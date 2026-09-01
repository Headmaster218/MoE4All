//! Host-to-device transfer primitives shared by paged weights and elastic arena clients.
//!
//! Callers name an opaque device target. Whether the target is CPU-mapped is handled here:
//! mapped targets take the direct write fast path, while ordinary device-local targets use a
//! host-visible staging allocation and `vkCmdCopyBuffer`.

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;

use infr_core::backend::Buffer;
use infr_core::error::Result;
use infr_core::pager_profile;

use crate::{as_vk_buf, be, copy_to_mapped, VkBuffer, VulkanBackend};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostTransferPath {
    DirectMapped,
    Staged,
}

/// A byte range that can be consumed by Vulkan. `mapped_ptr` points at the start of this exact
/// range when the physical backing is host-visible; it is absent for ordinary device-local VRAM.
#[derive(Clone)]
pub(crate) struct DeviceTransferTarget {
    buffer: Arc<dyn Buffer>,
    /// Offset relative to this logical `Buffer` handle (used by `Recorder`, which adds sub_offset).
    buffer_offset: usize,
    /// Absolute offset in the underlying VkBuffer (used by raw one-shot submissions).
    vk_offset: usize,
    mapped_ptr: Option<usize>,
    len: usize,
}

impl DeviceTransferTarget {
    pub(crate) fn new(buffer: Arc<dyn Buffer>, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| be("device transfer target range overflow"))?;
        if end > buffer.len_bytes() {
            return Err(be(format!(
                "device transfer target {offset}..{end} exceeds its {}-byte buffer",
                buffer.len_bytes()
            )));
        }
        let vk_buffer = as_vk_buf(buffer.as_ref())?;
        let vk_offset = vk_buffer
            .sub_offset
            .checked_add(offset)
            .ok_or_else(|| be("device transfer target Vulkan offset overflow"))?;
        let mapped_ptr = vk_buffer
            .mapped_ptr()
            .map(|ptr| unsafe { ptr.add(offset) } as usize);
        Ok(Self {
            buffer,
            buffer_offset: offset,
            vk_offset,
            mapped_ptr,
            len,
        })
    }

    pub(crate) fn buffer(&self) -> &dyn Buffer {
        self.buffer.as_ref()
    }

    pub(crate) fn buffer_arc(&self) -> Arc<dyn Buffer> {
        Arc::clone(&self.buffer)
    }

    pub(crate) fn buffer_offset(&self) -> usize {
        self.buffer_offset
    }

    pub(crate) fn vk_offset(&self) -> usize {
        self.vk_offset
    }

    pub(crate) fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr.map(|ptr| ptr as *mut u8)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_mapped(&self) -> bool {
        self.mapped_ptr.is_some()
    }

    pub(crate) fn subtarget(&self, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| be("device transfer sub-range overflow"))?;
        if end > self.len {
            return Err(be("device transfer sub-range exceeds its parent target"));
        }
        Ok(Self {
            buffer: Arc::clone(&self.buffer),
            buffer_offset: self
                .buffer_offset
                .checked_add(offset)
                .ok_or_else(|| be("device transfer sub-range Vulkan offset overflow"))?,
            vk_offset: self
                .vk_offset
                .checked_add(offset)
                .ok_or_else(|| be("device transfer sub-range Vulkan offset overflow"))?,
            mapped_ptr: self.mapped_ptr.map(|ptr| ptr + offset),
            len,
        })
    }
}

impl VulkanBackend {
    /// Allocate staging explicitly on the host-visible non-device-local heap when the device
    /// exposes one. This prevents a small RDNA2 ReBAR heap from being consumed by the fallback
    /// that exists precisely because the expert arena did not fit that heap.
    fn make_host_transfer_buffer(&self, size: usize) -> Result<VkBuffer> {
        let Some(memory_type) = self.shared.host_overflow_type else {
            return self.make_buf(size, MemoryLocation::CpuToGpu, "host-transfer-staging");
        };
        let info = vk::BufferCreateInfo::default()
            .size(crate::fill_span(size))
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.shared.device.create_buffer(&info, None) }
            .map_err(|error| be(format!("create_buffer(host-transfer-staging): {error}")))?;
        let requirements = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };
        if requirements.memory_type_bits & (1 << memory_type) == 0 {
            unsafe { self.shared.device.destroy_buffer(buffer, None) };
            return self.make_buf(size, MemoryLocation::CpuToGpu, "host-transfer-staging");
        }
        self.alloc_vram_mapped(buffer, size, &requirements, memory_type, true, false, false)
            .inspect_err(|_| unsafe { self.shared.device.destroy_buffer(buffer, None) })
    }

    /// Materialize bytes in a temporary host-visible Vulkan buffer without submitting work. The
    /// returned owner must stay alive until every command that reads it has completed.
    pub(crate) fn stage_host_bytes(&self, src: &[u8]) -> Result<(Arc<dyn Buffer>, usize)> {
        let staging = self.make_host_transfer_buffer(src.len())?;
        let ptr = staging
            .mapped_ptr()
            .ok_or_else(|| be("host transfer staging allocation is not mapped"))?;
        let started = pager_profile::active().then(std::time::Instant::now);
        copy_to_mapped(src, ptr);
        if let Some(t0) = started {
            pager_profile::record_memcpy(src.len(), t0.elapsed());
        }
        Ok((Arc::new(staging), ptr as usize))
    }

    /// Fill one target range. The callback always receives writable host memory, either the final
    /// mapped destination or a temporary staging allocation. The staged path is synchronous; it
    /// is the universal correctness fallback used only when a direct/imported batch is unavailable.
    pub(crate) fn write_device_target(
        &self,
        target: &DeviceTransferTarget,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<HostTransferPath> {
        if let Some(ptr) = target.mapped_ptr() {
            let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, target.len()) };
            fill(bytes)?;
            return Ok(HostTransferPath::DirectMapped);
        }

        let staging = self.make_host_transfer_buffer(target.len())?;
        let staging_ptr = staging
            .mapped_ptr()
            .ok_or_else(|| be("host transfer staging allocation is not mapped"))?;
        let staging_bytes = unsafe { std::slice::from_raw_parts_mut(staging_ptr, target.len()) };
        fill(staging_bytes)?;
        let source: Arc<dyn Buffer> = Arc::new(staging);
        self.copy_transfer_targets_now(&[(source, 0, target.clone(), target.len())])?;
        Ok(HostTransferPath::Staged)
    }

    pub(crate) fn upload_device_target(
        &self,
        target: &DeviceTransferTarget,
        src: &[u8],
    ) -> Result<HostTransferPath> {
        if src.len() != target.len() {
            return Err(be(format!(
                "host upload has {} bytes but its device target has {}",
                src.len(),
                target.len()
            )));
        }
        if let Some(dst) = target.mapped_ptr() {
            let started = pager_profile::active().then(std::time::Instant::now);
            copy_to_mapped(src, dst);
            if let Some(t0) = started {
                pager_profile::record_memcpy(src.len(), t0.elapsed());
            }
            return Ok(HostTransferPath::DirectMapped);
        }
        let (staging, _) = self.stage_host_bytes(src)?;
        self.copy_transfer_targets_now(&[(staging, 0, target.clone(), src.len())])?;
        Ok(HostTransferPath::Staged)
    }

    /// Execute already-materialized buffer copies immediately on the main queue. Used by the
    /// Decode overlap compatibility path when no ambient recorder exists; mapped destinations can
    /// still avoid this through their direct CPU fallback.
    pub(crate) fn copy_transfer_targets_now(
        &self,
        copies: &[(Arc<dyn Buffer>, usize, DeviceTransferTarget, usize)],
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let mut resolved = Vec::with_capacity(copies.len());
        for (source, source_offset, target, len) in copies {
            if *len > target.len() {
                return Err(be("immediate transfer exceeds its device target"));
            }
            resolved.push((
                as_vk_buf(source.as_ref())?.buffer,
                *source_offset as u64,
                as_vk_buf(target.buffer())?.buffer,
                target.vk_offset() as u64,
                *len as u64,
            ));
        }
        let shared = Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            let host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            shared.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[host],
                &[],
                &[],
            );
            for &(src, src_offset, dst, dst_offset, len) in &resolved {
                shared.device.cmd_copy_buffer(
                    cmd,
                    src,
                    dst,
                    &[vk::BufferCopy::default()
                        .src_offset(src_offset)
                        .dst_offset(dst_offset)
                        .size(len)],
                );
            }
        })?;
        if pager_profile::active() {
            let bytes = copies
                .iter()
                .fold(0usize, |total, copy| total.saturating_add(copy.3));
            pager_profile::record_gpu_copy(bytes);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::HostTransferPath;

    #[test]
    fn transfer_paths_distinguish_optional_mapping_from_correctness_fallback() {
        assert_ne!(HostTransferPath::DirectMapped, HostTransferPath::Staged);
    }
}
