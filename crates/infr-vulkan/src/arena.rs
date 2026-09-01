//! Physical Vulkan backing for device-addressable arenas.
//!
//! Allocation policy lives above this module. An arena shard guarantees device residency,
//! transfer access and a buffer device address; host mapping is an optional acceleration, not a
//! correctness requirement.

use std::sync::Arc;

use ash::vk;

use infr_core::backend::Buffer;
use infr_core::error::Result;

use crate::{as_vk_buf, be, VulkanBackend};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceArenaBacking {
    MappedDeviceLocal,
    DeviceLocal,
}

fn choose_backing(required: u64, mapped_heap_capacity: u64) -> DeviceArenaBacking {
    if required <= mapped_heap_capacity {
        DeviceArenaBacking::MappedDeviceLocal
    } else {
        DeviceArenaBacking::DeviceLocal
    }
}

/// Capacity of heaps reachable through a DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT memory type.
/// Count each heap once even when the driver exposes several compatible memory types for it.
fn mapped_device_local_heap_capacity(vk: &VulkanBackend) -> u64 {
    let properties = unsafe {
        vk.shared
            .instance
            .get_physical_device_memory_properties(vk.shared.physical_device)
    };
    let wanted = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mut counted = vec![false; properties.memory_heap_count as usize];
    let mut bytes = 0u64;
    for index in 0..properties.memory_type_count as usize {
        let memory_type = properties.memory_types[index];
        if !memory_type.property_flags.contains(wanted) {
            continue;
        }
        let heap = memory_type.heap_index as usize;
        if heap < counted.len() && !counted[heap] {
            counted[heap] = true;
            bytes = bytes.saturating_add(properties.memory_heaps[heap].size);
        }
    }
    bytes
}

/// One physical device-addressable allocation. The CPU mapping exists only on a mapped-ReBAR/UMA
/// backing; every caller must be prepared to upload through a Vulkan transfer when it is absent.
pub(crate) struct DeviceArenaShard {
    buffer: Arc<dyn Buffer>,
    base_addr: u64,
    mapped_ptr: Option<usize>,
    bytes: usize,
}

impl DeviceArenaShard {
    pub(crate) fn buffer(&self) -> &dyn Buffer {
        self.buffer.as_ref()
    }

    pub(crate) fn buffer_arc(&self) -> Arc<dyn Buffer> {
        Arc::clone(&self.buffer)
    }

    pub(crate) fn base_addr(&self) -> u64 {
        self.base_addr
    }

    pub(crate) fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr.map(|ptr| ptr as *mut u8)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Physical shard set selected independently from the logical allocator layered above it.
pub(crate) struct DeviceArena {
    backing: DeviceArenaBacking,
    shards: Vec<Arc<DeviceArenaShard>>,
}

impl DeviceArena {
    pub(crate) fn new(vk: &VulkanBackend, shard_sizes: &[usize]) -> Result<Self> {
        if shard_sizes.is_empty() || shard_sizes.contains(&0) {
            return Err(be("device arena needs non-empty physical shards"));
        }
        let required = shard_sizes.iter().try_fold(0u64, |total, &bytes| {
            total
                .checked_add(bytes as u64)
                .ok_or_else(|| be("device arena byte size overflow"))
        })?;
        let mapped_capacity = mapped_device_local_heap_capacity(vk);
        let preferred = choose_backing(required, mapped_capacity);

        if preferred == DeviceArenaBacking::MappedDeviceLocal {
            match Self::allocate(vk, shard_sizes, preferred) {
                Ok(arena) => return Ok(arena),
                Err(mapped_error) => {
                    tracing::warn!(
                        "[infr] mapped device-local arena unavailable ({mapped_error}); falling back to ordinary device-local VRAM with staged transfers"
                    );
                }
            }
        } else {
            tracing::info!(
                "[infr] mapped device-local heap is {:.2} MiB, below the {:.2} MiB arena; using ordinary device-local VRAM with staged transfers",
                mapped_capacity as f64 / (1u64 << 20) as f64,
                required as f64 / (1u64 << 20) as f64,
            );
        }

        Self::allocate(vk, shard_sizes, DeviceArenaBacking::DeviceLocal).map_err(|error| {
            be(format!(
                "device arena allocation failed on the ordinary device-local fallback: {error}"
            ))
        })
    }

    fn allocate(
        vk: &VulkanBackend,
        shard_sizes: &[usize],
        backing: DeviceArenaBacking,
    ) -> Result<Self> {
        let mut shards = Vec::with_capacity(shard_sizes.len());
        for &bytes in shard_sizes {
            let (buffer, base_addr) = match backing {
                DeviceArenaBacking::MappedDeviceLocal => vk.alloc_mapped_arena_bda(bytes)?,
                DeviceArenaBacking::DeviceLocal => vk.alloc_device_local_arena_bda(bytes)?,
            };
            let buffer: Arc<dyn Buffer> = Arc::from(buffer);
            let mapped_ptr = as_vk_buf(buffer.as_ref())?
                .mapped_ptr()
                .map(|ptr| ptr as usize);
            if backing == DeviceArenaBacking::MappedDeviceLocal && mapped_ptr.is_none() {
                return Err(be("mapped device-local arena shard has no CPU mapping"));
            }
            shards.push(Arc::new(DeviceArenaShard {
                buffer,
                base_addr,
                mapped_ptr,
                bytes,
            }));
        }
        tracing::debug!(
            "[infr] device arena: {} bytes across {} shard(s), backing={backing:?}",
            shard_sizes.iter().sum::<usize>(),
            shards.len(),
        );
        Ok(Self { backing, shards })
    }

    pub(crate) fn backing(&self) -> DeviceArenaBacking {
        self.backing
    }

    pub(crate) fn shard(&self, index: usize) -> Option<Arc<DeviceArenaShard>> {
        self.shards.get(index).cloned()
    }

    pub(crate) fn shard_sizes(&self) -> Vec<usize> {
        self.shards.iter().map(|shard| shard.bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{choose_backing, DeviceArena, DeviceArenaBacking};
    use crate::transfer::{DeviceTransferTarget, HostTransferPath};
    use crate::VulkanBackend;
    use ash::vk;
    use infr_core::Backend;

    #[test]
    fn mapped_backing_is_only_selected_when_the_whole_arena_fits_its_heap() {
        assert_eq!(
            choose_backing(256 << 20, 256 << 20),
            DeviceArenaBacking::MappedDeviceLocal
        );
        assert_eq!(
            choose_backing((256 << 20) + 1, 256 << 20),
            DeviceArenaBacking::DeviceLocal
        );
    }

    #[test]
    fn devices_without_mapped_device_local_memory_use_device_local() {
        assert_eq!(choose_backing(1, 0), DeviceArenaBacking::DeviceLocal);
    }

    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn unmapped_device_arena_roundtrips_through_staging() {
        let vk = match VulkanBackend::new() {
            Ok(vk) => vk,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let arena = DeviceArena::allocate(&vk, &[1 << 20], DeviceArenaBacking::DeviceLocal)
            .expect("allocate forced device-local arena");
        let shard = arena.shard(0).expect("one shard");
        assert!(shard.mapped_ptr().is_none());

        let bytes: Vec<u8> = (0..64 * 1024)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let target = DeviceTransferTarget::new(shard.buffer_arc(), 0, bytes.len())
            .expect("build transfer target");
        assert_eq!(
            vk.upload_device_target(&target, &bytes)
                .expect("staged upload"),
            HostTransferPath::Staged
        );
        let mut back = vec![0u8; bytes.len()];
        vk.download(shard.buffer(), &mut back).expect("download");
        assert_eq!(back, bytes);

        let replacement: Vec<u8> = bytes.iter().map(|byte| byte ^ 0x5a).collect();
        let (staging, _) = vk
            .stage_host_bytes(&replacement)
            .expect("materialize recorded staging source");
        let recorder = vk.recorder().expect("create transfer recorder");
        recorder.host_transfer_barrier();
        recorder.copy_regions(
            staging.as_ref(),
            target.buffer(),
            &[vk::BufferCopy::default().size(replacement.len() as u64)],
        );
        recorder.retain_buffer(staging);
        recorder
            .finish_nowait()
            .expect("submit recorded staging copy")
            .wait()
            .expect("wait recorded staging copy");
        vk.download(shard.buffer(), &mut back).expect("download");
        assert_eq!(back, replacement);
    }
}
