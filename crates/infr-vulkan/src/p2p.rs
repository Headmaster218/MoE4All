//! Host-LESS cross-device buffer sharing over Vulkan external memory (the multi-GPU P2P slice).
//!
//! The host-bounce baseline (Slice 0, `tests/interconnect_probe.rs`) moves a tensor between two
//! physical GPUs as `device A → host RAM → device B`: two DDR round-trips over PCIe. This module
//! removes the bounce. It allocates a buffer's backing `VkDeviceMemory` on device **A** with an
//! EXTERNAL handle type (dma-buf, or opaque-fd), exports that memory as a POSIX fd
//! (`vkGetMemoryFdKHR`), and IMPORTS the fd on device **B** (`VkImportMemoryFdInfoKHR`), binding a
//! device-B `VkBuffer` to it. The device-B buffer then ALIASES device A's physical bytes — device B
//! reads/writes A's memory directly over PCIe, no host copy.
//!
//! This is a gated, ADDITIVE capability: a device without `VK_KHR_external_memory_fd` (and, for the
//! dma-buf handle type, `VK_EXT_external_memory_dma_buf`) simply reports [`VulkanBackend::p2p_supported`]
//! `false` and offers no P2P path. Nothing on the default single-device path changes.
//!
//! ## Synchronization (correctness first)
//! Device A writes the shared buffer, device B reads it — an ordering the caller establishes with
//! HOST-SIDE sync: `A.upload(...); A.sync();` (the backend's `upload`/`copy_buffer` submit through
//! `one_shot`, whose `queue_wait_idle` is a full fence) completes and flushes A's writes before B's
//! submit is recorded. This is the simplest CORRECT scheme. A zero-host-stall optimization replaces
//! it with `VK_KHR_external_semaphore_fd` (export a semaphore signalled by A, wait on it on B, no
//! host round-trip) — see [`crate::tp_sem`].
//!
//! ## Queue-family ownership (EXCLUSIVE external buffers)
//! The export/import buffers are created `SharingMode::EXCLUSIVE` (each backend has a single queue
//! family, so `CONCURRENT` — which needs ≥2 distinct family indices — is not available). A buffer
//! whose backing memory is written by device A's queue family and read by device B's therefore needs
//! an explicit queue-family ownership transfer through `VK_QUEUE_FAMILY_EXTERNAL`: the transport
//! (`tp_sem` / the host-fence and pipeline copies) RELEASES the exported buffer to
//! `QUEUE_FAMILY_EXTERNAL` after A's write and ACQUIRES the imported buffer from it before B's read,
//! satisfying the spec's cross-family requirement (the host fence alone does not).
//!
//! ## fd ownership
//! [`P2pExport`] owns the exported fd and closes it on drop. Each [`import`](VulkanBackend::p2p_import)
//! DUPLICATES the fd (`dup(2)`) so the import consumes the DUP (Vulkan takes ownership of an fd it
//! imports on success) while the export keeps its original — a `P2pExport` can be imported more than
//! once (a dma-buf may have many importers) and there is never a double-close.

use std::os::fd::RawFd;

use ash::vk;

use crate::{be, Backing, VkBuffer, VulkanBackend, BUFFER_USAGE};
use gpu_allocator::MemoryLocation;
use infr_core::backend::Buffer;
use infr_core::error::Result;
use std::sync::Arc;

/// External-memory handle type for a P2P export/import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pHandleType {
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` — the Linux dma-buf handle. The portable
    /// choice for CROSS-GPU sharing: a dma-buf can be imported by a different physical device
    /// (needs `VK_EXT_external_memory_dma_buf` on both sides).
    DmaBuf,
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` — the driver's opaque fd. Meant for
    /// cross-process / cross-API sharing on the SAME device; a driver may reject importing it on a
    /// different physical device. Probed here so the report can say whether it works cross-GPU.
    OpaqueFd,
}

impl P2pHandleType {
    fn vk(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            P2pHandleType::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            P2pHandleType::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
        }
    }
}

/// A buffer whose backing memory has been exported from device A as an fd, for import on device B.
///
/// Holds the device-A `VkBuffer` (so the underlying pages stay alive, and device A can
/// upload/download into it through the exporting backend) plus the exported fd and the metadata the
/// importer needs. Dropping it destroys the device-A buffer and closes the fd.
pub struct P2pExport {
    /// Device-A buffer bound to the exportable memory. Exposed via [`buffer`](Self::buffer) so the
    /// exporting backend can `upload`/`download`/`copy_buffer` into it like any other buffer.
    buf: VkBuffer,
    /// The exported fd (dma-buf / opaque-fd). Owned here; `dup`ed per import; closed on drop.
    fd: RawFd,
    /// Size of the underlying `VkDeviceMemory` allocation (`memory_requirements.size` on A). The
    /// import binds an allocation of exactly this many bytes.
    mem_size: u64,
    /// Logical byte length the caller asked for (`<= mem_size`).
    size: usize,
    /// The memory-type INDEX the exporter (device A) placed the backing allocation on. Threaded to
    /// the importer so a same-driver import (`OPAQUE_FD`, and `DmaBuf` when the exporter's type is
    /// importable) binds the EXPORTER's type rather than blindly picking the lowest allowed bit —
    /// the lowest bit can be a host-visible/uncached type the exporter never used, spuriously
    /// rejecting a valid cross-device import.
    mem_type: u32,
    handle_type: P2pHandleType,
}

// The fd + VkBuffer are Send/Sync under the same whole-backend mutex discipline as VkBuffer.
unsafe impl Send for P2pExport {}
unsafe impl Sync for P2pExport {}

impl P2pExport {
    /// The device-A buffer aliasing the exported memory — upload/download/copy into it through the
    /// backend that produced this export (device A) exactly like any other buffer.
    pub fn buffer(&self) -> &dyn Buffer {
        &self.buf
    }

    /// Logical byte length of the shared buffer.
    pub fn len_bytes(&self) -> usize {
        self.size
    }

    /// The handle type this memory was exported as.
    pub fn handle_type(&self) -> P2pHandleType {
        self.handle_type
    }
}

impl Drop for P2pExport {
    fn drop(&mut self) {
        // `buf` (the device-A `VkBuffer`, `Backing::External`) drops with the struct and frees the
        // memory + destroys the buffer. Close our copy of the exported fd — imports used DUPs, so
        // this never double-closes an fd Vulkan already owns.
        if self.fd >= 0 {
            unsafe { libc::close(self.fd) };
        }
    }
}

impl VulkanBackend {
    /// Can this backend's device participate in the host-less P2P transport for `handle_type`?
    ///
    /// Needs `VK_KHR_external_memory_fd` (the fd export/import ops) always, plus
    /// `VK_EXT_external_memory_dma_buf` for the dma-buf handle type. `false` ⇒ this device offers no
    /// P2P path (the caller falls back to the host-bounce transport).
    pub fn p2p_supported(&self, handle_type: P2pHandleType) -> bool {
        self.shared.external_memory_fd.is_some()
            && match handle_type {
                P2pHandleType::DmaBuf => self.shared.has_dma_buf,
                P2pHandleType::OpaqueFd => true,
            }
    }

    /// Can this device order a cross-device dependency with an EXPORTED/IMPORTED timeline/binary
    /// semaphore (`VK_KHR_external_semaphore_fd`)? When every rank in a tensor-parallel group says
    /// yes, the all-reduce waits on a peer's GPU-side signal with no host round-trip
    /// (`AllReduceMode::P2pSemaphore`); otherwise the P2P data path is ordered by the host fence.
    ///
    /// This returns `true` whenever the device enabled `VK_KHR_external_semaphore_fd`, i.e. the
    /// semaphore-ordered all-reduce path is LIVE (not gated off) on such a device — the all-reduce
    /// selects [`AllReduceMode::P2pSemaphore`] and orders the peer read GPU-side. A device/driver
    /// that cannot import a cross-device `OPAQUE_FD` semaphore reports the exact failure and the
    /// all-reduce falls back to the host fence (`AllReduceMode::P2pHostFence`).
    pub fn external_semaphore_supported(&self) -> bool {
        self.shared.external_semaphore_fd.is_some()
    }

    /// Allocate a `size`-byte device-local buffer on THIS backend (device A) whose backing memory is
    /// EXPORTABLE as `handle_type`, and export it as an fd. The returned [`P2pExport`] owns the
    /// device-A buffer (upload into it via [`P2pExport::buffer`]) and the fd; import it on another
    /// backend with [`p2p_import`](Self::p2p_import).
    ///
    /// The memory is placed on a DEVICE_LOCAL type when the buffer's requirements allow one (so a
    /// discrete card exports real VRAM); otherwise the first allowed type. It is NOT counted against
    /// the VRAM budget guard — this is a probe/transport capability, explicit and self-bounded, not
    /// a model allocation.
    pub fn p2p_export(&self, size: usize, handle_type: P2pHandleType) -> Result<P2pExport> {
        if !self.p2p_supported(handle_type) {
            return Err(be(format!(
                "p2p_export: this device does not support the {handle_type:?} external-memory \
                 handle type (needs VK_KHR_external_memory_fd{})",
                match handle_type {
                    P2pHandleType::DmaBuf => " + VK_EXT_external_memory_dma_buf",
                    P2pHandleType::OpaqueFd => "",
                }
            )));
        }
        let ext_fd = self.shared.external_memory_fd.as_ref().unwrap();
        let device = &self.shared.device;
        let ht = handle_type.vk();

        // ── exportable buffer ────────────────────────────────────────────────────────────────
        let mut ext_ci = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
        let buf_ci = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(BUFFER_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_ci);
        let buffer = unsafe { device.create_buffer(&buf_ci, None) }
            .map_err(|e| be(format!("p2p_export create_buffer: {e}")))?;

        // From here on, any early return must destroy `buffer`.
        let cleanup_buf = |device: &ash::Device| unsafe { device.destroy_buffer(buffer, None) };

        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        // Prefer real device-local memory (VRAM on a discrete card); fall back to any allowed type.
        let ty = self
            .find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| (req.memory_type_bits != 0).then(|| req.memory_type_bits.trailing_zeros()))
            .ok_or_else(|| {
                cleanup_buf(device);
                be("p2p_export: no memory type satisfies the exportable buffer")
            })?;

        // ── dedicated, exportable allocation ─────────────────────────────────────────────────
        let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(ht);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let alloc_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(ty)
            .push_next(&mut export_info)
            .push_next(&mut dedicated);
        let memory = match unsafe { device.allocate_memory(&alloc_ci, None) } {
            Ok(m) => m,
            Err(e) => {
                cleanup_buf(device);
                return Err(be(format!(
                    "p2p_export allocate_memory({} bytes, {handle_type:?}): {e}",
                    req.size
                )));
            }
        };
        let cleanup_mem = |device: &ash::Device| unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        };

        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            cleanup_mem(device);
            return Err(be(format!("p2p_export bind_buffer_memory: {e}")));
        }

        // ── export the fd ────────────────────────────────────────────────────────────────────
        let get_fd = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(ht);
        let fd = match unsafe { ext_fd.get_memory_fd(&get_fd) } {
            Ok(fd) => fd,
            Err(e) => {
                cleanup_mem(device);
                return Err(be(format!(
                    "p2p_export vkGetMemoryFdKHR({handle_type:?}): {e}"
                )));
            }
        };

        Ok(P2pExport {
            buf: VkBuffer {
                shared: Arc::clone(&self.shared),
                buffer,
                backing: Backing::External { memory },
                size,
                mem_size: req.size,
                location: MemoryLocation::GpuOnly,
                sub_offset: 0,
                own_addr: None,
                act_bytes: 0,
            },
            fd,
            mem_size: req.size,
            size,
            mem_type: ty,
            handle_type,
        })
    }

    /// Import `export`'s memory on THIS backend (device B), returning a device-B buffer that ALIASES
    /// device A's physical bytes. Reads/writes on the returned buffer touch A's memory directly over
    /// PCIe — no host copy. The caller MUST ensure device A's writes are complete before device B
    /// reads (host-side sync: `A.sync()` before B's submit — see the module docs).
    ///
    /// The `export` must outlive the returned buffer (it owns the underlying pages).
    pub fn p2p_import(&self, export: &P2pExport) -> Result<Box<dyn Buffer>> {
        if !self.p2p_supported(export.handle_type) {
            return Err(be(format!(
                "p2p_import: this device does not support the {:?} external-memory handle type",
                export.handle_type
            )));
        }
        let ext_fd = self.shared.external_memory_fd.as_ref().unwrap();
        let device = &self.shared.device;
        let ht = export.handle_type.vk();

        // DUP the export's fd: a successful import CONSUMES the fd it is given, so hand Vulkan a copy
        // and let `P2pExport` keep (and eventually close) the original. `< 0` ⇒ `dup` failed.
        let dup_fd: RawFd = unsafe { libc::dup(export.fd) };
        if dup_fd < 0 {
            return Err(be(format!(
                "p2p_import dup(fd={}): {}",
                export.fd,
                std::io::Error::last_os_error()
            )));
        }
        // Any early return before a SUCCESSFUL allocate_memory must close `dup_fd` (Vulkan has not
        // taken ownership yet).
        let close_dup = || unsafe {
            libc::close(dup_fd);
        };

        // ── device-B buffer with the same external handle type ───────────────────────────────
        let mut ext_ci = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
        let buf_ci = vk::BufferCreateInfo::default()
            .size(export.size as u64)
            .usage(BUFFER_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_ci);
        let buffer = match unsafe { device.create_buffer(&buf_ci, None) } {
            Ok(b) => b,
            Err(e) => {
                close_dup();
                return Err(be(format!("p2p_import create_buffer: {e}")));
            }
        };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Which memory types can this fd be imported into on device B?
        //   * dma-buf (and other "host-derived" handles): the DRIVER decides — query it with
        //     `vkGetMemoryFdPropertiesKHR` and intersect with the buffer's own requirement bits.
        //   * opaque-fd: the spec FORBIDS calling `vkGetMemoryFdPropertiesKHR` for OPAQUE_FD
        //     (VUID-vkGetMemoryFdPropertiesKHR-handleType-00674) — the memory type is fixed by the
        //     exporting device, so we take the buffer's requirement bits directly and let the
        //     `vkAllocateMemory` import be the arbiter (it returns ERROR_INVALID_EXTERNAL_HANDLE if
        //     the opaque fd is not importable on this device — the clean cross-device rejection).
        let allowed = match export.handle_type {
            P2pHandleType::DmaBuf => {
                let mut fd_props = vk::MemoryFdPropertiesKHR::default();
                if let Err(e) =
                    unsafe { ext_fd.get_memory_fd_properties(ht, dup_fd, &mut fd_props) }
                {
                    unsafe { device.destroy_buffer(buffer, None) };
                    close_dup();
                    return Err(be(format!(
                        "p2p_import vkGetMemoryFdPropertiesKHR(DmaBuf): {e} — the importing device \
                         rejected the fd (dma-buf not importable cross-device here)"
                    )));
                }
                fd_props.memory_type_bits & req.memory_type_bits
            }
            P2pHandleType::OpaqueFd => req.memory_type_bits,
        };
        if allowed == 0 {
            unsafe { device.destroy_buffer(buffer, None) };
            close_dup();
            return Err(be(format!(
                "p2p_import: no memory type on this device accepts the imported {:?} fd \
                 (buffer requires={:#x}) — cross-device import of this handle type is not usable on \
                 this device pair",
                export.handle_type, req.memory_type_bits
            )));
        }
        // Prefer the EXPORTER's memory-type index when it is among the allowed set (it names the type
        // the memory actually lives on — required to match for a same-driver import); otherwise fall
        // back to the lowest allowed bit. Picking the lowest bit unconditionally can select a type
        // the exporter never used and spuriously reject the import.
        let ty = if export.mem_type < 32 && (allowed & (1u32 << export.mem_type)) != 0 {
            export.mem_type
        } else {
            allowed.trailing_zeros()
        };

        // ── import + bind ────────────────────────────────────────────────────────────────────
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(ht)
            .fd(dup_fd);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let alloc_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(export.mem_size)
            .memory_type_index(ty)
            .push_next(&mut import_info)
            .push_next(&mut dedicated);
        let memory = match unsafe { device.allocate_memory(&alloc_ci, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                close_dup();
                return Err(be(format!(
                    "p2p_import allocate_memory (import {:?} fd): {e} — the driver refused to import \
                     the fd into memory type {ty}",
                    export.handle_type
                )));
            }
        };
        // From here Vulkan OWNS `dup_fd` (consumed by the successful import); do NOT close it.

        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(be(format!("p2p_import bind_buffer_memory: {e}")));
        }

        Ok(Box::new(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            backing: Backing::External { memory },
            size: export.size,
            mem_size: export.mem_size,
            location: MemoryLocation::GpuOnly,
            sub_offset: 0,
            own_addr: None,
            act_bytes: 0,
        }))
    }
}
