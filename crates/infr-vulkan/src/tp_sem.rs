//! GPU-side cross-device synchronization for the tensor-parallel all-reduce, via
//! `VK_KHR_external_semaphore_fd` — a TIMELINE semaphore exported as an fd on one device and imported
//! on another, so a peer's read waits on the producer's GPU-side signal with NO host round-trip.
//!
//! Mirrors the memory export/import in [`crate::p2p`], for semaphores instead of buffers:
//! * [`VulkanBackend::tp_export_timeline`] creates a timeline `VkSemaphore` with an `OPAQUE_FD`
//!   export handle and exports the fd (`vkGetSemaphoreFdKHR`).
//! * [`VulkanBackend::tp_import_timeline`] creates a timeline semaphore on the consumer and imports
//!   the (dup'd) fd (`vkImportSemaphoreFdKHR`, permanent) so it shares the producer's payload — a
//!   value signalled on the producer's semaphore is observed by the consumer's.
//!
//! The submit primitives record a buffer copy and submit it with timeline wait/signal semaphores and
//! NO `queue_wait_idle`, so the host issues the producer's publish and the consumer's gather
//! back-to-back and the GPUs enforce the ordering themselves. `OPAQUE_FD` semaphore sharing is
//! same-driver cross-device (the dGPU+iGPU pair here are both RADV), and is GATED behind
//! [`VulkanBackend::external_semaphore_supported`] — a device/driver that can't import one makes the
//! all-reduce fall back to the host fence.

#[cfg(target_os = "windows")]
type RawFd = std::os::raw::c_int;
#[cfg(not(target_os = "windows"))]
use std::os::fd::RawFd;
use std::sync::Arc;

use ash::vk;

use crate::{as_vk_buf, be, VulkanBackend, VulkanShared};
use infr_core::backend::Buffer;
use infr_core::error::Result;

/// A timeline semaphore exported as an fd on the PRODUCER device. The producer signals a monotonic
/// value on it (per all-reduce); importers on other devices wait that value. Owns the semaphore + fd,
/// destroyed/closed on drop.
pub struct TpExportSemaphore {
    shared: Arc<VulkanShared>,
    sem: vk::Semaphore,
    /// The exported fd (OPAQUE_FD). Owned here; `dup`ed per import; closed on drop.
    fd: RawFd,
}

// The semaphore + fd are Send/Sync under the same whole-backend mutex discipline as the buffers.
unsafe impl Send for TpExportSemaphore {}
unsafe impl Sync for TpExportSemaphore {}

impl Drop for TpExportSemaphore {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_semaphore(self.sem, None) };
        if self.fd >= 0 {
            unsafe { libc::close(self.fd) };
        }
    }
}

/// A timeline semaphore on the CONSUMER device that shares a [`TpExportSemaphore`]'s payload (imported
/// from its fd). Waiting a value on it blocks until the producer signals that value.
pub struct TpImportSemaphore {
    shared: Arc<VulkanShared>,
    sem: vk::Semaphore,
}

unsafe impl Send for TpImportSemaphore {}
unsafe impl Sync for TpImportSemaphore {}

impl Drop for TpImportSemaphore {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_semaphore(self.sem, None) };
    }
}

impl VulkanBackend {
    /// Create a timeline semaphore on THIS device with an `OPAQUE_FD` export handle and export it as
    /// an fd, for [`tp_import_timeline`](Self::tp_import_timeline) on a peer.
    pub fn tp_export_timeline(&self) -> Result<TpExportSemaphore> {
        let ext = self
            .shared
            .external_semaphore_fd
            .as_ref()
            .ok_or_else(|| be("tp: this device has no VK_KHR_external_semaphore_fd"))?;
        let device = &self.shared.device;
        let ht = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;

        let mut type_ci = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export_ci = vk::ExportSemaphoreCreateInfo::default().handle_types(ht);
        let ci = vk::SemaphoreCreateInfo::default()
            .push_next(&mut type_ci)
            .push_next(&mut export_ci);
        let sem = unsafe { device.create_semaphore(&ci, None) }
            .map_err(|e| be(format!("tp_export_timeline create_semaphore: {e}")))?;

        let get = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(sem)
            .handle_type(ht);
        let fd = match unsafe { ext.get_semaphore_fd(&get) } {
            Ok(fd) => fd,
            Err(e) => {
                unsafe { device.destroy_semaphore(sem, None) };
                return Err(be(format!("tp_export_timeline vkGetSemaphoreFdKHR: {e}")));
            }
        };
        Ok(TpExportSemaphore {
            shared: Arc::clone(&self.shared),
            sem,
            fd,
        })
    }

    /// Import `export`'s timeline payload on THIS device (a peer of the exporter): create a timeline
    /// semaphore and permanently import the (dup'd) fd, so a value the exporter signals is observed
    /// here. The `export` must outlive the returned import (it owns the fd).
    ///
    /// Returns the exact `VkResult` on rejection — a driver that refuses a cross-device OPAQUE_FD
    /// semaphore import is a valid hardware finding the caller reports and falls back to host-fence.
    pub fn tp_import_timeline(&self, export: &TpExportSemaphore) -> Result<TpImportSemaphore> {
        let ext = self
            .shared
            .external_semaphore_fd
            .as_ref()
            .ok_or_else(|| be("tp: this device has no VK_KHR_external_semaphore_fd"))?;
        let device = &self.shared.device;
        let ht = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;

        let mut type_ci = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_ci);
        let sem = unsafe { device.create_semaphore(&ci, None) }
            .map_err(|e| be(format!("tp_import_timeline create_semaphore: {e}")))?;

        // dup: a successful import CONSUMES the fd it is given, so hand Vulkan a copy and let the
        // export keep (and eventually close) the original.
        let dup_fd: RawFd = unsafe { libc::dup(export.fd) };
        if dup_fd < 0 {
            unsafe { device.destroy_semaphore(sem, None) };
            return Err(be(format!(
                "tp_import_timeline dup(fd={}): {}",
                export.fd,
                std::io::Error::last_os_error()
            )));
        }
        let import = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(sem)
            .handle_type(ht)
            .flags(vk::SemaphoreImportFlags::empty()) // permanent import (shared payload)
            .fd(dup_fd);
        match unsafe { ext.import_semaphore_fd(&import) } {
            Ok(()) => {}
            Err(e) => {
                unsafe {
                    libc::close(dup_fd);
                    device.destroy_semaphore(sem, None);
                };
                return Err(be(format!(
                    "tp_import_timeline vkImportSemaphoreFdKHR (cross-device OPAQUE_FD): {e}"
                )));
            }
        }
        Ok(TpImportSemaphore {
            shared: Arc::clone(&self.shared),
            sem,
        })
    }

    /// Record `src → dst` (`bytes`) into a one-time command buffer and submit it SIGNALLING `sem` to
    /// `value` (timeline), with NO `queue_wait_idle` — the copy runs async and the signal fires on its
    /// completion. Returns the command buffer, which the caller MUST keep alive until it frees it
    /// (after a later [`tp_queue_wait_idle`](Self::tp_queue_wait_idle)).
    pub fn tp_submit_copy_signal(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        bytes: usize,
        sem: &TpExportSemaphore,
        value: u64,
    ) -> Result<vk::CommandBuffer> {
        // Publish: copy into the exported buffer, then RELEASE its queue-family ownership to
        // QUEUE_FAMILY_EXTERNAL so the peer device may acquire + read it (the EXCLUSIVE external
        // buffer needs the ownership transfer — see `p2p.rs`).
        let cmd = self.tp_record_copies(&[(src, dst, bytes)], &[], &[dst])?;
        let cmds = [cmd];
        let sems = [sem.sem];
        let vals = [value];
        let mut tl = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&vals);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(&sems)
            .push_next(&mut tl);
        if let Err(e) = unsafe {
            self.shared
                .device
                .queue_submit(self.shared.queue, &[submit], vk::Fence::null())
        } {
            // Submit failed → the cmd never runs; free it so it does not leak the shared pool.
            self.tp_free_cmds(&[cmd]);
            return Err(be(format!("tp_submit_copy_signal queue_submit: {e}")));
        }
        Ok(cmd)
    }

    /// Record all `copies` into one command buffer and submit it WAITING on each `(sem, value)` (at
    /// the TRANSFER stage) before the copies run, with NO `queue_wait_idle` — the waits are enforced
    /// GPU-side, so the host does not block on the producers. Returns the command buffer (free it
    /// after a later [`tp_queue_wait_idle`](Self::tp_queue_wait_idle)).
    pub fn tp_submit_copies_wait(
        &self,
        copies: &[(&dyn Buffer, &dyn Buffer, usize)],
        waits: &[(&TpImportSemaphore, u64)],
    ) -> Result<vk::CommandBuffer> {
        // Gather: ACQUIRE each imported (peer-exported) buffer from QUEUE_FAMILY_EXTERNAL before the
        // copies read it (the src side of every copy is a cross-device imported buffer).
        let acquire: Vec<&dyn Buffer> = copies.iter().map(|(s, _, _)| *s).collect();
        let cmd = self.tp_record_copies(copies, &acquire, &[])?;
        let cmds = [cmd];
        let wait_sems: Vec<vk::Semaphore> = waits.iter().map(|(s, _)| s.sem).collect();
        let wait_vals: Vec<u64> = waits.iter().map(|(_, v)| *v).collect();
        let wait_stages: Vec<vk::PipelineStageFlags> = waits
            .iter()
            .map(|_| vk::PipelineStageFlags::TRANSFER)
            .collect();
        let mut tl = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_vals);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .push_next(&mut tl);
        if let Err(e) = unsafe {
            self.shared
                .device
                .queue_submit(self.shared.queue, &[submit], vk::Fence::null())
        } {
            self.tp_free_cmds(&[cmd]);
            return Err(be(format!("tp_submit_copies_wait queue_submit: {e}")));
        }
        Ok(cmd)
    }

    /// Host-fence PUBLISH copy (`src → export`) + a release of `export` to QUEUE_FAMILY_EXTERNAL,
    /// submitted and DRAINED (`queue_wait_idle`) here, freeing the command buffer on every path. The
    /// host-fence / pipeline transports use this (no external semaphore); the release still satisfies
    /// the EXCLUSIVE external buffer's ownership-transfer requirement.
    pub fn p2p_publish_copy(
        &self,
        src: &dyn Buffer,
        export: &dyn Buffer,
        bytes: usize,
    ) -> Result<()> {
        let cmd = self.tp_record_copies(&[(src, export, bytes)], &[], &[export])?;
        self.tp_submit_wait_free(cmd, "p2p_publish_copy")
    }

    /// Host-fence GATHER copy: ACQUIRE `imported` from QUEUE_FAMILY_EXTERNAL then copy
    /// `imported → dst`, submitted + drained here, freeing the command buffer on every path.
    pub fn p2p_gather_copy(
        &self,
        imported: &dyn Buffer,
        dst: &dyn Buffer,
        bytes: usize,
    ) -> Result<()> {
        let cmd = self.tp_record_copies(&[(imported, dst, bytes)], &[imported], &[])?;
        self.tp_submit_wait_free(cmd, "p2p_gather_copy")
    }

    /// Submit `cmd` with no semaphores, wait the queue idle, and free `cmd` — used by the host-fence
    /// barriered copies. Frees the command buffer on the submit-error path too (no leak).
    fn tp_submit_wait_free(&self, cmd: vk::CommandBuffer, what: &str) -> Result<()> {
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let r = unsafe {
            self.shared
                .device
                .queue_submit(self.shared.queue, &[submit], vk::Fence::null())
        };
        if let Err(e) = r {
            self.tp_free_cmds(&[cmd]);
            return Err(be(format!("{what} queue_submit: {e}")));
        }
        let wait = self.tp_queue_wait_idle();
        self.tp_free_cmds(&[cmd]);
        wait
    }

    /// Allocate + record a one-time command buffer that copies each `(src, dst, bytes)`, optionally
    /// wrapped in cross-device queue-family ownership transfers: each buffer in `acquire_external` is
    /// ACQUIRED from `QUEUE_FAMILY_EXTERNAL` before the copies (an imported peer buffer being read),
    /// and each in `release_external` is RELEASED to `QUEUE_FAMILY_EXTERNAL` after them (an exported
    /// buffer just written, handed to a peer). Not submitted here — the caller submits it with the
    /// semaphores/fence it wants. Frees the command buffer on any record error (no leak).
    fn tp_record_copies(
        &self,
        copies: &[(&dyn Buffer, &dyn Buffer, usize)],
        acquire_external: &[&dyn Buffer],
        release_external: &[&dyn Buffer],
    ) -> Result<vk::CommandBuffer> {
        let device = &self.shared.device;
        let qf = self.shared.queue_family_index;
        let pool = *self.shared.cmd_pool.lock().unwrap();
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("tp_record_copies allocate: {e}")))?[0];
        // Any record error past this point must free `cmd` before returning (else it leaks the pool).
        let record = || -> Result<()> {
            unsafe {
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| be(format!("tp_record_copies begin: {e}")))?;
                // ── acquire imported peer buffers from EXTERNAL (before the reads) ────────────────
                if !acquire_external.is_empty() {
                    let barriers: Vec<vk::BufferMemoryBarrier> = acquire_external
                        .iter()
                        .map(|b| {
                            Ok(vk::BufferMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::empty())
                                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                                .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                                .dst_queue_family_index(qf)
                                .buffer(as_vk_buf(*b)?.buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &barriers,
                        &[],
                    );
                }
                for &(src, dst, bytes) in copies {
                    let (s, d) = (as_vk_buf(src)?, as_vk_buf(dst)?);
                    let region = vk::BufferCopy {
                        src_offset: s.sub_offset as u64,
                        dst_offset: d.sub_offset as u64,
                        size: bytes as u64,
                    };
                    device.cmd_copy_buffer(cmd, s.buffer, d.buffer, &[region]);
                }
                // ── release exported buffers to EXTERNAL (after the writes) ───────────────────────
                if !release_external.is_empty() {
                    let barriers: Vec<vk::BufferMemoryBarrier> = release_external
                        .iter()
                        .map(|b| {
                            Ok(vk::BufferMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::empty())
                                .src_queue_family_index(qf)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                                .buffer(as_vk_buf(*b)?.buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &barriers,
                        &[],
                    );
                }
                device
                    .end_command_buffer(cmd)
                    .map_err(|e| be(format!("tp_record_copies end: {e}")))?;
            }
            Ok(())
        };
        if let Err(e) = record() {
            self.tp_free_cmds(&[cmd]);
            return Err(e);
        }
        Ok(cmd)
    }

    /// Block until THIS device's queue is idle (a full memory barrier) — the single residual host
    /// wait per all-reduce, after which the summed partials in scratch are safe for the reduce
    /// dispatch to read. All the cross-device ordering already happened GPU-side on the semaphores.
    pub fn tp_queue_wait_idle(&self) -> Result<()> {
        unsafe { self.shared.device.queue_wait_idle(self.shared.queue) }
            .map_err(|e| be(format!("tp_queue_wait_idle: {e}")))
    }

    /// Free command buffers returned by the semaphore-submit primitives (call only after their work
    /// has completed — i.e. after [`tp_queue_wait_idle`](Self::tp_queue_wait_idle)).
    pub fn tp_free_cmds(&self, cmds: &[vk::CommandBuffer]) {
        if cmds.is_empty() {
            return;
        }
        let pool = *self.shared.cmd_pool.lock().unwrap();
        unsafe { self.shared.device.free_command_buffers(pool, cmds) };
    }
}
