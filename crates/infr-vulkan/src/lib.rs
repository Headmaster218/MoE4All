//! Vulkan backend (`ash` + SPIR-V). The MVP `Backend` impl.
//!
//! Reference: `~/Projects/llama.cpp/ggml/src/ggml-vulkan/` and its `vulkan-shaders/*.comp`
//! (reuse the tuned quant matmul / dequant / attention shaders). Enable device features
//! `VK_KHR_cooperative_matrix`, `shaderFloat16`, `VK_KHR_16bit_storage`,
//! `VK_KHR_shader_subgroup_extended_types`. See docs/plan.md.
#![allow(dead_code)]
// GPU kernel record/dispatch APIs bind many distinct buffers (weights, scales, activations,
// scratch) — wide signatures are inherent here, not a refactor smell.
#![allow(clippy::too_many_arguments)]

mod adapter;
pub mod ep;
mod gemm;
pub mod linear;
mod matmul;
mod ops;
pub mod p2p;
pub mod pager;
mod pcache;
pub mod pipeline;
mod recorder;
pub mod tp;
pub mod tp_allreduce;
pub mod tp_sem;

pub use ep::{EpBuffer, ExpertParallelBackend};
pub use p2p::{P2pExport, P2pHandleType};
pub use pipeline::{PipelineBackend, PipelineBuffer};
pub use recorder::{FlashStage, RecordedCmd, Recorder};
pub use tp::{TensorParallelBackend, TpBuffer, TpRole};
pub use tp_allreduce::{AllReduce, AllReduceMode};
pub use tp_sem::{TpExportSemaphore, TpImportSemaphore};

/// Shared-memory bytes consumed per query row of a flash-attention prefill tile
/// (`Ss` + `Ps` + `Os` + softmax state, at `BN=64` / `HD=128`). The tile height is chosen so
/// `rows * FLASH_SHARED_PER_ROW <= maxComputeSharedMemorySize`; `use_flash` needs the smallest
/// tile (`BM=32`) to fit. Keep in sync with `attn_flash{,_warp,_partial}.comp`.
pub const FLASH_SHARED_PER_ROW: u32 = 908;
/// Same, for the register-O flash tile (`sfsh` + `Psh` + `pvsh` + state); smallest tile is `BR=64`.
/// Keep in sync with `attn_flash_reg.comp`.
pub const FLASH_REG_SHARED_PER_ROW: u32 = 460;

use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;

use infr_core::{
    backend::{Bindings, Buffer, BufferUsage, Capabilities, Plan},
    budget::{spill_report_line, SpillNouns, SpillTally},
    config::Config,
    error::Result,
    graph::Graph,
    Backend,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Terse local shorthand for the shared [`infr_core::error::backend`] constructor.
use infr_core::error::backend as be;

/// Resolve an `INFR_DEV` value to a Vulkan physical-device index (`None` = "use the discrete
/// default"). `INFR_DEV` is the SINGLE device-selection env, sharing the CLI's `--dev` grammar, so
/// it can hold a non-Vulkan spec:
///   * `None` / empty / whitespace → `None` (discrete default),
///   * `metal` / `cpu` (case-insensitive) → `None` — TOLERATED, not an error: a process only
///     reaches the Vulkan constructor when it actually built a Vulkan backend (e.g. a non-macOS
///     build), so a leftover non-Vulkan spec must not hard-fail device selection,
///   * anything else is treated as a Vulkan index (`VulkanN`, `vulkanN`, or a bare `N`): parsed,
///     and range-checked against `device_names.len()`. An unparseable or out-of-range value is a
///     HARD ERROR — silently running on a different GPU than asked produces plausible-but-wrong
///     numbers, so a typo must fail loudly. `device_names` (`"Vulkan0=<name>"`, …) feeds the
///     "no such device" message.
fn resolve_infr_dev_index(spec: Option<&str>, device_names: &[String]) -> Result<Option<usize>> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let s = spec.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let lower = s.to_ascii_lowercase();
    if lower == "metal" || lower == "cpu" {
        return Ok(None);
    }
    let idx_str = lower.strip_prefix("vulkan").unwrap_or(&lower);
    let idx: usize = idx_str.parse().map_err(|_| {
        be(format!(
            "INFR_DEV/--dev: expected `VulkanN` (e.g. Vulkan0, Vulkan1), got `{spec}`"
        ))
    })?;
    if idx >= device_names.len() {
        return Err(be(format!(
            "INFR_DEV/--dev `{spec}`: no such Vulkan device (this system has {}: {})",
            device_names.len(),
            device_names.join(", ")
        )));
    }
    Ok(Some(idx))
}

/// Downcast `&dyn Buffer` → `&VkBuffer`, CHECKED.
///
/// This used to be an `unsafe fn` that reinterpreted the trait object's data pointer
/// (`b as *const dyn Buffer as *const () as *const VkBuffer`) with no type check at all, its
/// "must only be called with buffers returned by `VulkanBackend::alloc`" contract enforced purely
/// by convention at ~30 call sites. The `Buffer` trait carries `as_any`, so the check is one
/// `TypeId` comparison — and there are now several other `Buffer` impls a `&dyn Buffer` can
/// legitimately be (`TpBuffer`, `EpBuffer`, `PipelineBuffer`, `infr-cpu`'s `CpuBuffer`,
/// `infr-metal`'s `MetalBuffer`). `infr multi` hosts several backends in ONE process and the MTP
/// draft path can mix them, so a mis-routed buffer is no longer a hypothetical: unchecked, it reads
/// a foreign struct's first fields as a `vk::Buffer` handle plus offsets and hands them to the
/// driver — undiagnosable memory corruption or a device loss, arbitrarily far from the routing bug.
/// Checked, it is an ordinary backend error naming the operation.
///
/// Hot path: this runs per dispatch, so the success path must stay a `TypeId` compare and nothing
/// else — the message is built in the failure branch only (`ok_or_else`), never formatted eagerly.
/// `&dyn Buffer` exposes no type name, so the error reports what it CAN see (the logical extent and
/// whether the buffer carries a device address) plus the impls it plausibly was.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn as_vk_buf(b: &dyn Buffer) -> Result<&VkBuffer> {
    b.as_any().downcast_ref::<VkBuffer>().ok_or_else(|| {
        be(format!(
            "vulkan: buffer was not allocated by this VulkanBackend ({} bytes, device_addr={}) — \
             it is some other `Buffer` impl (a TpBuffer/EpBuffer/PipelineBuffer wrapper, or another \
             backend's buffer entirely). Unwrap it to the underlying VkBuffer before handing it to \
             a Vulkan op.",
            b.len_bytes(),
            if b.device_addr().is_some() {
                "yes"
            } else {
                "no"
            },
        ))
    })
}

/// Bounds-check ONE side of a device transfer: `bytes` must fit inside a buffer's LOGICAL extent
/// ([`VkBuffer::size`] — what the caller asked for, and what [`Buffer::len_bytes`] reports).
///
/// `size`, NOT the underlying `vk::Buffer`'s extent, is the right bound. A resident-BDA sub-tensor
/// (`Backing::BdaSub`) shares one big arena `vk::Buffer` with its neighbours and lives at
/// `sub_offset` inside it, so the whole-object extent would happily "validate" a transfer that runs
/// straight over the next sub-tensor. Every transfer path already folds `sub_offset` into its copy
/// region and then moves exactly `bytes`, so `bytes <= size` is precisely the condition that keeps
/// the touched range `[sub_offset, sub_offset + bytes)` inside this tensor's own slice of the block.
///
/// `op`/`role` only shape the message (`upload`'s wording, which predates this helper, is the
/// template): "upload: 64 bytes into a 32-byte buffer".
fn check_extent(op: &str, role: &str, bytes: usize, size: usize) -> Result<()> {
    if bytes > size {
        return Err(be(format!(
            "{op}: {bytes} bytes {role} a {size}-byte buffer"
        )));
    }
    Ok(())
}

// ── device class (process-global) ─────────────────────────────────────────────

/// The class of the Vulkan device this PROCESS opened — see [`device_class`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceClass {
    /// `deviceType == INTEGRATED_GPU` (see [`Capabilities::integrated`]).
    pub integrated: bool,
    /// Compute units, or 0 = unknown (see [`Capabilities::compute_units`]).
    pub compute_units: u32,
}

/// Set ONCE by the first [`VulkanBackend::new`] in the process.
static DEVICE_CLASS: std::sync::OnceLock<DeviceClass> = std::sync::OnceLock::new();

/// The class of the Vulkan device this process opened, or `None` when no Vulkan backend has been
/// constructed (a CPU/Metal run, or a GPU-less box).
///
/// A PROCESS-GLOBAL because its one consumer, the seam's `ubatch_rows`, is itself a process-global
/// funnel: the prefill loop, the activation reserve, and the SWA ring sizing must all agree on ONE
/// chunk height, and they are reached from call sites that hold no backend handle. Same shape and
/// lifetime as the seam's existing `PINNED_UBATCH`. A multi-GPU process mixing an iGPU and a dGPU
/// would pin whichever opened first; infr opens exactly one device per process today.
pub fn device_class() -> Option<DeviceClass> {
    DEVICE_CLASS.get().copied()
}

// ── shared GPU state ──────────────────────────────────────────────────────────

/// Device memory snapshot from [`VulkanBackend::vram`]. `available` is live free bytes when
/// `live` is true (VK_EXT_memory_budget present), otherwise it equals `total` (best-effort).
///
/// WHICH HEAPS THIS COUNTS depends on the device class (see [`vram_info`]): device-local only on a
/// discrete card, ALL heaps on a unified-memory part where they are the same physical DDR.
#[derive(Clone, Copy, Debug)]
pub struct VramInfo {
    pub total: u64,
    pub available: u64,
    pub live: bool,
    /// True when this snapshot counted every heap because the device has unified memory (see
    /// [`Capabilities::unified_memory`]) — only affects how the guard words its error.
    pub uma: bool,
}

struct VulkanShared {
    // NOTE: field declaration order matters for drop.
    // Rust drops struct fields in *declaration order*.  We keep `allocator`
    // in a `ManuallyDrop` so we can drop it explicitly before calling
    // `destroy_device` in the `Drop` impl.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    /// Serialises all one-shot command-buffer submissions.
    cmd_pool: Mutex<vk::CommandPool>,
    /// Must be dropped before the device is destroyed.
    allocator: ManuallyDrop<Mutex<Allocator>>,
    caps: Capabilities,
    /// VK_EXT_memory_budget enabled → `vram()` can report live free bytes (else total only).
    has_mem_budget: bool,
    /// `maxMemoryAllocationSize` — the largest single `vkAllocateMemory` this device accepts
    /// (`VkPhysicalDeviceMaintenance3Properties`, ~4 GiB on RADV). The weight arena
    /// (`reserve_weights`) splits its up-front reservation into blocks no larger than this, since a
    /// single alloc of a whole multi-GiB model is impossible. Falls back to a conservative 1 GiB
    /// (the Vulkan-guaranteed floor) if the device reports 0.
    max_mem_alloc_size: u64,
    /// `VK_KHR_push_descriptor` loader, when the device supports it — every dispatch's
    /// descriptor binding then records via `cmd_push_descriptor_set` (recorder.rs
    /// `bind_descriptors`) instead of a pooled `alloc_set` + `update_descriptor_sets` +
    /// `cmd_bind_descriptor_sets` per op. `None` falls back to the pooled path.
    push_descriptor: Option<ash::khr::push_descriptor::Device>,
    /// `VK_KHR_external_memory_fd` loader (`vkGetMemoryFdKHR` / `vkGetMemoryFdPropertiesKHR`), when
    /// the device enabled it. `Some` is the gate for the host-less cross-device P2P transport (a
    /// buffer's memory exported as an fd on one backend and imported on another — see `p2p.rs`).
    /// `None` on any device/driver without the extension, in which case no P2P path is offered and
    /// the default single-device behaviour is unchanged.
    external_memory_fd: Option<ash::khr::external_memory_fd::Device>,
    /// True when this device enabled `VK_EXT_external_memory_dma_buf`, so the P2P export/import may
    /// use the dma-buf handle type (the cross-GPU-portable one on Linux) in addition to opaque-fd.
    has_dma_buf: bool,
    /// `VK_KHR_external_semaphore_fd` loader — exports/imports a semaphore fd so a tensor-parallel
    /// all-reduce can order a peer's read after this device's GPU-side signal with no host round-trip
    /// (`AllReduceMode::P2pSemaphore`). `None` = the all-reduce uses the host fence (`queue_wait_idle`)
    /// instead. `Some` whenever the device enabled the extension, in which case the semaphore-ordered
    /// all-reduce path is LIVE (see `external_semaphore_supported`); a device that can't import a
    /// cross-device semaphore falls back to the host fence.
    external_semaphore_fd: Option<ash::khr::external_semaphore_fd::Device>,
    /// Generic cache of compute kernels by name (see `ops.rs`).
    kernels: Mutex<HashMap<&'static str, crate::ops::ComputeKernel>>,
    /// Device pipeline cache, seeded from disk at init and persisted back (see `pcache.rs`) so
    /// pipeline creation after the first-ever launch reuses cached driver binaries. Null when
    /// creation failed (caching is then simply off — Vulkan accepts a null cache everywhere).
    pipeline_cache: vk::PipelineCache,
    /// Disk persistence for `pipeline_cache`; `None` = INFR_NO_PIPELINE_CACHE or no cache dir.
    pcache: Option<crate::pcache::PcachePersist>,
    /// Active weight-load progress bar (see [`VulkanBackend::weight_progress`]). Every
    /// `BufferUsage::Weights` allocation advances it, so no model loader can forget to tick it.
    weight_pb: Mutex<Option<indicatif::ProgressBar>>,
    /// Cumulative device-local bytes THIS backend has committed (pooled/dedicated allocations +
    /// weight-arena blocks). The VRAM budget guard's fallback accounting when
    /// VK_EXT_memory_budget is absent — the live per-heap budget is preferred when present
    /// (it also sees other processes' VRAM).
    device_used: AtomicU64,
    /// SUBMIT SPLITTER: the most dispatches `execute_static` will record into one command buffer
    /// before submitting it and opening the next (`0` = unlimited, never split).
    ///
    /// The GPU hang watchdog is armed per SUBMIT, so a forward pass recorded as one command buffer
    /// is one indivisible watchdog job. On a 2-CU integrated part that job is ~2.05 s of real GPU
    /// work and the device kills it at ~2.06 s — a margin so thin it was a coin flip, which is the
    /// `ring gfx_0.0.0 timeout` -> `VK_ERROR_DEVICE_LOST` this exists to prevent. Splitting the
    /// SAME work across N command buffers divides the per-job duration by N without removing any
    /// work: the segments still run back-to-back on the queue (`finish_nowait`, no host sync), the
    /// watchdog just gets N short jobs to watch instead of one long one.
    ///
    /// Seeded from `infr_core::initial_submit_dispatch_cap` (unlimited on discrete — a dGPU
    /// forward is tens of ms and must not pay for barriers it does not need) and then RE-TUNED
    /// from measurement after every forward (`infr_core::submit_cap_from_measurement`), so the
    /// bound tracks whatever the device actually is rather than a table of magic numbers.
    submit_dispatch_cap: AtomicUsize,
    /// UNIFIED-MEMORY parts only (`None` on every discrete GPU): the host-visible memory type on
    /// the non-device-local heap that `GpuOnly` allocations SPILL into once the device-local heap
    /// is full. See [`probe_uma_overflow_type`] for why counting that heap in the budget is not
    /// enough on its own — the bytes have to be able to land there too.
    uma_overflow_type: Option<u32>,
    /// The host-visible memory type on a NON-device-local heap, probed on EVERY device (unlike
    /// `uma_overflow_type`, which is UMA-only). On a discrete card this heap is system RAM across
    /// PCIe. `None` if the device exposes no such type. Used ONLY by the opt-in
    /// `INFR_KV_OVERFLOW` path to place the KV cache in system RAM (read by attention over PCIe
    /// via its device address — the KV read seam is 100% `bufferDeviceAddress`, so the bytes may
    /// live off-device with no shader change). See [`Self::alloc_kv_host`].
    host_overflow_type: Option<u32>,
    /// VRAM-first KV-overflow placement tally (`INFR_KV_OVERFLOW`): how many `BufferUsage::KvCache`
    /// buffers landed in device-local VRAM vs spilled to system RAM, and their byte totals. Purely
    /// for the one-shot placement banner (`kv_overflow_report`) so the user sees the resident/spilled
    /// split; the actual budgeting is `device_used` alone. All 0 when the flag is off. The
    /// bookkeeping (and the cumulative-cap gate) is the shared [`SpillTally`].
    kv_spill: SpillTally,
    /// Reused staging ring for weight uploads (see [`StagingRing`]). Built lazily on
    /// the first staged weight upload of a load and torn down with the weight scope.
    staging_ring: Mutex<Option<StagingRing>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanShared {
    /// Wait for every in-flight staging copy and tear the ring down. Called when the weight-load
    /// scope ends, so all weights are fully resident before any forward is recorded — this is the
    /// synchronization point that replaced the old per-tensor `queue_wait_idle`.
    fn drain_staging_ring(&self) {
        let Some(mut ring) = self.staging_ring.lock().unwrap().take() else {
            return;
        };
        let pending: Vec<vk::Fence> = (0..RING_SLOTS)
            .filter(|&i| ring.busy[i])
            .map(|i| ring.fences[i])
            .collect();
        unsafe {
            if !pending.is_empty() {
                let _ = self.device.wait_for_fences(&pending, true, u64::MAX);
            }
            let pool = *self.cmd_pool.lock().unwrap();
            self.device.free_command_buffers(pool, &ring.cmds);
            for f in ring.fences.drain(..) {
                self.device.destroy_fence(f, None);
            }
        }
        // `ring.bufs` drop here → the staging slots are freed.
    }
}

// ash Instances/Devices/handles are Send+Sync per the Vulkan spec when
// accessed through our Mutexes.
unsafe impl Send for VulkanShared {}
unsafe impl Sync for VulkanShared {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanShared {
    /// Debounced disk save of the pipeline cache — call after a NEW pipeline lands so long-lived
    /// processes (serve) persist without waiting for a clean Drop.
    pub(crate) fn persist_pipeline_cache(&self) {
        if let Some(pc) = &self.pcache {
            pc.maybe_save(&self.device, self.pipeline_cache);
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for VulkanShared {
    fn drop(&mut self) {
        unsafe {
            // Also the pipeline cache's TRIPWIRE verdict (see `pcache.rs`): VK_ERROR_DEVICE_LOST is
            // STICKY — once the device is lost every call returns it — so this one drain doubles as
            // "did this run hang the GPU?", with no flag to thread through every submit site.
            let device_lost = matches!(
                self.device.device_wait_idle(),
                Err(vk::Result::ERROR_DEVICE_LOST)
            );
            if let Ok(map) = self.kernels.lock() {
                for k in map.values() {
                    crate::ops::destroy_compute_kernel(&self.device, k);
                }
            }
            // Persist the pipeline cache (final save — the debounced mid-run saves may have
            // missed the tail) and destroy it. On a LOST device this discards the file instead of
            // saving it, and either way it disarms this process's tripwire marker.
            if let Some(pc) = &self.pcache {
                pc.finish(&self.device, self.pipeline_cache, device_lost);
            }
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            // Destroy command pool.
            let pool = *self.cmd_pool.lock().unwrap();
            self.device.destroy_command_pool(pool, None);
            // Drop the allocator *before* destroying the device.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── VkBuffer ──────────────────────────────────────────────────────────────────

/// How a `VkBuffer`'s device memory is owned.
enum Backing {
    /// A gpu-allocator sub-allocation — freed back to the allocator on drop (transient buffers,
    /// host-visible staging/readback).
    Pooled(ManuallyDrop<Allocation>),
    /// A DEDICATED `VkDeviceMemory` this buffer owns outright, PERSISTENTLY MAPPED — today only the
    /// UNIFIED-MEMORY overflow spill (`spilled: true`, see `probe_uma_overflow_type`): a GpuOnly
    /// buffer placed on the non-device-local heap once the synthetic device-local heap is full.
    /// `upload` memcpys straight through the mapped pointer. Freed (unmapped + `vkFreeMemory`) on
    /// drop.
    Vram {
        memory: vk::DeviceMemory,
        ptr: *mut u8,
        /// True when this is a UNIFIED-MEMORY SPILL (see `probe_uma_overflow_type`): the memory came
        /// from the non-device-local overflow heap, so it is NOT charged to `device_used` (the
        /// budget guard's device-local tally) — leaving the spill decision to ask "is the
        /// DEVICE-LOCAL heap full?" without the answer being polluted by the bytes it already
        /// spilled elsewhere. A `false` (device-local mapped) buffer is charged to `device_used`
        /// like any other GpuOnly allocation.
        spilled: bool,
    },
    /// A logical BYTE RANGE of a [`BdaWeightArena`] block's single big `vk::Buffer` (resident weight
    /// sub-tensors — see [`VulkanBackend::bda_weight_alloc`]). Unlike every other
    /// variant, `VkBuffer::buffer` here is NOT this handle's own object: it is the block's buffer,
    /// shared byte-for-byte with every other sub-tensor carved from the same block, and with the
    /// block's own keepalive copy. The `Arc` is what keeps the block (and therefore its memory and
    /// buffer handle) alive for as long as any sub-tensor referencing it is alive; dropping this
    /// variant frees NOTHING — no `destroy_buffer`, no memory free — that happens exactly once, when
    /// the last `Arc<BdaBlockHandle>` clone (the arena's own, or the last live sub-tensor's) drops
    /// and `BdaBlockHandle::buf`'s ordinary `VkBuffer::drop` runs.
    ///
    /// Descriptor binds of a sub-tensor (`recorder::Recorder::vkb`) are legal as long as they carry
    /// this tensor's own `(sub_offset, range)`, never `(0, WHOLE_SIZE)` — the latter describes the
    /// whole shared block, not the tensor. A big matmul weight is instead read through its 64-bit
    /// `device_addr()` by a `-DSTREAMED` shader twin, required once the range would exceed
    /// `maxStorageBufferRange`/4 GiB and preferred for the big matmul families regardless.
    BdaSub(Arc<BdaBlockHandle>),
    /// A DEDICATED `VkDeviceMemory` allocated with an EXTERNAL handle type (dma-buf / opaque-fd) —
    /// the cross-device P2P path (see `p2p.rs`). Two flavours share this variant. On the EXPORT
    /// side (device A) the memory is allocated with `VkExportMemoryAllocateInfo` and its fd handed
    /// out by `vkGetMemoryFdKHR`; this buffer keeps the underlying pages alive for device B. On the
    /// IMPORT side (device B) the memory is allocated with `VkImportMemoryFdInfoKHR`, ALIASING
    /// device A's physical bytes — reads/writes here go straight to A's memory over PCIe, no host
    /// copy.
    ///
    /// Never host-mapped (`mapped_ptr` = `None`, so `upload`/`download` use the staging path).
    /// Freed with a plain `vkFreeMemory` on drop (no unmap). Deliberately OUTSIDE the VRAM budget
    /// accounting: this is a gated probe/transport capability, not wired into any model path, and
    /// the import side aliases memory already owned by the exporting backend.
    External { memory: vk::DeviceMemory },
}

struct VkBuffer {
    shared: Arc<VulkanShared>,
    buffer: vk::Buffer,
    backing: Backing,
    /// Logical buffer size (what the caller asked for and what `upload`/`fill_buf` touch).
    size: usize,
    /// Device-memory bytes actually committed for this buffer (`requirements.size`, i.e. `size`
    /// rounded up for alignment). Charged to / released from the VRAM budget guard's accounting
    /// by [`Backing::Vram`], which owns its `VkDeviceMemory` outright.
    mem_size: u64,
    location: MemoryLocation,
    /// Byte offset of this tensor's logical range within `buffer` — `0` for every buffer except a
    /// resident-BDA weight sub-tensor (`Backing::BdaSub`, see [`BdaWeightArena`]), where several
    /// `VkBuffer` handles share ONE big `vk::Buffer` (the arena block) and this field is what tells
    /// them apart. Every upload/download/fill site that touches `buffer` at a byte offset must add
    /// this in, and [`Buffer::device_addr`] is `block_base_addr + sub_offset`.
    sub_offset: usize,
    /// `vkGetBufferDeviceAddress(buffer)` for a buffer that owns ITS OWN `SHADER_DEVICE_ADDRESS`
    /// buffer object (unlike `Backing::BdaSub`, which shares an arena block's handle and derives
    /// its address from the block's `base_addr + sub_offset` instead). Populated by `make_buf_ex`/
    /// `alloc_vram_mapped` whenever they were asked for a device address — today that's the
    /// resident-BDA/paged-MoE arena blocks themselves (`Backing::Pooled`/`Backing::Vram`) and
    /// `BufferUsage::KvCache` allocations (see [`Buffer::device_addr`]). `None` for every buffer
    /// that never requested one.
    own_addr: Option<u64>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VkBuffer {
    /// Persistently-mapped host pointer for host-visible buffers — pooled host-visible allocations
    /// AND [`Backing::Vram`] weights (device-local VRAM the host can write through, via ReBAR).
    /// `None` for plain device-local or arena buffers, which are filled via a staging copy.
    /// Every "can I just memcpy into this?" decision (`upload`, `fill_buf`) keys off this.
    fn mapped_ptr(&self) -> Option<*mut u8> {
        match &self.backing {
            Backing::Pooled(a) => a.mapped_ptr().map(|p| p.as_ptr() as *mut u8),
            Backing::Vram { ptr, .. } => Some(*ptr),
            // Defensive, not currently reachable: `bda_weight_alloc`'s blocks are plain `GpuOnly`
            // dedicated allocations (never host-mapped), so `buf.mapped_ptr()` is `None` today. If a
            // future block ever WERE host-visible, offsetting by `sub_offset` here is what keeps
            // every "can I just memcpy?" call site (`upload`/`fill_buf`) correct without change.
            Backing::BdaSub(block) => block
                .buf
                .mapped_ptr()
                .map(|p| unsafe { p.add(self.sub_offset) }),
            // External P2P memory is never host-mapped — reads/writes route through the staging
            // copy path (device A owns the pages; device B aliases them over PCIe).
            Backing::External { .. } => None,
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for VkBuffer {
    fn drop(&mut self) {
        unsafe {
            match &mut self.backing {
                Backing::Pooled(alloc) => {
                    let alloc = ManuallyDrop::take(alloc);
                    // Keep the budget guard's fallback accounting balanced.
                    if self.location == MemoryLocation::GpuOnly {
                        self.shared
                            .device_used
                            .fetch_sub(alloc.size(), Ordering::Relaxed);
                    }
                    self.shared.allocator.lock().unwrap().free(alloc).ok();
                }
                // A dedicated VkDeviceMemory we own outright — today only the UMA overflow spill
                // (`spilled: true`). Only a device-local mapped buffer (`spilled: false`) is charged
                // to `device_used` at allocation (see `make_buf_ex`) — a UMA spill lives off the
                // device-local heap and is never counted there — so balance that same charge here.
                Backing::Vram {
                    memory, spilled, ..
                } => {
                    if !*spilled {
                        self.shared
                            .device_used
                            .fetch_sub(self.mem_size, Ordering::Relaxed);
                    }
                    self.shared.device.unmap_memory(*memory);
                    self.shared.device.free_memory(*memory, None);
                }
                // Shares the block's `vk::Buffer` handle byte-for-byte with every other sub-tensor
                // and the block's own keepalive copy — this handle owns NEITHER the buffer object
                // NOR its memory, only an `Arc` clone. Dropping the `Arc` (below, implicitly, when
                // `self.backing` itself drops) is the whole of this handle's cleanup; the actual
                // `destroy_buffer`/memory-free happens once, inside `BdaBlockHandle::buf`'s own
                // `VkBuffer::drop`, when the last clone goes away.
                Backing::BdaSub(_) => {}
                // A dedicated external-memory allocation (P2P export or import). Never host-mapped,
                // so no unmap — just free the memory. On the export side this releases device A's
                // pages once no importer still references the underlying dma-buf/fd (each side owns
                // an independent `VkDeviceMemory` over the same pages, freed independently).
                Backing::External { memory } => {
                    self.shared.device.free_memory(*memory, None);
                }
            }
            // Every OTHER variant owns `self.buffer` outright and must destroy it here. A `BdaSub`
            // handle's `buffer` is an alias of the block's — destroying it here would destroy the
            // block out from under every other sub-tensor still referencing it (and double-free when
            // the block's own `VkBuffer` later drops), so it is the one variant that skips this.
            if !matches!(self.backing, Backing::BdaSub(_)) {
                self.shared.device.destroy_buffer(self.buffer, None);
            }
        }
    }
}

unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Buffer for VkBuffer {
    fn len_bytes(&self) -> usize {
        self.size
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device_addr(&self) -> Option<u64> {
        if let Some(addr) = self.own_addr {
            return Some(addr);
        }
        match &self.backing {
            Backing::BdaSub(block) => Some(block.base_addr + self.sub_offset as u64),
            _ => None,
        }
    }
}

// ── weight arena ────────────────────────────────────────────────────────────────

/// Buffer usage flags for every device buffer (must match across the arena probe and all
/// allocations so their memory-type bits / alignment agree).
const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        // Any buffer may serve as vkCmdDispatchIndirect args (the split-K replay prologue writes
        // the partial pass's workgroup count GPU-side).
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
);

/// On-demand block size floor for a resident-BDA arena block (see [`BdaWeightArena`]) — big enough
/// to amortize a dedicated `vkAllocateMemory` across a run of small tensors, small enough not to
/// waste much on the tail.
const ARENA_OVERFLOW_BLOCK: u64 = 64 * 1024 * 1024;

/// The UMA OVERFLOW memory type: a host-visible type on a NON-device-local heap. `None` on a
/// discrete GPU (never probed) and on any UMA part that doesn't expose one.
///
/// This is the other half of the unified-memory fix, and without it widening the budget is not
/// merely useless but actively harmful. `vram_info` budgets a UMA part against ALL heaps, but
/// gpu-allocator resolves `MemoryLocation::GpuOnly` to the FIRST DEVICE_LOCAL memory type and
/// never falls back — so every allocation lands on the device-local heap no matter how full it is.
/// RADV does not enforce the heap size (a 41 GiB run of 1 GiB allocations succeeded on a
/// "21.47 GiB" heap), so nothing errors; the kernel simply can no longer validate the buffer list
/// and the next SUBMIT dies with "Not enough memory for command submission" — a device-lost, i.e.
/// exactly the silent-degradation failure the guard exists to prevent, just moved later.
/// MEASURED on RAPHAEL_MENDOCINO with gemma-4-31B: weights + KV + activations cross the
/// device-local heap's 21.47 GiB and the guard sits there reporting 10.70 GiB "available" (which
/// is precisely heap 0's size — capacity nothing could reach) while the submit fails.
///
/// So the overflow must be PLACED, not just counted. On an APU heap 0 is the same DDR at the same
/// bandwidth as the synthetic device-local heap — the weights are read out of GTT either way
/// (`mem_info_gtt_used` accounts for them on both paths) — so spilling there costs no bandwidth.
/// It is only on a DISCRETE card that the non-device-local heap means "across PCIe", which is why
/// this is probed for UMA parts alone.
fn probe_uma_overflow_type(mp: &vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    (0..mp.memory_type_count).find(|&i| {
        let t = mp.memory_types[i as usize];
        let heap = mp.memory_heaps[t.heap_index as usize];
        t.property_flags.contains(want)
            && !t
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            && !heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
    })
}

/// Opt-in: place the KV cache VRAM-FIRST, spilling to system RAM only what does not fit. Each
/// per-layer/per-side KV buffer is tried in device-local VRAM (budget-guarded); once the VRAM
/// budget is reached, that buffer — and, since the budget only shrinks as later buffers land, every
/// subsequent one — is placed in host RAM and read by attention over PCIe via its device address
/// (the KV read seam is 100% `bufferDeviceAddress`, so off-device bytes need no shader change). So a
/// context whose KV overflows VRAM by a modest amount keeps most layers resident and pays PCIe only
/// on the spilled tail; a context that overflows entirely spills entirely (slice-1 whole-host as the
/// limiting case). `INFR_KV_OVERFLOW=1` (config `kv.overflow`). Default OFF (empty or `0` = off) =
/// today's VRAM-only behavior, unchanged. See [`VulkanBackend::alloc_kv_host`],
/// [`VulkanBackend::vram_budget_fits`], and the ctx-clamp ladder's last rung.
///
/// Read off the backend's `Config`, not the process environment; `budget::flag_from`'s grammar
/// (empty and `"0"` are OFF, unlike every `is_ok()` knob) is preserved by the config layer.
fn kv_overflow_enabled(cfg: &infr_core::config::Config) -> bool {
    cfg.kv.overflow
}

/// Nouns for this backend's KV placement banner (see [`spill_report_line`], which owns the
/// skeleton every spill class shares). The spill clause is Vulkan-specific twice over: the host
/// bytes are ordinary host-visible memory (not page-locked) and attention reaches them
/// by DEVICE ADDRESS, which is the fact that makes off-device KV work at all here.
const KV_SPILL: SpillNouns<'static> = SpillNouns {
    env: "INFR_KV_OVERFLOW",
    noun: "KV buffers",
    resident_note: "no PCIe KV reads.",
    spill_note: "SYSTEM RAM — attention reads those K/V over PCIe by device address (PCIe-bound \
                 on the spilled layers). Spilled KV bytes are exempt from the VRAM budget guard.",
};

/// Diagnostic cap (in MiB) on CUMULATIVE KV bytes the VRAM-first spill will place in device-local
/// VRAM before spilling the rest to host: `INFR_KV_OVERFLOW_VRAM_MB`. Unset ⇒ no cap (spill only
/// when VRAM is genuinely full). `0` ⇒ nothing resident (whole-host, the slice-1 case). Its ONLY
/// purpose is to make the partial-spill mix and the whole-host case reproducible on models that
/// would otherwise fit entirely — for tests and apples-to-apples benchmarking. Gates KV placement
/// alone, never the real VRAM guard. Ignored when `INFR_KV_OVERFLOW` is off.
///
/// Carried on the `Config` in MiB (`kv.overflow_vram_mb`); the byte conversion is this accessor's.
fn kv_overflow_vram_cap(cfg: &infr_core::config::Config) -> Option<u64> {
    infr_core::budget::mib_bytes(cfg.kv.overflow_vram_mb)
}

/// Headroom the VRAM budget reserves below the true heap size, shared by the hard guard
/// ([`VulkanBackend::check_vram_budget`]) and the non-erroring probe
/// ([`VulkanBackend::vram_budget_fits`]) so VRAM-first KV spill and the guard agree to the byte on
/// where "full" is. Absorbs allocation slop (alignment, gpu-allocator block rounding) and
/// driver-internal allocations (descriptor pools, pipeline/shader memory, command buffers).
const GUARD_HEADROOM: u64 = 256 * 1024 * 1024;

/// Free bytes on the DEVICE-LOCAL heaps alone — what the UMA spill decision keys off (unlike
/// [`vram_info`]'s UMA figure, which spans every heap). Live VK_EXT_memory_budget when present, so
/// a device-local heap another process has filled reads as full here too; otherwise the heap size
/// minus this process's tracked device-local bytes (`device_used`; UMA-spilled bytes never touch
/// `device_used`, so they are correctly excluded).
fn device_local_room(s: &VulkanShared) -> u64 {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default();
    if s.has_mem_budget {
        props2 = props2.push_next(&mut budget);
    }
    unsafe {
        s.instance
            .get_physical_device_memory_properties2(s.physical_device, &mut props2)
    };
    let mp = props2.memory_properties;
    let (mut size, mut avail) = (0u64, 0u64);
    for i in 0..mp.memory_heap_count as usize {
        if mp.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        {
            size += mp.memory_heaps[i].size;
            avail += budget.heap_budget[i]
                .saturating_sub(budget.heap_usage[i])
                .min(mp.memory_heaps[i].size);
        }
    }
    if s.has_mem_budget {
        avail
    } else {
        size.saturating_sub(s.device_used.load(Ordering::Relaxed))
    }
}

/// Human byte count for the budget guard's error, in the LARGEST unit that keeps a significant
/// digit. A fixed `{:.2} GiB` printed "0.00 GiB requested" for anything under ~5 MiB — a guard
/// error that reads as nonsense exactly when it fires on a small allocation (the last straw on a
/// budget the big tensors already filled).
fn fmt_bytes(b: u64) -> String {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    match b {
        b if b >= GIB => format!("{:.2} GiB", b as f64 / GIB as f64),
        b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
        b if b >= KIB => format!("{:.1} KiB", b as f64 / KIB as f64),
        b => format!("{b} B"),
    }
}

/// Byte span a device-local `vkCmdFillBuffer` must cover to fully zero-init (calloc contract) a
/// buffer whose LOGICAL size is `logical_size`. `vkCmdFillBuffer` requires a 4-byte-multiple size,
/// so round UP — the old `size / 4 * 4` truncation left the trailing 1-3 bytes of a
/// non-multiple-of-4 buffer holding recycled VRAM, violating `Backend::alloc`'s zero-init
/// guarantee. The backing buffer is CREATED at this same rounded size (see `make_buf_ex` /
/// `alloc_kv_host`), so the rounded fill stays in-bounds (`dstOffset + size <= buffer size`).
/// Identity for any 4-aligned size — every current tensor is 4-aligned, so the fill is byte-for-
/// byte what it was before this rounding existed.
fn fill_span(logical_size: usize) -> u64 {
    (logical_size as u64).next_multiple_of(4)
}

/// Human-readable Vulkan device class for the enumeration listing.
fn device_type_str(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

/// A physical device as seen by [`VulkanBackend::enumerate_devices`]. `index` is the `VulkanN` /
/// `INFR_DEV` / [`VulkanBackend::new_on`] handle. The `external_memory*` flags report whether the
/// device could, in principle, participate in a host-less GPU↔GPU transfer (dma-buf / fd import) —
/// the P2P feasibility signal for the multi-GPU campaign; they do NOT imply a P2P path is wired.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub device_type: &'static str,
    pub integrated: bool,
    /// Sum of DEVICE_LOCAL heap sizes (a UMA part reports its GTT-backed heap here).
    pub vram_bytes: u64,
    /// True for the device `VulkanBackend::new()` would bind today (INFR_DEV, else discrete, else 0).
    pub is_default_pick: bool,
    pub external_memory: bool,
    pub external_memory_fd: bool,
    pub external_memory_dma_buf: bool,
}

/// Copy `src` into a persistently-mapped destination, in PARALLEL for large buffers.
///
/// For a ReBAR weight the destination is write-combined VRAM across PCIe, where a single core
/// cannot saturate the link (measured ~8.8 GB/s single-threaded on a 7900 XTX / PCIe 4.0 x16).
/// Splitting the copy across cores lets several write-combine streams be in flight at once. Small
/// buffers copy inline — below the threshold the rayon fork/join costs more than it saves.
fn copy_to_mapped(src: &[u8], dst: *mut u8) {
    /// Below this, a plain memcpy beats paying for fork/join.
    const PAR_MIN: usize = 4 * 1024 * 1024;
    /// Chunk per task: big enough to amortize scheduling, small enough to spread over cores.
    const PAR_CHUNK: usize = 2 * 1024 * 1024;

    if src.len() < PAR_MIN {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    // `dst` is valid for `src.len()` bytes and the chunks are disjoint, so the per-chunk raw
    // writes never alias. `usize` is carried across the thread boundary because `*mut u8` is
    // not `Send`.
    let base = dst as usize;
    src.par_chunks(PAR_CHUNK).enumerate().for_each(|(i, c)| {
        let off = i * PAR_CHUNK;
        unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), (base + off) as *mut u8, c.len()) };
    });
}

/// A ring of REUSED, fixed-size staging buffers — the weight-upload path on every device.
///
/// The old path allocated a fresh DEDICATED staging buffer as large as each tensor, memcpy'd into
/// it, submitted a copy, then `vkQueueWaitIdle`'d and freed it — per tensor. That serialized the
/// host memcpy against the DMA and paid an allocate/submit/stall/free cycle 443 times.
///
/// Here the ring is allocated ONCE per load. Big tensors are chunked across slots, and each slot
/// carries its own command buffer + fence, so while slot N's DMA is in flight the host is already
/// memcpy'ing into slot N+1 — the copy engine and the CPU overlap instead of taking turns. A slot
/// is only waited on when it is reused (its fence), never after every tensor. That overlap is the
/// whole point: it hides the PCIe crossing behind a full-speed system-RAM memcpy.
struct StagingRing {
    /// Fixed-size host-visible staging slots (`RING_SLOTS` × `RING_SLOT_BYTES`).
    bufs: Vec<VkBuffer>,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    /// Whether slot `i` has work in flight that its fence must be waited on before reuse.
    busy: Vec<bool>,
    next: usize,
}

/// Staging-ring geometry: enough slots to keep the copy engine fed while the host fills the next.
const RING_SLOTS: usize = 4;
const RING_SLOT_BYTES: usize = 32 * 1024 * 1024;

// ── resident-BDA weight arena ──────────────────────────────────────────────────
//
// Allocation-side plumbing for resident weights: they live in big BDA arena blocks addressed by
// 64-bit device address — the same `bufferDeviceAddress` scheme the paged-MoE/dense-streaming
// kernels already read weights through (see `pager.rs`, `alloc_arena_bda`), applied to the RESIDENT
// path. This is the ONLY weight path: every `BufferUsage::Weights` alloc routes through here (see
// `make_alloc`).

/// Byte alignment for resident-BDA weight sub-tensors within a block. Sub-tensors share one buffer
/// object rather than getting their own `VkMemoryRequirements`, so this is a fixed constant rather
/// than a probed one — 256 comfortably covers every access width a weight-reading shader uses (the
/// widest today is a 16-byte vec4 load) with headroom to spare for future wider kernels.
const BDA_WEIGHT_ALIGN: u64 = 256;

/// Minimum size for a resident-BDA arena block (see [`VulkanBackend::bda_weight_alloc`]). Blocks
/// are opened ON DEMAND — one per call that outgrows the current block's remainder — so without a
/// floor a run of small tensors would each pay for a separate dedicated `vkAllocateMemory`. Reuses
/// [`ARENA_OVERFLOW_BLOCK`]'s size: the same "big enough to amortize, small enough not to waste"
/// reasoning applies unchanged.
const BDA_BLOCK_MIN: u64 = ARENA_OVERFLOW_BLOCK;

/// Upper bound on ONE resident-BDA addressing unit's byte size (see [`BdaWeightArena`]'s addressing
/// invariant). The 64-bit promotion protects the arena/expert BASE, but a tensor's (or a per-expert
/// slice's) intra-unit byte offsets ride u32 push-constants / u32 in-kernel indices — a unit >= 4
/// GiB truncates into a coherent-but-wrong pointer. This is also `maxStorageBufferRange` on RADV,
/// the cap on the sub-range a `vkb` descriptor can bind. Enforced today by model reality; a single
/// >4 GiB tensor would need a wider addressing scheme, not just a bigger allocation.
const BDA_ADDRESSING_UNIT_MAX: u64 = 1 << 32;

/// Keepalive + addressing info for one [`BdaWeightArena`] block: a single dedicated
/// `bufferDeviceAddress` buffer (exactly what [`VulkanBackend::alloc_arena_bda`] builds) that
/// resident weight tensors bump-allocate BYTE RANGES within, never separate buffer objects (see
/// [`Backing::BdaSub`]). Held behind an `Arc` so a sub-tensor's `VkBuffer` can keep the block (and
/// therefore its memory and buffer handle) alive without owning it: the block is destroyed by
/// `buf`'s own `Drop` exactly once, when the last `Arc` clone — the arena's own plus every live
/// sub-tensor's — goes away. Never stored directly on `VulkanShared` (see
/// `VulkanBackend::bda_weight_arena`'s doc for why: `buf` holds an `Arc<VulkanShared>` clone, so
/// that would form a reference cycle and leak the whole device, exactly the bug
/// `backend_drop_frees_device_after_moe_pager` guards against for `moe_pager`/`dense_pager`).
struct BdaBlockHandle {
    /// The whole block's buffer: a `force_dedicated`, `device_address` allocation
    /// (`Backing::Pooled`), built by `make_buf_ex` exactly like `alloc_arena_bda`'s paged-MoE arena.
    buf: VkBuffer,
    /// `vkGetBufferDeviceAddress(buf.buffer)` — this block's byte-0 device address. A sub-tensor's
    /// `Buffer::device_addr` is `base_addr + sub_offset`.
    base_addr: u64,
}

/// One [`BdaWeightArena`] block: the shared keepalive handle plus its live bump cursor. Split apart
/// from `BdaBlockHandle` so the cursor can be mutated (behind `VulkanBackend::bda_weight_arena`'s
/// `Mutex`) while sub-tensors hold their own `Arc` clone of the handle without needing `&mut`
/// through it.
struct BdaArenaBlock {
    handle: Arc<BdaBlockHandle>,
    /// This block's total capacity in bytes (`<= max_mem_alloc_size`).
    size: u64,
    /// Next free byte offset (pre-alignment). Monotonic — sub-tensors are never freed individually.
    cursor: u64,
}

/// The resident-weight sub-allocator (the ONLY weight path — see `make_alloc`). Blocks are created
/// ON DEMAND as `bda_weight_alloc` calls outgrow the current block; this is pure allocation/upload
/// plumbing and doesn't thread a loader's total through this path, so an up-front-sized version can
/// be layered on the same block/bump primitives later. Each block is capped at `max_mem_alloc_size`
/// (a whole multi-GiB model can't be one `vkAllocateMemory`); a tensor never straddles a block
/// boundary — one that doesn't fit the current block's remainder opens a fresh one.
///
/// Sub-tensors CAN be bound as descriptors (unlike the paged-MoE/dense-streaming `alloc_arena_bda`
/// blocks, which are only ever read by device address): `recorder::Recorder::vkb` binds each
/// sub-tensor's own `(sub_offset, range)` rather than the whole block's `(0, WHOLE_SIZE)`, which is
/// what makes it safe for the small unforked weight consumers (norm gammas, biases, rope tables)
/// that have no `-DSTREAMED` twin to read this arena the same way they'd read any ordinary buffer.
///
/// Addressing invariant (audited): the 64-bit promotion protects the arena/expert BASE; intra-tensor
/// offsets remain u32 — each addressing unit (one dense tensor / one per-expert slice) must stay
/// < 4 Gi elements and < 4 GiB bytes; enforced today by model reality, revisit for >4 GiB single
/// tensors.
#[derive(Default)]
struct BdaWeightArena {
    blocks: Vec<BdaArenaBlock>,
}

// ── VulkanBackend ─────────────────────────────────────────────────────────────

/// Vulkan device + allocator + pipeline cache.
pub struct VulkanBackend {
    // NOTE: `moe_pager` is declared before `shared` so the session's buffers are freed first on
    // drop (each holds its own `Arc<VulkanShared>` clone, so the device outlives them either way).
    /// Paged MoE expert cache (see `pager::MoePagerSession`) — `Some` only when the loaded model's
    /// expert banks don't fit VRAM and the seam's placement policy chose paging over the legacy
    /// host-visible split (see `infr-llama`'s `generate_dense_vulkan_session`). `None` is the
    /// overwhelming common case (fits resident) and costs nothing beyond one `Mutex` lock check
    /// per `Backend::moe_paged` call.
    ///
    /// Owned by the BACKEND handle, NOT `VulkanShared`: the session's arena/LUT/ring buffers each
    /// hold an `Arc<VulkanShared>` clone, so parking the session on `VulkanShared` formed an Arc
    /// CYCLE — the shared state (device, allocator, weight arena, the pager arenas themselves:
    /// ~23 GiB after a Scout load) never dropped until process exit, and every LATER model load
    /// in the same process hit the VRAM budget guard with "N GiB already in use" (the
    /// `cpu_backend` gpu_ test-suite flake; see `backend_drop_frees_device_after_moe_pager`).
    /// The session still lives exactly as long as a loaded paged model can be generated with:
    /// `infr-llama`'s sessions own the `VulkanBackend`, and a new backend is a new device whose
    /// buffers couldn't read the old session anyway.
    moe_pager: crate::pager::MoePagerCell,
    /// Dense layer-streaming cache (see `pager::DensePagerSession`) — `Some` only when the loaded
    /// DENSE model's per-layer weights don't fit VRAM and the seam's placement chose streaming.
    /// Same drop-ordering/ownership story as `moe_pager` (declared before `shared` so its
    /// arena/ring buffers free first; owned by the backend HANDLE, never `VulkanShared` — the Arc
    /// cycle lesson on `moe_pager`'s doc applies unchanged).
    dense_pager: crate::pager::DensePagerCell,
    /// Resident-weight sub-allocator (see [`BdaWeightArena`]) — `None` until the first weight alloc;
    /// `make_alloc` routes every `BufferUsage::Weights` here (the sole weight path).
    ///
    /// Same drop-ordering/ownership story as `moe_pager`/`dense_pager` above and for the identical
    /// reason: each block's `BdaBlockHandle::buf` holds its own `Arc<VulkanShared>` clone, so
    /// parking this arena ON `VulkanShared` would form the same reference cycle that leaked the
    /// whole device for a paged-MoE session before `moe_pager` was moved off it — see
    /// `backend_drop_frees_device_after_moe_pager`.
    bda_weight_arena: Mutex<Option<BdaWeightArena>>,
    /// The engine configuration this backend reads its knobs from — one value, held for the
    /// backend's whole life, borrowed (never cloned) at every read site (`docs/config-plan.md`
    /// R4/R6).
    ///
    /// Handed in by [`VulkanBackend::new_with`] (S5a); the S2 `Config::load_from_env()` bridge that
    /// used to build it inside `new_selected` is GONE. `infr-cli` resolves the four layers once in
    /// `main()` and the `Arc` reaches here through the seam, so nothing on the production path
    /// crosses this boundary through the environment. The env-only [`VulkanBackend::new`] entry
    /// point remains for this crate's own GPU tests and for external library callers.
    cfg: Arc<Config>,
    shared: Arc<VulkanShared>,
}

/// Device memory info for a backend's shared state — the body of [`VulkanBackend::vram`],
/// factored out so scopes that only hold the `Arc<VulkanShared>` (e.g. [`WeightProgress`]'s
/// post-load log) can read it too.
///
/// WHICH HEAPS COUNT — the whole point of this function, and the difference between refusing a
/// model the box could run and TDR-ing one it could not:
///
/// DISCRETE card — device-local heaps ONLY. The other heap is host RAM reachable over PCIe (GTT).
/// It is NOT capacity: RADV happily accepts a device-local allocation past the VRAM heap's size and
/// quietly spills the excess into GTT — MEASURED on a 7900 XTX, where a 41 GiB run of 1 GiB
/// device-local allocations succeeded and landed as `mem_info_vram_used` 23.08 GiB +
/// `mem_info_gtt_used` 18.01 GiB. Every byte on the GTT side is then read across PCIe at a fraction
/// of VRAM bandwidth. Counting it would turn a clean load error into a mysteriously slow model, so
/// the guard budgets device-local alone. THIS IS THE PRE-EXISTING BEHAVIOR AND MUST NOT CHANGE.
///
/// UNIFIED-MEMORY part (an APU — see [`Capabilities::unified_memory`]) — ALL heaps. There is no
/// VRAM here to spill out of; both heaps are the same DDR at the same bandwidth, and the driver's
/// split between them is bookkeeping, not physics. MEASURED on RADV RAPHAEL_MENDOCINO: it
/// advertises a 21.47 GiB "DEVICE_LOCAL" heap and a 10.73 GiB host-visible one, which sum to
/// EXACTLY `mem_info_vram_total` (2 GiB carveout) + `mem_info_gtt_total` (30.20 GiB) — RADV
/// synthesizes the device-local heap as 2/3 of (carveout + GTT). The same 41 GiB device-local probe
/// on that device landed as `mem_info_gtt_used` 30.01 GiB and `mem_info_vram_used` 1.03 GiB: the
/// "device-local" heap IS system RAM through the GART, and the 2 GiB carveout is not where the
/// weights go. So the honest capacity is the SUM of the heaps, and budgeting against the
/// device-local slice alone refuses models (gemma-4-31B UD-Q5_K_XL: 20.37 GiB of weights against a
/// 21.22 GiB budget) that fit the machine with room to spare.
///
/// Counting the overflow heap is only half of it — `probe_uma_overflow_type` is what lets bytes
/// actually LAND there once the device-local heap is full. Above the summed budget the failure mode
/// is the same on both classes (the driver oversubscribes and starts evicting), which is why the
/// guard exists at all — it just now guards the right number on each.
fn vram_info(s: &VulkanShared) -> VramInfo {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default();
    if s.has_mem_budget {
        props2 = props2.push_next(&mut budget);
    }
    unsafe {
        s.instance
            .get_physical_device_memory_properties2(s.physical_device, &mut props2)
    };
    let mp = props2.memory_properties;
    let uma = s.caps.unified_memory;

    // Discrete: device-local heaps only. UMA: every heap (they are one pool of DDR). The live
    // VK_EXT_memory_budget figure is used on BOTH — it is what accounts for other processes, and
    // on a shared-memory part that matters more, not less: a second infr holding 21 GiB of the
    // same DDR is exactly the thing a UMA guard must see.
    let mut total = 0u64;
    let mut available = 0u64;
    for i in 0..mp.memory_heap_count as usize {
        let device_local = mp.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        if uma || device_local {
            total += mp.memory_heaps[i].size;
            available += if s.has_mem_budget {
                // Live free = budget - usage (the budget is a CEILING, not free bytes). Clamped to
                // the heap size so a driver that reports usage past the heap (RADV on an APU, once
                // something has oversubscribed the synthetic split) can't hand back a bogus figure.
                budget.heap_budget[i]
                    .saturating_sub(budget.heap_usage[i])
                    .min(mp.memory_heaps[i].size)
            } else {
                mp.memory_heaps[i].size
            };
        }
    }
    VramInfo {
        total,
        available,
        live: s.has_mem_budget,
        uma,
    }
}

/// RAII scope for a weight-load progress bar (see [`VulkanBackend::weight_progress`]). While alive,
/// `BufferUsage::Weights` allocations advance the bar; on drop it finishes and clears it.
pub struct WeightProgress {
    shared: Arc<VulkanShared>,
    /// `prof.vram` (`INFR_PROF_VRAM`), copied off the backend's `Config` when the scope opens —
    /// `Drop` holds only the shared state, so the flag rides along rather than being looked up
    /// (S5a; R4 — no global, no second config).
    vram_log: bool,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for WeightProgress {
    fn drop(&mut self) {
        // Drain the staging ring FIRST: its copies are still in flight (we fence per slot instead
        // of stalling the queue per tensor), and the weights must be fully resident before the
        // loader records a forward.
        self.shared.drain_staging_ring();
        if let Some(pb) = self.shared.weight_pb.lock().unwrap().take() {
            pb.finish_and_clear();
        }
        // Post-load memory-hygiene visibility (`prof.vram`): the LIVE in-use figure right
        // after the LAST weight upload — the number the VRAM-audit residual math (in-use minus
        // weights+KV estimate) starts from. The upload staging that ran under this scope was
        // dedicated-allocated (see `Backend::upload`), so by this drop it has fully returned
        // its device memory; what remains is weights + already-allocated session buffers.
        if self.vram_log {
            let v = vram_info(&self.shared);
            eprintln!(
                "post-load vram in use: {:.2} GiB of {:.2} GiB ({})",
                v.total.saturating_sub(v.available) as f64 / (1u64 << 30) as f64,
                v.total as f64 / (1u64 << 30) as f64,
                if v.live { "live" } else { "tracked" },
            );
        }
    }
}

impl infr_core::backend::ProgressScope for WeightProgress {}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// `maxComputeSharedMemorySize` for the active device — the per-workgroup shared-memory budget
    /// the flash-attention tile height is sized against (cheap accessor; avoids cloning caps).
    pub fn max_shared_memory_bytes(&self) -> u32 {
        self.shared.caps.max_shared_memory_bytes
    }

    /// Borrowed capabilities — the kernel-tier fallback ladder's gate (`caps.f16_coopmat`,
    /// `caps.f16`, `caps.i8_dot`). Cheap: a reference, not the [`Backend::capabilities`] clone (which
    /// copies the `name: String`) — safe to call per-op inside the adapter's hot lowering loop.
    pub(crate) fn caps(&self) -> &Capabilities {
        &self.shared.caps
    }

    /// Borrowed engine configuration — every knob this backend (and the seam code holding it)
    /// steers on. A REFERENCE, never a clone: the adapter reads it inside per-op lowering
    /// (`docs/config-plan.md` R6).
    ///
    /// `pub` because `infr-llama`'s seam reads the paging/KV knobs off the backend it already
    /// holds rather than growing a second env-sourced config of its own.
    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// Initialize Vulkan: create instance, pick a GPU (prefer discrete), create a logical
    /// device + compute queue with the required extensions/features, set up the allocator.
    /// `Err` on Apple (Vulkan is unsupported there — use the Metal backend), `Ok(())` everywhere
    /// else. Split into two `cfg` bodies so the guard is a runtime `Result` the caller `?`s: that
    /// keeps the Vulkan body in [`new`](Self::new) compiling on macOS while never executing it.
    #[cfg(target_os = "macos")]
    fn reject_on_apple() -> Result<()> {
        Err(be(
            "Vulkan is not supported on Apple. Use the native Metal backend: it is the default on \
             macOS, or select it explicitly with `--dev metal` (or INFR_DEV=metal). (The only \
             Vulkan on Apple is MoltenVK, which this backend deliberately does not target.)",
        ))
    }
    #[cfg(not(target_os = "macos"))]
    fn reject_on_apple() -> Result<()> {
        Ok(())
    }

    /// The historical default-device rule, EXACTLY preserved for the Vulkan case: honor
    /// `INFR_DEV=VulkanN` (the CLI's `--dev`, matching llama.cpp's naming) if set, else the first
    /// `DISCRETE_GPU`, else device 0. An out-of-range / unparseable Vulkan index is a hard error
    /// (silently running on a different GPU than asked produces plausible-but-wrong numbers).
    ///
    /// `INFR_DEV` is now the SINGLE device-selection env, so it can also hold `metal`/`cpu` (the
    /// non-Vulkan backends). The index resolver ([`resolve_infr_dev_index`]) TOLERATES those — a
    /// `metal`/`cpu` (or empty/unset) value falls back to the discrete default rather than erroring
    /// — since a process that reaches this Vulkan constructor built a Vulkan backend regardless.
    /// Split out of `new()` so `new_on` can bypass it, and so the behavior is a single named unit.
    ///
    /// `spec` is `device.dev` off the caller's [`Config`] (S5a) — the raw string, unparsed, exactly
    /// as `INFR_DEV` delivered it before. §6.12/§10.8: the CLI's `parse_dev_spec` and this
    /// crate's [`resolve_infr_dev_index`] are two DIFFERENT parsers of that one string and are
    /// deliberately NOT unified.
    fn pick_default_device(
        instance: &ash::Instance,
        pdevices: &[vk::PhysicalDevice],
        spec: Option<&str>,
    ) -> Result<vk::PhysicalDevice> {
        // A pinned Vulkan index needs the device names for the "no such device" message; build them
        // once (cold init path, a handful of devices).
        let names: Vec<String> = pdevices
            .iter()
            .enumerate()
            .map(|(i, &pd)| {
                let p = unsafe { instance.get_physical_device_properties(pd) };
                let n = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                format!("Vulkan{i}={n}")
            })
            .collect();
        match resolve_infr_dev_index(spec, &names)? {
            Some(idx) => Ok(pdevices[idx]), // range already checked by the resolver
            None => Ok(pdevices
                .iter()
                .copied()
                .find(|&pd| {
                    let p = unsafe { instance.get_physical_device_properties(pd) };
                    p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .unwrap_or(pdevices[0])),
        }
    }

    /// Enumerate ALL Vulkan physical devices WITHOUT building a backend (a cheap instance +
    /// `enumerate_physical_devices`, torn down before returning). Feeds the `infr devices` listing
    /// and the interconnect probe. Each entry's `index` is the `VulkanN` / `INFR_DEV` / `new_on`
    /// handle. Also reports the external-memory extensions each device exposes — the P2P /
    /// host-less-transfer feasibility signal the multi-GPU campaign needs.
    ///
    /// Takes the caller's [`Config`] because `is_default_pick` marks the device
    /// [`new_with`](Self::new_with) WOULD bind, which `device.dev` (`INFR_DEV`) steers (S5a).
    pub fn enumerate_devices(cfg: &Config) -> Result<Vec<DeviceInfo>> {
        Self::reject_on_apple()?;
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| be(format!("ash::Entry::load: {e}")))?;
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .map_err(|e| be(format!("create_instance: {e}")))?;

        let result = (|| -> Result<Vec<DeviceInfo>> {
            let pdevices = unsafe { instance.enumerate_physical_devices() }
                .map_err(|e| be(format!("enumerate_physical_devices: {e}")))?;
            let default_pick =
                Self::pick_default_device(&instance, &pdevices, cfg.device.dev.as_deref()).ok();
            let mut out = Vec::with_capacity(pdevices.len());
            for (index, &pd) in pdevices.iter().enumerate() {
                let p = unsafe { instance.get_physical_device_properties(pd) };
                let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                let mp = unsafe { instance.get_physical_device_memory_properties(pd) };
                let vram_bytes: u64 = (0..mp.memory_heap_count as usize)
                    .filter(|&h| {
                        mp.memory_heaps[h]
                            .flags
                            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                    })
                    .map(|h| mp.memory_heaps[h].size)
                    .sum();
                let exts = unsafe { instance.enumerate_device_extension_properties(pd) }
                    .unwrap_or_default();
                let has = |name: &CStr| {
                    exts.iter()
                        .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == name })
                };
                out.push(DeviceInfo {
                    index,
                    name,
                    device_type: device_type_str(p.device_type),
                    integrated: p.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU,
                    vram_bytes,
                    is_default_pick: default_pick == Some(pd),
                    external_memory: has(c"VK_KHR_external_memory"),
                    external_memory_fd: has(c"VK_KHR_external_memory_fd"),
                    external_memory_dma_buf: has(c"VK_EXT_external_memory_dma_buf"),
                });
            }
            Ok(out)
        })();
        unsafe { instance.destroy_instance(None) };
        result
    }

    /// **The real constructor (S5a).** Build a backend on the default device, reading every
    /// construction-time knob — device pick, capability masking, subgroup preference, the submit
    /// splitter, the VRAM guard, the pipeline-cache and pager diagnostics — from `cfg` instead of
    /// the process environment. The `Arc` is held for the backend's whole life and borrowed
    /// (never cloned) at each read site (`docs/config-plan.md` R4/R6).
    ///
    /// Device pick: `cfg.device.dev` (`INFR_DEV`) if it names a `VulkanN`, else the first discrete
    /// GPU, else device 0. See [`new_on_with`](Self::new_on_with) to pin a SPECIFIC index.
    pub fn new_with(cfg: Arc<Config>) -> Result<Self> {
        Self::new_selected(None, cfg)
    }

    /// [`new_with`](Self::new_with) pinned to physical-device `index`. See
    /// [`new_on`](Self::new_on) for the index semantics.
    pub fn new_on_with(index: usize, cfg: Arc<Config>) -> Result<Self> {
        Self::new_selected(Some(index), cfg)
    }

    /// Default-device constructor for callers that have no [`Config`] to hand in — this crate's own
    /// `#[cfg(test)]` GPU tests and external library users. Resolves `Default` < environment once
    /// (the same fold [`Config::load_from_env`] performs, but FALLIBLE, so `INFR_SG` /
    /// `INFR_SUBMIT_DISPATCHES` / the device lists still reject a bad value LOUDLY here exactly as
    /// the pre-S5a read sites did) and forwards to [`new_with`](Self::new_with). Every caller
    /// inside `infr-llama` and `infr-cli` passes its own `Arc<Config>` instead.
    pub fn new() -> Result<Self> {
        Self::new_with(Self::cfg_from_env()?)
    }

    /// Construct a backend pinned to physical-device `index` (enumeration order, matching
    /// [`enumerate_devices`](Self::enumerate_devices) and `INFR_DEV=VulkanN`), IGNORING the
    /// `device.dev` spec and the discrete-default rule. This is the multi-device foundation: two
    /// backends built with different indices can be held live simultaneously (each owns its own
    /// instance + logical device + allocator), enabling later tensor/expert-parallel slices. An
    /// out-of-range `index` is a hard error, never a silent fallback.
    ///
    /// The environment-sourced twin of [`new_on_with`](Self::new_on_with); see [`new`](Self::new).
    pub fn new_on(index: usize) -> Result<Self> {
        Self::new_on_with(index, Self::cfg_from_env()?)
    }

    /// `Default` < environment, for the [`new`](Self::new)/[`new_on`](Self::new_on)/
    /// [`enumerate_devices_from_env`](Self::enumerate_devices_from_env) entry points that are handed
    /// no `Config`.
    ///
    /// Deliberately NOT [`Config::load_from_env`]: that one is infallible (S2 needed it to be,
    /// because the loud keys were still read — and still rejected — at their own sites). S5a moved
    /// the last two of those five (`INFR_SG`, `INFR_SUBMIT_DISPATCHES`) onto `Config`, so swallowing
    /// a layer error here would SILENTLY drop an error a `VulkanBackend::new()` caller gets today
    /// (R1). No file layer either — these knobs were env-only, and a second diagnostics banner
    /// would print.
    fn cfg_from_env() -> Result<Arc<Config>> {
        let layer = infr_core::config::ConfigLayer::env().map_err(|e| be(e.to_string()))?;
        Ok(Arc::new(Config::load_from_layers(&[layer])))
    }

    /// [`enumerate_devices`](Self::enumerate_devices) for callers with no [`Config`] — see
    /// [`new`](Self::new).
    pub fn enumerate_devices_from_env() -> Result<Vec<DeviceInfo>> {
        let cfg = Self::cfg_from_env()?;
        Self::enumerate_devices(&cfg)
    }

    fn new_selected(explicit_index: Option<usize>, cfg: Arc<Config>) -> Result<Self> {
        // Apple: the Vulkan backend is DELIBERATELY unsupported (the only Vulkan on Apple is
        // MoltenVK, which lacks features this backend depends on — e.g. `bufferDeviceAddress` for
        // the paged MoE arena — and is slower than talking to Metal directly). infr ships a NATIVE
        // Metal backend for Apple GPUs. The `?` on a runtime `Result` here does NOT trip
        // `unreachable_code`, so the Vulkan body below still compiles clean on macOS (it is simply
        // never reached). See `reject_on_apple`.
        Self::reject_on_apple()?;

        // Every knob below is resolved ONCE, here, from the caller's config — the construction-time
        // tier (`docs/config-plan.md` §7 S5a). Borrowed, never cloned (R6); `cfg` itself moves onto
        // the backend at the bottom of this function.
        let vkcfg = &cfg.kernels.vulkan;

        // ── entry ──────────────────────────────────────────────────────────────
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| be(format!("ash::Entry::load: {e}")))?;

        // ── instance (Vulkan 1.3) ──────────────────────────────────────────────
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"infr")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"infr-vulkan")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        // Native Vulkan drivers only (Linux/Windows AMD/NVIDIA/Intel). The MoltenVK portability
        // opt-in that used to live here was removed with the Apple guard above — Apple is the only
        // place a portability driver is enumerated, and infr no longer runs Vulkan there.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .map_err(|e| be(format!("create_instance: {e}")))?;

        // RAII cleanup for the partially-built device. `ash::Instance`/`Device` have NO `Drop`
        // (only `VulkanShared::Drop` frees them), so every `Err`/`?` return below this point would
        // otherwise leak the `VkInstance` — and, once they exist, the `VkDevice` + command pool —
        // for the whole process life. That is a RECOVERABLE path (the seam catches the `Err` and
        // falls back to CPU), so the leak is permanent per launch: the subgroup-32 rejection, the
        // `INFR_SG`/`INFR_SUBMIT_DISPATCHES` env guards, `!has_bda`, a failed `create_command_pool`
        // or allocator build all return here. Mirror `enumerate_devices`, which destroys its
        // instance unconditionally on the way out. Holds independent handle CLONES (ash handles are
        // trivially copyable and their `Drop` is a no-op), so the success path — which DISARMS this
        // just before the originals move into `VulkanShared` — is byte-for-byte unchanged and never
        // double-frees.
        struct InstanceCleanup {
            instance: ash::Instance,
            device: Option<ash::Device>,
            pool: vk::CommandPool,
            armed: bool,
        }
        impl Drop for InstanceCleanup {
            fn drop(&mut self) {
                if !self.armed {
                    return;
                }
                unsafe {
                    if let Some(device) = &self.device {
                        if self.pool != vk::CommandPool::null() {
                            device.destroy_command_pool(self.pool, None);
                        }
                        device.destroy_device(None);
                    }
                    self.instance.destroy_instance(None);
                }
            }
        }
        let mut cleanup = InstanceCleanup {
            instance: instance.clone(),
            device: None,
            pool: vk::CommandPool::null(),
            armed: true,
        };

        // ── physical device: `INFR_DEV` if set, else prefer discrete ──────────
        // `INFR_DEV=VulkanN` (set by the CLI's `--dev`) pins the Nth device in ENUMERATION order,
        // matching llama.cpp's `--dev VulkanN` naming so the two tools address the same GPU on a
        // multi-GPU box. Unset => the historical rule: first DISCRETE_GPU, else device 0.
        //
        // An out-of-range / unparseable INFR_DEV is a hard error, NOT a fallback: silently running
        // on a different GPU than the one asked for produces numbers that look plausible and are
        // wrong, which is far worse than refusing to start.
        let pdevices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| be(format!("enumerate_physical_devices: {e}")))?;
        if pdevices.is_empty() {
            return Err(be("no Vulkan physical devices"));
        }

        // Make enumeration VISIBLE (the campaign asked for it): one line per physical device at
        // init — index (the `VulkanN` / `INFR_DEV` / `new_on` handle), name, class, device-local
        // heap. Silent-picking one GPU on a multi-GPU box is exactly what hid device selection.
        for (i, &pd) in pdevices.iter().enumerate() {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }.to_string_lossy();
            let mp = unsafe { instance.get_physical_device_memory_properties(pd) };
            let dev_local: u64 = (0..mp.memory_heap_count as usize)
                .filter(|&h| {
                    mp.memory_heaps[h]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|h| mp.memory_heaps[h].size)
                .sum();
            eprintln!(
                "[infr] vulkan device Vulkan{i}: {name} ({}, {})",
                device_type_str(p.device_type),
                fmt_bytes(dev_local),
            );
        }

        // Explicit index (`new_on`) wins outright and bypasses the env/discrete rule — the
        // multi-device path. `None` = the historical default, byte-for-byte unchanged below.
        let physical_device = match explicit_index {
            Some(idx) => *pdevices.get(idx).ok_or_else(|| {
                be(format!(
                    "VulkanBackend::new_on({idx}): no such Vulkan device (this system has {})",
                    pdevices.len()
                ))
            })?,
            None => Self::pick_default_device(&instance, &pdevices, cfg.device.dev.as_deref())?,
        };

        // Selection log: which of the enumerated devices this backend actually bound.
        {
            let p = unsafe { instance.get_physical_device_properties(physical_device) };
            let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }.to_string_lossy();
            let idx = pdevices.iter().position(|&pd| pd == physical_device);
            eprintln!(
                "[infr] vulkan: selected {}{name} ({})",
                idx.map(|i| format!("Vulkan{i}=")).unwrap_or_default(),
                device_type_str(p.device_type),
            );
        }

        // ── compute queue family ───────────────────────────────────────────────
        // The first COMPUTE-capable family. On amdgpu this is the universal (graphics) family, so
        // the work lands on the `gfx` ring. Moving it to a compute-ONLY family (the `comp` rings)
        // was tried and is strictly WORSE on the surveyed integrated part: the same ~2 s forward
        // that is an intermittent device-lost on `gfx` is a DETERMINISTIC one on `comp` (measured
        // 4/4 runs), i.e. the compute ring's effective hang budget there is TIGHTER, not the 60 s
        // its module-parameter default advertises. The submit splitter is what actually bounds the
        // job (see `VulkanShared::submit_dispatch_cap`); the ring choice is not a lever.
        let qf_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = qf_props
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|i| i as u32)
            .ok_or_else(|| be("no compute queue family found"))?;

        // ── probe device extensions ────────────────────────────────────────────
        let avail_exts = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|e| be(format!("enumerate device extensions: {e}")))?;

        let has_ext = |name: &CStr| -> bool {
            avail_exts
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == name })
        };

        let has_coop_matrix = has_ext(c"VK_KHR_cooperative_matrix");
        let has_16bit_storage = has_ext(c"VK_KHR_16bit_storage");
        let has_8bit_storage = has_ext(c"VK_KHR_8bit_storage");
        // Packed i8 (int8) dot (dp4a) — the decode i8 `mmv` accumulate. Promoted to core in Vulkan
        // 1.3; probed via the KHR ext for pre-1.3 drivers. Detection-only here (caps.i8_dot); the
        // adapter's i8-mmv gate consults it so a device without packed dot routes to the scalar
        // dequant GEMV instead of dispatching a dp4a kernel it can't run.
        let has_i8_dot_ext = has_ext(c"VK_KHR_shader_integer_dot_product");
        // f8 (== fp8, E4M3/E5M2) storage/convert support. ash 0.38 has no constant for the ext, so
        // match the raw name. Absent on RDNA3 → caps.f8 false.
        let has_f8_ext = has_ext(c"VK_EXT_shader_float8");
        // bf16 (bfloat16) storage/convert. ash 0.38 has no constant for the ext → match the raw
        // name. Absent on RDNA3 → caps.bf16 false; present on RDNA4/Navi44.
        let has_bf16_ext = has_ext(c"VK_KHR_shader_bfloat16");
        let has_subgroup_ext = has_ext(c"VK_KHR_shader_subgroup_extended_types");
        let has_mem_budget = has_ext(c"VK_EXT_memory_budget");
        // External-memory (host-LESS cross-device P2P): export a buffer's memory as an fd on one
        // device and import it on another so device B reads/writes device A's physical bytes over
        // PCIe with no host bounce (see `p2p.rs`). `VK_KHR_external_memory` is core in Vulkan 1.1
        // (this backend targets 1.3), so the `VkExternalMemoryBufferCreateInfo` /
        // `VkExportMemoryAllocateInfo` / `VkImportMemoryFdInfoKHR` structs need no extension enable —
        // only the fd op extension (`vkGetMemoryFdKHR`/`vkGetMemoryFdPropertiesKHR`) and, for the
        // dma-buf handle type (the cross-GPU-portable one on Linux), `VK_EXT_external_memory_dma_buf`.
        // Both are GATED: a device lacking them simply reports no P2P support (`caps` below), the
        // default single-device path is untouched, and no P2P handle type is offered.
        let has_ext_mem_fd = has_ext(c"VK_KHR_external_memory_fd");
        let has_ext_mem_dma_buf = has_ext(c"VK_EXT_external_memory_dma_buf");
        // External SEMAPHORE fd — the tensor-parallel all-reduce orders a peer's cross-device read
        // after this device's GPU-side signal with NO host round-trip: export a timeline semaphore as
        // an fd here, import it on the peer, signal a value on this device's submit and wait it on the
        // peer's (see `tp_sem.rs`). `VK_KHR_external_semaphore` is core in Vulkan 1.1, so only the fd
        // op extension needs enabling; the timeline-semaphore feature (core 1.2) is probed + enabled
        // below. GATED: a device lacking either reports no support and the all-reduce falls back to
        // the host fence. `VK_EXT_external_semaphore_fd` uses OPAQUE_FD (same-driver cross-device,
        // which the dGPU+iGPU pair here both being RADV satisfies).
        let has_ext_sem_fd = has_ext(c"VK_KHR_external_semaphore_fd");
        // Lets every dispatch bind its buffers with one `cmd_push_descriptor_set` recorded
        // straight into the command buffer instead of `alloc_set` (pool allocate) +
        // `update_descriptor_sets` (a separate driver call) + `cmd_bind_descriptor_sets` per op —
        // measured as a real per-forward host-side cost at small-m shapes (many-op graphs where
        // GPU busy time is small, so the fixed per-dispatch descriptor churn is a bigger fraction
        // of wall time). Near-universally supported (desktop RADV/NVIDIA/Intel); the pooled path
        // stays as a fallback for drivers that lack it (e.g. some portability/MoltenVK builds).
        // `kernels.vulkan.push_desc = false` (`INFR_NO_PUSH_DESC=1`) forces the pooled-classic
        // fallback even when the extension exists — lets a RADV dev box exercise the code path a
        // driver WITHOUT push descriptors takes (field report: teardown validation findings on
        // Intel Arc/ANV that RADV runs never reproduce because the classic pools are never created
        // here). Test/diagnosis knob only.
        let has_push_descriptor = has_ext(c"VK_KHR_push_descriptor") && vkcfg.push_desc;

        // ── probe features (via VK 1.1 get_physical_device_features2) ─────────
        // Memory model and subgroup-size-control are probed rather than assumed: a portability
        // device (MoltenVK) may lack either, and enabling an unsupported feature fails
        // create_device outright.
        let mut f16_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut memmodel_feat = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        let mut sgsize_feat = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut coopmat_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut intdot_feat = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        // Buffer-device-address: lets a shader read a buffer via a 64-bit `VkDeviceAddress`
        // (`GL_EXT_buffer_reference`), bypassing one SSBO binding's `maxStorageBufferRange` (~4 GiB
        // on RADV). infr's paged-MoE expert arena REQUIRES it — a per-role pool now spans as much
        // VRAM as the budget allows, addressed by a raw pointer. Core in Vulkan 1.2, so it is
        // hard-required below (not an opt-in ladder like coopmat).
        let mut bda_feat = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        // Timeline semaphore (core 1.2) — required by the tensor-parallel external-semaphore
        // all-reduce (a shared timeline signalled on one device, waited on another). Probed so we
        // never try to enable it where absent (which would fail create_device).
        let mut timeline_feat = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut feat2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f16_feat)
            .push_next(&mut memmodel_feat)
            .push_next(&mut sgsize_feat)
            .push_next(&mut coopmat_feat)
            .push_next(&mut intdot_feat)
            .push_next(&mut bda_feat)
            .push_next(&mut timeline_feat);
        unsafe { instance.get_physical_device_features2(physical_device, &mut feat2) };
        // Core Vulkan 1.0 feature (no extension struct — `get_physical_device_features2` always
        // populates the chain's base `.features`). Several KV-cache dequant/attention shaders
        // (dequant_turbo_f16.comp, dequant_q8_f16.comp, attn_*.comp) declare SPIR-V's `Int16`
        // capability (16-bit integer arithmetic, e.g. GL_EXT_shader_explicit_arithmetic_types_int16
        // int16_t/uint16_t locals — distinct from `storageBuffer16BitAccess`, which only covers
        // 16-bit SSBO/UBO *storage*, not arithmetic): VUID-VkShaderModuleCreateInfo-pCode-08740
        // requires `shaderInt16` enabled on the DEVICE for that capability, same class of bug as the
        // `shaderIntegerDotProduct` one fixed below — detected via caps but never chained into
        // `device_ci`, so vkCreateShaderModule for those kernels violated the VUID under validation.
        let has_int16 = feat2.features.shader_int16 != 0;
        // Same 08740 class as `shaderInt16` above, for 64-bit integer arithmetic: the BDA arena
        // helper `native_weight_addr.glsl` (paged-MoE `-DPAGED` builds AND dense-streaming
        // `-DSTREAMED` builds) composes a 64-bit slot address from lo/hi u32 halves
        // (`uint64_t(hi) << 32 | uint64_t(lo)`, `GL_EXT_shader_explicit_arithmetic_types_int64`),
        // which emits SPIR-V's `Int64` capability — so `shaderInt64` MUST be enabled on the device
        // or vkCreateShaderModule for those kernels violates the VUID under validation. Core 1.0
        // feature; RADV/desktop support it universally, probed here for portability devices.
        let has_int64 = feat2.features.shader_int64 != 0;
        // Read AFTER the `feat2.features` access above: `feat2` holds a mutable borrow of every
        // pushed feature struct (incl. `bda_feat`) until its last use, so the pushed structs can
        // only be read once `feat2` itself is done being touched.
        let has_bda = bda_feat.buffer_device_address != 0;
        // The external-semaphore all-reduce needs BOTH the fd extension and the timeline feature
        // (read here, after `feat2`'s last use, for the same borrow reason as `has_bda`).
        let has_ext_sem = has_ext_sem_fd && timeline_feat.timeline_semaphore != 0;
        // Hard requirement, not a fallback: the paged-MoE arena is addressed by a 64-bit device
        // pointer, so a device that cannot hand out one has no 64-bit address space for infr to
        // use. bufferDeviceAddress is core in Vulkan 1.2 and this backend targets 1.3, so on any
        // real target this never fires — it is the clean guard for a driver/portability layer that
        // somehow omits it, failing at init with a clear message instead of miscompiling a shader.
        if !has_bda {
            let p = unsafe { instance.get_physical_device_properties(physical_device) };
            let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let (major, minor, patch) = (
                vk::api_version_major(p.api_version),
                vk::api_version_minor(p.api_version),
                vk::api_version_patch(p.api_version),
            );
            return Err(be(format!(
                "this GPU/driver does not support bufferDeviceAddress (the 64-bit shader address \
                 space), which infr's paged-MoE arena requires — {name} / Vulkan \
                 {major}.{minor}.{patch}. bufferDeviceAddress is core in Vulkan 1.2; update the \
                 driver or use a device that exposes it."
            )));
        }
        let has_f16 = f16_feat.shader_float16 != 0;
        let has_memmodel = memmodel_feat.vulkan_memory_model != 0;
        let has_memmodel_dev = memmodel_feat.vulkan_memory_model_device_scope != 0;
        let has_sgsize = sgsize_feat.subgroup_size_control != 0;
        let has_full_sg = sgsize_feat.compute_full_subgroups != 0;
        // Packed i8 dot: ext advertised AND the feature bit set (same ext-AND-feature discipline as
        // coopmat). Detection-only — the current i8 mmv is DEFAULT-OFF at m=1 (scalar wins), and no
        // shader here uses the ext builtin yet, so we don't add it to the enabled feature chain; the
        // adapter's i8-mmv gate reads `caps.i8_dot` before ever dispatching a dp4a kernel.
        let has_i8_dot = has_i8_dot_ext && intdot_feat.shader_integer_dot_product != 0;
        // i8 (int8) shader storage/math — the same `shaderFloat16Int8` feature struct carries it.
        let has_int8 = f16_feat.shader_int8 != 0;
        // Extension presence alone doesn't guarantee the FEATURE bit (a driver may advertise
        // VK_KHR_cooperative_matrix with cooperativeMatrix=false — enabling it then fails
        // create_device, the same failure class #32 fixed for memmodel/sgsize). This is the
        // PREREQUISITE (unit exists + usable); which COMPONENT TYPES it accepts is a separate
        // enumeration below — the ext bit does NOT imply f16 support (the spec only promises a unit
        // exists, not that it does f16), so we don't assume it.
        let has_coop_ext_feat = has_coop_matrix && coopmat_feat.cooperative_matrix != 0;

        // Enumerate the device's cooperative-matrix configs ONCE — the AUTHORITATIVE source for
        // which component types AND tile dimensions the matrix unit accepts. Each config lists
        // m/n/k size + a/b/c/result types; the ext's presence alone tells us nothing about them.
        // Empty when the ext/feature is absent. Extract a Copy tuple — the returned structs borrow
        // the loader `cm`, so we can't let them outlive this block.
        type CoopmatConfig = (
            u32,
            u32,
            u32,
            vk::ComponentTypeKHR,
            vk::ComponentTypeKHR,
            vk::ComponentTypeKHR,
            vk::ComponentTypeKHR,
        );
        let coopmat_configs: Vec<CoopmatConfig> = if has_coop_ext_feat {
            let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
            unsafe { cm.get_physical_device_cooperative_matrix_properties(physical_device) }
                .map(|v| {
                    v.iter()
                        .map(|p| {
                            (
                                p.m_size,
                                p.n_size,
                                p.k_size,
                                p.a_type,
                                p.b_type,
                                p.c_type,
                                p.result_type,
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        // Diagnostic: dump every enumerated (M,N,K,aType,bType,cType,resultType) — the definitive
        // list of what the matrix unit accepts, for bringing up new HW (RDNA4 fp8, bf16, Intel's
        // 8/8/16 tiles) and sanity-checking the per-type/per-dim detection below.
        // `debug.coopmat` (`INFR_DEBUG_COOPMAT=1`).
        if cfg.debug.coopmat {
            let raws: Vec<(u32, u32, u32, i32, i32, i32, i32)> = coopmat_configs
                .iter()
                .map(|&(m, n, k, a, b, c, r)| {
                    (m, n, k, a.as_raw(), b.as_raw(), c.as_raw(), r.as_raw())
                })
                .collect();
            eprintln!(
                "[infr] coopmat configs (M,N,K,aType_raw,bType_raw,cType_raw,resultType_raw): \
                 {raws:?}"
            );
        }
        // f16 coopmat: configs with f16 A AND B operands (accumulator/result f16 or f32), reduced
        // to ONE chosen (M,N,K) tile by `select_coopmat_shape`'s preference order: 16x16x16 first
        // (the shape every production coopmat shader is built for — every `coopmat<...,16,16,...>`
        // declaration across gemm_coopmat*/gemm_warp/native_gemm*/attn_*/deltanet_prep), then
        // 8x8x16 (Intel Arc/ANV XMX — ONLY under the `INFR_CM_8X8=1` opt-in, and only the
        // `native_gemm_warp` `_cm8` builds exist at that shape). Component types alone are NOT
        // sufficient — an Intel A770 (Mesa ANV) advertises f16×f16→f32 only at M=8,N=8,K=16;
        // creating our 16x16x16 pipeline on such a device silently fails
        // vkCreateComputePipelines (the segfault bug — the result wasn't checked, see
        // `create_compute_pipeline` below). Requiring the shape match here makes an unsupported
        // device fall back to the non-coopmat ladder instead of crashing. Derived from the
        // enumeration, not assumed from the ext bit. `has_coop_matrix` (any usable f16 shape)
        // keeps its downstream role (ext-enable, feature chain).
        let f16c = vk::ComponentTypeKHR::FLOAT16;
        let cm8_env = vkcfg.coopmat_8x8;
        let coopmat_f16 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, _)| a == f16c && b == f16c)
                .map(|&(m, n, k, ..)| (m, n, k)),
            cm8_env,
        );
        // Extension-added ComponentTypeKHR raw values (ash 0.38 predates these variants, so match by
        // raw i32). CONFIRMED on RDNA4/Navi44 via INFR_DEBUG_COOPMAT — all CORE types are 0..=10
        // (FLOAT16=0/FLOAT32=1/SINT8=3/UINT8=7/…), these are the KHR-standard ext values:
        const CT_E4M3: i32 = 1_000_491_002; // VK_COMPONENT_TYPE_FLOAT_E4M3_KHR
        const CT_E5M2: i32 = 1_000_491_003; // VK_COMPONENT_TYPE_FLOAT_E5M2_KHR
        const CT_BF16: i32 = 1_000_141_000; // VK_COMPONENT_TYPE_BFLOAT16_KHR
        let is_f8 = |t: i32| t == CT_E4M3 || t == CT_E5M2;
        // f8 coopmat: configs with fp8 (E4M3/E5M2) A AND B operands, 16x16x16 ONLY (no f8 shader
        // exists at any other shape → `allow_8x8x16 = false`). Uses the KHR-standard fp8 raw
        // values (confirmed on RDNA4) rather than the older `>= 1e9` heuristic, which also
        // matched bf16 (`CT_BF16` is ext-range too) and would false-positive f8 on a bf16-only
        // unit. Also requires the float8 storage ext. NEVER Some on RDNA3 (enumerates no fp8
        // config).
        let coopmat_f8 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, _)| is_f8(a.as_raw()) && is_f8(b.as_raw()))
                .map(|&(m, n, k, ..)| (m, n, k)),
            false,
        )
        .filter(|_| has_f8_ext);
        // bf16 coopmat: BFLOAT16 A AND B operands, 16x16x16 only. Confirmed on RDNA4/Navi44
        // (bf16×bf16→bf16 and →f32); RDNA3 enumerates none. Same discipline as f8/f16 above.
        let coopmat_bf16 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, _)| a.as_raw() == CT_BF16 && b.as_raw() == CT_BF16)
                .map(|&(m, n, k, ..)| (m, n, k)),
            false,
        );
        // i8 coopmat: configs with SINT8 A AND B operands and a SINT32 result, 16x16x16 only (the
        // shape every int8 coopmat shader here uses) — same discipline as `coopmat_f16`'s shape
        // selection above. DETECTION ONLY (see the `coopmat_i8` doc comment on `Capabilities`):
        // the standalone `coopmat_int8_test` harness confirmed this exact config
        // (SINT8xSINT8->SINT32, subgroup-pinned 32, A RowMajor/B ColumnMajor) dispatches correctly
        // on this driver, but int8 coopmat hung an OLDER Mesa (commit ad82a77) despite enumerating
        // fine there too — so detection alone does NOT make this capability a safe default; the
        // adapter requires `INFR_I8_COOPMAT=1` in addition to `caps.i8_coopmat()` before ever
        // dispatching the kernel.
        let i8c = vk::ComponentTypeKHR::SINT8;
        let i32c = vk::ComponentTypeKHR::SINT32;
        let coopmat_i8 = select_coopmat_shape(
            coopmat_configs
                .iter()
                .filter(|&&(_, _, _, a, b, _, r)| a == i8c && b == i8c && r == i32c)
                .map(|&(m, n, k, ..)| (m, n, k)),
            false,
        );

        // ── force-disable capabilities for fallback-path testing on capable HW ──
        // These knobs drop a DETECTED capability so the next kernel tier down is exercised on a
        // device that actually has the feature — otherwise the portability fallbacks are only
        // reachable on hardware we may not own. This is `docs/config-plan.md` §5.2: `Capabilities`
        // stays a PROBE result (no config fields on it, nothing downstream re-reads a knob), and
        // the config MASKS the probe right here, at construction. Applied before the ext list /
        // feature chain so a forced-off feature is genuinely NOT enabled on the device (a faithful
        // simulation, not just a caps flag flip). f16 is a coopmat prerequisite, so `!f16` ⇒ NO
        // coopmat too — preserve that AND, it is not a bug.
        let has_f16 = has_f16 && vkcfg.f16;
        let coopmat_f16 = coopmat_f16
            .filter(|_| has_f16 && vkcfg.coopmat)
            .filter(|&s| {
                // 8x8x16 additionally needs a pinnable subgroup size 16: the `_cm8` builds run
                // 128 threads = 8 warps × 16 lanes (XMX/DPAS is SIMD16-native) and reuse the
                // kernel's 8-warp index math, so a device that can't pin 16 can't run them.
                // (Checked here so the coopmat ext is never enabled for a shape we then refuse.
                // The physical-device subgroup range is queried again for `caps` below; this
                // early copy exists because that query currently runs after device creation.)
                if s != infr_core::COOPMAT_TILE_8 {
                    return true;
                }
                let mut sgp = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sgp);
                unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
                has_sgsize && sgp.min_subgroup_size <= 16 && 16 <= sgp.max_subgroup_size
            });
        // `has_coop_matrix` = ANY usable f16 coopmat shape — drives the ext enable + feature
        // chain below. On a 16x16x16 device this is exactly the old boolean; on an 8x8x16-only
        // device it is false unless INFR_CM_8X8=1 selected the 8x8x16 shape above (default OFF:
        // the ext is then NOT enabled, byte-identical to the pre-shape-table behavior there).
        let has_coop_matrix = coopmat_f16.is_some();
        // f8 coopmat is a coopmat sub-tier, so dropping coopmat drops it too.
        let coopmat_f8 = coopmat_f8.filter(|_| has_coop_matrix);
        // bf16 coopmat: same coopmat sub-tier dependency (rides the coopmat device-feature enable).
        let coopmat_bf16 = coopmat_bf16.filter(|_| has_coop_matrix);
        // i8 coopmat rides the SAME device feature enable (coopmat_ci is only chained into
        // device_ci below when `has_coop_matrix`) — without it the extension isn't enabled on the
        // logical device even if int8 configs were enumerated, so this is a real dependency, not
        // just symmetry with coopmat_f8 above. `!coopmat`/`!f16` drop it too.
        let coopmat_i8 = coopmat_i8.filter(|_| has_coop_matrix);
        let has_i8_dot = has_i8_dot && vkcfg.i8_dot;
        // INFR_CM_8X8=1 outcome notice (once, at device init): the tester A/B knob must be loud
        // about whether it actually engaged — on RADV (16x16x16 enumerated) or any device without
        // an 8x8x16 f16 config it changes NOTHING, and the kernel set stays identical.
        if cm8_env {
            match coopmat_f16 {
                Some(infr_core::COOPMAT_TILE_8) => eprintln!(
                    "[infr] INFR_CM_8X8=1: 8x8x16 f16 coopmat selected — native_gemm_warp _cm8 \
                     prefill tier live (other coopmat families stay on their non-coopmat \
                     fallbacks)"
                ),
                Some(_) => eprintln!(
                    "[infr] INFR_CM_8X8=1 has no effect: device provides the default 16x16x16 \
                     f16 coopmat tile — kernel set unchanged"
                ),
                None => eprintln!(
                    "[infr] INFR_CM_8X8=1 has no effect: device enumerates no usable 8x8x16 f16 \
                     coopmat config (or coopmat is disabled) — kernel set unchanged"
                ),
            }
        }
        // Extend the `debug.coopmat` dump with the CHOSEN shape per component type (the raw
        // enumeration is printed above, before selection).
        if cfg.debug.coopmat {
            eprintln!(
                "[infr] coopmat chosen shapes (M,N,K): f16={coopmat_f16:?} bf16={coopmat_bf16:?} \
                 f8={coopmat_f8:?} i8={coopmat_i8:?}"
            );
        }

        // ── build extension name list (only available ones) ────────────────────
        let mut ext_ptrs: Vec<*const i8> = Vec::new();
        if has_coop_matrix {
            ext_ptrs.push(c"VK_KHR_cooperative_matrix".as_ptr());
        }
        if has_16bit_storage {
            ext_ptrs.push(c"VK_KHR_16bit_storage".as_ptr());
        }
        if has_8bit_storage {
            ext_ptrs.push(c"VK_KHR_8bit_storage".as_ptr());
        }
        if has_subgroup_ext {
            ext_ptrs.push(c"VK_KHR_shader_subgroup_extended_types".as_ptr());
        }
        if has_mem_budget {
            ext_ptrs.push(c"VK_EXT_memory_budget".as_ptr());
        }
        if has_push_descriptor {
            ext_ptrs.push(c"VK_KHR_push_descriptor".as_ptr());
        }
        // The int8 dp4a decode GEMVs (native_mmv.comp, native_mmv_mrow.comp, native_mmv_id_q4k.comp,
        // mul_mat_vec_q.comp's dotPacked builtins) compile to SPIR-V with the DotProduct /
        // DotProductInput4x8BitPacked capabilities, which VUID-VkShaderModuleCreateInfo-pCode-08740
        // requires `shaderIntegerDotProduct` to be enabled on the DEVICE for — not just detected.
        // This was previously probed into `caps.i8_dot` (detection-only, per an now-stale comment
        // claiming no shader used the builtin yet) but never actually enabled, so vkCreateShaderModule
        // for those kernels violated the VUID on any driver that validates it (reproduced on the
        // 7900 XTX under validation layers with an 8B model, which is wide enough to select the mmv
        // dp4a tier — the small model's shapes never hit it, hence the bug staying latent).
        if has_i8_dot {
            ext_ptrs.push(c"VK_KHR_shader_integer_dot_product".as_ptr());
        }
        // External-memory fd ops + dma-buf handle type for the cross-device P2P transport (gated —
        // see the probe above). `VK_KHR_external_memory` itself is core in 1.1 and needs no enable.
        if has_ext_mem_fd {
            ext_ptrs.push(c"VK_KHR_external_memory_fd".as_ptr());
        }
        if has_ext_mem_dma_buf {
            ext_ptrs.push(c"VK_EXT_external_memory_dma_buf".as_ptr());
        }
        // External-semaphore fd ops for the tensor-parallel all-reduce's GPU-side cross-device sync
        // (gated). `VK_KHR_external_semaphore` is core in 1.1 (no enable); the timeline-semaphore
        // feature is enabled via `timeline_sem_ci` chained into `device_ci` below.
        if has_ext_sem {
            ext_ptrs.push(c"VK_KHR_external_semaphore_fd".as_ptr());
        }
        // A portability (layered) device REQUIRES VK_KHR_portability_subset to be enabled when
        // it advertises it (Vulkan valid-usage rule); MoltenVK does.
        if has_ext(c"VK_KHR_portability_subset") {
            ext_ptrs.push(c"VK_KHR_portability_subset".as_ptr());
        }

        // ── logical device ─────────────────────────────────────────────────────
        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        // Feature chain — needed for cooperative-matrix kernels:
        //   shaderFloat16 (f16 math), 16-bit storage (f16 SSBOs), Vulkan memory model
        //   (required by coopmat), cooperativeMatrix itself.
        let mut shader_f16_ci = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(has_f16)
            .shader_int8(true);
        let mut storage16_ci = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(has_16bit_storage);
        let mut storage8_ci = vk::PhysicalDevice8BitStorageFeatures::default()
            .storage_buffer8_bit_access(has_8bit_storage);
        let mut memmodel_ci = vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
            .vulkan_memory_model(has_memmodel)
            .vulkan_memory_model_device_scope(has_memmodel_dev);
        // Chained below only when `has_coop_matrix` (ext AND probed feature) — see the probe.
        let mut coopmat_ci =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        // Lets us pin the subgroup size to 32 (RDNA3 coopmat is wave32) for the tiled GEMM.
        let mut sgsize_ci = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(has_sgsize)
            .compute_full_subgroups(has_full_sg);
        // Chained below only when `has_i8_dot` — see the ext_ptrs comment above.
        let mut intdot_ci = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(true);
        // Buffer-device-address — hard-required above, so always enabled (the paged-MoE arena is
        // addressed by a `VkDeviceAddress`). Core in 1.2, promoted from VK_KHR_buffer_device_address,
        // so it needs no device extension on a 1.3 device — only the feature enable.
        let mut bda_ci =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        // Timeline semaphore — enabled only when the external-semaphore all-reduce path is available
        // (chained below when `has_ext_sem`).
        let mut timeline_sem_ci =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);

        // Core 1.0 features (shaderInt16 — see the probe comment above): passed via
        // `enabled_features`, NOT a pNext-chained `PhysicalDeviceFeatures2` (the two are mutually
        // exclusive per the spec; this device_ci never chains `PhysicalDeviceFeatures2` itself, only
        // extension-specific feature structs, so `enabled_features` is the correct, conflict-free
        // slot for it).
        let core_features = vk::PhysicalDeviceFeatures::default()
            .shader_int16(has_int16)
            .shader_int64(has_int64);
        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&ext_ptrs)
            .enabled_features(&core_features)
            .push_next(&mut shader_f16_ci)
            .push_next(&mut storage16_ci)
            .push_next(&mut storage8_ci)
            .push_next(&mut memmodel_ci)
            .push_next(&mut sgsize_ci)
            .push_next(&mut bda_ci);
        if has_i8_dot {
            device_ci = device_ci.push_next(&mut intdot_ci);
        }
        if has_coop_matrix {
            device_ci = device_ci.push_next(&mut coopmat_ci);
        }
        if has_ext_sem {
            device_ci = device_ci.push_next(&mut timeline_sem_ci);
        }

        let device = unsafe { instance.create_device(physical_device, &device_ci, None) }
            .map_err(|e| be(format!("create_device: {e}")))?;
        // Register the device so any Err below (subgroup-32/env guards, allocator build) destroys
        // it instead of leaking it — see `InstanceCleanup` above.
        cleanup.device = Some(device.clone());

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // ── command pool ───────────────────────────────────────────────────────
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| be(format!("create_command_pool: {e}")))?;
        // Register the pool too, so a later Err frees it alongside the device.
        cleanup.pool = cmd_pool;

        // ── capabilities ───────────────────────────────────────────────────────
        // Query base limits + the subgroup-size range together via properties2 (the coopmat GEMM
        // pins requiredSubgroupSize=32, so the fallback ladder needs to know whether 32 is in range).
        let mut sgsize_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        // Maintenance3 (core in Vulkan 1.1) carries `maxMemoryAllocationSize` — the weight arena
        // splits its up-front reservation into blocks no larger than this.
        let mut maint3_props = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut sgsize_props)
            .push_next(&mut maint3_props);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let props = props2.properties;
        // 0 = not reported → fall back to the Vulkan-guaranteed floor (2^30 = 1 GiB).
        let max_mem_alloc_size = if maint3_props.max_memory_allocation_size == 0 {
            1 << 30
        } else {
            maint3_props.max_memory_allocation_size
        };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        // (0,0) when subgroup-size-control is unsupported: can't pin any size — the adapter treats
        // that as "no 32-pin available" and uses the driver's default subgroup for the fallback.
        let (subgroup_min, subgroup_max) = if has_sgsize {
            (
                sgsize_props.min_subgroup_size,
                sgsize_props.max_subgroup_size,
            )
        } else {
            (0, 0)
        };

        // infr's Vulkan compute kernels are written for a PINNED subgroup size of 32 (RDNA3 wave32):
        // rmsnorm / softmax / quant_q8 / the coopmat GEMM / attention QK+PV / flash / DeltaNet all
        // dispatch via `kernel_sg(..., 32)`, which sets `requiredSubgroupSize=32` and FAILS pipeline
        // creation on any device that can't provide a size-32 subgroup (no `subgroup_size_control`,
        // or 32 outside `[minSubgroupSize, maxSubgroupSize]`). rmsnorm/softmax run on EVERY forward
        // and are NOT coopmat-gated, so gating only the coopmat caps wouldn't prevent the crash —
        // the whole backend needs 32. Refuse the Vulkan backend here (a clean Err, not a mid-forward
        // panic) so `gpu_available()`/the seam falls back to CPU. Every real target — RADV (32-64),
        // NVIDIA (32), Intel Arc (…-32) — provides 32; this only rejects exotic no-32 /
        // no-size-control devices (older/mobile/llvmpipe), which can't run these wave32 kernels
        // correctly anyway.
        if !(has_sgsize && subgroup_min <= 32 && 32 <= subgroup_max) {
            return Err(be(format!(
                "infr's Vulkan backend requires a pinnable subgroup size of 32 (wave32); this \
                 device's subgroup range is [{subgroup_min}, {subgroup_max}] and \
                 subgroup_size_control={has_sgsize} — falling back to another backend"
            )));
        }

        // ── sg_pref: pinned subgroup size for the decode GEMV/reduction family ────────────────
        // Capability-driven: any device whose smallest pinnable subgroup is ≤16 likely has
        // SIMD8/SIMD16 EUs (Intel Arc, future small-subgroup GPUs) where pinning the decode
        // GEMV family at 32 starves per-lane registers (llama.cpp pins 16 for mul_mat_vec
        // on Intel for exactly this). `max(16, subgroup_min)` keeps this Battlemage-proof
        // (min=8 SKUs still get 16, never 8 — the kernels' lane math is only built for 16/32).
        // Everything else (RADV 32-64, NVIDIA 32) keeps 32, so the default kernel/pipeline
        // set there is byte-identical to before this field existed.
        let sg_default = if subgroup_min <= 16 {
            16u32.max(subgroup_min)
        } else {
            32
        };
        // `device.subgroup_pref` (`INFR_SG=16|32`): A/B override (Intel testers; inert on devices
        // that can't pin the value). The env layer already rejects anything but "16"/"32" with the
        // pre-S5a wording, but the field is now also reachable from the TOML file and `--set`, so
        // the "only 16 and 32 have builds" POLICY stays here at the consumer (R5) and still refuses
        // a value it has no kernels for.
        let sg_pref = match cfg.device.subgroup_pref {
            Some(16) => 16,
            Some(32) => 32,
            Some(other) => {
                return Err(be(format!(
                    "device.subgroup_pref (INFR_SG) must be 16 or 32 (got {other}) — the decode \
                     GEMV family only has subgroup-16 and subgroup-32 builds"
                )))
            }
            None => sg_default,
        };
        // A 16 request/default is only usable where 16 is pinnable; otherwise CLEANLY fall back
        // to 32 (e.g. INFR_SG=16 on RADV wave32: subgroup_min == 32 → stays 32, path set
        // unchanged). 32 is always pinnable here (hard-required above).
        let sg_pref = if sg_pref == 16 && !(subgroup_min <= 16 && 16 <= subgroup_max) {
            eprintln!(
                "[infr] INFR_SG=16 requested but this device's subgroup range \
                 [{subgroup_min}, {subgroup_max}] cannot pin 16 — keeping 32"
            );
            32
        } else {
            sg_pref
        };

        // ── integrated GPU + compute-unit count ───────────────────────────────────────────────
        // An iGPU/APU is NOT just "a slow discrete card": it is forced onto the non-coopmat kernel
        // tier (RDNA2/Raphael enumerates no cooperative matrix at all) AND carries ~1/50th the
        // compute, so a prefill chunk sized for a 96-CU card becomes a single multi-SECOND command
        // buffer — past the ~10 s `gfx`-ring watchdog it is a GPU reset, not merely slow. Detect the
        // device class here and let the seam bound its per-submit work (`Capabilities::integrated`).
        let integrated = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
        // Best-effort CU count (AMD only; 0 = unknown). `VK_AMD_shader_core_properties` is a
        // properties2 pNext, so it needs no device-extension ENABLE — only that the driver
        // advertises it, hence the `has_ext` guard (chaining an unsupported struct is UB).
        let compute_units = if has_ext(c"VK_AMD_shader_core_properties") {
            let mut sc = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sc);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
            sc.shader_engine_count
                * sc.shader_arrays_per_engine_count
                * sc.compute_units_per_shader_array
        } else {
            0
        };

        let caps = Capabilities {
            name: device_name,
            f16: has_f16,
            coopmat_f16,
            f8: has_f8_ext,
            coopmat_f8,
            i8: has_int8,
            i8_dot: has_i8_dot,
            coopmat_i8,
            bf16: has_bf16_ext,
            coopmat_bf16,
            subgroup_min,
            subgroup_max,
            sg_pref,
            integrated,
            compute_units,
            buffer_device_address: has_bda,
            max_shared_memory_bytes: props.limits.max_compute_shared_memory_size,
            // An INTEGRATED_GPU has no VRAM to be separate FROM: its "device-local" heap is system
            // DDR reached through the GART (proven on RADV RAPHAEL_MENDOCINO — see `vram_info`,
            // which is the only consumer that matters today). A DISCRETE_GPU is never UMA, and
            // that is the class this must not perturb, so key off the device type exactly like
            // `integrated` above (llama.cpp's Vulkan backend sets its `uma` flag the same way).
            // Note this is a strictly WEAKER claim than `integrated`, which additionally means
            // "submits must stay under a TDR watchdog" — the two happen to coincide on Vulkan.
            unified_memory: integrated,
            // The seam adapter records the decode graph once and replays it (params-driven `_dyn`
            // kernels); the runner compiles the eligible qwen3 decode graph once.
            decode_replay: true,
            combined_gu: true,
            embed_gather: true,
            gpu_sample: true,
            argmax_rows: true,
            argmax_prob: true,
            // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate) — the
            // `rmsnorm_gate` kernel (rmsnorm.comp's -DGATE build). Collapses QkNorm→GatedAct's
            // read-after-write barrier into one dispatch. INFR_NO_GATED_RMSNORM forces the split
            // form for A/B.
            gated_rmsnorm: true,
            // Every KV write/read kernel maps position -> row modulo the cache's row capacity
            // (identity on full-context caches), so SWA layers may get window-sized ring caches.
            kv_swa_ring: true,
        };

        // Publish the device class BEFORE any caller can size a prefill chunk against it (the seam
        // reads this in `ubatch_rows`, which runs on the first session/KV allocation — strictly
        // after `VulkanBackend::new` returns).
        let _ = DEVICE_CLASS.set(DeviceClass {
            integrated: caps.integrated,
            compute_units: caps.compute_units,
        });

        // One-line device banner (stderr) — the first thing to check on a portability bug report:
        // which GPU was picked and which kernel tiers are live. `y`/`n` per capability + the
        // subgroup range + shared-mem budget. Printed on every `VulkanBackend::new()` (no
        // process-wide dedup): a single run constructing several backends on the same device (an
        // `infr bench` MTP rep loop; a CPU/Vulkan parity check) now genuinely means one construction
        // per printed line, not a duplicate — `DenseSeamChat`'s MTP chat path shares ONE backend
        // across `warmup()` + every turn (see `chat/vulkan.rs`'s `mtp_vk`), so an ordinary
        // `INFR_MTP=1` run prints exactly one banner again without needing this dedup.
        let yn = |b: bool| if b { "y" } else { "n" };
        eprintln!(
            "[infr] GPU: {} | f16:{} f16cm:{} bf16:{} bf16cm:{} f8:{} f8cm:{} i8:{} i8dot:{} i8cm:{} \
             subgroup:{}-{} sgp:{} shared:{}KB",
            caps.name,
            yn(caps.f16),
            yn(caps.f16_coopmat()),
            yn(caps.bf16),
            yn(caps.bf16_coopmat()),
            yn(caps.f8),
            yn(caps.f8_coopmat()),
            yn(caps.i8),
            yn(caps.i8_dot),
            yn(caps.i8_coopmat()),
            caps.subgroup_min,
            caps.subgroup_max,
            caps.sg_pref,
            caps.max_shared_memory_bytes / 1024,
        );
        // Submit splitter (see `VulkanShared::submit_dispatch_cap`): the initial, pre-measurement
        // cap. `device.submit_dispatches` (`INFR_SUBMIT_DISPATCHES`) overrides it (`0` = never
        // split) — the kill switch if this ever misjudges a device. A non-numeric value is still
        // rejected loudly, now by the env layer, with the SAME text
        // (`ConfigError::Env` renders as `INFR_SUBMIT_DISPATCHES: expected a dispatch count …`).
        let submit_dispatch_cap = cfg
            .device
            .submit_dispatches
            .unwrap_or_else(|| infr_core::initial_submit_dispatch_cap(caps.integrated));
        // Integrated GPUs run a DIFFERENT shape of forward (smaller prefill chunk, and the whole
        // pass split across several submits so no single command buffer can trip the GPU's hang
        // watchdog), so say so out loud: it is the first thing to check when an iGPU run hangs or
        // prefills slowly. Silent on every discrete device (nothing changed there).
        if caps.integrated {
            eprintln!(
                "[infr] GPU: INTEGRATED (cu:{}) — prefill chunk {} rows, forward split every {} \
                 dispatches to stay under the GPU hang watchdog; INFR_UBATCH / \
                 INFR_SUBMIT_DISPATCHES override",
                if caps.compute_units > 0 {
                    caps.compute_units.to_string()
                } else {
                    "?".to_string()
                },
                infr_core::integrated_ubatch_rows(caps.compute_units),
                submit_dispatch_cap,
            );
        }
        // On a unified-memory part the budget guard counts EVERY heap, not just the device-local
        // one (see `vram_info`) — a materially different capacity, and the second thing to check
        // when an iGPU either loads a model you didn't expect to fit or starts swapping. Print the
        // number it will actually budget against. Silent on every discrete device.
        if caps.unified_memory {
            let mp = unsafe { instance.get_physical_device_memory_properties(physical_device) };
            let (mut all, mut dev_local) = (0u64, 0u64);
            for i in 0..mp.memory_heap_count as usize {
                all += mp.memory_heaps[i].size;
                if mp.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                {
                    dev_local += mp.memory_heaps[i].size;
                }
            }
            eprintln!(
                "[infr] GPU: UNIFIED MEMORY — budgeting against all {} heaps ({}), not the \
                 device-local slice alone ({}); this GPU's memory IS system RAM",
                mp.memory_heap_count,
                fmt_bytes(all),
                fmt_bytes(dev_local),
            );
        }

        // ── gpu-allocator ──────────────────────────────────────────────────────
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            // Every gpu-allocator allocation gets VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, which the
            // paged-MoE arena buffer (created with SHADER_DEVICE_ADDRESS usage) requires before it
            // can be bound. Harmless for all other buffers. `has_bda` is hard-required above.
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| be(format!("gpu_allocator::Allocator::new: {e}")))?;

        // ── on-disk pipeline cache (see `pcache.rs`) ───────────────────────────
        let pcache = crate::pcache::PcachePersist::new(&props, vkcfg.pipeline_cache_disk);
        let initial = pcache.as_ref().and_then(|p| p.load()).unwrap_or_default();
        let mut pc_info = vk::PipelineCacheCreateInfo::default();
        if !initial.is_empty() {
            pc_info = pc_info.initial_data(&initial);
        }
        // A corrupt-but-well-enveloped blob can still fail creation: retry empty, never fatal.
        let pipeline_cache = unsafe { device.create_pipeline_cache(&pc_info, None) }
            .or_else(|_| unsafe {
                device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
            })
            .unwrap_or(vk::PipelineCache::null());

        // Built before `instance`/`device` move into `VulkanShared` below.
        let push_descriptor =
            has_push_descriptor.then(|| ash::khr::push_descriptor::Device::new(&instance, &device));

        // External-memory fd loader (`vkGetMemoryFdKHR`/`vkGetMemoryFdPropertiesKHR`) — present only
        // when the device extension was enabled above. `Some` here is the sole gate the P2P path
        // checks (`p2p.rs`); `None` = this backend offers no host-less cross-device transport.
        let external_memory_fd =
            has_ext_mem_fd.then(|| ash::khr::external_memory_fd::Device::new(&instance, &device));
        // External-semaphore fd loader (`vkGetSemaphoreFdKHR`/`vkImportSemaphoreFdKHR`) — the gate for
        // the tensor-parallel GPU-side all-reduce sync. `Some` only when the fd ext AND the timeline
        // feature are both present (both enabled above); else the all-reduce uses the host fence.
        let external_semaphore_fd =
            has_ext_sem.then(|| ash::khr::external_semaphore_fd::Device::new(&instance, &device));

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        // Probed for UMA parts ONLY — on a discrete card the non-device-local heap is host RAM
        // across PCIe and must never receive a GpuOnly buffer (see `probe_uma_overflow_type`).
        let uma_overflow_type = caps
            .unified_memory
            .then(|| probe_uma_overflow_type(&mem_props))
            .flatten();
        // Same probe, but WITHOUT the UMA gate — on a discrete card this resolves to the GTT
        // host-visible type (system RAM over PCIe). Only the opt-in KV-overflow path uses it.
        let host_overflow_type = probe_uma_overflow_type(&mem_props);

        // Success: the instance/device/pool now move into `VulkanShared` (which owns their
        // destruction). Disarm so `cleanup`'s Drop is a no-op and never double-frees them.
        cleanup.armed = false;

        Ok(Self {
            moe_pager: Mutex::new(None),
            dense_pager: Mutex::new(None),
            bda_weight_arena: Mutex::new(None),
            cfg,
            shared: Arc::new(VulkanShared {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family_index,
                cmd_pool: Mutex::new(cmd_pool),
                allocator: ManuallyDrop::new(Mutex::new(allocator)),
                caps,
                has_mem_budget,
                max_mem_alloc_size,
                push_descriptor,
                external_memory_fd,
                has_dma_buf: has_ext_mem_dma_buf,
                external_semaphore_fd,
                kernels: Mutex::new(HashMap::new()),
                pipeline_cache,
                pcache,
                weight_pb: Mutex::new(None),
                device_used: AtomicU64::new(0),
                submit_dispatch_cap: AtomicUsize::new(submit_dispatch_cap),
                uma_overflow_type,
                host_overflow_type,
                kv_spill: SpillTally::default(),
                staging_ring: Mutex::new(None),
            }),
        })
    }

    /// The submit splitter's current cap — see `VulkanShared::submit_dispatch_cap`. `0` =
    /// unlimited (record the whole forward into one command buffer, the discrete default).
    pub(crate) fn submit_dispatch_cap(&self) -> usize {
        self.shared.submit_dispatch_cap.load(Ordering::Relaxed)
    }

    /// Feed a completed forward's measurement back into the submit splitter: `elapsed` of wall
    /// across `dispatches` dispatches, measured with the queue drained at both ends. Re-tunes the
    /// cap so the NEXT forward's segments land inside `infr_core::SUBMIT_BUDGET_NS` on whatever
    /// device this actually is — the loop that makes the splitter hardware-agnostic instead of a
    /// table of per-GPU constants.
    ///
    /// The cap only ever RATCHETS DOWN within a process. A forward's wall time is a noisy sample
    /// (a cold pipeline compile, a busy host, a first-touch page fault all inflate it), and the
    /// asymmetry of the two mistakes is total: too small a cap costs a few extra submits, too
    /// large a cap costs a device-lost. So a slow sample tightens the bound and a fast one is
    /// simply ignored.
    pub(crate) fn observe_forward(&self, elapsed: std::time::Duration, dispatches: usize) {
        let ns = elapsed.as_nanos() as u64;
        let cur = self.submit_dispatch_cap();
        // A device that has never split (every discrete GPU) only starts splitting if a forward
        // actually lands in watchdog territory — see `infr_core::SUBMIT_DANGER_NS`. Without this,
        // the measured cap (finite, just very large) would eventually split a big enough graph on
        // a perfectly healthy dGPU, which is a tuned path this has no business touching.
        if cur == 0 && ns < infr_core::SUBMIT_DANGER_NS {
            return;
        }
        let want = infr_core::submit_cap_from_measurement(ns, dispatches);
        if want == 0 {
            return; // measurement says "no split needed"; never loosen an existing cap on it
        }
        let next = if cur == 0 { want } else { cur.min(want) };
        if next != cur {
            self.shared
                .submit_dispatch_cap
                .store(next, Ordering::Relaxed);
        }
    }

    /// Begin a "loading weights" progress bar covering `total_bytes` (pass `None` for an
    /// indeterminate byte spinner when the total isn't known up front). Every subsequent
    /// `BufferUsage::Weights` allocation advances it automatically — the ticking lives in `alloc`,
    /// so a model loader cannot forget it; it only has to open the scope once. The returned guard
    /// finishes and clears the bar on drop, so the bar's lifetime is the loader's scope.
    fn weight_progress_scope(&self, total_bytes: Option<u64>) -> WeightProgress {
        // Weights are read by 64-bit device address and sub-allocate from the BDA arena
        // (`bda_weight_alloc`, opened lazily on the first `Weights` alloc); there is no separate
        // up-front SSBO reservation or ReBAR direct-write path to arm here. The upload path is the
        // reused, pipelined staging ring (`upload_staged_ring`) on every device.
        let pb = infr_core::progress::bar(
            total_bytes,
            "loading weights",
            infr_core::progress::Unit::Bytes,
        );
        *self.shared.weight_pb.lock().unwrap() = Some(pb);
        WeightProgress {
            shared: self.shared.clone(),
            vram_log: self.cfg.prof.vram,
        }
    }

    /// Install this model's paged-MoE session (see `pager::MoePagerSession`), sized but with no
    /// tensors registered yet — called BEFORE the seam's weight-load closure runs (see
    /// `pager::MoePagerLayout`'s doc for why the ordering matters: `Backend::moe_paged` must
    /// already read true by the time that closure's placeholder buffers are bound). Replaces any
    /// previous session (there is only ever one loaded model per process today).
    pub fn init_moe_pager(&self, layout: crate::pager::MoePagerLayout) -> Result<()> {
        let session = crate::pager::MoePagerSession::new(self, layout)?;
        *self.moe_pager.lock().unwrap() = Some(session);
        Ok(())
    }

    /// Register one paged layer's role tensor with the session `init_moe_pager` already installed
    /// — called from the seam's weight-load closure instead of uploading the tensor's full bytes.
    /// Panics if no session is installed (a caller bug: `init_moe_pager` must run first); errors
    /// if the layout has no pool matching the tensor's (role, per-expert bytes).
    pub fn register_paged_expert(
        &self,
        role: crate::pager::Role,
        buf_id: usize,
        source: crate::pager::ExpertSource,
    ) -> Result<()> {
        self.moe_pager
            .lock()
            .unwrap()
            .as_mut()
            .expect("register_paged_expert called before init_moe_pager")
            .register(role, buf_id, source)
    }

    /// `INFR_PAGER_STATS=1` reporting hook — a no-op when no paged model is loaded.
    pub fn print_moe_pager_stats(&self) {
        if let Some(s) = self.moe_pager.lock().unwrap().as_ref() {
            s.print_stats_if_enabled();
        }
    }

    /// Locked access to the paged-MoE session for the adapter's `execute_static` — `pub(crate)`
    /// (only `adapter.rs` reaches into this); see `pager.rs`'s module doc for why this lives
    /// outside the `Graph`/`Bindings` seam instead of a per-op flag.
    pub(crate) fn moe_pager(&self) -> &crate::pager::MoePagerCell {
        &self.moe_pager
    }

    /// Install this model's dense layer-streaming session (see `pager::DensePagerSession`) —
    /// `init_moe_pager`'s dense twin, same call-order contract (BEFORE the seam's weight-load
    /// closure binds the first placeholder, so `Backend::dense_paged` already reads true).
    pub fn init_dense_pager(&self, layout: crate::pager::DensePagerLayout) -> Result<()> {
        let session = crate::pager::DensePagerSession::new(self, layout)?;
        *self.dense_pager.lock().unwrap() = Some(session);
        Ok(())
    }

    /// Register one streamed dense block with the session `init_dense_pager` installed — called
    /// from the seam's weight-load closure instead of uploading the block's full bytes. Panics if
    /// no session is installed (a caller bug: `init_dense_pager` must run first).
    pub fn register_dense_stream(
        &self,
        pool: usize,
        buf_id: usize,
        source: crate::pager::DenseSource,
    ) -> Result<()> {
        self.dense_pager
            .lock()
            .unwrap()
            .as_mut()
            .expect("register_dense_stream called before init_dense_pager")
            .register(pool, buf_id, source)
    }

    /// `INFR_PAGER_STATS=1` reporting hook — a no-op when no dense-streamed model is loaded.
    pub fn print_dense_pager_stats(&self) {
        if let Some(s) = self.dense_pager.lock().unwrap().as_ref() {
            s.print_stats_if_enabled();
        }
    }

    /// [`Self::moe_pager`]'s dense twin — locked access for the adapter's `execute_static`.
    pub(crate) fn dense_pager(&self) -> &crate::pager::DensePagerCell {
        &self.dense_pager
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Create a `vk::Buffer` + gpu-allocator sub-allocation of the requested size/location.
    /// Device-local VRAM: total heap size and currently-available bytes. `available` comes from
    /// VK_EXT_memory_budget (live, accounts for other processes + our own allocations) when the
    /// extension is present; otherwise it falls back to the total heap size (best effort).
    /// NOTE: the extension's `heapBudget` is a CEILING (how much this process may use in total),
    /// not free bytes — live free = `heapBudget - heapUsage`. Reporting the raw budget here once
    /// made `available` sit ~constant while we allocated GBs, which let the VRAM guard sail past
    /// a 53 GiB KV cache into VK_ERROR_DEVICE_LOST.
    pub fn vram(&self) -> VramInfo {
        vram_info(&self.shared)
    }

    /// Device-memory budget guard: hard-error BEFORE a device-local allocation of `want` bytes
    /// that would exceed the budget. Over-committing does not fail cleanly on GPUs — the driver
    /// accepts the allocation and then evicts, which on a discrete card means reading weights back
    /// across PCIe (measured: a 41 GiB device-local run on a 24 GiB 7900 XTX quietly placed 18 GiB
    /// in GTT) and can end in a device-lost (TDR) mid-inference. The only safe failure point is
    /// here, at allocation time (mirrors the Metal backend's working-set guard).
    ///
    /// The budget comes from [`vram_info`], which counts device-local heaps on a discrete card and
    /// EVERY heap on a unified-memory part (where they are one pool of DDR — see its doc). Uses the
    /// LIVE per-heap budget when VK_EXT_memory_budget is present (it accounts for other processes
    /// and everything we already hold); otherwise falls back to this backend's tracked bytes
    /// against the total heap. `GUARD_HEADROOM` absorbs allocation slop (alignment, gpu-allocator
    /// block rounding) and driver-internal allocations. `INFR_NO_VRAM_GUARD=1` disables the check
    /// (restoring the old fail-late behavior).
    ///
    /// Sub-MiB allocations skip the check (no per-tiny-alloc driver query; they cannot
    /// individually blow the budget and stay covered by the next large allocation's check).
    fn check_vram_budget(&self, want: u64) -> Result<()> {
        const CHECK_MIN: u64 = 1 << 20; // 1 MiB
        if want < CHECK_MIN || self.cfg.kernels.vulkan.no_vram_guard {
            return Ok(());
        }
        // The probe is the single source of truth for "does `want` fit?" so the hard guard here and
        // the VRAM-first KV spill can never disagree (the guard errors iff the probe says it won't
        // fit). Only build the detailed error — and re-query the driver for its fields — on failure.
        if self.vram_budget_fits(want) {
            return Ok(());
        }
        let v = self.vram();
        let used = if v.live {
            v.total.saturating_sub(v.available)
        } else {
            self.shared.device_used.load(Ordering::Relaxed)
        };
        let budget = v.total.saturating_sub(GUARD_HEADROOM);
        let pool = if v.uma {
            "unified memory (all heaps — this GPU shares system RAM)"
        } else {
            "device-local"
        };
        Err(be(format!(
            "{} budget exceeded: {} requested + {} already in use ({}) > {} budget ({} {pool} \
                 minus 256 MiB headroom). Refusing to over-commit: exceeding it doesn't fail \
                 cleanly — the driver evicts (weights get read back over the bus) or the device is \
                 lost (TDR) mid-inference. Use a smaller context (INFR_CTX), a smaller/more- \
                 quantized model, close other GPU processes, or run on the CPU backend \
                 (INFR_DEV=cpu). INFR_NO_VRAM_GUARD=1 overrides at your own risk.",
            if v.uma { "Unified-memory" } else { "VRAM" },
            fmt_bytes(want),
            fmt_bytes(used),
            if v.live {
                "live driver budget"
            } else {
                "tracked by this process; no VK_EXT_memory_budget"
            },
            fmt_bytes(budget),
            fmt_bytes(v.total),
        )))
    }

    /// Non-erroring budget probe: would a device-local allocation of `want` bytes fit under the SAME
    /// budget [`check_vram_budget`](Self::check_vram_budget) enforces? The VRAM-first KV-overflow
    /// path (`make_alloc`'s `KvCache` arm with `INFR_KV_OVERFLOW`) needs "would this fit?" to choose
    /// VRAM vs host placement, not "error if not" — but it must agree with the guard to the byte, so
    /// this and the guard share `GUARD_HEADROOM` and the same `used`/`budget` math. A `true` here
    /// means `check_vram_budget(want)` returns `Ok` (the guard errors iff this returns `false`);
    /// unlike the guard it ignores `INFR_NO_VRAM_GUARD` and the sub-MiB skip — it is a placement
    /// decision, not a safety gate, so it always reports the honest budget answer.
    fn vram_budget_fits(&self, want: u64) -> bool {
        let v = self.vram();
        let used = if v.live {
            v.total.saturating_sub(v.available)
        } else {
            self.shared.device_used.load(Ordering::Relaxed)
        };
        let budget = v.total.saturating_sub(GUARD_HEADROOM);
        used.saturating_add(want) <= budget
    }

    /// First device-local memory type compatible with `type_bits` (from a buffer's requirements).
    fn find_memory_type(&self, type_bits: u32, props: vk::MemoryPropertyFlags) -> Option<u32> {
        let mp = unsafe {
            self.shared
                .instance
                .get_physical_device_memory_properties(self.shared.physical_device)
        };
        (0..mp.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(props)
        })
    }

    /// Bind `buffer` to a fresh, PERSISTENTLY MAPPED dedicated allocation of memory type `ty`. The
    /// UNIFIED-MEMORY overflow spill (`spilled == true` — the non-device-local heap of a UMA part),
    /// the sole caller today. See [`Backing::Vram`]. (`spilled == false` is still threaded for the
    /// budget-accounting split described below, in case a device-local mapped caller returns.)
    ///
    /// The caller owns `buffer` and must destroy it if this returns `Err`. Budget-guarded and
    /// charged to `device_used` like any other allocation UNLESS `spilled` (a UMA-overflow buffer
    /// lives off the device-local heap and is deliberately not counted against it).
    ///
    /// `device_address` must mirror the buffer's `SHADER_DEVICE_ADDRESS` usage: when the buffer was
    /// created with that usage (a paged-MoE `alloc_arena_bda` / resident-BDA `bda_weight_alloc`
    /// block that spilled onto the UMA overflow heap), its backing memory MUST carry the
    /// `DEVICE_ADDRESS` alloc flag too, or binding it violates
    /// VUID-VkMemoryAllocateInfo-flags-03331 (validation error / a bogus 64-bit `device_addr()`).
    /// This is the manual mirror of the flag gpu-allocator sets on its own path (it was built with
    /// `buffer_device_address: true`). Every non-device_address caller passes `false`.
    fn alloc_vram_mapped(
        &self,
        buffer: vk::Buffer,
        size: usize,
        requirements: &vk::MemoryRequirements,
        ty: u32,
        spilled: bool,
        device_address: bool,
        budget_check: bool,
    ) -> Result<VkBuffer> {
        // The KV-overflow path (`alloc_kv_host`) places bytes in SYSTEM RAM across PCIe, not VRAM,
        // so it opts out of the device-local budget guard entirely (`budget_check == false`). The
        // UMA spill keeps it: on a unified part every heap IS the same DDR pool and the guard
        // budgets against all of them.
        if budget_check {
            self.check_vram_budget(requirements.size)?;
        }

        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(ty);
        if device_address {
            alloc_info = alloc_info.push_next(&mut flags_info);
        }
        let memory = unsafe { self.shared.device.allocate_memory(&alloc_info, None) }
            .map_err(|e| be(format!("rebar allocate_memory({}): {e}", requirements.size)))?;

        let ptr = match unsafe {
            self.shared
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(p) => p as *mut u8,
            Err(e) => {
                unsafe { self.shared.device.free_memory(memory, None) };
                return Err(be(format!("rebar map_memory: {e}")));
            }
        };

        if let Err(e) = unsafe { self.shared.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.shared.device.unmap_memory(memory);
                self.shared.device.free_memory(memory, None);
            }
            return Err(be(format!("rebar bind_buffer_memory: {e}")));
        }

        // Charge the device-local budget tally — UNLESS this is a UMA/host spill, which lives off
        // the device-local heap and must not count against it (balanced in `VkBuffer::drop`).
        if !spilled {
            self.shared
                .device_used
                .fetch_add(requirements.size, Ordering::Relaxed);
        }

        // Mirror `make_buf_ex`'s own-address computation: `device_address` here means the caller
        // already added `SHADER_DEVICE_ADDRESS` to the buffer's usage AND (just above) chained the
        // matching `DEVICE_ADDRESS` memory-allocate flag, so the query is valid.
        let own_addr = device_address.then(|| unsafe {
            self.shared
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        });

        Ok(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            backing: Backing::Vram {
                memory,
                ptr,
                spilled,
            },
            // Logical size = what the caller asked for; `requirements.size` only rounds it up for
            // alignment, and `fill_buf`/`upload` must not touch past the logical extent.
            size,
            mem_size: requirements.size,
            location: MemoryLocation::GpuOnly,
            sub_offset: 0,
            own_addr,
        })
    }

    fn make_buf(&self, size: usize, location: MemoryLocation, label: &str) -> Result<VkBuffer> {
        self.make_buf_ex(size, location, label, false, false)
    }

    /// Allocate one KV-cache buffer in SYSTEM RAM (host-visible, non-device-local heap) WITH a
    /// device address — the opt-in `INFR_KV_OVERFLOW` path. The KV read seam is 100%
    /// `bufferDeviceAddress` (issue #74: `attn_partial`/`attention_kv`/dequant read K/V only
    /// through `k_addr`/`v_addr` pointers), so a KV buffer whose bytes live off-device is read by
    /// attention over PCIe with NO shader change; only the store→read barrier's inert bound
    /// descriptors still bind the same buffer, which is valid on any heap. Same bytes, different
    /// heap ⇒ bit-identical logits to a VRAM KV cache, at PCIe bandwidth.
    ///
    /// This is SYSTEM RAM, not VRAM, so it does NOT go through the device-local budget guard
    /// (`budget_check == false`) and is NOT charged to `device_used` (`spilled == true`) — leaving
    /// the VRAM guard to protect only the weights + activations that live on-device. Requires
    /// `host_overflow_type` (present on RDNA3 = the GTT host-visible type); the caller has already
    /// checked the flag, so a missing type here is a hard error rather than a silent VRAM fallback.
    fn alloc_kv_host(&self, size: usize) -> Result<VkBuffer> {
        let ty =
            self.shared.host_overflow_type.ok_or_else(|| {
                be("INFR_KV_OVERFLOW set but this device exposes no host-visible non-device-local \
                memory type to place the KV cache in — unset it to run in VRAM".to_string())
            })?;
        // KV buffers are addressed by device address (issue #74), so this buffer needs the
        // SHADER_DEVICE_ADDRESS usage exactly like the VRAM KV path (`make_buf_ex(.., true)`).
        let usage = vk::BufferUsageFlags::from_raw(
            BUFFER_USAGE.as_raw() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
        );
        // 4-byte-rounded create size (`fill_span`) — matches `make_buf_ex`; identity for any
        // 4-aligned `size` (all current KV buffers), keeps device-local zero-init in-bounds.
        let buf_ci = vk::BufferCreateInfo::default()
            .size(fill_span(size))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.shared.device.create_buffer(&buf_ci, None) }
            .map_err(|e| be(format!("create_buffer(kv-host): {e}")))?;
        let requirements = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };
        if requirements.memory_type_bits & (1 << ty) == 0 {
            unsafe { self.shared.device.destroy_buffer(buffer, None) };
            return Err(be(
                "INFR_KV_OVERFLOW: the host-visible overflow memory type is not compatible with a \
                 KV storage buffer on this device"
                    .to_string(),
            ));
        }
        // spilled=true (NOT charged to device_used), device_address=true,
        // budget_check=false (this is system RAM). On Err, `alloc_vram_mapped` leaves the buffer
        // to us.
        self.alloc_vram_mapped(buffer, size, &requirements, ty, true, true, false)
            .inspect_err(|_| unsafe { self.shared.device.destroy_buffer(buffer, None) })
    }

    /// Allocate the paged-MoE expert arena as a `bufferDeviceAddress` buffer and return it with its
    /// 64-bit `VkDeviceAddress`. Unlike a plain SSBO arena (capped at `maxStorageBufferRange`), the
    /// paged expert kernels read this through a `GL_EXT_buffer_reference` pointer, so it may be as
    /// large as VRAM allows. Always a dedicated GpuOnly allocation, budget-guarded like any weight;
    /// the `SHADER_DEVICE_ADDRESS` usage + the allocator's DEVICE_ADDRESS memory flag are what let
    /// `get_buffer_device_address` succeed.
    pub fn alloc_arena_bda(&self, bytes: usize) -> Result<(Box<dyn Buffer>, u64)> {
        let buf = self.make_buf_ex(bytes, MemoryLocation::GpuOnly, "moe-arena", true, true)?;
        // `make_buf_ex(device_address=true)` already queried + stored this handle's address in
        // `own_addr` — reuse it rather than issuing a second identical `get_buffer_device_address`.
        let addr = buf
            .own_addr
            .expect("moe-arena built with device_address=true carries an own_addr");
        Ok((Box::new(buf) as Box<dyn Buffer>, addr))
    }

    /// Sub-allocate `size` bytes for a resident weight tensor from the BDA arena (see
    /// [`BdaWeightArena`]). Bump-allocates within the current block at [`BDA_WEIGHT_ALIGN`]; when the
    /// request doesn't fit the current block's remainder, opens a fresh dedicated block (never
    /// splitting a tensor across two blocks). Returns a `VkBuffer` that shares the
    /// block's single `vk::Buffer` handle (`Backing::BdaSub`) with `sub_offset` set to this tensor's
    /// offset within it — see that variant's doc for why the handle must never be bound as a
    /// descriptor at its full range.
    fn bda_weight_alloc(&self, size: usize) -> Result<VkBuffer> {
        // Load-time guard for the BYTE half of the addressing invariant (see
        // [`BDA_ADDRESSING_UNIT_MAX`] / [`BdaWeightArena`]): a single arena tensor at or above 4 GiB
        // would wrap the u32 intra-unit byte offsets the STREAMED kernels apply, silently reading
        // the wrong weights. Reject it LOUDLY here rather than let it corrupt output in-kernel. This
        // is distinct from the `max_mem_alloc_size` block error below (a device-allocation limit) —
        // this is an addressing limit that a bigger device or heap does NOT lift. The ELEMENT half
        // of the invariant (< 4 Gi elements, the binding cap for sub-byte quants) needs shape+dtype
        // and is enforced at the geometry chokepoints (`expert_stride_bytes`, the loader seam).
        if size as u64 >= BDA_ADDRESSING_UNIT_MAX {
            return Err(be(format!(
                "resident-BDA weight tensor ({size} bytes) exceeds the u32 addressing unit \
                 ({BDA_ADDRESSING_UNIT_MAX} bytes / 4 GiB) — intra-tensor offsets are u32; a single \
                 tensor this large needs a wider addressing scheme, not a bigger allocation"
            )));
        }
        let want = (size as u64).max(1).next_multiple_of(BDA_WEIGHT_ALIGN);
        let mut guard = self.bda_weight_arena.lock().unwrap();
        let arena = guard.get_or_insert_with(BdaWeightArena::default);

        // First-fit over ALL open blocks, not just the last: a big tensor opens an exact-size
        // (fully consumed) block, so last-block-only would strand every earlier block's tail and
        // open a fresh `BDA_BLOCK_MIN` block for EACH tiny tensor that follows a big one — ~64 MiB
        // stranded per norm gamma, GiBs across a model's layers (caught live: Qwen3-30B-A3B
        // tripped the VRAM guard flag-on while fitting comfortably flag-off).
        for b in arena.blocks.iter_mut() {
            let off = b.cursor.div_ceil(BDA_WEIGHT_ALIGN) * BDA_WEIGHT_ALIGN;
            if off + want <= b.size {
                b.cursor = off + want;
                return Ok(VkBuffer {
                    shared: Arc::clone(&self.shared),
                    buffer: b.handle.buf.buffer,
                    backing: Backing::BdaSub(Arc::clone(&b.handle)),
                    size,
                    mem_size: 0,
                    location: MemoryLocation::GpuOnly,
                    sub_offset: off as usize,
                    // `device_addr()` derives a `BdaSub`'s address from the block's `base_addr` +
                    // `sub_offset` — no own address needed here.
                    own_addr: None,
                });
            }
        }

        // The current block (if any) has no room: open a fresh dedicated block sized to the
        // request, floored at `BDA_BLOCK_MIN` and capped at `max_mem_alloc_size` — a whole
        // multi-GiB model can't be one `vkAllocateMemory`, same reasoning as `WeightArena`'s block
        // cap. A single tensor bigger than the cap can't be placed at all; report it rather than
        // silently truncating.
        let block_bytes = want.max(BDA_BLOCK_MIN).min(self.shared.max_mem_alloc_size);
        if want > block_bytes {
            return Err(be(format!(
                "resident-BDA weight tensor ({want} bytes) exceeds max_mem_alloc_size ({} bytes) \
                 — cannot fit in a single arena block",
                self.shared.max_mem_alloc_size
            )));
        }
        let block_buf = self.make_buf_ex(
            block_bytes as usize,
            MemoryLocation::GpuOnly,
            "resident-bda",
            true,
            true,
        )?;
        let buffer = block_buf.buffer;
        // `make_buf_ex(device_address=true)` already stored the block's address in `own_addr`;
        // reuse it instead of a second identical `get_buffer_device_address`.
        let base_addr = block_buf
            .own_addr
            .expect("resident-bda block built with device_address=true carries an own_addr");
        let handle = Arc::new(BdaBlockHandle {
            buf: block_buf,
            base_addr,
        });
        arena.blocks.push(BdaArenaBlock {
            handle: Arc::clone(&handle),
            size: block_bytes,
            cursor: want,
        });
        Ok(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            backing: Backing::BdaSub(handle),
            size,
            mem_size: 0,
            location: MemoryLocation::GpuOnly,
            sub_offset: 0,
            own_addr: None,
        })
    }

    /// Test-support hook: sub-allocate a resident-BDA weight tensor via [`Self::bda_weight_alloc`]
    /// directly — the same "construct the arena alloc directly" approach
    /// `resident_bda_weight_arena_roundtrip` (this module's own `#[cfg(test)]`) uses, exposed as
    /// `pub` so an external `tests/*.rs` integration binary (which only links the crate's public
    /// API, never its private items) can build a buffer whose `device_addr()` reports `Some` and
    /// drive dispatch routing on it. Boxed as `Box<dyn Buffer>` since [`VkBuffer`] itself is
    /// private.
    pub fn bda_weight_alloc_for_test(&self, size: usize) -> Result<Box<dyn Buffer>> {
        self.bda_weight_alloc(size)
            .map(|b| Box::new(b) as Box<dyn Buffer>)
    }

    /// [`make_buf`](Self::make_buf) with an explicit dedicated-allocation override. Post-load
    /// memory hygiene: `force_dedicated` bypasses gpu-allocator's general (sub-allocating)
    /// memory blocks entirely, so a TRANSIENT buffer frees its `VkDeviceMemory` fully on drop.
    /// Without it, sub-block transients grow general blocks the allocator then RETAINS: the
    /// vendored gpu-allocator (0.27) frees an emptied general block only while another general
    /// block exists in the same memory type (`active_general_blocks > 1` in its `free()`), and
    /// exposes no purge/trim API — so the last 64 MiB host-visible block (and a 256 MiB
    /// device-local one) would sit empty in the ReBAR heap for the whole session. Used by the
    /// weight-upload staging path below; never on a per-token path (a dedicated allocation costs
    /// a `vkAllocateMemory`, fine once per tensor at load, wrong per token).
    fn make_buf_ex(
        &self,
        size: usize,
        location: MemoryLocation,
        label: &str,
        force_dedicated: bool,
        device_address: bool,
    ) -> Result<VkBuffer> {
        // `device_address` (the paged-MoE / resident-BDA arena blocks): add SHADER_DEVICE_ADDRESS
        // so the buffer can be handed to a shader as a 64-bit pointer. Its backing memory needs the
        // matching DEVICE_ADDRESS alloc flag — gpu-allocator sets it (built with
        // `buffer_device_address: true`) on the pooled path below, and the UMA-overflow spill passes
        // it through to `alloc_vram_mapped` explicitly.
        let usage = if device_address {
            vk::BufferUsageFlags::from_raw(
                BUFFER_USAGE.as_raw() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
            )
        } else {
            BUFFER_USAGE
        };
        // Created at the 4-byte-rounded size (`fill_span`) so the device-local zero-init can cover
        // the whole logical extent — see `fill_span`. Identity for any 4-aligned `size` (all current
        // tensors), so `requirements.size`/the address are unchanged on every live path.
        let buf_ci = vk::BufferCreateInfo::default()
            .size(fill_span(size))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.shared.device.create_buffer(&buf_ci, None) }
            .map_err(|e| be(format!("create_buffer: {e}")))?;

        let requirements = unsafe { self.shared.device.get_buffer_memory_requirements(buffer) };

        // Large buffers (KV cache, big weights) get a DEDICATED exact-size VkDeviceMemory; otherwise
        // they sub-allocate into gpu-allocator's 256MB blocks and waste the remainder (e.g. 3×67MB
        // KV buffers per block leave ~55MB unused — ~0.7GB across a long-context KV cache). Small/
        // transient buffers stay sub-allocated (cheap, pooled).
        const DEDICATED_MIN: u64 = 32 * 1024 * 1024;
        let scheme = if force_dedicated || requirements.size >= DEDICATED_MIN {
            AllocationScheme::DedicatedBuffer(buffer)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };
        // Budget guard: fail fast, with a clear error, BEFORE committing device-local memory the
        // budget can't cover (host-visible staging/readback/host-weights are exempt — the guard
        // protects VRAM only).
        if location == MemoryLocation::GpuOnly {
            if let Err(e) = self.check_vram_budget(requirements.size) {
                unsafe { self.shared.device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        }

        // ── UMA spill: device-local heap full, put this on the other heap ─────────────────────
        // UNIFIED-MEMORY PARTS ONLY (`uma_overflow_type` is `None` on every discrete GPU, so a
        // dGPU never even evaluates this — it falls straight through to gpu-allocator exactly as
        // before). Once the synthetic device-local heap is full, gpu-allocator would keep resolving
        // GpuOnly to it and RADV would keep saying yes, right up until the kernel can't validate
        // the buffer list and the SUBMIT dies. Place the overflow on the non-device-local heap
        // instead: same DDR, same bandwidth on an APU. See `probe_uma_overflow_type`.
        if location == MemoryLocation::GpuOnly {
            if let Some(ty) = self.shared.uma_overflow_type {
                // Leave the device-local heap a little slack rather than filling it to the last
                // byte: the driver makes its own internal allocations there (descriptor pools,
                // pipeline/shader memory, the command buffers themselves), and a heap with zero
                // room is how the "not enough memory for command submission" failure starts.
                const UMA_SPILL_MARGIN: u64 = 256 * 1024 * 1024;
                let dl_avail = device_local_room(&self.shared);
                let fits_device_local =
                    dl_avail >= requirements.size.saturating_add(UMA_SPILL_MARGIN);
                if !fits_device_local && requirements.memory_type_bits & (1 << ty) != 0 {
                    // Both heaps are out if this fails, so report it rather than retrying on the
                    // device-local heap we just established has no room (that path would recurse
                    // straight back to here, and RADV would accept the allocation anyway and hand
                    // the failure to the next submit as a device-lost — the exact thing this
                    // spill exists to prevent).
                    return match self.alloc_vram_mapped(
                        buffer,
                        size,
                        &requirements,
                        ty,
                        true,
                        device_address,
                        // budget_check=false: the identical `check_vram_budget(requirements.size)`
                        // at the top of `make_buf_ex` already ran for this GpuOnly buffer — don't
                        // repeat it (and its memory-property driver round-trip) here.
                        false,
                    ) {
                        Ok(b) => Ok(b),
                        Err(e) => {
                            // `alloc_vram_mapped` leaves the buffer to us on failure.
                            unsafe { self.shared.device.destroy_buffer(buffer, None) };
                            Err(be(format!(
                                "unified memory exhausted: {} for {label} did not fit the \
                                 device-local heap ({} free) and the overflow heap rejected it \
                                 too ({e})",
                                fmt_bytes(requirements.size),
                                fmt_bytes(dl_avail),
                            )))
                        }
                    };
                }
            }
        }

        let allocation = {
            let mut alloc = self.shared.allocator.lock().unwrap();
            alloc
                .allocate(&AllocationCreateDesc {
                    name: label,
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: scheme,
                })
                .map_err(|e| {
                    // Clean up the buffer we created if allocation fails.
                    unsafe { self.shared.device.destroy_buffer(buffer, None) };
                    be(format!("gpu_allocator::allocate: {e}"))
                })?
        };

        unsafe {
            self.shared
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .map_err(|e| be(format!("bind_buffer_memory: {e}")))?;

        // Charge the budget guard's fallback accounting (balanced by `VkBuffer::drop`).
        if location == MemoryLocation::GpuOnly {
            self.shared
                .device_used
                .fetch_add(allocation.size(), Ordering::Relaxed);
        }

        // `device_address` callers added `SHADER_DEVICE_ADDRESS` to the buffer's usage above; the
        // allocator itself was built with `buffer_device_address: true` (see `VulkanShared::new`),
        // so every block it hands back — pooled sub-allocation or dedicated alike — already carries
        // the matching memory-allocate flag and the query below is valid without any extra plumbing.
        let own_addr = device_address.then(|| unsafe {
            self.shared
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        });

        Ok(VkBuffer {
            shared: Arc::clone(&self.shared),
            buffer,
            backing: Backing::Pooled(ManuallyDrop::new(allocation)),
            size,
            mem_size: requirements.size,
            location,
            sub_offset: 0,
            own_addr,
        })
    }

    /// Fill a buffer with the repeated byte `byte` (0x00 = zero-init, 0xFF = poison). Host-visible
    /// buffers are memset through the mapped pointer (no submit); device-local buffers use
    /// `vkCmdFillBuffer` via a one-shot submit. Every OTHER `VkBuffer` owns a distinct `vk::Buffer`
    /// handle addressing its region from offset 0 (plain `WeightArena` buffers included), so filling
    /// `[0, size)` of the handle is correct; a resident-BDA sub-tensor (`Backing::BdaSub`) instead
    /// shares its block's ONE big handle, so the fill must start at `buf.sub_offset` or it would
    /// clobber whatever tensor happens to sit at the block's byte 0.
    fn fill_buf(&self, buf: &VkBuffer, byte: u8) -> Result<()> {
        if let Some(ptr) = buf.mapped_ptr() {
            unsafe { std::ptr::write_bytes(ptr, byte, buf.size) };
        } else {
            let word = u32::from_ne_bytes([byte; 4]);
            let size = fill_span(buf.size); // round UP to a 4-byte multiple: cover the whole extent
            if size > 0 {
                let vkbuf = buf.buffer;
                let off = buf.sub_offset as u64;
                let shared = Arc::clone(&self.shared);
                self.one_shot(move |cmd| unsafe {
                    shared.device.cmd_fill_buffer(cmd, vkbuf, off, size, word);
                })?;
            }
        }
        Ok(())
    }

    /// Allocate `sizes.len()` buffers and zero-init them with (at most) ONE submit — the batched
    /// twin of [`Backend::alloc`], same calloc contract. `alloc`'s per-buffer `fill_buf` costs a
    /// one-shot submit + `queue_wait_idle` per device-local buffer; a graph execute's scratch set
    /// (~70 Internal tensors) paid ~2.5ms of pure submit overhead per call on a 7900 XTX. Here
    /// host-visible buffers are memset through their mapped pointer and every device-local fill is
    /// recorded into a single one-shot command buffer.
    pub(crate) fn alloc_zeroed_batch(
        &self,
        sizes: &[usize],
        usage: BufferUsage,
    ) -> Result<Vec<Box<dyn Buffer>>> {
        let bufs: Vec<VkBuffer> = sizes
            .iter()
            .map(|&b| self.make_alloc(b, usage))
            .collect::<Result<_>>()?;
        let mut dev: Vec<(vk::Buffer, u64, u64)> = Vec::new();
        for buf in &bufs {
            if let Some(ptr) = buf.mapped_ptr() {
                unsafe { std::ptr::write_bytes(ptr, 0u8, buf.size) };
            } else {
                let size = fill_span(buf.size); // round UP to a 4-byte multiple: cover the whole extent
                if size > 0 {
                    dev.push((buf.buffer, buf.sub_offset as u64, size));
                }
            }
        }
        if !dev.is_empty() {
            let shared = Arc::clone(&self.shared);
            self.one_shot(move |cmd| unsafe {
                for (b, off, size) in dev {
                    shared.device.cmd_fill_buffer(cmd, b, off, size, 0);
                }
            })?;
        }
        Ok(bufs
            .into_iter()
            .map(|b| Box::new(b) as Box<dyn Buffer>)
            .collect())
    }

    /// The shared body of `alloc`/`alloc_uninit`: pick the memory location + tick the weight-load
    /// progress bar. Zero/poison filling is applied by the callers.
    fn make_alloc(&self, bytes: usize, usage: BufferUsage) -> Result<VkBuffer> {
        // Weights are addressed exclusively by 64-bit device address: every `BufferUsage::Weights`
        // alloc sub-allocates from the BDA arena (`bda_weight_alloc`), the ONE weight path. Every
        // other `BufferUsage` takes the gpu-allocator path in `make_buf` below.
        if usage == BufferUsage::Weights {
            let buf = self.bda_weight_alloc(bytes)?;
            if let Some(pb) = self.shared.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
            return Ok(buf);
        }
        // KV cache: slice 0 of the u64/BDA migration (issue #74) — allocation-only enablement, NO
        // kernel reads this address yet (every attention/store/dequant dispatch still binds these
        // buffers exactly as before `vkb`). Unlike `Weights`, this is deliberately NOT an arena
        // sub-allocation: each `kbufs[l]`/`vbufs[l]` (and its fork/checkpoint/MTP-draft twins)
        // stays its OWN dedicated-or-pooled buffer object via the ordinary `make_buf_ex` path — the
        // per-layer/per-side structure is unchanged, it just gains `SHADER_DEVICE_ADDRESS` usage +
        // an `own_addr`. Smallest blast radius: only KV buffers get an address, not every
        // `Activations` scratch/partial/logits allocation in the engine.
        if usage == BufferUsage::KvCache {
            // Opt-in overflow (issue: KV-in-system-RAM), VRAM-FIRST: keep this KV buffer resident in
            // device-local VRAM while it fits the guard's budget; once VRAM is full, place it — and,
            // since the budget only shrinks as later buffers land, every subsequent one — in host RAM,
            // read by attention over PCIe via its device address (the read seam is 100% BDA — see
            // `alloc_kv_host`). This bounds PCIe cost to the overflow tail instead of paying it on the
            // whole cache; whole-host (slice-1 behavior) is now just the case where nothing fits.
            // Off by default ⇒ unchanged device-local VRAM KV.
            if kv_overflow_enabled(&self.cfg) {
                // Probe agrees with the guard to the byte (both key off `vram_budget_fits`), so a
                // `true` here means the VRAM alloc's own guard will pass. Guard against the rounding
                // slop between the requested `bytes` and the allocation's aligned size at the exact
                // budget edge by treating a VRAM alloc failure as "spill it" rather than propagating —
                // the whole point of overflow mode is to degrade to host, never to hard-error.
                //
                // `INFR_KV_OVERFLOW_VRAM_MB` (diagnostic) additionally caps CUMULATIVE KV-in-VRAM
                // bytes: it forces a partial (or, at 0, whole-host) spill on a model that would
                // otherwise fit entirely, so the mix path is exercisable on small models and the
                // whole-host case is reproducible apples-to-apples for benchmarking. It gates ONLY
                // this KV placement — never the real guard (`vram_budget_fits`/`check_vram_budget`),
                // which keeps protecting weights + activations against true VRAM.
                let cap_ok = self
                    .shared
                    .kv_spill
                    .admits(kv_overflow_vram_cap(&self.cfg), bytes as u64);
                if cap_ok && self.vram_budget_fits(bytes as u64) {
                    if let Ok(buf) =
                        self.make_buf_ex(bytes, MemoryLocation::GpuOnly, "kv-cache", false, true)
                    {
                        self.shared.kv_spill.record_vram(bytes as u64);
                        return Ok(buf);
                    }
                }
                let buf = self.alloc_kv_host(bytes)?;
                self.shared.kv_spill.record_host(bytes as u64);
                return Ok(buf);
            }
            return self.make_buf_ex(bytes, MemoryLocation::GpuOnly, "kv-cache", false, true);
        }
        let (location, label) = match usage {
            BufferUsage::Weights => unreachable!("Weights routed to bda_weight_alloc above"),
            BufferUsage::KvCache => unreachable!("KvCache routed to make_buf_ex above"),
            BufferUsage::Activations => (MemoryLocation::GpuOnly, "activations"),
            BufferUsage::Staging => (MemoryLocation::CpuToGpu, "staging"),
            BufferUsage::Readback => (MemoryLocation::GpuToCpu, "readback"),
            // GpuToCpu = HOST_VISIBLE|HOST_CACHED system RAM — the point of the class is NOT
            // living in VRAM.
            BufferUsage::HostWeights => (MemoryLocation::GpuToCpu, "host-weights"),
        };
        let buf = self.make_buf(bytes, location, label)?;
        // Advance the weight-load progress bar for host-weights too (the single funnel every
        // weight upload passes through, so no loader can forget to account for a tensor).
        if matches!(usage, BufferUsage::HostWeights) {
            if let Some(pb) = self.shared.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(buf)
    }

    /// Copy `src` into device-local `dst_buf` through the REUSED staging ring (see [`StagingRing`])
    /// — the weight-load path on a device without ReBAR.
    ///
    /// The tensor is chunked across fixed-size slots. For each chunk we wait only on the fence of
    /// the slot we are about to REUSE (not on the queue as a whole), memcpy into it, and submit its
    /// copy. With `RING_SLOTS` slots in flight the host's memcpy for chunk N+1 overlaps the DMA of
    /// chunk N, instead of the old `queue_wait_idle`-after-every-tensor lockstep.
    ///
    /// Uploads are not awaited here; [`WeightProgress::drop`] drains the ring, which happens long
    /// before any forward is submitted.
    ///
    /// `dst_base` is added to every chunk's destination offset — `0` for an ordinary weight buffer
    /// (the tensor owns `dst_buf` outright, so its region starts at byte 0), or a resident-BDA
    /// sub-tensor's `sub_offset` when `dst_buf` is actually a whole arena BLOCK shared with other
    /// tensors (see [`Backing::BdaSub`]) — without it every such tensor would land at the block's
    /// byte 0 and overwrite whatever the previous tensor wrote there.
    fn upload_staged_ring(&self, dst_buf: vk::Buffer, dst_base: u64, src: &[u8]) -> Result<()> {
        let device = &self.shared.device;
        let mut guard = self.shared.staging_ring.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.make_staging_ring()?);
        }
        let ring = guard.as_mut().expect("just built");

        let mut off = 0usize;
        while off < src.len() {
            let n = (src.len() - off).min(RING_SLOT_BYTES);
            let i = ring.next;
            ring.next = (ring.next + 1) % RING_SLOTS;

            // Reuse of a slot is the ONLY place we block: wait for its previous copy to land.
            if ring.busy[i] {
                unsafe { device.wait_for_fences(&[ring.fences[i]], true, u64::MAX) }
                    .map_err(|e| be(format!("staging ring wait_for_fences: {e}")))?;
                ring.busy[i] = false;
            }
            unsafe { device.reset_fences(&[ring.fences[i]]) }
                .map_err(|e| be(format!("staging ring reset_fences: {e}")))?;

            let ptr = ring.bufs[i]
                .mapped_ptr()
                .ok_or_else(|| be("staging ring slot is not mapped"))?;
            copy_to_mapped(&src[off..off + n], ptr);

            let cmd = ring.cmds[i];
            unsafe {
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| be(format!("staging ring reset_command_buffer: {e}")))?;
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| be(format!("staging ring begin_command_buffer: {e}")))?;
                device.cmd_copy_buffer(
                    cmd,
                    ring.bufs[i].buffer,
                    dst_buf,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: dst_base + off as u64,
                        size: n as u64,
                    }],
                );
                device
                    .end_command_buffer(cmd)
                    .map_err(|e| be(format!("staging ring end_command_buffer: {e}")))?;

                let cmds = [cmd];
                let submit = vk::SubmitInfo::default().command_buffers(&cmds);
                device
                    .queue_submit(self.shared.queue, &[submit], ring.fences[i])
                    .map_err(|e| be(format!("staging ring queue_submit: {e}")))?;
            }
            ring.busy[i] = true;
            off += n;
        }
        Ok(())
    }

    /// Allocate the staging ring's slots, command buffers and fences — ONCE per load.
    fn make_staging_ring(&self) -> Result<StagingRing> {
        let device = &self.shared.device;
        let pool = *self.shared.cmd_pool.lock().unwrap();
        let cmds = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(RING_SLOTS as u32),
            )
        }
        .map_err(|e| be(format!("staging ring allocate_command_buffers: {e}")))?;

        let mut bufs = Vec::with_capacity(RING_SLOTS);
        let mut fences = Vec::with_capacity(RING_SLOTS);
        for _ in 0..RING_SLOTS {
            bufs.push(self.make_buf(
                RING_SLOT_BYTES,
                MemoryLocation::CpuToGpu,
                "upload_staging",
            )?);
            fences.push(
                unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                    .map_err(|e| be(format!("staging ring create_fence: {e}")))?,
            );
        }
        Ok(StagingRing {
            bufs,
            cmds,
            fences,
            busy: vec![false; RING_SLOTS],
            next: 0,
        })
    }

    /// Record a single command into a one-shot command buffer, submit it to the
    /// compute queue, and block until idle.
    ///
    /// The closure receives the command buffer handle to record into.
    /// All operations are serialised through the `cmd_pool` mutex.
    fn one_shot(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.shared.device;
        let pool = *self.shared.cmd_pool.lock().unwrap();

        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("allocate_command_buffers: {e}")))?[0];

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| be(format!("begin_command_buffer: {e}")))?;

        f(cmd);

        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| be(format!("end_command_buffer: {e}")))?;

        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe { device.queue_submit(self.shared.queue, &[submit], vk::Fence::null()) }
            .map_err(|e| be(format!("queue_submit: {e}")))?;

        unsafe { device.queue_wait_idle(self.shared.queue) }
            .map_err(|e| be(format!("queue_wait_idle: {e}")))?;

        unsafe { device.free_command_buffers(pool, &cmds) };
        Ok(())
    }
}

// ── Backend impl ──────────────────────────────────────────────────────────────

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn capabilities(&self) -> Capabilities {
        self.shared.caps.clone()
    }

    fn weight_progress(
        &self,
        total_bytes: Option<u64>,
    ) -> Box<dyn infr_core::backend::ProgressScope> {
        Box::new(self.weight_progress_scope(total_bytes))
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // calloc contract: zero-init so recycled/uninitialized VRAM can't leak into a read-before-write.
        let buf = self.make_alloc(bytes, usage)?;
        self.fill_buf(&buf, 0x00)?;
        Ok(Box::new(buf))
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Opt-out: skip the zero-fill (caller guarantees the full extent is written before any read).
        // Debug builds poison with 0xFF (= NaN as f32) so a misuse surfaces loudly in tests;
        // `debug.poison_uninit` (`INFR_POISON_UNINIT=1`) forces the poison in release too — for
        // hunting layout-sensitive read-before-write bugs whose output shifts with unrelated code
        // changes.
        let buf = self.make_alloc(bytes, usage)?;
        #[cfg(debug_assertions)]
        self.fill_buf(&buf, 0xFF)?;
        #[cfg(not(debug_assertions))]
        if self.cfg.debug.poison_uninit {
            self.fill_buf(&buf, 0xFF)?;
        }
        Ok(Box::new(buf))
    }

    /// Copy `src` (host slice) into `dst` (device buffer).
    ///
    /// If `dst` is host-visible (`CpuToGpu`), writes directly through the
    /// Device-side prefix copy (`vkCmdCopyBuffer` region `[0, bytes)`) — no host bounce.
    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        let (s, d) = (as_vk_buf(src)?, as_vk_buf(dst)?);
        // BOTH sides, unlike `upload`/`download` which each have only one device buffer to police.
        // An oversize `bytes` here is a `vkCmdCopyBuffer` region that runs off the end of the source
        // and/or the destination — a VUID violation the driver is free to turn into a GPU fault or a
        // silent clobber of whatever sub-tensor happens to sit next in the arena block.
        check_extent("copy_buffer", "out of", bytes, s.size)?;
        check_extent("copy_buffer", "into", bytes, d.size)?;
        let (sb, db) = (s.buffer, d.buffer);
        // `sub_offset` — 0 for every ordinary buffer (all of today's callers pass KV/state
        // Activations buffers, whose handle IS the tensor), but a resident-BDA sub-tensor shares
        // its block's `vk::Buffer` and lives at its offset within it (see `Backing::BdaSub`), so
        // fold each side's `sub_offset` in the same way `upload`/`download`/`fill_buf` do.
        let (src_off, dst_off) = (s.sub_offset as u64, d.sub_offset as u64);
        // one_shot's queue_wait_idle provides the ordering fence vs prior/following work.
        self.one_shot(move |cmd| unsafe {
            let region = vk::BufferCopy {
                src_offset: src_off,
                dst_offset: dst_off,
                size: bytes as u64,
            };
            self.shared.device.cmd_copy_buffer(cmd, sb, db, &[region]);
        })
    }

    /// persistent mapped pointer.  Otherwise, creates a temporary staging buffer,
    /// writes there, then submits a `cmd_copy_buffer` to the compute queue.
    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let vk_dst = as_vk_buf(dst)?;
        // Same guard as `download`/`copy_buffer`, same wording — this one is the original, the
        // helper just stops the three from drifting apart again.
        check_extent("upload", "into", src.len(), vk_dst.size)?;

        // ── Direct write: any PERSISTENTLY MAPPED destination ─────────────────────────────────
        // Host-visible staging/readback buffers, AND a UMA overflow-spill buffer (`Backing::Vram`):
        // the host writes straight through the mapped pointer. One pass over the bytes, no staging
        // buffer, no `vkCmdCopyBuffer`, no queue stall. The memory is HOST_COHERENT so no explicit
        // flush is needed, and the host writes are made visible to the device by the implicit
        // host-write domain operation that `vkQueueSubmit` performs.
        if let Some(ptr) = vk_dst.mapped_ptr() {
            copy_to_mapped(src, ptr);
            return Ok(());
        }

        // ── Staged write: device-local destination with no host mapping ───────────────────────
        // During a weight load this goes through the REUSED, pipelined staging ring; anywhere else
        // (and for a tensor larger than the ring's slot on a non-load path) it is a single
        // synchronous copy.
        if self.shared.weight_pb.lock().unwrap().is_some() {
            return self.upload_staged_ring(vk_dst.buffer, vk_dst.sub_offset as u64, src);
        }

        let staging = self.make_buf(src.len(), MemoryLocation::CpuToGpu, "upload_staging")?;
        let stg_ptr = staging
            .mapped_ptr()
            .ok_or_else(|| be("staging buffer is not mapped"))?;
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), stg_ptr, src.len()) };

        let stg_buf = staging.buffer;
        let dst_buf = vk_dst.buffer;
        // `dst.sub_offset` — 0 for every ordinary buffer; a resident-BDA sub-tensor's offset within
        // its block's shared `vk::Buffer` otherwise (see `Backing::BdaSub`).
        let dst_off = vk_dst.sub_offset as u64;
        let size = src.len() as u64;
        // Clone the Arc so the closure is independent of `self`.
        let shared = Arc::clone(&self.shared);
        self.one_shot(move |cmd| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: dst_off,
                size,
            };
            unsafe {
                shared
                    .device
                    .cmd_copy_buffer(cmd, stg_buf, dst_buf, &[region])
            };
        })?;
        // `staging` is dropped here → frees vk::Buffer + gpu-allocator sub-allocation.
        Ok(())
    }

    /// Copy `src` (device buffer) into `dst` (host slice).
    ///
    /// If `src` is host-visible (persistently mapped — `Readback`/`GpuToCpu` OR `Staging`/CpuToGpu),
    /// reads STRAIGHT from the mapped pointer: zero submit/sync. Only a truly device-local
    /// (`GpuOnly`, unmapped) source copies via a temporary readback staging buffer + submit + wait.
    ///
    /// Covering CpuToGpu here matters on the hot decode loop: the record-once replay binds the
    /// device sampler's id output to a `Staging` buffer (`dec_ids_buf`, dual-purposed as the next
    /// iteration's on-device embed-gather input), and the per-token fallback / E2B path reads that
    /// id back every step. The old `GpuToCpu`-only check bounced it through a staging alloc +
    /// one_shot copy + `queue_wait_idle` PER TOKEN — the exact per-token full-sync cost `read_pos0`
    /// already dodges for `positions`. The mapped read carries the same contract the `GpuToCpu`
    /// path always had: the caller must have completed the GPU work that wrote `src` (every decode
    /// site does — `execute`/`replay` end in `queue_wait_idle`; the buffers are HOST_COHERENT so
    /// the write is visible with no explicit invalidate).
    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let vk_src = as_vk_buf(src)?;
        // The mirror of `upload`'s guard, and it is the one with TEETH: the mapped fast path below
        // is a raw `copy_nonoverlapping` of `dst.len()` bytes out of the mapping, so an oversize
        // `dst` is an out-of-bounds READ — undefined behaviour that hands the caller whatever
        // happens to follow the buffer (another tensor, another allocation) instead of failing.
        // The staging path's failure is tamer but still wrong: a `vkCmdCopyBuffer` whose `size`
        // overruns the source.
        check_extent("download", "out of", dst.len(), vk_src.size)?;

        if let Some(ptr) = vk_src.mapped_ptr() {
            // Host-visible (Readback or Staging): direct read from the persistently-mapped pointer.
            unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, dst.as_mut_ptr(), dst.len()) };
        } else {
            // Readback path: device-local → staging → host.
            let staging = self.make_buf(dst.len(), MemoryLocation::GpuToCpu, "download_staging")?;

            let src_buf = vk_src.buffer;
            let stg_buf = staging.buffer;
            // `src.sub_offset` — see `upload`'s `dst_off`; a resident-BDA sub-tensor reads from its
            // offset within the block's shared `vk::Buffer`, not from byte 0.
            let src_off = vk_src.sub_offset as u64;
            let size = dst.len() as u64;
            let shared = Arc::clone(&self.shared);
            self.one_shot(move |cmd| {
                let region = vk::BufferCopy {
                    src_offset: src_off,
                    dst_offset: 0,
                    size,
                };
                unsafe {
                    shared
                        .device
                        .cmd_copy_buffer(cmd, src_buf, stg_buf, &[region])
                };
            })?;

            // GPU→staging transfer is complete (queue_wait_idle returned).
            let ptr = staging
                .mapped_ptr()
                .ok_or_else(|| be("readback staging is not mapped"))?
                as *const u8;
            unsafe { std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len()) };
            // `staging` dropped here.
        }
        Ok(())
    }

    /// VRAM-first KV-overflow placement banner (see `make_alloc`'s `KvCache` arm). One shot after the
    /// runner's KV loop: how many KV buffers stayed resident in VRAM vs spilled to system RAM, so the
    /// partial split is visible. All-resident (nothing spilled) and all-spilled (slice-1 whole-host)
    /// are both reported. Nothing printed with the flag off or when no KV was allocated.
    fn kv_overflow_report(&self) {
        if !kv_overflow_enabled(&self.cfg) {
            return;
        }
        if let Some(line) = spill_report_line(self.shared.kv_spill.counts(), &KV_SPILL, fmt_bytes) {
            eprintln!("{line}");
        }
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        adapter::compile(self, graph)
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        adapter::execute(self, plan, bindings)
    }

    /// See `Backend::max_decode_chain`. A device that needs its FORWARD split into several
    /// submits (`submit_dispatch_cap` — every integrated part measured so far) cannot also afford
    /// to pack several decode steps into one: a decode graph is hundreds of dispatches, i.e.
    /// already at or past that cap on its own, so the honest bound there is a chain of ONE. A
    /// device that never splits (every discrete GPU) keeps the unbounded default, and the tuned
    /// chained-decode fast path is untouched.
    fn max_decode_chain(&self) -> usize {
        if self.submit_dispatch_cap() == 0 {
            usize::MAX
        } else {
            1
        }
    }

    fn execute_chain(
        &self,
        plan: &dyn Plan,
        bindings: &Bindings,
        n: usize,
    ) -> Result<Option<Vec<u32>>> {
        adapter::execute_chain(self, plan, bindings, n)
    }

    fn sync(&self) -> Result<()> {
        unsafe { self.shared.device.device_wait_idle() }
            .map_err(|e| be(format!("device_wait_idle: {e}")))
    }

    fn moe_paged(&self) -> bool {
        self.moe_pager.lock().unwrap().is_some()
    }

    fn dense_paged(&self) -> bool {
        self.dense_pager.lock().unwrap().is_some()
    }

    /// DiffusionGemma perf slice 3 (docs/diffusion-gemma.md): one eager dispatch of
    /// `dg_eb_sample` + a synchronous wait (`Recorder::finish`) — this isn't part of a cached
    /// [`Plan`], it runs once per denoise step right after that step's forward `execute()`, on
    /// the same `logits` buffer the forward just wrote (still GPU-resident).
    fn eb_sample_reduce(
        &self,
        logits: &dyn Buffer,
        u: &dyn Buffer,
        rows: usize,
        dim: usize,
        temp_inv: f32,
        argmax_out: &dyn Buffer,
        entropy_out: &dyn Buffer,
        sampled_out: &dyn Buffer,
    ) -> Result<bool> {
        let rec = self.recorder()?;
        rec.dg_eb_sample(
            logits,
            u,
            argmax_out,
            entropy_out,
            sampled_out,
            rows,
            dim,
            temp_inv,
        );
        rec.finish()?;
        Ok(true)
    }
}

/// Pick ONE cooperative-matrix (M,N,K) tile for a component type from the device's enumerated
/// shape list, by preference order:
///
/// 1. [`infr_core::COOPMAT_TILE_16`] (16x16x16) — the shape EVERY production coopmat shader is
///    built for; a device that enumerates it (RADV/RDNA3+, NVIDIA, and reportedly some
///    Battlemage drivers) always gets it, regardless of `allow_8x8x16` — the env knob must never
///    move a device off the proven kernel set.
/// 2. [`infr_core::COOPMAT_TILE_8`] (8x8x16, Intel Arc/ANV XMX) — only when `allow_8x8x16`
///    (the `INFR_CM_8X8=1` opt-in; only `native_gemm_warp`'s `_cm8` builds exist at this shape,
///    and Alchemist coopmat is a llama.cpp-documented regression, so it stays default-OFF).
/// 3. `None` — no shape any kernel here is built for; the non-coopmat tiers take over.
///
/// Pure function of the enumerated list + the opt-in flag (no env reads) so the selection is
/// unit-testable with synthetic property lists.
fn select_coopmat_shape(
    shapes: impl IntoIterator<Item = (u32, u32, u32)>,
    allow_8x8x16: bool,
) -> Option<(u32, u32, u32)> {
    let mut has_8x8x16 = false;
    for s in shapes {
        if s == infr_core::COOPMAT_TILE_16 {
            return Some(infr_core::COOPMAT_TILE_16);
        }
        has_8x8x16 |= s == infr_core::COOPMAT_TILE_8;
    }
    (allow_8x8x16 && has_8x8x16).then_some(infr_core::COOPMAT_TILE_8)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::Backend;

    /// A `Buffer` impl that is NOT a `VkBuffer` — stands in for what the multi-backend paths can
    /// actually hand a Vulkan op (another backend's buffer, or a `TpBuffer`/`EpBuffer` wrapper).
    struct ForeignBuffer {
        size: usize,
    }
    impl Buffer for ForeignBuffer {
        fn len_bytes(&self) -> usize {
            self.size
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// `as_vk_buf` must REJECT a buffer this backend did not allocate. Before the checked downcast
    /// it reinterpreted any `&dyn Buffer`'s data pointer as a `VkBuffer` — a foreign struct's
    /// leading bytes became a `vk::Buffer` handle and offsets, which then went to the driver. Since
    /// `infr multi` hosts several backends in one process (and the MTP draft path can mix them),
    /// this is a reachable mis-route, and it must be an `Error::Backend` rather than memory
    /// corruption. Needs no GPU: the downcast is pure type identity.
    #[test]
    fn as_vk_buf_rejects_a_foreign_buffer() {
        let foreign = ForeignBuffer { size: 4096 };
        let err = as_vk_buf(&foreign).map(|_| ());
        match err {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("not allocated by this VulkanBackend") && msg.contains("4096"),
                    "error should say the buffer is foreign and report what it could see: {msg}"
                );
            }
            Ok(()) => panic!("a non-VkBuffer must not downcast to &VkBuffer"),
        }
    }

    /// Finding #2: the device-local zero-init fill must cover the WHOLE logical extent, not the old
    /// `size / 4 * 4` truncation that left the trailing 1-3 bytes of a non-multiple-of-4 buffer
    /// holding recycled VRAM. `fill_span` rounds UP to the 4-byte multiple `vkCmdFillBuffer` needs,
    /// and the buffer is CREATED at that same span so the fill stays in-bounds.
    #[test]
    fn fill_span_covers_the_whole_buffer() {
        // 4-aligned sizes are unchanged (identity) — every current tensor, so the fill is
        // byte-for-byte what it was.
        for aligned in [0usize, 4, 8, 64, 4096] {
            assert_eq!(
                fill_span(aligned),
                aligned as u64,
                "4-aligned size is identity"
            );
        }
        // Non-multiple-of-4 sizes round UP and FULLY cover the logical extent (never truncate).
        for size in [1usize, 2, 3, 5, 6, 7, 13, 4095] {
            let span = fill_span(size);
            assert!(
                span >= size as u64,
                "fill must reach the last byte of size {size}"
            );
            assert!(
                span - size as u64 <= 3,
                "rounds up by at most 3 bytes (size {size})"
            );
            assert_eq!(
                span % 4,
                0,
                "vkCmdFillBuffer size must be a 4-byte multiple"
            );
            // The key regression guard: the OLD truncation would drop the tail.
            assert!(
                span > (size / 4 * 4) as u64,
                "must not truncate the tail of size {size}"
            );
        }
    }

    /// The transfer bounds guard shared by `upload`/`download`/`copy_buffer` (no GPU needed — this
    /// is why the check is a free function over two `usize`s instead of inline in each method).
    ///
    /// `download` used to have NO guard at all: it `copy_nonoverlapping`'d `dst.len()` bytes out of
    /// the source's mapping, so an oversize host slice read past the end of the buffer. `copy_buffer`
    /// checked neither side. The boundary that matters is `bytes == size` — a full-buffer transfer is
    /// legal and every real caller sits exactly there.
    #[test]
    fn check_extent_rejects_only_out_of_bounds_transfers() {
        // In-bounds, including the exact-fit and empty edges: never an error.
        for bytes in [0usize, 1, 31, 32] {
            assert!(
                check_extent("download", "out of", bytes, 32).is_ok(),
                "{bytes} bytes out of a 32-byte buffer is in bounds"
            );
        }
        // One past the end, and the pathological ask.
        assert!(check_extent("download", "out of", 33, 32).is_err());
        assert!(check_extent("copy_buffer", "into", usize::MAX, 32).is_err());
        // A zero-length buffer accepts only a zero-length transfer.
        assert!(check_extent("upload", "into", 0, 0).is_ok());
        assert!(check_extent("upload", "into", 1, 0).is_err());
        // The message keeps `upload`'s long-standing wording (it is the template the other two
        // callers were written to match), so a failure reads the same whichever path raised it.
        let msg = check_extent("upload", "into", 64, 32)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("upload: 64 bytes into a 32-byte buffer"),
            "unexpected message: {msg}"
        );
    }

    /// `INFR_DEV` index resolution (no GPU needed). Now the SINGLE device-selection env, it can hold
    /// `metal`/`cpu` — the Vulkan reader must TOLERATE those (fall back to the discrete default,
    /// `None`) rather than hard-erroring — while still hard-erroring on an out-of-range/garbage
    /// Vulkan index (typo protection preserved).
    #[test]
    fn infr_dev_index_tolerates_non_vulkan_specs() {
        let names: Vec<String> = vec!["Vulkan0=A".into(), "Vulkan1=B".into()];
        // Unset / empty → discrete default.
        assert_eq!(resolve_infr_dev_index(None, &names).unwrap(), None);
        assert_eq!(resolve_infr_dev_index(Some(""), &names).unwrap(), None);
        assert_eq!(resolve_infr_dev_index(Some("  "), &names).unwrap(), None);
        // metal / cpu (case-insensitive) → tolerated, discrete default (NOT an error).
        assert_eq!(resolve_infr_dev_index(Some("metal"), &names).unwrap(), None);
        assert_eq!(resolve_infr_dev_index(Some("cpu"), &names).unwrap(), None);
        assert_eq!(resolve_infr_dev_index(Some("Metal"), &names).unwrap(), None);
        assert_eq!(resolve_infr_dev_index(Some("CPU"), &names).unwrap(), None);
        // Valid Vulkan indices (VulkanN / bare N), case-insensitive.
        assert_eq!(
            resolve_infr_dev_index(Some("Vulkan0"), &names).unwrap(),
            Some(0)
        );
        assert_eq!(
            resolve_infr_dev_index(Some("vulkan1"), &names).unwrap(),
            Some(1)
        );
        assert_eq!(resolve_infr_dev_index(Some("1"), &names).unwrap(), Some(1));
        // Out-of-range → HARD ERROR (typo protection).
        assert!(resolve_infr_dev_index(Some("Vulkan99"), &names).is_err());
        // Unparseable Vulkan spec → HARD ERROR.
        assert!(resolve_infr_dev_index(Some("VulkanX"), &names).is_err());
    }

    /// A `Config` built as a VALUE reaches `pick_default_device` (S5a): `device.dev` is threaded in
    /// as a parameter, so the two consumers of the one raw string (`infr-cli`'s `parse_dev_spec`
    /// and this crate's index resolver, §10.8) each keep their own parse and neither reads the
    /// environment. No GPU needed — this pins the plumbing, `dev_from_config_selects_the_device`
    /// below pins the effect.
    #[test]
    fn config_device_dev_feeds_the_index_resolver() {
        let names: Vec<String> = vec!["Vulkan0=A".into(), "Vulkan1=B".into()];
        let spec = |v: Option<&str>| {
            let mut cfg = Config::default();
            cfg.device.dev = v.map(str::to_string);
            resolve_infr_dev_index(cfg.device.dev.as_deref(), &names)
        };
        assert_eq!(spec(None).unwrap(), None, "unset ⇒ the discrete default");
        assert_eq!(spec(Some("Vulkan1")).unwrap(), Some(1));
        assert_eq!(
            spec(Some("metal")).unwrap(),
            None,
            "tolerated, not an error"
        );
        assert!(spec(Some("Vulkan99")).is_err(), "typo protection preserved");
    }

    /// `kernels.vulkan.pipeline_cache_disk = false` (`INFR_NO_PIPELINE_CACHE`) must suppress DISK
    /// persistence — the whole knob — while the default keeps it. Driven through the field, not the
    /// environment (R7); no GPU needed since the constructor only reads properties + `$HOME`.
    #[test]
    fn pipeline_cache_disk_flag_gates_persistence() {
        let props = vk::PhysicalDeviceProperties::default();
        assert!(
            crate::pcache::PcachePersist::new(&props, false).is_none(),
            "cleared `pipeline_cache_disk` must disable on-disk persistence"
        );
        // The default is ON; it can still be `None` when there is no cache dir (no `HOME`/
        // `XDG_CACHE_HOME`), which is not this knob's doing — assert only the knob's direction.
        let has_cache_dir =
            std::env::var_os("XDG_CACHE_HOME").is_some() || std::env::var_os("HOME").is_some();
        assert_eq!(
            crate::pcache::PcachePersist::new(&props, true).is_some(),
            has_cache_dir,
            "with the flag set, persistence is on iff a cache dir exists"
        );
    }

    /// §5.2, on real hardware: the capability MASKERS are configuration, `Capabilities` stays a
    /// probe result. Clearing `kernels.vulkan.f16` must drop `caps.f16` AND `caps.f16_coopmat()`
    /// (f16 is a coopmat prerequisite — the AND `INFR_NO_F16` has always implied); clearing
    /// `coopmat` alone must drop the coopmat tiers while LEAVING `caps.f16` up; clearing `i8_dot`
    /// must drop `caps.i8_dot` alone. Each backend is built from a VALUE — no env, no `EnvGuard`,
    /// no ordering hazard (R7).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn config_masks_the_capability_probe() {
        let build = |f: &dyn Fn(&mut Config)| {
            let mut cfg = Config::default();
            f(&mut cfg);
            VulkanBackend::new_with(Arc::new(cfg)).expect("VulkanBackend::new_with")
        };
        let base = build(&|_| {});
        let (f16, cm, i8dot) = {
            let c = base.caps();
            (c.f16, c.f16_coopmat(), c.i8_dot)
        };
        drop(base);
        println!("baseline caps: f16={f16} f16cm={cm} i8dot={i8dot}");

        let no_f16 = build(&|c| c.kernels.vulkan.f16 = false);
        assert!(!no_f16.caps().f16, "`f16 = false` must clear caps.f16");
        assert!(
            !no_f16.caps().f16_coopmat(),
            "f16 is a coopmat prerequisite: clearing it must drop coopmat too"
        );
        drop(no_f16);

        let no_cm = build(&|c| c.kernels.vulkan.coopmat = false);
        assert!(
            !no_cm.caps().f16_coopmat(),
            "`coopmat = false` must drop the f16 coopmat tier"
        );
        assert_eq!(
            no_cm.caps().f16,
            f16,
            "`coopmat = false` must NOT touch plain f16"
        );
        drop(no_cm);

        let no_i8 = build(&|c| c.kernels.vulkan.i8_dot = false);
        assert!(
            !no_i8.caps().i8_dot,
            "`i8_dot = false` must clear caps.i8_dot"
        );
        assert_eq!(no_i8.caps().f16, f16, "i8_dot must not perturb f16");
    }

    /// `device.subgroup_pref` (`INFR_SG`) reaches `caps.sg_pref`, and a value the kernel set has no
    /// builds for is still refused LOUDLY — the policy moved to the consumer (R5) but did not
    /// vanish. `16` is only pinnable where the device's subgroup range admits it; on RADV wave32
    /// the documented fallback to 32 applies, so assert the reachable outcome rather than the
    /// request.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn config_subgroup_pref_drives_sg_pref() {
        let build = |sg: Option<u32>| {
            let mut cfg = Config::default();
            cfg.device.subgroup_pref = sg;
            VulkanBackend::new_with(Arc::new(cfg))
        };
        let be32 = build(Some(32)).expect("sg_pref 32 is always pinnable");
        assert_eq!(be32.caps().sg_pref, 32);
        let (min, max) = (be32.caps().subgroup_min, be32.caps().subgroup_max);
        drop(be32);

        let be16 = build(Some(16)).expect("sg_pref 16 request");
        let want16 = min <= 16 && 16 <= max;
        assert_eq!(
            be16.caps().sg_pref,
            if want16 { 16 } else { 32 },
            "a 16 request is honored iff the range [{min}, {max}] can pin it, else it falls back"
        );
        drop(be16);

        let msg = match build(Some(17)) {
            Ok(_) => panic!("subgroup_pref 17 must be refused: no subgroup-17 kernel builds exist"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("must be 16 or 32") && msg.contains("INFR_SG"),
            "the loud rejection kept its wording: {msg}"
        );
    }

    /// `kernels.vulkan.push_desc = false` (`INFR_NO_PUSH_DESC`) forces the pooled-classic
    /// descriptor path even where `VK_KHR_push_descriptor` exists — the loader must be absent, so
    /// the extension is genuinely not enabled on the logical device (a faithful simulation, not a
    /// flag flip). Skips itself on a device that has no push descriptors to drop.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn config_push_desc_forces_the_pooled_fallback() {
        let build = |on: bool| {
            let mut cfg = Config::default();
            cfg.kernels.vulkan.push_desc = on;
            VulkanBackend::new_with(Arc::new(cfg)).expect("VulkanBackend::new_with")
        };
        let dflt = build(true);
        let available = dflt.shared.push_descriptor.is_some();
        drop(dflt);
        if !available {
            eprintln!("skip: this device does not expose VK_KHR_push_descriptor");
            return;
        }
        assert!(
            build(false).shared.push_descriptor.is_none(),
            "`push_desc = false` must drop the push-descriptor loader"
        );
    }

    /// `kernels.vulkan.no_vram_guard` (`INFR_NO_VRAM_GUARD`) — a SANCTIONED negative field (§4).
    /// Set, it turns the alloc-time budget check into a no-op; clear, an ask several times the
    /// device's capacity is refused. Drives `check_vram_budget` directly, so it asserts the guard
    /// itself and allocates nothing.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn config_no_vram_guard_disables_the_budget_check() {
        let build = |off: bool| {
            let mut cfg = Config::default();
            cfg.kernels.vulkan.no_vram_guard = off;
            VulkanBackend::new_with(Arc::new(cfg)).expect("VulkanBackend::new_with")
        };
        let guarded = build(false);
        let huge = guarded.vram().total.saturating_mul(4);
        assert!(
            guarded.check_vram_budget(huge).is_err(),
            "the guard must refuse {huge} bytes on a device with {} total",
            guarded.vram().total
        );
        // The sub-MiB skip is unconditional and unchanged.
        assert!(guarded.check_vram_budget(1024).is_ok());
        drop(guarded);

        let unguarded = build(true);
        assert!(
            unguarded.check_vram_budget(huge).is_ok(),
            "`no_vram_guard = true` must make the check a no-op"
        );
    }

    /// `device.submit_dispatches` (`INFR_SUBMIT_DISPATCHES`) overrides the measured initial submit
    /// cap, `0` meaning "never split"; unset keeps `initial_submit_dispatch_cap(integrated)`.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn config_submit_dispatches_overrides_the_splitter_cap() {
        let build = |n: Option<usize>| {
            let mut cfg = Config::default();
            cfg.device.submit_dispatches = n;
            VulkanBackend::new_with(Arc::new(cfg)).expect("VulkanBackend::new_with")
        };
        let dflt = build(None);
        let integrated = dflt.caps().integrated;
        assert_eq!(
            dflt.submit_dispatch_cap(),
            infr_core::initial_submit_dispatch_cap(integrated)
        );
        drop(dflt);
        assert_eq!(build(Some(7)).submit_dispatch_cap(), 7);
        assert_eq!(build(Some(0)).submit_dispatch_cap(), 0, "0 = no split");
    }

    /// Shape selection over synthetic property lists (no GPU needed) — the caps-table core of the
    /// shape-aware coopmat gate. RADV-like (16x16x16 present, plus the other shapes RADV
    /// enumerates) must pick 16x16x16 regardless of the 8x8 opt-in; ANV-like (8x8x16 only) must
    /// stay dark by default and pick 8x8x16 only under the opt-in; empty (no coopmat) is None.
    #[test]
    fn coopmat_shape_selection() {
        let t16 = infr_core::COOPMAT_TILE_16;
        let t8 = infr_core::COOPMAT_TILE_8;
        // RADV-like: 16x16x16 f16 (RDNA3 WMMA). Opt-in must NOT move it off 16x16x16.
        let radv = [t16];
        assert_eq!(select_coopmat_shape(radv, false), Some(t16));
        assert_eq!(select_coopmat_shape(radv, true), Some(t16));
        // Device enumerating BOTH shapes: 16x16x16 preferred, opt-in irrelevant.
        let both = [t8, t16];
        assert_eq!(select_coopmat_shape(both, false), Some(t16));
        assert_eq!(select_coopmat_shape(both, true), Some(t16));
        // ANV-like (Intel Arc A770): 8x8x16 only — default OFF, opt-in selects it.
        let anv = [t8];
        assert_eq!(select_coopmat_shape(anv, false), None);
        assert_eq!(select_coopmat_shape(anv, true), Some(t8));
        // No configs (ext absent / feature off): None either way.
        assert_eq!(select_coopmat_shape([], false), None);
        assert_eq!(select_coopmat_shape([], true), None);
        // A shape no kernel is built for (e.g. a hypothetical 32x32x16): None even with opt-in.
        assert_eq!(select_coopmat_shape([(32, 32, 16)], true), None);
    }

    /// Resident-BDA weight arena: sub-allocate three odd-sized weight buffers directly from
    /// `bda_weight_alloc`, and verify:
    ///   * every sub-tensor reports `Some(device_addr)`,
    ///   * addresses are 256-byte aligned and strictly increasing within the (single, since the
    ///     three sizes together are far under `BDA_BLOCK_MIN`) block,
    ///   * distinct byte patterns uploaded to each round-trip intact — proving the `sub_offset`
    ///     plumbing on both `upload` (staged one-shot path, no weight-load scope open) and
    ///     `download`,
    ///   * a plain `Activations` alloc through the ordinary `Backend::alloc` path still reports
    ///     `device_addr() == None` — only `Weights` allocs route through `bda_weight_alloc`.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn resident_bda_weight_arena_roundtrip() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let sizes = [1000usize, 4096, 300_000];
        let mut addrs = Vec::new();
        let mut bufs = Vec::new();
        for (bi, &sz) in sizes.iter().enumerate() {
            let data: Vec<u8> = (0..sz)
                .map(|i| (i as u8).wrapping_add(bi as u8 * 53))
                .collect();
            let buf = be.bda_weight_alloc(sz).expect("bda_weight_alloc");
            let addr = buf
                .device_addr()
                .expect("resident-BDA sub-tensor must report Some(device_addr)");
            assert_eq!(
                addr % 256,
                0,
                "sub-tensor {bi} device_addr {addr:#x} is not 256-byte aligned"
            );
            if let Some(&prev) = addrs.last() {
                assert!(
                    addr > prev,
                    "sub-tensor {bi} device_addr {addr:#x} did not increase past the previous \
                     tensor's {prev:#x}"
                );
            }
            addrs.push(addr);

            be.upload(&buf, &data).expect("upload");
            let mut back = vec![0u8; sz];
            be.download(&buf, &mut back).expect("download");
            assert_eq!(
                back, data,
                "sub-tensor {bi} (size {sz}) round-trip mismatch"
            );
            bufs.push(buf);
        }
        // All three coexist in distinct byte ranges of the same block — re-check the first after
        // the later uploads landed, proving they didn't overlap/clobber it.
        let mut back0 = vec![0u8; sizes[0]];
        be.download(&bufs[0], &mut back0).expect("re-download");
        assert_eq!(
            back0[1], 1u8,
            "first sub-tensor corrupted by later resident-BDA allocs"
        );

        // A plain Activations alloc is unaffected by any of this — never routed through
        // `bda_weight_alloc`, so it must report no device address.
        let act = be
            .alloc(64, BufferUsage::Activations)
            .expect("Activations alloc");
        assert!(
            act.device_addr().is_none(),
            "an ordinary Activations buffer must not report a device_addr"
        );
    }

    /// Slice 0 of the KV-cache u64/BDA migration (issue #74): pure allocator-seam enablement — a
    /// `BufferUsage::KvCache` allocation (the exact usage class `infr-llama`'s `kbufs[l]`/
    /// `vbufs[l]`, their `fork()`/MTP-checkpoint/MTP-draft twins all route through as of this
    /// slice) must report `Some(device_addr)`. `kbufs` itself is `pub(super)` inside
    /// `infr_llama::seam` (invisible to any integration test), so this exercises the mechanism
    /// those call sites share rather than a live model session — exactly the "allocator seam
    /// only, zero behavioral change" scope of this slice (no kernel reads this address yet).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn kv_cache_buffer_reports_device_addr() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        // Two independent buffers, mirroring a K/V pair for one layer. UNLIKE the resident-weight
        // BDA arena, KV buffers are deliberately NOT consolidated into one shared block in this
        // slice (see `make_alloc`'s `KvCache` arm) — each stays its own dedicated-or-pooled
        // object, so their addresses must be distinct, non-null VkDeviceAddress values.
        let kbuf = be.alloc(4096, BufferUsage::KvCache).expect("KvCache alloc");
        let vbuf = be.alloc(4096, BufferUsage::KvCache).expect("KvCache alloc");
        let kaddr = kbuf
            .device_addr()
            .expect("kbuf must report Some(device_addr)");
        let vaddr = vbuf
            .device_addr()
            .expect("vbuf must report Some(device_addr)");
        assert_ne!(
            kaddr, 0,
            "device_addr must be a real (non-null) VkDeviceAddress"
        );
        assert_ne!(
            vaddr, 0,
            "device_addr must be a real (non-null) VkDeviceAddress"
        );
        assert_ne!(
            kaddr, vaddr,
            "K and V buffers must be independent objects, not shared/aliased"
        );

        // The buffer is still an ordinary bound-descriptor-usable SSBO — upload/download work
        // exactly as before (zero behavioral change to the actual KV read/write path; this slice
        // only adds the address, no kernel forks on it yet).
        let data: Vec<u8> = (0..4096u32).map(|i| i as u8).collect();
        be.upload(kbuf.as_ref(), &data).expect("upload");
        let mut back = vec![0u8; 4096];
        be.download(kbuf.as_ref(), &mut back).expect("download");
        assert_eq!(
            back, data,
            "KvCache buffer upload/download round-trip mismatch"
        );

        // A plain Activations alloc must still report no address — smallest blast radius: only
        // KvCache buffers gain one, not every scratch/partial/logits allocation in the engine.
        let act = be
            .alloc(64, BufferUsage::Activations)
            .expect("Activations alloc");
        assert!(
            act.device_addr().is_none(),
            "an ordinary Activations buffer must not report a device_addr"
        );
    }

    /// Dropping the backend must actually drop `VulkanShared` (device, allocator, weight arena —
    /// i.e. free the VRAM) even after a paged-MoE session was installed. The session's arena/LUT/
    /// ring buffers each hold an `Arc<VulkanShared>` clone, so parking the session ON
    /// `VulkanShared` formed an Arc cycle that leaked the whole device (~23 GiB after the Scout
    /// paged test) until process exit — every later model load in the same process then hit the
    /// VRAM budget guard with "N GiB already in use" (the cpu_backend gpu_ suite flake).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn backend_drop_frees_device_after_moe_pager() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        be.init_moe_pager(crate::pager::MoePagerLayout {
            n_blocks: 4,
            pools: vec![crate::pager::MoePoolSpec {
                role: crate::pager::Role::Gate,
                slot_bytes: 4096,
                n_slots: 2,
            }],
            ring_bytes: 1 << 20,
        })
        .expect("init_moe_pager");
        let weak = Arc::downgrade(&be.shared);
        drop(be);
        assert!(
            weak.upgrade().is_none(),
            "VulkanShared leaked after dropping the backend (Arc cycle via the paged-MoE session)"
        );
    }

    /// GPU f32 matmul correctness: compares `VulkanBackend::matmul_f32` against a CPU
    /// reference; asserts max relative error < 1e-3.
    ///
    /// Run with: `cargo test -p infr-vulkan -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn test_matmul_f32() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new failed");
        let caps = backend.capabilities();
        println!("device: {}", caps.name);

        let (m, k, n) = (32usize, 32usize, 32usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        // CPU reference
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        let c_gpu = backend
            .matmul_f32(&a, &b, m, k, n)
            .expect("matmul_f32 failed");

        let max_abs = c_gpu
            .iter()
            .zip(c_ref.iter())
            .map(|(g, r)| (*g - r).abs())
            .fold(0.0f32, f32::max);
        let max_ref = c_ref.iter().map(|r| r.abs()).fold(0.0f32, f32::max);
        let rel_err = if max_ref > 1e-6 {
            max_abs / max_ref
        } else {
            max_abs
        };

        println!("matmul {m}×{k}×{n}: max_rel_err = {rel_err:.2e}");
        assert!(rel_err < 1e-3, "matmul rel error too large: {rel_err:.2e}");
        println!("matmul GPU test PASS");
    }

    /// End-to-end roundtrip: init → alloc (device-local) → upload → download → assert.
    ///
    /// Marked `#[ignore]` so CI without a GPU passes; run manually with:
    /// ```text
    /// cargo test -p infr-vulkan -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn roundtrip_upload_download() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new failed");

        let caps = backend.capabilities();
        println!("=== Capabilities ===\n{caps:#?}\n");

        const N: usize = 1024;
        // Pattern: bytes 0x00..0xFF repeating.
        let pattern: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();

        // Alloc a device-local buffer (exercises the staging copy path).
        let buf = backend
            .alloc(N, BufferUsage::Weights)
            .expect("alloc Weights buffer");

        backend
            .upload(buf.as_ref(), &pattern)
            .expect("upload host→device");

        let mut got = vec![0u8; N];
        backend
            .download(buf.as_ref(), &mut got)
            .expect("download device→host");

        assert_eq!(pattern, got, "roundtrip data mismatch at 1024 bytes");

        backend.sync().expect("sync");

        println!("roundtrip OK — {N} bytes match");
    }
}

// qwen35 (Qwen3.5) SSM kernels: the GPU conv1d+SiLU and gated-DeltaNet recurrence must match the CPU
// reference. Self-skip without a GPU (so CI passes, runs locally with a device).
#[cfg(test)]
mod ssm_tests {
    use super::*;
    use infr_core::backend::{Buffer, BufferUsage};

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn softplus(x: f32) -> f32 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }
    fn det(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.137 + seed).sin()).collect()
    }
    fn dev(be: &VulkanBackend, data: &[f32]) -> Box<dyn Buffer> {
        let b = be
            .alloc((data.len() * 4).max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
        b
    }
    fn read(be: &VulkanBackend, buf: &dyn Buffer, n: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; n * 4];
        be.download(buf, &mut bytes).unwrap();
        bytemuck::cast_slice(&bytes).to_vec()
    }
    fn maxerr(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn softcap_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (n, cap) = (100usize, 30.0f32);
        let x = det(n, 0.5);
        let out_cpu: Vec<f32> = x.iter().map(|&v| cap * (v / cap).tanh()).collect();
        let xb = dev(&be, &x);
        let ob = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.softcap(xb.as_ref(), ob.as_ref(), cap, n);
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), n);
        let e = maxerr(&out_cpu, &out_gpu);
        assert!(e < 1e-4, "softcap err {e}");
    }

    #[test]
    fn deltanet_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (nv, nk, kd, vd) = (4usize, 2usize, 8usize, 8usize);
        let eps = 1e-6f32;
        let q = det(nk * kd, 0.1);
        let k = det(nk * kd, 0.7);
        let v = det(nv * vd, 1.3);
        let blog = det(nv, 2.0);
        let alpha = det(nv, 0.5);
        let acoef: Vec<f32> = (0..nv).map(|i| -(0.2 + 0.1 * i as f32)).collect();
        let dtbias = det(nv, -0.3);
        let state0 = det(nv * kd * vd, 0.05);

        // CPU reference (mirrors shaders/deltanet.comp + the qwen35 CPU mixer).
        let qscale = 1.0 / (kd as f32).sqrt();
        let mut s = state0.clone();
        let mut out_cpu = vec![0f32; nv * vd];
        for h in 0..nv {
            let khid = h % nk;
            let mut qh = q[khid * kd..khid * kd + kd].to_vec();
            let mut kh = k[khid * kd..khid * kd + kd].to_vec();
            let qn = (qh.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
            let kn = (kh.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
            for x in qh.iter_mut() {
                *x = *x / qn * qscale;
            }
            for x in kh.iter_mut() {
                *x /= kn;
            }
            let beta = sigmoid(blog[h]);
            let decay = (acoef[h] * softplus(alpha[h] + dtbias[h])).exp();
            let sb = h * kd * vd;
            for d in 0..vd {
                let mut kvv = 0.0;
                for kk in 0..kd {
                    let sv = s[sb + kk * vd + d] * decay;
                    s[sb + kk * vd + d] = sv;
                    kvv += kh[kk] * sv;
                }
                let delta = (v[h * vd + d] - kvv) * beta;
                let mut o = 0.0;
                for kk in 0..kd {
                    let sv = s[sb + kk * vd + d] + kh[kk] * delta;
                    s[sb + kk * vd + d] = sv;
                    o += qh[kk] * sv;
                }
                out_cpu[h * vd + d] = o;
            }
        }

        let (qb, kb, vb) = (dev(&be, &q), dev(&be, &k), dev(&be, &v));
        let (bb, ab) = (dev(&be, &blog), dev(&be, &alpha));
        let (acb, dtb) = (dev(&be, &acoef), dev(&be, &dtbias));
        let sbuf = dev(&be, &state0);
        let ob = be.alloc(nv * vd * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.deltanet(
            qb.as_ref(),
            kb.as_ref(),
            vb.as_ref(),
            bb.as_ref(),
            ab.as_ref(),
            acb.as_ref(),
            dtb.as_ref(),
            sbuf.as_ref(),
            ob.as_ref(),
            1, // rows: single-token bespoke path
            nv,
            nk,
            kd,
            vd,
            eps,
        );
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), nv * vd);
        let s_gpu = read(&be, sbuf.as_ref(), nv * kd * vd);
        assert!(
            maxerr(&out_cpu, &out_gpu) < 1e-4,
            "deltanet out err {}",
            maxerr(&out_cpu, &out_gpu)
        );
        assert!(
            maxerr(&s, &s_gpu) < 1e-4,
            "deltanet state err {}",
            maxerr(&s, &s_gpu)
        );
    }

    #[test]
    fn conv1d_silu_matches_cpu() {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        let (cc, kconv) = (40usize, 4usize);
        let qkv = det(cc, 0.2);
        let w = det(cc * kconv, 1.1);
        let state0 = det((kconv - 1) * cc, 0.3);

        // CPU reference (mirrors shaders/conv1d_silu.comp).
        let mut st = state0.clone();
        let mut out_cpu = vec![0f32; cc];
        let km1 = kconv - 1;
        for ch in 0..cc {
            let mut acc = 0.0;
            for k in 0..km1 {
                acc += st[k * cc + ch] * w[ch * kconv + k];
            }
            acc += qkv[ch] * w[ch * kconv + km1];
            out_cpu[ch] = acc * sigmoid(acc);
            for k in 0..km1 - 1 {
                st[k * cc + ch] = st[(k + 1) * cc + ch];
            }
            st[(km1 - 1) * cc + ch] = qkv[ch];
        }

        let xb = dev(&be, &qkv);
        // `rec.conv1d_silu` resolves the weight's own BDA device address (resident-BDA — see that
        // fn's doc), so `wb` must be a real `BufferUsage::Weights` allocation, not a plain
        // activation buffer (which has no `device_addr()`).
        let wb = be.upload_weight(&w).unwrap();
        let sbuf = dev(&be, &state0);
        let ob = be.alloc(cc * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.conv1d_silu(
            xb.as_ref(),
            wb.as_ref(),
            sbuf.as_ref(),
            ob.as_ref(),
            1, // rows: single-token bespoke path
            cc,
            kconv,
        );
        rec.finish().unwrap();
        let out_gpu = read(&be, ob.as_ref(), cc);
        let s_gpu = read(&be, sbuf.as_ref(), (kconv - 1) * cc);
        assert!(
            maxerr(&out_cpu, &out_gpu) < 1e-5,
            "conv out err {}",
            maxerr(&out_cpu, &out_gpu)
        );
        assert!(
            maxerr(&st, &s_gpu) < 1e-5,
            "conv state err {}",
            maxerr(&st, &s_gpu)
        );
    }
}
