//! The ROCm/HIP backend — mirrors `infr-metal`'s structure: backend struct, buffer,
//! and the `Backend` trait impl.
//!
//! Compiled only when `cfg(all(target_os = "linux", feature = "rocm"))`.

use crate::exec;
use crate::ffi::{self, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};
use crate::kernels::Pipelines;
use infr_core::backend::{
    Backend, Bindings, Buffer, BufferUsage, Capabilities, GraphPlan, Plan, ProgressScope,
    COOPMAT_TILE_16,
};
use infr_core::budget::{mib_bytes, reserve_bytes, spill_report_line, SpillNouns, SpillTally};
use infr_core::config::Config;
use infr_core::error::Result;
use infr_core::graph::Graph;
use std::ffi::{c_int, c_void};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

/// Terse local shorthand for the shared backend-error constructor.
use infr_core::error::backend as be;

/// VRAM headroom (bytes) kept free when placing KV in VRAM under `INFR_KV_OVERFLOW`: the spill
/// decision reserves this much so the per-forward activation scratch (the pooled GEMV/attention/FFN
/// buffers, whose peak scales with the prefill ubatch), the weight-dequant cache, and the rocBLAS
/// workspace all still have room after the KV cache lands. Once `free - reserve < bytes` the buffer
/// (and every later one, as the budget only shrinks) spills to host RAM; a VRAM `hipMalloc` failure
/// ALSO spills, so the reserve just moves the spill *before* the card is bone dry — otherwise those
/// scratch allocations OOM mid-forward (the exec pool panics, it is infallible by contract).
///
/// Default: 12% of total VRAM, floored at 2 GiB — enough for a several-hundred-row prefill on a
/// large-vocab model. `INFR_KV_OVERFLOW_RESERVE_MB` overrides it (raise it if a big prefill ubatch
/// still OOMs the scratch pool; lower it to keep more KV resident when the model's scratch is small).
fn kv_overflow_vram_reserve(cfg: &Config, total_vram: u64) -> u64 {
    reserve_bytes(total_vram, cfg.kv.overflow_reserve_mb)
}

/// `INFR_KV_OVERFLOW=1` opt-in: spill the KV cache to host RAM (read by attention over PCIe) when
/// it would not fit VRAM. Empty / `0` = off (unchanged VRAM-only KV). Mirrors the Vulkan backend's
/// own local `kv_overflow_enabled`.
fn kv_overflow_enabled(cfg: &Config) -> bool {
    cfg.kv.overflow
}

/// Nouns for the KV placement banner (see [`spill_report_line`], which owns the skeleton every
/// spill class shares). ROCm's host allocations are `hipHostMalloc`'d page-locked pages, which the
/// wording names because the pinning is what makes the PCIe read possible.
const KV_SPILL: SpillNouns<'static> = SpillNouns {
    env: "INFR_KV_OVERFLOW",
    noun: "KV buffers",
    resident_note: "no PCIe KV reads.",
    spill_note: "page-locked SYSTEM RAM — attention reads those K/V over PCIe (PCIe-bound on the \
                 spilled layers). Spilled KV is exempt from VRAM.",
};

/// Nouns for the dense-weight placement banner — the same skeleton, a different class.
const WEIGHT_SPILL: SpillNouns<'static> = SpillNouns {
    env: "INFR_ROCM_WEIGHT_OVERFLOW",
    noun: "weight banks",
    resident_note: "no PCIe weight reads.",
    spill_note: "page-locked SYSTEM RAM — the native Linear/EmbedGather GEMV reads those over \
                 PCIe (PCIe-bound on the spilled banks). Spilled weights are exempt from VRAM.",
};

/// Diagnostic cumulative cap (MiB) on KV-in-VRAM bytes before spilling the rest to host:
/// `INFR_KV_OVERFLOW_VRAM_MB`. Unset ⇒ no cap (VRAM-first up to the real headroom). At 0 it forces
/// the whole KV cache to host — makes the spill path reproducible on models that would otherwise
/// fit. Ignored when `INFR_KV_OVERFLOW` is off. Mirrors the Vulkan backend's `kv_overflow_vram_cap`.
fn kv_overflow_vram_cap(cfg: &Config) -> Option<u64> {
    mib_bytes(cfg.kv.overflow_vram_mb)
}

/// `INFR_ROCM_WEIGHT_OVERFLOW=1` opt-in (Slice 35): spill dense weight banks to page-locked,
/// device-mapped HOST RAM (read by the native Linear/EmbedGather GEMV over PCIe) when they would not
/// fit VRAM. Empty / `0` = off (unchanged VRAM-only weights). The ROCm twin of the Vulkan
/// `dense_paged` capability, but via the same zero-copy host-visible trick as the KV path — no
/// prefetch ring, no per-slot weight offsets, no kernel changes: a covered-format (Q4_K/Q6_K/Q8_0/
/// Q5_0) weight in host RAM is decoded in place exactly like a resident one. The uncovered formats
/// that dequant→f16 into a VRAM cache do NOT benefit (the f16 copy re-lands in VRAM), so spilling
/// them saves nothing — this path is meaningful for the native-decode formats.
fn weight_overflow_enabled(cfg: &Config) -> bool {
    cfg.paging.rocm_weight_overflow
}

/// Cumulative cap (MiB) on weight-in-VRAM bytes before spilling the rest to host under
/// `INFR_ROCM_WEIGHT_OVERFLOW`: `INFR_ROCM_WEIGHT_VRAM_MB`. Unset ⇒ no cap (VRAM-first up to the
/// real headroom minus the reserve). At 0 it forces the WHOLE weight set to host — makes the spill
/// path reproducible on a model that would otherwise fit (the sanctioned way to demonstrate the
/// capability on a card big enough for the weights). Ignored when the flag is off. Twin of
/// `INFR_KV_OVERFLOW_VRAM_MB`.
fn weight_overflow_vram_cap(cfg: &Config) -> Option<u64> {
    mib_bytes(cfg.paging.rocm_weight_vram_mb)
}

/// VRAM headroom (bytes) kept free when placing weights in VRAM under `INFR_ROCM_WEIGHT_OVERFLOW`.
/// Weights load FIRST (before the KV cache and before any per-forward activation scratch), so this
/// reserve is what leaves room for those later consumers: the KV cache (which may itself spill under
/// `INFR_KV_OVERFLOW`, but wants VRAM first), the paged-MoE expert arena, and the pooled
/// GEMV/attention/FFN scratch whose peak scales with the prefill ubatch + the token_embd/uncovered
/// dequant→f16 cache. Once `free - reserve < bytes` the bank (and every later one, as the budget
/// only shrinks) spills to host RAM. Default: 12% of total VRAM floored at 2 GiB, same as the KV
/// reserve; `INFR_ROCM_WEIGHT_OVERFLOW_RESERVE_MB` overrides it (raise it to keep more headroom for
/// a big KV context / MoE arena, lower it to keep more weights resident on a weight-dominated dense
/// model).
fn weight_overflow_vram_reserve(cfg: &Config, total_vram: u64) -> u64 {
    reserve_bytes(total_vram, cfg.paging.rocm_weight_reserve_mb)
}

/// Human-readable byte count for the KV-overflow placement banner.
fn fmt_bytes(n: u64) -> String {
    const U: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

// ── RocmBuffer ───────────────────────────────────────────────────────────────

/// A device buffer. Normally `hipMalloc`'d VRAM; for the `INFR_KV_OVERFLOW` spill path it is a
/// `hipHostMalloc`'d, page-locked, device-mapped HOST allocation (`host_ptr` set), whose device
/// alias (`ptr`, from `hipHostGetDevicePointer`) the KV kernels bind and read/write over PCIe.
pub struct RocmBuffer {
    /// Device pointer (null if len == 0). For a host-spilled KV buffer this is the device alias of
    /// `host_ptr`, so `WriteKv`/`Attention` bind it exactly like a VRAM pointer.
    pub(crate) ptr: *mut c_void,
    /// Byte length.
    pub(crate) len: usize,
    /// Whether `drop` should call `hipFree` (false for a slice/view into another buffer, and for a
    /// host-spilled buffer which is freed via `hipHostFree(host_ptr)` instead).
    pub(crate) owned: bool,
    /// Non-null only for a `hipHostMalloc`'d spilled buffer (KV cache under `INFR_KV_OVERFLOW`, or a
    /// dense weight bank under `INFR_ROCM_WEIGHT_OVERFLOW`): the HOST pointer, freed with
    /// `hipHostFree` in `drop` (which takes the host pointer, not the device alias in `ptr`).
    pub(crate) host_ptr: *mut c_void,
}

// Raw device pointers are Send/Sync (they identify a VRAM region, not a CPU address).
unsafe impl Send for RocmBuffer {}
unsafe impl Sync for RocmBuffer {}

impl RocmBuffer {
    /// Allocate `bytes` of **zero-initialized** device memory (calloc contract), returning
    /// `Err` if `hipMalloc` (OOM) or the `hipMemset` zero-fill fails — both are recoverable,
    /// never a panic. A failed `hipMemset` MUST error: silently handing back uninitialized VRAM
    /// breaks the calloc contract (`infr_core::backend::Backend::alloc`) and yields the classic
    /// CPU-works/GPU-garbage trap.
    pub fn try_alloc(bytes: usize, _stream: ffi::hipStream_t) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if bytes > 0 {
            let rc = unsafe { ffi::hipMalloc(&mut ptr, bytes) };
            if rc != HIP_SUCCESS {
                return Err(be(format!("hipMalloc({bytes}): rc={rc}")));
            }
            // Zero-init (calloc contract) — a failed memset is fatal, not ignorable.
            let rc = unsafe { ffi::hipMemset(ptr, 0, bytes) };
            if rc != HIP_SUCCESS {
                unsafe { ffi::hipFree(ptr) };
                return Err(be(format!("hipMemset({bytes}): rc={rc}")));
            }
        }
        Ok(Self {
            ptr,
            len: bytes,
            owned: true,
            host_ptr: std::ptr::null_mut(),
        })
    }

    /// Allocate device memory WITHOUT zero-init, returning `Err` on `hipMalloc` failure (OOM).
    /// Only for buffers whose full extent is written before any read (e.g. weights uploaded
    /// immediately).
    pub fn try_alloc_uninit(bytes: usize, _stream: ffi::hipStream_t) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if bytes > 0 {
            let rc = unsafe { ffi::hipMalloc(&mut ptr, bytes) };
            if rc != HIP_SUCCESS {
                return Err(be(format!("hipMalloc({bytes}): rc={rc}")));
            }
        }
        Ok(Self {
            ptr,
            len: bytes,
            owned: true,
            host_ptr: std::ptr::null_mut(),
        })
    }

    /// Allocate one buffer in HOST memory (page-locked, device-mapped) — the shared spill path for
    /// both `INFR_KV_OVERFLOW` (KV cache, Slice 34) and `INFR_ROCM_WEIGHT_OVERFLOW` (dense weight
    /// banks, Slice 35). Returns `Err` on `hipHostMalloc` failure so overflow mode degrades
    /// gracefully rather than aborting. The device alias (`ptr`, from `hipHostGetDevicePointer`) is
    /// what the WriteKv/Attention or native Linear/EmbedGather GEMV kernels bind; the device
    /// reads/writes it directly over PCIe with NO explicit per-token copy and NO kernel changes.
    ///
    /// `zero_init` honors the calloc contract for buffers read before written (KV padding rows):
    /// a plain host `write_bytes` — CPU-addressable, no device memset/sync. Weight banks are
    /// overwritten in full by the immediate `upload`, so they pass `false` and skip the memset of a
    /// multi-GiB region.
    pub fn try_alloc_host(bytes: usize, zero_init: bool) -> Result<Self> {
        let mut host_ptr: *mut c_void = std::ptr::null_mut();
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        if bytes > 0 {
            let rc = unsafe {
                ffi::hipHostMalloc(
                    &mut host_ptr,
                    bytes,
                    ffi::HIP_HOST_MALLOC_PORTABLE | ffi::HIP_HOST_MALLOC_MAPPED,
                )
            };
            if rc != HIP_SUCCESS {
                return Err(be(format!("hipHostMalloc({bytes}): rc={rc}")));
            }
            if zero_init {
                // Zero-init on the host side (calloc contract) — CPU-addressable, no device sync.
                unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, bytes) };
            }
            let rc = unsafe { ffi::hipHostGetDevicePointer(&mut dev_ptr, host_ptr, 0) };
            if rc != HIP_SUCCESS {
                unsafe { ffi::hipHostFree(host_ptr) };
                return Err(be(format!("hipHostGetDevicePointer: rc={rc}")));
            }
        }
        Ok(Self {
            ptr: dev_ptr,
            len: bytes,
            owned: true,
            host_ptr,
        })
    }

    /// Zero-initialized device memory, panicking on failure. Convenience for the exec-internal
    /// scratch path (activations/intermediates) where a fallible signature would ripple through
    /// every op; the recoverable trait-level entry points use [`try_alloc`](Self::try_alloc).
    pub fn alloc(bytes: usize, stream: ffi::hipStream_t) -> Self {
        Self::try_alloc(bytes, stream).expect("hipMalloc/hipMemset (exec scratch)")
    }

    /// Alias for [`alloc`](Self::alloc) — zero-initialized device memory (calloc contract).
    pub fn alloc_zero(bytes: usize, stream: ffi::hipStream_t) -> Self {
        Self::alloc(bytes, stream)
    }

    /// Upload host bytes to this device buffer.
    pub fn upload(&mut self, src: &[u8], _stream: ffi::hipStream_t) {
        if src.is_empty() || self.ptr.is_null() {
            return;
        }
        let n = src.len().min(self.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                n,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if rc != HIP_SUCCESS {
            panic!("hipMemcpy H2D: rc={rc}");
        }
    }

    /// Download device bytes to host.
    // `stream` is an opaque HIP handle passed straight to the driver, not a Rust-dereferenced
    // pointer — the not_unsafe_ptr_arg_deref lint doesn't apply to a handle-passing helper.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn download(&self, dst: &mut [u8], stream: ffi::hipStream_t) {
        if dst.is_empty() || self.ptr.is_null() {
            return;
        }
        let n = dst.len().min(self.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                n,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != HIP_SUCCESS {
            panic!("hipMemcpy D2H: rc={rc}");
        }
        // Wait for the copy to finish
        unsafe { ffi::hipStreamSynchronize(stream) };
    }
}

impl Buffer for RocmBuffer {
    fn len_bytes(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ── BufferPool ───────────────────────────────────────────────────────────────

/// Round a byte request up to its pool bucket. Distinct-but-close sizes share a bucket so a
/// prefill (m=N) row and a decode (m=1) row of the same op don't fragment the free-list too
/// finely; 256 B granularity keeps waste ≤ 256 B/alloc while the same graph replayed every
/// decode step maps to the exact same buckets → perfect reuse, zero churn.
pub(crate) fn bucket_bytes(bytes: usize) -> usize {
    const GRAN: usize = 256;
    bytes.max(1).div_ceil(GRAN) * GRAN
}

/// A free-list of reusable device scratch allocations, keyed by bucket byte size. Op scratch
/// (`zero_dev` / transient GEMV buffers) is drawn from here and returned at end-of-forward
/// instead of `hipMalloc`/`hipFree`'d per op — on a blocking stream each malloc/free implicitly
/// syncs the device, so the per-op allocation churn (not the explicit sync) was the decode
/// bottleneck. The pool lives on the backend, so it persists across decode replay steps and the
/// hot loop allocates nothing after the first pass.
pub(crate) struct BufferPool {
    free: std::collections::HashMap<usize, Vec<*mut c_void>>,
}

// The pool holds raw device pointers (VRAM regions, not CPU addresses) — Send/Sync like RocmBuffer.
unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}

impl BufferPool {
    pub(crate) fn new() -> Self {
        Self {
            free: std::collections::HashMap::new(),
        }
    }

    /// Get a device pointer for `bucket` bytes (already rounded via [`bucket_bytes`]): reuse a
    /// free one if present, else `hipMalloc` a fresh bucket-sized allocation. Panics on OOM — the
    /// exec-internal scratch path, like [`RocmBuffer::alloc`], is infallible by contract.
    pub(crate) fn take(&mut self, bucket: usize) -> *mut c_void {
        if let Some(v) = self.free.get_mut(&bucket) {
            if let Some(p) = v.pop() {
                return p;
            }
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { ffi::hipMalloc(&mut ptr, bucket) };
        if rc != HIP_SUCCESS {
            panic!("BufferPool hipMalloc({bucket}): rc={rc}");
        }
        ptr
    }

    /// Return a pointer to its bucket free-list for reuse by the next op / forward pass.
    pub(crate) fn give(&mut self, bucket: usize, ptr: *mut c_void) {
        self.free.entry(bucket).or_default().push(ptr);
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        for (_, v) in self.free.drain() {
            for p in v {
                if !p.is_null() {
                    unsafe { ffi::hipFree(p) };
                }
            }
        }
    }
}

impl Drop for RocmBuffer {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        // A host-spilled KV buffer (KV overflow): free the HOST allocation via `hipHostFree`; its
        // device alias in `ptr` is NOT separately `hipFree`d (it is not an independent allocation).
        if !self.host_ptr.is_null() {
            unsafe { ffi::hipHostFree(self.host_ptr) };
        } else if !self.ptr.is_null() {
            unsafe { ffi::hipFree(self.ptr) };
        }
    }
}

// ── RocmBackend ──────────────────────────────────────────────────────────────

/// The ROCm/HIP compute backend.
pub struct RocmBackend {
    /// Active device index.
    device: c_int,
    /// Non-blocking stream for all work.
    stream: ffi::hipStream_t,
    /// Compiled kernel module + function cache.
    pipelines: Pipelines,
    /// Dequantized-weight cache: (bound-buffer device address, byte length) → f16 device buffer.
    /// Single-generation lifetime (one backend per generation); keys are stable. The byte length
    /// is part of the key so a freed weight buffer whose device address is later RECYCLED for a
    /// DIFFERENTLY-SIZED weight cannot collide (address alone aliases the stale dequant — the
    /// classic "wrong scalar for a later head" corruption when two DeltaNet shapes share a backend).
    pub(crate) weight_cache: Mutex<std::collections::HashMap<(usize, usize), RocmBuffer>>,
    /// Reusable op-scratch pool (see [`BufferPool`]). Persists across `execute` calls so the
    /// decode replay loop draws from the free-list instead of `hipMalloc`/`hipFree` per op.
    pub(crate) pool: Mutex<BufferPool>,
    /// rocBLAS handle for the OPT-IN Slice-26 f16 prefill GEMM (`INFR_ROCM_BLAS=1`). `null` by
    /// default (and if `rocblas_create_handle` fails), so the prefill path uses the int8 WMMA kernel.
    rocblas: ffi::rocblas_handle,
    /// Active weight-load progress bar.
    weight_pb: Arc<Mutex<Option<indicatif::ProgressBar>>>,
    /// Paged MoE expert cache (Slice 33 — see `crate::pager`). `Some` only when the loaded model's
    /// expert banks are paged (too big to keep resident, or forced via `INFR_ROCM_EXPERT_BUDGET`);
    /// `None` (the common case) means every expert is resident, zero change. `Backend::moe_paged`
    /// reads this.
    pub(crate) moe_pager: Mutex<Option<crate::pager::RocmMoePager>>,
    /// Dense-weight prefetch ring (Slice 37 — see `crate::weight_pager`). Lazily built on the first
    /// `execute` that sees a spilled-native dense Linear bank under `INFR_ROCM_WEIGHT_OVERFLOW`;
    /// `None` (the common case) means no dense bank is spilled, zero change. Streams the NEXT
    /// spilled bank into a VRAM staging slot on a copy stream while the current layer computes, so
    /// the Linear GEMV reads a resident slot instead of the bank over PCIe.
    pub(crate) weight_ring: Mutex<Option<crate::weight_pager::RocmWeightRing>>,
    /// VRAM-first KV-overflow placement tally (`INFR_KV_OVERFLOW`): how many `BufferUsage::KvCache`
    /// buffers (and how many bytes) landed in device-local VRAM vs spilled to host RAM. Fed by the
    /// `alloc` KvCache branch, drained once by `kv_overflow_report`. Zero unless the flag is on.
    /// The bookkeeping (and the cumulative-cap gate) is the shared [`SpillTally`].
    kv_spill: SpillTally,
    /// VRAM-first WEIGHT-overflow placement tally (`INFR_ROCM_WEIGHT_OVERFLOW`, Slice 35): how many
    /// `BufferUsage::Weights`/`HostWeights` banks (and how many bytes) landed in device-local VRAM
    /// vs spilled to page-locked host RAM. Fed by the `alloc`/`alloc_uninit` weight branch, drained
    /// once by `weight_overflow_report` (printed on the first `execute`). Zero unless the flag is on.
    wt_spill: SpillTally,
    /// One-shot latch so the weight-overflow banner prints exactly once, on the first `execute`
    /// (after the whole weight-load walk has run). `false` until printed.
    wt_reported: std::sync::atomic::AtomicBool,
    /// The engine configuration this backend reads its knobs from — one value, HANDED IN by the
    /// caller ([`RocmBackend::new_with`]), held for the backend's whole life, and borrowed (never
    /// cloned) at every read site including the per-forward `execute_graph` walk
    /// (`docs/config-plan.md` R4/R6). S6 replaced S2's `Config::load_from_env()` bridge with it.
    cfg: Arc<Config>,
}

// The backend owns streams and device handles which are Send/Sync.
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Borrowed engine configuration — every knob this backend steers on. A REFERENCE, never a
    /// clone: `execute_graph` reads it per forward and the kernel-tier helpers read it per op
    /// (`docs/config-plan.md` R6).
    ///
    /// `pub` so `infr-llama`'s seam and this crate's own probes can read the knobs off the backend
    /// they already hold instead of growing a second env-sourced config.
    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// `Default` < environment, for the [`new`](Self::new) entry point that is handed no
    /// [`Config`] — this crate's own tests/examples and external library callers. Fallible for the
    /// same reason `VulkanBackend::cfg_from_env` is (S5a): the five LOUD keys (`INFR_SG`,
    /// `INFR_SUBMIT_DISPATCHES`, the three device lists) are `Config`-sourced now, so swallowing a
    /// layer error would silently drop a rejection.
    fn cfg_from_env() -> Result<Arc<Config>> {
        let layer = infr_core::config::ConfigLayer::env().map_err(|e| be(e.to_string()))?;
        Ok(Arc::new(Config::load_from_layers(&[layer])))
    }

    /// Create a ROCm backend on `device_id`, resolving `Default` < environment for itself. Every
    /// caller inside `infr-llama` passes its own `Arc<Config>` to
    /// [`new_with`](Self::new_with) instead.
    pub fn new(device_id: c_int) -> Result<Self> {
        Self::new_with(device_id, Self::cfg_from_env()?)
    }

    /// **The real constructor (S6).** Build a backend on `device_id`, reading every knob — the
    /// rocBLAS opt-in, the kernel tiers, the pager/prefetch diagnostics, the KV/weight overflow
    /// budgets — from the `cfg` the caller hands in rather than the process environment.
    pub fn new_with(device_id: c_int, cfg: Arc<Config>) -> Result<Self> {
        let mut count: c_int = 0;
        let rc = unsafe { ffi::hipGetDeviceCount(&mut count) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipGetDeviceCount: rc={rc}")));
        }
        if count == 0 {
            return Err(be("no HIP-capable devices found"));
        }
        if device_id >= count {
            return Err(be(format!(
                "HIP device {device_id} out of range (count={count})"
            )));
        }

        let device: c_int = device_id;
        let rc = unsafe { ffi::hipSetDevice(device) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipSetDevice({device}): rc={rc}")));
        }

        let mut stream: ffi::hipStream_t = std::ptr::null_mut();
        let rc = unsafe { ffi::hipStreamCreate(&mut stream) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipStreamCreate: rc={rc}")));
        }

        let pipelines = Pipelines::build(device)?;

        // rocBLAS handle for the OPT-IN f16 prefill GEMM (Slice 26), bound once to our work stream.
        // OFF by default: the isolated GEMM wins 3.6-5.9× (examples/blas_probe), but the per-forward
        // dequant→f16 tax makes it a NET LOSS end-to-end (~0.88× pp512 on 0.6B-8B) AND its transient
        // f16 pool buffers reintroduce the Phase-3 VRAM blowup (OOM on 8B). So the default prefill
        // path stays on the hand int8 WMMA kernel; `INFR_ROCM_BLAS=1` opts into the library GEMM for
        // experimentation. A create failure is non-fatal — the handle stays null and WMMA is used.
        let mut rocblas: ffi::rocblas_handle = std::ptr::null_mut();
        if cfg.kernels.rocm.blas {
            let rc = unsafe { ffi::rocblas_create_handle(&mut rocblas) };
            if rc == ffi::ROCBLAS_STATUS_SUCCESS {
                unsafe { ffi::rocblas_set_stream(rocblas, stream) };
            } else {
                rocblas = std::ptr::null_mut();
            }
        }

        Ok(Self {
            device,
            stream,
            pipelines,
            weight_cache: Mutex::new(std::collections::HashMap::new()),
            pool: Mutex::new(BufferPool::new()),
            rocblas,
            weight_pb: Arc::new(Mutex::new(None)),
            moe_pager: Mutex::new(None),
            weight_ring: Mutex::new(None),
            kv_spill: SpillTally::default(),
            wt_spill: SpillTally::default(),
            wt_reported: std::sync::atomic::AtomicBool::new(false),
            cfg,
        })
    }

    /// Allocate one KV-cache buffer under `INFR_KV_OVERFLOW`, VRAM-first: keep it resident in VRAM
    /// while the live free budget (minus [`KV_OVERFLOW_VRAM_RESERVE`], and under the diagnostic
    /// `INFR_KV_OVERFLOW_VRAM_MB` cumulative cap) still fits it; otherwise — and for every later
    /// buffer, since the budget only shrinks — place it in page-locked, device-mapped HOST RAM read
    /// by attention over PCIe. A VRAM `hipMalloc` failure at the exact budget edge ALSO spills
    /// rather than propagating: overflow mode degrades to host, never hard-errors. Bumps the
    /// resident/spilled tally for the one-shot `kv_overflow_report` banner.
    fn alloc_kv_overflow(&self, bytes: usize) -> Result<Box<dyn Buffer>> {
        let cap_ok = self
            .kv_spill
            .admits(kv_overflow_vram_cap(&self.cfg), bytes as u64);
        let (free, total) = self.vram_info();
        let reserve = kv_overflow_vram_reserve(&self.cfg, total as u64);
        let budget_ok = (free as u64) >= bytes as u64 + reserve;
        if cap_ok && budget_ok {
            if let Ok(buf) = RocmBuffer::try_alloc(bytes, self.stream) {
                self.kv_spill.record_vram(bytes as u64);
                return Ok(Box::new(buf));
            }
        }
        let buf = RocmBuffer::try_alloc_host(bytes, true)?;
        self.kv_spill.record_host(bytes as u64);
        Ok(Box::new(buf))
    }

    /// Allocate one dense weight bank under `INFR_ROCM_WEIGHT_OVERFLOW`, VRAM-first: keep it
    /// resident in device-local VRAM while the live free budget (minus
    /// [`weight_overflow_vram_reserve`], and under the diagnostic `INFR_ROCM_WEIGHT_VRAM_MB`
    /// cumulative cap) still fits it; otherwise — and for every later bank, since the budget only
    /// shrinks — place it in page-locked, device-mapped HOST RAM read by the native Linear/
    /// EmbedGather GEMV over PCIe. A VRAM `hipMalloc` failure at the budget edge ALSO spills rather
    /// than propagating: overflow mode degrades to host, never hard-errors (mirrors the KV path).
    /// Weight banks are overwritten in full by the immediate `upload`, so VRAM placement uses the
    /// UNINIT alloc and the host placement skips the calloc memset. Bumps the resident/spilled tally
    /// for the one-shot `weight_overflow_report` banner.
    fn alloc_weight_overflow(&self, bytes: usize) -> Result<Box<dyn Buffer>> {
        let cap_ok = self
            .wt_spill
            .admits(weight_overflow_vram_cap(&self.cfg), bytes as u64);
        let (free, total) = self.vram_info();
        let reserve = weight_overflow_vram_reserve(&self.cfg, total as u64);
        let budget_ok = (free as u64) >= bytes as u64 + reserve;
        if cap_ok && budget_ok {
            if let Ok(buf) = RocmBuffer::try_alloc_uninit(bytes, self.stream) {
                self.wt_spill.record_vram(bytes as u64);
                return Ok(Box::new(buf));
            }
        }
        let buf = RocmBuffer::try_alloc_host(bytes, false)?;
        self.wt_spill.record_host(bytes as u64);
        Ok(Box::new(buf))
    }

    /// One-shot weight-placement summary for the `INFR_ROCM_WEIGHT_OVERFLOW` VRAM-first spill
    /// (mirrors the KV banner): how many weight banks stayed resident in VRAM vs spilled to host
    /// RAM. Printed lazily on the FIRST `execute`, by which point the whole weight-load walk has
    /// run. No-op with the flag off (nothing was tallied) so normal runs print nothing.
    fn weight_overflow_report(&self) {
        if !weight_overflow_enabled(&self.cfg) {
            return;
        }
        if self.wt_reported.swap(true, Ordering::Relaxed) {
            return;
        }
        if let Some(line) = spill_report_line(self.wt_spill.counts(), &WEIGHT_SPILL, fmt_bytes) {
            eprintln!("{line}");
        }
    }

    /// Read a device property field.
    fn prop(&self) -> ffi::hipDeviceProp_t {
        let mut props: ffi::hipDeviceProp_t = unsafe { std::mem::zeroed() };
        unsafe { ffi::hipGetDeviceProperties(&mut props, self.device) };
        props
    }

    /// `(free, total)` device memory in bytes — the paged-MoE budget input and the peak-VRAM
    /// report. Returns `(0, 0)` if the query fails (a caller treats free==0 as "unknown").
    pub fn vram_info(&self) -> (usize, usize) {
        let (mut free, mut total) = (0usize, 0usize);
        let rc = unsafe { ffi::hipMemGetInfo(&mut free, &mut total) };
        if rc != HIP_SUCCESS {
            return (0, 0);
        }
        (free, total)
    }

    /// Install this model's paged-MoE session (see [`crate::pager::RocmMoePager`]), sized but with
    /// no tensor registered yet. Called by the seam BEFORE the weight-load walk, so
    /// `Backend::moe_paged` answers truthy and the first paged tensor's placeholder is registered
    /// as it is bound. The arenas allocate here (one contiguous VRAM buffer per pool).
    pub fn init_moe_pager(&self, layout: crate::pager::MoePagerLayout) -> Result<()> {
        let session = crate::pager::RocmMoePager::new(layout, self.stream, &self.cfg.paging)?;
        *self.moe_pager.lock().unwrap() = Some(session);
        Ok(())
    }

    /// Register one paged layer's role tensor with the session `init_moe_pager` installed. Panics
    /// if no session is installed (a caller bug: `init_moe_pager` must run first).
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

    /// `INFR_PAGER_STATS=1` hit/miss/eviction dump for the paged session (no-op when unpaged).
    pub fn print_moe_pager_stats(&self) {
        if let Some(p) = self.moe_pager.lock().unwrap().as_ref() {
            p.print_stats_if_enabled();
        }
    }
}

impl Backend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn capabilities(&self) -> Capabilities {
        let props = self.prop();
        Capabilities {
            name: "AMD ROCm/HIP".into(),
            f16: true,
            coopmat_f16: None,
            f8: false,
            coopmat_f8: None,
            // Phase 4: int8-activation dp4a decode GEMV is the default path for the covered
            // formats (Q4_K/Q6_K/Q8_0), quantizing the activation row to int8 and integer-dotting
            // (V_DOT4/`__builtin_amdgcn_sdot4`) against the native weight codes. These caps are
            // informational for the seam runner (it does not branch on them), but flipped to report
            // the backend honestly. Phase 5: prefill (m>1) runs on the RDNA3 wave32 int8 matrix
            // core (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`), so `coopmat_i8` reports the real
            // 16×16×16 tile.
            i8: true,
            i8_dot: true,
            coopmat_i8: Some(COOPMAT_TILE_16),
            bf16: false,
            coopmat_bf16: None,
            subgroup_min: 0,
            subgroup_max: 0,
            sg_pref: 0,
            vendor_intel: false,
            integrated: false,
            compute_units: props.multi_processor_count as u32,
            buffer_device_address: false,
            max_shared_memory_bytes: props.shared_mem_per_block as u32,
            unified_memory: false,
            // ── correctness-dial: start with NOTHING fused ──
            decode_replay: false,
            combined_gu: false,
            embed_gather: false,
            gpu_sample: false,
            argmax_rows: false,
            argmax_prob: false,
            gated_rmsnorm: false,
            kv_swa_ring: false,
        }
    }

    fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Opt-in KV overflow (`INFR_KV_OVERFLOW`): place KV VRAM-first, spilling the tail to host
        // RAM read by attention over PCIe. Off by default ⇒ unchanged VRAM-only KV below. Only KV
        // buffers are eligible; weights/activations always stay device-local.
        if matches!(_usage, BufferUsage::KvCache) && kv_overflow_enabled(&self.cfg) {
            return self.alloc_kv_overflow(bytes);
        }
        let is_weight = matches!(_usage, BufferUsage::Weights | BufferUsage::HostWeights);
        // Opt-in dense WEIGHT overflow (`INFR_ROCM_WEIGHT_OVERFLOW`, Slice 35): place weight banks
        // VRAM-first, spilling the tail to host RAM read by the native GEMV over PCIe. Off by
        // default ⇒ unchanged VRAM-only weights below.
        let buf = if is_weight && weight_overflow_enabled(&self.cfg) {
            let b = self.alloc_weight_overflow(bytes)?;
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
            return Ok(b);
        } else {
            // Zero-init (calloc contract); OOM or a failed zero-fill returns Err (recoverable).
            RocmBuffer::try_alloc(bytes, self.stream)?
        };
        // Advance weight progress bar for weight/host-weight allocations
        if is_weight {
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(Box::new(buf))
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        let is_weight = matches!(usage, BufferUsage::Weights | BufferUsage::HostWeights);
        if is_weight && weight_overflow_enabled(&self.cfg) {
            let b = self.alloc_weight_overflow(bytes)?;
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
            return Ok(b);
        }
        // Skip zero-init for weight buffers (they get uploaded immediately); OOM returns Err.
        let buf = RocmBuffer::try_alloc_uninit(bytes, self.stream)?;
        if is_weight {
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(Box::new(buf))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let buf = dst
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: buffer is not a RocmBuffer");
        if src.is_empty() || buf.ptr.is_null() {
            return Ok(());
        }
        let n = src.len().min(buf.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                buf.ptr,
                src.as_ptr() as *const c_void,
                n,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipMemcpy H2D: rc={rc}")));
        }
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let buf = src
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: buffer is not a RocmBuffer");
        if dst.is_empty() || buf.ptr.is_null() {
            return Ok(());
        }
        let n = dst.len().min(buf.len);
        let rc = unsafe {
            ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                buf.ptr,
                n,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipMemcpy D2H: rc={rc}")));
        }
        // Wait for download to complete
        unsafe { ffi::hipStreamSynchronize(self.stream) };
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(GraphPlan::boxed(graph))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        // One-shot weight-overflow banner: the whole weight-load walk has run by the first execute.
        self.weight_overflow_report();
        exec::execute_graph(
            &self.pipelines,
            &self.weight_cache,
            &self.pool,
            &self.moe_pager,
            &self.weight_ring,
            self.stream,
            self.rocblas,
            plan,
            bindings,
            &self.cfg,
        )
    }

    /// A paged MoE model (Slice 33 — `crate::pager`) keeps its expert banks in host memory and
    /// pages the routed experts into a VRAM slot arena. `true` only while such a session is
    /// installed; `false` for every resident model (the common case, zero change).
    fn moe_paged(&self) -> bool {
        self.moe_pager.lock().unwrap().is_some()
    }

    /// One-shot KV placement summary for the `INFR_KV_OVERFLOW` VRAM-first spill (mirrors the
    /// Vulkan backend): how many KV buffers stayed resident in VRAM vs spilled to host RAM. The
    /// runner calls this once, right after the per-layer KV allocation loop. No-op with the flag off
    /// (nothing was tallied) so normal runs print nothing.
    fn kv_overflow_report(&self) {
        if !kv_overflow_enabled(&self.cfg) {
            return;
        }
        if let Some(line) = spill_report_line(self.kv_spill.counts(), &KV_SPILL, fmt_bytes) {
            eprintln!("{line}");
        }
    }

    fn sync(&self) -> Result<()> {
        let rc = unsafe { ffi::hipStreamSynchronize(self.stream) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipStreamSynchronize: rc={rc}")));
        }
        Ok(())
    }

    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        // Bound-check BOTH ends: a `bytes > dst.len_bytes()` `hipMemcpyDtoD` is a device-side
        // out-of-bounds write (VRAM corruption), just as `bytes > src` is an OOB read.
        infr_core::backend::check_copy_bytes(bytes, src.len_bytes())?;
        infr_core::backend::check_copy_bytes(bytes, dst.len_bytes())?;
        let src_buf = src
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: src is not a RocmBuffer");
        let dst_buf = dst
            .as_any()
            .downcast_ref::<RocmBuffer>()
            .expect("rocm backend: dst is not a RocmBuffer");
        let rc = unsafe { ffi::hipMemcpyDtoD(dst_buf.ptr, src_buf.ptr, bytes) };
        if rc != HIP_SUCCESS {
            return Err(be(format!("hipMemcpyDtoD: rc={rc}")));
        }
        Ok(())
    }

    fn weight_progress(&self, total_bytes: Option<u64>) -> Box<dyn ProgressScope> {
        struct RocmProgress {
            pb: Arc<Mutex<Option<indicatif::ProgressBar>>>,
        }
        impl ProgressScope for RocmProgress {}
        impl Drop for RocmProgress {
            fn drop(&mut self) {
                if let Some(pb) = self.pb.lock().unwrap().take() {
                    pb.finish_and_clear();
                }
            }
        }
        let pb = total_bytes.map(|total| {
            let style = indicatif::ProgressStyle::with_template(
                "  {spinner} ROCm weights {bytes}/{total_bytes} [{elapsed_precise}] {msg}",
            )
            .unwrap();
            let pb = indicatif::ProgressBar::new(total);
            pb.set_style(style);
            pb
        });
        *self.weight_pb.lock().unwrap() = pb;
        Box::new(RocmProgress {
            pb: self.weight_pb.clone(),
        })
    }
}

impl Drop for RocmBackend {
    fn drop(&mut self) {
        if !self.rocblas.is_null() {
            unsafe { ffi::rocblas_destroy_handle(self.rocblas) };
        }
        if !self.stream.is_null() {
            unsafe {
                ffi::hipStreamSynchronize(self.stream);
                ffi::hipStreamDestroy(self.stream);
            }
        }
    }
}
