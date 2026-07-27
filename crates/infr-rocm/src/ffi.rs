//! HIP FFI — hand-rolled `extern "C"` bindings to `libamdhip64` and `libhiprtc`.
//!
//! Compiled only when `cfg(all(target_os = "linux", feature = "rocm"))`. Every function
//! returns its natural error code; the caller checks against the success constant (0).

// The type aliases deliberately keep HIP's C spelling (`hipStream_t`, `hipMemcpyKind`, …) so
// this module reads 1:1 against the HIP headers; and the binding surface intentionally declares
// the full set of entry points / constants even though later phases consume some of them.
#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_int, c_void};

// ── libamdhip64 ──────────────────────────────────────────────────────────────

#[link(name = "amdhip64")]
extern "C" {
    /// Number of HIP-capable devices on this node.
    pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    /// Select the active device.
    pub fn hipSetDevice(device: c_int) -> c_int;
    /// Query device properties into `props` (allocated by caller). Typed as the LEGACY
    /// [`hipDeviceProp_tR0000`] because that is the layout the linked symbol actually fills — see
    /// the note beside that struct; declaring the current `R0600` layout here reads garbage.
    pub fn hipGetDeviceProperties(props: *mut hipDeviceProp_tR0000, device: c_int) -> c_int;
    /// Allocate `size` bytes of device memory.
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    /// Free device memory.
    pub fn hipFree(ptr: *mut c_void) -> c_int;
    /// Copy `count` bytes between host and device (direction `kind`).
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
    ) -> c_int;
    /// Fill `count` bytes of device memory with `value`.
    pub fn hipMemset(dst: *mut c_void, value: c_int, count: usize) -> c_int;
    /// Fill `count` bytes of device memory with `value`, asynchronously on `stream`
    /// (no implicit device sync — the buffer-pool zero-on-reuse primitive).
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        count: usize,
        stream: hipStream_t,
    ) -> c_int;
    /// Copy `count` bytes between host and device (direction `kind`), asynchronously on `stream`.
    /// The paged-MoE slot fill (`pager::RocmMoePager::ensure_slot`): stream-ordered so the copy
    /// completes before the expert GEMV enqueued after it on the same stream reads the slot.
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> c_int;
    /// Free and total device memory in bytes (the paged-MoE budget + peak-VRAM report).
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    /// Allocate `size` bytes of HOST memory that is page-locked and (with `HIP_HOST_MALLOC_MAPPED`)
    /// mapped into the device address space — the `INFR_KV_OVERFLOW` spill path. The device reads
    /// this memory over PCIe through the device pointer from `hipHostGetDevicePointer`; no explicit
    /// per-token copy is needed. `flags` = OR of the `HIP_HOST_MALLOC_*` bits below.
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    /// Return the DEVICE-side pointer that aliases a `hipHostMalloc(..MAPPED)` host allocation, so a
    /// kernel binding this pointer reads/writes the host buffer directly over PCIe. `flags` is 0.
    pub fn hipHostGetDevicePointer(
        dev_ptr: *mut *mut c_void,
        host_ptr: *mut c_void,
        flags: u32,
    ) -> c_int;
    /// Free a `hipHostMalloc` allocation (takes the HOST pointer, not the device alias).
    pub fn hipHostFree(ptr: *mut c_void) -> c_int;
    /// Create a non-blocking stream.
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> c_int;
    /// Block until all work on `stream` finishes.
    pub fn hipStreamSynchronize(stream: hipStream_t) -> c_int;
    /// Destroy a stream.
    pub fn hipStreamDestroy(stream: hipStream_t) -> c_int;
    /// Create an event with `flags` (see `HIP_EVENT_*`). The paged-MoE pager records one on its
    /// copy stream after each async slot fill, then makes the compute stream wait on it so an
    /// expert GEMV overlaps the NEXT expert's page-in copy.
    pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: u32) -> c_int;
    /// Record `event` in `stream` (captures the stream's progress at this point).
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> c_int;
    /// Make `stream` wait until `event` has been reached (cross-stream ordering, no host sync).
    pub fn hipStreamWaitEvent(stream: hipStream_t, event: hipEvent_t, flags: u32) -> c_int;
    /// Destroy an event.
    pub fn hipEventDestroy(event: hipEvent_t) -> c_int;
    /// Load a code object (PTX-alike, from hiprtc or hipcc) into a module.
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> c_int;
    /// Get a kernel function from a module by name.
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> c_int;
    /// Launch a kernel with the given grid/block dimensions, shared-mem bytes, and args.
    #[allow(improper_ctypes)]
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    /// Block until all work on the device finishes.
    pub fn hipDeviceSynchronize() -> c_int;
    /// Copy from one device buffer to another.
    pub fn hipMemcpyDtoD(dst: *mut c_void, src: *const c_void, count: usize) -> c_int;
    /// HIP runtime version (`major*10000000 + minor*100000 + patch`). Folded into the module
    /// cache's key: a runtime upgrade re-generates code objects, and an old one must not be
    /// re-loaded (see `kernels.rs`).
    pub fn hipRuntimeGetVersion(version: *mut c_int) -> c_int;
    /// HIP driver version — same role in the key. The runtime and the driver ship separately, and
    /// either moving is enough to invalidate a cached code object.
    pub fn hipDriverGetVersion(version: *mut c_int) -> c_int;
}

// ── libhiprtc ────────────────────────────────────────────────────────────────

#[link(name = "hiprtc")]
extern "C" {
    /// Create a program from `src` (null-terminated) with optional `name`.
    pub fn hiprtcCreateProgram(
        prog: *mut hiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> c_int;
    /// Compile the program with the given options.
    pub fn hiprtcCompileProgram(
        prog: hiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> c_int;
    /// Get the compiled code object (binary, NOT null-terminated).
    pub fn hiprtcGetCode(prog: hiprtcProgram, code: *mut c_char) -> c_int;
    /// Get the size of the compiled code object.
    pub fn hiprtcGetCodeSize(prog: hiprtcProgram, size: *mut usize) -> c_int;
    /// Get the compile log (null-terminated).
    pub fn hiprtcGetProgramLog(prog: hiprtcProgram, log: *mut c_char) -> c_int;
    /// Get the compile log size.
    pub fn hiprtcGetProgramLogSize(prog: hiprtcProgram, log_size: *mut usize) -> c_int;
    /// Destroy a program, freeing its resources.
    pub fn hiprtcDestroyProgram(prog: *mut hiprtcProgram) -> c_int;
    /// hiprtc's own version — the COMPILER that produced a cached code object, which can move
    /// independently of the HIP runtime. Third version in the module cache's key.
    pub fn hiprtcVersion(major: *mut c_int, minor: *mut c_int) -> c_int;
}

// ── librocblas ───────────────────────────────────────────────────────────────
//
// Slice 26: the dequant→f16→library-GEMM prefill path. `rocblas_gemm_ex` runs the
// f16 (fp32-accumulate) GEMM that beats the hand int8 WMMA kernel 3.6-5.9× on the
// isolated GEMM (see `examples/blas_probe`). Only the four entry points the prefill
// path calls are declared; the search path is emitted by `build.rs` (rocm feature).

#[link(name = "rocblas")]
extern "C" {
    /// Create a rocBLAS library handle (device context + workspace).
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> c_int;
    /// Destroy a rocBLAS handle.
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> c_int;
    /// Bind all subsequent rocBLAS calls on `handle` to `stream`.
    pub fn rocblas_set_stream(handle: rocblas_handle, stream: hipStream_t) -> c_int;
    /// General mixed-precision GEMM: `D = alpha·op(A)·op(B) + beta·C` (column-major).
    #[allow(clippy::too_many_arguments)]
    pub fn rocblas_gemm_ex(
        handle: rocblas_handle,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        lda: c_int,
        b: *const c_void,
        b_type: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *const c_void,
        c_type: c_int,
        ldc: c_int,
        d: *mut c_void,
        d_type: c_int,
        ldd: c_int,
        compute_type: c_int,
        algo: c_int,
        solution_index: i32,
        flags: u32,
    ) -> c_int;
}

/// An opaque rocBLAS library handle.
pub type rocblas_handle = *mut c_void;

/// rocBLAS success return code.
pub const ROCBLAS_STATUS_SUCCESS: c_int = 0;
// rocblas_operation
pub const ROCBLAS_OPERATION_NONE: c_int = 111;
pub const ROCBLAS_OPERATION_TRANSPOSE: c_int = 112;
// rocblas_datatype (real)
pub const ROCBLAS_DATATYPE_F16_R: c_int = 150;
pub const ROCBLAS_DATATYPE_F32_R: c_int = 151;
// rocblas_gemm_algo
pub const ROCBLAS_GEMM_ALGO_STANDARD: c_int = 0;

// ── Type aliases ─────────────────────────────────────────────────────────────

/// An opaque HIP stream.
pub type hipStream_t = *mut c_void;
/// An opaque HIP event.
pub type hipEvent_t = *mut c_void;
/// An opaque HIP module (compiled code object).
pub type hipModule_t = *mut c_void;
/// An opaque HIP kernel function.
pub type hipFunction_t = *mut c_void;
/// An opaque hiprtc program handle.
pub type hiprtcProgram = *mut c_void;

// ── Return-code constants ────────────────────────────────────────────────────

/// HIP / hiprtc success return code.
pub const HIP_SUCCESS: c_int = 0;
pub const HIPRTC_SUCCESS: c_int = 0;

// ── hipMemcpyKind ────────────────────────────────────────────────────────────

pub type hipMemcpyKind = c_int;
pub const HIP_MEMCPY_HOST_TO_DEVICE: hipMemcpyKind = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: hipMemcpyKind = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: hipMemcpyKind = 3;

// ── hipEvent flags ───────────────────────────────────────────────────────────

/// Event flag: skip timing bookkeeping (cheaper record/wait) — the pager only needs ordering.
pub const HIP_EVENT_DISABLE_TIMING: u32 = 0x2;

// ── hipHostMalloc flags ──────────────────────────────────────────────────────

/// The allocation is usable by all HIP contexts (safe default).
pub const HIP_HOST_MALLOC_PORTABLE: u32 = 0x1;
/// Map the allocation into the device address space so a device pointer aliases it
/// (`hipHostGetDevicePointer`) — required for the KV-overflow read-over-PCIe path.
pub const HIP_HOST_MALLOC_MAPPED: u32 = 0x2;

// ── hipDeviceProp_tR0000 — the ONLY device-properties layout this crate reads ─
//
// The linked `hipGetDeviceProperties` symbol (`hip_4.2`) fills the LEGACY layout, which ROCm 6+
// froze for ABI compatibility when it renamed the current struct to `hipDeviceProp_tR0600`. So
// that is what is declared here, and it is the only one — a struct shaped like `R0600` reads
// garbage out of this symbol, field-by-field, with no error to notice.
//
// This crate used to carry exactly such a struct, and it cost real numbers: `multiProcessorCount`
// read 1024 (it was landing on `maxThreadsDim[1]`) and `Capabilities::compute_units` reported that.
// Measured on ROCm 7 / RX 7900 XTX, every field below reads coherently (gfx1100, warp 32, 384-bit
// bus, 6 MiB L2, 48 WGPs) where the `R0600` shape did not.
//
// Reading a field through the WRONG layout is not merely useless: `gcnArchName` NAMES the kernel
// module cache's blob, and a stable-but-wrong token could name one cache file for two different
// archs. The `gcnArchName` caller therefore also demands the value start with `gfx` and disables
// caching otherwise.

/// The legacy (`R0000`) device-properties layout, named down to `gcnArchName` and padded past it.
/// Offsets are asserted below for EVERY field this crate reads, so a future ROCm that moves one is
/// a COMPILE error rather than a silently wrong capability or cache key.
#[repr(C)]
pub struct hipDeviceProp_tR0000 {
    pub name: [c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub total_const_mem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub multi_processor_count: c_int,
    pub l2_cache_size: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub compute_mode: c_int,
    pub clock_instruction_rate: c_int,
    /// 17 single-bit fields in ONE 32-bit storage unit — 4 bytes, not 8.
    pub arch: u32,
    pub concurrent_kernels: c_int,
    pub pci_domain_id: c_int,
    pub pci_bus_id: c_int,
    pub pci_device_id: c_int,
    pub max_shared_memory_per_multi_processor: usize,
    pub is_multi_gpu_board: c_int,
    pub can_map_host_memory: c_int,
    pub gcn_arch: c_int,
    pub gcn_arch_name: [c_char; 256],
    /// Everything after the arch name, padded to the 1472 bytes the caller's buffer already
    /// reserves — the runtime writes its own `sizeof`, which is no larger.
    _tail: [u8; 820],
}

// Every field this crate reads — `name`, `shared_mem_per_block`, `warp_size`,
// `multi_processor_count`, `gcn_arch_name` — pinned at COMPILE time against the layout above, plus
// the neighbours that fence them in. Offsets measured on ROCm 7 / RX 7900 XTX; if a future ROCm
// release reshuffles the frozen legacy struct, this fails to BUILD instead of quietly reporting a
// field that has moved (which is exactly how `compute_units` came to be 1024).
const _: () = assert!(std::mem::offset_of!(hipDeviceProp_tR0000, name) == 0);
const _: () = assert!(std::mem::offset_of!(hipDeviceProp_tR0000, total_global_mem) == 256);
const _: () = assert!(std::mem::offset_of!(hipDeviceProp_tR0000, shared_mem_per_block) == 264);
const _: () = assert!(std::mem::offset_of!(hipDeviceProp_tR0000, warp_size) == 276);
const _: () = assert!(std::mem::offset_of!(hipDeviceProp_tR0000, multi_processor_count) == 336);
const _: () =
    assert!(std::mem::offset_of!(hipDeviceProp_tR0000, max_threads_per_multi_processor) == 344);
const _: () = assert!(
    std::mem::offset_of!(hipDeviceProp_tR0000, max_shared_memory_per_multi_processor) == 376
);
const _: () = assert!(std::mem::offset_of!(hipDeviceProp_tR0000, gcn_arch_name) == 396);
const _: () = assert!(std::mem::size_of::<hipDeviceProp_tR0000>() == 1472);
