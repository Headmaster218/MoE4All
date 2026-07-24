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
use infr_core::error::{Error, Result};
use infr_core::graph::Graph;
use std::ffi::{c_int, c_void};
use std::sync::{Arc, Mutex};

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

// ── RocmBuffer ───────────────────────────────────────────────────────────────

/// A device buffer allocated with `hipMalloc`.
pub struct RocmBuffer {
    /// Device pointer (null if len == 0).
    pub(crate) ptr: *mut c_void,
    /// Byte length.
    pub(crate) len: usize,
    /// Whether `drop` should call `hipFree` (false for a slice/view into another buffer).
    pub(crate) owned: bool,
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
        if self.owned && !self.ptr.is_null() {
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
    /// Active weight-load progress bar.
    weight_pb: Arc<Mutex<Option<indicatif::ProgressBar>>>,
}

// The backend owns streams and device handles which are Send/Sync.
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl RocmBackend {
    /// Create a new ROCm backend on the given device index.
    pub fn new(device_id: c_int) -> Result<Self> {
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

        Ok(Self {
            device,
            stream,
            pipelines,
            weight_cache: Mutex::new(std::collections::HashMap::new()),
            pool: Mutex::new(BufferPool::new()),
            weight_pb: Arc::new(Mutex::new(None)),
        })
    }

    /// Read a device property field.
    fn prop(&self) -> ffi::hipDeviceProp_t {
        let mut props: ffi::hipDeviceProp_t = unsafe { std::mem::zeroed() };
        unsafe { ffi::hipGetDeviceProperties(&mut props, self.device) };
        props
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
        // Zero-init (calloc contract); OOM or a failed zero-fill returns Err (recoverable).
        let buf = RocmBuffer::try_alloc(bytes, self.stream)?;
        // Advance weight progress bar for weight/host-weight allocations
        if matches!(_usage, BufferUsage::Weights | BufferUsage::HostWeights) {
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(Box::new(buf))
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Skip zero-init for weight buffers (they get uploaded immediately); OOM returns Err.
        let buf = RocmBuffer::try_alloc_uninit(bytes, self.stream)?;
        if matches!(usage, BufferUsage::Weights | BufferUsage::HostWeights) {
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
        exec::execute_graph(
            &self.pipelines,
            &self.weight_cache,
            &self.pool,
            self.stream,
            plan,
            bindings,
        )
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
        if !self.stream.is_null() {
            unsafe {
                ffi::hipStreamSynchronize(self.stream);
                ffi::hipStreamDestroy(self.stream);
            }
        }
    }
}
