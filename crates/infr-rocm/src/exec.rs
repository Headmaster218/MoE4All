//! Graph execution: walk ops → resolve bound buffers → dispatch HIP kernels.
//!
//! Covered quant formats (Q4_K/Q6_K/Q8_0, see `native_decode_fmt`) are decoded in-kernel
//! from their RAW bytes on the `Linear`/`EmbedGather` paths — no f16 cache, VRAM ≈ quant_size.
//! Uncovered quantized weight tensors are dequantized to f16 on the host on first touch and
//! cached by the raw device-pointer address of their bound buffer.

use crate::backend::{bucket_bytes, BufferPool};
use crate::ffi::{self, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};
use crate::kernels::Pipelines;
use half::f16;
use infr_core::backend::{Bindings, GraphPlan, Plan};
use infr_core::error::{Error, Result};
use infr_core::graph::{AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_gguf::dequant;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

fn rocm_buf(b: &dyn infr_core::backend::Buffer) -> &crate::RocmBuffer {
    b.as_any()
        .downcast_ref::<crate::RocmBuffer>()
        .expect("rocm backend: buffer is not a RocmBuffer")
}

fn read_bytes(b: &crate::RocmBuffer, stream: ffi::hipStream_t) -> Vec<u8> {
    let mut v = vec![0u8; b.len];
    if b.len > 0 {
        // Sync the work stream BEFORE the readback: with the per-op sync removed, this is the
        // barrier that guarantees every queued async kernel/memset that produced `b` has retired
        // before we copy it to the host — independent of HIP's (per-thread vs legacy) default-stream
        // mode. This is one of the only two sync points kept on the hot path (host readbacks + the
        // final writeback barrier).
        unsafe {
            ffi::hipStreamSynchronize(stream);
        }
        unsafe {
            ffi::hipMemcpy(
                v.as_mut_ptr() as *mut c_void,
                b.ptr,
                b.len,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
        }
    }
    v
}

fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Result<Vec<f32>> {
    match dtype {
        DType::F32 => {
            // Raw f32 bytes — reinterpret directly.
            let f32s: &[f32] = bytemuck::cast_slice(bytes);
            Ok(f32s.to_vec())
        }
        DType::F16 => {
            // Raw f16 bytes — convert each half to f32.
            let f16s: &[u16] = bytemuck::cast_slice(bytes);
            Ok(f16s
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect())
        }
        DType::I32 => {
            // Bias / position tensor — bitcast i32 to f32.
            let i32s: &[i32] = bytemuck::cast_slice(bytes);
            Ok(i32s.iter().map(|&v| f32::from_bits(v as u32)).collect())
        }
        _ => dequant::dequant_block(dtype, bytes)
            .map_err(|e| be(format!("dequant {dtype:?} weight: {e}"))),
    }
}

/// Formats decoded natively in-kernel (Phase 3): the GEMV / EmbedGather reads the RAW quant
/// bytes and decodes each block on the fly, so no f16 cache is materialized in VRAM. Returns
/// `(elems_per_block, bytes_per_block, linear_kernel, embed_kernel)` for a covered dtype, else
/// `None` (uncovered formats keep the dequant→f16 fallback). The decode is bit-faithful to the
/// old cache path (see `kernels.rs` NATIVE_DECODE), so goldens do not move.
fn native_decode_fmt(dt: DType) -> Option<(usize, usize, &'static str, &'static str)> {
    match dt {
        DType::Q8_0 => Some((32, 34, "linear_q80", "embed_q80")),
        DType::Q4K => Some((256, 144, "linear_q4k", "embed_q4k")),
        DType::Q6K => Some((256, 210, "linear_q6k", "embed_q6k")),
        DType::Q5_0 => Some((32, 22, "linear_q50", "embed_q50")),
        _ => None,
    }
}

/// Int8-activation dp4a GEMV kernel (Phase 4) for a covered dtype: `(bytes_per_block, kernel)`.
/// The activation row is quantized to int8 once (`quant_i8_32`) and integer-dotted against the
/// decoded weight codes (scale-after) — dropping the Phase-3 per-element f16 round-trip. Returns
/// `None` for uncovered formats (they keep the Phase-3 native decode / dequant→f16 fallback), or
/// when `INFR_ROCM_NO_I8` selects the Phase-3 path for A/B benchmarking.
fn native_i8_fmt(dt: DType) -> Option<(usize, &'static str)> {
    if std::env::var_os("INFR_ROCM_NO_I8").is_some() {
        return None;
    }
    match dt {
        DType::Q8_0 => Some((34, "linear_i8_q80")),
        DType::Q4K => Some((144, "linear_i8_q4k")),
        DType::Q6K => Some((210, "linear_i8_q6k")),
        DType::Q5_0 => Some((22, "linear_i8_q50")),
        _ => None,
    }
}

/// Dequant-to-f16 kernel name (`deqf16_*`, kernels.rs `DEQUANT_F16`) for a covered dtype — the
/// weight decoder feeding the Slice-26 rocBLAS f16 prefill GEMM. Same covered set as
/// [`native_decode_fmt`] (Q8_0/Q4_K/Q6_K/Q5_0); `None` keeps a format on the WMMA/GEMV path.
fn deqf16_fmt(dt: DType) -> Option<&'static str> {
    match dt {
        DType::Q8_0 => Some("deqf16_q80"),
        DType::Q4K => Some("deqf16_q4k"),
        DType::Q6K => Some("deqf16_q6k"),
        DType::Q5_0 => Some("deqf16_q50"),
        _ => None,
    }
}

/// Explicit tile override for A/B benchmarking (`INFR_ROCM_WMMA_TILE=RxC`, one of 1x1/2x1/2x2).
/// `None` when unset → the shape-driven auto tier in [`wmma_tile`] is used.
fn wmma_tile_forced() -> Option<(u32, u32)> {
    match std::env::var("INFR_ROCM_WMMA_TILE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("1x1") => Some((1, 1)),
        Some("2x1") => Some((2, 1)),
        Some("2x2") => Some((2, 2)),
        _ => None,
    }
}

/// Register tile `(RM, CN)` for the WMMA prefill GEMM (Slice 25): each wave computes an RM×CN grid of
/// 16×16 output tiles, reusing every loaded A fragment across the CN weight-column tiles and every
/// decoded weight tile across the RM row tiles. Measured on gfx1100 (isolated-GEMM GFLOP/s sweep, see
/// `examples/wmma_bench`): blocking M (`2x1`) strictly beats the un-blocked Slice-15 tile (`1x1`) on
/// every shape (+2..16%); the wider `2x2` additionally wins the wide-N shapes (out_f ≥ 2048: up/gate,
/// wide projections) but loses ~11-14% on the square/narrow ones (qkv, down). So the auto tier is
/// `2x2` for wide-N GEMMs and `2x1` otherwise. `INFR_ROCM_WMMA_TILE` overrides for benchmarking.
fn wmma_tile(out_f: u32) -> (u32, u32) {
    if let Some(t) = wmma_tile_forced() {
        return t;
    }
    if out_f >= 2048 {
        (2, 2)
    } else {
        (2, 1)
    }
}

/// Matrix-core (WMMA) int8 prefill GEMM kernel (Phase 5, Slice-25 RM×CN-tiled) for a covered dtype.
/// Routed only for `m > 1` (prefill); decode (`m == 1`) stays on the `linear_i8_*` GEMV, which WMMA
/// can't help. Same int8 precision as `native_i8_fmt` (identical activation quant + weight codes),
/// and bit-identical f32 accumulation order to the Slice-15 kernel, so it holds the blessed goldens.
/// `INFR_ROCM_NO_WMMA` forces the GEMV path for A/B benchmarking. Returns `(kernel_name, RM, CN)`, or
/// `None` when the int8 path itself is disabled (`INFR_ROCM_NO_I8`).
fn native_wmma_fmt(dt: DType, out_f: u32) -> Option<(&'static str, u32, u32)> {
    if std::env::var_os("INFR_ROCM_NO_WMMA").is_some()
        || std::env::var_os("INFR_ROCM_NO_I8").is_some()
    {
        return None;
    }
    // Slice-27: Q4_K prefill defaults to the software-pipelined (prefetched double-buffered nibble)
    // 2x1 kernel — its overlapped load/decode/WMMA schedule beats the Slice-25 auto-tier on EVERY
    // representative shape in the isolated-GEMM micro-bench (`examples/wmma_bench`), including the
    // wide-N GEMMs where the un-pipelined 2x2 previously won, so it supersedes the 2x1/2x2 split for
    // Q4_K. Bit-identical math to `wmma_i8_q4k_2x1` (goldens unmoved). `INFR_ROCM_NO_PIPE=1` falls
    // back to the Slice-25 auto-tier for A/B benchmarking.
    if dt == DType::Q4K && std::env::var_os("INFR_ROCM_NO_PIPE").is_none() {
        return Some(("wmma_i8_q4k_pipe_2x1", 2, 1));
    }
    let (rm, cn) = wmma_tile(out_f);
    let name = match (dt, rm, cn) {
        (DType::Q8_0, 1, 1) => "wmma_i8_q80_1x1",
        (DType::Q8_0, 2, 2) => "wmma_i8_q80_2x2",
        (DType::Q8_0, _, _) => "wmma_i8_q80_2x1",
        (DType::Q4K, 1, 1) => "wmma_i8_q4k_1x1",
        (DType::Q4K, 2, 2) => "wmma_i8_q4k_2x2",
        (DType::Q4K, _, _) => "wmma_i8_q4k_2x1",
        (DType::Q6K, 1, 1) => "wmma_i8_q6k_1x1",
        (DType::Q6K, 2, 2) => "wmma_i8_q6k_2x2",
        (DType::Q6K, _, _) => "wmma_i8_q6k_2x1",
        (DType::Q5_0, 1, 1) => "wmma_i8_q50_1x1",
        (DType::Q5_0, 2, 2) => "wmma_i8_q50_2x2",
        (DType::Q5_0, _, _) => "wmma_i8_q50_2x1",
        _ => return None,
    };
    Some((name, rm, cn))
}

/// Native in-kernel decode for a MoE expert weight bank (Phase-3 for MoE). Returns
/// `(suffix, elems_per_block, bytes_per_block)` for a covered dtype — the suffix names the
/// `deq_*` decoder baked into the `moe_ffn_expert_<gu>_<dn>` kernel and the block geometry gives
/// the per-expert byte offset. `None` for uncovered formats (they keep the dequant→f16 fallback).
fn moe_native_fmt(dt: DType) -> Option<(&'static str, usize, usize)> {
    match dt {
        DType::Q8_0 => Some(("q80", 32, 34)),
        DType::Q4K => Some(("q4k", 256, 144)),
        DType::Q6K => Some(("q6k", 256, 210)),
        _ => None,
    }
}

/// Static kernel name for the `(gate/up format, down format)` combo — the 9 covered pairs over
/// {q80, q4k, q6k} (e.g. Q4_K_M is `("q4k", "q6k")`). Both suffixes come from `moe_native_fmt`, so
/// the `_` arm is unreachable.
fn moe_expert_kernel(gu: &str, dn: &str) -> &'static str {
    match (gu, dn) {
        ("q80", "q80") => "moe_ffn_expert_q80_q80",
        ("q80", "q4k") => "moe_ffn_expert_q80_q4k",
        ("q80", "q6k") => "moe_ffn_expert_q80_q6k",
        ("q4k", "q80") => "moe_ffn_expert_q4k_q80",
        ("q4k", "q4k") => "moe_ffn_expert_q4k_q4k",
        ("q4k", "q6k") => "moe_ffn_expert_q4k_q6k",
        ("q6k", "q80") => "moe_ffn_expert_q6k_q80",
        ("q6k", "q4k") => "moe_ffn_expert_q6k_q4k",
        ("q6k", "q6k") => "moe_ffn_expert_q6k_q6k",
        _ => unreachable!("moe_expert_kernel: uncovered ({gu}, {dn})"),
    }
}

/// Whether the int8-activation dp4a MoE expert path (Slice 20) is enabled. Reuses the dense
/// `INFR_ROCM_NO_I8` A/B switch: when set, MoE falls back to the Phase-3 f16-decode expert kernel.
fn moe_i8_enabled() -> bool {
    std::env::var_os("INFR_ROCM_NO_I8").is_none()
}

/// Static gate/up int8 kernel name for the gate/up format suffix (`moe_gate_up_act_i8_<gu>`). The
/// suffix comes from `moe_native_fmt`, so the `_` arm is unreachable.
fn moe_gate_up_i8_kernel(gu: &str) -> &'static str {
    match gu {
        "q80" => "moe_gate_up_act_i8_q80",
        "q4k" => "moe_gate_up_act_i8_q4k",
        "q6k" => "moe_gate_up_act_i8_q6k",
        _ => unreachable!("moe_gate_up_i8_kernel: uncovered ({gu})"),
    }
}

/// Static down int8 kernel name for the down format suffix (`moe_down_i8_<dn>`).
fn moe_down_i8_kernel(dn: &str) -> &'static str {
    match dn {
        "q80" => "moe_down_i8_q80",
        "q4k" => "moe_down_i8_q4k",
        "q6k" => "moe_down_i8_q6k",
        _ => unreachable!("moe_down_i8_kernel: uncovered ({dn})"),
    }
}

fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        let h = f16::from_f32(*x);
        out.extend_from_slice(&h.to_bits().to_le_bytes());
    }
    out
}

// ── Kernel dispatch helpers ──────────────────────────────────────────────────

fn dispatch_1d(
    pipelines: &Pipelines,
    stream: ffi::hipStream_t,
    kernel_name: &'static str,
    total_threads: u32,
    block_size: u32,
    args: Vec<Vec<u8>>,
) -> Result<()> {
    let func = pipelines.get(kernel_name)?;
    let grid_x = total_threads.div_ceil(block_size);
    let mut storage = args;
    let mut arg_ptrs: Vec<*mut c_void> = Vec::with_capacity(storage.len());
    for ab in storage.iter_mut() {
        arg_ptrs.push(ab.as_mut_ptr() as *mut c_void);
    }
    let rc = unsafe {
        ffi::hipModuleLaunchKernel(
            func,
            grid_x,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            arg_ptrs.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if rc != HIP_SUCCESS {
        return Err(be(format!("hipModuleLaunchKernel({kernel_name}): rc={rc}")));
    }
    Ok(())
}

/// Launch `kernel_name` with an explicit `(grid_x, grid_y)` grid of `block_size`-thread blocks.
/// Used by the int8 GEMV, whose grid is (out_f output rows, m activation rows).
fn dispatch_grid(
    pipelines: &Pipelines,
    stream: ffi::hipStream_t,
    kernel_name: &'static str,
    grid_x: u32,
    grid_y: u32,
    block_size: u32,
    args: Vec<Vec<u8>>,
) -> Result<()> {
    let func = pipelines.get(kernel_name)?;
    let mut storage = args;
    let mut arg_ptrs: Vec<*mut c_void> = Vec::with_capacity(storage.len());
    for ab in storage.iter_mut() {
        arg_ptrs.push(ab.as_mut_ptr() as *mut c_void);
    }
    let rc = unsafe {
        ffi::hipModuleLaunchKernel(
            func,
            grid_x,
            grid_y,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            arg_ptrs.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if rc != HIP_SUCCESS {
        return Err(be(format!("hipModuleLaunchKernel({kernel_name}): rc={rc}")));
    }
    Ok(())
}

/// Launch `kernel_name` with an explicit block count (`grid_x` blocks of `block_size` threads) and a
/// dynamic shared-memory allocation of `smem_bytes`. Used by the chunked DeltaNet prefill kernel,
/// which runs one block per value head and stashes its per-chunk tensors in dynamic LDS.
fn dispatch_blocks_smem(
    pipelines: &Pipelines,
    stream: ffi::hipStream_t,
    kernel_name: &'static str,
    grid_x: u32,
    block_size: u32,
    smem_bytes: u32,
    args: Vec<Vec<u8>>,
) -> Result<()> {
    let func = pipelines.get(kernel_name)?;
    let mut storage = args;
    let mut arg_ptrs: Vec<*mut c_void> = Vec::with_capacity(storage.len());
    for ab in storage.iter_mut() {
        arg_ptrs.push(ab.as_mut_ptr() as *mut c_void);
    }
    let rc = unsafe {
        ffi::hipModuleLaunchKernel(
            func,
            grid_x,
            1,
            1,
            block_size,
            1,
            1,
            smem_bytes,
            stream,
            arg_ptrs.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if rc != HIP_SUCCESS {
        return Err(be(format!("hipModuleLaunchKernel({kernel_name}): rc={rc}")));
    }
    Ok(())
}

fn arg_ptr(p: *mut c_void) -> Vec<u8> {
    (p as u64).to_le_bytes().to_vec()
}
fn arg_i32(v: i32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn arg_f32(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

// ── ExecCtx ──────────────────────────────────────────────────────────────────

struct ExecCtx<'a> {
    dev: Vec<Option<crate::RocmBuffer>>,
    vals: Vec<Option<Vec<f32>>>,
    weight_cache: &'a Mutex<HashMap<(usize, usize), crate::RocmBuffer>>,
    /// Reusable device-scratch pool (persists across `execute` calls on the backend).
    pool: &'a Mutex<BufferPool>,
    /// Pool draws made this forward pass: `(ptr, bucket_bytes)`, returned to `pool` on `Drop`
    /// (both the success path and any early-error return) so nothing is `hipFree`'d per op.
    pooled: Vec<(*mut c_void, usize)>,
    stream: ffi::hipStream_t,
    /// rocBLAS handle bound to `stream` for the OPT-IN Slice-26 f16 prefill GEMM (`INFR_ROCM_BLAS=1`),
    /// or `null` (the default) — in which case the prefill path uses the int8 WMMA kernel.
    rocblas: ffi::rocblas_handle,
}

impl<'a> ExecCtx<'a> {
    fn f16_dev(&self, data: &[u8]) -> crate::RocmBuffer {
        // The dequant→f16 weight cache is long-lived (backend lifetime), NOT per-forward scratch,
        // so it allocates directly and is owned by `weight_cache` — never routed through the pool.
        let mut buf = crate::RocmBuffer::alloc(data.len().max(1), self.stream);
        buf.upload(data, self.stream);
        buf
    }

    /// Draw a `bytes`-byte scratch buffer from the pool. When `zero`, the reused region is cleared
    /// with an ASYNC memset (calloc contract, no host sync) — required for accumulators and
    /// partial-write outputs (`Copy`/`CopyStrided`/MoE dst/unproduced tensors). Fully-written
    /// outputs (GEMV / elementwise) pass `zero = false` and skip the clear. The returned
    /// `RocmBuffer` is `owned: false` (its `Drop` is a no-op); the allocation is returned to the
    /// pool via `ExecCtx::Drop`. `len` is the LOGICAL byte length (≤ bucket), so downstream
    /// `min(len, …)` copy clamps stay correct.
    fn pool_buf(&mut self, bytes: usize, zero: bool) -> crate::RocmBuffer {
        let len = bytes.max(1);
        let bucket = bucket_bytes(len);
        let ptr = self.pool.lock().unwrap().take(bucket);
        if zero {
            let rc = unsafe { ffi::hipMemsetAsync(ptr, 0, len, self.stream) };
            debug_assert_eq!(rc, HIP_SUCCESS, "hipMemsetAsync(pool zero-on-reuse)");
        }
        self.pooled.push((ptr, bucket));
        crate::RocmBuffer {
            ptr,
            len,
            owned: false,
        }
    }

    /// Zeroed scratch for `n` f32 ELEMENTS (calloc contract). Pooled + async-cleared. Every op
    /// `dst` uses this: the async memset is near-free (no host sync) and keeping the calloc
    /// contract universal guarantees the goldens can't move on a partial-write op. Genuinely
    /// fully-overwritten transient scratch (the int8 `qx`/`xs`, the aliased-copy clone) instead
    /// calls [`pool_buf`](Self::pool_buf) with `zero = false` to skip even that memset.
    fn zero_dev(&mut self, n: usize) -> crate::RocmBuffer {
        self.pool_buf((n * 4).max(1), true)
    }

    fn host_vals(&mut self, id: TensorId, g: &Graph, bindings: &Bindings) -> Result<&[f32]> {
        let i = id.0 as usize;
        if self.vals[i].is_none() {
            let decl = &g.tensors[i];
            let val = match decl.kind {
                TensorKind::Input | TensorKind::Weight => {
                    let b = rocm_buf(bindings.get(id).expect("rocm: unbound Input/Weight"));
                    let raw = read_bytes(b, self.stream);
                    bytes_to_f32(&raw, decl.desc.dtype)?
                }
                TensorKind::Internal | TensorKind::Output => {
                    if let Some(ref db) = self.dev[i] {
                        let raw = read_bytes(db, self.stream);
                        bytes_to_f32(&raw, decl.desc.dtype)?
                    } else {
                        vec![0f32; decl.desc.numel()]
                    }
                }
            };
            self.vals[i] = Some(val);
        }
        Ok(self.vals[i].as_ref().unwrap())
    }

    fn ensure_device(
        &mut self,
        id: TensorId,
        g: &Graph,
        bindings: &Bindings,
    ) -> Result<*mut c_void> {
        let i = id.0 as usize;
        if let Some(ref db) = self.dev[i] {
            return Ok(db.ptr);
        }
        // For Input/Weight tensors, use the bound buffer directly (no host download).
        let decl = &g.tensors[i];
        let ptr = match decl.kind {
            TensorKind::Input | TensorKind::Weight => {
                let b = rocm_buf(bindings.get(id).expect("rocm: unbound Input/Weight"));
                let p = b.ptr;
                // Track in dev so subsequent accesses find it.
                self.dev[i] = Some(crate::RocmBuffer {
                    ptr: p,
                    len: b.len,
                    owned: false,
                });
                p
            }
            TensorKind::Internal | TensorKind::Output => {
                // Not yet produced — allocate a zero-filled buffer.
                let db = self.zero_dev(decl.desc.numel());
                let p = db.ptr;
                self.dev[i] = Some(db);
                p
            }
        };
        Ok(ptr)
    }

    /// For an in-place `Copy`/`CopyStrided` where `src == dst`, return a temp device buffer holding
    /// a full DtoD clone of the source so the kernel reads a stable snapshot (the read window can't
    /// be clobbered by the in-place write). Returns `None` when `src != dst` (the common case), and
    /// the caller reads `src` directly. Both `src` and `dst` must already be on device.
    fn stage_if_aliased(&mut self, src: TensorId, dst: TensorId) -> Option<crate::RocmBuffer> {
        if src.0 != dst.0 {
            return None;
        }
        let (sptr, slen) = {
            let sb = self.dev[src.0 as usize].as_ref().unwrap();
            (sb.ptr, sb.len)
        };
        // Fully overwritten by the DtoD clone below → un-cleared pool scratch.
        let tmp = self.pool_buf(slen.max(1), false);
        if slen > 0 {
            unsafe {
                ffi::hipMemcpyDtoD(tmp.ptr, sptr, slen);
            }
        }
        Some(tmp)
    }

    fn dequant_weight_or_cache(
        &mut self,
        id: TensorId,
        g: &Graph,
        bindings: &Bindings,
    ) -> Result<*mut c_void> {
        let i = id.0 as usize;
        let b = rocm_buf(bindings.get(id).expect("rocm: unbound Weight"));
        // Key on (address, byte length): a recycled device address that now backs a differently-
        // sized weight must MISS (its stale dequant has the wrong length), forcing a re-dequant.
        let key = (b.ptr as usize, b.len);
        {
            let cache = self.weight_cache.lock().unwrap();
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.ptr);
            }
        }
        let dt = g.desc(id).dtype;
        let raw = read_bytes(b, self.stream);
        let f32s = bytes_to_f32(&raw, dt)?;
        let f16_bytes = f32_to_f16_bytes(&f32s);
        let dq = self.f16_dev(&f16_bytes);
        let ptr = dq.ptr;
        let len = dq.len;
        {
            let mut cache = self.weight_cache.lock().unwrap();
            // Cache owns the device memory (owned: true)
            cache.insert(
                key,
                crate::RocmBuffer {
                    ptr: dq.ptr,
                    len: dq.len,
                    owned: true,
                },
            );
        }
        // Store a non-owned reference in dev so ctx.drop doesn't free it.
        // Prevent dq from dropping (cache owns the allocation now).
        std::mem::forget(dq);
        self.dev[i] = Some(crate::RocmBuffer {
            ptr,
            len,
            owned: false,
        });
        Ok(ptr)
    }
}

impl Drop for ExecCtx<'_> {
    fn drop(&mut self) {
        // Return every pool draw to the free-list (success OR early-error path). The pooled
        // `RocmBuffer`s stored in `dev` are `owned: false`, so their own `Drop` frees nothing —
        // this is the sole owner of the reuse lifetime. The caller has already synced the stream
        // before we drop on the success path; on an error path the backend is being torn down.
        let mut pool = self.pool.lock().unwrap();
        for (ptr, bucket) in self.pooled.drain(..) {
            pool.give(bucket, ptr);
        }
    }
}

// ── Main execute walk ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn execute_graph(
    pipelines: &Pipelines,
    weight_cache: &Mutex<HashMap<(usize, usize), crate::RocmBuffer>>,
    pool: &Mutex<BufferPool>,
    stream: ffi::hipStream_t,
    rocblas: ffi::rocblas_handle,
    plan: &dyn Plan,
    bindings: &Bindings,
) -> Result<()> {
    let g = &plan
        .as_any()
        .downcast_ref::<GraphPlan>()
        .expect("rocm backend: plan is not a GraphPlan")
        .graph;
    let n = g.tensors.len();
    let mut ctx = ExecCtx {
        dev: (0..n).map(|_| None).collect(),
        vals: (0..n).map(|_| None).collect(),
        weight_cache,
        pool,
        pooled: Vec::new(),
        stream,
        rocblas,
    };

    // No per-op sync: the whole op list queues on ONE stream, which serializes device work, so
    // intra-graph producer→consumer ordering holds without a host round-trip. The only syncs are
    // (a) inside `read_bytes`/`host_vals`, immediately before a host readback, and (b) the single
    // barrier below, before the cross-stream writeback DtoD + the final checked sync. With the
    // allocation churn gone (buffer pool), those per-op `hipMalloc`/`hipFree`/`hipStreamSynchronize`
    // device syncs — the real decode bottleneck — are all off the hot path.
    for op in g.ops.iter() {
        run_op(op, g, bindings, pipelines, &mut ctx)?;
    }

    // Barrier all queued op work before the writeback: the writeback `hipMemcpyDtoD` runs on the
    // NULL stream, which is NOT ordered against our non-default work stream, so it must observe a
    // completed stream first.
    unsafe { ffi::hipStreamSynchronize(stream) };
    let direct = g.in_place_inputs();
    for (i, decl) in g.tensors.iter().enumerate() {
        let id = TensorId(i as u32);
        let wb = matches!(decl.kind, TensorKind::Output)
            || (decl.kind == TensorKind::Input
                && decl.desc.dtype == DType::F32
                && !direct.contains(&id));
        if !wb {
            continue;
        }
        if let Some(b) = bindings.get(id) {
            let dst = rocm_buf(b);
            if let Some(ref dev_buf) = ctx.dev[i] {
                if dev_buf.len > 0 {
                    unsafe {
                        ffi::hipMemcpyDtoD(dst.ptr, dev_buf.ptr, dev_buf.len.min(dst.len));
                    }
                }
            } else if let Some(ref vals) = ctx.vals[i] {
                let bytes = bytemuck::cast_slice::<f32, u8>(vals);
                let n = bytes.len().min(dst.len);
                if n > 0 {
                    unsafe {
                        ffi::hipMemcpy(
                            dst.ptr,
                            bytes.as_ptr() as *const c_void,
                            n,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        );
                    }
                }
            }
        }
    }

    let rc = unsafe { ffi::hipStreamSynchronize(stream) };
    if rc != HIP_SUCCESS {
        return Err(be(format!("hipStreamSynchronize: rc={rc}")));
    }
    Ok(())
}

// ── Per-op dispatch ──────────────────────────────────────────────────────────

macro_rules! args { ($($e:expr),* $(,)?) => { vec![$($e),*] }; }

fn run_op(
    op: &Op,
    g: &Graph,
    bindings: &Bindings,
    pipelines: &Pipelines,
    ctx: &mut ExecCtx,
) -> Result<()> {
    match *op {
        Op::RmsNorm {
            x,
            weight,
            dst,
            rows,
            dim,
            eps,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * dim as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            // One block per row; the block reduces the sum-of-squares across a wave.
            dispatch_grid(
                pipelines,
                ctx.stream,
                "rmsnorm",
                rows,
                1,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(dim as i32),
                    arg_f32(eps),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::RmsNormAdd {
            x,
            weight,
            dst,
            rows,
            dim,
            eps,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(dst, g, bindings)?;
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let dd = ctx.dev[dst.0 as usize].as_ref().unwrap();
            // One block per row; the block reduces the sum-of-squares across a wave.
            dispatch_grid(
                pipelines,
                ctx.stream,
                "rmsnorm_add",
                rows,
                1,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(dim as i32),
                    arg_f32(eps),
                ],
            )?;
        }
        Op::Linear {
            x,
            weight,
            dst,
            m,
            in_f,
            out_f,
            w_off,
        } => {
            let wdt = g.desc(weight).dtype;
            if let (Some((qpb, bpb, _, _)), Some((bpb_i8, i8_kernel))) =
                (native_decode_fmt(wdt), native_i8_fmt(wdt))
            {
                // Int8-activation dp4a decode (Phase 4): quantize the `m×in_f` activation to int8
                // ONCE (`quant_i8_32`, per-32-block scale), then integer-dot against the decoded
                // weight codes (scale-after) via `linear_i8_*`. Drops the Phase-3 per-element f16
                // round-trip → decode is no longer ALU-bound. `bpb == bpb_i8` (same weight layout);
                // the bound quant buffer is pre-advanced past `w_off`, a whole number of output
                // rows × `in_f` (a multiple of `qpb`), so `(w_off/qpb)*bpb` is exact.
                debug_assert_eq!(bpb, bpb_i8);
                let wptr = ctx.ensure_device(weight, g, bindings)?;
                ctx.ensure_device(x, g, bindings)?;
                let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
                let mu = m as usize;
                let inu = in_f as usize;
                let ou = out_f as usize;
                let blk_off = (w_off as usize / qpb) * bpb;
                let wptr_off = unsafe { (wptr as *mut u8).add(blk_off) as *mut c_void };
                match (m > 1 && !ctx.rocblas.is_null())
                    .then(|| deqf16_fmt(wdt))
                    .flatten()
                {
                    Some(deq_kernel) => {
                        // Prefill (m>1), OPT-IN (`INFR_ROCM_BLAS=1` → handle live): Slice-26 rocBLAS
                        // f16 GEMM. Dequantize the weight to a POOLED transient f16 buffer and cast the
                        // activation to f16, then `dst[m,out_f] = x[m,in_f] · Wᵀ` via `rocblas_gemm_ex`
                        // (f16 in, f32 out, f32 accumulate). The library GEMM peaks 3.6-5.9× over the
                        // hand int8 WMMA kernel on the ISOLATED GEMM (examples/blas_probe) — but the
                        // per-forward dequant→f16 tax makes it a NET LOSS end-to-end (~0.88× pp512) and
                        // the transient f16 pool buffers OOM at 8B, so it is OFF by default (the WMMA
                        // arm below is the shipping path). All three buffers live on `ctx.stream`, to
                        // which the handle is bound, so the dequant/cast → GEMM ordering holds sync-free.
                        let wf16 = ctx.pool_buf((ou * inu * 2).max(1), false);
                        dispatch_1d(
                            pipelines,
                            ctx.stream,
                            deq_kernel,
                            (ou * inu) as u32,
                            256,
                            args![
                                arg_ptr(wptr_off),
                                arg_ptr(wf16.ptr),
                                arg_i32((ou * inu) as i32),
                            ],
                        )?;
                        let xf16 = ctx.pool_buf((mu * inu * 2).max(1), false);
                        dispatch_1d(
                            pipelines,
                            ctx.stream,
                            "cast_f32_f16",
                            (mu * inu) as u32,
                            256,
                            args![
                                arg_ptr(bx_ptr),
                                arg_ptr(xf16.ptr),
                                arg_i32((mu * inu) as i32),
                            ],
                        )?;
                        let dd = ctx.zero_dev(mu * ou);
                        // Column-major rocBLAS: computing Cᵀ[out_f,m] = W[out_f,in_f]·Xᵀ[in_f,m] with
                        // A=W transposed, B=X none yields exactly the row-major dst[m,out_f]. Weight
                        // row-major [out_f,in_f] == col-major [in_f,out_f] (lda=in_f); activation
                        // row-major [m,in_f] == col-major [in_f,m] (ldb=in_f); output ldc=out_f.
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        let rc = unsafe {
                            ffi::rocblas_gemm_ex(
                                ctx.rocblas,
                                ffi::ROCBLAS_OPERATION_TRANSPOSE,
                                ffi::ROCBLAS_OPERATION_NONE,
                                out_f as i32,
                                m as i32,
                                in_f as i32,
                                &alpha as *const f32 as *const c_void,
                                wf16.ptr,
                                ffi::ROCBLAS_DATATYPE_F16_R,
                                in_f as i32,
                                xf16.ptr,
                                ffi::ROCBLAS_DATATYPE_F16_R,
                                in_f as i32,
                                &beta as *const f32 as *const c_void,
                                dd.ptr,
                                ffi::ROCBLAS_DATATYPE_F32_R,
                                out_f as i32,
                                dd.ptr,
                                ffi::ROCBLAS_DATATYPE_F32_R,
                                out_f as i32,
                                ffi::ROCBLAS_DATATYPE_F32_R,
                                ffi::ROCBLAS_GEMM_ALGO_STANDARD,
                                0,
                                0,
                            )
                        };
                        if rc != ffi::ROCBLAS_STATUS_SUCCESS {
                            return Err(be(format!("rocblas_gemm_ex: rc={rc}")));
                        }
                        ctx.dev[dst.0 as usize] = Some(dd);
                    }
                    None => {
                        // Int8-activation dp4a path: quantize the `m×in_f` activation to int8 ONCE
                        // (`quant_i8_32`, per-32-block scale), then integer-dot against the decoded
                        // weight codes (scale-after). `bpb == bpb_i8` (same layout). The int8 codes /
                        // scales are drawn from the scratch pool (fully written before any read → `out`,
                        // un-cleared) and stay live until end-of-forward, so the async GEMM/GEMV that
                        // reads them never races a pool reuse.
                        let nb = inu / 32; // in_f is 32-aligned for every covered format
                        let qx = ctx.pool_buf((mu * inu).max(1), false);
                        let xs = ctx.pool_buf((mu * nb * 4).max(1), false);
                        dispatch_1d(
                            pipelines,
                            ctx.stream,
                            "quant_i8_32",
                            (mu * nb) as u32,
                            256,
                            args![
                                arg_ptr(bx_ptr),
                                arg_ptr(qx.ptr),
                                arg_ptr(xs.ptr),
                                arg_i32(m as i32),
                                arg_i32(in_f as i32),
                            ],
                        )?;
                        let dd = ctx.zero_dev(mu * ou);
                        match (m > 1).then(|| native_wmma_fmt(wdt, out_f)).flatten() {
                            Some((wmma_kernel, rm, cn)) => {
                                // Prefill (m>1), BLAS disabled: matrix-core int8 GEMM. Grid =
                                // (ceil(out_f/(16*CN)), ceil(m/(16*RM))), one wave32 block per
                                // 16*RM × 16*CN output tile — reuses each A fragment across the CN
                                // weight-column tiles and each decoded weight tile across the RM row
                                // tiles. Bit-identical f32 accumulation to the Slice-15 kernel.
                                dispatch_grid(
                                    pipelines,
                                    ctx.stream,
                                    wmma_kernel,
                                    out_f.div_ceil(16 * cn),
                                    m.div_ceil(16 * rm),
                                    32,
                                    args![
                                        arg_ptr(qx.ptr),
                                        arg_ptr(xs.ptr),
                                        arg_ptr(wptr_off),
                                        arg_ptr(dd.ptr),
                                        arg_i32(m as i32),
                                        arg_i32(in_f as i32),
                                        arg_i32(out_f as i32),
                                    ],
                                )?;
                            }
                            None => {
                                // Decode (m==1) or WMMA disabled: the dp4a GEMV. Grid = (out_f, m):
                                // one wave32 block per (output row, activation row).
                                dispatch_grid(
                                    pipelines,
                                    ctx.stream,
                                    i8_kernel,
                                    out_f,
                                    m,
                                    32,
                                    args![
                                        arg_ptr(qx.ptr),
                                        arg_ptr(xs.ptr),
                                        arg_ptr(wptr_off),
                                        arg_ptr(dd.ptr),
                                        arg_i32(m as i32),
                                        arg_i32(in_f as i32),
                                        arg_i32(out_f as i32),
                                    ],
                                )?;
                            }
                        }
                        ctx.dev[dst.0 as usize] = Some(dd);
                    }
                }
            } else if let Some((qpb, bpb, kname, _)) = native_decode_fmt(wdt) {
                // Native in-kernel decode: read the RAW quant bytes (no f16 cache → VRAM drops).
                // The bound quant buffer is pre-advanced past `w_off`; `w_off` is always a whole
                // number of output rows × `in_f`, hence a multiple of `qpb`, so the block offset
                // `(w_off / qpb) * bpb` is exact.
                let wptr = ctx.ensure_device(weight, g, bindings)?;
                ctx.ensure_device(x, g, bindings)?;
                let dd = ctx.zero_dev(m as usize * out_f as usize);
                let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
                let blk_off = (w_off as usize / qpb) * bpb;
                let wptr_off = unsafe { (wptr as *mut u8).add(blk_off) as *mut c_void };
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    kname,
                    m * 256,
                    256,
                    args![
                        arg_ptr(bx.ptr),
                        arg_ptr(wptr_off),
                        arg_ptr(dd.ptr),
                        arg_i32(m as i32),
                        arg_i32(in_f as i32),
                        arg_i32(out_f as i32),
                    ],
                )?;
                ctx.dev[dst.0 as usize] = Some(dd);
            } else {
                let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
                ctx.ensure_device(x, g, bindings)?;
                let dd = ctx.zero_dev(m as usize * out_f as usize);
                let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
                let wptr_off = unsafe { (wptr as *mut u8).add(w_off as usize * 2) as *mut c_void };
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "linear_f16",
                    m * 256,
                    256,
                    args![
                        arg_ptr(bx.ptr),
                        arg_ptr(wptr_off),
                        arg_ptr(dd.ptr),
                        arg_i32(m as i32),
                        arg_i32(in_f as i32),
                        arg_i32(out_f as i32),
                    ],
                )?;
                ctx.dev[dst.0 as usize] = Some(dd);
            }
        }
        Op::Softmax {
            x,
            dst,
            rows,
            dim,
            scale,
            scale_buf,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            let s = if let Some(sid) = scale_buf {
                ctx.host_vals(sid, g, bindings)?
                    .first()
                    .copied()
                    .unwrap_or(scale)
            } else {
                scale
            };
            let dd = ctx.zero_dev(rows as usize * dim as usize);
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            let dd_ptr = dd.ptr;
            dispatch_1d(
                pipelines,
                ctx.stream,
                "softmax",
                rows,
                256,
                args![
                    arg_ptr(bx_ptr),
                    arg_ptr(dd_ptr),
                    arg_i32(rows as i32),
                    arg_i32(dim as i32),
                    arg_f32(s),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::QkNorm {
            x,
            weight,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
            x_stride,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "qk_norm",
                rows * n_head,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_f32(eps),
                    arg_i32(x_stride as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::GatedRmsNorm {
            x,
            weight,
            gate,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(gate, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bg = ctx.dev[gate.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "gated_rmsnorm",
                rows * n_head,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(bg.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_f32(eps),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Rope {
            x,
            positions,
            dst,
            rows,
            n_head,
            head_dim,
            rope_dim,
            theta,
            freq_factors,
            x_stride,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(positions, g, bindings)?;
            let ff_ptr = if let Some(fid) = freq_factors {
                ctx.ensure_device(fid, g, bindings)?;
                ctx.dev[fid.0 as usize].as_ref().unwrap().ptr
            } else {
                std::ptr::null_mut()
            };
            // Re-fetch after ensure_device calls (borrow lifetime)
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            let bp_ptr = ctx.dev[positions.0 as usize].as_ref().unwrap().ptr;
            // Per-row stride in elements (0 = packed n_head*head_dim). Mirrors the fused
            // qk_norm_rope stride convention: heads stay packed within a strided row.
            let stride_elems = if x_stride > 0 {
                x_stride as usize
            } else {
                n_head as usize * head_dim as usize
            };
            if dst == x {
                let rope_args = args![
                    arg_ptr(bx_ptr),
                    arg_ptr(bp_ptr),
                    arg_ptr(ff_ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_i32(rope_dim as i32),
                    arg_f32(theta),
                    arg_i32(x_stride as i32),
                ];
                dispatch_1d(pipelines, ctx.stream, "rope", rows, 256, rope_args)?;
            } else {
                // Copy the FULL (possibly strided) source so both the pass-through dims and the
                // inter-row gaps survive, then rotate in place. A packed input (x_stride == 0)
                // allocs the natural rows*n_head*head_dim; a strided view needs rows*stride so the
                // kernel's off = row*stride + h*head_dim stays in bounds for every row.
                let dd = ctx.zero_dev(rows as usize * stride_elems);
                unsafe {
                    ffi::hipMemcpyDtoD(
                        dd.ptr,
                        bx_ptr,
                        dd.len.min(ctx.dev[x.0 as usize].as_ref().unwrap().len),
                    );
                }
                let dst_args = args![
                    arg_ptr(dd.ptr),
                    arg_ptr(bp_ptr),
                    arg_ptr(ff_ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_head as i32),
                    arg_i32(head_dim as i32),
                    arg_i32(rope_dim as i32),
                    arg_f32(theta),
                    arg_i32(x_stride as i32),
                ];
                dispatch_1d(pipelines, ctx.stream, "rope", rows, 256, dst_args)?;
                ctx.dev[dst.0 as usize] = Some(dd);
            }
        }
        Op::QkNormRope {
            x,
            weight,
            positions,
            dst,
            rows,
            n_head,
            head_dim,
            rope_dim,
            eps,
            theta,
            freq_factors,
            x_stride,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(positions, g, bindings)?;
            let ff_ptr = if let Some(fid) = freq_factors {
                ctx.ensure_device(fid, g, bindings)?;
                ctx.dev[fid.0 as usize].as_ref().unwrap().ptr
            } else {
                std::ptr::null_mut()
            };
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            let bp_ptr = ctx.dev[positions.0 as usize].as_ref().unwrap().ptr;
            let total = rows * n_head;
            // Output is ALWAYS a fresh PACKED [rows, n_head, head_dim] buffer: the kernel reads the
            // (possibly strided/interleaved q+g) input and writes the packed query — so no in-place
            // rotation and no strided-source copy (the old copy grabbed a packed prefix of a wider
            // row and then indexed it with the strided stride → out-of-bounds on multi-row prefill).
            // Matches infr-cpu QkNormRope, which always produces a fresh packed `out`.
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let qnr_args = args![
                arg_ptr(bx_ptr),
                arg_ptr(wptr),
                arg_ptr(bp_ptr),
                arg_ptr(ff_ptr),
                arg_ptr(dd.ptr),
                arg_i32(rows as i32),
                arg_i32(n_head as i32),
                arg_i32(head_dim as i32),
                arg_i32(rope_dim as i32),
                arg_f32(eps),
                arg_f32(theta),
                arg_i32(x_stride as i32),
            ];
            dispatch_1d(pipelines, ctx.stream, "qk_norm_rope", total, 256, qnr_args)?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::WriteKv {
            src,
            cache,
            pos,
            rows,
            row_stride,
        } => {
            ctx.ensure_device(src, g, bindings)?;
            let bs = ctx.dev[src.0 as usize].as_ref().unwrap();
            let bc = rocm_buf(bindings.get(cache).expect("rocm: unbound KV cache"));
            dispatch_1d(
                pipelines,
                ctx.stream,
                "write_kv",
                rows,
                256,
                args![
                    arg_ptr(bs.ptr),
                    arg_ptr(bc.ptr),
                    arg_i32(pos as i32),
                    arg_i32(rows as i32),
                    arg_i32(row_stride as i32),
                    arg_i32(0), // src_stride (0 = packed = row_stride)
                ],
            )?;
        }
        Op::Attention {
            q,
            k_cache,
            v_cache,
            dst,
            rows,
            kv_len,
            n_head,
            n_kv,
            head_dim,
            scale,
            mask,
            pos,
        } => {
            ctx.ensure_device(q, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize);
            let bk = rocm_buf(bindings.get(k_cache).expect("rocm: unbound K cache"));
            let bv = rocm_buf(bindings.get(v_cache).expect("rocm: unbound V cache"));
            let (bk_ptr, bv_ptr) = (bk.ptr, bv.ptr);
            let bq_ptr = ctx.dev[q.0 as usize].as_ref().unwrap().ptr;
            let dd_ptr = dd.ptr;
            let (mt, swa): (i32, i32) = match mask {
                AttnMask::Causal => (0, 0),
                AttnMask::SlidingWindow(w) => (1, w as i32),
                AttnMask::Canvas { lo } => (2, lo as i32),
            };

            // Split-KV (flash-decoding) for DECODE (rows==1). The single-wave `attention` kernel runs
            // ONE wave per (row, head) that scans ALL kv serially — fine at low depth, but at long
            // context that one wave crawls while ~95 CUs idle. Split-KV partitions kv into `n_chunks`
            // contiguous chunks, launches one wave per (row, head, chunk) to compute per-chunk
            // online-softmax partials, then a combine wave merges them. Adaptive chunking (mirrors the
            // Vulkan attn_partial policy): aim ~32 chunks/head, each 64..512 keys. Only worth it when
            // rows==1 AND the derived n_chunks>1 (short-context decode → n_chunks==1 → plain kernel,
            // no scratch, no combine). Prefill (rows>1) already fills the grid with rows*n_head waves
            // and stays on the plain kernel.
            let heads = rows as usize * n_head as usize;
            let kvl = kv_len as usize;
            let chunk_size = kvl.div_ceil(32).clamp(64, 512);
            let n_chunks = kvl.div_ceil(chunk_size).max(1);
            if rows == 1 && n_chunks > 1 {
                let hd = head_dim as usize;
                let pm = ctx.pool_buf(heads * n_chunks * 4, false);
                let pl = ctx.pool_buf(heads * n_chunks * 4, false);
                let pacc = ctx.pool_buf(heads * n_chunks * hd * 4, false);
                // Pass 1: one wave per (row, head, chunk).
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "attention_split_partial",
                    (heads * n_chunks) as u32 * 32,
                    32,
                    args![
                        arg_ptr(bq_ptr),
                        arg_ptr(bk_ptr),
                        arg_ptr(bv_ptr),
                        arg_ptr(pm.ptr),
                        arg_ptr(pl.ptr),
                        arg_ptr(pacc.ptr),
                        arg_i32(rows as i32),
                        arg_i32(kv_len as i32),
                        arg_i32(n_head as i32),
                        arg_i32(n_kv as i32),
                        arg_i32(head_dim as i32),
                        arg_f32(scale),
                        arg_i32(pos as i32),
                        arg_i32(mt),
                        arg_i32(swa),
                        arg_i32(chunk_size as i32),
                        arg_i32(n_chunks as i32),
                    ],
                )?;
                // Combine: one wave per (row, head), fixed chunk order → deterministic reduction.
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "attention_split_combine",
                    heads as u32 * 32,
                    32,
                    args![
                        arg_ptr(pm.ptr),
                        arg_ptr(pl.ptr),
                        arg_ptr(pacc.ptr),
                        arg_ptr(dd_ptr),
                        arg_i32(rows as i32),
                        arg_i32(n_head as i32),
                        arg_i32(head_dim as i32),
                        arg_i32(n_chunks as i32),
                    ],
                )?;
            } else {
                // One 32-lane WAVE per (row, head): grid = rows*n_head blocks of 32 threads. The
                // kernel reads `blockIdx.x` as the head index, so pass heads*32 with block=32.
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "attention",
                    rows * n_head * 32,
                    32,
                    args![
                        arg_ptr(bq_ptr),
                        arg_ptr(bk_ptr),
                        arg_ptr(bv_ptr),
                        arg_ptr(dd_ptr),
                        arg_i32(rows as i32),
                        arg_i32(kv_len as i32),
                        arg_i32(n_head as i32),
                        arg_i32(n_kv as i32),
                        arg_i32(head_dim as i32),
                        arg_f32(scale),
                        arg_i32(pos as i32),
                        arg_i32(mt),
                        arg_i32(swa),
                    ],
                )?;
            }
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::GatedAct {
            gate,
            up,
            dst,
            rows,
            nff,
            act,
            up_off,
            up_stride,
            gate_stride,
            gate_block_width,
        } => {
            ctx.ensure_device(gate, g, bindings)?;
            ctx.ensure_device(up, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * nff as usize);
            let bg = ctx.dev[gate.0 as usize].as_ref().unwrap();
            let bu = ctx.dev[up.0 as usize].as_ref().unwrap();
            let at: i32 = match act {
                infr_core::graph::Activation::Silu => 0,
                infr_core::graph::Activation::Gelu => 1,
                infr_core::graph::Activation::Sigmoid => 2,
            };
            dispatch_1d(
                pipelines,
                ctx.stream,
                "gated_act",
                rows * nff,
                256,
                args![
                    arg_ptr(bg.ptr),
                    arg_ptr(bu.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(nff as i32),
                    arg_i32(at),
                    arg_i32(up_off as i32),
                    arg_i32(up_stride as i32),
                    arg_i32(gate_stride as i32),
                    arg_i32(gate_block_width as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::GatedActFused {
            gu,
            dst,
            rows,
            nff,
            act,
        } => {
            ctx.ensure_device(gu, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * nff as usize);
            let bgu = ctx.dev[gu.0 as usize].as_ref().unwrap();
            let at: i32 = match act {
                infr_core::graph::Activation::Silu => 0,
                infr_core::graph::Activation::Gelu => 1,
                infr_core::graph::Activation::Sigmoid => 2,
            };
            dispatch_1d(
                pipelines,
                ctx.stream,
                "gated_act",
                rows * nff,
                256,
                args![
                    arg_ptr(bgu.ptr),
                    arg_ptr(bgu.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(nff as i32),
                    arg_i32(at),
                    arg_i32(nff as i32),
                    arg_i32((2 * nff) as i32),
                    arg_i32((2 * nff) as i32),
                    arg_i32(0),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Add { a, b, dst, n } => {
            ctx.ensure_device(a, g, bindings)?;
            ctx.ensure_device(b, g, bindings)?;
            let dd = ctx.zero_dev(n as usize);
            let ba = ctx.dev[a.0 as usize].as_ref().unwrap();
            let bb = ctx.dev[b.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "add",
                n,
                256,
                args![
                    arg_ptr(ba.ptr),
                    arg_ptr(bb.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::AddBias {
            x,
            bias,
            dst,
            rows,
            n,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(bias, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bb = ctx.dev[bias.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "add_bias",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(bb.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Scale { x, dst, s, n } => {
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "scale",
                n,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(dd.ptr),
                    arg_f32(s),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::MulVec {
            x,
            vec,
            dst,
            rows,
            n,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(vec, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bv = ctx.dev[vec.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "mul_vec",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(bv.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Softcap { x, dst, cap, n } => {
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(n as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "softcap",
                n,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(dd.ptr),
                    arg_f32(cap),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Copy {
            src,
            src_off,
            dst,
            dst_off,
            n,
        } => {
            ctx.ensure_device(src, g, bindings)?;
            // `dst` is a PRE-EXISTING tensor: `Copy` writes only the [dst_off, dst_off+n) slice and
            // must preserve the rest (matches the CPU reference, which copies into `vals[dst]`).
            // `ensure_device` allocates the full tensor extent (`numel`, zero-filled) if `dst` is
            // unproduced, or returns the already-produced buffer — never a wrong-sized fresh zero.
            ctx.ensure_device(dst, g, bindings)?;
            let dst_ptr = ctx.dev[dst.0 as usize].as_ref().unwrap().ptr;
            // Aliasing (src == dst): stage the source through a temp so the in-place copy can't
            // race the read (the CPU reference clones the read window for the same reason).
            let staged = ctx.stage_if_aliased(src, dst);
            let src_ptr = staged
                .as_ref()
                .map(|b| b.ptr)
                .unwrap_or_else(|| ctx.dev[src.0 as usize].as_ref().unwrap().ptr);
            dispatch_1d(
                pipelines,
                ctx.stream,
                "copy",
                n,
                256,
                args![
                    arg_ptr(src_ptr),
                    arg_i32(src_off as i32),
                    arg_ptr(dst_ptr),
                    arg_i32(dst_off as i32),
                    arg_i32(n as i32),
                ],
            )?;
        }
        Op::CopyStrided {
            src,
            src_off,
            src_stride,
            dst,
            dst_off,
            dst_stride,
            rows,
            n,
        } => {
            ctx.ensure_device(src, g, bindings)?;
            // See `Op::Copy`: write the strided rows in place into the full-extent, content-
            // preserving `dst` buffer instead of a wrong-sized fresh zero. The old
            // `rows*(dst_off+n+dst_stride)` sizing did not match a real row-major tensor and
            // dropped prior content on a partial/scatter update.
            ctx.ensure_device(dst, g, bindings)?;
            let dst_ptr = ctx.dev[dst.0 as usize].as_ref().unwrap().ptr;
            let staged = ctx.stage_if_aliased(src, dst);
            let src_ptr = staged
                .as_ref()
                .map(|b| b.ptr)
                .unwrap_or_else(|| ctx.dev[src.0 as usize].as_ref().unwrap().ptr);
            dispatch_1d(
                pipelines,
                ctx.stream,
                "copy_strided",
                rows,
                256,
                args![
                    arg_ptr(src_ptr),
                    arg_i32(src_off as i32),
                    arg_i32(src_stride as i32),
                    arg_ptr(dst_ptr),
                    arg_i32(dst_off as i32),
                    arg_i32(dst_stride as i32),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
        }
        Op::EmbedGather {
            ids,
            table,
            dst,
            rows,
            ne,
            scale,
        } => {
            let (kname, wptr) =
                if let Some((_, _, _, embed_k)) = native_decode_fmt(g.desc(table).dtype) {
                    // Native decode of the embedding table — avoids caching the whole (large) table
                    // as f16 in VRAM (the token_embd bank is a major VRAM cost on big models).
                    (embed_k, ctx.ensure_device(table, g, bindings)?)
                } else {
                    (
                        "embed_gather",
                        ctx.dequant_weight_or_cache(table, g, bindings)?,
                    )
                };
            ctx.ensure_device(ids, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * ne as usize);
            let bid = ctx.dev[ids.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                kname,
                rows,
                256,
                args![
                    arg_ptr(bid.ptr),
                    arg_ptr(wptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(ne as i32),
                    arg_f32(scale),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::Argmax { x, dst, n, rows } => {
            ctx.ensure_device(x, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            // One block per row; the block reduces the vocab argmax across a wave.
            dispatch_grid(
                pipelines,
                ctx.stream,
                "argmax",
                rows,
                1,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::ArgmaxProb { .. } => return Err(be("ArgmaxProb: Phase 2")),
        Op::Sample { .. } => return Err(be("Sample: Phase 2")),

        Op::MoeFfn {
            x,
            router_x,
            router,
            gate_exps,
            up_exps,
            down_exps,
            down_scale,
            dst,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            scale,
            act,
            gating,
            norm_w,
            weight_before,
            fused_gate_up,
            ep_band: _ep,
        } => {
            // Router weight [n_expert, ne] (dequantized to f16 and cached — the SAME handle
            // fed to the GEMV below; the previous code discarded it and softmaxed the raw
            // router_x row, selecting bogus "expert" indices out past the expert banks).
            let rw = ctx.dequant_weight_or_cache(router, g, bindings)?;

            // Native in-kernel expert decode (Phase-3 for MoE): when the gate/up/down banks are
            // covered quant formats, feed the RAW quant bytes to `moe_ffn_expert_<gu>_<dn>` and
            // decode per-block on the fly — NO f16 cache is materialized, so a big quantized MoE
            // fits in VRAM (footprint ≈ quant size, vs ~3.5× that as an f16 cache). Gate & up must
            // share a format (same tensor when fused; every GGUF stores them at the same type);
            // down may differ (Q4_K_M packs down as Q6_K). Any uncovered bank → the whole expert
            // falls back to the dequant→f16 `moe_ffn_expert` path so nothing breaks.
            let gate_dt = g.desc(gate_exps).dtype;
            let up_dt = g.desc(up_exps).dtype;
            let down_dt = g.desc(down_exps).dtype;
            let native = moe_native_fmt(gate_dt)
                .zip(moe_native_fmt(down_dt))
                .filter(|_| fused_gate_up || up_dt == gate_dt);

            let (gw_ptr, uw_ptr, dw_ptr) = if native.is_some() {
                // Raw quant device pointers (the bound buffers) — no dequant, no f16 cache.
                let gw = ctx.ensure_device(gate_exps, g, bindings)?;
                let uw = if fused_gate_up {
                    gw
                } else {
                    ctx.ensure_device(up_exps, g, bindings)?
                };
                let dw = ctx.ensure_device(down_exps, g, bindings)?;
                (gw, uw, dw)
            } else {
                let gw = ctx.dequant_weight_or_cache(gate_exps, g, bindings)?;
                let uw = if fused_gate_up {
                    gw
                } else {
                    ctx.dequant_weight_or_cache(up_exps, g, bindings)?
                };
                let dw = ctx.dequant_weight_or_cache(down_exps, g, bindings)?;
                (gw, uw, dw)
            };

            let neu = ne as usize;
            let nexp = n_expert as usize;
            let nu = n_used as usize;
            let nfu = n_ff_exp as usize;

            // `x` (and `router_x`, usually the same handle) carry `rows` token rows of `ne`.
            let x_ptr = ctx.ensure_device(x, g, bindings)?;
            let rx_ptr = if router_x != x {
                ctx.ensure_device(router_x, g, bindings)?
            } else {
                x_ptr
            };
            let rows = g.desc(x).numel() / neu;

            // Per-expert down-projection output scale (diffusion-gemma); 1.0 = none.
            let dsc_vals: Vec<f32> = match down_scale {
                Some(sid) => ctx.host_vals(sid, g, bindings)?.to_vec(),
                None => vec![1.0f32; nexp],
            };

            // Router logits = router · router_x, one dot per expert: reuse the linear_f16
            // GEMV to produce [rows, n_expert], then read them back for host-side gating.
            let logits_dev = ctx.zero_dev(rows * nexp);
            dispatch_1d(
                pipelines,
                ctx.stream,
                "linear_f16",
                (rows as u32) * 256,
                256,
                args![
                    arg_ptr(rx_ptr),
                    arg_ptr(rw),
                    arg_ptr(logits_dev.ptr),
                    arg_i32(rows as i32),
                    arg_i32(ne as i32),
                    arg_i32(n_expert as i32),
                ],
            )?;
            unsafe {
                ffi::hipStreamSynchronize(ctx.stream);
            }
            let logits_all: Vec<f32> = {
                let raw = read_bytes(&logits_dev, ctx.stream);
                bytemuck::cast_slice::<u8, f32>(&raw).to_vec()
            };

            let at: i32 = match act {
                infr_core::graph::Activation::Silu => 0,
                infr_core::graph::Activation::Gelu => 1,
                infr_core::graph::Activation::Sigmoid => 2,
            };
            let wb_flag: i32 = if weight_before { 1 } else { 0 };
            // Per-expert byte strides in the (f16) expert banks. Fused gate/up packs BOTH
            // roles per expert as [2*n_ff_exp, ne] (gate rows first, up second), so its expert
            // stride is DOUBLE the split-tensor stride.
            let ge_stride = if fused_gate_up {
                2 * nfu * neu
            } else {
                nfu * neu
            };

            // Int8-activation dp4a expert path (Slice 20): when the gate/up/down banks are covered
            // quant formats, decode+dot them via the int8 machinery (`quant_i8_32` + the `i8acc_*`
            // GEMVs) instead of the Phase-3 per-element f16 round-trip, parallelized across nff/ne
            // output rows. Scratch is drawn ONCE from the pool: the token's int8 input `qx_x`/`xs_x`
            // (re-quantized per token, reused across every expert + both gate & up) and the per-expert
            // activation `h_buf` → `hq`/`hs` (overwritten each expert; the stream serializes the
            // gate_up → quant_h → down chain so the reuse never races). All fully written before read,
            // so `zero = false`.
            let use_i8 = native.is_some() && moe_i8_enabled();
            let (qx_x, xs_x, h_buf, hq, hs) = if use_i8 {
                (
                    Some(ctx.pool_buf(neu.max(1), false)),
                    Some(ctx.pool_buf((neu / 32 * 4).max(1), false)),
                    Some(ctx.pool_buf((nfu * 4).max(1), false)),
                    Some(ctx.pool_buf(nfu.max(1), false)),
                    Some(ctx.pool_buf((nfu / 32 * 4).max(1), false)),
                )
            } else {
                (None, None, None, None, None)
            };

            let dd = ctx.zero_dev(rows * neu);
            for row in 0..rows {
                let logits = &logits_all[row * nexp..row * nexp + nexp];
                // Gating: softmax over experts (qwen3moe/…) or per-expert sigmoid (llama4).
                let probs: Vec<f32> = match gating {
                    infr_core::graph::MoeGating::Softmax => {
                        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
                        let sum: f32 = exps.iter().sum();
                        exps.iter().map(|v| v / sum).collect()
                    }
                    infr_core::graph::MoeGating::Sigmoid => {
                        logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
                    }
                };
                let mut idx: Vec<usize> = (0..nexp).collect();
                idx.sort_unstable_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                idx.truncate(nu);
                // `norm_w`: renormalize the selected weights to sum to 1 before scaling
                // (softmax MoE); llama4 uses the raw sigmoid prob × scale (no renorm).
                let wsum: f32 = if norm_w {
                    idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20)
                } else {
                    1.0
                };
                let x_row = unsafe { (x_ptr as *mut u8).add(row * neu * 4) as *mut c_void };
                let dst_row = unsafe { (dd.ptr as *mut u8).add(row * neu * 4) as *mut c_void };
                // Int8 path: quantize this token's input row ONCE (reused across all experts).
                if use_i8 {
                    dispatch_1d(
                        pipelines,
                        ctx.stream,
                        "quant_i8_32",
                        (neu / 32) as u32,
                        256,
                        args![
                            arg_ptr(x_row),
                            arg_ptr(qx_x.as_ref().unwrap().ptr),
                            arg_ptr(xs_x.as_ref().unwrap().ptr),
                            arg_i32(1),
                            arg_i32(ne as i32),
                        ],
                    )?;
                }
                for &ei in &idx {
                    let w = probs[ei] / wsum * scale;
                    // Per-expert pointers. Native path: byte offset = (element_offset / qpb) * bpb
                    // into the RAW quant bank (every element offset is a multiple of the block size,
                    // since the per-expert stride is a whole number of `ne`-wide rows and `ne` is a
                    // multiple of the block elem count). Fallback path: element_offset * 2 into the
                    // f16 cache. Gate/up share the `gu` format+geometry; down carries the `dn` one.
                    let (gs, us, ds, kname) =
                        if let Some(((gu, gu_qpb, gu_bpb), (dn, dn_qpb, dn_bpb))) = native {
                            let gs = unsafe {
                                (gw_ptr as *mut u8).add((ei * ge_stride / gu_qpb) * gu_bpb)
                                    as *mut c_void
                            };
                            let us = if fused_gate_up {
                                unsafe {
                                    (gw_ptr as *mut u8)
                                        .add(((ei * ge_stride + nfu * neu) / gu_qpb) * gu_bpb)
                                        as *mut c_void
                                }
                            } else {
                                unsafe {
                                    (uw_ptr as *mut u8).add((ei * nfu * neu / gu_qpb) * gu_bpb)
                                        as *mut c_void
                                }
                            };
                            let ds = unsafe {
                                (dw_ptr as *mut u8).add((ei * neu * nfu / dn_qpb) * dn_bpb)
                                    as *mut c_void
                            };
                            (gs, us, ds, moe_expert_kernel(gu, dn))
                        } else {
                            let gs = unsafe {
                                (gw_ptr as *mut u8).add(ei * ge_stride * 2) as *mut c_void
                            };
                            let us = if fused_gate_up {
                                unsafe {
                                    (gw_ptr as *mut u8).add((ei * ge_stride + nfu * neu) * 2)
                                        as *mut c_void
                                }
                            } else {
                                unsafe {
                                    (uw_ptr as *mut u8).add(ei * nfu * neu * 2) as *mut c_void
                                }
                            };
                            let ds = unsafe {
                                (dw_ptr as *mut u8).add(ei * neu * nfu * 2) as *mut c_void
                            };
                            (gs, us, ds, "moe_ffn_expert")
                        };
                    let dsc = dsc_vals.get(ei).copied().unwrap_or(1.0);
                    if let (true, Some(((gu, _, _), (dn, _, _)))) = (use_i8, native) {
                        // int8 dp4a: gate+up+activation (→ h_buf), quant h, then down (accumulate).
                        // The routing weight is folded into h via wg/wo (same split as the fused
                        // f16 kernel: `weight_before` applies it to the gate/up inputs, else output).
                        let wg = if weight_before { w } else { 1.0 };
                        let wo = if weight_before { 1.0 } else { w };
                        let h_ptr = h_buf.as_ref().unwrap().ptr;
                        dispatch_grid(
                            pipelines,
                            ctx.stream,
                            moe_gate_up_i8_kernel(gu),
                            n_ff_exp,
                            1,
                            32,
                            args![
                                arg_ptr(qx_x.as_ref().unwrap().ptr),
                                arg_ptr(xs_x.as_ref().unwrap().ptr),
                                arg_ptr(gs),
                                arg_ptr(us),
                                arg_ptr(h_ptr),
                                arg_i32(ne as i32),
                                arg_i32(n_ff_exp as i32),
                                arg_i32(at),
                                arg_f32(wg),
                                arg_f32(wo),
                                arg_f32(dsc),
                            ],
                        )?;
                        dispatch_1d(
                            pipelines,
                            ctx.stream,
                            "quant_i8_32",
                            (nfu / 32) as u32,
                            256,
                            args![
                                arg_ptr(h_ptr),
                                arg_ptr(hq.as_ref().unwrap().ptr),
                                arg_ptr(hs.as_ref().unwrap().ptr),
                                arg_i32(1),
                                arg_i32(n_ff_exp as i32),
                            ],
                        )?;
                        dispatch_grid(
                            pipelines,
                            ctx.stream,
                            moe_down_i8_kernel(dn),
                            ne,
                            1,
                            32,
                            args![
                                arg_ptr(hq.as_ref().unwrap().ptr),
                                arg_ptr(hs.as_ref().unwrap().ptr),
                                arg_ptr(ds),
                                arg_ptr(dst_row),
                                arg_i32(ne as i32),
                                arg_i32(n_ff_exp as i32),
                            ],
                        )?;
                    } else {
                        dispatch_1d(
                            pipelines,
                            ctx.stream,
                            kname,
                            n_ff_exp,
                            256,
                            args![
                                arg_ptr(x_row),
                                arg_ptr(gs),
                                arg_ptr(us),
                                arg_ptr(ds),
                                arg_ptr(dst_row),
                                arg_i32(ne as i32),
                                arg_i32(n_ff_exp as i32),
                                arg_i32(at),
                                arg_f32(w),
                                arg_f32(dsc),
                                arg_i32(wb_flag),
                            ],
                        )?;
                    }
                }
            }
            ctx.dev[dst.0 as usize] = Some(dd);
        }

        Op::Conv1dSilu {
            x,
            weight,
            state,
            dst,
            rows,
            channels,
            kernel,
        } => {
            let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(state, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * channels as usize);
            let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
            let bst = ctx.dev[state.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "conv1d_silu",
                rows,
                256,
                args![
                    arg_ptr(bx.ptr),
                    arg_ptr(wptr),
                    arg_ptr(bst.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(channels as i32),
                    arg_i32(kernel as i32),
                ],
            )?;
            // Host-side state update: the returned state is the trailing `km1` columns of the
            // virtual sequence seq = [state ‖ x] (km1 warmup columns then `rows` input columns),
            // i.e. new_state[j] = seq[rows + j] for j in 0..km1. This chains correctly for any
            // `rows`: for `rows >= km1` all km1 columns come from the last km1 input rows; for
            // `rows < km1` the leading entries carry over from the old state tail. For `rows == 1`
            // it reduces to the old "drop oldest, append x[0]" shift (decode is bit-identical).
            let km1 = (kernel - 1) as usize;
            let ch = channels as usize;
            let rows_u = rows as usize;
            // Read the OLD state and the conv input DIRECTLY from their device buffers — NOT via
            // `host_vals`, which caches by tensor id. `x` (`dn_qkvbuf`) and `state` (`k_cache[l]`)
            // are REUSED across every DeltaNet layer, so a cached read would hand back an earlier
            // layer's stale content and corrupt the rolling conv history for all deeper layers
            // (the first layer would carry correctly, later ones would not — the classic
            // "layer 0 fine, layer 2 diverges in decode" symptom).
            let hs = {
                let bst = ctx.dev[state.0 as usize].as_ref().unwrap();
                bytes_to_f32(&read_bytes(bst, ctx.stream), DType::F32)?
            };
            let hx = {
                let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
                bytes_to_f32(&read_bytes(bx, ctx.stream), DType::F32)?
            };
            let mut ns = vec![0f32; km1 * ch];
            for j in 0..km1 {
                let idx = rows_u + j; // virtual-sequence index of new_state column j
                for c in 0..ch {
                    ns[j * ch + c] = if idx < km1 {
                        hs[idx * ch + c] // still inside the old state tail
                    } else {
                        hx[(idx - km1) * ch + c] // an input column
                    };
                }
            }
            // Persist the rolling conv history IN PLACE into the bound (persistent) `state` buffer.
            // `state` is an in-place Input (`k_cache[l]`, repurposed as conv state): the end-of-graph
            // writeback skips in-place inputs, so `set_dev`-ing a FRESH buffer here would drop the
            // update and the history would never reach the next graph — the decode conv would read a
            // stale/zero history and diverge after the first token. Write the bound buffer directly.
            let sb = ctx.dev[state.0 as usize].as_ref().unwrap();
            let bytes = bytemuck::cast_slice::<f32, u8>(&ns);
            let n = bytes.len().min(sb.len);
            if n > 0 {
                unsafe {
                    ffi::hipMemcpy(
                        sb.ptr,
                        bytes.as_ptr() as *const c_void,
                        n,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    );
                }
            }
            ctx.dev[dst.0 as usize] = Some(dd);
        }

        Op::DeltaNet {
            q,
            k,
            v,
            b,
            a,
            a_coef,
            dt_bias,
            state,
            dst,
            rows,
            n_khead,
            n_vhead,
            head_k,
            head_v,
            eps,
            src_stride,
            ..
        } => {
            ctx.ensure_device(q, g, bindings)?;
            ctx.ensure_device(k, g, bindings)?;
            ctx.ensure_device(v, g, bindings)?;
            ctx.ensure_device(b, g, bindings)?;
            ctx.ensure_device(a, g, bindings)?;
            let ac = ctx.dequant_weight_or_cache(a_coef, g, bindings)?;
            let dt = ctx.dequant_weight_or_cache(dt_bias, g, bindings)?;
            ctx.ensure_device(state, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n_vhead as usize * head_v as usize);
            let bq = ctx.dev[q.0 as usize].as_ref().unwrap();
            let bk = ctx.dev[k.0 as usize].as_ref().unwrap();
            let bv = ctx.dev[v.0 as usize].as_ref().unwrap();
            let bb = ctx.dev[b.0 as usize].as_ref().unwrap();
            let ba = ctx.dev[a.0 as usize].as_ref().unwrap();
            let bst = ctx.dev[state.0 as usize].as_ref().unwrap();
            let dn_args = args![
                arg_ptr(bq.ptr),
                arg_ptr(bk.ptr),
                arg_ptr(bv.ptr),
                arg_ptr(bb.ptr),
                arg_ptr(ba.ptr),
                arg_ptr(ac),
                arg_ptr(dt),
                arg_ptr(bst.ptr),
                arg_ptr(dd.ptr),
                arg_i32(rows as i32),
                arg_i32(n_khead as i32),
                arg_i32(n_vhead as i32),
                arg_i32(head_k as i32),
                arg_i32(head_v as i32),
                arg_f32(eps),
                arg_i32(src_stride as i32),
            ];
            // Chunked/parallel prefill: DN_CHUNK=16, shared holds 2·C·kd + 2·C·C + 2·C floats
            // (≈18 KiB at kd=128). Use it for rows>1 when that footprint fits the 32 KiB dynamic-LDS
            // ceiling this GPU allows a launch without the MaxDynamicSharedMemorySize opt-in (an
            // over-budget launch silently corrupts LDS rather than erroring). Decode (rows==1) goes to
            // the column-parallel `deltanet_decode` (one block per value head, one thread per value
            // dim — bit-identical to the sequential scan, just spread over n_vhead·head_v threads
            // instead of n_vhead). The sequential per-head scan is the last-resort fallback (rows>1
            // that overflows LDS, or head_v==0).
            const DN_CHUNK: usize = 16;
            let smem_bytes =
                (2 * DN_CHUNK * head_k as usize + 2 * DN_CHUNK * DN_CHUNK + 2 * DN_CHUNK) * 4;
            if rows > 1 && head_v > 0 && smem_bytes <= 32 * 1024 {
                // One block per value head; one thread per value dim (≥ DN_CHUNK so Phase-1's
                // per-token threads and Phase-3's per-column threads are both covered).
                let block = head_v.max(DN_CHUNK as u32);
                dispatch_blocks_smem(
                    pipelines,
                    ctx.stream,
                    "deltanet_chunked",
                    n_vhead,
                    block,
                    smem_bytes as u32,
                    dn_args,
                )?;
            } else if rows == 1 && head_v > 0 {
                // Column-parallel decode: grid.x = n_vhead blocks (via total = n_vhead·block), one
                // thread per value dim d (grid-stride covers head_v > block).
                let block = head_v.clamp(1, 256);
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "deltanet_decode",
                    n_vhead * block,
                    block,
                    dn_args,
                )?;
            } else {
                dispatch_1d(pipelines, ctx.stream, "deltanet", n_vhead, 256, dn_args)?;
            }
            ctx.dev[dst.0 as usize] = Some(dd);
        }

        Op::MoeSharedExpertAdd {
            moe,
            shexp,
            gate,
            dst,
            rows,
            n,
        } => {
            ctx.ensure_device(moe, g, bindings)?;
            ctx.ensure_device(shexp, g, bindings)?;
            ctx.ensure_device(gate, g, bindings)?;
            let dd = ctx.zero_dev(rows as usize * n as usize);
            let bm = ctx.dev[moe.0 as usize].as_ref().unwrap();
            let bs = ctx.dev[shexp.0 as usize].as_ref().unwrap();
            let bg = ctx.dev[gate.0 as usize].as_ref().unwrap();
            dispatch_1d(
                pipelines,
                ctx.stream,
                "moe_shared_expert_add",
                rows,
                256,
                args![
                    arg_ptr(bm.ptr),
                    arg_ptr(bs.ptr),
                    arg_ptr(bg.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
    }
    Ok(())
}
