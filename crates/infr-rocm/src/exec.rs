//! Graph execution: walk ops → resolve bound buffers → dispatch HIP kernels.
//!
//! Covered quant formats (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/IQ2_XXS/
//! IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/TQ1_0/TQ2_0/Q2_0/MXFP4/NVFP4 — ALL 24 weight quants as of
//! R7, see `native_decode_fmt`) are decoded in-kernel from their RAW bytes on the
//! `Linear`/`EmbedGather` paths — no f16 cache, VRAM ≈ quant_size.
//! What is left on the host convert→f16 path (cached by the identity of its bound buffer) is only
//! the DENSE FLOAT dtypes F32/BF16, which have no quant decode at all — F16 is already native via
//! `linear_f16`. `native_decode_is_total_over_every_gguf_weight_dtype` pins that split.

use crate::backend::{bucket_bytes, BufferPool};
use crate::ffi::{self, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};
use crate::kernels::Pipelines;
use half::f16;
use infr_core::backend::{Bindings, GraphPlan, Plan};
use infr_core::error::Result;
use infr_core::graph::{Activation, AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_gguf::dequant;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_int, c_void};
use std::sync::Mutex;

/// Terse local shorthand for the shared backend-error constructor.
use infr_core::error::backend as be;

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
    // Kernel COVERAGE is the decision here; the block geometry is read from the shared decode spec
    // (`infr_core::decode_spec`) instead of re-spelling `(256, 144)` — see also `native_i8_fmt`
    // and `moe_native_fmt`, which read the same table.
    let (lin, emb) = match dt {
        DType::Q8_0 => ("linear_q80", "embed_q80"),
        DType::Q2K => ("linear_q2k", "embed_q2k"),
        DType::Q3K => ("linear_q3k", "embed_q3k"),
        DType::Q4K => ("linear_q4k", "embed_q4k"),
        DType::Q5K => ("linear_q5k", "embed_q5k"),
        DType::Q6K => ("linear_q6k", "embed_q6k"),
        DType::Q5_0 => ("linear_q50", "embed_q50"),
        DType::Q4_0 => ("linear_q40", "embed_q40"),
        DType::Q4_1 => ("linear_q41", "embed_q41"),
        DType::Q5_1 => ("linear_q51", "embed_q51"),
        DType::Iq4Nl => ("linear_iq4nl", "embed_iq4nl"),
        DType::Iq4Xs => ("linear_iq4xs", "embed_iq4xs"),
        DType::Iq2Xxs => ("linear_iq2xxs", "embed_iq2xxs"),
        DType::Iq2Xs => ("linear_iq2xs", "embed_iq2xs"),
        DType::Iq2S => ("linear_iq2s", "embed_iq2s"),
        DType::Iq3Xxs => ("linear_iq3xxs", "embed_iq3xxs"),
        DType::Iq3S => ("linear_iq3s", "embed_iq3s"),
        DType::Iq1S => ("linear_iq1s", "embed_iq1s"),
        DType::Iq1M => ("linear_iq1m", "embed_iq1m"),
        DType::Tq1_0 => ("linear_tq10", "embed_tq10"),
        DType::Tq2_0 => ("linear_tq20", "embed_tq20"),
        DType::Q2_0 => ("linear_q20", "embed_q20"),
        DType::Mxfp4 => ("linear_mxfp4", "embed_mxfp4"),
        DType::Nvfp4 => ("linear_nvfp4", "embed_nvfp4"),
        _ => return None,
    };
    let (elems, bytes) = infr_core::decode_spec::block_layout(dt);
    Some((elems, bytes, lin, emb))
}

/// Int8-activation dp4a GEMV kernel (Phase 4) for a covered dtype: `(bytes_per_block, kernel)`.
/// The activation row is quantized to int8 once (`quant_i8_32`) and integer-dotted against the
/// decoded weight codes (scale-after) — dropping the Phase-3 per-element f16 round-trip. Returns
/// `None` for uncovered formats (they keep the Phase-3 native decode / dequant→f16 fallback), or
/// when `INFR_ROCM_NO_I8` (config `kernels.rocm.i8`, POSITIVE polarity — presence of the env key,
/// including `=0`, clears it) selects the Phase-3 path for A/B benchmarking.
fn native_i8_fmt(dt: DType, rocm: &infr_core::config::RocmCfg) -> Option<(usize, &'static str)> {
    if !rocm.i8 {
        return None;
    }
    let kernel = match dt {
        DType::Q8_0 => "linear_i8_q80",
        DType::Q2K => "linear_i8_q2k",
        DType::Q3K => "linear_i8_q3k",
        DType::Q4K => "linear_i8_q4k",
        DType::Q5K => "linear_i8_q5k",
        DType::Q6K => "linear_i8_q6k",
        DType::Q5_0 => "linear_i8_q50",
        DType::Q4_0 => "linear_i8_q40",
        DType::Q4_1 => "linear_i8_q41",
        DType::Q5_1 => "linear_i8_q51",
        DType::Iq4Nl => "linear_i8_iq4nl",
        DType::Iq4Xs => "linear_i8_iq4xs",
        DType::Iq2Xxs => "linear_i8_iq2xxs",
        DType::Iq2Xs => "linear_i8_iq2xs",
        DType::Iq2S => "linear_i8_iq2s",
        DType::Iq3Xxs => "linear_i8_iq3xxs",
        DType::Iq3S => "linear_i8_iq3s",
        DType::Iq1S => "linear_i8_iq1s",
        DType::Iq1M => "linear_i8_iq1m",
        DType::Tq1_0 => "linear_i8_tq10",
        DType::Tq2_0 => "linear_i8_tq20",
        DType::Q2_0 => "linear_i8_q20",
        DType::Mxfp4 => "linear_i8_mxfp4",
        DType::Nvfp4 => "linear_i8_nvfp4",
        _ => return None,
    };
    Some((infr_core::decode_spec::block_layout(dt).1, kernel))
}

/// Output rows ONE wave of an int8 decode GEMV computes — the divisor on the launch's `grid.x`.
///
/// F4 (`kernels.rs`, `I8_MROW`): a wave that owns one output row has a single weight stream in
/// flight and nothing to overlap its latency with. The converted kernels take `I8_MROW` consecutive
/// rows instead, fetch the shared activation quad once, and issue all the rows' weight loads before
/// any of the math — worth 772 → 930 GB/s on the Q4_K lm_head shape. Every OTHER kernel in the
/// family still owns exactly one row and must keep `grid.x == out_f`, so this is a per-kernel fact,
/// not a global one. The 2 here MUST equal `I8_MROW` in the kernel source —
/// `mrow_matches_the_kernel_source` pins that, because a mismatch silently skips or double-writes
/// output rows rather than failing anything loudly.
fn i8_gemv_mrow(kernel: &str) -> u32 {
    match kernel {
        "linear_i8_q4k" | "linear_i8_q5k" => 2,
        _ => 1,
    }
}

/// Dequant-to-f16 kernel name (`deqf16_*`, kernels.rs `DEQUANT_F16`) for a covered dtype — the
/// weight decoder feeding the Slice-26 rocBLAS f16 prefill GEMM. Same covered set as
/// [`native_decode_fmt`], which after R7 is ALL 24 weight quants; `None` keeps a format off it
/// (only the dense float dtypes land there now, and they have nothing to decode).
fn deqf16_fmt(dt: DType) -> Option<&'static str> {
    match dt {
        DType::Q8_0 => Some("deqf16_q80"),
        DType::Q2K => Some("deqf16_q2k"),
        DType::Q3K => Some("deqf16_q3k"),
        DType::Q4K => Some("deqf16_q4k"),
        DType::Q5K => Some("deqf16_q5k"),
        DType::Q6K => Some("deqf16_q6k"),
        DType::Q5_0 => Some("deqf16_q50"),
        DType::Q4_0 => Some("deqf16_q40"),
        DType::Q4_1 => Some("deqf16_q41"),
        DType::Q5_1 => Some("deqf16_q51"),
        DType::Iq4Nl => Some("deqf16_iq4nl"),
        DType::Iq4Xs => Some("deqf16_iq4xs"),
        DType::Iq2Xxs => Some("deqf16_iq2xxs"),
        DType::Iq2Xs => Some("deqf16_iq2xs"),
        DType::Iq2S => Some("deqf16_iq2s"),
        DType::Iq3Xxs => Some("deqf16_iq3xxs"),
        DType::Iq3S => Some("deqf16_iq3s"),
        DType::Iq1S => Some("deqf16_iq1s"),
        DType::Iq1M => Some("deqf16_iq1m"),
        DType::Tq1_0 => Some("deqf16_tq10"),
        DType::Tq2_0 => Some("deqf16_tq20"),
        DType::Q2_0 => Some("deqf16_q20"),
        DType::Mxfp4 => Some("deqf16_mxfp4"),
        DType::Nvfp4 => Some("deqf16_nvfp4"),
        _ => None,
    }
}

/// Explicit tile override for A/B benchmarking (`INFR_ROCM_WMMA_TILE=RxC`, one of 1x1/2x1/2x2).
/// `None` when unset → the shape-driven auto tier in [`wmma_tile`] is used.
fn wmma_tile_forced(rocm: &infr_core::config::RocmCfg) -> Option<(u32, u32)> {
    // The env layer already trimmed the value (`opt_text_trimmed`); an unrecognized spelling is
    // treated as unset here, exactly as the `_ => None` arm always did.
    match rocm.wmma_tile.as_deref() {
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
fn wmma_tile(out_f: u32, rocm: &infr_core::config::RocmCfg) -> (u32, u32) {
    if let Some(t) = wmma_tile_forced(rocm) {
        return t;
    }
    if out_f >= 1024 {
        (2, 2)
    } else {
        (2, 1)
    }
}

/// Flash-decoding (split-KV) attention chunk policy for `attention_split_partial`: aim ~32
/// chunks/head, each 64..512 keys. Same window as Vulkan's `attn_partial`, but CEIL-rounded — a
/// real (small) divergence from Vulkan's floor, preserved here because neither has been shown
/// better and changing it re-shapes every decode dispatch. See
/// [`infr_core::tier::ChunkRounding`].
const ATTN_SPLIT: infr_core::tier::AttnSplitCfg = infr_core::tier::AttnSplitCfg {
    target_chunks: 64,
    min_chunk: 32,
    max_chunk: 512,
    rounding: infr_core::tier::ChunkRounding::Up,
};

/// Query rows one WAVE of `attention_prefill_flash` owns — mirrors the kernel's `ATTN_FLASH_QPW`,
/// which is compile-time there because `acc[u][c]` is register state. Swept on the pp512 shape at
/// a fixed `br` of 16: 2 -> 279 us/layer, 4 -> 313, 8 -> 441.
const ATTN_FLASH_QPW: usize = 2;

/// LDS budget for one `attention_prefill_flash` workgroup. Half of gfx1100's 64 KiB per CU, so two
/// workgroups stay co-resident; a `head_dim` that cannot be tiled inside it falls back to the plain
/// `attention` kernel rather than trading occupancy away for the tiling.
const ATTN_FLASH_LDS: usize = 32 * 1024;

/// The chosen shape of one `attention_prefill_flash` workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttnFlashTiling {
    /// Waves per workgroup (block = `nw * 32` threads).
    nw: usize,
    /// Keys per KV tile, `<= 32` — the kernel gives one LANE one key, so a wider tile has nowhere
    /// to put the extra keys.
    bc: usize,
}

impl AttnFlashTiling {
    /// Query rows one workgroup owns.
    fn br(&self) -> usize {
        self.nw * ATTN_FLASH_QPW
    }
    /// Dynamic LDS bytes: the `[br][head_dim]` f32 query tile plus the K and V half tiles at the
    /// bank-conflict-avoiding `head_dim + 2` row stride. Must match the kernel's own arithmetic.
    fn smem(&self, head_dim: usize) -> usize {
        self.br() * head_dim * 4 + 2 * self.bc * (head_dim + 2) * 2
    }
}

/// Pick the flash-prefill workgroup shape for `head_dim`, or `None` to keep the plain kernel.
///
/// Three hard requirements, each a way the kernel would be wrong or pointless:
///
/// 1. **`head_dim % 32 == 0`.** Three separate constraints land here, and the strictest wins. (a) The score
///    dot is rebuilt in the plain kernel's exact reduction tree — 32 lane-group partials, the
///    group for `t` taking `d = t, t+32, …` — so a head dim that does not fill every group whole
///    would need the plain kernel's `d < head_dim` guard to land on the same tree, and the point
///    of the rebuild is that it is exact. (b) The LDS K/V row stride is `head_dim + 2` HALVES =
///    `head_dim/2 + 1` 4-byte words, coprime to the 32 banks precisely when it is ODD, i.e. when
///    `head_dim` is a multiple of 4. (c) The tile staging reads K/V eight halves at a time as a
///    `uint4`, which needs each cache row 16 B-aligned off the `hipMalloc` base — a multiple of 8.
///    Every routed head dim is 64/128/256; anything else keeps the plain kernel.
/// 2. **`head_dim <= 256`.** The kernel's `ATTN_FLASH_MAXP2 = 4` output-dim PAIRS per lane. It is
///    sized to the models rather than to the plain kernel's 512-wide bound because
///    `acc[QPW][MAXP2]` is register state, allocated in full whether a head dim reaches it or not;
///    no model routed here has a head dim above gemma-3's 256.
/// 3. **The tile fits [`ATTN_FLASH_LDS`].** `bc` drops to 16 for the wide (gemma-3's 256) head
///    dims, then `nw` is the largest power of two that still fits — biggest `br` wins, because `br`
///    is exactly the factor by which global K/V traffic falls.
fn attn_flash_tiling(head_dim: usize) -> Option<AttnFlashTiling> {
    if head_dim == 0 || !head_dim.is_multiple_of(32) || head_dim > 256 {
        return None;
    }
    let bc = if head_dim <= 128 { 32 } else { 16 };
    [8usize, 4, 2, 1]
        .into_iter()
        .map(|nw| AttnFlashTiling { nw, bc })
        .find(|t| t.smem(head_dim) <= ATTN_FLASH_LDS)
}

/// The pair of P6 batched-prefetch DECODE attention entry points instantiated for one lane count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttnPfKernels {
    /// Single-wave-per-head kernel (`n_chunks == 1`, i.e. short context).
    plain: &'static str,
    /// Split-KV pass 1 (`n_chunks > 1`, every real depth).
    split_partial: &'static str,
}

/// Pick the batched-prefetch decode attention kernels for `head_dim`, or `None` to keep the generic
/// ones.
///
/// The selector is `npl = ceil(head_dim / 32)` — the head dims ONE LANE owns, which sizes the
/// kernels' register staging buffer and so has to be a template parameter (a runtime subscript
/// sinks that buffer to scratch). Only `npl ∈ {2, 4, 8}` —
/// head_dim 33..=64, 97..=128 and 225..=256 — are instantiated; those cover every head_dim the
/// supported models use. Anything else falls back, which is a perf choice and never a correctness
/// one: the two kernels compute bit-identical results (`attn_pf_falls_back_for_an_uninstantiated_\
/// lane_count` runs the fallback against the CPU oracle).
///
/// `head_dim == 0` is rejected so a degenerate graph cannot select a kernel whose `qreg` loop would
/// mask every lane off.
fn attn_pf_npl(head_dim: usize) -> Option<AttnPfKernels> {
    if head_dim == 0 {
        return None;
    }
    match head_dim.div_ceil(32) {
        2 => Some(AttnPfKernels {
            plain: "attention_pf_npl2",
            split_partial: "attention_split_partial_pf_npl2",
        }),
        4 => Some(AttnPfKernels {
            plain: "attention_pf_npl4",
            split_partial: "attention_split_partial_pf_npl4",
        }),
        8 => Some(AttnPfKernels {
            plain: "attention_pf_npl8",
            split_partial: "attention_split_partial_pf_npl8",
        }),
        _ => None,
    }
}

/// Matrix-core (WMMA) int8 prefill GEMM kernel (Phase 5, Slice-25 RM×CN-tiled) for a covered dtype.
/// Routed only for `m > 1` (prefill); decode (`m == 1`) stays on the `linear_i8_*` GEMV, which WMMA
/// can't help. Same int8 precision as `native_i8_fmt` (identical activation quant + weight codes),
/// and bit-identical f32 accumulation order to the Slice-15 kernel, so it holds the blessed goldens.
/// `INFR_ROCM_NO_WMMA` forces the GEMV path for A/B benchmarking. Returns `(kernel_name, RM, CN)`, or
/// `None` when the int8 path itself is disabled (`INFR_ROCM_NO_I8`).
fn native_wmma_fmt(
    dt: DType,
    out_f: u32,
    rocm: &infr_core::config::RocmCfg,
) -> Option<(&'static str, u32, u32)> {
    if rocm.no_wmma || !rocm.i8 {
        return None;
    }
    // Slice-27: Q4_K prefill defaults to the software-pipelined (prefetched double-buffered nibble)
    // 2x1 kernel — its overlapped load/decode/WMMA schedule beats the Slice-25 auto-tier on EVERY
    // representative shape in the isolated-GEMM micro-bench (`examples/wmma_bench`), including the
    // wide-N GEMMs where the un-pipelined 2x2 previously won, so it supersedes the 2x1/2x2 split for
    // Q4_K. Bit-identical math to `wmma_i8_q4k_2x1` (goldens unmoved). `INFR_ROCM_NO_PIPE=1` falls
    // back to the Slice-25 auto-tier for A/B benchmarking.
    if dt == DType::Q4K && rocm.pipe {
        return Some(("wmma_i8_q4k_pipe_2x1", 2, 1));
    }
    let (rm, cn) = wmma_tile(out_f, rocm);
    let name = match (dt, rm, cn) {
        (DType::Q8_0, 1, 1) => "wmma_i8_q80_1x1",
        (DType::Q8_0, 2, 2) => "wmma_i8_q80_2x2",
        (DType::Q8_0, _, _) => "wmma_i8_q80_2x1",
        (DType::Q2K, 1, 1) => "wmma_i8_q2k_1x1",
        (DType::Q2K, 2, 2) => "wmma_i8_q2k_2x2",
        (DType::Q2K, _, _) => "wmma_i8_q2k_2x1",
        (DType::Q3K, 1, 1) => "wmma_i8_q3k_1x1",
        (DType::Q3K, 2, 2) => "wmma_i8_q3k_2x2",
        (DType::Q3K, _, _) => "wmma_i8_q3k_2x1",
        (DType::Q4K, 1, 1) => "wmma_i8_q4k_1x1",
        (DType::Q4K, 2, 2) => "wmma_i8_q4k_2x2",
        (DType::Q4K, _, _) => "wmma_i8_q4k_2x1",
        (DType::Q5K, 1, 1) => "wmma_i8_q5k_1x1",
        (DType::Q5K, 2, 2) => "wmma_i8_q5k_2x2",
        (DType::Q5K, _, _) => "wmma_i8_q5k_2x1",
        (DType::Q6K, 1, 1) => "wmma_i8_q6k_1x1",
        (DType::Q6K, 2, 2) => "wmma_i8_q6k_2x2",
        (DType::Q6K, _, _) => "wmma_i8_q6k_2x1",
        (DType::Q5_0, 1, 1) => "wmma_i8_q50_1x1",
        (DType::Q5_0, 2, 2) => "wmma_i8_q50_2x2",
        (DType::Q5_0, _, _) => "wmma_i8_q50_2x1",
        (DType::Q4_0, 1, 1) => "wmma_i8_q40_1x1",
        (DType::Q4_0, 2, 2) => "wmma_i8_q40_2x2",
        (DType::Q4_0, _, _) => "wmma_i8_q40_2x1",
        (DType::Q4_1, 1, 1) => "wmma_i8_q41_1x1",
        (DType::Q4_1, 2, 2) => "wmma_i8_q41_2x2",
        (DType::Q4_1, _, _) => "wmma_i8_q41_2x1",
        (DType::Q5_1, 1, 1) => "wmma_i8_q51_1x1",
        (DType::Q5_1, 2, 2) => "wmma_i8_q51_2x2",
        (DType::Q5_1, _, _) => "wmma_i8_q51_2x1",
        (DType::Iq4Nl, 1, 1) => "wmma_i8_iq4nl_1x1",
        (DType::Iq4Nl, 2, 2) => "wmma_i8_iq4nl_2x2",
        (DType::Iq4Nl, _, _) => "wmma_i8_iq4nl_2x1",
        (DType::Iq4Xs, 1, 1) => "wmma_i8_iq4xs_1x1",
        (DType::Iq4Xs, 2, 2) => "wmma_i8_iq4xs_2x2",
        (DType::Iq4Xs, _, _) => "wmma_i8_iq4xs_2x1",
        (DType::Iq2Xxs, 1, 1) => "wmma_i8_iq2xxs_1x1",
        (DType::Iq2Xxs, 2, 2) => "wmma_i8_iq2xxs_2x2",
        (DType::Iq2Xxs, _, _) => "wmma_i8_iq2xxs_2x1",
        (DType::Iq2Xs, 1, 1) => "wmma_i8_iq2xs_1x1",
        (DType::Iq2Xs, 2, 2) => "wmma_i8_iq2xs_2x2",
        (DType::Iq2Xs, _, _) => "wmma_i8_iq2xs_2x1",
        (DType::Iq2S, 1, 1) => "wmma_i8_iq2s_1x1",
        (DType::Iq2S, 2, 2) => "wmma_i8_iq2s_2x2",
        (DType::Iq2S, _, _) => "wmma_i8_iq2s_2x1",
        (DType::Iq3Xxs, 1, 1) => "wmma_i8_iq3xxs_1x1",
        (DType::Iq3Xxs, 2, 2) => "wmma_i8_iq3xxs_2x2",
        (DType::Iq3Xxs, _, _) => "wmma_i8_iq3xxs_2x1",
        (DType::Iq3S, 1, 1) => "wmma_i8_iq3s_1x1",
        (DType::Iq3S, 2, 2) => "wmma_i8_iq3s_2x2",
        (DType::Iq3S, _, _) => "wmma_i8_iq3s_2x1",
        (DType::Iq1S, 1, 1) => "wmma_i8_iq1s_1x1",
        (DType::Iq1S, 2, 2) => "wmma_i8_iq1s_2x2",
        (DType::Iq1S, _, _) => "wmma_i8_iq1s_2x1",
        (DType::Iq1M, 1, 1) => "wmma_i8_iq1m_1x1",
        (DType::Iq1M, 2, 2) => "wmma_i8_iq1m_2x2",
        (DType::Iq1M, _, _) => "wmma_i8_iq1m_2x1",
        (DType::Tq1_0, 1, 1) => "wmma_i8_tq10_1x1",
        (DType::Tq1_0, 2, 2) => "wmma_i8_tq10_2x2",
        (DType::Tq1_0, _, _) => "wmma_i8_tq10_2x1",
        (DType::Tq2_0, 1, 1) => "wmma_i8_tq20_1x1",
        (DType::Tq2_0, 2, 2) => "wmma_i8_tq20_2x2",
        (DType::Tq2_0, _, _) => "wmma_i8_tq20_2x1",
        (DType::Q2_0, 1, 1) => "wmma_i8_q20_1x1",
        (DType::Q2_0, 2, 2) => "wmma_i8_q20_2x2",
        (DType::Q2_0, _, _) => "wmma_i8_q20_2x1",
        (DType::Mxfp4, 1, 1) => "wmma_i8_mxfp4_1x1",
        (DType::Mxfp4, 2, 2) => "wmma_i8_mxfp4_2x2",
        (DType::Mxfp4, _, _) => "wmma_i8_mxfp4_2x1",
        (DType::Nvfp4, 1, 1) => "wmma_i8_nvfp4_1x1",
        (DType::Nvfp4, 2, 2) => "wmma_i8_nvfp4_2x2",
        (DType::Nvfp4, _, _) => "wmma_i8_nvfp4_2x1",
        _ => return None,
    };
    Some((name, rm, cn))
}

/// Cooperative decode-once Q4_K prefill GEMM kernel (Phase 5, Slice-28). A multi-warp threadblock
/// decodes the BN-column weight tile into LDS int8 ONCE per K-step and reuses it across all BM rows,
/// killing the ~m/32× redundant weight decode of the single-wave `wmma_i8_q4k_*` kernels. Returns
/// `(kernel_name, BM, BN, threads)`; the launch grid is `(ceil(out_f/BN), ceil(m/BM))`.
///
/// OPT-IN (`INFR_ROCM_COOP=1`), NOT the default: the Slice-28 sweep measured this single-buffered
/// cooperative form as a REGRESSION vs the Slice-27 pipe on gfx1100 (isolated GEMM ~0.6×, pp512
/// ~0.90×). Root cause (docs/perf.md, `-Rpass-analysis` + micro-bench): the RX 7900 XTX Q4_K GEMM at
/// m=512 is NOT decode-bound — the baseline's barrier-free single-wave design already hides the
/// redundant decode behind full occupancy (thousands of 1-wave blocks). Cooperative trades that away:
/// the wide-tile variants are VGPR-starved (7 waves/SIMD vs 9-10), and even the occupancy-matched
/// `128x32` tile (10 waves/SIMD) loses, because the 2× `__syncthreads`/K-step and the serialized
/// cooperative-decode phase (only BN of the block's threads active) cost more than decode-once saves.
/// Removing the min-term ones-dot WMMA (`_rs` tiles) does not recover it → not matrix-core bound.
/// Closing the ~5.6× gap to llama.cpp needs the double-buffered async LDS pipeline (overlap
/// load+decode with WMMA so the barriers don't stall) — a larger change left for the next slice. The
/// kernels are kept (bit-faithful, parity-green) as that pipeline's foundation. `INFR_ROCM_COOP_TILE`
/// selects the tile for A/B benchmarking (`examples/wmma_bench`). Bit-identical math to
/// `wmma_i8_q4k_2x1` (same int8 codes + per-block scale-after order), so the blessed goldens hold.
fn q4k_coop_kernel(rocm: &infr_core::config::RocmCfg) -> Option<(&'static str, u32, u32, u32)> {
    // Opt-in gate: `kernels.rocm.coop` false (the default, i.e. `INFR_ROCM_COOP` absent) → `None`
    // (fall through to the default pipe path).
    if !rocm.coop {
        return None;
    }
    Some(
        // Trimmed by the env layer; an unrecognized tile falls to the `_` default, as before.
        match rocm.coop_tile.as_deref() {
            Some("128x32") => ("wmma_i8_q4k_coop_128x32_w8", 128, 32, 256),
            Some("64x64") => ("wmma_i8_q4k_coop_64x64_w4", 64, 64, 128),
            Some("64x32") => ("wmma_i8_q4k_coop_64x32_w8", 64, 32, 256),
            Some("128x128") => ("wmma_i8_q4k_coop_128x128_w8", 128, 128, 256),
            Some("256x64") => ("wmma_i8_q4k_coop_256x64_w16", 256, 64, 512),
            Some("rs_128x64") => ("wmma_i8_q4k_coop_rs_128x64_w8", 128, 64, 256),
            Some("rs_128x32") => ("wmma_i8_q4k_coop_rs_128x32_w8", 128, 32, 256),
            Some("rs_64x64") => ("wmma_i8_q4k_coop_rs_64x64_w4", 64, 64, 128),
            Some("rs_64x32") => ("wmma_i8_q4k_coop_rs_64x32_w8", 64, 32, 256),
            _ => ("wmma_i8_q4k_coop_128x64_w8", 128, 64, 256),
        },
    )
}

/// Native in-kernel decode for a MoE expert weight bank (Phase-3 for MoE). Returns
/// `(suffix, elems_per_block, bytes_per_block)` for a covered dtype — the suffix names the
/// `deq_*` decoder baked into the `moe_ffn_expert_<gu>_<dn>` kernel and the block geometry gives
/// the per-expert byte offset. `None` for uncovered formats (they keep the dequant→f16 fallback).
fn moe_native_fmt(dt: DType) -> Option<(&'static str, usize, usize)> {
    let suffix = match dt {
        DType::Q8_0 => "q80",
        DType::Q2K => "q2k",
        DType::Q3K => "q3k",
        DType::Q4K => "q4k",
        DType::Q5K => "q5k",
        DType::Q6K => "q6k",
        DType::Q4_0 => "q40",
        DType::Q4_1 => "q41",
        DType::Q5_1 => "q51",
        DType::Iq4Nl => "iq4nl",
        DType::Iq4Xs => "iq4xs",
        DType::Iq2Xxs => "iq2xxs",
        DType::Iq2Xs => "iq2xs",
        DType::Iq2S => "iq2s",
        DType::Iq3Xxs => "iq3xxs",
        DType::Iq3S => "iq3s",
        DType::Iq1S => "iq1s",
        DType::Iq1M => "iq1m",
        DType::Tq1_0 => "tq10",
        DType::Tq2_0 => "tq20",
        DType::Q2_0 => "q20",
        DType::Mxfp4 => "mxfp4",
        DType::Nvfp4 => "nvfp4",
        _ => return None,
    };
    let (elems, bytes) = infr_core::decode_spec::block_layout(dt);
    Some((suffix, elems, bytes))
}

/// Static kernel name for the `(gate/up format, down format)` combo of the Phase-3 f16-decode expert
/// FFN, or `None` when that pair is NOT instantiated in `kernels.rs` — the caller then keeps the
/// dequant→f16 `moe_ffn_expert` fallback. Instantiated set (116 of the 441 `moe_native_fmt` pairs):
///
/// * the full `{q80, q2k, q3k, q4k, q5k, q6k}²` (36 — e.g. Q4_K_M is `("q4k", "q6k")`, Q3_K_M is
///   `("q3k", "q5k")`),
/// * `{q40, q41, q51} × {q40, q41, q51, q80}` (12, R3),
/// * `{iq4nl, iq4xs} × {iq4nl, iq4xs, q4k, q5k, q6k, q80}` (12, R4),
/// * `{q2k, q3k} × {iq4nl}` (2, R4),
/// * `{iq2xxs, iq2xs, iq2s, iq3xxs, iq3s} × {iq2s, iq3xxs, iq3s, iq4nl, iq4xs, q4k, q6k}` (35, R5),
/// * `{iq1s, iq1m} × {iq1s, iq1m, iq2xxs, iq2s, iq3s, iq4xs, q4k, q6k}` (16, R6),
/// * the three ternary SELF pairs `{(tq10,tq10), (tq20,tq20), (q20,q20)}` (3, R6),
/// * the two fp4 SELF pairs `{(mxfp4,mxfp4), (nvfp4,nvfp4)}` (2, R7).
///
/// **Why not the full cross product** (R2 documented this escape hatch, R3 measured it and took it):
/// going 6×6 → 9×9 cost **+1.1 s of COLD hiprtc** — backend init plus a 1-token bench with
/// `~/.cache/comgr` cleared went 4.31 s → 6.27 s, against 5.44 s for the 48-pair set. R5 re-measured
/// the same way at the 16-format mark (3 reps each): **6.81-6.93 s** at R4's 62 pairs → **8.28-9.02
/// s** once R5's 55 DENSE kernels are added at the same 62 pairs → **9.14-9.30 s** at the shipped
/// 97. So the 35 pairs added above cost ~0.5 s (~14 ms each) while the dense kernels — the actual
/// feature — are ~1.6 s of the delta; the full 16×16 would have piled on 159 more cells (~2.2 s) for
/// nothing. Warm-cache startup is unchanged at ~0.51 s in every variant. The cells cut are the ones
/// nothing can reach:
///
/// * These kernels are NOT the shipping MoE path. The default int8 dp4a expert path dispatches the
///   per-FORMAT `moe_gate_up_act_i8_<gu>` + `moe_down_i8_<dn>` kernels, which ARE total over
///   `moe_native_fmt` (21 each). `moe_ffn_expert_<gu>_<dn>` runs only under `INFR_ROCM_NO_I8` — an
///   A/B benchmarking switch whose comparand is precisely the dequant→f16 path an absent pair
///   falls to.
/// * Legacy round quants never mix with K-quants across gate/up vs down. llama.cpp's
///   `llama_tensor_get_type` bumps a Q4_0/Q4_1/Q5_1 model's `ffn_down` to another legacy round quant
///   or to Q8_0, never to a K-quant, and the K-quant ftypes never step down to a legacy type.
/// * An IQ4 expert bank means an IQ-family ftype, whose `ffn_down` bump lands on another IQ4, on
///   Q4_K/Q5_K/Q6_K (`use_more_bits`), or on Q8_0 — never on Q2_K/Q3_K, and never on a legacy round
///   quant. The one direction that DOES reach a K-quant gate/up with an IQ4 down is
///   `convert_incompatible_tensor`, which rewrites a Q2_K/Q3_K tensor whose row is not 256-divisible
///   to **IQ4_NL** (Q4_K/Q5_K/Q6_K go to Q5_0/Q5_1/Q8_0 there instead, which is why only q2k/q3k
///   appear in that third group).
/// * The R5 grid quants are gate/up banks only. An IQ2/IQ3 gate/up means an IQ ftype, and `ffn_down`
///   is always bumped to something WIDER — never back down to IQ2_XXS/IQ2_XS, and never to a legacy
///   round quant or to Q2_K/Q3_K/Q5_K. That is what the 7 `dn` entries are: the same-or-wider IQ
///   ladder (`iq2s`, `iq3xxs`, `iq3s`, `iq4nl`, `iq4xs`) plus the two `use_more_bits` K-quant
///   targets `q4k`/`q6k`. The shape is not hypothetical — `Qwen3.6-35B-A3B-UD-IQ3_S` (cached on the
///   dev box) packs `ffn_gate_exps`/`ffn_up_exps` as **IQ2_S** with `ffn_down_exps` split **IQ3_S**
///   (37 tensors) / **IQ4_XS** (3), i.e. it needs exactly `("iq2s","iq3s")` and `("iq2s","iq4xs")`.
///   Nothing gives a grid quant as a `dn` under a K-quant or legacy `gu`, so those 55 cells are cut.
/// * The R6 IQ1 quants are gate/up banks too, but UNLIKE the grid quants they are also a legal `dn`
///   — an IQ1 ftype leaves most `ffn_down` tensors at the SAME type and only boosts a minority. The
///   `dn` set is read off the two cached UD-IQ1 GGUFs rather than guessed: `Qwen3-0.6B-UD-IQ1_S`
///   packs `ffn_gate`/`ffn_up` as **IQ1_S** with `ffn_down` split **IQ1_S** (18 layers) / **IQ2_S**
///   (5) / **IQ3_S** (5), and `Qwen3-0.6B-UD-IQ1_M` packs gate/up as **IQ1_M** (18) / IQ1_S (5) /
///   IQ2_XXS (5) with `ffn_down` split **IQ1_M** (18) / **IQ2_S** (5) / **IQ3_S** (5). (Those are
///   dense FFN tensors — the box has no IQ1 MoE cached — but `llama_tensor_get_type` applies the
///   same `ffn_down` rule to `ffn_down_exps`, and the IQ2_XXS/IQ2_S/IQ3_S `gu` cells that mix in
///   are already covered by R5's rectangle.) The 8 `dn` entries are that observed set (`iq1s`,
///   `iq1m`, `iq2s`, `iq3s`) plus the wider bumps a big-MoE IQ1 mix reaches — `iq2xxs` (the shape
///   `DeepSeek-R1-UD-IQ1_S` ships), `iq4xs`, and the two `use_more_bits` targets `q4k`/`q6k`.
///   Never a legacy round quant, never Q2_K/Q3_K/Q5_K, never a ternary type.
/// * The R6 ternary quants pair only with THEMSELVES. TQ1_0/TQ2_0/Q2_0 are not ftype mixes at all —
///   they are whole-model conversion targets for a natively ternary checkpoint (BitNet / TriLM /
///   Bonsai), so every FFN tensor in such a model carries the one type and there is no `ffn_down`
///   bump to model. Nothing mixes a ternary bank with any other family in either direction, so the
///   three self pairs are the complete reachable set for them.
/// * The R7 fp4 quants pair only with themselves too, and here the rule is WRITTEN DOWN rather than
///   inferred: `llama_tensor_get_type` handles `LLAMA_FTYPE_MOSTLY_MXFP4_MOE` before every other
///   branch as "MoE tensors (`ne[2] > 1`) → MXFP4, other tensors → Q8_0" — one unconditional arm,
///   no `use_more_bits`, no `ffn_down` bump, so gate/up AND down are the SAME type by construction
///   and no K-quant or IQ type can appear opposite an fp4 bank. The cached
///   `ggml-org/gpt-oss-20b-MXFP4` is exactly that: all 72 `ffn_{gate,up,down}_exps` MXFP4, every
///   dense tensor Q8_0 — i.e. it needs `("mxfp4","mxfp4")` and nothing else. NVFP4 has no ftype of
///   its own yet and no cached GGUF; it is the same microscaling family with the same
///   whole-MoE-conversion shape, so it gets the same single self pair.
///
/// An absent pair is not a bug and never panics: `None` here makes the `MoeFfn` arm drop `native`
/// (see the filter there), so the whole expert takes the correct, slower f16 path. Both this table
/// and `moe_expert_routed_kernel` must cover EXACTLY the same set — `moe_expert_pair_tables_agree`
/// pins that against `MOE_EXPERT_PAIRS`.
fn moe_expert_kernel(gu: &str, dn: &str) -> Option<&'static str> {
    match (gu, dn) {
        ("q80", "q80") => Some("moe_ffn_expert_q80_q80"),
        ("q80", "q2k") => Some("moe_ffn_expert_q80_q2k"),
        ("q80", "q3k") => Some("moe_ffn_expert_q80_q3k"),
        ("q80", "q4k") => Some("moe_ffn_expert_q80_q4k"),
        ("q80", "q5k") => Some("moe_ffn_expert_q80_q5k"),
        ("q80", "q6k") => Some("moe_ffn_expert_q80_q6k"),
        ("q2k", "q80") => Some("moe_ffn_expert_q2k_q80"),
        ("q2k", "q2k") => Some("moe_ffn_expert_q2k_q2k"),
        ("q2k", "q3k") => Some("moe_ffn_expert_q2k_q3k"),
        ("q2k", "q4k") => Some("moe_ffn_expert_q2k_q4k"),
        ("q2k", "q5k") => Some("moe_ffn_expert_q2k_q5k"),
        ("q2k", "q6k") => Some("moe_ffn_expert_q2k_q6k"),
        ("q3k", "q80") => Some("moe_ffn_expert_q3k_q80"),
        ("q3k", "q2k") => Some("moe_ffn_expert_q3k_q2k"),
        ("q3k", "q3k") => Some("moe_ffn_expert_q3k_q3k"),
        ("q3k", "q4k") => Some("moe_ffn_expert_q3k_q4k"),
        ("q3k", "q5k") => Some("moe_ffn_expert_q3k_q5k"),
        ("q3k", "q6k") => Some("moe_ffn_expert_q3k_q6k"),
        ("q4k", "q80") => Some("moe_ffn_expert_q4k_q80"),
        ("q4k", "q2k") => Some("moe_ffn_expert_q4k_q2k"),
        ("q4k", "q3k") => Some("moe_ffn_expert_q4k_q3k"),
        ("q4k", "q4k") => Some("moe_ffn_expert_q4k_q4k"),
        ("q4k", "q5k") => Some("moe_ffn_expert_q4k_q5k"),
        ("q4k", "q6k") => Some("moe_ffn_expert_q4k_q6k"),
        ("q5k", "q80") => Some("moe_ffn_expert_q5k_q80"),
        ("q5k", "q2k") => Some("moe_ffn_expert_q5k_q2k"),
        ("q5k", "q3k") => Some("moe_ffn_expert_q5k_q3k"),
        ("q5k", "q4k") => Some("moe_ffn_expert_q5k_q4k"),
        ("q5k", "q5k") => Some("moe_ffn_expert_q5k_q5k"),
        ("q5k", "q6k") => Some("moe_ffn_expert_q5k_q6k"),
        ("q6k", "q80") => Some("moe_ffn_expert_q6k_q80"),
        ("q6k", "q2k") => Some("moe_ffn_expert_q6k_q2k"),
        ("q6k", "q3k") => Some("moe_ffn_expert_q6k_q3k"),
        ("q6k", "q4k") => Some("moe_ffn_expert_q6k_q4k"),
        ("q6k", "q5k") => Some("moe_ffn_expert_q6k_q5k"),
        ("q6k", "q6k") => Some("moe_ffn_expert_q6k_q6k"),
        ("q40", "q40") => Some("moe_ffn_expert_q40_q40"),
        ("q40", "q41") => Some("moe_ffn_expert_q40_q41"),
        ("q40", "q51") => Some("moe_ffn_expert_q40_q51"),
        ("q40", "q80") => Some("moe_ffn_expert_q40_q80"),
        ("q41", "q40") => Some("moe_ffn_expert_q41_q40"),
        ("q41", "q41") => Some("moe_ffn_expert_q41_q41"),
        ("q41", "q51") => Some("moe_ffn_expert_q41_q51"),
        ("q41", "q80") => Some("moe_ffn_expert_q41_q80"),
        ("q51", "q40") => Some("moe_ffn_expert_q51_q40"),
        ("q51", "q41") => Some("moe_ffn_expert_q51_q41"),
        ("q51", "q51") => Some("moe_ffn_expert_q51_q51"),
        ("q51", "q80") => Some("moe_ffn_expert_q51_q80"),
        ("iq4nl", "iq4nl") => Some("moe_ffn_expert_iq4nl_iq4nl"),
        ("iq4nl", "iq4xs") => Some("moe_ffn_expert_iq4nl_iq4xs"),
        ("iq4nl", "q4k") => Some("moe_ffn_expert_iq4nl_q4k"),
        ("iq4nl", "q5k") => Some("moe_ffn_expert_iq4nl_q5k"),
        ("iq4nl", "q6k") => Some("moe_ffn_expert_iq4nl_q6k"),
        ("iq4nl", "q80") => Some("moe_ffn_expert_iq4nl_q80"),
        ("iq4xs", "iq4nl") => Some("moe_ffn_expert_iq4xs_iq4nl"),
        ("iq4xs", "iq4xs") => Some("moe_ffn_expert_iq4xs_iq4xs"),
        ("iq4xs", "q4k") => Some("moe_ffn_expert_iq4xs_q4k"),
        ("iq4xs", "q5k") => Some("moe_ffn_expert_iq4xs_q5k"),
        ("iq4xs", "q6k") => Some("moe_ffn_expert_iq4xs_q6k"),
        ("iq4xs", "q80") => Some("moe_ffn_expert_iq4xs_q80"),
        ("q2k", "iq4nl") => Some("moe_ffn_expert_q2k_iq4nl"),
        ("q3k", "iq4nl") => Some("moe_ffn_expert_q3k_iq4nl"),
        ("iq2xxs", "iq2s") => Some("moe_ffn_expert_iq2xxs_iq2s"),
        ("iq2xxs", "iq3xxs") => Some("moe_ffn_expert_iq2xxs_iq3xxs"),
        ("iq2xxs", "iq3s") => Some("moe_ffn_expert_iq2xxs_iq3s"),
        ("iq2xxs", "iq4nl") => Some("moe_ffn_expert_iq2xxs_iq4nl"),
        ("iq2xxs", "iq4xs") => Some("moe_ffn_expert_iq2xxs_iq4xs"),
        ("iq2xxs", "q4k") => Some("moe_ffn_expert_iq2xxs_q4k"),
        ("iq2xxs", "q6k") => Some("moe_ffn_expert_iq2xxs_q6k"),
        ("iq2xs", "iq2s") => Some("moe_ffn_expert_iq2xs_iq2s"),
        ("iq2xs", "iq3xxs") => Some("moe_ffn_expert_iq2xs_iq3xxs"),
        ("iq2xs", "iq3s") => Some("moe_ffn_expert_iq2xs_iq3s"),
        ("iq2xs", "iq4nl") => Some("moe_ffn_expert_iq2xs_iq4nl"),
        ("iq2xs", "iq4xs") => Some("moe_ffn_expert_iq2xs_iq4xs"),
        ("iq2xs", "q4k") => Some("moe_ffn_expert_iq2xs_q4k"),
        ("iq2xs", "q6k") => Some("moe_ffn_expert_iq2xs_q6k"),
        ("iq2s", "iq2s") => Some("moe_ffn_expert_iq2s_iq2s"),
        ("iq2s", "iq3xxs") => Some("moe_ffn_expert_iq2s_iq3xxs"),
        ("iq2s", "iq3s") => Some("moe_ffn_expert_iq2s_iq3s"),
        ("iq2s", "iq4nl") => Some("moe_ffn_expert_iq2s_iq4nl"),
        ("iq2s", "iq4xs") => Some("moe_ffn_expert_iq2s_iq4xs"),
        ("iq2s", "q4k") => Some("moe_ffn_expert_iq2s_q4k"),
        ("iq2s", "q6k") => Some("moe_ffn_expert_iq2s_q6k"),
        ("iq3xxs", "iq2s") => Some("moe_ffn_expert_iq3xxs_iq2s"),
        ("iq3xxs", "iq3xxs") => Some("moe_ffn_expert_iq3xxs_iq3xxs"),
        ("iq3xxs", "iq3s") => Some("moe_ffn_expert_iq3xxs_iq3s"),
        ("iq3xxs", "iq4nl") => Some("moe_ffn_expert_iq3xxs_iq4nl"),
        ("iq3xxs", "iq4xs") => Some("moe_ffn_expert_iq3xxs_iq4xs"),
        ("iq3xxs", "q4k") => Some("moe_ffn_expert_iq3xxs_q4k"),
        ("iq3xxs", "q6k") => Some("moe_ffn_expert_iq3xxs_q6k"),
        ("iq3s", "iq2s") => Some("moe_ffn_expert_iq3s_iq2s"),
        ("iq3s", "iq3xxs") => Some("moe_ffn_expert_iq3s_iq3xxs"),
        ("iq3s", "iq3s") => Some("moe_ffn_expert_iq3s_iq3s"),
        ("iq3s", "iq4nl") => Some("moe_ffn_expert_iq3s_iq4nl"),
        ("iq3s", "iq4xs") => Some("moe_ffn_expert_iq3s_iq4xs"),
        ("iq3s", "q4k") => Some("moe_ffn_expert_iq3s_q4k"),
        ("iq3s", "q6k") => Some("moe_ffn_expert_iq3s_q6k"),
        ("iq1s", "iq1s") => Some("moe_ffn_expert_iq1s_iq1s"),
        ("iq1s", "iq1m") => Some("moe_ffn_expert_iq1s_iq1m"),
        ("iq1s", "iq2xxs") => Some("moe_ffn_expert_iq1s_iq2xxs"),
        ("iq1s", "iq2s") => Some("moe_ffn_expert_iq1s_iq2s"),
        ("iq1s", "iq3s") => Some("moe_ffn_expert_iq1s_iq3s"),
        ("iq1s", "iq4xs") => Some("moe_ffn_expert_iq1s_iq4xs"),
        ("iq1s", "q4k") => Some("moe_ffn_expert_iq1s_q4k"),
        ("iq1s", "q6k") => Some("moe_ffn_expert_iq1s_q6k"),
        ("iq1m", "iq1s") => Some("moe_ffn_expert_iq1m_iq1s"),
        ("iq1m", "iq1m") => Some("moe_ffn_expert_iq1m_iq1m"),
        ("iq1m", "iq2xxs") => Some("moe_ffn_expert_iq1m_iq2xxs"),
        ("iq1m", "iq2s") => Some("moe_ffn_expert_iq1m_iq2s"),
        ("iq1m", "iq3s") => Some("moe_ffn_expert_iq1m_iq3s"),
        ("iq1m", "iq4xs") => Some("moe_ffn_expert_iq1m_iq4xs"),
        ("iq1m", "q4k") => Some("moe_ffn_expert_iq1m_q4k"),
        ("iq1m", "q6k") => Some("moe_ffn_expert_iq1m_q6k"),
        ("tq10", "tq10") => Some("moe_ffn_expert_tq10_tq10"),
        ("tq20", "tq20") => Some("moe_ffn_expert_tq20_tq20"),
        ("q20", "q20") => Some("moe_ffn_expert_q20_q20"),
        ("mxfp4", "mxfp4") => Some("moe_ffn_expert_mxfp4_mxfp4"),
        ("nvfp4", "nvfp4") => Some("moe_ffn_expert_nvfp4_nvfp4"),
        _ => None,
    }
}

/// Whether the int8-activation dp4a MoE expert path (Slice 20) is enabled. Reuses the dense
/// `INFR_ROCM_NO_I8` A/B switch: when set, MoE falls back to the Phase-3 f16-decode expert kernel.
fn moe_i8_enabled(rocm: &infr_core::config::RocmCfg) -> bool {
    rocm.i8
}

/// Static gate/up int8 kernel name for the gate/up format suffix (`moe_gate_up_act_i8_<gu>`). The
/// suffix comes from `moe_native_fmt`, so the `_` arm is unreachable.
fn moe_gate_up_i8_kernel(gu: &str) -> &'static str {
    match gu {
        "q80" => "moe_gate_up_act_i8_q80",
        "q2k" => "moe_gate_up_act_i8_q2k",
        "q3k" => "moe_gate_up_act_i8_q3k",
        "q4k" => "moe_gate_up_act_i8_q4k",
        "q5k" => "moe_gate_up_act_i8_q5k",
        "q6k" => "moe_gate_up_act_i8_q6k",
        "q40" => "moe_gate_up_act_i8_q40",
        "q41" => "moe_gate_up_act_i8_q41",
        "q51" => "moe_gate_up_act_i8_q51",
        "iq4nl" => "moe_gate_up_act_i8_iq4nl",
        "iq4xs" => "moe_gate_up_act_i8_iq4xs",
        "iq2xxs" => "moe_gate_up_act_i8_iq2xxs",
        "iq2xs" => "moe_gate_up_act_i8_iq2xs",
        "iq2s" => "moe_gate_up_act_i8_iq2s",
        "iq3xxs" => "moe_gate_up_act_i8_iq3xxs",
        "iq3s" => "moe_gate_up_act_i8_iq3s",
        "iq1s" => "moe_gate_up_act_i8_iq1s",
        "iq1m" => "moe_gate_up_act_i8_iq1m",
        "tq10" => "moe_gate_up_act_i8_tq10",
        "tq20" => "moe_gate_up_act_i8_tq20",
        "q20" => "moe_gate_up_act_i8_q20",
        "mxfp4" => "moe_gate_up_act_i8_mxfp4",
        "nvfp4" => "moe_gate_up_act_i8_nvfp4",
        _ => unreachable!("moe_gate_up_i8_kernel: uncovered ({gu})"),
    }
}

/// Static down int8 kernel name for the down format suffix (`moe_down_i8_<dn>`).
fn moe_down_i8_kernel(dn: &str) -> &'static str {
    match dn {
        "q80" => "moe_down_i8_q80",
        "q2k" => "moe_down_i8_q2k",
        "q3k" => "moe_down_i8_q3k",
        "q4k" => "moe_down_i8_q4k",
        "q5k" => "moe_down_i8_q5k",
        "q6k" => "moe_down_i8_q6k",
        "q40" => "moe_down_i8_q40",
        "q41" => "moe_down_i8_q41",
        "q51" => "moe_down_i8_q51",
        "iq4nl" => "moe_down_i8_iq4nl",
        "iq4xs" => "moe_down_i8_iq4xs",
        "iq2xxs" => "moe_down_i8_iq2xxs",
        "iq2xs" => "moe_down_i8_iq2xs",
        "iq2s" => "moe_down_i8_iq2s",
        "iq3xxs" => "moe_down_i8_iq3xxs",
        "iq3s" => "moe_down_i8_iq3s",
        "iq1s" => "moe_down_i8_iq1s",
        "iq1m" => "moe_down_i8_iq1m",
        "tq10" => "moe_down_i8_tq10",
        "tq20" => "moe_down_i8_tq20",
        "q20" => "moe_down_i8_q20",
        "mxfp4" => "moe_down_i8_mxfp4",
        "nvfp4" => "moe_down_i8_nvfp4",
        _ => unreachable!("moe_down_i8_kernel: uncovered ({dn})"),
    }
}

/// Device-routed (Slice 38) twin of `moe_expert_kernel` — the `moe_ffn_expert_routed_<gu>_<dn>`
/// native-decode expert FFN that resolves the per-expert bank pointer + routing weight on-device.
/// Instantiated over the SAME [`MOE_EXPERT_PAIRS`] set, so one availability check covers both.
fn moe_expert_routed_kernel(gu: &str, dn: &str) -> Option<&'static str> {
    match (gu, dn) {
        ("q80", "q80") => Some("moe_ffn_expert_routed_q80_q80"),
        ("q80", "q2k") => Some("moe_ffn_expert_routed_q80_q2k"),
        ("q80", "q3k") => Some("moe_ffn_expert_routed_q80_q3k"),
        ("q80", "q4k") => Some("moe_ffn_expert_routed_q80_q4k"),
        ("q80", "q5k") => Some("moe_ffn_expert_routed_q80_q5k"),
        ("q80", "q6k") => Some("moe_ffn_expert_routed_q80_q6k"),
        ("q2k", "q80") => Some("moe_ffn_expert_routed_q2k_q80"),
        ("q2k", "q2k") => Some("moe_ffn_expert_routed_q2k_q2k"),
        ("q2k", "q3k") => Some("moe_ffn_expert_routed_q2k_q3k"),
        ("q2k", "q4k") => Some("moe_ffn_expert_routed_q2k_q4k"),
        ("q2k", "q5k") => Some("moe_ffn_expert_routed_q2k_q5k"),
        ("q2k", "q6k") => Some("moe_ffn_expert_routed_q2k_q6k"),
        ("q3k", "q80") => Some("moe_ffn_expert_routed_q3k_q80"),
        ("q3k", "q2k") => Some("moe_ffn_expert_routed_q3k_q2k"),
        ("q3k", "q3k") => Some("moe_ffn_expert_routed_q3k_q3k"),
        ("q3k", "q4k") => Some("moe_ffn_expert_routed_q3k_q4k"),
        ("q3k", "q5k") => Some("moe_ffn_expert_routed_q3k_q5k"),
        ("q3k", "q6k") => Some("moe_ffn_expert_routed_q3k_q6k"),
        ("q4k", "q80") => Some("moe_ffn_expert_routed_q4k_q80"),
        ("q4k", "q2k") => Some("moe_ffn_expert_routed_q4k_q2k"),
        ("q4k", "q3k") => Some("moe_ffn_expert_routed_q4k_q3k"),
        ("q4k", "q4k") => Some("moe_ffn_expert_routed_q4k_q4k"),
        ("q4k", "q5k") => Some("moe_ffn_expert_routed_q4k_q5k"),
        ("q4k", "q6k") => Some("moe_ffn_expert_routed_q4k_q6k"),
        ("q5k", "q80") => Some("moe_ffn_expert_routed_q5k_q80"),
        ("q5k", "q2k") => Some("moe_ffn_expert_routed_q5k_q2k"),
        ("q5k", "q3k") => Some("moe_ffn_expert_routed_q5k_q3k"),
        ("q5k", "q4k") => Some("moe_ffn_expert_routed_q5k_q4k"),
        ("q5k", "q5k") => Some("moe_ffn_expert_routed_q5k_q5k"),
        ("q5k", "q6k") => Some("moe_ffn_expert_routed_q5k_q6k"),
        ("q6k", "q80") => Some("moe_ffn_expert_routed_q6k_q80"),
        ("q6k", "q2k") => Some("moe_ffn_expert_routed_q6k_q2k"),
        ("q6k", "q3k") => Some("moe_ffn_expert_routed_q6k_q3k"),
        ("q6k", "q4k") => Some("moe_ffn_expert_routed_q6k_q4k"),
        ("q6k", "q5k") => Some("moe_ffn_expert_routed_q6k_q5k"),
        ("q6k", "q6k") => Some("moe_ffn_expert_routed_q6k_q6k"),
        ("q40", "q40") => Some("moe_ffn_expert_routed_q40_q40"),
        ("q40", "q41") => Some("moe_ffn_expert_routed_q40_q41"),
        ("q40", "q51") => Some("moe_ffn_expert_routed_q40_q51"),
        ("q40", "q80") => Some("moe_ffn_expert_routed_q40_q80"),
        ("q41", "q40") => Some("moe_ffn_expert_routed_q41_q40"),
        ("q41", "q41") => Some("moe_ffn_expert_routed_q41_q41"),
        ("q41", "q51") => Some("moe_ffn_expert_routed_q41_q51"),
        ("q41", "q80") => Some("moe_ffn_expert_routed_q41_q80"),
        ("q51", "q40") => Some("moe_ffn_expert_routed_q51_q40"),
        ("q51", "q41") => Some("moe_ffn_expert_routed_q51_q41"),
        ("q51", "q51") => Some("moe_ffn_expert_routed_q51_q51"),
        ("q51", "q80") => Some("moe_ffn_expert_routed_q51_q80"),
        ("iq4nl", "iq4nl") => Some("moe_ffn_expert_routed_iq4nl_iq4nl"),
        ("iq4nl", "iq4xs") => Some("moe_ffn_expert_routed_iq4nl_iq4xs"),
        ("iq4nl", "q4k") => Some("moe_ffn_expert_routed_iq4nl_q4k"),
        ("iq4nl", "q5k") => Some("moe_ffn_expert_routed_iq4nl_q5k"),
        ("iq4nl", "q6k") => Some("moe_ffn_expert_routed_iq4nl_q6k"),
        ("iq4nl", "q80") => Some("moe_ffn_expert_routed_iq4nl_q80"),
        ("iq4xs", "iq4nl") => Some("moe_ffn_expert_routed_iq4xs_iq4nl"),
        ("iq4xs", "iq4xs") => Some("moe_ffn_expert_routed_iq4xs_iq4xs"),
        ("iq4xs", "q4k") => Some("moe_ffn_expert_routed_iq4xs_q4k"),
        ("iq4xs", "q5k") => Some("moe_ffn_expert_routed_iq4xs_q5k"),
        ("iq4xs", "q6k") => Some("moe_ffn_expert_routed_iq4xs_q6k"),
        ("iq4xs", "q80") => Some("moe_ffn_expert_routed_iq4xs_q80"),
        ("q2k", "iq4nl") => Some("moe_ffn_expert_routed_q2k_iq4nl"),
        ("q3k", "iq4nl") => Some("moe_ffn_expert_routed_q3k_iq4nl"),
        ("iq2xxs", "iq2s") => Some("moe_ffn_expert_routed_iq2xxs_iq2s"),
        ("iq2xxs", "iq3xxs") => Some("moe_ffn_expert_routed_iq2xxs_iq3xxs"),
        ("iq2xxs", "iq3s") => Some("moe_ffn_expert_routed_iq2xxs_iq3s"),
        ("iq2xxs", "iq4nl") => Some("moe_ffn_expert_routed_iq2xxs_iq4nl"),
        ("iq2xxs", "iq4xs") => Some("moe_ffn_expert_routed_iq2xxs_iq4xs"),
        ("iq2xxs", "q4k") => Some("moe_ffn_expert_routed_iq2xxs_q4k"),
        ("iq2xxs", "q6k") => Some("moe_ffn_expert_routed_iq2xxs_q6k"),
        ("iq2xs", "iq2s") => Some("moe_ffn_expert_routed_iq2xs_iq2s"),
        ("iq2xs", "iq3xxs") => Some("moe_ffn_expert_routed_iq2xs_iq3xxs"),
        ("iq2xs", "iq3s") => Some("moe_ffn_expert_routed_iq2xs_iq3s"),
        ("iq2xs", "iq4nl") => Some("moe_ffn_expert_routed_iq2xs_iq4nl"),
        ("iq2xs", "iq4xs") => Some("moe_ffn_expert_routed_iq2xs_iq4xs"),
        ("iq2xs", "q4k") => Some("moe_ffn_expert_routed_iq2xs_q4k"),
        ("iq2xs", "q6k") => Some("moe_ffn_expert_routed_iq2xs_q6k"),
        ("iq2s", "iq2s") => Some("moe_ffn_expert_routed_iq2s_iq2s"),
        ("iq2s", "iq3xxs") => Some("moe_ffn_expert_routed_iq2s_iq3xxs"),
        ("iq2s", "iq3s") => Some("moe_ffn_expert_routed_iq2s_iq3s"),
        ("iq2s", "iq4nl") => Some("moe_ffn_expert_routed_iq2s_iq4nl"),
        ("iq2s", "iq4xs") => Some("moe_ffn_expert_routed_iq2s_iq4xs"),
        ("iq2s", "q4k") => Some("moe_ffn_expert_routed_iq2s_q4k"),
        ("iq2s", "q6k") => Some("moe_ffn_expert_routed_iq2s_q6k"),
        ("iq3xxs", "iq2s") => Some("moe_ffn_expert_routed_iq3xxs_iq2s"),
        ("iq3xxs", "iq3xxs") => Some("moe_ffn_expert_routed_iq3xxs_iq3xxs"),
        ("iq3xxs", "iq3s") => Some("moe_ffn_expert_routed_iq3xxs_iq3s"),
        ("iq3xxs", "iq4nl") => Some("moe_ffn_expert_routed_iq3xxs_iq4nl"),
        ("iq3xxs", "iq4xs") => Some("moe_ffn_expert_routed_iq3xxs_iq4xs"),
        ("iq3xxs", "q4k") => Some("moe_ffn_expert_routed_iq3xxs_q4k"),
        ("iq3xxs", "q6k") => Some("moe_ffn_expert_routed_iq3xxs_q6k"),
        ("iq3s", "iq2s") => Some("moe_ffn_expert_routed_iq3s_iq2s"),
        ("iq3s", "iq3xxs") => Some("moe_ffn_expert_routed_iq3s_iq3xxs"),
        ("iq3s", "iq3s") => Some("moe_ffn_expert_routed_iq3s_iq3s"),
        ("iq3s", "iq4nl") => Some("moe_ffn_expert_routed_iq3s_iq4nl"),
        ("iq3s", "iq4xs") => Some("moe_ffn_expert_routed_iq3s_iq4xs"),
        ("iq3s", "q4k") => Some("moe_ffn_expert_routed_iq3s_q4k"),
        ("iq3s", "q6k") => Some("moe_ffn_expert_routed_iq3s_q6k"),
        ("iq1s", "iq1s") => Some("moe_ffn_expert_routed_iq1s_iq1s"),
        ("iq1s", "iq1m") => Some("moe_ffn_expert_routed_iq1s_iq1m"),
        ("iq1s", "iq2xxs") => Some("moe_ffn_expert_routed_iq1s_iq2xxs"),
        ("iq1s", "iq2s") => Some("moe_ffn_expert_routed_iq1s_iq2s"),
        ("iq1s", "iq3s") => Some("moe_ffn_expert_routed_iq1s_iq3s"),
        ("iq1s", "iq4xs") => Some("moe_ffn_expert_routed_iq1s_iq4xs"),
        ("iq1s", "q4k") => Some("moe_ffn_expert_routed_iq1s_q4k"),
        ("iq1s", "q6k") => Some("moe_ffn_expert_routed_iq1s_q6k"),
        ("iq1m", "iq1s") => Some("moe_ffn_expert_routed_iq1m_iq1s"),
        ("iq1m", "iq1m") => Some("moe_ffn_expert_routed_iq1m_iq1m"),
        ("iq1m", "iq2xxs") => Some("moe_ffn_expert_routed_iq1m_iq2xxs"),
        ("iq1m", "iq2s") => Some("moe_ffn_expert_routed_iq1m_iq2s"),
        ("iq1m", "iq3s") => Some("moe_ffn_expert_routed_iq1m_iq3s"),
        ("iq1m", "iq4xs") => Some("moe_ffn_expert_routed_iq1m_iq4xs"),
        ("iq1m", "q4k") => Some("moe_ffn_expert_routed_iq1m_q4k"),
        ("iq1m", "q6k") => Some("moe_ffn_expert_routed_iq1m_q6k"),
        ("tq10", "tq10") => Some("moe_ffn_expert_routed_tq10_tq10"),
        ("tq20", "tq20") => Some("moe_ffn_expert_routed_tq20_tq20"),
        ("q20", "q20") => Some("moe_ffn_expert_routed_q20_q20"),
        ("mxfp4", "mxfp4") => Some("moe_ffn_expert_routed_mxfp4_mxfp4"),
        ("nvfp4", "nvfp4") => Some("moe_ffn_expert_routed_nvfp4_nvfp4"),
        _ => None,
    }
}

/// Device-routed twin of `moe_gate_up_i8_kernel`.
fn moe_gate_up_i8_routed_kernel(gu: &str) -> &'static str {
    match gu {
        "q80" => "moe_gate_up_act_i8_routed_q80",
        "q2k" => "moe_gate_up_act_i8_routed_q2k",
        "q3k" => "moe_gate_up_act_i8_routed_q3k",
        "q4k" => "moe_gate_up_act_i8_routed_q4k",
        "q5k" => "moe_gate_up_act_i8_routed_q5k",
        "q6k" => "moe_gate_up_act_i8_routed_q6k",
        "q40" => "moe_gate_up_act_i8_routed_q40",
        "q41" => "moe_gate_up_act_i8_routed_q41",
        "q51" => "moe_gate_up_act_i8_routed_q51",
        "iq4nl" => "moe_gate_up_act_i8_routed_iq4nl",
        "iq4xs" => "moe_gate_up_act_i8_routed_iq4xs",
        "iq2xxs" => "moe_gate_up_act_i8_routed_iq2xxs",
        "iq2xs" => "moe_gate_up_act_i8_routed_iq2xs",
        "iq2s" => "moe_gate_up_act_i8_routed_iq2s",
        "iq3xxs" => "moe_gate_up_act_i8_routed_iq3xxs",
        "iq3s" => "moe_gate_up_act_i8_routed_iq3s",
        "iq1s" => "moe_gate_up_act_i8_routed_iq1s",
        "iq1m" => "moe_gate_up_act_i8_routed_iq1m",
        "tq10" => "moe_gate_up_act_i8_routed_tq10",
        "tq20" => "moe_gate_up_act_i8_routed_tq20",
        "q20" => "moe_gate_up_act_i8_routed_q20",
        "mxfp4" => "moe_gate_up_act_i8_routed_mxfp4",
        "nvfp4" => "moe_gate_up_act_i8_routed_nvfp4",
        _ => unreachable!("moe_gate_up_i8_routed_kernel: uncovered ({gu})"),
    }
}

/// Device-routed twin of `moe_down_i8_kernel`.
fn moe_down_i8_routed_kernel(dn: &str) -> &'static str {
    match dn {
        "q80" => "moe_down_i8_routed_q80",
        "q2k" => "moe_down_i8_routed_q2k",
        "q3k" => "moe_down_i8_routed_q3k",
        "q4k" => "moe_down_i8_routed_q4k",
        "q5k" => "moe_down_i8_routed_q5k",
        "q6k" => "moe_down_i8_routed_q6k",
        "q40" => "moe_down_i8_routed_q40",
        "q41" => "moe_down_i8_routed_q41",
        "q51" => "moe_down_i8_routed_q51",
        "iq4nl" => "moe_down_i8_routed_iq4nl",
        "iq4xs" => "moe_down_i8_routed_iq4xs",
        "iq2xxs" => "moe_down_i8_routed_iq2xxs",
        "iq2xs" => "moe_down_i8_routed_iq2xs",
        "iq2s" => "moe_down_i8_routed_iq2s",
        "iq3xxs" => "moe_down_i8_routed_iq3xxs",
        "iq3s" => "moe_down_i8_routed_iq3s",
        "iq1s" => "moe_down_i8_routed_iq1s",
        "iq1m" => "moe_down_i8_routed_iq1m",
        "tq10" => "moe_down_i8_routed_tq10",
        "tq20" => "moe_down_i8_routed_tq20",
        "q20" => "moe_down_i8_routed_q20",
        "mxfp4" => "moe_down_i8_routed_mxfp4",
        "nvfp4" => "moe_down_i8_routed_nvfp4",
        _ => unreachable!("moe_down_i8_routed_kernel: uncovered ({dn})"),
    }
}

/// Token rows per id-indexed MoE expert-GEMV dispatch (R8) — the `moe_*_idm_*` tier's batch bound.
///
/// **This is not Vulkan's `MOE_SMALL_M` with a different number, and it is deliberately not a
/// crossover.** Vulkan's threshold picks between its id-GEMV tier and a bucket-sorted batched
/// expert GEMM, and 8 is where the batched path's cross-token weight reuse starts to pay for its
/// quant/count/scan/scatter prologue. ROCm HAS NO BATCHED EXPERT PATH: above the threshold it
/// would fall back to the same per-`(row, slot)` host loop the id tier replaces, over the *same*
/// per-slot weight traffic — so the id tier is a strict improvement at every `m` here and the
/// only thing left to bound is the per-slot scratch (`[rows·n_used, n_ff_exp]` activations plus
/// `[rows·n_used, ne]` partial outputs), which grows linearly in `rows`.
///
/// So `rows` is CHUNKED at this value rather than the tier being abandoned past it. Measured on
/// the RX 7900 XTX with Qwen3-30B-A3B Q4_K_M `pp512`, 3 reps (see `docs/rocm-plan.md` R8):
/// 8 → 237.1, 32 → 249.0, 128 → 253.3, unchunked → 261.2 t/s. Throughput rises with the chunk and
/// flattens; 128 is NOT the top of that curve, and the last ~3% is deliberately left on the table,
/// because it is not free: at `-p 1024` an unchunked (or 512-row) chunk asks ~50-100 MiB of pool
/// on top of a 17 GiB weight set plus its KV and `BufferPool`'s `hipMalloc` FAILS — reproduced on
/// this box, and today that aborts the process rather than degrading. `MOE_ID_SCRATCH_CAP` (at the
/// use site) is the backstop that makes the ceiling a byte count as well as a row count.
///
/// The clamp is [`infr_core::tier::EnvRows`]' — the SHARED policy half, so this knob cannot grow
/// its own bounds grammar. `0` keeps the meaning it has on Vulkan's crossover knob, "never take
/// the id tier": it drops the resident MoE path back to the pre-R8 per-`(row, slot)` loop, which
/// is the A/B comparand `moe_ffn_id_tier_matches_the_serial_tier_bitwise` runs against. The
/// ceiling matches `INFR_UBATCH`'s default prefill chunk — a larger value cannot widen a dispatch
/// beyond what the seam hands the op, it can only inflate the scratch.
const MOE_ID_ROWS: infr_core::tier::EnvRows = infr_core::tier::EnvRows {
    // No env key — a typed config field (`kernels.rocm.moe_id_rows`), like `module_cache`. The
    // manifest has nothing to map, so this names the config path it answers to instead.
    env: "kernels.rocm.moe_id_rows",
    default: 128,
    min: 0,
    max: 1024,
};

/// Multi-slot id-indexed twin of [`moe_gate_up_i8_routed_kernel`] (R8).
fn moe_gate_up_i8_idm_kernel(gu: &str) -> &'static str {
    match gu {
        "q80" => "moe_gate_up_act_i8_idm_q80",
        "q2k" => "moe_gate_up_act_i8_idm_q2k",
        "q3k" => "moe_gate_up_act_i8_idm_q3k",
        "q4k" => "moe_gate_up_act_i8_idm_q4k",
        "q5k" => "moe_gate_up_act_i8_idm_q5k",
        "q6k" => "moe_gate_up_act_i8_idm_q6k",
        "q40" => "moe_gate_up_act_i8_idm_q40",
        "q41" => "moe_gate_up_act_i8_idm_q41",
        "q51" => "moe_gate_up_act_i8_idm_q51",
        "iq4nl" => "moe_gate_up_act_i8_idm_iq4nl",
        "iq4xs" => "moe_gate_up_act_i8_idm_iq4xs",
        "iq2xxs" => "moe_gate_up_act_i8_idm_iq2xxs",
        "iq2xs" => "moe_gate_up_act_i8_idm_iq2xs",
        "iq2s" => "moe_gate_up_act_i8_idm_iq2s",
        "iq3xxs" => "moe_gate_up_act_i8_idm_iq3xxs",
        "iq3s" => "moe_gate_up_act_i8_idm_iq3s",
        "iq1s" => "moe_gate_up_act_i8_idm_iq1s",
        "iq1m" => "moe_gate_up_act_i8_idm_iq1m",
        "tq10" => "moe_gate_up_act_i8_idm_tq10",
        "tq20" => "moe_gate_up_act_i8_idm_tq20",
        "q20" => "moe_gate_up_act_i8_idm_q20",
        "mxfp4" => "moe_gate_up_act_i8_idm_mxfp4",
        "nvfp4" => "moe_gate_up_act_i8_idm_nvfp4",
        _ => unreachable!("moe_gate_up_i8_idm_kernel: uncovered ({gu})"),
    }
}

/// Multi-slot id-indexed twin of [`moe_down_i8_routed_kernel`] (R8).
fn moe_down_i8_idm_kernel(dn: &str) -> &'static str {
    match dn {
        "q80" => "moe_down_i8_idm_q80",
        "q2k" => "moe_down_i8_idm_q2k",
        "q3k" => "moe_down_i8_idm_q3k",
        "q4k" => "moe_down_i8_idm_q4k",
        "q5k" => "moe_down_i8_idm_q5k",
        "q6k" => "moe_down_i8_idm_q6k",
        "q40" => "moe_down_i8_idm_q40",
        "q41" => "moe_down_i8_idm_q41",
        "q51" => "moe_down_i8_idm_q51",
        "iq4nl" => "moe_down_i8_idm_iq4nl",
        "iq4xs" => "moe_down_i8_idm_iq4xs",
        "iq2xxs" => "moe_down_i8_idm_iq2xxs",
        "iq2xs" => "moe_down_i8_idm_iq2xs",
        "iq2s" => "moe_down_i8_idm_iq2s",
        "iq3xxs" => "moe_down_i8_idm_iq3xxs",
        "iq3s" => "moe_down_i8_idm_iq3s",
        "iq1s" => "moe_down_i8_idm_iq1s",
        "iq1m" => "moe_down_i8_idm_iq1m",
        "tq10" => "moe_down_i8_idm_tq10",
        "tq20" => "moe_down_i8_idm_tq20",
        "q20" => "moe_down_i8_idm_q20",
        "mxfp4" => "moe_down_i8_idm_mxfp4",
        "nvfp4" => "moe_down_i8_idm_nvfp4",
        _ => unreachable!("moe_down_i8_idm_kernel: uncovered ({dn})"),
    }
}

/// Widest `n_expert` the P2 bucket sort's LDS histogram holds — mirrors `MOE_BUCKET_MAX_EXPERT`
/// in `kernels.rs`. Wider MoEs keep the id tier (the sort would need dynamic shared memory).
const MOE_BUCKET_MAX_EXPERT: usize = 1024;

/// Wave32s per P2 batched-tier workgroup — mirrors `MOE_IDB_WAVES` in `kernels.rs`.
///
/// Each wave still owns ONE output row and runs the id tier's arithmetic; this only packs several
/// of them into one workgroup. It is an OCCUPANCY knob: a 32-thread (one-wave) workgroup hits the
/// hardware's workgroups-per-CU limit long before its waves-per-SIMD limit, so the single-wave
/// grid the id tier uses cannot keep enough waves resident to hide the decode+load latency.
const MOE_IDB_WAVES: u32 = 4;

/// Minimum AVERAGE bucket occupancy (`n_slots / n_expert`) for the P2 batched tier to be worth
/// taking. The batched grid is `(output row, expert)` — it launches a block per expert whether or
/// not any token routed there, so below ~2 slots an expert the empty blocks cost more than the
/// saved weight traffic returns. Decode (`n_used` slots over `n_expert` banks) is always well
/// under this, which is how the tier stays a PREFILL change.
const MOE_BUCKET_MIN_OCC: usize = 2;

/// Bucket-sorted batched twin of [`moe_gate_up_i8_idm_kernel`] (P2).
fn moe_gate_up_i8_idb_kernel(gu: &str) -> &'static str {
    match gu {
        "q80" => "moe_gate_up_act_i8_idb_q80",
        "q2k" => "moe_gate_up_act_i8_idb_q2k",
        "q3k" => "moe_gate_up_act_i8_idb_q3k",
        "q4k" => "moe_gate_up_act_i8_idb_q4k",
        "q5k" => "moe_gate_up_act_i8_idb_q5k",
        "q6k" => "moe_gate_up_act_i8_idb_q6k",
        "q40" => "moe_gate_up_act_i8_idb_q40",
        "q41" => "moe_gate_up_act_i8_idb_q41",
        "q51" => "moe_gate_up_act_i8_idb_q51",
        "iq4nl" => "moe_gate_up_act_i8_idb_iq4nl",
        "iq4xs" => "moe_gate_up_act_i8_idb_iq4xs",
        "iq2xxs" => "moe_gate_up_act_i8_idb_iq2xxs",
        "iq2xs" => "moe_gate_up_act_i8_idb_iq2xs",
        "iq2s" => "moe_gate_up_act_i8_idb_iq2s",
        "iq3xxs" => "moe_gate_up_act_i8_idb_iq3xxs",
        "iq3s" => "moe_gate_up_act_i8_idb_iq3s",
        "iq1s" => "moe_gate_up_act_i8_idb_iq1s",
        "iq1m" => "moe_gate_up_act_i8_idb_iq1m",
        "tq10" => "moe_gate_up_act_i8_idb_tq10",
        "tq20" => "moe_gate_up_act_i8_idb_tq20",
        "q20" => "moe_gate_up_act_i8_idb_q20",
        "mxfp4" => "moe_gate_up_act_i8_idb_mxfp4",
        "nvfp4" => "moe_gate_up_act_i8_idb_nvfp4",
        _ => unreachable!("moe_gate_up_i8_idb_kernel: uncovered ({gu})"),
    }
}

/// Bucket-sorted batched twin of [`moe_down_i8_idm_kernel`] (P2).
fn moe_down_i8_idb_kernel(dn: &str) -> &'static str {
    match dn {
        "q80" => "moe_down_i8_idb_q80",
        "q2k" => "moe_down_i8_idb_q2k",
        "q3k" => "moe_down_i8_idb_q3k",
        "q4k" => "moe_down_i8_idb_q4k",
        "q5k" => "moe_down_i8_idb_q5k",
        "q6k" => "moe_down_i8_idb_q6k",
        "q40" => "moe_down_i8_idb_q40",
        "q41" => "moe_down_i8_idb_q41",
        "q51" => "moe_down_i8_idb_q51",
        "iq4nl" => "moe_down_i8_idb_iq4nl",
        "iq4xs" => "moe_down_i8_idb_iq4xs",
        "iq2xxs" => "moe_down_i8_idb_iq2xxs",
        "iq2xs" => "moe_down_i8_idb_iq2xs",
        "iq2s" => "moe_down_i8_idb_iq2s",
        "iq3xxs" => "moe_down_i8_idb_iq3xxs",
        "iq3s" => "moe_down_i8_idb_iq3s",
        "iq1s" => "moe_down_i8_idb_iq1s",
        "iq1m" => "moe_down_i8_idb_iq1m",
        "tq10" => "moe_down_i8_idb_tq10",
        "tq20" => "moe_down_i8_idb_tq20",
        "q20" => "moe_down_i8_idb_q20",
        "mxfp4" => "moe_down_i8_idb_mxfp4",
        "nvfp4" => "moe_down_i8_idb_nvfp4",
        _ => unreachable!("moe_down_i8_idb_kernel: uncovered ({dn})"),
    }
}

/// MMQ decode-once-reuse gate+up kernel name for a covered format suffix.
/// Returns `None` when MMQ is not yet implemented for this format.
fn mmq_up_kernel(gu: &str) -> Option<&'static str> {
    Some(match gu {
        "q4k" => "moe_mmq_up_i8_q4k",
        "q6k" => "moe_mmq_up_i8_q6k",
        "q5k" => "moe_mmq_up_i8_q5k",
        "q80" => "moe_mmq_up_i8_q80",
        "q2k" => "moe_mmq_up_i8_q2k",
        "q3k" => "moe_mmq_up_i8_q3k",
        "iq2xxs" => "moe_mmq_up_i8_iq2xxs",
        "iq2xs" => "moe_mmq_up_i8_iq2xs",
        "iq2s" => "moe_mmq_up_i8_iq2s",
        "iq3xxs" => "moe_mmq_up_i8_iq3xxs",
        "iq3s" => "moe_mmq_up_i8_iq3s",
        "iq1s" => "moe_mmq_up_i8_iq1s",
        "iq1m" => "moe_mmq_up_i8_iq1m",
        "tq10" => "moe_mmq_up_i8_tq10",
        "tq20" => "moe_mmq_up_i8_tq20",
        "q20" => "moe_mmq_up_i8_q20",
        "mxfp4" => "moe_mmq_up_i8_mxfp4",
        "nvfp4" => "moe_mmq_up_i8_nvfp4",
        "iq4nl" => "moe_mmq_up_i8_iq4nl",
        "iq4xs" => "moe_mmq_up_i8_iq4xs",
        "q40" => "moe_mmq_up_i8_q40",
        "q41" => "moe_mmq_up_i8_q41",
        "q51" => "moe_mmq_up_i8_q51",
        _ => return None,
    })
}

/// Activation kernel name for MMQ raw gate+up → silu(gate)*up + route-weight + scale.
fn mmq_act_kernel(gu: &str) -> &'static str {
    match gu {
        "q4k" => "moe_act_mul_q4k",
        "q6k" => "moe_act_mul_q6k",
        "q5k" => "moe_act_mul_q5k",
        "q80" => "moe_act_mul_q80",
        "q2k" => "moe_act_mul_q2k",
        "q3k" => "moe_act_mul_q3k",
        "iq2xxs" => "moe_act_mul_iq2xxs",
        "iq2xs" => "moe_act_mul_iq2xs",
        "iq2s" => "moe_act_mul_iq2s",
        "iq3xxs" => "moe_act_mul_iq3xxs",
        "iq3s" => "moe_act_mul_iq3s",
        "iq1s" => "moe_act_mul_iq1s",
        "iq1m" => "moe_act_mul_iq1m",
        "tq10" => "moe_act_mul_tq10",
        "tq20" => "moe_act_mul_tq20",
        "q20" => "moe_act_mul_q20",
        "mxfp4" => "moe_act_mul_mxfp4",
        "nvfp4" => "moe_act_mul_nvfp4",
        "iq4nl" => "moe_act_mul_iq4nl",
        "iq4xs" => "moe_act_mul_iq4xs",
        "q40" => "moe_act_mul_q40",
        "q41" => "moe_act_mul_q41",
        "q51" => "moe_act_mul_q51",
        _ => "moe_act_mul_q4k", // unreachable: mmq_up_kernel gates first
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
fn arg_i64(v: i64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}
fn arg_f32(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

// ── ExecCtx ──────────────────────────────────────────────────────────────────

/// Decode activation-quantization memo (F1b).
///
/// The int8 GEMV path quantizes its activation row before every projection. On every arch the seam
/// emits, SIBLING projections read the SAME row: `q`/`k`/`v` all consume one input norm, `gate`/`up`
/// (when not concatenated) one post-attention norm, and a fused-QKV upload issues several `w_off`
/// slices of one weight. Each of those re-ran `rmsnorm_quant_i8_32` / `quant_i8_32` over identical
/// input and wrote identical bytes to a fresh scratch pair — pure redundancy, and it also forced a
/// WAR hazard between the siblings (each quant rewrote the buffer the previous GEMV was reading).
///
/// So remember the pass: the NEXT op, if it is a decode GEMV over the same source row with the same
/// norm, binds the same `(qx, xs)` instead of recomputing them. Byte-identical by construction — the
/// GEMV reads the exact bytes the elided pass would have written — so it needs no capability, no
/// tolerance and no golden movement.
///
/// The memo is `take()`n at the top of every [`run_op`] and restored ONLY by the int8 branch, so any
/// other op in between (which may rewrite the row) invalidates it by construction; the branch
/// additionally refuses to publish a memo whose source row is what this very GEMV just wrote (the
/// fused-residual epilogue writes into the residual stream, which is also some norms' input).
#[derive(Clone, Copy, PartialEq, Eq)]
struct QuantKey {
    /// Source activation row: the raw pre-norm row when the RmsNorm→Linear fold is active, else the
    /// already-normalized `x`. Held as BOTH the graph tensor and the device pointer the quant
    /// kernel actually read — a hit needs both, so neither a rebound tensor nor a recycled handle
    /// can match a stale memo.
    src: (TensorId, *mut c_void),
    /// Norm weight, or `None` for the plain (`quant_i8_32`) pass. Two Linears over the same raw row
    /// but under DIFFERENT norm weights are different activations.
    norm: Option<(TensorId, *mut c_void)>,
    /// `eps` by bit pattern — the key is an exact match, never a float compare.
    eps_bits: u32,
    m: u32,
    in_f: u32,
}

#[derive(Clone, Copy)]
struct QuantMemo {
    key: QuantKey,
    /// The int8 codes + per-32-block scales that pass wrote. Pool draws live until end-of-forward
    /// (`ExecCtx::pooled`), so these stay valid for every op that can still read the memo.
    qx: *mut c_void,
    xs: *mut c_void,
}

struct ExecCtx<'a> {
    dev: Vec<Option<crate::RocmBuffer>>,
    vals: Vec<Option<Vec<f32>>>,
    weight_cache: &'a crate::backend::WeightCache,
    /// Reusable device-scratch pool (persists across `execute` calls on the backend).
    pool: &'a Mutex<BufferPool>,
    /// Paged MoE expert cache (Slice 33 — `crate::pager`), `None` for a resident model. The
    /// `Op::MoeFfn` arm resolves per-expert slot pointers through it instead of the resident bank.
    moe_pager: &'a Mutex<Option<crate::pager::RocmMoePager>>,
    /// Pool draws made this forward pass: `(ptr, bucket_bytes)`, returned to `pool` on `Drop`
    /// (both the success path and any early-error return) so nothing is `hipFree`'d per op.
    pooled: Vec<(*mut c_void, usize)>,
    /// Dense-weight prefetch ring (Slice 37 — `crate::weight_pager`), `None` for a model with no
    /// spilled dense bank. The `Op::Linear` arm routes a spilled-native weight through it (staged
    /// VRAM slot) instead of reading the Slice-35 host alias over PCIe.
    weight_ring: &'a Mutex<Option<crate::weight_pager::RocmWeightRing>>,
    /// `true` once the ring is built AND primed for this forward (a non-empty spilled schedule).
    /// The Linear arm checks it before consulting the ring; `false` ⇒ every weight uses the
    /// resident / Slice-35 host-alias path unchanged.
    weight_ring_active: bool,
    /// Per-bank size cap for staging (`weight_pager::max_bank_bytes`): a spilled-native Linear bank
    /// is staged only when `len <= cap`. The SINGLE predicate the schedule build and the per-op
    /// staged decision share so the ring's cursor stays in lockstep with the op walk.
    weight_prefetch_cap: usize,
    stream: ffi::hipStream_t,
    /// rocBLAS handle bound to `stream` for the OPT-IN Slice-26 f16 prefill GEMM (`INFR_ROCM_BLAS=1`),
    /// or `null` (the default) — in which case the prefill path uses the int8 WMMA kernel.
    rocblas: ffi::rocblas_handle,
    /// The backend's ROCm kernel-tier config, BORROWED for the whole forward (S6, R6): the int8 /
    /// WMMA / pipe / cooperative selectors read it per op instead of calling `getenv`.
    rocm: &'a infr_core::config::RocmCfg,
    /// F1b sibling-GEMV activation-quant memo — see [`QuantMemo`]. `None` at the start of every
    /// forward and after any op that is not a memo-publishing int8 GEMV.
    qmemo: Option<QuantMemo>,
    /// F5: force the un-cleared pool draw ([`pool_buf`](ExecCtx::pool_buf) with `zero = false`) to
    /// fill with the POISON byte instead of leaving the recycled bytes alone. Debug builds do this
    /// unconditionally; this flag (`debug.poison_uninit`) turns it on in RELEASE too, for hunting a
    /// read-before-write whose output only shifts when an unrelated change reshuffles the pool.
    poison: bool,
}

/// Byte written over an un-cleared pool draw when poisoning is active. `0xFF` in all four bytes is
/// f32 NaN (and `-1` as i32/i8), so a slot the kernel was supposed to overwrite but did not turns
/// the whole downstream row into NaN — a LOUD failure in the goldens, instead of the silently
/// plausible answer a recycled-but-happens-to-be-zero block would produce. Same byte and same
/// reasoning as `VulkanBackend::alloc_uninit`.
const POISON_BYTE: c_int = 0xFF;

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
    ///
    /// F5 — the poison. `zero = false` is a CLAIM ("this kernel writes every byte first"), and a
    /// wrong claim is a silent wrong-answer bug, because the pool hands back a block some earlier
    /// op already wrote — frequently one that happens to make the output look plausible, or that
    /// happens to be zero on the first pass. So the un-cleared draw is filled with [`POISON_BYTE`]
    /// in debug builds (and in release under `debug.poison_uninit`): any byte the kernel then fails
    /// to write reads back as NaN and the goldens move loudly. The poison is NOT free — it is the
    /// very memset this slice removes — so release builds skip it, which is exactly why the parity
    /// suite has to be run in DEBUG for the claim to have been tested at all.
    fn pool_buf(&mut self, bytes: usize, zero: bool) -> crate::RocmBuffer {
        let len = bytes.max(1);
        let bucket = bucket_bytes(len);
        let ptr = self.pool.lock().unwrap().take(bucket);
        let fill = if zero {
            Some(0)
        } else if cfg!(debug_assertions) || self.poison {
            Some(POISON_BYTE)
        } else {
            None
        };
        if let Some(byte) = fill {
            let rc = unsafe { ffi::hipMemsetAsync(ptr, byte, len, self.stream) };
            debug_assert_eq!(rc, HIP_SUCCESS, "hipMemsetAsync(pool zero-on-reuse)");
        }
        self.pooled.push((ptr, bucket));
        crate::RocmBuffer {
            ptr,
            len,
            owned: false,
            host_ptr: std::ptr::null_mut(),
            // A fresh draw is a NEW logical buffer even though the pool recycled the bytes behind
            // it — stamp it accordingly so nothing can memoize across two unrelated draws.
            uid: crate::backend::next_buffer_uid(),
        }
    }

    /// Zeroed scratch for `n` f32 ELEMENTS (calloc contract). Pooled + async-cleared. Reserved for
    /// the `dst`s that genuinely READ what they were handed: accumulators (`moe_accum_idm`'s
    /// `dst += acc`, the pre-R8 per-slot `atomicAdd`), partial writes (`Copy`/`CopyStrided`'s
    /// content-preserving destination, the strided `Rope` clone whose memcpy is length-clamped) and
    /// unproduced tensors reached by [`ensure_device`](Self::ensure_device).
    ///
    /// A `dst` the kernel overwrites in full must use [`uninit_dev`](Self::uninit_dev) instead —
    /// see F5 in `docs/rocm-plan.md` for why this memset was ~20% of a decode token.
    fn zero_dev(&mut self, n: usize) -> crate::RocmBuffer {
        self.pool_buf((n * 4).max(1), true)
    }

    /// UN-cleared scratch for `n` f32 ELEMENTS — the same pooled draw as [`zero_dev`](Self::zero_dev)
    /// without the memset, for a `dst` whose kernel writes every byte before anything reads it.
    ///
    /// F5. `hipMemsetAsync` is ~3.8 µs of real GPU work on gfx1100 (a null kernel is 2.7 µs), and
    /// the 0.6B decodes ~423 ops in 7.6 ms — so clearing a `dst` the GEMV then overwrites cost
    /// ~1.6 ms/token, ~20% of decode, for nothing. The obligation that comes with calling this is
    /// exact: EVERY element of `[0, n)` must be stored by the dispatch(es) below, on every shape
    /// the op can take, before any read. Mirrors `VulkanBackend::alloc_uninit` (and carries the
    /// same debug poison, see [`pool_buf`](Self::pool_buf)).
    fn uninit_dev(&mut self, n: usize) -> crate::RocmBuffer {
        self.pool_buf((n * 4).max(1), false)
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
                // Track in dev so subsequent accesses find it. A view of the bound allocation, so
                // it carries that allocation's identity.
                self.dev[i] = Some(crate::RocmBuffer {
                    ptr: p,
                    len: b.len,
                    owned: false,
                    host_ptr: std::ptr::null_mut(),
                    uid: b.uid,
                });
                p
            }
            TensorKind::Internal | TensorKind::Output => {
                // Not yet produced — allocate a zero-filled buffer. F5 keeps this calloc: reaching
                // here means an op wants a tensor NOTHING has written, and the callers that do
                // (`Copy`/`CopyStrided`'s content-preserving dst, the fused-residual `add_dst`) all
                // read what they are handed. The CPU reference's `vals[dst]` is zeroed too.
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
        // (address, byte length) is only the SLOT — HIP recycles a freed address for the next
        // same-sized allocation, so it is not an identity. The bound buffer's `uid` is; a hit is
        // served only when it matches, otherwise the address was recycled for different weights and
        // the stale dequant must not be handed out (see `RocmBackend::weight_cache`).
        let key = (b.ptr as usize, b.len);
        {
            let cache = self.weight_cache.lock().unwrap();
            if let Some((uid, cached)) = cache.get(&key) {
                if *uid == b.uid {
                    return Ok(cached.ptr);
                }
            }
        }
        let dt = g.desc(id).dtype;
        let raw = read_bytes(b, self.stream);
        let f32s = bytes_to_f32(&raw, dt)?;
        let f16_bytes = f32_to_f16_bytes(&f32s);
        let dq = self.f16_dev(&f16_bytes);
        let ptr = dq.ptr;
        let len = dq.len;
        let dq_uid = dq.uid;
        {
            let mut cache = self.weight_cache.lock().unwrap();
            // Cache owns the device memory (owned: true), stamped with the identity of the WEIGHT
            // buffer it was dequantized from. Inserting on an already-occupied key is the
            // recycled-address case: the replaced entry drops here, freeing the stale dequant.
            cache.insert(
                key,
                (
                    b.uid,
                    crate::RocmBuffer {
                        ptr: dq.ptr,
                        len: dq.len,
                        owned: true,
                        host_ptr: std::ptr::null_mut(),
                        uid: dq_uid,
                    },
                ),
            );
        }
        // Store a non-owned reference in dev so ctx.drop doesn't free it.
        // Prevent dq from dropping (cache owns the allocation now).
        std::mem::forget(dq);
        self.dev[i] = Some(crate::RocmBuffer {
            ptr,
            len,
            owned: false,
            host_ptr: std::ptr::null_mut(),
            uid: dq_uid,
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

/// The staged VRAM slot pointer to feed a spilled-native Linear GEMV, or `None` when this weight is
/// NOT prefetched (resident, or an oversized bank kept on the Slice-35 host-alias read). Consults
/// the prefetch ring under the SAME predicate [`build_spilled_schedule`] used, so the ring's cursor
/// tracks the op walk exactly. On the staged path the compute stream is made to wait on the bank's
/// fill; the caller MUST call [`weight_staged_done`] after dispatching the GEMV.
fn weight_staged_ptr(
    ctx: &ExecCtx,
    weight: TensorId,
    bindings: &Bindings,
) -> Result<Option<*mut c_void>> {
    if !ctx.weight_ring_active {
        return Ok(None);
    }
    let wb = rocm_buf(bindings.get(weight).expect("rocm: unbound Weight"));
    // Spilled (host_ptr set) AND within the staging cap ⇒ prefetched; else fall through to the
    // resident / Slice-35 host-alias read.
    if wb.host_ptr.is_null() || wb.len > ctx.weight_prefetch_cap {
        return Ok(None);
    }
    let mut guard = ctx.weight_ring.lock().unwrap();
    let ring = guard
        .as_mut()
        .expect("weight_ring_active implies a live ring");
    Ok(Some(ring.stage(wb.ptr)?))
}

/// Record that a staged Linear's GEMV was dispatched: record the slot's `free` event + kick off the
/// next bank's prefetch, advancing the ring cursor. Call once, after the GEMV, iff
/// [`weight_staged_ptr`] returned `Some`.
fn weight_staged_done(ctx: &ExecCtx) -> Result<()> {
    let mut guard = ctx.weight_ring.lock().unwrap();
    guard
        .as_mut()
        .expect("weight_staged_done without a live ring")
        .consumed()
}

/// The ordered list of spilled dense Linear banks this graph will read that are eligible for the
/// prefetch ring: `Op::Linear`s (in the same walk order the executor dispatches, skip-set applied)
/// whose weight is a NATIVE-decode format, is spilled to host under Slice 35 (`host_ptr` set), and
/// is `<= cap` bytes. The uncovered (dequant→f16) formats are excluded (their host bank is read once
/// into a VRAM f16 cache, not per-token), as are oversized banks (the lm_head / token_embd, kept on
/// the host-alias read). Empty ⇒ no ring is built.
fn build_spilled_schedule(
    g: &Graph,
    bindings: &Bindings,
    skip: &HashSet<usize>,
    cap: usize,
) -> Vec<crate::weight_pager::SpilledBank> {
    let mut sched = Vec::new();
    // The SAME walk the executor dispatches (`infr_core::exec::live_ops` — graph order, fused-away
    // indices elided), so the ring's cursor can't drift from the op loop below.
    for (_, op) in infr_core::exec::live_ops(&g.ops, skip) {
        let Op::Linear { weight, .. } = *op else {
            continue;
        };
        if native_decode_fmt(g.desc(weight).dtype).is_none() {
            continue;
        }
        let Some(b) = bindings.get(weight) else {
            continue;
        };
        let wb = rocm_buf(b);
        if wb.host_ptr.is_null() || wb.len > cap {
            continue;
        }
        sched.push(crate::weight_pager::SpilledBank {
            host_src: wb.host_ptr,
            dev_alias: wb.ptr,
            len: wb.len,
        });
    }
    sched
}

#[allow(clippy::too_many_arguments)]
pub fn execute_graph(
    pipelines: &Pipelines,
    weight_cache: &crate::backend::WeightCache,
    pool: &Mutex<BufferPool>,
    moe_pager: &Mutex<Option<crate::pager::RocmMoePager>>,
    weight_ring: &Mutex<Option<crate::weight_pager::RocmWeightRing>>,
    stream: ffi::hipStream_t,
    rocblas: ffi::rocblas_handle,
    plan: &dyn Plan,
    bindings: &Bindings,
    cfg: &infr_core::config::Config,
) -> Result<()> {
    let g = &plan
        .as_any()
        .downcast_ref::<GraphPlan>()
        .expect("rocm backend: plan is not a GraphPlan")
        .graph;
    let n = g.tensors.len();

    // Slice 37: build this forward's spilled-dense-Linear prefetch schedule (needs the fusion skip
    // set so it walks the exact ops the executor dispatches). Lazily build / resize the ring, then
    // prime it. Any failure leaves the ring `None` ⇒ the Linear arm falls back to the Slice-35
    // host-alias read (correct, un-overlapped).
    let fusion = decode_fusion(g, cfg);
    let prefetch_cap = crate::weight_pager::max_bank_bytes(&cfg.paging);
    let schedule = build_spilled_schedule(g, bindings, &fusion.skip, prefetch_cap);
    let weight_ring_active = if schedule.is_empty() {
        false
    } else {
        let max_bank = schedule.iter().map(|s| s.len).max().unwrap_or(0);
        let mut guard = weight_ring.lock().unwrap();
        // (Re)build the ring if absent or if a bank now exceeds its slot (a fixed weight set never
        // grows, but the guard keeps the slot arithmetic sound if it ever did).
        if guard.as_ref().is_none_or(|r| r.slot_bytes() < max_bank) {
            *guard = crate::weight_pager::RocmWeightRing::try_new(max_bank, stream, &cfg.paging);
        }
        match guard.as_mut() {
            Some(ring) => {
                ring.begin_execute(schedule)?;
                true
            }
            None => false, // ring build failed → Slice-35 fallback
        }
    };

    let mut ctx = ExecCtx {
        dev: (0..n).map(|_| None).collect(),
        vals: (0..n).map(|_| None).collect(),
        weight_cache,
        pool,
        moe_pager,
        weight_ring,
        weight_ring_active,
        weight_prefetch_cap: prefetch_cap,
        pooled: Vec::new(),
        stream,
        rocblas,
        rocm: &cfg.kernels.rocm,
        qmemo: None,
        poison: cfg.debug.poison_uninit,
    };

    // No per-op sync: the whole op list queues on ONE stream, which serializes device work, so
    // intra-graph producer→consumer ordering holds without a host round-trip. The only syncs are
    // (a) inside `read_bytes`/`host_vals`, immediately before a host readback, and (b) the single
    // barrier below, before the cross-stream writeback DtoD + the final checked sync. With the
    // allocation churn gone (buffer pool), those per-op `hipMalloc`/`hipFree`/`hipStreamSynchronize`
    // device syncs — the real decode bottleneck — are all off the hot path. `fusion` was computed
    // above (it also drives the Slice-37 prefetch schedule). The walk itself (graph order, the
    // Slice-32-elided indices skipped) is the shared `infr_core::exec` skeleton; only `run_op`'s
    // body is per-backend.
    // Per-op profiling is the shared predicate now (`prof.per_op()` — INFR_PROF_OPS or INFR_PROF_OPS
    // — ANDed with the warmup-suppression flag). It used to be `kernels.rocm.prof_ops`, which had
    // no env var at all and ignored suppression, so every table rocm printed silently included the
    // bench's untimed warmup forward.
    let mut prof = infr_core::prof::enabled(&cfg.prof).then(OpProf::new);
    infr_core::exec::run_ops(
        &g.ops,
        &fusion.skip,
        &mut RocmDispatch {
            g,
            bindings,
            pipelines,
            fusion: &fusion,
            ctx: &mut ctx,
            prof: prof.as_mut(),
        },
    )?;

    // Barrier all queued op work before the writeback: the writeback `hipMemcpyDtoD` runs on the
    // NULL stream, which is NOT ordered against our non-default work stream, so it must observe a
    // completed stream first.
    unsafe { ffi::hipStreamSynchronize(stream) };
    // P1: the events are all retired now, so this is a pure host-side read of their timestamps.
    if let Some(p) = prof.take() {
        p.flush();
    }
    // Outputs + mutated f32 Inputs; the in-place KV caches are already current in their bound
    // buffers (shared predicate — `infr_core::exec::writes_back`, same set cpu/metal copy back).
    for id in infr_core::exec::write_back_targets(g) {
        let i = id.0 as usize;
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

// ── Decode op-fusion peephole (Slice 32) ─────────────────────────────────────
//
// Five adjacent-op merges the backend detects on the AGNOSTIC graph (so they apply to every arch),
// each with a scalar fallback when the pattern doesn't match:
//
//   1. `RmsNorm → Linear` (input_norm→qkv, post_attn_norm→gate/up): elide the standalone `rmsnorm`
//      kernel + its normalized-activation DRAM round-trip; every consuming decode GEMV normalizes
//      and int8-quantizes its RAW input row in one `rmsnorm_quant_i8_32` pass (bit-faithful).
//   2. `Linear → Add(residual)` (o_proj, down_proj): fold the residual Add into the GEMV epilogue
//      (`dst = gemv + residual`), killing the standalone `add` kernel + its round-trip.
//   3. F1c `RmsNorm → MoeFfn` (post_attn_norm→experts on a pure-MoE layer): the same fold as (1) for
//      the MoE sublayer, whose input norm has no `Linear` consumer at all — the router GEMV lives
//      INSIDE `Op::MoeFfn`. The MoE arm's own `rmsnorm_quant_i8_32` emits both the int8 codes the
//      experts read and the normalized f32 row the router reads.
//   4. F1c `MoeFfn → Add(residual)`: the residual folds into the ordered expert-accumulate epilogue
//      (`moe_accum_idm`), killing the last standalone `add` a pure-MoE decode paid.
//   5. F1d `QkNormRope → WriteKv`: the rotated K row is written STRAIGHT into the f16 KV cache at
//      the write row by `qk_norm_rope` itself, killing the standalone `write_kv` kernel AND the
//      whole f32 K scratch (its pooled alloc, its zeroing memset, and the DRAM round-trip).
//
// The first four are gated to decode (`m == 1` / one token row) int8 paths — the shipping default
// (every `native_i8_fmt` / `moe_native_fmt` format).
// Prefill (m>1, WMMA/rocBLAS) and uncovered formats keep the split ops. (5) is dtype-free and
// applies to prefill too — the fused kernel's grid is already one wave per (row, head).
// Escape hatches: `INFR_ROCM_NO_FUSE_NORM` / `INFR_ROCM_NO_FUSE_ADD` (each covers its dense AND its
// MoE fold), and `kernels.rocm.fuse_kv_write` for (5).
#[derive(Default)]
struct DecodeFusion {
    /// Linear/MoeFfn op idx → (raw pre-norm x, norm weight, eps): run `rmsnorm_quant_i8_32` on the
    /// raw row instead of `quant_i8_32` on the (elided) normalized input.
    norm: HashMap<usize, (TensorId, TensorId, f32)>,
    /// Linear/MoeFfn op idx → (residual operand, add dst): fold the following `Add` into that op's
    /// write epilogue (the GEMV's, or the MoE expert accumulate's).
    add: HashMap<usize, (TensorId, TensorId)>,
    /// F1d: `QkNormRope` op idx → the absorbed `WriteKv`'s target. The rope kernel writes the f16
    /// cache instead of its f32 `dst` scratch.
    kv: HashMap<usize, KvFuse>,
    /// Op indices to elide entirely (the fused-away `RmsNorm` / `Add` / `WriteKv` / `GatedAct`).
    skip: HashSet<usize>,
    /// E2B per-layer inp_gate: Linear op index → elided GatedAct's payload.
    e2b_gate: HashMap<usize, E2bGateFuse>,
}

/// The absorbed `WriteKv`'s target, resolved at plan time (F1d).
#[derive(Clone, Copy)]
struct KvFuse {
    /// The KV cache tensor — always a BOUND buffer (like `Op::WriteKv`'s own `cache`), never a
    /// `ctx.dev` scratch, so the fused write lands in exactly the allocation the elided kernel used.
    cache: TensorId,
    /// First cache ROW to write — the SHARED plan's row, which `kv_fuse_ok` gate 3 has checked
    /// equals the `WriteKv`'s raw `pos` (the fold is declined outright when a ring cache would have
    /// made it `pos % cap_rows`, because ROCm's `write_kv` does not wrap).
    row: u32,
    /// Per-row elements in the cache (`= n_head * head_dim` of the rope, checked at plan time).
    stride: u32,
}

/// Fused E2B per-layer inp_gate: `Linear` (f32 weight, m ≤ 4) → `GatedAct` (Gelu, strided)
/// collapsed into one `e2b_gate` dispatch. The GatedAct at `i+1` is elided; when `run_op`
/// reaches the `Linear` at `i`, it dispatches `e2b_gate` with the up buffer pointers instead.
#[derive(Clone, Copy)]
struct E2bGateFuse {
    up: TensorId,
    up_off: u32,
    up_stride: u32,
}

/// Weight-dtype predicate for BOTH dense decode fusions: a covered int8-decode GEMV format
/// (`native_i8_fmt`, i.e. every natively decoded format, or `None` under `INFR_ROCM_NO_I8`). The `rmsnorm→
/// int8-decode-Linear` and `int8-decode-Linear→Add` folds share it.
fn fuse_weight_ok(dt: DType, rocm: &infr_core::config::RocmCfg) -> bool {
    native_i8_fmt(dt, rocm).is_some()
}

/// Expert-bank predicate for BOTH F1c MoE fusions: the R8 id-indexed int8 expert tier
/// (`moe_*_idm_*`), which is the only one whose accumulate is an ORDERED reduction with an epilogue
/// to fold into, and the only one whose activation quantize is a single whole-block pass.
///
/// `up` is not required to be natively covered on its own: a `fused_gate_up` bank stores gate|up in
/// ONE tensor (so `up_exps == gate_exps`), and a split bank always stores both at the same type —
/// the executor's own `native` resolution asserts that. Whatever this predicate lets through, the
/// executor still re-checks at dispatch time and REPLAYS the elided op if the tier it actually
/// takes (paged, `INFR_ROCM_NO_I8`, `moe_id_rows = 0`) cannot carry the fold.
fn fuse_experts_ok(
    gate: DType,
    _up: DType,
    down: DType,
    rocm: &infr_core::config::RocmCfg,
) -> bool {
    moe_i8_enabled(rocm)
        && MOE_ID_ROWS.clamped(rocm.moe_id_rows) > 0
        && moe_native_fmt(gate).is_some()
        && moe_native_fmt(down).is_some()
}

fn decode_fusion(g: &Graph, engine: &infr_core::config::Config) -> DecodeFusion {
    // The two decode folds (`RmsNorm → int8 Linear` normalize-in-kernel, `int8 Linear → Add`
    // residual epilogue) are the shared `plan_fusions` rmsnorm_linear + linear_add passes, gated to
    // the int8-decode GEMV coverage. Escape hatches: `INFR_ROCM_NO_FUSE_NORM`/`INFR_ROCM_NO_FUSE_ADD`.
    // The int8-coverage predicate now needs the ROCm tier config (`INFR_ROCM_NO_I8` disables the
    // int8 GEMV, and with it BOTH folds), so it is a closure borrowing `engine` rather than the
    // former `static fn(DType) -> bool`. `FusionCfg::weight_ok` is a `&dyn Fn`, so one binding
    // serves both passes; it is built ONCE per forward, not per op.
    let weight_ok = |dt: DType| fuse_weight_ok(dt, &engine.kernels.rocm);
    let experts_ok =
        |g_: DType, u: DType, d: DType| fuse_experts_ok(g_, u, d, &engine.kernels.rocm);
    let cfg = infr_core::fusion::FusionCfg {
        linear_add: Some(infr_core::fusion::LinearAddCfg {
            weight_ok: &weight_ok,
            // `INFR_ROCM_NO_FUSE_ADD` (config `kernels.rocm.fuse_add`, positive polarity):
            // PRESENCE of the env key — including `=0` — turns the fold off.
            enabled: engine.kernels.rocm.fuse_add,
        }),
        rmsnorm_linear: Some(infr_core::fusion::RmsNormLinearCfg {
            weight_ok: &weight_ok,
            // F1c: a single-row `MoeFfn` is a fusable consumer too — ROCm is the one backend whose
            // MoE arm can produce the normalized row itself.
            moe_ok: Some(&experts_ok),
            // `INFR_ROCM_NO_FUSE_NORM` (config `kernels.rocm.fuse_norm`), same polarity.
            enabled: engine.kernels.rocm.fuse_norm,
        }),
        moe_add: Some(infr_core::fusion::MoeAddCfg {
            experts_ok: &experts_ok,
            // Same hatch as the dense `Linear → Add` fold: it is the same rewrite.
            enabled: engine.kernels.rocm.fuse_add,
        }),
        // F1d. The shared pass has no per-backend predicate hook here (its own gate is fixed: an
        // Internal f16 rope `dst` feeding an immediately-following `WriteKv` into an f16 cache), so
        // ROCm's coverage is applied as a POST-FILTER below — `kv_fuse_ok` drops every planned
        // entry this backend's `qk_norm_rope` cannot reproduce exactly, and un-skips its `WriteKv`.
        kv_write: engine.kernels.rocm.fuse_kv_write,
    };
    let plan = infr_core::fusion::plan_fusions(g, &cfg);
    // Both residual folds land in ONE map: they are keyed by op index and carry the same
    // `(residual, add dst)` payload, and `run_op` looks the entry up by index without caring whether
    // the producer was a `Linear` or a `MoeFfn`. The two key sets are disjoint by construction (an
    // `Add` at `i + 1` has exactly one producer at `i`).
    let mut add = plan.linear_add;
    add.extend(plan.moe_add);
    let mut skip = plan.skip;
    // F1d post-filter: keep only the planned K writes this backend's fused kernel reproduces
    // EXACTLY, and replay the standalone `write_kv` for the rest (un-skip its op index).
    let mut kv = HashMap::new();
    for (i, (cache, row)) in plan.kv_write {
        match kv_fuse_ok(g, i, cache, row) {
            Some(f) => {
                kv.insert(i, f);
            }
            None => {
                skip.remove(&(i + 1));
            }
        }
    }
    // E2B per-layer inp_gate peephole: Linear(f32, m ≤ 4) → GatedAct(Gelu, strided up)
    // collapses into one `e2b_gate` dispatch. The pattern only fires for E2B models
    // where the weight is f32, m is small (decode or a micro-prefill), and the up buffer
    // carries a per-row stride.
    let mut e2b_gate = HashMap::new();
    for i in 0..g.ops.len().saturating_sub(1) {
        if skip.contains(&i) || skip.contains(&(i + 1)) {
            continue;
        }
        let Op::Linear { weight, dst, m, .. } = g.ops[i] else {
            continue;
        };
        if g.desc(weight).dtype != DType::F32 || m > 4 {
            continue;
        }
        let Op::GatedAct {
            gate,
            up,
            dst: ga_dst,
            act,
            up_off,
            up_stride,
            gate_stride,
            gate_block_width,
            ..
        } = g.ops[i + 1]
        else {
            continue;
        };
        if act != Activation::Gelu
            || gate != dst
            || ga_dst != dst
            || up_stride == 0
            || gate_stride != 0
            || gate_block_width != 0
        {
            continue;
        }
        e2b_gate.insert(
            i,
            E2bGateFuse {
                up,
                up_off,
                up_stride,
            },
        );
        skip.insert(i + 1);
    }
    DecodeFusion {
        norm: plan.rmsnorm_linear,
        add,
        kv,
        skip,
        e2b_gate,
    }
}

/// F1d coverage filter for one planned `kv_write` entry (rope at op `i`, absorbed `WriteKv` at
/// `i + 1`). `None` = decline, and the standalone `write_kv` runs as before.
///
/// The four gates, each a way the fused kernel could differ from the pair it replaces:
///
/// 1. **`Op::QkNormRope` only.** The shared pass also matches an f16-out `Op::Rope` (llama's K path),
///    but ROCm's `rope` kernel rotates an f32 buffer IN PLACE after a DtoD copy — it has no output
///    pointer to redirect, let alone an f16 one. Those keep the split pair.
/// 2. **The rope must tile the cache row exactly**: `row_stride == n_head * head_dim` and the
///    `WriteKv` must cover the same `rows`. The elided kernel copied `rows × row_stride` packed
///    elements; the fused kernel's grid is `rows × n_head` waves each owning `head_dim` elements, so
///    anything else would leave part of the row unwritten.
/// 3. **NO RING WRAP.** ROCm reports `kv_swa_ring: false`, so the seam gives it full-context caches
///    and `write_kv` indexes `pos + row` with no modulo at all. The shared plan hands back
///    `pos % cap_rows`; requiring it to EQUAL `pos` is what makes "the fused variant does what the
///    write path does today" a checked property rather than a comment. If ROCm ever gains ring
///    semantics, `write_kv` learns them first and this gate is what forces the question.
/// 4. **Live range.** The plan carries no live-range bound (Vulkan's record-once decode REQUIRES its
///    K write fused, so it cannot afford one), but eliding the write leaves the rope's `dst` scratch
///    unwritten — safe only if nothing reads it before it is next rewritten.
fn kv_fuse_ok(g: &Graph, i: usize, cache: TensorId, row: usize) -> Option<KvFuse> {
    let Op::QkNormRope {
        dst,
        rows,
        n_head,
        head_dim,
        ..
    } = g.ops[i]
    else {
        return None; // (1)
    };
    let Some(&Op::WriteKv {
        pos,
        rows: w_rows,
        row_stride,
        ..
    }) = g.ops.get(i + 1)
    else {
        return None;
    };
    if row_stride != n_head * head_dim || w_rows != rows {
        return None; // (2)
    }
    if row != pos as usize {
        return None; // (3): the plan wrapped this write; ROCm's write path does not wrap.
    }
    if !infr_core::fusion::dst_only_read_by_next(g, i + 2, dst) {
        return None; // (4)
    }
    // The row the fused kernel writes is the PLAN's row, not a locally re-derived `pos` — gate (3)
    // is what makes those the same number, so dropping it cannot quietly leave a correct write
    // behind. (`row == pos` there, so the cast is exact.)
    Some(KvFuse {
        cache,
        row: row as u32,
        stride: row_stride,
    })
}

// ── Per-op GPU-time profiler (P1; unified onto the shared seam in U3) ────────
//
// ROCm had no per-op profiler at all, which is why every perf slice before P1 had to reason about
// where a token went from launch COUNTS and isolated micro-benches.
//
// This type is now only the ACQUISITION half — recording HIP timing events around each op and
// resolving them to microseconds. The accounting, the label grammar, the report format and the
// process-wide aggregate all live in `infr_core::prof`, shared with vulkan/metal/cpu.
//
// Why HIP events and not a host timer: the ops all queue on ONE stream with no per-op sync (see
// `execute_graph`), so a host `Instant` around `run_op` measures ENQUEUE, not execution. Putting a
// `hipStreamSynchronize` per op instead would measure execution but serialize the pipeline and add
// the very ~2.7 µs floor F4 measured to every sample. A recorded timing-event pair costs the stream
// nothing but a timestamp write, and `hipEventElapsedTime` reads the command processor's own
// clocks AFTER the forward's existing barrier — so the profile is of the shipping schedule.
struct OpProf {
    /// One entry per dispatched op, in walk order: `(label, start, end)`.
    spans: Vec<(String, ffi::hipEvent_t, ffi::hipEvent_t)>,
}

impl OpProf {
    fn new() -> Self {
        OpProf { spans: Vec::new() }
    }

    /// Create a TIMING event (no `HIP_EVENT_DISABLE_TIMING`) and record it on `stream`.
    fn mark(stream: ffi::hipStream_t) -> ffi::hipEvent_t {
        let mut ev: ffi::hipEvent_t = std::ptr::null_mut();
        unsafe {
            ffi::hipEventCreateWithFlags(&mut ev, 0);
            ffi::hipEventRecord(ev, stream);
        }
        ev
    }

    fn begin(&mut self, label: String, stream: ffi::hipStream_t) {
        let start = Self::mark(stream);
        self.spans.push((label, start, std::ptr::null_mut()));
    }

    fn end(&mut self, stream: ffi::hipStream_t) {
        let ev = Self::mark(stream);
        if let Some(last) = self.spans.last_mut() {
            last.2 = ev;
        }
    }

    /// Read every span (the caller has already synchronized the stream), hand the durations to the
    /// shared collector to account and report, and destroy the events.
    fn flush(self) {
        let mut p = infr_core::prof::OpProf::new("rocm", infr_core::prof::Unit::Device);
        for (label, start, end) in self.spans {
            let mut ms = 0.0f32;
            if !end.is_null()
                && unsafe { ffi::hipEventElapsedTime(&mut ms, start, end) } == HIP_SUCCESS
            {
                p.add(label, ms as f64 * 1000.0);
            }
            unsafe {
                ffi::hipEventDestroy(start);
                if !end.is_null() {
                    ffi::hipEventDestroy(end);
                }
            }
        }
        p.flush();
    }
}

// ── Per-op dispatch ──────────────────────────────────────────────────────────

/// This backend's hook into the shared op walk ([`infr_core::exec::run_ops`]): the ambient state
/// [`run_op`] needs, plus the per-op fusion payload lookup the walk used to do inline. One method
/// over the whole `&Op` — the `match Op::` and every HIP decision inside it stay below, in
/// [`run_op`]. Monomorphized by `run_ops`, so the per-op call is the same direct call the
/// hand-written loop made.
struct RocmDispatch<'a, 'b, 'c> {
    g: &'a Graph,
    bindings: &'a Bindings<'b>,
    pipelines: &'a Pipelines,
    fusion: &'a DecodeFusion,
    ctx: &'a mut ExecCtx<'c>,
    /// `Some` only under `kernels.rocm.prof_ops` (P1).
    prof: Option<&'a mut OpProf>,
}

impl infr_core::exec::OpDispatch for RocmDispatch<'_, '_, '_> {
    fn dispatch(&mut self, i: usize, op: &Op) -> Result<()> {
        // The event pair brackets EVERYTHING the op queues — every kernel of a multi-dispatch arm
        // (split-KV attention, the MoE expert loop) plus any copy — which is what "where does the
        // forward go" wants, not per-kernel accounting.
        if let Some(p) = self.prof.as_deref_mut() {
            p.begin(infr_core::prof::op_label(op, self.g), self.ctx.stream);
        }
        let r = run_op(
            op,
            self.g,
            self.bindings,
            self.pipelines,
            self.ctx,
            self.fusion.norm.get(&i).copied(),
            self.fusion.add.get(&i).copied(),
            self.fusion.kv.get(&i).copied(),
            self.fusion.e2b_gate.get(&i).copied(),
        );
        if let Some(p) = self.prof.as_deref_mut() {
            p.end(self.ctx.stream);
        }
        r
    }
}

macro_rules! args { ($($e:expr),* $(,)?) => { vec![$($e),*] }; }

#[allow(clippy::too_many_arguments)]
fn run_op(
    op: &Op,
    g: &Graph,
    bindings: &Bindings,
    pipelines: &Pipelines,
    ctx: &mut ExecCtx,
    norm_fuse: Option<(TensorId, TensorId, f32)>,
    add_fuse: Option<(TensorId, TensorId)>,
    kv_fuse: Option<KvFuse>,
    e2b_fuse: Option<E2bGateFuse>,
) -> Result<()> {
    // F1b: the sibling-GEMV quant memo is valid only for the op IMMEDIATELY after the pass that
    // published it. Clearing it here and republishing it from the int8 GEMV branch alone means any
    // op that could have rewritten the activation row invalidates the memo without having to know
    // about it. (Same discipline as Vulkan's `mmv_memo`.)
    let qmemo_prev = ctx.qmemo.take();
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
            // F5 fully-overwritten: `rmsnorm`'s block per row stores `d[i]` for every
            // `i = tid, tid+nt, … < dim`, so all `rows * dim` slots land.
            let dd = ctx.uninit_dev(rows as usize * dim as usize);
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
                (native_decode_fmt(wdt), native_i8_fmt(wdt, ctx.rocm))
            {
                // Int8-activation dp4a decode (Phase 4): quantize the `m×in_f` activation to int8
                // ONCE (`quant_i8_32`, per-32-block scale), then integer-dot against the decoded
                // weight codes (scale-after) via `linear_i8_*`. Drops the Phase-3 per-element f16
                // round-trip → decode is no longer ALU-bound. `bpb == bpb_i8` (same weight layout);
                // the bound quant buffer is pre-advanced past `w_off`, a whole number of output
                // rows × `in_f` (a multiple of `qpb`), so `(w_off/qpb)*bpb` is exact.
                debug_assert_eq!(bpb, bpb_i8);
                // Slice 37: a spilled bank is streamed into a resident VRAM staging slot ahead of
                // this GEMV (the compute stream already waits on its fill inside `weight_staged_ptr`);
                // otherwise `wptr` is the resident VRAM / Slice-35 host-alias pointer as before.
                let (wptr, wt_staged) = match weight_staged_ptr(ctx, weight, bindings)? {
                    Some(p) => (p, true),
                    None => (ctx.ensure_device(weight, g, bindings)?, false),
                };
                let mu = m as usize;
                let inu = in_f as usize;
                let ou = out_f as usize;
                let blk_off = (w_off as usize / qpb) * bpb;
                let wptr_off = unsafe { (wptr as *mut u8).add(blk_off) as *mut c_void };
                // The plain int8 quant reads the (already-normalized) `x`; the RmsNorm→Linear fusion
                // instead reads the RAW pre-norm row inside `rmsnorm_quant_i8_32`, so `x` (the elided
                // norm's output) is never materialized on device.
                let bx_ptr = if norm_fuse.is_none() {
                    ctx.ensure_device(x, g, bindings)?;
                    ctx.dev[x.0 as usize].as_ref().unwrap().ptr
                } else {
                    std::ptr::null_mut()
                };
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
                        // F5: `beta = 0` means BLAS does not reference C, so this IS fully written —
                        // but it is left on the calloc draw deliberately. The arm is opt-in
                        // (`INFR_ROCM_BLAS=1`), measured a net loss, and ships off, so there is no
                        // dispatch cost here worth trading against depending on a library's
                        // handling of `0 * NaN` for the poison build.
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
                        // ── Resolve the activation device pointer (shared by both paths) ──
                        let (q_src, q_norm, q_eps) = match norm_fuse {
                            Some((x_raw, norm_w, eps)) => {
                                let wnptr = ctx.dequant_weight_or_cache(norm_w, g, bindings)?;
                                let xrp = ctx.ensure_device(x_raw, g, bindings)?;
                                ((x_raw, xrp), Some((norm_w, wnptr)), eps)
                            }
                            None => ((x, bx_ptr), None, 0.0),
                        };
                        // ── f16 A_GLOBAL WMMA prefill (Slice S2) ──────────────────────
                        // OPT-IN behind `INFR_ROCM_A_GLOBAL=1`. Skip int8 quant entirely —
                        // convert f32 activations to f16 and multiply directly with
                        // `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`. Q4_K only.
                        if m > 1 && wdt == DType::Q4K && ctx.rocm.a_global && !ctx.rocm.no_wmma {
                            // Convert f32 activations → f16 (one element per thread).
                            let a_elems = mu * inu;
                            let a_f16 = ctx.pool_buf(a_elems * 2, false);
                            dispatch_1d(
                                pipelines,
                                ctx.stream,
                                "convert_f32_to_f16",
                                a_elems as u32,
                                256,
                                args![
                                    arg_ptr(q_src.1),
                                    arg_ptr(a_f16.ptr),
                                    arg_i32(a_elems as i32),
                                ],
                            )?;
                            // Allocate output (same as the int8 path's fully-overwritten dst).
                            let dd = match add_fuse {
                                Some((_, add_dst)) => {
                                    ctx.ensure_device(add_dst, g, bindings)?;
                                    let b = ctx.dev[add_dst.0 as usize].as_ref().unwrap();
                                    crate::RocmBuffer {
                                        ptr: b.ptr,
                                        len: b.len,
                                        owned: false,
                                        host_ptr: std::ptr::null_mut(),
                                        uid: b.uid,
                                    }
                                }
                                None => ctx.uninit_dev(mu * ou),
                            };
                            // Grid = (ceil(out_f/32), ceil(m/32)), one wave32 per 32×32 tile.
                            dispatch_grid(
                                pipelines,
                                ctx.stream,
                                "wmma_f16_q4k_2x2",
                                out_f.div_ceil(32),
                                m.div_ceil(32),
                                32,
                                args![
                                    arg_ptr(a_f16.ptr),
                                    arg_ptr(wptr_off),
                                    arg_ptr(dd.ptr),
                                    arg_i32(m as i32),
                                    arg_i32(in_f as i32),
                                    arg_i32(out_f as i32),
                                ],
                            )?;
                            let dd_ptr = dd.ptr;
                            if add_fuse.is_none() {
                                ctx.dev[dst.0 as usize] = Some(dd);
                            }
                            // No qmemo: the f16 path has no int8 codes to reuse.
                            // But we MUST clear any stale memo so a later GEMV over the same
                            // row does not pick up int8 codes that were never written this pass.
                            if dd_ptr != q_src.1 {
                                ctx.qmemo = None;
                            }
                        } else {
                            // Int8-activation dp4a path: quantize the `m×in_f` activation to int8 ONCE
                            // (`quant_i8_32`, per-32-block scale), then integer-dot against the decoded
                            // weight codes (scale-after). `bpb == bpb_i8` (same layout). The int8 codes /
                            // scales are drawn from the scratch pool (fully written before any read → `out`,
                            // un-cleared) and stay live until end-of-forward, so the async GEMM/GEMV that
                            // reads them never races a pool reuse.
                            let nb = inu / 32; // in_f is 32-aligned for every covered format

                            // F1b: (q_src, q_norm, q_eps) resolved above — shared by both the f16
                            // and int8 paths so the quant memo key is consistent.
                            let qkey = QuantKey {
                                src: q_src,
                                norm: q_norm,
                                eps_bits: q_eps.to_bits(),
                                m,
                                in_f,
                            };
                            let hit = qmemo_prev.filter(|p| p.key == qkey);
                            let (qx_ptr, xs_ptr) = match hit {
                                // Sibling projection off the same activation row: the previous GEMV's
                                // codes/scales ARE this one's, bit for bit. Skip the pass.
                                Some(p) => (p.qx, p.xs),
                                None => {
                                    let qx = ctx.pool_buf((mu * inu).max(1), false);
                                    let xs = ctx.pool_buf((mu * nb * 4).max(1), false);
                                    if norm_fuse.is_some() {
                                        // Slice-32 RmsNorm→Linear: one block per row reduces the
                                        // sum-of-squares over the RAW row, then int8-quantizes the
                                        // normalized row in registers (bit-identical to `rmsnorm` then
                                        // `quant_i8_32`), killing the `rmsnorm` launch + the
                                        // normalized-activation DRAM round-trip.
                                        dispatch_grid(
                                            pipelines,
                                            ctx.stream,
                                            "rmsnorm_quant_i8_32",
                                            m,
                                            1,
                                            256,
                                            args![
                                                arg_ptr(q_src.1),
                                                arg_ptr(q_norm.unwrap().1),
                                                arg_ptr(qx.ptr),
                                                arg_ptr(xs.ptr),
                                                // No normalized-row output: a dense GEMV consumes the
                                                // int8 codes only (the F1c `xn` arm is the MoE router's).
                                                arg_ptr(std::ptr::null_mut()),
                                                arg_i32(m as i32),
                                                arg_i32(in_f as i32),
                                                arg_f32(q_eps),
                                            ],
                                        )?;
                                    } else {
                                        dispatch_1d(
                                            pipelines,
                                            ctx.stream,
                                            "quant_i8_32",
                                            (mu * nb) as u32,
                                            256,
                                            args![
                                                arg_ptr(q_src.1),
                                                arg_ptr(qx.ptr),
                                                arg_ptr(xs.ptr),
                                                arg_i32(m as i32),
                                                arg_i32(in_f as i32),
                                            ],
                                        )?;
                                    }
                                    (qx.ptr, xs.ptr)
                                }
                            };
                            // Slice-32 Linear→Add: when the following residual Add is fused in, the GEMV
                            // writes (and adds) straight into the residual stream's live buffer — no
                            // fresh zeroed dst, no standalone `add`. `resid_ptr` is null otherwise, so
                            // the GEMV epilogue is bit-identical to the pre-fusion write.
                            let resid_ptr = match add_fuse {
                                Some((resid, _)) => ctx.ensure_device(resid, g, bindings)?,
                                None => std::ptr::null_mut(),
                            };
                            let dd = match add_fuse {
                                Some((_, add_dst)) => {
                                    ctx.ensure_device(add_dst, g, bindings)?;
                                    let b = ctx.dev[add_dst.0 as usize].as_ref().unwrap();
                                    crate::RocmBuffer {
                                        ptr: b.ptr,
                                        len: b.len,
                                        owned: false,
                                        host_ptr: std::ptr::null_mut(),
                                        uid: b.uid,
                                    }
                                }
                                // F5 fully-overwritten: all three arms below tile `dst[m, out_f]`
                                // exactly — the mrow GEMV writes every `ov[r] < out_f` once (grid
                                // `ceil(out_f/I8_MROW) × m`, the `ov[r] != o0` guard suppressing the
                                // clamped duplicate), and the WMMA / coop kernels store every
                                // `(re < m, col < out_f)` of their tile.
                                None => ctx.uninit_dev(mu * ou),
                            };
                            // Slice-28: Q4_K prefill (m>1) can OPT IN (`INFR_ROCM_COOP=1`) to the
                            // cooperative decode-once GEMM (multi-warp threadblock, LDS-shared weight
                            // tile). It is bit-faithful to `wmma_i8_q4k_2x1` (goldens hold) but measured
                            // a regression on gfx1100 (see `q4k_coop_kernel`), so the DEFAULT stays the
                            // Slice-27 pipe. When not opted in, this falls through to the pipe / GEMV.
                            let coop =
                                (m > 1 && wdt == DType::Q4K && !ctx.rocm.no_wmma && ctx.rocm.i8)
                                    .then(|| q4k_coop_kernel(ctx.rocm))
                                    .flatten();
                            match coop {
                                Some((coop_kernel, bm, bn, threads)) => {
                                    // Grid = (ceil(out_f/BN), ceil(m/BM)); one multi-warp threadblock per
                                    // BM×BN output tile decodes its BN-column weight tile into LDS once and
                                    // reuses it across all BM rows. m/out_f edges are masked in-kernel.
                                    dispatch_grid(
                                        pipelines,
                                        ctx.stream,
                                        coop_kernel,
                                        out_f.div_ceil(bn),
                                        m.div_ceil(bm),
                                        threads,
                                        args![
                                            arg_ptr(qx_ptr),
                                            arg_ptr(xs_ptr),
                                            arg_ptr(wptr_off),
                                            arg_ptr(dd.ptr),
                                            arg_i32(m as i32),
                                            arg_i32(in_f as i32),
                                            arg_i32(out_f as i32),
                                        ],
                                    )?;
                                }
                                None => match (m > 1)
                                    .then(|| native_wmma_fmt(wdt, out_f, ctx.rocm))
                                    .flatten()
                                {
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
                                                arg_ptr(qx_ptr),
                                                arg_ptr(xs_ptr),
                                                arg_ptr(wptr_off),
                                                arg_ptr(dd.ptr),
                                                arg_i32(m as i32),
                                                arg_i32(in_f as i32),
                                                arg_i32(out_f as i32),
                                            ],
                                        )?;
                                    }
                                    None => {
                                        // Decode (m==1) or WMMA disabled: the dp4a GEMV. Grid =
                                        // (out_f / rows-per-wave, m): one wave32 block per (group of
                                        // output rows, activation row). `resid_ptr` (null unless the
                                        // Slice-32 residual Add is fused) folds the add into the
                                        // epilogue.
                                        dispatch_grid(
                                            pipelines,
                                            ctx.stream,
                                            i8_kernel,
                                            out_f.div_ceil(i8_gemv_mrow(i8_kernel)),
                                            m,
                                            32,
                                            args![
                                                arg_ptr(qx_ptr),
                                                arg_ptr(xs_ptr),
                                                arg_ptr(wptr_off),
                                                arg_ptr(dd.ptr),
                                                arg_ptr(resid_ptr),
                                                arg_i32(m as i32),
                                                arg_i32(in_f as i32),
                                                arg_i32(out_f as i32),
                                            ],
                                        )?;
                                    }
                                },
                            }
                            let dd_ptr = dd.ptr;
                            // When the residual Add is fused, `dd` aliases the residual stream buffer
                            // (already mapped in `ctx.dev` via `ensure_device(add_dst)`) and the result
                            // is written in place — nothing to remap. Otherwise publish the fresh dst.
                            if add_fuse.is_none() {
                                ctx.dev[dst.0 as usize] = Some(dd);
                            }
                            // F1b: publish the memo for the next op — UNLESS this GEMV just wrote the
                            // very row it quantized (the fused-residual epilogue writes into the
                            // residual stream, which is also the input norm's `x`). In that case the
                            // codes no longer describe the row and the memo must not survive.
                            if dd_ptr != q_src.1 {
                                ctx.qmemo = Some(QuantMemo {
                                    key: qkey,
                                    qx: qx_ptr,
                                    xs: xs_ptr,
                                });
                            }
                        }
                    }
                }
                // Slice 37: GEMV dispatched — record the slot free + prefetch the next spilled bank.
                if wt_staged {
                    weight_staged_done(ctx)?;
                }
            } else if let Some((qpb, bpb, kname, _)) = native_decode_fmt(wdt) {
                // Native in-kernel decode: read the RAW quant bytes (no f16 cache → VRAM drops).
                // The bound quant buffer is pre-advanced past `w_off`; `w_off` is always a whole
                // number of output rows × `in_f`, hence a multiple of `qpb`, so the block offset
                // `(w_off / qpb) * bpb` is exact.
                // Slice 37: same staging seam as the int8 path — a spilled bank is read from a
                // resident VRAM slot prefetched on the copy stream, not over PCIe in-kernel.
                let (wptr, wt_staged) = match weight_staged_ptr(ctx, weight, bindings)? {
                    Some(p) => (p, true),
                    None => (ctx.ensure_device(weight, g, bindings)?, false),
                };
                ctx.ensure_device(x, g, bindings)?;
                // F5 fully-overwritten: the native-decode GEMV runs one block per activation row
                // and stores `dst[row*out_f + o]` for every `o = tid, tid+blockDim, … < out_f`.
                let dd = ctx.uninit_dev(m as usize * out_f as usize);
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
                if wt_staged {
                    weight_staged_done(ctx)?;
                }
            } else if let Some(f) = e2b_fuse {
                // E2B per-layer inp_gate fusion: `e2b_gate` combines Linear(f32→f16)
                // + GatedAct(gelu, stride) into one kernel dispatch. The up buffer
                // is resolved here; w_off must be 0 (the peephole only fires when
                // weight dtype is F32 with no concatenated upload).
                let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
                ctx.ensure_device(x, g, bindings)?;
                ctx.ensure_device(f.up, g, bindings)?;
                // F5 fully-overwritten: same one-thread-per-output tiling as `linear_f16`.
                let dd = ctx.uninit_dev(m as usize * out_f as usize);
                let bx = ctx.dev[x.0 as usize].as_ref().unwrap();
                let bu = ctx.dev[f.up.0 as usize].as_ref().unwrap();
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "e2b_gate",
                    m * out_f,
                    256,
                    args![
                        arg_ptr(bx.ptr),
                        arg_ptr(wptr),
                        arg_ptr(bu.ptr),
                        arg_ptr(dd.ptr),
                        arg_i32(m as i32),
                        arg_i32(in_f as i32),
                        arg_i32(out_f as i32),
                        arg_i32(f.up_off as i32),
                        arg_i32(f.up_stride as i32),
                    ],
                )?;
                ctx.dev[dst.0 as usize] = Some(dd);
            } else {
                let wptr = ctx.dequant_weight_or_cache(weight, g, bindings)?;
                ctx.ensure_device(x, g, bindings)?;
                // F5 fully-overwritten: `linear_f16` is the same one-block-per-row tiling.
                let dd = ctx.uninit_dev(m as usize * out_f as usize);
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
            // F5 fully-overwritten: one thread per row writes `dr[0..dim)` twice over (exp then
            // normalize), so every one of the `rows * dim` slots is stored.
            let dd = ctx.uninit_dev(rows as usize * dim as usize);
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
            // F5: `qk_norm` writes `dst[off + i]`, and `off` is built from the SOURCE stride —
            // `r * x_stride + h * head_dim`. Packed (`x_stride == 0`) that is exactly the
            // `rows * n_head * head_dim` dst, tiled once: fully overwritten. Strided it is not the
            // dst's own layout at all, so rows>1 would leave inter-row gaps unwritten. Only the
            // packed case takes the un-cleared draw.
            let dd = if x_stride == 0 {
                ctx.uninit_dev(rows as usize * n_head as usize * head_dim as usize)
            } else {
                ctx.zero_dev(rows as usize * n_head as usize * head_dim as usize)
            };
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
            // F5 fully-overwritten: one thread per (row, head) writes `dst[head*head_dim + i]` for
            // every `i < head_dim`; the grid covers all `rows * n_head` heads.
            let dd = ctx.uninit_dev(rows as usize * n_head as usize * head_dim as usize);
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
                // F5 PARTIAL WRITE — stays calloc. The clone below copies `min(dd.len, src.len)`
                // bytes, so a source shorter than `rows * stride_elems` leaves a tail the `rope`
                // dispatch does not write either (it only touches `[0, rope_dim)` of each head).
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
            //
            // F1d: when the `kv_write` peephole absorbed the following `WriteKv`, there is no `out`
            // at all — the kernel casts each element to f16 and stores it in the KV cache row the
            // elided `write_kv` would have filled. No scratch draw, no zeroing memset, no round
            // trip, and `ctx.dev[dst]` deliberately stays unset (nothing may read it — `kv_fuse_ok`
            // gate 4). The cache is the BOUND buffer, exactly as `Op::WriteKv` takes it.
            let dd = match kv_fuse {
                Some(_) => None,
                // F5: the wave per (row, head) covers `[0, head_dim)` exactly once — the
                // pass-through loop takes `[rope_dim, head_dim)` strided by 32, the rotation loop
                // writes both `i` and `i + half` for every `i < half`. That is a full tiling
                // PROVIDED `rope_dim` is even; an odd one would leave `rope_dim - 1` in neither
                // loop's range, so it keeps the calloc draw. The write base `doff = head*head_dim`
                // is the packed dst regardless of `x_stride`, and the grid covers every head.
                None => {
                    let n = rows as usize * n_head as usize * head_dim as usize;
                    Some(if rope_dim % 2 == 0 {
                        ctx.uninit_dev(n)
                    } else {
                        ctx.zero_dev(n)
                    })
                }
            };
            let (kv_ptr, kv_row, kv_stride) = match kv_fuse {
                Some(f) => (
                    rocm_buf(bindings.get(f.cache).expect("rocm: unbound KV cache")).ptr,
                    f.row,
                    f.stride,
                ),
                None => (std::ptr::null_mut(), 0, 0),
            };
            let qnr_args = args![
                arg_ptr(bx_ptr),
                arg_ptr(wptr),
                arg_ptr(bp_ptr),
                arg_ptr(ff_ptr),
                arg_ptr(dd.as_ref().map_or(std::ptr::null_mut(), |d| d.ptr)),
                arg_i32(rows as i32),
                arg_i32(n_head as i32),
                arg_i32(head_dim as i32),
                arg_i32(rope_dim as i32),
                arg_f32(eps),
                arg_f32(theta),
                arg_i32(x_stride as i32),
                arg_ptr(kv_ptr),
                arg_i32(kv_row as i32),
                arg_i32(kv_stride as i32),
            ];
            // One 32-lane WAVE per (row, head): grid = rows*n_head blocks of 32 threads. The kernel
            // reads `blockIdx.x` as the head index, so pass total*32 with block=32.
            dispatch_1d(
                pipelines,
                ctx.stream,
                "qk_norm_rope",
                total * 32,
                32,
                qnr_args,
            )?;
            if let Some(d) = dd {
                ctx.dev[dst.0 as usize] = Some(d);
            }
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
            let cache_dtype = g.desc(cache).dtype;
            if cache_dtype == DType::Q8_0 {
                // Q8_0 planar KV cache: quantize f32 → int8 codes + f16 scales, one 32-lane
                // wave per 32-element block. The reader (`q8kv_decode` inline in the attention
                // flash kernel) reads this exact planar layout.
                let n = (rows * row_stride) as i32;
                let off = (pos as i32) * (row_stride as i32);
                let cap = g.desc(cache).shape[0] as i32;
                let total = n as u32;
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "store_q8",
                    total,
                    32,
                    args![
                        arg_ptr(bs.ptr),
                        arg_ptr(bc.ptr),
                        arg_i32(n),
                        arg_i32(off),
                        arg_i32(cap),
                        arg_i32(0), // src_off
                    ],
                )?;
            } else if cache_dtype == DType::Q4_0 {
                // Q4_0 GGUF KV cache: quantize f32 → 4-bit codes + f16 scale per 32-element
                // block. The reader (`q40kv_decode` inline in the attention flash kernel)
                // reads this exact GGUF block layout.
                let n = (rows * row_stride) as i32;
                let off = (pos as i32) * (row_stride as i32);
                let cap = g.desc(cache).shape[0] as i32;
                let total = n as u32;
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "store_kv_q4_0",
                    total,
                    32,
                    args![
                        arg_ptr(bs.ptr),
                        arg_ptr(bc.ptr),
                        arg_i32(n),
                        arg_i32(off),
                        arg_i32(cap),
                        arg_i32(0), // src_off
                    ],
                )?;
            } else {
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "write_kv",
                    rows * row_stride, // one thread per (row, element): decode fans across CUs
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
            // F5 fully-overwritten: both arms write `dst[head*head_dim + d]` for every `d`
            // (`attention` partitions the head dims across the wave's lanes with `d < head_dim`;
            // `attention_split_combine` uses the identical lane partition), over a grid that covers
            // every `head < rows * n_head`.
            let dd = ctx.uninit_dev(rows as usize * n_head as usize * head_dim as usize);
            let bk = rocm_buf(bindings.get(k_cache).expect("rocm: unbound K cache"));
            let bv = rocm_buf(bindings.get(v_cache).expect("rocm: unbound V cache"));
            let (bk_ptr, bv_ptr) = (bk.ptr, bv.ptr);
            let bq_ptr = ctx.dev[q.0 as usize].as_ref().unwrap().ptr;
            let dd_ptr = dd.ptr;
            // KV quant state: dtype encodes any quantized format (Q8_0, block, turbo), and k_cap
            // is the total element count (rows * n_kv * head_dim) per layer — the flash kernel
            // uses it to locate the Q8_0 scale section at offset cap/4 uint32 words.
            let k_dtype = match g.desc(k_cache).dtype {
                DType::Q8_0 => 1,
                DType::Q4_0 => 2,
                DType::Q4_1 => 3,
                DType::Q5_0 => 4,
                DType::Q5_1 => 5,
                _ => 0, // F16 (or other: the inline path handles F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1)
            };
            let v_dtype = match g.desc(v_cache).dtype {
                DType::Q8_0 => 1,
                DType::Q4_0 => 2,
                DType::Q4_1 => 3,
                DType::Q5_0 => 4,
                DType::Q5_1 => 5,
                _ => 0,
            };
            let k_cap = g.desc(k_cache).shape[0] as i32;
            let (mt, swa): (i32, i32) = match mask {
                AttnMask::Causal => (0, 0),
                AttnMask::SlidingWindow(w) => (1, w as i32),
                AttnMask::Canvas { lo } => (2, lo as i32),
            };

            // Split-KV (flash-decoding) for DECODE (rows==1). The single-wave `attention` kernel runs
            // ONE wave per (row, head) that scans ALL kv serially — fine at low depth, but at long
            // context that one wave crawls while ~95 CUs idle. Split-KV partitions kv into `n_chunks`
            // contiguous chunks, launches one wave per (row, head, chunk) to compute per-chunk
            // online-softmax partials, then a combine wave merges them. Adaptive chunking via the
            // shared `infr_core::tier` policy ([`ATTN_SPLIT`]): aim ~32 chunks/head, each 64..512
            // keys. Only worth it when rows==1 AND the derived n_chunks>1 (short-context decode →
            // n_chunks==1 → plain kernel, no scratch, no combine). Prefill (rows>1) already fills
            // the grid with rows*n_head waves and stays on the plain kernel. No chunk-COUNT cap
            // here: `attention_split_combine` walks n_chunks straight out of global memory (no
            // fixed shared array), unlike Vulkan's attn_combine.
            let heads = rows as usize * n_head as usize;
            let kvl = kv_len as usize;
            let chunk_size = infr_core::tier::adaptive_chunk(kvl, &ATTN_SPLIT);
            let n_chunks = infr_core::tier::n_chunks(kvl, chunk_size).max(1);
            // P6: the batched-prefetch DECODE variant, when one is instantiated for this
            // head_dim's lane count. `None` ⇒ the generic kernel below. Decode only (`rows == 1`):
            // prefill fills the grid with `rows*n_head` waves and is not request-starved, and
            // holding it to the generic kernel keeps pp byte-identical in dispatch structure.
            let pf = (rows == 1 && ctx.rocm.attn_pf)
                .then(|| attn_pf_npl(head_dim as usize))
                .flatten();
            if rows == 1 && n_chunks > 1 {
                let hd = head_dim as usize;
                let pm = ctx.pool_buf(heads * n_chunks * 4, false);
                let pl = ctx.pool_buf(heads * n_chunks * 4, false);
                let pacc = ctx.pool_buf(heads * n_chunks * hd * 4, false);
                // Pass 1: one wave per (row, head, chunk).
                // P7: one-pass online-softmax + one-key-per-lane when `attn_split_flash` is on
                // and head_dim is a multiple of 32 (the lane-per-key tile requires it).
                // The args are identical — only the internal algorithm differs.
                let use_flash = ctx.rocm.attn_split_flash && hd % 32 == 0;
                if use_flash {
                    let flash_kernel = match hd {
                        128 => "attention_split_partial_flash_hd128",
                        256 => "attention_split_partial_flash_hd256",
                        _ => "attention_split_partial_flash",
                    };
                    dispatch_blocks_smem(
                        pipelines,
                        ctx.stream,
                        flash_kernel,
                        (heads * n_chunks) as u32,
                        32,
                        (hd * 4) as u32, // smem: head_dim floats for Q staging
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
                } else {
                    dispatch_1d(
                        pipelines,
                        ctx.stream,
                        pf.map_or("attention_split_partial", |p| p.split_partial),
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
                }
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
            } else if let Some(t) = (rows > 1 && ctx.rocm.attn_flash)
                .then(|| attn_flash_tiling(head_dim as usize))
                .flatten()
            {
                // P1: TILED FLASH PREFILL. A workgroup owns `t.br()` consecutive query rows of one
                // head, streams the kv range in `t.bc`-key tiles through LDS, and runs a one-pass
                // online softmax — so K/V is read once per query TILE instead of once per query
                // ROW, and the causal/SWA-masked half of the score matrix is never visited. The
                // grid is `head * n_qtiles + qtile`, linearized because `dispatch_blocks_smem`
                // takes a 1-D block count.
                let br = t.br();
                let n_qtiles = (rows as usize).div_ceil(br);
                dispatch_blocks_smem(
                    pipelines,
                    ctx.stream,
                    "attention_prefill_flash",
                    (n_qtiles * n_head as usize) as u32,
                    (t.nw * 32) as u32,
                    t.smem(head_dim as usize) as u32,
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
                        arg_i32(k_dtype),
                        arg_i32(v_dtype),
                        arg_i32(k_cap),
                        arg_i32(t.bc as i32),
                        arg_i32(n_qtiles as i32),
                    ],
                )?;
            } else if rows > 1
                && ctx.rocm.attn_flash_wmma
                && !ctx.rocm.no_wmma
                && matches!(head_dim as usize, 64 | 128 | 256)
            {
                // P8: WMMA f16 flash prefill (OPT-IN). 256 threads (8 warps 4×2),
                // br=64 bc=64, K/V read directly from global. Scalar-ALU placeholder
                // until WMMA intrinsics land; gated behind `attn_flash_wmma` (default off)
                // so the P1 scalar flash still handles all default prefill traffic.
                let n_qtiles = (rows as usize).div_ceil(64);

                // Convert Q from f32 to f16 for the WMMA kernel.
                let q_elems = rows as usize * n_head as usize * head_dim as usize;
                let q_f16 = ctx.pool_buf(q_elems * 2, false);
                let bq_f16_ptr = q_f16.ptr;
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "convert_f32_to_f16",
                    q_elems as u32,
                    256,
                    args![
                        arg_ptr(bq_ptr),
                        arg_ptr(bq_f16_ptr),
                        arg_i32(q_elems as i32),
                    ],
                )?;

                dispatch_blocks_smem(
                    pipelines,
                    ctx.stream,
                    "attention_prefill_flash_wmma",
                    (n_qtiles * n_head as usize) as u32,
                    256,
                    0,
                    args![
                        arg_ptr(bq_f16_ptr),
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
                        arg_i32(64),
                        arg_i32(n_qtiles as i32),
                    ],
                )?;
            } else {
                // One 32-lane WAVE per (row, head): grid = rows*n_head blocks of 32 threads. The
                // kernel reads `blockIdx.x` as the head index, so pass heads*32 with block=32.
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    pf.map_or("attention", |p| p.plain),
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
            // F5 fully-overwritten: one thread per OUTPUT element, `dst[row*nff + i]`, over a grid
            // of `rows * nff` threads. Only the READ side is ever strided/interleaved.
            let dd = ctx.uninit_dev(rows as usize * nff as usize);
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
            // F5 fully-overwritten: same `gated_act` kernel, same one-thread-per-output tiling.
            let dd = ctx.uninit_dev(rows as usize * nff as usize);
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
            // F5 fully-overwritten: one thread per element, `dst[i] = a[i] + b[i]` for all `i < n`.
            let dd = ctx.uninit_dev(n as usize);
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
            // F5 fully-overwritten: one thread per row writes `dr[0..n)`, grid covers every row.
            let dd = ctx.uninit_dev(rows as usize * n as usize);
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
            // F5 fully-overwritten: one thread per element.
            let dd = ctx.uninit_dev(n as usize);
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
            // F5 fully-overwritten: one thread per row writes `dr[0..n)`, grid covers every row.
            let dd = ctx.uninit_dev(rows as usize * n as usize);
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
            // F5 fully-overwritten: one thread per element.
            let dd = ctx.uninit_dev(n as usize);
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
            // P7b: parallelise within each row (one block per row, blockDim.x threads per row)
            // instead of the old one-thread-per-row. A single-lane serial loop over n=2048
            // floats has exactly one memory request in flight; bs=min(n,256) threads per row
            // with float4 loads brings n/bs × 4 iterations per thread and n concurrent requests.
            let bs = (n as u32).min(256);
            dispatch_1d(
                pipelines,
                ctx.stream,
                "copy_strided",
                rows * bs,
                bs,
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
            // F5 fully-overwritten: one thread per row writes `dr[0..ne)` — both the f16 gather and
            // every native-decode `embed_gather_*` variant fill the whole row.
            let dd = ctx.uninit_dev(rows as usize * ne as usize);
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
            let dd = ctx.uninit_dev(rows as usize);
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            // P7c: multi-block reduction. n_chunks blocks of 256 threads each process
            // ARGMAX_CHUNK floats cooperatively, then one block per row merges the partials.
            // For decode (rows=1, vocab=151936 → 75 blocks) this fills the GPU instead of
            // stranding the whole 608 KB reduction on one block.
            const ARGMAX_CHUNK: usize = 2048;
            let n_chunks = (n as usize).div_ceil(ARGMAX_CHUNK);
            let pval = ctx.pool_buf(rows as usize * n_chunks * 4, false);
            let pidx = ctx.pool_buf(rows as usize * n_chunks * 4, false);
            dispatch_1d(
                pipelines,
                ctx.stream,
                "argmax_partial",
                rows as u32 * n_chunks as u32 * 256,
                256,
                args![
                    arg_ptr(bx_ptr),
                    arg_ptr(pval.ptr),
                    arg_ptr(pidx.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n as i32),
                    arg_i32(n_chunks as i32),
                ],
            )?;
            dispatch_1d(
                pipelines,
                ctx.stream,
                "argmax_combine",
                rows as u32 * 256,
                256,
                args![
                    arg_ptr(pval.ptr),
                    arg_ptr(pidx.ptr),
                    arg_ptr(dd.ptr),
                    arg_i32(rows as i32),
                    arg_i32(n_chunks as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }
        Op::ArgmaxProb {
            x,
            dst_id,
            dst_prob,
            n,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            let dd_id = ctx.uninit_dev(1);
            let dd_prob = ctx.uninit_dev(1);
            const ARGMAX_CHUNK: usize = 2048;
            let n_chunks = (n as usize).div_ceil(ARGMAX_CHUNK);
            let part = ctx.pool_buf(n_chunks * 3 * 4, false);
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            dispatch_1d(
                pipelines,
                ctx.stream,
                "argmax_prob_partial",
                n_chunks as u32 * 256,
                256,
                args![
                    arg_ptr(bx_ptr),
                    arg_ptr(part.ptr),
                    arg_i32(n as i32),
                    arg_i32(n_chunks as i32),
                ],
            )?;
            dispatch_1d(
                pipelines,
                ctx.stream,
                "argmax_prob_combine",
                256,
                256,
                args![
                    arg_ptr(part.ptr),
                    arg_ptr(dd_id.ptr),
                    arg_ptr(dd_prob.ptr),
                    arg_i32(n_chunks as i32),
                ],
            )?;
            ctx.dev[dst_id.0 as usize] = Some(dd_id);
            ctx.dev[dst_prob.0 as usize] = Some(dd_prob);
        }
        Op::Sample {
            x,
            u,
            dst,
            n,
            top_k,
            temp,
            top_p,
        } => {
            ctx.ensure_device(x, g, bindings)?;
            ctx.ensure_device(u, g, bindings)?;
            let dd = ctx.uninit_dev(1);
            let top_k = top_k as usize;
            let n_chunks: usize = 256; // fixed: 256 workgroups of 256 threads
            let cand_bytes = n_chunks * top_k * 2 * 4; // values + idx pairs, 4 bytes each
            let cand = ctx.pool_buf(cand_bytes, false);
            let bx_ptr = ctx.dev[x.0 as usize].as_ref().unwrap().ptr;
            let bu_ptr = ctx.dev[u.0 as usize].as_ref().unwrap().ptr;
            dispatch_1d(
                pipelines,
                ctx.stream,
                "sample_topk_partial",
                n_chunks as u32 * 256,
                256,
                args![
                    arg_ptr(bx_ptr),
                    arg_ptr(cand.ptr),
                    arg_i32(n as i32),
                    arg_i32(top_k as i32),
                ],
            )?;
            dispatch_1d(
                pipelines,
                ctx.stream,
                "sample_topk_combine",
                256,
                256,
                args![
                    arg_ptr(cand.ptr),
                    arg_ptr(bu_ptr),
                    arg_ptr(dd.ptr),
                    arg_i32((n_chunks * top_k) as i32),
                    arg_i32(top_k as i32),
                    arg_i32(temp.to_bits() as i32),
                    arg_i32(top_p.to_bits() as i32),
                ],
            )?;
            ctx.dev[dst.0 as usize] = Some(dd);
        }

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
                .filter(|_| fused_gate_up || up_dt == gate_dt)
                // The default int8 expert path only needs the per-FORMAT `moe_gate_up_act_i8_<gu>` /
                // `moe_down_i8_<dn>` kernels, which exist for every `moe_native_fmt` format. The
                // Phase-3 `moe_ffn_expert_<gu>_<dn>` cross product (`INFR_ROCM_NO_I8` only) is
                // instantiated for the reachable pairs alone (`MOE_EXPERT_PAIRS`), so when int8 is
                // OFF and this pair is not one of them, drop the whole expert to the dequant→f16
                // path — which is exactly that A/B switch's comparand. Filtering here (not at the
                // dispatch site) is what keeps `gw_ptr`/`dw_ptr` in step: the f16 kernel must be fed
                // dequantized banks, and the raw-vs-dequant pointer choice below reads `native`.
                .filter(|((gu, ..), (dn, ..))| {
                    moe_i8_enabled(ctx.rocm) || moe_expert_kernel(gu, dn).is_some()
                });

            // Paged MoE (Slice 33): when the bound `_exps` buffers are registered with the pager,
            // the expert banks live in HOST memory and each routed expert is paged into a VRAM
            // slot on demand (host routing already happens below). The bound buffer's device
            // pointer is the pager identity. Paging is only installed for native-covered banks
            // (the seam gates on it), so `is_paged` implies `native.is_some()`.
            let gate_buf_id =
                rocm_buf(bindings.get(gate_exps).expect("rocm: unbound gate_exps")).ptr as usize;
            let up_buf_id = if fused_gate_up {
                gate_buf_id
            } else {
                rocm_buf(bindings.get(up_exps).expect("rocm: unbound up_exps")).ptr as usize
            };
            let down_buf_id =
                rocm_buf(bindings.get(down_exps).expect("rocm: unbound down_exps")).ptr as usize;
            let is_paged = {
                let mp = ctx.moe_pager.lock().unwrap();
                mp.as_ref()
                    .is_some_and(|p| p.is_paged(crate::pager::Role::Gate, gate_buf_id))
            };
            if is_paged {
                if native.is_none() {
                    return Err(be(
                        "rocm moe pager: paged expert bank has a non-native quant format \
                         (only Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_1/IQ4_NL/IQ4_XS/ \
                          IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/TQ1_0/TQ2_0/Q2_0 page \
                          — the fp4 formats await native MoE decode)",
                    ));
                }
                // Open one touch batch per pool for this (layer) op, so every expert this op
                // pages is eviction-protected from the op's own later touches, and reset the
                // copy-stream overlap engine's per-op in-flight cursor (the router-logit readback
                // `hipStreamSynchronize` below drains prior copies before any slot is reused).
                let mut mp = ctx.moe_pager.lock().unwrap();
                let p = mp.as_mut().unwrap();
                p.begin_paged_op();
                p.begin_batch(gate_buf_id)?;
                if !fused_gate_up {
                    p.begin_batch(up_buf_id)?;
                }
                p.begin_batch(down_buf_id)?;
            }

            let (gw_ptr, uw_ptr, dw_ptr) = if is_paged {
                // Bank pointers are unused on the paged path — per-expert slot pointers are
                // resolved through the pager in the routing loop below.
                (
                    std::ptr::null_mut::<c_void>(),
                    std::ptr::null_mut::<c_void>(),
                    std::ptr::null_mut::<c_void>(),
                )
            } else if native.is_some() {
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
            let rows = g.desc(x).numel() / neu;

            // Which expert tier this op will take, resolved BEFORE the input is prepared because
            // both F1c folds are scoped to the R8 ordered-accumulate int8 tier (see `use_idm`'s
            // full rationale at its original site below — the exclusions are unchanged).
            let use_i8 = native.is_some() && moe_i8_enabled(ctx.rocm);
            let id_rows = MOE_ID_ROWS.clamped(ctx.rocm.moe_id_rows);
            let use_idm = use_i8 && !is_paged && id_rows > 0;

            // ── F1c `RmsNorm → MoeFfn`: the elided norm's row is produced HERE. ──
            // The shared pass only plans this for a single-row op whose `x` and `router_x` are both
            // the norm output, so ONE `rmsnorm_quant_i8_32` serves both consumers: its int8 codes go
            // to the experts (replacing the `quant_i8_32` the chunk loop would have run) and its
            // `xn` output is the normalized f32 row the router GEMV reads — byte for byte what the
            // standalone `rmsnorm` wrote. Net: one launch per MoE layer instead of two.
            //
            // A tier that cannot take it (paged, `INFR_ROCM_NO_I8`, `moe_id_rows = 0`) REPLAYS the
            // elided `rmsnorm` into scratch, so the fold never changes what those paths compute.
            let (x_ptr, rx_ptr, pre_quant) = match norm_fuse {
                Some((x_raw, norm_w, eps)) => {
                    let wnptr = ctx.dequant_weight_or_cache(norm_w, g, bindings)?;
                    let xrp = ctx.ensure_device(x_raw, g, bindings)?;
                    let xn = ctx.pool_buf((rows * neu * 4).max(1), false);
                    if use_idm && rows == 1 {
                        let qx = ctx.pool_buf((rows * neu).max(1), false);
                        let xs = ctx.pool_buf((rows * (neu / 32) * 4).max(1), false);
                        dispatch_grid(
                            pipelines,
                            ctx.stream,
                            "rmsnorm_quant_i8_32",
                            rows as u32,
                            1,
                            256,
                            args![
                                arg_ptr(xrp),
                                arg_ptr(wnptr),
                                arg_ptr(qx.ptr),
                                arg_ptr(xs.ptr),
                                arg_ptr(xn.ptr),
                                arg_i32(rows as i32),
                                arg_i32(ne as i32),
                                arg_f32(eps),
                            ],
                        )?;
                        (xn.ptr, xn.ptr, Some((qx, xs)))
                    } else {
                        dispatch_grid(
                            pipelines,
                            ctx.stream,
                            "rmsnorm",
                            rows as u32,
                            1,
                            256,
                            args![
                                arg_ptr(xrp),
                                arg_ptr(wnptr),
                                arg_ptr(xn.ptr),
                                arg_i32(rows as i32),
                                arg_i32(ne as i32),
                                arg_f32(eps),
                            ],
                        )?;
                        (xn.ptr, xn.ptr, None)
                    }
                }
                None => {
                    // `x` (and `router_x`, usually the same handle) carry `rows` token rows of `ne`.
                    let xp = ctx.ensure_device(x, g, bindings)?;
                    let rxp = if router_x != x {
                        ctx.ensure_device(router_x, g, bindings)?
                    } else {
                        xp
                    };
                    (xp, rxp, None)
                }
            };

            // Per-expert down-projection output scale (diffusion-gemma); 1.0 = none.
            let dsc_vals: Vec<f32> = match down_scale {
                Some(sid) => ctx.host_vals(sid, g, bindings)?.to_vec(),
                None => vec![1.0f32; nexp],
            };

            // Router logits = router · router_x, one dot per expert: reuse the linear_f16
            // GEMV to produce [rows, n_expert], then read them back for host-side gating.
            // F5 fully-overwritten: `linear_f16` runs one block per row and stores every
            // `[row, expert]` of the `[rows, n_expert]` logits — the only consumers (`moe_topk`,
            // and the paged path's host readback) read exactly that extent.
            let logits_dev = ctx.uninit_dev(rows * nexp);
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
            // ── R8: the id-indexed multi-slot expert GEMV tier (`moe_*_idm_*`). ──
            // Takes over the RESIDENT int8 expert path entirely — every (row, slot) pair in one
            // dispatch per stage instead of the `3 * rows * n_used` serialized launches below.
            // Scoped to `use_i8 && !is_paged`, and the exclusions are not arbitrary:
            //
            //  * PAGED stays on the per-expert loop. The pager routes on the HOST (it must know
            //    which experts to page in before it can page them), and Slice 36 deliberately
            //    INTERLEAVES each expert's H2D fill with the previous expert's GEMV on a separate
            //    copy stream — collapsing the loop into one dispatch would serialize every fill
            //    ahead of all compute and throw that overlap away for the sake of ~20 launches.
            //    A paged bank's slot index is also not its expert id, so an id-GEMV would need
            //    Vulkan's device LUT (`native_gemv_id*.comp`'s `-DPAGED` build, a per-layer window
            //    into a slot-index tape) which the host-routing design has no need of. See the R8
            //    entry in `docs/rocm-plan.md` for the measurement.
            //  * `INFR_ROCM_NO_I8` / the dequant→f16 fallback stay on the loop: both are A/B
            //    comparands whose whole job is to be the OTHER path, and neither ships.
            //  * `kernels.rocm.moe_id_rows = 0` turns the tier off outright — the third A/B
            //    comparand, and the one that isolates R8 itself (see [`MOE_ID_ROWS`]).
            //
            // (`use_i8` / `id_rows` / `use_idm` are resolved above, before the input is prepared —
            // the F1c norm fold needs to know which tier this op takes.)
            let (qx_x, xs_x, h_buf, hq, hs) = if use_i8 && !use_idm {
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

            // ── F1c `MoeFfn → Add`: the residual folds into the expert-accumulate epilogue. ──
            // ONLY on the R8 tier. The pre-R8 loop `atomicAdd`s each slot's contribution into `dst`
            // and is deterministic only because the host serializes the slots — seeding `dst` with
            // the residual there would re-associate the sum (`((h + s0) + s1)…` instead of
            // `h + ((s0 + s1)…)`) and move the golden. The R8 tier reduces the slots in
            // `moe_accum_idm`, which joins the residual ONCE, at the very end, exactly where the
            // elided `add` joined it — bit-identical (see the kernel's header).
            //
            // When the fold is on, `dd` ALIASES the residual stream's live buffer (the `Add`'s dst)
            // and the accumulate writes the final sum straight into it, so there is no zeroed MoE
            // scratch and no standalone `add`. Off (or on a tier that cannot take it), `dd` is the
            // fresh zeroed scratch it always was.
            let fold_resid = add_fuse.filter(|_| use_idm);
            let dd = match fold_resid {
                Some((_, add_dst)) => {
                    ctx.ensure_device(add_dst, g, bindings)?;
                    let b = ctx.dev[add_dst.0 as usize].as_ref().unwrap();
                    crate::RocmBuffer {
                        ptr: b.ptr,
                        len: b.len,
                        owned: false,
                        host_ptr: std::ptr::null_mut(),
                        uid: b.uid,
                    }
                }
                // F5 ACCUMULATOR — stays calloc, on every tier. The pre-R8 per-(row, slot) loop and
                // the paged loop `atomicAdd` each expert's contribution into `dst`, and even the R8
                // tier's `moe_accum_idm` does `dst[i] += acc` when the residual is not folded in
                // (`res == null`) precisely so the two paths sum in the same order. Its seed must
                // be +0.0.
                None => ctx.zero_dev(rows * neu),
            };
            let resid_ptr = match fold_resid {
                Some((resid, _)) => ctx.ensure_device(resid, g, bindings)?,
                None => std::ptr::null_mut(),
            };

            // ── RESIDENT path (Slice 38): GPU-side top-k routing, no host readback. ──
            // `moe_topk` reads the router logits (already on-device in `logits_dev`), computes the
            // top-`n_used` experts + gate weights per row into device buffers, and the `*_routed_*`
            // expert kernels resolve the per-expert bank pointer from `expert_id` in-kernel. The host
            // loop below issues a FIXED `rows * n_used` grid of dispatches with NO knowledge of which
            // experts were picked, so nothing is read back — removing the per-MoE-layer D2H stall.
            if !is_paged {
                let route_ids = ctx.pool_buf((rows * nu * 4).max(4), false);
                let route_wts = ctx.pool_buf((rows * nu * 4).max(4), false);
                let gating_flag: i32 = match gating {
                    infr_core::graph::MoeGating::Softmax => 0,
                    infr_core::graph::MoeGating::Sigmoid => 1,
                };
                let normw_flag: i32 = if norm_w { 1 } else { 0 };
                dispatch_1d(
                    pipelines,
                    ctx.stream,
                    "moe_topk",
                    (rows as u32) * 128,
                    128,
                    args![
                        arg_ptr(logits_dev.ptr),
                        arg_ptr(route_ids.ptr),
                        arg_ptr(route_wts.ptr),
                        arg_i32(nexp as i32),
                        arg_i32(nu as i32),
                        arg_f32(scale),
                        arg_i32(gating_flag),
                        arg_i32(normw_flag),
                    ],
                )?;

                // Per-expert down-projection scale on device (diffusion-gemma); null ⇒ all 1.0. The
                // synchronous copy is tiny (n_expert floats) and only runs for models that carry a
                // `down_scale` (the coherence-critical qwen3moe has none → null → no copy).
                let dsc_dev = if down_scale.is_some() {
                    let b = ctx.pool_buf((nexp * 4).max(4), false);
                    unsafe {
                        ffi::hipMemcpy(
                            b.ptr,
                            dsc_vals.as_ptr() as *const c_void,
                            nexp * 4,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        );
                    }
                    Some(b)
                } else {
                    None
                };
                let dsc_ptr = dsc_dev
                    .as_ref()
                    .map(|b| b.ptr)
                    .unwrap_or(std::ptr::null_mut());
                let fused_flag: i32 = if fused_gate_up { 1 } else { 0 };

                if use_idm {
                    // ── R8 id-indexed MULTI-SLOT tier: 5 dispatches per row-chunk, whatever
                    // `n_used` is, and all `rows * n_used` experts run CONCURRENTLY. ──
                    let ((gu, gu_qpb, gu_bpb), (dn, dn_qpb, dn_bpb)) =
                        native.expect("use_idm implies native");
                    // Per-expert BYTE strides — computed on the host in `usize` and passed as
                    // `i64`, so the in-kernel `base + (long)expert_id * stride` is a 64-bit
                    // multiply on a 64-bit pointer. Element-count strides scaled inside the kernel
                    // are what overflowed 32 bits on Vulkan; see MOE_ID_MULTI's header.
                    let gate_bstride = ((ge_stride / gu_qpb) * gu_bpb) as i64;
                    let up_bstride = ((nfu * neu / gu_qpb) * gu_bpb) as i64;
                    let up_half_boff = up_bstride;
                    let down_bstride = ((neu * nfu / dn_qpb) * dn_bpb) as i64;
                    // Rows are CHUNKED (see `MOE_ID_ROWS`) — the scratch below is sized for ONE
                    // chunk and reused across them, so a 1024-row prefill ubatch costs the same
                    // VRAM as a 128-row one. `rows == 1` (decode) is a single chunk of 1.
                    //
                    // The chunk is bounded by BYTES as well as by rows, because a row count is not
                    // a fixed footprint:
                    // `ne` and `n_ff_exp` vary ~4× across MoE architectures and `n_used` 1..8, so
                    // the same 128 rows is 13 MiB on qwen3moe and could be several times that
                    // elsewhere. MEASURED, not hypothetical: at `-p 1024` on Qwen3-30B-A3B an
                    // unchunked (or 512-row) chunk asks for ~50-100 MiB of pool on top of a 17 GiB
                    // weight set plus its KV, and `BufferPool`'s `hipMalloc` fails — which today
                    // aborts the process rather than degrading. The cap makes the knob a ceiling
                    // on rows AND on bytes, so no shape can walk into that through the default.
                    const MOE_ID_SCRATCH_CAP: usize = 16 << 20;
                    let per_row =
                        nu * (nfu * 4 + neu * 4 + nfu + (nfu / 32) * 4) + neu * 4 + (neu / 32) * 4;
                    let chunk = id_rows
                        .min(rows.max(1))
                        .min((MOE_ID_SCRATCH_CAP / per_row.max(1)).max(1))
                        .max(1);
                    let max_slots = chunk * nu;
                    // Every byte of each is written before it is read within the chunk (the quant
                    // passes fill `qx`/`xs` and `hq`/`hs` completely, the two GEMVs fill `h`/`y`
                    // completely), so `zero = false` — same argument the per-row scratch makes.
                    //
                    // F1c: under the `RmsNorm → MoeFfn` fold the activation is ALREADY int8 — the
                    // normalize pass above quantized it (that fold is single-row, so there is
                    // exactly one chunk and `pre_quant`'s buffers are chunk-sized by construction).
                    let (qxb, xsb, prequantized) = match pre_quant {
                        Some((qx, xs)) => {
                            debug_assert_eq!(rows, 1);
                            (qx, xs, true)
                        }
                        None => (
                            ctx.pool_buf((chunk * neu).max(1), false),
                            ctx.pool_buf((chunk * (neu / 32) * 4).max(1), false),
                            false,
                        ),
                    };
                    // ── S5: MMQ decode-once-reuse MoE GEMM (opt-in) ──
                    // Each expert's weight column tile decoded ONCE into LDS and reused
                    // across all routing rows, eliminating per-wave re-decode overhead.
                    let mmq_kernel = mmq_up_kernel(gu);
                    let use_mmq = ctx.rocm.mmq && mmq_kernel.is_some() && fused_gate_up;
                    let hb = ctx.pool_buf(
                        (max_slots * nfu * if use_mmq { 2 } else { 1 } * 4).max(1),
                        false,
                    );
                    // MMQ activation output buffer — silu(gate)*up applied, n_ff_exp columns.
                    let ab = if use_mmq {
                        Some(ctx.pool_buf((max_slots * nfu * 4).max(1), false))
                    } else {
                        None
                    };
                    let hqb = ctx.pool_buf((max_slots * nfu).max(1), false);
                    let hsb = ctx.pool_buf((max_slots * (nfu / 32) * 4).max(1), false);
                    let yb = ctx.pool_buf((max_slots * neu * 4).max(1), false);
                    // ── P2: the bucket-sorted BATCHED arm. ──
                    // R8 fixed the launch count and left the WEIGHT TRAFFIC alone: its
                    // `(output row, slot)` grid re-reads an expert's whole bank once per slot, so
                    // Qwen3-30B-A3B `pp512` moves 12.5 GB of expert weights per layer against
                    // 391 MB of distinct bytes — a 32× re-read (`rows·n_used / n_expert`), and
                    // `Op::MoeFfn` measured 97.1% of the forward because of it. Bucketing the
                    // slots by expert and giving each expert ONE block per output row collapses
                    // that to once per ROW-CHUNK. Same kernels' arithmetic, same per-slot
                    // destinations, so the outputs are bit-identical (see MOE_ID_BUCKET's header).
                    //
                    // `nexp` bounds are the sort's LDS histogram; the occupancy floor keeps the
                    // empty `(row, expert)` blocks from outweighing the saved traffic, and is what
                    // leaves DECODE (`nu` slots over `nexp` banks) on the id tier untouched.
                    let use_idb = ctx.rocm.moe_bucket
                        && nexp > 0
                        && nexp <= MOE_BUCKET_MAX_EXPERT
                        && chunk * nu >= nexp * MOE_BUCKET_MIN_OCC;
                    // MMQ always needs bucket-sorted data; force bucket alloc when MMQ.
                    let (bslot, eoff, ecnt) = if use_mmq || use_idb {
                        (
                            Some(ctx.pool_buf((max_slots * 4).max(4), false)),
                            Some(ctx.pool_buf(nexp * 4, false)),
                            Some(ctx.pool_buf(nexp * 4, false)),
                        )
                    } else {
                        (None, None, None)
                    };
                    let mut r0 = 0usize;
                    while r0 < rows {
                        let nr = chunk.min(rows - r0);
                        let n_slots = nr * nu;
                        // `route_ids`/`route_wts` are `[rows, n_used]` flat, so a row chunk is a
                        // contiguous window of slots — the kernels' `slot` is chunk-local and
                        // `row = slot / n_used` recovers the chunk-local token row.
                        let x_c = unsafe { (x_ptr as *mut u8).add(r0 * neu * 4) as *mut c_void };
                        let dst_c = unsafe { (dd.ptr as *mut u8).add(r0 * neu * 4) as *mut c_void };
                        let ids_c =
                            unsafe { (route_ids.ptr as *mut u8).add(r0 * nu * 4) as *mut c_void };
                        let wts_c =
                            unsafe { (route_wts.ptr as *mut u8).add(r0 * nu * 4) as *mut c_void };
                        // int8 quant of the chunk's WHOLE activation block in one dispatch — the
                        // per-row loop's `quant_i8_32` with its own `m` finally carrying more
                        // than 1. Skipped when the F1c norm fold already produced these codes.
                        if !prequantized {
                            dispatch_1d(
                                pipelines,
                                ctx.stream,
                                "quant_i8_32",
                                (nr * (neu / 32)) as u32,
                                256,
                                args![
                                    arg_ptr(x_c),
                                    arg_ptr(qxb.ptr),
                                    arg_ptr(xsb.ptr),
                                    arg_i32(nr as i32),
                                    arg_i32(ne as i32),
                                ],
                            )?;
                        }
                        // P2: the chunk's slots sorted into per-expert buckets, once, ahead of
                        // both GEMVs — the gate/up and down arms walk the SAME bucket list. The
                        // last chunk of a prefill can be short, so the occupancy floor is re-checked
                        // per chunk rather than assumed from `chunk`.
                        let batched = (use_mmq || use_idb) && n_slots >= nexp * MOE_BUCKET_MIN_OCC;
                        // S5: MMQ always needs bucket-sorted data — sort even when
                        // batched occupancy floor isn't met (decode with few slots).
                        if use_mmq || batched {
                            dispatch_1d(
                                pipelines,
                                ctx.stream,
                                "moe_bucket_sort",
                                256,
                                256,
                                args![
                                    arg_ptr(ids_c),
                                    arg_ptr(ecnt.as_ref().unwrap().ptr),
                                    arg_ptr(eoff.as_ref().unwrap().ptr),
                                    arg_ptr(bslot.as_ref().unwrap().ptr),
                                    arg_i32(n_slots as i32),
                                    arg_i32(nexp as i32),
                                ],
                            )?;
                        }
                        if use_mmq {
                            // S5: MMQ GEMM — one workgroup per (expert, column tile).
                            // Outputs raw gate+up dot products: cols [0, n_ff_exp) = gate,
                            // cols [n_ff_exp, 2*n_ff_exp) = up. Then `moe_act_mul_*` applies
                            // silu(gate)*up + route weight + scale → activated [n_slots, n_ff_exp].
                            let n_ff_exp2 = (n_ff_exp * 2) as u32; // combined gate+up output cols
                            let mmq_col_tiles = n_ff_exp2.div_ceil(64); // BN=64
                            dispatch_grid(
                                pipelines,
                                ctx.stream,
                                mmq_kernel.unwrap(),
                                mmq_col_tiles,
                                nexp as u32,
                                128,
                                args![
                                    arg_ptr(qxb.ptr),
                                    arg_ptr(xsb.ptr),
                                    arg_ptr(gw_ptr),
                                    arg_ptr(uw_ptr),
                                    arg_ptr(hb.ptr), // raw gate+up [n_slots, 2*n_ff_exp]
                                    arg_i32(ne as i32),
                                    arg_i32(n_ff_exp2 as i32),
                                    arg_i32(n_slots as i32),
                                    arg_i32(nexp as i32),
                                    arg_ptr(eoff.as_ref().unwrap().ptr),
                                    arg_ptr(ecnt.as_ref().unwrap().ptr),
                                    arg_ptr(bslot.as_ref().unwrap().ptr),
                                    arg_i32(nu as i32),
                                    arg_i64(gate_bstride),
                                    arg_i64(up_bstride),
                                    arg_i32(neu as i32),
                                ],
                            )?;
                            // Apply activation: silu(gate)*up + route weight + per-expert scale.
                            let act_kernel = mmq_act_kernel(gu);
                            let act_flag: i32 = match act {
                                Activation::Silu => 0,
                                Activation::Gelu => 1,
                                Activation::Sigmoid => 2,
                            };
                            dispatch_1d(
                                pipelines,
                                ctx.stream,
                                act_kernel,
                                nexp as u32 * 256,
                                256,
                                args![
                                    arg_ptr(hb.ptr),                   // src: raw gate+up [n_slots, 2*n_ff_exp]
                                    arg_ptr(ab.as_ref().unwrap().ptr), // dst: activated [n_slots, n_ff_exp]
                                    arg_i32(n_ff_exp as i32),          // n_ff (per-gate dim)
                                    arg_i32(n_ff_exp2 as i32),         // n_ff_exp = 2*n_ff
                                    arg_i32(n_slots as i32),
                                    arg_i32(nexp as i32),
                                    arg_ptr(eoff.as_ref().unwrap().ptr),
                                    arg_ptr(ecnt.as_ref().unwrap().ptr),
                                    arg_ptr(bslot.as_ref().unwrap().ptr),
                                    arg_ptr(wts_c),   // route_wts
                                    arg_ptr(dsc_ptr), // per-expert scale (null = 1.0)
                                    arg_i32(act_flag),
                                    arg_i32(wb_flag),
                                ],
                            )?;
                        } else if batched {
                            dispatch_grid(
                                pipelines,
                                ctx.stream,
                                moe_gate_up_i8_idb_kernel(gu),
                                n_ff_exp.div_ceil(MOE_IDB_WAVES),
                                nexp as u32,
                                32 * MOE_IDB_WAVES,
                                args![
                                    arg_ptr(qxb.ptr),
                                    arg_ptr(xsb.ptr),
                                    arg_ptr(gw_ptr),
                                    arg_ptr(uw_ptr),
                                    arg_ptr(hb.ptr),
                                    arg_i32(ne as i32),
                                    arg_i32(n_ff_exp as i32),
                                    arg_i32(at),
                                    arg_i32(wb_flag),
                                    arg_ptr(dsc_ptr),
                                    arg_ptr(wts_c),
                                    arg_i32(nu as i32),
                                    arg_i64(gate_bstride),
                                    arg_i64(up_bstride),
                                    arg_i32(fused_flag),
                                    arg_i64(up_half_boff),
                                    arg_ptr(bslot.as_ref().unwrap().ptr),
                                    arg_ptr(eoff.as_ref().unwrap().ptr),
                                    arg_ptr(ecnt.as_ref().unwrap().ptr),
                                ],
                            )?;
                        } else {
                            // P7f: Q4_K CN=2 column tiling — two output columns per wave
                            // instead of one, halving the wave count on the idm tier.
                            let (gu_kernel, gu_grid_x) = if gu == "q4k" {
                                ("moe_gate_up_act_i8_idm_q4k_cn2", n_ff_exp.div_ceil(2))
                            } else {
                                (moe_gate_up_i8_idm_kernel(gu), n_ff_exp)
                            };
                            dispatch_grid(
                                pipelines,
                                ctx.stream,
                                gu_kernel,
                                gu_grid_x,
                                n_slots as u32,
                                32,
                                args![
                                    arg_ptr(qxb.ptr),
                                    arg_ptr(xsb.ptr),
                                    arg_ptr(gw_ptr),
                                    arg_ptr(uw_ptr),
                                    arg_ptr(hb.ptr),
                                    arg_i32(ne as i32),
                                    arg_i32(n_ff_exp as i32),
                                    arg_i32(at),
                                    arg_i32(wb_flag),
                                    arg_ptr(dsc_ptr),
                                    arg_ptr(ids_c),
                                    arg_ptr(wts_c),
                                    arg_i32(n_slots as i32),
                                    arg_i32(nu as i32),
                                    arg_i64(gate_bstride),
                                    arg_i64(up_bstride),
                                    arg_i32(fused_flag),
                                    arg_i64(up_half_boff),
                                ],
                            )?;
                        }
                        let src_act = if use_mmq {
                            ab.as_ref().unwrap().ptr
                        } else {
                            hb.ptr
                        };
                        dispatch_1d(
                            pipelines,
                            ctx.stream,
                            "quant_i8_32",
                            (n_slots * (nfu / 32)) as u32,
                            256,
                            args![
                                arg_ptr(src_act),
                                arg_ptr(hqb.ptr),
                                arg_ptr(hsb.ptr),
                                arg_i32(n_slots as i32),
                                arg_i32(n_ff_exp as i32),
                            ],
                        )?;
                        if batched {
                            dispatch_grid(
                                pipelines,
                                ctx.stream,
                                moe_down_i8_idb_kernel(dn),
                                ne.div_ceil(MOE_IDB_WAVES),
                                nexp as u32,
                                32 * MOE_IDB_WAVES,
                                args![
                                    arg_ptr(hqb.ptr),
                                    arg_ptr(hsb.ptr),
                                    arg_ptr(dw_ptr),
                                    arg_ptr(yb.ptr),
                                    arg_i32(ne as i32),
                                    arg_i32(n_ff_exp as i32),
                                    arg_i64(down_bstride),
                                    arg_ptr(bslot.as_ref().unwrap().ptr),
                                    arg_ptr(eoff.as_ref().unwrap().ptr),
                                    arg_ptr(ecnt.as_ref().unwrap().ptr),
                                ],
                            )?;
                        } else {
                            dispatch_grid(
                                pipelines,
                                ctx.stream,
                                moe_down_i8_idm_kernel(dn),
                                ne,
                                n_slots as u32,
                                32,
                                args![
                                    arg_ptr(hqb.ptr),
                                    arg_ptr(hsb.ptr),
                                    arg_ptr(dw_ptr),
                                    arg_ptr(yb.ptr),
                                    arg_i32(ne as i32),
                                    arg_i32(n_ff_exp as i32),
                                    arg_ptr(ids_c),
                                    arg_i32(n_slots as i32),
                                    arg_i64(down_bstride),
                                ],
                            )?;
                        }
                        // Ordered slot reduction — NOT atomics; see MOE_ID_MULTI's header for why
                        // the golden hash depends on it. `res_c` (null unless the F1c residual fold
                        // is on) joins the residual in the epilogue, leaving that order untouched.
                        let res_c = if resid_ptr.is_null() {
                            std::ptr::null_mut()
                        } else {
                            unsafe { (resid_ptr as *mut u8).add(r0 * neu * 4) as *mut c_void }
                        };
                        dispatch_1d(
                            pipelines,
                            ctx.stream,
                            "moe_accum_idm",
                            (nr * neu) as u32,
                            256,
                            args![
                                arg_ptr(yb.ptr),
                                arg_ptr(dst_c),
                                arg_ptr(res_c),
                                arg_i32(ne as i32),
                                arg_i32(nr as i32),
                                arg_i32(nu as i32),
                            ],
                        )?;
                        r0 += nr;
                    }
                } else {
                    // ── Pre-R8 per-(row, slot) tier. Still the only path for the two A/B comparands
                    // (`INFR_ROCM_NO_I8`, and a bank pair with no native decode at all). ──
                    for row in 0..rows {
                        let x_row = unsafe { (x_ptr as *mut u8).add(row * neu * 4) as *mut c_void };
                        let dst_row =
                            unsafe { (dd.ptr as *mut u8).add(row * neu * 4) as *mut c_void };
                        if use_i8 {
                            // int8 activation quant of this token's input row (reused across all n_used).
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
                        for k in 0..nu {
                            let slot = (row * nu + k) as i32;
                            if let (true, Some(((gu, gu_qpb, gu_bpb), (dn, dn_qpb, dn_bpb)))) =
                                (use_i8, native)
                            {
                                let gate_bstride = ((ge_stride / gu_qpb) * gu_bpb) as i64;
                                let up_bstride = ((nfu * neu / gu_qpb) * gu_bpb) as i64;
                                let up_half_boff = ((nfu * neu / gu_qpb) * gu_bpb) as i64;
                                let down_bstride = ((neu * nfu / dn_qpb) * dn_bpb) as i64;
                                let h_ptr = h_buf.as_ref().unwrap().ptr;
                                dispatch_grid(
                                    pipelines,
                                    ctx.stream,
                                    moe_gate_up_i8_routed_kernel(gu),
                                    n_ff_exp,
                                    1,
                                    32,
                                    args![
                                        arg_ptr(qx_x.as_ref().unwrap().ptr),
                                        arg_ptr(xs_x.as_ref().unwrap().ptr),
                                        arg_ptr(gw_ptr),
                                        arg_ptr(uw_ptr),
                                        arg_ptr(h_ptr),
                                        arg_i32(ne as i32),
                                        arg_i32(n_ff_exp as i32),
                                        arg_i32(at),
                                        arg_i32(wb_flag),
                                        arg_ptr(dsc_ptr),
                                        arg_ptr(route_ids.ptr),
                                        arg_ptr(route_wts.ptr),
                                        arg_i32(slot),
                                        arg_i64(gate_bstride),
                                        arg_i64(up_bstride),
                                        arg_i32(fused_flag),
                                        arg_i64(up_half_boff),
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
                                    moe_down_i8_routed_kernel(dn),
                                    ne,
                                    1,
                                    32,
                                    args![
                                        arg_ptr(hq.as_ref().unwrap().ptr),
                                        arg_ptr(hs.as_ref().unwrap().ptr),
                                        arg_ptr(dw_ptr),
                                        arg_ptr(dst_row),
                                        arg_i32(ne as i32),
                                        arg_i32(n_ff_exp as i32),
                                        arg_ptr(route_ids.ptr),
                                        arg_i32(slot),
                                        arg_i64(down_bstride),
                                    ],
                                )?;
                            } else if let Some(((gu, gu_qpb, gu_bpb), (dn, dn_qpb, dn_bpb))) =
                                native
                            {
                                let gate_bstride = ((ge_stride / gu_qpb) * gu_bpb) as i64;
                                let up_bstride = ((nfu * neu / gu_qpb) * gu_bpb) as i64;
                                let up_half_boff = ((nfu * neu / gu_qpb) * gu_bpb) as i64;
                                let down_bstride = ((neu * nfu / dn_qpb) * dn_bpb) as i64;
                                dispatch_1d(
                                    pipelines,
                                    ctx.stream,
                                    moe_expert_routed_kernel(gu, dn)
                                        .expect("non-int8 native expert without a kernel"),
                                    n_ff_exp,
                                    256,
                                    args![
                                        arg_ptr(x_row),
                                        arg_ptr(gw_ptr),
                                        arg_ptr(uw_ptr),
                                        arg_ptr(dw_ptr),
                                        arg_ptr(dst_row),
                                        arg_i32(ne as i32),
                                        arg_i32(n_ff_exp as i32),
                                        arg_i32(at),
                                        arg_i32(wb_flag),
                                        arg_ptr(dsc_ptr),
                                        arg_ptr(route_ids.ptr),
                                        arg_ptr(route_wts.ptr),
                                        arg_i32(slot),
                                        arg_i64(gate_bstride),
                                        arg_i64(up_bstride),
                                        arg_i64(down_bstride),
                                        arg_i32(fused_flag),
                                        arg_i64(up_half_boff),
                                    ],
                                )?;
                            } else {
                                // f16 dequant-cache fallback: element strides into the __half banks.
                                let gate_estride = ge_stride as i64;
                                let up_estride = (nfu * neu) as i64;
                                let down_estride = (neu * nfu) as i64;
                                let up_half_eoff = (nfu * neu) as i64;
                                dispatch_1d(
                                    pipelines,
                                    ctx.stream,
                                    "moe_ffn_expert_routed",
                                    n_ff_exp,
                                    256,
                                    args![
                                        arg_ptr(x_row),
                                        arg_ptr(gw_ptr),
                                        arg_ptr(uw_ptr),
                                        arg_ptr(dw_ptr),
                                        arg_ptr(dst_row),
                                        arg_i32(ne as i32),
                                        arg_i32(n_ff_exp as i32),
                                        arg_i32(at),
                                        arg_i32(wb_flag),
                                        arg_ptr(dsc_ptr),
                                        arg_ptr(route_ids.ptr),
                                        arg_ptr(route_wts.ptr),
                                        arg_i32(slot),
                                        arg_i64(gate_estride),
                                        arg_i64(up_estride),
                                        arg_i64(down_estride),
                                        arg_i32(fused_flag),
                                        arg_i64(up_half_eoff),
                                    ],
                                )?;
                            }
                        }
                    }
                }
            }

            // ── PAGED path: host-side routing (the pager must know WHICH experts to page in, so the
            //    router logits are read back to the host). Unchanged from the pre-Slice-38 flow. ──
            if is_paged {
                unsafe {
                    ffi::hipStreamSynchronize(ctx.stream);
                }
                let logits_all: Vec<f32> = {
                    let raw = read_bytes(&logits_dev, ctx.stream);
                    bytemuck::cast_slice::<u8, f32>(&raw).to_vec()
                };
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
                        let (gs, us, ds, kname) = if is_paged {
                            // Page each routed expert into its VRAM slot; the slot holds exactly this
                            // expert's raw quant bytes at offset 0, so there is no per-expert bank
                            // offset — the slot base IS the expert pointer. A fused gate_up slot is
                            // double-width: gate at 0, up at the within-slot half offset.
                            let ((gu, gu_qpb, gu_bpb), (dn, _dqpb, _dbpb)) =
                                native.expect("is_paged implies native");
                            let mut mp = ctx.moe_pager.lock().unwrap();
                            let p = mp.as_mut().unwrap();
                            let gs =
                                p.ensure_slot(crate::pager::Role::Gate, gate_buf_id, ei as u32)?;
                            let us = if fused_gate_up {
                                unsafe {
                                    (gs as *mut u8).add((nfu * neu / gu_qpb) * gu_bpb)
                                        as *mut c_void
                                }
                            } else {
                                p.ensure_slot(crate::pager::Role::Up, up_buf_id, ei as u32)?
                            };
                            let ds =
                                p.ensure_slot(crate::pager::Role::Down, down_buf_id, ei as u32)?;
                            (gs, us, ds, moe_expert_kernel(gu, dn))
                        } else if let Some(((gu, gu_qpb, gu_bpb), (dn, dn_qpb, dn_bpb))) = native {
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
                            (gs, us, ds, Some("moe_ffn_expert"))
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
                                // `native` is filtered so an int8-off expert always has an
                                // instantiated cross-product kernel (see `MOE_EXPERT_PAIRS`).
                                kname.expect("non-int8 native expert without a kernel"),
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
            }
            match (add_fuse, fold_resid) {
                // Folded: `dd` IS the `Add`'s dst buffer (already mapped in `ctx.dev` by the
                // `ensure_device` above) and the accumulate wrote the final sum in place. The MoE
                // op's own `dst` handle stays unpublished — the shared pass's live-range bound
                // guarantees nothing reads it before it is next rewritten, exactly as for the
                // dense `Linear → Add` fold.
                (Some(_), Some(_)) => {}
                // Planned but declined: this op took a tier whose accumulate has no epilogue to
                // fold into (paged / `INFR_ROCM_NO_I8` / `moe_id_rows = 0`). The `Add` op was
                // elided from the walk, so REPLAY it — the same `add` kernel over the same
                // operands, giving those paths the pre-fusion numbers exactly.
                (Some((resid, add_dst)), None) => {
                    let rp = ctx.ensure_device(resid, g, bindings)?;
                    let ap = ctx.ensure_device(add_dst, g, bindings)?;
                    let n = rows * neu;
                    dispatch_1d(
                        pipelines,
                        ctx.stream,
                        "add",
                        n as u32,
                        256,
                        args![arg_ptr(rp), arg_ptr(dd.ptr), arg_ptr(ap), arg_i32(n as i32),],
                    )?;
                    ctx.dev[dst.0 as usize] = Some(dd);
                }
                (None, _) => ctx.dev[dst.0 as usize] = Some(dd),
            }
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
            // F5 fully-overwritten: one thread per row writes `dst[row*channels + c]` for every
            // channel; the grid covers every row.
            let dd = ctx.uninit_dev(rows as usize * channels as usize);
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
            // F5 fully-overwritten: all three arms below store the output of every (row, value
            // head, value dim) — `deltanet`'s per-head `dr[d]` over the whole `d` loop,
            // `deltanet_decode`'s thread-per-`d` (grid-stride past `blockDim`), and
            // `deltanet_chunked`'s `dst[t*n_vhead*vd + vh*vd + d]` for every token of every chunk.
            let dd = ctx.uninit_dev(rows as usize * n_vhead as usize * head_v as usize);
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
            // F5 fully-overwritten: one thread per row writes `dr[0..n)`.
            let dd = ctx.uninit_dev(rows as usize * n as usize);
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

#[cfg(test)]
mod decode_spec_tests {
    use super::{moe_native_fmt, native_decode_fmt, native_i8_fmt};
    use infr_core::config::RocmCfg;
    use infr_core::DType;

    /// The three kernel tables now derive their block geometry from
    /// `infr_core::decode_spec::block_layout` instead of carrying inline `(256, 144)` literals.
    /// Pin them against the numbers that were spelled out here before the rewire, so the hoist
    /// stays behavior-preserving and a wrong spec entry cannot silently reshape a HIP dispatch
    /// (the block stride is what the kernels' byte addressing is built on).
    #[test]
    fn native_tables_reproduce_the_inline_block_geometry() {
        // S6: the int8 table is selected off a `RocmCfg` VALUE, not the environment — `default()`
        // is `INFR_ROCM_NO_I8` unset (`i8: true`), so the int8 arm below is always exercised now.
        let rocm = RocmCfg::default();
        // (dtype, elems/block, bytes/block) — verbatim from the pre-hoist tables.
        for (dt, elems, bytes) in [
            (DType::Q8_0, 32usize, 34usize),
            // R2: Q2_K (256 elems / 84 bytes = 16 scale bytes + 64 qs + 2×f16) and Q3_K (256 elems
            // / 110 bytes = 32 hmask + 64 qs + 12 packed 6-bit scales + f16) joined the natively
            // decoded set; the kernels' byte addressing is built on these strides.
            (DType::Q2K, 256, 84),
            (DType::Q3K, 256, 110),
            (DType::Q4K, 256, 144),
            // R1: Q5_K joined the natively-decoded set (256 elems / 176 bytes = 2×f16 + 12 scale
            // bytes + 32 qh + 128 qs); the kernels' byte addressing is built on this stride.
            (DType::Q5K, 256, 176),
            (DType::Q6K, 256, 210),
            // The legacy 32-element round quants: Q5_0 (22 B = f16 + 4 qh + 16 qs), and R3's
            // Q4_0 (18 B = f16 + 16 qs), Q4_1 (20 B = 2×f16 + 16 qs), Q5_1 (24 B = 2×f16 + 4 qh
            // + 16 qs). All four decode one 32-element block per header — no super-block.
            (DType::Q5_0, 32, 22),
            (DType::Q4_0, 32, 18),
            (DType::Q4_1, 32, 20),
            (DType::Q5_1, 32, 24),
            // R4's codebook quants: IQ4_NL is Q4_0's block shape (18 B = f16 + 16 packed nibbles)
            // and IQ4_XS is a 256-element super-block (136 B = f16 + u16 scales_h + 4 scales_l +
            // 128 qs) whose 8 sub-blocks of 32 each carry their own 6-bit scale.
            (DType::Iq4Nl, 32, 18),
            (DType::Iq4Xs, 256, 136),
            // R5's grid quants — all 256-element super-blocks, differing only in how much sign /
            // high-index / scale side-data rides along with the 8-, 9- or 10-bit grid codes.
            (DType::Iq2Xxs, 256, 66),
            (DType::Iq2Xs, 256, 74),
            (DType::Iq2S, 256, 82),
            (DType::Iq3Xxs, 256, 98),
            (DType::Iq3S, 256, 110),
            // R6's IQ1 quants — 256-element super-blocks sharing the 2048-entry IQ1 grid. IQ1_S
            // (50 B = f16 + 32 qs + 8×u16 qh) carries a standalone `d`; IQ1_M (56 B = 32 qs + 16 qh
            // + 4×u16 scales) has NONE — its `d` is split across the scale words' top nibbles.
            (DType::Iq1S, 256, 50),
            (DType::Iq1M, 256, 56),
            // R6's ternary quants. TQ1_0 (54 B = 48 qs + 4 qh + f16) packs 5 base-3 digits per
            // byte; TQ2_0 (66 B = 64 qs + f16) packs 4 elements per byte at 2 bits. Q2_0 is infr's
            // OWN format and the only 64-ELEMENT block in the covered set (18 B = f16 + 16 qs), so
            // one activation 32-block is HALF a Q2_0 block — the geometry every byte offset in
            // `wdec_q20` is built on.
            (DType::Tq1_0, 256, 54),
            (DType::Tq2_0, 256, 66),
            (DType::Q2_0, 64, 18),
            // R7's fp4 microscaling quants. MXFP4 (17 B = ONE E8M0 exponent byte + 16 packed
            // nibbles) is IQ4_NL's block with a 1-byte scale instead of an f16 — the smallest
            // header in the covered set. NVFP4 is a 64-ELEMENT block (36 B = 4 UE4M3 scale bytes +
            // 32 packed nibbles), so like Q2_0 one weight block spans TWO activation 32-blocks —
            // but with FOUR scales, one per 16 elements, which is the geometry `wdec_nvfp4`'s
            // `blk>>1` block index, `blk&1` half and per-half `s0`/`s1` pair are built on.
            (DType::Mxfp4, 32, 17),
            (DType::Nvfp4, 64, 36),
        ] {
            let (e, b, _, _) = native_decode_fmt(dt).expect("covered by native decode");
            assert_eq!((e, b), (elems, bytes), "{dt:?} native_decode_fmt geometry");
            let (b8, _) = native_i8_fmt(dt, &rocm).expect("covered by the int8 decode");
            assert_eq!(b8, bytes, "{dt:?} native_i8_fmt bytes/block");
        }
        for (dt, elems, bytes) in [
            (DType::Q8_0, 32usize, 34usize),
            (DType::Q2K, 256, 84),
            (DType::Q3K, 256, 110),
            (DType::Q4K, 256, 144),
            (DType::Q5K, 256, 176),
            (DType::Q6K, 256, 210),
            (DType::Q4_0, 32, 18),
            (DType::Q4_1, 32, 20),
            (DType::Q5_1, 32, 24),
            (DType::Iq4Nl, 32, 18),
            (DType::Iq4Xs, 256, 136),
            (DType::Iq2Xxs, 256, 66),
            (DType::Iq2Xs, 256, 74),
            (DType::Iq2S, 256, 82),
            (DType::Iq3Xxs, 256, 98),
            (DType::Iq3S, 256, 110),
            (DType::Iq1S, 256, 50),
            (DType::Iq1M, 256, 56),
            (DType::Tq1_0, 256, 54),
            (DType::Tq2_0, 256, 66),
            (DType::Q2_0, 64, 18),
            (DType::Mxfp4, 32, 17),
            (DType::Nvfp4, 64, 36),
        ] {
            let (_, e, b) = moe_native_fmt(dt).expect("covered by MoE native decode");
            assert_eq!((e, b), (elems, bytes), "{dt:?} moe_native_fmt geometry");
        }
        // Q5_0 is native on the DENSE paths but has no MoE expert kernel — no shipped GGUF packs
        // expert banks as Q5_0, so it stays off the (gate/up × down) cross product.
        assert!(moe_native_fmt(DType::Q5_0).is_none());
    }

    /// **R7: there is no coverage boundary left.** R1-R6 each maintained an assertion naming the
    /// weight quants still on the host dequant→f16 fallback; MXFP4 and NVFP4 were the last two, so
    /// that assertion is replaced by its complement — a TOTALITY check over every `DType` a GGUF
    /// can carry.
    ///
    /// The `match` is deliberately EXHAUSTIVE (no `_` arm): a new `DType` variant does not compile
    /// until someone decides here whether it is natively decoded or is one of the enumerated
    /// intentional exclusions, which is the property the old boundary assertion could not give
    /// (it only listed what was known-missing at the time). Every quant arm additionally requires
    /// the whole fast-path family — the Phase-3 GEMV/EmbedGather, the int8 dp4a GEMV, the WMMA
    /// prefill GEMM and the `deqf16_*` rocBLAS feeder — not just `native_decode_fmt`, because a
    /// format registered in one table and missing from another is exactly the drift that used to
    /// show up only on the box.
    #[test]
    fn native_decode_is_total_over_every_gguf_weight_dtype() {
        use super::{deqf16_fmt, native_wmma_fmt};
        let rocm = RocmCfg::default();
        // Why each excluded dtype is NOT a native-decode gap. Kept as a per-variant string so the
        // failure message names the reason rather than just the variant.
        const DENSE_FLOAT: &str =
            "dense float weight — nothing to decode (F16 rides `linear_f16`; \
             F32/BF16 are cast to f16 once at first use, a format change, not a quant decode)";
        const NOT_A_WEIGHT: &str = "never a weight tensor (bias / position payload)";
        const HOST_CONVERTED: &str =
            "host-converted to f16 at weight load — no backend ever sees this dtype";
        const KV_ONLY: &str = "KV-cache-only TurboQuant format — never a GGUF weight";
        for dt in [
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::I32,
            DType::U32,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Q8_0,
            DType::Q2K,
            DType::Q3K,
            DType::Q4K,
            DType::Q5K,
            DType::Q6K,
            DType::Iq1S,
            DType::Iq1M,
            DType::Iq2Xxs,
            DType::Iq2Xs,
            DType::Iq2S,
            DType::Iq3Xxs,
            DType::Iq3S,
            DType::Iq4Nl,
            DType::Iq4Xs,
            DType::Tq1_0,
            DType::Tq2_0,
            DType::I2S,
            DType::Q2_0,
            DType::Mxfp4,
            DType::Nvfp4,
            DType::Turbo2,
            DType::Turbo3,
            DType::Turbo4,
        ] {
            // EXHAUSTIVE — adding a `DType` variant breaks this match, which is the point.
            let excused: Option<&str> = match dt {
                DType::F32 | DType::F16 | DType::Bf16 => Some(DENSE_FLOAT),
                DType::I32 | DType::U32 => Some(NOT_A_WEIGHT),
                DType::I2S => Some(HOST_CONVERTED),
                DType::Turbo2 | DType::Turbo3 | DType::Turbo4 => Some(KV_ONLY),
                DType::Q4_0
                | DType::Q4_1
                | DType::Q5_0
                | DType::Q5_1
                | DType::Q8_0
                | DType::Q2K
                | DType::Q3K
                | DType::Q4K
                | DType::Q5K
                | DType::Q6K
                | DType::Iq1S
                | DType::Iq1M
                | DType::Iq2Xxs
                | DType::Iq2Xs
                | DType::Iq2S
                | DType::Iq3Xxs
                | DType::Iq3S
                | DType::Iq4Nl
                | DType::Iq4Xs
                | DType::Tq1_0
                | DType::Tq2_0
                | DType::Q2_0
                | DType::Mxfp4
                | DType::Nvfp4 => None,
            };
            match excused {
                Some(why) => assert!(
                    native_decode_fmt(dt).is_none(),
                    "{dt:?} is listed as intentionally unsupported ({why}) but native_decode_fmt \
                     claims it — update the exclusion list or the table"
                ),
                None => {
                    assert!(
                        native_decode_fmt(dt).is_some(),
                        "{dt:?} has no native decode kernel and is not excused"
                    );
                    assert!(native_i8_fmt(dt, &rocm).is_some(), "{dt:?} int8 dp4a GEMV");
                    assert!(deqf16_fmt(dt).is_some(), "{dt:?} deqf16_* rocBLAS feeder");
                    assert!(
                        native_wmma_fmt(dt, 4096, &rocm).is_some(),
                        "{dt:?} WMMA prefill GEMM"
                    );
                }
            }
        }
        // …and the not-excused set is EXACTLY the shared weight-quant roster, so this test cannot
        // pass by having quietly excused a real quant.
        for &dt in infr_core::decode_spec::WEIGHT_QUANTS {
            assert!(
                native_decode_fmt(dt).is_some(),
                "{dt:?} is in decode_spec::WEIGHT_QUANTS but has no native decode kernel"
            );
        }
        assert_eq!(
            infr_core::decode_spec::WEIGHT_QUANTS.len(),
            24,
            "the roster this slice claims to have closed"
        );
    }

    /// The `(gate/up, down)` pairs `kernels.rs` instantiates the cross-product expert kernels for —
    /// the expected set both mappers are pinned against. Rationale lives on `moe_expert_kernel`.
    const MOE_EXPERT_PAIRS: &[(&str, &str)] = &{
        // The K-quant square, then five rectangles and a diagonal: the R3 legacy round quants, the
        // R4 codebook quants, the `convert_incompatible_tensor` K-quant→IQ4_NL down bump, the R5
        // grid quants (gate/up only — see `moe_expert_kernel` for why nothing yields a grid `dn`),
        // the R6 IQ1 quants (which ARE also a legal `dn`), and the R6 ternary + R7 fp4 SELF pairs.
        const K: [&str; 6] = ["q80", "q2k", "q3k", "q4k", "q5k", "q6k"];
        const L: [&str; 3] = ["q40", "q41", "q51"];
        const LD: [&str; 4] = ["q40", "q41", "q51", "q80"];
        const I: [&str; 2] = ["iq4nl", "iq4xs"];
        const ID: [&str; 6] = ["iq4nl", "iq4xs", "q4k", "q5k", "q6k", "q80"];
        const XK: [&str; 2] = ["q2k", "q3k"];
        const G: [&str; 5] = ["iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s"];
        const GD: [&str; 7] = ["iq2s", "iq3xxs", "iq3s", "iq4nl", "iq4xs", "q4k", "q6k"];
        const O: [&str; 2] = ["iq1s", "iq1m"];
        const OD: [&str; 8] = [
            "iq1s", "iq1m", "iq2xxs", "iq2s", "iq3s", "iq4xs", "q4k", "q6k",
        ];
        // The SELF-pair diagonals: R6's ternary trio and R7's fp4 pair. Both families are
        // whole-model / whole-MoE conversion targets with no `ffn_down` bump, so the diagonal IS
        // their reachable set.
        const T: [&str; 5] = ["tq10", "tq20", "q20", "mxfp4", "nvfp4"];
        let mut out = [("", "");
            K.len() * K.len()
                + L.len() * LD.len()
                + I.len() * ID.len()
                + XK.len()
                + G.len() * GD.len()
                + O.len() * OD.len()
                + T.len()];
        let (mut n, mut i) = (0, 0);
        while i < K.len() {
            let mut j = 0;
            while j < K.len() {
                out[n] = (K[i], K[j]);
                n += 1;
                j += 1;
            }
            i += 1;
        }
        let mut i = 0;
        while i < L.len() {
            let mut j = 0;
            while j < LD.len() {
                out[n] = (L[i], LD[j]);
                n += 1;
                j += 1;
            }
            i += 1;
        }
        let mut i = 0;
        while i < I.len() {
            let mut j = 0;
            while j < ID.len() {
                out[n] = (I[i], ID[j]);
                n += 1;
                j += 1;
            }
            i += 1;
        }
        let mut i = 0;
        while i < XK.len() {
            out[n] = (XK[i], "iq4nl");
            n += 1;
            i += 1;
        }
        let mut i = 0;
        while i < G.len() {
            let mut j = 0;
            while j < GD.len() {
                out[n] = (G[i], GD[j]);
                n += 1;
                j += 1;
            }
            i += 1;
        }
        let mut i = 0;
        while i < O.len() {
            let mut j = 0;
            while j < OD.len() {
                out[n] = (O[i], OD[j]);
                n += 1;
                j += 1;
            }
            i += 1;
        }
        let mut i = 0;
        while i < T.len() {
            out[n] = (T[i], T[i]);
            n += 1;
            i += 1;
        }
        out
    };

    /// R3 escape hatch (extended by R4 and R5): the Phase-3 cross-product expert kernels are instantiated for
    /// [`MOE_EXPERT_PAIRS`] only, and BOTH mappers must cover exactly that set — the `MoeFfn` arm
    /// checks availability once (via `moe_expert_kernel`) and then dispatches EITHER the host-routed
    /// or the device-routed kernel, so a table that drifted would `expect`-panic at run time on a
    /// real GGUF. Also pins that the mappers never claim a pair `kernels.rs` does not instantiate.
    #[test]
    fn moe_expert_pair_tables_agree() {
        use super::{moe_expert_kernel, moe_expert_routed_kernel};
        const ALL: [&str; 24] = [
            "q80", "q2k", "q3k", "q4k", "q5k", "q6k", "q40", "q41", "q51", "iq4nl", "iq4xs",
            "iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq1s", "iq1m", "tq10", "tq20", "q20",
            "mxfp4", "nvfp4", "q50",
        ];
        assert_eq!(
            MOE_EXPERT_PAIRS.len(),
            118,
            "6×6 + 3×4 + 2×6 + 2 + 5×7 + 2×8 + 5 instantiated pairs"
        );
        for gu in ALL {
            for dn in ALL {
                let want = MOE_EXPERT_PAIRS.contains(&(gu, dn));
                let host = moe_expert_kernel(gu, dn);
                let routed = moe_expert_routed_kernel(gu, dn);
                assert_eq!(host.is_some(), want, "moe_expert_kernel({gu}, {dn})");
                assert_eq!(
                    routed.is_some(),
                    want,
                    "moe_expert_routed_kernel({gu}, {dn})"
                );
                if want {
                    assert_eq!(host.unwrap(), format!("moe_ffn_expert_{gu}_{dn}"));
                    assert_eq!(routed.unwrap(), format!("moe_ffn_expert_routed_{gu}_{dn}"));
                }
            }
        }
        // Every `moe_native_fmt` format has BOTH per-format int8 expert kernels — that is what makes
        // the shipping (int8) MoE path total over the covered set even though the cross product is not.
        let src = crate::kernels::hip_source();
        for dt in [
            DType::Q8_0,
            DType::Q2K,
            DType::Q3K,
            DType::Q4K,
            DType::Q5K,
            DType::Q6K,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_1,
            DType::Iq4Nl,
            DType::Iq4Xs,
            DType::Iq2Xxs,
            DType::Iq2Xs,
            DType::Iq2S,
            DType::Iq3Xxs,
            DType::Iq3S,
            DType::Iq1S,
            DType::Iq1M,
            DType::Tq1_0,
            DType::Tq2_0,
            DType::Q2_0,
            DType::Mxfp4,
            DType::Nvfp4,
        ] {
            let (s, _, _) = moe_native_fmt(dt).expect("covered by MoE native decode");
            assert_eq!(
                super::moe_gate_up_i8_kernel(s),
                format!("moe_gate_up_act_i8_{s}")
            );
            assert_eq!(super::moe_down_i8_kernel(s), format!("moe_down_i8_{s}"));
            assert_eq!(
                super::moe_gate_up_i8_routed_kernel(s),
                format!("moe_gate_up_act_i8_routed_{s}")
            );
            assert_eq!(
                super::moe_down_i8_routed_kernel(s),
                format!("moe_down_i8_routed_{s}")
            );
            // R8: the id-indexed multi-slot tier is total over the same set — it IS the shipping
            // resident path now, so a format in `moe_native_fmt` without an `_idm_` kernel is a
            // panic on a real model, not a slow fallback.
            assert_eq!(
                super::moe_gate_up_i8_idm_kernel(s),
                format!("moe_gate_up_act_i8_idm_{s}")
            );
            assert_eq!(
                super::moe_down_i8_idm_kernel(s),
                format!("moe_down_i8_idm_{s}")
            );
            // …and each name is actually INSTANTIATED in the module source (the kernels come out
            // of a token-pasting `GEN_*` macro, so the instantiation LINE is what to look for — a
            // table entry whose `GEN_*(fmt)` line was never added compiles fine on the host and
            // fails only on the box, at the launch).
            assert!(
                src.contains(&format!("GEN_MOE_GATE_UP_IDM({s})\n")),
                "moe_gate_up_act_i8_idm_{s} named but not instantiated"
            );
            // P7f: Q4_K CN=2 variant — standalone kernel, not a macro instantiation.
            if s == "q4k" {
                assert!(
                    src.contains("moe_gate_up_act_i8_idm_q4k_cn2"),
                    "moe_gate_up_act_i8_idm_q4k_cn2 not found in kernel source"
                );
            }
            assert!(
                src.contains(&format!("GEN_MOE_DOWN_IDM({s})\n")),
                "moe_down_i8_idm_{s} named but not instantiated"
            );
            // P2: the bucket-sorted batched tier is total over the SAME set for the same reason —
            // the executor picks it on shape alone (occupancy + `n_expert`), never on format, so a
            // format that reached `use_idb` without an `_idb_` kernel is a panic on a real model.
            assert_eq!(
                super::moe_gate_up_i8_idb_kernel(s),
                format!("moe_gate_up_act_i8_idb_{s}")
            );
            assert_eq!(
                super::moe_down_i8_idb_kernel(s),
                format!("moe_down_i8_idb_{s}")
            );
            assert!(
                src.contains(&format!("GEN_MOE_GATE_UP_IDB({s})\n")),
                "moe_gate_up_act_i8_idb_{s} named but not instantiated"
            );
            assert!(
                src.contains(&format!("GEN_MOE_DOWN_IDB({s})\n")),
                "moe_down_i8_idb_{s} named but not instantiated"
            );
        }
    }
}

/// R8: the id-GEMV's expert address must be a 64-BIT byte offset on a 64-bit pointer.
///
/// The Vulkan u64/BDA campaign's finding, transplanted: an expert base computed as an ELEMENT
/// count scaled inside the kernel wraps 32 bits on a real MoE bank (its `native_gemv_id` STREAMED
/// build had to move to `uint64_t(ids[slot]) * uint64_t(stride)` after the u32 element-space
/// multiply went coherent-but-wrong past ~102 Scout-sized slots). HIP pointers are 64-bit and
/// `long` is 64-bit on AMDGCN, so `base + (long)e * bstride` is 64-bit BY CONSTRUCTION — but only
/// while `bstride` stays a `long` PARAMETER. Narrow it to `int` and `e * bstride` silently becomes
/// a 32-bit multiply that wraps at 2 GiB, which llama4-Scout's 16 × 8192 × 5120 Q4_K down bank
/// (2.7 GiB) clears on its own. Nothing on a CPU-only box would catch that, so it is pinned here
/// against the emitted source; the host half (`usize` arithmetic widened to `i64`) is checked by
/// the arithmetic assertions below.
#[cfg(test)]
mod moe_id_multi_addressing_tests {
    /// The stride parameters, exactly as the `_idm_` kernels must declare them.
    const LONG_PARAMS: [&str; 4] = [
        "long gate_bstride",
        "long up_bstride",
        "long fused_up_half_boff",
        "long down_bstride",
    ];

    #[test]
    fn moe_id_multi_strides_are_64_bit() {
        let src = crate::kernels::hip_source();
        for p in LONG_PARAMS {
            assert!(src.contains(p), "id-GEMV stride param not 64-bit: `{p}`");
            // The `int` spelling must not appear anywhere — including in the `*_routed_*` twins,
            // which share the contract.
            let narrowed = format!("int {}", p.trim_start_matches("long "));
            assert!(
                !src.contains(&narrowed),
                "a 32-bit stride parameter survives: `{narrowed}`"
            );
        }
        // The multiply itself is on the WIDENED id, not on a 32-bit product later cast.
        for expr in [
            "gate_base + (long)e * gate_bstride",
            "up_base + (long)e * up_bstride",
            "down_base + (long)e * down_bstride",
        ] {
            assert!(
                src.contains(expr),
                "id-GEMV expert address is not a 64-bit multiply: expected `{expr}`"
            );
        }
        // Per-slot scratch indexing is likewise widened before the multiply — `n_slots * n_ff_exp`
        // passes 2^31 elements well before VRAM runs out on a long prefill chunk.
        for expr in [
            "h_out[(long)slot * nff + o]",
            "y[(long)slot * ne + d]",
            "hq + (long)slot * nff",
        ] {
            assert!(src.contains(expr), "scratch index not widened: `{expr}`");
        }
    }

    /// The host half: every byte stride the executor passes is computed in `usize` and handed over
    /// as `i64`, and the value is the one the bank layout implies. Reproduces the executor's own
    /// expressions over llama4-Scout's shape, where the products exceed `u32`.
    #[test]
    fn moe_id_multi_host_strides_exceed_u32_without_wrapping() {
        // Scout: ne = 5120, n_ff_exp = 8192, Q4_K banks (256 elems / 144 bytes per block).
        let (ne, nff, qpb, bpb) = (5120usize, 8192usize, 256usize, 144usize);
        let gate_bstride = ((nff * ne / qpb) * bpb) as i64;
        let down_bstride = ((ne * nff / qpb) * bpb) as i64;
        assert_eq!(gate_bstride, 23_592_960);
        assert_eq!(gate_bstride, down_bstride);
        // Expert 127's base is past 2^31 bytes — the boundary a 32-bit multiply would wrap at.
        let far = 127i64 * gate_bstride;
        assert!(
            far > i64::from(i32::MAX),
            "test shape no longer probes the boundary"
        );
        assert_eq!(far, 2_996_305_920);
        // The same product taken through i32 (what an `int` stride parameter would do) does NOT
        // agree — this is the bug the `long` declarations above prevent.
        assert_ne!(far, i64::from((127i32).wrapping_mul(gate_bstride as i32)));
    }
}

/// S6 (`docs/config-plan.md` §8): every ROCm kernel-tier knob drives its selector from a
/// `RocmCfg` VALUE. No environment, no `EnvGuard`, no GPU — these are pure kernel-name pickers.
#[cfg(test)]
mod config_tier_tests {
    use super::{
        fuse_weight_ok, i8_gemv_mrow, moe_i8_enabled, native_i8_fmt, native_wmma_fmt,
        q4k_coop_kernel, wmma_tile,
    };
    use infr_core::config::RocmCfg;
    use infr_core::DType;

    /// F4: the host divides `grid.x` by [`i8_gemv_mrow`], the kernel multiplies `blockIdx.x` by its
    /// own `I8_MROW` — a mismatch would silently skip or double-write output rows, and no parity
    /// test that runs a whole GEMV can distinguish "wrote row 2 twice" from "never wrote row 3"
    /// on a zeroed dst. Pin both halves against the kernel source: the constant's value, and the
    /// exact set of kernels that opted in (every other `linear_i8_*` still owns one row per wave).
    #[test]
    fn mrow_matches_the_kernel_source() {
        let src = crate::kernels::hip_source();
        assert!(
            src.contains("#define I8_MROW 2"),
            "kernel I8_MROW is not 2 — update `i8_gemv_mrow`"
        );
        for k in ["linear_i8_q4k", "linear_i8_q5k"] {
            assert_eq!(i8_gemv_mrow(k), 2, "{k} should be on the mrow grid");
            assert!(
                src.contains(&format!("void {k}(")),
                "{k} is not in the kernel source"
            );
        }
        // Every other covered format keeps `grid.x == out_f`.
        let cfg = RocmCfg::default();
        for dt in [
            DType::Q8_0,
            DType::Q2K,
            DType::Q3K,
            DType::Q6K,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Iq4Nl,
            DType::Iq4Xs,
            DType::Iq1S,
            DType::Tq2_0,
            DType::Mxfp4,
        ] {
            let (_, k) = native_i8_fmt(dt, &cfg).expect("covered by the int8 decode");
            assert_eq!(i8_gemv_mrow(k), 1, "{k} is one row per wave");
        }
    }

    fn cfg(f: impl FnOnce(&mut RocmCfg)) -> RocmCfg {
        let mut c = RocmCfg::default();
        f(&mut c);
        c
    }

    /// `INFR_ROCM_NO_I8` (`kernels.rocm.i8`, POSITIVE, default `true`): clearing it drops the int8
    /// GEMV, the WMMA prefill GEMM, the int8 MoE expert path AND both decode fusions (whose
    /// coverage predicate is the int8 table). Setting the env to `0` clears it too — presence is
    /// all that matters — which the env-layer polarity test in `infr-core` pins.
    #[test]
    fn i8_flag_gates_the_whole_int8_family() {
        let on = RocmCfg::default();
        assert!(on.i8);
        assert!(native_i8_fmt(DType::Q4K, &on).is_some());
        assert!(native_wmma_fmt(DType::Q4K, 4096, &on).is_some());
        assert!(moe_i8_enabled(&on));
        assert!(fuse_weight_ok(DType::Q4K, &on));

        let off = cfg(|c| c.i8 = false);
        assert!(native_i8_fmt(DType::Q4K, &off).is_none());
        assert!(native_wmma_fmt(DType::Q4K, 4096, &off).is_none());
        assert!(!moe_i8_enabled(&off));
        assert!(!fuse_weight_ok(DType::Q4K, &off));
    }

    /// `INFR_ROCM_NO_WMMA` (`kernels.rocm.no_wmma`, a `presence` knob on a NEGATIVE field) forces
    /// the dp4a GEMV without touching the int8 family.
    #[test]
    fn no_wmma_forces_the_gemv_but_keeps_int8() {
        let off = cfg(|c| c.no_wmma = true);
        assert!(native_wmma_fmt(DType::Q4K, 4096, &off).is_none());
        assert!(native_i8_fmt(DType::Q4K, &off).is_some());
    }

    /// `INFR_ROCM_NO_PIPE` (`kernels.rocm.pipe`, POSITIVE, default `true`): with it the Q4_K
    /// prefill is the Slice-27 software-pipelined kernel; without it, the Slice-25 auto tier
    /// (`2x2` for wide-N, `2x1` otherwise).
    #[test]
    fn pipe_flag_selects_the_q4k_prefill_kernel() {
        let on = RocmCfg::default();
        assert_eq!(
            native_wmma_fmt(DType::Q4K, 4096, &on),
            Some(("wmma_i8_q4k_pipe_2x1", 2, 1))
        );
        let off = cfg(|c| c.pipe = false);
        assert_eq!(
            native_wmma_fmt(DType::Q4K, 4096, &off),
            Some(("wmma_i8_q4k_2x2", 2, 2))
        );
        assert_eq!(
            native_wmma_fmt(DType::Q4K, 1024, &off),
            Some(("wmma_i8_q4k_2x1", 2, 1))
        );
    }

    /// `INFR_ROCM_WMMA_TILE` is matched against the EXACT strings `1x1`/`2x1`/`2x2` (§10.4);
    /// anything else — including a typo — is treated as unset and the shape-driven auto tier wins.
    #[test]
    fn wmma_tile_override_takes_only_the_three_literals() {
        assert_eq!(wmma_tile(4096, &RocmCfg::default()), (2, 2));
        assert_eq!(wmma_tile(1024, &RocmCfg::default()), (2, 1));
        for (spec, want) in [("1x1", (1, 1)), ("2x1", (2, 1)), ("2x2", (2, 2))] {
            let c = cfg(|c| c.wmma_tile = Some(spec.to_string()));
            assert_eq!(wmma_tile(4096, &c), want, "{spec}");
            assert_eq!(wmma_tile(1024, &c), want, "{spec}");
        }
        let bogus = cfg(|c| c.wmma_tile = Some("4x4".to_string()));
        assert_eq!(wmma_tile(4096, &bogus), (2, 2), "unrecognized ⇒ auto tier");
    }

    /// `INFR_ROCM_COOP` is the opt-in gate (default OFF); `INFR_ROCM_COOP_TILE` picks the tile and
    /// an unrecognized/absent name falls to the `128x64` default.
    #[test]
    fn coop_is_opt_in_and_its_tile_defaults() {
        assert!(q4k_coop_kernel(&RocmCfg::default()).is_none());
        let on = cfg(|c| c.coop = true);
        assert_eq!(
            q4k_coop_kernel(&on),
            Some(("wmma_i8_q4k_coop_128x64_w8", 128, 64, 256))
        );
        let tiled = cfg(|c| {
            c.coop = true;
            c.coop_tile = Some("64x64".to_string());
        });
        assert_eq!(
            q4k_coop_kernel(&tiled),
            Some(("wmma_i8_q4k_coop_64x64_w4", 64, 64, 128))
        );
        let bogus = cfg(|c| {
            c.coop = true;
            c.coop_tile = Some("nope".to_string());
        });
        assert_eq!(
            q4k_coop_kernel(&bogus),
            Some(("wmma_i8_q4k_coop_128x64_w8", 128, 64, 256))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flash-prefill tiling policy, pinned at every head dim the backend actually sees plus the
    /// three ways a shape is refused. `br` is the factor by which the kernel divides global K/V
    /// traffic, so "the widest workgroup that fits" IS the policy, and a regression here is a
    /// silent perf loss rather than a wrong answer.
    #[test]
    fn attn_flash_tiling_picks_the_widest_query_tile_that_fits_lds() {
        // Refusals: head dims whose padded LDS stride is not an odd word count (bank conflicts)
        // or whose rows are not 16 B-aligned for the `uint4` stage, one past the kernel's
        // `ATTN_FLASH_MAXP2` bound, and the degenerate zero.
        for hd in [0, 4, 20, 100, 320, 512] {
            assert_eq!(attn_flash_tiling(hd), None, "head_dim {hd} must not tile");
        }
        // qwen3 / llama / qwen3moe: the 32-key tile, and the widest workgroup inside the budget.
        for hd in [64, 96, 128] {
            let t = attn_flash_tiling(hd).unwrap_or_else(|| panic!("head_dim {hd} must tile"));
            assert_eq!(t.bc, 32, "head_dim {hd}");
            assert!(
                t.smem(hd) <= ATTN_FLASH_LDS,
                "head_dim {hd} overflows the LDS budget"
            );
            // One `nw` up either exceeds the budget or is off the top of the candidate list —
            // that is what makes this the WIDEST tile rather than merely a fitting one.
            let wider = AttnFlashTiling {
                nw: t.nw * 2,
                bc: t.bc,
            };
            assert!(
                t.nw == 8 || wider.smem(hd) > ATTN_FLASH_LDS,
                "head_dim {hd}: nw={} left a wider tile on the table",
                t.nw
            );
        }
        assert_eq!(
            attn_flash_tiling(128),
            Some(AttnFlashTiling { nw: 8, bc: 32 })
        );
        // gemma-3's 256: the K/V tiles alone would be 33 KiB at bc=32, so the policy narrows the
        // KEY tile first and only then the query tile.
        let g = attn_flash_tiling(256).expect("head_dim 256 must tile");
        assert_eq!((g.bc, g.nw), (16, 4));
        assert_eq!(g.br(), 8);
        assert!(g.smem(256) <= ATTN_FLASH_LDS);
    }

    /// **The P6 selector routes exactly the head dims that are instantiated, and nothing else.** A
    /// `head_dim` whose lane count has no template instantiation would resolve a kernel name that
    /// is not in the module — a `hipModuleGetFunction` failure at the first decode, not a slow
    /// path — so the two lists have to be checked against each other rather than kept in step by
    /// hand. Both directions: every selected name is present in the HIP source, and every
    /// uninstantiated lane count declines.
    #[test]
    fn attn_pf_selects_only_instantiated_kernels() {
        let src = crate::kernels::hip_source();
        // head_dim -> the lane count it selects. 32/64 -> npl 1/2, 128 -> 4, 256 -> 8, and the
        // ragged dims in between (they round UP, and the kernel masks the spare lanes).
        for hd in [64usize, 40, 128, 100, 256, 225] {
            let k = attn_pf_npl(hd).unwrap_or_else(|| panic!("head_dim {hd} must select a kernel"));
            for name in [k.plain, k.split_partial] {
                assert!(
                    src.contains(&format!("{name},")),
                    "head_dim {hd} selects `{name}`, which the module never instantiates"
                );
            }
        }
        // npl 1, 3, 5, 6 and the degenerate zero have no instantiation and must decline.
        for hd in [0usize, 20, 32, 96, 160, 190] {
            assert_eq!(attn_pf_npl(hd), None, "head_dim {hd} must not select");
        }
        // The plain and split names must never be crossed — they take different argument lists, so
        // a swap is a silent stack smash rather than a compile error.
        for hd in [64usize, 128, 256] {
            let k = attn_pf_npl(hd).unwrap();
            assert!(k.plain.starts_with("attention_pf_npl"));
            assert!(k
                .split_partial
                .starts_with("attention_split_partial_pf_npl"));
        }
    }
}
