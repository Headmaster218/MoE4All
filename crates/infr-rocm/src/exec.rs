//! Graph execution: walk ops → resolve bound buffers → dispatch HIP kernels.
//!
//! Covered quant formats (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS, see
//! `native_decode_fmt`) are decoded in-kernel from their RAW bytes on the `Linear`/`EmbedGather`
//! paths — no f16 cache, VRAM ≈ quant_size.
//! Uncovered quantized weight tensors are dequantized to f16 on the host on first touch and
//! cached by the raw device-pointer address of their bound buffer.

use crate::backend::{bucket_bytes, BufferPool};
use crate::ffi::{self, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};
use crate::kernels::Pipelines;
use half::f16;
use infr_core::backend::{Bindings, GraphPlan, Plan};
use infr_core::error::Result;
use infr_core::graph::{AttnMask, Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_gguf::dequant;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
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
        _ => return None,
    };
    Some((infr_core::decode_spec::block_layout(dt).1, kernel))
}

/// Dequant-to-f16 kernel name (`deqf16_*`, kernels.rs `DEQUANT_F16`) for a covered dtype — the
/// weight decoder feeding the Slice-26 rocBLAS f16 prefill GEMM. Same covered set as
/// [`native_decode_fmt`] (Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS);
/// `None` keeps a format off it.
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
    if out_f >= 2048 {
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
    target_chunks: 32,
    min_chunk: 64,
    max_chunk: 512,
    rounding: infr_core::tier::ChunkRounding::Up,
};

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
        _ => return None,
    };
    let (elems, bytes) = infr_core::decode_spec::block_layout(dt);
    Some((suffix, elems, bytes))
}

/// Static kernel name for the `(gate/up format, down format)` combo of the Phase-3 f16-decode expert
/// FFN, or `None` when that pair is NOT instantiated in `kernels.rs` — the caller then keeps the
/// dequant→f16 `moe_ffn_expert` fallback. Instantiated set (62 of the 121 `moe_native_fmt` pairs):
///
/// * the full `{q80, q2k, q3k, q4k, q5k, q6k}²` (36 — e.g. Q4_K_M is `("q4k", "q6k")`, Q3_K_M is
///   `("q3k", "q5k")`),
/// * `{q40, q41, q51} × {q40, q41, q51, q80}` (12, R3),
/// * `{iq4nl, iq4xs} × {iq4nl, iq4xs, q4k, q5k, q6k, q80}` (12, R4),
/// * `{q2k, q3k} × {iq4nl}` (2, R4).
///
/// **Why not the full cross product** (R2 documented this escape hatch, R3 measured it and took it):
/// going 6×6 → 9×9 cost **+1.1 s of COLD hiprtc** — backend init plus a 1-token bench with
/// `~/.cache/comgr` cleared went 4.31 s → 6.27 s, against 5.44 s for the 48-pair set. R4 re-measured
/// the same way at the 11-format mark (3 reps each): **5.50-5.55 s** at R3's 48 pairs → **6.39-6.60
/// s** once R4's 24 DENSE kernels are added at the same 48 pairs → **6.72-6.75 s** at the shipped
/// 62. So the 14 pairs added above cost only ~0.25 s (~18 ms each) — the dense kernels, i.e. the
/// actual feature, are ~0.9 s of the delta — while the full 11×11 would have piled on ~59 more
/// cells. Warm-cache startup is unchanged at ~0.48 s in every variant. The cells cut are the ones
/// nothing can reach:
///
/// * These kernels are NOT the shipping MoE path. The default int8 dp4a expert path dispatches the
///   per-FORMAT `moe_gate_up_act_i8_<gu>` + `moe_down_i8_<dn>` kernels, which ARE total over
///   `moe_native_fmt` (11 each). `moe_ffn_expert_<gu>_<dn>` runs only under `INFR_ROCM_NO_I8` — an
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
        _ => unreachable!("moe_down_i8_routed_kernel: uncovered ({dn})"),
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

struct ExecCtx<'a> {
    dev: Vec<Option<crate::RocmBuffer>>,
    vals: Vec<Option<Vec<f32>>>,
    weight_cache: &'a Mutex<HashMap<(usize, usize), crate::RocmBuffer>>,
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
            host_ptr: std::ptr::null_mut(),
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
                    host_ptr: std::ptr::null_mut(),
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
                    host_ptr: std::ptr::null_mut(),
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
            host_ptr: std::ptr::null_mut(),
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
    weight_cache: &Mutex<HashMap<(usize, usize), crate::RocmBuffer>>,
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
    infr_core::exec::run_ops(
        &g.ops,
        &fusion.skip,
        &mut RocmDispatch {
            g,
            bindings,
            pipelines,
            fusion: &fusion,
            ctx: &mut ctx,
        },
    )?;

    // Barrier all queued op work before the writeback: the writeback `hipMemcpyDtoD` runs on the
    // NULL stream, which is NOT ordered against our non-default work stream, so it must observe a
    // completed stream first.
    unsafe { ffi::hipStreamSynchronize(stream) };
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
// Two adjacent-op merges the backend detects on the AGNOSTIC graph (so they apply to every arch),
// each with a scalar fallback when the pattern doesn't match:
//
//   1. `RmsNorm → Linear` (input_norm→qkv, post_attn_norm→gate/up): elide the standalone `rmsnorm`
//      kernel + its normalized-activation DRAM round-trip; every consuming decode GEMV normalizes
//      and int8-quantizes its RAW input row in one `rmsnorm_quant_i8_32` pass (bit-faithful).
//   2. `Linear → Add(residual)` (o_proj, down_proj): fold the residual Add into the GEMV epilogue
//      (`dst = gemv + residual`), killing the standalone `add` kernel + its round-trip.
//
// Both are gated to decode (`m == 1`) int8 GEMVs — the shipping default path (every `native_i8_fmt`
// format).
// Prefill (m>1, WMMA/rocBLAS) and uncovered formats keep the split ops. Escape hatches:
// `INFR_ROCM_NO_FUSE_NORM` / `INFR_ROCM_NO_FUSE_ADD`.
#[derive(Default)]
struct DecodeFusion {
    /// Linear op idx → (raw pre-norm x, norm weight, eps): run `rmsnorm_quant_i8_32` on the raw row
    /// instead of `quant_i8_32` on the (elided) normalized input.
    norm: HashMap<usize, (TensorId, TensorId, f32)>,
    /// Linear op idx → (residual operand, add dst): fold the following `Add` into the GEMV epilogue.
    add: HashMap<usize, (TensorId, TensorId)>,
    /// Op indices to elide entirely (the fused-away `RmsNorm` / `Add`).
    skip: HashSet<usize>,
}

/// Weight-dtype predicate for BOTH decode fusions: a covered int8-decode GEMV format
/// (`native_i8_fmt`, i.e. every natively decoded format, or `None` under `INFR_ROCM_NO_I8`). The `rmsnorm→
/// int8-decode-Linear` and `int8-decode-Linear→Add` folds share it.
fn fuse_weight_ok(dt: DType, rocm: &infr_core::config::RocmCfg) -> bool {
    native_i8_fmt(dt, rocm).is_some()
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
    let cfg = infr_core::fusion::FusionCfg {
        linear_add: Some(infr_core::fusion::LinearAddCfg {
            weight_ok: &weight_ok,
            // `INFR_ROCM_NO_FUSE_ADD` (config `kernels.rocm.fuse_add`, positive polarity):
            // PRESENCE of the env key — including `=0` — turns the fold off.
            enabled: engine.kernels.rocm.fuse_add,
        }),
        rmsnorm_linear: Some(infr_core::fusion::RmsNormLinearCfg {
            weight_ok: &weight_ok,
            // `INFR_ROCM_NO_FUSE_NORM` (config `kernels.rocm.fuse_norm`), same polarity.
            enabled: engine.kernels.rocm.fuse_norm,
        }),
        kv_write: false,
    };
    let plan = infr_core::fusion::plan_fusions(g, &cfg);
    DecodeFusion {
        norm: plan.rmsnorm_linear,
        add: plan.linear_add,
        skip: plan.skip,
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
}

impl infr_core::exec::OpDispatch for RocmDispatch<'_, '_, '_> {
    fn dispatch(&mut self, i: usize, op: &Op) -> Result<()> {
        run_op(
            op,
            self.g,
            self.bindings,
            self.pipelines,
            self.ctx,
            self.fusion.norm.get(&i).copied(),
            self.fusion.add.get(&i).copied(),
        )
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
                        if let Some((x_raw, norm_w, eps)) = norm_fuse {
                            // Slice-32 RmsNorm→Linear: one block per row reduces the sum-of-squares
                            // over the RAW row, then int8-quantizes the normalized row in registers
                            // (bit-identical to `rmsnorm` then `quant_i8_32`), killing the `rmsnorm`
                            // launch + the normalized-activation DRAM round-trip.
                            let wnptr = ctx.dequant_weight_or_cache(norm_w, g, bindings)?;
                            let xrp = ctx.ensure_device(x_raw, g, bindings)?;
                            dispatch_grid(
                                pipelines,
                                ctx.stream,
                                "rmsnorm_quant_i8_32",
                                m,
                                1,
                                256,
                                args![
                                    arg_ptr(xrp),
                                    arg_ptr(wnptr),
                                    arg_ptr(qx.ptr),
                                    arg_ptr(xs.ptr),
                                    arg_i32(m as i32),
                                    arg_i32(in_f as i32),
                                    arg_f32(eps),
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
                                    arg_ptr(bx_ptr),
                                    arg_ptr(qx.ptr),
                                    arg_ptr(xs.ptr),
                                    arg_i32(m as i32),
                                    arg_i32(in_f as i32),
                                ],
                            )?;
                        }
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
                                }
                            }
                            None => ctx.zero_dev(mu * ou),
                        };
                        // Slice-28: Q4_K prefill (m>1) can OPT IN (`INFR_ROCM_COOP=1`) to the
                        // cooperative decode-once GEMM (multi-warp threadblock, LDS-shared weight
                        // tile). It is bit-faithful to `wmma_i8_q4k_2x1` (goldens hold) but measured
                        // a regression on gfx1100 (see `q4k_coop_kernel`), so the DEFAULT stays the
                        // Slice-27 pipe. When not opted in, this falls through to the pipe / GEMV.
                        let coop = (m > 1 && wdt == DType::Q4K && !ctx.rocm.no_wmma && ctx.rocm.i8)
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
                                    // one wave32 block per (output row, activation row). `resid_ptr`
                                    // (null unless the Slice-32 residual Add is fused) folds the add
                                    // into the epilogue.
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
                                            arg_ptr(resid_ptr),
                                            arg_i32(m as i32),
                                            arg_i32(in_f as i32),
                                            arg_i32(out_f as i32),
                                        ],
                                    )?;
                                }
                            },
                        }
                        // When the residual Add is fused, `dd` aliases the residual stream buffer
                        // (already mapped in `ctx.dev` via `ensure_device(add_dst)`) and the result
                        // is written in place — nothing to remap. Otherwise publish the fresh dst.
                        if add_fuse.is_none() {
                            ctx.dev[dst.0 as usize] = Some(dd);
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
                if wt_staged {
                    weight_staged_done(ctx)?;
                }
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
                         (only Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_1/IQ4_NL/IQ4_XS page \
                          — the remaining IQ/fp4/ternary formats await native MoE decode)",
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
            let use_i8 = native.is_some() && moe_i8_enabled(ctx.rocm);
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

                for row in 0..rows {
                    let x_row = unsafe { (x_ptr as *mut u8).add(row * neu * 4) as *mut c_void };
                    let dst_row = unsafe { (dd.ptr as *mut u8).add(row * neu * 4) as *mut c_void };
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
                        } else if let Some(((gu, gu_qpb, gu_bpb), (dn, dn_qpb, dn_bpb))) = native {
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
        ] {
            let (_, e, b) = moe_native_fmt(dt).expect("covered by MoE native decode");
            assert_eq!((e, b), (elems, bytes), "{dt:?} moe_native_fmt geometry");
        }
        // Coverage boundary: the formats still awaiting a native kernel stay off the fast path.
        assert!(native_decode_fmt(DType::Iq2Xxs).is_none());
        assert!(native_i8_fmt(DType::Iq2Xxs, &rocm).is_none());
        // Q5_0 is native on the DENSE paths but has no MoE expert kernel — no shipped GGUF packs
        // expert banks as Q5_0, so it stays off the (gate/up × down) cross product.
        assert!(moe_native_fmt(DType::Q5_0).is_none());
    }

    /// The `(gate/up, down)` pairs `kernels.rs` instantiates the cross-product expert kernels for —
    /// the expected set both mappers are pinned against. Rationale lives on `moe_expert_kernel`.
    const MOE_EXPERT_PAIRS: &[(&str, &str)] = &{
        // The K-quant square, then three rectangles: the R3 legacy round quants, the R4 codebook
        // quants, and the `convert_incompatible_tensor` K-quant→IQ4_NL down bump.
        const K: [&str; 6] = ["q80", "q2k", "q3k", "q4k", "q5k", "q6k"];
        const L: [&str; 3] = ["q40", "q41", "q51"];
        const LD: [&str; 4] = ["q40", "q41", "q51", "q80"];
        const I: [&str; 2] = ["iq4nl", "iq4xs"];
        const ID: [&str; 6] = ["iq4nl", "iq4xs", "q4k", "q5k", "q6k", "q80"];
        const XK: [&str; 2] = ["q2k", "q3k"];
        let mut out =
            [("", ""); K.len() * K.len() + L.len() * LD.len() + I.len() * ID.len() + XK.len()];
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
        out
    };

    /// R3 escape hatch (extended by R4): the Phase-3 cross-product expert kernels are instantiated for
    /// [`MOE_EXPERT_PAIRS`] only, and BOTH mappers must cover exactly that set — the `MoeFfn` arm
    /// checks availability once (via `moe_expert_kernel`) and then dispatches EITHER the host-routed
    /// or the device-routed kernel, so a table that drifted would `expect`-panic at run time on a
    /// real GGUF. Also pins that the mappers never claim a pair `kernels.rs` does not instantiate.
    #[test]
    fn moe_expert_pair_tables_agree() {
        use super::{moe_expert_kernel, moe_expert_routed_kernel};
        const ALL: [&str; 12] = [
            "q80", "q2k", "q3k", "q4k", "q5k", "q6k", "q40", "q41", "q51", "iq4nl", "iq4xs", "q50",
        ];
        assert_eq!(
            MOE_EXPERT_PAIRS.len(),
            62,
            "6×6 + 3×4 + 2×6 + 2 instantiated pairs"
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
        }
    }
}

/// S6 (`docs/config-plan.md` §8): every ROCm kernel-tier knob drives its selector from a
/// `RocmCfg` VALUE. No environment, no `EnvGuard`, no GPU — these are pure kernel-name pickers.
#[cfg(test)]
mod config_tier_tests {
    use super::{
        fuse_weight_ok, moe_i8_enabled, native_i8_fmt, native_wmma_fmt, q4k_coop_kernel, wmma_tile,
    };
    use infr_core::config::RocmCfg;
    use infr_core::DType;

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
