//! MoE expert-bank GEMM dispatch: activation-representation selection (`ActsKind` /
//! `expert_acts_kind`) and the per-o-range GEMM (`expert_gemm_range`) that fans out to the
//! quant-specific batch kernels.
use crate::bytes_to_f32;
use crate::kernels::{
    dot, dot_bf16, dot_f16, vec_dot_q32_batch8, vec_dot_q4k_batch, vec_dot_q4k_batch8,
    vec_dot_q5_0_32_batch, vec_dot_q5k_batch, vec_dot_q6k_batch, vec_dot_q8_0_32_batch,
    vec_dot_q8_0_batch,
};
use crate::quant::{Q8x32, Q8};
#[cfg(target_arch = "x86_64")]
use crate::repack::q4k_gemm_group;
use crate::repack::Q4kPack;
use infr_core::tensor::DType;

/// Activation batch for [`expert_gemm_range`] — the representation the weight dtype dictates
/// (see [`expert_acts_kind`]). The staged `Op::MoeFfn` pipeline builds these ONCE per stage
/// (each distinct hidden row quantized a single time, then cloned per routed pair) and every
/// o-range task borrows a bucket's slice.
pub(crate) enum ExpertActs<'a> {
    /// 256-super-block int8 (`quantize_q8`): Q4K/Q6K/Q8_0/Q5K weights with `in_f % 256 == 0`.
    Super(&'a [Q8]),
    /// 32-block int8 (`quantize_q8_32`): Q8_0/Q5_0 weights at misaligned `in_f % 32 == 0`
    /// (e.g. DiffusionGemma's down `in_f = 704`).
    Blk32(&'a [Q8x32]),
    /// Row-major f32 `[count, in_f]`: f16/bf16/f32 weights, the dequant fallback, and ALL
    /// single-row (`int8_ok == false`) calls, which stay byte-for-byte exact (Metal parity).
    Raw(&'a [f32]),
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ActsKind {
    Super,
    Blk32,
    Raw,
}

/// Which activation representation an expert bank of this `(dtype, in_f)` uses — the SAME
/// dispatch order the old `expert_matvec_batch` fast paths had, so every (weights, activations)
/// pairing lands on the identical kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn expert_acts_kind(dt: DType, in_f: usize, int8_ok: bool) -> ActsKind {
    if int8_ok
        && matches!(dt, DType::Q4K | DType::Q6K | DType::Q8_0 | DType::Q5K)
        && in_f.is_multiple_of(256)
    {
        ActsKind::Super
    } else if int8_ok && matches!(dt, DType::Q8_0 | DType::Q5_0) && in_f.is_multiple_of(32) {
        ActsKind::Blk32
    } else {
        ActsKind::Raw
    }
}

/// Compute output rows `[o0, o1)` of ONE expert bank into `out_t` — an o-major slice of exactly
/// `(o1-o0) * count` floats (`out_t[(o - o0) * count + r]`). `o0` MUST be 8-aligned: the
/// Q4_K / 32-block 8-row tiles are anchored at `o = 0`, so 8-aligned chunking reproduces the
/// exact tile boundaries (and therefore bit-identical per-element results) of a whole-bank call,
/// no matter how a task list splits the range. The kernels and their dispatch mirror the old
/// `expert_matvec_batch` fast paths 1/2 + fallback one-for-one.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn expert_gemm_range(
    wbytes: &[u8],
    dt: DType,
    in_f: usize,
    out_f: usize,
    acts: &ExpertActs,
    q4k_pack: Option<&Q4kPack>,
    o0: usize,
    o1: usize,
    out_t: &mut [f32],
) {
    debug_assert!(o0.is_multiple_of(8) && o0 <= o1 && o1 <= out_f);
    let bpr = wbytes.len() / out_f;
    let count = if o1 > o0 { out_t.len() / (o1 - o0) } else { 0 };
    match acts {
        ExpertActs::Super(q8s) => {
            let dot_col = |o: usize, col: &mut [f32]| {
                let row = &wbytes[o * bpr..o * bpr + bpr];
                match dt {
                    DType::Q4K => vec_dot_q4k_batch(row, q8s, in_f, col),
                    DType::Q6K => vec_dot_q6k_batch(row, q8s, in_f, col),
                    DType::Q8_0 => vec_dot_q8_0_batch(row, q8s, in_f, col),
                    DType::Q5K => vec_dot_q5k_batch(row, q8s, in_f, col),
                    _ => unreachable!("Super acts imply a 256-super-block quant dtype"),
                }
            };
            // 8-row tiling for Q4_K only (`vec_dot_q4k_batch8` / the cached ilv pack) — the same
            // gate the old fast path 1 had; Q6K/Q8_0/Q5K run per-row.
            let tiled_end = if dt == DType::Q4K { out_f / 8 * 8 } else { 0 };
            let mut o = o0;
            while o + 8 <= o1.min(tiled_end) {
                let cols = &mut out_t[(o - o0) * count..(o - o0 + 8) * count];
                #[cfg(target_arch = "x86_64")]
                if let Some(pack) = q4k_pack {
                    // SAFETY: packs are only built when the VNNI ilv dispatch applies.
                    unsafe { q4k_gemm_group(&pack.groups[o / 8], pack.nb, q8s, cols) };
                    o += 8;
                    continue;
                }
                let rows: [&[u8]; 8] =
                    std::array::from_fn(|i| &wbytes[(o + i) * bpr..(o + i) * bpr + bpr]);
                let mut it = cols.chunks_mut(count);
                let outs: [&mut [f32]; 8] = std::array::from_fn(|_| it.next().unwrap());
                vec_dot_q4k_batch8(rows, q8s, in_f, outs);
                o += 8;
            }
            while o < o1 {
                dot_col(o, &mut out_t[(o - o0) * count..(o - o0 + 1) * count]);
                o += 1;
            }
        }
        ExpertActs::Blk32(q8s) => {
            let q5 = dt == DType::Q5_0;
            let dot_col = |o: usize, col: &mut [f32]| {
                let row = &wbytes[o * bpr..o * bpr + bpr];
                match dt {
                    DType::Q8_0 => vec_dot_q8_0_32_batch(row, q8s, in_f, col),
                    DType::Q5_0 => vec_dot_q5_0_32_batch(row, q8s, in_f, col),
                    _ => unreachable!("Blk32 acts imply Q8_0/Q5_0"),
                }
            };
            // Both Q8_0 and Q5_0 ride the interleaved 8-row tile (old fast path 2).
            let tiled_end = out_f / 8 * 8;
            let mut o = o0;
            while o + 8 <= o1.min(tiled_end) {
                let cols = &mut out_t[(o - o0) * count..(o - o0 + 8) * count];
                let rows: [&[u8]; 8] =
                    std::array::from_fn(|i| &wbytes[(o + i) * bpr..(o + i) * bpr + bpr]);
                let mut it = cols.chunks_mut(count);
                let outs: [&mut [f32]; 8] = std::array::from_fn(|_| it.next().unwrap());
                vec_dot_q32_batch8(rows, q8s, in_f, outs, q5);
                o += 8;
            }
            while o < o1 {
                dot_col(o, &mut out_t[(o - o0) * count..(o - o0 + 1) * count]);
                o += 1;
            }
        }
        ExpertActs::Raw(xin) => {
            // f32 dots against raw activation rows: identical math to the old fallback (the
            // weight row is dequantized ONCE, reused across all rows), o-major placement.
            for o in o0..o1 {
                let row = &wbytes[o * bpr..o * bpr + bpr];
                let col = &mut out_t[(o - o0) * count..(o - o0 + 1) * count];
                match dt {
                    DType::F32 => {
                        let w32: &[f32] = bytemuck::cast_slice(row);
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dot(w32, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                    DType::F16 => {
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dot_f16(row, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                    DType::Bf16 => {
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dot_bf16(row, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                    _ => {
                        let wf = bytes_to_f32(row, dt);
                        for (r, dst) in col.iter_mut().enumerate() {
                            *dst = dot(&wf, &xin[r * in_f..r * in_f + in_f]);
                        }
                    }
                }
            }
        }
    }
}
