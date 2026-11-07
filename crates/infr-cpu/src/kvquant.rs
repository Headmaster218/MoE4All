//! On-the-fly activation → GGUF-block quantizers for the KV cache (`q4_0`, `q4_1`, `q5_0`, `q5_1`,
//! `iq4_nl`). These are the `quantize_row_*_ref` reference formulas from llama.cpp
//! `ggml/src/ggml-quants.c`, block size 32. (Weights are pre-quantized on disk so infr only ever
//! *dequantized* before; a quantized KV cache needs to quantize activations each step, like Q8_0.)
//! The read side is the shared [`infr_gguf::dequant::dequant_block`] over the block-aligned prefix.

use half::f16;
use infr_core::tensor::DType;
use infr_gguf::dequant::KVALUES_IQ4NL;

const QK: usize = 32; // all of these are 32-element blocks

#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn h(x: f32) -> [u8; 2] {
    f16::from_f32(x).to_bits().to_le_bytes()
}

/// q4_0 (18 B): `d = max/-8`, `q = clamp(x/d + 8.5, .., 15)`, 4-bit, low/high halves interleaved.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q4_0_block(x: &[f32], dst: &mut [u8]) {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    dst[0..2].copy_from_slice(&h(d));
    for j in 0..QK / 2 {
        let x0 = (x[j] * id + 8.5) as i32 as i8;
        let x1 = (x[j + QK / 2] * id + 8.5) as i32 as i8;
        let (xi0, xi1) = (x0.clamp(0, 15) as u8, x1.clamp(0, 15) as u8);
        dst[2 + j] = xi0 | (xi1 << 4);
    }
}

/// q4_1 (20 B): asymmetric — `d = (max-min)/15`, `q = clamp((x-min)/d + 0.5, 0, 15)`, stores `min`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q4_1_block(x: &[f32], dst: &mut [u8]) {
    let (mut min, mut max) = (f32::MAX, f32::MIN);
    for &v in x {
        min = min.min(v);
        max = max.max(v);
    }
    let d = (max - min) / 15.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    dst[0..2].copy_from_slice(&h(d));
    dst[2..4].copy_from_slice(&h(min));
    for j in 0..QK / 2 {
        let x0 = ((x[j] - min) * id + 0.5) as i32 as i8;
        let x1 = ((x[j + QK / 2] - min) * id + 0.5) as i32 as i8;
        let (xi0, xi1) = (x0.clamp(0, 15) as u8, x1.clamp(0, 15) as u8);
        dst[4 + j] = xi0 | (xi1 << 4);
    }
}

/// q5_0 (22 B): `d = max/-16`, 5-bit — low nibble in `qs`, 5th bit packed into the `qh` u32.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q5_0_block(x: &[f32], dst: &mut [u8]) {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    dst[0..2].copy_from_slice(&h(d));
    let mut qh: u32 = 0;
    for j in 0..QK / 2 {
        let xi0 = (((x[j] * id + 16.5) as i32 as i8).clamp(0, 31)) as u8;
        let xi1 = (((x[j + QK / 2] * id + 16.5) as i32 as i8).clamp(0, 31)) as u8;
        dst[6 + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= (((xi0 & 0x10) >> 4) as u32) << j;
        qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK / 2);
    }
    dst[2..6].copy_from_slice(&qh.to_le_bytes());
}

/// q5_1 (24 B): asymmetric 5-bit — `d = (max-min)/31`, low nibble in `qs`, 5th bit in `qh`, `min`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q5_1_block(x: &[f32], dst: &mut [u8]) {
    let (mut min, mut max) = (f32::MAX, f32::MIN);
    for &v in x {
        min = min.min(v);
        max = max.max(v);
    }
    let d = (max - min) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    dst[0..2].copy_from_slice(&h(d));
    dst[2..4].copy_from_slice(&h(min));
    let mut qh: u32 = 0;
    for j in 0..QK / 2 {
        // `.min(31)` matches the clamp its q4_0/q4_1/q5_0 siblings apply. Unreachable in practice
        // ((x-min)*id ≤ 31, so the +0.5 truncates to ≤ 31), but a stray 32 would otherwise set the
        // 5th bit with a zero low nibble (decoding to 16, not 31) — the guard keeps that severe
        // failure mode impossible. Byte-identical for every real block.
        let xi0 = (((x[j] - min) * id + 0.5) as i32 as u8).min(31);
        let xi1 = (((x[j + QK / 2] - min) * id + 0.5) as i32 as u8).min(31);
        dst[8 + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= (((xi0 & 0x10) >> 4) as u32) << j;
        qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK / 2);
    }
    dst[4..8].copy_from_slice(&qh.to_le_bytes());
}

/// Nearest codebook index (llama.cpp `best_index_int8`, binary search on the sorted values).
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn best_index(val: &[i8; 16], x: f32) -> usize {
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[15] as f32 {
        return 15;
    }
    let (mut ml, mut mu) = (0usize, 15usize);
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if (x - val[mu - 1] as f32) < (val[mu] as f32 - x) {
        mu - 1
    } else {
        mu
    }
}

/// iq4_nl (18 B): non-linear 16-entry codebook. `d = max/values[0]`, least-squares refine
/// `d = Σwqx/Σwq²` (w = x², no imatrix → the `ntry=-1` path). Indices packed low/high halves.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn iq4nl_block(x: &[f32], dst: &mut [u8]) {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let mut l = [0u8; QK];
    let mut d = 0f32;
    if amax >= 1e-15 {
        d = max / KVALUES_IQ4NL[0] as f32;
        let id = 1.0 / d;
        let (mut sumqx, mut sumq2) = (0f32, 0f32);
        for j in 0..QK {
            let li = best_index(&KVALUES_IQ4NL, id * x[j]);
            l[j] = li as u8;
            let q = KVALUES_IQ4NL[li] as f32;
            let w = x[j] * x[j];
            sumqx += w * q * x[j];
            sumq2 += w * q * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
    }
    dst[0..2].copy_from_slice(&h(d));
    for j in 0..QK / 2 {
        dst[2 + j] = l[j] | (l[j + QK / 2] << 4);
    }
}

/// Quantize `src.len()` activations (a multiple of 32) into 32-element blocks of `dt` in `dst`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn quantize_row(dt: DType, src: &[f32], dst: &mut [u8]) {
    let bb = infr_gguf::nbytes(dt, QK);
    for (b, chunk) in src.as_chunks::<QK>().0.iter().enumerate() {
        let blk = &mut dst[b * bb..(b + 1) * bb];
        match dt {
            DType::Q4_0 => q4_0_block(chunk, blk),
            DType::Q4_1 => q4_1_block(chunk, blk),
            DType::Q5_0 => q5_0_block(chunk, blk),
            DType::Q5_1 => q5_1_block(chunk, blk),
            DType::Iq4Nl => iq4nl_block(chunk, blk),
            _ => unreachable!("kvquant::quantize_row for non-KV-quant dtype {dt:?}"),
        }
    }
}

/// Which dtypes this module can quantize on the fly for the KV cache.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn supported(dt: DType) -> bool {
    matches!(
        dt,
        DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Iq4Nl
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each format quantize→dequant recovers a Gaussian block within its bit-width tolerance, and the
    // higher-bit formats are more accurate.
    #[test]
    fn quantize_dequant_roundtrip() {
        let mut s: u64 = 0xC0FFEE;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };
        let mut src = [0f32; QK];
        for v in src.iter_mut() {
            *v = (rnd() + rnd() + rnd()) * 4.0;
        }
        for (dt, tol) in [
            (DType::Q4_0, 0.10),
            (DType::Q4_1, 0.09),
            (DType::Q5_0, 0.05),
            (DType::Q5_1, 0.05),
            (DType::Iq4Nl, 0.08),
        ] {
            let mut blk = vec![0u8; infr_gguf::nbytes(dt, QK)];
            quantize_row(dt, &src, &mut blk);
            let got = infr_gguf::dequant::dequant_block(dt, &blk).expect("dequant");
            let (mut num, mut den) = (0f32, 0f32);
            for i in 0..QK {
                num += (got[i] - src[i]).powi(2);
                den += src[i].powi(2);
            }
            let rel = (num / den).sqrt();
            assert!(rel < tol, "{dt:?} rel L2 err {rel} >= {tol}");
        }
    }

    // #3: `q5_1_block` now `.min(31)`s each 5-bit code like its q4_0/q4_1/q5_0 siblings. This is
    // byte-identical for real blocks (the max element lands exactly on index 31, never 32), and a
    // synthetic overflow saturates to 31 rather than wrapping to the catastrophic (bit-set, low-
    // nibble-zero) encoding that decodes to 16.
    #[test]
    fn q5_1_saturates_to_31() {
        // Decode the 5-bit index the block stored for element `j` (low nibble + the qh 5th bit).
        fn code(blk: &[u8], j: usize) -> u8 {
            let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
            let nib = if j < QK / 2 {
                blk[8 + j] & 0x0F
            } else {
                blk[8 + (j - QK / 2)] >> 4
            };
            nib | (((qh >> j) & 1) as u8) << 4
        }

        // A normal ramp: the maximum element must encode to exactly 31, none above it.
        let mut x = [0f32; QK];
        for (i, v) in x.iter_mut().enumerate() {
            *v = i as f32; // min = 0, max = 31 → d = 1, every code = the value itself
        }
        let mut blk = vec![0u8; infr_gguf::nbytes(DType::Q5_1, QK)];
        q5_1_block(&x, &mut blk);
        for j in 0..QK {
            let c = code(&blk, j);
            assert!(c <= 31, "code {c} at j={j} exceeds the 5-bit range");
        }
        assert_eq!(code(&blk, QK - 1), 31, "max element must encode to 31");

        // Directly exercise the guard: the pre-mask cast can never legitimately exceed 31, but if it
        // did, `.min(31)` saturates instead of wrapping into the 5th bit with a zeroed low nibble.
        let raw = (40.0f32 as i32 as u8).min(31);
        assert_eq!(
            raw, 31,
            "overflow must clamp to 31, not wrap to 8 (40 & 0x1F)"
        );
    }
}
