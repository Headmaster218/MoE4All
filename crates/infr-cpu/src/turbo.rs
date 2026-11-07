//! TurboQuant KV-cache quantization (`turbo2` / `turbo3` / `turbo4`) — a from-scratch port of
//! llama.cpp's `GGML_TYPE_TURBO{2,3,4}_0` (`ggml/src/ggml-turbo-quant.c`), bit-compatible layouts.
//!
//! All three share ONE pipeline over each 128-element group (a head_dim slice): L2-normalize →
//! forward Walsh–Hadamard Transform (fixed ±1 sign flips) → nearest optimal-centroid scalar quant →
//! pack; store `norm = grp_norm / ‖reconstructed centroids‖`. They differ only in bit width /
//! centroid table / packing:
//!
//! | fmt | bits | centroids | block (128 elems) | packing |
//! |-----|------|-----------|-------------------|---------|
//! | turbo2 | 2 | 4  | 34 B = norm(2)+qs[32]           | 4 idx/byte in qs |
//! | turbo3 | 3 | 8  | 50 B = norm(2)+qs[32]+signs[16] | low2 in qs, bit2 in signs |
//! | turbo4 | 4 | 16 | 66 B = norm(2)+qs[64]           | nibble (2 idx/byte) in qs |
//!
//! Dequant returns the values in the **rotated (WHT) domain**; the CPU reference then applies the
//! inverse WHT ([`dequant_prefix_orig`]) to recover the original domain so the existing f32 SDPA runs
//! unchanged (the fused rotate-Q approach is the GPU optimization). NOTE: the reference's struct byte
//! comments are STALE (32-elem era); the real sizes come from `QK_TURBO*=128` + the static_asserts.

use half::f16;
use infr_core::tensor::DType;

const QK: usize = 128; // elements per block / WHT group

/// 1/sqrt(128) — the WHT normalization for a 128-element group.
const INV_SQRT: f32 = 0.088_388_35;

// Optimal (Lloyd-Max, N(0,1/128)) centroids + the midpoint thresholds between them, per bit width.
const C2: [f32; 4] = [-0.133462, -0.039994, 0.039994, 0.133462];
const M2: [f32; 3] = [-0.086728, 0.0, 0.086728];
const C3: [f32; 8] = [
    -0.190207, -0.118786, -0.066822, -0.021663, 0.021663, 0.066822, 0.118786, 0.190207,
];
const M3: [f32; 7] = [
    -0.154496, -0.092804, -0.044243, 0.0, 0.044243, 0.092804, 0.154496,
];
const C4: [f32; 16] = [
    -0.241529, -0.182877, -0.143016, -0.111036, -0.083292, -0.058050, -0.034299, -0.011349,
    0.011349, 0.034299, 0.058050, 0.083292, 0.111036, 0.143016, 0.182877, 0.241529,
];
const M4: [f32; 15] = [
    -0.212203, -0.162947, -0.127026, -0.097164, -0.070671, -0.046174, -0.022824, 0.0, 0.022824,
    0.046174, 0.070671, 0.097164, 0.127026, 0.162947, 0.212203,
];

// Fixed ±1 diagonal sign vectors for the randomized WHT (llama.cpp `turbo_cpu_s1`/`s2`, extracted
// verbatim — shared by all turbo widths). Stored as i8; applied as f32.
const S1: [i8; 128] = [
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1,
    1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1,
    -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1,
    -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
    -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1,
];
const S2: [i8; 128] = [
    1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1,
    1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
    1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
];

/// Per-format codec: bit width, block byte size, centroid + midpoint tables.
struct Codec {
    bits: u32,
    block_bytes: usize,
    centroids: &'static [f32],
    mids: &'static [f32],
}

const TURBO2: Codec = Codec {
    bits: 2,
    block_bytes: 34,
    centroids: &C2,
    mids: &M2,
};
const TURBO3: Codec = Codec {
    bits: 3,
    block_bytes: 50,
    centroids: &C3,
    mids: &M3,
};
const TURBO4: Codec = Codec {
    bits: 4,
    block_bytes: 66,
    centroids: &C4,
    mids: &M4,
};

/// The codec for a turbo KV dtype (panics on a non-turbo dtype — callers gate on the DType).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn codec(dt: DType) -> &'static Codec {
    match dt {
        DType::Turbo2 => &TURBO2,
        DType::Turbo3 => &TURBO3,
        DType::Turbo4 => &TURBO4,
        _ => unreachable!("turbo codec for non-turbo dtype {dt:?}"),
    }
}

/// Bytes per 128-element turbo block for `dt`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn block_bytes(dt: DType) -> usize {
    codec(dt).block_bytes
}

/// Nearest centroid index via the ascending midpoint thresholds (matches the reference if-chains).
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn nearest(mids: &[f32], v: f32) -> usize {
    for (i, &m) in mids.iter().enumerate() {
        if v < m {
            return i;
        }
    }
    mids.len()
}

/// In-place Hadamard butterfly (unnormalized), shared by the forward/inverse WHT.
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn butterfly(x: &mut [f32; QK]) {
    let mut h = 1;
    while h < QK {
        let mut i = 0;
        while i < QK {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Forward WHT: `y = D(s2)·N·H·D(s1)·x` (N = 1/√128).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn fwht(x: &mut [f32; QK]) {
    for i in 0..QK {
        x[i] *= S1[i] as f32;
    }
    butterfly(x);
    for i in 0..QK {
        x[i] *= INV_SQRT * S2[i] as f32;
    }
}

/// Inverse WHT: swap the sign diagonals (`H·N` self-inverse; s1/s2 are ±1 diagonals).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn fwht_inverse(x: &mut [f32; QK]) {
    for i in 0..QK {
        x[i] *= S2[i] as f32;
    }
    butterfly(x);
    for i in 0..QK {
        x[i] *= INV_SQRT * S1[i] as f32;
    }
}

/// Pack the quantized indices of a rotated group into `body` (the block after the 2-byte norm),
/// returning `Σ centroid²` for the norm correction. Packing depends on bit width.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn pack(c: &Codec, buf: &[f32; QK], body: &mut [u8]) -> f32 {
    body.fill(0);
    let mut recon_sq = 0f32;
    match c.bits {
        2 => {
            for j in 0..QK {
                let idx = nearest(c.mids, buf[j]);
                body[j / 4] |= (idx as u8) << ((j % 4) * 2);
                recon_sq += c.centroids[idx] * c.centroids[idx];
            }
        }
        3 => {
            let (qs, signs) = body.split_at_mut(QK / 4); // 32 | 16
            for j in 0..QK {
                let idx = nearest(c.mids, buf[j]);
                qs[j / 4] |= ((idx & 0x3) as u8) << ((j % 4) * 2);
                if idx & 0x4 != 0 {
                    signs[j / 8] |= 1u8 << (j % 8);
                }
                recon_sq += c.centroids[idx] * c.centroids[idx];
            }
        }
        4 => {
            for j in 0..QK {
                let idx = nearest(c.mids, buf[j]);
                body[j / 2] |= (idx as u8) << ((j % 2) * 4);
                recon_sq += c.centroids[idx] * c.centroids[idx];
            }
        }
        _ => unreachable!(),
    }
    recon_sq
}

/// Unpack a block into the ROTATED (WHT) domain: `centroid[idx]·norm`, no inverse WHT.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn unpack_rot(c: &Codec, blk: &[u8], out: &mut [f32; QK]) {
    let norm = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
    let body = &blk[2..];
    match c.bits {
        2 => {
            for j in 0..QK {
                let idx = ((body[j / 4] >> ((j % 4) * 2)) & 0x3) as usize;
                out[j] = c.centroids[idx] * norm;
            }
        }
        3 => {
            let (qs, signs) = body.split_at(QK / 4);
            for j in 0..QK {
                let low2 = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                let hi1 = (signs[j / 8] >> (j % 8)) & 0x1;
                out[j] = c.centroids[(low2 | (hi1 << 2)) as usize] * norm;
            }
        }
        4 => {
            for j in 0..QK {
                let idx = ((body[j / 2] >> ((j % 2) * 4)) & 0xF) as usize;
                out[j] = c.centroids[idx] * norm;
            }
        }
        _ => unreachable!(),
    }
}

/// Quantize one 128-element group into a turbo block of `dt` (`dst.len() >= block_bytes(dt)`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn quantize_block(dt: DType, src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), QK);
    let c = codec(dt);
    let mut buf = [0f32; QK];
    let mut norm_sq = 0f32;
    for j in 0..QK {
        buf[j] = src[j];
        norm_sq += buf[j] * buf[j];
    }
    let grp_norm = norm_sq.sqrt();
    let inv = if grp_norm > 1e-10 {
        1.0 / grp_norm
    } else {
        0.0
    };
    for v in buf.iter_mut() {
        *v *= inv;
    }
    fwht(&mut buf);

    let (norm_b, body) = dst[..c.block_bytes].split_at_mut(2);
    let recon_sq = pack(c, &buf, body);
    // Norm correction: centroid·norm reproduces the original magnitude on average.
    let recon_norm = recon_sq.sqrt();
    let corrected = if recon_norm > 1e-10 {
        grp_norm / recon_norm
    } else {
        grp_norm
    };
    norm_b.copy_from_slice(&f16::from_f32(corrected).to_bits().to_le_bytes());
}

/// Dequant the first `need` elements (a multiple of 128) to the ORIGINAL domain (rotated dequant +
/// inverse WHT), so the CPU reference attention reads normal-domain K/V. Mirrors
/// [`crate::dequant_prefix_q8_0`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn dequant_prefix_orig(dt: DType, bytes: &[u8], need: usize) -> Vec<f32> {
    let c = codec(dt);
    let nb = need / QK;
    let mut out = vec![0f32; need];
    for b in 0..nb {
        let blk = &bytes[b * c.block_bytes..(b + 1) * c.block_bytes];
        let mut v = [0f32; QK];
        unpack_rot(c, blk, &mut v);
        fwht_inverse(&mut v);
        out[b * QK..(b + 1) * QK].copy_from_slice(&v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_roundtrip_identity() {
        let mut x = [0f32; QK];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i as f32) - 64.0) * 0.013;
        }
        let orig = x;
        fwht(&mut x);
        fwht_inverse(&mut x);
        for i in 0..QK {
            assert!(
                (x[i] - orig[i]).abs() < 1e-4,
                "i={i}: {} vs {}",
                x[i],
                orig[i]
            );
        }
    }

    // Every width recovers a Gaussian vector; error shrinks with bits (2 > 3 > 4).
    #[test]
    fn quantize_dequant_widths() {
        let mut s: u64 = 0x1234_5678;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };
        let mut src = [0f32; QK];
        for v in src.iter_mut() {
            *v = rnd() + rnd() + rnd();
        }
        let mut prev = f32::INFINITY;
        for (dt, tol) in [
            (DType::Turbo2, 0.40),
            (DType::Turbo3, 0.25),
            (DType::Turbo4, 0.15),
        ] {
            let mut blk = vec![0u8; block_bytes(dt)];
            quantize_block(dt, &src, &mut blk);
            let got = dequant_prefix_orig(dt, &blk, QK);
            let (mut num, mut den) = (0f32, 0f32);
            for i in 0..QK {
                num += (got[i] - src[i]).powi(2);
                den += src[i].powi(2);
            }
            let rel = (num / den).sqrt();
            assert!(rel < tol, "{dt:?} rel L2 err {rel} >= {tol}");
            assert!(
                rel <= prev + 0.02,
                "{dt:?} err {rel} not <= wider width {prev}"
            );
            prev = rel;
        }
    }

    // Heavy-tailed vector (one dominant component) stays quantizable thanks to the randomized WHT.
    #[test]
    fn quantize_dequant_heavy_tailed() {
        let mut s: u64 = 0xBEEF;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let mut worst = 0f32;
        for trial in 0..16 {
            let mut src = [0f32; QK];
            for v in src.iter_mut() {
                *v = rnd() * 3.0;
            }
            src[trial * 7 % QK] = 500.0;
            let mut blk = [0u8; 50];
            quantize_block(DType::Turbo3, &src, &mut blk);
            let got = dequant_prefix_orig(DType::Turbo3, &blk, QK);
            let (mut num, mut den) = (0f32, 0f32);
            for i in 0..QK {
                num += (got[i] - src[i]).powi(2);
                den += src[i].powi(2);
            }
            worst = worst.max((num / den).sqrt());
        }
        assert!(
            worst < 0.25,
            "RHT failed to bound heavy-tailed error: {worst}"
        );
    }
}
