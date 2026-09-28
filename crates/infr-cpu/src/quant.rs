//! Activation quantization: Q8 (256-superblock) and Q8x32 (native 32-block) int8 formats,
//! plus small SIMD horizontal-reduction helpers shared by the kernels.

/// Activation quantized to Q8 over 256-element super-blocks: `qs[i] = round(x[i]/d[blk])` (int8),
/// `d[blk] = max|x|/127`. Quantize the activation ONCE per matvec, then integer-dot it against the
/// quantized weight rows (llama.cpp's q8_K path) — no per-row f32 weight expansion.
#[derive(Clone)]
pub(crate) struct Q8 {
    pub(crate) qs: Vec<i8>,
    pub(crate) d: Vec<f32>,
    /// Sub-block sums: `bsums[b*8+s]` = Σ `qs[b*256 + s*32 .. +32]` as i32.
    /// One entry per 32-element sub-block (8 per 256-element super-block).
    /// Precomputed once at quantization time; reused across all weight rows so the
    /// `sm` accumulation in `vec_dot_q4k` (Σ m·Σq8) avoids O(rows·256) re-summation.
    /// Mirrors llama.cpp's `block_q8_K.bsums`.
    pub(crate) bsums: Vec<i32>,
    /// Per-16-element sub-block sums (`bsums16[b*16+s]` = Σ `qs[b*256 + s*16 .. +16]`) — Q6_K's
    /// `-32` offset correction is per 16-element scale group. Precomputed here for the same
    /// reason as `bsums`: `vec_dot_q6k_batch` used to re-derive these sums with SIMD for EVERY
    /// weight row (262k redundant recomputes per lm_head GEMM — ~22% of a DG denoise step).
    /// Only the x86 SIMD kernel reads it — dead code on aarch64 (macOS CI builds with -D warnings).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) bsums16: Vec<i32>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quantize_q8(x: &[f32]) -> Q8 {
    let nb = x.len() / 256;
    let mut qs = vec![0i8; nb * 256];
    let mut d = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * 256..b * 256 + 256];
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        d[b] = dd;
        for (i, &v) in blk.iter().enumerate() {
            qs[b * 256 + i] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
    }
    // Precompute per-16- and per-32-elem sub-block sums (Q6_K offset term / Q4_K min-scale term).
    let mut bsums16 = vec![0i32; nb * 16];
    for b in 0..nb {
        for s in 0..16usize {
            bsums16[b * 16 + s] = qs[b * 256 + s * 16..b * 256 + s * 16 + 16]
                .iter()
                .map(|&q| q as i32)
                .sum();
        }
    }
    let bsums = bsums16
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| p[0] + p[1])
        .collect();
    Q8 {
        qs,
        d,
        bsums,
        bsums16,
    }
}

/// Horizontal reduction: sum 8 × i32 in a ymm register to a scalar i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn hadd_i32_ymm(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let h1 = _mm256_hadd_epi32(v, v);
    let h2 = _mm256_hadd_epi32(h1, h1);
    let lo = _mm256_castsi256_si128(h2);
    let hi = _mm256_extracti128_si256::<1>(h2);
    _mm_cvtsi128_si32(_mm_add_epi32(lo, hi))
}

/// Horizontal reduction: sum 4 × i32 in an xmm register to a scalar i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn hadd_i32_xmm(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let h = _mm_hadd_epi32(v, v); // [a+b, c+d, a+b, c+d]
    let hh = _mm_hadd_epi32(h, h); // [a+b+c+d, ...]
    _mm_cvtsi128_si32(hh)
}

/// Activation quantized to int8 per NATIVE 32-element block (mirrors [`Q8`] but without the
/// 256-superblock grouping). `bsum` is `Σqs` per block — Q5_0's constant `-16` offset needs `Σx`,
/// which `d[b] * bsum[b]` approximates the same way `Q8::bsums` does for the K-quant min term.
#[derive(Clone)]
pub(crate) struct Q8x32 {
    pub(crate) qs: Vec<i8>,
    pub(crate) d: Vec<f32>,
    pub(crate) bsum: Vec<i32>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quantize_q8_32(x: &[f32]) -> Q8x32 {
    let nb = x.len() / 32;
    let mut qs = vec![0i8; nb * 32];
    let mut d = vec![0f32; nb];
    let mut bsum = vec![0i32; nb];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        d[b] = dd;
        let mut s = 0i32;
        for (i, &v) in blk.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            qs[b * 32 + i] = q;
            s += q as i32;
        }
        bsum[b] = s;
    }
    Q8x32 { qs, d, bsum }
}
