//! Per-format vec-dot kernels (scalar/AVX2/AVX-512 + dispatchers): Q4_K, Q6_K, Q8_0, Q5_K
//! single-row and batch families, the Q8_0_32/Q5_0_32 native-block family, and the plain
//! f32/f16/bf16 dots + gated-FFN activation.
#[cfg(target_arch = "x86_64")]
use crate::quant::{hadd_i32_xmm, hadd_i32_ymm};
use crate::quant::{Q8x32, Q8};
use infr_core::graph::Activation;
use infr_gguf::dequant::{
    e8m0_to_fp32_half, k4, rdf16, ue4m3_to_fp32, KVALUES_IQ4NL, KVALUES_MXFP4,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm256_loadu_si256, _mm512_loadu_si512, _mm_loadu_si128,
};

/// Load the 32 bytes starting at ELEMENT offset `off` of `s` into a ymm register.
///
/// WHY THIS EXISTS. Every wide load in this file used to be written inline as
/// `_mm256_loadu_si256(s[off..].as_ptr() as *const __m256i)`, and the runtime dispatchers above
/// them carry the comment "pointer bounds checked by slice indexing". Only half of that is true.
/// The feature half is real — `is_x86_feature_detected!` gates the call. The bounds half is FALSE:
/// `s[off..]` is a `RangeFrom` index, so it panics only when the START of the load is out of
/// bounds. It says nothing whatsoever about the 32 bytes the intrinsic then reads past that point.
/// The slice type is erased into a raw pointer one token later, and from there the load width is
/// invisible to the compiler.
///
/// The loads are all in bounds today, but only because the block geometry happens to work out
/// exactly, with ZERO slack. Q6_K is the sharpest example: its `qh` sub-slice is exactly 64 bytes,
/// and the `half == 1` load reads `qh[32..]` — precisely the last 32 bytes, not one to spare. Any
/// future change to a block layout (a header field added, a sub-block count changed, a format
/// packed differently) turns that into a read of the NEXT block's bytes, which produces silently
/// wrong dot products with no crash and no test failure outside the parity suite; or, on the last
/// block of a tensor backed by an mmap, a read past the mapping and a SIGSEGV. Nothing in the old
/// expression could catch either case.
///
/// Routing every load through this helper makes the geometry a CHECKED invariant instead of an
/// unwritten one: the `debug_assert!` fires in debug and test builds — which is where the whole
/// per-format kernel parity suite runs — the moment a load would read past the end of the slice it
/// was handed, and it compiles to nothing in release. The `s[off..]` indexing is deliberately kept
/// verbatim rather than replaced with `add(off)`, so the release-mode start-of-load bounds check,
/// and therefore the emitted code, is byte-for-byte what it was before this helper existed.
///
/// `off` counts ELEMENTS of `T`, exactly matching the index that was already written at the call
/// site, so no call site had to change its arithmetic. The width check is done in bytes because
/// the bases in this file are not all `&[u8]`: activations are `&[i8]` and Q6_K's per-16 sub-block
/// sums are `&[i32]`, where 32 bytes is 32 elements and 8 elements respectively.
///
/// `#[inline]` rather than `#[inline(always)]`: rustc rejects `#[inline(always)]` on a
/// `#[target_feature]` function outright (rust-lang/rust#145574), and `#[target_feature]` is the
/// part that actually matters here — without it the avx-gated intrinsic cannot be inlined into
/// this wrapper and every load would degrade into a real call. With it, the callee feature set is
/// a subset of every caller's (`avx` ⊆ `avx2` ⊆ `avx512bw`), so LLVM inlines it to the bare
/// `vmovups` at each site.
///
/// # Safety
/// `avx` must be available. Every caller is inside an `avx2`/`avx512*` `#[target_feature]`
/// function, all of which imply it.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn load256<T>(s: &[T], off: usize) -> __m256i {
    debug_assert!(
        off * size_of::<T>() + 32 <= size_of_val(s),
        "load256 reads out of bounds: offset {off} elems ({} bytes) + 32 bytes > slice of {} elems \
         ({} bytes) — block geometry changed and the load width no longer fits",
        off * size_of::<T>(),
        s.len(),
        size_of_val(s)
    );
    _mm256_loadu_si256(s[off..].as_ptr() as *const __m256i)
}

/// Load the 64 bytes starting at ELEMENT offset `off` of `s` into a zmm register.
///
/// The 512-bit twin of [`load256`] — see that function for the full rationale. The failure mode is
/// the same and strictly worse here: a `RangeFrom` slice index bounds only the first byte, while
/// this intrinsic reads SIXTY-FOUR, so a layout change that leaves a block one 32-byte chunk short
/// silently pulls in the following block (wrong output, no crash) or walks off the end of the
/// tensor mapping (SIGSEGV). The `debug_assert!` turns that from an invisible geometry assumption
/// into a test-suite failure at the exact load that broke.
///
/// # Safety
/// `avx512f` must be available. Every caller is inside an `avx512bw` `#[target_feature]` function,
/// which implies it.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load512<T>(s: &[T], off: usize) -> __m512i {
    debug_assert!(
        off * size_of::<T>() + 64 <= size_of_val(s),
        "load512 reads out of bounds: offset {off} elems ({} bytes) + 64 bytes > slice of {} elems \
         ({} bytes) — block geometry changed and the load width no longer fits",
        off * size_of::<T>(),
        s.len(),
        size_of_val(s)
    );
    _mm512_loadu_si512(s[off..].as_ptr() as *const __m512i)
}

/// Load the 16 bytes starting at ELEMENT offset `off` of `s` into an xmm register.
///
/// The 128-bit twin of [`load256`] — see that function for the full rationale. The failure mode is
/// identical: a `RangeFrom` slice index bounds only the first byte while the intrinsic reads
/// SIXTEEN, so the load is correct by block geometry alone and a layout change breaks it silently.
///
/// Several call sites read a fixed-size codebook (`KVALUES_IQ4NL` / `KVALUES_MXFP4`, both
/// `[i8; 16]`) where the bound is trivially exact; they go through the helper anyway so the
/// invariant is uniform across every wide load in this file and a future codebook resize cannot
/// quietly widen the read.
///
/// # Safety
/// `sse2` must be available. It is part of the x86_64 baseline, and every caller is additionally
/// inside an `avx2`/`avx512*` `#[target_feature]` function, all of which imply it.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn load128<T>(s: &[T], off: usize) -> __m128i {
    debug_assert!(
        off * size_of::<T>() + 16 <= size_of_val(s),
        "load128 reads out of bounds: offset {off} elems ({} bytes) + 16 bytes > slice of {} elems \
         ({} bytes) — block geometry changed and the load width no longer fits",
        off * size_of::<T>(),
        s.len(),
        size_of_val(s)
    );
    _mm_loadu_si128(s[off..].as_ptr() as *const __m128i)
}

/// `Σ weight·x` for one Q4_K row (144 bytes / 256 elems) against the Q8 activation. Weight value is
/// `d·sc_s·q4 − dmin·m_s` over 8 sub-blocks of 32; dispatches to the best SIMD path available at
/// runtime (avx512bw → avx2 → scalar). The `sm` term uses `q8.bsums` (precomputed in `quantize_q8`)
/// instead of re-summing q8 values per row.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q4k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(in_f % 256, 0, "vec_dot_q4k: in_f must be a multiple of 256");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q4k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q4k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q4k_scalar(row, q8, in_f)
}

/// Scalar fallback for `vec_dot_q4k`; also used on non-x86 targets. Uses `q8.bsums` for `isum`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q4k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        for s in 0..8usize {
            let (sc, m) = k4(s, scales);
            let (half, hi) = (s / 2, s % 2 == 1);
            let qbyte = &qs[half * 32..half * 32 + 32];
            let q8s = &q8b[s * 32..s * 32 + 32];
            let mut iprod = 0i32;
            for l in 0..32 {
                let q4 = if hi {
                    (qbyte[l] >> 4) as i32
                } else {
                    (qbyte[l] & 0xF) as i32
                };
                iprod += q4 * q8s[l] as i32;
            }
            // isum = Σ q8s — precomputed in Q8::bsums, avoids re-summing per weight row.
            let isum = q8.bsums[b * 8 + s];
            sd += sc as i32 * iprod;
            sm += m as i32 * isum;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q4k`: one 32-element sub-block per iteration with 256-bit SIMD.
/// Nibbles are unpacked with `_mm256_maddubs_epi16` (unsigned×signed → i16 pair-sum) then widened
/// to i32 via `_mm256_madd_epi16`. `isum` comes from `q8.bsums`, not the inner loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        for s in 0..8usize {
            let (sc, m) = k4(s, scales);
            let hi = s % 2 == 1;
            let half = s / 2;
            // 32-byte nibble chunk shared by sub-blocks `2*half` (lo) and `2*half+1` (hi).
            let nibbles = load256(qs, half * 32);
            // Unpack nibbles: low or high 4 bits of each byte → u8 values 0–15.
            let q4 = if hi {
                _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo)
            } else {
                _mm256_and_si256(nibbles, mask_lo)
            };
            // Load 32 signed Q8 bytes for this sub-block.
            let q8v = load256(q8b, s * 32);
            // maddubs: a=u8 (q4, 0–15), b=i8 (q8) → 16×i16 pair-sums. No i16 overflow:
            // max pair = 15·127 + 15·127 = 3810 < 32767.
            let prod = _mm256_maddubs_epi16(q4, q8v);
            // madd with 1: widen 16×i16 → 8×i32 (pairs summed).
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            let isum = q8.bsums[b * 8 + s];
            sd += sc as i32 * iprod;
            sm += m as i32 * isum;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q4k`: processes TWO sub-blocks per iteration (64 elements) with
/// 512-bit SIMD. For each pair (s=2k even, s=2k+1 odd), the 32-byte nibble chunk is unpacked into
/// a zmm (low nibbles in lower 256 bits, high nibbles in upper 256 bits) and multiplied against the
/// corresponding 64 contiguous Q8 bytes. The zmm result is split back to two ymm sums, giving both
/// `iprod_even` and `iprod_odd` in one pass — half the memory traffic of the avx2 path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        // k=0..4 covers sub-block pairs (0,1), (2,3), (4,5), (6,7).
        for k in 0..4usize {
            let (sc_e, m_e) = k4(2 * k, scales);
            let (sc_o, m_o) = k4(2 * k + 1, scales);
            // Load 32 nibble bytes serving both sub-blocks 2k (low) and 2k+1 (high).
            let nibbles = load256(qs, k * 32);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo); // sub-block 2k  (u8, 0–15)
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo); // sub-block 2k+1
                                                                                   // Pack into zmm: lower 256 = lo_nib (for 2k), upper 256 = hi_nib (for 2k+1).
            let q4_zmm = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo_nib), hi_nib);
            // Load 64 Q8 bytes: [2k*32..(2k+1)*32] in lower, [(2k+1)*32..(2k+2)*32] in upper.
            let q8_zmm = load512(q8b, k * 64);
            // maddubs 512-bit: u8(q4)×i8(q8) → 32×i16 pair-sums.
            let prod = _mm512_maddubs_epi16(q4_zmm, q8_zmm);
            // madd with 1: widen 32×i16 → 16×i32.
            let sum32 = _mm512_madd_epi16(prod, ones512);
            // Lower 256 = 8×i32 for sub-block 2k; upper 256 = 8×i32 for sub-block 2k+1.
            let lo_ymm = _mm512_castsi512_si256(sum32);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
            let iprod_e = hadd_i32_ymm(lo_ymm);
            let iprod_o = hadd_i32_ymm(hi_ymm);
            let isum_e = q8.bsums[b * 8 + 2 * k];
            let isum_o = q8.bsums[b * 8 + 2 * k + 1];
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// `Σ weight·x` for one Q2_K row (84 bytes / 256 elems) against the Q8 activation. Q2_K is an
/// AFFINE k-quant like Q4_K — weight value is `d·scale·q2 − dmin·min` — but with 2-bit codes, a
/// 4-bit unsigned scale AND 4-bit unsigned min per PER-16-element sub-block (16 of them, vs Q4_K's
/// 8 per-32). Same `Σ scale·(Σ q2·qa) − min·(Σ qa)` shape as Q4_K; the min correction uses the
/// activation's per-16 sub-block sums (`q8.bsums16`, not the per-32 `bsums` Q4_K uses). Dispatches
/// avx512bw → avx2 → scalar.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q2k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(in_f % 256, 0, "vec_dot_q2k: in_f must be a multiple of 256");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q2k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q2k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q2k_scalar(row, q8, in_f)
}

/// Scalar fallback for `vec_dot_q2k`; also used on non-x86 targets. Traverses the 16 per-16-element
/// sub-blocks in the EXACT order `dequant_block(DType::Q2K)` does (`n`×`shift`×`half`), so element
/// `is*16 + l` maps to the same `(sub-block, code)` the dequant reference produces. Uses
/// `q8.bsums16` for the `sm` (min) correction.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q2k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 84..b * 84 + 84];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = rdf16(&blk[80..82]);
        let dmin = rdf16(&blk[82..84]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        // 16 sub-blocks of 16, consumed in the dequant reference's traversal order.
        let mut is = 0usize;
        let mut qoff = 0usize;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                for half in 0..2 {
                    let sc = scales[is];
                    let scale = (sc & 0xF) as i32;
                    let minv = (sc >> 4) as i32;
                    let qbyte = &qs[qoff + half * 16..qoff + half * 16 + 16];
                    let q8s = &q8b[is * 16..is * 16 + 16];
                    let mut iprod = 0i32;
                    for l in 0..16 {
                        let q2 = ((qbyte[l] >> shift) & 3) as i32;
                        iprod += q2 * q8s[l] as i32;
                    }
                    sd += scale * iprod;
                    sm += minv * q8.bsums16[b * 16 + is];
                    is += 1;
                }
                shift += 2;
            }
            qoff += 32;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q2k`: one 32-byte `qs` chunk (per `n` half) serves all 4 shifts. For
/// shift-index `k` the byte-wise `(qs >> 2k) & 3` (running 16-bit shift + mask, same cross-byte-safe
/// trick Q6_K uses) aligns EXACTLY with `q8b[n*128 + k*32 .. +32]` — 32 elements = sub-block
/// `8n+2k` (low 16) + `8n+2k+1` (high 16). `maddubs(q2_u8, q8_i8) → madd(1)` then split the ymm into
/// two 16-elem sub-block dots. The `sm` (min) term is an order-free i32 sum over `q8.bsums16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q2k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_03 = _mm256_set1_epi8(0x03_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 84..b * 84 + 84];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = rdf16(&blk[80..82]);
        let dmin = rdf16(&blk[82..84]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut sd = 0i32;
        for n in 0..2usize {
            // 32 code bytes shared by the 4 shifts (sub-blocks 8n..8n+8).
            let mut cur = load256(qs, n * 32);
            for k in 0..4usize {
                // (cur & 3) = the 2-bit codes at the current shift, one per byte (0–3).
                let q2 = _mm256_and_si256(cur, mask_03);
                let q8v = load256(q8b, n * 128 + k * 32);
                // maddubs: u8(0–3) × i8 → 16×i16 pair-sums (max |pair| = 2·3·127 = 762, no overflow).
                let prod = _mm256_maddubs_epi16(q2, q8v);
                let sum32 = _mm256_madd_epi16(prod, ones);
                // Lower 128 = elements 0..15 (sub-block 8n+2k); upper 128 = 16..31 (sub-block 8n+2k+1).
                let iprod_lo = hadd_i32_xmm(_mm256_castsi256_si128(sum32));
                let iprod_hi = hadd_i32_xmm(_mm256_extracti128_si256::<1>(sum32));
                let is0 = 8 * n + 2 * k;
                sd += (scales[is0] & 0xF) as i32 * iprod_lo
                    + (scales[is0 + 1] & 0xF) as i32 * iprod_hi;
                // Advance to the next 2-bit field. 16-bit lane shift is safe: masking with 0x03
                // discards any cross-byte bleed (see the Q6_K `qh_sr2` note).
                cur = _mm256_srli_epi16(cur, 2);
            }
        }
        // sm (min correction) — order-free i32 sum over the per-16 activation sub-block sums.
        let mut sm = 0i32;
        for is in 0..16usize {
            sm += (scales[is] >> 4) as i32 * q8.bsums16[b * 16 + is];
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q2k`: both `n` halves' 32-byte `qs` chunks are contiguous (64 B),
/// so one zmm holds them. For each shift-index `k`, `(qs_zmm >> 2k) & 3` yields 64 codes — lower 256
/// = `n=0` (sub-blocks `2k`,`2k+1`), upper 256 = `n=1` (`8+2k`,`8+2k+1`). The matching q8 bytes
/// (`q8b[k*32..]` for `n=0`, `q8b[128+k*32..]` for `n=1`) are non-contiguous → 2 ymm loads + insert.
/// One `maddubs512 → madd512` covers 64 elements; the 16×i32 result splits into four 16-elem dots.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q2k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_03 = _mm512_set1_epi8(0x03_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 84..b * 84 + 84];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = rdf16(&blk[80..82]);
        let dmin = rdf16(&blk[82..84]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        // qs[0..64] = both n halves contiguous → one zmm.
        let mut cur = load512(qs, 0);
        let mut sd = 0i32;
        for k in 0..4usize {
            let q2 = _mm512_and_si512(cur, mask_03);
            // q8 for n=0 (lower 256) and n=1 (upper 256), 32 bytes each, non-contiguous.
            let q8_lo = load256(q8b, k * 32);
            let q8_hi = load256(q8b, 128 + k * 32);
            let q8_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q8_lo), q8_hi);
            let prod = _mm512_maddubs_epi16(q2, q8_z);
            let sum32 = _mm512_madd_epi16(prod, ones512);
            // Lower ymm = n=0 (sub-blocks 2k, 2k+1); upper ymm = n=1 (sub-blocks 8+2k, 8+2k+1).
            let lo_ymm = _mm512_castsi512_si256(sum32);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
            let iprod_n0_0 = hadd_i32_xmm(_mm256_castsi256_si128(lo_ymm));
            let iprod_n0_1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(lo_ymm));
            let iprod_n1_0 = hadd_i32_xmm(_mm256_castsi256_si128(hi_ymm));
            let iprod_n1_1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(hi_ymm));
            let is0 = 2 * k; // n=0
            let is1 = 8 + 2 * k; // n=1
            sd += (scales[is0] & 0xF) as i32 * iprod_n0_0
                + (scales[is0 + 1] & 0xF) as i32 * iprod_n0_1
                + (scales[is1] & 0xF) as i32 * iprod_n1_0
                + (scales[is1 + 1] & 0xF) as i32 * iprod_n1_1;
            cur = _mm512_srli_epi16(cur, 2);
        }
        let mut sm = 0i32;
        for is in 0..16usize {
            sm += (scales[is] >> 4) as i32 * q8.bsums16[b * 16 + is];
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// `Σ weight·x` for one Q6_K row (210 bytes / 256 elems). Dispatches to the best SIMD path
/// available at runtime (avx512bw → avx2 → scalar). Weight value is `d·sc·(q6−32)` over 16
/// sub-blocks of 16 (int8 scales).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q6k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(in_f % 256, 0, "vec_dot_q6k: in_f must be a multiple of 256");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q6k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q6k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q6k_scalar(row, q8, in_f)
}

/// Scalar fallback for `vec_dot_q6k`; also used on non-x86 targets.
/// Accumulates `Σ q6·q8` and `Σ q8` per 16-element sub-block, then applies int8 scales.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q6k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = rdf16(&blk[208..210]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut sumi = [0i32; 16];
        let mut bsum = [0i32; 16];
        for half in 0..2 {
            let (qlo, qho, sco, base) = (half * 64, half * 32, half * 8, half * 128);
            for l in 0..32 {
                let is = l / 16;
                let q1 = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                let q2 = (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                let q3 = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                let q4 = (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                for (off, q, sci) in [(0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)] {
                    let sub = sco + is + sci;
                    let v = q8b[base + l + off] as i32;
                    sumi[sub] += q as i32 * v;
                    bsum[sub] += v;
                }
            }
        }
        // INTEGER epilogue (llama.cpp's `ggml_vec_dot_q6_K_q8_K` shape): the scale-weighted
        // sub-block dots accumulate exactly in i32 (|Σ| < 2^29, no overflow possible), ONE f32
        // multiply per super-block — fewer roundings than the old per-sub f32 chain, and the
        // i32 sum is order-free, so SIMD variants no longer need an order-pinned f32 tail.
        let mut isum = 0i32;
        for sub in 0..16 {
            isum += scales[sub] as i8 as i32 * (sumi[sub] - 32 * bsum[sub]);
        }
        sumf += d * q8.d[b] * isum as f32;
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q6k`: processes each of the 4 "columns" of 32 q6 values per half
/// using 256-bit SIMD. Each column maps to two consecutive 16-element sub-blocks (lower/upper
/// 128 bits after `madd`). q6 is reconstructed from `ql`/`qh` via byte-wise mask+shift; the
/// dot uses `maddubs(q6_u8, q8_i8) → madd(-, 1)`. The −32·bsum correction is computed from a
/// parallel `maddubs(1_u8, q8_i8)` sum so no per-element loop is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q6k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;

    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8); // low nibble
    let mask_30 = _mm256_set1_epi8(0x30_u8 as i8); // bits 4-5
    let mask_03 = _mm256_set1_epi8(0x03_u8 as i8); // low 2 bits
    let ones_u8 = _mm256_set1_epi8(1i8); // for bsum via maddubs(1, q8)
    let ones_i16 = _mm256_set1_epi16(1i16);

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = rdf16(&blk[208..210]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];

        let mut s = 0i32; // integer epilogue — see `vec_dot_q6k_scalar`

        // Process two halves (each 128 elements = 8 sub-blocks of 16).
        for half in 0..2usize {
            let qlo = half * 64;
            let qho = half * 32;
            let sco = half * 8;
            let base = half * 128;

            // Load qh (32 bytes) and the two ql halves for this block-half.
            let qh_ymm = load256(qh, qho);
            let ql_lo = load256(ql, qlo);
            let ql_hi = load256(ql, qlo + 32);

            // (qh >> 2) & 0x03 per byte — reused in col2 and col6 reconstructions.
            // Trick: _mm256_srli_epi16(v, 2) shifts 16-bit lanes; masking with 0x03 per byte
            // gives the correct byte-wise >>2 result for each byte (cross-byte bleed is masked
            // away: (high_byte << 6) & 0x03 = 0 since high_byte << 6 occupies bits 6-7).
            let qh_sr2 = _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_03);

            // Reconstruct 4 × 32 q6 byte columns (values 0–63, stored as u8 in __m256i).
            //
            // col0: (ql_lo & 0x0F) | ((qh & 0x03) << 4)  →  q8b[base..base+32]
            // col2: (ql_hi & 0x0F) | ((qh>>2 & 0x03) << 4) → q8b[base+32..base+64]
            // col4: (ql_lo >> 4) | (qh & 0x30)             → q8b[base+64..base+96]
            // col6: (ql_hi >> 4) | ((qh>>2) & 0x30)        → q8b[base+96..base+128]
            //
            // Left-shift by 4 via _mm256_slli_epi16: for bytes 0–3 in range the low byte
            // result (v << 4) & 0xFF is correct (high byte has no bleed since input ≤ 3).
            let q6_c0 = _mm256_or_si256(
                _mm256_and_si256(ql_lo, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(qh_ymm, mask_03), 4),
            );
            let q6_c2 = _mm256_or_si256(
                _mm256_and_si256(ql_hi, mask_0f),
                _mm256_slli_epi16(qh_sr2, 4),
            );
            // col4: (qh >> 4 & 3) << 4 = qh & 0x30 (bits 4-5 of qh[i]).
            let q6_c4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_0f),
                _mm256_and_si256(qh_ymm, mask_30),
            );
            // col6: (qh >> 6 & 3) << 4 = (qh >> 2) & 0x30 (bits 6-7 of qh[i] → positions 4-5).
            let q6_c6 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_0f),
                _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_30),
            );

            // For each of the 4 columns: dot with the corresponding 32 q8 bytes, then split the
            // 8×i32 ymm result into lower 4×i32 (sub-block `sco+ci*2`) and upper 4×i32
            // (sub-block `sco+ci*2+1`). Reduction order matches scalar sub=0..16, ensuring
            // bit-identical f32 accumulation.
            for (ci, q6_col) in [q6_c0, q6_c2, q6_c4, q6_c6].iter().enumerate() {
                let q8_ymm = load256(q8b, base + ci * 32);

                // maddubs: q6_u8 (0–63) × q8_i8 → 16×i16 pair-sums. Each pair sums two adjacent
                // products, so the bound is 2·63·127 = 16002 (min −16128 at q8 = −128), < 32767.
                // madd with 1: widen 16×i16 → 8×i32 (pairs summed).
                let prod = _mm256_maddubs_epi16(*q6_col, q8_ymm);
                let sum32 = _mm256_madd_epi16(prod, ones_i16);

                // bsum: maddubs(1_u8, q8_i8) = q8 pair-sums as i16; madd → 4-group i32 sums.
                let bsum_i16 = _mm256_maddubs_epi16(ones_u8, q8_ymm);
                let bsum_i32 = _mm256_madd_epi16(bsum_i16, ones_i16);

                // Lower 128 bits = elements 0..15 (sub-block `sco+ci*2`).
                // Upper 128 bits = elements 16..31 (sub-block `sco+ci*2+1`).
                let sum_lo = _mm256_castsi256_si128(sum32);
                let sum_hi = _mm256_extracti128_si256::<1>(sum32);
                let bs_lo = _mm256_castsi256_si128(bsum_i32);
                let bs_hi = _mm256_extracti128_si256::<1>(bsum_i32);

                let iprod_0 = hadd_i32_xmm(sum_lo);
                let iprod_1 = hadd_i32_xmm(sum_hi);
                let bs_0 = hadd_i32_xmm(bs_lo);
                let bs_1 = hadd_i32_xmm(bs_hi);

                let sub_0 = sco + ci * 2;
                let sub_1 = sco + ci * 2 + 1;
                s += scales[sub_0] as i8 as i32 * (iprod_0 - 32 * bs_0);
                s += scales[sub_1] as i8 as i32 * (iprod_1 - 32 * bs_1);
            }
        }
        sumf += d * q8.d[b] * s as f32;
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q6k`: processes BOTH halves of a 256-element block simultaneously
/// using zmm registers (512-bit). The two ql half-lo slices are packed into one zmm via
/// `_mm512_inserti64x4`; qh loads as a single 64-byte zmm (the two halves are contiguous). For
/// each of the 4 q6 columns, both halves' q8 are also packed into one zmm, so a single
/// `maddubs512 → madd512` covers 64 elements at once. Results are split back to two ymm
/// (h0/h1) then two xmm per ymm (per-sub-block) for the scalar scale accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q6k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;

    let mask_0f_z = _mm512_set1_epi8(0x0F_u8 as i8);
    let mask_30_z = _mm512_set1_epi8(0x30_u8 as i8);
    let mask_03_z = _mm512_set1_epi8(0x03_u8 as i8);
    let ones_u8_z = _mm512_set1_epi8(1i8);
    let ones_i16_z = _mm512_set1_epi16(1i16);

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = rdf16(&blk[208..210]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];

        // qh is 64 contiguous bytes: h0 in bits 0-255, h1 in bits 256-511 → one zmm load.
        let qh_z = load512(qh, 0);

        // ql_lo_z: lower 256 = ql[0..32] (h0_lo), upper 256 = ql[64..96] (h1_lo).
        // ql_hi_z: lower 256 = ql[32..64] (h0_hi), upper 256 = ql[96..128] (h1_hi).
        // h0 and h1 slices are non-contiguous (separated by 32 bytes), so 2 ymm loads + insert.
        let ql_lo_h0 = load256(ql, 0);
        let ql_lo_h1 = load256(ql, 64);
        let ql_lo_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_lo_h0), ql_lo_h1);

        let ql_hi_h0 = load256(ql, 32);
        let ql_hi_h1 = load256(ql, 96);
        let ql_hi_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_hi_h0), ql_hi_h1);

        // Same byte-wise shift/mask tricks as AVX2 but on 512-bit registers.
        let qh_sr2_z = _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_03_z);

        let q6_c0_z = _mm512_or_si512(
            _mm512_and_si512(ql_lo_z, mask_0f_z),
            _mm512_slli_epi16(_mm512_and_si512(qh_z, mask_03_z), 4),
        );
        let q6_c2_z = _mm512_or_si512(
            _mm512_and_si512(ql_hi_z, mask_0f_z),
            _mm512_slli_epi16(qh_sr2_z, 4),
        );
        let q6_c4_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_lo_z, 4), mask_0f_z),
            _mm512_and_si512(qh_z, mask_30_z),
        );
        let q6_c6_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_hi_z, 4), mask_0f_z),
            _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_30_z),
        );

        // Collect per-sub-block i32 values in arrays; accumulate in 0..16 order (scalar-identical).
        let mut simd_sumi = [0i32; 16];
        let mut simd_bsum = [0i32; 16];

        for (ci, q6_col_z) in [q6_c0_z, q6_c2_z, q6_c4_z, q6_c6_z].iter().enumerate() {
            // q8 for h0 column ci: q8b[ci*32..ci*32+32]; h1: q8b[128+ci*32..128+ci*32+32].
            let q8_h0 = load256(q8b, ci * 32);
            let q8_h1 = load256(q8b, 128 + ci * 32);
            let q8_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q8_h0), q8_h1);

            // 512-bit dot: maddubs → madd.
            let prod = _mm512_maddubs_epi16(*q6_col_z, q8_z);
            let sum32_z = _mm512_madd_epi16(prod, ones_i16_z);

            let bsum_i16_z = _mm512_maddubs_epi16(ones_u8_z, q8_z);
            let bsum_i32_z = _mm512_madd_epi16(bsum_i16_z, ones_i16_z);

            // Lower ymm = h0 (sub-blocks ci*2, ci*2+1); upper ymm = h1 (sub-blocks 8+ci*2, 8+ci*2+1).
            let sum_h0 = _mm512_castsi512_si256(sum32_z);
            let sum_h1 = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let bsum_h0 = _mm512_castsi512_si256(bsum_i32_z);
            let bsum_h1 = _mm512_extracti64x4_epi64::<1>(bsum_i32_z);

            // Each ymm: lower xmm = first 16 elements (is=0), upper xmm = elements 16..31 (is=1).
            let s_h0_lo = _mm256_castsi256_si128(sum_h0);
            let s_h0_hi = _mm256_extracti128_si256::<1>(sum_h0);
            let s_h1_lo = _mm256_castsi256_si128(sum_h1);
            let s_h1_hi = _mm256_extracti128_si256::<1>(sum_h1);
            let b_h0_lo = _mm256_castsi256_si128(bsum_h0);
            let b_h0_hi = _mm256_extracti128_si256::<1>(bsum_h0);
            let b_h1_lo = _mm256_castsi256_si128(bsum_h1);
            let b_h1_hi = _mm256_extracti128_si256::<1>(bsum_h1);

            simd_sumi[ci * 2] = hadd_i32_xmm(s_h0_lo);
            simd_sumi[ci * 2 + 1] = hadd_i32_xmm(s_h0_hi);
            simd_sumi[8 + ci * 2] = hadd_i32_xmm(s_h1_lo);
            simd_sumi[8 + ci * 2 + 1] = hadd_i32_xmm(s_h1_hi);

            simd_bsum[ci * 2] = hadd_i32_xmm(b_h0_lo);
            simd_bsum[ci * 2 + 1] = hadd_i32_xmm(b_h0_hi);
            simd_bsum[8 + ci * 2] = hadd_i32_xmm(b_h1_lo);
            simd_bsum[8 + ci * 2 + 1] = hadd_i32_xmm(b_h1_hi);
        }

        // Integer epilogue — see `vec_dot_q6k_scalar` (i32 sum is order-free and exact).
        let mut s = 0i32;
        for sub in 0..16 {
            s += scales[sub] as i8 as i32 * (simd_sumi[sub] - 32 * simd_bsum[sub]);
        }
        sumf += d * q8.d[b] * s as f32;
    }
    sumf
}

/// Decode Q3_K's 16 packed 6-bit sub-block scales into signed `(sc6 − 32)` values, indexed by the
/// natural sub-block counter `is` (matching `dequant_block(DType::Q3K)`'s `sc6(is)`). The 12 scale
/// bytes encode 16 × 6-bit fields via llama.cpp's aux bit-shuffle — this is NOT a plain field read;
/// the shuffle is ported EXACTLY from `dequant_block`'s Q3_K case (`ggml-quants.c`).
#[inline]
fn q3k_scales(scales_raw: &[u8]) -> [i16; 16] {
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes(scales_raw[0..4].try_into().unwrap());
    aux[1] = u32::from_le_bytes(scales_raw[4..8].try_into().unwrap());
    aux[2] = u32::from_le_bytes(scales_raw[8..12].try_into().unwrap());
    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let mut sc = [0i16; 16];
    for (i, s) in sc.iter_mut().enumerate() {
        *s = (((aux[i / 4] >> ((i % 4) * 8)) & 0xFF) as u8 as i8) as i16 - 32;
    }
    sc
}

/// `Σ weight·x` for one Q3_K row (110 bytes / 256 elems) against the Q8 activation. Structurally a
/// Q6_K sibling: weight value is `d·(sc6−32)·(q3−4)` over 16 sub-blocks of 16 — a per-16 SIGNED
/// scale `(sc6−32)` times a signed offset code `(q3−4)` — so it uses Q6_K's signed-scale × int8
/// reduction with the offset 32→4. The 3-bit code is 2 low bits from `qs` (traversed exactly like
/// Q2_K) plus 1 high bit from `hmask`. The `−4·bsum` correction reuses `q8.bsums16` (per-16 sums).
/// Dispatches avx512bw → avx2 → scalar.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q3k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(in_f % 256, 0, "vec_dot_q3k: in_f must be a multiple of 256");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q3k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q3k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q3k_scalar(row, q8, in_f)
}

/// Scalar fallback for `vec_dot_q3k`; also used on non-x86 targets. Traverses the 16 per-16-element
/// sub-blocks in the EXACT order `dequant_block(DType::Q3K)` does (`n`×`shift`×`half`), assembling
/// each code `q3 = low2 | (high<<2)` from `qs`'s 2-bit field plus `hmask`'s bit-plane `m`. NOTE `m`
/// advances once per `_j` (8 planes total across the block, 4 per `n`) — NOT once per `n`. The
/// integer epilogue folds the signed `(sc6−32)` scale into `sc·(iprod − 4·bsum16)`, one f32 multiply
/// per super-block (order-free i32 sum — see `vec_dot_q6k_scalar`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q3k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 110..b * 110 + 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let sc = q3k_scales(&blk[96..108]);
        let d = rdf16(&blk[108..110]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut isum = 0i32;
        let mut is = 0usize;
        let mut qoff = 0usize;
        let mut m = 1u8;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                for half in 0..2 {
                    let qbyte = &qs[qoff + half * 16..qoff + half * 16 + 16];
                    let hbyte = &hmask[half * 16..half * 16 + 16];
                    let q8s = &q8b[is * 16..is * 16 + 16];
                    let mut iprod = 0i32;
                    for l in 0..16 {
                        let low2 = ((qbyte[l] >> shift) & 3) as i32;
                        let high = if hbyte[l] & m != 0 { 4 } else { 0 };
                        iprod += (low2 | high) * q8s[l] as i32;
                    }
                    isum += sc[is] as i32 * (iprod - 4 * q8.bsums16[b * 16 + is]);
                    is += 1;
                }
                shift += 2;
                m <<= 1;
            }
            qoff += 32;
        }
        sumf += d * q8.d[b] * isum as f32;
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q3k`: one 32-byte `qs` chunk (per `n` half) serves all 4 shifts, same
/// running 16-bit-shift + mask trick as Q2_K. The extra Q3_K high bit comes from `hmask[0..32]`
/// (loaded once per super-block, byte layout aligns with the 32-byte `cur`): for shift-index `k`
/// the plane bit is `m = 1 << (n*4 + k)`, and `(hmask & m) != 0 → 4` is folded into the code via a
/// `cmpeq`/`andnot`. `maddubs(q3_u8, q8_i8) → madd(1)` then split the ymm into two 16-elem
/// sub-block dots. Signed `(sc6−32)` scale × `(iprod − 4·bsum16)` in the order-free i32 epilogue.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q3k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_03 = _mm256_set1_epi8(0x03_u8 as i8);
    let four = _mm256_set1_epi8(4i8);
    let zero = _mm256_setzero_si256();
    let ones_i16 = _mm256_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 110..b * 110 + 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let sc = q3k_scales(&blk[96..108]);
        let d = rdf16(&blk[108..110]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let hmask_ymm = load256(hmask, 0);
        let mut isum = 0i32;
        for n in 0..2usize {
            // 32 code bytes shared by the 4 shifts (sub-blocks 8n..8n+8).
            let mut cur = load256(qs, n * 32);
            for k in 0..4usize {
                let m = 1u8 << (n * 4 + k);
                // low 2 bits of the current shift; high bit selected from hmask plane m → value 4.
                let low2 = _mm256_and_si256(cur, mask_03);
                let hbit = _mm256_and_si256(hmask_ymm, _mm256_set1_epi8(m as i8));
                let iszero = _mm256_cmpeq_epi8(hbit, zero);
                let high4 = _mm256_andnot_si256(iszero, four);
                let q3 = _mm256_or_si256(low2, high4); // codes 0..7
                let q8v = load256(q8b, n * 128 + k * 32);
                // maddubs: u8(0–7) × i8 → 16×i16 pair-sums (|pair| ≤ 2·7·127 = 1778, no overflow).
                let prod = _mm256_maddubs_epi16(q3, q8v);
                let sum32 = _mm256_madd_epi16(prod, ones_i16);
                let iprod_lo = hadd_i32_xmm(_mm256_castsi256_si128(sum32));
                let iprod_hi = hadd_i32_xmm(_mm256_extracti128_si256::<1>(sum32));
                let is0 = 8 * n + 2 * k;
                isum += sc[is0] as i32 * (iprod_lo - 4 * q8.bsums16[b * 16 + is0]);
                isum += sc[is0 + 1] as i32 * (iprod_hi - 4 * q8.bsums16[b * 16 + is0 + 1]);
                cur = _mm256_srli_epi16(cur, 2);
            }
        }
        sumf += d * q8.d[b] * isum as f32;
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q3k`: both `n` halves' 32-byte `qs` chunks are contiguous (64 B),
/// so one zmm holds them. `hmask[0..32]` is broadcast into both 256-lanes; for shift-index `k` the
/// plane byte differs per half (`1<<k` for `n=0` lower lane, `1<<(4+k)` for `n=1` upper lane), built
/// as a zmm and tested with `_mm512_test_epi8_mask` → `maskz_mov` to fold the high bit (value 4).
/// The matching q8 bytes for the two halves are non-contiguous → 2 ymm loads + insert. One
/// `maddubs512 → madd512` covers 64 elements; the result splits into four 16-elem sub-block dots.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q3k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_03 = _mm512_set1_epi8(0x03_u8 as i8);
    let four = _mm512_set1_epi8(4i8);
    let ones512 = _mm512_set1_epi16(1i16);
    for b in 0..nb {
        let blk = &row[b * 110..b * 110 + 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let sc = q3k_scales(&blk[96..108]);
        let d = rdf16(&blk[108..110]);
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        // qs[0..64] = both n halves contiguous → one zmm. hmask[0..32] broadcast to both lanes.
        let mut cur = load512(qs, 0);
        let hmask_ymm = load256(hmask, 0);
        let hmask_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(hmask_ymm), hmask_ymm);
        let mut isum = 0i32;
        for k in 0..4usize {
            // Plane byte per lane: n=0 → 1<<k (lower 256), n=1 → 1<<(4+k) (upper 256).
            let m_lo = _mm256_set1_epi8((1u8 << k) as i8);
            let m_hi = _mm256_set1_epi8((1u8 << (4 + k)) as i8);
            let m_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(m_lo), m_hi);
            let low2 = _mm512_and_si512(cur, mask_03);
            let hset = _mm512_test_epi8_mask(hmask_z, m_z); // bytes with the plane bit set
            let high4 = _mm512_maskz_mov_epi8(hset, four); // → 4 where set, else 0
            let q3 = _mm512_or_si512(low2, high4);
            // q8 for n=0 (lower 256) and n=1 (upper 256), 32 bytes each, non-contiguous.
            let q8_lo = load256(q8b, k * 32);
            let q8_hi = load256(q8b, 128 + k * 32);
            let q8_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q8_lo), q8_hi);
            let prod = _mm512_maddubs_epi16(q3, q8_z);
            let sum32 = _mm512_madd_epi16(prod, ones512);
            // Lower ymm = n=0 (sub-blocks 2k, 2k+1); upper ymm = n=1 (sub-blocks 8+2k, 8+2k+1).
            let lo_ymm = _mm512_castsi512_si256(sum32);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
            let ip_n0_0 = hadd_i32_xmm(_mm256_castsi256_si128(lo_ymm));
            let ip_n0_1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(lo_ymm));
            let ip_n1_0 = hadd_i32_xmm(_mm256_castsi256_si128(hi_ymm));
            let ip_n1_1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(hi_ymm));
            let is0 = 2 * k; // n=0
            let is1 = 8 + 2 * k; // n=1
            isum += sc[is0] as i32 * (ip_n0_0 - 4 * q8.bsums16[b * 16 + is0]);
            isum += sc[is0 + 1] as i32 * (ip_n0_1 - 4 * q8.bsums16[b * 16 + is0 + 1]);
            isum += sc[is1] as i32 * (ip_n1_0 - 4 * q8.bsums16[b * 16 + is1]);
            isum += sc[is1 + 1] as i32 * (ip_n1_1 - 4 * q8.bsums16[b * 16 + is1 + 1]);
            cur = _mm512_srli_epi16(cur, 2);
        }
        sumf += d * q8.d[b] * isum as f32;
    }
    sumf
}

/// `Σ weight·x` for one IQ4_XS row (136 bytes / 256 elems) against the Q8 activation. Weight value
/// is `d·(ls−32)·KVALUES_IQ4NL[code]` over 8 sub-blocks of 32 (6-bit `ls` scale per sub-block, a
/// signed 16-entry codebook). Dispatches to the best SIMD path available at runtime
/// (avx512bw → avx2 → scalar). Same signed-weight × int8-activation regime as Q6_K, but the weight
/// is a codebook lookup (Q8_0's `abs/sign` maddubs trick) rather than the `q6−32` linear offset, and
/// the per-sub-block `(ls−32)` scale folds into the SAME i32 integer epilogue Q6_K uses.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq4xs(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq4xs: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_iq4xs_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_iq4xs_avx2(row, q8, in_f) };
        }
    }
    vec_dot_iq4xs_scalar(row, q8, in_f)
}

/// Decode the 6-bit `ls` scale for sub-block `ib` of an IQ4_XS block and return `ls − 32`.
/// `lo` = low 4 bits from `scales_l[ib/2]`, `hi` = high 2 bits from `scales_h`.
#[inline]
fn iq4xs_ls_minus_32(scales_h: u16, scales_l: &[u8], ib: usize) -> i32 {
    let lo = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32;
    let hi = ((scales_h >> (2 * ib)) & 3) as i32;
    (lo | (hi << 4)) - 32
}

/// Scalar fallback for `vec_dot_iq4xs`; also used on non-x86 targets. Accumulates a signed integer
/// dot per 32-element sub-block, weights the dots by the integer `(ls−32)` scale in an i32 epilogue
/// (order-free, exact — see `vec_dot_q6k_scalar`), then ONE f32 multiply per super-block.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_iq4xs_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 136..b * 136 + 136];
        let d = rdf16(&blk[0..2]);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut isum = 0i32;
        for ib in 0..8usize {
            let qsb = &qs[ib * 16..ib * 16 + 16];
            let a = &q8b[ib * 32..ib * 32 + 32];
            let mut dot = 0i32;
            for j in 0..16usize {
                // low nibble → element j (0..15); high nibble → element j+16 (16..31).
                let w_lo = KVALUES_IQ4NL[(qsb[j] & 0xF) as usize] as i32;
                let w_hi = KVALUES_IQ4NL[(qsb[j] >> 4) as usize] as i32;
                dot += w_lo * a[j] as i32;
                dot += w_hi * a[j + 16] as i32;
            }
            isum += iq4xs_ls_minus_32(scales_h, scales_l, ib) * dot;
        }
        sumf += d * q8.d[b] * isum as f32;
    }
    sumf
}

/// AVX2 kernel for `vec_dot_iq4xs`: one 32-element sub-block per iteration. The 16 code bytes are
/// nibble-unpacked into a 32-byte codes ymm (low nibbles → elems 0–15 in the low lane, high nibbles
/// → elems 16–31 in the high lane); a `pshufb` against the broadcast codebook table yields the 32
/// signed i8 weights. The signed×int8 dot reuses Q8_0's `abs(w)`/`sign(w)·a` maddubs trick, then the
/// same i32 `(ls−32)`-weighted epilogue as the scalar path → bit-identical (integer dot, order-free).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_iq4xs_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones_i16 = _mm256_set1_epi16(1i16);
    // Codebook table broadcast into both 128-bit lanes for lane-local pshufb.
    let table = _mm256_broadcastsi128_si256(load128(&KVALUES_IQ4NL, 0));
    for b in 0..nb {
        let blk = &row[b * 136..b * 136 + 136];
        let d = rdf16(&blk[0..2]);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut isum = 0i32;
        for ib in 0..8usize {
            // 16 code bytes → xmm; nibbles → 32 codes ([lo16 | hi16]).
            let codes16 = load128(qs, ib * 16);
            let lo =
                _mm256_castsi128_si256(_mm_and_si128(codes16, _mm256_castsi256_si128(mask_0f)));
            let hi = _mm256_castsi128_si256(_mm_and_si128(
                _mm_srli_epi16(codes16, 4),
                _mm256_castsi256_si128(mask_0f),
            ));
            // codes ymm: low lane = lo nibbles (elems 0–15), high lane = hi nibbles (elems 16–31).
            let codes = _mm256_inserti128_si256::<1>(lo, _mm256_castsi256_si128(hi));
            let w = _mm256_shuffle_epi8(table, codes); // 32 signed i8 codebook weights
            let a = load256(q8b, ib * 32);
            // Q8_0 sign trick: |w| unsigned × sign(w)·a signed → maddubs. Pair sum ≤ 2·127·127 < 2^15.
            let w_abs = _mm256_abs_epi8(w);
            let a_signed = _mm256_sign_epi8(a, w);
            let prod = _mm256_maddubs_epi16(w_abs, a_signed);
            let sum32 = _mm256_madd_epi16(prod, ones_i16);
            let dot = hadd_i32_ymm(sum32);
            isum += iq4xs_ls_minus_32(scales_h, scales_l, ib) * dot;
        }
        sumf += d * q8.d[b] * isum as f32;
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_iq4xs`: TWO sub-blocks (64 elems) per zmm. Both sub-blocks' 16 code
/// bytes are nibble-unpacked and lane-shuffled into a 64-byte codes zmm laid out
/// `[lo(ib) | hi(ib) | lo(ib+1) | hi(ib+1)]` to match the contiguous 64-byte activation load; a
/// single `pshufb512` produces 64 signed weights. The signed×int8 dot uses the Q8_0 abs/sign trick
/// at ymm granularity (no `_mm512_sign_epi8`), then splits the 16 i32 lanes back into the two
/// per-sub-block dots for the SAME i32 `(ls−32)` epilogue → bit-identical to scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512f")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_iq4xs_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones_i16_z = _mm512_set1_epi16(1i16);
    let table_z = _mm512_broadcast_i32x4(load128(&KVALUES_IQ4NL, 0));
    for b in 0..nb {
        let blk = &row[b * 136..b * 136 + 136];
        let d = rdf16(&blk[0..2]);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut isum = 0i32;
        for k in 0..4usize {
            let ib = 2 * k;
            // 32 code bytes covering sub-blocks ib (low lane) and ib+1 (high lane).
            let codes32 = load256(qs, ib * 16);
            let lo = _mm256_and_si256(codes32, mask_0f); // lanes: [lo(ib), lo(ib+1)]
            let hi = _mm256_and_si256(_mm256_srli_epi16(codes32, 4), mask_0f); // [hi(ib), hi(ib+1)]
                                                                               // Reorder 128-bit lanes into [lo(ib), hi(ib)] and [lo(ib+1), hi(ib+1)].
            let low256 = _mm256_permute2x128_si256::<0x20>(lo, hi);
            let up256 = _mm256_permute2x128_si256::<0x31>(lo, hi);
            let codes_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low256), up256);
            let w_z = _mm512_shuffle_epi8(table_z, codes_z); // 64 signed weights
                                                             // 64 contiguous activation bytes = [a(ib) | a(ib+1)], lanes align with w_z.
            let a_z = load512(q8b, ib * 32);
            // Sign trick at ymm level (no _mm512_sign_epi8), then repack into zmm.
            let w0 = _mm512_castsi512_si256(w_z);
            let w1 = _mm512_extracti64x4_epi64::<1>(w_z);
            let a0 = _mm512_castsi512_si256(a_z);
            let a1 = _mm512_extracti64x4_epi64::<1>(a_z);
            let wabs_z = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_abs_epi8(w0)),
                _mm256_abs_epi8(w1),
            );
            let as_z = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_sign_epi8(a0, w0)),
                _mm256_sign_epi8(a1, w1),
            );
            let prod = _mm512_maddubs_epi16(wabs_z, as_z);
            let sum32 = _mm512_madd_epi16(prod, ones_i16_z);
            // Lower ymm = sub-block ib's 8 i32; upper ymm = sub-block ib+1's.
            let dot_ib = hadd_i32_ymm(_mm512_castsi512_si256(sum32));
            let dot_ib1 = hadd_i32_ymm(_mm512_extracti64x4_epi64::<1>(sum32));
            isum += iq4xs_ls_minus_32(scales_h, scales_l, ib) * dot_ib;
            isum += iq4xs_ls_minus_32(scales_h, scales_l, ib + 1) * dot_ib1;
        }
        sumf += d * q8.d[b] * isum as f32;
    }
    sumf
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q8_0(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q8_0: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q8_0_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q8_0_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q8_0_scalar(row, q8, in_f)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q8_0_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let bpr = 34usize; // bytes per Q8_0 weight block (32 elems)
    let nb_super = in_f / 256; // activation super-blocks
    let mut sumf = 0f32;
    for b in 0..nb_super {
        let d8 = q8.d[b];
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            let d_w = rdf16(&blk[0..2]);
            let qw = &blk[2..34];
            let qx = &q8.qs[b * 256 + s * 32..b * 256 + s * 32 + 32];
            let iprod: i32 = (0..32).map(|i| qw[i] as i8 as i32 * qx[i] as i32).sum();
            sumf += d8 * d_w * iprod as f32;
        }
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q8_0`: one 32-element Q8_0 weight block per iteration.
/// Sign trick: `maddubs(abs(qw), sign(qw)·qx)` handles i8×i8 without overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q8_0_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let bpr = 34usize;
    let nb_super = in_f / 256;
    let ones = _mm256_set1_epi16(1i16);
    let mut sumf = 0f32;
    for b in 0..nb_super {
        let d8 = q8.d[b];
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            let d_w = rdf16(&blk[0..2]);
            let qw = load256(blk, 2);
            let qx = load256(&q8.qs, b * 256 + s * 32);
            // sign trick: qx_signed = sign(qw) * qx;  abs(qw) stays unsigned
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(qx, qw);
            let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d8 * d_w * iprod as f32;
        }
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q8_0`: TWO 32-element Q8_0 blocks per iteration (64 elems / zmm).
/// Sign trick is applied at ymm level (no `_mm512_sign_epi8`), then results are packed into zmm
/// for a single `maddubs512 → madd512` pass.  Activation bytes for the pair are contiguous in
/// `q8.qs`, so a single `_mm512_loadu_si512` covers both.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q8_0_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let bpr = 34usize;
    let nb_super = in_f / 256;
    let ones512 = _mm512_set1_epi16(1i16);
    let mut sumf = 0f32;
    for b in 0..nb_super {
        let d8 = q8.d[b];
        for k in 0..4usize {
            let s0 = 2 * k;
            let s1 = 2 * k + 1;
            let wb0 = b * 8 + s0;
            let wb1 = b * 8 + s1;
            let d_w0 = rdf16(&row[wb0 * bpr..wb0 * bpr + 2]);
            let d_w1 = rdf16(&row[wb1 * bpr..wb1 * bpr + 2]);
            // Load weight i8 bytes (32 each, non-contiguous due to f16 header)
            let qw0 = load256(row, wb0 * bpr + 2);
            let qw1 = load256(row, wb1 * bpr + 2);
            // Load 64 contiguous activation bytes as zmm (s0*32 and s1*32 are adjacent)
            let qx_z = load512(&q8.qs, b * 256 + s0 * 32);
            // Sign trick at ymm level (no avx512 sign_epi8)
            let qx0 = _mm512_castsi512_si256(qx_z);
            let qx1 = _mm512_extracti64x4_epi64::<1>(qx_z);
            let qw_abs0 = _mm256_abs_epi8(qw0);
            let qw_abs1 = _mm256_abs_epi8(qw1);
            let qx_s0 = _mm256_sign_epi8(qx0, qw0);
            let qx_s1 = _mm256_sign_epi8(qx1, qw1);
            // Pack into zmm
            let qw_a_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qw_abs0), qw_abs1);
            let qx_s_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qx_s0), qx_s1);
            // 512-bit dot
            let prod = _mm512_maddubs_epi16(qw_a_z, qx_s_z);
            let sum32_z = _mm512_madd_epi16(prod, ones512);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d8 * (d_w0 * iprod0 as f32 + d_w1 * iprod1 as f32);
        }
    }
    sumf
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q5k(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(in_f % 256, 0, "vec_dot_q5k: in_f must be a multiple of 256");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q5k_avx512bw(row, q8, in_f) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q5k_avx2(row, q8, in_f) };
        }
    }
    vec_dot_q5k_scalar(row, q8, in_f)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q5k_scalar(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    let nb = in_f / 256;
    let mut sumf = 0f32;
    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = k4(2 * j, scales);
            let (sc_o, m_o) = k4(2 * j + 1, scales);
            let ql_chunk = &ql[j * 32..j * 32 + 32];
            let q8_e = &q8b[2 * j * 32..(2 * j + 1) * 32];
            let q8_o = &q8b[(2 * j + 1) * 32..(2 * j + 2) * 32];
            let bsum_e = q8.bsums[b * 8 + 2 * j];
            let bsum_o = q8.bsums[b * 8 + 2 * j + 1];
            let mut iprod_e = 0i32;
            let mut iprod_o = 0i32;
            for l in 0..32 {
                let v = ql_chunk[l];
                let q5_e = (v & 0xF) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
                let q5_o = (v >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                iprod_e += q5_e * q8_e[l] as i32;
                iprod_o += q5_o * q8_o[l] as i32;
            }
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * bsum_e + m_o as i32 * bsum_o;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX2 kernel for `vec_dot_q5k`: one nibble-pair per iteration (64 elements = two 32-elem
/// sub-blocks).  High bit per element is extracted from `qh` using per-j bit masks: if the
/// bit is set the value adds 16.  `maddubs` works directly since q5 ∈ 0..31 (unsigned).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5k_avx2(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);
    let zero = _mm256_setzero_si256();
    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let qh_ymm = load256(qh, 0);
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = k4(2 * j, scales);
            let (sc_o, m_o) = k4(2 * j + 1, scales);
            // Unpack nibbles from ql[j*32..+32]
            let nibbles = load256(ql, j * 32);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            // High-bit extraction: if (qh[l] & u) != 0 → add 16 to that element.
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            // andnot(cmpeq_zero, 0x10) = 0x10 where nonzero, 0 otherwise
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero), sixteen);
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            // Load Q8 activation bytes for both sub-blocks
            let q8_e = load256(q8b, 2 * j * 32);
            let q8_o = load256(q8b, (2 * j + 1) * 32);
            // maddubs: q5 is u8 (0..31), q8 is i8 → direct, no sign trick needed
            let prod_e = _mm256_maddubs_epi16(q5_e, q8_e);
            let sum32_e = _mm256_madd_epi16(prod_e, ones);
            let iprod_e = hadd_i32_ymm(sum32_e);
            let prod_o = _mm256_maddubs_epi16(q5_o, q8_o);
            let sum32_o = _mm256_madd_epi16(prod_o, ones);
            let iprod_o = hadd_i32_ymm(sum32_o);
            let isum_e = q8.bsums[b * 8 + 2 * j];
            let isum_o = q8.bsums[b * 8 + 2 * j + 1];
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

/// AVX-512BW kernel for `vec_dot_q5k`: identical structure to the Q4_K AVX-512BW kernel but with
/// the 5th bit ORed in from `qh`.  Each iteration (k=0..4) processes one nibble pair (64 elements)
/// via a zmm.  The high bit is extracted from `qh_ymm` using per-k bit masks at ymm width; results
/// are inserted into zmm for the 512-bit `maddubs → madd` pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5k_avx512bw(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    use std::arch::x86_64::*;
    let nb = in_f / 256;
    let mut sumf = 0f32;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);
    let zero256 = _mm256_setzero_si256();
    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        let d = rdf16(&blk[0..2]);
        let dmin = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let qh_ymm = load256(qh, 0);
        let (mut sd, mut sm) = (0i32, 0i32);
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for k in 0..4usize {
            let (sc_e, m_e) = k4(2 * k, scales);
            let (sc_o, m_o) = k4(2 * k + 1, scales);
            let nibbles = load256(ql, k * 32);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            // High-bit extraction per bit pair
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero256), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero256), sixteen);
            // q5 values (0..31, unsigned)
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            // Pack into zmm: lower 256 = even sub-block, upper 256 = odd sub-block
            let q5_zmm = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q5_e), q5_o);
            // 64 contiguous Q8 activation bytes for this pair
            let q8_zmm = load512(q8b, k * 64);
            let prod = _mm512_maddubs_epi16(q5_zmm, q8_zmm);
            let sum32_z = _mm512_madd_epi16(prod, ones512);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod_e = hadd_i32_ymm(lo_ymm);
            let iprod_o = hadd_i32_ymm(hi_ymm);
            let isum_e = q8.bsums[b * 8 + 2 * k];
            let isum_o = q8.bsums[b * 8 + 2 * k + 1];
            sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
            sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            u1 <<= 2;
            u2 <<= 2;
        }
        sumf += q8.d[b] * (d * sd as f32 - dmin * sm as f32);
    }
    sumf
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q4k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;

    // Decode weight row once: per-block d/dmin/sc/m, expanded nibbles (0–15).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![0u32; nb * 8];
    let mut m_arr = vec![0u32; nb * 8];
    let mut q4_flat = vec![0u8; nb * 256]; // one byte per element, value 0–15

    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b * 8 + s] = sc;
            m_arr[b * 8 + s] = mv;
            let hi = s % 2 == 1;
            let half = s / 2;
            let qbyte = &qs[half * 32..half * 32 + 32];
            let flat = &mut q4_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
            for l in 0..32 {
                flat[l] = if hi { qbyte[l] >> 4 } else { qbyte[l] & 0xF };
            }
        }
    }

    // Per-token dot using pre-expanded data (identical order to scalar single-token).
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            for s in 0..8usize {
                let mut iprod = 0i32;
                let fb = &q4_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
                let q8b = &q8.qs[b * 256 + s * 32..b * 256 + s * 32 + 32];
                for l in 0..32 {
                    iprod += fb[l] as i32 * q8b[l] as i32;
                }
                sd += sc_arr[b * 8 + s] as i32 * iprod;
                sm += m_arr[b * 8 + s] as i32 * q8.bsums[b * 8 + s];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// Decode one Q4_K weight row into the batch kernels' shared scratch: per-block `d`/`dmin`, the
/// eight `sc`/`m` sub-block scales, and nibble-expanded `q4_flat` in the pair-interleaved layout
/// `flat[b*256 + k*64 .. +64] = [lo_nib(sub-block 2k) (32 B), hi_nib(sub-block 2k+1) (32 B)]`,
/// directly loadable as a ymm/zmm per pair `k`. This is the exact decode the AVX2/AVX512BW/VNNI
/// Q4_K batch kernels shared verbatim — factored out so the nibble-unpack lives in one place.
/// Bit-identical by construction: only integer/f16 decode, no float reduction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn q4k_decode_row(
    row: &[u8],
    nb: usize,
    d_arr: &mut [f32],
    dmin_arr: &mut [f32],
    sc_arr: &mut [[u32; 8]],
    m_arr: &mut [[u32; 8]],
    q4_flat: &mut [u8],
) {
    use std::arch::x86_64::*;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    for b in 0..nb {
        let blk = &row[b * 144..b * 144 + 144];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b][s] = sc;
            m_arr[b][s] = mv;
        }
        // Unpack 4 nibble pairs with SIMD, store lo then hi in contiguous slots.
        let flat = &mut q4_flat[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibbles = load256(qs, k * 32);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo); // sub-block 2k
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo); // sub-block 2k+1
            _mm256_storeu_si256(flat[k * 64..].as_mut_ptr() as *mut __m256i, lo_nib);
            _mm256_storeu_si256(flat[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi_nib);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let ones = _mm256_set1_epi16(1i16);

    // Pre-expand nibbles into flat[b*256..b*256+256] once (see q4k_decode_row for the layout).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q4_flat = vec![0u8; nb * 256];

    q4k_decode_row(
        row,
        nb,
        &mut d_arr,
        &mut dmin_arr,
        &mut sc_arr,
        &mut m_arr,
        &mut q4_flat,
    );

    // Per-token dots: load pre-expanded q4 ymm + q8 ymm, maddubs → madd → hadd.
    // Accumulation order identical to single-token AVX2 kernel → bit-identical result.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q4_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for s in 0..8usize {
                let q4 = load256(flat, s * 32);
                let q8v = load256(q8b, s * 32);
                let prod = _mm256_maddubs_epi16(q4, q8v);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod = hadd_i32_ymm(sum32);
                let isum = q8.bsums[b * 8 + s];
                sd += sc_arr[b][s] as i32 * iprod;
                sm += m_arr[b][s] as i32 * isum;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let ones512 = _mm512_set1_epi16(1i16);

    // Pre-expand nibbles (same layout as AVX2 batch, see q4k_decode_row): flat[b*256 + k*64..+64]
    // = [lo_nib_2k (32 B), hi_nib_2k+1 (32 B)], directly loadable as a zmm per pair k.
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q4_flat = vec![0u8; nb * 256];

    q4k_decode_row(
        row,
        nb,
        &mut d_arr,
        &mut dmin_arr,
        &mut sc_arr,
        &mut m_arr,
        &mut q4_flat,
    );

    // Per-token: one zmm load per pair (64 pre-expanded bytes) + one zmm q8 load.
    // maddubs512 → madd512 → split ymm → hadd×2. Identical to single-token avx512bw kernel.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q4_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for k in 0..4usize {
                let (sc_e, m_e) = (sc_arr[b][2 * k], m_arr[b][2 * k]);
                let (sc_o, m_o) = (sc_arr[b][2 * k + 1], m_arr[b][2 * k + 1]);
                // flat[k*64..k*64+64]: lower 256 = sub-block 2k, upper 256 = sub-block 2k+1
                let q4_zmm = load512(flat, k * 64);
                let q8_zmm = load512(q8b, k * 64);
                let prod = _mm512_maddubs_epi16(q4_zmm, q8_zmm);
                let sum32 = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                let iprod_e = hadd_i32_ymm(lo_ymm);
                let iprod_o = hadd_i32_ymm(hi_ymm);
                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];
                sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
                sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI variant of [`vec_dot_q4k_batch_avx512bw`]: `_mm512_dpbusd_epi32` fuses the
/// maddubs+madd pair into ONE u8×s8→i32 dot-accumulate. Bit-identical — the i32 lanes hold the
/// same per-4-byte-group sums the madd chain produced (q4 nibbles ≤15 × |q8| ≤127 never
/// saturated maddubs' i16 either), and the hadd/scale order is unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_batch_vnni(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;

    // Pre-expand nibbles (same layout as AVX2 batch, see q4k_decode_row).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q4_flat = vec![0u8; nb * 256];

    q4k_decode_row(
        row,
        nb,
        &mut d_arr,
        &mut dmin_arr,
        &mut sc_arr,
        &mut m_arr,
        &mut q4_flat,
    );

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q4_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for k in 0..4usize {
                let (sc_e, m_e) = (sc_arr[b][2 * k], m_arr[b][2 * k]);
                let (sc_o, m_o) = (sc_arr[b][2 * k + 1], m_arr[b][2 * k + 1]);
                let q4_zmm = load512(flat, k * 64);
                let q8_zmm = load512(q8b, k * 64);
                let sum32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_zmm, q8_zmm);
                let lo_ymm = _mm512_castsi512_si256(sum32);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                let iprod_e = hadd_i32_ymm(lo_ymm);
                let iprod_o = hadd_i32_ymm(hi_ymm);
                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];
                sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
                sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// Batch Q4_K dot: `out[r] = vec_dot_q4k(row, &q8s[r], in_f)` for all r, bit-identical to
/// the single-token kernel. The weight row is decoded ONCE; per-token work is the integer dot only.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q4k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q4k_batch: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q4k_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q4k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q4k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q4k_batch_scalar(row, q8s, in_f, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_batch2_avx512bw(
    row_a: &[u8],
    row_b: &[u8],
    q8s: &[Q8],
    in_f: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);

    // Pre-expand both weight rows.
    let mut d_a = vec![0f32; nb];
    let mut dmin_a = vec![0f32; nb];
    let mut sc_a = vec![[0u32; 8]; nb];
    let mut m_a = vec![[0u32; 8]; nb];
    let mut flat_a = vec![0u8; nb * 256];

    let mut d_b = vec![0f32; nb];
    let mut dmin_b = vec![0f32; nb];
    let mut sc_b = vec![[0u32; 8]; nb];
    let mut m_b = vec![[0u32; 8]; nb];
    let mut flat_b = vec![0u8; nb * 256];

    for b in 0..nb {
        // Row A
        let blk_a = &row_a[b * 144..b * 144 + 144];
        d_a[b] = rdf16(&blk_a[0..2]);
        dmin_a[b] = rdf16(&blk_a[2..4]);
        let scales_a = &blk_a[4..16];
        let qs_a = &blk_a[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales_a);
            sc_a[b][s] = sc;
            m_a[b][s] = mv;
        }
        let fa = &mut flat_a[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibs = load256(qs_a, k * 32);
            let lo = _mm256_and_si256(nibs, mask_lo);
            let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
            _mm256_storeu_si256(fa[k * 64..].as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(fa[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
        }

        // Row B
        let blk_b = &row_b[b * 144..b * 144 + 144];
        d_b[b] = rdf16(&blk_b[0..2]);
        dmin_b[b] = rdf16(&blk_b[2..4]);
        let scales_b = &blk_b[4..16];
        let qs_b = &blk_b[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales_b);
            sc_b[b][s] = sc;
            m_b[b][s] = mv;
        }
        let fb = &mut flat_b[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibs = load256(qs_b, k * 32);
            let lo = _mm256_and_si256(nibs, mask_lo);
            let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
            _mm256_storeu_si256(fb[k * 64..].as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(fb[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
        }
    }

    // Per-token: load q8 ONCE per block per pair k; compute both row A and row B dots.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf_a = 0f32;
        let mut sumf_b = 0f32;
        for b in 0..nb {
            let (mut sd_a, mut sm_a) = (0i32, 0i32);
            let (mut sd_b, mut sm_b) = (0i32, 0i32);
            let fa = &flat_a[b * 256..b * 256 + 256];
            let fb = &flat_b[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            for k in 0..4usize {
                let (sca_e, ma_e) = (sc_a[b][2 * k], m_a[b][2 * k]);
                let (sca_o, ma_o) = (sc_a[b][2 * k + 1], m_a[b][2 * k + 1]);
                let (scb_e, mb_e) = (sc_b[b][2 * k], m_b[b][2 * k]);
                let (scb_o, mb_o) = (sc_b[b][2 * k + 1], m_b[b][2 * k + 1]);

                // Load q8 ONCE for this pair (shared by both row A and row B).
                let q8_zmm = load512(q8b, k * 64);

                // Row A dot
                let qa_zmm = load512(fa, k * 64);
                let prod_a = _mm512_maddubs_epi16(qa_zmm, q8_zmm);
                let sum32_a = _mm512_madd_epi16(prod_a, ones512);
                let lo_a = _mm512_castsi512_si256(sum32_a);
                let hi_a = _mm512_extracti64x4_epi64::<1>(sum32_a);

                // Row B dot (q8_zmm reused, no reload)
                let qb_zmm = load512(fb, k * 64);
                let prod_b = _mm512_maddubs_epi16(qb_zmm, q8_zmm);
                let sum32_b = _mm512_madd_epi16(prod_b, ones512);
                let lo_b = _mm512_castsi512_si256(sum32_b);
                let hi_b = _mm512_extracti64x4_epi64::<1>(sum32_b);

                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];

                sd_a += sca_e as i32 * hadd_i32_ymm(lo_a) + sca_o as i32 * hadd_i32_ymm(hi_a);
                sm_a += ma_e as i32 * isum_e + ma_o as i32 * isum_o;

                sd_b += scb_e as i32 * hadd_i32_ymm(lo_b) + scb_o as i32 * hadd_i32_ymm(hi_b);
                sm_b += mb_e as i32 * isum_e + mb_o as i32 * isum_o;
            }
            sumf_a += q8.d[b] * (d_a[b] * sd_a as f32 - dmin_a[b] * sm_a as f32);
            sumf_b += q8.d[b] * (d_b[b] * sd_b as f32 - dmin_b[b] * sm_b as f32);
        }
        out_a[r] = sumf_a;
        out_b[r] = sumf_b;
    }
}

/// AVX512-VNNI variant of [`vec_dot_q4k_batch2_avx512bw`] (2-row tile, q8 loaded once per pair,
/// dpbusd replacing maddubs+madd — see [`vec_dot_q4k_batch_vnni`]'s bit-identity note).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_batch2_vnni(
    row_a: &[u8],
    row_b: &[u8],
    q8s: &[Q8],
    in_f: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);

    let mut d_a = vec![0f32; nb];
    let mut dmin_a = vec![0f32; nb];
    let mut sc_a = vec![[0u32; 8]; nb];
    let mut m_a = vec![[0u32; 8]; nb];
    let mut flat_a = vec![0u8; nb * 256];

    let mut d_b = vec![0f32; nb];
    let mut dmin_b = vec![0f32; nb];
    let mut sc_b = vec![[0u32; 8]; nb];
    let mut m_b = vec![[0u32; 8]; nb];
    let mut flat_b = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk_a = &row_a[b * 144..b * 144 + 144];
        d_a[b] = rdf16(&blk_a[0..2]);
        dmin_a[b] = rdf16(&blk_a[2..4]);
        let scales_a = &blk_a[4..16];
        let qs_a = &blk_a[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales_a);
            sc_a[b][s] = sc;
            m_a[b][s] = mv;
        }
        let fa = &mut flat_a[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibs = load256(qs_a, k * 32);
            let lo = _mm256_and_si256(nibs, mask_lo);
            let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
            _mm256_storeu_si256(fa[k * 64..].as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(fa[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
        }

        let blk_b = &row_b[b * 144..b * 144 + 144];
        d_b[b] = rdf16(&blk_b[0..2]);
        dmin_b[b] = rdf16(&blk_b[2..4]);
        let scales_b = &blk_b[4..16];
        let qs_b = &blk_b[16..144];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales_b);
            sc_b[b][s] = sc;
            m_b[b][s] = mv;
        }
        let fb = &mut flat_b[b * 256..b * 256 + 256];
        for k in 0..4usize {
            let nibs = load256(qs_b, k * 32);
            let lo = _mm256_and_si256(nibs, mask_lo);
            let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
            _mm256_storeu_si256(fb[k * 64..].as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(fb[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf_a = 0f32;
        let mut sumf_b = 0f32;
        for b in 0..nb {
            let (mut sd_a, mut sm_a) = (0i32, 0i32);
            let (mut sd_b, mut sm_b) = (0i32, 0i32);
            let fa = &flat_a[b * 256..b * 256 + 256];
            let fb = &flat_b[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            for k in 0..4usize {
                let (sca_e, ma_e) = (sc_a[b][2 * k], m_a[b][2 * k]);
                let (sca_o, ma_o) = (sc_a[b][2 * k + 1], m_a[b][2 * k + 1]);
                let (scb_e, mb_e) = (sc_b[b][2 * k], m_b[b][2 * k]);
                let (scb_o, mb_o) = (sc_b[b][2 * k + 1], m_b[b][2 * k + 1]);

                let q8_zmm = load512(q8b, k * 64);

                let qa_zmm = load512(fa, k * 64);
                let sum32_a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qa_zmm, q8_zmm);
                let lo_a = _mm512_castsi512_si256(sum32_a);
                let hi_a = _mm512_extracti64x4_epi64::<1>(sum32_a);

                let qb_zmm = load512(fb, k * 64);
                let sum32_b = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qb_zmm, q8_zmm);
                let lo_b = _mm512_castsi512_si256(sum32_b);
                let hi_b = _mm512_extracti64x4_epi64::<1>(sum32_b);

                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];

                sd_a += sca_e as i32 * hadd_i32_ymm(lo_a) + sca_o as i32 * hadd_i32_ymm(hi_a);
                sm_a += ma_e as i32 * isum_e + ma_o as i32 * isum_o;

                sd_b += scb_e as i32 * hadd_i32_ymm(lo_b) + scb_o as i32 * hadd_i32_ymm(hi_b);
                sm_b += mb_e as i32 * isum_e + mb_o as i32 * isum_o;
            }
            sumf_a += q8.d[b] * (d_a[b] * sd_a as f32 - dmin_a[b] * sm_a as f32);
            sumf_b += q8.d[b] * (d_b[b] * sd_b as f32 - dmin_b[b] * sm_b as f32);
        }
        out_a[r] = sumf_a;
        out_b[r] = sumf_b;
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q4k_batch2(
    row_a: &[u8],
    row_b: &[u8],
    q8s: &[Q8],
    in_f: usize,
    out_a: &mut [f32],
    out_b: &mut [f32],
) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q4k_batch2: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q4k_batch2_vnni(row_a, row_b, q8s, in_f, out_a, out_b) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q4k_batch2_avx512bw(row_a, row_b, q8s, in_f, out_a, out_b) };
        }
    }
    // Scalar fallback: call single-row batch twice (still unpack once per row).
    vec_dot_q4k_batch(row_a, q8s, in_f, out_a);
    vec_dot_q4k_batch(row_b, q8s, in_f, out_b);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_batch8_avx512bw(
    rows: [&[u8]; 8],
    q8s: &[Q8],
    in_f: usize,
    outs: [&mut [f32]; 8],
) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);

    // Pre-expand all 8 weight rows once. Layout identical to the single-row AVX-512BW
    // batch kernel: flat[i][b*256 + k*64 .. +64] = [lo_nib 32B, hi_nib 32B] for pair k.
    let mut flats: [Vec<u8>; 8] = std::array::from_fn(|_| vec![0u8; nb * 256]);
    let mut d_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut dmin_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut sc_arr: [Vec<[u32; 8]>; 8] = std::array::from_fn(|_| vec![[0u32; 8]; nb]);
    let mut m_arr: [Vec<[u32; 8]>; 8] = std::array::from_fn(|_| vec![[0u32; 8]; nb]);

    for i in 0..8 {
        for b in 0..nb {
            let blk = &rows[i][b * 144..b * 144 + 144];
            d_arr[i][b] = rdf16(&blk[0..2]);
            dmin_arr[i][b] = rdf16(&blk[2..4]);
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            for s in 0..8usize {
                let (sc, mv) = k4(s, scales);
                sc_arr[i][b][s] = sc;
                m_arr[i][b][s] = mv;
            }
            let f = &mut flats[i][b * 256..b * 256 + 256];
            for k in 0..4usize {
                let nibs = load256(qs, k * 32);
                let lo = _mm256_and_si256(nibs, mask_lo);
                let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
                _mm256_storeu_si256(f[k * 64..].as_mut_ptr() as *mut __m256i, lo);
                _mm256_storeu_si256(f[k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
            }
        }
    }

    // Destructure outs so we can write to 8 independent &mut [f32] without aliasing.
    let [o0, o1, o2, o3, o4, o5, o6, o7] = outs;

    // Per-token dot: for each token r, load the Q8 activation zmm ONCE per (b, k) pair
    // and reuse it across all 8 weight rows — 8× the FMAs per activation load.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = [0f32; 8];

        for b in 0..nb {
            // ── Deferred-hadd accumulation ──────────────────────────────────────────
            // Instead of hadd_i32_ymm inside the k loop (8 rows × 2 hadd × 4 k = 64
            // hadd calls/block, all on port 5), we accumulate scaled ymm vectors and
            // hadd once per row after all k-pairs are done (16 hadd calls/block).
            //
            // Bit-identical: hadd(Σ_k scale[k]·v[k]) = Σ_k scale[k]·hadd(v[k])
            // because hadd is a linear sum and integer mullo is exact (no overflow for
            // our value ranges: sc≤63, per-element sum≤4×15×127 ≈ 7620 → product ≤ ~480k
            // → fits i32; 8-element accumulation ≤ ~3.8M → fits i32).
            //
            // acc_lo[i] = Σ_k ( sc_e[k] × lo_ymm[k] )   — 8 × i32 lanes
            // acc_hi[i] = Σ_k ( sc_o[k] × hi_ymm[k] )   — 8 × i32 lanes
            // sd_i     = hadd(acc_lo[i]) + hadd(acc_hi[i])
            let mut acc_lo = [_mm256_setzero_si256(); 8];
            let mut acc_hi = [_mm256_setzero_si256(); 8];
            let mut sm = [0i32; 8];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            for k in 0..4usize {
                // ── ONE activation load for pair k, shared by all 8 weight rows ──
                let q8_zmm = load512(q8b, k * 64);
                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];

                // ── 8 weight row dots against the shared q8_zmm ──
                for i in 0..8usize {
                    let (sc_e, ma_e) = (sc_arr[i][b][2 * k], m_arr[i][b][2 * k]);
                    let (sc_o, ma_o) = (sc_arr[i][b][2 * k + 1], m_arr[i][b][2 * k + 1]);
                    let qi_zmm = load512(&flats[i], b * 256 + k * 64);
                    let prod = _mm512_maddubs_epi16(qi_zmm, q8_zmm);
                    let sum32 = _mm512_madd_epi16(prod, ones512);
                    let lo_ymm = _mm512_castsi512_si256(sum32);
                    let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                    // Scale each 8×i32 sub-block by its per-sub-block scale and
                    // accumulate into ymm registers — no hadd in the hot path.
                    acc_lo[i] = _mm256_add_epi32(
                        acc_lo[i],
                        _mm256_mullo_epi32(lo_ymm, _mm256_set1_epi32(sc_e as i32)),
                    );
                    acc_hi[i] = _mm256_add_epi32(
                        acc_hi[i],
                        _mm256_mullo_epi32(hi_ymm, _mm256_set1_epi32(sc_o as i32)),
                    );
                    sm[i] += ma_e as i32 * isum_e + ma_o as i32 * isum_o;
                }
            }
            // ── 2 hadd per row per block (vs 8 in eager version) ──────────────────
            for i in 0..8 {
                let sd_i = hadd_i32_ymm(acc_lo[i]) + hadd_i32_ymm(acc_hi[i]);
                sumf[i] += q8.d[b] * (d_arr[i][b] * sd_i as f32 - dmin_arr[i][b] * sm[i] as f32);
            }
        }
        o0[r] = sumf[0];
        o1[r] = sumf[1];
        o2[r] = sumf[2];
        o3[r] = sumf[3];
        o4[r] = sumf[4];
        o5[r] = sumf[5];
        o6[r] = sumf[6];
        o7[r] = sumf[7];
    }
}

/// Interleaved-x8 Q4_K GEMM tile (AVX512-VNNI) — the ggml `block_q4_Kx8` idea applied per call:
/// the 8 weight rows' expanded nibbles are INTERLEAVED per (super-block, sub-block, 8-byte group)
/// so one contiguous zmm load carries the SAME 8 activation positions of ALL 8 rows (qword lane i
/// = row i), and the activation qword is broadcast once. Per sub-block that's 4 dpbusd + ONE
/// hadd + ONE mullo for all 8 rows — vs the flat batch8's 8 dpbusd + 16 mullo + per-row hadds —
/// and every weight byte the inner loop touches is sequential.
///
/// `_mm256_hadd_epi32(lo, hi)` yields per-row sums in the fixed permutation [0,1,4,5,2,3,6,7]
/// (its 128-bit-lane semantics); scales are pre-permuted to match and lanes un-permute only at
/// the per-super-block extraction. Bit-identical to the scalar single-row kernel: the integer
/// sums are exact regardless of association, and the per-(row, super-block) f32 expression and
/// its accumulation order are unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4k_batch8_ilv_vnni(
    rows: [&[u8]; 8],
    q8s: &[Q8],
    in_f: usize,
    outs: [&mut [f32]; 8],
) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    // hadd_epi32(lo, hi) sum-position j holds row PERM[j].
    const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

    // ── repack: expand nibbles per row (as the flat kernels do), then interleave ──
    // ilv[b*2048 + s*256 + g*64 + i*8 .. +8] = row i, sub-block s, byte group g (8 B).
    let mut d_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut dmin_arr: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    let mut m_arr: [Vec<[u32; 8]>; 8] = std::array::from_fn(|_| vec![[0u32; 8]; nb]);
    // per-(b, s) PAIR-DUPLICATED scale vector in dpbusd lane order (lanes 2i, 2i+1 = row i) —
    // consumed by the vertical 512-bit mullo; see q4k_gemm_group's sc16.
    let mut sc16_vec = vec![[_mm512_setzero_si512(); 8]; nb];
    // PERM-ordered per-(b,s) m-scales and per-b f32 d/dmin vectors: lets the sm accumulation and
    // the final `q8.d*(d*sd - dmin*sm)` run 8-rows-wide in lane space (un-permuted only at the
    // very end). No FMA anywhere — each mul/sub/add rounds separately, matching the scalar order.
    let mut m_vec = vec![[_mm256_setzero_si256(); 8]; nb];
    let mut d_vec = vec![_mm256_setzero_ps(); nb];
    let mut dmin_vec = vec![_mm256_setzero_ps(); nb];
    let mut ilv = vec![0u8; nb * 2048];
    {
        let mut sc_rows = [[0u32; 8]; 8]; // [row][s]
        let mut tmp = [[0u8; 256]; 8]; // expanded nibbles, one superblock, all 8 rows
        for b in 0..nb {
            for i in 0..8 {
                let blk = &rows[i][b * 144..b * 144 + 144];
                d_arr[i][b] = rdf16(&blk[0..2]);
                dmin_arr[i][b] = rdf16(&blk[2..4]);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                for s in 0..8usize {
                    let (sc, mv) = k4(s, scales);
                    sc_rows[i][s] = sc;
                    m_arr[i][b][s] = mv;
                }
                for k in 0..4usize {
                    let nibs = load256(qs, k * 32);
                    let lo = _mm256_and_si256(nibs, mask_lo);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
                    _mm256_storeu_si256(tmp[i][k * 64..].as_mut_ptr() as *mut __m256i, lo);
                    _mm256_storeu_si256(tmp[i][k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
                }
            }
            d_vec[b] = _mm256_setr_ps(
                d_arr[PERM[0]][b],
                d_arr[PERM[1]][b],
                d_arr[PERM[2]][b],
                d_arr[PERM[3]][b],
                d_arr[PERM[4]][b],
                d_arr[PERM[5]][b],
                d_arr[PERM[6]][b],
                d_arr[PERM[7]][b],
            );
            dmin_vec[b] = _mm256_setr_ps(
                dmin_arr[PERM[0]][b],
                dmin_arr[PERM[1]][b],
                dmin_arr[PERM[2]][b],
                dmin_arr[PERM[3]][b],
                dmin_arr[PERM[4]][b],
                dmin_arr[PERM[5]][b],
                dmin_arr[PERM[6]][b],
                dmin_arr[PERM[7]][b],
            );
            for s in 0..8usize {
                sc16_vec[b][s] = _mm512_setr_epi32(
                    sc_rows[0][s] as i32,
                    sc_rows[0][s] as i32,
                    sc_rows[1][s] as i32,
                    sc_rows[1][s] as i32,
                    sc_rows[2][s] as i32,
                    sc_rows[2][s] as i32,
                    sc_rows[3][s] as i32,
                    sc_rows[3][s] as i32,
                    sc_rows[4][s] as i32,
                    sc_rows[4][s] as i32,
                    sc_rows[5][s] as i32,
                    sc_rows[5][s] as i32,
                    sc_rows[6][s] as i32,
                    sc_rows[6][s] as i32,
                    sc_rows[7][s] as i32,
                    sc_rows[7][s] as i32,
                );
                m_vec[b][s] = _mm256_setr_epi32(
                    m_arr[PERM[0]][b][s] as i32,
                    m_arr[PERM[1]][b][s] as i32,
                    m_arr[PERM[2]][b][s] as i32,
                    m_arr[PERM[3]][b][s] as i32,
                    m_arr[PERM[4]][b][s] as i32,
                    m_arr[PERM[5]][b][s] as i32,
                    m_arr[PERM[6]][b][s] as i32,
                    m_arr[PERM[7]][b][s] as i32,
                );
                for g in 0..4usize {
                    let dst =
                        &mut ilv[b * 2048 + s * 256 + g * 64..b * 2048 + s * 256 + g * 64 + 64];
                    for i in 0..8 {
                        dst[i * 8..i * 8 + 8]
                            .copy_from_slice(&tmp[i][s * 32 + g * 8..s * 32 + g * 8 + 8]);
                    }
                }
            }
        }
    }

    let [o0, o1, o2, o3, o4, o5, o6, o7] = outs;
    for r in 0..m {
        let q8 = &q8s[r];
        // Whole-row f32 accumulator, 8 rows wide in PERM lane order — un-permuted only at the
        // final store. Every f32 op below is a separate mul/sub/add (NO fma): identical rounding
        // sequence to the scalar per-row expression.
        let mut sumf_v = _mm256_setzero_ps();
        for b in 0..nb {
            // sd/sm per row: sd vertical in 512-bit pair-lane space, sm in PERM lane order.
            let mut sd_zmm = _mm512_setzero_si512();
            let mut sm_ymm = _mm256_setzero_si256();
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for s in 0..8usize {
                let mut acc = _mm512_setzero_si512();
                for g in 0..4usize {
                    let w = load512(&ilv, b * 2048 + s * 256 + g * 64);
                    // q8b is &[i8]; read the 8 activation bytes as one little-endian qword.
                    let a = _mm512_set1_epi64(
                        (q8b[s * 32 + g * 8..].as_ptr() as *const i64).read_unaligned(),
                    );
                    acc = _mm512_dpbusd_epi32(acc, w, a);
                }
                // Vertical: scale pair lanes in 512-bit space; pair-merge deferred to once per
                // super-block (integer-exact — see q4k_gemm_group's note).
                sd_zmm = _mm512_add_epi32(sd_zmm, _mm512_mullo_epi32(acc, sc16_vec[b][s]));
                let isum = _mm256_set1_epi32(q8.bsums[b * 8 + s]);
                sm_ymm = _mm256_add_epi32(sm_ymm, _mm256_mullo_epi32(m_vec[b][s], isum));
            }
            let sd_lo = _mm512_castsi512_si256(sd_zmm);
            let sd_hi = _mm512_extracti64x4_epi64::<1>(sd_zmm);
            let sd_ymm = _mm256_hadd_epi32(sd_lo, sd_hi); // pair-merge -> PERM order
                                                          // q8.d[b] * (d*sd - dmin*sm), 8 rows at once (cvtepi32→f32 is exact for these
                                                          // magnitudes; mul/sub/mul/add sequence matches the scalar expression's rounding).
            let sd_f = _mm256_cvtepi32_ps(sd_ymm);
            let sm_f = _mm256_cvtepi32_ps(sm_ymm);
            let t = _mm256_sub_ps(
                _mm256_mul_ps(d_vec[b], sd_f),
                _mm256_mul_ps(dmin_vec[b], sm_f),
            );
            sumf_v = _mm256_add_ps(sumf_v, _mm256_mul_ps(_mm256_set1_ps(q8.d[b]), t));
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sumf_v);
        o0[r] = lanes[0];
        o1[r] = lanes[1];
        o4[r] = lanes[2];
        o5[r] = lanes[3];
        o2[r] = lanes[4];
        o3[r] = lanes[5];
        o6[r] = lanes[6];
        o7[r] = lanes[7];
    }
}

/// Batch Q4_K 8-row tile: `outs[i][r] = vec_dot_q4k(rows[i], &q8s[r], in_f)` for all i,r.
/// Bit-identical to the single-token kernel. On AVX-512BW machines the Q8 activation is
/// loaded once per (block, nibble-pair) and dotted against all 8 weight rows — 8× activation
/// reuse over single-row, 4× over 2-row. Falls back to 8× `vec_dot_q4k_batch` on older CPUs.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q4k_batch8(rows: [&[u8]; 8], q8s: &[Q8], in_f: usize, outs: [&mut [f32]; 8]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q4k_batch8: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q4k_batch8_ilv_vnni(rows, q8s, in_f, outs) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q4k_batch8_avx512bw(rows, q8s, in_f, outs) };
        }
    }
    // Fallback: call the per-row batch kernel (avx2/scalar dispatch) for each of the 8 rows.
    let [row0, row1, row2, row3, row4, row5, row6, row7] = rows;
    let [out0, out1, out2, out3, out4, out5, out6, out7] = outs;
    vec_dot_q4k_batch(row0, q8s, in_f, out0);
    vec_dot_q4k_batch(row1, q8s, in_f, out1);
    vec_dot_q4k_batch(row2, q8s, in_f, out2);
    vec_dot_q4k_batch(row3, q8s, in_f, out3);
    vec_dot_q4k_batch(row4, q8s, in_f, out4);
    vec_dot_q4k_batch(row5, q8s, in_f, out5);
    vec_dot_q4k_batch(row6, q8s, in_f, out6);
    vec_dot_q4k_batch(row7, q8s, in_f, out7);
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q6k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;

    // Pre-expand Q6 nibbles + high bits into flat[b*256..b*256+256] (values 0–63).
    // Layout: flat[b*256 + half*128 + ci*32 .. +32] = q6 column ci for that half
    // (matches the AVX2 single-token column layout; also usable by scalar per sub-block).
    let mut d_arr = vec![0f32; nb];
    let mut scales_arr = vec![0i8; nb * 16];
    let mut q6_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        d_arr[b] = rdf16(&blk[208..210]);
        for i in 0..16 {
            scales_arr[b * 16 + i] = scales[i] as i8;
        }
        let flat = &mut q6_flat[b * 256..b * 256 + 256];
        for half in 0..2usize {
            let (qlo, qho) = (half * 64, half * 32);
            for l in 0..32usize {
                let q1 = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                let q2 = (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                let q3 = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                let q4 = (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                // column-first layout: ci=0→col0(q1), ci=1→col1(q2), ci=2→col2(q3), ci=3→col3(q4)
                flat[half * 128 + l] = q1;
                flat[half * 128 + 32 + l] = q2;
                flat[half * 128 + 64 + l] = q3;
                flat[half * 128 + 96 + l] = q4;
            }
        }
    }

    // Per-token: accumulate sumi/bsum per sub-block, apply int8 scales.
    // sub_off(sub) = (sub/8)*128 + ((sub%8)/2)*32 + ((sub%8)%2)*16
    // This matches both flat q6 and q8.qs layout (same 16-element stride).
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut sumi = [0i32; 16];
            let mut bsum = [0i32; 16];
            for sub in 0..16usize {
                let sub_off = (sub / 8) * 128 + ((sub % 8) / 2) * 32 + ((sub % 8) % 2) * 16;
                let q6_ptr = &q6_flat[b * 256 + sub_off..b * 256 + sub_off + 16];
                let q8_ptr = &q8.qs[b * 256 + sub_off..b * 256 + sub_off + 16];
                for i in 0..16 {
                    let q6v = q6_ptr[i] as i32;
                    let v = q8_ptr[i] as i32;
                    sumi[sub] += q6v * v;
                    bsum[sub] += v;
                }
            }
            // Integer epilogue — see `vec_dot_q6k_scalar`.
            let mut s = 0i32;
            for sub in 0..16 {
                s += scales_arr[b * 16 + sub] as i32 * (sumi[sub] - 32 * bsum[sub]);
            }
            sumf += d_arr[b] * q8.d[b] * s as f32;
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q6k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;

    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);
    let mask_30 = _mm256_set1_epi8(0x30_u8 as i8);
    let mask_03 = _mm256_set1_epi8(0x03_u8 as i8);
    let ones_u8 = _mm256_set1_epi8(1i8);
    let ones_i16 = _mm256_set1_epi16(1i16);

    // Pre-expand all Q6 nibbles using AVX2 shifts (same ops as single-token, done once).
    let mut d_arr = vec![0f32; nb];
    let mut scales_arr = vec![0i8; nb * 16];
    let mut q6_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        d_arr[b] = rdf16(&blk[208..210]);
        for i in 0..16 {
            scales_arr[b * 16 + i] = scales[i] as i8;
        }
        let flat = &mut q6_flat[b * 256..b * 256 + 256];

        for half in 0..2usize {
            let qlo = half * 64;
            let qho = half * 32;

            let qh_ymm = load256(qh, qho);
            let ql_lo = load256(ql, qlo);
            let ql_hi = load256(ql, qlo + 32);

            let qh_sr2 = _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_03);

            let q6_c0 = _mm256_or_si256(
                _mm256_and_si256(ql_lo, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(qh_ymm, mask_03), 4),
            );
            let q6_c2 = _mm256_or_si256(
                _mm256_and_si256(ql_hi, mask_0f),
                _mm256_slli_epi16(qh_sr2, 4),
            );
            let q6_c4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_0f),
                _mm256_and_si256(qh_ymm, mask_30),
            );
            let q6_c6 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_0f),
                _mm256_and_si256(_mm256_srli_epi16(qh_ymm, 2), mask_30),
            );

            // Store columns contiguously: flat[half*128 + ci*32..+32]
            _mm256_storeu_si256(flat[half * 128..].as_mut_ptr() as *mut __m256i, q6_c0);
            _mm256_storeu_si256(flat[half * 128 + 32..].as_mut_ptr() as *mut __m256i, q6_c2);
            _mm256_storeu_si256(flat[half * 128 + 64..].as_mut_ptr() as *mut __m256i, q6_c4);
            _mm256_storeu_si256(flat[half * 128 + 96..].as_mut_ptr() as *mut __m256i, q6_c6);
        }
    }

    // Per-token: dot each column (32 elements) with q8, compute bsum simultaneously.
    // Identical column/scale accumulation order as single-token AVX2 Q6K → bit-identical.
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let d = d_arr[b];
            let flat = &q6_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];

            let mut s = 0i32; // integer epilogue — see `vec_dot_q6k_scalar`
            for half in 0..2usize {
                let sco = half * 8;
                let base = half * 128;

                for ci in 0..4usize {
                    let q6_ymm = load256(flat, half * 128 + ci * 32);
                    let q8_ymm = load256(q8b, base + ci * 32);

                    let prod = _mm256_maddubs_epi16(q6_ymm, q8_ymm);
                    let sum32 = _mm256_madd_epi16(prod, ones_i16);
                    let bsum_i16 = _mm256_maddubs_epi16(ones_u8, q8_ymm);
                    let bsum_i32 = _mm256_madd_epi16(bsum_i16, ones_i16);

                    let sum_lo = _mm256_castsi256_si128(sum32);
                    let sum_hi = _mm256_extracti128_si256::<1>(sum32);
                    let bs_lo = _mm256_castsi256_si128(bsum_i32);
                    let bs_hi = _mm256_extracti128_si256::<1>(bsum_i32);

                    let iprod_0 = hadd_i32_xmm(sum_lo);
                    let iprod_1 = hadd_i32_xmm(sum_hi);
                    let bs_0 = hadd_i32_xmm(bs_lo);
                    let bs_1 = hadd_i32_xmm(bs_hi);

                    let sub_0 = sco + ci * 2;
                    let sub_1 = sco + ci * 2 + 1;
                    s += scales_arr[b * 16 + sub_0] as i32 * (iprod_0 - 32 * bs_0);
                    s += scales_arr[b * 16 + sub_1] as i32 * (iprod_1 - 32 * bs_1);
                }
            }
            sumf += d * q8.d[b] * s as f32;
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q6k_batch_vnni(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;

    let mask_0f_z = _mm512_set1_epi8(0x0F_u8 as i8);
    let mask_30_z = _mm512_set1_epi8(0x30_u8 as i8);
    let mask_03_z = _mm512_set1_epi8(0x03_u8 as i8);

    // Pre-expand Q6 both halves simultaneously via zmm, store to q6_flat.
    // flat[b*256 + half*128 + ci*32..+32] = q6 column ci for half.
    let mut d_arr = vec![0f32; nb];
    let mut scales_arr = vec![0i8; nb * 16];
    let mut q6_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        d_arr[b] = rdf16(&blk[208..210]);
        for i in 0..16 {
            scales_arr[b * 16 + i] = scales[i] as i8;
        }
        let flat = &mut q6_flat[b * 256..b * 256 + 256];

        // qh is 64 contiguous bytes for both halves → single zmm load.
        let qh_z = load512(qh, 0);

        let ql_lo_h0 = load256(ql, 0);
        let ql_lo_h1 = load256(ql, 64);
        let ql_lo_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_lo_h0), ql_lo_h1);
        let ql_hi_h0 = load256(ql, 32);
        let ql_hi_h1 = load256(ql, 96);
        let ql_hi_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ql_hi_h0), ql_hi_h1);

        let qh_sr2_z = _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_03_z);

        let q6_c0_z = _mm512_or_si512(
            _mm512_and_si512(ql_lo_z, mask_0f_z),
            _mm512_slli_epi16(_mm512_and_si512(qh_z, mask_03_z), 4),
        );
        let q6_c2_z = _mm512_or_si512(
            _mm512_and_si512(ql_hi_z, mask_0f_z),
            _mm512_slli_epi16(qh_sr2_z, 4),
        );
        let q6_c4_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_lo_z, 4), mask_0f_z),
            _mm512_and_si512(qh_z, mask_30_z),
        );
        let q6_c6_z = _mm512_or_si512(
            _mm512_and_si512(_mm512_srli_epi16(ql_hi_z, 4), mask_0f_z),
            _mm512_and_si512(_mm512_srli_epi16(qh_z, 2), mask_30_z),
        );

        // Each zmm has h0 in lower 256 and h1 in upper 256.
        // Store as [h0(ci), h1(ci)] per column → deinterleave into two 32-byte stores.
        for (ci, q6_z) in [q6_c0_z, q6_c2_z, q6_c4_z, q6_c6_z].iter().enumerate() {
            let h0 = _mm512_castsi512_si256(*q6_z);
            let h1 = _mm512_extracti64x4_epi64::<1>(*q6_z);
            _mm256_storeu_si256(flat[ci * 32..].as_mut_ptr() as *mut __m256i, h0);
            _mm256_storeu_si256(flat[128 + ci * 32..].as_mut_ptr() as *mut __m256i, h1);
        }
    }

    // Per-token: `flat` is plain linear element order (see the store above: sub-block s = elems
    // 16s..16s+16), so each super-block is 4 straight 64-byte chunks. Each chunk's madd yields
    // 16 i32 lanes = 4 sub-blocks × 4 lanes; collapse in-register (2 permute+add steps per chunk,
    // 3 cross-chunk permutes) into ONE zmm holding the 16 per-sub-block dots in order — replacing
    // the 32-extract + 32-hadd storm the old loop paid per (row, super-block). The `-32` offset
    // correction reads `q8.bsums16` (computed once at quantize time) instead of re-deriving the
    // activation sums per weight row — the same values, so bit-identity holds, minus 262k
    // redundant recomputes per lm_head GEMM. The f32 epilogue is UNCHANGED (ascending sub-block
    // chain, then d·d8·s). Rows go in PAIRS sharing the weight loads so the two sequential f32
    // chains overlap in the OoO window instead of exposing their latency back to back.
    let idx_pair = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    let idx_half = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    let idx_q14 = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 28, 24, 20, 16, 12, 8, 4, 0);
    let idx_cat = _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0);

    // One row's integer dots for super-block `b` against the four preloaded weight chunks:
    // sumi16[s] = Σ_{i∈sub-block s} q6[i]·q8[i] (q6 still biased +32 — corrected in the epilogue).
    macro_rules! q6k_sumi16 {
        ($q8:expr, $b:expr, $w0:expr, $w1:expr, $w2:expr, $w3:expr) => {{
            let qs = &$q8.qs[$b * 256..$b * 256 + 256];
            let a0 = load512(qs, 0);
            let a1 = load512(qs, 64);
            let a2 = load512(qs, 128);
            let a3 = load512(qs, 192);
            // dpbusd: one op per 64-byte chunk instead of the maddubs+madd pair —
            // integer-exact both ways (u8≤63 × i8 groups can't saturate the i16 pairs).
            let z = _mm512_setzero_si512();
            let s0 = _mm512_dpbusd_epi32(z, $w0, a0);
            let s1 = _mm512_dpbusd_epi32(z, $w1, a1);
            let s2 = _mm512_dpbusd_epi32(z, $w2, a2);
            let s3 = _mm512_dpbusd_epi32(z, $w3, a3);
            let c0 = _mm512_add_epi32(s0, _mm512_permutexvar_epi32(idx_pair, s0));
            let c0 = _mm512_add_epi32(c0, _mm512_permutexvar_epi32(idx_half, c0));
            let c1 = _mm512_add_epi32(s1, _mm512_permutexvar_epi32(idx_pair, s1));
            let c1 = _mm512_add_epi32(c1, _mm512_permutexvar_epi32(idx_half, c1));
            let c2 = _mm512_add_epi32(s2, _mm512_permutexvar_epi32(idx_pair, s2));
            let c2 = _mm512_add_epi32(c2, _mm512_permutexvar_epi32(idx_half, c2));
            let c3 = _mm512_add_epi32(s3, _mm512_permutexvar_epi32(idx_pair, s3));
            let c3 = _mm512_add_epi32(c3, _mm512_permutexvar_epi32(idx_half, c3));
            let lo = _mm512_permutex2var_epi32(c0, idx_q14, c1); // subs 0..7 in lanes 0..7
            let hi = _mm512_permutex2var_epi32(c2, idx_q14, c3); // subs 8..15 in lanes 0..7
            _mm512_permutex2var_epi32(lo, idx_cat, hi)
        }};
    }
    // INTEGER epilogue in SIMD (see `vec_dot_q6k_scalar`): `Σ_s sc_s·(dp_s − 32·bsum_s)` fully
    // in 16 i32 lanes (mullo + one reduce), replacing the order-pinned 16-step scalar f32 chain
    // that used to dominate this kernel's critical path.
    macro_rules! q6k_epilogue {
        ($q8:expr, $b:expr, $sumi_z:expr, $scales_z:expr, $sumf:expr) => {{
            let bs = load512(&$q8.bsums16, $b * 16);
            let corr = _mm512_sub_epi32($sumi_z, _mm512_slli_epi32::<5>(bs));
            let isum = _mm512_reduce_add_epi32(_mm512_mullo_epi32(corr, $scales_z));
            $sumf += d_arr[$b] * $q8.d[$b] * isum as f32;
        }};
    }

    for rp in 0..m / 2 {
        let (ra, rb) = (2 * rp, 2 * rp + 1);
        let (q8a, q8b) = (&q8s[ra], &q8s[rb]);
        let (mut sumf_a, mut sumf_b) = (0f32, 0f32);
        for b in 0..nb {
            let flat = &q6_flat[b * 256..b * 256 + 256];
            let w0 = load512(flat, 0);
            let w1 = load512(flat, 64);
            let w2 = load512(flat, 128);
            let w3 = load512(flat, 192);
            let sc_z = _mm512_cvtepi8_epi32(load128(&scales_arr, b * 16));
            let sumi_a = q6k_sumi16!(q8a, b, w0, w1, w2, w3);
            let sumi_b = q6k_sumi16!(q8b, b, w0, w1, w2, w3);
            q6k_epilogue!(q8a, b, sumi_a, sc_z, sumf_a);
            q6k_epilogue!(q8b, b, sumi_b, sc_z, sumf_b);
        }
        out[ra] = sumf_a;
        out[rb] = sumf_b;
    }
    if m % 2 == 1 {
        let r = m - 1;
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let flat = &q6_flat[b * 256..b * 256 + 256];
            let w0 = load512(flat, 0);
            let w1 = load512(flat, 64);
            let w2 = load512(flat, 128);
            let w3 = load512(flat, 192);
            let sc_z = _mm512_cvtepi8_epi32(load128(&scales_arr, b * 16));
            let sumi = q6k_sumi16!(q8, b, w0, w1, w2, w3);
            q6k_epilogue!(q8, b, sumi, sc_z, sumf);
        }
        out[r] = sumf;
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q6k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q6k_batch: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q6k_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q6k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q6k_batch_scalar(row, q8s, in_f, out);
}

/// Batched `Σ weight·x` for one IQ4_XS row against `m` Q8 activations (`out[r]`). Dispatches
/// avx2 → scalar. Mirrors `vec_dot_q6k_batch`: the weight row's codebook lookup (nibble unpack +
/// `pshufb`) is done ONCE into a flat signed-i8 buffer, then reused across all `m` token
/// activations — amortising the decode the single-token path would repeat per token.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq4xs_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq4xs_batch: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_iq4xs_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_iq4xs_batch_scalar(row, q8s, in_f, out);
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_iq4xs_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;
    // Pre-expand codebook weights (signed i8), (ls−32) scales, and d ONCE per super-block.
    let mut d_arr = vec![0f32; nb];
    let mut ls_arr = vec![0i32; nb * 8];
    let mut w_flat = vec![0i8; nb * 256];
    for b in 0..nb {
        let blk = &row[b * 136..b * 136 + 136];
        d_arr[b] = rdf16(&blk[0..2]);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        for ib in 0..8usize {
            ls_arr[b * 8 + ib] = iq4xs_ls_minus_32(scales_h, scales_l, ib);
            let qsb = &qs[ib * 16..ib * 16 + 16];
            let dst = &mut w_flat[b * 256 + ib * 32..b * 256 + ib * 32 + 32];
            for j in 0..16usize {
                dst[j] = KVALUES_IQ4NL[(qsb[j] & 0xF) as usize];
                dst[j + 16] = KVALUES_IQ4NL[(qsb[j] >> 4) as usize];
            }
        }
    }
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut isum = 0i32;
            for ib in 0..8usize {
                let w = &w_flat[b * 256 + ib * 32..b * 256 + ib * 32 + 32];
                let a = &q8b[ib * 32..ib * 32 + 32];
                let mut dot = 0i32;
                for i in 0..32usize {
                    dot += w[i] as i32 * a[i] as i32;
                }
                isum += ls_arr[b * 8 + ib] * dot;
            }
            sumf += d_arr[b] * q8.d[b] * isum as f32;
        }
        out[r] = sumf;
    }
}

/// AVX2 batch kernel for IQ4_XS: expand the weight row's codebook values once (same nibble+pshufb
/// as single-token), store signed i8 to `w_flat`, then per token run the Q8_0 abs/sign maddubs dot.
/// Integer dot is order-free, and the epilogue formula matches the scalar path → bit-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_iq4xs_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);
    let ones_i16 = _mm256_set1_epi16(1i16);
    let table = _mm256_broadcastsi128_si256(load128(&KVALUES_IQ4NL, 0));

    let mut d_arr = vec![0f32; nb];
    let mut ls_arr = vec![0i32; nb * 8];
    let mut w_flat = vec![0i8; nb * 256];
    for b in 0..nb {
        let blk = &row[b * 136..b * 136 + 136];
        d_arr[b] = rdf16(&blk[0..2]);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        for ib in 0..8usize {
            ls_arr[b * 8 + ib] = iq4xs_ls_minus_32(scales_h, scales_l, ib);
            let codes16 = load128(qs, ib * 16);
            let lo =
                _mm256_castsi128_si256(_mm_and_si128(codes16, _mm256_castsi256_si128(mask_0f)));
            let hi = _mm256_castsi128_si256(_mm_and_si128(
                _mm_srli_epi16(codes16, 4),
                _mm256_castsi256_si128(mask_0f),
            ));
            let codes = _mm256_inserti128_si256::<1>(lo, _mm256_castsi256_si128(hi));
            let w = _mm256_shuffle_epi8(table, codes);
            _mm256_storeu_si256(w_flat[b * 256 + ib * 32..].as_mut_ptr() as *mut __m256i, w);
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut isum = 0i32;
            for ib in 0..8usize {
                let w = load256(&w_flat, b * 256 + ib * 32);
                let a = load256(q8b, ib * 32);
                let w_abs = _mm256_abs_epi8(w);
                let a_signed = _mm256_sign_epi8(a, w);
                let prod = _mm256_maddubs_epi16(w_abs, a_signed);
                let sum32 = _mm256_madd_epi16(prod, ones_i16);
                let dot = hadd_i32_ymm(sum32);
                isum += ls_arr[b * 8 + ib] * dot;
            }
            sumf += d_arr[b] * q8.d[b] * isum as f32;
        }
        out[r] = sumf;
    }
}

/// Expand a whole IQ2_S weight row (82 bytes / 256 elems per super-block) to signed `i8` codebook
/// weights ONCE, alongside the per-16-element `dl` scales — mirrors `dequant_block(DType::Iq2S)`
/// exactly. IQ2_S is a GRID-codebook quant: each group of 8 weights is a 10-bit index into
/// `IQ2S_GRID` (each entry packs 8 signed i8), then sign-flipped per `KMASK_IQ2XS` from a per-group
/// sign byte. The grid gather can't be SIMD-vectorized, so the expansion is scalar here and the
/// caller runs the int dot against the Q8 activation (amortising this decode across all `m` tokens,
/// like `vec_dot_q6k_batch` decodes its row once). `weights[b*256 + g*8 + k]` and `dls[b*32 + g]`
/// are in the SAME element order the dequant writes: groups iterate `ib32` (0..8) then `l` (0..4),
/// 8 elements each. `in_f` must be a multiple of 256.
fn iq2s_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>) {
    use infr_core::iquant_grids::{IQ2S_GRID, KMASK_IQ2XS};
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut dls = vec![0f32; nb * 32];
    for b in 0..nb {
        let blk = &row[b * 82..b * 82 + 82];
        let d = rdf16(&blk[0..2]);
        let qs_all = &blk[2..66]; // 64 bytes: [0..32] grid-idx low bytes, [32..64] sign bytes
        let qh = &blk[66..74]; // 8 bytes: 2-bit high index bits, one byte per ib32
        let scales = &blk[74..82]; // 8 bytes: two 4-bit scales per byte
        for ib32 in 0..8usize {
            let sc = scales[ib32];
            let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
            let qh_byte = qh[ib32];
            for l in 0..4usize {
                let g = ib32 * 4 + l; // group index within block (0..32)
                let qs_byte = qs_all[ib32 * 4 + l];
                let sign_byte = qs_all[ib32 * 4 + l + 32];
                let hi = ((qh_byte as u32).wrapping_shl((8 - 2 * l) as u32)) & 0x300;
                let grid_idx = (qs_byte as usize) | (hi as usize);
                let grid_u64 = IQ2S_GRID[grid_idx];
                let dl = if l < 2 { db0 } else { db1 };
                dls[b * 32 + g] = dl;
                let dst = &mut weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                for k in 0..8usize {
                    let gv = ((grid_u64 >> (8 * k)) & 0xFF) as i8;
                    // Negate per KMASK_IQ2XS[k], matching `apply_signs` in dequant.rs.
                    dst[k] = if sign_byte & KMASK_IQ2XS[k] != 0 {
                        gv.wrapping_neg()
                    } else {
                        gv
                    };
                }
            }
        }
    }
    (weights, dls)
}

/// `Σ weight·x` for one IQ2_S row against the Q8 activation. Expands the grid-codebook weight row to
/// signed i8 once (`iq2s_expand_row`), then per group of 8 runs an integer dot scaled by the per-16
/// `dl`; the block's `dl·iprod` terms accumulate in an f32 running sum, then ONE multiply by the
/// super-block activation scale `q8.d[b]`. No offset/min/bsum term — grid weights are already signed
/// and zero-centred. Same per-sub-block→super-block accumulation shape as `vec_dot_iq4xs`, minus the
/// SIMD dot (the grid gather is inherently scalar). `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq2s(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq2s: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, dls) = iq2s_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut block_sum = 0f32;
        for g in 0..32usize {
            let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
            let a = &q8b[g * 8..g * 8 + 8];
            let mut iprod = 0i32;
            for k in 0..8usize {
                iprod += w[k] as i32 * a[k] as i32;
            }
            block_sum += dls[b * 32 + g] * iprod as f32;
        }
        sumf += q8.d[b] * block_sum;
    }
    sumf
}

/// Batched `Σ weight·x` for one IQ2_S row against `m` Q8 activations (`out[r]`). Expands the
/// grid-codebook weight row to signed i8 ONCE (`iq2s_expand_row`), then reuses it across all `m`
/// token activations — the amortisation the single-token path can't do. Bit-identical accumulation
/// to `vec_dot_iq2s` (same group order, same f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq2s_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq2s_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, dls) = iq2s_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut block_sum = 0f32;
            for g in 0..32usize {
                let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                let a = &q8b[g * 8..g * 8 + 8];
                let mut iprod = 0i32;
                for k in 0..8usize {
                    iprod += w[k] as i32 * a[k] as i32;
                }
                block_sum += dls[b * 32 + g] * iprod as f32;
            }
            sumf += q8.d[b] * block_sum;
        }
        out[r] = sumf;
    }
}

/// Expand a whole IQ2_XXS weight row (66 bytes / 256 elems per super-block) to signed `i8` codebook
/// weights ONCE, alongside the per-32-element `db` scales — mirrors `dequant_block(DType::Iq2Xxs)`
/// exactly. IQ2_XXS is a GRID-codebook quant: each super-block is 8 sub-blocks of 32; for sub-block
/// `ib32` two LE32 words `aux0`/`aux1` are read from `qs[ib32*8..]` — `aux0` holds four 8-bit indices
/// into the 256-entry `IQ2XXS_GRID` (each entry packs 8 signed i8), `aux1` holds four 7-bit sign
/// indices (bits 0..27) plus the 4-bit scale magnitude (bits 28..31). Per-32 scale
/// `db = d * (0.5 + (aux1>>28)) * 0.25`, and each group of 8 is sign-flipped per `KMASK_IQ2XS` from
/// `KSIGNS_IQ2XS[sign_idx]`. The grid gather can't be SIMD-vectorized, so the expansion is scalar and
/// the caller runs the int dot against the Q8 activation, amortising this decode across all `m`
/// tokens. `weights[b*256 + g*8 + k]` and `dls[b*32 + g]` are in the SAME element order the dequant
/// writes (`outoff` progression: groups iterate `ib32` (0..8) then `l` (0..4), 8 elements each);
/// `dls[g]` is the group's `db` (same `db` for the 4 groups-of-8 inside each 32-sub-block). `in_f`
/// must be a multiple of 256.
fn iq2xxs_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>) {
    use infr_core::iquant_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut dls = vec![0f32; nb * 32];
    for b in 0..nb {
        let blk = &row[b * 66..b * 66 + 66];
        let d = rdf16(&blk[0..2]);
        let qs = &blk[2..66]; // 64 bytes: 8 sub-blocks × (aux0[4] grid idx + aux1[4] sign+scale)
        for ib32 in 0..8usize {
            let off = ib32 * 8;
            let aux0 = u32::from_le_bytes(qs[off..off + 4].try_into().unwrap());
            let aux1 = u32::from_le_bytes(qs[off + 4..off + 8].try_into().unwrap());
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            let aux0_bytes = aux0.to_le_bytes();
            for l in 0..4usize {
                let g = ib32 * 4 + l; // group index within block (0..32)
                let grid_idx = aux0_bytes[l] as usize; // 8-bit → IQ2XXS_GRID[256]
                let sign_idx = ((aux1 >> (7 * l)) & 127) as usize;
                let grid_u64 = IQ2XXS_GRID[grid_idx];
                let sign_byte = KSIGNS_IQ2XS[sign_idx];
                dls[b * 32 + g] = db;
                let dst = &mut weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                for k in 0..8usize {
                    let gv = ((grid_u64 >> (8 * k)) & 0xFF) as i8;
                    // Negate per KMASK_IQ2XS[k], matching `apply_signs` in dequant.rs.
                    dst[k] = if sign_byte & KMASK_IQ2XS[k] != 0 {
                        gv.wrapping_neg()
                    } else {
                        gv
                    };
                }
            }
        }
    }
    (weights, dls)
}

/// `Σ weight·x` for one IQ2_XXS row against the Q8 activation. Expands the grid-codebook weight row
/// to signed i8 once (`iq2xxs_expand_row`), then per group of 8 runs an integer dot scaled by the
/// per-32 `db`; the block's `db·iprod` terms accumulate in an f32 running sum, then ONE multiply by
/// the super-block activation scale `q8.d[b]`. No offset/min/bsum term — grid weights are already
/// signed and zero-centred. Same per-group→super-block accumulation shape as `vec_dot_iq2s` (the
/// grid gather is inherently scalar). `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq2xxs(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq2xxs: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, dls) = iq2xxs_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut block_sum = 0f32;
        for g in 0..32usize {
            let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
            let a = &q8b[g * 8..g * 8 + 8];
            let mut iprod = 0i32;
            for k in 0..8usize {
                iprod += w[k] as i32 * a[k] as i32;
            }
            block_sum += dls[b * 32 + g] * iprod as f32;
        }
        sumf += q8.d[b] * block_sum;
    }
    sumf
}

/// Batched `Σ weight·x` for one IQ2_XXS row against `m` Q8 activations (`out[r]`). Expands the
/// grid-codebook weight row to signed i8 ONCE (`iq2xxs_expand_row`), then reuses it across all `m`
/// token activations — the amortisation the single-token path can't do. Bit-identical accumulation
/// to `vec_dot_iq2xxs` (same group order, same f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq2xxs_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq2xxs_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, dls) = iq2xxs_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut block_sum = 0f32;
            for g in 0..32usize {
                let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                let a = &q8b[g * 8..g * 8 + 8];
                let mut iprod = 0i32;
                for k in 0..8usize {
                    iprod += w[k] as i32 * a[k] as i32;
                }
                block_sum += dls[b * 32 + g] * iprod as f32;
            }
            sumf += q8.d[b] * block_sum;
        }
        out[r] = sumf;
    }
}

/// Expand a whole TQ2_0 weight row (66 bytes / 256 elems per super-block) to signed `i8` ternary
/// weights ONCE, alongside the per-256 `d` scale — mirrors `dequant_block(DType::Tq2_0)` (dequant.rs)
/// EXACTLY. TQ2_0 (BitNet/TriLM ternary) is the SIMPLEST k-quant-sized format: ONE f16 scale `d` per
/// 256-block, no sub-block scales, no grid, no signs; each 2-bit code `q ∈ 0..3` dequants to
/// `y = (q − 1)·d`. Folding the `−1` into the stored i8 weight (`(q − 1) ∈ {−1,0,1,2}`) means the dot
/// needs NO offset/bsum correction term — a plain signed int dot × `d` × `q8.d[b]`. Element order
/// mirrors the dequant's `outoff` progression EXACTLY: two 32-byte chunks (chunk 0 → `qs[0..32]`,
/// chunk 1 → `qs[32..64]`), then `l in 0..4`, then `m in 0..32`, output element index
/// `chunk*128 + l*32 + m`. Block = 66 bytes: `[u8 qs[64]]`, `[f16 d]`. `in_f` must be a multiple of
/// 256; `ds[b]` is the block's `d` (one per super-block, unlike the per-32 grid scales).
fn tq2_0_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>) {
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut ds = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * 66..b * 66 + 66];
        let qs = &blk[0..64];
        ds[b] = rdf16(&blk[64..66]);
        let base = b * 256;
        let mut outoff = 0usize;
        // Two 32-byte chunks (chunk 0 → qs[0..32], chunk 1 → qs[32..64]).
        for chunk in 0..2usize {
            let j = chunk * 32;
            for l in 0..4usize {
                for m in 0..32usize {
                    let q = ((qs[j + m] >> (l * 2)) & 3) as i8; // ∈ 0..3
                    weights[base + outoff] = q - 1; // (q − 1) ∈ {−1,0,1,2}
                    outoff += 1;
                }
            }
        }
    }
    (weights, ds)
}

/// `Σ weight·x` for one TQ2_0 row against the Q8 activation. Expands the ternary weight row to signed
/// i8 once (`tq2_0_expand_row`, `−1` already folded in), then per 256-block runs ONE integer dot and
/// scales by `d · q8.d[b]` — a single term per super-block, no min/bsum correction (the weights are
/// already zero-centred). `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_tq2_0(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_tq2_0: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, ds) = tq2_0_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let w = &weights[b * 256..b * 256 + 256];
        let a = &q8.qs[b * 256..b * 256 + 256];
        let mut iprod = 0i32;
        for i in 0..256usize {
            iprod += w[i] as i32 * a[i] as i32;
        }
        sumf += ds[b] * q8.d[b] * iprod as f32;
    }
    sumf
}

/// Batched `Σ weight·x` for one TQ2_0 row against `m` Q8 activations (`out[r]`). Expands the ternary
/// weight row to signed i8 ONCE (`tq2_0_expand_row`), then reuses it across all `m` token activations
/// — the amortisation the single-token path can't do. Bit-identical accumulation to `vec_dot_tq2_0`
/// (same 256-block int dot, same `d·q8.d[b]` f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_tq2_0_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_tq2_0_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, ds) = tq2_0_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let w = &weights[b * 256..b * 256 + 256];
            let a = &q8.qs[b * 256..b * 256 + 256];
            let mut iprod = 0i32;
            for i in 0..256usize {
                iprod += w[i] as i32 * a[i] as i32;
            }
            sumf += ds[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Expand a whole TQ1_0 weight row (54 bytes / 256 elems per super-block) to signed `i8` ternary
/// weights ONCE, alongside the per-256 `d` scale — mirrors `dequant_block(DType::Tq1_0)` (dequant.rs)
/// EXACTLY. TQ1_0 (BitNet/TriLM 1.6-bit ternary) is the base-3-PACKED sibling of TQ2_0: ONE f16 scale
/// `d` per 256-block, no sub-block scales, no grid, no signs; each ternary digit `digit ∈ 0..2`
/// dequants to `y = (digit − 1)·d`. Folding the `−1` into the stored i8 weight (`(digit − 1) ∈
/// {−1,0,1}`) means the dot needs NO offset/bsum correction term — a plain signed int dot × `d` ×
/// `q8.d[b]`. Digit extraction packs 5 ternary digits per byte: for byte `b` and digit index `n`,
/// `q = b·POW3[n]` (wrapping u8), `digit = ((q as u16)·3) >> 8`. Element order mirrors the dequant's
/// `outoff` progression EXACTLY — THREE segments: (1) `qs[0..32]`, 5 digit passes → 160 elems
/// (`n*32 + m`); (2) `qs[32..48]`, 5 digit passes → 80 elems (`160 + n*16 + m`); (3) `qh[0..4]`, 4
/// digit passes → 16 elems (`240 + n*4 + j`). Block = 54 bytes: `[u8 qs[48]]`, `[u8 qh[4]]`,
/// `[f16 d]`. `in_f` must be a multiple of 256; `ds[b]` is the block's `d`.
fn tq1_0_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut ds = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * 54..b * 54 + 54];
        let qs = &blk[0..48];
        let qh = &blk[48..52];
        ds[b] = rdf16(&blk[52..54]);
        let base = b * 256;
        // Segment 1: qs[0..32], 5 digit passes → 160 elems (outoff 0..160).
        for n in 0..5usize {
            let p3 = POW3[n];
            for m in 0..32usize {
                let q = qs[m].wrapping_mul(p3);
                let digit = (((q as u16) * 3) >> 8) as i8; // ∈ 0..2
                weights[base + n * 32 + m] = digit - 1; // ∈ {−1,0,1}
            }
        }
        // Segment 2: qs[32..48], 5 digit passes → 80 elems (outoff 160..240).
        for n in 0..5usize {
            let p3 = POW3[n];
            for m in 0..16usize {
                let q = qs[32 + m].wrapping_mul(p3);
                let digit = (((q as u16) * 3) >> 8) as i8; // ∈ 0..2
                weights[base + 160 + n * 16 + m] = digit - 1; // ∈ {−1,0,1}
            }
        }
        // Segment 3: qh[0..4], 4 digit passes → 16 elems (outoff 240..256).
        for n in 0..4usize {
            let p3 = POW3[n];
            for j in 0..4usize {
                let q = qh[j].wrapping_mul(p3);
                let digit = (((q as u16) * 3) >> 8) as i8; // ∈ 0..2
                weights[base + 240 + n * 4 + j] = digit - 1; // ∈ {−1,0,1}
            }
        }
    }
    (weights, ds)
}

/// `Σ weight·x` for one TQ1_0 row against the Q8 activation. Expands the ternary weight row to signed
/// i8 once (`tq1_0_expand_row`, `−1` already folded in), then per 256-block runs ONE integer dot and
/// scales by `d · q8.d[b]` — a single term per super-block, no min/bsum correction (the weights are
/// already zero-centred). `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_tq1_0(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_tq1_0: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, ds) = tq1_0_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let w = &weights[b * 256..b * 256 + 256];
        let a = &q8.qs[b * 256..b * 256 + 256];
        let mut iprod = 0i32;
        for i in 0..256usize {
            iprod += w[i] as i32 * a[i] as i32;
        }
        sumf += ds[b] * q8.d[b] * iprod as f32;
    }
    sumf
}

/// Batched `Σ weight·x` for one TQ1_0 row against `m` Q8 activations (`out[r]`). Expands the ternary
/// weight row to signed i8 ONCE (`tq1_0_expand_row`), then reuses it across all `m` token activations
/// — the amortisation the single-token path can't do. Bit-identical accumulation to `vec_dot_tq1_0`
/// (same 256-block int dot, same `d·q8.d[b]` f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_tq1_0_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_tq1_0_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, ds) = tq1_0_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let w = &weights[b * 256..b * 256 + 256];
            let a = &q8.qs[b * 256..b * 256 + 256];
            let mut iprod = 0i32;
            for i in 0..256usize {
                iprod += w[i] as i32 * a[i] as i32;
            }
            sumf += ds[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Expand a whole IQ1_S weight row (50 bytes / 256 elems per super-block) to signed `i8` grid
/// weights ONCE — mirrors `dequant_block(DType::Iq1S)` (dequant.rs) EXACTLY. IQ1_S is a
/// grid-codebook quant with a per-32 fractional `delta` that is NOT part of the signed codebook, so
/// unlike IQ2_XXS it can't be folded into the i8 weights: this fn stores the raw signed grid value
/// (`g_j`, delta NOT baked in) plus the per-8-group `dl` scale and `delta` offset (both constant
/// across the 4 groups-of-8 inside a sub-block, i.e. per-32). Block = 50 bytes: `[f16 d]` (2),
/// `[u8 qs[32]]`, `[u8 qh[16]]` (= 8 × `u16` LE). 8 sub-blocks (`ib`); per sub-block
/// `dl = d*(2*((qh>>12)&7)+1)`, `delta = ±IQ1S_DELTA` per the `0x8000` bit; each of the 4 groups
/// (`l`) reads an 11-bit grid index `qs[l] | (((qh>>3l)&7)<<8)` into the 2048-entry `IQ1S_GRID`
/// (each entry packs 8 signed i8). `weights[b*256 + g*8 + k]`, `dls[b*32 + g]`, `deltas[b*32 + g]`
/// are in the SAME `outoff` element order the dequant writes. `in_f` must be a multiple of 256.
fn iq1s_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    use infr_core::iquant_grids::IQ1S_GRID;
    use infr_gguf::dequant::IQ1S_DELTA;
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut dls = vec![0f32; nb * 32];
    let mut deltas = vec![0f32; nb * 32];
    for b in 0..nb {
        let blk = &row[b * 50..b * 50 + 50];
        let d = rdf16(&blk[0..2]);
        let qs = &blk[2..34]; // 32 bytes
        let qh_raw = &blk[34..50]; // 16 bytes = 8 × u16 LE
        let mut outoff = 0usize;
        let mut qs_off = 0usize;
        for ib in 0..8usize {
            let qh = u16::from_le_bytes(qh_raw[2 * ib..2 * ib + 2].try_into().unwrap());
            // Per-32 scale & delta (constant across the 4 groups-of-8 in this sub-block).
            let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4usize {
                let g = ib * 4 + l; // group index within block (0..32); outoff == g*8
                let grid_idx = qs[qs_off + l] as usize | (((qh >> (3 * l)) & 7) as usize) << 8;
                let grid_u64 = IQ1S_GRID[grid_idx];
                dls[b * 32 + g] = dl;
                deltas[b * 32 + g] = delta;
                let dst = &mut weights[b * 256 + outoff..b * 256 + outoff + 8];
                for (j, dj) in dst.iter_mut().enumerate() {
                    // Raw signed grid i8 — delta is applied at dot time, NOT baked in.
                    *dj = ((grid_u64 >> (8 * j)) & 0xFF) as i8;
                }
                outoff += 8;
            }
            qs_off += 4;
        }
    }
    (weights, dls, deltas)
}

/// `Σ weight·x` for one IQ1_S row against the Q8 activation. Expands the grid row to signed i8 once
/// (`iq1s_expand_row`), then per group of 8 forms `iprod = Σ grid·qa` AND `asum = Σ qa` (the
/// activation sum), accumulating `dl·(iprod + delta·asum)` in an f32 block sum — the delta split of
/// `Σ dl·(grid+delta)·(as·qa) = dl·as·(iprod + delta·asum)`, since the fractional `delta` can't be
/// folded into the i8 grid. ONE multiply by the super-block activation scale `q8.d[b]` per 256.
/// `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq1s(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq1s: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, dls, deltas) = iq1s_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut block_sum = 0f32;
        for g in 0..32usize {
            let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
            let a = &q8b[g * 8..g * 8 + 8];
            let mut iprod = 0i32;
            let mut asum = 0i32;
            for k in 0..8usize {
                iprod += w[k] as i32 * a[k] as i32;
                asum += a[k] as i32;
            }
            block_sum += dls[b * 32 + g] * (iprod as f32 + deltas[b * 32 + g] * asum as f32);
        }
        sumf += q8.d[b] * block_sum;
    }
    sumf
}

/// Batched `Σ weight·x` for one IQ1_S row against `m` Q8 activations (`out[r]`). Expands the grid
/// row to signed i8 ONCE (`iq1s_expand_row`), then reuses it across all `m` token activations — the
/// amortisation the single-token path can't do. Bit-identical accumulation to `vec_dot_iq1s` (same
/// group order, same `iprod`/`asum` split, same f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq1s_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq1s_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, dls, deltas) = iq1s_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut block_sum = 0f32;
            for g in 0..32usize {
                let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                let a = &q8b[g * 8..g * 8 + 8];
                let mut iprod = 0i32;
                let mut asum = 0i32;
                for k in 0..8usize {
                    iprod += w[k] as i32 * a[k] as i32;
                    asum += a[k] as i32;
                }
                block_sum += dls[b * 32 + g] * (iprod as f32 + deltas[b * 32 + g] * asum as f32);
            }
            sumf += q8.d[b] * block_sum;
        }
        out[r] = sumf;
    }
}

/// Expand a whole IQ1_M weight row (56 bytes / 256 elems per super-block) to signed `i8` grid
/// weights ONCE — mirrors `dequant_block(DType::Iq1M)` (dequant.rs) EXACTLY. IQ1_M shares the IQ1_S
/// grid codebook + fractional `delta` (`y = dl·(grid + delta)`, delta NOT part of the signed
/// codebook, so it can't be folded into the i8 weights). It differs from IQ1_S in three ways:
/// (1) NO separate `d` field — `d` is a f16 packed into the high nibble of each of the four `u16`
/// scale words; (2) the `dl` scale varies PER-16 within a sub-block (`dl1` for groups l=0,1 vs
/// `dl2` for l=2,3), not per-32; (3) the `delta` sign varies PER-8 (per group), from two `qh` bytes.
/// Block = 56 bytes: `[u8 qs[32]]` (32), `[u8 qh[16]]` (16), `[u8 scales[8]]` (8 = 4 × `u16` LE).
/// 8 sub-blocks (`ib`); per sub-block `dl1/dl2 = d*(2*((sc[ib/2] >> (6*(ib%2)(+3))) & 7) + 1)`,
/// `qh0 = qh[qh_off]`, `qh1 = qh[qh_off+1]`; each of the 4 groups (`l`) reads an 11-bit grid index
/// `qs[qs_off+l] | ((qh<<(8|4)) & 0x700)` into the 2048-entry `IQ1S_GRID` (each entry packs 8 signed
/// i8) and a per-8 `delta = ±IQ1S_DELTA` per the `0x08`/`0x80` bit. `weights[b*256 + g*8 + k]`,
/// `dls[b*32 + g]`, `deltas[b*32 + g]` are in the SAME `outoff` element order the dequant writes
/// (`g == ib*4 + l`). `in_f` must be a multiple of 256.
fn iq1m_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    use infr_core::iquant_grids::IQ1S_GRID;
    use infr_gguf::dequant::IQ1S_DELTA;
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut dls = vec![0f32; nb * 32];
    let mut deltas = vec![0f32; nb * 32];
    for b in 0..nb {
        let blk = &row[b * 56..b * 56 + 56];
        let qs = &blk[0..32]; // 32 bytes
        let qh = &blk[32..48]; // 16 bytes
        let scales_raw = &blk[48..56]; // 8 bytes = 4 × u16 LE
        let sc = [
            u16::from_le_bytes(scales_raw[0..2].try_into().unwrap()),
            u16::from_le_bytes(scales_raw[2..4].try_into().unwrap()),
            u16::from_le_bytes(scales_raw[4..6].try_into().unwrap()),
            u16::from_le_bytes(scales_raw[6..8].try_into().unwrap()),
        ];
        // `d` is a f16 packed into the high nibble of each scale word.
        let d_bits: u16 =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = half::f16::from_bits(d_bits).to_f32();
        let mut outoff = 0usize;
        let mut qs_off = 0usize;
        let mut qh_off = 0usize;
        for ib in 0..8usize {
            // Per-16 scales: dl1 for groups l=0,1; dl2 for l=2,3.
            let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) as f32 + 1.0);
            let qh0 = qh[qh_off];
            let qh1 = qh[qh_off + 1];
            // Per-8 grid indices (8-bit qs + 3 high bits from qh, mask 0x700).
            let idx = [
                qs[qs_off] as usize | (((qh0 as usize) << 8) & 0x700),
                qs[qs_off + 1] as usize | (((qh0 as usize) << 4) & 0x700),
                qs[qs_off + 2] as usize | (((qh1 as usize) << 8) & 0x700),
                qs[qs_off + 3] as usize | (((qh1 as usize) << 4) & 0x700),
            ];
            // Per-8 delta sign from qh bits 0x08/0x80.
            let delta = [
                if qh0 & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh0 & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh1 & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh1 & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];
            for l in 0..4usize {
                let g = ib * 4 + l; // group index within block (0..32); outoff == g*8
                let dl = if l < 2 { dl1 } else { dl2 };
                let grid_u64 = IQ1S_GRID[idx[l]];
                dls[b * 32 + g] = dl;
                deltas[b * 32 + g] = delta[l];
                let dst = &mut weights[b * 256 + outoff..b * 256 + outoff + 8];
                for (j, dj) in dst.iter_mut().enumerate() {
                    // Raw signed grid i8 — delta is applied at dot time, NOT baked in.
                    *dj = ((grid_u64 >> (8 * j)) & 0xFF) as i8;
                }
                outoff += 8;
            }
            qs_off += 4;
            qh_off += 2;
        }
    }
    (weights, dls, deltas)
}

/// `Σ weight·x` for one IQ1_M row against the Q8 activation. Expands the grid row to signed i8 once
/// (`iq1m_expand_row`), then per group of 8 forms `iprod = Σ grid·qa` AND `asum = Σ qa`, accumulating
/// `dl·(iprod + delta·asum)` in an f32 block sum — the delta split of
/// `Σ dl·(grid+delta)·(as·qa) = dl·as·(iprod + delta·asum)`, since the fractional `delta` can't be
/// folded into the i8 grid. ONE multiply by the super-block activation scale `q8.d[b]` per 256.
/// Bit-identical accumulation to `vec_dot_iq1s`. `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq1m(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq1m: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, dls, deltas) = iq1m_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut block_sum = 0f32;
        for g in 0..32usize {
            let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
            let a = &q8b[g * 8..g * 8 + 8];
            let mut iprod = 0i32;
            let mut asum = 0i32;
            for k in 0..8usize {
                iprod += w[k] as i32 * a[k] as i32;
                asum += a[k] as i32;
            }
            block_sum += dls[b * 32 + g] * (iprod as f32 + deltas[b * 32 + g] * asum as f32);
        }
        sumf += q8.d[b] * block_sum;
    }
    sumf
}

/// Batched `Σ weight·x` for one IQ1_M row against `m` Q8 activations (`out[r]`). Expands the grid row
/// to signed i8 ONCE (`iq1m_expand_row`), then reuses it across all `m` token activations — the
/// amortisation the single-token path can't do. Bit-identical accumulation to `vec_dot_iq1m` (same
/// group order, same `iprod`/`asum` split, same f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq1m_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq1m_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, dls, deltas) = iq1m_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut block_sum = 0f32;
            for g in 0..32usize {
                let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                let a = &q8b[g * 8..g * 8 + 8];
                let mut iprod = 0i32;
                let mut asum = 0i32;
                for k in 0..8usize {
                    iprod += w[k] as i32 * a[k] as i32;
                    asum += a[k] as i32;
                }
                block_sum += dls[b * 32 + g] * (iprod as f32 + deltas[b * 32 + g] * asum as f32);
            }
            sumf += q8.d[b] * block_sum;
        }
        out[r] = sumf;
    }
}

/// Expand a whole IQ2_XS weight row (74 bytes / 256 elems per super-block) to signed `i8` codebook
/// weights ONCE, alongside the per-16-element `dl` scales — mirrors `dequant_codebook(DType::Iq2Xs)`
/// exactly. IQ2_XS is a GRID-codebook quant: each super-block is 8 sub-blocks of 32; sub-block
/// `ib32` carries a scale byte `scales[ib32]` giving `db0 = d*(0.5 + (sc & 0xf))*0.25` and
/// `db1 = d*(0.5 + (sc >> 4))*0.25`. Each sub-block is 4 groups of 8 (`l in 0..4`); group `l` reads
/// a `u16` `qs16` from `qs[(ib32*4 + l)*2..]` whose low 9 bits (`qs16 & 511`) index the 512-entry
/// `IQ2XS_GRID` (each entry packs 8 signed i8) and whose high 7 bits (`qs16 >> 9`) index the
/// 128-entry `KSIGNS_IQ2XS` sign table; the group scale is `dl = if l < 2 { db0 } else { db1 }`
/// (per-16). Each group of 8 is sign-flipped per `KMASK_IQ2XS`. The grid gather can't be
/// SIMD-vectorized, so the expansion is scalar and the caller runs the int dot against the Q8
/// activation, amortising this decode across all `m` tokens. `weights[b*256 + g*8 + k]` and
/// `dls[b*32 + g]` are in the SAME element order the dequant writes (`outoff` progression: groups
/// iterate `ib32` (0..8) then `l` (0..4), 8 elements each). `in_f` must be a multiple of 256.
fn iq2xs_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>) {
    use infr_core::iquant_grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut dls = vec![0f32; nb * 32];
    for b in 0..nb {
        let blk = &row[b * 74..b * 74 + 74];
        let d = rdf16(&blk[0..2]);
        let qs_raw = &blk[2..66]; // 64 bytes: 32 u16 LE grid-idx + 7-bit sign-idx entries
        let scales = &blk[66..74]; // 8 bytes: two 4-bit scales per sub-block
        for ib32 in 0..8usize {
            let sc = scales[ib32];
            let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
            for l in 0..4usize {
                let g = ib32 * 4 + l; // group index within block (0..32)
                let qoff = (ib32 * 4 + l) * 2;
                let qs16 = u16::from_le_bytes(qs_raw[qoff..qoff + 2].try_into().unwrap());
                let grid_idx = (qs16 & 511) as usize; // 9-bit → IQ2XS_GRID[512]
                let sign_idx = (qs16 >> 9) as usize; // 7-bit → KSIGNS_IQ2XS[128]
                let grid_u64 = IQ2XS_GRID[grid_idx];
                let sign_byte = KSIGNS_IQ2XS[sign_idx];
                let dl = if l < 2 { db0 } else { db1 };
                dls[b * 32 + g] = dl;
                let dst = &mut weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                for k in 0..8usize {
                    let gv = ((grid_u64 >> (8 * k)) & 0xFF) as i8;
                    // Negate per KMASK_IQ2XS[k], matching `apply_signs` in dequant.rs.
                    dst[k] = if sign_byte & KMASK_IQ2XS[k] != 0 {
                        gv.wrapping_neg()
                    } else {
                        gv
                    };
                }
            }
        }
    }
    (weights, dls)
}

/// `Σ weight·x` for one IQ2_XS row against the Q8 activation. Expands the grid-codebook weight row
/// to signed i8 once (`iq2xs_expand_row`), then per group of 8 runs an integer dot scaled by the
/// per-16 `dl`; the block's `dl·iprod` terms accumulate in an f32 running sum, then ONE multiply by
/// the super-block activation scale `q8.d[b]`. No offset/min/bsum term — grid weights are already
/// signed and zero-centred. Same per-group→super-block accumulation shape as `vec_dot_iq2s` (the
/// grid gather is inherently scalar). `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq2xs(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq2xs: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, dls) = iq2xs_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut block_sum = 0f32;
        for g in 0..32usize {
            let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
            let a = &q8b[g * 8..g * 8 + 8];
            let mut iprod = 0i32;
            for k in 0..8usize {
                iprod += w[k] as i32 * a[k] as i32;
            }
            block_sum += dls[b * 32 + g] * iprod as f32;
        }
        sumf += q8.d[b] * block_sum;
    }
    sumf
}

/// Batched `Σ weight·x` for one IQ2_XS row against `m` Q8 activations (`out[r]`). Expands the
/// grid-codebook weight row to signed i8 ONCE (`iq2xs_expand_row`), then reuses it across all `m`
/// token activations — the amortisation the single-token path can't do. Bit-identical accumulation
/// to `vec_dot_iq2xs` (same group order, same f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq2xs_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq2xs_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, dls) = iq2xs_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut block_sum = 0f32;
            for g in 0..32usize {
                let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                let a = &q8b[g * 8..g * 8 + 8];
                let mut iprod = 0i32;
                for k in 0..8usize {
                    iprod += w[k] as i32 * a[k] as i32;
                }
                block_sum += dls[b * 32 + g] * iprod as f32;
            }
            sumf += q8.d[b] * block_sum;
        }
        out[r] = sumf;
    }
}

/// Expand a whole IQ3_S weight row (110 bytes / 256 elems per super-block) to signed `i8` codebook
/// weights ONCE, alongside the per-32-element `db` scales — mirrors `dequant_codebook(DType::Iq3S)`
/// exactly. IQ3_S is a GRID-codebook quant like IQ2_S, but each group of 8 weights is TWO 9-bit
/// indices into the 512-entry `IQ3S_GRID` (each entry packs 4 signed i8): `g1` fills elements 0..3,
/// `g2` fills 4..7, then a per-group sign byte flips them via `KMASK_IQ2XS` (sign_off 0 for g1, 4 for
/// g2 — the two-half split of `apply_signs` in dequant.rs). The 9th index bit comes from `qh`. The
/// grid gather can't be SIMD-vectorized, so the expansion is scalar and the caller runs the int dot
/// against the Q8 activation, amortising this decode across all `m` tokens. `weights[b*256 + g*8 + k]`
/// and `dls[b*32 + g]` are in the SAME element order the dequant writes (`outoff` progression: pair
/// 0..4 → two 32-groups → l 0..4, 8 elems each); `dls[g]` is the group's `db` (same for the 4
/// groups-of-8 inside each 32-group). `in_f` must be a multiple of 256.
fn iq3s_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>) {
    use infr_core::iquant_grids::{IQ3S_GRID, KMASK_IQ2XS};
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut dls = vec![0f32; nb * 32];
    for b in 0..nb {
        let blk = &row[b * 110..b * 110 + 110];
        let d = rdf16(&blk[0..2]);
        let qs = &blk[2..66]; // 64 bytes: grid-idx low bytes
        let qh = &blk[66..74]; // 8 bytes: 9th grid-idx bit per group
        let signs = &blk[74..106]; // 32 bytes: per-group sign masks
        let scales = &blk[106..110]; // 4 bytes: two 4-bit scales per byte
        let mut outoff = 0usize; // element offset within block (0..256)
        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        let mut qh_off = 0usize;
        // 4 pairs → each pair is two 32-element groups (db1 from low nibble, db2 from high).
        for pair in 0..4usize {
            let db1 = d * (1.0 + 2.0 * (scales[pair] & 0xf) as f32);
            let db2 = d * (1.0 + 2.0 * (scales[pair] >> 4) as f32);
            for (qh_byte, db) in [(qh[qh_off], db1), (qh[qh_off + 1], db2)] {
                for l in 0..4usize {
                    let g1_idx = qs[qs_off + 2 * l] as usize
                        | (((qh_byte as u32).wrapping_shl((8 - 2 * l) as u32)) & 256) as usize;
                    let g2_idx = qs[qs_off + 2 * l + 1] as usize
                        | (((qh_byte as u32).wrapping_shl((7 - 2 * l) as u32)) & 256) as usize;
                    let g1 = IQ3S_GRID[g1_idx] as u64;
                    let g2 = IQ3S_GRID[g2_idx] as u64;
                    let sb = signs[signs_off + l];
                    let wo = b * 256 + outoff;
                    // g1 → elems 0..3 (sign_off 0), g2 → elems 4..7 (sign_off 4).
                    for k in 0..4usize {
                        let gv = ((g1 >> (8 * k)) & 0xFF) as i8;
                        weights[wo + k] = if sb & KMASK_IQ2XS[k] != 0 {
                            gv.wrapping_neg()
                        } else {
                            gv
                        };
                    }
                    for k in 0..4usize {
                        let gv = ((g2 >> (8 * k)) & 0xFF) as i8;
                        weights[wo + 4 + k] = if sb & KMASK_IQ2XS[k + 4] != 0 {
                            gv.wrapping_neg()
                        } else {
                            gv
                        };
                    }
                    dls[b * 32 + outoff / 8] = db;
                    outoff += 8;
                }
                qs_off += 8;
                signs_off += 4;
            }
            qh_off += 2;
        }
    }
    (weights, dls)
}

/// `Σ weight·x` for one IQ3_S row against the Q8 activation. Expands the grid-codebook weight row to
/// signed i8 once (`iq3s_expand_row`), then per group of 8 runs an integer dot scaled by the per-32
/// `dl`; the block's `dl·iprod` terms accumulate in an f32 running sum, then ONE multiply by the
/// super-block activation scale `q8.d[b]`. No offset/min/bsum term — grid weights are already signed
/// and zero-centred. Identical accumulation shape to `vec_dot_iq2s`. `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq3s(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq3s: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, dls) = iq3s_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut block_sum = 0f32;
        for g in 0..32usize {
            let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
            let a = &q8b[g * 8..g * 8 + 8];
            let mut iprod = 0i32;
            for k in 0..8usize {
                iprod += w[k] as i32 * a[k] as i32;
            }
            block_sum += dls[b * 32 + g] * iprod as f32;
        }
        sumf += q8.d[b] * block_sum;
    }
    sumf
}

/// Batched `Σ weight·x` for one IQ3_S row against `m` Q8 activations (`out[r]`). Expands the
/// grid-codebook weight row to signed i8 ONCE (`iq3s_expand_row`), then reuses it across all `m`
/// token activations — the amortisation the single-token path can't do. Bit-identical accumulation
/// to `vec_dot_iq3s` (same group order, same f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq3s_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq3s_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, dls) = iq3s_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut block_sum = 0f32;
            for g in 0..32usize {
                let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                let a = &q8b[g * 8..g * 8 + 8];
                let mut iprod = 0i32;
                for k in 0..8usize {
                    iprod += w[k] as i32 * a[k] as i32;
                }
                block_sum += dls[b * 32 + g] * iprod as f32;
            }
            sumf += q8.d[b] * block_sum;
        }
        out[r] = sumf;
    }
}

/// Expand a whole IQ3_XXS weight row (98 bytes / 256 elems per super-block) to signed `i8` codebook
/// weights ONCE, alongside the per-32-element `db` scales — mirrors `dequant_codebook(DType::Iq3Xxs)`
/// exactly. IQ3_XXS is a GRID-codebook quant like IQ3_S but SIMPLER (no `qh` high bits): each group
/// of 8 weights is TWO 8-bit indices into the 256-entry `IQ3XXS_GRID` (each entry packs 4 signed i8):
/// `g1` fills elements 0..3, `g2` fills 4..7. Signs and the per-32 scale ride together in the
/// `scales_and_signs` `aux32` words: the top nibble (`aux32 >> 28`) is the scale (`db = d*(0.5+s)*0.5`)
/// and each of the four 7-bit fields indexes `KSIGNS_IQ2XS` for a group's sign byte (flipping via
/// `KMASK_IQ2XS`, offset 0 for g1's four elems, 4 for g2's — the two-half split of `apply_signs` in
/// dequant.rs). The grid gather can't be SIMD-vectorized, so the expansion is scalar and the caller
/// runs the int dot against the Q8 activation, amortising this decode across all `m` tokens.
/// `weights[b*256 + g*8 + k]` and `dls[b*32 + g]` are in the SAME element order the dequant writes
/// (`outoff` progression: ib32 0..8 → l 0..4, 8 elems each); `dls[g]` is the group's `db` (same for the
/// 4 groups-of-8 inside each 32-sub-block). `in_f` must be a multiple of 256.
fn iq3xxs_expand_row(row: &[u8], in_f: usize) -> (Vec<i8>, Vec<f32>) {
    use infr_core::iquant_grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
    let nb = in_f / 256;
    let mut weights = vec![0i8; nb * 256];
    let mut dls = vec![0f32; nb * 32];
    for b in 0..nb {
        let blk = &row[b * 98..b * 98 + 98];
        let d = rdf16(&blk[0..2]);
        let qs = &blk[2..66]; // 64 bytes: grid indices (two per group of 8)
        let sas = &blk[66..98]; // 32 bytes: scales-and-signs (8 × u32)
        let mut outoff = 0usize; // element offset within block (0..256)
        let mut qs_off = 0usize;
        for ib32 in 0..8usize {
            let aux32 = u32::from_le_bytes(sas[4 * ib32..4 * ib32 + 4].try_into().unwrap());
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4usize {
                let sign_idx = ((aux32 >> (7 * l)) & 127) as usize;
                let sb = KSIGNS_IQ2XS[sign_idx];
                let g1 = IQ3XXS_GRID[qs[qs_off + 2 * l] as usize] as u64;
                let g2 = IQ3XXS_GRID[qs[qs_off + 2 * l + 1] as usize] as u64;
                let wo = b * 256 + outoff;
                // g1 → elems 0..3 (sign_off 0), g2 → elems 4..7 (sign_off 4).
                for k in 0..4usize {
                    let gv = ((g1 >> (8 * k)) & 0xFF) as i8;
                    weights[wo + k] = if sb & KMASK_IQ2XS[k] != 0 {
                        gv.wrapping_neg()
                    } else {
                        gv
                    };
                }
                for k in 0..4usize {
                    let gv = ((g2 >> (8 * k)) & 0xFF) as i8;
                    weights[wo + 4 + k] = if sb & KMASK_IQ2XS[k + 4] != 0 {
                        gv.wrapping_neg()
                    } else {
                        gv
                    };
                }
                dls[b * 32 + outoff / 8] = db;
                outoff += 8;
            }
            qs_off += 8;
        }
    }
    (weights, dls)
}

/// `Σ weight·x` for one IQ3_XXS row against the Q8 activation. Expands the grid-codebook weight row to
/// signed i8 once (`iq3xxs_expand_row`), then per group of 8 runs an integer dot scaled by the per-32
/// `dl`; the block's `dl·iprod` terms accumulate in an f32 running sum, then ONE multiply by the
/// super-block activation scale `q8.d[b]`. No offset/min/bsum term — grid weights are already signed
/// and zero-centred. Identical accumulation shape to `vec_dot_iq3s`. `in_f` must be a multiple of 256.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq3xxs(row: &[u8], q8: &Q8, in_f: usize) -> f32 {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq3xxs: in_f must be a multiple of 256"
    );
    let nb = in_f / 256;
    let (weights, dls) = iq3xxs_expand_row(row, in_f);
    let mut sumf = 0f32;
    for b in 0..nb {
        let q8b = &q8.qs[b * 256..b * 256 + 256];
        let mut block_sum = 0f32;
        for g in 0..32usize {
            let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
            let a = &q8b[g * 8..g * 8 + 8];
            let mut iprod = 0i32;
            for k in 0..8usize {
                iprod += w[k] as i32 * a[k] as i32;
            }
            block_sum += dls[b * 32 + g] * iprod as f32;
        }
        sumf += q8.d[b] * block_sum;
    }
    sumf
}

/// Batched `Σ weight·x` for one IQ3_XXS row against `m` Q8 activations (`out[r]`). Expands the
/// grid-codebook weight row to signed i8 ONCE (`iq3xxs_expand_row`), then reuses it across all `m`
/// token activations — the amortisation the single-token path can't do. Bit-identical accumulation
/// to `vec_dot_iq3xxs` (same group order, same f32 epilogue), so batch and single agree to the bit.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq3xxs_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_iq3xxs_batch: in_f must be a multiple of 256"
    );
    let m = q8s.len();
    let nb = in_f / 256;
    let (weights, dls) = iq3xxs_expand_row(row, in_f);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            let mut block_sum = 0f32;
            for g in 0..32usize {
                let w = &weights[b * 256 + g * 8..b * 256 + g * 8 + 8];
                let a = &q8b[g * 8..g * 8 + 8];
                let mut iprod = 0i32;
                for k in 0..8usize {
                    iprod += w[k] as i32 * a[k] as i32;
                }
                block_sum += dls[b * 32 + g] * iprod as f32;
            }
            sumf += q8.d[b] * block_sum;
        }
        out[r] = sumf;
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q8_0_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb_super = in_f / 256;
    let bpr = 34usize;

    // Pre-read weight scales and i8 values (no nibble extraction needed for Q8_0).
    let mut dw_arr = vec![0f32; nb_super * 8];
    let mut qw_flat = vec![0i8; nb_super * 256]; // raw i8 weight bytes

    for b in 0..nb_super {
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            dw_arr[b * 8 + s] = rdf16(&blk[0..2]);
            let src = &blk[2..34];
            let dst = &mut qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
            for i in 0..32 {
                dst[i] = src[i] as i8;
            }
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb_super {
            let d8 = q8.d[b];
            for s in 0..8usize {
                let qw = &qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32];
                let qx = &q8.qs[b * 256 + s * 32..b * 256 + s * 32 + 32];
                let iprod: i32 = (0..32).map(|i| qw[i] as i32 * qx[i] as i32).sum();
                sumf += d8 * dw_arr[b * 8 + s] * iprod as f32;
            }
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q8_0_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb_super = in_f / 256;
    let bpr = 34usize;
    let ones = _mm256_set1_epi16(1i16);

    // Pre-read weight scales + abs(qw) (u8) + sign bytes (for _mm256_sign_epi8 per token).
    // Storing qw as raw i8 and applying abs+sign per token avoids two extra arrays.
    let mut dw_arr = vec![0f32; nb_super * 8];
    let mut qw_flat = vec![0u8; nb_super * 256]; // i8 as u8 bytes, to be cast per-kernel call

    for b in 0..nb_super {
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            dw_arr[b * 8 + s] = rdf16(&blk[0..2]);
            let src = &blk[2..34];
            qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32].copy_from_slice(src);
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb_super {
            let d8 = q8.d[b];
            for s in 0..8usize {
                let qw = load256(&qw_flat, b * 256 + s * 32);
                let qx = load256(&q8.qs, b * 256 + s * 32);
                let qw_abs = _mm256_abs_epi8(qw);
                let qx_signed = _mm256_sign_epi8(qx, qw);
                let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod = hadd_i32_ymm(sum32);
                sumf += d8 * dw_arr[b * 8 + s] * iprod as f32;
            }
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q8_0_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb_super = in_f / 256;
    let bpr = 34usize;
    let ones512 = _mm512_set1_epi16(1i16);

    let mut dw_arr = vec![0f32; nb_super * 8];
    let mut qw_flat = vec![0u8; nb_super * 256];

    for b in 0..nb_super {
        for s in 0..8usize {
            let wb = b * 8 + s;
            let blk = &row[wb * bpr..wb * bpr + bpr];
            dw_arr[b * 8 + s] = rdf16(&blk[0..2]);
            qw_flat[b * 256 + s * 32..b * 256 + s * 32 + 32].copy_from_slice(&blk[2..34]);
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb_super {
            let d8 = q8.d[b];
            // Process pairs of sub-blocks with zmm (identical to single-token avx512bw).
            for k in 0..4usize {
                let s0 = 2 * k;
                let s1 = 2 * k + 1;
                let qw0 = load256(&qw_flat, b * 256 + s0 * 32);
                let qw1 = load256(&qw_flat, b * 256 + s1 * 32);
                // Load 64 contiguous activation bytes
                let qx_z = load512(&q8.qs, b * 256 + s0 * 32);
                let qx0 = _mm512_castsi512_si256(qx_z);
                let qx1 = _mm512_extracti64x4_epi64::<1>(qx_z);
                let qw_abs0 = _mm256_abs_epi8(qw0);
                let qw_abs1 = _mm256_abs_epi8(qw1);
                let qx_s0 = _mm256_sign_epi8(qx0, qw0);
                let qx_s1 = _mm256_sign_epi8(qx1, qw1);
                let qw_a_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qw_abs0), qw_abs1);
                let qx_s_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qx_s0), qx_s1);
                let prod = _mm512_maddubs_epi16(qw_a_z, qx_s_z);
                let sum32_z = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32_z);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
                let iprod0 = hadd_i32_ymm(lo_ymm);
                let iprod1 = hadd_i32_ymm(hi_ymm);
                sumf +=
                    d8 * (dw_arr[b * 8 + s0] * iprod0 as f32 + dw_arr[b * 8 + s1] * iprod1 as f32);
            }
        }
        out[r] = sumf;
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q8_0_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q8_0_batch: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q8_0_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q8_0_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q8_0_batch_scalar(row, q8s, in_f, out);
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q5k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;

    // Pre-expand Q5 values (0–31) into flat[b*256..b*256+256].
    // Layout: flat[b*256 + s*32 .. +32] = q5 for sub-block s (same as Q4K flat layout).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![0u32; nb * 8];
    let mut m_arr = vec![0u32; nb * 8];
    let mut q5_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4usize {
            let (sc_e, m_e) = k4(2 * j, scales);
            let (sc_o, m_o) = k4(2 * j + 1, scales);
            sc_arr[b * 8 + 2 * j] = sc_e;
            m_arr[b * 8 + 2 * j] = m_e;
            sc_arr[b * 8 + 2 * j + 1] = sc_o;
            m_arr[b * 8 + 2 * j + 1] = m_o;
            let base_e = b * 256 + (2 * j) * 32;
            let base_o = b * 256 + (2 * j + 1) * 32;
            for l in 0..32 {
                let v = ql[j * 32 + l];
                q5_flat[base_e + l] = (v & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
                q5_flat[base_o + l] = (v >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            for j in 0..4usize {
                let mut iprod_e = 0i32;
                let mut iprod_o = 0i32;
                let fe = &q5_flat[b * 256 + (2 * j) * 32..b * 256 + (2 * j) * 32 + 32];
                let fo = &q5_flat[b * 256 + (2 * j + 1) * 32..b * 256 + (2 * j + 1) * 32 + 32];
                let q8e = &q8.qs[b * 256 + (2 * j) * 32..b * 256 + (2 * j) * 32 + 32];
                let q8o = &q8.qs[b * 256 + (2 * j + 1) * 32..b * 256 + (2 * j + 1) * 32 + 32];
                for l in 0..32 {
                    iprod_e += fe[l] as i32 * q8e[l] as i32;
                    iprod_o += fo[l] as i32 * q8o[l] as i32;
                }
                sd += sc_arr[b * 8 + 2 * j] as i32 * iprod_e
                    + sc_arr[b * 8 + 2 * j + 1] as i32 * iprod_o;
                sm += m_arr[b * 8 + 2 * j] as i32 * q8.bsums[b * 8 + 2 * j]
                    + m_arr[b * 8 + 2 * j + 1] as i32 * q8.bsums[b * 8 + 2 * j + 1];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones = _mm256_set1_epi16(1i16);
    let zero = _mm256_setzero_si256();

    // Pre-expand q5 values (0–31) using the same AVX2 logic as the single-token kernel.
    // flat[b*256 + k*64..+32] = even sub-block 2k (lo nibble + high bit),
    // flat[b*256 + k*64+32..+32] = odd sub-block 2k+1 (hi nibble + high bit).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q5_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b][s] = sc;
            m_arr[b][s] = mv;
        }
        let qh_ymm = load256(qh, 0);
        let flat = &mut q5_flat[b * 256..b * 256 + 256];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for k in 0..4usize {
            let nibbles = load256(ql, k * 32);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero), sixteen);
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            _mm256_storeu_si256(flat[k * 64..].as_mut_ptr() as *mut __m256i, q5_e);
            _mm256_storeu_si256(flat[k * 64 + 32..].as_mut_ptr() as *mut __m256i, q5_o);
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q5_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for j in 0..4usize {
                let (sc_e, m_e) = (sc_arr[b][2 * j], m_arr[b][2 * j]);
                let (sc_o, m_o) = (sc_arr[b][2 * j + 1], m_arr[b][2 * j + 1]);
                let q5_e = load256(flat, j * 64);
                let q5_o = load256(flat, j * 64 + 32);
                let q8_e = load256(q8b, 2 * j * 32);
                let q8_o = load256(q8b, (2 * j + 1) * 32);
                let prod_e = _mm256_maddubs_epi16(q5_e, q8_e);
                let sum32_e = _mm256_madd_epi16(prod_e, ones);
                let iprod_e = hadd_i32_ymm(sum32_e);
                let prod_o = _mm256_maddubs_epi16(q5_o, q8_o);
                let sum32_o = _mm256_madd_epi16(prod_o, ones);
                let iprod_o = hadd_i32_ymm(sum32_o);
                let isum_e = q8.bsums[b * 8 + 2 * j];
                let isum_o = q8.bsums[b * 8 + 2 * j + 1];
                sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
                sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let sixteen = _mm256_set1_epi8(0x10_u8 as i8);
    let ones512 = _mm512_set1_epi16(1i16);
    let zero256 = _mm256_setzero_si256();

    // Pre-expand q5: same flat layout as AVX2 batch (k*64 = [even 32B, odd 32B]).
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![[0u32; 8]; nb];
    let mut m_arr = vec![[0u32; 8]; nb];
    let mut q5_flat = vec![0u8; nb * 256];

    for b in 0..nb {
        let blk = &row[b * 176..b * 176 + 176];
        d_arr[b] = rdf16(&blk[0..2]);
        dmin_arr[b] = rdf16(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        for s in 0..8usize {
            let (sc, mv) = k4(s, scales);
            sc_arr[b][s] = sc;
            m_arr[b][s] = mv;
        }
        let qh_ymm = load256(qh, 0);
        let flat = &mut q5_flat[b * 256..b * 256 + 256];
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for k in 0..4usize {
            let nibbles = load256(ql, k * 32);
            let lo_nib = _mm256_and_si256(nibbles, mask_lo);
            let hi_nib = _mm256_and_si256(_mm256_srli_epi16(nibbles, 4), mask_lo);
            let u1v = _mm256_set1_epi8(u1 as i8);
            let u2v = _mm256_set1_epi8(u2 as i8);
            let has_e = _mm256_and_si256(qh_ymm, u1v);
            let has_o = _mm256_and_si256(qh_ymm, u2v);
            let high_e = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_e, zero256), sixteen);
            let high_o = _mm256_andnot_si256(_mm256_cmpeq_epi8(has_o, zero256), sixteen);
            let q5_e = _mm256_or_si256(lo_nib, high_e);
            let q5_o = _mm256_or_si256(hi_nib, high_o);
            _mm256_storeu_si256(flat[k * 64..].as_mut_ptr() as *mut __m256i, q5_e);
            _mm256_storeu_si256(flat[k * 64 + 32..].as_mut_ptr() as *mut __m256i, q5_o);
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q5_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for k in 0..4usize {
                let (sc_e, m_e) = (sc_arr[b][2 * k], m_arr[b][2 * k]);
                let (sc_o, m_o) = (sc_arr[b][2 * k + 1], m_arr[b][2 * k + 1]);
                // flat[k*64..+64]: lower 32B = even sub-block, upper 32B = odd
                let q5_zmm = load512(flat, k * 64);
                let q8_zmm = load512(q8b, k * 64);
                let prod = _mm512_maddubs_epi16(q5_zmm, q8_zmm);
                let sum32_z = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32_z);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
                let iprod_e = hadd_i32_ymm(lo_ymm);
                let iprod_o = hadd_i32_ymm(hi_ymm);
                let isum_e = q8.bsums[b * 8 + 2 * k];
                let isum_o = q8.bsums[b * 8 + 2 * k + 1];
                sd += sc_e as i32 * iprod_e + sc_o as i32 * iprod_o;
                sm += m_e as i32 * isum_e + m_o as i32 * isum_o;
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q5k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q5k_batch: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q5k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q5k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q5k_batch_scalar(row, q8s, in_f, out);
}

/// Scalar Q2_K batch: decode the weight row's 2-bit codes into natural element order ONCE (into
/// `q2_flat[b*256 + is*16 + l]`, matching `q8b` layout) plus per-16 sub-block scale/min arrays,
/// then run the integer dot against every one of the `m` Q8 activations. Bit-identical to
/// `vec_dot_q2k_scalar` (same `is=0..16` accumulation order, same `sm` via `bsums16`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q2k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![0i32; nb * 16];
    let mut min_arr = vec![0i32; nb * 16];
    let mut q2_flat = vec![0u8; nb * 256];
    for b in 0..nb {
        let blk = &row[b * 84..b * 84 + 84];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        d_arr[b] = rdf16(&blk[80..82]);
        dmin_arr[b] = rdf16(&blk[82..84]);
        let mut is = 0usize;
        let mut qoff = 0usize;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                for half in 0..2 {
                    let sc = scales[is];
                    sc_arr[b * 16 + is] = (sc & 0xF) as i32;
                    min_arr[b * 16 + is] = (sc >> 4) as i32;
                    let qbyte = &qs[qoff + half * 16..qoff + half * 16 + 16];
                    let dst = &mut q2_flat[b * 256 + is * 16..b * 256 + is * 16 + 16];
                    for l in 0..16 {
                        dst[l] = (qbyte[l] >> shift) & 3;
                    }
                    is += 1;
                }
                shift += 2;
            }
            qoff += 32;
        }
    }
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let (mut sd, mut sm) = (0i32, 0i32);
            let flat = &q2_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for is in 0..16usize {
                let f = &flat[is * 16..is * 16 + 16];
                let a = &q8b[is * 16..is * 16 + 16];
                let mut iprod = 0i32;
                for l in 0..16 {
                    iprod += f[l] as i32 * a[l] as i32;
                }
                sd += sc_arr[b * 16 + is] * iprod;
                sm += min_arr[b * 16 + is] * q8.bsums16[b * 16 + is];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// AVX2 Q2_K batch: same decode-once flat buffer as the scalar path; per token, each 32-byte flat
/// chunk `c` holds two 16-elem sub-blocks (`2c` lo, `2c+1` hi) aligned with `q8b[c*32..]`. One
/// `maddubs → madd` → split ymm gives both sub-block dots. Integer dot is order-free and the
/// epilogue matches scalar → bit-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q2k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let ones = _mm256_set1_epi16(1i16);
    let (d_arr, dmin_arr, sc_arr, min_arr, q2_flat) = q2k_decode_row(row, nb);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut sd = 0i32;
            let flat = &q2_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for c in 0..8usize {
                let f = load256(flat, c * 32);
                let a = load256(q8b, c * 32);
                let prod = _mm256_maddubs_epi16(f, a);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod_lo = hadd_i32_xmm(_mm256_castsi256_si128(sum32));
                let iprod_hi = hadd_i32_xmm(_mm256_extracti128_si256::<1>(sum32));
                sd += sc_arr[b * 16 + 2 * c] * iprod_lo + sc_arr[b * 16 + 2 * c + 1] * iprod_hi;
            }
            let mut sm = 0i32;
            for is in 0..16usize {
                sm += min_arr[b * 16 + is] * q8.bsums16[b * 16 + is];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// AVX-512BW Q2_K batch: process sub-block quads via zmm (64 contiguous flat + q8 bytes = pair of
/// 32-byte chunks `2cp`,`2cp+1` → sub-blocks `4cp..4cp+4`). `maddubs512 → madd512`, split to four
/// 16-elem dots. Bit-identical to the scalar path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q2k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let ones512 = _mm512_set1_epi16(1i16);
    let (d_arr, dmin_arr, sc_arr, min_arr, q2_flat) = q2k_decode_row(row, nb);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut sd = 0i32;
            let flat = &q2_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for cp in 0..4usize {
                let f = load512(flat, cp * 64);
                let a = load512(q8b, cp * 64);
                let prod = _mm512_maddubs_epi16(f, a);
                let sum32 = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                let ip0 = hadd_i32_xmm(_mm256_castsi256_si128(lo_ymm));
                let ip1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(lo_ymm));
                let ip2 = hadd_i32_xmm(_mm256_castsi256_si128(hi_ymm));
                let ip3 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(hi_ymm));
                let base = b * 16 + 4 * cp;
                sd += sc_arr[base] * ip0
                    + sc_arr[base + 1] * ip1
                    + sc_arr[base + 2] * ip2
                    + sc_arr[base + 3] * ip3;
            }
            let mut sm = 0i32;
            for is in 0..16usize {
                sm += min_arr[b * 16 + is] * q8.bsums16[b * 16 + is];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI variant of [`vec_dot_q2k_batch_avx512bw`]: `_mm512_dpbusd_epi32` fuses the
/// maddubs+madd pair into ONE u8×s8→i32 dot-accumulate. Bit-identical — the codes (0–3) × |q8| ≤127
/// never saturated maddubs' i16 either, so the i32 lanes hold the same per-4-byte-group sums, and
/// the hadd/scale order is unchanged.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q2k_batch_vnni(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let (d_arr, dmin_arr, sc_arr, min_arr, q2_flat) = q2k_decode_row(row, nb);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut sd = 0i32;
            let flat = &q2_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for cp in 0..4usize {
                let f = load512(flat, cp * 64);
                let a = load512(q8b, cp * 64);
                let sum32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), f, a);
                let lo_ymm = _mm512_castsi512_si256(sum32);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                let ip0 = hadd_i32_xmm(_mm256_castsi256_si128(lo_ymm));
                let ip1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(lo_ymm));
                let ip2 = hadd_i32_xmm(_mm256_castsi256_si128(hi_ymm));
                let ip3 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(hi_ymm));
                let base = b * 16 + 4 * cp;
                sd += sc_arr[base] * ip0
                    + sc_arr[base + 1] * ip1
                    + sc_arr[base + 2] * ip2
                    + sc_arr[base + 3] * ip3;
            }
            let mut sm = 0i32;
            for is in 0..16usize {
                sm += min_arr[b * 16 + is] * q8.bsums16[b * 16 + is];
            }
            sumf += q8.d[b] * (d_arr[b] * sd as f32 - dmin_arr[b] * sm as f32);
        }
        out[r] = sumf;
    }
}

/// Decode a Q2_K weight row ONCE for the batch kernels: 2-bit codes into natural element order
/// (`q2_flat[b*256 + is*16 + l]`, mirroring `q8b`), plus per-16 sub-block `scale`/`min` and the
/// `d`/`dmin` super-block scales. Traversal order matches `dequant_block(DType::Q2K)` exactly.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::type_complexity)]
fn q2k_decode_row(row: &[u8], nb: usize) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>, Vec<u8>) {
    let mut d_arr = vec![0f32; nb];
    let mut dmin_arr = vec![0f32; nb];
    let mut sc_arr = vec![0i32; nb * 16];
    let mut min_arr = vec![0i32; nb * 16];
    let mut q2_flat = vec![0u8; nb * 256];
    for b in 0..nb {
        let blk = &row[b * 84..b * 84 + 84];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        d_arr[b] = rdf16(&blk[80..82]);
        dmin_arr[b] = rdf16(&blk[82..84]);
        let mut is = 0usize;
        let mut qoff = 0usize;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                for half in 0..2 {
                    let sc = scales[is];
                    sc_arr[b * 16 + is] = (sc & 0xF) as i32;
                    min_arr[b * 16 + is] = (sc >> 4) as i32;
                    let qbyte = &qs[qoff + half * 16..qoff + half * 16 + 16];
                    let dst = &mut q2_flat[b * 256 + is * 16..b * 256 + is * 16 + 16];
                    for l in 0..16 {
                        dst[l] = (qbyte[l] >> shift) & 3;
                    }
                    is += 1;
                }
                shift += 2;
            }
            qoff += 32;
        }
    }
    (d_arr, dmin_arr, sc_arr, min_arr, q2_flat)
}

/// Batch Q2_K dot: `out[r] = vec_dot_q2k(row, &q8s[r], in_f)` for all r, bit-identical to the
/// single-token kernel. The weight row is decoded ONCE; per-token work is the integer dot only.
/// Dispatches VNNI → avx512bw → avx2 → scalar.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q2k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q2k_batch: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q2k_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q2k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q2k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q2k_batch_scalar(row, q8s, in_f, out);
}

/// Decode a Q3_K weight row ONCE for the batch kernels: 3-bit codes (`low2` from `qs` + `high` bit
/// from `hmask`) into natural element order (`q3_flat[b*256 + is*16 + l]`, mirroring `q8b`), plus
/// the per-16 signed `(sc6−32)` scale and the super-block `d`. Traversal (and the `m` bit-plane
/// advance — once per `_j`, 8 planes total) matches `dequant_block(DType::Q3K)` exactly.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::type_complexity)]
fn q3k_decode_row(row: &[u8], nb: usize) -> (Vec<f32>, Vec<i32>, Vec<u8>) {
    let mut d_arr = vec![0f32; nb];
    let mut sc_arr = vec![0i32; nb * 16];
    let mut q3_flat = vec![0u8; nb * 256];
    for b in 0..nb {
        let blk = &row[b * 110..b * 110 + 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let sc = q3k_scales(&blk[96..108]);
        d_arr[b] = rdf16(&blk[108..110]);
        for is in 0..16 {
            sc_arr[b * 16 + is] = sc[is] as i32;
        }
        let mut is = 0usize;
        let mut qoff = 0usize;
        let mut m = 1u8;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                for half in 0..2 {
                    let qbyte = &qs[qoff + half * 16..qoff + half * 16 + 16];
                    let hbyte = &hmask[half * 16..half * 16 + 16];
                    let dst = &mut q3_flat[b * 256 + is * 16..b * 256 + is * 16 + 16];
                    for l in 0..16 {
                        let low2 = (qbyte[l] >> shift) & 3;
                        let high = if hbyte[l] & m != 0 { 4u8 } else { 0 };
                        dst[l] = low2 | high;
                    }
                    is += 1;
                }
                shift += 2;
                m <<= 1;
            }
            qoff += 32;
        }
    }
    (d_arr, sc_arr, q3_flat)
}

/// Scalar Q3_K batch: decode the weight row's 3-bit codes into natural element order ONCE (into
/// `q3_flat[b*256 + is*16 + l]`, matching `q8b`) plus the per-16 signed scale array, then run the
/// integer dot against every one of the `m` Q8 activations. Bit-identical to `vec_dot_q3k_scalar`
/// (same `is=0..16` accumulation order, same `−4·bsum` correction via `q8.bsums16`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q3k_batch_scalar(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    let m = q8s.len();
    let nb = in_f / 256;
    let mut d_arr = vec![0f32; nb];
    let mut sc_arr = vec![0i32; nb * 16];
    let mut q3_flat = vec![0u8; nb * 256];
    for b in 0..nb {
        let blk = &row[b * 110..b * 110 + 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let sc = q3k_scales(&blk[96..108]);
        d_arr[b] = rdf16(&blk[108..110]);
        for is in 0..16 {
            sc_arr[b * 16 + is] = sc[is] as i32;
        }
        let mut is = 0usize;
        let mut qoff = 0usize;
        let mut mp = 1u8;
        for _n in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                for half in 0..2 {
                    let qbyte = &qs[qoff + half * 16..qoff + half * 16 + 16];
                    let hbyte = &hmask[half * 16..half * 16 + 16];
                    let dst = &mut q3_flat[b * 256 + is * 16..b * 256 + is * 16 + 16];
                    for l in 0..16 {
                        let low2 = (qbyte[l] >> shift) & 3;
                        let high = if hbyte[l] & mp != 0 { 4u8 } else { 0 };
                        dst[l] = low2 | high;
                    }
                    is += 1;
                }
                shift += 2;
                mp <<= 1;
            }
            qoff += 32;
        }
    }
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut isum = 0i32;
            let flat = &q3_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for is in 0..16usize {
                let f = &flat[is * 16..is * 16 + 16];
                let a = &q8b[is * 16..is * 16 + 16];
                let mut iprod = 0i32;
                for l in 0..16 {
                    iprod += f[l] as i32 * a[l] as i32;
                }
                isum += sc_arr[b * 16 + is] * (iprod - 4 * q8.bsums16[b * 16 + is]);
            }
            sumf += d_arr[b] * q8.d[b] * isum as f32;
        }
        out[r] = sumf;
    }
}

/// AVX2 Q3_K batch: same decode-once flat buffer as the scalar path; per token, each 32-byte flat
/// chunk `c` holds two 16-elem sub-blocks (`2c` lo, `2c+1` hi) aligned with `q8b[c*32..]`. One
/// `maddubs → madd` → split ymm gives both sub-block dots. Integer dot is order-free and the
/// signed-scale × `(iprod − 4·bsum16)` epilogue matches scalar → bit-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q3k_batch_avx2(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let ones = _mm256_set1_epi16(1i16);
    let (d_arr, sc_arr, q3_flat) = q3k_decode_row(row, nb);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut isum = 0i32;
            let flat = &q3_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for c in 0..8usize {
                let f = load256(flat, c * 32);
                let a = load256(q8b, c * 32);
                let prod = _mm256_maddubs_epi16(f, a);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod_lo = hadd_i32_xmm(_mm256_castsi256_si128(sum32));
                let iprod_hi = hadd_i32_xmm(_mm256_extracti128_si256::<1>(sum32));
                let is0 = 2 * c;
                isum += sc_arr[b * 16 + is0] * (iprod_lo - 4 * q8.bsums16[b * 16 + is0]);
                isum += sc_arr[b * 16 + is0 + 1] * (iprod_hi - 4 * q8.bsums16[b * 16 + is0 + 1]);
            }
            sumf += d_arr[b] * q8.d[b] * isum as f32;
        }
        out[r] = sumf;
    }
}

/// AVX-512BW Q3_K batch: process sub-block quads via zmm (64 contiguous flat + q8 bytes = pair of
/// 32-byte chunks `2cp`,`2cp+1` → sub-blocks `4cp..4cp+4`). `maddubs512 → madd512`, split to four
/// 16-elem dots. Bit-identical to the scalar path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q3k_batch_avx512bw(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let ones512 = _mm512_set1_epi16(1i16);
    let (d_arr, sc_arr, q3_flat) = q3k_decode_row(row, nb);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut isum = 0i32;
            let flat = &q3_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for cp in 0..4usize {
                let f = load512(flat, cp * 64);
                let a = load512(q8b, cp * 64);
                let prod = _mm512_maddubs_epi16(f, a);
                let sum32 = _mm512_madd_epi16(prod, ones512);
                let lo_ymm = _mm512_castsi512_si256(sum32);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                let ip0 = hadd_i32_xmm(_mm256_castsi256_si128(lo_ymm));
                let ip1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(lo_ymm));
                let ip2 = hadd_i32_xmm(_mm256_castsi256_si128(hi_ymm));
                let ip3 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(hi_ymm));
                let base = b * 16 + 4 * cp;
                isum += sc_arr[base] * (ip0 - 4 * q8.bsums16[base]);
                isum += sc_arr[base + 1] * (ip1 - 4 * q8.bsums16[base + 1]);
                isum += sc_arr[base + 2] * (ip2 - 4 * q8.bsums16[base + 2]);
                isum += sc_arr[base + 3] * (ip3 - 4 * q8.bsums16[base + 3]);
            }
            sumf += d_arr[b] * q8.d[b] * isum as f32;
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI variant of [`vec_dot_q3k_batch_avx512bw`]: `_mm512_dpbusd_epi32` fuses the
/// maddubs+madd pair into ONE u8×s8→i32 dot-accumulate. Bit-identical — codes (0–7) × |q8| ≤127
/// never saturate the i16 pairs either, so the i32 lanes hold the same per-4-byte-group sums.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q3k_batch_vnni(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    let nb = in_f / 256;
    let (d_arr, sc_arr, q3_flat) = q3k_decode_row(row, nb);
    for r in 0..m {
        let q8 = &q8s[r];
        let mut sumf = 0f32;
        for b in 0..nb {
            let mut isum = 0i32;
            let flat = &q3_flat[b * 256..b * 256 + 256];
            let q8b = &q8.qs[b * 256..b * 256 + 256];
            for cp in 0..4usize {
                let f = load512(flat, cp * 64);
                let a = load512(q8b, cp * 64);
                let sum32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), f, a);
                let lo_ymm = _mm512_castsi512_si256(sum32);
                let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32);
                let ip0 = hadd_i32_xmm(_mm256_castsi256_si128(lo_ymm));
                let ip1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(lo_ymm));
                let ip2 = hadd_i32_xmm(_mm256_castsi256_si128(hi_ymm));
                let ip3 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(hi_ymm));
                let base = b * 16 + 4 * cp;
                isum += sc_arr[base] * (ip0 - 4 * q8.bsums16[base]);
                isum += sc_arr[base + 1] * (ip1 - 4 * q8.bsums16[base + 1]);
                isum += sc_arr[base + 2] * (ip2 - 4 * q8.bsums16[base + 2]);
                isum += sc_arr[base + 3] * (ip3 - 4 * q8.bsums16[base + 3]);
            }
            sumf += d_arr[b] * q8.d[b] * isum as f32;
        }
        out[r] = sumf;
    }
}

/// Batch Q3_K dot: `out[r] = vec_dot_q3k(row, &q8s[r], in_f)` for all r, bit-identical to the
/// single-token kernel. The weight row is decoded ONCE; per-token work is the integer dot only.
/// Dispatches VNNI → avx512bw → avx2 → scalar.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q3k_batch(row: &[u8], q8s: &[Q8], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 256,
        0,
        "vec_dot_q3k_batch: in_f must be a multiple of 256"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q3k_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { vec_dot_q3k_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q3k_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q3k_batch_scalar(row, q8s, in_f, out);
}

/// Batched Q8_0 dot at native 32-block granularity: `y = d_w·qw` exactly (no min term — the i8 sign
/// already encodes it, see `dequant_block`'s Q8_0 case), so `Σy·x = d_w·Σ(qw·x) ≈ d_w·d8·Σ(qw·q8)`.
/// The weight row's per-block `d_w` is read once, then dotted against every one of the `count` token
/// activations — same amortization as the other `_batch` kernels, just without an f32 intermediate.
/// Dispatches to the best SIMD path available at runtime (avx512bw → avx2 → scalar) — this is
/// DiffusionGemma's `down` projection kernel (`n_ff_exp=704` isn't a multiple of 256, so it can't use
/// the K-quant/Q8_0-256 batch path above), previously the single largest scalar hot loop in the MoE
/// arm (measured via `prof.prof_ops`'s per-stage MoeFfn breakdown).
/// AVX512-VNNI variant of [`vec_dot_q8_0_32_batch_avx512bw`]: dpbusd fuses the maddubs+madd pair
/// (sign trick unchanged: `dpbusd(|qw|, sign(q8,qw))`; |qw| ≤128 × |q8| ≤127 never saturated the
/// i16 chain either, so this is bit-identical — same per-4-byte i32 group sums, same hadd and the
/// same two SEPARATE f32 adds per pair as the scalar oracle).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q8_0_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 34usize;
    let pairs = nb / 2;
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let qw0 = load256(row, b0 * bpr + 2);
            let qw1 = load256(row, b1 * bpr + 2);
            let qx_z = load512(&q8.qs, b0 * 32);
            let qx0 = _mm512_castsi512_si256(qx_z);
            let qx1 = _mm512_extracti64x4_epi64::<1>(qx_z);
            let qw_abs0 = _mm256_abs_epi8(qw0);
            let qw_abs1 = _mm256_abs_epi8(qw1);
            let qx_s0 = _mm256_sign_epi8(qx0, qw0);
            let qx_s1 = _mm256_sign_epi8(qx1, qw1);
            let qw_a_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qw_abs0), qw_abs1);
            let qx_s_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qx_s0), qx_s1);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qw_a_z, qx_s_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d_arr[b0] * q8.d[b0] * iprod0 as f32;
            sumf += d_arr[b1] * q8.d[b1] * iprod1 as f32;
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let qw = load256(row, b * bpr + 2);
            let q8v = load256(&q8.qs, b * 32);
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(q8v, qw);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qw_abs, qx_signed);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q8_0_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    // Native 32-block weight: `nb = in_f / 32` covers the whole row only when 32-aligned. This
    // guard is the 32-block sibling of the `in_f % 256` asserts on the K-quant kernels — it turns
    // a mis-route (e.g. a non-32-multiple `in_f`, or a wrongly-dispatched dtype) into a loud debug
    // panic instead of a silent truncated dot (the Q8_0-on-256-block bug class).
    debug_assert_eq!(
        in_f % 32,
        0,
        "vec_dot_q8_0_32_batch: in_f must be a multiple of 32"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q8_0_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: avx512bw detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q8_0_32_batch_avx512bw(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q8_0_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q8_0_32_batch_scalar(row, q8s, in_f, out);
}

/// Scalar fallback for `vec_dot_q8_0_32_batch`; also used on non-x86 targets, and the exactness
/// oracle the SIMD kernels below are tested against.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q8_0_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 34usize; // f16 d (2B) + 32 × i8 qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let qw = &row[b * bpr + 2..b * bpr + bpr];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for i in 0..32 {
                iprod += qw[i] as i8 as i32 * q8b[i] as i32;
            }
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// AVX2 kernel for `vec_dot_q8_0_32_batch`: one native 32-element Q8_0 block per iteration. Same
/// sign trick as `vec_dot_q8_0_avx2` (`maddubs(abs(qw), sign(qw)·qx)`), just scaled per-block instead
/// of per-256-superblock (each native block carries its own `d_w`, and the activation's own `d[b]`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q8_0_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 34usize;
    let ones = _mm256_set1_epi16(1i16);
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let qw = load256(row, b * bpr + 2);
            let q8v = load256(&q8.qs, b * 32);
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(q8v, qw);
            let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// AVX-512BW kernel for `vec_dot_q8_0_32_batch`: TWO native 32-element blocks per iteration (64
/// elems / zmm), mirroring `vec_dot_q8_0_avx512bw`'s pairing trick — sign trick applied at ymm level,
/// then packed into a zmm for one `maddubs512 → madd512` pass. `nb` is even for every shape this
/// crate has seen (e.g. DiffusionGemma's 22), but an odd tail block is handled scalarly for safety.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q8_0_32_batch_avx512bw(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 34usize;
    let ones512 = _mm512_set1_epi16(1i16);
    let ones256 = _mm256_set1_epi16(1i16);
    let pairs = nb / 2;
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let qw0 = load256(row, b0 * bpr + 2);
            let qw1 = load256(row, b1 * bpr + 2);
            // b0/b1 activation bytes are contiguous (`Q8x32::qs` is laid out `[b*32..b*32+32]`).
            let qx_z = load512(&q8.qs, b0 * 32);
            let qx0 = _mm512_castsi512_si256(qx_z);
            let qx1 = _mm512_extracti64x4_epi64::<1>(qx_z);
            let qw_abs0 = _mm256_abs_epi8(qw0);
            let qw_abs1 = _mm256_abs_epi8(qw1);
            let qx_s0 = _mm256_sign_epi8(qx0, qw0);
            let qx_s1 = _mm256_sign_epi8(qx1, qw1);
            let qw_a_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qw_abs0), qw_abs1);
            let qx_s_z = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(qx_s0), qx_s1);
            let prod = _mm512_maddubs_epi16(qw_a_z, qx_s_z);
            let sum32_z = _mm512_madd_epi16(prod, ones512);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            // Two SEPARATE f32 adds (not `sumf += a + b`) — matches the scalar oracle's per-block
            // sequential accumulation order exactly (f32 addition isn't associative).
            sumf += d_arr[b0] * q8.d[b0] * iprod0 as f32;
            sumf += d_arr[b1] * q8.d[b1] * iprod1 as f32;
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let qw = load256(row, b * bpr + 2);
            let q8v = load256(&q8.qs, b * 32);
            let qw_abs = _mm256_abs_epi8(qw);
            let qx_signed = _mm256_sign_epi8(q8v, qw);
            let prod = _mm256_maddubs_epi16(qw_abs, qx_signed);
            let sum32 = _mm256_madd_epi16(prod, ones256);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Batched Q5_0 dot at native 32-block granularity: `y = d_w·(code−16)`, `code ∈ 0..31` (4 nibble
/// bits from `qs` + 1 high bit from `qh`, per `dequant_block`'s Q5_0 case) — so
/// `Σy·x = d_w·(Σcode·x − 16·Σx) ≈ d_w·d8·(Σcode·q8 − 16·bsum)`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q5_0_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 32,
        0,
        "vec_dot_q5_0_32_batch: in_f must be a multiple of 32"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q5_0_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q5_0_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q5_0_32_batch_scalar(row, q8s, in_f, out);
}

/// Interleaved-x8 tile over native 32-element blocks (AVX512-VNNI) for the MoE `down`
/// projections — the [`vec_dot_q4k_batch8_ilv_vnni`] structure at Q8_0/Q5_0's own block size.
/// Weights become UNSIGNED for dpbusd: Q5_0's 5-bit codes already are (0..31; the −16 rides the
/// `−16·bsum` term); Q8_0's signed bytes are biased `+128` at repack and the exact integer
/// correction `−128·bsum[b]` is applied per block — both integer-exact, so results stay
/// bit-identical to the scalar oracles (same per-block f32 expression and order).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q32_batch8_ilv_vnni(
    rows: [&[u8]; 8],
    q8s: &[Q8x32],
    in_f: usize,
    outs: [&mut [f32]; 8],
    q5: bool, // false = Q8_0 (bias +128), true = Q5_0 (codes, −16·bsum)
) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bias = _mm_set1_epi8(-128i8); // +128 == xor 0x80 on u8 lanes

    // ── repack: per row, unsigned block bytes; then interleave 8-byte groups across rows ──
    // ilv[b*256 + g*64 + i*8 .. +8] = row i, block b, byte group g.
    let mut d_w: [Vec<f32>; 8] = std::array::from_fn(|_| vec![0f32; nb]);
    // PERM-ordered per-block f32 weight-scale vector (see the q4k ilv kernel's d_vec).
    let mut d_vec = vec![_mm256_setzero_ps(); nb];
    let mut ilv = vec![0u8; nb * 256];
    {
        let bpr = if q5 { 22usize } else { 34usize };
        let mut tmp = [[0u8; 32]; 8];
        for b in 0..nb {
            for i in 0..8 {
                let blk = &rows[i][b * bpr..b * bpr + bpr];
                d_w[i][b] = rdf16(&blk[0..2]);
                if q5 {
                    let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                    let qs = &blk[6..22];
                    for j in 0..16 {
                        let xh0 = ((qh >> j) << 4) & 0x10;
                        let xh1 = (qh >> (j + 12)) & 0x10;
                        tmp[i][j] = (qs[j] as u32 & 0x0F | xh0) as u8;
                        tmp[i][j + 16] = (qs[j] as u32 >> 4 | xh1) as u8;
                    }
                } else {
                    // Q8_0: bias the signed bytes to unsigned (two 16-byte xors).
                    let q = &blk[2..34];
                    let v0 = load128(q, 0);
                    let v1 = load128(q, 16);
                    _mm_storeu_si128(tmp[i].as_mut_ptr() as *mut __m128i, _mm_xor_si128(v0, bias));
                    _mm_storeu_si128(
                        tmp[i][16..].as_mut_ptr() as *mut __m128i,
                        _mm_xor_si128(v1, bias),
                    );
                }
            }
            for g in 0..4usize {
                let dst = &mut ilv[b * 256 + g * 64..b * 256 + g * 64 + 64];
                for i in 0..8 {
                    dst[i * 8..i * 8 + 8].copy_from_slice(&tmp[i][g * 8..g * 8 + 8]);
                }
            }
            const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
            d_vec[b] = _mm256_setr_ps(
                d_w[PERM[0]][b],
                d_w[PERM[1]][b],
                d_w[PERM[2]][b],
                d_w[PERM[3]][b],
                d_w[PERM[4]][b],
                d_w[PERM[5]][b],
                d_w[PERM[6]][b],
                d_w[PERM[7]][b],
            );
        }
    }

    let sub = if q5 { 16i32 } else { 128i32 }; // per-block bsum multiplier to subtract
    let [o0, o1, o2, o3, o4, o5, o6, o7] = outs;
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf_v = _mm256_setzero_ps();
        for b in 0..nb {
            let mut acc = _mm512_setzero_si512();
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            for g in 0..4usize {
                let w = load512(&ilv, b * 256 + g * 64);
                let a = _mm512_set1_epi64((q8b[g * 8..].as_ptr() as *const i64).read_unaligned());
                acc = _mm512_dpbusd_epi32(acc, w, a);
            }
            let lo = _mm512_castsi512_si256(acc);
            let hi = _mm512_extracti64x4_epi64::<1>(acc);
            let sums_perm = _mm256_hadd_epi32(lo, hi);
            // Same expression as the scalar oracles, 8 rows wide (no FMA — each mul/add rounds
            // separately, matching the scalar `d_w * q8.d * iprod` left-assoc sequence): Q8_0 has
            // no bsum term of its own (the −128·bsum was the bias correction, folded into
            // `corr`); Q5_0's −16·bsum likewise.
            let corr = _mm256_set1_epi32(sub * q8.bsum[b]);
            let iprod_f = _mm256_cvtepi32_ps(_mm256_sub_epi32(sums_perm, corr));
            let scale = _mm256_mul_ps(d_vec[b], _mm256_set1_ps(q8.d[b]));
            sumf_v = _mm256_add_ps(sumf_v, _mm256_mul_ps(scale, iprod_f));
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sumf_v);
        o0[r] = lanes[0];
        o1[r] = lanes[1];
        o4[r] = lanes[2];
        o5[r] = lanes[3];
        o2[r] = lanes[4];
        o3[r] = lanes[5];
        o6[r] = lanes[6];
        o7[r] = lanes[7];
    }
}

/// 8-row tile dispatcher for the native-32-block dots (Q8_0/Q5_0): the interleaved VNNI kernel
/// on capable hardware, else 8x the per-row batch kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q32_batch8(
    rows: [&[u8]; 8],
    q8s: &[Q8x32],
    in_f: usize,
    outs: [&mut [f32]; 8],
    q5: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") {
            return unsafe { vec_dot_q32_batch8_ilv_vnni(rows, q8s, in_f, outs, q5) };
        }
    }
    let [r0, r1, r2, r3, r4, r5, r6, r7] = rows;
    let [mut_o0, mut_o1, mut_o2, mut_o3, mut_o4, mut_o5, mut_o6, mut_o7] = outs;
    let per = |row: &[u8], out: &mut [f32]| {
        if q5 {
            vec_dot_q5_0_32_batch(row, q8s, in_f, out);
        } else {
            vec_dot_q8_0_32_batch(row, q8s, in_f, out);
        }
    };
    per(r0, mut_o0);
    per(r1, mut_o1);
    per(r2, mut_o2);
    per(r3, mut_o3);
    per(r4, mut_o4);
    per(r5, mut_o5);
    per(r6, mut_o6);
    per(r7, mut_o7);
}

/// Expand one weight row's Q5_0 codes (5-bit, 0..31, the UNSIGNED pre-`−16` values) into a flat
/// `[nb*32]` u8 buffer ONCE per row — the scalar kernel re-decoded nibble+high-bit per
/// (activation-row, block), which multiplied the decode cost by the batch size. Shared by the
/// SIMD kernels; layout `flat[b*32 + j]` = code j of block b (j 0..15 = lo nibbles, 16..31 = hi).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q5_0_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<u8>, Vec<f32>) {
    let mut flat = vec![0u8; nb * 32];
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = rdf16(&blk[0..2]);
        let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
        let qs = &blk[6..22];
        let f = &mut flat[b * 32..b * 32 + 32];
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            f[j] = (qs[j] as u32 & 0x0F | xh0) as u8;
            f[j + 16] = (qs[j] as u32 >> 4 | xh1) as u8;
        }
    }
    (flat, d_arr)
}

/// AVX2 kernel for `vec_dot_q5_0_32_batch`: codes pre-expanded once (see [`q5_0_expand_codes`]),
/// then one `maddubs(code_u8, q8_s8)` block dot per (row, block) — codes ≤31 × |q8| ≤127 can't
/// saturate the i16 pair sums. Bit-identical to the scalar oracle (integer dot exact; the
/// per-block f32 accumulation expression and order are unchanged).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5_0_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 22usize;
    let ones = _mm256_set1_epi16(1i16);
    let (flat, d_arr) = q5_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let prod = _mm256_maddubs_epi16(code, q8v);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 16.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_q5_0_32_batch`: two blocks per zmm, `dpbusd` in place of the
/// maddubs+madd pair (see the AVX2 variant's bit-identity note; the two per-block f32 adds stay
/// SEPARATE and in scalar order).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5_0_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 22usize;
    let pairs = nb / 2;
    let (flat, d_arr) = q5_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let code_z = load512(&flat, b0 * 32);
            let qx_z = load512(&q8.qs, b0 * 32);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), code_z, qx_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d_arr[b0] * q8.d[b0] * (iprod0 as f32 - 16.0 * q8.bsum[b0] as f32);
            sumf += d_arr[b1] * q8.d[b1] * (iprod1 as f32 - 16.0 * q8.bsum[b1] as f32);
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), code, q8v);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 16.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_q5_0_32_batch` (also the non-x86 path).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q5_0_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 22usize; // f16 d (2B) + u32 qh (4B) + 16 × packed-nibble qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
            let qs = &blk[6..22];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let code0 = ((qs[j] as u32 & 0x0F) | xh0) as i32;
                let code1 = ((qs[j] as u32 >> 4) | xh1) as i32;
                iprod += code0 * q8b[j] as i32 + code1 * q8b[j + 16] as i32;
            }
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 16.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Batched Q4_0 dot at native 32-block granularity: `y = d_w·(code−8)`, `code ∈ 0..15` (4 nibble
/// bits from `qs`, per `dequant_block`'s Q4_0 case) — so
/// `Σy·x = d_w·(Σcode·x − 8·Σx) ≈ d_w·d8·(Σcode·q8 − 8·bsum)`. Q4_0 is [`vec_dot_q5_0_32_batch`]
/// without the 5th (`qh`) bit and with offset 8 not 16; block stride is 18 bytes not 22.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q4_0_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 32,
        0,
        "vec_dot_q4_0_32_batch: in_f must be a multiple of 32"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q4_0_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q4_0_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q4_0_32_batch_scalar(row, q8s, in_f, out);
}

/// Expand one weight row's Q4_0 codes (4-bit, 0..15, the UNSIGNED pre-`−8` values) into a flat
/// `[nb*32]` u8 buffer ONCE per row (mirrors [`q5_0_expand_codes`] without the `qh` high bit).
/// Shared by the SIMD kernels; layout `flat[b*32 + j]` = code j of block b (j 0..15 = lo nibbles,
/// 16..31 = hi).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q4_0_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<u8>, Vec<f32>) {
    let mut flat = vec![0u8; nb * 32];
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = rdf16(&blk[0..2]);
        let qs = &blk[2..18];
        let f = &mut flat[b * 32..b * 32 + 32];
        for j in 0..16 {
            f[j] = qs[j] & 0x0F;
            f[j + 16] = qs[j] >> 4;
        }
    }
    (flat, d_arr)
}

/// AVX2 kernel for `vec_dot_q4_0_32_batch`: codes pre-expanded once (see [`q4_0_expand_codes`]),
/// then one `maddubs(code_u8, q8_s8)` block dot per (row, block) — codes ≤15 × |q8| ≤127 can't
/// saturate the i16 pair sums. Bit-identical to the scalar oracle (integer dot exact; the
/// per-block f32 accumulation expression and order are unchanged). Mirrors
/// [`vec_dot_q5_0_32_batch_avx2`] with offset 8 instead of 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4_0_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 18usize;
    let ones = _mm256_set1_epi16(1i16);
    let (flat, d_arr) = q4_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let prod = _mm256_maddubs_epi16(code, q8v);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 8.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_q4_0_32_batch`: two blocks per zmm, `dpbusd` in place of the
/// maddubs+madd pair (see the AVX2 variant's bit-identity note; the two per-block f32 adds stay
/// SEPARATE and in scalar order). Mirrors [`vec_dot_q5_0_32_batch_vnni`] with offset 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4_0_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 18usize;
    let pairs = nb / 2;
    let (flat, d_arr) = q4_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let code_z = load512(&flat, b0 * 32);
            let qx_z = load512(&q8.qs, b0 * 32);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), code_z, qx_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d_arr[b0] * q8.d[b0] * (iprod0 as f32 - 8.0 * q8.bsum[b0] as f32);
            sumf += d_arr[b1] * q8.d[b1] * (iprod1 as f32 - 8.0 * q8.bsum[b1] as f32);
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), code, q8v);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 8.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_q4_0_32_batch` (also the non-x86 path).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q4_0_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 18usize; // f16 d (2B) + 16 × packed-nibble qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qs = &blk[2..18];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for j in 0..16 {
                let code0 = (qs[j] & 0x0F) as i32;
                let code1 = (qs[j] >> 4) as i32;
                iprod += code0 * q8b[j] as i32 + code1 * q8b[j + 16] as i32;
            }
            sumf += d_arr[b] * q8.d[b] * (iprod as f32 - 8.0 * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Batched Q2_0 dot ("Bonsai ternary", a simple LINEAR-offset quant like Q4_0, 2-bit codes on a
/// **64-weight** block with offset −1: `y = d·(q−1)`, `q ∈ 0..3`). The `Q8x32` activation is
/// 32-element, so each 64-weight Q2_0 block `b` maps to TWO consecutive activation sub-blocks
/// `2b` (elements 0..32) and `2b+1` (32..64), each carrying its own `d`/`bsum`. Per Q2_0 block
/// `Σ_{i} d·(q_i−1)·a_i = Σ_{s∈{2b,2b+1}} q8.d[s]·d·(iprod_s − bsum[s])`, where `iprod_s` is the
/// unsigned·signed dot of the 2-bit codes with that sub-block's int8 activation. `in_f` must be a
/// multiple of 64 (⇒ an even sub-block count); the m>1 dispatch guards on `is_multiple_of(64)`.
/// Block stride is 18 bytes (`[f16 d][u8 qs[16]]`); code `j` is `(qs[j/4] >> ((j%4)*2)) & 3`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q2_0_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    // Q2_0 is a 64-weight block (two 32-elem activation sub-blocks): `nb = in_f / 64`.
    debug_assert_eq!(
        in_f % 64,
        0,
        "vec_dot_q2_0_batch: in_f must be a multiple of 64"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q2_0_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q2_0_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q2_0_batch_scalar(row, q8s, in_f, out);
}

/// Expand one weight row's Q2_0 codes (2-bit, 0..3, the UNSIGNED pre-`−1` values) into a flat
/// `[nb*64]` u8 buffer ONCE per row (mirrors [`q4_0_expand_codes`] with a 64-weight block). Layout
/// `flat[b*64 + j]` = code j of block b, so it aligns 1:1 with `Q8x32::qs` (block b's two sub-blocks
/// occupy flat elements `b*64..b*64+32` and `b*64+32..b*64+64`). `code j = (qs[j/4] >> ((j%4)*2))&3`.
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q2_0_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<u8>, Vec<f32>) {
    let mut flat = vec![0u8; nb * 64];
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = rdf16(&blk[0..2]);
        let qs = &blk[2..18];
        let f = &mut flat[b * 64..b * 64 + 64];
        for (j, fj) in f.iter_mut().enumerate() {
            *fj = (qs[j / 4] >> ((j % 4) * 2)) & 3;
        }
    }
    (flat, d_arr)
}

/// AVX2 kernel for `vec_dot_q2_0_batch`: codes pre-expanded once (see [`q2_0_expand_codes`]), then
/// one `maddubs(code_u8, q8_s8)` block dot per (row, 32-sub-block) — codes ≤3 × |q8| ≤127 can't
/// saturate the i16 pair sums. Bit-identical to the scalar oracle (integer dot exact; the two
/// per-sub-block f32 adds stay SEPARATE and in scalar order). Mirrors [`vec_dot_q4_0_32_batch_avx2`]
/// with offset 1 and two sub-blocks per 64-weight block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q2_0_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 64;
    let bpr = 18usize;
    let ones = _mm256_set1_epi16(1i16);
    let (flat, d_arr) = q2_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let d = d_arr[b];
            for sub in 0..2 {
                let s = 2 * b + sub;
                let code = load256(&flat, s * 32);
                let q8v = load256(&q8.qs, s * 32);
                let prod = _mm256_maddubs_epi16(code, q8v);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod = hadd_i32_ymm(sum32);
                sumf += d * q8.d[s] * (iprod as f32 - q8.bsum[s] as f32);
            }
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_q2_0_batch`: one zmm per 64-weight block (its two 32-sub-blocks
/// fill the 64 bytes exactly), `dpbusd` in place of the maddubs+madd pair — the low 8 i32 lanes are
/// sub-block `2b`, the high 8 are `2b+1`. See the AVX2 variant's bit-identity note; the two per-
/// sub-block f32 adds stay SEPARATE and in scalar order. Mirrors [`vec_dot_q4_0_32_batch_vnni`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q2_0_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 64;
    let bpr = 18usize;
    let (flat, d_arr) = q2_0_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let d = d_arr[b];
            let (s0, s1) = (2 * b, 2 * b + 1);
            let code_z = load512(&flat, s0 * 32);
            let qx_z = load512(&q8.qs, s0 * 32);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), code_z, qx_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d * q8.d[s0] * (iprod0 as f32 - q8.bsum[s0] as f32);
            sumf += d * q8.d[s1] * (iprod1 as f32 - q8.bsum[s1] as f32);
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_q2_0_batch` (also the non-x86 path). Per 64-weight block: two 32-
/// element sub-blocks, each `d · q8.d[s] · (iprod_s − bsum[s])` with `iprod_s = Σ code·qa`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q2_0_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 64;
    let bpr = 18usize; // f16 d (2B) + 16 × 2-bit-packed qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qs = &blk[2..18];
            let d = d_arr[b];
            for sub in 0..2 {
                let s = 2 * b + sub;
                let q8b = &q8.qs[s * 32..s * 32 + 32];
                let mut iprod = 0i32;
                for (p, &qa) in q8b.iter().enumerate() {
                    let j = sub * 32 + p;
                    let code = ((qs[j / 4] >> ((j % 4) * 2)) & 3) as i32;
                    iprod += code * qa as i32;
                }
                sumf += d * q8.d[s] * (iprod as f32 - q8.bsum[s] as f32);
            }
        }
        out[r] = sumf;
    }
}

/// Batched Q4_1 dot at native 32-block granularity. Q4_1 is the AFFINE sibling of Q4_0: same
/// 32-weight block and nibble layout, but `y = d_w·q4 + m_w` (`q4 ∈ 0..15`, NO −8 offset; a
/// separate per-block f16 `m` min is added). Per activation block (`a_i ≈ as·qa_i`, `Q8x32` carries
/// `as` and `bsum = Σqa_i`):
/// `Σy·x = Σ(d_w·q4_i + m_w)·a_i = as·( d_w·Σ(q4_i·qa_i) + m_w·Σqa_i ) = as·( d_w·iprod + m_w·bsum )`.
/// Block stride is 20 bytes (`[f16 d][f16 m][u8 qs[16]]`), not Q4_0's 18. Reuses the same `Q8x32`
/// activation as Q4_0 unchanged; `iprod` is the identical `maddubs/dpbusd` unsigned·signed dot.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q4_1_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 32,
        0,
        "vec_dot_q4_1_32_batch: in_f must be a multiple of 32"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q4_1_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q4_1_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q4_1_32_batch_scalar(row, q8s, in_f, out);
}

/// Expand one weight row's Q4_1 codes (4-bit, 0..15) into a flat `[nb*32]` u8 buffer ONCE per row
/// (mirrors [`q4_0_expand_codes`] with the affine layout: `d` AND `m` per block, block stride 20,
/// `qs` starting at byte 4). Returns `(flat, d_arr, m_arr)`; `flat[b*32 + j]` = code j of block b
/// (j 0..15 = lo nibbles, 16..31 = hi).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q4_1_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut flat = vec![0u8; nb * 32];
    let mut d_arr = vec![0f32; nb];
    let mut m_arr = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = rdf16(&blk[0..2]);
        m_arr[b] = rdf16(&blk[2..4]);
        let qs = &blk[4..20];
        let f = &mut flat[b * 32..b * 32 + 32];
        for j in 0..16 {
            f[j] = qs[j] & 0x0F;
            f[j + 16] = qs[j] >> 4;
        }
    }
    (flat, d_arr, m_arr)
}

/// AVX2 kernel for `vec_dot_q4_1_32_batch`: codes pre-expanded once, one `maddubs(code_u8, q8_s8)`
/// block dot per (row, block) — codes ≤15 × |q8| ≤127 can't saturate the i16 pair sums.
/// Bit-identical to the scalar oracle (integer dot exact; the affine per-block f32 accumulation
/// expression `as·(d·iprod + m·bsum)` and order are unchanged). Mirrors [`vec_dot_q4_0_32_batch_avx2`]
/// with `+m·bsum` in place of `−8·bsum`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4_1_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 20usize;
    let ones = _mm256_set1_epi16(1i16);
    let (flat, d_arr, m_arr) = q4_1_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let prod = _mm256_maddubs_epi16(code, q8v);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += q8.d[b] * (d_arr[b] * iprod as f32 + m_arr[b] * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_q4_1_32_batch`: two blocks per zmm, `dpbusd` in place of the
/// maddubs+madd pair (see the AVX2 variant's bit-identity note; the two per-block f32 adds stay
/// SEPARATE and in scalar order). Mirrors [`vec_dot_q4_0_32_batch_vnni`] with the affine form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q4_1_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 20usize;
    let pairs = nb / 2;
    let (flat, d_arr, m_arr) = q4_1_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let code_z = load512(&flat, b0 * 32);
            let qx_z = load512(&q8.qs, b0 * 32);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), code_z, qx_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += q8.d[b0] * (d_arr[b0] * iprod0 as f32 + m_arr[b0] * q8.bsum[b0] as f32);
            sumf += q8.d[b1] * (d_arr[b1] * iprod1 as f32 + m_arr[b1] * q8.bsum[b1] as f32);
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), code, q8v);
            let iprod = hadd_i32_ymm(sum32);
            sumf += q8.d[b] * (d_arr[b] * iprod as f32 + m_arr[b] * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_q4_1_32_batch` (also the non-x86 path). Affine: `as·(d·iprod + m·bsum)`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q4_1_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 20usize; // f16 d (2B) + f16 m (2B) + 16 × packed-nibble qs
    let mut d_arr = vec![0f32; nb];
    let mut m_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
        m_arr[b] = rdf16(&row[b * bpr + 2..b * bpr + 4]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qs = &blk[4..20];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for j in 0..16 {
                let code0 = (qs[j] & 0x0F) as i32;
                let code1 = (qs[j] >> 4) as i32;
                iprod += code0 * q8b[j] as i32 + code1 * q8b[j + 16] as i32;
            }
            sumf += q8.d[b] * (d_arr[b] * iprod as f32 + m_arr[b] * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Batched Q5_1 dot at native 32-block granularity. Q5_1 is the AFFINE sibling of Q5_0: the same
/// 32-weight, 5-bit block (4 nibble bits from `qs` + 1 high bit from `qh`, `code ∈ 0..31`), but
/// `y = d_w·q5 + m_w` (like Q4_1's `+m`, NOT Q5_0's `−16` offset; a separate per-block f16 `m` min
/// is added). Per activation block (`a_i ≈ as·qa_i`, `Q8x32` carries `as` and `bsum = Σqa_i`):
/// `Σy·x = Σ(d_w·q5_i + m_w)·a_i = as·( d_w·Σ(q5_i·qa_i) + m_w·Σqa_i ) = as·( d_w·iprod + m_w·bsum )`.
/// Block stride is 24 bytes (`[f16 d][f16 m][u8 qh[4]][u8 qs[16]]`); the `qh` 5th-bit assembly is
/// identical to [`vec_dot_q5_0_32_batch`] (per `dequant_block`'s Q5_1 case). Reuses the same `Q8x32`
/// activation unchanged; `iprod` is the identical `maddubs/dpbusd` unsigned·signed dot.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_q5_1_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 32,
        0,
        "vec_dot_q5_1_32_batch: in_f must be a multiple of 32"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_q5_1_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_q5_1_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_q5_1_32_batch_scalar(row, q8s, in_f, out);
}

/// Expand one weight row's Q5_1 codes (5-bit, 0..31, the UNSIGNED values) into a flat `[nb*32]` u8
/// buffer ONCE per row — the affine analogue of [`q5_0_expand_codes`] (`d` AND `m` per block, block
/// stride 24, `qh` at byte 4, `qs` at byte 8). Returns `(flat, d_arr, m_arr)`; `flat[b*32 + j]` =
/// code j of block b (j 0..15 = lo nibbles + `xh0` high bit, 16..31 = hi nibbles + `xh1` high bit).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn q5_1_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut flat = vec![0u8; nb * 32];
    let mut d_arr = vec![0f32; nb];
    let mut m_arr = vec![0f32; nb];
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = rdf16(&blk[0..2]);
        m_arr[b] = rdf16(&blk[2..4]);
        let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
        let qs = &blk[8..24];
        let f = &mut flat[b * 32..b * 32 + 32];
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            f[j] = (qs[j] as u32 & 0x0F | xh0) as u8;
            f[j + 16] = (qs[j] as u32 >> 4 | xh1) as u8;
        }
    }
    (flat, d_arr, m_arr)
}

/// AVX2 kernel for `vec_dot_q5_1_32_batch`: codes pre-expanded once, one `maddubs(code_u8, q8_s8)`
/// block dot per (row, block) — codes ≤31 × |q8| ≤127 can't saturate the i16 pair sums.
/// Bit-identical to the scalar oracle (integer dot exact; the affine per-block f32 accumulation
/// expression `as·(d·iprod + m·bsum)` and order are unchanged). Mirrors [`vec_dot_q5_0_32_batch_avx2`]
/// with `+m·bsum` in place of `−16·bsum`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5_1_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 24usize;
    let ones = _mm256_set1_epi16(1i16);
    let (flat, d_arr, m_arr) = q5_1_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let prod = _mm256_maddubs_epi16(code, q8v);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += q8.d[b] * (d_arr[b] * iprod as f32 + m_arr[b] * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_q5_1_32_batch`: two blocks per zmm, `dpbusd` in place of the
/// maddubs+madd pair (see the AVX2 variant's bit-identity note; the two per-block f32 adds stay
/// SEPARATE and in scalar order). Mirrors [`vec_dot_q5_0_32_batch_vnni`] with the affine form.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_q5_1_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 24usize;
    let pairs = nb / 2;
    let (flat, d_arr, m_arr) = q5_1_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let code_z = load512(&flat, b0 * 32);
            let qx_z = load512(&q8.qs, b0 * 32);
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), code_z, qx_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += q8.d[b0] * (d_arr[b0] * iprod0 as f32 + m_arr[b0] * q8.bsum[b0] as f32);
            sumf += q8.d[b1] * (d_arr[b1] * iprod1 as f32 + m_arr[b1] * q8.bsum[b1] as f32);
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let code = load256(&flat, b * 32);
            let q8v = load256(&q8.qs, b * 32);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), code, q8v);
            let iprod = hadd_i32_ymm(sum32);
            sumf += q8.d[b] * (d_arr[b] * iprod as f32 + m_arr[b] * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_q5_1_32_batch` (also the non-x86 path). Affine `as·(d·iprod + m·bsum)`
/// with Q5_0's 5-bit (4 nibble + 1 high) code assembly.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_q5_1_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 24usize; // f16 d (2B) + f16 m (2B) + u32 qh (4B) + 16 × packed-nibble qs
    let mut d_arr = vec![0f32; nb];
    let mut m_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
        m_arr[b] = rdf16(&row[b * bpr + 2..b * bpr + 4]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
            let qs = &blk[8..24];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let code0 = ((qs[j] as u32 & 0x0F) | xh0) as i32;
                let code1 = ((qs[j] as u32 >> 4) | xh1) as i32;
                iprod += code0 * q8b[j] as i32 + code1 * q8b[j + 16] as i32;
            }
            sumf += q8.d[b] * (d_arr[b] * iprod as f32 + m_arr[b] * q8.bsum[b] as f32);
        }
        out[r] = sumf;
    }
}

/// Batched IQ4_NL dot at native 32-block granularity. IQ4_NL is the FLAT cousin of IQ4_XS: the same
/// signed 16-entry `KVALUES_IQ4NL` codebook, but ONE f16 scale `d` per 32-weight block and NO
/// super-block / per-sub-block `(ls−32)` scales (pure codebook, no affine offset). So it wires
/// IQ4_XS's codebook signed-dot into [`vec_dot_q4_0_32_batch`]'s 32-block shape (block stride 18:
/// `[f16 d][u8 qs[16]]`, `Q8x32` activation). Per block `Σy·x = d·as·Σ(KV[code_i]·qa_i)`, where the
/// codebook weight is signed i8 and the activation is int8 → Q8_0's `abs(w)`/`sign(w)·a` maddubs
/// trick. No `bsum` term (no offset). Dispatches vnni → avx2 → scalar.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_iq4nl_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 32,
        0,
        "vec_dot_iq4nl_32_batch: in_f must be a multiple of 32"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_iq4nl_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_iq4nl_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_iq4nl_32_batch_scalar(row, q8s, in_f, out);
}

/// Expand one weight row's IQ4_NL codebook values into a flat `[nb*32]` SIGNED-i8 buffer ONCE per
/// row (mirrors [`q4_0_expand_codes`], but the nibble is a `pshufb` lookup into `KVALUES_IQ4NL`
/// rather than the raw 0..15 code — the same expansion IQ4_XS's batch kernel does). Block stride 18,
/// `qs` at byte 2; `flat[b*32 + j]` = weight j of block b (j 0..15 = lo nibbles, 16..31 = hi).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[target_feature(enable = "avx2")]
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn iq4nl_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<i8>, Vec<f32>) {
    use std::arch::x86_64::*;
    let mut flat = vec![0i8; nb * 32];
    let mut d_arr = vec![0f32; nb];
    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);
    let table = _mm256_broadcastsi128_si256(load128(&KVALUES_IQ4NL, 0));
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = rdf16(&blk[0..2]);
        let qs = &blk[2..18];
        // 16 code bytes → xmm; nibbles → 32 codes ([lo16 | hi16]); pshufb → 32 signed weights.
        let codes16 = load128(qs, 0);
        let lo = _mm256_castsi128_si256(_mm_and_si128(codes16, _mm256_castsi256_si128(mask_0f)));
        let hi = _mm256_castsi128_si256(_mm_and_si128(
            _mm_srli_epi16(codes16, 4),
            _mm256_castsi256_si128(mask_0f),
        ));
        let codes = _mm256_inserti128_si256::<1>(lo, _mm256_castsi256_si128(hi));
        let w = _mm256_shuffle_epi8(table, codes);
        _mm256_storeu_si256(flat[b * 32..].as_mut_ptr() as *mut __m256i, w);
    }
    (flat, d_arr)
}

/// AVX2 kernel for `vec_dot_iq4nl_32_batch`: signed codebook weights pre-expanded once (see
/// [`iq4nl_expand_codes`]), then per (row, block) the Q8_0 abs/sign maddubs dot (|w| unsigned ×
/// sign(w)·a signed → maddubs; pair sums ≤ 2·127·127 < 2^15). Integer dot is order-free and the
/// per-block f32 accumulation `d·as·iprod` and order match the scalar oracle → bit-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_iq4nl_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 18usize;
    let ones = _mm256_set1_epi16(1i16);
    let (w_flat, d_arr) = iq4nl_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let w = load256(&w_flat, b * 32);
            let a = load256(&q8.qs, b * 32);
            let w_abs = _mm256_abs_epi8(w);
            let a_signed = _mm256_sign_epi8(a, w);
            let prod = _mm256_maddubs_epi16(w_abs, a_signed);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_iq4nl_32_batch`: two blocks per zmm, `dpbusd` in place of the
/// maddubs+madd pair. `dpbusd` is unsigned×signed, so the Q8_0 abs/sign transform is applied per
/// (block, token) — `|w|` unsigned, `sign(w)·a` signed — at ymm granularity (no `_mm512_sign_epi8`),
/// then repacked to zmm (as IQ4_XS's avx512 path does). The two per-block f32 adds stay SEPARATE and
/// in scalar order → bit-identical to the scalar oracle. Mirrors [`vec_dot_q4_0_32_batch_vnni`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_iq4nl_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 18usize;
    let pairs = nb / 2;
    let (w_flat, d_arr) = iq4nl_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let w_z = load512(&w_flat, b0 * 32);
            let a_z = load512(&q8.qs, b0 * 32);
            // Sign trick at ymm level (no _mm512_sign_epi8), then repack into zmm.
            let w0 = _mm512_castsi512_si256(w_z);
            let w1 = _mm512_extracti64x4_epi64::<1>(w_z);
            let a0 = _mm512_castsi512_si256(a_z);
            let a1 = _mm512_extracti64x4_epi64::<1>(a_z);
            let wabs_z = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_abs_epi8(w0)),
                _mm256_abs_epi8(w1),
            );
            let as_z = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_sign_epi8(a0, w0)),
                _mm256_sign_epi8(a1, w1),
            );
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wabs_z, as_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d_arr[b0] * q8.d[b0] * iprod0 as f32;
            sumf += d_arr[b1] * q8.d[b1] * iprod1 as f32;
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let w = load256(&w_flat, b * 32);
            let a = load256(&q8.qs, b * 32);
            let w_abs = _mm256_abs_epi8(w);
            let a_signed = _mm256_sign_epi8(a, w);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_abs, a_signed);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_iq4nl_32_batch` (also the non-x86 path). Signed codebook weight
/// `KV[code]` × int8 activation, summed to an exact i32 `iprod` per 32-block, then ONE f32 multiply
/// `d·as·iprod` per block (no offset/min term).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_iq4nl_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 18usize; // f16 d (2B) + 16 × packed-nibble qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = rdf16(&row[b * bpr..b * bpr + 2]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qs = &blk[2..18];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for j in 0..16 {
                // low nibble → element j (0..15); high nibble → element j+16 (16..31).
                let w_lo = KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as i32;
                let w_hi = KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i32;
                iprod += w_lo * q8b[j] as i32 + w_hi * q8b[j + 16] as i32;
            }
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Batched MXFP4 dot at native 32-block granularity. MXFP4 is structurally identical to IQ4_NL (the
/// FLAT 32-weight-block codebook shape) with exactly two changes: (a) the block is 17 bytes —
/// `[u8 e]` (1-byte E8M0 exponent scale) + `[u8 qs[16]]` — instead of IQ4_NL's 18-byte `[f16 d]` +
/// `qs`, so the per-block scale is `d = 2^(e−128)` decoded by [`e8m0_to_fp32_half`] rather than an
/// f16 read; and (b) the codebook is [`KVALUES_MXFP4`] (E2M1 integer values), not `KVALUES_IQ4NL`.
/// Everything else — the signed-i8 codebook × int8 activation reduction (`Q8x32`), the per-block
/// `d·as·Σ(KV[code_i]·qa_i)` form, and the accumulation order — matches IQ4_NL. Dispatches vnni →
/// avx2 → scalar.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_mxfp4_32_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    debug_assert_eq!(
        in_f % 32,
        0,
        "vec_dot_mxfp4_32_batch: in_f must be a multiple of 32"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_mxfp4_32_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_mxfp4_32_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_mxfp4_32_batch_scalar(row, q8s, in_f, out);
}

/// Expand one MXFP4 weight row's codebook values into a flat `[nb*32]` SIGNED-i8 buffer ONCE per row
/// (mirror of [`iq4nl_expand_codes`]; the two differences are the 17-byte block stride with the
/// E8M0 scale byte at offset 0, and the [`KVALUES_MXFP4`] pshufb table). `qs` at byte 1;
/// `flat[b*32 + j]` = weight j of block b (j 0..15 = lo nibbles, 16..31 = hi).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[target_feature(enable = "avx2")]
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn mxfp4_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<i8>, Vec<f32>) {
    use std::arch::x86_64::*;
    let mut flat = vec![0i8; nb * 32];
    let mut d_arr = vec![0f32; nb];
    let mask_0f = _mm256_set1_epi8(0x0F_u8 as i8);
    let table = _mm256_broadcastsi128_si256(load128(&KVALUES_MXFP4, 0));
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        d_arr[b] = e8m0_to_fp32_half(blk[0]);
        let qs = &blk[1..17];
        // 16 code bytes → xmm; nibbles → 32 codes ([lo16 | hi16]); pshufb → 32 signed weights.
        let codes16 = load128(qs, 0);
        let lo = _mm256_castsi128_si256(_mm_and_si128(codes16, _mm256_castsi256_si128(mask_0f)));
        let hi = _mm256_castsi128_si256(_mm_and_si128(
            _mm_srli_epi16(codes16, 4),
            _mm256_castsi256_si128(mask_0f),
        ));
        let codes = _mm256_inserti128_si256::<1>(lo, _mm256_castsi256_si128(hi));
        let w = _mm256_shuffle_epi8(table, codes);
        _mm256_storeu_si256(flat[b * 32..].as_mut_ptr() as *mut __m256i, w);
    }
    (flat, d_arr)
}

/// AVX2 kernel for `vec_dot_mxfp4_32_batch` (clone of [`vec_dot_iq4nl_32_batch_avx2`]; block stride
/// 17, [`mxfp4_expand_codes`] for the codebook + E8M0 scale). Same Q8_0 abs/sign maddubs dot and
/// per-block `d·as·iprod` accumulation → bit-identical to the scalar oracle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_mxfp4_32_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 17usize;
    let ones = _mm256_set1_epi16(1i16);
    let (w_flat, d_arr) = mxfp4_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let w = load256(&w_flat, b * 32);
            let a = load256(&q8.qs, b * 32);
            let w_abs = _mm256_abs_epi8(w);
            let a_signed = _mm256_sign_epi8(a, w);
            let prod = _mm256_maddubs_epi16(w_abs, a_signed);
            let sum32 = _mm256_madd_epi16(prod, ones);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_mxfp4_32_batch` (clone of [`vec_dot_iq4nl_32_batch_vnni`]; block
/// stride 17, [`mxfp4_expand_codes`] for the codebook + E8M0 scale). Two blocks per zmm via
/// `dpbusd`; the two per-block f32 adds stay SEPARATE and in scalar order → bit-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_mxfp4_32_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 32;
    let bpr = 17usize;
    let pairs = nb / 2;
    let (w_flat, d_arr) = mxfp4_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for k in 0..pairs {
            let (b0, b1) = (2 * k, 2 * k + 1);
            let w_z = load512(&w_flat, b0 * 32);
            let a_z = load512(&q8.qs, b0 * 32);
            // Sign trick at ymm level (no _mm512_sign_epi8), then repack into zmm.
            let w0 = _mm512_castsi512_si256(w_z);
            let w1 = _mm512_extracti64x4_epi64::<1>(w_z);
            let a0 = _mm512_castsi512_si256(a_z);
            let a1 = _mm512_extracti64x4_epi64::<1>(a_z);
            let wabs_z = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_abs_epi8(w0)),
                _mm256_abs_epi8(w1),
            );
            let as_z = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_sign_epi8(a0, w0)),
                _mm256_sign_epi8(a1, w1),
            );
            let sum32_z = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wabs_z, as_z);
            let lo_ymm = _mm512_castsi512_si256(sum32_z);
            let hi_ymm = _mm512_extracti64x4_epi64::<1>(sum32_z);
            let iprod0 = hadd_i32_ymm(lo_ymm);
            let iprod1 = hadd_i32_ymm(hi_ymm);
            sumf += d_arr[b0] * q8.d[b0] * iprod0 as f32;
            sumf += d_arr[b1] * q8.d[b1] * iprod1 as f32;
        }
        if nb % 2 == 1 {
            let b = nb - 1;
            let w = load256(&w_flat, b * 32);
            let a = load256(&q8.qs, b * 32);
            let w_abs = _mm256_abs_epi8(w);
            let a_signed = _mm256_sign_epi8(a, w);
            let sum32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_abs, a_signed);
            let iprod = hadd_i32_ymm(sum32);
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_mxfp4_32_batch` (also the non-x86 path). Clone of
/// [`vec_dot_iq4nl_32_batch_scalar`] with the 17-byte block (E8M0 scale byte at offset 0, `qs` at
/// offset 1) and the [`KVALUES_MXFP4`] codebook. Signed codebook weight × int8 activation → exact
/// i32 `iprod` per 32-block, then ONE f32 multiply `d·as·iprod` per block (no offset/min term).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_mxfp4_32_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 32;
    let bpr = 17usize; // E8M0 e (1B) + 16 × packed-nibble qs
    let mut d_arr = vec![0f32; nb];
    for b in 0..nb {
        d_arr[b] = e8m0_to_fp32_half(row[b * bpr]);
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let blk = &row[b * bpr..b * bpr + bpr];
            let qs = &blk[1..17];
            let q8b = &q8.qs[b * 32..b * 32 + 32];
            let mut iprod = 0i32;
            for j in 0..16 {
                // low nibble → element j (0..15); high nibble → element j+16 (16..31).
                let w_lo = KVALUES_MXFP4[(qs[j] & 0x0F) as usize] as i32;
                let w_hi = KVALUES_MXFP4[(qs[j] >> 4) as usize] as i32;
                iprod += w_lo * q8b[j] as i32 + w_hi * q8b[j + 16] as i32;
            }
            sumf += d_arr[b] * q8.d[b] * iprod as f32;
        }
        out[r] = sumf;
    }
}

/// Batched NVFP4 dot at native 64-block granularity. NVFP4 shares the [`KVALUES_MXFP4`] E2M1
/// codebook with MXFP4 but replaces MXFP4's single per-32-block E8M0 scale with FOUR per-16-element
/// sub-block UE4M3 scales (the same per-sub-block-scale structure IQ4_XS has). Block = 36 bytes,
/// 64 weights: `[u8 scales[4]]` (one UE4M3 per 16-elem sub-block, decoded by [`ue4m3_to_fp32`]) +
/// `[u8 qs[32]]`. For sub-block `s` the 8 code bytes are `qs[s*8..s*8+8]`; low nibble → element
/// `s*16 + j` (j 0..7), high nibble → element `s*16 + j + 8`.
///
/// The `Q8x32` activation is 32-element, so a 64-weight block `b` maps to TWO consecutive activation
/// sub-blocks `t = 2b` and `t = 2b+1`, and EACH `Q8x32` block spans TWO NVFP4 sub-blocks (16+16)
/// with DIFFERENT weight scales but the SAME activation scale `q8.d[t]`. Per NVFP4 sub-block `s`
/// (16 elems), with `t = 2b + s/2`: `Σ (KV[code]·d_s)·a = d_s · q8.d[t] · Σ(KV[code]·qa)` — a
/// 16-element signed int dot, one f32 multiply, summed over the 4 sub-blocks (no offset/min term).
/// `in_f` must be a multiple of 64. Dispatches vnni → avx2 → scalar.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn vec_dot_nvfp4_batch(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    // NVFP4 is a 64-weight block (two 32-elem activation sub-blocks): `nb = in_f / 64`.
    debug_assert_eq!(
        in_f % 64,
        0,
        "vec_dot_nvfp4_batch: in_f must be a multiple of 64"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: features detected at runtime; pointer bounds checked by slice indexing.
            return unsafe { vec_dot_nvfp4_batch_vnni(row, q8s, in_f, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_nvfp4_batch_avx2(row, q8s, in_f, out) };
        }
    }
    vec_dot_nvfp4_batch_scalar(row, q8s, in_f, out);
}

/// Expand one NVFP4 weight row's codebook values into a flat `[nb*64]` SIGNED-i8 buffer ONCE per row
/// (per NVFP4 sub-block `s`: 8 code bytes → 16 signed weights, lo nibbles → the sub-block's first 8
/// elements, hi nibbles → its last 8, via [`KVALUES_MXFP4`] pshufb) and decode the four per-sub-block
/// UE4M3 scales into `d_arr[b*4 + s]`. `flat[b*64 + s*16 + k]` = element `k` of sub-block `s` of
/// block `b`, so it aligns 1:1 with `Q8x32::qs` (element `b*64 + s*16 + k` lives at `q8.qs` index
/// `(2b + s/2)*32 + (s%2)*16 + k` = `b*64 + s*16 + k`).
#[cfg(target_arch = "x86_64")] // only the x86 SIMD kernels call this — dead code on aarch64
#[target_feature(enable = "avx2")]
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn nvfp4_expand_codes(row: &[u8], nb: usize, bpr: usize) -> (Vec<i8>, Vec<f32>) {
    use std::arch::x86_64::*;
    let mut flat = vec![0i8; nb * 64];
    let mut d_arr = vec![0f32; nb * 4];
    let mask_0f = _mm_set1_epi8(0x0F_u8 as i8);
    let table = load128(&KVALUES_MXFP4, 0);
    for b in 0..nb {
        let blk = &row[b * bpr..b * bpr + bpr];
        for s in 0..4usize {
            d_arr[b * 4 + s] = ue4m3_to_fp32(blk[s]);
        }
        let qs = &blk[4..36];
        for s in 0..4usize {
            // 8 code bytes → xmm low 8 bytes; nibbles → 16 codes ([lo8 | hi8]); pshufb → 16 weights.
            let c8 = _mm_loadl_epi64(qs[s * 8..].as_ptr() as *const __m128i);
            let lo = _mm_and_si128(c8, mask_0f);
            let hi = _mm_and_si128(_mm_srli_epi16(c8, 4), mask_0f);
            let codes = _mm_or_si128(lo, _mm_slli_si128(hi, 8));
            let w = _mm_shuffle_epi8(table, codes);
            _mm_storeu_si128(flat[b * 64 + s * 16..].as_mut_ptr() as *mut __m128i, w);
        }
    }
    (flat, d_arr)
}

/// AVX2 kernel for `vec_dot_nvfp4_batch`: codes pre-expanded once (see [`nvfp4_expand_codes`]), then
/// one 32-element `maddubs` per (row, `Q8x32` sub-block `t`). The two NVFP4 sub-blocks packed in that
/// 32-block share the activation scale but carry different weight scales, so the `madd` output's low
/// 4 i32 lanes (elements 0..15) and high 4 (16..31) are reduced SEPARATELY into the two per-sub-block
/// `iprod`s. Bit-identical to the scalar oracle (integer dot exact; per-sub-block f32 adds in scalar
/// order 0,1,2,3).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_nvfp4_batch_avx2(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 64;
    let bpr = 36usize;
    let ones = _mm256_set1_epi16(1i16);
    let (w_flat, d_arr) = nvfp4_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            for u in 0..2usize {
                let t = 2 * b + u; // Q8x32 activation sub-block
                let s0 = 2 * u; // NVFP4 sub-blocks s0 (elems 0..15) and s0+1 (16..31) of this 32-block
                let w = load256(&w_flat, b * 64 + s0 * 16);
                let a = load256(&q8.qs, t * 32);
                let w_abs = _mm256_abs_epi8(w);
                let a_signed = _mm256_sign_epi8(a, w);
                let prod = _mm256_maddubs_epi16(w_abs, a_signed);
                let sum32 = _mm256_madd_epi16(prod, ones);
                let iprod0 = hadd_i32_xmm(_mm256_castsi256_si128(sum32));
                let iprod1 = hadd_i32_xmm(_mm256_extracti128_si256::<1>(sum32));
                sumf += d_arr[b * 4 + s0] * q8.d[t] * iprod0 as f32;
                sumf += d_arr[b * 4 + s0 + 1] * q8.d[t] * iprod1 as f32;
            }
        }
        out[r] = sumf;
    }
}

/// AVX512-VNNI kernel for `vec_dot_nvfp4_batch`: one 128-bit `dpbusd` per 16-element NVFP4 sub-block
/// (the scale granularity), reducing its 4 i32 lanes to `iprod_s`. Adapts MXFP4's VNNI dot to the
/// per-sub-block scale layout. Bit-identical to the scalar oracle (integer dot exact; per-sub-block
/// f32 adds in scalar order 0,1,2,3).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni,avx512vl")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
unsafe fn vec_dot_nvfp4_batch_vnni(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = in_f / 64;
    let bpr = 36usize;
    let (w_flat, d_arr) = nvfp4_expand_codes(row, nb, bpr);
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            for s in 0..4usize {
                let t = 2 * b + s / 2;
                let w = load128(&w_flat, b * 64 + s * 16);
                let a = load128(&q8.qs, t * 32 + (s % 2) * 16);
                let w_abs = _mm_abs_epi8(w);
                let a_signed = _mm_sign_epi8(a, w);
                let sum32 = _mm_dpbusd_epi32(_mm_setzero_si128(), w_abs, a_signed);
                let iprod = hadd_i32_xmm(sum32);
                sumf += d_arr[b * 4 + s] * q8.d[t] * iprod as f32;
            }
        }
        out[r] = sumf;
    }
}

/// Scalar oracle for `vec_dot_nvfp4_batch` (also the non-x86 path). Per 64-weight block: four 16-
/// element sub-blocks, each `d_s · q8.d[t] · iprod_s` with `t = 2b + s/2`, `iprod_s = Σ KV[code]·qa`
/// (signed codebook weight × int8 activation). No offset/min term.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn vec_dot_nvfp4_batch_scalar(row: &[u8], q8s: &[Q8x32], in_f: usize, out: &mut [f32]) {
    let nb = in_f / 64;
    let bpr = 36usize; // 4 × UE4M3 sub-block scale + 32 × packed-nibble qs
    let mut d_arr = vec![0f32; nb * 4];
    for b in 0..nb {
        for s in 0..4usize {
            d_arr[b * 4 + s] = ue4m3_to_fp32(row[b * bpr + s]);
        }
    }
    for (r, q8) in q8s.iter().enumerate() {
        let mut sumf = 0f32;
        for b in 0..nb {
            let qs = &row[b * bpr + 4..b * bpr + 36];
            for s in 0..4usize {
                let t = 2 * b + s / 2; // Q8x32 activation sub-block
                let qa = &q8.qs[t * 32 + (s % 2) * 16..t * 32 + (s % 2) * 16 + 16];
                let code = &qs[s * 8..s * 8 + 8];
                let mut iprod = 0i32;
                for j in 0..8usize {
                    // low nibble → element j (0..7); high nibble → element j+8 (8..15).
                    let w_lo = KVALUES_MXFP4[(code[j] & 0x0F) as usize] as i32;
                    let w_hi = KVALUES_MXFP4[(code[j] >> 4) as usize] as i32;
                    iprod += w_lo * qa[j] as i32 + w_hi * qa[j + 8] as i32;
                }
                sumf += d_arr[b * 4 + s] * q8.d[t] * iprod as f32;
            }
        }
        out[r] = sumf;
    }
}

/// `Σ f16_weight·x` (weight is 2 bytes/elem). `target-cpu=native` lowers the f16→f32 to F16C.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dot_f16(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        for (j, ac) in acc.iter_mut().enumerate() {
            let i = c * 8 + j;
            let wv = half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]]).to_f32();
            *ac = wv.mul_add(x[i], *ac);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s = half::f16::from_le_bytes([w[i * 2], w[i * 2 + 1]])
            .to_f32()
            .mul_add(x[i], s);
    }
    s
}

/// `Σ bf16_weight·x` (bf16 = top 16 bits of f32). Uses 8 independent accumulators — the same
/// chunked structure as [`dot`]/[`dot_f16`] — so the reduction isn't a latency-bound FMA chain
/// (the old single serial accumulator was several× slower on the bf16-weight hot path). NOTE:
/// the 8-lane summation order differs from that serial chain, so this is NOT bit-identical to the
/// previous `dot_bf16`; it now matches `dot`/`dot_f16` bit-for-bit for the same bf16-rounded
/// weights (a legitimate float reorder, not a numeric bug).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dot_bf16(w: &[u8], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = x.len();
    let chunks = n / 8;
    for c in 0..chunks {
        for (j, ac) in acc.iter_mut().enumerate() {
            let i = c * 8 + j;
            let wv = f32::from_bits((u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]) as u32) << 16);
            *ac = wv.mul_add(x[i], *ac);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        let wv = f32::from_bits((u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]) as u32) << 16);
        s = wv.mul_add(x[i], s);
    }
    s
}

/// Dot product with 8 independent accumulators so the reduction isn't latency-bound — lets the
/// autovectorizer (with `target-cpu=native`) keep several AVX FMA lanes in flight. `mul_add`
/// fuses each lane's multiply+add into one FMA (numerics policy: llama.cpp's f32 dots are FMA
/// too) — this fn is attention's QK score and the F32-weight GEMM fallbacks (e.g. DG's router).
#[inline]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: a and b must have equal length");
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = [0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for (j, ac) in acc.iter_mut().enumerate() {
            *ac = a[base + j].mul_add(b[base + j], *ac);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..n {
        s = a[i].mul_add(b[i], s);
    }
    s
}

/// Gated-FFN activation applied to the gate value.
// per-call leaf, too small to probe (see docs/perf/playbook.md)
#[cfg_attr(infr_profile, infr_prof::skip)]
pub(crate) fn act_fn(act: Activation, g: f32) -> f32 {
    match act {
        Activation::Silu => g / (1.0 + (-g).exp()),
        // gelu_pytorch_tanh: 0.5 g (1 + tanh(√(2/π)·(g + 0.044715 g³)))
        Activation::Gelu => 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
        Activation::Sigmoid => 1.0 / (1.0 + (-g).exp()),
    }
}

/// One gated-FFN output element, `act(gate) * up`, with DeepSeek V4's optional per-layer SwiGLU
/// clamp: `Some(limit)` gives `act(min(gate, limit)) * clamp(up, -limit, limit)` — `up` clamped
/// symmetrically, the gate one-sided and BEFORE the activation (see `Op::GatedAct::swiglu_clamp`).
/// `None` compiles to exactly the un-clamped expression this replaced.
#[inline]
pub(crate) fn gated_act_fn(act: Activation, clamp: Option<f32>, gate: f32, up: f32) -> f32 {
    match clamp {
        None => act_fn(act, gate) * up,
        Some(l) => act_fn(act, gate.min(l)) * up.clamp(-l, l),
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;
    use crate::quant::{quantize_q8, quantize_q8_32};
    // These pack/gemm helpers exist only on x86_64 (AVX2/AVX512-VNNI); the tests that use them
    // gate their bodies on the same arch, so the import must be gated too or it fails to resolve
    // on aarch64.
    #[cfg(target_arch = "x86_64")]
    use crate::repack::{q4k_gemm_group, q4k_pack, q6k_gemm_group, q6k_pack};
    use infr_core::tensor::DType;
    use infr_gguf::dequant::{dequant_block, dequant_codebook};

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed
    }
    fn det_bytes(n: usize, mut seed: u64) -> Vec<u8> {
        (0..n).map(|_| (lcg(&mut seed) >> 33) as u8).collect()
    }
    fn det_x(n: usize, mut seed: u64) -> Vec<f32> {
        (0..n)
            .map(|_| ((lcg(&mut seed) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
            .collect()
    }
    fn put_f16(b: &mut [u8], v: f32) {
        b.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    /// The reference activation the integer kernels actually see: `d8 * q8` per super-block.
    fn dequant_q8(q8: &Q8) -> Vec<f32> {
        let mut x = vec![0f32; q8.qs.len()];
        for (b, &d) in q8.d.iter().enumerate() {
            for i in 0..256 {
                x[b * 256 + i] = d * q8.qs[b * 256 + i] as f32;
            }
        }
        x
    }
    /// The reference activation the `_32` native-block int8 kernels actually see: `d*q8` per 32-block.
    fn dequant_q8_32(q8: &Q8x32) -> Vec<f32> {
        let mut x = vec![0f32; q8.qs.len()];
        for (b, &d) in q8.d.iter().enumerate() {
            for i in 0..32 {
                x[b * 32 + i] = d * q8.qs[b * 32 + i] as f32;
            }
        }
        x
    }
    fn rel_err(got: f32, want: f32) -> f32 {
        (got - want).abs() / want.abs().max(1.0)
    }

    #[test]
    fn q4k_dot_matches_dequant_reference() {
        let in_f = 768; // 3 super-blocks
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 144, 1);
        for k in 0..nb {
            put_f16(&mut w[k * 144..k * 144 + 2], 0.05); // d
            put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015); // dmin
        }
        let wref = dequant_block(DType::Q4K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 2));
        let got = vec_dot_q4k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q4k: got {got}, want {want}");
    }

    #[test]
    fn q6k_dot_matches_dequant_reference() {
        let in_f = 768;
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 210, 3);
        for k in 0..nb {
            put_f16(&mut w[k * 210 + 208..k * 210 + 210], 0.04); // d
        }
        let wref = dequant_block(DType::Q6K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 4));
        let got = vec_dot_q6k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q6k: got {got}, want {want}");
    }

    #[test]
    fn q8_0_dot_matches_dequant_reference() {
        // Q8_0: 34 bytes / 32 elems. in_f must be a multiple of 256 (activation super-block size).
        let in_f = 512; // 2 super-blocks = 16 Q8_0 weight blocks
        let nb_w = in_f / 32;
        let mut w = det_bytes(nb_w * 34, 9);
        for k in 0..nb_w {
            put_f16(&mut w[k * 34..k * 34 + 2], 0.03); // d
        }
        let wref = dequant_block(DType::Q8_0, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 10));
        let got = vec_dot_q8_0(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q8_0: got {got}, want {want}");
    }

    #[test]
    fn q5k_dot_matches_dequant_reference() {
        // Q5_K: 176 bytes / 256 elems. in_f must be a multiple of 256.
        let in_f = 512; // 2 Q5K blocks = 2 super-blocks
        let nb = in_f / 256;
        let mut w = det_bytes(nb * 176, 11);
        for k in 0..nb {
            put_f16(&mut w[k * 176..k * 176 + 2], 0.05); // d
            put_f16(&mut w[k * 176 + 2..k * 176 + 4], 0.01); // dmin
        }
        let wref = dequant_block(DType::Q5K, &w).unwrap();
        let q8 = quantize_q8(&det_x(in_f, 12));
        let got = vec_dot_q5k(&w, &q8, in_f);
        let want = dot(&wref, &dequant_q8(&q8));
        assert!(rel_err(got, want) < 1e-3, "q5k: got {got}, want {want}");
    }

    /// MoE `down`'s native-32-block fast path (`in_f` not a multiple of 256, e.g. DiffusionGemma's
    /// 704): the batched Q8_0 kernel must match the trusted `dequant_block` reference, for BOTH a
    /// 256-aligned and a NON-256-aligned `in_f` (704 = 22×32, not a multiple of 256), and across a
    /// multi-row batch (exercises the "decode weight row once, reuse across tokens" path itself).
    #[test]
    fn q8_0_32_batch_matches_dequant_reference() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let mut w = det_bytes(nb * 34, 20);
            for k in 0..nb {
                put_f16(&mut w[k * 34..k * 34 + 2], 0.02);
            }
            let wref = dequant_block(DType::Q8_0, &w).unwrap();
            let xs: Vec<Vec<f32>> = (0..3).map(|i| det_x(in_f, 21 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();
            let mut got = vec![0f32; 3];
            vec_dot_q8_0_32_batch(&w, &q8s, in_f, &mut got);
            for (r, q8) in q8s.iter().enumerate() {
                let want = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want) < 1e-3,
                    "q8_0_32 in_f={in_f} row={r}: got {got_r}, want {want}",
                    got_r = got[r]
                );
            }
        }
    }

    /// The AVX2/AVX-512BW kernels for `vec_dot_q8_0_32_batch` must be BIT-IDENTICAL to the scalar
    /// oracle (`_scalar`) — integer dot products have no rounding, so summation order doesn't matter,
    /// only the final `d_w * d8 * iprod` formula does, which is shared. Runs whichever SIMD tier this
    /// CPU actually has (falls through to the same scalar fn on non-x86 or pre-AVX2 hardware, in which
    /// case the assertion is trivially true).
    #[test]
    fn q6k_pack_gemm_bit_identical_to_batch() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")) {
                return; // the pack/gemm pair only exists on the VNNI path
            }
            let (in_f, out_f, m) = (512usize, 16usize, 5usize);
            let nb256 = in_f / 256;
            let mut w = det_bytes(out_f * nb256 * 210, 90);
            for o in 0..out_f {
                for b in 0..nb256 {
                    put_f16(
                        &mut w[(o * nb256 + b) * 210 + 208..(o * nb256 + b) * 210 + 210],
                        0.03,
                    );
                }
            }
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let pack = q6k_pack(&w, in_f, out_f);
            let bpr = w.len() / out_f;
            for (g, pg) in pack.groups.iter().enumerate() {
                let mut cols = vec![0f32; 8 * m];
                unsafe { q6k_gemm_group(pg, pack.nb, &q8s, &mut cols) };
                for i in 0..8 {
                    let o = g * 8 + i;
                    let mut want = vec![0f32; m];
                    vec_dot_q6k_batch(&w[o * bpr..o * bpr + bpr], &q8s, in_f, &mut want);
                    for r in 0..m {
                        assert_eq!(
                            cols[i * m + r].to_bits(),
                            want[r].to_bits(),
                            "q6k pack/gemm g={g} i={i} r={r}: got {}, want {}",
                            cols[i * m + r],
                            want[r],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q4k_pack_gemm_bit_identical_to_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !(is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")) {
                return; // the pack/gemm pair only exists on the VNNI path
            }
            let (in_f, out_f, m) = (512usize, 16usize, 5usize);
            let nb256 = in_f / 256;
            let mut w = det_bytes(out_f * nb256 * 144, 80);
            for o in 0..out_f {
                for b in 0..nb256 {
                    put_f16(
                        &mut w[(o * nb256 + b) * 144..(o * nb256 + b) * 144 + 2],
                        0.02,
                    );
                    put_f16(
                        &mut w[(o * nb256 + b) * 144 + 2..(o * nb256 + b) * 144 + 4],
                        0.01,
                    );
                }
            }
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 81 + r as u64)))
                .collect();
            let pack = unsafe { q4k_pack(&w, in_f, out_f) };
            let bpr = w.len() / out_f;
            for (g, pg) in pack.groups.iter().enumerate() {
                let mut cols = vec![0f32; 8 * m];
                unsafe { q4k_gemm_group(pg, pack.nb, &q8s, &mut cols) };
                for i in 0..8 {
                    let o = g * 8 + i;
                    let mut want = vec![0f32; m];
                    vec_dot_q4k_batch_scalar(&w[o * bpr..o * bpr + bpr], &q8s, in_f, &mut want);
                    for r in 0..m {
                        assert_eq!(
                            cols[i * m + r].to_bits(),
                            want[r].to_bits(),
                            "pack/gemm g={g} i={i} r={r}: got {}, want {}",
                            cols[i * m + r],
                            want[r]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn q32_batch8_bit_identical_to_scalar() {
        for q5 in [false, true] {
            let bpr = if q5 { 22usize } else { 34 };
            let in_f = 704usize;
            let nb = in_f / 32;
            let m = 5usize;
            let mut ws: Vec<Vec<u8>> = (0..8)
                .map(|i| {
                    let mut w = det_bytes(nb * bpr, 60 + i as u64);
                    for k in 0..nb {
                        put_f16(&mut w[k * bpr..k * bpr + 2], 0.02);
                    }
                    w
                })
                .collect();
            let _ = &mut ws;
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 70 + r as u64)))
                .collect();
            let mut got: Vec<Vec<f32>> = vec![vec![0f32; m]; 8];
            {
                let rows: [&[u8]; 8] = std::array::from_fn(|i| ws[i].as_slice());
                let mut it = got.iter_mut();
                let outs: [&mut [f32]; 8] =
                    std::array::from_fn(|_| it.next().unwrap().as_mut_slice());
                vec_dot_q32_batch8(rows, &q8s, in_f, outs, q5);
            }
            for i in 0..8 {
                let mut want = vec![0f32; m];
                if q5 {
                    vec_dot_q5_0_32_batch_scalar(&ws[i], &q8s, in_f, &mut want);
                } else {
                    vec_dot_q8_0_32_batch_scalar(&ws[i], &q8s, in_f, &mut want);
                }
                for r in 0..m {
                    assert_eq!(
                        got[i][r].to_bits(),
                        want[r].to_bits(),
                        "q32 batch8 q5={q5} row={i} r={r}: got {}, want {}",
                        got[i][r],
                        want[r]
                    );
                }
            }
        }
    }

    #[test]
    fn q5_0_32_batch_simd_bit_identical_to_scalar() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let mut w = det_bytes(nb * 22, 41);
            for k in 0..nb {
                put_f16(&mut w[k * 22..k * 22 + 2], 0.02);
            }
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 42 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q5_0_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q5_0_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q5_0_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
        }
    }

    #[test]
    fn q8_0_32_batch_simd_bit_identical_to_scalar() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let mut w = det_bytes(nb * 34, 25);
            for k in 0..nb {
                put_f16(&mut w[k * 34..k * 34 + 2], 0.02);
            }
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 26 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q8_0_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q8_0_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q8_0_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r],
                );
            }
        }
    }

    /// Same shape of check as `q8_0_32_batch_matches_dequant_reference`, for Q5_0's 5-bit (4 nibble
    /// + 1 high) native block.
    #[test]
    fn q5_0_32_batch_matches_dequant_reference() {
        for in_f in [256usize, 704] {
            let nb = in_f / 32;
            let mut w = det_bytes(nb * 22, 30);
            for k in 0..nb {
                put_f16(&mut w[k * 22..k * 22 + 2], 0.03);
            }
            let wref = dequant_block(DType::Q5_0, &w).unwrap();
            let xs: Vec<Vec<f32>> = (0..3).map(|i| det_x(in_f, 31 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();
            let mut got = vec![0f32; 3];
            vec_dot_q5_0_32_batch(&w, &q8s, in_f, &mut got);
            for (r, q8) in q8s.iter().enumerate() {
                let want = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want) < 1e-3,
                    "q5_0_32 in_f={in_f} row={r}: got {got_r}, want {want}",
                    got_r = got[r]
                );
            }
        }
    }

    /// SIMD (VNNI/AVX2, whichever this host has) must match the Q4_0 scalar oracle bit-for-bit:
    /// the integer dot is exact and the per-block f32 accumulation order is identical. 96 blocks-odd
    /// exercises the VNNI pair-plus-tail path.
    #[test]
    fn q4_0_32_batch_simd_bit_identical_to_scalar() {
        for in_f in [96usize, 256, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let mut w = det_bytes(nb * 18, 72);
            for k in 0..nb {
                put_f16(&mut w[k * 18..k * 18 + 2], 0.02);
            }
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 73 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q4_0_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q4_0_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q4_0_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity for the Q4_0 native-block int8 kernel: the int8-quantized-
    /// activation dot must track the FULL-PRECISION `d*(q−8)`-dequant · f32-activation reference.
    /// int8 activation quant is lossy, so this is a tolerance (not bit-identity). Covers the batch
    /// (m>1) and single-row (m=1) entries, several in_f, both nibble halves nonzero.
    #[test]
    fn q4_0_32_batch_matches_dequant_reference() {
        for in_f in [32usize, 256, 512] {
            let nb = in_f / 32;
            let mut w = det_bytes(nb * 18, 70);
            for k in 0..nb {
                put_f16(&mut w[k * 18..k * 18 + 2], 0.03);
            }
            let wref = dequant_block(DType::Q4_0, &w).unwrap();
            // Assert both nibble halves are actually nonzero somewhere (decode exercises hi+lo).
            assert!(wref[..32].iter().any(|&v| v != 0.0));
            // Full-precision activations (NOT the quantized ones) → proves end-to-end accuracy.
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // Batch entry (m>1). Tight 1e-3 vs the QUANTIZED activation the kernel actually sees
            // (isolates integer-dot correctness, exactly as the Q5_0 sibling), plus a looser 2e-2
            // vs the full-precision activation (absorbs the lossy int8 activation quant; the small
            // in_f=32 single-block case has little sign cancellation, so 1e-2 would be marginal).
            let mut got = vec![0f32; xs.len()];
            vec_dot_q4_0_32_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "q4_0_32 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "q4_0_32 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row entry (m=1): one activation block-set, one-element out.
            for (r, x) in xs.iter().enumerate() {
                let mut one = [0f32; 1];
                vec_dot_q4_0_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                let want = dot(&wref, x);
                assert!(
                    rel_err(one[0], want) < 2e-2,
                    "q4_0_32 single in_f={in_f} row={r}: got {}, want {want}",
                    one[0]
                );
            }
        }
    }

    /// SIMD (VNNI/AVX2, whichever this host has) must match the Q2_0 scalar oracle bit-for-bit: the
    /// integer dot is exact and the per-sub-block f32 accumulation order is identical. Covers the
    /// single-block (in_f=64), 256-aligned, and non-256 (704 = 11×64) cases; full-random qs so all
    /// four 2-bit codes appear (asserted); batch (m>1) and single-row (m=1) dispatch entries.
    #[test]
    fn q2_0_simd_bit_identical_to_scalar() {
        for in_f in [64usize, 256, 704] {
            let nb = in_f / 64;
            let m = 5usize;
            let mut w = det_bytes(nb * 18, 200);
            for k in 0..nb {
                put_f16(&mut w[k * 18..k * 18 + 2], 0.02);
            }
            // All four 2-bit code values must be present (exercises q ∈ {0,1,2,3} decode).
            let mut seen = [false; 4];
            for b in 0..nb {
                let qs = &w[b * 18 + 2..b * 18 + 18];
                for j in 0..64usize {
                    seen[((qs[j / 4] >> ((j % 4) * 2)) & 3) as usize] = true;
                }
            }
            assert!(
                seen.iter().all(|&s| s),
                "q2_0 in_f={in_f}: not all four codes present"
            );
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 201 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q2_0_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q2_0_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q2_0 batch in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
            // Single-row (m=1) dispatch must agree too.
            for r in 0..m {
                let mut one = [0f32; 1];
                vec_dot_q2_0_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                assert_eq!(
                    one[0].to_bits(),
                    scalar_out[r].to_bits(),
                    "q2_0 single in_f={in_f} row={r}"
                );
            }
        }
    }

    /// End-to-end tolerance-parity for the Q2_0 native-block int8 kernel: the int8-quantized-
    /// activation dot must track the FULL-PRECISION `d·(q−1)`-dequant · f32-activation reference.
    /// int8 activation quant is lossy, so this is a tolerance (not bit-identity). Covers batch
    /// (m>1) and single-row (m=1), several in_f; tight 1e-3 vs the quantized activation the kernel
    /// actually sees, loose 2e-2 vs the full-precision activation.
    #[test]
    fn q2_0_matches_dequant_reference() {
        for in_f in [64usize, 256, 704] {
            let nb = in_f / 64;
            let mut w = det_bytes(nb * 18, 210);
            for k in 0..nb {
                put_f16(&mut w[k * 18..k * 18 + 2], 0.03);
            }
            let wref = dequant_block(DType::Q2_0, &w).unwrap();
            // Full-random qs → decode exercises the whole {−d,0,d,2d} range.
            assert!(wref[..64].iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 211 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // Batch entry (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_q2_0_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "q2_0 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "q2_0 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row entry (m=1).
            for (r, x) in xs.iter().enumerate() {
                let mut one = [0f32; 1];
                vec_dot_q2_0_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                let want = dot(&wref, x);
                assert!(
                    rel_err(one[0], want) < 2e-2,
                    "q2_0 single in_f={in_f} row={r}: got {}, want {want}",
                    one[0]
                );
            }
        }
    }

    /// Build `nb` real 20-byte Q4_1 blocks: fixed per-block `d`/`m`, random nibbles (both halves
    /// exercised). `d`/`m` land weights at Q4_0 scale so the int8-activation dot tracks the
    /// full-precision reference within a tight tolerance.
    fn build_q4_1(nb: usize, d: f32, m: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 20, seed);
        for k in 0..nb {
            put_f16(&mut w[k * 20..k * 20 + 2], d);
            put_f16(&mut w[k * 20 + 2..k * 20 + 4], m);
        }
        w
    }

    /// SIMD (VNNI/AVX2, whichever this host has) must match the Q4_1 scalar oracle bit-for-bit: the
    /// integer dot is exact and the affine per-block f32 accumulation order is identical. 96
    /// blocks-odd exercises the VNNI pair-plus-tail path.
    #[test]
    fn q4_1_32_batch_simd_bit_identical_to_scalar() {
        for in_f in [96usize, 256, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let w = build_q4_1(nb, 0.02, -0.01, 82);
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 83 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q4_1_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q4_1_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q4_1_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
            // Single-row (m=1) dispatch must agree too.
            for r in 0..m {
                let mut one = [0f32; 1];
                vec_dot_q4_1_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                assert_eq!(
                    one[0].to_bits(),
                    scalar_out[r].to_bits(),
                    "q4_1_32 single in_f={in_f} row={r}"
                );
            }
        }
    }

    /// End-to-end tolerance-parity for the Q4_1 native-block int8 kernel: the int8-quantized-
    /// activation dot must track the FULL-PRECISION `d*q4 + m`-dequant · f32-activation reference.
    /// int8 activation quant is lossy, so this is a tolerance (not bit-identity). Covers the batch
    /// (m>1) and single-row (m=1) entries, several in_f, both nibble halves nonzero.
    #[test]
    fn q4_1_32_batch_matches_dequant_reference() {
        for in_f in [32usize, 256, 512] {
            let nb = in_f / 32;
            let w = build_q4_1(nb, 0.03, -0.02, 90);
            let wref = dequant_block(DType::Q4_1, &w).unwrap();
            // Assert both nibble halves decode to varying values (decode exercises lo AND hi).
            assert!(wref[..16].iter().any(|&v| v != wref[0]));
            assert!(wref[16..32].iter().any(|&v| v != wref[16]));
            // Full-precision activations (NOT the quantized ones) → proves end-to-end accuracy.
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 91 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // Batch entry (m>1). Tight 1e-3 vs the QUANTIZED activation the kernel actually sees
            // (isolates integer-dot correctness), plus a looser 2e-2 vs the full-precision
            // activation (absorbs the lossy int8 activation quant).
            let mut got = vec![0f32; xs.len()];
            vec_dot_q4_1_32_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "q4_1_32 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "q4_1_32 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row entry (m=1): one activation block-set, one-element out.
            for (r, x) in xs.iter().enumerate() {
                let mut one = [0f32; 1];
                vec_dot_q4_1_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                let want = dot(&wref, x);
                assert!(
                    rel_err(one[0], want) < 2e-2,
                    "q4_1_32 single in_f={in_f} row={r}: got {}, want {want}",
                    one[0]
                );
            }
        }
    }

    /// Build `nb` real 24-byte Q5_1 blocks: fixed per-block `d`/`m`, random `qh`+`qs` (both nibble
    /// halves AND the 5th `qh` bit exercised). `d`/`m` land weights at Q4_0 scale so the
    /// int8-activation dot tracks the full-precision reference within a tight tolerance.
    fn build_q5_1(nb: usize, d: f32, m: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 24, seed);
        for k in 0..nb {
            put_f16(&mut w[k * 24..k * 24 + 2], d);
            put_f16(&mut w[k * 24 + 2..k * 24 + 4], m);
        }
        w
    }

    /// Count Q5_1 codes that exceed 15 (i.e. the `qh` 5th bit is set) across a built row — used to
    /// prove the tests actually exercise the high bit, not just the low nibble.
    fn q5_1_codes_over_15(w: &[u8], nb: usize) -> usize {
        let mut n = 0usize;
        for k in 0..nb {
            let blk = &w[k * 24..k * 24 + 24];
            let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
            let qs = &blk[8..24];
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let c0 = (qs[j] as u32 & 0x0F) | xh0;
                let c1 = (qs[j] as u32 >> 4) | xh1;
                n += (c0 > 15) as usize + (c1 > 15) as usize;
            }
        }
        n
    }

    /// SIMD (VNNI/AVX2, whichever this host has) must match the Q5_1 scalar oracle bit-for-bit: the
    /// integer dot is exact and the affine per-block f32 accumulation order is identical. 96
    /// blocks-odd exercises the VNNI pair-plus-tail path. `qh`+`qs` are fully random, so both nibble
    /// halves and the 5th (`qh`) bit are exercised (asserted via `q5_1_codes_over_15`).
    #[test]
    fn q5_1_32_simd_bit_identical_to_scalar() {
        for in_f in [96usize, 256, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let w = build_q5_1(nb, 0.02, -0.01, 111);
            // The 5th bit must actually fire somewhere, else the qh-assembly path is untested.
            assert!(
                q5_1_codes_over_15(&w, nb) > 0,
                "q5_1_32 in_f={in_f}: no code exceeded 15 — qh 5th bit never exercised"
            );
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 112 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_q5_1_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q5_1_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q5_1_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
            // Single-row (m=1) dispatch must agree too.
            for r in 0..m {
                let mut one = [0f32; 1];
                vec_dot_q5_1_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                assert_eq!(
                    one[0].to_bits(),
                    scalar_out[r].to_bits(),
                    "q5_1_32 single in_f={in_f} row={r}"
                );
            }
        }
    }

    /// End-to-end tolerance-parity for the Q5_1 native-block int8 kernel: the int8-quantized-
    /// activation dot must track the FULL-PRECISION `d*q5 + m`-dequant · f32-activation reference.
    /// int8 activation quant is lossy, so this is a tolerance (not bit-identity). Covers the batch
    /// (m>1) and single-row (m=1) entries, several in_f, both nibble halves + the 5th `qh` bit.
    #[test]
    fn q5_1_32_matches_dequant_reference() {
        for in_f in [96usize, 256, 512] {
            let nb = in_f / 32;
            let w = build_q5_1(nb, 0.03, -0.02, 120);
            let wref = dequant_block(DType::Q5_1, &w).unwrap();
            // Both nibble halves decode to varying values (decode exercises lo AND hi).
            assert!(wref[..16].iter().any(|&v| v != wref[0]));
            assert!(wref[16..32].iter().any(|&v| v != wref[16]));
            // And the 5th bit must fire (codes > 15 → the qh-assembly path is verified vs dequant).
            assert!(
                q5_1_codes_over_15(&w, nb) > 0,
                "q5_1_32 ref in_f={in_f}: no code exceeded 15 — qh 5th bit never exercised"
            );
            // Full-precision activations (NOT the quantized ones) → proves end-to-end accuracy.
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 121 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // Batch entry (m>1). Tight 1e-3 vs the QUANTIZED activation the kernel actually sees
            // (isolates integer-dot correctness), plus a looser 2e-2 vs the full-precision
            // activation (absorbs the lossy int8 activation quant).
            let mut got = vec![0f32; xs.len()];
            vec_dot_q5_1_32_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "q5_1_32 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "q5_1_32 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row entry (m=1): one activation block-set, one-element out.
            for (r, x) in xs.iter().enumerate() {
                let mut one = [0f32; 1];
                vec_dot_q5_1_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                let want = dot(&wref, x);
                assert!(
                    rel_err(one[0], want) < 2e-2,
                    "q5_1_32 single in_f={in_f} row={r}: got {}, want {want}",
                    one[0]
                );
            }
        }
    }

    /// Build `nb` real 18-byte IQ4_NL blocks: fixed per-block `d`, random `qs` nibbles (both halves
    /// exercised, every code 0..15 reachable). Any byte pattern is a legal IQ4_NL block.
    fn build_iq4nl(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 18, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 18..b * 18 + 2], d);
        }
        w
    }

    /// SIMD (VNNI/AVX2, whichever this host has) must match the IQ4_NL scalar oracle bit-for-bit: the
    /// codebook signed-dot is an exact integer sum and the per-block f32 accumulation order is
    /// identical. `in_f=96` (3 blocks) exercises the VNNI pair-plus-odd-tail path.
    #[test]
    fn iq4nl_32_simd_bit_identical_to_scalar() {
        for in_f in [96usize, 256, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let w = build_iq4nl(nb, 0.02, 112);
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 113 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_iq4nl_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_iq4nl_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "iq4nl_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
            // Single-row (m=1) dispatch must agree too.
            for r in 0..m {
                let mut one = [0f32; 1];
                vec_dot_iq4nl_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                assert_eq!(
                    one[0].to_bits(),
                    scalar_out[r].to_bits(),
                    "iq4nl_32 single in_f={in_f} row={r}"
                );
            }
        }
    }

    /// End-to-end tolerance-parity for the IQ4_NL native-block int8 kernel: the int8-quantized-
    /// activation codebook dot must track the FULL-PRECISION `d·KV[code]`-dequant · f32-activation
    /// reference. int8 activation quant is lossy, so this is a tolerance (not bit-identity). Covers
    /// the batch (m>1) and single-row (m=1) entries, several in_f, both nibble halves nonzero.
    #[test]
    fn iq4nl_32_matches_dequant_reference() {
        for in_f in [32usize, 256, 512] {
            let nb = in_f / 32;
            let w = build_iq4nl(nb, 0.03, 120);
            let wref = dequant_block(DType::Iq4Nl, &w).unwrap();
            // Assert both nibble halves decode to varying values (decode exercises lo AND hi).
            assert!(wref[..16].iter().any(|&v| v != wref[0]));
            assert!(wref[16..32].iter().any(|&v| v != wref[16]));
            // Full-precision activations (NOT the quantized ones) → proves end-to-end accuracy.
            // Seed base 140: well-conditioned (worst full-precision rel-err ~1e-2 across these in_f;
            // IQ4_NL's wide asymmetric codebook makes int8-activation quant error data-dependent).
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 140 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // Batch entry (m>1). Tight 1e-3 vs the QUANTIZED activation the kernel actually sees
            // (isolates codebook-dot correctness), plus a looser 2e-2 vs the full-precision
            // activation (absorbs the lossy int8 activation quant).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq4nl_32_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq4nl_32 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "iq4nl_32 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row entry (m=1): one activation block-set, one-element out.
            for (r, x) in xs.iter().enumerate() {
                let mut one = [0f32; 1];
                vec_dot_iq4nl_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                let want = dot(&wref, x);
                assert!(
                    rel_err(one[0], want) < 2e-2,
                    "iq4nl_32 single in_f={in_f} row={r}: got {}, want {want}",
                    one[0]
                );
            }
        }
    }

    /// Build `nb` real 17-byte MXFP4 blocks: fully random bytes (E8M0 scale byte `e` at offset 0,
    /// 16 packed-nibble `qs`). Any byte pattern is a legal MXFP4 block, so the full `e` range and
    /// every code 0..15 (both nibble halves) are exercised — used by the bit-identity oracle test.
    fn build_mxfp4(nb: usize, seed: u64) -> Vec<u8> {
        det_bytes(nb * 17, seed)
    }

    /// Build `nb` real 17-byte MXFP4 blocks with the E8M0 scale byte clamped to a narrow band
    /// (`127..130` → `d ∈ {0.5, 1, 2}`) so the codebook · f32-activation reference stays well inside
    /// f32 range (the full E8M0 range spans 2^-127..2^127, which overflows/underflows the reference
    /// dot) AND the per-block `d` dynamic range stays small (4×). A wide `d` spread lets one
    /// high-scale block's absolute contribution dwarf the (mean-zero) total, amplifying that block's
    /// lossy int8-activation-quant error past the tolerance; keeping the band narrow keeps the
    /// end-to-end reference well-conditioned, exactly as IQ4_NL's fixed-`d` reference is. `qs` bytes
    /// stay fully random (both nibble halves, every code reachable); the full `e` range is exercised
    /// against the scalar oracle by the bit-identity test instead.
    fn build_mxfp4_band(nb: usize, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 17, seed);
        for b in 0..nb {
            w[b * 17] = 127 + (w[b * 17] % 3); // e ∈ {127,128,129} → d ∈ {0.5,1,2}
        }
        w
    }

    /// SIMD (VNNI/AVX2, whichever this host has) must match the MXFP4 scalar oracle bit-for-bit: the
    /// codebook signed-dot is an exact integer sum and the per-block f32 accumulation order is
    /// identical. Full random `e` + `qs`. `in_f=96` (3 blocks) exercises the VNNI pair-plus-tail.
    #[test]
    fn mxfp4_32_simd_bit_identical_to_scalar() {
        for in_f in [96usize, 256, 704] {
            let nb = in_f / 32;
            let m = 5usize;
            let w = build_mxfp4(nb, 212);
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 213 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_mxfp4_32_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_mxfp4_32_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "mxfp4_32 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
            // Single-row (m=1) dispatch must agree too.
            for r in 0..m {
                let mut one = [0f32; 1];
                vec_dot_mxfp4_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                assert_eq!(
                    one[0].to_bits(),
                    scalar_out[r].to_bits(),
                    "mxfp4_32 single in_f={in_f} row={r}"
                );
            }
        }
    }

    /// End-to-end tolerance-parity for the MXFP4 native-block int8 kernel: the int8-quantized-
    /// activation codebook dot must track the FULL-PRECISION `d·KVALUES_MXFP4[code]`-dequant ·
    /// f32-activation reference. int8 activation quant is lossy, so this is a tolerance (not
    /// bit-identity). `e` is kept in a moderate band (see [`build_mxfp4_band`]) so the reference dot
    /// stays in f32 range. Covers the batch (m>1) and single-row (m=1) entries, several in_f.
    #[test]
    fn mxfp4_32_matches_dequant_reference() {
        for in_f in [32usize, 256, 512] {
            let nb = in_f / 32;
            let w = build_mxfp4_band(nb, 282);
            let wref = dequant_block(DType::Mxfp4, &w).unwrap();
            // Assert both nibble halves decode to varying values (decode exercises lo AND hi).
            assert!(wref[..16].iter().any(|&v| v != wref[0]));
            assert!(wref[16..32].iter().any(|&v| v != wref[16]));
            // Full-precision activations (NOT the quantized ones) → proves end-to-end accuracy.
            // Seed bases (w=282, x=287) picked well-conditioned: the mean-zero signed dot with
            // narrow-band per-block `d` makes int8-activation rel-err data-dependent (worst ~2.4e-3
            // across these in_f), so a seed that avoids near-cancellation flakes keeps 2e-2 firm.
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 287 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // Batch entry (m>1). Tight 1e-3 vs the QUANTIZED activation the kernel actually sees
            // (isolates codebook-dot correctness), plus a looser 2e-2 vs the full-precision
            // activation (absorbs the lossy int8 activation quant).
            let mut got = vec![0f32; xs.len()];
            vec_dot_mxfp4_32_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "mxfp4_32 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "mxfp4_32 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row entry (m=1): one activation block-set, one-element out.
            for (r, x) in xs.iter().enumerate() {
                let mut one = [0f32; 1];
                vec_dot_mxfp4_32_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                let want = dot(&wref, x);
                assert!(
                    rel_err(one[0], want) < 2e-2,
                    "mxfp4_32 single in_f={in_f} row={r}: got {}, want {want}",
                    one[0]
                );
            }
        }
    }

    /// Build `nb` real 36-byte NVFP4 blocks: fully random bytes (4 UE4M3 sub-block scales + 32
    /// packed-nibble `qs`). Any byte pattern is a legal NVFP4 block (UE4M3 decodes every byte, incl.
    /// the 0/0x7F → 0 specials; codes are nibbles 0..15), so the full scale range and both nibble
    /// halves are exercised — used by the bit-identity oracle test.
    fn build_nvfp4(nb: usize, seed: u64) -> Vec<u8> {
        det_bytes(nb * 36, seed)
    }

    /// Build `nb` real 36-byte NVFP4 blocks with the four UE4M3 scale bytes clamped to a moderate
    /// band (`64..72` → exp 8, `d ∈ [1.0, 1.875]`) so the codebook · f32-activation reference
    /// stays well inside f32 range and the per-sub-block `d` dynamic range stays small (like the
    /// MXFP4 band builder). `qs` bytes stay fully random (both nibble halves, every code reachable);
    /// the full UE4M3 scale range is exercised against the scalar oracle by the bit-identity test.
    fn build_nvfp4_band(nb: usize, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 36, seed);
        for b in 0..nb {
            for s in 0..4usize {
                // exp 8 (bytes 64..71); mantissa spans the low 3 bits → d ∈ [1.0, 1.875]. Keeping the
                // per-sub-block scale spread tight (1.875×) bounds the int8-activation-quant error a
                // wide spread would amplify through the mean-zero (cancelling) codebook dot.
                w[b * 36 + s] = 64 + (w[b * 36 + s] % 8);
            }
        }
        w
    }

    /// SIMD (VNNI/AVX2, whichever this host has) must match the NVFP4 scalar oracle bit-for-bit: the
    /// codebook signed-dot is an exact integer sum and the per-sub-block f32 accumulation order is
    /// identical across all three tiers. Full random scales + `qs`. `in_f ∈ {64,256,704}` (multiples
    /// of 64) cover several block counts; single (m=1) and batch (m>1) both checked.
    #[test]
    fn nvfp4_simd_bit_identical_to_scalar() {
        for in_f in [64usize, 256, 704] {
            let nb = in_f / 64;
            let m = 5usize;
            let w = build_nvfp4(nb, 512);
            let q8s: Vec<Q8x32> = (0..m)
                .map(|r| quantize_q8_32(&det_x(in_f, 513 + r as u64)))
                .collect();
            let mut simd_out = vec![0f32; m];
            vec_dot_nvfp4_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_nvfp4_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "nvfp4 in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
            // Single-row (m=1) dispatch must agree too.
            for r in 0..m {
                let mut one = [0f32; 1];
                vec_dot_nvfp4_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                assert_eq!(
                    one[0].to_bits(),
                    scalar_out[r].to_bits(),
                    "nvfp4 single in_f={in_f} row={r}"
                );
            }
        }
    }

    /// End-to-end tolerance-parity for the NVFP4 native-block int8 kernel: the int8-quantized-
    /// activation codebook dot must track the FULL-PRECISION `dequant_block(Nvfp4)` · f32-activation
    /// reference. int8 activation quant is lossy, so this is a tolerance (not bit-identity). Scales
    /// are kept in a moderate band (see [`build_nvfp4_band`]) so the reference dot is well-
    /// conditioned. Covers the batch (m>1) and single-row (m=1) entries, several in_f.
    #[test]
    fn nvfp4_matches_dequant_reference() {
        for in_f in [64usize, 256, 704] {
            let nb = in_f / 64;
            let w = build_nvfp4_band(nb, 582);
            let wref = dequant_block(DType::Nvfp4, &w).unwrap();
            // Assert the four sub-blocks and both nibble halves decode to varying values.
            assert!(wref[..8].iter().any(|&v| v != wref[0]));
            assert!(wref[8..16].iter().any(|&v| v != wref[8]));
            assert!(wref[16..32].iter().any(|&v| v != wref[16]));
            // Seed bases (w=582, x=111) picked well-conditioned: the mean-zero signed codebook dot
            // makes int8-activation rel-err data-dependent (worst ~8.4e-3 across these in_f), so a
            // seed that avoids near-cancellation flakes keeps the 2e-2 bound firm.
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 111 + i)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // Batch entry (m>1). Tight 1e-3 vs the QUANTIZED activation the kernel actually sees
            // (isolates codebook-dot correctness), plus a looser 2e-2 vs the full-precision
            // activation (absorbs the lossy int8 activation quant).
            let mut got = vec![0f32; xs.len()];
            vec_dot_nvfp4_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "nvfp4 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "nvfp4 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row entry (m=1): one activation block-set, one-element out.
            for (r, x) in xs.iter().enumerate() {
                let mut one = [0f32; 1];
                vec_dot_nvfp4_batch(&w, std::slice::from_ref(&q8s[r]), in_f, &mut one);
                let want = dot(&wref, x);
                assert!(
                    rel_err(one[0], want) < 2e-2,
                    "nvfp4 single in_f={in_f} row={r}: got {}, want {want}",
                    one[0]
                );
            }
        }
    }

    /// Build `nb` real 136-byte IQ4_XS super-blocks with a fixed `d` and otherwise random (but
    /// structurally valid) `scales_h`/`scales_l`/`qs` bytes — any byte pattern is a legal IQ4_XS
    /// block (6-bit `ls` from arbitrary bits, 4-bit codes are nibbles 0–15). The full `ls` range
    /// (0..63) and every code is exercised — used by the bit-identity oracle test.
    fn build_iq4xs_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 136, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 136..b * 136 + 2], d);
        }
        w
    }

    /// Build well-conditioned IQ4_XS blocks: a small mixed-sign `ls` band around 32 (uniform weight
    /// magnitude, no pathological dynamic range → little catastrophic cancellation) plus random
    /// codes. Weights land at Q4_0 scale, so the int8-activation dot tracks the full-precision
    /// reference within a tight rel tolerance (the fully-random builder makes weights up to ±120
    /// with heavy cancellation, where int8 activation-quant noise legitimately dominates).
    fn build_iq4xs_wc(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = vec![0u8; nb * 136];
        for b in 0..nb {
            let blk = &mut w[b * 136..b * 136 + 136];
            put_f16(&mut blk[0..2], d);
            let mut scales_h = 0u16;
            for ib in 0..8usize {
                let ls = 28 + (lcg(&mut seed) >> 40) % 9; // ls ∈ 28..36 → ls−32 ∈ [−4,4]
                let lo = (ls & 0xF) as u8;
                let hi = ((ls >> 4) & 3) as u8;
                blk[4 + ib / 2] |= lo << (4 * (ib % 2));
                scales_h |= (hi as u16) << (2 * ib);
            }
            blk[2..4].copy_from_slice(&scales_h.to_le_bytes());
            for j in 0..128usize {
                blk[8 + j] = (lcg(&mut seed) >> 33) as u8;
            }
        }
        w
    }

    /// The SIMD dispatch (whatever tier this CPU has) must be BIT-IDENTICAL to the scalar oracle for
    /// both the single-token and batch IQ4_XS kernels: the per-sub-block dot is an integer sum (no
    /// rounding, order-free) and the `d·d8·isum` epilogue is shared, so every tier collapses to the
    /// same bits. Falls through to the same scalar fn on non-x86, where the assertion is trivial.
    #[test]
    fn iq4xs_simd_bit_identical_to_scalar() {
        for in_f in [256usize, 512, 768] {
            let nb = in_f / 256;
            let w = build_iq4xs_rand(nb, 0.05, 40);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 41 + r as u64)))
                .collect();
            // Single-token: SIMD dispatch vs scalar oracle.
            for (r, q8) in q8s.iter().enumerate() {
                let simd = vec_dot_iq4xs(&w, q8, in_f);
                let scalar = vec_dot_iq4xs_scalar(&w, q8, in_f);
                assert_eq!(
                    simd.to_bits(),
                    scalar.to_bits(),
                    "iq4xs single in_f={in_f} row={r}: simd {simd}, scalar {scalar}"
                );
            }
            // Batch: SIMD dispatch vs scalar oracle.
            let mut simd_out = vec![0f32; m];
            vec_dot_iq4xs_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_iq4xs_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "iq4xs batch in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ4_XS kernel must track the FULL-PRECISION
    /// codebook dequant (`d·(ls−32)·kv[code]`) · f32-activation reference. IQ4_XS weights are stored
    /// exactly (no weight-quant loss), so vs the QUANTIZED activation the kernel sees, the only slack
    /// is f32 accumulation order → tight 1e-3 (isolates integer-dot correctness, as the Q4_0 sibling
    /// does). vs the full-precision activation, the looser 2e-2 absorbs the lossy int8 activation
    /// quant. Covers single-row (m=1) and batch (m>1).
    #[test]
    fn iq4xs_matches_dequant_reference() {
        for in_f in [256usize, 512, 768] {
            let nb = in_f / 256;
            let w = build_iq4xs_wc(nb, 0.0006, 50);
            let wref = dequant_codebook(DType::Iq4Xs, &w);
            // Sanity: reference actually spans both nibble halves / nonzero scales.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 51 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq4xs_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq4xs batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "iq4xs batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq4xs(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq4xs single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 2e-2,
                    "iq4xs single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` fully-random 82-byte IQ2_S super-blocks with a fixed `d`. Every byte pattern is a
    /// legal IQ2_S block: the grid index is 10-bit into the 1024-entry `IQ2S_GRID`, and the sign /
    /// scale bytes are arbitrary — so random bytes exercise the full grid, all `KMASK_IQ2XS` sign
    /// combinations, both 4-bit scale nibbles, and every 2-bit `qh` high-bit pattern.
    fn build_iq2s_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 82, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 82..b * 82 + 2], d);
        }
        w
    }

    /// Build well-conditioned IQ2_S blocks: scale nibbles pinned to a narrow band (uniform `dl` →
    /// little f32 cancellation) with random grid indices / signs / high bits, so the int8-activation
    /// dot tracks the full-precision grid dequant within a tight rel tolerance. (Fully-random blocks
    /// can drive `dl` across an 8× range with heavy per-group cancellation, where the activation
    /// int8-quant noise legitimately dominates — same reasoning as `build_iq4xs_wc`.)
    fn build_iq2s_wc(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = vec![0u8; nb * 82];
        for b in 0..nb {
            let blk = &mut w[b * 82..b * 82 + 82];
            put_f16(&mut blk[0..2], d);
            // qs [2..66]: grid-idx low bytes [0..32] + sign bytes [32..64], fully random.
            // qh [66..74]: 2-bit high index bits, random for full 10-bit grid coverage.
            for j in 0..72usize {
                blk[2 + j] = (lcg(&mut seed) >> 33) as u8;
            }
            // scales [74..82]: both nibbles in a narrow band {6..9} → dl within a ~1.46× spread.
            for j in 0..8usize {
                let lo = 6 + (lcg(&mut seed) >> 40) % 4;
                let hi = 6 + (lcg(&mut seed) >> 40) % 4;
                blk[74 + j] = (lo | (hi << 4)) as u8;
            }
        }
        w
    }

    /// Batch and single-token IQ2_S kernels share the SAME scalar grid expansion and f32
    /// accumulation order, so they must agree to the bit for every row. Fully-random blocks exercise
    /// the whole grid / all sign masks. (No SIMD variant exists — the grid gather is inherently
    /// scalar — so this bit-identity check stands in for the SIMD-vs-scalar oracle IQ4_XS has.)
    #[test]
    fn iq2s_batch_matches_single() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq2s_rand(nb, 0.05, 80);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 81 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_iq2s_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_iq2s(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "iq2s batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ2_S kernel must track the FULL-PRECISION
    /// grid dequant (`dl·signed_grid`) · f32-activation reference. IQ2_S weights are stored exactly
    /// (the grid value IS the weight, no weight-quant loss), so vs the QUANTIZED activation the
    /// kernel actually sees, the only slack is f32 accumulation order → tight 1e-3 (isolates the
    /// grid+sign+scale decode AND the integer dot — this is the primary correctness proof, no model
    /// available). vs the full-precision activation, the looser 3e-2 absorbs the lossy int8
    /// activation quant, whose relative error the signed-grid dot's sign cancellation amplifies
    /// beyond IQ4_XS's 2e-2. Covers single-row (m=1) and batch (m>1).
    #[test]
    fn iq2s_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq2s_wc(nb, 0.03, 70);
            let wref = dequant_codebook(DType::Iq2S, &w);
            // Sanity: reference spans nonzero grid values / both scale nibbles.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq2s_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq2s batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 3e-2,
                    "iq2s batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq2s(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq2s single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 3e-2,
                    "iq2s single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` fully-random 66-byte IQ2_XXS super-blocks with a fixed `d`. Every byte pattern is a
    /// legal IQ2_XXS block: each 8-bit grid index selects into the 256-entry `IQ2XXS_GRID`, each 7-bit
    /// sign index into the 128-entry `KSIGNS_IQ2XS`, and the 4-bit scale magnitude (`aux1>>28`) is
    /// arbitrary — so random bytes exercise the full grid, all sign masks, and every scale nibble.
    fn build_iq2xxs_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 66, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 66..b * 66 + 2], d);
        }
        w
    }

    /// Build well-conditioned IQ2_XXS blocks: the per-32 scale magnitude (`aux1>>28`, the high nibble
    /// of byte `qs[ib32*8+7]`) pinned to a narrow band {6..9} → `db = d*(0.5+mag)*0.25` within a
    /// ~1.46× spread, with random grid indices / sign indices, so the int8-activation dot tracks the
    /// full-precision grid dequant within a tight rel tolerance. (Fully-random blocks drive `db`
    /// across an 8× range with heavy per-group cancellation, where the activation int8-quant noise
    /// legitimately dominates — same reasoning as `build_iq2s_wc`.)
    fn build_iq2xxs_wc(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = vec![0u8; nb * 66];
        for b in 0..nb {
            let blk = &mut w[b * 66..b * 66 + 66];
            put_f16(&mut blk[0..2], d);
            // qs[2..66]: 8 sub-blocks × 8 bytes (aux0[4] grid idx + aux1[4] sign+scale), random.
            for j in 0..64usize {
                blk[2 + j] = (lcg(&mut seed) >> 33) as u8;
            }
            // Pin scale magnitude of each sub-block: aux1>>28 = high nibble of byte qs[ib32*8+7].
            for ib32 in 0..8usize {
                let mag = 6 + (lcg(&mut seed) >> 40) % 4; // {6..9}
                let byte = 2 + ib32 * 8 + 7;
                blk[byte] = (blk[byte] & 0x0f) | ((mag as u8) << 4);
            }
        }
        w
    }

    /// Batch and single-token IQ2_XXS kernels share the SAME scalar grid expansion and f32
    /// accumulation order, so they must agree to the bit for every row. Fully-random blocks exercise
    /// the whole grid / all sign masks. (No SIMD variant exists — the grid gather is inherently
    /// scalar — so this bit-identity check stands in for the SIMD-vs-scalar oracle IQ4_XS has.)
    #[test]
    fn iq2xxs_batch_matches_single() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq2xxs_rand(nb, 0.05, 80);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 81 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_iq2xxs_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_iq2xxs(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "iq2xxs batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ2_XXS kernel must track the FULL-PRECISION
    /// grid dequant (`db·signed_grid`) · f32-activation reference. IQ2_XXS weights are stored exactly
    /// (the grid value IS the weight, no weight-quant loss), so vs the QUANTIZED activation the kernel
    /// actually sees, the only slack is f32 accumulation order → tight 1e-3 (isolates the grid+sign+
    /// scale decode AND the integer dot — this is the primary correctness proof, no model available).
    /// vs the full-precision activation, the looser 3e-2 absorbs the lossy int8 activation quant,
    /// whose relative error the signed-grid dot's sign cancellation amplifies beyond IQ4_XS's 2e-2
    /// (same bound IQ2_S uses). Covers single-row (m=1) and batch (m>1).
    #[test]
    fn iq2xxs_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            // wseed=60 is well-conditioned for the loose full-precision bound: IQ2_XXS packs all 8
            // group elements from ONE grid entry, so per-group sign cancellation is more correlated
            // than IQ2_S and a few weight seeds flake the 3e-2 vs-full-precision check even though the
            // tight quant-ref (the real proof) always holds. 60 keeps worst full-ref rel_err ~9e-3.
            let w = build_iq2xxs_wc(nb, 0.03, 60);
            let wref = dequant_codebook(DType::Iq2Xxs, &w);
            // Sanity: reference spans nonzero grid values / both scale nibbles.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq2xxs_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq2xxs batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 3e-2,
                    "iq2xxs batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq2xxs(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq2xxs single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 3e-2,
                    "iq2xxs single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` random 66-byte TQ2_0 super-blocks with a fixed `d`. TQ2_0 has NO structural
    /// constraints on `qs`: every one of the 64 bytes packs four arbitrary 2-bit codes `q ∈ 0..3`,
    /// so fully-random bytes exercise all four code values (→ all `(q − 1) ∈ {−1,0,1,2}` weights) in
    /// every block. Only the trailing f16 `d` is pinned.
    fn build_tq2_0_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 66, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 66 + 64..b * 66 + 66], d);
        }
        w
    }

    /// Batch and single-token TQ2_0 kernels share the SAME i8 expansion and f32 accumulation order,
    /// so they must agree to the bit for every row. Fully-random blocks exercise all four 2-bit codes.
    #[test]
    fn tq2_0_batch_matches_single() {
        for in_f in [256usize, 512] {
            let w = build_tq2_0_rand(in_f / 256, 0.05, 90);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_tq2_0_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_tq2_0(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "tq2_0 batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation TQ2_0 kernel must track the full-precision
    /// dequant (`(q − 1)·d`) · f32-activation reference. TQ2_0 weights are stored EXACTLY (the ternary
    /// code IS the weight, no weight-quant loss beyond the shared f16 `d`), so vs the QUANTIZED
    /// activation the kernel actually sees, the only slack is f32 accumulation order → tight 1e-3
    /// (this is the primary correctness proof: it isolates the packing / chunk-l-m element order and
    /// the integer dot, no model available). vs the full-precision activation the looser 2e-2 absorbs
    /// the lossy int8 activation quant; ternary weights have far less sign cancellation than the grid
    /// quants, so the bound holds comfortably (seed 70 keeps worst full-ref rel_err ~5e-3). Covers
    /// single-row (m=1) and batch (m>1).
    #[test]
    fn tq2_0_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let w = build_tq2_0_rand(in_f / 256, 0.05, 70);
            let wref = dequant_block(DType::Tq2_0, &w).unwrap();
            // Sanity: reference spans all four dequant weight values → all 2-bit codes exercised.
            // Targets are multiples of the f16-rounded `d` the dequant actually applies.
            let dr = half::f16::from_f32(0.05).to_f32();
            for target in [-dr, 0.0, dr, 2.0 * dr] {
                assert!(
                    wref.iter().any(|&v| (v - target).abs() < 1e-6),
                    "tq2_0 ref missing weight {target} in_f={in_f}"
                );
            }
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_tq2_0_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "tq2_0 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "tq2_0 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_tq2_0(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "tq2_0 single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 2e-2,
                    "tq2_0 single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` random 54-byte TQ1_0 super-blocks with a fixed `d`. TQ1_0's base-3 packing places
    /// NO constraint on `qs`/`qh`: EVERY byte value 0..255 is a legal base-3 packing (the digit
    /// extraction `((b·POW3[n])·3 >> 8)` maps any byte to five ternary digits ∈ 0..2), so fully-random
    /// bytes exercise all three dequant weight values (`(digit − 1) ∈ {−1,0,1}`) across the block.
    /// Only the trailing f16 `d` is pinned.
    fn build_tq1_0_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 54, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 54 + 52..b * 54 + 54], d);
        }
        w
    }

    /// Batch and single-token TQ1_0 kernels share the SAME i8 expansion and f32 accumulation order,
    /// so they must agree to the bit for every row. Fully-random blocks exercise all three ternary
    /// digits across the base-3 packing and all three segments (qs[0..32], qs[32..48], qh[0..4]).
    #[test]
    fn tq1_0_batch_matches_single() {
        for in_f in [256usize, 512] {
            let w = build_tq1_0_rand(in_f / 256, 0.05, 90);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_tq1_0_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_tq1_0(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "tq1_0 batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation TQ1_0 kernel must track the full-precision
    /// dequant (`(digit − 1)·d`) · f32-activation reference. TQ1_0 ternary weights are stored EXACTLY
    /// (the ternary digit IS the weight, no weight-quant loss beyond the shared f16 `d`), so vs the
    /// QUANTIZED activation the kernel actually sees, the only slack is f32 accumulation order → tight
    /// 1e-3 (this is the primary correctness proof: it isolates the base-3 digit extraction, the
    /// 3-segment 160+80+16 element order, and the integer dot, no model available). vs the
    /// full-precision activation the looser 2e-2 absorbs the lossy int8 activation quant; ternary
    /// weights have far less sign cancellation than the grid quants, so the bound holds comfortably.
    /// Covers single-row (m=1) and batch (m>1).
    #[test]
    fn tq1_0_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let w = build_tq1_0_rand(in_f / 256, 0.05, 70);
            let wref = dequant_block(DType::Tq1_0, &w).unwrap();
            // Sanity: reference spans all three dequant weight values → all ternary digits exercised.
            // Targets are multiples of the f16-rounded `d` the dequant actually applies.
            let dr = half::f16::from_f32(0.05).to_f32();
            for target in [-dr, 0.0, dr] {
                assert!(
                    wref.iter().any(|&v| (v - target).abs() < 1e-6),
                    "tq1_0 ref missing weight {target} in_f={in_f}"
                );
            }
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_tq1_0_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "tq1_0 batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "tq1_0 batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_tq1_0(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "tq1_0 single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 2e-2,
                    "tq1_0 single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` random 50-byte IQ1_S super-blocks with a fixed `d`. Every generated block is a
    /// legal IQ1_S block: `qs[32]` are the low 8 bits of each 11-bit grid index and the `qh[8]`
    /// `u16`s supply the high 3 bits per group (`(qh>>3l)&7`) → any 11-bit index ∈ 0..2047 hits the
    /// 2048-entry `IQ1S_GRID`. The per-sub-block scale bits `(qh>>12)&7` are drawn across their full
    /// 0..7 range and the `0x8000` delta-sign bit across {0,1}, so both `dl` magnitudes and both
    /// `delta` signs are exercised (per the TDD requirement).
    fn build_iq1s_rand(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = vec![0u8; nb * 50];
        for b in 0..nb {
            let blk = &mut w[b * 50..b * 50 + 50];
            put_f16(&mut blk[0..2], d);
            // qs[32]: random grid low bytes.
            for j in 0..32usize {
                blk[2 + j] = (lcg(&mut seed) >> 33) as u8;
            }
            // qh[16] = 8 × u16 LE. Low 12 bits carry the per-group grid high bits (3 bits each for
            // l=0..3, i.e. bits 0..11); bits 12..14 are the scale, bit 15 the delta sign.
            for ib in 0..8usize {
                let low = (lcg(&mut seed) >> 20) as u16 & 0x0fff;
                let scale = (lcg(&mut seed) % 8) as u16; // (qh>>12)&7 across 0..7 → both dl mags
                let sign = ((lcg(&mut seed) >> 40) & 1) as u16; // 0x8000 across both delta signs
                let qh = low | (scale << 12) | (sign << 15);
                blk[34 + 2 * ib..34 + 2 * ib + 2].copy_from_slice(&qh.to_le_bytes());
            }
        }
        w
    }

    /// Batch and single-token IQ1_S kernels share the SAME scalar grid expansion and f32
    /// accumulation order (incl. the `iprod`/`asum` delta split), so they must agree to the bit for
    /// every row. Random valid blocks exercise the whole grid, all scale bits, and both delta signs.
    #[test]
    fn iq1s_batch_matches_single() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq1s_rand(nb, 0.05, 90);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_iq1s_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_iq1s(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "iq1s batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ1_S kernel must track the FULL-PRECISION
    /// grid+delta dequant (`dl·(grid+delta)`) · f32-activation reference from `dequant_block`. IQ1_S
    /// weights are stored exactly (grid value + fractional delta, no weight-quant loss), so vs the
    /// QUANTIZED activation the kernel actually sees, the only slack is f32 accumulation order → tight
    /// 1e-3. This isolates the grid+delta+scale decode, the integer `iprod` dot, AND the `delta·asum`
    /// split (the wrinkle vs IQ2_XXS) as the primary correctness proof (no model available). vs the
    /// full-precision activation, the looser 3e-2 absorbs the lossy int8 activation quant. Covers
    /// single-row (m=1) and batch (m>1). Seed 267 is well-conditioned for the loose full-ref bound
    /// (worst full-ref rel_err ~2.4e-3; fully-random weight seeds routinely drive it past 3e-2 as the
    /// small {-1,0,1} grid values + ±0.125 delta make per-group sign cancellation dominate the int8
    /// activation-quant noise — the tight quant-ref bound below is the real proof and holds for any).
    #[test]
    fn iq1s_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq1s_rand(nb, 0.05, 267);
            let wref = dequant_block(DType::Iq1S, &w).unwrap();
            // Sanity: reference spans nonzero grid values.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq1s_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq1s batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 3e-2,
                    "iq1s batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq1s(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq1s single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 3e-2,
                    "iq1s single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` random 56-byte IQ1_M super-blocks with a fixed `d`. Every generated block is a
    /// legal IQ1_M block: `qs[32]` supply the low 8 bits of each grid index and each `qh` byte's low
    /// bits supply the 3 high bits (`(qh<<8|<<4)&0x700`) per group → any 11-bit index ∈ 0..2047 hits
    /// the 2048-entry `IQ1S_GRID`. The four `u16` scale words carry random 12-bit low halves (so the
    /// per-16 `dl1`/`dl2` nibbles `(sc >> {0,3,6,9}) & 7` span 0..7) with `d`'s f16 bits packed into
    /// their high nibbles (the `dequant` d-extraction round-trips). The `qh` `0x08`/`0x80` delta
    /// bits are drawn fully random so both per-8 delta signs are exercised (per the TDD requirement).
    fn build_iq1m_rand(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let d_bits = half::f16::from_f32(d).to_bits();
        let mut w = vec![0u8; nb * 56];
        for b in 0..nb {
            let blk = &mut w[b * 56..b * 56 + 56];
            // qs[32]: random grid low bytes.
            for j in 0..32usize {
                blk[j] = (lcg(&mut seed) >> 33) as u8;
            }
            // qh[16]: random — low 3 bits = grid high, bit3/bit7 = delta signs, bits4-6 = grid high.
            for j in 0..16usize {
                blk[32 + j] = (lcg(&mut seed) >> 33) as u8;
            }
            // scales[8] = 4 × u16 LE. Low 12 bits carry the per-16 scale nibbles; the high nibble of
            // word i carries nibble i of d_bits so `dequant`'s d-extraction reconstructs `d`.
            for i in 0..4usize {
                let low = (lcg(&mut seed) >> 20) as u16 & 0x0fff;
                let d_nib = (d_bits >> (4 * i)) & 0xf;
                let sc = low | (d_nib << 12);
                blk[48 + 2 * i..48 + 2 * i + 2].copy_from_slice(&sc.to_le_bytes());
            }
        }
        w
    }

    /// Batch and single-token IQ1_M kernels share the SAME scalar grid expansion and f32
    /// accumulation order (incl. the per-16 `dl` and per-8 `delta·asum` split), so they must agree to
    /// the bit for every row. Random valid blocks exercise the whole grid, all scale nibbles, and
    /// both per-8 delta signs.
    #[test]
    fn iq1m_batch_matches_single() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq1m_rand(nb, 0.05, 90);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_iq1m_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_iq1m(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "iq1m batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ1_M kernel must track the FULL-PRECISION
    /// grid+delta dequant (`dl·(grid+delta)`) · f32-activation reference from `dequant_block`. IQ1_M
    /// weights are stored exactly (grid value + fractional delta, no weight-quant loss), so vs the
    /// QUANTIZED activation the kernel actually sees, the only slack is f32 accumulation order → tight
    /// 1e-3. This isolates the d-extraction (bit-packed f16), the per-16 `dl1`/`dl2` selection, the
    /// per-8 delta decode, the grid decode, the integer `iprod` dot, AND the `delta·asum` split as the
    /// primary correctness proof (no model available). vs the full-precision activation, the looser
    /// 3e-2 absorbs the lossy int8 activation quant. Covers single-row (m=1) and batch (m>1). Seed 5
    /// is well-conditioned for the loose full-ref bound (worst ~3.7e-3; the {-1,0,1} grid + ±0.125 make
    /// per-group sign cancellation dominate int8 activation-quant noise, so fully-random seeds can
    /// drive the full-ref rel_err past 3e-2 — the tight quant-ref bound is the real proof; holds any).
    #[test]
    fn iq1m_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq1m_rand(nb, 0.05, 5);
            let wref = dequant_block(DType::Iq1M, &w).unwrap();
            // Sanity: reference spans nonzero grid values.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq1m_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq1m batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 3e-2,
                    "iq1m batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq1m(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq1m single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 3e-2,
                    "iq1m single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` fully-random 74-byte IQ2_XS super-blocks with a fixed `d`. Every byte pattern is a
    /// legal IQ2_XS block: each `u16` in `qs` splits into a 9-bit grid index (`&511`) into the
    /// 512-entry `IQ2XS_GRID` and a 7-bit sign index (`>>9`) into the 128-entry `KSIGNS_IQ2XS`, and
    /// the scale bytes are arbitrary — so random bytes exercise the full grid, all sign masks, and
    /// both 4-bit scale nibbles.
    fn build_iq2xs_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 74, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 74..b * 74 + 2], d);
        }
        w
    }

    /// Build well-conditioned IQ2_XS blocks: scale nibbles pinned to a narrow band (uniform `dl` →
    /// little f32 cancellation) with random grid / sign indices, so the int8-activation dot tracks
    /// the full-precision grid dequant within a tight rel tolerance. (Fully-random blocks can drive
    /// `dl` across an 8× range with heavy per-group cancellation, where the activation int8-quant
    /// noise legitimately dominates — same reasoning as `build_iq2s_wc`.)
    fn build_iq2xs_wc(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = vec![0u8; nb * 74];
        for b in 0..nb {
            let blk = &mut w[b * 74..b * 74 + 74];
            put_f16(&mut blk[0..2], d);
            // qs [2..66]: 32 u16 = 9-bit grid idx + 7-bit sign idx, fully random.
            for j in 0..64usize {
                blk[2 + j] = (lcg(&mut seed) >> 33) as u8;
            }
            // scales [66..74]: both nibbles in a narrow band {6..9} → dl within a ~1.46× spread.
            for j in 0..8usize {
                let lo = 6 + (lcg(&mut seed) >> 40) % 4;
                let hi = 6 + (lcg(&mut seed) >> 40) % 4;
                blk[66 + j] = (lo | (hi << 4)) as u8;
            }
        }
        w
    }

    /// Batch and single-token IQ2_XS kernels share the SAME scalar grid expansion and f32
    /// accumulation order, so they must agree to the bit for every row. Fully-random blocks exercise
    /// the whole grid / all sign masks. (No SIMD variant exists — the grid gather is inherently
    /// scalar — so this bit-identity check stands in for the SIMD-vs-scalar oracle IQ4_XS has.)
    #[test]
    fn iq2xs_batch_matches_single() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq2xs_rand(nb, 0.05, 80);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 81 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_iq2xs_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_iq2xs(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "iq2xs batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ2_XS kernel must track the FULL-PRECISION
    /// grid dequant (`dl·signed_grid`) · f32-activation reference. IQ2_XS weights are stored exactly
    /// (the grid value IS the weight, no weight-quant loss), so vs the QUANTIZED activation the kernel
    /// actually sees, the only slack is f32 accumulation order → tight 1e-3 (isolates the grid+sign+
    /// scale decode AND the integer dot — this is the primary correctness proof, no model available).
    /// vs the full-precision activation, the looser 3e-2 absorbs the lossy int8 activation quant,
    /// whose relative error the signed-grid dot's sign cancellation amplifies beyond IQ4_XS's 2e-2
    /// (same bound IQ2_S/IQ2_XXS use). Covers single-row (m=1) and batch (m>1).
    #[test]
    fn iq2xs_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            // wseed=60 is well-conditioned for the loose full-precision bound (worst full-ref
            // rel_err ~4.4e-3): a few weight seeds drive heavier per-group sign cancellation and flake
            // the 3e-2 vs-full-precision check even though the tight quant-ref (the real proof) always
            // holds regardless of seed.
            let w = build_iq2xs_wc(nb, 0.03, 60);
            let wref = dequant_codebook(DType::Iq2Xs, &w);
            // Sanity: reference spans nonzero grid values / both scale nibbles.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq2xs_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq2xs batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 3e-2,
                    "iq2xs batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq2xs(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq2xs single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 3e-2,
                    "iq2xs single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` fully-random 110-byte IQ3_S super-blocks with a fixed `d`. Every byte pattern is a
    /// legal IQ3_S block: each grid index is 9-bit (8 low bits from `qs`, 1 high bit from `qh`) into
    /// the 512-entry `IQ3S_GRID`, and the sign / scale bytes are arbitrary — so random bytes exercise
    /// the full grid, all `KMASK_IQ2XS` sign combinations, both 4-bit scale nibbles, and every `qh`
    /// high-bit pattern.
    fn build_iq3s_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 110, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 110..b * 110 + 2], d);
        }
        w
    }

    /// Build well-conditioned IQ3_S blocks: scale nibbles pinned to a narrow band {5..8} (`db =
    /// d*(1+2*s)` → dl within a ~1.55× spread) with random grid indices / signs / high bits, so the
    /// int8-activation dot tracks the full-precision grid dequant within a tight rel tolerance.
    /// (Fully-random blocks drive `dl` across a wide range with heavy per-group cancellation, where
    /// the activation int8-quant noise legitimately dominates — same reasoning as `build_iq2s_wc`.)
    fn build_iq3s_wc(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = vec![0u8; nb * 110];
        for b in 0..nb {
            let blk = &mut w[b * 110..b * 110 + 110];
            put_f16(&mut blk[0..2], d);
            // qs[2..66] + qh[66..74] + signs[74..106] = 104 random bytes (full grid / all signs).
            for j in 0..104usize {
                blk[2 + j] = (lcg(&mut seed) >> 33) as u8;
            }
            // scales[106..110]: both nibbles in {5..8} → db within a ~1.55× spread.
            for j in 0..4usize {
                let lo = 5 + (lcg(&mut seed) >> 40) % 4;
                let hi = 5 + (lcg(&mut seed) >> 40) % 4;
                blk[106 + j] = (lo | (hi << 4)) as u8;
            }
        }
        w
    }

    /// Batch and single-token IQ3_S kernels share the SAME scalar grid expansion and f32 accumulation
    /// order, so they must agree to the bit for every row. Fully-random blocks exercise the whole
    /// grid / all sign masks. (No SIMD variant exists — the grid gather is inherently scalar — so this
    /// bit-identity check stands in for the SIMD-vs-scalar oracle.)
    #[test]
    fn iq3s_batch_matches_single() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq3s_rand(nb, 0.05, 90);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_iq3s_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_iq3s(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "iq3s batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ3_S kernel must track the FULL-PRECISION
    /// grid dequant (`db·signed_grid`) · f32-activation reference. IQ3_S weights are stored exactly
    /// (the grid value IS the weight, no weight-quant loss), so vs the QUANTIZED activation the kernel
    /// actually sees, the only slack is f32 accumulation order → tight 1e-3 (isolates the
    /// grid+sign+scale decode AND the integer dot — this is the primary correctness proof, no model
    /// available). vs the full-precision activation, the looser 3e-2 absorbs the lossy int8 activation
    /// quant, whose relative error the signed-grid dot's sign cancellation amplifies. Covers
    /// single-row (m=1) and batch (m>1).
    #[test]
    fn iq3s_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            // Seed 70 is well-conditioned (max full-ref rel err ~0.005 across all rows / in_f) — a
            // handful of seeds drive `dl` cancellation that legitimately inflates the int8-activation
            // error past 3e-2, same as IQ2_S; the tight quant-ref bound below holds for ALL seeds.
            let w = build_iq3s_wc(nb, 0.02, 70);
            let wref = dequant_codebook(DType::Iq3S, &w);
            // Sanity: reference spans nonzero grid values / both scale nibbles.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq3s_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq3s batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 3e-2,
                    "iq3s batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq3s(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq3s single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 3e-2,
                    "iq3s single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` fully-random 98-byte IQ3_XXS super-blocks with a fixed `d`. Every byte pattern is a
    /// legal IQ3_XXS block: each grid index is an 8-bit index into the 256-entry `IQ3XXS_GRID`, and
    /// the scales-and-signs `aux32` words are arbitrary — so random bytes exercise the full grid, all
    /// `KMASK_IQ2XS` sign combinations (via every `KSIGNS_IQ2XS` entry) and every top-nibble scale.
    fn build_iq3xxs_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 98, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 98..b * 98 + 2], d);
        }
        w
    }

    /// Build well-conditioned IQ3_XXS blocks: the per-32 scale nibble (`aux32 >> 28`) pinned to a
    /// narrow band {6..9} (`db = d*(0.5+s)*0.5` → dl within a ~1.36× spread) with random grid indices
    /// and random 7-bit sign fields, so the int8-activation dot tracks the full-precision grid dequant
    /// within a tight rel tolerance. (Fully-random blocks drive `dl` across a wide range with heavy
    /// per-group cancellation, where the activation int8-quant noise legitimately dominates — same
    /// reasoning as `build_iq3s_wc`.)
    fn build_iq3xxs_wc(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = vec![0u8; nb * 98];
        for b in 0..nb {
            let blk = &mut w[b * 98..b * 98 + 98];
            put_f16(&mut blk[0..2], d);
            // qs[2..66] = 64 random grid-index bytes (full grid coverage).
            for j in 0..64usize {
                blk[2 + j] = (lcg(&mut seed) >> 33) as u8;
            }
            // sas[66..98] = 8 × u32 (aux32). Randomize the 28-bit sign region (four 7-bit fields),
            // then pin the top nibble (scale) into {6..9} → db within a ~1.36× spread.
            for ib32 in 0..8usize {
                let signs = (lcg(&mut seed) as u32) & 0x0FFF_FFFF;
                let scale = 6 + ((lcg(&mut seed) >> 40) % 4) as u32;
                let aux32 = signs | (scale << 28);
                blk[66 + 4 * ib32..66 + 4 * ib32 + 4].copy_from_slice(&aux32.to_le_bytes());
            }
        }
        w
    }

    /// Batch and single-token IQ3_XXS kernels share the SAME scalar grid expansion and f32
    /// accumulation order, so they must agree to the bit for every row. Fully-random blocks exercise
    /// the whole grid / all sign masks. (No SIMD variant exists — the grid gather is inherently scalar
    /// — so this bit-identity check stands in for the SIMD-vs-scalar oracle.)
    #[test]
    fn iq3xxs_batch_matches_single() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            let w = build_iq3xxs_rand(nb, 0.05, 90);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 91 + r as u64)))
                .collect();
            let mut batch_out = vec![0f32; m];
            vec_dot_iq3xxs_batch(&w, &q8s, in_f, &mut batch_out);
            for (r, q8) in q8s.iter().enumerate() {
                let single = vec_dot_iq3xxs(&w, q8, in_f);
                assert_eq!(
                    batch_out[r].to_bits(),
                    single.to_bits(),
                    "iq3xxs batch vs single in_f={in_f} row={r}: batch {}, single {single}",
                    batch_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation IQ3_XXS kernel must track the FULL-PRECISION
    /// grid dequant (`db·signed_grid`) · f32-activation reference. IQ3_XXS weights are stored exactly
    /// (the grid value IS the weight, no weight-quant loss), so vs the QUANTIZED activation the kernel
    /// actually sees, the only slack is f32 accumulation order → tight 1e-3 (isolates the
    /// grid+sign+scale decode AND the integer dot — this is the primary correctness proof, no model
    /// available). vs the full-precision activation, the looser 3e-2 absorbs the lossy int8 activation
    /// quant, whose relative error the signed-grid dot's sign cancellation amplifies. Covers
    /// single-row (m=1) and batch (m>1).
    #[test]
    fn iq3xxs_matches_dequant_reference() {
        for in_f in [256usize, 512] {
            let nb = in_f / 256;
            // Seed 70 is well-conditioned (max full-ref rel err well under 3e-2 across all rows /
            // in_f) — a handful of seeds drive `dl` cancellation that legitimately inflates the
            // int8-activation error past 3e-2, same as IQ3_S; the tight quant-ref bound holds for ALL
            // seeds.
            let w = build_iq3xxs_wc(nb, 0.02, 70);
            let wref = dequant_codebook(DType::Iq3Xxs, &w);
            // Sanity: reference spans nonzero grid values / scale range.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 71 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_iq3xxs_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "iq3xxs batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 3e-2,
                    "iq3xxs batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_iq3xxs(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "iq3xxs single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 3e-2,
                    "iq3xxs single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` fully-random 84-byte Q2_K super-blocks with fixed `d`/`dmin` — every scale nibble
    /// (0–15), min nibble (0–15) and 2-bit code (0–3) is exercised. Any byte pattern is a legal
    /// Q2_K block. Used by the bit-identity oracle test.
    fn build_q2k_rand(nb: usize, d: f32, dmin: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 84, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 84 + 80..b * 84 + 82], d);
            put_f16(&mut w[b * 84 + 82..b * 84 + 84], dmin);
        }
        w
    }

    /// Build well-conditioned Q2_K blocks: ZERO-CENTERED weights, the same trick that makes the
    /// Q4_0 sibling's `d·(q−8)` robust. With `dmin = d`, `scale ∈ {2,4}` and `min = 1.5·scale`,
    /// the weight is `d·scale·q2 − d·min = d·scale·(q2 − 1.5)` — zero-mean over q2 ∈ {0,1,2,3}. That
    /// gives the dot a robust magnitude (no pathological sign cancellation), so the int8-activation
    /// result tracks the full-precision reference within the loose rel tolerance. Codes (`qs`) stay
    /// random (0–3); fully-random scales/mins (mostly-positive weights) make the dot small and the
    /// loose bound flaky, which is why the bit-identity test — not this one — owns the full range.
    fn build_q2k_wc(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 84, seed ^ 0x9E37);
        let mut s = seed;
        for b in 0..nb {
            let blk = &mut w[b * 84..b * 84 + 84];
            for is in 0..16usize {
                let scale = 2 + 2 * ((lcg(&mut s) >> 40) % 2); // {2, 4}
                let minv = scale * 3 / 2; // {3, 6} = 1.5·scale → zero-mean weight
                blk[is] = (scale | (minv << 4)) as u8;
            }
            // qs[16..80] stays random → codes 0..3.
            put_f16(&mut blk[80..82], d); // d
            put_f16(&mut blk[82..84], d); // dmin = d (required for the zero-centering algebra)
        }
        w
    }

    /// The SIMD dispatch (whatever tier this CPU has) must be BIT-IDENTICAL to the scalar oracle for
    /// both the single-token and batch Q2_K kernels: the per-sub-block dot is an integer sum (exact,
    /// order-free) and the `q8.d·(d·sd − dmin·sm)` epilogue is shared, so every tier collapses to the
    /// same bits. Full-random scale/min/code range. Trivial on non-x86 (dispatch == scalar).
    #[test]
    fn q2k_simd_bit_identical_to_scalar() {
        for in_f in [256usize, 512, 768] {
            let nb = in_f / 256;
            let w = build_q2k_rand(nb, 0.05, 0.015, 60);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 61 + r as u64)))
                .collect();
            // Single-token: SIMD dispatch vs scalar oracle.
            for (r, q8) in q8s.iter().enumerate() {
                let simd = vec_dot_q2k(&w, q8, in_f);
                let scalar = vec_dot_q2k_scalar(&w, q8, in_f);
                assert_eq!(
                    simd.to_bits(),
                    scalar.to_bits(),
                    "q2k single in_f={in_f} row={r}: simd {simd}, scalar {scalar}"
                );
            }
            // Batch: SIMD dispatch vs scalar oracle.
            let mut simd_out = vec![0f32; m];
            vec_dot_q2k_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q2k_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q2k batch in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation Q2_K kernel must track the full-precision
    /// affine dequant (`d·scale·q2 − dmin·min`) · f32-activation reference. Q2_K weights are computed
    /// exactly by the kernel (no weight-quant loss beyond what `dequant_block` also applies), so vs
    /// the QUANTIZED activation the kernel sees the only slack is f32 accumulation order → tight 1e-3
    /// (isolates integer-dot correctness, as the Q4_0/IQ4_XS siblings do). vs the full-precision
    /// activation, the looser 2e-2 absorbs the lossy int8 activation quant. Covers m=1 and m>1.
    #[test]
    fn q2k_matches_dequant_reference() {
        for in_f in [256usize, 512, 768] {
            let nb = in_f / 256;
            let w = build_q2k_wc(nb, 0.05, 62);
            let wref = dequant_block(DType::Q2K, &w).unwrap();
            // Sanity: reference actually spans nonzero weights.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 63 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_q2k_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "q2k batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "q2k batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_q2k(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "q2k single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 2e-2,
                    "q2k single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    /// Build `nb` fully-random 110-byte Q3_K super-blocks with fixed `d` — every 2-bit `qs` field,
    /// `hmask` bit-plane and 6-bit packed scale is exercised (any byte pattern is a legal Q3_K
    /// block). Used by the bit-identity oracle test.
    fn build_q3k_rand(nb: usize, d: f32, seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 110, seed);
        for b in 0..nb {
            put_f16(&mut w[b * 110 + 108..b * 110 + 110], d);
        }
        w
    }

    /// Inverse of `q3k_scales`'s aux bit-shuffle: pack 16 chosen 6-bit scale values (`dvals`) into the
    /// 12-byte field so the kernel/`dequant_block` decode them back. Used only to place the scales in
    /// a well-conditioned band; both the kernel and the reference share the forward shuffle, so this
    /// need not be exact for correctness — but it IS the true inverse (verified against `q3k_scales`).
    fn pack_q3k_scales(dvals: &[u8; 16]) -> [u8; 12] {
        let mut s = [0u8; 12];
        for j in 0..4usize {
            s[j] = (dvals[j] & 0xF) | ((dvals[8 + j] & 0xF) << 4);
            s[4 + j] = (dvals[4 + j] & 0xF) | ((dvals[12 + j] & 0xF) << 4);
            s[8 + j] = ((dvals[j] >> 4) & 3)
                | (((dvals[4 + j] >> 4) & 3) << 2)
                | (((dvals[8 + j] >> 4) & 3) << 4)
                | (((dvals[12 + j] >> 4) & 3) << 6);
        }
        s
    }

    /// Build well-conditioned Q3_K blocks: a small mixed-sign scale band (`sc6 ∈ 28..36` → `sc6−32 ∈
    /// [−4,4]`) with random `hmask`/`qs` codes. Weight is `d·(sc6−32)·(q3−4)`; the mixed-sign scale ×
    /// signed offset code (`q3−4 ∈ [−4,3]`) gives the dot a robust magnitude (no pathological sign
    /// cancellation), so the int8-activation result tracks the full-precision reference within the
    /// loose tolerance — the bit-identity test, not this one, owns the full random scale range.
    fn build_q3k_wc(nb: usize, d: f32, mut seed: u64) -> Vec<u8> {
        let mut w = det_bytes(nb * 110, seed ^ 0x51D3);
        for b in 0..nb {
            let blk = &mut w[b * 110..b * 110 + 110];
            // hmask[0..32] and qs[32..96] stay random → codes 0..7.
            let mut dvals = [0u8; 16];
            for d6 in dvals.iter_mut() {
                *d6 = (28 + (lcg(&mut seed) >> 40) % 9) as u8; // sc6 ∈ 28..36
            }
            blk[96..108].copy_from_slice(&pack_q3k_scales(&dvals));
            put_f16(&mut blk[108..110], d);
        }
        w
    }

    /// The SIMD dispatch (whatever tier this CPU has) must be BIT-IDENTICAL to the scalar oracle for
    /// both the single-token and batch Q3_K kernels: the per-sub-block dot is an integer sum (exact,
    /// order-free) and the `d·d8·isum` epilogue is shared, so every tier collapses to the same bits.
    /// Full-random scale/code/hmask range. Trivial on non-x86 (dispatch == scalar).
    #[test]
    fn q3k_simd_bit_identical_to_scalar() {
        for in_f in [256usize, 512, 768] {
            let nb = in_f / 256;
            let w = build_q3k_rand(nb, 0.05, 70);
            let m = 5usize;
            let q8s: Vec<Q8> = (0..m)
                .map(|r| quantize_q8(&det_x(in_f, 71 + r as u64)))
                .collect();
            // Single-token: SIMD dispatch vs scalar oracle.
            for (r, q8) in q8s.iter().enumerate() {
                let simd = vec_dot_q3k(&w, q8, in_f);
                let scalar = vec_dot_q3k_scalar(&w, q8, in_f);
                assert_eq!(
                    simd.to_bits(),
                    scalar.to_bits(),
                    "q3k single in_f={in_f} row={r}: simd {simd}, scalar {scalar}"
                );
            }
            // Batch: SIMD dispatch vs scalar oracle.
            let mut simd_out = vec![0f32; m];
            vec_dot_q3k_batch(&w, &q8s, in_f, &mut simd_out);
            let mut scalar_out = vec![0f32; m];
            vec_dot_q3k_batch_scalar(&w, &q8s, in_f, &mut scalar_out);
            for r in 0..m {
                assert_eq!(
                    simd_out[r].to_bits(),
                    scalar_out[r].to_bits(),
                    "q3k batch in_f={in_f} row={r}: simd {}, scalar {}",
                    simd_out[r],
                    scalar_out[r]
                );
            }
        }
    }

    /// End-to-end tolerance-parity: the int8-activation Q3_K kernel must track the full-precision
    /// dequant (`d·(sc6−32)·(q3−4)`) · f32-activation reference. Q3_K weights are computed exactly by
    /// the kernel (no weight-quant loss beyond what `dequant_block` also applies), so vs the QUANTIZED
    /// activation the kernel sees the only slack is f32 accumulation order → tight 1e-3 (isolates
    /// integer-dot correctness, as the Q2_K/IQ4_XS siblings do). vs the full-precision activation, the
    /// looser 2e-2 absorbs the lossy int8 activation quant. Covers m=1 and m>1.
    #[test]
    fn q3k_matches_dequant_reference() {
        for in_f in [256usize, 512, 768] {
            let nb = in_f / 256;
            let w = build_q3k_wc(nb, 0.0006, 72);
            let wref = dequant_block(DType::Q3K, &w).unwrap();
            // Sanity: reference actually spans nonzero weights.
            assert!(wref.iter().any(|&v| v != 0.0));
            let xs: Vec<Vec<f32>> = (0..4).map(|i| det_x(in_f, 73 + i)).collect();
            let q8s: Vec<Q8> = xs.iter().map(|x| quantize_q8(x)).collect();

            // Batch (m>1).
            let mut got = vec![0f32; xs.len()];
            vec_dot_q3k_batch(&w, &q8s, in_f, &mut got);
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got[r], want_q) < 1e-3,
                    "q3k batch(quant-ref) in_f={in_f} row={r}: got {}, want {want_q}",
                    got[r]
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got[r], want_f) < 2e-2,
                    "q3k batch(full-ref) in_f={in_f} row={r}: got {}, want {want_f}",
                    got[r]
                );
            }
            // Single-row (m=1).
            for (r, (x, q8)) in xs.iter().zip(q8s.iter()).enumerate() {
                let got1 = vec_dot_q3k(&w, q8, in_f);
                let want_q = dot(&wref, &dequant_q8(q8));
                assert!(
                    rel_err(got1, want_q) < 1e-3,
                    "q3k single(quant-ref) in_f={in_f} row={r}: got {got1}, want {want_q}"
                );
                let want_f = dot(&wref, x);
                assert!(
                    rel_err(got1, want_f) < 2e-2,
                    "q3k single(full-ref) in_f={in_f} row={r}: got {got1}, want {want_f}"
                );
            }
        }
    }

    #[test]
    fn f16_dot_matches_reference() {
        let n = 257; // odd, exercises the tail past the 8-wide chunks
        let x = det_x(n, 5);
        let wf = det_x(n, 6);
        let wbytes: Vec<u8> = wf
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let wref: Vec<f32> = wf
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        assert!(rel_err(dot_f16(&wbytes, &x), dot(&wref, &x)) < 1e-4);
    }

    #[test]
    fn bf16_dot_matches_reference() {
        let n = 130;
        let x = det_x(n, 7);
        let wf = det_x(n, 8);
        let wbytes: Vec<u8> = wf
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()) // bf16 = top 16 bits
            .collect();
        let wref: Vec<f32> = wf
            .iter()
            .map(|&v| f32::from_bits((v.to_bits() >> 16) << 16))
            .collect();
        // dot_bf16 now uses the same 8-accumulator chunked structure as `dot`, so on the same
        // bf16-rounded weights it is bit-identical to `dot` — not merely close.
        assert_eq!(
            dot_bf16(&wbytes, &x).to_bits(),
            dot(&wref, &x).to_bits(),
            "dot_bf16 must match dot's 8-accumulator reduction bit-for-bit",
        );
    }

    // #1: dot_bf16 must reproduce an explicit 8-accumulator reference (the new structure) and
    // that reduction order must differ from the old single serial accumulator on this vector —
    // documenting that the bf16 result changed bit-for-bit (a legitimate float reorder).
    #[test]
    fn bf16_dot_is_8_accumulator_reorder() {
        let n = 2050; // many chunks + a tail so lane grouping actually reorders the sum
        let x = det_x(n, 11);
        let wf = det_x(n, 12);
        let wbytes: Vec<u8> = wf
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let wref: Vec<f32> = wf
            .iter()
            .map(|&v| f32::from_bits((v.to_bits() >> 16) << 16))
            .collect();

        // Explicit 8-lane reference (mirrors dot/dot_f16).
        let mut acc = [0f32; 8];
        let chunks = n / 8;
        for c in 0..chunks {
            for (j, ac) in acc.iter_mut().enumerate() {
                let i = c * 8 + j;
                *ac = wref[i].mul_add(x[i], *ac);
            }
        }
        let mut want = acc.iter().sum::<f32>();
        for i in chunks * 8..n {
            want = wref[i].mul_add(x[i], want);
        }

        // Old serial single-accumulator reference (what dot_bf16 used to compute).
        let mut serial = 0f32;
        for i in 0..n {
            serial = wref[i].mul_add(x[i], serial);
        }

        assert_eq!(
            dot_bf16(&wbytes, &x).to_bits(),
            want.to_bits(),
            "dot_bf16 must equal the 8-accumulator reference bit-for-bit",
        );
        assert_ne!(
            want.to_bits(),
            serial.to_bits(),
            "8-accumulator reorder must differ from the old serial chain (bf16 golden shifts)",
        );
    }

    // #2: the 256-block debug_assert fires when in_f is not a multiple of 256 (debug builds).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must be a multiple of 256")]
    fn q4k_non_multiple_of_256_debug_asserts() {
        let q8 = quantize_q8(&det_x(256, 1));
        let w = det_bytes(144, 2);
        let _ = vec_dot_q4k(&w, &q8, 200); // 200 % 256 != 0
    }

    // #4: q4k_decode_row reproduces the inline scalar decode bit-for-bit on a sample row.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k_decode_row_matches_scalar_decode() {
        if !is_x86_feature_detected!("avx2") {
            return; // helper is avx2-gated; nothing to check on this host
        }
        let nb = 3usize;
        let mut row = det_bytes(nb * 144, 99);
        for b in 0..nb {
            put_f16(&mut row[b * 144..b * 144 + 2], 0.05);
            put_f16(&mut row[b * 144 + 2..b * 144 + 4], 0.015);
        }
        let mut d_arr = vec![0f32; nb];
        let mut dmin_arr = vec![0f32; nb];
        let mut sc_arr = vec![[0u32; 8]; nb];
        let mut m_arr = vec![[0u32; 8]; nb];
        let mut q4_flat = vec![0u8; nb * 256];
        unsafe {
            q4k_decode_row(
                &row,
                nb,
                &mut d_arr,
                &mut dmin_arr,
                &mut sc_arr,
                &mut m_arr,
                &mut q4_flat,
            );
        }
        for b in 0..nb {
            let blk = &row[b * 144..b * 144 + 144];
            assert_eq!(d_arr[b].to_bits(), rdf16(&blk[0..2]).to_bits());
            assert_eq!(dmin_arr[b].to_bits(), rdf16(&blk[2..4]).to_bits());
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            for s in 0..8usize {
                let (sc, mv) = k4(s, scales);
                assert_eq!(sc_arr[b][s], sc, "sc mismatch b={b} s={s}");
                assert_eq!(m_arr[b][s], mv, "m mismatch b={b} s={s}");
                let hi = s % 2 == 1;
                let half = s / 2;
                let qbyte = &qs[half * 32..half * 32 + 32];
                for l in 0..32 {
                    let want = if hi { qbyte[l] >> 4 } else { qbyte[l] & 0xF };
                    assert_eq!(
                        q4_flat[b * 256 + s * 32 + l],
                        want,
                        "flat b={b} s={s} l={l}"
                    );
                }
            }
        }
    }

    // ── Batch kernel bit-identity tests ──────────────────────────────────────
    //
    // For each quant type, assert that `vec_dot_qXk_batch` produces the same
    // f32 bits as calling `vec_dot_qXk` per-activation on every token.

    #[test]
    fn q4k_batch_bit_identical_to_single() {
        let in_f = 512; // 2 super-blocks
        let nb = in_f / 256;
        let m = 7usize; // odd count to test non-power-of-2
        let mut w = det_bytes(nb * 144, 20);
        for k in 0..nb {
            put_f16(&mut w[k * 144..k * 144 + 2], 0.05);
            put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 30 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q4k_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q4k(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q4k batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    #[test]
    fn q6k_batch_bit_identical_to_single() {
        let in_f = 512;
        let nb = in_f / 256;
        let m = 5usize;
        let mut w = det_bytes(nb * 210, 21);
        for k in 0..nb {
            put_f16(&mut w[k * 210 + 208..k * 210 + 210], 0.04);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 40 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q6k_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q6k(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q6k batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    #[test]
    fn q8_0_batch_bit_identical_to_single() {
        let in_f = 512; // 2 super-blocks = 16 Q8_0 weight blocks
        let nb_w = in_f / 32;
        let m = 6usize;
        let mut w = det_bytes(nb_w * 34, 22);
        for k in 0..nb_w {
            put_f16(&mut w[k * 34..k * 34 + 2], 0.03);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 50 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q8_0_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q8_0(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q8_0 batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    #[test]
    fn q5k_batch_bit_identical_to_single() {
        let in_f = 512;
        let nb = in_f / 256;
        let m = 4usize;
        let mut w = det_bytes(nb * 176, 23);
        for k in 0..nb {
            put_f16(&mut w[k * 176..k * 176 + 2], 0.05);
            put_f16(&mut w[k * 176 + 2..k * 176 + 4], 0.01);
        }
        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 60 + r as u64)))
            .collect();
        let mut batch_out = vec![0f32; m];
        vec_dot_q5k_batch(&w, &q8s, in_f, &mut batch_out);
        for r in 0..m {
            let want = vec_dot_q5k(&w, &q8s[r], in_f);
            assert_eq!(
                batch_out[r].to_bits(),
                want.to_bits(),
                "q5k batch[{r}]: got {}, want {}",
                batch_out[r],
                want,
            );
        }
    }

    /// Assert the 8-row tile produces bit-identical results to `vec_dot_q4k` per (row, token).
    /// Uses 11 rows to exercise the full remainder path: 8-row tile + 2-row pair + 1-row tail.
    #[test]
    fn q4k_batch8_bit_identical_to_single() {
        let n_rows = 11usize; // 8 + 2-row-pair + 1-row-tail
        let in_f = 512; // 2 super-blocks
        let nb = in_f / 256;
        let m = 9usize; // non-power-of-2 token count

        // Build n_rows random weight rows with valid f16 d/dmin.
        let ws: Vec<Vec<u8>> = (0..n_rows)
            .map(|i| {
                let mut w = det_bytes(nb * 144, 100 + i as u64);
                for k in 0..nb {
                    put_f16(&mut w[k * 144..k * 144 + 2], 0.05);
                    put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015);
                }
                w
            })
            .collect();

        let q8s: Vec<Q8> = (0..m)
            .map(|r| quantize_q8(&det_x(in_f, 200 + r as u64)))
            .collect();

        // Reference: vec_dot_q4k per (row, token) — the scalar single-token oracle.
        let want: Vec<Vec<f32>> = (0..n_rows)
            .map(|i| (0..m).map(|r| vec_dot_q4k(&ws[i], &q8s[r], in_f)).collect())
            .collect();

        // Test the 8-row tile for rows 0..8.
        let mut got0 = vec![0f32; m];
        let mut got1 = vec![0f32; m];
        let mut got2 = vec![0f32; m];
        let mut got3 = vec![0f32; m];
        let mut got4 = vec![0f32; m];
        let mut got5 = vec![0f32; m];
        let mut got6 = vec![0f32; m];
        let mut got7 = vec![0f32; m];
        vec_dot_q4k_batch8(
            [
                &ws[0], &ws[1], &ws[2], &ws[3], &ws[4], &ws[5], &ws[6], &ws[7],
            ],
            &q8s,
            in_f,
            [
                &mut got0, &mut got1, &mut got2, &mut got3, &mut got4, &mut got5, &mut got6,
                &mut got7,
            ],
        );

        // Test 2-row remainder (rows 8..10) — same path as the dispatch remainder.
        let mut got8 = vec![0f32; m];
        let mut got9 = vec![0f32; m];
        vec_dot_q4k_batch2(&ws[8], &ws[9], &q8s, in_f, &mut got8, &mut got9);

        // Test 1-row tail (row 10).
        let mut got10 = vec![0f32; m];
        vec_dot_q4k_batch(&ws[10], &q8s, in_f, &mut got10);

        let got_all: [&Vec<f32>; 11] = [
            &got0, &got1, &got2, &got3, &got4, &got5, &got6, &got7, &got8, &got9, &got10,
        ];
        for i in 0..n_rows {
            for r in 0..m {
                assert_eq!(
                    got_all[i][r].to_bits(),
                    want[i][r].to_bits(),
                    "q4k_batch8 row {i} token {r}: got {}, want {}",
                    got_all[i][r],
                    want[i][r],
                );
            }
        }
    }

    /// Regression: Q8_0 is a NATIVE 32-block weight, so a tensor's `in_f` can be a multiple of 32
    /// but NOT 256 (gemma3-1b's attention projections are 1152 = 36×32 = 4.5×256). The `Op::Linear`
    /// dispatch used to route Q8_0 through the 256-superblock `quantize_q8` path, whose
    /// `nb = in_f/256` truncates to `floor(1152/256)*256 = 1024` elements — silently dropping the
    /// last 128 (~11%) of every dot. The fix routes such tensors to the native 32-block
    /// `vec_dot_q8_0_32_batch` (the kernel the dispatch now invokes for both the m==1 single-row
    /// slice and the m>1 batch). The dispatch method lives on the graph executor and is awkward to
    /// drive from a unit test, so this exercises the routed kernel directly at in_f=1152, plus a
    /// 256-aligned case to confirm the fast path is unchanged, and asserts the sub-256 tail the old
    /// path dropped is numerically significant (so the bug was real, not cosmetic).
    #[test]
    fn q8_0_dispatch_sub256_in_f_1152() {
        for &in_f in &[1152usize, 256usize] {
            let nb = in_f / 32;
            let mut w = det_bytes(nb * 34, 77);
            for k in 0..nb {
                put_f16(&mut w[k * 34..k * 34 + 2], 0.03);
            }
            let wref = dequant_block(DType::Q8_0, &w).unwrap();
            assert_eq!(wref.len(), in_f);

            let m = 4usize;
            let xs: Vec<Vec<f32>> = (0..m).map(|r| det_x(in_f, 300 + r as u64)).collect();
            let q8s: Vec<Q8x32> = xs.iter().map(|x| quantize_q8_32(x)).collect();

            // m>1 batch entry: the arm the dispatch's `else if q8_0_blk32` branch calls.
            let mut got_batch = vec![0f32; m];
            vec_dot_q8_0_32_batch(&w, &q8s, in_f, &mut got_batch);

            for (r, q8) in q8s.iter().enumerate() {
                // full-precision weight · the int8 activation the kernel actually sees.
                let want = dot(&wref, &dequant_q8_32(q8));
                assert!(
                    rel_err(got_batch[r], want) < 1e-3,
                    "q8_0 in_f={in_f} batch row {r}: got {}, want {want}",
                    got_batch[r],
                );

                // m==1 entry: the dispatch quantizes a single row and passes a 1-element slice —
                // must equal the batch result for that row (same kernel, one token).
                let mut got1 = [0f32];
                vec_dot_q8_0_32_batch(&w, std::slice::from_ref(q8), in_f, &mut got1);
                assert_eq!(
                    got1[0].to_bits(),
                    got_batch[r].to_bits(),
                    "q8_0 in_f={in_f} m==1 row {r} diverges from batch",
                );
            }

            // Prove the old truncation mattered: at in_f=1152 a dot over only the first 1024
            // elements (what the 256-block `nb = in_f/256` path summed) differs from the full dot
            // by far more than the 1e-3 tolerance above.
            if in_f == 1152 {
                let x = &xs[0];
                let full = dot(&wref, x);
                let trunc = dot(&wref[..1024], &x[..1024]);
                assert!(
                    rel_err(trunc, full) > 1e-2,
                    "expected sub-256 tail to be significant: full {full}, trunc {trunc}",
                );
            }
        }
    }

    // ── Op::Linear dispatch: awkward-in_f coverage for EVERY weight quant dtype ───────────────
    //
    // The Q8_0 bug lived in the `Op::Linear` DISPATCH (lib.rs), not in a kernel: the executor
    // picked the 256-superblock `quantize_q8` activation for a NATIVE 32-block Q8_0 weight, and
    // `quantize_q8`/`vec_dot_q8_0`'s `nb = in_f/256` then silently dropped the sub-256 tail
    // (gemma3-1b's `in_f = 1152 = 4.5×256` → summed only 1024/1152 ≈ 89%). The kernel-direct
    // tests above cannot see a *routing* mistake — only driving the real graph executor can. This
    // test builds a small `Op::Linear` graph per dtype and runs it through the actual
    // `CpuBackend::execute` path at a format-valid but non-256 (or non-super-block) `in_f`, for
    // BOTH the m==1 (single-row) and m>1 (batch) dispatch arms, and checks the output against the
    // full-precision `dequant_block`·f32-activation reference. The only error source is the lossy
    // int8 activation quant, so the tolerance is 2e-2 of the output's dynamic range — a ~11%
    // truncation shifts every output far past that.
    //
    // Confirmed to catch the class: on the pre-fix code, Q8_0 @ in_f=1152 routes to
    // `quantize_q8` + `vec_dot_q8_0` (256-block), which sum `floor(1152/256)*256 = 1024` elements
    // — every output deviates ~11% (≫ the 2e-2 bound) so the test fails. (In debug it also trips
    // the `in_f % 256` assert now on `vec_dot_q8_0`.) Post-fix it routes to the 32-block
    // `vec_dot_q8_0_32_batch` and passes.
    #[test]
    fn op_linear_dispatch_awkward_in_f_all_quants() {
        use crate::CpuBackend;
        use infr_core::backend::{Backend, Bindings, BufferUsage};
        use infr_core::graph::{Graph, Op};
        use infr_core::tensor::TensorDesc;

        // f32 reference: out[r*out_f + o] = Σ_k x[r*in_f+k] · w[o*in_f+k].
        fn ref_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
            let mut out = vec![0f32; m * out_f];
            for r in 0..m {
                for o in 0..out_f {
                    out[r * out_f + o] =
                        dot(&w[o * in_f..o * in_f + in_f], &x[r * in_f..r * in_f + in_f]);
                }
            }
            out
        }

        // Drive ONE Op::Linear through the real CPU graph executor and read the Output back.
        fn run_linear(
            dtype: DType,
            wbytes: &[u8],
            xs: &[f32],
            m: usize,
            in_f: usize,
            out_f: usize,
        ) -> Vec<f32> {
            let mut g = Graph::new();
            let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
            let w = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
            let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
            g.push(Op::Linear {
                x,
                weight: w,
                dst,
                m: m as u32,
                in_f: in_f as u32,
                out_f: out_f as u32,
                w_off: 0,
            });
            let be = CpuBackend::new();
            let alloc_upload = |bytes: &[u8]| -> Box<dyn infr_core::backend::Buffer> {
                let b = be
                    .alloc(bytes.len().max(4), BufferUsage::Activations)
                    .unwrap();
                be.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = alloc_upload(bytemuck::cast_slice(xs));
            let wb = alloc_upload(wbytes);
            let ob = be
                .alloc((m * out_f * 4).max(4), BufferUsage::Activations)
                .unwrap();
            let mut binds = Bindings::new();
            binds.bind(x, xb.as_ref());
            binds.bind(w, wb.as_ref());
            binds.bind(dst, ob.as_ref());
            let plan = be.compile(&g).unwrap();
            be.execute(plan.as_ref(), &binds).unwrap();
            let mut bytes = vec![0u8; m * out_f * 4];
            be.download(ob.as_ref(), &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        }

        // Valid GGUF bytes for `tb` format-blocks (the whole [out_f, in_f] tensor is just a flat run
        // of blocks). Reuses the per-dtype block builders the kernel tests already validate; the
        // few "legacy" formats without a helper are trivial (fixed f16 scale + random codes).
        fn synth(dtype: DType, tb: usize) -> Vec<u8> {
            let put_scale = |bpb: usize, doff: usize, dval: f32, seed: u64| -> Vec<u8> {
                let mut w = det_bytes(tb * bpb, seed);
                for k in 0..tb {
                    put_f16(&mut w[k * bpb + doff..k * bpb + doff + 2], dval);
                }
                w
            };
            match dtype {
                DType::Q4_0 => put_scale(18, 0, 0.03, 501),
                DType::Q5_0 => put_scale(22, 0, 0.03, 502),
                DType::Q8_0 => put_scale(34, 0, 0.03, 503),
                DType::Q2_0 => put_scale(18, 0, 0.03, 504),
                DType::Q6K => put_scale(210, 208, 0.04, 505),
                DType::Q4K => {
                    let mut w = det_bytes(tb * 144, 506);
                    for k in 0..tb {
                        put_f16(&mut w[k * 144..k * 144 + 2], 0.05);
                        put_f16(&mut w[k * 144 + 2..k * 144 + 4], 0.015);
                    }
                    w
                }
                DType::Q5K => {
                    let mut w = det_bytes(tb * 176, 507);
                    for k in 0..tb {
                        put_f16(&mut w[k * 176..k * 176 + 2], 0.05);
                        put_f16(&mut w[k * 176 + 2..k * 176 + 4], 0.01);
                    }
                    w
                }
                DType::Q4_1 => build_q4_1(tb, 0.02, -0.01, 508),
                DType::Q5_1 => build_q5_1(tb, 0.02, -0.01, 509),
                DType::Q2K => build_q2k_rand(tb, 0.05, 0.015, 510),
                DType::Q3K => build_q3k_rand(tb, 0.05, 511),
                DType::Iq4Nl => build_iq4nl(tb, 0.02, 512),
                DType::Iq4Xs => build_iq4xs_rand(tb, 0.05, 513),
                DType::Iq2S => build_iq2s_rand(tb, 0.05, 514),
                DType::Iq2Xs => build_iq2xs_rand(tb, 0.05, 515),
                DType::Iq2Xxs => build_iq2xxs_rand(tb, 0.05, 516),
                DType::Iq3S => build_iq3s_rand(tb, 0.05, 517),
                DType::Iq3Xxs => build_iq3xxs_rand(tb, 0.05, 518),
                DType::Iq1S => build_iq1s_rand(tb, 0.05, 519),
                DType::Iq1M => build_iq1m_rand(tb, 0.05, 520),
                DType::Tq1_0 => build_tq1_0_rand(tb, 0.05, 521),
                DType::Tq2_0 => build_tq2_0_rand(tb, 0.05, 522),
                // `_band` variants clamp the per-block exponent/scale to a narrow range: the
                // fully-random `e`/UE4M3 byte spans 2^±127, which overflows the f32 reference dot.
                DType::Mxfp4 => build_mxfp4_band(tb, 523),
                DType::Nvfp4 => build_nvfp4_band(tb, 524),
                other => panic!("synth: unhandled dtype {other:?}"),
            }
        }

        // (dtype, weight-block elems, awkward in_f set). Every in_f is a whole number of weight
        // blocks (format-valid) but deliberately NOT super-block-friendly — the exact shape that
        // exposed the Q8_0 mis-route: 1152/320 are ×32 not ×256; 128/320 are ×64 not ×256.
        let cases: &[(DType, usize, &[usize])] = &[
            // 32-block quants → Q8x32 activation; in_f ×32 but not ×256.
            (DType::Q4_0, 32, &[1152, 320]),
            (DType::Q4_1, 32, &[1152, 320]),
            (DType::Q5_0, 32, &[1152, 320]),
            (DType::Q5_1, 32, &[1152, 320]),
            (DType::Q8_0, 32, &[1152, 320]),
            (DType::Iq4Nl, 32, &[1152, 320]),
            (DType::Mxfp4, 32, &[1152, 320]),
            // 64-block quants → Q8x32 activation, 2 sub-blocks/block; in_f ×64 but not ×256.
            (DType::Q2_0, 64, &[128, 320]),
            (DType::Nvfp4, 64, &[128, 320]),
            // 256-block quants → Q8 super-block activation; both in_f are ×256 (all valid).
            (DType::Q4K, 256, &[256, 512]),
            (DType::Q5K, 256, &[256, 512]),
            (DType::Q6K, 256, &[256, 512]),
            (DType::Q2K, 256, &[256, 512]),
            (DType::Q3K, 256, &[256, 512]),
            (DType::Iq4Xs, 256, &[256, 512]),
            (DType::Iq2S, 256, &[256, 512]),
            (DType::Iq2Xs, 256, &[256, 512]),
            (DType::Iq2Xxs, 256, &[256, 512]),
            (DType::Iq3S, 256, &[256, 512]),
            (DType::Iq3Xxs, 256, &[256, 512]),
            (DType::Iq1S, 256, &[256, 512]),
            (DType::Iq1M, 256, &[256, 512]),
            (DType::Tq1_0, 256, &[256, 512]),
            (DType::Tq2_0, 256, &[256, 512]),
        ];

        // out_f = 13: odd and > 8, so the m>1 arm exercises Q4_K/Q6_K's 8-row batch group PLUS the
        // 2-row pair and the odd single-row remainder tails.
        let out_f = 13usize;
        for &(dtype, blk, in_fs) in cases {
            for &in_f in in_fs {
                let tb = out_f * (in_f / blk);
                let wbytes = synth(dtype, tb);
                let wref = dequant_block(dtype, &wbytes).unwrap();
                assert_eq!(wref.len(), out_f * in_f, "{dtype:?}: dequant length");
                for &m in &[1usize, 5] {
                    let xs = det_x(m * in_f, 900 + in_f as u64 + m as u64);
                    let got = run_linear(dtype, &wbytes, &xs, m, in_f, out_f);
                    let reference = ref_linear(&xs, &wref, m, in_f, out_f);
                    // Absolute tolerance = 2e-2 × the output's dynamic range: robust to isolated
                    // near-zero (cancelled) outputs, yet a truncated/mis-routed dot moves the
                    // large-magnitude outputs by a big fraction of that range and is caught.
                    let scale = reference.iter().fold(1e-3f32, |a, &v| a.max(v.abs()));
                    for (i, (&g_i, &r_i)) in got.iter().zip(reference.iter()).enumerate() {
                        let diff = (g_i - r_i).abs();
                        assert!(
                            diff <= 2e-2 * scale,
                            "{dtype:?} in_f={in_f} m={m} out[{i}]: got {g_i}, want {r_i} (|diff| {diff} > {})",
                            2e-2 * scale,
                        );
                    }
                }
            }
        }
    }

    // ── SIMD tier agreement ────────────────────────────────────────────────────────────────────
    //
    // Every `vec_dot_*` is a runtime dispatch (avx512vnni → avx512bw → avx2 → scalar), so a plain
    // `cargo test` only ever exercises the TOP tier the test box happens to have — on this repo's
    // AMD dev box that is avx512vnni, which means the avx2 and scalar kernels ship untested and a
    // format decoded wrongly in one of them surfaces only on a user's older machine.
    //
    // These tests call each tier DIRECTLY on identical inputs and hold it to the scalar kernel,
    // which is the readable port of llama.cpp's `ggml_vec_dot_*` (and the only tier the non-x86
    // build uses). They are the per-kernel counterpart of `CpuBackend::reference` vs production:
    // that pair bounds the int8-activation error at the BACKEND level, this one proves the tiers
    // implement the same arithmetic at the KERNEL level.
    //
    // Bound: the tiers accumulate the same integer products and differ only in the ORDER of the
    // per-block f32 accumulation, so agreement is near-exact. `TIER_TOL` is relative to the
    // scalar result's magnitude — a mis-decoded format lands at O(1) relative and cannot hide.

    /// Relative bound for a SIMD tier against the scalar kernel — f32 accumulation order only.
    ///
    /// Measured worst on the dev box (Zen5, all of avx2 / avx512bw / avx512vnni present, so every
    /// tier below actually ran): single-row `9.7e-8`, batch `2.3e-7`, and the native-32-block
    /// family `0.0` — those kernels are BIT-IDENTICAL to their scalar twins. `1e-5` is ~40x
    /// headroom over the worst, and mutation-checked: at `1e-9` the avx512bw arms fail, and
    /// scoring one format against another format's scalar kernel fails outright.
    const TIER_TOL: f32 = 1e-5;

    #[cfg(target_arch = "x86_64")]
    fn tier_rel(got: f32, want: f32) -> f32 {
        (got - want).abs() / want.abs().max(1e-3)
    }

    /// Single-row (`m=1`, Q8 super-block activation) int8 dots: avx2 + avx512bw vs scalar.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_tiers_match_the_scalar_kernel() {
        type Dot = fn(&[u8], &Q8, usize) -> f32;
        let has2 = is_x86_feature_detected!("avx2");
        let has5 = is_x86_feature_detected!("avx512bw");
        let in_f = 512usize; // 2 super-blocks: exercises the per-block accumulation, not just one
        let q8 = quantize_q8(&det_x(in_f, 0xA11CE));
        // (label, dtype, scalar, avx2, avx512bw)
        let cases: &[(&str, DType, Dot, Dot, Dot)] = &[
            (
                "q4k",
                DType::Q4K,
                vec_dot_q4k_scalar,
                |r, q, n| unsafe { vec_dot_q4k_avx2(r, q, n) },
                |r, q, n| unsafe { vec_dot_q4k_avx512bw(r, q, n) },
            ),
            (
                "q2k",
                DType::Q2K,
                vec_dot_q2k_scalar,
                |r, q, n| unsafe { vec_dot_q2k_avx2(r, q, n) },
                |r, q, n| unsafe { vec_dot_q2k_avx512bw(r, q, n) },
            ),
            (
                "q3k",
                DType::Q3K,
                vec_dot_q3k_scalar,
                |r, q, n| unsafe { vec_dot_q3k_avx2(r, q, n) },
                |r, q, n| unsafe { vec_dot_q3k_avx512bw(r, q, n) },
            ),
            (
                "q5k",
                DType::Q5K,
                vec_dot_q5k_scalar,
                |r, q, n| unsafe { vec_dot_q5k_avx2(r, q, n) },
                |r, q, n| unsafe { vec_dot_q5k_avx512bw(r, q, n) },
            ),
            (
                "q6k",
                DType::Q6K,
                vec_dot_q6k_scalar,
                |r, q, n| unsafe { vec_dot_q6k_avx2(r, q, n) },
                |r, q, n| unsafe { vec_dot_q6k_avx512bw(r, q, n) },
            ),
            (
                "q8_0",
                DType::Q8_0,
                vec_dot_q8_0_scalar,
                |r, q, n| unsafe { vec_dot_q8_0_avx2(r, q, n) },
                |r, q, n| unsafe { vec_dot_q8_0_avx512bw(r, q, n) },
            ),
            (
                "iq4xs",
                DType::Iq4Xs,
                vec_dot_iq4xs_scalar,
                |r, q, n| unsafe { vec_dot_iq4xs_avx2(r, q, n) },
                |r, q, n| unsafe { vec_dot_iq4xs_avx512bw(r, q, n) },
            ),
        ];
        let mut worst = 0f32;
        for &(name, dtype, scalar, avx2, avx512) in cases {
            let w = infr_testkit::synth_weight(dtype, in_f, 0x5EED);
            let want = scalar(&w, &q8, in_f);
            assert!(
                want.abs() > 1e-3,
                "{name}: scalar result is ~0, case is vacuous"
            );
            for (tier, on, f) in [("avx2", has2, avx2), ("avx512bw", has5, avx512)] {
                if !on {
                    eprintln!("skip {name} {tier}: not supported by this CPU");
                    continue;
                }
                let got = f(&w, &q8, in_f);
                let rel = tier_rel(got, want);
                worst = worst.max(rel);
                assert!(
                    rel < TIER_TOL,
                    "{name} {tier}: {got} vs scalar {want} (rel {rel:.3e} > {TIER_TOL:.0e})"
                );
            }
        }
        eprintln!("simd tier worst relative deviation from scalar: {worst:.3e}");
    }

    /// Multi-row (`_batch`, Q8 super-block activation) int8 dots: same tiers, plus the AVX-512
    /// VNNI kernels, which are SEPARATE implementations the dispatch prefers when available.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_batch_tiers_match_the_scalar_kernel() {
        type Bat = fn(&[u8], &[Q8], usize, &mut [f32]);
        /// (tier label, supported by this CPU, kernel).
        type Tier<'a> = (&'a str, bool, Bat);
        /// (label, weight dtype, scalar kernel, the SIMD tiers to hold to it).
        type Case<'a> = (&'a str, DType, Bat, &'a [Tier<'a>]);
        let has2 = is_x86_feature_detected!("avx2");
        let has5 = is_x86_feature_detected!("avx512bw");
        let hasv = has5 && is_x86_feature_detected!("avx512vnni");
        let in_f = 512usize;
        // m = 3: odd, so the row-pair/row-8 groupings inside the batch kernels hit their tails.
        let q8s: Vec<Q8> = (0..3)
            .map(|r| quantize_q8(&det_x(in_f, 0xB0B + r as u64)))
            .collect();
        // (label, dtype, scalar, [(tier, supported, kernel)])
        let cases: &[Case] = &[
            (
                "q4k_batch",
                DType::Q4K,
                vec_dot_q4k_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q4k_batch_avx2(r, q, n, o)
                    }),
                    ("avx512bw", has5, |r, q, n, o| unsafe {
                        vec_dot_q4k_batch_avx512bw(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q4k_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q2k_batch",
                DType::Q2K,
                vec_dot_q2k_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q2k_batch_avx2(r, q, n, o)
                    }),
                    ("avx512bw", has5, |r, q, n, o| unsafe {
                        vec_dot_q2k_batch_avx512bw(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q2k_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q3k_batch",
                DType::Q3K,
                vec_dot_q3k_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q3k_batch_avx2(r, q, n, o)
                    }),
                    ("avx512bw", has5, |r, q, n, o| unsafe {
                        vec_dot_q3k_batch_avx512bw(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q3k_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q5k_batch",
                DType::Q5K,
                vec_dot_q5k_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q5k_batch_avx2(r, q, n, o)
                    }),
                    ("avx512bw", has5, |r, q, n, o| unsafe {
                        vec_dot_q5k_batch_avx512bw(r, q, n, o)
                    }),
                ],
            ),
            (
                "q6k_batch",
                DType::Q6K,
                vec_dot_q6k_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q6k_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q6k_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q8_0_batch",
                DType::Q8_0,
                vec_dot_q8_0_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q8_0_batch_avx2(r, q, n, o)
                    }),
                    ("avx512bw", has5, |r, q, n, o| unsafe {
                        vec_dot_q8_0_batch_avx512bw(r, q, n, o)
                    }),
                ],
            ),
            (
                "iq4xs_batch",
                DType::Iq4Xs,
                vec_dot_iq4xs_batch_scalar,
                &[("avx2", has2, |r, q, n, o| unsafe {
                    vec_dot_iq4xs_batch_avx2(r, q, n, o)
                })],
            ),
        ];
        let m = q8s.len();
        let mut worst = 0f32;
        for &(name, dtype, scalar, tiers) in cases {
            let w = infr_testkit::synth_weight(dtype, in_f, 0x5EED);
            let mut want = vec![0f32; m];
            scalar(&w, &q8s, in_f, &mut want);
            assert!(
                want.iter().any(|v| v.abs() > 1e-3),
                "{name}: scalar result is ~0, case is vacuous"
            );
            for &(tier, on, f) in tiers {
                if !on {
                    eprintln!("skip {name} {tier}: not supported by this CPU");
                    continue;
                }
                let mut got = vec![0f32; m];
                f(&w, &q8s, in_f, &mut got);
                for (r, (&g, &v)) in got.iter().zip(&want).enumerate() {
                    let rel = tier_rel(g, v);
                    worst = worst.max(rel);
                    assert!(
                        rel < TIER_TOL,
                        "{name} {tier} row {r}: {g} vs scalar {v} (rel {rel:.3e} > {TIER_TOL:.0e})"
                    );
                }
            }
        }
        eprintln!("simd batch tier worst relative deviation from scalar: {worst:.3e}");
    }

    /// The NATIVE 32-block family (`_32_batch` / `Q8x32` activations): the legacy round quants and
    /// the 4-bit float formats, whose only SIMD tiers are avx2 and VNNI.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_blk32_tiers_match_the_scalar_kernel() {
        type Bat32 = fn(&[u8], &[Q8x32], usize, &mut [f32]);
        /// (tier label, supported by this CPU, kernel).
        type Tier32<'a> = (&'a str, bool, Bat32);
        /// (label, weight dtype, scalar kernel, the SIMD tiers to hold to it).
        type Case32<'a> = (&'a str, DType, Bat32, &'a [Tier32<'a>]);
        let has2 = is_x86_feature_detected!("avx2");
        let hasv = is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni");
        let has5 = is_x86_feature_detected!("avx512bw");
        let in_f = 512usize;
        let q8s: Vec<Q8x32> = (0..3)
            .map(|r| quantize_q8_32(&det_x(in_f, 0xC0DE + r as u64)))
            .collect();
        let cases: &[Case32] = &[
            (
                "q8_0_32",
                DType::Q8_0,
                vec_dot_q8_0_32_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q8_0_32_batch_avx2(r, q, n, o)
                    }),
                    ("avx512bw", has5, |r, q, n, o| unsafe {
                        vec_dot_q8_0_32_batch_avx512bw(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q8_0_32_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q5_0_32",
                DType::Q5_0,
                vec_dot_q5_0_32_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q5_0_32_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q5_0_32_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q4_0_32",
                DType::Q4_0,
                vec_dot_q4_0_32_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q4_0_32_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q4_0_32_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q4_1_32",
                DType::Q4_1,
                vec_dot_q4_1_32_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q4_1_32_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q4_1_32_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q5_1_32",
                DType::Q5_1,
                vec_dot_q5_1_32_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q5_1_32_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q5_1_32_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "q2_0",
                DType::Q2_0,
                vec_dot_q2_0_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_q2_0_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_q2_0_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "iq4nl_32",
                DType::Iq4Nl,
                vec_dot_iq4nl_32_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_iq4nl_32_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_iq4nl_32_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "mxfp4_32",
                DType::Mxfp4,
                vec_dot_mxfp4_32_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_mxfp4_32_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_mxfp4_32_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
            (
                "nvfp4",
                DType::Nvfp4,
                vec_dot_nvfp4_batch_scalar,
                &[
                    ("avx2", has2, |r, q, n, o| unsafe {
                        vec_dot_nvfp4_batch_avx2(r, q, n, o)
                    }),
                    ("vnni", hasv, |r, q, n, o| unsafe {
                        vec_dot_nvfp4_batch_vnni(r, q, n, o)
                    }),
                ],
            ),
        ];
        let m = q8s.len();
        let mut worst = 0f32;
        for &(name, dtype, scalar, tiers) in cases {
            let w = infr_testkit::synth_weight(dtype, in_f, 0x5EED);
            let mut want = vec![0f32; m];
            scalar(&w, &q8s, in_f, &mut want);
            assert!(
                want.iter().any(|v| v.abs() > 1e-3),
                "{name}: scalar result is ~0, case is vacuous"
            );
            for &(tier, on, f) in tiers {
                if !on {
                    eprintln!("skip {name} {tier}: not supported by this CPU");
                    continue;
                }
                let mut got = vec![0f32; m];
                f(&w, &q8s, in_f, &mut got);
                for (r, (&g, &v)) in got.iter().zip(&want).enumerate() {
                    let rel = tier_rel(g, v);
                    worst = worst.max(rel);
                    assert!(
                        rel < TIER_TOL,
                        "{name} {tier} row {r}: {g} vs scalar {v} (rel {rel:.3e} > {TIER_TOL:.0e})"
                    );
                }
            }
        }
        eprintln!("simd blk32 tier worst relative deviation from scalar: {worst:.3e}");
    }
}
