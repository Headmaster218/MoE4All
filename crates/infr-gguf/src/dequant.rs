//! Pure dequantization kernels — DType + bytes → f32. No GGUF parsing, no weight loading.
//! Extracted from `infr-llama::quant` so that `infr-cpu` can depend on `infr-gguf` without
//! creating a cycle through `infr-llama`.
#![allow(clippy::needless_range_loop)]
use anyhow::{bail, Result};

/// Dequantize a tensor's raw `bytes` of `dtype` into host f32. Handles plain floats
/// (F32/F16/BF16), affine quants (via [`dequant_factored`]), and codebook quants (via
/// [`dequant_codebook`]). The single host-side dequant entry point.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn dequant_block(dtype: infr_core::DType, bytes: &[u8]) -> Result<Vec<f32>> {
    use infr_core::DType::*;
    Ok(match dtype {
        F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        // Affine quants: decode once into the compact factored form, then expand in a
        // single fused pass — hoist the f16 super-scale `dd.to_f32()` out to once per
        // `dblk` and the `(sc, m)` multipliers to once per 16-element block, writing
        // straight into the output. Bit-identical to the old
        // factored→unified(materialize sc/mn)→expand chain: the arithmetic
        // `(dd·sc)·code + (dd·m)` and its evaluation order are unchanged, only the
        // redundant per-element f16→f32 reconversions and the two numel-sized `sc`/`mn`
        // allocations are removed.
        d if is_quant(d) => {
            let f = dequant_factored(d, bytes)?;
            let mut out = vec![0f32; f.codes.len()];
            let n16 = out.len() / 16;
            for b16 in 0..n16 {
                let base = b16 * 16;
                let db = base / f.dblk; // dblk is a multiple of 16 → constant over the block
                let dsc = f.dd[2 * db].to_f32();
                let dmn = f.dd[2 * db + 1].to_f32();
                let sc = dsc * f.scm[2 * b16] as f32;
                let mn = dmn * f.scm[2 * b16 + 1] as f32;
                for g in 0..16 {
                    out[base + g] = sc * f.codes[base + g] as f32 + mn;
                }
            }
            out
        }
        d if is_codebook_quant(d) => dequant_codebook(d, bytes),
        other => bail!("unsupported dtype {other:?} (host dequant wants F16/F32/BF16/quant)"),
    })
}

/// f32 → f16, saturating to ±65504 (f16 max) instead of overflowing to ±inf. Preserves NaN. Used
/// when down-converting bf16/f32 weights for the f16 fused path so a large-magnitude weight clips
/// to the largest finite f16 rather than corrupting the matmul with inf/NaN.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn f32_to_f16_sat(x: f32) -> half::f16 {
    const F16_MAX: f32 = 65504.0;
    if x.is_nan() {
        half::f16::NAN
    } else {
        half::f16::from_f32(x.clamp(-F16_MAX, F16_MAX))
    }
}

// per-call leaf, too small to probe (see docs/perf.md)
#[cfg_attr(infr_profile, infr_prof::skip)]
pub fn rdf16(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}
// per-call leaf, too small to probe (see docs/perf.md)
#[cfg_attr(infr_profile, infr_prof::skip)]
pub fn k4(j: usize, q: &[u8]) -> (u32, u32) {
    // get_scale_min_k4: extract 6-bit scale `d` and min `m` for sub-block j (0..8) from scales[12]
    if j < 4 {
        ((q[j] & 63) as u32, (q[j + 4] & 63) as u32)
    } else {
        (
            ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)) as u32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as u32,
        )
    }
}

/// A quant tensor in FACTORED unified form: `weight[g] = (d·sc)·code + (dmin·m)`, with one u8
/// `code` per element, one `(sc, m)` i16 pair per consecutive 16-element block, and one `(d, dmin)`
/// f16 pair per `dblk` elements (32 for the legacy formats, 256 for K-quants) — the two-level
/// scale structure every affine GGUF quant actually has. Recomputing `f32(d) * f32(sc)` yields
/// [`dequant_block`]'s per-block f32 scale bit-for-bit (it is the same f32 multiply; sign flips
/// and the ×4/×32 factors folded into `m` are exact power-of-two scalings), so a consumer keeping
/// this compact form reconstructs the exact dequant reference while reading ~2 bits/elem of scale
/// metadata instead of 64.
pub struct Factored {
    /// Per-element quant index.
    pub codes: Vec<u8>,
    /// Per-16-element-block `(sc, m)` multipliers, interleaved `[sc0, m0, sc1, m1, ..]`.
    pub scm: Vec<i16>,
    /// Per-`dblk` `(d, dmin)` f16 super scales, interleaved `[d0, dmin0, d1, dmin1, ..]`.
    pub dd: Vec<half::f16>,
    /// Elements per `(d, dmin)` pair.
    pub dblk: usize,
}

// per-call leaf, too small to probe (see docs/perf.md)
#[cfg_attr(infr_profile, infr_prof::skip)]
fn rf16(b: &[u8]) -> half::f16 {
    half::f16::from_le_bytes([b[0], b[1]])
}

/// Decode any affine quant into [`Factored`] form. Each arm mirrors the llama.cpp reference
/// dequant (see the per-format comments), but instead of expanding to per-element f32 scale/min it
/// records the format's own structure: the f16 super scale(s) per block and the small integer
/// multipliers per 16-element sub-block.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn dequant_factored(dtype: infr_core::DType, bytes: &[u8]) -> Result<Factored> {
    use infr_core::DType::*;
    // Only affine quants have a factored form; a non-affine dtype (F16/Iq*/Tq*/fp4)
    // must error, never panic — callers can pass an untrusted-model dtype here.
    // Which formats have a factored form is the decision here; the block GEOMETRY comes from the
    // shared decode spec (`infr_core::decode_spec::block_layout`) rather than a second hand-kept
    // `(elems, bytes)` table that could drift from the loader's.
    match dtype {
        Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2K | Q3K | Q4K | Q5K | Q6K => {}
        other => bail!("dequant_factored: unsupported (non-affine) dtype {other:?}"),
    }
    let (qpb, bpb) = infr_core::decode_spec::block_layout(dtype);
    let nblk = bytes.len() / bpb;
    let numel = nblk * qpb;
    let dblk = qpb; // every affine format's (d, dmin) covers exactly one quant block
    let mut codes = vec![0u8; numel];
    let mut scm = vec![0i16; numel / 16 * 2];
    let mut dd = vec![half::f16::ZERO; nblk * 2];
    for b in 0..nblk {
        let blk = &bytes[b * bpb..(b + 1) * bpb];
        // (sc, m) for the two/sixteen 16-blocks of this quant block, written via `s16`.
        let mut s16 = |b16: usize, sc: i16, m: i16| {
            scm[2 * b16] = sc;
            scm[2 * b16 + 1] = m;
        };
        match dtype {
            // ── Q4_0: y = d*(q4 - 8), q4 ∈ 0..15 ──────────────────────────────
            // Ref: llama.cpp dequantize_row_q4_0 (ggml-quants.c l.401)
            // Block: [half d][uint8 qs[16]] → scale = d·1, min = d·(-8)
            Q4_0 => {
                let d = rf16(blk);
                dd[2 * b] = d;
                dd[2 * b + 1] = d;
                s16(b * 2, 1, -8);
                s16(b * 2 + 1, 1, -8);
                let qs = &blk[2..18];
                for j in 0..16 {
                    codes[b * 32 + j] = qs[j] & 0x0F;
                    codes[b * 32 + j + 16] = qs[j] >> 4;
                }
            }
            // ── Q4_1: y = d*q4 + m ─────────────────────────────────────────────
            // Ref: llama.cpp dequantize_row_q4_1 (ggml-quants.c l.421)
            // Block: [half d][half m][uint8 qs[16]] → scale = d·1, min = m·1
            Q4_1 => {
                let d = rf16(blk);
                let m = rf16(&blk[2..4]);
                dd[2 * b] = d;
                dd[2 * b + 1] = m;
                s16(b * 2, 1, 1);
                s16(b * 2 + 1, 1, 1);
                let qs = &blk[4..20];
                for j in 0..16 {
                    codes[b * 32 + j] = qs[j] & 0x0F;
                    codes[b * 32 + j + 16] = qs[j] >> 4;
                }
            }
            // ── Q5_0: y = d*(q5 - 16), q5 ∈ 0..31 ──────────────────────────────
            // Ref: llama.cpp dequantize_row_q5_0 (ggml-quants.c l.442)
            // Block: [half d][uint8 qh[4]][uint8 qs[16]] → scale = d·1, min = d·(-16)
            Q5_0 => {
                let d = rf16(blk);
                dd[2 * b] = d;
                dd[2 * b + 1] = d;
                s16(b * 2, 1, -16);
                s16(b * 2 + 1, 1, -16);
                let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                let qs = &blk[6..22];
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    codes[b * 32 + j] = ((qs[j] as u32 & 0x0F) | xh0) as u8;
                    codes[b * 32 + j + 16] = ((qs[j] as u32 >> 4) | xh1) as u8;
                }
            }
            // ── Q5_1: y = d*q5 + m ─────────────────────────────────────────────
            // Ref: llama.cpp dequantize_row_q5_1 (ggml-quants.c l.468)
            // Block: [half d][half m][uint8 qh[4]][uint8 qs[16]] → scale = d·1, min = m·1
            Q5_1 => {
                let d = rf16(blk);
                let m = rf16(&blk[2..4]);
                dd[2 * b] = d;
                dd[2 * b + 1] = m;
                s16(b * 2, 1, 1);
                s16(b * 2 + 1, 1, 1);
                let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
                let qs = &blk[8..24];
                for j in 0..16 {
                    let xh0 = ((qh >> j) << 4) & 0x10;
                    let xh1 = (qh >> (j + 12)) & 0x10;
                    codes[b * 32 + j] = ((qs[j] as u32 & 0x0F) | xh0) as u8;
                    codes[b * 32 + j + 16] = ((qs[j] as u32 >> 4) | xh1) as u8;
                }
            }
            // ── Q8_0: y = d*q8, q8 i8 stored biased by +128 → scale = d·1, min = d·(-128)
            Q8_0 => {
                let d = rf16(blk);
                dd[2 * b] = d;
                dd[2 * b + 1] = d;
                s16(b * 2, 1, -128);
                s16(b * 2 + 1, 1, -128);
                for i in 0..32 {
                    codes[b * 32 + i] = (blk[2 + i] as i8 as i16 + 128) as u8;
                }
            }
            // ── Q2_K: y = d*(sc&0xF)*q2 - dmin*(sc>>4); per-16-elem sub-block scale/min ──
            // Ref: llama.cpp dequantize_row_q2_K (ggml-quants.c l.903)
            // Block (84 bytes): [uint8 scales[16]][uint8 qs[64]][half d][half dmin]
            Q2K => {
                let scales = &blk[0..16];
                let qs = &blk[16..80];
                dd[2 * b] = rf16(&blk[80..82]);
                dd[2 * b + 1] = rf16(&blk[82..84]);
                let base = b * 256;
                let mut out = 0usize;
                let mut is = 0usize;
                let mut qoff = 0usize;
                for _n in 0..2 {
                    let mut shift = 0u32;
                    for _j in 0..4 {
                        for half in 0..2 {
                            let sc = scales[is];
                            is += 1;
                            s16((base + out) / 16, (sc & 0xF) as i16, -((sc >> 4) as i16));
                            for l in 0..16 {
                                codes[base + out] = (qs[qoff + half * 16 + l] >> shift) & 3;
                                out += 1;
                            }
                        }
                        shift += 2;
                    }
                    qoff += 32;
                }
            }
            // ── Q3_K: y = d*(sc6-32)*(q3u - 4); per-16-elem 6-bit sub-block scale ──
            // Ref: llama.cpp dequantize_row_q3_K (ggml-quants.c l.1247)
            // Block (110 bytes): [uint8 hmask[32]][uint8 qs[64]][uint8 scales[12]][half d]
            // scale = d·(sc6-32), min = -4·scale = d·(-4·(sc6-32))
            Q3K => {
                let hmask = &blk[0..32];
                let qs = &blk[32..96];
                let scales_raw = &blk[96..108];
                dd[2 * b] = rf16(&blk[108..110]);
                dd[2 * b + 1] = dd[2 * b];
                let base = b * 256;

                // Decode 6-bit scales: port of the llama.cpp bit manipulation exactly.
                // scales_raw is 12 bytes encoding 16 × 6-bit values.
                // kmask1=0x03030303, kmask2=0x0f0f0f0f
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
                // aux as i8[16] gives decoded 6-bit scales; subtract 32 to get signed scale.
                let sc6 = |i: usize| -> i16 {
                    let sc = ((aux[i / 4] >> ((i % 4) * 8)) & 0xFF) as u8 as i8;
                    sc as i16 - 32
                };

                let mut out = 0usize;
                let mut is = 0usize;
                let mut qoff = 0usize;
                let mut m = 1u8;
                for _n in 0..2 {
                    let mut shift = 0u32;
                    for _j in 0..4 {
                        for half in 0..2 {
                            let s = sc6(is);
                            is += 1;
                            s16((base + out) / 16, s, -4 * s);
                            for l in 0..16 {
                                let low2 = (qs[qoff + half * 16 + l] >> shift) & 3;
                                let high = if hmask[half * 16 + l] & m != 0 {
                                    1u8
                                } else {
                                    0u8
                                };
                                codes[base + out] = low2 | (high << 2); // 0..7
                                out += 1;
                            }
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    qoff += 32;
                }
            }
            // ── Q4_K: y = d*sc6·q4 - dmin*m6; 6-bit scale/min per 32-elem sub-block ──
            // Ref: llama.cpp dequantize_row_q4_K (via get_scale_min_k4)
            Q4K => {
                dd[2 * b] = rf16(&blk[0..2]);
                dd[2 * b + 1] = rf16(&blk[2..4]);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let base = b * 256;
                for j in 0..4 {
                    let (sc1, m1) = k4(2 * j, scales);
                    let (sc2, m2) = k4(2 * j + 1, scales);
                    let b16 = b * 16 + j * 4;
                    s16(b16, sc1 as i16, -(m1 as i16));
                    s16(b16 + 1, sc1 as i16, -(m1 as i16));
                    s16(b16 + 2, sc2 as i16, -(m2 as i16));
                    s16(b16 + 3, sc2 as i16, -(m2 as i16));
                    for l in 0..32 {
                        let v = qs[j * 32 + l];
                        codes[base + j * 64 + l] = v & 0xF;
                        codes[base + j * 64 + 32 + l] = v >> 4;
                    }
                }
            }
            // ── Q5_K: like Q4_K with a 5th code bit from qh ──
            Q5K => {
                dd[2 * b] = rf16(&blk[0..2]);
                dd[2 * b + 1] = rf16(&blk[2..4]);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let base = b * 256;
                let (mut u1, mut u2) = (1u8, 2u8);
                for j in 0..4 {
                    let (sc1, m1) = k4(2 * j, scales);
                    let (sc2, m2) = k4(2 * j + 1, scales);
                    let b16 = b * 16 + j * 4;
                    s16(b16, sc1 as i16, -(m1 as i16));
                    s16(b16 + 1, sc1 as i16, -(m1 as i16));
                    s16(b16 + 2, sc2 as i16, -(m2 as i16));
                    s16(b16 + 3, sc2 as i16, -(m2 as i16));
                    for l in 0..32 {
                        let v = qs[j * 32 + l];
                        codes[base + j * 64 + l] = (v & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
                        codes[base + j * 64 + 32 + l] =
                            (v >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            // ── Q6_K: y = d*sc_i8*(q6 - 32); i8 scale per 16-elem sub-block ──
            // scale = d·sc, min = -32·scale = d·(-32·sc); -32·sc ∈ [-4064, 4096] needs i16.
            Q6K => {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let scales = &blk[192..208]; // 16 × int8
                dd[2 * b] = rf16(&blk[208..210]);
                dd[2 * b + 1] = dd[2 * b];
                for half in 0..2 {
                    let (qlo, qho, sco, base) =
                        (half * 64, half * 32, half * 8, b * 256 + half * 128);
                    for sci in [0usize, 2, 4, 6] {
                        for is in 0..2usize {
                            let s = scales[sco + is + sci] as i8 as i16;
                            s16(base / 16 + sci + is, s, -32 * s);
                        }
                    }
                    for l in 0..32 {
                        codes[base + l] = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                        codes[base + l + 32] =
                            (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                        codes[base + l + 64] = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                        codes[base + l + 96] =
                            (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(Factored {
        codes,
        scm,
        dd,
        dblk,
    })
}

// ─── codebook (non-affine) dequant ───────────────────────────────────────────

/// IQ1_S / IQ1_M delta offset applied to each grid element.
/// Ref: llama.cpp ggml-common.h l.1121: `#define IQ1S_DELTA 0.125f`
pub const IQ1S_DELTA: f32 = 0.125;

/// MXFP4 / NVFP4 signed 4-bit codebook (E2M1 format × 2).
/// Ref: llama.cpp ggml-common.h l.1116: `kvalues_mxfp4`
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Decode an E8M0 exponent byte to float32, halved (= 2^(x-128)).
/// Matches llama.cpp `ggml_e8m0_to_fp32_half` (ggml-impl.h l.477).
#[inline(always)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn e8m0_to_fp32_half(x: u8) -> f32 {
    if x < 2 {
        f32::from_bits(0x0020_0000u32 << x)
    } else {
        f32::from_bits(((x as u32) - 1) << 23)
    }
}

/// Decode a UE4M3 byte (unsigned, 4 exp bits bias=7, 3 mantissa bits) to float32, halved.
/// Matches llama.cpp `ggml_ue4m3_to_fp32` (ggml-impl.h l.502).
#[inline(always)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 {
        man * f32::powi(2.0, -9)
    } else {
        (1.0 + man / 8.0) * f32::powi(2.0, exp - 7)
    };
    raw * 0.5
}

/// IQ4_NL / IQ4_XS 16-entry signed-integer codebook.
/// Ref: llama.cpp ggml-common.h `kvalues_iq4nl` (l.1110)
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// True for codebook quants (IQ*/TQ*/fp4) that go host-dequant → f16, NOT the GPU affine path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn is_codebook_quant(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        d,
        Iq1S | Iq1M
            | Iq2Xxs
            | Iq2Xs
            | Iq2S
            | Iq3Xxs
            | Iq3S
            | Iq4Nl
            | Iq4Xs
            | Tq1_0
            | Tq2_0
            | Q2_0
            | Mxfp4
            | Nvfp4
            | I2S
    )
}

/// Shared IQ sign+scale expansion: for `n` elements, decode signed byte `j` from the
/// packed `grid`, negate it per `KMASK_IQ2XS[sign_off + j]`, scale by `db`, and write
/// into `out[..n]`. Factors the sign-application block that was copy-pasted across
/// IQ2_XXS / IQ2_XS / IQ2_S (n=8, sign_off=0) and IQ3_XXS / IQ3_S (two n=4 halves,
/// sign_off 0 and 4). The multiply order `db * gv * sign` is unchanged from the inlined
/// versions.
#[inline]
fn apply_signs(out: &mut [f32], grid: u64, signs: u8, sign_off: usize, n: usize, db: f32) {
    use infr_core::iquant_grids::KMASK_IQ2XS;
    for j in 0..n {
        let gv = ((grid >> (8 * j)) & 0xFF) as i8 as f32;
        let sign = if signs & KMASK_IQ2XS[sign_off + j] != 0 {
            -1.0
        } else {
            1.0
        };
        out[j] = db * gv * sign;
    }
}

/// Dequantize a codebook (non-affine) quant to f32. Ported from llama.cpp `ggml-quants.c`.
/// Returns a `Vec<f32>` of length `numel` in natural tensor order.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn dequant_codebook(dtype: infr_core::DType, bytes: &[u8]) -> Vec<f32> {
    use infr_core::DType::*;
    match dtype {
        // ── IQ4_NL: y = d * kvalues_iq4nl[q4], QK4_NL=32 ───────────────────────
        // Block: [half d][uint8 qs[16]], 18 bytes
        // Ref: llama.cpp dequantize_row_iq4_nl (ggml-quants.c l.2653)
        Iq4Nl => {
            // Ref: llama.cpp dequantize_row_iq4_nl (ggml-quants.c l.2653)
            // y[j]    = d * kv[qs[j] & 0xF]  for j in 0..16 → elements  0..15
            // y[j+16] = d * kv[qs[j] >>  4]  for j in 0..16 → elements 16..31
            let bpb = 18usize;
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 32];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs = &blk[2..18];
                let base = b * 32;
                for j in 0..16 {
                    out[base + j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                    out[base + j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
            out
        }
        // ── IQ4_XS: y = d*(ls-32) * kvalues_iq4nl[q4], QK_K=256 ────────────────
        // Block: [half d][uint16 scales_h][uint8 scales_l[4]][uint8 qs[128]], 136 bytes
        // Ref: llama.cpp dequantize_row_iq4_xs (ggml-quants.c l.2671)
        Iq4Xs => {
            // Ref: llama.cpp dequantize_row_iq4_xs (ggml-quants.c l.2671)
            // Block: [half d][uint16 scales_h][uint8 scales_l[4]][uint8 qs[128]], 136 bytes
            // 8 sub-blocks of 32 elements each; ls = 6-bit scale per sub-block
            // y[j+0]  = dl * kv[qs[j] & 0xF] for j in 0..16 → elements  0..15 of sub-block
            // y[j+16] = dl * kv[qs[j] >>  4] for j in 0..16 → elements 16..31 of sub-block
            let bpb = 136usize;
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let scales_h = u16::from_le_bytes(blk[2..4].try_into().unwrap());
                let scales_l = &blk[4..8];
                let qs = &blk[8..136];
                let base = b * 256;
                let mut qoff = 0usize;
                let mut outoff = 0usize;
                for ib in 0..8usize {
                    // 6-bit ls: lower 4 bits from scales_l, upper 2 bits from scales_h
                    let lo = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as u32;
                    let hi = ((scales_h >> (2 * ib)) & 3) as u32;
                    let ls = lo | (hi << 4);
                    let dl = d * (ls as i32 - 32) as f32;
                    for j in 0..16 {
                        out[base + outoff + j] =
                            dl * KVALUES_IQ4NL[(qs[qoff + j] & 0xF) as usize] as f32;
                        out[base + outoff + j + 16] =
                            dl * KVALUES_IQ4NL[(qs[qoff + j] >> 4) as usize] as f32;
                    }
                    qoff += 16;
                    outoff += 32;
                }
            }
            out
        }
        // ── IQ2_XXS: block = [half d][uint16 qs[32]], 66 bytes, QK_K=256 ────────
        // Each super-block has 8 sub-blocks of 32 elements.  For each sub-block:
        //   aux32[0] = 4 grid indices (one byte each, into iq2xxs_grid[256])
        //   aux32[1] = sign pack: bits[6:0]*4 = four 7-bit sign indices + bits[31:28] = scale
        //   db = d * (0.5 + (aux32[1] >> 28)) * 0.25
        //   grid[j] = ((iq2xxs_grid[idx] >> (8*j)) & 0xFF) as i8  (for j in 0..8)
        //   y[j] = db * grid[j] * (if ksigns[sign_idx] & (1<<j) { -1 } else { 1 })
        // Ref: llama.cpp dequantize_row_iq2_xxs (ggml-quants.c l.2416)
        Iq2Xxs => {
            use infr_core::iquant_grids::{IQ2XXS_GRID, KSIGNS_IQ2XS};
            let bpb = 66usize; // 2 + 32*2
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                // qs as raw bytes (64 bytes = 32 uint16s)
                let qs = &blk[2..66];
                let base = b * 256;
                let mut outoff = 0usize;
                for ib32 in 0..8usize {
                    // read 8 bytes at offset 8*ib32 within qs
                    let off = ib32 * 8;
                    let aux0 = u32::from_le_bytes(qs[off..off + 4].try_into().unwrap());
                    let aux1 = u32::from_le_bytes(qs[off + 4..off + 8].try_into().unwrap());
                    let scale_mag = aux1 >> 28;
                    let db = d * (0.5 + scale_mag as f32) * 0.25;
                    let aux0_bytes = aux0.to_le_bytes();
                    for l in 0..4usize {
                        let grid_idx = aux0_bytes[l] as usize;
                        let sign_idx = ((aux1 >> (7 * l)) & 127) as usize;
                        let grid_u64 = IQ2XXS_GRID[grid_idx];
                        let signs = KSIGNS_IQ2XS[sign_idx];
                        apply_signs(&mut out[base + outoff..], grid_u64, signs, 0, 8, db);
                        outoff += 8;
                    }
                }
            }
            out
        }
        // ── IQ2_XS: block = [half d][uint16 qs[32]][uint8 scales[8]], 74 bytes ──
        // 8 sub-blocks of 32 elements each (4 groups of 8 per sub-block).
        //   db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25
        //   db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25
        //   For l in 0..4: grid = iq2xs_grid[qs16[l] & 511]  (9-bit index)
        //                  signs = ksigns[qs16[l] >> 9]        (7-bit sign)
        //                  dl = db[l/2]
        // Ref: llama.cpp dequantize_row_iq2_xs (ggml-quants.c l.2444)
        Iq2Xs => {
            use infr_core::iquant_grids::{IQ2XS_GRID, KSIGNS_IQ2XS};
            let bpb = 74usize; // 2 + 32*2 + 8
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                // qs as uint16 array (64 bytes = 32 entries)
                let qs_raw = &blk[2..66];
                let scales = &blk[66..74];
                let base = b * 256;
                let mut outoff = 0usize;
                for ib32 in 0..8usize {
                    let sc = scales[ib32];
                    let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
                    let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
                    for l in 0..4usize {
                        let qoff = (ib32 * 4 + l) * 2;
                        let qs16 = u16::from_le_bytes(qs_raw[qoff..qoff + 2].try_into().unwrap());
                        let grid_idx = (qs16 & 511) as usize;
                        let sign_idx = (qs16 >> 9) as usize;
                        let grid_u64 = IQ2XS_GRID[grid_idx];
                        let signs = KSIGNS_IQ2XS[sign_idx];
                        let dl = if l < 2 { db0 } else { db1 };
                        apply_signs(&mut out[base + outoff..], grid_u64, signs, 0, 8, dl);
                        outoff += 8;
                    }
                }
            }
            out
        }
        // ── IQ2_S: block = [half d][u8 qs[64]][u8 qh[8]][u8 scales[8]], 82 bytes ─
        // First 32 bytes of qs = 8-bit grid indices (low bits);
        // next 32 bytes = per-group sign bytes. qh = 2-bit high bits per entry.
        //   grid_idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)  (10-bit → iq2s_grid[1024])
        // Ref: llama.cpp dequantize_row_iq2_s (ggml-quants.c l.2471)
        Iq2S => {
            use infr_core::iquant_grids::IQ2S_GRID;
            let bpb = 82usize; // 2 + 64 + 8 + 8
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs_all = &blk[2..66]; // 64 bytes
                let qh = &blk[66..74]; // 8 bytes
                let scales = &blk[74..82]; // 8 bytes
                let base = b * 256;
                let mut outoff = 0usize;
                for ib32 in 0..8usize {
                    let sc = scales[ib32];
                    let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
                    let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
                    let qh_byte = qh[ib32];
                    for l in 0..4usize {
                        let qs_idx = ib32 * 4 + l;
                        let sgn_idx = ib32 * 4 + l + 32; // signs start at qs_all[32]
                        let qs_byte = qs_all[qs_idx];
                        let sign_byte = qs_all[sgn_idx];
                        // high 2 bits: shift = 8 - 2*l; mask 0x300
                        let hi = ((qh_byte as u32).wrapping_shl((8 - 2 * l) as u32)) & 0x300;
                        let grid_idx = (qs_byte as usize) | (hi as usize);
                        let grid_u64 = IQ2S_GRID[grid_idx];
                        let dl = if l < 2 { db0 } else { db1 };
                        apply_signs(&mut out[base + outoff..], grid_u64, sign_byte, 0, 8, dl);
                        outoff += 8;
                    }
                }
            }
            out
        }
        // ── IQ3_XXS: block = [half d][u8 qs[96]], 98 bytes, QK_K=256 ─────────────
        // qs[0..64] = 8-bit indices (two per group of 8: grid1, grid2)
        // qs[64..96] = scales_and_signs: 4 bytes per sub-block (aux32)
        //   db = d * (0.5 + (aux32 >> 28)) * 0.5
        //   For l in 0..4: signs = ksigns[(aux32 >> 7*l) & 127]
        //     grid1 = iq3xxs_grid[qs[2*l+0]];  grid2 = iq3xxs_grid[qs[2*l+1]]
        //     y[j+0] = db * grid1[j] * sign;  y[j+4] = db * grid2[j] * sign
        // Ref: llama.cpp dequantize_row_iq3_xxs (ggml-quants.c l.2503)
        Iq3Xxs => {
            use infr_core::iquant_grids::{IQ3XXS_GRID, KSIGNS_IQ2XS};
            let bpb = 98usize; // 2 + 96
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs = &blk[2..66]; // first 64 bytes = grid indices
                let sas = &blk[66..98]; // scales_and_signs (32 bytes = 8 × u32)
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                for ib32 in 0..8usize {
                    let aux32 = u32::from_le_bytes(sas[4 * ib32..4 * ib32 + 4].try_into().unwrap());
                    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                    for l in 0..4usize {
                        let sign_idx = ((aux32 >> (7 * l)) & 127) as usize;
                        let signs = KSIGNS_IQ2XS[sign_idx];
                        let g1 = IQ3XXS_GRID[qs[qs_off + 2 * l] as usize];
                        let g2 = IQ3XXS_GRID[qs[qs_off + 2 * l + 1] as usize];
                        apply_signs(&mut out[base + outoff..], g1 as u64, signs, 0, 4, db);
                        apply_signs(&mut out[base + outoff + 4..], g2 as u64, signs, 4, 4, db);
                        outoff += 8;
                    }
                    qs_off += 8;
                }
            }
            out
        }
        // ── IQ3_S: block = [half d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]], 110 bytes
        // Outer loop steps ib32 in 0,2,4,6 (pairs → two groups of 32 each = 64 elements/iter).
        //   db1 = d * (1 + 2*(scales[ib32/2] & 0xf))
        //   db2 = d * (1 + 2*(scales[ib32/2] >> 4))
        //   For first group (l in 0..4, using qh[0], db1):
        //     grid1 = iq3s_grid[qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)]
        //     grid2 = iq3s_grid[qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)]
        //   For second group (l in 0..4, using qh[1], db2): similarly
        // Ref: llama.cpp dequantize_row_iq3_s (ggml-quants.c l.2535)
        Iq3S => {
            use infr_core::iquant_grids::IQ3S_GRID;
            let bpb = 110usize; // 2 + 64 + 8 + 32 + 4
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs_arr = &blk[2..66]; // 64 bytes
                let qh_arr = &blk[66..74]; // 8 bytes
                let signs_arr = &blk[74..106]; // 32 bytes
                let scales = &blk[106..110]; // 4 bytes
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                let mut signs_off = 0usize;
                let mut qh_off = 0usize;
                // outer loop: ib32 steps 0,2,4,6 → 4 pairs
                for pair in 0..4usize {
                    let db1 = d * (1.0 + 2.0 * (scales[pair] & 0xf) as f32);
                    let db2 = d * (1.0 + 2.0 * (scales[pair] >> 4) as f32);
                    // first group of 32 elements (using qh[qh_off], db1)
                    let qh0 = qh_arr[qh_off];
                    for l in 0..4usize {
                        let g1_idx = qs_arr[qs_off + 2 * l] as usize
                            | (((qh0 as u32).wrapping_shl((8 - 2 * l) as u32)) & 256) as usize;
                        let g2_idx = qs_arr[qs_off + 2 * l + 1] as usize
                            | (((qh0 as u32).wrapping_shl((7 - 2 * l) as u32)) & 256) as usize;
                        let g1 = IQ3S_GRID[g1_idx];
                        let g2 = IQ3S_GRID[g2_idx];
                        let sb = signs_arr[signs_off + l];
                        apply_signs(&mut out[base + outoff..], g1 as u64, sb, 0, 4, db1);
                        apply_signs(&mut out[base + outoff + 4..], g2 as u64, sb, 4, 4, db1);
                        outoff += 8;
                    }
                    qs_off += 8;
                    signs_off += 4;
                    // second group of 32 elements (using qh[qh_off+1], db2)
                    let qh1 = qh_arr[qh_off + 1];
                    for l in 0..4usize {
                        let g1_idx = qs_arr[qs_off + 2 * l] as usize
                            | (((qh1 as u32).wrapping_shl((8 - 2 * l) as u32)) & 256) as usize;
                        let g2_idx = qs_arr[qs_off + 2 * l + 1] as usize
                            | (((qh1 as u32).wrapping_shl((7 - 2 * l) as u32)) & 256) as usize;
                        let g1 = IQ3S_GRID[g1_idx];
                        let g2 = IQ3S_GRID[g2_idx];
                        let sb = signs_arr[signs_off + l];
                        apply_signs(&mut out[base + outoff..], g1 as u64, sb, 0, 4, db2);
                        apply_signs(&mut out[base + outoff + 4..], g2 as u64, sb, 4, 4, db2);
                        outoff += 8;
                    }
                    qh_off += 2;
                    qs_off += 8;
                    signs_off += 4;
                }
            }
            out
        }
        // ── IQ1_S: block = [half d][u8 qs[32]][u16 qh[8]], 50 bytes, QK_K=256 ────
        // 8 sub-blocks of 32 elements (4 groups of 8 each).
        //   dl = d * (2*((qh[ib] >> 12) & 7) + 1)
        //   delta = if qh[ib] & 0x8000 { -IQ1S_DELTA } else { IQ1S_DELTA }
        //   For l in 0..4: grid_idx = qs[l] | (((qh[ib] >> 3*l) & 7) << 8)
        //     y[j] = dl * (grid[j] as f32 + delta)
        // Ref: llama.cpp dequantize_row_iq1_s (ggml-quants.c l.2578)
        Iq1S => {
            use infr_core::iquant_grids::IQ1S_GRID;
            let bpb = 50usize; // 2 + 32 + 16
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs = &blk[2..34]; // 32 bytes
                let qh_raw = &blk[34..50]; // 16 bytes = 8 × u16
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                for ib in 0..8usize {
                    let qh = u16::from_le_bytes(qh_raw[2 * ib..2 * ib + 2].try_into().unwrap());
                    let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
                    let delta = if qh & 0x8000 != 0 {
                        -IQ1S_DELTA
                    } else {
                        IQ1S_DELTA
                    };
                    for l in 0..4usize {
                        let grid_idx =
                            qs[qs_off + l] as usize | (((qh >> (3 * l)) & 7) as usize) << 8;
                        let grid_u64 = IQ1S_GRID[grid_idx];
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8 as f32;
                            out[base + outoff + j] = dl * (gv + delta);
                        }
                        outoff += 8;
                    }
                    qs_off += 4;
                }
            }
            out
        }
        // ── IQ1_M: block = [u8 qs[32]][u8 qh[16]][u8 scales[8]], 56 bytes, QK_K=256
        // No separate `d` field — d is a f16 packed into the high 4 bits of each u16 of scales.
        //   sc[i] = scales reinterpreted as u16[4]; d extracted via:
        //   scale.u16 = (sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00) | (sc[3]&0xf000)
        //   dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1)
        //   dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1)
        //   idx[0..4] from qs[0..3] and qh[0..1]:
        //     idx[0] = qs[0] | ((qh[0] << 8) & 0x700);  delta[0] = qh[0]&0x08 ? neg : pos
        //     idx[1] = qs[1] | ((qh[0] << 4) & 0x700);  delta[1] = qh[0]&0x80 ? neg : pos
        //     idx[2] = qs[2] | ((qh[1] << 8) & 0x700);  delta[2] = qh[1]&0x08 ? neg : pos
        //     idx[3] = qs[3] | ((qh[1] << 4) & 0x700);  delta[3] = qh[1]&0x80 ? neg : pos
        //   l=0,1: y[j] = dl1*(grid[j]+delta[l]); l=2,3: y[j] = dl2*(grid[j]+delta[l])
        // Ref: llama.cpp dequantize_row_iq1_m (ggml-quants.c l.2603)
        Iq1M => {
            use infr_core::iquant_grids::IQ1S_GRID;
            let bpb = 56usize; // 32 + 16 + 8
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let qs_arr = &blk[0..32]; // 32 bytes
                let qh_arr = &blk[32..48]; // 16 bytes
                let scales_raw = &blk[48..56]; // 8 bytes = 4 × u16
                                               // Extract d from high-nibble packing of the 4 scale u16s
                let sc0 = u16::from_le_bytes(scales_raw[0..2].try_into().unwrap());
                let sc1 = u16::from_le_bytes(scales_raw[2..4].try_into().unwrap());
                let sc2 = u16::from_le_bytes(scales_raw[4..6].try_into().unwrap());
                let sc3 = u16::from_le_bytes(scales_raw[6..8].try_into().unwrap());
                let sc = [sc0, sc1, sc2, sc3];
                let d_bits: u16 =
                    (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
                let d = half::f16::from_bits(d_bits).to_f32();
                let base = b * 256;
                let mut outoff = 0usize;
                let mut qs_off = 0usize;
                let mut qh_off = 0usize;
                for ib in 0..8usize {
                    let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) as f32 + 1.0);
                    let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) as f32 + 1.0);
                    let qh0 = qh_arr[qh_off];
                    let qh1 = qh_arr[qh_off + 1];
                    let idx = [
                        qs_arr[qs_off] as usize | (((qh0 as usize) << 8) & 0x700),
                        qs_arr[qs_off + 1] as usize | (((qh0 as usize) << 4) & 0x700),
                        qs_arr[qs_off + 2] as usize | (((qh1 as usize) << 8) & 0x700),
                        qs_arr[qs_off + 3] as usize | (((qh1 as usize) << 4) & 0x700),
                    ];
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
                    // l=0,1: use dl1; l=2,3: use dl2
                    for l in 0..2usize {
                        let grid_u64 = IQ1S_GRID[idx[l]];
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8 as f32;
                            out[base + outoff + j] = dl1 * (gv + delta[l]);
                        }
                        outoff += 8;
                    }
                    for l in 2..4usize {
                        let grid_u64 = IQ1S_GRID[idx[l]];
                        for j in 0..8usize {
                            let gv = ((grid_u64 >> (8 * j)) & 0xFF) as i8 as f32;
                            out[base + outoff + j] = dl2 * (gv + delta[l]);
                        }
                        outoff += 8;
                    }
                    qs_off += 4;
                    qh_off += 2;
                }
            }
            out
        }
        // ── TQ1_0: block = [u8 qs[48]][u8 qh[4]][half d], 54 bytes, QK_K=256 ────
        // 5-ternary-digits-per-byte encoding.
        //   Main loop: j=0..31 (32 bytes), 5 passes → 32*5=160 elements
        //   Second:   j=32..47 (16 bytes), 5 passes → 16*5=80 elements
        //   qh loop:  4 bytes, 4 passes →  4*4=16 elements
        //   Total 256.
        //   digit_n(b) = ((b * pow3[n] as u16) * 3 >> 8) as i16 → 0,1, or 2
        //   y = (digit - 1) * d
        // Ref: llama.cpp dequantize_row_tq1_0 (ggml-quants.c l.2356)
        Tq1_0 => {
            let bpb = 54usize; // 48 + 4 + 2
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let qs = &blk[0..48];
                let qh = &blk[48..52];
                let d = rdf16(&blk[52..54]);
                let base = b * 256;
                let mut outoff = 0usize;
                // qs[0..32]: 32 bytes, 5 digit passes
                for n in 0..5usize {
                    let p3 = POW3[n];
                    for m in 0..32usize {
                        let q = qs[m].wrapping_mul(p3);
                        let xi = (((q as u16) * 3) >> 8) as i16;
                        out[base + outoff + n * 32 + m] = (xi - 1) as f32 * d;
                    }
                }
                outoff += 5 * 32; // 160
                                  // qs[32..48]: 16 bytes, 5 digit passes
                for n in 0..5usize {
                    let p3 = POW3[n];
                    for m in 0..16usize {
                        let q = qs[32 + m].wrapping_mul(p3);
                        let xi = (((q as u16) * 3) >> 8) as i16;
                        out[base + outoff + n * 16 + m] = (xi - 1) as f32 * d;
                    }
                }
                outoff += 5 * 16; // 80
                                  // qh[0..4]: 4 bytes, 4 digit passes
                for n in 0..4usize {
                    let p3 = POW3[n];
                    for j in 0..4usize {
                        let q = qh[j].wrapping_mul(p3);
                        let xi = (((q as u16) * 3) >> 8) as i16;
                        out[base + outoff + n * 4 + j] = (xi - 1) as f32 * d;
                    }
                }
                // outoff += 16 (unused but for clarity)
                let _ = outoff;
            }
            out
        }
        // ── TQ2_0: block = [u8 qs[64]][half d], 66 bytes, QK_K=256 ──────────────
        // 2 bits per element; 4 elements packed per byte; two 32-byte passes.
        //   For each 32-byte chunk j, for l in 0..4, for m in 0..32:
        //     q = (qs[j+m] >> (l*2)) & 3  ∈ {0,1,2,3}
        //     y = (q - 1) * d
        // Ref: llama.cpp dequantize_row_tq2_0 (ggml-quants.c l.2395)
        Tq2_0 => {
            let bpb = 66usize; // 64 + 2
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 256];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let qs = &blk[0..64];
                let d = rdf16(&blk[64..66]);
                let base = b * 256;
                let mut outoff = 0usize;
                // Two 32-byte chunks (j=0, j=32)
                for chunk in 0..2usize {
                    let j = chunk * 32;
                    for l in 0..4usize {
                        for m in 0..32usize {
                            let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                            out[base + outoff] = (q - 1) as f32 * d;
                            outoff += 1;
                        }
                    }
                }
            }
            out
        }
        // ── Q2_0: block = [half d][u8 qs[16]], 18 bytes, QK2_0=64 (Bonsai ternary) ──
        // 2 bits per element, packed sequentially: element j → byte j/4, shift (j%4)*2.
        //   q = (qs[j/4] >> ((j%4)*2)) & 3  ∈ {0,1,2,3}
        //   y = (q - 1) * d                 → {-d, 0, +d, +2d}
        // Ref: llama.cpp dequantize_row_q2_0 (ggml-quants.c l.439)
        Q2_0 => {
            let bpb = 18usize; // 2 + 16
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 64];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = rdf16(blk);
                let qs = &blk[2..18];
                let base = b * 64;
                for j in 0..64usize {
                    let q = ((qs[j / 4] >> ((j % 4) * 2)) & 3) as i32;
                    out[base + j] = (q - 1) as f32 * d;
                }
            }
            out
        }
        // ── MXFP4: block = [u8 e][u8 qs[16]], 17 bytes, QK_MXFP4=32 ─────────────
        // E8M0 shared exponent + nibble-packed E2M1 4-bit values.
        //   d = e8m0_to_fp32_half(e) = 2^(e-128)
        //   x0 = kvalues_mxfp4[qs[j] & 0xF]; x1 = kvalues_mxfp4[qs[j] >> 4]
        //   y[j+0] = x0*d; y[j+16] = x1*d
        // Ref: llama.cpp dequantize_row_mxfp4 (ggml-quants.c l.511)
        Mxfp4 => {
            let bpb = 17usize; // 1 + 16
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 32];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let d = e8m0_to_fp32_half(blk[0]);
                let qs = &blk[1..17];
                let base = b * 32;
                for j in 0..16usize {
                    let x0 = KVALUES_MXFP4[(qs[j] & 0xF) as usize] as f32;
                    let x1 = KVALUES_MXFP4[(qs[j] >> 4) as usize] as f32;
                    out[base + j] = x0 * d;
                    out[base + j + 16] = x1 * d;
                }
            }
            out
        }
        // ── NVFP4: block = [u8 d[4]][u8 qs[32]], 36 bytes, QK_NVFP4=64 ──────────
        // 4 sub-blocks of 16 elements each; UE4M3 scale per sub-block.
        //   For s in 0..4: d = ue4m3_to_fp32(scales[s])
        //     For j in 0..7: v0 = kvalues_mxfp4[qs[s*8+j] & 0xF]; v1 = qs[s*8+j] >> 4
        //     yb[j] = v0*d; yb[j+8] = v1*d
        // Ref: llama.cpp dequantize_row_nvfp4 (ggml-quants.c l.531)
        Nvfp4 => {
            let bpb = 36usize; // 4 + 32
            let nblk = bytes.len() / bpb;
            let mut out = vec![0.0f32; nblk * 64];
            for b in 0..nblk {
                let blk = &bytes[b * bpb..(b + 1) * bpb];
                let scales = &blk[0..4];
                let qs = &blk[4..36];
                let base = b * 64;
                for s in 0..4usize {
                    let d = ue4m3_to_fp32(scales[s]);
                    let ybase = base + s * 16;
                    for j in 0..8usize {
                        let v0 = KVALUES_MXFP4[(qs[s * 8 + j] & 0xF) as usize] as f32;
                        let v1 = KVALUES_MXFP4[(qs[s * 8 + j] >> 4) as usize] as f32;
                        out[ybase + j] = v0 * d;
                        out[ybase + j + 8] = v1 * d;
                    }
                }
            }
            out
        }
        // ── I2S (BitNet i2_s): ternary {-1,0,+1}, 2 bits/elem + ONE per-tensor f32 scale ────────
        // Layout (microsoft/BitNet, quantize_i2_s / ggml_vec_dot_i2_i8_s, x86 build): `numel/4`
        // packed bytes, then a trailing f32 `scale`. Codes are packed INTERLEAVED in 128-element
        // groups: element `e = i*128 + j` (natural row-major order) → byte `i*32 + (j%32)`, in the
        // 2-bit field at shift `6 - 2*(j/32)`. Code `q ∈ {0,1,2}` maps to `q-1 ∈ {-1,0,+1}` and
        // `y = (q-1) * scale`. (Rows never split a 128-group: every ne0 in these GGUFs is a
        // multiple of 128.) Verified against the real 2B-4T model: scale ≈ absmax, ~50% zeros.
        I2S => {
            // The slice is `numel/4` packed bytes + a 4-byte f32 scale at the tail (see
            // `tensor_nbytes`). Recover numel from the packed-code length.
            let packed = bytes.len().saturating_sub(4);
            let numel = packed * 4;
            let scale = f32::from_le_bytes([
                bytes[packed],
                bytes[packed + 1],
                bytes[packed + 2],
                bytes[packed + 3],
            ]);
            let mut out = vec![0.0f32; numel];
            for e in 0..numel {
                let i = e / 128;
                let j = e % 128;
                let byte = bytes[i * 32 + (j % 32)];
                let shift = 6 - 2 * (j / 32);
                let code = (byte >> shift) & 3;
                out[e] = (code as f32 - 1.0) * scale;
            }
            out
        }
        other => unimplemented!("codebook dequant for {other:?} not yet implemented"),
    }
}

/// True for types that go through the GPU in-kernel affine dequant path (`Wt::Q`).
/// All affine quants: legacy round quants + k-quants (Q2K–Q6K).
/// Codebook quants (IQ*/TQ*/fp4) are NOT included — they go host-dequant → f16.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn is_quant(d: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(
        d,
        Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2K | Q3K | Q4K | Q5K | Q6K
    )
}

#[cfg(test)]
mod dequant_tests {
    use super::*;

    // ── factored fn hardening: non-affine dtype must error, not panic ──────────
    // `dequant_factored` is `pub` and used by infr-metal; passing a non-affine dtype
    // (F16 / codebook quant) previously hit `unreachable!()`. It must now return Err.
    #[test]
    fn dequant_factored_non_affine_errors() {
        for d in [
            infr_core::DType::F16,
            infr_core::DType::Iq4Nl,
            infr_core::DType::Iq2Xxs,
            infr_core::DType::Tq1_0,
        ] {
            assert!(
                dequant_factored(d, &[0u8; 256]).is_err(),
                "dequant_factored({d:?}) must be Err, not panic/unreachable"
            );
        }
    }

    // ── bit-identity characterization for the single-pass affine expansion ──────
    // Reproduces the exact pre-refactor path (dequant_factored → materialize sc/mn →
    // expand `sc*code + mn`) and asserts the new fused `dequant_block` matches it
    // bit-for-bit (compare `to_bits()`, so any rounding/ordering drift fails).
    fn old_affine_expand(d: infr_core::DType, bytes: &[u8]) -> Vec<f32> {
        let f = dequant_factored(d, bytes).unwrap();
        let n = f.codes.len();
        let (mut sc, mut mn) = (vec![0f32; n], vec![0f32; n]);
        for g in 0..n {
            let (b, db) = (g / 16, g / f.dblk);
            sc[g] = f.dd[2 * db].to_f32() * f.scm[2 * b] as f32;
            mn[g] = f.dd[2 * db + 1].to_f32() * f.scm[2 * b + 1] as f32;
        }
        (0..n).map(|g| sc[g] * f.codes[g] as f32 + mn[g]).collect()
    }

    fn assert_bit_identical(d: infr_core::DType, bytes: &[u8]) {
        let expected = old_affine_expand(d, bytes);
        let got = dequant_block(d, bytes).unwrap();
        assert_eq!(got.len(), expected.len(), "{d:?} length mismatch");
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{d:?} element {i} not bit-identical: got {a} ({:#010x}) vs expected {b} ({:#010x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    #[test]
    fn affine_single_pass_bit_identical_q4k() {
        // Two full Q4_K blocks (144 bytes each) of patterned bytes; f16 scale slots
        // overwritten with finite values so codes span the full range.
        let mut bytes = vec![0u8; 144 * 2];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 37 + 11) & 0xFF) as u8;
        }
        for blk in 0..2 {
            let base = blk * 144;
            bytes[base..base + 2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
            bytes[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
        }
        assert_bit_identical(infr_core::DType::Q4K, &bytes);
    }

    #[test]
    fn affine_single_pass_bit_identical_q6k() {
        // Two full Q6_K blocks (210 bytes each) of patterned bytes; f16 super-scale slot
        // (bytes 208..210) set to a finite value.
        let mut bytes = vec![0u8; 210 * 2];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((i * 53 + 7) & 0xFF) as u8;
        }
        for blk in 0..2 {
            let base = blk * 210;
            bytes[base + 208..base + 210].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        }
        assert_bit_identical(infr_core::DType::Q6K, &bytes);
    }

    // Cover every affine format's single-pass path against the old expansion.
    #[test]
    fn affine_single_pass_bit_identical_all_formats() {
        use infr_core::DType::*;
        for (d, bpb) in [
            (Q4_0, 18usize),
            (Q4_1, 20),
            (Q5_0, 22),
            (Q5_1, 24),
            (Q8_0, 34),
            (Q2K, 84),
            (Q3K, 110),
            (Q4K, 144),
            (Q5K, 176),
            (Q6K, 210),
        ] {
            let mut bytes = vec![0u8; bpb * 2];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = ((i * 29 + 3) & 0xFF) as u8;
            }
            assert_bit_identical(d, &bytes);
        }
    }

    // ── IQ4_NL ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qs[16]], 32 elements, 18 bytes
    // y[j] = d * KVALUES_IQ4NL[qs[j] & 0xF]; y[j+16] = d * KVALUES_IQ4NL[qs[j] >> 4]
    // Reference: llama.cpp dequantize_row_iq4_nl (ggml-quants.c l.2653)
    #[test]
    fn iq4nl_single_block() {
        // d=1.0, qs[0]=0x80 (lo=0, hi=8)
        // KVALUES_IQ4NL[0] = -127, KVALUES_IQ4NL[8] = 1
        // y[0] = 1.0 * (-127) = -127.0
        // y[16] = 1.0 * 1 = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x80; // lo=0→-127, hi=8→1
        let y = dequant_codebook(infr_core::DType::Iq4Nl, &block);
        assert_eq!(y.len(), 32);
        assert!(
            (y[0] - (-127.0)).abs() < 1e-3,
            "iq4nl y[0] expected -127.0, got {}",
            y[0]
        );
        assert!(
            (y[16] - 1.0).abs() < 1e-3,
            "iq4nl y[16] expected 1.0, got {}",
            y[16]
        );
    }

    // ── IQ4_XS ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 scales_h][uint8 scales_l[4]][uint8 qs[128]], 256 elements, 136 bytes
    // y = d*(ls-32) * KVALUES_IQ4NL[q4], ls is 6-bit per 32-elem sub-block
    // Reference: llama.cpp dequantize_row_iq4_xs (ggml-quants.c l.2671)
    #[test]
    fn iq4xs_single_block() {
        // d=1.0, scales: all sub-blocks have ls=32 → dl=d*(32-32)=0 → y=0
        // Verify: all 256 outputs are 0.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&d_bytes);
        // scales_h=0, scales_l=[0x00,0x00,0x00,0x00]: all lo=0, all hi=0 → ls=0 → dl=-32
        // Wait: ls=lo|(hi<<4). With scales_h=0 and scales_l=0, ls=0. dl=1.0*(0-32)=-32.
        // qs all 0: qs[j]&0xF=0 → KVALUES_IQ4NL[0]=-127; qs[j]>>4=0 → -127
        // y = -32 * (-127) = 4064.0 (all elements)
        let y = dequant_codebook(infr_core::DType::Iq4Xs, &block);
        assert_eq!(y.len(), 256);
        let expected = -32.0_f32 * KVALUES_IQ4NL[0] as f32; // 4064.0
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 0.5,
                "iq4xs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ1_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[32]][u16 qh[8]], 50 bytes, QK_K=256
    // All-zero block: d=1.0, qh=0 → dl=1.0*(2*0+1)=1.0, delta=+0.125, grid_idx=0
    //   IQ1S_GRID[0] = 0xffffffffffffffff → gv=-1 for all j
    //   y[j] = 1.0 * (-1.0 + 0.125) = -0.875 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq1_s (ggml-quants.c l.2578)
    #[test]
    fn iq1s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&d_bytes);
        // qs=0, qh=0 → grid_idx=0, dl=1.0, delta=+0.125
        let y = dequant_codebook(infr_core::DType::Iq1S, &block);
        assert_eq!(y.len(), 256);
        // IQ1S_GRID[0] = 0xffffffffffffffff → all bytes 0xFF = -1i8
        let expected = 1.0_f32 * (-1.0_f32 + IQ1S_DELTA);
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq1s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── MXFP4 ───────────────────────────────────────────────────────────────────
    // Block: [u8 e][u8 qs[16]], 17 bytes, QK_MXFP4=32
    // e=128 → d=e8m0_to_fp32_half(128)=2^(128-128)=1.0; qs[0]=0x21 → lo=1, hi=2
    //   y[0] = KVALUES_MXFP4[1]*1.0 = 1.0; y[16] = KVALUES_MXFP4[2]*1.0 = 2.0
    // Ref: llama.cpp dequantize_row_mxfp4 (ggml-quants.c l.511)
    #[test]
    fn mxfp4_single_block() {
        let mut block = vec![0u8; 17];
        block[0] = 128; // e=128 → d=1.0
        block[1] = 0x21; // lo nibble=1→1, hi nibble=2→2
        let y = dequant_codebook(infr_core::DType::Mxfp4, &block);
        assert_eq!(y.len(), 32);
        assert!(
            (y[0] - 1.0).abs() < 1e-5,
            "mxfp4 y[0] expected 1.0, got {}",
            y[0]
        );
        assert!(
            (y[16] - 2.0).abs() < 1e-5,
            "mxfp4 y[16] expected 2.0, got {}",
            y[16]
        );
        // rest of qs=0 → x0=x1=0 → y=0.0
        for i in 1..16 {
            assert!(y[i].abs() < 1e-5, "mxfp4 y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── NVFP4 ───────────────────────────────────────────────────────────────────
    // Block: [u8 d[4]][u8 qs[32]], 36 bytes, QK_NVFP4=64
    // All-zero scales: d=ue4m3_to_fp32(0)=0.0 → all y=0.0
    // Ref: llama.cpp dequantize_row_nvfp4 (ggml-quants.c l.531)
    #[test]
    fn nvfp4_single_block() {
        let block = vec![0u8; 36];
        let y = dequant_codebook(infr_core::DType::Nvfp4, &block);
        assert_eq!(y.len(), 64);
        for i in 0..64 {
            assert!(y[i].abs() < 1e-5, "nvfp4 y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── IQ1_M ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[32]][u8 qh[16]][u8 scales[8]], 56 bytes, QK_K=256
    // All-zero: scales=0 → d_bits=0 → d=0.0 → all y=0.0
    // Ref: llama.cpp dequantize_row_iq1_m (ggml-quants.c l.2603)
    #[test]
    fn iq1m_single_block() {
        let block = vec![0u8; 56];
        let y = dequant_codebook(infr_core::DType::Iq1M, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(y[i].abs() < 1e-4, "iq1m y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── TQ1_0 ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[48]][u8 qh[4]][half d], 54 bytes, QK_K=256
    // All-zero qs/qh: q=0 → xi=0 → y=(0-1)*d = -d for all 256 elements
    // Ref: llama.cpp dequantize_row_tq1_0 (ggml-quants.c l.2356)
    #[test]
    fn tq1_0_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 54];
        block[52..54].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Tq1_0, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "tq1_0 y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
    }

    // ── TQ2_0 ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[64]][half d], 66 bytes, QK_K=256
    // All-zero qs: q=(0>>l*2)&3=0 → y=(0-1)*d = -d for all 256 elements
    // Ref: llama.cpp dequantize_row_tq2_0 (ggml-quants.c l.2395)
    #[test]
    fn tq2_0_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 66];
        block[64..66].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Tq2_0, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "tq2_0 y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
    }

    // ── Q2_0 (Bonsai ternary) ───────────────────────────────────────────────────
    // Block: [half d][u8 qs[16]], 18 bytes, QK2_0=64
    // qs[0]=0b11100100 encodes q = 0,1,2,3 for elements 0..3 → y = -d, 0, +d, +2d.
    // Remaining qs zero: q=0 → y=-d for elements 4..63.
    // Ref: llama.cpp dequantize_row_q2_0 (ggml-quants.c l.439)
    #[test]
    fn q2_0_single_block() {
        let d_bytes = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0b1110_0100; // elems 0..3 → codes 0,1,2,3
        let y = dequant_codebook(infr_core::DType::Q2_0, &block);
        assert_eq!(y.len(), 64);
        let want = [-0.5f32, 0.0, 0.5, 1.0];
        for (i, w) in want.iter().enumerate() {
            assert!(
                (y[i] - w).abs() < 1e-4,
                "q2_0 y[{i}] expected {w}, got {}",
                y[i]
            );
        }
        for i in 4..64 {
            assert!(
                (y[i] - (-0.5)).abs() < 1e-4,
                "q2_0 y[{i}] expected -0.5, got {}",
                y[i]
            );
        }
    }

    // ── I2S (BitNet i2_s) ────────────────────────────────────────────────────────
    // Layout: `numel/4` packed bytes (2 bits/elem, interleaved in 128-elem groups) + a trailing
    // f32 scale. Element `e = i*128+j` → byte `i*32 + j%32`, field shift `6 - 2*(j/32)`; code
    // q∈{0,1,2} → (q-1)*scale. One 128-elem group here (32 packed bytes + 4-byte scale = 36 bytes).
    // byte[0]'s four 2-bit fields (hi→lo) govern elements 0, 32, 64, 96; bytes[1..32]=0x55 sets
    // every other element's field to code 1 (→ 0). Ref: microsoft/BitNet quantize_i2_s / dot.
    #[test]
    fn i2s_single_group() {
        let mut block = vec![0u8; 36];
        // byte[0] = 0b00_01_10_00: elem0=code0(-1), elem32=code1(0), elem64=code2(+1), elem96=code0(-1)
        block[0] = 0b00_01_10_00;
        for b in &mut block[1..32] {
            *b = 0x55; // 0b01_01_01_01 → all fields code 1 → 0
        }
        let scale = 2.0_f32;
        block[32..36].copy_from_slice(&scale.to_le_bytes());
        let y = dequant_codebook(infr_core::DType::I2S, &block);
        assert_eq!(y.len(), 128);
        assert!(
            (y[0] - (-2.0)).abs() < 1e-6,
            "i2s elem0 expected -2.0, got {}",
            y[0]
        );
        assert!(
            (y[32] - 0.0).abs() < 1e-6,
            "i2s elem32 expected 0.0, got {}",
            y[32]
        );
        assert!(
            (y[64] - 2.0).abs() < 1e-6,
            "i2s elem64 expected 2.0, got {}",
            y[64]
        );
        assert!(
            (y[96] - (-2.0)).abs() < 1e-6,
            "i2s elem96 expected -2.0, got {}",
            y[96]
        );
        // every other element decodes to code 1 → 0
        for (e, v) in y.iter().enumerate() {
            if e == 0 || e == 32 || e == 64 || e == 96 {
                continue;
            }
            assert!(v.abs() < 1e-6, "i2s elem{e} expected 0.0, got {v}");
        }
        // Round-trip via the public dequant_block entry (mirrors what the seam calls at load).
        let y2 = dequant_block(infr_core::DType::I2S, &block).unwrap();
        assert_eq!(y2.len(), 128);
        assert!(y2.iter().all(|v| v.is_finite()));
        assert_eq!(y, y2);
    }

    // ── IQ2_XXS ─────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 qs[32]], 66 bytes, QK_K=256
    // Sub-block 0: aux0=0 → 4 grid indices all 0; aux1=0 → scale_mag=0, sign_idx=0
    //   IQ2XXS_GRID[0] = 0x0808080808080808 → 8 bytes all 0x08
    //   KSIGNS_IQ2XS[0] = 0 → no negations
    //   db = 1.0*(0.5+0)*0.25 = 0.125
    //   y = 0.125 * 8 = 1.0 for each of 8 elements × 4 groups = 32 elements
    // Ref: llama.cpp dequantize_row_iq2_xxs (ggml-quants.c l.2416)
    #[test]
    fn iq2xxs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&d_bytes);
        // all qs = 0: aux0=0 (grid_idx=0), aux1=0 (scale_mag=0, sign_idx=0)
        let y = dequant_codebook(infr_core::DType::Iq2Xxs, &block);
        assert_eq!(y.len(), 256);
        // first sub-block, first element
        let expected = 0.125 * 8.0_f32;
        for i in 0..32 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
        // remaining sub-blocks: same pattern (all zeros)
        for i in 32..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_XS ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 qs[32]][uint8 scales[8]], 74 bytes, QK_K=256
    // All zeros: scales[0]=0 → db0=db1=0.125; qs16=0 → grid_idx=0, sign_idx=0
    //   IQ2XS_GRID[0] = 0x0808080808080808 → gv=8; KSIGNS[0]=0 → +1
    //   y = 0.125 * 8 = 1.0 for first 32 elements
    // Ref: llama.cpp dequantize_row_iq2_xs (ggml-quants.c l.2444)
    #[test]
    fn iq2xs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq2Xs, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.125 * 8.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[64]][u8 qh[8]][u8 scales[8]], 82 bytes, QK_K=256
    // All zeros: scales=0 → db0=db1=0.125; qs_all[0]=0, qh[0]=0 → grid_idx=0
    //   IQ2S_GRID[0] = 0x0808080808080808 → gv=8; signs[32]=0 → +1
    //   y = 0.125 * 8 = 1.0 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq2_s (ggml-quants.c l.2471)
    #[test]
    fn iq2s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq2S, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.125 * 8.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ3_XXS ─────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[96]], 98 bytes, QK_K=256
    // qs[0..64]=0 → grid_idx=0 for all; qs[64..96]=0 → aux32=0 → scale_mag=0, sign_idx=0
    //   IQ3XXS_GRID[0] = 0x04040404 → gv for j=0..3: 4; KSIGNS[0]=0 → +1
    //   db = 1.0*(0.5+0)*0.5 = 0.25
    //   y = 0.25 * 4 = 1.0 for first 32 elements
    // Ref: llama.cpp dequantize_row_iq3_xxs (ggml-quants.c l.2503)
    #[test]
    fn iq3xxs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq3Xxs, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.25 * 4.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq3xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ3_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]], 110 bytes
    // All zeros: scales=0 → db1=db2=1.0*(1+2*0)=1.0; qs=0, qh=0 → grid_idx=0
    //   IQ3S_GRID[0] = 0x01010101 → gv for j=0..3: 1; signs[0]=0 → +1
    //   y = 1.0 * 1 = 1.0 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq3_s (ggml-quants.c l.2535)
    #[test]
    fn iq3s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq3S, &block);
        assert_eq!(y.len(), 256);
        let expected = 1.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq3s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── Q2_K ────────────────────────────────────────────────────────────────────
    // Block: [uint8 scales[16]][uint8 qs[64]][half d][half dmin]
    // y = d*(sc&0xF)*q2 - dmin*(sc>>4), q2 ∈ 0..3
    // Reference: llama.cpp dequantize_row_q2_K (ggml-quants.c l.903)
    #[test]
    fn q2k_single_block() {
        // d=1.0, dmin=2.0
        // scales[0]=0x23 → lo=3, hi=2 → first sub-block: dl=3.0, ml=4.0
        // scales[1]=0x23 → second 16-elem sub-block (qs[16..32]): same dl/ml
        // qs[0..16]=0x55 → q2 (shift=0) = 0x55 & 3 = 1
        // Expected y[0] = 3.0*1 - 4.0 = -1.0
        let mut block = vec![0u8; 84];
        // scales[0..16]
        block[0] = 0x23; // lo=3, hi=2
        block[1] = 0x23; // same for second sub-block
                         // qs[16..80]
        for b in &mut block[16..80] {
            *b = 0x55; // any bits; q2 at shift=0 for first 16 = 1
        }
        // d[80..82] = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[80..82].copy_from_slice(&d_bytes);
        // dmin[82..84] = 2.0
        let dmin_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        block[82..84].copy_from_slice(&dmin_bytes);

        let y = dequant_block(infr_core::DType::Q2K, &block).unwrap();
        assert_eq!(y.len(), 256);
        // First sub-block, first element: q2=1, y=3.0*1-4.0=-1.0
        assert!(
            (y[0] - (-1.0)).abs() < 1e-4,
            "q2k y[0] expected -1.0, got {}",
            y[0]
        );
        // All elements in first sub-block same q2=1 → same y
        for i in 0..16 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "q2k y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
        // Second sub-block (16..32): same scales, qs=0x55, q2=(0x55>>2)&3=(0x15)&3=1
        // Wait: shift=0 for j=0 applies to BOTH first and second 16-elem groups of the
        // same j-iteration. Let me re-check the llama logic.
        // In the llama code, for j=0 (shift=0):
        //   sc=scales[0], dl=d*(sc&0xF)=3, ml=dmin*(sc>>4)=4
        //   for l in 0..16: q2 = (q[l] >> 0) & 3 = qs[l] & 3 = 0x55 & 3 = 1
        //   sc=scales[1], dl=d*(sc&0xF)=3, ml=dmin*(sc>>4)=4
        //   for l in 0..16: q2 = (q[l+16] >> 0) & 3 = qs[l+16] & 3 = 1
        // So elements 16..32 also have dl=3, ml=4, q2=1 → y=-1.0
        assert!(
            (y[16] - (-1.0)).abs() < 1e-4,
            "q2k y[16] expected -1.0, got {}",
            y[16]
        );
    }

    // ── Q3_K ────────────────────────────────────────────────────────────────────
    // Block: [uint8 hmask[32]][uint8 qs[64]][uint8 scales[12]][half d]
    // y = d*(sc6-32)*(q3u - 4), q3u = (low2 | high_bit<<2) ∈ 0..7
    // Reference: llama.cpp dequantize_row_q3_K (ggml-quants.c l.1247)
    #[test]
    fn q3k_single_block() {
        // d=1.0
        // Choose scales to decode as sc6=36 for all sub-blocks → sc6-32=4 → dl=4.0
        // Encode sc6=36 for first sub-block in scales_raw:
        //   After decode, aux bytes give sc6 values. Simpler: set all scales[0..12]=0
        //   so that after bit manipulation aux has all-zero lower nibbles → sc6=0 for all.
        //   Then dl=0 → y=0 everywhere. That's a trivial test.
        //
        // Better: set scales bytes to give sc6=32 for first two sub-blocks (dl=0, y=0)
        // and verify that y[0..32]=0. Then set hmask and qs to anything.
        //
        // Even simpler: set scales_raw all-zero. After bit manipulation:
        //   aux[0]=0, aux[1]=0, aux[2]=0, aux[3]=0
        //   sc6(0)= aux[0] byte0 = 0 → sc6-32 = -32 → dl=-32
        //   hmask[0..16]=0 (high bit=0), qs[0..16]=0x00 (low2=0 at shift=0)
        //   q3u = 0 | (0<<2) = 0. y = -32*0 + (-4)*(-32) = 128... wait
        //   y = dl*q3u + (-4*dl) = -32*0 + (-4*(-32)) = 128
        //
        // Let me verify this explicitly:
        //   q3u=0, dl=-32, min=-4*dl=128. y = -32*0 + 128 = 128. ✓
        //
        // Alternatively: set scales_raw to encode sc6=32 for sub-block 0.
        //   When tmp=aux[2]=0, aux[0]=scales_bytes[0..4] as u32.
        //   For sc6=32 after decode:
        //     sc6(0) = (aux[0] & 0xFF) = 32 → need aux[0] byte 0 = 32 = 0x20
        //     After bit manip (tmp=0): aux[0] = (orig_aux0 & 0x0F0F0F0F) | ...
        //     So (orig_aux0 & 0xF) = 32? 32 > 15, so the lower 4 bits can't encode 32.
        //
        // The scale decoding is complex. Let me just use all-zero scales (sc6=0, dl=-32*1=-32)
        // with hmask[0..16]=0 and qs[0..16]=0x00:
        // y = -32*0 + (-4*(-32)) = 128.0
        let mut block = vec![0u8; 110];
        // hmask[0..32] = all 0 (high bit not set for any elem)
        // qs[32..96] = all 0 (low2=0 at any shift)
        // scales[96..108] = all 0 (encodes sc6=0 after bit manipulation → dl=-32)
        // d[108..110] = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[108..110].copy_from_slice(&d_bytes);

        let y = dequant_block(infr_core::DType::Q3K, &block).unwrap();
        assert_eq!(y.len(), 256);
        // sc6=0 → dl = 1.0*(0-32) = -32.0
        // q3u = 0 (hmask=0, qs=0), min = -4*(-32) = 128
        // y[0] = -32*0 + 128 = 128.0
        assert!(
            (y[0] - 128.0).abs() < 1e-3,
            "q3k y[0] expected 128.0, got {}",
            y[0]
        );
        // All elements should be 128.0 (same scale, q3u=0 everywhere)
        for i in 0..256 {
            assert!(
                (y[i] - 128.0).abs() < 1e-3,
                "q3k y[{i}] expected 128.0, got {}",
                y[i]
            );
        }
    }

    // ── Q4_0 ────────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qs[16]]; y = d * (q4 - 8), q4 ∈ 0..15
    // Reference: llama.cpp dequantize_row_q4_0 (ggml-quants.c l.401)
    #[test]
    fn q4_0_single_block() {
        // d = 2.0 (f16 = 0x4000), qs[0] = 0x89 (lo=9, hi=8), rest = 0x88 (lo=8, hi=8)
        let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x89; // qs[0]: lo=9, hi=8
        for b in &mut block[3..18] {
            *b = 0x88; // lo=8, hi=8 → y = d*(8-8) = 0
        }
        let y = dequant_block(infr_core::DType::Q4_0, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 2.0*(9-8) = 2.0
        assert!(
            (y[0] - 2.0).abs() < 1e-5,
            "q4_0 y[0] expected 2.0, got {}",
            y[0]
        );
        // y[16] = 2.0*(8-8) = 0.0
        assert!(y[16].abs() < 1e-5, "q4_0 y[16] expected 0.0, got {}", y[16]);
        // y[1] = 2.0*(8-8) = 0.0
        assert!(y[1].abs() < 1e-5, "q4_0 y[1] expected 0.0, got {}", y[1]);
    }

    // ── Q4_1 ────────────────────────────────────────────────────────────────────
    // Block: [half d][half m][uint8 qs[16]]; y = d*q4 + m, q4 ∈ 0..15
    // Reference: llama.cpp dequantize_row_q4_1 (ggml-quants.c l.421)
    #[test]
    fn q4_1_single_block() {
        // d=1.0, m=0.5, qs[0]=0x30 (lo=0, hi=3), rest=0x00
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        block[4] = 0x30; // lo=0, hi=3
        let y = dequant_block(infr_core::DType::Q4_1, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 1.0*0 + 0.5 = 0.5
        assert!(
            (y[0] - 0.5).abs() < 1e-4,
            "q4_1 y[0] expected 0.5, got {}",
            y[0]
        );
        // y[16] = 1.0*3 + 0.5 = 3.5
        assert!(
            (y[16] - 3.5).abs() < 1e-4,
            "q4_1 y[16] expected 3.5, got {}",
            y[16]
        );
    }

    // ── Q5_0 ────────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qh[4]][uint8 qs[16]]; y = d*(q5 - 16), q5 ∈ 0..31
    // Reference: llama.cpp dequantize_row_q5_0 (ggml-quants.c l.442)
    #[test]
    fn q5_0_single_block() {
        // d=1.0, qh=[0x01,0,0,0] (bit 0 → element 0 gets high bit → q5=15|16=31)
        // qs[0]=0x0F (lo=15, hi=0), rest=0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x01; // qh[0]: bit 0 set
        block[6] = 0x0F; // qs[0]: lo=15, hi=0
        let y = dequant_block(infr_core::DType::Q5_0, &block).unwrap();
        assert_eq!(y.len(), 32);
        // j=0: xh0 = ((1>>0)<<4)&0x10 = 16. q5 = 15|16=31. y[0] = 1.0*(31-16) = 15.0
        assert!(
            (y[0] - 15.0).abs() < 1e-5,
            "q5_0 y[0] expected 15.0, got {}",
            y[0]
        );
        // j=0: xh1 = (1>>12)&0x10 = 0. q5 = 0. y[16] = 1.0*(0-16) = -16.0
        assert!(
            (y[16] - (-16.0)).abs() < 1e-5,
            "q5_0 y[16] expected -16.0, got {}",
            y[16]
        );
    }

    // ── Q5_1 ────────────────────────────────────────────────────────────────────
    // Block: [half d][half m][uint8 qh[4]][uint8 qs[16]]; y = d*q5 + m, q5 ∈ 0..31
    // Reference: llama.cpp dequantize_row_q5_1 (ggml-quants.c l.468)
    #[test]
    fn q5_1_single_block() {
        // d=2.0, m=-1.0, qh=[0,0,0,0], qs[0]=0x1F (lo=15, hi=1)
        let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(-1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        // qh[4] all zero → no high bits
        block[8] = 0x1F; // qs[0]: lo=15, hi=1
        let y = dequant_block(infr_core::DType::Q5_1, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 2.0*15 + (-1.0) = 29.0
        assert!(
            (y[0] - 29.0).abs() < 1e-4,
            "q5_1 y[0] expected 29.0, got {}",
            y[0]
        );
        // y[16] = 2.0*1 + (-1.0) = 1.0
        assert!(
            (y[16] - 1.0).abs() < 1e-4,
            "q5_1 y[16] expected 1.0, got {}",
            y[16]
        );
    }
}
