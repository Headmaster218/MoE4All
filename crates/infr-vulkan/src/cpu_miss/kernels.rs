// Validated packed-weight kernels; all entry points require AVX2 and FMA3.
use infr_core::iquant_grids::{IQ2S_GRID, IQ3S_GRID};
use infr_gguf::dequant::KVALUES_IQ4NL;
use std::arch::x86_64::*;
use std::sync::OnceLock;

#[repr(align(32))]
pub struct Tables {
    iq2: [[f32; 8]; 1024],
    iq3: [[f32; 4]; 512],
    signs: [[u32; 8]; 256],
    iq2_scale: [f32; 16],
    iq3_scale: [f32; 16],
}

pub fn tables() -> &'static Tables {
    static TABLES: OnceLock<Box<Tables>> = OnceLock::new();
    TABLES.get_or_init(|| {
        Box::new(Tables {
            iq2: std::array::from_fn(|i| IQ2S_GRID[i].to_le_bytes().map(|v| v as f32)),
            iq3: std::array::from_fn(|i| IQ3S_GRID[i].to_le_bytes().map(|v| v as f32)),
            signs: std::array::from_fn(|s| {
                std::array::from_fn(|i| if s & (1 << i) != 0 { 0x80000000 } else { 0 })
            }),
            iq2_scale: std::array::from_fn(|i| (0.5 + i as f32) * 0.25),
            iq3_scale: std::array::from_fn(|i| 1.0 + 2.0 * i as f32),
        })
    })
}

const fn half_bits(bits: u16) -> u32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 31;
    let mut mantissa = (bits & 1023) as u32;
    if exp == 0 {
        if mantissa == 0 {
            return sign;
        }
        let mut shifts = 0;
        while mantissa & 1024 == 0 {
            mantissa <<= 1;
            shifts += 1;
        }
        sign | ((113 - shifts) << 23) | ((mantissa & 1023) << 13)
    } else if exp == 31 {
        sign | 0x7f800000 | (mantissa << 13) | if mantissa != 0 { 0x00400000 } else { 0 }
    } else {
        sign | (((exp as u32) + 112) << 23) | (mantissa << 13)
    }
}
const fn half_table() -> [u32; 65536] {
    let mut table = [0u32; 65536];
    let mut i = 0;
    while i < table.len() {
        table[i] = half_bits(i as u16);
        i += 1;
    }
    table
}
static HALF_TABLE: [u32; 65536] = half_table();

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn scale(p: *const u8) -> f32 {
    let bits = p.cast::<u16>().read_unaligned();
    f32::from_bits(*HALF_TABLE.as_ptr().add(bits as usize))
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn signed(packed: [u64; 4], p: *const u8) -> __m256i {
    let values = _mm256_set_epi64x(
        packed[3] as i64,
        packed[2] as i64,
        packed[1] as i64,
        packed[0] as i64,
    );
    let signs = _mm256_set1_epi32(p.cast::<i32>().read_unaligned());
    let broadcast = _mm256_shuffle_epi8(
        signs,
        _mm256_set_epi64x(
            0x0303030303030303,
            0x0202020202020202,
            0x0101010101010101,
            0,
        ),
    );
    let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    let negate = _mm256_cmpeq_epi8(_mm256_and_si256(broadcast, bits), bits);
    _mm256_sub_epi8(_mm256_xor_si256(values, negate), negate)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn accumulate(
    w: __m256i,
    lo_scale: f32,
    hi_scale: f32,
    x: [__m256; 4],
    mut acc: __m256,
) -> __m256 {
    let lo = _mm256_castsi256_si128(w);
    let hi = _mm256_extracti128_si256::<1>(w);
    let s0 = _mm256_set1_ps(lo_scale);
    let s1 = _mm256_set1_ps(hi_scale);
    let parts = [lo, _mm_srli_si128::<8>(lo), hi, _mm_srli_si128::<8>(hi)];
    for i in 0..4 {
        let weights = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(parts[i])),
            if i < 2 { s0 } else { s1 },
        );
        acc = _mm256_fmadd_ps(weights, x[i], acc);
    }
    acc
}

// Callers validate each row and input before entering this raw-pointer inner loop.
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile<const KIND: u8, const NR: usize>(rows: [*const u8; NR], x: &[f32]) -> [f32; NR] {
    packed::<KIND, NR>(rows, x)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn packed<const KIND: u8, const NR: usize>(rows: [*const u8; NR], x: &[f32]) -> [f32; NR] {
    assert!(NR > 0 && NR <= 8);
    let input = x.len();
    let mut acc = [_mm256_setzero_ps(); NR];
    if KIND == 4 {
        assert!(input.is_multiple_of(32));
        let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()));
        let mask = _mm_set1_epi8(15);
        for b in 0..input / 32 {
            let xp = x.as_ptr().add(b * 32);
            let activation = std::array::from_fn(|i| _mm256_loadu_ps(xp.add(i * 8)));
            for r in 0..NR {
                let p = rows[r].add(b * 18);
                let codes = _mm_loadu_si128(p.add(2).cast());
                let lo = _mm_and_si128(codes, mask);
                let hi = _mm_and_si128(_mm_srli_epi16(codes, 4), mask);
                let w = _mm256_shuffle_epi8(
                    table,
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi),
                );
                let d = scale(p);
                acc[r] = accumulate(w, d, d, activation, acc[r]);
            }
        }
    } else {
        assert!((KIND == 2 || KIND == 3) && input.is_multiple_of(256));
        let size = if KIND == 2 { 82 } else { 110 };
        for b in 0..input / 256 {
            let base = rows.map(|p| p.add(b * size));
            let d = base.map(|p| scale(p));
            for group in 0..8 {
                let xp = x.as_ptr().add(b * 256 + group * 32);
                let activation = std::array::from_fn(|i| _mm256_loadu_ps(xp.add(i * 8)));
                for r in 0..NR {
                    let p = base[r];
                    let high = *p.add(66 + group);
                    let (w, s0, s1) = if KIND == 2 {
                        let packed = std::array::from_fn(|l| {
                            let index = *p.add(2 + group * 4 + l) as usize
                                | (((high as u32) << (8 - 2 * l)) & 0x300) as usize;
                            *IQ2S_GRID.as_ptr().add(index)
                        });
                        let sc = *p.add(74 + group);
                        (
                            signed(packed, p.add(34 + group * 4)),
                            d[r] * (0.5 + (sc & 15) as f32) * 0.25,
                            d[r] * (0.5 + (sc >> 4) as f32) * 0.25,
                        )
                    } else {
                        let packed = std::array::from_fn(|l| {
                            let a = *p.add(2 + group * 8 + l * 2) as usize
                                | (((high as u32) << (8 - 2 * l)) & 256) as usize;
                            let b = *p.add(3 + group * 8 + l * 2) as usize
                                | (((high as u32) << (7 - 2 * l)) & 256) as usize;
                            *IQ3S_GRID.as_ptr().add(a) as u64
                                | (*IQ3S_GRID.as_ptr().add(b) as u64) << 32
                        });
                        let sc = (*p.add(106 + group / 2) >> ((group % 2) * 4)) & 15;
                        let ds = d[r] * (1.0 + 2.0 * sc as f32);
                        (signed(packed, p.add(74 + group * 4)), ds, ds)
                    };
                    acc[r] = accumulate(w, s0, s1, activation, acc[r]);
                }
            }
        }
    }
    std::array::from_fn(|r| {
        let mut values = [0.0; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), acc[r]);
        values.iter().sum()
    })
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_lut<const KIND: u8, const NR: usize>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    lookup::<KIND, NR, false>(rows, x, tables)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_stream<const KIND: u8, const NR: usize, const COMPACT: bool>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    stream_chains::<KIND, NR, COMPACT, 1, false>(rows, x, tables)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_stream_split<const KIND: u8, const NR: usize, const COMPACT: bool>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    stream_chains::<KIND, NR, COMPACT, 4, false>(rows, x, tables)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_stream_grouped<const KIND: u8, const NR: usize>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    stream_chains::<KIND, NR, true, 1, true>(rows, x, tables)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_stream_grouped_lut<const KIND: u8, const NR: usize>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    stream_chains::<KIND, NR, false, 1, true>(rows, x, tables)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_stream_dual<const KIND: u8, const NR: usize>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    stream_chains::<KIND, NR, true, 2, false>(rows, x, tables)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_iq4_grouped<const NR: usize>(rows: [*const u8; NR], x: &[f32]) -> [f32; NR] {
    iq4_grouped::<NR, false>(rows, x)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn tile_iq4_grouped_dual<const NR: usize>(
    rows: [*const u8; NR],
    x: &[f32],
) -> [f32; NR] {
    iq4_grouped::<NR, true>(rows, x)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn iq4_grouped<const NR: usize, const DUAL: bool>(
    rows: [*const u8; NR],
    x: &[f32],
) -> [f32; NR] {
    assert!(NR > 0 && NR <= 8 && x.len().is_multiple_of(32));
    let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()));
    let mask = _mm_set1_epi8(15);
    let mut acc = [_mm256_setzero_ps(); NR];
    for b in 0..x.len() / 32 {
        let xp = x.as_ptr().add(b * 32);
        let activation: [__m256; 4] = std::array::from_fn(|i| _mm256_loadu_ps(xp.add(i * 8)));
        for r in 0..NR {
            let p = rows[r].add(b * 18);
            let codes = _mm_loadu_si128(p.add(2).cast());
            let lo = _mm_and_si128(codes, mask);
            let hi = _mm_and_si128(_mm_srli_epi16(codes, 4), mask);
            let w = _mm256_shuffle_epi8(
                table,
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi),
            );
            let lo = _mm256_castsi256_si128(w);
            let hi = _mm256_extracti128_si256::<1>(w);
            let parts = [lo, _mm_srli_si128::<8>(lo), hi, _mm_srli_si128::<8>(hi)];
            let mut partial = [_mm256_setzero_ps(); 2];
            for i in 0..4 {
                let values = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(parts[i]));
                let chain = if DUAL { i % 2 } else { 0 };
                partial[chain] = _mm256_fmadd_ps(values, activation[i], partial[chain]);
            }
            let sum = if DUAL {
                _mm256_add_ps(partial[0], partial[1])
            } else {
                partial[0]
            };
            acc[r] = _mm256_fmadd_ps(sum, _mm256_set1_ps(scale(p)), acc[r]);
        }
    }
    std::array::from_fn(|r| {
        let mut values = [0.0; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), acc[r]);
        values.iter().sum()
    })
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn stream_chains<
    const KIND: u8,
    const NR: usize,
    const COMPACT: bool,
    const CHAINS: usize,
    const GROUPED: bool,
>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    assert!(KIND == 2 || KIND == 3);
    assert!(CHAINS == 1 || CHAINS == 2 || CHAINS == 4);
    assert!(x.len().is_multiple_of(256));
    // Four independent chains avoid serializing every 8-lane FMA on one accumulator.
    let mut acc = [[_mm256_setzero_ps(); NR]; 4];
    let size = if KIND == 2 { 82 } else { 110 };
    for b in 0..x.len() / 256 {
        let base = rows.map(|p| p.add(b * size));
        let d = base.map(|p| scale(p));
        for group in 0..8 {
            let mut partial = [[_mm256_setzero_ps(); NR]; 2];
            let high = base.map(|p| *p.add(66 + group));
            let sc = base.map(|p| {
                if KIND == 2 {
                    *p.add(74 + group)
                } else {
                    (*p.add(106 + group / 2) >> ((group % 2) * 4)) & 15
                }
            });
            let low = std::array::from_fn::<_, NR, _>(|r| {
                if KIND == 2 {
                    d[r] * tables.iq2_scale[(sc[r] & 15) as usize]
                } else {
                    d[r] * tables.iq3_scale[sc[r] as usize]
                }
            });
            let hi = std::array::from_fn::<_, NR, _>(|r| {
                if KIND == 2 {
                    d[r] * tables.iq2_scale[(sc[r] >> 4) as usize]
                } else {
                    low[r]
                }
            });
            for l in 0..4 {
                let activation = _mm256_loadu_ps(x.as_ptr().add(b * 256 + group * 32 + l * 8));
                for r in 0..NR {
                    let p = base[r];
                    let (value, signs) = if KIND == 2 {
                        let index = *p.add(2 + group * 4 + l) as usize
                            | (((high[r] as u32) << (8 - 2 * l)) & 0x300) as usize;
                        (
                            if COMPACT {
                                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(
                                    *IQ2S_GRID.as_ptr().add(index) as i64,
                                )))
                            } else {
                                _mm256_loadu_ps(tables.iq2.as_ptr().add(index).cast())
                            },
                            *p.add(34 + group * 4 + l) as usize,
                        )
                    } else {
                        let a = *p.add(2 + group * 8 + l * 2) as usize
                            | (((high[r] as u32) << (8 - 2 * l)) & 256) as usize;
                        let b = *p.add(3 + group * 8 + l * 2) as usize
                            | (((high[r] as u32) << (7 - 2 * l)) & 256) as usize;
                        (
                            if COMPACT {
                                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(
                                    (*IQ3S_GRID.as_ptr().add(a) as u64
                                        | (*IQ3S_GRID.as_ptr().add(b) as u64) << 32)
                                        as i64,
                                )))
                            } else {
                                _mm256_insertf128_ps::<1>(
                                    _mm256_castps128_ps256(_mm_loadu_ps(
                                        tables.iq3.as_ptr().add(a).cast(),
                                    )),
                                    _mm_loadu_ps(tables.iq3.as_ptr().add(b).cast()),
                                )
                            },
                            *p.add(74 + group * 4 + l) as usize,
                        )
                    };
                    let mask = _mm256_loadu_si256(tables.signs.as_ptr().add(signs).cast());
                    let signed = _mm256_xor_ps(value, _mm256_castsi256_ps(mask));
                    if GROUPED {
                        let half = if KIND == 2 { l / 2 } else { 0 };
                        partial[half][r] = _mm256_fmadd_ps(signed, activation, partial[half][r]);
                    } else {
                        let weight = _mm256_mul_ps(
                            signed,
                            _mm256_set1_ps(if l < 2 { low[r] } else { hi[r] }),
                        );
                        let chain = l % CHAINS;
                        acc[chain][r] = _mm256_fmadd_ps(weight, activation, acc[chain][r]);
                    }
                }
            }
            if GROUPED {
                for r in 0..NR {
                    acc[0][r] = _mm256_fmadd_ps(partial[0][r], _mm256_set1_ps(low[r]), acc[0][r]);
                    if KIND == 2 {
                        acc[0][r] =
                            _mm256_fmadd_ps(partial[1][r], _mm256_set1_ps(hi[r]), acc[0][r]);
                    }
                }
            }
        }
    }
    std::array::from_fn(|r| {
        let mut values = [0.0; 8];
        let sum = if CHAINS == 4 {
            _mm256_add_ps(
                _mm256_add_ps(acc[0][r], acc[1][r]),
                _mm256_add_ps(acc[2][r], acc[3][r]),
            )
        } else if CHAINS == 2 {
            _mm256_add_ps(acc[0][r], acc[1][r])
        } else {
            acc[0][r]
        };
        _mm256_storeu_ps(values.as_mut_ptr(), sum);
        values.iter().sum()
    })
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn lookup<const KIND: u8, const NR: usize, const COMPACT: bool>(
    rows: [*const u8; NR],
    x: &[f32],
    tables: &Tables,
) -> [f32; NR] {
    assert!(KIND == 2 || KIND == 3);
    assert!(x.len().is_multiple_of(256));
    let mut acc = [_mm256_setzero_ps(); NR];
    let size = if KIND == 2 { 82 } else { 110 };
    for b in 0..x.len() / 256 {
        let base = rows.map(|p| p.add(b * size));
        let d = base.map(|p| scale(p));
        for group in 0..8 {
            let xp = x.as_ptr().add(b * 256 + group * 32);
            let activation: [__m256; 4] = std::array::from_fn(|i| _mm256_loadu_ps(xp.add(i * 8)));
            for r in 0..NR {
                let p = base[r];
                let high = *p.add(66 + group);
                let sc = if KIND == 2 {
                    *p.add(74 + group)
                } else {
                    (*p.add(106 + group / 2) >> ((group % 2) * 4)) & 15
                };
                let s0 = if KIND == 2 {
                    d[r] * tables.iq2_scale[(sc & 15) as usize]
                } else {
                    d[r] * tables.iq3_scale[sc as usize]
                };
                let s1 = if KIND == 2 {
                    d[r] * tables.iq2_scale[(sc >> 4) as usize]
                } else {
                    s0
                };
                // Lane indices also address packed weight groups; keep the fixed SIMD layout.
                #[allow(clippy::needless_range_loop)]
                for l in 0..4 {
                    let (value, signs) = if KIND == 2 {
                        let index = *p.add(2 + group * 4 + l) as usize
                            | (((high as u32) << (8 - 2 * l)) & 0x300) as usize;
                        (
                            if COMPACT {
                                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(
                                    *IQ2S_GRID.as_ptr().add(index) as i64,
                                )))
                            } else {
                                _mm256_loadu_ps(tables.iq2.as_ptr().add(index).cast())
                            },
                            *p.add(34 + group * 4 + l) as usize,
                        )
                    } else {
                        let a = *p.add(2 + group * 8 + l * 2) as usize
                            | (((high as u32) << (8 - 2 * l)) & 256) as usize;
                        let b = *p.add(3 + group * 8 + l * 2) as usize
                            | (((high as u32) << (7 - 2 * l)) & 256) as usize;
                        (
                            if COMPACT {
                                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(
                                    (*IQ3S_GRID.as_ptr().add(a) as u64
                                        | (*IQ3S_GRID.as_ptr().add(b) as u64) << 32)
                                        as i64,
                                )))
                            } else {
                                _mm256_insertf128_ps::<1>(
                                    _mm256_castps128_ps256(_mm_loadu_ps(
                                        tables.iq3.as_ptr().add(a).cast(),
                                    )),
                                    _mm_loadu_ps(tables.iq3.as_ptr().add(b).cast()),
                                )
                            },
                            *p.add(74 + group * 4 + l) as usize,
                        )
                    };
                    let mask = _mm256_loadu_si256(tables.signs.as_ptr().add(signs).cast());
                    let weights = _mm256_mul_ps(
                        _mm256_xor_ps(value, _mm256_castsi256_ps(mask)),
                        _mm256_set1_ps(if l < 2 { s0 } else { s1 }),
                    );
                    acc[r] = _mm256_fmadd_ps(weights, activation[l], acc[r]);
                }
            }
        }
    }
    std::array::from_fn(|r| {
        let mut values = [0.0; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), acc[r]);
        values.iter().sum()
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn half_scales_match_all_65536_half_values() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        for bits in 0..=u16::MAX {
            let bytes = bits.to_le_bytes();
            let actual = unsafe { super::scale(bytes.as_ptr()) };
            let expected = half::f16::from_bits(bits).to_f32();
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}
