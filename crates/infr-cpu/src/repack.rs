//! Interleaved 8-row weight packs (Q4_K, Q6_K) for the VNNI GEMM path, and the GEMM kernels
//! that consume them — the (layer, expert) repack-cache payload.
#[cfg(target_arch = "x86_64")]
use crate::quant::Q8;
#[cfg(target_arch = "x86_64")]
use infr_gguf::dequant::{k4, rdf16};
use std::collections::HashMap;
use std::sync::Arc;

/// One 8-output-row group of a Q4_K weight bank in the interleaved-x8 form the VNNI GEMM
/// consumes (see [`vec_dot_q4k_batch8_ilv_vnni`]'s layout doc): `ilv` is the byte-interleaved
/// nibble expansion, the metadata arrays are PERM-lane-ordered per (super-block, sub-block) so
/// the GEMM can `loadu` them straight into vectors. Cacheable: building this is the per-call
/// repack cost the (layer, expert) cache eliminates (ggml pays it once at load via its
/// `block_q4_Kx8` buffers; we pay it once per cached expert).
/// Identity of one cached pack: `(buffer uid, byte offset within that buffer, byte length)`.
///
/// The uid — not the slice's ADDRESS — is what makes this key sound. A pack cache outlives the
/// weights it was built from (the same `CpuBackend` survives a serve model reload), and both an
/// allocator and an mmap hand the same address out again afterwards, so an address key returns a
/// pack built from a different weight of equal length: plausible numbers, no error. See
/// `CpuBuffer::uid`.
pub(crate) type PackKey = (u64, usize, usize);

/// `(entries keyed by [`PackKey`], total cached bytes)` — see `repack_cache`'s doc.
pub(crate) type RepackCacheState = (HashMap<PackKey, Arc<Q4kPack>>, usize);

pub(crate) type Repack6CacheState = (HashMap<PackKey, Arc<Q6kPack>>, usize);

/// Insert `pack` under `key` if still absent, then return the *resident* `Arc` — the caller holds
/// the cache lock. This closes the check-then-insert race in `q4k_pack_for`/`q6k_pack_for`: the lock
/// drops between the fast-path miss and the (expensive) build, so a second thread may have inserted
/// the same key meanwhile. Re-checking under the lock means only the first insertion counts the
/// bytes (against `budget`) and only one entry lands — the loser returns the winner's pack and drops
/// its own build. Byte accounting happens ONLY on a genuine insert.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn cache_insert_if_absent<P>(
    state: &mut (HashMap<PackKey, Arc<P>>, usize),
    key: PackKey,
    pack: Arc<P>,
    bytes: usize,
    budget: usize,
) -> Arc<P> {
    if let Some(p) = state.0.get(&key) {
        return p.clone();
    }
    if state.1 + bytes <= budget {
        state.1 += bytes;
        state.0.insert(key, pack.clone());
    }
    pack
}

// Only the x86 ilv kernels read these — plain data everywhere else (aarch64 CI builds with
// -D warnings, so the not-x86 dead-code must be explicitly allowed rather than cfg'd away:
// the types appear in cross-target signatures like `expert_matvec_batch`).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Q4kPackGroup {
    ilv: Vec<u8>, // [nb * 2048]
    // Pair-duplicated sub-block scales in dpbusd lane order (lanes 2i, 2i+1 = row i): the GEMM
    // scales each sub-block's 16-lane accumulator with ONE 512-bit mullo and adds vertically,
    // deferring the pair-merge to once per SUPER-block (the hadd then lands in PERM order
    // exactly as before). [nb * 8 subs * 16 lanes].
    sc16: Vec<i32>,
    msc: Vec<i32>,  // [nb * 8 subs * 8 lanes], PERM order
    d: Vec<f32>,    // [nb * 8 lanes], PERM order
    dmin: Vec<f32>, // [nb * 8 lanes], PERM order
}

/// A whole Q4_K weight bank (e.g. one MoE expert's fused gate_up) packed as full 8-row groups —
/// `out_f % 8` tail rows are NOT packed (callers run them through the per-row kernel).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Q4kPack {
    pub(crate) groups: Vec<Q4kPackGroup>,
    pub(crate) nb: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Q4kPack {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) fn bytes(&self) -> usize {
        self.groups
            .iter()
            .map(|g| {
                g.ilv.len() + (g.sc16.len() + g.msc.len()) * 4 + (g.d.len() + g.dmin.len()) * 4
            })
            .sum()
    }
}

/// Build the pack for `wbytes` (a `[out_f, in_f]` Q4_K bank). AVX2 is enough for the expansion.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q4k_pack(wbytes: &[u8], in_f: usize, out_f: usize) -> Q4kPack {
    use std::arch::x86_64::*;
    const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
    let nb = in_f / 256;
    let bpr = wbytes.len() / out_f;
    let mask_lo = _mm256_set1_epi8(0x0F_u8 as i8);
    let n_groups = out_f / 8;
    let mut groups = Vec::with_capacity(n_groups);
    let mut tmp = [[0u8; 256]; 8];
    let mut sc_rows = [[0u32; 8]; 8];
    let mut m_rows = [[0u32; 8]; 8];
    let mut d_rows = [0f32; 8];
    let mut dmin_rows = [0f32; 8];
    for g in 0..n_groups {
        let mut pg = Q4kPackGroup {
            ilv: vec![0u8; nb * 2048],
            sc16: vec![0i32; nb * 128],
            msc: vec![0i32; nb * 64],
            d: vec![0f32; nb * 8],
            dmin: vec![0f32; nb * 8],
        };
        for b in 0..nb {
            for i in 0..8 {
                let row = &wbytes[(g * 8 + i) * bpr..(g * 8 + i) * bpr + bpr];
                let blk = &row[b * 144..b * 144 + 144];
                d_rows[i] = rdf16(&blk[0..2]);
                dmin_rows[i] = rdf16(&blk[2..4]);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                for s in 0..8usize {
                    let (sc, mv) = k4(s, scales);
                    sc_rows[i][s] = sc;
                    m_rows[i][s] = mv;
                }
                for k in 0..4usize {
                    let nibs = _mm256_loadu_si256(qs[k * 32..].as_ptr() as *const __m256i);
                    let lo = _mm256_and_si256(nibs, mask_lo);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(nibs, 4), mask_lo);
                    _mm256_storeu_si256(tmp[i][k * 64..].as_mut_ptr() as *mut __m256i, lo);
                    _mm256_storeu_si256(tmp[i][k * 64 + 32..].as_mut_ptr() as *mut __m256i, hi);
                }
            }
            for (j, &row) in PERM.iter().enumerate() {
                pg.d[b * 8 + j] = d_rows[row];
                pg.dmin[b * 8 + j] = dmin_rows[row];
            }
            for st in 0..8usize {
                for (j, &row) in PERM.iter().enumerate() {
                    pg.msc[b * 64 + st * 8 + j] = m_rows[row][st] as i32;
                }
                for i in 0..8usize {
                    pg.sc16[b * 128 + st * 16 + 2 * i] = sc_rows[i][st] as i32;
                    pg.sc16[b * 128 + st * 16 + 2 * i + 1] = sc_rows[i][st] as i32;
                }
                for gg in 0..4usize {
                    let dst = &mut pg.ilv
                        [b * 2048 + st * 256 + gg * 64..b * 2048 + st * 256 + gg * 64 + 64];
                    for i in 0..8 {
                        dst[i * 8..i * 8 + 8]
                            .copy_from_slice(&tmp[i][st * 32 + gg * 8..st * 32 + gg * 8 + 8]);
                    }
                }
            }
        }
        groups.push(pg);
    }
    Q4kPack { groups, nb }
}

/// The GEMM half of [`vec_dot_q4k_batch8_ilv_vnni`] over a prebuilt [`Q4kPackGroup`] — identical
/// math and rounding sequence, so bit-identity with the scalar oracle is preserved.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q4k_gemm_group(pg: &Q4kPackGroup, nb: usize, q8s: &[Q8], cols: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    // Raw-pointer inner loops: this GEMM's slice-range indexing was ~7% of ALL CPU as bounds
    // checks (samply leaf `index.rs`). The offsets are validated here once instead — pack layout
    // by construction of `q4k_pack`, activation layout by `quantize_q8`.
    assert_eq!(cols.len(), 8 * m);
    assert!(pg.ilv.len() >= nb * 2048 && pg.sc16.len() >= nb * 128 && pg.msc.len() >= nb * 64);
    assert!(pg.d.len() >= nb * 8 && pg.dmin.len() >= nb * 8);
    let ilv_p = pg.ilv.as_ptr();
    let sc16_p = pg.sc16.as_ptr();
    let msc_p = pg.msc.as_ptr();
    let d_p = pg.d.as_ptr();
    let dmin_p = pg.dmin.as_ptr();
    let cols_p = cols.as_mut_ptr();
    for r in 0..m {
        let q8 = &q8s[r];
        assert!(q8.qs.len() >= nb * 256 && q8.bsums.len() >= nb * 8 && q8.d.len() >= nb);
        let qs_p = q8.qs.as_ptr();
        let bsums_p = q8.bsums.as_ptr();
        let qd_p = q8.d.as_ptr();
        let mut sumf_v = _mm256_setzero_ps();
        for b in 0..nb {
            // Vertical accumulation: each sub-block's 16-lane sums are scaled with ONE 512-bit
            // mullo (pair-duplicated scales, see `sc16`) and added vertically; the pair-merge
            // hadd runs ONCE per super-block instead of per sub-block. Integer-exact
            // (Σ_s sc·(a+b) = Σ_s (sc·a + sc·b)) — still bit-identical to the scalar oracle.
            let mut sd_zmm = _mm512_setzero_si512();
            let mut sm_ymm = _mm256_setzero_si256();
            let q8b = qs_p.add(b * 256);
            for st in 0..8usize {
                let mut acc = _mm512_setzero_si512();
                for gg in 0..4usize {
                    let w = _mm512_loadu_si512(
                        ilv_p.add(b * 2048 + st * 256 + gg * 64) as *const __m512i
                    );
                    let a = _mm512_set1_epi64(
                        (q8b.add(st * 32 + gg * 8) as *const i64).read_unaligned(),
                    );
                    acc = _mm512_dpbusd_epi32(acc, w, a);
                }
                let sc16 = _mm512_loadu_si512(sc16_p.add(b * 128 + st * 16) as *const __m512i);
                sd_zmm = _mm512_add_epi32(sd_zmm, _mm512_mullo_epi32(acc, sc16));
                let m_v = _mm256_loadu_si256(msc_p.add(b * 64 + st * 8) as *const __m256i);
                let isum = _mm256_set1_epi32(*bsums_p.add(b * 8 + st));
                sm_ymm = _mm256_add_epi32(sm_ymm, _mm256_mullo_epi32(m_v, isum));
            }
            let sd_lo = _mm512_castsi512_si256(sd_zmm);
            let sd_hi = _mm512_extracti64x4_epi64::<1>(sd_zmm);
            let sd_ymm = _mm256_hadd_epi32(sd_lo, sd_hi); // pair-merge -> PERM order, once per block
            let sd_f = _mm256_cvtepi32_ps(sd_ymm);
            let sm_f = _mm256_cvtepi32_ps(sm_ymm);
            let d_v = _mm256_loadu_ps(d_p.add(b * 8));
            let dmin_v = _mm256_loadu_ps(dmin_p.add(b * 8));
            let t = _mm256_sub_ps(_mm256_mul_ps(d_v, sd_f), _mm256_mul_ps(dmin_v, sm_f));
            sumf_v = _mm256_add_ps(sumf_v, _mm256_mul_ps(_mm256_set1_ps(*qd_p.add(b)), t));
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sumf_v);
        const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
        for (j, &row) in PERM.iter().enumerate() {
            // SAFETY: row < 8, r < m, cols.len() == 8*m (asserted above).
            *cols_p.add(row * m + r) = lanes[j];
        }
    }
}

/// [`Q4kPackGroup`]'s Q6_K sibling: 16 sub-blocks of 16 (2 qwords each) instead of 8×32, int8
/// scales instead of the 6-bit sc/min pairs, and no min term — the `-32` offset rides the
/// activation `bsums16` at GEMM time (split as `-16·bsum` per pair lane, integer-exact).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Q6kPackGroup {
    ilv: Vec<u8>, // [nb * 2048]: [st 0..16][qw 0..2][row-interleaved 64 B], codes BIASED (q+32)
    sc16: Vec<i32>, // [nb * 256]: pair-duplicated int8 scales (lanes 2i, 2i+1 = row i)
    d: Vec<f32>,  // [nb * 8], PERM order
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Q6kPack {
    pub(crate) groups: Vec<Q6kPackGroup>,
    pub(crate) nb: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Q6kPack {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) fn bytes(&self) -> usize {
        self.groups
            .iter()
            .map(|g| g.ilv.len() + g.sc16.len() * 4 + g.d.len() * 4)
            .sum()
    }
}

/// Build the interleaved pack for a `[out_f, in_f]` Q6_K bank (e.g. the tied Q6_K lm_head —
/// ~740 MiB of expanded codes for gemma's 262k vocab, built once per session, rayon over groups).
/// `out_f % 8` tail rows are NOT packed.
#[cfg(target_arch = "x86_64")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn q6k_pack(wbytes: &[u8], in_f: usize, out_f: usize) -> Q6kPack {
    const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
    let nb = in_f / 256;
    let bpr = wbytes.len() / out_f;
    let n_groups = out_f / 8;
    use rayon::prelude::*;
    let groups: Vec<Q6kPackGroup> = (0..n_groups)
        .into_par_iter()
        .map(|g| {
            let mut pg = Q6kPackGroup {
                ilv: vec![0u8; nb * 2048],
                sc16: vec![0i32; nb * 256],
                d: vec![0f32; nb * 8],
            };
            let mut tmp = [[0u8; 256]; 8];
            let mut sc_rows = [[0i32; 16]; 8];
            let mut d_rows = [0f32; 8];
            for b in 0..nb {
                for i in 0..8 {
                    let row = &wbytes[(g * 8 + i) * bpr..(g * 8 + i) * bpr + bpr];
                    let blk = &row[b * 210..b * 210 + 210];
                    let ql = &blk[0..128];
                    let qh = &blk[128..192];
                    let scales = &blk[192..208];
                    d_rows[i] = rdf16(&blk[208..210]);
                    for (st, sv) in sc_rows[i].iter_mut().enumerate() {
                        *sv = scales[st] as i8 as i32;
                    }
                    // Linear element order, codes biased (q+32, 0..63) — the same mapping the
                    // batch kernels' flat expansion uses.
                    for half in 0..2usize {
                        let (qlo, qho, base) = (half * 64, half * 32, half * 128);
                        for l in 0..32 {
                            let f = &mut tmp[i];
                            f[base + l] = (ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4);
                            f[base + l + 32] =
                                (ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4);
                            f[base + l + 64] = (ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4);
                            f[base + l + 96] =
                                (ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4);
                        }
                    }
                }
                for (j, &row) in PERM.iter().enumerate() {
                    pg.d[b * 8 + j] = d_rows[row];
                }
                for st in 0..16usize {
                    for i in 0..8usize {
                        pg.sc16[b * 256 + st * 16 + 2 * i] = sc_rows[i][st];
                        pg.sc16[b * 256 + st * 16 + 2 * i + 1] = sc_rows[i][st];
                    }
                    for qw in 0..2usize {
                        let dst = &mut pg.ilv
                            [b * 2048 + st * 128 + qw * 64..b * 2048 + st * 128 + qw * 64 + 64];
                        for i in 0..8 {
                            dst[i * 8..i * 8 + 8]
                                .copy_from_slice(&tmp[i][st * 16 + qw * 8..st * 16 + qw * 8 + 8]);
                        }
                    }
                }
            }
            pg
        })
        .collect();
    Q6kPack { groups, nb }
}

/// The GEMM half over a prebuilt [`Q6kPackGroup`] — 8 output rows × all `m` activations. Same
/// integer core as `vec_dot_q6k_batch_vnni` (dpbusd on biased codes, `-32·bsum16` split as
/// `-16·bsum16` per pair lane — exact) and the same per-super-block f32 sequence
/// (`(d · q8.d) · isum`), so results are bit-identical to the batch/single kernels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vnni")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) unsafe fn q6k_gemm_group(pg: &Q6kPackGroup, nb: usize, q8s: &[Q8], cols: &mut [f32]) {
    use std::arch::x86_64::*;
    let m = q8s.len();
    assert_eq!(cols.len(), 8 * m);
    assert!(pg.ilv.len() >= nb * 2048 && pg.sc16.len() >= nb * 256 && pg.d.len() >= nb * 8);
    let ilv_p = pg.ilv.as_ptr();
    let sc16_p = pg.sc16.as_ptr();
    let d_p = pg.d.as_ptr();
    let cols_p = cols.as_mut_ptr();
    for r in 0..m {
        let q8 = &q8s[r];
        assert!(q8.qs.len() >= nb * 256 && q8.bsums16.len() >= nb * 16 && q8.d.len() >= nb);
        let qs_p = q8.qs.as_ptr();
        let bs16_p = q8.bsums16.as_ptr();
        let qd_p = q8.d.as_ptr();
        let mut sumf_v = _mm256_setzero_ps();
        for b in 0..nb {
            let mut sd_zmm = _mm512_setzero_si512();
            let q8b = qs_p.add(b * 256);
            for st in 0..16usize {
                let mut acc = _mm512_setzero_si512();
                for qw in 0..2usize {
                    let w = _mm512_loadu_si512(
                        ilv_p.add(b * 2048 + st * 128 + qw * 64) as *const __m512i
                    );
                    let a = _mm512_set1_epi64(
                        (q8b.add(st * 16 + qw * 8) as *const i64).read_unaligned(),
                    );
                    acc = _mm512_dpbusd_epi32(acc, w, a);
                }
                // -32·bsum16 per sub-block, split as -16·bsum16 per pair lane (exact:
                // (a1-16bs)+(a2-16bs) = dp-32bs).
                let corr = _mm512_set1_epi32(16 * *bs16_p.add(b * 16 + st));
                let sc16 = _mm512_loadu_si512(sc16_p.add(b * 256 + st * 16) as *const __m512i);
                sd_zmm = _mm512_add_epi32(
                    sd_zmm,
                    _mm512_mullo_epi32(_mm512_sub_epi32(acc, corr), sc16),
                );
            }
            let sd_lo = _mm512_castsi512_si256(sd_zmm);
            let sd_hi = _mm512_extracti64x4_epi64::<1>(sd_zmm);
            let sd_ymm = _mm256_hadd_epi32(sd_lo, sd_hi); // pair-merge -> PERM order
            let s_f = _mm256_cvtepi32_ps(sd_ymm);
            let d_v = _mm256_loadu_ps(d_p.add(b * 8));
            let t = _mm256_mul_ps(d_v, _mm256_set1_ps(*qd_p.add(b)));
            sumf_v = _mm256_add_ps(sumf_v, _mm256_mul_ps(t, s_f));
        }
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sumf_v);
        const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
        for (j, &row) in PERM.iter().enumerate() {
            // SAFETY: row < 8, r < m, cols.len() == 8*m (asserted above).
            *cols_p.add(row * m + r) = lanes[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #4: `cache_insert_if_absent` closes the check-then-insert race — the re-check under the lock
    // must (a) insert + count bytes exactly once for a fresh key, (b) NOT re-count or overwrite when
    // the key is already resident (the race loser gets the winner's Arc), and (c) respect the budget.
    #[test]
    fn cache_insert_if_absent_is_idempotent_and_budgeted() {
        let mut state: (HashMap<PackKey, Arc<u32>>, usize) = (HashMap::new(), 0);
        let budget = 100usize;

        // First insert of a fresh key: lands, counts its bytes.
        let win = cache_insert_if_absent(&mut state, (1, 2, 3), Arc::new(42), 10, budget);
        assert_eq!(*win, 42);
        assert_eq!(state.1, 10, "bytes counted once");
        assert_eq!(state.0.len(), 1);

        // Simulated race: a second builder reaches the lock with the SAME key but its own Arc. It
        // must get the RESIDENT Arc back (Arc::ptr_eq the winner) and must NOT double-count bytes.
        let resident = state.0.get(&(1, 2, 3)).unwrap().clone();
        let loser = cache_insert_if_absent(&mut state, (1, 2, 3), Arc::new(999), 10, budget);
        assert!(
            Arc::ptr_eq(&loser, &resident),
            "loser gets the resident pack"
        );
        assert_eq!(state.1, 10, "no double byte accounting on a re-check hit");
        assert_eq!(state.0.len(), 1);

        // A pack that would blow the budget is built-and-returned but NOT inserted/counted.
        let big = cache_insert_if_absent(&mut state, (4, 5, 6), Arc::new(7), 1_000, budget);
        assert_eq!(*big, 7, "over-budget pack still usable, just transient");
        assert_eq!(state.1, 10, "over-budget insert must not count bytes");
        assert_eq!(state.0.len(), 1, "over-budget key not inserted");
    }
}
