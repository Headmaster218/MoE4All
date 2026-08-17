//! Single-command-buffer forward recorder. Records many op dispatches (linear, rmsnorm,
//! rope, attention, silu_mul, add) into ONE command buffer with a conservative global
//! compute→compute barrier between each, then submits once on `finish()`. This collapses
//! the ~per-op submit+wait round-trips (the real bottleneck) to one per forward.
//!
//! Buffers are caller-owned `Box<dyn Buffer>` (from `Backend::alloc`); the recorder only
//! binds them. Reuses the cached kernels in `VulkanShared.kernels`.

use std::cell::RefCell;
use std::collections::HashSet;

use ash::vk;

use infr_core::{backend::Buffer, error::Result, pager_profile};

use super::ops::ComputeKernel;
use super::{as_vk_buf, be, Backing, VkBuffer, VulkanBackend};

/// Coopmat prefill shmem-staging mode for `attention_prefill_flash` (Levers 2 & 5,
/// kv-decode-perf-levers). `Off` = the default direct coopMatLoad from the bound descriptor.
/// `Stage` (INFR_FLASH_STAGE) stages the f16 K/V tile into shmem and coopMatLoads from there.
/// `Dequant(dt)` (INFR_FLASH_DEQUANT) decodes the quantized GGUF block cache `dt` to f16 in the
/// stage, skipping the dequant_kv_f16 prepass. Both route through `attn_flash_partial`'s staged
/// builds and force the split-K partial+combine path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlashStage {
    Off,
    Stage,
    Dequant(infr_core::DType),
}

/// State columns carried per subgroup by the token-serial DeltaNet scan; must match
/// `deltanet_seq.comp`'s `NCOL`.
///
/// Raising NCOL cuts the redundant k̂/q̂ cache re-reads (every column-workgroup of a head reads the
/// same row each token, ~1.07 GB/layer of L2 traffic at NCOL=1) and lets the NCOL reductions
/// pipeline — but it divides the wave count by NCOL, and this kernel is latency-bound, so the two
/// effects fight and the curve has an interior optimum. Measured, Ornith-35B pp512, scan total over
/// the 30 DeltaNet layers: NCOL=1 → 10.07ms (2048 waves), NCOL=2 → 8.29ms (1024), NCOL=4 → 9.42ms
/// (512), NCOL=8 → 14.06ms (256, register pressure). NOTE this is NOT a bandwidth-bound kernel: the
/// 4x traffic cut from NCOL=1→4 bought only ~6%, which is what ruled the bandwidth theory out.
pub(crate) const DN_SEQ_NCOL: usize = 2;

/// `matmul_mmq_experts`' BM=64 → BM=32 row-tile crossover, in average rows-per-expert
/// (`rows·n_used/n_expert`). Below this, the BM=32 `_xp32` kernel wins (less masked-tile waste);
/// at/above it BM=64 (default) wins. Picked from `moe_expert_grid_bound_bench`'s tile sweep —
/// bumped past qwen3.6-MoE's pp512 avg (~16/expert) but below qwen3-30B-A3B's (~32/expert), so a
/// deeper-context prefill that pushes avg rows/expert up for either model naturally slides back to
/// BM=64 as the segments fill in.
const MOE_EXPERT_SMALL_TILE_AVG_ROWS: usize = 24;

/// Max K for which the DEFAULT (BM=64) expert tile takes the BN=128 wide-N kernel (`_xp128`).
///
/// Wide-N halves the per-workgroup activation (As) staging and the workgroup count, but costs
/// occupancy (THREADS = (64/4)·(128/4) = 512 vs 256). Which way that trades depends on K, because
/// K sets how many k-block iterations the one-time staging is amortized over:
///
/// * SHALLOW k (the MoE `down` proj — k = n_ff_exp, 768 on qwen3-30B-A3B): the k-loop is short
///   (nblk = 24), staging is a large share of the kernel, and halving As traffic WINS —
///   expert_down 56.7ms → 50.0ms (−11.8%) at pp512.
/// * DEEP k (the `gate`/`up` projs — k = n_embd, 2048): nblk = 64 already amortizes staging, so
///   the As saving is small and the 512-thread occupancy hit dominates — expert_gateup
///   65.0ms → 69.2ms (+6.5%, a LOSS).
///
/// 1024 sits between the two (768 vs 2048) with room on both sides. The `small_tile` (BM=32)
/// path keeps its own unconditional n%128 wide-N rule — at 256 threads it has no occupancy
/// penalty to trade against, so it wants wide-N at every K.
const MOE_EXPERT_WIDE_BN_MAX_K: usize = 1024;

/// Dense A_GLOBAL warp-GEMM `matmul_native_f16a` (n128_ag, non-split-K family) BM=64 → BM=32
/// row-tile crossover, in real batched-prefill rows `m`. MTP verify's draft window runs a batched
/// prefill of the drafted suffix (m≈6-8 steady-state, growing up to ~m30-50 under the no-rewind
/// fallback's carried-forward unaccepted prefix) through every dense projection GEMM on the SAME
/// BM=64 tile prefill uses at m=512+ — at m=8 that's 87.5% masked waste (56 of 64 rows discarded
/// at the store). Below this threshold the BM=32 `_bm32` kernel variant wins (halves the waste,
/// same per-thread math as `matmul_mmq_experts`' `_xp32`); at/above it BM=64 (default, unchanged)
/// wins — picked from `dense_small_m_row_tile_bench`'s crossover sweep (BM=32 wins clearly through
/// m=24, a wash by m=32, a loss by m=64). `INFR_NO_SMALL_BM` forces BM=64 (A/B escape). Same
/// K-accumulation order either way — tile GRANULARITY only, bit-identical to the BM=64 path.
///
/// NOT applied to `matmul_native_splitk`'s sk_ag family (o/down/kv-proj at these same m — the
/// OTHER half of verify's dense GEMM traffic): the same bench shows a net LOSS there across the
/// whole m≈4-64 sweep — split-K's own `splits` dimension already fans the GEMM across enough
/// workgroups to fill the device, so a smaller row tile only adds fixed per-workgroup cost with
/// no occupancy win to pay for it (see `matmul_native_splitk`'s doc).
const DENSE_SMALL_TILE_MAX_M: usize = 24;

/// Dense A_GLOBAL warp-GEMM `matmul_native_f16a` (n128_ag family) BM=32 → BM=16 row-tile
/// crossover, in real batched-prefill rows `m`. BM=16 is one coopmat M-frag — the tiling floor —
/// so it halves BM=32's masked waste again (m=8 on BM=32 is 75% masked; on BM=16 it's 50%), but it
/// ALSO halves WN (doubling WARPS_N to keep all 8 launched warps mapped to a valid tile), so it's
/// a genuinely different, thinner-frag tile shape, not just "BM=32 again but smaller" — worth
/// measuring rather than assuming.
///
/// `bm16_crossover_bench` (gemm_bench.rs) swept m ∈ {4,6,8,12,16,24,32} on the real qwen35-4B-
/// UD-Q4_K_XL n128_ag verify shapes (gate_up: Q4_K k=2560 n=18432; vocab_head: Q6_K k=2560
/// n=248320) on the 7900 XTX (RADV). BM=16 wins clearly and by a growing margin through m=16 (the
/// single-M-tile regime: ceil(m/16)==1) — e.g. gate_up m=8: 70.3us vs BM=32's 89.5us (+27%); m=16:
/// 75.3us vs 90.8us (+21%). At m=24 (ceil(24/16)==2 — a SECOND M-tile appears) it falls off a
/// cliff: the weight-decode work (independent of M-tile, redone once per M-tile per column-tile)
/// now happens twice, and on the wide vocab_head weight (n=248320, decode-heavy) that alone makes
/// BM=16 77% SLOWER than BM=32 (2771us vs 1564us) despite the smaller per-tile masked waste. m=32
/// is a wash on both shapes (~same µs ± noise). So the crossover is exactly the BM=16 tiling floor
/// itself: profitable only while m stays within a single M-tile.
///
/// Capped at 16 (not `DENSE_SMALL_TILE_MAX_M`'s 24) for that reason: above the single-M-tile
/// regime (ceil(m/16) > 1) BM=16 redoes the weight-decode per extra M-tile and loses. Default-ON
/// (`INFR_NO_BM16` A/B escape → BM=32); a win on both 4B and 9B MTP (see the gate's comment). Only
/// applies within the small-tile band and never disturbs `INFR_NO_SMALL_BM`'s BM=64 override.
const DENSE_SMALL_TILE_MAX_M16: usize = 16;

/// Upper bound on the number of buffers any single recorded dispatch binds (the widest kernel
/// binds ≤ ~9). Sizes the stack scratch arrays in `dispatch3`/`dispatch_indirect`/`bind_descriptors`
/// so those hot recording paths never touch the heap; a `debug_assert` guards the bound.
const MAX_DISPATCH_BINDINGS: usize = 16;

/// Rows-per-workgroup for the multi-output-row decode GEMV, or `None` to keep the RM=1 path.
/// The RM variant packs RM output rows into one workgroup (RM× the in-flight weight streams per
/// wave) to feed enough MLP at low `out_f`, where the RM=1 grid (out_f workgroups) is too shallow
/// to hide DRAM latency — the big GEMVs (lm_head/gate+up, out_f ≥ ~16k) already saturate on RM=1
/// so they stay there. Only Q4_K/Q6_K have RM builds ([`crate::gemm::native_rm_streamed_build_spv`]).
/// Tunable for A/B: `INFR_NO_GEMV_RM` forces the RM=1 path; `INFR_GEMV_RM`=2|4 forces the factor;
/// `INFR_GEMV_RM_MAXOUT` overrides the out_f gate.
///
/// `k` is [`GemvCfg`](infr_core::config::GemvCfg) off the backend's `Config` — borrowed from the
/// recorder, never re-resolved (`docs/config-plan.md` R6). It replaced a `OnceLock<GemvKnobs>` that
/// re-read ~10 env vars process-once (§10.6): the memo is gone AND the value is now settable per
/// backend, at the same per-GEMV cost (a field read off a reference).
fn native_rm_choice(
    dtype: infr_core::DType,
    out_f: usize,
    k: &infr_core::config::GemvCfg,
) -> Option<u32> {
    use infr_core::DType::*;
    if !matches!(dtype, Q4K | Q6K) || k.no_rm {
        return None;
    }
    let max_out = k.rm_maxout;
    // Below ~2k outputs the RM=1 grid is already shallow; halving it starves the machine (k/v,
    // out_f=1024, regressed in the cold A/B), so those stay on RM=1.
    let min_out = k.rm_minout;
    if out_f > max_out || out_f < min_out {
        return None;
    }
    let rm = k.rm;
    (rm == 2 || rm == 4).then_some(rm)
}

/// NUM_ROWS for the reassociation-tolerant subgroup decode GEMV (`native_gemv_sg.comp`), or `None`
/// to keep the bit-identical tree/RM path. The subgroup form (wave32 + subgroupAdd, NR output rows)
/// keeps 32×NR weight-load requests in flight per wave — the extra MLP that the shallow low-out_f
/// projection grid (q/k/v/o/down) needs to saturate DRAM. It REORDERS the accumulation (not
/// bit-identical), so it is gated to the latency-starved projection BAND ONLY: the big lm_head /
/// gate+up GEMVs (out_f ≥ ~16k) already saturate on the tree kernel and subgroupAdd measured a ~20%
/// occupancy regression there. Q6_K ONLY (see below) — no Q4_K SG build exists.
/// Env: `INFR_NO_GEMV_SG` forces the tree/RM path; `INFR_GEMV_SG_NR`=2|4|8 overrides NR;
/// `INFR_GEMV_SG_MINOUT`/`INFR_GEMV_SG_MAXOUT` override the band.
fn native_sg_choice(
    dtype: infr_core::DType,
    in_f: usize,
    out_f: usize,
    k: &infr_core::config::GemvCfg,
) -> Option<u32> {
    use infr_core::DType::*;
    // Q6_K ONLY: the heavier Q6_K decode (210-byte super-blocks, more unpack ALU) is where wave32 +
    // subgroupAdd's lower register/barrier overhead nets out. On Q4_K the tree/RM kernel already
    // saturates and SG regressed at every measured shape (decode_gemv_bw A/B). Q6_K also loses below
    // out_f≈2048 (attn_v, out_f=1024: grid too shallow) and above the band (lm_head, out_f=151936:
    // already peak on the tree kernel + subgroupAdd occupancy regression) — so the band is the
    // out_f≈4096 Q6_K projections (ffn_down / o). Default NR=2 (best at every winning shape).
    if !matches!(dtype, Q6K) || k.no_sg {
        return None;
    }
    // wave32 lanes stride 32-elem sub-blocks; in_f must be a multiple of 32 (all projections are).
    if !in_f.is_multiple_of(32) {
        return None;
    }
    let min_out = k.sg_minout;
    let max_out = k.sg_maxout;
    if out_f < min_out || out_f > max_out {
        return None;
    }
    let nr = k.sg_nr;
    matches!(nr, 2 | 4 | 8).then_some(nr)
}

/// NUM_ROWS for the reassociation-tolerant subgroup id-indexed decode GEMV
/// (`native_gemv_id_multi_sg.comp`), or `None` to keep the tree id kernel. Same projection-band
/// discipline as [`native_sg_choice`], but covers BOTH Q6_K (Qwen3-30B expert down, out_f≈2048 — the
/// largest SG win of all decode shapes) AND Q5_K (Qwen3.6-A3B UD-quant stores most expert downs as
/// Q5_K at the same out_f≈2048 shape; A/B-confirmed a win — Q4_K idm already saturates and stays on
/// the tree). Kept independent of `native_sg_choice` so the byte-identical dense SG path is untouched.
/// Env: `INFR_NO_GEMV_ID_SG` forces the tree id path (id-only escape); `INFR_NO_GEMV_SG` disables BOTH
/// the dense and id SG routes; `INFR_GEMV_SG_MINOUT`/`MAXOUT`/`INFR_GEMV_SG_NR` are the shared knobs.
fn native_id_sg_choice(
    dtype: infr_core::DType,
    in_f: usize,
    out_f: usize,
    k: &infr_core::config::GemvCfg,
) -> Option<u32> {
    use infr_core::DType::*;
    if k.no_id_sg || k.no_sg {
        return None;
    }
    // Heavy K-quant decodes only (Q6_K/Q5_K): the extra unpack ALU is where wave32 + subgroupAdd's
    // lower register/barrier overhead nets out. Q4_K's tree id kernel already saturates (SG regressed).
    //
    // IQ3_S joins on the SAME band and the same NR=2 default: its expert DOWN projection is exactly
    // the out_f≈2048 shape the band was cut for, and the grid codebook makes it the heaviest unpack
    // of the set. NR rows/workgroup also amortizes `grid_init()`'s per-workgroup LDS staging of the
    // codebook, which the one-row-per-workgroup tree kernel pays in full for every output row —
    // measured on Qwen3.6-35B-A3B-UD-IQ3_S tg64: native_idm_iq3s 42.0 → 34.8ms, tg128 135.6 → 137.5.
    //
    // IQ2_S is deliberately NOT here. Its expert gate/up is the mirrored shape (in_f=2048,
    // out_f=512), and out_f=512 both falls under this band AND is where the SG route loses badly:
    // forced on (MINOUT=512), native_idm_iq2s went 49.8 → 117.2ms (NR=2) / 85.8ms (NR=4) / 86.8ms
    // (NR=8), dragging tg128 136 → 121-127. A 32-lane workgroup has to stage the same 8 KB table
    // with half the tree kernel's threads, and 8 KB of LDS on a single-wave workgroup collapses
    // occupancy — the staging amortization NR buys never pays that back. Eliminating the staging
    // (codebook in an L2-resident BUFFER instead of per-workgroup LDS) is the real fix for the
    // gate/up shape and is NOT what this tier does; ablating grid_init() outright measured
    // native_idm_iq2s 49.8 → 23.3ms, i.e. ~50ms of a 505ms decode still on the table.
    if !matches!(dtype, Q6K | Q5K | Iq3S) {
        return None;
    }
    // wave32 lanes stride 32-elem sub-blocks; in_f must be a multiple of 32 (all projections are).
    if !in_f.is_multiple_of(32) {
        return None;
    }
    let min_out = k.sg_minout;
    let max_out = k.sg_maxout;
    if out_f < min_out || out_f > max_out {
        return None;
    }
    let nr = k.sg_nr;
    matches!(nr, 2 | 4 | 8).then_some(nr)
}

/// Converts an element-space per-expert stride to the per-expert BYTE stride the STREAMED
/// (resident-BDA arena) id-indexed/expert-grid kernels apply on the 64-bit pointer (`w_addr +=
/// uint64_t(ex) * uint64_t(pc.m)` / `arena + uint64_t(ids[slot]) * uint64_t(stride)`) — see
/// `native_gemv_id.comp`'s and the `native_gemm_mmq_*.comp` family's `#elif defined(STREAMED)`
/// docs. `stride` is always a whole number of the dtype's GGUF blocks (weight strides never split
/// a block), so `stride_bytes = stride / block_elems * block_bytes` is exact.
fn expert_stride_bytes(dtype: infr_core::DType, stride: usize) -> u32 {
    let (block_elems, block_bytes) = infr_gguf::block_layout(dtype);
    debug_assert!(
        stride.is_multiple_of(block_elems),
        "expert stride {stride} is not a whole number of {dtype:?} blocks ({block_elems} elems/block)"
    );
    let bytes = (stride / block_elems) * block_bytes;
    // The kernels index this byte stride as u32 (`arena + uint64_t(ids[slot]) * uint64_t(stride)`
    // promotes to 64-bit, but the stride VALUE itself rides a u32 push-constant). A per-expert
    // slice >= 4 GiB would truncate here into a coherent-but-wrong pointer — the same class of
    // bug as the old u32 element-space multiply. `bda_weight_alloc`'s byte cap already rejects
    // any single arena tensor this large, so this only ever fires on a geometry bug upstream.
    debug_assert!(
        bytes < u32::MAX as usize,
        "per-expert byte stride {bytes} ({dtype:?}) exceeds the u32 addressing unit (4 GiB)"
    );
    bytes as u32
}

/// Reject a native-block dispatch whose K axis is off the 32-element sub-block grid.
///
/// Every `native_decode.glsl` consumer walks K as `nsub = in_f / 32` whole 32-element sub-blocks —
/// the `native_gemv*`, `native_mmv*` and `embed_gather` shader families. That integer division
/// turns an off-grid K into a wrong answer with nothing to notice it by. A ragged `in_f % 32`
/// drops the leftover elements from the dot product; `in_f < 32` runs ZERO iterations, and because
/// the GEMV kernels write their accumulator unconditionally, the dispatch completes cleanly and
/// stores an exact `0.0` over every output — measured, not inferred: a BF16 id-GEMV at `in_f = 8`
/// overwrote a `-7.0` sentinel with zeros while the host reference was ~3.1. No driver error, no
/// validation-layer message, no seam error, and no zero-init check can tell it from a real answer.
/// (`embed_gather` differs in shape only — it writes inside the sub-block loop, so a short row
/// leaves `dst` untouched instead — not in consequence.)
///
/// No GGUF reaches either case — quant block sizes are 32/64/256, and every float projection width
/// in a real model is 32-aligned — so this guards the floor rather than lifting it with a masked
/// tail block across the whole shader family. A hard `assert!` and not a `debug_assert!`: the
/// failure being guarded is a plausible-looking RELEASE-build result, which is exactly what a
/// debug-only check does not catch. Cost is two integer ops in front of a GPU dispatch.
#[track_caller]
fn assert_native_k(kernel: &str, in_f: usize) {
    assert!(
        in_f != 0 && in_f.is_multiple_of(32),
        "{kernel}: native-block K must be a nonzero multiple of 32 (got in_f={in_f}) — the \
         shaders loop `nsub = in_f / 32` whole sub-blocks, so an off-grid K silently drops the \
         tail, and in_f < 32 silently yields all-zero output"
    );
}

/// A batched-MoE expert-GEMM kernel variant: `(kernel name, SPIR-V)`.
type MmqKern = (&'static str, &'static [u32]);

/// ONE dtype-keyed source of truth for batched-MoE expert-GEMM kernel selection, driving BOTH the
/// resident (`_xp`) dispatch in [`Recorder::matmul_mmq_experts`] and the paged (`_xpg`) one in
/// [`Recorder::matmul_mmq_experts_paged`]. Each previously carried its own ~180-line dtype→kernel
/// match whose covered set and `unreachable!` arms had to be hand-kept in sync — a dtype added to
/// one but not the other would silently `unreachable!` at runtime. Here the two share this single
/// `match dtype`, so the set can't drift; the callers pick the variant (tile/wide) and apply the
/// paged +1 LUT binding. `resident_nb` is the resident binding count; paged is always
/// `resident_nb + 1`. `moe_mmq_kernels_match` (in `#[cfg(test)]`) freezes the whole mapping so a
/// refactor can't change any `(name, nb)` the two old tables produced.
struct MoeMmqDesc {
    /// Resident (`_xp`) STORAGE_BUFFER binding count; the paged (`_xpg`) build adds one for the LUT.
    resident_nb: usize,
    /// Resident non-wide: `_xp` (big BM=64 tile) / `_xp32` (small BM=32 tile).
    xp: MmqKern,
    xp32: MmqKern,
    /// Resident wide-BN (`_xp128` big-tile / `_xp32w` small-tile) — Q4_K/Q6_K only, else `None`.
    /// The caller only reads this when `wide_bn`, which it only sets for Q4_K/Q6_K.
    xp_wide: Option<(MmqKern, MmqKern)>,
    /// Paged: `_xpg` (big-tile) / `_xpg32` (small-tile).
    xpg: MmqKern,
    xpg32: MmqKern,
}

/// The single dtype→kernel table (see [`MoeMmqDesc`]). `None` for any dtype outside the mmq set.
fn moe_mmq_desc(dtype: infr_core::DType) -> Option<MoeMmqDesc> {
    use crate::gemm as g;
    use infr_core::DType as D;
    let d = match dtype {
        D::Q4K => MoeMmqDesc {
            resident_nb: 7,
            xp: ("native_gemm_mmq_q4k_xp", g::native_gemm_mmq_q4k_xp_spv()),
            xp32: (
                "native_gemm_mmq_q4k_xp32",
                g::native_gemm_mmq_q4k_xp32_spv(),
            ),
            xp_wide: Some((
                (
                    "native_gemm_mmq_q4k_xp128",
                    g::native_gemm_mmq_q4k_xp128_spv(),
                ),
                (
                    "native_gemm_mmq_q4k_xp32w",
                    g::native_gemm_mmq_q4k_xp32w_spv(),
                ),
            )),
            xpg: ("native_gemm_mmq_q4k_xpg", g::native_gemm_mmq_q4k_xpg_spv()),
            xpg32: (
                "native_gemm_mmq_q4k_xpg32",
                g::native_gemm_mmq_q4k_xpg32_spv(),
            ),
        },
        D::Q6K => MoeMmqDesc {
            resident_nb: 6,
            xp: ("native_gemm_mmq_q6k_xp", g::native_gemm_mmq_q6k_xp_spv()),
            xp32: (
                "native_gemm_mmq_q6k_xp32",
                g::native_gemm_mmq_q6k_xp32_spv(),
            ),
            xp_wide: Some((
                (
                    "native_gemm_mmq_q6k_xp128",
                    g::native_gemm_mmq_q6k_xp128_spv(),
                ),
                (
                    "native_gemm_mmq_q6k_xp32w",
                    g::native_gemm_mmq_q6k_xp32w_spv(),
                ),
            )),
            xpg: ("native_gemm_mmq_q6k_xpg", g::native_gemm_mmq_q6k_xpg_spv()),
            xpg32: (
                "native_gemm_mmq_q6k_xpg32",
                g::native_gemm_mmq_q6k_xpg32_spv(),
            ),
        },
        D::Q8_0 => MoeMmqDesc {
            resident_nb: 6,
            xp: ("native_gemm_mmq_q8_0_xp", g::native_gemm_mmq_q8_0_xp_spv()),
            xp32: (
                "native_gemm_mmq_q8_0_xp32",
                g::native_gemm_mmq_q8_0_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q8_0_xpg",
                g::native_gemm_mmq_q8_0_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q8_0_xpg32",
                g::native_gemm_mmq_q8_0_xpg32_spv(),
            ),
        },
        D::Q5_0 => MoeMmqDesc {
            resident_nb: 6,
            xp: ("native_gemm_mmq_q5_0_xp", g::native_gemm_mmq_q5_0_xp_spv()),
            xp32: (
                "native_gemm_mmq_q5_0_xp32",
                g::native_gemm_mmq_q5_0_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q5_0_xpg",
                g::native_gemm_mmq_q5_0_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q5_0_xpg32",
                g::native_gemm_mmq_q5_0_xpg32_spv(),
            ),
        },
        D::Q5K => MoeMmqDesc {
            resident_nb: 7,
            xp: ("native_gemm_mmq_q5k_xp", g::native_gemm_mmq_q5k_xp_spv()),
            xp32: (
                "native_gemm_mmq_q5k_xp32",
                g::native_gemm_mmq_q5k_xp32_spv(),
            ),
            xp_wide: None,
            xpg: ("native_gemm_mmq_q5k_xpg", g::native_gemm_mmq_q5k_xpg_spv()),
            xpg32: (
                "native_gemm_mmq_q5k_xpg32",
                g::native_gemm_mmq_q5k_xpg32_spv(),
            ),
        },
        D::Q5_1 => MoeMmqDesc {
            resident_nb: 7,
            xp: ("native_gemm_mmq_q5_1_xp", g::native_gemm_mmq_q5_1_xp_spv()),
            xp32: (
                "native_gemm_mmq_q5_1_xp32",
                g::native_gemm_mmq_q5_1_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q5_1_xpg",
                g::native_gemm_mmq_q5_1_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q5_1_xpg32",
                g::native_gemm_mmq_q5_1_xpg32_spv(),
            ),
        },
        D::Q2K => MoeMmqDesc {
            resident_nb: 6,
            xp: ("native_gemm_mmq_q2_k_xp", g::native_gemm_mmq_q2_k_xp_spv()),
            xp32: (
                "native_gemm_mmq_q2_k_xp32",
                g::native_gemm_mmq_q2_k_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q2_k_xpg",
                g::native_gemm_mmq_q2_k_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q2_k_xpg32",
                g::native_gemm_mmq_q2_k_xpg32_spv(),
            ),
        },
        D::Q3K => MoeMmqDesc {
            resident_nb: 6,
            xp: ("native_gemm_mmq_q3_k_xp", g::native_gemm_mmq_q3_k_xp_spv()),
            xp32: (
                "native_gemm_mmq_q3_k_xp32",
                g::native_gemm_mmq_q3_k_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q3_k_xpg",
                g::native_gemm_mmq_q3_k_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q3_k_xpg32",
                g::native_gemm_mmq_q3_k_xpg32_spv(),
            ),
        },
        D::Q4_0 => MoeMmqDesc {
            resident_nb: 6,
            xp: ("native_gemm_mmq_q4_0_xp", g::native_gemm_mmq_q4_0_xp_spv()),
            xp32: (
                "native_gemm_mmq_q4_0_xp32",
                g::native_gemm_mmq_q4_0_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q4_0_xpg",
                g::native_gemm_mmq_q4_0_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q4_0_xpg32",
                g::native_gemm_mmq_q4_0_xpg32_spv(),
            ),
        },
        D::Q4_1 => MoeMmqDesc {
            resident_nb: 7,
            xp: ("native_gemm_mmq_q4_1_xp", g::native_gemm_mmq_q4_1_xp_spv()),
            xp32: (
                "native_gemm_mmq_q4_1_xp32",
                g::native_gemm_mmq_q4_1_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q4_1_xpg",
                g::native_gemm_mmq_q4_1_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q4_1_xpg32",
                g::native_gemm_mmq_q4_1_xpg32_spv(),
            ),
        },
        D::Iq4Nl => MoeMmqDesc {
            resident_nb: 6,
            xp: (
                "native_gemm_mmq_iq4_nl_xp",
                g::native_gemm_mmq_iq4_nl_xp_spv(),
            ),
            xp32: (
                "native_gemm_mmq_iq4_nl_xp32",
                g::native_gemm_mmq_iq4_nl_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_iq4_nl_xpg",
                g::native_gemm_mmq_iq4_nl_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_iq4_nl_xpg32",
                g::native_gemm_mmq_iq4_nl_xpg32_spv(),
            ),
        },
        D::Iq4Xs => MoeMmqDesc {
            resident_nb: 6,
            xp: (
                "native_gemm_mmq_iq4_xs_xp",
                g::native_gemm_mmq_iq4_xs_xp_spv(),
            ),
            xp32: (
                "native_gemm_mmq_iq4_xs_xp32",
                g::native_gemm_mmq_iq4_xs_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_iq4_xs_xpg",
                g::native_gemm_mmq_iq4_xs_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_iq4_xs_xpg32",
                g::native_gemm_mmq_iq4_xs_xpg32_spv(),
            ),
        },
        D::Iq2S => MoeMmqDesc {
            resident_nb: 6,
            xp: (
                "native_gemm_mmq_iq2_s_xp",
                g::native_gemm_mmq_iq2_s_xp_spv(),
            ),
            xp32: (
                "native_gemm_mmq_iq2_s_xp32",
                g::native_gemm_mmq_iq2_s_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_iq2_s_xpg",
                g::native_gemm_mmq_iq2_s_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_iq2_s_xpg32",
                g::native_gemm_mmq_iq2_s_xpg32_spv(),
            ),
        },
        D::Iq3S => MoeMmqDesc {
            resident_nb: 6,
            xp: (
                "native_gemm_mmq_iq3_s_xp",
                g::native_gemm_mmq_iq3_s_xp_spv(),
            ),
            xp32: (
                "native_gemm_mmq_iq3_s_xp32",
                g::native_gemm_mmq_iq3_s_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_iq3_s_xpg",
                g::native_gemm_mmq_iq3_s_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_iq3_s_xpg32",
                g::native_gemm_mmq_iq3_s_xpg32_spv(),
            ),
        },
        D::Mxfp4 => MoeMmqDesc {
            resident_nb: 6,
            xp: (
                "native_gemm_mmq_mxfp4_xp",
                g::native_gemm_mmq_mxfp4_xp_spv(),
            ),
            xp32: (
                "native_gemm_mmq_mxfp4_xp32",
                g::native_gemm_mmq_mxfp4_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_mxfp4_xpg",
                g::native_gemm_mmq_mxfp4_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_mxfp4_xpg32",
                g::native_gemm_mmq_mxfp4_xpg32_spv(),
            ),
        },
        D::Nvfp4 => MoeMmqDesc {
            resident_nb: 6,
            xp: (
                "native_gemm_mmq_nvfp4_xp",
                g::native_gemm_mmq_nvfp4_xp_spv(),
            ),
            xp32: (
                "native_gemm_mmq_nvfp4_xp32",
                g::native_gemm_mmq_nvfp4_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_nvfp4_xpg",
                g::native_gemm_mmq_nvfp4_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_nvfp4_xpg32",
                g::native_gemm_mmq_nvfp4_xpg32_spv(),
            ),
        },
        D::Q2_0 => MoeMmqDesc {
            resident_nb: 6,
            xp: ("native_gemm_mmq_q2_0_xp", g::native_gemm_mmq_q2_0_xp_spv()),
            xp32: (
                "native_gemm_mmq_q2_0_xp32",
                g::native_gemm_mmq_q2_0_xp32_spv(),
            ),
            xp_wide: None,
            xpg: (
                "native_gemm_mmq_q2_0_xpg",
                g::native_gemm_mmq_q2_0_xpg_spv(),
            ),
            xpg32: (
                "native_gemm_mmq_q2_0_xpg32",
                g::native_gemm_mmq_q2_0_xpg32_spv(),
            ),
        },
        _ => return None,
    };
    Some(d)
}

/// The u32 addressing-unit caps a single dense weight (or one output-row chunk of it) must stay
/// under: `< 2^32` ELEMENTS (in-kernel element/word indices are u32 — see native_weight_addr.glsl)
/// AND `< 2^32` BYTES (the `wi << 2` word→byte offset inside `arena_word` is also u32). Chunked
/// dispatch (see [`Recorder::dispatch_gemv_chunked`]) splits an over-cap dense output projection
/// (lm_head / embed) into output-row ranges each strictly under BOTH — the DISPATCH-level answer to
/// a `>= 2^32`-element tensor that does NOT widen any in-kernel index (widening per-element indices
/// to u64 was measured at a 2.4x RDNA3 regression — see the resident-BDA campaign). Both default to
/// `2^32`; `INFR_BDA_CHUNK_ELEMS` / `INFR_BDA_CHUNK_BYTES` (`kernels.vulkan.bda_chunk_{elems,bytes}`)
/// lower them to FORCE a normal model's lm_head/embed to split so the always-1-chunk path can be
/// proven bitwise-identical to the split one on real hardware.
///
/// **S5b:** this pair used to be two `AtomicU64` cells seeded from the environment on first read
/// (`cap_from_env`) — a memo, and therefore unsettable from a test (§10.6). The caps are now
/// resolved ONCE per `Recorder` at construction (`Recorder::bda_chunk_elem_cap` /
/// `bda_chunk_byte_cap` fields, hoisted from the backend's `Config`) and passed down, so nothing
/// reads the environment and nothing latches process-wide.
const BDA_CHUNK_UNIT_MAX: u64 = 1 << 32;

/// `(elem_cap, byte_cap)` from a backend's Vulkan config. `None` (unset, or a value below 2, which
/// can never fit even one row) = the real u32 cap — exactly what `cap_from_env`'s
/// `.filter(|&n| n >= 2).unwrap_or(BDA_CHUNK_UNIT_MAX)` did.
pub(crate) fn bda_chunk_caps(vk: &infr_core::config::VulkanCfg) -> (u64, u64) {
    (
        vk.bda_chunk_elems.unwrap_or(BDA_CHUNK_UNIT_MAX),
        vk.bda_chunk_bytes.unwrap_or(BDA_CHUNK_UNIT_MAX),
    )
}

// In-process cap override for the chunked-vs-unchunked GPU parity test (recorder `mod tests`).
// Scoped to the calling thread (recording is synchronous on the test thread), so it is race-free
// under parallel test execution and cannot be seen by any other test. NOT the environment and NOT
// a config layer: the test drives ONE backend and has to flip the caps between two recordings on
// it, which a per-backend `Config` cannot express. Compiled only under `cfg(test)`.
#[cfg(test)]
thread_local! {
    static TEST_CHUNK_CAP: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
fn test_chunk_cap_override() -> Option<(u64, u64)> {
    TEST_CHUNK_CAP.with(|c| c.get())
}
#[cfg(test)]
fn with_bda_chunk_caps<R>(elem_cap: u64, byte_cap: u64, f: impl FnOnce() -> R) -> R {
    let prev = TEST_CHUNK_CAP.with(|c| c.replace(Some((elem_cap, byte_cap))));
    let r = f();
    TEST_CHUNK_CAP.with(|c| c.set(prev));
    r
}

/// Whether a `[out_f, in_f]` dense weight read at within-tensor element offset `w_base` breaches
/// either u32 addressing cap and therefore needs chunked dispatch. Cheap (two multiplies and two
/// comparisons against already-resolved caps, no allocation, no env lookup) so it sits on the decode
/// hot path in front of the LITERAL original single dispatch — the always-on chunk loop runs exactly
/// one iteration (this returns `false`) for every current model.
fn bda_gemv_needs_chunk(
    dtype: infr_core::DType,
    in_f: usize,
    out_f: usize,
    w_base: usize,
    caps: (u64, u64),
) -> bool {
    let (block_elems, block_bytes) = infr_gguf::block_layout(dtype);
    let whole_elems = w_base as u64 + (out_f as u64) * (in_f as u64);
    // Row bytes are exact only when a row is a whole number of blocks (always true for a real weight
    // row); if it somehow isn't, fall back to the element cap alone rather than mis-estimate bytes.
    let whole_bytes = if in_f.is_multiple_of(block_elems) {
        (out_f as u64) * ((in_f / block_elems) as u64) * (block_bytes as u64)
    } else {
        0
    };
    whole_elems >= caps.0 || whole_bytes >= caps.1
}

/// Whether a `[out_f, in_f]` dense weight breaches a u32 addressing cap (at slice offset 0). The
/// tiled-GEMM prefill tiers (coopmat / nc_mmq / nc_fma / streamed_prefill_gemm) cannot be
/// output-row chunked the way the GEMV can (their 64-row M-tile is not decomposable into single
/// input rows), so a breaching weight on those paths — a multi-row lm_head, i.e. MTP speculative
/// verify or an all-position-logits request on a big-vocab model — must fail LOUDLY rather than
/// wrap. The adapter guards the GEMM dispatch with this. (Decode m=1 and the per-row GEMV ARE
/// chunk-covered — see [`Recorder::dispatch_gemv_chunked`].)
pub(crate) fn bda_weight_breaches(
    dtype: infr_core::DType,
    in_f: usize,
    out_f: usize,
    vk: &infr_core::config::VulkanCfg,
) -> bool {
    #[cfg(test)]
    if let Some(caps) = test_chunk_cap_override() {
        return bda_gemv_needs_chunk(dtype, in_f, out_f, 0, caps);
    }
    bda_gemv_needs_chunk(dtype, in_f, out_f, 0, bda_chunk_caps(vk))
}

/// The maximum output-row count of a `[out_f, in_f]` weight per addressing unit. See [`OutRowChunk`].
const BDA_CHUNK_ROW_ALIGN: usize = 64;

/// One contiguous run of output rows (weight rows / vocab logits) that a chunked dense GEMV
/// dispatches together, plus the BYTE offset of its first row from the tensor's base address (added
/// to the 64-bit arena base so the kernel's per-output element index stays chunk-local and u32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutRowChunk {
    /// First output row (element index into `out_f`) — always a multiple of [`BDA_CHUNK_ROW_ALIGN`]
    /// so the f32 output/activation binding offset `o0 * 4` meets any device's
    /// `minStorageBufferOffsetAlignment` (Vulkan spec max 256 bytes).
    o0: usize,
    /// Number of output rows in this chunk.
    n: usize,
    /// Byte offset of row `o0` from the tensor base (`chunk_base = tensor.device_addr() + byte_off`).
    byte_off: u64,
}

/// Split the `out_f` output rows of a `[out_f, in_f]` dense weight into dispatch chunks each
/// strictly under BOTH the element cap (`elem_cap`) and the byte cap (`byte_cap`). Returns a single
/// whole-tensor chunk (`o0=0, n=out_f, byte_off=0`) whenever the tensor already fits — the
/// zero-cost path for every current model. Chunk starts are [`BDA_CHUNK_ROW_ALIGN`]-aligned.
///
/// Errors (never loops forever) when a SINGLE output row already breaches a cap — `in_f >= elem_cap`
/// or one row `>= byte_cap`, or a forced cap so small it can't hold even one aligned group of rows.
/// A nonzero `w_base` (a fused-QKV slice offset) on a breaching tensor is rejected: the chunk base
/// folds into the 64-bit arena address, leaving `w_base` for the shader's per-output index, and
/// mixing a breaching tensor with a nonzero slice offset is not wired (no such tensor exists — only
/// lm_head/embed breach, and they carry `w_base == 0`).
fn bda_out_f_chunks(
    dtype: infr_core::DType,
    in_f: usize,
    out_f: usize,
    w_base: usize,
    elem_cap: u64,
    byte_cap: u64,
) -> std::result::Result<Vec<OutRowChunk>, String> {
    let (block_elems, block_bytes) = infr_gguf::block_layout(dtype);
    if !in_f.is_multiple_of(block_elems) {
        return Err(format!(
            "chunk-split: in_f {in_f} is not a whole number of {dtype:?} blocks ({block_elems})"
        ));
    }
    let row_elems = in_f as u64;
    let row_bytes = ((in_f / block_elems) * block_bytes) as u64;
    let whole_elems = w_base as u64 + (out_f as u64) * row_elems;
    let whole_bytes = (out_f as u64) * row_bytes;
    if whole_elems < elem_cap && whole_bytes < byte_cap {
        return Ok(vec![OutRowChunk {
            o0: 0,
            n: out_f,
            byte_off: 0,
        }]);
    }
    if w_base != 0 {
        return Err(format!(
            "chunk-split: a breaching weight with a nonzero slice offset (w_base={w_base}) is \
             unsupported — the chunk base folds into the 64-bit arena address"
        ));
    }
    // Rows per chunk under each cap (strict `<`), then floored to the row alignment so every chunk
    // start keeps the output/x binding offset device-aligned. A zero row size (impossible for a
    // real weight — guarded only so the division can't trap) imposes no limit.
    let by_elem = (elem_cap - 1).checked_div(row_elems).unwrap_or(u64::MAX);
    let by_byte = (byte_cap - 1).checked_div(row_bytes).unwrap_or(u64::MAX);
    let max_rows = (by_elem.min(by_byte) as usize) / BDA_CHUNK_ROW_ALIGN * BDA_CHUNK_ROW_ALIGN;
    if max_rows == 0 {
        return Err(format!(
            "chunk-split: a single dense output row (in_f={in_f}, {row_bytes} bytes, {dtype:?}) \
             cannot be split under the addressing caps (elems {elem_cap}, bytes {byte_cap}) — this \
             tensor needs a wider addressing scheme, not chunking"
        ));
    }
    let mut chunks = Vec::with_capacity(out_f.div_ceil(max_rows));
    let mut o0 = 0usize;
    while o0 < out_f {
        chunks.push(OutRowChunk {
            o0,
            n: max_rows.min(out_f - o0),
            byte_off: (o0 as u64) * row_bytes,
        });
        o0 += max_rows;
    }
    Ok(chunks)
}

pub struct Recorder<'a> {
    be: &'a VulkanBackend,
    cmd: vk::CommandBuffer,
    /// Descriptor pools, GROWN on exhaustion (a batched-MoE prefill chunk records ~50k dispatches
    /// — far beyond any fixed max_sets). The last entry is the active pool; `alloc_set` appends a
    /// fresh one on ERROR_OUT_OF_POOL_MEMORY.
    pools: std::cell::RefCell<Vec<vk::DescriptorPool>>,
    /// Buffers written since the last barrier (for read-after-write / write-after-write detection).
    dirty_writes: RefCell<HashSet<vk::Buffer>>,
    /// Buffers read since the last barrier (for write-after-read detection).
    dirty_reads: RefCell<HashSet<vk::Buffer>>,
    /// Whether any un-barriered write was produced by a transfer (copy) rather than a shader.
    dirty_transfer: std::cell::Cell<bool>,
    barriers: RefCell<usize>,
    /// Set while recording an indirect dispatch so `sync` widens the barrier to cover the
    /// indirect-command read of GPU-written dispatch args.
    indirect_pending: std::cell::Cell<bool>,
    /// Debug knobs, resolved ONCE at construction from the backend's `Config` — `debug.no_barrier`,
    /// `debug.full_barrier`, `prof.stages` (S5b). Per-dispatch paths read these fields, never a
    /// config tree walk and never the environment.
    no_barrier: bool,
    full_barrier: bool,
    stages: bool,
    /// Per-op GPU timestamp profiling (INFR_PROF_OPS): a timestamp before EVERY dispatch — labeled
    /// automatically with the dispatched kernel's name at the `dispatch3`/`dispatch_indirect`
    /// chokepoints — plus one at finish. No manual stamp calls; aggregation by label happens at
    /// report time.
    prof_ops: bool,
    /// INFR_PROF_OP_SHAPES (on top of INFR_PROF_OPS): GEMV/GEMM dispatches get shape-itemized labels
    /// (`kind:mM:KxN`) instead of the bare kernel name. Read once at construction.
    op_shapes: bool,
    query_pool: vk::QueryPool,
    ts_labels: RefCell<Vec<&'static str>>,
    /// Dispatches past the query-pool capacity: counted (and reported) instead of stamped.
    ts_dropped: std::cell::Cell<usize>,
    /// One-shot prof_ops label override: the NEXT dispatch's timestamp uses this instead of the
    /// kernel name, then clears it. Lets a caller attribute a generic kernel (e.g. the lm_head
    /// GEMM vs a projection, expert gate_up vs down) to a distinct bucket without per-op API
    /// plumbing.
    next_label: std::cell::Cell<Option<&'static str>>,
    /// Record-once: when set, the command buffer is begun resubmittable (no ONE_TIME_SUBMIT) and
    /// `finish_record` hands back its cmd buffer + pool (a [`RecordedCmd`]) instead of submitting and
    /// freeing — so the caller can replay it across tokens.
    persistent: bool,
    /// Creation time — INFR_PROF prints the host record time vs the submit+GPU wait in `finish`.
    t0: std::time::Instant,
    /// See [`Self::suppress_sync`]: while set, `sync` accumulates hazards without emitting.
    suppress: std::cell::Cell<bool>,
    /// Dispatches recorded into THIS command buffer so far (counted at the `stamp` chokepoint).
    /// The submit splitter reads it to decide when the segment has grown to the device's cap —
    /// see `VulkanShared::submit_dispatch_cap` and `adapter::execute_static`.
    dispatches: std::cell::Cell<usize>,
    /// `kernels.vulkan.bda_chunk_elems` / `_bytes`, HOISTED here at construction (§10.6 — they used
    /// to be two env-seeded `AtomicU64` memos). Read on the decode GEMV path via
    /// [`Self::chunk_caps`].
    bda_chunk_elem_cap: u64,
    bda_chunk_byte_cap: u64,
}

/// Intern a profiling shape label, leaking each DISTINCT string at most ONCE and handing back the
/// same `&'static str` for every later occurrence.
///
/// WHY AN INTERNER AND NOT `Box::leak`. The timestamp-label plumbing wants a `&'static str`, and
/// the obvious way to satisfy that from a `format!` is `Box::leak`. But the shape itemizers that
/// build these labels ([`Recorder::label_gemv`] / [`Recorder::label_gemm`]) run at RECORD time,
/// not at graph-build time: `execute_static` re-walks `for op in &graph.ops` and re-records every
/// dispatch on EVERY execute. A per-call `Box::leak` therefore leaked one small string per op per
/// forward pass — on a ~400-op graph at 50 tok/s that is ~20k leaked strings a second, on the
/// order of tens of MB per minute, growing without bound for as long as the profile runs. The
/// `next_label.get().is_none()` guard did not help: it only skips labels that an explicit
/// `label_next` already claimed, it is not a shape registry.
///
/// The set of DISTINCT shape strings a model produces is tiny and fixed — a few dozen, one per
/// (kernel, m, k, n) combination in the graph — so interning turns an unbounded per-token leak
/// into a bounded one-time cost, while keeping the `&'static str` the label plumbing needs.
///
/// The `Mutex` is fine here: this function is only ever reached under INFR_PROF_OP_SHAPES. The
/// non-profiling path never calls it, never formats a string, and never touches this lock —
/// callers check `self.op_shapes` first. A poisoned lock is recovered rather than propagated: a
/// panic elsewhere must not turn profiling into a second, unrelated crash.
fn intern_label(s: String) -> &'static str {
    static INTERN: std::sync::LazyLock<std::sync::Mutex<HashSet<&'static str>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));
    let mut set = INTERN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&hit) = set.get(s.as_str()) {
        return hit;
    }
    let leaked: &'static str = Box::leak(s.into_boxed_str());
    set.insert(leaked);
    leaked
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<'a> Recorder<'a> {
    pub(crate) fn new(backend: &'a VulkanBackend) -> Result<Self> {
        Self::new_inner(backend, false)
    }

    /// A recorder whose command buffer is resubmittable (no ONE_TIME_SUBMIT). `finish_record` returns
    /// a [`RecordedCmd`] the caller can replay instead of re-recording. Profiling is disabled on this
    /// path (the recorder is gone after recording, so per-replay timestamps can't be reported).
    pub(crate) fn new_persistent(backend: &'a VulkanBackend) -> Result<Self> {
        Self::new_inner(backend, true)
    }

    fn new_inner(backend: &'a VulkanBackend, persistent: bool) -> Result<Self> {
        let device = &backend.shared.device;
        let cmd_pool = *backend.shared.cmd_pool.lock().unwrap();
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| be(format!("alloc cmd buffer: {e}")))?[0];
        let begin_flags = if persistent {
            // SIMULTANEOUS_USE: the chained decode submits the SAME recorded command buffer n
            // times in one vkQueueSubmit (see RecordedCmd::replay_n) — the buffer must be legal
            // in the pending state more than once.
            vk::CommandBufferUsageFlags::SIMULTANEOUS_USE
        } else {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        };
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(begin_flags),
            )
        }
        .map_err(|e| be(format!("begin cmd buffer: {e}")))?;

        // Big pool: one descriptor set per recorded op.
        let pool = Self::new_desc_pool(device)?;

        // Per-DISPATCH stamps (auto-labels): a batched prefill chunk can record tens of
        // thousands of dispatches; 64k × 8 B = 512 KiB of query pool, profiling runs only.
        // Overflow is counted (`ts_dropped`) and reported, never UB.
        const MAX_TS: u32 = Recorder::<'_>::MAX_TS;
        // No per-op profiling on the persistent (replayed) path — the recorder is dropped after
        // recording, so it can't report timestamps for replays.
        // `profiling_suppressed` is the non-env suppression path (see `infr_prof_rt`): benches /
        // warmups toggle it around untimed work instead of mutating the `INFR_PROF_OPS` env var.
        let profcfg = &backend.cfg().prof;
        let dbgcfg = &backend.cfg().debug;
        let (bda_chunk_elem_cap, bda_chunk_byte_cap) =
            bda_chunk_caps(&backend.cfg().kernels.vulkan);
        let prof_ops = infr_core::prof::enabled(profcfg) && !persistent;
        let op_shapes = prof_ops && profcfg.op_shapes;
        let query_pool = if prof_ops {
            let qp = unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(MAX_TS),
                    None,
                )
            }
            .map_err(|e| be(format!("create query pool: {e}")))?;
            unsafe { device.cmd_reset_query_pool(cmd, qp, 0, MAX_TS) };
            qp
        } else {
            vk::QueryPool::null()
        };

        Ok(Self {
            be: backend,
            cmd,
            pools: std::cell::RefCell::new(vec![pool]),
            dirty_writes: RefCell::new(HashSet::new()),
            dirty_reads: RefCell::new(HashSet::new()),
            dirty_transfer: std::cell::Cell::new(false),
            barriers: RefCell::new(0),
            indirect_pending: std::cell::Cell::new(false),
            no_barrier: dbgcfg.no_barrier,
            full_barrier: dbgcfg.full_barrier,
            stages: profcfg.stages,
            prof_ops,
            op_shapes,
            query_pool,
            ts_labels: RefCell::new(Vec::new()),
            ts_dropped: std::cell::Cell::new(0),
            next_label: std::cell::Cell::new(None),
            persistent,
            t0: std::time::Instant::now(),
            suppress: std::cell::Cell::new(false),
            dispatches: std::cell::Cell::new(0),
            bda_chunk_elem_cap,
            bda_chunk_byte_cap,
        })
    }

    /// The Vulkan kernel-tier config this recorder steers on — BORROWED off the backend that
    /// created it, never cloned and never re-resolved (`docs/config-plan.md` R6). Every per-op /
    /// per-dispatch knob below reads through here instead of `std::env::var` (S5b).
    #[inline]
    pub(crate) fn vk(&self) -> &infr_core::config::VulkanCfg {
        &self.be.cfg().kernels.vulkan
    }

    /// The native-GEMV routing sub-config (`INFR_GEMV_*` / `INFR_NO_GEMV_*`), borrowed.
    #[inline]
    fn gemv(&self) -> &infr_core::config::GemvCfg {
        &self.vk().gemv
    }

    /// The `(elem, byte)` u32 addressing caps for chunked dense GEMV dispatch — the two values
    /// hoisted at construction. The `cfg(test)` arm is the chunk-parity test's thread-local
    /// override (see [`with_bda_chunk_caps`]); production compiles to two field reads.
    #[inline]
    fn chunk_caps(&self) -> (u64, u64) {
        #[cfg(test)]
        if let Some(caps) = test_chunk_cap_override() {
            return caps;
        }
        (self.bda_chunk_elem_cap, self.bda_chunk_byte_cap)
    }

    /// Dispatches recorded into this command buffer so far — the submit splitter's trigger.
    pub fn dispatches(&self) -> usize {
        self.dispatches.get()
    }

    /// Disjoint-batch barrier suppression: while ON, recorded dispatches accumulate hazard state
    /// but emit NO pipeline barriers. For batches whose members are KNOWN to touch disjoint
    /// regions of the same buffers (the batched-MoE per-expert stage loop — 128 experts' gathers
    /// all write `xe`, each at its routed offset). Leave the batch's FIRST dispatch unsuppressed
    /// so the stage orders after its producers, and turn suppression OFF after the batch (the
    /// next normal dispatch then fences the whole batch with ONE barrier).
    pub fn suppress_sync(&self, on: bool) {
        self.suppress.set(on);
    }

    /// Override the label of the NEXT dispatch's timestamp (INFR_PROF_OPS). Consumed once — a
    /// pre-set override wins over the automatic kernel-name label AND over the shape itemizers
    /// below. No-op without prof_ops.
    pub fn label_next(&self, label: &'static str) {
        if self.prof_ops {
            self.next_label.set(Some(label));
        }
    }

    /// Shape-itemize the next GEMV-class dispatch: with INFR_PROF_OP_SHAPES set (on top of
    /// INFR_PROF_OPS), its label becomes `kind:mM:in_fxout_f` (e.g. `mmvr:m4:1536x24576`) instead of
    /// the bare kernel name, splitting the bucket per route and per projection shape. This is
    /// what made the E2B pp4 63%-GEMV attribution actionable (llama.cpp's perf logger itemizes
    /// per shape; a flat bucket didn't). A pending `label_next` override wins. Profiling-only:
    /// without `op_shapes` this does nothing at all — no format, no lock, no allocation.
    fn label_gemv(&self, kind: &'static str, rows: usize, in_f: usize, out_f: usize) {
        if self.op_shapes && self.next_label.get().is_none() {
            let s = intern_label(format!("{kind}:m{rows}:{in_f}x{out_f}"));
            self.next_label.set(Some(s));
        }
    }

    /// Shape-itemize the next GEMM-class dispatch (`kernel:mM:kxn` under INFR_PROF_OP_SHAPES —
    /// same rationale as [`label_gemv`](Self::label_gemv)). Profiling-only; zero work when
    /// `op_shapes` is off.
    fn label_gemm(&self, kernel: &'static str, m: usize, k: usize, n: usize) {
        if self.op_shapes && self.next_label.get().is_none() {
            let s = intern_label(format!("{kernel}:m{m}:{k}x{n}"));
            self.next_label.set(Some(s));
        }
    }

    /// Sets per descriptor-pool tranche.
    const DESC_POOL_MAX_SETS: u32 = 4096;
    /// Upper bound on STORAGE_BUFFER bindings any single recorded dispatch uses (the widest kernel
    /// binds ≤ ~9; 12 leaves headroom). The tranche is sized so its sets and its descriptors
    /// exhaust TOGETHER: a set-count-limited pool with fewer descriptors (the old
    /// `descriptor_count: 16384` against `max_sets: 4096` = 4 descriptors/set) ran dry on
    /// descriptors at ~2048 heavily-bound sets — half its set capacity — doubling how often
    /// `alloc_set` has to grow the chain.
    const DESC_POOL_MAX_BINDINGS_PER_SET: u32 = 12;

    /// Create one descriptor pool tranche (the chain grows by these on exhaustion).
    fn new_desc_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: Self::DESC_POOL_MAX_SETS * Self::DESC_POOL_MAX_BINDINGS_PER_SET,
        }];
        unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(Self::DESC_POOL_MAX_SETS)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| be(format!("create recorder pool: {e}")))
    }

    /// Allocate a descriptor set from the active pool, growing the chain when it runs dry.
    fn alloc_set(&self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let device = &self.be.shared.device;
        let try_alloc = |pool: vk::DescriptorPool| unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&layout)),
            )
        };
        let cur = *self.pools.borrow().last().expect("≥1 descriptor pool");
        match try_alloc(cur) {
            Ok(sets) => sets[0],
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                let fresh = Self::new_desc_pool(device).expect("grow descriptor pool");
                self.pools.borrow_mut().push(fresh);
                try_alloc(fresh).expect("alloc descriptor set (fresh pool)")[0]
            }
            Err(e) => panic!("alloc descriptor set: {e}"),
        }
    }

    /// Query-pool capacity (timestamps per recorder). Stamps past `MAX_TS - 1` (one slot is
    /// reserved for the closing timestamp in `finish`) are dropped and counted.
    const MAX_TS: u32 = 65536;

    /// Record a profiling timestamp (BOTTOM_OF_PIPE) tagged with an op label, if INFR_PROF_OPS.
    /// Called automatically at the dispatch chokepoints with the kernel name — op methods no
    /// longer stamp by hand (use [`label_next`](Self::label_next) to override a too-generic
    /// kernel name).
    fn stamp(&self, label: &'static str) {
        // Every dispatch funnels through here, so this is also where the recorder counts them —
        // the submit splitter (`Recorder::dispatches`) needs the count whether or not INFR_PROF_OPS
        // is on.
        self.dispatches.set(self.dispatches.get() + 1);
        if !self.prof_ops {
            return;
        }
        let label = self.next_label.take().unwrap_or(label);
        let idx = self.ts_labels.borrow().len() as u32;
        if idx >= Self::MAX_TS - 1 {
            self.ts_dropped.set(self.ts_dropped.get() + 1);
            return;
        }
        unsafe {
            self.be.shared.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                idx,
            );
        }
        self.ts_labels.borrow_mut().push(label);
    }

    /// Emit a global compute/transfer barrier only if `reads`/`writes` collide with work recorded
    /// since the last barrier (RAW / WAR / WAW). Independent dispatches then overlap on the GPU.
    /// `dst_transfer` = this op consumes via a transfer (copy) rather than a compute shader.
    /// Returns once any required barrier is recorded; updates the hazard-tracking sets.
    fn sync(&self, reads: &[vk::Buffer], writes: &[vk::Buffer], dst_transfer: bool) {
        if self.no_barrier {
            return;
        }
        // Suppressed (disjoint-batch) dispatch: accumulate the hazard state but emit NO barrier —
        // the caller guarantees this dispatch touches only regions disjoint from the batch's other
        // members (per-expert MoE stage loop). The batch's first dispatch runs unsuppressed, so
        // the stage as a whole still orders after its producers; the NEXT unsuppressed dispatch
        // sees the accumulated dirty state and fences the whole batch at once.
        if self.suppress.get() {
            self.dirty_reads.borrow_mut().extend(reads.iter().copied());
            self.dirty_writes
                .borrow_mut()
                .extend(writes.iter().copied());
            return;
        }
        let dw = self.dirty_writes.borrow();
        let dr = self.dirty_reads.borrow();
        let raw = self.full_barrier || reads.iter().any(|b| dw.contains(b));
        let waw = writes.iter().any(|b| dw.contains(b));
        let war = writes.iter().any(|b| dr.contains(b));
        drop(dw);
        drop(dr);
        if raw || waw || war {
            // Access flags must only name flags supported by a stage present in the paired stage
            // mask (VUID-vkCmdPipelineBarrier-srcAccessMask-02815 / -dstAccessMask-02816) — e.g.
            // TRANSFER_WRITE/TRANSFER_READ are only valid alongside PipelineStageFlags::TRANSFER.
            // The old code set the access masks unconditionally (always including the TRANSFER
            // bits) while only conditionally widening the STAGE masks, so a pure compute→compute
            // barrier (the common case) claimed TRANSFER_WRITE/TRANSFER_READ access against a
            // COMPUTE_SHADER-only stage mask — a real spec violation RADV tolerates but Intel's
            // Mesa ANV validation layer (and, per spec, any conformant driver) rejects. Build both
            // masks from the SAME conditions so they can never drift apart again.
            let mut src = vk::PipelineStageFlags::COMPUTE_SHADER;
            let mut src_access = vk::AccessFlags::SHADER_WRITE;
            if self.dirty_transfer.get() {
                src |= vk::PipelineStageFlags::TRANSFER;
                src_access |= vk::AccessFlags::TRANSFER_WRITE;
            }
            let mut dst = vk::PipelineStageFlags::COMPUTE_SHADER;
            let mut dst_access = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
            if dst_transfer {
                dst |= vk::PipelineStageFlags::TRANSFER;
                dst_access |= vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
            }
            if self.indirect_pending.get() {
                // The consumer reads GPU-written dispatch args: cover the indirect-command read.
                dst |= vk::PipelineStageFlags::DRAW_INDIRECT;
                dst_access |= vk::AccessFlags::INDIRECT_COMMAND_READ;
            }
            let mb = vk::MemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(dst_access);
            unsafe {
                self.be.shared.device.cmd_pipeline_barrier(
                    self.cmd,
                    src,
                    dst,
                    vk::DependencyFlags::empty(),
                    &[mb],
                    &[],
                    &[],
                );
            }
            self.dirty_writes.borrow_mut().clear();
            self.dirty_reads.borrow_mut().clear();
            self.dirty_transfer.set(false);
            *self.barriers.borrow_mut() += 1;
        }
        self.dirty_reads.borrow_mut().extend(reads.iter().copied());
        self.dirty_writes
            .borrow_mut()
            .extend(writes.iter().copied());
    }

    fn dispatch(
        &self,
        k: ComputeKernel,
        buffers: &[vk::DescriptorBufferInfo],
        n_out: usize,
        push: &[u8],
        groups: u32,
    ) {
        self.dispatch3(k, buffers, n_out, push, groups, 1, 1);
    }

    /// The spec-guaranteed-minimum `VkPhysicalDeviceLimits::maxComputeWorkGroupCount[0]` (every
    /// conformant Vulkan implementation supports at least this many workgroups in dimension 0).
    /// Used as the safe split point below rather than threading the ACTUAL queried device limit
    /// through every GEMV call site — RDNA3/most desktop GPUs report a much larger x-limit, but an
    /// Intel A770 (Mesa ANV) enforces exactly the guaranteed minimum, and a wide GEMV (a
    /// ~150k-vocab lm_head, or `rows * gate_up_intermediate` at multi-row prefill) can dispatch
    /// past it in one dimension — the segfault-adjacent bug this splits around.
    const MAX_GROUP_COUNT_X: u32 = 65_535;

    /// Dispatch a FLAT `n`-workgroup 1-D grid, splitting into a safe 2-D grid
    /// (`MAX_GROUP_COUNT_X` × `ceil(n / MAX_GROUP_COUNT_X)`) whenever `n` would otherwise exceed
    /// the guaranteed limit. Every paired shader recovers the flat index as
    /// `gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x` (identical to `gl_WorkGroupID.x`
    /// when `gy == 1`, i.e. a total no-op for every dispatch that already fit in one dimension —
    /// which is every dispatch on hardware/models exercised so far) and bounds-checks it against
    /// the true element count, since `gx * gy` can overshoot `n` when `n` isn't a multiple of
    /// `gx` — those padding workgroups must return immediately without touching any buffer.
    fn dispatch_wide(
        &self,
        k: ComputeKernel,
        buffers: &[vk::DescriptorBufferInfo],
        n_out: usize,
        push: &[u8],
        n: u32,
    ) {
        if n <= Self::MAX_GROUP_COUNT_X {
            self.dispatch3(k, buffers, n_out, push, n, 1, 1);
        } else {
            let gy = n.div_ceil(Self::MAX_GROUP_COUNT_X);
            if self.be.cfg().debug.wide_dispatch {
                eprintln!(
                    "[infr] dispatch_wide SPLIT kernel={:?} n={n} -> gx={} gy={gy}",
                    k.name,
                    Self::MAX_GROUP_COUNT_X
                );
            }
            self.dispatch3(k, buffers, n_out, push, Self::MAX_GROUP_COUNT_X, gy, 1);
        }
    }

    /// Bind `k`'s pipeline and all of `buffers` to descriptor set 0. `VK_KHR_push_descriptor`
    /// (when the device has it) records the bindings straight into the command buffer with one
    /// `cmd_push_descriptor_set` call — no host-side descriptor-pool allocation or
    /// `vkUpdateDescriptorSets` syscall, which the pooled path below pays on EVERY dispatch. That
    /// per-dispatch churn measured as a real chunk of the host-side (non-GPU-timestamped) gap at
    /// small-m shapes (many-op graphs where GPU busy time is small — perf.md class 4). Falls back
    /// to the pooled alloc_set + update_descriptor_sets + cmd_bind_descriptor_sets sequence when
    /// the extension is unavailable; `k.ds_layout` was built to match (see `ops.rs`).
    fn bind_descriptors(&self, k: ComputeKernel, buffers: &[vk::DescriptorBufferInfo]) {
        let device = &self.be.shared.device;
        unsafe { device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline) };
        let n = buffers.len();
        debug_assert!(
            n <= MAX_DISPATCH_BINDINGS,
            "dispatch binds {n} buffers; raise MAX_DISPATCH_BINDINGS (currently {MAX_DISPATCH_BINDINGS})"
        );
        // Stack-built descriptor writes (binding counts are statically ≤ ~9) — no per-dispatch
        // `Vec` heap churn on this host-bound recording path (a batched MoE prefill chunk records
        // tens of thousands of dispatches).
        let mut ds_writes: [vk::WriteDescriptorSet; MAX_DISPATCH_BINDINGS] =
            std::array::from_fn(|_| vk::WriteDescriptorSet::default());
        if let Some(pd) = &self.be.shared.push_descriptor {
            for i in 0..n {
                ds_writes[i] = vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffers[i..i + 1]);
            }
            unsafe {
                pd.cmd_push_descriptor_set(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    k.pipeline_layout,
                    0,
                    &ds_writes[..n],
                );
            }
        } else {
            let set = self.alloc_set(k.ds_layout);
            for i in 0..n {
                ds_writes[i] = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffers[i..i + 1]);
            }
            unsafe {
                device.update_descriptor_sets(&ds_writes[..n], &[]);
                device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    k.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
            }
        }
    }

    /// Like [`Self::dispatch`], but the workgroup count comes from `args` (a GPU-written
    /// `[gx,gy,gz]` u32 triple at offset 0 — vkCmdDispatchIndirect). `args` joins the hazard reads
    /// so the barrier after its producer covers the indirect-command read (the barrier's dst stage
    /// widens to DRAW_INDIRECT whenever an indirect consumer follows).
    fn dispatch_indirect(
        &self,
        k: ComputeKernel,
        buffers: &[vk::DescriptorBufferInfo],
        n_out: usize,
        push: &[u8],
        args: vk::Buffer,
        args_off: u64,
    ) {
        // Auto-label (INFR_PROF_OPS): every dispatch stamps a timestamp tagged with its kernel name
        // — the chokepoint knows the kernel, so no op method needs a manual stamp call. Placed
        // before `sync` so a barrier's cost lands in the op it fences (same as the old
        // stamp-at-op-start convention). One `prof_ops` branch when profiling is off.
        self.stamp(k.name);
        let n = buffers.len();
        debug_assert!(
            n <= MAX_DISPATCH_BINDINGS,
            "indirect dispatch binds {n} buffers"
        );
        let split = n - n_out;
        // Stack scratch (no per-dispatch `Vec`): reads get `args` appended (the indirect-command
        // read joins the hazard reads), writes are the trailing `n_out` bindings.
        let mut all_reads: [vk::Buffer; MAX_DISPATCH_BINDINGS + 1] =
            [vk::Buffer::null(); MAX_DISPATCH_BINDINGS + 1];
        for (i, b) in buffers[..split].iter().enumerate() {
            all_reads[i] = b.buffer;
        }
        all_reads[split] = args;
        let mut write_bufs: [vk::Buffer; MAX_DISPATCH_BINDINGS] =
            [vk::Buffer::null(); MAX_DISPATCH_BINDINGS];
        for (i, b) in buffers[split..].iter().enumerate() {
            write_bufs[i] = b.buffer;
        }
        self.indirect_pending.set(true);
        self.sync(&all_reads[..split + 1], &write_bufs[..n_out], false);
        self.indirect_pending.set(false);
        self.bind_descriptors(k, buffers);
        let device = &self.be.shared.device;
        unsafe {
            if k.push_size > 0 {
                device.cmd_push_constants(
                    self.cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            device.cmd_dispatch_indirect(self.cmd, args, args_off);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch3(
        &self,
        k: ComputeKernel,
        buffers: &[vk::DescriptorBufferInfo],
        n_out: usize,
        push: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) {
        // Auto-label (INFR_PROF_OPS): see `dispatch_indirect` — kernel-name timestamp per dispatch.
        self.stamp(k.name);
        // The last `n_out` bound buffers are outputs; the rest are inputs. Inputs keep in-place
        // buffers (e.g. rope x==y) so a RAW from a prior op is still seen. Handles are copied into
        // one stack array (no per-dispatch `Vec`); reads/writes are then just subslices of it.
        let n = buffers.len();
        debug_assert!(n <= MAX_DISPATCH_BINDINGS, "dispatch binds {n} buffers");
        let split = n - n_out;
        let mut handles: [vk::Buffer; MAX_DISPATCH_BINDINGS] =
            [vk::Buffer::null(); MAX_DISPATCH_BINDINGS];
        for (i, b) in buffers.iter().enumerate() {
            handles[i] = b.buffer;
        }
        self.sync(&handles[..split], &handles[split..n], false);
        self.bind_descriptors(k, buffers);
        let device = &self.be.shared.device;

        unsafe {
            if k.push_size > 0 {
                device.cmd_push_constants(
                    self.cmd,
                    k.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            device.cmd_dispatch(self.cmd, gx, gy, gz);
        }
    }

    /// `&dyn Buffer` → `vk::DescriptorBufferInfo` — the SOLE choke point every dispatch call site in
    /// this file goes through before a buffer lands in a descriptor binding. An ordinary buffer
    /// binds `(offset=0, range=WHOLE_SIZE)` as always. A resident-BDA sub-tensor (`Buffer::device_addr`
    /// returns `Some`) is a logical byte range of a shared multi-GiB arena block — a `WHOLE_SIZE`
    /// bind would describe the whole BLOCK, not the tensor, so instead this binds the tensor's own
    /// `(sub_offset, range)`. The range is padded up to the next 256-byte word (some kernels read
    /// weight bytes as packed u32 words via the byte chokepoint `rb()`, which would clip the last
    /// partial word at a tight `range = size`) and clamped to the block's own end so it never reads
    /// past the arena's real allocation — the 256-byte pad can't run into a LATER tensor's data
    /// either, since the bump allocator aligns every sub-tensor start to that same 256 bytes. Every
    /// range fits `u32` (checked below): a tensor whose range would exceed 4 GiB cannot be bound at
    /// all (`maxStorageBufferRange` caps it) and must go via the `-DSTREAMED` device-address twin
    /// instead — see `Buffer::device_addr`'s doc and `Backing::BdaSub`.
    /// `&dyn Buffer` → `&VkBuffer` for the recorder's binding/handle helpers, PANICKING on a
    /// foreign buffer.
    ///
    /// The crate-level [`as_vk_buf`] is fallible (a checked `Any` downcast) precisely so a
    /// mis-routed buffer — `infr multi` runs several backends in one process — is a clean error
    /// instead of a reinterpreted pointer handed to the driver. The recorder cannot take that
    /// `Result` all the way out: `vkb` alone appears at ~360 sites, nearly all of them inside
    /// `&[vk::DescriptorBufferInfo]` array literals in dispatch bodies, and there is no error
    /// channel in that expression position. So the fallibility stops HERE, in one place, and turns
    /// into a panic that carries `as_vk_buf`'s full message.
    ///
    /// That is still the whole point of the change: the failure moves from silent memory
    /// corruption (a foreign struct's leading bytes read as a `vk::Buffer` handle + offsets, then
    /// dispatched — undiagnosable, arbitrarily far from the routing bug) to an immediate, named
    /// abort at the exact buffer that was mis-routed. Cost on the success path is one `TypeId`
    /// compare; the message is only formatted in the failure branch.
    fn vk_of(b: &dyn Buffer) -> &VkBuffer {
        match as_vk_buf(b) {
            Ok(v) => v,
            Err(e) => panic!("recorder: {e}"),
        }
    }

    fn vkb(b: &dyn Buffer) -> vk::DescriptorBufferInfo {
        let vb = Self::vk_of(b);
        let Backing::BdaSub(block) = &vb.backing else {
            return vk::DescriptorBufferInfo {
                buffer: vb.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            };
        };
        let block_len = block.buf.size as u64;
        let range = (vb.size as u64)
            .next_multiple_of(256)
            .min(block_len - vb.sub_offset as u64);
        // Checked in RELEASE too, and against the limit this DEVICE reported rather than the 4 GiB
        // a `u32` range field could hold (see `crate::max_storage_buffer_range`): a bind past
        // `maxStorageBufferRange` is a VUID violation whose consequence is whatever the driver
        // does with an out-of-range descriptor, and the old debug-only assert never ran in the
        // builds that ship. One compare on a path that already does two.
        let range_cap = crate::max_storage_buffer_range() as u64;
        assert!(
            range <= range_cap,
            "resident-BDA sub-tensor range ({range} bytes) exceeds this device's \
             maxStorageBufferRange ({range_cap}) — this tensor must be read through its 64-bit \
             device_addr() via a -DSTREAMED twin, never bound as a descriptor"
        );
        vk::DescriptorBufferInfo {
            buffer: vb.buffer,
            offset: vb.sub_offset as u64,
            range,
        }
    }

    /// [`Self::vkb`] with a leading f32-element offset, for binding one OUTPUT-row sub-range of an
    /// ordinary activation buffer (`x` / residual / `y`) during chunked dense dispatch (see
    /// [`Self::dispatch_gemv_chunked`]). The byte offset stays 256-byte aligned by construction —
    /// chunk starts are [`BDA_CHUNK_ROW_ALIGN`]-aligned and, at m>1, `in_f`/`out_f` are asserted
    /// multiples of that alignment — so it meets any device's `minStorageBufferOffsetAlignment`
    /// (Vulkan spec max 256). `elem_off == 0` reproduces [`Self::vkb`] exactly for an ordinary
    /// buffer, keeping the no-chunk fast path byte-identical. Never called on a resident-BDA weight
    /// sub-tensor (those are read by 64-bit address, not bound).
    fn vkb_off(b: &dyn Buffer, elem_off: usize) -> vk::DescriptorBufferInfo {
        let vb = Self::vk_of(b);
        debug_assert!(
            !matches!(vb.backing, Backing::BdaSub(_)),
            "vkb_off is for ordinary activation buffers, not resident-BDA weight sub-tensors"
        );
        vk::DescriptorBufferInfo {
            buffer: vb.buffer,
            offset: (elem_off as u64) * 4,
            range: vk::WHOLE_SIZE,
        }
    }

    /// Chunked twin of one `dispatch_wide` GEMV: when a `[out_f, in_f]` dense weight (lm_head /
    /// embed) breaches a u32 addressing cap, split the dispatch into output-row chunks — each under
    /// BOTH caps, its weight read from the pre-offset base `arena_addr + chunk.byte_off` (u64 base,
    /// u32 in-kernel indices, NO widening) — and, at m>1, into single input rows so every
    /// sub-dispatch writes a CONTIGUOUS slice of `y` bound at its own offset. Bitwise-identical to
    /// the single big dispatch: each output row's dot-product reduction is independent of how the
    /// output rows are partitioned across dispatches. The caller has already ruled out the no-chunk
    /// case with [`bda_gemv_needs_chunk`], so this only runs on a genuinely over-cap projection.
    /// `residual` present ⇒ the fused-add binding order `[filler, x, residual, y]`; absent ⇒
    /// `[filler, x, y]`. `grid(chunk_n)` gives the workgroup count for a `chunk_n`-row chunk at ONE
    /// input row. `push` carries `in_f` (`[4..8]`) and `w_base` (`[12..16]`) unchanged; this
    /// overwrites `rows` (`[0..4]` → 1), `out_f` (`[8..12]`) and the arena base (`[16..24]`).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemv_chunked(
        &self,
        k: ComputeKernel,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        x: &dyn Buffer,
        residual: Option<&dyn Buffer>,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
        mut push: [u8; 24],
        grid: impl Fn(u32) -> u32,
    ) {
        let (elem_cap, byte_cap) = self.chunk_caps();
        let chunks = bda_out_f_chunks(dtype, in_f, out_f, w_base, elem_cap, byte_cap).expect(
            "chunked dense GEMV reached an un-chunkable weight — the load-time addressing guard \
             (check_bda_element_cap / bda_weight_alloc) should have rejected it",
        );
        if rows > 1 {
            // Per-row binding offsets (r*in_f for x, r*out_f for y/residual) must stay
            // 256-byte aligned; the chunk-start alignment only covers o0.
            assert!(
                in_f.is_multiple_of(BDA_CHUNK_ROW_ALIGN)
                    && out_f.is_multiple_of(BDA_CHUNK_ROW_ALIGN),
                "chunked dense GEMV at m>1 needs in_f/out_f multiples of {BDA_CHUNK_ROW_ALIGN} to \
                 keep every per-row binding offset device-aligned (in_f={in_f}, out_f={out_f})"
            );
        }
        if self.be.cfg().debug.bda_chunk {
            eprintln!(
                "[infr] dispatch_gemv_chunked {dtype:?} rows={rows} in_f={in_f} out_f={out_f} \
                 -> {} chunks (caps: elems {elem_cap}, bytes {byte_cap})",
                chunks.len(),
            );
        }
        push[0..4].copy_from_slice(&1u32.to_ne_bytes()); // each sub-dispatch is a single input row
        for r in 0..rows {
            for c in &chunks {
                let base = arena_addr + c.byte_off;
                push[8..12].copy_from_slice(&(c.n as u32).to_ne_bytes());
                push[16..20].copy_from_slice(&(base as u32).to_ne_bytes());
                push[20..24].copy_from_slice(&((base >> 32) as u32).to_ne_bytes());
                let out_off = r * out_f + c.o0;
                let xb = Self::vkb_off(x, r * in_f);
                let yb = Self::vkb_off(y, out_off);
                match residual {
                    Some(res) => self.dispatch_wide(
                        k,
                        &[Self::vkb(x), xb, Self::vkb_off(res, out_off), yb],
                        1,
                        &push,
                        grid(c.n as u32),
                    ),
                    None => {
                        self.dispatch_wide(k, &[Self::vkb(x), xb, yb], 1, &push, grid(c.n as u32))
                    }
                }
            }
        }
    }

    /// `&dyn Buffer` → raw `vk::Buffer` handle, for the few call sites that consume the handle with
    /// EXPLICIT byte offsets rather than a descriptor binding: `dispatch_indirect`'s `args` buffer
    /// (`vkCmdDispatchIndirect`), [`Self::copy`] (`vkCmdCopyBuffer`), and [`Self::zero`]
    /// (`vkCmdFillBuffer`). Unlike `vkb`, no sub-range is encoded in the return value — each of
    /// those call sites is responsible for adding the buffer's `sub_offset` into its own offset
    /// arithmetic (a resident-BDA sub-tensor's logical byte 0 lives at `sub_offset` within the
    /// shared arena block; `args` is always ordinary scratch, so `dispatch_indirect` skips it).
    fn vk_handle(b: &dyn Buffer) -> vk::Buffer {
        Self::vk_of(b).buffer
    }

    /// Pinned subgroup size for the decode GEMV/reduction family (`caps.sg_pref`: 16 on Intel,
    /// 32 everywhere else — see the caps field doc). Every other pinned kernel stays at 32.
    fn sgp(&self) -> u32 {
        self.be.caps().sg_pref
    }

    /// Whether the decode GEMV family should use its `-DSG=16` SPIR-V twins (`sg_pref == 16`).
    fn sg16(&self) -> bool {
        self.sgp() == 16
    }

    /// `y[rows,out] = x[rows,in] · Wᵀ` (W stored `[out,in]`). Resident-BDA route: `w.device_addr()`
    /// `Some` means `w` is a sub-tensor of a `bufferDeviceAddress` arena block (see
    /// `Buffer::device_addr`'s doc) — always fork to the `-DSTREAMED` twin here rather than binding
    /// it, addressing the weight by its 64-bit pointer instead. This big-matmul weight family is
    /// exactly the case a bind can't cover unconditionally (a fused/wide weight's range can exceed
    /// `maxStorageBufferRange`/4 GiB), so streamed stays the required/preferred path regardless of
    /// `Self::vkb`'s sub-range bind support for the small unforked weight consumers elsewhere.
    pub fn linear(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear requires a u64 BDA device address");
        self.linear_at(arena_addr, x, y, rows, in_f, out_f);
    }

    /// `y[rows,out] = x[rows,in] · Wᵀ` (W stored `[out,in]` f16) — the `!caps.f16` tier: same shape
    /// as [`Self::linear`] but the weight buffer is read as packed u32 + `unpackHalf2x16` instead
    /// of a `float16_t` SSBO element, so it needs no shaderFloat16/Float16 SPIR-V capability. See
    /// `linear_f16_noext.comp`. Correctness-first — not row-tiled like the f32 mrow variants.
    pub fn linear_f16_noext(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_f16_noext requires a u64 BDA device address");
        self.linear_f16_noext_at(arena_addr, x, y, rows, in_f, out_f);
    }

    /// f32-weight GEMV `y = x·Wᵀ` — full-precision projection weights (gemma4 E2B's per-layer
    /// inp_gate/proj and qwen3moe's router ship as F32; reading them through the f16 kernel
    /// produced garbage). Dispatches the `linear_f32r` family (see
    /// [`Self::linear_f32_at`]'s mrow/vec4 tile pick) — these weights are small, so the
    /// simple kernels are fine. (The old eager thread-per-output `linear_f32.comp` is deleted;
    /// see build.rs's note where its entry used to live.)
    pub fn linear_f32(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_f32 requires a u64 BDA device address");
        self.linear_f32_at(arena_addr, x, y, rows, in_f, out_f);
    }

    /// Arena-addressed body of [`Self::linear`] (f16 weight via a typed 64-bit buffer_reference).
    /// Binding 0 takes a FILLER (`x`). Production routes here through [`Self::linear`] (the
    /// weight's own BDA address); parity tests call it directly with an explicit arena address.
    pub fn linear_at(
        &self,
        arena_addr: u64,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("lin_f16", rows, in_f, out_f);
        let k = self
            .be
            .kernel("linear_f16", crate::gemm::linear_f16_spv(), 3, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// `-DSTREAMED` twin of [`Self::linear_f16_noext`] (slice A4). Parity-test entry.
    pub fn linear_f16_noext_at(
        &self,
        arena_addr: u64,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("lin_f16_noext_streamed", rows, in_f, out_f);
        let k = self.be.kernel(
            "linear_f16_noext",
            crate::gemm::linear_f16_noext_spv(),
            3,
            20,
        );
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// bf16-weight GEMV `y = x·Wᵀ` (`linear_bf16.comp`) — recorder-level twin of the eager
    /// `ops::linear_bf16` (which owns its host round-trip); exists so the streamed parity test can
    /// drive the resident kernel in a recorded submit. Plain 1-D grid (the shader does no 2-D
    /// index recovery).
    pub fn linear_bf16(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_bf16 requires a u64 BDA device address");
        self.linear_bf16_at(arena_addr, x, y, rows, in_f, out_f);
    }

    /// `-DSTREAMED` twin of the bf16 GEMV (`linear_bf16.comp`; slice A4). Parity-test entry.
    pub fn linear_bf16_at(
        &self,
        arena_addr: u64,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("lin_bf16", rows, in_f, out_f);
        let k = self
            .be
            .kernel("linear_bf16", crate::gemm::linear_bf16_spv(), 3, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        // linear_bf16's grid is NOT dispatch_wide-recovered in the shader (plain gl_WorkGroupID.x)
        // — mirror the resident call shape: rows*out_f 1-D groups.
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// Arena-addressed body of [`Self::linear_f32`] — mirrors the eager dispatcher's mrow/vec4
    /// variant routing (same env escapes) so parity comparisons land on the same kernel.
    /// Production routes here through [`Self::linear_f32`] (the weight's own BDA address); parity
    /// tests call it directly with an explicit arena address.
    pub fn linear_f32_at(
        &self,
        arena_addr: u64,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        self.label_gemv("lin_f32_streamed", rows, in_f, out_f);
        let use_mrow = rows > 1 && self.vk().f32_mrow;
        let use_v4 = use_mrow && in_f.is_multiple_of(4) && self.vk().f32_v4;
        let (name, spv, groups) = if use_v4 && rows <= 4 {
            (
                "linear_f32r_mrow4_v4",
                crate::gemm::linear_f32r_mrow4_v4_spv(),
                (out_f * rows.div_ceil(4)) as u32,
            )
        } else if use_v4 {
            (
                "linear_f32r_mrow8_v4",
                crate::gemm::linear_f32r_mrow8_v4_spv(),
                (out_f * rows.div_ceil(8)) as u32,
            )
        } else if use_mrow {
            (
                "linear_f32r_mrow8",
                crate::gemm::linear_f32r_mrow8_spv(),
                (out_f * rows.div_ceil(8)) as u32,
            )
        } else {
            (
                "linear_f32r",
                crate::gemm::linear_f32r_spv(),
                (rows * out_f) as u32,
            )
        };
        let k = self.be.kernel(name, spv, 3, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            groups,
        );
    }

    /// Arena-addressed body of [`Self::linear_add`] (fused-residual f16 GEMV, `linear_res.comp`).
    /// Production routes here through [`Self::linear_add`] (the weight's own BDA address); parity
    /// tests call it directly with an explicit arena address.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_at(
        &self,
        arena_addr: u64,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let k = self
            .be
            .kernel("linear_res", crate::gemm::linear_res_spv(), 4, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(x),
                Self::vkb(x),
                Self::vkb(residual),
                Self::vkb(y),
            ],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// E2B per-layer inp_gate: fused f32 GEMV + GELU activation * strided up-read.
    /// `dst[r,o] = gelu(sum_k x[r,k] * w[o,k]) * up[up_off + r*up_stride + o]`.
    /// Scalar MROW=4 kernel, bit-identical to separate Linear(f32) + GatedAct(gelu, stride).
    /// Resident-BDA route: `w.device_addr()` `Some` forks to the `-DSTREAMED` twin (the only
    /// weight build — see [`Self::e2b_gate_at`]'s doc).
    pub fn e2b_gate(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        up: &dyn Buffer,
        up_off: usize,
        up_stride: usize,
        y: &dyn Buffer,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: e2b_gate requires a u64 BDA device address");
        self.e2b_gate_at(arena_addr, x, up, up_off, up_stride, y, m, in_f, out_f);
    }

    /// Arena-addressed body of [`Self::e2b_gate`] (weight read through a typed 64-bit
    /// buffer_reference — see `shaders/e2b_gate.comp`'s STREAMED doc). Binding 0 (the old
    /// resident build's weight SSBO) takes a harmless FILLER (`x`). Production routes here
    /// through [`Self::e2b_gate`] (the weight's own BDA address); parity tests call it directly
    /// with an explicit arena address.
    pub fn e2b_gate_at(
        &self,
        arena_addr: u64,
        x: &dyn Buffer,
        up: &dyn Buffer,
        up_off: usize,
        up_stride: usize,
        y: &dyn Buffer,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let k = self
            .be
            .kernel("e2b_gate", crate::gemm::e2b_gate_spv(), 4, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(up_off as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(up_stride as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let groups = out_f as u32 * (m as u32).div_ceil(4);
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(up), Self::vkb(y)],
            1,
            &push,
            groups,
        );
    }

    /// Prefill projection GEMM: `c[m,n] = a[m,k] · Wᵀ` on the matrix cores (coopmat). `a` is f32;
    /// `wq` is the f16 weight, packed 2/u32. `c` MUST be allocated `ceil(m/64)*64` rows (the kernel
    /// writes padded rows as 0). `n%64==0`, `k%32==0`.
    pub fn matmul_proj(
        &self,
        a: &dyn Buffer,
        wq: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let arena_addr = wq
            .device_addr()
            .expect("resident-BDA weight: matmul_proj requires a u64 BDA device address");
        self.matmul_proj_at(a, arena_addr, c, m, k, n);
    }

    /// `-DSTREAMED` twin of [`Self::matmul_proj`] (the coopmat f16 projection GEMM: see
    /// `gemm_proj.comp`'s STREAMED doc). Same arena-by-address convention as
    /// [`Self::matmul_mmq_at`] — the resident build's weight-SSBO slot (binding 1) takes a
    /// harmless FILLER (`a`) since the arena is never bound as a descriptor. Not wired into any
    /// production dispatch yet; exists so a parity test can exercise the `_streamed` SPV directly.
    pub fn matmul_proj_at(
        &self,
        a: &dyn Buffer,
        arena_addr: u64,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // Same M/N routing as matmul_proj — see its comment for the warp-tile rationale.
        let warp = m >= 768 && n.is_multiple_of(256);
        let (name, spv, tiles_n) = if warp {
            ("gemm_proj_warp", crate::gemm::gemm_proj_warp_spv(), n / 256)
        } else {
            ("gemm_proj", crate::gemm::gemm_proj_spv(), n / 64)
        };
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 3, 20, 32);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * tiles_n) as u32; // both kernels use BM=64
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(a), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Native-block projection GEMM `c = a · Wᵀ` for prefill: raw GGUF blocks dequantized in-shader
    /// during coopmat tiled staging (decode-once per weight element, reused across the 64-row tile).
    /// `c` is allocated `ceil(m/64)*64` rows. Requires `n%64==0`, `k%32==0`.
    pub fn matmul_native(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let w_addr = w
            .device_addr()
            .expect("resident-BDA weight: matmul_native requires a u64 BDA device address");
        self.matmul_native_off(dtype, a, w_addr, 0, c, m, k, n);
    }

    /// Native-block tiled coopmat GEMM reading the weight from element offset `w_base` — lets one
    /// stacked MoE expert tensor serve all experts (`w_base = expert_id * out_f * in_f`), so each
    /// expert weight is decoded ONCE and reused across the 64-row tile (vs the per-row GEMV re-read).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_off(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        // The weight's 64-bit base address: a resident weight's own BDA address
        // (`w.device_addr()`), or a dense-pager streamed block's slot base in the pager arena (see
        // adapter.rs `streamed_prefill_gemm`). The picked tile reads the weight by this address
        // (native_weight_addr.glsl); it is never bound as a descriptor.
        w_addr: u64,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // Large-warptile variant (8-warp BM=64×BN=256): 4× the math per staged A-row / decoded
        // W-column vs the 64×64 tile, and the extra warps hide the dequant latency. Needs k%32;
        // only the hot formats are compiled — everything else stays on the 64×64 kernel.
        // Tile pick: the wide BN=256 tile when its grid fills the device, else the NARROW BN=128
        // tile (same per-thread math, 2× the workgroups) — n=1024/2048 GEMMs underfilled a 96-wg
        // part at 5-9 TFLOPS with the wide tile. INFR_NO_GEMM_WARP forces the 64×64 tile (A/B).
        // Empirically the wide tile only wins once its own grid saturates (~2× device capacity).
        const WIDE_GRID_MIN: usize = 128;
        let wide_grid = m.div_ceil(64) * (n / 256).max(1);
        let use_wide = n.is_multiple_of(256) && wide_grid >= WIDE_GRID_MIN;
        // `Some(bn)` selects a warptile (BN=256 wide or BN=128 narrow) for `dtype`; `None` falls
        // back to the 64×64 kernel. The getters report tile availability only — weights are read by
        // 64-bit address via the `_streamed` twin below, so no resident SPV is loaded here.
        let warp = if k.is_multiple_of(32) && self.vk().gemm_warp {
            if use_wide {
                crate::gemm::native_gemm_warp_kernel_name(dtype).map(|_| 256usize)
            } else if n.is_multiple_of(128) {
                crate::gemm::native_gemm_warp_n128_kernel_name(dtype).map(|_| 128usize)
            } else {
                None
            }
        } else {
            None
        };
        // Resident tile NAME the arena-addressed `_streamed` twin keys off (`native_gemm_spv`
        // below) — the pick stays 1:1 with the resident tables, but nothing binds a resident SSBO.
        let name = match (warp, dtype) {
            (Some(256), infr_core::DType::Bf16) => "native_gemm_warp_bf16",
            (Some(256), infr_core::DType::Q3K) => "native_gemm_warp_q3k",
            (Some(256), infr_core::DType::Q5_0) => "native_gemm_warp_q5_0",
            (Some(256), infr_core::DType::Q5_1) => "native_gemm_warp_q5_1",
            (Some(256), infr_core::DType::Iq4Nl) => "native_gemm_warp_iq4nl",
            (Some(256), infr_core::DType::Iq4Xs) => "native_gemm_warp_iq4xs",
            (Some(256), infr_core::DType::Q2K) => "native_gemm_warp_q2k",
            (Some(256), infr_core::DType::Q4_0) => "native_gemm_warp_q4_0",
            (Some(256), infr_core::DType::Q4K) => "native_gemm_warp_q4k",
            (Some(256), infr_core::DType::Q5K) => "native_gemm_warp_q5k",
            (Some(256), infr_core::DType::Q6K) => "native_gemm_warp_q6k",
            (Some(256), infr_core::DType::Q8_0) => "native_gemm_warp_q8_0",
            (Some(_), infr_core::DType::Bf16) => "native_gemm_warp_bf16_n128",
            (Some(_), infr_core::DType::Q3K) => "native_gemm_warp_q3k_n128",
            (Some(_), infr_core::DType::Q5_0) => "native_gemm_warp_q5_0_n128",
            (Some(_), infr_core::DType::Q5_1) => "native_gemm_warp_q5_1_n128",
            (Some(_), infr_core::DType::Iq4Nl) => "native_gemm_warp_iq4nl_n128",
            (Some(_), infr_core::DType::Iq4Xs) => "native_gemm_warp_iq4xs_n128",
            (Some(_), infr_core::DType::Q2K) => "native_gemm_warp_q2k_n128",
            (Some(_), infr_core::DType::Q4_0) => "native_gemm_warp_q4_0_n128",
            (Some(_), infr_core::DType::Q4K) => "native_gemm_warp_q4k_n128",
            (Some(_), infr_core::DType::Q5K) => "native_gemm_warp_q5k_n128",
            (Some(_), infr_core::DType::Q6K) => "native_gemm_warp_q6k_n128",
            (Some(_), infr_core::DType::Q8_0) => "native_gemm_warp_q8_0_n128",
            _ => crate::linear::native_gemm_kernel_name(dtype),
        };
        self.label_gemm(name, m, k, n);
        // The weight is read by its 64-bit address (`w_addr`): the SAME tile the name pick chose,
        // arena base appended to the push, `a` bound as the binding-1 filler (the kernel reads the
        // weight by address, never an SSBO — the arena is NEVER bound as a descriptor).
        let (kname, kspv) = crate::gemm::native_gemm_spv(name);
        let push_size: u32 = 24;
        let kern = self.be.kernel_sg(kname, kspv, 3, push_size, 32);
        let groups_n = match warp {
            Some(bn) => n / bn,
            None => n / 64,
        };
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(w_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((w_addr >> 32) as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * groups_n) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(a), Self::vkb(c)],
            1,
            &push[..push_size as usize],
            groups,
        );
    }

    /// 8x8x16-fragment coopmat warptile GEMM (`_cm8` builds — Intel Arc/ANV XMX). One fixed tile:
    /// NARROW_N+BM32 (BM=32, BN=128, BK=64) with CM_M=CM_N=8 fragments, 128 threads pinned at
    /// subgroup size 16 (XMX/DPAS is SIMD16-native; 8 subgroups keep the kernel's 8-warp index
    /// math). Caller (the adapter's `cm8_ok` gate) guarantees: the device's selected f16 coopmat
    /// shape is 8x8x16 (`caps.f16_coopmat_8x8()` — never true on RADV), n%128==0, k%64==0, and
    /// the `_cm8` SPIR-V exists for `dtype`. No wide/split-K/A_GLOBAL variants yet — this is the
    /// minimal plumbing-proof tier; widen after the first A770/B580 A/B reports.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_cm8(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let (name, spv) = crate::gemm::native_gemm_warp_cm8_build_spv(dtype).expect("cm8 GEMM spv");
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 3, 16, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        // BM=32 row tiles × BN=128 column tiles (the kernel zero-pads A rows past `m`; C pad rows
        // land in the adapter's ceil(m/64)*64-row padded dst, same convention as the 16x16x16
        // warptiles).
        let groups = (m.div_ceil(32) * (n / 128)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// [`matmul_native_off`](Self::matmul_native_off) with A already converted to f16 (padded to
    /// ceil(m/64)·64 rows) — the A_GLOBAL warptiles coopMatLoad A straight from global, dropping
    /// the As stage/LDS (occupancy 2→3 wgs/WGP: ~1.5x on the 8B shapes). Same tile pick as the
    /// f32 path; caller guarantees k%32==0, n%128==0 and that the _ag SPIR-V exists for `dtype`
    /// (`native_gemm_warp_ag_kernel_name(dtype).is_some()`). Numerics are bit-identical to the f32
    /// path: the staging loop rounded A to f16 anyway, and the MMA order is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_f16a(
        &self,
        dtype: infr_core::DType,
        a16: &dyn Buffer,
        // The weight's 64-bit base address — see `matmul_native_off`'s `w_addr` doc.
        w_addr: u64,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // The BN=128 (n128) ag tile beats the BN=256 (wide) ag tile on EVERY shape this decision
        // can reach, measured on RDNA3 (7900 XTX): the wide tile's WN=64 → 2×4 = 8 accumulator
        // coopmat frags per warp cost enough VGPRs to drop occupancy, and the wide-square n=4096
        // shapes (o-proj, down) land at exactly ceil(m/64)·(n/256) = 128 workgroups — one
        // under-filled wave on the 48-WGP part. Halving the N tile doubles the workgroups AND
        // halves the accumulator footprint.
        //
        // The old gate `use_wide = wide_grid >= 128` had the crossover BACKWARDS: the
        // `wide_n128_crossover_sweep` bench shows n128 wins across the whole wide_grid >= 128
        // regime (the ONLY regime the old gate picked wide) — 8B o 40→70 TF, gate+up 53→77 TF,
        // down 42→73 TF, and the shallow-k small-model shapes too (1152×13824 43→68 TF,
        // 1024×6144 53→69 TF). The wide tile only edges n128 on wide_grid < 128 (small-N) shapes,
        // which the old code ALREADY routed to n128 and which production sends to split-K anyway —
        // so no shape reachable here prefers wide. In-model pp512: 8B +27%, qwen35-4B +10%,
        // gemma-4-E2B +10%; gemma-3-1b/qwen3-0.6b flat (their hot GEMMs are Q5_0 off this path /
        // split-K). Bit-identical to the wide tile (both BK=64, same k-accumulation order).
        // INFR_GEMM_WIDE_TILE restores the old BN=256 tile for A/B.
        let wide_grid = m.div_ceil(64) * (n / 256).max(1);
        let use_wide = n.is_multiple_of(256) && wide_grid >= 128 && self.vk().gemm_wide_tile;
        // Small-m batched prefill (MTP verify's draft window, m≈6-24): the default BM=64 row tile
        // is mostly masked waste (see DENSE_SMALL_TILE_MAX_M's doc) — switch to the BM=32 `_bm32`
        // n128_ag variant when it exists for `dtype` (only the K-quant formats verify actually
        // hits: Q4_K/Q5_K/Q6_K/Q8_0) and the wide tile wasn't explicitly requested (wide_grid>=128
        // never fires at these m anyway, short of the vocab-head GEMM under INFR_GEMM_WIDE_TILE).
        let small_bm = !use_wide && m <= DENSE_SMALL_TILE_MAX_M && self.vk().small_bm;
        // Sub-band of small_bm: at the very smallest m, the BM=32 tile itself is still ~half
        // masked — BM=16 (one coopmat M-frag, the tiling floor) halves that again. It also HALVES
        // WN (doubling WARPS_N), a thinner-frag shape, so both model sizes were measured interleaved
        // rather than trusting one run: MTP +3.5% on qwen35-4B (k=2560) AND +2.8% on qwen35-9B
        // (k=4096), consistent across passes — a real win on both. `INFR_NO_BM16` is a standalone A/B
        // escape (falls back to BM=32, not BM=64).
        let bm16 = small_bm && m <= DENSE_SMALL_TILE_MAX_M16 && self.vk().bm16;
        let bm16 = bm16
            .then(|| crate::gemm::native_gemm_warp_n128_ag_bm16_kernel_name(dtype))
            .flatten();
        let bm32 = small_bm
            .then(|| crate::gemm::native_gemm_warp_n128_ag_bm32_kernel_name(dtype))
            .flatten();
        // Resident A_GLOBAL tile NAME the `_streamed` twin keys off — the weight is read by 64-bit
        // address, so nothing binds a resident SSBO; the getters report the tile name/availability.
        let (name, bn, bm): (&str, usize, usize) = if use_wide {
            (
                crate::gemm::native_gemm_warp_ag_kernel_name(dtype).expect("ag name"),
                256,
                64,
            )
        } else if let Some(name) = bm16 {
            (name, 128, 16)
        } else if let Some(name) = bm32 {
            (name, 128, 32)
        } else {
            (
                crate::gemm::native_gemm_warp_n128_ag_kernel_name(dtype).expect("ag n128 name"),
                128,
                64,
            )
        };
        self.label_gemm(name, m, k, n);
        // Weight read by its 64-bit address (`w_addr`): the picked A_GLOBAL tile, arena base
        // appended to the push, `a16` doubles as the binding-1 filler.
        let (kname, kspv) = crate::gemm::native_gemm_spv(name);
        let push_size: u32 = 24;
        let kern = self.be.kernel_sg(kname, kspv, 3, push_size, 32);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(w_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((w_addr >> 32) as u32).to_ne_bytes());
        let groups = (m.div_ceil(bm) * (n / bn)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a16), Self::vkb(a16), Self::vkb(c)],
            1,
            &push[..push_size as usize],
            groups,
        );
    }

    /// [`matmul_native_f16a`](Self::matmul_native_f16a) with f32 activations read directly — no
    /// separate `store_f16` pass. The `_direct` kernel variants stage f32→f16 to As shared memory
    /// and coopMatLoad from there. Same tile shapes as the n128_ag family (BM=64/32/16, BN=128,
    /// BK=64). Requires `n%128==0`, `k%32==0`, and that the `_direct` SPIR-V exists for `dtype`
    /// (`native_gemm_warp_n128_direct_kernel_name(dtype).is_some()`). Numerics: the f32→f16 cast
    /// is the same rounding the old `store_f16` pass did, so the GEMM is bit-identical to the
    /// store_f16 + n128_ag path.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_f32a(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer, // [M, K] f32 activations (NOT pre-converted)
        w_addr: u64,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // Same tile-selection as the A_GLOBAL path: n128 always (wide_grid gate irrelevant here —
        // the _direct variants are only n128, and wide_grid≥128 never fires at the m these hit).
        let wide_grid = m.div_ceil(64) * (n / 256).max(1);
        let use_wide = n.is_multiple_of(256) && wide_grid >= 128 && self.vk().gemm_wide_tile;
        // Small-m batched prefill: same crossover constants as the A_GLOBAL path.
        let small_bm = !use_wide && m <= DENSE_SMALL_TILE_MAX_M && self.vk().small_bm;
        let bm16 = small_bm && m <= DENSE_SMALL_TILE_MAX_M16 && self.vk().bm16;
        let bm16 = bm16
            .then(|| crate::gemm::native_gemm_warp_n128_direct_bm16_kernel_name(dtype))
            .flatten();
        let bm32 = small_bm
            .then(|| crate::gemm::native_gemm_warp_n128_direct_bm32_kernel_name(dtype))
            .flatten();
        let (name, bm): (&str, usize) = if let Some(name) = bm16 {
            (name, 16)
        } else if let Some(name) = bm32 {
            (name, 32)
        } else {
            (
                crate::gemm::native_gemm_warp_n128_direct_kernel_name(dtype)
                    .expect("_direct kernel name"),
                64,
            )
        };
        self.label_gemm(name, m, k, n);
        let (kname, kspv) = crate::gemm::native_gemm_spv(name);
        let push_size: u32 = 24;
        let kern = self.be.kernel_sg(kname, kspv, 3, push_size, 32);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(w_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((w_addr >> 32) as u32).to_ne_bytes());
        let groups = (m.div_ceil(bm) * (n / 128)) as u32;
        self.dispatch(
            kern,
            // Binding 0 is the f32 activation buffer; binding 1 filler bound to a (inert, the
            // weight is read by device address); binding 2 is the output.
            &[Self::vkb(a), Self::vkb(a), Self::vkb(c)],
            1,
            &push[..push_size as usize],
            groups,
        );
    }

    /// SPLIT-K narrow-warptile GEMM: `splits` k-partials into `partials` ([splits, mpad, n] f32),
    /// then the deterministic fixed-order reduce into `c`. The occupancy fix for narrow-n GEMMs
    /// with deep k (o/down projections: n = n_embd, k = 2-3·n_embd — 64 workgroups on a 96-wg
    /// part with the plain tile). Requires n%128==0, k%32==0; caller sizes `partials` to
    /// `splits · ceil(m/64)·64 · n · 4` bytes. `a_is_f16`: A is the caller's f16 cast-copy
    /// (padded rows) and the A_GLOBAL tile is used — see [`matmul_native_f16a`](Self::matmul_native_f16a).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_native_splitk(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        // The weight's 64-bit base address — see `matmul_native_off`'s `w_addr` doc. Only the
        // k-partial GEMM reads the weight; the reduce pass (partials→c) is unchanged.
        w_addr: u64,
        w_base: usize,
        partials: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
        splits: usize,
        a_is_f16: bool,
    ) {
        // NB: `dense_small_m_row_tile_bench` also probed a BM=32 tile here (mirroring
        // `matmul_native_f16a`'s small-m gate) and found a NET LOSS across the whole m≈4-64
        // sweep on every sk_ag shape tested (down/attn_out/kv/q_proj/o_small) — split-K's own
        // `splits` dimension already fans this GEMM out across enough extra workgroups to fill
        // the device, so a smaller row tile only adds fixed per-workgroup cost (barrier/staging)
        // without an occupancy win to pay for it. BM=64 stays unconditional here; the win lives
        // in `matmul_native_f16a` (the non-split-K n128_ag family), which had no such fill floor.
        // Resident split-K tile NAME the `_streamed` twin keys off — the weight is read by 64-bit
        // address, so nothing binds a resident SSBO; the getters report the tile name/availability.
        let name = if a_is_f16 {
            crate::gemm::native_gemm_warp_sk_ag_kernel_name(dtype).expect("split-k ag name")
        } else {
            crate::gemm::native_gemm_warp_sk_kernel_name(dtype).expect("split-k name")
        };
        self.label_gemm(name, m, k, n);
        // Weight read by its 64-bit address (`w_addr`): the picked split-K tile, arena base
        // appended AFTER the splits/mpad fields, `a` doubles as the binding-1 filler.
        let (kname, kspv) = crate::gemm::native_gemm_spv(name);
        let push_size: u32 = 32;
        let mpad = m.div_ceil(64) * 64;
        let kern = self.be.kernel_sg(kname, kspv, 3, push_size, 32);
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(splits as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(mpad as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(w_addr as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&((w_addr >> 32) as u32).to_ne_bytes());
        let groups = ((mpad / 64) * (n / 128) * splits) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(a), Self::vkb(partials)],
            1,
            &push[..push_size as usize],
            groups,
        );
        // reduce: out[i] = Σ_s partials[s·plane + i]
        self.label_gemm("splitk_reduce", m, k, n);
        let rk = self
            .be
            .kernel("splitk_reduce", crate::gemm::splitk_reduce_spv(), 2, 12);
        let n_elems = mpad * n;
        let mut rp = [0u8; 12];
        rp[0..4].copy_from_slice(&(n_elems as u32).to_ne_bytes());
        rp[4..8].copy_from_slice(&(splits as u32).to_ne_bytes());
        rp[8..12].copy_from_slice(&(n_elems as u32).to_ne_bytes());
        self.dispatch(
            rk,
            &[Self::vkb(partials), Self::vkb(c)],
            1,
            &rp,
            (n_elems as u32).div_ceil(64),
        );
    }

    /// Tiled Q4_K dp4a (mmq) GEMM for a stacked expert: `c = qa·W[w_base]ᵀ` using hardware int8
    /// dot-product (activations pre-quantized via `quant_q8`). `c` is `ceil(m/64)*64` rows. Faster
    /// than the coopmat-f16 `matmul_native` for u4 weights (the dense path's default for u4).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_q4k(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_mmq_q4k", m, k, n);
        let kern = self.be.kernel(
            "native_gemm_mmq_q4k",
            crate::gemm::native_gemm_mmq_q4k_spv(),
            5,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        self.dispatch(
            kern,
            &[
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(w),
                Self::vkb(c),
            ],
            1,
            &push,
            groups,
        );
    }

    /// int8 cooperative-matrix (WMMA) prefill GEMM for Q8_0 weights — MEASUREMENT kernel, gated by
    /// the adapter behind [`VulkanBackend::i8_coopmat_ready`] (`INFR_I8_COOPMAT=1` +
    /// `caps.i8_coopmat` + this driver having PASSED the accumulator-layout probe; see
    /// `native_gemm_i8cm_q8_0.comp` for the design doc: 16x16 tile per workgroup, per-Q8_0-block
    /// int32 WMMA dot + in-fragment scale epilogue). The layout gate is the ADAPTER's, not this
    /// method's — a direct caller (the `parity`-feature resident-vs-streamed test) dispatches the
    /// kernel deliberately and compares two spellings of it against each other, which stays
    /// meaningful whatever the fragment layout is. `qa`/`dact` from `quant_q8`. `c` is
    /// `ceil(m/64)*64` rows (same padding convention as `matmul_native_off`). Requires `n%16==0`,
    /// `k%32==0`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_i8cm_q8_0(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_i8cm_q8_0", m, k, n);
        let kern = self.be.kernel_sg(
            "native_gemm_i8cm_q8_0",
            crate::gemm::native_gemm_i8cm_q8_0_spv(),
            4,
            16,
            32,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        // BM=128 (see the shader header): one workgroup covers 128 rows x 16 cols.
        let groups = (m.div_ceil(128) * (n / 16)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(dact), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// `-DSTREAMED` twin of [`Self::matmul_i8cm_q8_0`]: the raw Q8_0 weight bytes read through an
    /// inline 64-bit buffer_reference into the resident arena (`native_gemm_i8cm_q8_0.comp`'s
    /// STREAMED doc) instead of the bound binding-2 SSBO. The resident build's weight-SSBO slot
    /// takes a harmless FILLER (`qa`) since the arena is never bound as a descriptor — same
    /// convention as [`Self::matmul_proj_at`]. Not wired into any production dispatch (verified: no
    /// non-test caller) — it exists ONLY so a parity test can exercise the `_streamed` SPV
    /// directly, so it is compiled out of a normal build (`parity` feature / `test`). Kept `pub`
    /// (not `pub(crate)`) because its sole caller is the `weight_addr_parity` INTEGRATION test,
    /// which links the crate externally.
    #[cfg(any(test, feature = "parity"))]
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_i8cm_q8_0_at(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        arena_addr: u64,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_i8cm_q8_0_streamed", m, k, n);
        let kern = self.be.kernel_sg(
            "native_gemm_i8cm_q8_0_streamed",
            crate::gemm::native_gemm_i8cm_q8_0_streamed_spv(),
            4,
            24,
            32,
        );
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        // BM=128 (see the shader header): one workgroup covers 128 rows x 16 cols.
        let groups = (m.div_ceil(128) * (n / 16)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(dact), Self::vkb(qa), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// "Idea 2" measurement variant of `matmul_i8cm_q8_0`: `dact_row` is ONE f16 scale per row
    /// (whole-K, from `quant_q8_row`) instead of per-32-block — coarser precision, but lets the
    /// shader pull the activation scale out of its per-block sum. Gated by `INFR_I8_ROW_SCALE=1`
    /// (see `native_gemm_i8cm_q8_0.comp` #ifdef ROW_SCALE).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_i8cm_q8_0_rowscale(
        &self,
        qa: &dyn Buffer,
        dact_row: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_i8cm_q8_0_rowscale", m, k, n);
        let kern = self.be.kernel_sg(
            "native_gemm_i8cm_q8_0_rowscale",
            crate::gemm::native_gemm_i8cm_q8_0_rowscale_spv(),
            4,
            16,
            32,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(128) * (n / 16)) as u32;
        self.dispatch(
            kern,
            &[
                Self::vkb(qa),
                Self::vkb(dact_row),
                Self::vkb(w),
                Self::vkb(c),
            ],
            1,
            &push,
            groups,
        );
    }

    /// fp8 (E4M3) cooperative-matrix (WMMA) prefill GEMM for Q8_0 weights — gated by the adapter
    /// behind `INFR_F8_COOPMAT=1` + `caps.f8_coopmat` (see `native_gemm_f8cm_q8_0.comp` for the
    /// design doc: this is the SAME 256-thread/8-warp warptile as `native_gemm_warp`'s f16 GEMM,
    /// fp8-typed — a DIRECT matmul, no per-block int rescale like `matmul_i8cm_q8_0` needs). `qa`
    /// is the PRE-SCALED fp8 activation buffer from `quant_f8_row` (E4M3's range is tiny — raw f32
    /// activations must be range-scaled before the cast or they overflow to inf/NaN), `srow` is
    /// its per-row f32 descale factor, applied to the output in the shader's epilogue. `c` is
    /// `ceil(m/64)*64` rows (same padding convention as `matmul_native`). Picks the WIDE (BN=256,
    /// BK=32) tile when `n%256==0`, else the NARROW_N (BN=128, BK=64) tile — mirrors
    /// `matmul_native_off`'s self-contained wide/narrow pick; the caller (adapter.rs `f8cm_ok`)
    /// already checked shape eligibility for one of these before calling in. Requires `k%32==0`
    /// (wide) or `k%64==0` (narrow). Correctness UNVALIDATED on hardware without fp8 coopmat.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f8cm_q8_0(
        &self,
        qa: &dyn Buffer,
        srow: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let wide = n.is_multiple_of(256);
        let (name, spv, bn) = if wide {
            (
                "native_gemm_f8cm_q8_0",
                crate::gemm::native_gemm_f8cm_q8_0_spv(),
                256,
            )
        } else {
            (
                "native_gemm_f8cm_q8_0_n128",
                crate::gemm::native_gemm_f8cm_q8_0_n128_spv(),
                128,
            )
        };
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 4, 16, 32);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        // BM=64 (both tiles): one workgroup covers 64 rows x `bn` cols.
        let groups = (m.div_ceil(64) * (n / bn)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(w), Self::vkb(srow), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// NATIVE bf16 cooperative-matrix (WMMA) prefill GEMM: `c = a · Wᵀ`, `a` f32 activations, `w`
    /// raw bf16 weight (native_decode.glsl FMT_BF16), gated by the adapter behind
    /// `INFR_BF16_COOPMAT=1` + `caps.bf16_coopmat`. This is the `-DBF16CM` build of the SAME
    /// production `native_gemm_warp.comp` kernel `native_gemm_warp_bf16` (the f16-clamped path)
    /// uses — only the coopmat A/B operand type differs (bfloat16_t, no f16 clamp on FMT_BF16
    /// weights); staging/tiling/epilogue are byte-for-byte the same tuned warptile, so this should
    /// match `native_gemm_warp_bf16`'s speed while keeping bf16's full exponent range (see
    /// `native_gemm_warp.comp`'s BF16CM doc). `c` is `ceil(m/64)*64` rows (same padding convention
    /// as `matmul_native`). Picks the WIDE (BN=256, BK=32) tile when `n%256==0`, else the NARROW_N
    /// (BN=128, BK=64) tile — mirrors `matmul_f8cm_q8_0`'s wide/narrow pick; the caller
    /// (adapter.rs `bf16cm_ok`) already checked shape eligibility for one of these before calling
    /// in. Requires `k%32==0` (wide) or `k%64==0` (narrow). No A_GLOBAL variant (mirrors
    /// `native_gemm_warp_bf16`, which also has none). Correctness UNVALIDATED on hardware without
    /// bf16 coopmat.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_bf16cm(
        &self,
        a: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let wide = n.is_multiple_of(256);
        let (name, spv, bn) = if wide {
            (
                "native_gemm_warp_bf16cm",
                crate::gemm::native_gemm_warp_bf16cm_spv(),
                256,
            )
        } else {
            (
                "native_gemm_warp_bf16cm_n128",
                crate::gemm::native_gemm_warp_bf16cm_n128_spv(),
                128,
            )
        };
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 3, 16, 32);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        // BM=64 (both tiles): one workgroup covers 64 rows x `bn` cols.
        let groups = (m.div_ceil(64) * (n / bn)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// `-DPREPACK` measurement variant of `matmul_f8cm_q8_0`: `w8` is a PRE-PACKED E4M3 weight
    /// buffer (from `repack_q8_to_f8`, block scale already baked in) instead of raw Q8_0 blocks —
    /// the Bs staging in `native_gemm_f8cm_q8_0.comp` reads it DIRECTLY, no in-shader dqblk. Tests
    /// whether removing that dequant-ALU cost (the SAME cost f16 pays, which is why fp8-dqblk
    /// measured 0.73x f16 on RDNA4) lets fp8's 2x WMMA rate actually win. `qa`/`srow` are the
    /// SAME pre-scaled fp8 activations / per-row descale `matmul_f8cm_q8_0` uses (from
    /// `quant_f8_row`) — only the weight path differs. `w8` is always its own base (element 0):
    /// the adapter repacks into a per-shape pooled scratch buffer every call (see adapter.rs
    /// `f8cm_ok`'s INFR_F8_PREPACK arm), so no `w_base` parameter is needed (the push constant's
    /// `w_base` field is unused under `-DPREPACK`, kept at 0 only to share the push-const layout).
    /// `c` is `ceil(m/64)*64` rows, same convention as `matmul_f8cm_q8_0`. Gated by
    /// `INFR_F8_COOPMAT=1` + `INFR_F8_PREPACK=1`. Correctness UNVALIDATED on hardware without fp8
    /// coopmat.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f8cm_q8_0_prepacked(
        &self,
        qa: &dyn Buffer,
        srow: &dyn Buffer,
        w8: &dyn Buffer,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let wide = n.is_multiple_of(256);
        let (name, spv, bn) = if wide {
            (
                "native_gemm_f8cm_q8_0_prepack",
                crate::gemm::native_gemm_f8cm_q8_0_prepack_spv(),
                256,
            )
        } else {
            (
                "native_gemm_f8cm_q8_0_prepack_n128",
                crate::gemm::native_gemm_f8cm_q8_0_prepack_n128_spv(),
                128,
            )
        };
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel_sg(name, spv, 4, 16, 32);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&0u32.to_ne_bytes()); // w_base: unused under -DPREPACK
                                                           // BM=64 (both tiles): one workgroup covers 64 rows x `bn` cols.
        let groups = (m.div_ceil(64) * (n / bn)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(w8), Self::vkb(srow), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Prepack a Q8_0 weight tensor into E4M3 (`w8`), block scale BAKED IN — decode-once via
    /// `dqblk` (see `repack_q8_to_f8.comp`), producing the weight buffer
    /// `matmul_f8cm_q8_0_prepacked` reads directly. `w` is the Q8_0 SOURCE weight buffer, `w_base`
    /// the usual element-offset convention into it (fused-QKV slices / stacked-MoE expert offset);
    /// `w8` is written starting at element 0 (the caller's pooled per-shape scratch — see
    /// adapter.rs `f8cm_ok`'s INFR_F8_PREPACK arm). `w8` is `[n, k]` E4M3, 1 byte/elem. Gated by
    /// `INFR_F8_COOPMAT=1` + `INFR_F8_PREPACK=1`. Repacks EVERY call (no cross-call weight cache)
    /// — a real deployment would cache the repack at load time; this is a per-op profiling
    /// isolation path.
    pub fn repack_q8_to_f8(
        &self,
        w: &dyn Buffer,
        w_base: usize,
        w8: &dyn Buffer,
        n: usize,
        k: usize,
    ) {
        let kern = self
            .be
            .kernel("repack_q8_to_f8", crate::gemm::repack_q8_to_f8_spv(), 2, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let total_blocks = ((n * k) / 32) as u32;
        let groups = total_blocks.div_ceil(256);
        self.dispatch_wide(kern, &[Self::vkb(w), Self::vkb(w8)], 1, &push, groups);
    }

    /// `-DSTREAMED` twin of [`Self::repack_q8_to_f8`]: the Q8_0 source is read through
    /// `native_weight_addr.glsl`'s `NW()` chokepoint into the resident arena instead of the bound
    /// binding-0 SSBO (`repack_q8_to_f8.comp`'s STREAMED doc). The vacated binding-0 slot takes a
    /// harmless FILLER (`w8`, the only other buffer this dispatch touches) since the arena is never
    /// bound as a descriptor. Not wired into any production dispatch (verified: no non-test caller)
    /// — it exists ONLY so a parity test can exercise the `_streamed` SPV directly, so it is
    /// compiled out of a normal build (`parity` feature / `test`). Kept `pub` (not `pub(crate)`)
    /// because its sole caller is the `weight_addr_parity` INTEGRATION test, which links the crate
    /// externally.
    #[cfg(any(test, feature = "parity"))]
    #[allow(clippy::too_many_arguments)]
    pub fn repack_q8_to_f8_at(
        &self,
        arena_addr: u64,
        w_base: usize,
        w8: &dyn Buffer,
        n: usize,
        k: usize,
    ) {
        let kern = self.be.kernel(
            "repack_q8_to_f8_streamed",
            crate::gemm::repack_q8_to_f8_streamed_spv(),
            2,
            20,
        );
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let total_blocks = ((n * k) / 32) as u32;
        let groups = total_blocks.div_ceil(256);
        self.dispatch_wide(kern, &[Self::vkb(w8), Self::vkb(w8)], 1, &push, groups);
    }

    /// Row-wise (whole-K) fp8 (E4M3) activation quantize pass — the activation-scaling step for
    /// `matmul_f8cm_q8_0` (see `quant_f8_row.comp`): ONE f32 scale per row (`srow[m]`), computed
    /// from the row's amax and applied so activations land inside E4M3's +-448 range before the
    /// f32->fp8 cast. Structurally mirrors `quant_q8_row` (one workgroup per row, 256 threads),
    /// but the scale is a pure range rescale (`amax/448`), not a fixed-point quant step.
    pub fn quant_f8_row(
        &self,
        a: &dyn Buffer,
        qa: &dyn Buffer,
        srow: &dyn Buffer,
        m: usize,
        k: usize,
    ) {
        let kq = self
            .be
            .kernel_sg("quant_f8_row", crate::gemm::quant_f8_row_spv(), 3, 8, 32);
        let mut p = [0u8; 8];
        p[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        self.dispatch(
            kq,
            &[Self::vkb(a), Self::vkb(qa), Self::vkb(srow)],
            2,
            &p,
            m as u32,
        );
    }

    /// Row-wise (whole-K) int8 activation quantize pass — int8-coopmat GEMM "Idea 2" measurement
    /// variant (see `quant_q8_row.comp` / `native_gemm_i8cm_q8_0.comp`): ONE f16 scale per row
    /// (`dact_row[m]`) instead of `quant_q8`'s per-32-block scale, coarser but block-invariant so
    /// the i8cm kernel can pull it out of its per-block sum. Gated by `INFR_I8_ROW_SCALE=1`; used
    /// only by the int8-coopmat GEMM path, `quant_q8` (every other int8 kernel) is untouched.
    pub fn quant_q8_row(
        &self,
        a: &dyn Buffer,
        qa: &dyn Buffer,
        dact_row: &dyn Buffer,
        m: usize,
        k: usize,
    ) {
        let (name, spv) = crate::gemm::quant_q8_row_build_spv(self.sg16());
        let kq = self.be.kernel_sg(name, spv, 3, 8, self.sgp());
        let mut p = [0u8; 8];
        p[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        self.dispatch(
            kq,
            &[Self::vkb(a), Self::vkb(qa), Self::vkb(dact_row)],
            2,
            &p,
            m as u32,
        );
    }

    /// Tiled Q6_K dp4a (mmq) GEMM for a stacked expert (the MoE down projection): `c = qa·W[w_base]ᵀ`
    /// using hardware int8 dot-product (activations pre-quantized via `quant_q8`; the per-block sum is
    /// unused — Q6_K is symmetric, no min). `c` is `ceil(m/64)*64` rows. Faster than the coopmat-f16
    /// `matmul_native` for u6 weights. Requires `n%64`, `k%16`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_q6k(
        &self,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        self.label_gemm("native_gemm_mmq_q6k", m, k, n);
        let kern = self.be.kernel(
            "native_gemm_mmq_q6k",
            crate::gemm::native_gemm_mmq_q6k_spv(),
            4,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(qa), Self::vkb(dact), Self::vkb(w), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Dtype-generic DENSE dp4a (mmq) tiled GEMM: `c = qa·W[w_base]ᵀ` for ANY
    /// `infr_core::tensor::MOE_MMQ_DTYPES` member — the non-coopmat prefill tier's GEMM
    /// (adapter.rs `nc_mmq`: no usable f16 coopmat, packed int8 dot present — Intel Arc/ANV).
    /// Same kernels/conventions as [`Self::matmul_mmq_q4k`]/[`Self::matmul_mmq_q6k`] (which it
    /// delegates to for those dtypes' names), extended to the full dense build set
    /// (`native_gemm_mmq_dense_spv`): activations pre-quantized via `quant_q8`, `c` is
    /// `ceil(m/64)*64` rows, requires `n%64`, `k%32`. `sact` (the per-32-block Σx term) is
    /// always passed but bound ONLY for the min-carrying dtypes
    /// (`infr_core::tensor::moe_mmq_needs_sact` — the same SSOT the `_xp` expert dispatch uses;
    /// Q2_K self-computes its narrower Σx in-shader).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq(
        &self,
        dtype: infr_core::DType,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: matmul_mmq requires a u64 BDA device address");
        self.matmul_mmq_at(dtype, qa, dact, sact, arena_addr, w_base, c, m, k, n);
    }

    /// `-DSTREAMED` twin of [`Self::matmul_mmq`] (slice A3 build-variant: see
    /// `crate::gemm::native_gemm_mmq_dense_spv`). Same arena-by-address convention as
    /// [`Self::linear_native_at`] — the resident build's weight SSBO slot takes a small
    /// FILLER (`qa`) since the arena is never bound as a descriptor. Not wired into any production
    /// dispatch yet; exists so a parity test can exercise the `_streamed` SPV directly.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_at(
        &self,
        dtype: infr_core::DType,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        arena_addr: u64,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let (name, spv) = crate::gemm::native_gemm_mmq_dense_spv(dtype)
            .expect("matmul_mmq_at: dtype gated by native_gemm_mmq_dense_spv");
        let needs_sact = infr_core::tensor::moe_mmq_needs_sact(dtype);
        self.label_gemm(name, m, k, n);
        let nb = if needs_sact { 5 } else { 4 };
        let kern = self.be.kernel(name, spv, nb, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        let mut bufs = vec![Self::vkb(qa), Self::vkb(dact)];
        if needs_sact {
            bufs.push(Self::vkb(sact));
        }
        bufs.extend_from_slice(&[Self::vkb(qa), Self::vkb(c)]);
        self.dispatch(kern, &bufs, 1, &push, groups);
    }

    /// Arena-addressed body of [`Self::matmul_fma`] (see `crate::gemm::native_gemm_fma_spv`).
    /// The vacated binding-1 slot takes a FILLER (`a`). Production routes here through
    /// [`Self::matmul_fma`] (the weight's own BDA address); parity tests call it directly with an
    /// explicit arena address.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_fma_at(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        arena_addr: u64,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let (name, spv) = crate::gemm::native_gemm_fma_spv(dtype)
            .expect("matmul_fma_at: dtype gated by native_gemm_fma_spv");
        self.label_gemm(name, m, k, n);
        let kern = self.be.kernel(name, spv, 3, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let groups = (m.div_ceil(64) * (n / 64)) as u32;
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(a), Self::vkb(c)],
            1,
            &push,
            groups,
        );
    }

    /// Non-coopmat float-weight prefill GEMM (the "fma-warp" tier): `c = a·W[w_base]ᵀ`, `a` f32
    /// activations, `w` raw f16/bf16/f32 words read NATIVELY (bf16 keeps its full exponent
    /// range — bits shifted `<<16` into f32 registers, no f16 clamp and no upconverted weight
    /// copy). Shared-memory fma warptile (BM=64×BN=64×BK=32, 256 threads, TM=TN=4 register
    /// block — see `native_gemm_fma.comp` for the Intel Arc design constraints: no subgroup
    /// ops, no f16 extensions, 16.5 KB LDS, modest registers). Selected by the adapter's
    /// non-coopmat tier (`nc_fma`, `!caps.f16_coopmat`); beats the per-row scalar GEMVs by
    /// amortizing each weight element across the 64-row tile, does NOT need to beat coopmat.
    /// `c` is `ceil(m/64)*64` rows (the usual padded-GEMM convention). Requires `n%64`, `k%32`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_fma(
        &self,
        dtype: infr_core::DType,
        a: &dyn Buffer,
        w: &dyn Buffer,
        w_base: usize,
        c: &dyn Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: matmul_fma requires a u64 BDA device address");
        self.matmul_fma_at(dtype, a, arena_addr, w_base, c, m, k, n);
    }

    /// Native-block dequant GEMV `y = x·Wᵀ`. Raw GGUF block bytes in `w` (padded to u32); format
    /// identified by `dtype`. Dispatch `rows*out_f` workgroups. Reads the weight from element offset
    /// `w_base` (pass `0` for the whole tensor) — lets one stacked MoE expert tensor serve all
    /// experts (`w_base = expert_id * out_f * in_f`).
    ///
    /// Resident-BDA route (`w.device_addr()` is `Some`): no single `_streamed` twin mirrors this
    /// dispatcher's own SG → variant → RM → plain precedence the way [`Self::linear_add_native`]'s
    /// twin ([`Self::linear_add_native_at`]) does — each branch's kernel has its OWN
    /// `_streamed` twin instead (see [`Self::linear_native_sg_at`],
    /// [`Self::linear_native_rm_v2_at`], [`Self::linear_native_rm_at`],
    /// [`Self::linear_native_at`]), built per-kernel in slice A1. This fork replicates the
    /// SAME precedence by hand, calling each streamed twin at the exact point its resident
    /// counterpart would fire — the build tables are 1:1 mirrored (verified against `gemm.rs`), so
    /// every branch reachable here has a streamed kernel to route to.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_native requires a u64 BDA device address");
        // Replicate the resident SG → RM-v2 → RM → plain precedence, routing each branch to its own
        // `_streamed` twin (the build tables are 1:1 mirrored — see this fn's doc). Every branch
        // reachable here has a streamed kernel to route to.
        if rows == 1 {
            if let Some(nr) = native_sg_choice(dtype, in_f, out_f, self.gemv()) {
                if crate::gemm::native_sg_streamed_build_spv(dtype, false, nr, self.sg16())
                    .is_some()
                {
                    self.linear_native_sg_at(dtype, arena_addr, w_base, x, y, in_f, out_f, nr);
                    return;
                }
            }
            if let Some(v) = &self.gemv().variant {
                if crate::gemm::native_rm_v2_streamed_build_spv(v, dtype, false).is_some() {
                    self.linear_native_rm_v2_at(v, dtype, arena_addr, w_base, x, y, in_f, out_f);
                    return;
                }
            }
            if let Some(rm) = native_rm_choice(dtype, out_f, self.gemv()) {
                if crate::gemm::native_rm_streamed_build_spv(dtype, false, rm).is_some() {
                    self.linear_native_rm_at(dtype, arena_addr, w_base, x, y, in_f, out_f, rm);
                    return;
                }
            }
        }
        self.linear_native_at(dtype, arena_addr, w_base, x, y, rows, in_f, out_f);
    }

    /// Dense layer-streaming GEMV (`native_gemv.comp -DSTREAMED`): `y = x·Wᵀ` where the weight
    /// lives in a `bufferDeviceAddress` arena pool (see [`crate::pager::DensePagerSession`])
    /// addressed by `arena_addr` — the resident slot's base BYTE address — NOT a bound SSBO. This
    /// lifts the ~4 GiB `maxStorageBufferRange` cap the resident/SSBO GEMV binding has. `w_base` is
    /// the op's own WITHIN-slot element offset (a fused-QKV slice, usually 0), added to the
    /// per-output index exactly as the resident path does. All streamed dense Linears (any m) route
    /// here — the rows loop handles m>1 (re-streaming the weight per row: correct, and streaming is
    /// already a memory-bound fallback where kernel micro-optimization is dominated by PCIe/residency
    /// cost).
    ///
    /// The arena is read purely by the `arena_addr` push constant — it is NEVER bound as a
    /// descriptor (a `WHOLE_SIZE` binding of this multi-GiB pager arena would re-impose
    /// `maxStorageBufferRange` and trip a descriptor-range VUID, exactly what pins the SSBO path at
    /// ~4 GiB — unlike the smaller resident-BDA weight arena's sub-tensors, whose tight
    /// `(sub_offset, range)` binds via `Self::vkb` stay well under the cap and are bound routinely).
    /// Binding 0 (the resident build's weight SSBO) takes a small read-only FILLER (`x`) so the
    /// descriptor set stays fully written — same convention as the MoE `-DPAGED` dispatch, which
    /// fills binding 0 with `lut`, not the giant arena (see [`Self::linear_native_id_paged`]). The
    /// ring→arena copy vs pointer-read hazard is ordered explicitly by
    /// [`Self::arena_stream_barrier`] at the staging site (the pointer read is invisible to the
    /// buffer hazard tracker).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_native_at", in_f);
        self.label_gemv("gemv_streamed", rows, in_f, out_f);
        let (name, spv) =
            crate::gemm::native_streamed_build_spv(dtype, false).expect("native streamed GEMV spv");
        let k = self.be.kernel(name, spv, 3, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        // Arena base BYTE address (lo/hi split — a uvec2 avoids the 8-byte push alignment a
        // uint64_t member forces; see native_weight_addr.glsl). `w_base` rides on top unchanged.
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        // Chunked dispatch when the weight breaches a u32 addressing cap (a `>= 2^32`-element
        // lm_head/embed); the always-on check is `false` for every current model, taking the
        // literal single dispatch below unchanged.
        if bda_gemv_needs_chunk(dtype, in_f, out_f, w_base, self.chunk_caps()) {
            self.dispatch_gemv_chunked(
                k,
                dtype,
                arena_addr,
                w_base,
                x,
                None,
                y,
                rows,
                in_f,
                out_f,
                push,
                |n| n,
            );
            return;
        }
        // Binding 0 = `x` (small filler, unread by the STREAMED shader); 1 = x; 2 = y (the sole
        // write, kept LAST so `dispatch_wide`'s tail-n_out split marks it the write).
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// Multi-row native GEMV (`2 <= rows <= 8`): the GEMV's out_f-wide cooperative-over-K grid,
    /// each workgroup decoding a weight sub-block ONCE and dotting it against every row — the
    /// spec-decode verify / short-suffix-prefill shape, where the single-M-tile coopmat GEMM
    /// underfills the GPU (measured 51-182 GB/s effective vs the GEMV's 292-651 on a 7900 XTX)
    /// and the plain GEMV re-streams the weight per row. Same push layout as the GEMV; `w_off`
    /// (fused-QKV slices) rides `w_base`. Caller gates on [`crate::gemm::native_mrow_kernel_name`].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_mrow(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        debug_assert!((2..=8).contains(&rows));
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_native_mrow requires a u64 BDA device address");
        self.linear_native_mrow_at(dtype, arena_addr, w_base, x, y, rows, in_f, out_f);
    }

    /// `-DSTREAMED` twin of [`Self::linear_native_mrow`] (slice A1 build-variant: see
    /// `crate::gemm::native_mrow_spv`). Same arena-by-address convention as
    /// [`Self::linear_native_at`] — binding 0 (the resident build's weight SSBO) takes a
    /// small FILLER (`x`) since the arena is never bound as a descriptor. Not wired into any
    /// production dispatch yet; exists so a parity test can exercise the `_streamed` SPV directly.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_mrow_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        debug_assert!((2..=8).contains(&rows));
        assert_native_k("linear_native_mrow_at", in_f);
        self.label_gemv("mrow_streamed", rows, in_f, out_f);
        let (name, spv) = crate::gemm::native_mrow_spv(dtype).expect("native mrow streamed spv");
        let k = self.be.kernel(name, spv, 3, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            out_f as u32,
        );
    }

    /// `-DSTREAMED` twin of the reassociation-tolerant subgroup+NUM_ROWS decode GEMV
    /// (`native_gemv_sg.comp`, slice A1 build-variant: see `crate::gemm::native_sg_streamed_build_spv`).
    /// `m=1` decode only (matches the resident SG route's own gate); `nr` ∈ {2,4,8}. Not wired into
    /// any production dispatch yet; exists so a parity test can exercise the `_streamed` SPV
    /// directly.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_sg_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
        nr: u32,
    ) {
        assert_native_k("linear_native_sg_at", in_f);
        self.label_gemv("gemv_sg_streamed", 1, in_f, out_f);
        let (name, spv) = crate::gemm::native_sg_streamed_build_spv(dtype, false, nr, self.sg16())
            .expect("native sg streamed spv");
        let k = self.be.kernel_sg(name, spv, 3, 24, self.sgp());
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes()); // rows: SG serves m=1 decode only
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        if bda_gemv_needs_chunk(dtype, in_f, out_f, w_base, self.chunk_caps()) {
            self.dispatch_gemv_chunked(
                k,
                dtype,
                arena_addr,
                w_base,
                x,
                None,
                y,
                1,
                in_f,
                out_f,
                push,
                |n| n.div_ceil(nr),
            );
            return;
        }
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (out_f as u32).div_ceil(nr),
        );
    }

    /// `-DSTREAMED` twin of the multi-output-row decode GEMV (`native_gemv_rm.comp`, slice A1
    /// build-variant: see `crate::gemm::native_rm_streamed_build_spv`). `m=1` decode only (matches
    /// the resident RM route's own gate); `rm` ∈ {2,4}. Not wired into any production dispatch yet;
    /// exists so a parity test can exercise the `_streamed` SPV directly.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_rm_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
        rm: u32,
    ) {
        assert_native_k("linear_native_rm_at", in_f);
        self.label_gemv("gemv_rm_streamed", 1, in_f, out_f);
        let (name, spv) = crate::gemm::native_rm_streamed_build_spv(dtype, false, rm)
            .expect("native rm streamed spv");
        let k = self.be.kernel(name, spv, 3, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes()); // rows: RM serves m=1 decode only
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        if bda_gemv_needs_chunk(dtype, in_f, out_f, w_base, self.chunk_caps()) {
            self.dispatch_gemv_chunked(
                k,
                dtype,
                arena_addr,
                w_base,
                x,
                None,
                y,
                1,
                in_f,
                out_f,
                push,
                |n| n.div_ceil(rm),
            );
            return;
        }
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (out_f as u32).div_ceil(rm),
        );
    }

    /// `-DSTREAMED` twin of an experimental RM kernel variant (`native_gemv_rm_v2.comp`, slice A1
    /// build-variant: see `crate::gemm::native_rm_v2_streamed_build_spv`). `m=1` decode only, `RM=2`
    /// fixed (all current variant builds). Not wired into any production dispatch yet; exists so a
    /// parity test can exercise the `_streamed` SPV directly.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_rm_v2_at(
        &self,
        variant: &str,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_native_rm_v2_at", in_f);
        self.label_gemv("gemv_rm_v2_streamed", 1, in_f, out_f);
        let (name, spv) = crate::gemm::native_rm_v2_streamed_build_spv(variant, dtype, false)
            .expect("native rm_v2 streamed spv");
        let k = self.be.kernel(name, spv, 3, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes()); // rows: rm_v2 serves m=1 decode only
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        if bda_gemv_needs_chunk(dtype, in_f, out_f, w_base, self.chunk_caps()) {
            self.dispatch_gemv_chunked(
                k,
                dtype,
                arena_addr,
                w_base,
                x,
                None,
                y,
                1,
                in_f,
                out_f,
                push,
                |n| n.div_ceil(2),
            );
            return;
        }
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (out_f as u32).div_ceil(2),
        );
    }

    /// `-DSTREAMED` twin of [`Self::linear_mmv`] (slice A2 build-variant: see
    /// `crate::gemm::native_mmv_spv`). Same arena-by-address convention as
    /// [`Self::linear_native_at`] — binding 0 (the resident build's weight SSBO) takes a
    /// small FILLER (`qa`) since the arena is never bound as a descriptor. Not wired into any
    /// production dispatch yet; exists so a parity test can exercise the `_streamed` SPV directly.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_mmv_at", in_f);
        self.label_gemv("mmv_streamed", 1, in_f, out_f);
        let (name, spv) =
            crate::gemm::native_mmv_spv(dtype, false).expect("native mmv streamed spv");
        let k = self.be.kernel(name, spv, 5, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(qa),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(y),
            ],
            1,
            &push,
            (out_f as u32).div_ceil(2),
        );
    }

    /// `-DSTREAMED` twin of [`Self::linear_mmv_mrow`] (slice A2 build-variant: see
    /// `crate::gemm::native_mmv_mrow_variant_spv`). Mirrors the resident dispatcher's
    /// o4/m4/m16 routing EXACTLY (same env escapes) so a parity test comparing the two at the same
    /// shape lands on the same layout variant. `residual`: fused `y = residual + x·Wᵀ`, same
    /// legality rule as the resident dispatcher (`rows == 1` only — asserted; never legal at the
    /// `rows > 8` m16 tier). Binding count grows 5→6 when `Some`, same shape as
    /// [`Self::linear_mmv_mrow`]'s res arm — the arena address still rides the FILLER'd binding 0,
    /// residual inserted right before the trailing `y` write.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_mrow_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        residual: Option<&dyn Buffer>,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        debug_assert!((1..=16).contains(&rows));
        debug_assert!(
            residual.is_none() || rows == 1,
            "fused residual is decode-only (rows=1)"
        );
        // Explicit res-legality gate (AUDIT #2): not every mrow-eligible dtype has a fused-residual
        // build (Iq4Xs has none). Assert on the predicate so an illegal residual decode is caught
        // here with a clear message rather than in the SPV `expect()` below.
        debug_assert!(
            residual.is_none() || crate::gemm::native_mmv_mrow_res_supported(dtype),
            "no fused-residual mrow build for {dtype:?} — dtype is decode-eligible but res-illegal"
        );
        assert_native_k("linear_mmv_mrow_at", in_f);
        self.label_gemv("mmvr_streamed", rows, in_f, out_f);
        let o4 = in_f < 2048 && self.vk().mmv_o4;
        let m4 = rows <= 4 && self.vk().mmv_m4;
        let res = residual.is_some();
        let (name, spv) = if rows > 8 {
            crate::gemm::native_mmv_mrow_m16_spv(dtype).expect("native mmv mrow m16 streamed spv")
        } else {
            crate::gemm::native_mmv_mrow_variant_spv(dtype, o4, m4, res)
                .expect("native mmv mrow streamed spv")
        };
        let groups = (out_f as u32).div_ceil(if o4 && rows <= 8 { 4 } else { 2 });
        let nbind = if res { 6 } else { 5 };
        let k = self.be.kernel(name, spv, nbind, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let mut bufs = vec![
            Self::vkb(qa),
            Self::vkb(qa),
            Self::vkb(dact),
            Self::vkb(sact),
        ];
        if let Some(r) = residual {
            bufs.push(Self::vkb(r));
        }
        bufs.push(Self::vkb(y));
        self.dispatch_wide(k, &bufs, 1, &push, groups);
    }

    /// `-DSTREAMED` twin of [`Self::linear_mmv_mw`] (slice A2 build-variant: see
    /// `crate::gemm::native_mmv_mw_spv`). `residual`: fused `y = residual + x·Wᵀ`,
    /// same convention as the resident fn — the `_res_streamed` SPV already exists in the build
    /// table (see `native_mmv_mw_spv`'s `res` column), this wrapper just wasn't
    /// exposing it until the resident-BDA routing fork in [`Self::linear_mmv_mw`] needed it (a
    /// non-unified-mmv-row1 GPU can reach the fused-add peephole on a wave32-native device with a
    /// resident-BDA weight).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_mw_at(
        &self,
        dtype: infr_core::DType,
        warps: u32,
        arena_addr: u64,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        residual: Option<&dyn Buffer>,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_mmv_mw_at", in_f);
        self.label_gemv("mmv_mw_streamed", 1, in_f, out_f);
        let res = residual.is_some();
        let (name, spv) = crate::gemm::native_mmv_mw_spv(dtype, res, warps, self.sg16())
            .expect("native mmv_mw streamed spv");
        let nbind = if res { 6 } else { 5 };
        let k = self.be.kernel_sg(name, spv, nbind, 24, self.sgp());
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let mut bufs = vec![
            Self::vkb(qa),
            Self::vkb(qa),
            Self::vkb(dact),
            Self::vkb(sact),
        ];
        if let Some(r) = residual {
            bufs.push(Self::vkb(r));
        }
        bufs.push(Self::vkb(y));
        self.dispatch_wide(k, &bufs, 1, &push, (out_f as u32).div_ceil(warps));
    }

    /// Multi-row int8 dp4a GEMV (`1 <= rows <= 16`): [`Self::linear_native_mrow`]'s shape with
    /// `native_mmv.comp`'s integer-dot math — the weight sub-block is unpacked once into packed
    /// int8 words and dp4a'd against every row's pre-quantized activation block (`qa`/`dact`/
    /// `sact` from [`Self::quant_q8`] over `rows` rows). NUM_ROWS=2 outputs per workgroup
    /// (grid = ceil(out_f/2)). Caller gates on [`crate::gemm::native_mmv_mrow_kernel_name`].
    ///
    /// `rows == 1` is not a special case: it is THE m=1 decode kernel now (see
    /// `native_mmv_mrow.comp`'s header). A row's math depends only on its own `qa`/`dact`/`sact`
    /// slice and is bit-identical whether `rows` is 1 or 16 — same threads, same sub-block
    /// striding, same accumulation order, same subgroupAdd width — which is exactly the guarantee
    /// `mmv_row1_bit_identical` proves and MTP token-identity needs. `residual`: fused
    /// `y = residual + x·Wᵀ` (only legal at `rows == 1`, the decode Linear+Add fusion — asserted)
    /// or `None` for the plain `y = x·Wᵀ` write used at every `rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_mrow(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        residual: Option<&dyn Buffer>,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        debug_assert!((1..=16).contains(&rows));
        debug_assert!(
            residual.is_none() || rows == 1,
            "fused residual is decode-only (rows=1)"
        );
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_mmv_mrow requires a u64 BDA device address");
        self.linear_mmv_mrow_at(
            dtype, arena_addr, w_base, qa, dact, sact, residual, y, rows, in_f, out_f,
        );
    }

    /// FUSE_QUANT variant of [`Self::linear_mmv_mrow`] — reads f32 activations directly and
    /// quantizes per 32-block inline, saving the separate `quant_q8` dispatch. Same push layout
    /// and BDA weight convention, but bindings are reduced (no qa/dact/sact buffers).
    /// Opt-in behind INFR_MMV_FUSE_QUANT=1; Q4_K/Q6_K only.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_mrow_fuseq(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        x: &dyn Buffer, // [rows, in_f] f32 activations
        residual: Option<&dyn Buffer>,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        debug_assert!(
            (1..=4).contains(&rows),
            "FUSE_QUANT is decode-only (rows<=4)"
        );
        debug_assert!(
            residual.is_none() || rows == 1,
            "fused residual is decode-only (rows=1)"
        );
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_mmv_mrow_fuseq requires a u64 BDA device address");
        let o4 = in_f < 2048 && self.vk().mmv_o4;
        let m4 = rows <= 4;
        let res = residual.is_some();
        let (name, spv) = crate::gemm::native_mmv_mrow_variant_fuseq_spv(dtype, o4, m4, res)
            .expect("FUSE_QUANT SPIR-V");
        let nbind: usize = if res { 3 } else { 2 };
        let k = self.be.kernel(name, spv, nbind, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let groups = (out_f as u32).div_ceil(if o4 && rows <= 8 { 4 } else { 2 });
        let mut bufs = vec![Self::vkb(x)];
        if let Some(r) = residual {
            bufs.push(Self::vkb(r));
        }
        bufs.push(Self::vkb(y));
        // FILLER: the fused kernel has only 2-3 bindings but the BDA-read weight's SSBO slot
        // (binding 0) must still be a legal descriptor. x is already at binding 0 — ok.
        self.dispatch_wide(k, &bufs, 1, &push, groups);
    }

    /// Native-block dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_native(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_add_native requires a u64 BDA device address");
        self.linear_add_native_at(dtype, arena_addr, x, residual, y, rows, in_f, out_f);
    }

    /// `-DSTREAMED` twin of [`Self::linear_add_native`] (fused-residual native dequant GEMV).
    /// Mirrors the resident dispatcher's SG → variant → RM → plain-GEMV routing precedence EXACTLY
    /// (same heuristics, same env escapes) so a parity test comparing the two at the same shape
    /// lands on the same kernel. Same arena-by-address convention as
    /// [`Self::linear_native_at`] — binding 0 (the resident build's weight SSBO) takes a small
    /// FILLER (`x`) since the arena is never bound as a descriptor; `arena_lo`/`arena_hi` are
    /// appended LAST to the push block, after the resident `w_base` slot (always 0 — the residual
    /// native GEMV is not used for stacked experts). Not wired into any production dispatch yet
    /// (today's fused-add peephole filters streamed weights out before this path is ever reached —
    /// see adapter.rs's Linear+Add lowering); exists so a parity test can exercise the `_streamed`
    /// SPV directly ahead of the resident-BDA endgame that will need it.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_native_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_add_native_at", in_f);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        // push[12..16] = w_base, 0 (residual native GEMV is not used for stacked experts).
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let bufs = [
            Self::vkb(x),
            Self::vkb(x),
            Self::vkb(residual),
            Self::vkb(y),
        ];
        // Reassociation-tolerant subgroup route — precedence over RM, mirrors linear_add_native.
        if rows == 1 {
            if let Some(nr) = native_sg_choice(dtype, in_f, out_f, self.gemv()) {
                if let Some((name, spv)) =
                    crate::gemm::native_sg_streamed_build_spv(dtype, true, nr, self.sg16())
                {
                    let k = self.be.kernel_sg(name, spv, 4, 24, self.sgp());
                    self.dispatch_wide(k, &bufs, 1, &push, (out_f as u32).div_ceil(nr));
                    return;
                }
            }
        }
        // Multi-output-row route — mirrors linear_add_native's variant-then-RM fallback.
        if rows == 1 {
            if let Some(v) = &self.gemv().variant {
                if let Some((name, spv)) =
                    crate::gemm::native_rm_v2_streamed_build_spv(v, dtype, true)
                {
                    let rm: u32 = 2;
                    let k = self.be.kernel(name, spv, 4, 24);
                    self.dispatch_wide(k, &bufs, 1, &push, (out_f as u32).div_ceil(rm));
                    return;
                }
            }
            if let Some(rm) = native_rm_choice(dtype, out_f, self.gemv()) {
                if let Some((name, spv)) =
                    crate::gemm::native_rm_streamed_build_spv(dtype, true, rm)
                {
                    let k = self.be.kernel(name, spv, 4, 24);
                    self.dispatch_wide(k, &bufs, 1, &push, (out_f as u32).div_ceil(rm));
                    return;
                }
            }
        }
        let (name, spv) = crate::gemm::native_streamed_build_spv(dtype, true)
            .expect("native streamed GEMV res spv");
        let k = self.be.kernel(name, spv, 4, 24);
        self.dispatch_wide(k, &bufs, 1, &push, (rows * out_f) as u32);
    }

    /// Gather + dequantize embedding rows (`Op::EmbedGather`): `dst[r,:] = table[ids[r],:] *
    /// scale`. One workgroup per row; per-format decode from native_decode.glsl. Caller gates on
    /// [`crate::gemm::embed_gather_kernel_name`].
    #[allow(clippy::too_many_arguments)]
    pub fn embed_gather(
        &self,
        dtype: infr_core::DType,
        table: &dyn Buffer,
        ids: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        ne: usize,
        scale: f32,
    ) {
        let arena_addr = table
            .device_addr()
            .expect("resident-BDA weight: embed_gather requires a u64 BDA embedding table address");
        // A table whose element count `>= 2^32` (a 256k-vocab frontier model) would wrap the
        // shader's u32 `ids*ne` row base — pass the row byte stride so it folds `ids` into the
        // 64-bit table base instead (issue #77). Every current table fits u32 → `row_bytes = 0` →
        // the shader keeps its original element-offset path, byte-identical (incl. gemma's
        // non-word-aligned quant rows, which the 64-bit fold could not honor). The fold requires a
        // word-aligned row stride, asserted loudly ONLY when it actually engages.
        let (block_elems, block_bytes) = infr_gguf::block_layout(dtype);
        assert!(
            ne.is_multiple_of(block_elems),
            "embed_gather: table row (ne={ne}) is not a whole number of {dtype:?} blocks ({block_elems})"
        );
        let row_stride = (ne / block_elems) * block_bytes;
        let vocab = table.len_bytes() / row_stride.max(1);
        // Fold when the table's element count reaches the addressing cap (the `ids*ne` u32 base
        // would wrap). Driving this off `bda_chunk_elem_cap` — not a hardcoded 2^32 — lets
        // `INFR_BDA_CHUNK_ELEMS` force the fold path on a normal model, so the forced-chunking proof
        // exercises BOTH the lm_head GEMV split and this embed fold in one run.
        let row_bytes = if (vocab as u64) * (ne as u64) >= self.chunk_caps().0 {
            assert!(
                row_stride.is_multiple_of(4),
                "embed_gather: a >= 2^32-element {dtype:?} table (vocab={vocab}, ne={ne}) has a \
                 non-word-aligned row stride ({row_stride} bytes) — the 64-bit-based gather it needs \
                 would misalign the buffer_reference reads; this table is unsupported"
            );
            row_stride as u32
        } else {
            0 // fits u32: keep the original element-offset gather path
        };
        if row_bytes != 0 && self.be.cfg().debug.bda_chunk {
            eprintln!(
                "[infr] embed_gather {dtype:?} vocab={vocab} ne={ne} -> 64-bit row-base fold \
                 (row_bytes={row_bytes})"
            );
        }
        self.embed_gather_at(dtype, arena_addr, ids, dst, rows, ne, scale, row_bytes);
    }

    /// `-DSTREAMED` twin of [`Self::embed_gather`] (slice A5 build-variant: see
    /// `crate::gemm::embed_gather_spv`). The token-embedding table is read from a
    /// resident `bufferDeviceAddress` arena instead of a bound SSBO — binding 0 (the resident
    /// build's weight SSBO) takes a harmless FILLER (`ids`) since the arena is never bound as a
    /// descriptor. `row_bytes` is the mode selector (0 = element-offset gather for a u32-fitting
    /// table; nonzero = 64-bit row-base fold for a `>= 2^32`-element table — see the `.comp` PC doc).
    #[allow(clippy::too_many_arguments)]
    pub fn embed_gather_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        ids: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        ne: usize,
        scale: f32,
        row_bytes: u32,
    ) {
        // The shader gathers in whole 32-element sub-blocks (`nsub = ne / 32`), so a ragged tail of
        // `ne % 32` elements is left unwritten (uninitialized embedding rows) and `ne < 32` writes
        // nothing at all — the same silent-zeros class as the GEMV families, guarded the same way
        // (see [`assert_native_k`], which also says why this is a hard assert and not debug-only).
        assert_native_k("embed_gather_at (row width ne)", ne);
        let (name, spv) = crate::gemm::embed_gather_spv(dtype).expect("embed_gather streamed spv");
        let k = self.be.kernel(name, spv, 3, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&scale.to_ne_bytes());
        // Arena base BYTE address (lo/hi split — a uvec2 avoids the 8-byte push alignment a
        // uint64_t member forces; see native_weight_addr.glsl), appended before row_bytes.
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&row_bytes.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(ids), Self::vkb(ids), Self::vkb(dst)],
            1,
            &push,
            rows as u32,
        );
    }

    /// I32 lookup-table row gather (`gather_i32.comp`, `Op::GatherI32`):
    /// `dst[r, j] = table[ids[r], j]`, integers copied verbatim. One thread per output element.
    ///
    /// The table is BOUND (not read through the resident-BDA arena like [`Self::embed_gather`]'s):
    /// `ffn_gate_tid2eid` is `n_expert_used * n_vocab` dwords, far inside `maxStorageBufferRange`,
    /// which [`Self::vkb`] asserts for a resident-BDA sub-tensor anyway.
    pub fn gather_i32(
        &self,
        table: &dyn Buffer,
        ids: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        ne: usize,
    ) {
        let k = self
            .be
            .kernel("gather_i32", crate::gemm::gather_i32_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(table), Self::vkb(ids), Self::vkb(dst)],
            1,
            &push,
            (rows * ne).div_ceil(64) as u32,
        );
    }

    /// Int8 dp4a decode GEMV (m=1): `y = x·Wᵀ` with `x` pre-quantized via [`Self::quant_q8`]
    /// (qa/dact/sact). NUM_ROWS=2 — one workgroup per 2 consecutive outputs (`ceil(out_f/2)`
    /// grid), the activation block read once for both. `w_base` = element offset (fused-QKV
    /// slices). Caller gates on [`crate::gemm::native_mmv_kernel_name`].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_mmv requires a u64 BDA device address");
        self.linear_mmv_at(dtype, arena_addr, w_base, qa, dact, sact, y, in_f, out_f);
    }

    /// Int8 dp4a decode GEMV with fused residual add: `y = residual + x·Wᵀ` (see
    /// [`Self::linear_mmv`]).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_mmv(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_add_mmv requires a u64 BDA device address");
        self.linear_add_mmv_at(dtype, arena_addr, qa, dact, sact, residual, y, in_f, out_f);
    }

    /// `-DSTREAMED` twin of [`Self::linear_add_mmv`] (fused-residual int8 dp4a decode GEMV). Same
    /// arena-by-address / FILLER convention as [`Self::linear_mmv_at`] — binding 0 (the
    /// resident build's weight SSBO) takes a small FILLER (`qa`) since the arena is never bound as a
    /// descriptor; `residual` is inserted right before the trailing `y` write (6 bindings total, vs
    /// the plain arm's 5), and `arena_lo`/`arena_hi` are appended LAST to the push block. Not wired
    /// into any production dispatch yet (today's fused-add peephole filters streamed weights out
    /// before this path is ever reached — see adapter.rs's Linear+Add lowering); exists so a parity
    /// test can exercise the `_streamed` SPV directly ahead of the resident-BDA endgame.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add_mmv_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_add_mmv_at", in_f);
        self.label_gemv("mmv_add_streamed", 1, in_f, out_f);
        let (name, spv) =
            crate::gemm::native_mmv_spv(dtype, true).expect("native mmv res streamed spv");
        let k = self.be.kernel(name, spv, 6, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&1u32.to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        // push[12..16] = w_base, 0 (the residual GEMV never reads stacked experts).
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(qa),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(residual),
                Self::vkb(y),
            ],
            1,
            &push,
            (out_f as u32).div_ceil(2),
        );
    }

    /// Multi-warp int8 dp4a decode GEMV (`native_mmv_mw.comp`, `warps` rows/block, subgroupAdd).
    /// Wave32-native GPUs only. `res` selects the fused-residual variant (extra `residual` binding).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_mw(
        &self,
        dtype: infr_core::DType,
        warps: u32,
        w: &dyn Buffer,
        w_base: usize,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        residual: Option<&dyn Buffer>,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_mmv_mw requires a u64 BDA device address");
        self.linear_mmv_mw_at(
            dtype, warps, arena_addr, w_base, qa, dact, sact, residual, y, in_f, out_f,
        );
    }

    /// `y = residual + x·Wᵀ` (fused residual add). `residual` and `y` may be the same buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_add(
        &self,
        w: &dyn Buffer,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_add requires a u64 BDA device address");
        self.linear_add_at(arena_addr, x, residual, y, rows, in_f, out_f);
    }

    /// Fused RMSNorm + in-place add: `dst[r*..] += rmsnorm(x)[r*..] * w`. One workgroup per row.
    /// Replaces the RmsNorm + Add dispatch pair (E2B per-layer projection tail).
    pub fn rmsnorm_add(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        let k = self
            .be
            .kernel_sg("rmsnorm_add", crate::gemm::rmsnorm_add_spv(), 3, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(dst)],
            1,
            &push,
            rows as u32,
        );
    }

    /// MLA attention (DeepSeek V2/V3 absorbed form). One workgroup per (token, head) — the dispatch
    /// below is `rows * n_head` groups — and that group's 128 lanes cooperate on the one head's
    /// math. The KV cache holds ONE row per token (key_length = kv_lora_rank + qk_rope_dim wide).
    ///
    /// The kernel's `sq`/`sv` shared arrays are fixed-size; the caller must have checked the dims
    /// against `adapter::MLA_MAX_KEY_LEN` / `adapter::MLA_MAX_KV_LORA_RANK` (the `Op::Mla` arm of
    /// `lower_op` does) — a wider latent overruns shared memory here with no diagnostic.
    #[allow(clippy::too_many_arguments)]
    pub fn mla(
        &self,
        q: &dyn Buffer,
        k_cache: &dyn Buffer,
        wk_b: &dyn Buffer,
        wv_b: &dyn Buffer,
        dst: &dyn Buffer,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        q_head_dim: u32,
        kv_lora_rank: u32,
        qk_nope_dim: u32,
        qk_rope_dim: u32,
        v_head_dim: u32,
        scale: f32,
        pos: u32,
        mask_type: u32,
        window: u32,
        theta: f32,
        cache_cap_rows: u32,
        // `AttnMask::Canvas { lo }`'s lower bound. Read by the shader on `mask_type == 2` only;
        // pass 0 for the causal and sliding-window masks, which derive their own `lo`.
        canvas_lo: u32,
        // YaRN per-pair frequency divisors for the internal q_pe rope (`qk_rope_dim/2` floats) —
        // selects the `-DFREQ_FACTORS` kernel build (a compile-time presence; the plain `mla`
        // build has no ff binding).
        freq_factors: Option<&dyn Buffer>,
        // deepseek32's additive `[rows, kv_len]` top-k score mask — selects the `-DKEY_BIAS`
        // build. Independent of `freq_factors`, hence the four kernels. Both optional reads bind
        // BEFORE `dst`, which must stay the trailing (hazard-tracked write) binding.
        key_bias: Option<&dyn Buffer>,
    ) {
        let mut push = [0u8; 60];
        push[0..4].copy_from_slice(&rows.to_ne_bytes());
        push[4..8].copy_from_slice(&kv_len.to_ne_bytes());
        push[8..12].copy_from_slice(&n_head.to_ne_bytes());
        push[12..16].copy_from_slice(&q_head_dim.to_ne_bytes());
        push[16..20].copy_from_slice(&kv_lora_rank.to_ne_bytes());
        push[20..24].copy_from_slice(&qk_nope_dim.to_ne_bytes());
        push[24..28].copy_from_slice(&qk_rope_dim.to_ne_bytes());
        push[28..32].copy_from_slice(&v_head_dim.to_ne_bytes());
        push[32..36].copy_from_slice(&scale.to_ne_bytes());
        push[36..40].copy_from_slice(&pos.to_ne_bytes());
        push[40..44].copy_from_slice(&mask_type.to_ne_bytes());
        push[44..48].copy_from_slice(&window.to_ne_bytes());
        push[48..52].copy_from_slice(&theta.to_ne_bytes());
        push[52..56].copy_from_slice(&cache_cap_rows.to_ne_bytes());
        push[56..60].copy_from_slice(&canvas_lo.to_ne_bytes());
        // The optional reads bind BEFORE dst: the hazard tracker treats the trailing `n_out`
        // bindings as writes, and `dst` must be the tracked one — binding ff/kbias last would mark
        // read-only buffers as written and leave `dst` untracked (no barrier before the
        // wo-projection read → nondeterminism).
        let mut bufs = vec![
            Self::vkb(q),
            Self::vkb(k_cache),
            Self::vkb(wk_b),
            Self::vkb(wv_b),
        ];
        bufs.extend(freq_factors.map(Self::vkb));
        bufs.extend(key_bias.map(Self::vkb));
        bufs.push(Self::vkb(dst));
        let sg = crate::gemm::mla_sg_tier(self.vk(), self.be.caps());
        let (name, spv) = crate::gemm::mla_kernel(sg, freq_factors.is_some(), key_bias.is_some());
        // The `-DMLA_SG` builds read `gl_SubgroupID` and index a `MLA_NSG`-wide shared array, both
        // sized from the pinned width — so they go through `kernel_sg`, which sets
        // `requiredSubgroupSize` AND `REQUIRE_FULL_SUBGROUPS`. The scalar builds need neither and
        // keep the unpinned pipeline they have always had.
        let k = if sg {
            self.be
                .kernel_sg(name, spv, bufs.len(), 60, crate::gemm::MLA_SG_WIDTH)
        } else {
            self.be.kernel(name, spv, bufs.len(), 60)
        };
        self.dispatch_wide(k, &bufs, 1, &push, rows * n_head);
    }

    /// Expand `Op::LightningIndexer`'s `[rows, top_k]` key indices into the additive `[rows,
    /// kv_len]` f32 score mask `Recorder::mla`'s `key_bias` adds (`topk_mask.comp`,
    /// `Op::TopkMask`): one workgroup per query row, `-inf` fill then a scatter of `0.0`.
    pub fn topk_mask(
        &self,
        idx: &dyn Buffer,
        dst: &dyn Buffer,
        rows: u32,
        kv_len: u32,
        top_k: u32,
    ) {
        let k = self
            .be
            .kernel("topk_mask", crate::gemm::topk_mask_spv(), 2, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&rows.to_ne_bytes());
        push[4..8].copy_from_slice(&kv_len.to_ne_bytes());
        push[8..12].copy_from_slice(&top_k.to_ne_bytes());
        self.dispatch_wide(k, &[Self::vkb(idx), Self::vkb(dst)], 1, &push, rows);
    }

    /// DeepSeek V4 hyper-connection coefficients (`hyper_mix.comp`, `Op::HyperConnectMix`): one
    /// thread per token computes `pre`, and — when `gates` is `Some((post, comb))` — `post` plus
    /// the Sinkhorn-normalised `hc x hc` mixing matrix. `mix_dim` is `(2 + hc)*hc` with gates and
    /// `hc` without (`build_hc_head`'s narrower `mixes`).
    #[allow(clippy::too_many_arguments)]
    pub fn hyper_mix(
        &self,
        mixes: &dyn Buffer,
        scale: &dyn Buffer,
        base: &dyn Buffer,
        pre: &dyn Buffer,
        gates: Option<(&dyn Buffer, &dyn Buffer)>,
        rows: u32,
        hc: u32,
        eps: f32,
        n_iter: u32,
    ) {
        let mut bufs = vec![
            Self::vkb(mixes),
            Self::vkb(scale),
            Self::vkb(base),
            Self::vkb(pre),
        ];
        let mix_dim = match gates {
            Some((post, comb)) => {
                bufs.push(Self::vkb(post));
                bufs.push(Self::vkb(comb));
                (2 + hc) * hc
            }
            None => hc,
        };
        let (name, spv) = match gates {
            Some(_) => ("hyper_mix_gates", crate::gemm::hyper_mix_gates_spv()),
            None => ("hyper_mix", crate::gemm::hyper_mix_spv()),
        };
        let k = self.be.kernel(name, spv, bufs.len(), 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&rows.to_ne_bytes());
        push[4..8].copy_from_slice(&hc.to_ne_bytes());
        push[8..12].copy_from_slice(&mix_dim.to_ne_bytes());
        push[12..16].copy_from_slice(&eps.to_ne_bytes());
        push[16..20].copy_from_slice(&n_iter.to_ne_bytes());
        // `pre` and (with gates) `post` + `comb` are the writes — the trailing bindings.
        let n_out = if gates.is_some() { 3 } else { 1 };
        self.dispatch(k, &bufs, n_out, &push, rows.div_ceil(64));
    }

    /// DeepSeek V4 hyper-connection stream collapse (`hyper_pre.comp`, `Op::HyperConnectPre`):
    /// `dst[t, i] = Σ_h x[t, h, i] * w[t, h]`, one thread per output element.
    pub fn hyper_pre(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        dst: &dyn Buffer,
        rows: u32,
        hc: u32,
        n_embd: u32,
    ) {
        let total = rows * n_embd;
        let k = self
            .be
            .kernel("hyper_pre", crate::gemm::hyper_pre_spv(), 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&hc.to_ne_bytes());
        push[4..8].copy_from_slice(&n_embd.to_ne_bytes());
        push[8..12].copy_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(dst)],
            1,
            &push,
            total.div_ceil(64),
        );
    }

    /// DeepSeek V4 hyper-connection stream re-expansion (`hyper_post.comp`,
    /// `Op::HyperConnectPost`): `dst[t, d, i] = x[t,i]*post[t,d] + Σ_src res[t,src,i]*comb[t,
    /// d + hc*src]`, one thread per output element.
    #[allow(clippy::too_many_arguments)]
    pub fn hyper_post(
        &self,
        x: &dyn Buffer,
        residual: &dyn Buffer,
        post: &dyn Buffer,
        comb: &dyn Buffer,
        dst: &dyn Buffer,
        rows: u32,
        hc: u32,
        n_embd: u32,
    ) {
        let total = rows * hc * n_embd;
        let k = self
            .be
            .kernel("hyper_post", crate::gemm::hyper_post_spv(), 5, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&hc.to_ne_bytes());
        push[4..8].copy_from_slice(&n_embd.to_ne_bytes());
        push[8..12].copy_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(x),
                Self::vkb(residual),
                Self::vkb(post),
                Self::vkb(comb),
                Self::vkb(dst),
            ],
            1,
            &push,
            total.div_ceil(64),
        );
    }

    /// DeepSeek V3.2 lightning indexer (`lightning_indexer.comp`, `Op::LightningIndexer`): scores
    /// every cached key against the row's indexer heads and writes the top-`top_k` key indices.
    /// One workgroup per query row — the dispatch below is `rows` groups, and each group's 256
    /// lanes split the KEY axis.
    ///
    /// `scores` is caller-provided scratch of at least `rows * kv_len` f32s; the kernel fills it
    /// (phase 1) and reads it back (phase 2), so it must not alias anything else live. It is
    /// counted as a written binding alongside `dst` so the hazard tracker barriers it.
    #[allow(clippy::too_many_arguments)]
    pub fn lightning_indexer(
        &self,
        q: &dyn Buffer,
        k_cache: &dyn Buffer,
        w: &dyn Buffer,
        scores: &dyn Buffer,
        dst: &dyn Buffer,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        head_dim: u32,
        top_k: u32,
        scale: f32,
        pos: u32,
    ) {
        let k = self.be.kernel(
            "lightning_indexer",
            crate::gemm::lightning_indexer_spv(),
            5,
            28,
        );
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&rows.to_ne_bytes());
        push[4..8].copy_from_slice(&kv_len.to_ne_bytes());
        push[8..12].copy_from_slice(&n_head.to_ne_bytes());
        push[12..16].copy_from_slice(&head_dim.to_ne_bytes());
        push[16..20].copy_from_slice(&top_k.to_ne_bytes());
        push[20..24].copy_from_slice(&scale.to_ne_bytes());
        push[24..28].copy_from_slice(&pos.to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(q),
                Self::vkb(k_cache),
                Self::vkb(w),
                Self::vkb(scores),
                Self::vkb(dst),
            ],
            2, // scores + dst are both written
            &push,
            rows,
        );
    }

    pub fn rmsnorm(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        // 256-thread subgroup kernel (requiredSubgroupSize=32): more load/store parallelism and a
        // single barrier vs the 64-thread WGSL shared-tree. ~2.6× faster as a kernel.
        //
        // DECODE (rows==1) takes the -DWIDE twin instead. The kernel is one-workgroup-per-row, so
        // rows==1 is a SINGLE workgroup: at 256 threads that is 8 wave32s on one WGP with a ~21 KB
        // row, i.e. memory-LATENCY bound with barely any requests in flight. Profiled on
        // gemma-4-31B (dim 5376) it cost 10.5 us/dispatch — vs ~1.2 us for `add` over the SAME
        // vector, which fans out to dim/64 workgroups — and 301 dispatches/token made it 8.9% of
        // ALL decode GPU time. WIDE keeps the one workgroup (the reduction can't be split without a
        // second dispatch) but runs it 1024-wide with vec4 loads: 4x the waves x 4x the bytes per
        // request. Gate is exactly the decode hidden-norm shape — prefill (rows>1) already fills
        // the GPU with row-workgroups, and the per-head Q/K/V norms (rows=n_head, small dim) would
        // just idle 3/4 of the lanes, so both keep the 256-thread build.
        let wide = rows == 1 && dim >= 2048 && dim.is_multiple_of(4);
        let k = if wide {
            self.be
                .kernel_sg("rmsnorm_wide", crate::gemm::rmsnorm_wide_spv(), 3, 12, 32)
        } else {
            self.be
                .kernel_sg("rmsnorm", crate::gemm::rmsnorm_spv(), 3, 12, 32)
        };
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(y)],
            1,
            &push,
            rows as u32, // one workgroup per row (cooperative reduction)
        );
    }

    /// WEIGHTLESS RMSNorm: `y[i] = x[i] * inversesqrt(mean(x²) + eps)`, one workgroup per row
    /// (`rmsnorm.comp`'s -DNO_WEIGHT build) — `Op::QkNorm { weight: None }`, deepseek4's bare
    /// per-head `ggml_rms_norm` on Q. Same reduction as [`Self::rmsnorm`], one binding fewer.
    pub fn rmsnorm_nw(&self, x: &dyn Buffer, y: &dyn Buffer, rows: usize, dim: usize, eps: f32) {
        let k = self
            .be
            .kernel_sg("rmsnorm_nw", crate::gemm::rmsnorm_nw_spv(), 2, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            rows as u32, // one workgroup per row (cooperative reduction)
        );
    }

    /// Mean-centred LayerNorm: `y[i] = (x[i] - mean) * inversesqrt(var + eps) * w[i] + b[i]` per
    /// row (`layernorm.comp`, `Op::LayerNorm`) — deepseek32's `indexer_k_norm`, the only non-RMS
    /// norm in the family. One workgroup per row, like [`Self::rmsnorm`], but two cooperative
    /// reductions instead of one (the mean has to land before the variance sum can start).
    pub fn layernorm(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        b: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        let k = self
            .be
            .kernel_sg("layernorm", crate::gemm::layernorm_spv(), 4, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(b), Self::vkb(y)],
            1,
            &push,
            rows as u32, // one workgroup per row (cooperative reduction)
        );
    }

    /// Like [`Self::rmsnorm`], but `x`/`y` are f16 instead of f32 (`rmsnorm.comp`'s -DF16IO build)
    /// — llama4's post-rope weightless Q/K L2-norm (`Op::QkNorm` in-place on the f16 rope scratch,
    /// `x == dst`). `w` stays f32 either way.
    pub fn rmsnorm_f16(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        let k = self
            .be
            .kernel_sg("rmsnorm_f16", crate::gemm::rmsnorm_f16_spv(), 3, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(y)],
            1,
            &push,
            rows as u32,
        );
    }

    /// Fused per-head RMSNorm + SiLU gate multiply: `y[i] = rmsnorm(x)[i] * silu(z[i])`, one
    /// workgroup per (row, head) — `rmsnorm.comp`'s -DGATE build (`Op::GatedRmsNorm`, qwen35
    /// DeltaNet z-gate). Same reduction as `rmsnorm`; `z` must be the SAME shape/layout as `x`
    /// (`rows*dim` flat, indexed identically — see `Op::GatedRmsNorm`'s doc). Removes the
    /// read-after-write barrier a separate `rmsnorm`→`silu_mul` pair would need (GatedAct reading
    /// QkNorm's freshly-written output).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_gate(
        &self,
        x: &dyn Buffer,
        w: &dyn Buffer,
        z: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        let k = self
            .be
            .kernel_sg("rmsnorm_gate", crate::gemm::rmsnorm_gate_spv(), 4, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&eps.to_ne_bytes());
        // Binding order MUST be x, w, z, y (matches the shader's -DGATE layout) — `y` last so the
        // hazard tracker's `n_out=1` marks only the true output as written (see the shader's doc).
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(w), Self::vkb(z), Self::vkb(y)],
            1,
            &push,
            rows as u32, // one workgroup per row (cooperative reduction)
        );
    }

    /// Row-wise softmax: `y[r,:] = softmax(x[r,:] * scale)` over `dim` columns, one workgroup per
    /// row (diffusion-gemma's in-graph self-conditioning — see docs/diffusion-gemma.md's Phase-B
    /// and the reference's `dg_canvas_embed`). Same 256-thread subgroup-reduction shape as
    /// `rmsnorm` — `dim` here is the vocab, so the cooperative reduction matters just as much.
    pub fn softmax(&self, x: &dyn Buffer, y: &dyn Buffer, rows: usize, dim: usize, scale: f32) {
        let k = self
            .be
            .kernel_sg("softmax", crate::gemm::softmax_spv(), 2, 12, 32);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(x), Self::vkb(y)], 1, &push, rows as u32);
    }

    /// Like [`Self::softmax`], but the scale comes from `scale_buf[0]` (a 1-element device buffer,
    /// host-updated via a tiny 4-byte `upload` between calls) instead of a push constant — the
    /// DiffusionGemma denoise self-conditioning path uses this to vary the softmax temperature
    /// every step on a plan compiled/cached ONCE (see `Op::Softmax::scale_buf`'s doc and
    /// docs/diffusion-gemma.md's Phase-B). `USE_SCALE_BUF`-compiled variant of the same shader.
    pub fn softmax_dyn(
        &self,
        x: &dyn Buffer,
        scale_buf: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        dim: usize,
    ) {
        let k = self
            .be
            .kernel_sg("softmax_dyn", crate::gemm::softmax_dyn_spv(), 3, 8, 32);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(dim as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(scale_buf), Self::vkb(y)],
            1,
            &push,
            rows as u32,
        );
    }

    /// RoPE in place is allowed (`x` and `y` may be the same buffer). `pos_offset` shifts the
    /// absolute position of the first row (for cached decode).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &dyn Buffer,
        y: &dyn Buffer,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        pos_offset: usize,
        // YaRN per-pair frequency divisors (`rope_dim/2` floats) — binds at 2 and selects the
        // `rope_ff` kernel build (compile-time presence; the plain `rope` build has no ff binding).
        freq_factors: Option<&dyn Buffer>,
        // NEOX split-half pairing instead of the interleaved NORM one (`Op::Rope::neox`) — a
        // second compile-time build rather than a push-constant branch, so the existing NORM
        // pipelines and their push layout are untouched.
        neox: bool,
        // De-rope: negate `sin`, leave `cos` (`Op::Rope::backward` / `ggml_rope_ext_back`). Same
        // compile-time-build treatment as `neox`. NORM only — deepseek4's de-rope is NORM, and a
        // build nothing dispatches is a build nothing tests, so the adapter refuses neox+backward.
        backward: bool,
    ) {
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        // [24..28] out_base: 0 (in-place f32 rope has no output shift)
        match freq_factors {
            Some(ff) => {
                let k = if neox {
                    self.be
                        .kernel("rope_ff_neox", crate::gemm::rope_ff_neox_spv(), 3, 28)
                } else if backward {
                    self.be
                        .kernel("rope_ff_back", crate::gemm::rope_ff_back_spv(), 3, 28)
                } else {
                    self.be.kernel("rope_ff", crate::gemm::rope_ff_spv(), 3, 28)
                };
                self.dispatch(
                    k,
                    // ff BEFORE y: the trailing `n_out` bindings are the hazard-tracked writes,
                    // and `y` (index 2) must be that write — binding ff last would mark the
                    // read-only divisors as written and leave `y` untracked (same race as mla_ff).
                    &[Self::vkb(x), Self::vkb(ff), Self::vkb(y)],
                    1,
                    &push,
                    (t * n_heads) as u32,
                );
            }
            None => {
                let k = if neox {
                    self.be
                        .kernel("rope_neox", crate::gemm::rope_neox_spv(), 2, 28)
                } else if backward {
                    self.be
                        .kernel("rope_back", crate::gemm::rope_back_spv(), 2, 28)
                } else {
                    self.be.kernel("rope", crate::gemm::rope_spv(), 2, 28)
                };
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(y)],
                    1,
                    &push,
                    (t * n_heads) as u32,
                );
            }
        }
    }

    /// Interleaved (llama NORM) RoPE writing f16 — the llama q/k analogue of `qk_norm_rope`'s
    /// f16 output: `out_base` shifts the output row (0 for the Q scratch; `pos` for the fused
    /// K cache write via the kv_write peephole).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f16(
        &self,
        x: &dyn Buffer,
        y: &dyn Buffer,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        pos_offset: usize,
        out_base: usize,
    ) {
        let k = self
            .be
            .kernel("rope_f16", crate::gemm::rope_f16_spv(), 2, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// Record-once variant of [`Self::rope_f16`]: pos from `params`; `out_base_mul` is the 0/1
    /// multiplier the shader scales by pos (1 -> write cache row pos, 0 -> row 0 of the Q scratch).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f16_dyn(
        &self,
        x: &dyn Buffer,
        params: &dyn Buffer,
        y: &dyn Buffer,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
    ) {
        let k = self
            .be
            .kernel("rope_f16_dyn", crate::gemm::rope_f16_dyn_spv(), 3, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] pos_offset: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(params), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// Cached attention: q `[q_len,nh,hd]` (abs positions `pos_offset..`) over the K/V cache
    /// `[kv_len,nkv,hd]`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        // Sliding-window attention: a query at abs pos `p` attends only keys `j > p - window`.
        // `0` = full causal (llama/qwen3 + gemma full-attention layers).
        window: usize,
        // QK scale: `> 0` overrides the default 1/√hd (gemma4 passes 1.0). `0.0` = default.
        scale: f32,
        // Per-side planar Q8_0 KV read (K and V independent). `cap` = total cache elements (planar
        // scales base), unused when both are f16.
        k_q8: bool,
        v_q8: bool,
        cap: usize,
    ) {
        let (name, spv) = match (k_q8, v_q8) {
            (false, false) => ("attention_kv", crate::gemm::attention_kv_spv()),
            (true, false) => ("attention_kv_kq8", crate::gemm::attention_kv_kq8_spv()),
            (false, true) => ("attention_kv_vq8", crate::gemm::attention_kv_vq8_spv()),
            (true, true) => ("attention_kv_q8", crate::gemm::attention_kv_q8_spv()),
        };
        let kern = self.be.kernel(name, spv, 4, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        push[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(o)],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// [`Self::attention_kv`] plus per-head ATTENTION SINKS (`attention_kv.comp`'s -DSINKS build,
    /// `Op::Attention::sinks`): `sinks[nh]` f32 binds at slot 3 (output last at 4), and each head's
    /// sink logit joins the softmax max and denominator without contributing a value row.
    ///
    /// f16 KV, bound, static — no Q8/BDA/params twins, so the adapter must have already refused
    /// those shapes. Doing it that way (rather than adding SINKS to every tier) keeps the sinks
    /// arithmetic in ONE kernel; deepseek4's wiring slice can widen it if the perf demands.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_sinks(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        sinks: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale: f32,
        cap: usize,
    ) {
        let kern = self.be.kernel(
            "attention_kv_sinks",
            crate::gemm::attention_kv_sinks_spv(),
            5,
            36,
        );
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        push[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch(
            kern,
            // `o` LAST: the trailing `n_out` bindings are the hazard-tracked writes, so the sinks
            // weight must bind before it (same rule as `rope_ff`'s divisors and `mla_bias`).
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(sinks),
                Self::vkb(o),
            ],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// `-DKV_BDA` twin of [`Self::attention_kv`] (#74 slice 1): reads the K/V cache by 64-bit device
    /// address (`k_addr`/`v_addr`, kv_addr.glsl) instead of the bound SSBOs at slots 1/2. Bit-
    /// identical to the bound build (proven by kv_addr_parity.rs); production forks here from
    /// adapter.rs when both KV buffers report a `device_addr` (attention_kv stays dual-arm: the
    /// quantized-KV-prefill scalar read off the Activations dequant scratch has none → bound). Push
    /// grows by k_lo/k_hi/v_lo/v_hi (uvec2 splits, avoiding 8-byte push alignment).
    ///
    /// `kc`/`vc` are still bound at slots 1/2 as INERT descriptors the shader never reads — solely so
    /// `Recorder::sync`'s hazard tracker still sees the KV read and orders it after the store kernel's
    /// write (a BDA pointer read is invisible to that descriptor-based tracker — the dropped-barrier
    /// class fixed for the MoE pager in 843a759). `k_addr`/`v_addr` need not equal
    /// `kc`/`vc`.device_addr(): the parity test parks the SAME cache behind a non-zero prefix and
    /// passes base+prefix, while binding the prefixed buffer for the hazard edge.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_at(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        k_addr: u64,
        v_addr: u64,
        o: &dyn Buffer,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale: f32,
        k_q8: bool,
        v_q8: bool,
        cap: usize,
    ) {
        let (name, spv) = match (k_q8, v_q8) {
            (false, false) => ("attention_kv_bda", crate::gemm::attention_kv_bda_spv()),
            (true, false) => (
                "attention_kv_kq8_bda",
                crate::gemm::attention_kv_kq8_bda_spv(),
            ),
            (false, true) => (
                "attention_kv_vq8_bda",
                crate::gemm::attention_kv_vq8_bda_spv(),
            ),
            (true, true) => (
                "attention_kv_q8_bda",
                crate::gemm::attention_kv_q8_bda_spv(),
            ),
        };
        // n_buf stays 4 (q, K, V, o): slots 1/2 keep the KV binds the -DKV_BDA shader ignores — see
        // the doc above (hazard-tracking only), mirroring the weight side's linear_f16 filler bind.
        let kern = self.be.kernel(name, spv, 4, 52);
        let mut push = [0u8; 52];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        push[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        push[36..40].copy_from_slice(&(k_addr as u32).to_ne_bytes());
        push[40..44].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
        push[44..48].copy_from_slice(&(v_addr as u32).to_ne_bytes());
        push[48..52].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(o)],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// Non-FA prefill attention: clean coopmat QK → row softmax → coopmat PV (ollama's approach).
    /// `q`=[mpad,nh,hd] f16, `kc`/`vc`=[kv_len,nkv,hd] f16, `attn`=[mpad,nh*hd] f32 out, `s`=
    /// [nh,mpad,kv_pad] f16 scratch (mpad=ceil(n/64)*64, kv_pad=ceil(kv_len/64)*64). `pos_offset` is
    /// the absolute position of query row 0 (for causal masking).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_nonfa(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        s: &dyn Buffer,
        pv_part: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale: f32,
    ) {
        let mpad = (n.div_ceil(64) * 64) as u32;
        // kv padded to 256 (the 8-warp attn_qk's BN); extra cols are masked in softmax. Still %64 so
        // the 4-warp fallback + softmax + attn_pv are unaffected.
        let kv_pad = (kv_len.div_ceil(256) * 256) as u32;
        let hdu = hd as u32;
        // scale override: >0 uses the caller's value (gemma4 = 1.0, QK-norm controls magnitude);
        // 0 keeps the default 1/√hd (qwen3, gemma3).
        let scale = if scale > 0.0 {
            scale
        } else {
            1.0f32 / (hd as f32).sqrt()
        };

        // stage 1: S = scale·Q·Kᵀ. 8-warp/256-thread warptile (BN=256, matches ollama's mul_mm)
        // unless INFR_NO_QK_WARP forces the 4-warp/2×2 attn_qk.
        let qk_warp = self.vk().qk_warp;
        let (qk_name, qk_spv, qk_bn) = if qk_warp {
            ("attn_qk_warp", crate::gemm::attn_qk_warp_spv(), 256u32)
        } else {
            ("attn_qk", crate::gemm::attn_qk_spv(), 64u32)
        };
        let kqk = self.be.kernel_sg(qk_name, qk_spv, 3, 24, 32);
        let mut p = [0u8; 24];
        p[0..4].copy_from_slice(&mpad.to_ne_bytes());
        p[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        p[8..12].copy_from_slice(&hdu.to_ne_bytes());
        p[12..16].copy_from_slice(&(nh as u32).to_ne_bytes());
        p[16..20].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p[20..24].copy_from_slice(&scale.to_ne_bytes());
        let qk_tiles = (mpad / 64) * (kv_pad / qk_bn);
        self.dispatch3(
            kqk,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(s)],
            1,
            &p,
            qk_tiles,
            1,
            nh as u32,
        );

        // stage 2: row softmax (causal + optional sliding window), in place S → P
        let ksm = self
            .be
            .kernel("attn_softmax", crate::gemm::attn_softmax_spv(), 1, 20);
        let mut ps = [0u8; 20];
        ps[0..4].copy_from_slice(&mpad.to_ne_bytes());
        ps[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        ps[8..12].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        ps[12..16].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        ps[16..20].copy_from_slice(&(window as u32).to_ne_bytes());
        self.dispatch3(ksm, &[Self::vkb(s)], 1, &ps, mpad, 1, nh as u32);

        // stage 3: O = P·V  (one coopmat GEMM per head). Split-K when under-occupied: at high ctx
        // mpad is small so the base workgroup count ((mpad/64)*(hd/64)*nh) is far below the GPU's
        // capacity while each grinds a huge kv reduction → split the kv dim across gl_WorkGroupID.y
        // into partials, then sum them.
        let pv_base_wg = (mpad / 64) * (hdu / 64) * nh as u32;
        // `pv_splits` is `Option<usize>` in the config; the dispatch count is a u32, so an
        // out-of-range override falls back to the adaptive policy exactly as `.parse::<u32>().ok()`
        // did at the old env read.
        let n_splits = match self.vk().pv_splits.and_then(|v| u32::try_from(v).ok()) {
            Some(v) => v,
            None => {
                if pv_base_wg >= 1024 || kv_pad < 4096 {
                    1u32
                } else {
                    (2048 / pv_base_wg.max(1))
                        .clamp(2, 8)
                        .min(kv_pad / 2048)
                        .max(1)
                }
            }
        };
        let ksplit = kv_pad.div_ceil(n_splits).div_ceil(32) * 32;
        // 8-warp/256-thread PV warptile (BN=128=hd, matches ollama's mul_mm) when hd%128; else the
        // 4-warp/2×2 attn_pv (also handles hd<128, e.g. hd=64). INFR_NO_PV_WARP forces the 4-warp.
        let pv_warp = hdu.is_multiple_of(128) && self.vk().pv_warp;
        let (pv_name, pv_spv, pv_bn) = if pv_warp {
            ("attn_pv_warp", crate::gemm::attn_pv_warp_spv(), 128u32)
        } else {
            ("attn_pv", crate::gemm::attn_pv_spv(), 64u32)
        };
        // attn_pv{,_warp}.comp derive `tilesN = hd/BN` as a comment-only precondition: hd<BN
        // div-by-zeros, a non-multiple hd truncates and never covers the top dims. All shipped head
        // dims are BN-multiples (64/128/256), so this never fires — it pins the invariant at dispatch.
        debug_assert!(
            hdu.is_multiple_of(pv_bn),
            "attn_pv: hd ({hdu}) must be a multiple of BN ({pv_bn}); tilesN=hd/BN would truncate dims / div-by-zero"
        );
        let kpv = self.be.kernel_sg(pv_name, pv_spv, 3, 28, 32);
        let mut pp = [0u8; 28];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&kv_pad.to_ne_bytes());
        pp[8..12].copy_from_slice(&hdu.to_ne_bytes());
        pp[12..16].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&n_splits.to_ne_bytes());
        pp[24..28].copy_from_slice(&ksplit.to_ne_bytes());
        let pv_tiles = (mpad / 64) * (hdu / pv_bn);
        let pv_out = if n_splits == 1 { attn } else { pv_part };
        self.dispatch3(
            kpv,
            &[Self::vkb(s), Self::vkb(vc), Self::vkb(pv_out)],
            1,
            &pp,
            pv_tiles,
            n_splits,
            nh as u32,
        );
        if n_splits > 1 {
            // sum the per-split partials → attn
            let total = mpad * nh as u32 * hdu;
            let kr = self
                .be
                .kernel("attn_pv_reduce", crate::gemm::attn_pv_reduce_spv(), 2, 8);
            let mut pr = [0u8; 8];
            pr[0..4].copy_from_slice(&total.to_ne_bytes());
            pr[4..8].copy_from_slice(&n_splits.to_ne_bytes());
            self.dispatch(
                kr,
                &[Self::vkb(pv_part), Self::vkb(attn)],
                1,
                &pr,
                total.div_ceil(256),
            );
        }
    }

    /// Flash-attention prefill: fused QK→softmax→PV, no materialized S buffer. `q`=[mpad,nh,hd] f16,
    /// `kc`/`vc`=[kv_len,nkv,hd] f16, `attn`=[mpad,nh*hd] f32 out. `pos_offset` = abs position of row
    /// 0 (causal). Split-K over kv for occupancy at high ctx (few q tiles): each (q-tile, head, split)
    /// emits an online-softmax partial into `po`/`pm`/`pl`, merged by attn_flash_combine. Scratch
    /// (caller, sized for ≤8 splits): `po` = 8·mpad·nh·hd f32, `pm`/`pl` = 8·mpad·nh f32. n_splits==1
    /// → single fused kernel writing `attn` directly (no scratch touched).
    ///
    /// `stage` (Levers 2 & 5): `Off` = the default (direct coopMatLoad from bound descriptor, f16
    /// `kc`/`vc`). `Dequant(dt)` (Lever 2) decodes the QUANTIZED GGUF block cache (dt) to f16 in the
    /// QK/PV stage — the caller passes the cache directly and skips the dequant_kv_f16 prepass; on the
    /// warp-eligible hd==128 geometry it uses the register-tiled `attn_flash_warp_deq_*` build (keeps
    /// the fast kernel), else the `attn_flash_partial` staged build. `Stage` (Lever 5, f16) has builds
    /// only for `attn_flash_partial`, so it alone forces the split-K partial+combine path (no fused
    /// `attn_flash`, no `attn_flash_warp`). Both staged modes still use partial+combine at n_splits==1.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_flash(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        po: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        stage: FlashStage,
        // #74 slice 4 resurrection: Some((k_addr, v_addr)) → read the KV cache by 64-bit device address
        // (kv_addr.glsl coopMatLoad base) via the `_bda` builds, opt-in (INFR_KV_COOPMAT_BDA) from the
        // adapter. None → bound descriptors, EXACTLY as before. Only honored on the split-K Off path
        // (partial/warp) — the fused single-kernel `attn_flash` and the Stage/Dequant experimental
        // builds have no BDA arm, so they stay bound even when this is Some. kc/vc are ALWAYS bound
        // (inert under BDA) so Recorder::sync keeps the KV store→read hazard edge.
        kv_addr: Option<(u64, u64)>,
    ) {
        let mpad = (n.div_ceil(64) * 64) as u32;
        let base_wg = (mpad / 64) * nh as u32;
        let n_splits = match self.vk().flash_splits.and_then(|v| u32::try_from(v).ok()) {
            Some(v) => v,
            None => {
                if base_wg >= 1024 || kv_len < 4096 {
                    1u32
                } else {
                    (2048 / base_wg.max(1))
                        .clamp(2, 8)
                        .min((kv_len / 2048).max(1) as u32)
                }
            }
        };
        // hd=128 → register-blocked partial (always via partial+combine). Other hd → the
        // 4-subgroup path: single fused kernel for n_splits==1, else the scalar partial.
        //
        // The warp partial's shared scratch is bm*908 B (Ss+Ps+Os+mrow/lrow/corr, BN=64/HD=128).
        // Pick the largest tile the device's maxComputeSharedMemorySize allows: bm=64 → 58112 B (needs
        // 64 KB, e.g. RADV); bm=32 → 29056 B (fits NVIDIA 48 KB / MoltenVK 32 KB). The transformer
        // skips flash entirely when even bm=32 won't fit, so one of these always fits here.
        let shared_limit = self.be.max_shared_memory_bytes();
        let bm64_shared = 64 * crate::FLASH_SHARED_PER_ROW; // 58112 B
                                                            // `INFR_FLASH_BM=32` (`kernels.vulkan.flash_bm32` — compared to the LITERAL "32", §10.4)
                                                            // forces the small (29056 B) tile even on a 64 KB device, so the bm=32 shaders get
                                                            // numeric-parity coverage on any GPU (they otherwise only run on sub-64 KB ones).
        let force_bm32 = self.vk().flash_bm32;
        // Partial tiles (rows < 64, the small-m deep-kv tier) take BM=32: half the padded-row
        // waste and half the shared scratch (2x resident workgroups). Measured @d16384 on a
        // 7900 XTX: pp24 1122 -> 1312 t/s, pp32 1510 -> 1734 (llama.cpp: 1313/1933). Full
        // tiles (rows >= 64) keep the device-based pick — BM=64 wins there when shared fits.
        let bm: u32 = if !force_bm32 && shared_limit >= bm64_shared && n >= 64 {
            64
        } else {
            32
        };
        // Each workgroup covers `bm` query rows → mpad/bm×nh groups (mpad is 64-aligned → /32 exact).
        let tile_wg = (mpad / bm) * nh as u32;
        // INFR_NO_FLASH_WARP routes to the non-warp partial. Both warp and non-warp paths have a
        // bm=32 build (29056 B) that fits sub-64 KB devices, so the knob is honored everywhere —
        // no longer forced back to warp on NVIDIA / MoltenVK.
        // Dequant-in-flash (Lever 2) now has a WARP build (attn_flash_warp_deq_*), so `Dequant` also
        // keeps the fast register-tiled warp path here — it no longer forfeits warp to the ~2x-slower
        // `attn_flash_partial` (kv-decode-perf-levers). `Stage` (Lever 5, f16) still has builds only
        // for the split-K `attn_flash_partial`, so it alone forces that path (no fused `attn_flash`).
        let warp = matches!(stage, FlashStage::Off | FlashStage::Dequant(_))
            && hd == 128
            && self.vk().flash_warp;
        if n_splits == 1 && !warp && matches!(stage, FlashStage::Off) {
            let (fname, fspv): (&'static str, &[u32]) = if bm == 32 {
                ("attn_flash_bm32", crate::gemm::attn_flash_bm32_spv())
            } else {
                ("attn_flash", crate::gemm::attn_flash_spv())
            };
            let k = self.be.kernel_sg(fname, fspv, 4, 24, 32);
            let mut p = [0u8; 24];
            p[0..4].copy_from_slice(&mpad.to_ne_bytes());
            p[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
            p[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
            p[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
            p[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
            p[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
            self.dispatch(
                k,
                &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(attn)],
                1,
                &p,
                tile_wg,
            );
            return;
        }
        // split-K partials
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        // BDA is only wired for the Off (f16) split-K path — the Stage/Dequant experimental builds have
        // no `_bda` twin (a separate, mutually-exclusive flag family). So opt into BDA only when Off.
        let bda = kv_addr.is_some() && matches!(stage, FlashStage::Off);
        // hd=128 → register-blocked warp partial; else the 4-subgroup partial. Each picks its
        // bm-sized shared build (warp/partial share the bm*908 B footprint) and covers tile_wg groups.
        let (pname, pspv): (&'static str, &[u32]) = match stage {
            FlashStage::Stage => crate::gemm::attn_flash_partial_stage_spv(bm == 32),
            // Dequant on the warp-eligible geometry keeps the register-tiled warp path; otherwise
            // (hd!=128 impossible here, or INFR_NO_FLASH_WARP) falls to the split-K partial build.
            FlashStage::Dequant(dt) if warp => crate::gemm::attn_flash_warp_deq_spv(dt, bm == 32)
                .expect("attention_prefill_flash: no warp dequant-in-flash build for KV dtype"),
            FlashStage::Dequant(dt) => crate::gemm::attn_flash_partial_deq_spv(dt, bm == 32)
                .expect("attention_prefill_flash: no dequant-in-flash build for KV dtype"),
            FlashStage::Off => match (warp, bm, bda) {
                (true, 32, true) => (
                    "attn_flash_warp_bm32_bda",
                    crate::gemm::attn_flash_warp_bm32_bda_spv(),
                ),
                (true, 32, false) => (
                    "attn_flash_warp_bm32",
                    crate::gemm::attn_flash_warp_bm32_spv(),
                ),
                (true, _, true) => (
                    "attn_flash_warp_bda",
                    crate::gemm::attn_flash_warp_bda_spv(),
                ),
                (true, _, false) => ("attn_flash_warp", crate::gemm::attn_flash_warp_spv()),
                (false, 32, true) => (
                    "attn_flash_partial_bm32_bda",
                    crate::gemm::attn_flash_partial_bm32_bda_spv(),
                ),
                (false, 32, false) => (
                    "attn_flash_partial_bm32",
                    crate::gemm::attn_flash_partial_bm32_spv(),
                ),
                (false, _, true) => (
                    "attn_flash_partial_bda",
                    crate::gemm::attn_flash_partial_bda_spv(),
                ),
                (false, _, false) => ("attn_flash_partial", crate::gemm::attn_flash_partial_spv()),
            },
        };
        // BDA grows the push by four u32 (k_lo/k_hi/v_lo/v_hi) after the 8 base fields; bound keeps 32.
        let plen: usize = if bda { 48 } else { 32 };
        let kp = self.be.kernel_sg(pname, pspv, 6, plen as u32, 32);
        let mut pp = [0u8; 48];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        pp[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        pp[24..28].copy_from_slice(&n_splits.to_ne_bytes());
        pp[28..32].copy_from_slice(&ksplit.to_ne_bytes());
        if let Some((k_addr, v_addr)) = kv_addr.filter(|_| bda) {
            pp[32..36].copy_from_slice(&(k_addr as u32).to_ne_bytes());
            pp[36..40].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
            pp[40..44].copy_from_slice(&(v_addr as u32).to_ne_bytes());
            pp[44..48].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        }
        self.dispatch3(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(po),
                Self::vkb(pm),
                Self::vkb(pl),
            ],
            3,
            &pp[..plen],
            tile_wg,
            n_splits,
            1,
        );
        // combine → attn
        let kc2 = self.be.kernel_sg(
            "attn_flash_combine",
            crate::gemm::attn_flash_combine_spv(),
            4,
            16,
            32,
        );
        let mut pc2 = [0u8; 16];
        pc2[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pc2[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        pc2[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        pc2[12..16].copy_from_slice(&n_splits.to_ne_bytes());
        self.dispatch(
            kc2,
            &[Self::vkb(po), Self::vkb(pm), Self::vkb(pl), Self::vkb(attn)],
            1,
            &pc2,
            mpad * nh as u32,
        );
    }

    /// hd=256 FlashAttention prefill for 32 KB-shared devices. A fixed BM=16 tile and four
    /// subgroup column workers plus safe final-tile staging fit in 30,912 B shared memory.
    /// This is deliberately separate from [`Self::attention_prefill_flash`]: its hd=128 shader
    /// selection, split heuristic and combine SPIR-V remain untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_flash_hd256(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        po: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        pos_offset: usize,
    ) {
        const HD: u32 = 256;
        const BM: u32 = 16;
        let mpad = (n.div_ceil(64) * 64) as u32;
        let base_wg = (mpad / BM) * nh as u32;
        let n_splits = match self.vk().flash_splits.and_then(|v| u32::try_from(v).ok()) {
            Some(v) => v,
            None => {
                if base_wg >= 1024 || kv_len < 4096 {
                    1u32
                } else {
                    (2048 / base_wg.max(1))
                        .clamp(2, 8)
                        .min((kv_len / 2048).max(1) as u32)
                }
            }
        };
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        let kp = self.be.kernel_sg(
            "attn_flash_warp_hd256_bm16",
            crate::gemm::attn_flash_warp_hd256_bm16_spv(),
            6,
            32,
            32,
        );
        let mut pp = [0u8; 32];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        pp[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&HD.to_ne_bytes());
        pp[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        pp[24..28].copy_from_slice(&n_splits.to_ne_bytes());
        pp[28..32].copy_from_slice(&ksplit.to_ne_bytes());
        self.dispatch3(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(po),
                Self::vkb(pm),
                Self::vkb(pl),
            ],
            3,
            &pp,
            base_wg,
            n_splits,
            1,
        );

        let kc2 = self.be.kernel_sg(
            "attn_flash_combine_hd256",
            crate::gemm::attn_flash_combine_hd256_spv(),
            4,
            16,
            32,
        );
        let mut pc2 = [0u8; 16];
        pc2[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pc2[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        pc2[8..12].copy_from_slice(&HD.to_ne_bytes());
        pc2[12..16].copy_from_slice(&n_splits.to_ne_bytes());
        self.dispatch(
            kc2,
            &[Self::vkb(po), Self::vkb(pm), Self::vkb(pl), Self::vkb(attn)],
            1,
            &pc2,
            mpad * nh as u32,
        );
    }

    /// Non-coopmat shared-memory fma flash-attention prefill (`attn_nc_fa.comp`, adapter.rs
    /// `nc_fa_ok` — the Intel Arc tier): fused tile QK^T → online softmax → PV, one workgroup per
    /// (BM-row query tile, head), no subgroup ops, no pinned subgroup size, no scratch buffers.
    /// Handles causal + sliding-window masks, ring KV caches (position → row via the `cap` push
    /// constant, attn_partial's mapping), caller scale (0 → 1/√hd) and GQA. f16 KV only (the
    /// adapter dequants quantized KV to the f16 ring-layout scratch before rows>1 attention).
    /// `q`/`attn` rows need NOT be tile-padded: pad rows stage zeros / are never written.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_nc_fa(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        rows: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos: usize,
        window: usize,
        scale: f32,
        cap: usize,
    ) {
        let (name, spv, bm) =
            crate::gemm::attn_nc_fa_spv(hd).expect("attention_prefill_nc_fa: hd gated <= 512");
        let k = self.be.kernel(name, spv, 4, 36);
        let mut p = [0u8; 36];
        p[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        p[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        p[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        p[20..24].copy_from_slice(&(pos as u32).to_ne_bytes());
        p[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        p[28..32].copy_from_slice(&scale.to_ne_bytes());
        p[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        let groups = (rows.div_ceil(bm) * nh) as u32;
        self.dispatch(
            k,
            &[Self::vkb(q), Self::vkb(kc), Self::vkb(vc), Self::vkb(attn)],
            1,
            &p,
            groups,
        );
    }

    /// FlashAttention-2 register-O prefill (Br=128, per-thread register accumulator → no [Br][HD]
    /// shared O; 2× the query tile of the shared-Os flash → fewer q-tiles). hd MUST be 128 and the
    /// caller MUST allocate q/attn/po to mpad128 = ceil(n/128)*128 rows. Split-K → partials → combine.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_flash_reg(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        attn: &dyn Buffer,
        po: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        n: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        // #74 slice 4 resurrection: Some((k_addr, v_addr)) → read KV by device address via the `_bda`
        // builds (kc/vc stay bound inert for the store→read hazard edge); None → bound, as before.
        kv_addr: Option<(u64, u64)>,
    ) {
        // mpad is 128-aligned → divisible by both BR tiles.
        let mpad = (n.div_ceil(128) * 128) as u32;
        // Register-O shared = BR*FLASH_REG_SHARED_PER_ROW: BR=128 → 58880 B (needs 64 KB); BR=64 →
        // 29440 B (NVIDIA 48 KB / MoltenVK 32 KB). Largest that fits; transformer skips reg if neither.
        let br: u32 = if self.be.max_shared_memory_bytes() >= 128 * crate::FLASH_REG_SHARED_PER_ROW
        {
            128
        } else {
            64
        };
        let base_wg = (mpad / br) * nh as u32;
        let n_splits = match self.vk().flash_splits.and_then(|v| u32::try_from(v).ok()) {
            Some(v) => v,
            None => {
                if base_wg >= 1024 || kv_len < 4096 {
                    1u32
                } else {
                    (2048 / base_wg.max(1))
                        .clamp(2, 8)
                        .min((kv_len / 2048).max(1) as u32)
                }
            }
        };
        let ksplit = (kv_len as u32).div_ceil(n_splits).div_ceil(64) * 64;
        let bda = kv_addr.is_some();
        let (rname, rspv): (&'static str, &[u32]) = match (br == 64, bda) {
            (true, true) => (
                "attn_flash_reg_br64_bda",
                crate::gemm::attn_flash_reg_br64_bda_spv(),
            ),
            (true, false) => (
                "attn_flash_reg_br64",
                crate::gemm::attn_flash_reg_br64_spv(),
            ),
            (false, true) => ("attn_flash_reg_bda", crate::gemm::attn_flash_reg_bda_spv()),
            (false, false) => ("attn_flash_reg", crate::gemm::attn_flash_reg_spv()),
        };
        let plen: usize = if bda { 48 } else { 32 };
        let kp = self.be.kernel_sg(rname, rspv, 6, plen as u32, 32);
        let mut pp = [0u8; 48];
        pp[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pp[4..8].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        pp[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        pp[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        pp[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        pp[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        pp[24..28].copy_from_slice(&n_splits.to_ne_bytes());
        pp[28..32].copy_from_slice(&ksplit.to_ne_bytes());
        if let Some((k_addr, v_addr)) = kv_addr {
            pp[32..36].copy_from_slice(&(k_addr as u32).to_ne_bytes());
            pp[36..40].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
            pp[40..44].copy_from_slice(&(v_addr as u32).to_ne_bytes());
            pp[44..48].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        }
        self.dispatch3(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(po),
                Self::vkb(pm),
                Self::vkb(pl),
            ],
            3,
            &pp[..plen],
            base_wg,
            n_splits,
            1,
        );
        let kc2 = self.be.kernel_sg(
            "attn_flash_combine",
            crate::gemm::attn_flash_combine_spv(),
            4,
            16,
            32,
        );
        let mut pc2 = [0u8; 16];
        pc2[0..4].copy_from_slice(&mpad.to_ne_bytes());
        pc2[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        pc2[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        pc2[12..16].copy_from_slice(&n_splits.to_ne_bytes());
        self.dispatch(
            kc2,
            &[Self::vkb(po), Self::vkb(pm), Self::vkb(pl), Self::vkb(attn)],
            1,
            &pc2,
            mpad * nh as u32,
        );
    }

    /// Cast-copy f32 `src[0..n]` → f16 `dst[off..off+n]` (write f32 activations into the f16 cache).
    pub fn store_f16(&self, src: &dyn Buffer, dst: &dyn Buffer, n: usize, off: usize) {
        self.store_f16_off(src, dst, n, off, 0)
    }

    /// [`Self::store_f16`] with a SOURCE element offset — the second half of a KV write that
    /// crosses an SWA ring cache's wrap boundary reads the source's tail (`src[src_off..]`).
    pub fn store_f16_off(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        src_off: usize,
    ) {
        let k = self
            .be
            .kernel("store_f16", crate::gemm::store_f16_spv(), 2, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(src_off as u32).to_ne_bytes());
        // [12..16] cap: dyn-only (ring row modulo), zero here.
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    // ── #74 slice 3: KV-cache u64/BDA store + dequant-read `_at` twins ──────────────────────────
    // Each `_at` forks to the `-DKV_BDA` build (writes the DEST cache / reads the SOURCE cache by the
    // 64-bit `addr`) but keeps the KV buffer BOUND at its original slot as an INERT descriptor: a BDA
    // access is invisible to `Recorder::sync`'s descriptor hazard tracker, so the bound-at-write-slot
    // dst (store) / bound-at-read-slot src (dequant) is exactly what still orders the store→(later)
    // read barrier — the write-side analog of the slice-1 inert read bind. `n_buf`/`n_out` are
    // UNCHANGED from the bound sibling (the KV binding stays in the reads|writes split); only the push
    // grows by the addr uvec2 (lo/hi). Production forks here from adapter.rs when the KV buffer exposes
    // a device address; bit-identical to the bound dispatch (store_f16 stays dual-arm — its f32-cast
    // dequant read writes a non-KV Activations scratch that has none → bound).

    /// `-DKV_BDA` twin of [`Self::store_f16_off`]: writes the f16 cache at `dst_addr`. `dst` stays
    /// bound at slot 1 as the inert write descriptor keeping the store→read barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn store_f16_off_at(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        dst_addr: u64,
        n: usize,
        off: usize,
        src_off: usize,
    ) {
        let k = self
            .be
            .kernel("store_f16_bda", crate::gemm::store_f16_bda_spv(), 2, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(src_off as u32).to_ne_bytes());
        // [12..16] cap: dyn-only, zero here.
        push[16..20].copy_from_slice(&(dst_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((dst_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// `-DKV_BDA` twin of [`Self::store_f16_dyn`]: writes the f16 cache at `dst_addr` (row from
    /// `params`). `dst` stays bound at slot 2 (inert) for the store→read barrier.
    pub fn store_f16_dyn_at(
        &self,
        src: &dyn Buffer,
        params: &dyn Buffer,
        dst: &dyn Buffer,
        dst_addr: u64,
        n: usize,
        cap_rows: usize,
    ) {
        let k = self.be.kernel(
            "store_f16_dyn_bda",
            crate::gemm::store_f16_dyn_bda_spv(),
            3,
            24,
        );
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(cap_rows as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(dst_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((dst_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(params), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// `-DKV_BDA` twin of [`Self::rope_f16`]: writes the f16 K cache at `y_addr`. `y` stays bound
    /// at slot 1 (inert) for the store→read barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f16_at(
        &self,
        x: &dyn Buffer,
        y: &dyn Buffer,
        y_addr: u64,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        pos_offset: usize,
        out_base: usize,
    ) {
        let k = self
            .be
            .kernel("rope_f16_bda", crate::gemm::rope_f16_bda_spv(), 2, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(pos_offset as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&(y_addr as u32).to_ne_bytes());
        push[32..36].copy_from_slice(&((y_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// `-DKV_BDA` twin of [`Self::rope_f16_dyn`]: writes the f16 K cache at `y_addr` (pos from
    /// `params`). `y` stays bound at slot 2 (inert) for the store→read barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f16_dyn_at(
        &self,
        x: &dyn Buffer,
        params: &dyn Buffer,
        y: &dyn Buffer,
        y_addr: u64,
        t: usize,
        n_heads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
    ) {
        let k = self.be.kernel(
            "rope_f16_dyn_bda",
            crate::gemm::rope_f16_dyn_bda_spv(),
            3,
            36,
        );
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_heads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] pos_offset: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&(y_addr as u32).to_ne_bytes());
        push[32..36].copy_from_slice(&((y_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(params), Self::vkb(y)],
            1,
            &push,
            (t * n_heads) as u32,
        );
    }

    /// `-DKV_BDA` twin of [`Self::qk_norm_rope`]: writes the fused K cache at `y_addr`. `y` stays
    /// bound at its (last) write slot (inert) for the store→read barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_at(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        y: &dyn Buffer,
        y_addr: u64,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        freq_factors: Option<&dyn Buffer>,
    ) {
        let mut push = [0u8; 44];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_pos as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        // [32..36] kcap: dyn-only, zero here.
        push[36..40].copy_from_slice(&(y_addr as u32).to_ne_bytes());
        push[40..44].copy_from_slice(&((y_addr >> 32) as u32).to_ne_bytes());
        match freq_factors {
            Some(ff) => {
                let k = self.be.kernel(
                    "qk_norm_rope_ff_bda",
                    crate::gemm::qk_norm_rope_ff_bda_spv(),
                    4,
                    44,
                );
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(ff), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
            None => {
                let k = self.be.kernel(
                    "qk_norm_rope_bda",
                    crate::gemm::qk_norm_rope_bda_spv(),
                    3,
                    44,
                );
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
        }
    }

    /// `-DKV_BDA` twin of [`Self::qk_norm_rope_interleaved`]: writes the fused K cache at `y_addr`.
    /// `y` stays bound at slot 2 (inert) for the store→read barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_interleaved_at(
        &self,
        qg: &dyn Buffer,
        nw: &dyn Buffer,
        y: &dyn Buffer,
        y_addr: u64,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        src_stride: usize,
    ) {
        let k = self.be.kernel(
            "qk_norm_rope_interleaved_bda",
            crate::gemm::qk_norm_rope_interleaved_bda_spv(),
            3,
            44,
        );
        let mut push = [0u8; 44];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_pos as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        push[32..36].copy_from_slice(&(src_stride as u32).to_ne_bytes());
        push[36..40].copy_from_slice(&(y_addr as u32).to_ne_bytes());
        push[40..44].copy_from_slice(&((y_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(qg), Self::vkb(nw), Self::vkb(y)],
            1,
            &push,
            (rows * nheads) as u32,
        );
    }

    /// `-DKV_BDA` twin of [`Self::qk_norm_rope_dyn`]: writes the fused K cache at `y_addr` (pos from
    /// `params`). `y` stays bound at its (last) write slot (inert) for the store→read barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_dyn_at(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        params: &dyn Buffer,
        ff: Option<&dyn Buffer>,
        y: &dyn Buffer,
        y_addr: u64,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
        eps: f32,
        kcap: usize,
    ) {
        let k = match ff {
            Some(_) => self.be.kernel(
                "qk_norm_rope_dyn_ff_bda",
                crate::gemm::qk_norm_rope_dyn_ff_bda_spv(),
                5,
                44,
            ),
            None => self.be.kernel(
                "qk_norm_rope_dyn_bda",
                crate::gemm::qk_norm_rope_dyn_bda_spv(),
                4,
                44,
            ),
        };
        let mut push = [0u8; 44];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] rope_pos: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        push[32..36].copy_from_slice(&(kcap as u32).to_ne_bytes());
        push[36..40].copy_from_slice(&(y_addr as u32).to_ne_bytes());
        push[40..44].copy_from_slice(&((y_addr >> 32) as u32).to_ne_bytes());
        match ff {
            Some(f) => self.dispatch(
                k,
                &[
                    Self::vkb(x),
                    Self::vkb(nw),
                    Self::vkb(params),
                    Self::vkb(f),
                    Self::vkb(y),
                ],
                1,
                &push,
                (rows * nheads) as u32,
            ),
            None => self.dispatch(
                k,
                &[Self::vkb(x), Self::vkb(nw), Self::vkb(params), Self::vkb(y)],
                1,
                &push,
                (rows * nheads) as u32,
            ),
        }
    }

    /// `-DKV_BDA` twin of [`Self::qk_norm_rope_interleaved_dyn`]: writes the fused K cache at
    /// `y_addr` (pos from `params`). `y` stays bound at slot 3 (inert) for the store→read barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_interleaved_dyn_at(
        &self,
        qg: &dyn Buffer,
        nw: &dyn Buffer,
        params: &dyn Buffer,
        y: &dyn Buffer,
        y_addr: u64,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
        eps: f32,
        src_stride: usize,
    ) {
        let k = self.be.kernel(
            "qk_norm_rope_interleaved_dyn_bda",
            crate::gemm::qk_norm_rope_interleaved_dyn_bda_spv(),
            4,
            44,
        );
        let mut push = [0u8; 44];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] rope_pos: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        push[32..36].copy_from_slice(&(src_stride as u32).to_ne_bytes());
        push[36..40].copy_from_slice(&(y_addr as u32).to_ne_bytes());
        push[40..44].copy_from_slice(&((y_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(qg),
                Self::vkb(nw),
                Self::vkb(params),
                Self::vkb(y),
            ],
            1,
            &push,
            (rows * nheads) as u32,
        );
    }

    /// Quantize `src[src_off..src_off+n]` → the Vulkan planar Q8_0 KV cache at element offset
    /// `off`. `cap` = total cache elements (the scales region begins at byte `cap`). `src_f16`
    /// selects the f16-source variant (the un-fused roped K staging); f32 otherwise (the V
    /// projection). `n`/`off`/`src_off` are block-aligned (KV row width is a multiple of 32);
    /// `src_off` > 0 only for the second half of a write crossing an SWA ring cache's wrap.
    #[allow(clippy::too_many_arguments)]
    pub fn store_q8(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        cap: usize,
        src_f16: bool,
        src_off: usize,
    ) {
        // #74 (slice 5): BDA-only. `dst` (the planar-Q8 KV cache) is written by device address; it
        // stays bound at slot 1 as an inert descriptor so Recorder::sync keeps the store→read barrier.
        let dst_addr = dst
            .device_addr()
            .expect("store_q8: KV cache dst must have a device address (BufferUsage::KvCache)");
        let (name, spv) = if src_f16 {
            ("store_q8_f16", crate::gemm::store_q8_f16_spv())
        } else {
            ("store_q8", crate::gemm::store_q8_spv())
        };
        let k = self.be.kernel(name, spv, 2, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(cap as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(src_off as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(dst_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((dst_addr >> 32) as u32).to_ne_bytes());
        // One workgroup (32 lanes) per 32-element block.
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32) / 32,
        );
    }

    /// Record-once decode variant of [`Recorder::store_q8`]: the write offset is `p_pos*n` (one KV
    /// row at the token's position), `p_pos` read from the `params` SSBO so the buffer replays.
    pub fn store_q8_dyn(
        &self,
        src: &dyn Buffer,
        params: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        cap: usize,
        src_f16: bool,
    ) {
        // #74 (slice 5): BDA-only. `dst` (the planar-Q8 KV cache) is written by device address; it
        // stays bound at slot 2 as an inert descriptor so Recorder::sync keeps the store→read barrier.
        let dst_addr = dst
            .device_addr()
            .expect("store_q8_dyn: KV cache dst must have a device address (BufferUsage::KvCache)");
        let (name, spv) = if src_f16 {
            ("store_q8_f16_dyn", crate::gemm::store_q8_f16_dyn_spv())
        } else {
            ("store_q8_dyn", crate::gemm::store_q8_dyn_spv())
        };
        let k = self.be.kernel(name, spv, 3, 24);
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        // [4..8] off: unused (from params). [12..16] src_off: always 0 here (a single decode row
        // never crosses the ring wrap). The kernel derives the ring row capacity from cap/n.
        push[8..12].copy_from_slice(&(cap as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(dst_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((dst_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(params), Self::vkb(dst)],
            1,
            &push,
            (n as u32) / 32,
        );
    }

    /// Expand `n` elements of the planar Q8_0 cache → an f16 buffer (dequant the KV prefix so the
    /// f16 flash / non-FA prefill kernels can read it; the persistent cache stays Q8). `cap` = total
    /// cache elements (the planar scales region base).
    pub fn dequant_q8_f16(&self, src: &dyn Buffer, dst: &dyn Buffer, n: usize, cap: usize) {
        // #74 (slice 5): BDA-only. `src` (the planar-Q8 KV cache) is read by device address; it stays
        // bound at slot 0 as an inert descriptor so Recorder::sync keeps the store→dequant-read barrier.
        let src_addr = src.device_addr().expect(
            "dequant_q8_f16: KV cache src must have a device address (BufferUsage::KvCache)",
        );
        let k = self
            .be
            .kernel("dequant_q8_f16", crate::gemm::dequant_q8_f16_spv(), 2, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cap as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(src_addr as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&((src_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Quantize `src[0..n]` → the standard GGUF-block KV cache of `dt` at element offset `off` (one
    /// thread per 32-block). `src_f16` = f16 source (roped K); f32 otherwise (V).
    pub fn quant_kv(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        src_f16: bool,
    ) {
        // #74 (slice 5): BDA-only. `dst` (the GGUF-block KV cache) is written by device address; it
        // stays bound at slot 1 as an inert descriptor so Recorder::sync keeps the store→read barrier.
        let dst_addr = dst
            .device_addr()
            .expect("quant_kv: KV cache dst must have a device address (BufferUsage::KvCache)");
        let (name, spv) = crate::gemm::quant_kv_kernel(dt, src_f16);
        let k = self.be.kernel(name, spv, 2, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(dst_addr as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&((dst_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((n / 32) as u32).div_ceil(64),
        );
    }

    /// Quantize `src[0..n]` → a TurboQuant KV cache of `dt` at element offset `off` (one thread per
    /// 128-block: L2-norm + WHT + centroid). `src_f16` = f16 K source; f32 V otherwise.
    pub fn quant_turbo(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        src_f16: bool,
    ) {
        let (name, spv) = crate::gemm::quant_turbo_kernel(dt, src_f16);
        let k = self.be.kernel(name, spv, 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((n / 128) as u32).div_ceil(64),
        );
    }

    /// Expand `n` elements of a TurboQuant KV cache of `dt` → an f16 buffer (unpack + inverse WHT).
    /// One thread per 128-block.
    pub fn dequant_turbo_f16(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
    ) {
        let (name, spv) = crate::gemm::dequant_turbo_kernel(dt);
        let k = self.be.kernel(name, spv, 2, 4);
        let mut push = [0u8; 4];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((n / 128) as u32).div_ceil(64),
        );
    }

    /// Cast-store `src[0..n]` → a DENSE KV cache of `dst_dt` (F32/Bf16) at element offset `off`.
    /// `src_f16` = f16 source (roped K); f32 otherwise (V). One thread per element.
    pub fn store_kv_dense(
        &self,
        dst_dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        off: usize,
        src_f16: bool,
    ) {
        // #74 (slice 5): BDA-only. `dst` (the dense KV cache) is written by device address; it stays
        // bound at slot 1 as an inert descriptor so Recorder::sync keeps the store→read barrier.
        let dst_addr = dst.device_addr().expect(
            "store_kv_dense: KV cache dst must have a device address (BufferUsage::KvCache)",
        );
        let (name, spv) = crate::gemm::store_kv_dense_kernel(dst_dt, src_f16);
        let k = self.be.kernel(name, spv, 2, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(off as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(dst_addr as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&((dst_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Expand `n` elements of a standard-GGUF-block KV cache of `dt` → an f16 buffer (reuses the
    /// native_decode `dq()`), so the f16 attention can read a quantized KV prefix. One thread/element.
    pub fn dequant_kv_f16(
        &self,
        dt: infr_core::DType,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
    ) {
        // #74 (slice 5): BDA-only. `src` (the GGUF-block KV cache) is read by device address; it stays
        // bound at slot 0 as an inert descriptor so Recorder::sync keeps the store→dequant-read barrier.
        let src_addr = src.device_addr().expect(
            "dequant_kv_f16: KV cache src must have a device address (BufferUsage::KvCache)",
        );
        let (name, spv) = crate::gemm::dequant_kv_kernel(dt);
        let k = self.be.kernel(name, spv, 2, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(src_addr as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&((src_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Qwen3 QK-norm + RoPE over `x[rows, nheads, hd]` → `y` at rows `out_base..`. `nw` is the
    /// per-head [hd] norm weight. (q: out_base=0; k: out_base=pos so it lands in the cache.)
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        // gemma4 full-attention layers: per-pair RoPE frequency divisors (`Some`); `None` = normal
        // RoPE (qwen3 / gemma3 / gemma4 SWA layers).
        freq_factors: Option<&dyn Buffer>,
    ) {
        // 36 bytes: the trailing `kcap` (ring row capacity) is dyn-only and stays 0 here — the
        // STATIC fused-K cache write pre-mods its baked `out_base` row in the adapter instead.
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_pos as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        match freq_factors {
            Some(ff) => {
                let k =
                    self.be
                        .kernel("qk_norm_rope_ff", crate::gemm::qk_norm_rope_ff_spv(), 4, 36);
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(ff), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
            None => {
                let k = self
                    .be
                    .kernel("qk_norm_rope", crate::gemm::qk_norm_rope_spv(), 3, 36);
                self.dispatch(
                    k,
                    &[Self::vkb(x), Self::vkb(nw), Self::vkb(y)],
                    1,
                    &push,
                    (rows * nheads) as u32,
                );
            }
        }
    }

    /// Fused QkNormRope reading from an INTERLEAVED q+g buffer (stride = nh*2*hd per row,
    /// query for head h at offset h*2*hd). Eliminates per-head CopyStrided dispatches for
    /// qwen35 attention layers (768 dispatches on 27B, 204 on 4B).
    pub fn qk_norm_rope_interleaved(
        &self,
        qg: &dyn Buffer, // interleaved q+g [rows, nh*2*hd]
        nw: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        src_stride: usize, // per-row stride in the qg buffer = nh*2*hd
    ) {
        let k = self.be.kernel(
            "qk_norm_rope_interleaved",
            crate::gemm::qk_norm_rope_interleaved_spv(),
            3,
            36,
        );
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&(rope_pos as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(out_base as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        push[32..36].copy_from_slice(&(src_stride as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(qg), Self::vkb(nw), Self::vkb(y)],
            1,
            &push,
            (rows * nheads) as u32,
        );
    }

    /// Fused QkNormRope reading from interleaved buffer, Dynamic/record-once variant.
    /// Same as qk_norm_rope_interleaved but with USE_PARAMS: pos from SSBO for decode replay.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_interleaved_dyn(
        &self,
        qg: &dyn Buffer,
        nw: &dyn Buffer,
        params: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
        eps: f32,
        src_stride: usize,
    ) {
        let k = self.be.kernel(
            "qk_norm_rope_interleaved_dyn",
            crate::gemm::qk_norm_rope_interleaved_dyn_spv(),
            4,
            36,
        );
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        push[20..24].copy_from_slice(&0u32.to_ne_bytes()); // unused rope_pos
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        push[32..36].copy_from_slice(&(src_stride as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(qg),
                Self::vkb(nw),
                Self::vkb(params),
                Self::vkb(y),
            ],
            1,
            &push,
            (rows * nheads) as u32,
        );
    }

    /// Flash-decoding attention (q_len==1): split the KV range into `n_chunks` chunks of `chunk`,
    /// compute per-chunk softmax partials in parallel (`pm`/`pl`/`pacc`), then combine into `o`.
    /// Parallelizes attention across `nh*n_chunks` workgroups so it stays fast at long context.
    /// Bound-SSBO K/V reads (bindings 1/2). See [`Self::attention_kv_split_at`] for the #74-slice-2
    /// pointer-read twin.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        rows: usize,
        pos: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        canvas_lo: Option<usize>,
        k_q8: bool,
        v_q8: bool,
        cap: usize,
        batched: bool,
    ) {
        self.attention_kv_split_impl(
            q, kc, vc, o, pm, pl, pacc, rows, pos, kv_len, nh, nkv, hd, chunk, n_chunks, scale,
            window, canvas_lo, k_q8, v_q8, cap, batched, None, None,
        );
    }

    /// `-DKV_BDA` twin of [`Self::attention_kv_split`] (#74 slice 2): the flash-decoding partial pass
    /// reads the K/V cache by 64-bit device address (`k_addr`/`v_addr`, kv_addr.glsl) instead of the
    /// bound SSBOs at slots 1/2. Bit-identical to the bound build (proven by kv_addr_parity.rs). `kc`/
    /// `vc` stay BOUND at 1/2 as INERT descriptors the `-DKV_BDA` shader never reads — solely so
    /// `Recorder::sync`'s hazard tracker still orders this read after the KV store's write (a BDA read
    /// is invisible to the descriptor tracker). The combine pass (pass 2) reads no KV and is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_at(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        k_addr: u64,
        v_addr: u64,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        rows: usize,
        pos: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        canvas_lo: Option<usize>,
        k_q8: bool,
        v_q8: bool,
        cap: usize,
        batched: bool,
        // Lever 1: `Some(dt)` routes pass 1 through the mainline-INLINE `attn_partial` build for the
        // low-bit block quant `dt` (K and V share the dtype) — the cache is read/dequanted in-kernel,
        // no dequant→f16 prepass. Requires the BDA base (this `_at` method); `None` keeps f16/Q8.
        kv_ml: Option<infr_core::DType>,
    ) {
        self.attention_kv_split_impl(
            q,
            kc,
            vc,
            o,
            pm,
            pl,
            pacc,
            rows,
            pos,
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            window,
            canvas_lo,
            k_q8,
            v_q8,
            cap,
            batched,
            Some((k_addr, v_addr)),
            kv_ml,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_kv_split_impl(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        // Small-m generalization: `rows` query rows at absolute positions pos..pos+rows-1 (decode
        // = 1 row at kv_len-1). Row r's causal end is pos+r+1; workgroup y picks the row; the
        // partial scratch and `o` are [rows*nh, ...] row-major — the combine treats (rows*nh) as
        // its head count unchanged, and o IS the [rows, nh, hd] dst.
        rows: usize,
        pos: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        // DiffusionGemma canvas denoise (docs/diffusion-gemma.md, `AttnMask::Canvas`): every row
        // attends the SAME fixed bidirectional `[lo, kv_len)`, overriding BOTH the per-row causal
        // end (`qpos1 = pos+row+1` → `kv_len`) and the sliding-window `lo` formula. Rides the
        // shader's otherwise-dead `rows` push-constant slot (see `attn_partial.comp`: the
        // dispatch grid already encodes the row via `gl_WorkGroupID.y`, so the field was never
        // read) as `lo+1` (0 = disabled) — no push-constant layout change, no effect on any
        // existing (non-canvas) caller.
        canvas_lo: Option<usize>,
        // Per-side planar Q8_0 KV read (K and V independent). Planar `cap` = total cache elements
        // (the scales-region base), unused when both are f16.
        k_q8: bool,
        v_q8: bool,
        cap: usize,
        // Rows-batched pass 1 (attn_partial_mrows_c256, K/V once per 4-row group): the caller
        // gates on rows/kv_len/hd (see the adapter's `batched_attn`) and MUST pass chunk <= 256.
        // Mutually exclusive with `canvas_lo` (the adapter's `batched_attn` gate excludes Canvas).
        batched: bool,
        // #74 slice 2: Some((k_addr, v_addr)) routes pass 1's K/V reads through the `-DKV_BDA`
        // attn_partial twins (kv_addr.glsl device-address reads); None keeps the bound-SSBO bindings.
        // kc/vc stay bound at slots 1/2 either way for the store→read hazard edge.
        kv_addr: Option<(u64, u64)>,
        // Lever 1: `Some(dt)` → mainline-INLINE decode build (reads the compact block quant directly,
        // no dequant→f16 prepass). Only ever set with `kv_addr` Some (inline decode needs the BDA
        // base) and with `k_q8`/`v_q8`/`batched` false (mainline is a distinct read path).
        kv_ml: Option<infr_core::DType>,
    ) {
        let bda = kv_addr.is_some();
        // pass 1: per-chunk partials (subgroup-reduction QK; needs requiredSubgroupSize=32). Each
        // f16 variant has a `_bda` twin selected when `kv_addr` is Some (device-address K/V reads).
        let (p1name, p1spv) = if let Some(dt) = kv_ml {
            // Lever 1: mainline low-bit KV decode reads the cache INLINE (per-format FMT build).
            debug_assert!(bda && !k_q8 && !v_q8 && !batched);
            crate::gemm::attn_partial_ml_kernel(dt, crate::gemm::attn_hd_spec_disabled(self.vk()))
        } else {
            match (k_q8, v_q8) {
                (false, false) if crate::gemm::attn_hd_spec_disabled(self.vk()) => {
                    if bda {
                        (
                            "attn_partial_nohd_bda",
                            crate::gemm::attn_partial_nohd_bda_spv(),
                        )
                    } else {
                        ("attn_partial_nohd", crate::gemm::attn_partial_nohd_spv())
                    }
                }
                (false, false) if bda => ("attn_partial_bda", crate::gemm::attn_partial_bda_spv()),
                (false, false) => ("attn_partial", crate::gemm::attn_partial_spv()),
                (true, false) if bda => (
                    "attn_partial_kq8_bda",
                    crate::gemm::attn_partial_kq8_bda_spv(),
                ),
                (true, false) => ("attn_partial_kq8", crate::gemm::attn_partial_kq8_spv()),
                (false, true) if bda => (
                    "attn_partial_vq8_bda",
                    crate::gemm::attn_partial_vq8_bda_spv(),
                ),
                (false, true) => ("attn_partial_vq8", crate::gemm::attn_partial_vq8_spv()),
                (true, true) if bda => (
                    "attn_partial_q8_bda",
                    crate::gemm::attn_partial_q8_bda_spv(),
                ),
                (true, true) => ("attn_partial_q8", crate::gemm::attn_partial_q8_spv()),
            }
        };
        // Rows-BATCHED pass 1 (INFR_MROWS_ATTN=1 [+ INFR_MROWS_CHUNK=256], OFF by default): one
        // workgroup per (head, chunk) streams K/V ONCE for a 4-row group — one subgroupAdd(vec4)
        // per key, scores staged in LDS. The occupancy sweep on a 7900 XTX (pp4@d16384) was
        // monotone in LDS — 22KB (RB=8) 188 t/s, 11KB (RB=4) 325, 7KB (RB=4 + c256) 456, vs the
        // per-row grid's 548 — but even at 7KB the design beats per-row only in a NARROW band:
        // rows >= ~12 AND deep kv (pp16@d16384 832 -> 925, pp32 963 -> 1123; pp4-8 lose or wash,
        // pp16@d4096 loses). On RDNA3 the per-row grid's rows-x extra workgroups fill the DRAM
        // queue better than the batched form's rows-/ bandwidth saving, so the spec-verify band
        // (m <= 8) keeps per-row. If the m >= 12-at-depth band ever matters, route here on
        // (rows, kv_len) — or better, build the LDS-staged K-TILE kernel (per-thread full dots,
        // no cross-lane reductions), which is how llama.cpp wins that cell (1056 t/s).
        // The batched kernel is built `attn_partial_mrows_c256` (-DSC_MAX=256u): its `sc[RB*SC]`
        // score slab holds only SC=256 keys/row, so a chunk > 256 would overrun `sc` (nothing clamps
        // the `sc[i*SC + (j-j0)]` write index in-shader). Pin chunk <= 256 = SC_MAX here.
        debug_assert!(!batched || (chunk <= 256 && hd <= 128 && !k_q8 && !v_q8 && rows >= 2));
        let (p1name, p1spv) = if batched {
            if bda {
                (
                    "attn_partial_mrows_c256_bda",
                    crate::gemm::attn_partial_mrows_c256_bda_spv(),
                )
            } else {
                (
                    "attn_partial_mrows_c256",
                    crate::gemm::attn_partial_mrows_c256_spv(),
                )
            }
        } else {
            (p1name, p1spv)
        };
        // B7 slice 3a — DECODE-ONLY pass 1 (`attn_decode.comp`). Same grid, same chunk policy, same
        // pm/pl/pacc layout, same 60-byte BDA push, same 6 bindings, so this is a pure kernel swap;
        // pass 2 below is untouched. It is `attn_partial_bda`'s hd=128 f16 inner loop with every
        // other arm deleted, which is worth 120 → 96 VGPRs and 4.75 → 2.75 KB LDS (occupancy 12 →
        // 16 waves/SIMD). The reduction is the same plain `subgroupAdd`, so the output is
        // BIT-identical — `tests/attn_decode_parity.rs` asserts that on raw bits.
        //
        // Every condition below is an arm no member of the specialized family has. Two axes ARE
        // covered (slice 3b): the head dim, via the `hd` argument to `attn_decode_kernel` (128 /
        // 256 / 512 have their own builds, anything else falls back), and sliding-window, via the
        // `-DSWA -DRING` builds.
        //
        // The ring-cache condition is the subtle one: `attn_partial` maps position j to cache row
        // `j % (cap/(nkv*hd))` whenever `cap` is nonzero, and the NON-`SWA` builds assume the
        // identity. `window == 0` already implies a full-context (non-ring) cache — only
        // sliding-window layers are allocated as rings (`seam::mod`'s `window + ubatch` sizing) —
        // but the row bound is checked rather than inferred, since a wrong answer here is silent
        // garbage rather than a crash. `window > 0` needs no such check: those builds carry the
        // modulo, which is the identity when the cache happens to be full-context.
        let ring_safe = window > 0 || cap == 0 || pos < cap / (nkv * hd);
        let fast = if !batched
            && !crate::gemm::attn_decode_disabled(self.vk())
            && bda
            && kv_ml.is_none()
            && !k_q8
            && !v_q8
            && rows == 1
            && canvas_lo.is_none()
            && chunk <= 512
            && ring_safe
            // INFR_NO_ATTN_HD=1 exists to A/B the hd 256/512 specialization against the general
            // runtime-hd4 loops; the family's hd256/hd512 builds ARE that specialization, so honor
            // the knob by falling back to `attn_partial_nohd_bda` for those dims. hd == 128 is
            // unaffected by NO_HD_SPEC either way.
            && (hd == 128 || !crate::gemm::attn_hd_spec_disabled(self.vk()))
        {
            crate::gemm::attn_decode_kernel(hd, window > 0, false)
        } else {
            None
        };
        let (p1name, p1spv) = fast.unwrap_or((p1name, p1spv));
        // The -DKV_BDA push grows by k_lo/k_hi/v_lo/v_hi (uvec2 splits) → 60 bytes; the bound push is
        // the base 44. n_buf stays 6 (q, kc, vc, pm, pl, pacc): kc/vc are inert-but-bound under BDA.
        let plen: usize = if bda { 60 } else { 44 };
        let k1 = self.be.kernel_sg(p1name, p1spv, 6, plen as u32, 32);
        let mut p1 = [0u8; 60];
        p1[0..4].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        p1[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        p1[36..40].copy_from_slice(&(pos as u32).to_ne_bytes());
        // `rows` is dead in `attn_partial.comp` (this slot carries `canvas_lo+1`, 0 = disabled,
        // instead — see `canvas_lo`'s doc above) but `attn_partial_mrows.comp` (the `batched`
        // kernel) genuinely reads it (`rb = min(RB, pc.rows - row0)`, the short-last-row-group
        // clamp) — canvas is never batched (the adapter's `batched_attn` gate excludes Canvas),
        // so keep sending the real row count there.
        debug_assert!(!batched || canvas_lo.is_none());
        let rows_field = if batched {
            rows as u32
        } else {
            canvas_lo.map(|lo| lo as u32 + 1).unwrap_or(0)
        };
        p1[40..44].copy_from_slice(&rows_field.to_ne_bytes());
        if let Some((k_addr, v_addr)) = kv_addr {
            p1[44..48].copy_from_slice(&(k_addr as u32).to_ne_bytes());
            p1[48..52].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
            p1[52..56].copy_from_slice(&(v_addr as u32).to_ne_bytes());
            p1[56..60].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        }
        // Batched: workgroup y = 4-row group; per-row: y = row.
        let gy = if batched { rows.div_ceil(4) } else { rows };
        self.dispatch3(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1[..plen],
            (nh * n_chunks) as u32,
            gy as u32,
            1,
        );
        // pass 2: combine — split each (row, head)'s hd outputs across `ntile` workgroups for
        // occupancy. The combine is row-agnostic: rows*nh independent [n_chunks] partial sets.
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        // attn_combine.comp splits hd as `hdt = hd/ntile`; a non-divisor ntile drops the top
        // hd-ntile*hdt dims (left uninitialized). ntile is 4 only when hd%4==0, else 1, so hd%ntile==0
        // always holds — this pins that invariant at the call site (never fires for shipped hd).
        debug_assert!(
            hd.is_multiple_of(ntile as usize),
            "attn_combine: hd ({hd}) must be a multiple of ntile ({ntile}); hdt=hd/ntile would drop dims"
        );
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&((rows * nh) as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            (rows * nh) as u32 * ntile,
        );
    }

    /// PROBE (B7 slice 1) — LDS-staged **K-tile** flash-decoding pass 1 + the stock `attn_combine`.
    ///
    /// Measurement-only alternative to [`Self::attention_kv_split_at`]'s pass 1: `attn_ktile.comp`
    /// stages a tile of K in shared memory with coalesced global reads and gives each THREAD a
    /// whole 128-dim dot, so `attn_partial`'s per-key cross-lane `subgroupAdd` — 59% of decode GPU
    /// time at d32768 — disappears. It writes `pm`/`pl`/`pacc` in exactly `attn_partial`'s layout,
    /// so pass 2 is the same `attn_combine` dispatch byte for byte.
    ///
    /// **Nothing in production calls this**, and nothing should until the probe shows a win: the
    /// only caller is `tests/attn_ktile_probe.rs`. `cfg` picks the tile configuration (0 = 64-key
    /// tile, 1 = 64-key unpadded LDS rows [bank-conflict A/B], 2 = 128-key tile, 3 = 64-key tile
    /// staging half the head dim per pass).
    ///
    /// Specialization asserted below (the kernel handles nothing else — `attention_kv_split_at`
    /// remains the general path): `hd == 128`, `rows == 1`, f16 K/V by device address, causal,
    /// `window == 0`, no canvas, no Q8, no ring cache, `chunk <= 512`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_ktile_at(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        k_addr: u64,
        v_addr: u64,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        pos: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        cfg: u32,
    ) {
        assert_eq!(hd, 128, "attn_ktile probe is hd=128 only");
        assert!(chunk <= 512, "attn_ktile probe: sc[] holds 512 keys");
        assert!(nh.is_multiple_of(nkv), "attn_ktile probe: nh % nkv != 0");
        let (p1name, p1spv) = match cfg {
            0 => ("attn_ktile_w64", crate::gemm::attn_ktile_w64_spv()),
            1 => (
                "attn_ktile_w64_nopad",
                crate::gemm::attn_ktile_w64_nopad_spv(),
            ),
            2 => ("attn_ktile_w128", crate::gemm::attn_ktile_w128_spv()),
            3 => (
                "attn_ktile_w64_dw32",
                crate::gemm::attn_ktile_w64_dw32_spv(),
            ),
            _ => panic!("attn_ktile probe: unknown cfg {cfg}"),
        };
        // Same 60-byte `-DKV_BDA` push block attn_partial uses, packed identically (the shader
        // reuses that layout verbatim); `window`/`cap`/`rows` stay 0 — the probe's specialization.
        let k1 = self.be.kernel_sg(p1name, p1spv, 6, 60, 32);
        let mut p1 = [0u8; 60];
        p1[0..4].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        p1[36..40].copy_from_slice(&(pos as u32).to_ne_bytes());
        p1[44..48].copy_from_slice(&(k_addr as u32).to_ne_bytes());
        p1[48..52].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
        p1[52..56].copy_from_slice(&(v_addr as u32).to_ne_bytes());
        p1[56..60].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        self.dispatch3(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1,
            (nh * n_chunks) as u32,
            1,
            1,
        );
        // pass 2: the STOCK combine, unchanged (rows == 1).
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = 4u32;
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    /// PROBE (B7 slice 2) — `attn_partial`'s hd==128 QK loop at a PARAMETERIZED reduction width,
    /// plus the stock `attn_combine`.
    ///
    /// The shipped pass 1 gives all 32 lanes of a wave one key and reduces its 128-dim dot with a
    /// full `subgroupAdd` (reduction width 32, one key in flight per wave); the declined k-tile
    /// probe was width 1. `attn_partial_dsplit.comp` makes the width a build-time constant, so
    /// `width` lanes cooperate on one key (`subgroupClusteredAdd(., width)`) and `32/width` keys
    /// are in flight. Grid, chunk policy and the pm/pl/pacc layout are `attn_partial`'s exactly —
    /// pass 2 is the same `attn_combine` dispatch — so this A/Bs directly against
    /// [`Self::attention_kv_split_at`]. `wg` picks the workgroup size (64 = shipped 2 waves,
    /// 128 = llama.cpp's 4).
    ///
    /// **Nothing in production calls this**, and nothing should until the probe shows a win: the
    /// only caller is `tests/attn_dsplit_probe.rs`.
    ///
    /// Specialization asserted below (identical to the k-tile probe's): `hd == 128`, `rows == 1`,
    /// f16 K/V by device address, causal, `window == 0`, no canvas, no Q8, no ring cache,
    /// `chunk <= 512`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dsplit_at(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        k_addr: u64,
        v_addr: u64,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        pos: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        width: u32,
        wg: u32,
    ) {
        assert_eq!(hd, 128, "attn_dsplit probe is hd=128 only");
        assert!(chunk <= 512, "attn_dsplit probe: sc[] holds 512 keys");
        assert!(nh.is_multiple_of(nkv), "attn_dsplit probe: nh % nkv != 0");
        let (p1name, p1spv) = match (width, wg) {
            (1, 64) => ("attn_dsplit_w1", crate::gemm::attn_dsplit_w1_spv()),
            (2, 64) => ("attn_dsplit_w2", crate::gemm::attn_dsplit_w2_spv()),
            (4, 64) => ("attn_dsplit_w4", crate::gemm::attn_dsplit_w4_spv()),
            (8, 64) => ("attn_dsplit_w8", crate::gemm::attn_dsplit_w8_spv()),
            (16, 64) => ("attn_dsplit_w16", crate::gemm::attn_dsplit_w16_spv()),
            (32, 64) => ("attn_dsplit_w32", crate::gemm::attn_dsplit_w32_spv()),
            (1, 128) => (
                "attn_dsplit_w1_wg128",
                crate::gemm::attn_dsplit_w1_wg128_spv(),
            ),
            (2, 128) => (
                "attn_dsplit_w2_wg128",
                crate::gemm::attn_dsplit_w2_wg128_spv(),
            ),
            (4, 128) => (
                "attn_dsplit_w4_wg128",
                crate::gemm::attn_dsplit_w4_wg128_spv(),
            ),
            (8, 128) => (
                "attn_dsplit_w8_wg128",
                crate::gemm::attn_dsplit_w8_wg128_spv(),
            ),
            (16, 128) => (
                "attn_dsplit_w16_wg128",
                crate::gemm::attn_dsplit_w16_wg128_spv(),
            ),
            (32, 128) => (
                "attn_dsplit_w32_wg128",
                crate::gemm::attn_dsplit_w32_wg128_spv(),
            ),
            _ => panic!("attn_dsplit probe: unknown (width {width}, wg {wg})"),
        };
        // Same 60-byte `-DKV_BDA` push block attn_partial uses, packed identically (the shader
        // reuses that layout verbatim); `window`/`cap`/`rows` stay 0 — the probe's specialization.
        let k1 = self.be.kernel_sg(p1name, p1spv, 6, 60, 32);
        let mut p1 = [0u8; 60];
        p1[0..4].copy_from_slice(&(kv_len as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        p1[36..40].copy_from_slice(&(pos as u32).to_ne_bytes());
        p1[44..48].copy_from_slice(&(k_addr as u32).to_ne_bytes());
        p1[48..52].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
        p1[52..56].copy_from_slice(&(v_addr as u32).to_ne_bytes());
        p1[56..60].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        self.dispatch3(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1,
            (nh * n_chunks) as u32,
            1,
            1,
        );
        // pass 2: the STOCK combine, unchanged (rows == 1).
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = 4u32;
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    // ---- Record-once decode variants (`_dyn`) ----
    // These read the per-token `pos`/`kv_len` from a host-updated `params` SSBO ([pos, kv_len] u32)
    // instead of push constants, so the decode command buffer can be recorded once and replayed every
    // token (only `params` + the embedding change). Used ONLY by the GPU-resident decode path; every
    // other caller keeps the push-constant kernels. `params` is inserted before the output(s) so the
    // recorder's reads|writes split stays output-last.

    /// QK-norm + RoPE, pos from `params`. `out_base_mul` = 0 for Q (write to a temp), 1 for K (write
    /// to the cache at row pos).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_dyn(
        &self,
        x: &dyn Buffer,
        nw: &dyn Buffer,
        params: &dyn Buffer,
        ff: Option<&dyn Buffer>,
        y: &dyn Buffer,
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        out_base_mul: usize,
        eps: f32,
        // Fused-K cache write only: the cache's ROW capacity — an SWA ring cache writes row
        // `pos % kcap` (identity on a full-context cache). 0 (no modulo) for the Q write.
        kcap: usize,
    ) {
        // With freq_factors (gemma4 full-attention layers) `ff` binds at 3 and the output shifts
        // to 4 — same PC layout either way.
        let k = match ff {
            Some(_) => self.be.kernel(
                "qk_norm_rope_dyn_ff",
                crate::gemm::qk_norm_rope_dyn_ff_spv(),
                5,
                36,
            ),
            None => self.be.kernel(
                "qk_norm_rope_dyn",
                crate::gemm::qk_norm_rope_dyn_spv(),
                4,
                36,
            ),
        };
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nheads as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(hd as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(rope_dim as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&theta.to_ne_bytes());
        // [20..24] rope_pos: unused (from params)
        push[24..28].copy_from_slice(&(out_base_mul as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&eps.to_ne_bytes());
        push[32..36].copy_from_slice(&(kcap as u32).to_ne_bytes());
        match ff {
            Some(f) => self.dispatch(
                k,
                &[
                    Self::vkb(x),
                    Self::vkb(nw),
                    Self::vkb(params),
                    Self::vkb(f),
                    Self::vkb(y),
                ],
                1,
                &push,
                (rows * nheads) as u32,
            ),
            None => self.dispatch(
                k,
                &[Self::vkb(x), Self::vkb(nw), Self::vkb(params), Self::vkb(y)],
                1,
                &push,
                (rows * nheads) as u32,
            ),
        }
    }

    /// Cast-copy f32 `src[0..n]` → f16 `dst[row*n..]` (one KV row at position pos from `params`;
    /// `cap_rows` = the cache's row capacity, so an SWA ring cache writes row `pos % cap_rows` —
    /// identity on a full-context cache).
    pub fn store_f16_dyn(
        &self,
        src: &dyn Buffer,
        params: &dyn Buffer,
        dst: &dyn Buffer,
        n: usize,
        cap_rows: usize,
    ) {
        let k = self
            .be
            .kernel("store_f16_dyn", crate::gemm::store_f16_dyn_spv(), 3, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        // [4..8] off: unused (computed as (pos % cap)*n from params). [8..12] src_off: 0.
        push[12..16].copy_from_slice(&(cap_rows as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(src), Self::vkb(params), Self::vkb(dst)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Causal GQA over the KV cache (q_len==1), pos_offset from `params`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_dyn(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        params: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        window: usize,
        // Q8_0 KV cache (K==V==q8): planar dequant-on-read variant. `false` = f16 cache.
        q8: bool,
        // Planar Q8 scales region base = total cache elements (`cap`). Unused when `q8` is false.
        cap: usize,
    ) {
        let (name, spv) = if q8 {
            (
                "attention_kv_dyn_q8",
                crate::gemm::attention_kv_dyn_q8_spv(),
            )
        } else {
            ("attention_kv_dyn", crate::gemm::attention_kv_dyn_spv())
        };
        let kern = self.be.kernel(name, spv, 5, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        // [4..8] kv_len: unused
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        // [20..24] pos_offset: unused (from params)
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        push[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(o),
            ],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// `-DKV_BDA` twin of [`Self::attention_kv_dyn`] (#74 slice 1): record-once decode attention that
    /// reads the K/V cache by 64-bit device address (`k_addr`/`v_addr`). The KV cache buffers are
    /// persistent across the replay, so their addresses are stable — baked into the recorded push
    /// constants, replay-safe. `kc`/`vc` stay bound at slots 1/2 as inert descriptors purely for the
    /// hazard edge (see [`Self::attention_kv_at`]). Push grows by k_lo/k_hi/v_lo/v_hi.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_dyn_at(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        k_addr: u64,
        v_addr: u64,
        params: &dyn Buffer,
        o: &dyn Buffer,
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        window: usize,
        q8: bool,
        cap: usize,
    ) {
        let (name, spv) = if q8 {
            (
                "attention_kv_dyn_q8_bda",
                crate::gemm::attention_kv_dyn_q8_bda_spv(),
            )
        } else {
            (
                "attention_kv_dyn_bda",
                crate::gemm::attention_kv_dyn_bda_spv(),
            )
        };
        let kern = self.be.kernel(name, spv, 5, 52);
        let mut push = [0u8; 52];
        push[0..4].copy_from_slice(&(q_len as u32).to_ne_bytes());
        // [4..8] kv_len: unused (from params)
        push[8..12].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(hd as u32).to_ne_bytes());
        // [20..24] pos_offset: unused (from params)
        push[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&scale.to_ne_bytes());
        push[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        push[36..40].copy_from_slice(&(k_addr as u32).to_ne_bytes());
        push[40..44].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
        push[44..48].copy_from_slice(&(v_addr as u32).to_ne_bytes());
        push[48..52].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(o),
            ],
            1,
            &push,
            (q_len * nh) as u32,
        );
    }

    /// Split-K decode attention, kv_len from `params`. `chunk`/`n_chunks` stay push constants (they
    /// define the dispatch structure; the caller re-records when they change).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dyn(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
    ) {
        self.attention_kv_split_dyn_inner(
            q, kc, vc, o, pm, pl, pacc, params, nh, nkv, hd, chunk, n_chunks, scale, window, None,
        )
    }

    /// `-DKV_BDA` twin of [`Self::attention_kv_split_dyn`] (#74 slice 2): the plain record-once
    /// split-K partial pass reading K/V by 64-bit device address. Bit-identical to the bound build;
    /// `kc`/`vc` stay bound at slots 1/2 as inert descriptors for the store→read hazard edge.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dyn_at(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        k_addr: u64,
        v_addr: u64,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
    ) {
        self.attention_kv_split_dyn_inner(
            q,
            kc,
            vc,
            o,
            pm,
            pl,
            pacc,
            params,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            window,
            Some((k_addr, v_addr)),
        )
    }

    /// Record the split-K replay prologue ONCE per execute: derives the live chunk count from the
    /// params SSBO's kv_len and writes `args = [nh·live, 1, 1, live]` — the indirect dispatch args
    /// and combine loop bound that every subsequent [`Self::attention_kv_split_dynac`] in the same
    /// execute shares (kv_len is the same for every layer of a token).
    pub fn attn_live_prologue(
        &self,
        params: &dyn Buffer,
        args: &dyn Buffer,
        nh: usize,
        chunk: usize,
        window: usize,
    ) {
        let kl = self
            .be
            .kernel("attn_live", crate::gemm::attn_live_spv(), 2, 12);
        let mut p0 = [0u8; 12];
        p0[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p0[4..8].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p0[8..12].copy_from_slice(&(window as u32).to_ne_bytes());
        self.dispatch(kl, &[Self::vkb(params), Self::vkb(args)], 1, &p0, 1);
    }

    /// SELF-CHUNKING variant for record-once REPLAY over a growing kv_len. A one-thread prologue
    /// (`attn_live`) derives the adaptive chunk (~32 chunks/head, 64..512, floored by the baked
    /// minimum `chunk`) from the LIVE kv_len and writes the partial pass's INDIRECT dispatch args
    /// (`args`, ≥16 bytes: [gx,gy,gz,live]) — so one recorded plan launches exactly nh·live
    /// workgroups at every depth (no dead workgroups shallow, no re-record deep). The combine
    /// loops the prologue's live count. `n_chunks` is only the pm/pl/pacc scratch STRIDE/capacity
    /// (cap.div_ceil(chunk), ≤1024).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dynac(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        args: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        q8: bool,
        cap: usize,
    ) {
        self.attention_kv_split_dynac_impl(
            q, kc, vc, o, pm, pl, pacc, params, args, nh, nkv, hd, chunk, n_chunks, scale, window,
            q8, cap, None,
        );
    }

    /// `-DKV_BDA` twin of [`Self::attention_kv_split_dynac`] (#74 slice 2) — the HOTTEST decode read
    /// path (record-once replay). Pass 1 reads K/V by 64-bit device address (`k_addr`/`v_addr`);
    /// `kc`/`vc` stay bound at slots 1/2 as inert descriptors purely for the store→read hazard edge.
    /// Bit-identical to the bound build. The caches persist across replay, so their addresses are
    /// stable in the baked push.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kv_split_dynac_at(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        k_addr: u64,
        v_addr: u64,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        args: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        q8: bool,
        cap: usize,
    ) {
        self.attention_kv_split_dynac_impl(
            q,
            kc,
            vc,
            o,
            pm,
            pl,
            pacc,
            params,
            args,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            window,
            q8,
            cap,
            Some((k_addr, v_addr)),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_kv_split_dynac_impl(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        args: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        // Q8_0 KV cache (K==V==q8): coalesced planar Q8 read variant. `false` = f16 cache.
        q8: bool,
        // Planar Q8 scales region base = total cache elements (`cap`). Unused when `q8` is false.
        cap: usize,
        // #74 slice 2: Some((k_addr, v_addr)) selects the `-DKV_BDA` twins; None keeps bound SSBOs.
        kv_addr: Option<(u64, u64)>,
    ) {
        let bda = kv_addr.is_some();
        // pass 1: self-chunking partials, workgroup count from `args` (the caller records ONE
        // `attn_live_prologue` per (nh, chunk, window) key — kv_len is identical for every layer
        // of a token, so the args buffer is shared across same-key attention ops instead of
        // re-derived per layer). Each f16 variant has a `_bda` device-address twin.
        let (p1name, p1spv) = match (q8, crate::gemm::attn_hd_spec_disabled(self.vk()), bda) {
            (true, _, false) => (
                "attn_partial_dynac_q8",
                crate::gemm::attn_partial_dynac_q8_spv(),
            ),
            (true, _, true) => (
                "attn_partial_dynac_q8_bda",
                crate::gemm::attn_partial_dynac_q8_bda_spv(),
            ),
            (false, true, false) => (
                "attn_partial_dynac_nohd",
                crate::gemm::attn_partial_dynac_nohd_spv(),
            ),
            (false, true, true) => (
                "attn_partial_dynac_nohd_bda",
                crate::gemm::attn_partial_dynac_nohd_bda_spv(),
            ),
            (false, false, false) => ("attn_partial_dynac", crate::gemm::attn_partial_dynac_spv()),
            (false, false, true) => (
                "attn_partial_dynac_bda",
                crate::gemm::attn_partial_dynac_bda_spv(),
            ),
        };
        // B7 slice 3a — DECODE-ONLY pass 1 on the REPLAY path (`attn_decode.comp -DUSE_PARAMS
        // -DSELF_CHUNK`). This is the one that matters end-to-end: real decode replays a recorded
        // tape, so a specialization wired only into the static path above measures as zero gain.
        // Its SELF_CHUNK arm re-derives the identical adaptive chunk (`max(pc.chunk,
        // min(max(span/32,64),512))`) from the live kv_len, so the `attn_live` prologue's indirect
        // grid — nh·live workgroups, one per (query head, chunk) — is unchanged and the prologue
        // needs no edit. `chunk <= 512` keeps the derived chunk inside the kernel's `sc[512]` slab
        // (the policy's own `min(.., 512)` only binds above the baked floor).
        //
        // Slice 3b widens this to `window > 0` (the `-DSWA -DRING` builds: the window mask floor
        // AND the ring row modulo an SWA layer's `window + ubatch`-row cache needs — this path
        // cannot check the row bound the static gate does, because the replayed kv_len is not known
        // at record time, so covering the ring is a precondition for covering SWA at all) and to
        // hd 256/512 (the `-DDHD4=64/128` builds). Anything else still falls back.
        let fast = if !crate::gemm::attn_decode_disabled(self.vk())
            && bda
            && !q8
            && chunk <= 512
            // INFR_NO_ATTN_HD=1 A/Bs the hd 256/512 specialization; the family's hd256/hd512
            // builds ARE it, so leave those dims on `attn_partial_dynac_nohd_bda` when it is set.
            && (hd == 128 || !crate::gemm::attn_hd_spec_disabled(self.vk()))
        {
            crate::gemm::attn_decode_kernel(hd, window > 0, true)
        } else {
            None
        };
        let (p1name, p1spv) = fast.unwrap_or((p1name, p1spv));
        // attn_partial.comp's `layout(push_constant) uniform PC {...}` block is declared
        // UNCONDITIONALLY (11 x 4-byte members = 44 bytes) — `pos`/`rows` (offsets 36..44) are
        // dead code under `-DUSE_PARAMS` (never read in the compiled SPIR-V for this variant, see
        // the shader's `#ifdef USE_PARAMS` guard), so a range of 36 looked sufficient. It isn't:
        // vkCreateComputePipelines validates the pipeline layout's range against the shader's
        // declared Block SIZE, not per-member static use (VUID-VkComputePipelineCreateInfo-layout-
        // 10069 — confirmed firing on the 7900 XTX under validation layers: "push constant buffer
        // Block with range [0, 44] which is outside the VkPushConstantRange of [0, 36]"). The
        // range must cover the full declared block; the trailing 8 bytes stay zero and unread.
        // BDA push grows to 60 (k_lo/k_hi/v_lo/v_hi at 44..60); bound stays 44. n_buf stays 7.
        let plen: usize = if bda { 60 } else { 44 };
        let k1 = self.be.kernel_sg(p1name, p1spv, 7, plen as u32, 32);
        let mut p1 = [0u8; 60];
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        p1[32..36].copy_from_slice(&(cap as u32).to_ne_bytes());
        if let Some((k_addr, v_addr)) = kv_addr {
            p1[44..48].copy_from_slice(&(k_addr as u32).to_ne_bytes());
            p1[48..52].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
            p1[52..56].copy_from_slice(&(v_addr as u32).to_ne_bytes());
            p1[56..60].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        }
        self.dispatch_indirect(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1[..plen],
            Self::vk_handle(args),
            0,
        );

        // pass 2: combine over the live chunks
        let k2 = self.be.kernel(
            "attn_combine_live",
            crate::gemm::attn_combine_live_spv(),
            5,
            16,
        );
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        // attn_combine.comp splits hd as `hdt = hd/ntile`; a non-divisor ntile drops the top
        // hd-ntile*hdt dims (left uninitialized). ntile is 4 only when hd%4==0, else 1, so hd%ntile==0
        // always holds — this pins that invariant at the call site (never fires for shipped hd).
        debug_assert!(
            hd.is_multiple_of(ntile as usize),
            "attn_combine: hd ({hd}) must be a multiple of ntile ({ntile}); hdt=hd/ntile would drop dims"
        );
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
                Self::vkb(args),
                Self::vkb(o),
            ],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_kv_split_dyn_inner(
        &self,
        q: &dyn Buffer,
        kc: &dyn Buffer,
        vc: &dyn Buffer,
        o: &dyn Buffer,
        pm: &dyn Buffer,
        pl: &dyn Buffer,
        pacc: &dyn Buffer,
        params: &dyn Buffer,
        nh: usize,
        nkv: usize,
        hd: usize,
        chunk: usize,
        n_chunks: usize,
        scale: f32,
        window: usize,
        // #74 slice 2: Some((k_addr, v_addr)) selects the `-DKV_BDA` twins; None keeps bound SSBOs.
        kv_addr: Option<(u64, u64)>,
    ) {
        let bda = kv_addr.is_some();
        let (p1name, p1spv) = match (crate::gemm::attn_hd_spec_disabled(self.vk()), bda) {
            (true, false) => (
                "attn_partial_dyn_nohd",
                crate::gemm::attn_partial_dyn_nohd_spv(),
            ),
            (true, true) => (
                "attn_partial_dyn_nohd_bda",
                crate::gemm::attn_partial_dyn_nohd_bda_spv(),
            ),
            (false, false) => ("attn_partial_dyn", crate::gemm::attn_partial_dyn_spv()),
            (false, true) => (
                "attn_partial_dyn_bda",
                crate::gemm::attn_partial_dyn_bda_spv(),
            ),
        };
        // Same fix as `attention_kv_split_dynac` above: attn_partial.comp's PC block is always 44
        // bytes (60 under -DKV_BDA); a shorter declared range fails
        // VUID-VkComputePipelineCreateInfo-layout-10069 even though `cap`/`pos`/`rows` are dead
        // code in this (`-DUSE_PARAMS`, no `-DSELF_CHUNK`) variant.
        let plen: usize = if bda { 60 } else { 44 };
        let k1 = self.be.kernel_sg(p1name, p1spv, 7, plen as u32, 32);
        let mut p1 = [0u8; 60];
        // [0..4] kv_len: unused (from params)
        p1[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        p1[16..20].copy_from_slice(&(chunk as u32).to_ne_bytes());
        p1[20..24].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p1[24..28].copy_from_slice(&(window as u32).to_ne_bytes());
        p1[28..32].copy_from_slice(&scale.to_ne_bytes());
        if let Some((k_addr, v_addr)) = kv_addr {
            p1[44..48].copy_from_slice(&(k_addr as u32).to_ne_bytes());
            p1[48..52].copy_from_slice(&((k_addr >> 32) as u32).to_ne_bytes());
            p1[52..56].copy_from_slice(&(v_addr as u32).to_ne_bytes());
            p1[56..60].copy_from_slice(&((v_addr >> 32) as u32).to_ne_bytes());
        }
        self.dispatch(
            k1,
            &[
                Self::vkb(q),
                Self::vkb(kc),
                Self::vkb(vc),
                Self::vkb(params),
                Self::vkb(pm),
                Self::vkb(pl),
                Self::vkb(pacc),
            ],
            3,
            &p1[..plen],
            (nh * n_chunks) as u32,
        );
        // pass 2: combine (structure-only, unchanged from the push-constant path)
        let k2 = self
            .be
            .kernel("attn_combine", crate::gemm::attn_combine_spv(), 4, 16);
        let ntile = if hd.is_multiple_of(4) { 4u32 } else { 1u32 };
        // attn_combine.comp splits hd as `hdt = hd/ntile`; a non-divisor ntile drops the top
        // hd-ntile*hdt dims (left uninitialized). ntile is 4 only when hd%4==0, else 1, so hd%ntile==0
        // always holds — this pins that invariant at the call site (never fires for shipped hd).
        debug_assert!(
            hd.is_multiple_of(ntile as usize),
            "attn_combine: hd ({hd}) must be a multiple of ntile ({ntile}); hdt=hd/ntile would drop dims"
        );
        let mut p2 = [0u8; 16];
        p2[0..4].copy_from_slice(&(nh as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(hd as u32).to_ne_bytes());
        p2[8..12].copy_from_slice(&(n_chunks as u32).to_ne_bytes());
        p2[12..16].copy_from_slice(&ntile.to_ne_bytes());
        self.dispatch(
            k2,
            &[Self::vkb(pm), Self::vkb(pl), Self::vkb(pacc), Self::vkb(o)],
            1,
            &p2,
            nh as u32 * ntile,
        );
    }

    /// Record a buffer→buffer copy of `bytes` from `src[src_offset..]` into `dst[dst_offset..]`.
    pub fn copy(
        &self,
        src: &dyn Buffer,
        src_offset: usize,
        dst: &dyn Buffer,
        dst_offset: usize,
        bytes: usize,
    ) {
        // Transfer chokepoint: previously un-stamped, so copy time silently landed in the
        // preceding label's bucket. Distinct from the `copy_strided` compute kernel.
        self.stamp("copy_buffer");
        let device = &self.be.shared.device;
        // `cmd_copy_buffer` takes raw handles + explicit byte offsets, not a descriptor — go
        // through `vk_handle`, not `vkb` (whose sub-range `DescriptorBufferInfo` doesn't apply
        // here). A resident-BDA sub-tensor's logical byte 0 sits at `sub_offset` within its shared
        // arena block, so that offset joins the caller's — for every ordinary buffer it is 0 and
        // this is the plain copy it always was.
        let src_h = Self::vk_handle(src);
        let dst_h = Self::vk_handle(dst);
        let src_sub = Self::vk_of(src).sub_offset;
        let dst_sub = Self::vk_of(dst).sub_offset;
        self.sync(&[src_h], &[dst_h], true);
        self.dirty_transfer.set(true);
        unsafe {
            device.cmd_copy_buffer(
                self.cmd,
                src_h,
                dst_h,
                &[vk::BufferCopy {
                    src_offset: (src_offset + src_sub) as u64,
                    dst_offset: (dst_offset + dst_sub) as u64,
                    size: bytes as u64,
                }],
            );
        }
    }

    pub fn attention(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        o: &dyn Buffer,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        let kern = self
            .be
            .kernel("attention", crate::gemm::attention_spv(), 4, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(t as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nh as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nkv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(hd as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(q), Self::vkb(k), Self::vkb(v), Self::vkb(o)],
            1,
            &push,
            (t * nh) as u32,
        );
    }

    /// Fused SwiGLU over a combined `gu` `[rows, 2*nff]` → `y` `[rows, nff]`. `clamp` is DeepSeek
    /// V4's per-layer SwiGLU limit (`Op::GatedActFused::swiglu_clamp`); `None` for every other arch.
    pub fn silu_mul_fused(
        &self,
        gu: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nff: usize,
        clamp: Option<f32>,
    ) {
        let k = self
            .be
            .kernel("silu_mul_fused", crate::gemm::silu_mul_fused_spv(), 2, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nff as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(u32::from(clamp.is_some())).to_ne_bytes());
        push[12..16].copy_from_slice(&clamp.unwrap_or(0.0).to_ne_bytes());
        // Thread-per-element kernel (local_size 64) — dispatch element-count/64 workgroups.
        self.dispatch(
            k,
            &[Self::vkb(gu), Self::vkb(y)],
            1,
            &push,
            ((rows * nff) as u32).div_ceil(64),
        );
    }

    /// Fused GeGLU (GELU tanh-approx gate) over a combined `gu` `[rows, 2*nff]` → `y` `[rows, nff]`.
    /// Same layout/dispatch as [`silu_mul_fused`]; gemma uses GELU instead of SiLU.
    pub fn gelu_mul_fused(
        &self,
        gu: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        nff: usize,
        clamp: Option<f32>,
    ) {
        let k = self
            .be
            .kernel("gelu_mul_fused", crate::gemm::gelu_mul_fused_spv(), 2, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nff as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(u32::from(clamp.is_some())).to_ne_bytes());
        push[12..16].copy_from_slice(&clamp.unwrap_or(0.0).to_ne_bytes());
        // Thread-per-element kernel (local_size 64) — dispatch element-count/64 workgroups.
        self.dispatch(
            k,
            &[Self::vkb(gu), Self::vkb(y)],
            1,
            &push,
            ((rows * nff) as u32).div_ceil(64),
        );
    }

    /// SwiGLU over separate contiguous `gate`/`up` buffers. `clamp` is DeepSeek V4's per-layer
    /// SwiGLU limit (`Op::GatedAct::swiglu_clamp`); `None` for every other arch.
    pub fn silu_mul(
        &self,
        gate: &dyn Buffer,
        up: &dyn Buffer,
        y: &dyn Buffer,
        n: usize,
        clamp: Option<f32>,
    ) {
        let k = self
            .be
            .kernel("silu_mul", crate::gemm::silu_mul_spv(), 3, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&(u32::from(clamp.is_some())).to_ne_bytes());
        push[32..36].copy_from_slice(&clamp.unwrap_or(0.0).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Gated-DeltaNet recurrence, one token (qwen35 SSM). The persistent `state` buffer
    /// `[nv*kd*vd]` is updated in place; `out` is `[nv*vd]`. One workgroup per value head; the
    /// `nk` q/k heads are tiled up to `nv`. See shaders/deltanet.comp.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        // The shader caches each column block's state [kd, 32] in shared memory (`ss[128*32]`), so kd
        // must be ≤ 128. qwen35 uses kd=128; assert so a larger head_k_dim fails loudly instead of
        // corrupting LDS.
        debug_assert!(
            kd <= 128,
            "deltanet shared-state block assumes kd ≤ 128, got {kd}"
        );
        let kern = self
            .be
            .kernel("deltanet", crate::gemm::deltanet_spv(), 9, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&eps.to_ne_bytes());
        push[24..28].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        // One workgroup per (value head, block of 32 state columns); local_size_x=32.
        let n_blk = vd.div_ceil(32);
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (nv * n_blk) as u32,
        );
    }

    /// Strided DeltaNet: q/k/v read from same source buffer at offsets 0, nk*kd, 2*nk*kd.
    /// Env-gated via INFR_DELTA_STRIDED=1. Push constants 32B (adds src_stride to standard).
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_strided(
        &self,
        q: &dyn Buffer, // all three are the same convout buffer when strided
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
        src_stride: usize,
    ) {
        debug_assert!(kd <= 128);
        let kern = self.be.kernel(
            "deltanet_strided",
            crate::gemm::deltanet_strided_spv(),
            9,
            32,
        );
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&eps.to_ne_bytes());
        push[24..28].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        push[28..32].copy_from_slice(&(src_stride as u32).to_ne_bytes());
        let n_blk = vd.div_ceil(32);
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2,
            &push,
            (nv * n_blk) as u32,
        );
    }

    /// CHUNKED gated-DeltaNet (chunkwise delta rule, C=32, one chunk when rows < 32): per 32-token
    /// chunk the recurrence collapses to dense matmuls + one unit-lower-triangular solve, so the
    /// state is traversed ⌈rows/32⌉ times instead of `rows`. Same signature/bindings as `deltanet`;
    /// math validated against the sequential form in tests/chunked_delta_math.rs (including a
    /// partial-chunk case). Use for rows ≥ 2 — prefill AND the MTP verify batch (m≈2-19 rows), which
    /// measurably beats the sequential kernel down to rows=2 (rows==1 decode keeps the sequential
    /// kernel; see the call site in adapter.rs for why). See shaders/deltanet_chunked.comp.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_chunked(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        debug_assert!(
            kd <= 128,
            "deltanet_chunked LDS chunk tiles assume kd ≤ 128, got {kd}"
        );
        let kern = self.be.kernel(
            "deltanet_chunked",
            crate::gemm::deltanet_chunked_spv(),
            9,
            28,
        );
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&eps.to_ne_bytes());
        push[24..28].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        // One workgroup per (value head, block of 32 state columns); local_size_x=256.
        // (COLS=16 was tried for occupancy and REGRESSED 2670→3668µs — the per-block duplicated
        // A/Wq dots dominate; that's what the split prep+scan variant hoists out.)
        let n_blk = vd.div_ceil(32);
        self.dispatch(
            kern,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(v),
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (nv * n_blk) as u32,
        );
    }

    /// Chunked gated-DeltaNet prefill, SPLIT variant (prep + gates + scan): the chunk-parallel
    /// work (q/k normalization, intra-chunk D=K̂K̂ᵀ / Dq=Q̂K̂ᵀ dot matrices, gates) is hoisted into
    /// two fully-parallel passes, so the sequential scan pass does ONLY state-coupled work — which
    /// parallelizes cleanly over small column blocks (COLS=16 → nv·(vd/16) workgroups). The
    /// monolithic `deltanet_chunked` duplicated that shared work per column block (~37% of it).
    /// Scratch (caller-alloc'd, alloc_uninit-safe — every read slot is written by prep/gates):
    /// kn/qn [rows·nk·kd] f32, dk/dq [nchunk·nk·C·C] f32, betag/gg [nchunk·nv·C] f32, C=32.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_chunked_split(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        kn: &dyn Buffer,
        qn: &dyn Buffer,
        dk: &dyn Buffer,
        dq: &dyn Buffer,
        betag: &dyn Buffer,
        gg: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        debug_assert!(kd <= 128, "deltanet split assumes kd ≤ 128, got {kd}");
        debug_assert!(
            kd.is_multiple_of(16),
            "deltanet_prep's coopmat D/Dq tiles assume kd a multiple of 16, got {kd}"
        );
        let nchunk = rows.div_ceil(32);
        // pass 1: prep — (chunk, k-head) grid. kernel_sg: the D/Dq dot matrices are computed via
        // coopmat (RDNA3 needs requiredSubgroupSize=32 pinned for coopmat correctness).
        let kp = self
            .be
            .kernel_sg("deltanet_prep", crate::gemm::deltanet_prep_spv(), 6, 20, 32);
        let mut p1 = [0u8; 20];
        p1[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nk as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(kd as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&eps.to_ne_bytes());
        p1[16..20].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        self.dispatch(
            kp,
            &[
                Self::vkb(q),
                Self::vkb(k),
                Self::vkb(kn),
                Self::vkb(qn),
                Self::vkb(dk),
                Self::vkb(dq),
            ],
            4, // kn, qn, dk, dq
            &p1,
            (nchunk * nk) as u32,
        );
        // pass 2: gates — (chunk, value-head) grid. Pinned to requiredSubgroupSize=32 (like the
        // rest of the DeltaNet family) so the 32-lane workgroup is exactly one subgroup and the
        // prefix log-decay scan can use subgroupInclusiveAdd.
        let kg = self.be.kernel_sg(
            "deltanet_gates",
            crate::gemm::deltanet_gates_spv(),
            6,
            8,
            32,
        );
        let mut p2 = [0u8; 8];
        p2[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        self.dispatch(
            kg,
            &[
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(betag),
                Self::vkb(gg),
            ],
            2, // betag, gg
            &p2,
            (nchunk * nv) as u32,
        );
        // pass 3: scan — (value head, column block) grid, sequential over chunks inside
        let ks = self
            .be
            .kernel_sg("deltanet_scan", crate::gemm::deltanet_scan_spv(), 9, 20, 32);
        let mut p3 = [0u8; 20];
        p3[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p3[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        p3[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        p3[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        p3[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        let n_blk = vd.div_ceil(8); // COLS=8, keep in sync with deltanet_scan.comp
        self.dispatch(
            ks,
            &[
                Self::vkb(v),
                Self::vkb(kn),
                Self::vkb(qn),
                Self::vkb(dk),
                Self::vkb(dq),
                Self::vkb(betag),
                Self::vkb(gg),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &p3,
            (nv * n_blk) as u32,
        );
    }

    /// Token-serial gated-DeltaNet prefill with the state column REGISTER-resident (norm + gates +
    /// seq). This is the default prefill path at `kd == 128`; `deltanet_chunked_split` remains the
    /// fallback (and the `INFR_DN_CHUNK_SCAN=1` A/B).
    ///
    /// The chunked delta rule does not actually do less arithmetic than the plain recurrence (~420M
    /// vs ~402M FMA per layer at qwen35/Ornith dims) — it only shortens the dependency chain, and it
    /// pays for that with LDS-resident state, unrollable-blocking runtime trip counts, ~96 workgroup
    /// barriers, and just nv·(vd/8)=256 workgroups. This form keeps the state in registers, uses
    /// single-subgroup workgroups (zero barriers; the two kd-contractions are subgroupAdd), and
    /// launches nv·vd = 2048 workgroups at the same total thread count. See deltanet_seq.comp.
    ///
    /// Scratch (caller-alloc'd, alloc_uninit-safe — every read slot is written by norm/gates):
    /// kn/qn `[rows·nk·kd]` f32, bet/dec `[rows·nv]` f32. No dk/dq — the D/Dq chunk dot matrices
    /// exist only for the chunked triangular solve, which this path does not perform.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_seq_split(
        &self,
        q: &dyn Buffer,
        k: &dyn Buffer,
        v: &dyn Buffer,
        blog: &dyn Buffer,
        alpha: &dyn Buffer,
        acoef: &dyn Buffer,
        dtbias: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        kn: &dyn Buffer,
        qn: &dyn Buffer,
        bet: &dyn Buffer,
        dec: &dyn Buffer,
        rows: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) {
        debug_assert_eq!(
            kd, 128,
            "deltanet_seq assumes kd == 128 (RPL=4 register shards); caller must fall back"
        );
        debug_assert!(
            vd.is_multiple_of(DN_SEQ_NCOL),
            "deltanet_seq's NCOL={DN_SEQ_NCOL} column blocking needs NCOL | vd, got vd={vd}"
        );
        // pass 1: norm — (token, k-head) grid, 32 lanes each
        let kn_k = self
            .be
            .kernel_sg("deltanet_norm", crate::gemm::deltanet_norm_spv(), 4, 20, 32);
        let mut p1 = [0u8; 20];
        p1[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p1[4..8].copy_from_slice(&(nk as u32).to_ne_bytes());
        p1[8..12].copy_from_slice(&(kd as u32).to_ne_bytes());
        p1[12..16].copy_from_slice(&eps.to_ne_bytes());
        p1[16..20].copy_from_slice(&(1.0f32 / (kd as f32).sqrt()).to_ne_bytes());
        self.dispatch(
            kn_k,
            &[Self::vkb(q), Self::vkb(k), Self::vkb(kn), Self::vkb(qn)],
            2, // kn, qn
            &p1,
            (rows * nk) as u32,
        );
        // pass 2: gates — flat rows·nv grid, 256 threads per workgroup
        let kg = self.be.kernel(
            "deltanet_gates_seq",
            crate::gemm::deltanet_gates_seq_spv(),
            6,
            8,
        );
        let mut p2 = [0u8; 8];
        p2[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p2[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        self.dispatch(
            kg,
            &[
                Self::vkb(blog),
                Self::vkb(alpha),
                Self::vkb(acoef),
                Self::vkb(dtbias),
                Self::vkb(bet),
                Self::vkb(dec),
            ],
            2, // bet, dec
            &p2,
            (rows * nv).div_ceil(256) as u32,
        );
        // pass 3: seq scan — one 32-lane workgroup per (value head, state column)
        let ks = self
            .be
            .kernel_sg("deltanet_seq", crate::gemm::deltanet_seq_spv(), 7, 20, 32);
        let mut p3 = [0u8; 20];
        p3[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        p3[4..8].copy_from_slice(&(nv as u32).to_ne_bytes());
        p3[8..12].copy_from_slice(&(nk as u32).to_ne_bytes());
        p3[12..16].copy_from_slice(&(kd as u32).to_ne_bytes());
        p3[16..20].copy_from_slice(&(vd as u32).to_ne_bytes());
        self.dispatch(
            ks,
            &[
                Self::vkb(kn),
                Self::vkb(qn),
                Self::vkb(v),
                Self::vkb(bet),
                Self::vkb(dec),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &p3,
            (nv * (vd / DN_SEQ_NCOL)) as u32,
        );
    }

    /// Causal depthwise conv1d + SiLU, one token (qwen35 SSM input conv). The per-channel history
    /// `state` `[(kconv-1)*cc]` is updated in place; `out` is `[cc]`. See shaders/conv1d_silu.comp.
    /// Resident-BDA route: `w.device_addr()` `Some` forks to the `-DSTREAMED` twin (the only
    /// weight build — see [`Self::conv1d_silu_at`]'s doc).
    pub fn conv1d_silu(
        &self,
        qkv: &dyn Buffer,
        w: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: conv1d_silu requires a u64 BDA device address");
        self.conv1d_silu_at(qkv, arena_addr, state, out, rows, cc, kconv);
    }

    /// BATCH depthwise conv1d + SiLU (rows ≥ kconv-1): pass 1 computes ALL rows·cc outputs in
    /// parallel from the virtual sequence [state ‖ qkv] (the sequential kernel walked the rows
    /// one by one, shuffling the history each token); pass 2 rebuilds the history from the last
    /// kconv-1 input rows. The recorder's hazard tracking orders pass 2 after pass 1 (pass 1
    /// reads the old state pass 2 overwrites). See shaders/conv1d_silu_par.comp / conv1d_shift.comp.
    /// Resident-BDA route: `w.device_addr()` `Some` forks pass 1 to the `-DSTREAMED` twin (the
    /// only weight build — see [`Self::conv1d_silu_batch_at`]'s doc); pass 2 never reads the
    /// weight, so it is unaffected.
    pub fn conv1d_silu_batch(
        &self,
        qkv: &dyn Buffer,
        w: &dyn Buffer,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: conv1d_silu_batch requires a u64 BDA device address");
        self.conv1d_silu_batch_at(qkv, arena_addr, state, out, rows, cc, kconv);
    }

    /// Arena-addressed body of [`Self::conv1d_silu`] (the qwen35 SSM input conv weight read
    /// through a typed 64-bit buffer_reference — see `shaders/conv1d_silu.comp`'s STREAMED doc).
    /// Binding 1 (the old resident build's weight SSBO) takes a harmless FILLER (`qkv`).
    /// Production routes here through [`Self::conv1d_silu`] (the weight's own BDA address);
    /// parity tests call it directly with an explicit arena address.
    pub fn conv1d_silu_at(
        &self,
        qkv: &dyn Buffer,
        arena_addr: u64,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        let kern = self
            .be
            .kernel("conv1d_silu", crate::gemm::conv1d_silu_spv(), 4, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cc as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(kconv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[
                Self::vkb(qkv),
                Self::vkb(qkv),
                Self::vkb(state),
                Self::vkb(out),
            ],
            2, // state (in/out) + out
            &push,
            (cc as u32).div_ceil(256),
        );
    }

    /// Arena-addressed body of [`Self::conv1d_silu_batch`]'s pass-1 dispatch (see
    /// `shaders/conv1d_silu_par.comp`'s STREAMED doc). Pass 2 (`conv1d_shift`) never reads the
    /// weight, so it stays the same dispatch regardless of the weight's residency — both legs
    /// share it. Binding 1 takes a harmless FILLER (`qkv`). Production routes here through
    /// [`Self::conv1d_silu_batch`] (the weight's own BDA address); parity tests call it directly
    /// with an explicit arena address.
    pub fn conv1d_silu_batch_at(
        &self,
        qkv: &dyn Buffer,
        arena_addr: u64,
        state: &dyn Buffer,
        out: &dyn Buffer,
        rows: usize,
        cc: usize,
        kconv: usize,
    ) {
        debug_assert!(
            rows >= kconv - 1,
            "conv1d_silu_batch_at needs rows ≥ kconv-1"
        );
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(cc as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(kconv as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let k1 = self
            .be
            .kernel("conv1d_silu_par", crate::gemm::conv1d_silu_par_spv(), 4, 20);
        self.dispatch(
            k1,
            &[
                Self::vkb(qkv),
                Self::vkb(qkv),
                Self::vkb(state),
                Self::vkb(out),
            ],
            1, // out only (state is read-only here)
            &push,
            ((rows * cc) as u32).div_ceil(256),
        );
        let mut push2 = [0u8; 12];
        push2[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push2[4..8].copy_from_slice(&(cc as u32).to_ne_bytes());
        push2[8..12].copy_from_slice(&(kconv as u32).to_ne_bytes());
        let k2 = self
            .be
            .kernel("conv1d_shift", crate::gemm::conv1d_shift_spv(), 2, 12);
        self.dispatch(
            k2,
            &[Self::vkb(qkv), Self::vkb(state)],
            1, // state out
            &push2,
            (((kconv - 1) * cc) as u32).div_ceil(256),
        );
    }

    /// Batched strided row copy in ONE dispatch (word granularity): `rows` slices of `nw` u32
    /// words, `dst[dst_off + r*dst_stride + ..nw] = src[src_off + r*src_stride + ..nw]` (all in
    /// words). Replaces the per-row copy-command loop for Op::CopyStrided — at rows=512 that was
    /// 512 vkCmdCopyBuffer + hazard checks per split op, dwarfing the bytes moved.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_strided(
        &self,
        src: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        nw: usize,
        src_off: usize,
        src_stride: usize,
        dst_off: usize,
        dst_stride: usize,
    ) {
        let kern = self
            .be
            .kernel("copy_strided", crate::gemm::copy_strided_spv(), 2, 24);
        let mut push = [0u8; 24];
        for (i, v) in [rows, nw, src_off, src_stride, dst_off, dst_stride]
            .iter()
            .enumerate()
        {
            push[i * 4..i * 4 + 4].copy_from_slice(&(*v as u32).to_ne_bytes());
        }
        self.dispatch(
            kern,
            &[Self::vkb(src), Self::vkb(dst)],
            1,
            &push,
            ((rows * nw) as u32).div_ceil(256),
        );
    }

    /// Elementwise sigmoid gate: `y[i] = a[i] * sigmoid(b[i])` (Qwen3-Next attention output gate).
    /// `gate_stride_elements`, `gate_row_width`, `gate_block_width`: interleaved gate buffer params.
    /// All zero = flat gate read.
    pub fn mul_sigmoid(
        &self,
        a: &dyn Buffer,
        b: &dyn Buffer,
        y: &dyn Buffer,
        n: usize,
        gate_stride_el: usize,
        gate_row_width: usize,
        gate_block_width: usize,
    ) {
        let kern = self
            .be
            .kernel("mul_sigmoid", crate::gemm::mul_sigmoid_spv(), 3, 16);
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(gate_stride_el as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(gate_row_width as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(gate_block_width as u32).to_ne_bytes());
        self.dispatch(
            kern,
            &[Self::vkb(a), Self::vkb(b), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// GeGLU with separate gate/up buffers: `y[i] = gelu(gate[i]) * up[idx(i)]` (GELU
    /// tanh-approx). `up_off_bytes` lets a layer-major slice of a larger buffer be read in place
    /// (gemma4 per-layer-embd gate: `gelu(inp_gate·hidden) * inp_per_layer[il]`).
    /// `up_stride_bytes` is the per-row byte stride when reading from a wider row-major tensor;
    /// 0 means rows are tightly packed. `row_width` is the element count per row.
    pub fn gelu_mul_off(
        &self,
        gate: &dyn Buffer,
        up: &dyn Buffer,
        up_off_bytes: usize,
        up_stride_bytes: usize,
        row_width: usize,
        gate_stride_bytes: usize,
        gate_row_width: usize,
        gate_block_width: usize,
        y: &dyn Buffer,
        n: usize,
    ) {
        let k = self
            .be
            .kernel("gelu_mul", crate::gemm::gelu_mul_spv(), 3, 36);
        // The strided GeGLU shares silu_mul.comp (`-DGELU`), so it pushes that kernel's FULL
        // params — the trailing `do_clamp`/`limit` pair included, left zero here: no caller clamps
        // through this form (gemma4's per-layer-embd gate), and a short push would leave the
        // shader reading whatever the last dispatch left in those two words.
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&((up_off_bytes / 4) as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&((up_stride_bytes / 4) as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(row_width as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&((gate_stride_bytes / 4) as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(gate_row_width as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&((gate_block_width / 4) as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(gate), Self::vkb(up), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Whether the id-indexed native GEMV (GPU-resident MoE routing) supports this expert format.
    pub fn native_id_supported(dtype: infr_core::DType) -> bool {
        crate::linear::native_id_kernel_name(dtype).is_some()
    }

    /// GPU MoE router top-k for `n_tokens` tokens (one workgroup per token): top-`n_used` over each
    /// token's `logits[n_expert]` → selected expert `ids` + `wts` (per token, `n_used` each), all in
    /// VRAM (no host round-trip). `scale` = routing scale. `gating` (0 = softmax, 1 = sigmoid) and
    /// `norm_w` (renormalize the selected weights to sum to 1) select the weighting formula —
    /// qwen3moe/qwen35moe/diffusion-gemma pass (softmax, true); llama4 passes (sigmoid, false).
    /// Bound-buffer order matters: `dispatch` treats the last `n_out` buffers as writes, so the two
    /// WRITTEN buffers (`ids`, `wts`) must occupy the trailing slots for the hazard tracker to see
    /// the `ids` write; `bias` is read-only.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk(
        &self,
        logits: &dyn Buffer,
        ids: &dyn Buffer,
        wts: &dyn Buffer,
        bias: &dyn Buffer,
        n_tokens: usize,
        n_expert: usize,
        n_used: usize,
        scale: f32,
        gating: u32,
        norm_w: bool,
        has_bias: bool,
        n_expert_groups: u32,
        n_expert_groups_used: u32,
        hash_ids: &dyn Buffer,
        hash: bool,
    ) {
        let k = self
            .be
            .kernel("moe_topk", crate::gemm::moe_topk_spv(), 5, 36);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(n_expert as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&scale.to_ne_bytes());
        push[12..16].copy_from_slice(&gating.to_ne_bytes());
        push[16..20].copy_from_slice(&(norm_w as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(has_bias as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&n_expert_groups.to_ne_bytes());
        push[28..32].copy_from_slice(&n_expert_groups_used.to_ne_bytes());
        push[32..36].copy_from_slice(&(hash as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(logits),
                Self::vkb(bias),
                Self::vkb(hash_ids),
                Self::vkb(ids),
                Self::vkb(wts),
            ],
            2,
            &push,
            n_tokens as u32,
        );
    }

    /// Expert-parallel (multi-GPU EP) band remap: after the replicated router `moe_topk`, rewrite
    /// this rank's copy of the GLOBAL top-k `ids` into LOCAL expert-shard indices for the band
    /// `[lo, hi)` (`id-lo` in-band; `0` out-of-band) and zero the out-of-band `wts`, so THIS rank's
    /// expert GEMVs/GEMMs compute only its owned experts and the weighted combine drops the rest.
    /// `n = rows * n_used` packed (token, slot) assignments. Both buffers are read-modify-write. See
    /// `moe_ep_band_remap.comp` and `Op::MoeFfn`'s `ep_band`.
    pub fn moe_ep_band_remap(
        &self,
        ids: &dyn Buffer,
        wts: &dyn Buffer,
        n: usize,
        lo: u32,
        hi: u32,
    ) {
        let k = self.be.kernel(
            "moe_ep_band_remap",
            crate::gemm::moe_ep_band_remap_spv(),
            2,
            12,
        );
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&lo.to_ne_bytes());
        push[8..12].copy_from_slice(&hi.to_ne_bytes());
        // ids + wts are both read-modify-write — bind both in the WRITE slots (n_out = 2) so the
        // downstream expert GEMVs get a RAW barrier against this rewrite.
        self.dispatch(k, &[Self::vkb(ids), Self::vkb(wts)], 2, &push, n as u32);
    }

    /// llama4 weight-before-FFN: in-place per-row scale `y[row*width+i] *= w[row]` — applied to the
    /// gate/up projection OUTPUT (before the activation), exact-equivalent to the CPU's
    /// input-side fold since gate/up are linear (see `Op::MoeFfn`'s `weight_before` doc and the
    /// shader comment). `n_rows`/`w` are at whatever granularity `y` is packed at (small-m: n_slots
    /// = rows*n_used; batched: n_pairs bucket-sorted assignments).
    pub fn moe_weight_scale(&self, y: &dyn Buffer, w: &dyn Buffer, n_rows: usize, width: usize) {
        let k = self.be.kernel(
            "moe_weight_scale",
            crate::gemm::moe_weight_scale_spv(),
            2,
            8,
        );
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n_rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(width as u32).to_ne_bytes());
        // w first, y LAST: dispatch3 hazard-tracks the last n_out buffers as writes — y is the
        // read-modify-write target, so it must sit in the write slot or the downstream activation
        // kernel races the in-place scale (shader bindings match this order).
        self.dispatch(
            k,
            &[Self::vkb(w), Self::vkb(y)],
            1,
            &push,
            ((n_rows * width) as u32).div_ceil(64),
        );
    }

    /// Greedy argmax over `[rows, n]` logits → token ids (u32) in `out_id[0..rows]`. Lets greedy
    /// decode (rows == 1) and the MTP verify accept (rows == m, issue #31) read back 4 bytes/row
    /// instead of the whole vocab logits.
    pub fn argmax(
        &self,
        logits: &dyn Buffer,
        part: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        rows: usize,
    ) {
        // Two-stage (see argmax.comp): 256 slice partials in parallel across the GPU, then a
        // one-workgroup reduce. `part` = 512 f32 scratch (vals + idx bit-patterns), reused across
        // rows — the hazard tracker orders row r+1's partial pass after row r's reduce (WAR).
        // Rows loop sequentially (m is small, 2-7 for MTP); a single-workgroup whole-vocab scan
        // measured 18% SLOWER than the download it replaced, so each row keeps the full
        // 256-workgroup two-stage shape.
        let k1 = self
            .be
            .kernel("argmax_part", crate::gemm::argmax_part_spv(), 2, 8);
        let k2 = self.be.kernel("argmax", crate::gemm::argmax_spv(), 2, 8);
        for row in 0..rows {
            let mut p1 = [0u8; 8];
            p1[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
            p1[4..8].copy_from_slice(&((row * n) as u32).to_ne_bytes());
            self.dispatch(k1, &[Self::vkb(logits), Self::vkb(part)], 1, &p1, 256);
            let mut p2 = [0u8; 8];
            p2[0..4].copy_from_slice(&256u32.to_ne_bytes());
            p2[4..8].copy_from_slice(&(row as u32).to_ne_bytes());
            self.dispatch(k2, &[Self::vkb(part), Self::vkb(out_id)], 1, &p2, 1);
        }
    }

    /// Fused single-row argmax + softmax top-1 probability (`Op::ArgmaxProb`) — the MTP
    /// draft-loop accept (issue #33 follow-up to [`Recorder::argmax`]'s VERIFY-side fusion,
    /// de35727): `out_id[0]`/`out_prob[0]` read back 8 bytes total instead of the `[n]` logits row
    /// plus the host argmax + `sum exp` scan it replaces (measured the dominant per-step draft
    /// cost, not the download itself — see `Op::ArgmaxProb`'s doc). Two-stage, the SAME
    /// 256-workgroup-partials -> 1-workgroup-merge shape as `argmax`, with an online-softmax
    /// `sum_exp` carried alongside the (max, idx) reduction (see argmax_prob.comp). Single row
    /// only — the draft loop self-chains one token at a time.
    pub fn argmax_prob(
        &self,
        logits: &dyn Buffer,
        part: &dyn Buffer,
        out_id: &dyn Buffer,
        out_prob: &dyn Buffer,
        n: usize,
    ) {
        let k1 = self.be.kernel(
            "argmax_prob_part",
            crate::gemm::argmax_prob_part_spv(),
            2,
            4,
        );
        let k2 = self
            .be
            .kernel("argmax_prob", crate::gemm::argmax_prob_spv(), 3, 4);
        let p1 = (n as u32).to_ne_bytes();
        self.dispatch(k1, &[Self::vkb(logits), Self::vkb(part)], 1, &p1, 256);
        let p2 = 256u32.to_ne_bytes();
        // TWO output buffers (`out_id` AND `out_prob`) — `n_out` must be 2, not 1, or the hazard
        // tracker's `dispatch3` split treats `out_id` as a READ (see that fn's doc), so a LATER op
        // in the SAME graph that reads `out_id` (issue #33's chained draft loop feeds it straight
        // into the next step's `Op::EmbedGather` `ids`) gets no RAW barrier against this write —
        // previously harmless because nothing downstream of a fused `id_prob` ever read `out_id`
        // within the same `execute()` call (the caller always downloaded it after a full
        // queue-idle `execute`), until `build_mtp_draft_chain_graph` did exactly that.
        self.dispatch(
            k2,
            &[Self::vkb(part), Self::vkb(out_id), Self::vkb(out_prob)],
            2,
            &p2,
            1,
        );
    }

    /// Emit an UNCONDITIONAL full compute→compute memory barrier. The chained decode replays one
    /// recording back-to-back in a single submission — commands from consecutive replays may
    /// otherwise overlap (the hazard tracker only orders dispatches WITHIN a recording), so a
    /// global barrier at the top of the recording fences iteration i's tail writes (sampled id,
    /// params) from iteration i+1's head reads.
    pub fn global_barrier(&self) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            self.be.shared.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );
        }
        *self.barriers.borrow_mut() += 1;
    }

    /// Full COMPUTE↔TRANSFER memory+execution barrier for the dense-streaming BDA arena. The
    /// streamed GEMV reads the arena by 64-bit pointer (native_weight_addr.glsl), so the buffer-based
    /// hazard tracker (`sync`) never sees the read — the ordering MUST be recorded explicitly around
    /// the ring→arena staging copies, the way the paged MoE flow orders its own pointer-read arena.
    /// Bracketing each miss's copy with this barrier gives both directions in one primitive:
    ///   • RAW — the ring→arena copy (`TRANSFER_WRITE`) is visible to the reading dispatch
    ///     (`SHADER_READ`) [`TRANSFER`→`COMPUTE_SHADER`];
    ///   • WAR — a prior dispatch's arena read (`SHADER_READ`) completes before a later copy
    ///     overwrites that slot (`TRANSFER_WRITE`) [`COMPUTE_SHADER`→`TRANSFER`].
    /// Streaming is PCIe/residency-bound, so the extra barrier per staged block is noise.
    pub fn arena_stream_barrier(&self) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            self.be.shared.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );
        }
        *self.barriers.borrow_mut() += 1;
    }

    /// Log the sampled token id into the chained-decode ring: `ring[params[0] & 63] = id[0]`.
    /// Recorded LAST in the decode replay (after Argmax/Sample wrote `id`) — the host reads the
    /// whole chunk's ids from the ring after ONE n-replay submission.
    pub fn id_log(&self, params: &dyn Buffer, id: &dyn Buffer, ring: &dyn Buffer) {
        let k = self.be.kernel("id_log", crate::gemm::id_log_spv(), 3, 0);
        self.dispatch(
            k,
            &[Self::vkb(params), Self::vkb(id), Self::vkb(ring)],
            1,
            &[],
            1,
        );
    }

    /// Advance the decode-replay `params` SSBO on the device: `[pos, kv_len] -> [pos+1, pos+2]`.
    /// Recorded FIRST in the decode replay so the position stream never leaves the GPU.
    pub fn params_advance(&self, params: &dyn Buffer) {
        let k = self
            .be
            .kernel("params_advance", crate::gemm::params_advance_spv(), 1, 0);
        self.dispatch(k, &[Self::vkb(params)], 1, &[], 1);
    }

    /// Largest `top_k` the GPU stochastic sampler handles; above this the caller samples on the host.
    pub const SAMPLE_KMAX: usize = crate::gemm::SAMPLE_KMAX;

    /// Vocab-scale GPU stochastic sampling (`Op::Sample`): temperature + top-k + top-p over `n`
    /// logits → token id in `out_id[0]`, with the uniform draw read from the 1-float `u` BUFFER
    /// (record-once replayable — the host rewrites 4 bytes per token). Two-stage (see
    /// sample_topk.comp); `cand` = 2*256*top_k f32 scratch. Requires `2 <= top_k <= SAMPLE_KMAX`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_topk(
        &self,
        logits: &dyn Buffer,
        cand: &dyn Buffer,
        u: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        top_k: usize,
        temp: f32,
        top_p: f32,
    ) {
        debug_assert!((2..=Self::SAMPLE_KMAX).contains(&top_k));
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(top_k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&temp.to_ne_bytes());
        push[12..16].copy_from_slice(&top_p.to_ne_bytes());
        let k1 = self.be.kernel(
            "sample_topk_part",
            crate::gemm::sample_topk_part_spv(),
            2,
            16,
        );
        self.dispatch(k1, &[Self::vkb(logits), Self::vkb(cand)], 1, &push, 256);
        let mut push2 = [0u8; 16];
        push2[0..4].copy_from_slice(&(256u32 * top_k as u32).to_ne_bytes());
        push2[4..16].copy_from_slice(&push[4..16]);
        let k2 = self
            .be
            .kernel("sample_topk", crate::gemm::sample_topk_spv(), 3, 16);
        self.dispatch(
            k2,
            &[Self::vkb(cand), Self::vkb(u), Self::vkb(out_id)],
            1,
            &push2,
            1,
        );
    }

    /// Vocab-scale GPU stochastic sampling for the CHAINED/record-once decode replay (`Op::Sample`
    /// under `RopeMode::Dynamic`): identical to [`Recorder::sample_topk`], but stage 2 reads its
    /// uniform draw from the 64-slot `u` ring at `u_buf[params[0] & 63]` instead of `u_buf[0]` —
    /// the same ring geometry `id_log` keys its ring on, so ONE recording replays correctly whether
    /// the caller drives it singly (`execute`) or chained (`execute_chain`, N replays/submission),
    /// consuming the host RNG stream in lockstep with the per-token host sampler either way.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_topk_chain(
        &self,
        logits: &dyn Buffer,
        cand: &dyn Buffer,
        u: &dyn Buffer,
        params: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        top_k: usize,
        temp: f32,
        top_p: f32,
    ) {
        debug_assert!((2..=Self::SAMPLE_KMAX).contains(&top_k));
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(top_k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&temp.to_ne_bytes());
        push[12..16].copy_from_slice(&top_p.to_ne_bytes());
        let k1 = self.be.kernel(
            "sample_topk_part",
            crate::gemm::sample_topk_part_spv(),
            2,
            16,
        );
        self.dispatch(k1, &[Self::vkb(logits), Self::vkb(cand)], 1, &push, 256);
        let mut push2 = [0u8; 16];
        push2[0..4].copy_from_slice(&(256u32 * top_k as u32).to_ne_bytes());
        push2[4..16].copy_from_slice(&push[4..16]);
        // Binding order matches sample_topk.comp's CHAIN layout: cand(0), u_buf(1), params(2),
        // out_id(3) — `params` is a READ so it sits before the write (`dispatch`'s hazard tracker
        // treats the trailing `n_out` bound buffers as writes).
        let k2 = self.be.kernel(
            "sample_topk_chain",
            crate::gemm::sample_topk_chain_spv(),
            4,
            16,
        );
        self.dispatch(
            k2,
            &[
                Self::vkb(cand),
                Self::vkb(u),
                Self::vkb(params),
                Self::vkb(out_id),
            ],
            1,
            &push2,
            1,
        );
    }

    /// GPU stochastic sampling over `n` logits → token id in `out_id[0]`: temperature + top-k +
    /// top-p (nucleus) via a radix N-ary select, inverse-CDF sampled with the host-drawn uniform `u`.
    /// Requires `2 ≤ top_k ≤ SAMPLE_KMAX`. Only the token reads back — the vocab logits stay in VRAM.
    pub fn sample(
        &self,
        logits: &dyn Buffer,
        out_id: &dyn Buffer,
        n: usize,
        top_k: usize,
        temp: f32,
        top_p: f32,
        u: f32,
    ) {
        let k = self
            .be
            .kernel("moe_sample", crate::gemm::moe_sample_spv(), 2, 20);
        let mut push = [0u8; 20];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(top_k as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&temp.to_ne_bytes());
        push[12..16].copy_from_slice(&top_p.to_ne_bytes());
        push[16..20].copy_from_slice(&u.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(logits), Self::vkb(out_id)], 1, &push, 1);
    }

    /// DiffusionGemma perf slice 3 (docs/diffusion-gemma.md): fused per-canvas-row entropy-bound
    /// sampler reduction — argmax/entropy/CDF-sample over `[rows, dim]` logits, one workgroup per
    /// row. `u` is `rows` host-drawn uniform `[0,1)` floats (the CDF-inversion target draw).
    /// Writes `argmax_out`/`entropy_out`/`sampled_out` (each `[rows]`) — only those tiny arrays
    /// need to leave the GPU, not the `[rows, dim]` logits themselves. See `dg_eb_sample.comp`.
    #[allow(clippy::too_many_arguments)]
    pub fn dg_eb_sample(
        &self,
        logits: &dyn Buffer,
        u: &dyn Buffer,
        argmax_out: &dyn Buffer,
        entropy_out: &dyn Buffer,
        sampled_out: &dyn Buffer,
        rows: usize,
        dim: usize,
        temp_inv: f32,
    ) {
        let k = self
            .be
            .kernel_sg("dg_eb_sample", crate::gemm::dg_eb_sample_spv(), 5, 8, 32);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(dim as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&temp_inv.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(logits),
                Self::vkb(u),
                Self::vkb(argmax_out),
                Self::vkb(entropy_out),
                Self::vkb(sampled_out),
            ],
            3,
            &push,
            rows as u32,
        );
    }

    /// Zero a buffer's first `n` 4-byte elements (cmd_fill_buffer) — clears the bucket counters.
    pub fn zero(&self, buf: &dyn Buffer, n: usize) {
        // Transfer chokepoint (see `copy`): previously an un-stamped blind spot.
        self.stamp("zero_fill");
        // `cmd_fill_buffer` takes a raw handle + explicit byte offset, not a descriptor — go
        // through `vk_handle`, not `vkb`. The fill starts at the buffer's `sub_offset` (0 for every
        // ordinary buffer): a resident-BDA sub-tensor shares its arena block's handle, and a fill
        // at byte 0 would clobber whatever tensor sits at the block's front — same rule as
        // `fill_buf`'s arena arm in lib.rs.
        let buf_h = Self::vk_handle(buf);
        let sub = Self::vk_of(buf).sub_offset;
        self.sync(&[], &[buf_h], true);
        unsafe {
            self.be
                .shared
                .device
                .cmd_fill_buffer(self.cmd, buf_h, sub as u64, (n * 4) as u64, 0);
        }
    }

    /// MoE bucketing pass 1 (count): tally assignments per expert into `counts` (pre-zeroed).
    pub fn moe_bucket_count(&self, tok_ids: &dyn Buffer, counts: &dyn Buffer, n_pairs: usize) {
        let k = self.be.kernel(
            "moe_bucket_count",
            crate::gemm::moe_bucket_count_spv(),
            2,
            4,
        );
        self.dispatch(
            k,
            &[Self::vkb(tok_ids), Self::vkb(counts)],
            1,
            &(n_pairs as u32).to_ne_bytes(),
            (n_pairs as u32).div_ceil(64),
        );
    }

    /// MoE bucketing pass 2 (scan): exclusive prefix sum `counts → offsets`. The scatter's `fill`
    /// counters are zeroed separately (`rec.zero`) so the clear overlaps the count/scan instead of
    /// riding this 1-lane serial pass.
    pub fn moe_bucket_scan(&self, counts: &dyn Buffer, offsets: &dyn Buffer, n_expert: usize) {
        let k = self
            .be
            .kernel("moe_bucket_scan", crate::gemm::moe_bucket_scan_spv(), 2, 4);
        self.dispatch(
            k,
            &[Self::vkb(counts), Self::vkb(offsets)],
            1,
            &(n_expert as u32).to_ne_bytes(),
            1,
        );
    }

    /// MoE bucketing pass 3 (scatter): group token rows + weights by expert into `bucket_rows` /
    /// `bucket_wts` (each expert's run starts at `offsets[e]`). `dscale`, when given, is a
    /// per-expert weight (diffusion-gemma's `ffn_down_exps.scale`) baked into `bucket_wts` at
    /// scatter time — the scatter already has the expert id (`e`) in hand to index it, so this
    /// is a free multiply here vs. a separate post-GEMM pass, and `moe_scatter_reduce` needs no
    /// changes at all (it just sums already-scaled weights). Equivalent to the per-token path's
    /// `moe_accumulate_scaled` since the scale is linear in the down output.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_bucket_scatter(
        &self,
        tok_ids: &dyn Buffer,
        tok_wts: &dyn Buffer,
        offsets: &dyn Buffer,
        fill: &dyn Buffer,
        bucket_rows: &dyn Buffer,
        bucket_wts: &dyn Buffer,
        inv_pos: &dyn Buffer,
        dscale: Option<&dyn Buffer>,
        n_pairs: usize,
        n_used: usize,
    ) {
        let (name, spv): (_, _) = match dscale {
            Some(_) => (
                "moe_bucket_scatter_scaled",
                crate::gemm::moe_bucket_scatter_scaled_spv(),
            ),
            None => ("moe_bucket_scatter", crate::gemm::moe_bucket_scatter_spv()),
        };
        let nb = if dscale.is_some() { 8 } else { 7 };
        let k = self.be.kernel(name, spv, nb, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n_pairs as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        let mut bufs = vec![Self::vkb(tok_ids), Self::vkb(tok_wts), Self::vkb(offsets)];
        if let Some(ds) = dscale {
            bufs.push(Self::vkb(ds));
        }
        bufs.extend_from_slice(&[
            Self::vkb(fill),
            Self::vkb(bucket_rows),
            Self::vkb(bucket_wts),
            Self::vkb(inv_pos),
        ]);
        self.dispatch(k, &bufs, 4, &push, (n_pairs as u32).div_ceil(64));
    }

    /// Fused gather+quant for the batched MoE pipeline: quantize `n_slots` BUCKET rows (each
    /// reading its source token row via `bucket_rows`) straight into the packed expert-grouped
    /// layout — one dispatch replaces the per-expert gather and per-expert quant stages.
    #[allow(clippy::too_many_arguments)]
    pub fn quant_q8_gather(
        &self,
        a: &dyn Buffer,
        bucket_rows: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        n_slots: usize,
        k_dim: usize,
    ) {
        let kq = self.be.kernel_sg(
            "quant_q8_gather",
            crate::gemm::quant_q8_gather_spv(),
            5,
            12,
            32,
        );
        let mut p = [0u8; 12];
        p[0..4].copy_from_slice(&(n_slots as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k_dim as u32).to_ne_bytes());
        p[8..12].copy_from_slice(&32u32.to_ne_bytes());
        self.dispatch(
            kq,
            &[
                Self::vkb(a),
                Self::vkb(bucket_rows),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
            ],
            3,
            &p,
            (n_slots * (k_dim / 32)) as u32,
        );
    }

    /// ALL experts' mmq GEMMs in ONE dispatch (`gl_WorkGroupID.y` = expert): activation rows are
    /// packed by expert (bucket layout, segment e = offsets[e]..+counts[e]); the weight bank is
    /// indexed `w_base + e·stride`. Grid x covers the worst-case row tiles (`ceil(rows/BM)` — the
    /// whole chunk landing on one expert); tiles past a segment exit immediately, so the empty
    /// launches cost ~nothing while the dispatch count drops from ~n_expert·stages per layer to
    /// stages. `sact` is Q4_K/Q5_K/Q5_1's min-term row sums (None for Q6_K/Q8_0/Q5_0/Q3_K, which
    /// have no min, AND for Q2_K — also min-carrying, but its 16-elem sub-block is HALF the
    /// activation's 32-elem `sact` granularity, so it self-computes its own 16-wide Σx in-shader
    /// instead of binding this buffer; see `native_gemm_mmq_q2_k.comp`'s doc).
    ///
    /// `rows` is a GRID-SIZING BOUND ONLY (never read by the shader — the kernel derives every
    /// real row range from `counts`/`offsets`): it must be `>=` the largest possible `counts[e]`
    /// over any expert `e`. Pass the CHUNK's TOKEN count (`x.numel()/ne`), not `n_pairs` (=
    /// `tokens·n_used`, the total packed-buffer length) — a top-k router never assigns a token to
    /// the same expert twice, so `counts[e] <= tokens` always (one assignment per token, at most),
    /// an `n_used`×-tighter bound than `n_pairs`. Passing `n_pairs` here is still CORRECT (a valid
    /// superset of row tiles, just with more early-exiting empty ones) but launches `n_used`× the
    /// necessary workgroups — caught during the diffusion-gemma perf campaign (slice 5): DG's
    /// caller was passing `n_pairs` (n_used=8 → 8× the row-tile grid) alongside qwen3moe's caller,
    /// which had the same bug. Fixed at both call sites in `adapter.rs`'s `Op::MoeFfn` lowering.
    ///
    /// `n_used`: NOT read by the shader — used HERE only to pick the row TILE size (BM). The
    /// default kernel tiles M in BM=64 chunks; at small rows-per-expert (Qwen3.6-MoE's 256-expert
    /// pool averages `rows·n_used/n_expert` ≈ 16/expert at pp512) that tile is mostly masked
    /// waste — a full 64-row tile launches, does the full BM×BN×K compute, but only ~16 of the 64
    /// rows are real (the rest are past the expert's segment and their results are discarded at
    /// the store). `rows·n_used/n_expert` (integer division; sum(counts) == rows·n_used == n_pairs
    /// spread evenly is the EXACT mean, not a bound) selects a BM=32 tile
    /// (`native_gemm_mmq_*_xp32`) below `MOE_EXPERT_SMALL_TILE_AVG_ROWS`, halving the average
    /// waste — measured via `moe_expert_grid_bound_bench`'s tile sweep (see its doc): BM=32 wins at
    /// avg~16 (qwen3.6-MoE), BM=64 (unchanged) still wins at avg~32+ (qwen3-30B-A3B, DG) where
    /// tiles are fuller and the smaller tile's extra per-workgroup fixed cost (barrier/staging,
    /// amortized over less work) stops paying for itself. Same K-accumulation order either way —
    /// this is tile GRANULARITY only, so results stay bit-identical to the BM=64 path.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_experts(
        &self,
        dtype: infr_core::DType,
        stage: &'static str,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: Option<&dyn Buffer>,
        w: &dyn Buffer,
        w_base: usize,
        stride: usize,
        counts: &dyn Buffer,
        offsets: &dyn Buffer,
        c: &dyn Buffer,
        rows: usize,
        k: usize,
        n: usize,
        n_expert: usize,
        n_used: usize,
    ) {
        // The stacked expert bank is read from its resident-BDA arena address, never a bound SSBO.
        // `pc.m` is REPURPOSED as the per-expert BYTE stride applied on the pointer
        // (`w_addr += uint64_t(ex) * uint64_t(pc.m)`, see the shaders' STREAMED EXPERT_GRID arm) —
        // [`expert_stride_bytes`] converts the caller's element-space `stride` using the dtype's
        // GGUF block layout. `w_base` passes through unchanged (the small within-bank element
        // offset the shader still adds). The vacated weight-SSBO slot takes a small FILLER (`qa`,
        // same convention as [`Self::matmul_mmq`]) so the binding count/order matches the shader's
        // layout. Push layout matches the shaders' STREAMED PC struct: `m(=stride bytes), n, k,
        // w_base, arena_lo, arena_hi`.
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: matmul_mmq_experts requires a u64 BDA device address");
        self.label_next(stage);
        let avg_rows = rows.saturating_mul(n_used) / n_expert.max(1);
        let small_tile = avg_rows <= MOE_EXPERT_SMALL_TILE_AVG_ROWS;
        let wide_bn = n.is_multiple_of(128)
            && (small_tile || k <= MOE_EXPERT_WIDE_BN_MAX_K)
            && matches!(dtype, infr_core::DType::Q4K | infr_core::DType::Q6K);
        let bm: usize = if small_tile { 32 } else { 64 };
        let bn: usize = if wide_bn { 128 } else { 64 };
        // Kernel selection from the SINGLE dtype→kernel table shared with the paged twin (see
        // [`moe_mmq_desc`]) — resident suffix + tile picked here, byte-identical to the old inline
        // match. `wide_bn` is only ever set for Q4_K/Q6_K (see its definition above), which are the
        // only dtypes carrying `xp_wide`.
        let desc = moe_mmq_desc(dtype).unwrap_or_else(|| {
            unreachable!(
                "batched MoE expert GEMM (streamed): the MOE_MMQ_DTYPES set only (Q4_0/Q4_1/Q4_K/\
                 Q5_K/Q6_K/Q8_0/Q5_0/Q5_1/Q2_K/Q3_K/Q2_0/IQ4_NL/IQ4_XS/IQ2_S/IQ3_S/MXFP4/NVFP4)"
            )
        });
        let (name, spv) = if wide_bn {
            let (w128, w32w) = desc
                .xp_wide
                .expect("wide_bn is set only for Q4_K/Q6_K, which carry xp_wide");
            if small_tile {
                w32w
            } else {
                w128
            }
        } else if small_tile {
            desc.xp32
        } else {
            desc.xp
        };
        let nb = desc.resident_nb;
        let kern = self.be.kernel(name, spv, nb, 24);
        let mut push = [0u8; 24];
        let stride_bytes = expert_stride_bytes(dtype, stride);
        push[0..4].copy_from_slice(&stride_bytes.to_ne_bytes()); // pc.m = per-expert BYTE stride
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(w_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let gx = (rows.div_ceil(bm) * (n / bn)) as u32;
        let mut bufs = vec![Self::vkb(qa), Self::vkb(dact)];
        if let Some(sa) = sact {
            bufs.push(Self::vkb(sa));
        }
        // The vacated weight-SSBO slot takes a small FILLER (`qa`) — same convention as
        // `matmul_mmq_at`'s dense twin; the arena is read by device address, not this slot.
        bufs.extend_from_slice(&[
            Self::vkb(qa),
            Self::vkb(counts),
            Self::vkb(offsets),
            Self::vkb(c),
        ]);
        self.dispatch3(kern, &bufs, 1, &push, gx, n_expert as u32, 1);
    }

    /// [`Self::matmul_mmq_experts`]'s paged twin: `arena` is a `GpuPager` arena (uniform slots)
    /// and `lut` a buffer of resident SLOT INDICES; the kernel resolves each bucket's arena
    /// address as `arena_base + uint64_t(lut[lut_base + expert]) * slot_bytes` instead of
    /// `w_base + expert·stride` (the shaders' `-DPAGED` build — slot indirection through
    /// `w_addr`/`arena_word()` at the `rb()` chokepoint, replacing the old u32-element-space
    /// multiply that overflowed past ~102 Scout-sized slots). `lut`/`lut_base` name a run of
    /// `n_expert` slot indices for exactly this layer — the session's write-once LUT tape window
    /// (`MoePagerSession::lut_window`), frozen at
    /// record time so later staging can't mutate what an in-flight segment reads. The caller must
    /// have staged every ROUTED expert of this layer resident (all of them simultaneously —
    /// buckets run in one dispatch) before recording this; buckets with count 0 exit before any
    /// LUT/weight read, so unrouted experts may be absent (their window entries are whatever the
    /// mirror held — never dereferenced). Covers every dtype `paged_mmq_ok` accepts — the FULL
    /// `MOE_MMQ_DTYPES` set, since
    /// `MOE_MMQ_PAGED_DTYPES` now mirrors it (see adapter.rs's drift-guard doc); `sact` is `Some`
    /// only for the min-carrying dtypes (Q4_K/Q5_K/Q5_1/Q4_1 — the rest are
    /// symmetric-or-self-summed, same `MOE_MMQ_SACT_DTYPES` split the resident twin uses). Tile
    /// selection (BM=64 vs 32) matches the resident twin.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mmq_experts_paged(
        &self,
        dtype: infr_core::DType,
        stage: &'static str,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: Option<&dyn Buffer>,
        arena_addr: u64,
        slot_bytes: u32,
        lut: &dyn Buffer,
        lut_base: usize,
        counts: &dyn Buffer,
        offsets: &dyn Buffer,
        c: &dyn Buffer,
        rows: usize,
        k: usize,
        n: usize,
        n_expert: usize,
        n_used: usize,
    ) {
        self.label_next(stage);
        let avg_rows = rows.saturating_mul(n_used) / n_expert.max(1);
        let small_tile = avg_rows <= MOE_EXPERT_SMALL_TILE_AVG_ROWS;
        let bm: usize = if small_tile { 32 } else { 64 };
        // Kernel selection from the SAME dtype→kernel table as the resident twin (see
        // [`moe_mmq_desc`]): paged uses the `_xpg`/`_xpg32` variants and one MORE binding than
        // resident (the LUT slot — `nb = resident_nb + 1`). Min-carrying dtypes (Q4_K/Q5_K/Q5_1/
        // Q4_1) thus land at 8 bindings, the rest at 7; byte-identical to the old inline match.
        let desc = moe_mmq_desc(dtype).unwrap_or_else(|| {
            unreachable!(
                "paged batched MoE expert GEMM: the MOE_MMQ_PAGED_DTYPES set only \
                 (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q2_0/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/IQ2_S/\
                 IQ3_S/MXFP4/NVFP4)"
            )
        });
        let (name, spv) = if small_tile { desc.xpg32 } else { desc.xpg };
        let nb = desc.resident_nb + 1;
        let kern = self.be.kernel(name, spv, nb, 28);
        let mut push = [0u8; 28];
        // pc.m unused in the PAGED build (slot bases come from the LUT); pc.w_base carries the
        // layer's LUT window base.
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(lut_base as u32).to_ne_bytes());
        // Arena base address (lo/hi) + per-slot byte stride: the `-DPAGED` mmq shader reads the
        // arena by device address (native_weight_addr.glsl), not through the old WB binding.
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&slot_bytes.to_ne_bytes());
        let gx = (rows.div_ceil(bm) * (n / 64)) as u32;
        let mut bufs = vec![Self::vkb(qa), Self::vkb(dact)];
        if let Some(sa) = sact {
            bufs.push(Self::vkb(sa));
        }
        // The old arena WB binding is now unread by the shader (arena is device-address addressed);
        // its descriptor slot takes a harmless filler (`lut` — a valid, small, read-only buffer),
        // keeping the binding count and every subsequent binding number unchanged.
        bufs.extend_from_slice(&[
            Self::vkb(lut),
            Self::vkb(lut),
            Self::vkb(counts),
            Self::vkb(offsets),
            Self::vkb(c),
        ]);
        self.dispatch3(kern, &bufs, 1, &push, gx, n_expert as u32, 1);
    }

    /// Batched-MoE epilogue: `dst[t] = Σ_s bucket_wts[p]·y_all[p]` over the token's `n_used`
    /// assignments (p = inv_pos[t·n_used+s]) — fixed slot order, deterministic, no atomics, and
    /// dst is written directly (no zero + per-expert accumulate passes). `prescaled` (llama4's
    /// weight-before-FFN): `bucket_wts` was already folded into the packed activations upstream
    /// (`moe_weight_scale`) — sum `y_all` unweighted instead of multiplying by `bucket_wts` again.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_scatter_reduce(
        &self,
        y_all: &dyn Buffer,
        bucket_wts: &dyn Buffer,
        inv_pos: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        ne: usize,
        n_used: usize,
        prescaled: bool,
    ) {
        let k = self.be.kernel(
            "moe_scatter_reduce",
            crate::gemm::moe_scatter_reduce_spv(),
            4,
            16,
        );
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(prescaled as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(y_all),
                Self::vkb(bucket_wts),
                Self::vkb(inv_pos),
                Self::vkb(dst),
            ],
            1,
            &push,
            ((rows * ne) as u32).div_ceil(64),
        );
    }

    /// Id-indexed native-block GEMV `y = x · W[ids[slot]]ᵀ` from a stacked expert tensor (element
    /// stride per expert). Lets GPU-resident MoE decode pick the expert from a GPU buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        ids: &dyn Buffer,
        slot: usize,
        stride: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w
            .device_addr()
            .expect("resident-BDA weight: linear_native_id requires a u64 BDA device address");
        let stride_bytes = expert_stride_bytes(dtype, stride);
        self.linear_native_id_at(
            dtype,
            arena_addr,
            stride_bytes,
            ids,
            slot,
            x,
            y,
            rows,
            in_f,
            out_f,
        );
    }

    /// Multi-slot id GEMV: all `n_used` experts in ONE dispatch → `y` is [n_used, out_f]. The experts
    /// run concurrently (no inter-expert barrier). `x_per_slot`: false → all slots read the same row
    /// `x` (gate/up); true → slot reads `x[slot*in_f..]` (down). Decode FFN fusion.
    ///
    /// `rows`: widens the SAME dispatch to `rows` independent tokens — `ids`/`wts`/`y` become
    /// `[rows, n_used, ...]` flat (the shader splits its flat `slot_global` index back into
    /// `row`/`slot` using `n_used`, so no push-constant change is needed here). `rows == 1` is the
    /// original decode call. This is the MoE small-m fast path (see `Op::MoeFfn` in adapter.rs):
    /// for a handful of prefill tokens it dispatches per-ACTIVE-expert GEMVs over only the routed
    /// rows, instead of the batched path's whole-expert-bank streaming GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id_multi(
        &self,
        dtype: infr_core::DType,
        w: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        stride: usize,
        x: &dyn Buffer,
        x_per_slot: bool,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
        rows: usize,
    ) {
        let arena_addr = w.device_addr().expect(
            "resident-BDA weight: linear_native_id_multi requires a u64 BDA device address",
        );
        let stride_bytes = expert_stride_bytes(dtype, stride);
        self.linear_native_id_multi_at(
            dtype,
            arena_addr,
            stride_bytes,
            ids,
            n_used,
            x,
            x_per_slot,
            y,
            in_f,
            out_f,
            rows,
        );
    }

    /// `-DSTREAMED` twin of [`Self::linear_native_id`] (slice A4 build-variant: see
    /// `crate::gemm::native_id_build_spv`). The stacked expert tensor lives at
    /// `arena_addr` in a resident BDA arena; `stride_bytes` is the per-expert BYTE stride the
    /// shader applies on the 64-bit pointer (`w_addr = arena + u64(ids[slot]) * u64(stride_bytes)`)
    /// — the u32 element-space `ids*stride` multiply this replaces wraps past 2^32 elements.
    /// Binding 0 (the resident weight SSBO) takes a small FILLER (`x`). [`Self::linear_native_id`]
    /// forks here when its weight buffer is arena-allocated (`w.device_addr()`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        stride_bytes: u32,
        ids: &dyn Buffer,
        slot: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_native_id_at", in_f);
        self.label_gemv("gemv_id_streamed", rows, in_f, out_f);
        let (name, spv) = crate::gemm::native_id_build_spv(dtype).expect("native id streamed spv");
        let k = self.be.kernel(name, spv, 4, 28);
        let mut push = [0u8; 28];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(slot as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&stride_bytes.to_ne_bytes());
        push[20..24].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[Self::vkb(x), Self::vkb(x), Self::vkb(ids), Self::vkb(y)],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// `-DSTREAMED` twin of [`Self::linear_native_id_multi`] (slice A4 build-variant: see
    /// `crate::gemm::native_idm_build_spv`). Same BYTE-stride-on-the-pointer convention
    /// as [`Self::linear_native_id_at`]. Mirrors the resident dispatcher's SG routing (m=1
    /// decode, Q6_K/Q5_K/IQ3_S projection band) so a parity test comparing the two at the same
    /// shape lands on the same kernel. [`Self::linear_native_id_multi`] forks here when its weight
    /// buffer is arena-allocated (`w.device_addr()`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id_multi_at(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        stride_bytes: u32,
        ids: &dyn Buffer,
        n_used: usize,
        x: &dyn Buffer,
        x_per_slot: bool,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
        rows: usize,
    ) {
        assert_native_k("linear_native_id_multi_at", in_f);
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&stride_bytes.to_ne_bytes());
        push[16..20].copy_from_slice(&(x_per_slot as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        let bufs = [Self::vkb(x), Self::vkb(x), Self::vkb(ids), Self::vkb(y)];
        if rows == 1 {
            if let Some(nr) = native_id_sg_choice(dtype, in_f, out_f, self.gemv()) {
                if let Some((name, spv)) =
                    crate::gemm::native_idm_sg_build_spv(dtype, nr, self.sg16())
                {
                    let k = self.be.kernel_sg(name, spv, 4, 32, self.sgp());
                    let groups = (n_used * out_f.div_ceil(nr as usize)) as u32;
                    self.dispatch_wide(k, &bufs, 1, &push, groups);
                    return;
                }
            }
        }
        let (name, spv) =
            crate::gemm::native_idm_build_spv(dtype).expect("native idm streamed spv");
        let k = self.be.kernel(name, spv, 4, 32);
        self.dispatch_wide(k, &bufs, 1, &push, (rows * n_used * out_f) as u32);
    }

    /// [`linear_native_id`]'s paged twin: `w` is a `GpuPager` arena (fixed uniform slots, not one
    /// contiguous per-expert tensor) and `lut` a run of per-expert resident SLOT INDICES for
    /// exactly this layer — the kernel sets `w_addr = arena_base + uint64_t(lut[lut_base +
    /// ids[slot_idx]]) * slot_bytes` with the layer-LOCAL expert ids moe_topk produced (no host
    /// global-id rewrite; see `shaders/native_gemv_id.comp`'s `-DPAGED` branch — `lut_base` rides
    /// the otherwise-unused `stride` push slot). `lut` is the session's write-once LUT tape window in the inline paths
    /// (see `MoePagerSession::lut_window`). Always the tree kernel — the SG variant has no paged
    /// build (Q6_K/Q5_K only, gated off; see `native_id_sg_choice`'s doc — a paged model never
    /// reaches that gate today since MoE auto-fit picks the pager only on overflow, but this
    /// keeps the two builds from silently diverging if that changes).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id_paged(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        slot_bytes: u32,
        lut: &dyn Buffer,
        ids: &dyn Buffer,
        slot: usize,
        lut_base: usize,
        x: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_native_id_paged", in_f);
        self.label_gemv("gemv_id_paged", rows, in_f, out_f);
        let name =
            crate::linear::native_id_paged_kernel_name(dtype).expect("native id paged kernel");
        let spv = crate::gemm::native_id_paged_build_spv(dtype).expect("native id paged spv");
        let k = self.be.kernel(name, spv, 5, 32);
        let mut push = [0u8; 32];
        push[0..4].copy_from_slice(&(rows as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(slot as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(lut_base as u32).to_ne_bytes());
        // Arena base address (lo/hi) + per-slot byte stride: the `-DPAGED` shader reads the arena
        // by device address, not through binding 0 (see native_weight_addr.glsl).
        push[20..24].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[24..28].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&slot_bytes.to_ne_bytes());
        // Binding order MUST match the shader's `-DPAGED` layout (`native_gemv_id.comp`) AND keep
        // `y` last: `dispatch3`'s hazard tracking treats the tail `n_out` buffers as writes and
        // everything before as reads (one contiguous split) — `y` bound before `lut` would get
        // mis-tracked as a read, silently dropping the barrier a later dispatch needs. Binding 0
        // (the old arena SSBO) is now unread by the shader; it takes a harmless filler (`lut`
        // again — a valid, small, read-only buffer) so the descriptor set stays fully written.
        self.dispatch_wide(
            k,
            &[
                Self::vkb(lut),
                Self::vkb(x),
                Self::vkb(ids),
                Self::vkb(lut),
                Self::vkb(y),
            ],
            1,
            &push,
            (rows * out_f) as u32,
        );
    }

    /// [`linear_native_id_multi`]'s paged twin — same local-ids + LUT-window hop as
    /// [`linear_native_id_paged`] (`w_addr = arena_base + uint64_t(lut[lut_base + ids[slot]]) *
    /// slot_bytes`), for the decode/small-m all-`n_used`-experts-in-one-dispatch path. Always the
    /// tree kernel (see
    /// [`linear_native_id_paged`]'s doc for why the SG fast path is skipped).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_native_id_multi_paged(
        &self,
        dtype: infr_core::DType,
        arena_addr: u64,
        slot_bytes: u32,
        lut: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        lut_base: usize,
        x: &dyn Buffer,
        x_per_slot: bool,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
        rows: usize,
    ) {
        assert_native_k("linear_native_id_multi_paged", in_f);
        let mut push = [0u8; 36];
        push[0..4].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&(lut_base as u32).to_ne_bytes());
        push[16..20].copy_from_slice(&(x_per_slot as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&(rows as u32).to_ne_bytes());
        // Arena base address (lo/hi) + per-slot byte stride — see `linear_native_id_paged`.
        push[24..28].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[28..32].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        push[32..36].copy_from_slice(&slot_bytes.to_ne_bytes());
        // `y` last — see `linear_native_id_paged`'s doc on why binding order matters for
        // `dispatch3`'s hazard tracking. Binding 0 (old arena SSBO) is unread; `lut` fills it.
        let bufs = [
            Self::vkb(lut),
            Self::vkb(x),
            Self::vkb(ids),
            Self::vkb(lut),
            Self::vkb(y),
        ];
        let name =
            crate::linear::native_idm_paged_kernel_name(dtype).expect("native idm paged kernel");
        let spv = crate::gemm::native_idm_paged_build_spv(dtype).expect("native idm paged spv");
        let k = self.be.kernel(name, spv, 5, 36);
        self.dispatch_wide(k, &bufs, 1, &push, (rows * n_used * out_f) as u32);
    }

    /// Quantize f32 activations `a` [m,k] → int8 `qa` [m,k] + per-32-block f16 `dact`/`sact`
    /// ([m, k/32]) for the dp4a mmq matmul. (Pass 1 of mmq, reusable standalone.)
    pub fn quant_q8(
        &self,
        a: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        m: usize,
        k: usize,
    ) {
        let kq = self
            .be
            .kernel_sg("quant_q8", crate::gemm::quant_q8_spv(), 4, 12, 32);
        let mut p = [0u8; 12];
        p[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        p[4..8].copy_from_slice(&(k as u32).to_ne_bytes());
        p[8..12].copy_from_slice(&32u32.to_ne_bytes());
        self.dispatch(
            kq,
            &[
                Self::vkb(a),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
            ],
            3,
            &p,
            (m * (k / 32)) as u32,
        );
    }

    /// Multi-slot id-indexed Q4_K dp4a (mmq) GEMV: like `linear_native_id_multi` but using hardware
    /// int8 dot-product against pre-quantized activations (`qa`/`dact`/`sact` from `quant_q8`, shared
    /// across slots). Q4_K weights only. `y` is [n_used, out_f].
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_id_multi_q4k(
        &self,
        w: &dyn Buffer,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        stride: usize,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        let arena_addr = w.device_addr().expect(
            "resident-BDA weight: linear_mmv_id_multi_q4k requires a u64 BDA device address",
        );
        let stride_bytes = expert_stride_bytes(infr_core::DType::Q4K, stride);
        self.linear_mmv_id_multi_q4k_at(
            arena_addr,
            stride_bytes,
            qa,
            dact,
            sact,
            ids,
            n_used,
            y,
            in_f,
            out_f,
        );
    }

    /// `-DSTREAMED` twin of [`Self::linear_mmv_id_multi_q4k`] (slice A4 build-variant: see
    /// `crate::gemm::native_mmv_id_q4k_spv`). Same BYTE-stride-on-the-pointer convention
    /// as [`Self::linear_native_id_at`]; binding 0 takes a small FILLER (`qa`).
    /// [`Self::linear_mmv_id_multi_q4k`] forks here when its weight buffer is arena-allocated
    /// (`w.device_addr()`).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mmv_id_multi_q4k_at(
        &self,
        arena_addr: u64,
        stride_bytes: u32,
        qa: &dyn Buffer,
        dact: &dyn Buffer,
        sact: &dyn Buffer,
        ids: &dyn Buffer,
        n_used: usize,
        y: &dyn Buffer,
        in_f: usize,
        out_f: usize,
    ) {
        assert_native_k("linear_mmv_id_multi_q4k_at", in_f);
        let k = self.be.kernel(
            "native_mmv_id_q4k",
            crate::gemm::native_mmv_id_q4k_spv(),
            6,
            24,
        );
        let mut push = [0u8; 24];
        push[0..4].copy_from_slice(&(in_f as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(out_f as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[12..16].copy_from_slice(&stride_bytes.to_ne_bytes());
        push[16..20].copy_from_slice(&(arena_addr as u32).to_ne_bytes());
        push[20..24].copy_from_slice(&((arena_addr >> 32) as u32).to_ne_bytes());
        self.dispatch_wide(
            k,
            &[
                Self::vkb(qa),
                Self::vkb(qa),
                Self::vkb(dact),
                Self::vkb(sact),
                Self::vkb(ids),
                Self::vkb(y),
            ],
            1,
            &push,
            (n_used * out_f) as u32,
        );
    }

    /// Weighted accumulate of all selected experts' down outputs into hidden:
    /// `hidden[row*ne+i] += Σ_slot wts[row*n_used+slot] * down[(row*n_used+slot)*ne + i]`. Folds the
    /// per-expert axpys into one op. `rows` widens this to independent tokens via grid.y (the MoE
    /// small-m fast path — see `Op::MoeFfn` in adapter.rs); `rows == 1` is the original decode call.
    /// `prescaled` (llama4's weight-before-FFN): `wts` was already folded into the gate/up
    /// activations upstream (`moe_weight_scale`) — sum `down` unweighted instead of multiplying by
    /// `wts` again.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_accumulate(
        &self,
        down: &dyn Buffer,
        wts: &dyn Buffer,
        hidden: &dyn Buffer,
        ne: usize,
        n_used: usize,
        rows: usize,
        prescaled: bool,
    ) {
        let k = self
            .be
            .kernel("moe_accumulate", crate::gemm::moe_accumulate_spv(), 3, 12);
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(prescaled as u32).to_ne_bytes());
        self.dispatch3(
            k,
            &[Self::vkb(down), Self::vkb(wts), Self::vkb(hidden)],
            1,
            &push,
            (ne as u32).div_ceil(64),
            rows as u32,
            1,
        );
    }

    /// `acc += wts[slot] * x` (indexed axpy) — the scale is read from a GPU buffer (the on-GPU router
    /// weights), so the weighted MoE expert accumulate needs no host scale.
    pub fn add_scaled_id(
        &self,
        x: &dyn Buffer,
        wts: &dyn Buffer,
        slot: usize,
        acc: &dyn Buffer,
        n: usize,
    ) {
        let k = self
            .be
            .kernel("add_scaled_id", crate::gemm::add_scaled_id_spv(), 3, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(slot as u32).to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(wts), Self::vkb(acc)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// `acc += scale * x` (axpy), in place into `acc`. Accumulates weighted MoE expert outputs into
    /// the resident hidden state on the GPU (chained across experts via WAW barriers on `acc`).
    /// In-place elementwise scalar multiply: `y[i] *= scale` for `i < n` (gemma4 layer output scale).
    pub fn scale(&self, y: &dyn Buffer, scale: f32, n: usize) {
        let k = self.be.kernel("scale", crate::gemm::scale_spv(), 1, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(k, &[Self::vkb(y)], 1, &push, (n as u32).div_ceil(64));
    }

    /// Elementwise softcap `y[i] = cap·tanh(x[i]/cap)` (gemma final-logit / attn softcap). In-place
    /// safe — bind `x == y`.
    pub fn softcap(&self, x: &dyn Buffer, y: &dyn Buffer, cap: f32, n: usize) {
        let k = self.be.kernel("softcap", crate::gemm::softcap_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&cap.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(y)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    pub fn add_scaled(&self, x: &dyn Buffer, acc: &dyn Buffer, scale: f32, n: usize) {
        let k = self
            .be
            .kernel("add_scaled", crate::gemm::add_scaled_spv(), 2, 8);
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(n as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&scale.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(acc)],
            1,
            &push,
            (n as u32).div_ceil(64),
        );
    }

    /// Elementwise add; in place allowed (`a` may equal `y`).
    pub fn add(&self, a: &dyn Buffer, b: &dyn Buffer, y: &dyn Buffer, n: usize) {
        let k = self.be.kernel("add", crate::gemm::add_spv(), 3, 4);
        self.dispatch(
            k,
            &[Self::vkb(a), Self::vkb(b), Self::vkb(y)],
            1,
            &(n as u32).to_ne_bytes(),
            (n as u32).div_ceil(64),
        );
    }

    /// Broadcast bias add: `y[i] = x[i] + bias[i % n]` over `rows*n` elements (Qwen2 q/k/v `Wx+b`).
    pub fn add_bias(
        &self,
        x: &dyn Buffer,
        bias: &dyn Buffer,
        y: &dyn Buffer,
        rows: usize,
        n: usize,
    ) {
        let total = (rows * n) as u32;
        let k = self
            .be
            .kernel("add_bias", crate::gemm::add_bias_spv(), 3, 8);
        let mut pc = (n as u32).to_ne_bytes().to_vec();
        pc.extend_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(bias), Self::vkb(y)],
            1,
            &pc,
            total.div_ceil(64),
        );
    }

    /// Broadcast multiply: `dst[i] = x[i] * vec[i % n]` over `total = rows*n` elements
    /// (diffusion-gemma's router input scale — the multiplicative twin of `add_bias`).
    pub fn mul_vec(&self, x: &dyn Buffer, vec: &dyn Buffer, y: &dyn Buffer, rows: usize, n: usize) {
        let total = (rows * n) as u32;
        let k = self.be.kernel("mul_vec", crate::gemm::mul_vec_spv(), 3, 8);
        let mut pc = (n as u32).to_ne_bytes().to_vec();
        pc.extend_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[Self::vkb(x), Self::vkb(vec), Self::vkb(y)],
            1,
            &pc,
            total.div_ceil(64),
        );
    }

    /// qwen35moe shared-expert combine: `dst[r,c] = moe[r,c] + sigmoid(gate[r]) * shexp[r,c]` over
    /// `total = rows*n` elements — `gate` is a length-`rows` buffer (one raw pre-sigmoid logit per
    /// row), broadcast across each row's `n` columns (the row-broadcast twin of `mul_vec`'s
    /// column broadcast). See `Op::MoeSharedExpertAdd`'s doc.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_shared_expert_add(
        &self,
        moe: &dyn Buffer,
        shexp: &dyn Buffer,
        gate: &dyn Buffer,
        dst: &dyn Buffer,
        rows: usize,
        n: usize,
    ) {
        let total = (rows * n) as u32;
        let k = self.be.kernel(
            "moe_shared_expert_add",
            crate::gemm::moe_shared_expert_add_spv(),
            4,
            8,
        );
        let mut pc = (n as u32).to_ne_bytes().to_vec();
        pc.extend_from_slice(&total.to_ne_bytes());
        self.dispatch(
            k,
            &[
                Self::vkb(moe),
                Self::vkb(shexp),
                Self::vkb(gate),
                Self::vkb(dst),
            ],
            1,
            &pc,
            total.div_ceil(64),
        );
    }

    /// Like [`Self::moe_accumulate`], but scales each selected expert's down output by a per-expert
    /// weight BEFORE the weighted sum: `hidden[row*ne+i] += sum_slot wts[row*n_used+slot] *
    /// dscale[ids[row*n_used+slot]] * down[(row*n_used+slot)*ne+i]` (diffusion-gemma
    /// `ffn_down_exps.scale`). `ids` is the same expert-id buffer `moe_topk` filled. `rows` widens
    /// this to independent tokens via grid.y (the MoE small-m fast path); `rows == 1` is the
    /// original per-token call.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_accumulate_scaled(
        &self,
        down: &dyn Buffer,
        wts: &dyn Buffer,
        ids: &dyn Buffer,
        dscale: &dyn Buffer,
        hidden: &dyn Buffer,
        ne: usize,
        n_used: usize,
        rows: usize,
    ) {
        let k = self.be.kernel(
            "moe_accumulate_scaled",
            crate::gemm::moe_accumulate_scaled_spv(),
            5,
            8,
        );
        let mut push = [0u8; 8];
        push[0..4].copy_from_slice(&(ne as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n_used as u32).to_ne_bytes());
        self.dispatch3(
            k,
            &[
                Self::vkb(down),
                Self::vkb(wts),
                Self::vkb(ids),
                Self::vkb(dscale),
                Self::vkb(hidden),
            ],
            1,
            &push,
            (ne as u32).div_ceil(64),
            rows as u32,
            1,
        );
    }

    /// Free the recorder's transient Vulkan objects — the query pool (if any), the command buffer,
    /// and every descriptor-pool tranche. `Recorder` has no `Drop` (by design — the arena pools
    /// would otherwise be reported live at `vkDestroyDevice`), so `finish`/`finish_nowait` must
    /// call this on BOTH their success AND their `?`-error exits; a bare early return would leak
    /// these — and it leaks precisely on device-lost, exactly when those live pools then trip
    /// `vkDestroyDevice`. Call once per recorder. Mirrors `discard`'s cleanup block.
    fn free_transient(&self) {
        let device = &self.be.shared.device;
        unsafe {
            // query_pool is null unless prof_ops created one; the null check is equivalent to `prof_ops`.
            if self.query_pool != vk::QueryPool::null() {
                device.destroy_query_pool(self.query_pool, None);
            }
            let cmd_pool = *self.be.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            for p in self.pools.borrow().iter() {
                device.destroy_descriptor_pool(*p, None);
            }
        }
    }

    /// End recording, submit once, wait, and release transient objects.
    pub fn finish(self) -> Result<()> {
        let device = &self.be.shared.device;
        let dispatches = self.dispatches.get();
        let t_record = self.t0.elapsed();
        if self.stages {
            eprintln!("[prof] barriers emitted = {}", self.barriers.borrow());
        }
        // Final timestamp so the last op has an interval to close.
        if self.prof_ops {
            let idx = self.ts_labels.borrow().len() as u32;
            unsafe {
                device.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    idx,
                );
            }
        }
        let queue = self.be.shared.queue;
        // Each fallible step frees the transient objects before propagating — see `free_transient`.
        if let Err(e) = unsafe { device.end_command_buffer(self.cmd) } {
            self.free_transient();
            return Err(be(format!("end cmd: {e}")));
        }
        let prof = pager_profile::active();
        let submit_t0 = prof.then(std::time::Instant::now);
        let submit = unsafe {
            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd))],
                vk::Fence::null(),
            )
        };
        if let Some(t0) = submit_t0 {
            pager_profile::record_queue_submit(dispatches, t0.elapsed());
        }
        if let Err(e) = submit {
            self.free_transient();
            return Err(be(format!("queue_submit: {e}")));
        }
        let wait_t0 = prof.then(std::time::Instant::now);
        let wait = unsafe { device.queue_wait_idle(queue) };
        if let Some(t0) = wait_t0 {
            pager_profile::record_sync_wait(pager_profile::SyncKind::QueueIdle, t0.elapsed());
        }
        if let Err(e) = wait {
            self.free_transient();
            return Err(be(format!("queue_wait_idle: {e}")));
        }
        if self.stages {
            eprintln!(
                "[prof] record={:.1}ms submit+gpu={:.1}ms",
                t_record.as_secs_f64() * 1e3,
                (self.t0.elapsed() - t_record).as_secs_f64() * 1e3,
            );
        }
        // Report BEFORE freeing (report_timestamps reads the query pool); `free_transient` then
        // destroys it. Order matches the old success path.
        if self.prof_ops {
            self.report_timestamps();
        }
        self.free_transient();
        Ok(())
    }

    /// Abandon a partially recorded command buffer WITHOUT submitting it, releasing exactly what
    /// [`Self::finish`] releases (command buffer, descriptor pools, query pool) — the shutdown path
    /// (`adapter::execute_static`'s abort on the `infr_core::shutdown` latch).
    ///
    /// The point is that the work recorded so far is WASTE: the forward it belongs to is being
    /// abandoned, so nothing will ever read what those dispatches would write. Submitting it anyway
    /// (the obvious `finish()`) would add a whole segment of GPU time to how long Ctrl-C takes —
    /// and would hand the driver more work at the exact moment we are trying to stop giving it any.
    /// Dropping the recorder instead is not an option: it has no `Drop`, so its descriptor pools
    /// would outlive it and `vkDestroyDevice` would (rightly) be reported as destroying a device
    /// with live pools.
    ///
    /// Only the NOT-yet-submitted buffer is abandoned. Anything already on the queue is still
    /// drained by the caller — a submitted command buffer cannot be cancelled, only waited on.
    pub fn discard(self) -> Result<()> {
        let device = &self.be.shared.device;
        unsafe {
            device
                .end_command_buffer(self.cmd)
                .map_err(|e| be(format!("end cmd: {e}")))?;
            if self.prof_ops {
                device.destroy_query_pool(self.query_pool, None);
            }
            let cmd_pool = *self.be.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            for p in self.pools.borrow().iter() {
                device.destroy_descriptor_pool(*p, None);
            }
        }
        Ok(())
    }

    /// [`Self::finish`]'s pipelined twin: end recording and submit WITHOUT waiting, returning a
    /// [`PendingSegment`] whose [`PendingSegment::wait`] blocks on a fence and releases the
    /// command buffer / descriptor pools. The paged-MoE executor uses this to keep staging the
    /// next segment's expert uploads on the CPU while the GPU chews the one just submitted
    /// (`adapter::execute_paged_moe`'s ring rotation). Profiling recorders (INFR_PROF /
    /// INFR_PROF_OPS) degrade to the blocking [`Self::finish`] — their reports need completed
    /// timestamps — and hand back an already-drained segment, so callers need no special case.
    ///
    /// The NEXT recorder on this queue must open with [`Self::seed_barrier`]: hazard tracking is
    /// per-recorder, so cross-submission RAW/WAR ordering (scratch reuse, arena-slot rewrites vs
    /// a prior segment's reads) is otherwise invisible to it.
    pub fn finish_nowait(self) -> Result<PendingSegment> {
        let dispatches = self.dispatches.get();
        if self.stages || self.prof_ops {
            let shared = std::sync::Arc::clone(&self.be.shared);
            self.finish()?;
            return Ok(PendingSegment {
                shared,
                fence: None,
                cmd: vk::CommandBuffer::null(),
                pools: Vec::new(),
                dispatches,
            });
        }
        let device = &self.be.shared.device;
        // On SUCCESS the cmd buffer + pools are handed to the returned `PendingSegment` (freed in
        // its `wait`); on every ERROR exit we own them still and must free them here (`Recorder`
        // has no `Drop`) — see `free_transient`.
        if let Err(e) = unsafe { device.end_command_buffer(self.cmd) } {
            self.free_transient();
            return Err(be(format!("end cmd: {e}")));
        }
        let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
            Ok(f) => f,
            Err(e) => {
                self.free_transient();
                return Err(be(format!("create fence: {e}")));
            }
        };
        let prof = pager_profile::active();
        let submit_t0 = prof.then(std::time::Instant::now);
        let submit = unsafe {
            device.queue_submit(
                self.be.shared.queue,
                &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd))],
                fence,
            )
        };
        if let Some(t0) = submit_t0 {
            pager_profile::record_queue_submit(dispatches, t0.elapsed());
        }
        if let Err(e) = submit {
            unsafe { device.destroy_fence(fence, None) };
            self.free_transient();
            return Err(be(format!("queue_submit: {e}")));
        }
        Ok(PendingSegment {
            shared: std::sync::Arc::clone(&self.be.shared),
            fence: Some(fence),
            cmd: self.cmd,
            pools: self.pools.borrow().clone(),
            dispatches,
        })
    }

    /// Unconditional global compute+transfer barrier — the mandatory FIRST command of a recorder
    /// that continues a [`Self::finish_nowait`]-submitted stream. Hazard tracking starts empty per
    /// recorder, so the ordering between this segment's dispatches/copies and the in-flight
    /// previous segment's (pooled scratch reuse, pager arena-slot rewrites after reads) must be
    /// seeded explicitly; `vkCmdPipelineBarrier`'s scope spans submission order on the queue, so
    /// this one barrier orders the whole segment after everything already submitted.
    pub fn seed_barrier(&self) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            self.be.shared.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            );
        }
        *self.barriers.borrow_mut() += 1;
    }

    /// End recording WITHOUT submitting, returning ONE resubmittable [`RecordedSegment`] — the
    /// building block of a [`RecordedCmd`]. Only meaningful for a `new_persistent` recorder; the
    /// descriptor sets bind the (persistent) decode buffers, so replays stay valid as long as those
    /// buffers live.
    ///
    /// A decode recording that must be watchdog-split records several segments back to back
    /// (`adapter::record_decode_replay`): each continuation recorder opens with [`Self::seed_barrier`]
    /// and its own segment is a SEPARATE submit at replay time, so no single submit can trip the
    /// per-submit GPU hang watchdog. A discrete GPU records the whole decode into one segment.
    pub(crate) fn end_segment(self) -> Result<RecordedSegment> {
        if self.stages {
            eprintln!("[prof] barriers emitted = {}", self.barriers.borrow());
        }
        unsafe { self.be.shared.device.end_command_buffer(self.cmd) }
            .map_err(|e| be(format!("end cmd: {e}")))?;
        Ok(RecordedSegment {
            cmd: self.cmd,
            pools: self.pools.borrow().clone(),
            dispatches: self.dispatches.get(),
        })
    }

    /// End recording WITHOUT submitting, returning a single-segment [`RecordedCmd`] the caller can
    /// replay across tokens (skipping per-token re-recording). The one-segment shape is the tuned
    /// fast path taken by every discrete GPU; a device that watchdog-splits its decode builds a
    /// multi-segment [`RecordedCmd`] from [`Self::end_segment`] instead.
    pub fn finish_record(self) -> Result<RecordedCmd> {
        let shared = std::sync::Arc::clone(&self.be.shared);
        let seg = self.end_segment()?;
        Ok(RecordedCmd::from_shared_segments(shared, vec![seg]))
    }

    /// Read back per-dispatch timestamps and print GPU time aggregated by label (the kernel
    /// name, unless overridden via [`label_next`](Self::label_next) or the shape itemizers).
    /// Each label's total also feeds the process-wide aggregate (`infr_prof_rt::gpu_add`),
    /// printed once at exit — and merged with the host-side function table when the binary was
    /// ALSO built with `INFR_PROFILE=1`.
    fn report_timestamps(&self) {
        let labels = self.ts_labels.borrow();
        let n = labels.len();
        if n == 0 {
            return;
        }
        if self.ts_dropped.get() > 0 {
            eprintln!(
                "[prof_ops] WARNING: {} dispatches past the {}-timestamp query pool went unstamped",
                self.ts_dropped.get(),
                Self::MAX_TS,
            );
        }
        let mut ticks = vec![0u64; n + 1];
        unsafe {
            self.be
                .shared
                .device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .expect("get_query_pool_results");
        }
        let period = unsafe {
            self.be
                .shared
                .instance
                .get_physical_device_properties(self.be.shared.physical_device)
        }
        .limits
        .timestamp_period; // ns per tick
                           // Accounted, aggregated and printed by the shared reporter (`infr_core::prof::OpProf`),
                           // which is where the `gpu_add` fold into the process-wide exit table now happens too — the
                           // same code path metal and the cpu backend report through, so the three tables are
                           // column-for-column comparable.
        let mut p = infr_core::prof::OpProf::new("vulkan", infr_core::prof::Unit::Device);
        for i in 0..n {
            let us = (ticks[i + 1].wrapping_sub(ticks[i]) as f64) * period as f64 / 1000.0;
            p.add(labels[i], us);
        }
        p.flush();
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// Start recording a single-submit forward.
    pub fn recorder(&self) -> Result<Recorder<'_>> {
        Recorder::new(self)
    }

    /// Start recording a resubmittable forward — `finish_record` returns a [`RecordedCmd`] to replay.
    pub fn recorder_persistent(&self) -> Result<Recorder<'_>> {
        Recorder::new_persistent(self)
    }
}

/// One resubmittable command buffer of a [`RecordedCmd`] (from [`Recorder::end_segment`]). Owns
/// its command buffer + descriptor pools (whose sets bind the persistent decode buffers); a
/// `RecordedCmd` frees them all on drop after the GPU drains.
pub(crate) struct RecordedSegment {
    cmd: vk::CommandBuffer,
    pools: Vec<vk::DescriptorPool>,
    /// Dispatches recorded into this one segment.
    dispatches: usize,
}

/// A pre-recorded, resubmittable decode step (from [`Recorder::finish_record`], or accumulated from
/// [`Recorder::end_segment`] on a splitting device). Replaying it skips per-token re-recording in
/// the GPU-resident decode loop.
///
/// The step is one or more [`RecordedSegment`]s. A discrete GPU (no submit cap) records the whole
/// decode into ONE segment and replays it in a single submit — the tuned fast path, unchanged. A
/// device that must split its forward against the per-submit GPU hang watchdog records the decode
/// across several watchdog-sized segments (`adapter::record_decode_replay`), and [`Self::replay`]
/// submits each SEPARATELY so no single submit is ever the whole (potentially >2 s) decode. The
/// same `_dyn`/params/ring decode semantics are preserved either way — splitting only distributes
/// the identical dispatch stream across command buffers, with a seeded global barrier at each
/// segment head carrying cross-segment ordering.
pub struct RecordedCmd {
    shared: std::sync::Arc<crate::VulkanShared>,
    /// One (discrete) or several (watchdog-split) resubmittable segments, in execution order.
    segments: Vec<RecordedSegment>,
    /// Dispatches in ONE replay of the WHOLE recording (summed across segments). `replay_n(n)` puts
    /// `n` copies of a single-segment recording into one submit, so this is what bounds a chain
    /// against the GPU hang watchdog — see `Self::max_chain`.
    dispatches: usize,
}

impl RecordedCmd {
    pub(crate) fn from_shared_segments(
        shared: std::sync::Arc<crate::VulkanShared>,
        segments: Vec<RecordedSegment>,
    ) -> Self {
        let dispatches = segments.iter().map(|s| s.dispatches).sum();
        Self {
            shared,
            segments,
            dispatches,
        }
    }

    /// Assemble a (possibly multi-segment) recording — the watchdog-split decode path.
    pub(crate) fn from_segments(
        be_: &crate::VulkanBackend,
        segments: Vec<RecordedSegment>,
    ) -> Self {
        Self::from_shared_segments(std::sync::Arc::clone(&be_.shared), segments)
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl RecordedCmd {
    /// The longest chain `replay_n` may be given on this device: `n` copies of the recording go
    /// into ONE submit, and the GPU hang watchdog is armed per submit, so an unbounded chain is
    /// the same device-lost trap the forward-pass splitter exists to close (on the surveyed 2-CU
    /// iGPU a decode step is ~213 ms, so the old ceiling of 64 would have been ~13.6 s in a single
    /// command buffer). Bounded by the SAME measured dispatch cap the splitter uses; `usize::MAX`
    /// (no bound) on a device that never splits, which is every discrete GPU. A multi-segment
    /// recording only exists on a splitting device whose decode already EXCEEDS the cap, so this
    /// returns 1 there — `replay_n` never packs copies of a split decode.
    pub fn max_chain(&self) -> usize {
        let cap = self
            .shared
            .submit_dispatch_cap
            .load(std::sync::atomic::Ordering::Relaxed);
        if cap == 0 || self.dispatches == 0 {
            return usize::MAX;
        }
        // Always at least one: a single decode step is indivisible, and it is short enough on
        // every device we have measured.
        (cap / self.dispatches).max(1)
    }

    /// Resubmit the recording `n` times back to back and wait once — the chained decode's n
    /// iterations. On the single-segment (discrete) fast path this is ONE queue submission of `n`
    /// copies (legal via SIMULTANEOUS_USE); the recording's leading global barrier orders
    /// consecutive iterations. Callers must clamp `n` to [`Self::max_chain`] — see
    /// `adapter::execute_chain`. A multi-segment (watchdog-split) recording only ever reaches here
    /// with `n == 1` (its `max_chain` is 1), and each iteration is a full `Self::replay`.
    pub fn replay_n(&self, n: usize) -> Result<()> {
        let [seg] = &self.segments[..] else {
            // Watchdog-split recording: replay the whole (already per-segment-split) decode n times.
            for _ in 0..n {
                self.replay()?;
            }
            return Ok(());
        };
        let device = &self.shared.device;
        let cmds = vec![seg.cmd; n];
        let prof = pager_profile::active();
        let submit_t0 = prof.then(std::time::Instant::now);
        let submit = unsafe {
            device.queue_submit(
                self.shared.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                vk::Fence::null(),
            )
        };
        if let Some(t0) = submit_t0 {
            pager_profile::record_queue_submit(seg.dispatches * n, t0.elapsed());
        }
        submit.map_err(|e| be(format!("replay_n submit: {e}")))?;
        let wait_t0 = prof.then(std::time::Instant::now);
        let wait = unsafe { device.queue_wait_idle(self.shared.queue) };
        if let Some(t0) = wait_t0 {
            pager_profile::record_sync_wait(pager_profile::SyncKind::QueueIdle, t0.elapsed());
        }
        wait.map_err(|e| be(format!("replay_n wait: {e}")))?;
        Ok(())
    }

    /// Resubmit the recording and wait for completion. Each segment is a SEPARATE `vkQueueSubmit`,
    /// so the GPU hang watchdog — armed per submit — sees a split decode as several short jobs
    /// running back-to-back on the queue rather than one long one. The segments still execute in
    /// submission order and each continuation segment's leading `seed_barrier` carries the
    /// cross-segment RAW/WAR memory dependency (a pipeline barrier's first scope spans everything
    /// submitted earlier on the same queue). A discrete GPU has exactly ONE segment, so this is a
    /// single submit + wait — byte-identical to the record-once fast path.
    pub fn replay(&self) -> Result<()> {
        let device = &self.shared.device;
        let prof = pager_profile::active();
        for seg in &self.segments {
            let submit_t0 = prof.then(std::time::Instant::now);
            let submit = unsafe {
                device.queue_submit(
                    self.shared.queue,
                    &[vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&seg.cmd))],
                    vk::Fence::null(),
                )
            };
            if let Some(t0) = submit_t0 {
                pager_profile::record_queue_submit(seg.dispatches, t0.elapsed());
            }
            submit.map_err(|e| be(format!("replay submit: {e}")))?;
        }
        let wait_t0 = prof.then(std::time::Instant::now);
        let wait = unsafe { device.queue_wait_idle(self.shared.queue) };
        if let Some(t0) = wait_t0 {
            pager_profile::record_sync_wait(pager_profile::SyncKind::QueueIdle, t0.elapsed());
        }
        wait.map_err(|e| be(format!("replay wait: {e}")))?;
        Ok(())
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Drop for RecordedCmd {
    fn drop(&mut self) {
        let device = &self.shared.device;
        unsafe {
            let _ = device.queue_wait_idle(self.shared.queue);
            let cmd_pool = *self.shared.cmd_pool.lock().unwrap();
            for seg in &self.segments {
                device.free_command_buffers(cmd_pool, &[seg.cmd]);
                for p in &seg.pools {
                    device.destroy_descriptor_pool(*p, None);
                }
            }
        }
    }
}

/// A submitted-but-unwaited segment from [`Recorder::finish_nowait`]: holds the fence its submit
/// signals plus the transient objects (command buffer, descriptor pools) that must outlive GPU
/// execution. [`Self::wait`] blocks on the fence and releases everything; `Drop` does the same
/// (best-effort) so an error path can never free a command buffer the GPU still owns.
pub struct PendingSegment {
    shared: std::sync::Arc<crate::VulkanShared>,
    /// `None` = already drained (the profiling degrade path in `finish_nowait`, or after `wait`).
    fence: Option<vk::Fence>,
    cmd: vk::CommandBuffer,
    pools: Vec<vk::DescriptorPool>,
    /// Dispatches this segment carried — the submit splitter sums them across a forward to feed
    /// `VulkanBackend::observe_forward`.
    dispatches: usize,
}

impl PendingSegment {
    /// Dispatches recorded into this segment's command buffer.
    pub fn dispatches(&self) -> usize {
        self.dispatches
    }
}

impl PendingSegment {
    /// Block until the segment's GPU work completes, then release its transient objects.
    pub fn wait(mut self) -> Result<()> {
        self.drain()
    }

    fn drain(&mut self) -> Result<()> {
        let Some(fence) = self.fence.take() else {
            return Ok(()); // already drained
        };
        let device = &self.shared.device;
        let wait_t0 = pager_profile::start();
        let waited = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) };
        if let Some(elapsed) = pager_profile::elapsed(wait_t0) {
            pager_profile::record_sync_wait(pager_profile::SyncKind::Fence, elapsed);
        }
        unsafe {
            device.destroy_fence(fence, None);
            let cmd_pool = *self.shared.cmd_pool.lock().unwrap();
            device.free_command_buffers(cmd_pool, &[self.cmd]);
            for p in self.pools.drain(..) {
                device.destroy_descriptor_pool(p, None);
            }
        }
        waited.map_err(|e| be(format!("wait segment fence: {e}")))
    }
}

impl Drop for PendingSegment {
    fn drop(&mut self) {
        let _ = self.drain();
    }
}

#[cfg(test)]
mod chunk_math_tests {
    //! Pure geometry tests for the chunked-dispatch split (`bda_out_f_chunks`) — no GPU. Proves the
    //! output-row chunking stays under both u32 addressing caps, covers every row exactly once,
    //! keeps chunk starts binding-offset aligned, and fails loudly (never loops) on an un-chunkable
    //! single row. This is the math a `>= 2^32`-element lm_head/embed relies on to run at all.
    use super::{bda_out_f_chunks, OutRowChunk, BDA_CHUNK_ROW_ALIGN};
    use infr_core::DType;

    const CAP: u64 = 1 << 32; // the production element/byte cap

    fn assert_covers(
        chunks: &[OutRowChunk],
        out_f: usize,
        in_f: usize,
        dtype: DType,
        byte_cap: u64,
    ) {
        let (be, bb) = infr_gguf::block_layout(dtype);
        let row_bytes = ((in_f / be) * bb) as u64;
        // Contiguous, gap-free cover of [0, out_f) with aligned, cap-respecting chunks.
        let mut next = 0usize;
        for c in chunks {
            assert_eq!(c.o0, next, "chunks must be contiguous");
            assert_eq!(
                c.o0 % BDA_CHUNK_ROW_ALIGN,
                0,
                "chunk start must be row-aligned"
            );
            assert_eq!(
                c.byte_off,
                (c.o0 as u64) * row_bytes,
                "byte_off = o0 * row_bytes"
            );
            assert!(c.n >= 1);
            assert!(
                (c.n as u64) * (in_f as u64) < CAP,
                "chunk elements under the cap"
            );
            assert!(
                (c.n as u64) * row_bytes < byte_cap,
                "chunk bytes under the cap"
            );
            next += c.n;
        }
        assert_eq!(
            next, out_f,
            "chunks must cover every output row exactly once"
        );
    }

    #[test]
    fn whole_tensor_is_one_chunk_at_production_caps() {
        // A real 151936-vocab x 5120 lm_head (778M elems, ~1.5 GiB Q4_K) fits under both caps: one
        // chunk, byte_off 0 — byte-identical to the unchunked dispatch. Every shipping model.
        let c = bda_out_f_chunks(DType::Q4K, 5120, 151936, 0, CAP, CAP).unwrap();
        assert_eq!(
            c,
            vec![OutRowChunk {
                o0: 0,
                n: 151936,
                byte_off: 0
            }]
        );
    }

    #[test]
    fn breaching_lm_head_splits_and_covers() {
        // 262144-vocab x 16384 hidden = 4.295G elems (> 2^32) at Q4_K (~2.3 GiB, under the byte
        // cap): the first plausible real breach. Must split on element boundaries and cover fully.
        let (in_f, out_f) = (16384, 262144);
        let c = bda_out_f_chunks(DType::Q4K, in_f, out_f, 0, CAP, CAP).unwrap();
        assert!(c.len() >= 2, "a 4.3G-element tensor must split");
        assert_covers(&c, out_f, in_f, DType::Q4K, CAP);
    }

    #[test]
    fn exact_cap_and_one_below() {
        // `< cap` is strict: a whole-tensor element count of exactly 2^32 splits; 2^32 - align does
        // not. Use F16 (1 elem/row-unit) with in_f == align so out_f is the element count / align.
        let in_f = BDA_CHUNK_ROW_ALIGN; // 64
        let at_cap = (CAP / in_f as u64) as usize; // out_f * in_f == 2^32 exactly
        let split = bda_out_f_chunks(DType::F16, in_f, at_cap, 0, CAP, u64::MAX).unwrap();
        assert!(split.len() >= 2, "exactly-at-cap must split");
        assert_covers(&split, at_cap, in_f, DType::F16, u64::MAX);
        // One aligned group of rows below the cap still fits in a single chunk.
        let below = bda_out_f_chunks(DType::F16, in_f, at_cap - in_f, 0, CAP, u64::MAX).unwrap();
        assert_eq!(below.len(), 1);
    }

    #[test]
    fn one_over_cap_splits() {
        let in_f = 1024usize;
        let out_f = (CAP / in_f as u64) as usize + 1; // one row past 2^32 elements
        let c = bda_out_f_chunks(DType::F16, in_f, out_f, 0, CAP, u64::MAX).unwrap();
        assert!(c.len() >= 2);
        assert_covers(&c, out_f, in_f, DType::F16, u64::MAX);
    }

    #[test]
    fn non_divisible_row_count_has_remainder_tail() {
        // A forced-small cap that yields max_rows = 640, out_f = 2000 → 640,640,640,80.
        let (in_f, out_f) = (1024usize, 2000usize);
        let elem_cap = 640 * 1024 + 5; // by_elem floor(645/1024*... )-> 640 rows
        let c = bda_out_f_chunks(DType::F16, in_f, out_f, 0, elem_cap as u64, u64::MAX).unwrap();
        assert_covers(&c, out_f, in_f, DType::F16, u64::MAX);
        assert_eq!(
            c.last().unwrap().n,
            out_f % 640,
            "the tail chunk is the remainder"
        );
        assert!(c.last().unwrap().n < 640);
    }

    #[test]
    fn byte_cap_binds_before_element_cap_for_f16() {
        // F16 is 2 bytes/elem, so the 4 GiB byte cap trips at 2^31 elements — half the element cap.
        // With the element cap lifted, chunking is driven purely by bytes.
        let (in_f, out_f) = (2048usize, 1_500_000usize); // 3.07G elems, 6.14 GiB
        let c = bda_out_f_chunks(DType::F16, in_f, out_f, 0, u64::MAX, CAP).unwrap();
        assert!(
            c.len() >= 2,
            "the byte cap must force a split even below the element cap"
        );
        assert_covers(&c, out_f, in_f, DType::F16, CAP);
    }

    #[test]
    fn single_row_too_big_errors_no_infinite_loop() {
        // A single row that already breaches (in_f itself past the element cap) is un-chunkable —
        // it must Err, not loop forever trying to fit a zero-row chunk.
        let huge_in_f = (CAP as usize) + 4096; // one row >= 2^32 elements
        assert!(bda_out_f_chunks(DType::F16, huge_in_f, 10, 0, CAP, u64::MAX).is_err());
        // Same guard under a forced-tiny cap smaller than one aligned group of rows.
        assert!(bda_out_f_chunks(DType::F16, 4096, 10, 0, 4096, u64::MAX).is_err());
    }

    #[test]
    fn nonzero_slice_offset_on_breach_errors() {
        // A breaching tensor with a nonzero within-tensor slice offset (fused-QKV) is unsupported —
        // no such tensor exists (only lm_head/embed breach, at w_base 0), and the chunk base folds
        // into the arena address, so mixing the two must fail loudly rather than mis-address.
        let out_f = (CAP / 1024) as usize + 1;
        assert!(bda_out_f_chunks(DType::F16, 1024, out_f, 512, CAP, u64::MAX).is_err());
        // A nonzero offset on a NON-breaching tensor is fine (the common fused-QKV case).
        assert!(bda_out_f_chunks(DType::F16, 1024, 4096, 512, CAP, CAP).is_ok());
    }

    #[test]
    fn in_f_not_block_multiple_errors() {
        // A row that isn't a whole number of blocks would mis-map byte offsets — reject.
        assert!(bda_out_f_chunks(DType::Q4K, 100, 4096, 0, CAP, CAP).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::{backend::BufferUsage, Backend};

    /// A backend built on an EXPLICIT Vulkan kernel-tier configuration (`docs/config-plan.md` S5b).
    ///
    /// This is what replaced `EnvGuard::with([("INFR_FLASH_SPLITS", "4"), …])` in every test below:
    /// the tier knobs are a VALUE handed to `VulkanBackend::new_with`, so nothing is process-global,
    /// nothing has to be restored, and two backends with different tiers can be live at once (R7).
    /// The cost is one device per configuration — these tests already built one per case.
    fn be_with(f: impl FnOnce(&mut infr_core::config::VulkanCfg)) -> VulkanBackend {
        let mut cfg = infr_core::config::Config::default();
        f(&mut cfg.kernels.vulkan);
        VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap()
    }

    /// #1 drift guard: `kernels.vulkan.gemv` must resolve, for any env state, EXACTLY what the
    /// pre-config per-call `std::env::var` reads in the GEMV dispatchers would have returned — so
    /// neither the (deleted) `OnceLock<GemvKnobs>` memo nor its replacement can change a
    /// kernel-selection decision.
    ///
    /// S5b: the struct under test is now `infr_core::config::GemvCfg`, resolved through the config
    /// ENV LAYER with an injected reader — the same `HashMap` states as before, and still zero
    /// contact with the process environment (R7).
    #[test]
    fn gemv_knobs_resolve_matches_env_reads() {
        use std::collections::HashMap;
        let states: Vec<HashMap<&str, &str>> = vec![
            // all defaults
            HashMap::new(),
            // every disable flag set (value present, even empty)
            HashMap::from([
                ("INFR_NO_GEMV_RM", ""),
                ("INFR_NO_GEMV_SG", ""),
                ("INFR_NO_GEMV_ID_SG", ""),
                ("INFR_NO_GEMV_REG", ""),
            ]),
            // explicit overrides on the numeric knobs + a variant
            HashMap::from([
                ("INFR_GEMV_RM", "4"),
                ("INFR_GEMV_RM_MAXOUT", "9000"),
                ("INFR_GEMV_RM_MINOUT", "1000"),
                ("INFR_GEMV_SG_MINOUT", "512"),
                ("INFR_GEMV_SG_MAXOUT", "4096"),
                ("INFR_GEMV_SG_NR", "8"),
                ("INFR_GEMV_VARIANT", "o4"),
            ]),
            // unparseable values must fall back to the defaults, same as `.parse().ok()` did
            HashMap::from([("INFR_GEMV_RM", "xx"), ("INFR_GEMV_SG_NR", "yy")]),
        ];
        for st in states {
            let get = |name: &str| st.get(name).map(|s| s.to_string());
            let layer = infr_core::config::env::parse(&get).expect("gemv keys never error");
            let k = &infr_core::config::Config::load_from_layers(&[layer])
                .kernels
                .vulkan
                .gemv;
            // The OLD inline reads, replicated verbatim against the same env map:
            let old_variant = if get("INFR_NO_GEMV_REG").is_some() {
                None
            } else {
                get("INFR_GEMV_VARIANT").or_else(|| Some("reg".to_string()))
            };
            assert_eq!(k.no_rm, get("INFR_NO_GEMV_RM").is_some());
            assert_eq!(
                k.rm_maxout,
                get("INFR_GEMV_RM_MAXOUT")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(usize::MAX)
            );
            assert_eq!(
                k.rm_minout,
                get("INFR_GEMV_RM_MINOUT")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2048usize)
            );
            assert_eq!(
                k.rm,
                get("INFR_GEMV_RM")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2u32)
            );
            assert_eq!(k.no_sg, get("INFR_NO_GEMV_SG").is_some());
            assert_eq!(k.no_id_sg, get("INFR_NO_GEMV_ID_SG").is_some());
            assert_eq!(
                k.sg_minout,
                get("INFR_GEMV_SG_MINOUT")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2048usize)
            );
            assert_eq!(
                k.sg_maxout,
                get("INFR_GEMV_SG_MAXOUT")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8192usize)
            );
            assert_eq!(
                k.sg_nr,
                get("INFR_GEMV_SG_NR")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2u32)
            );
            assert_eq!(k.variant, old_variant);
        }
    }

    /// #4 drift guard: the ONE unified [`moe_mmq_desc`] table must produce, for every dtype in
    /// `MOE_MMQ_DTYPES`, the SAME `(kernel_name, nbind)` the two old hand-synced match tables did —
    /// resident (`_xp`/`_xp32` + Q4_K/Q6_K wide `_xp128`/`_xp32w`) and paged (`_xpg`/`_xpg32`,
    /// `nb = resident_nb + 1`). Golden is an independent stem+nb copy so a name/nb edit in
    /// `moe_mmq_desc` is caught here rather than at a mid-inference `unreachable!`.
    #[test]
    fn moe_mmq_table_matches_frozen_names() {
        use infr_core::DType as D;
        // (dtype, kernel stem, resident binding count) — the frozen snapshot of both old tables.
        let golden: &[(D, &str, usize)] = &[
            (D::Q4_0, "q4_0", 6),
            (D::Q4_1, "q4_1", 7),
            (D::Q5_0, "q5_0", 6),
            (D::Q5_1, "q5_1", 7),
            (D::Q8_0, "q8_0", 6),
            (D::Q2K, "q2_k", 6),
            (D::Q3K, "q3_k", 6),
            (D::Q4K, "q4k", 7),
            (D::Q5K, "q5k", 7),
            (D::Q6K, "q6k", 6),
            (D::Iq4Nl, "iq4_nl", 6),
            (D::Iq4Xs, "iq4_xs", 6),
            (D::Iq2S, "iq2_s", 6),
            (D::Iq3S, "iq3_s", 6),
            (D::Mxfp4, "mxfp4", 6),
            (D::Nvfp4, "nvfp4", 6),
            (D::Q2_0, "q2_0", 6),
        ];
        // Golden and MOE_MMQ_DTYPES must cover exactly the same set.
        assert_eq!(golden.len(), infr_core::tensor::MOE_MMQ_DTYPES.len());
        for dt in infr_core::tensor::MOE_MMQ_DTYPES {
            assert!(
                golden.iter().any(|(g, _, _)| g == dt),
                "MOE_MMQ_DTYPES member {dt:?} missing from golden"
            );
        }
        let wide = [D::Q4K, D::Q6K];
        for &(dt, stem, nb) in golden {
            let d = moe_mmq_desc(dt).unwrap_or_else(|| panic!("moe_mmq_desc None for {dt:?}"));
            assert_eq!(d.resident_nb, nb, "{dt:?} resident_nb");
            assert_eq!(
                d.xp.0,
                format!("native_gemm_mmq_{stem}_xp").as_str(),
                "{dt:?} xp"
            );
            assert_eq!(
                d.xp32.0,
                format!("native_gemm_mmq_{stem}_xp32").as_str(),
                "{dt:?} xp32"
            );
            // paged variants + the +1 LUT binding are applied by the caller; names must match.
            assert_eq!(
                d.xpg.0,
                format!("native_gemm_mmq_{stem}_xpg").as_str(),
                "{dt:?} xpg"
            );
            assert_eq!(
                d.xpg32.0,
                format!("native_gemm_mmq_{stem}_xpg32").as_str(),
                "{dt:?} xpg32"
            );
            if wide.contains(&dt) {
                let (w128, w32w) = d.xp_wide.expect("Q4_K/Q6_K carry xp_wide");
                assert_eq!(
                    w128.0,
                    format!("native_gemm_mmq_{stem}_xp128").as_str(),
                    "{dt:?} xp128"
                );
                assert_eq!(
                    w32w.0,
                    format!("native_gemm_mmq_{stem}_xp32w").as_str(),
                    "{dt:?} xp32w"
                );
            } else {
                assert!(d.xp_wide.is_none(), "{dt:?} must not carry xp_wide");
            }
        }
        // A dtype outside the mmq set maps to None (drives the callers' `unreachable!`).
        assert!(moe_mmq_desc(D::F32).is_none());
    }

    fn rmsnorm_cpu(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale = 1.0 / (ss + eps).sqrt();
        (0..n).map(|i| x[i] * scale * w[i]).collect()
    }
    fn dot(w: &[f32], row: usize, ne: usize, x: &[f32]) -> f32 {
        (0..ne).map(|k| w[row * ne + k] * x[k]).sum()
    }
    // ggml NORM-interleaved rope of a head-dim vector in place over the first rope_dim entries.
    fn rope_head(v: &mut [f32], hd: usize, rope_dim: usize, theta: f32, pos: usize) {
        for i in 0..rope_dim / 2 {
            let freq = theta.powf(-2.0 * i as f32 / rope_dim as f32);
            let ang = pos as f32 * freq;
            let (s, co) = (ang.sin(), ang.cos());
            let a = v[2 * i];
            let b = v[2 * i + 1];
            v[2 * i] = a * co - b * s;
            v[2 * i + 1] = a * s + b * co;
        }
        let _ = hd;
    }

    // round f32 → f16 → f32 (matches what the f16 q/k/v buffers store)
    fn r16(v: &[f32]) -> Vec<f32> {
        v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
    }
    // upload f32 values as an f16 buffer
    fn upf16(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let bits: Vec<u16> = v
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let b = be.alloc(bits.len() * 2, BufferUsage::Staging).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(&bits)).unwrap();
        b
    }

    // Reference attention. `window`>0 = sliding-window lower bound (gemma SWA); `scale_in`>0 overrides
    // the default 1/√hd (gemma4 = 1.0). Matches attention_kv.comp and attn_softmax.comp semantics.
    #[allow(clippy::too_many_arguments)]
    fn attn_kv_cpu(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        window: usize,
        scale_in: f32,
    ) -> Vec<f32> {
        let scale = if scale_in > 0.0 {
            scale_in
        } else {
            1.0 / (hd as f32).sqrt()
        };
        let mut o = vec![0f32; q_len * nh * hd];
        for ti in 0..q_len {
            let abs = pos_offset + ti;
            let lo = if window > 0 && abs + 1 > window {
                abs + 1 - window
            } else {
                0
            };
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let qb = (ti * nh + h) * hd;
                let mut sc = vec![0f32; abs + 1 - lo]; // valid keys [lo, abs]
                let mut mx = f32::NEG_INFINITY;
                for (jj, scj) in sc.iter_mut().enumerate() {
                    let kb = ((lo + jj) * nkv + kvh) * hd;
                    let d: f32 = (0..hd).map(|x| q[qb + x] * k[kb + x]).sum();
                    *scj = d * scale;
                    mx = mx.max(*scj);
                }
                let mut l = 0f32;
                for s in &sc {
                    l += (s - mx).exp();
                }
                let ob = (ti * nh + h) * hd;
                for (jj, s) in sc.iter().enumerate() {
                    let p = (s - mx).exp() / l;
                    let vb = ((lo + jj) * nkv + kvh) * hd;
                    for x in 0..hd {
                        o[ob + x] += p * v[vb + x];
                    }
                }
            }
        }
        o
    }

    fn run_attn_kv(q_len: usize, kv_len: usize, nh: usize, nkv: usize, hd: usize) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len; // new tokens are the last q_len of the cache
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        // q/k/v are f16 on the GPU; round the reference inputs to f16 too so the test isolates the
        // attention math (not f16 rounding).
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be
            .alloc(q_len * nh * hd * 4, BufferUsage::Readback)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            0,     // full causal (no sliding window)
            0.0,   // default 1/√hd scale
            false, // k f16
            false, // v f16
            0,     // cap (unused for f16)
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; q_len * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attention_kv q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attention_kv mismatch: {err}");
    }

    fn run_attn_kv_split(kv_len: usize, nh: usize, nkv: usize, hd: usize, scale: f32, win: usize) {
        let be = VulkanBackend::new().unwrap();
        let chunk = 512usize;
        let n_chunks = kv_len.div_ceil(chunk);
        let pos_offset = kv_len - 1; // decode: one new token at the end
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be.alloc(nh * hd * 4, BufferUsage::Readback).unwrap();
        let pm = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pl = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pacc = be
            .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_kv_split(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            1,          // rows (decode shape)
            kv_len - 1, // pos of the single query row
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            win,
            None,  // canvas_lo: causal decode, not DiffusionGemma canvas
            false, // k f16
            false, // v f16
            0,     // cap (unused for f16)
            false, // batched: decode shape stays on the per-row grid
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, 1, nh, nkv, hd, pos_offset, win, scale);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_kv_split kv_len={kv_len} n_chunks={n_chunks} win={win} max_err={err:e}");
        assert!(err < 5e-3, "split mismatch: {err}");
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attn_kv_split_dynac(
        kv_len: usize,
        cap: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        win: usize,
    ) {
        let be = VulkanBackend::new().unwrap();
        let chunk = cap.div_ceil(1024).max(64);
        let n_chunks = cap.div_ceil(chunk);
        let pos_offset = kv_len - 1;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &k);
        let bv = upf16(&be, &v);
        let bo = be.alloc(nh * hd * 4, BufferUsage::Readback).unwrap();
        let pm = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pl = be
            .alloc(nh * n_chunks * 4, BufferUsage::Activations)
            .unwrap();
        let pacc = be
            .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
            .unwrap();
        let params = be.alloc(8, BufferUsage::Activations).unwrap();
        be.upload(
            params.as_ref(),
            bytemuck::cast_slice(&[pos_offset as u32, kv_len as u32]),
        )
        .unwrap();
        let args = be.alloc(16, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, chunk, win);
        rec.attention_kv_split_dynac(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            params.as_ref(),
            args.as_ref(),
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            scale,
            win,
            false, // f16 KV cache
            0,     // cap (unused for f16)
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, 1, nh, nkv, hd, pos_offset, win, scale);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "attn_kv_split_dynac kv={kv_len} cap={cap} n_chunks={n_chunks} win={win} max_err={err:e}"
        );
        assert!(err < 5e-3, "dynac split mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_split_dynac_matches_cpu() {
        run_attn_kv_split_dynac(20, 68, 16, 8, 128, 0.0, 0); // shallow: 1 live chunk, tiny cap
        run_attn_kv_split_dynac(600, 8065, 9, 3, 64, 0.0, 0); // wide bake, few live
        run_attn_kv_split_dynac(8000, 8065, 16, 8, 128, 0.0, 0); // deep: 32 live chunks
        run_attn_kv_split_dynac(130, 40960, 4, 2, 128, 0.0, 0); // huge cap (chunk floor rises)
                                                                // gemma-family replay shapes: SWA windows (span-chunked grid) + explicit scale.
        run_attn_kv_split_dynac(2050, 8065, 16, 8, 256, 0.0, 512); // gemma3 SWA deep (window << kv)
        run_attn_kv_split_dynac(300, 8065, 16, 8, 256, 0.0, 512); // SWA shallow (kv < window)
        run_attn_kv_split_dynac(700, 8065, 16, 8, 256, 0.0, 640); // window not chunk-aligned
        run_attn_kv_split_dynac(4000, 8065, 8, 2, 512, 1.0, 1024); // gemma4 hd=512, scale=1.0, SWA
        run_attn_kv_split_dynac(900, 8065, 8, 4, 256, 1.0, 0); // gemma4 full-attn (scale only)
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_split_matches_cpu() {
        run_attn_kv_split(600, 9, 3, 64, 0.0, 0); // 2 chunks
        run_attn_kv_split(2050, 9, 3, 64, 0.0, 0); // 5 chunks, partial last
        run_attn_kv_split(8000, 4, 2, 32, 0.0, 0); // 16 chunks
        run_attn_kv_split(830, 16, 2, 256, 0.0, 0); // qwen35 full-attn decode (hd=256 general path)
        run_attn_kv_split(2050, 16, 8, 256, 0.0, 0); // gemma SWA-shape decode (hd=256, GQA 16:8)
        run_attn_kv_split(2050, 16, 8, 256, 0.0, 512); // SWA window (chunks below lo → empty)
        run_attn_kv_split(4000, 8, 2, 512, 1.0, 1024); // gemma4 hd=512, scale=1.0, SWA
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_decode_matches_cpu() {
        // decode: 1 new token over a cache; 2500 exercises multi-tile flash (>TILE=1024)
        run_attn_kv(1, 200, 9, 3, 64);
        run_attn_kv(1, 13, 4, 2, 32);
        run_attn_kv(1, 1, 4, 2, 32);
        run_attn_kv(1, 2500, 9, 3, 64);
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_kv_prefill_matches_cpu() {
        run_attn_kv(17, 17, 9, 3, 64);
        run_attn_kv(40, 1500, 9, 3, 64); // multi-tile prefill (kv_len > TILE)
                                         // gemma4: SWA layers (hd=256, GQA 16:8) and full layers (hd=512, GQA 16:1).
        run_attn_kv(17, 17, 16, 8, 256);
        run_attn_kv(17, 17, 16, 1, 512);
        run_attn_kv(1, 200, 16, 1, 512); // gemma4 full-layer decode
    }

    // upload f32 values as an f32 buffer (qk_norm_rope reads x / nw / freq_factors as f32).
    fn upf32(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    }

    /// CPU reference for fused per-head QK-norm (RMSNorm over hd) + NEOX RoPE (optional freq_factors).
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_cpu(
        x: &[f32],
        nw: &[f32],
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        eps: f32,
        ff: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut y = vec![0f32; (out_base + rows) * nheads * hd];
        let hf = rope_dim / 2;
        for r in 0..rows {
            for h in 0..nheads {
                let ib = (r * nheads + h) * hd;
                let ss: f32 = (0..hd).map(|i| x[ib + i] * x[ib + i]).sum::<f32>() / hd as f32;
                let scale = 1.0 / (ss + eps).sqrt();
                let ob = ((out_base + r) * nheads + h) * hd;
                for p in 0..hf {
                    let (i0, i1) = (p, p + hf);
                    let a = x[ib + i0] * scale * nw[i0];
                    let b = x[ib + i1] * scale * nw[i1];
                    let freq = theta.powf(-2.0 * p as f32 / rope_dim as f32);
                    let mut ang = (rope_pos + r) as f32 * freq;
                    if let Some(ff) = ff {
                        ang /= ff[p];
                    }
                    let (s, c) = (ang.sin(), ang.cos());
                    y[ob + i0] = a * c - b * s;
                    y[ob + i1] = a * s + b * c;
                }
            }
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qk_norm_rope(
        rows: usize,
        nheads: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        rope_pos: usize,
        out_base: usize,
        with_ff: bool,
    ) {
        let be = VulkanBackend::new().unwrap();
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let x = gen(rows * nheads * hd, 1);
        let nw: Vec<f32> = gen(hd, 2).iter().map(|v| v + 1.05).collect(); // ~gemma norm weights near 1
                                                                          // proportional-rope freq_factors: first quarter rotate (1.0), rest unrotated (1e30) — like gemma4
        let ff: Vec<f32> = (0..rope_dim / 2)
            .map(|p| if p < rope_dim / 4 { 1.0 } else { 1e30 })
            .collect();
        let bx = upf32(&be, &x);
        let bnw = upf32(&be, &nw);
        let bff = upf32(&be, &ff);
        let y_len = (out_base + rows) * nheads * hd;
        let by = be.alloc(y_len * 2, BufferUsage::Readback).unwrap(); // f16 out
        let rec = be.recorder().unwrap();
        rec.qk_norm_rope(
            bx.as_ref(),
            bnw.as_ref(),
            by.as_ref(),
            rows,
            nheads,
            hd,
            rope_dim,
            theta,
            rope_pos,
            out_base,
            1e-6,
            if with_ff { Some(bff.as_ref()) } else { None },
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; y_len * 2];
        be.download(by.as_ref(), &mut bytes).unwrap();
        let got: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let want = qk_norm_rope_cpu(
            &x,
            &nw,
            rows,
            nheads,
            hd,
            rope_dim,
            theta,
            rope_pos,
            out_base,
            1e-6,
            if with_ff { Some(&ff) } else { None },
        );
        // compare only the rows the kernel actually wrote (out_base..out_base+rows)
        let mut err = 0f32;
        for r in 0..rows {
            for h in 0..nheads {
                for i in 0..hd {
                    let idx = ((out_base + r) * nheads + h) * hd + i;
                    err = err.max((got[idx] - want[idx]).abs());
                }
            }
        }
        println!(
            "qk_norm_rope rows={rows} nheads={nheads} hd={hd} rope_dim={rope_dim} ff={with_ff} max_err={err:e}"
        );
        assert!(err < 1e-2, "qk_norm_rope mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn qk_norm_rope_matches_cpu() {
        run_qk_norm_rope(8, 4, 128, 128, 1e6, 0, 0, false); // qwen3 hd=128
        run_qk_norm_rope(8, 4, 256, 256, 1e4, 0, 0, false); // gemma3 hd=256
        run_qk_norm_rope(17, 16, 256, 256, 1e4, 5, 0, false); // gemma4 SWA Q (out_base=0)
        run_qk_norm_rope(17, 8, 256, 256, 1e4, 5, 5, false); // gemma4 SWA K (out_base=pos)
        run_qk_norm_rope(17, 16, 512, 512, 1e6, 5, 0, false); // gemma4 full Q
        run_qk_norm_rope(17, 1, 512, 512, 1e6, 5, 5, false); // gemma4 full K
        run_qk_norm_rope(17, 16, 512, 512, 1e6, 5, 0, true); // gemma4 full Q + freq_factors
        run_qk_norm_rope(17, 1, 512, 512, 1e6, 5, 5, true); // gemma4 full K + freq_factors
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_nonfa_matches_cpu() {
        let be = be_with(|_| {});
        // hd=128 (qwen3), full causal, default scale — the pre-existing regression cases.
        run_attn_prefill_nonfa(&be, 64, 64, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(&be, 128, 200, 4, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(&be, 70, 70, 2, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(&be, 192, 500, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(&be, 80, 300, 9, 3, 64, 0, 0.0);
        // gemma: hd=256 (SWA, GQA 16:8) and hd=512 (full, GQA 16:1), with the sliding-window mask
        // and the scale override (gemma4 = 1.0). These are the paths the new routing exercises.
        run_attn_prefill_nonfa(&be, 128, 300, 16, 8, 256, 100, 1.0); // gemma4 SWA (window)
        run_attn_prefill_nonfa(&be, 128, 300, 16, 1, 512, 0, 1.0); // gemma4 full (scale=1)
        run_attn_prefill_nonfa(&be, 128, 300, 16, 8, 256, 100, 0.0); // gemma3 SWA (window, 1/√hd)
        run_attn_prefill_nonfa(&be, 70, 400, 16, 8, 256, 64, 1.0); // SWA, non-64-aligned q
                                                                   // Force the split-K PV path (n_splits>1) and verify the partial-sum reduce is correct,
                                                                   // incl. with a window (split reduce must respect the softmax mask) and hd=512.
        let split = be_with(|v| v.pv_splits = Some(4));
        run_attn_prefill_nonfa(&split, 70, 300, 4, 2, 128, 0, 0.0);
        run_attn_prefill_nonfa(&split, 128, 500, 2, 1, 128, 0, 0.0);
        run_attn_prefill_nonfa(&split, 128, 3000, 16, 8, 256, 200, 1.0); // SWA, long kv, split-K
        run_attn_prefill_nonfa(&split, 128, 3000, 16, 1, 512, 0, 1.0); // hd=512, long kv, split-K
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attn_prefill_nonfa(
        be: &VulkanBackend,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
        scale: f32,
    ) {
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(64) * 64;
        let kv_pad = kv_len.div_ceil(256) * 256; // recorder pads kv to 256 (8-warp attn_qk BN)
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        // K/V must cover the padded kv (the kernel reads padded rows; softmax masks them).
        let mut kp = k.clone();
        kp.resize((kv_pad + 64) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_pad + 64) * nkv * hd, 0.0);
        let bq = upf16(be, &qp);
        let bk = upf16(be, &kp);
        let bv = upf16(be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let bs = be
            .alloc(nh * mpad * kv_pad * 2, BufferUsage::Activations)
            .unwrap();
        let bpv = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_nonfa(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            bs.as_ref(),
            bpv.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            window,
            scale,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, window, scale);
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_prefill_nonfa q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attn_prefill_nonfa mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_flash_matches_cpu() {
        let be = be_with(|_| {});
        for &(q, kv, nh, nkv, hd) in &[
            (64usize, 64usize, 2usize, 1usize, 128usize),
            (128, 200, 4, 2, 128),
            (70, 70, 2, 2, 128),
            (192, 500, 2, 1, 128),
            (80, 300, 9, 3, 64),
            (448, 2000, 16, 8, 128), // qwen3-shaped, multi-block kv
            (16, 64, 4, 2, 256),     // hd256 minimum BM tile
            (70, 200, 8, 2, 256),    // hd256 padded query rows + GQA
            (128, 500, 8, 2, 256),   // hd256 multi-BN-block kv
        ] {
            run_attn_prefill_flash(&be, q, kv, nh, nkv, hd);
        }
        // force the split-K flash path (partial+combine) and verify the merge is correct
        let split = be_with(|v| v.flash_splits = Some(4));
        run_attn_prefill_flash(&split, 64, 2000, 16, 8, 128);
        run_attn_prefill_flash(&split, 128, 500, 2, 1, 128);
        run_attn_prefill_flash(&split, 64, 2000, 8, 2, 256);
        // Force the bm=32 tile (otherwise only selected on sub-64 KB-shared devices like NVIDIA /
        // MoltenVK) so the small shaders get numeric-parity coverage on any GPU: the fused kernel
        // (hd=64), the warp split-K partial+combine (hd=128), and the non-warp partial
        // (`flash_warp = false`). Without this, a 64 KB device only ever exercises the bm=64 build.
        let bm32 = be_with(|v| v.flash_bm32 = true);
        run_attn_prefill_flash(&bm32, 80, 300, 9, 3, 64); // fused attn_flash_bm32
        run_attn_prefill_flash(&bm32, 128, 200, 4, 2, 128); // warp partial+combine (bm32)
        run_attn_prefill_flash(&bm32, 448, 2000, 16, 8, 128); // warp, multi-block kv
        let bm32_nowarp = be_with(|v| {
            v.flash_bm32 = true;
            v.flash_warp = false;
        });
        run_attn_prefill_flash(&bm32_nowarp, 128, 500, 2, 1, 128); // non-warp attn_flash_partial_bm32
    }

    fn run_attn_prefill_flash(
        be: &VulkanBackend,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(64) * 64;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        let mut kp = k.clone();
        kp.resize((kv_len + 64) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_len + 64) * nkv * hd, 0.0);
        let bq = upf16(be, &qp);
        let bk = upf16(be, &kp);
        let bv = upf16(be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        if hd == 256 {
            rec.attention_prefill_flash_hd256(
                bq.as_ref(),
                bk.as_ref(),
                bv.as_ref(),
                bo.as_ref(),
                po.as_ref(),
                pmb.as_ref(),
                plb.as_ref(),
                q_len,
                kv_len,
                nh,
                nkv,
                pos_offset,
            );
        } else {
            rec.attention_prefill_flash(
                bq.as_ref(),
                bk.as_ref(),
                bv.as_ref(),
                bo.as_ref(),
                po.as_ref(),
                pmb.as_ref(),
                plb.as_ref(),
                q_len,
                kv_len,
                nh,
                nkv,
                hd,
                pos_offset,
                FlashStage::Off,
                None,
            );
        }
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_prefill_flash q_len={q_len} kv_len={kv_len} max_err={err:e}");
        assert!(err < 5e-3, "attn_prefill_flash mismatch: {err}");
    }

    /// Run one `attention_prefill_flash` and return the raw f32 output bytes (for BITWISE compares).
    #[allow(clippy::too_many_arguments)]
    fn run_flash_bytes(
        be: &VulkanBackend,
        bq: &dyn Buffer,
        bk: &dyn Buffer,
        bv: &dyn Buffer,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        stage: FlashStage,
        kv_addr: Option<(u64, u64)>,
    ) -> Vec<u8> {
        let mpad = q_len.div_ceil(64) * 64;
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_flash(
            bq,
            bk,
            bv,
            bo.as_ref(),
            po.as_ref(),
            pmb.as_ref(),
            plb.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            stage,
            kv_addr,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        bytes
    }

    /// Levers 2 & 5 (kv-decode-perf-levers): the shmem-staged `attn_flash_partial` builds compute
    /// BIT-IDENTICALLY to the direct (non-staged) `attn_flash_partial` — they differ ONLY in where
    /// the K/V f16 the coopmat consumes comes from (a direct descriptor coopMatLoad vs a shmem slab
    /// filled by an f16 copy (STAGE) or `native_decode` `dqblk` decode (DEQUANT)); the QK/PV
    /// coopMatMulAdd sequence is the SAME order (kf ascends 0,16,32,48… in both), so anything short
    /// of bitwise equality is a staging/decoding bug, not a tolerance question. Reference is the
    /// DIRECT split-K partial (INFR_NO_FLASH_WARP + forced splits pin Off onto attn_flash_partial,
    /// the same kernel the staged builds fork from — NOT the production warp/fused default, whose
    /// different accumulation order would make a bitwise compare meaningless).
    ///
    /// Lever 5: staged f16 == direct f16. Lever 2: dequant-in-flash on the quant cache == direct on
    /// the dequant_kv_f16 PREPASS output (proving the in-stage `dqblk` decode reproduces the prepass
    /// f16 exactly, so the prepass can be skipped). GQA (nkv<nh) + multi-BN-block kv + non-zero
    /// pos_offset exercise the head/kv-block addressing (the bound coopmat path has no per-tensor
    /// arena base offset — vkb binds the whole cache — so BDA-style base offset-invariance N/A here).
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attn_flash_stage_dequant_parity() {
        use infr_core::DType;
        // Pin Off onto the DIRECT attn_flash_partial (same kernel the staged builds fork from):
        // no register warp, forced split-K partial+combine (n_splits==1 else takes the fused
        // attn_flash). Staged/Dequant force the partial path themselves but read the same splits.
        // A per-backend `Config` value, not an `EnvGuard`: this test and
        // `attn_flash_warp_dequant_parity` both drive `flash_splits`, and while they were env-global
        // one would clear the knob the other was still running under (both failed together, both
        // passed individually). Two backends, two values, no lock, no restore (S5b / R7).
        let be = be_with(|v| {
            v.flash_warp = false;
            v.flash_splits = Some(2);
        });
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        // finite quant block bytes (every f16 scale field's high byte < 0x40 → finite; codes/qh any)
        let synth = |n: usize, seed: usize| -> Vec<u8> {
            (0..n)
                .map(|i| {
                    ((i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7) as u8 % 0x40
                })
                .collect::<Vec<u8>>()
        };
        for &(q_len, kv_len, nh, nkv) in &[
            (128usize, 512usize, 8usize, 2usize),
            (64, 256, 4, 4),
            (128, 320, 16, 8),
        ] {
            let hd = 128usize;
            let pos_offset = kv_len - q_len;
            let mpad = q_len.div_ceil(64) * 64;
            let padrows = kv_len + 64; // Off reads tile-pad rows past kv_len (masked; must be finite)
            let bq = {
                let mut v = r16(&gen(q_len * nh * hd, 1));
                v.resize(mpad * nh * hd, 0.0);
                upf16(&be, &v)
            };
            let bk = {
                let mut v = r16(&gen(kv_len * nkv * hd, 2));
                v.resize(padrows * nkv * hd, 0.0);
                upf16(&be, &v)
            };
            let bv = {
                let mut v = r16(&gen(kv_len * nkv * hd, 3));
                v.resize(padrows * nkv * hd, 0.0);
                upf16(&be, &v)
            };
            let direct = run_flash_bytes(
                &be,
                bq.as_ref(),
                bk.as_ref(),
                bv.as_ref(),
                q_len,
                kv_len,
                nh,
                nkv,
                hd,
                pos_offset,
                FlashStage::Off,
                None,
            );
            // sanity: not vacuously all-zero / NaN (a bitwise compare would pass vacuously otherwise)
            let df: &[f32] = bytemuck::cast_slice(&direct);
            assert!(
                df[..q_len * nh * hd]
                    .iter()
                    .any(|x| x.is_finite() && *x != 0.0),
                "reference flash output degenerate (all zero/NaN)"
            );
            // ---- Lever 5: STAGE (f16) == direct ----
            let staged = run_flash_bytes(
                &be,
                bq.as_ref(),
                bk.as_ref(),
                bv.as_ref(),
                q_len,
                kv_len,
                nh,
                nkv,
                hd,
                pos_offset,
                FlashStage::Stage,
                None,
            );
            assert_eq!(
                direct, staged,
                "INFR_FLASH_STAGE not bit-identical to direct attn_flash_partial \
                 (q={q_len} kv={kv_len} nh={nh} nkv={nkv})"
            );
            // ---- Lever 2: DEQUANT (quant cache) == direct on the dequant_kv_f16 prepass output ----
            let ne = kv_len * nkv * hd;
            for &(dt, blk) in &[
                (DType::Q4_0, 18usize),
                (DType::Q4_1, 20),
                (DType::Q5_0, 22),
                (DType::Q5_1, 24),
            ] {
                let nblk = ne / 32;
                let kq = synth(nblk * blk, 100 + blk);
                let vq = synth(nblk * blk, 200 + blk);
                let bkq = be.alloc(kq.len(), BufferUsage::KvCache).unwrap();
                let bvq = be.alloc(vq.len(), BufferUsage::KvCache).unwrap();
                be.upload(bkq.as_ref(), &kq).unwrap();
                be.upload(bvq.as_ref(), &vq).unwrap();
                // reference: prepass-dequant the quant cache → padded f16 scratch (pad rows stay 0 →
                // finite), then the DIRECT partial reads the scratch.
                let ksc = be
                    .alloc(padrows * nkv * hd * 2, BufferUsage::Activations)
                    .unwrap();
                let vsc = be
                    .alloc(padrows * nkv * hd * 2, BufferUsage::Activations)
                    .unwrap();
                {
                    let rec = be.recorder().unwrap();
                    rec.dequant_kv_f16(dt, bkq.as_ref(), ksc.as_ref(), ne);
                    rec.dequant_kv_f16(dt, bvq.as_ref(), vsc.as_ref(), ne);
                    rec.finish().unwrap();
                }
                let deq_ref = run_flash_bytes(
                    &be,
                    bq.as_ref(),
                    ksc.as_ref(),
                    vsc.as_ref(),
                    q_len,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos_offset,
                    FlashStage::Off,
                    None,
                );
                let deq_test = run_flash_bytes(
                    &be,
                    bq.as_ref(),
                    bkq.as_ref(),
                    bvq.as_ref(),
                    q_len,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos_offset,
                    FlashStage::Dequant(dt),
                    None,
                );
                assert_eq!(
                    deq_ref, deq_test,
                    "INFR_FLASH_DEQUANT {dt:?} not bit-identical to direct+prepass \
                     (q={q_len} kv={kv_len} nh={nh} nkv={nkv})"
                );
                println!("flash dequant parity OK {dt:?} q={q_len} kv={kv_len} nh={nh} nkv={nkv}");
            }
        }
    }

    /// Lever 2 on the WARP kernel (kv-decode-perf-levers): the register-tiled `attn_flash_warp_deq_*`
    /// dequant-in-flash build computes BIT-IDENTICALLY to warp+prepass. Unlike the partial-parity test
    /// above (which pins INFR_NO_FLASH_WARP to compare on `attn_flash_partial`), this leaves warp ON:
    /// both the reference (warp on the dequant_kv_f16 f16 scratch, `Off`) and the test (warp on the
    /// quant cache, `Dequant(dt)`) dispatch `attn_flash_warp[_deq]` — proving the in-stage `dqblk`
    /// decode reproduces the prepass f16 exactly AND the staged QK/PV accumulation order (dcol×kf2 in
    /// QK, kf in PV) matches the direct warp path bit-for-bit, so the warp default can skip the prepass
    /// while keeping the fast kernel. Forced splits exercise the split-K partial+combine reduce.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attn_flash_warp_dequant_parity() {
        use infr_core::DType;
        // Leave warp ON (`flash_warp` default `true`). Force split-K so the warp partial+combine
        // reduce is covered in addition to the n_splits==1 warp path. Its own backend, so the knob
        // cannot collide with `attn_flash_stage_dequant_parity` running in parallel.
        let be = be_with(|v| v.flash_splits = Some(2));
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        // finite quant block bytes (every f16 scale field's high byte < 0x40 → finite; codes/qh any)
        let synth = |n: usize, seed: usize| -> Vec<u8> {
            (0..n)
                .map(|i| {
                    ((i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7) as u8 % 0x40
                })
                .collect::<Vec<u8>>()
        };
        for &(q_len, kv_len, nh, nkv) in &[
            (128usize, 512usize, 8usize, 2usize),
            (64, 256, 4, 4),
            (128, 320, 16, 8),
        ] {
            let hd = 128usize;
            let pos_offset = kv_len - q_len;
            let mpad = q_len.div_ceil(64) * 64;
            let padrows = kv_len + 64; // warp reads tile-pad rows past kv_len (masked; must be finite)
            let bq = {
                let mut v = r16(&gen(q_len * nh * hd, 1));
                v.resize(mpad * nh * hd, 0.0);
                upf16(&be, &v)
            };
            let ne = kv_len * nkv * hd;
            for &(dt, blk) in &[
                (DType::Q4_0, 18usize),
                (DType::Q4_1, 20),
                (DType::Q5_0, 22),
                (DType::Q5_1, 24),
            ] {
                let nblk = ne / 32;
                let kq = synth(nblk * blk, 100 + blk);
                let vq = synth(nblk * blk, 200 + blk);
                let bkq = be.alloc(kq.len(), BufferUsage::KvCache).unwrap();
                let bvq = be.alloc(vq.len(), BufferUsage::KvCache).unwrap();
                be.upload(bkq.as_ref(), &kq).unwrap();
                be.upload(bvq.as_ref(), &vq).unwrap();
                // reference: prepass-dequant the quant cache → padded f16 scratch (pad rows 0 →
                // finite), then warp (Off) reads the scratch.
                let ksc = be
                    .alloc(padrows * nkv * hd * 2, BufferUsage::Activations)
                    .unwrap();
                let vsc = be
                    .alloc(padrows * nkv * hd * 2, BufferUsage::Activations)
                    .unwrap();
                {
                    let rec = be.recorder().unwrap();
                    rec.dequant_kv_f16(dt, bkq.as_ref(), ksc.as_ref(), ne);
                    rec.dequant_kv_f16(dt, bvq.as_ref(), vsc.as_ref(), ne);
                    rec.finish().unwrap();
                }
                let warp_ref = run_flash_bytes(
                    &be,
                    bq.as_ref(),
                    ksc.as_ref(),
                    vsc.as_ref(),
                    q_len,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos_offset,
                    FlashStage::Off,
                    None,
                );
                // sanity: not vacuously all-zero / NaN
                let rf: &[f32] = bytemuck::cast_slice(&warp_ref);
                assert!(
                    rf[..q_len * nh * hd]
                        .iter()
                        .any(|x| x.is_finite() && *x != 0.0),
                    "reference warp flash output degenerate (all zero/NaN)"
                );
                let warp_deq = run_flash_bytes(
                    &be,
                    bq.as_ref(),
                    bkq.as_ref(),
                    bvq.as_ref(),
                    q_len,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos_offset,
                    FlashStage::Dequant(dt),
                    None,
                );
                assert_eq!(
                    warp_ref, warp_deq,
                    "warp INFR_FLASH_DEQUANT {dt:?} not bit-identical to warp+prepass \
                     (q={q_len} kv={kv_len} nh={nh} nkv={nkv})"
                );
                println!(
                    "warp flash dequant parity OK {dt:?} q={q_len} kv={kv_len} nh={nh} nkv={nkv}"
                );
            }
        }
    }

    /// Upload f16 K/V into a `KvCache` buffer (device-addressed → `device_addr()` is `Some`), so the
    /// coopmat flash-prefill `-DKV_COOPMAT_BDA` builds can read it by 64-bit pointer.
    fn upf16_kv(be: &VulkanBackend, v: &[f32]) -> Box<dyn Buffer> {
        let bits: Vec<u16> = v
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let b = be.alloc(bits.len() * 2, BufferUsage::KvCache).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(&bits)).unwrap();
        b
    }

    /// Run one `attention_prefill_flash_reg` and return the raw f32 output bytes (BITWISE compares).
    #[allow(clippy::too_many_arguments)]
    fn run_flash_reg_bytes(
        be: &VulkanBackend,
        bq: &dyn Buffer,
        bk: &dyn Buffer,
        bv: &dyn Buffer,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        pos_offset: usize,
        kv_addr: Option<(u64, u64)>,
    ) -> Vec<u8> {
        let mpad = q_len.div_ceil(128) * 128;
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_flash_reg(
            bq,
            bk,
            bv,
            bo.as_ref(),
            po.as_ref(),
            pmb.as_ref(),
            plb.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            kv_addr,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        bytes
    }

    /// #74 slice 4 resurrection: the coopmat flash-prefill `-DKV_COOPMAT_BDA` builds (K/V read by 64-bit
    /// device address via kv_addr.glsl `KvMat(k_addr)/(v_addr)`) compute BIT-IDENTICALLY to the bound
    /// descriptor default — the ONLY change is the KV source (a bound-SSBO coopMatLoad vs a
    /// buffer_reference coopMatLoad off the same bytes), the QK/PV coopMatMulAdd order is untouched, so
    /// anything short of bytewise equality is an addressing bug. This is the correctness bar for the
    /// opt-in flag (on RADV/RDNA3 the BDA arm merely regresses perf; it must never change results).
    /// Covers the hd==128 coopmat-prefill kernels gpu_seam hits: the register-O warp default
    /// (`attn_flash_warp`), the split-K partial reduce, the non-warp `attn_flash_partial`, the bm=32
    /// tile, and the (test-only) register-O `attn_flash_reg`. GQA (nkv<nh), multi-BN-block kv, and a
    /// non-zero pos_offset exercise the head/kv addressing the BDA offset must reproduce exactly.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attn_flash_coopmat_bda_parity() {
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        // The four `attention_prefill_flash` build selections, each as its own backend + tier
        // config. Pre-S5b these were four `EnvGuard` cases over ONE backend; the tiers are now
        // resolved at construction, so a case IS a backend (R7 — no env, no serialization).
        #[allow(clippy::type_complexity)]
        let cases: Vec<(VulkanBackend, &str)> = vec![
            (be_with(|_| {}), "warp (n_splits default)"),
            (
                be_with(|v| v.flash_splits = Some(2)),
                "warp split-K partial+combine",
            ),
            (
                be_with(|v| {
                    v.flash_warp = false;
                    v.flash_splits = Some(2);
                }),
                "non-warp attn_flash_partial",
            ),
            (
                be_with(|v| {
                    v.flash_bm32 = true;
                    v.flash_splits = Some(2);
                }),
                "bm=32 tile",
            ),
        ];
        for &(q_len, kv_len, nh, nkv) in &[
            (128usize, 512usize, 8usize, 2usize),
            (64, 256, 4, 4),
            (128, 320, 16, 8),
            (192, 500, 2, 1),
        ] {
            let hd = 128usize;
            let pos_offset = kv_len - q_len;
            // `padrows`: pad rows read past kv_len (masked; must be finite/zero).
            let padrows = kv_len + 128;
            // Buffers belong to the backend that allocated them, so each case makes its own set
            // from the SAME synthetic bytes.
            #[allow(clippy::type_complexity)]
            let mk = |be: &VulkanBackend| -> (
                Box<dyn Buffer>,
                Box<dyn Buffer>,
                Box<dyn Buffer>,
                Option<(u64, u64)>,
            ) {
                let bq = {
                    let mut v = r16(&gen(q_len * nh * hd, 1));
                    v.resize(q_len.div_ceil(128) * 128 * nh * hd, 0.0);
                    upf16(be, &v)
                };
                // K/V in KvCache buffers so both bound and BDA reads hit the same bytes.
                let bk = {
                    let mut v = r16(&gen(kv_len * nkv * hd, 2));
                    v.resize(padrows * nkv * hd, 0.0);
                    upf16_kv(be, &v)
                };
                let bv = {
                    let mut v = r16(&gen(kv_len * nkv * hd, 3));
                    v.resize(padrows * nkv * hd, 0.0);
                    upf16_kv(be, &v)
                };
                let ka = bk
                    .device_addr()
                    .expect("KvCache K must have a device address");
                let va = bv
                    .device_addr()
                    .expect("KvCache V must have a device address");
                (bq, bk, bv, Some((ka, va)))
            };

            for (be, label) in &cases {
                let (bq, bk, bv, addr) = mk(be);
                let bound = run_flash_bytes(
                    be,
                    bq.as_ref(),
                    bk.as_ref(),
                    bv.as_ref(),
                    q_len,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos_offset,
                    FlashStage::Off,
                    None,
                );
                let bda = run_flash_bytes(
                    be,
                    bq.as_ref(),
                    bk.as_ref(),
                    bv.as_ref(),
                    q_len,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos_offset,
                    FlashStage::Off,
                    addr,
                );
                let bf: &[f32] = bytemuck::cast_slice(&bound);
                assert!(
                    bf[..q_len * nh * hd]
                        .iter()
                        .any(|x| x.is_finite() && *x != 0.0),
                    "reference flash output degenerate (all zero/NaN) [{label}]"
                );
                assert_eq!(
                    bound, bda,
                    "INFR_KV_COOPMAT_BDA not bit-identical to bound flash [{label}] \
                     (q={q_len} kv={kv_len} nh={nh} nkv={nkv})"
                );
                println!(
                    "flash coopmat BDA parity OK [{label}] q={q_len} kv={kv_len} nh={nh} nkv={nkv}"
                );
            }

            // The (test-only) register-O `attn_flash_reg` kernel, bound vs BDA, on the DEFAULT tier.
            let be = &cases[0].0;
            let (bq, bk, bv, addr) = mk(be);
            let reg_bound = run_flash_reg_bytes(
                be,
                bq.as_ref(),
                bk.as_ref(),
                bv.as_ref(),
                q_len,
                kv_len,
                nh,
                nkv,
                hd,
                pos_offset,
                None,
            );
            let reg_bda = run_flash_reg_bytes(
                be,
                bq.as_ref(),
                bk.as_ref(),
                bv.as_ref(),
                q_len,
                kv_len,
                nh,
                nkv,
                hd,
                pos_offset,
                addr,
            );
            assert_eq!(
                reg_bound, reg_bda,
                "INFR_KV_COOPMAT_BDA not bit-identical to bound attn_flash_reg \
                 (q={q_len} kv={kv_len} nh={nh} nkv={nkv})"
            );
            println!("flash_reg coopmat BDA parity OK q={q_len} kv={kv_len} nh={nh} nkv={nkv}");
        }
    }

    /// Non-coopmat fma flash prefill (`attn_nc_fa`): host-reference parity across causal /
    /// windowed / ring-wrapped / GQA configs on all three HD_MAX builds, PLUS same-dispatch-3x
    /// bitwise determinism (the mmq barrier-race lesson — goldens can't catch intra-dispatch
    /// races). Runs on any GPU: the kernel uses no coopmat and no subgroup ops.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_nc_fa_matches_cpu() {
        // causal, GQA, rows not a tile multiple (pad-row guards), hd128 build
        run_attn_nc_fa(100, 160, 4, 2, 128, 0, 0.0, None);
        // multi-block kv, qwen3-shaped
        run_attn_nc_fa(448, 2000, 16, 8, 128, 0, 0.0, None);
        // sliding window on a full-context cache + non-default scale (gemma4 pushes 1.0)
        run_attn_nc_fa(100, 300, 4, 2, 128, 64, 1.0, None);
        // hd=64 (BK tail guards: hd < the build's HD_MAX)
        run_attn_nc_fa(80, 300, 9, 3, 64, 0, 0.0, None);
        // hd256 build, windowed (gemma4 SWA shape)
        run_attn_nc_fa(96, 400, 8, 4, 256, 128, 1.0, None);
        // hd512 build (BM=16 tile)
        run_attn_nc_fa(70, 200, 2, 1, 512, 0, 0.0, None);
        // RING cache wrapped past capacity (kv_len > ring rows): window 64, ring 192 —
        // union span (kvend - first tile's window base) = 163 <= 192, the runner's invariant
        run_attn_nc_fa(100, 260, 4, 2, 128, 64, 0.0, Some(192));
        // ring + hd256 + gemma scale
        run_attn_nc_fa(96, 288, 8, 4, 256, 96, 1.0, Some(192));
        // gemma-4-12b mid-prefill ubatch chunk, EXACT runner shapes (INFR_UBATCH=256): SWA layer
        // (hd 256, window 1024, ring = round64(window+ubatch) = 1280, wrapped: kv_len 1536)…
        run_attn_nc_fa(256, 1536, 16, 8, 256, 1024, 1.0, Some(1280));
        // …and the paired global layer (hd 512, BM=16 build, full-context cache, deep pos)
        run_attn_nc_fa(256, 1536, 16, 1, 512, 0, 1.0, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attn_nc_fa(
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
        scale: f32,
        ring_rows: Option<usize>,
    ) {
        let be = VulkanBackend::new().unwrap();
        let pos_offset = kv_len - q_len;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        // Build the cache buffers in RING layout: position j lives at row j % rcap, ascending j
        // so later positions overwrite recycled rows (exactly what the runner's ring writes do).
        // rcap = kv_len for the full-context configs (the modulo is an identity there).
        let rcap = ring_rows.unwrap_or(kv_len);
        let rw = nkv * hd;
        let mut kr = vec![0f32; rcap * rw];
        let mut vr = vec![0f32; rcap * rw];
        for j in 0..kv_len {
            let r = j % rcap;
            kr[r * rw..(r + 1) * rw].copy_from_slice(&k[j * rw..(j + 1) * rw]);
            vr[r * rw..(r + 1) * rw].copy_from_slice(&v[j * rw..(j + 1) * rw]);
        }
        let bq = upf16(&be, &q);
        let bk = upf16(&be, &kr);
        let bv = upf16(&be, &vr);
        // dst is NOT tile-padded — the kernel guards pad-row writes (no Internal requirement)
        let outs: Vec<_> = (0..3)
            .map(|_| {
                be.alloc(q_len * nh * hd * 4, BufferUsage::Readback)
                    .unwrap()
            })
            .collect();
        let cap = rcap * rw;
        let rec = be.recorder().unwrap();
        for o in &outs {
            rec.attention_prefill_nc_fa(
                bq.as_ref(),
                bk.as_ref(),
                bv.as_ref(),
                o.as_ref(),
                q_len,
                kv_len,
                nh,
                nkv,
                hd,
                pos_offset,
                window,
                scale,
                cap,
            );
        }
        rec.finish().unwrap();
        let dl = |b: &dyn Buffer| -> Vec<f32> {
            let mut bytes = vec![0u8; q_len * nh * hd * 4];
            be.download(b, &mut bytes).unwrap();
            bytemuck::cast_slice(&bytes).to_vec()
        };
        let got = dl(outs[0].as_ref());
        assert!(
            got.iter().all(|x| x.is_finite()),
            "attn_nc_fa q={q_len} kv={kv_len} hd={hd}: non-finite output"
        );
        for (run, o) in outs.iter().enumerate().skip(1) {
            let other = dl(o.as_ref());
            let ndiff = got.iter().zip(&other).filter(|(a, b)| a != b).count();
            assert_eq!(
                ndiff, 0,
                "attn_nc_fa q={q_len} kv={kv_len} hd={hd} ring={ring_rows:?}: \
                 nondeterministic — dispatch 0 vs {run}: {ndiff} differ"
            );
        }
        // reference over the LOGICAL (position-ordered) K/V — the ring layout must be invisible
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, window, scale);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "attn_nc_fa q={q_len} kv={kv_len} nh={nh}/{nkv} hd={hd} win={window} \
             ring={ring_rows:?} max_err={err:e}"
        );
        assert!(err < 5e-3, "attn_nc_fa mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn attention_prefill_flash_reg_matches_cpu() {
        let be = be_with(|_| {});
        for &(q, kv, nh, nkv) in &[
            (128usize, 128usize, 4usize, 2usize),
            (128, 300, 2, 1),
            (200, 600, 8, 4),
            (448, 2000, 16, 8), // qwen3-shaped
            (100, 100, 2, 2),   // q<128 → padded tile
        ] {
            run_attn_flash_reg(&be, q, kv, nh, nkv, 128);
        }
        let split = be_with(|v| v.flash_splits = Some(4));
        run_attn_flash_reg(&split, 128, 2000, 16, 8, 128);
        run_attn_flash_reg(&split, 200, 600, 2, 1, 128);
    }

    fn run_attn_flash_reg(
        be: &VulkanBackend,
        q_len: usize,
        kv_len: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        let pos_offset = kv_len - q_len;
        let mpad = q_len.div_ceil(128) * 128;
        let gen = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
                .collect()
        };
        let q = r16(&gen(q_len * nh * hd, 1));
        let k = r16(&gen(kv_len * nkv * hd, 2));
        let v = r16(&gen(kv_len * nkv * hd, 3));
        let mut qp = q.clone();
        qp.resize(mpad * nh * hd, 0.0);
        let mut kp = k.clone();
        kp.resize((kv_len + 128) * nkv * hd, 0.0);
        let mut vp = v.clone();
        vp.resize((kv_len + 128) * nkv * hd, 0.0);
        let bq = upf16(be, &qp);
        let bk = upf16(be, &kp);
        let bv = upf16(be, &vp);
        let bo = be.alloc(mpad * nh * hd * 4, BufferUsage::Readback).unwrap();
        let po = be
            .alloc(8 * mpad * nh * hd * 4, BufferUsage::Activations)
            .unwrap();
        let pmb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let plb = be
            .alloc(8 * mpad * nh * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.attention_prefill_flash_reg(
            bq.as_ref(),
            bk.as_ref(),
            bv.as_ref(),
            bo.as_ref(),
            po.as_ref(),
            pmb.as_ref(),
            plb.as_ref(),
            q_len,
            kv_len,
            nh,
            nkv,
            hd,
            pos_offset,
            None,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * nh * hd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let want = attn_kv_cpu(&q, &k, &v, q_len, nh, nkv, hd, pos_offset, 0, 0.0);
        let err = got[..q_len * nh * hd]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn_flash_reg q_len={q_len} kv_len={kv_len} nh={nh} max_err={err:e}");
        assert!(err < 5e-3, "attn_flash_reg mismatch: {err}");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn native_gemm_mmq_q6k_matches_cpu() {
        // Q6_K dp4a GEMM vs a CPU reference: build a synthetic Q6_K weight (q6 ∈ 0..63, i8 sub-scale
        // per 16, f16 super-scale per 256), pack it into the 210-byte block layout, and check the
        // int8-dot GEMM matches the dequantized matmul. k must be a multiple of 256 (a Q6_K superblock)
        // so each column's k packs into whole blocks — exactly the real (down-proj) layout.
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (70usize, 256usize, 128usize); // m not %64 → padding; k = 1 superblock; n%64
        let nblk = k / 32;
        let mpad = m.div_ceil(64) * 64;
        let nsb = k / 256; // superblocks per column
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
            .collect();
        // q6 quants, per-16 i8 scales, per-256 f16 super-scale (weight g = col*k + kk).
        // NOT 64-periodic: the original `i*13 % 64` satisfied q6[p+64]==q6[p], which made the
        // shader's og=1/og=2 ql-source transposition (fixed 2026-07, see the .comp) invisible —
        // the `(i/64)*31` term breaks the period so a nibble-map regression fails loudly here.
        let q6: Vec<u32> = (0..n * k)
            .map(|i| ((i * 13 + (i / 64) * 31) % 64) as u32)
            .collect();
        let sc: Vec<i8> = (0..n * k / 16)
            .map(|b| ((b % 11) as i32 - 5) as i8)
            .collect();
        let d: Vec<half::f16> = (0..n * nsb)
            .map(|b| half::f16::from_f32(0.008 + (b % 5) as f32 * 0.001))
            .collect();
        // pack into 210-byte Q6_K blocks: [ql[128]][qh[64]][i8 scales[16]][f16 d]
        let mut blk = vec![0u8; n * nsb * 210];
        let sc_index = |g: usize| -> usize {
            // global i8-scale index for weight g, mirroring the shader's sc_idx within the superblock
            let p = g % 256;
            let (hf, ph) = (p / 128, p % 128);
            let (og, l) = (ph / 32, ph % 32);
            (g / 256) * 16 + hf * 8 + l / 16 + 2 * og
        };
        for (col, _) in (0..n).map(|c| (c, ())) {
            for sbk in 0..nsb {
                let base = (col * nsb + sbk) * 210;
                for p in 0..256usize {
                    let g = col * k + sbk * 256 + p;
                    let q = q6[g];
                    let (hf, ph) = (p / 128, p % 128);
                    let (og, l) = (ph / 32, ph % 32);
                    let lo = hf * 64;
                    let qh_byte = base + 128 + hf * 32 + l;
                    match og {
                        0 => blk[base + lo + l] |= (q & 0xF) as u8,
                        1 => blk[base + lo + l + 32] |= (q & 0xF) as u8,
                        2 => blk[base + lo + l] |= ((q & 0xF) << 4) as u8,
                        _ => blk[base + lo + l + 32] |= ((q & 0xF) << 4) as u8,
                    }
                    blk[qh_byte] |= (((q >> 4) & 3) << (2 * og)) as u8;
                }
                // i8 scales at +192, f16 d at +208
                for s in 0..16 {
                    blk[base + 192 + s] = sc[(col * nsb + sbk) * 16 + s] as u8;
                }
                let db = d[col * nsb + sbk].to_bits().to_le_bytes();
                blk[base + 208] = db[0];
                blk[base + 209] = db[1];
            }
        }
        let dq = |g: usize| -> f32 {
            d[g / 256].to_f32() * sc[sc_index(g)] as f32 * (q6[g] as f32 - 32.0)
        };

        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let bw = be.upload_weight_bytes(&blk).unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let qa = be.alloc(mpad * k, BufferUsage::Activations).unwrap();
        let dact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(mpad * nblk * 2, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.quant_q8(ba.as_ref(), qa.as_ref(), dact.as_ref(), sact.as_ref(), m, k);
        rec.matmul_mmq_q6k(
            qa.as_ref(),
            dact.as_ref(),
            bw.as_ref(),
            0,
            bc.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * dq(col * k + x)).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("native_gemm_mmq_q6k max_err={e:e}");
        assert!(e < 2e-2, "native_gemm_mmq_q6k mismatch: {e}"); // int8 activation quant tolerance
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn matmul_proj_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (800usize, 64usize, 256usize); // m≥768 & not %64 → warp path + padding; n%256
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        // weight W[N,K] (row-major [out,in]), f16-rounded
        let w: Vec<f32> = (0..n * k)
            .map(|i| half::f16::from_f32(((i * 13 % 23) as f32 - 11.0) * 0.02).to_f32())
            .collect();
        let upf = |v: &[f32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Staging).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let ba = upf(&a);
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let cpu = |label: &str, c: &[f32]| {
            let mut e = 0f32;
            for r in 0..m {
                for col in 0..n {
                    let want: f32 = (0..k).map(|x| a[r * k + x] * w[col * k + x]).sum();
                    e = e.max((c[r * n + col] - want).abs());
                }
            }
            println!("matmul_proj {label} max_err={e:e}");
            assert!(e < 5e-3, "matmul_proj {label} mismatch: {e}");
        };

        // --- f16 weights ---
        let wf16: Vec<u16> = w
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let bw = be.upload_weight_bytes(bytemuck::cast_slice(&wf16)).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_proj(ba.as_ref(), bw.as_ref(), bc.as_ref(), m, k, n);
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        cpu("f16", bytemuck::cast_slice(&bytes));
    }

    /// The F16-weight SPLIT-K warptile (DG slice-7: the SC soft-embedding GEMM's route — deep k,
    /// narrow n, m below the wide-warp gate) vs a host reference on the SAME f16-rounded weights.
    /// m=70 (not %64) exercises the padded-row store; splits=4 exercises the partial planes +
    /// fixed-order reduce; k=2048 gives each split multiple BK=64 stages.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn matmul_native_splitk_f16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n, splits) = (70usize, 2048usize, 256usize, 4usize);
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| half::f16::from_f32(((i * 13 % 23) as f32 - 11.0) * 0.02).to_f32())
            .collect();
        let wf16: Vec<u16> = w
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let ba = be.alloc(a.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(ba.as_ref(), bytemuck::cast_slice(&a)).unwrap();
        let bw = be.upload_weight_bytes(bytemuck::cast_slice(&wf16)).unwrap();
        let pk = be
            .alloc(splits * mpad * n * 4, BufferUsage::Activations)
            .unwrap();
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_native_splitk(
            infr_core::DType::F16,
            ba.as_ref(),
            bw.device_addr().unwrap(),
            0,
            pk.as_ref(),
            bc.as_ref(),
            m,
            k,
            n,
            splits,
            false,
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; mpad * n * 4];
        be.download(bc.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        let mut e = 0f32;
        for r in 0..m {
            for col in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * w[col * k + x]).sum();
                e = e.max((got[r * n + col] - want).abs());
            }
        }
        println!("matmul_native_splitk f16 max_err={e:e}");
        assert!(e < 5e-2, "matmul_native_splitk f16 mismatch: {e}"); // f16 A rounding at k=2048
    }

    /// The IQ4_NL warp-GEMM family vs a CPU dequant reference — wide (BN=256), n128, n128_ag,
    /// sk, and sk_ag tiles. Synthetic IQ4_NL blocks ([f16 d][u8 qs[16]] = 18 bytes / 32 elems,
    /// value = d·kvalues[nibble]); m=70 (not %64) exercises padded rows, k=1024 gives split-K
    /// multiple BK stages, n=16384 fills the wide-tile grid gate (ceil(70/64)·(16384/256) = 128).
    /// The host reference rounds A to f16 first — every warp tile does (stage or A_GLOBAL cast).
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn native_gemm_warp_iq4nl_matches_cpu() {
        const KV: [i32; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let be = VulkanBackend::new().unwrap();
        let (m, k, n_max) = (70usize, 1024usize, 16384usize);
        let mpad = m.div_ceil(64) * 64;
        let a: Vec<f32> = (0..m * k)
            .map(|i| half::f16::from_f32(((i % 23) as f32 - 11.0) * 0.04).to_f32())
            .collect();
        // weight element g = col*k + kk; blocks packed col-major contiguous
        let q: Vec<u32> = (0..n_max * k).map(|i| (i * 7 % 16) as u32).collect();
        let nblk = n_max * k / 32;
        let d: Vec<half::f16> = (0..nblk)
            .map(|b| half::f16::from_f32(0.01 + (b % 7) as f32 * 0.002))
            .collect();
        let mut blk = vec![0u8; nblk * 18];
        for b in 0..nblk {
            let db = d[b].to_bits().to_le_bytes();
            blk[b * 18] = db[0];
            blk[b * 18 + 1] = db[1];
            for j in 0..16 {
                blk[b * 18 + 2 + j] = (q[b * 32 + j] | (q[b * 32 + 16 + j] << 4)) as u8;
            }
        }
        let dq = |g: usize| -> f32 { d[g / 32].to_f32() * KV[q[g] as usize] as f32 };

        let ba = be.alloc(a.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(ba.as_ref(), bytemuck::cast_slice(&a)).unwrap();
        let bw = be.upload_weight_bytes(&blk).unwrap();
        let a16 = be.alloc(mpad * k * 2, BufferUsage::Activations).unwrap();
        {
            let rec = be.recorder().unwrap();
            rec.store_f16(ba.as_ref(), a16.as_ref(), m * k, 0);
            rec.finish().unwrap();
        }
        let check = |bc: &dyn Buffer, n: usize, label: &str| {
            let mut bytes = vec![0u8; mpad * n * 4];
            be.download(bc, &mut bytes).unwrap();
            let got: &[f32] = bytemuck::cast_slice(&bytes);
            let mut e = 0f32;
            for r in 0..m {
                for col in 0..n {
                    let want: f32 = (0..k).map(|x| a[r * k + x] * dq(col * k + x)).sum();
                    e = e.max((got[r * n + col] - want).abs());
                }
            }
            println!("native_gemm_warp_iq4nl {label} max_err={e:e}");
            assert!(e < 5e-2, "native_gemm_warp_iq4nl {label} mismatch: {e}");
        };

        // wide BN=256 tile (grid gate: ceil(70/64)·(16384/256) = 128 ≥ 128)
        let bc = be.alloc(mpad * n_max * 4, BufferUsage::Readback).unwrap();
        {
            let rec = be.recorder().unwrap();
            rec.matmul_native(
                infr_core::DType::Iq4Nl,
                ba.as_ref(),
                bw.as_ref(),
                bc.as_ref(),
                m,
                k,
                n_max,
            );
            rec.finish().unwrap();
        }
        check(bc.as_ref(), n_max, "wide");
        // n128 tile (n=128: below the wide gate, n%128==0)
        let n = 128usize;
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        {
            let rec = be.recorder().unwrap();
            rec.matmul_native(
                infr_core::DType::Iq4Nl,
                ba.as_ref(),
                bw.as_ref(),
                bc.as_ref(),
                m,
                k,
                n,
            );
            rec.finish().unwrap();
        }
        check(bc.as_ref(), n, "n128");
        // n128_ag (A pre-cast to f16)
        let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
        {
            let rec = be.recorder().unwrap();
            rec.matmul_native_f16a(
                infr_core::DType::Iq4Nl,
                a16.as_ref(),
                bw.device_addr().unwrap(),
                0,
                bc.as_ref(),
                m,
                k,
                n,
            );
            rec.finish().unwrap();
        }
        check(bc.as_ref(), n, "n128_ag");
        // sk (split-K, f32 A) and sk_ag (split-K, f16 A)
        let splits = 4usize;
        let pk = be
            .alloc(splits * mpad * n * 4, BufferUsage::Activations)
            .unwrap();
        for (a_is_f16, label) in [(false, "sk"), (true, "sk_ag")] {
            let bc = be.alloc(mpad * n * 4, BufferUsage::Readback).unwrap();
            let ab: &dyn Buffer = if a_is_f16 { a16.as_ref() } else { ba.as_ref() };
            let rec = be.recorder().unwrap();
            rec.matmul_native_splitk(
                infr_core::DType::Iq4Nl,
                ab,
                bw.device_addr().unwrap(),
                0,
                pk.as_ref(),
                bc.as_ref(),
                m,
                k,
                n,
                splits,
                a_is_f16,
            );
            rec.finish().unwrap();
            check(bc.as_ref(), n, label);
        }
    }

    // ── chunked dense GEMV parity (issue #77) ────────────────────────────────────────────────────
    // Proves the DISPATCH-level output-row chunking (`dispatch_gemv_chunked`) is BITWISE-identical
    // to the single unchunked dispatch of the SAME kernel — the whole guarantee that lets a
    // `>= 2^32`-element lm_head/embed run without widening any in-kernel index. Forces chunking on a
    // small synthetic weight via the thread-local cap override (`with_bda_chunk_caps`), so it needs
    // no `>4 GiB` tensor. Runs under the validation layer with zero VUIDs (same as the sibling
    // weight_addr_parity suite). Covers m=1 (decode, the always-run path) and m>1 (the per-input-row
    // decompose), a remainder tail (non-divisible row count), and the non-zero arena offset.
    fn synth_w_bytes(n: usize, seed: usize) -> Vec<u8> {
        (0..n)
            .map(|i| {
                let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
                (h % 0x40) as u8
            })
            .collect()
    }

    /// One chunked-vs-unchunked leg: dispatch `linear_native_at` unchunked (production caps) and
    /// chunked (a forced elem cap of ~`chunk_rows` rows) against the SAME arena weight, return both
    /// output-bit vectors. `arena_prefix` parks the weight behind a garbage prefix at a non-zero
    /// arena offset (exercises the folded chunk base + the address split together).
    fn chunked_gemv_bits(
        be: &VulkanBackend,
        dtype: infr_core::DType,
        rows: usize,
        in_f: usize,
        out_f: usize,
        chunk_rows: usize,
        arena_prefix: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        use infr_core::backend::BufferUsage;
        let (be_blk, bb) = infr_gguf::block_layout(dtype);
        let w_bytes = (in_f * out_f) / be_blk * bb;
        let w = synth_w_bytes(w_bytes, in_f * 31 + out_f * 7 + rows);
        let x: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let x_buf = be.alloc(rows * in_f * 4, BufferUsage::Activations).unwrap();
        be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        // Weight parked at a block-aligned, 256-aligned non-zero offset in a shared arena.
        let prefix = arena_prefix.div_ceil(bb.max(256)) * bb.max(256);
        let prefix = prefix.div_ceil(256) * 256;
        let mut backing = synth_w_bytes(prefix, 0xBAD);
        backing.extend_from_slice(&w);
        let (arena, addr0) = be.alloc_arena_bda(backing.len()).unwrap();
        be.upload(arena.as_ref(), &backing).unwrap();
        let addr = addr0 + prefix as u64;

        let run = |chunked: bool| -> Vec<u32> {
            let y = be
                .alloc(rows * out_f * 4, BufferUsage::Activations)
                .unwrap();
            let rec = be.recorder().unwrap();
            let go = || {
                rec.linear_native_at(
                    dtype,
                    addr,
                    0,
                    x_buf.as_ref(),
                    y.as_ref(),
                    rows,
                    in_f,
                    out_f,
                )
            };
            if chunked {
                // Force ~chunk_rows rows/chunk (byte cap left wide so the element cap drives it).
                with_bda_chunk_caps((chunk_rows * in_f + 1) as u64, u64::MAX, go);
            } else {
                go();
            }
            rec.finish().unwrap();
            let mut out = vec![0u8; rows * out_f * 4];
            be.download(y.as_ref(), &mut out).unwrap();
            bytemuck::cast_slice::<u8, u32>(&out).to_vec()
        };
        (run(false), run(true))
    }

    fn assert_chunk_parity(
        be: &VulkanBackend,
        dtype: infr_core::DType,
        rows: usize,
        in_f: usize,
        out_f: usize,
        chunk_rows: usize,
        off: usize,
    ) {
        // Sanity: this shape MUST actually split under the forced cap (else the test proves nothing).
        let chunks = bda_out_f_chunks(
            dtype,
            in_f,
            out_f,
            0,
            (chunk_rows * in_f + 1) as u64,
            u64::MAX,
        )
        .unwrap();
        assert!(
            chunks.len() >= 2,
            "{dtype:?} r{rows} {in_f}x{out_f}: forced cap did not split (got {} chunk)",
            chunks.len()
        );
        let (unchunked, chunked) = chunked_gemv_bits(be, dtype, rows, in_f, out_f, chunk_rows, off);
        assert!(
            unchunked.iter().any(|&b| b != 0),
            "{dtype:?} r{rows} {in_f}x{out_f}: unchunked output is all zeros — kernel not exercised"
        );
        assert_eq!(
            unchunked,
            chunked,
            "{dtype:?} r{rows} {in_f}x{out_f} chunk_rows={chunk_rows} off={off}: chunked dispatch \
             ({} chunks) is NOT bitwise-identical to the unchunked dispatch",
            chunks.len()
        );
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn chunked_dense_gemv_matches_unchunked() {
        let be = VulkanBackend::new().unwrap();
        for dtype in [
            infr_core::DType::Q4K,
            infr_core::DType::Q6K,
            infr_core::DType::Q8_0,
        ] {
            // m=1 decode, even split (256 rows → 4×64) at arena offset 0.
            assert_chunk_parity(&be, dtype, 1, 512, 256, 64, 0);
            // m=1 decode, REMAINDER tail (200 rows → 64,64,64,8), non-zero arena offset.
            assert_chunk_parity(&be, dtype, 1, 512, 200, 64, 4096);
            // m>1 (per-input-row decompose), out_f 64-aligned, non-zero offset.
            assert_chunk_parity(&be, dtype, 3, 512, 256, 64, 8192);
        }
    }

    /// B44: `moe_topk`'s softmax weight branch must shift by the max unbiased logit over the WHOLE
    /// selection, not over the first pick.
    ///
    /// A router bias breaks the "first pick carries the largest logit" assumption — selection is by
    /// `logit + bias`, weights come from the unbiased logits — so the first pick can be an expert
    /// whose logit is far BELOW another selected one. Shifting by the first pick's logit then makes
    /// `exp(logit - shift)` overflow to `+inf`, and under `norm_w` the renormalization divides
    /// `inf/inf` = NaN. The gap here (0 vs 100) is synthetic: this is the hazard the shift protects
    /// against, made large enough to actually cross f32's `exp` range (~88.7), so the test fails
    /// with a NaN weight on a kernel that shifts by the wrong max.
    ///
    /// Under renormalization the shift otherwise cancels algebraically, so a moderate gap could not
    /// tell the two maxima apart at all.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn moe_topk_softmax_bias_shifts_by_selection_max() {
        let be = be_with(|_| {});
        let (n_tokens, n_expert, n_used) = (1usize, 8usize, 2usize);
        // Expert 0 wins selection on bias alone; expert 1 carries the far larger logit.
        let mut logits = vec![-50.0f32; n_expert];
        logits[0] = 0.0;
        logits[1] = 100.0;
        let mut bias = vec![0.0f32; n_expert];
        bias[0] = 200.0;
        let blog = upf32(&be, &logits);
        let bbias = upf32(&be, &bias);
        let bids = be
            .alloc(n_tokens * n_used * 4, BufferUsage::Readback)
            .unwrap();
        let bwts = be
            .alloc(n_tokens * n_used * 4, BufferUsage::Readback)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.moe_topk(
            blog.as_ref(),
            bids.as_ref(),
            bwts.as_ref(),
            bbias.as_ref(),
            n_tokens,
            n_expert,
            n_used,
            1.0,            // scale
            0,              // gating: softmax
            true,           // norm_w — the branch that used to read the first-pick max
            true,           // has_bias
            0,              // n_expert_groups
            0,              // n_expert_groups_used
            bbias.as_ref(), // hash_ids: unused dummy (hash == false)
            false,          // hash
        );
        rec.finish().unwrap();
        let mut idb = vec![0u8; n_tokens * n_used * 4];
        be.download(bids.as_ref(), &mut idb).unwrap();
        let ids: &[u32] = bytemuck::cast_slice(&idb);
        let mut wb = vec![0u8; n_tokens * n_used * 4];
        be.download(bwts.as_ref(), &mut wb).unwrap();
        let wts: &[f32] = bytemuck::cast_slice(&wb);
        // Selection is by `logit + bias`: 200 beats 100 beats the rest.
        assert_eq!(ids, &[0u32, 1], "selection order changed: {ids:?}");
        // Weights come from the UNBIASED logits, renormalized over the selection: expert 1's 100
        // dominates expert 0's 0, so the pair is ~[0, 1].
        assert!(
            wts.iter().all(|w| w.is_finite()),
            "softmax overflowed — weights {wts:?} (shift taken from the first pick, not the \
             selection max)"
        );
        assert!(
            wts[0] < 1e-30,
            "expected ~0 for the low-logit pick: {wts:?}"
        );
        assert!(
            (wts[1] - 1.0).abs() < 1e-6,
            "expected ~1 for the high-logit pick: {wts:?}"
        );
    }

    /// MLA (DeepSeek V2/V3 absorbed form) on the REAL Vulkan path, vs a CPU reference — the
    /// smallest possible dispatch of `recorder.mla` (no model, no seam). Mirrors `mla_parity`'s
    /// synthetic inputs in infr-llama/tests/seam_op_parity.rs. This is the regression test for the
    /// "logical device has been lost" hang the kernel caused on RDNA3 (see the .comp's header).
    ///
    /// Run against BOTH tiers. They are separate SPIR-V with separate reductions and separate
    /// softmax structure (`mla.comp`'s `-DMLA_SG`), and the shipped default only ever dispatches
    /// one of them — so a scalar-only regression would otherwise sit unexecuted behind a knob
    /// nothing exercises, which is the same thing as not having a fallback.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn mla_matches_cpu_reference() {
        for sg in [true, false] {
            mla_matches_cpu_reference_at(sg);
        }
    }

    fn mla_matches_cpu_reference_at(sg: bool) {
        let be = be_with(|c| c.mla_sg = sg);
        // Seam-family shape (rows=1 decode, ring cap=12) but with TWO attended keys (pos=1, causal)
        // so the absorbed-query scores actually shape the softmax — at kv_len=1 softmax is trivial
        // and the output is independent of q, which hid the wk_b transposition bug for months.
        // Exercises the ring-buffer path with cap > kv_len and the full shared-memory arrays.
        let (rows, nh, kv_lora, qk_nope, qk_rope, vhd) =
            (1usize, 16usize, 512usize, 128usize, 64usize, 128usize);
        let cap_rows = 12usize; // ring capacity — bigger than kv_len
        let kv_len = 2usize; // attended keys dispatched (pos=1 causal → keys 0..=1)
        let key_len = kv_lora + qk_rope; // 576
        let q_head_dim = qk_nope + qk_rope; // 192
        let scale = 1.0 / ((qk_nope + qk_rope) as f32).sqrt();
        let theta = 10000.0_f32;
        // Synthetic identity-ish wk_b/wv_b: wk_b[h] maps nope dims onto distinct latent slots,
        // wv_b[h] is a shifted identity — the CPU reference reproduces the same math.
        let mut wk: Vec<f32> = vec![0f32; nh * kv_lora * qk_nope];
        let s = kv_lora * qk_nope;
        for h in 0..nh {
            for a in 0..qk_nope {
                let slot = (h + 3 * a) % kv_lora;
                wk[h * s + slot * qk_nope + a] = 1.0;
            }
        }
        let mut wv: Vec<f32> = vec![0f32; nh * kv_lora * vhd];
        for h in 0..nh {
            let off = h * kv_lora * vhd;
            for a in 0..kv_lora.min(vhd) {
                wv[off + a * vhd + a] = 1.0;
            }
        }
        // Same synthetic inputs as the CPU mla_parity test (see seam_op_parity.rs).
        let qi: Vec<f32> = (1..=((rows * nh * q_head_dim) as i32))
            .map(|x| x as f32)
            .collect();
        // Random-but-deterministic K rows (xorshift, values < 2048 so the f16 cache cast is exact):
        // a fixed column pattern (10+col) makes every key row a constant shift of the others, and
        // with a near-one-hot softmax the output then collapses to the winner's V regardless of q —
        // which accidentally made the output invariant to a transposed wk_b. Random rows make the
        // scores, and hence the output, genuinely depend on the absorbed query.
        let mut state = 0x1234_5678u32;
        let mut ki = Vec::with_capacity(cap_rows * key_len);
        for _ in 0..cap_rows * key_len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ki.push((state % 1999) as f32);
        }
        let bq = upf32(&be, &qi);
        let bwk = upf32(&be, &wk);
        let bwv = upf32(&be, &wv);
        // k_cache: allocate a REAL cache buffer and write it via rec.copy (the same f16 WriteKv
        // path the seam uses) INSIDE the same recorder as the mla — reproducing the real graph's
        // store→read hazard sequence.
        let s = kv_lora * qk_nope;
        let bk = be
            .alloc(cap_rows * key_len * 2, BufferUsage::KvCache)
            .unwrap();
        let ksrc = upf16(&be, &ki);
        let bo = be
            .alloc(rows * nh * vhd * 4, BufferUsage::Readback)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.copy(ksrc.as_ref(), 0, bk.as_ref(), 0, cap_rows * key_len * 2);
        rec.mla(
            bq.as_ref(),
            bk.as_ref(),
            bwk.as_ref(),
            bwv.as_ref(),
            bo.as_ref(),
            rows as u32,
            kv_len as u32, // two DIFFERENT keys so the absorbed-query scores matter (kv_len=1 makes
            // softmax trivial and would hide a transposed wk_b)
            nh as u32,
            q_head_dim as u32,
            kv_lora as u32,
            qk_nope as u32,
            qk_rope as u32,
            vhd as u32,
            scale,
            1, // pos — causal: attends keys 0..=1
            0, // mask_type: causal
            0, // window
            theta,
            cap_rows as u32, // cache_cap_rows — ring capacity > kv_len
            0,               // canvas_lo — unused by the causal mask
            None,            // freq_factors — no YaRN divisors in the reference test
            None,            // key_bias — deepseek2 attends every causally-eligible key
        );
        rec.finish().unwrap();
        let mut bytes = vec![0u8; rows * nh * vhd * 4];
        be.download(bo.as_ref(), &mut bytes).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        // CPU reference: absorb → rope → SDPA → wv_b (see seam_op_parity.rs mla_parity).
        let mut want = vec![0f32; rows * nh * vhd];
        for t in 0..rows {
            let abs_pos = 1u32; // pos = 1
            for h in 0..nh {
                let q_off = (t * nh + h) * q_head_dim;
                let mut q_full = vec![0f32; key_len];
                for j in 0..kv_lora {
                    let mut acc = 0.0f32;
                    for i in 0..qk_nope {
                        acc += wk[h * s + i + j * qk_nope] * qi[q_off + i];
                    }
                    q_full[j] = acc;
                }
                for p in 0..qk_rope / 2 {
                    let ang = abs_pos as f32 * theta.powf(-2.0 * p as f32 / qk_rope as f32);
                    let (i0, i1) = (2 * p, 2 * p + 1);
                    let (a, b) = (qi[q_off + qk_nope + i0], qi[q_off + qk_nope + i1]);
                    q_full[kv_lora + i0] = a * ang.cos() - b * ang.sin();
                    q_full[kv_lora + i1] = a * ang.sin() + b * ang.cos();
                }
                let mut vacc = vec![0f32; kv_lora];
                let mut sumw = 0.0f32;
                // Two-pass with the true lmax (max over attended keys) for exact parity.
                let lo = 0usize;
                let hi = (abs_pos as usize + 1).min(kv_len);
                let mut lmax = f32::NEG_INFINITY;
                let mut scores = Vec::new();
                for j in lo..hi {
                    let mut d = 0.0f32;
                    for di in 0..key_len {
                        d += q_full[di] * ki[j * key_len + di];
                    }
                    d *= scale;
                    lmax = lmax.max(d);
                    scores.push(d);
                }
                for (jj, &d) in scores.iter().enumerate() {
                    let p = (d - lmax).exp();
                    sumw += p;
                    for di in 0..kv_lora {
                        vacc[di] += p * ki[(lo + jj) * key_len + di];
                    }
                }
                let inv = 1.0 / sumw.max(1e-20);
                for d in 0..vhd {
                    let mut acc = 0.0f32;
                    for i in 0..kv_lora {
                        acc += vacc[i] * inv * wv[h * kv_lora * vhd + i + d * kv_lora];
                    }
                    want[t * nh * vhd + h * vhd + d] = acc;
                }
            }
        }
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let tier = if sg { "mla_sg" } else { "mla" };
        println!("mla vulkan {tier} max_err={err:e}");
        assert!(err < 1e-3, "MLA vulkan {tier} diverges: max_err={err:e}");
    }

    /// One `mla_ring_and_mask_matches_cpu_reference` case.
    struct MlaCase {
        name: &'static str,
        rows: usize,
        pos: u32,
        kv_len: usize,
        /// Ring row capacity. `cap < kv_len` is a genuinely WRAPPED cache, so `mla.comp`'s
        /// `(lo + jj) % cap` has to fold rather than being the identity it is at `cap >= kv_len`.
        cap: usize,
        /// 0 = causal, 1 = sliding window, 2 = canvas (see `mla.comp`'s `pc.mask_type`).
        mask_type: u32,
        window: u32,
        /// `AttnMask::Canvas { lo }`'s lower bound — read on `mask_type == 2` only.
        canvas_lo: u32,
    }

    /// `mla.comp` over the axes `mla_matches_cpu_reference` never moves: a WRAPPED ring cache
    /// (`cache_cap_rows < kv_len`), `mask_type` 1 (sliding window) and 2 (canvas), and a `pos`
    /// past the first — the gap recorded as `docs/backlog.md` B46's second bullet.
    ///
    /// The cache is filled by an explicit ring WRITER (absolute position `j` → row `j % cap`,
    /// ascending), and the reference below reads keys by ABSOLUTE POSITION, never computing a row
    /// index. Kernel and reference therefore agree only if the kernel's `(lo + jj) % cap` resolves
    /// to the row the writer used — which is what makes the wrapped cases informative rather than
    /// a restatement of the shader's own expression.
    ///
    /// Dims are tiny (and mirror `infr-llama`'s `mla_mask_ring_parity` case for case) so a failure
    /// is hand-checkable; `mla_matches_cpu_reference` above keeps the DeepSeek-sized dispatch that
    /// guards the shared-memory/lane layout.
    ///
    /// Run against BOTH tiers, for the reason `mla_matches_cpu_reference` gives.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn mla_ring_and_mask_matches_cpu_reference() {
        for sg in [true, false] {
            mla_ring_and_mask_matches_cpu_reference_at(sg);
        }
    }

    fn mla_ring_and_mask_matches_cpu_reference_at(sg: bool) {
        let be = be_with(|c| c.mla_sg = sg);
        // key_len = kv_lora + qk_rope must be EVEN: `kread` addresses the f16 cache as u32-packed
        // PAIRS, so an odd row width would put a row's last element in a half-word past the end of
        // the allocation.
        let (nh, kv_lora, qk_nope, qk_rope, vhd) = (2usize, 4usize, 2usize, 2usize, 2usize);
        let key_len = kv_lora + qk_rope; // 6
        let q_head_dim = qk_nope + qk_rope; // 4
        let scale = 1.0 / (q_head_dim as f32).sqrt();
        let theta = 10000.0f32;
        // One-hot wk_b/wv_b in the READ convention the shader uses (`i` / `a` the FAST dim).
        let mut wk: Vec<f32> = vec![0f32; nh * kv_lora * qk_nope];
        let mut wv: Vec<f32> = vec![0f32; nh * kv_lora * vhd];
        for h in 0..nh {
            for i in 0..qk_nope {
                wk[h * kv_lora * qk_nope + i + ((h + i) % kv_lora) * qk_nope] = 1.0;
            }
            for o in 0..vhd {
                wv[h * kv_lora * vhd + (h + o) % kv_lora + o * kv_lora] = 1.0;
            }
        }
        // One distinct key per absolute position. Values are 1/16ths, so the f16 cache round-trip
        // is EXACT and the tolerance is measuring the kernel, not the cast. They are also all
        // O(1), which keeps the softmax SOFT — under the near-one-hot softmax that large values
        // give, the output is just the winning key's V and adding or dropping a LOSING key (what
        // an off-by-one in `lo`/`hi` does) would leave the output unchanged.
        let key_at = |j: usize| -> Vec<f32> {
            (0..key_len)
                .map(|d| ((j * 7 + d * 3) % 13) as f32 / 16.0 + 0.125)
                .collect()
        };
        let q_at = |i: usize| ((i * 5 + 3) % 11) as f32 / 8.0 - 0.5;
        // Which ABSOLUTE positions a query at `abs` may attend to — stated from what each mask
        // MEANS, not from the shader's expression for it.
        let attends = |c: &MlaCase, abs: usize| {
            match c.mask_type {
                // Every earlier position plus its own.
                0 => 0..(abs + 1).min(c.kv_len),
                // The `window` most recent positions, its own included.
                1 => (abs + 1).saturating_sub(c.window as usize)..(abs + 1).min(c.kv_len),
                // One fixed span for EVERY row — `abs` is not consulted at all.
                _ => c.canvas_lo as usize..c.kv_len,
            }
        };

        let cases = [
            MlaCase {
                name: "causal pos=0, no wrap",
                rows: 2,
                pos: 0,
                kv_len: 2,
                cap: 2,
                mask_type: 0,
                window: 0,
                canvas_lo: 0,
            },
            MlaCase {
                name: "causal pos=3, no wrap",
                rows: 2,
                pos: 3,
                kv_len: 5,
                cap: 8,
                mask_type: 0,
                window: 0,
                canvas_lo: 0,
            },
            MlaCase {
                name: "sliding window w=3, pos=3, no wrap",
                rows: 2,
                pos: 3,
                kv_len: 5,
                cap: 8,
                mask_type: 1,
                window: 3,
                canvas_lo: 0,
            },
            // cap=5 over 14 positions is two full laps plus four rows, and the abs=12 row attends
            // 9..13 → rows 4,0,1,2: `lo` and `hi-1` land on OPPOSITE sides of the wrap boundary,
            // which a single lap starting at row 0 would not have caught.
            MlaCase {
                name: "sliding window w=4, wrapped ring (lo/hi straddle)",
                rows: 2,
                pos: 12,
                kv_len: 14,
                cap: 5,
                mask_type: 1,
                window: 4,
                canvas_lo: 0,
            },
            MlaCase {
                name: "sliding window w=cap=5, wrapped ring",
                rows: 1,
                pos: 13,
                kv_len: 14,
                cap: 5,
                mask_type: 1,
                window: 5,
                canvas_lo: 0,
            },
            MlaCase {
                name: "canvas lo=0, pos=3",
                rows: 2,
                pos: 3,
                kv_len: 5,
                cap: 8,
                mask_type: 2,
                window: 0,
                canvas_lo: 0,
            },
            // Canvas ignores `abs` entirely: both rows attend 2..5 even though their causal bounds
            // differ, and `pos` still moves the internal q_pe rope. `lo = 0` cannot see the
            // `pc.canvas_lo` field at all — the hardcoded `lo = 0u` this replaces passed that case.
            MlaCase {
                name: "canvas lo=2, pos=3",
                rows: 2,
                pos: 3,
                kv_len: 5,
                cap: 8,
                mask_type: 2,
                window: 0,
                canvas_lo: 2,
            },
            // A bounded canvas span over a WRAPPED ring: 9..14 is five positions in a cap=5 ring,
            // rows 4,0,1,2,3 — `lo` and `hi-1` on opposite sides of the wrap boundary. This is the
            // only span shape well defined over a wrap (B46: a causal query would attend rows the
            // ring has already overwritten).
            MlaCase {
                name: "canvas lo=9, wrapped ring (straddles)",
                rows: 1,
                pos: 13,
                kv_len: 14,
                cap: 5,
                mask_type: 2,
                window: 0,
                canvas_lo: 9,
            },
        ];

        for c in cases {
            // Ring writer: absolute position j lands in row j % cap, ascending.
            let keys: Vec<Vec<f32>> = (0..c.kv_len).map(key_at).collect();
            let mut cache = vec![0f32; c.cap * key_len];
            for (j, k) in keys.iter().enumerate() {
                cache[(j % c.cap) * key_len..][..key_len].copy_from_slice(k);
            }
            // A ring holds only the last `cap` positions. If an attended position's row was reused
            // by a LATER one, the cache no longer holds the key the reference expects and the case
            // is asking a question with no answer — catch that here, not as a mystery max_err.
            for ti in 0..c.rows {
                let abs = c.pos as usize + ti;
                for j in attends(&c, abs) {
                    let last = (0..c.kv_len)
                        .rfind(|p| p % c.cap == j % c.cap)
                        .expect("attended position is inside 0..kv_len");
                    assert_eq!(
                        last, j,
                        "{}: attended position {j} was overwritten by {last} in the ring — the \
                         case attends a wider span than cap={} holds",
                        c.name, c.cap
                    );
                }
            }
            let qi: Vec<f32> = (0..c.rows * nh * q_head_dim).map(q_at).collect();

            let bq = upf32(&be, &qi);
            let bwk = upf32(&be, &wk);
            let bwv = upf32(&be, &wv);
            let ksrc = upf16(&be, &cache);
            let bk = be.alloc(c.cap * key_len * 2, BufferUsage::KvCache).unwrap();
            let bo = be
                .alloc(c.rows * nh * vhd * 4, BufferUsage::Readback)
                .unwrap();
            let rec = be.recorder().unwrap();
            // Write the cache through `rec.copy` inside the SAME recorder as the mla, reproducing
            // the real graph's store→read hazard sequence (as `mla_matches_cpu_reference` does).
            rec.copy(ksrc.as_ref(), 0, bk.as_ref(), 0, c.cap * key_len * 2);
            rec.mla(
                bq.as_ref(),
                bk.as_ref(),
                bwk.as_ref(),
                bwv.as_ref(),
                bo.as_ref(),
                c.rows as u32,
                c.kv_len as u32,
                nh as u32,
                q_head_dim as u32,
                kv_lora as u32,
                qk_nope as u32,
                qk_rope as u32,
                vhd as u32,
                scale,
                c.pos,
                c.mask_type,
                c.window,
                theta,
                c.cap as u32,
                c.canvas_lo,
                None,
                None,
            );
            rec.finish().unwrap();
            let mut bytes = vec![0u8; c.rows * nh * vhd * 4];
            be.download(bo.as_ref(), &mut bytes).unwrap();
            let got: &[f32] = bytemuck::cast_slice(&bytes);

            // CPU reference: absorb → rope → SDPA over the attended ABSOLUTE positions → wv_b.
            let mut want = vec![0f32; c.rows * nh * vhd];
            for ti in 0..c.rows {
                let abs = c.pos as usize + ti;
                for h in 0..nh {
                    let q_off = (ti * nh + h) * q_head_dim;
                    let mut q_full = vec![0f32; key_len];
                    for (j, qf) in q_full.iter_mut().enumerate().take(kv_lora) {
                        let mut acc = 0.0f32;
                        for i in 0..qk_nope {
                            acc += wk[h * kv_lora * qk_nope + i + j * qk_nope] * qi[q_off + i];
                        }
                        *qf = acc;
                    }
                    for p in 0..qk_rope / 2 {
                        let ang = abs as f32 * theta.powf(-2.0 * p as f32 / qk_rope as f32);
                        let (i0, i1) = (2 * p, 2 * p + 1);
                        let (a, b) = (qi[q_off + qk_nope + i0], qi[q_off + qk_nope + i1]);
                        q_full[kv_lora + i0] = a * ang.cos() - b * ang.sin();
                        q_full[kv_lora + i1] = a * ang.sin() + b * ang.cos();
                    }
                    let span = attends(&c, abs);
                    let sc: Vec<f32> = span
                        .clone()
                        .map(|j| {
                            keys[j].iter().zip(&q_full).map(|(k, q)| k * q).sum::<f32>() * scale
                        })
                        .collect();
                    let lmax = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut vacc = vec![0f32; kv_lora];
                    let mut sumw = 0.0f32;
                    for (j, &d) in span.zip(&sc) {
                        let p = (d - lmax).exp();
                        sumw += p;
                        for (di, v) in vacc.iter_mut().enumerate() {
                            *v += p * keys[j][di];
                        }
                    }
                    let inv = 1.0 / sumw.max(1e-20);
                    for d in 0..vhd {
                        let mut acc = 0.0f32;
                        for i in 0..kv_lora {
                            acc += vacc[i] * inv * wv[h * kv_lora * vhd + i + d * kv_lora];
                        }
                        want[(ti * nh + h) * vhd + d] = acc;
                    }
                }
            }
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let tier = if sg { "mla_sg" } else { "mla" };
            println!(
                "mla vulkan {tier} {}: max_err={err:e}\n  got ={got:?}\n  want={want:?}",
                c.name
            );
            assert!(
                err < 1e-4,
                "MLA vulkan {tier} {} diverges: max_err={err:e}",
                c.name
            );
        }
    }
}
