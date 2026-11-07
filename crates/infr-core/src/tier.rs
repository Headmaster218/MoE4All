//! Shared capability-tiering / kernel-selection policy.
//!
//! Picking WHICH kernel runs a `Linear` or an `Attention` is two separable things: the
//! device-specific half (does this dtype have a build? does the device have dp4a / coopmat? does
//! the pipeline fit a 128-thread threadgroup?) and a device-INDEPENDENT half — pure arithmetic on
//! `m` / `out_f` / the KV span, plus the `INFR_*` knobs that override it. The second half was
//! re-derived per backend and drifted. This module hosts it ONCE.
//!
//! Backends stay **parameterized, not flattened**: every number a backend measured on its own
//! hardware is passed IN as config ([`MultiRowBand`], [`AttnSplitCfg`], [`EnvRows`]), and the
//! policy returns an enum ([`LinearTier`]) the backend maps to its own kernels. Nothing here
//! knows about a shader, a pipeline, or a dtype.
//!
//! What is here:
//!
//! - [`EnvRows`] — the numeric row/count knob (`INFR_MOE_SMALL_M`, `INFR_CANVAS_CHUNK_N`): the
//!   measured default plus the safe clamp range the configured value is bounded into. (An
//!   UNCLAMPED override is not a slow run but a dead GPU — see Vulkan's `INFR_MOE_SMALL_M` doc.)
//! - [`linear_tier`] / [`MultiRowBand`] — the dense `Linear` m-ladder: single-row decode GEMV,
//!   the small-m multi-row GEMV band(s), the tiled prefill GEMM.
//! - [`adaptive_chunk`] / [`cap_chunk_count`] / [`baked_chunk`] / [`n_chunks`] — the flash-decoding
//!   (split-K) attention chunk policy shared by Vulkan's `attn_partial`.
//!
//! What is deliberately NOT here (device-specific facts that only LOOK like duplicates):
//!
//! - The prefill GEMM's narrow-n split-K count (Vulkan `split_k_plan`, Metal's `ks_split`). Both
//!   estimate "does the tile grid fill the device" and split k to fill it, but over DIFFERENT tile
//!   geometry (Vulkan 64x128 coopmat vs Metal 32x64 threadgroup) with different fill targets and
//!   different tail rules. Unifying them would mean a config struct with one field per line of
//!   either formula.
//! - Per-dtype / per-vendor kernel-coverage sets (`native_mrow_kernel_name`,
//!   `mrow_int8_dtype_ok`, Metal's `prefer_*`): capability, not arithmetic — candidate G's
//!   territory, and already predicate-shaped.

/// A numeric row/count knob: the measured default, the safe clamp range, and the `INFR_*` name it
/// answers to. `usize`-valued (rows, chunk counts) — byte-valued knobs use [`crate::parse_size`]
/// instead.
///
/// The VALUE comes from the [`Config`](crate::config::Config) the backend was built with; this
/// table is the accessor's policy half ([`clamped`](Self::clamped)) plus the shipped default,
/// which must equal the corresponding `Config::default()` field (the backends pin that).
///
/// The clamp is not cosmetic. A tier threshold pushed far past its measured range does not just
/// mis-schedule: an unclamped `INFR_MOE_SMALL_M` turns a 1024-row prefill chunk into one
/// multi-second dispatch and trips the amdgpu ring watchdog, taking the process down with a device
/// loss. Every knob therefore names its own ceiling.
#[derive(Debug, Clone, Copy)]
pub struct EnvRows {
    /// The `INFR_*` name this knob is spelled with — its identity in `config::manifest`, and the
    /// name a doc/error message quotes. Nothing here READS it (R3: the environment is read once,
    /// in `config::env`).
    pub env: &'static str,
    /// The shipped value, used when no layer specified the knob. Must match the knob's
    /// `Config::default()` field. NOT clamped separately — a default outside `[min, max]` is a
    /// config bug, and [`resolve`](Self::resolve) would silently clamp it.
    pub default: usize,
    /// Inclusive lower clamp for an override (`0` when the knob has no floor).
    pub min: usize,
    /// Inclusive upper clamp for an override (`usize::MAX` when the knob has no ceiling).
    pub max: usize,
}

impl EnvRows {
    /// The knob's value as CONFIGURED, clamped into `[min, max]` — the accessor half of the
    /// migrated knob (`docs/config-plan.md` R5: the config layer only parses, the clamp is policy
    /// and lives here). The caller passes `cfg.kernels.vulkan.moe_small_m` (etc.), whose
    /// `Config::default()` is this table's [`default`](Self::default) — pinned by the backend's
    /// own knob-table test, since the two spellings must not drift.
    pub fn clamped(&self, value: usize) -> usize {
        value.clamp(self.min, self.max)
    }

    /// The WHOLE knob grammar over an explicit override string: parse, else `default`, then
    /// [`clamped`](Self::clamped). This is what the env layer + `Config::default()` + `clamped`
    /// reproduce between them, and the test that pins the grammar drives it — over plain strings,
    /// so it never touches the process environment.
    pub fn resolve(&self, raw: Option<&str>) -> usize {
        self.clamped(
            raw.and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(self.default),
        )
    }
}

/// One small-m multi-row GEMV band: the `m` window (and the `out_f` shapes) on which re-reading
/// the weight per row-tile beats handing the shape to a tiled GEMM.
///
/// The band exists because a single-M-tile GEMM launches only `out_f/tile` workgroups and
/// underfills the GPU, while a plain GEMV re-streams the weight once per row. In between, a
/// multi-row GEMV reads the weight once and reuses it across `m` rows. The band's upper edge is
/// shape-dependent: once `out_f` is wide enough that the GEMM's tile grid DOES fill the device,
/// the GEMM wins again — hence [`Self::narrow_max_m`] / [`Self::wide_out_f_max`].
///
/// Every field is a MEASURED per-backend number; nothing here is a shared constant.
#[derive(Debug, Clone, Copy)]
pub struct MultiRowBand {
    /// Lowest `m` on the band (usually 2 — `m == 1` is [`LinearTier::Decode`]).
    pub min_m: usize,
    /// Highest `m` on the band at any covered `out_f`.
    pub max_m: usize,
    /// Rows at or below this take the band at ANY `out_f` (no wide-n cutoff). Set to `0` for a
    /// band whose every row count is subject to [`Self::wide_out_f_max`].
    pub narrow_max_m: usize,
    /// Above [`Self::narrow_max_m`], the band applies only while `out_f <= wide_out_f_max` — past
    /// it the GEMM's tile grid fills the device and wins. `usize::MAX` disables the cutoff.
    pub wide_out_f_max: usize,
    /// Hard inclusive `out_f` ceiling for the WHOLE band, independent of `m` — the vocab-width
    /// `lm_head` shape, where the multi-row GEMV's per-row workgroup explosion loses to a GEMM
    /// that streams the weight once. `usize::MAX` disables it.
    pub out_f_max: usize,
}

impl MultiRowBand {
    /// Is `(m, out_f)` inside this band? Pure shape arithmetic — the caller still has to confirm
    /// the band's kernel exists for the weight dtype and that its env hatch is unset.
    pub const fn contains(&self, m: usize, out_f: usize) -> bool {
        m >= self.min_m
            && m <= self.max_m
            && (m <= self.narrow_max_m || out_f <= self.wide_out_f_max)
            && out_f <= self.out_f_max
    }
}

/// Which m-tier a dense `Linear` falls in, before any device/dtype gating. The backend maps this
/// onto its own kernels (and may still fall back to [`LinearTier::Gemm`] when the band's kernel
/// has no build for the weight dtype, or its `INFR_*` hatch is set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearTier {
    /// `m == 1` — the single-row decode GEMV. Its numerics are the model's precision contract
    /// (every backend keeps decode on its exact path), so this tier is never merged upward.
    Decode,
    /// A small multi-row batch — a speculative/MTP verify batch, a chat turn's short suffix
    /// prefill. `band` indexes the `bands` slice passed to [`linear_tier`]; a backend with several
    /// bands (different kernels, different capability requirements) reads it to pick which.
    MultiRow {
        /// Index into the `bands` slice of the FIRST band containing the shape.
        band: usize,
    },
    /// Everything else — the tiled prefill GEMM.
    Gemm,
}

/// Classify a dense `Linear` shape against the backend's multi-row bands, in order: `m == 1` is
/// [`LinearTier::Decode`], the first band containing `(m, out_f)` wins, otherwise
/// [`LinearTier::Gemm`]. `bands` may be empty (a backend with no multi-row GEMV — every `m > 1`
/// shape goes straight to the GEMM).
pub fn linear_tier(m: usize, out_f: usize, bands: &[MultiRowBand]) -> LinearTier {
    if m == 1 {
        return LinearTier::Decode;
    }
    match bands.iter().position(|b| b.contains(m, out_f)) {
        Some(band) => LinearTier::MultiRow { band },
        None => LinearTier::Gemm,
    }
}

/// Rounding used when deriving the adaptive attention chunk size from a KV span.
///
/// `Down` — `span / target_chunks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkRounding {
    /// `span / target_chunks`.
    Down,
}

/// Flash-decoding (split-K) attention chunk policy: aim for ~`target_chunks` chunks per head, each
/// clamped into `[min_chunk, max_chunk]` keys.
///
/// A single wave per (row, head) that scans the whole cache serially is fine at low depth and
/// leaves the device idle at long context; splitting the KV range into chunks with per-chunk
/// online-softmax partials plus a combine pass fills it. Too few chunks and the split does not pay
/// for the combine; too many and the partials scratch (`[rows, heads, n_chunks, head_dim]`)
/// balloons — hence the clamp.
#[derive(Debug, Clone, Copy)]
pub struct AttnSplitCfg {
    /// Chunk count aimed for before clamping (both backends: 32).
    pub target_chunks: usize,
    /// Inclusive lower clamp on the chunk SIZE in keys.
    pub min_chunk: usize,
    /// Inclusive upper clamp on the chunk SIZE in keys.
    pub max_chunk: usize,
    /// How `span / target_chunks` rounds — see [`ChunkRounding`].
    pub rounding: ChunkRounding,
}

/// The adaptive chunk SIZE (in keys) for a KV `span`. Does NOT apply any chunk-COUNT cap — a
/// backend whose combine kernel has a fixed per-chunk shared array must additionally run the
/// result through [`cap_chunk_count`].
pub fn adaptive_chunk(span: usize, cfg: &AttnSplitCfg) -> usize {
    let target = cfg.target_chunks.max(1);
    let base = span / target;
    base.clamp(cfg.min_chunk, cfg.max_chunk)
}

/// Chunks needed to cover `span` at `chunk` keys each.
pub fn n_chunks(span: usize, chunk: usize) -> usize {
    span.div_ceil(chunk.max(1))
}

/// Raise `chunk` just enough that `n_chunks(span, chunk) <= max_chunks`.
///
/// Bounds the chunk COUNT, not its size: a combine kernel that indexes a fixed
/// `shared float w[max_chunks]` by chunk would OOB-write past it. Raising the chunk floor to
/// `span.div_ceil(max_chunks)` caps the count; for every span where the caller's own policy
/// already produced a small enough count this returns `chunk` UNCHANGED, so the recorded dispatch
/// is byte-identical (Vulkan: it only bites above `max_chunk * max_chunks` ≈ 524k keys, reachable
/// only under `INFR_KV_OVERFLOW`).
pub fn cap_chunk_count(span: usize, chunk: usize, max_chunks: usize) -> usize {
    chunk.max(span.div_ceil(max_chunks.max(1)))
}

/// The chunk size BAKED into a record-once decode replay, derived from the KV cache's row
/// `capacity` instead of a live span: `chunk`/`n_chunks` are push constants (dispatch structure),
/// so the recording must cover the worst case the cache can reach. Chunks past the live `kv_len`
/// produce zero-weight partials the combine ignores.
///
/// Starts at `min_chunk` and rises only when `capacity` would exceed the `max_chunks` cap.
pub fn baked_chunk(capacity: usize, max_chunks: usize, min_chunk: usize) -> usize {
    capacity.div_ceil(max_chunks.max(1)).max(min_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vulkan's shipping decode split-K policy, mirrored here so the boundary sweeps below run
    /// against real numbers (the crate itself owns the constant).
    const VK_SPLIT: AttnSplitCfg = AttnSplitCfg {
        target_chunks: 32,
        min_chunk: 64,
        max_chunk: 512,
        rounding: ChunkRounding::Down,
    };
    /// Vulkan `attn_combine.comp`'s `shared float wexp[1024]`.
    const VK_MAX_CHUNKS: usize = 1024;

    /// The knob grammar, over explicit strings via [`EnvRows::resolve`] — NO process-environment
    /// mutation, so this cannot race another test in the same binary.
    /// The knob's value now arrives from a `Config` (the env layer parses, `Config::default()`
    /// holds the default) and only [`EnvRows::clamped`] runs at the read site; `resolve` is the
    /// composition of the three and stays the one place the whole grammar is pinned.
    #[test]
    fn env_rows_defaults_parses_and_clamps() {
        let knob = EnvRows {
            env: "INFR_TIER_TEST_ROWS",
            default: 8,
            min: 0,
            max: 64,
        };
        // Unset (and every unparseable spelling) yields the default untouched.
        assert_eq!(knob.resolve(None), 8);
        for (raw, want) in [
            ("0", 0), // no floor on this knob — 0 is a legal "never take the path"
            ("1", 1),
            ("8", 8),
            ("64", 64),     // exactly the ceiling
            ("65", 64),     // clamped
            ("100000", 64), // THE watchdog case: an unclamped huge value hangs the GPU
            ("", 8),        // unparseable → unset
            ("abc", 8),
            ("-1", 8), // negative doesn't parse as usize → unset, NOT 0
            ("4.5", 8),
        ] {
            assert_eq!(knob.resolve(Some(raw)), want, "raw={raw:?}");
        }

        // A knob with a FLOOR instead of a ceiling (the canvas chunk divisor shape): 0 would be a
        // division by zero downstream, so it clamps up.
        let floored = EnvRows {
            env: "INFR_TIER_TEST_FLOOR",
            default: 3,
            min: 1,
            max: usize::MAX,
        };
        assert_eq!(floored.resolve(Some("0")), 1);
        assert_eq!(floored.resolve(Some("7")), 7);
        assert_eq!(floored.resolve(None), 3);
    }

    /// The read-site half on its own: whatever the config resolved to is clamped into the knob's
    /// safe range, so a config file / `--set` / env override past the ceiling is as harmless as
    /// the env override always was (the unclamped `INFR_MOE_SMALL_M=100000` device-loss).
    #[test]
    fn env_rows_clamped_bounds_a_configured_value() {
        let knob = EnvRows {
            env: "INFR_TIER_TEST_ROWS",
            default: 8,
            min: 0,
            max: 64,
        };
        assert_eq!(knob.clamped(8), 8);
        assert_eq!(knob.clamped(64), 64);
        assert_eq!(knob.clamped(65), 64);
        assert_eq!(knob.clamped(100_000), 64);
        // A floored knob (the canvas chunk DIVISOR) clamps up instead — 0 would divide by zero.
        let floored = EnvRows {
            env: "INFR_TIER_TEST_FLOOR",
            default: 3,
            min: 1,
            max: usize::MAX,
        };
        assert_eq!(floored.clamped(0), 1);
        assert_eq!(floored.clamped(7), 7);
    }

    #[test]
    fn linear_tier_decode_and_empty_bands() {
        // m == 1 is Decode regardless of the bands (its exactness is the precision contract).
        let band = MultiRowBand {
            min_m: 1,
            max_m: 16,
            narrow_max_m: 16,
            wide_out_f_max: usize::MAX,
            out_f_max: usize::MAX,
        };
        assert_eq!(linear_tier(1, 4096, &[band]), LinearTier::Decode);
        assert_eq!(linear_tier(1, 4096, &[]), LinearTier::Decode);
        // No bands ⟹ every m > 1 shape is the GEMM.
        for m in [2usize, 8, 64, 1024] {
            assert_eq!(linear_tier(m, 4096, &[]), LinearTier::Gemm);
        }
    }

    #[test]
    fn multirow_band_boundaries_wide_n_cutoff() {
        // Vulkan's m<=8 mrow band: 2..=4 at any width, 5..=8 only while out_f <= 8192.
        let vk_mrow = MultiRowBand {
            min_m: 2,
            max_m: 8,
            narrow_max_m: 4,
            wide_out_f_max: 8192,
            out_f_max: usize::MAX,
        };
        let old = |m: usize, out_f: usize| {
            (2..=4).contains(&m) || ((5..=8).contains(&m) && out_f <= 8192)
        };
        for m in 0usize..24 {
            for out_f in [1usize, 4096, 8191, 8192, 8193, 24576, 262_144] {
                assert_eq!(
                    vk_mrow.contains(m, out_f),
                    old(m, out_f),
                    "vk mrow band m={m} out_f={out_f}"
                );
            }
        }
        // The exact edges the old inline expression encoded.
        assert!(!vk_mrow.contains(1, 4096)); // decode, not the band
        assert!(vk_mrow.contains(2, usize::MAX)); // narrow rows ignore the width cutoff
        assert!(vk_mrow.contains(4, 24576));
        assert!(!vk_mrow.contains(5, 24576)); // wide-n: GEMM tiles fill the device
        assert!(vk_mrow.contains(5, 8192));
        assert!(vk_mrow.contains(8, 8192));
        assert!(!vk_mrow.contains(9, 8192)); // above the band
    }

    #[test]
    fn multirow_band_boundaries_m16_and_lmhead_ceiling() {
        // Vulkan's 9..=16 int8 band: no narrow rows, so the width cutoff applies throughout.
        let vk_mrow16 = MultiRowBand {
            min_m: 9,
            max_m: 16,
            narrow_max_m: 0,
            wide_out_f_max: 8192,
            out_f_max: usize::MAX,
        };
        let old16 = |m: usize, out_f: usize| (9..=16).contains(&m) && out_f <= 8192;
        for m in 0usize..24 {
            for out_f in [1usize, 8192, 8193, 65536] {
                assert_eq!(
                    vk_mrow16.contains(m, out_f),
                    old16(m, out_f),
                    "vk mrow16 band m={m} out_f={out_f}"
                );
            }
        }
        assert!(!vk_mrow16.contains(8, 4096)); // the two vulkan bands don't overlap
        assert!(vk_mrow16.contains(9, 4096));
        assert!(vk_mrow16.contains(16, 8192));
        assert!(!vk_mrow16.contains(17, 8192));

        // Metal's mrv band: 2..=8 at any width UP TO the lm_head ceiling (out_f < 65536).
        let mtl_mrv = MultiRowBand {
            min_m: 2,
            max_m: 8,
            narrow_max_m: 8,
            wide_out_f_max: usize::MAX,
            out_f_max: 65535,
        };
        let old_mrv = |m: usize, out_f: usize| (2..=8).contains(&m) && out_f < 65536;
        for m in 0usize..12 {
            for out_f in [1usize, 9216, 65535, 65536, 248_320] {
                assert_eq!(
                    mtl_mrv.contains(m, out_f),
                    old_mrv(m, out_f),
                    "metal mrv band m={m} out_f={out_f}"
                );
            }
        }
        assert!(mtl_mrv.contains(8, 9216));
        assert!(mtl_mrv.contains(8, 65535));
        assert!(!mtl_mrv.contains(8, 65536)); // the tied lm_head goes to the GEMM tile
        assert!(!mtl_mrv.contains(1, 9216));

        // Band ORDER decides ties: the first match wins, and `band` names which kernel.
        let bands = [
            MultiRowBand {
                min_m: 2,
                max_m: 8,
                narrow_max_m: 4,
                wide_out_f_max: 8192,
                out_f_max: usize::MAX,
            },
            vk_mrow16,
        ];
        assert_eq!(
            linear_tier(3, 24576, &bands),
            LinearTier::MultiRow { band: 0 }
        );
        assert_eq!(
            linear_tier(12, 4096, &bands),
            LinearTier::MultiRow { band: 1 }
        );
        assert_eq!(linear_tier(12, 24576, &bands), LinearTier::Gemm);
        assert_eq!(linear_tier(1024, 4096, &bands), LinearTier::Gemm);
    }

    #[test]
    fn adaptive_chunk_clamps_and_rounds() {
        // Below the ramp both roundings floor at min_chunk; above it both cap at max_chunk.
        for span in [0usize, 1, 64, 2047, 2048] {
            assert_eq!(adaptive_chunk(span, &VK_SPLIT), 64, "span {span}");
        }
        for span in [16_384usize, 16_385, 1 << 20] {
            assert_eq!(adaptive_chunk(span, &VK_SPLIT), 512, "span {span}");
        }
        assert_eq!(adaptive_chunk(4096, &VK_SPLIT), 128);
        assert_eq!(adaptive_chunk(2049, &VK_SPLIT), 64);
        assert_eq!(adaptive_chunk(4097, &VK_SPLIT), 128);
    }

    #[test]
    fn cap_chunk_count_is_identity_for_realistic_spans() {
        // The cap must NOT change `chunk` (hence `n_chunks`, hence the recorded dispatch) for any
        // realistic span — it may only kick in past `max_chunk * max_chunks`. Sweep every regime
        // of the adaptive policy (the 64 floor, the /32 ramp, the 512 ceiling) plus the exact
        // boundary where chunk=512 first yields n_chunks=1024.
        let boundary = 512 * VK_MAX_CHUNKS;
        let mut spans: Vec<usize> = (1..=boundary).step_by(97).collect();
        spans.extend([
            1,
            2,
            63,
            64,
            2047,
            2048,
            16_383,
            16_384,
            boundary - 1,
            boundary,
        ]);
        for span in spans {
            let base = adaptive_chunk(span, &VK_SPLIT);
            let capped = cap_chunk_count(span, base, VK_MAX_CHUNKS);
            assert_eq!(
                base, capped,
                "cap changed chunk for realistic span {span}: {base} -> {capped}"
            );
            assert!(
                n_chunks(span, capped) <= VK_MAX_CHUNKS,
                "n_chunks OOB at span {span}: {}",
                n_chunks(span, capped)
            );
        }
    }

    #[test]
    fn cap_chunk_count_bounds_huge_spans() {
        // Past the boundary the cap must engage: the count stays <= max_chunks for every base
        // chunk a caller can arrive with (the adaptive value, the ring/cap 512, the batched 256).
        for span in (1..=8_000_000).step_by(7919) {
            for base in [adaptive_chunk(span, &VK_SPLIT), 512usize, 256usize] {
                let capped = cap_chunk_count(span, base, VK_MAX_CHUNKS);
                let n = n_chunks(span, capped);
                assert!(
                    n <= VK_MAX_CHUNKS,
                    "n_chunks {n} > cap at span {span}, base {base}"
                );
                assert!(capped >= base, "cap must only RAISE the chunk floor");
            }
        }
    }

    #[test]
    fn baked_chunk_holds_the_min_until_the_cap_bites() {
        // Record-once replay bakes chunk/n_chunks from the cache CAPACITY. Below the cap it stays
        // at the floor; above it, it rises just enough to keep the count bounded.
        for cap_rows in [1usize, 64, 4096, 65_536, 1024 * 64] {
            assert_eq!(
                baked_chunk(cap_rows, VK_MAX_CHUNKS, 64),
                64,
                "cap_rows {cap_rows}"
            );
            assert!(n_chunks(cap_rows, baked_chunk(cap_rows, VK_MAX_CHUNKS, 64)) <= VK_MAX_CHUNKS);
        }
        assert_eq!(baked_chunk(1024 * 64 + 1, VK_MAX_CHUNKS, 64), 65);
        for cap_rows in (1..=4_000_000).step_by(4093) {
            let chunk = baked_chunk(cap_rows, VK_MAX_CHUNKS, 64);
            assert!(chunk >= 64);
            assert!(
                n_chunks(cap_rows, chunk) <= VK_MAX_CHUNKS,
                "baked n_chunks OOB at cap_rows {cap_rows}"
            );
        }
    }
}
