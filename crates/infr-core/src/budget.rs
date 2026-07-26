//! Shared KV-cache / VRAM-budget arithmetic.
//!
//! Deciding *where* a KV cache or a weight bank lives is two separable things: the device half
//! (`vkCreateBuffer` vs `hipMalloc`, the memory type, the arena, the staging ring) and a
//! device-INDEPENDENT half — how many bytes a KV format costs, which `INFR_*` knob spells what,
//! the cumulative-cap gate the VRAM-first spill applies, and the one-shot placement banner. The
//! second half was re-derived per backend (and, inside `infr-vulkan`, per multi-device wrapper)
//! and drifted. This module hosts it ONCE.
//!
//! What is here:
//!
//! - [`parse_kv_dtype`] — the `INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` name grammar (llama's
//!   `--cache-type-k` / `--cache-type-v`), the single spelling table.
//! - [`kv_fmt_bytes`] — the EXACT allocation size of a KV buffer of `elems` elements, and
//!   [`kv_bytes_per_elem`] — the same thing as a fractional per-element rate, which is what the
//!   context-fit ESTIMATES divide by. The two must agree; a test pins them against each other.
//! - [`flag_from`] / [`mib_bytes`] / [`reserve_bytes`] — the grammars and accessors the spill
//!   knobs share (`empty`/`0` = off; a MiB count → bytes; the 12%-of-VRAM-floored-at-2-GiB
//!   reserve). The VALUES come from the backend's [`Config`](crate::config::Config); nothing here
//!   reads the environment (R3).
//! - [`SpillTally`] / [`SpillCounts`] / [`spill_report_line`] — the VRAM-first placement
//!   bookkeeping and its banner. The counters and the message SKELETON are shared; every noun and
//!   every device-specific clause is passed in ([`SpillNouns`]), and so is the byte formatter,
//!   because Vulkan and ROCm round MiB differently and the banner text must not move.
//!
//! What is deliberately NOT here (device facts that only look shared):
//!
//! - The placement itself. Whether VRAM has room is `vram_budget_fits` (Vulkan's live
//!   `VK_EXT_memory_budget` heap probe minus a guard headroom) vs `hipMemGetInfo` minus a reserve
//!   — different questions with different answers. Only the cumulative *cap* gate
//!   ([`SpillTally::admits`]), which is pure arithmetic over the tally, is shared.
//! - The human-readable byte formatter. Vulkan prints MiB to one decimal, ROCm to two; unifying
//!   them would move user-visible text for no behavioral gain, so [`spill_report_line`] takes the
//!   formatter as a parameter instead.
//! - The per-format GATES (`kv_align_ok`, which backends have a Q8 KV kernel, TurboQuant's
//!   `head_dim % 128`). Those are capabilities, and they stay at the seam that knows the backend.

use crate::decode_spec::block_layout;
use crate::tensor::DType;
use std::sync::atomic::{AtomicU64, Ordering};

// ── KV format sizing ─────────────────────────────────────────────────────────

/// The `INFR_KV_TYPE_K` / `INFR_KV_TYPE_V` name grammar (llama.cpp's `--cache-type-k` /
/// `--cache-type-v`): a spelling → KV dtype, or `None` for unset/unrecognized, which every caller
/// treats as "fall through to the default ladder".
///
/// This is the ONLY table of KV format names. The runner turns the dtype into an allocation after
/// applying its backend/alignment gates; the placement estimator turns the same dtype into a
/// bytes-per-element rate via [`kv_bytes_per_elem`]. Both used to spell the names themselves, so a
/// format added to one and not the other was priced as f16 while being allocated as something
/// else.
pub fn parse_kv_dtype(name: &str) -> Option<DType> {
    Some(match name {
        // TurboQuant: WHT-rotated, 128-elem blocks.
        "turbo2" => DType::Turbo2,
        "turbo3" => DType::Turbo3,
        "turbo4" => DType::Turbo4,
        "q8_0" | "q8" | "Q8_0" => DType::Q8_0,
        // Mainline low-bit block quants (32-elem blocks).
        "q4_0" => DType::Q4_0,
        "q4_1" => DType::Q4_1,
        "q5_0" => DType::Q5_0,
        "q5_1" => DType::Q5_1,
        "iq4_nl" => DType::Iq4Nl,
        // Dense.
        "bf16" => DType::Bf16,
        "f32" => DType::F32,
        "f16" | "F16" => DType::F16,
        _ => return None,
    })
}

/// EXACT byte size of one KV-cache buffer holding `elems` elements in `dt` — the number the runner
/// hands `Backend::alloc`, so it is also what the VRAM budget guard sees.
///
/// Q8_0 additionally rounds up to a `u32` multiple: the GPU KV kernels address the cache as
/// `uint`s, and a 34-B-per-block cache is only 2-byte aligned at odd block counts. The other block
/// formats have even block sizes, so their products are already aligned.
///
/// Anything not a KV format falls to `elems * 4` (f32), which is what the seam has always done —
/// the KV dtype is chosen by [`parse_kv_dtype`], so the fallthrough is only ever reached for F32
/// itself.
pub fn kv_fmt_bytes(dt: DType, elems: usize) -> usize {
    match dt {
        DType::Q8_0 => (elems / 32 * 34).next_multiple_of(4),
        // TurboQuant 128-elem blocks: turbo2 = 34 B, turbo3 = 50 B, turbo4 = 66 B.
        DType::Turbo2 => elems / 128 * 34,
        DType::Turbo3 => elems / 128 * 50,
        DType::Turbo4 => elems / 128 * 66,
        // Mainline low-bit KV quants (32-elem blocks) + bf16.
        DType::Q4_0 | DType::Iq4Nl => elems / 32 * 18,
        DType::Q4_1 => elems / 32 * 20,
        DType::Q5_0 => elems / 32 * 22,
        DType::Q5_1 => elems / 32 * 24,
        DType::F16 | DType::Bf16 => elems * 2,
        _ => elems * 4, // F32
    }
}

/// KV bytes per ELEMENT in `dt`, as a fraction (e.g. Q8_0 = 34/32 = 1.0625).
///
/// The context-fit estimators need a rate, not a size: they divide a free-byte budget by
/// "bytes per token across all layers" to solve for a context length, which the exact
/// [`kv_fmt_bytes`] (with its per-buffer flooring and alignment rounding) cannot express. It is
/// nonetheless the SAME number — read straight off [`block_layout`], the one decode spec, instead
/// of a second hand-written table of ratios — and `kv_fmt_bytes_matches_the_per_elem_rate` pins
/// that the two agree on every whole block.
pub fn kv_bytes_per_elem(dt: DType) -> f64 {
    let (elems, bytes) = block_layout(dt);
    bytes as f64 / elems as f64
}

// ── grammars + accessors shared by the spill knobs ───────────────────────────

/// The boolean-flag grammar every overflow knob uses, over an explicit value: `Some(v)` with `v`
/// neither empty nor `"0"` ⇒ on; unset ⇒ off. (`INFR_KV_OVERFLOW`,
/// `INFR_ROCM_WEIGHT_OVERFLOW`.) NOT the same as the `is_ok()` presence grammar, where an empty
/// value is ON — `config::env`'s `flag` reader calls this so the difference survives the
/// migration; the read sites take the resolved `bool` off their backend's `Config`.
pub fn flag_from(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| !v.is_empty() && v != "0")
}

/// A MiB-valued knob's CONFIG field in bytes. (`Some(0)` stays `Some(0)` — the diagnostic caps use
/// it to mean "nothing resident", which is NOT the same as "no cap".)
pub fn mib_bytes(mib: Option<u64>) -> Option<u64> {
    mib.map(|mb| mb * 1024 * 1024)
}

/// The MiB grammar over an explicit string — the env spelling of [`mib_bytes`], `None` when absent
/// or unparseable.
pub fn mib_from(raw: Option<&str>) -> Option<u64> {
    mib_bytes(raw.and_then(|v| v.trim().parse::<u64>().ok()))
}

/// VRAM headroom a VRAM-first spill keeps free below the live free-byte figure, so the consumers
/// that allocate AFTER the spilled class (per-forward activation scratch, dequant caches, BLAS
/// workspace, a paged-MoE arena) still have room. Default 12% of total VRAM floored at 2 GiB —
/// enough for a several-hundred-row prefill on a large-vocab model — overridden by the knob's
/// config field (`kv.overflow_reserve_mb` / `paging.rocm_weight_reserve_mb`, in MiB).
///
/// Over-reserving costs residency; under-reserving OOMs mid-forward in an allocator that is
/// infallible by contract, so the floor is deliberate.
pub fn reserve_bytes(total_vram: u64, override_mib: Option<u64>) -> u64 {
    if let Some(bytes) = mib_bytes(override_mib) {
        return bytes;
    }
    (total_vram / 100 * 12).max(2 * 1024 * 1024 * 1024)
}

/// [`reserve_bytes`] over an explicit override STRING — the env spelling of the same policy, kept
/// so the grammar (MiB, trimmed, unparseable ⇒ the default formula) is pinned in one place.
pub fn reserve_from(total_vram: u64, raw: Option<&str>) -> u64 {
    reserve_bytes(total_vram, raw.and_then(|v| v.trim().parse::<u64>().ok()))
}

// ── VRAM-first spill bookkeeping + banner ────────────────────────────────────

/// Resident/spilled counters for one VRAM-first placement class (the KV cache, or ROCm's dense
/// weight banks). Bumped from `alloc` on whichever arm won, drained once by the banner.
///
/// All-zero (the `Default`) means the flag was off or nothing of that class was allocated — both
/// print nothing.
#[derive(Debug, Default)]
pub struct SpillTally {
    vram_bufs: AtomicU64,
    vram_bytes: AtomicU64,
    host_bufs: AtomicU64,
    host_bytes: AtomicU64,
}

/// A [`SpillTally`] read at one instant — plain values, so the banner formats from a consistent
/// snapshot instead of four independent loads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpillCounts {
    pub vram_bufs: u64,
    pub vram_bytes: u64,
    pub host_bufs: u64,
    pub host_bytes: u64,
}

impl SpillCounts {
    /// Buffers placed either way. `0` ⇒ nothing to report.
    pub fn total_bufs(&self) -> u64 {
        self.vram_bufs + self.host_bufs
    }
}

impl SpillTally {
    /// Record one buffer that stayed resident in device-local VRAM.
    pub fn record_vram(&self, bytes: u64) {
        self.vram_bufs.fetch_add(1, Ordering::Relaxed);
        self.vram_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record one buffer that spilled to host RAM.
    pub fn record_host(&self, bytes: u64) {
        self.host_bufs.fetch_add(1, Ordering::Relaxed);
        self.host_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Snapshot for the banner.
    pub fn counts(&self) -> SpillCounts {
        SpillCounts {
            vram_bufs: self.vram_bufs.load(Ordering::Relaxed),
            vram_bytes: self.vram_bytes.load(Ordering::Relaxed),
            host_bufs: self.host_bufs.load(Ordering::Relaxed),
            host_bytes: self.host_bytes.load(Ordering::Relaxed),
        }
    }

    /// The diagnostic CUMULATIVE-cap gate: may `bytes` more still be placed in VRAM?
    ///
    /// `cap` is `None` for the normal case (no cap — spill only when VRAM is genuinely full);
    /// `Some(0)` forces the whole class to host, which is how the whole-host path stays
    /// reproducible on a model that would otherwise fit entirely. The cap gates PLACEMENT only,
    /// never the real VRAM guard.
    pub fn admits(&self, cap: Option<u64>, bytes: u64) -> bool {
        match cap {
            None => true,
            Some(cap) => self.vram_bytes.load(Ordering::Relaxed) + bytes <= cap,
        }
    }
}

/// The per-class nouns of the placement banner — everything [`spill_report_line`] cannot know.
#[derive(Debug, Clone, Copy)]
pub struct SpillNouns<'a> {
    /// Env var that enabled the spill, printed as the line's tag (`INFR_KV_OVERFLOW`).
    pub env: &'a str,
    /// Plural noun for the counted buffers (`"KV buffers"`, `"weight banks"`).
    pub noun: &'a str,
    /// Clause closing the all-resident line, after `"none spilled; "` (`"no PCIe KV reads."`).
    pub resident_note: &'a str,
    /// Clause closing the partial-spill line, after `"in "` — the placement AND who reads it over
    /// PCIe (`"SYSTEM RAM — attention reads those K/V over PCIe ..."`). Device-specific: ROCm's
    /// host allocations are page-locked and Vulkan's KV reads are by device address, and the
    /// existing wording of each is user-visible.
    pub spill_note: &'a str,
}

/// The one-shot VRAM-first placement banner, or `None` when nothing of the class was allocated.
///
/// Two shapes: everything stayed resident (worth saying — it means no PCIe reads at all), or the
/// cache split, in which case the counts on BOTH sides are named so a partial spill is visible
/// rather than looking like a slow run. Callers gate on their own enable flag and `eprintln!` the
/// result; keeping this a pure `String` builder is what lets a test pin the text.
pub fn spill_report_line(
    c: SpillCounts,
    n: &SpillNouns<'_>,
    fmt_bytes: impl Fn(u64) -> String,
) -> Option<String> {
    let total = c.total_bufs();
    if total == 0 {
        return None;
    }
    let (env, noun) = (n.env, n.noun);
    Some(if c.host_bufs == 0 {
        format!(
            "[infr] {env}: all {total} {noun} ({}) fit in VRAM — none spilled; {}",
            fmt_bytes(c.vram_bytes),
            n.resident_note,
        )
    } else {
        format!(
            "[infr] {env}: {} of {total} {noun} ({} resident) in VRAM, the remaining {} ({}) in {}",
            c.vram_bufs,
            fmt_bytes(c.vram_bytes),
            c.host_bufs,
            fmt_bytes(c.host_bytes),
            n.spill_note,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every KV spelling the two pre-hoist tables (the runner's `parse_kv_fmt` and the
    /// estimator's `kv_bytes_per_elem`) accepted, and the dtype each produced. A name dropped here
    /// silently reverts that format to f16 at runtime.
    #[test]
    fn kv_dtype_names_match_the_inline_tables() {
        for (name, want) in [
            ("f32", DType::F32),
            ("bf16", DType::Bf16),
            ("f16", DType::F16),
            ("F16", DType::F16),
            ("q8_0", DType::Q8_0),
            ("q8", DType::Q8_0),
            ("Q8_0", DType::Q8_0),
            ("q4_0", DType::Q4_0),
            ("q4_1", DType::Q4_1),
            ("q5_0", DType::Q5_0),
            ("q5_1", DType::Q5_1),
            ("iq4_nl", DType::Iq4Nl),
            ("turbo2", DType::Turbo2),
            ("turbo3", DType::Turbo3),
            ("turbo4", DType::Turbo4),
        ] {
            assert_eq!(parse_kv_dtype(name), Some(want), "{name}");
        }
        // Unset / misspelled / a weight-only quant / case variants that were never accepted all
        // fall through to the caller's default ladder.
        for name in ["", "q4_k", "Q4_0", "F32", "BF16", "turbo", "q8_1", "f 16"] {
            assert_eq!(parse_kv_dtype(name), None, "{name}");
        }
    }

    /// The per-element rates, verbatim from the pre-hoist `kv_bytes_per_elem` table in
    /// `infr-llama`'s `seam/model.rs`. These divide the free-VRAM budget in the default-ctx clamp,
    /// so a wrong one moves the context a user gets.
    #[test]
    fn kv_bytes_per_elem_matches_the_inline_table() {
        for (dt, want) in [
            (DType::F32, 4.0),
            (DType::Bf16, 2.0),
            (DType::F16, 2.0),
            (DType::Q8_0, 34.0 / 32.0),
            (DType::Q4_0, 18.0 / 32.0),
            (DType::Iq4Nl, 18.0 / 32.0),
            (DType::Q4_1, 20.0 / 32.0),
            (DType::Q5_0, 22.0 / 32.0),
            (DType::Q5_1, 24.0 / 32.0),
            (DType::Turbo2, 34.0 / 128.0),
            (DType::Turbo3, 50.0 / 128.0),
            (DType::Turbo4, 66.0 / 128.0),
        ] {
            // Bit-exact, not approximate: both sides are the same ratio of small integers, and an
            // estimator that drifts by an ULP is an estimator that was recomputed differently.
            assert_eq!(kv_bytes_per_elem(dt), want, "{dt:?}");
        }
    }

    /// The exact allocation sizes, verbatim from the pre-hoist `kv_fmt_bytes` in `infr-llama`'s
    /// `seam/mod.rs` (and the copy in `infr-metal`'s parity suite).
    #[test]
    fn kv_fmt_bytes_matches_the_inline_table() {
        let e = 4096usize;
        for (dt, want) in [
            (DType::Q8_0, (e / 32 * 34usize).next_multiple_of(4)),
            (DType::Turbo2, e / 128 * 34),
            (DType::Turbo3, e / 128 * 50),
            (DType::Turbo4, e / 128 * 66),
            (DType::Q4_0, e / 32 * 18),
            (DType::Iq4Nl, e / 32 * 18),
            (DType::Q4_1, e / 32 * 20),
            (DType::Q5_0, e / 32 * 22),
            (DType::Q5_1, e / 32 * 24),
            (DType::F16, e * 2),
            (DType::Bf16, e * 2),
            (DType::F32, e * 4),
        ] {
            assert_eq!(kv_fmt_bytes(dt, e), want, "{dt:?}");
        }
        // The Q8_0 u32 rounding is load-bearing at an odd block count: 33 blocks x 34 B = 1122,
        // which is not a u32 multiple.
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 33 * 32), 1124);
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 32 * 32), 32 * 34);
        // Sub-block element counts floor to zero bytes, as they always have (the runner never
        // allocates one — every KV row is gate-checked for block alignment first).
        assert_eq!(kv_fmt_bytes(DType::Q8_0, 31), 0);
        assert_eq!(kv_fmt_bytes(DType::Turbo3, 127), 0);
    }

    /// The exact sizer and the estimator rate are two views of one number: on any whole-block
    /// count they must agree (modulo Q8_0's u32 rounding, which only ever rounds UP).
    #[test]
    fn kv_fmt_bytes_matches_the_per_elem_rate() {
        for dt in [
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::Q8_0,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Iq4Nl,
            DType::Turbo2,
            DType::Turbo3,
            DType::Turbo4,
        ] {
            for blocks in [1usize, 3, 128, 1024] {
                let elems = blocks * 128; // a multiple of both block sizes
                let exact = kv_fmt_bytes(dt, elems) as f64;
                let rate = kv_bytes_per_elem(dt) * elems as f64;
                assert!(
                    exact >= rate && exact - rate < 4.0,
                    "{dt:?} x {elems}: exact {exact} vs rate {rate}"
                );
            }
        }
    }

    /// `empty`/`0`/unset = off; anything else = on — over explicit values via [`flag_from`], so
    /// no test in this binary has to touch the process environment (see `crate::test_env`).
    #[test]
    fn flag_grammar() {
        assert!(!flag_from(None));
        for (raw, want) in [
            ("", false),
            ("0", false),
            ("1", true),
            ("00", true), // only the exact string "0" is off — matches the shipped grammar
            ("true", true),
            ("no", true),
        ] {
            assert_eq!(flag_from(Some(raw)), want, "raw={raw:?}");
        }
    }

    /// The same grammar as the CONFIG layer resolves it: `config::env`'s `flag` reader is
    /// `flag_from`, so a `Config` built from these spellings carries exactly the booleans above —
    /// which is what the spill sites now read.
    #[test]
    fn kv_overflow_config_field_follows_the_flag_grammar() {
        use crate::config::Config;
        for (raw, want) in [("", false), ("0", false), ("1", true), ("true", true)] {
            for key in ["INFR_KV_OVERFLOW", "INFR_ROCM_WEIGHT_OVERFLOW"] {
                let layer =
                    crate::config::env::parse(&|k| (k == key).then(|| raw.to_string())).expect(key);
                let cfg = Config::load_from_layers(&[layer]);
                let got = match key {
                    "INFR_KV_OVERFLOW" => cfg.kv.overflow,
                    _ => cfg.paging.rocm_weight_overflow,
                };
                assert_eq!(got, want, "{key}={raw:?}");
            }
        }
        // Unset ⇒ the shipped default, off — both spill classes.
        assert!(!Config::default().kv.overflow);
        assert!(!Config::default().paging.rocm_weight_overflow);
    }

    /// MiB → bytes, with `0` distinct from unset (it forces whole-host placement) — over explicit
    /// values via [`mib_from`], no environment.
    #[test]
    fn mib_grammar() {
        assert_eq!(mib_from(None), None);
        for (raw, want) in [
            ("0", Some(0)),
            ("1", Some(1024 * 1024)),
            ("  512  ", Some(512 * 1024 * 1024)),
            ("", None),
            ("abc", None),
            ("-1", None),
            ("1.5", None),
        ] {
            assert_eq!(mib_from(Some(raw)), want, "raw={raw:?}");
        }
    }

    /// The MiB CAPS as the read sites now take them: a `Config` field in MiB, [`mib_bytes`] to
    /// bytes. `Some(0)` (whole-host placement) must survive as `Some(0)`, not collapse to "no cap".
    #[test]
    fn mib_caps_come_off_the_config_in_mib() {
        use crate::config::Config;
        assert_eq!(mib_bytes(None), None);
        assert_eq!(mib_bytes(Some(0)), Some(0));
        assert_eq!(mib_bytes(Some(512)), Some(512 * 1024 * 1024));
        // Unset ⇒ no cap on either class; set ⇒ the MiB count, byte-converted at the site.
        assert_eq!(Config::default().kv.overflow_vram_mb, None);
        assert_eq!(Config::default().paging.rocm_weight_vram_mb, None);
        let layer = crate::config::env::parse(&|k| match k {
            "INFR_KV_OVERFLOW_VRAM_MB" => Some("  512  ".to_string()),
            "INFR_ROCM_WEIGHT_VRAM_MB" => Some("0".to_string()),
            _ => None,
        })
        .expect("env layer");
        let cfg = Config::load_from_layers(&[layer]);
        assert_eq!(mib_bytes(cfg.kv.overflow_vram_mb), Some(512 * 1024 * 1024));
        assert_eq!(mib_bytes(cfg.paging.rocm_weight_vram_mb), Some(0));
    }

    /// 12% of total, floored at 2 GiB, override in MiB — verbatim from ROCm's two inline copies
    /// (`kv_overflow_vram_reserve` / `weight_overflow_vram_reserve`), over [`reserve_from`] so the
    /// case sweep needs no environment.
    #[test]
    fn reserve_from_matches_the_inline_formula() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Below the crossover (2 GiB / 0.12 ≈ 16.67 GiB) the floor wins.
        assert_eq!(reserve_from(8 * GIB, None), 2 * GIB);
        assert_eq!(reserve_from(16 * GIB, None), 2 * GIB);
        // Above it, 12% — computed as `total / 100 * 12`, the shipped (truncating) order.
        assert_eq!(reserve_from(24 * GIB, None), 24 * GIB / 100 * 12);
        assert_eq!(reserve_from(0, None), 2 * GIB);
        // The override wins outright, including below the floor and at zero.
        assert_eq!(reserve_from(24 * GIB, Some("128")), 128 * 1024 * 1024);
        assert_eq!(reserve_from(24 * GIB, Some("0")), 0);
        assert_eq!(reserve_from(24 * GIB, Some("abc")), 24 * GIB / 100 * 12);
    }

    /// The same policy driven off the CONFIG field (MiB) instead of a string — what the two ROCm
    /// reserve sites now call. `reserve_from` is the string spelling of exactly this.
    #[test]
    fn reserve_bytes_takes_the_override_from_the_config() {
        use crate::config::Config;
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(reserve_bytes(24 * GIB, None), 24 * GIB / 100 * 12);
        assert_eq!(reserve_bytes(8 * GIB, None), 2 * GIB);
        assert_eq!(reserve_bytes(24 * GIB, Some(128)), 128 * 1024 * 1024);
        assert_eq!(reserve_bytes(24 * GIB, Some(0)), 0);
        // Unset in the shipped config ⇒ the formula, for both spill classes.
        let d = Config::default();
        assert_eq!(d.kv.overflow_reserve_mb, None);
        assert_eq!(d.paging.rocm_weight_reserve_mb, None);
        let layer = crate::config::env::parse(&|k| {
            (k == "INFR_KV_OVERFLOW_RESERVE_MB").then(|| "128".to_string())
        })
        .expect("env layer");
        let cfg = Config::load_from_layers(&[layer]);
        assert_eq!(
            reserve_bytes(24 * GIB, cfg.kv.overflow_reserve_mb),
            128 * 1024 * 1024
        );
    }

    /// The cumulative cap admits up to and including the cap, and `Some(0)` admits nothing —
    /// the whole-host reproducibility case.
    #[test]
    fn spill_tally_cap_boundaries() {
        let t = SpillTally::default();
        assert!(t.admits(None, u64::MAX));
        assert!(t.admits(Some(0), 0));
        assert!(!t.admits(Some(0), 1));
        assert!(t.admits(Some(100), 100)); // exactly the cap fits
        assert!(!t.admits(Some(100), 101));

        t.record_vram(60);
        assert!(t.admits(Some(100), 40));
        assert!(!t.admits(Some(100), 41));
        t.record_vram(40);
        assert!(t.admits(Some(100), 0));
        assert!(!t.admits(Some(100), 1));
        // Host placements never consume the VRAM cap.
        t.record_host(1 << 40);
        assert!(t.admits(Some(200), 100));

        let c = t.counts();
        assert_eq!(
            c,
            SpillCounts {
                vram_bufs: 2,
                vram_bytes: 100,
                host_bufs: 1,
                host_bytes: 1 << 40,
            }
        );
        assert_eq!(c.total_bufs(), 3);
    }

    /// The banner, byte-for-byte against the three pre-hoist `eprintln!`s (Vulkan KV, ROCm KV,
    /// ROCm weights). This text is what a user reads to tell a partial spill from a slow run.
    #[test]
    fn spill_report_line_reproduces_the_inline_banners() {
        /// Vulkan's `fmt_bytes` (MiB to one decimal).
        fn vk_fmt(b: u64) -> String {
            const KIB: u64 = 1 << 10;
            const MIB: u64 = 1 << 20;
            const GIB: u64 = 1 << 30;
            match b {
                b if b >= GIB => format!("{:.2} GiB", b as f64 / GIB as f64),
                b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
                b if b >= KIB => format!("{:.1} KiB", b as f64 / KIB as f64),
                b => format!("{b} B"),
            }
        }
        const VK_KV: SpillNouns = SpillNouns {
            env: "INFR_KV_OVERFLOW",
            noun: "KV buffers",
            resident_note: "no PCIe KV reads.",
            spill_note: "SYSTEM RAM — attention reads those K/V over PCIe by device address \
                         (PCIe-bound on the spilled layers). Spilled KV bytes are exempt from the \
                         VRAM budget guard.",
        };
        const ROCM_WT: SpillNouns = SpillNouns {
            env: "INFR_ROCM_WEIGHT_OVERFLOW",
            noun: "weight banks",
            resident_note: "no PCIe weight reads.",
            spill_note: "page-locked SYSTEM RAM — the native Linear/EmbedGather GEMV reads those \
                         over PCIe (PCIe-bound on the spilled banks). Spilled weights are exempt \
                         from VRAM.",
        };

        // Nothing allocated (or the flag off, which leaves the tally at zero) ⇒ no line.
        assert_eq!(
            spill_report_line(SpillCounts::default(), &VK_KV, vk_fmt),
            None
        );

        let all_resident = SpillCounts {
            vram_bufs: 56,
            vram_bytes: 3 << 30,
            host_bufs: 0,
            host_bytes: 0,
        };
        assert_eq!(
            spill_report_line(all_resident, &VK_KV, vk_fmt).unwrap(),
            "[infr] INFR_KV_OVERFLOW: all 56 KV buffers (3.00 GiB) fit in VRAM — none spilled; \
             no PCIe KV reads."
        );

        let split = SpillCounts {
            vram_bufs: 40,
            vram_bytes: 3 << 30,
            host_bufs: 16,
            host_bytes: 1 << 30,
        };
        assert_eq!(
            spill_report_line(split, &VK_KV, vk_fmt).unwrap(),
            "[infr] INFR_KV_OVERFLOW: 40 of 56 KV buffers (3.00 GiB resident) in VRAM, the \
             remaining 16 (1.00 GiB) in SYSTEM RAM — attention reads those K/V over PCIe by \
             device address (PCIe-bound on the spilled layers). Spilled KV bytes are exempt from \
             the VRAM budget guard."
        );

        assert_eq!(
            spill_report_line(all_resident, &ROCM_WT, vk_fmt).unwrap(),
            "[infr] INFR_ROCM_WEIGHT_OVERFLOW: all 56 weight banks (3.00 GiB) fit in VRAM — none \
             spilled; no PCIe weight reads."
        );
        assert_eq!(
            spill_report_line(split, &ROCM_WT, vk_fmt).unwrap(),
            "[infr] INFR_ROCM_WEIGHT_OVERFLOW: 40 of 56 weight banks (3.00 GiB resident) in VRAM, \
             the remaining 16 (1.00 GiB) in page-locked SYSTEM RAM — the native \
             Linear/EmbedGather GEMV reads those over PCIe (PCIe-bound on the spilled banks). \
             Spilled weights are exempt from VRAM."
        );
    }
}
