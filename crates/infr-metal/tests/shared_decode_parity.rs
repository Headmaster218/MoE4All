//! Metal block-decode parity on the **shared** `infr-testkit` harness (backend-unification
//! candidate H).
//!
//! `tests/parity.rs` already asserts per-format quant `Linear` parity, but it does so through ~60
//! hand-written tests over a per-suite family of `synth_q4k`/`synth_q6k`/`synth_mxfp4`/… builders
//! that ROCm's suite independently re-implements. This file gets the same coverage as ONE sweep,
//! from `infr_core::decode_spec` via `infr_testkit::synth_weight` — so a new quant format is
//! covered on cpu / rocm / metal / vulkan the moment it is added to the spec, with no fourth copy
//! of the block layouts.
//!
//! It also closes a real gap in the existing suite: the per-format tests each pick one or two
//! shapes ad hoc, so no single thing sweeps ALL 24 formats through the same m-tier. Here `m=1`
//! (the decode GEMV) and `m=4` (the multi-row `MRV_BAND` tier, see `infr_core::tier`) are both
//! swept across the whole roster.
//!
//! macOS-only (the backend is), and `#[ignore]`d — needs a real Metal device:
//!
//!   cargo test -p infr-metal --test shared_decode_parity -- --include-ignored --nocapture
#![cfg(target_os = "macos")]

use infr_core::DType;
use infr_metal::MetalBackend;
use infr_testkit::{sweep_linear_on, weight_quant_cases, LinearCase};

/// Relative tolerance for a Metal quant `Linear` against the f32 host oracle.
///
/// Metal decodes the raw GGUF block in-kernel for every one of the 24 weight formats
/// (`weight_qui`'s `native_kern` table) and accumulates in float, so the gap here is the decode's
/// own f16/half-precision scale arithmetic plus f32 reassociation — the same two lossy stages the
/// ROCm sweep bounds at `2e-2`, which is why this carries the identical figure.
///
/// It is deliberately NOT tighter than that: this sweep's job is COVERAGE (all 24 formats × two
/// m-tiers from one source), and the existing per-format tests in `tests/parity.rs` already hold
/// Metal to a considerably tighter `1e-3` on the shapes they pick. Tighten this once it has been
/// measured on a Mac — the author of this slice has ROCm/Vulkan hardware only, so the number
/// chosen here is the defensible upper bound rather than a measured one.
const METAL_TOL: f32 = 2e-2;

/// All 24 weight quants at **m=1** — the decode GEMV tier (`linear_*` native block decode).
#[test]
#[ignore = "requires a Metal GPU"]
fn metal_linear_all_weight_quants_m1_match_the_host_oracle() {
    let Ok(be) = MetalBackend::new() else {
        return; // no Metal device — self-skip
    };
    // in_f=256 divides by every block size (32/64/256), so one shape covers all 24 formats.
    let cases = weight_quant_cases(1, 256, 8);
    sweep_linear_on("metal Linear m=1", &be, &cases, |_| METAL_TOL).assert_ok();
}

/// All 24 weight quants at **m=4** — the multi-row (`MRV_BAND`, m=2..=8) tier, which is a
/// different kernel family from the m=1 GEMV. A format whose two tiers disagree is exactly the bug
/// class that broke Q5_K MTP token identity on Vulkan.
///
/// The `m >= 16` half-fragment coop-GEMM tier is deliberately NOT swept here: that kernel rounds
/// BOTH weights and activations to f16 before the dot, so an f32 oracle is the wrong comparand for
/// it (`tests/parity.rs` handles it by mirroring the rounding into its reference via `half_ops`).
/// Teaching the harness that mode is follow-up work.
#[test]
#[ignore = "requires a Metal GPU"]
fn metal_linear_all_weight_quants_m4_match_the_host_oracle() {
    let Ok(be) = MetalBackend::new() else {
        return;
    };
    let cases = weight_quant_cases(4, 256, 32);
    sweep_linear_on("metal Linear m=4", &be, &cases, |_| METAL_TOL).assert_ok();
}

/// The dense float weight paths, which take no quant decode — the oracle reads the SAME bytes the
/// GPU does, so the only gap is f32 summation order.
#[test]
#[ignore = "requires a Metal GPU"]
fn metal_linear_dense_dtypes_match_the_host_oracle() {
    let Ok(be) = MetalBackend::new() else {
        return;
    };
    let cases = [
        LinearCase::new(DType::F32, 1, 256, 8).with_seed(0x501),
        LinearCase::new(DType::F16, 1, 256, 8).with_seed(0x502),
        LinearCase::new(DType::F16, 4, 256, 32).with_seed(0x503),
    ];
    sweep_linear_on("metal Linear dense", &be, &cases, |_| 1e-4).assert_ok();
}
