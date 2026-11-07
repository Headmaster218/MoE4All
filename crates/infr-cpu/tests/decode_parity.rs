//! Per-dtype block-decode parity for the CPU interpreter, on the shared `infr-testkit` harness.
//!
//! This is the harness's **GPU-less** instantiation: it needs no device, so it runs in a plain
//! `cargo test` and is the thing that proves the harness itself (synth → oracle → seam → score)
//! before any GPU test relies on it.
//!
//! Why it is not redundant with the oracle it compares against: `infr-cpu`'s quant `Linear` does
//! NOT call `dequant_block`. It quantizes the activation row to int8 and runs a per-format
//! `vec_dot_*` kernel (`infr_cpu::kernels`) that decodes the weight bytes itself — an INDEPENDENT
//! decoder from `infr_gguf::dequant`. So this sweep is a genuine two-implementation cross-check of
//! all 24 weight formats, not a tautology.
//!
//! Run: `cargo test -p infr-cpu --test decode_parity -- --nocapture`

use infr_core::config::{Config, CpuCfg, KernelCfg};
use infr_core::DType;
use infr_testkit::{sweep_linear_on, weight_quant_cases, LinearCase};
use std::sync::Arc;

/// A reference-mode backend built the way the config campaign says a mode is chosen: a typed field
/// on a `Config` the caller owns (`kernels.cpu.reference`), handed to
/// [`infr_cpu::CpuBackend::new_with`]. No environment, no global, no ordering hazard — these tests
/// run in parallel with the production sweeps in the same binary and cannot disturb them.
///
/// [`infr_cpu::CpuBackend::reference()`] is this same config, spelled shorter, and
/// `cpu_production_matches_the_reference_backend` below keeps using it so the shorthand stays
/// pinned to the field.
fn reference_backend() -> infr_cpu::CpuBackend {
    let cfg = Config {
        kernels: KernelCfg {
            cpu: CpuCfg {
                reference: true,
                ..CpuCfg::default()
            },
            ..KernelCfg::default()
        },
        ..Config::default()
    };
    infr_cpu::CpuBackend::new_with(Arc::new(cfg))
}

/// Relative tolerance for the CPU quant `Linear` path against the f32 oracle.
///
/// The gap is NOT decode error — both sides decode the same bytes — it is the CPU kernel's **int8
/// activation quantization**: `x` is quantized to 32-element int8 blocks before the dot, so each
/// activation carries up to `amax/254` of error, and the dot sums `in_f` of them. Relative to
/// `max|oracle|` that lands at the low `1e-2`s for a 256-deep dot over the harness's activation
/// band, regardless of the WEIGHT format — which is why this is one number and not a per-format
/// table. `2e-2` is that bound with headroom; anything above it is a decode bug, not rounding.
///
/// The one exception is Q8_0, whose weight codes are already int8: nothing about it is tighter on
/// the CPU path (the activation quant dominates), so it takes the same bound.
///
/// Measured worst case over all 24 formats at both `m=1` and `m=2`: `7.6e-3` (Q2_K), i.e. this
/// bound carries ~2.6x headroom — enough that a different host/opt-level cannot flake it, tight
/// enough that a mis-decoded format (which lands at O(1) relative) cannot hide under it.
const CPU_INT8_ACT_TOL: f32 = 2e-2;

/// Every real GGUF weight quant (24) through a one-`Op::Linear` graph on `CpuBackend`, scored
/// against `dequant_block` + a f64-accumulated reference GEMV.
#[test]
fn cpu_linear_all_weight_quants_match_the_host_oracle() {
    let be = infr_cpu::CpuBackend::new();
    // in_f=256 is divisible by every block size (32/64/256), so one shape covers all 24 formats;
    // m=2 exercises the batched (multi-row) dot path.
    let cases = weight_quant_cases(2, 256, 8);
    sweep_linear_on("cpu Linear m=2", &be, &cases, |_| CPU_INT8_ACT_TOL).assert_ok();
}

/// The same sweep at `m=1` — the DECODE shape, which routes through the single-row `vec_dot_*`
/// kernels rather than the `*_batch` ones. A format whose batched and single-row decoders disagree
/// (the bug class that broke Q5_K MTP token identity on Vulkan) shows up as one of these failing.
#[test]
fn cpu_linear_all_weight_quants_match_the_host_oracle_at_m1() {
    let be = infr_cpu::CpuBackend::new();
    let cases = weight_quant_cases(1, 256, 8);
    sweep_linear_on("cpu Linear m=1", &be, &cases, |_| CPU_INT8_ACT_TOL).assert_ok();
}

/// A backend with `kernels.cpu.reference` SET, on the SAME sweep: it decodes the weight to f32 and
/// takes an f32 dot, so the int8-activation term above is gone entirely and only summation order is
/// left. That is what makes it usable as the oracle a GPU-vs-CPU token comparison is scored against
/// (`gpu_seam_matches_cpu_qwen3_q2k` failed for exactly this reason: the production CPU leg's ~4e-3
/// moved a greedy token at Q2_K while the GPU sat at ~1e-7).
///
/// This is also the `kernels.cpu.reference` knob's behaviour test (`docs/config-plan.md` §8): the
/// mode is selected purely by the `Config` handed to the backend, and the assertion is the numeric
/// consequence of picking it.
///
/// The `1e-5` bound is ~3 orders tighter than [`CPU_INT8_ACT_TOL`] — if reference mode ever silently
/// falls back to an int8 kernel, this test fails while the production sweeps stay green.
#[test]
fn cpu_reference_mode_is_exact_for_every_weight_quant() {
    let be = reference_backend();
    for m in [1usize, 2] {
        let cases = weight_quant_cases(m, 256, 8);
        sweep_linear_on(&format!("cpu REFERENCE Linear m={m}"), &be, &cases, |_| {
            1e-5
        })
        .assert_ok();
    }
}

/// The shorthand constructor and the config field are the SAME backend: `CpuBackend::reference()`
/// is `new_with(cfg)` with `kernels.cpu.reference` set, so the two must agree bit-for-bit (not
/// within a tolerance). This is what lets the GPU-vs-CPU seam oracle keep calling the short
/// spelling while the mode itself lives in the `Config`.
#[test]
fn reference_shorthand_is_the_config_field() {
    let shorthand = infr_cpu::CpuBackend::reference();
    let from_cfg = reference_backend();
    for case in weight_quant_cases(1, 256, 8) {
        assert_eq!(
            infr_testkit::run_linear(&shorthand, case),
            infr_testkit::run_linear(&from_cfg, case),
            "{:?}: CpuBackend::reference() diverged from kernels.cpu.reference",
            case.dtype
        );
    }
}

/// PRODUCTION vs REFERENCE, in-engine: the same `Op::Linear` graph run on `CpuBackend::new()` and
/// on `CpuBackend::reference()`, scored against each other rather than against the host oracle.
///
/// The two sweeps above each pin one backend mode to `dequant_block`; this pins the two modes to
/// EACH OTHER, which is what the GPU-vs-CPU seam comparison actually depends on. The reference is
/// the slow-but-accurate implementation (~1e-5 vs the oracle); production is the fast int8-activation
/// one (~4e-3). So this test states the production path's error BUDGET against the accurate path:
/// inside `CPU_INT8_ACT_TOL` is rounding, outside it is a kernel bug — including a SIMD kernel that
/// decodes a format wrongly on one machine's tier and not another's (see
/// `kernels::kernel_tests::simd_tiers_match_the_scalar_kernel` for the per-tier version).
#[test]
fn cpu_production_matches_the_reference_backend() {
    let prod = infr_cpu::CpuBackend::new();
    let refb = infr_cpu::CpuBackend::reference();
    for m in [1usize, 2] {
        for case in weight_quant_cases(m, 256, 8) {
            let got = infr_testkit::run_linear(&prod, case);
            let want = infr_testkit::run_linear(&refb, case);
            let mag = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            assert!(
                mag > 1e-3,
                "{:?} m={m}: reference output is all-zero",
                case.dtype
            );
            let rel = got
                .iter()
                .zip(&want)
                .fold(0f32, |a, (g, w)| a.max((g - w).abs()))
                / mag;
            assert!(
                rel < CPU_INT8_ACT_TOL,
                "{:?} m={m}: production deviates {rel:.3e} from the reference backend \
                 (budget {CPU_INT8_ACT_TOL:.1e})",
                case.dtype
            );
        }
    }
}

/// The dense float weight paths, which take no int8 activation quantization at all — so the bound
/// is the f16 weight rounding (F16) or bit-exactness modulo f32 reassociation (F32).
#[test]
fn cpu_linear_dense_dtypes_match_the_host_oracle() {
    let be = infr_cpu::CpuBackend::new();
    let cases = [
        LinearCase::new(DType::F32, 2, 256, 8).with_seed(0x301),
        LinearCase::new(DType::F16, 2, 256, 8).with_seed(0x302),
    ];
    sweep_linear_on("cpu Linear dense", &be, &cases, |dt| match dt {
        // f32 weights: only the summation ORDER differs from the f64 reference.
        DType::F32 => 1e-5,
        // f16 weights: the oracle reads the same f16 bytes, so this too is reassociation-only.
        _ => 1e-5,
    })
    .assert_ok();
}
