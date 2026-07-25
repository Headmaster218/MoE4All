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

use infr_core::DType;
use infr_testkit::{sweep_linear_on, weight_quant_cases, LinearCase};

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
