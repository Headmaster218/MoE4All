//! ROCm block-decode parity on the **shared** `infr-testkit` harness (backend-unification
//! candidate H).
//!
//! What this adds over `tests/parity.rs`'s `all_quant_linear_matches_cpu`:
//!
//! 1. **A stronger oracle.** The legacy sweep compares ROCm against `infr_cpu::CpuBackend`, which
//!    int8-quantizes the activation row for a quant `Linear` — so it is itself lossy, and a shared
//!    decode bug in a format BOTH backends get wrong would cancel out. Here the comparand is
//!    `infr_gguf::dequant::dequant_block` (the host oracle) plus an f64-accumulated reference GEMV,
//!    which has no int8 stage at all.
//! 2. **The tiers the legacy sweep does not reach.** It runs one shape (`m=2`); ROCm routes
//!    `m == 1` to the `linear_i8_*` dp4a GEMV / native-decode GEMV and `m > 1` to the WMMA
//!    matrix-core prefill GEMM (`native_wmma_fmt`, `wmma_i8_q4k_pipe_2x1` for Q4_K). Both are
//!    swept here across all 24 weight formats.
//! 3. **One source.** The synthetic weights come from `infr_core::decode_spec` via
//!    `infr_testkit::synth_weight`, not from a per-suite `synth_q`/`synth_mxfp4`/`synth_nvfp4`/
//!    `synth_iq1m` table — so adding a quant format gives cpu / rocm / metal / vulkan coverage at
//!    once.
//!
//! Every test is `#[ignore]`d (needs the RX 7900 XTX dev box):
//!
//!   cargo test -p infr-rocm --features rocm -- --include-ignored

#![cfg(all(target_os = "linux", feature = "rocm"))]

use infr_core::DType;
use infr_rocm::RocmBackend;
use infr_testkit::{sweep_linear, weight_quant_cases, LinearCase};

/// A fresh backend on device 0. `dequant_weight_or_cache` keys its dequantized-weight cache by the
/// weight's raw device POINTER, and a weight buffer freed at the end of one case can have its VRAM
/// address recycled by the next — a stale hit would feed the previous format's rows. So every case
/// gets its own backend (and therefore its own empty cache), exactly as `tests/parity.rs` does.
fn fresh() -> RocmBackend {
    RocmBackend::new(0).expect("ROCm device 0")
}

fn have_rocm() -> bool {
    RocmBackend::new(0).is_ok()
}

/// Relative tolerance for a ROCm quant `Linear` against the f32 host oracle.
///
/// Two lossy stages sit between them, and neither is a decode error:
///   * the **weight** round-trip — the uncovered formats are host-dequantized to **f16** before
///     the GEMV, so each weight carries up to one f16 ULP (~5e-4 relative);
///   * the **activation** round-trip — the covered formats (Q8_0/Q4_K/Q6_K/Q5_0) take the int8
///     dp4a path, quantizing `x` to 32-element int8 blocks (~`amax/254` per element).
///
/// Both accumulate over a 256-deep dot with partial cancellation, so `2e-2` relative to
/// `max|oracle|` is the honest joint bound — the same figure `tests/parity.rs` settled on against
/// the (also-lossy) CPU comparand.
///
/// **Measured on an RX 7900 XTX (gfx1100), all 24 formats at both m-tiers: worst case `4.7e-3`**
/// (Q6_K at m=1, the int8 dp4a GEMV), so this carries ~4x headroom. The formats that stay on the
/// f32-exact path (TQ1_0/TQ2_0/Q2_0/MXFP4/NVFP4) come in at ~3e-7. A genuinely mis-decoded format
/// lands at O(1) relative, orders of magnitude above the bound — it is not hiding anything.
const ROCM_TOL: f32 = 2e-2;

/// All 24 weight quants at **m=1** — the decode GEMV tier (`linear_i8_*` for the four natively
/// covered formats, dequant→f16 GEMV for the rest). Not covered by the legacy `m=2` sweep.
#[test]
#[ignore = "requires a ROCm GPU"]
fn rocm_linear_all_weight_quants_m1_match_the_host_oracle() {
    if !have_rocm() {
        return;
    }
    // in_f=256 divides by every block size (32/64/256), so one shape covers all 24 formats.
    let cases = weight_quant_cases(1, 256, 8);
    sweep_linear("rocm Linear m=1", &cases, |_| ROCM_TOL, fresh).assert_ok();
}

/// All 24 weight quants at **m=16** — the WMMA matrix-core prefill GEMM tier. `out_f=32` (two
/// 16-wide column tiles) and `m=16` (one row tile) so the RM×CN register-tiled kernels are
/// actually entered rather than falling into a tail guard.
#[test]
#[ignore = "requires a ROCm GPU"]
fn rocm_linear_all_weight_quants_m16_match_the_host_oracle() {
    if !have_rocm() {
        return;
    }
    let cases = weight_quant_cases(16, 256, 32);
    sweep_linear("rocm Linear m=16", &cases, |_| ROCM_TOL, fresh).assert_ok();
}

/// The dense float weight paths (F16), which take no quant decode at all — the oracle reads the
/// SAME f16 bytes the GPU does, so the only gap is f32 summation order.
#[test]
#[ignore = "requires a ROCm GPU"]
fn rocm_linear_f16_matches_the_host_oracle() {
    if !have_rocm() {
        return;
    }
    let cases = [
        LinearCase::new(DType::F16, 1, 256, 8).with_seed(0x401),
        LinearCase::new(DType::F16, 16, 256, 32).with_seed(0x402),
    ];
    // Reassociation-only: a 256-term f32 dot reordered against an f64 reference.
    sweep_linear("rocm Linear F16", &cases, |_| 1e-4, fresh).assert_ok();
}
