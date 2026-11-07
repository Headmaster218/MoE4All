//! Vulkan block-decode parity on the **shared** `infr-testkit` harness (backend-unification
//! candidate H).
//!
//! Vulkan already has ~75 `*_matches_host` tests, so this file is not closing a coverage hole the
//! way the Metal ones do. Its jobs are:
//!
//! 1. **Pin the dense-Linear dtype ROSTER against the decode spec.** `linear::native_dense_dtypes`
//!    claims every one of `infr_core::decode_spec::WEIGHT_QUANTS`; this sweep actually runs all 24
//!    through `Op::Linear` and scores them against the host oracle, so "claimed" and "works" stay
//!    the same set. The existing per-format tests each cover one or two formats at hand-picked
//!    shapes — none of them sweeps the roster.
//! 2. **Prove the harness is backend-agnostic**, by getting the same coverage on a third backend
//!    from the same source (`infr_testkit::synth_weight`, driven by the shared spec) with no
//!    Vulkan-specific API involved — only the `Backend` seam.
//!
//! `#[ignore]`d like every Vulkan GPU test:
//!
//!   cargo test -p infr-vulkan --test shared_decode_parity -- --include-ignored --nocapture

use infr_core::DType;
use infr_testkit::{sweep_linear_on, weight_quant_cases, LinearCase};
use infr_vulkan::VulkanBackend;

/// Relative tolerance for a Vulkan quant `Linear` against the f32 host oracle.
///
/// **Measured on an RX 7900 XTX (RADV, Navi31), all 24 formats at both m-tiers: worst case
/// `1.6e-7`** — i.e. f32 reassociation noise and nothing else. At these narrow shapes
/// (`out_f` 8/32) the tier policy keeps every format on the f32-exact GEMV rather than the int8
/// dp4a decode kernels, so no activation quantization enters the comparison at all.
///
/// The bound is nonetheless set at the same `2e-2` ceiling the Metal sweep carries, deliberately:
/// the m-tier and per-dtype int8 routing (`adapter::mmv_int8_decode_dtypes`,
/// `infr_core::tier::linear_tier`) is a POLICY that legitimately moves, and when a shape here
/// lands on `quant_q8` → `linear_mmv_mw`/`linear_mmv_mrow` its error jumps to the ~5e-3 band that
/// int8 activations cost (that is exactly what the Metal sweep measures on the same formats). A
/// tighter number would turn a deliberate tier change into a spurious parity failure. What this
/// sweep is for is the DECODE — and a mis-decoded format lands at O(1) relative, four orders of
/// magnitude above the ceiling.
const VK_TOL: f32 = 2e-2;

/// All 24 weight quants at **m=1** — the decode GEMV tier.
#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn vulkan_linear_all_weight_quants_m1_match_the_host_oracle() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // in_f=256 divides by every block size (32/64/256), so one shape covers all 24 formats.
    let cases = weight_quant_cases(1, 256, 8);
    sweep_linear_on("vulkan Linear m=1", &be, &cases, |_| VK_TOL).assert_ok();
}

/// All 24 weight quants at **m=4** — the multi-row (`MROW_BANDS`, m=2..=8) tier, a different
/// kernel family from the m=1 GEMV. A dtype that is int8 in one stream and f32-exact in the other
/// is exactly the bug class that broke Q5_K MTP token identity.
#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn vulkan_linear_all_weight_quants_m4_match_the_host_oracle() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let cases = weight_quant_cases(4, 256, 32);
    sweep_linear_on("vulkan Linear m=4", &be, &cases, |_| VK_TOL).assert_ok();
}

/// The dense float weight paths, which take no quant decode — the oracle reads the SAME bytes the
/// GPU does, so the only gap is f32 summation order.
#[test]
#[ignore = "requires a Vulkan-capable GPU"]
fn vulkan_linear_dense_dtypes_match_the_host_oracle() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let cases = [
        LinearCase::new(DType::F32, 1, 256, 8).with_seed(0x601),
        LinearCase::new(DType::F16, 1, 256, 8).with_seed(0x602),
        LinearCase::new(DType::F16, 4, 256, 32).with_seed(0x603),
    ];
    sweep_linear_on("vulkan Linear dense", &be, &cases, |_| 1e-4).assert_ok();
}
