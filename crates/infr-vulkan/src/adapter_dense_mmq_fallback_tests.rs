#![cfg(test)]

use super::*;
use infr_core::tensor::{DType, TensorDesc};

#[test]
fn dense_mmq_coverage_distinguishes_expert_only_formats() {
    for &dtype in infr_core::tensor::MOE_MMQ_DTYPES {
        let kernel = crate::gemm::native_gemm_mmq_dense_spv(dtype);
        if matches!(dtype, DType::Iq2Xs | DType::Iq3Xxs) {
            assert!(
                kernel.is_none(),
                "{dtype:?} unexpectedly has a dense MMQ build"
            );
            assert!(native_dense_supported(dtype), "{dtype:?} lost its fallback");
        } else {
            let (name, spv) =
                kernel.unwrap_or_else(|| panic!("{dtype:?} lost its dense MMQ build"));
            assert!(name.starts_with("native_gemm_mmq_"));
            assert_eq!(spv.first(), Some(&0x0723_0203), "{name} is not SPIR-V");
        }
    }
    for &dtype in crate::linear::native_dense_dtypes() {
        if crate::gemm::native_gemm_mmq_dense_spv(dtype).is_some() {
            assert!(infr_core::tensor::moe_mmq_ok(dtype));
        }
    }
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        assert!(crate::gemm::native_gemm_mmq_dense_spv(dtype).is_none());
    }
}

fn backend(coopmat: bool) -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = std::env::var("MOE4ALL_TEST_DEVICE").ok();
    cfg.kernels.vulkan.coopmat = coopmat;
    let be = VulkanBackend::new_with(Arc::new(cfg)).expect("Vulkan test backend");
    println!(
        "device={} coopmat={}",
        be.caps().name,
        be.caps().f16_coopmat()
    );
    be
}

fn check_linear(be: &VulkanBackend, dtype: DType, rows: usize, offset_rows: usize) {
    let (in_f, out_f) = (256usize, 64usize);
    // Every 32-value block has amax=127/128, so quant_q8 is exact. This lets the same
    // CPU reference check both the MMQ route and the unquantized GEMV fallback.
    let x = (0..rows * in_f)
        .map(|i| {
            let code = if i % 32 == 0 {
                127
            } else {
                ((i * 17 + i / in_f * 13) % 255) as i32 - 127
            };
            code as f32 / 128.0
        })
        .collect::<Vec<_>>();
    let mut w = infr_testkit::synth_weight(dtype, (out_f + offset_rows) * in_f, 0x127d_ea5e);
    if be.caps().f16_coopmat() {
        // Keep the control's decoded weights exactly representable in F16, so its
        // staging precision does not obscure the routing regression being checked.
        let spec = infr_core::decode_spec::block_spec(dtype);
        for block in w.chunks_exact_mut(spec.block_bytes) {
            spec.write_scales(block, 1.0 / 1024.0, 1.0 / 1024.0);
        }
    }
    let decoded = infr_testkit::dequant_oracle(dtype, &w);
    if be.caps().f16_coopmat() {
        assert!(decoded
            .iter()
            .all(|&v| half::f16::from_f32(v).to_f32() == v));
    }
    let offset = offset_rows * in_f;
    let want = infr_testkit::ref_linear(
        &x,
        &decoded[offset..offset + out_f * in_f],
        rows,
        in_f,
        out_f,
    );
    assert!(want.iter().all(|v| v.is_finite()));
    assert!(want.iter().any(|v| v.abs() > 1e-3));
    let mut graph = Graph::new();
    let xi = graph.input(TensorDesc::new(vec![rows, in_f], DType::F32));
    let wi = graph.weight(TensorDesc::new(vec![out_f + offset_rows, in_f], dtype));
    let yi = graph.output(TensorDesc::new(vec![rows, out_f], DType::F32));
    graph.push(Op::Linear {
        x: xi,
        weight: wi,
        dst: yi,
        m: rows as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: offset as u32,
    });
    let got = infr_testkit::run_graph(
        be,
        &graph,
        &[(xi, infr_testkit::f32_bytes(&x)), (wi, w)],
        yi,
        rows * out_f,
    );
    let mut max_error = 0.0f32;
    for (index, (&actual, &expected)) in got.iter().zip(&want).enumerate() {
        let error = (actual - expected).abs();
        max_error = max_error.max(error);
        assert!(
            actual.is_finite() && error <= 2e-3 * expected.abs().max(1.0),
            "{dtype:?} rows={rows} offset_rows={offset_rows} index={index}: \
             got {actual}, expected {expected}"
        );
    }
    println!(
        "dense Linear {dtype:?} rows={rows} offset_rows={offset_rows} max_abs_error={max_error:e}"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU with integer dot products"]
fn missing_dense_mmq_formats_fall_back_and_match_host() {
    let be = backend(false);
    assert!(!be.caps().f16_coopmat());
    assert!(be.caps().i8_dot, "test must exercise MMQ admission");
    for (dtype, fallback) in [
        (DType::Iq2Xs, "native_iq2xs"),
        (DType::Iq3Xxs, "native_iq3xxs"),
    ] {
        // Start with a genuine GEMM-shaped prefill; before the fix this panics.
        for rows in [70, 128, 1, 2] {
            for offset_rows in [0, 1] {
                check_linear(&be, dtype, rows, offset_rows);
            }
        }
        assert!(be.built_kernel_names().contains(&fallback));
    }
}

#[test]
#[ignore = "requires a Vulkan GPU with integer dot products"]
fn supported_dense_mmq_formats_keep_their_kernels_and_match_host() {
    let be = backend(false);
    assert!(!be.caps().f16_coopmat());
    assert!(be.caps().i8_dot, "test must exercise MMQ admission");
    for (dtype, expected) in [
        (DType::Q8_0, "native_gemm_mmq_q8_0"),
        (DType::Q4K, "native_gemm_mmq_q4k_streamed"),
        (DType::Iq4Xs, "native_gemm_mmq_iq4_xs"),
    ] {
        for rows in [70, 128] {
            for offset_rows in [0, 1] {
                check_linear(&be, dtype, rows, offset_rows);
            }
        }
        assert!(be.built_kernel_names().contains(&expected));
    }
}

#[test]
#[ignore = "requires a Vulkan GPU with cooperative matrix"]
fn coopmat_dense_routes_still_match_host() {
    let be = backend(true);
    assert!(be.caps().f16_coopmat());
    for dtype in [DType::Iq2Xs, DType::Iq3Xxs, DType::Q4K, DType::Q8_0] {
        for rows in [70, 128] {
            check_linear(&be, dtype, rows, 0);
        }
    }
}
