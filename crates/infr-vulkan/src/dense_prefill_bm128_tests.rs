//! Extends the original Q8 Graph/Recorder contract to large Q8 and Q6_K projections.
//! Mounted under the existing contract module to share its guard/offset fixtures.
use super::*;

#[test]
fn dense_prefill_policy_preserves_unqualified_routes() {
    use crate::recorder::dense_prefill_bm128_applies as applies;
    for dtype in [DType::Q8_0, DType::Q6K] {
        for (k, n) in [(2048, 4096), (2048, 8192)] {
            let min_rows: usize = if dtype == DType::Q8_0 { 1024 } else { 2048 };
            for m in [
                1, 8, 16, 512, 1024, 1536, 2047, 2048, 2049, 4096, 8192, 12288, 12416,
            ] {
                let expected = (min_rows..=12288).contains(&m) && m.is_multiple_of(128);
                assert_eq!(applies(true, true, true, dtype, m, k, n), expected);
                assert!(!applies(false, true, true, dtype, m, k, n));
                assert!(!applies(true, false, true, dtype, m, k, n));
                assert!(!applies(true, true, false, dtype, m, k, n));
            }
        }
    }
    for dtype in [
        DType::F16,
        DType::Bf16,
        DType::Q4K,
        DType::Q5K,
        DType::Iq4Nl,
    ] {
        assert!(!applies(true, true, true, dtype, 4096, 2048, 8192));
    }
    for (k, n) in [(2048, 2048), (2048, 4097), (512, 2048), (2560, 640)] {
        assert!(!applies(true, true, true, DType::Q8_0, 4096, k, n));
    }
    for (k, n, m) in QUALIFIED {
        assert!(applies(true, true, true, DType::Q8_0, m, k, n));
    }
    for (k, n, m) in EXCLUDED_FLASH_M512 {
        assert!(!applies(true, true, true, DType::Q8_0, m, k, n));
    }
    for n in [10240, 12288] {
        assert!(applies(true, true, true, DType::Q8_0, 4096, 2560, n));
        assert!(!applies(true, true, true, DType::Q6K, 4096, 2560, n));
        assert!(!applies(true, true, true, DType::Q8_0, 4097, 2560, n));
    }
    assert!(crate::gemm::native_gemm_warp_n128_ag_bm128_kernel_name(DType::Bf16).is_none());
}

fn q6_raw(k: usize, n: usize) -> Vec<u8> {
    let mut bytes = vec![0; (n + 1) * k / 256 * 210];
    for (i, block) in bytes.chunks_exact_mut(210).enumerate() {
        let mut rng = (i as u32).wrapping_mul(0x85ebca6b) ^ 0x6b128;
        for byte in &mut block[..208] {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            *byte = rng as u8;
        }
        block[208..].copy_from_slice(&half::f16::from_f32(1.0 / 8192.0).to_le_bytes());
    }
    bytes
}

#[test]
#[ignore = "GPU: explicit XTX; actual Q8/Q6_K Linear, nonzero weight offset and output guards"]
fn dense_prefill_bm128_graph_exact_and_ragged_fallback() {
    let off = backend(false);
    let on = backend(true);
    for (dtype, k, n, m) in [
        (DType::Q6K, 2048, 8192, 2048),
        (DType::Q6K, 2048, 4096, 2048),
        (DType::Q8_0, 2048, 4096, 1024),
        (DType::Q8_0, 2560, 12288, 2048),
        (DType::Q8_0, 2560, 10240, 2048),
        (DType::Q6K, 2048, 4096, 2049),
    ] {
        let bytes = if dtype == DType::Q6K {
            q6_raw(k, n)
        } else {
            raw(k, n)
        };
        let qualified =
            crate::recorder::dense_prefill_bm128_applies(true, true, true, dtype, m, k, n);
        assert_eq!(split_k_plan(m, k, n), 1);
        for internal in [false, true] {
            let a = Case::new_typed(&off, k, n, m, internal, &bytes, dtype);
            let b = Case::new_typed(&on, k, n, m, internal, &bytes, dtype);
            for seed in 0..2 {
                let values = input(m, k, seed);
                let baseline = a.execute(&off, &values);
                let candidate = b.execute(&on, &values);
                let label = format!("{dtype:?} M{m} K{k} N{n} internal={internal} seed={seed}");
                exact_bits(&label, &baseline, &candidate);
                // Direct Recorder outputs require row padding; the Graph exercises
                // its own padded-destination/copy contract for ragged external outputs.
                if m.is_multiple_of(64) {
                    exact_bits(&format!("Recorder {label}"), &candidate, &b.recorder(&on));
                }
            }
            a.unchanged_weights(&off, &bytes);
            b.unchanged_weights(&on, &bytes);
        }
        if qualified {
            let name = crate::gemm::native_gemm_warp_n128_ag_bm128_kernel_name(dtype).unwrap();
            assert!(on.built_kernel_names().contains(&name));
            assert!(!off.built_kernel_names().contains(&name));
        }
        eprintln!("DENSE_BM128 dtype={dtype:?} M={m} K={k} N={n} qualified={qualified} exact=true offset=true guards=true");
    }
}
