//! Production graph coverage; ignored GPU cases require explicit XTX selection.
use super::*;
use crate::narrow_projection::{device_eligible, scratch_bytes, MAX_SCRATCH_BYTES};
use infr_core::{DType, TensorDesc};

#[test]
fn narrow_projection_policy_boundaries_and_fallbacks() {
    let cfg = VulkanCfg::default();
    for dt in [DType::Q8_0, DType::Bf16] {
        for (k, n, min_m) in [(2560, 48, 256), (10240, 4, 1024)] {
            for m in [min_m, min_m + 3, 4096] {
                for off in [0, k, k * 3] {
                    let bytes = scratch_bytes(&cfg, true, dt, m, k, n, off).unwrap();
                    assert_eq!(bytes, k * n * 4);
                    assert!(bytes <= MAX_SCRATCH_BYTES);
                }
            }
            for m in [0, 1, 2, 8, 16, min_m - 1] {
                assert_eq!(scratch_bytes(&cfg, true, dt, m, k, n, 0), None);
            }
            assert_eq!(scratch_bytes(&cfg, false, dt, 4096, k, n, 0), None);
            assert_eq!(scratch_bytes(&cfg, true, dt, 4096, k, n, 1), None);
            assert_eq!(scratch_bytes(&cfg, true, dt, usize::MAX, k, n, 0), None);
            assert_eq!(scratch_bytes(&cfg, true, dt, 4096, k, n, usize::MAX), None);
            for knob in 0..3 {
                let mut disabled = cfg.clone();
                match knob {
                    0 => disabled.narrow_f32_gemv = false,
                    1 => disabled.f32_mrow = false,
                    _ => disabled.f32_v4 = false,
                }
                assert_eq!(scratch_bytes(&disabled, true, dt, 4096, k, n, 0), None);
            }
        }
    }
    for dt in [DType::F16, DType::F32, DType::Q4K, DType::Q6K] {
        assert_eq!(scratch_bytes(&cfg, true, dt, 4096, 2560, 48, 0), None);
    }
    for (k, n) in [
        (2560, 64),
        (2560, 47),
        (10240, 48),
        (0, 4),
        (usize::MAX, 48),
    ] {
        assert_eq!(scratch_bytes(&cfg, true, DType::Q8_0, 4096, k, n, 0), None);
    }
}

#[test]
fn narrow_projection_requires_reported_measured_device_and_bda() {
    use crate::caps::{DeviceArch, DeviceProbe, VENDOR_AMD};
    let p = DeviceProbe {
        vendor_id: VENDOR_AMD,
        device_id: 0x744c,
        driver_id_reported: true,
        driver_id: ash::vk::DriverId::AMD_PROPRIETARY,
        ..Default::default()
    };
    let arch = DeviceArch::AmdRdna3;
    assert!(device_eligible(&p, arch, true, true));
    assert!(!device_eligible(&p, arch, false, true));
    assert!(!device_eligible(&p, arch, true, false));
    assert!(!device_eligible(&p, DeviceArch::Other, true, true));
    for bad in [
        DeviceProbe {
            driver_id_reported: false,
            ..p
        },
        DeviceProbe {
            driver_id: ash::vk::DriverId::MESA_RADV,
            ..p
        },
        DeviceProbe {
            integrated: true,
            ..p
        },
        DeviceProbe {
            vendor_id: 0x10de,
            ..p
        },
        DeviceProbe {
            device_id: 0x7550,
            ..p
        },
    ] {
        assert!(!device_eligible(&bad, arch, true, true));
    }
}

fn backend(enabled: bool) -> VulkanBackend {
    backend_with_arena(enabled, true)
}

fn backend_with_arena(enabled: bool, arena: bool) -> VulkanBackend {
    let selected = std::env::var("MOE4ALL_TEST_DEVICE")
        .expect("explicit MOE4ALL_TEST_DEVICE=VulkanN required; selected device must be XTX");
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(selected);
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    cfg.kernels.vulkan.narrow_f32_gemv = enabled;
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(be.caps().name.contains("7900 XTX"));
    assert!(
        be.shared.narrow_f32_device,
        "requires qualified Windows AMD_PROPRIETARY Navi31"
    );
    assert_eq!(
        narrow_projection_scratch(&be, DType::Q8_0, 259, 2560, 48, 0),
        None
    );
    if arena {
        be.init_unified_vram(256 << 20).unwrap();
    }
    be
}

// Two independent Linears in one graph exercise scratch write/read/rewrite
// ordering. Streamed tests force one-slot eviction and change source bytes.
fn graph_case(be: &VulkanBackend, dt: DType, m: usize, k: usize, n: usize, streamed: bool) {
    const GUARD: usize = 64;
    const POISON: f32 = -777.25;
    let offset = k; // Nonzero, row-aligned weight offset in BOTH dtypes/routes.
    let x = (0..m * k)
        .map(|i| ((i * 17 % 127) as f32 - 63.0) / 128.0)
        .collect::<Vec<_>>();
    let raw: [Vec<u8>; 2] = std::array::from_fn(|j| {
        if dt == DType::Bf16 {
            (0..(n + 1) * k)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 29 + j * 13) % 113) as f32 / 128.0 - 0.4375)
                        .to_le_bytes()
                })
                .collect()
        } else {
            infr_testkit::synth_weight(dt, (n + 1) * k, 0x1460 + j as u32)
        }
    });
    let weights: [Box<dyn Buffer>; 2] = std::array::from_fn(|j| {
        let buf = be
            .alloc(
                if streamed { 4 } else { raw[j].len() },
                BufferUsage::Weights,
            )
            .unwrap();
        if !streamed {
            be.upload(buf.as_ref(), &raw[j]).unwrap();
        }
        buf
    });
    if streamed {
        // Q8_0 stride must be divisible by 34 and 4; BF16 is naturally aligned.
        let slot = raw[0].len().div_ceil(68) * 68;
        be.init_dense_pager(crate::pager::DensePagerLayout {
            pools: vec![crate::pager::DensePoolSpec {
                slot_bytes: slot,
                n_slots: 1,
                n_blocks: 2,
                host: None,
            }],
            ring_bytes: slot * 2,
        })
        .unwrap();
        for j in 0..2 {
            be.register_dense_stream(
                0,
                crate::pager::buffer_identity(weights[j].as_ref()),
                crate::pager::DenseSource {
                    bytes: crate::pager::DenseBytes::Mmap(vec![Arc::new(raw[j].clone())]),
                    block_id: j as u32,
                },
            )
            .unwrap();
        }
    }
    let mut g = Graph::new();
    let xi = g.input(TensorDesc::new(vec![m, k], DType::F32));
    let wi = std::array::from_fn::<_, 2, _>(|_| g.weight(TensorDesc::new(vec![n + 1, k], dt)));
    let yi = std::array::from_fn::<_, 2, _>(|_| g.output(TensorDesc::new(vec![m, n], DType::F32)));
    for j in 0..2 {
        g.push(Op::Linear {
            x: xi,
            weight: wi[j],
            dst: yi[j],
            m: m as u32,
            in_f: k as u32,
            out_f: n as u32,
            w_off: offset as u32,
        });
    }
    let admitted = narrow_projection_scratch(be, dt, m, k, n, offset);
    let requests = stages::compiled_workspace_requests(be, &g).unwrap();
    assert_eq!(
        requests
            .activation
            .iter()
            .find(|(tag, _)| *tag == "lin_w32")
            .map(|(_, b)| *b),
        admitted
    );
    let input = be
        .alloc((x.len() + GUARD) * 4, BufferUsage::Activations)
        .unwrap();
    let mut padded = x.clone();
    padded.extend(std::iter::repeat_n(POISON, GUARD));
    be.upload(input.as_ref(), bytemuck::cast_slice(&padded))
        .unwrap();
    let outputs: [Box<dyn Buffer>; 2] = std::array::from_fn(|_| {
        let out = be
            .alloc((m * n + GUARD) * 4, BufferUsage::Readback)
            .unwrap();
        be.upload(
            out.as_ref(),
            bytemuck::cast_slice(&vec![POISON; m * n + GUARD]),
        )
        .unwrap();
        out
    });
    let plan = be.compile(&g).unwrap();
    let mut bindings = Bindings::new();
    bindings.bind(xi, input.as_ref());
    for j in 0..2 {
        bindings.bind(wi[j], weights[j].as_ref());
        bindings.bind(yi[j], outputs[j].as_ref());
    }
    for _ in 0..2 {
        be.execute(plan.as_ref(), &bindings).unwrap();
    }
    for j in 0..2 {
        let decoded = infr_testkit::dequant_oracle(dt, &raw[j]);
        let expected = infr_testkit::ref_linear(&x, &decoded[offset..offset + k * n], m, k, n);
        let mut got = vec![0f32; m * n + GUARD];
        be.download(outputs[j].as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for (i, (&actual, &want)) in got.iter().zip(&expected).enumerate() {
            assert!(
                actual.is_finite() && (actual - want).abs() <= 3e-3 * want.abs().max(1.0),
                "{dt:?} streamed={streamed} M={m} K={k} N={n} [{j}][{i}]: {actual} vs {want}"
            );
        }
        assert!(got[m * n..].iter().all(|v| v.to_bits() == POISON.to_bits()));
        if !streamed {
            let mut unchanged = vec![0u8; raw[j].len()];
            be.download(weights[j].as_ref(), &mut unchanged).unwrap();
            assert_eq!(unchanged, raw[j]);
        }
    }
    let mut unchanged = vec![0u8; padded.len() * 4];
    be.download(input.as_ref(), &mut unchanged).unwrap();
    assert_eq!(unchanged, bytemuck::cast_slice::<f32, u8>(&padded));
    let expanded = be
        .built_kernel_names()
        .iter()
        .any(|name| name.starts_with("weight_f32_"));
    assert_eq!(expanded, admitted.is_some());
    if admitted.is_some() {
        assert!(be.built_kernel_names().contains(&"linear_f32r_mrow8_v4"));
    }
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE=VulkanN selecting XTX required; real resident/streamed graphs"]
fn narrow_projection_actual_graph_offsets_tails_guards() {
    for dt in [DType::Q8_0, DType::Bf16] {
        for (k, n, min_m) in [(2560, 48, 256), (10240, 4, 1024)] {
            for streamed in [false, true] {
                let be = backend(true);
                graph_case(&be, dt, min_m + 3, k, n, streamed);
            }
        }
    }
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE=VulkanN selecting XTX required; real fallback graphs"]
fn narrow_projection_actual_graph_fallbacks() {
    for dt in [DType::Q8_0, DType::Bf16] {
        for streamed in [false, true] {
            for (enabled, m) in [(true, 1), (true, 255), (false, 259)] {
                let be = backend(enabled);
                graph_case(&be, dt, m, 2560, 48, streamed);
            }
        }
    }
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE=VulkanN selecting XTX; standalone scratch fallback"]
fn narrow_projection_plain_scratch_keeps_native() {
    for dt in [DType::Q8_0, DType::Bf16] {
        for streamed in [false, true] {
            let be = backend_with_arena(true, false);
            graph_case(&be, dt, 259, 2560, 48, streamed);
        }
    }
}
