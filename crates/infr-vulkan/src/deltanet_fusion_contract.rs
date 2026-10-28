use super::*;
use infr_core::tensor::{DType, TensorDesc};

#[test]
fn deltanet_norm_v2b_embedded_push_abi_is_twenty_bytes() {
    // Reflect the actual embedded modules without constructing a Vulkan backend.
    for sigmoid in [false, true] {
        let (name, words) = crate::gemm::deltanet_fused_norm_kernel(sigmoid);
        assert_eq!(words.first(), Some(&0x0723_0203));
        let mut inst = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            assert!(count > 0 && at + count <= words.len(), "malformed {name}");
            inst.push((words[at] & 0xffff, &words[at..at + count]));
            at += count;
        }
        // OpVariable -> OpTypePointer(PushConstant=9) -> OpTypeStruct.
        let vars: Vec<_> = inst
            .iter()
            .filter(|(op, w)| *op == 59 && w.len() >= 4 && w[3] == 9)
            .collect();
        assert_eq!(vars.len(), 1, "{name}: expected one push block");
        let pointer = inst
            .iter()
            .find(|(op, w)| *op == 32 && w.len() == 4 && w[1] == vars[0].1[1])
            .expect("push pointer type");
        assert_eq!(pointer.1[2], 9);
        let structure = inst
            .iter()
            .find(|(op, w)| *op == 30 && w[1] == pointer.1[3])
            .expect("push struct type");
        assert_eq!(structure.1.len(), 7, "{name}: expected five scalar fields");
        for member in 0..5usize {
            assert!(
                inst.iter().any(|(op, w)| {
                    *op == 72
                        && w.len() == 5
                        && w[1] == structure.1[1]
                        && w[2] == member as u32
                        && w[3] == 35
                        && w[4] == (member * 4) as u32
                }),
                "{name}: wrong offset for push member {member}"
            );
            let scalar = structure.1[2 + member];
            assert!(
                inst.iter().any(|(op, w)| {
                    if member < 2 {
                        *op == 21 && w.len() == 4 && w[1] == scalar && w[2] == 32 && w[3] == 0
                    } else {
                        *op == 22 && w.len() == 3 && w[1] == scalar && w[2] == 32
                    }
                }),
                "{name}: wrong scalar type for push member {member}"
            );
        }
    }
    let kd = 128usize;
    assert_eq!((1.0f32 / (kd as f32).sqrt()).to_bits(), 0x3db5_04f3);
}

fn graph(nv: u32, act: Activation) -> Graph {
    let mut g = Graph::new();
    let qkv = g.input(TensorDesc::new(vec![(32 + nv as usize) * 128], DType::F32));
    let b = g.input(TensorDesc::new(vec![nv as usize], DType::F32));
    let a = g.input(TensorDesc::new(vec![nv as usize], DType::F32));
    let ac = g.weight(TensorDesc::new(vec![nv as usize], DType::F32));
    let dt = g.weight(TensorDesc::new(vec![nv as usize], DType::F32));
    let state = g.input(TensorDesc::new(vec![nv as usize, 128, 128], DType::F32));
    let out = g.internal(TensorDesc::new(vec![nv as usize * 128], DType::F32));
    let gate = g.input(TensorDesc::new(vec![nv as usize * 128], DType::F32));
    let w = g.weight(TensorDesc::new(vec![128], DType::F32));
    g.push(Op::DeltaNet {
        q: qkv,
        k: qkv,
        v: qkv,
        b,
        a,
        a_coef: ac,
        dt_bias: dt,
        state,
        state_trace: None,
        dst: out,
        rows: 1,
        n_vhead: nv,
        n_khead: 16,
        head_k: 128,
        head_v: 128,
        eps: 1e-6,
        src_stride: 0,
    });
    g.push(Op::GatedRmsNorm {
        x: out,
        weight: w,
        gate,
        dst: out,
        rows: 1,
        n_head: nv,
        head_dim: 128,
        eps: 1e-6,
        act,
    });
    g
}

#[test]
fn all_measured_head_ratios_and_both_gates_match() {
    for nv in [32, 48, 64] {
        for act in [Activation::Silu, Activation::Sigmoid] {
            let g = graph(nv, act);
            let matched = deltanet_norm_pair(&g, 0).expect("measured one-row contract");
            assert_eq!(matched.sigmoid, act == Activation::Sigmoid);
            assert_eq!(matched.eps, 1e-6);
            assert!(deltanet_norm_pair(&g, 1).is_none());
            assert!(deltanet_norm_pair(&g, usize::MAX).is_none());
        }
    }
}

#[test]
#[ignore = "requires the measured Vulkan GPU"]
fn recurrent_norm_fusion_keeps_speculative_runtime_on_original_path() {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.kernels.vulkan.delta_fused_norm = true;
    cfg.kernels.vulkan.delta_strided = true;
    cfg.spec.mtp = true;
    let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(backend.prefers_deltanet_fused_norm());
    assert!(!deltanet_norm_enabled(&backend));
}

#[test]
fn state_batch_dtype_alias_and_trace_fallbacks_are_preserved() {
    for case in 0..14 {
        let mut g = graph(48, Activation::Sigmoid);
        match case {
            0 => g.independent_rows = true,
            1 => g.mtp_verify = true,
            2 => {
                if let Op::DeltaNet { rows, .. } = &mut g.ops[0] {
                    *rows = 2;
                }
            }
            3 => {
                if let Op::DeltaNet {
                    state_trace, state, ..
                } = &mut g.ops[0]
                {
                    *state_trace = Some(*state);
                }
            }
            4 => {
                if let Op::DeltaNet { k, b, .. } = &mut g.ops[0] {
                    *k = *b;
                }
            }
            5 => {
                if let Op::DeltaNet { n_vhead, .. } = &mut g.ops[0] {
                    *n_vhead = 16;
                }
            }
            6 => {
                if let Op::DeltaNet { head_k, .. } = &mut g.ops[0] {
                    *head_k = 64;
                }
            }
            7 => {
                if let Op::GatedRmsNorm { rows, .. } = &mut g.ops[1] {
                    *rows = 2;
                }
            }
            8 => {
                if let Op::GatedRmsNorm { n_head, .. } = &mut g.ops[1] {
                    *n_head = 32;
                }
            }
            9 => {
                if let Op::GatedRmsNorm { act, .. } = &mut g.ops[1] {
                    *act = Activation::Gelu;
                }
            }
            10 => {
                if let Op::GatedRmsNorm { dst, gate, .. } = &mut g.ops[1] {
                    *dst = *gate;
                }
            }
            11 => {
                if let Op::GatedRmsNorm { weight, .. } = &g.ops[1] {
                    g.tensors[weight.0 as usize].kind = TensorKind::Internal;
                }
            }
            12 => {
                if let Op::DeltaNet { q, .. } = &g.ops[0] {
                    g.tensors[q.0 as usize].desc.dtype = DType::F16;
                }
            }
            13 => {
                if let Op::GatedRmsNorm { gate, dst, .. } = &mut g.ops[1] {
                    *gate = *dst;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            deltanet_norm_pair(&g, 0).is_none(),
            "unsafe pattern {case} fused"
        );
    }
}

#[test]
#[ignore = "parent-serialized RX7900XTX complete-graph parity"]
fn deltanet_norm_complete_graph_matches_split() {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.spec.mtp = false;
    cfg.kernels.vulkan.delta_strided = true;
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    cfg.kernels.vulkan.delta_fused_norm = false;
    let baseline = VulkanBackend::new_with(std::sync::Arc::new(cfg.clone())).unwrap();
    cfg.kernels.vulkan.delta_fused_norm = true;
    let candidate = VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap();
    assert!(!deltanet_norm_enabled(&baseline));
    assert!(deltanet_norm_enabled(&candidate));
    let props = unsafe {
        candidate
            .shared
            .instance
            .get_physical_device_properties(candidate.shared.physical_device)
    };
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy();
    assert!(name.contains("7900 XTX"), "wrong GPU: {name}");
    for nv in [32usize, 48, 64] {
        for act in [Activation::Silu, Activation::Sigmoid] {
            for scale in [0.0f32, 1e-4, 0.7, 4.0] {
                let mut g = graph(nv as u32, act);
                let (normal, state) = match g.ops[0] {
                    Op::DeltaNet { dst, state, .. } => (dst, state),
                    _ => unreachable!(),
                };
                let normal_n = nv * 128;
                let state_n = nv * 128 * 128;
                let output = g.output(TensorDesc::new(vec![normal_n + state_n], DType::F32));
                g.push(Op::Copy {
                    src: normal,
                    src_off: 0,
                    dst: output,
                    dst_off: 0,
                    n: normal_n as u32,
                });
                g.push(Op::Copy {
                    src: state,
                    src_off: 0,
                    dst: output,
                    dst_off: normal_n as u32,
                    n: state_n as u32,
                });
                let inputs: Vec<_> = g
                    .tensors
                    .iter()
                    .enumerate()
                    .filter_map(|(id, t)| {
                        if !matches!(t.kind, TensorKind::Input | TensorKind::Weight) {
                            return None;
                        }
                        let data: Vec<f32> = if id == 3 {
                            vec![-0.2; nv]
                        } else if id == 8 {
                            infr_testkit::gen_f32(128, 11)
                                .iter()
                                .map(|v| 1.0 + v * 0.3)
                                .collect()
                        } else {
                            infr_testkit::gen_f32(t.desc.numel(), id + 5)
                                .iter()
                                .map(|v| v * scale)
                                .collect()
                        };
                        Some((TensorId(id as u32), infr_testkit::f32_bytes(&data)))
                    })
                    .collect();
                let expected =
                    infr_testkit::run_graph(&baseline, &g, &inputs, output, normal_n + state_n);
                let actual =
                    infr_testkit::run_graph(&candidate, &g, &inputs, output, normal_n + state_n);
                assert!(expected.iter().chain(actual.iter()).all(|v| v.is_finite()));
                assert_eq!(
                    bytemuck::cast_slice::<f32, u8>(&expected[normal_n..]),
                    bytemuck::cast_slice::<f32, u8>(&actual[normal_n..]),
                    "persistent state changed nv={nv} act={act:?} scale={scale}"
                );
                for (i, (&got, &want)) in actual[..normal_n]
                    .iter()
                    .zip(&expected[..normal_n])
                    .enumerate()
                {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "norm bit drift nv={nv} act={act:?} scale={scale} i={i} got={got} want={want}"
                    );
                }
                let baseline_gate = if act == Activation::Sigmoid {
                    "rmsnorm_gate_sigmoid"
                } else {
                    "rmsnorm_gate"
                };
                let fused_gate = if act == Activation::Sigmoid {
                    "deltanet_fused_norm_sigmoid"
                } else {
                    "deltanet_fused_norm"
                };
                let split_names = baseline.built_kernel_names();
                let fused_names = candidate.built_kernel_names();
                assert!(split_names.contains(&"deltanet_strided"));
                assert!(split_names.contains(&baseline_gate));
                assert!(!split_names
                    .iter()
                    .any(|n| n.starts_with("deltanet_fused_norm")));
                assert!(fused_names.contains(&fused_gate));
                {
                    let kernels = candidate.shared.kernels.lock().unwrap();
                    let fused = kernels.get(fused_gate).expect("fused gate pipeline");
                    assert_eq!(fused.push_size, 20, "v2b runtime push size");
                    assert_eq!(fused.n_buf, 9, "v2b descriptor count");
                }
                assert!(!fused_names.contains(&"rmsnorm_gate"));
                assert!(!fused_names.contains(&"rmsnorm_gate_sigmoid"));
                eprintln!("G01v2b graph nv={nv} act={act:?} scale={scale} state_byte_exact=true norm_byte_exact=true push_bytes=20 bindings=9 split_and_fused_routes_verified=true");
            }
        }
    }
}
