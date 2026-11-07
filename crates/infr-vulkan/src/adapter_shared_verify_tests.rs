//! Actual-graph qualification, not the already-fused shared NR1/NR8 module comparison.
use std::sync::atomic::AtomicU64;

pub(super) static ADMISSIONS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod fixtures {
    use super::super::*;
    use crate::pager::{ExpertSource, MoeHostChunkSpec, MoePagerLayout, MoePoolSpec, Role};
    use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
    use infr_core::{DType, TensorDesc};

    const ROWS: usize = 4;
    const K: usize = 2560;
    const N: usize = 640;
    const EXP: usize = 512;
    const USED: usize = 10;
    const GU_BYTES: usize = K * N / 256 * 82;
    const DOWN_BYTES: usize = K * N / 32 * 18;
    const GU_SLOTS: usize = 96;
    const DOWN_SLOTS: usize = 51;
    const POISON: u32 = 0x7fc0_1234;
    const CANARY: u32 = 0xc5f3_5555;

    struct SharedGraph {
        graph: Graph,
        x: TensorId,
        routing: TensorId,
        router: TensorId,
        roles: [TensorId; 3],
        scalar: TensorId,
        shared: [TensorId; 3],
        out: TensorId,
    }

    impl SharedGraph {
        fn new(rows: usize) -> Self {
            let mut graph = Graph::new();
            graph.mtp_verify = true;
            let f32_desc = |n| TensorDesc::new(vec![n], DType::F32);
            let x = graph.input(f32_desc(rows * K));
            let routing = graph.input(f32_desc(rows * K));
            let router = graph.weight(f32_desc(EXP * K));
            let roles = [DType::Iq2S, DType::Iq2S, DType::Iq4Nl]
                .map(|dtype| graph.weight(TensorDesc::new(vec![EXP * K * N], dtype)));
            let scalar = graph.weight(f32_desc(K));
            let shared =
                std::array::from_fn(|_| graph.weight(TensorDesc::new(vec![K * N], DType::Q8_0)));
            let moe = graph.internal(f32_desc(rows * K));
            let gate = graph.internal(f32_desc(rows));
            let gbuf = graph.internal(f32_desc(rows * N));
            let ubuf = graph.internal(f32_desc(rows * N));
            let act = graph.internal(f32_desc(rows * N));
            let down = graph.internal(f32_desc(rows * K));
            let out = graph.output(f32_desc(rows * K));
            graph.push(Op::MoeFfn {
                x,
                router_x: routing,
                router,
                gate_exps: roles[0],
                up_exps: roles[1],
                down_exps: roles[2],
                dst: moe,
                ne: K as u32,
                n_expert: EXP as u32,
                n_used: USED as u32,
                n_ff_exp: N as u32,
                down_scale: None,
                fused_gate_up: false,
                scale: 1.0,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            graph.push(Op::Linear {
                x,
                weight: scalar,
                dst: gate,
                m: rows as u32,
                in_f: K as u32,
                out_f: 1,
                w_off: 0,
            });
            for (dst, weight) in [(gbuf, shared[0]), (ubuf, shared[1])] {
                graph.push(Op::Linear {
                    x,
                    weight,
                    dst,
                    m: rows as u32,
                    in_f: K as u32,
                    out_f: N as u32,
                    w_off: 0,
                });
            }
            graph.push(Op::GatedAct {
                gate: gbuf,
                up: ubuf,
                dst: act,
                rows: rows as u32,
                nff: N as u32,
                act: Activation::Silu,
                up_off: 0,
                up_stride: 0,
                gate_stride: 0,
                gate_block_width: 0,
                swiglu_clamp: None,
            });
            graph.push(Op::Linear {
                x: act,
                weight: shared[2],
                dst: down,
                m: rows as u32,
                in_f: N as u32,
                out_f: K as u32,
                w_off: 0,
            });
            graph.push(Op::MoeSharedExpertAdd {
                moe,
                shexp: down,
                gate,
                dst: out,
                rows: rows as u32,
                n: K as u32,
            });
            Self {
                graph,
                x,
                routing,
                router,
                roles,
                scalar,
                shared,
                out,
            }
        }
    }

    #[test]
    fn shared_verify_policy_exact_shape_and_legacy_opt_out() {
        let f = SharedGraph::new(ROWS);
        assert!(paged_moe_shared_at(&f.graph, 0).is_none());
        assert!(paged_moe_shared_at_with_policy(&f.graph, 0, false).is_none());
        let expected = paged_moe_shared_at_with_policy(&f.graph, 0, true).unwrap();
        assert_eq!(
            paged_moe_shared_candidates_with_policy(&f.graph, false, true).collect::<Vec<_>>(),
            vec![(0, expected)],
        );
        assert_eq!(
            paged_moe_shared_candidates_with_policy(&f.graph, true, true).count(),
            0
        );
        // Ordinary M1 retains the old matcher, independently of this opt-in.
        let scalar = SharedGraph::new(1);
        assert_eq!(
            paged_moe_shared_at(&scalar.graph, 0),
            paged_moe_shared_at_with_policy(&scalar.graph, 0, true)
        );
        assert!(paged_moe_shared_at(&scalar.graph, 0).is_some());
        for rows in [2, 3, 8] {
            assert!(
                paged_moe_shared_at_with_policy(&SharedGraph::new(rows).graph, 0, true).is_none()
            );
        }
    }

    #[test]
    fn shared_verify_policy_rejects_near_misses_without_graph_mutation() {
        let f = SharedGraph::new(ROWS);
        let reject = |mutate: &dyn Fn(&mut Graph)| {
            let mut graph = f.graph.clone();
            mutate(&mut graph);
            assert!(paged_moe_shared_at_with_policy(&graph, 0, true).is_none());
        };
        reject(&|g| g.mtp_verify = false);
        reject(&|g| g.independent_rows = true);
        reject(&|g| {
            if let Op::MoeFfn { n_expert, .. } = &mut g.ops[0] {
                *n_expert = 511
            }
        });
        reject(&|g| {
            if let Op::MoeFfn { n_used, .. } = &mut g.ops[0] {
                *n_used = 9
            }
        });
        reject(&|g| {
            if let Op::MoeFfn { ne, .. } = &mut g.ops[0] {
                *ne = 2048
            }
        });
        reject(&|g| {
            if let Op::MoeFfn { n_ff_exp, .. } = &mut g.ops[0] {
                *n_ff_exp = 512
            }
        });
        reject(&|g| {
            if let Op::MoeFfn { fused_gate_up, .. } = &mut g.ops[0] {
                *fused_gate_up = true
            }
        });
        reject(&|g| {
            if let Op::MoeFfn { swiglu_clamp, .. } = &mut g.ops[0] {
                *swiglu_clamp = Some(7.0)
            }
        });
        for (id, dtype) in [
            (f.roles[0], DType::Iq3S),
            (f.roles[1], DType::Iq3S),
            (f.roles[2], DType::Iq2S),
            (f.shared[0], DType::Q5K),
            (f.shared[1], DType::Q5K),
            (f.shared[2], DType::Q5K),
            (f.scalar, DType::Q8_0),
        ] {
            reject(&|g| g.tensors[id.0 as usize].desc.dtype = dtype);
        }
        reject(&|g| {
            g.ops.pop();
        });
        reject(&|g| {
            if let Op::Linear { w_off, .. } = &mut g.ops[2] {
                *w_off = 1
            }
        });
        reject(&|g| {
            if let Op::MoeSharedExpertAdd { shexp, .. } = &mut g.ops[6] {
                *shexp = f.x
            }
        });
        // Failed matches cannot poison a later valid scan or change the graph's tail.
        assert_eq!(f.graph.ops.len(), 7);
        assert_eq!(
            paged_moe_shared_candidates_with_policy(&f.graph, false, true).count(),
            1
        );
    }

    fn next(seed: &mut u32) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *seed
    }

    fn quant_data(len: usize, block: usize, mut seed: u32) -> Vec<u8> {
        assert_eq!(len % block, 0);
        let mut data = vec![0u8; len];
        for bytes in data.chunks_exact_mut(block) {
            // Finite, nonzero GGUF f16 scale; all payload bits are valid IQ/Q8 codes.
            bytes[..2].copy_from_slice(&0x1000u16.to_le_bytes());
            for byte in &mut bytes[2..] {
                *byte = next(&mut seed) as u8;
            }
        }
        data
    }

    struct QuantIo;
    impl BlockIo for QuantIo {
        fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
            let (stride, block) = if desc.id == 2 {
                (DOWN_BYTES, 18)
            } else {
                (GU_BYTES, 82)
            };
            assert_eq!(desc.nbytes(), stride * EXP);
            assert!(dst.len() >= desc.nbytes());
            for (expert, bytes) in dst[..desc.nbytes()].chunks_exact_mut(stride).enumerate() {
                // Repeated valid quant-block tiles keep initialization bounded. The tile period
                // differs from matrix row stride, and every role/expert has distinct payloads.
                let seed = 0x735a_19cdu32
                    .wrapping_add(desc.id * 0x10007)
                    .wrapping_add(expert as u32 * 0x103);
                let tile = quant_data(1024 * block, block, seed);
                for chunk in bytes.chunks_mut(tile.len()) {
                    chunk.copy_from_slice(&tile[..chunk.len()]);
                }
            }
            Ok(())
        }
    }

    fn upload(be: &VulkanBackend, values: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
        let buffer = be.alloc(values.len(), usage).unwrap();
        be.upload(buffer.as_ref(), values).unwrap();
        buffer
    }

    fn initialize_pager(be: &VulkanBackend, placeholders: &[Box<dyn Buffer>; 3]) -> [usize; 3] {
        let sizes = [GU_BYTES, GU_BYTES, DOWN_BYTES];
        let mut offset = 0;
        let chunks = sizes.map(|bytes| {
            let chunk = MoeHostChunkSpec {
                base_offset: offset,
                bytes: bytes * EXP,
            };
            offset += chunk.bytes;
            chunk
        });
        be.init_moe_pager(MoePagerLayout {
            load_reserve_bytes: 0,
            n_blocks: EXP,
            pools: vec![
                MoePoolSpec {
                    slot_bytes: GU_BYTES,
                    n_slots: GU_SLOTS,
                    min_enabled_slots: 80,
                },
                MoePoolSpec {
                    slot_bytes: DOWN_BYTES,
                    n_slots: DOWN_SLOTS,
                    min_enabled_slots: 40,
                },
            ],
            host_tier: None,
            host_store_io: Some(Arc::new(QuantIo)),
            dynamic_state_reserve_bytes: 0,
            dynamic_state_max_allocation_bytes: 0,
            prefill_min_lane_bytes: 0,
            runtime_reserve_bytes: 16 * 1024 * 1024,
            host_chunks: chunks
                .iter()
                .map(|c| MoeHostChunkSpec {
                    base_offset: c.base_offset,
                    bytes: c.bytes,
                })
                .collect(),
            prefill_target_lanes: 1,
            prefill_cache_bytes: (GU_BYTES * GU_SLOTS + DOWN_BYTES * DOWN_SLOTS) as u64,
        })
        .unwrap();
        let identities =
            std::array::from_fn(|i| crate::pager::buffer_identity(placeholders[i].as_ref()));
        for (i, role) in [Role::Gate, Role::Up, Role::Down].into_iter().enumerate() {
            // Registration validates bank extent even when BlockIo fills the permanent host store.
            // Drop this validation bank after each registration instead of retaining three copies.
            be.register_paged_expert(
                role,
                identities[i],
                ExpertSource {
                    bank: Arc::new(vec![0u8; sizes[i] * EXP]),
                    stride_bytes: sizes[i],
                    layer_base: 0,
                    host_offset: chunks[i].base_offset,
                    file: Some(BlockDesc {
                        id: i as u32,
                        extents: vec![BlockExtent {
                            offset: 0,
                            len: sizes[i] * EXP,
                        }],
                    }),
                },
                EXP,
            )
            .unwrap();
        }
        identities
    }

    fn route_ids(pairs: usize) -> Vec<u32> {
        assert!(matches!(pairs, 6 | 10));
        let mut ids: Vec<_> = (0..ROWS * USED).map(|i| i as u32).collect();
        for first in [0, 2] {
            for slot in 0..pairs / 2 {
                ids[(first + 1) * USED + slot] = ids[first * USED + (slot + 3) % USED];
            }
        }
        ids
    }

    fn router_values(ids: &[u32]) -> Vec<f32> {
        let mut values = vec![0f32; EXP * K];
        for row in 0..ROWS {
            for slot in 0..USED {
                values[ids[row * USED + slot] as usize * K + row] = (USED - slot) as f32;
            }
        }
        values
    }

    fn run_graph(enabled: bool, resident: bool, device: usize) -> Vec<Vec<f32>> {
        let mut cfg = infr_core::config::Config::default();
        cfg.spec.mtp = true;
        cfg.paging.expert_prefetch = false;
        cfg.kernels.vulkan.verify_shared_iq2_nr8 = enabled;
        cfg.kernels.vulkan.gemv.id_grid_nr = 8;
        cfg.kernels.vulkan.gemv.id_grid_buffer = Some(false);
        cfg.kernels.vulkan.decode_gpu_resident = resident;
        cfg.kernels.vulkan.decode_gpu_resident_copy = Some(false);
        cfg.kernels.vulkan.decode_gpu_resident_copy_overlap = false;
        let be = VulkanBackend::new_on_with(device, Arc::new(cfg)).unwrap();
        assert_eq!(
            crate::recorder::shared_iq2_verify_fusion_allowed(&be),
            enabled
        );
        let f = SharedGraph::new(ROWS);
        let placeholders = std::array::from_fn(|_| be.alloc(4, BufferUsage::Staging).unwrap());
        let roles = initialize_pager(&be, &placeholders);
        let mut seed = 0x418c_057bu32;
        let input: Vec<_> = (0..ROWS * K)
            .map(|_| (next(&mut seed) as i32 as f32 / i32::MAX as f32) * 0.125)
            .collect();
        let x = upload(&be, bytemuck::cast_slice(&input), BufferUsage::Activations);
        let mut routing = vec![0f32; ROWS * K];
        for row in 0..ROWS {
            routing[row * K + row] = 1.0;
        }
        let routing = upload(
            &be,
            bytemuck::cast_slice(&routing),
            BufferUsage::Activations,
        );
        let router = be.alloc(EXP * K * 4, BufferUsage::Weights).unwrap();
        let scalar: Vec<_> = (0..K).map(|i| (i % 17) as f32 / 32.0 - 0.25).collect();
        let scalar = upload(&be, bytemuck::cast_slice(&scalar), BufferUsage::Weights);
        let shared: [Box<dyn Buffer>; 3] = std::array::from_fn(|role| {
            upload(
                &be,
                &quant_data(K * N / 32 * 34, 34, 0x17ca_439b + role as u32),
                BufferUsage::Weights,
            )
        });
        let out = be
            .alloc((ROWS * K + 16) * 4, BufferUsage::Readback)
            .unwrap();
        let mut bindings = Bindings::new();
        bindings.bind(f.x, x.as_ref());
        bindings.bind(f.routing, routing.as_ref());
        bindings.bind(f.router, router.as_ref());
        bindings.bind(f.scalar, scalar.as_ref());
        bindings.bind(f.out, out.as_ref());
        for (id, buffer) in f.roles.iter().zip(&placeholders) {
            bindings.bind(*id, buffer.as_ref());
        }
        for (id, buffer) in f.shared.iter().zip(&shared) {
            bindings.bind(*id, buffer.as_ref());
        }
        be.finish_session_allocations().unwrap();
        let plan = be.compile(&f.graph).unwrap();
        let mut outputs = Vec::new();
        for (case, pairs) in [6, 10, 6, 10].into_iter().enumerate() {
            let ids = route_ids(pairs);
            be.upload(router.as_ref(), bytemuck::cast_slice(&router_values(&ids)))
                .unwrap();
            // Start cold, repeat hot, and force a coherent one-ID miss across all three roles.
            if case == 2 {
                let mut guard = be.moe_pager().lock().unwrap();
                let pager = guard.as_mut().unwrap();
                assert_eq!(
                    pager.routed_roles_resident_mask(&roles, &[ids[0]]).unwrap(),
                    1
                );
                pager.discard_cpu_miss(roles, &[ids[0]]);
                assert_eq!(
                    pager.routed_roles_resident_mask(&roles, &[ids[0]]).unwrap(),
                    0
                );
            }
            let mut words = vec![POISON; ROWS * K + 16];
            words[ROWS * K..].fill(CANARY);
            be.upload(out.as_ref(), bytemuck::cast_slice(&words))
                .unwrap();
            let before = super::ADMISSIONS.load(Ordering::Relaxed);
            be.execute(plan.as_ref(), &bindings).unwrap();
            be.download(out.as_ref(), bytemuck::cast_slice_mut(&mut words))
                .unwrap();
            let admitted = super::ADMISSIONS.load(Ordering::Relaxed) - before;
            assert_eq!(
                admitted > 0,
                enabled,
                "actual execute scan must adopt the requested path"
            );
            let names = be.built_kernel_names();
            assert_eq!(
                names.contains(&"native_idm_iq2s_nr8_paged_shexp"),
                enabled,
                "candidate must build shared NR8; the nonfused reference must not: {names:?}"
            );
            if resident {
                assert!(
                    be.runtime_phase.lock().unwrap().resident_buffers.is_some(),
                    "resident leg must exercise the real controller"
                );
            }
            let guard = be.moe_pager().lock().unwrap();
            let pager = guard.as_ref().unwrap();
            for row in ids.chunks_exact(USED) {
                assert_eq!(
                    pager.routed_roles_resident_mask(&roles, row).unwrap(),
                    (1 << USED) - 1,
                    "every routed role must be resident after recovery"
                );
            }
            drop(guard);
            assert!(words[ROWS * K..].iter().all(|&v| v == CANARY));
            let values: Vec<_> = words[..ROWS * K]
                .iter()
                .map(|&v| f32::from_bits(v))
                .collect();
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().any(|v| v.abs() > 1e-7), "vacuous full FFN");
            assert!(values
                .chunks_exact(K)
                .zip(input.chunks_exact(K))
                .any(|(a, b)| a != b));
            outputs.push(values);
        }
        outputs
    }

    #[test]
    #[ignore = "explicit GPU; full paged IQ banks, about 2 GiB peak host RAM; run serially"]
    fn shared_verify_actual_graph_flag_ab_numeric_and_adoption() {
        let selected = std::env::var("MOE4ALL_TEST_DEVICE")
            .expect("set MOE4ALL_TEST_DEVICE=VulkanN explicitly; no fallback device");
        let device: usize = selected
            .strip_prefix("Vulkan")
            .unwrap_or(&selected)
            .parse()
            .expect("MOE4ALL_TEST_DEVICE must be VulkanN or a numeric index");
        for resident in [false, true] {
            let baseline = run_graph(false, resident, device);
            let candidate = run_graph(true, resident, device);
            let mut max_abs = 0f32;
            let mut changed = 0usize;
            for (case, (reference, actual)) in baseline.iter().zip(&candidate).enumerate() {
                for (index, (&a, &b)) in reference.iter().zip(actual).enumerate() {
                    let error = (a - b).abs();
                    max_abs = max_abs.max(error);
                    changed += usize::from(a.to_bits() != b.to_bits());
                    // Dense mrow Q8 and the shared-slot Q8 use different sum grouping. This is
                    // the existing full-FFN tolerance, NOT a bit-exact or token-level claim.
                    assert!(
                        error <= 1e-5f32.max(a.abs() * 1e-5),
                        "resident={resident} case={case} element={index}: {a} != {b}"
                    );
                }
            }
            eprintln!("shared Verify ACTUAL graph: resident={resident} max_abs={max_abs:e} changed_bits={changed}; runtime adoption checked, E2E still required");
        }
    }
}
