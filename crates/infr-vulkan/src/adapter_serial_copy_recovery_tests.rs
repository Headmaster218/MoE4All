//! Copy A, CPU-recover B into the entire cache, then demand the evicted A again.
use super::*;
use crate::pager::{ExpertSource, MoeHostChunkSpec, MoePagerLayout, MoePoolSpec, Role};
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::{DType, SizeSpec, TensorDesc};

const NE: usize = 32;
const EXP: usize = 4;
const SLOT_BYTES: usize = 64 * 1024;
const BANK_BYTES: usize = EXP * SLOT_BYTES;
const SLOTS: usize = 6;
const RUNTIME_BYTES: usize = 2 * 1024 * 1024;
const AUX_SLOT_BYTES: usize = 4 * 1024 * 1024;
const AUX_SLOTS: usize = 4;

fn fixture_pools() -> Vec<MoePoolSpec> {
    vec![
        MoePoolSpec {
            slot_bytes: SLOT_BYTES,
            n_slots: SLOTS,
            min_enabled_slots: 3,
        },
        // No source has this stride. Two surplus auxiliary cells place the final
        // 4 MiB cell above the routed pool's surplus, supplying real runtime storage.
        MoePoolSpec {
            slot_bytes: AUX_SLOT_BYTES,
            n_slots: AUX_SLOTS,
            min_enabled_slots: 1,
        },
    ]
}

fn assert_fixture_layout(layout: &crate::unified::ExpertArenaLayout) {
    assert_eq!(
        layout.total_bytes(),
        SLOTS * SLOT_BYTES + AUX_SLOTS * AUX_SLOT_BYTES
    );
    let routed = layout.slots(0).unwrap();
    assert_eq!(routed.len(), SLOTS);
    let runtime = layout.runtime_corridor();
    assert_eq!(runtime.len(), RUNTIME_BYTES);
    assert!(routed
        .iter()
        .all(|slot| slot.logical_offset + slot.len <= runtime.start));
    assert!(layout.slots(1).unwrap().iter().any(|slot| {
        slot.logical_offset <= runtime.start && slot.logical_offset + slot.len >= runtime.end
    }));
}

#[test]
fn serial_copy_redemand_layout_has_runtime_space_outside_six_expert_slots() {
    let specs: Vec<_> = fixture_pools()
        .iter()
        // MoePagerSession::new adds one physical exchange-floor cell to each dispatch floor.
        .map(|pool| (pool.slot_bytes, pool.n_slots, pool.min_enabled_slots + 1))
        .collect();
    for max_shard in [AUX_SLOT_BYTES, 2 * AUX_SLOT_BYTES, 3 * 1024 * 1024 * 1024] {
        let build = |specs: &[(usize, usize, usize)]| {
            crate::unified::ExpertArenaLayout::build(specs, max_shard, 0, 0, 0, RUNTIME_BYTES)
        };
        assert!(
            build(&specs[..1]).is_err(),
            "original undersized layout must fail"
        );
        assert_fixture_layout(&build(&specs).unwrap());
    }
}

struct ExpertIo;

impl BlockIo for ExpertIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        assert_eq!(desc.nbytes(), BANK_BYTES);
        assert!(desc.id < 6);
        dst[..BANK_BYTES].fill(0);
        for expert in 0..EXP {
            let value = 0.5 + desc.id as f32 / 8.0 + expert as f32 / 16.0;
            for diagonal in 0..NE {
                let offset = expert * SLOT_BYTES + (diagonal * NE + diagonal) * 4;
                dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(())
    }
}

struct Fixture {
    graph: Graph,
    outputs: Vec<TensorId>,
    buffers: Vec<(TensorId, Box<dyn Buffer>)>,
    roles: [[usize; 3]; 2],
    backend: VulkanBackend,
}

impl Fixture {
    fn new(resident: bool, early: bool, overlap: bool) -> Self {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU explicitly"));
        cfg.device.submit_dispatches = Some(0);
        cfg.device.decode_command_ahead = false;
        cfg.paging.expert_prefetch = false;
        cfg.paging.host_dma = true;
        cfg.paging.ring = Some(SizeSpec::Bytes((4 * SLOT_BYTES) as u64));
        cfg.kernels.vulkan.cpu_miss_threads = 0;
        cfg.kernels.vulkan.static_scratch_memo = true;
        cfg.kernels.vulkan.decode_gpu_resident = resident;
        cfg.kernels.vulkan.decode_gpu_resident_copy = Some(resident);
        cfg.kernels.vulkan.decode_gpu_resident_copy_overlap = overlap;
        cfg.kernels.vulkan.decode_gpu_resident_layers = 3;
        cfg.kernels.vulkan.decode_gpu_resident_submit_layers = 1;
        cfg.kernels.vulkan.decode_gpu_resident_inflight = 1;
        cfg.kernels.vulkan.decode_gpu_resident_hit_first = true;
        cfg.kernels.vulkan.decode_gpu_resident_fast_lut = true;
        cfg.kernels.vulkan.decode_gpu_resident_mapped_journal = true;
        cfg.kernels.vulkan.decode_gpu_resident_immutable_predicates = true;
        cfg.kernels.vulkan.decode_gpu_resident_indirect_predicates = false;
        cfg.kernels.vulkan.decode_gpu_resident_notice_thread = true;
        cfg.kernels.vulkan.decode_gpu_resident_early_recovery = early;
        let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        backend
            .init_moe_pager(MoePagerLayout {
                load_reserve_bytes: 0,
                n_blocks: 2 * EXP,
                pools: fixture_pools(),
                host_tier: None,
                host_store_io: Some(Arc::new(ExpertIo)),
                dynamic_state_reserve_bytes: 0,
                dynamic_state_max_allocation_bytes: 0,
                prefill_min_lane_bytes: 0,
                runtime_reserve_bytes: RUNTIME_BYTES as u64,
                host_chunks: vec![MoeHostChunkSpec {
                    base_offset: 0,
                    bytes: 6 * BANK_BYTES,
                }],
                prefill_target_lanes: 1,
                prefill_cache_bytes: (SLOTS * SLOT_BYTES) as u64,
            })
            .unwrap();
        assert_fixture_layout(backend.unified_vram().unwrap().expert_layout().unwrap());

        let mut graph = Graph::new();
        let input = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
        let router = graph.weight(TensorDesc::new(vec![EXP, NE], DType::F32));
        let mut buffers = Vec::new();
        let input_buf = backend.alloc(NE * 4, BufferUsage::Activations).unwrap();
        backend
            .upload(input_buf.as_ref(), bytemuck::cast_slice(&[1f32; NE]))
            .unwrap();
        buffers.push((input, input_buf));
        let router_buf = backend.alloc(EXP * NE * 4, BufferUsage::Weights).unwrap();
        let mut router_values = vec![0f32; EXP * NE];
        for expert in 0..EXP {
            router_values[expert * NE] = (EXP - expert) as f32;
        }
        backend
            .upload(router_buf.as_ref(), bytemuck::cast_slice(&router_values))
            .unwrap();
        buffers.push((router, router_buf));
        let mut handles = Vec::new();
        let mut role_ids = Vec::new();
        let bank: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(vec![0u8; BANK_BYTES]);
        for index in 0..6 {
            let id = graph.weight(TensorDesc::new(vec![EXP, NE, NE], DType::F32));
            let placeholder = backend.alloc(4, BufferUsage::Staging).unwrap();
            let role_id = crate::pager::buffer_identity(placeholder.as_ref());
            backend
                .register_paged_expert(
                    [Role::Gate, Role::Up, Role::Down][index % 3],
                    role_id,
                    ExpertSource {
                        bank: Arc::clone(&bank),
                        stride_bytes: SLOT_BYTES,
                        layer_base: (index / 3 * EXP) as u32,
                        host_offset: index * BANK_BYTES,
                        file: Some(BlockDesc {
                            id: index as u32,
                            extents: vec![BlockExtent {
                                offset: 0,
                                len: BANK_BYTES,
                            }],
                        }),
                    },
                    EXP,
                )
                .unwrap();
            handles.push(id);
            role_ids.push(role_id);
            buffers.push((id, placeholder));
        }
        backend.finish_session_allocations().unwrap();
        let mut outputs = Vec::new();
        for (bank, top) in [(0, 1), (1, 2), (0, 1)] {
            let dst = graph.output(TensorDesc::new(vec![1, NE], DType::F32));
            graph.push(Op::MoeFfn {
                x: input,
                router_x: input,
                router,
                gate_exps: handles[bank * 3],
                up_exps: handles[bank * 3 + 1],
                down_exps: handles[bank * 3 + 2],
                down_scale: None,
                fused_gate_up: false,
                dst,
                ne: NE as u32,
                n_expert: EXP as u32,
                n_used: top,
                n_ff_exp: NE as u32,
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
            outputs.push(dst);
            buffers.push((dst, backend.alloc(NE * 4, BufferUsage::Readback).unwrap()));
        }
        Self {
            graph,
            outputs,
            buffers,
            roles: [
                role_ids[..3].try_into().unwrap(),
                role_ids[3..].try_into().unwrap(),
            ],
            backend,
        }
    }

    fn bindings(&self) -> Bindings<'_> {
        let mut bindings = Bindings::new();
        for (id, buffer) in &self.buffers {
            bindings.bind(*id, buffer.as_ref());
        }
        bindings
    }

    fn outputs(&self) -> Vec<Vec<f32>> {
        self.outputs
            .iter()
            .map(|id| {
                let buffer = &self.buffers.iter().find(|(key, _)| key == id).unwrap().1;
                let mut values = vec![0f32; NE];
                self.backend
                    .download(buffer.as_ref(), bytemuck::cast_slice_mut(&mut values))
                    .unwrap();
                assert!(values.iter().all(|value| value.is_finite() && *value > 0.0));
                values
            })
            .collect()
    }
}

#[test]
#[ignore = "requires MOE4ALL_TEST_DEVICE selecting an idle GPU with conditional rendering and imported-host BDA"]
fn gpu_serial_copy_cpu_fallback_redemands_evicted_victim() {
    redemand_fixture(false);
}

#[test]
#[ignore = "requires MOE4ALL_TEST_DEVICE selecting an idle GPU with conditional rendering and imported-host BDA"]
fn gpu_overlap_copy_cpu_fallback_redemands_evicted_victim() {
    redemand_fixture(true);
}

fn redemand_fixture(overlap: bool) {
    let reference = {
        let fixture = Fixture::new(false, false, false);
        let plan = fixture.backend.compile(&fixture.graph).unwrap();
        fixture
            .backend
            .execute(plan.as_ref(), &fixture.bindings())
            .unwrap();
        fixture.outputs()
    };
    assert_eq!(reference[0], reference[2]);
    assert!((reference[0][0] - reference[1][0]).abs() > 0.01);

    for early in [false, true] {
        let fixture = Fixture::new(true, early, overlap);
        let plan = fixture.backend.compile(&fixture.graph).unwrap();
        // Warm fixed metadata/workspace before making the measured epoch completely cold.
        fixture
            .backend
            .execute(plan.as_ref(), &fixture.bindings())
            .unwrap();
        let copy = {
            let mut phase = fixture.backend.runtime_phase.lock().unwrap();
            let controller = ResidentChunk::new(
                &fixture.backend,
                &fixture.graph,
                true,
                &mut phase.resident_buffers,
            )
            .unwrap()
            .expect("resident execution required");
            assert_eq!(controller.early_recovery, early);
            controller
                .gpu_copy
                .clone()
                .expect("GPU copy required, not silent CPU fallback")
        };
        assert_eq!(
            copy.miss_predicates.is_some(),
            overlap,
            "requested overlap must be available"
        );
        {
            let mut pager = fixture.backend.moe_pager().lock().unwrap();
            let session = pager.as_mut().unwrap();
            for roles in fixture.roles {
                session.discard_cpu_miss(roles, &[0, 1, 2, 3]);
                assert_eq!(
                    session.routed_roles_resident_mask(&roles, &[0, 1]).unwrap(),
                    0
                );
            }
        }
        // Hide B only. Its first missing source fails validation before any tentative placement.
        // B's top-two triplets require all six physical slots, so its successful CPU FFN must
        // evict every A block installed by the first GPU copy. The third call demands A again.
        let mut sources = vec![0u32; copy.sources.len_bytes() / 4];
        fixture
            .backend
            .download(
                copy.sources.as_ref(),
                bytemuck::cast_slice_mut(&mut sources),
            )
            .unwrap();
        let mut hidden = sources.clone();
        for role in 0..3 {
            for expert in 0..EXP {
                let block = role * (2 * EXP) + EXP + expert;
                assert_ne!(&sources[block * 4..block * 4 + 2], &[0, 0]);
                hidden[block * 4..block * 4 + 2].fill(0);
            }
        }
        let ptr = crate::as_vk_buf(copy.sources.as_ref())
            .unwrap()
            .mapped_ptr()
            .unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(hidden.as_ptr().cast::<u8>(), ptr, hidden.len() * 4);
        }
        let before = crate::pager::completed_copies().0;
        fixture
            .backend
            .execute(plan.as_ref(), &fixture.bindings())
            .unwrap();
        let copies = crate::pager::completed_copies().0 - before;
        for (layer, (actual, expected)) in fixture.outputs().iter().zip(&reference).enumerate() {
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() <= 1e-6 * b.abs().max(1.0)),
                "early={early} layer={layer}: {actual:?} != {expected:?}"
            );
        }
        {
            let phase = fixture.backend.runtime_phase.lock().unwrap();
            assert!(
                Arc::ptr_eq(
                    &copy,
                    phase
                        .resident_buffers
                        .as_ref()
                        .unwrap()
                        .gpu_copy
                        .as_ref()
                        .unwrap()
                ),
                "the same fixed directory must survive recovery"
            );
        }
        // Capture host-derived LUT windows after all work drained; A must own three distinct
        // physical addresses, and no other resident global block may alias those slots.
        let (tape, windows) = {
            let mut pager = fixture.backend.moe_pager().lock().unwrap();
            let session = pager.as_mut().unwrap();
            assert_eq!(
                session
                    .routed_roles_resident_mask(&fixture.roles[0], &[0])
                    .unwrap(),
                1
            );
            let mut cursor = 0;
            let windows: Vec<_> = fixture
                .roles
                .iter()
                .flatten()
                .map(|&role| session.lut_window(&mut cursor, role, EXP).unwrap() as usize)
                .collect();
            (session.retained_tape(), windows)
        };
        let mut addresses = vec![0u64; 6 * EXP];
        fixture
            .backend
            .download(tape.as_ref(), bytemuck::cast_slice_mut(&mut addresses))
            .unwrap();
        let mut overlay = vec![0u32; copy.overlay.len_bytes() / 4];
        fixture
            .backend
            .download(
                copy.overlay.as_ref(),
                bytemuck::cast_slice_mut(&mut overlay),
            )
            .unwrap();
        let resident: Vec<_> = addresses.iter().copied().filter(|&a| a != 0).collect();
        assert_eq!(
            resident.len(),
            SLOTS,
            "runtime claims must leave all six routed expert slots available"
        );
        assert_eq!(
            resident.iter().copied().collect::<HashSet<_>>().len(),
            resident.len()
        );
        let mut final_gpu_roles = 0;
        for role in 0..3 {
            let address = addresses[windows[role]];
            let block = role * (2 * EXP);
            let gpu = overlay[block * 4] as u64 | ((overlay[block * 4 + 1] as u64) << 32);
            assert_ne!(address, 0);
            if gpu != 0 {
                final_gpu_roles += 1;
                assert_eq!(
                    address, gpu,
                    "early={early} role={role}: host LUT vs fresh GPU overlay"
                );
            }
        }
        // The resumed B reader can still protect the whole pool when the fresh A epoch
        // selects candidates. A may then recover on CPU instead of copying on GPU again.
        assert!(final_gpu_roles == 0 || final_gpu_roles == 3);
        assert_eq!(
            copies,
            1 + u64::from(final_gpu_roles == 3),
            "early={early}: first A must copy; final A must not reuse an old overlay"
        );
    }
}
