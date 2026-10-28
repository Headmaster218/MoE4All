//! Production graph differential: real payload copies/FFNs, not fake-BDA prepare fixtures.
use super::*;
use crate::recorder::owned_topk;

const EXPERTS: usize = 16;
const BANK_BYTES_16: usize = EXPERTS * SLOT_BYTES;

struct TopkExpertIo;
impl BlockIo for TopkExpertIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        if desc.nbytes() != BANK_BYTES_16 || dst.len() < BANK_BYTES_16 || desc.id >= 6 {
            return Err(be("invalid TopK graph fixture bank"));
        }
        dst[..BANK_BYTES_16].fill(0);
        for expert in 0..EXPERTS {
            let scale = 0.125 + desc.id as f32 / 64.0 + expert as f32 / 32.0;
            for diagonal in 0..NE {
                let offset = expert * SLOT_BYTES + (diagonal * NE + diagonal) * 4;
                dst[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
            }
        }
        Ok(())
    }
}

fn fixture(used: usize, overlap: bool) -> Fixture {
    assert!(matches!(used, 8 | 10));
    let (submit_layers, producer, copy) = (1, true, Some(true));
    let (kv_reserve_bytes, kv_segment_bytes) = (0usize, 0usize);
    let mut cfg = infr_core::config::Config::default();
    cfg.device.submit_dispatches = Some(0);
    cfg.device.decode_command_ahead = true;
    cfg.device.decode_command_early_submit = true;
    cfg.device.decode_command_resident = true;
    cfg.paging.expert_prefetch = false;
    cfg.paging.host_dma = true;
    cfg.paging.ring = Some(SizeSpec::Bytes((4 * SLOT_BYTES) as u64));
    cfg.kernels.vulkan.cpu_miss_threads = 0;
    cfg.kernels.vulkan.static_scratch_memo = true;
    cfg.kernels.vulkan.decode_gpu_resident = true;
    cfg.kernels.vulkan.decode_gpu_resident_copy = copy;
    cfg.kernels.vulkan.decode_gpu_resident_copy_overlap = overlap;
    cfg.kernels.vulkan.decode_gpu_resident_layers = 3;
    cfg.kernels.vulkan.decode_gpu_resident_submit_layers = submit_layers;
    cfg.kernels.vulkan.decode_gpu_resident_inflight = 1;
    cfg.kernels.vulkan.decode_gpu_resident_hit_first = true;
    cfg.kernels.vulkan.decode_gpu_resident_fast_lut = true;
    cfg.kernels.vulkan.decode_gpu_resident_immutable_predicates = true;
    cfg.kernels.vulkan.decode_gpu_resident_indirect_predicates = false;
    cfg.kernels.vulkan.decode_gpu_resident_notice_thread = producer;
    cfg.kernels.vulkan.decode_gpu_resident_early_recovery = producer;
    cfg.kernels.vulkan.decode_gpu_resident_waiter = producer;
    cfg.prof.ops = false;
    cfg.prof.stages = false;
    cfg.prof.timeline_path = None;

    if let Ok(device) = std::env::var("MOE4ALL_TEST_DEVICE") {
        cfg.device.dev = Some(device);
    }
    let backend = VulkanBackend::new_with(Arc::new(cfg)).expect("Vulkan fixture backend");
    assert!(backend.shared.conditional_rendering.is_some());
    backend
        .init_moe_pager(MoePagerLayout {
            load_reserve_bytes: 0,
            n_blocks: LAYERS * EXPERTS,
            pools: vec![MoePoolSpec {
                slot_bytes: SLOT_BYTES,
                n_slots: 128,
                min_enabled_slots: LAYERS * EXPERTS * 3,
            }],
            host_tier: None,
            host_store_io: Some(Arc::new(TopkExpertIo)),
            dynamic_state_reserve_bytes: kv_reserve_bytes as u64,
            dynamic_state_max_allocation_bytes: kv_segment_bytes as u64,
            prefill_min_lane_bytes: 0,
            runtime_reserve_bytes: 1024 * 1024,
            host_chunks: vec![MoeHostChunkSpec {
                base_offset: 0,
                bytes: LAYERS * 3 * BANK_BYTES_16,
            }],
            prefill_target_lanes: 1,
            prefill_cache_bytes: (128 * SLOT_BYTES) as u64,
        })
        .expect("small padded expert pool");
    assert!(
        command_prefix::enabled(&backend),
        "preparation must be active"
    );

    let mut graph = Graph::new();
    let frontend = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
    let one = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
    let input = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
    let routing = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
    let router = graph.weight(TensorDesc::new(vec![EXPERTS, NE], DType::F32));
    let mut role_handles = Vec::new();
    let mut outputs = Vec::new();
    let mut x = input;
    for _ in 0..LAYERS {
        graph.push(Op::Add {
            a: frontend,
            b: one,
            dst: frontend,
            n: NE as u32,
        });
        let gate_exps = graph.weight(TensorDesc::new(vec![EXPERTS, FF, NE], DType::F32));
        let up_exps = graph.weight(TensorDesc::new(vec![EXPERTS, FF, NE], DType::F32));
        let down_exps = graph.weight(TensorDesc::new(vec![EXPERTS, NE, FF], DType::F32));
        role_handles.extend([gate_exps, up_exps, down_exps]);
        let dst = graph.output(TensorDesc::new(vec![1, NE], DType::F32));
        graph.push(Op::MoeFfn {
            x,
            router_x: routing,
            router,
            gate_exps,
            up_exps,
            down_exps,
            down_scale: None,
            fused_gate_up: false,
            dst,
            ne: NE as u32,
            n_expert: EXPERTS as u32,
            n_used: used as u32,
            n_ff_exp: FF as u32,
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
        x = dst;
    }
    let mut buffers = Vec::new();
    for (id, values, usage) in [
        (frontend, vec![0.0f32; NE], BufferUsage::Staging),
        (one, vec![1.0f32; NE], BufferUsage::Activations),
        (input, vec![1.0f32; NE], BufferUsage::Activations),
        (routing, vec![1.0f32; NE], BufferUsage::Activations),
    ] {
        let buffer = backend.alloc(NE * 4, usage).unwrap();
        backend
            .upload(buffer.as_ref(), bytemuck::cast_slice(&values))
            .unwrap();
        buffers.push((id, buffer));
    }
    let router_buffer = backend
        .alloc(EXPERTS * NE * 4, BufferUsage::Weights)
        .unwrap();
    let mut router_values = vec![0.0f32; EXPERTS * NE];
    for expert in 0..EXPERTS {
        router_values[expert * NE] = (EXPERTS - expert) as f32;
    }
    backend
        .upload(router_buffer.as_ref(), bytemuck::cast_slice(&router_values))
        .unwrap();
    buffers.push((router, router_buffer));
    let bank = Arc::new(vec![0u8; BANK_BYTES_16]);
    let mut role_ids = Vec::new();
    for (index, id) in role_handles.into_iter().enumerate() {
        let placeholder = backend.alloc(4, BufferUsage::Staging).unwrap();
        let role_id = crate::pager::buffer_identity(placeholder.as_ref());
        backend
            .register_paged_expert(
                [Role::Gate, Role::Up, Role::Down][index % 3],
                role_id,
                ExpertSource {
                    bank: bank.clone(),
                    stride_bytes: SLOT_BYTES,
                    layer_base: (index / 3 * EXPERTS) as u32,
                    host_offset: index * BANK_BYTES_16,
                    file: Some(BlockDesc {
                        id: index as u32,
                        extents: vec![BlockExtent {
                            offset: 0,
                            len: BANK_BYTES_16,
                        }],
                    }),
                },
                EXPERTS,
            )
            .unwrap();
        buffers.push((id, placeholder));
        role_ids.push(role_id);
    }
    backend.finish_session_allocations().unwrap();
    for &id in &outputs {
        buffers.push((id, backend.alloc(NE * 4, BufferUsage::Readback).unwrap()));
    }
    Fixture {
        graph,
        frontend,
        outputs: outputs.try_into().unwrap(),
        buffers,
        roles: [
            role_ids[..3].try_into().unwrap(),
            role_ids[3..].try_into().unwrap(),
        ],
        backend,
    }
}

fn preload(f: &Fixture, ids: &[u32]) {
    let mut guard = f.backend.moe_pager().lock().unwrap();
    let session = guard.as_mut().unwrap();
    session.handoff_gpu_decode(&f.backend).unwrap();
    for roles in f.roles {
        session.begin_role_batches(&roles).unwrap();
        session
            .push_roles_cpu(&f.backend, &roles.map(|r| (r, ids)), false)
            .unwrap()
            .complete_without_recorder(&f.backend)
            .unwrap();
    }
}

fn cpu_miss_last_layer(f: &Fixture, missing: u32) {
    let (sources, seed) = {
        let mut guard = f.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        let previous = session.gpu_decode_owner.clone().expect("warm owned graph");
        session.handoff_gpu_decode(&f.backend).unwrap();
        let bases = previous.role_params(session, &f.roles[LAYERS - 1]).unwrap();
        session.discard_cpu_miss(f.roles[LAYERS - 1], &[missing]);
        session.gpu_decode_owner = None;
        let mut sources = session.gpu_copy_sources().unwrap().unwrap();
        for base in &bases[..3] {
            sources.test_cpu_only_source(*base as usize + missing as usize);
        }
        (sources, crate::pager::owned_seed(session, 0).unwrap())
    };
    let owner = crate::pager::GpuCopyBuffers::new_owned(&f.backend, sources, LAYERS, seed)
        .unwrap()
        .unwrap();
    assert!(!owner.owned_self_service());
    f.backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .gpu_decode_owner = Some(Arc::clone(&owner));
    f.backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_mut()
        .unwrap()
        .gpu_copy = Some(owner);
}

#[derive(Debug, PartialEq)]
struct ResultState {
    outputs: Vec<Vec<u32>>,
    directory: Vec<u32>,
    overlay: Vec<u32>,
    nodes: Vec<u32>,
    blocks: Vec<u32>,
}

fn normalize_addresses(nodes: &mut [u32], blocks: &mut [u32]) {
    let first = nodes[7] as usize;
    let slots = nodes[2] as usize;
    for slot in 0..slots {
        nodes[(first + slot) * 4..(first + slot) * 4 + 2].fill(0);
    }
    for record in blocks.chunks_exact_mut(4) {
        record[..2].fill(0);
    }
}

fn run(used: usize, enabled: bool, cpu_miss: bool) -> ResultState {
    run_mode(used, enabled, cpu_miss, false, false)
}

fn run_mode(
    used: usize,
    enabled: bool,
    cpu_miss: bool,
    overlap: bool,
    prepared: bool,
) -> ResultState {
    assert!(
        !prepared || !cpu_miss,
        "prepared CPU mutation is a separate fixture"
    );
    let f = fixture(used, overlap);
    let _outer = owned_topk::test_scope(&f.backend, enabled);
    let ids = (0..used as u32).collect::<Vec<_>>();
    preload(&f, &ids);
    let plan = f.backend.compile(&f.graph).unwrap();
    f.backend.execute(plan.as_ref(), &f.bindings()).unwrap();
    if cpu_miss {
        cpu_miss_last_layer(&f, 1);
    }
    let next = prepared.then(|| f.backend.compile(&f.graph).unwrap());
    let fused_supported = enabled && owned_topk::test_supported_mode(&f.backend, overlap);
    let mut prepared_layers = 0;
    if let Some(next) = &next {
        f.reset_frontend();
        let before = owned_topk::test_counts(&f.backend);
        f.backend
            .execute_with_lookahead(
                plan.as_ref(),
                &f.bindings(),
                Some((next.as_ref(), &f.bindings(), &[0])),
            )
            .unwrap();
        prepared_layers = next
            .as_any()
            .downcast_ref::<VkDecodePlan>()
            .unwrap()
            .prepared_prefix
            .lock()
            .unwrap()
            .as_ref()
            .expect("actual packet must be prepared")
            .recorded_shape()
            .3;
        assert!(prepared_layers > 0 && prepared_layers <= LAYERS);
        let recorded = owned_topk::test_counts(&f.backend);
        let route_count = LAYERS + prepared_layers;
        assert_eq!(recorded.admissions - before.admissions, route_count);
        assert_eq!(recorded.recovery - before.recovery, 0);
        assert_eq!(
            recorded.fused - before.fused,
            if fused_supported { route_count } else { 0 }
        );
        assert_eq!(
            recorded.routers - before.routers,
            if fused_supported { 0 } else { route_count }
        );
    }
    f.reset_frontend();
    let _measured = owned_topk::test_scope(&f.backend, enabled);
    f.backend
        .execute(next.as_deref().unwrap_or(plan.as_ref()), &f.bindings())
        .unwrap();
    if let Some(next) = &next {
        assert!(
            next.as_any()
                .downcast_ref::<VkDecodePlan>()
                .unwrap()
                .prepared_prefix
                .lock()
                .unwrap()
                .is_none(),
            "packet must be consumed"
        );
    }
    f.assert_frontend(f.buffer(f.frontend), LAYERS as f32);
    let counts = owned_topk::test_counts(&f.backend);
    // These counters measure CPU encoding. Already prepared layers execute without
    // entering the encoder again; complete replay must therefore record zero routes.
    let encoded_layers = LAYERS - prepared_layers;
    assert_eq!(
        counts.admissions,
        encoded_layers + counts.recovery,
        "{counts:?}"
    );
    assert_eq!(
        counts.fused + counts.routers,
        encoded_layers,
        "recovery must reuse its IDs/weights, never record a second router: {counts:?}"
    );
    if fused_supported {
        assert_eq!(
            counts.fused, encoded_layers,
            "eligible encoding must use fusion"
        );
    } else {
        assert_eq!(counts.fused, 0);
        assert_eq!(counts.routers, encoded_layers);
    }
    assert_eq!(counts.recovery != 0, cpu_miss, "{counts:?}");
    let outputs = f
        .outputs()
        .into_iter()
        .map(|row| {
            assert!(row.iter().all(|x| x.is_finite() && *x > 0.0));
            row.into_iter().map(f32::to_bits).collect()
        })
        .collect();
    let owner = f
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .gpu_decode_owner
        .clone()
        .unwrap();
    let mut directory = vec![0u32; owner.candidates.len_bytes() / 4];
    let mut overlay = vec![0u32; owner.overlay.len_bytes() / 4];
    f.backend
        .download(
            owner.candidates.as_ref(),
            bytemuck::cast_slice_mut(&mut directory),
        )
        .unwrap();
    f.backend
        .download(
            owner.overlay.as_ref(),
            bytemuck::cast_slice_mut(&mut overlay),
        )
        .unwrap();
    normalize_addresses(&mut directory, &mut overlay);
    let mut guard = f.backend.moe_pager().lock().unwrap();
    let session = guard.as_mut().unwrap();
    session.handoff_gpu_decode(&f.backend).unwrap();
    for roles in f.roles {
        assert_eq!(
            session.routed_roles_resident_mask(&roles, &ids).unwrap(),
            (1 << used) - 1
        );
    }
    let mut seed = crate::pager::owned_seed(session, 0).unwrap();
    // Separate backends have different BDAs. Compare raw live directory/overlay above,
    // and the independent validated handoff/reseed below; normalize only address words.
    normalize_addresses(&mut seed.nodes, &mut seed.blocks);
    ResultState {
        outputs,
        directory,
        overlay,
        nodes: seed.nodes,
        blocks: seed.blocks,
    }
}

#[test]
#[ignore = "central GPU runner only; production used8/10 router+owner versus separate dispatches"]
fn gpu_owned_topk_actual_graph_used8_used10_exact_and_admissions() {
    for used in [8, 10] {
        assert_eq!(
            run(used, true, false),
            run(used, false, false),
            "used={used}"
        );
    }
}

#[test]
#[ignore = "central GPU runner only; production CPU admission recovery must not reroute"]
fn gpu_owned_topk_actual_graph_used8_used10_cpu_recovery_no_reroute() {
    for used in [8, 10] {
        assert_eq!(run(used, true, true), run(used, false, true), "used={used}");
    }
}

#[test]
#[ignore = "central GPU runner only; fused overlap adapter wiring, real copies and CPU recovery"]
fn gpu_owned_topk_actual_overlap_graph_exact_and_recovery() {
    for used in [8, 10] {
        for cpu_miss in [false, true] {
            assert_eq!(
                run_mode(used, true, cpu_miss, true, false),
                run_mode(used, false, cpu_miss, true, false),
                "overlap used={used} cpu_miss={cpu_miss}"
            );
        }
    }
}

#[test]
#[ignore = "central GPU runner only; prepared overlap packets must consume fused routing"]
fn gpu_owned_topk_actual_overlap_prepared_packet_exact() {
    for used in [8, 10] {
        assert_eq!(
            run_mode(used, true, false, true, true),
            run_mode(used, false, false, true, true),
            "prepared overlap used={used}"
        );
    }
}
