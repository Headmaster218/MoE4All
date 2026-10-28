//! Candidate only: model-free graph qualification, not an image-model acceptance test.
use super::single_verify_tests::VerifyGraph;
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub(super) record: usize,
    pub(super) overlap: usize,
    serial: usize,
    host_ready_create: usize,
    host_ready_created: usize,
    pub(super) early_submit: usize,
}

fn observations() -> &'static Mutex<HashMap<usize, Counts>> {
    static STATES: OnceLock<Mutex<HashMap<usize, Counts>>> = OnceLock::new();
    STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

// Called only from cfg(test) sites; key follows the real backend onto its recording worker.
pub(crate) fn note(be: &VulkanBackend, event: &'static str) {
    let mut states = observations().lock().unwrap();
    let Some(counts) = states.get_mut(&(Arc::as_ptr(&be.shared) as usize)) else {
        return;
    };
    match event {
        "record" => counts.record += 1,
        "overlap" => counts.overlap += 1,
        "serial" => counts.serial += 1,
        "host_ready_create" => counts.host_ready_create += 1,
        "host_ready_created" => counts.host_ready_created += 1,
        "early_submit" => counts.early_submit += 1,
        _ => panic!("unknown promotion observation {event}"),
    }
}

pub(super) struct Trace(Arc<crate::VulkanShared>);

impl Trace {
    pub(super) fn begin(be: &VulkanBackend) -> Self {
        assert!(observations()
            .lock()
            .unwrap()
            .insert(Arc::as_ptr(&be.shared) as usize, Counts::default(),)
            .is_none());
        Self(Arc::clone(&be.shared))
    }

    pub(super) fn counts(&self) -> Counts {
        observations().lock().unwrap()[&(Arc::as_ptr(&self.0) as usize)]
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        observations()
            .lock()
            .unwrap()
            .remove(&(Arc::as_ptr(&self.0) as usize));
    }
}

fn all5(cfg: &mut infr_core::config::Config, enabled: bool) {
    cfg.device.decode_command_ahead = enabled;
    cfg.device.decode_command_early_submit = enabled;
    cfg.device.decode_command_resident = enabled;
    cfg.kernels.vulkan.decode_gpu_resident_copy_overlap = enabled;
    cfg.kernels.vulkan.decode_gpu_resident_host_ready = enabled;
}

fn fixture(mtp: bool) -> Fixture {
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let cfg = Arc::make_mut(&mut fixture.backend.cfg);
    all5(cfg, true);
    cfg.spec.mtp = mtp;
    fixture
}

fn no_new_mtp_path(trace: &Trace) {
    let counts = trace.counts();
    assert_eq!(
        counts.record, 0,
        "MTP must not attempt optional recording: {counts:?}"
    );
    assert_eq!(
        counts.overlap, 0,
        "MTP must retain serial completion: {counts:?}"
    );
    assert_eq!(
        counts.host_ready_create, 0,
        "MTP must not even create the new worker: {counts:?}"
    );
}

fn verify_bindings<'a>(fixture: &'a Fixture, verify: &'a VerifyGraph) -> Bindings<'a> {
    let mut bindings = fixture.bindings();
    for (id, buffer) in &verify.buffers {
        bindings.bind(*id, buffer.as_ref());
    }
    bindings
}

#[test]
fn promotion_isolation_cpu_feature_switches_do_not_enable_mtp_owner() {
    let mut cfg = infr_core::config::Config::default();
    all5(&mut cfg, true);
    for mtp in [false, true] {
        cfg.spec.mtp = mtp;
        for verify in [false, true] {
            let mut graph = Graph::new();
            graph.mtp_verify = verify;
            assert_eq!(
                resident_copy_owned_allowed(cfg.spec.mtp, &graph),
                !mtp && !verify
            );
            let mut creates = 0;
            assert_eq!(
                resident_host_ready_worker(true, mtp, !verify, || {
                    creates += 1;
                    Ok(17u32)
                }),
                (!mtp && !verify).then_some(17)
            );
            assert_eq!(creates, usize::from(!mtp && !verify));
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; all5 MTP target/marked Verify/no-MoE draft isolation"]
fn gpu_promotion_isolation_all5_mtp_target_verify_and_draft() {
    let fixture = fixture(true);
    let trace = Trace::begin(&fixture.backend);
    assert!(!command_prefix::enabled(&fixture.backend));
    assert!(!fixture.backend.supports_execution_lookahead());
    fixture.preload();
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.reset_frontend();
    execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
    let expected = fixture.outputs();
    let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        session.discard_cpu_miss(fixture.roles[0], &[ROUTED[0]]);
    }
    fixture.reset_frontend();
    execute_static_next(
        &fixture.backend,
        &current,
        &fixture.bindings(),
        Some((&next, &fixture.bindings(), &[1])),
    )
    .unwrap();
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_close(&fixture.outputs(), &expected);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
    {
        let arena = fixture.backend.runtime_phase.lock().unwrap();
        let copy = arena
            .resident_buffers
            .as_ref()
            .unwrap()
            .gpu_copy
            .as_ref()
            .expect("explicit legacy copy remains available to the ordinary target");
        assert!(!copy.is_owned());
        assert!(
            copy.miss_predicates.is_none(),
            "MTP cannot allocate overlap predicates"
        );
    }
    assert!(
        trace.counts().serial > 0,
        "must exercise real legacy recording"
    );
    no_new_mtp_path(&trace);

    let verify = VerifyGraph::new(&fixture, 2, 1);
    verify.run(&fixture, true);
    let verify_current = VkDecodePlan::new_cpu(&verify.graph, false, true);
    let verify_next = VkDecodePlan::new_cpu(&verify.graph, false, true);
    let bindings = verify_bindings(&fixture, &verify);
    execute_static_next(
        &fixture.backend,
        &verify_current,
        &bindings,
        Some((&verify_next, &bindings, &[1])),
    )
    .unwrap();
    assert!(verify_next.prepared_prefix.lock().unwrap().is_none());
    no_new_mtp_path(&trace);

    let mut draft_graph = fixture.graph.clone();
    draft_graph.ops.retain(|op| matches!(op, Op::Add { .. }));
    assert!(!draft_graph.ops.is_empty());
    assert!(!draft_graph
        .ops
        .iter()
        .any(|op| matches!(op, Op::MoeFfn { .. })));
    let draft = VkDecodePlan::new_cpu(&draft_graph, false, true);
    let draft_next = VkDecodePlan::new_cpu(&draft_graph, false, true);
    let before = trace.counts();
    fixture.reset_frontend();
    execute_static_next(
        &fixture.backend,
        &draft,
        &fixture.bindings(),
        Some((&draft_next, &fixture.bindings(), &[1])),
    )
    .unwrap();
    fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
    assert!(draft_next.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(
        trace.counts(),
        before,
        "no-MoE draft must not record resident/copy work"
    );
}

#[test]
#[ignore = "parent serial GPU only; explicit legacy Verify copy records the serial fallback"]
fn gpu_promotion_isolation_verify_explicit_copy_records_serial_fallback() {
    let mut fixture = fixture(false);
    // Make the probe specifically about the new Verify guard, not pair exclusivity.
    Arc::make_mut(&mut fixture.backend.cfg)
        .kernels
        .vulkan
        .verify_pair = false;
    let trace = Trace::begin(&fixture.backend);
    for rows in [2, 4] {
        let verify = VerifyGraph::new(&fixture, rows, 1);
        verify.run(&fixture, true); // Existing scalar oracle, forced real miss and exact owner checks.
        let before = trace.counts();
        let current = VkDecodePlan::new_cpu(&verify.graph, false, true);
        let next = VkDecodePlan::new_cpu(&verify.graph, false, true);
        let bindings = verify_bindings(&fixture, &verify);
        execute_static_next(
            &fixture.backend,
            &current,
            &bindings,
            Some((&next, &bindings, &[1])),
        )
        .unwrap();
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        assert_eq!(trace.counts().record, before.record);
        let arena = fixture.backend.runtime_phase.lock().unwrap();
        let copy = arena
            .resident_buffers
            .as_ref()
            .unwrap()
            .gpu_copy
            .as_ref()
            .expect("explicit Some(true) must really instantiate legacy Verify copy");
        assert!(!copy.is_owned());
        assert!(
            copy.miss_predicates.is_some(),
            "probe must not pass just because predicates are absent"
        );
    }
    let counts = trace.counts();
    assert!(
        counts.serial > 0,
        "actual serial-copy command stream required: {counts:?}"
    );
    assert_eq!(
        counts.overlap, 0,
        "Verify must never select overlap: {counts:?}"
    );
    assert_eq!(
        counts.host_ready_create, 0,
        "Verify must not create an owned completion worker"
    );
}

#[test]
#[ignore = "parent serial GPU only; unmarked row1 prompt-feed graph to accepted NEXT decode"]
fn gpu_promotion_isolation_unmarked_prompt_feed_to_next_decode() {
    let fixture = fixture(false);
    let trace = Trace::begin(&fixture.backend);
    fixture.preload();
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    assert!(
        !current.graph.mtp_verify
            && current.graph.prefill_target_tokens.is_none()
            && !current.graph.prefill_next_chunk
            && !current.graph.independent_rows
    );
    fixture.reset_frontend();
    execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
    let expected = fixture.outputs();
    let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.reset_frontend();
    execute_static_next(
        &fixture.backend,
        &current,
        &fixture.bindings(),
        Some((&next, &fixture.bindings(), &[1])),
    )
    .unwrap();
    assert_close(&fixture.outputs(), &expected);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
    assert!(
        next.prepared_prefix
            .lock()
            .unwrap()
            .as_ref()
            .expect("NEXT must contain a real resident packet")
            .resident_layers()
            > 0
    );
    assert!(fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_ref()
        .unwrap()
        .gpu_copy
        .as_ref()
        .unwrap()
        .is_owned());
    assert!(trace.counts().record > 0);
    let accepted = command_prefix::test_accepted_prefixes();
    let early_submits = trace.counts().early_submit;
    fixture.reset_frontend();
    execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
    assert_eq!(command_prefix::test_accepted_prefixes(), accepted + 1);
    assert!(
        trace.counts().early_submit > early_submits,
        "accepted resident packet must actually submit before main recording"
    );
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_close(&fixture.outputs(), &expected);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
    assert!(
        trace.counts().overlap > 0,
        "all5 must actually exercise overlap"
    );
    assert!(
        trace.counts().host_ready_created > 0,
        "all5 must successfully create host-ready support, not merely attempt it"
    );
}

#[test]
#[ignore = "parent serial GPU only; warmed all5 Lookup excludes pre-recording without changing owner/math"]
fn gpu_promotion_isolation_lookup_warmed_disables_only_next_recording() {
    let mut fixture = fixture(false);
    fixture.preload();
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.reset_frontend();
    execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
    let expected = fixture.outputs();
    let owner = fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .gpu_decode_owner
        .clone()
        .expect("actual warm owner, not an uninitialized pager");
    assert!(fixture.backend.moe_paged() && owner.is_owned());
    assert!(fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_ref()
        .unwrap()
        .gpu_copy
        .as_ref()
        .is_some_and(|copy| Arc::ptr_eq(copy, &owner)));
    let trace = Trace::begin(&fixture.backend);
    // Positive controls on both sides prevent supported=false due to cold/unsupported state
    // from disguising a missing Lookup gate. No resources or owner policy are replaced.
    for lookup in [false, true, false] {
        Arc::make_mut(&mut fixture.backend.cfg).spec.lookup = lookup;
        assert_eq!(command_prefix::supported(&fixture.backend), !lookup);
        assert_eq!(fixture.backend.supports_execution_lookahead(), !lookup);
        let before = trace.counts();
        let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
        fixture.reset_frontend();
        execute_static_next(
            &fixture.backend,
            &current,
            &fixture.bindings(),
            Some((&next, &fixture.bindings(), &[1])),
        )
        .unwrap();
        assert_eq!(next.prepared_prefix.lock().unwrap().is_some(), !lookup);
        if lookup {
            assert_eq!(trace.counts().record, before.record);
        } else {
            assert!(trace.counts().record > before.record);
        }
        fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
        assert_close(&fixture.outputs(), &expected);
        let accepted = command_prefix::test_accepted_prefixes();
        fixture.reset_frontend();
        execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
        assert_eq!(
            command_prefix::test_accepted_prefixes(),
            accepted + usize::from(!lookup)
        );
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
        assert_close(&fixture.outputs(), &expected);
        assert!(
            trace.counts().overlap > before.overlap,
            "ordinary Lookup fallback remains eligible for copy overlap"
        );
        assert!(Arc::ptr_eq(
            fixture
                .backend
                .moe_pager()
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .gpu_decode_owner
                .as_ref()
                .unwrap(),
            &owner
        ));
    }
}

struct PositionInputs {
    x: TensorId,
    weight: TensorId,
    positions: TensorId,
    positions4: TensorId,
    dst: TensorId,
}

impl PositionInputs {
    fn add(graph: &mut Graph) -> Self {
        Self {
            x: graph.input(TensorDesc::new([1, NE], DType::F32)),
            weight: graph.weight(TensorDesc::new([NE], DType::F32)),
            positions: graph.input(TensorDesc::new([1], DType::I32)),
            positions4: graph.input(TensorDesc::new([1, 4], DType::I32)),
            dst: graph.output(TensorDesc::new([1, NE], DType::F16)),
        }
    }

    fn op(&self, mrope: bool) -> Op {
        if mrope {
            Op::QkNormMrope {
                x: self.x,
                weight: self.weight,
                positions4: self.positions4,
                dst: self.dst,
                rows: 1,
                n_head: 1,
                head_dim: NE as u32,
                rope_dim: 8,
                theta: 10000.0,
                eps: 1e-6,
                sections: [1; 4],
                x_stride: 0,
            }
        } else {
            Op::QkNormRope {
                x: self.x,
                weight: self.weight,
                positions: self.positions,
                dst: self.dst,
                rows: 1,
                n_head: 1,
                head_dim: NE as u32,
                rope_dim: 8,
                theta: 10000.0,
                eps: 1e-6,
                freq_factors: None,
                x_stride: 0,
            }
        }
    }

    fn bind(&self, fixture: &mut Fixture) {
        for (id, bytes, usage) in [
            (
                self.x,
                bytemuck::cast_slice(&[1.0f32; NE]).to_vec(),
                BufferUsage::Activations,
            ),
            (
                self.weight,
                bytemuck::cast_slice(&[1.0f32; NE]).to_vec(),
                BufferUsage::Weights,
            ),
            (
                self.positions,
                0i32.to_le_bytes().to_vec(),
                BufferUsage::Staging,
            ),
            (
                self.positions4,
                bytemuck::cast_slice(&[2i32, 3, 5, 0]).to_vec(),
                BufferUsage::Staging,
            ),
            (self.dst, vec![0u8; NE * 2], BufferUsage::Readback),
        ] {
            let buffer = fixture.backend.alloc(bytes.len(), usage).unwrap();
            fixture.backend.upload(buffer.as_ref(), &bytes).unwrap();
            fixture.buffers.push((id, buffer));
        }
    }

    fn position(&self, fixture: &Fixture, value: i32) {
        fixture
            .backend
            .upload(fixture.buffer(self.positions), &value.to_le_bytes())
            .unwrap();
    }

    fn bytes(&self, fixture: &Fixture) -> Vec<u8> {
        let mut bytes = vec![0; NE * 2];
        fixture
            .backend
            .download(fixture.buffer(self.dst), &mut bytes)
            .unwrap();
        bytes
    }
}

#[test]
fn promotion_isolation_cpu_mrope_markers_and_hidden_shape_boundaries() {
    // No model/backend needed: add the same real op ABI to a minimal supported MoE graph.
    let mut graph = Graph::new();
    let x = graph.input(TensorDesc::new([1, NE], DType::F32));
    let router = graph.weight(TensorDesc::new([EXP, NE], DType::F32));
    let gate = graph.weight(TensorDesc::new([EXP, FF, NE], DType::F32));
    let up = graph.weight(TensorDesc::new([EXP, FF, NE], DType::F32));
    let down = graph.weight(TensorDesc::new([EXP, NE, FF], DType::F32));
    let dst = graph.output(TensorDesc::new([1, NE], DType::F32));
    let p = PositionInputs::add(&mut graph);
    graph.push(p.op(false));
    graph.push(Op::MoeFfn {
        x,
        router_x: x,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: NE as u32,
        n_expert: EXP as u32,
        n_used: TOP as u32,
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
    assert!(command_prefix::prefix_end(&graph).is_some());
    for mode in 0..4 {
        let mut rejected = graph.clone();
        match mode {
            0 => rejected.ops[0] = p.op(true),
            1 => rejected.mtp_verify = true,
            2 => rejected.prefill_target_tokens = Some(1),
            _ => rejected.prefill_next_chunk = true,
        }
        assert_eq!(command_prefix::prefix_end(&rejected), None);
    }
    let scratch: ScratchSet = scratch_layout(&graph)
        .unwrap()
        .iter()
        .map(|n| n.map(|n| Box::new(CpuBuffer(n)) as Box<dyn Buffer>))
        .collect();
    let mut hidden = graph.clone();
    let tap = hidden.output(TensorDesc::new([1, NE], DType::F32));
    hidden.push(Op::Copy {
        src: dst,
        src_off: 0,
        dst: tap,
        dst_off: 0,
        n: NE as u32,
    });
    assert!(!command_prefix::compatible(&graph, &hidden, &scratch));
}

#[test]
#[ignore = "parent serial GPU only; real MRoPE rejection then plain text/rebind/hidden shape"]
fn gpu_promotion_isolation_mrope_plain_rebind_and_hidden_shape() {
    let mut fixture = fixture(false);
    let p = PositionInputs::add(&mut fixture.graph);
    p.bind(&mut fixture);
    let mut mrope_graph = fixture.graph.clone();
    mrope_graph.ops.insert(0, p.op(true));
    let mut plain_graph = fixture.graph.clone();
    plain_graph.ops.insert(0, p.op(false));
    fixture.preload();
    let mrope = VkDecodePlan::new_cpu(&mrope_graph, false, true);
    let declined = VkDecodePlan::new_cpu(&mrope_graph, false, true);
    let trace = Trace::begin(&fixture.backend);
    fixture.reset_frontend();
    execute_static(&fixture.backend, &mrope, &fixture.bindings()).unwrap();
    let expected_mrope = p.bytes(&fixture);
    assert!(expected_mrope.chunks_exact(2).all(|bytes| {
        half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).is_finite()
    }));
    fixture.reset_frontend();
    execute_static_next(
        &fixture.backend,
        &mrope,
        &fixture.bindings(),
        Some((&declined, &fixture.bindings(), &[1])),
    )
    .unwrap();
    assert!(declined.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(
        trace.counts().record,
        0,
        "unsupported MRoPE must decline before recording"
    );
    assert_eq!(p.bytes(&fixture), expected_mrope);
    let expected_moe = fixture.outputs();

    // Oracle uses the ordinary recorder and exactly the same plain-RoPE kernel/position.
    let plain = VkDecodePlan::new_cpu(&plain_graph, false, true);
    p.position(&fixture, 1);
    fixture.reset_frontend();
    execute_static(&fixture.backend, &plain, &fixture.bindings()).unwrap();
    let expected_plain = p.bytes(&fixture);
    assert_ne!(
        expected_mrope, expected_plain,
        "the MRoPE and plain NEXT probes must be distinguishable"
    );
    p.position(&fixture, 0);
    let next = VkDecodePlan::new_cpu(&plain_graph, false, true);
    fixture.reset_frontend();
    execute_static_next(
        &fixture.backend,
        &mrope,
        &fixture.bindings(),
        Some((&next, &fixture.bindings(), &[1])),
    )
    .unwrap();
    assert!(
        next.prepared_prefix.lock().unwrap().is_some(),
        "plain text should remain eligible after MRoPE"
    );
    assert_eq!(
        p.bytes(&fixture),
        expected_mrope,
        "unsubmitted plain NEXT must not execute over the current MRoPE result"
    );
    p.position(&fixture, 1);
    fixture.reset_frontend();
    let accepted = command_prefix::test_accepted_prefixes();
    execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
    assert_eq!(command_prefix::test_accepted_prefixes(), accepted + 1);
    assert_eq!(p.bytes(&fixture), expected_plain);
    assert_close(&fixture.outputs(), &expected_moe);

    // A changed physical input rejects the actual recorded packet. The ordinary fallback
    // writes the replacement exactly once, without touching the original frontend input.
    p.position(&fixture, 1);
    let rebound = VkDecodePlan::new_cpu(&plain_graph, false, true);
    fixture.reset_frontend();
    execute_static_next(
        &fixture.backend,
        &plain,
        &fixture.bindings(),
        Some((&rebound, &fixture.bindings(), &[2])),
    )
    .unwrap();
    assert!(rebound.prepared_prefix.lock().unwrap().is_some());
    let replacement = fixture.backend.alloc(NE * 4, BufferUsage::Staging).unwrap();
    fixture
        .backend
        .upload(replacement.as_ref(), bytemuck::cast_slice(&[0.0f32; NE]))
        .unwrap();
    let mut bindings = fixture.bindings();
    bindings.bind(fixture.frontend, replacement.as_ref());
    p.position(&fixture, 2);
    fixture.reset_frontend();
    let accepted = command_prefix::test_accepted_prefixes();
    execute_static(&fixture.backend, &rebound, &bindings).unwrap();
    assert_eq!(
        command_prefix::test_accepted_prefixes(),
        accepted,
        "rebind cannot accept stale commands"
    );
    fixture.assert_frontend(replacement.as_ref(), LAYERS as f32);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 0.0);
    assert!(rebound.prepared_prefix.lock().unwrap().is_none());
    assert_close(&fixture.outputs(), &expected_moe);

    // A new hidden output changes the next plan's physical layout. Offer it through the
    // real lookahead entry; it must fall back, not claim a packet for the old shape.
    let mut hidden_graph = plain_graph.clone();
    let hidden = hidden_graph.output(TensorDesc::new([1, NE], DType::F32));
    hidden_graph.push(Op::Copy {
        src: fixture.outputs[1],
        src_off: 0,
        dst: hidden,
        dst_off: 0,
        n: NE as u32,
    });
    let hidden_buffer = fixture
        .backend
        .alloc(NE * 4, BufferUsage::Readback)
        .unwrap();
    let mut hidden_bindings = fixture.bindings();
    hidden_bindings.bind(hidden, hidden_buffer.as_ref());
    let hidden_plan = VkDecodePlan::new_cpu(&hidden_graph, false, true);
    let recorded = trace.counts().record;
    p.position(&fixture, 2);
    fixture.reset_frontend();
    execute_static_next(
        &fixture.backend,
        &plain,
        &fixture.bindings(),
        Some((&hidden_plan, &hidden_bindings, &[3])),
    )
    .unwrap();
    assert!(hidden_plan.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(trace.counts().record, recorded);
    let accepted = command_prefix::test_accepted_prefixes();
    p.position(&fixture, 3);
    fixture.reset_frontend();
    execute_static(&fixture.backend, &hidden_plan, &hidden_bindings).unwrap();
    assert_eq!(command_prefix::test_accepted_prefixes(), accepted);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
    assert_close(&fixture.outputs(), &expected_moe);
    assert_eq!(fixture.read(hidden_buffer.as_ref()), expected_moe[1]);
}
