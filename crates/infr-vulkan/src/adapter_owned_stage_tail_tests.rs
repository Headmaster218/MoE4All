//! Chunk-boundary provenance and terminal pointwise guarded-recovery suffixes.
use super::*;

#[test]
fn owned_prepared_boundary_empty_requires_submitted_chunk_not_complete_graph() {
    // The graph may still have HC ops, but this full chunk has no new commands/jobs.
    assert!(owned_prepared_stage_empty_eligible(
        true, true, true, 1, 1, 1, 0
    ));
    for (early, boundary, no_parts, entries, submitted, sent, dispatches) in [
        (false, true, true, 1, 1, 1, 0),
        (true, false, true, 1, 1, 1, 0),
        (true, true, false, 1, 1, 1, 0),
        (true, true, true, 0, 0, 0, 0),
        (true, true, true, 2, 1, 1, 0),
        (true, true, true, 1, 1, 0, 0),
        (true, true, true, 1, 1, 1, 1),
    ] {
        assert!(!owned_prepared_stage_empty_eligible(
            early, boundary, no_parts, entries, submitted, sent, dispatches,
        ));
    }
}

#[test]
fn owned_recovery_stage_tail_whitelist_is_all_or_nothing_and_capacity_checked() {
    let mut graph = Graph::new();
    let x = graph.input(TensorDesc::new([32], DType::F32));
    let y = graph.output(TensorDesc::new([32], DType::F32));
    let gate = graph.input(TensorDesc::new([1], DType::F32));
    let bx = CpuBuffer(128);
    let by = CpuBuffer(128);
    let bg = CpuBuffer(4);
    let mut bindings = Bindings::new();
    bindings.bind(x, &bx);
    bindings.bind(y, &by);
    bindings.bind(gate, &bg);
    let scratch = (0..graph.tensors.len()).map(|_| None).collect();
    let hc = Op::QwenHcInject {
        residual: x,
        block: x,
        gate,
        dst: y,
        rows: 1,
        hc: 1,
        n_embd: 32,
    };
    let scale = Op::Scale {
        x: y,
        dst: y,
        s: 2.0,
        n: 32,
    };
    assert!(owned_recovery_stage_tail_ready(
        &graph,
        &[(0, &hc), (1, &scale)],
        &scratch,
        &bindings
    ));
    assert!(!owned_recovery_stage_tail_ready(
        &graph,
        &[],
        &scratch,
        &bindings
    ));
    let short = CpuBuffer(124);
    bindings.bind(y, &short);
    assert!(!owned_recovery_stage_tail_ready(
        &graph,
        &[(0, &hc)],
        &scratch,
        &bindings
    ));
    bindings.bind(y, &by);
    // Transfer/host-sensitive/unknown lowering is rejected even AFTER a valid prefix.
    let copy = Op::Copy {
        src: x,
        src_off: 0,
        dst: y,
        dst_off: 0,
        n: 32,
    };
    let sample = Op::Argmax {
        x,
        dst: y,
        n: 32,
        rows: 1,
    };
    let unknown = Op::Softcap {
        x,
        dst: y,
        cap: 1.0,
        n: 32,
    };
    let host_boundary = Op::WriteKv {
        src: x,
        cache: y,
        rows: 1,
        row_stride: 32,
        pos: 0,
    };
    for op in [&copy, &sample, &unknown, &host_boundary] {
        assert!(!owned_recovery_stage_tail_ready(
            &graph,
            &[(0, &hc), (1, op)],
            &scratch,
            &bindings,
        ));
    }
    let overflow = Op::QwenHcInject {
        residual: x,
        block: x,
        gate,
        dst: y,
        rows: u32::MAX,
        hc: u32::MAX,
        n_embd: u32::MAX,
    };
    assert!(!owned_recovery_stage_tail_ready(
        &graph,
        &[(0, &overflow)],
        &scratch,
        &bindings
    ));
    let router = setup_cache_graph()
        .0
        .ops
        .into_iter()
        .find(|op| matches!(op, Op::Linear { .. }))
        .unwrap();
    assert!(!owned_recovery_stage_tail_ready(
        &graph,
        &[(0, &hc), (1, &router)],
        &scratch,
        &bindings
    ));
}

struct TraceGuard;

impl TraceGuard {
    fn begin(disable_merge: bool, corrupt_guard: bool, failure: Option<(usize, u8)>) -> Self {
        OWNED_STAGE_TAIL_TEST_STATE.with(|state| {
            assert!(state
                .replace(Some(OwnedStageTailTestState {
                    disable_merge,
                    corrupt_guard,
                    failure,
                    ..Default::default()
                }))
                .is_none());
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| assert!(state.replace(Some([0; 3])).is_none()));
        Self
    }

    fn counts(&self) -> (usize, usize, usize, Vec<usize>) {
        OWNED_STAGE_TAIL_TEST_STATE.with(|state| {
            let state = state.borrow();
            let state = state.as_ref().unwrap();
            (
                state.recorded_ops,
                state.merged_submits,
                state.standalone_tail_submits,
                state.recovery_dispatches.clone(),
            )
        })
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        OWNED_STAGE_TAIL_TEST_STATE.with(|state| {
            state.take();
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.set(None));
    }
}

fn stage_fixture(extra_copy: bool) -> (Fixture, VkDecodePlan, TensorId, Vec<f32>) {
    stage_fixture_rows(extra_copy, 1)
}

fn stage_fixture_rows(
    extra_copy: bool,
    rows: usize,
) -> (Fixture, VkDecodePlan, TensorId, Vec<f32>) {
    stage_fixture_with_cadence(extra_copy, rows, 1)
}

fn mapped_stage_moe(fixture: &Fixture, op: &Op) -> Op {
    let roles = fixture.roles[1].map(|role| {
        fixture
            .buffers
            .iter()
            .find(|(_, buffer)| crate::pager::buffer_identity(buffer.as_ref()) == role)
            .expect("registered mapped bank binding")
            .0
    });
    let mut op = op.clone();
    let Op::MoeFfn {
        gate_exps,
        up_exps,
        down_exps,
        ..
    } = &mut op
    else {
        panic!("expected paged MoE");
    };
    *gate_exps = roles[0];
    *up_exps = roles[1];
    *down_exps = roles[2];
    op
}

fn stage_fixture_with_cadence(
    extra_copy: bool,
    rows: usize,
    submit_layers: usize,
) -> (Fixture, VkDecodePlan, TensorId, Vec<f32>) {
    assert!(matches!(submit_layers, 1 | 2));
    let mut fixture = Fixture::new_with_copy_policy(submit_layers, true, Some(true));
    if rows != 1 {
        for (index, tensor) in fixture.graph.tensors.iter_mut().enumerate() {
            if tensor.desc.shape.as_slice() != [1, NE] {
                continue;
            }
            tensor.desc.shape = vec![rows, NE].into();
            if let Some((_, buffer)) = fixture
                .buffers
                .iter_mut()
                .find(|(id, _)| id.0 as usize == index)
            {
                let mut bytes = vec![0u8; NE * 4];
                fixture
                    .backend
                    .download(buffer.as_ref(), &mut bytes)
                    .unwrap();
                let replacement = fixture
                    .backend
                    .alloc(rows * NE * 4, BufferUsage::Staging)
                    .unwrap();
                fixture
                    .backend
                    .upload(replacement.as_ref(), &bytes.repeat(rows))
                    .unwrap();
                *buffer = replacement;
            }
        }
        for op in &mut fixture.graph.ops {
            if let Op::Add { n, .. } = op {
                *n *= rows as u32;
            }
        }
        fixture.graph.independent_rows = true;
        fixture.graph.sequence_spans = (0..rows)
            .map(|row| infr_core::graph::SequenceSpan {
                row_start: row as u32,
                rows: 1,
                start_pos: (3 + row * 4) as u32,
            })
            .collect();
    }
    fixture.graph.ops.truncate(2);
    if submit_layers == 2 {
        // A two-layer cadence needs a two-layer head. Keep the CPU miss last so its
        // real guarded recovery still owns the terminal pointwise stage suffix.
        let mapped = mapped_stage_moe(&fixture, &fixture.graph.ops[1]);
        fixture.graph.ops.insert(1, mapped);
    }
    let gate = fixture.graph.input(TensorDesc::new([rows], DType::F32));
    let out = fixture
        .graph
        .output(TensorDesc::new([rows, NE], DType::F32));
    let gate_buffer = fixture
        .backend
        .alloc(rows * 4, BufferUsage::Activations)
        .unwrap();
    fixture
        .backend
        .upload(
            gate_buffer.as_ref(),
            bytemuck::cast_slice(&vec![0f32; rows]),
        )
        .unwrap();
    fixture.buffers.push((gate, gate_buffer));
    fixture.buffers.push((
        out,
        fixture
            .backend
            .alloc(rows * NE * 4, BufferUsage::Activations)
            .unwrap(),
    ));
    fixture.graph.push(Op::QwenHcInject {
        residual: fixture.frontend,
        block: fixture.outputs[0],
        gate,
        dst: out,
        rows: rows as u32,
        hc: 1,
        n_embd: NE as u32,
    });
    fixture.graph.push(Op::Scale {
        x: out,
        dst: out,
        s: 2.0,
        n: (rows * NE) as u32,
    });
    if extra_copy {
        fixture.graph.push(Op::Copy {
            src: out,
            src_off: 0,
            dst: fixture.outputs[1],
            dst_off: 0,
            n: (rows * NE) as u32,
        });
    }
    fixture.preload();
    let plan = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    let mut expected = vec![0f32; rows * NE];
    fixture
        .backend
        .download(fixture.buffer(out), bytemuck::cast_slice_mut(&mut expected))
        .unwrap();
    assert!(expected.iter().all(|x| x.is_finite() && *x > 0.0));
    install_cpu_head_owner(&fixture);
    fixture
        .backend
        .upload(
            fixture.buffer(fixture.frontend),
            bytemuck::cast_slice(&vec![0f32; rows * NE]),
        )
        .unwrap();
    (fixture, plan, out, expected)
}

fn install_cpu_head_owner(fixture: &Fixture) {
    install_cpu_head_owner_with_capacity(fixture, 0);
}

fn install_cpu_head_owner_with_capacity(fixture: &Fixture, minimum_capacity: usize) {
    let capacity = fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_ref()
        .unwrap()
        .capacity
        .max(minimum_capacity);
    let (sources, seed) = {
        let mut pager = fixture.backend.moe_pager().lock().unwrap();
        let session = pager.as_mut().unwrap();
        let previous = session.gpu_decode_owner.clone().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        let banks = previous.role_params(session, &fixture.roles[0]).unwrap();
        session.discard_cpu_miss(fixture.roles[0], &[ROUTED[1]]);
        session.gpu_decode_owner = None;
        let mut sources = session.gpu_copy_sources().unwrap().unwrap();
        for base in &banks[..3] {
            sources.test_cpu_only_source(*base as usize + ROUTED[1] as usize);
        }
        (sources, crate::pager::owned_seed(session, 0).unwrap())
    };
    let owner = crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, capacity, seed)
        .unwrap()
        .unwrap();
    fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .gpu_decode_owner = Some(Arc::clone(&owner));
    fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_mut()
        .unwrap()
        .gpu_copy = Some(owner);
}

fn read_bytes(fixture: &Fixture, buffer: &dyn Buffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.len_bytes()];
    fixture.backend.download(buffer, &mut bytes).unwrap();
    bytes
}

fn prepared_head(fixture: &Fixture, current: &VkDecodePlan) -> VkDecodePlan {
    let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.reset_frontend();
    {
        let bindings = fixture.bindings();
        execute_static_next(
            &fixture.backend,
            current,
            &bindings,
            Some((&next, &bindings, &[1])),
        )
        .unwrap();
    }
    {
        let packet = next.prepared_prefix.lock().unwrap();
        let (head_end, head_layers, end, layers, self_service, owned) =
            packet.as_ref().unwrap().recorded_shape();
        assert_eq!((head_end, end, head_layers, layers), (2, 2, 1, 1));
        assert!(end < fixture.graph.ops.len() && owned && !self_service);
    }
    {
        let mut pager = fixture.backend.moe_pager().lock().unwrap();
        let session = pager.as_mut().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        session.discard_cpu_miss(fixture.roles[0], &[ROUTED[1]]);
    }
    fixture.reset_frontend();
    next
}

#[test]
#[ignore = "parent serial GPU only; same-seed legacy/merged HC outputs and full owner metadata differential"]
fn gpu_owned_stage_tail_merges_one_recovery_and_preserves_exact_metadata() {
    let (fixture, plan, out, expected) = stage_fixture(false);
    let (copy, tape) = {
        let mut pager = fixture.backend.moe_pager().lock().unwrap();
        let session = pager.as_mut().unwrap();
        let copy = session.gpu_decode_owner.clone().unwrap();
        copy.begin(&fixture.backend, session).unwrap();
        copy.finish_owned_forward();
        (copy, session.retained_tape())
    };
    // Restore only when all readers/ACKs drained and no forward owns the directory.
    // Same owner, physical addresses and generation: byte-for-byte, no normalization.
    let buffers = [
        copy.sources.clone(),
        copy.candidates.clone(),
        copy.overlay.clone(),
        copy.jobs.clone(),
        tape,
    ];
    let initial: Vec<_> = buffers
        .iter()
        .map(|b| read_bytes(&fixture, b.as_ref()))
        .collect();
    let mut reference = None;
    let mut dispatches = 0;
    for disable_merge in [true, false] {
        assert!(copy.owned_active() && !copy.owned_in_forward());
        for (buffer, bytes) in buffers.iter().zip(&initial) {
            fixture.backend.upload(buffer.as_ref(), bytes).unwrap();
        }
        fixture.reset_frontend();
        let trace = TraceGuard::begin(disable_merge, false, None);
        execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
        assert_eq!(fixture.read(fixture.buffer(out)), expected);
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
        let (ops, merged, standalone, submitted) = trace.counts();
        assert_eq!(submitted.len(), 1, "one actual recovery CB");
        assert_eq!(
            (ops, merged, standalone),
            if disable_merge { (0, 0, 1) } else { (2, 1, 0) }
        );
        let actual: Vec<_> = buffers
            .iter()
            .map(|b| read_bytes(&fixture, b.as_ref()))
            .collect();
        if disable_merge {
            reference = Some(actual);
            dispatches = submitted[0];
        } else {
            assert_eq!(
                &actual,
                reference.as_ref().unwrap(),
                "full owner directory/jobs/Tape equality"
            );
            assert_eq!(
                submitted[0],
                dispatches + 2,
                "HC and Scale appended to recovery"
            );
        }
    }
    fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .handoff_gpu_decode(&fixture.backend)
        .unwrap();
}

#[test]
#[ignore = "parent serial GPU only; prepared chunk ending before HC discards only proven boundary CB"]
fn gpu_owned_prepared_boundary_before_hc_discards_and_recovers_once() {
    let (fixture, current, out, expected) = stage_fixture(false);
    let next = prepared_head(&fixture, &current);
    let before = OWNED_PREPARED_STAGE_TEST_DISCARDS.with(|n| n.get());
    let trace = TraceGuard::begin(false, false, None);
    execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
    assert_eq!(
        OWNED_PREPARED_STAGE_TEST_DISCARDS.with(|n| n.get()) - before,
        1
    );
    assert_eq!(trace.counts().0, 2);
    assert_eq!(trace.counts().1, 1);
    assert_eq!(trace.counts().2, 0);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
    assert_eq!(fixture.read(fixture.buffer(out)), expected);
    assert!(next.prepared_prefix.lock().unwrap().is_none());
}

#[test]
#[ignore = "parent serial GPU only; invalid GPU generation guard must not run appended HC"]
fn gpu_owned_stage_tail_invalid_guard_never_runs_hc() {
    let (fixture, plan, out, _) = stage_fixture(false);
    let sentinel = vec![-83.0f32; NE];
    fixture
        .backend
        .upload(fixture.buffer(out), bytemuck::cast_slice(&sentinel))
        .unwrap();
    let trace = TraceGuard::begin(false, true, None);
    assert!(execute_static(&fixture.backend, &plan, &fixture.bindings()).is_err());
    assert_eq!(
        trace.counts().1,
        1,
        "rejected guard still encoded the terminal suffix"
    );
    assert_eq!(fixture.read(fixture.buffer(out)), sentinel);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
}

#[test]
#[ignore = "parent serial GPU only; terminal transfer stays on legacy nonempty tail, never dispatch-zero discard"]
fn gpu_owned_stage_tail_copy_falls_back_and_remains_submitted() {
    let (fixture, plan, out, expected) = stage_fixture(true);
    let trace = TraceGuard::begin(false, false, None);
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    assert_eq!(trace.counts().1, 0);
    assert_eq!(trace.counts().2, 1);
    assert_eq!(fixture.read(fixture.buffer(out)), expected);
    assert_eq!(fixture.read(fixture.buffer(fixture.outputs[1])), expected);
}

#[test]
#[ignore = "parent serial GPU only; Moe followed ONLY by Copy retains a nonempty resident tail"]
fn gpu_owned_stage_tail_only_copy_is_never_zero_dispatch_discarded() {
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    fixture.graph.ops.truncate(2);
    fixture.graph.push(Op::Copy {
        src: fixture.outputs[0],
        src_off: 0,
        dst: fixture.outputs[1],
        dst_off: 0,
        n: NE as u32,
    });
    fixture.preload();
    let plan = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    let expected = fixture.read(fixture.buffer(fixture.outputs[0]));
    install_cpu_head_owner(&fixture);
    fixture.reset_frontend();
    let trace = TraceGuard::begin(false, false, None);
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    assert_eq!((trace.counts().1, trace.counts().2), (0, 1));
    OWNED_STAGE_TAIL_TEST_STATE.with(|state| {
        let state = state.borrow();
        assert!(state
            .as_ref()
            .unwrap()
            .standalone_tail_dispatches
            .iter()
            .all(|&n| n > 0));
    });
    assert_eq!(fixture.read(fixture.buffer(fixture.outputs[1])), expected);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
}

#[test]
#[ignore = "parent serial GPU only; a later router/Moe forces the original guarded submission boundary"]
fn gpu_owned_stage_tail_does_not_cross_next_router_or_layer() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    fixture.preload();
    let plan = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    let expected = fixture.outputs();
    install_cpu_head_owner(&fixture);
    fixture.reset_frontend();
    let trace = TraceGuard::begin(false, false, None);
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    assert_eq!(trace.counts().1, 0);
    assert!(!trace.counts().3.is_empty(), "real L0 CPU recovery");
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
    assert_eq!(fixture.outputs(), expected);
}

#[test]
#[ignore = "parent serial GPU only; tail error/cancel/unwind drains prepared L0 and allows a new request"]
fn gpu_owned_stage_tail_error_cancel_unwind_never_resubmits_l0() {
    for failure in 0..3 {
        let (fixture, current, out, expected) = stage_fixture(false);
        let plan = prepared_head(&fixture, &current);
        let sentinel = vec![-91.0f32; NE];
        fixture
            .backend
            .upload(fixture.buffer(out), bytemuck::cast_slice(&sentinel))
            .unwrap();
        let trace = TraceGuard::begin(false, false, Some((1, failure)));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute_static(&fixture.backend, &plan, &fixture.bindings())
        }));
        match failure {
            0 => assert!(result
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("stage-tail lowering error")),
            1 => assert!(matches!(result.unwrap(), Err(Error::Aborted))),
            _ => assert!(result.is_err()),
        }
        assert_eq!(
            trace.counts().1,
            0,
            "unsubmitted recovery cannot be replayed"
        );
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
        assert_eq!(fixture.read(fixture.buffer(out)), sentinel);
        assert!(!fixture.backend.runtime_phase.is_poisoned());
        assert!(plan.prepared_prefix.lock().unwrap().is_none());
        drop(trace);
        fixture
            .backend
            .moe_pager()
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .handoff_gpu_decode(&fixture.backend)
            .unwrap();
        fixture.reset_frontend();
        execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
        assert_eq!(fixture.read(fixture.buffer(out)), expected);
    }
}

struct InheritedTraceGuard;

impl InheritedTraceGuard {
    fn begin(retirement_failure: Option<u8>) -> Self {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            assert!(state
                .replace(Some(StagedActivationTestState {
                    events: Vec::new(),
                    setup_failure: None,
                    retirement_failure,
                }))
                .is_none());
        });
        ROLLING_MAPPED_TEST_STATE.with(|state| {
            assert!(state
                .replace(Some(RollingMappedTestState {
                    decisions: Vec::new(),
                    admission_proofs: 0,
                    fail_after_layers: None,
                    confirm_recovery: false,
                }))
                .is_none());
        });
        PREPARED_MAPPED_TEST_STATE.with(|state| {
            assert!(state
                .replace(Some(PreparedMappedTestState {
                    decisions: Vec::new(),
                    fail_after_layers: None,
                }))
                .is_none());
        });
        Self
    }

    fn events(&self) -> Vec<StagedActivationTestEvent> {
        STAGED_ACTIVATION_TEST_STATE.with(|state| state.borrow().as_ref().unwrap().events.clone())
    }

    fn allow_retry(&self) {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            state.borrow_mut().as_mut().unwrap().retirement_failure = None;
        });
    }
}

impl Drop for InheritedTraceGuard {
    fn drop(&mut self) {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            state.take();
        });
        ROLLING_MAPPED_TEST_STATE.with(|state| {
            state.take();
        });
        PREPARED_MAPPED_TEST_STATE.with(|state| {
            state.take();
        });
    }
}

#[test]
#[ignore = "parent serial GPU only; inherited-tail retirement restores mapped windows at one/two-layer cadence and drains failures"]
fn gpu_owned_staged_inherited_pending_retires_after_prefix_and_restores_zero_recovery_window() {
    use infr_core::backend::ExecutionStage;
    for cadence in [1, 2] {
        for failure in [None, Some(0), Some(1), Some(2)] {
            let (fixture, head, out, expected) = stage_fixture_with_cadence(false, 1, cadence);
            let mapped = mapped_stage_moe(
                &fixture,
                head.graph
                    .ops
                    .iter()
                    .find(|op| matches!(op, Op::MoeFfn { .. }))
                    .unwrap(),
            );
            let mut tail_graph = head.graph.clone();
            tail_graph.ops.clear();
            let input = tail_graph.input(TensorDesc::new([NE], DType::F32));
            let staged_input = tail_graph.internal(TensorDesc::new([NE], DType::F32));
            let output = tail_graph.output(TensorDesc::new([NE], DType::F32));
            // A supported nonempty preamble is required by prefix_end. Keep its result
            // in retained scratch, so an accepted prefix cannot touch the final sentinel.
            tail_graph.push(Op::Scale {
                x: input,
                dst: staged_input,
                s: 1.0,
                n: NE as u32,
            });
            tail_graph.ops.extend(vec![mapped; 6]);
            tail_graph.push(Op::Add {
                a: out,
                b: staged_input,
                dst: output,
                n: NE as u32,
            });
            let mut tail = VkDecodePlan::new_cpu(&tail_graph, false, true);
            tail.phase_requests =
                Some(stages::compiled_workspace_requests(&fixture.backend, &tail.graph).unwrap());
            let input_buffer = fixture.backend.alloc(NE * 4, BufferUsage::Staging).unwrap();
            let output_buffer = fixture
                .backend
                .alloc(NE * 4, BufferUsage::Readback)
                .unwrap();
            fixture
                .backend
                .upload(input_buffer.as_ref(), bytemuck::cast_slice(&vec![3f32; NE]))
                .unwrap();
            let head_bindings = fixture.bindings();
            let mut tail_bindings = fixture.bindings();
            tail_bindings
                .bind(input, input_buffer.as_ref())
                .bind(output, output_buffer.as_ref());
            // Repeated banks still need six REAL job/predicate directories. Do not reuse
            // the head's three-ordinal owner while warming the longer execution window.
            install_cpu_head_owner_with_capacity(&fixture, 6);
            execute_static(&fixture.backend, &tail, &tail_bindings).unwrap();
            // This fixture accepts one bounded prefix, then walks a mapped rolling
            // suffix in the SAME initial window, not recovery re-admission. It does
            // not exercise the separate prepared-parts submission path.
            install_cpu_head_owner(&fixture);
            fixture.reset_frontend();
            let sentinel = vec![-7f32; NE];
            fixture
                .backend
                .upload(output_buffer.as_ref(), bytemuck::cast_slice(&sentinel))
                .unwrap();
            let inputs = [input];
            let stages = [
                ExecutionStage::new(&head, &head_bindings),
                ExecutionStage::new(&tail, &tail_bindings).with_independent_host_inputs(&inputs),
            ];
            let _trace = TraceGuard::begin(false, false, None);
            let inherited = InheritedTraceGuard::begin(failure);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                fixture.backend.execute_staged(
                    &stages,
                    &mut |index| {
                        staged_activation_test_note(StagedActivationTestEvent::Ready(index));
                        Ok(())
                    },
                    None,
                )
            }));
            let events = inherited.events();
            let position = |expected| {
                events
                    .iter()
                    .position(|event| *event == expected)
                    .unwrap_or_else(|| panic!("missing {expected:?}: {events:?}"))
            };
            let handoff = position(StagedActivationTestEvent::InheritedPending(1));
            let retiring = position(StagedActivationTestEvent::InheritedRetiring(1));
            let retired = position(StagedActivationTestEvent::InheritedRetired(1));
            let submitted = events[..retiring]
                .iter()
                .rposition(|event| *event == StagedActivationTestEvent::Submitted)
                .expect("actual next prefix submitted before old retirement");
            assert!(handoff < position(StagedActivationTestEvent::Ready(1)));
            assert!(
                handoff < submitted && submitted < retiring && retiring < retired,
                "{events:?}"
            );
            assert_eq!(
                OWNED_RECOVERY_TEST_COUNTS.with(|state| state.get().unwrap()[0]),
                1,
                "only the head may recover, cadence={cadence}: {events:?}"
            );
            let (owner, generation) = drained_owner_epoch(&fixture);
            assert_eq!(owner.owned_active(), failure.is_none());
            match failure {
                None => {
                    result.unwrap().unwrap();
                    assert!(
                        events[retired + 1..].contains(
                            &StagedActivationTestEvent::WindowAdmission {
                                rolling: true,
                                prepared: false,
                            }
                        ),
                        "restored rolling admission after old retirement: {events:?}"
                    );
                    ROLLING_MAPPED_TEST_STATE.with(|state| {
                        let state = state.borrow();
                        let state = state.as_ref().unwrap();
                        assert!(state.decisions.iter().any(|d| d.eligible && d.over_budget && !d.after_recovery),
                            "restored real rolling bypass without tail recovery, cadence={cadence}: {:?}", state.decisions);
                    });
                    PREPARED_MAPPED_TEST_STATE.with(|state| {
                        assert!(
                            state
                                .borrow()
                                .as_ref()
                                .unwrap()
                                .decisions
                                .is_empty(),
                            "bounded-prefix fixture must not claim prepared-parts coverage, cadence={cadence}"
                        );
                    });
                    assert_eq!(
                        fixture.read(output_buffer.as_ref()),
                        expected.iter().map(|x| x + 3.0).collect::<Vec<_>>()
                    );
                }
                Some(0) => assert!(result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("inherited stage retirement error")),
                Some(1) => assert!(matches!(result.unwrap(), Err(Error::Aborted))),
                _ => assert!(result.is_err()),
            }
            if failure.is_some() {
                assert!(
                    !events[retired + 1..].iter().any(|event| matches!(
                        event,
                        StagedActivationTestEvent::WindowAdmission { .. }
                    )),
                    "{events:?}"
                );
                assert_eq!(fixture.read(output_buffer.as_ref()), sentinel);
            }
            fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
            assert_eq!(fixture.read(fixture.buffer(out)), expected);
            assert!(!fixture.backend.runtime_phase.is_poisoned());
            // The actual next prefix also drains on retirement error/cancel/panic. Retry
            // through normal same-owner recovery, without a test handoff or replay of L0.
            inherited.allow_retry();
            fixture.reset_frontend();
            fixture
                .backend
                .execute_staged(&stages, &mut |_| Ok(()), None)
                .unwrap();
            fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
            assert_eq!(
                fixture.read(output_buffer.as_ref()),
                expected.iter().map(|x| x + 3.0).collect::<Vec<_>>()
            );
            let (reused, reused_generation) = drained_owner_epoch(&fixture);
            assert!(Arc::ptr_eq(&owner, &reused));
            assert!(reused.owned_active());
            assert_eq!(generation + u64::from(failure.is_some()), reused_generation);
            fixture
                .backend
                .moe_pager()
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .handoff_gpu_decode(&fixture.backend)
                .unwrap();
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; independent host input overlaps a retained GPU tail without replay"]
fn gpu_owned_staged_tail_independent_input_and_failure_preserve_forward_lifetimes() {
    use infr_core::backend::ExecutionStage;
    for outcome in 0..6 {
        let (fixture, head, out, expected) = stage_fixture(false);
        let mut tail_graph = head.graph.clone();
        tail_graph.ops.clear();
        // Keep a real paged-MoE Decode stage, rather than an auxiliary Add-only graph.
        tail_graph.push(head.graph.ops[1].clone());
        let input = tail_graph.input(TensorDesc::new([NE], DType::F32));
        let output = tail_graph.output(TensorDesc::new([NE], DType::F32));
        tail_graph.push(Op::Add {
            a: out,
            b: input,
            dst: output,
            n: NE as u32,
        });
        let tail = VkDecodePlan::new_cpu(&tail_graph, false, true);
        let input_buffer = fixture.backend.alloc(NE * 4, BufferUsage::Staging).unwrap();
        let output_buffer = fixture
            .backend
            .alloc(NE * 4, BufferUsage::Activations)
            .unwrap();
        let head_bindings = fixture.bindings();
        let mut tail_bindings = fixture.bindings();
        tail_bindings.bind(input, input_buffer.as_ref());
        tail_bindings.bind(output, output_buffer.as_ref());
        let inputs = [input];
        let main = ExecutionStage::new(&tail, &tail_bindings);
        let main = if outcome == 5 {
            main
        } else {
            main.with_independent_host_inputs(&inputs)
        };
        let stages = [ExecutionStage::new(&head, &head_bindings), main];
        let before = STAGED_TAIL_DEFERRED.with(|count| count.get());
        let pending_before = STAGED_TAIL_PENDING_HANDOFF.with(|count| count.get());
        let mut saw_callback = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fixture.backend.execute_staged(
                &stages,
                &mut |index| {
                    if index == 1 {
                        saw_callback = true;
                        let expected = usize::from(outcome != 5);
                        assert_eq!(
                            STAGED_TAIL_DEFERRED.with(|count| count.get()),
                            before + expected
                        );
                        assert_eq!(
                            STAGED_TAIL_PENDING_HANDOFF.with(|count| count.get()),
                            pending_before + expected
                        );
                        match outcome {
                            1 => std::thread::sleep(std::time::Duration::from_millis(2)),
                            2 => return Err(be("independent input failed")),
                            3 => return Err(Error::Aborted),
                            4 => panic!("independent input panic"),
                            _ => (),
                        }
                        fixture
                            .backend
                            .upload(input_buffer.as_ref(), bytemuck::cast_slice(&vec![3f32; NE]))?;
                    }
                    Ok(())
                },
                None,
            )
        }));
        assert!(saw_callback, "stage 0 failed before callback: {result:?}");
        let (owner, generation) = drained_owner_epoch(&fixture);
        let failed = matches!(outcome, 2..=4);
        assert_eq!(owner.owned_active(), !failed);
        match outcome {
            0 | 1 | 5 => {
                result.unwrap().unwrap();
                assert_eq!(
                    fixture.read(output_buffer.as_ref()),
                    expected.iter().map(|x| x + 3.0).collect::<Vec<_>>()
                );
            }
            2 => assert!(result
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("independent input failed")),
            3 => assert!(matches!(result.unwrap(), Err(Error::Aborted))),
            _ => assert!(result.is_err()),
        }
        // Even when stage 1 never starts, the already submitted L0 must finish exactly once.
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
        assert_eq!(fixture.read(fixture.buffer(out)), expected);
        assert!(!fixture.backend.runtime_phase.is_poisoned());
        fixture.reset_frontend();
        execute_static(&fixture.backend, &head, &head_bindings).unwrap();
        assert_eq!(fixture.read(fixture.buffer(out)), expected);
        let (reused, reused_generation) = drained_owner_epoch(&fixture);
        assert!(Arc::ptr_eq(&owner, &reused));
        assert!(reused.owned_active());
        assert_eq!(
            generation + u64::from(failed),
            reused_generation,
            "normal reuse must not reseed; errors use the backend's one conservative recovery"
        );
        fixture
            .backend
            .moe_pager()
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .handoff_gpu_decode(&fixture.backend)
            .unwrap();
    }
}

#[test]
#[ignore = "parent serial GPU only; physical alias and CPU metadata-read admission proof"]
fn gpu_owned_staged_independent_input_proof_rejects_aliases_and_preserves_segmented_kv() {
    use infr_core::backend::{ExecutionStage, SegmentedKvSpec};
    let fixture = Fixture::new_with_copy_policy_and_kv(1, true, Some(true), 16384, 4096);
    let be_ = &fixture.backend;
    let mut head = Graph::new();
    let x = head.input(TensorDesc::new([32], DType::F32));
    let y = head.output(TensorDesc::new([32], DType::F32));
    let unused = head.input(TensorDesc::new([1], DType::F32));
    head.push(Op::Scale {
        x,
        dst: y,
        s: 2.0,
        n: 32,
    });
    let head_plan = VkDecodePlan::new_cpu(&head, false, true);
    let mut tail = head.clone();
    let input = tail.input(TensorDesc::new([32], DType::F32));
    let mut tail_plan = VkDecodePlan::new_cpu(&tail, false, true);
    let xb = be_.alloc(128, BufferUsage::Staging).unwrap();
    let yb = be_.alloc(128, BufferUsage::Staging).unwrap();
    let ib = be_.alloc(128, BufferUsage::Staging).unwrap();
    let short = be_.alloc(4, BufferUsage::Staging).unwrap();
    let unknown = CpuBuffer(4);
    let mut head_bindings = Bindings::new();
    head_bindings
        .bind(x, xb.as_ref())
        .bind(y, yb.as_ref())
        .bind(unused, &unknown);
    let mut tail_bindings = Bindings::new();
    tail_bindings.bind(input, ib.as_ref()).bind(y, yb.as_ref());
    let scratch: ScratchSet = (0..head.tensors.len()).map(|_| None).collect();
    let pool = ScratchPool::default();
    let prove = |ids: Option<&[TensorId]>,
                 bindings: &Bindings,
                 head_bindings: &Bindings,
                 plan: &VkDecodePlan,
                 scratch: &ScratchSet,
                 pool: &ScratchPool| {
        let mut stage = ExecutionStage::new(plan, bindings);
        stage.independent_host_inputs = ids;
        stages::independent_stage_inputs(
            &stage,
            plan,
            &ExecutionStage::new(&head_plan, head_bindings),
            &head_plan,
            scratch,
            pool,
        )
    };
    assert!(prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &scratch,
        &pool
    ));
    for ids in [
        None,
        Some([].as_slice()),
        Some([input, input].as_slice()),
        Some([y].as_slice()),
        Some([TensorId(u32::MAX)].as_slice()),
    ] {
        assert!(!prove(
            ids,
            &tail_bindings,
            &head_bindings,
            &tail_plan,
            &scratch,
            &pool
        ));
    }
    tail_bindings.bind(input, short.as_ref());
    assert!(!prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &scratch,
        &pool
    ));
    tail_bindings.bind(input, xb.as_ref());
    assert!(!prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &scratch,
        &pool
    ));
    tail_bindings.bind(input, ib.as_ref());
    // A backend CPU position read is unsafe even when the callback's write is isolated.
    tail_plan.setup_analysis.rope_ids.push(y);
    assert!(!prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &scratch,
        &pool
    ));
    tail_plan.setup_analysis.rope_ids.clear();
    head_bindings.bind_rows(x, vec![xb.as_ref(), ib.as_ref()]);
    assert!(!prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &scratch,
        &pool
    ));
    head_bindings.bind(x, xb.as_ref());
    let alias = || {
        Box::new(unsafe { crate::as_vk_buf(ib.as_ref()).unwrap().recording_alias() })
            as Box<dyn Buffer>
    };
    let mut alias_scratch: ScratchSet = (0..head.tensors.len()).map(|_| None).collect();
    alias_scratch[x.0 as usize] = Some(alias());
    assert!(!prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &alias_scratch,
        &pool
    ));
    let mut alias_pool = ScratchPool::default();
    alias_pool
        .buffers
        .insert(("host_input_alias", 128), alias());
    assert!(!prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &scratch,
        &alias_pool
    ));
    let kv = be_
        .alloc_segmented_kv(SegmentedKvSpec {
            logical_bytes: 8192,
            segment_bytes: 4096,
            segment_elements: 1024,
            max_segments: 2,
        })
        .unwrap()
        .unwrap();
    be_.ensure_segmented_kv(kv.as_ref(), 1).unwrap();
    head_bindings.bind(x, kv.as_ref());
    assert!(prove(
        Some(&[input]),
        &tail_bindings,
        &head_bindings,
        &tail_plan,
        &scratch,
        &pool
    ));
}

#[test]
#[ignore = "parent serial GPU only; independent two-lane input and unequal sequence positions"]
fn gpu_owned_staged_tail_two_lanes_preserve_rows_and_independent_input() {
    use infr_core::backend::ExecutionStage;
    let (fixture, head, out, expected) = stage_fixture_rows(false, 2);
    let mut tail_graph = head.graph.clone();
    tail_graph.ops.clear();
    tail_graph.push(head.graph.ops[1].clone());
    let input = tail_graph.input(TensorDesc::new([2, NE], DType::F32));
    let output = tail_graph.output(TensorDesc::new([2, NE], DType::F32));
    tail_graph.push(Op::Add {
        a: out,
        b: input,
        dst: output,
        n: (2 * NE) as u32,
    });
    let tail = VkDecodePlan::new_cpu(&tail_graph, false, true);
    let ib = fixture
        .backend
        .alloc(2 * NE * 4, BufferUsage::Staging)
        .unwrap();
    let ob = fixture
        .backend
        .alloc(2 * NE * 4, BufferUsage::Readback)
        .unwrap();
    let hb = fixture.bindings();
    let mut tb = fixture.bindings();
    tb.bind(input, ib.as_ref()).bind(output, ob.as_ref());
    let input_ids = [input];
    let stages = [
        ExecutionStage::new(&head, &hb),
        ExecutionStage::new(&tail, &tb).with_independent_host_inputs(&input_ids),
    ];
    let values: Vec<f32> = [vec![3f32; NE], vec![7f32; NE]].concat();
    let before = STAGED_TAIL_DEFERRED.with(|count| count.get());
    let pending_before = STAGED_TAIL_PENDING_HANDOFF.with(|count| count.get());
    fixture
        .backend
        .execute_staged(
            &stages,
            &mut |index| {
                if index == 1 {
                    assert_eq!(STAGED_TAIL_DEFERRED.with(|count| count.get()), before + 1);
                    assert_eq!(
                        STAGED_TAIL_PENDING_HANDOFF.with(|count| count.get()),
                        pending_before + 1
                    );
                    fixture
                        .backend
                        .upload(ib.as_ref(), bytemuck::cast_slice(&values))?;
                }
                Ok(())
            },
            None,
        )
        .unwrap();
    let mut actual = vec![0f32; 2 * NE];
    fixture
        .backend
        .download(ob.as_ref(), bytemuck::cast_slice_mut(&mut actual))
        .unwrap();
    assert_eq!(
        actual,
        expected
            .iter()
            .zip(&values)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>()
    );
    fixture
        .backend
        .download(
            fixture.buffer(fixture.frontend),
            bytemuck::cast_slice_mut(&mut actual),
        )
        .unwrap();
    assert_eq!(actual, vec![1f32; 2 * NE]);
    fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .handoff_gpu_decode(&fixture.backend)
        .unwrap();
}
