//! Same-ACK mapped prepared runs may queue without blocking on the cadence fence.
use super::*;

#[test]
fn prepared_mapped_run_requires_owned_scope_and_no_unconfirmed_host_notice() {
    for mask in 0..16 {
        let entries = notice_entries(&std::array::from_fn::<_, 4, _>(|i| mask & (1 << i) != 0));
        for same_packet in [false, true] {
            for confirmed in 0..=4 {
                for submitted in 0..=4 {
                    for end in 0..=5 {
                        let expected = same_packet
                            && confirmed <= submitted
                            && submitted < end
                            && end <= entries.len()
                            && entries
                                .iter()
                                .enumerate()
                                .all(|(i, entry)| i < confirmed || i >= end || !entry.host_notice);
                        assert_eq!(
                            prepared_mapped_run_eligible(
                                same_packet, &entries, confirmed, submitted, end,
                            ),
                            expected,
                            "mask={mask} scope={same_packet} cursor={confirmed} submitted={submitted} end={end}"
                        );
                    }
                }
            }
        }
    }
    let entries = notice_entries(&[false, true, false]);
    assert!(!prepared_mapped_run_eligible(true, &entries, 1, 2, 3));
    assert!(prepared_mapped_run_eligible(true, &entries, 2, 2, 3));
}

struct TraceGuard;

impl TraceGuard {
    fn begin(fail_after_layers: Option<(usize, bool)>) -> Self {
        PREPARED_MAPPED_TEST_STATE.with(|state| {
            assert!(state
                .replace(Some(PreparedMappedTestState {
                    decisions: Vec::new(),
                    fail_after_layers,
                }))
                .is_none());
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| assert!(state.replace(Some([0; 3])).is_none()));
        Self
    }

    fn decisions(&self) -> Vec<PreparedMappedTestDecision> {
        PREPARED_MAPPED_TEST_STATE.with(|state| state.borrow().as_ref().unwrap().decisions.clone())
    }

    fn recovery_count(&self) -> usize {
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.get().unwrap()[0])
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        PREPARED_MAPPED_TEST_STATE.with(|state| {
            state.take();
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.set(None));
    }
}

pub(super) fn install_mixed_owner(fixture: &Fixture) {
    let capacity = fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_ref()
        .unwrap()
        .capacity;
    let (sources, seed) = {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        let previous = session.gpu_decode_owner.clone().expect("warm owner");
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        let banks = previous.role_params(session, &fixture.roles[1]).unwrap();
        session.gpu_decode_owner = None;
        let mut sources = session.gpu_copy_sources().unwrap().unwrap();
        for base in &banks[..3] {
            sources.test_cpu_only_source(*base as usize + ROUTED[1] as usize);
        }
        (sources, crate::pager::owned_seed(session, 0).unwrap())
    };
    let copy = crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, capacity, seed)
        .unwrap()
        .unwrap();
    assert!(!copy.owned_self_service());
    fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .gpu_decode_owner = Some(Arc::clone(&copy));
    fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_mut()
        .unwrap()
        .gpu_copy = Some(copy);
}

fn prepared_case(host_boundary: bool) -> (Fixture, VkDecodePlan, Vec<(TensorId, Vec<f32>)>) {
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let original = fixture.graph.ops.clone();
    // Same registered mapped bank can appear repeatedly; do not assume layer/bank uniqueness.
    // The second registered bank remains CPU-sensitive even when absent from this graph.
    fixture.graph.ops.clear();
    for layer in [0, usize::from(host_boundary), 0] {
        fixture
            .graph
            .ops
            .extend_from_slice(&original[layer * 2..layer * 2 + 2]);
    }
    fixture.preload();
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    {
        let bindings = fixture.bindings();
        execute_static(&fixture.backend, &current, &bindings).unwrap();
    }
    let expected = fixture.outputs[..if host_boundary { 2 } else { 1 }]
        .iter()
        .map(|&id| {
            let values = fixture.read(fixture.buffer(id));
            assert!(values.iter().all(|x| x.is_finite() && *x > 0.0));
            (id, values)
        })
        .collect();
    install_mixed_owner(&fixture);
    // This regression deliberately needs the full queued suffix, not a shorter candidate.
    let _legacy_horizon = command_prefix::source_horizon_scope(&fixture.backend, false);
    let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.reset_frontend();
    {
        let bindings = fixture.bindings();
        execute_static_next(
            &fixture.backend,
            &current,
            &bindings,
            Some((&next, &bindings, &[1])),
        )
        .unwrap();
    }
    let (head_end, head_layers, last_end, layers, self_service, owned) = next
        .prepared_prefix
        .lock()
        .unwrap()
        .as_ref()
        .expect("actual three-part prepared packet")
        .recorded_shape();
    assert_eq!((head_layers, layers), (1, 3));
    assert!(head_end < last_end);
    assert_eq!(last_end, fixture.graph.ops.len());
    assert!(owned && !self_service);
    fixture.reset_frontend();
    (fixture, next, expected)
}

fn assert_complete(fixture: &Fixture, expected: &[(TensorId, Vec<f32>)]) {
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 3.0);
    for (id, values) in expected {
        assert_eq!(fixture.read(fixture.buffer(*id)), *values);
    }
    let mut guard = fixture.backend.moe_pager().lock().unwrap();
    let session = guard.as_mut().unwrap();
    session.handoff_gpu_decode(&fixture.backend).unwrap();
    assert!(!session.gpu_decode_in_forward());
}

#[test]
#[ignore = "parent serial GPU only; three pre-recorded mapped parts bypass cadence without re-recording"]
fn gpu_prepared_mapped_run_queues_ready_parts_and_preserves_outputs() {
    let (fixture, next, expected) = prepared_case(false);
    let trace = TraceGuard::begin(None);
    execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
    let decisions = trace.decisions();
    assert_eq!(decisions.len(), 2, "{decisions:?}");
    for (i, decision) in decisions.iter().enumerate() {
        assert_eq!((decision.first, decision.end), (i + 1, i + 2));
        assert!(decision.eligible && decision.over_budget, "{decisions:?}");
    }
    assert_eq!(trace.recovery_count(), 0);
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_complete(&fixture, &expected);
}

#[test]
#[ignore = "parent serial GPU only; CPU-sensitive part and actual miss retain legacy cadence/recovery"]
fn gpu_prepared_mapped_run_does_not_queue_past_unconfirmed_cpu_boundary() {
    for missing in [false, true] {
        let (fixture, next, expected) = prepared_case(true);
        if missing {
            let mut guard = fixture.backend.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            session.handoff_gpu_decode(&fixture.backend).unwrap();
            session.discard_cpu_miss(fixture.roles[1], &[ROUTED[1]]);
        }
        let trace = TraceGuard::begin(None);
        execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
        let decisions = trace.decisions();
        assert!(!decisions.is_empty(), "actual prepared CPU boundary");
        assert_eq!((decisions[0].first, decisions[0].end), (1, 2));
        assert!(!decisions[0].eligible && decisions[0].over_budget);
        for decision in &decisions {
            if decision.eligible {
                assert_eq!((decision.first, decision.end), (2, 3));
                assert!(decision.confirmed >= 2, "{decisions:?}");
            }
        }
        if missing {
            assert!(decisions.iter().all(|decision| !decision.eligible));
            assert!(trace.recovery_count() > 0, "actual CPU miss recovery");
        } else {
            assert_eq!(trace.recovery_count(), 0);
        }
        assert_complete(&fixture, &expected);
    }
}

#[test]
#[ignore = "parent serial GPU only; cancellation/error after safe enqueue drain and never replay L0"]
fn gpu_prepared_mapped_run_cancel_and_error_drain_submitted_parts() {
    for abort in [false, true] {
        let (fixture, next, expected) = prepared_case(false);
        let trace = TraceGuard::begin(Some((2, abort)));
        let result = execute_static(&fixture.backend, &next, &fixture.bindings());
        if abort {
            assert!(matches!(result, Err(Error::Aborted)));
        } else {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("injected prepared mapped-run submission error"));
        }
        let decisions = trace.decisions();
        assert_eq!(decisions.len(), 1, "{decisions:?}");
        assert!(decisions[0].eligible && decisions[0].over_budget);
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
        assert_eq!(
            fixture.read(fixture.buffer(fixture.outputs[0])),
            expected[0].1
        );
        drop(trace);
        // A new request can run normally after all error-path readers have drained.
        fixture.reset_frontend();
        execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
        assert_complete(&fixture, &expected);
    }
}
