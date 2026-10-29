//! Normal (not prepared) rolling-window admission, queue bounds and recovery re-certification.
use super::*;

#[test]
fn rolling_admission_lazy_skips_prepared_and_under_budget_paths() {
    let mut cache = RollingMappedAdmissionCache::default();
    // Complete prepared walks never reach an ordinary rolling boundary. Neither
    // an in-budget submit nor an ineligible host boundary may request the proof.
    assert!(cache.result.is_none());
    for _ in 0..3 {
        assert!(!cache.check_if(false, || panic!("resource scan"), || panic!("proof")));
    }
    assert!(cache.result.is_none());
}

#[test]
fn rolling_admission_caches_failures_and_invalidates_resource_changes() {
    let mut cache = RollingMappedAdmissionCache::default();
    let mut key = RollingMappedAdmissionKey {
        resource_epoch: 1,
        pool_buffers: 2,
        needs_compaction: false,
    };
    let proofs = std::cell::Cell::new(0);
    for _ in 0..4 {
        assert!(!cache.check_if(
            true,
            || key,
            || {
                proofs.set(proofs.get() + 1);
                false
            }
        ));
    }
    assert_eq!(proofs.get(), 1);
    key.resource_epoch += 1;
    assert!(cache.check_if(true, || key, || true));
    key.pool_buffers += 1;
    assert!(cache.check_if(true, || key, || true));
    // Same-count capacity replacement dirties the pool and invalidates success.
    key.needs_compaction = true;
    assert!(!cache.check_if(true, || key, || false));
    assert!(!cache.check_if(true, || key, || panic!("repeated failed proof")));
    key.needs_compaction = false;
    assert!(cache.check_if(true, || key, || true));
}

#[test]
fn rolling_admission_recovery_requires_scope_then_recertifies_new_epoch() {
    let mut cache = RollingMappedAdmissionCache::default();
    let mut key = RollingMappedAdmissionKey {
        resource_epoch: 1,
        pool_buffers: 2,
        needs_compaction: false,
    };
    assert!(cache.check_if(true, || key, || true));
    // CPU work/unconfirmed notices clear eligibility, regardless of cached success.
    assert!(!cache.check_if(false, || panic!("unconfirmed"), || panic!("unconfirmed")));
    // Real retirement + ACK restores provenance without rescanning stable resources.
    assert!(cache.check_if(true, || key, || panic!("unchanged resources")));
    key.resource_epoch += 1;
    let proved = std::cell::Cell::new(false);
    assert!(cache.check_if(
        true,
        || key,
        || {
            proved.set(true);
            true
        }
    ));
    assert!(proved.get());
}

#[test]
fn rolling_mapped_run_bounds_pending_by_admitted_window_and_confirmed_notices() {
    let entries = notice_entries(&[false, false, false]);
    assert!(rolling_mapped_run_eligible(true, &entries, 0, 2, 3, 3));
    for (scope, confirmed, submitted, capacity, pending) in [
        (false, 0, 2, 3, 3),
        (true, 0, 2, 2, 3),
        (true, 0, 2, 3, 4),
        (true, 0, 2, 3, 0),
        (true, 3, 2, 3, 3),
        (true, 0, 3, 3, 3),
    ] {
        assert!(!rolling_mapped_run_eligible(
            scope, &entries, confirmed, submitted, capacity, pending,
        ));
    }
    let entries = notice_entries(&[false, true, false]);
    assert!(!rolling_mapped_run_eligible(true, &entries, 1, 2, 3, 2));
    assert!(rolling_mapped_run_eligible(true, &entries, 2, 2, 3, 2));
    let mut stream = PagedStream::default();
    assert!(rolling_mapped_stream_clear(&stream));
    stream.cpu_transfer_wait = 1;
    assert!(!rolling_mapped_stream_clear(&stream));
    stream.cpu_transfer_wait = 0;
    stream.guarded_retire_after_submit = Some(1);
    assert!(!rolling_mapped_stream_clear(&stream));
    stream.guarded_retire_after_submit = None;
    stream.resume_ids = Some(vec![0]);
    assert!(!rolling_mapped_stream_clear(&stream));
}

struct TraceGuard;

impl TraceGuard {
    fn begin(fail_after_layers: Option<(usize, bool)>, confirm_recovery: bool) -> Self {
        ROLLING_MAPPED_TEST_STATE.with(|state| {
            assert!(state
                .replace(Some(RollingMappedTestState {
                    decisions: Vec::new(),
                    admission_proofs: 0,
                    fail_after_layers,
                    confirm_recovery,
                }))
                .is_none());
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| assert!(state.replace(Some([0; 3])).is_none()));
        Self
    }

    fn decisions(&self) -> Vec<RollingMappedTestDecision> {
        ROLLING_MAPPED_TEST_STATE.with(|state| state.borrow().as_ref().unwrap().decisions.clone())
    }

    fn recovery_count(&self) -> usize {
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.get().unwrap()[0])
    }

    fn admission_proofs(&self) -> usize {
        ROLLING_MAPPED_TEST_STATE.with(|state| state.borrow().as_ref().unwrap().admission_proofs)
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        ROLLING_MAPPED_TEST_STATE.with(|state| {
            state.take();
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.set(None));
    }
}

fn ordinary_case(pattern: &[usize]) -> (Fixture, VkDecodePlan, Vec<(TensorId, Vec<f32>)>) {
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let original = fixture.graph.ops.clone();
    fixture.graph.ops.clear();
    for &layer in pattern {
        fixture
            .graph
            .ops
            .extend_from_slice(&original[layer * 2..layer * 2 + 2]);
    }
    fixture.preload();
    let mut plan = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    // Same complete receipt that VkDecodePlan::boxed installs in production.
    plan.phase_requests =
        Some(stages::compiled_workspace_requests(&fixture.backend, &fixture.graph).unwrap());
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    let expected = fixture
        .outputs
        .iter()
        .enumerate()
        .filter(|(layer, _)| pattern.contains(layer))
        .map(|(_, &id)| {
                let values = fixture.read(fixture.buffer(id));
                assert!(values.iter().all(|value| value.is_finite() && *value > 0.0));
                (id, values)
        })
        .collect();
    prepared_mapped_run_tests::install_mixed_owner(&fixture);
    fixture.reset_frontend();
    assert!(plan.prepared_prefix.lock().unwrap().is_none());
    (fixture, plan, expected)
}

fn assert_complete(fixture: &Fixture, expected: &[(TensorId, Vec<f32>)], layers: usize) {
    fixture.assert_frontend(fixture.buffer(fixture.frontend), layers as f32);
    for (id, values) in expected {
        assert_eq!(fixture.read(fixture.buffer(*id)), *values);
    }
    let mut guard = fixture.backend.moe_pager().lock().unwrap();
    let session = guard.as_mut().unwrap();
    session.handoff_gpu_decode(&fixture.backend).unwrap();
    assert!(!session.gpu_decode_in_forward());
}

#[test]
#[ignore = "parent serial GPU only; true ordinary rolling mapped run and missing-receipt fallback"]
fn gpu_rolling_mapped_run_bypasses_wait_without_prepared_packets() {
    for admitted in [true, false] {
        let (fixture, mut plan, expected) = ordinary_case(&[0, 0, 0]);
        if !admitted {
            // Ordinary preflight still allocates correctly, but no complete cached proof
            // is supplied to the optimizer. It MUST preserve the old cadence.
            plan.phase_requests = None;
        }
        let trace = TraceGuard::begin(None, false);
        execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
        let decisions = trace.decisions();
        assert_eq!(decisions.len(), 2, "{decisions:?}");
        assert_eq!(
            decisions
                .iter()
                .filter(|d| d.eligible && d.over_budget)
                .count(),
            usize::from(admitted),
            "real bypass count: {decisions:?}"
        );
        for (i, decision) in decisions.iter().enumerate() {
            assert_eq!((decision.first, decision.end), (i, i + 1));
            assert_eq!(
                decision.eligible,
                admitted && decision.over_budget,
                "{decisions:?}"
            );
            assert!(decision.pending <= decision.end && decision.pending <= decision.limit);
        }
        assert_eq!(trace.recovery_count(), 0);
        assert_eq!(
            trace.admission_proofs(),
            1,
            "success and failure both cached"
        );
        eprintln!(
            "normal rolling admitted={admitted} bypasses={} decisions={decisions:?}",
            decisions
                .iter()
                .filter(|d| d.eligible && d.over_budget)
                .count()
        );
        assert!(plan.prepared_prefix.lock().unwrap().is_none());
        assert_complete(&fixture, &expected, 3);
    }
}

#[test]
#[ignore = "parent serial GPU only; complete prepared forward must never request rolling proof"]
fn gpu_rolling_admission_complete_prepared_forward_does_not_scan() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let (mut next, expected) = fixture.prepare(1, true);
    next.phase_requests =
        Some(stages::compiled_workspace_requests(&fixture.backend, &fixture.graph).unwrap());
    {
        let packet = next.prepared_prefix.lock().unwrap();
        let (_, _, end, _, self_service, owned) = packet.as_ref().unwrap().recorded_shape();
        assert_eq!(end, fixture.graph.ops.len());
        assert!(owned && self_service);
    }
    fixture.reset_frontend();
    let trace = TraceGuard::begin(None, false);
    execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
    assert_eq!(trace.admission_proofs(), 0);
    assert!(trace.decisions().is_empty());
    assert_close(&fixture.outputs(), &expected);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
}

#[test]
#[ignore = "parent serial GPU only; real CPU miss then guarded retirement and mapped-window re-certification"]
fn gpu_rolling_mapped_run_recertifies_after_cpu_recovery_and_ack() {
    let (fixture, plan, expected) = ordinary_case(&[0, 1, 0, 0, 0, 0]);
    {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        session.discard_cpu_miss(fixture.roles[1], &[ROUTED[1]]);
    }
    // Wait for the real recovery notice in this correctness test so the transition is
    // deterministic. Production retains nonblocking polling; no ACK/data is synthesized.
    let trace = TraceGuard::begin(None, true);
    execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
    let decisions = trace.decisions();
    let cpu_boundary = decisions
        .iter()
        .find(|d| !d.after_recovery && d.end == 2)
        .expect("normal rolling CPU boundary");
    assert!(
        !cpu_boundary.eligible && cpu_boundary.over_budget,
        "{decisions:?}"
    );
    for decision in decisions.iter().filter(|d| !d.after_recovery && d.end > 2) {
        assert!(!decision.eligible, "unconfirmed CPU suffix: {decisions:?}");
    }
    let bypasses = decisions
        .iter()
        .filter(|d| d.after_recovery && d.eligible && d.over_budget)
        .count();
    assert!(
        bypasses > 0,
        "mapped run must consume re-certification: {decisions:?}"
    );
    for decision in decisions.iter().filter(|d| d.after_recovery && d.eligible) {
        assert!(
            decision.confirmed >= 1,
            "recovery notice must be confirmed: {decisions:?}"
        );
        assert!(decision.pending <= decision.end && decision.pending <= decision.limit);
    }
    eprintln!("normal rolling post-recovery mapped bypasses={bypasses} decisions={decisions:?}");
    assert!(trace.recovery_count() > 0);
    assert!(plan.prepared_prefix.lock().unwrap().is_none());
    assert_complete(&fixture, &expected, 6);
}

#[test]
#[ignore = "parent serial GPU only; error/cancel after ordinary safe enqueue drain before a new request"]
fn gpu_rolling_mapped_run_cancel_error_drain_and_new_request_continues() {
    for abort in [false, true] {
        let (fixture, plan, expected) = ordinary_case(&[0, 0, 0]);
        let trace = TraceGuard::begin(Some((2, abort)), false);
        let result = execute_static(&fixture.backend, &plan, &fixture.bindings());
        if abort {
            assert!(matches!(result, Err(Error::Aborted)));
        } else {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("injected rolling mapped-run submission error"));
        }
        assert!(trace
            .decisions()
            .iter()
            .any(|d| d.end == 2 && d.eligible && d.over_budget));
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
        assert_eq!(
            fixture.read(fixture.buffer(fixture.outputs[0])),
            expected[0].1
        );
        assert!(plan.prepared_prefix.lock().unwrap().is_none());
        drop(trace);
        fixture.reset_frontend();
        execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
        assert_complete(&fixture, &expected, 3);
    }
}
