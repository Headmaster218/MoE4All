//! Real NEXT recording/activation plus submitted host-wait CPU faults, short horizon.
//! Child of host_ready_combined_tests; deliberately does not call Fixture::prepare,
//! whose legacy coverage overrides the source-certificate horizon.
use super::*;
use crate::adapter::resident_ahead_tests::promotion_isolation_tests::Trace;

fn run_prepared_combined(fault: Fault) {
    let fixture = Fixture::new_with_combined_policy(1, true, Some(true), 0, 0, true, 3, true);
    let backend = &fixture.backend;
    assert!(backend.cfg().device.decode_command_ahead);
    assert!(backend.cfg().device.decode_command_early_submit);
    assert!(backend.cfg().device.decode_command_resident);
    assert!(
        backend
            .cfg()
            .kernels
            .vulkan
            .decode_gpu_resident_copy_overlap
    );
    assert!(backend.cfg().kernels.vulkan.decode_gpu_resident_host_ready);
    assert!(
        backend
            .cfg()
            .kernels
            .vulkan
            .decode_gpu_resident_upload_worker
    );
    assert!(!backend.cfg().spec.mtp);
    let trace = Trace::begin(backend);
    let current = backend.compile(&fixture.graph).unwrap();
    let next = backend.compile(&fixture.graph).unwrap();
    let next_decode = next.as_any().downcast_ref::<VkDecodePlan>().unwrap();
    fixture.preload();
    backend
        .execute(current.as_ref(), &fixture.bindings())
        .unwrap();
    let expected = fixture.outputs.map(|id| fixture.read(fixture.buffer(id)));
    assert!(expected
        .iter()
        .flatten()
        .all(|value| value.is_finite() && *value > 0.0));

    // Install permanent non-imported RAM/CPU-only source certificates BEFORE any NEXT
    // snapshot. This owner must be retained unchanged through recording and activation.
    install_cpu_tail(&fixture);
    let owner = backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .gpu_decode_owner
        .clone()
        .unwrap();
    assert!(owner.is_owned() && !owner.owned_self_service());
    let recorded_before = trace.counts().record;
    fixture.reset_frontend();
    let bindings = fixture.bindings();
    backend
        .execute_with_lookahead(
            current.as_ref(),
            &bindings,
            Some((next.as_ref(), &bindings, &[1])),
        )
        .unwrap();
    assert!(
        trace.counts().record > recorded_before,
        "NEXT must execute real StageRecording, not just enable its flags"
    );
    fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
    assert_eq!(
        fixture.outputs.map(|id| fixture.read(fixture.buffer(id))),
        expected
    );
    {
        let packet = next_decode.prepared_prefix.lock().unwrap();
        let packet = packet.as_ref().expect("actual prepared NEXT packet");
        assert!(packet.resident_layers() > 0);
        // Natural production source horizon: the first CPU-service group ends at layer 1,
        // not the complete two-layer graph and not a forced legacy sixteen-layer window.
        assert_eq!(packet.resident_layers(), 1);
        assert!(packet.end < fixture.graph.ops.len());
        let (_, head_layers, _, layers, self_service, owned) = packet.recorded_shape();
        assert_eq!(
            (head_layers, layers, self_service, owned),
            (1, 1, false, true)
        );
    }
    let epoch = backend.shared.resource_epoch.load(Ordering::Acquire);
    {
        let mut guard = backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        assert!(Arc::ptr_eq(
            session.gpu_decode_owner.as_ref().unwrap(),
            &owner
        ));
        session.handoff_gpu_decode(backend).unwrap();
        session.discard_cpu_miss(fixture.roles[0], &[ROUTED[1]]);
        assert_eq!(
            session
                .routed_roles_resident_mask(&fixture.roles[0], &ROUTED)
                .unwrap()
                & 2,
            0
        );
        assert_eq!(
            session
                .routed_roles_resident_mask(&fixture.roles[1], &ROUTED)
                .unwrap(),
            HIT_MASK
        );
    }
    fixture.reset_frontend();
    for output in fixture.outputs {
        backend
            .upload(
                fixture.buffer(output),
                bytemuck::cast_slice(&[-777.0f32; NE]),
            )
            .unwrap();
    }
    assert_eq!(
        backend.shared.resource_epoch.load(Ordering::Acquire),
        epoch,
        "forcing the miss and resetting mapped inputs cannot invalidate the NEXT packet"
    );
    assert!(Arc::ptr_eq(
        backend
            .runtime_phase
            .lock()
            .unwrap()
            .resident_buffers
            .as_ref()
            .unwrap()
            .gpu_copy
            .as_ref()
            .unwrap(),
        &owner
    ));
    OWNED_HOST_READY_TEST_ARMS.with(|arms| arms.set(0));
    OWNED_HOST_READY_TEST_OUTCOME.with(|outcome| assert!(outcome.get().is_none()));
    let accepted_before = command_prefix::test_accepted_prefixes();
    let counts_before = trace.counts();
    let registration = Registration::new(&backend.shared, Arc::clone(&backend.unified_exec), fault);
    let result = backend.execute(next.as_ref(), &bindings);
    let proof = registration.0.evidence();
    drop(registration);
    let counts_after = trace.counts();
    eprintln!("prepared host-ready {fault:?}: before={counts_before:?} after={counts_after:?} proof={proof:?}");
    assert_eq!(
        command_prefix::test_accepted_prefixes(),
        accepted_before + 1,
        "must consume/accept the recorded current prefix, not silently run ordinary recording"
    );
    assert_eq!(counts_after.early_submit, counts_before.early_submit + 1,
        "existing early_submit observation is AFTER successful submit and early_prefix=Some(PendingSegment)");
    assert!(
        counts_after.overlap > counts_before.overlap,
        "accepted prepared prefix plus recovery must actually record the overlap path"
    );
    assert!(next_decode.prepared_prefix.lock().unwrap().is_none());
    assert!(proof.entered && proof.jobs > 1);
    assert!(proof.submitted && proof.wait_value != 0);
    assert!(proof.unsignaled_at_submit && proof.fence_pending_at_submit);
    assert_eq!(proof.copied_at_submit, 0);
    assert!(proof.pending_owned && proof.shared_gate_blocked);
    assert_eq!(proof.controller_cancelled, fault == Fault::SubmittedCancel);
    OWNED_HOST_READY_TEST_ARMS.with(|arms| assert_eq!(arms.get(), 1));
    if fault == Fault::None {
        result.unwrap();
        assert_eq!(proof.copied, proof.jobs);
        fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
        assert_eq!(
            fixture.outputs.map(|id| fixture.read(fixture.buffer(id))),
            expected
        );
    } else {
        assert!(result.is_err());
        assert_eq!(
            proof.copied, 1,
            "actual first memcpy completed, remaining jobs did not"
        );
        for output in fixture.outputs {
            assert_eq!(fixture.read(fixture.buffer(output)), vec![-777.0f32; NE]);
        }
        let guard = backend.moe_pager().lock().unwrap();
        let session = guard.as_ref().unwrap();
        assert!(!session.gpu_decode_in_forward());
        assert_eq!(
            session
                .routed_roles_resident_mask(&fixture.roles[0], &ROUTED)
                .unwrap()
                & 2,
            0,
            "failed partial payload cannot survive recovery as READY residency"
        );
    }
    // Same plan/backend/producer/source owner; existing failure signaling and Drop drains only.
    check_request(&fixture, next.as_ref(), &expected);
    assert_handoff(&fixture);
}

#[test]
#[ignore = "parent serial GPU only; accepted short-horizon NEXT/early Pending plus host-ready CPU fault/cancel"]
fn gpu_host_ready_prepared_short_horizon_fault_cancel_retry() {
    for fault in [Fault::None, Fault::AfterFirstError, Fault::SubmittedCancel] {
        run_prepared_combined(fault);
    }
}
