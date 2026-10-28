//! Complete early-prepared stages retain their useful fence, not a barrier-only tail fence.
use super::*;

#[test]
fn owned_prepared_stage_empty_requires_complete_submitted_stage_provenance() {
    assert!(owned_prepared_stage_empty_eligible(
        true, true, true, 1, 1, 1, 0
    ));
    for (early, complete, no_parts, entries, submitted, sent, dispatches) in [
        (false, true, true, 1, 1, 1, 0),
        (true, false, true, 1, 1, 1, 0),
        (true, true, false, 1, 1, 1, 0),
        (true, true, true, 0, 0, 0, 0),
        (true, true, true, 2, 1, 2, 0),
        (true, true, true, 2, 2, 1, 0),
        (true, true, true, 1, 1, 1, 1),
    ] {
        assert!(!owned_prepared_stage_empty_eligible(
            early, complete, no_parts, entries, submitted, sent, dispatches,
        ));
    }
    // A zero-dispatch ordinary recorder can contain a transfer/event. Without the
    // private early-prepared origin proof it must never become eligible.
    assert!(!owned_prepared_stage_empty_eligible(
        false, true, true, 1, 1, 1, 0
    ));
}

fn install_cpu_only_head_source(fixture: &Fixture) {
    let (sources, seed) = {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        let previous = session.gpu_decode_owner.clone().expect("warm owner");
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        let banks = previous.role_params(session, &fixture.roles[0]).unwrap();
        session.gpu_decode_owner = None;
        let mut sources = session.gpu_copy_sources().unwrap().unwrap();
        for base in &banks[..3] {
            sources.test_cpu_only_source(*base as usize + ROUTED[1] as usize);
        }
        (sources, crate::pager::owned_seed(session, 0).unwrap())
    };
    // Allocate outside the pager mutex, as in ensure_gpu_copy and the upload regressions.
    let copy = crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, LAYERS, seed)
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

struct Counts {
    before: usize,
}

impl Counts {
    fn begin() -> Self {
        OWNED_RECOVERY_TEST_COUNTS.with(|counts| {
            assert!(counts.replace(Some([0; 3])).is_none());
        });
        Self {
            before: OWNED_PREPARED_STAGE_TEST_DISCARDS.with(|count| count.get()),
        }
    }

    fn values(&self) -> (usize, [usize; 3]) {
        (
            OWNED_PREPARED_STAGE_TEST_DISCARDS.with(|count| count.get() - self.before),
            OWNED_RECOVERY_TEST_COUNTS.with(|count| count.get().unwrap()),
        )
    }
}

impl Drop for Counts {
    fn drop(&mut self) {
        OWNED_RECOVERY_TEST_COUNTS.with(|count| count.set(None));
    }
}

#[test]
#[ignore = "parent serial GPU only; complete prepared L0 boundary, real CPU recovery and discard provenance"]
fn gpu_owned_prepared_complete_stage_discards_empty_boundary_and_recovers_cpu_miss() {
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    // A complete first stage: frontend plus L0. The later model stage is deliberately
    // outside this graph, exactly the complete-stage boundary being optimized.
    fixture.graph.ops.truncate(2);
    fixture.preload();
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    let bindings = fixture.bindings();
    let initial_discards = OWNED_PREPARED_STAGE_TEST_DISCARDS.with(|count| count.get());
    execute_static(&fixture.backend, &current, &bindings).unwrap();
    assert_eq!(
        OWNED_PREPARED_STAGE_TEST_DISCARDS.with(|count| count.get()),
        initial_discards,
        "ordinary recording must not acquire the prepared-stage provenance"
    );
    let expected = fixture.read(fixture.buffer(fixture.outputs[0]));
    assert!(expected.iter().all(|x| x.is_finite() && *x > 0.0));
    install_cpu_only_head_source(&fixture);

    for missing in [false, true] {
        let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
        fixture.reset_frontend();
        execute_static_next(
            &fixture.backend,
            &current,
            &bindings,
            Some((&next, &bindings, &[1])),
        )
        .unwrap();
        {
            let packet = next.prepared_prefix.lock().unwrap();
            let packet = packet.as_ref().expect("actual prepared stage");
            let (head_end, head_layers, last_end, layers, self_service, owned) =
                packet.recorded_shape();
            assert_eq!(
                (head_end, last_end),
                (fixture.graph.ops.len(), fixture.graph.ops.len())
            );
            assert_eq!(
                (head_layers, layers),
                (1, 1),
                "no outstanding prepared parts"
            );
            assert!(
                owned && !self_service,
                "must exercise the notice/producer path"
            );
        }
        {
            let mut guard = fixture.backend.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            session.handoff_gpu_decode(&fixture.backend).unwrap();
            if missing {
                session.discard_cpu_miss(fixture.roles[0], &[ROUTED[1]]);
            }
        }
        fixture.reset_frontend();
        let counts = Counts::begin();
        execute_static(&fixture.backend, &next, &bindings).unwrap();
        let (discarded, recovery) = counts.values();
        assert_eq!(
            discarded, 1,
            "the fresh boundary recorder must take the explicit discard arm"
        );
        assert_eq!(
            recovery[0],
            usize::from(missing),
            "CPU miss must still enter guarded recovery"
        );
        assert_eq!(
            recovery[1],
            usize::from(missing),
            "real recovery submission must survive"
        );
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
        assert_eq!(fixture.read(fixture.buffer(fixture.outputs[0])), expected);
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        assert_eq!(
            session
                .routed_roles_resident_mask(&fixture.roles[0], &ROUTED)
                .unwrap(),
            HIT_MASK
        );
        assert!(
            !session.gpu_decode_in_forward(),
            "forward and producer ACK lifecycle must drain"
        );
    }
}
