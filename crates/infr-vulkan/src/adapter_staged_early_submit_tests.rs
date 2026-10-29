//! Parent-serialized GPU regressions for the accepted-L0-before-tail-analysis boundary.
use super::*;

#[test]
fn staged_binding_setup_reads_retained_scratch_without_switching_or_growing() {
    let mut arena = RuntimePhaseArena {
        scratch: vec![Some(Box::new(CpuBuffer(16)) as Box<dyn Buffer>)],
        scratch_layout: vec![Some(16)],
        scratch_topology: vec![true],
        ..RuntimePhaseArena::default()
    };
    arena.parked_scratch.push(PhaseScratch {
        topology: vec![false, true],
        layout: vec![None, Some(32)],
        scratch: vec![None, Some(Box::new(CpuBuffer(32)) as Box<dyn Buffer>)],
    });
    assert!(std::ptr::eq(
        arena.retained_setup_scratch(&vec![Some(16)]).unwrap(),
        &arena.scratch
    ));
    assert!(std::ptr::eq(
        arena.retained_setup_scratch(&vec![None, Some(24)]).unwrap(),
        &arena.parked_scratch[0].scratch
    ));
    assert!(
        arena
            .retained_setup_scratch(&vec![None, Some(33)])
            .is_none()
    );
    assert!(
        arena
            .retained_setup_scratch(&vec![Some(16), Some(16)])
            .is_none()
    );
    assert_eq!(arena.scratch_topology, [true]);
    assert_eq!(arena.parked_scratch.len(), 1);
    assert_eq!(arena.parked_scratch[0].topology, [false, true]);
}

#[test]
fn inherited_stage_retirement_requires_exact_nonempty_count_and_actual_prefix() {
    for submitted in [false, true] {
        assert_eq!(
            inherited_stage_retirement_count(None, 0, submitted).unwrap(),
            0
        );
        assert_eq!(
            inherited_stage_retirement_count(None, 2, submitted).unwrap(),
            0
        );
        for count in 1..=3 {
            assert_eq!(
                inherited_stage_retirement_count(Some(count), count, submitted).unwrap(),
                if submitted { count } else { 0 },
            );
            for pending in 0..=4 {
                if pending != count {
                    assert!(
                        inherited_stage_retirement_count(Some(count), pending, submitted).is_err()
                    );
                }
            }
        }
        assert!(inherited_stage_retirement_count(Some(0), 0, submitted).is_err());
    }
}

struct TraceGuard;

impl TraceGuard {
    fn begin(setup_failure: Option<(usize, bool)>) -> Self {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            assert!(
                state
                    .replace(Some(StagedActivationTestState {
                        events: Vec::new(),
                        setup_failure,
                        retirement_failure: None,
                    }))
                    .is_none()
            );
        });
        Self
    }

    fn events(&self) -> Vec<StagedActivationTestEvent> {
        STAGED_ACTIVATION_TEST_STATE.with(|state| state.borrow().as_ref().unwrap().events.clone())
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            state.take();
        });
    }
}

#[test]
#[ignore = "parent serial GPU only; missing prefix and mismatched inherited count cannot retire a pending"]
fn gpu_staged_inherited_retirement_without_prefix_or_exact_count_stays_conservative() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let trace = TraceGuard::begin(None);
    let old = fixture.backend.recorder().unwrap();
    old.zero(fixture.buffer(fixture.frontend), NE);
    let old = old.finish_nowait().unwrap();
    let old_fence = old.notice_fence();
    assert!(old_fence.is_some());
    let mut segments = std::collections::VecDeque::from([old]);
    retire_inherited_stage_pending(&mut segments, Some(1), None).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments.front().unwrap().notice_fence(), old_fence);
    assert_eq!(
        trace.events(),
        [StagedActivationTestEvent::InheritedNoPrefix(1)]
    );
    // The unchanged initial admission gate must still see the old stage's pending.
    assert!(!(segments.is_empty() && rolling_mapped_stream_clear(&PagedStream::default())));

    let current = fixture.backend.recorder().unwrap();
    current.zero(fixture.buffer(fixture.frontend), NE);
    let current = current.finish_nowait().unwrap();
    let current_fence = current.notice_fence();
    assert!(current_fence.is_some());
    assert_ne!(old_fence, current_fence);
    assert!(retire_inherited_stage_pending(&mut segments, Some(2), Some(&current)).is_err());
    assert_eq!(segments.len(), 1);
    assert_eq!(segments.front().unwrap().notice_fence(), old_fence);
    assert_eq!(current.notice_fence(), current_fence);
    assert_eq!(
        trace.events(),
        [StagedActivationTestEvent::InheritedNoPrefix(1)]
    );
    segments.pop_front().unwrap().wait().unwrap();
    current.wait().unwrap();
}

fn prepared_pair() -> (Fixture, VkDecodePlan, VkDecodePlan, Vec<Vec<f32>>) {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    fixture.preload();
    let (head, tail, next) = split_plans(&fixture);
    {
        let bindings = fixture.bindings();
        let full = VkDecodePlan::new_cpu(&fixture.graph, false, true);
        execute_static(&fixture.backend, &full, &bindings).unwrap();
        fixture.reset_frontend();
        fixture
            .backend
            .execute_staged(
                &[
                    ExecutionStage::new(&head, &bindings),
                    ExecutionStage::new(&tail, &bindings),
                ],
                &mut |_| Ok(()),
                Some((&next, &bindings, &[1])),
            )
            .unwrap();
        assert!(
            next.prepared_prefix.lock().unwrap().is_some(),
            "actual retained L0 packet"
        );
    }
    let expected = fixture.outputs();
    fixture.reset_frontend();
    (fixture, next, tail, expected)
}

#[test]
#[ignore = "parent serial GPU only; accepted L0 submit precedes tail binding analysis"]
fn gpu_staged_early_prefix_submits_before_tail_setup_and_preserves_ready_order() {
    let (fixture, head, tail, expected) = prepared_pair();
    let bindings = fixture.bindings();
    let trace = TraceGuard::begin(None);
    fixture
        .backend
        .execute_staged(
            &[
                ExecutionStage::new(&head, &bindings),
                ExecutionStage::new(&tail, &bindings),
            ],
            &mut |index| {
                staged_activation_test_note(StagedActivationTestEvent::Ready(index));
                Ok(())
            },
            None,
        )
        .unwrap();
    let events = trace.events();
    let first = |event| {
        events
            .iter()
            .position(|found| *found == event)
            .expect("event")
    };
    assert!(
        first(StagedActivationTestEvent::Ready(0)) < first(StagedActivationTestEvent::Submitted),
        "{events:?}"
    );
    assert!(
        first(StagedActivationTestEvent::Submitted) < first(StagedActivationTestEvent::Setup(1)),
        "{events:?}"
    );
    assert!(
        first(StagedActivationTestEvent::Setup(1)) < first(StagedActivationTestEvent::Ready(1)),
        "{events:?}"
    );
    assert_eq!(
        events[..first(StagedActivationTestEvent::Ready(1))]
            .iter()
            .filter(|event| **event == StagedActivationTestEvent::Submitted)
            .count(),
        1,
        "L0 is submitted once, never reactivated by the walk: {events:?}"
    );
    assert!(head.prepared_prefix.lock().unwrap().is_none());
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
    assert_eq!(fixture.outputs(), expected);
}

#[test]
#[ignore = "parent serial GPU only; middle-stage admission precedes its accepted prefix in a three-stage chain"]
fn gpu_staged_three_stage_middle_admits_following_template_before_prefix_activation() {
    let (fixture, head, tail, expected) = prepared_pair();
    let repeated_tail = VkDecodePlan::new_cpu(&tail.graph, false, true);
    let bindings = fixture.bindings();
    let trace = TraceGuard::begin(None);
    fixture
        .backend
        .execute_staged(
            &[
                ExecutionStage::new(&head, &bindings),
                ExecutionStage::new(&tail, &bindings),
                ExecutionStage::new(&repeated_tail, &bindings),
            ],
            &mut |index| {
                staged_activation_test_note(StagedActivationTestEvent::Ready(index));
                Ok(())
            },
            None,
        )
        .unwrap();
    let events = trace.events();
    let position = |expected| {
        events
            .iter()
            .position(|event| *event == expected)
            .unwrap_or_else(|| panic!("missing {expected:?}: {events:?}"))
    };
    let submitted = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| {
            (*event == StagedActivationTestEvent::Submitted).then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        submitted.len(),
        3,
        "all three prepared prefixes must be accepted exactly once, not fall back: {events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                StagedActivationTestEvent::Ready(index) => Some(*index),
                _ => None,
            })
            .collect::<Vec<_>>(),
        [0, 1, 2],
        "{events:?}"
    );
    let ready = |index| position(StagedActivationTestEvent::Ready(index));
    let following = |index| position(StagedActivationTestEvent::FollowingTemplate(index));
    assert!(
        ready(0) < following(0) && following(0) < submitted[0],
        "{events:?}"
    );
    assert!(submitted[0] < ready(1), "{events:?}");
    assert!(
        ready(1) < following(1) && following(1) < submitted[1],
        "{events:?}"
    );
    assert!(
        submitted[1] < ready(2) && ready(2) < submitted[2],
        "{events:?}"
    );
    assert!(!events.contains(&StagedActivationTestEvent::FollowingTemplate(2)));
    assert!(
        !events.iter().any(|event| matches!(
            event,
            StagedActivationTestEvent::InheritedPending(_)
                | StagedActivationTestEvent::InheritedRetiring(_)
                | StagedActivationTestEvent::InheritedRetired(_)
                | StagedActivationTestEvent::InheritedNoPrefix(_)
        )),
        "three-stage chains must not inherit the two-stage deferral proof: {events:?}"
    );
    assert!(head.prepared_prefix.lock().unwrap().is_none());
    assert!(tail.prepared_prefix.lock().unwrap().is_none());
    assert!(repeated_tail.prepared_prefix.lock().unwrap().is_none());
    assert!(!fixture.backend.runtime_phase.is_poisoned());
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 3.0);
    // Both tails read the unchanged L0 result; only the independent visit counter advances.
    assert_close(&fixture.outputs(), &expected);
}

#[test]
#[ignore = "parent serial GPU only; stage1 submit precedes next-token aliases, failed snapshots drain once"]
fn gpu_staged_following_prefix_precedes_next_snapshots_and_errors_drain_before_unlock() {
    use infr_core::backend::ExecutionLookahead;
    for failure in [None, Some(false), Some(true)] {
        let (fixture, head, tail, expected) = prepared_pair();
        let next_head = VkDecodePlan::new_cpu(&head.graph, false, true);
        let next_tail = VkDecodePlan::new_cpu(&tail.graph, false, true);
        let bindings = fixture.bindings();
        let current = [
            ExecutionStage::new(&head, &bindings),
            ExecutionStage::new(&tail, &bindings),
        ];
        let next = [
            ExecutionStage::new(&next_head, &bindings),
            ExecutionStage::new(&next_tail, &bindings),
        ];
        STAGED_FOLLOW_TEST_STATE.with(|state| {
            assert!(
                state
                    .replace(Some(StagedFollowTestState {
                        events: Vec::new(),
                        failure,
                    }))
                    .is_none()
            )
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fixture.backend.execute_staged_lookahead(
                &current,
                &mut |_| Ok(()),
                Some(ExecutionLookahead::new(&next, &[1])),
            )
        }));
        let observed = STAGED_FOLLOW_TEST_STATE.with(|state| state.take().unwrap());
        let (owner, generation) = drained_owner_epoch(&fixture);
        assert_eq!(owner.owned_active(), failure.is_none());
        assert_eq!(observed.events.first(), Some(&"submitted"));
        assert!(observed.events.contains(&"snapshot"));
        match failure {
            None => {
                result.unwrap().unwrap();
            }
            Some(false) => assert!(
                result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("post-activation snapshot error")
            ),
            Some(true) => assert!(result.is_err()),
        }
        assert!(!fixture.backend.runtime_phase.is_poisoned());
        // Both accepted prefixes executed once, even if host alias preparation failed.
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
        assert_close(&fixture.outputs(), &expected);
        if failure.is_some() {
            assert!(next_head.prepared_prefix.lock().unwrap().is_none());
            assert!(next_tail.prepared_prefix.lock().unwrap().is_none());
        }
        fixture.reset_frontend();
        fixture
            .backend
            .execute_staged(&current, &mut |_| Ok(()), None)
            .unwrap();
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
        assert_close(&fixture.outputs(), &expected);
        let (reused, reused_generation) = drained_owner_epoch(&fixture);
        assert!(Arc::ptr_eq(&owner, &reused));
        assert!(reused.owned_active());
        assert_eq!(
            generation + u64::from(failure.is_some()),
            reused_generation,
            "normal reuse must retain epoch; errors recover exactly once without test handoff"
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
#[ignore = "parent serial GPU only; post-submit setup/callback error, cancellation and unwind drain without replay"]
fn gpu_staged_early_prefix_errors_cancel_and_unwind_drain_without_replay() {
    // Tail setup Err/panic, then input-ready Err/panic/Aborted. Only these bounded failures.
    for mode in 0..5 {
        let (fixture, head, tail, expected) = prepared_pair();
        let bindings = fixture.bindings();
        let sentinel = vec![-71.0f32; NE];
        fixture
            .backend
            .upload(
                fixture.buffer(fixture.outputs[1]),
                bytemuck::cast_slice(&sentinel),
            )
            .unwrap();
        let trace = TraceGuard::begin((mode < 2).then_some((1, mode == 1)));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fixture.backend.execute_staged(
                &[
                    ExecutionStage::new(&head, &bindings),
                    ExecutionStage::new(&tail, &bindings),
                ],
                &mut |index| {
                    staged_activation_test_note(StagedActivationTestEvent::Ready(index));
                    if index == 1 {
                        match mode {
                            2 => return Err(be("injected staged input-ready error")),
                            3 => panic!("injected staged input-ready panic"),
                            4 => return Err(Error::Aborted),
                            _ => (),
                        }
                    }
                    Ok(())
                },
                None,
            )
        }));
        if mode == 1 || mode == 3 {
            assert!(result.is_err());
        } else if mode == 4 {
            assert!(matches!(result.unwrap(), Err(Error::Aborted)));
        } else {
            assert!(
                result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains(if mode == 0 {
                        "injected staged binding setup error"
                    } else {
                        "injected staged input-ready error"
                    })
            );
        }
        let events = trace.events();
        assert_eq!(
            events
                .iter()
                .filter(|event| **event == StagedActivationTestEvent::Submitted)
                .count(),
            1,
            "{events:?}"
        );
        if mode < 2 {
            assert_eq!(
                events.last(),
                Some(&StagedActivationTestEvent::Drained),
                "{events:?}"
            );
            assert!(
                !events.contains(&StagedActivationTestEvent::Ready(1)),
                "tail input-ready must not advance after setup failure"
            );
        }
        assert!(
            head.prepared_prefix.lock().unwrap().is_none(),
            "consumed L0 cannot retry automatically"
        );
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
        assert_eq!(
            fixture.read(fixture.buffer(fixture.outputs[0])),
            expected[0]
        );
        assert_eq!(
            fixture.read(fixture.buffer(fixture.outputs[1])),
            sentinel,
            "tail never ran"
        );
        assert!(
            !fixture.backend.runtime_phase.is_poisoned(),
            "setup panic must unlock before resuming"
        );
        drop(trace);
        // A separate request can retry normally; it is not an automatic L0 replay on error.
        fixture.reset_frontend();
        fixture
            .backend
            .execute_staged(
                &[
                    ExecutionStage::new(&head, &bindings),
                    ExecutionStage::new(&tail, &bindings),
                ],
                &mut |_| Ok(()),
                None,
            )
            .unwrap();
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
        assert_eq!(fixture.outputs(), expected);
    }
}
