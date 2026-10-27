//! Parent-only serial GPU qualification; never run by the TMP preparation task.
use super::super::pure_mapped_coalesce::TestTrace;
use super::*;

fn fixture(pattern: &[usize], overlap: bool, aliases: bool) -> Fixture {
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    Arc::make_mut(&mut fixture.backend.cfg)
        .kernels
        .vulkan
        .decode_gpu_resident_layers = 16;
    Arc::make_mut(&mut fixture.backend.cfg)
        .kernels
        .vulkan
        .decode_gpu_resident_copy_overlap = overlap;
    let original = fixture.graph.ops.clone();
    fixture.graph.ops.clear();
    for &layer in pattern {
        fixture
            .graph
            .ops
            .extend_from_slice(&original[layer * 2..layer * 2 + 2]);
    }
    if aliases {
        for op in &mut fixture.graph.ops {
            if let Op::MoeFfn {
                gate_exps, up_exps, ..
            } = op
            {
                *up_exps = *gate_exps;
            }
        }
    }
    fixture
}

fn plan(fixture: &Fixture) -> VkDecodePlan {
    let mut plan = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    plan.phase_requests =
        Some(stages::compiled_workspace_requests(&fixture.backend, &fixture.graph).unwrap());
    plan
}

fn policy(fixture: &mut Fixture, enabled: bool) {
    Arc::make_mut(&mut fixture.backend.cfg)
        .kernels
        .vulkan
        .decode_gpu_resident_pure_mapped_layers = if enabled { 4 } else { 0 };
}

fn install(fixture: &Fixture, plan: &VkDecodePlan) {
    fixture.preload();
    fixture.reset_frontend();
    execute_static(&fixture.backend, plan, &fixture.bindings()).unwrap();
    prepared_mapped_run_tests::install_mixed_owner(fixture);
}

fn prepare_case(fixture: &Fixture, missing: bool) {
    fixture.preload();
    if missing {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.discard_cpu_miss(fixture.roles[1], &[ROUTED[1]]);
    }
    fixture.reset_frontend();
}

fn lru(fixture: &Fixture) -> Vec<(Vec<u32>, usize)> {
    let mut guard = fixture.backend.moe_pager().lock().unwrap();
    let session = guard.as_mut().unwrap();
    assert!(!session.gpu_decode_in_forward());
    session.handoff_gpu_decode(&fixture.backend).unwrap();
    // Compare block recency, not slot numbers from different initial free-list histories.
    // Existing raw metadata oracles retain physical-slot coverage.
    session.test_resident_lru_snapshot().unwrap()
}

#[test]
#[ignore = "parent serial GPU only; puremapped coalescing ordinary path, overlap and alias exact LRU"]
fn gpu_pure_mapped_coalesce_ordinary_exact_and_actual_cuts() {
    for overlap in [false, true] {
        let pattern = [0; 9];
        let mut fixture = fixture(&pattern, overlap, true);
        let plan = plan(&fixture);
        install(&fixture, &plan);
        prepare_case(&fixture, false);
        execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
        let expected = fixture.read(fixture.buffer(fixture.outputs[0]));
        let expected_lru = lru(&fixture);
        policy(&mut fixture, true);
        prepare_case(&fixture, false);
        let trace = TestTrace::begin();
        execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
        assert_eq!(fixture.read(fixture.buffer(fixture.outputs[0])), expected);
        fixture.assert_frontend(fixture.buffer(fixture.frontend), pattern.len() as f32);
        assert_eq!(lru(&fixture), expected_lru);
        let decisions = trace.decisions();
        assert!(
            decisions
                .iter()
                .any(|d| !d.preparing && d.entries == 1 && d.submitted == 0 && d.due),
            "first submit unchanged: {decisions:?}"
        );
        assert!(
            decisions
                .iter()
                .any(|d| !d.preparing && d.entries == 5 && d.submitted == 1 && d.due),
            "actual four-layer cut: {decisions:?}"
        );
        assert!(
            decisions
                .iter()
                .any(|d| !d.preparing && d.entries == 4 && d.submitted == 1 && !d.due),
            "must hold a real group, not merely rename a counter: {decisions:?}"
        );
    }
}

#[test]
#[ignore = "parent serial GPU only; host at first/middle/last, real CPU recovery, prepared and runtime grouping"]
fn gpu_pure_mapped_coalesce_host_boundaries_and_prepared_recovery() {
    for host in [0, 4, 8] {
        let mut pattern = [0; 9];
        pattern[host] = 1;
        let mut fixture = fixture(&pattern, true, false);
        let current = plan(&fixture);
        install(&fixture, &current);
        prepare_case(&fixture, true);
        execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
        let expected = fixture.outputs();
        let expected_lru = lru(&fixture);
        policy(&mut fixture, true);
        // Produce a real next packet. Its horizon still comes only from the existing config.
        let next = plan(&fixture);
        prepare_case(&fixture, false);
        execute_static_next(
            &fixture.backend,
            &current,
            &fixture.bindings(),
            Some((&next, &fixture.bindings(), &[1])),
        )
        .unwrap();
        let ends = next
            .prepared_prefix
            .lock()
            .unwrap()
            .as_ref()
            .expect("actual prepared packet")
            .coalesced_test_parts();
        assert_eq!(ends[0], 1, "first useful submit must not grow");
        let mut first = 0;
        for &end in &ends {
            assert!(end > first && end - first <= 4, "{ends:?}");
            if (first..end).contains(&host) {
                assert_eq!(end - first, 1, "host cadence changed: {ends:?}");
            }
            first = end;
        }
        // Prove actual prepared submission, not just successful optional recording.
        prepare_case(&fixture, false);
        PREPARED_MAPPED_TEST_STATE.with(|state| {
            assert!(
                state
                    .replace(Some(PreparedMappedTestState {
                        decisions: Vec::new(),
                        fail_after_layers: None,
                    }))
                    .is_none()
            )
        });
        let consumed = execute_static(&fixture.backend, &next, &fixture.bindings());
        let decisions = PREPARED_MAPPED_TEST_STATE.with(|state| state.take().unwrap().decisions);
        consumed.unwrap();
        assert!(
            decisions.iter().any(|d| d.end - d.first > 1 && d.eligible),
            "must really submit a coalesced prepared part, not silently re-record: {decisions:?}"
        );
        let recovery_next = plan(&fixture);
        execute_static_next(
            &fixture.backend,
            &current,
            &fixture.bindings(),
            Some((&recovery_next, &fixture.bindings(), &[1])),
        )
        .unwrap();
        assert!(recovery_next.prepared_prefix.lock().unwrap().is_some());
        prepare_case(&fixture, true);
        OWNED_RECOVERY_TEST_COUNTS.with(|counts| assert!(counts.replace(Some([0; 3])).is_none()));
        let result = execute_static(&fixture.backend, &recovery_next, &fixture.bindings());
        let recovery = OWNED_RECOVERY_TEST_COUNTS.with(|counts| counts.replace(None).unwrap());
        result.unwrap();
        assert!(recovery[0] > 0, "fixture must actually recover a CPU miss");
        fixture.assert_frontend(fixture.buffer(fixture.frontend), pattern.len() as f32);
        assert_eq!(fixture.outputs(), expected);
        assert_eq!(lru(&fixture), expected_lru);
        assert!(recovery_next.prepared_prefix.lock().unwrap().is_none());
    }
}

#[test]
#[ignore = "parent serial GPU only; coalesced enqueue error/cancel must drain before an unassisted retry"]
fn gpu_pure_mapped_coalesce_cancel_error_and_retry() {
    for abort in [false, true] {
        let mut fixture = fixture(&[0; 9], false, false);
        let plan = plan(&fixture);
        install(&fixture, &plan);
        policy(&mut fixture, true);
        prepare_case(&fixture, false);
        ROLLING_MAPPED_TEST_STATE.with(|state| {
            assert!(
                state
                    .replace(Some(RollingMappedTestState {
                        decisions: Vec::new(),
                        admission_proofs: 0,
                        fail_after_layers: Some((5, abort)),
                        confirm_recovery: false,
                    }))
                    .is_none()
            )
        });
        let result = execute_static(&fixture.backend, &plan, &fixture.bindings());
        let decisions = ROLLING_MAPPED_TEST_STATE.with(|state| state.take().unwrap().decisions);
        if abort {
            assert!(matches!(result, Err(Error::Aborted)));
        } else {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("injected rolling mapped-run")
            );
        }
        assert!(
            decisions
                .iter()
                .any(|d| d.end == 5 && d.eligible && d.over_budget)
        );
        // No manual handoff/reset before retry; parent recovery is responsible for safe reuse.
        fixture.reset_frontend();
        execute_static(&fixture.backend, &plan, &fixture.bindings()).unwrap();
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 9.0);
        lru(&fixture);
    }
}

#[test]
#[ignore = "parent serial GPU only; independent two-row routes, real CPU miss and coalesced exact LRU"]
fn gpu_pure_mapped_coalesce_two_lanes_with_distinct_routes_and_cpu_miss() {
    for overlap in [false, true] {
        let pattern = [0, 0, 0, 0, 1, 0, 0, 0, 0];
        let mut fixture = fixture(&pattern, overlap, false);
        for (index, tensor) in fixture.graph.tensors.iter_mut().enumerate() {
            if tensor.desc.shape.as_slice() != [1, NE] {
                continue;
            }
            tensor.desc.shape = vec![2, NE].into();
            if let Some((_, buffer)) = fixture.buffers.iter_mut().find(|(id, _)| id.0 as usize == index) {
                let mut bytes = vec![0u8; NE * 4];
                fixture.backend.download(buffer.as_ref(), &mut bytes).unwrap();
                let expanded = fixture.backend.alloc(2 * NE * 4, BufferUsage::Staging).unwrap();
                fixture.backend.upload(expanded.as_ref(), &bytes.repeat(2)).unwrap();
                *buffer = expanded;
            }
        }
        for op in &mut fixture.graph.ops {
            if let Op::Add { n, .. } = op {
                *n *= 2;
            }
        }
        fixture.graph.independent_rows = true;
        fixture.graph.sequence_spans = [3, 11].into_iter().enumerate().map(|(row, start)| {
            infr_core::graph::SequenceSpan { row_start: row as u32, rows: 1, start_pos: start }
        }).collect();
        let router_x = fixture.graph.ops.iter().find_map(|op| match op {
            Op::MoeFfn { router_x, .. } => Some(*router_x),
            _ => None,
        }).unwrap();
        let mut routing = vec![1f32; 2 * NE];
        routing[NE] = -1.0;
        fixture.backend.upload(fixture.buffer(router_x), bytemuck::cast_slice(&routing)).unwrap();
        let current = plan(&fixture);
        install(&fixture, &current);
        let reset = |fixture: &Fixture| {
            fixture.backend.upload(fixture.buffer(fixture.frontend),
                bytemuck::cast_slice(&vec![0f32; 2 * NE])).unwrap();
        };
        let outputs = |fixture: &Fixture| {
            fixture.outputs.iter().map(|&id| {
                let mut values = vec![0f32; 2 * NE];
                fixture.backend.download(fixture.buffer(id), bytemuck::cast_slice_mut(&mut values)).unwrap();
                values
            }).collect::<Vec<_>>()
        };
        prepare_case(&fixture, true);
        reset(&fixture);
        execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
        let expected = outputs(&fixture);
        let expected_lru = lru(&fixture);

        policy(&mut fixture, true);
        prepare_case(&fixture, true);
        reset(&fixture);
        OWNED_RECOVERY_TEST_COUNTS.with(|counts| assert!(counts.replace(Some([0; 3])).is_none()));
        let trace = TestTrace::begin();
        let result = execute_static(&fixture.backend, &current, &fixture.bindings());
        let recovery = OWNED_RECOVERY_TEST_COUNTS.with(|counts| counts.replace(None).unwrap());
        result.unwrap();
        assert!(recovery[0] > 0, "two-row case must really recover a CPU miss");
        assert_eq!(outputs(&fixture), expected, "overlap={overlap}");
        let mut frontend = vec![0f32; 2 * NE];
        fixture.backend.download(fixture.buffer(fixture.frontend), bytemuck::cast_slice_mut(&mut frontend)).unwrap();
        assert_eq!(frontend, vec![pattern.len() as f32; 2 * NE]);
        assert_eq!(lru(&fixture), expected_lru, "overlap={overlap}");
        assert!(trace.decisions().iter().any(|d| !d.preparing && d.entries - d.submitted > 1),
            "two-row execution must actually hold a pure mapped group");
    }
}
