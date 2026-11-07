#[cfg(test)]
mod fixtures {
    use super::super::*;
    use std::cell::Cell;

    struct CpuBuffer;
    impl Buffer for CpuBuffer {
        fn len_bytes(&self) -> usize {
            0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn buffers() -> ResidentBuffers {
        ResidentBuffers {
            control: Arc::new(CpuBuffer),
            journal: Arc::new(CpuBuffer),
            capacity: 3,
            notices: None,
            producer: None,
            waiter: None,
            indirect: None,
            gpu_copy: None,
            gpu_copy_init_attempted: false,
        }
    }

    fn layouts(rows: usize) -> [ResidentMoeLayout; 2] {
        [ResidentMoeLayout {
            rows,
            n_expert: 4,
            n_used: 2,
        }; 2]
    }

    #[test]
    fn resident_copy_policy_auto_single_explicit_multirow_and_verify() {
        for verify in [false, true] {
            for independent in [false, true] {
                for rows in 1..=crate::GPU_RESIDENT_MAX_ROWS {
                    let mut graph = Graph::new();
                    graph.mtp_verify = verify;
                    graph.independent_rows = independent;
                    if independent {
                        graph.sequence_spans = (0..rows)
                            .map(|row| SequenceSpan {
                                row_start: row as u32,
                                rows: 1,
                                start_pos: 30000,
                            })
                            .collect();
                    }
                    let layout = layouts(rows);
                    assert_eq!(
                        resident_copy_requested(None, &graph, &layout),
                        !verify && (independent || rows == 1)
                    );
                    assert!(!resident_copy_requested(Some(false), &graph, &layout));
                    assert_eq!(
                        resident_copy_requested(Some(true), &graph, &layout),
                        !verify || (!independent && (2..=4).contains(&rows))
                    );
                }
            }
        }
        let graph = Graph::new();
        assert_eq!(decode_token_rows(&graph), Some(1));
        assert!(
            !resident_copy_requested(None, &graph, &layouts(2)),
            "nominal one-row decode must not override actual MoE rows"
        );
        let mut mixed = layouts(1);
        mixed[1].rows = 2;
        assert!(!resident_copy_requested(None, &graph, &mixed));
        assert!(!resident_copy_requested(None, &graph, &[]));
        assert!(resident_copy_requested(
            Some(true),
            &graph,
            &layouts(1)[..1]
        ));
    }

    #[test]
    fn resident_owned_copy_excludes_both_mtp_sessions_and_verify_graphs() {
        for mtp_enabled in [false, true] {
            for verify in [false, true] {
                let mut graph = Graph::new();
                graph.mtp_verify = verify;
                assert_eq!(
                    resident_copy_owned_allowed(mtp_enabled, &graph),
                    !mtp_enabled && !verify
                );
                for rows in 2..=4 {
                    assert!(resident_copy_requested(Some(true), &graph, &layouts(rows)));
                    if verify || mtp_enabled {
                        assert!(!resident_copy_owned_allowed(mtp_enabled, &graph));
                    }
                }
            }
        }
    }

    #[test]
    fn resident_copy_mode_reset_only_retries_a_present_mismatched_kind() {
        let mut retained = buffers();
        let copy = crate::pager::GpuCopyBuffers::test_stub(Arc::new(CpuBuffer));
        retained.gpu_copy = Some(copy.clone());
        retained.gpu_copy_init_attempted = true;
        retained.reset_gpu_copy_mode(false);
        assert!(retained.gpu_copy_init_attempted);
        assert!(Arc::ptr_eq(retained.gpu_copy.as_ref().unwrap(), &copy));
        retained.reset_gpu_copy_mode(true);
        assert!(retained.gpu_copy.is_none());
        assert!(!retained.gpu_copy_init_attempted);
        retained.ensure_gpu_copy_with(true, |_| Ok(None)).unwrap();
        for owned_allowed in [false, true] {
            retained.reset_gpu_copy_mode(owned_allowed);
            assert!(retained.gpu_copy_init_attempted);
            assert!(retained.gpu_copy.is_none());
        }
    }

    #[test]
    fn resident_copy_lazy_init_and_prepared_selection_follow_each_graph() {
        let mut retained = buffers();
        let calls = Cell::new(0);
        let copy = crate::pager::GpuCopyBuffers::test_stub(Arc::new(CpuBuffer));
        let init = |capacity| {
            assert_eq!(capacity, 3);
            calls.set(calls.get() + 1);
            Ok(Some(copy.clone()))
        };
        // The first multirow/Verify graph owns the common buffers, but not copy metadata.
        retained.ensure_gpu_copy_with(false, init).unwrap();
        assert!(!retained.gpu_copy_init_attempted);
        assert_eq!(calls.get(), 0);
        assert!(command_prefix::copy_binding_matches(
            retained.selected_gpu_copy(false),
            None
        ));

        retained.ensure_gpu_copy_with(true, init).unwrap();
        assert!(retained.gpu_copy_init_attempted);
        assert_eq!(calls.get(), 1);
        assert!(Arc::ptr_eq(
            retained.selected_gpu_copy(true).unwrap(),
            &copy
        ));
        assert!(
            !command_prefix::copy_binding_matches(retained.selected_gpu_copy(true), None),
            "a no-copy prepared packet must be rejected after lazy allocation"
        );
        for requested in [false, true, false, false, true] {
            retained.ensure_gpu_copy_with(requested, init).unwrap();
            assert_eq!(calls.get(), 1);
            assert_eq!(retained.selected_gpu_copy(requested).is_some(), requested);
            assert!(Arc::ptr_eq(retained.gpu_copy.as_ref().unwrap(), &copy));
            assert!(command_prefix::copy_binding_matches(
                retained.selected_gpu_copy(requested),
                requested.then_some(&copy)
            ));
        }
        assert!(
            !command_prefix::copy_binding_matches(retained.selected_gpu_copy(false), Some(&copy)),
            "a warm copy packet must not activate on an auto-ineligible graph"
        );
        let replacement = crate::pager::GpuCopyBuffers::test_stub(Arc::new(CpuBuffer));
        assert!(!command_prefix::copy_binding_matches(
            retained.selected_gpu_copy(true),
            Some(&replacement)
        ));
        // Common-buffer replacement invalidates the old command and restarts lazy admission.
        retained = buffers();
        retained.ensure_gpu_copy_with(true, init).unwrap();
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn resident_copy_unavailable_or_failed_init_is_not_retried_each_token() {
        for fails in [false, true] {
            let mut retained = buffers();
            let calls = Cell::new(0);
            let init = |_| {
                calls.set(calls.get() + 1);
                if fails {
                    Err(be("fixture allocation failure"))
                } else {
                    Ok(None)
                }
            };
            assert_eq!(retained.ensure_gpu_copy_with(true, init).is_err(), fails);
            assert!(retained.gpu_copy_init_attempted);
            retained.ensure_gpu_copy_with(true, init).unwrap();
            retained.ensure_gpu_copy_with(false, init).unwrap();
            assert_eq!(calls.get(), 1);
            assert!(retained.selected_gpu_copy(true).is_none());
            assert!(command_prefix::copy_binding_matches(
                retained.selected_gpu_copy(true),
                None
            ));
        }
    }
}
