#![cfg(test)]

use super::*;
use std::cell::Cell;

struct DummyBuffer(usize);

impl Buffer for DummyBuffer {
    fn len_bytes(&self) -> usize {
        self.0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn import_prefix(source: &[u8], imported_len: usize) -> SessionTransferPlan {
    SessionTransferPlan::new(vec![crate::ImportedHostAllocation {
        base: source.as_ptr() as usize,
        logical_len: source.len(),
        imported_len,
        shards: vec![crate::ImportedHostShard {
            offset: 0,
            len: imported_len,
            buffer: Arc::new(DummyBuffer(imported_len)),
        }],
    }])
}

#[test]
fn owned_upload_class_rejects_empty_bounded_and_unmapped_without_import_lookup() {
    for (bounded, mapped) in [
        (false, vec![]),
        (true, vec![true]),
        (false, vec![false]),
        (false, vec![true, false, true]),
    ] {
        let class = OwnedUploadClass::from_targets(bounded, mapped.into_iter());
        assert!(!class.mapped);
        assert!(!class.host_ready_eligible(|| panic!("ineligible batch scanned imports")));
        assert!(class.no_imported_sources.get().is_none());
    }
}

#[test]
fn owned_upload_class_caches_each_result_once_and_never_shares_between_plans() {
    for eligible in [false, true] {
        let calls = Cell::new(0);
        let class = OwnedUploadClass::from_targets(false, [true, true].into_iter());
        assert!(class.mapped);
        for _ in 0..3 {
            assert_eq!(
                class.host_ready_eligible(|| {
                    calls.set(calls.get() + 1);
                    eligible
                }),
                eligible
            );
        }
        assert_eq!(calls.get(), 1);
        let next = OwnedUploadClass::from_targets(false, [true, true].into_iter());
        assert_eq!(next.host_ready_eligible(|| !eligible), !eligible);
    }
}

#[test]
fn owned_upload_class_stays_conservative_when_partial_imports_are_shed() {
    let source = [0u8; 128];
    let plan = import_prefix(&source, 64);
    let imported = OwnedUploadClass::from_targets(false, [true].into_iter());
    let unmapped_source = OwnedUploadClass::from_targets(false, [true].into_iter());
    assert!(!imported.host_ready_eligible(|| !plan.contains_imported_upload(&source[..32])));
    assert!(unmapped_source.host_ready_eligible(|| !plan.contains_imported_upload(&source[64..96])));
    assert_eq!(plan.shed_unused_import_tails(64), 64);
    assert!(!plan.contains_imported_upload(&source[..32]));
    assert!(!imported.host_ready_eligible(|| panic!("cached negative must remain conservative")));
    assert!(
        unmapped_source.host_ready_eligible(|| panic!("non-imported source stayed non-imported"))
    );
    let next = OwnedUploadClass::from_targets(false, [true].into_iter());
    assert!(next.host_ready_eligible(|| !plan.contains_imported_upload(&source[..32])));
}

#[test]
fn owned_upload_mixed_import_batch_cannot_become_host_ready_from_a_partial_check() {
    let source = [0u8; 128];
    let plan = import_prefix(&source, 64);
    let class = OwnedUploadClass::from_targets(false, [true, true].into_iter());
    assert!(
        !class.host_ready_eligible(|| [&source[64..96], &source[..32]]
            .into_iter()
            .all(|range| !plan.contains_imported_upload(range)))
    );
}

#[test]
fn owned_upload_empty_plan_still_rejects_mapped_execution_before_any_write() {
    let plan = OwnedUploadPlan {
        transfer_plan: Arc::new(SessionTransferPlan::default()),
        uploads: Vec::new(),
        bounded: Vec::new(),
        requested: 0,
        bytes: 0,
        class: OwnedUploadClass::from_targets(false, std::iter::empty()),
    };
    assert!(!plan.record_while_uploading());
    assert!(!plan.host_ready_eligible());
    assert!(plan.execute_mapped_cpu(0).is_err());
}

#[test]
#[ignore = "manual optimized CPU microbenchmark; per-plan classification/import lookup only, no GPU"]
fn d2_original_vs_cached_classification_microbenchmark() {
    use std::hint::black_box;
    use std::time::Instant;

    fn original(plan: &SessionTransferPlan, source: &[u8], mapped: &[bool]) -> bool {
        let recordable = || !mapped.is_empty() && mapped.iter().all(|&value| value);
        let eligible = || {
            recordable()
                && source
                    .chunks_exact(64)
                    .all(|range| !plan.contains_imported_upload(range))
        };
        // Started checks recordability, then eligibility; CPU execution checks eligibility again.
        recordable() && eligible() && eligible()
    }

    fn cached(plan: &SessionTransferPlan, source: &[u8], mapped: &[bool]) -> bool {
        // Include construction/first lookup for each plan, not just warmed cache hits.
        let class = OwnedUploadClass::from_targets(false, mapped.iter().copied());
        let eligible = || {
            source
                .chunks_exact(64)
                .all(|range| !plan.contains_imported_upload(range))
        };
        class.mapped && class.host_ready_eligible(eligible) && class.host_ready_eligible(eligible)
    }

    const ITERATIONS: usize = 2_000;
    for rows in [1, 2, 4, infr_core::backend::MAX_DECODE_ROWS] {
        for mode in ["no-import", "mixed-import", "all-import", "unmapped-target"] {
            let jobs = rows * 8 * 3;
            let source = vec![0u8; jobs * 64];
            let mut mapped = vec![true; jobs];
            let plan = match mode {
                "mixed-import" => import_prefix(&source, source.len() / 2),
                "all-import" => import_prefix(&source, source.len()),
                _ => SessionTransferPlan::default(),
            };
            if mode == "unmapped-target" {
                mapped[jobs - 1] = false;
            }
            let expected = original(&plan, &source, &mapped);
            assert_eq!(cached(&plan, &source, &mapped), expected);
            let mut totals = [0u128; 2];
            for arm in [0, 1, 1, 0] {
                let started = Instant::now();
                let mut count = 0;
                for _ in 0..ITERATIONS {
                    let plan = black_box(&plan);
                    let source = black_box(source.as_slice());
                    let mapped = black_box(mapped.as_slice());
                    count += usize::from(black_box(if arm == 0 {
                        original(plan, source, mapped)
                    } else {
                        cached(plan, source, mapped)
                    }));
                }
                totals[arm] += started.elapsed().as_nanos();
                assert_eq!(count, ITERATIONS * usize::from(expected));
            }
            eprintln!(
                "D2 rows={rows} jobs={jobs} mode={mode} iterations_per_arm={} original_ns={} cached_ns={}",
                ITERATIONS * 2, totals[0], totals[1]
            );
        }
    }
}
