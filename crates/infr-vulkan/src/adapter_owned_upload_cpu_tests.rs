use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

#[test]
fn decode_mtp_isolation_host_ready_creation_cpu() {
    for requested in [false, true] {
        for mtp_enabled in [false, true] {
            for owned_copy_requested in [false, true] {
                let attempts = AtomicUsize::new(0);
                let worker = resident_host_ready_worker(
                    requested,
                    mtp_enabled,
                    owned_copy_requested,
                    || {
                        attempts.fetch_add(1, Ordering::Relaxed);
                        Ok(37u32)
                    },
                );
                let expected = requested && !mtp_enabled && owned_copy_requested;
                assert_eq!(worker, expected.then_some(37));
                assert_eq!(attempts.load(Ordering::Relaxed), usize::from(expected));
            }
        }
    }
}

#[test]
fn decode_mtp_isolation_host_ready_create_failure_falls_back_cpu() {
    let attempts = AtomicUsize::new(0);
    let worker = resident_host_ready_worker(true, false, true, || -> Result<()> {
        attempts.fetch_add(1, Ordering::Relaxed);
        Err(be("injected host-ready creation failure"))
    });
    assert!(
        worker.is_none(),
        "failure must retain the old completion gate"
    );
    assert_eq!(attempts.load(Ordering::Relaxed), 1);
    // Creation is optional, not sticky: a later producer may enable it successfully.
    assert_eq!(
        resident_host_ready_worker(true, false, true, || Ok(41u32)),
        Some(41),
    );
}

// Preserve the pre-D1 implementation as an independent validation/order oracle.
fn original_notice_slots(
    words: &[u32; crate::recorder::RESIDENT_JOURNAL_WORDS],
    layout: ResidentMoeLayout,
) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    let count = layout.count();
    if layout.rows == 0
        || layout.rows > crate::recorder::RESIDENT_JOURNAL_MASK_WORDS
        || layout.n_used == 0
        || layout.n_used > 16
        || count > crate::recorder::RESIDENT_JOURNAL_ID_WORDS
        || words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] != 2
    {
        return Err(be("invalid resident miss notice shape/status"));
    }
    let masks = words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD
        ..crate::recorder::RESIDENT_JOURNAL_MASK_WORD + layout.rows]
        .to_vec();
    let valid_mask = (1u32 << layout.n_used) - 1;
    if masks.iter().any(|&mask| mask & !valid_mask != 0)
        || words[..count]
            .iter()
            .any(|&id| id >= layout.n_expert as u32)
    {
        return Err(be("invalid resident miss notice mask/id"));
    }
    let mut hits = Vec::new();
    let mut misses = Vec::new();
    for (slot, &id) in words[..count].iter().enumerate() {
        if masks[slot / layout.n_used] & (1 << (slot % layout.n_used)) != 0 {
            hits.push(id);
        } else {
            misses.push(id);
        }
    }
    if misses.is_empty() {
        return Err(be("resident miss notice contains only hits"));
    }
    Ok((masks, hits, misses))
}

fn assert_routes_match_original(
    words: &[u32; crate::recorder::RESIDENT_JOURNAL_WORDS],
    layout: ResidentMoeLayout,
) {
    let original = original_notice_slots(words, layout).map_err(|error| error.to_string());
    let full = resident_notice_slots(words, layout).map_err(|error| error.to_string());
    let misses = resident_notice_misses(words, layout).map_err(|error| error.to_string());
    assert_eq!(full, original);
    assert_eq!(misses, original.map(|(_, _, misses)| misses));
}

#[test]
fn owned_misses_only_preserves_all_rows_duplicates_and_padding_rules() {
    for rows in [1, 2, 4, infr_core::backend::MAX_DECODE_ROWS] {
        for n_used in [1, 8, 16] {
            let layout = ResidentMoeLayout {
                rows,
                n_expert: 512,
                n_used,
            };
            for pattern in [0, 1, 0x55, 0xffff] {
                let mut words = [u32::MAX; crate::recorder::RESIDENT_JOURNAL_WORDS];
                words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 2;
                for (slot, id) in words[..layout.count()].iter_mut().enumerate() {
                    *id = (slot % 3) as u32;
                }
                let valid_mask = (1u32 << n_used) - 1;
                for row in 0..rows {
                    words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + row] =
                        (pattern >> (row % 2)) & valid_mask;
                }
                assert_routes_match_original(&words, layout);
            }
        }
    }
}

#[test]
fn owned_misses_only_keeps_original_errors_and_validates_hit_ids_too() {
    let layout = ResidentMoeLayout {
        rows: 2,
        n_expert: 32,
        n_used: 8,
    };
    let mut words = [0; crate::recorder::RESIDENT_JOURNAL_WORDS];
    words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 2;
    for status in [0, 1, 2, 3, 4, 5, u32::MAX] {
        let mut changed = words;
        changed[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = status;
        assert_routes_match_original(&changed, layout);
    }
    for row in 0..layout.rows {
        let mut changed = words;
        changed[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + row] = 1 << layout.n_used;
        assert_routes_match_original(&changed, layout);
    }
    for slot in 0..layout.count() {
        let mut changed = words;
        changed[slot] = layout.n_expert as u32;
        changed[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + slot / layout.n_used] =
            1 << (slot % layout.n_used);
        assert_routes_match_original(&changed, layout);
    }
    for bad in [
        ResidentMoeLayout { rows: 0, ..layout },
        ResidentMoeLayout {
            rows: crate::recorder::RESIDENT_JOURNAL_MASK_WORDS + 1,
            ..layout
        },
        ResidentMoeLayout {
            n_used: 0,
            ..layout
        },
        ResidentMoeLayout {
            n_used: 17,
            ..layout
        },
        ResidentMoeLayout {
            n_expert: 0,
            ..layout
        },
    ] {
        assert_routes_match_original(&words, bad);
    }
    words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD
        ..crate::recorder::RESIDENT_JOURNAL_MASK_WORD + layout.rows]
        .fill(0xff);
    assert_routes_match_original(&words, layout);
}

#[test]
fn producer_drained_completion_keeps_gpu_dependency_without_a_second_wait() {
    for value in [None, Some(7), Some(u64::MAX - 1)] {
        let (done, receiver) = mpsc::channel();
        let producer_waits = AtomicUsize::new(0);
        let lease = Arc::new(());
        let weak = Arc::downgrade(&lease);
        let result = drain_owned_upload(
            &|seen| {
                assert_eq!(Some(seen), value);
                assert!(weak.upgrade().is_some());
                producer_waits.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
            value,
            lease,
        );
        assert!(weak.upgrade().is_none());
        done.send(result).unwrap();
        let mut completion = OwnedUploadCompletion::new_producer_drained(
            |_| -> Result<()> { panic!("consumer must not repeat a successful producer DMA wait") },
            receiver,
        );
        assert_eq!(completion.wait().unwrap(), value);
        assert_eq!(completion.wait().unwrap(), None);
        drop(completion);
        assert_eq!(
            producer_waits.load(Ordering::Relaxed),
            usize::from(value.is_some())
        );
    }
}

#[test]
fn producer_drained_completion_propagates_failed_wait_and_disconnect() {
    let (done, receiver) = mpsc::channel();
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let result = drain_owned_upload(
        &|_| {
            assert!(weak.upgrade().is_some());
            Err(be("original producer drain failure"))
        },
        Some(19),
        lease,
    );
    assert!(weak.upgrade().is_none());
    done.send(result).unwrap();
    let mut completion = OwnedUploadCompletion::new_producer_drained(
        |_| -> Result<()> { panic!("failed producer completion is not a pending DMA value") },
        receiver,
    );
    assert!(completion
        .wait()
        .unwrap_err()
        .to_string()
        .contains("original producer drain failure"));
    let (done, receiver) = mpsc::channel();
    drop(done);
    let mut completion = OwnedUploadCompletion::new_producer_drained(|_| Ok(()), receiver);
    assert!(completion
        .wait()
        .unwrap_err()
        .to_string()
        .contains("disconnected"));
}

#[test]
fn default_completion_still_retries_failed_dma_on_drop() {
    let (done, receiver) = mpsc::channel();
    let waits = AtomicUsize::new(0);
    let mut completion = OwnedUploadCompletion::new(
        |value| {
            assert_eq!(value, 23);
            if waits.fetch_add(1, Ordering::Relaxed) == 0 {
                Err(be("retry the pending DMA"))
            } else {
                Ok(())
            }
        },
        receiver,
    );
    done.send(Ok(Some(23))).unwrap();
    assert!(completion.wait().is_err());
    assert_eq!(completion.pending_dma, Some(23));
    drop(completion);
    assert_eq!(waits.load(Ordering::Relaxed), 2);
}

#[test]
fn producer_drained_cancel_and_unwind_keep_leases_until_real_wait_returns() {
    for unwind in [false, true] {
        let lease = Arc::new(());
        let weak = Arc::downgrade(&lease);
        let (entered, waiting) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let (done, receiver) = mpsc::channel();
        let (dropping, drop_started) = mpsc::channel();
        let (finished, dropped) = mpsc::channel();
        std::thread::scope(|scope| {
            let producer = scope.spawn(move || {
                let result = drain_owned_upload(
                    &|_| {
                        entered.send(()).unwrap();
                        released.recv().unwrap();
                        Ok(())
                    },
                    Some(31),
                    lease,
                );
                let _ = done.send(result);
            });
            let consumer = scope.spawn(move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _completion = OwnedUploadCompletion::new_producer_drained(
                        |_| -> Result<()> { panic!("consumer repeated producer wait") },
                        receiver,
                    );
                    dropping.send(()).unwrap();
                    if unwind {
                        panic!("synthetic recovery unwind");
                    }
                }));
                assert_eq!(outcome.is_err(), unwind);
                finished.send(()).unwrap();
            });
            waiting.recv_timeout(Duration::from_secs(5)).unwrap();
            drop_started.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(weak.upgrade().is_some());
            assert!(matches!(dropped.try_recv(), Err(mpsc::TryRecvError::Empty)));
            release.send(()).unwrap();
            dropped.recv_timeout(Duration::from_secs(5)).unwrap();
            producer.join().unwrap();
            consumer.join().unwrap();
        });
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn producer_wait_panic_disconnects_without_publishing_success() {
    let (done, receiver) = mpsc::channel();
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let result = drain_owned_upload(
            &|_| -> Result<()> { panic!("producer wait panicked") },
            Some(41),
            lease,
        );
        let _ = done.send(result);
    }));
    assert!(outcome.is_err());
    assert!(weak.upgrade().is_none());
    let mut completion = OwnedUploadCompletion::new_producer_drained(|_| Ok(()), receiver);
    assert!(completion
        .wait()
        .unwrap_err()
        .to_string()
        .contains("disconnected"));
}

#[test]
#[ignore = "manual optimized CPU microbenchmark; route validation/allocation only, no GPU"]
fn d1_notice_routes_original_vs_misses_only_microbenchmark() {
    use std::hint::black_box;
    use std::time::Instant;

    const ITERATIONS: usize = 10_000;
    for rows in [1, 2, 4, infr_core::backend::MAX_DECODE_ROWS] {
        for n_used in [8, 16] {
            for mode in ["one-miss", "mixed", "all-miss"] {
                let layout = ResidentMoeLayout {
                    rows,
                    n_expert: 512,
                    n_used,
                };
                let mut words = [0; crate::recorder::RESIDENT_JOURNAL_WORDS];
                words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 2;
                for (slot, id) in words[..layout.count()].iter_mut().enumerate() {
                    *id = (slot % 7) as u32;
                }
                let full = (1u32 << n_used) - 1;
                words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD
                    ..crate::recorder::RESIDENT_JOURNAL_MASK_WORD + rows]
                    .fill(match mode {
                        "one-miss" => full,
                        "mixed" => 0x5555 & full,
                        _ => 0,
                    });
                if mode == "one-miss" {
                    words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + rows - 1] &=
                        !(1 << (n_used - 1));
                }
                let expected = original_notice_slots(&words, layout).unwrap().2;
                assert_eq!(resident_notice_misses(&words, layout).unwrap(), expected);
                let mut totals = [0u128; 2];
                // ABBA includes allocation/destruction in both arms; no timing assertion.
                for arm in [0, 1, 1, 0] {
                    let started = Instant::now();
                    let mut count = 0;
                    for _ in 0..ITERATIONS {
                        let words = black_box(&words);
                        let layout = black_box(layout);
                        let misses = if arm == 0 {
                            let (_, _, misses) = original_notice_slots(words, layout).unwrap();
                            misses
                        } else {
                            resident_notice_misses(words, layout).unwrap()
                        };
                        count += black_box(misses).len();
                    }
                    totals[arm] += started.elapsed().as_nanos();
                    assert_eq!(count, ITERATIONS * expected.len());
                }
                eprintln!(
                    "D1 rows={rows} used={n_used} mode={mode} iterations_per_arm={} original_ns={} misses_only_ns={}",
                    ITERATIONS * 2, totals[0], totals[1]
                );
            }
        }
    }
}
