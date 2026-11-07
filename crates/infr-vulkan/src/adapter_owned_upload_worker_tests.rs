use super::super::{drain_owned_upload, OwnedUploadCompletion};
use super::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

thread_local! {
    static PENDING_WAIT_ENTRY: RefCell<Option<mpsc::Sender<()>>> = const { RefCell::new(None) };
}

pub(super) fn pending_wait_entered() {
    PENDING_WAIT_ENTRY.with(|entry| {
        if let Some(entered) = entry.borrow_mut().take() {
            let _ = entered.send(());
        }
    });
}

fn job_ack() -> (Arc<NoticeProducerAck>, mpsc::Receiver<()>) {
    let (done, receiver) = mpsc::channel();
    (Arc::new(NoticeProducerAck(done)), receiver)
}

fn no_consumer_dma_wait(_: u64) -> Result<()> {
    panic!("consumer must retain the dependency value without a duplicate DMA wait")
}

#[test]
fn old_skip_jobs_advance_during_upload_but_original_ack_waits() {
    let mut worker = OwnedUploadWorker::new(()).unwrap();
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let (entered, waiting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let (completed, completion) = mpsc::channel();
    let (ack, acknowledged) = job_ack();
    worker.schedule(
        &(),
        move |_| {
            drain_owned_upload(
                &|value| {
                    assert_eq!(value, 37);
                    entered.send(()).unwrap();
                    released
                        .recv_timeout(TIMEOUT)
                        .map_err(|e| be(e.to_string()))?;
                    Ok(())
                },
                Some(37),
                lease,
            )
        },
        completed,
        Arc::clone(&ack),
        None,
    );
    waiting.recv_timeout(TIMEOUT).unwrap();
    // Model the Started job ending and subsequent jobs being observed on this
    // same thread. Only their SKIP path can advance before upload finishes.
    drop(ack);
    let (results, observed) = mpsc::channel();
    for ordinal in 1..=8 {
        let (skip_ack, skip_done) = job_ack();
        worker.before_notice(0).unwrap();
        results.send(ordinal).unwrap();
        drop(skip_ack);
        skip_done.recv_timeout(TIMEOUT).unwrap();
    }
    assert_eq!(
        observed.try_iter().collect::<Vec<_>>(),
        (1..=8).collect::<Vec<_>>()
    );
    assert!(weak.upgrade().is_some());
    assert!(matches!(
        acknowledged.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    assert!(matches!(
        completion.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    release.send(()).unwrap();
    let mut upload = OwnedUploadCompletion::new_producer_drained(no_consumer_dma_wait, completion);
    assert_eq!(upload.wait().unwrap(), Some(37));
    assert!(weak.upgrade().is_none());
    assert_eq!(upload.wait().unwrap(), None);
    acknowledged.recv_timeout(TIMEOUT).unwrap();
    assert!(matches!(
        acknowledged.try_recv(),
        Err(mpsc::TryRecvError::Disconnected)
    ));
    worker.wait_idle().unwrap();
}

#[test]
fn upload_finishing_first_does_not_ack_an_active_observer() {
    let mut worker = OwnedUploadWorker::new(()).unwrap();
    let (ack, acknowledged) = job_ack();
    let (completed, completion) = mpsc::channel();
    worker.schedule(&(), |_| Ok(None), completed, Arc::clone(&ack), None);
    assert_eq!(completion.recv_timeout(TIMEOUT).unwrap().unwrap(), None);
    worker.wait_idle().unwrap();
    assert!(matches!(
        acknowledged.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop(ack);
    acknowledged.recv_timeout(TIMEOUT).unwrap();
}

#[test]
fn every_non_skip_notice_waits_for_upload_retirement() {
    for status in [1, 2, 4, 5, u32::MAX] {
        let mut worker = OwnedUploadWorker::new(()).unwrap();
        let (release, released) = mpsc::channel();
        let (entered, waiting) = mpsc::channel();
        let (completed, completion) = mpsc::channel();
        let (ack, acknowledged) = job_ack();
        worker.schedule(
            &(),
            move |_| {
                entered.send(()).unwrap();
                released
                    .recv_timeout(TIMEOUT)
                    .map_err(|e| be(e.to_string()))?;
                Ok(None)
            },
            completed,
            ack,
            None,
        );
        waiting.recv_timeout(TIMEOUT).unwrap();
        let (wait_entry, wait_entered) = mpsc::channel();
        let (finished, checked) = mpsc::channel();
        let observer = std::thread::spawn(move || {
            PENDING_WAIT_ENTRY.with(|entry| {
                assert!(entry.borrow_mut().replace(wait_entry).is_none());
            });
            worker.before_notice(status).unwrap();
            finished.send(()).unwrap();
        });
        // The checkpoint is inside wait_idle's Some(pending) branch, not before
        // before_notice. Skipping the wait call must fail this receive, regardless
        // of whether the observer has been scheduled before the following probe.
        let entered = wait_entered.recv_timeout(TIMEOUT);
        let premature_return = checked.try_recv();
        // Always release and join before asserting, including a missing checkpoint.
        let released = release.send(());
        let joined = observer.join();
        assert!(
            entered.is_ok(),
            "status={status} never entered the pending wait: {entered:?}"
        );
        assert!(matches!(premature_return, Err(mpsc::TryRecvError::Empty)));
        released.unwrap();
        joined.unwrap();
        checked.recv_timeout(TIMEOUT).unwrap();
        completion.recv_timeout(TIMEOUT).unwrap().unwrap();
        acknowledged.recv_timeout(TIMEOUT).unwrap();
    }
}

#[test]
fn completion_failure_and_panic_release_captures_and_worker_is_reused() {
    let mut worker = OwnedUploadWorker::new(()).unwrap();
    let mut thread_ids = Vec::new();
    for mode in 0..3 {
        let lease = Arc::new(());
        let weak = Arc::downgrade(&lease);
        let (seen, observed) = mpsc::channel();
        let (completed, completion) = mpsc::channel();
        let (ack, acknowledged) = job_ack();
        worker.schedule(
            &(),
            move |_| {
                seen.send(std::thread::current().id()).unwrap();
                // A mock wait failure preserves the existing D3 cleanup contract;
                // this test does not establish safety of a real failed GPU wait.
                drain_owned_upload(
                    &|_| match mode {
                        0 => Err(be("synthetic upload completion failure")),
                        1 => panic!("synthetic upload panic"),
                        _ => Ok(()),
                    },
                    Some(53),
                    lease,
                )
            },
            completed,
            ack,
            None,
        );
        thread_ids.push(observed.recv_timeout(TIMEOUT).unwrap());
        let result = completion.recv_timeout(TIMEOUT).unwrap();
        match mode {
            0 => assert!(result
                .unwrap_err()
                .to_string()
                .contains("synthetic upload completion failure")),
            1 => assert!(result.unwrap_err().to_string().contains("panicked")),
            _ => assert_eq!(result.unwrap(), Some(53)),
        }
        assert!(weak.upgrade().is_none());
        acknowledged.recv_timeout(TIMEOUT).unwrap();
        worker.wait_idle().unwrap();
    }
    assert!(thread_ids.iter().all(|id| *id == thread_ids[0]));
    assert_ne!(thread_ids[0], std::thread::current().id());
}

#[test]
fn cancelled_receiver_does_not_cancel_running_upload_or_release_lease() {
    let mut worker = OwnedUploadWorker::new(()).unwrap();
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let (entered, waiting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let (completed, completion) = mpsc::channel();
    let (ack, acknowledged) = job_ack();
    worker.schedule(
        &(),
        move |_| {
            let _lease = lease;
            entered.send(()).unwrap();
            released
                .recv_timeout(TIMEOUT)
                .map_err(|e| be(e.to_string()))?;
            Ok(Some(59))
        },
        completed,
        ack,
        None,
    );
    waiting.recv_timeout(TIMEOUT).unwrap();
    drop(completion);
    assert!(weak.upgrade().is_some());
    assert!(matches!(
        acknowledged.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    release.send(()).unwrap();
    worker.wait_idle().unwrap();
    assert!(weak.upgrade().is_none());
    acknowledged.recv_timeout(TIMEOUT).unwrap();
}

#[test]
fn completion_drop_and_consumer_unwind_wait_for_real_upload() {
    for unwind in [false, true] {
        let mut worker = OwnedUploadWorker::new(()).unwrap();
        let lease = Arc::new(());
        let weak = Arc::downgrade(&lease);
        let (entered, waiting) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let (completed, completion) = mpsc::channel();
        let (ack, acknowledged) = job_ack();
        worker.schedule(
            &(),
            move |_| {
                drain_owned_upload(
                    &|_| {
                        entered.send(()).unwrap();
                        released
                            .recv_timeout(TIMEOUT)
                            .map_err(|e| be(e.to_string()))?;
                        Ok(())
                    },
                    Some(61),
                    lease,
                )
            },
            completed,
            ack,
            None,
        );
        waiting.recv_timeout(TIMEOUT).unwrap();
        let (dropping, drop_started) = mpsc::channel();
        let (finished, dropped) = mpsc::channel();
        let consumer = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _upload =
                    OwnedUploadCompletion::new_producer_drained(no_consumer_dma_wait, completion);
                dropping.send(()).unwrap();
                if unwind {
                    panic!("synthetic consumer unwind");
                }
            }));
            assert_eq!(result.is_err(), unwind);
            finished.send(()).unwrap();
        });
        drop_started.recv_timeout(TIMEOUT).unwrap();
        assert!(weak.upgrade().is_some());
        assert!(matches!(dropped.try_recv(), Err(mpsc::TryRecvError::Empty)));
        release.send(()).unwrap();
        dropped.recv_timeout(TIMEOUT).unwrap();
        consumer.join().unwrap();
        worker.wait_idle().unwrap();
        assert!(weak.upgrade().is_none());
        acknowledged.recv_timeout(TIMEOUT).unwrap();
    }
}

#[test]
fn worker_drop_joins_accepted_work_and_retains_executor() {
    let executor = Arc::new(());
    let executor_weak = Arc::downgrade(&executor);
    let mut worker = OwnedUploadWorker::new(Arc::clone(&executor)).unwrap();
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let (entered, waiting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let (completed, completion) = mpsc::channel();
    let (ack, acknowledged) = job_ack();
    worker.schedule(
        &executor,
        move |_| {
            let _lease = lease;
            entered.send(()).unwrap();
            released
                .recv_timeout(TIMEOUT)
                .map_err(|e| be(e.to_string()))?;
            Ok(None)
        },
        completed,
        ack,
        None,
    );
    waiting.recv_timeout(TIMEOUT).unwrap();
    drop(executor);
    drop(completion);
    let (dropping, drop_started) = mpsc::channel();
    let (finished, dropped) = mpsc::channel();
    let shutdown = std::thread::spawn(move || {
        dropping.send(()).unwrap();
        drop(worker);
        finished.send(()).unwrap();
    });
    drop_started.recv_timeout(TIMEOUT).unwrap();
    assert!(executor_weak.upgrade().is_some());
    assert!(weak.upgrade().is_some());
    assert!(matches!(dropped.try_recv(), Err(mpsc::TryRecvError::Empty)));
    assert!(matches!(
        acknowledged.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    release.send(()).unwrap();
    dropped.recv_timeout(TIMEOUT).unwrap();
    shutdown.join().unwrap();
    assert!(executor_weak.upgrade().is_none());
    assert!(weak.upgrade().is_none());
    acknowledged.recv_timeout(TIMEOUT).unwrap();
}

#[test]
fn disabled_and_failed_worker_creation_keep_inline_fallback_available() {
    let calls = AtomicUsize::new(0);
    let disabled = OwnedUploadWorker::<()>::optional(false, || {
        calls.fetch_add(1, Ordering::Relaxed);
        panic!("MTP/legacy must not create an upload worker");
    });
    assert!(disabled.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let failed = OwnedUploadWorker::<()>::optional(true, || {
        calls.fetch_add(1, Ordering::Relaxed);
        Err(be("synthetic worker spawn failure"))
    });
    assert!(failed.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn full_or_disconnected_queue_falls_back_inline_exactly_once() {
    for full in [false, true] {
        let (sender, jobs) = mpsc::sync_channel(1);
        let mut receiver = Some(jobs);
        let mut filler_completion = None;
        if full {
            // Synthetic full queue: normal schedule waits for its previous task,
            // so this defensive fallback cannot arise from ordinary operation.
            let (completed, completion) = mpsc::channel();
            filler_completion = Some(completion);
            let (ack, _) = job_ack();
            assert!(sender
                .try_send(UploadJob {
                    work: Some(Box::new(|_: &usize| -> Outcome {
                        panic!("queued filler must not execute")
                    })),
                    completed: Some(completed),
                    acknowledgement: Some(ack),
                    retired: None,
                    context: None,
                })
                .is_ok());
        } else {
            drop(receiver.take());
        }
        let mut worker = OwnedUploadWorker {
            sender: Some(sender),
            pending: None,
            thread: None,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let ran = Arc::clone(&calls);
        let caller = std::thread::current().id();
        let (completed, completion) = mpsc::channel();
        let (ack, acknowledged) = job_ack();
        worker.schedule(
            &71,
            move |executor| {
                assert_eq!(*executor, 71);
                assert_eq!(std::thread::current().id(), caller);
                ran.fetch_add(1, Ordering::Relaxed);
                Ok(Some(*executor as u64))
            },
            completed,
            ack,
            None,
        );
        assert_eq!(completion.recv_timeout(TIMEOUT).unwrap().unwrap(), Some(71));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        acknowledged.recv_timeout(TIMEOUT).unwrap();
        worker.wait_idle().unwrap();
        drop(worker);
        drop(receiver);
        if let Some(completion) = filler_completion {
            assert!(completion.recv_timeout(TIMEOUT).unwrap().is_err());
        }
    }
}

#[test]
fn unexecuted_job_drop_publishes_error_after_releasing_captures() {
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let (completed, completion) = mpsc::channel();
    let (retired, retirement) = mpsc::channel();
    let (ack, acknowledged) = job_ack();
    let job = UploadJob {
        work: Some(Box::new(move |_: &()| -> Outcome {
            let _lease = lease;
            panic!("cancelled job must not run")
        })),
        completed: Some(completed),
        acknowledgement: Some(ack),
        retired: Some(retired),
        context: None,
    };
    drop(job);
    assert!(weak.upgrade().is_none());
    assert!(completion
        .recv_timeout(TIMEOUT)
        .unwrap()
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    acknowledged.recv_timeout(TIMEOUT).unwrap();
    retirement.recv_timeout(TIMEOUT).unwrap();
}

#[test]
fn started_send_failure_releases_blocking_completion_before_unsent_notice_drop() {
    let (started, receiver) = mpsc::channel();
    drop(receiver);
    let (completed, completion) = mpsc::channel();
    let upload = OwnedUploadCompletion::new_producer_drained(no_consumer_dma_wait, completion);
    let unsent = match started.send(upload) {
        Err(unsent) => unsent,
        Ok(()) => panic!("Started unexpectedly delivered"),
    };
    // This is the existing producer send-failure ordering, before worker.schedule.
    assert!(completed
        .send(Err(be("owned upload cancelled before Started delivery")))
        .is_ok());
    drop(unsent);
}

#[test]
fn upload_context_is_carried_per_job_and_does_not_leak_on_reuse() {
    fn key(context: Option<infr_core::timeline::Context>) -> Option<(u64, usize, i32)> {
        context.map(|context| (context.step, context.lanes, context.layer))
    }
    let mut worker = OwnedUploadWorker::new(()).unwrap();
    for context in [
        Some(infr_core::timeline::Context {
            step: 101,
            lanes: 8,
            layer: 17,
        }),
        None,
        Some(infr_core::timeline::Context {
            step: 102,
            lanes: 1,
            layer: 3,
        }),
    ] {
        let expected = key(context);
        let (observed, received) = mpsc::channel();
        let (completed, completion) = mpsc::channel();
        let (ack, acknowledged) = job_ack();
        worker.schedule(
            &(),
            move |_| {
                observed
                    .send((
                        infr_core::timeline::enabled(),
                        key(infr_core::timeline::context()),
                    ))
                    .unwrap();
                Ok(None)
            },
            completed,
            ack,
            context,
        );
        let (enabled, observed) = received.recv_timeout(TIMEOUT).unwrap();
        // Do not change process-global timeline settings in a CPU-only test.
        assert_eq!(observed, if enabled { expected } else { None });
        completion.recv_timeout(TIMEOUT).unwrap().unwrap();
        acknowledged.recv_timeout(TIMEOUT).unwrap();
        worker.wait_idle().unwrap();
    }
}
