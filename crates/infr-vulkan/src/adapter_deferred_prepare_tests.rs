//! CPU-only regressions for deferring preparation until dependent transfers have completed.
use super::command_prefix::DeferredPreparation;
use crate::transfer::parallel_copy_to_mapped;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

const COPY_BYTES: usize = (4 << 20) + 256;
const GUARD: usize = 64;
const WATCHDOG: Duration = Duration::from_secs(10);
const BLOCKED_PROBE: Duration = Duration::from_millis(100);

struct HostCopy {
    source: Vec<u8>,
    destination: Vec<u8>,
    pattern: u8,
}

impl HostCopy {
    fn new(pattern: u8) -> Self {
        let mut source = vec![0x3c; COPY_BYTES + 2 * GUARD];
        source[GUARD..GUARD + COPY_BYTES].fill(pattern);
        Self {
            source,
            destination: vec![0xa5; COPY_BYTES + 2 * GUARD],
            pattern,
        }
    }

    fn copy(&mut self) {
        // Ordinary, disjoint CPU allocations stand in for coherent mapped source/target ranges.
        // Each range exceeds the production helper's 4 MiB parallel-copy threshold.
        let destination = self.destination[GUARD..GUARD + COPY_BYTES].as_mut_ptr();
        parallel_copy_to_mapped(&self.source[GUARD..GUARD + COPY_BYTES], destination);
    }

    fn verify(&self) {
        assert!(self.source[..GUARD].iter().all(|&byte| byte == 0x3c));
        assert!(self.source[GUARD + COPY_BYTES..]
            .iter()
            .all(|&byte| byte == 0x3c));
        assert!(self.destination[..GUARD].iter().all(|&byte| byte == 0xa5));
        assert!(self.destination[GUARD + COPY_BYTES..]
            .iter()
            .all(|&byte| byte == 0xa5));
        assert!(self.source[GUARD..GUARD + COPY_BYTES]
            .iter()
            .all(|&byte| byte == self.pattern));
        assert_eq!(
            self.destination[GUARD..GUARD + COPY_BYTES],
            self.source[GUARD..GUARD + COPY_BYTES]
        );
    }
}

fn copy_pair() -> [HostCopy; 2] {
    let mut left = HostCopy::new(0x17);
    let mut right = HostCopy::new(0x83);
    rayon::join(|| left.copy(), || right.copy());
    [left, right]
}

#[test]
fn deferred_preparation_allows_external_parallel_copy_before_tail() {
    for workers in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let copied = AtomicBool::new(false);
        let recorded = AtomicUsize::new(0);
        let early = AtomicBool::new(false);
        let (done, completed) = mpsc::channel();
        let (finished, before_tail, copies) = std::thread::scope(|threads| {
            // The controller stays outside the pool: blocking it must not occupy a Rayon worker.
            pool.in_place_scope(|scope| {
                let mut preparation = DeferredPreparation::new(Some(|| {
                    if !copied.load(Ordering::Acquire) {
                        early.store(true, Ordering::Release);
                    }
                    recorded.fetch_add(1, Ordering::AcqRel);
                }));
                let producer = threads.spawn(|| {
                    let copies = pool.install(copy_pair);
                    copied.store(true, Ordering::Release);
                    let _ = done.send(());
                    copies
                });
                let finished = completed.recv_timeout(WATCHDOG).is_ok();
                let before_tail = recorded.load(Ordering::Acquire);
                if finished {
                    preparation.start(scope);
                    preparation.start(scope);
                }
                (finished, before_tail, producer.join())
            })
        });
        assert!(
            finished,
            "{workers} worker(s): transfer did not complete before tail"
        );
        assert_eq!(before_tail, 0);
        assert_eq!(recorded.load(Ordering::Acquire), 1);
        assert!(!early.load(Ordering::Acquire));
        for copy in copies.expect("external copy producer panicked") {
            copy.verify();
        }
    }
}

#[test]
fn eager_readiness_wait_blocks_external_copy_until_gate_is_released() {
    for workers in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let (spare_entered, spare_started) = mpsc::channel();
        let (release_spare, spare_gate) = mpsc::channel();
        let (spare_exited, spare_exit) = mpsc::channel();
        let (eager_entered, eager_started) = mpsc::channel();
        let (release_eager, eager_gate) = mpsc::channel();
        let (eager_exited, eager_exit) = mpsc::channel();
        let (attempting, attempted) = mpsc::channel();
        let (done, completed) = mpsc::channel();

        let (spare_ready, eager_ready, producer_ready, blocked, recovered, copies) =
            std::thread::scope(|threads| {
                pool.in_place_scope(|scope| {
                    // Occupy the spare worker so both pool sizes leave exactly one worker for
                    // the old eagerly queued readiness waiter and the external upload.
                    if workers == 2 {
                        scope.spawn(move |_| {
                            let _ = spare_entered.send(());
                            let released = spare_gate.recv_timeout(WATCHDOG).is_ok();
                            let _ = spare_exited.send(released);
                        });
                    }
                    let spare_ready = workers == 1 || spare_started.recv_timeout(WATCHDOG).is_ok();
                    scope.spawn(move |_| {
                        let _ = eager_entered.send(());
                        let released = eager_gate.recv_timeout(WATCHDOG).is_ok();
                        let _ = eager_exited.send(released);
                    });
                    let eager_ready = eager_started.recv_timeout(WATCHDOG).is_ok();
                    let producer = threads.spawn(|| {
                        let _ = attempting.send(());
                        let copies = pool.install(copy_pair);
                        let _ = done.send(());
                        copies
                    });
                    let producer_ready = attempted.recv_timeout(WATCHDOG).is_ok();
                    let probe = completed.recv_timeout(BLOCKED_PROBE);
                    let blocked = matches!(probe, Err(mpsc::RecvTimeoutError::Timeout));

                    // Release every gate before joining or asserting, even if any watchdog fired.
                    let _ = release_eager.send(());
                    let _ = release_spare.send(());
                    let recovered = probe.is_ok() || completed.recv_timeout(WATCHDOG).is_ok();
                    (
                        spare_ready,
                        eager_ready,
                        producer_ready,
                        blocked,
                        recovered,
                        producer.join(),
                    )
                })
            });
        let eager_released = eager_exit.recv_timeout(WATCHDOG).unwrap_or(false);
        let spare_released = workers == 1 || spare_exit.recv_timeout(WATCHDOG).unwrap_or(false);
        assert!(
            spare_ready && eager_ready && producer_ready,
            "negative-control setup timed out"
        );
        assert!(
            blocked,
            "{workers} worker(s): eager waiter must prevent upload progress"
        );
        assert!(
            recovered,
            "upload must finish after releasing readiness gates"
        );
        assert!(
            eager_released && spare_released,
            "a worker escaped through its watchdog"
        );
        for copy in copies.expect("external copy producer panicked") {
            copy.verify();
        }
    }
}

struct DropProbe(Arc<AtomicUsize>);

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

#[test]
fn deferred_cancellation_drops_unstarted_work_and_joins_started_resources() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let ran = AtomicUsize::new(0);
    let probe = DropProbe(Arc::clone(&dropped));
    let ran_ref = &ran;
    let preparation = DeferredPreparation::new(Some(move || {
        ran_ref.fetch_add(1, Ordering::AcqRel);
        drop(probe);
    }));
    drop(preparation);
    assert_eq!(ran.load(Ordering::Acquire), 0);
    assert_eq!(dropped.load(Ordering::Acquire), 1);

    for workers in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        let cancelled = AtomicBool::new(false);
        let published = AtomicUsize::new(0);
        let observed = AtomicUsize::new(0);
        let borrowed = [37usize; 8];
        let probe = DropProbe(Arc::clone(&dropped));
        let (entered, started) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let (exited, exit) = mpsc::channel();
        let cancelled_ref = &cancelled;
        let published_ref = &published;
        let observed_ref = &observed;
        let borrowed_ref = &borrowed;
        let (started_ok, drops_before_release) = pool.in_place_scope(|scope| {
            let mut preparation = DeferredPreparation::new(Some(move || {
                let _ = entered.send(());
                let released = gate.recv_timeout(WATCHDOG).is_ok();
                observed_ref.store(borrowed_ref[0], Ordering::Release);
                if !cancelled_ref.load(Ordering::Acquire) {
                    published_ref.fetch_add(1, Ordering::AcqRel);
                }
                drop(probe);
                let _ = exited.send(released);
            }));
            preparation.start(scope);
            let started_ok = started.recv_timeout(WATCHDOG).is_ok();
            cancelled.store(true, Ordering::Release);
            let drops_before_release = dropped.load(Ordering::Acquire);
            let _ = release.send(());
            (started_ok, drops_before_release)
        });
        let released = exit.recv_timeout(WATCHDOG).unwrap_or(false);
        assert!(
            started_ok && released,
            "started-work lifetime handshake timed out"
        );
        assert_eq!(drops_before_release, 0);
        assert_eq!(dropped.load(Ordering::Acquire), 1);
        assert_eq!(observed.load(Ordering::Acquire), borrowed[0]);
        assert_eq!(published.load(Ordering::Acquire), 0);
    }
}
