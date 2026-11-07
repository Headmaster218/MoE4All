//! CPU ownership/locking model plus opt-in real queue lifetime regressions.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex, Weak};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

fn empty_queue() -> DedicatedTransferQueue {
    DedicatedTransferQueue {
        queue: vk::Queue::null(),
        family_index: 0,
        pool: vk::CommandPool::null(),
        timeline: vk::Semaphore::null(),
        slots: (0..DEDICATED_TRANSFER_COMMAND_SLOTS)
            .map(|_| DedicatedTransferCommandSlot {
                cmd: vk::CommandBuffer::null(),
                pending_value: 0,
                keepalive: Vec::new(),
                profile_pending: false,
                profile_bytes: 0,
                timeline_query: None,
            })
            .collect(),
        cursor: 0,
        next_value: 1,
        profile_query_pool: vk::QueryPool::null(),
        profile_timestamp_period_ns: 1.0,
        profile_timestamp_valid_bits: 64,
    }
}

struct Model {
    queue: Mutex<DedicatedTransferQueue>,
    pending: AtomicBool,
    queries: AtomicUsize,
}

impl Model {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(empty_queue()),
            pending: AtomicBool::new(false),
            queries: AtomicUsize::new(0),
        })
    }

    fn publish(self: &Arc<Self>, slot: usize, value: u64) -> (Weak<dyn Buffer>, Arc<AtomicBool>) {
        let unlocked = Arc::new(AtomicBool::new(false));
        let buffer: Arc<dyn Buffer> = Arc::new(ModelBuffer {
            owner: Arc::clone(self),
            dropped_outside_lock: Arc::clone(&unlocked),
        });
        let weak = Arc::downgrade(&buffer);
        let mut queue = self.queue.lock().unwrap();
        self.pending.store(true, Ordering::Release);
        assert!(queue.slots[slot].keepalive.is_empty());
        queue.slots[slot].pending_value = value;
        queue.slots[slot].keepalive.push(buffer);
        self.pending
            .store(queue.has_pending_keepalive(), Ordering::Release);
        (weak, unlocked)
    }

    // Models the success/query guard in Shared; the detach operation is production code.
    fn probe(&self, observed: std::result::Result<u64, vk::Result>) {
        if !self.pending.load(Ordering::Acquire) {
            return;
        }
        let retired = {
            let Ok(mut queue) = self.queue.try_lock() else {
                return;
            };
            if !queue.has_pending_keepalive() {
                self.pending.store(false, Ordering::Release);
                return;
            }
            self.queries.fetch_add(1, Ordering::Relaxed);
            let Ok(completed) = observed else { return };
            let retired = queue.take_completed_keepalive(completed);
            self.pending
                .store(queue.has_pending_keepalive(), Ordering::Release);
            retired
        };
        drop(retired);
    }

    fn through(&self, completed: u64) {
        if completed == 0 || !self.pending.load(Ordering::Acquire) {
            return;
        }
        let retired = {
            let Ok(mut queue) = self.queue.try_lock() else {
                return;
            };
            let retired = queue.take_completed_keepalive(completed);
            self.pending
                .store(queue.has_pending_keepalive(), Ordering::Release);
            retired
        };
        drop(retired);
    }

    fn waited(&self, completed: u64) {
        if completed == 0 || !self.pending.load(Ordering::Acquire) {
            return;
        }
        let retired = {
            let mut queue = self.queue.lock().unwrap();
            let retired = queue.take_completed_keepalive(completed);
            self.pending
                .store(queue.has_pending_keepalive(), Ordering::Release);
            retired
        };
        drop(retired);
    }
}

struct ModelBuffer {
    owner: Arc<Model>,
    dropped_outside_lock: Arc<AtomicBool>,
}

impl Buffer for ModelBuffer {
    fn len_bytes(&self) -> usize {
        64
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for ModelBuffer {
    fn drop(&mut self) {
        let unlocked = match self.owner.queue.try_lock() {
            Ok(_) | Err(std::sync::TryLockError::Poisoned(_)) => true,
            Err(std::sync::TryLockError::WouldBlock) => false,
        };
        self.dropped_outside_lock.store(unlocked, Ordering::Release);
    }
}

#[test]
fn success_retires_all_proven_slots_and_breaks_shared_cycle_outside_lock() {
    let model = Model::new();
    let shared = Arc::downgrade(&model);
    let buffers = [
        model.publish(0, 3),
        model.publish(1, 1),
        model.publish(2, 2),
    ];
    assert!(buffers.iter().all(|(weak, _)| weak.upgrade().is_some()));
    model.probe(Ok(3));
    assert!(!model.pending.load(Ordering::Acquire));
    assert!(buffers
        .iter()
        .all(|(weak, unlocked)| { weak.upgrade().is_none() && unlocked.load(Ordering::Acquire) }));
    drop(model);
    assert!(shared.upgrade().is_none());
}

#[test]
fn completed_prefix_preserves_other_slots_and_timestamp_state() {
    let model = Model::new();
    let first = model.publish(0, 4);
    let second = model.publish(1, 7);
    let third = model.publish(2, 9);
    {
        let mut queue = model.queue.lock().unwrap();
        for (i, slot) in queue.slots.iter_mut().enumerate() {
            slot.profile_pending = true;
            slot.profile_bytes = 100 + i as u64;
        }
    }
    model.probe(Ok(7));
    assert!(first.0.upgrade().is_none() && second.0.upgrade().is_none());
    assert!(third.0.upgrade().is_some());
    assert!(model.pending.load(Ordering::Acquire));
    {
        let queue = model.queue.lock().unwrap();
        assert_eq!(
            queue
                .slots
                .iter()
                .map(|s| s.pending_value)
                .collect::<Vec<_>>(),
            [4, 7, 9]
        );
        for (i, slot) in queue.slots.iter().enumerate() {
            assert!(slot.profile_pending);
            assert_eq!(slot.profile_bytes, 100 + i as u64);
        }
    }
    model.probe(Ok(9));
    assert!(third.0.upgrade().is_none());
}

#[test]
fn stale_completion_cannot_retire_a_reused_slot() {
    let model = Model::new();
    let old = model.publish(0, 1);
    model.through(1);
    assert!(old.0.upgrade().is_none());
    let new = model.publish(0, 5);
    model.through(1);
    assert!(new.0.upgrade().is_some());
    assert!(model.pending.load(Ordering::Acquire));
    model.through(5);
    assert!(new.0.upgrade().is_none());
    assert_eq!(model.queries.load(Ordering::Relaxed), 0);
}

#[test]
fn zero_fence_dependency_skips_pending_copies_without_counter_query() {
    let model = Model::new();
    let earlier = model.publish(0, 1);
    let later = model.publish(1, 2);
    model.through(0);
    assert!(earlier.0.upgrade().is_some() && later.0.upgrade().is_some());
    assert!(model.pending.load(Ordering::Acquire));
    model.through(1);
    assert!(earlier.0.upgrade().is_none());
    assert!(later.0.upgrade().is_some());
    assert!(model.pending.load(Ordering::Acquire));
    model.waited(2);
    assert!(later.0.upgrade().is_none());
    assert!(!model.pending.load(Ordering::Acquire));
    assert_eq!(model.queries.load(Ordering::Relaxed), 0);
}

#[test]
fn zero_or_failed_observation_does_not_retire_unproven_resources() {
    let model = Model::new();
    let buffer = model.publish(0, 2);
    for observed in [
        Ok(0),
        Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
        Err(vk::Result::ERROR_DEVICE_LOST),
    ] {
        model.probe(observed);
        assert!(buffer.0.upgrade().is_some());
        assert!(model.pending.load(Ordering::Acquire));
        assert!(!buffer.1.load(Ordering::Acquire));
    }
    model.probe(Ok(2));
    assert!(buffer.0.upgrade().is_none());
}

#[test]
fn empty_and_repeated_retirement_skip_queries() {
    let model = Model::new();
    model.probe(Ok(u64::MAX));
    assert_eq!(model.queries.load(Ordering::Relaxed), 0);
    model.pending.store(true, Ordering::Release);
    model.probe(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    assert!(!model.pending.load(Ordering::Acquire));
    assert_eq!(model.queries.load(Ordering::Relaxed), 0);
    let buffer = model.publish(0, 1);
    model.probe(Ok(1));
    assert!(buffer.0.upgrade().is_none());
    for _ in 0..8 {
        model.probe(Ok(1));
    }
    assert_eq!(model.queries.load(Ordering::Relaxed), 1);
}

#[test]
fn new_submission_after_last_retirement_republishes_pending() {
    let model = Model::new();
    let old = model.publish(0, 1);
    let (locked, lock_seen) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let retire_owner = Arc::clone(&model);
    let retiring = std::thread::spawn(move || {
        let mut retired: RetiredTransferKeepalive = std::array::from_fn(|_| Vec::new());
        {
            let mut queue = retire_owner.queue.lock().unwrap();
            for (owners, completed) in retired.iter_mut().zip(queue.take_completed_keepalive(1)) {
                *owners = completed;
            }
            retire_owner
                .pending
                .store(queue.has_pending_keepalive(), Ordering::Release);
            locked.send(()).unwrap();
            released.recv_timeout(TIMEOUT).unwrap();
        }
        drop(retired);
    });
    lock_seen.recv_timeout(TIMEOUT).unwrap();
    let producer_owner = Arc::clone(&model);
    let producing = std::thread::spawn(move || producer_owner.publish(0, 2));
    release.send(()).unwrap();
    retiring.join().unwrap();
    let new = producing.join().unwrap();
    assert!(old.0.upgrade().is_none());
    assert!(model.pending.load(Ordering::Acquire));
    model.probe(Ok(1));
    assert!(new.0.upgrade().is_some());
    model.probe(Ok(2));
    assert!(new.0.upgrade().is_none());
}

#[test]
fn contended_gpu_cleanup_never_waits_and_preserves_pending_for_later_boundary() {
    let model = Model::new();
    let buffer = model.publish(0, 1);
    let guard = model.queue.lock().unwrap();
    let worker_owner = Arc::clone(&model);
    let (returned, result) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        worker_owner.through(1);
        returned.send(()).unwrap();
    });
    let nonblocking = result.recv_timeout(TIMEOUT);
    // Always release/join before assertions, even if a regression used blocking lock().
    drop(guard);
    worker.join().unwrap();
    assert!(
        nonblocking.is_ok(),
        "opportunistic cleanup waited for the queue mutex"
    );
    assert!(model.pending.load(Ordering::Acquire));
    assert_eq!(model.queries.load(Ordering::Relaxed), 0);
    assert!(buffer.0.upgrade().is_some());
    model.waited(1);
    assert!(buffer.0.upgrade().is_none());
    assert!(buffer.1.load(Ordering::Acquire));
    assert!(!model.pending.load(Ordering::Acquire));
}

#[test]
fn submission_that_wins_queue_lock_is_not_erased_by_older_proof() {
    let model = Model::new();
    let old = model.publish(0, 1);
    let new = model.publish(1, 2);
    model.probe(Ok(1));
    assert!(old.0.upgrade().is_none());
    assert!(new.0.upgrade().is_some());
    assert!(model.pending.load(Ordering::Acquire));
    model.probe(Ok(2));
    assert!(new.0.upgrade().is_none());
}

#[test]
fn unwind_after_proven_detach_still_drops_owners_outside_queue_lock() {
    let model = Model::new();
    let buffer = model.publish(0, 1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut _retired: RetiredTransferKeepalive = std::array::from_fn(|_| Vec::new());
        {
            let mut queue = model.queue.lock().unwrap();
            _retired[0] = std::mem::take(&mut queue.slots[0].keepalive);
            // Models submit's reuse path AFTER its successful wait, before a later error.
            panic!("synthetic recording failure after successful slot wait");
        }
    }));
    assert!(result.is_err());
    assert!(buffer.0.upgrade().is_none());
    assert!(buffer.1.load(Ordering::Acquire));
    // The lock is poisoned, but it must be unlocked during buffer destruction.
    assert!(model.queue.try_lock().is_err());
}

fn gpu_backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(
        std::env::var("MOE4ALL_TEST_DEVICE")
            .expect("central runner must explicitly select the idle GPU"),
    );
    cfg.paging.host_dma = true;
    cfg.paging.expert_prefetch = false;
    cfg.prof.ops = false;
    cfg.prof.stages = false;
    cfg.prof.pager_profile = false;
    let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(
        backend.shared.dedicated_transfer.is_some(),
        "test requires a dedicated transfer queue"
    );
    assert!(
        backend.shared.host_overflow_type.is_some(),
        "test requires host-visible staging"
    );
    backend
}

const PAYLOAD: usize = 64 * 1024;

fn gpu_batch(backend: &VulkanBackend, salt: u8) -> (TransferCopyBatch, Vec<u8>) {
    let bytes = (0..PAYLOAD)
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(salt))
        .collect::<Vec<_>>();
    let src: Arc<dyn Buffer> =
        Arc::new(make_dedicated_host_transfer_buffer(&backend.shared, PAYLOAD).unwrap());
    copy_to_mapped(
        &bytes,
        as_vk_buf(src.as_ref()).unwrap().mapped_ptr().unwrap(),
    );
    let (dst, _) = backend.alloc_device_local_arena_bda(PAYLOAD).unwrap();
    (
        TransferCopyBatch {
            src,
            dst: Arc::from(dst),
            regions: vec![vk::BufferCopy::default().size(PAYLOAD as u64)],
        },
        bytes,
    )
}

#[test]
#[ignore = "central serial GPU: actual WeakShared and arena allocation after CPU timeline wait"]
fn gpu_cpu_wait_success_releases_all_completed_transfer_owners() {
    let backend = gpu_backend();
    let shared = Arc::downgrade(&backend.shared);
    assert!(backend
        .shared
        .submit_transfer_batches(&[])
        .unwrap()
        .is_none());
    assert!(!backend
        .shared
        .dedicated_transfer_keepalive_pending
        .load(Ordering::Acquire));
    let mut owners = Vec::new();
    let mut value = 0;
    for salt in 1..=3 {
        let (batch, _) = gpu_batch(&backend, salt);
        owners.push((Arc::downgrade(&batch.src), Arc::downgrade(&batch.dst)));
        value = backend
            .shared
            .submit_transfer_batches(&[batch])
            .unwrap()
            .unwrap();
    }
    assert!(owners
        .iter()
        .all(|(src, dst)| src.upgrade().is_some() && dst.upgrade().is_some()));
    assert!(backend
        .shared
        .submit_transfer_batches(&[])
        .unwrap()
        .is_none());
    assert!(backend
        .shared
        .dedicated_transfer_keepalive_pending
        .load(Ordering::Acquire));
    assert!(owners
        .iter()
        .all(|(src, dst)| src.upgrade().is_some() && dst.upgrade().is_some()));
    // An unrelated, successful fence with transfer proof zero must not retire these owners,
    // even if the dedicated queue has already completed them. Do not query its counter here.
    backend
        .recorder()
        .unwrap()
        .finish_nowait()
        .unwrap()
        .wait()
        .unwrap();
    assert!(owners
        .iter()
        .all(|(src, dst)| src.upgrade().is_some() && dst.upgrade().is_some()));
    backend.shared.wait_dedicated_transfer(value).unwrap();
    assert!(owners
        .iter()
        .all(|(src, dst)| src.upgrade().is_none() && dst.upgrade().is_none()));
    assert!(!backend
        .shared
        .dedicated_transfer_keepalive_pending
        .load(Ordering::Acquire));
    drop(backend);
    assert!(
        shared.upgrade().is_none(),
        "completed transfer slots retained VulkanShared"
    );
}

fn gpu_only_dependency_retirement(cancel: bool) {
    let backend = gpu_backend();
    let shared = Arc::downgrade(&backend.shared);
    let (batch, expected) = gpu_batch(&backend, 17);
    let source = Arc::downgrade(&batch.src);
    let arena = Arc::downgrade(&batch.dst);
    let readback: Arc<dyn Buffer> = Arc::new(
        backend
            .make_buf(
                PAYLOAD,
                MemoryLocation::GpuToCpu,
                "transfer-retirement-readback",
            )
            .unwrap(),
    );
    let recorder = backend.recorder().unwrap();
    assert!(recorder
        .submit_dedicated_transfer(std::slice::from_ref(&batch))
        .unwrap());
    recorder.copy_regions(
        batch.dst.as_ref(),
        readback.as_ref(),
        &[vk::BufferCopy::default().size(PAYLOAD as u64)],
    );
    recorder.retain_buffer(Arc::clone(&batch.dst));
    recorder.retain_buffer(Arc::clone(&readback));
    recorder.publish_host_reads();
    drop(batch);
    let pending = recorder.finish_nowait().unwrap();
    if cancel {
        // Cancellation of an already submitted consumer must run the same real fence drain.
        drop(pending);
    } else {
        pending.wait().unwrap();
    }
    // No manual retirement helper, CPU timeline wait, download, or queue-idle here: removing
    // PendingSegment::drain's captured-value success hook must make this regression fail.
    assert!(
        source.upgrade().is_none(),
        "GPU-only fence drain retained transfer source"
    );
    assert!(
        arena.upgrade().is_none(),
        "GPU-only fence drain retained arena allocation"
    );
    let ptr = as_vk_buf(readback.as_ref()).unwrap().mapped_ptr().unwrap();
    let actual = unsafe { std::slice::from_raw_parts(ptr, PAYLOAD) };
    assert_eq!(actual, expected.as_slice());
    drop(readback);
    drop(backend);
    assert!(
        shared.upgrade().is_none(),
        "GPU-only completion retained VulkanShared"
    );
}

#[test]
#[ignore = "central serial GPU: requires PendingSegment success-retirement hook"]
fn gpu_only_dependency_fence_releases_transfer_owners() {
    gpu_only_dependency_retirement(false);
}

#[test]
#[ignore = "central serial GPU: requires PendingSegment cancellation Drop-drain hook"]
fn gpu_only_dependency_cancel_drain_releases_transfer_owners() {
    gpu_only_dependency_retirement(true);
}
