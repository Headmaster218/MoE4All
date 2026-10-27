//! Bounded hardware experiment only; this does not model an upload ticket or model speedup.
use super::*;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(5);
const WARMUP_BLOCKS: usize = 2;
const SAMPLE_BLOCKS: usize = 16;
const SENTINEL: f32 = -4096.0;

fn fatal(message: &str) -> ! {
    eprintln!("HOST_READY_PROBE fatal: {message}; aborting before resource destructors");
    std::process::abort()
}

struct Ready {
    shared: Arc<crate::VulkanShared>,
    semaphore: vk::Semaphore,
}

impl Drop for Ready {
    fn drop(&mut self) {
        unsafe { self.shared.device.destroy_semaphore(self.semaphore, None) };
    }
}

// Ownership moves into the queued job, so dropped jobs and unwinding copies also release waits.
struct Release {
    ready: Arc<Ready>,
    value: u64,
}

impl Drop for Release {
    fn drop(&mut self) {
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.ready.semaphore)
            .value(self.value);
        if unsafe { self.ready.shared.device.signal_semaphore(&info) }.is_err() {
            fatal("host signal failed: a submitted timeline wait cannot be cancelled");
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CopyMode {
    Complete,
    FailHalfway,
    PanicHalfway,
    CancelBeforeCopy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Copied,
    Failed,
    Panicked,
    Cancelled,
}

struct CopyOutcome {
    status: Status,
    copy: Duration,
}

struct CopyJob {
    release: Release,
    source: Arc<Vec<f32>>,
    destination: Arc<dyn Buffer>,
    mode: CopyMode,
    done: Sender<CopyOutcome>,
}

fn mapped_copy(source: &[f32], destination: &dyn Buffer, mode: CopyMode) -> CopyOutcome {
    let target = Recorder::vk_of(destination).mapped_ptr().unwrap();
    let bytes: &[u8] = bytemuck::cast_slice(source);
    let count = if mode == CopyMode::Complete {
        bytes.len()
    } else {
        bytes.len() / 2
    };
    let start = Instant::now();
    // Both cases use the same single-core memcpy, without upload helpers or staging copies.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), target, count) };
    let copy = start.elapsed();
    match mode {
        CopyMode::Complete => CopyOutcome {
            status: Status::Copied,
            copy,
        },
        CopyMode::FailHalfway => CopyOutcome {
            status: Status::Failed,
            copy,
        },
        CopyMode::PanicHalfway => panic!("injected host-ready memcpy panic after half the payload"),
        CopyMode::CancelBeforeCopy => unreachable!("cancelled jobs must never reach memcpy"),
    }
}

struct Worker {
    sender: Option<Sender<CopyJob>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<CopyJob>();
        let thread = std::thread::Builder::new()
            .name("host-ready-probe-copy".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let CopyJob {
                        release,
                        source,
                        destination,
                        mode,
                        done,
                    } = job;
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        mapped_copy(&source, destination.as_ref(), mode)
                    }))
                    .unwrap_or(CopyOutcome {
                        status: Status::Panicked,
                        copy: Duration::ZERO,
                    });
                    // Signal only after copying has stopped; acknowledge after releasing owners.
                    drop(release);
                    drop(source);
                    drop(destination);
                    let _ = done.send(outcome);
                }
            })
            .unwrap();
        Self {
            sender: Some(sender),
            thread: Some(thread),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                fatal("copy worker exited outside the per-job panic boundary");
            }
        }
    }
}

struct Case {
    // Keep the recording ahead of its bound buffer in field drop order.
    command: RecordedCmd,
    destination: Arc<dyn Buffer>,
    ready: Arc<Ready>,
    fence: vk::Fence,
    bytes: usize,
    value: u64,
}

impl Case {
    fn new(backend: &VulkanBackend, bytes: usize) -> Self {
        // This allocator requires DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT; no fallback.
        let destination: Arc<dyn Buffer> = backend.alloc_mapped_arena_bda(bytes).unwrap().0.into();
        let recorder = backend.recorder_persistent().unwrap();
        // In B, host writes may happen after submit. Preserve the explicit HOST_WRITE scope.
        recorder.seed_barrier();
        recorder.scale(destination.as_ref(), 2.0, bytes / 4);
        recorder.publish_host_reads();
        let command = recorder.finish_record().unwrap();
        assert_eq!(command.segments.len(), 1);
        let device = &backend.shared.device;
        let mut kind =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .unwrap();
        let ready = Arc::new(Ready {
            shared: Arc::clone(&backend.shared),
            semaphore,
        });
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
        Self {
            command,
            destination,
            ready,
            fence,
            bytes,
            value: 0,
        }
    }

    fn run(&mut self, worker: &Worker, early_submit: bool, mode: CopyMode, epoch: usize) -> Sample {
        assert!(early_submit || mode == CopyMode::Complete);
        self.value += 1;
        let device = &self.ready.shared.device;
        unsafe { device.reset_fences(&[self.fence]) }.unwrap();
        let words = self.bytes / 4;
        let source: Arc<Vec<f32>> = Arc::new(
            (0..words)
                .map(|i| ((i * 17 + epoch * 31) % 4093 + 1) as f32 * 0.125)
                .collect(),
        );
        let pointer = Recorder::vk_of(self.destination.as_ref())
            .mapped_ptr()
            .unwrap();
        unsafe { std::slice::from_raw_parts_mut(pointer.cast::<f32>(), words) }.fill(SENTINEL);
        let (done, completion) = mpsc::channel();
        // Arm before submit, and keep any unsent release inside the drain guard. Its Drop
        // releases that job before waiting, even if the controller unwinds before handoff.
        let job = CopyJob {
            release: Release {
                ready: Arc::clone(&self.ready),
                value: self.value,
            },
            source: Arc::clone(&source),
            destination: Arc::clone(&self.destination),
            mode,
            done,
        };
        let mut flight = Flight {
            case: self,
            unsent: Some(job),
            completion: Some(completion),
            outcome: None,
            pending: false,
        };
        let commands = [self.command.segments[0].cmd];
        let semaphores = [self.ready.semaphore];
        let values = [self.value];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
        let mut submit = vk::SubmitInfo::default().command_buffers(&commands);
        if early_submit {
            submit = submit
                .wait_semaphores(&semaphores)
                .wait_dst_stage_mask(&stages)
                .push_next(&mut timeline);
        }
        let start = Instant::now();
        if mode == CopyMode::Complete {
            // Both arms pay for the same worker copy, release/signal and ACK. Only A
            // waits for that ACK before submit; B may overlap copy with queue_submit.
            flight.handoff(worker, mode);
            if !early_submit {
                flight.wait_copy();
            }
        }
        let submit_start = Instant::now();
        {
            // Raw, serialized, single-attempt submit: no recovery helper may queue-idle a
            // host-gated wait before its job is handed off. This is cfg(test) only.
            let _queue = self.ready.shared.queue_access.lock().unwrap();
            if unsafe { device.queue_submit(self.ready.shared.queue, &[submit], self.fence) }
                .is_err()
            {
                fatal("raw submit failed; do not guess whether resources can be reclaimed");
            }
            flight.pending = true;
        }
        let submit_time = submit_start.elapsed();
        let finish_start = Instant::now();
        if early_submit && mode != CopyMode::Complete {
            // Failure validation must exercise a real pending wait, regardless of scheduling.
            assert!(
                unsafe { device.get_semaphore_counter_value(self.ready.semaphore) }.unwrap()
                    < self.value
            );
            assert!(!unsafe { device.get_fence_status(self.fence) }.unwrap());
            flight.handoff(worker, mode);
        }
        flight.complete();
        let finish = finish_start.elapsed();
        let total = start.elapsed();
        let outcome = flight.outcome.as_ref().unwrap();
        // Full readback is outside all timing windows and follows copy completion + fence.
        let actual = unsafe { std::slice::from_raw_parts(pointer.cast::<f32>(), words) };
        for (index, (&actual, &input)) in actual.iter().zip(source.iter()).enumerate() {
            let copied = mode == CopyMode::Complete
                || (mode != CopyMode::CancelBeforeCopy && index < words / 2);
            let expected = if copied { input * 2.0 } else { SENTINEL * 2.0 };
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "bytes={} early={early_submit} mode={mode:?} epoch={epoch} index={index}",
                self.bytes
            );
        }
        let expected = match mode {
            CopyMode::Complete => Status::Copied,
            CopyMode::FailHalfway => Status::Failed,
            CopyMode::PanicHalfway => Status::Panicked,
            CopyMode::CancelBeforeCopy => Status::Cancelled,
        };
        assert_eq!(outcome.status, expected);
        assert_eq!(
            unsafe { device.get_semaphore_counter_value(self.ready.semaphore) }.unwrap(),
            self.value
        );
        Sample {
            copy: outcome.copy,
            submit: submit_time,
            finish,
            total,
        }
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        // Every successful submit is drained by Flight before this owner can be dropped.
        unsafe { self.ready.shared.device.destroy_fence(self.fence, None) };
    }
}

struct Flight<'a> {
    case: &'a Case,
    unsent: Option<CopyJob>,
    completion: Option<Receiver<CopyOutcome>>,
    outcome: Option<CopyOutcome>,
    pending: bool,
}

impl Flight<'_> {
    fn handoff(&mut self, worker: &Worker, mode: CopyMode) {
        let job = self.unsent.take().unwrap();
        if mode == CopyMode::CancelBeforeCopy {
            let (cancelled, receiver) = mpsc::channel::<CopyJob>();
            drop(receiver);
            // SendError owns the job: dropping it must signal the pending GPU wait.
            drop(cancelled.send(job).unwrap_err().0);
        } else if let Err(error) = worker.sender.as_ref().unwrap().send(job) {
            drop(error.0);
        }
    }

    fn wait_copy(&mut self) {
        if let Some(completion) = self.completion.take() {
            self.outcome = Some(match completion.recv_timeout(WAIT) {
                Ok(outcome) => outcome,
                Err(RecvTimeoutError::Disconnected) => CopyOutcome {
                    status: Status::Cancelled,
                    copy: Duration::ZERO,
                },
                Err(RecvTimeoutError::Timeout) => {
                    fatal("copy worker did not finish within five seconds")
                }
            });
        }
    }

    fn complete(&mut self) {
        drop(self.unsent.take());
        self.wait_copy();
        if self.pending {
            if unsafe {
                self.case.ready.shared.device.wait_for_fences(
                    &[self.case.fence],
                    true,
                    WAIT.as_nanos() as u64,
                )
            }
            .is_err()
            {
                fatal("submitted work did not drain within five seconds");
            }
            self.pending = false;
        }
    }
}

impl Drop for Flight<'_> {
    fn drop(&mut self) {
        self.complete();
    }
}

struct Sample {
    copy: Duration,
    submit: Duration,
    finish: Duration,
    total: Duration,
}

fn report(bytes: usize, label: &str, samples: &[Sample]) {
    for (metric, durations) in [
        ("copy", samples.iter().map(|s| s.copy).collect::<Vec<_>>()),
        ("submit", samples.iter().map(|s| s.submit).collect()),
        ("total", samples.iter().map(|s| s.total).collect()),
        ("finish", samples.iter().map(|s| s.finish).collect()),
    ] {
        let raw: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        let mut sorted = raw.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = (sorted[(n - 1) / 2] + sorted[n / 2]) / 2.0;
        let p90 = sorted[(n * 90).div_ceil(100) - 1];
        println!("HOST_READY_PROBE bytes={bytes} case={label} metric={metric} samples={n} median_us={median:.3} p90_us={p90:.3} raw_us={raw:?}");
    }
}

#[test]
#[ignore = "parent serial GPU runner only; explicitly set MOE4ALL_TEST_DEVICE=Vulkan1"]
fn host_ready_mapped_memcpy_vs_host_ready_worker_abba() {
    let selected = std::env::var("MOE4ALL_TEST_DEVICE")
        .expect("parent must explicitly set MOE4ALL_TEST_DEVICE=Vulkan1 on an idle GPU");
    assert_eq!(
        selected, "Vulkan1",
        "this isolated probe is assigned to Vulkan1"
    );
    let mut config = infr_core::config::Config::default();
    config.device.dev = Some(selected);
    let backend = VulkanBackend::new_with(Arc::new(config)).unwrap();
    // One persistent CPU worker for BOTH sizes, every trial, and all injected failures.
    let worker = Worker::new();
    println!("HOST_READY_PROBE device=Vulkan1 sizes_bytes=[524288,2097152] warmup_abba_blocks={WARMUP_BLOCKS} measured_abba_blocks={SAMPLE_BLOCKS} copy=pure_single_core_memcpy submit=host_call_including_queue_lock finish=submit_return_to_copy_ack_and_fence total=start_to_copy_ack_and_fence validation=full_payload_scale_x2");
    println!("HOST_READY_PROBE both cases use the same persistent worker copy/release/signal/ACK; A waits for ACK before submit without semaphore wait; B submits with ready wait before ACK; total includes handoff/ACK in both cases; B copy may overlap submit; metrics overlap and are not additive; no model-benefit inference");
    for bytes in [512 * 1024, 2 * 1024 * 1024] {
        let mut case = Case::new(&backend, bytes);
        let mut epoch = 0;
        let mut a = Vec::with_capacity(SAMPLE_BLOCKS * 2);
        let mut b = Vec::with_capacity(SAMPLE_BLOCKS * 2);
        for block in 0..WARMUP_BLOCKS + SAMPLE_BLOCKS {
            for early in [false, true, true, false] {
                epoch += 1;
                let sample = case.run(&worker, early, CopyMode::Complete, epoch);
                if block >= WARMUP_BLOCKS {
                    if early {
                        b.push(sample);
                    } else {
                        a.push(sample);
                    }
                }
            }
        }
        report(bytes, "A_worker_ack_then_submit", &a);
        report(bytes, "B_submit_with_worker_signal", &b);
        for mode in [
            CopyMode::FailHalfway,
            CopyMode::PanicHalfway,
            CopyMode::CancelBeforeCopy,
        ] {
            epoch += 1;
            case.run(&worker, true, mode, epoch);
            // Prove the same worker, timeline, recording and fence remain reusable.
            epoch += 1;
            case.run(&worker, true, CopyMode::Complete, epoch);
            println!("HOST_READY_PROBE bytes={bytes} injected={mode:?} release_and_full_payload=ok next_job=ok");
        }
    }
}
