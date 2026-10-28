//! Host release for one already-owned recovery recording, not a new notice epoch.
//!
//! Work/failure callbacks must make progress independently of the submitting thread:
//! no queue submission, queue-idle wait, or locks held by that thread. Before signaling,
//! work drains the upload and old notice ACKs, resets events, and remarks the current
//! notice. Failure must establish equally safe retirement and invalidate GPU recovery.
//! The caller joins before new notice jobs/readers/recording, and leases HostReadyWait
//! through GPU fence completion: CPU completion alone does not retire a semaphore wait.

use crate::{VulkanBackend, VulkanShared};
use ash::vk;
use infr_core::error::{backend as be, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};

type Callback = Box<dyn FnOnce() -> Result<()> + Send>;

#[derive(Clone)]
pub(crate) struct HostReadyWait {
    pub(crate) timeline: Arc<HostReadyTimeline>,
    pub(crate) value: u64,
}

pub(crate) struct HostReadyTimeline {
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) shared: Arc<VulkanShared>,
    signaled: Mutex<u64>,
}

impl HostReadyTimeline {
    fn signal(&self, value: u64) -> Result<()> {
        let mut previous = self
            .signaled
            .lock()
            .map_err(|_| be("host-ready signal ordering lock poisoned"))?;
        // A higher value would also release every older wait. Never skip a job whose
        // failure/retirement callback has not run, even during unexpected receiver teardown.
        if previous.checked_add(1) != Some(value) {
            return Err(be("host-ready jobs released out of sequence"));
        }
        unsafe {
            self.shared.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.semaphore)
                    .value(value),
            )
        }
        .map_err(|error| be(format!("signal host-ready timeline: {error}")))?;
        *previous = value;
        Ok(())
    }
}

impl Drop for HostReadyTimeline {
    fn drop(&mut self) {
        // shared keeps the device alive. The submitter must retain its own wait lease
        // until the pending fence retires, including error and cancellation paths.
        unsafe { self.shared.device.destroy_semaphore(self.semaphore, None) };
    }
}

fn fatal(error: impl std::fmt::Display) -> ! {
    // Once scheduled, a GPU wait may already exist. Failed invalidation cannot safely
    // signal success, and failed signaling cannot safely unwind into fence destructors.
    // Do not retry a callback/host signal or treat DEVICE_LOST as recoverable here:
    // that would require a backend-wide proof that no device work can execute again.
    eprintln!("[infr] fatal host-ready release failure: {error}");
    std::process::abort()
}

fn invoke(callback: Callback, context: &str) -> Result<()> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)).unwrap_or_else(|payload| {
        // A panic payload may itself have a panicking destructor. Forget it rather
        // than letting a second panic bypass failure publication and GPU release.
        std::mem::forget(payload);
        Err(be(format!("host-ready {context} panicked")))
    })
}

struct ReleaseJob {
    work: Option<Callback>,
    failure: Option<Callback>,
    signal: Option<Callback>,
    done: Option<Sender<Result<()>>>,
}

impl ReleaseJob {
    fn run(mut self) {
        let result = invoke(self.work.take().expect("owned host-ready work"), "work");
        self.finish(result);
    }

    fn finish(&mut self, result: Result<()>) {
        // Queued cancellation must drop work's retained guards before invalidating and
        // signaling. If those destructors panic, quiescence is unknown: fail closed.
        if let Some(work) = self.work.take() {
            if invoke(
                Box::new(move || {
                    drop(work);
                    Ok(())
                }),
                "cancelled work cleanup",
            )
            .is_err()
            {
                fatal("cancelled work captures could not be safely retired");
            }
        }
        let failure = self
            .failure
            .take()
            .unwrap_or_else(|| fatal("missing failure callback"));
        if result.is_err() {
            if let Err(error) = invoke(failure, "failure callback") {
                fatal(error);
            }
        } else if invoke(
            Box::new(move || {
                drop(failure);
                Ok(())
            }),
            "failure capture cleanup",
        )
        .is_err()
        {
            fatal("unused failure captures could not be safely retired");
        }
        if let Err(error) = invoke(
            self.signal
                .take()
                .unwrap_or_else(|| fatal("missing signal owner")),
            "signal",
        ) {
            fatal(error);
        }
        // All callback captures, including their source/target/fence leases, are now
        // gone. Preserve the first work/cancellation error after successful failure handling.
        if let Some(done) = self.done.take() {
            let _ = done.send(result);
        }
    }
}

impl Drop for ReleaseJob {
    fn drop(&mut self) {
        if self.done.is_some() {
            self.finish(Err(be("host-ready job cancelled before completion")));
        }
    }
}

pub(crate) struct HostReadyCompletion {
    pub(crate) wait: HostReadyWait,
    done: Option<Receiver<Result<()>>>,
    result: Option<std::result::Result<(), String>>,
}

impl HostReadyCompletion {
    pub(crate) fn join(&mut self) -> Result<()> {
        if let Some(done) = self.done.take() {
            let result = done.recv().unwrap_or_else(|_| {
                Err(be(
                    "host-ready worker disconnected before completion acknowledgement",
                ))
            });
            self.result = Some(result.map_err(|error| error.to_string()));
        }
        match self.result.as_ref().expect("joined host-ready result") {
            Ok(()) => Ok(()),
            Err(error) => Err(be(format!("host-ready completion: {error}"))),
        }
    }
}

impl Drop for HostReadyCompletion {
    fn drop(&mut self) {
        let _ = self.join();
    }
}

pub(crate) struct HostReadyWorker {
    timeline: Arc<HostReadyTimeline>,
    // Serialize value reservation with enqueue: concurrent callers cannot reorder signals.
    sender: Mutex<Option<SyncSender<ReleaseJob>>>,
    sequence: AtomicU64,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HostReadyWorker {
    pub(crate) fn new(backend: &VulkanBackend) -> Result<Self> {
        let mut kind =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = unsafe {
            backend.shared.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )
        }
        .map_err(|error| be(format!("create host-ready timeline: {error}")))?;
        let timeline = Arc::new(HostReadyTimeline {
            semaphore,
            shared: Arc::clone(&backend.shared),
            signaled: Mutex::new(0),
        });
        // At most one queued release in addition to the job currently running. Normal
        // recovery joins every ticket before requesting another, so it never queues ahead.
        let (sender, jobs) = mpsc::sync_channel::<ReleaseJob>(1);
        let thread = std::thread::Builder::new()
            .name("resident-host-ready-release".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    job.run();
                }
            })
            .map_err(|error| be(format!("create host-ready worker: {error}")))?;
        Ok(Self {
            timeline,
            sender: Mutex::new(Some(sender)),
            sequence: AtomicU64::new(0),
            thread: Some(thread),
        })
    }

    pub(crate) fn schedule(
        &self,
        work: Box<dyn FnOnce() -> Result<()> + Send>,
        failure: Box<dyn FnOnce() -> Result<()> + Send>,
    ) -> Result<HostReadyCompletion> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| be("host-ready scheduler poisoned"))?;
        let sender_ref = sender
            .as_ref()
            .ok_or_else(|| be("host-ready worker closed"))?;
        let previous = self
            .sequence
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| be("host-ready timeline sequence exhausted"))?;
        let wait = HostReadyWait {
            timeline: Arc::clone(&self.timeline),
            value: previous + 1,
        };
        let signal_wait = wait.clone();
        let (done, receiver) = mpsc::channel();
        let job = ReleaseJob {
            work: Some(work),
            failure: Some(failure),
            signal: Some(Box::new(move || {
                signal_wait.timeline.signal(signal_wait.value)
            })),
            done: Some(done),
        };
        let sent = sender_ref.send(job);
        drop(sender);
        if let Err(unsent) = sent {
            // The returned job still owns failure + signal; run its Drop before reporting
            // rejection. No wait token escapes until the handoff succeeds.
            drop(unsent.0);
            return Err(be(
                "host-ready release worker disconnected during scheduling",
            ));
        }
        Ok(HostReadyCompletion {
            wait,
            done: Some(receiver),
            result: None,
        })
    }
}

impl Drop for HostReadyWorker {
    fn drop(&mut self) {
        self.sender
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        if let Some(thread) = self.thread.take() {
            if thread.thread().id() == std::thread::current().id() {
                fatal("callback dropped the last owner of its own release worker");
            }
            if thread.join().is_err() {
                fatal("release worker exited outside its per-job panic boundary");
            }
        }
        // No shared -> worker ownership edge: timeline retains shared, never the reverse.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Capture(Arc<Mutex<Vec<&'static str>>>, &'static str);
    impl Drop for Capture {
        fn drop(&mut self) {
            self.0.lock().unwrap().push(self.1);
        }
    }

    fn job(log: &Arc<Mutex<Vec<&'static str>>>, mode: u8) -> (ReleaseJob, Receiver<Result<()>>) {
        let work_log = Arc::clone(log);
        let work_capture = Capture(Arc::clone(log), "work_drop");
        let failure_log = Arc::clone(log);
        let failure_capture = Capture(Arc::clone(log), "failure_drop");
        let signal_log = Arc::clone(log);
        let signal_capture = Capture(Arc::clone(log), "signal_drop");
        let (done, receiver) = mpsc::channel();
        (
            ReleaseJob {
                work: Some(Box::new(move || {
                    let _capture = work_capture;
                    work_log.lock().unwrap().push("work");
                    match mode {
                        1 => Err(be("first work failure")),
                        2 => panic!("injected work panic"),
                        _ => Ok(()),
                    }
                })),
                failure: Some(Box::new(move || {
                    let _capture = failure_capture;
                    failure_log.lock().unwrap().push("failure");
                    Ok(())
                })),
                signal: Some(Box::new(move || {
                    let _capture = signal_capture;
                    signal_log.lock().unwrap().push("signal");
                    Ok(())
                })),
                done: Some(done),
            },
            receiver,
        )
    }

    #[test]
    fn success_drops_all_captures_before_ack() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (job, done) = job(&log, 0);
        job.run();
        done.recv().unwrap().unwrap();
        assert_eq!(
            *log.lock().unwrap(),
            ["work", "work_drop", "failure_drop", "signal", "signal_drop"]
        );
    }

    #[test]
    fn error_and_panic_invalidate_before_signal_and_preserve_first_error() {
        for mode in [1, 2] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let (job, done) = job(&log, mode);
            job.run();
            let error = done.recv().unwrap().unwrap_err().to_string();
            assert!(error.contains(if mode == 1 {
                "first work failure"
            } else {
                "work panicked"
            }));
            assert_eq!(
                *log.lock().unwrap(),
                [
                    "work",
                    "work_drop",
                    "failure",
                    "failure_drop",
                    "signal",
                    "signal_drop"
                ]
            );
        }
    }

    #[test]
    fn queued_channel_cancellation_runs_failure_and_signal_without_work() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (job, done) = job(&log, 0);
        let (sender, receiver) = mpsc::channel();
        assert!(sender.send(job).is_ok());
        drop(receiver);
        assert!(done
            .recv()
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("cancelled"));
        assert_eq!(
            *log.lock().unwrap(),
            [
                "work_drop",
                "failure",
                "failure_drop",
                "signal",
                "signal_drop"
            ]
        );
    }

    #[test]
    fn dropped_completion_receiver_still_releases_job() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (job, done) = job(&log, 1);
        drop(done);
        job.run();
        assert_eq!(
            *log.lock().unwrap(),
            [
                "work",
                "work_drop",
                "failure",
                "failure_drop",
                "signal",
                "signal_drop"
            ]
        );
    }
}
