//! Bounded transfer-only work owned by one cached ResidentNoticeProducer.
use super::{be, NoticeProducerAck};
use infr_core::error::Result;
use std::sync::{mpsc, Arc};

type Outcome = Result<Option<u64>>;
type Work<E> = Box<dyn FnOnce(&E) -> Outcome + Send>;

struct UploadJob<E> {
    work: Option<Work<E>>,
    completed: Option<mpsc::Sender<Outcome>>,
    acknowledgement: Option<Arc<NoticeProducerAck>>,
    retired: Option<mpsc::Sender<()>>,
    context: Option<infr_core::timeline::Context>,
}

fn invoke(work: impl FnOnce() -> Outcome) -> Outcome {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        // A panic payload may panic on drop; it must not bypass completion publication.
        std::mem::forget(payload);
        Err(be("owned upload worker panicked"))
    })
}

impl<E> UploadJob<E> {
    fn run(mut self, executor: &E) {
        let _context = infr_core::timeline::scoped_context(self.context);
        let work = self.work.take().expect("owned upload work");
        let result = invoke(|| work(executor));
        let _complete = infr_core::timeline::background_span("resident_owned_upload_completed");
        self.finish(result);
    }

    fn finish(&mut self, result: Outcome) {
        // A rejected/unexecuted job still owns its plan. Drop those captures before
        // waking a blocking completion receiver and before releasing its shared ACK.
        let cleanup = self.work.take().map(|work| {
            invoke(|| {
                drop(work);
                Ok(None)
            })
        });
        let result = match (result, cleanup) {
            (Ok(_), Some(Err(error))) => Err(error),
            (result, _) => result,
        };
        if let Some(completed) = self.completed.take() {
            let _ = completed.send(result);
        }
        drop(self.acknowledgement.take());
        if let Some(retired) = self.retired.take() {
            let _ = retired.send(());
        }
    }
}

impl<E> Drop for UploadJob<E> {
    fn drop(&mut self) {
        if self.completed.is_some() {
            self.finish(Err(be("owned upload cancelled before execution")));
        }
    }
}

pub(super) struct OwnedUploadWorker<E> {
    sender: Option<mpsc::SyncSender<UploadJob<E>>>,
    pending: Option<mpsc::Receiver<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl<E: Send + 'static> OwnedUploadWorker<E> {
    pub(super) fn optional(enabled: bool, create: impl FnOnce() -> Result<Self>) -> Option<Self> {
        if !enabled {
            return None;
        }
        match create() {
            Ok(worker) => Some(worker),
            Err(error) => {
                tracing::warn!(%error, "owned upload helper unavailable; retaining inline upload");
                None
            }
        }
    }

    pub(super) fn new(executor: E) -> Result<Self> {
        // Only one upload may be outstanding. Bound the channel as well so future
        // callers cannot silently accumulate prepared plans and source/target leases.
        let (sender, jobs) = mpsc::sync_channel::<UploadJob<E>>(1);
        let thread = std::thread::Builder::new()
            .name("resident-owned-upload".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    job.run(&executor);
                }
            })
            .map_err(|error| be(format!("create owned upload worker: {error}")))?;
        Ok(Self {
            sender: Some(sender),
            pending: None,
            thread: Some(thread),
        })
    }

    pub(super) fn wait_idle(&mut self) -> Result<()> {
        if let Some(pending) = self.pending.take() {
            #[cfg(test)]
            tests::pending_wait_entered();
            pending
                .recv()
                .map_err(|_| be("owned upload worker disconnected before retirement"))?;
        }
        Ok(())
    }

    pub(super) fn before_notice(&mut self, status: u32) -> Result<()> {
        if status != 0 {
            self.wait_idle()?;
        }
        Ok(())
    }

    pub(super) fn schedule(
        &mut self,
        fallback_executor: &E,
        work: impl FnOnce(&E) -> Outcome + Send + 'static,
        completed: mpsc::Sender<Outcome>,
        acknowledgement: Arc<NoticeProducerAck>,
        context: Option<infr_core::timeline::Context>,
    ) {
        let (retired, pending) = mpsc::channel();
        let mut job = UploadJob {
            work: Some(Box::new(work)),
            completed: Some(completed),
            acknowledgement: Some(acknowledgement),
            retired: Some(retired),
            context,
        };
        if let Err(error) = self.wait_idle() {
            job.finish(Err(error));
            return;
        }
        let Some(sender) = &self.sender else {
            job.run(fallback_executor);
            return;
        };
        // Fallback is serial: any earlier task has retired before try_send. Do not
        // add a second Started result if handoff fails; this job owns completion.
        match sender.try_send(job) {
            Ok(()) => self.pending = Some(pending),
            Err(mpsc::TrySendError::Full(job) | mpsc::TrySendError::Disconnected(job)) => {
                job.run(fallback_executor);
            }
        }
    }
}

impl<E> Drop for OwnedUploadWorker<E> {
    fn drop(&mut self) {
        // Receiver cancellation is not upload cancellation. Keep executor/captures
        // live until accepted work has actually returned, without any pager lock.
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
#[path = "adapter_owned_upload_worker_tests.rs"]
mod tests;
