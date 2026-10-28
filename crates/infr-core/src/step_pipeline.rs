//! Bounded CPU preparation for the next execution step, never speculative model execution.
//!
//! Each pipeline is scoped to one stable execution cohort. Only one job may be outstanding;
//! dropping it closes both channels and joins its worker before the cohort's resources expire.
//! A caller must not put uploads, KV growth, sampling or mutable bindings in `prepare`.

use std::sync::mpsc::{self, Receiver, SyncSender};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparationError {
    AlreadyPending,
    UnexpectedStep,
    WorkerStopped,
    WorkerPanicked,
}

impl std::fmt::Display for PreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "step preparation: {self:?}")
    }
}

impl std::error::Error for PreparationError {}

type Prepared<T> = (usize, Result<T, PreparationError>);

enum Mode<'a, J, T> {
    Inline(&'a dyn Fn(J) -> T, Option<T>),
    Worker(SyncSender<(usize, J)>, Receiver<Prepared<T>>),
}

pub struct StepPreparation<'a, J, T> {
    mode: Mode<'a, J, T>,
    pending: Option<usize>,
}

impl<J, T> StepPreparation<'_, J, T> {
    pub fn is_async(&self) -> bool {
        matches!(self.mode, Mode::Worker(..))
    }

    pub fn pending_step(&self) -> Option<usize> {
        self.pending
    }

    pub fn schedule(&mut self, step: usize, job: J) -> Result<(), PreparationError> {
        if self.pending.is_some() {
            return Err(PreparationError::AlreadyPending);
        }
        match &mut self.mode {
            Mode::Inline(prepare, ready) => *ready = Some(prepare(job)),
            Mode::Worker(jobs, _) => jobs
                .send((step, job))
                .map_err(|_| PreparationError::WorkerStopped)?,
        }
        self.pending = Some(step);
        Ok(())
    }

    pub fn take(&mut self, step: usize) -> Result<T, PreparationError> {
        if self.pending != Some(step) {
            return Err(PreparationError::UnexpectedStep);
        }
        self.pending = None;
        match &mut self.mode {
            Mode::Inline(_, ready) => ready.take().ok_or(PreparationError::WorkerStopped),
            Mode::Worker(_, results) => {
                let (actual, prepared) = results
                    .recv()
                    .map_err(|_| PreparationError::WorkerStopped)?;
                if actual != step {
                    return Err(PreparationError::UnexpectedStep);
                }
                prepared
            }
        }
    }
}

/// Fall back to inline preparation if disabled or if the OS cannot create a worker. The body
/// runs on the caller thread, so its callbacks need not be `Send`. Job errors belong in `T` and
/// are observed only when consumed; an unused next-step error must not invalidate a final token.
pub fn with_step_preparation<J: Send, T: Send, R>(
    enabled: bool,
    prepare: impl Fn(J) -> T + Sync,
    body: impl FnOnce(&mut StepPreparation<'_, J, T>) -> R,
) -> R {
    std::thread::scope(|scope| {
        let mut pipeline = StepPreparation {
            mode: Mode::Inline(&prepare, None),
            pending: None,
        };
        if enabled {
            let (jobs, requests) = mpsc::sync_channel::<(usize, J)>(1);
            let (finished, results) = mpsc::sync_channel(1);
            let prepare = &prepare;
            if std::thread::Builder::new()
                .name("infr-step-prepare".into())
                .spawn_scoped(scope, move || {
                    while let Ok((step, job)) = requests.recv() {
                        let prepared =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare(job)))
                                .map_err(|_| PreparationError::WorkerPanicked);
                        let failed = prepared.is_err();
                        if finished.send((step, prepared)).is_err() || failed {
                            break;
                        }
                    }
                })
                .is_ok()
            {
                pipeline.mode = Mode::Worker(jobs, results);
            }
        }
        body(&mut pipeline)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    #[test]
    fn inline_and_worker_preserve_step_order_and_values() {
        for enabled in [false, true] {
            let values = with_step_preparation(
                enabled,
                |x: usize| x * 7,
                |pipeline| {
                    (0..32)
                        .map(|step| {
                            pipeline.schedule(step, step).unwrap();
                            pipeline.take(step).unwrap()
                        })
                        .collect::<Vec<_>>()
                },
            );
            assert_eq!(values, (0..32).map(|x| x * 7).collect::<Vec<_>>());
        }
    }

    #[test]
    fn slow_next_prepare_does_not_block_current_tail_or_ack() {
        let gate = Barrier::new(2);
        with_step_preparation(
            true,
            |_: ()| gate.wait(),
            |pipeline| {
                assert!(pipeline.is_async());
                pipeline.schedule(1, ()).unwrap();
                // This stands for current readback/stop/ACK; a join before tail would deadlock.
                gate.wait();
                pipeline.take(1).unwrap();
            },
        );
    }

    #[test]
    fn bounds_one_pending_and_does_not_consume_wrong_step() {
        for enabled in [false, true] {
            with_step_preparation(
                enabled,
                |x| x,
                |pipeline| {
                    pipeline.schedule(4, 17).unwrap();
                    assert_eq!(
                        pipeline.schedule(5, 18),
                        Err(PreparationError::AlreadyPending)
                    );
                    assert_eq!(pipeline.take(5), Err(PreparationError::UnexpectedStep));
                    assert_eq!(pipeline.pending_step(), Some(4));
                    assert_eq!(pipeline.take(4), Ok(17));
                    assert_eq!(pipeline.pending_step(), None);
                },
            );
        }
    }

    #[test]
    fn unused_next_error_does_not_fail_stopped_sequence() {
        for enabled in [false, true] {
            let result: Result<(), &str> = with_step_preparation(
                enabled,
                |_: ()| Err::<(), _>("unneeded next shape"),
                |pipeline| {
                    pipeline.schedule(1, ()).unwrap();
                    Ok(())
                },
            );
            assert_eq!(result, Ok(()));
        }
    }

    #[test]
    fn exit_and_error_drain_worker_before_borrowed_resources_expire() {
        struct Job(Arc<AtomicUsize>);
        impl Drop for Job {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        for fail in [false, true] {
            let released = Arc::new(AtomicUsize::new(0));
            let result = with_step_preparation(
                true,
                |job: Job| drop(job),
                |pipeline| {
                    pipeline.schedule(1, Job(Arc::clone(&released))).unwrap();
                    if fail {
                        Err("execute failed")
                    } else {
                        Ok(())
                    }
                },
            );
            assert_eq!(result.is_err(), fail);
            assert_eq!(released.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn worker_panic_is_a_consumed_preparation_error_not_a_scope_panic() {
        with_step_preparation(
            true,
            |_: ()| panic!("test preparation panic"),
            |pipeline| {
                pipeline.schedule(0, ()).unwrap();
                assert_eq!(pipeline.take(0), Err(PreparationError::WorkerPanicked));
            },
        );
    }

    #[test]
    fn new_cohort_cannot_receive_previous_cohort_result() {
        for cohort in 0..16 {
            with_step_preparation(
                true,
                |step| (cohort, step),
                |pipeline| {
                    pipeline.schedule(0, 0).unwrap();
                    assert_eq!(pipeline.take(0), Ok((cohort, 0)));
                    pipeline.schedule(1, 1).unwrap();
                    // Simulate Prefill insertion / resize / cancellation with unused lookahead.
                },
            );
        }
    }

    #[test]
    fn result_processing_overlaps_prepare_without_executing_unconfirmed_tokens() {
        for lanes in [1, 2, 3, 8] {
            for stop in 0..4 {
                let mut executed = vec![0; lanes];
                let mut delivered = vec![0; lanes];
                with_step_preparation(
                    true,
                    |step| vec![step; lanes],
                    |pipeline| {
                        pipeline.schedule(0, 0).unwrap();
                        for step in 0..4 {
                            let plan = pipeline.take(step).unwrap();
                            assert_eq!(plan, vec![step; lanes]);
                            if step + 1 < 4 {
                                pipeline.schedule(step + 1, step + 1).unwrap();
                            }
                            // Only the caller executes/commits, then resolves stop/ACK for every lane.
                            executed.iter_mut().for_each(|n| *n += 1);
                            delivered.iter_mut().for_each(|n| *n += 1);
                            if step == stop {
                                break;
                            }
                        }
                    },
                );
                assert_eq!(executed, vec![stop + 1; lanes]);
                assert_eq!(executed, delivered);
            }
        }
    }

    #[test]
    fn worker_is_drained_on_caller_unwind() {
        let finished = AtomicUsize::new(0);
        let caught = std::panic::catch_unwind(|| {
            with_step_preparation(
                true,
                |_: ()| {
                    finished.fetch_add(1, Ordering::SeqCst);
                },
                |pipeline| {
                    pipeline.schedule(0, ()).unwrap();
                    panic!("test caller panic");
                },
            );
        });
        assert!(caught.is_err());
        assert_eq!(finished.load(Ordering::SeqCst), 1);
    }
}
