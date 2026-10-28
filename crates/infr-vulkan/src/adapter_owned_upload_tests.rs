use super::*;
use std::sync::mpsc;
use std::time::Duration;

struct OwnedRecoveryCounts;

impl OwnedRecoveryCounts {
    fn begin() -> Self {
        OWNED_RECOVERY_TEST_COUNTS.with(|counts| {
            assert!(counts.replace(Some([0; 3])).is_none());
        });
        Self
    }

    fn values(&self) -> [usize; 3] {
        OWNED_RECOVERY_TEST_COUNTS.with(|counts| counts.get().unwrap())
    }
}

impl Drop for OwnedRecoveryCounts {
    fn drop(&mut self) {
        OWNED_RECOVERY_TEST_COUNTS.with(|counts| counts.set(None));
    }
}

#[test]
fn owned_upload_empty_tail_requires_specific_fresh_recorder_provenance() {
    assert!(owned_guarded_empty_tail_eligible(true, true, 1, 2, 0));
    assert!(owned_guarded_empty_tail_eligible(true, false, 1, 1, 0));
    assert!(!owned_guarded_empty_tail_eligible(true, false, 1, 2, 0));
    assert!(!owned_guarded_empty_tail_eligible(true, true, 1, 2, 1));
    // Transfers, event commands and waits need not increase dispatches. Zero alone
    // cannot certify a recorder; only the known newly-created boundary recorder is armed.
    for graph_end in [false, true] {
        assert!(!owned_guarded_empty_tail_eligible(
            false, graph_end, 1, 1, 0
        ));
    }
}

#[test]
fn owned_upload_guarded_recovery_requires_every_existing_scratch_capacity() {
    let mut pool = ScratchPool::default();
    let requirements = [
        ("moe_paged_g", 4096),
        ("moe_paged_y", 512),
        ("moe_paged_u", 0),
    ];
    assert!(!owned_recovery_scratch_capacities_ready(
        &pool,
        &requirements
    ));
    pool.buffers
        .insert(("moe_paged_g", 8192), Box::new(CpuBuffer(8192)));
    pool.buffers
        .insert(("moe_paged_y", 508), Box::new(CpuBuffer(508)));
    assert!(!owned_recovery_scratch_capacities_ready(
        &pool,
        &requirements
    ));
    pool.buffers
        .insert(("moe_paged_y", 512), Box::new(CpuBuffer(512)));
    assert!(owned_recovery_scratch_capacities_ready(
        &pool,
        &requirements
    ));
    // Larger requests fall back before the new recorder can grow/reclaim unified scratch.
    assert!(!owned_recovery_scratch_capacities_ready(
        &pool,
        &[("moe_paged_g", 8196)]
    ));
    assert!(
        pool.in_use.is_empty(),
        "the proof is read-only, not a speculative allocation"
    );
}

fn promotion(pending: CpuPromotionPending) -> CpuMissPromotion {
    CpuMissPromotion {
        roles: [1, 2, 3],
        ids: vec![0, 1],
        layer: 0,
        sources: Vec::new(),
        started: true,
        pending: Some(pending),
        _owned_sources: Vec::new(),
    }
}

fn producer() -> (
    Arc<ResidentNoticeProducer>,
    mpsc::Sender<(usize, Result<ProducedNotice>)>,
) {
    let (output, results) = mpsc::channel();
    (
        Arc::new(ResidentNoticeProducer {
            host_ready: None,
            sender: None,
            results: Mutex::new(results),
            thread: None,
            announced_miss: Arc::new(AtomicUsize::new(0)),
            pager: Arc::new(Mutex::new(None)),
        }),
        output,
    )
}

#[test]
fn owned_upload_retired_tail_preserves_first_error_and_next_request_is_clean() {
    for earlier_error in [false, true] {
        let (producer, output) = producer();
        assert!(output
            .send((17, Err(be("first remaining tail failure"))))
            .is_ok());
        assert!(output
            .send((
                18,
                Ok(ProducedNotice {
                    words: [0; crate::recorder::RESIDENT_JOURNAL_WORDS],
                    promotion: None,
                    owned_cpu: false,
                })
            ))
            .is_ok());
        assert!(output
            .send((19, Err(be("later remaining tail failure"))))
            .is_ok());
        let mut result = if earlier_error {
            Err(be("original tail failure"))
        } else {
            Ok(())
        };
        producer.discard_retired_results(&mut result);
        assert!(result.unwrap_err().to_string().contains(if earlier_error {
            "original tail failure"
        } else {
            "first remaining tail failure"
        }));
        assert!(producer.receive(false).unwrap().is_none());
        let mut words = [0; crate::recorder::RESIDENT_JOURNAL_WORDS];
        words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 1;
        words[0] = 9123;
        assert!(output
            .send((
                0,
                Ok(ProducedNotice {
                    words,
                    promotion: None,
                    owned_cpu: false
                })
            ))
            .is_ok());
        let (ordinal, notice) = producer.receive(true).unwrap().unwrap();
        assert_eq!(ordinal, 0);
        assert_eq!(notice.unwrap().words, words);
        assert!(producer.receive(false).unwrap().is_none());
    }
}

#[test]
fn owned_upload_completion_retains_dma_value_and_errors() {
    for value in [None, Some(7), Some(u64::MAX - 1)] {
        let (completed, completion) = mpsc::channel();
        let waited = Mutex::new(Vec::new());
        let mut pending = OwnedUploadCompletion::new(
            |value| {
                waited.lock().unwrap().push(value);
                Ok(())
            },
            completion,
        );
        completed.send(Ok(value)).unwrap();
        assert_eq!(pending.wait().unwrap(), value);
        assert_eq!(pending.wait().unwrap(), None);
        assert_eq!(
            *waited.lock().unwrap(),
            value.into_iter().collect::<Vec<_>>()
        );
    }
    let (completed, completion) = mpsc::channel();
    let mut pending = OwnedUploadCompletion::new(|_| Ok(()), completion);
    completed.send(Err(be("original upload failure"))).unwrap();
    assert!(pending
        .wait()
        .unwrap_err()
        .to_string()
        .contains("original upload failure"));
    let (completed, completion) = mpsc::channel();
    drop(completed);
    let mut pending = OwnedUploadCompletion::new(|_| Ok(()), completion);
    assert!(pending
        .wait()
        .unwrap_err()
        .to_string()
        .contains("disconnected"));
}

#[test]
fn owned_upload_cancel_and_unwind_wait_for_completion_and_release_retained_sources() {
    for unwind in [false, true] {
        let (completed, completion) = mpsc::channel();
        let (dropping, started_drop) = mpsc::channel();
        let (finished, finish) = mpsc::channel();
        let source = Arc::new(vec![17u8; 64]);
        let weak = Arc::downgrade(&source);
        std::thread::scope(|scope| {
            let worker = scope.spawn(move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _source = source;
                    let _completion = OwnedUploadCompletion::new(|_| Ok(()), completion);
                    dropping.send(()).unwrap();
                    if unwind {
                        panic!("synthetic recovery cancellation");
                    }
                }));
                assert_eq!(outcome.is_err(), unwind);
                finished.send(()).unwrap();
            });
            started_drop.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(weak.upgrade().is_some());
            assert!(matches!(finish.try_recv(), Err(mpsc::TryRecvError::Empty)));
            completed
                .send(Err(be("cancelled copy completed with failure")))
                .unwrap();
            finish.recv_timeout(Duration::from_secs(5)).unwrap();
            worker.join().unwrap();
        });
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn owned_upload_undelivered_started_releases_completion_before_notice_drop() {
    let (completed, completion) = mpsc::channel();
    let (output, results) = mpsc::channel();
    drop(results);
    let completion = OwnedUploadCompletion::new(|_| Ok(()), completion);
    let Err(unsent) = output.send((0usize, completion)) else {
        panic!("closed receiver");
    };
    // This is the send-error ordering in the producer: dropping unsent first deadlocks.
    completed.send(Err(be("Started not delivered"))).unwrap();
    drop(unsent);
}

#[test]
fn owned_upload_dma_drain_failure_keeps_value_for_drop_before_lease_release() {
    let (completed, completion) = mpsc::channel();
    let (draining, drain_started) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let (finished, finish) = mpsc::channel();
    let source = Arc::new(vec![23u8; 64]);
    let weak = Arc::downgrade(&source);
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            let _source = source;
            let calls = AtomicUsize::new(0);
            let mut pending = OwnedUploadCompletion::new(
                |value: u64| -> Result<()> {
                    assert_eq!(value, 91);
                    if calls.fetch_add(1, Ordering::AcqRel) == 0 {
                        return Err(be("first DMA drain failed"));
                    }
                    draining.send(()).unwrap();
                    released.recv().unwrap();
                    Ok(())
                },
                completion,
            );
            assert!(pending
                .wait()
                .unwrap_err()
                .to_string()
                .contains("first DMA drain failed"));
            assert_eq!(pending.pending_dma, Some(91));
            drop(pending); // Must retry the retained value before its enclosing leases drop.
            assert_eq!(calls.load(Ordering::Acquire), 2);
            finished.send(()).unwrap();
        });
        completed.send(Ok(Some(91))).unwrap();
        drain_started.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(weak.upgrade().is_some());
        assert!(matches!(finish.try_recv(), Err(mpsc::TryRecvError::Empty)));
        release.send(()).unwrap();
        finish.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
    });
    assert!(weak.upgrade().is_none());
}

#[test]
#[ignore = "central GPU runner only; real recorder blocks before submit until upload plus ACK"]
fn gpu_owned_upload_recording_overlaps_delayed_copy_and_gates_every_submit() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let backend = &fixture.backend;
    let target = fixture.buffer(fixture.outputs[0]);
    // Two successful epochs reuse the same backend, events and producer.
    // Error/disconnect/cancel/unwind must retain the unsubmitted output and the old event.
    for outcome in [0, 1, 2, 3, 4, 5, 6] {
        let notices = Arc::new(ResidentNotices::new(backend, 1).unwrap());
        let (producer, output) = producer();
        for miss in 0..if outcome == 0 { 2 } else { 1 } {
            backend
                .upload(target, bytemuck::cast_slice(&[19.0f32; NE]))
                .unwrap();
            unsafe {
                notices.shared.device.set_event(notices.events[0]).unwrap();
            }
            notices.dirty.mark(0);
            let (completed, completion) = mpsc::channel();
            let (acknowledge, ack) = mpsc::channel();
            let (recorded, recording) = mpsc::channel();
            let (release_recording, release) = mpsc::channel();
            let (finished, finish) = mpsc::channel();
            if outcome == 6 {
                assert!(output
                    .send((2, Err(be("old producer tail failed"))))
                    .is_ok());
                assert!(output.send((47, Err(be("later old tail failure")))).is_ok());
                assert!(output
                    .send((
                        48,
                        Ok(ProducedNotice {
                            words: [0; crate::recorder::RESIDENT_JOURNAL_WORDS],
                            promotion: None,
                            owned_cpu: false,
                        })
                    ))
                    .is_ok());
            }
            std::thread::scope(|scope| {
                let worker_producer = Arc::clone(&producer);
                let worker_notices = Arc::clone(&notices);
                let executor = backend
                    .background_expert_transfer_executor()
                    .expect("dedicated upload executor");
                let worker = scope.spawn(move || {
                    // AffinityGuard deliberately makes PagedStream !Send. Construct it on
                    // its recording thread, rather than weakening that production contract.
                    let mut stream = PagedStream {
                        cpu_promotion: Some(promotion(CpuPromotionPending::OwnedUpload(
                            OwnedUploadCompletion::new(executor, completion),
                        ))),
                        deferred_owned_notices: Some(DeferredOwnedNotices {
                            producer: worker_producer,
                            notices: worker_notices,
                            entries: notice_entries(&[false, true, outcome == 6]),
                            // The CPU miss was consumed; only the successor remains.
                            cursor: 2,
                            sent: 3,
                            acknowledgements: vec![ack],
                            drained: false,
                            guarded_seed: None,
                            failure_fence: None,
                        }),
                        ..Default::default()
                    };
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                            let recorder = backend.recorder()?;
                            recorder.seed_barrier();
                            recorder.scale(target, 0.0, NE);
                            assert!(recorder.dispatches() > 0);
                            // A real GPU write has already been encoded while the modeled mapped
                            // copy is deliberately parked before Completed. This is not a span-name test.
                            recorded.send(recorder.dispatches()).unwrap();
                            release.recv().unwrap();
                            if matches!(outcome, 3 | 4) {
                                let cancelled = std::mem::take(&mut stream);
                                if outcome == 4 {
                                    let _cancelled = cancelled;
                                    panic!("synthetic recording unwind");
                                }
                                drop(cancelled);
                                recorder.discard()?;
                                return Err(Error::Aborted);
                            }
                            let result = stream.complete_owned_recording(&recorder, 0, true);
                            if result.is_err() {
                                recorder.discard()?;
                                return result;
                            }
                            recorder.finish()
                        }));
                    finished
                        .send(result.map(|result| result.is_ok()).unwrap_or(false))
                        .unwrap();
                });
                assert!(recording.recv_timeout(Duration::from_secs(5)).unwrap() > 0);
                assert_eq!(fixture.read(target), vec![19.0; NE]);
                assert!(unsafe {
                    notices
                        .shared
                        .device
                        .get_event_status(notices.events[0])
                        .unwrap()
                });
                release_recording.send(()).unwrap();
                assert!(matches!(finish.try_recv(), Err(mpsc::TryRecvError::Empty)));
                match outcome {
                    0 | 3 | 4 | 5 | 6 => completed.send(Ok(None)).unwrap(),
                    1 => completed
                        .send(Err(be("delayed mapped upload failure")))
                        .unwrap(),
                    2 => drop(completed),
                    _ => unreachable!(),
                }
                // Completed alone cannot recycle the event or submit: the old worker still
                // holds its ACK and could be reading this epoch's event/journal.
                assert_eq!(fixture.read(target), vec![19.0; NE]);
                assert!(unsafe {
                    notices
                        .shared
                        .device
                        .get_event_status(notices.events[0])
                        .unwrap()
                });
                if outcome == 5 {
                    drop(acknowledge);
                } else {
                    acknowledge.send(()).unwrap();
                }
                assert_eq!(
                    finish.recv_timeout(Duration::from_secs(5)).unwrap(),
                    outcome == 0
                );
                worker.join().unwrap();
            });
            assert!(
                producer.receive(false).unwrap().is_none(),
                "no stale ordinal after old ACKs"
            );
            if outcome == 6 {
                let mut words = [0; crate::recorder::RESIDENT_JOURNAL_WORDS];
                words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 1;
                assert!(output
                    .send((
                        0,
                        Ok(ProducedNotice {
                            words,
                            promotion: None,
                            owned_cpu: false
                        })
                    ))
                    .is_ok());
                let (ordinal, notice) = producer.receive(true).unwrap().unwrap();
                assert_eq!(ordinal, 0);
                assert_eq!(notice.unwrap().words, words);
            }
            assert_eq!(
                fixture.read(target),
                vec![if outcome == 0 { 0.0 } else { 19.0 }; NE]
            );
            if outcome == 0 {
                assert!(!unsafe {
                    notices
                        .shared
                        .device
                        .get_event_status(notices.events[0])
                        .unwrap()
                });
                let mut dirty = Vec::new();
                notices
                    .dirty
                    .reset_with(1, |ordinal| {
                        dirty.push(ordinal);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(
                    dirty,
                    [0],
                    "recovery mark must survive old epoch reset, miss {miss}"
                );
            }
            assert_eq!(producer.announced_miss.load(Ordering::Acquire), usize::MAX);
        }
    }
}

#[test]
#[ignore = "central GPU runner only; two independent owned CPU misses within one forward"]
fn gpu_owned_upload_started_recovers_multiple_cpu_layers_once() {
    run_owned_upload_two_cpu_misses(false, false);
}

#[test]
#[ignore = "central GPU runner only; nonempty post-recovery continuation must still submit"]
fn gpu_owned_upload_guarded_recovery_keeps_live_tail_commands() {
    run_owned_upload_two_cpu_misses(true, false);
}

#[test]
#[ignore = "central GPU runner only; guarded CPU upload releases submitted recovery before controller ACK"]
fn gpu_owned_host_ready_recovers_multiple_cpu_layers_and_stage_tail() {
    for continuation in [false, true] {
        run_owned_upload_two_cpu_misses(continuation, true);
    }
}

fn run_owned_upload_two_cpu_misses(with_continuation: bool, host_ready: bool) {
    run_owned_upload_two_cpu_misses_outcome(with_continuation, host_ready, None);
}

#[test]
#[ignore = "central GPU runner only; failed host release skips recovery and next request remains correct"]
fn gpu_owned_host_ready_error_panic_and_pre_submit_cancel_preserve_next_request() {
    for outcome in [1, 2, 3] {
        run_owned_upload_two_cpu_misses_outcome(false, true, Some(outcome));
    }
}

fn run_owned_upload_two_cpu_misses_outcome(
    with_continuation: bool,
    host_ready: bool,
    outcome: Option<u8>,
) {
    let mut fixture =
        Fixture::new_with_copy_policy_and_kv_ready(1, true, Some(true), 0, 0, host_ready);
    if with_continuation {
        fixture.graph.push(Op::Add {
            a: fixture.frontend,
            b: fixture.frontend,
            dst: fixture.frontend,
            n: NE as u32,
        });
    }
    let plan = fixture.backend.compile(&fixture.graph).unwrap();
    fixture.preload();
    fixture
        .backend
        .execute(plan.as_ref(), &fixture.bindings())
        .unwrap();
    let expected = fixture.outputs.map(|id| fixture.read(fixture.buffer(id)));
    let (sources, seed) = {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        let previous = session.gpu_decode_owner.clone().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        let banks = fixture
            .roles
            .map(|roles| previous.role_params(session, &roles).unwrap());
        for roles in fixture.roles {
            session.discard_cpu_miss(roles, &[ROUTED[1]]);
        }
        session.gpu_decode_owner = None;
        if host_ready {
            // Keep the real dedicated queue/executor, but model the unimported RAM tail.
            // Keep the imported fixture buffers too: only these expert sources become CPU-only.
            session.test_unimport_host_uploads();
        }
        let mut sources = session.gpu_copy_sources().unwrap().unwrap();
        for roles in banks {
            for base in &roles[..3] {
                sources.test_cpu_only_source(*base as usize + ROUTED[1] as usize);
            }
        }
        (sources, crate::pager::owned_seed(session, 0).unwrap())
    };
    // Unified allocation can consult the pager. Match production ensure_gpu_copy: do not
    // retain its mutex across construction, then publish the finished owner under the lock.
    let owner = crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, LAYERS, seed)
        .unwrap()
        .unwrap();
    assert!(!owner.owned_self_service());
    fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .gpu_decode_owner = Some(Arc::clone(&owner));
    fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_mut()
        .unwrap()
        .gpu_copy = Some(owner);
    fixture.reset_frontend();
    OWNED_HOST_READY_TEST_ARMS.with(|arms| arms.set(0));
    OWNED_HOST_READY_TEST_OUTCOME.with(|value| value.set(outcome));
    if outcome.is_some() {
        for output in fixture.outputs {
            fixture
                .backend
                .upload(
                    fixture.buffer(output),
                    bytemuck::cast_slice(&vec![-777.0f32; NE]),
                )
                .unwrap();
        }
    }
    let counts = OwnedRecoveryCounts::begin();
    let result = fixture.backend.execute(plan.as_ref(), &fixture.bindings());
    if outcome.is_some() {
        assert!(
            result.is_err(),
            "injected failure must not return a successful forward"
        );
        OWNED_HOST_READY_TEST_ARMS.with(|arms| assert_eq!(arms.get(), 1));
        for output in fixture.outputs {
            assert_eq!(
                fixture.read(fixture.buffer(output)),
                vec![-777.0f32; NE]
            );
        }
        fixture.reset_frontend();
        fixture
            .backend
            .execute(plan.as_ref(), &fixture.bindings())
            .unwrap();
    } else {
        result.unwrap();
        OWNED_HOST_READY_TEST_ARMS
            .with(|arms| assert_eq!(arms.get(), if host_ready { 2 } else { 0 }));
        assert_eq!(counts.values(), [2, 2, usize::from(!with_continuation)],
            "both misses must enter and submit guarded recovery; only the proven empty boundary tail is discarded");
    }
    OWNED_HOST_READY_TEST_OUTCOME.with(|value| assert_eq!(value.get(), None));
    fixture.assert_frontend(
        fixture.buffer(fixture.frontend),
        (LAYERS * if with_continuation { 2 } else { 1 }) as f32,
    );
    assert_eq!(
        fixture.outputs.map(|id| fixture.read(fixture.buffer(id))),
        expected
    );
    fixture.backend.with_unified_exclusive(|| {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        for roles in fixture.roles {
            assert_eq!(
                session.routed_roles_resident_mask(&roles, &ROUTED).unwrap(),
                HIT_MASK
            );
        }
    });
}

#[test]
#[ignore = "central GPU runner only; ordinary native transfer must be submitted, not discarded"]
fn gpu_owned_upload_empty_tail_fix_preserves_normal_transfer() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let be = &fixture.backend;
    let plan = be.compile(&fixture.graph).unwrap();
    fixture.preload();
    be.execute(plan.as_ref(), &fixture.bindings()).unwrap();
    let mut retained = be.runtime_phase.lock().unwrap().resident_buffers.take();
    let mut chunk = ResidentChunk::new(be, &fixture.graph, true, &mut retained)
        .unwrap()
        .unwrap();
    let mut words = vec![0; crate::recorder::RESIDENT_CONTROL_BYTES / 4];
    words[0] = 1;
    words[1] = u32::MAX;
    be.upload(chunk.control.as_ref(), bytemuck::cast_slice(&words))
        .unwrap();
    let source = be.alloc(16, BufferUsage::Staging).unwrap();
    let target = be.alloc(16, BufferUsage::Readback).unwrap();
    let expected = [0x1020_3040u32, 2, 3, 4];
    be.upload(source.as_ref(), bytemuck::cast_slice(&expected))
        .unwrap();
    be.upload(target.as_ref(), &[0u8; 16]).unwrap();
    let recorder = be.recorder().unwrap();
    recorder.copy(source.as_ref(), 0, target.as_ref(), 0, 16);
    recorder.publish_host_reads();
    // Recorder::copy emits vkCmdCopyBuffer outside a conditional region. Its profiling
    // stamp counts that native transfer as one recorded operation, not a compute dispatch.
    assert_eq!(
        recorder.dispatches(),
        1,
        "copy_buffer is counted by the recorder stamp"
    );
    let counts = OwnedRecoveryCounts::begin();
    let mut rec = Some(recorder);
    let mut ps = PagedStream::default();
    assert!(!ps.guarded_empty_tail_recorder);
    let mut segments = std::collections::VecDeque::new();
    let mut prepare_tail = || {};
    chunk
        .finish_with_notice(
            be,
            &mut rec,
            &mut ps,
            &ScratchPool::default(),
            &fixture.graph,
            &HashMap::new(),
            &mut segments,
            true,
            &mut prepare_tail,
        )
        .unwrap();
    assert_eq!(counts.values(), [0, 0, 0]);
    let mut actual = [0u32; 4];
    be.download(target.as_ref(), bytemuck::cast_slice_mut(&mut actual))
        .unwrap();
    assert_eq!(
        actual, expected,
        "an ordinary native transfer must be submitted"
    );
    if let Some(rec) = rec {
        rec.discard().unwrap();
    }
    drop(chunk);
    be.runtime_phase.lock().unwrap().resident_buffers = retained;
}

#[path = "adapter_host_ready_combined_tests.rs"]
mod host_ready_combined_tests;
