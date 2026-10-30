#![cfg(test)]

//! Nested under resident_ahead_tests; model-free staged next-token packet contracts.
use super::*;
use infr_core::backend::{ExecutionStage, SegmentedKvSpec};
#[path = "adapter_staged_early_submit_tests.rs"]
mod early_submit_tests;

fn split_plans(fixture: &Fixture) -> (VkDecodePlan, VkDecodePlan, VkDecodePlan) {
    let mut head = fixture.graph.clone();
    head.ops.truncate(2);
    let mut tail = fixture.graph.clone();
    tail.ops.drain(..2);
    let extra = tail.internal(TensorDesc::new([1, NE * 2], DType::F32));
    // Make the tail's extra topology a real, fully written activation, not an unused tensor.
    tail.push(Op::Copy {
        src: fixture.frontend,
        src_off: 0,
        dst: extra,
        dst_off: 0,
        n: NE as u32,
    });
    tail.push(Op::Copy {
        src: fixture.frontend,
        src_off: 0,
        dst: extra,
        dst_off: NE as u32,
        n: NE as u32,
    });
    let next = head.clone();
    (
        VkDecodePlan::new_cpu(&head, false, true),
        VkDecodePlan::new_cpu(&tail, false, true),
        VkDecodePlan::new_cpu(&next, false, true),
    )
}

#[test]
fn staged_next_output_guard_empty_cancellation_and_unwind_are_inert() {
    let output = Mutex::new(None);
    drop(command_prefix::PreparedOutputGuard::new(Some(&output)));
    assert!(output.lock().unwrap().is_none());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = command_prefix::PreparedOutputGuard::new(Some(&output));
        panic!("synthetic preparation unwind");
    }));
    assert!(result.is_err());
    assert!(output.lock().unwrap().is_none());
    let mut guard = command_prefix::PreparedOutputGuard::new(None);
    guard.commit();
    drop(guard);
}

#[test]
#[ignore = "parent serial GPU only; retained heterogeneous topology and actual next-token reuse"]
fn gpu_staged_next_heterogeneous_workspace_reuses_prefix_and_recovers_miss() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    fixture.preload();
    let bindings = fixture.bindings();
    let full = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    execute_static(&fixture.backend, &full, &bindings).unwrap();
    let expected = fixture.outputs();
    let (head, tail, next) = split_plans(&fixture);
    let stages = [
        ExecutionStage::new(&head, &bindings),
        ExecutionStage::new(&tail, &bindings),
    ];
    let head_layout = scratch_layout(&head.graph).unwrap();
    let tail_layout = scratch_layout(&tail.graph).unwrap();
    assert_ne!(head_layout.len(), tail_layout.len());
    let head_scratch: ScratchSet = head_layout
        .iter()
        .map(|bytes| bytes.map(|n| Box::new(CpuBuffer(n)) as Box<dyn Buffer>))
        .collect();
    let tail_scratch: ScratchSet = tail_layout
        .iter()
        .map(|bytes| bytes.map(|n| Box::new(CpuBuffer(n)) as Box<dyn Buffer>))
        .collect();
    assert!(command_prefix::compatible(
        &head.graph,
        &next.graph,
        &head_scratch
    ));
    assert!(!command_prefix::compatible(
        &tail.graph,
        &next.graph,
        &tail_scratch
    ));
    for missing in [false, true] {
        fixture.reset_frontend();
        fixture
            .backend
            .execute_staged(&stages, &mut |_| Ok(()), Some((&next, &bindings, &[1])))
            .unwrap();
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
        assert_close(&fixture.outputs(), &expected);
        assert!(
            next.prepared_prefix.lock().unwrap().is_some(),
            "heterogeneous tail must publish next-L0 packet from retained L0 workspace"
        );
        if missing {
            let mut guard = fixture.backend.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            session.handoff_gpu_decode(&fixture.backend).unwrap();
            session.discard_cpu_miss(fixture.roles[0], &[ROUTED[0]]);
        }
        fixture.reset_frontend();
        fixture
            .backend
            .execute_staged(
                &[
                    ExecutionStage::new(&next, &bindings),
                    ExecutionStage::new(&tail, &bindings),
                ],
                &mut |_| Ok(()),
                None,
            )
            .unwrap();
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
        assert_close(&fixture.outputs(), &expected);
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        for roles in &fixture.roles {
            assert_eq!(
                session.routed_roles_resident_mask(roles, &ROUTED).unwrap(),
                HIT_MASK
            );
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; snapshot canonical bindings, positions, cancellation and unwind"]
fn gpu_staged_next_snapshot_rejects_changes_and_guard_discards_recorded_packets() {
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let mut graph = fixture.graph.clone();
    graph.ops.truncate(2);
    let Op::MoeFfn { x, .. } = graph.ops[1] else {
        unreachable!()
    };
    let position = graph.input(TensorDesc::new([1], DType::I32));
    let norm = graph.weight(TensorDesc::new([NE], DType::F32));
    let rotated = graph.internal(TensorDesc::new([1, NE], DType::F16));
    graph.ops.insert(
        1,
        Op::QkNormRope {
            x,
            weight: norm,
            positions: position,
            dst: rotated,
            rows: 1,
            n_head: 1,
            head_dim: NE as u32,
            rope_dim: NE as u32,
            theta: 10000.0,
            eps: 1e-6,
            freq_factors: None,
            x_stride: 0,
        },
    );
    for (id, bytes, usage) in [
        (
            position,
            bytemuck::cast_slice(&[0u32]).to_vec(),
            BufferUsage::Staging,
        ),
        (
            norm,
            bytemuck::cast_slice(&[1.0f32; NE]).to_vec(),
            BufferUsage::Weights,
        ),
    ] {
        let buffer = fixture.backend.alloc(bytes.len(), usage).unwrap();
        fixture.backend.upload(buffer.as_ref(), &bytes).unwrap();
        fixture.buffers.push((id, buffer));
    }
    let next = VkDecodePlan::new_cpu(&graph, false, true);
    let scratch = alloc_scratch(&fixture.backend, &graph).unwrap();
    let replacement = fixture.backend.alloc(NE * 4, BufferUsage::Staging).unwrap();
    let bindings = fixture.bindings();
    let mut pool = ScratchPool::default();
    fixture.reset_frontend();
    fixture.backend.with_unified_exclusive(|| {
        // A cancelled job never publishes or executes its captured frontend.
        let job = command_prefix::StageRecording::snapshot_next_token(
            &fixture.backend,
            &next,
            &bindings,
            &scratch,
            &mut pool,
            &[1],
        )
        .unwrap()
        .unwrap();
        assert!(job.is_next_token());
        job.record(&fixture.backend, &AtomicBool::new(true))
            .unwrap();
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        for mode in 0..3 {
            fixture
                .backend
                .upload(fixture.buffer(position), bytemuck::cast_slice(&[1u32]))
                .unwrap();
            let job = command_prefix::StageRecording::snapshot_next_token(
                &fixture.backend,
                &next,
                &bindings,
                &scratch,
                &mut pool,
                &[1],
            )
            .unwrap()
            .unwrap();
            job.record(&fixture.backend, &AtomicBool::new(false))
                .unwrap();
            assert!(next.prepared_prefix.lock().unwrap().is_some());
            let packet = next.prepared_prefix.lock().unwrap().take().unwrap();
            let mut changed = fixture.bindings();
            if mode == 1 {
                changed.bind(fixture.frontend, replacement.as_ref());
            }
            if mode == 2 {
                fixture
                    .backend
                    .upload(fixture.buffer(position), bytemuck::cast_slice(&[2u32]))
                    .unwrap();
            }
            let activated =
                packet.take_if_valid(&fixture.backend, &scratch, &mut pool, &changed, None, true);
            assert_eq!(
                activated.is_some(),
                mode == 0,
                "only exact canonical scratch, bindings and position may activate"
            );
            drop(activated);
        }
        // Exercise populated guards, not only empty Option teardown.
        for unwind in [false, true] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = command_prefix::PreparedOutputGuard::new(Some(&next.prepared_prefix));
                let job = command_prefix::StageRecording::snapshot_next_token(
                    &fixture.backend,
                    &next,
                    &bindings,
                    &scratch,
                    &mut pool,
                    &[1],
                )
                .unwrap()
                .unwrap();
                job.record(&fixture.backend, &AtomicBool::new(false))
                    .unwrap();
                assert!(next.prepared_prefix.lock().unwrap().is_some());
                if unwind {
                    panic!("synthetic failure after next packet publication");
                }
            }));
            assert_eq!(result.is_err(), unwind);
            assert!(next.prepared_prefix.lock().unwrap().is_none());
        }
        let mut guard = command_prefix::PreparedOutputGuard::new(Some(&next.prepared_prefix));
        let job = command_prefix::StageRecording::snapshot_next_token(
            &fixture.backend,
            &next,
            &bindings,
            &scratch,
            &mut pool,
            &[1],
        )
        .unwrap()
        .unwrap();
        job.record(&fixture.backend, &AtomicBool::new(false))
            .unwrap();
        guard.commit();
        drop(guard);
        assert!(next.prepared_prefix.lock().unwrap().is_some());
        next.prepared_prefix.lock().unwrap().take();
    });
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 0.0);
}

#[test]
#[ignore = "parent serial GPU only; staged callback failure must clear next output and not rerun L0"]
fn gpu_staged_next_callback_failure_clears_output_without_running_next_frontend() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    fixture.preload();
    let bindings = fixture.bindings();
    let full = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    execute_static(&fixture.backend, &full, &bindings).unwrap();
    let (head, tail, next) = split_plans(&fixture);
    let stages = [
        ExecutionStage::new(&head, &bindings),
        ExecutionStage::new(&tail, &bindings),
    ];
    fixture
        .backend
        .execute_staged(&stages, &mut |_| Ok(()), Some((&next, &bindings, &[1])))
        .unwrap();
    assert!(next.prepared_prefix.lock().unwrap().is_some());
    let sentinel = [-19.0f32; NE];
    fixture
        .backend
        .upload(
            fixture.buffer(fixture.outputs[1]),
            bytemuck::cast_slice(&sentinel),
        )
        .unwrap();
    fixture.reset_frontend();
    let error = fixture
        .backend
        .execute_staged(
            &stages,
            &mut |index| {
                if index == 1 {
                    Err(be("synthetic callback failure before tail"))
                } else {
                    Ok(())
                }
            },
            Some((&next, &bindings, &[1])),
        )
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("synthetic callback failure before tail"));
    assert!(
        next.prepared_prefix.lock().unwrap().is_none(),
        "old or newly recorded next output must not survive an errored forward"
    );
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
    assert_eq!(fixture.read(fixture.buffer(fixture.outputs[1])), sentinel);
}

struct TwoLaneFixture {
    fixture: Fixture,
    position: TensorId,
    caches: [TensorId; 2],
    states: Vec<Box<dyn Buffer>>,
    initial_positions: [u32; 2],
}

impl TwoLaneFixture {
    const CACHE_ROWS: usize = 16;

    fn new() -> Self {
        Self::new_with_kv(None, [3, 7])
    }

    fn segmented_spec() -> SegmentedKvSpec {
        let segment_elements = 4 * NE;
        assert!(segment_elements.is_power_of_two());
        SegmentedKvSpec {
            logical_bytes: DType::F16.dense_bytes(Self::CACHE_ROWS * NE).unwrap(),
            segment_bytes: DType::F16.dense_bytes(segment_elements).unwrap(),
            segment_elements,
            max_segments: Self::CACHE_ROWS / 4,
        }
    }

    fn new_with_kv(segmented: Option<SegmentedKvSpec>, initial_positions: [u32; 2]) -> Self {
        let mut fixture = Fixture::new_with_copy_policy_and_kv(
            1,
            true,
            Some(true),
            segmented.map_or(0, |spec| 4 * spec.logical_bytes),
            segmented.map_or(0, |spec| spec.segment_bytes),
        );
        for (index, tensor) in fixture.graph.tensors.iter_mut().enumerate() {
            if tensor.desc.shape.as_slice() != [1, NE] {
                continue;
            }
            tensor.desc = TensorDesc::new([2, NE], tensor.desc.dtype);
            if let Some((_, buffer)) = fixture
                .buffers
                .iter_mut()
                .find(|(id, _)| id.0 as usize == index)
            {
                let mut row = vec![0u8; NE * 4];
                fixture.backend.download(buffer.as_ref(), &mut row).unwrap();
                let replacement = fixture
                    .backend
                    .alloc(2 * NE * 4, BufferUsage::Staging)
                    .unwrap();
                let values = [row.as_slice(), row.as_slice()].concat();
                fixture
                    .backend
                    .upload(replacement.as_ref(), &values)
                    .unwrap();
                *buffer = replacement;
            }
        }
        for op in &mut fixture.graph.ops {
            if let Op::Add { n, .. } = op {
                *n = (2 * NE) as u32;
            }
        }
        fixture.graph.independent_rows = true;
        fixture.graph.sequence_spans = initial_positions
            .into_iter()
            .enumerate()
            .map(|(lane, start_pos)| SequenceSpan {
                row_start: lane as u32,
                rows: 1,
                start_pos,
            })
            .collect();
        let position = fixture.graph.input(TensorDesc::new([2], DType::I32));
        let norm = fixture.graph.weight(TensorDesc::new([NE], DType::F32));
        let rotated = fixture.graph.internal(TensorDesc::new([2, NE], DType::F16));
        let caches = std::array::from_fn(|_| {
            fixture
                .graph
                .input(TensorDesc::new([Self::CACHE_ROWS, NE], DType::F16))
        });
        let attended = fixture.graph.internal(TensorDesc::new([2, NE], DType::F32));
        let Op::MoeFfn { x, .. } = fixture.graph.ops[1] else {
            unreachable!()
        };
        let input = x;
        if let Op::MoeFfn { x, .. } = &mut fixture.graph.ops[1] {
            *x = attended;
        }
        fixture.graph.ops.splice(
            1..1,
            [
                Op::QkNormRope {
                    x: input,
                    weight: norm,
                    positions: position,
                    dst: rotated,
                    rows: 2,
                    n_head: 1,
                    head_dim: NE as u32,
                    rope_dim: NE as u32,
                    theta: 10000.0,
                    eps: 1e-6,
                    freq_factors: None,
                    x_stride: 0,
                },
                Op::WriteKv {
                    src: rotated,
                    cache: caches[0],
                    rows: 2,
                    row_stride: NE as u32,
                    pos: 0,
                },
                Op::WriteKv {
                    src: input,
                    cache: caches[1],
                    rows: 2,
                    row_stride: NE as u32,
                    pos: 0,
                },
                Op::Attention {
                    q: rotated,
                    k_cache: caches[0],
                    v_cache: caches[1],
                    dst: attended,
                    rows: 2,
                    kv_len: 8,
                    n_head: 1,
                    n_kv: 1,
                    head_dim: NE as u32,
                    scale: 1.0 / (NE as f32).sqrt(),
                    mask: AttnMask::Causal,
                    pos: 0,
                    sinks: None,
                },
            ],
        );
        for (id, data, usage) in [
            (
                position,
                bytemuck::cast_slice(&initial_positions).to_vec(),
                BufferUsage::Staging,
            ),
            (
                norm,
                bytemuck::cast_slice(&[1.0f32; NE]).to_vec(),
                BufferUsage::Weights,
            ),
        ] {
            let buffer = fixture.backend.alloc(data.len(), usage).unwrap();
            fixture.backend.upload(buffer.as_ref(), &data).unwrap();
            fixture.buffers.push((id, buffer));
        }
        let states = (0..4)
            .map(|_| {
                if let Some(spec) = segmented {
                    fixture
                        .backend
                        .alloc_segmented_kv(spec)
                        .unwrap()
                        .expect("segmented Vulkan fixture cache")
                } else {
                    fixture
                        .backend
                        .alloc(
                            DType::F16.dense_bytes(Self::CACHE_ROWS * NE).unwrap(),
                            BufferUsage::KvCache,
                        )
                        .unwrap()
                }
            })
            .collect();
        let result = Self {
            fixture,
            position,
            caches,
            states,
            initial_positions,
        };
        if let Some(spec) = segmented {
            for (lane, position) in initial_positions.into_iter().enumerate() {
                result.grow_lane(
                    lane,
                    ((position as usize + 1) * NE).div_ceil(spec.segment_elements),
                );
            }
        }
        result.reset();
        result
    }

    fn bindings(&self, order: [usize; 2]) -> Bindings<'_> {
        let mut bindings = self.fixture.bindings();
        for (side, &id) in self.caches.iter().enumerate() {
            bindings.bind_rows(
                id,
                order
                    .iter()
                    .map(|&lane| self.states[side * 2 + lane].as_ref())
                    .collect(),
            );
        }
        bindings
    }

    fn positions(&self, positions: [u32; 2]) {
        self.fixture
            .backend
            .upload(
                self.fixture.buffer(self.position),
                bytemuck::cast_slice(&positions),
            )
            .unwrap();
    }

    fn reset(&self) {
        self.fixture
            .backend
            .upload(
                self.fixture.buffer(self.fixture.frontend),
                bytemuck::cast_slice(&vec![0f32; 2 * NE]),
            )
            .unwrap();
        for (index, buffer) in self.states.iter().enumerate() {
            // Distinct nonzero histories expose crossed cache bindings, not just output shapes.
            let values = vec![
                half::f16::from_f32(0.1 + index as f32 / 10.0).to_bits();
                Self::backed_bytes(buffer.as_ref())
                    / DType::F16.dense_bytes(1).unwrap()
            ];
            self.fixture
                .backend
                .upload_range(buffer.as_ref(), 0, bytemuck::cast_slice(&values))
                .unwrap();
        }
        self.positions(self.initial_positions);
    }

    fn backed_bytes(buffer: &dyn Buffer) -> usize {
        crate::as_segmented_kv(buffer).map_or_else(
            || buffer.len_bytes(),
            |kv| kv.committed() * kv.spec().segment_bytes,
        )
    }

    fn grow_lane(&self, lane: usize, segments: usize) {
        self.fixture
            .backend
            .ensure_segmented_kv_batch(
                &[self.states[lane].as_ref(), self.states[2 + lane].as_ref()],
                segments,
            )
            .unwrap();
    }

    fn state(&self) -> Vec<Vec<u8>> {
        self.fixture
            .outputs
            .iter()
            .map(|&id| self.fixture.buffer(id))
            .chain(self.states.iter().map(|buffer| buffer.as_ref()))
            .chain(std::iter::once(self.fixture.buffer(self.fixture.frontend)))
            .map(|buffer| {
                let mut data = vec![0; Self::backed_bytes(buffer)];
                self.fixture
                    .backend
                    .download_range(buffer, 0, &mut data)
                    .unwrap();
                data
            })
            .collect()
    }

    fn plans(&self) -> (VkDecodePlan, VkDecodePlan) {
        let current = VkDecodePlan::new_cpu(&self.fixture.graph, false, true);
        let mut next = self.fixture.graph.clone();
        for span in &mut next.sequence_spans {
            span.start_pos += 1;
        }
        (current, VkDecodePlan::new_cpu(&next, false, true))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SegmentedCacheSnapshot {
    identity: (usize, usize, Option<u64>, u64, usize, usize),
    committed: usize,
    addresses: Vec<u64>,
    bytes: Vec<u8>,
}

fn segmented_cache_snapshots(multi: &TwoLaneFixture) -> Vec<SegmentedCacheSnapshot> {
    use ash::vk::Handle;
    multi
        .states
        .iter()
        .map(|buffer| {
            let buffer = buffer.as_ref();
            let kv = crate::as_segmented_kv(buffer).expect("segmented cache fixture");
            let table = kv.table_buffer();
            let mut addresses = vec![0u8; kv.spec().max_segments * std::mem::size_of::<u64>()];
            multi
                .fixture
                .backend
                .download(table, &mut addresses)
                .unwrap();
            let mut bytes = vec![0; TwoLaneFixture::backed_bytes(buffer)];
            multi
                .fixture
                .backend
                .download_range(buffer, 0, &mut bytes)
                .unwrap();
            SegmentedCacheSnapshot {
                identity: (
                    buffer as *const dyn Buffer as *const () as usize,
                    buffer.len_bytes(),
                    buffer.device_addr(),
                    table.buffer.as_raw(),
                    table.sub_offset,
                    kv.spec().segment_elements,
                ),
                committed: kv.committed(),
                addresses: addresses
                    .chunks_exact(8)
                    .map(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect(),
                bytes,
            }
        })
        .collect()
}

fn assert_segment_growth_preserves_prefix(
    before: &[SegmentedCacheSnapshot],
    after: &[SegmentedCacheSnapshot],
    lane: usize,
) {
    assert_eq!(before.len(), 4);
    assert_eq!(after.len(), 4);
    for (index, (before, after)) in before.iter().zip(after).enumerate() {
        assert_eq!(
            after.identity, before.identity,
            "growth must retain object, logical extent and address-table identity"
        );
        assert_eq!(
            &after.bytes[..before.bytes.len()],
            before.bytes.as_slice(),
            "growth changed committed cache prefix {index}"
        );
        assert_eq!(
            &after.addresses[..before.committed],
            &before.addresses[..before.committed]
        );
        if index % 2 == lane {
            assert_eq!(after.committed, before.committed + 1);
            assert_eq!(before.addresses[before.committed], 0);
            assert_ne!(after.addresses[before.committed], 0);
            assert!(after.addresses[after.committed..]
                .iter()
                .all(|&address| address == 0));
        } else {
            assert_eq!(after, before);
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; uncommitted independent KV boundary declines, then grows and resumes exactly"]
fn gpu_prepared_segmented_kv_declines_uncommitted_next_lane_boundary() {
    let spec = TwoLaneFixture::segmented_spec();
    let rows_per_segment = spec.segment_elements / NE;
    let current_positions = [
        (rows_per_segment - 2) as u32,
        (2 * rows_per_segment - 1) as u32,
    ];
    let next_positions = current_positions.map(|position| position + 1);
    let next_usize = next_positions.map(|position| position as usize);
    let multi = TwoLaneFixture::new_with_kv(Some(spec), current_positions);
    let fixture = &multi.fixture;
    fixture.preload();
    let bindings = multi.bindings([0, 1]);
    let (current, next) = multi.plans();
    assert_eq!(
        segmented_cache_snapshots(&multi)
            .iter()
            .map(|cache| cache.committed)
            .collect::<Vec<_>>(),
        vec![1, 2, 1, 2]
    );
    execute_static(&fixture.backend, &current, &bindings).unwrap();
    let before = segmented_cache_snapshots(&multi);
    let before_state = multi.state();
    fixture.backend.with_unified_exclusive(|| {
        let mut phase = fixture.backend.runtime_phase.lock().unwrap();
        let RuntimePhaseArena { scratch, pool, .. } = &mut *phase;
        assert!(command_prefix::StageRecording::snapshot_next_token(
            &fixture.backend,
            &next,
            &bindings,
            scratch,
            pool,
            &next_usize
        )
        .unwrap()
        .is_none());
    });
    assert_eq!(segmented_cache_snapshots(&multi), before);
    assert_eq!(
        multi.state(),
        before_state,
        "declined preparation must not mutate or execute either lane"
    );

    multi.reset();
    execute_static(&fixture.backend, &current, &bindings).unwrap();
    let expected_current = multi.state();
    multi.reset();
    execute_static_next(
        &fixture.backend,
        &current,
        &bindings,
        Some((&next, &bindings, &next_usize)),
    )
    .unwrap();
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(
        multi.state(),
        expected_current,
        "uncommitted lookahead must retain ordinary current-forward behavior"
    );

    let before_growth = segmented_cache_snapshots(&multi);
    multi.grow_lane(1, 3);
    assert_segment_growth_preserves_prefix(&before_growth, &segmented_cache_snapshots(&multi), 1);
    // Establish an ordinary reference only after the newly backed ranges are initialized.
    multi.reset();
    execute_static(&fixture.backend, &current, &bindings).unwrap();
    multi.positions(next_positions);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    let expected = multi.state();
    multi.reset();
    execute_static_next(
        &fixture.backend,
        &current,
        &bindings,
        Some((&next, &bindings, &next_usize)),
    )
    .unwrap();
    assert!(
        next.prepared_prefix.lock().unwrap().is_some(),
        "committed next depth must restore preparation"
    );
    multi.positions(next_positions);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(
        multi.state(),
        expected,
        "post-growth prepared outputs and every committed KV byte must match ordinary Decode"
    );
}

#[test]
#[ignore = "parent serial GPU only; same-object segmented KV growth invalidates an existing prepared packet"]
fn gpu_prepared_segmented_kv_growth_invalidates_same_object_packet() {
    let spec = TwoLaneFixture::segmented_spec();
    let rows_per_segment = spec.segment_elements / NE;
    let current_positions = [
        (rows_per_segment - 2) as u32,
        (2 * rows_per_segment - 2) as u32,
    ];
    let next_positions = current_positions.map(|position| position + 1);
    let next_usize = next_positions.map(|position| position as usize);
    let multi = TwoLaneFixture::new_with_kv(Some(spec), current_positions);
    let fixture = &multi.fixture;
    fixture.preload();
    let bindings = multi.bindings([0, 1]);
    let (_, next) = multi.plans();
    multi.positions(next_positions);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    let before = segmented_cache_snapshots(&multi);
    let before_state = multi.state();
    let epoch_before = fixture
        .backend
        .shared
        .resource_epoch
        .load(Ordering::Acquire);
    let packet = fixture.backend.with_unified_exclusive(|| {
        let mut phase = fixture.backend.runtime_phase.lock().unwrap();
        let RuntimePhaseArena { scratch, pool, .. } = &mut *phase;
        // A no-growth control proves the same scratch, positions and row bindings can activate.
        for control in [true, false] {
            let job = command_prefix::StageRecording::snapshot_next_token(
                &fixture.backend,
                &next,
                &bindings,
                scratch,
                pool,
                &next_usize,
            )
            .unwrap()
            .unwrap();
            job.record(&fixture.backend, &AtomicBool::new(false))
                .unwrap();
            let packet = next.prepared_prefix.lock().unwrap().take().unwrap();
            if control {
                assert!(packet
                    .take_if_valid(&fixture.backend, scratch, pool, &bindings, None, true)
                    .is_some());
            } else {
                return packet;
            }
        }
        unreachable!()
    });
    assert_eq!(
        multi.state(),
        before_state,
        "recording and control activation cannot submit the prefix"
    );
    multi.grow_lane(1, 3);
    let after_growth = segmented_cache_snapshots(&multi);
    assert_segment_growth_preserves_prefix(&before, &after_growth, 1);
    // These are the segmented-key components; no object/table replacement can hide the depth change.
    assert_ne!(
        (&before[1].identity, before[1].committed),
        (&after_growth[1].identity, after_growth[1].committed)
    );
    assert_ne!(
        (&before[3].identity, before[3].committed),
        (&after_growth[3].identity, after_growth[3].committed)
    );
    let epoch_after = fixture
        .backend
        .shared
        .resource_epoch
        .load(Ordering::Acquire);
    eprintln!("segmented packet growth: epoch {epoch_before}->{epoch_after}; lane1 committed 2->3, object/table identities retained");
    fixture.backend.with_unified_exclusive(|| {
        let mut phase = fixture.backend.runtime_phase.lock().unwrap();
        let RuntimePhaseArena { scratch, pool, .. } = &mut *phase;
        assert!(packet
            .take_if_valid(&fixture.backend, scratch, pool, &bindings, None, true)
            .is_none());
    });
    assert_eq!(
        segmented_cache_snapshots(&multi),
        after_growth,
        "rejection must not mutate any cache or table"
    );
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    multi.reset();
    let initialized = segmented_cache_snapshots(&multi);
    multi.positions(next_positions);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    let actual = multi.state();
    assert_eq!(actual.len(), before_state.len());
    for (actual, expected) in actual.iter().zip(&before_state) {
        assert_eq!(&actual[..expected.len()], expected.as_slice(), "ordinary continuation must retain the pre-growth result for every previously committed byte");
    }
    for ((old, initialized), actual) in before
        .iter()
        .zip(&initialized)
        .zip(segmented_cache_snapshots(&multi))
    {
        assert_eq!(
            &actual.bytes[old.bytes.len()..],
            &initialized.bytes[old.bytes.len()..],
            "the continued forward must not write newly committed but invisible cache rows"
        );
    }
}

#[test]
#[ignore = "parent serial GPU only; two independent unequal-position rows and exact KV/output parity"]
fn gpu_prepared_two_lane_unequal_positions_matches_ordinary_kv_and_outputs() {
    let multi = TwoLaneFixture::new();
    let fixture = &multi.fixture;
    fixture.preload();
    let bindings = multi.bindings([0, 1]);
    let (current, next) = multi.plans();
    // Warm both attention scratch geometries before recording retained commands.
    execute_static(&fixture.backend, &current, &bindings).unwrap();
    multi.positions([4, 8]);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    multi.reset();
    execute_static(&fixture.backend, &current, &bindings).unwrap();
    multi.positions([4, 8]);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    let expected = multi.state();
    multi.reset();
    execute_static_next(
        &fixture.backend,
        &current,
        &bindings,
        Some((&next, &bindings, &[4, 8])),
    )
    .unwrap();
    assert!(
        next.prepared_prefix.lock().unwrap().is_some(),
        "must actually prepare a two-lane packet"
    );
    multi.positions([4, 8]);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(
        multi.state(),
        expected,
        "all output bytes and each lane's K/V state must match ordinary Decode"
    );
}

#[test]
#[ignore = "parent serial GPU only; two-lane binding/order/position rejection, cancellation and unwind"]
fn gpu_prepared_two_lane_guards_reject_partial_advance_rebind_and_eos() {
    let multi = TwoLaneFixture::new();
    let fixture = &multi.fixture;
    fixture.preload();
    let bindings = multi.bindings([0, 1]);
    let (_, next) = multi.plans();
    multi.positions([4, 8]);
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    let state_before = multi.state();
    let replacement = fixture
        .backend
        .alloc(16 * NE * 2, BufferUsage::KvCache)
        .unwrap();
    fixture.backend.with_unified_exclusive(|| {
        let mut phase = fixture.backend.runtime_phase.lock().unwrap();
        let RuntimePhaseArena { scratch, pool, .. } = &mut *phase;
        for mode in 0..6 {
            multi.positions([4, 8]);
            let job = command_prefix::StageRecording::snapshot_next_token(
                &fixture.backend,
                &next,
                &bindings,
                scratch,
                pool,
                &[4, 8],
            )
            .unwrap()
            .unwrap();
            job.record(&fixture.backend, &AtomicBool::new(false))
                .unwrap();
            let packet = next.prepared_prefix.lock().unwrap().take().unwrap();
            let mut changed = multi.bindings(if mode == 1 { [1, 0] } else { [0, 1] });
            if mode == 2 {
                changed.bind_rows(multi.caches[0], vec![multi.states[0].as_ref()]);
            }
            if mode == 3 {
                multi.positions([4, 7]);
            }
            if mode == 4 {
                multi.positions([8, 4]);
            }
            if mode == 5 {
                changed.bind_rows(
                    multi.caches[0],
                    vec![multi.states[0].as_ref(), replacement.as_ref()],
                );
            }
            let activated =
                packet.take_if_valid(&fixture.backend, scratch, pool, &changed, None, true);
            assert_eq!(activated.is_some(), mode == 0, "guard case {mode}");
            drop(activated);
        }
        multi.positions([4, 8]);
        // An EOS/cancelled candidate and a forward failure both discard unsubmitted work.
        let job = command_prefix::StageRecording::snapshot_next_token(
            &fixture.backend,
            &next,
            &bindings,
            scratch,
            pool,
            &[4, 8],
        )
        .unwrap()
        .unwrap();
        job.record(&fixture.backend, &AtomicBool::new(true))
            .unwrap();
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        for unwind in [false, true] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = command_prefix::PreparedOutputGuard::new(Some(&next.prepared_prefix));
                let job = command_prefix::StageRecording::snapshot_next_token(
                    &fixture.backend,
                    &next,
                    &bindings,
                    scratch,
                    pool,
                    &[4, 8],
                )
                .unwrap()
                .unwrap();
                job.record(&fixture.backend, &AtomicBool::new(false))
                    .unwrap();
                assert!(next.prepared_prefix.lock().unwrap().is_some());
                if unwind {
                    panic!("two-lane forward failure after publication");
                }
            }));
            assert_eq!(result.is_err(), unwind);
            assert!(next.prepared_prefix.lock().unwrap().is_none());
        }
    });
    assert_eq!(
        multi.state(),
        state_before,
        "recording/rejected activation must not execute or mutate any lane"
    );
}

// Separate from the existing segmented tests: each stage has a distinct namespace and
// writes real KV. The tail additionally consumes an input uploaded only by its callback.
struct StagedBundleFixture {
    multi: TwoLaneFixture,
    tail_input: TensorId,
    tail_buffer: Box<dyn Buffer>,
    plans: [[VkDecodePlan; 2]; 2],
}

impl StagedBundleFixture {
    fn new() -> Self {
        let multi = TwoLaneFixture::new();
        let mut head = multi.fixture.graph.clone();
        let boundary = head
            .ops
            .iter()
            .position(|op| matches!(op, Op::MoeFfn { .. }))
            .unwrap()
            + 1;
        head.ops.truncate(boundary);
        let mut tail = multi.fixture.graph.clone();
        tail.ops.drain(..boundary);
        let tail_input = tail.input(TensorDesc::new([2, NE], DType::F32));
        let merged = tail.internal(TensorDesc::new([2, NE], DType::F32));
        let rotated = tail.internal(TensorDesc::new([2, NE], DType::F16));
        let Op::MoeFfn { x, .. } = &mut tail.ops[1] else {
            unreachable!()
        };
        let preceding = *x;
        *x = merged;
        let mut rope = head
            .ops
            .iter()
            .find(|op| matches!(op, Op::QkNormRope { .. }))
            .unwrap()
            .clone();
        if let Op::QkNormRope { x, dst, .. } = &mut rope {
            *x = merged;
            *dst = rotated;
        }
        tail.ops.splice(
            1..1,
            [
                Op::Add {
                    a: preceding,
                    b: tail_input,
                    dst: merged,
                    n: (2 * NE) as u32,
                },
                rope,
                Op::WriteKv {
                    src: rotated,
                    cache: multi.caches[0],
                    rows: 2,
                    row_stride: NE as u32,
                    pos: 0,
                },
            ],
        );
        let current = [
            VkDecodePlan::new_cpu(&head, false, true),
            VkDecodePlan::new_cpu(&tail, false, true),
        ];
        for graph in [&mut head, &mut tail] {
            for span in &mut graph.sequence_spans {
                span.start_pos += 1;
            }
        }
        let next = [
            VkDecodePlan::new_cpu(&head, false, true),
            VkDecodePlan::new_cpu(&tail, false, true),
        ];
        assert_ne!(
            scratch_layout(&current[0].graph).unwrap().len(),
            scratch_layout(&current[1].graph).unwrap().len()
        );
        let tail_buffer = multi
            .fixture
            .backend
            .alloc(2 * NE * 4, BufferUsage::Staging)
            .unwrap();
        let result = Self {
            multi,
            tail_input,
            tail_buffer,
            plans: [current, next],
        };
        result.reset();
        result
    }

    fn bindings(&self) -> Bindings<'_> {
        let mut bindings = self.multi.bindings([0, 1]);
        bindings.bind(self.tail_input, self.tail_buffer.as_ref());
        bindings
    }

    fn stages<'a>(&'a self, token: usize, bindings: &'a Bindings<'a>) -> [ExecutionStage<'a>; 2] {
        self.plans[token]
            .each_ref()
            .map(|plan| ExecutionStage::new(plan, bindings))
    }

    fn upload_tail(&self, token: usize) {
        let values: Vec<f32> = (0..2 * NE)
            .map(|word| 0.125 + token as f32 * 0.375 + word as f32 / (8 * NE) as f32)
            .collect();
        self.multi
            .fixture
            .backend
            .upload(self.tail_buffer.as_ref(), bytemuck::cast_slice(&values))
            .unwrap();
    }

    fn ready(&self, token: usize, stage: usize) {
        if stage == 0 {
            self.multi
                .positions(if token == 0 { [3, 7] } else { [4, 8] });
        } else {
            assert_eq!(stage, 1);
            self.upload_tail(token);
        }
    }

    fn reset(&self) {
        self.multi.reset();
        self.upload_tail(0);
        for &output in &self.multi.fixture.outputs {
            self.multi
                .fixture
                .backend
                .upload(
                    self.multi.fixture.buffer(output),
                    bytemuck::cast_slice(&[-23.0f32; 2 * NE]),
                )
                .unwrap();
        }
    }

    fn clear_packets(&self) {
        for plan in self.plans.iter().flatten() {
            plan.prepared_prefix.lock().unwrap().take();
        }
    }

    fn ordinary(&self, token: usize) {
        let bindings = self.bindings();
        let stages = self.stages(token, &bindings);
        self.multi
            .fixture
            .backend
            .execute_staged(
                &stages,
                &mut |stage| {
                    self.ready(token, stage);
                    Ok(())
                },
                None,
            )
            .unwrap();
    }

    fn warm(&self) {
        self.multi.fixture.preload();
        self.ordinary(0);
        self.ordinary(1);
        self.clear_packets();
        self.reset();
    }

    fn state(&self) -> Vec<Vec<u8>> {
        let mut state = self.multi.state();
        for buffer in [
            self.multi.fixture.buffer(self.multi.position),
            self.tail_buffer.as_ref(),
        ] {
            let mut bytes = vec![0; buffer.len_bytes()];
            self.multi
                .fixture
                .backend
                .download(buffer, &mut bytes)
                .unwrap();
            state.push(bytes);
        }
        state
    }

    fn assert_frontend(&self, count: f32) {
        let mut values = [0f32; 2 * NE];
        self.multi
            .fixture
            .backend
            .download(
                self.multi.fixture.buffer(self.multi.fixture.frontend),
                bytemuck::cast_slice_mut(&mut values),
            )
            .unwrap();
        assert_eq!(
            values,
            [count; 2 * NE],
            "each authorized stage frontend runs exactly once"
        );
    }

    fn future_kv_rows(&self, state: &[Vec<u8>]) -> Vec<Vec<u8>> {
        (0..4)
            .map(|cache| {
                let position = [4, 8][cache % 2];
                let start = position * NE * 2;
                state[self.multi.fixture.outputs.len() + cache][start..start + NE * 2].to_vec()
            })
            .collect()
    }
}

#[test]
#[ignore = "parent serial GPU only; both future stage packets, callback input and exact KV/output parity"]
fn gpu_staged_lookahead_publishes_both_stages_without_running_before_ack() {
    use infr_core::backend::ExecutionLookahead;
    let fixture = StagedBundleFixture::new();
    fixture.warm();
    fixture.ordinary(0);
    let current_expected = fixture.state();
    fixture.ordinary(1);
    let next_expected = fixture.state();
    fixture.reset();
    let bindings = fixture.bindings();
    let current = fixture.stages(0, &bindings);
    let next = fixture.stages(1, &bindings);
    let positions = [4usize, 8];
    let mut callbacks = Vec::new();
    fixture
        .multi
        .fixture
        .backend
        .execute_staged_lookahead(
            &current,
            &mut |stage| {
                callbacks.push(stage);
                fixture.ready(0, stage);
                Ok(())
            },
            Some(ExecutionLookahead::new(&next, &positions)),
        )
        .unwrap();
    assert_eq!(callbacks, [0, 1]);
    for plan in &fixture.plans[1] {
        assert!(
            plan.prepared_prefix.lock().unwrap().is_some(),
            "both heterogeneous future stages must actually publish a packet"
        );
    }
    assert_eq!(
        fixture.state(),
        current_expected,
        "recording future L0/tail must not advance inputs, outputs or KV"
    );
    fixture.assert_frontend(2.0);
    callbacks.clear();
    fixture
        .multi
        .fixture
        .backend
        .execute_staged_lookahead(
            &next,
            &mut |stage| {
                callbacks.push(stage);
                if stage == 0 {
                    assert_eq!(
                        fixture.state(),
                        current_expected,
                        "no future command may execute before the next first-stage callback"
                    );
                    assert!(fixture.plans[1][0]
                        .prepared_prefix
                        .lock()
                        .unwrap()
                        .is_some());
                } else {
                    assert!(
                        fixture.plans[1][0]
                            .prepared_prefix
                            .lock()
                            .unwrap()
                            .is_none(),
                        "L0 consumed its packet before tail authorization"
                    );
                    assert!(
                        fixture.plans[1][1]
                            .prepared_prefix
                            .lock()
                            .unwrap()
                            .is_some(),
                        "tail must remain unsubmitted until its input callback"
                    );
                    fixture.assert_frontend(3.0);
                    assert_eq!(
                        fixture.state()[1],
                        current_expected[1],
                        "tail output cannot change before its callback"
                    );
                }
                fixture.ready(1, stage);
                Ok(())
            },
            None,
        )
        .unwrap();
    assert_eq!(callbacks, [0, 1]);
    for plan in &fixture.plans[1] {
        assert!(plan.prepared_prefix.lock().unwrap().is_none());
    }
    fixture.assert_frontend(4.0);
    assert_eq!(fixture.state(), next_expected, "prepared L0 and freshly uploaded tail input must match ordinary two-stage execution byte for byte");
    assert!(!fixture
        .multi
        .fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .gpu_decode_in_forward());
}

#[test]
#[ignore = "parent serial GPU only; legacy staged callbacks unwind without poisoning the execution lease"]
fn gpu_staged_without_lookahead_unwind_releases_lease_for_retry() {
    let fixture = StagedBundleFixture::new();
    fixture.warm();
    fixture.ordinary(0);
    let expected = fixture.state();
    let bindings = fixture.bindings();
    let stages = fixture.stages(0, &bindings);
    for failed_stage in [0, 1] {
        fixture.reset();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fixture.multi.fixture.backend.execute_staged(
                &stages,
                &mut |stage| {
                    assert_ne!(stage, failed_stage, "synthetic legacy callback unwind");
                    fixture.ready(0, stage);
                    Ok(())
                },
                None,
            )
        }));
        assert!(result.is_err());
        assert!(!fixture
            .multi
            .fixture
            .backend
            .moe_pager()
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .gpu_decode_in_forward());
        fixture.reset();
        fixture.ordinary(0);
        assert_eq!(fixture.state(), expected);
    }
}

#[test]
#[ignore = "parent serial GPU only; callback errors and unwind drain then discard every future stage packet"]
fn gpu_staged_lookahead_callback_failure_and_unwind_discard_all_future_packets() {
    use infr_core::backend::ExecutionLookahead;
    let fixture = StagedBundleFixture::new();
    fixture.warm();
    let bindings = fixture.bindings();
    let current = fixture.stages(0, &bindings);
    let next = fixture.stages(1, &bindings);
    let positions = [4usize, 8];
    for fail_stage in [0, 1] {
        for unwind in [false, true] {
            fixture.clear_packets();
            fixture.reset();
            fixture
                .multi
                .fixture
                .backend
                .execute_staged_lookahead(
                    &current,
                    &mut |stage| {
                        fixture.ready(0, stage);
                        Ok(())
                    },
                    Some(ExecutionLookahead::new(&next, &positions)),
                )
                .unwrap();
            for plan in &fixture.plans[1] {
                assert!(plan.prepared_prefix.lock().unwrap().is_some());
            }
            let before = fixture.state();
            let mut callbacks = Vec::new();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                fixture.multi.fixture.backend.execute_staged_lookahead(
                    &current,
                    &mut |stage| {
                        callbacks.push(stage);
                        if stage == fail_stage {
                            if unwind {
                                panic!("synthetic future-bundle callback unwind");
                            }
                            return Err(be("synthetic future-bundle callback failure"));
                        }
                        fixture.ready(0, stage);
                        Ok(())
                    },
                    Some(ExecutionLookahead::new(&next, &positions)),
                )
            }));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(result
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("synthetic future-bundle callback failure"));
            }
            assert_eq!(callbacks, (0..=fail_stage).collect::<Vec<_>>());
            for plan in &fixture.plans[1] {
                assert!(
                    plan.prepared_prefix.lock().unwrap().is_none(),
                    "first/tail errors and unwind must clear ALL next packets"
                );
            }
            assert!(
                !fixture
                    .multi
                    .fixture
                    .backend
                    .moe_pager()
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .gpu_decode_in_forward(),
                "owner is released only after forward drain"
            );
            let after = fixture.state();
            if fail_stage == 0 {
                assert_eq!(
                    after, before,
                    "first callback failed before any stage execution"
                );
            } else {
                fixture.assert_frontend(3.0);
                assert_eq!(
                    after[1], before[1],
                    "failed tail callback never executes the tail"
                );
                assert_eq!(
                    after.last(),
                    before.last(),
                    "failed tail input was never uploaded"
                );
            }
            assert_eq!(
                fixture.future_kv_rows(&after),
                fixture.future_kv_rows(&before),
                "no recorded future packet may write next-position KV on failure"
            );
            assert_eq!(
                fixture.state(),
                after,
                "state is stable on return; no submitted stage is left running"
            );
            fixture.ordinary(0);
            assert!(!fixture
                .multi
                .fixture
                .backend
                .moe_pager()
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .gpu_decode_in_forward());
        }
    }
}
