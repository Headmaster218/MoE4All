//! Whole-expert span payloads and standalone cross-shard PAGED Q8_0 correctness.
//!
//! The cross-shard fixtures do not install their pool into the backend or construct a model
//! session. A separate session fixture forces gaps with live blockers in an ordinary shard.
//! The central runner owns GPU serialization; these are correctness tests, not PCIe benchmarks.
use super::{
    prefill_bank_payloads, ExpertSource, MoeHostChunkSpec, MoePagerLayout, MoePagerSession,
    MoePoolSpec, PrefillBankPayload, PrefillBankSpan, Role,
};
use crate::transfer::{
    BackgroundTransferExecutor, DeviceTransferTarget, PreparedTransfer, SessionTransferPlan,
};
use crate::unified::{
    ExpertArenaLayout, PlannedRange, UnifiedAllocationHandle, UnifiedRange, UnifiedVramClass,
    UnifiedVramPool, UnitAllocationRequest, UnitClaimPlan,
};
use crate::VulkanBackend;
use infr_core::backend::{Buffer, BufferUsage};
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::error::Result;
use infr_core::hostpager::{InclusiveBlockState, InclusiveHostTier};
use infr_core::{Backend, DType};
use std::sync::Arc;

fn capacity_spans() -> [PrefillBankSpan; 3] {
    [
        PrefillBankSpan {
            allocation: 7,
            first_expert: 0,
            experts: 3,
        },
        PrefillBankSpan {
            allocation: 2,
            first_expert: 3,
            experts: 2,
        },
        PrefillBankSpan {
            allocation: 11,
            first_expert: 5,
            experts: 3,
        },
    ]
}

#[test]
fn payloads_preserve_expert_order_and_allocation_identity() {
    assert_eq!(
        prefill_bank_payloads(&capacity_spans(), 256, 8).unwrap(),
        vec![
            PrefillBankPayload {
                allocation: 7,
                first_expert: 0,
                experts: 3,
                source_offset: 0,
                bytes: 768,
            },
            PrefillBankPayload {
                allocation: 2,
                first_expert: 3,
                experts: 2,
                source_offset: 768,
                bytes: 512,
            },
            PrefillBankPayload {
                allocation: 11,
                first_expert: 5,
                experts: 3,
                source_offset: 1280,
                bytes: 768,
            },
        ],
    );
}

#[test]
fn smaller_stride_and_expert_count_pack_each_span_prefix() {
    let spans = capacity_spans();
    let original = spans;
    assert_eq!(
        prefill_bank_payloads(&spans, 96, 4).unwrap(),
        vec![
            PrefillBankPayload {
                allocation: 7,
                first_expert: 0,
                experts: 3,
                source_offset: 0,
                bytes: 288,
            },
            PrefillBankPayload {
                allocation: 2,
                first_expert: 3,
                experts: 1,
                source_offset: 288,
                bytes: 96,
            },
        ],
    );
    assert_eq!(
        spans, original,
        "a smaller layer must not repack lane capacity"
    );
    let restored = prefill_bank_payloads(&spans, 256, 8).unwrap();
    assert_eq!(restored.len(), 3);
    assert_eq!(restored[2].first_expert, 5);
    assert_eq!(restored[2].experts, 3);
}

#[test]
fn payloads_cover_every_requested_expert_once_at_span_boundaries() {
    let spans = capacity_spans();
    for stride in [1, 96, 256, 17_408] {
        for experts in 1..=8 {
            let payloads = prefill_bank_payloads(&spans, stride, experts).unwrap();
            let mut coverage = vec![0; experts];
            for payload in &payloads {
                let span = spans
                    .iter()
                    .find(|span| span.allocation == payload.allocation)
                    .unwrap();
                assert_eq!(payload.first_expert, span.first_expert);
                assert!(payload.experts > 0 && payload.experts <= span.experts);
                assert_eq!(payload.bytes, payload.experts * stride);
                for local in 0..payload.experts {
                    let expert = payload.first_expert + local;
                    coverage[expert] += 1;
                    assert_eq!(payload.source_offset + local * stride, expert * stride);
                }
            }
            assert_eq!(coverage, vec![1; experts]);
            assert_eq!(
                payloads.iter().map(|payload| payload.bytes).sum::<usize>(),
                experts * stride
            );
        }
    }
}

#[test]
fn payloads_reject_empty_zero_and_insufficient_capacity() {
    let spans = capacity_spans();
    assert!(prefill_bank_payloads(&[], 256, 1).is_err());
    assert!(prefill_bank_payloads(&spans, 0, 8).is_err());
    assert!(prefill_bank_payloads(&spans, 256, 0).is_err());
    assert!(prefill_bank_payloads(&spans, 256, 9).is_err());
}

#[test]
fn payloads_reject_gaps_overlaps_reordering_and_empty_spans() {
    for (index, first_expert, experts) in [(0, 1, 3), (1, 4, 2), (1, 2, 2), (0, 0, 0), (1, 3, 0)] {
        let mut spans = capacity_spans();
        spans[index].first_expert = first_expert;
        spans[index].experts = experts;
        assert!(prefill_bank_payloads(&spans, 256, 8).is_err(), "{spans:?}");
    }
    let mut spans = capacity_spans();
    spans.swap(0, 1);
    assert!(prefill_bank_payloads(&spans, 256, 8).is_err());
}

#[test]
fn payloads_reject_payload_and_later_source_offset_overflow() {
    let stride = usize::MAX / 2 + 1;
    let spans = [PrefillBankSpan {
        allocation: 0,
        first_expert: 0,
        experts: 2,
    }];
    assert!(prefill_bank_payloads(&spans, stride, 2)
        .err()
        .unwrap()
        .to_string()
        .contains("payload size overflow"));

    let spans: Vec<_> = (0..3)
        .map(|first_expert| PrefillBankSpan {
            allocation: first_expert,
            first_expert,
            experts: 1,
        })
        .collect();
    assert!(prefill_bank_payloads(&spans, stride, 3)
        .err()
        .unwrap()
        .to_string()
        .contains("source offset overflow"));
}

const K: usize = 256;
const N: usize = 64;
const EXPERTS: usize = 8;
const BANKS: usize = 3;
const ROWS: usize = 128;
// Expert-grid tiles may load complete activation tiles beyond the final ragged bucket.
const PADDED_ROWS: usize = ROWS + 64;
const EXPERT_BYTES: usize = K * N / 32 * 34;
const MAX_STRIDE: usize = EXPERT_BYTES + 256;
const MAX_SHARD: usize = 64 * 1024;
const GUARD_BYTES: usize = 256;
const GUARD: u8 = 0xa5;
const OUTPUT_GUARD: f32 = -12345.0;
const LUT_PREFIX: usize = 3;

struct PhysicalShard {
    buffer: Arc<dyn Buffer>,
    address: u64,
}

struct SpanUpload {
    bank: usize,
    payload: PrefillBankPayload,
    target: DeviceTransferTarget,
}

fn gpu_backend() -> VulkanBackend {
    gpu_backend_with_host_dma(false)
}

fn gpu_backend_with_host_dma(host_dma: bool) -> VulkanBackend {
    let device = std::env::var("MOE4ALL_TEST_DEVICE")
        .expect("central runner must explicitly select the idle GPU");
    assert!(
        !device.trim().is_empty(),
        "MOE4ALL_TEST_DEVICE must not be empty"
    );
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(device);
    cfg.paging.host_dma = host_dma;
    cfg.paging.dedicated_transfer = true;
    cfg.paging.expert_prefetch = false;
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    let backend = VulkanBackend::new_with(Arc::new(cfg)).expect("configured Vulkan GPU");
    assert!(
        backend.caps().i8_dot,
        "Q8_0 MMQ requires integer dot products"
    );
    backend
}

fn live_ranges(pool: &UnifiedVramPool) -> Vec<UnifiedRange> {
    let mut ranges = pool.allocations();
    ranges.sort_unstable_by_key(|range| range.id);
    ranges
}

// Expert filler is reclaimable in Prefill, so allocator free bytes alone understate the usable
// gaps. Clip the live non-Expert blockers to the actual Prefill corridor in each physical shard.
fn largest_prefill_gap(pool: &UnifiedVramPool) -> usize {
    let live = pool.allocations();
    let mut remaining = pool.prefill_loan_bytes();
    let mut largest = 0;
    for (shard, bytes) in pool.shard_sizes().into_iter().enumerate() {
        let end = bytes.min(remaining);
        remaining = remaining.saturating_sub(bytes);
        let mut blocked: Vec<_> = live
            .iter()
            .filter(|range| range.shard == shard && range.class != UnifiedVramClass::Expert)
            .filter(|range| range.offset < end)
            .map(|range| (range.offset, (range.offset + range.len).min(end)))
            .collect();
        blocked.sort_unstable();
        let mut cursor = 0;
        for (start, stop) in blocked {
            largest = largest.max(start.saturating_sub(cursor));
            cursor = cursor.max(stop);
        }
        largest = largest.max(end.saturating_sub(cursor));
    }
    largest
}

fn assert_failed_plan_is_read_only(pool: &UnifiedVramPool) {
    let before = live_ranges(pool);
    let stats = pool.stats();
    let generation = pool.generation();
    // This request starts consuming gaps in its private plan, then runs out of physical units.
    let impossible = [UnitAllocationRequest {
        unit_bytes: MAX_STRIDE,
        units: stats.capacity_bytes / MAX_STRIDE + 1,
    }];
    assert!(pool
        .plan_unit_owner_claim(&impossible, UnifiedVramClass::Prefill, &[])
        .is_err());
    assert_eq!(live_ranges(pool), before);
    assert_eq!(pool.stats(), stats);
    assert_eq!(pool.generation(), generation);
}

fn activation(row: usize, column: usize) -> i8 {
    ((row * 7 + column * 11 + column / 32 * 3) % 17) as i8 - 8
}

fn weight(salt: usize, bank: usize, expert: usize, output: usize, column: usize) -> i8 {
    ((salt * 3 + bank * 11 + expert * 7 + output * 5 + column * 3 + column / 32) % 19) as i8 - 9
}

fn q8_bank(stride: usize, salt: usize, bank: usize) -> Vec<u8> {
    assert!(stride >= EXPERT_BYTES);
    let mut bytes = vec![0x5d; EXPERTS * stride];
    for expert in 0..EXPERTS {
        for output in 0..N {
            for block in 0..K / 32 {
                let offset = expert * stride + (output * (K / 32) + block) * 34;
                bytes[offset..offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16(1)
                for lane in 0..32 {
                    bytes[offset + 2 + lane] =
                        weight(salt, bank, expert, output, block * 32 + lane) as u8;
                }
            }
        }
    }
    bytes
}

fn upload_buffer(backend: &VulkanBackend, bytes: &[u8]) -> Box<dyn Buffer> {
    let buffer = backend
        .alloc_uninit(bytes.len(), BufferUsage::Activations)
        .expect("fixture buffer");
    backend
        .upload(buffer.as_ref(), bytes)
        .expect("fixture upload");
    buffer
}

fn check_integer_mmq(
    backend: &VulkanBackend,
    addresses: &[u64],
    representative: u64,
    stride: usize,
    bank: usize,
    salt: usize,
) {
    assert_eq!(addresses.len(), EXPERTS);
    assert!(addresses.iter().all(|address| *address != 0));
    // Each two-expert span has a routed expert; also test empty buckets and a 64-row bucket.
    let counts = [0u32, 5, 0, 17, 1, 0, 64, 41];
    let mut offsets = [0u32; EXPERTS];
    let mut total = 0;
    for expert in 0..EXPERTS {
        offsets[expert] = total;
        total += counts[expert];
    }
    assert_eq!(total as usize, ROWS);
    let mut qa = vec![0u32; PADDED_ROWS * K / 4];
    for row in 0..ROWS {
        for column in 0..K {
            qa[row * (K / 4) + column / 4] |=
                (activation(row, column) as u8 as u32) << ((column % 4) * 8);
        }
    }
    let dact = vec![0x3c00u16; PADDED_ROWS * K / 32];
    let mut lut = vec![addresses[0]; LUT_PREFIX];
    lut.extend_from_slice(addresses);
    let qa = upload_buffer(backend, bytemuck::cast_slice(&qa));
    let dact = upload_buffer(backend, bytemuck::cast_slice(&dact));
    let lut = upload_buffer(backend, bytemuck::cast_slice(&lut));
    let counts_buffer = upload_buffer(backend, bytemuck::cast_slice(&counts));
    let offsets_buffer = upload_buffer(backend, bytemuck::cast_slice(&offsets));
    let mut output = vec![OUTPUT_GUARD; PADDED_ROWS * N];
    let output_buffer = upload_buffer(backend, bytemuck::cast_slice(&output));

    let recorder = backend.recorder().expect("MMQ recorder");
    recorder.arena_stream_barrier();
    recorder.matmul_mmq_experts_paged(
        DType::Q8_0,
        "expert_gateup",
        qa.as_ref(),
        dact.as_ref(),
        None,
        representative, // Compatibility only: every expert is resolved through the absolute LUT.
        stride as u32,
        lut.as_ref(),
        LUT_PREFIX,
        counts_buffer.as_ref(),
        offsets_buffer.as_ref(),
        output_buffer.as_ref(),
        ROWS,
        K,
        N,
        EXPERTS,
        1,
    );
    recorder.finish().expect("finish PAGED Q8_0 MMQ");
    backend
        .download(
            output_buffer.as_ref(),
            bytemuck::cast_slice_mut(&mut output),
        )
        .expect("MMQ readback");
    for expert in 0..EXPERTS {
        let first = offsets[expert] as usize;
        for row in first..first + counts[expert] as usize {
            for column in 0..N {
                let dot: i32 = (0..K)
                    .map(|k| {
                        i32::from(activation(row, k))
                            * i32::from(weight(salt, bank, expert, column, k))
                    })
                    .sum();
                // Small integer operands and unit scales make every partial sum exact in f32.
                assert_eq!(
                    output[row * N + column],
                    dot as f32,
                    "bank={bank} salt={salt} expert={expert} row={row} column={column}",
                );
            }
        }
    }
    assert!(output[ROWS * N..]
        .iter()
        .all(|value| *value == OUTPUT_GUARD));
}

fn exercise_fragmented_banks(force_device_local_targets: bool) {
    let backend = gpu_backend();
    assert_eq!(EXPERT_BYTES, 17_408);
    let layout = ExpertArenaLayout::build(&[(EXPERT_BYTES, 48, 3)], MAX_SHARD, 0, 0, 0, 0)
        .expect("small expert-aware physical geometry");
    let pool = UnifiedVramPool::new_for_experts(&backend, layout).expect("standalone expert pool");
    assert!(
        backend.unified_vram().is_none(),
        "fixture must not install its pool"
    );
    let shard_sizes = pool.shard_sizes();
    assert!(shard_sizes.len() > 1);
    assert!(shard_sizes.iter().all(|size| *size <= MAX_SHARD));
    assert!(shard_sizes
        .iter()
        .all(|size| *size < EXPERTS * EXPERT_BYTES));

    // One live guard at every shard's beginning forces all payloads to use nonzero offsets.
    // There are no live Expert filler leases, so direct claim commit needs no pager retirement.
    let blockers: Vec<_> = (0..shard_sizes.len())
        .map(|shard| {
            pool.try_claim_exact(shard, 0, GUARD_BYTES, UnifiedVramClass::LlmRuntime)
                .expect("claim a live shard guard")
        })
        .collect();
    let backing: Vec<_> = blockers
        .iter()
        .enumerate()
        .map(|(shard, blocker)| {
            if force_device_local_targets {
                // A transport twin with exactly the claimed geometry. The unified pool still
                // uses its preferred backing: forcing that pool itself requires an arena helper.
                let (buffer, address) = backend
                    .alloc_device_local_arena_bda(shard_sizes[shard])
                    .expect("forced ordinary device-local shard");
                assert!(crate::as_vk_buf(buffer.as_ref())
                    .unwrap()
                    .mapped_ptr()
                    .is_none());
                PhysicalShard {
                    buffer: Arc::from(buffer),
                    address,
                }
            } else {
                PhysicalShard {
                    buffer: blocker.buffer_arc(),
                    address: blocker.base_addr(),
                }
            }
        })
        .collect();
    let guard_bytes = vec![GUARD; GUARD_BYTES];
    for shard in &backing {
        backend
            .upload_range(shard.buffer.as_ref(), 0, &guard_bytes)
            .unwrap();
    }
    assert_failed_plan_is_read_only(&pool);
    let before = live_ranges(&pool);
    let stats = pool.stats();
    let largest_gap_before = largest_prefill_gap(&pool);
    let generation = pool.generation();
    assert!(pool
        .plan_owner_claim(&[EXPERTS * MAX_STRIDE], UnifiedVramClass::Prefill, &[])
        .is_err());
    let requests = [UnitAllocationRequest {
        unit_bytes: MAX_STRIDE,
        units: EXPERTS,
    }; BANKS];
    let UnitClaimPlan { claim, spans } = pool
        .plan_unit_owner_claim(&requests, UnifiedVramClass::Prefill, &[])
        .expect("fragmented capacity must fit the complete three-bank layer");
    assert!(claim.victims().is_empty());
    assert_eq!(
        live_ranges(&pool),
        before,
        "successful planning is read-only too"
    );
    assert_eq!(pool.stats(), stats);
    assert_eq!(pool.generation(), generation);
    let handles = pool
        .commit_claim(claim)
        .expect("commit the whole layer transaction");
    let ring_ranges: Vec<_> = handles.iter().map(|handle| handle.range()).collect();
    let ring_bytes: usize = ring_ranges.iter().map(|range| range.len).sum();
    let mut banks: Vec<Vec<PrefillBankSpan>> = (0..BANKS).map(|_| Vec::new()).collect();
    for span in spans {
        let range = handles[span.allocation].range();
        assert!(range.offset >= GUARD_BYTES);
        assert!(range.offset + range.len <= shard_sizes[range.shard]);
        assert!(span.units * MAX_STRIDE <= range.len);
        banks[span.request].push(PrefillBankSpan {
            allocation: span.allocation,
            first_expert: span.first_unit,
            experts: span.units,
        });
    }
    for bank in &banks {
        assert!(
            bank.len() > 1,
            "a complete bank cannot fit any physical shard"
        );
        let shards: std::collections::HashSet<_> = bank
            .iter()
            .map(|span| handles[span.allocation].range().shard)
            .collect();
        assert!(
            shards.len() > 1,
            "exercise actual cross-shard expert addresses"
        );
    }
    let spans_per_bank: Vec<_> = banks.iter().map(Vec::len).collect();
    let ring_requested_bytes: usize = ring_ranges.iter().map(|range| range.requested_len).sum();
    assert_failed_plan_is_read_only(&pool);

    let mut expected_ranges: Vec<Vec<u8>> = ring_ranges
        .iter()
        .map(|range| vec![GUARD; range.len])
        .collect();
    for (range, expected) in ring_ranges.iter().zip(&expected_ranges) {
        backend
            .upload_range(backing[range.shard].buffer.as_ref(), range.offset, expected)
            .unwrap();
    }
    // A -> B -> A: smaller B packs each span's prefix and must preserve its unused capacity tail.
    for (stride, salt) in [(MAX_STRIDE, 0), (EXPERT_BYTES, 13), (MAX_STRIDE, 0)] {
        let sources: Vec<_> = (0..BANKS).map(|bank| q8_bank(stride, salt, bank)).collect();
        let mut uploads = Vec::new();
        let mut addresses = vec![vec![0u64; EXPERTS]; BANKS];
        for (bank, bank_spans) in banks.iter().enumerate() {
            for payload in prefill_bank_payloads(bank_spans, stride, EXPERTS).unwrap() {
                let handle = &handles[payload.allocation];
                let range = handle.range();
                let physical = &backing[range.shard];
                let buffer = if force_device_local_targets {
                    Arc::clone(&physical.buffer)
                } else {
                    handle.buffer_arc()
                };
                let target = DeviceTransferTarget::new(buffer, range.offset, payload.bytes)
                    .expect("exact whole-expert span target");
                assert!(payload.bytes <= range.len);
                let source =
                    &sources[bank][payload.source_offset..payload.source_offset + payload.bytes];
                expected_ranges[payload.allocation][..payload.bytes].copy_from_slice(source);
                for local in 0..payload.experts {
                    addresses[bank][payload.first_expert + local] =
                        physical.address + range.offset as u64 + (local * stride) as u64;
                }
                uploads.push(SpanUpload {
                    bank,
                    payload,
                    target,
                });
            }
        }
        let plan = SessionTransferPlan::default();
        let mut prepared = PreparedTransfer::default();
        if force_device_local_targets {
            // Also exercise the filled-staging API: one prepared batch, no per-span submit.
            for upload in &uploads {
                assert!(upload.target.mapped_ptr().is_none());
                let start = upload.payload.source_offset;
                let source = &sources[upload.bank][start..start + upload.payload.bytes];
                plan.prepare_fill(
                    &backend,
                    &upload.target,
                    |destination| {
                        destination.copy_from_slice(source);
                        Ok(())
                    },
                    &mut prepared,
                )
                .expect("prepare device-local filled span");
            }
        } else {
            plan.prepare_upload_batch(
                &backend,
                uploads.iter().map(|upload| {
                    let start = upload.payload.source_offset;
                    (
                        &sources[upload.bank][start..start + upload.payload.bytes],
                        &upload.target,
                    )
                }),
                &mut prepared,
            )
            .expect("prepare all span uploads together");
        }
        let recorder = backend.recorder().expect("one layer transfer recorder");
        prepared
            .record(&recorder)
            .expect("record the complete layer batch");
        recorder.arena_stream_barrier();
        recorder
            .finish()
            .expect("complete every span before publishing the LUT");
        for (range, expected) in ring_ranges.iter().zip(&expected_ranges) {
            let mut actual = vec![0; range.len];
            backend
                .download_range(
                    backing[range.shard].buffer.as_ref(),
                    range.offset,
                    &mut actual,
                )
                .unwrap();
            assert_eq!(
                &actual, expected,
                "payload/tail mismatch at {range:?}, stride={stride}"
            );
        }
        for (bank, bank_addresses) in addresses.iter().enumerate() {
            let first = handles[banks[bank][0].allocation].range();
            let representative = backing[first.shard].address + first.offset as u64;
            check_integer_mmq(&backend, bank_addresses, representative, stride, bank, salt);
        }
        for shard in &backing {
            let mut actual = vec![0; GUARD_BYTES];
            backend
                .download_range(shard.buffer.as_ref(), 0, &mut actual)
                .unwrap();
            assert_eq!(actual, guard_bytes, "live blocker was overwritten");
        }
    }

    // A transfer target retains a physical buffer, while a handle retains the owner transaction.
    // One remaining handle must protect ALL sibling ranges, even after GPU work has drained.
    let retained = Arc::clone(&handles[0]);
    drop(handles);
    assert_eq!(
        pool.stats().class_bytes(UnifiedVramClass::Prefill),
        ring_bytes
    );
    for range in &ring_ranges {
        assert!(pool
            .try_claim_exact(
                range.shard,
                range.offset,
                range.len,
                UnifiedVramClass::Prefill
            )
            .is_none());
    }
    drop(retained);
    assert_eq!(pool.stats().class_bytes(UnifiedVramClass::Prefill), 0);
    assert_eq!(
        live_ranges(&pool),
        before,
        "only the original live blockers remain"
    );
    assert_eq!(pool.stats(), stats);
    for range in &ring_ranges {
        let lease: Arc<UnifiedAllocationHandle> = pool
            .try_claim_exact(
                range.shard,
                range.offset,
                range.len,
                UnifiedVramClass::Prefill,
            )
            .expect("released span must be reclaimable at its exact coordinates");
        drop(lease);
    }
    drop(blockers);
    assert!(pool.allocations().is_empty());
    assert_eq!(pool.stats().free_bytes, pool.stats().capacity_bytes);
    eprintln!(
        "PREFILL_FRAGMENTED_CAPACITY case={} largest_gap_before={} allocator_largest_free_before={} logical_bank_bytes={} ring_requested_bytes={} ring_physical_bytes={} spans_per_bank={:?} physical_shards={} payload_exact=true integer_mmq_exact=true",
        if force_device_local_targets { "standalone_device_local_targets" } else { "standalone_preferred_pool" },
        largest_gap_before, stats.largest_free_bytes, EXPERTS * MAX_STRIDE,
        ring_requested_bytes, ring_bytes, spans_per_bank, shard_sizes.len(),
    );
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE"]
fn gpu_fragmented_q8_banks_preferred_pool_roundtrip_mmq_and_leases() {
    exercise_fragmented_banks(false);
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE"]
fn gpu_fragmented_q8_banks_forced_device_local_targets_and_batched_fill() {
    exercise_fragmented_banks(true);
}

struct PatternIo;

impl PatternIo {
    fn byte(offset: usize) -> u8 {
        (offset.wrapping_mul(29)
            ^ (offset / 257).wrapping_mul(71)
            ^ (offset / 65_536).wrapping_mul(17)) as u8
    }

    fn expected(offset: usize, bytes: usize) -> Vec<u8> {
        (offset..offset + bytes).map(Self::byte).collect()
    }
}

impl BlockIo for PatternIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        if desc.nbytes() > dst.len() {
            return Err(crate::be("fragmented session fixture destination is short"));
        }
        let mut written = 0;
        for extent in &desc.extents {
            let start = usize::try_from(extent.offset)
                .map_err(|_| crate::be("fragmented session fixture extent exceeds usize"))?;
            for local in 0..extent.len {
                dst[written + local] = Self::byte(start + local);
            }
            written += extent.len;
        }
        Ok(())
    }
}

fn exercise_fragmented_session(host_dma: bool, tiered: bool) {
    const LARGE: usize = 64 * 1024;
    const SMALL: usize = 32 * 1024;
    const SESSION_EXPERTS: usize = 4;
    const HOST_BYTES: usize = 3 * SESSION_EXPERTS * (LARGE + SMALL);
    assert!(
        !(host_dma && tiered),
        "DMA proof uses the permanent host store"
    );
    let backend = gpu_backend_with_host_dma(host_dma);
    let host_tier = tiered.then(|| {
        Arc::new(
            InclusiveHostTier::new(
                2 * (LARGE + SMALL),
                &[(LARGE, 3 * SESSION_EXPERTS), (SMALL, 3 * SESSION_EXPERTS)],
                Arc::new(PatternIo),
            )
            .expect("two bounded RAM size classes with two slots each"),
        )
    });
    let mut session = MoePagerSession::new(
        &backend,
        MoePagerLayout {
            load_reserve_bytes: 0,
            n_blocks: 8,
            pools: vec![
                MoePoolSpec {
                    slot_bytes: LARGE,
                    n_slots: 96,
                    min_enabled_slots: 1,
                },
                MoePoolSpec {
                    slot_bytes: SMALL,
                    n_slots: 64,
                    min_enabled_slots: 1,
                },
            ],
            host_tier: host_tier.clone(),
            host_store_io: if tiered {
                None
            } else {
                Some(Arc::new(PatternIo))
            },
            dynamic_state_reserve_bytes: 0,
            dynamic_state_max_allocation_bytes: 0,
            prefill_min_lane_bytes: 0,
            runtime_reserve_bytes: 2 * 1024 * 1024,
            host_chunks: if tiered {
                Vec::new()
            } else {
                vec![MoeHostChunkSpec {
                    base_offset: 0,
                    bytes: HOST_BYTES,
                }]
            },
            prefill_target_lanes: 1,
            prefill_cache_bytes: (96 * LARGE + 64 * SMALL) as u64,
        },
    )
    .expect("actual pager session with two expert size classes");
    let ids = [[101usize, 102, 103], [201, 202, 203]];
    let mut host_offsets = [[0usize; 3]; 2];
    let mut host_offset = 0;
    for (layer, stride) in [LARGE, SMALL].into_iter().enumerate() {
        for (bank, role) in [Role::Gate, Role::Up, Role::Down].into_iter().enumerate() {
            let bytes = SESSION_EXPERTS * stride;
            host_offsets[layer][bank] = host_offset;
            if let Some(tier) = &host_tier {
                let cache = tier.cache(stride).expect("original source size class");
                for expert in 0..SESSION_EXPERTS {
                    cache
                        .register(BlockDesc {
                            id: (bank * 8 + layer * SESSION_EXPERTS + expert) as u32,
                            extents: vec![BlockExtent {
                                offset: (host_offset + expert * stride) as u64,
                                len: stride,
                            }],
                        })
                        .expect("register source-role expert descriptor before the bank");
                }
            }
            session
                .register(
                    role,
                    ids[layer][bank],
                    ExpertSource {
                        bank: Arc::new(vec![0u8; bytes]),
                        stride_bytes: stride,
                        layer_base: (layer * SESSION_EXPERTS) as u32,
                        host_offset,
                        file: Some(BlockDesc {
                            id: (layer * 3 + bank) as u32,
                            extents: vec![BlockExtent {
                                offset: host_offset as u64,
                                len: bytes,
                            }],
                        }),
                    },
                    SESSION_EXPERTS,
                )
                .expect("register layer-major source bank");
            assert_eq!(
                session.sources[&ids[layer][bank]].2.block_base,
                (bank * 8 + layer * SESSION_EXPERTS) as u32,
            );
            host_offset += bytes;
        }
    }
    assert_eq!(host_offset, HOST_BYTES);
    assert_eq!(session.host_store.is_empty(), tiered);
    if let Some(tier) = &host_tier {
        for (layer, stride) in [LARGE, SMALL].into_iter().enumerate() {
            let cache = tier.cache(stride).unwrap();
            assert_eq!(cache.n_slots(), 2);
            // Both hits are after expert zero, including one in Down's distinct role block space.
            let resident = [
                (layer * SESSION_EXPERTS + 3) as u32,
                (2 * 8 + layer * SESSION_EXPERTS + 1) as u32,
            ];
            assert_eq!(cache.preload(&resident).unwrap(), (2, 2 * stride));
            for id in resident {
                assert_eq!(cache.block_state(id).unwrap(), InclusiveBlockState::Ready);
            }
        }
    }
    let background = if host_dma {
        let plan = backend.build_session_transfer_plan(session.take_transfer_sources());
        let source = &session.sources[&ids[0][0]].2;
        let bank = session.host_store[source.host_chunk.unwrap()]
            .range(source.host_offset, source.bank_bytes)
            .unwrap();
        assert!(
            plan.contains_imported_upload(bank),
            "fixture requires an actual imported expert bank"
        );
        session.install_transfer_plan(Arc::new(plan));
        Some(
            BackgroundTransferExecutor::from_backend(&backend)
                .expect("DMA fixture requires a dedicated transfer queue and host staging heap"),
        )
    } else {
        None
    };
    let pool = Arc::clone(&session.unified_pool);
    assert_eq!(
        pool.shard_sizes().len(),
        1,
        "use ordinary small-pool shard geometry"
    );
    let floor_end = pool.expert_layout().unwrap().floor_corridor().end;
    assert_eq!(floor_end, 2 * LARGE + 2 * SMALL);
    let loan_end = pool.prefill_loan_bytes();
    let blocker_ranges: Vec<_> = (floor_end..loan_end)
        .step_by(2 * LARGE)
        .map(|offset| PlannedRange {
            shard: 0,
            offset,
            len: LARGE.min(loan_end - offset),
            requested_len: LARGE.min(loan_end - offset),
        })
        .collect();
    let blocker_plan = pool
        .plan_exact_owner_claim(&blocker_ranges, UnifiedVramClass::LlmRuntime, &[])
        .expect("plan alternating persistent runtime blockers");
    // Existing Expert filler must retire through the session, not a raw exact allocator claim.
    let blockers = session
        .commit_unified_claim(&backend, blocker_plan)
        .expect("retire filler for exact live blockers");
    for blocker in &blockers {
        let range = blocker.range();
        backend
            .upload_range(blocker.buffer(), range.offset, &vec![GUARD; range.len])
            .unwrap();
    }
    let expert_bytes_before = pool.stats().class_bytes(UnifiedVramClass::Expert);
    let largest_gap_before = largest_prefill_gap(&pool);
    let allocator_largest_free_before = pool.stats().largest_free_bytes;
    assert_eq!(largest_gap_before, floor_end);
    assert!(
        pool.plan_owner_claim(
            &[SESSION_EXPERTS * LARGE; 3],
            UnifiedVramClass::Prefill,
            &[]
        )
        .is_err(),
        "the floor gap and all alternating gaps are smaller than a bank"
    );

    // Reject one prepared source/target size mismatch before any copy is submitted, then use the
    // same explicit abort path as a drained failed producer. Retrying must not remain pending.
    let mut failed = session
        .prepare_prefill_layer_cpu(&backend, ids[0][0])
        .unwrap()
        .expect("first layer copy job");
    assert!(session.layer_bank_pending(ids[0][0]).unwrap());
    assert!(!session.layer_bank_current(ids[0][0]).unwrap());
    assert!(
        session
            .prepare_prefill_layer_cpu(&backend, ids[0][1])
            .unwrap()
            .is_none(),
        "another role may not duplicate the same pending layer"
    );
    match failed.copies.first_mut().expect("whole-layer copies") {
        super::PrefillCopy::Memory { len, .. } => *len -= 1,
        super::PrefillCopy::Tiered { n_blocks, .. } => *n_blocks -= 1,
    }
    assert!(failed.execute(&backend).is_err());
    session.abort_prefill_layer_cpu(ids[0][0]);
    for id in ids[0] {
        assert!(!session.layer_bank_pending(id).unwrap());
        assert!(!session.layer_bank_current(id).unwrap());
    }

    let ring_requested_bytes: usize = session
        .prefill_allocations
        .iter()
        .map(|allocation| allocation.range().requested_len)
        .sum();
    let ring_physical_bytes: usize = session
        .prefill_allocations
        .iter()
        .map(|allocation| allocation.range().len)
        .sum();
    let spans_per_bank: Vec<_> = ids[0]
        .iter()
        .map(|id| session.prefill_placement[id].spans.len())
        .collect();
    let mut dma_values = Vec::new();
    let mut tier_reads = Vec::new();
    let mut cursor = 5usize; // Exercise actual tape windows with a nonzero base.
    let mut frozen_windows = Vec::new();
    for layer in [0usize, 1, 0] {
        let stride = [LARGE, SMALL][layer];
        let job = session
            .prepare_prefill_layer_cpu(&backend, ids[layer][0])
            .unwrap()
            .expect("retry or replace the one-lane layer");
        assert_eq!(job.buf_id(), ids[layer][0]);
        for id in ids[layer] {
            assert!(session.layer_bank_pending(id).unwrap());
            assert!(!session.layer_bank_current(id).unwrap());
        }
        let groups: Vec<_> = job
            .copies
            .chunk_by(|left, right| left.bank() == right.bank())
            .map(|copies| (copies[0].bank(), copies.len()))
            .collect();
        assert_eq!(
            groups.iter().map(|group| group.0).collect::<Vec<_>>(),
            ids[layer]
        );
        assert!(
            groups.iter().all(|group| group.1 > 1),
            "every bank has multiple spans"
        );
        let tier_cache = host_tier.as_ref().map(|tier| tier.cache(stride).unwrap());
        let tier_before = tier_cache.as_ref().map(|cache| cache.stats());
        if let Some(executor) = &background {
            assert!(
                job.copies.iter().any(|copy| match copy {
                    super::PrefillCopy::Memory { src, len, .. } => {
                        let bytes = unsafe { std::slice::from_raw_parts(*src as *const u8, *len) };
                        session.transfer_plan.contains_imported_upload(bytes)
                    }
                    _ => false,
                }),
                "every DMA layer must contain an imported source span"
            );
            let value = job
                .execute_on_background(executor)
                .expect("execute actual imported-host DMA on the background queue")
                .expect("imported-host DMA must produce a timeline value");
            assert!(value > 0);
            if let Some(previous) = dma_values.last() {
                assert!(value > *previous);
            }
            dma_values.push(value);
        } else {
            job.execute(&backend)
                .expect("execute all real session span copies");
        }
        if let (Some(cache), Some(before)) = (&tier_cache, tier_before) {
            let after = cache.stats();
            assert_eq!(after.ram_hits - before.ram_hits, 2);
            assert_eq!(after.ssd_reads - before.ssd_reads, 10);
            tier_reads.push((
                stride,
                after.ram_hits - before.ram_hits,
                after.ssd_reads - before.ssd_reads,
            ));
            assert_eq!(after.bytes_read - before.bytes_read, (10 * stride) as u64);
            assert_eq!(
                after.bytes_promoted - before.bytes_promoted,
                (12 * stride) as u64
            );
            assert_eq!(
                after.ram_evictions, before.ram_evictions,
                "Prefill must not admit its misses"
            );
            for bank in 0..3 {
                for expert in 0..SESSION_EXPERTS {
                    let id = (bank * 8 + layer * SESSION_EXPERTS + expert) as u32;
                    let resident = (bank == 0 && expert == 3) || (bank == 2 && expert == 1);
                    assert_eq!(
                        cache.block_state(id).unwrap(),
                        if resident {
                            InclusiveBlockState::Ready
                        } else {
                            InclusiveBlockState::Absent
                        }
                    );
                }
            }
        }
        assert!(!session.layer_bank_current(ids[layer][0]).unwrap());
        session.complete_prefill_layer_cpu(ids[layer][0]).unwrap();
        for (bank, id) in ids[layer].into_iter().enumerate() {
            assert!(session.layer_bank_current(id).unwrap());
            assert!(!session.layer_bank_pending(id).unwrap());
            let spans = Arc::clone(&session.prefill_placement[&id].spans);
            assert!(spans.len() > 1, "the real session bank must be fragmented");
            assert!(session.layer_byte_offset(id).is_err());
            let first = &session.prefill_allocations[spans[0].allocation];
            let representative = first.base_addr() + first.range().offset as u64;
            // Compatibility helpers must not reject a whole bank larger than the first span.
            assert_eq!(session.arena_addr(id).unwrap(), representative);
            assert_eq!(
                crate::as_vk_buf(session.arena(id).unwrap()).unwrap().buffer,
                crate::as_vk_buf(first.buffer()).unwrap().buffer,
            );
            let mut addresses = vec![0u64; SESSION_EXPERTS];
            for payload in prefill_bank_payloads(&spans, stride, SESSION_EXPERTS).unwrap() {
                let allocation = &session.prefill_allocations[payload.allocation];
                let range = allocation.range();
                let mut actual = vec![0; payload.bytes];
                backend
                    .download_range(allocation.buffer(), range.offset, &mut actual)
                    .unwrap();
                assert_eq!(
                    actual,
                    PatternIo::expected(
                        host_offsets[layer][bank] + payload.source_offset,
                        payload.bytes
                    ),
                    "real layer={layer} role={bank} first_expert={}",
                    payload.first_expert,
                );
                for local in 0..payload.experts {
                    addresses[payload.first_expert + local] =
                        allocation.base_addr() + range.offset as u64 + (local * stride) as u64;
                }
            }
            let window = session
                .layer_lut_window(&mut cursor, id, SESSION_EXPERTS)
                .unwrap();
            let tape = crate::as_vk_buf(session.tape())
                .unwrap()
                .mapped_ptr()
                .unwrap();
            let words = unsafe {
                std::slice::from_raw_parts(
                    tape.add(window as usize * 8).cast::<u64>(),
                    SESSION_EXPERTS,
                )
            };
            assert_eq!(words, addresses.as_slice());
            frozen_windows.push((window, addresses));
        }
        for blocker in &blockers {
            let range = blocker.range();
            let mut actual = vec![0; range.len];
            backend
                .download_range(blocker.buffer(), range.offset, &mut actual)
                .unwrap();
            assert!(
                actual.iter().all(|byte| *byte == GUARD),
                "live runtime blocker changed"
            );
        }
    }
    // Later layer windows must never rewrite earlier addresses in the immutable tape.
    let tape = crate::as_vk_buf(session.tape())
        .unwrap()
        .mapped_ptr()
        .unwrap();
    for (window, addresses) in frozen_windows {
        let words = unsafe {
            std::slice::from_raw_parts(tape.add(window as usize * 8).cast::<u64>(), SESSION_EXPERTS)
        };
        assert_eq!(words, addresses.as_slice());
    }

    let detached = session
        .prepare_prefill_layer_cpu(&backend, ids[1][0])
        .unwrap()
        .unwrap();
    let ring_bytes = pool.stats().class_bytes(UnifiedVramClass::Prefill);
    assert!(ring_bytes > 0);
    assert!(session.enter_decode());
    assert_eq!(
        pool.stats().class_bytes(UnifiedVramClass::Prefill),
        ring_bytes,
        "the detached job must retain the released session ring leases"
    );
    drop(detached);
    session.enter_decode();
    assert_eq!(pool.stats().class_bytes(UnifiedVramClass::Prefill), 0);
    assert_eq!(
        pool.stats().class_bytes(UnifiedVramClass::Expert),
        expert_bytes_before
    );
    drop(blockers);
    session.enter_decode();
    assert_eq!(
        pool.stats().class_bytes(UnifiedVramClass::Expert),
        pool.stats().capacity_bytes
    );
    eprintln!(
        "PREFILL_FRAGMENTED_CAPACITY case=session host_dma={} tiered={} largest_gap_before={} allocator_largest_free_before={} logical_bank_bytes={:?} ring_requested_bytes={} ring_physical_bytes={} spans_per_bank={:?} physical_shards={} dma_values={:?} tier_reads_stride_ram_ssd={:?} payload_lut_exact=true",
        host_dma, tiered, largest_gap_before, allocator_largest_free_before,
        [SESSION_EXPERTS * LARGE, SESSION_EXPERTS * SMALL], ring_requested_bytes,
        ring_physical_bytes, spans_per_bank, pool.shard_sizes().len(), dma_values, tier_reads,
    );
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE"]
fn gpu_fragmented_session_jobs_lut_stride_reuse_abort_retry_and_decode_restore() {
    exercise_fragmented_session(false, false);
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE, host import and dedicated transfer queue"]
fn gpu_fragmented_session_imported_host_dma_background() {
    exercise_fragmented_session(true, false);
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE"]
fn gpu_fragmented_session_bounded_ram_ssd_split_block_offsets() {
    exercise_fragmented_session(false, true);
}
