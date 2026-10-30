#![cfg(test)]

//! Split/fused owner preparation differential probes; TopK routing stays separate.
use super::*;

#[test]
fn gpu_owned_wider_prepare_admission_covers_qualified_shapes() {
    let mut push = [0; 19];
    for used in 1..=16 {
        push[6] = used;
        for rows in [1, 2, 4, 8] {
            push[0] = rows * used;
            assert_eq!(owned_prepare_fused_eligible(true, &push, false), rows == 1);
            assert_eq!(
                owned_prepare_fused_eligible(true, &push, true),
                rows == 1 || (rows == 2 && matches!(used, 8 | 10))
            );
            assert!(!owned_prepare_fused_eligible(false, &push, true));
        }
    }
    for (count, used) in [(0, 8), (17, 8), (19, 10), (21, 10), (u32::MAX, 10)] {
        push[0] = count;
        push[6] = used;
        assert!(!owned_prepare_fused_eligible(true, &push, true));
    }
}

#[test]
fn gpu_owned_recovery_prepare_push_uses_pool_start0_word12() {
    for recovery in [0, 1, 2] {
        let base = gpu_copy_prepare_push(
            [40, 512, 0, 512, 1024, 0, 10, 1 << 10],
            true,
            [100, 612, 1124, recovery, 0, 0, 0, 0, 0],
            8,
        );
        assert_eq!(base[12], recovery, "extra[3] is pool_start0, not base[3]");
        assert_eq!(base[3], 512, "Up Tape base remains independent");
        assert_eq!(gpu_owned_fast_push(&base, &[0; 12])[12], recovery);
        assert_eq!(
            owned_prepare_fused_eligible(true, &base, true),
            recovery == 1
        );
    }
}

#[test]
fn gpu_owned_recovery_fused_prepare_gate_only_qualified_shapes() {
    let mut base = [0u32; 19];
    for used in 1..=16 {
        base[6] = used;
        for rows in 1..=9 {
            base[0] = rows * used;
            for recovery in [0, 1, 2, u32::MAX] {
                base[12] = recovery;
                for wider in [false, true] {
                    let existing = rows == 1 || (wider && rows == 2 && matches!(used, 8 | 10));
                    let added = wider
                        && recovery == 1
                        && (((3..=8).contains(&rows) && matches!(used, 8 | 10))
                            || (rows == 8 && used == 16));
                    assert_eq!(
                        owned_prepare_fused_eligible(true, &base, wider),
                        existing || added,
                        "rows={rows} used={used} recovery={recovery} wider={wider}"
                    );
                    assert!(!owned_prepare_fused_eligible(false, &base, wider));
                }
            }
        }
    }
    base[12] = 1;
    for used in [0, 8, 10, 16] {
        base[6] = used;
        for count in [17, 19, 21, 129, u32::MAX] {
            base[0] = count;
            assert!(!owned_prepare_fused_eligible(true, &base, true));
        }
    }
}

struct WiderProbe {
    pool: PoolProbe,
    overlap: bool,
    miss: Box<dyn Buffer>,
}

impl WiderProbe {
    fn new(
        be: &VulkanBackend,
        slots: usize,
        duplicate: bool,
        workload: Workload,
        overlap: bool,
    ) -> Self {
        let used = probe_used();
        assert!(matches!(used, 8 | 10), "run separate used8/10 subprocesses");
        let mut pool = PoolProbe::new(be, slots, used * 2, workload);
        assert_eq!(
            pool.repeats, 1,
            "no warm-hit or external shader probe switches"
        );
        if duplicate {
            let (first, second) = pool.initial[0][..used * 2].split_at_mut(used);
            second.copy_from_slice(first);
        }
        let roles = [pool.push[9], pool.push[10], pool.push[11]].map(|x| x as usize);
        (pool.expected, pool.misses) = reference(
            &pool.initial,
            &pool.initial[0][..used * 2],
            roles,
            slots,
            pool.address_base,
        );
        if overlap {
            for row in 0..2 {
                let hits = pool.expected[3][RESIDENT_JOURNAL_MASK_WORD + row];
                pool.expected[0][used * 2 + row] = hits;
                pool.expected[0][used * 2 + 2 + row] = ((1 << used) - 1) ^ hits;
            }
        }
        pool.buffers[5] = be
            .alloc_mapped_arena_bda((pool.heap_base + 2 + slots * 4) * 16)
            .unwrap()
            .0;
        let miss = be.alloc_mapped_arena_bda(16).unwrap().0;
        Self {
            pool,
            overlap,
            miss,
        }
    }

    fn reset(&self, be: &VulkanBackend) {
        self.pool.reset(be, false, true);
        upload(be, self.miss.as_ref(), &[0xa5a5_a5a5; 4]);
    }

    fn bindings(&self) -> Vec<vk::DescriptorBufferInfo> {
        let mut bindings: Vec<_> = self
            .pool
            .buffers
            .iter()
            .map(|b| Recorder::vkb_exact(b.as_ref()))
            .collect();
        if self.overlap {
            bindings.push(Recorder::vkb_exact(self.miss.as_ref()));
        }
        bindings
    }

    fn record(&self, rec: &Recorder<'_>, trial: bool, recovery: bool) {
        let slots = self.pool.slots_per_pool as u32;
        let geometry = [
            0,
            SLOT_BYTES[0],
            0,
            slots,
            0,
            SLOT_BYTES[0],
            0,
            slots,
            1,
            SLOT_BYTES[1],
            slots,
            slots,
        ];
        let mut push = self.pool.push;
        push[12] = u32::from(recovery);
        assert_eq!(owned_prepare_fused_eligible(true, &push, trial), trial);
        assert!(rec.dispatch_owned_prepare_geometry(
            &self.bindings(),
            &push,
            self.overlap,
            &geometry,
            trial,
        ));
    }

    fn snapshot(&self, be: &VulkanBackend) -> Vec<Vec<u32>> {
        let mut result: Vec<_> = self
            .pool
            .buffers
            .iter()
            .map(|b| read(be, b.as_ref()))
            .collect();
        if self.overlap {
            result.push(read(be, self.miss.as_ref()));
        }
        result
    }

    fn restore(&self, be: &VulkanBackend, initial: &[Vec<u32>]) {
        for (buffer, words) in self.pool.buffers.iter().zip(initial) {
            upload(be, buffer.as_ref(), words);
        }
        if self.overlap {
            upload(be, self.miss.as_ref(), &initial[8]);
        }
    }

    fn assert_oracle(&self, be: &VulkanBackend) {
        self.pool.assert_result(be, false, true, false, false);
        if self.overlap {
            assert_eq!(
                read(be, self.miss.as_ref()),
                [
                    self.pool.misses as u32,
                    if self.pool.misses == 0 { 0 } else { 2 },
                    1,
                    1
                ]
            );
        }
    }
}

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicit idle device"));
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(be
        .shared
        .compute_subgroup_operations
        .contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC));
    be
}

#[test]
#[ignore = "central GPU runner only; used8/10 subprocesses, exact split/fused metadata and rejection"]
fn gpu_owned_wider_prepare_exact_and_rejection() {
    let be = backend();
    for overlap in [false, true] {
        for duplicate in [false, true] {
            for workload in [Workload::AllHit, Workload::OneExpertMiss] {
                let fixture = WiderProbe::new(&be, 128, duplicate, workload, overlap);
                let mut baseline = None;
                for trial in [false, true] {
                    fixture.reset(&be);
                    let rec = be.recorder().unwrap();
                    rec.owned_copy_barrier(true, false);
                    fixture.record(&rec, trial, false);
                    rec.owned_copy_barrier(false, true);
                    rec.finish().unwrap();
                    fixture.assert_oracle(&be);
                    let actual = fixture.snapshot(&be);
                    if let Some(expected) = &baseline {
                        assert_eq!(
                            &actual, expected,
                            "all raw buffers, duplicate={duplicate} overlap={overlap} {workload:?}"
                        );
                    } else {
                        baseline = Some(actual);
                    }
                }
            }
        }
        for fault in ["id", "header", "tape_alias", "source_stride"] {
            let fixture = WiderProbe::new(&be, 128, false, Workload::OneExpertMiss, overlap);
            fixture.reset(&be);
            let mut initial = fixture.snapshot(&be);
            let mut bad_push = fixture.pool.push;
            match fault {
                "id" => initial[0][0] = EXPERTS as u32,
                "header" => initial[5][0] = 2,
                "tape_alias" => bad_push[3] = bad_push[2],
                "source_stride" => {
                    let block = fixture.pool.push[10] as usize;
                    initial[4][block * 4 + 2] = SLOT_BYTES[1];
                    initial[4][block * 4 + 3] = 1;
                }
                _ => unreachable!(),
            }
            let mut fixture = fixture;
            fixture.pool.push = bad_push;
            let mut baseline = None;
            for trial in [false, true] {
                fixture.restore(&be, &initial);
                let rec = be.recorder().unwrap();
                rec.owned_copy_barrier(true, false);
                fixture.record(&rec, trial, false);
                rec.owned_copy_barrier(false, true);
                rec.finish().unwrap();
                let actual = fixture.snapshot(&be);
                assert_eq!(actual[3][RESIDENT_JOURNAL_STATUS_WORD], 3, "{fault}");
                assert_eq!(&actual[2][..4], &[0, 0, 1, 1], "{fault}");
                assert_eq!(actual[3][RESIDENT_JOURNAL_JOB_COUNT_WORD], 0);
                for binding in [0, 1, 4, 5, 6, 7] {
                    assert_eq!(
                        actual[binding], initial[binding],
                        "{fault} binding={binding}"
                    );
                }
                if let Some(expected) = &baseline {
                    assert_eq!(
                        &actual, expected,
                        "exact rejection {fault} overlap={overlap}"
                    );
                } else {
                    baseline = Some(actual);
                }
            }
        }
    }
    eprintln!(
        "WIDER_OWNER_EXACT used={} rows=2 routes=distinct,duplicate serial_and_overlap=true raw=all lru=exact rejection=pass reference=split candidate=fused",
        probe_used()
    );
}

fn assert_fifo(snapshot: &[Vec<u32>], fixture: &WiderProbe) {
    let nodes = &snapshot[5];
    let slots = fixture.pool.slots_per_pool;
    for pool in 0..2 {
        let base = nodes[(POOL_BASE + pool * 2) * 4] as usize;
        let queue = &nodes[base * 4..base * 4 + 4];
        let (head, tail) = (queue[0] as usize, queue[1] as usize);
        assert!(head <= tail && tail <= slots * 2);
        assert_eq!(&queue[2..], &[slots as u32 * 2, 0]);
        let mut actual = Vec::new();
        for i in head..tail {
            let record = &nodes[(base + 1 + i) * 4..(base + 2 + i) * 4];
            let slot = record[2] as usize;
            assert!((pool * slots..(pool + 1) * slots).contains(&slot));
            assert_eq!(record[3], 0);
            let live = &nodes[(SLOT_BASE + slot) * 4..(SLOT_BASE + slot + 1) * 4];
            if matches!(live[3], 1 | 2) && stamp(live) == stamp(record) {
                actual.push((stamp(record), slot));
            }
        }
        let mut expected: Vec<_> = (pool * slots..(pool + 1) * slots)
            .filter(|&slot| matches!(nodes[(SLOT_BASE + slot) * 4 + 3], 1 | 2))
            .map(|slot| (stamp(&nodes[(SLOT_BASE + slot) * 4..]), slot))
            .collect();
        expected.sort_unstable();
        assert_eq!(
            actual, expected,
            "complete stable exact LRU, no duplicate live log records"
        );
        assert_eq!(
            nodes[(POOL_BASE + pool * 2) * 4 + 1] as usize,
            expected.len()
        );
    }
}

#[test]
#[ignore = "central GPU runner only; real BDA payload, CPU-only miss and guarded recovery float bits"]
fn gpu_owned_wider_prepare_cpu_recovery_exact() {
    let be = backend();
    for duplicate in [false, true] {
        let mut fixture = WiderProbe::new(&be, 128, duplicate, Workload::OneExpertMiss, false);
        let (source, source_address) = be.alloc_mapped_arena_bda(SLOT_BYTES[1] as usize).unwrap();
        let bytes = fixture.pool.slots_per_pool * (SLOT_BYTES[0] + SLOT_BYTES[1]) as usize;
        let (destination, destination_address) = be.alloc_mapped_arena_bda(bytes).unwrap();
        let pattern: Vec<u32> = (0..SLOT_BYTES[1] / 4).map(|i| 0x3f00_0000 ^ i).collect();
        upload(&be, source.as_ref(), &pattern);
        for block in 0..BLOCKS {
            fixture.pool.initial[4][block * 4..block * 4 + 2]
                .copy_from_slice(&pair(source_address));
        }
        let cpu_block = fixture.pool.push[9] as usize;
        fixture.pool.initial[4][cpu_block * 4..cpu_block * 4 + 2].fill(0);
        for slot in 0..fixture.pool.slots_per_pool * 2 {
            let pool = slot / fixture.pool.slots_per_pool;
            let offset = if pool == 0 {
                slot * SLOT_BYTES[0] as usize
            } else {
                fixture.pool.slots_per_pool * SLOT_BYTES[0] as usize
                    + (slot - fixture.pool.slots_per_pool) * SLOT_BYTES[1] as usize
            };
            let address = pair(destination_address + offset as u64);
            let at = (fixture.pool.address_base + slot) * 4;
            fixture.pool.initial[5][at..at + 2].copy_from_slice(&address);
            let block = fixture.pool.initial[5][(SLOT_BASE + slot) * 4 + 2] as usize;
            fixture.pool.initial[6][block * 4..block * 4 + 2].copy_from_slice(&address);
        }
        let mut baseline = None;
        for trial in [false, true] {
            fixture.reset(&be);
            upload(&be, destination.as_ref(), &vec![0xa5a5_a5a5; bytes / 4]);
            let before = fixture.snapshot(&be);
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            fixture.record(&rec, trial, false);
            rec.owned_copy_barrier(false, false);
            let bindings = fixture.bindings();
            let payload = be.kernel(
                "gpu_owned_payload",
                crate::gemm::gpu_resident_copy_payload_spv(false),
                2,
                4,
            );
            rec.dispatch(
                payload,
                &[bindings[3], bindings[7]],
                2,
                bytemuck::bytes_of(&0u32),
                2,
            );
            rec.owned_copy_barrier(false, false);
            let publish = be.kernel(
                "gpu_owned_publish",
                crate::gemm::gpu_owned_publish_spv(),
                4,
                8,
            );
            let publish_bindings = [bindings[3], bindings[7], bindings[5], bindings[6]];
            rec.dispatch(
                publish,
                &publish_bindings,
                2,
                bytemuck::cast_slice(&[0u32, 0]),
                1,
            );
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            let stopped = fixture.snapshot(&be);
            assert_eq!(stopped[3][RESIDENT_JOURNAL_STATUS_WORD], 5);
            assert_eq!(&stopped[2][..4], &[0, 0, 1, 0]);
            assert_fifo(&stopped, &fixture);
            let job_count = stopped[3][RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
            assert_eq!(job_count, 3, "one missing expert, three distinct roles");
            let mut expected_payload = vec![0xa5a5_a5a5; bytes / 4];
            let mut cpu_count = 0;
            for job in stopped[7][..job_count * 8].chunks_exact(8) {
                let offset = stamp(&job[2..4]).checked_sub(destination_address).unwrap() as usize;
                let len = job[4] as usize;
                assert!(offset <= bytes && len <= bytes - offset);
                if stamp(job) == 0 {
                    cpu_count += 1;
                    let ptr = as_vk_buf(destination.as_ref())
                        .unwrap()
                        .mapped_ptr()
                        .unwrap();
                    crate::copy_to_mapped(
                        &bytemuck::cast_slice::<u32, u8>(&pattern)[..len],
                        unsafe { ptr.add(offset) },
                    );
                }
                expected_payload[offset / 4..(offset + len) / 4]
                    .copy_from_slice(&pattern[..len / 4]);
            }
            assert_eq!(
                cpu_count, 1,
                "actual unmapped CPU-only job, not a mapped miss"
            );
            let rec = be.recorder().unwrap();
            rec.seed_barrier();
            rec.gpu_owned_recovery_seed(
                fixture.pool.buffers[2].as_ref(),
                fixture.pool.buffers[3].as_ref(),
                fixture.pool.buffers[5].as_ref(),
                [0, 1, 41, 7],
            );
            rec.begin_immutable_conditional_at(
                fixture.pool.buffers[2].as_ref(),
                RESIDENT_SEED_WORD,
            );
            rec.owned_copy_barrier(true, false);
            rec.dispatch(
                publish,
                &publish_bindings,
                2,
                bytemuck::cast_slice(&[0u32, 1]),
                1,
            );
            rec.owned_copy_barrier(false, false);
            rec.end_conditional();
            fixture.record(&rec, trial, true);
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            let actual = fixture.snapshot(&be);
            assert_eq!(actual[3][RESIDENT_JOURNAL_STATUS_WORD], 1);
            assert_eq!(&actual[2][..4], &[1, NONE, 1, 0]);
            assert_eq!(actual[2][RESIDENT_SEED_WORD], 1);
            assert_eq!(
                actual[5][10..12],
                stopped[5][10..12],
                "recovery does not touch clock twice"
            );
            assert_eq!(
                stamp(&actual[5][10..12]),
                stamp(&before[5][10..12]) + fixture.pool.push[0] as u64 * 3
            );
            assert_fifo(&actual, &fixture);
            for role in 0..3 {
                for &id in &fixture.pool.initial[0][..fixture.pool.push[0] as usize] {
                    let block = fixture.pool.push[9 + role] as usize + id as usize;
                    assert_eq!(actual[6][block * 4 + 2], 1);
                    let tape = (fixture.pool.push[2 + role] as usize + id as usize) * 2;
                    assert_eq!(
                        actual[1][tape..tape + 2],
                        actual[6][block * 4..block * 4 + 2]
                    );
                }
            }
            let bits = read(&be, destination.as_ref());
            assert_eq!(
                bits, expected_payload,
                "complete raw payload / f32::to_bits, including untouched bytes"
            );
            let result = (stopped, actual, bits);
            if let Some(expected) = &baseline {
                assert_eq!(
                    &result, expected,
                    "full raw before/after recovery and float bits duplicate={duplicate}"
                );
            } else {
                baseline = Some(result);
            }
        }
        // The failed-generation branch is intentionally separate from valid recovery:
        // CPU bytes may have arrived, but they must remain unpublished and unusable.
        let mut baseline = None;
        for trial in [false, true] {
            fixture.reset(&be);
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            fixture.record(&rec, trial, false);
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            let stopped = fixture.snapshot(&be);
            assert_eq!(stopped[3][RESIDENT_JOURNAL_STATUS_WORD], 5);
            let rec = be.recorder().unwrap();
            rec.seed_barrier();
            rec.gpu_owned_recovery_seed(
                fixture.pool.buffers[2].as_ref(),
                fixture.pool.buffers[3].as_ref(),
                fixture.pool.buffers[5].as_ref(),
                [0, 1, 42, 7],
            );
            let bindings = fixture.bindings();
            let publish = be.kernel(
                "gpu_owned_publish",
                crate::gemm::gpu_owned_publish_spv(),
                4,
                8,
            );
            rec.begin_immutable_conditional_at(
                fixture.pool.buffers[2].as_ref(),
                RESIDENT_SEED_WORD,
            );
            rec.owned_copy_barrier(true, false);
            rec.dispatch(
                publish,
                &[bindings[3], bindings[7], bindings[5], bindings[6]],
                2,
                bytemuck::cast_slice(&[0u32, 1]),
                1,
            );
            rec.owned_copy_barrier(false, false);
            rec.end_conditional();
            fixture.record(&rec, trial, true);
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            let actual = fixture.snapshot(&be);
            assert_eq!(actual[2][0], 0);
            assert_eq!(actual[2][3], 1);
            assert_eq!(actual[2][RESIDENT_SEED_WORD], 0);
            assert_eq!(actual[3][RESIDENT_JOURNAL_STATUS_WORD], 0);
            for binding in [0, 1, 4, 5, 6, 7] {
                assert_eq!(
                    actual[binding], stopped[binding],
                    "rejected recovery binding={binding}"
                );
            }
            if let Some(expected) = &baseline {
                assert_eq!(&actual, expected, "failed-generation recovery exact");
            } else {
                baseline = Some(actual);
            }
        }
    }
    eprintln!(
        "WIDER_OWNER_CPU_RECOVERY used={} rows=2 routes=distinct,duplicate real_bda=true cpu_only_jobs=1 exact_raw_floatbits_lru=true",
        probe_used()
    );
}

#[test]
#[ignore = "central GPU runner only; run exact/recovery first, paired module timestamps, not E2E"]
fn gpu_owned_wider_prepare_paired_latency_probe() {
    let be = backend();
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    let queries = Queries {
        be: &be,
        pool: unsafe {
            be.shared.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .unwrap(),
    };
    let period = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits
    .timestamp_period;
    for overlap in [false, true] {
        for duplicate in [false, true] {
            for workload in [Workload::AllHit, Workload::OneExpertMiss] {
                let fixture = WiderProbe::new(&be, 4096, duplicate, workload, overlap);
                let mut baseline = None;
                let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..10 {
                    for variant in [round % 2, 1 - round % 2] {
                        fixture.reset(&be);
                        let rec = be.recorder().unwrap();
                        rec.owned_copy_barrier(true, false);
                        unsafe {
                            be.shared
                                .device
                                .cmd_reset_query_pool(rec.cmd, queries.pool, 0, 2);
                            be.shared.device.cmd_write_timestamp(
                                rec.cmd,
                                vk::PipelineStageFlags::TOP_OF_PIPE,
                                queries.pool,
                                0,
                            );
                        }
                        fixture.record(&rec, variant == 1, false);
                        unsafe {
                            be.shared.device.cmd_write_timestamp(
                                rec.cmd,
                                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                queries.pool,
                                1,
                            );
                        }
                        rec.owned_copy_barrier(false, true);
                        rec.finish().unwrap();
                        let mut ticks = [0u64; 2];
                        unsafe {
                            be.shared.device.get_query_pool_results(
                                queries.pool,
                                0,
                                &mut ticks,
                                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                            )
                        }
                        .unwrap();
                        let mut delta = ticks[1].wrapping_sub(ticks[0]);
                        if be.shared.submit_timestamp_valid_bits < 64 {
                            delta &= (1u64 << be.shared.submit_timestamp_valid_bits) - 1;
                        }
                        fixture.assert_oracle(&be);
                        let actual = fixture.snapshot(&be);
                        if let Some(expected) = &baseline {
                            assert_eq!(&actual, expected, "every timed sample full raw exact");
                        } else {
                            baseline = Some(actual);
                        }
                        if round >= 2 {
                            samples[variant].push(delta as f64 * period as f64 / 1000.0);
                        }
                    }
                }
                let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
                let split = mean(&samples[0]);
                let fused = mean(&samples[1]);
                let paired: Vec<_> = samples[0]
                    .iter()
                    .zip(&samples[1])
                    .map(|(a, b)| b - a)
                    .collect();
                eprintln!(
                    "WIDER_OWNER_PAIRED used={} rows=2 slots=4096 duplicate={duplicate} overlap={overlap} workload={workload:?} pairs={} split_total_mean_us={split:.6} fused_total_mean_us={fused:.6} paired_delta_mean_us={:.6} owner_per_sample=1 reset_outside_timestamps=true exact_raw_lru=pass module_only=true reference=split candidate=fused",
                    probe_used(),
                    paired.len(),
                    mean(&paired)
                );
            }
        }
    }
}
