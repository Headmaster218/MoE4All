//! Payload-only imported-host BDA sweep. No planner, FFN, DMA, or production policy changes.
#[cfg(test)]
mod fixtures {
    use super::super::*;
    use crate::AlignedHostBuffer;
    use std::ops::Range;
    use std::sync::Arc;

    const MIB: usize = 1024 * 1024;
    const RING_BYTES: usize = 256 * MIB;
    const GUARD: usize = 256;
    const POISON: u8 = 0xa5;
    const PAIRS: usize = 12;
    const JOB_CAPACITY: usize = RESIDENT_JOURNAL_ID_WORDS * 3;

    fn spans(bytes: usize, experts: usize) -> (Vec<Range<usize>>, usize) {
        let mut remaining = bytes / 16;
        let mut cursor = GUARD;
        let count = experts * 3;
        let ranges = (0..count)
            .map(|job| {
                // Gate/Up IQ2_S : Down IQ4_NL byte ratio for equal matrix dimensions.
                let units = if job + 1 == count {
                    remaining
                } else {
                    (bytes / 16) * [82, 82, 144][job % 3] / (experts * 308)
                };
                remaining -= units;
                let start = cursor;
                cursor += units * 16 + GUARD;
                start..cursor - GUARD
            })
            .collect();
        (ranges, cursor)
    }

    fn pattern(word: u32) -> [u8; 4] {
        let mut value = word.wrapping_add(0x9e37_79b9);
        value = (value ^ (value >> 16)).wrapping_mul(0x85eb_ca6b);
        value = (value ^ (value >> 13)).wrapping_mul(0xc2b2_ae35);
        ((value ^ (value >> 16)) | 0x0101_0101).to_le_bytes()
    }

    fn write_mapped(buffer: &dyn Buffer, bytes: &[u8]) {
        assert!(bytes.len() <= buffer.len_bytes());
        let pointer = as_vk_buf(buffer)
            .unwrap()
            .mapped_ptr()
            .expect("coherent mapped fixture buffer");
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
        }
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5
    }

    struct TimestampPool<'a> {
        device: &'a ash::Device,
        handle: vk::QueryPool,
    }

    impl Drop for TimestampPool<'_> {
        fn drop(&mut self) {
            unsafe {
                self.device.destroy_query_pool(self.handle, None);
            }
        }
    }

    #[test]
    fn gpu_copy_payload_sweep_layout_visits_over_128_mib_with_byte_guards() {
        for bytes in [2 * MIB, 8 * MIB, 16 * MIB] {
            for experts in [4, 16] {
                let (ranges, span) = spans(bytes, experts);
                assert!(ranges.len() <= JOB_CAPACITY);
                assert_eq!(ranges.iter().map(|range| range.len()).sum::<usize>(), bytes);
                assert_eq!(ranges[0].start, GUARD);
                assert_eq!(span - ranges.last().unwrap().end, GUARD);
                assert!(ranges.iter().all(|range| !range.is_empty()
                    && range.start.is_multiple_of(16)
                    && range.len().is_multiple_of(16)));
                assert!(ranges
                    .windows(2)
                    .all(|pair| pair[1].start - pair[0].end == GUARD));
                let banks = RING_BYTES / span;
                let samples = 5 * (PAIRS + 1) * 2;
                assert!(banks.min(samples) * bytes > 128 * MIB);
            }
        }
        let bytes: Vec<_> = (0..256).flat_map(pattern).collect();
        assert!(bytes.iter().all(|&value| value != 0));
        assert!(bytes.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    #[ignore = "requires MOE4ALL_TEST_DEVICE selecting an idle GPU, coherent imported-host BDA, 1024-thread workgroups and timestamps"]
    fn gpu_copy_payload_groups_imported_host_ring_sweep() {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the idle GPU explicitly"));
        cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
        let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        let device = &backend.shared.device;
        let properties = unsafe {
            backend
                .shared
                .instance
                .get_physical_device_properties(backend.shared.physical_device)
        };
        assert!(properties.limits.max_compute_work_group_invocations >= 1024);
        assert!(properties.limits.max_compute_work_group_size[0] >= 1024);
        let bits = backend.shared.submit_timestamp_valid_bits;
        assert!(bits > 0);
        let period_ns = backend.shared.submit_timestamp_period_ns as f64;
        assert!(period_ns.is_finite() && period_ns > 0.0);

        // Commit and populate ordinary RAM before importing. Every sampled bank has distinct data.
        let host = Arc::new(AlignedHostBuffer::new(RING_BYTES).unwrap());
        let source = unsafe { std::slice::from_raw_parts_mut(host.as_ptr(), RING_BYTES) };
        for (word, output) in source.chunks_exact_mut(4).enumerate() {
            output.copy_from_slice(&pattern(word as u32));
        }
        let imported = backend
            .try_import_host_shard(
                backend
                    .shared
                    .external_memory_host
                    .as_ref()
                    .expect("host import required"),
                Arc::clone(&host),
                0,
                host.allocated_len(),
            )
            .unwrap();
        assert!(matches!(
            &as_vk_buf(imported.buffer.as_ref()).unwrap().backing,
            Backing::ImportedHost { .. }
        ));
        let source_address = imported
            .buffer
            .device_addr()
            .expect("import must expose BDA, not ordinary DMA fallback");
        let source = unsafe { std::slice::from_raw_parts(host.as_ptr(), RING_BYTES) };
        let kernels = [false, true].map(|wide| {
            backend.kernel(
                if wide {
                    "gpu_resident_copy_payload1024"
                } else {
                    "gpu_resident_copy_payload"
                },
                crate::gemm::gpu_resident_copy_payload_spv(wide),
                2,
                4,
            )
        });
        let variants = [
            (256, 2, kernels[0]),
            (256, 4, kernels[0]),
            (256, 8, kernels[0]),
            (1024, 2, kernels[1]),
            (1024, 4, kernels[1]),
            (1024, 8, kernels[1]),
        ];
        let baseline = variants[3];
        let query = TimestampPool {
            device,
            handle: unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
            }
            .unwrap(),
        };
        println!("PAYLOAD_SWEEP_SETUP ring_bytes={RING_BYTES} source=ImportedHostBDA device_type={:?} timestamp_bits={bits} timestamp_period_ns={period_ns} pairs={PAIRS} timings=payload_only submit=one_per_payload units=decimal_GBps", properties.device_type);

        for bytes in [2 * MIB, 8 * MIB, 16 * MIB] {
            for experts in [4, 16] {
                let (ranges, span) = spans(bytes, experts);
                let bank_count = RING_BYTES / span;
                let (destination, destination_address) =
                    backend.alloc_mapped_arena_bda(2 * span).unwrap();
                let journal = backend
                    .alloc_mapped_arena_bda(2 * RESIDENT_JOURNAL_BYTES)
                    .unwrap()
                    .0;
                let jobs = backend
                    .alloc_mapped_arena_bda(2 * JOB_CAPACITY * 32)
                    .unwrap()
                    .0;
                let mut journal_words = vec![0u32; 2 * RESIDENT_JOURNAL_WORDS];
                for ordinal in 0..2 {
                    journal_words
                        [ordinal * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_STATUS_WORD] = 4;
                    journal_words
                        [ordinal * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_JOB_COUNT_WORD] =
                        ranges.len() as u32;
                }
                write_mapped(journal.as_ref(), bytemuck::cast_slice(&journal_words));
                let mut job_words = vec![0u32; 2 * JOB_CAPACITY * 8];
                let mut expected = vec![POISON; 2 * span];
                let mut observed = vec![0u8; 2 * span];
                let mut cursor = 0usize;
                let mut visited = HashSet::new();
                let mut measure_pair =
                    |order: [(u32, u32, ComputeKernel); 2]| -> ([f64; 2], [usize; 2]) {
                        expected.fill(POISON);
                        let banks = std::array::from_fn(|_| {
                            let bank = cursor % bank_count;
                            cursor += 1;
                            visited.insert(bank);
                            bank
                        });
                        assert_ne!(banks[0], banks[1]);
                        for (ordinal, &bank) in banks.iter().enumerate() {
                            let source_base = bank * span;
                            let destination_base = ordinal * span;
                            for (job, range) in ranges.iter().enumerate() {
                                let src = source_address + (source_base + range.start) as u64;
                                let dst =
                                    destination_address + (destination_base + range.start) as u64;
                                let index = (ordinal * JOB_CAPACITY + job) * 8;
                                job_words[index..index + 8].copy_from_slice(&[
                                    src as u32,
                                    (src >> 32) as u32,
                                    dst as u32,
                                    (dst >> 32) as u32,
                                    range.len() as u32,
                                    job as u32,
                                    job as u32,
                                    job as u32,
                                ]);
                                expected
                                    [destination_base + range.start..destination_base + range.end]
                                    .copy_from_slice(
                                        &source[source_base + range.start..source_base + range.end],
                                    );
                            }
                        }
                        write_mapped(jobs.as_ref(), bytemuck::cast_slice(&job_words));
                        let pointer = as_vk_buf(destination.as_ref())
                            .unwrap()
                            .mapped_ptr()
                            .unwrap();
                        unsafe {
                            std::ptr::write_bytes(pointer, POISON, 2 * span);
                        }
                        let mask = if bits >= 64 {
                            u64::MAX
                        } else {
                            (1u64 << bits) - 1
                        };
                        let mut elapsed = [0f64; 2];
                        for ordinal in 0..2 {
                            // Finish and wait each payload before recording the next TOP timestamp.
                            // CPU encoding, submission and fence wait are outside the query interval.
                            let rec = backend.recorder().unwrap();
                            assert!(
                                !rec.prof_ops && rec.timeline_query.borrow().is_none(),
                                "disable per-op/timeline profiling for this timestamp fixture"
                            );
                            assert!(!rec.no_barrier && !rec.full_barrier);
                            unsafe {
                                device.cmd_reset_query_pool(rec.cmd, query.handle, 0, 2);
                            }
                            rec.seed_barrier();
                            // Payload destinations are BDA writes, invisible to descriptor tracking.
                            // Join previous payloads before each timed dispatch and every destination reuse.
                            let join = vk::MemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_access_mask(
                                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                                );
                            unsafe {
                                device.cmd_pipeline_barrier(
                                    rec.cmd,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::DependencyFlags::empty(),
                                    &[join],
                                    &[],
                                    &[],
                                );
                            }
                            unsafe {
                                device.cmd_write_timestamp(
                                    rec.cmd,
                                    vk::PipelineStageFlags::TOP_OF_PIPE,
                                    query.handle,
                                    0,
                                );
                            }
                            rec.dispatch(
                                order[ordinal].2,
                                &[
                                    Recorder::vkb(journal.as_ref()),
                                    Recorder::vkb(jobs.as_ref()),
                                ],
                                0,
                                bytemuck::bytes_of(&(ordinal as u32)),
                                order[ordinal].1,
                            );
                            unsafe {
                                device.cmd_write_timestamp(
                                    rec.cmd,
                                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                    query.handle,
                                    1,
                                );
                            }
                            rec.publish_host_reads();
                            rec.finish().unwrap();
                            let mut ticks = [0u64; 2];
                            unsafe {
                                device.get_query_pool_results(
                                    query.handle,
                                    0,
                                    &mut ticks,
                                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                                )
                            }
                            .unwrap();
                            let us = (ticks[1].wrapping_sub(ticks[0]) & mask) as f64 * period_ns
                                / 1000.0;
                            assert!(us.is_finite() && us > 0.0);
                            elapsed[ordinal] = us;
                        }
                        backend
                            .download(destination.as_ref(), &mut observed)
                            .unwrap();
                        if let Some(index) = observed
                            .iter()
                            .zip(&expected)
                            .position(|(actual, expected)| actual != expected)
                        {
                            panic!("payload/guard mismatch: bytes={bytes} experts={experts} index={index} banks={banks:?} configs={:?} actual={} expected={}",
                        order.map(|(local, groups, _)| (local, groups)), observed[index], expected[index]);
                        }
                        (elapsed, banks)
                    };
                for candidate in variants
                    .iter()
                    .copied()
                    .filter(|variant| (variant.0, variant.1) != (baseline.0, baseline.1))
                {
                    // One complete warm-up pair; its fresh source banks still count as visited.
                    measure_pair([baseline, candidate]);
                    let mut base_us = Vec::with_capacity(PAIRS);
                    let mut candidate_us = Vec::with_capacity(PAIRS);
                    let mut ratios = Vec::with_capacity(PAIRS);
                    let mut source_banks = Vec::with_capacity(PAIRS);
                    for pair in 0..PAIRS {
                        let reverse = pair % 2 != 0;
                        let (times, banks) = measure_pair(if reverse {
                            [candidate, baseline]
                        } else {
                            [baseline, candidate]
                        });
                        let (base, compared) = if reverse {
                            (times[1], times[0])
                        } else {
                            (times[0], times[1])
                        };
                        base_us.push(base);
                        candidate_us.push(compared);
                        ratios.push(base / compared);
                        source_banks.push(banks);
                    }
                    let base_median = median(&base_us);
                    let candidate_median = median(&candidate_us);
                    println!("PAYLOAD_SWEEP {{\"bytes\":{bytes},\"experts\":{experts},\"jobs\":{},\"baseline_local\":1024,\"baseline_groups\":2,\"candidate_local\":{},\"candidate_groups\":{},\"pairs\":{PAIRS},\"order\":\"AB_BA\",\"baseline_us\":{base_us:?},\"candidate_us\":{candidate_us:?},\"source_banks_execution_order\":{source_banks:?},\"baseline_median_us\":{base_median},\"candidate_median_us\":{candidate_median},\"baseline_GBps\":{},\"candidate_GBps\":{},\"median_paired_speedup\":{},\"guards\":\"all_bytes_every_pair\"}}",
                    ranges.len(), candidate.0, candidate.1, bytes as f64 / (base_median * 1000.0), bytes as f64 / (candidate_median * 1000.0), median(&ratios));
                }
                let visited_bytes = visited.len() * bytes;
                assert!(
                    visited_bytes > 128 * MIB,
                    "allocated capacity is not a visited working set"
                );
                println!("PAYLOAD_SWEEP_WORKING_SET payload_bytes={bytes} experts={experts} bank_span={span} banks={bank_count} visited_banks={} visited_payload_bytes={visited_bytes} trials={cursor}", visited.len());
            }
        }
    }
}
