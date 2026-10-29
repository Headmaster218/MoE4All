//! Metadata-only recovery qualification; synthetic BDAs never reach payload dispatches.
//! Child of topk_many_row_tests to reuse its guarded Probe and the performance helpers.
use super::*;

#[derive(Clone, Copy, Debug)]
enum Arm {
    Split,
    Fused,
}

struct RecoveryKernels {
    fast: ComputeKernel,
    slow: ComputeKernel,
    fused: ComputeKernel,
    publish: ComputeKernel,
}

impl RecoveryKernels {
    fn new(be: &VulkanBackend) -> Self {
        assert!(be
            .shared
            .compute_subgroup_operations
            .contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC));
        Self {
            fast: be.kernel_sg(
                "recovery_candidate_split_fast",
                crate::gemm::gpu_owned_prepare_fast_split_spv(true),
                9,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            ),
            // Match dispatch_owned_prepare_geometry's device-selected subgroup slow arm.
            slow: be.kernel_native(
                "recovery_candidate_split_slow",
                crate::gemm::gpu_owned_prepare_slow_subgroup_spv(true),
                9,
                GPU_COPY_PREPARE_PUSH_BYTES,
            ),
            fused: be.kernel_sg(
                "recovery_candidate_non_topk_fused",
                crate::gemm::gpu_owned_prepare_fused_spv(true),
                9,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            ),
            publish: be.kernel(
                "recovery_candidate_publish",
                crate::gemm::gpu_owned_publish_spv(),
                4,
                8,
            ),
        }
    }

    fn prepare(&self, rec: &Recorder<'_>, p: &Probe, arm: Arm) {
        let buffers: Vec<_> = p.buffers[..9]
            .iter()
            .map(|b| Recorder::vkb_exact(b.as_ref()))
            .collect();
        match arm {
            Arm::Split => {
                rec.dispatch(self.fast, &buffers, 9, bytemuck::cast_slice(&p.push), 1);
                // Normal dispatch tracking supplies the same WAW dependency as production.
                rec.dispatch(
                    self.slow,
                    &buffers,
                    9,
                    bytemuck::cast_slice(&p.push[..19]),
                    1,
                );
            }
            Arm::Fused => rec.dispatch(self.fused, &buffers, 9, bytemuck::cast_slice(&p.push), 1),
        }
    }

    fn publish(&self, be: &VulkanBackend, p: &Probe, mode: u32) {
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        rec.dispatch(
            self.publish,
            &[
                Recorder::vkb_exact(p.buffers[3].as_ref()),
                Recorder::vkb_exact(p.buffers[7].as_ref()),
                Recorder::vkb_exact(p.buffers[5].as_ref()),
                Recorder::vkb_exact(p.buffers[6].as_ref()),
            ],
            2,
            bytemuck::cast_slice(&[p.shape.ordinal as u32, mode]),
            1,
        );
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
    }
}

fn raw(be: &VulkanBackend, p: &Probe) -> Raw {
    p.buffers.iter().map(|b| read(be, b.as_ref())).collect()
}

fn run(
    be: &VulkanBackend,
    kernels: &RecoveryKernels,
    p: &Probe,
    arm: Arm,
    queries: Option<&Queries<'_>>,
) -> (Raw, Option<f64>) {
    for (buffer, words) in p.buffers.iter().zip(&p.seed) {
        upload(be, buffer.as_ref(), words);
    }
    let rec = be.recorder().unwrap();
    rec.owned_copy_barrier(true, false);
    if let Some(q) = queries {
        unsafe {
            be.shared.device.cmd_reset_query_pool(rec.cmd, q.pool, 0, 2);
            be.shared.device.cmd_write_timestamp(
                rec.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                q.pool,
                0,
            );
        }
    }
    kernels.prepare(&rec, p, arm);
    if let Some(q) = queries {
        unsafe {
            be.shared.device.cmd_write_timestamp(
                rec.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                q.pool,
                1,
            );
        }
    }
    rec.owned_copy_barrier(false, true);
    rec.finish().unwrap();
    let elapsed = queries.map(|q| {
        let mut ticks = [0u64; 2];
        unsafe {
            be.shared.device.get_query_pool_results(
                q.pool,
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
        delta as f64 * be.shared.submit_timestamp_period_ns as f64 / 1000.0
    });
    (raw(be, p), elapsed)
}

struct ReadyRecovery {
    probe: Probe,
    original_hits: Vec<u32>,
}

impl ReadyRecovery {
    fn new(
        be: &VulkanBackend,
        kernels: &RecoveryKernels,
        rows: usize,
        used: usize,
        slots: usize,
        duplicate: bool,
        ordinal: usize,
    ) -> Self {
        let mut p = Probe::new(
            be,
            Shape {
                mixed: true,
                overlap: true,
                ordinal,
                ..Shape::new(rows, used)
            },
            slots,
        );
        // Non-TopK ABI: words 13..18 are not borrowed for router scale/gating/predicate.
        p.push[13..18].fill(0);
        let count = rows * used;
        if duplicate {
            for i in 0..count {
                p.seed[0][i] = (i % used) as u32;
            }
        }
        let gate = p.push[9] as usize;
        // A real GPU admission must generate status5, with CPU and imported jobs.
        p.seed[4][(gate + 1) * 4..(gate + 1) * 4 + 2].fill(0);
        let (admitted, _) = run(be, kernels, &p, Arm::Split, None);
        let base = ordinal * RESIDENT_JOURNAL_WORDS;
        assert_eq!(admitted[3][base + RESIDENT_JOURNAL_STATUS_WORD], 5);
        let jobs = admitted[3][base + RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
        assert!(jobs > 1 && jobs <= 3 * count);
        let job_base = ordinal * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
        let live_jobs = &admitted[7][job_base..job_base + jobs * 8];
        let cpu_jobs = live_jobs
            .chunks_exact(8)
            .filter(|job| job[0] == 0 && job[1] == 0)
            .count();
        assert!(cpu_jobs > 0 && cpu_jobs < jobs);
        let original_hits = admitted[3]
            [base + RESIDENT_JOURNAL_MASK_WORD..base + RESIDENT_JOURNAL_MASK_WORD + rows]
            .to_vec();
        assert!(original_hits.iter().any(|hits| *hits != (1u32 << used) - 1));

        // Only metadata is exercised. Treat transfers as completed, without dispatching
        // payload against synthetic BDAs. Execute both real publication modes in order.
        kernels.publish(be, &p, 0);
        let mapped_ready = raw(be, &p);
        for job in live_jobs.chunks_exact(8) {
            let state = if job[0] == 0 && job[1] == 0 { 2 } else { 1 };
            assert_eq!(
                mapped_ready[5][(SLOT_BASE + job[6] as usize) * 4 + 3],
                state
            );
            assert_eq!(mapped_ready[6][job[5] as usize * 4 + 2], state);
        }
        kernels.publish(be, &p, 1);
        let ready = raw(be, &p);
        let mut expected_ready = admitted.clone();
        for job in live_jobs.chunks_exact(8) {
            expected_ready[5][(SLOT_BASE + job[6] as usize) * 4 + 3] = 1;
            expected_ready[6][job[5] as usize * 4 + 2] = 1;
        }
        assert_eq!(
            ready, expected_ready,
            "publication changes readiness only, all buffers exact"
        );
        p.seed = ready;
        p.push[12] = 1;
        p.seed[2][0] = 1;
        p.seed[2][1] = NONE;
        p.seed[2][3] = 0;
        // Force observable address/mask refresh and rejection of stale copy arguments.
        p.seed[1][..3 * EXPERTS * 2].fill(GUARD);
        p.seed[0][count..count + rows * 2].fill(GUARD);
        p.seed[8][ordinal * 4..ordinal * 4 + 4].fill(GUARD);
        Self {
            probe: p,
            original_hits,
        }
    }

    fn expected(&self, status: u32, masked: bool) -> Raw {
        let p = &self.probe;
        let mut expected = p.seed.clone();
        let ordinal = p.shape.ordinal;
        let base = ordinal * RESIDENT_JOURNAL_WORDS;
        let count = p.shape.count();
        let routed = (1u32 << p.shape.used) - 1;
        expected[3]
            [base + RESIDENT_JOURNAL_JOB_COUNT_WORD..base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 4]
            .fill(0);
        expected[3][base + RESIDENT_JOURNAL_STATUS_WORD] = status;
        expected[8][ordinal * 4..ordinal * 4 + 4].copy_from_slice(&[0, 0, 1, 1]);
        expected[2][4] = 0;
        expected[2][5 + ordinal * 2..7 + ordinal * 2].fill(0);
        match status {
            1 => {
                expected[2][0] = 1;
                expected[2][2] = ordinal as u32 + 1;
                expected[2][4] = 1;
                expected[2][5 + ordinal * 2..7 + ordinal * 2].fill(1);
                expected[3][base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 1] = p.seed[5][3];
                expected[3][base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 2] = p.seed[5][4];
                for row in 0..p.shape.rows {
                    expected[3][base + RESIDENT_JOURNAL_MASK_WORD + row] = routed;
                    expected[0][count + row] = if masked {
                        routed ^ self.original_hits[row]
                    } else {
                        routed | p.push[7]
                    };
                    expected[0][count + p.shape.rows + row] = 0;
                }
                for i in 0..count {
                    let id = p.seed[0][i] as usize;
                    expected[3][base + i] = id as u32;
                    for role in 0..3 {
                        let block = p.push[9 + role] as usize + id;
                        let tape = (p.push[2 + role] as usize + id) * 2;
                        expected[1][tape..tape + 2]
                            .copy_from_slice(&p.seed[6][block * 4..block * 4 + 2]);
                    }
                }
            }
            3 => {
                expected[2][0] = 0;
                expected[2][1] = ordinal as u32;
                expected[2][2] = ordinal as u32 + 1;
                expected[2][3] = 1;
            }
            0 => assert_eq!(p.seed[2][0], 0),
            _ => panic!("unsupported expected status {status}"),
        }
        expected
    }

    fn check(&self, actual: &Raw, status: u32, masked: bool) {
        assert_eq!(
            *actual,
            self.expected(status, masked),
            "full raw expected: shape={:?} status={status} masked={masked}",
            self.probe.shape
        );
        for binding in [4, 5, 6, 7, 9, 10] {
            assert_eq!(
                actual[binding], self.probe.seed[binding],
                "sources, full ownership/FIFO/stat/clock, stale jobs and router guards unchanged"
            );
        }
    }

    fn check_masks(&self, be: &VulkanBackend) {
        let p = &self.probe;
        let rec = be.recorder().unwrap();
        rec.gpu_owned_recovery_miss_masks(
            p.buffers[0].as_ref(),
            p.shape.count(),
            p.shape.used,
            &self.original_hits,
        );
        rec.publish_host_reads();
        rec.finish().unwrap();
        self.check(&raw(be, p), 1, true);
    }
}

#[test]
#[ignore = "central serial GPU only; real status5 metadata/publish READY, recovery-only split vs existing fused"]
fn gpu_owned_recovery_non_topk_fused_exact_3_to_8() {
    let be = backend();
    let kernels = RecoveryKernels::new(&be);
    for (rows, used, duplicate, ordinal) in (3..=8)
        .flat_map(|rows| [8, 10].map(move |used| (rows, used, false, 0)))
        .chain([(8, 16, false, 0), (4, 10, true, 1), (8, 10, true, 127)])
    {
        let fixture = ReadyRecovery::new(&be, &kernels, rows, used, 256, duplicate, ordinal);
        let mut previous = None;
        for arm in [Arm::Split, Arm::Fused] {
            let (actual, _) = run(&be, &kernels, &fixture.probe, arm, None);
            fixture.check(&actual, 1, false);
            if let Some(old) = previous {
                assert_eq!(actual, old, "split vs fused exact");
            }
            previous = Some(actual);
            fixture.check_masks(&be);
        }
    }
}

#[test]
#[ignore = "central serial GPU only; unready/invalid recovery rejects without ownership writes, SKIP clears work"]
fn gpu_owned_recovery_non_topk_fused_rejection_and_skip() {
    let be = backend();
    let kernels = RecoveryKernels::new(&be);
    for fault in 0..5 {
        let mut fixture = ReadyRecovery::new(&be, &kernels, 4, 10, 256, false, 0);
        let p = &mut fixture.probe;
        let gate = p.push[9] as usize;
        let slot = p.seed[6][(gate + 1) * 4 + 3] as usize;
        let status = match fault {
            0 => {
                p.seed[6][(gate + 1) * 4 + 2] = 2;
                p.seed[5][(SLOT_BASE + slot) * 4 + 3] = 2;
                3
            }
            1 => {
                p.seed[6][(gate + 1) * 4..(gate + 1) * 4 + 4].copy_from_slice(&[0, 0, 0, NONE]);
                3
            }
            2 => {
                p.seed[5][(SLOT_BASE + slot) * 4 + 2] = gate as u32;
                3
            }
            3 => {
                p.push[3] = p.push[2] + 1;
                3
            }
            _ => {
                p.seed[2][0] = 0;
                0
            }
        };
        for arm in [Arm::Split, Arm::Fused] {
            let (actual, _) = run(&be, &kernels, &fixture.probe, arm, None);
            fixture.check(&actual, status, false);
        }
    }
}

#[test]
#[ignore = "central serial GPU only; rows4 used8/10, eight outer-only split vs existing non-TopK fused pairs"]
fn gpu_owned_recovery_non_topk_fused_paired_rows4() {
    let be = backend();
    let kernels = RecoveryKernels::new(&be);
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
    for used in [8, 10] {
        let fixture = ReadyRecovery::new(&be, &kernels, 4, used, 4096, false, 0);
        for arm in [Arm::Split, Arm::Fused] {
            let (actual, _) = run(&be, &kernels, &fixture.probe, arm, None);
            fixture.check(&actual, 1, false);
        }
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..8 {
            for index in [round % 2, 1 - round % 2] {
                let arm = [Arm::Split, Arm::Fused][index];
                let (actual, us) = run(&be, &kernels, &fixture.probe, arm, Some(&queries));
                fixture.check(&actual, 1, false);
                samples[index].push(us.unwrap());
            }
        }
        let mean = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
        let paired: Vec<_> = samples[0]
            .iter()
            .zip(&samples[1])
            .map(|(old, new)| new - old)
            .collect();
        eprintln!(
            "RECOVERY_NON_TOPK_PAIRED rows=4 used={used} slots_per_pool=4096 pairs=8 reference=production_fast_split_plus_native_slow candidate=existing_non_topk_fused_wg64_fixed32 split_us={:.6} fused_us={:.6} paired_delta_us={:.6} fresh_status5_ready_snapshot=true every_full_buffer_exact=true outer_only=true reset_readback_outside_timestamps=true metadata_only=true payload_dispatches=0 router_dispatches=0",
            mean(&samples[0]),
            mean(&samples[1]),
            mean(&paired),
        );
    }
}
