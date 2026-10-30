#![cfg(test)]

//! Continuous metadata-only replay of the existing serial external-SPV TopK fixture.
use super::*;

type ReplayRaw = [Vec<u32>; 10];
const USED: usize = 8;
const ACTIVE_EXPERTS: usize = 256;
const SLOTS: usize = 4096;

fn snapshot(be: &VulkanBackend, probe: &TopkProbe) -> ReplayRaw {
    std::array::from_fn(|i| match i {
        8 => read(be, probe.weights.as_ref()),
        9 => read(be, probe.logits.as_ref()),
        _ => read(be, probe.fixture.buffers[i].as_ref()),
    })
}

fn new_probe(be: &VulkanBackend, kernels: &TopkKernels) -> TopkProbe {
    TopkProbe::new_with_experts(
        be,
        kernels,
        Case::normal("continuous_compaction", USED, Workload::AllHit),
        SLOTS,
        ACTIVE_EXPERTS,
    )
}

#[derive(Clone, Copy)]
struct Touch {
    block: usize,
    slot: usize,
    pool: usize,
    tape: usize,
}

struct ReplayOracle {
    raw: ReplayRaw,
    seed: ReplayRaw,
    touches: Vec<Touch>,
    ids: Vec<u32>,
    owners: usize,
    compactions: [usize; 2],
    dropped_records: [usize; 2],
}

impl ReplayOracle {
    fn new(probe: &TopkProbe, seed: ReplayRaw) -> Self {
        assert_eq!(probe.fused_push[0], USED as u32);
        assert_eq!(probe.fused_push[1], ACTIVE_EXPERTS as u32);
        assert_eq!(probe.fused_push[5], 0);
        assert_eq!(probe.fused_push[6], USED as u32);
        assert_eq!(probe.fused_push[12], 0, "all-hit, not recovery");
        assert_eq!(probe.fixture.misses, 0);
        assert_eq!(probe.fixture.repeats, 1);
        let ids = probe.fixture.expected[0][..USED].to_vec();
        let mut unique_ids = ids.clone();
        unique_ids.sort_unstable();
        assert_eq!(unique_ids, (0..USED as u32).collect::<Vec<_>>());
        let mut blocks = HashSet::new();
        let mut touches = Vec::new();
        for role in 0..3 {
            for &id in &ids {
                let block = probe.fused_push[9 + role] as usize + id as usize;
                assert!(
                    blocks.insert(block),
                    "this one shape has disjoint role banks"
                );
                let overlay = &seed[6][block * 4..block * 4 + 4];
                assert_eq!(overlay[2], 1);
                let pool = seed[4][block * 4 + 3] as usize;
                assert_eq!(pool, usize::from(role == 2));
                touches.push(Touch {
                    block,
                    slot: overlay[3] as usize,
                    pool,
                    tape: probe.fused_push[2 + role] as usize + id as usize,
                });
            }
        }
        assert_eq!(touches.len(), USED * 3);
        Self {
            raw: seed.clone(),
            seed,
            touches,
            ids,
            owners: 0,
            compactions: [0; 2],
            dropped_records: [0; 2],
        }
    }

    fn advance(&mut self, probe: &TopkProbe, owners: usize) {
        for _ in 0..owners {
            // Compact against the OLD slot stamps, before this request updates them.
            for pool in 0..2 {
                let pn = (POOL_BASE + pool * 2) * 4;
                let header = self.raw[5][pn] as usize * 4;
                let head = self.raw[5][header] as usize;
                let tail = self.raw[5][header + 1] as usize;
                let capacity = self.raw[5][header + 2] as usize;
                let appended = self.touches.iter().filter(|t| t.pool == pool).count();
                assert_eq!(capacity, 2 * SLOTS);
                assert!(head <= tail && tail <= capacity);
                if appended > capacity - tail {
                    let mut kept = Vec::new();
                    for index in head..tail {
                        let at = header + 4 + index * 4;
                        let record = &self.raw[5][at..at + 4];
                        let slot = record[2] as usize;
                        assert!((pool * SLOTS..(pool + 1) * SLOTS).contains(&slot));
                        assert_eq!(record[3], 0);
                        let resident = &self.raw[5][(SLOT_BASE + slot) * 4..][..4];
                        if matches!(resident[3], 1 | 2) && stamp(record) == stamp(resident) {
                            kept.push([record[0], record[1], record[2], 0]);
                        }
                    }
                    assert_eq!(kept.len(), self.raw[5][pn + 1] as usize);
                    assert_eq!(kept.len(), SLOTS);
                    assert!(kept.len() + appended < tail, "a real tail drop is required");
                    for (index, record) in kept.iter().enumerate() {
                        let at = header + 4 + index * 4;
                        self.raw[5][at..at + 4].copy_from_slice(record);
                    }
                    // The shader preserves unused physical log words; never zero the suffix.
                    self.raw[5][header] = 0;
                    self.raw[5][header + 1] = kept.len() as u32;
                    self.compactions[pool] += 1;
                    self.dropped_records[pool] += tail - head - kept.len();
                }
            }
            let old_clock = stamp(&self.raw[5][10..12]);
            for (q, touch) in self.touches.iter().enumerate() {
                let next = pair(old_clock.checked_add(q as u64 + 1).unwrap());
                let slot_at = (SLOT_BASE + touch.slot) * 4;
                assert_eq!(self.raw[5][slot_at + 2], touch.block as u32);
                assert_eq!(self.raw[5][slot_at + 3], 1);
                self.raw[5][slot_at..slot_at + 2].copy_from_slice(&next);
                let pn = (POOL_BASE + touch.pool * 2) * 4;
                let header = self.raw[5][pn] as usize * 4;
                let tail = self.raw[5][header + 1] as usize;
                assert!(tail < self.raw[5][header + 2] as usize);
                let at = header + 4 + tail * 4;
                self.raw[5][at..at + 4].copy_from_slice(&[next[0], next[1], touch.slot as u32, 0]);
                self.raw[5][header + 1] += 1;
                self.raw[5][pn + 4] = self.raw[5][pn + 4].checked_add(1).unwrap();
                let address = [
                    self.raw[6][touch.block * 4],
                    self.raw[6][touch.block * 4 + 1],
                ];
                self.raw[1][touch.tape * 2..touch.tape * 2 + 2].copy_from_slice(&address);
            }
            self.raw[5][8] = self.raw[5][8].checked_add((USED * 3) as u32).unwrap();
            self.raw[5][10..12].copy_from_slice(&pair(old_clock + (USED * 3) as u64));
            self.raw[0][..USED].copy_from_slice(&self.ids);
            self.raw[0][USED] = (1 << USED) - 1;
            self.raw[0][USED + 1] = 0;
            self.raw[2][2] = 1;
            self.raw[2][4] = 1;
            self.raw[2][5..7].copy_from_slice(&[1, 1]);
            self.raw[3][..USED].copy_from_slice(&self.ids);
            self.raw[3][RESIDENT_JOURNAL_MASK_WORD] = (1 << USED) - 1;
            let generation = [0, self.raw[5][3], self.raw[5][4]];
            self.raw[3][RESIDENT_JOURNAL_JOB_COUNT_WORD..RESIDENT_JOURNAL_JOB_COUNT_WORD + 3]
                .copy_from_slice(&generation);
            self.raw[3][RESIDENT_JOURNAL_STATUS_WORD] = 1;
            self.raw[8].copy_from_slice(&probe.expected_weights);
            self.owners += 1;
        }
    }

    fn check_existing_lru_reference(&self, probe: &TopkProbe) {
        assert!(self.owners > 0);
        // The workload repeats the same real all-hit routes. Move only additive counters
        // to the penultimate request, then use the existing CPU exact-LRU oracle for the last.
        let mut before_last: [Vec<u32>; 8] = std::array::from_fn(|i| self.seed[i].clone());
        before_last[0][..USED].copy_from_slice(&self.ids);
        let prior = self.owners - 1;
        for pool in 0..2 {
            let count = self.touches.iter().filter(|t| t.pool == pool).count();
            let stats = (POOL_BASE + pool * 2 + 1) * 4;
            before_last[5][stats] += u32::try_from(prior * count).unwrap();
        }
        before_last[5][8] += u32::try_from(prior * USED * 3).unwrap();
        let clock = stamp(&self.seed[5][10..12]) + (prior * USED * 3) as u64;
        before_last[5][10..12].copy_from_slice(&pair(clock));
        let roles = std::array::from_fn(|r| probe.fused_push[9 + r] as usize);
        let (mut expected, misses) = reference(
            &before_last,
            &self.ids,
            roles,
            SLOTS,
            probe.fixture.address_base,
        );
        assert_eq!(misses, 0);
        // The inherited CPU fixture defaults to used10; only these single-row masks differ.
        expected[0][USED] = (1 << USED) - 1;
        expected[0][USED + 1] = 0;
        expected[3][RESIDENT_JOURNAL_MASK_WORD] = (1 << USED) - 1;
        for i in [0, 1, 2, 3, 4, 6, 7] {
            assert_eq!(self.raw[i], expected[i], "existing CPU oracle binding={i}");
        }
        assert_eq!(
            &self.raw[5][..probe.fixture.heap_base * 4],
            &expected[5][..probe.fixture.heap_base * 4],
            "CPU exact LRU, clock, owner slots, geometry and statistics"
        );
    }

    fn assert_final(&self, be: &VulkanBackend, probe: &TopkProbe) -> ReplayRaw {
        self.check_existing_lru_reference(probe);
        let actual = snapshot(be, probe);
        assert_eq!(
            actual, self.raw,
            "all8 raw buffers plus every weight/logit bit and tail"
        );
        let clock = stamp(&actual[5][10..12]);
        let mut tails = [0usize; 2];
        for pool in 0..2 {
            let pn = (POOL_BASE + pool * 2) * 4;
            let header = actual[5][pn] as usize * 4;
            let head = actual[5][header] as usize;
            let tail = actual[5][header + 1] as usize;
            assert_eq!(actual[5][header + 2], (SLOTS * 2) as u32);
            assert_eq!(actual[5][header + 3], 0);
            assert!(head <= tail && tail <= 2 * SLOTS);
            tails[pool] = tail;
            let mut live = Vec::new();
            let mut seen = HashSet::new();
            for index in head..tail {
                let at = header + 4 + index * 4;
                let record = &actual[5][at..at + 4];
                let slot = record[2] as usize;
                assert!((pool * SLOTS..(pool + 1) * SLOTS).contains(&slot));
                assert_eq!(record[3], 0);
                assert!(stamp(record) != 0 && stamp(record) <= clock);
                let node = &actual[5][(SLOT_BASE + slot) * 4..][..4];
                if matches!(node[3], 1 | 2) && stamp(record) == stamp(node) {
                    assert!(seen.insert(slot), "exactly one matching log per resident");
                    live.push((stamp(record), slot));
                }
            }
            let mut lru = Vec::new();
            for slot in pool * SLOTS..(pool + 1) * SLOTS {
                let node = &actual[5][(SLOT_BASE + slot) * 4..][..4];
                assert_eq!(node[3], 1, "all-hit fixture has only READY slots");
                assert!(stamp(node) != 0 && stamp(node) <= clock);
                let block = node[2] as usize;
                let overlay = &actual[6][block * 4..block * 4 + 4];
                let address = &actual[5][(probe.fixture.address_base + slot) * 4..][..4];
                assert_eq!(&overlay[..2], &address[..2]);
                assert_eq!(&overlay[2..], &[1, slot as u32]);
                assert_eq!(address[2], pool as u32);
                assert_eq!(address[3], (slot - pool * SLOTS) as u32);
                assert_eq!(actual[4][block * 4 + 3], pool as u32);
                assert_eq!(actual[4][block * 4 + 2], SLOT_BYTES[pool]);
                lru.push((stamp(node), slot));
            }
            lru.sort_unstable();
            assert_eq!(live, lru, "stable FIFO covers complete CPU exact LRU");
            let added = self.touches.iter().filter(|t| t.pool == pool).count() * self.owners;
            assert!(SLOTS + added > 2 * SLOTS);
            assert!(
                tail < SLOTS + added,
                "persistent metadata proves history was reclaimed"
            );
        }
        assert!(self.compactions.iter().sum::<usize>() >= 4);
        assert!(self.dropped_records.iter().all(|&n| n > 0));
        eprintln!(
            "GPU_OWNER_COMPACTION_REPLAY_COVERAGE owners={} used=8 experts=256 slots_per_pool=4096 expected_compactions={:?} reclaimed_records={:?} final_tails={tails:?} all10_raw=pass float_bits=pass exact_lru=pass seed_once=true per_step_readback=false",
            self.owners, self.compactions, self.dropped_records
        );
        actual
    }
}

fn replay_batch(
    be: &VulkanBackend,
    probe: &TopkProbe,
    kernel: ComputeKernel,
    owners: usize,
    query: Option<&Queries<'_>>,
) -> Option<f64> {
    assert!([512, 1024].contains(&owners));
    let mut bindings: Vec<_> = probe
        .fixture
        .buffers
        .iter()
        .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
        .collect();
    bindings.extend([
        Recorder::vkb_exact(probe.logits.as_ref()),
        Recorder::vkb_exact(probe.weights.as_ref()),
    ]);
    let rec = be.recorder().unwrap();
    rec.owned_copy_barrier(true, false);
    if let Some(query) = query {
        unsafe {
            be.shared
                .device
                .cmd_reset_query_pool(rec.cmd, query.pool, 0, 2);
            be.shared.device.cmd_write_timestamp(
                rec.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                query.pool,
                0,
            );
        }
    }
    // Both variants get the identical production dependency barrier between owners.
    // Never reset directory, control, journal, IDs or weights inside the continuous chain.
    for index in 0..owners {
        if index != 0 {
            rec.owned_copy_barrier(false, false);
        }
        rec.dispatch(
            kernel,
            &bindings,
            10,
            bytemuck::cast_slice(&probe.fused_push),
            1,
        );
    }
    if let Some(query) = query {
        unsafe {
            be.shared.device.cmd_write_timestamp(
                rec.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                query.pool,
                1,
            );
        }
    }
    rec.owned_copy_barrier(false, true);
    rec.finish().unwrap();
    query.map(|query| {
        let mut ticks = [0u64; 2];
        unsafe {
            be.shared.device.get_query_pool_results(
                query.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .unwrap();
        let bits = be.shared.submit_timestamp_valid_bits;
        let mut delta = ticks[1].wrapping_sub(ticks[0]);
        if bits < 64 {
            delta &= (1u64 << bits) - 1;
        }
        delta as f64 * be.shared.submit_timestamp_period_ns as f64 / 1000.0
    })
}

fn exact_gate(be: &VulkanBackend, kernels: &TopkKernels, previous: &TopkKernels) {
    let probes = [new_probe(be, kernels), new_probe(be, kernels)];
    let handles = [previous.fused, kernels.fused];
    let mut results = Vec::new();
    for (probe, kernel) in probes.iter().zip(handles) {
        probe.reset(be); // One seed per complete correctness chain, outside timestamps.
        let mut oracle = ReplayOracle::new(probe, snapshot(be, probe));
        replay_batch(be, probe, kernel, 1024, None);
        oracle.advance(probe, 1024);
        assert_eq!(oracle.compactions, [3, 1]);
        results.push(oracle.assert_final(be, probe));
    }
    assert_eq!(
        results[0], results[1],
        "baseline/candidate complete physical FIFO storage"
    );
    eprintln!(
        "GPU_OWNER_COMPACTION_REPLAY_GATE PASS owners=1024 expected_compactions=[3,1] timing_started=false"
    );
}

#[test]
#[ignore = "parent serial GPU only; continuous external TopK FIFO oracle and actual compaction coverage"]
fn gpu_owned_topk_continuous_compaction_exact() {
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let previous = kernels.previous_fused(&be);
    exact_gate(&be, &kernels, &previous);
}

#[test]
#[ignore = "parent serial idle GPU only; correctness gate then persistent 512/1024-owner batches"]
fn gpu_owned_topk_continuous_compaction_paired_latency_probe() {
    const ROUNDS: usize = 6;
    const WARMUP: usize = 2;
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let previous = kernels.previous_fused(&be);
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    exact_gate(&be, &kernels, &previous); // No timing until real compaction/raw/LRU checks pass.
    let probes = [new_probe(&be, &kernels), new_probe(&be, &kernels)];
    let handles = [previous.fused, kernels.fused];
    let mut oracles: Vec<_> = probes
        .iter()
        .map(|probe| {
            probe.reset(&be); // One seed per timing stream; never inside either rounds loop.
            ReplayOracle::new(probe, snapshot(&be, probe))
        })
        .collect();
    assert_eq!(oracles[0].seed, oracles[1].seed);
    let queries = Queries {
        be: &be,
        pool: unsafe {
            be.shared
                .device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
                .unwrap()
        },
    };
    let mut reports = Vec::new();
    let mut total_owners = [0usize; 2];
    for owners in [512usize, 1024] {
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..ROUNDS {
            for offset in 0..2 {
                let arm = (round + offset) % 2;
                let elapsed =
                    replay_batch(&be, &probes[arm], handles[arm], owners, Some(&queries)).unwrap();
                total_owners[arm] += owners;
                if round >= WARMUP {
                    samples[arm].push(elapsed);
                }
            }
        }
        reports.push((owners, samples));
    }
    // Keep CPU simulation out of the gaps between timed GPU arms.
    for (arm, &owners) in total_owners.iter().enumerate() {
        oracles[arm].advance(&probes[arm], owners);
    }
    // Only timestamp queries are read per batch. Metadata is downloaded once at stream end.
    let results: Vec<_> = probes
        .iter()
        .zip(&oracles)
        .map(|(probe, oracle)| oracle.assert_final(&be, probe))
        .collect();
    assert_eq!(results[0], results[1]);
    assert_eq!(oracles[0].owners, ROUNDS * (512 + 1024));
    assert_eq!(oracles[0].compactions, [35, 17]);
    for (owners, samples) in reports {
        for (name, values) in ["baseline", "candidate"].into_iter().zip(samples) {
            assert_eq!(values.len(), ROUNDS - WARMUP);
            let total_mean = values.iter().sum::<f64>() / values.len() as f64;
            eprintln!(
                "GPU_OWNER_COMPACTION_REPLAY_PAIRED variant={name} batch_owners={owners} measured_batches={} warmup_batches={WARMUP} total_mean_us={total_mean:.3} amortized_mean_us_per_owner={:.6} min_batch_us={:.3} max_batch_us={:.3} stream_owners={} expected_stream_compactions={:?} seed_once=true reset_between_batches=false all10_raw=pass float_bits=pass exact_lru=pass module_only=true",
                values.len(),
                total_mean / owners as f64,
                values.iter().copied().fold(f64::INFINITY, f64::min),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                oracles[0].owners,
                oracles[0].compactions,
            );
        }
    }
}
