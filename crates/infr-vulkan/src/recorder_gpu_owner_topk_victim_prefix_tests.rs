//! Metadata-only external-SPV differential; no payload dereferences synthetic BDAs.
use super::*;

type Raw = [Vec<u32>; 10];
type Seed = [Vec<u32>; 8];
const USED: usize = 10;
const SLOTS: usize = 2048;

#[derive(Clone, Copy, Debug)]
enum Head {
    Victim,
    ProtectedReady,
    Copying,
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    None,
    BeforeLive,
    LiveHead,
    AfterCutoff,
}

struct PrefixCase {
    seed: Seed,
    ids: Vec<u32>,
    roles: [usize; 3],
    heads: [usize; 2],
    cutoffs: [usize; 2],
}

fn protected(ids: &[u32], roles: [usize; 3]) -> HashSet<usize> {
    roles
        .into_iter()
        .flat_map(|base| ids.iter().map(move |&id| base + id as usize))
        .collect()
}

fn record(words: &[u32], index: usize) -> [u32; 4] {
    words[index * 4..index * 4 + 4].try_into().unwrap()
}

fn put(words: &mut [u32], index: usize, value: [u32; 4]) {
    words[index * 4..index * 4 + 4].copy_from_slice(&value);
}

impl PrefixCase {
    fn new(be: &VulkanBackend, probe: &TopkProbe, prefix: usize, head: Head) -> Self {
        assert!([0, 64, 512, 1024].contains(&prefix));
        assert_eq!(probe_used(), USED);
        assert_eq!(probe.fused_push[0], USED as u32);
        assert_eq!(probe.fused_push[5], 0);
        assert_eq!(probe.fused_push[6], USED as u32);
        assert_eq!(probe.fused_push[12], 0);
        probe.reset(be);
        let mut seed: Seed = std::array::from_fn(|i| read(be, probe.fixture.buffers[i].as_ref()));
        let ids = probe.fixture.expected[0][..USED].to_vec();
        let roles = std::array::from_fn(|r| probe.fused_push[9 + r] as usize);
        let routed = protected(&ids, roles);
        let mut demand = [0usize; 2];
        for &block in &routed {
            if seed[6][block * 4 + 2] != 1 {
                demand[seed[4][block * 4 + 3] as usize] += 1;
            }
        }
        assert_eq!(demand, [2, 1], "real missing ID0 in all three role banks");
        let old_clock = stamp(&seed[5][10..12]);
        let mut heads = [0; 2];
        let mut cutoffs = [0; 2];
        for pool in 0..2 {
            let span = pool * SLOTS..(pool + 1) * SLOTS;
            let head_slot = span
                .clone()
                .find(|&slot| {
                    let node = record(&seed[5], SLOT_BASE + slot);
                    assert_eq!(node[3], 1, "start from a fully READY valid seed");
                    routed.contains(&(node[2] as usize)) == matches!(head, Head::ProtectedReady)
                })
                .unwrap();
            let oldest = span
                .clone()
                .map(|slot| stamp(&record(&seed[5], SLOT_BASE + slot)))
                .min()
                .unwrap();
            assert!(oldest > prefix as u64 + 1);
            let mut node = record(&seed[5], SLOT_BASE + head_slot);
            node[..2].copy_from_slice(&pair(oldest - 1));
            if matches!(head, Head::Copying) {
                assert!(!routed.contains(&(node[2] as usize)));
                node[3] = 2;
                seed[6][node[2] as usize * 4 + 2] = 2;
            }
            put(&mut seed[5], SLOT_BASE + head_slot, node);
            let mut live: Vec<_> = span
                .clone()
                .map(|slot| {
                    let node = record(&seed[5], SLOT_BASE + slot);
                    [node[0], node[1], slot as u32, 0]
                })
                .collect();
            live.sort_unstable_by_key(|r| (stamp(r), r[2]));
            assert_eq!(live[0][2], head_slot as u32);
            let header = seed[5][(POOL_BASE + pool * 2) * 4] as usize;
            assert_eq!(seed[5][(POOL_BASE + pool * 2) * 4 + 2], NONE);
            put(
                &mut seed[5],
                header,
                [0, (prefix + SLOTS) as u32, (2 * SLOTS) as u32, 0],
            );
            for index in 0..2 * SLOTS {
                let value = if index < prefix {
                    let old = pair(index as u64 + 1);
                    [old[0], old[1], head_slot as u32, 0]
                } else if index < prefix + SLOTS {
                    live[index - prefix]
                } else {
                    [
                        TAIL_CANARY,
                        !TAIL_CANARY,
                        TAIL_CANARY ^ index as u32,
                        TAIL_CANARY,
                    ]
                };
                put(&mut seed[5], header + 1 + index, value);
            }
            // Independent seed proof: the stale prefix is well formed, every resident
            // has exactly one live log, and all live records are in exact stamp order.
            let mut seen = HashSet::new();
            let mut previous = None;
            let mut eligible = 0;
            for index in 0..prefix + SLOTS {
                let log = record(&seed[5], header + 1 + index);
                assert_eq!(log[3], 0);
                assert!(stamp(&log) != 0 && stamp(&log) <= old_clock);
                assert!(span.contains(&(log[2] as usize)));
                let slot = record(&seed[5], SLOT_BASE + log[2] as usize);
                if stamp(&log) != stamp(&slot) {
                    assert!(index < prefix);
                    continue;
                }
                assert!(matches!(slot[3], 1 | 2));
                assert!(seen.insert(log[2]));
                let key = (stamp(&log), log[2]);
                if let Some(old) = previous {
                    assert!(old < key);
                }
                previous = Some(key);
                if slot[3] == 1 && !routed.contains(&(slot[2] as usize)) {
                    eligible += 1;
                    if eligible == demand[pool] {
                        cutoffs[pool] = index;
                    }
                }
            }
            assert_eq!(seen.len(), SLOTS);
            assert!(eligible >= demand[pool]);
            assert!(cutoffs[pool] >= prefix);
            heads[pool] = prefix;
            assert!(prefix + SLOTS + USED * 3 < 2 * SLOTS, "no compaction");
        }
        Self {
            seed,
            ids,
            roles,
            heads,
            cutoffs,
        }
    }

    fn corrupt(&mut self, fault: Fault) {
        if matches!(fault, Fault::None) {
            return;
        }
        let header = self.seed[5][POOL_BASE * 4] as usize;
        let index = match fault {
            Fault::BeforeLive => self.heads[0].checked_sub(1).unwrap(),
            Fault::LiveHead => self.heads[0],
            Fault::AfterCutoff => self.cutoffs[0] + 1,
            Fault::None => unreachable!(),
        };
        assert!(index < self.seed[5][header * 4 + 1] as usize);
        if matches!(fault, Fault::AfterCutoff) {
            // Intentionally malformed duplicate of the protected/COPYING-looking
            // head AFTER the original cutoff, in the same GPU scan chunk.
            assert_eq!(index / 64, self.cutoffs[0] / 64);
            let head = record(&self.seed[5], header + 1 + self.heads[0]);
            put(&mut self.seed[5], header + 1 + index, head);
        }
        self.seed[5][(header + 1 + index) * 4 + 3] = 1;
    }

    fn expected(&self, probe: &TopkProbe) -> Raw {
        let mut routed = self.seed.clone();
        routed[0][..USED].copy_from_slice(&self.ids);
        let (mut expected, misses) = reference(
            &routed,
            &self.ids,
            self.roles,
            SLOTS,
            probe.fixture.address_base,
        );
        assert_eq!(misses, 3);
        expected[0][USED + 1] = 0;
        let requests: Vec<_> = self
            .roles
            .iter()
            .flat_map(|&base| self.ids.iter().map(move |&id| base + id as usize))
            .collect();
        let old_clock = stamp(&self.seed[5][10..12]);
        for pool in 0..2 {
            let header = self.seed[5][(POOL_BASE + pool * 2) * 4] as usize;
            let tail = self.seed[5][header * 4 + 1] as usize;
            let mut appended = 0;
            for (q, &block) in requests.iter().enumerate() {
                if requests[q + 1..].contains(&block)
                    || self.seed[4][block * 4 + 3] as usize != pool
                {
                    continue;
                }
                let value = pair(old_clock + q as u64 + 1);
                let slot = expected[6][block * 4 + 3];
                put(
                    &mut expected[5],
                    header + 1 + tail + appended,
                    [value[0], value[1], slot, 0],
                );
                appended += 1;
            }
            put(
                &mut expected[5],
                header,
                [
                    self.heads[pool] as u32,
                    (tail + appended) as u32,
                    (SLOTS * 2) as u32,
                    0,
                ],
            );
        }
        std::array::from_fn(|i| match i {
            8 => probe.expected_weights.clone(),
            9 => probe.logits_bits.clone(),
            _ => expected[i].clone(),
        })
    }
}

fn execute(be: &VulkanBackend, probe: &TopkProbe, kernel: ComputeKernel, seed: &Seed) -> Raw {
    // Reset only once per independent arm, then upload the actual dynamic queue.
    // No call to PoolProbe::assert_result: it normalizes physical FIFO tail words.
    probe.raw_fused(be, kernel, |state, _| *state = seed.clone())
}

fn check_case(
    be: &VulkanBackend,
    probe: &TopkProbe,
    kernels: &TopkKernels,
    previous: &TopkKernels,
    prefix: usize,
    head: Head,
    fault: Fault,
) {
    let mut case = PrefixCase::new(be, probe, prefix, head);
    case.corrupt(fault);
    let old = execute(be, probe, previous.fused, &case.seed);
    let new = execute(be, probe, kernels.fused, &case.seed);
    assert_eq!(
        old, new,
        "prefix={prefix} head={head:?} fault={fault:?} all10 raw"
    );
    assert_eq!(
        new[8], probe.expected_weights,
        "all routed weight bits/canaries"
    );
    assert_eq!(new[9], probe.logits_bits, "all logits/canaries");
    if matches!(fault, Fault::BeforeLive | Fault::LiveHead) {
        assert_eq!(new[2][0], 0);
        assert_eq!(new[2][1], 0);
        assert_eq!(new[2][2], 1);
        assert_eq!(new[2][3], 1);
        assert_eq!(new[3][RESIDENT_JOURNAL_STATUS_WORD], 3);
        for i in [1, 4, 5, 6, 7] {
            assert_eq!(
                new[i], case.seed[i],
                "rejected before metadata commit binding={i}"
            );
        }
    } else {
        assert_eq!(
            new,
            case.expected(probe),
            "independent complete physical FIFO/CPU LRU"
        );
        assert_eq!(new[3][RESIDENT_JOURNAL_STATUS_WORD], 4);
        for pool in 0..2 {
            let header = case.seed[5][(POOL_BASE + pool * 2) * 4] as usize;
            assert_eq!(new[5][header * 4] as usize, prefix, "exact first-live head");
        }
    }
    eprintln!("GPU_OWNER_VICTIM_PREFIX prefix={prefix} head={head:?} fault={fault:?} cutoff={:?} all10_raw=pass physical_fifo_unmasked=true cpu_oracle=true", case.cutoffs);
}

#[test]
#[ignore = "parent serial GPU; external candidate/reference, valid 0/64/512/1024 stale-prefix receipts"]
fn gpu_owned_topk_victim_prefix_heads_exact() {
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let previous = kernels.previous_fused(&be);
    let probe = TopkProbe::new(
        &be,
        &kernels,
        Case::normal("victim_prefix", USED, Workload::OneExpertMiss),
        SLOTS,
    );
    for prefix in [0, 64, 512, 1024] {
        for head in [Head::Victim, Head::ProtectedReady, Head::Copying] {
            check_case(&be, &probe, &kernels, &previous, prefix, head, Fault::None);
        }
    }
}

#[test]
#[ignore = "parent serial GPU; original malformed prefix/head rejection and post-cutoff non-rejection"]
fn gpu_owned_topk_victim_prefix_malformed_cutoff_exact() {
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let previous = kernels.previous_fused(&be);
    let probe = TopkProbe::new(
        &be,
        &kernels,
        Case::normal("victim_prefix_fault", USED, Workload::OneExpertMiss),
        SLOTS,
    );
    for prefix in [64, 512, 1024] {
        for head in [Head::ProtectedReady, Head::Copying] {
            for fault in [Fault::BeforeLive, Fault::LiveHead, Fault::AfterCutoff] {
                check_case(&be, &probe, &kernels, &previous, prefix, head, fault);
            }
        }
    }
}
