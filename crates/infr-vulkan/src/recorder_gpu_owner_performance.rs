//! Metadata-only measurements. Synthetic BDAs must never reach payload dispatches.
use super::*;
use std::collections::HashSet;

#[cfg(test)]
#[path = "recorder_gpu_owner_topk_tests.rs"]
mod topk_tests;

#[path = "recorder_gpu_owner_wider_tests.rs"]
mod wider_tests;

#[path = "recorder_gpu_owner_topk_two_row_tests.rs"]
mod topk_two_row_tests;

#[path = "recorder_gpu_owner_topk_many_row_tests.rs"]
mod topk_many_row_tests;

#[path = "recorder_gpu_owner_legacy_touch_tests.rs"]
mod legacy_touch_tests;

const NONE: u32 = u32::MAX;
const LAYERS: usize = 48;
const EXPERTS: usize = 512;
const BLOCKS: usize = LAYERS * 3 * EXPERTS;
const POOL_BASE: usize = 3;
const SLOT_BASE: usize = POOL_BASE + 4;
const SLOT_BYTES: [u32; 2] = [4096, 8192];
const CERTIFIED_DIRECTORY_PUSH_TAG: u32 = 0x4f44_4952; // ODIR, test-only shader contract.

fn certified_directory_push(push: &[u32; 31], header: &[u32; 8]) -> [u32; 31] {
    assert_eq!(
        header[0], 4,
        "directory certificate requires the reset v4 header"
    );
    let mut packed = *push;
    packed[13..18].copy_from_slice(&[
        header[1],
        header[2],
        header[3],
        header[4],
        CERTIFIED_DIRECTORY_PUSH_TAG,
    ]);
    packed
}

#[test]
fn gpu_owned_certified_directory_push_preserves_other_words() {
    for recovery in [0, 1] {
        let mut base = std::array::from_fn(|i| 0x1000 + i as u32);
        base[12] = recovery;
        let geometry = std::array::from_fn(|i| 0x2000 + i as u32);
        let original = gpu_owned_fast_push(&base, &geometry);
        let header = [4, 2, 8192, 0xfedc_ba98, 0x7654_3210, 3, 7, 8199];
        let packed = certified_directory_push(&original, &header);
        assert_eq!(
            packed[13..18],
            [2, 8192, 0xfedc_ba98, 0x7654_3210, 0x4f44_4952]
        );
        for index in 0..original.len() {
            if !(13..18).contains(&index) {
                assert_eq!(
                    packed[index], original[index],
                    "unchanged push word {index}"
                );
            }
        }
        assert_eq!(packed[12], recovery, "pool_start0 recovery is not borrowed");
    }
}

fn probe_used() -> usize {
    let used = std::env::var("MOE4ALL_OWNER_POOL_USED")
        .map(|value| value.parse().unwrap())
        .unwrap_or(10);
    assert!((1..=16).contains(&used));
    used
}

fn pool_of(block: usize) -> usize {
    usize::from(block / EXPERTS % 3 == 2)
}

fn stamp(words: &[u32]) -> u64 {
    u64::from(words[0]) | (u64::from(words[1]) << 32)
}

fn pair(value: u64) -> [u32; 2] {
    [value as u32, (value >> 32) as u32]
}

fn upload(be: &VulkanBackend, buffer: &dyn Buffer, words: &[u32]) {
    be.upload(buffer, bytemuck::cast_slice(words)).unwrap();
}

fn read(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u32> {
    let mut words = vec![0; buffer.len_bytes() / 4];
    be.download(buffer, bytemuck::cast_slice_mut(&mut words))
        .unwrap();
    words
}

#[derive(Clone, Copy, Debug)]
enum Workload {
    AllHit,
    OneExpertMiss,
    OneExpertStale,
    MixedFresh,
    MixedStale,
    AliasedStale,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Self::AllHit => "all_hit",
            Self::OneExpertMiss => "one_expert_miss",
            Self::OneExpertStale => "one_expert_stale",
            Self::MixedFresh => "mixed_fresh",
            Self::MixedStale => "mixed_stale",
            Self::AliasedStale => "aliased_stale",
        }
    }

    fn stale(self) -> bool {
        matches!(
            self,
            Self::MixedStale | Self::AliasedStale | Self::OneExpertStale
        )
    }
}

struct PoolProbe {
    buffers: Vec<Box<dyn Buffer>>,
    initial: [Vec<u32>; 8],
    expected: [Vec<u32>; 8],
    slots_per_pool: usize,
    address_base: usize,
    heap_base: usize,
    push: [u32; 19],
    misses: usize,
    repeats: usize,
}

impl PoolProbe {
    fn new(be: &VulkanBackend, slots_per_pool: usize, count: usize, workload: Workload) -> Self {
        let total = slots_per_pool * 2;
        let address_base = SLOT_BASE + total;
        let heap_base = address_base + total;
        let roles = [0, 1, 2].map(|role| ((LAYERS - 1) * 3 + role) * EXPERTS);
        let ids: Vec<_> = (0..count)
            .map(|i| {
                if matches!(workload, Workload::AliasedStale) {
                    (i % (count / 2)) as u32
                } else {
                    i as u32
                }
            })
            .collect();
        let requested: HashSet<_> = roles
            .iter()
            .flat_map(|&base| ids.iter().map(move |&id| base + id as usize))
            .collect();
        let mut nodes = vec![[0u32; 4]; heap_base + total];
        let clock = (1u64 << 32) - 10;
        nodes[0] = [2, 2, total as u32, 41];
        nodes[1] = [7, POOL_BASE as u32, SLOT_BASE as u32, address_base as u32];
        nodes[2] = [17, 23, pair(clock)[0], pair(clock)[1]];
        let mut sources = vec![[0u32; 4]; BLOCKS];
        let mut overlay = vec![[0, 0, 0, NONE]; BLOCKS];
        for (block, source) in sources.iter_mut().enumerate() {
            let address = pair(0x20_0000_0000 + block as u64 * 16384);
            *source = [
                address[0],
                address[1],
                SLOT_BYTES[pool_of(block)],
                pool_of(block) as u32,
            ];
        }
        for pool in 0..2 {
            let start = pool * slots_per_pool;
            let header = POOL_BASE + pool * 2;
            nodes[header] = [
                (heap_base + start) as u32,
                slots_per_pool as u32,
                NONE,
                slots_per_pool as u32,
            ];
            nodes[header + 1] = [11, 7, 5, 0x1234];
            let mut residents: Vec<_> = requested
                .iter()
                .copied()
                .filter(|&block| {
                    pool_of(block) == pool
                        && (matches!(workload, Workload::AllHit)
                            || if matches!(
                                workload,
                                Workload::OneExpertMiss | Workload::OneExpertStale
                            ) {
                                block % EXPERTS != 0
                            } else {
                                block % EXPERTS % 2 == 0
                            })
                })
                .collect();
            residents.sort_unstable();
            residents.extend(
                (0..BLOCKS)
                    .filter(|block| pool_of(*block) == pool && !requested.contains(block))
                    .take(slots_per_pool - residents.len()),
            );
            assert_eq!(residents.len(), slots_per_pool);
            for (local, block) in residents.into_iter().enumerate() {
                let slot = start + local;
                let live = clock - total as u64 + slot as u64 + 1;
                let cached = if workload.stale() {
                    slot as u64 + 1
                } else {
                    live
                };
                let address = pair(0x10_0000_0000 + slot as u64 * 16384);
                let pool_requests = requested
                    .iter()
                    .filter(|&&block| pool_of(block) == pool)
                    .count();
                let state = if workload.stale()
                    && local + 1 == slots_per_pool
                    && slots_per_pool > pool_requests
                {
                    2
                } else {
                    1
                };
                nodes[SLOT_BASE + slot] = [pair(live)[0], pair(live)[1], block as u32, state];
                nodes[address_base + slot] = [address[0], address[1], pool as u32, local as u32];
                nodes[heap_base + slot] = [pair(cached)[0], pair(cached)[1], slot as u32, 0];
                overlay[block] = [address[0], address[1], state, slot as u32];
            }
        }
        let used = probe_used();
        assert!(count.is_multiple_of(used));
        let rows = count / used;
        assert!((1..=infr_core::backend::MAX_DECODE_ROWS).contains(&rows));
        let mut initial: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        initial[0] = vec![0; count + rows * 2];
        initial[0][..count].copy_from_slice(&ids);
        initial[1] = vec![0xdead_beef; 3 * EXPERTS * 2];
        initial[2] = vec![0; RESIDENT_CONTROL_BYTES / 4];
        initial[2][0] = 1;
        initial[2][1] = NONE;
        initial[3] = vec![0; RESIDENT_JOURNAL_WORDS];
        initial[4] = bytemuck::cast_slice(&sources).to_vec();
        initial[5] = bytemuck::cast_slice(&nodes).to_vec();
        initial[6] = bytemuck::cast_slice(&overlay).to_vec();
        initial[7] = vec![0; RESIDENT_JOURNAL_ID_WORDS * 3 * 8];
        let (mut expected, misses) = reference(&initial, &ids, roles, slots_per_pool, address_base);
        let repeats = if matches!(workload, Workload::AllHit)
            && std::env::var("MOE4ALL_OWNER_POOL_WARM_HITS").as_deref() == Ok("1")
        {
            64
        } else {
            1
        };
        for _ in 1..repeats {
            (expected, _) = reference(&expected, &ids, roles, slots_per_pool, address_base);
        }
        let reserve_fifo = std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_SPV").is_some()
            || std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_PRESELECTED_SPV").is_some()
            || certified_probe();
        let buffers = initial
            .iter()
            .enumerate()
            .map(|(binding, words)| {
                let bytes = if binding == 5 && reserve_fifo {
                    (heap_base + 2 + total * 2) * 16
                } else {
                    words.len() * 4
                };
                be.alloc_mapped_arena_bda(bytes).unwrap().0
            })
            .collect();
        let push = gpu_copy_prepare_push(
            [
                count as u32,
                EXPERTS as u32,
                0,
                EXPERTS as u32,
                (2 * EXPERTS) as u32,
                0,
                used as u32,
                0,
            ],
            true,
            [
                roles[0] as u32,
                roles[1] as u32,
                roles[2] as u32,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            2,
        );
        Self {
            buffers,
            initial,
            expected,
            slots_per_pool,
            address_base,
            heap_base,
            push,
            misses,
            repeats,
        }
    }

    fn reset(&self, be: &VulkanBackend, list: bool, fifo: bool) {
        let mut initial = self.initial.clone();
        if fifo {
            let mut nodes = self.initial[5][..self.heap_base * 4].to_vec();
            nodes[0] = 4;
            for pool in 0..2 {
                let start = pool * self.slots_per_pool;
                let header = nodes.len() / 4;
                nodes[(POOL_BASE + pool * 2) * 4] = header as u32;
                nodes.extend_from_slice(&[
                    0,
                    self.slots_per_pool as u32,
                    (2 * self.slots_per_pool) as u32,
                    0,
                ]);
                let mut order: Vec<_> = (start..start + self.slots_per_pool).collect();
                order.sort_unstable_by_key(|&slot| (stamp(&nodes[(SLOT_BASE + slot) * 4..]), slot));
                for slot in order {
                    let value = pair(stamp(&nodes[(SLOT_BASE + slot) * 4..]));
                    nodes.extend_from_slice(&[value[0], value[1], slot as u32, 0]);
                }
                nodes.resize((header + 1 + self.slots_per_pool * 2) * 4, 0);
            }
            initial[5] = nodes;
        }
        if list {
            let nodes = &mut initial[5];
            nodes[0] = 3;
            for pool in 0..2 {
                let start = pool * self.slots_per_pool;
                let mut order: Vec<_> = (start..start + self.slots_per_pool).collect();
                order.sort_unstable_by_key(|&slot| (stamp(&nodes[(SLOT_BASE + slot) * 4..]), slot));
                nodes[(POOL_BASE + pool * 2 + 1) * 4 + 3] = order[0] as u32;
                for (index, &slot) in order.iter().enumerate() {
                    let previous = order[(index + order.len() - 1) % order.len()];
                    let next = order[(index + 1) % order.len()];
                    nodes[(self.heap_base + slot) * 4..(self.heap_base + slot + 1) * 4]
                        .copy_from_slice(&[previous as u32, next as u32, 0, 0]);
                }
            }
        }
        for (buffer, words) in self.buffers.iter().zip(&initial) {
            upload(be, buffer.as_ref(), words);
        }
    }

    fn assert_result(
        &self,
        be: &VulkanBackend,
        list: bool,
        fifo: bool,
        skip_hit_updates: bool,
        cache_snapshot: bool,
    ) {
        for index in [0, 1, 2, 3, 4, 6, 7] {
            let mut expected = self.expected[index].clone();
            let count = self.push[0] as usize;
            let requests = count * 3;
            if index == 7
                && cache_snapshot
                && self.misses != 0
                && requests * 2 <= RESIDENT_JOURNAL_ID_WORDS * 3
            {
                // The declared upper-half workspace is checked exactly too, not masked away.
                // Real jobs, unused middle/tail and every other buffer retain full comparison.
                let base = RESIDENT_JOURNAL_ID_WORDS * 3 * 4;
                for q in 0..requests {
                    let block =
                        self.push[9 + q / count] as usize + self.initial[0][q % count] as usize;
                    let overlay = &self.initial[6][block * 4..block * 4 + 4];
                    let source = &self.initial[4][block * 4..block * 4 + 4];
                    let valid = overlay[2] == 1;
                    let start = base + q * 8;
                    expected[start..start + 4].copy_from_slice(if valid {
                        overlay
                    } else {
                        &[0; 4]
                    });
                    expected[start + 4..start + 8].copy_from_slice(&[
                        u32::from(valid),
                        source[3],
                        source[2],
                        block as u32,
                    ]);
                }
            }
            assert_eq!(
                read(be, self.buffers[index].as_ref()),
                expected,
                "binding {index}"
            );
        }
        let mut nodes = read(be, self.buffers[5].as_ref());
        let expected_nodes = if skip_hit_updates {
            assert_eq!(self.misses, 0, "maintenance attribution is all-hit only");
            &self.initial[5]
        } else {
            &self.expected[5]
        };
        if fifo {
            assert_eq!(nodes[0], 4);
            let mut header = self.heap_base;
            for pool in 0..2 {
                let start = pool * self.slots_per_pool;
                assert_eq!(nodes[(POOL_BASE + pool * 2) * 4] as usize, header);
                let queue = &nodes[header * 4..header * 4 + 4];
                let (head, tail, capacity) =
                    (queue[0] as usize, queue[1] as usize, queue[2] as usize);
                assert_eq!(capacity, self.slots_per_pool * 2);
                assert!(head <= tail && tail <= capacity);
                assert_eq!(queue[3], 0);
                let mut live_order = Vec::new();
                let mut seen = HashSet::new();
                let mut previous = None;
                for index in head..tail {
                    let record = &nodes[(header + 1 + index) * 4..(header + 2 + index) * 4];
                    let slot = record[2] as usize;
                    assert!((start..start + self.slots_per_pool).contains(&slot));
                    assert_eq!(record[3], 0);
                    let live = &nodes[(SLOT_BASE + slot) * 4..(SLOT_BASE + slot + 1) * 4];
                    if !matches!(live[3], 1 | 2) || stamp(live) != stamp(record) {
                        continue;
                    }
                    let key = (stamp(record), slot);
                    if let Some(previous) = previous {
                        assert!(previous < key, "FIFO chronological order");
                    }
                    previous = Some(key);
                    assert!(seen.insert(slot), "one live FIFO entry per slot");
                    live_order.push(slot);
                }
                let mut expected_order: Vec<_> = (start..start + self.slots_per_pool)
                    .filter(|&slot| matches!(nodes[(SLOT_BASE + slot) * 4 + 3], 1 | 2))
                    .collect();
                expected_order
                    .sort_unstable_by_key(|&slot| (stamp(&nodes[(SLOT_BASE + slot) * 4..]), slot));
                assert_eq!(
                    live_order, expected_order,
                    "FIFO covers exact READY/COPYING LRU order"
                );
                nodes[(POOL_BASE + pool * 2) * 4] = self.initial[5][(POOL_BASE + pool * 2) * 4];
                header += 1 + capacity;
            }
            assert_eq!(header * 4, nodes.len());
            nodes[0] = 2;
        }
        if list {
            assert_eq!(nodes[0], 3);
            for pool in 0..2 {
                let start = pool * self.slots_per_pool;
                let stats = (POOL_BASE + pool * 2 + 1) * 4;
                let mut order: Vec<_> = (start..start + self.slots_per_pool).collect();
                order.sort_unstable_by_key(|&slot| (stamp(&nodes[(SLOT_BASE + slot) * 4..]), slot));
                assert_eq!(nodes[stats + 3] as usize, order[0], "exact LRU head");
                for (index, &slot) in order.iter().enumerate() {
                    let links =
                        &nodes[(self.heap_base + slot) * 4..(self.heap_base + slot + 1) * 4];
                    assert_eq!(
                        links,
                        &[
                            order[(index + order.len() - 1) % order.len()] as u32,
                            order[(index + 1) % order.len()] as u32,
                            0,
                            0
                        ],
                        "circular LRU links for slot {slot}"
                    );
                }
                // Normalize representation only; owners, timestamps, masks, jobs and statistics
                // still compare against the independent CPU replacement reference below.
                nodes[stats + 3] = self.initial[5][stats + 3];
                for (index, &slot) in order.iter().enumerate() {
                    let value = stamp(&nodes[(SLOT_BASE + slot) * 4..]);
                    let key = pair(value);
                    nodes[(self.heap_base + start + index) * 4
                        ..(self.heap_base + start + index + 1) * 4]
                        .copy_from_slice(&[key[0], key[1], slot as u32, 0]);
                }
            }
            nodes[0] = 2;
        }
        assert_eq!(
            &nodes[..self.heap_base * 4],
            &expected_nodes[..self.heap_base * 4],
            "headers, statistics, clock, exact owners/stamps/states, canonical addresses"
        );
        if fifo {
            return;
        }
        let clock = stamp(&nodes[10..12]);
        for pool in 0..2 {
            let start = pool * self.slots_per_pool;
            let base = self.heap_base + start;
            let mut seen = vec![false; self.slots_per_pool];
            for index in 0..self.slots_per_pool {
                let entry = &nodes[(base + index) * 4..(base + index + 1) * 4];
                let slot = entry[2] as usize;
                assert!((start..start + self.slots_per_pool).contains(&slot));
                assert!(
                    !std::mem::replace(&mut seen[slot - start], true),
                    "duplicate heap slot"
                );
                let live = &nodes[(SLOT_BASE + slot) * 4..(SLOT_BASE + slot + 1) * 4];
                assert!(matches!(live[3], 1 | 2));
                assert!(stamp(entry) > 0 && stamp(entry) <= stamp(live) && stamp(live) <= clock);
                assert_eq!(entry[3], 0);
                if index != 0 {
                    let parent = &nodes[(base + (index - 1) / 2) * 4..];
                    assert!(
                        (stamp(parent), parent[2]) <= (stamp(entry), entry[2]),
                        "heap order"
                    );
                }
                let canonical = &nodes[(self.address_base + slot) * 4..];
                assert_eq!(&canonical[2..4], &[pool as u32, (slot - start) as u32]);
            }
            assert!(seen.into_iter().all(|value| value));
        }
    }
}

// Independent exact LRU: choose the minimum live stamp, without reading any heap key.
fn reference(
    initial: &[Vec<u32>; 8],
    ids: &[u32],
    roles: [usize; 3],
    slots_per_pool: usize,
    address_base: usize,
) -> ([Vec<u32>; 8], usize) {
    let mut expected = initial.clone();
    let mut nodes = initial[5].clone();
    let mut overlay = initial[6].clone();
    let requests: Vec<_> = roles
        .iter()
        .flat_map(|&base| ids.iter().map(move |&id| base + id as usize))
        .collect();
    let protected: HashSet<_> = requests.iter().copied().collect();
    let old_clock = stamp(&nodes[10..12]);
    let last_stamp =
        |block| old_clock + requests.iter().rposition(|&b| b == block).unwrap() as u64 + 1;
    for &block in &protected {
        if initial[6][block * 4 + 2] == 1 {
            let slot = initial[6][block * 4 + 3] as usize;
            nodes[(SLOT_BASE + slot) * 4..(SLOT_BASE + slot) * 4 + 2]
                .copy_from_slice(&pair(last_stamp(block)));
        }
    }
    let mut touches = [0u32; 2];
    let mut misses = [0u32; 2];
    let mut admitted = HashSet::new();
    let mut job_count = 0;
    for (q, &block) in requests.iter().enumerate() {
        let pool = pool_of(block);
        touches[pool] += 1;
        if initial[6][block * 4 + 2] == 1 || !admitted.insert(block) {
            continue;
        }
        let slot = (pool * slots_per_pool..(pool + 1) * slots_per_pool)
            .filter(|&slot| {
                let n = (SLOT_BASE + slot) * 4;
                nodes[n + 3] == 1 && !protected.contains(&(nodes[n + 2] as usize))
            })
            .min_by_key(|&slot| (stamp(&nodes[(SLOT_BASE + slot) * 4..]), slot))
            .unwrap();
        let n = (SLOT_BASE + slot) * 4;
        let evicted = nodes[n + 2] as usize;
        overlay[evicted * 4..evicted * 4 + 4].copy_from_slice(&[0, 0, 0, NONE]);
        let a = (address_base + slot) * 4;
        let destination = [nodes[a], nodes[a + 1]];
        nodes[n..n + 4].copy_from_slice(&[
            pair(last_stamp(block))[0],
            pair(last_stamp(block))[1],
            block as u32,
            2,
        ]);
        overlay[block * 4..block * 4 + 4].copy_from_slice(&[
            destination[0],
            destination[1],
            2,
            slot as u32,
        ]);
        let s = block * 4;
        let role = q / ids.len();
        let job = [
            initial[4][s],
            initial[4][s + 1],
            destination[0],
            destination[1],
            initial[4][s + 2],
            block as u32,
            slot as u32,
            (role * EXPERTS + ids[q % ids.len()] as usize) as u32,
        ];
        expected[7][job_count * 8..job_count * 8 + 8].copy_from_slice(&job);
        job_count += 1;
        misses[pool] += 1;
    }
    for (q, &block) in requests.iter().enumerate() {
        let tape = (q / ids.len() * EXPERTS + ids[q % ids.len()] as usize) * 2;
        expected[1][tape..tape + 2].copy_from_slice(&overlay[block * 4..block * 4 + 2]);
    }
    for pool in 0..2 {
        let stats = (POOL_BASE + pool * 2 + 1) * 4;
        nodes[stats] += touches[pool] - misses[pool];
        nodes[stats + 1] += misses[pool];
        nodes[stats + 2] += misses[pool];
    }
    nodes[8] += requests.len() as u32 - job_count as u32;
    nodes[9] += job_count as u32;
    nodes[10..12].copy_from_slice(&pair(old_clock + requests.len() as u64));
    expected[2][2] = 1;
    expected[2][4] = 1;
    expected[2][5..7].copy_from_slice(&[1, 1]);
    expected[3][..ids.len()].copy_from_slice(ids);
    let used = probe_used();
    let rows = ids.len() / used;
    for row in 0..rows {
        let mut hits = 0;
        for lane in 0..used {
            let id = ids[row * used + lane] as usize;
            if roles
                .iter()
                .all(|&base| initial[6][(base + id) * 4 + 2] == 1)
            {
                hits |= 1 << lane;
            }
        }
        expected[3][RESIDENT_JOURNAL_MASK_WORD + row] = hits;
        expected[0][ids.len() + row] = (1 << used) - 1;
    }
    expected[3][RESIDENT_JOURNAL_JOB_COUNT_WORD..RESIDENT_JOURNAL_JOB_COUNT_WORD + 3]
        .copy_from_slice(&[job_count as u32, 41, 7]);
    expected[3][RESIDENT_JOURNAL_STATUS_WORD] = if job_count == 0 { 1 } else { 4 };
    expected[5] = nodes;
    expected[6] = overlay;
    (expected, job_count)
}

struct ProbeShader {
    name: &'static str,
    words: Vec<u32>,
    list: bool,
    fifo: bool,
    skip_hit_updates: bool,
    preselected: bool,
}

impl ProbeShader {
    fn certified_push(&self) -> bool {
        matches!(
            self.name,
            "gpu_owned_pool_certified" | "gpu_owned_pool_embedded_certified"
        )
    }

    fn cold_only(&self) -> bool {
        matches!(
            self.name,
            "gpu_owned_pool_fifo_cold" | "gpu_owned_pool_fifo_cold_preselected"
        )
    }
}

fn certified_probe() -> bool {
    std::env::var("MOE4ALL_OWNER_PREPARE_CERTIFIED").as_deref() == Ok("1")
}

fn single_prepare_dispatch(name: &str, candidate: bool, compare: bool) -> bool {
    match name {
        "gpu_owned_pool_certified" => candidate,
        "gpu_owned_pool_embedded_certified" => compare,
        _ => false,
    }
}

#[test]
fn gpu_owned_monolithic_dispatch_flags_are_independent() {
    for candidate in [false, true] {
        for compare in [false, true] {
            assert_eq!(
                single_prepare_dispatch("gpu_owned_pool_certified", candidate, compare),
                candidate
            );
            assert_eq!(
                single_prepare_dispatch("gpu_owned_pool_embedded_certified", candidate, compare),
                compare
            );
            for name in ["gpu_owned_pool_current", "gpu_owned_pool_fifo"] {
                assert!(!single_prepare_dispatch(name, candidate, compare));
            }
        }
    }
}

fn shaders() -> Vec<ProbeShader> {
    let mut shaders = vec![ProbeShader {
        name: "gpu_owned_pool_current",
        words: crate::gemm::gpu_owned_prepare_reference_spv().to_vec(),
        list: false,
        fifo: false,
        skip_hit_updates: false,
        preselected: false,
    }];
    if certified_probe() {
        let words = std::env::var_os("MOE4ALL_OWNER_PREPARE_CERTIFIED_SPV")
            .map(|path| {
                eprintln!(
                    "GPU_OWNER_POOL_CERTIFIED_SHADER path={}",
                    std::path::Path::new(&path).display()
                );
                let bytes = std::fs::read(path).unwrap();
                ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap()
            })
            .unwrap_or_else(|| crate::gemm::gpu_owned_prepare_fast_spv(false).to_vec());
        shaders.push(ProbeShader {
            name: "gpu_owned_pool_certified",
            words,
            list: false,
            fifo: true,
            skip_hit_updates: false,
            preselected: false,
        });
    }
    if std::env::var("MOE4ALL_OWNER_COMPARE_EMBEDDED_CERTIFIED").as_deref() == Ok("1") {
        let words = std::env::var_os("MOE4ALL_OWNER_COMPARE_FAST_SPV")
            .map(|path| {
                eprintln!(
                    "GPU_OWNER_POOL_COMPARE_FAST_SHADER push_bytes=124 path={}",
                    std::path::Path::new(&path).display()
                );
                let bytes = std::fs::read(path).unwrap();
                ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap()
            })
            .unwrap_or_else(|| crate::gemm::gpu_owned_prepare_fast_spv(false).to_vec());
        shaders.push(ProbeShader {
            name: "gpu_owned_pool_embedded_certified",
            words,
            list: false,
            fifo: true,
            skip_hit_updates: false,
            preselected: false,
        });
    }
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("MOE4ALL_OWNER_PREPARE_V2_SPV") {
        paths.push(std::path::PathBuf::from(path));
    }
    if let Some(list) = std::env::var_os("MOE4ALL_OWNER_PREPARE_V2_SPVS") {
        paths.extend(std::env::split_paths(&list));
    }
    let names = [
        "gpu_owned_pool_external0",
        "gpu_owned_pool_external1",
        "gpu_owned_pool_external2",
        "gpu_owned_pool_external3",
        "gpu_owned_pool_external4",
        "gpu_owned_pool_external5",
        "gpu_owned_pool_external6",
        "gpu_owned_pool_external7",
    ];
    assert!(
        paths.len() <= names.len(),
        "at most eight external variants"
    );
    for (name, path) in names.into_iter().zip(paths) {
        eprintln!(
            "GPU_OWNER_POOL_SHADER variant={name} path={}",
            path.display()
        );
        let bytes = std::fs::read(path).unwrap();
        shaders.push(ProbeShader {
            name,
            words: ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap(),
            list: false,
            fifo: false,
            skip_hit_updates: false,
            preselected: false,
        });
    }
    if let Some(path) = std::env::var_os("MOE4ALL_OWNER_PREPARE_LIST_SPV") {
        let bytes = std::fs::read(&path).unwrap();
        eprintln!(
            "GPU_OWNER_POOL_SHADER variant=gpu_owned_pool_list path={}",
            std::path::Path::new(&path).display()
        );
        shaders.push(ProbeShader {
            name: "gpu_owned_pool_list",
            words: ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap(),
            list: true,
            fifo: false,
            skip_hit_updates: false,
            preselected: false,
        });
    }
    if let Some(path) = std::env::var_os("MOE4ALL_OWNER_PREPARE_NO_HIT_LRU_SPV") {
        let bytes = std::fs::read(&path).unwrap();
        eprintln!(
            "GPU_OWNER_POOL_SHADER variant=gpu_owned_pool_no_hit_lru path={}",
            std::path::Path::new(&path).display()
        );
        shaders.push(ProbeShader {
            name: "gpu_owned_pool_no_hit_lru",
            words: ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap(),
            list: false,
            fifo: false,
            skip_hit_updates: true,
            preselected: false,
        });
    }
    if let Some(path) = std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_SPV") {
        let bytes = std::fs::read(&path).unwrap();
        eprintln!(
            "GPU_OWNER_POOL_SHADER variant=gpu_owned_pool_fifo path={}",
            std::path::Path::new(&path).display()
        );
        shaders.push(ProbeShader {
            name: "gpu_owned_pool_fifo",
            words: ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap(),
            list: false,
            fifo: true,
            skip_hit_updates: false,
            preselected: false,
        });
    }
    if let Some(path) = std::env::var_os("MOE4ALL_OWNER_PREPARE_PRESELECTED_SPV") {
        let bytes = std::fs::read(&path).unwrap();
        eprintln!(
            "GPU_OWNER_POOL_SHADER variant=gpu_owned_pool_preselected path={}",
            std::path::Path::new(&path).display()
        );
        shaders.push(ProbeShader {
            name: "gpu_owned_pool_preselected",
            words: ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap(),
            list: false,
            fifo: false,
            skip_hit_updates: false,
            preselected: true,
        });
    }
    if let Some(path) = std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_PRESELECTED_SPV") {
        // Both arms start with identical reference jobs and status6, outside timestamps.
        // This diagnoses selector sensitivity; it is not an alternate residency protocol.
        let baseline = std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_COLD_SPV")
            .map(|path| {
                eprintln!(
                    "GPU_OWNER_POOL_COLD_BASELINE path={}",
                    std::path::Path::new(&path).display()
                );
                let bytes = std::fs::read(path).unwrap();
                ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap()
            })
            .unwrap_or_else(|| crate::gemm::gpu_owned_prepare_slow_spv(false).to_vec());
        eprintln!(
            "GPU_OWNER_POOL_COLD_PRESELECTED path={}",
            std::path::Path::new(&path).display()
        );
        let bytes = std::fs::read(path).unwrap();
        let preselected = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
        for (name, words, preselected) in [
            ("gpu_owned_pool_fifo_cold", baseline, false),
            ("gpu_owned_pool_fifo_cold_preselected", preselected, true),
        ] {
            shaders.push(ProbeShader {
                name,
                words,
                list: false,
                fifo: true,
                skip_hit_updates: false,
                preselected,
            });
        }
    }
    if std::env::var("MOE4ALL_OWNER_PREPARE_REVERSE").as_deref() == Ok("1") {
        shaders.reverse();
    }
    shaders
}

struct Queries<'a> {
    be: &'a VulkanBackend,
    pool: vk::QueryPool,
}

impl Drop for Queries<'_> {
    fn drop(&mut self) {
        unsafe { self.be.shared.device.destroy_query_pool(self.pool, None) };
    }
}

#[test]
#[ignore = "central GPU runner only; realistic pools, synthetic BDAs, prepare-only exact LRU"]
fn gpu_owned_realistic_pool_prepare_latency_probe() {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicitly select an idle device"));
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let variants = shaders();
    let candidate_monolithic =
        std::env::var("MOE4ALL_OWNER_PREPARE_MONOLITHIC").as_deref() == Ok("1");
    let compare_monolithic =
        std::env::var("MOE4ALL_OWNER_COMPARE_MONOLITHIC").as_deref() == Ok("1");
    assert!(
        !compare_monolithic
            || (variants
                .iter()
                .any(|variant| variant.name == "gpu_owned_pool_embedded_certified")
                && std::env::var_os("MOE4ALL_OWNER_COMPARE_FAST_SPV").is_some()),
        "compare monolithic requires the certified comparator and explicit COMPARE_FAST_SPV"
    );
    eprintln!(
        "GPU_OWNER_POOL_MONOLITHIC candidate={candidate_monolithic} compare={compare_monolithic}"
    );
    let directory_push =
        std::env::var("MOE4ALL_OWNER_CERTIFIED_DIRECTORY_PUSH").as_deref() == Ok("1");
    assert!(
        !directory_push
            || (certified_probe()
                && std::env::var_os("MOE4ALL_OWNER_PREPARE_CERTIFIED_SPV").is_some()),
        "directory push requires the explicit external certified fast124 candidate"
    );
    let fifo_slow_124 = std::env::var("MOE4ALL_OWNER_PREPARE_FIFO_SLOW_124").as_deref() == Ok("1");
    assert!(
        !fifo_slow_124 || std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_SLOW_SPV").is_some(),
        "slow124 requires an explicit external FIFO_SLOW_SPV"
    );
    let fifo_slow_external = std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_SLOW_SPV");
    let fifo_slow_native = fifo_slow_external.is_some()
        && std::env::var("MOE4ALL_OWNER_PREPARE_SLOW_NATIVE").as_deref() == Ok("1");
    let fifo_slow = fifo_slow_external
        .map(|path| {
            let bytes = std::fs::read(path).unwrap();
            ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap()
        })
        .or_else(|| {
            certified_probe().then(|| crate::gemm::gpu_owned_prepare_slow_spv(false).to_vec())
        })
        .map(|words| {
            let make_kernel = if fifo_slow_native {
                VulkanBackend::kernel_native
            } else {
                VulkanBackend::kernel
            };
            make_kernel(
                &be,
                "gpu_owned_pool_fifo_slow",
                &words,
                8,
                if fifo_slow_124 {
                    GPU_OWNED_FAST_PUSH_BYTES
                } else {
                    GPU_COPY_PREPARE_PUSH_BYTES
                },
            )
        });
    let rounds = std::env::var("MOE4ALL_OWNER_POOL_ROUNDS")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(8);
    assert!((4..=64).contains(&rounds));
    let compare_slow_words = std::env::var_os("MOE4ALL_OWNER_COMPARE_SLOW_SPV")
        .map(|path| {
            eprintln!(
                "GPU_OWNER_POOL_COMPARE_SLOW_SHADER push_bytes=76 path={}",
                std::path::Path::new(&path).display()
            );
            let bytes = std::fs::read(path).unwrap();
            ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap()
        })
        .unwrap_or_else(|| crate::gemm::gpu_owned_prepare_slow_spv(false).to_vec());
    let embedded_slow = be.kernel(
        "gpu_owned_pool_embedded_certified_slow",
        &compare_slow_words,
        8,
        GPU_COPY_PREPARE_PUSH_BYTES,
    );
    assert!(
        be.shared.submit_timestamp_valid_bits > 0,
        "queue timestamps required"
    );
    let device = &be.shared.device;
    let queries = Queries {
        be: &be,
        pool: unsafe {
            device.create_query_pool(
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
    let smoke = std::env::var("MOE4ALL_OWNER_POOL_SMOKE").as_deref() == Ok("1");
    let used = probe_used();
    let wide = std::env::var("MOE4ALL_OWNER_POOL_WIDE").as_deref() == Ok("1");
    let counts = if wide {
        vec![
            used,
            used * 2,
            used * 4,
            used * infr_core::backend::MAX_DECODE_ROWS,
        ]
    } else {
        vec![used, used * 2]
    };
    for count in counts {
        for slots in [count * 2, 4096, 16384] {
            if (smoke && !wide && count != used) || (smoke && slots == 16384) {
                continue;
            }
            let workloads = if std::env::var("MOE4ALL_OWNER_POOL_ONE_MISS").as_deref() == Ok("1") {
                vec![
                    Workload::AllHit,
                    Workload::OneExpertMiss,
                    Workload::OneExpertStale,
                ]
            } else {
                vec![
                    Workload::AllHit,
                    Workload::MixedFresh,
                    Workload::MixedStale,
                    Workload::AliasedStale,
                ]
            };
            for workload in workloads {
                let fixture = PoolProbe::new(&be, slots, count, workload);
                let bindings: Vec<_> = fixture
                    .buffers
                    .iter()
                    .map(|buffer| Recorder::vkb(buffer.as_ref()))
                    .collect();
                let kernels: Vec<_> = variants
                    .iter()
                    .map(|variant| {
                        let push_bytes = if variant.certified_push() {
                            GPU_OWNED_FAST_PUSH_BYTES
                        } else {
                            GPU_COPY_PREPARE_PUSH_BYTES
                        };
                        be.kernel(variant.name, &variant.words, 8, push_bytes)
                    })
                    .collect();
                let geometry = [
                    0,
                    SLOT_BYTES[0],
                    0,
                    slots as u32,
                    0,
                    SLOT_BYTES[0],
                    0,
                    slots as u32,
                    1,
                    SLOT_BYTES[1],
                    slots as u32,
                    slots as u32,
                ];
                let fast_push = gpu_owned_fast_push(&fixture.push, &geometry);
                let mut samples = vec![Vec::new(); variants.len()];
                for round in 0..rounds {
                    // Rotate variant order and reset outside timestamps: every sample sees the same misses.
                    for offset in 0..variants.len() {
                        let variant = (round + offset) % variants.len();
                        if variants[variant].skip_hit_updates
                            && !matches!(workload, Workload::AllHit)
                        {
                            continue;
                        }
                        if variants[variant].cold_only() && matches!(workload, Workload::AllHit) {
                            // Cold miss attribution only; warm-hit repeats need a fresh status6.
                            continue;
                        }
                        fixture.reset(&be, variants[variant].list, variants[variant].fifo);
                        // Candidate-only certificate from the actual reset v4 storage,
                        // outside timestamps. Comparison and slow arms keep the old push.
                        let candidate_fast_push = if directory_push
                            && variants[variant].name == "gpu_owned_pool_certified"
                        {
                            let mut header = [0u32; 8];
                            be.download(
                                fixture.buffers[5].as_ref(),
                                bytemuck::cast_slice_mut(&mut header),
                            )
                            .unwrap();
                            certified_directory_push(&fast_push, &header)
                        } else {
                            fast_push
                        };
                        if variants[variant].preselected || variants[variant].cold_only() {
                            upload(&be, fixture.buffers[7].as_ref(), &fixture.expected[7]);
                        }
                        if variants[variant].cold_only() {
                            let mut journal = fixture.initial[3].clone();
                            journal[RESIDENT_JOURNAL_STATUS_WORD] = 6;
                            upload(&be, fixture.buffers[3].as_ref(), &journal);
                        }
                        let rec = be.recorder().unwrap();
                        rec.owned_copy_barrier(true, false);
                        unsafe {
                            device.cmd_reset_query_pool(rec.cmd, queries.pool, 0, 2);
                            device.cmd_write_timestamp(
                                rec.cmd,
                                vk::PipelineStageFlags::TOP_OF_PIPE,
                                queries.pool,
                                0,
                            );
                        }
                        // Deliberately dispatch only prepare. No fake BDA is ever dereferenced.
                        for iteration in 0..fixture.repeats {
                            if iteration != 0 {
                                rec.owned_copy_barrier(false, false);
                            }
                            let push = if variants[variant].certified_push() {
                                bytemuck::cast_slice(&candidate_fast_push)
                            } else {
                                bytemuck::cast_slice(&fixture.push)
                            };
                            rec.dispatch(kernels[variant], &bindings, 8, push, 1);
                            let single_dispatch = single_prepare_dispatch(
                                variants[variant].name,
                                candidate_monolithic,
                                compare_monolithic,
                            );
                            if variants[variant].fifo
                                && !variants[variant].cold_only()
                                && !single_dispatch
                            {
                                if let Some(kernel) = if variants[variant].name
                                    == "gpu_owned_pool_embedded_certified"
                                {
                                    Some(embedded_slow)
                                } else {
                                    fifo_slow
                                } {
                                    rec.owned_copy_barrier(false, false);
                                    // The embedded baseline always uses the original slow76 ABI.
                                    let slow_push = if fifo_slow_124
                                        && variants[variant].name
                                            != "gpu_owned_pool_embedded_certified"
                                    {
                                        bytemuck::cast_slice(&fast_push)
                                    } else {
                                        bytemuck::cast_slice(&fixture.push)
                                    };
                                    rec.dispatch(kernel, &bindings, 8, slow_push, 1);
                                }
                            }
                        }
                        unsafe {
                            device.cmd_write_timestamp(
                                rec.cmd,
                                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                queries.pool,
                                1,
                            )
                        };
                        rec.owned_copy_barrier(false, true);
                        rec.finish().unwrap();
                        let mut ticks = [0u64; 2];
                        unsafe {
                            device.get_query_pool_results(
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
                        fixture.assert_result(
                            &be,
                            variants[variant].list,
                            variants[variant].fifo,
                            variants[variant].skip_hit_updates,
                            variants[variant].name == "gpu_owned_pool_certified"
                                && std::env::var("MOE4ALL_OWNER_PREPARE_CACHE_SNAPSHOT").as_deref()
                                    == Ok("1"),
                        );
                        if round >= 2 {
                            samples[variant].push(
                                delta as f64 * period as f64 / 1000.0 / fixture.repeats as f64,
                            );
                        }
                    }
                }
                for (variant, mut values) in variants.iter().zip(samples) {
                    if values.is_empty() {
                        continue;
                    }
                    values.sort_by(f64::total_cmp);
                    eprintln!("GPU_OWNER_POOL_PREPARE variant={} slots_per_pool={slots} pools=2 layers={LAYERS} experts={EXPERTS} routed={count} workload={} misses={} repeats={} samples={} mean_us={:.3} p50_us={:.3} min_us={:.3} exact_lru=pass",
                        variant.name, workload.name(), fixture.misses, fixture.repeats, values.len(), values.iter().sum::<f64>() / values.len() as f64, values[values.len() / 2], values[0]);
                }
            }
        }
    }
}

#[test]
#[ignore = "central GPU runner only; explicit subgroup-clock feature and external serial slow76 SPV"]
fn gpu_owned_cold_phase_clock_probe() {
    const ORIGINAL_SLOW_SHA256: &str =
        "572033E6D16B5CD7582EED5CDE5E094756B7ADDF70BA97C797D17DF51D0C6FE3";
    const PHASES: [&str; 12] = [
        "initialize",
        "collect",
        "classify_plans",
        "preflight",
        "select_fifo",
        "prepare_queues",
        "compact_queues",
        "install_barrier",
        "report_barrier",
        "rebind_barrier",
        "append_lane0",
        "account_finish_lane0",
    ];
    const CLOCK_WORDS: usize = 32;
    const MARKERS: usize = PHASES.len() + 1;
    const ROUNDS: usize = 6;
    const WARMUP: usize = 2;
    assert_eq!(probe_used(), 10, "clock probe is restricted to used10");
    assert!(
        std::env::var("MOE4ALL_TEST_SHADER_CLOCK").as_deref() == Ok("1"),
        "clock probe requires the test-only MOE4ALL_TEST_SHADER_CLOCK=1 device opt-in"
    );
    let path = std::env::var_os("MOE4ALL_OWNER_PREPARE_CLOCK_SPV")
        .expect("provide instrumented original serial slow76 SPV; enable shaderSubgroupClock");
    eprintln!(
        "GPU_OWNER_COLD_CLOCK_SHADER path={} original_sha256={ORIGINAL_SLOW_SHA256} feature_required=shaderSubgroupClock serial_only=true",
        std::path::Path::new(&path).display()
    );
    let bytes = std::fs::read(path).unwrap();
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicitly select an idle device"));
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let fast = be.kernel(
        "gpu_owned_cold_clock_fast_builtin",
        crate::gemm::gpu_owned_prepare_fast_spv(false),
        8,
        GPU_OWNED_FAST_PUSH_BYTES,
    );
    let slow = be.kernel(
        "gpu_owned_cold_clock_slow_external",
        &words,
        9,
        GPU_COPY_PREPARE_PUSH_BYTES,
    );
    for workload in [Workload::OneExpertMiss, Workload::OneExpertStale] {
        let mut fixture = PoolProbe::new(&be, 4096, 10, workload);
        assert!(fixture.misses > 0);
        assert_eq!(fixture.repeats, 1);
        assert_eq!(fixture.push[12], 0, "recovery is outside this phase probe");
        // Explicitly size FIFO storage: this test does not depend on other probe env flags.
        let slots = fixture.slots_per_pool;
        let fifo_bytes = (fixture.heap_base + 2 + slots * 4) * 16;
        fixture.buffers[5] = be.alloc_mapped_arena_bda(fifo_bytes).unwrap().0;
        let clock = be.alloc_mapped_arena_bda(CLOCK_WORDS * 4).unwrap().0;
        let bindings: Vec<_> = fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
            .collect();
        let mut clock_bindings = bindings.clone();
        clock_bindings.push(Recorder::vkb_exact(clock.as_ref()));
        let geometry = [
            0,
            SLOT_BYTES[0],
            0,
            slots as u32,
            0,
            SLOT_BYTES[0],
            0,
            slots as u32,
            1,
            SLOT_BYTES[1],
            slots as u32,
            slots as u32,
        ];
        let fast_push = gpu_owned_fast_push(&fixture.push, &geometry);
        let mut samples: [Vec<u64>; PHASES.len()] = std::array::from_fn(|_| Vec::new());
        let mut totals = Vec::new();
        for round in 0..ROUNDS {
            fixture.reset(&be, false, true);
            upload(&be, clock.as_ref(), &[0; CLOCK_WORDS]);
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            rec.dispatch(fast, &bindings, 8, bytemuck::cast_slice(&fast_push), 1);
            rec.owned_copy_barrier(false, false);
            rec.dispatch(
                slow,
                &clock_bindings,
                9,
                bytemuck::cast_slice(&fixture.push),
                1,
            );
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            fixture.assert_result(&be, false, true, false, false);
            let output = read(&be, clock.as_ref());
            assert_eq!(output.len(), CLOCK_WORDS);
            assert_eq!(output[26], 0x434c_4b31, "complete CLK1 sample required");
            assert_eq!(output[27], MARKERS as u32);
            assert_eq!(
                output[28],
                fixture.expected[3][RESIDENT_JOURNAL_STATUS_WORD]
            );
            assert_eq!(output[29], 1, "fixture must take the missing-request path");
            assert_eq!(output[30], fixture.push[5]);
            assert_eq!(output[31], fixture.push[0] * 3);
            let ticks: [u64; MARKERS] = std::array::from_fn(|i| stamp(&output[i * 2..i * 2 + 2]));
            assert!(
                ticks.windows(2).all(|pair| pair[0] <= pair[1]),
                "lane0 clock markers must be monotonic: {ticks:?}"
            );
            let total = ticks[MARKERS - 1] - ticks[0];
            assert!(total > 0, "subgroup clock must advance");
            let deltas: [u64; PHASES.len()] = std::array::from_fn(|i| ticks[i + 1] - ticks[i]);
            assert_eq!(deltas.iter().sum::<u64>(), total);
            if round >= WARMUP {
                for (samples, delta) in samples.iter_mut().zip(deltas) {
                    samples.push(delta);
                }
                totals.push(total);
            }
        }
        eprintln!(
            "GPU_OWNER_COLD_CLOCK workload={} slots_per_pool={slots} routed=10 misses={} samples={} mean_total_ticks={:.1} units=unscaled_subgroup_clock_ticks lane0_progress=true fifo_history=seeded_live exact_lru=pass",
            workload.name(),
            fixture.misses,
            totals.len(),
            totals.iter().map(|&value| value as f64).sum::<f64>() / totals.len() as f64
        );
        for (phase, mut values) in PHASES.into_iter().zip(samples) {
            values.sort_unstable();
            eprintln!(
                "GPU_OWNER_COLD_CLOCK_PHASE workload={} phase={phase} samples={} mean_ticks={:.1} p50_ticks={} min_ticks={}",
                workload.name(),
                values.len(),
                values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64,
                values[values.len() / 2],
                values[0]
            );
        }
    }
}

#[test]
#[ignore = "central GPU runner only; explicit subgroup-clock feature and external serial fast124 SPV"]
fn gpu_owned_fast_phase_clock_probe() {
    const ORIGINAL_FAST_SHA256: &str =
        "DB85A9F5326CF2D141F8EE6775681A09C8AD546E578F71788DA029CA7DEB78BF";
    const PHASES: [&str; 6] = [
        "initialize_header_prefix",
        "gather_queue_preflight",
        "work_invalid_decision",
        "compact_or_noop",
        "ids_tape_stamps_lane0",
        "stats_finish_lane0",
    ];
    const CLOCK_WORDS: usize = 32;
    const MARKERS: usize = PHASES.len() + 1;
    const ROUNDS: usize = 6;
    const WARMUP: usize = 2;
    let used = probe_used();
    assert!(
        [8, 10].contains(&used),
        "fast clock probe supports used8/10"
    );
    assert!(
        std::env::var("MOE4ALL_TEST_SHADER_CLOCK").as_deref() == Ok("1"),
        "clock probe requires the test-only MOE4ALL_TEST_SHADER_CLOCK=1 device opt-in"
    );
    let path = std::env::var_os("MOE4ALL_OWNER_PREPARE_FAST_CLOCK_SPV")
        .expect("provide instrumented original serial fast124 SPV");
    eprintln!(
        "GPU_OWNER_FAST_CLOCK_SHADER path={} original_sha256={ORIGINAL_FAST_SHA256} feature_required=shaderSubgroupClock serial_only=true",
        std::path::Path::new(&path).display()
    );
    let bytes = std::fs::read(path).unwrap();
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicitly select an idle device"));
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let fast = be.kernel(
        "gpu_owned_fast_clock_external",
        &words,
        9,
        GPU_OWNED_FAST_PUSH_BYTES,
    );
    let mut fixture = PoolProbe::new(&be, 4096, used, Workload::AllHit);
    assert_eq!(fixture.misses, 0);
    assert_eq!(fixture.push[12], 0, "normal all-hit only, not recovery");
    let slots = fixture.slots_per_pool;
    let fifo_bytes = (fixture.heap_base + 2 + slots * 4) * 16;
    fixture.buffers[5] = be.alloc_mapped_arena_bda(fifo_bytes).unwrap().0;
    let clock = be.alloc_mapped_arena_bda(CLOCK_WORDS * 4).unwrap().0;
    let mut bindings: Vec<_> = fixture
        .buffers
        .iter()
        .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
        .collect();
    bindings.push(Recorder::vkb_exact(clock.as_ref()));
    let geometry = [
        0,
        SLOT_BYTES[0],
        0,
        slots as u32,
        0,
        SLOT_BYTES[0],
        0,
        slots as u32,
        1,
        SLOT_BYTES[1],
        slots as u32,
        slots as u32,
    ];
    let fast_push = gpu_owned_fast_push(&fixture.push, &geometry);
    let mut samples: [Vec<u64>; PHASES.len()] = std::array::from_fn(|_| Vec::new());
    let mut totals = Vec::new();
    for round in 0..ROUNDS {
        fixture.reset(&be, false, true);
        upload(&be, clock.as_ref(), &[0; CLOCK_WORDS]);
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        for iteration in 0..fixture.repeats {
            if iteration != 0 {
                rec.owned_copy_barrier(false, false);
            }
            rec.dispatch(fast, &bindings, 9, bytemuck::cast_slice(&fast_push), 1);
        }
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
        fixture.assert_result(&be, false, true, false, false);
        let output = read(&be, clock.as_ref());
        assert_eq!(output.len(), CLOCK_WORDS);
        assert_eq!(output[26], 0x4643_4c31, "complete FCL1 sample required");
        assert_eq!(output[27], MARKERS as u32);
        assert_eq!(output[28], 1, "all-hit fast must finish without slow");
        assert_eq!(output[29], 0, "ordinary hit path, no compaction/recovery");
        assert_eq!(output[30], fixture.push[5]);
        assert_eq!(output[31], fixture.push[0] * 3);
        assert!(output[MARKERS * 2..26].iter().all(|&word| word == 0));
        let ticks: [u64; MARKERS] = std::array::from_fn(|i| stamp(&output[i * 2..i * 2 + 2]));
        assert!(
            ticks.windows(2).all(|pair| pair[0] <= pair[1]),
            "lane0 fast clock markers must be monotonic: {ticks:?}"
        );
        let total = ticks[MARKERS - 1] - ticks[0];
        assert!(total > 0, "subgroup clock must advance");
        let deltas: [u64; PHASES.len()] = std::array::from_fn(|i| ticks[i + 1] - ticks[i]);
        assert_eq!(deltas.iter().sum::<u64>(), total);
        if round >= WARMUP {
            for (samples, delta) in samples.iter_mut().zip(deltas) {
                samples.push(delta);
            }
            totals.push(total);
        }
    }
    eprintln!(
        "GPU_OWNER_FAST_CLOCK workload=all_hit slots_per_pool={slots} routed={used} repeats={} last_iteration=true samples={} mean_total_ticks={:.1} units=unscaled_subgroup_clock_ticks lane0_progress=true exact_lru=pass",
        fixture.repeats,
        totals.len(),
        totals.iter().map(|&value| value as f64).sum::<f64>() / totals.len() as f64
    );
    for (phase, mut values) in PHASES.into_iter().zip(samples) {
        values.sort_unstable();
        eprintln!(
            "GPU_OWNER_FAST_CLOCK_PHASE phase={phase} samples={} mean_ticks={:.1} p50_ticks={} min_ticks={}",
            values.len(),
            values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64,
            values[values.len() / 2],
            values[0]
        );
    }
}

#[test]
#[ignore = "central GPU runner only; metadata-only rejection stress; use a subprocess deadline"]
fn gpu_owned_collect_classify_rejection_stress() {
    const SLOTS: usize = 4096;
    const REPEATS: usize = 16;
    let used = probe_used();
    assert!([8, 10, 16].contains(&used), "bounded used8/10/16 fixtures");
    let count = used * 4;
    assert!(count * 3 > 64, "exercise more than one WG64 request pass");
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicitly select an idle device"));
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(
        be.shared
            .compute_subgroup_operations
            .contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC),
        "explicit test device must support COMPUTE BASIC|ARITHMETIC for all three paths"
    );
    let subgroup = crate::gemm::gpu_owned_prepare_slow_subgroup_spv(false);
    let variants = [
        (
            "fallback_legacy76",
            be.kernel(
                "gpu_owned_classify_reject_legacy76",
                crate::gemm::gpu_owned_prepare_slow_spv(false),
                8,
                GPU_COPY_PREPARE_PUSH_BYTES,
            ),
        ),
        (
            "subgroup_native76",
            be.kernel_native(
                "gpu_owned_classify_reject_native76",
                subgroup,
                8,
                GPU_COPY_PREPARE_PUSH_BYTES,
            ),
        ),
        (
            "subgroup_fixed32_76",
            be.kernel_sg(
                "gpu_owned_classify_reject_fixed32_76",
                subgroup,
                8,
                GPU_COPY_PREPARE_PUSH_BYTES,
                32,
            ),
        ),
        (
            "fused_fixed32_124",
            be.kernel(
                "gpu_owned_classify_reject_fused124",
                crate::gemm::gpu_owned_prepare_fused_spv(false),
                8,
                GPU_OWNED_FAST_PUSH_BYTES,
            ),
        ),
    ];
    for case in ["conflicting_tape_alias", "nonuniform_bank_pool_stride"] {
        let mut fixture = PoolProbe::new(&be, SLOTS, count, Workload::OneExpertMiss);
        assert_eq!(fixture.push[5], 0);
        assert_eq!(fixture.push[12], 0);
        assert_eq!(fixture.repeats, 1);
        match case {
            "conflicting_tape_alias" => {
                fixture.push[3] = fixture.push[2];
                assert_ne!(fixture.push[9], fixture.push[10]);
            }
            "nonuniform_bank_pool_stride" => {
                let block = fixture.push[10] as usize + fixture.initial[0][0] as usize;
                assert_eq!(
                    fixture.initial[6][block * 4 + 2],
                    0,
                    "alter only a FREE request"
                );
                fixture.initial[4][block * 4 + 2] = SLOT_BYTES[1];
                fixture.initial[4][block * 4 + 3] = 1;
                let next = fixture.push[10] as usize + fixture.initial[0][1] as usize;
                assert_ne!(
                    &fixture.initial[4][block * 4 + 2..block * 4 + 4],
                    &fixture.initial[4][next * 4 + 2..next * 4 + 4]
                );
            }
            _ => unreachable!(),
        }
        let fifo_bytes = (fixture.heap_base + 2 + SLOTS * 4) * 16;
        fixture.buffers[5] = be.alloc_mapped_arena_bda(fifo_bytes).unwrap().0;
        fixture.reset(&be, false, true);
        let mut initial: [Vec<u32>; 8] =
            std::array::from_fn(|i| read(&be, fixture.buffers[i].as_ref()));
        initial[3][RESIDENT_JOURNAL_STATUS_WORD] = 6;
        // Assert the intended phase precondition independently: all per-request collect
        // checks pass; only cross-request Tape/class consistency is malformed.
        let clock = stamp(&initial[5][10..12]);
        for q in 0..count * 3 {
            let role = q / count;
            let id = initial[0][q % count] as usize;
            let tb = fixture.push[2 + role] as usize;
            let bb = fixture.push[9 + role] as usize;
            assert!(id < EXPERTS && tb + EXPERTS <= initial[1].len() / 2);
            assert!(bb + EXPERTS <= initial[4].len() / 4);
            assert!(bb + EXPERTS <= initial[6].len() / 4);
            let block = bb + id;
            let source = &initial[4][block * 4..block * 4 + 4];
            let pool = source[3] as usize;
            assert!(pool < 2 && source[2] > 0 && source[2] & 15 == 0);
            let source_address = stamp(source);
            assert!(source_address != 0 && source_address != u64::MAX);
            assert_eq!(source_address & 15, 0);
            assert!(source_address
                .checked_add(u64::from(source[2]) - 1)
                .is_some());
            let entry = &initial[6][block * 4..block * 4 + 4];
            assert!(entry[2] <= 1);
            if entry[2] == 1 {
                let slot = entry[3] as usize;
                assert!((pool * SLOTS..(pool + 1) * SLOTS).contains(&slot));
                let canonical = &initial[5][(fixture.address_base + slot) * 4..];
                assert_eq!(
                    &canonical[2..4],
                    &[pool as u32, (slot - pool * SLOTS) as u32]
                );
                assert_eq!(&entry[..2], &canonical[..2]);
                let address = stamp(canonical);
                assert!(address != 0 && address != u64::MAX && address & 15 == 0);
                assert!(address.checked_add(u64::from(source[2]) - 1).is_some());
                let owner = &initial[5][(SLOT_BASE + slot) * 4..];
                assert_eq!(&owner[2..4], &[block as u32, 1]);
                assert!(stamp(owner) > 0 && stamp(owner) <= clock);
            }
        }
        let mut expected = initial.clone();
        expected[2][..5].copy_from_slice(&[0, 0, 1, 1, 0]);
        if fixture.push[8] != 0 {
            expected[2][5..7].fill(0);
        }
        expected[3][RESIDENT_JOURNAL_JOB_COUNT_WORD..RESIDENT_JOURNAL_STATUS_WORD].fill(0);
        expected[3][RESIDENT_JOURNAL_STATUS_WORD] = 3;
        let bindings: Vec<_> = fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
            .collect();
        let mut geometry = [0u32; 12];
        for role in 0..3 {
            let source = fixture.push[9 + role] as usize * 4;
            let pool = initial[4][source + 3];
            geometry[role * 4..role * 4 + 4].copy_from_slice(&[
                pool,
                initial[4][source + 2],
                pool * SLOTS as u32,
                SLOTS as u32,
            ]);
        }
        let fused_push = gpu_owned_fast_push(&fixture.push, &geometry);
        for &(name, slow) in &variants {
            eprintln!("GPU_OWNER_CLASSIFY_REJECT_BEGIN case={case} variant={name} requests={} repeats={REPEATS} fixed32_has_two_subgroups=true native_width=device_selected subprocess_deadline_required=true", count * 3);
            for iteration in 0..REPEATS {
                // Complete FIFO, journal, control and jobs reset, not just the run flag.
                for (buffer, words) in fixture.buffers.iter().zip(&initial) {
                    upload(&be, buffer.as_ref(), words);
                }
                let rec = be.recorder().unwrap();
                rec.owned_copy_barrier(true, false);
                let push = if name == "fused_fixed32_124" {
                    bytemuck::cast_slice(&fused_push)
                } else {
                    bytemuck::cast_slice(&fixture.push)
                };
                rec.dispatch(slow, &bindings, 8, push, 1);
                rec.owned_copy_barrier(false, true);
                rec.finish().unwrap();
                for (binding, expected) in expected.iter().enumerate() {
                    assert_eq!(
                        read(&be, fixture.buffers[binding].as_ref()),
                        *expected,
                        "{case}/{name}/iteration{iteration}: exact rejection binding {binding}"
                    );
                }
            }
            eprintln!("GPU_OWNER_CLASSIFY_REJECT_PASS case={case} variant={name} requests={} repeats={REPEATS} status=3 jobs=0 all8_exact=true payload=false", count * 3);
        }
    }
}

#[test]
#[ignore = "central GPU runner only; external serial slow76 subgroup scan, metadata only"]
fn gpu_owned_slow_scan_differential_probe() {
    assert_eq!(probe_used(), 10, "bounded scan fixtures require used10");
    let path = std::env::var_os("MOE4ALL_OWNER_PREPARE_FIFO_SLOW_SPV")
        .expect("provide the external serial slow76 scan SPV");
    let bytes = std::fs::read(&path).unwrap();
    let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicitly select an idle device"));
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let fast = be.kernel(
        "gpu_owned_scan_diff_fast_builtin124",
        crate::gemm::gpu_owned_prepare_fast_spv(false),
        8,
        GPU_OWNED_FAST_PUSH_BYTES,
    );
    let variants = [
        (
            "embedded76",
            be.kernel(
                "gpu_owned_scan_diff_slow_builtin76",
                crate::gemm::gpu_owned_prepare_slow_spv(false),
                8,
                GPU_COPY_PREPARE_PUSH_BYTES,
            ),
        ),
        (
            "external_native76",
            be.kernel_native(
                "gpu_owned_scan_diff_slow_native76",
                &words,
                8,
                GPU_COPY_PREPARE_PUSH_BYTES,
            ),
        ),
        (
            "external_fixed32_76",
            be.kernel_sg(
                "gpu_owned_scan_diff_slow_fixed32_76",
                &words,
                8,
                GPU_COPY_PREPARE_PUSH_BYTES,
                32,
            ),
        ),
    ];
    for (case, slots, prepend, corrupt) in [
        ("stale64_clean", 4096, 64, None),
        ("stale64_bad_before", 4096, 64, Some(63)),
        ("stale64_bad_after", 4096, 64, Some(66)),
        ("partial44_clean", 40, 4, None),
    ] {
        let mut fixture = PoolProbe::new(&be, slots, 10, Workload::OneExpertMiss);
        assert_eq!(fixture.misses, 3);
        assert_eq!(fixture.repeats, 1);
        let roles = [0, 1, 2].map(|r| fixture.push[9 + r] as usize);
        let ids = fixture.initial[0][..10].to_vec();
        let protected: HashSet<_> = roles
            .iter()
            .flat_map(|&base| ids.iter().map(move |&id| base + id as usize))
            .collect();
        // The stock seed puts protected hits first. Retimestamp pool0 so its first two
        // live records really are eligible at prepend/ prepend+1, without changing owners.
        let clock = stamp(&fixture.initial[5][10..12]);
        let mut order: Vec<_> = (0..slots).collect();
        order.sort_unstable_by_key(|&slot| {
            (
                protected.contains(&(fixture.initial[5][(SLOT_BASE + slot) * 4 + 2] as usize)),
                slot,
            )
        });
        for (rank, slot) in order.into_iter().enumerate() {
            fixture.initial[5][(SLOT_BASE + slot) * 4..(SLOT_BASE + slot) * 4 + 2]
                .copy_from_slice(&pair(clock - (slots * 2) as u64 + rank as u64 + 1));
        }
        (fixture.expected, fixture.misses) =
            reference(&fixture.initial, &ids, roles, slots, fixture.address_base);
        let fifo_bytes = (fixture.heap_base + 2 + slots * 4) * 16;
        fixture.buffers[5] = be.alloc_mapped_arena_bda(fifo_bytes).unwrap().0;
        fixture.reset(&be, false, true);
        let mut initial: [Vec<u32>; 8] =
            std::array::from_fn(|i| read(&be, fixture.buffers[i].as_ref()));
        let queue_base = initial[5][POOL_BASE * 4] as usize;
        let records = (queue_base + 1) * 4;
        assert_eq!(initial[5][queue_base * 4], 0);
        assert_eq!(initial[5][queue_base * 4 + 1], slots as u32);
        assert!(slots + prepend <= initial[5][queue_base * 4 + 2] as usize);
        let first_slot = initial[5][records + 2];
        initial[5].copy_within(records..records + slots * 4, records + prepend * 4);
        for index in 0..prepend {
            initial[5][records + index * 4..records + (index + 1) * 4]
                .copy_from_slice(&[1, 0, first_slot, 0]);
        }
        initial[5][queue_base * 4 + 1] = (slots + prepend) as u32;
        assert!(stamp(&initial[5][(SLOT_BASE + first_slot as usize) * 4..]) > 1);
        for index in prepend..prepend + 2 {
            let slot = initial[5][records + index * 4 + 2] as usize;
            let live = &initial[5][(SLOT_BASE + slot) * 4..];
            assert_eq!(live[3], 1);
            assert!(!protected.contains(&(live[2] as usize)));
            assert_eq!(stamp(live), stamp(&initial[5][records + index * 4..]));
        }
        if let Some(index) = corrupt {
            initial[5][records + index * 4 + 2] = NONE;
        }
        let rejected = corrupt == Some(63);
        // Independent oracle for the bounded no-compaction cases. Preserve all original
        // log bytes (including ignored corruption), append exact final role-major touches.
        let mut expected = fixture.expected.clone();
        expected[5] = initial[5].clone();
        expected[5][..fixture.heap_base * 4]
            .copy_from_slice(&fixture.expected[5][..fixture.heap_base * 4]);
        expected[5][0] = 4;
        for pool in 0..2 {
            let pn = (POOL_BASE + pool * 2) * 4;
            let base = initial[5][pn] as usize;
            expected[5][pn] = base as u32;
            let header = base * 4;
            let tail = initial[5][header + 1] as usize;
            expected[5][header] = if pool == 0 { prepend as u32 } else { 0 };
            let mut rank = 0;
            for &role in &roles {
                for &id in &ids {
                    let block = role + id as usize;
                    if pool_of(block) != pool {
                        continue;
                    }
                    let slot = expected[6][block * 4 + 3] as usize;
                    let value = pair(stamp(&expected[5][(SLOT_BASE + slot) * 4..]));
                    let target = (base + 1 + tail + rank) * 4;
                    expected[5][target..target + 4].copy_from_slice(&[
                        value[0],
                        value[1],
                        slot as u32,
                        0,
                    ]);
                    rank += 1;
                }
            }
            assert!(tail + rank <= initial[5][header + 2] as usize);
            expected[5][header + 1] = (tail + rank) as u32;
        }
        let bindings: Vec<_> = fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
            .collect();
        let geometry = [
            0,
            SLOT_BYTES[0],
            0,
            slots as u32,
            0,
            SLOT_BYTES[0],
            0,
            slots as u32,
            1,
            SLOT_BYTES[1],
            slots as u32,
            slots as u32,
        ];
        let fast_push = gpu_owned_fast_push(&fixture.push, &geometry);
        let mut baseline: Option<[Vec<u32>; 8]> = None;
        for &(name, slow) in &variants {
            for (buffer, words) in fixture.buffers.iter().zip(&initial) {
                upload(&be, buffer.as_ref(), words);
            }
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            rec.dispatch(fast, &bindings, 8, bytemuck::cast_slice(&fast_push), 1);
            rec.owned_copy_barrier(false, false);
            rec.dispatch(slow, &bindings, 8, bytemuck::cast_slice(&fixture.push), 1);
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            let output: [Vec<u32>; 8] =
                std::array::from_fn(|i| read(&be, fixture.buffers[i].as_ref()));
            assert_eq!(
                output[3][RESIDENT_JOURNAL_STATUS_WORD],
                if rejected { 3 } else { 4 },
                "{case}/{name}: exact visited-prefix status"
            );
            if rejected {
                assert_eq!(output[2][0], 0, "{case}/{name}: stop on invalid directory");
                assert_eq!(output[3][RESIDENT_JOURNAL_JOB_COUNT_WORD], 0);
                for binding in [0, 1, 4, 5, 6, 7] {
                    assert_eq!(
                        output[binding], initial[binding],
                        "{case}/{name}: rejected preflight mutated binding {binding}"
                    );
                }
            } else {
                for binding in 0..8 {
                    assert_eq!(
                        output[binding], expected[binding],
                        "{case}/{name}: independent exact oracle binding {binding}"
                    );
                }
                if corrupt.is_none() {
                    fixture.assert_result(&be, false, true, false, false);
                }
            }
            if let Some(original) = &baseline {
                for binding in 0..8 {
                    assert_eq!(
                        output[binding], original[binding],
                        "{case}/{name}: raw differential binding {binding}, no normalization"
                    );
                }
            } else {
                baseline = Some(output);
            }
            eprintln!("GPU_OWNER_SCAN_DIFFERENTIAL case={case} variant={name} slots_per_pool={slots} stale_prefix={prepend} status={} all8_raw_equal=true independent_oracle=pass payload=false external={}",
                if rejected { 3 } else { 4 }, std::path::Path::new(&path).display());
        }
    }
}

#[test]
#[ignore = "central GPU runner only; metadata publication never dereferences synthetic BDAs"]
fn gpu_owned_parallel_publish_latency_probe() {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select an idle device"));
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let bytes = std::fs::read(std::env::var_os("MOE4ALL_OWNER_PUBLISH_SPV").unwrap()).unwrap();
    let parallel = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    let kernels = [
        be.kernel(
            "gpu_owned_publish_probe_builtin",
            crate::gemm::gpu_owned_publish_spv(),
            4,
            8,
        ),
        be.kernel("gpu_owned_publish_probe_parallel", &parallel, 4, 8),
    ];
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
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    let period = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits
    .timestamp_period;
    for workload in [Workload::OneExpertMiss, Workload::MixedFresh] {
        let fixture = PoolProbe::new(&be, 4096, probe_used(), workload);
        // Independent prepare reference supplies all input metadata. No payload is read or copied.
        for (mixed_cpu, mode, corrupt) in [
            (false, 0u32, false),
            (true, 0, false),
            (true, 1, false),
            (true, 1, true),
        ] {
            let mut initial = fixture.expected.clone();
            if mixed_cpu {
                initial[3][RESIDENT_JOURNAL_STATUS_WORD] = 5;
                for j in (0..fixture.misses).step_by(2) {
                    initial[7][j * 8..j * 8 + 2].fill(0);
                }
            }
            if corrupt {
                // A bad final destination must suppress publication of every preceding job.
                initial[7][(fixture.misses - 1) * 8 + 6] = u32::MAX;
            }
            let mut expected = initial.clone();
            if !corrupt {
                for j in 0..fixture.misses {
                    let job = &initial[7][j * 8..(j + 1) * 8];
                    if mode == 0 && job[..2] == [0, 0] {
                        continue;
                    }
                    expected[5][(SLOT_BASE + job[6] as usize) * 4 + 3] = 1;
                    expected[6][job[5] as usize * 4 + 2] = 1;
                }
            }
            let bindings: Vec<_> = [3, 7, 5, 6]
                .into_iter()
                .map(|index| Recorder::vkb(fixture.buffers[index].as_ref()))
                .collect();
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..8 {
                for offset in 0..kernels.len() {
                    let variant = (round + offset) % kernels.len();
                    for (buffer, words) in fixture.buffers.iter().zip(&initial) {
                        upload(&be, buffer.as_ref(), words);
                    }
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
                    rec.dispatch(
                        kernels[variant],
                        &bindings,
                        4,
                        bytemuck::cast_slice(&[0u32, mode]),
                        1,
                    );
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
                    for (index, expected) in expected.iter().enumerate() {
                        assert_eq!(read(&be, fixture.buffers[index].as_ref()), *expected,
                            "publish {variant} binding {index}, mixed={mixed_cpu}, mode={mode}, corrupt={corrupt}");
                    }
                    if round >= 2 {
                        samples[variant].push(delta as f64 * f64::from(period) / 1000.0);
                    }
                }
            }
            eprintln!("GPU_OWNER_PUBLISH jobs={} mixed_cpu={mixed_cpu} mode={mode} corrupt={corrupt} builtin_us={:.3} parallel_us={:.3} exact=pass",
                fixture.misses, samples[0].iter().sum::<f64>() / samples[0].len() as f64,
                samples[1].iter().sum::<f64>() / samples[1].len() as f64);
        }
    }
}
