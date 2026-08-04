//! Host-paged weights for the CPU backend: weights read from the model file into a bounded DRAM
//! arena instead of being mapped and left to the OS page cache
//! (`docs/disk-streaming-plan.md` §3.6).
//!
//! # Why pools
//! [`infr_core::hostpager::HostPager`] holds UNIFORM slots, and a model's tensors are anything but
//! — a norm is kilobytes, an lm_head is gigabytes. One arena sized to the largest would spend most
//! of its budget on padding. So this keeps one pager per distinct BLOCK SIZE. That is not an
//! arbitrary bucketing: a transformer repeats the same layer shape, so every layer's `attn_q` is
//! byte-identical in size to every other's, and a dense model lands in a handful of classes with
//! no padding at all.
//!
//! # Why the pools are planned before the load
//! Every pool exists before the first weight is bound, sized from the tensor directory the GGUF
//! already carries. Creating them lazily instead would mean growing the pool list while bound
//! buffers hold references into it, and it would hand the whole budget to whichever size class
//! happened to be bound first — the last layer's classes appear last in a load, so first-come is
//! the wrong order to spend a budget in.
//!
//! # What stays mapped
//! Weights below [`MIN_PAGED_BYTES`], and any size class the budget cannot seat even once. Those
//! keep the zero-copy mmap path, so a budget too small to be useful degrades to today's behaviour
//! instead of failing the load.

use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::error::Result;
use infr_core::hostpager::{HostPager, HostPagerStats, Pin};
use infr_core::loader::TensorInfo;
use infr_core::pager::Insert;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Smallest weight worth paging. Below this the arena bookkeeping outweighs the bytes: a 1 MiB
/// floor keeps norms, biases and rope tables mapped (a few MB across a whole model) while covering
/// every projection and expert bank, which are what a forward pass streams.
pub const MIN_PAGED_BYTES: usize = 1 << 20;

/// Handle to one registered paged weight — what a [`crate::CpuBuffer`] carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedId {
    pool: usize,
    block: infr_core::pager::BlockId,
}

/// One planned size class: `n_slots` resident blocks of `slot_bytes` each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolPlan {
    pub slot_bytes: usize,
    pub n_slots: usize,
    /// How many blocks of this size the model has — the ceiling on useful slots.
    pub n_blocks: usize,
}

/// Decide the pools for a model's tensors within `budget_bytes`.
///
/// One class per distinct weight size above [`MIN_PAGED_BYTES`]; the budget splits across them by
/// [`infr_core::hostpager::plan_slots`], which is the same rule the Vulkan dense session's host
/// tier uses. A class the budget cannot seat once is dropped — it stays mapped.
pub fn plan_pools(budget_bytes: usize, tensors: &[TensorInfo]) -> Vec<PoolPlan> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for t in tensors.iter().filter(|t| t.nbytes >= MIN_PAGED_BYTES) {
        *counts.entry(t.nbytes).or_insert(0) += 1;
    }
    let mut classes: Vec<(usize, usize)> = counts.into_iter().collect();
    // A `HashMap`'s iteration order is not stable, and `plan_slots` only promises to preserve the
    // order it is GIVEN — sort here so the same model and budget produce the same plan every run.
    classes.sort_unstable();
    infr_core::hostpager::plan_slots(budget_bytes, &classes)
        .into_iter()
        .zip(classes)
        .filter(|&(n_slots, _)| n_slots > 0)
        .map(|(n_slots, (slot_bytes, n_blocks))| PoolPlan {
            slot_bytes,
            n_slots,
            n_blocks,
        })
        .collect()
}

/// The CPU backend's host weight cache: one [`HostPager`] per planned size class.
pub struct PagedWeights {
    pools: Vec<HostPager>,
    /// Block size -> pool index. Fixed at construction.
    classes: HashMap<usize, usize>,
    /// Per-pool next block id. Registration is load-time only, but it takes `&self` so that bound
    /// buffers can already hold an `Arc` of this store.
    next_block: Mutex<Vec<u32>>,
}

impl PagedWeights {
    /// Build the pools named by [`plan_pools`]. Allocates every arena up front, so a budget that
    /// cannot be committed fails here rather than part-way through a generation.
    pub fn new(plans: &[PoolPlan], io: Arc<dyn BlockIo>) -> Result<Self> {
        let mut pools = Vec::with_capacity(plans.len());
        let mut classes = HashMap::with_capacity(plans.len());
        for (i, p) in plans.iter().enumerate() {
            pools.push(HostPager::new(p.n_slots, p.slot_bytes, Arc::clone(&io))?);
            classes.insert(p.slot_bytes, i);
        }
        Ok(Self {
            next_block: Mutex::new(vec![0; pools.len()]),
            pools,
            classes,
        })
    }

    /// Register one weight's file ranges, or decline it.
    ///
    /// `Ok(None)` means this weight has no pool (too small, or its class did not fit the budget) —
    /// the caller maps it instead.
    pub fn register(&self, ranges: &[(u64, usize)]) -> Result<Option<PagedId>> {
        let nbytes: usize = ranges.iter().map(|(_, l)| l).sum();
        let Some(&pool) = self.classes.get(&nbytes) else {
            return Ok(None);
        };
        let block = {
            let mut next = self.next_block.lock().unwrap();
            let id = next[pool];
            next[pool] += 1;
            id
        };
        self.pools[pool].register(BlockDesc {
            id: block,
            extents: ranges
                .iter()
                .map(|&(offset, len)| BlockExtent { offset, len })
                .collect(),
        })?;
        Ok(Some(PagedId { pool, block }))
    }

    /// Make `id`'s bytes resident and pin them, reading from the model file on a miss.
    ///
    /// [`Insert::Cold`] is the policy: a forward pass visits weights in a fixed order and returns
    /// to the first only after touching every other, so the block just used is the one whose next
    /// use is furthest away. Recency is the pathological choice for that shape — it is what the OS
    /// page cache does, and what `docs/perf/results.md` measured collapsing.
    pub fn pin(&self, id: PagedId) -> Result<Pin<'_>> {
        self.pools[id.pool].pin(id.block, Insert::Cold)
    }

    /// Pin `id` only if it is already resident — the read path, which runs after the op's pre-step
    /// has pinned everything the op names.
    pub fn try_pin(&self, id: PagedId) -> Option<Pin<'_>> {
        self.pools[id.pool].try_pin(id.block)
    }

    /// Per-pool `(slot bytes, slot count, stats)`, for the paging report.
    pub fn pool_stats(&self) -> Vec<(usize, usize, HostPagerStats)> {
        self.pools
            .iter()
            .map(|p| (p.slot_bytes(), p.n_slots(), p.stats()))
            .collect()
    }

    /// Bytes committed to arenas — what the process actually holds.
    pub fn arena_bytes(&self) -> usize {
        self.pools.iter().map(|p| p.arena_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::tensor::DType;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeIo(AtomicUsize);
    impl BlockIo for FakeIo {
        fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
            self.0.fetch_add(1, Ordering::SeqCst);
            let n = desc.nbytes();
            dst[..n].fill(desc.id as u8);
            Ok(())
        }
    }

    const MIB: usize = 1 << 20;

    fn tensors(spec: &[(usize, usize)]) -> Vec<TensorInfo> {
        let mut out = Vec::new();
        for (i, &(nbytes, count)) in spec.iter().enumerate() {
            for k in 0..count {
                out.push(TensorInfo {
                    name: format!("t{i}_{k}"),
                    shape: vec![nbytes],
                    dtype: DType::F32,
                    offset: 0,
                    nbytes,
                });
            }
        }
        out
    }

    #[test]
    fn tensors_below_the_floor_get_no_pool() {
        let plans = plan_pools(64 * MIB, &tensors(&[(MIN_PAGED_BYTES - 1, 8)]));
        assert!(plans.is_empty(), "sub-floor tensors must not be paged");
    }

    /// The proportional split: a class holding most of the bytes gets most of the slots, and no
    /// class gets more slots than it has blocks.
    #[test]
    fn slots_follow_each_class_share_and_never_exceed_its_blocks() {
        // 32 x 4 MiB = 128 MiB, and 2 x 8 MiB = 16 MiB. Budget 72 MiB.
        let plans = plan_pools(72 * MIB, &tensors(&[(4 * MIB, 32), (8 * MIB, 2)]));
        let big = plans.iter().find(|p| p.slot_bytes == 4 * MIB).expect("4M");
        let small = plans.iter().find(|p| p.slot_bytes == 8 * MIB).expect("8M");
        assert!(
            big.n_slots > small.n_slots,
            "the dominant class must get more slots ({} vs {})",
            big.n_slots,
            small.n_slots
        );
        assert!(small.n_slots <= small.n_blocks, "more slots than blocks");
        let committed: usize = plans.iter().map(|p| p.n_slots * p.slot_bytes).sum();
        assert!(committed <= 72 * MIB, "plan overspent: {committed}");
    }

    /// A class the remaining budget cannot seat once is dropped, not squeezed in — and the classes
    /// that matter most are seated first.
    #[test]
    fn a_class_that_cannot_be_seated_is_dropped() {
        // 8 MiB budget: 4 MiB x 8 blocks (32 MiB total) dominates; 6 MiB x 1 is the minor class.
        let plans = plan_pools(8 * MIB, &tensors(&[(4 * MIB, 8), (6 * MIB, 1)]));
        assert!(plans.iter().any(|p| p.slot_bytes == 4 * MIB));
        let committed: usize = plans.iter().map(|p| p.n_slots * p.slot_bytes).sum();
        assert!(committed <= 8 * MIB, "plan overspent: {committed}");
    }

    /// The plan is deterministic: a `HashMap`'s iteration order must not reach it, or residency
    /// would differ run to run for the same model and budget.
    #[test]
    fn the_plan_is_stable_across_runs() {
        let t = tensors(&[(4 * MIB, 3), (8 * MIB, 3), (2 * MIB, 3)]);
        let first = plan_pools(64 * MIB, &t);
        for _ in 0..8 {
            assert_eq!(plan_pools(64 * MIB, &t), first);
        }
    }

    #[test]
    fn registration_maps_equal_sizes_to_one_pool() {
        let io = Arc::new(FakeIo(AtomicUsize::new(0)));
        let plans = plan_pools(64 * MIB, &tensors(&[(4 * MIB, 4), (8 * MIB, 2)]));
        let w = PagedWeights::new(&plans, io).expect("pools");
        let a = w.register(&[(0, 4 * MIB)]).unwrap().expect("paged");
        let b = w.register(&[(99, 4 * MIB)]).unwrap().expect("paged");
        let c = w.register(&[(0, 8 * MIB)]).unwrap().expect("paged");
        assert_eq!(a.pool, b.pool);
        assert_ne!(a.block, b.block, "each weight needs its own block id");
        assert_ne!(a.pool, c.pool);
        // A size with no pool is declined rather than mis-filed into a neighbouring class.
        assert_eq!(w.register(&[(0, 3 * MIB)]).unwrap(), None);
    }

    #[test]
    fn pin_reads_once_then_hits() {
        let io = Arc::new(FakeIo(AtomicUsize::new(0)));
        let plans = plan_pools(64 * MIB, &tensors(&[(MIB, 4)]));
        let w = PagedWeights::new(&plans, io.clone()).expect("pools");
        let id = w.register(&[(0, MIB)]).unwrap().expect("paged");
        assert_eq!(&w.pin(id).unwrap()[..8], &[0u8; 8]);
        assert_eq!(io.0.load(Ordering::SeqCst), 1);
        assert!(w.try_pin(id).is_some());
        assert_eq!(io.0.load(Ordering::SeqCst), 1, "a hit must not re-read");
    }
}
