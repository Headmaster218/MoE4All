//! Logical sub-allocation for the elastic VRAM arena shared by paged experts and auxiliary
//! engines.  The Vulkan backing is layered on this allocator in `lib.rs`; keeping range
//! bookkeeping independent makes alignment, coalescing, accounting and exact slot restoration
//! testable without a GPU.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use infr_core::backend::Buffer;
use infr_core::error::Result;

use super::{as_vk_buf, be, VulkanBackend};

/// Owner of a live range in the unified elastic arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnifiedVramClass {
    Expert,
    EmbeddingWeights,
    EmbeddingRuntime,
    VisionWeights,
    VisionRuntime,
    DraftWeights,
    DraftRuntime,
}

impl UnifiedVramClass {
    const COUNT: usize = 7;

    const fn index(self) -> usize {
        match self {
            Self::Expert => 0,
            Self::EmbeddingWeights => 1,
            Self::EmbeddingRuntime => 2,
            Self::VisionWeights => 3,
            Self::VisionRuntime => 4,
            Self::DraftWeights => 5,
            Self::DraftRuntime => 6,
        }
    }
}

/// Immutable coordinates of one allocation. A range never crosses a physical Vulkan shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnifiedRange {
    pub id: u64,
    pub shard: usize,
    pub offset: usize,
    /// Aligned physical span returned to the allocator on release.
    pub len: usize,
    /// Logical byte count requested by the caller.
    pub requested_len: usize,
    pub class: UnifiedVramClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnifiedVramStats {
    pub capacity_bytes: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub largest_free_bytes: usize,
    pub live_allocations: usize,
    pub bytes_by_class: [usize; UnifiedVramClass::COUNT],
}

impl UnifiedVramStats {
    pub fn class_bytes(&self, class: UnifiedVramClass) -> usize {
        self.bytes_by_class[class.index()]
    }

    /// `0.0` means all free bytes form one range; `1.0` approaches maximally fragmented.
    pub fn fragmentation(&self) -> f64 {
        if self.free_bytes == 0 {
            0.0
        } else {
            1.0 - self.largest_free_bytes as f64 / self.free_bytes as f64
        }
    }
}

#[derive(Debug)]
struct ShardState {
    capacity: usize,
    /// start -> byte length, always sorted, disjoint and maximally coalesced.
    free: BTreeMap<usize, usize>,
}

#[derive(Debug)]
struct PoolState {
    shards: Vec<ShardState>,
    live: HashMap<u64, UnifiedRange>,
    next_id: u64,
    bytes_by_class: [usize; UnifiedVramClass::COUNT],
}

#[derive(Debug)]
struct PoolInner {
    state: Mutex<PoolState>,
    generation: AtomicU64,
}

/// Cloneable owner of one logical set of physical arena shards.
#[derive(Clone, Debug)]
pub struct UnifiedRangePool {
    inner: Arc<PoolInner>,
}

/// RAII lease. The aligned physical range returns to its pool when the final `Arc` is dropped.
#[derive(Debug)]
pub struct UnifiedAllocation {
    range: UnifiedRange,
    owner: Weak<PoolInner>,
}

impl UnifiedAllocation {
    pub fn range(&self) -> UnifiedRange {
        self.range
    }
}

impl Drop for UnifiedAllocation {
    fn drop(&mut self) {
        let Some(owner) = self.owner.upgrade() else {
            return;
        };
        let mut state = owner.state.lock().unwrap();
        if state.live.remove(&self.range.id).is_none() {
            return;
        }
        state.bytes_by_class[self.range.class.index()] =
            state.bytes_by_class[self.range.class.index()].saturating_sub(self.range.len);
        insert_free(
            &mut state.shards[self.range.shard].free,
            self.range.offset,
            self.range.len,
        );
        owner.generation.fetch_add(1, Ordering::Release);
    }
}

impl UnifiedRangePool {
    pub fn new(shard_sizes: impl IntoIterator<Item = usize>) -> Option<Self> {
        let shards: Vec<_> = shard_sizes
            .into_iter()
            .filter(|&capacity| capacity != 0)
            .map(|capacity| ShardState {
                capacity,
                free: BTreeMap::from([(0, capacity)]),
            })
            .collect();
        if shards.is_empty() {
            return None;
        }
        Some(Self {
            inner: Arc::new(PoolInner {
                state: Mutex::new(PoolState {
                    shards,
                    live: HashMap::new(),
                    next_id: 1,
                    bytes_by_class: [0; UnifiedVramClass::COUNT],
                }),
                generation: AtomicU64::new(0),
            }),
        })
    }

    /// Allocate inside one shard. Best-fit limits fragmentation without moving live allocations.
    pub fn allocate(
        &self,
        requested_len: usize,
        align: usize,
        class: UnifiedVramClass,
    ) -> Option<Arc<UnifiedAllocation>> {
        self.allocate_with_policy(requested_len, align, class, true)
    }

    fn allocate_first_fit(
        &self,
        requested_len: usize,
        align: usize,
        class: UnifiedVramClass,
    ) -> Option<Arc<UnifiedAllocation>> {
        self.allocate_with_policy(requested_len, align, class, false)
    }

    fn allocate_with_policy(
        &self,
        requested_len: usize,
        align: usize,
        class: UnifiedVramClass,
        best_fit: bool,
    ) -> Option<Arc<UnifiedAllocation>> {
        if requested_len == 0 || align == 0 || !align.is_power_of_two() {
            return None;
        }
        let len = align_up(requested_len, align)?;
        let mut state = self.inner.state.lock().unwrap();
        let mut best: Option<(usize, usize, usize, usize)> = None;
        'shards: for (shard_idx, shard) in state.shards.iter().enumerate() {
            for (&start, &span) in &shard.free {
                let aligned = align_up(start, align)?;
                let end = aligned.checked_add(len)?;
                let range_end = start.checked_add(span)?;
                if end > range_end {
                    continue;
                }
                let waste = span - len;
                let candidate = (waste, shard_idx, start, aligned);
                if best.is_none_or(|old| candidate < old) {
                    best = Some(candidate);
                }
                if !best_fit {
                    break 'shards;
                }
            }
        }
        let (_, shard, free_start, offset) = best?;
        take_range(&mut state.shards[shard].free, free_start, offset, len);
        Some(make_allocation(
            &self.inner,
            &mut state,
            shard,
            offset,
            len,
            requested_len,
            class,
        ))
    }

    /// Reclaim a known slot after a borrower released it. Fails rather than moving another range.
    pub fn try_claim_exact(
        &self,
        shard: usize,
        offset: usize,
        len: usize,
        class: UnifiedVramClass,
    ) -> Option<Arc<UnifiedAllocation>> {
        if len == 0 {
            return None;
        }
        let mut state = self.inner.state.lock().unwrap();
        let shard_state = state.shards.get_mut(shard)?;
        let (&free_start, &free_len) = shard_state.free.range(..=offset).next_back()?;
        let free_end = free_start.checked_add(free_len)?;
        let end = offset.checked_add(len)?;
        if end > free_end {
            return None;
        }
        take_range(&mut shard_state.free, free_start, offset, len);
        Some(make_allocation(
            &self.inner,
            &mut state,
            shard,
            offset,
            len,
            len,
            class,
        ))
    }

    pub fn generation(&self) -> u64 {
        self.inner.generation.load(Ordering::Acquire)
    }

    pub fn allocations(&self) -> Vec<UnifiedRange> {
        let state = self.inner.state.lock().unwrap();
        let mut result: Vec<_> = state.live.values().copied().collect();
        result.sort_by_key(|range| (range.shard, range.offset));
        result
    }

    pub fn stats(&self) -> UnifiedVramStats {
        let state = self.inner.state.lock().unwrap();
        let capacity_bytes = state.shards.iter().map(|shard| shard.capacity).sum();
        let free_bytes = state
            .shards
            .iter()
            .flat_map(|shard| shard.free.values())
            .sum();
        let largest_free_bytes = state
            .shards
            .iter()
            .flat_map(|shard| shard.free.values())
            .copied()
            .max()
            .unwrap_or(0);
        UnifiedVramStats {
            capacity_bytes,
            allocated_bytes: capacity_bytes - free_bytes,
            free_bytes,
            largest_free_bytes,
            live_allocations: state.live.len(),
            bytes_by_class: state.bytes_by_class,
        }
    }
}

/// One physical mapped-ReBAR shard. The shard owns the Vulkan allocation while logical leases
/// merely name byte ranges inside it.
struct UnifiedVramShard {
    buffer: Arc<dyn Buffer>,
    base_addr: u64,
    mapped_ptr: usize,
    bytes: usize,
}

/// A Vulkan-backed range lease. Keeping the physical shard and logical lease in the same handle
/// lets a `VkBuffer` view outlive the backend handle without forming a cycle through
/// `VulkanShared`.
pub(crate) struct UnifiedAllocationHandle {
    lease: Arc<UnifiedAllocation>,
    shard: Arc<UnifiedVramShard>,
}

impl UnifiedAllocationHandle {
    pub(crate) fn range(&self) -> UnifiedRange {
        self.lease.range()
    }

    pub(crate) fn buffer(&self) -> &dyn Buffer {
        self.shard.buffer.as_ref()
    }

    pub(crate) fn buffer_arc(&self) -> Arc<dyn Buffer> {
        Arc::clone(&self.shard.buffer)
    }

    pub(crate) fn base_addr(&self) -> u64 {
        self.shard.base_addr
    }

    pub(crate) fn mapped_ptr(&self) -> *mut u8 {
        self.shard.mapped_ptr as *mut u8
    }

    pub(crate) fn shard_bytes(&self) -> usize {
        self.shard.bytes
    }
}

/// Physical elastic VRAM arena. Shards work around Windows/driver single-allocation limits while
/// `ranges` exposes one allocation policy and one accounting surface across all of them.
pub struct UnifiedVramPool {
    ranges: UnifiedRangePool,
    shards: Vec<Arc<UnifiedVramShard>>,
}

impl UnifiedVramPool {
    pub(crate) fn new(vk: &VulkanBackend, capacity: usize) -> Result<Arc<Self>> {
        if capacity == 0 {
            return Err(be("unified VRAM arena cannot have zero capacity"));
        }
        const WINDOWS_MAX_SHARD: usize = 3 * 1024 * 1024 * 1024;
        let driver_max = usize::try_from(vk.shared.max_mem_alloc_size)
            .unwrap_or(usize::MAX)
            .max(256);
        let platform_max = if cfg!(target_os = "windows") {
            WINDOWS_MAX_SHARD
        } else {
            driver_max
        };
        let max_shard = platform_max.min(driver_max) / 256 * 256;
        let mut remaining = capacity;
        let mut shard_sizes = Vec::new();
        while remaining != 0 {
            let bytes = remaining.min(max_shard);
            shard_sizes.push(bytes);
            remaining -= bytes;
        }
        Self::new_with_shards(vk, &shard_sizes)
    }

    pub(crate) fn new_with_shards(vk: &VulkanBackend, shard_sizes: &[usize]) -> Result<Arc<Self>> {
        if shard_sizes.is_empty() || shard_sizes.contains(&0) {
            return Err(be("unified VRAM arena needs non-empty physical shards"));
        }
        let mut shards = Vec::new();
        for &bytes in shard_sizes {
            let (buffer, base_addr) = vk.alloc_mapped_arena_bda(bytes)?;
            let buffer: Arc<dyn Buffer> = Arc::from(buffer);
            let mapped_ptr = as_vk_buf(buffer.as_ref())?
                .mapped_ptr()
                .ok_or_else(|| be("unified VRAM shard is not persistently mapped"))?
                as usize;
            shards.push(Arc::new(UnifiedVramShard {
                buffer,
                base_addr,
                mapped_ptr,
                bytes,
            }));
        }
        let ranges = UnifiedRangePool::new(shards.iter().map(|shard| shard.bytes))
            .ok_or_else(|| be("unified VRAM arena has no physical shards"))?;
        tracing::info!(
            "[infr] unified VRAM arena: {} bytes across {} mapped ReBAR shard(s)",
            shard_sizes.iter().sum::<usize>(),
            shards.len(),
        );
        Ok(Arc::new(Self { ranges, shards }))
    }

    pub(crate) fn allocate(
        &self,
        bytes: usize,
        class: UnifiedVramClass,
    ) -> Option<Arc<UnifiedAllocationHandle>> {
        let lease = if class == UnifiedVramClass::Expert {
            self.ranges.allocate_first_fit(bytes, 256, class)?
        } else {
            self.ranges.allocate(bytes, 256, class)?
        };
        let shard = Arc::clone(self.shards.get(lease.range().shard)?);
        Some(Arc::new(UnifiedAllocationHandle { lease, shard }))
    }

    pub(crate) fn try_claim_exact(
        &self,
        shard: usize,
        offset: usize,
        bytes: usize,
        class: UnifiedVramClass,
    ) -> Option<Arc<UnifiedAllocationHandle>> {
        let lease = self.ranges.try_claim_exact(shard, offset, bytes, class)?;
        let physical = Arc::clone(self.shards.get(shard)?);
        Some(Arc::new(UnifiedAllocationHandle {
            lease,
            shard: physical,
        }))
    }

    pub fn generation(&self) -> u64 {
        self.ranges.generation()
    }

    pub fn stats(&self) -> UnifiedVramStats {
        self.ranges.stats()
    }

    pub fn allocations(&self) -> Vec<UnifiedRange> {
        self.ranges.allocations()
    }

    pub fn shard_sizes(&self) -> Vec<usize> {
        self.shards.iter().map(|shard| shard.bytes).collect()
    }
}

fn make_allocation(
    inner: &Arc<PoolInner>,
    state: &mut PoolState,
    shard: usize,
    offset: usize,
    len: usize,
    requested_len: usize,
    class: UnifiedVramClass,
) -> Arc<UnifiedAllocation> {
    let id = state.next_id;
    state.next_id = state.next_id.wrapping_add(1).max(1);
    let range = UnifiedRange {
        id,
        shard,
        offset,
        len,
        requested_len,
        class,
    };
    state.live.insert(id, range);
    state.bytes_by_class[class.index()] += len;
    inner.generation.fetch_add(1, Ordering::Release);
    Arc::new(UnifiedAllocation {
        range,
        owner: Arc::downgrade(inner),
    })
}

fn align_up(value: usize, align: usize) -> Option<usize> {
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

fn take_range(free: &mut BTreeMap<usize, usize>, free_start: usize, offset: usize, len: usize) {
    let old_len = free
        .remove(&free_start)
        .expect("selected free range disappeared while locked");
    let old_end = free_start + old_len;
    if offset > free_start {
        free.insert(free_start, offset - free_start);
    }
    let end = offset + len;
    if end < old_end {
        free.insert(end, old_end - end);
    }
}

fn insert_free(free: &mut BTreeMap<usize, usize>, offset: usize, len: usize) {
    let mut start = offset;
    let mut end = offset + len;
    if let Some((&prev_start, &prev_len)) = free.range(..=offset).next_back() {
        let prev_end = prev_start + prev_len;
        debug_assert!(
            prev_end <= offset,
            "released range overlaps previous free range"
        );
        if prev_end == offset {
            start = prev_start;
            free.remove(&prev_start);
        }
    }
    if let Some((&next_start, &next_len)) = free.range(offset..).next() {
        debug_assert!(end <= next_start, "released range overlaps next free range");
        if end == next_start {
            end = next_start + next_len;
            free.remove(&next_start);
        }
    }
    free.insert(start, end - start);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_best_fit_and_drop_coalesce() {
        let pool = UnifiedRangePool::new([1024, 512]).unwrap();
        let a = pool.allocate(100, 256, UnifiedVramClass::Expert).unwrap();
        let b = pool
            .allocate(200, 256, UnifiedVramClass::EmbeddingWeights)
            .unwrap();
        assert_eq!(
            (a.range().shard, a.range().offset, a.range().len),
            (1, 0, 256)
        );
        assert_eq!(
            (b.range().shard, b.range().offset, b.range().len),
            (1, 256, 256)
        );
        assert_eq!(pool.stats().largest_free_bytes, 1024);
        drop(a);
        drop(b);
        let stats = pool.stats();
        assert_eq!(stats.free_bytes, 1536);
        assert_eq!(stats.largest_free_bytes, 1024);
        assert_eq!(stats.fragmentation(), 1.0 - 1024.0 / 1536.0);
    }

    #[test]
    fn final_arc_releases_and_updates_class_accounting() {
        let pool = UnifiedRangePool::new([1024]).unwrap();
        let allocation = pool
            .allocate(300, 64, UnifiedVramClass::EmbeddingRuntime)
            .unwrap();
        let keepalive = Arc::clone(&allocation);
        assert_eq!(
            pool.stats().class_bytes(UnifiedVramClass::EmbeddingRuntime),
            320
        );
        drop(allocation);
        assert_eq!(pool.stats().allocated_bytes, 320);
        drop(keepalive);
        assert_eq!(pool.stats().allocated_bytes, 0);
    }

    #[test]
    fn exact_claim_restores_released_slot_only_when_free() {
        let pool = UnifiedRangePool::new([1024]).unwrap();
        let slot = pool
            .try_claim_exact(0, 256, 256, UnifiedVramClass::Expert)
            .unwrap();
        assert!(pool
            .try_claim_exact(0, 256, 256, UnifiedVramClass::Expert)
            .is_none());
        drop(slot);
        let restored = pool
            .try_claim_exact(0, 256, 256, UnifiedVramClass::Expert)
            .unwrap();
        assert_eq!(restored.range().offset, 256);
    }

    #[test]
    fn allocations_never_cross_shards() {
        let pool = UnifiedRangePool::new([512, 512]).unwrap();
        assert!(pool
            .allocate(768, 256, UnifiedVramClass::EmbeddingWeights)
            .is_none());
        assert_eq!(pool.stats().allocated_bytes, 0);
    }

    #[test]
    fn generation_tracks_topology_changes() {
        let pool = UnifiedRangePool::new([1024]).unwrap();
        let before = pool.generation();
        let allocation = pool.allocate(64, 64, UnifiedVramClass::Expert).unwrap();
        let allocated = pool.generation();
        assert!(allocated > before);
        drop(allocation);
        assert!(pool.generation() > allocated);
    }
}
