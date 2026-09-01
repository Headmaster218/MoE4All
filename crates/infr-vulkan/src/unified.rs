//! Logical sub-allocation for the elastic VRAM arena shared by paged experts and auxiliary
//! engines. The physical Vulkan backing is supplied by `arena.rs`; keeping range
//! bookkeeping independent makes alignment, coalescing, accounting and exact slot restoration
//! testable without a GPU.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use infr_core::backend::Buffer;
use infr_core::error::Result;

use super::{be, VulkanBackend};
use crate::arena::{DeviceArena, DeviceArenaShard};

const UNIFIED_ALIGN: usize = 256;

/// Stable identity of one expert filler cell. Pool-local slot numbers remain the pager's cache
/// indices; this pair is the only identity the global arena layout needs in order to report which
/// cells a higher-priority range will cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExpertSlotId {
    pub pool: usize,
    pub slot: usize,
}

/// One expert cell's immutable physical coordinates inside a large arena shard. Cells from
/// different size classes may be interleaved, but no cell crosses a Vulkan allocation boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertSlotPlacement {
    pub id: ExpertSlotId,
    pub logical_offset: usize,
    pub shard: usize,
    pub offset: usize,
    pub len: usize,
}

/// Load-time geometry of the elastic VRAM arena. Physical storage remains a small number of large
/// Vulkan shards; the entries below are only a logical filler directory over those shards.
///
/// The arena has three priority corridors:
///
/// - `[0, kv_reserve_bytes)` is reserved for lazy low-address KV growth;
/// - the middle corridor is available to the fixed Prefill lane while active;
/// - the high suffix is the planned runtime reserve.
///
/// Expert cells initially fill all three corridors and are disabled only as their bytes are
/// claimed by a higher-priority owner.
#[derive(Clone, Debug)]
pub(crate) struct ExpertArenaLayout {
    shard_sizes: Vec<usize>,
    slots_by_pool: Vec<Vec<ExpertSlotPlacement>>,
    total_bytes: usize,
    kv_reserve_bytes: usize,
    runtime_reserve_bytes: usize,
}

impl ExpertArenaLayout {
    pub(crate) fn build(
        specs: &[(usize, usize)],
        max_shard: usize,
        kv_reserve_bytes: usize,
        runtime_reserve_bytes: usize,
    ) -> Result<Self> {
        if specs.is_empty() || max_shard < UNIFIED_ALIGN {
            return Err(be(
                "expert arena layout needs pools and a usable shard limit",
            ));
        }
        let mut total_bytes = 0usize;
        let mut total_slots = 0usize;
        for &(slot_bytes, n_slots) in specs {
            if slot_bytes == 0
                || n_slots == 0
                || !slot_bytes.is_multiple_of(UNIFIED_ALIGN)
                || slot_bytes > max_shard
            {
                return Err(be(format!(
                    "invalid expert arena pool: {n_slots} slot(s) x {slot_bytes} bytes, shard limit {max_shard}"
                )));
            }
            total_bytes = total_bytes
                .checked_add(
                    slot_bytes
                        .checked_mul(n_slots)
                        .ok_or_else(|| be("expert arena pool byte size overflow"))?,
                )
                .ok_or_else(|| be("expert arena total byte size overflow"))?;
            total_slots = total_slots
                .checked_add(n_slots)
                .ok_or_else(|| be("expert arena slot count overflow"))?;
        }
        let kv_reserve_bytes = align_up(kv_reserve_bytes, UNIFIED_ALIGN)
            .ok_or_else(|| be("KV reserve alignment overflow"))?;
        let runtime_reserve_bytes = align_up(runtime_reserve_bytes, UNIFIED_ALIGN)
            .ok_or_else(|| be("runtime reserve alignment overflow"))?;
        if kv_reserve_bytes.saturating_add(runtime_reserve_bytes) > total_bytes {
            return Err(be(format!(
                "elastic arena has {total_bytes} bytes, below its {kv_reserve_bytes}-byte KV and {runtime_reserve_bytes}-byte runtime reserves"
            )));
        }

        // Smooth weighted round-robin preserves each planner-selected pool's slot fraction in
        // every broad physical prefix. A growing low-address KV frontier therefore sheds all size
        // classes proportionally instead of exhausting whichever class happened to be allocated
        // first. The weights are the final slot counts, not raw expert counts.
        let weights: Vec<i128> = specs.iter().map(|&(_, n)| n as i128).collect();
        let total_weight = total_slots as i128;
        let mut current = vec![0i128; specs.len()];
        let mut remaining: Vec<usize> = specs.iter().map(|&(_, n)| n).collect();
        let mut slots_by_pool: Vec<Vec<ExpertSlotPlacement>> =
            specs.iter().map(|&(_, n)| Vec::with_capacity(n)).collect();
        let mut shard_sizes = Vec::new();
        let mut shard = 0usize;
        let mut shard_offset = 0usize;
        let mut logical_offset = 0usize;

        for _ in 0..total_slots {
            let mut selected = None;
            for pool in 0..specs.len() {
                if remaining[pool] == 0 {
                    continue;
                }
                current[pool] += weights[pool];
                if selected.is_none_or(|old| {
                    current[pool] > current[old] || (current[pool] == current[old] && pool < old)
                }) {
                    selected = Some(pool);
                }
            }
            let pool = selected.expect("total_slots tracks every remaining expert cell");
            current[pool] -= total_weight;
            remaining[pool] -= 1;

            let slot_bytes = specs[pool].0;
            if shard_offset != 0 && shard_offset.saturating_add(slot_bytes) > max_shard {
                shard_sizes.push(shard_offset);
                shard += 1;
                shard_offset = 0;
            }
            let slot = slots_by_pool[pool].len();
            slots_by_pool[pool].push(ExpertSlotPlacement {
                id: ExpertSlotId { pool, slot },
                logical_offset,
                shard,
                offset: shard_offset,
                len: slot_bytes,
            });
            shard_offset += slot_bytes;
            logical_offset += slot_bytes;
        }
        if shard_offset != 0 {
            shard_sizes.push(shard_offset);
        }
        debug_assert_eq!(logical_offset, total_bytes);
        debug_assert!(remaining.iter().all(|&n| n == 0));

        Ok(Self {
            shard_sizes,
            slots_by_pool,
            total_bytes,
            kv_reserve_bytes,
            runtime_reserve_bytes,
        })
    }

    pub(crate) fn shard_sizes(&self) -> &[usize] {
        &self.shard_sizes
    }

    pub(crate) fn slots(&self, pool: usize) -> Option<&[ExpertSlotPlacement]> {
        self.slots_by_pool.get(pool).map(Vec::as_slice)
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub(crate) fn kv_corridor(&self) -> Range<usize> {
        0..self.kv_reserve_bytes
    }

    pub(crate) fn prefill_corridor(&self) -> Range<usize> {
        self.kv_reserve_bytes..self.total_bytes - self.runtime_reserve_bytes
    }

    pub(crate) fn runtime_corridor(&self) -> Range<usize> {
        self.total_bytes - self.runtime_reserve_bytes..self.total_bytes
    }

    fn physical_pieces(&self, logical: Range<usize>) -> Vec<ArenaPiece> {
        let mut pieces = Vec::new();
        let mut base = 0usize;
        for (shard, &size) in self.shard_sizes.iter().enumerate() {
            let shard_end = base + size;
            let start = logical.start.max(base);
            let end = logical.end.min(shard_end);
            if start < end {
                pieces.push(ArenaPiece {
                    shard,
                    start: start - base,
                    end: end - base,
                });
            }
            base = shard_end;
        }
        pieces
    }

    fn plan_claim(
        &self,
        live: &[UnifiedRange],
        requested: &[usize],
        class: UnifiedVramClass,
        corridor: Range<usize>,
        direction: ClaimDirection,
        protected_experts: &[ExpertSlotId],
    ) -> Result<UnifiedClaimPlan> {
        if class == UnifiedVramClass::Expert || requested.is_empty() {
            return Err(be("elastic claim needs non-Expert bytes"));
        }
        let live_experts: HashSet<_> = live
            .iter()
            .filter(|range| range.class == UnifiedVramClass::Expert)
            .map(|range| (range.shard, range.offset, range.len))
            .collect();
        let mut blockers: Vec<PlannedRange> = live
            .iter()
            .filter(|range| range.class != UnifiedVramClass::Expert)
            .map(|range| PlannedRange {
                shard: range.shard,
                offset: range.offset,
                len: range.len,
                requested_len: range.requested_len,
            })
            .collect();
        for &id in protected_experts {
            let placement = self
                .slots_by_pool
                .get(id.pool)
                .and_then(|slots| slots.get(id.slot))
                .ok_or_else(|| {
                    be(format!(
                        "protected Expert slot {}/{} is unknown",
                        id.pool, id.slot
                    ))
                })?;
            if live_experts.contains(&(placement.shard, placement.offset, placement.len)) {
                blockers.push(PlannedRange {
                    shard: placement.shard,
                    offset: placement.offset,
                    len: placement.len,
                    requested_len: placement.len,
                });
            }
        }
        let pieces = self.physical_pieces(corridor);
        let mut ordered = Vec::with_capacity(requested.len());
        for (index, &requested_len) in requested.iter().enumerate() {
            if requested_len == 0 {
                return Err(be("elastic claim cannot contain a zero-byte range"));
            }
            let len = align_up(requested_len, UNIFIED_ALIGN)
                .ok_or_else(|| be("elastic claim alignment overflow"))?;
            ordered.push((index, requested_len, len));
        }
        // Packing the largest ranges first avoids stranding a large KV plane, Prefill bank or
        // runtime workspace behind small shard-tail fragments. Restore caller order before commit
        // so returned handles still align exactly with the request slice.
        ordered.sort_unstable_by_key(|&(index, _, len)| (std::cmp::Reverse(len), index));
        let mut ranges = vec![None; requested.len()];
        for (index, requested_len, len) in ordered {
            let candidate = match direction {
                ClaimDirection::Low => find_low_gap(&pieces, &blockers, len),
                ClaimDirection::High => find_high_gap(&pieces, &blockers, len),
            }
            .ok_or_else(|| {
                be(format!(
                    "{class:?} cannot fit {len} contiguous bytes in its frozen arena corridor"
                ))
            })?;
            let range = PlannedRange {
                requested_len,
                ..candidate
            };
            blockers.push(range);
            ranges[index] = Some(range);
        }
        let ranges: Vec<_> = ranges
            .into_iter()
            .map(|range| range.expect("every validated elastic request was planned"))
            .collect();

        let mut victims = Vec::new();
        for placement in self.slots_by_pool.iter().flatten() {
            if !live_experts.contains(&(placement.shard, placement.offset, placement.len)) {
                continue;
            }
            if ranges.iter().any(|range| {
                range.shard == placement.shard
                    && ranges_overlap(range.offset, range.len, placement.offset, placement.len)
            }) {
                victims.push(placement.id);
            }
        }
        let mapped_experts = self
            .slots_by_pool
            .iter()
            .flatten()
            .filter(|placement| {
                live_experts.contains(&(placement.shard, placement.offset, placement.len))
            })
            .count();
        if mapped_experts != live_experts.len() {
            return Err(be(
                "elastic arena contains an Expert allocation outside its frozen slot directory",
            ));
        }
        victims.sort_unstable_by_key(|id| (id.pool, id.slot));
        victims.dedup();
        Ok(UnifiedClaimPlan {
            class,
            ranges,
            victims,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ArenaPiece {
    shard: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
enum ClaimDirection {
    Low,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedRange {
    shard: usize,
    offset: usize,
    len: usize,
    requested_len: usize,
}

/// A side-effect-free arena mutation plan. Pager retirement happens between planning and commit;
/// callers hold the backend's unified execution guard across both operations, so no third party
/// can invalidate the range calculation.
#[derive(Debug)]
pub(crate) struct UnifiedClaimPlan {
    class: UnifiedVramClass,
    ranges: Vec<PlannedRange>,
    victims: Vec<ExpertSlotId>,
}

impl UnifiedClaimPlan {
    pub(crate) fn victims(&self) -> &[ExpertSlotId] {
        &self.victims
    }

    pub(crate) fn len(&self) -> usize {
        self.ranges.len()
    }

    pub(crate) fn class(&self) -> UnifiedVramClass {
        self.class
    }
}

fn find_low_gap(
    pieces: &[ArenaPiece],
    blockers: &[PlannedRange],
    len: usize,
) -> Option<PlannedRange> {
    for piece in pieces {
        let mut occupied: Vec<_> = blockers
            .iter()
            .filter(|range| range.shard == piece.shard)
            .map(|range| (range.offset, range.offset.saturating_add(range.len)))
            .filter(|&(start, end)| start < piece.end && end > piece.start)
            .collect();
        occupied.sort_unstable();
        let mut cursor = align_up(piece.start, UNIFIED_ALIGN)?;
        for (start, end) in occupied {
            if cursor.checked_add(len)? <= start {
                return Some(PlannedRange {
                    shard: piece.shard,
                    offset: cursor,
                    len,
                    requested_len: len,
                });
            }
            cursor = align_up(cursor.max(end), UNIFIED_ALIGN)?;
            if cursor >= piece.end {
                break;
            }
        }
        if cursor.checked_add(len)? <= piece.end {
            return Some(PlannedRange {
                shard: piece.shard,
                offset: cursor,
                len,
                requested_len: len,
            });
        }
    }
    None
}

fn find_high_gap(
    pieces: &[ArenaPiece],
    blockers: &[PlannedRange],
    len: usize,
) -> Option<PlannedRange> {
    for piece in pieces.iter().rev() {
        let mut occupied: Vec<_> = blockers
            .iter()
            .filter(|range| range.shard == piece.shard)
            .map(|range| {
                (
                    range.offset.max(piece.start),
                    range.offset.saturating_add(range.len).min(piece.end),
                )
            })
            .filter(|&(start, end)| start < end)
            .collect();
        occupied.sort_unstable();
        let mut gaps = Vec::with_capacity(occupied.len() + 1);
        let mut cursor = piece.start;
        for (start, end) in occupied {
            if cursor < start {
                gaps.push((cursor, start));
            }
            cursor = cursor.max(end);
        }
        if cursor < piece.end {
            gaps.push((cursor, piece.end));
        }
        for (start, end) in gaps.into_iter().rev() {
            let Some(latest) = end.checked_sub(len) else {
                continue;
            };
            let offset = latest & !(UNIFIED_ALIGN - 1);
            if offset >= start {
                return Some(PlannedRange {
                    shard: piece.shard,
                    offset,
                    len,
                    requested_len: len,
                });
            }
        }
    }
    None
}

fn ranges_overlap(a_offset: usize, a_len: usize, b_offset: usize, b_len: usize) -> bool {
    a_offset < b_offset.saturating_add(b_len) && b_offset < a_offset.saturating_add(a_len)
}

/// Owner of a live range in the unified elastic arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnifiedVramClass {
    Expert,
    KvCache,
    LlmRuntime,
    Prefill,
    EmbeddingWeights,
    EmbeddingRuntime,
    VisionWeights,
    VisionRuntime,
    DraftWeights,
    DraftRuntime,
}

impl UnifiedVramClass {
    const COUNT: usize = 10;

    const fn index(self) -> usize {
        match self {
            Self::Expert => 0,
            Self::KvCache => 1,
            Self::LlmRuntime => 2,
            Self::Prefill => 3,
            Self::EmbeddingWeights => 4,
            Self::EmbeddingRuntime => 5,
            Self::VisionWeights => 6,
            Self::VisionRuntime => 7,
            Self::DraftWeights => 8,
            Self::DraftRuntime => 9,
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

    /// Allocate from the highest fitting address. Variable-sized auxiliary weights and runtime
    /// workspaces grow down from the opposite end of each shard to the fixed-size Expert slots,
    /// so releasing them exposes a coalesced suffix instead of leaving holes throughout the LRU.
    fn allocate_high(
        &self,
        requested_len: usize,
        align: usize,
        class: UnifiedVramClass,
    ) -> Option<Arc<UnifiedAllocation>> {
        if requested_len == 0 || align == 0 || !align.is_power_of_two() {
            return None;
        }
        let len = align_up(requested_len, align)?;
        let mut state = self.inner.state.lock().unwrap();
        let mut selected = None;
        'shards: for shard_idx in (0..state.shards.len()).rev() {
            let shard = &state.shards[shard_idx];
            for (&free_start, &span) in shard.free.iter().rev() {
                let range_end = free_start.checked_add(span)?;
                let Some(latest) = range_end.checked_sub(len) else {
                    continue;
                };
                let offset = latest & !(align - 1);
                if offset >= free_start {
                    selected = Some((shard_idx, free_start, offset));
                    break 'shards;
                }
            }
        }
        let (shard, free_start, offset) = selected?;
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

    fn try_claim_planned(
        &self,
        class: UnifiedVramClass,
        ranges: &[PlannedRange],
    ) -> Option<Vec<Arc<UnifiedAllocation>>> {
        if ranges.is_empty()
            || ranges.iter().any(|range| range.len == 0)
            || ranges.iter().enumerate().any(|(index, range)| {
                ranges[index + 1..].iter().any(|other| {
                    range.shard == other.shard
                        && ranges_overlap(range.offset, range.len, other.offset, other.len)
                })
            })
        {
            return None;
        }
        let mut state = self.inner.state.lock().unwrap();
        for range in ranges {
            let shard = state.shards.get(range.shard)?;
            let (&free_start, &free_len) = shard.free.range(..=range.offset).next_back()?;
            let free_end = free_start.checked_add(free_len)?;
            if range.offset.checked_add(range.len)? > free_end {
                return None;
            }
        }

        let mut out = Vec::with_capacity(ranges.len());
        for range in ranges {
            let (&free_start, _) = state.shards[range.shard]
                .free
                .range(..=range.offset)
                .next_back()
                .expect("the complete claim batch was validated while holding this lock");
            take_range(
                &mut state.shards[range.shard].free,
                free_start,
                range.offset,
                range.len,
            );
            out.push(make_allocation(
                &self.inner,
                &mut state,
                range.shard,
                range.offset,
                range.len,
                range.requested_len,
                class,
            ));
        }
        Some(out)
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

/// A Vulkan-backed range lease. Keeping the physical shard and logical lease in the same handle
/// lets a `VkBuffer` view outlive the backend handle without forming a cycle through
/// `VulkanShared`.
pub(crate) struct UnifiedAllocationHandle {
    lease: Arc<UnifiedAllocation>,
    shard: Arc<DeviceArenaShard>,
}

impl UnifiedAllocationHandle {
    pub(crate) fn range(&self) -> UnifiedRange {
        self.lease.range()
    }

    pub(crate) fn buffer(&self) -> &dyn Buffer {
        self.shard.buffer()
    }

    pub(crate) fn buffer_arc(&self) -> Arc<dyn Buffer> {
        self.shard.buffer_arc()
    }

    pub(crate) fn base_addr(&self) -> u64 {
        self.shard.base_addr()
    }

    pub(crate) fn mapped_ptr(&self) -> Option<*mut u8> {
        self.shard.mapped_ptr()
    }

    pub(crate) fn shard_bytes(&self) -> usize {
        self.shard.bytes()
    }
}

/// Elastic VRAM arena. `ranges` owns placement/accounting while `arena` independently selects
/// mapped or ordinary device-local physical shards.
pub struct UnifiedVramPool {
    ranges: UnifiedRangePool,
    arena: DeviceArena,
    expert_layout: Option<ExpertArenaLayout>,
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
        Self::new_with_layout(vk, shard_sizes, None)
    }

    pub(crate) fn new_for_experts(
        vk: &VulkanBackend,
        layout: ExpertArenaLayout,
    ) -> Result<Arc<Self>> {
        let shard_sizes = layout.shard_sizes().to_vec();
        Self::new_with_layout(vk, &shard_sizes, Some(layout))
    }

    fn new_with_layout(
        vk: &VulkanBackend,
        shard_sizes: &[usize],
        expert_layout: Option<ExpertArenaLayout>,
    ) -> Result<Arc<Self>> {
        if shard_sizes.is_empty() || shard_sizes.contains(&0) {
            return Err(be("unified VRAM arena needs non-empty physical shards"));
        }
        let arena = DeviceArena::new(vk, shard_sizes)?;
        let backing = arena.backing();
        let ranges = UnifiedRangePool::new(arena.shard_sizes())
            .ok_or_else(|| be("unified VRAM arena has no physical shards"))?;
        tracing::info!(
            "[infr] unified VRAM arena: {} bytes across {} shard(s), backing={backing:?}",
            shard_sizes.iter().sum::<usize>(),
            shard_sizes.len(),
        );
        Ok(Arc::new(Self {
            ranges,
            arena,
            expert_layout,
        }))
    }

    pub(crate) fn expert_layout(&self) -> Option<&ExpertArenaLayout> {
        self.expert_layout.as_ref()
    }

    pub(crate) fn claim_expert_slot(
        &self,
        id: ExpertSlotId,
    ) -> Option<Arc<UnifiedAllocationHandle>> {
        let placement = *self.expert_layout.as_ref()?.slots(id.pool)?.get(id.slot)?;
        debug_assert_eq!(placement.id, id);
        let lease = self.ranges.try_claim_exact(
            placement.shard,
            placement.offset,
            placement.len,
            UnifiedVramClass::Expert,
        )?;
        let shard = self.arena.shard(placement.shard)?;
        Some(Arc::new(UnifiedAllocationHandle { lease, shard }))
    }

    pub(crate) fn plan_kv_claim(
        &self,
        requested: &[usize],
        protected_experts: &[ExpertSlotId],
    ) -> Result<UnifiedClaimPlan> {
        let layout = self
            .expert_layout
            .as_ref()
            .ok_or_else(|| be("KV corridor requires an expert-aware unified VRAM layout"))?;
        layout.plan_claim(
            &self.ranges.allocations(),
            requested,
            UnifiedVramClass::KvCache,
            layout.kv_corridor(),
            ClaimDirection::Low,
            protected_experts,
        )
    }

    pub(crate) fn plan_prefill_claim(
        &self,
        requested: &[usize],
        protected_experts: &[ExpertSlotId],
    ) -> Result<UnifiedClaimPlan> {
        let layout = self
            .expert_layout
            .as_ref()
            .ok_or_else(|| be("Prefill corridor requires an expert-aware unified VRAM layout"))?;
        layout.plan_claim(
            &self.ranges.allocations(),
            requested,
            UnifiedVramClass::Prefill,
            layout.prefill_corridor(),
            ClaimDirection::Low,
            protected_experts,
        )
    }

    /// Plan a non-KV, non-Prefill owner from high addresses. The maximum KV corridor remains a
    /// hard lower boundary even before those lazy segments are physically committed.
    pub(crate) fn plan_high_claim(
        &self,
        requested: &[usize],
        class: UnifiedVramClass,
        protected_experts: &[ExpertSlotId],
    ) -> Result<UnifiedClaimPlan> {
        if matches!(
            class,
            UnifiedVramClass::Expert | UnifiedVramClass::KvCache | UnifiedVramClass::Prefill
        ) {
            return Err(be(format!(
                "{class:?} cannot use the high-address elastic claim path"
            )));
        }
        let layout = self
            .expert_layout
            .as_ref()
            .ok_or_else(|| be("high-address claim requires an expert-aware unified VRAM layout"))?;
        layout.plan_claim(
            &self.ranges.allocations(),
            requested,
            class,
            layout.kv_corridor().end..layout.total_bytes(),
            ClaimDirection::High,
            protected_experts,
        )
    }

    /// Commit a previously planned claim as one allocator transaction. Callers must retire every
    /// reported Expert victim first while holding the backend's unified execution gate.
    pub(crate) fn commit_claim(
        &self,
        plan: UnifiedClaimPlan,
    ) -> Result<Vec<Arc<UnifiedAllocationHandle>>> {
        let class = plan.class;
        let leases = self
            .ranges
            .try_claim_planned(class, &plan.ranges)
            .ok_or_else(|| be(format!("stale {class:?} unified VRAM claim plan")))?;
        leases
            .into_iter()
            .map(|lease| {
                let range = lease.range();
                let shard = self
                    .arena
                    .shard(range.shard)
                    .ok_or_else(|| be("planned unified VRAM range has no physical shard"))?;
                Ok(Arc::new(UnifiedAllocationHandle { lease, shard }))
            })
            .collect()
    }

    pub(crate) fn allocate(
        &self,
        bytes: usize,
        class: UnifiedVramClass,
    ) -> Option<Arc<UnifiedAllocationHandle>> {
        let lease = if class == UnifiedVramClass::Expert {
            self.ranges.allocate_first_fit(bytes, 256, class)?
        } else {
            self.ranges.allocate_high(bytes, 256, class)?
        };
        let shard = self.arena.shard(lease.range().shard)?;
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
        let physical = self.arena.shard(shard)?;
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
        self.arena.shard_sizes()
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
    fn expert_layout_interleaves_planner_selected_pool_ratios() {
        let specs = [(512, 2), (256, 1), (256, 17)];
        let layout = ExpertArenaLayout::build(&specs, 4096, 0, 0).unwrap();
        let mut ordered: Vec<_> = layout.slots_by_pool.iter().flatten().copied().collect();
        ordered.sort_unstable_by_key(|slot| slot.logical_offset);
        assert_eq!(ordered.len(), 20);

        let mut seen = vec![0usize; specs.len()];
        for (prefix, slot) in ordered.iter().enumerate() {
            seen[slot.id.pool] += 1;
            let prefix = prefix + 1;
            for (pool, &(_, wanted)) in specs.iter().enumerate() {
                let error = (seen[pool] * ordered.len()) as isize - (prefix * wanted) as isize;
                assert!(
                    error.unsigned_abs() <= ordered.len(),
                    "pool {pool} drifted by more than one slot at prefix {prefix}: {seen:?}"
                );
            }
        }
        assert_eq!(seen, vec![2, 1, 17]);
        let first_pool_positions: Vec<_> = ordered
            .iter()
            .enumerate()
            .filter_map(|(position, slot)| (slot.id.pool == 0).then_some(position))
            .collect();
        assert!(first_pool_positions[1] - first_pool_positions[0] > 1);
    }

    #[test]
    fn expert_layout_uses_large_shards_without_crossing_or_holes() {
        let specs = [(1024, 5), (512, 7), (256, 9)];
        let layout = ExpertArenaLayout::build(&specs, 2048, 0, 0).unwrap();
        assert!(layout.shard_sizes().iter().all(|&bytes| bytes <= 2048));
        assert_eq!(
            layout.shard_sizes().iter().sum::<usize>(),
            layout.total_bytes()
        );

        for (shard, &size) in layout.shard_sizes().iter().enumerate() {
            let mut ranges: Vec<_> = layout
                .slots_by_pool
                .iter()
                .flatten()
                .filter(|slot| slot.shard == shard)
                .map(|slot| (slot.offset, slot.offset + slot.len))
                .collect();
            ranges.sort_unstable();
            assert_eq!(ranges.first().map(|range| range.0), Some(0));
            assert_eq!(ranges.last().map(|range| range.1), Some(size));
            assert!(ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));
        }
    }

    #[test]
    fn arena_corridors_align_and_leave_the_middle_for_experts_and_prefill() {
        let layout = ExpertArenaLayout::build(&[(256, 20)], 4096, 257, 513).unwrap();
        assert_eq!(layout.kv_corridor(), 0..512);
        assert_eq!(layout.runtime_corridor(), 4352..5120);
        assert_eq!(layout.prefill_corridor(), 512..4352);
    }

    #[test]
    fn arena_corridors_reject_reserves_larger_than_the_physical_pool() {
        let error = ExpertArenaLayout::build(&[(256, 4)], 4096, 768, 512).unwrap_err();
        assert!(error.to_string().contains("below its"));
    }

    #[test]
    fn low_kv_claim_reports_exact_interleaved_expert_victims() {
        let layout = ExpertArenaLayout::build(&[(256, 8), (512, 4)], 4096, 1024, 512).unwrap();
        let ranges = UnifiedRangePool::new(layout.shard_sizes().iter().copied()).unwrap();
        let leases: Vec<_> = layout
            .slots_by_pool
            .iter()
            .flatten()
            .map(|slot| {
                ranges
                    .try_claim_exact(slot.shard, slot.offset, slot.len, UnifiedVramClass::Expert)
                    .unwrap()
            })
            .collect();
        let plan = layout
            .plan_claim(
                &ranges.allocations(),
                &[300, 300],
                UnifiedVramClass::KvCache,
                layout.kv_corridor(),
                ClaimDirection::Low,
                &[],
            )
            .unwrap();
        assert_eq!(
            plan.ranges
                .iter()
                .map(|range| range.len)
                .collect::<Vec<_>>(),
            vec![512, 512]
        );
        assert_eq!(
            plan.victims,
            vec![
                ExpertSlotId { pool: 0, slot: 0 },
                ExpertSlotId { pool: 0, slot: 1 },
                ExpertSlotId { pool: 1, slot: 0 },
            ]
        );
        drop(leases);
    }

    #[test]
    fn protected_exchange_slot_is_a_hard_claim_blocker() {
        let layout = ExpertArenaLayout::build(&[(256, 8), (512, 4)], 4096, 1024, 512).unwrap();
        let ranges = UnifiedRangePool::new(layout.shard_sizes().iter().copied()).unwrap();
        let _leases: Vec<_> = layout
            .slots_by_pool
            .iter()
            .flatten()
            .map(|slot| {
                ranges
                    .try_claim_exact(slot.shard, slot.offset, slot.len, UnifiedVramClass::Expert)
                    .unwrap()
            })
            .collect();
        let exchange = ExpertSlotId { pool: 1, slot: 0 };
        let error = layout
            .plan_claim(
                &ranges.allocations(),
                &[512],
                UnifiedVramClass::KvCache,
                layout.kv_corridor(),
                ClaimDirection::Low,
                &[exchange],
            )
            .unwrap_err();
        assert!(error.to_string().contains("cannot fit"));
    }

    #[test]
    fn high_claim_never_enters_the_maximum_kv_corridor() {
        let layout = ExpertArenaLayout::build(&[(256, 16)], 4096, 1024, 512).unwrap();
        let plan = layout
            .plan_claim(
                &[],
                &[300],
                UnifiedVramClass::LlmRuntime,
                layout.kv_corridor().end..layout.total_bytes(),
                ClaimDirection::High,
                &[],
            )
            .unwrap();
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(plan.ranges[0].offset, layout.total_bytes() - 512);
        assert!(plan.ranges[0].offset >= layout.kv_corridor().end);
    }

    #[test]
    fn planned_claim_is_all_or_nothing() {
        let pool = UnifiedRangePool::new([1024]).unwrap();
        let first = pool
            .try_claim_exact(0, 0, 256, UnifiedVramClass::Expert)
            .unwrap();
        let second = pool
            .try_claim_exact(0, 256, 256, UnifiedVramClass::Expert)
            .unwrap();
        drop(first);
        let plan = [
            PlannedRange {
                shard: 0,
                offset: 0,
                len: 256,
                requested_len: 200,
            },
            PlannedRange {
                shard: 0,
                offset: 256,
                len: 256,
                requested_len: 200,
            },
        ];
        assert!(pool
            .try_claim_planned(UnifiedVramClass::LlmRuntime, &plan)
            .is_none());
        let stats = pool.stats();
        assert_eq!(stats.class_bytes(UnifiedVramClass::LlmRuntime), 0);
        assert_eq!(stats.free_bytes, 768);

        drop(second);
        let claimed = pool
            .try_claim_planned(UnifiedVramClass::LlmRuntime, &plan)
            .unwrap();
        assert_eq!(claimed.len(), 2);
    }

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

    #[test]
    fn experts_grow_low_and_variable_objects_grow_high() {
        let pool = UnifiedRangePool::new([4096]).unwrap();
        let expert = pool
            .allocate_first_fit(512, 256, UnifiedVramClass::Expert)
            .unwrap();
        let weights = pool
            .allocate_high(768, 256, UnifiedVramClass::EmbeddingWeights)
            .unwrap();
        let runtime = pool
            .allocate_high(256, 256, UnifiedVramClass::EmbeddingRuntime)
            .unwrap();
        assert_eq!(expert.range().offset, 0);
        assert_eq!(weights.range().offset, 4096 - 768);
        assert_eq!(runtime.range().offset, 4096 - 768 - 256);
    }

    #[test]
    fn high_allocation_skips_an_undersized_higher_range() {
        let pool = UnifiedRangePool::new([1024]).unwrap();
        let barrier = pool
            .try_claim_exact(0, 512, 256, UnifiedVramClass::Expert)
            .unwrap();
        let allocation = pool
            .allocate_high(512, 256, UnifiedVramClass::EmbeddingWeights)
            .unwrap();
        assert_eq!(allocation.range().offset, 0);
        assert_eq!(allocation.range().len, 512);
        drop(barrier);
    }
}
