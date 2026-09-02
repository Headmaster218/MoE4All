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
/// The arena has four priority corridors:
///
/// - `[0, kv_reserve_bytes)` is reserved for lazy low-address KV growth;
/// - one middle band retains every pool's physical dispatch floor;
/// - the remaining middle corridor is available to the fixed Prefill lane while active;
/// - the high suffix is the planned runtime reserve.
///
/// Surplus Expert cells initially fill the elastic corridors and are disabled only as their bytes
/// are claimed by a higher-priority owner. Floor cells never enter those corridors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpertArenaLayout {
    shard_sizes: Vec<usize>,
    slots_by_pool: Vec<Vec<ExpertSlotPlacement>>,
    total_bytes: usize,
    kv_reserve_bytes: usize,
    runtime_reserve_bytes: usize,
    floor_corridor: Range<usize>,
}

fn smooth_weighted_pool_order(counts: &[usize]) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return Vec::new();
    }
    let weights: Vec<i128> = counts.iter().map(|&count| count as i128).collect();
    let total_weight = total as i128;
    let mut current = vec![0i128; counts.len()];
    let mut remaining = counts.to_vec();
    let mut order = Vec::with_capacity(total);
    for _ in 0..total {
        let mut selected = None;
        for pool in 0..counts.len() {
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
        let pool = selected.expect("total tracks every remaining weighted item");
        current[pool] -= total_weight;
        remaining[pool] -= 1;
        order.push(pool);
    }
    order
}

impl ExpertArenaLayout {
    pub(crate) fn build(
        specs: &[(usize, usize, usize)],
        max_shard: usize,
        kv_reserve_bytes: usize,
        kv_max_allocation_bytes: usize,
        prefill_min_lane_bytes: usize,
        runtime_reserve_bytes: usize,
    ) -> Result<Self> {
        if specs.is_empty() || max_shard < UNIFIED_ALIGN {
            return Err(be(
                "expert arena layout needs pools and a usable shard limit",
            ));
        }
        let mut total_bytes = 0usize;
        let mut total_slots = 0usize;
        let mut floor_bytes = 0usize;
        for &(slot_bytes, n_slots, floor_slots) in specs {
            if slot_bytes == 0
                || n_slots == 0
                || floor_slots > n_slots
                || !slot_bytes.is_multiple_of(UNIFIED_ALIGN)
                || slot_bytes > max_shard
            {
                return Err(be(format!(
                    "invalid expert arena pool: {n_slots} slot(s), {floor_slots} protected, x {slot_bytes} bytes, shard limit {max_shard}"
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
            floor_bytes = floor_bytes
                .checked_add(
                    slot_bytes
                        .checked_mul(floor_slots)
                        .ok_or_else(|| be("expert arena floor byte size overflow"))?,
                )
                .ok_or_else(|| be("expert arena total floor byte size overflow"))?;
        }
        let kv_reserve_bytes = align_up(kv_reserve_bytes, UNIFIED_ALIGN)
            .ok_or_else(|| be("KV reserve alignment overflow"))?;
        let kv_max_allocation_bytes = align_up(kv_max_allocation_bytes, UNIFIED_ALIGN)
            .ok_or_else(|| be("maximum KV allocation alignment overflow"))?;
        let prefill_min_lane_bytes = align_up(prefill_min_lane_bytes, UNIFIED_ALIGN)
            .ok_or_else(|| be("minimum Prefill lane alignment overflow"))?;
        let runtime_reserve_bytes = align_up(runtime_reserve_bytes, UNIFIED_ALIGN)
            .ok_or_else(|| be("runtime reserve alignment overflow"))?;
        if kv_max_allocation_bytes > max_shard {
            return Err(be(format!(
                "maximum KV segment is {kv_max_allocation_bytes} bytes, above the {max_shard}-byte Vulkan arena shard limit"
            )));
        }
        // The logical KV estimate is byte-exact, while every physical segment must fit wholly in
        // one Vulkan allocation. Expert-sized shard tails can therefore strand less than one
        // largest KV segment per shard. Price that packing loss inside the existing arena instead
        // of discovering it only at the final 32K growth boundary.
        let max_slot_bytes = specs
            .iter()
            .map(|&(slot_bytes, _, _)| slot_bytes)
            .max()
            .unwrap_or(0);
        let minimum_shard_payload = max_shard.saturating_sub(max_slot_bytes).max(UNIFIED_ALIGN);
        let guaranteed_kv_payload = minimum_shard_payload
            .saturating_sub(kv_max_allocation_bytes)
            .max(UNIFIED_ALIGN);
        let packing_shards = if kv_reserve_bytes == 0 || kv_max_allocation_bytes == 0 {
            0
        } else {
            kv_reserve_bytes.div_ceil(guaranteed_kv_payload)
        };
        let kv_packing_slack = packing_shards
            .checked_mul(kv_max_allocation_bytes)
            .ok_or_else(|| be("KV shard-packing reserve overflow"))?;
        let kv_corridor_target = kv_reserve_bytes
            .checked_add(kv_packing_slack)
            .ok_or_else(|| be("KV corridor size overflow"))?;
        let loanable_bytes = total_bytes.saturating_sub(floor_bytes);
        if kv_corridor_target.saturating_add(runtime_reserve_bytes) > loanable_bytes {
            return Err(be(format!(
                "elastic arena has {loanable_bytes} loanable bytes after its {floor_bytes}-byte expert floor, below its {kv_reserve_bytes}-byte KV reserve, {kv_packing_slack}-byte KV packing reserve and {runtime_reserve_bytes}-byte runtime reserve"
            )));
        }

        // Only slots above each pool's dispatch floor may enter an elastic corridor. Interleave
        // that surplus in weighted-fair order, put enough at low addresses for maximum KV, keep
        // every physical floor together in the middle, then leave the remaining surplus at high
        // addresses for Prefill/runtime owners. A pool already at its floor therefore cannot be
        // clipped by an unrelated KV claim merely because its cells happened to land in a broad
        // weighted prefix.
        let loanable_counts: Vec<_> = specs
            .iter()
            .map(|&(_, n_slots, floor_slots)| n_slots - floor_slots)
            .collect();
        let mut loanable_order = smooth_weighted_pool_order(&loanable_counts);
        let mut low_bytes = 0usize;
        let mut low_count = 0usize;
        while low_bytes < kv_corridor_target && low_count < loanable_order.len() {
            low_bytes = low_bytes
                .checked_add(specs[loanable_order[low_count]].0)
                .ok_or_else(|| be("low-address expert surplus size overflow"))?;
            low_count += 1;
        }
        if low_bytes < kv_corridor_target
            || loanable_bytes.saturating_sub(low_bytes) < runtime_reserve_bytes
        {
            return Err(be(format!(
                "expert slot granularity cannot split {loanable_bytes} loanable bytes into a {kv_corridor_target}-byte packed-KV prefix and {runtime_reserve_bytes}-byte runtime suffix"
            )));
        }
        let prefill_bytes = floor_bytes
            .checked_add(
                loanable_bytes
                    .saturating_sub(low_bytes)
                    .saturating_sub(runtime_reserve_bytes),
            )
            .ok_or_else(|| be("Prefill corridor size overflow"))?;
        if prefill_bytes < prefill_min_lane_bytes {
            return Err(be(format!(
                "elastic arena leaves a {prefill_bytes}-byte phase-exclusive Prefill corridor, below its {prefill_min_lane_bytes}-byte minimum lane"
            )));
        }
        let high_order = loanable_order.split_off(low_count);
        let low_order = loanable_order;
        let floor_counts: Vec<_> = specs
            .iter()
            .map(|&(_, _, floor_slots)| floor_slots)
            .collect();
        let floor_order = smooth_weighted_pool_order(&floor_counts);

        let mut loanable_slot = vec![0usize; specs.len()];
        let mut floor_slot = loanable_counts.clone();
        let mut physical_order = Vec::with_capacity(total_slots);
        for pool in low_order {
            let slot = loanable_slot[pool];
            loanable_slot[pool] += 1;
            physical_order.push((pool, slot));
        }
        let floor_start = low_bytes;
        for pool in floor_order {
            let slot = floor_slot[pool];
            floor_slot[pool] += 1;
            physical_order.push((pool, slot));
        }
        let floor_end = floor_start
            .checked_add(floor_bytes)
            .ok_or_else(|| be("expert floor corridor overflow"))?;
        for pool in high_order {
            let slot = loanable_slot[pool];
            loanable_slot[pool] += 1;
            physical_order.push((pool, slot));
        }
        debug_assert_eq!(physical_order.len(), total_slots);
        debug_assert!(loanable_slot
            .iter()
            .zip(&loanable_counts)
            .all(|(placed, wanted)| placed == wanted));
        debug_assert!(floor_slot
            .iter()
            .zip(specs)
            .all(|(placed, &(_, wanted, _))| placed == &wanted));

        let mut slot_table: Vec<Vec<Option<ExpertSlotPlacement>>> = specs
            .iter()
            .map(|&(_, n_slots, _)| vec![None; n_slots])
            .collect();
        let mut shard_sizes = Vec::new();
        let mut shard = 0usize;
        let mut shard_offset = 0usize;
        let mut logical_offset = 0usize;

        for (position, (pool, slot)) in physical_order.into_iter().enumerate() {
            let slot_bytes = specs[pool].0;
            // KV can never borrow the Decode floor. End its physical shard before the floor so the
            // phase-exclusive Prefill corridor can borrow floor + high filler from a clean Vulkan
            // allocation instead of inheriting an arbitrary KV-shard tail.
            let starts_prefill_corridor = position == low_count && low_count != total_slots;
            if shard_offset != 0
                && (starts_prefill_corridor || shard_offset.saturating_add(slot_bytes) > max_shard)
            {
                shard_sizes.push(shard_offset);
                shard += 1;
                shard_offset = 0;
            }
            slot_table[pool][slot] = Some(ExpertSlotPlacement {
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
        let slots_by_pool = slot_table
            .into_iter()
            .map(|slots| {
                slots
                    .into_iter()
                    .map(|slot| slot.expect("every planned expert slot has physical coordinates"))
                    .collect()
            })
            .collect();

        Ok(Self {
            shard_sizes,
            slots_by_pool,
            total_bytes,
            kv_reserve_bytes,
            runtime_reserve_bytes,
            floor_corridor: floor_start..floor_end,
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
        // The low surplus is selected in whole Expert slots, so its physical boundary can extend
        // slightly past the byte-exact model estimate. Keep that otherwise unreachable tail in
        // the KV corridor; it supplies useful shard-packing slack without growing the arena.
        0..self.floor_corridor.start
    }

    pub(crate) fn prefill_corridor(&self) -> Range<usize> {
        // Prefill and Decode are mutually exclusive phases. The whole-layer ring may therefore
        // evict and borrow Decode-floor cells; release restores those physical slots before the
        // next Decode dispatch. KV and runtime owners continue to exclude the floor.
        self.floor_corridor.start..self.total_bytes - self.runtime_reserve_bytes
    }

    pub(crate) fn runtime_corridor(&self) -> Range<usize> {
        self.total_bytes - self.runtime_reserve_bytes..self.total_bytes
    }

    pub(crate) fn floor_corridor(&self) -> Range<usize> {
        self.floor_corridor.clone()
    }

    pub(crate) fn floor_slots(&self, pool: usize) -> Vec<usize> {
        let floor = &self.floor_corridor;
        self.slots(pool)
            .into_iter()
            .flatten()
            .filter(|slot| {
                slot.logical_offset >= floor.start
                    && slot.logical_offset.saturating_add(slot.len) <= floor.end
            })
            .map(|slot| slot.id.slot)
            .collect()
    }

    pub(crate) fn slot_is_in_floor(&self, id: ExpertSlotId) -> bool {
        let Some(slot) = self.slots(id.pool).and_then(|slots| slots.get(id.slot)) else {
            return false;
        };
        slot.logical_offset >= self.floor_corridor.start
            && slot.logical_offset.saturating_add(slot.len) <= self.floor_corridor.end
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
        self.plan_claim_with_reservations(
            live,
            requested,
            class,
            corridor,
            direction,
            protected_experts,
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_claim_with_reservations(
        &self,
        live: &[UnifiedRange],
        requested: &[usize],
        class: UnifiedVramClass,
        corridor: Range<usize>,
        direction: ClaimDirection,
        protected_experts: &[ExpertSlotId],
        reservations: &[PlannedRange],
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
        blockers.extend_from_slice(reservations);
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

    fn plan_exact_claim(
        &self,
        live: &[UnifiedRange],
        ranges: &[PlannedRange],
        class: UnifiedVramClass,
        corridor: Range<usize>,
        protected_experts: &[ExpertSlotId],
    ) -> Result<UnifiedClaimPlan> {
        if class == UnifiedVramClass::Expert || ranges.is_empty() {
            return Err(be("exact elastic claim needs non-Expert ranges"));
        }
        let pieces = self.physical_pieces(corridor);
        for (index, range) in ranges.iter().enumerate() {
            if range.requested_len == 0
                || range.len < range.requested_len
                || !range.len.is_multiple_of(UNIFIED_ALIGN)
                || !pieces.iter().any(|piece| {
                    piece.shard == range.shard
                        && range.offset >= piece.start
                        && range.offset.saturating_add(range.len) <= piece.end
                })
            {
                return Err(be(format!(
                    "exact {class:?} range {index} is outside its frozen arena corridor"
                )));
            }
            if ranges[..index].iter().any(|old| {
                old.shard == range.shard
                    && ranges_overlap(old.offset, old.len, range.offset, range.len)
            }) {
                return Err(be(format!(
                    "exact {class:?} range {index} overlaps an earlier range"
                )));
            }
        }

        let live_experts: HashSet<_> = live
            .iter()
            .filter(|range| range.class == UnifiedVramClass::Expert)
            .map(|range| (range.shard, range.offset, range.len))
            .collect();
        for allocation in live
            .iter()
            .filter(|allocation| allocation.class != UnifiedVramClass::Expert)
        {
            if ranges.iter().any(|range| {
                range.shard == allocation.shard
                    && ranges_overlap(range.offset, range.len, allocation.offset, allocation.len)
            }) {
                return Err(be(format!(
                    "exact {class:?} range overlaps a live {:?} allocation",
                    allocation.class
                )));
            }
        }
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
            if live_experts.contains(&(placement.shard, placement.offset, placement.len))
                && ranges.iter().any(|range| {
                    range.shard == placement.shard
                        && ranges_overlap(range.offset, range.len, placement.offset, placement.len)
                })
            {
                return Err(be(format!(
                    "exact {class:?} range overlaps protected Expert slot {}/{}",
                    id.pool, id.slot
                )));
            }
        }

        let mut victims = Vec::new();
        for placement in self.slots_by_pool.iter().flatten() {
            if live_experts.contains(&(placement.shard, placement.offset, placement.len))
                && ranges.iter().any(|range| {
                    range.shard == placement.shard
                        && ranges_overlap(range.offset, range.len, placement.offset, placement.len)
                })
            {
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
            ranges: ranges.to_vec(),
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
pub(crate) struct PlannedRange {
    pub(crate) shard: usize,
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) requested_len: usize,
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

    pub(crate) fn ranges(&self) -> &[PlannedRange] {
        &self.ranges
    }

    pub(crate) fn class(&self) -> UnifiedVramClass {
        self.class
    }
}

#[derive(Default)]
struct KvReservationState {
    next_id: u64,
    ranges: HashMap<u64, Vec<PlannedRange>>,
}

/// Frozen coordinates for every segment of one or more logical KV planes. A reservation owns no
/// VRAM bytes: Expert filler remains live until an exact segment claim retires the overlapping
/// cells. It only prevents later segmented-KV buffers from choosing the same future coordinates.
pub(crate) struct UnifiedKvReservation {
    id: u64,
    ranges: Vec<PlannedRange>,
    state: Arc<Mutex<KvReservationState>>,
}

impl UnifiedKvReservation {
    pub(crate) fn ranges(&self) -> &[PlannedRange] {
        &self.ranges
    }
}

impl Drop for UnifiedKvReservation {
    fn drop(&mut self) {
        self.state.lock().unwrap().ranges.remove(&self.id);
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
    kv_reservations: Arc<Mutex<KvReservationState>>,
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
            kv_reservations: Arc::new(Mutex::new(KvReservationState::default())),
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

    pub(crate) fn reserve_kv_layout(
        &self,
        requested: &[usize],
        protected_experts: &[ExpertSlotId],
    ) -> Result<Arc<UnifiedKvReservation>> {
        let layout = self
            .expert_layout
            .as_ref()
            .ok_or_else(|| be("KV reservations require an expert-aware unified VRAM layout"))?;
        let live = self.ranges.allocations();
        let mut state = self.kv_reservations.lock().unwrap();
        let existing: Vec<_> = state.ranges.values().flatten().copied().collect();
        let plan = layout.plan_claim_with_reservations(
            &live,
            requested,
            UnifiedVramClass::KvCache,
            layout.kv_corridor(),
            ClaimDirection::Low,
            protected_experts,
            &existing,
        )?;
        let id = state.next_id;
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| be("KV reservation id overflow"))?;
        let ranges = plan.ranges().to_vec();
        state.ranges.insert(id, ranges.clone());
        Ok(Arc::new(UnifiedKvReservation {
            id,
            ranges,
            state: Arc::clone(&self.kv_reservations),
        }))
    }

    pub(crate) fn plan_exact_kv_claim(
        &self,
        ranges: &[PlannedRange],
        protected_experts: &[ExpertSlotId],
    ) -> Result<UnifiedClaimPlan> {
        let layout = self
            .expert_layout
            .as_ref()
            .ok_or_else(|| be("exact KV claims require an expert-aware unified VRAM layout"))?;
        layout.plan_exact_claim(
            &self.ranges.allocations(),
            ranges,
            UnifiedVramClass::KvCache,
            layout.kv_corridor(),
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

    /// Plan a non-KV, non-Prefill owner from high addresses. The physical expert-floor corridor
    /// remains a hard lower boundary even before lazy KV segments are committed.
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
            layout.floor_corridor().end..layout.total_bytes(),
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
        let specs = [(512, 2, 0), (256, 1, 0), (256, 17, 0)];
        let layout = ExpertArenaLayout::build(&specs, 4096, 0, 0, 0, 0).unwrap();
        let mut ordered: Vec<_> = layout.slots_by_pool.iter().flatten().copied().collect();
        ordered.sort_unstable_by_key(|slot| slot.logical_offset);
        assert_eq!(ordered.len(), 20);

        let mut seen = vec![0usize; specs.len()];
        for (prefix, slot) in ordered.iter().enumerate() {
            seen[slot.id.pool] += 1;
            let prefix = prefix + 1;
            for (pool, &(_, wanted, _)) in specs.iter().enumerate() {
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
        let specs = [(1024, 5, 0), (512, 7, 0), (256, 9, 0)];
        let layout = ExpertArenaLayout::build(&specs, 2048, 0, 0, 0, 0).unwrap();
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
        let layout = ExpertArenaLayout::build(&[(256, 20, 0)], 4096, 257, 0, 0, 513).unwrap();
        assert_eq!(layout.kv_corridor(), 0..512);
        assert_eq!(layout.runtime_corridor(), 4352..5120);
        assert_eq!(layout.prefill_corridor(), 512..4352);
    }

    #[test]
    fn arena_corridors_reject_reserves_larger_than_the_physical_pool() {
        let error = ExpertArenaLayout::build(&[(256, 4, 0)], 4096, 768, 0, 0, 512).unwrap_err();
        assert!(error.to_string().contains("below its"));
    }

    #[test]
    fn persistent_corridors_preserve_floor_while_prefill_may_borrow_it() {
        let layout =
            ExpertArenaLayout::build(&[(256, 4, 4), (512, 8, 2)], 4096, 1024, 0, 512, 512).unwrap();
        let floor = layout.floor_corridor();
        assert_eq!(floor, 1024..3072);
        assert_eq!(layout.prefill_corridor(), 1024..4608);
        assert!(layout.slots(0).unwrap().iter().all(|slot| {
            slot.logical_offset >= floor.start && slot.logical_offset + slot.len <= floor.end
        }));

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
        let kv = layout
            .plan_claim(
                &ranges.allocations(),
                &[1024],
                UnifiedVramClass::KvCache,
                layout.kv_corridor(),
                ClaimDirection::Low,
                &[],
            )
            .unwrap();
        assert!(kv.victims.iter().all(|victim| victim.pool == 1));
        let prefill = layout
            .plan_claim(
                &ranges.allocations(),
                &[512],
                UnifiedVramClass::Prefill,
                layout.prefill_corridor(),
                ClaimDirection::Low,
                &[],
            )
            .unwrap();
        assert!(prefill.victims.iter().any(|victim| victim.pool == 0));
        let runtime = layout
            .plan_claim(
                &ranges.allocations(),
                &[512],
                UnifiedVramClass::LlmRuntime,
                layout.floor_corridor().end..layout.total_bytes(),
                ClaimDirection::High,
                &[],
            )
            .unwrap();
        assert!(runtime.victims.iter().all(|victim| victim.pool == 1));
    }

    #[test]
    fn prefill_corridor_starts_on_a_clean_shard_after_the_expert_floor() {
        let layout =
            ExpertArenaLayout::build(&[(256, 4, 4), (512, 8, 2)], 4096, 1024, 0, 512, 512).unwrap();
        let pieces = layout.physical_pieces(layout.prefill_corridor());

        assert!(!pieces.is_empty());
        assert_eq!(pieces[0].start, 0);
        assert_eq!(pieces[0].end - pieces[0].start, 3584);
    }

    #[test]
    fn frozen_kv_layout_survives_incremental_exact_claims() {
        let layout = ExpertArenaLayout::build(&[(256, 32, 4)], 2048, 4096, 512, 0, 0).unwrap();
        let ranges = UnifiedRangePool::new(layout.shard_sizes().iter().copied()).unwrap();
        let mut experts: HashMap<_, _> = layout
            .slots_by_pool
            .iter()
            .flatten()
            .map(|slot| {
                let lease = ranges
                    .try_claim_exact(slot.shard, slot.offset, slot.len, UnifiedVramClass::Expert)
                    .unwrap();
                (slot.id, lease)
            })
            .collect();

        // Three logical planes grow together through four depth increments. Reserve every future
        // segment up front, but leave the backing Expert cells live until each increment commits.
        let requested = [512, 512, 512, 512, 256, 256, 256, 256, 256, 256, 256, 256];
        let frozen = layout
            .plan_claim_with_reservations(
                &ranges.allocations(),
                &requested,
                UnifiedVramClass::KvCache,
                layout.kv_corridor(),
                ClaimDirection::Low,
                &[],
                &[],
            )
            .unwrap();
        let mut kv = Vec::new();
        for segment in 0..4 {
            let exact_ranges = [
                frozen.ranges[segment],
                frozen.ranges[4 + segment],
                frozen.ranges[8 + segment],
            ];
            let claim = layout
                .plan_exact_claim(
                    &ranges.allocations(),
                    &exact_ranges,
                    UnifiedVramClass::KvCache,
                    layout.kv_corridor(),
                    &[],
                )
                .unwrap();
            for victim in claim.victims() {
                drop(experts.remove(victim).unwrap());
            }
            kv.extend(
                ranges
                    .try_claim_planned(UnifiedVramClass::KvCache, claim.ranges())
                    .unwrap(),
            );
        }

        assert_eq!(kv.len(), requested.len());
        assert_eq!(ranges.stats().class_bytes(UnifiedVramClass::KvCache), 4096);
        assert_eq!(experts.len(), 16);
    }

    #[test]
    fn low_kv_claim_reports_exact_interleaved_expert_victims() {
        let layout =
            ExpertArenaLayout::build(&[(256, 8, 0), (512, 4, 0)], 4096, 1024, 0, 0, 512).unwrap();
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
        let layout =
            ExpertArenaLayout::build(&[(256, 8, 0), (512, 4, 0)], 4096, 1024, 0, 0, 512).unwrap();
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
        let layout = ExpertArenaLayout::build(&[(256, 16, 0)], 4096, 1024, 0, 0, 512).unwrap();
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
        let pieces = layout.physical_pieces(layout.kv_corridor().end..layout.total_bytes());
        let high_piece = pieces.last().unwrap();
        assert_eq!(plan.ranges.len(), 1);
        assert_eq!(plan.ranges[0].shard, high_piece.shard);
        assert_eq!(plan.ranges[0].offset, high_piece.end - 512);
        assert!(plan.ranges[0].offset >= high_piece.start);
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
