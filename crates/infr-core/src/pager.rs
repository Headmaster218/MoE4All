//! Generic, backend-agnostic block-paging bookkeeping: a fixed-slot LRU cache mapping an opaque
//! `BlockId` to a slot index, for any backend that wants to keep a working set of uniform-ish
//! byte blocks resident in a budget smaller than the full set.
//!
//! This module holds ONLY the host-side residency/eviction *decision* (no GPU types, no bytes) —
//! [`Pager::touch`] is pure bookkeeping, cheap to unit test without a device. A backend wraps this
//! with the actual VRAM arena + device LUT buffer + upload machinery (see `infr-vulkan`'s
//! `GpuPager`) and drives it by calling `touch` for every block a dispatch is about to read,
//! before recording that dispatch.
//!
//! ## Today: MoE expert paging
//! The Vulkan MoE lowering plugs in `BlockId = (layer, role, expert_id)` packed into a `u32` (see
//! `infr-vulkan`'s `pager` module) with a demand/LRU policy — an expert is paged in on first use
//! by a layer and evicted least-recently-used when the arena is full.
//!
//! ## Dense layer streaming
//! The second policy reuses the SAME slot bookkeeping for `BlockId = layer_idx` (or a per-tensor
//! block within a layer) in a dense model whose weights don't fit VRAM: a dense forward visits
//! layers in a fixed, known order every step, so residency is schedule-driven via
//! [`Pager::schedule`] (exact cyclic-sweep eviction, Belady-parity — see its doc) rather than
//! demand/LRU. The PREFETCH side (staging layer `l+1`'s bytes while `l` executes) is the
//! backend's job — `infr-vulkan`'s `DensePagerSession` records uploads ahead of GPU execution
//! through its pipelined ring; this module stays pure residency/eviction bookkeeping.
use std::collections::HashMap;

/// Opaque identifier for one pageable block. The pager never interprets this — callers pack
/// whatever key space they need (an expert id, a `(layer, role, expert)` tuple, a layer index for
/// the planned dense-streaming policy, ...) into it.
pub type BlockId = u32;

/// Sentinel LUT value meaning "not resident" — mirrors what a device-side LUT buffer should hold
/// for any block the pager hasn't (yet) admitted, so an accidental stale read is loud (an
/// out-of-range slot index) rather than silently aliasing slot 0.
pub const NOT_RESIDENT: u32 = u32::MAX;

/// Outcome of [`Pager::touch`]: whether the block was already resident (no upload needed) or had
/// to be paged in (possibly evicting another block first).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// Already resident at `slot` — no upload needed, LUT already correct.
    Hit { slot: u32 },
    /// Now resident at `slot` after an upload the caller must perform. `evicted` is the block
    /// that previously occupied `slot`, if any (so the caller can invalidate its LUT entry, or
    /// just leave it — it's never read again unless `evicted` is re-touched, which re-admits it
    /// through the normal miss path).
    Miss { slot: u32, evicted: Option<BlockId> },
}

/// Why [`Pager::admit_at`] could not replay an admission. Every error leaves the pager unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    /// `NOT_RESIDENT` cannot name a block in an admission replay.
    InvalidBlockId,
    SlotOutOfRange,
    SlotDisabled,
    /// The requested block already owns a different physical slot.
    ResidentElsewhere {
        slot: u32,
    },
    /// The target's current owner has an outstanding explicit pin.
    Pinned {
        id: BlockId,
    },
    /// The target's current owner was accessed in the current batch.
    CurrentBatch {
        id: BlockId,
    },
}

/// Cumulative pager activity — the `INFR_PAGER_STATS` hit-rate counter the task's validation step
/// asks for rides this.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagerStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl PagerStats {
    /// Hit rate over all `touch` calls so far; `1.0` (vacuously) before any calls.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Immutable cache ownership at a completed-dispatch handoff boundary.
///
/// This describes bookkeeping only: the backend must synchronize outstanding readers and keep
/// slot bytes and its device LUT consistent with these mappings. Batch epochs and explicit pins
/// are not transferable. Restoring starts with no batch protection; begin a new batch before
/// resolving the next dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagerSnapshot {
    disabled: Vec<bool>,
    free: Vec<u32>,
    resident_lru: Vec<(BlockId, u32)>,
    stats: PagerStats,
}

impl PagerSnapshot {
    /// Assemble a snapshot, for example from GPU-authoritative cache state.
    ///
    /// `disabled` is indexed by physical slot (true means disabled) and its length is the total
    /// slot capacity. `free` is a stack of enabled empty slots: `pop()` allocates the LAST element
    /// next, exactly as in the pager. `resident_lru` lists `(block, slot)` pairs exactly once,
    /// oldest to newest. Every
    /// enabled slot must occur in exactly one of those lists. IDs are opaque, possibly sparse;
    /// only `NOT_RESIDENT` is reserved. `stats` contains cumulative totals, including any activity
    /// while the GPU owned the cache. Construction does not validate: [`Pager::restore_handoff`]
    /// checks the entire snapshot before changing the destination.
    pub fn from_parts(
        disabled: Vec<bool>,
        free: Vec<u32>,
        resident_lru: Vec<(BlockId, u32)>,
        stats: PagerStats,
    ) -> Self {
        Self {
            disabled,
            free,
            resident_lru,
            stats,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.disabled.len()
    }

    /// Physical-slot mask; true means disabled, false means enabled.
    pub fn disabled(&self) -> &[bool] {
        &self.disabled
    }

    /// Enabled empty-slot stack in the pager's internal order; the LAST element allocates next.
    pub fn free(&self) -> &[u32] {
        &self.free
    }

    /// Exact resident `(block, slot)` mapping and eviction order, oldest to newest.
    pub fn resident_lru(&self) -> &[(BlockId, u32)] {
        &self.resident_lru
    }

    pub fn stats(&self) -> PagerStats {
        self.stats
    }
}

/// A rejected handoff changes no pager state, including batch protection and counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HandoffError {
    #[error("pager handoff: {blocks} blocks have active pins")]
    ActivePins { blocks: usize },
    #[error("pager handoff: expected {expected} slots, snapshot has {actual}")]
    CapacityMismatch { expected: usize, actual: usize },
    #[error("pager handoff: NOT_RESIDENT is not a valid block ID")]
    InvalidBlockId,
    #[error("pager handoff: slot {slot} is out of range")]
    SlotOutOfRange { slot: u32 },
    #[error("pager handoff: slot {slot} is disabled")]
    SlotDisabled { slot: u32 },
    #[error("pager handoff: block {id} appears more than once")]
    DuplicateBlockId { id: BlockId },
    #[error("pager handoff: slot {slot} appears more than once")]
    SlotConflict { slot: u32 },
    #[error("pager handoff: enabled slot {slot} is neither free nor resident")]
    MissingSlot { slot: u32 },
}

#[derive(Clone, Copy, Default)]
struct LruNode {
    id: Option<BlockId>,
    prev: Option<u32>,
    next: Option<u32>,
}

/// Fixed-`n_slots` LRU residency map. `n_slots` slots are handed out first-come (no eviction)
/// until exhausted, then every further miss evicts the least-recently-touched resident block.
///
/// # Within-batch safety
/// A caller MUST resolve every block a single dispatch batch needs (e.g. all `n_used` experts a
/// decode step's top-k picked, or every distinct expert a prefill ubatch's bucket counts named)
/// via `touch` BEFORE recording/dispatching anything that reads the result — and must do so as one
/// uninterrupted sequence of `touch` calls. `touch` marks a block most-recently-used the instant
/// it resolves, and eviction only ever pops the LEAST-recently-used entry, so a block touched
/// earlier in the same batch can never be evicted by a later touch in that SAME batch. This only
/// holds if `n_slots >= ` the number of DISTINCT blocks one batch touches — [`Pager::new`]'s doc
/// repeats this invariant; violating it (a cache budget too small to even hold one batch) is a
/// configuration error the caller should surface, not silently thrash.
pub struct Pager {
    n_slots: usize,
    /// Slots temporarily loaned to another consumer of a shared device-memory arena. Disabled
    /// slots keep their stable numeric identity but never appear in `free` or the LRU until they
    /// are enabled again. This lets a backend carve a contiguous module allocation out of an
    /// expert cache without rebuilding every block -> slot mapping.
    disabled: Vec<bool>,
    enabled_slots: usize,
    /// block_id -> slot index for every currently-resident block.
    resident: HashMap<BlockId, u32>,
    /// LRU order, oldest (least-recently-used) at the front. A block appears at most once;
    /// `touch` on an existing entry removes-then-repushes it (O(n_slots), fine at the slot counts
    /// this cache runs at — tens to low hundreds; an intrusive doubly-linked list is the upgrade
    /// path if that ever isn't true, per the module doc's SLRU note).
    lru: Vec<LruNode>,
    lru_head: Option<u32>,
    lru_tail: Option<u32>,
    lru_len: usize,
    /// Slot indices never yet handed out (drained before any eviction kicks in).
    free: Vec<u32>,
    /// block_id -> the [`Self::begin_batch`] epoch of its last touch. Eviction skips blocks
    /// touched in the CURRENT batch — the explicit form of the within-batch safety invariant
    /// (MRU ordering used to carry it implicitly; the scan-resistant [`Self::touch_cold`] path
    /// inserts at the COLD end, so it needs the epoch guard instead).
    epoch: HashMap<BlockId, u64>,
    cur_epoch: u64,
    /// Next LRU node the scan-resistant cold path should inspect in this batch. `touch_cold`
    /// inserts new misses before this cursor and every node skipped by the cursor is protected by
    /// `cur_epoch`, so a node never needs to be visited twice until [`Self::begin_batch`] resets
    /// the cursor. This turns a wide cold batch from O(batch_width * protected_prefix) victim
    /// scans into one amortized pass over the LRU list.
    cold_victim_cursor: Option<u32>,
    /// block_id -> outstanding pin count, for blocks a caller is READING right now
    /// ([`Self::resolve_and_pin`]). Only present while non-zero. A pinned block is never an
    /// eviction victim: the epoch guard above cannot express this, because it is scoped to one
    /// dispatch batch while a pin lasts as long as the borrow — a CPU kernel reads a weight for a
    /// whole op, spanning batches.
    pinned: HashMap<BlockId, u32>,
    stats: PagerStats,
    profile_work: crate::pager_profile::LruWorkStats,
}

/// Where a miss inserts into the LRU order — the two policies this pager already had, named so a
/// caller can pick one at the call site instead of through three near-identical entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insert {
    /// Most-recently-used end: recency (`touch`). Right when reuse follows use.
    Mru,
    /// Cold end: scan-resistant / cyclic-sweep (`touch_cold`, `schedule`). Right when the block
    /// just used is the one whose next use is furthest away.
    Cold,
}

impl Pager {
    /// A pager with `n_slots` uniform slots and nothing resident. `n_slots` must be at least the
    /// largest number of distinct blocks any single dispatch batch will `touch` (see the
    /// within-batch safety note) — the pager can't check that itself (it doesn't know about
    /// batches), so a caller sizing `n_slots` from a VRAM budget must clamp it to at least that
    /// floor and error out earlier if the budget can't cover even one batch.
    ///
    /// # Panics
    /// If `n_slots == 0`. A zero-slot pager cannot satisfy even a one-block batch, so there is no
    /// correct behaviour to fall back to: the first `touch` would find `free` empty, scan an empty
    /// `lru` for a victim, and die inside [`Self::take_slot`] on an `expect` about a `position()`
    /// index — a panic that describes the wrong thing entirely, several frames away from the
    /// mistake. Clamping to 1 instead would be worse than either: it manufactures a pager that
    /// silently misses on EVERY touch and evicts the block the caller resolved a moment ago,
    /// violating the within-batch safety invariant above for any batch wider than one block —
    /// precisely the coherent-but-wrong failure this module refuses elsewhere (see `take_slot`'s
    /// batch-overflow assert). The contract quoted above already makes the FLOOR the caller's job,
    /// so a 0 here is a caller bug that must surface at the mistake. The one in-tree caller
    /// (`infr-vulkan`'s `validate_pager_dims`) already rejects a 0-slot budget with a clean `Err`
    /// before constructing anything, so this can only fire for a NEW caller that skipped that step
    /// — never on a shipping path. `new` also runs once per pool at model load, so a check here
    /// costs nothing measurable.
    pub fn new(n_slots: usize) -> Self {
        assert!(
            n_slots >= 1,
            "pager: n_slots must be at least 1 (a zero-slot pager can never satisfy a touch) — \
             size the slot count from the paging budget and reject a budget that can't hold one \
             dispatch batch BEFORE constructing the pager"
        );
        Self {
            n_slots,
            disabled: vec![false; n_slots],
            enabled_slots: n_slots,
            resident: HashMap::with_capacity(n_slots),
            lru: vec![LruNode::default(); n_slots],
            lru_head: None,
            lru_tail: None,
            lru_len: 0,
            free: (0..n_slots as u32).rev().collect(), // pop() hands out slot 0 first
            epoch: HashMap::with_capacity(n_slots),
            cur_epoch: 0,
            cold_victim_cursor: None,
            pinned: HashMap::new(),
            stats: PagerStats::default(),
            profile_work: crate::pager_profile::LruWorkStats::default(),
        }
    }

    /// Open a new touch batch (one dispatch's worth of `touch`/`touch_cold` calls). Blocks
    /// touched under the current epoch are never eviction victims, which makes the within-batch
    /// safety invariant explicit for BOTH insertion orders (classic MRU and `touch_cold`'s
    /// cold-end scan inserts, where LRU position alone no longer protects batch siblings).
    pub fn begin_batch(&mut self) {
        self.cur_epoch += 1;
        self.cold_victim_cursor = self.lru_head;
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    /// Number of slots currently available to this pager. This can be smaller than `n_slots`
    /// while part of the backing arena is loaned to another model component.
    pub fn enabled_slots(&self) -> usize {
        self.enabled_slots
    }

    /// Temporarily remove one physical slot from the cache, returning the resident block evicted
    /// from it (if any). The slot number remains stable and can later be restored with
    /// [`Self::enable_slot`].
    ///
    /// # Panics
    ///
    /// Panics when the slot is out of range or its block is pinned. A caller must only resize a
    /// cache between dispatches; removing memory a live reader owns would be silent corruption.
    pub fn disable_slot(&mut self, slot: u32) -> Option<BlockId> {
        let idx = self.slot_idx(slot);
        assert!(
            !self.disabled[idx],
            "pager: slot {slot} is already disabled"
        );
        let resident = self.lru[idx].id;
        if let Some(id) = resident {
            assert!(
                !self.pinned.contains_key(&id),
                "pager: disable of pinned block {id}"
            );
            let removed = self.resident.remove(&id);
            debug_assert_eq!(removed, Some(slot));
            let unlinked = self.unlink_lru(slot);
            debug_assert_eq!(unlinked, Some(id));
            self.epoch.remove(&id);
            self.stats.evictions += 1;
        } else if let Some(pos) = self.free.iter().position(|&free| free == slot) {
            self.free.swap_remove(pos);
        }
        self.disabled[idx] = true;
        self.enabled_slots -= 1;
        resident
    }

    /// Return a previously disabled physical slot to the free list.
    pub fn enable_slot(&mut self, slot: u32) {
        let idx = self.slot_idx(slot);
        assert!(self.disabled[idx], "pager: slot {slot} is already enabled");
        debug_assert!(self.lru[idx].id.is_none());
        self.disabled[idx] = false;
        self.enabled_slots += 1;
        self.free.push(slot);
    }

    pub fn slot_enabled(&self, slot: u32) -> bool {
        !self.disabled[self.slot_idx(slot)]
    }

    pub fn stats(&self) -> PagerStats {
        self.stats
    }

    /// Capture exact slot ownership, allocation/eviction order and cumulative demand stats.
    ///
    /// Requires no explicit pins. The caller must also have completed outstanding dispatches;
    /// epoch protection is deliberately omitted from the snapshot (see [`PagerSnapshot`]).
    pub fn snapshot_for_handoff(&self) -> Result<PagerSnapshot, HandoffError> {
        if !self.pinned.is_empty() {
            return Err(HandoffError::ActivePins {
                blocks: self.pinned.len(),
            });
        }
        if self.resident.contains_key(&NOT_RESIDENT) {
            return Err(HandoffError::InvalidBlockId);
        }
        Ok(PagerSnapshot::from_parts(
            self.disabled.clone(),
            self.free.clone(),
            self.resident_slots_lru(),
            self.stats,
        ))
    }

    /// Check whether this pager can accept a handoff without changing any state.
    ///
    /// Uses the same checks as [`Self::restore_handoff`], including destination pins and total
    /// physical capacity. Callers can validate every pool before restoring any pool, keeping all
    /// pool states unchanged between validation and restore. This does not reserve slots or
    /// synchronize device readers; restore revalidates against the destination's current state.
    pub fn validate_handoff(&self, snapshot: &PagerSnapshot) -> Result<(), HandoffError> {
        if !self.pinned.is_empty() {
            return Err(HandoffError::ActivePins {
                blocks: self.pinned.len(),
            });
        }
        if snapshot.n_slots() != self.n_slots {
            return Err(HandoffError::CapacityMismatch {
                expected: self.n_slots,
                actual: snapshot.n_slots(),
            });
        }

        let mut seen = vec![false; self.n_slots];
        let mut resident = HashMap::with_capacity(snapshot.resident_lru.len().min(self.n_slots));
        let mut claim_slot = |slot: u32| -> Result<(), HandoffError> {
            let idx = slot as usize;
            if idx >= self.n_slots {
                return Err(HandoffError::SlotOutOfRange { slot });
            }
            if snapshot.disabled[idx] {
                return Err(HandoffError::SlotDisabled { slot });
            }
            if seen[idx] {
                return Err(HandoffError::SlotConflict { slot });
            }
            seen[idx] = true;
            Ok(())
        };
        for &slot in &snapshot.free {
            claim_slot(slot)?;
        }
        for &(id, slot) in &snapshot.resident_lru {
            if id == NOT_RESIDENT {
                return Err(HandoffError::InvalidBlockId);
            }
            if resident.insert(id, slot).is_some() {
                return Err(HandoffError::DuplicateBlockId { id });
            }
            claim_slot(slot)?;
        }
        for (idx, &claimed) in seen.iter().enumerate() {
            if !snapshot.disabled[idx] && !claimed {
                return Err(HandoffError::MissingSlot { slot: idx as u32 });
            }
        }
        Ok(())
    }

    /// Replace ownership from a completed-dispatch handoff without replaying any accesses.
    ///
    /// The snapshot must have the same total physical capacity; its enabled/disabled mask
    /// replaces this pager's mask. All mappings, free slots and IDs are validated before any
    /// mutation. A rejection leaves EVERY field unchanged, including pins, epochs and profiling.
    /// Success preserves the supplied cumulative demand stats without counting admissions or
    /// evictions. CPU profiling stays local to this pager. Old batch guards are cleared, so the
    /// caller must synchronize readers before restore and begin a new batch for the next dispatch.
    pub fn restore_handoff(&mut self, snapshot: &PagerSnapshot) -> Result<(), HandoffError> {
        self.validate_handoff(snapshot)?;

        // Build the intrusive list directly, in slot space, after validation. No touch/admit
        // replay: those would reorder the cache, recreate batch guards and count false traffic.
        let mut lru = vec![LruNode::default(); self.n_slots];
        let entries = &snapshot.resident_lru;
        let resident = entries.iter().copied().collect();
        for (pos, &(id, slot)) in entries.iter().enumerate() {
            lru[slot as usize] = LruNode {
                id: Some(id),
                prev: pos.checked_sub(1).map(|prev| entries[prev].1),
                next: entries.get(pos + 1).map(|entry| entry.1),
            };
        }
        let disabled = snapshot.disabled.clone();
        let enabled_slots = disabled.iter().filter(|&&disabled| !disabled).count();
        let free = snapshot.free.clone();

        self.disabled = disabled;
        self.enabled_slots = enabled_slots;
        self.resident = resident;
        self.lru = lru;
        self.lru_head = entries.first().map(|entry| entry.1);
        self.lru_tail = entries.last().map(|entry| entry.1);
        self.lru_len = entries.len();
        self.free = free;
        self.epoch.clear();
        self.cur_epoch = 0;
        self.cold_victim_cursor = self.lru_head;
        self.stats = snapshot.stats;
        Ok(())
    }

    /// Number of currently-resident blocks (== `n_slots` once the cache is full).
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Resident `(block, slot)` pairs from coldest to hottest.
    ///
    /// Stage-transition code uses this snapshot to borrow the coldest physical ranges for a
    /// temporary whole-layer Prefill ring without discarding unrelated hot Decode entries. It is
    /// intentionally a snapshot rather than an iterator borrowing the pager: callers may evict
    /// selected blocks immediately afterwards.
    pub fn resident_slots_lru(&self) -> Vec<(BlockId, u32)> {
        let mut out = Vec::with_capacity(self.lru_len);
        let mut slot = self.lru_head;
        while let Some(s) = slot {
            let node = self.lru[self.slot_idx(s)];
            out.push((node.id.expect("linked LRU node has a block id"), s));
            slot = node.next;
        }
        out
    }

    /// Free slots followed by unpinned residents, coldest first, without allocating a full snapshot.
    /// This is selection only: callers still own batch/read protection and physical reservations.
    pub fn replacement_slots(&self) -> impl Iterator<Item = (Option<BlockId>, u32)> + '_ {
        let free = self.free.iter().rev().copied().map(|slot| (None, slot));
        let mut next = self.lru_head;
        let resident = std::iter::from_fn(move || {
            while let Some(slot) = next {
                let node = self.lru[self.slot_idx(slot)];
                next = node.next;
                let id = node.id.expect("linked LRU node has a block id");
                if !self.pinned.contains_key(&id) {
                    return Some((Some(id), slot));
                }
            }
            None
        });
        free.chain(resident)
    }

    /// Number of entries in the `epoch` map — test-only, to assert eviction keeps it bounded by
    /// `n_slots` rather than growing per distinct BlockId ever touched.
    #[cfg(test)]
    pub(crate) fn epoch_len(&self) -> usize {
        self.epoch.len()
    }

    fn slot_idx(&self, slot: u32) -> usize {
        let idx = slot as usize;
        debug_assert!(idx < self.lru.len(), "slot {slot} out of range");
        idx
    }

    fn push_back_lru(&mut self, slot: u32, id: BlockId) {
        let idx = self.slot_idx(slot);
        debug_assert!(
            self.lru[idx].id.is_none(),
            "slot {slot} already has an LRU node"
        );
        self.lru[idx] = LruNode {
            id: Some(id),
            prev: self.lru_tail,
            next: None,
        };
        if let Some(tail) = self.lru_tail {
            let tail_idx = self.slot_idx(tail);
            self.lru[tail_idx].next = Some(slot);
        } else {
            self.lru_head = Some(slot);
        }
        self.lru_tail = Some(slot);
        self.lru_len += 1;
    }

    fn push_front_lru(&mut self, slot: u32, id: BlockId) {
        let idx = self.slot_idx(slot);
        debug_assert!(
            self.lru[idx].id.is_none(),
            "slot {slot} already has an LRU node"
        );
        self.lru[idx] = LruNode {
            id: Some(id),
            prev: None,
            next: self.lru_head,
        };
        if let Some(head) = self.lru_head {
            let head_idx = self.slot_idx(head);
            self.lru[head_idx].prev = Some(slot);
        } else {
            self.lru_tail = Some(slot);
        }
        self.lru_head = Some(slot);
        self.lru_len += 1;
    }

    fn unlink_lru(&mut self, slot: u32) -> Option<BlockId> {
        let idx = self.slot_idx(slot);
        let node = self.lru[idx];
        let id = node.id?;
        // MRU promotion or an external eviction can remove the next cold victim. Advance
        // before unlinking so a mixed touch/touch_cold batch cannot skip the remaining list.
        if self.cold_victim_cursor == Some(slot) {
            self.cold_victim_cursor = node.next;
        }
        if let Some(prev) = node.prev {
            let prev_idx = self.slot_idx(prev);
            self.lru[prev_idx].next = node.next;
        } else {
            self.lru_head = node.next;
        }
        if let Some(next) = node.next {
            let next_idx = self.slot_idx(next);
            self.lru[next_idx].prev = node.prev;
        } else {
            self.lru_tail = node.prev;
        }
        self.lru[idx] = LruNode::default();
        self.lru_len -= 1;
        Some(id)
    }

    fn move_to_back_lru(&mut self, slot: u32) {
        if self.lru_tail == Some(slot) {
            let idx = self.slot_idx(slot);
            debug_assert!(self.lru[idx].id.is_some());
            return;
        }
        let id = self
            .unlink_lru(slot)
            .expect("resident block has an LRU node");
        self.push_back_lru(slot, id);
    }

    fn first_lru_slot_matching<F>(&self, mut eligible: F) -> (Option<u32>, usize)
    where
        F: FnMut(BlockId) -> bool,
    {
        let mut scan_steps = 0usize;
        let mut slot = self.lru_head;
        while let Some(s) = slot {
            let node = self.lru[self.slot_idx(s)];
            let id = node.id.expect("linked LRU node has a block id");
            scan_steps += 1;
            if eligible(id) {
                return (Some(s), scan_steps);
            }
            slot = node.next;
        }
        (None, scan_steps)
    }

    #[cfg(test)]
    fn lru_order(&self) -> Vec<BlockId> {
        let mut out = Vec::new();
        let mut slot = self.lru_head;
        while let Some(s) = slot {
            let node = self.lru[self.slot_idx(s)];
            out.push(node.id.expect("linked LRU node has a block id"));
            slot = node.next;
            assert!(out.len() <= self.n_slots, "LRU list contains a cycle");
        }
        out
    }

    pub fn slot_of(&self, id: BlockId) -> Option<u32> {
        self.resident.get(&id).copied()
    }

    /// Move `id` to the most-recently-used end without changing residency (internal to `touch`,
    /// exposed for the doubly-recorded case: a batch that touches the same id twice).
    fn mark_mru(&mut self, id: BlockId) {
        let slot = *self
            .resident
            .get(&id)
            .expect("mark_mru is only called for resident blocks");
        if crate::pager_profile::active() {
            self.profile_work.record_mark_mru(self.lru_len, 0);
        }
        self.move_to_back_lru(slot);
    }

    /// Ensure `id` is resident, evicting the least-recently-used unprotected block if the cache
    /// is full. Returns whether it was already resident (caller skips the upload) or had to be
    /// paged in (caller must upload the block's bytes into `slot` and write the device LUT entry
    /// `id -> slot` before any dispatch reads it).
    pub fn touch(&mut self, id: BlockId) -> Resolution {
        self.epoch.insert(id, self.cur_epoch);
        if let Some(&slot) = self.resident.get(&id) {
            self.stats.hits += 1;
            self.mark_mru(id);
            return Resolution::Hit { slot };
        }
        self.stats.misses += 1;
        let (slot, evicted) = self.take_slot();
        self.resident.insert(id, slot);
        self.push_back_lru(slot, id);
        Resolution::Miss { slot, evicted }
    }

    /// Replay a demand admission into an exact, previously selected physical `slot`.
    ///
    /// This is host bookkeeping for a backend reconciling a GPU-selected admission. It neither
    /// copies bytes nor reserves the slot: the backend must protect and validate its selection
    /// before copying, then reconcile its LUT using the returned `Resolution`. A `Miss` describes
    /// the ownership change even when the backend has already filled the slot.
    ///
    /// A hit is accepted only at the block's existing slot, including when that block is pinned
    /// or current-batch protected. A miss may replace only an unpinned owner outside the current
    /// batch. As with normal demand access, epoch protection starts at the first `begin_batch`;
    /// successful hits and misses mark the block MRU, join the epoch and update demand stats.
    /// No explicit pin is acquired. Errors change no state, including counters and LRU order.
    ///
    /// Only the `NOT_RESIDENT` sentinel is rejected as an ID here; the pager has no model-specific
    /// ID bound, so validating that namespace remains the backend's responsibility.
    pub fn admit_at(&mut self, id: BlockId, slot: u32) -> Result<Resolution, AdmissionError> {
        if id == NOT_RESIDENT {
            return Err(AdmissionError::InvalidBlockId);
        }
        let idx = slot as usize;
        if idx >= self.n_slots {
            return Err(AdmissionError::SlotOutOfRange);
        }
        if self.disabled[idx] {
            return Err(AdmissionError::SlotDisabled);
        }
        if let Some(actual) = self.slot_of(id) {
            if actual != slot {
                return Err(AdmissionError::ResidentElsewhere { slot: actual });
            }
            return Ok(self.touch(id));
        }

        let evicted = self.lru[idx].id;
        if let Some(owner) = evicted {
            if self.pinned.contains_key(&owner) {
                return Err(AdmissionError::Pinned { id: owner });
            }
            if self.cur_epoch != 0 && self.epoch.get(&owner) == Some(&self.cur_epoch) {
                return Err(AdmissionError::CurrentBatch { id: owner });
            }
            let removed = self.unlink_lru(slot);
            debug_assert_eq!(removed, Some(owner));
            let old_slot = self.resident.remove(&owner);
            debug_assert_eq!(old_slot, Some(slot));
            self.epoch.remove(&owner);
            self.stats.evictions += 1;
        } else {
            let pos = self
                .free
                .iter()
                .position(|&free| free == slot)
                .expect("enabled empty slot is on the free list");
            // Preserve the allocation order of every other free slot.
            self.free.remove(pos);
        }
        self.resident.insert(id, slot);
        self.epoch.insert(id, self.cur_epoch);
        self.push_back_lru(slot, id);
        self.stats.misses += 1;
        Ok(Resolution::Miss { slot, evicted })
    }

    /// Ensure `id` is resident AND pin it, so it cannot be evicted until [`Self::unpin`].
    ///
    /// This is the entry point for a tier whose slots are read DIRECTLY (the host DRAM arena: a
    /// CPU kernel dereferences the slot for a whole op, and a staging copy reads it until the copy
    /// is recorded), as opposed to the VRAM pagers, whose callers upload a slot's bytes and then
    /// let residency ride the batch epoch alone.
    ///
    /// `None` means every resident block is protected right now — all pinned, or all touched by
    /// the current batch — so there is no slot to give. Nothing is evicted, nothing is inserted and
    /// no counter moves in that case: the caller has to release pins (or was sized below its own
    /// working set, which is a configuration error it should report). This is a `None` and not a
    /// panic precisely because it IS reachable at runtime, unlike the unpinned callers' floor.
    ///
    /// On a `Miss` the caller must fill the slot before reading it; on a `Hit` it is already valid.
    /// Either way the block comes back pinned exactly once.
    pub fn resolve_and_pin(&mut self, id: BlockId, insert: Insert) -> Option<Resolution> {
        if let Some(&slot) = self.resident.get(&id) {
            self.epoch.insert(id, self.cur_epoch);
            self.stats.hits += 1;
            if insert == Insert::Mru {
                self.mark_mru(id);
            }
            *self.pinned.entry(id).or_insert(0) += 1;
            return Some(Resolution::Hit { slot });
        }
        let (slot, evicted) = self.take_slot_opt(true)?;
        self.epoch.insert(id, self.cur_epoch);
        self.stats.misses += 1;
        self.resident.insert(id, slot);
        match insert {
            Insert::Mru => self.push_back_lru(slot, id),
            Insert::Cold => self.push_front_lru(slot, id),
        }
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(Resolution::Miss { slot, evicted })
    }

    /// Take ANOTHER pin on a block that is already resident, without counting anything.
    ///
    /// This is not a residency decision — it is a second borrow of a block the caller has already
    /// resolved (the CPU read path re-borrows what its op's pin pre-step resolved a moment ago).
    /// Counting it would inflate the hit rate by exactly one hit per access, which makes a
    /// thrashing cache report ~50% and a perfect one report 100%: the two become indistinguishable
    /// where it matters. Returns `None` if `id` is not resident.
    pub fn repin(&mut self, id: BlockId) -> Option<u32> {
        let slot = *self.resident.get(&id)?;
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(slot)
    }

    /// Pin `id` only if it is ALREADY resident — never reads, never evicts, never inserts.
    ///
    /// The hit-only probe a tier above this one uses to decide between "copy from here" and "go to
    /// the tier below". A hit counts as a hit; a miss counts as nothing, because no residency
    /// decision was made.
    pub fn pin_if_resident(&mut self, id: BlockId) -> Option<u32> {
        let slot = *self.resident.get(&id)?;
        self.epoch.insert(id, self.cur_epoch);
        self.stats.hits += 1;
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(slot)
    }

    /// Release one pin taken by [`Self::resolve_and_pin`] / [`Self::pin_if_resident`].
    ///
    /// # Panics
    /// If `id` holds no pin. An unbalanced release is not a state to recover from: it would let a
    /// slot be evicted while a reader still holds bytes from it, which is silent wrong output
    /// rather than a failure. The RAII guard in the host tier is what makes this balanced.
    pub fn unpin(&mut self, id: BlockId) {
        match self.pinned.get_mut(&id) {
            Some(n) if *n > 1 => *n -= 1,
            Some(_) => {
                self.pinned.remove(&id);
            }
            None => panic!("pager: unpin of block {id}, which holds no pin"),
        }
    }

    /// Release one pin and make the still-resident block MRU without counting another access.
    /// An inclusive tier uses this when an upper-tier resident becomes a cold lower-tier entry:
    /// the bytes stay in the same slot, but their replacement priority starts at the warm end.
    pub fn unpin_mru(&mut self, id: BlockId) {
        self.unpin(id);
        self.mark_mru(id);
    }

    /// How many distinct blocks are pinned right now — the number a caller sizes `n_slots`
    /// against (see [`Self::resolve_and_pin`]'s `None`).
    pub fn pinned_blocks(&self) -> usize {
        self.pinned.len()
    }

    /// Forget `id` and return its slot to the free list, if it was resident.
    ///
    /// For the caller whose fill FAILED: the slot holds partial bytes, so the block must not stay
    /// resident (the next request would be served that garbage as a hit). This is not a capacity
    /// eviction and is not counted as one — nothing was displaced to make room.
    ///
    /// # Panics
    /// If `id` is pinned. Removing a block someone is reading is the exact failure pins exist to
    /// prevent, so it is a caller bug, not a recoverable state.
    pub fn evict(&mut self, id: BlockId) -> Option<u32> {
        assert!(
            !self.pinned.contains_key(&id),
            "pager: evict of pinned block {id}"
        );
        let slot = self.resident.remove(&id)?;
        let removed = self.unlink_lru(slot);
        debug_assert_eq!(removed, Some(id));
        self.epoch.remove(&id);
        self.free.push(slot);
        Some(slot)
    }

    /// Replace one resident block with another in the SAME physical slot and put the replacement
    /// at the requested end of the LRU. This is the bookkeeping primitive used by an exclusive
    /// upper/lower cache swap: the promoted block leaves its RAM slot only after its bytes have
    /// reached VRAM, then the demoted VRAM block takes that exact RAM slot.
    ///
    /// No hit/miss counter is changed here. The access that caused the swap was already counted
    /// by the tier that served it; this operation only changes ownership of an existing slot.
    pub fn replace_resident(&mut self, old: BlockId, new: BlockId, insert: Insert) -> Option<u32> {
        assert!(
            !self.pinned.contains_key(&old),
            "pager: replacement of pinned block {old}"
        );
        assert!(
            !self.resident.contains_key(&new),
            "pager: replacement block {new} is already resident"
        );
        let slot = self.resident.remove(&old)?;
        let removed = self.unlink_lru(slot);
        debug_assert_eq!(removed, Some(old));
        self.epoch.remove(&old);
        self.resident.insert(new, slot);
        self.epoch.insert(new, self.cur_epoch);
        match insert {
            Insert::Mru => self.push_back_lru(slot, new),
            Insert::Cold => self.push_front_lru(slot, new),
        }
        Some(slot)
    }

    /// Move a resident block from its current enabled slot into one disabled spare slot, making
    /// the old slot the next spare. The bytes are NOT moved here; a backend first writes the
    /// promoted block into `spare`, then copies the evicted block out of the returned old slot.
    ///
    /// The LRU position is preserved exactly. In the tiered MoE path this follows [`Self::touch`]
    /// on a miss, so the moved block is already MRU and remains there while the physical exchange
    /// slot rotates underneath it.
    pub fn rotate_resident_to_spare(&mut self, id: BlockId, spare: u32) -> u32 {
        let spare_idx = self.slot_idx(spare);
        assert!(
            self.disabled[spare_idx],
            "pager: exchange slot {spare} is not disabled"
        );
        assert!(
            self.lru[spare_idx].id.is_none(),
            "pager: disabled exchange slot {spare} is occupied"
        );
        let old = *self
            .resident
            .get(&id)
            .expect("pager: only a resident block can rotate into a spare");
        let old_idx = self.slot_idx(old);
        assert!(
            !self.disabled[old_idx],
            "pager: resident block {id} is in disabled slot {old}"
        );

        let node = self.lru[old_idx];
        debug_assert_eq!(node.id, Some(id));
        if let Some(prev) = node.prev {
            let prev_idx = self.slot_idx(prev);
            self.lru[prev_idx].next = Some(spare);
        } else {
            self.lru_head = Some(spare);
        }
        if let Some(next) = node.next {
            let next_idx = self.slot_idx(next);
            self.lru[next_idx].prev = Some(spare);
        } else {
            self.lru_tail = Some(spare);
        }
        if self.cold_victim_cursor == Some(old) {
            self.cold_victim_cursor = Some(spare);
        }
        self.lru[spare_idx] = LruNode {
            id: Some(id),
            prev: node.prev,
            next: node.next,
        };
        self.lru[old_idx] = LruNode::default();
        self.resident.insert(id, spare);
        self.disabled[spare_idx] = false;
        self.disabled[old_idx] = true;
        old
    }

    /// Schedule-driven residency for a DETERMINISTIC cyclic sweep — the dense layer-streaming
    /// policy the module doc names (`BlockId = layer`, every forward pass visits blocks in the
    /// same fixed order). This is NOT demand/LRU: under a cyclic sweep the block whose next use
    /// is FURTHEST away is precisely the one the sweep just passed, so a miss inserts at the COLD
    /// end (it won't be needed again until every other block has been visited) and a hit keeps
    /// its position (promotion would let the sweep evict the stable set). The cache converges to
    /// a stable resident prefix of `n_slots - 1` blocks re-hit every pass plus one churn slot the
    /// remaining blocks cycle through — the same hit count Belady's offline-optimal policy
    /// achieves on a cyclic sweep (see `schedule_cyclic_sweep_is_belady_optimal`).
    ///
    /// Mechanically this shares [`Self::touch_cold`]'s cold-end insertion; it is a separate entry
    /// point because the CONTRACT differs: `touch_cold` promises scan-resistance for full-set
    /// sweeps MIXED with recency (`touch`) traffic on the same pager, while `schedule` promises
    /// Belady-parity for a pager driven EXCLUSIVELY by one fixed cyclic order (dense layer
    /// streaming never mixes policies on a pool). Within-batch safety still rides the batch
    /// epoch ([`Self::begin_batch`]), not insertion order.
    pub fn schedule(&mut self, id: BlockId) -> Resolution {
        self.touch_cold(id)
    }

    /// [`Self::touch`]'s SCAN-RESISTANT twin, for full-set sweeps (a batched-prefill layer
    /// pre-ensuring all `n_expert` experts, layer 0..n cyclically): hits keep their LRU position
    /// (no MRU promotion) and misses insert at the COLD end. Under plain LRU a cyclic sweep
    /// larger than the cache is the pathological worst case — every block evicted exactly before
    /// its next use, 0% steady-state reuse (measured on Scout pp512: 768/768 blocks re-uploaded
    /// per rep). Cold insertion makes the sweep churn a single victim region instead and
    /// converges to a stable resident prefix of ~`n_slots` blocks — near the offline-optimal
    /// hit count for a repeated sweep — while recency-driven [`Self::touch`] users (decode's
    /// routed-only path) keep their hot set at the MRU end, out of the sweep's victim zone.
    /// Within-batch safety comes from the batch epoch (see [`Self::begin_batch`]), NOT ordering.
    pub fn touch_cold(&mut self, id: BlockId) -> Resolution {
        self.epoch.insert(id, self.cur_epoch);
        if let Some(&slot) = self.resident.get(&id) {
            self.stats.hits += 1;
            return Resolution::Hit { slot };
        }
        self.stats.misses += 1;
        let (slot, evicted) = self.take_slot_cold();
        self.resident.insert(id, slot);
        self.push_front_lru(slot, id);
        Resolution::Miss { slot, evicted }
    }

    /// Speculatively admit `id` at the cold end without changing demand statistics.
    ///
    /// A resident prediction keeps its exact LRU position. Both hits and successful admissions
    /// join the current batch epoch, so later predictions cannot evict current-dispatch weights or
    /// an earlier prediction. `None` means every remaining slot is protected; speculative work
    /// simply stops instead of weakening the within-batch safety invariant.
    pub fn prefetch_cold(&mut self, id: BlockId) -> Option<Resolution> {
        if let Some(&slot) = self.resident.get(&id) {
            self.epoch.insert(id, self.cur_epoch);
            return Some(Resolution::Hit { slot });
        }
        let (slot, evicted) = self.take_slot_opt(false)?;
        self.epoch.insert(id, self.cur_epoch);
        self.resident.insert(id, slot);
        self.push_front_lru(slot, id);
        Some(Resolution::Miss { slot, evicted })
    }

    /// Cold-path counterpart of [`Self::take_slot`]. Within one batch it resumes from the node
    /// after the previous victim instead of restarting at `lru_head`: nodes it passes are current-
    /// epoch protected, the victim is removed, and newly admitted cold blocks are inserted before
    /// the cursor and are current-epoch protected too. Consequently every linked node is examined
    /// at most once per cold batch while victim choice remains identical to a fresh head scan.
    fn take_slot_cold(&mut self) -> (u32, Option<BlockId>) {
        if self.free.is_empty() {
            debug_assert!(
                self.pinned.is_empty(),
                "pins are for the fallible resolve_and_pin path; touch/schedule cannot see them"
            );
        }
        if let Some(s) = self.free.pop() {
            return (s, None);
        }

        let queue_len = self.lru_len;
        let mut scan_steps = 0usize;
        let mut cursor = self.cold_victim_cursor;
        let victim_slot = loop {
            let Some(slot) = cursor else {
                break None;
            };
            let node = self.lru[self.slot_idx(slot)];
            let id = node.id.expect("linked LRU node has a block id");
            scan_steps += 1;
            cursor = node.next;
            if self.epoch.get(&id) != Some(&self.cur_epoch) {
                break Some(slot);
            }
        };
        // Advance before unlinking: `victim_slot`'s node is cleared by `unlink_lru`, while its
        // saved successor remains linked and is exactly where the next cold miss must resume.
        self.cold_victim_cursor = cursor;
        if crate::pager_profile::active() {
            self.profile_work
                .record_victim_select(queue_len, scan_steps);
        }
        let victim_slot = victim_slot.unwrap_or_else(|| {
            assert!(
                self.cur_epoch == 0,
                "pager: a single dispatch batch touched more distinct blocks than the \
                 n_slots={} this cache holds — within-batch eviction safety is unsatisfiable. \
                 Refusing to degrade: evicting a block this same batch already resolved \
                 doesn't fail cleanly, it feeds the dispatch another block's bytes (silent \
                 wrong output, not an error). Raise the paging budget (INFR_CACHE) so every \
                 pool gets at least one slot per block a batch touches, or leave it unset to \
                 take the auto budget, which is sized to that floor.",
                self.n_slots
            );
            self.lru_head.expect("full pager has an LRU head")
        });
        let victim = self
            .unlink_lru(victim_slot)
            .expect("victim slot has an LRU node");
        let vslot = self
            .resident
            .remove(&victim)
            .expect("every lru entry has a resident mapping");
        debug_assert_eq!(vslot, victim_slot);
        self.epoch.remove(&victim);
        self.stats.evictions += 1;
        (vslot, Some(victim))
    }

    /// A free slot, evicting if none remain, or `None` when every resident block is protected
    /// (pinned, or touched by the current batch). [`Self::take_slot`] is the infallible form the
    /// unpinned callers use.
    fn take_slot_opt(&mut self, count_eviction: bool) -> Option<(u32, Option<BlockId>)> {
        if let Some(s) = self.free.pop() {
            return Some((s, None));
        }
        // A block is protected by its PIN always, and by the batch epoch only once batches are in
        // use: a caller that never opens one leaves `cur_epoch` at 0, which every block's default
        // epoch also is, so an unconditional epoch test would mark the whole cache as in-batch and
        // find no victim at all. (The infallible `take_slot` expresses the same exemption as its
        // `cur_epoch == 0` assert.)
        let batches = self.cur_epoch != 0;
        let cur_epoch = self.cur_epoch;
        let epoch = &self.epoch;
        let pinned = &self.pinned;
        let queue_len = self.lru_len;
        let (victim_slot, scan_steps) = self.first_lru_slot_matching(|b| {
            !(batches && epoch.get(&b) == Some(&cur_epoch)) && !pinned.contains_key(&b)
        });
        if crate::pager_profile::active() {
            self.profile_work
                .record_victim_select(queue_len, scan_steps);
        }
        let victim_slot = victim_slot?;
        let victim = self
            .unlink_lru(victim_slot)
            .expect("victim slot has an LRU node");
        let vslot = self
            .resident
            .remove(&victim)
            .expect("every lru entry has a resident mapping");
        debug_assert_eq!(vslot, victim_slot);
        // Drop the victim's epoch entry too — otherwise `epoch` is the one pager structure not
        // bounded by `n_slots`, accumulating a stale entry per distinct BlockId ever touched. A
        // stale entry could also mask an id as "current batch" across a `cur_epoch` wraparound.
        self.epoch.remove(&victim);
        if count_eviction {
            self.stats.evictions += 1;
        }
        Some((vslot, Some(victim)))
    }

    /// A free slot, evicting if none remain. The victim is the coldest block NOT touched in the
    /// current batch (epoch guard — see [`Self::begin_batch`]); legacy callers that never open a
    /// batch share epoch 0 everywhere, for which the guard degrades to plain LRU `pop_front`.
    fn take_slot(&mut self) -> (u32, Option<BlockId>) {
        if self.free.is_empty() {
            // Only reachable through the infallible entry points, whose callers hold no pins.
            debug_assert!(
                self.pinned.is_empty(),
                "pins are for the fallible resolve_and_pin path; touch/schedule cannot see them"
            );
        }
        if let Some(s) = self.free.pop() {
            return (s, None);
        }
        let cur_epoch = self.cur_epoch;
        let epoch = &self.epoch;
        let queue_len = self.lru_len;
        let (victim_slot, scan_steps) =
            self.first_lru_slot_matching(|b| epoch.get(&b) != Some(&cur_epoch));
        if crate::pager_profile::active() {
            self.profile_work
                .record_victim_select(queue_len, scan_steps);
        }
        let victim_slot = victim_slot
            // Every resident block touched by the CURRENT batch means the batch exceeded
            // n_slots — the sizing floor (`Pager::new`'s doc) was violated upstream. Falling
            // back to pop_front would silently evict a batch sibling mid-flight (the
            // coherent-but-wrong class); fail loudly instead when batches are in use.
            .unwrap_or_else(|| {
                // Actionable in the voice of the Vulkan VRAM guard (`check_vram_budget`): say what
                // was asked for, say why it cannot be tolerated rather than degraded, and name the
                // knob to turn. The knob is `INFR_CACHE` — the paging budget (`paging.cache`) the
                // seam splits proportionally into per-pool slot counts, so raising it is what makes
                // `n_slots` clear the batch's width. NOT reachable on a default configuration: the
                // MoE seam floors every pool at `min(n_expert, n_blocks)` slots (a batch is one
                // (layer, role) resolution, at most `n_expert` distinct experts) and the dense
                // streaming path opens a fresh batch around each single `schedule` call, so both
                // shipping callers satisfy the floor by construction — reaching this means an
                // explicit `INFR_CACHE` too small for one dispatch, or a new caller that skipped
                // the floor.
                assert!(
                    self.cur_epoch == 0,
                    "pager: a single dispatch batch touched more distinct blocks than the \
                     n_slots={} this cache holds — within-batch eviction safety is unsatisfiable. \
                     Refusing to degrade: evicting a block this same batch already resolved \
                     doesn't fail cleanly, it feeds the dispatch another block's bytes (silent \
                     wrong output, not an error). Raise the paging budget (INFR_CACHE) so every \
                     pool gets at least one slot per block a batch touches, or leave it unset to \
                     take the auto budget, which is sized to that floor.",
                    self.n_slots
                );
                self.lru_head.expect("full pager has an LRU head")
            });
        let victim = self
            .unlink_lru(victim_slot)
            .expect("victim slot has an LRU node");
        let vslot = self
            .resident
            .remove(&victim)
            .expect("every lru entry has a resident mapping");
        debug_assert_eq!(vslot, victim_slot);
        // Drop the victim's epoch entry too — otherwise `epoch` is the one pager structure not
        // bounded by `n_slots`, accumulating a stale entry per distinct BlockId ever touched. A
        // stale entry could also mask an id as "current batch" across a `cur_epoch` wraparound.
        self.epoch.remove(&victim);
        self.stats.evictions += 1;
        (vslot, Some(victim))
    }
}

impl Drop for Pager {
    fn drop(&mut self) {
        crate::pager_profile::record_lru_work(self.profile_work);
    }
}

/// Upload-ring sizing policy: the configured ring size (`paging.ring`, from `INFR_PAGER_RING` or a
/// config file, in the shared size grammar) wins; otherwise an eighth of the pager budget, clamped
/// to [256 MiB, 2 GiB].
///
/// Bigger halves = fewer pipeline rotations, and each rotation stalls the CPU on the other half's
/// fence — measured on Scout pp512 (miss-heavy steady state, ~22 GiB staged/rep): 256 MiB →
/// 224 t/s, 1 GiB → 324, 2 GiB → 404, flat beyond. The budget fraction keeps small explicit
/// `INFR_CACHE` runs from spending most of their grant on staging instead of arena slots.
///
/// Pure budget arithmetic — no device types — so the seam's placement math can price the ring
/// without reaching into a backend crate, and a second paging backend gets the same policy rather
/// than a fourth copy of the clamp. The staging ring ITSELF (the buffers, the fences, the half
/// rotation) stays per-backend.
pub fn ring_bytes(pager_budget: u64, ring: Option<crate::SizeSpec>) -> usize {
    const MIB: u64 = 1024 * 1024;
    // A zero (or absent) override can never be honoured — a 0-byte ring means "never stage a
    // slot" — so it falls through to the budget fraction, exactly as an unparseable value does.
    if let Some(b) = ring.map(|s| s.resolve(0) as usize).filter(|&b| b > 0) {
        return b;
    }
    (pager_budget / 8).clamp(256 * MIB, 2048 * MIB) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_slots_are_bounded_read_only_and_skip_explicit_pins() {
        let mut pager = Pager::new(6);
        for id in 10..14 {
            pager.touch(id);
        }
        pager.disable_slot(5);
        pager.repin(11).unwrap();
        let before = admission_state(&pager);
        assert_eq!(
            pager.replacement_slots().take(2).collect::<Vec<_>>(),
            vec![(None, 4), (Some(10), 0)]
        );
        assert_eq!(
            pager.replacement_slots().collect::<Vec<_>>(),
            vec![(None, 4), (Some(10), 0), (Some(12), 2), (Some(13), 3)]
        );
        assert_eq!(admission_state(&pager), before);
    }

    #[derive(Debug, PartialEq, Eq)]
    struct AdmissionState {
        n_slots: usize,
        disabled: Vec<bool>,
        enabled_slots: usize,
        resident: HashMap<BlockId, u32>,
        lru: Vec<(Option<BlockId>, Option<u32>, Option<u32>)>,
        lru_head: Option<u32>,
        lru_tail: Option<u32>,
        lru_len: usize,
        free: Vec<u32>,
        epoch: HashMap<BlockId, u64>,
        cur_epoch: u64,
        cold_victim_cursor: Option<u32>,
        pinned: HashMap<BlockId, u32>,
        stats: (u64, u64, u64),
        profile_work: crate::pager_profile::LruWorkStats,
    }

    fn admission_state(p: &Pager) -> AdmissionState {
        AdmissionState {
            n_slots: p.n_slots,
            disabled: p.disabled.clone(),
            enabled_slots: p.enabled_slots,
            resident: p.resident.clone(),
            lru: p.lru.iter().map(|n| (n.id, n.prev, n.next)).collect(),
            lru_head: p.lru_head,
            lru_tail: p.lru_tail,
            lru_len: p.lru_len,
            free: p.free.clone(),
            epoch: p.epoch.clone(),
            cur_epoch: p.cur_epoch,
            cold_victim_cursor: p.cold_victim_cursor,
            pinned: p.pinned.clone(),
            stats: (p.stats.hits, p.stats.misses, p.stats.evictions),
            profile_work: p.profile_work,
        }
    }

    fn mixed_handoff_pager() -> Pager {
        let mut pager = Pager::new(8);
        pager.begin_batch();
        for id in [7, 4_000, u32::MAX - 1, 99] {
            pager.touch(id);
        }
        pager.touch(7);
        pager.evict(4_000);
        pager.disable_slot(3);
        pager.disable_slot(6);
        pager.enable_slot(6);
        pager.disable_slot(5);
        pager
    }

    #[test]
    fn handoff_roundtrip_preserves_mixed_slots_orders_sparse_ids_and_stats() {
        let mut source = mixed_handoff_pager();
        let before = admission_state(&source);
        let snapshot = source.snapshot_for_handoff().unwrap();
        assert_eq!(admission_state(&source), before, "snapshot is read-only");
        assert_eq!(snapshot.n_slots(), 8);
        assert_eq!(
            snapshot.disabled(),
            &[false, false, false, true, false, true, false, false]
        );
        assert_eq!(snapshot.free(), &[7, 1, 6, 4]);
        assert_eq!(snapshot.resident_lru(), &[(u32::MAX - 1, 2), (7, 0)]);
        assert_eq!(
            snapshot.stats(),
            PagerStats {
                hits: 1,
                misses: 4,
                evictions: 1
            }
        );

        let mut restored = Pager::new(8);
        restored.touch(123);
        restored.disable_slot(0);
        restored.profile_work.mark_calls = 17;
        let profile_before = restored.profile_work;
        restored.restore_handoff(&snapshot).unwrap();
        assert_eq!(restored.snapshot_for_handoff().unwrap(), snapshot);
        assert_eq!(restored.profile_work, profile_before);
        assert_eq!(restored.enabled_slots(), 6);
        assert!(restored.epoch.is_empty());
        assert_eq!(restored.cur_epoch, 0);
        assert_eq!(restored.cold_victim_cursor, restored.lru_head);
        for &(id, slot) in snapshot.resident_lru() {
            assert_eq!(restored.slot_of(id), Some(slot));
        }

        // Verify behavior, not just serialization: pop every free slot in its original order,
        // then evict in the supplied oldest-to-newest order, using both insertion policies.
        for (id, slot, evicted) in [
            (100, 4, None),
            (101, 6, None),
            (102, 1, None),
            (103, 7, None),
            (104, 2, Some(u32::MAX - 1)),
            (105, 0, Some(7)),
        ] {
            source.begin_batch();
            restored.begin_batch();
            let expected = Resolution::Miss { slot, evicted };
            assert_eq!(source.touch(id), expected);
            assert_eq!(restored.touch(id), expected);
            assert_eq!(
                restored.snapshot_for_handoff(),
                source.snapshot_for_handoff()
            );
        }
        source.begin_batch();
        restored.begin_batch();
        assert_eq!(restored.touch_cold(106), source.touch_cold(106));
        assert_eq!(
            restored.snapshot_for_handoff(),
            source.snapshot_for_handoff()
        );
        assert_admission_invariants(&restored);
    }

    #[test]
    fn malformed_handoff_is_transactional_for_every_validation_failure() {
        let good = mixed_handoff_pager().snapshot_for_handoff().unwrap();
        let mut cases = Vec::new();

        let mut bad = good.clone();
        bad.disabled.pop();
        cases.push((
            bad,
            HandoffError::CapacityMismatch {
                expected: 8,
                actual: 7,
            },
        ));
        let mut bad = good.clone();
        bad.resident_lru[0].0 = NOT_RESIDENT;
        cases.push((bad, HandoffError::InvalidBlockId));
        let mut bad = good.clone();
        bad.resident_lru[1].0 = bad.resident_lru[0].0;
        cases.push((bad, HandoffError::DuplicateBlockId { id: u32::MAX - 1 }));
        let mut bad = good.clone();
        bad.resident_lru[1].1 = bad.resident_lru[0].1;
        cases.push((bad, HandoffError::SlotConflict { slot: 2 }));
        let mut bad = good.clone();
        bad.resident_lru[1].1 = good.free[0];
        cases.push((bad, HandoffError::SlotConflict { slot: 7 }));
        let mut bad = good.clone();
        bad.free.push(good.free[0]);
        cases.push((bad, HandoffError::SlotConflict { slot: 7 }));
        let mut bad = good.clone();
        bad.free.pop();
        cases.push((bad, HandoffError::MissingSlot { slot: 4 }));
        let mut bad = good.clone();
        bad.resident_lru.pop();
        cases.push((bad, HandoffError::MissingSlot { slot: 0 }));
        for resident in [false, true] {
            for (slot, error) in [
                (8, HandoffError::SlotOutOfRange { slot: 8 }),
                (u32::MAX, HandoffError::SlotOutOfRange { slot: u32::MAX }),
                (3, HandoffError::SlotDisabled { slot: 3 }),
            ] {
                let mut bad = good.clone();
                if resident {
                    bad.resident_lru[0].1 = slot;
                } else {
                    bad.free[0] = slot;
                }
                cases.push((bad, error));
            }
        }

        let mut destination = mixed_handoff_pager();
        destination.begin_batch();
        destination.touch_cold(7);
        destination.profile_work.victim_calls = 11;
        let before = admission_state(&destination);
        for (snapshot, error) in cases {
            assert_eq!(destination.validate_handoff(&snapshot), Err(error));
            assert_eq!(admission_state(&destination), before);
            assert_eq!(destination.restore_handoff(&snapshot), Err(error));
            assert_eq!(admission_state(&destination), before);
        }
    }

    #[test]
    fn handoff_rejects_nested_active_pins_without_changing_any_state() {
        let mut pager = mixed_handoff_pager();
        let snapshot = pager.snapshot_for_handoff().unwrap();
        pager.repin(7).unwrap();
        pager.repin(7).unwrap();
        for _ in 0..2 {
            let before = admission_state(&pager);
            let error = HandoffError::ActivePins { blocks: 1 };
            assert_eq!(pager.snapshot_for_handoff(), Err(error));
            assert_eq!(pager.validate_handoff(&snapshot), Err(error));
            assert_eq!(admission_state(&pager), before);
            assert_eq!(pager.restore_handoff(&snapshot), Err(error));
            assert_eq!(admission_state(&pager), before);
            pager.unpin(7);
        }
        assert_eq!(pager.snapshot_for_handoff().unwrap(), snapshot);
        pager.restore_handoff(&snapshot).unwrap();
    }

    #[test]
    fn handoff_validation_accepts_new_ownership_without_changing_destination() {
        let mut pager = mixed_handoff_pager();
        pager.begin_batch();
        pager.touch_cold(7);
        pager.profile_work.victim_calls = 11;
        let mut snapshot = pager.snapshot_for_handoff().unwrap();
        snapshot.free.reverse();
        snapshot.resident_lru.reverse();
        snapshot.stats.hits += 5;
        let before = admission_state(&pager);
        let snapshot_before = snapshot.clone();
        let shared: &Pager = &pager;
        assert_eq!(shared.validate_handoff(&snapshot), Ok(()));
        assert_eq!(shared.validate_handoff(&snapshot), Ok(()));
        assert_eq!(admission_state(&pager), before);
        assert_eq!(snapshot, snapshot_before);

        pager.restore_handoff(&snapshot).unwrap();
        assert_eq!(pager.snapshot_for_handoff().unwrap(), snapshot);
    }

    #[test]
    fn handoff_validation_can_reject_a_later_pool_before_restoring_any_pool() {
        let mut first = mixed_handoff_pager();
        let mut second = mixed_handoff_pager();
        let first_snapshot = first.snapshot_for_handoff().unwrap();
        let second_snapshot = second.snapshot_for_handoff().unwrap();
        let mut invalid_second = second_snapshot.clone();
        invalid_second.resident_lru[0].1 = 3;
        let first_before = admission_state(&first);
        let second_before = admission_state(&second);

        let validation = first
            .validate_handoff(&first_snapshot)
            .and_then(|()| second.validate_handoff(&invalid_second));
        assert_eq!(validation, Err(HandoffError::SlotDisabled { slot: 3 }));
        assert_eq!(admission_state(&first), first_before);
        assert_eq!(admission_state(&second), second_before);

        first.validate_handoff(&first_snapshot).unwrap();
        second.validate_handoff(&second_snapshot).unwrap();
        first.restore_handoff(&first_snapshot).unwrap();
        second.restore_handoff(&second_snapshot).unwrap();
        assert_eq!(first.snapshot_for_handoff().unwrap(), first_snapshot);
        assert_eq!(second.snapshot_for_handoff().unwrap(), second_snapshot);
    }

    #[test]
    fn handoff_imports_gpu_order_and_cumulative_stats_without_access_replay() {
        let stats = PagerStats {
            hits: 900,
            misses: 70,
            evictions: 40,
        };
        let snapshot = PagerSnapshot::from_parts(
            vec![false, true, false, false],
            vec![3],
            vec![(u32::MAX - 1, 2), (0, 0)],
            stats,
        );
        let mut pager = Pager::new(4);
        pager.begin_batch();
        pager.touch(100);
        pager.restore_handoff(&snapshot).unwrap();
        assert_eq!(pager.stats(), stats);
        assert_eq!(pager.snapshot_for_handoff().unwrap(), snapshot);
        assert_eq!(
            pager.touch(1),
            Resolution::Miss {
                slot: 3,
                evicted: None
            }
        );
        // No old/current-batch guard may prevent the GPU's oldest resident being evicted.
        assert_eq!(
            pager.touch_cold(2),
            Resolution::Miss {
                slot: 2,
                evicted: Some(u32::MAX - 1)
            }
        );
        assert_eq!(
            pager.stats(),
            PagerStats {
                hits: 900,
                misses: 72,
                evictions: 41
            }
        );
        assert_eq!(pager.slot_of(0), Some(0));
    }

    #[test]
    fn handoff_handles_empty_full_and_all_disabled_caches() {
        for (disabled, free, residents) in [
            (vec![false; 2], vec![1, 0], vec![]),
            (vec![false; 2], vec![], vec![(0, 1), (u32::MAX - 1, 0)]),
            (vec![true; 2], vec![], vec![]),
        ] {
            let snapshot =
                PagerSnapshot::from_parts(disabled, free, residents, PagerStats::default());
            let mut pager = Pager::new(2);
            pager.restore_handoff(&snapshot).unwrap();
            assert_eq!(pager.snapshot_for_handoff().unwrap(), snapshot);
            // Enabling a spare after restore must retain every other slot's ownership/order.
            if pager.enabled_slots() == 0 {
                pager.enable_slot(1);
                assert_eq!(
                    pager.touch(8),
                    Resolution::Miss {
                        slot: 1,
                        evicted: None
                    }
                );
            }
        }
    }

    fn assert_admission_rejected(p: &mut Pager, id: BlockId, slot: u32, error: AdmissionError) {
        let before = admission_state(p);
        assert_eq!(p.admit_at(id, slot), Err(error));
        assert_eq!(admission_state(p), before, "a refusal must change no state");
    }

    fn assert_admission_invariants(p: &Pager) {
        assert_eq!(p.enabled_slots, p.disabled.iter().filter(|&&d| !d).count());
        assert_eq!(p.lru_len, p.resident.len());
        assert_eq!(p.free.len() + p.resident.len(), p.enabled_slots);
        assert_eq!(p.epoch.len(), p.resident.len());
        let mut seen = vec![false; p.n_slots];
        let mut prev = None;
        let mut slot = p.lru_head;
        let mut linked = 0;
        while let Some(s) = slot {
            let idx = s as usize;
            assert!(!seen[idx], "duplicate slot or LRU cycle");
            seen[idx] = true;
            let node = p.lru[idx];
            let id = node.id.expect("linked slot must have an owner");
            assert!(!p.disabled[idx]);
            assert_eq!(p.slot_of(id), Some(s));
            assert!(p.epoch.contains_key(&id));
            assert_eq!(node.prev, prev);
            prev = Some(s);
            slot = node.next;
            linked += 1;
        }
        assert_eq!(prev, p.lru_tail);
        assert_eq!(linked, p.lru_len);
        if let Some(cursor) = p.cold_victim_cursor {
            assert!(
                seen[cursor as usize],
                "cold cursor must point to a linked slot"
            );
        }
        for &s in &p.free {
            let idx = s as usize;
            assert!(!seen[idx], "free slot is duplicated or still resident");
            seen[idx] = true;
            assert!(!p.disabled[idx]);
            assert!(p.lru[idx].id.is_none());
        }
        for (idx, &disabled) in p.disabled.iter().enumerate() {
            assert_eq!(seen[idx], !disabled);
            if p.lru[idx].id.is_none() {
                assert!(p.lru[idx].prev.is_none() && p.lru[idx].next.is_none());
            }
        }
        for (&id, &count) in &p.pinned {
            assert!(count > 0 && p.resident.contains_key(&id));
        }
    }

    #[test]
    fn admission_rejects_invalid_ids_slots_and_resident_mismatches_without_mutation() {
        let mut p = Pager::new(4);
        p.touch(10);
        p.touch(11);
        p.begin_batch();
        p.disable_slot(3);
        assert_admission_rejected(&mut p, NOT_RESIDENT, 2, AdmissionError::InvalidBlockId);
        for slot in [4, u32::MAX] {
            assert_admission_rejected(&mut p, 12, slot, AdmissionError::SlotOutOfRange);
        }
        assert_admission_rejected(&mut p, 12, 3, AdmissionError::SlotDisabled);
        // A mismatch must not evict either an occupied target or consume a free target.
        for slot in [1, 2] {
            assert_admission_rejected(
                &mut p,
                10,
                slot,
                AdmissionError::ResidentElsewhere { slot: 0 },
            );
        }
        assert_admission_invariants(&p);
    }

    #[test]
    fn admission_preserves_explicit_pins_and_current_batch_owners() {
        let mut p = Pager::new(3);
        p.resolve_and_pin(10, Insert::Mru).unwrap();
        p.repin(10).unwrap();
        p.touch(11);
        p.begin_batch();
        p.touch(11);
        // Refuse the exact protected target even though slot 2 is free.
        assert_admission_rejected(&mut p, 12, 0, AdmissionError::Pinned { id: 10 });
        assert_admission_rejected(&mut p, 12, 1, AdmissionError::CurrentBatch { id: 11 });
        assert_eq!(p.admit_at(10, 0), Ok(Resolution::Hit { slot: 0 }));
        assert_eq!(p.pinned.get(&10), Some(&2), "a hit acquires no new pin");
        p.unpin(10);
        p.begin_batch();
        assert_admission_rejected(&mut p, 12, 0, AdmissionError::Pinned { id: 10 });
        p.unpin(10);
        assert_eq!(
            p.admit_at(12, 0),
            Ok(Resolution::Miss {
                slot: 0,
                evicted: Some(10),
            })
        );
        assert_eq!(p.pinned_blocks(), 0);
        assert_admission_invariants(&p);
    }

    #[test]
    fn admission_hit_marks_mru_counts_once_and_protects_the_epoch() {
        let mut p = Pager::new(3);
        for id in 10..13 {
            p.touch(id);
        }
        p.begin_batch();
        assert_eq!(p.admit_at(10, 0), Ok(Resolution::Hit { slot: 0 }));
        assert_eq!(p.lru_order(), vec![11, 12, 10]);
        assert_eq!(p.stats.hits, 1);
        assert_eq!((p.stats.misses, p.stats.evictions), (3, 0));
        assert_eq!(p.epoch.get(&10), Some(&p.cur_epoch));
        assert_admission_rejected(&mut p, 20, 0, AdmissionError::CurrentBatch { id: 10 });
        assert_eq!(p.admit_at(10, 0), Ok(Resolution::Hit { slot: 0 }));
        assert_eq!(p.stats.hits, 2);
        assert_eq!(p.lru_order(), vec![11, 12, 10]);
        assert_admission_invariants(&p);
    }

    #[test]
    fn admission_consumes_any_free_slot_without_reordering_or_duplicate_allocation() {
        for target in 0..4u32 {
            let mut p = Pager::new(4);
            let remaining: Vec<_> = (0..4u32).filter(|&s| s != target).collect();
            assert_eq!(
                p.admit_at(10, target),
                Ok(Resolution::Miss {
                    slot: target,
                    evicted: None,
                })
            );
            assert_admission_invariants(&p);
            for (id, slot) in (11..).zip(remaining) {
                assert_eq!(
                    p.touch(id),
                    Resolution::Miss {
                        slot,
                        evicted: None
                    }
                );
                assert_admission_invariants(&p);
            }
            assert_eq!(p.stats.evictions, 0);
            assert_eq!(p.stats.misses, 4);
            assert_eq!(p.slot_of(10), Some(target));
        }
    }

    #[test]
    fn admission_reuses_evicted_and_reenabled_slots() {
        let mut p = Pager::new(3);
        p.touch(10);
        p.touch(11);
        assert_eq!(p.disable_slot(0), Some(10));
        assert_admission_rejected(&mut p, 12, 0, AdmissionError::SlotDisabled);
        p.enable_slot(0);
        assert_eq!(
            p.admit_at(12, 0),
            Ok(Resolution::Miss {
                slot: 0,
                evicted: None
            })
        );
        assert_admission_invariants(&p);
        assert_eq!(p.evict(12), Some(0));
        assert_eq!(
            p.admit_at(13, 0),
            Ok(Resolution::Miss {
                slot: 0,
                evicted: None
            })
        );
        assert_eq!(
            p.touch(14),
            Resolution::Miss {
                slot: 2,
                evicted: None
            }
        );
        assert_eq!(
            p.stats.evictions, 1,
            "only disable_slot displaced a resident"
        );
        assert_admission_invariants(&p);
    }

    #[test]
    fn admission_replaces_the_selected_head_middle_or_tail_even_with_free_slots() {
        for target in 0..4u32 {
            let mut p = Pager::new(5);
            for id in 0..4 {
                p.touch(id);
            }
            p.begin_batch();
            let free = p.free.clone();
            assert_eq!(
                p.admit_at(10, target),
                Ok(Resolution::Miss {
                    slot: target,
                    evicted: Some(target),
                })
            );
            let mut order: Vec<_> = (0..4).filter(|&id| id != target).collect();
            order.push(10);
            assert_eq!(p.lru_order(), order);
            assert_eq!(p.free, free);
            assert_eq!(p.slot_of(target), None);
            assert_eq!(p.slot_of(10), Some(target));
            assert!(!p.epoch.contains_key(&target));
            assert_eq!((p.stats.hits, p.stats.misses, p.stats.evictions), (0, 5, 1));
            assert_admission_rejected(&mut p, 11, target, AdmissionError::CurrentBatch { id: 10 });
            assert_admission_invariants(&p);
        }
    }

    #[test]
    fn admission_supports_legacy_epoch_zero_and_bounds_ownership_metadata() {
        let mut p = Pager::new(1);
        for id in 0..100 {
            assert_eq!(
                p.admit_at(id, 0),
                Ok(Resolution::Miss {
                    slot: 0,
                    evicted: id.checked_sub(1),
                })
            );
            assert_eq!(p.slot_of(id), Some(0));
            assert_eq!(p.lru_order(), vec![id]);
            assert_admission_invariants(&p);
        }
        assert_eq!(
            (p.stats.hits, p.stats.misses, p.stats.evictions),
            (0, 100, 99)
        );
    }

    #[test]
    fn admission_keeps_cold_cursor_valid_after_hits_and_selected_evictions() {
        for hit in [false, true] {
            for target in 0..4u32 {
                let mut p = Pager::new(4);
                for id in 0..4 {
                    p.touch(id);
                }
                p.begin_batch();
                let id = if hit { target } else { 10 };
                let result = p.admit_at(id, target).unwrap();
                assert_eq!(matches!(result, Resolution::Hit { .. }), hit);
                assert_admission_invariants(&p);
                for (new, victim) in (20..).zip((0..4).filter(|&old| old != target)) {
                    assert_eq!(
                        p.touch_cold(new),
                        Resolution::Miss {
                            slot: victim,
                            evicted: Some(victim),
                        }
                    );
                    assert_admission_invariants(&p);
                }
                assert_eq!(p.slot_of(id), Some(target));
                assert_eq!(p.prefetch_cold(30), None);
                p.begin_batch();
                assert!(matches!(p.touch_cold(30), Resolution::Miss { .. }));
                assert_admission_invariants(&p);
            }
        }
    }

    #[test]
    fn admission_reconciles_a_simulated_gpu_lut_and_physical_owners() {
        let mut p = Pager::new(4);
        let mut lut = [NOT_RESIDENT; 8];
        let mut physical = [None; 4];
        let selected = [
            (0, 2),
            (1, 0),
            (2, 3),
            (0, 2),
            (3, 1),
            (4, 0),
            (5, 2),
            (2, 3),
            (1, 1),
            (4, 0),
            (6, 3),
        ];
        for (id, slot) in selected {
            p.begin_batch();
            // The backend has already copied bytes and published its ownership/LUT change.
            let old = physical[slot as usize].replace(id);
            if let Some(old) = old {
                if old != id {
                    lut[old as usize] = NOT_RESIDENT;
                }
            }
            lut[id as usize] = slot;
            let expected = if old == Some(id) {
                Resolution::Hit { slot }
            } else {
                Resolution::Miss { slot, evicted: old }
            };
            assert_eq!(p.admit_at(id, slot), Ok(expected));
            for (id, &mapped) in lut.iter().enumerate() {
                assert_eq!(p.slot_of(id as u32).unwrap_or(NOT_RESIDENT), mapped);
                if mapped != NOT_RESIDENT {
                    assert_eq!(physical[mapped as usize], Some(id as u32));
                }
            }
            assert_admission_invariants(&p);
        }
        assert!(p.stats.hits > 0 && p.stats.evictions > 0);
    }

    #[test]
    fn admission_matches_normal_demand_lru_epochs_and_statistics() {
        let mut normal = Pager::new(4);
        let mut replay = Pager::new(4);
        for id in [0, 1, 2, 0, 1, 3, 4, 0, 1, 5, 0, 1, 6, 7, 0, 1] {
            normal.begin_batch();
            replay.begin_batch();
            let expected = normal.touch(id);
            let slot = match expected {
                Resolution::Hit { slot } | Resolution::Miss { slot, .. } => slot,
            };
            assert_eq!(replay.admit_at(id, slot), Ok(expected));
            assert_eq!(replay.resident_slots_lru(), normal.resident_slots_lru());
            assert_eq!(replay.epoch, normal.epoch);
            assert_eq!(replay.free, normal.free);
            assert_eq!(replay.cold_victim_cursor, normal.cold_victim_cursor);
            assert_eq!(
                (
                    replay.stats.hits,
                    replay.stats.misses,
                    replay.stats.evictions
                ),
                (
                    normal.stats.hits,
                    normal.stats.misses,
                    normal.stats.evictions
                ),
            );
            assert_admission_invariants(&replay);
        }
    }

    #[test]
    fn legacy_mru_then_cold_touch_preserves_remaining_victims() {
        let mut pager = Pager::new(4);
        for id in 0..4 {
            pager.touch(id);
        }
        pager.begin_batch();
        pager.touch(0);
        assert!(matches!(
            pager.touch_cold(4),
            Resolution::Miss {
                evicted: Some(1),
                ..
            }
        ));
        assert!(pager.slot_of(0).is_some());
        assert!(pager.slot_of(4).is_some());
    }

    /// The ring clamp's boundaries, pinning the pre-hoist inline expression
    /// `(budget / 8).clamp(256 MiB, 2048 MiB)` — the seam subtracts this from the pager budget
    /// before splitting arena shares, so a moved edge silently re-sizes every MoE arena — plus the
    /// override's fall-through rule.
    ///
    /// Driven through [`ring_bytes`] itself, over resolved [`SizeSpec`] values. It used to go
    /// through a `&str` façade (`ring_bytes_from`) so the sweep could avoid the process
    /// environment, which the value-taking function already allows; the SPELLINGS it exercised
    /// (`1GiB`, `1g`, `512m`, a bare byte count, and the unparseable `banana`) are
    /// `crate::parse_size`'s grammar and are pinned by `parse_size_grammar` and by
    /// `config`'s `mib_and_size_string_spellings` — everything unparseable arrives here as `None`.
    #[test]
    fn ring_bytes_clamp_boundaries_and_override_grammar() {
        const MIB: u64 = 1024 * 1024;
        // Below the floor's crossover (8 x 256 MiB = 2 GiB of budget) the floor wins — including
        // the `0` budget the pager passes when it has no budget figure at all.
        assert_eq!(ring_bytes(0, None), 256 * MIB as usize);
        assert_eq!(ring_bytes(2048 * MIB, None), 256 * MIB as usize);
        // Exactly at the crossover, and just past it, the eighth wins.
        assert_eq!(ring_bytes(2048 * MIB + 8, None), (256 * MIB + 1) as usize);
        // At the ceiling's crossover (8 x 2 GiB) and beyond, the ceiling wins.
        assert_eq!(ring_bytes(16 * 1024 * MIB, None), (2048 * MIB) as usize);
        assert_eq!(ring_bytes(64 * 1024 * MIB, None), (2048 * MIB) as usize);

        // An explicit override wins outright; a ZERO one (or an absent/unparseable one, which the
        // config layer turns into `None`) falls through to the policy — a `0` ring would mean
        // "never stage a slot".
        let bytes = crate::SizeSpec::Bytes;
        for (spec, want) in [
            (Some(bytes(1024 * MIB)), Some(1024 * MIB as usize)),
            (Some(bytes(512 * MIB)), Some(512 * MIB as usize)),
            (Some(bytes(3_221_225_472)), Some(3 * 1024 * MIB as usize)),
            (Some(bytes(0)), None),
            (None, None),
        ] {
            let got = ring_bytes(64 * 1024 * MIB, spec);
            match want {
                Some(w) => assert_eq!(got, w, "spec={spec:?}"),
                // Falls through to the policy, which caps at 2 GiB on this budget.
                None => assert_eq!(got, (2048 * MIB) as usize, "spec={spec:?}"),
            }
        }
        // A PERCENT override resolves against a zero base (the ring has no base to scale), so it
        // is zero and falls through too — the same arm the `0` case above takes.
        assert_eq!(
            ring_bytes(64 * 1024 * MIB, Some(crate::SizeSpec::Percent(0.5))),
            (2048 * MIB) as usize
        );
    }

    /// The override as the read sites now take it: a `paging.ring` field off the backend's
    /// `Config`, already through the shared size grammar. `None` (unset, or a value the grammar
    /// rejects) falls through to the budget fraction.
    #[test]
    fn ring_bytes_takes_the_override_from_the_config() {
        use crate::config::Config;
        const MIB: u64 = 1024 * 1024;
        assert_eq!(Config::default().paging.ring, None);
        assert_eq!(
            ring_bytes(64 * 1024 * MIB, Config::default().paging.ring),
            (2048 * MIB) as usize
        );
        for (raw, want) in [
            ("1GiB", 1024 * MIB as usize),
            ("1g", 1024 * MIB as usize),
            ("512m", 512 * MIB as usize),
            ("banana", (2048 * MIB) as usize), // rejected by the grammar ⇒ field stays None
            ("0", (2048 * MIB) as usize),      // a 0-byte ring is not a ring
        ] {
            let layer =
                crate::config::env::parse(&|k| (k == "INFR_PAGER_RING").then(|| raw.to_string()))
                    .expect("env layer");
            let cfg = Config::load_from_layers(&[layer]);
            assert_eq!(
                ring_bytes(64 * 1024 * MIB, cfg.paging.ring),
                want,
                "INFR_PAGER_RING={raw:?}"
            );
        }
    }

    #[test]
    fn fresh_pager_is_all_misses_until_full() {
        let mut p = Pager::new(3);
        assert_eq!(
            p.touch(10),
            Resolution::Miss {
                slot: 0,
                evicted: None
            }
        );
        assert_eq!(
            p.touch(11),
            Resolution::Miss {
                slot: 1,
                evicted: None
            }
        );
        assert_eq!(
            p.touch(12),
            Resolution::Miss {
                slot: 2,
                evicted: None
            }
        );
        assert_eq!(p.resident_count(), 3);
        assert_eq!(p.stats().misses, 3);
        assert_eq!(p.stats().hits, 0);
    }

    #[test]
    fn repeat_touch_is_a_hit_at_the_same_slot() {
        let mut p = Pager::new(2);
        let Resolution::Miss { slot, .. } = p.touch(5) else {
            panic!("expected miss")
        };
        assert_eq!(p.touch(5), Resolution::Hit { slot });
        assert_eq!(p.touch(5), Resolution::Hit { slot });
        assert_eq!(p.stats().hits, 2);
        assert_eq!(p.stats().misses, 1);
    }

    #[test]
    fn eviction_picks_least_recently_used() {
        let mut p = Pager::new(2);
        p.touch(1); // slot 0
        p.touch(2); // slot 1, full now
        p.touch(1); // hit, 1 is now MRU; 2 is LRU
                    // 3 must evict 2 (LRU), not 1 (just touched).
        let Resolution::Miss { slot, evicted } = p.touch(3) else {
            panic!("expected miss")
        };
        assert_eq!(evicted, Some(2));
        assert_eq!(p.slot_of(1), Some(0)); // 1 kept its original slot
        assert_eq!(p.slot_of(3), Some(slot));
        assert_eq!(p.slot_of(2), None);
        assert_eq!(p.stats().evictions, 1);
    }

    #[test]
    fn resident_slot_snapshot_is_cold_to_hot_and_tracks_physical_slots() {
        let mut p = Pager::new(3);
        p.touch(10); // slot 0
        p.touch(11); // slot 1
        p.touch(12); // slot 2
        p.touch(10); // move slot 0 to MRU

        assert_eq!(p.resident_slots_lru(), vec![(11, 1), (12, 2), (10, 0)]);

        p.evict(12);
        assert_eq!(p.resident_slots_lru(), vec![(11, 1), (10, 0)]);
    }

    #[test]
    fn explicit_evict_unlinks_head_middle_and_tail() {
        let mut p = Pager::new(4);
        for id in [1, 2, 3, 4] {
            p.touch(id);
        }
        assert_eq!(p.lru_order(), vec![1, 2, 3, 4]);

        assert!(p.evict(1).is_some());
        assert_eq!(p.lru_order(), vec![2, 3, 4]);

        assert!(p.evict(3).is_some());
        assert_eq!(p.lru_order(), vec![2, 4]);

        assert!(p.evict(4).is_some());
        assert_eq!(p.lru_order(), vec![2]);

        p.touch(5);
        assert_eq!(p.lru_order(), vec![2, 5]);
    }

    #[test]
    fn slot_reuse_after_eviction_is_exact() {
        // The freed slot from an eviction is the ONLY slot available for the next miss (n_slots
        // fixed) — assert the pager actually reuses it rather than growing.
        let mut p = Pager::new(1);
        let Resolution::Miss { slot: s0, .. } = p.touch(1) else {
            panic!()
        };
        let Resolution::Miss {
            slot: s1,
            evicted: e1,
        } = p.touch(2)
        else {
            panic!()
        };
        assert_eq!(s0, s1); // only one slot exists, must be reused
        assert_eq!(e1, Some(1));
        assert_eq!(p.resident_count(), 1);
    }

    #[test]
    fn within_batch_touch_order_protects_earlier_ids_from_later_ones() {
        // Simulates a decode step's top-k = 3 experts touched in a fixed order against a 3-slot
        // cache with 1 already resident from a PRIOR step — the prior step's expert should be the
        // only eviction victim; none of the 3 in-flight ids may evict each other.
        let mut p = Pager::new(3);
        p.touch(100); // prior step's expert, now LRU-oldest
        for id in [7u32, 8, 9] {
            p.touch(id);
        }
        // Cache is now full (100 evicted by whichever of 7/8/9 needed a slot first); crucially,
        // ALL of 7, 8, 9 must be resident simultaneously — that's the within-batch invariant.
        assert!(p.slot_of(7).is_some());
        assert!(p.slot_of(8).is_some());
        assert!(p.slot_of(9).is_some());
        assert_eq!(p.slot_of(100), None);
    }

    #[test]
    fn lut_coherence_across_a_simulated_token_sequence() {
        // Drives a small cache through a token sequence with a repeating expert-access pattern
        // (like a real MoE decode loop) and checks the (block_id -> slot) mapping stays exactly
        // what a host-mirrored LUT array would need at every step.
        let mut p = Pager::new(4);
        let mut lut = vec![NOT_RESIDENT; 16]; // host mirror of the device LUT buffer
        let apply = |p: &mut Pager, lut: &mut [u32], id: BlockId| match p.touch(id) {
            Resolution::Hit { slot } => assert_eq!(lut[id as usize], slot, "hit must match LUT"),
            Resolution::Miss { slot, evicted } => {
                if let Some(e) = evicted {
                    lut[e as usize] = NOT_RESIDENT;
                }
                lut[id as usize] = slot;
            }
        };
        // A token sequence revisiting a hot set {0,1} plus a cold long tail.
        let seq = [0u32, 1, 2, 0, 1, 3, 4, 0, 1, 5, 0, 1, 6, 7, 0, 1];
        for &id in &seq {
            apply(&mut p, &mut lut, id);
        }
        // The hot set must still be resident (never evicted across the whole run) and every
        // resident block's LUT entry must exactly match the pager's own view.
        for id in [0u32, 1] {
            let slot = p.slot_of(id).expect("hot id stays resident");
            assert_eq!(lut[id as usize], slot);
        }
        for id in 0..16u32 {
            match p.slot_of(id) {
                Some(slot) => assert_eq!(lut[id as usize], slot),
                None => assert_eq!(lut[id as usize], NOT_RESIDENT),
            }
        }
        assert!(p.stats().hits > 0);
        assert!(p.stats().evictions > 0);
    }

    #[test]
    fn cold_touch_sweep_converges_to_a_stable_prefix() {
        // A cyclic sweep of 6 blocks through 3 slots: plain LRU would miss EVERY touch from the
        // second pass on (the pathological case touch_cold exists for); cold insertion must
        // instead keep a stable resident prefix and only churn the cold end.
        let mut p = Pager::new(3);
        for _pass in 0..4 {
            for id in 0..6u32 {
                p.begin_batch(); // one "layer" per id in this reduced shape
                p.touch_cold(id);
            }
        }
        // Steady state: passes 2..4 must hit the stable prefix — strictly more than plain LRU's
        // zero. (Exact count: 2 hits/pass here — n_slots-1 stable + 1 churn slot.)
        assert!(
            p.stats().hits >= 6,
            "cold sweep should retain a stable prefix (hits={})",
            p.stats().hits
        );
        // And the prefix blocks are still resident for the next pass.
        assert!(p.slot_of(0).is_some());
        assert!(p.slot_of(1).is_some());
    }

    #[test]
    fn cold_touch_batch_cannot_evict_its_own_siblings() {
        // Within-batch safety under COLD insertion: LRU position no longer protects batch
        // siblings (they sit at the cold end), so the epoch guard must. A 3-slot cache fully
        // occupied by an old batch, then one batch cold-touching 3 fresh blocks: all 3 must be
        // simultaneously resident at the end — no sibling may evict another.
        let mut p = Pager::new(3);
        p.begin_batch();
        for id in [100u32, 101, 102] {
            p.touch_cold(id);
        }
        p.begin_batch();
        for id in [7u32, 8, 9] {
            p.touch_cold(id);
        }
        for id in [7u32, 8, 9] {
            assert!(p.slot_of(id).is_some(), "batch sibling {id} was evicted");
        }
    }

    #[test]
    fn cold_prefetch_preserves_order_safety_and_demand_stats() {
        let mut p = Pager::new(3);
        p.begin_batch();
        for id in [1u32, 2, 3] {
            p.touch(id);
        }

        p.begin_batch();
        p.touch(3); // current dispatch protects 3
        let before = p.stats();
        assert_eq!(p.lru_order(), vec![1, 2, 3]);

        assert!(matches!(p.prefetch_cold(2), Some(Resolution::Hit { .. })));
        assert_eq!(p.lru_order(), vec![1, 2, 3], "a predicted hit stays put");
        assert!(matches!(
            p.prefetch_cold(4),
            Some(Resolution::Miss {
                evicted: Some(1),
                ..
            })
        ));
        assert_eq!(p.lru_order(), vec![4, 2, 3]);
        assert_eq!(
            p.prefetch_cold(5),
            None,
            "current demand and admitted predictions exhaust safe victims"
        );
        let after = p.stats();
        assert_eq!(
            (after.hits, after.misses, after.evictions),
            (before.hits, before.misses, before.evictions),
            "speculation must not pollute demand cache statistics"
        );

        p.begin_batch();
        let hits = p.stats().hits;
        assert!(matches!(p.touch(4), Resolution::Hit { .. }));
        assert_eq!(p.stats().hits, hits + 1);
        assert_eq!(p.lru_order(), vec![2, 3, 4]);
    }

    #[test]
    fn cold_batch_cursor_keeps_the_same_victim_order_as_head_scans() {
        let mut p = Pager::new(4);
        p.begin_batch();
        for id in [10u32, 11, 12, 13] {
            p.touch_cold(id);
        }

        // The cold order is [13, 12, 11, 10]. Protecting 13 makes 12 the first victim; later
        // misses must resume at 11 and 10 rather than revisiting the protected prefix.
        p.begin_batch();
        assert!(matches!(p.touch_cold(13), Resolution::Hit { .. }));
        for (id, expected_victim) in [(20u32, 12u32), (21, 11), (22, 10)] {
            assert!(matches!(
                p.touch_cold(id),
                Resolution::Miss {
                    evicted: Some(victim),
                    ..
                } if victim == expected_victim
            ));
        }
        for id in [13u32, 20, 21, 22] {
            assert!(p.slot_of(id).is_some());
        }

        // A new batch resets the cursor to the new cold head.
        p.begin_batch();
        assert!(matches!(
            p.touch_cold(30),
            Resolution::Miss {
                evicted: Some(22),
                ..
            }
        ));
    }

    #[test]
    #[should_panic(expected = "within-batch eviction safety")]
    fn batch_larger_than_cache_fails_loudly() {
        // The sizing-floor violation must panic (silent sibling eviction = the
        // coherent-but-wrong output class), not degrade.
        let mut p = Pager::new(2);
        p.begin_batch();
        for id in 0..3u32 {
            p.touch_cold(id);
        }
    }

    #[test]
    #[should_panic(expected = "n_slots must be at least 1")]
    fn zero_slot_pager_is_rejected_at_construction() {
        // A 0-slot pager used to build fine and then die on the FIRST touch, inside `take_slot`,
        // with a victim-selection assertion several frames away from the caller — a message that
        // nothing about the actual mistake (a paging budget that couldn't buy a single slot). The
        // guard must fire in the constructor, at the mistake.
        let _ = Pager::new(0);
    }

    #[test]
    fn one_slot_pager_is_the_smallest_legal_one() {
        // The guard's boundary: 1 slot is a legitimate (if maximally thrashy) configuration and
        // must keep working — `slot_reuse_after_eviction_is_exact` relies on it.
        let mut p = Pager::new(1);
        assert_eq!(p.n_slots(), 1);
        assert_eq!(
            p.touch(1),
            Resolution::Miss {
                slot: 0,
                evicted: None
            }
        );
    }

    #[test]
    fn disabled_slot_is_never_reused_until_enabled() {
        let mut p = Pager::new(3);
        assert_eq!(
            p.touch(10),
            Resolution::Miss {
                slot: 0,
                evicted: None
            }
        );
        assert_eq!(
            p.touch(11),
            Resolution::Miss {
                slot: 1,
                evicted: None
            }
        );
        assert_eq!(p.disable_slot(0), Some(10));
        assert_eq!(p.enabled_slots(), 2);
        assert!(!p.slot_enabled(0));
        assert_eq!(p.slot_of(10), None);

        // Slot 2 is the only never-used enabled slot. Once full, eviction must recycle slot 1 or
        // 2 rather than handing the loaned slot 0 back to the pager.
        assert_eq!(
            p.touch(12),
            Resolution::Miss {
                slot: 2,
                evicted: None
            }
        );
        let replacement = p.touch(13);
        assert!(matches!(replacement, Resolution::Miss { slot: 1 | 2, .. }));

        p.enable_slot(0);
        assert_eq!(p.enabled_slots(), 3);
        assert!(p.slot_enabled(0));
        assert!(matches!(
            p.touch(14),
            Resolution::Miss {
                slot: 0,
                evicted: None
            }
        ));
    }

    #[test]
    fn disabling_a_free_slot_removes_it_from_the_free_list() {
        let mut p = Pager::new(2);
        assert_eq!(p.disable_slot(0), None);
        assert_eq!(
            p.touch(1),
            Resolution::Miss {
                slot: 1,
                evicted: None
            }
        );
        assert!(matches!(
            p.touch(2),
            Resolution::Miss {
                slot: 1,
                evicted: Some(1)
            }
        ));
    }

    #[test]
    fn classic_touches_keep_their_hot_set_against_cold_sweeps() {
        // Decode's recency set (classic touch → MRU end) must survive a prefill-style cold
        // sweep, which churns the COLD end only.
        let mut p = Pager::new(4);
        p.begin_batch();
        p.touch(1); // decode-hot
        p.touch(2); // decode-hot
        for id in 10..16u32 {
            p.begin_batch();
            p.touch_cold(id); // sweep bigger than the remaining capacity
        }
        assert!(
            p.slot_of(1).is_some(),
            "hot block 1 evicted by a cold sweep"
        );
        assert!(
            p.slot_of(2).is_some(),
            "hot block 2 evicted by a cold sweep"
        );
    }

    #[test]
    fn schedule_cyclic_sweep_is_belady_optimal() {
        // Dense layer streaming's access shape: N blocks visited 0..N in order, repeated every
        // forward pass, through C < N slots. Belady (evict the furthest-next-use block) yields
        // exactly C-1 hits per steady-state pass on this trace; `schedule` must match it —
        // plain LRU would yield ZERO (every block evicted right before its reuse).
        let (n_blocks, n_slots, passes) = (10u32, 4usize, 5u64);
        let mut p = Pager::new(n_slots);
        for _ in 0..passes {
            for id in 0..n_blocks {
                p.begin_batch(); // one layer's weights = one batch
                p.schedule(id);
            }
        }
        // Pass 1 is all misses; every later pass hits the stable prefix of n_slots-1 blocks.
        let want_hits = (passes - 1) * (n_slots as u64 - 1);
        assert_eq!(
            p.stats().hits,
            want_hits,
            "schedule policy fell short of the Belady hit count on a cyclic sweep"
        );
    }

    #[test]
    fn schedule_stable_prefix_keeps_its_slots_across_passes() {
        // The stable prefix must not only stay resident but keep the SAME slot assignment pass
        // over pass — the property that makes the streamed dispatch's per-pass offsets cheap to
        // reason about (only the churn slot's occupant changes).
        let mut p = Pager::new(3);
        let record = |p: &Pager| (0..3u32).map(|id| p.slot_of(id)).collect::<Vec<_>>();
        for id in 0..8u32 {
            p.begin_batch();
            p.schedule(id);
        }
        let first = record(&p);
        for _ in 0..3 {
            for id in 0..8u32 {
                p.begin_batch();
                p.schedule(id);
            }
            assert_eq!(
                record(&p),
                first,
                "stable-prefix slots moved between passes"
            );
        }
        // And the prefix really is resident (blocks 0..n_slots-1 minus the churn slot).
        assert!(p.slot_of(0).is_some());
        assert!(p.slot_of(1).is_some());
    }

    #[test]
    fn schedule_batch_siblings_are_eviction_safe() {
        // A layer whose weights span several blocks in ONE pool touches them as one batch; the
        // epoch guard must keep earlier siblings resident while later ones evict (same invariant
        // as touch_cold, restated for the schedule entry point).
        let mut p = Pager::new(3);
        p.begin_batch();
        for id in [100u32, 101, 102] {
            p.schedule(id);
        }
        p.begin_batch();
        for id in [7u32, 8, 9] {
            p.schedule(id);
        }
        for id in [7u32, 8, 9] {
            assert!(p.slot_of(id).is_some(), "batch sibling {id} was evicted");
        }
    }

    /// Eviction must drop the victim's `epoch` entry, so `epoch` stays bounded by `n_slots` (never
    /// growing an entry per distinct BlockId ever touched). Exercised via the test-only
    /// `epoch_len()` accessor.
    #[test]
    fn eviction_drops_the_victim_epoch_entry() {
        let mut p = Pager::new(2);
        p.touch(1); // slot 0
        p.touch(2); // slot 1, full
                    // 3 evicts 1 (LRU); 1's epoch entry must go with it.
        assert_eq!(
            p.touch(3),
            Resolution::Miss {
                slot: 0,
                evicted: Some(1)
            }
        );
        assert!(p.slot_of(1).is_none());
        // epoch tracks only the two currently-resident blocks (2 and 3), not the evicted 1.
        assert_eq!(p.epoch_len(), 2);
        assert!(p.epoch_len() <= p.n_slots());
        // Churn many more distinct ids through the 2-slot cache: epoch stays bounded, not linear
        // in the number of distinct ids ever seen.
        for id in 100..200u32 {
            p.touch(id);
        }
        assert!(
            p.epoch_len() <= p.n_slots(),
            "epoch grew past n_slots ({} > {})",
            p.epoch_len(),
            p.n_slots()
        );
    }

    /// The pin's whole purpose: a pinned block is not a victim even when it is the COLDEST thing
    /// in the cache. Without the guard, LRU order alone would hand slot 0 straight back.
    #[test]
    fn a_pinned_block_is_never_the_victim() {
        let mut p = Pager::new(2);
        let Some(Resolution::Miss { slot: s1, .. }) = p.resolve_and_pin(1, Insert::Mru) else {
            panic!("expected miss")
        };
        p.touch(2); // unpinned, and now the MRU end — but 1 is pinned, so 2 is the only victim
        let Some(Resolution::Miss { evicted, .. }) = p.resolve_and_pin(3, Insert::Mru) else {
            panic!("expected miss")
        };
        assert_eq!(evicted, Some(2), "the unpinned block must be the victim");
        assert_eq!(p.slot_of(1), Some(s1), "pinned block moved or was evicted");
    }

    /// Exhaustion is a `None`, not a panic and not a silent eviction — and it must leave the cache
    /// exactly as it was, so a caller that releases a pin and retries gets a working pager.
    #[test]
    fn every_slot_pinned_yields_none_and_changes_nothing() {
        let mut p = Pager::new(2);
        p.resolve_and_pin(1, Insert::Mru).expect("slot 1");
        p.resolve_and_pin(2, Insert::Mru).expect("slot 2");
        let before = p.stats();
        assert!(
            p.resolve_and_pin(3, Insert::Mru).is_none(),
            "a fully pinned cache cannot admit a third block"
        );
        assert_eq!(p.resident_count(), 2);
        assert!(p.slot_of(1).is_some() && p.slot_of(2).is_some());
        assert_eq!(
            p.slot_of(3),
            None,
            "the rejected block must not be resident"
        );
        let after = p.stats();
        assert_eq!(
            (after.hits, after.misses, after.evictions),
            (before.hits, before.misses, before.evictions),
            "a refused resolve must not move any counter"
        );

        // Releasing one pin makes exactly one slot available again.
        p.unpin(1);
        let Some(Resolution::Miss { evicted, .. }) = p.resolve_and_pin(3, Insert::Mru) else {
            panic!("expected the freed slot")
        };
        assert_eq!(evicted, Some(1));
    }

    /// A pin outlives the batch epoch — the reason it exists at all. The epoch guard expires the
    /// moment the next batch opens; the pin does not.
    #[test]
    fn a_pin_outlives_its_batch() {
        let mut p = Pager::new(2);
        p.begin_batch();
        p.resolve_and_pin(1, Insert::Cold).expect("resident");
        p.begin_batch(); // block 1's epoch protection is now stale; only its pin remains
        p.touch_cold(2);
        p.begin_batch(); // ...and now block 2's is stale too, so it is the only legal victim
        let Some(Resolution::Miss { evicted, .. }) = p.resolve_and_pin(3, Insert::Cold) else {
            panic!("expected miss")
        };
        assert_eq!(evicted, Some(2), "the pin must still protect block 1");
        assert!(p.slot_of(1).is_some());
    }

    /// Nesting: two pins on one block need two releases. A single `unpin` must NOT make it
    /// evictable while the second reader still holds it.
    #[test]
    fn pins_are_counted_not_boolean() {
        let mut p = Pager::new(1);
        p.resolve_and_pin(1, Insert::Mru).expect("resident");
        p.resolve_and_pin(1, Insert::Mru).expect("hit, second pin");
        assert_eq!(p.pinned_blocks(), 1);
        p.unpin(1);
        assert!(
            p.resolve_and_pin(2, Insert::Mru).is_none(),
            "one release of two must leave the block pinned"
        );
        p.unpin(1);
        assert!(p.resolve_and_pin(2, Insert::Mru).is_some());
    }

    /// `pin_if_resident` is a probe: a miss must not read, insert, evict or count a miss.
    #[test]
    fn pin_if_resident_never_admits_a_block() {
        let mut p = Pager::new(2);
        assert_eq!(p.pin_if_resident(9), None);
        assert_eq!(p.resident_count(), 0);
        assert_eq!(p.stats().misses, 0, "a probe miss is not a residency miss");

        let Some(Resolution::Miss { slot, .. }) = p.resolve_and_pin(9, Insert::Mru) else {
            panic!("expected miss")
        };
        assert_eq!(p.pin_if_resident(9), Some(slot));
        // Two pins now (resolve + probe): both must be released before 9 can be evicted.
        p.unpin(9);
        p.unpin(9);
        assert_eq!(p.pinned_blocks(), 0);
    }

    #[test]
    #[should_panic(expected = "holds no pin")]
    fn unpinning_an_unpinned_block_panics() {
        // An unbalanced release would let a slot be evicted under a live reader — wrong bytes with
        // no error, which is worse than the panic.
        let mut p = Pager::new(1);
        p.touch(1);
        p.unpin(1);
    }

    #[test]
    fn hit_rate_reports_sane_values() {
        let mut p = Pager::new(2);
        assert_eq!(p.stats().hit_rate(), 1.0); // vacuous
        p.touch(1);
        p.touch(1);
        assert!((p.stats().hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn exchange_slot_rotates_without_changing_lru_capacity() {
        let mut p = Pager::new(3);
        assert_eq!(p.disable_slot(2), None);
        p.touch(10);
        p.touch(11);
        p.begin_batch();
        let Resolution::Miss {
            slot,
            evicted: Some(10),
        } = p.touch(12)
        else {
            panic!("full active set must evict its LRU")
        };
        let old = p.rotate_resident_to_spare(12, 2);
        assert_eq!(old, slot);
        assert_eq!(p.slot_of(12), Some(2));
        assert!(!p.slot_enabled(old));
        assert!(p.slot_enabled(2));
        assert_eq!(p.enabled_slots(), 2);
        assert_eq!(p.lru_order(), vec![11, 12]);
    }
}
