//! Pure bookkeeping for the unified VRAM+RAM paged-KV frame space.
//!
//! Phase A of the paged hierarchical cache plan (`.trae/documents/
//! paged-hierarchical-kv-cache.md`): this module deliberately owns no buffers
//! and touches no execution path — the Vulkan wiring lands in Phase B/C, so
//! `dead_code` here is expected, not a mistake.
//!
//! A *frame* is one physical page of KV storage: one logical page x every
//! attention layer x every K/V/QSA plane, laid out contiguously so a migration
//! is a single `vkCmdCopyBuffer` region per page. Every frame is the same fixed
//! byte size, which is the whole point: a fixed-size allocator never hunts for
//! a hole, so VRAM external fragmentation cannot form. Frames live in two tiers
//! behind one handle type — `FrameId` bit 63 selects VRAM vs imported host RAM.
//! Migration is a *new id*, never an in-place tier flip: the caller DMAs the
//! bytes into a slot of the target tier, then rewrites its page->frame mapping
//! (the radix-tree chain, Phase B) and releases the source slot. The caller
//! must hold the GPU baton for the whole transfer — the PR #86 idle-worker
//! black-hole window (slot unpublished while bytes in flight) is the cautionary
//! tale this API refuses to encode: every mutation here is synchronous and
//! complete by return.

use std::collections::VecDeque;

/// Bit 63 of a packed [`FrameId`]: clear = VRAM tier, set = imported host-RAM
/// tier. RAM-tier page-table entries are never written into the kernel-visible
/// address tables (BDA stale-read + PCIe bandwidth-wall convictions); the bit
/// only names where the bytes physically sit for the host-side bookkeeper.
pub(crate) const RAM_TIER_BIT: u64 = 1 << 63;

/// Physical tier a frame's bytes live in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Tier {
    Vram,
    Ram,
}

/// Handle to one frame slot inside its tier's pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct FrameId(u64);

impl FrameId {
    pub(crate) fn new(tier: Tier, slot: usize) -> Self {
        let bit = if matches!(tier, Tier::Ram) {
            RAM_TIER_BIT
        } else {
            0
        };
        Self(bit | slot as u64)
    }

    pub(crate) fn tier(self) -> Tier {
        if self.0 & RAM_TIER_BIT != 0 {
            Tier::Ram
        } else {
            Tier::Vram
        }
    }

    pub(crate) fn slot(self) -> usize {
        (self.0 & !RAM_TIER_BIT) as usize
    }

    /// Wire form for later serialization (page chains in telemetry dumps).
    pub(crate) fn raw(self) -> u64 {
        self.0
    }

    pub(crate) fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

/// One frame slot's bookkeeping.
#[derive(Clone, Copy, Debug)]
struct SlotState {
    /// Live owners (page chains sharing the frame's bytes). Zero means the slot
    /// sits on its tier's free list; a handle into a freed slot is a logic bug.
    refcount: u32,
    /// Set while the running session's page table points at this frame. Active
    /// frames are never eviction candidates — "pages in inference stay pinned
    /// to VRAM" is enforced here, not by caller discipline alone.
    active: bool,
    /// Monotonic clock stamp of the last access; the LRU key for cold-leaf
    /// pruning (tip-first: the coldest leaf's tail page is picked first).
    lru: u64,
}

/// Fixed-capacity slot array + free list for one tier.
#[derive(Clone, Debug)]
struct TierState {
    slots: Vec<SlotState>,
    free: VecDeque<usize>,
    held: usize,
}

impl TierState {
    fn new(capacity: usize) -> Self {
        Self {
            slots: vec![
                SlotState {
                    refcount: 0,
                    active: false,
                    lru: 0
                };
                capacity
            ],
            free: (0..capacity).rev().collect(),
            held: 0,
        }
    }
}

/// The two-tier fixed-frame allocator shared by the paged KV cache.
#[derive(Clone, Debug)]
pub(crate) struct FramePool {
    frame_bytes: u64,
    clock: u64,
    tiers: [TierState; 2],
    /// Device base address of each tier's contiguous corridor (VRAM: the unified-arena
    /// `alloc_kv_frames` corridor; RAM: 0 until Phase C imports a host arena — RAM frames are
    /// never device-addressable anyway). `frame_addr` is pure arithmetic on this base, which is
    /// what lets a page-table entry be `base + slot * frame_bytes + plane_offset` without the
    /// pool ever owning a buffer.
    base: [u64; 2],
}

impl FramePool {
    pub(crate) fn new(frame_bytes: u64, vram_frames: usize, ram_frames: usize) -> Self {
        assert!(frame_bytes > 0, "frame_bytes must be positive");
        Self {
            frame_bytes,
            clock: 0,
            tiers: [TierState::new(vram_frames), TierState::new(ram_frames)],
            base: [0; 2],
        }
    }

    /// Attach a tier's corridor base once the backend has reserved it (Phase B: VRAM only).
    pub(crate) fn set_base(&mut self, tier: Tier, base: u64) {
        self.base[Self::idx(tier)] = base;
    }

    /// Device address of a frame's first byte. Debug-asserts the tier has a corridor: a RAM-tier
    /// frame must never reach a page table (the caller writes 0 for RAM-tier entries instead).
    pub(crate) fn frame_addr(&self, id: FrameId) -> u64 {
        let base = self.base[Self::idx(id.tier())];
        debug_assert!(
            base != 0 || matches!(id.tier(), Tier::Ram),
            "frame {id:?} used before its corridor was installed"
        );
        base + id.slot() as u64 * self.frame_bytes
    }

    fn idx(tier: Tier) -> usize {
        matches!(tier, Tier::Ram) as usize
    }

    pub(crate) fn frame_bytes(&self) -> u64 {
        self.frame_bytes
    }

    pub(crate) fn capacity_frames(&self, tier: Tier) -> usize {
        self.tiers[Self::idx(tier)].slots.len()
    }

    pub(crate) fn held_frames(&self, tier: Tier) -> usize {
        self.tiers[Self::idx(tier)].held
    }

    pub(crate) fn free_frames(&self, tier: Tier) -> usize {
        self.capacity_frames(tier) - self.held_frames(tier)
    }

    /// Bytes currently owned by live frames (the ledger's per-tier footing).
    pub(crate) fn held_bytes(&self, tier: Tier) -> u64 {
        self.held_frames(tier) as u64 * self.frame_bytes
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn slot(&self, id: FrameId) -> &SlotState {
        let s = &self.tiers[Self::idx(id.tier())].slots[id.slot()];
        debug_assert!(s.refcount > 0, "handle {id:?} references a freed slot");
        s
    }

    fn slot_mut(&mut self, id: FrameId) -> &mut SlotState {
        let s = &mut self.tiers[Self::idx(id.tier())].slots[id.slot()];
        debug_assert!(s.refcount > 0, "handle {id:?} references a freed slot");
        s
    }

    /// Reserve one frame owned by exactly one chain. `None` = tier exhausted;
    /// the caller trims (or grows the pool) before retrying.
    pub(crate) fn alloc(&mut self, tier: Tier) -> Option<FrameId> {
        let lru = self.tick();
        let t = &mut self.tiers[Self::idx(tier)];
        let slot = t.free.pop_back()?;
        t.slots[slot] = SlotState {
            refcount: 1,
            active: false,
            lru,
        };
        t.held += 1;
        Some(FrameId::new(tier, slot))
    }

    /// A second (and further) page chain now shares this frame's bytes — the
    /// zero-copy prefix-sharing primitive of Phase B.
    pub(crate) fn pin(&mut self, id: FrameId) {
        let lru = self.tick();
        let s = self.slot_mut(id);
        s.refcount += 1;
        s.lru = lru;
    }

    /// Record a fresh access without gaining an owner (LRU refresh on a hit).
    pub(crate) fn touch(&mut self, id: FrameId) {
        let lru = self.tick();
        self.slot_mut(id).lru = lru;
    }

    /// Drop one owner; returns `true` when the last owner left and the slot was
    /// returned to its tier's free list.
    pub(crate) fn unpin(&mut self, id: FrameId) -> bool {
        let t = &mut self.tiers[Self::idx(id.tier())];
        let s = &mut t.slots[id.slot()];
        debug_assert!(s.refcount > 0, "unpin below zero on {id:?}");
        s.refcount -= 1;
        if s.refcount == 0 {
            t.free.push_back(id.slot());
            t.held -= 1;
            true
        } else {
            false
        }
    }

    pub(crate) fn refcount(&self, id: FrameId) -> u32 {
        self.slot(id).refcount
    }

    pub(crate) fn is_active(&self, id: FrameId) -> bool {
        self.slot(id).active
    }

    /// Mark the frame as backing the running session's page table (pinned to
    /// its tier, excluded from [`cold_candidates`](Self::cold_candidates)).
    pub(crate) fn set_active(&mut self, id: FrameId, active: bool) {
        self.slot_mut(id).active = active;
    }

    /// Eviction candidates for a trim pass: owned, not active, oldest first,
    /// at most `want`. The tree feeds this list leaf-tip-first so a shared
    /// prefix survives while a cold branch's tail pages are pruned; the caller
    /// then `transfer`s each candidate to the other tier (DMA under the baton)
    /// or drops it when even RAM is full.
    pub(crate) fn cold_candidates(&self, tier: Tier, want: usize) -> Vec<FrameId> {
        let t = &self.tiers[Self::idx(tier)];
        let mut picked: Vec<(u64, usize)> = t
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.refcount > 0 && !s.active)
            .map(|(slot, s)| (s.lru, slot))
            .collect();
        picked.sort_unstable();
        picked.truncate(want);
        picked
            .into_iter()
            .map(|(_, slot)| FrameId::new(tier, slot))
            .collect()
    }

    /// Move a frame's bookkeeping to a fresh slot of `target` *after* the
    /// caller has DMA'd the bytes there (demote/promote). Owners and the cold
    /// LRU stamp carry over, so a demoted frame does not look brand-new to the
    /// next trim; the source slot returns to its tier's free list. `None` when
    /// the target tier is full — the source is left untouched, so the caller
    /// trims the target tier first. Active frames must not migrate (the
    /// in-VRAM pin invariant).
    pub(crate) fn transfer(&mut self, id: FrameId, target: Tier) -> Option<FrameId> {
        let source = id.tier();
        assert_ne!(source, target, "transfer between the same tier");
        let (owners, lru) = {
            let s = self.slot(id);
            debug_assert!(
                !s.active,
                "active frame {id:?} must stay pinned in {source:?}"
            );
            (s.refcount, s.lru)
        };
        let fresh = self.alloc(target)?;
        for _ in 1..owners {
            self.slot_mut(fresh).refcount += 1;
        }
        self.slot_mut(fresh).lru = lru;
        let t = &mut self.tiers[Self::idx(source)];
        t.slots[id.slot()].refcount = 0;
        t.slots[id.slot()].active = false;
        t.free.push_back(id.slot());
        t.held -= 1;
        Some(fresh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 MiB frames keep the byte-math assertions readable.
    const FB: u64 = 1 << 20;

    #[test]
    fn frame_id_packs_tier_in_bit_63() {
        let v = FrameId::new(Tier::Vram, 4095);
        let r = FrameId::new(Tier::Ram, 4095);
        assert_eq!(v.tier(), Tier::Vram);
        assert_eq!(r.tier(), Tier::Ram);
        assert_eq!(v.slot(), r.slot());
        assert!(v.raw() < RAM_TIER_BIT);
        assert_eq!(FrameId::from_raw(r.raw()), r);
        assert_eq!(FrameId::from_raw(v.raw()), v);
    }

    #[test]
    fn alloc_release_round_trip_reuses_free_slots() {
        let mut p = FramePool::new(FB, 2, 1);
        assert_eq!(p.free_frames(Tier::Vram), 2);
        let a = p.alloc(Tier::Vram).unwrap();
        let b = p.alloc(Tier::Vram).unwrap();
        assert_eq!(p.free_frames(Tier::Vram), 0);
        assert_eq!(p.held_bytes(Tier::Vram), 2 * FB);
        assert!(p.alloc(Tier::Vram).is_none(), "exhausted tier must refuse");
        assert!(p.unpin(a), "single owner frees on unpin");
        assert_eq!(p.free_frames(Tier::Vram), 1);
        let c = p.alloc(Tier::Vram).unwrap();
        assert_eq!(c, a, "freed slot is reused LIFO");
        assert!(p.unpin(b));
        assert_eq!(p.held_bytes(Tier::Vram), FB);
    }

    #[test]
    fn shared_frame_frees_only_after_last_owner() {
        let mut p = FramePool::new(FB, 4, 0);
        let a = p.alloc(Tier::Vram).unwrap();
        p.pin(a);
        assert_eq!(p.refcount(a), 2);
        assert!(!p.unpin(a));
        assert_eq!(p.held_frames(Tier::Vram), 1);
        assert!(p.unpin(a));
        assert_eq!(p.held_frames(Tier::Vram), 0);
    }

    #[test]
    fn cold_candidates_order_lru_and_skip_active_and_freed() {
        let mut p = FramePool::new(FB, 4, 0);
        let a = p.alloc(Tier::Vram).unwrap();
        let b = p.alloc(Tier::Vram).unwrap();
        let c = p.alloc(Tier::Vram).unwrap();
        let _d = p.alloc(Tier::Vram).unwrap();
        assert!(p.unpin(c));
        p.set_active(b, true);
        p.touch(a); // a is now warmer than b's original stamp, but b is pinned
                    // Candidates: a (touched last) and _d (untouched since alloc, older).
        let cold = p.cold_candidates(Tier::Vram, 8);
        assert_eq!(cold.len(), 2, "active + freed frames excluded");
        assert!(cold.contains(&a));
        p.set_active(b, false);
        assert_eq!(p.cold_candidates(Tier::Vram, 1)[0], b, "oldest first");
        assert_eq!(
            p.cold_candidates(Tier::Vram, 2).len(),
            2,
            "want caps output"
        );
    }

    #[test]
    fn transfer_moves_owners_and_recency_to_target_tier() {
        let mut p = FramePool::new(FB, 2, 2);
        let a = p.alloc(Tier::Vram).unwrap();
        p.pin(a); // two chains share the page
        let _b = p.alloc(Tier::Vram).unwrap(); // clock 2
        let r = p.transfer(a, Tier::Ram).unwrap();
        assert_eq!(r.tier(), Tier::Ram);
        assert_eq!(p.refcount(r), 2, "owners carry over");
        assert_eq!(p.held_frames(Tier::Vram), 1);
        assert_eq!(p.held_frames(Tier::Ram), 1);
        // The demoted frame keeps its cold stamp (clock 1), so a frame freshly
        // allocated in RAM after the transfer still sorts behind it.
        let c = p.alloc(Tier::Ram).unwrap();
        let cold = p.cold_candidates(Tier::Ram, 2);
        assert_eq!(cold[0], r);
        assert_eq!(cold[1], c);
    }

    #[test]
    fn transfer_without_target_capacity_leaves_source_intact() {
        let mut p = FramePool::new(FB, 1, 0);
        let a = p.alloc(Tier::Vram).unwrap();
        assert!(p.transfer(a, Tier::Ram).is_none());
        assert_eq!(p.refcount(a), 1);
        assert_eq!(p.held_frames(Tier::Vram), 1);
        assert_eq!(p.held_frames(Tier::Ram), 0);
    }

    #[test]
    fn frame_addr_is_slot_arithmetic_on_the_corridor_base() {
        let mut p = FramePool::new(FB, 4, 1);
        p.set_base(Tier::Vram, 0x1000_0000);
        let a = p.alloc(Tier::Vram).unwrap();
        let b = p.alloc(Tier::Vram).unwrap();
        assert_eq!(p.frame_addr(a), 0x1000_0000 + a.slot() as u64 * FB);
        assert_eq!(
            p.frame_addr(b) - p.frame_addr(a),
            FB,
            "frames tile the corridor"
        );
    }

    #[test]
    fn transfer_frees_source_slot_for_reuse() {
        let mut p = FramePool::new(FB, 1, 1);
        let a = p.alloc(Tier::Vram).unwrap();
        let r = p.transfer(a, Tier::Ram).unwrap();
        assert_eq!(p.free_frames(Tier::Vram), 1);
        let a2 = p.alloc(Tier::Vram).unwrap();
        assert_eq!(a2, a, "source slot returns to the free list");
        assert_eq!(p.held_frames(Tier::Ram), 1);
        assert!(p.unpin(r));
    }
}
