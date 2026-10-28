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
//! a hole, so VRAM external fragmentation cannot form. Frames live in three tiers
//! behind one handle type — `FrameId` bit 63 selects imported host RAM, bit 62 the
//! sparse disk backing file (Phase D); neither bit set = VRAM.
//! Migration is a *new id*, never an in-place tier flip: the caller DMAs the
//! bytes into a slot of the target tier, then rewrites its page->frame mapping
//! (the radix-tree chain, Phase B) and releases the source slot. The caller
//! must hold the GPU baton for the whole transfer — the PR #86 idle-worker
//! black-hole window (slot unpublished while bytes in flight) is the cautionary
//! tale this API refuses to encode: every mutation here is synchronous and
//! complete by return.

use infr_core::backend::KvFrameCorridor;
use std::collections::VecDeque;

/// Bit 63 of a packed [`FrameId`]: clear = VRAM tier, set = imported host-RAM
/// tier. RAM-tier page-table entries are never written into the kernel-visible
/// address tables (BDA stale-read + PCIe bandwidth-wall convictions); the bit
/// only names where the bytes physically sit for the host-side bookkeeper.
pub(crate) const RAM_TIER_BIT: u64 = 1 << 63;

/// Bit 62 of a packed [`FrameId`]: set = disk-tier frame (bytes live in the
/// sparse paged-KV backing file at `slot * frame_bytes`). Like the RAM tier, a
/// disk frame never enters a device page table; the bit only names the backing
/// store for the bookkeeper. Clear together with [`RAM_TIER_BIT`] = VRAM, so
/// Phase B/C encodings stay byte-identical and old telemetry parses unchanged.
pub(crate) const DISK_TIER_BIT: u64 = 1 << 62;

/// Physical tier a frame's bytes live in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Tier {
    Vram,
    Ram,
    Disk,
}

/// Handle to one frame slot inside its tier's pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct FrameId(u64);

impl FrameId {
    pub(crate) fn new(tier: Tier, slot: usize) -> Self {
        let bit = match tier {
            Tier::Vram => 0,
            Tier::Ram => RAM_TIER_BIT,
            Tier::Disk => DISK_TIER_BIT,
        };
        Self(bit | slot as u64)
    }

    pub(crate) fn tier(self) -> Tier {
        if self.0 & RAM_TIER_BIT != 0 {
            Tier::Ram
        } else if self.0 & DISK_TIER_BIT != 0 {
            Tier::Disk
        } else {
            Tier::Vram
        }
    }

    pub(crate) fn slot(self) -> usize {
        (self.0 & !(RAM_TIER_BIT | DISK_TIER_BIT)) as usize
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

/// One contiguous run of frames inside a tier's device corridor, plus the global slot index of
/// its first frame. `frame_addr` is arithmetic within a segment; the segment table is what makes
/// that arithmetic work when a tier is split across several device corridors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameSegment {
    base: u64,
    first_frame: usize,
    frames: usize,
}

/// The three-tier fixed-frame allocator shared by the paged KV cache.
#[derive(Clone, Debug)]
pub(crate) struct FramePool {
    frame_bytes: u64,
    clock: u64,
    tiers: [TierState; 3],
    /// Device corridors of each tier in ascending slot order (VRAM: the unified-arena
    /// `alloc_kv_frames` corridors; RAM: empty until Phase C imports a host arena — RAM frames
    /// are never device-addressable anyway). A tier may hold several segments: the unified arena
    /// is built from ~2 GiB ReBAR shard pieces, so a KV space larger than one piece's largest gap
    /// (ctx 262144 wants 4.31 GiB) is satisfied by tiling. Slot indices stay global 0..capacity —
    /// segments are pure addressing detail, so the radix tree, the LRU and the eviction logic
    /// never see them. A page-table entry is then `base + slot * frame_bytes + plane_offset`
    /// without the pool ever owning a buffer.
    segments: [Vec<FrameSegment>; 3],
}

impl FramePool {
    pub(crate) fn new(
        frame_bytes: u64,
        vram_frames: usize,
        ram_frames: usize,
        disk_frames: usize,
    ) -> Self {
        assert!(frame_bytes > 0, "frame_bytes must be positive");
        Self {
            frame_bytes,
            clock: 0,
            tiers: [
                TierState::new(vram_frames),
                TierState::new(ram_frames),
                TierState::new(disk_frames),
            ],
            segments: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    /// Install a tier's device corridors once the backend has reserved them (Phase B: VRAM only).
    /// The corridors must tile frames `0..capacity` exactly — a gap in the slot space would make
    /// `frame_addr` panic on a live handle.
    pub(crate) fn set_segments(&mut self, tier: Tier, corridors: &[KvFrameCorridor]) {
        let mut segments = Vec::with_capacity(corridors.len());
        let mut first_frame = 0usize;
        for corridor in corridors {
            assert!(corridor.frames > 0, "a KV frame corridor must hold frames");
            segments.push(FrameSegment {
                base: corridor.base,
                first_frame,
                frames: corridor.frames,
            });
            first_frame += corridor.frames;
        }
        assert_eq!(
            first_frame,
            self.capacity_frames(tier),
            "KV frame corridors must cover every {tier:?} slot"
        );
        self.segments[Self::idx(tier)] = segments;
    }

    /// Device address of a frame's first byte. Debug-asserts the tier has a corridor: a RAM-tier
    /// frame must never reach a page table (the caller writes 0 for RAM-tier entries instead).
    pub(crate) fn frame_addr(&self, id: FrameId) -> u64 {
        let tier = id.tier();
        let segments = &self.segments[Self::idx(tier)];
        let slot = id.slot();
        if segments.is_empty() {
            debug_assert!(
                !matches!(tier, Tier::Vram),
                "frame {id:?} used before its corridor was installed"
            );
            // Host-side tiers address their own backing stores: RAM slot = offset into
            // `ram_backing`, disk slot = offset into the sparse backing file.
            return slot as u64 * self.frame_bytes;
        }
        // Segments are sorted by `first_frame`, so the owning segment is the last one whose first
        // frame is not above this slot.
        let index = match segments.binary_search_by(|seg| seg.first_frame.cmp(&slot)) {
            Ok(index) => index,
            Err(0) => 0,
            Err(index) => index - 1,
        };
        let segment = &segments[index];
        debug_assert!(
            slot < segment.first_frame + segment.frames,
            "frame {id:?} falls outside its corridor segments"
        );
        segment.base + (slot - segment.first_frame) as u64 * self.frame_bytes
    }

    fn idx(tier: Tier) -> usize {
        match tier {
            Tier::Vram => 0,
            Tier::Ram => 1,
            Tier::Disk => 2,
        }
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

    /// Finish a migration whose bytes the caller already DMA'd into the PREALLOCATED `fresh`
    /// slot of the other tier (Phase C: the host-side bookkeeper needs the destination address —
    /// `frame_addr(fresh)` on the RAM tier is its offset into the host backing — BEFORE the bytes
    /// move, which `transfer`'s atomic alloc cannot offer). Owners and the cold LRU stamp carry
    /// over exactly as in [`Self::transfer`], and the source slot returns to its tier's free list.
    /// `fresh` must be a freshly allocated (single-owner, inactive) slot of the opposite tier.
    pub(crate) fn migrate_into(&mut self, id: FrameId, fresh: FrameId) {
        let source = id.tier();
        assert_ne!(source, fresh.tier(), "migrate_into between the same tier");
        let (owners, lru) = {
            let s = self.slot(id);
            debug_assert!(!s.active, "active frame {id:?} must stay pinned");
            (s.refcount, s.lru)
        };
        {
            let dest = self.slot_mut(fresh);
            debug_assert_eq!(
                dest.refcount, 1,
                "fresh slot {fresh:?} must be a fresh alloc"
            );
            debug_assert!(!dest.active, "fresh slot {fresh:?} must not be active");
            dest.refcount = owners;
            dest.lru = lru;
        }
        let t = &mut self.tiers[Self::idx(source)];
        t.slots[id.slot()].refcount = 0;
        t.slots[id.slot()].active = false;
        t.free.push_back(id.slot());
        t.held -= 1;
    }

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
        let mut p = FramePool::new(FB, 2, 1, 0);
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
        let mut p = FramePool::new(FB, 4, 0, 0);
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
        let mut p = FramePool::new(FB, 4, 0, 0);
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
        let mut p = FramePool::new(FB, 2, 2, 0);
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
        let mut p = FramePool::new(FB, 1, 0, 0);
        let a = p.alloc(Tier::Vram).unwrap();
        assert!(p.transfer(a, Tier::Ram).is_none());
        assert_eq!(p.refcount(a), 1);
        assert_eq!(p.held_frames(Tier::Vram), 1);
        assert_eq!(p.held_frames(Tier::Ram), 0);
    }

    #[test]
    fn frame_addr_is_slot_arithmetic_on_the_corridor_base() {
        let mut p = FramePool::new(FB, 4, 1, 0);
        p.set_segments(
            Tier::Vram,
            &[KvFrameCorridor {
                base: 0x1000_0000,
                frames: 4,
            }],
        );
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
    fn frame_addr_walks_several_corridors_by_global_slot() {
        // The unified arena is sharded, so a 5-frame tier arrives as 2 + 3 frames in two
        // corridors; global slot indices cross the segment boundary untouched.
        let mut p = FramePool::new(FB, 5, 0, 0);
        p.set_segments(
            Tier::Vram,
            &[
                KvFrameCorridor {
                    base: 0x2000_0000,
                    frames: 2,
                },
                KvFrameCorridor {
                    base: 0x9000_0000,
                    frames: 3,
                },
            ],
        );
        assert_eq!(p.frame_addr(FrameId::new(Tier::Vram, 0)), 0x2000_0000);
        assert_eq!(p.frame_addr(FrameId::new(Tier::Vram, 1)), 0x2000_0000 + FB);
        assert_eq!(
            p.frame_addr(FrameId::new(Tier::Vram, 2)),
            0x9000_0000,
            "slot 2 opens the second corridor"
        );
        assert_eq!(
            p.frame_addr(FrameId::new(Tier::Vram, 4)),
            0x9000_0000 + 2 * FB
        );
    }

    #[test]
    fn migrate_into_carries_owners_lru_and_frees_the_source() {
        let mut p = FramePool::new(FB, 2, 2, 0);
        p.set_segments(
            Tier::Vram,
            &[KvFrameCorridor {
                base: 0x1000_0000,
                frames: 2,
            }],
        );
        let a = p.alloc(Tier::Vram).unwrap(); // clock 1: cold stamp to carry over
        p.pin(a); // two entry chains share the frame
        let _b = p.alloc(Tier::Vram).unwrap(); // clock 2
        let r = p.alloc(Tier::Ram).unwrap();
        p.migrate_into(a, r);
        assert_eq!(p.refcount(r), 2, "owners carry over");
        assert_eq!(p.held_frames(Tier::Vram), 1);
        assert_eq!(p.held_frames(Tier::Ram), 1);
        assert_eq!(
            p.free_frames(Tier::Vram),
            1,
            "source slot is back on the list"
        );
        let a2 = p.alloc(Tier::Vram).unwrap();
        assert_eq!(a2, a, "the freed source slot is reusable");
        assert!(!p.is_active(r), "the migrated frame is not active");
        // The cold stamp carried over: a frame freshly allocated into RAM after the
        // migration sorts behind the demoted one.
        let c = p.alloc(Tier::Ram).unwrap();
        let cold = p.cold_candidates(Tier::Ram, 2);
        assert_eq!(cold[0], r, "demoted frame keeps its cold LRU");
        assert_eq!(cold[1], c);
        // The RAM frame's "address" is its offset into the host backing (no corridor).
        assert_eq!(p.frame_addr(r), r.slot() as u64 * FB);
        assert!(!p.unpin(r) && p.unpin(r), "both owners survive");
        let _ = p.unpin(c);
        let _ = p.unpin(a2);
    }

    #[test]
    fn transfer_frees_source_slot_for_reuse() {
        let mut p = FramePool::new(FB, 1, 1, 0);
        let a = p.alloc(Tier::Vram).unwrap();
        let r = p.transfer(a, Tier::Ram).unwrap();
        assert_eq!(p.free_frames(Tier::Vram), 1);
        let a2 = p.alloc(Tier::Vram).unwrap();
        assert_eq!(a2, a, "source slot returns to the free list");
        assert_eq!(p.held_frames(Tier::Ram), 1);
        assert!(p.unpin(r));
    }

    #[test]
    fn disk_tier_packs_bit_62_and_keeps_vram_ram_encodings() {
        let v = FrameId::new(Tier::Vram, 7);
        let r = FrameId::new(Tier::Ram, 7);
        let d = FrameId::new(Tier::Disk, 7);
        assert_eq!(d.tier(), Tier::Disk);
        assert_eq!(d.slot(), 7);
        // Phase B/C wire forms untouched: VRAM has neither bit, RAM only bit 63.
        assert_eq!(v.raw(), 7);
        assert_eq!(r.raw(), RAM_TIER_BIT | 7);
        assert_eq!(d.raw(), DISK_TIER_BIT | 7);
        assert_eq!(FrameId::from_raw(d.raw()), d);
    }

    #[test]
    fn migrate_ram_to_disk_and_back_carries_owners() {
        // Phase D round-trip through the pool ledger alone (bytes are the tree's job):
        // RAM -> Disk frees the RAM slot, Disk -> RAM frees the disk slot.
        let mut p = FramePool::new(FB, 0, 1, 1);
        let r = p.alloc(Tier::Ram).unwrap(); // clock 1: cold stamp to carry over
        p.pin(r); // two chains share the frame
        let d = p.alloc(Tier::Disk).unwrap();
        p.migrate_into(r, d);
        assert_eq!(p.refcount(d), 2);
        assert_eq!(p.held_frames(Tier::Ram), 0);
        assert_eq!(
            p.free_frames(Tier::Ram),
            1,
            "the RAM slot is claimable again"
        );
        assert_eq!(
            p.frame_addr(d),
            d.slot() as u64 * FB,
            "disk addr = file offset"
        );
        let r2 = p.alloc(Tier::Ram).unwrap();
        p.migrate_into(d, r2);
        assert_eq!(p.refcount(r2), 2);
        assert_eq!(p.held_frames(Tier::Disk), 0);
        assert_eq!(p.free_frames(Tier::Disk), 1);
        assert!(!p.unpin(r2) && p.unpin(r2), "both owners survived");
    }
}
