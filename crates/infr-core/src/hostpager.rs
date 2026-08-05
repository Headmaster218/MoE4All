//! The DRAM tier of the weight pager: a fixed-size host arena of uniform slots, filled from a
//! [`BlockIo`] on a miss and read IN PLACE while pinned (`docs/disk-streaming-plan.md` §3.3).
//!
//! Unlike the VRAM pagers, whose callers copy a slot's bytes out and then forget it, this tier's
//! callers dereference the slot itself — a CPU kernel reads a weight for a whole op, a staging copy
//! reads one until the copy is recorded. That is what the pins in [`crate::pager`] are for, and it
//! is why the arena is raw storage rather than a `Vec` behind a lock: readers of different slots
//! must not serialize against each other or against a fill.
//!
//! # Soundness
//! One `Mutex` guards ALL residency state (the [`Pager`], the per-block [`SlotState`]); the arena
//! bytes are outside it, reached only through raw pointers under these rules:
//!
//! - **A slot is written only by the thread that put its block into [`SlotState::Loading`]**, which
//!   happens under the lock, on a miss, for a block that thread has just pinned. A pinned block
//!   cannot be evicted, so no other thread can be handed the same slot meanwhile.
//! - **A slot is read only through a [`Pin`]**, which exists only for a block that is `Ready` and
//!   pinned. Any thread asking for a `Loading` block waits instead of reading it.
//! - **The arena is never reallocated**, and no reference to the whole buffer is ever formed —
//!   every access is a `from_raw_parts` over one slot's own range, so a reader of slot 3 and a
//!   filler of slot 7 never hold overlapping references.
//!
//! Together those give: for any byte, at most one writer and no concurrent reader.

use crate::blockio::{BlockDesc, BlockIo};
use crate::error::{Error, Result};
use crate::pager::{BlockId, Insert, Pager, PagerStats, Resolution};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Split an arena budget across uniform size classes, in slots per class.
///
/// `classes` is `(slot_bytes, n_blocks)` per class. Each gets a share of the budget proportional to
/// its share of the pageable bytes — byte share is access share, because a forward pass reads every
/// block exactly once — floored at one slot and capped at its block count, since slots past that
/// are unusable. Classes are seated largest-total-bytes first, so when the budget runs out it is
/// the classes that matter least that go unseated (`0` slots; the caller keeps those on whatever
/// path it had before). Returns one entry per input class, in the input's order.
///
/// Pure arithmetic, and shared by both consumers of the tier: the CPU backend sizes its pools per
/// weight-size class, and the Vulkan dense session sizes one host pool under each VRAM pool. Two
/// copies of this rule would be two budgets that drift.
pub fn plan_slots(budget_bytes: usize, classes: &[(usize, usize)]) -> Vec<usize> {
    let mut out = vec![0usize; classes.len()];
    let total: usize = classes.iter().map(|&(size, n)| size * n).sum();
    if total == 0 || budget_bytes == 0 {
        return out;
    }
    let mut order: Vec<usize> = (0..classes.len()).collect();
    // Largest total bytes first; size then index break ties, so the split is reproducible for a
    // given model and budget rather than depending on how the caller happened to enumerate.
    order.sort_unstable_by_key(|&i| {
        let (size, n) = classes[i];
        (std::cmp::Reverse(size * n), std::cmp::Reverse(size), i)
    });
    let mut left = budget_bytes;
    for i in order {
        let (slot_bytes, n_blocks) = classes[i];
        if slot_bytes == 0 || n_blocks == 0 {
            continue;
        }
        let share =
            (budget_bytes as u128 * (slot_bytes * n_blocks) as u128 / total as u128) as usize;
        let want = (share / slot_bytes).clamp(1, n_blocks);
        let n_slots = want.min(left / slot_bytes);
        if n_slots == 0 {
            continue; // cannot seat even one block of this class
        }
        left -= n_slots * slot_bytes;
        out[i] = n_slots;
    }
    out
}

/// Owned, never-resized slot storage, addressed through a raw pointer so that per-slot references
/// never alias (see the module doc's soundness rules). Zero-initialized, matching the calloc
/// contract every backend allocation in this workspace follows.
struct Arena {
    ptr: *mut u8,
    total: usize,
    slot_bytes: usize,
}

// SAFETY: `Arena` is a plain byte region. It carries no interior references and no thread-affine
// state; every access goes through the `unsafe` accessors below, whose contracts (upheld by
// `HostPager`'s locking) are what make concurrent use sound — not any property of the pointer
// itself. Sharing it across threads is therefore as safe as the accessors' callers make it.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    fn new(n_slots: usize, slot_bytes: usize) -> Self {
        let total = n_slots * slot_bytes;
        let buf = vec![0u8; total].into_boxed_slice();
        // Leak the box and keep only the pointer: holding both a `Box` and a raw pointer would
        // mean every access aliases the box's own reference, which is exactly the aliasing this
        // arena has to avoid. `Drop` reconstitutes it.
        let ptr = Box::into_raw(buf) as *mut u8;
        Self {
            ptr,
            total,
            slot_bytes,
        }
    }

    fn offset(&self, slot: u32) -> usize {
        slot as usize * self.slot_bytes
    }

    /// Address of `slot`'s first byte. Safe on its own — a pointer proves nothing; the exclusivity
    /// argument belongs at the site that forms a `&mut` from it, which is the one place that can
    /// make it (see [`HostPager::pin`]'s fill).
    fn slot_ptr(&self, slot: u32, len: usize) -> *mut u8 {
        debug_assert!(len <= self.slot_bytes);
        debug_assert!(self.offset(slot) + len <= self.total);
        // SAFETY: the offset is within the single allocation this arena owns, per the asserts.
        unsafe { self.ptr.add(self.offset(slot)) }
    }

    /// # Safety
    /// The caller must hold a pin on the block resident in `slot`, and that block must be `Ready`,
    /// so no writer can be active on this slot; `len <= slot_bytes`.
    unsafe fn slot_ref(&self, slot: u32, len: usize) -> &[u8] {
        debug_assert!(len <= self.slot_bytes);
        debug_assert!(self.offset(slot) + len <= self.total);
        std::slice::from_raw_parts(self.ptr.add(self.offset(slot)), len)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `Box::into_raw` of a `[u8]` of exactly `total` bytes, and no
        // `Pin` can outlive the `HostPager` that owns this arena (their lifetimes are tied).
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.ptr, self.total,
            )));
        }
    }
}

/// Whether a resident slot's bytes are usable yet. A block becomes `Loading` under the lock and
/// `Ready` once its fill returns; a failed fill removes it entirely, so a retry re-reads rather
/// than serving a half-filled slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Loading,
    Ready,
}

struct Inner {
    pager: Pager,
    state: HashMap<BlockId, SlotState>,
    descs: HashMap<BlockId, BlockDesc>,
    /// Blocks [`HostPager::fill`] has missed on at least once — the admission doorkeeper.
    ///
    /// A tier ABOVE this one keeps its own resident set, and it only calls down on ITS misses. On
    /// the first pass nothing is resident up there, so every block calls down and a
    /// first-miss-admits arena fills with the prefix the tier above is about to keep forever —
    /// blocks that then never call down again, holding slots that can never be hit. Measured on
    /// Qwen3-14B: 4 of 9 slots per pool dead, 44% of the arena.
    ///
    /// Requiring a SECOND miss fixes it with no knowledge of the tier above: a block that tier
    /// keeps resident never misses twice, so it is never admitted, and the arena fills with exactly
    /// the blocks that do keep coming back. Bounded by the pool's block count (one bit of interest
    /// per registered block), not by traffic.
    missed_once: HashSet<BlockId>,
}

/// What [`HostPager::fill`] did with one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    /// Resident: copied out of the arena, nothing read.
    Hit,
    /// Read into a free arena slot, then copied out. The block is now resident.
    Admitted,
    /// The arena was full, so the block was read straight into the caller's buffer and left
    /// unresident. One copy instead of two — see [`HostPager::fill`] for why that is the right
    /// trade on a sweep.
    Streamed,
}

/// Cumulative tier activity. The residency half comes from the [`Pager`]; the I/O half is what
/// says whether a good hit rate was earned or was never tested.
#[derive(Debug, Clone, Copy)]
pub struct HostPagerStats {
    pub pager: PagerStats,
    /// Blocks actually read from the tier below.
    pub reads: u64,
    pub bytes_read: u64,
    /// Of those reads, how many went STRAIGHT to the caller because the arena was full
    /// ([`Fill::Streamed`]). `reads - streamed` is what the arena absorbed.
    pub streamed: u64,
}

/// A fixed-budget host cache of uniform `slot_bytes` blocks, read in place.
pub struct HostPager {
    inner: Mutex<Inner>,
    /// Signalled whenever a block leaves [`SlotState::Loading`] — waiters for a block another
    /// thread is filling park here rather than reading its half-written slot.
    ready: Condvar,
    arena: Arena,
    io: Arc<dyn BlockIo>,
    slot_bytes: usize,
    /// How many blocks may be resident. Equal to the arena's slot count, EXCEPT in the arena-less
    /// mode built by [`HostPager::stream_only`], where it is zero and every fill streams.
    max_resident: usize,
    reads: AtomicU64,
    bytes_read: AtomicU64,
    streamed: AtomicU64,
}

impl HostPager {
    /// `n_slots` slots of `slot_bytes` each — the tier's whole budget, allocated up front so a
    /// budget that does not fit fails here rather than part-way through a generation.
    ///
    /// `n_slots` must be at least the number of blocks one caller pins simultaneously, times the
    /// number of concurrent callers (`infr serve`'s `--parallel`). Below that floor [`Self::pin`]
    /// returns the exhaustion error rather than evicting a block someone is reading.
    pub fn new(n_slots: usize, slot_bytes: usize, io: Arc<dyn BlockIo>) -> Result<Self> {
        if n_slots == 0 || slot_bytes == 0 {
            return Err(Error::backend(format!(
                "host pager: a {n_slots}-slot x {slot_bytes}-byte arena holds nothing"
            )));
        }
        Ok(Self {
            inner: Mutex::new(Inner {
                pager: Pager::new(n_slots),
                state: HashMap::new(),
                descs: HashMap::new(),
                missed_once: HashSet::new(),
            }),
            ready: Condvar::new(),
            arena: Arena::new(n_slots, slot_bytes),
            io,
            slot_bytes,
            max_resident: n_slots,
            reads: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            streamed: AtomicU64::new(0),
        })
    }

    /// A tier with NO arena: every [`Self::fill`] reads its block straight into the caller's
    /// buffer, and nothing is ever cached here.
    ///
    /// # Why an arena-less tier exists
    /// On a UNIFIED-memory device the arena ABOVE this one already lives in host RAM and is
    /// GPU-accessible. A host cache beneath it would be a second copy of the same bytes in the same
    /// RAM, readable only by the CPU — strictly worse than making the arena above bigger. But the
    /// tier below still has a job: serving that arena's misses by BLOCK-GRANULAR positioned reads
    /// (with [`crate::blockio`]'s concurrent reader) instead of through the GGUF mapping, whose
    /// page cache evicts by recency and so thrashes on the cyclic sweep a forward pass performs —
    /// the pathology this whole feature exists to fix (`docs/perf/results.md`).
    ///
    /// So on unified memory the ladder is `DISK → GPU-accessible RAM` with no host cache in
    /// between, and this is the bottom of it.
    ///
    /// `slot_bytes` still bounds what one block may be, because the caller's destination (a staging
    /// ring region) is sized from it. [`Self::pin`] and [`Self::try_pin`] are refused: they hand out
    /// a borrow of arena bytes, and there are none.
    pub fn stream_only(slot_bytes: usize, io: Arc<dyn BlockIo>) -> Result<Self> {
        if slot_bytes == 0 {
            return Err(Error::backend(
                "host pager: a stream-only tier still needs a non-zero block stride".to_string(),
            ));
        }
        Ok(Self {
            inner: Mutex::new(Inner {
                // One nominal slot so the bookkeeping type stays uniform; `max_resident == 0`
                // is what actually prevents admission, and nothing ever reaches this pager's
                // slot-handing paths (`fill` short-circuits, `pin` is refused).
                pager: Pager::new(1),
                state: HashMap::new(),
                descs: HashMap::new(),
                missed_once: HashSet::new(),
            }),
            ready: Condvar::new(),
            arena: Arena::new(0, slot_bytes),
            io,
            slot_bytes,
            max_resident: 0,
            reads: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            streamed: AtomicU64::new(0),
        })
    }

    /// Whether this tier caches anything, or only reads through ([`Self::stream_only`]).
    pub fn caches(&self) -> bool {
        self.max_resident > 0
    }

    /// Declare where one block's bytes live. Called once per block at load; a block must be
    /// registered before it can be pinned.
    pub fn register(&self, desc: BlockDesc) -> Result<()> {
        let n = desc.nbytes();
        if n > self.slot_bytes {
            return Err(Error::backend(format!(
                "host pager: block {} is {n} bytes, slot stride is {}",
                desc.id, self.slot_bytes
            )));
        }
        self.inner.lock().unwrap().descs.insert(desc.id, desc);
        Ok(())
    }

    /// How many bytes `id` occupies, or `None` if it was never registered. A tier above sizes its
    /// own slot against this rather than re-deriving the group's byte total from the model.
    pub fn block_bytes(&self, id: BlockId) -> Option<usize> {
        self.inner
            .lock()
            .unwrap()
            .descs
            .get(&id)
            .map(|d| d.nbytes())
    }

    /// Open a new touch batch — same meaning as [`Pager::begin_batch`].
    pub fn begin_batch(&self) {
        self.inner.lock().unwrap().pager.begin_batch();
    }

    pub fn stats(&self) -> HostPagerStats {
        HostPagerStats {
            pager: self.inner.lock().unwrap().pager.stats(),
            reads: self.reads.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            streamed: self.streamed.load(Ordering::Relaxed),
        }
    }

    /// Bytes this tier's arena occupies.
    pub fn arena_bytes(&self) -> usize {
        self.arena.total
    }

    /// One slot's byte stride — every block in this pager is at most this large.
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Blocks this tier may hold resident — `0` for an arena-less [`Self::stream_only`] tier, whose
    /// bookkeeping still carries one nominal slot it never uses.
    pub fn n_slots(&self) -> usize {
        self.max_resident
    }

    /// Pin `id`'s bytes, reading them from the tier below if they are not resident.
    ///
    /// Blocks on I/O when it misses, and blocks while another thread fills the same block — but
    /// never waits for a pin to be released: a caller that finds every slot pinned gets the
    /// exhaustion error instead, because waiting on an unordered set of pins acquired one at a time
    /// is a deadlock, not a slow path.
    pub fn pin(&self, id: BlockId, insert: Insert) -> Result<Pin<'_>> {
        if !self.caches() {
            // A `Pin` borrows arena bytes and there are none. Refuse rather than hand back a
            // zero-length view of an empty arena, which would decode as silent garbage.
            return Err(Error::backend(format!(
                "host pager: block {id} was pinned on an arena-less (stream-only) tier — this \
                 tier serves `fill` into a caller's buffer and has nothing to borrow"
            )));
        }
        let (slot, desc) = loop {
            let mut inner = self.inner.lock().unwrap();
            if !inner.descs.contains_key(&id) {
                return Err(Error::backend(format!(
                    "host pager: block {id} was never registered"
                )));
            }
            // Resident and readable? Take the pin and go.
            if inner.state.get(&id) == Some(&SlotState::Ready) {
                if let Some(slot) = inner.pager.pin_if_resident(id) {
                    return Ok(self.pinned(id, slot, &inner));
                }
            }
            // Being filled by someone else: wait for them rather than read a half-written slot.
            if inner.state.get(&id) == Some(&SlotState::Loading) {
                let _unused = self.ready.wait(inner).unwrap();
                continue;
            }
            match inner.pager.resolve_and_pin(id, insert) {
                Some(Resolution::Hit { slot }) => {
                    // Resident with no state entry cannot happen: state and residency are set
                    // together under this lock. Treat it as a fill to re-establish the invariant
                    // rather than handing out bytes nothing wrote.
                    debug_assert!(false, "resident block {id} had no slot state");
                    inner.state.insert(id, SlotState::Loading);
                    break (slot, inner.descs[&id].clone());
                }
                Some(Resolution::Miss { slot, evicted }) => {
                    if let Some(e) = evicted {
                        inner.state.remove(&e);
                    }
                    inner.state.insert(id, SlotState::Loading);
                    break (slot, inner.descs[&id].clone());
                }
                None => {
                    return Err(Error::backend(format!(
                        "host pager: every slot of the {}-slot host cache is pinned, so block {id} \
                         cannot be admitted. Raise the host paging budget (paging.dram) — it must \
                         hold at least one working set per concurrent request.",
                        inner.pager.n_slots()
                    )))
                }
            }
        };

        // Fill outside the lock: this is disk I/O, and holding the residency lock across it would
        // stall every other block's hit. The claim on `slot` is exclusive per the module doc (the
        // block is `Loading` and pinned by this thread), which is what makes the write sound.
        let len = desc.nbytes();
        // SAFETY: this thread set `id` to `Loading` under the lock, after the pager assigned it
        // `slot` and pinned it. A pinned block is never an eviction victim, so no other thread can
        // be handed this slot; a thread wanting this same block sees `Loading` and waits instead of
        // reading. That makes this `&mut` the only reference to these bytes for its whole life,
        // which ends before the state flips to `Ready` below. `len <= slot_bytes` per `register`.
        let dst = unsafe { std::slice::from_raw_parts_mut(self.arena.slot_ptr(slot, len), len) };
        let read = self.io.read_block(&desc, dst);
        let mut inner = self.inner.lock().unwrap();
        match read {
            Ok(()) => {
                inner.state.insert(id, SlotState::Ready);
                self.reads.fetch_add(1, Ordering::Relaxed);
                self.bytes_read.fetch_add(len as u64, Ordering::Relaxed);
                let pin = self.pinned(id, slot, &inner);
                drop(inner);
                self.ready.notify_all();
                Ok(pin)
            }
            Err(e) => {
                // Drop the failed block entirely: leaving it resident would serve a partly-written
                // slot to the next caller, and leaving it `Loading` would park every waiter for
                // good. The pin taken by `resolve_and_pin` goes with it.
                inner.state.remove(&id);
                inner.pager.unpin(id);
                inner.pager.evict(id);
                drop(inner);
                self.ready.notify_all();
                Err(e)
            }
        }
    }

    /// Deliver `id`'s bytes into `dst`, admitting the block only while the arena has room.
    ///
    /// The PARTITION shape of the tier (`docs/disk-streaming-plan.md` §3.6), for a caller that
    /// copies the bytes straight out — a GPU staging ring — rather than reading the slot in place:
    ///
    /// - resident: copied out of the arena, nothing read;
    /// - not resident, a slot free AND this block has missed before: read into the arena, then
    ///   copied out. The arena fills once;
    /// - otherwise: read STRAIGHT into `dst`, residency untouched.
    ///
    /// The "has missed before" condition is the admission doorkeeper ([`Inner::missed_once`]) and it
    /// is what keeps the arena from filling with the tier above's permanently-resident prefix.
    ///
    /// That last case is why this exists. Admitting by eviction would spend the copy AND evict a
    /// block whose next use is sooner: under a cyclic sweep the block that just missed is the one
    /// whose next use is furthest away, so a full arena is already holding the right set and the
    /// rest should stream through with ONE copy instead of two. A cache that keeps churning is the
    /// [`Self::pin`] shape, and that is the right one for MoE, where routing is skewed and
    /// unpredictable — not for a sweep.
    ///
    /// `dst` must be at least the block's length; only that prefix is written.
    ///
    /// There is no insertion-policy argument because nothing would read it: the LRU order exists to
    /// choose a victim, and this never has one. A caller that mixed this with [`Self::pin`] on one
    /// pager would make the order matter again — nothing does, and the cold-end insert below is the
    /// conservative choice if one ever did.
    pub fn fill(&self, id: BlockId, dst: &mut [u8]) -> Result<Fill> {
        // `(admitted slot, descriptor)` — the slot is `None` when this call will stream.
        let (slot, desc) = loop {
            let mut inner = self.inner.lock().unwrap();
            let Some(desc) = inner.descs.get(&id).cloned() else {
                return Err(Error::backend(format!(
                    "host pager: block {id} was never registered"
                )));
            };
            if inner.state.get(&id) == Some(&SlotState::Ready) {
                if let Some(slot) = inner.pager.pin_if_resident(id) {
                    let pin = self.pinned(id, slot, &inner);
                    drop(inner); // copy out of the arena without holding every other block's lock
                    dst[..pin.len()].copy_from_slice(&pin);
                    return Ok(Fill::Hit);
                }
            }
            if inner.state.get(&id) == Some(&SlotState::Loading) {
                let _unused = self.ready.wait(inner).unwrap();
                continue;
            }
            // Room to admit, and has this block earned admission? Only a FREE slot counts —
            // `Pager::take_slot_opt` drains the free list before it evicts, so this is exactly the
            // "admits without evicting" test — and only a block that has missed BEFORE is admitted,
            // so the tier above's permanently-resident prefix never takes a slot (see
            // `Inner::missed_once`).
            if inner.pager.resident_count() < self.max_resident && !inner.missed_once.insert(id) {
                match inner.pager.resolve_and_pin(id, Insert::Cold) {
                    Some(Resolution::Miss { slot, evicted }) => {
                        debug_assert!(evicted.is_none(), "a free slot cannot have evicted");
                        inner.state.insert(id, SlotState::Loading);
                        break (Some(slot), desc);
                    }
                    // Resident without a state entry, or every slot pinned. Neither is reachable
                    // here (the `Ready` arm above covers the first, and this path pins only across
                    // its own fill), and both are correctly served by streaming the block.
                    _ => break (None, desc),
                }
            }
            break (None, desc);
        };

        let Some(slot) = slot else {
            // Streamed: no arena involvement at all, so no lock and no residency change.
            let n = desc.nbytes();
            self.io.read_block(&desc, &mut dst[..n])?;
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
            self.streamed.fetch_add(1, Ordering::Relaxed);
            return Ok(Fill::Streamed);
        };

        // Admitting: fill the slot outside the lock, exactly as `pin` does and under the same
        // exclusivity argument (this thread set `Loading` and holds the pin).
        let n = desc.nbytes();
        // SAFETY: see `pin`'s fill — identical claim, identical proof.
        let arena = unsafe { std::slice::from_raw_parts_mut(self.arena.slot_ptr(slot, n), n) };
        let read = self.io.read_block(&desc, arena);
        let mut inner = self.inner.lock().unwrap();
        match read {
            Ok(()) => {
                inner.state.insert(id, SlotState::Ready);
                self.reads.fetch_add(1, Ordering::Relaxed);
                self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
                // The guard adopts the pin `resolve_and_pin` took and releases it on drop, so the
                // copy below runs with the slot un-evictable and nothing is released twice.
                let pin = self.pinned(id, slot, &inner);
                drop(inner);
                dst[..n].copy_from_slice(&pin);
                drop(pin);
                self.ready.notify_all();
                Ok(Fill::Admitted)
            }
            Err(e) => {
                inner.state.remove(&id);
                inner.pager.unpin(id);
                inner.pager.evict(id);
                drop(inner);
                self.ready.notify_all();
                Err(e)
            }
        }
    }

    /// Pin `id` only if it is already resident and readable — never reads from the tier below.
    ///
    /// Two callers, one shape: a reader re-borrowing a block it already pinned (the CPU op body,
    /// after its pre-step), and a tier above probing before going one tier down. Neither is a
    /// residency DECISION, so this moves no counter — see [`Pager::repin`]. A caller that wants the
    /// probe counted keeps its own tally.
    pub fn try_pin(&self, id: BlockId) -> Option<Pin<'_>> {
        if !self.caches() {
            return None; // nothing is ever resident on an arena-less tier
        }
        let mut inner = self.inner.lock().unwrap();
        if inner.state.get(&id) != Some(&SlotState::Ready) {
            return None;
        }
        let slot = inner.pager.repin(id)?;
        Some(self.pinned(id, slot, &inner))
    }

    /// Build the `Pin` for an already-pinned, `Ready` block. Takes the guard by reference to make
    /// the caller prove it holds the lock while reading the descriptor's length.
    fn pinned(&self, id: BlockId, slot: u32, inner: &Inner) -> Pin<'_> {
        let len = inner.descs[&id].nbytes();
        // SAFETY: `id` is `Ready` (fully written) and pinned by this caller, so it cannot be
        // evicted and no writer is active on `slot`; `len <= slot_bytes` per `register`.
        let bytes = unsafe { self.arena.slot_ref(slot, len) };
        Pin {
            pager: self,
            id,
            bytes,
        }
    }

    fn unpin(&self, id: BlockId) {
        self.inner.lock().unwrap().pager.unpin(id);
    }
}

/// A borrowed, un-evictable view of one block's bytes. Dropping it releases the pin.
///
/// `Debug` prints the block id and length only — a slot holds megabytes of weights, and a `Debug`
/// that dumps them turns one `expect_err` in a test into an unreadable wall.
pub struct Pin<'a> {
    pager: &'a HostPager,
    id: BlockId,
    bytes: &'a [u8],
}

impl std::ops::Deref for Pin<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.bytes
    }
}

impl std::fmt::Debug for Pin<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pin")
            .field("block", &self.id)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl Drop for Pin<'_> {
    fn drop(&mut self) {
        self.pager.unpin(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockio::BlockExtent;
    use std::sync::atomic::AtomicUsize;

    /// `plan_slots` returns one entry per input class, POSITIONALLY. Both callers index the result
    /// against their own class list — the Vulkan session pairs entry `i` with VRAM pool `i` — so a
    /// result that were sorted, filtered or compacted would attach each pool to another pool's host
    /// arena: the wrong slot stride, the wrong block set, and no error anywhere.
    #[test]
    fn plan_slots_answers_in_the_order_it_was_asked() {
        // Deliberately not in seating order: the middle class dominates the bytes.
        let classes = [(1 << 20, 2), (8 << 20, 16), (4 << 20, 1)];
        let slots = plan_slots(256 << 20, &classes);
        assert_eq!(slots.len(), classes.len());
        for (i, (&n, &(slot_bytes, n_blocks))) in slots.iter().zip(&classes).enumerate() {
            assert!(
                n <= n_blocks,
                "class {i} got {n} slots for {n_blocks} blocks of {slot_bytes}B"
            );
        }
        // The dominant class is the one that got the slots, and it is still at index 1.
        assert!(
            slots[1] > slots[0] && slots[1] > slots[2],
            "the dominant class did not get the largest share: {slots:?}"
        );
    }

    /// Seating is decided by each class's total bytes, not by where the caller happened to list it:
    /// permuting the input permutes the answer and changes nothing else. Without this, the split a
    /// model gets would depend on tensor enumeration order.
    #[test]
    fn plan_slots_is_independent_of_the_input_order() {
        let classes = [(1 << 20, 3), (8 << 20, 9), (2 << 20, 5)];
        let base = plan_slots(48 << 20, &classes);
        let permuted: Vec<(usize, usize)> = vec![classes[2], classes[0], classes[1]];
        let got = plan_slots(48 << 20, &permuted);
        assert_eq!(
            got,
            vec![base[2], base[0], base[1]],
            "a reordered input changed the split: {base:?} vs {got:?}"
        );
    }

    /// A budget that cannot seat one block of any class buys nothing — the caller keeps the path it
    /// had, rather than being handed a pool it cannot use.
    #[test]
    fn plan_slots_seats_nothing_it_cannot_afford() {
        assert_eq!(plan_slots(1 << 10, &[(1 << 20, 4)]), vec![0]);
        assert_eq!(plan_slots(0, &[(1 << 20, 4)]), vec![0]);
        assert!(plan_slots(1 << 30, &[]).is_empty());
    }

    /// A `BlockIo` with no file behind it: block `id` is `nbytes` copies of `id as u8`, so a slot
    /// filled from the wrong descriptor, or read at the wrong offset, is a value mismatch.
    struct FakeIo {
        reads: AtomicUsize,
        fail_on: Option<BlockId>,
        delay: Option<std::time::Duration>,
    }

    impl FakeIo {
        fn new() -> Self {
            Self {
                reads: AtomicUsize::new(0),
                fail_on: None,
                delay: None,
            }
        }
    }

    impl BlockIo for FakeIo {
        fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if let Some(d) = self.delay {
                std::thread::sleep(d);
            }
            if self.fail_on == Some(desc.id) {
                return Err(Error::backend(format!("injected failure on {}", desc.id)));
            }
            let n = desc.nbytes();
            dst[..n].fill(desc.id as u8);
            Ok(())
        }
    }

    fn desc(id: BlockId, len: usize) -> BlockDesc {
        BlockDesc {
            id,
            extents: vec![BlockExtent {
                offset: id as u64 * len as u64,
                len,
            }],
        }
    }

    fn pager_with(n_slots: usize, len: usize, io: Arc<FakeIo>, ids: &[BlockId]) -> HostPager {
        let p = HostPager::new(n_slots, len, io).expect("host pager");
        for &id in ids {
            p.register(desc(id, len)).expect("register");
        }
        p
    }

    /// The property every other test rests on: a pin's bytes are the block's OWN bytes, before and
    /// after the slot has been recycled for something else.
    #[test]
    fn a_pin_reads_the_blocks_own_bytes_across_eviction() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 64, io.clone(), &[1, 2, 3]);
        for id in [1u32, 2, 3, 1, 2, 3] {
            let pin = p.pin(id, Insert::Mru).expect("pin");
            assert_eq!(
                &pin[..],
                &vec![id as u8; 64][..],
                "block {id} read wrong bytes"
            );
        }
        // 2 slots, 3 blocks, cyclic: every access after the first pass is a miss, and each one
        // must have re-read rather than served the evicted block's stale slot.
        assert!(io.reads.load(Ordering::SeqCst) >= 6);
    }

    /// A hit must not touch the tier below at all — the entire point of the tier.
    #[test]
    fn a_hit_does_not_read() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(4, 32, io.clone(), &[7]);
        drop(p.pin(7, Insert::Mru).expect("pin"));
        assert_eq!(io.reads.load(Ordering::SeqCst), 1);
        drop(p.pin(7, Insert::Mru).expect("pin"));
        drop(p.pin(7, Insert::Mru).expect("pin"));
        assert_eq!(
            io.reads.load(Ordering::SeqCst),
            1,
            "a hit re-read the block"
        );
        let s = p.stats();
        assert_eq!((s.pager.hits, s.pager.misses), (2, 1));
        assert_eq!((s.reads, s.bytes_read), (1, 32));
    }

    /// A held pin survives a sweep that would otherwise evict it, and still reads its own bytes
    /// afterwards — the guarantee a CPU kernel holding a weight for a whole op depends on.
    #[test]
    fn a_held_pin_survives_a_sweep_over_the_whole_cache() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(3, 16, io, &[1, 2, 3, 4, 5, 6]);
        let held = p.pin(1, Insert::Cold).expect("pin");
        for id in [2u32, 3, 4, 5, 6] {
            drop(p.pin(id, Insert::Cold).expect("pin"));
        }
        assert_eq!(&held[..], &[1u8; 16][..], "the pinned slot was overwritten");
    }

    /// Exhaustion is an error naming the knob, not an eviction of someone's live bytes.
    #[test]
    fn all_slots_pinned_is_a_named_error() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 16, io, &[1, 2, 3]);
        let _a = p.pin(1, Insert::Mru).expect("pin");
        let _b = p.pin(2, Insert::Mru).expect("pin");
        let err = p.pin(3, Insert::Mru).expect_err("must refuse");
        assert!(err.to_string().contains("paging.dram"), "unexpected: {err}");
        // Releasing one pin makes the same call succeed.
        drop(_a);
        assert_eq!(&p.pin(3, Insert::Mru).expect("pin")[..], &[3u8; 16][..]);
    }

    /// A failed read must propagate AND leave nothing behind: the next attempt re-reads instead of
    /// serving the half-written slot, and the pin taken for the fill is released.
    #[test]
    fn a_failed_read_leaves_no_resident_block() {
        let io = Arc::new(FakeIo {
            reads: AtomicUsize::new(0),
            fail_on: Some(2),
            delay: None,
        });
        let p = pager_with(2, 16, io.clone(), &[1, 2]);
        let err = p.pin(2, Insert::Mru).expect_err("injected failure");
        assert!(err.to_string().contains("injected failure"));
        assert_eq!(p.stats().pager.hits, 0);
        // The slot is free again: a different block can take it, and retrying block 2 re-reads.
        assert_eq!(&p.pin(1, Insert::Mru).expect("pin")[..], &[1u8; 16][..]);
        let before = io.reads.load(Ordering::SeqCst);
        assert!(p.pin(2, Insert::Mru).is_err());
        assert_eq!(
            io.reads.load(Ordering::SeqCst),
            before + 1,
            "a failed block must be re-read, not served from its slot"
        );
    }

    /// `try_pin` never reads and never admits.
    #[test]
    fn try_pin_is_hit_only() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 16, io.clone(), &[1]);
        assert!(p.try_pin(1).is_none());
        assert_eq!(io.reads.load(Ordering::SeqCst), 0, "try_pin read the block");
        drop(p.pin(1, Insert::Mru).expect("pin"));
        assert_eq!(&p.try_pin(1).expect("now resident")[..], &[1u8; 16][..]);
    }

    /// Concurrent readers of the SAME block: exactly one fill happens, the other threads wait for
    /// it rather than reading a half-written slot, and every one of them sees complete bytes.
    #[test]
    fn concurrent_pins_of_one_block_fill_once() {
        let io = Arc::new(FakeIo {
            reads: AtomicUsize::new(0),
            fail_on: None,
            delay: Some(std::time::Duration::from_millis(20)),
        });
        let p = Arc::new(pager_with(4, 4096, io.clone(), &[5]));
        std::thread::scope(|s| {
            for _ in 0..8 {
                let p = Arc::clone(&p);
                s.spawn(move || {
                    let pin = p.pin(5, Insert::Mru).expect("pin");
                    assert_eq!(&pin[..], &vec![5u8; 4096][..], "torn or unfilled slot");
                });
            }
        });
        assert_eq!(
            io.reads.load(Ordering::SeqCst),
            1,
            "the block was filled more than once"
        );
    }

    /// Concurrent readers of DIFFERENT blocks must not serialize behind one another's I/O and must
    /// each get their own bytes — the case the per-slot pointer access exists for.
    #[test]
    fn concurrent_pins_of_distinct_blocks_are_independent() {
        let io = Arc::new(FakeIo {
            reads: AtomicUsize::new(0),
            fail_on: None,
            delay: Some(std::time::Duration::from_millis(5)),
        });
        let ids: Vec<BlockId> = (0..8).collect();
        let p = Arc::new(pager_with(8, 512, io, &ids));
        std::thread::scope(|s| {
            for id in ids {
                let p = Arc::clone(&p);
                s.spawn(move || {
                    for _ in 0..4 {
                        let pin = p.pin(id, Insert::Mru).expect("pin");
                        assert_eq!(
                            &pin[..],
                            &vec![id as u8; 512][..],
                            "block {id} got other bytes"
                        );
                    }
                });
            }
        });
    }

    /// `try_pin` must not move the counters. The CPU read path calls it once per op on top of the
    /// pin the op's pre-step already took, so counting it would add exactly one hit per access:
    /// a cache thrashing at 0% would report ~50%, and a perfect one 100% — the number stops
    /// distinguishing the two cases it exists to distinguish.
    #[test]
    fn try_pin_does_not_move_the_counters() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 32, io, &[1]);
        drop(p.pin(1, Insert::Mru).expect("pin")); // 1 miss
        let before = p.stats().pager;
        for _ in 0..10 {
            drop(p.try_pin(1).expect("resident"));
        }
        assert!(p.try_pin(999).is_none());
        let after = p.stats().pager;
        assert_eq!(
            (after.hits, after.misses),
            (before.hits, before.misses),
            "re-borrowing a pinned block is not a residency decision"
        );
    }

    /// `fill`'s three outcomes, each forced and each identified — and the bytes are the block's own
    /// in all three, which is what a caller staging them into a GPU ring depends on.
    ///
    /// Admission needs a SECOND miss, so the first pass over a block set streams entirely and the
    /// arena fills on the second.
    #[test]
    fn fill_admits_on_the_second_miss_then_streams_when_full() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 16, io.clone(), &[1, 2, 3]);
        let mut dst = [0u8; 16];

        // First sight of each block: streamed, arena untouched.
        for id in [1u32, 2, 3] {
            assert_eq!(p.fill(id, &mut dst).unwrap(), Fill::Streamed);
            assert_eq!(&dst, &[id as u8; 16]);
        }
        assert_eq!(p.stats().pager.misses, 0, "nothing may be admitted yet");

        // Second sight: admitted until the two slots are gone, then streamed again.
        assert_eq!(p.fill(1, &mut dst).unwrap(), Fill::Admitted);
        assert_eq!(dst, [1u8; 16]);
        assert_eq!(p.fill(2, &mut dst).unwrap(), Fill::Admitted);
        assert_eq!(dst, [2u8; 16]);
        // Arena full: block 3 must NOT evict either resident block, and must still deliver.
        assert_eq!(p.fill(3, &mut dst).unwrap(), Fill::Streamed);
        assert_eq!(dst, [3u8; 16]);
        // Re-asking for a resident block is a hit that reads nothing.
        let before = io.reads.load(Ordering::SeqCst);
        assert_eq!(p.fill(1, &mut dst).unwrap(), Fill::Hit);
        assert_eq!(dst, [1u8; 16]);
        assert_eq!(
            io.reads.load(Ordering::SeqCst),
            before,
            "a hit read the file"
        );

        let s = p.stats();
        assert_eq!(s.pager.evictions, 0, "a full arena must stream, not evict");
        assert_eq!((s.reads, s.streamed), (6, 4));
        assert_eq!(s.bytes_read, 96);
    }

    /// The doorkeeper's whole purpose: a block the tier ABOVE keeps resident calls down exactly
    /// once, and must never take an arena slot — otherwise the arena fills with blocks that can
    /// never be hit again. Measured on Qwen3-14B before this rule: 4 of 9 slots per pool dead.
    ///
    /// Modelled here as a tier above that keeps block 1 after its first miss (so 1 is never filled
    /// again) while blocks 2 and 3 keep coming back.
    #[test]
    fn a_block_the_tier_above_keeps_never_takes_a_slot() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 16, io, &[1, 2, 3]);
        let mut dst = [0u8; 16];

        // Pass 1: everything misses down to here.
        for id in [1u32, 2, 3] {
            assert_eq!(p.fill(id, &mut dst).unwrap(), Fill::Streamed);
        }
        // Passes 2 and 3: block 1 is resident above and never calls down again.
        for _ in 0..2 {
            for id in [2u32, 3] {
                p.fill(id, &mut dst).unwrap();
                assert_eq!(&dst, &[id as u8; 16]);
            }
        }
        // Both slots went to the blocks that kept coming back, not to block 1.
        assert_eq!(p.fill(2, &mut dst).unwrap(), Fill::Hit);
        assert_eq!(p.fill(3, &mut dst).unwrap(), Fill::Hit);
    }

    /// Streaming past a full arena must not disturb what the arena holds — that is the whole point
    /// of not evicting, and a slot quietly overwritten by a streamed block would be silent garbage.
    #[test]
    fn a_streamed_block_leaves_the_resident_set_alone() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 16, io, &[1, 2, 3, 4, 5]);
        let mut dst = [0u8; 16];
        // Two passes: the first only arms the doorkeeper, the second seats 1 and 2.
        for id in [1u32, 2, 1, 2] {
            p.fill(id, &mut dst).unwrap();
        }
        for id in [3u32, 4, 5, 3, 4, 5] {
            assert_eq!(p.fill(id, &mut dst).unwrap(), Fill::Streamed);
            assert_eq!(
                &dst, &[id as u8; 16],
                "streamed block {id} read wrong bytes"
            );
        }
        for id in [1u32, 2] {
            assert_eq!(p.fill(id, &mut dst).unwrap(), Fill::Hit);
            assert_eq!(&dst, &[id as u8; 16], "resident block {id} was disturbed");
        }
    }

    /// The unified-memory shape: an arena-less tier delivers every block's own bytes, caches
    /// nothing, and commits no host memory. Repeated asks must re-read rather than start hitting,
    /// because there is nowhere for a hit to come from.
    #[test]
    fn a_stream_only_tier_reads_through_and_caches_nothing() {
        let io = Arc::new(FakeIo::new());
        let p = HostPager::stream_only(16, io.clone()).expect("stream-only");
        for id in [1u32, 2, 3] {
            p.register(BlockDesc {
                id,
                extents: vec![BlockExtent {
                    offset: id as u64 * 16,
                    len: 16,
                }],
            })
            .expect("register");
        }
        assert!(!p.caches(), "a stream-only tier must not claim to cache");
        assert_eq!(p.arena_bytes(), 0, "it must commit no host memory");
        assert_eq!(p.n_slots(), 0);

        let mut dst = [0u8; 16];
        // Two full passes: every ask is a fresh read, and every ask gets the right bytes.
        for _ in 0..2 {
            for id in [1u32, 2, 3] {
                assert_eq!(p.fill(id, &mut dst).unwrap(), Fill::Streamed);
                assert_eq!(&dst, &[id as u8; 16], "block {id} read wrong bytes");
            }
        }
        let s = p.stats();
        assert_eq!(s.reads, 6, "every ask must reach the file");
        assert_eq!(s.streamed, 6, "and every read must be a streamed one");
        assert_eq!(s.pager.hits, 0, "nothing can hit with no arena");
        assert_eq!(s.pager.evictions, 0);
    }

    /// `pin` hands out a borrow of arena bytes, so it must be REFUSED rather than return an empty
    /// view of an arena that does not exist — that would decode as silent garbage.
    #[test]
    fn a_stream_only_tier_refuses_to_pin() {
        let io = Arc::new(FakeIo::new());
        let p = HostPager::stream_only(16, io).expect("stream-only");
        p.register(BlockDesc {
            id: 1,
            extents: vec![BlockExtent { offset: 0, len: 16 }],
        })
        .expect("register");
        let err = p.pin(1, Insert::Cold).expect_err("pin must be refused");
        assert!(
            err.to_string().contains("stream-only"),
            "unexpected error: {err}"
        );
        assert!(p.try_pin(1).is_none(), "try_pin must find nothing resident");
    }

    /// A failed read leaves nothing resident on the admit path, exactly as `pin` does.
    #[test]
    fn a_failed_fill_leaves_no_resident_block() {
        let io = Arc::new(FakeIo {
            reads: AtomicUsize::new(0),
            fail_on: Some(2),
            delay: None,
        });
        let p = pager_with(2, 16, io, &[1, 2]);
        let mut dst = [0u8; 16];
        // The doorkeeper is armed BEFORE the read is attempted, so the first call streams and
        // fails while still marking block 2 seen; the second is the one that admits — and it is
        // that admitted-then-failed fill whose cleanup this test is about.
        assert!(p.fill(2, &mut dst).is_err());
        assert!(p.fill(2, &mut dst).is_err());
        assert_eq!(p.stats().pager.hits, 0);
        // The slot is free again, so block 1 admits — once its own doorkeeper miss is spent.
        assert_eq!(p.fill(1, &mut dst).unwrap(), Fill::Streamed);
        assert_eq!(p.fill(1, &mut dst).unwrap(), Fill::Admitted);
    }

    #[test]
    fn an_unregistered_block_is_rejected() {
        let io = Arc::new(FakeIo::new());
        let p = pager_with(2, 16, io, &[1]);
        let err = p.pin(42, Insert::Mru).expect_err("must reject");
        assert!(err.to_string().contains("never registered"), "{err}");
    }

    #[test]
    fn a_block_larger_than_the_slot_is_rejected_at_registration() {
        let io = Arc::new(FakeIo::new());
        let p = HostPager::new(2, 16, io).expect("host pager");
        let err = p.register(desc(1, 17)).expect_err("must reject");
        assert!(err.to_string().contains("slot stride is 16"), "{err}");
    }
}
