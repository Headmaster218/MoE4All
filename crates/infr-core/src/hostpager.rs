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
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

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
}

/// Cumulative tier activity. The residency half comes from the [`Pager`]; the I/O half is what
/// says whether a good hit rate was earned or was never tested.
#[derive(Debug, Clone, Copy)]
pub struct HostPagerStats {
    pub pager: PagerStats,
    /// Blocks actually read from the tier below.
    pub reads: u64,
    pub bytes_read: u64,
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
    reads: AtomicU64,
    bytes_read: AtomicU64,
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
            }),
            ready: Condvar::new(),
            arena: Arena::new(n_slots, slot_bytes),
            io,
            slot_bytes,
            reads: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        })
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

    /// Open a new touch batch — same meaning as [`Pager::begin_batch`].
    pub fn begin_batch(&self) {
        self.inner.lock().unwrap().pager.begin_batch();
    }

    pub fn stats(&self) -> HostPagerStats {
        HostPagerStats {
            pager: self.inner.lock().unwrap().pager.stats(),
            reads: self.reads.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
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

    pub fn n_slots(&self) -> usize {
        self.inner.lock().unwrap().pager.n_slots()
    }

    /// Pin `id`'s bytes, reading them from the tier below if they are not resident.
    ///
    /// Blocks on I/O when it misses, and blocks while another thread fills the same block — but
    /// never waits for a pin to be released: a caller that finds every slot pinned gets the
    /// exhaustion error instead, because waiting on an unordered set of pins acquired one at a time
    /// is a deadlock, not a slow path.
    pub fn pin(&self, id: BlockId, insert: Insert) -> Result<Pin<'_>> {
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

    /// Pin `id` only if it is already resident and readable — never reads from the tier below.
    ///
    /// Two callers, one shape: a reader re-borrowing a block it already pinned (the CPU op body,
    /// after its pre-step), and a tier above probing before going one tier down. Neither is a
    /// residency DECISION, so this moves no counter — see [`Pager::repin`]. A caller that wants the
    /// probe counted keeps its own tally.
    pub fn try_pin(&self, id: BlockId) -> Option<Pin<'_>> {
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
