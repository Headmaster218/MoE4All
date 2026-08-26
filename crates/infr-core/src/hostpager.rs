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
use crate::pager_profile;
use std::collections::{HashMap, HashSet};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Stable, page-aligned host storage suitable for optional
/// `VK_EXT_external_memory_host` import. The logical byte count remains the pager budget; only the
/// final allocation page is rounded up. Allocation is lazy-zeroed by the operating system on the
/// native targets, matching the old zero-filled arena without eagerly touching a multi-GiB cache.
pub struct AlignedHostBuffer {
    ptr: NonNull<u8>,
    len: usize,
    allocated_len: usize,
}

// SAFETY: the allocation is plain bytes with a stable address. Users establish non-overlap and
// lifetime at the slot/cache layer; this owner neither creates references nor mutates metadata.
unsafe impl Send for AlignedHostBuffer {}
unsafe impl Sync for AlignedHostBuffer {}

impl AlignedHostBuffer {
    /// 64 KiB covers Windows allocation granularity and the 4 KiB import requirement reported by
    /// current AMD drivers. A Vulkan backend with a stricter runtime requirement simply declines
    /// the optional import and keeps the CPU-copy path.
    pub const ALIGNMENT: usize = 64 * 1024;

    pub fn new(len: usize) -> Result<Arc<Self>> {
        if len == 0 {
            return Ok(Arc::new(Self {
                ptr: NonNull::dangling(),
                len: 0,
                allocated_len: 0,
            }));
        }
        let allocated_len = len
            .checked_add(Self::ALIGNMENT - 1)
            .map(|n| n / Self::ALIGNMENT * Self::ALIGNMENT)
            .ok_or_else(|| Error::backend("aligned host allocation size overflow".to_string()))?;

        #[cfg(windows)]
        let ptr = {
            use windows::Win32::System::Memory::{
                VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
            };
            let raw = unsafe {
                VirtualAlloc(
                    None,
                    allocated_len,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            NonNull::new(raw.cast::<u8>()).ok_or_else(|| {
                Error::backend(format!(
                    "VirtualAlloc could not reserve {allocated_len} bytes for the host pager"
                ))
            })?
        };

        #[cfg(unix)]
        let ptr = {
            let raw = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    allocated_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if raw == libc::MAP_FAILED {
                return Err(Error::backend(format!(
                    "mmap could not reserve {allocated_len} bytes for the host pager"
                )));
            }
            NonNull::new(raw.cast::<u8>()).expect("mmap success returned null")
        };

        #[cfg(not(any(windows, unix)))]
        let ptr = {
            let layout = std::alloc::Layout::from_size_align(allocated_len, Self::ALIGNMENT)
                .map_err(|e| {
                    Error::backend(format!("invalid host pager allocation layout: {e}"))
                })?;
            NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).ok_or_else(|| {
                Error::backend(format!(
                    "allocator could not reserve {allocated_len} bytes for the host pager"
                ))
            })?
        };

        Ok(Arc::new(Self {
            ptr,
            len,
            allocated_len,
        }))
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn allocated_len(&self) -> usize {
        self.allocated_len
    }

    /// Return the byte offset of a range that lies wholly inside the logical allocation.
    pub fn offset_of(&self, ptr: *const u8, len: usize) -> Option<usize> {
        let base = self.ptr.as_ptr() as usize;
        let start = ptr as usize;
        let offset = start.checked_sub(base)?;
        (offset.checked_add(len)? <= self.len).then_some(offset)
    }

    /// # Safety
    /// The caller must guarantee exclusive access to this range for the duration of the returned
    /// slice. Pager slot state and model-load sequencing provide that guarantee at current uses.
    pub unsafe fn slice_mut(&self, offset: usize, len: usize) -> &mut [u8] {
        debug_assert!(offset.saturating_add(len) <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(offset), len) }
    }

    /// # Safety
    /// No writer may overlap the returned range for its lifetime.
    pub unsafe fn slice(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(offset.saturating_add(len) <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(offset), len) }
    }
}

impl Drop for AlignedHostBuffer {
    fn drop(&mut self) {
        if self.allocated_len == 0 {
            return;
        }
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
            let _ = VirtualFree(self.ptr.as_ptr().cast(), 0, MEM_RELEASE);
        }
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.allocated_len);
        }
        #[cfg(not(any(windows, unix)))]
        unsafe {
            let layout =
                std::alloc::Layout::from_size_align_unchecked(self.allocated_len, Self::ALIGNMENT);
            std::alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

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
    allocation: Arc<AlignedHostBuffer>,
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
    fn new(n_slots: usize, slot_bytes: usize) -> Result<Self> {
        let total = n_slots * slot_bytes;
        Ok(Self {
            allocation: AlignedHostBuffer::new(total)?,
            total,
            slot_bytes,
        })
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
        unsafe { self.allocation.as_ptr().add(self.offset(slot)) }
    }

    /// # Safety
    /// The caller must hold a pin on the block resident in `slot`, and that block must be `Ready`,
    /// so no writer can be active on this slot; `len <= slot_bytes`.
    unsafe fn slot_ref(&self, slot: u32, len: usize) -> &[u8] {
        debug_assert!(len <= self.slot_bytes);
        debug_assert!(self.offset(slot) + len <= self.total);
        std::slice::from_raw_parts(self.allocation.as_ptr().add(self.offset(slot)), len)
    }

    /// # Safety
    /// The caller must have exclusively reserved `slot` in `Loading` state, and no reader may
    /// hold a reference to it; `len <= slot_bytes`.
    unsafe fn slot_mut(&self, slot: u32, len: usize) -> &mut [u8] {
        debug_assert!(len <= self.slot_bytes);
        debug_assert!(self.offset(slot) + len <= self.total);
        std::slice::from_raw_parts_mut(self.allocation.as_ptr().add(self.offset(slot)), len)
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

/// Activity of the inclusive RAM tier used underneath the discrete-GPU MoE cache. GPU-resident
/// blocks remain pinned in this arena when capacity permits, so an upper-tier eviction releases a
/// shadow rather than copying immutable bytes back from mapped VRAM.
#[derive(Debug, Clone, Copy, Default)]
pub struct InclusiveHostStats {
    pub preload_reads: u64,
    pub bytes_preloaded: u64,
    pub ram_hits: u64,
    pub ssd_reads: u64,
    pub ram_evictions: u64,
    pub gpu_evictions: u64,
    pub shadow_promotions: u64,
    pub shadow_releases: u64,
    pub shadow_resident: usize,
    pub bytes_read: u64,
    pub bytes_promoted: u64,
}

struct InclusiveInner {
    pager: Option<Pager>,
    state: HashMap<BlockId, SlotState>,
    descs: HashMap<BlockId, BlockDesc>,
    /// GPU-resident blocks whose immutable bytes are retained in the host arena. Each member owns
    /// exactly one long-lived Pager pin and therefore cannot be selected as a cold RAM victim.
    shadows: HashSet<BlockId>,
    /// One per-size-class read buffer. It is transient working memory, not a weight mirror: an
    /// SSD miss uses it only when every RAM slot is already pinned by another GPU shadow.
    scratch: Box<[u8]>,
}

/// Temporary, non-touching borrows held while a Prefill bank is assembled from the inclusive
/// cache. `Pager::repin` leaves Decode's LRU/epoch/stats untouched; dropping the set releases only
/// these additional borrows, including when a parallel SSD read returns an error.
struct InclusiveReadPins<'a> {
    cache: &'a InclusiveHostCache,
    ids: Vec<BlockId>,
}

impl Drop for InclusiveReadPins<'_> {
    fn drop(&mut self) {
        if self.ids.is_empty() {
            return;
        }
        let mut inner = self.cache.inner.lock().unwrap();
        let pager = inner
            .pager
            .as_mut()
            .expect("resident inclusive blocks require a RAM arena");
        for &id in &self.ids {
            pager.unpin(id);
        }
    }
}

/// Fixed-budget, inclusive RAM cache beneath a faster cache of the same uniform blocks.
///
/// A promoted block stays in RAM and is pinned while it remains in GPU cache. When the GPU evicts
/// it, the pin is released and the same bytes become the MRU cold entry. This spends part of the
/// RAM budget on GPU shadows but makes GPU eviction metadata-only: no CPU read from mapped ReBAR.
/// If RAM has fewer slots than the GPU cache, additional GPU residents stream from SSD through the
/// scratch buffer and simply have no shadow; correctness never depends on a shadow being present.
pub struct InclusiveHostCache {
    inner: Mutex<InclusiveInner>,
    ready: Condvar,
    arena: Arena,
    io: Arc<dyn BlockIo>,
    slot_bytes: usize,
    preload_reads: AtomicU64,
    bytes_preloaded: AtomicU64,
    ram_hits: AtomicU64,
    ssd_reads: AtomicU64,
    ram_evictions: AtomicU64,
    gpu_evictions: AtomicU64,
    shadow_promotions: AtomicU64,
    shadow_releases: AtomicU64,
    bytes_read: AtomicU64,
    bytes_promoted: AtomicU64,
}

impl InclusiveHostCache {
    /// Build one RAM size class. `n_slots == 0` is a valid SSD-through mode; the one-block
    /// scratch still lets a miss be uploaded without retaining a host shadow.
    pub fn new(n_slots: usize, slot_bytes: usize, io: Arc<dyn BlockIo>) -> Result<Self> {
        if slot_bytes == 0 {
            return Err(Error::backend(
                "inclusive host cache needs a non-zero block stride".to_string(),
            ));
        }
        Ok(Self {
            inner: Mutex::new(InclusiveInner {
                pager: (n_slots > 0).then(|| Pager::new(n_slots)),
                state: HashMap::new(),
                descs: HashMap::new(),
                shadows: HashSet::new(),
                scratch: vec![0u8; slot_bytes].into_boxed_slice(),
            }),
            ready: Condvar::new(),
            arena: Arena::new(n_slots, slot_bytes)?,
            io,
            slot_bytes,
            preload_reads: AtomicU64::new(0),
            bytes_preloaded: AtomicU64::new(0),
            ram_hits: AtomicU64::new(0),
            ssd_reads: AtomicU64::new(0),
            ram_evictions: AtomicU64::new(0),
            gpu_evictions: AtomicU64::new(0),
            shadow_promotions: AtomicU64::new(0),
            shadow_releases: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_promoted: AtomicU64::new(0),
        })
    }

    pub fn register(&self, desc: BlockDesc) -> Result<()> {
        let n = desc.nbytes();
        if n > self.slot_bytes {
            return Err(Error::backend(format!(
                "inclusive host cache: block {} is {n} bytes, slot stride is {}",
                desc.id, self.slot_bytes
            )));
        }
        self.inner.lock().unwrap().descs.insert(desc.id, desc);
        Ok(())
    }

    pub fn block_bytes(&self, id: BlockId) -> Option<usize> {
        self.inner
            .lock()
            .unwrap()
            .descs
            .get(&id)
            .map(BlockDesc::nbytes)
    }

    pub fn arena_bytes(&self) -> usize {
        self.arena.total
    }

    pub fn arena_allocation(&self) -> Arc<AlignedHostBuffer> {
        Arc::clone(&self.arena.allocation)
    }

    pub fn n_slots(&self) -> usize {
        self.arena.total / self.slot_bytes
    }

    pub fn stats(&self) -> InclusiveHostStats {
        let shadow_resident = self.inner.lock().unwrap().shadows.len();
        InclusiveHostStats {
            preload_reads: self.preload_reads.load(Ordering::Relaxed),
            bytes_preloaded: self.bytes_preloaded.load(Ordering::Relaxed),
            ram_hits: self.ram_hits.load(Ordering::Relaxed),
            ssd_reads: self.ssd_reads.load(Ordering::Relaxed),
            ram_evictions: self.ram_evictions.load(Ordering::Relaxed),
            gpu_evictions: self.gpu_evictions.load(Ordering::Relaxed),
            shadow_promotions: self.shadow_promotions.load(Ordering::Relaxed),
            shadow_releases: self.shadow_releases.load(Ordering::Relaxed),
            shadow_resident,
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_promoted: self.bytes_promoted.load(Ordering::Relaxed),
        }
    }

    /// Fill empty RAM slots from registered blocks before the first upper-tier access.
    ///
    /// This is a cold-load operation: callers choose the block set and invoke it after all model
    /// banks are registered but before execution starts. It seeds the same Pager/LRU used by
    /// [`Self::promote`], so a later request is an ordinary RAM hit and can become a pinned GPU
    /// shadow without another disk read. Already-ready ids are harmless and skipped.
    pub fn preload(&self, ids: &[BlockId]) -> Result<(usize, usize)> {
        let mut inner = self.inner.lock().unwrap();
        if inner.pager.is_none() || ids.is_empty() {
            return Ok((0, 0));
        }

        let mut loaded = 0usize;
        let mut bytes = 0usize;
        for &id in ids {
            while inner.state.get(&id) == Some(&SlotState::Loading) {
                inner = self.ready.wait(inner).unwrap();
            }
            if inner.state.get(&id) == Some(&SlotState::Ready) {
                continue;
            }
            let desc = inner.descs.get(&id).cloned().ok_or_else(|| {
                Error::backend(format!(
                    "inclusive host cache: preload block {id} was never registered"
                ))
            })?;
            let len = desc.nbytes();
            let pager = inner.pager.as_mut().expect("preload checked the arena");
            if pager.resident_count() >= self.n_slots() {
                return Err(Error::backend(format!(
                    "inclusive host cache: preload selected more than {} RAM slots",
                    self.n_slots()
                )));
            }
            let slot = match pager.touch(id) {
                Resolution::Miss { slot, evicted } => {
                    debug_assert!(evicted.is_none(), "preload only fills empty RAM slots");
                    slot
                }
                Resolution::Hit { .. } => {
                    return Err(Error::backend(format!(
                        "inclusive host cache: preload block {id} is resident without ready bytes"
                    )));
                }
            };
            inner.state.insert(id, SlotState::Loading);
            let dst = unsafe { self.arena.slot_mut(slot, len) };
            if let Err(err) = self.io.read_block(&desc, dst) {
                inner.state.remove(&id);
                let removed = inner
                    .pager
                    .as_mut()
                    .expect("preload checked the arena")
                    .evict(id);
                debug_assert_eq!(removed, Some(slot));
                drop(inner);
                self.ready.notify_all();
                return Err(err);
            }
            inner.state.insert(id, SlotState::Ready);
            loaded += 1;
            bytes += len;
            self.preload_reads.fetch_add(1, Ordering::Relaxed);
            self.bytes_preloaded
                .fetch_add(len as u64, Ordering::Relaxed);
        }
        drop(inner);
        self.ready.notify_all();
        Ok((loaded, bytes))
    }

    /// Promote `requested` through `upload` while retaining its RAM bytes as a pinned GPU shadow.
    /// `evicted` is released to the cold RAM LRU before resolving the request, making one host slot
    /// eligible even when every retained byte was previously a shadow. No upper-tier bytes are
    /// ever read. `None` is the warm-up case where the GPU still had a free ordinary slot.
    pub fn promote<U>(&self, requested: BlockId, evicted: Option<BlockId>, upload: U) -> Result<()>
    where
        U: FnOnce(&[u8]) -> Result<()>,
    {
        let prof = pager_profile::active();
        let mut inner = self.inner.lock().unwrap();
        while inner.state.get(&requested) == Some(&SlotState::Loading) {
            let wait_t0 = prof.then(std::time::Instant::now);
            inner = self.ready.wait(inner).unwrap();
            if let Some(t0) = wait_t0 {
                pager_profile::record_host_wait(t0.elapsed());
            }
        }
        let desc = inner.descs.get(&requested).cloned().ok_or_else(|| {
            Error::backend(format!(
                "inclusive host cache: block {requested} was never registered"
            ))
        })?;
        let len = desc.nbytes();

        if let Some(victim) = evicted {
            self.gpu_evictions.fetch_add(1, Ordering::Relaxed);
            if Self::release_shadow_locked(&mut inner, victim) {
                self.shadow_releases.fetch_add(1, Ordering::Relaxed);
            }
        }

        let ram_slot = inner
            .state
            .get(&requested)
            .filter(|&&state| state == SlotState::Ready)
            .and_then(|_| inner.pager.as_ref()?.slot_of(requested));
        if let Some(slot) = ram_slot {
            let pinned_slot = inner
                .pager
                .as_mut()
                .and_then(|pager| pager.pin_if_resident(requested))
                .expect("ready RAM resident must still have a pager slot");
            debug_assert_eq!(pinned_slot, slot);
            let inserted = inner.shadows.insert(requested);
            debug_assert!(inserted, "GPU miss requested an existing host shadow");
            let src = unsafe { self.arena.slot_ref(slot, len) };
            if let Err(err) = upload(src) {
                inner.shadows.remove(&requested);
                inner
                    .pager
                    .as_mut()
                    .expect("RAM hit requires an arena")
                    .unpin(requested);
                return Err(err);
            }
            self.ram_hits.fetch_add(1, Ordering::Relaxed);
            self.bytes_promoted.fetch_add(len as u64, Ordering::Relaxed);
            self.shadow_promotions.fetch_add(1, Ordering::Relaxed);
            if prof {
                pager_profile::record_host_hit(len);
            }
            return Ok(());
        }

        // Admit the SSD result directly into an eligible cold RAM slot. If all host slots are
        // shadows (possible when the RAM budget is smaller than GPU capacity), stream through the
        // scratch buffer and leave this GPU resident unshadowed.
        let admission = inner
            .pager
            .as_mut()
            .and_then(|pager| pager.resolve_and_pin(requested, Insert::Mru));
        if let Some(Resolution::Miss { slot, evicted }) = admission {
            let host_evicted = evicted.is_some();
            if let Some(old) = evicted {
                let removed = inner.state.remove(&old);
                debug_assert_eq!(removed, Some(SlotState::Ready));
                self.ram_evictions.fetch_add(1, Ordering::Relaxed);
            }
            inner.state.insert(requested, SlotState::Loading);
            let dst = unsafe { self.arena.slot_mut(slot, len) };
            let read_t0 = prof.then(std::time::Instant::now);
            if let Err(err) = self.io.read_block(&desc, dst) {
                inner.state.remove(&requested);
                let pager = inner.pager.as_mut().expect("admission requires an arena");
                pager.unpin(requested);
                let removed = pager.evict(requested);
                debug_assert_eq!(removed, Some(slot));
                drop(inner);
                self.ready.notify_all();
                return Err(err);
            }
            if let Some(t0) = read_t0 {
                pager_profile::record_host_read(len, t0.elapsed(), false);
            }
            if let Err(err) = upload(dst) {
                inner.state.remove(&requested);
                let pager = inner.pager.as_mut().expect("admission requires an arena");
                pager.unpin(requested);
                let removed = pager.evict(requested);
                debug_assert_eq!(removed, Some(slot));
                drop(inner);
                self.ready.notify_all();
                return Err(err);
            }
            inner.state.insert(requested, SlotState::Ready);
            let inserted = inner.shadows.insert(requested);
            debug_assert!(inserted);
            self.shadow_promotions.fetch_add(1, Ordering::Relaxed);
            if prof {
                pager_profile::record_host_miss(len, host_evicted);
            }
            self.ready.notify_all();
        } else if admission.is_none() {
            let read_t0 = prof.then(std::time::Instant::now);
            self.io.read_block(&desc, &mut inner.scratch[..len])?;
            if let Some(t0) = read_t0 {
                pager_profile::record_host_read(len, t0.elapsed(), true);
            }
            upload(&inner.scratch[..len])?;
            if prof {
                pager_profile::record_host_miss(len, false);
            }
        } else {
            return Err(Error::backend(format!(
                "inclusive host cache: block {requested} became resident without ready bytes"
            )));
        }
        self.ssd_reads.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(len as u64, Ordering::Relaxed);
        self.bytes_promoted.fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Promote several independent upper-tier misses as one I/O batch.
    ///
    /// Residency and victim selection stay serial and in caller order under `inner`, exactly like
    /// repeated [`Self::promote`] calls. Only the expensive portion after those decisions is
    /// parallel: SSD fills and the caller's uploads. Holding the residency lock until every job
    /// completes keeps the same safety contract as `promote` (no external shadow release can make
    /// an arena slot reusable while a worker still reads or writes it).
    ///
    /// `requests` must contain unique block ids and independent upload destinations. `context` is
    /// opaque caller data, typically a mapped-device destination address.
    pub fn promote_batch<T, U>(
        &self,
        requests: &[(BlockId, Option<BlockId>, T)],
        upload: U,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        U: Fn(&[u8], T) -> Result<()> + Send + Sync,
    {
        use rayon::prelude::*;

        enum Source {
            Hit {
                slot: u32,
            },
            Fill {
                slot: u32,
                desc: BlockDesc,
                host_evicted: bool,
            },
            Stream {
                desc: BlockDesc,
            },
        }
        struct Work<T> {
            requested: BlockId,
            context: T,
            len: usize,
            source: Source,
        }

        if requests.is_empty() {
            return Ok(());
        }
        let prof = pager_profile::active();
        let mut inner = self.inner.lock().unwrap();
        let mut unique = HashSet::with_capacity(requests.len());
        for &(requested, _, _) in requests {
            if !unique.insert(requested) {
                return Err(Error::backend(format!(
                    "inclusive host cache batch contains duplicate block {requested}"
                )));
            }
            if !inner.descs.contains_key(&requested) {
                return Err(Error::backend(format!(
                    "inclusive host cache: block {requested} was never registered"
                )));
            }
        }

        let mut work = Vec::with_capacity(requests.len());
        for &(requested, upper_evicted, context) in requests {
            while inner.state.get(&requested) == Some(&SlotState::Loading) {
                let wait_t0 = prof.then(std::time::Instant::now);
                inner = self.ready.wait(inner).unwrap();
                if let Some(t0) = wait_t0 {
                    pager_profile::record_host_wait(t0.elapsed());
                }
            }
            let desc = inner.descs[&requested].clone();
            let len = desc.nbytes();

            if let Some(victim) = upper_evicted {
                self.gpu_evictions.fetch_add(1, Ordering::Relaxed);
                if Self::release_shadow_locked(&mut inner, victim) {
                    self.shadow_releases.fetch_add(1, Ordering::Relaxed);
                }
            }

            let ram_slot = inner
                .state
                .get(&requested)
                .filter(|&&state| state == SlotState::Ready)
                .and_then(|_| inner.pager.as_ref()?.slot_of(requested));
            if let Some(slot) = ram_slot {
                let pinned_slot = inner
                    .pager
                    .as_mut()
                    .and_then(|pager| pager.pin_if_resident(requested))
                    .expect("ready RAM resident must still have a pager slot");
                debug_assert_eq!(pinned_slot, slot);
                let inserted = inner.shadows.insert(requested);
                debug_assert!(inserted, "GPU miss requested an existing host shadow");
                work.push(Work {
                    requested,
                    context,
                    len,
                    source: Source::Hit { slot },
                });
                continue;
            }

            let admission = inner
                .pager
                .as_mut()
                .and_then(|pager| pager.resolve_and_pin(requested, Insert::Mru));
            match admission {
                Some(Resolution::Miss { slot, evicted }) => {
                    let host_evicted = evicted.is_some();
                    if let Some(old) = evicted {
                        let removed = inner.state.remove(&old);
                        debug_assert_eq!(removed, Some(SlotState::Ready));
                        self.ram_evictions.fetch_add(1, Ordering::Relaxed);
                    }
                    inner.state.insert(requested, SlotState::Loading);
                    work.push(Work {
                        requested,
                        context,
                        len,
                        source: Source::Fill {
                            slot,
                            desc,
                            host_evicted,
                        },
                    });
                }
                None => work.push(Work {
                    requested,
                    context,
                    len,
                    source: Source::Stream { desc },
                }),
                Some(Resolution::Hit { .. }) => {
                    return Err(Error::backend(format!(
                        "inclusive host cache: block {requested} became resident without ready bytes"
                    )));
                }
            }
        }

        // SAFETY of the arena slices below: every Hit is pinned as a shadow before later planning;
        // every Fill owns a distinct Loading+pin slot. `inner` remains locked until all workers
        // join, so no external release can remove those pins or recycle a slot in the meantime.
        let outcomes: Vec<Result<Option<std::time::Duration>>> = work
            .par_iter()
            .map(|job| match &job.source {
                Source::Hit { slot } => {
                    let src = unsafe { self.arena.slot_ref(*slot, job.len) };
                    upload(src, job.context)?;
                    Ok(None)
                }
                Source::Fill { slot, desc, .. } => {
                    let dst = unsafe { self.arena.slot_mut(*slot, job.len) };
                    let started = prof.then(std::time::Instant::now);
                    self.io.read_block(desc, dst)?;
                    let elapsed = started.map(|t| t.elapsed());
                    upload(dst, job.context)?;
                    Ok(elapsed)
                }
                Source::Stream { desc } => {
                    // This is only reachable when every RAM slot is a GPU shadow (including the
                    // zero-RAM mode). One temporary per in-flight miss lets the batch stay parallel
                    // without creating a persistent second weight store.
                    let mut bytes = vec![0u8; job.len];
                    let started = prof.then(std::time::Instant::now);
                    self.io.read_block(desc, &mut bytes)?;
                    let elapsed = started.map(|t| t.elapsed());
                    upload(&bytes, job.context)?;
                    Ok(elapsed)
                }
            })
            .collect();

        let mut first_error = None;
        let mut notify = false;
        for (job, outcome) in work.iter().zip(outcomes) {
            match outcome {
                Ok(read_elapsed) => {
                    match &job.source {
                        Source::Hit { .. } => {
                            self.ram_hits.fetch_add(1, Ordering::Relaxed);
                            self.shadow_promotions.fetch_add(1, Ordering::Relaxed);
                            if prof {
                                pager_profile::record_host_hit(job.len);
                            }
                        }
                        Source::Fill { host_evicted, .. } => {
                            inner.state.insert(job.requested, SlotState::Ready);
                            let inserted = inner.shadows.insert(job.requested);
                            debug_assert!(inserted);
                            self.shadow_promotions.fetch_add(1, Ordering::Relaxed);
                            self.ssd_reads.fetch_add(1, Ordering::Relaxed);
                            self.bytes_read.fetch_add(job.len as u64, Ordering::Relaxed);
                            if prof {
                                pager_profile::record_host_miss(job.len, *host_evicted);
                            }
                            notify = true;
                        }
                        Source::Stream { .. } => {
                            self.ssd_reads.fetch_add(1, Ordering::Relaxed);
                            self.bytes_read.fetch_add(job.len as u64, Ordering::Relaxed);
                            if prof {
                                pager_profile::record_host_miss(job.len, false);
                            }
                        }
                    }
                    if let Some(elapsed) = read_elapsed {
                        pager_profile::record_host_read(
                            job.len,
                            elapsed,
                            matches!(&job.source, Source::Stream { .. }),
                        );
                    }
                    self.bytes_promoted
                        .fetch_add(job.len as u64, Ordering::Relaxed);
                }
                Err(err) => {
                    match &job.source {
                        Source::Hit { .. } => {
                            if inner.shadows.remove(&job.requested) {
                                inner
                                    .pager
                                    .as_mut()
                                    .expect("RAM hit requires an arena")
                                    .unpin(job.requested);
                            }
                        }
                        Source::Fill { slot, .. } => {
                            inner.state.remove(&job.requested);
                            let pager = inner.pager.as_mut().expect("admission requires an arena");
                            pager.unpin(job.requested);
                            let removed = pager.evict(job.requested);
                            debug_assert_eq!(removed, Some(*slot));
                            notify = true;
                        }
                        Source::Stream { .. } => {}
                    }
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        drop(inner);
        if notify {
            self.ready.notify_all();
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Notify the host tier that GPU residency ended outside ordinary decode replacement, such as
    /// unified-arena borrowing or a Prefill lane overwriting Decode slots.
    pub fn release_gpu_blocks(&self, ids: &[BlockId]) {
        if ids.is_empty() {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        let mut released = 0u64;
        for &id in ids {
            released += u64::from(Self::release_shadow_locked(&mut inner, id));
        }
        self.gpu_evictions
            .fetch_add(ids.len() as u64, Ordering::Relaxed);
        self.shadow_releases.fetch_add(released, Ordering::Relaxed);
    }

    fn release_shadow_locked(inner: &mut InclusiveInner, id: BlockId) -> bool {
        if !inner.shadows.remove(&id) {
            return false;
        }
        debug_assert_eq!(inner.state.get(&id), Some(&SlotState::Ready));
        inner
            .pager
            .as_mut()
            .expect("a host shadow requires an arena")
            .unpin_mru(id);
        true
    }

    /// Assemble one contiguous Prefill bank from the existing inclusive RAM cache plus SSD misses.
    ///
    /// Ready blocks are borrowed with [`Pager::repin`], so this sequential whole-model sweep does
    /// not perturb Decode's LRU order, epoch protection, or residency counters. Missing blocks read
    /// directly into their final `dst` offsets and are deliberately not admitted: filling a bounded
    /// cache with a one-use Prefill sweep would evict the routed working set Decode is about to use.
    pub fn materialize_stream(
        &self,
        block_base: BlockId,
        n_blocks: usize,
        block_bytes: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        use rayon::prelude::*;

        enum Source {
            Ram { slot: u32 },
            Ssd { desc: BlockDesc },
        }

        struct Work {
            id: BlockId,
            offset: usize,
            len: usize,
            source: Source,
        }

        #[derive(Clone, Copy)]
        enum Kind {
            Ram,
            Ssd,
        }

        struct Outcome {
            kind: Kind,
            len: usize,
            elapsed: std::time::Duration,
        }

        if block_bytes == 0 {
            return Err(Error::backend(
                "inclusive host cache: Prefill block stride is zero".to_string(),
            ));
        }
        let need = n_blocks.checked_mul(block_bytes).ok_or_else(|| {
            Error::backend("inclusive host cache: Prefill bank size overflow".to_string())
        })?;
        if dst.len() < need {
            return Err(Error::backend(format!(
                "inclusive host cache: Prefill bank needs {need} bytes, destination holds {}",
                dst.len()
            )));
        }

        let mut ids = Vec::with_capacity(n_blocks);
        for block in 0..n_blocks {
            let local = u32::try_from(block).map_err(|_| {
                Error::backend("inclusive host cache: Prefill block count exceeds u32".to_string())
            })?;
            ids.push(block_base.checked_add(local).ok_or_else(|| {
                Error::backend("inclusive host cache: Prefill block id overflow".to_string())
            })?);
        }

        let mut inner = self.inner.lock().unwrap();
        for &id in &ids {
            let desc = inner.descs.get(&id).ok_or_else(|| {
                Error::backend(format!(
                    "inclusive host cache: Prefill block {id} was never registered"
                ))
            })?;
            if desc.nbytes() != block_bytes {
                return Err(Error::backend(format!(
                    "inclusive host cache: Prefill block {id} is {} bytes, expected {block_bytes}",
                    desc.nbytes()
                )));
            }
        }

        let mut work = Vec::with_capacity(n_blocks);
        let mut pinned = Vec::new();
        for (block, &id) in ids.iter().enumerate() {
            loop {
                if inner.state.get(&id) == Some(&SlotState::Ready) {
                    let slot = inner
                        .pager
                        .as_mut()
                        .and_then(|pager| pager.repin(id))
                        .expect("ready inclusive block must remain resident");
                    pinned.push(id);
                    work.push(Work {
                        id,
                        offset: block * block_bytes,
                        len: block_bytes,
                        source: Source::Ram { slot },
                    });
                    break;
                }
                if inner.state.get(&id) == Some(&SlotState::Loading) {
                    inner = self.ready.wait(inner).unwrap();
                    continue;
                }
                work.push(Work {
                    id,
                    offset: block * block_bytes,
                    len: block_bytes,
                    source: Source::Ssd {
                        desc: inner.descs[&id].clone(),
                    },
                });
                break;
            }
        }
        let pins = InclusiveReadPins {
            cache: self,
            ids: pinned,
        };
        drop(inner);

        let prof = pager_profile::active();
        let dst_addr = dst.as_mut_ptr() as usize;
        let outcomes: Vec<Result<Outcome>> = work
            .par_iter()
            .map(|job| {
                let started = std::time::Instant::now();
                // SAFETY: every job owns one distinct `[offset, offset + len)` block range in
                // `dst`. RAM sources remain Ready and un-evictable through `pins`; SSD jobs only
                // write their private destination range. All workers join before `dst` is reused.
                let out = unsafe {
                    std::slice::from_raw_parts_mut((dst_addr + job.offset) as *mut u8, job.len)
                };
                let kind = match &job.source {
                    Source::Ram { slot } => {
                        let src = unsafe { self.arena.slot_ref(*slot, job.len) };
                        out.copy_from_slice(src);
                        Kind::Ram
                    }
                    Source::Ssd { desc } => {
                        self.io.read_block(desc, out)?;
                        Kind::Ssd
                    }
                };
                Ok(Outcome {
                    kind,
                    len: job.len,
                    elapsed: started.elapsed(),
                })
            })
            .collect();

        let mut first_error = None;
        let mut ram_blocks = 0u64;
        let mut ram_bytes = 0u64;
        let mut ssd_blocks = 0u64;
        let mut ssd_bytes = 0u64;
        for (job, outcome) in work.iter().zip(outcomes) {
            match outcome {
                Ok(outcome) => match outcome.kind {
                    Kind::Ram => {
                        ram_blocks += 1;
                        ram_bytes += outcome.len as u64;
                        if prof {
                            pager_profile::record_host_hit(outcome.len);
                            pager_profile::record_memcpy(outcome.len, outcome.elapsed);
                        }
                    }
                    Kind::Ssd => {
                        ssd_blocks += 1;
                        ssd_bytes += outcome.len as u64;
                        if prof {
                            pager_profile::record_host_miss(outcome.len, false);
                            pager_profile::record_host_read(outcome.len, outcome.elapsed, true);
                        }
                    }
                },
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(Error::backend(format!(
                            "inclusive host cache: Prefill block {} failed: {err}",
                            job.id
                        )));
                    }
                }
            }
        }
        self.ram_hits.fetch_add(ram_blocks, Ordering::Relaxed);
        self.ssd_reads.fetch_add(ssd_blocks, Ordering::Relaxed);
        self.bytes_read.fetch_add(ssd_bytes, Ordering::Relaxed);
        self.bytes_promoted
            .fetch_add(ram_bytes.saturating_add(ssd_bytes), Ordering::Relaxed);
        drop(pins);

        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
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
            arena: Arena::new(n_slots, slot_bytes)?,
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
            arena: Arena::new(0, slot_bytes)?,
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
        let prof = pager_profile::active();
        let (slot, desc, evicted_slot) = loop {
            let mut inner = self.inner.lock().unwrap();
            if !inner.descs.contains_key(&id) {
                return Err(Error::backend(format!(
                    "host pager: block {id} was never registered"
                )));
            }
            // Resident and readable? Take the pin and go.
            if inner.state.get(&id) == Some(&SlotState::Ready) {
                if let Some(slot) = inner.pager.pin_if_resident(id) {
                    let pin = self.pinned(id, slot, &inner);
                    if prof {
                        pager_profile::record_host_hit(pin.len());
                    }
                    return Ok(pin);
                }
            }
            // Being filled by someone else: wait for them rather than read a half-written slot.
            if inner.state.get(&id) == Some(&SlotState::Loading) {
                let t0 = prof.then(std::time::Instant::now);
                let waited = self.ready.wait(inner).unwrap();
                if let Some(t0) = t0 {
                    pager_profile::record_host_wait(t0.elapsed());
                }
                drop(waited);
                continue;
            }
            match inner.pager.resolve_and_pin(id, insert) {
                Some(Resolution::Hit { slot }) => {
                    // Resident with no state entry cannot happen: state and residency are set
                    // together under this lock. Treat it as a fill to re-establish the invariant
                    // rather than handing out bytes nothing wrote.
                    debug_assert!(false, "resident block {id} had no slot state");
                    inner.state.insert(id, SlotState::Loading);
                    break (slot, inner.descs[&id].clone(), false);
                }
                Some(Resolution::Miss { slot, evicted }) => {
                    let evicted_slot = evicted.is_some();
                    if let Some(e) = evicted {
                        inner.state.remove(&e);
                    }
                    inner.state.insert(id, SlotState::Loading);
                    break (slot, inner.descs[&id].clone(), evicted_slot);
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
        let read_t0 = prof.then(std::time::Instant::now);
        let read = self.io.read_block(&desc, dst);
        let read_elapsed = read_t0.map(|t| t.elapsed());
        let mut inner = self.inner.lock().unwrap();
        match read {
            Ok(()) => {
                if prof {
                    pager_profile::record_host_miss(len, evicted_slot);
                    if let Some(elapsed) = read_elapsed {
                        pager_profile::record_host_read(len, elapsed, false);
                    }
                }
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
        let prof = pager_profile::active();
        // `(admitted slot, descriptor)` — the slot is `None` when this call will stream.
        let (slot, desc, evicted_slot) = loop {
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
                    let n = pin.len();
                    if prof {
                        pager_profile::record_host_hit(n);
                    }
                    let copy_t0 = prof.then(std::time::Instant::now);
                    dst[..n].copy_from_slice(&pin);
                    if let Some(t0) = copy_t0 {
                        pager_profile::record_memcpy(n, t0.elapsed());
                    }
                    return Ok(Fill::Hit);
                }
            }
            if inner.state.get(&id) == Some(&SlotState::Loading) {
                let t0 = prof.then(std::time::Instant::now);
                let waited = self.ready.wait(inner).unwrap();
                if let Some(t0) = t0 {
                    pager_profile::record_host_wait(t0.elapsed());
                }
                drop(waited);
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
                        let evicted_slot = evicted.is_some();
                        debug_assert!(evicted.is_none(), "a free slot cannot have evicted");
                        inner.state.insert(id, SlotState::Loading);
                        break (Some(slot), desc, evicted_slot);
                    }
                    // Resident without a state entry, or every slot pinned. Neither is reachable
                    // here (the `Ready` arm above covers the first, and this path pins only across
                    // its own fill), and both are correctly served by streaming the block.
                    _ => break (None, desc, false),
                }
            }
            break (None, desc, false);
        };

        let Some(slot) = slot else {
            // Streamed: no arena involvement at all, so no lock and no residency change.
            let n = desc.nbytes();
            let read_t0 = prof.then(std::time::Instant::now);
            self.io.read_block(&desc, &mut dst[..n])?;
            if prof {
                pager_profile::record_host_miss(n, false);
                if let Some(t0) = read_t0 {
                    pager_profile::record_host_read(n, t0.elapsed(), true);
                }
            }
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
        let read_t0 = prof.then(std::time::Instant::now);
        let read = self.io.read_block(&desc, arena);
        let read_elapsed = read_t0.map(|t| t.elapsed());
        let mut inner = self.inner.lock().unwrap();
        match read {
            Ok(()) => {
                if prof {
                    pager_profile::record_host_miss(n, evicted_slot);
                    if let Some(elapsed) = read_elapsed {
                        pager_profile::record_host_read(n, elapsed, false);
                    }
                }
                inner.state.insert(id, SlotState::Ready);
                self.reads.fetch_add(1, Ordering::Relaxed);
                self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
                // The guard adopts the pin `resolve_and_pin` took and releases it on drop, so the
                // copy below runs with the slot un-evictable and nothing is released twice.
                let pin = self.pinned(id, slot, &inner);
                drop(inner);
                let copy_t0 = prof.then(std::time::Instant::now);
                dst[..n].copy_from_slice(&pin);
                if let Some(t0) = copy_t0 {
                    pager_profile::record_memcpy(n, t0.elapsed());
                }
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

    #[test]
    fn aligned_host_buffer_is_zeroed_aligned_and_range_checked() {
        let logical = AlignedHostBuffer::ALIGNMENT + 17;
        let buffer = AlignedHostBuffer::new(logical).expect("allocate aligned host buffer");
        assert_eq!((buffer.as_ptr() as usize) % AlignedHostBuffer::ALIGNMENT, 0);
        assert_eq!(buffer.len(), logical);
        assert!(buffer.allocated_len() >= logical);
        assert!(buffer
            .allocated_len()
            .is_multiple_of(AlignedHostBuffer::ALIGNMENT));
        assert!(unsafe { buffer.slice(0, logical) }
            .iter()
            .all(|&byte| byte == 0));
        assert_eq!(
            buffer.offset_of(unsafe { buffer.as_ptr().add(9) }, 8),
            Some(9)
        );
        assert_eq!(
            buffer.offset_of(unsafe { buffer.as_ptr().add(logical - 1) }, 2),
            None,
        );
    }

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

    struct ConcurrentIo {
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl BlockIo for ConcurrentIo {
        fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            dst[..desc.nbytes()].fill(desc.id as u8);
            self.active.fetch_sub(1, Ordering::SeqCst);
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

    #[test]
    fn inclusive_batch_reads_distinct_misses_concurrently() {
        let io = Arc::new(ConcurrentIo {
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        });
        let cache = InclusiveHostCache::new(2, 16, io.clone()).expect("cache");
        cache.register(desc(1, 16)).expect("register 1");
        cache.register(desc(2, 16)).expect("register 2");
        let uploaded = std::sync::Mutex::new([[0u8; 16]; 2]);

        cache
            .promote_batch(&[(1, None, 0usize), (2, None, 1usize)], |bytes, dst| {
                uploaded.lock().unwrap()[dst].copy_from_slice(bytes);
                Ok(())
            })
            .expect("batch promotion");

        assert_eq!(*uploaded.lock().unwrap(), [[1u8; 16], [2u8; 16]]);
        assert_eq!(
            io.max_active.load(Ordering::SeqCst),
            2,
            "the two SSD misses were serialized"
        );
        let stats = cache.stats();
        assert_eq!(stats.ssd_reads, 2);
        assert_eq!(stats.shadow_resident, 2);
    }

    #[test]
    fn inclusive_prefill_stream_reuses_ram_without_polluting_decode_lru() {
        let io = Arc::new(FakeIo::new());
        let cache = InclusiveHostCache::new(2, 16, io.clone()).expect("cache");
        for id in 1..=4 {
            cache.register(desc(id, 16)).expect("register");
        }
        // Deliberate cold-to-hot order: if the Prefill scan touched its hits, block 3 would move
        // behind block 1 and the later Decode admission would evict the wrong block.
        cache.preload(&[3, 1]).expect("preload");

        let mut bank = [0u8; 64];
        cache
            .materialize_stream(1, 4, 16, &mut bank)
            .expect("materialize Prefill bank");
        for (block, bytes) in bank.chunks_exact(16).enumerate() {
            assert_eq!(bytes, &[(block + 1) as u8; 16]);
        }
        assert_eq!(
            io.reads.load(Ordering::SeqCst),
            4,
            "two preloaded blocks must replace two SSD reads"
        );
        let stats = cache.stats();
        assert_eq!(stats.ram_hits, 2);
        assert_eq!(stats.ssd_reads, 2);
        assert_eq!(stats.bytes_read, 32);
        assert_eq!(stats.bytes_promoted, 64);
        assert_eq!(
            stats.ram_evictions, 0,
            "Prefill misses must not be admitted"
        );

        cache
            .promote(2, None, |bytes| {
                assert_eq!(bytes, &[2u8; 16]);
                Ok(())
            })
            .expect("Decode promotion 2");
        cache
            .promote(3, Some(2), |bytes| {
                assert_eq!(bytes, &[3u8; 16]);
                Ok(())
            })
            .expect("Decode promotion 3");
        assert_eq!(
            cache.stats().ssd_reads,
            4,
            "block 3 must remain the LRU victim; the Prefill scan may not promote it"
        );
    }

    #[test]
    fn inclusive_prefill_stream_reads_distinct_ssd_misses_concurrently() {
        let io = Arc::new(ConcurrentIo {
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        });
        let cache = InclusiveHostCache::new(1, 16, io.clone()).expect("cache");
        for id in 1..=3 {
            cache.register(desc(id, 16)).expect("register");
        }
        cache.preload(&[1]).expect("preload");
        io.max_active.store(0, Ordering::SeqCst);

        let mut bank = [0u8; 48];
        cache
            .materialize_stream(1, 3, 16, &mut bank)
            .expect("materialize Prefill bank");
        assert!(
            io.max_active.load(Ordering::SeqCst) >= 2,
            "the two SSD-only blocks were serialized"
        );
        assert_eq!(cache.stats().ram_hits, 1);
        assert_eq!(cache.stats().ssd_reads, 2);
    }

    #[test]
    fn inclusive_cache_preload_becomes_gpu_shadows_without_runtime_reads() {
        let io = Arc::new(FakeIo::new());
        let cache = InclusiveHostCache::new(2, 16, io.clone()).expect("cache");
        for id in 1..=3 {
            cache.register(desc(id, 16)).expect("register");
        }

        assert_eq!(cache.preload(&[1, 3]).expect("preload"), (2, 32));
        let mut promoted = [0u8; 16];
        for (requested, victim) in [(3, None), (1, Some(3)), (3, Some(1))] {
            cache
                .promote(requested, victim, |bytes| {
                    promoted.copy_from_slice(bytes);
                    Ok(())
                })
                .expect("RAM promotion");
            assert_eq!(promoted, [requested as u8; 16]);
        }

        let stats = cache.stats();
        assert_eq!(stats.preload_reads, 2);
        assert_eq!(stats.bytes_preloaded, 32);
        assert_eq!(stats.ram_hits, 3);
        assert_eq!(stats.ssd_reads, 0);
        assert_eq!(stats.gpu_evictions, 2);
        assert_eq!(stats.shadow_promotions, 3);
        assert_eq!(stats.shadow_releases, 2);
        assert_eq!(stats.shadow_resident, 1);
        assert_eq!(io.reads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn inclusive_cache_evicts_cold_ram_but_never_a_gpu_shadow() {
        let io = Arc::new(FakeIo::new());
        let cache = InclusiveHostCache::new(2, 16, io.clone()).expect("cache");
        for id in 1..=3 {
            cache.register(desc(id, 16)).expect("register");
        }
        cache.preload(&[1, 2]).expect("preload");

        cache.promote(1, None, |_| Ok(())).expect("shadow 1");
        cache
            .promote(3, Some(1), |bytes| {
                assert_eq!(bytes, &[3u8; 16]);
                Ok(())
            })
            .expect("SSD promotion 3");
        cache
            .promote(1, Some(3), |bytes| {
                assert_eq!(bytes, &[1u8; 16]);
                Ok(())
            })
            .expect("retained cold block 1");

        let stats = cache.stats();
        assert_eq!(stats.ram_hits, 2);
        assert_eq!(stats.ssd_reads, 1);
        assert_eq!(stats.ram_evictions, 1);
        assert_eq!(stats.gpu_evictions, 2);
        assert_eq!(stats.shadow_resident, 1);
        assert_eq!(io.reads.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn inclusive_cache_streams_when_every_host_slot_is_a_shadow() {
        let io = Arc::new(FakeIo::new());
        let cache = InclusiveHostCache::new(2, 16, io.clone()).expect("cache");
        for id in 1..=4 {
            cache.register(desc(id, 16)).expect("register");
        }

        for id in [1, 2, 3] {
            cache
                .promote(id, None, |bytes| {
                    assert_eq!(bytes, &[id as u8; 16]);
                    Ok(())
                })
                .expect("warm GPU promotion");
        }
        cache
            .promote(4, Some(3), |bytes| {
                assert_eq!(bytes, &[4u8; 16]);
                Ok(())
            })
            .expect("unshadowed victim");
        cache
            .promote(3, Some(1), |bytes| {
                assert_eq!(bytes, &[3u8; 16]);
                Ok(())
            })
            .expect("released shadow makes an admission slot");

        let stats = cache.stats();
        assert_eq!(stats.ssd_reads, 5);
        assert_eq!(stats.ram_evictions, 1);
        assert_eq!(stats.gpu_evictions, 2);
        assert_eq!(stats.shadow_promotions, 3);
        assert_eq!(stats.shadow_releases, 1);
        assert_eq!(stats.shadow_resident, 2);
    }

    #[test]
    fn inclusive_cache_external_gpu_release_makes_shadow_cold() {
        let io = Arc::new(FakeIo::new());
        let cache = InclusiveHostCache::new(2, 16, io.clone()).expect("cache");
        for id in 1..=3 {
            cache.register(desc(id, 16)).expect("register");
        }
        cache.preload(&[1, 2]).expect("preload");
        cache.promote(1, None, |_| Ok(())).expect("shadow 1");
        cache.promote(2, None, |_| Ok(())).expect("shadow 2");

        cache.release_gpu_blocks(&[1]);
        cache.promote(3, None, |_| Ok(())).expect("admit 3");

        let stats = cache.stats();
        assert_eq!(stats.gpu_evictions, 1);
        assert_eq!(stats.shadow_releases, 1);
        assert_eq!(stats.shadow_resident, 2);
        assert_eq!(stats.ram_evictions, 1);
    }

    #[test]
    fn inclusive_zero_ram_mode_streams_and_drops_gpu_victims() {
        let io = Arc::new(FakeIo::new());
        let cache = InclusiveHostCache::new(0, 16, io.clone()).expect("cache");
        for id in 1..=2 {
            cache.register(desc(id, 16)).expect("register");
        }

        for (requested, victim) in [(1, None), (2, Some(1)), (1, Some(2))] {
            cache
                .promote(requested, victim, |bytes| {
                    assert_eq!(bytes, &[requested as u8; 16]);
                    Ok(())
                })
                .expect("direct SSD promotion");
        }
        let stats = cache.stats();
        assert_eq!(stats.ssd_reads, 3);
        assert_eq!(stats.ram_hits, 0);
        assert_eq!(stats.gpu_evictions, 2);
        assert_eq!(stats.shadow_promotions, 0);
        assert_eq!(stats.shadow_resident, 0);
        assert_eq!(cache.arena_bytes(), 0);
        assert_eq!(io.reads.load(Ordering::SeqCst), 3);
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
