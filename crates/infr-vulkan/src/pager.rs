//! GPU-resident paged weight caches. MoE owns one CPU-only layer-major expert store and a mapped
//! ReBAR VRAM arena: Prefill CPU-pushes complete layers into resident/A-B placements, while Decode
//! resolves `(layer, expert)` offsets into the same store and pushes misses into expert-LRU slots.
//! The full payload exists in physical RAM once and is never exposed as a GPU-visible HostWeights
//! mirror. Dense streaming retains its independent staging ring because its sources and scheduling
//! contract are different.
//!
//! # Design (block-agnostic core, MoE plugs in today)
//! [`GpuPager`] only knows about uniform `slot_bytes`-sized blocks keyed by an opaque
//! `infr_core::pager::BlockId` — it has no idea a block is "an expert". The MoE integration
//! (`infr-llama`'s seam / this crate's `adapter.rs`) packs a `BlockId` from `(layer, role,
//! expert_id)`. The MoE session calls `plan_host_dma` from its permanent store before dispatching
//! the id-indexed GEMV/GEMM through the LUT hop (the `PAGED` branch in
//! `shaders/native_gemv_id.comp` / `native_gemv_id_multi.comp`: `slot = lut[ids[slot]]`, scaled
//! onto the arena's 64-bit device address as `arena_addr + slot * slot_bytes` — see the `lut_host`
//! field's doc and `shaders/native_weight_addr.glsl`). Dense layer streaming below reuses the
//! same arena bookkeeping with schedule-driven rather than LRU placement.
//!
//! # LUT
//! The host keeps an `n_blocks`-entry mirror of per-block resident SLOT INDICES
//! (`infr_core::pager::NOT_RESIDENT` for an absent block). The paged EXECUTION path never reads a
//! live device LUT: each (layer, role) batch freezes its `n_expert`-entry window into the
//! session's append-only LUT tape ([`MoePagerSession::lut_window`]) at record time, so staging
//! for later layers can keep mutating the mirror while earlier recorded-but-in-flight segments
//! read a consistent view. The classic per-pager device LUT + [`GpuPager::flush_lut`] remain for
//! the standalone [`GpuPager::ensure_resident`] surface (parity tests / future non-MoE users).
//!
//! # Eviction policy
//! Classic LRU for recency-driven touches (decode's routed-only path) plus the scan-resistant
//! cold-end insertion (`infr_core::pager::Pager::touch_cold`) for the batched prefill's
//! full-set sweeps — see that method's doc for why plain LRU is pathological there. llama.cpp
//! issue #20757's SLRU-with-admission remains the documented upgrade if these thrash on an
//! adversarial pattern.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use ash::vk;

use infr_core::backend::{Buffer, BufferUsage};
use infr_core::error::Result;
use infr_core::hostpager::HostPager;
use infr_core::pager::{BlockId, Pager, PagerStats, Resolution, NOT_RESIDENT};
use infr_core::pager_profile;
use infr_core::Backend;

use super::{as_vk_buf, be, VulkanBackend};

/// Validate [`GpuPager::new`]'s block dimensions. Pure (no GPU) so it can be unit-tested and so a
/// bad seam budget (0 slots) or sizing bug (misaligned stride) returns `Err` before any allocation.
fn validate_pager_dims(n_slots: usize, slot_bytes: usize) -> Result<()> {
    if n_slots == 0 {
        return Err(be("GpuPager needs at least one slot"));
    }
    if !slot_bytes.is_multiple_of(4) {
        return Err(be(
            "GpuPager slot_bytes must be u32-aligned (the arena is read as u32 words)",
        ));
    }
    Ok(())
}

/// Apply one LUT-mirror placement to `lut_host`: clear an evicted block's entry to `NOT_RESIDENT`,
/// then record the newly-resident block's SLOT INDEX. Pure over the mirror slice (no `lut_dirty`,
/// no GPU) so the eviction/insert bookkeeping — the one place a wrong LUT entry becomes silent-zero
/// MoE output — is unit-testable. Out-of-range ids are ignored (mirrors the old inline
/// `get_mut(..)` guards). See [`GpuPager::record_placement`].
fn apply_placement(lut_host: &mut [u32], id: BlockId, slot: u32, evicted: Option<u32>) {
    if let Some(e) = evicted {
        if let Some(v) = lut_host.get_mut(e as usize) {
            *v = NOT_RESIDENT;
        }
    }
    if let Some(v) = lut_host.get_mut(id as usize) {
        // Slot index — the shader scales it onto the arena's 64-bit base address (see the
        // `lut_host` field's doc).
        *v = slot;
    }
}

/// Fixed-budget evictable VRAM cache of uniform `slot_bytes` blocks. See the module doc.
pub struct GpuPager {
    pager: Pager,
    slot_bytes: usize,
    /// Device-local arena: `n_slots * slot_bytes`, one contiguous `bufferDeviceAddress` buffer.
    /// Both the MoE pools and the dense-streaming pools allocate this and their paged/streamed
    /// kernels read it through a 64-bit pointer (see [`Self::arena_addr`]) — no
    /// `maxStorageBufferRange` cap.
    arena: Arc<dyn Buffer>,
    /// Byte offset of this pager's decode pool inside a shared MoE arena. Standalone/dense
    /// pagers own their arena and use zero.
    arena_offset: usize,
    /// This arena's 64-bit `VkDeviceAddress`. The MoE paged kernels compute a slot's byte address
    /// as `arena_addr + lut[...] * slot_bytes`; the dense streamed kernels take the resident
    /// slot's full base address `arena_addr + slot * slot_bytes` (host-computed in
    /// [`DensePagerSession::stage`]) directly as a push constant.
    arena_addr: u64,
    /// Host-visible LUT mirror (mutated in place, re-uploaded on change) + the device buffer it's
    /// pushed to. `n_blocks` entries, each the resident block's SLOT INDEX
    /// (`infr_core::pager::NOT_RESIDENT` for an absent block). The paged MoE kernels read this slot
    /// index (through the session's frozen tape window) and compute the slot's byte address as
    /// `arena_addr + uint64_t(slot) * slot_bytes` in 64-bit — the multiply that used to wrap u32 in
    /// element space (Scout: 41.9M elements/expert overflowed at slot ≥ ~102, the original
    /// coherent-but-wrong bug) is now done on the device address, so no arena size overflows it.
    /// The dense-streaming pool keeps this mirror coherent but never reads it (its dispatch bakes
    /// the slot into a weight element offset instead).
    lut_host: Vec<u32>,
    lut_dev: Box<dyn Buffer>,
    lut_dirty: bool,
}

impl GpuPager {
    /// `n_blocks`: total distinct `BlockId`s that can ever be named (the LUT's fixed size — for
    /// MoE, `n_paged_layers * n_roles * n_experts`). `n_slots`: the VRAM budget in blocks
    /// (`budget_bytes / slot_bytes`, computed by the caller from remaining VRAM — see the
    /// within-batch sizing note on `infr_core::pager::Pager::new`, which applies unchanged here).
    /// `slot_bytes`: one block's PADDED byte size (the largest block the model will ever page —
    /// MoE experts of one model are uniform per role, so this is exact, not a worst-case pad).
    /// Must be u32-aligned (`% 4 == 0`) — the arena is read back a word at a time (see
    /// `shaders/native_weight_addr.glsl`'s `arena_word`).
    ///
    /// The arena always allocates as a `bufferDeviceAddress` buffer, read through a 64-bit
    /// pointer, so it may be as large as VRAM allows — no `maxStorageBufferRange` cap (both the
    /// MoE pools and the dense-streaming pool have taken this path since `36bcbf5`).
    pub fn new(
        vk: &VulkanBackend,
        n_blocks: usize,
        n_slots: usize,
        slot_bytes: usize,
    ) -> Result<Self> {
        // Both are reachable from a too-small seam VRAM budget (0 slots) or a sizing bug, i.e.
        // recoverable input — return `Err` rather than aborting the process.
        validate_pager_dims(n_slots, slot_bytes)?;
        // Pointer-addressed: no per-arena binding cap — a pool spans as much VRAM as the budget
        // allows (the alloc-time VRAM budget guard is the only backstop).
        let (arena, arena_addr) = vk.alloc_arena_bda(n_slots * slot_bytes)?;
        Self::new_in_arena(
            vk,
            n_blocks,
            n_slots,
            slot_bytes,
            Arc::from(arena),
            arena_addr,
            0,
        )
    }

    /// MoE twin of [`Self::new`] whose final cache allocation is mapped ReBAR VRAM. Each pool gets
    /// one allocation so Windows AMD never has to map the whole multi-GiB logical cache as one
    /// `VkDeviceMemory`; the shader already consumes a per-pool BDA base.
    fn new_mapped(
        vk: &VulkanBackend,
        n_blocks: usize,
        n_slots: usize,
        slot_bytes: usize,
    ) -> Result<Self> {
        validate_pager_dims(n_slots, slot_bytes)?;
        let (arena, arena_addr) = vk.alloc_mapped_arena_bda(n_slots * slot_bytes)?;
        Self::new_in_arena(
            vk,
            n_blocks,
            n_slots,
            slot_bytes,
            Arc::from(arena),
            arena_addr,
            0,
        )
    }

    fn new_in_arena(
        vk: &VulkanBackend,
        n_blocks: usize,
        n_slots: usize,
        slot_bytes: usize,
        arena: Arc<dyn Buffer>,
        arena_addr: u64,
        arena_offset: usize,
    ) -> Result<Self> {
        validate_pager_dims(n_slots, slot_bytes)?;
        let lut_dev = vk.alloc_uninit(n_blocks.max(1) * 4, BufferUsage::Staging)?;
        let lut_host = vec![NOT_RESIDENT; n_blocks.max(1)];
        // Seed the device LUT with the same all-absent state (arena/LUT start coherent).
        vk.upload(lut_dev.as_ref(), bytemuck::cast_slice(&lut_host))?;
        Ok(Self {
            pager: Pager::new(n_slots),
            slot_bytes,
            arena,
            arena_offset,
            arena_addr,
            lut_host,
            lut_dev,
            lut_dirty: false,
        })
    }

    /// The arena's 64-bit `VkDeviceAddress`. The paged kernels take this as a push constant and
    /// add `lut_slot * slot_bytes` to reach an expert.
    pub fn arena_addr(&self) -> u64 {
        self.arena_addr
    }

    pub fn n_slots(&self) -> usize {
        self.pager.n_slots()
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub fn stats(&self) -> PagerStats {
        self.pager.stats()
    }

    pub fn arena_buffer(&self) -> &dyn Buffer {
        self.arena.as_ref()
    }

    pub fn lut_buffer(&self) -> &dyn Buffer {
        self.lut_dev.as_ref()
    }

    /// Already-resident check with NO mutation (for a caller that wants to decide whether it even
    /// needs `bytes` in hand before calling `ensure_resident` — e.g. skip a host dequant/gather on
    /// a hit).
    pub fn is_resident(&self, id: BlockId) -> bool {
        self.pager.slot_of(id).is_some()
    }

    /// Drop all host-side residency metadata without reallocating the arena. Used when the same
    /// physical bytes stop being a contiguous prefill-layer slot and become a decode expert LRU.
    fn reset_residency(&mut self) {
        self.pager = Pager::new(self.pager.n_slots());
        self.lut_host.fill(NOT_RESIDENT);
        self.lut_dirty = true;
    }

    /// [`Self::ensure_resident`]'s RECORDED twin: on a miss, memcpy `bytes` into the caller's
    /// staging ring at `ring_off` (a host-mapped write) and record the ring→arena slot copy
    /// through `rec` instead of submitting an immediate one-shot — the caller batches many
    /// misses (and whole layers of compute) into one submission. Contract: the ring region
    /// `[ring_off, ring_off + slot_bytes)` must stay untouched until that recording's submit
    /// completes (the adapter's fenced ring-half rotation enforces this). The HOST LUT mirror is
    /// updated exactly like `ensure_resident`; the device-visible copy is the caller's frozen
    /// tape window (see [`MoePagerSession::lut_window`]) — `flush_lut` is NOT required on this
    /// path. Returns the ring bytes consumed (0 on a hit).
    pub fn touch_staged(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        ring: &dyn Buffer,
        ring_off: usize,
        id: BlockId,
        bytes: &[u8],
        scan: bool,
    ) -> Result<usize> {
        debug_assert_eq!(
            bytes.len(),
            self.slot_bytes,
            "block byte size must match the arena's slot size"
        );
        let Some(dst) = self.plan_staged(rec, ring, ring_off, id, scan)? else {
            return Ok(0);
        };
        let copy_t0 = pager_profile::active().then(std::time::Instant::now);
        par_copy_to_mapped(bytes, dst as *mut u8);
        if let Some(t0) = copy_t0 {
            pager_profile::record_memcpy(bytes.len(), t0.elapsed());
        }
        Ok(self.slot_bytes)
    }

    /// Resolve and record one staged upload without copying its host bytes yet. The MoE session
    /// uses this to collect a whole layer/role's independent expert copies and execute them in one
    /// rayon batch; [`Self::touch_staged`] remains the one-copy wrapper for other callers.
    fn plan_staged(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        ring: &dyn Buffer,
        ring_off: usize,
        id: BlockId,
        scan: bool,
    ) -> Result<Option<usize>> {
        // `scan`: full-set sweep (batched prefill's touch-all) → the scan-resistant cold-end
        // policy; otherwise classic LRU (decode's routed-only touches). See
        // `infr_core::pager::Pager::touch_cold`.
        let prof = pager_profile::active();
        let lookup_t0 = prof.then(std::time::Instant::now);
        let resolution = if scan {
            self.pager.touch_cold(id)
        } else {
            self.pager.touch(id)
        };
        if let Some(t0) = lookup_t0 {
            let (hit, evicted) = match resolution {
                Resolution::Hit { .. } => (true, false),
                Resolution::Miss { evicted, .. } => (false, evicted.is_some()),
            };
            pager_profile::record_gpu_cache_lookup(hit, evicted, t0.elapsed());
        }
        match resolution {
            Resolution::Hit { .. } => Ok(None),
            Resolution::Miss { slot, evicted } => {
                let base = as_vk_buf(ring)?
                    .mapped_ptr()
                    .ok_or_else(|| be("pager staging ring is not persistently mapped"))?;
                rec.copy(
                    ring,
                    ring_off,
                    self.arena.as_ref(),
                    self.arena_offset + slot as usize * self.slot_bytes,
                    self.slot_bytes,
                );
                if prof {
                    pager_profile::record_gpu_copy(self.slot_bytes);
                }
                self.record_placement(id, slot, evicted);
                Ok(Some(unsafe { base.add(ring_off) } as usize))
            }
        }
    }

    /// Resolve one block and return its final mapped-ReBAR LRU destination on a miss. The caller
    /// CPU-pushes from the unique host store straight into that byte range; no GPU-visible host
    /// source or staging mirror exists.
    fn plan_cpu_push(&mut self, id: BlockId, scan: bool) -> Result<Option<usize>> {
        let prof = pager_profile::active();
        let lookup_t0 = prof.then(std::time::Instant::now);
        let resolution = if scan {
            self.pager.touch_cold(id)
        } else {
            self.pager.touch(id)
        };
        if let Some(t0) = lookup_t0 {
            let (hit, evicted) = match resolution {
                Resolution::Hit { .. } => (true, false),
                Resolution::Miss { evicted, .. } => (false, evicted.is_some()),
            };
            pager_profile::record_gpu_cache_lookup(hit, evicted, t0.elapsed());
        }
        match resolution {
            Resolution::Hit { .. } => Ok(None),
            Resolution::Miss { slot, evicted } => {
                self.record_placement(id, slot, evicted);
                Ok(Some(self.arena_offset + slot as usize * self.slot_bytes))
            }
        }
    }

    /// [`Self::touch_staged`]'s twin for a block whose bytes are READ rather than copied — the
    /// arena-less host tier a unified-memory device gets (`HostPager::stream_only`).
    ///
    /// Same residency decision and the same ring→arena copy; only the source differs, so the two
    /// cannot drift on policy. Kept separate rather than folded behind a closure because the copy
    /// they perform is genuinely different work: the mmap path memcpys a slice with
    /// [`par_copy_to_mapped`] (many threads over one already-resident buffer), while this issues a
    /// positioned read straight into the ring — one copy instead of the read-then-copy that going
    /// through a host arena would cost.
    pub fn touch_staged_read(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        ring: &dyn Buffer,
        ring_off: usize,
        id: BlockId,
        host: &HostPager,
        scan: bool,
    ) -> Result<usize> {
        let prof = pager_profile::active();
        let lookup_t0 = prof.then(std::time::Instant::now);
        let resolution = if scan {
            self.pager.touch_cold(id)
        } else {
            self.pager.touch(id)
        };
        if let Some(t0) = lookup_t0 {
            let (hit, evicted) = match resolution {
                Resolution::Hit { .. } => (true, false),
                Resolution::Miss { evicted, .. } => (false, evicted.is_some()),
            };
            pager_profile::record_gpu_cache_lookup(hit, evicted, t0.elapsed());
        }
        match resolution {
            Resolution::Hit { .. } => Ok(0),
            Resolution::Miss { slot, evicted } => {
                let n = host.block_bytes(id).ok_or_else(|| {
                    be(format!("moe pager: block {id} is unknown to the host tier"))
                })?;
                debug_assert!(
                    n <= self.slot_bytes,
                    "block bytes ({n}) exceed the arena's slot stride ({})",
                    self.slot_bytes
                );
                let base = as_vk_buf(ring)?
                    .mapped_ptr()
                    .ok_or_else(|| be("pager staging ring is not persistently mapped"))?;
                // SAFETY: `[ring_off, ring_off + n)` is this caller's own region of the
                // persistently-mapped ring — reserved by the cursor before this call and not
                // reused until the recording that reads it completes (the caller's ring
                // contract). No other thread holds a reference to it.
                let dst = unsafe { std::slice::from_raw_parts_mut(base.add(ring_off), n) };
                host.fill(id, dst)?;
                rec.copy(
                    ring,
                    ring_off,
                    self.arena.as_ref(),
                    self.arena_offset + slot as usize * self.slot_bytes,
                    self.slot_bytes,
                );
                if prof {
                    pager_profile::record_gpu_copy(self.slot_bytes);
                }
                self.record_placement(id, slot, evicted);
                Ok(self.slot_bytes)
            }
        }
    }

    /// `n` host-mirror LUT words starting at block id `base` — the source a frozen tape window
    /// copies from (see [`MoePagerSession::lut_window`]).
    fn lut_words(&self, base: usize, n: usize) -> &[u32] {
        &self.lut_host[base..base + n]
    }

    /// Mirror one miss's placement into the host LUT and mark it dirty — the shared
    /// eviction-then-insert bookkeeping formerly triplicated across [`Self::touch_staged`],
    /// [`Self::schedule_staged`] and [`Self::ensure_resident`]. Byte-for-byte the same writes those
    /// inline blocks made (see [`apply_placement`]); the one place a wrong LUT entry becomes
    /// silent-zero MoE output, so it lives in exactly one function now.
    fn record_placement(&mut self, id: BlockId, slot: u32, evicted: Option<u32>) {
        apply_placement(&mut self.lut_host, id, slot, evicted);
        self.lut_dirty = true;
    }

    /// [`Self::touch_staged`]'s DENSE-STREAMING twin: residency via the exact cyclic-sweep policy
    /// (`infr_core::pager::Pager::schedule` — dense layer order is deterministic, so every miss
    /// is known in advance and no LUT/readback machinery is involved) and the block's bytes given
    /// as SEGMENTS (a fused qkv/gate_up block keeps its component tensors' zero-copy mmap slices;
    /// materializing the concat would double the streamed model's host RAM). Returns
    /// `(slot, ring_bytes_consumed)` — 0 consumed on a hit; a miss memcpys the segments
    /// back-to-back into the ring at `ring_off` and records the ring→arena slot copy, exactly
    /// like `touch_staged` (same ring-region-lifetime contract). The segments' total may be up to
    /// `slot_bytes - 3` short of the slot (the stride is padded to the pool's block/word
    /// alignment); the pad tail is never read by a dispatch (every kernel read stays within the
    /// block's `numel`). The caller must have verified the current ring half fits `slot_bytes`
    /// BEFORE calling (a miss here always consumes a full slot stride of ring accounting).
    ///
    /// The host LUT mirror is kept coherent (eviction/insert) so a pager can't be silently
    /// half-adopted by a LUT-reading path later, but dense dispatch never reads it — the slot
    /// index returned here is baked into the dispatch's weight element offset instead.
    ///
    /// `host` is the pool's tier below, required exactly when `bytes` is [`DenseBytes::Host`]. That
    /// is the case that makes this three-tiered: a VRAM miss resolves against DRAM, which either
    /// hits (a memcpy out of its arena) or reads the model file into it first.
    pub fn schedule_staged(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        ring: &dyn Buffer,
        ring_off: usize,
        id: BlockId,
        bytes: &DenseBytes,
        host: Option<&HostPager>,
    ) -> Result<(u32, usize)> {
        let prof = pager_profile::active();
        let lookup_t0 = prof.then(std::time::Instant::now);
        let resolution = self.pager.schedule(id);
        if let Some(t0) = lookup_t0 {
            let (hit, evicted) = match resolution {
                Resolution::Hit { .. } => (true, false),
                Resolution::Miss { evicted, .. } => (false, evicted.is_some()),
            };
            pager_profile::record_gpu_cache_lookup(hit, evicted, t0.elapsed());
        }
        match resolution {
            Resolution::Hit { slot } => Ok((slot, 0)),
            Resolution::Miss { slot, evicted } => {
                let slot_bytes = self.slot_bytes;
                let fits = |total: usize| {
                    debug_assert!(
                        total <= slot_bytes,
                        "dense block bytes ({total}) exceed the pool's slot stride ({slot_bytes})"
                    );
                };
                let base = as_vk_buf(ring)?
                    .mapped_ptr()
                    .ok_or_else(|| be("pager staging ring is not persistently mapped"))?;
                let total = match bytes {
                    DenseBytes::Mmap(segments) => {
                        let total: usize = segments.iter().map(|s| expert_bytes(s).len()).sum();
                        fits(total);
                        let copy_t0 = prof.then(std::time::Instant::now);
                        let mut off = ring_off;
                        for s in segments {
                            let seg = expert_bytes(s);
                            par_copy_to_mapped(seg, unsafe { base.add(off) });
                            off += seg.len();
                        }
                        if let Some(t0) = copy_t0 {
                            let elapsed = t0.elapsed();
                            pager_profile::record_memcpy(total, elapsed);
                            pager_profile::record_mmap_fallback(total, elapsed);
                        }
                        total
                    }
                    DenseBytes::Host => {
                        let host = host.ok_or_else(|| {
                            be(format!(
                                "dense pager: block {id} has no host tier to read from"
                            ))
                        })?;
                        let n = host.block_bytes(id).ok_or_else(|| {
                            be(format!(
                                "dense pager: block {id} is unknown to the host tier"
                            ))
                        })?;
                        fits(n);
                        // Delivered STRAIGHT into the ring. `HostPager::fill` copies out of its
                        // arena on a hit and reads into a free slot while one remains, but once the
                        // arena is full it reads the block into this buffer directly — one copy on
                        // the streaming majority instead of two, which is the cost
                        // `docs/perf/results.md` measured the pin-then-memcpy shape paying.
                        //
                        // SAFETY: `[ring_off, ring_off + n)` is this caller's own region of the
                        // persistently-mapped ring — reserved by the cursor before this call, and
                        // not reused until the recording that reads it completes (the fn's ring
                        // contract). No other thread can hold a reference to it.
                        let dst = unsafe { std::slice::from_raw_parts_mut(base.add(ring_off), n) };
                        host.fill(id, dst)?;
                        n
                    }
                };
                // Word-align the copy length (the ring pad bytes it may carry are never read —
                // see the fn doc); `total <= slot_bytes` and `slot_bytes % 4 == 0` keep it in
                // the slot.
                rec.copy(
                    ring,
                    ring_off,
                    self.arena.as_ref(),
                    self.arena_offset + slot as usize * self.slot_bytes,
                    total.next_multiple_of(4),
                );
                if prof {
                    pager_profile::record_gpu_copy(total.next_multiple_of(4));
                }
                self.record_placement(id, slot, evicted);
                Ok((slot, self.slot_bytes))
            }
        }
    }

    /// Open a touch batch — see `infr_core::pager::Pager::begin_batch`. One batch = one
    /// (layer, role) residency resolution; blocks it touches are eviction-protected until the
    /// next batch opens.
    pub fn begin_batch(&mut self) {
        self.pager.begin_batch();
    }

    /// Ensure `id` is resident, uploading `bytes` (exactly `slot_bytes`) through `staging` if it's
    /// a miss. Updates the HOST lut mirror immediately; the device copy is deferred to
    /// [`flush_lut`](Self::flush_lut) so a caller resolving several ids for one batch (see
    /// `infr_core::pager`'s within-batch note, which applies here unchanged) pays for exactly one
    /// LUT upload per batch, not one per id.
    pub fn ensure_resident(
        &mut self,
        vk: &VulkanBackend,
        staging: &dyn Buffer,
        id: BlockId,
        bytes: &[u8],
    ) -> Result<u32> {
        debug_assert_eq!(
            bytes.len(),
            self.slot_bytes,
            "block byte size must match the arena's slot size"
        );
        let prof = pager_profile::active();
        let lookup_t0 = prof.then(std::time::Instant::now);
        let resolution = self.pager.touch(id);
        if let Some(t0) = lookup_t0 {
            let (hit, evicted) = match resolution {
                Resolution::Hit { .. } => (true, false),
                Resolution::Miss { evicted, .. } => (false, evicted.is_some()),
            };
            pager_profile::record_gpu_cache_lookup(hit, evicted, t0.elapsed());
        }
        match resolution {
            Resolution::Hit { slot } => Ok(slot),
            Resolution::Miss { slot, evicted } => {
                let upload_t0 = prof.then(std::time::Instant::now);
                vk.upload(staging, bytes)?;
                if let Some(t0) = upload_t0 {
                    pager_profile::record_memcpy(bytes.len(), t0.elapsed());
                }
                let sync_t0 = prof.then(std::time::Instant::now);
                copy_into_slot(
                    vk,
                    staging,
                    self.arena.as_ref(),
                    self.arena_offset + slot as usize * self.slot_bytes,
                    self.slot_bytes,
                )?;
                if let Some(t0) = sync_t0 {
                    pager_profile::record_gpu_copy(self.slot_bytes);
                    pager_profile::record_paging_sync_wait(t0.elapsed());
                }
                self.record_placement(id, slot, evicted);
                Ok(slot)
            }
        }
    }

    /// Push the host LUT mirror to the device if anything changed since the last flush. Callers
    /// resolving a whole batch of ids must call this exactly once, AFTER every `ensure_resident`
    /// for that batch and BEFORE recording any dispatch that reads the LUT — the within-batch
    /// eviction-safety argument on `infr_core::pager::Pager` only holds if the LUT a dispatch
    /// reads reflects EVERY id that batch touched, not a partial prefix.
    pub fn flush_lut(&mut self, vk: &VulkanBackend) -> Result<()> {
        if self.lut_dirty {
            vk.upload(self.lut_dev.as_ref(), bytemuck::cast_slice(&self.lut_host))?;
            self.lut_dirty = false;
        }
        Ok(())
    }
}

/// Parallel memcpy of one expert's bytes into the mapped staging ring. The single-thread copy is
/// the staging bottleneck (the bandwidth probe's 22 GB/s is a hot-source best case; streaming
/// distinct experts out of a 37 GB page-cache-backed mmap into write-combined ReBAR runs well
/// below that) — chunked `copy_nonoverlapping` across the rayon pool recovers most of the
/// PCIe/DRAM headroom. 4 MiB chunks: big enough for streaming stores, small enough to spread a
/// 14-18 MB expert across several workers.
fn par_copy_to_mapped(src: &[u8], dst: *mut u8) {
    use rayon::prelude::*;
    const CHUNK: usize = 4 << 20;
    if src.len() <= CHUNK {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let dst_addr = dst as usize; // Send-able; each chunk writes a disjoint range
    src.par_chunks(CHUNK).enumerate().for_each(|(i, c)| unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), (dst_addr + i * CHUNK) as *mut u8, c.len());
    });
}

#[derive(Clone, Copy)]
struct StagingCopy {
    src: usize,
    dst: usize,
    len: usize,
}

/// Copy independent expert blocks with one rayon entry per staged layer/role. Parallelizing
/// *inside* each sub-MiB expert costs more scheduler time than it saves; batching experts exposes
/// tens to hundreds of naturally independent, coarse jobs at once.
fn run_staging_copies(copies: &[StagingCopy]) {
    use rayon::prelude::*;
    copies.par_iter().for_each(|job| unsafe {
        std::ptr::copy_nonoverlapping(job.src as *const u8, job.dst as *mut u8, job.len);
    });
}

/// Device-to-device copy of `len` bytes from `src[0..len]` into `dst[slot*len .. (slot+1)*len]` —
/// the pager's slot placement, which the shared `Backend::copy_buffer` can't express (it always
/// copies `[0, bytes)` on both sides). Internal to this crate: raw `ash` calls mirroring
/// `VulkanBackend::upload`'s device-copy branch exactly, just with a nonzero destination offset.
fn copy_into_slot(
    vk: &VulkanBackend,
    src: &dyn Buffer,
    dst: &dyn Buffer,
    dst_offset: usize,
    len: usize,
) -> Result<()> {
    let (s, d) = (as_vk_buf(src)?, as_vk_buf(dst)?);
    let (sb, db) = (s.buffer, d.buffer);
    let dst_offset = dst_offset as u64;
    let shared = Arc::clone(&vk.shared);
    vk.one_shot(move |cmd| unsafe {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset,
            size: len as u64,
        };
        shared.device.cmd_copy_buffer(cmd, sb, db, &[region]);
    })
}

// ─── MoE expert-bank paging session (slice 2: wiring into the execution path) ─────────────────
//
// The pieces above are the block-agnostic host<->VRAM cache; everything below is the MoE-specific
// glue: one [`GpuPager`] POOL per (expert role, per-expert byte size) pair, a table mapping a
// bound weight BUFFER's identity to its offset in the unique CPU-only layer-major store, and the
// layer-granular Prefill interpretation of the same VRAM arena.
//
// Why (role, slot_bytes) pools and not one pager per role: the arena/LUT design requires every
// block sharing an arena to have the SAME byte size (fixed slot offsets + a word-base LUT), and
// the GEMV/GEMM kernels additionally assume the layer's dtype when decoding a slot's bytes. Two
// shapes break a naive per-role pager:
//   - MIXED-dtype roles: unsloth-dynamic (UD) quants bump a SUBSET of layers' banks to a wider
//     format for quality (gemma-4-MoE: down = Q5_1 on 29 layers + Q8_0 on 1; DiffusionGemma:
//     down = Q5_0/Q8_0 16/14; Qwen3.6-UD: down mixes Q4_K/Q6_K). Slot sizes differ per dtype, so
//     one arena can't hold both — but a pool PER byte-size can: each layer registers into the
//     pool matching its own per-expert byte size, and a dispatch only ever reads ids of ONE
//     layer (whose dtype it knows statically from the graph), so blocks of different dtypes that
//     happen to share a byte size may even share a pool safely.
//   - FUSED gate_up banks (gemma-4 MoE / DiffusionGemma `ffn_gate_up_exps`): a fused expert is
//     just a BIGGER uniform block ([ne, 2*n_ff_exp] instead of [ne, n_ff_exp]) — it pages under
//     `Role::Gate` with its own slot size, and the model simply has no `Role::Up` pool.
// Every pool shares the same GLOBAL block-id space (`layer_index * n_expert + local_id`), so the
// paged kernels' `lut[layer_base + expert]` hop is unchanged — a pool's LUT just holds
// NOT_RESIDENT for the layers that live in other pools (they are never asked for).
//
// Design note (see the task doc): `Op::MoeFfn` carries NO `paged` flag. A paged layer's graph is
// byte-for-byte the same shape as a resident one (same tensor roles, same op) — only the ACTUAL
// buffer bound at `gate_exps`/`up_exps`/`down_exps` differs (a tiny placeholder vs the full
// upload). Threading a per-layer paging flag through `generate_dense_backend` (~20 parameters, 16
// call sites shared by CPU/Vulkan/Metal) to recompute at every graph-build call is a much bigger,
// riskier diff than keying off the buffer ACTUALLY bound at execute time — which the adapter
// already has in hand via `Bindings`. So the placement decision lives entirely on this side: the
// seam registers each paged layer's source bytes once at weight-load time, keyed by the stable
// identity of the (tiny, otherwise-unread) placeholder buffer it bound in place of a real upload;
// `execute_static` looks up that identity when it meets a `MoeFfn` op, and only diverts to the
// segmented paged path on a hit. CPU and Metal never call any of this — zero changes there.
use std::sync::Mutex;

/// One paged expert role. A FUSED gate_up bank registers under `Gate` (see the module-section doc
/// above); a fused model simply has no `Up` sources. Roles with mixed per-expert byte sizes
/// across layers span several pools — the (role, slot_bytes) pair, not the role alone, names a
/// pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    Gate,
    Up,
    Down,
}

impl Role {
    fn name(self) -> &'static str {
        match self {
            Role::Gate => "gate",
            Role::Up => "up",
            Role::Down => "down",
        }
    }
}

/// Stable identity of a bound `&dyn Buffer` — a thin-pointer cast of the trait object's data
/// pointer, which Box/heap allocation guarantees stable for the buffer's whole lifetime (the
/// model's `SeamWeights::wbufs` never reallocates the Boxes themselves once loaded, only the Vec
/// that briefly held them during construction). Used to recognize "the SAME placeholder buffer
/// bound at this TensorId, across however many differently-shaped Graphs reuse it" without
/// depending on `TensorId` staying numerically stable across graphs (it doesn't — see the module
/// doc's design note).
pub fn buffer_identity(b: &dyn Buffer) -> usize {
    std::ptr::from_ref(b) as *const () as usize
}

/// One expert/segment source's bytes. `Arc<T>` itself implements `AsRef<T>`, so a bare
/// `arc.as_ref()` would resolve to THAT (returning the fat `&(dyn AsRef<[u8]> + Send + Sync)`)
/// instead of the inner `AsRef<[u8]>::as_ref` every caller needs — force the deref-to-trait-object
/// FIRST so only the trait object's own impl is a candidate. Factored so every call site shares
/// this one guarded deref (a copy that omits it compiles but resolves wrong).
fn expert_bytes(arc: &Arc<dyn AsRef<[u8]> + Send + Sync>) -> &[u8] {
    let inner: &(dyn AsRef<[u8]> + Send + Sync) = &**arc;
    inner.as_ref()
}

/// The shared `hits/misses/evictions/hit_rate` fragment of both sessions' `paging.stats` lines
/// (each session prepends its own label + slot size and appends its own slot-count suffix).
fn stats_suffix(s: &PagerStats) -> String {
    format!(
        "hits={} misses={} evictions={} hit_rate={:.3}",
        s.hits,
        s.misses,
        s.evictions,
        s.hit_rate(),
    )
}

/// Load-time description of one paged layer's per-role expert bank. `register` copies this bank
/// once into the session's CPU-only layer-major store, then drops the `Arc`; runtime paths retain
/// only offsets and never pin, repack, or reread the GGUF mapping.
pub struct ExpertSource {
    pub bank: Arc<dyn AsRef<[u8]> + Send + Sync>,
    pub stride_bytes: usize,
    /// This layer's offset into the role's shared LUT/arena block-id space
    /// (`layer_index * n_expert`) — turns a per-layer LOCAL expert id (what the router/top-k
    /// produces, `0..n_expert`) into a GLOBAL `BlockId` unique across every paged layer of this
    /// role, so one `Pager`/LUT can hold experts from many layers at once.
    pub layer_base: u32,
    /// Byte offset assigned by the seam's layer-major permanent host-store plan.
    pub host_offset: usize,
}

/// Runtime metadata for one bank after [`ExpertSource::bank`] has been copied into the permanent
/// host store. Keeping this separate is the ownership guarantee behind the single-copy design:
/// the session cannot accidentally retain the model mapping as a second runtime weight source.
#[derive(Clone, Copy, Debug)]
struct RegisteredExpertSource {
    stride_bytes: usize,
    layer_base: u32,
    host_chunk: usize,
    host_offset: usize,
    bank_bytes: usize,
}

struct HostStoreChunk {
    base_offset: usize,
    /// Ordinary CPU-owned memory. It has no VkBuffer, device address, GPU VA, shared-VRAM
    /// accounting, or second staging allocation. Chunks only avoid one enormous virtual address
    /// reservation; together they are the sole owned copy of the complete expert payload.
    bytes: Box<[u8]>,
}

/// One arena pool: every block in it shares `slot_bytes` (see the section doc above for why the
/// pool key is `(role, slot_bytes)`, not the role alone).
struct Pool {
    role: Role,
    slot_bytes: usize,
    pager: GpuPager,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoeArenaMode {
    DecodeLru,
    PrefillLayer,
}

#[derive(Clone, Copy, Debug)]
struct PrefillPlacement {
    /// Direct byte address of this role bank inside the shared arena. Prefill kernels add their
    /// identity expert id times the role's expert stride to this base; the decode pool ranges and
    /// their LUTs are deliberately bypassed.
    byte_offset: usize,
    pool: usize,
    /// `None` for a resident layer, A=0/B=1 for a streamed layer.
    lane: Option<usize>,
    layer_base: u32,
}

#[derive(Debug)]
struct PrefillLayerPlacement {
    layer_base: u32,
    banks: Vec<usize>,
}

const PREFILL_BANK_ALIGN: usize = 256;

#[inline]
fn prefill_align(bytes: usize) -> usize {
    bytes.next_multiple_of(PREFILL_BANK_ALIGN)
}

/// One model's whole paged-MoE session: the `(role, slot_bytes)` arena pools plus the permanent
/// CPU-only layer-major source. Lives on the `VulkanBackend` HANDLE
/// (NOT `VulkanShared` — the session's buffers hold `Arc<VulkanShared>` clones, and parking it on
/// the shared state made an Arc cycle that leaked the device's whole VRAM footprint until process
/// exit; see the `moe_pager` field doc in lib.rs) for as long as the backend that loaded the
/// paged model lives (`VulkanBackend::init_moe_pager`); `None` for every non-paged model — zero
/// cost, zero behavior change on the common (fits-in-VRAM) path.
pub struct MoePagerSession {
    pools: Vec<Pool>,
    /// The only owned host copy of all paged MoE weights. Plain CPU memory, deliberately not a
    /// Vulkan buffer: the full payload cannot be counted or accessed as shared/virtual VRAM.
    host_store: Vec<HostStoreChunk>,
    /// `buffer_identity(placeholder)` -> (role, pool index, this layer's expert source), for
    /// every PAGED `_exps` tensor. A non-paged layer's gate/up/down buffer is never registered
    /// here — the adapter's lookup simply misses and falls through to the ordinary
    /// resident-weight path.
    sources: HashMap<usize, (Role, usize, RegisteredExpertSource)>,
    /// LUT tape: an append-only run of frozen per-(layer, role) LUT windows (`n_expert` u32 slot
    /// indices each, written by [`Self::lut_window`]). Dispatches read `tape[window + local_id]`
    /// instead of the live pool LUT, so host-side staging for LATER layers can keep mutating the
    /// mirror while EARLIER layers' recorded-but-in-flight dispatches still read a consistent
    /// view — the in-flight-LUT rule that a single mutable device LUT cannot satisfy once
    /// several layers record into one submission. The cursor is the adapter's (reset only after
    /// a full drain).
    tape: Box<dyn Buffer>,
    tape_words: usize,
    print_stats: bool,
    /// Physical interpretation of every pool arena. Prefill owns slot 0..n_expert as one
    /// contiguous layer bank; decode restores the ordinary expert-LRU interpretation.
    mode: MoeArenaMode,
    /// Per pool, the complete bank currently occupying slots `0..n_expert` in layer-prefill
    /// mode. Layer-major prefill invokes the same layer once per chunk; those later chunks must
    /// reuse the first upload instead of streaming the bank again.
    prefill_lane_layer: [Option<u32>; 2],
    /// Per registered bank, its whole-layer placement inside the pool arena. Every complete
    /// layer slots except the final A/B pair are fixed cache entries; all later banks alternate
    /// over that pair. Decode ignores this map and uses every expert slot through its LRU.
    prefill_placement: HashMap<usize, PrefillPlacement>,
    prefill_layers: Vec<PrefillLayerPlacement>,
    prefill_loaded: HashSet<usize>,
    prefill_max_streamed_layer_bytes: usize,
}

/// One pool's spec in [`MoePagerLayout`]: slot counts are INDEPENDENT per pool. Each pool's arena
/// is a `bufferDeviceAddress` buffer (`48ad9c1`) addressed by 64-bit pointer — no per-arena
/// `maxStorageBufferRange` ceiling — but per-pool sizing
/// still matters because of unequal per-expert sizes (Scout: gate/up 13.8 MB, down 18 MB): a
/// shared slot count is dragged down to fit the LARGEST pool's per-slot bytes within the VRAM
/// budget and strands budget the smaller pools could have used as real hit rate (Scout: uniform
/// 238 slots everywhere left ~6 GB of a 19 GB budget unused; per-pool sizing gives gate/up 312
/// each). Each pool has its own LRU/LUT and `push_role_cpu` resolves pools independently, so
/// unequal counts are correctness-neutral — a pool with fewer slots just misses more often. Computed by
/// the caller (budget-driven count, then per-pool split — see `seam::mod`'s placement policy).
pub struct MoePoolSpec {
    pub role: Role,
    pub slot_bytes: usize,
    pub n_slots: usize,
}

/// One chunk of the unique permanent CPU store. Chunks are split only at complete layer boundaries
/// to avoid one enormous host allocation without ever splitting a Prefill layer push.
pub struct MoeHostChunkSpec {
    pub base_offset: usize,
    pub bytes: usize,
}

/// Fixed layout for [`MoePagerSession::new`] — sizes every arena/LUT UP FRONT, before any tensor
/// is registered. This split (layout now, registration per tensor later) matters for sequencing:
/// the session must exist and answer `is_paged`/`Backend::moe_paged` truthy BEFORE the seam's
/// weight-load closure runs (so a paged tensor's placeholder buffer is recognized the very first
/// time the adapter executes a graph, not just after the whole model is loaded) — see
/// `infr-llama`'s `generate_dense_vulkan_session` for the call order this enables.
pub struct MoePagerLayout {
    /// Total distinct experts nameable per pool's LUT = `n_paged_layers * n_expert` — the GLOBAL
    /// id space every pool shares (a pool only ever resolves ids of the layers registered into
    /// it; other layers' entries stay `NOT_RESIDENT`).
    pub n_blocks: usize,
    pub pools: Vec<MoePoolSpec>,
    /// Non-overlapping layer-boundary chunks covering the exact layer-major host-store extent.
    pub host_chunks: Vec<MoeHostChunkSpec>,
}

/// Upload-ring sizing policy — pure budget arithmetic, so it lives in the shared seam
/// ([`infr_core::pager::ring_bytes`], which owns the doc and the boundary tests). Re-exported
/// under this crate's old path because the ring it sizes is a Vulkan buffer pair and every call
/// site here reads better next to them. The `paging.ring` override comes off the backend's
/// `Config` (`INFR_PAGER_RING`), so the caller passes it in.
pub use infr_core::pager::ring_bytes;

// Vulkan permits 4-byte buffer-copy offsets, but 256 bytes also satisfies common
// `optimalBufferCopyOffsetAlignment` values. More importantly, it prevents a configured ring
// whose byte count is not evenly divisible by its slot count from giving every later region a
// pathologically misaligned base address.
const RING_REGION_ALIGN: usize = 256;

fn ring_region_bytes(total: usize, slots: usize, min_slot_bytes: usize) -> usize {
    debug_assert!(slots >= 2);
    let fair_share = total / slots;
    let aligned_share = fair_share / RING_REGION_ALIGN * RING_REGION_ALIGN;
    let aligned_min = min_slot_bytes.div_ceil(RING_REGION_ALIGN) * RING_REGION_ALIGN;
    aligned_share.max(aligned_min)
}

impl MoePagerSession {
    pub fn new(vk: &VulkanBackend, layout: MoePagerLayout) -> Result<Self> {
        if layout.host_chunks.is_empty() {
            return Err(be("moe pager: permanent host-store plan has no chunks"));
        }
        let mut host_store = Vec::with_capacity(layout.host_chunks.len());
        let mut previous_end = 0usize;
        for spec in &layout.host_chunks {
            if spec.bytes == 0 || spec.base_offset < previous_end {
                return Err(be("moe pager: invalid or overlapping host-store chunk"));
            }
            let end = spec
                .base_offset
                .checked_add(spec.bytes)
                .ok_or_else(|| be("moe pager: host-store chunk range overflow"))?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(spec.bytes).map_err(|e| {
                be(format!(
                    "moe pager: cannot reserve {} bytes for the unique CPU expert store: {e}",
                    spec.bytes,
                ))
            })?;
            bytes.resize(spec.bytes, 0);
            host_store.push(HostStoreChunk {
                base_offset: spec.base_offset,
                bytes: bytes.into_boxed_slice(),
            });
            previous_end = end;
        }
        let host_payload_bytes: usize = host_store.iter().map(|chunk| chunk.bytes.len()).sum();
        tracing::info!(
            "[infr] paged-MoE host store: {} bytes in {} CPU-only layer chunks; GPU-visible host payload = 0 bytes",
            host_payload_bytes,
            host_store.len(),
        );
        let mut pools = Vec::with_capacity(layout.pools.len());
        for spec in &layout.pools {
            pools.push(Pool {
                role: spec.role,
                slot_bytes: spec.slot_bytes,
                // MoE pools are pointer-addressed (`bufferDeviceAddress`) — no per-arena SSBO cap.
                pager: GpuPager::new_mapped(vk, layout.n_blocks, spec.n_slots, spec.slot_bytes)?,
            });
        }
        // One graph's windows = paged layers x roles x n_expert words (Scout: 48 x 3 x 16 = 2.3k)
        // — 64k words (256 KiB) leaves an order of magnitude of headroom; `lut_window` hard-errors
        // on overflow rather than wrapping into a region an in-flight segment may still read.
        let tape_words = 64 * 1024;
        let tape = vk.alloc_uninit(tape_words * 4, BufferUsage::Staging)?;
        Ok(Self {
            pools,
            host_store,
            sources: HashMap::new(),
            tape,
            tape_words,
            print_stats: vk.cfg().paging.stats,
            mode: MoeArenaMode::DecodeLru,
            prefill_lane_layer: [None, None],
            prefill_placement: HashMap::new(),
            prefill_layers: Vec::new(),
            prefill_loaded: HashSet::new(),
            prefill_max_streamed_layer_bytes: 0,
        })
    }

    /// Register one paged layer's `role` tensor — called from the seam's weight-load closure
    /// (once per paged `_exps` tensor) instead of uploading it. `buf_id` is the placeholder
    /// buffer's identity (see [`buffer_identity`]); `source` is where its bytes actually live.
    /// The pool is picked by `(role, source.stride_bytes)` — errors if the layout has no matching
    /// pool (a seam sizing bug: the layout enumeration and this registration must derive the slot
    /// size from the same tensor bytes).
    ///
    /// `n_expert` is how many experts this layer's bank holds, checked against the tier below when
    /// there is one: every one of the layer's blocks must already be registered there, or a routed
    /// id would fail only later, mid-generation, on the first miss that names it.
    pub fn register(
        &mut self,
        role: Role,
        buf_id: usize,
        source: ExpertSource,
        n_expert: usize,
    ) -> Result<()> {
        let pool = self
            .pools
            .iter()
            .position(|p| p.role == role && p.slot_bytes == source.stride_bytes)
            .ok_or_else(|| {
                be(format!(
                    "moe pager: no ({:?}, {} B/expert) pool in the layout for this tensor",
                    role, source.stride_bytes,
                ))
            })?;
        let bank = expert_bytes(&source.bank);
        let expected = source
            .stride_bytes
            .checked_mul(n_expert)
            .ok_or_else(|| be("moe pager: expert bank byte size overflow"))?;
        if bank.len() != expected {
            return Err(be(format!(
                "moe pager: bank is {} bytes, expected {n_expert} x {} = {expected}",
                bank.len(),
                source.stride_bytes,
            )));
        }
        let end = source
            .host_offset
            .checked_add(bank.len())
            .ok_or_else(|| be("moe pager: host-store offset overflow"))?;
        let (host_chunk, chunk) = self
            .host_store
            .iter_mut()
            .enumerate()
            .find(|(_, chunk)| {
                source.host_offset >= chunk.base_offset
                    && end <= chunk.base_offset + chunk.bytes.len()
            })
            .ok_or_else(|| {
                be(format!(
                    "moe pager: host-store bank range {}..{end} crosses or exceeds a chunk",
                    source.host_offset,
                ))
            })?;
        let chunk_offset = source.host_offset - chunk.base_offset;
        let copy_t0 = pager_profile::active().then(std::time::Instant::now);
        chunk.bytes[chunk_offset..chunk_offset + bank.len()].copy_from_slice(bank);
        if let Some(t0) = copy_t0 {
            pager_profile::record_memcpy(bank.len(), t0.elapsed());
        }
        self.sources.insert(
            buf_id,
            (
                role,
                pool,
                RegisteredExpertSource {
                    stride_bytes: source.stride_bytes,
                    layer_base: source.layer_base,
                    host_chunk,
                    host_offset: chunk_offset,
                    bank_bytes: bank.len(),
                },
            ),
        );
        // Registration happens only during model load, before execution. Keep this defensive so
        // a future loader that registers incrementally can never retain a layout built from an
        // incomplete source set.
        self.prefill_placement.clear();
        self.prefill_layers.clear();
        self.prefill_loaded.clear();
        self.prefill_lane_layer = [None, None];
        self.prefill_max_streamed_layer_bytes = 0;
        Ok(())
    }

    /// Whether `buf_id` (see [`buffer_identity`]) is a registered paged tensor of `role` — the
    /// adapter's per-`MoeFfn` dispatch check.
    pub fn is_paged(&self, role: Role, buf_id: usize) -> bool {
        self.sources.get(&buf_id).is_some_and(|(r, ..)| *r == role)
    }

    pub fn bank_bytes(&self, buf_id: usize) -> Result<usize> {
        let (_, _, src) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: bank size on an unregistered buffer"))?;
        Ok(src.bank_bytes)
    }

    /// Switch the shared arenas back to expert-LRU interpretation. A preceding prefill may have
    /// overwritten arbitrary LRU slots with contiguous whole-layer banks, so none of the old
    /// mappings are valid. The caller only invokes this at a new static execute boundary; the
    /// previous execute drained the queue before returning.
    pub fn enter_decode(&mut self) -> bool {
        if self.mode == MoeArenaMode::DecodeLru {
            return false;
        }
        for pool in &mut self.pools {
            pool.pager.reset_residency();
        }
        self.prefill_lane_layer = [None, None];
        self.prefill_loaded.clear();
        self.mode = MoeArenaMode::DecodeLru;
        true
    }

    fn build_prefill_layout(&mut self) -> Result<()> {
        if !self.prefill_placement.is_empty() {
            return Ok(());
        }
        let mut grouped: BTreeMap<u32, Vec<(u8, usize, usize, usize, usize, usize)>> =
            BTreeMap::new();
        for (&buf_id, (role, pool, src)) in &self.sources {
            let role_order = match role {
                Role::Gate => 0,
                Role::Up => 1,
                Role::Down => 2,
            };
            grouped.entry(src.layer_base).or_default().push((
                role_order,
                buf_id,
                *pool,
                src.bank_bytes,
                src.host_chunk,
                src.host_offset,
            ));
        }
        if grouped.is_empty() {
            return Err(be(
                "moe pager: cannot build a prefill layout without expert banks",
            ));
        }

        let mut layers: Vec<(u32, Vec<(usize, usize, usize, usize)>, usize)> = Vec::new();
        for (layer_base, mut banks) in grouped {
            banks.sort_unstable_by_key(|&(role, buf_id, _, _, _, _)| (role, buf_id));
            let mut offset = 0usize;
            let mut packed = Vec::with_capacity(banks.len());
            let (layer_host_chunk, layer_host_offset) = banks
                .first()
                .map(|&(_, _, _, _, host_chunk, host_offset)| (host_chunk, host_offset))
                .ok_or_else(|| be("moe pager: empty expert layer in host-store plan"))?;
            for (_, buf_id, pool, bytes, host_chunk, host_offset) in banks {
                offset = prefill_align(offset);
                if host_chunk != layer_host_chunk {
                    return Err(be(format!(
                        "moe pager: layer {layer_base} crosses permanent host-store chunks"
                    )));
                }
                let expected = layer_host_offset
                    .checked_add(offset)
                    .ok_or_else(|| be("moe pager: prefill layer host-store offset overflow"))?;
                if host_offset != expected {
                    return Err(be(format!(
                        "moe pager: layer {layer_base} bank host offset {host_offset} is not \
                         contiguous (expected {expected})"
                    )));
                }
                packed.push((buf_id, pool, offset, bytes));
                offset = offset
                    .checked_add(bytes)
                    .ok_or_else(|| be("moe pager: prefill layer byte size overflow"))?;
            }
            layers.push((layer_base, packed, prefill_align(offset)));
        }

        // Windows AMD rejects one mapped allocation spanning the whole logical cache, so each
        // existing (role, expert-size) pool is its own ReBAR allocation. Reserve two complete
        // BANK lanes in every pool, then choose only whole LAYERS whose every bank fits the
        // remaining resident capacity. A streamed layer uses the same A/B lane in all its pools.
        let mut pool_bank_bytes = vec![0usize; self.pools.len()];
        let mut pool_layer_counts = vec![0usize; self.pools.len()];
        for (_, banks, _) in &layers {
            for &(_, pool, _, bytes) in banks {
                if pool_bank_bytes[pool] != 0 && pool_bank_bytes[pool] != bytes {
                    return Err(be(
                        "moe pager: one pool contains unequal whole-bank byte sizes",
                    ));
                }
                pool_bank_bytes[pool] = bytes;
                pool_layer_counts[pool] += 1;
            }
        }
        let mut resident_capacity = vec![0usize; self.pools.len()];
        let mut stream_base = vec![0usize; self.pools.len()];
        let mut total_arena_bytes = 0usize;
        let mut total_ab_bytes = 0usize;
        for (pool_idx, pool) in self.pools.iter().enumerate() {
            let arena_bytes = pool
                .pager
                .n_slots()
                .checked_mul(pool.slot_bytes)
                .ok_or_else(|| be("moe pager: pool arena byte size overflow"))?;
            total_arena_bytes = total_arena_bytes.saturating_add(arena_bytes);
            let bank_bytes = pool_bank_bytes[pool_idx];
            if bank_bytes == 0 {
                continue;
            }
            let whole_banks = arena_bytes / bank_bytes;
            if whole_banks < 2 {
                return Err(be(format!(
                    "moe pager: pool {pool_idx} ({arena_bytes} bytes) cannot hold two {bank_bytes}-byte Prefill lanes"
                )));
            }
            resident_capacity[pool_idx] = whole_banks
                .saturating_sub(2)
                .min(pool_layer_counts[pool_idx]);
            stream_base[pool_idx] = arena_bytes - 2 * bank_bytes;
            total_ab_bytes = total_ab_bytes.saturating_add(2 * bank_bytes);
        }

        let mut resident = vec![false; layers.len()];
        let mut resident_used = vec![0usize; self.pools.len()];
        let mut order: Vec<usize> = (0..layers.len()).collect();
        order.sort_unstable_by_key(|&idx| std::cmp::Reverse(layers[idx].2));
        for idx in order {
            let mut need = vec![0usize; self.pools.len()];
            for &(_, pool, _, _) in &layers[idx].1 {
                need[pool] += 1;
            }
            if need
                .iter()
                .enumerate()
                .all(|(pool, &n)| resident_used[pool].saturating_add(n) <= resident_capacity[pool])
            {
                resident[idx] = true;
                for (pool, n) in need.into_iter().enumerate() {
                    resident_used[pool] += n;
                }
            }
        }

        let mut resident_cursor = vec![0usize; self.pools.len()];
        let mut streamed_ordinal = 0usize;
        let mut resident_bytes = 0usize;
        let mut max_streamed = 0usize;
        for (idx, (layer_base, banks, layer_bytes)) in layers.into_iter().enumerate() {
            let fixed = resident[idx];
            let lane = if fixed {
                resident_bytes += layer_bytes;
                None
            } else {
                let lane = streamed_ordinal % 2;
                streamed_ordinal += 1;
                max_streamed = max_streamed.max(layer_bytes);
                Some(lane)
            };
            let mut bank_ids = Vec::with_capacity(banks.len());
            for (buf_id, pool, _host_bank_offset, bank_bytes) in banks {
                let byte_offset = if fixed {
                    let off = resident_cursor[pool];
                    resident_cursor[pool] = resident_cursor[pool]
                        .checked_add(bank_bytes)
                        .ok_or_else(|| be("moe pager: resident bank offset overflow"))?;
                    if resident_cursor[pool] > stream_base[pool] {
                        return Err(be("moe pager: resident banks overlap a Prefill A/B lane"));
                    }
                    off
                } else {
                    stream_base[pool] + lane.expect("streamed layer has a lane") * bank_bytes
                };
                self.prefill_placement.insert(
                    buf_id,
                    PrefillPlacement {
                        byte_offset,
                        pool,
                        lane,
                        layer_base,
                    },
                );
                bank_ids.push(buf_id);
            }
            self.prefill_layers.push(PrefillLayerPlacement {
                layer_base,
                banks: bank_ids,
            });
        }
        self.prefill_max_streamed_layer_bytes = max_streamed;
        tracing::info!(
            "[moe-prefill] rebar_pool_arenas={} resident_layers={}/{} resident_bytes={} streamed_layer_max={} per_pool_A/B={} (decode reuses every pool)",
            total_arena_bytes,
            resident.iter().filter(|&&v| v).count(),
            self.prefill_layers.len(),
            resident_bytes,
            max_streamed,
            total_ab_bytes,
        );
        Ok(())
    }

    /// Select whole-layer interpretation for prefill. Direct layer staging ignores the decode
    /// LRU metadata; it is invalidated lazily by [`Self::enter_decode`] when decode begins.
    pub fn enter_prefill_layer(&mut self) -> Result<()> {
        self.build_prefill_layout()?;
        if self.mode != MoeArenaMode::PrefillLayer {
            self.prefill_lane_layer = [None, None];
            self.prefill_loaded.clear();
        }
        self.mode = MoeArenaMode::PrefillLayer;
        Ok(())
    }

    pub fn layer_bank_current(&self, buf_id: usize) -> Result<bool> {
        let placement = self
            .prefill_placement
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: no prefill placement for registered buffer"))?;
        Ok(self.mode == MoeArenaMode::PrefillLayer
            && self.prefill_loaded.contains(&buf_id)
            && placement
                .lane
                .is_none_or(|lane| self.prefill_lane_layer[lane] == Some(placement.layer_base)))
    }

    pub fn mark_layer_bank_current(&mut self, buf_id: usize) -> Result<()> {
        let placement = *self
            .prefill_placement
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: no prefill placement for registered buffer"))?;
        if let Some(lane) = placement.lane {
            if self.prefill_lane_layer[lane] != Some(placement.layer_base) {
                self.prefill_loaded.retain(|loaded| {
                    self.prefill_placement
                        .get(loaded)
                        .is_none_or(|p| p.lane != Some(lane))
                });
                self.prefill_lane_layer[lane] = Some(placement.layer_base);
            }
        }
        let banks = self
            .prefill_layers
            .iter()
            .find(|layer| layer.layer_base == placement.layer_base)
            .map(|layer| layer.banks.clone())
            .ok_or_else(|| be("moe pager: current layer missing from prefill layout"))?;
        self.prefill_loaded.extend(banks);
        Ok(())
    }

    pub fn layer_byte_offset(&self, buf_id: usize) -> Result<usize> {
        self.prefill_placement
            .get(&buf_id)
            .map(|p| p.byte_offset)
            .ok_or_else(|| be("moe pager: no prefill placement for registered buffer"))
    }

    /// One bank id identifying the next paged MoE layer. The layer-granular DMA path resolves it
    /// to the complete contiguous layer range; callers never iterate or upload role banks.
    pub fn next_layer(&self, buf_id: usize) -> Result<Option<usize>> {
        let layer_base = self
            .prefill_placement
            .get(&buf_id)
            .map(|p| p.layer_base)
            .ok_or_else(|| be("moe pager: next-layer query on an unplaced buffer"))?;
        let idx = self
            .prefill_layers
            .iter()
            .position(|layer| layer.layer_base == layer_base)
            .ok_or_else(|| be("moe pager: current layer missing from prefill order"))?;
        Ok(self
            .prefill_layers
            .get(idx + 1)
            .and_then(|layer| layer.banks.first().copied()))
    }

    /// The LUT tape buffer every windowed dispatch binds (see the `tape` field's doc).
    pub fn tape(&self) -> &dyn Buffer {
        self.tape.as_ref()
    }

    /// Whether ALL `n_expert` experts of `buf_id`'s layer are resident in its pool — the
    /// no-readback inline gate for a small-m (decode) layer: when true, any routing the GPU
    /// picks is covered, so the host needs no routing knowledge at all.
    pub fn all_resident(&self, buf_id: usize, n_expert: usize) -> bool {
        let (_, pool, src) = match self.sources.get(&buf_id) {
            Some(s) => s,
            None => return false,
        };
        let pager = &self.pools[*pool].pager;
        (0..n_expert as u32).all(|e| pager.is_resident(src.layer_base + e))
    }

    /// LRU maintenance for an inline-recorded (no-readback) layer: mark all `n_expert` blocks
    /// MRU. Callers gate on [`Self::all_resident`], so every touch is a hit — no uploads, no LUT
    /// mutation (the property that makes inline recording safe while earlier segments are still
    /// in flight).
    pub fn touch_all_hits(&mut self, buf_id: usize, n_expert: usize) -> Result<()> {
        let (_, pool, src) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: touch on an unregistered buffer"))?;
        let layer_base = src.layer_base;
        let pager = &mut self.pools[*pool].pager;
        pager.begin_batch();
        let prof = pager_profile::active();
        for e in 0..n_expert as u32 {
            let t0 = prof.then(std::time::Instant::now);
            let r = pager.pager.touch(layer_base + e);
            if let Some(t0) = t0 {
                let evicted = match r {
                    Resolution::Hit { .. } => false,
                    Resolution::Miss { evicted, .. } => evicted.is_some(),
                };
                pager_profile::record_gpu_cache_lookup(
                    matches!(r, Resolution::Hit { .. }),
                    evicted,
                    t0.elapsed(),
                );
            }
            debug_assert!(
                matches!(r, Resolution::Hit { .. }),
                "touch_all_hits on a non-resident block (all_resident gate violated)"
            );
        }
        Ok(())
    }

    /// Open a touch batch on `buf_id`'s pool — call once per (layer, role) residency resolution,
    /// BEFORE [`Self::push_role_cpu`] resolves that batch. The epoch protects earlier ids from
    /// later misses while multiple direct copies are recorded.
    pub fn begin_batch(&mut self, buf_id: usize) -> Result<()> {
        let (_, pool, _) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: begin_batch on an unregistered buffer"))?;
        self.pools[*pool].pager.begin_batch();
        Ok(())
    }

    /// Runtime Decode upload path backed by the unique CPU expert store. Every miss is copied
    /// directly into its final mapped-ReBAR LRU slot. The caller must have drained earlier arena
    /// readers before invoking this method; small-m Decode already does so before reading route ids.
    pub fn push_role_cpu(&mut self, buf_id: usize, local_ids: &[u32], scan: bool) -> Result<usize> {
        let (pool_idx, stride, layer_base, host_chunk, host_base) = {
            let (_, pool, src) = self
                .sources
                .get(&buf_id)
                .ok_or_else(|| be("moe pager: DMA stage on an unregistered buffer"))?;
            (
                *pool,
                src.stride_bytes,
                src.layer_base,
                src.host_chunk,
                src.host_offset,
            )
        };
        let Self {
            pools, host_store, ..
        } = self;
        let pager = &mut pools[pool_idx].pager;
        let arena_base = as_vk_buf(pager.arena_buffer())?
            .mapped_ptr()
            .ok_or_else(|| be("moe pager: ReBAR arena is not mapped"))?;
        let copy_t0 = pager_profile::active().then(std::time::Instant::now);
        let mut copied = 0usize;
        for &lid in local_ids {
            let local = lid as usize;
            let src = host_base
                .checked_add(local.saturating_mul(stride))
                .ok_or_else(|| be("moe pager: expert host offset overflow"))?;
            if let Some(dst) = pager.plan_cpu_push(layer_base + lid, scan)? {
                let bytes = host_store[host_chunk]
                    .bytes
                    .get(src..src + stride)
                    .ok_or_else(|| be("moe pager: expert CPU-store range out of bounds"))?;
                par_copy_to_mapped(bytes, unsafe { arena_base.add(dst) });
                copied += bytes.len();
            }
        }
        if let Some(t0) = copy_t0 {
            pager_profile::record_memcpy(copied, t0.elapsed());
        }
        Ok(local_ids.len())
    }

    /// CPU-push one whole layer from the unique host store straight into its final resident/A-B
    /// ReBAR placement. Load-time layout validation guarantees that every role bank and alignment
    /// gap has the same relative offset on both sides, so there is no pack/reorder/staging pass.
    pub fn push_prefill_layer_cpu(&mut self, buf_id: usize) -> Result<bool> {
        self.enter_prefill_layer()?;
        if self.layer_bank_current(buf_id)? {
            return Ok(false);
        }
        let layer_base = self
            .sources
            .get(&buf_id)
            .map(|(_, _, src)| src.layer_base)
            .ok_or_else(|| be("moe pager: layer DMA on an unregistered buffer"))?;
        let banks = self
            .prefill_layers
            .iter()
            .find(|layer| layer.layer_base == layer_base)
            .map(|layer| layer.banks.clone())
            .ok_or_else(|| be("moe pager: layer DMA missing from prefill layout"))?;
        let copy_t0 = pager_profile::active().then(std::time::Instant::now);
        let mut copied = 0usize;
        for bank_id in banks {
            let placement = *self
                .prefill_placement
                .get(&bank_id)
                .ok_or_else(|| be("moe pager: layer bank has no Prefill placement"))?;
            let (_, pool, source) = self
                .sources
                .get(&bank_id)
                .ok_or_else(|| be("moe pager: layer bank source disappeared"))?;
            if *pool != placement.pool {
                return Err(be("moe pager: Prefill placement points at the wrong pool"));
            }
            let src = self.host_store[source.host_chunk]
                .bytes
                .get(source.host_offset..source.host_offset + source.bank_bytes)
                .ok_or_else(|| be("moe pager: prefill CPU-store bank range out of bounds"))?;
            let dst = as_vk_buf(self.pools[*pool].pager.arena_buffer())?
                .mapped_ptr()
                .ok_or_else(|| be("moe pager: ReBAR pool arena is not mapped"))?;
            par_copy_to_mapped(src, unsafe { dst.add(placement.byte_offset) });
            copied += src.len();
        }
        if let Some(t0) = copy_t0 {
            pager_profile::record_memcpy(copied, t0.elapsed());
        }
        self.mark_layer_bank_current(buf_id)?;
        Ok(true)
    }

    /// Freeze the identity mapping used by whole-layer prefill. The dispatch base address points
    /// directly at this role bank, so local expert id is also its physical slot.
    pub fn layer_lut_window(
        &mut self,
        tape_cursor: &mut usize,
        _buf_id: usize,
        n_expert: usize,
    ) -> Result<u32> {
        if *tape_cursor + n_expert > self.tape_words {
            return Err(be(format!(
                "moe pager: LUT tape overflow ({} + {n_expert} > {} words)",
                *tape_cursor, self.tape_words,
            )));
        }
        let base = as_vk_buf(self.tape.as_ref())?
            .mapped_ptr()
            .ok_or_else(|| be("pager LUT tape is not persistently mapped"))?;
        let dst = unsafe { base.add(*tape_cursor * 4).cast::<u32>() };
        for e in 0..n_expert as u32 {
            unsafe { dst.add(e as usize).write(e) };
        }
        let window = *tape_cursor as u32;
        *tape_cursor += n_expert;
        Ok(window)
    }

    /// Freeze `buf_id`'s layer LUT window — `n_expert` slot indices starting at its `layer_base`,
    /// copied from the pool's host mirror into the tape at `*tape_cursor` — and return the tape
    /// word offset the layer's dispatches pass as `lut_base` (`lut[base + local_id]`). Must be
    /// called AFTER `push_role_cpu` for that (layer, role) batch completed (the
    /// within-batch LUT rule: the window must reflect every id the batch touched). Errors on
    /// tape overflow instead of wrapping — a wrapped window could alias one an in-flight segment
    /// still reads (the cursor only resets after a full drain; see the `tape` field's doc).
    pub fn lut_window(
        &mut self,
        tape_cursor: &mut usize,
        buf_id: usize,
        n_expert: usize,
    ) -> Result<u32> {
        let (_, pool, src) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: lut_window on an unregistered buffer"))?;
        if *tape_cursor + n_expert > self.tape_words {
            return Err(be(format!(
                "moe pager: LUT tape overflow ({} + {n_expert} > {} words) — one drain cycle \
                 recorded more layer windows than the tape holds",
                *tape_cursor, self.tape_words,
            )));
        }
        let window = self.pools[*pool]
            .pager
            .lut_words(src.layer_base as usize, n_expert);
        // The tape is session-owned Staging (persistently mapped) and the region written is
        // fresh this drain cycle — no in-flight reader can see a partial window.
        let base = as_vk_buf(self.tape.as_ref())?
            .mapped_ptr()
            .ok_or_else(|| be("pager LUT tape is not persistently mapped"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                window.as_ptr(),
                base.add(*tape_cursor * 4).cast::<u32>(),
                n_expert,
            );
        }
        let w = *tape_cursor as u32;
        *tape_cursor += n_expert;
        Ok(w)
    }

    fn pool_of(&self, buf_id: usize) -> Result<&Pool> {
        let (_, pool, _) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: arena/lut lookup on an unregistered buffer"))?;
        Ok(&self.pools[*pool])
    }

    /// The arena buffer `buf_id`'s pool dispatches against (callers gate on [`Self::is_paged`]
    /// first — this errors on an unregistered buffer).
    pub fn arena(&self, buf_id: usize) -> Result<&dyn Buffer> {
        Ok(self.pool_of(buf_id)?.pager.arena_buffer())
    }

    /// `buf_id`'s pool arena's 64-bit `VkDeviceAddress` — the base the paged kernels scale the LUT
    /// slot index onto (`arena_addr + slot * slot_bytes`). Passed to the shader as a push constant.
    pub fn arena_addr(&self, buf_id: usize) -> Result<u64> {
        if self.mode == MoeArenaMode::PrefillLayer {
            let placement = self
                .prefill_placement
                .get(&buf_id)
                .ok_or_else(|| be("moe pager: no Prefill placement for arena address"))?;
            return Ok(self.pools[placement.pool].pager.arena_addr() + placement.byte_offset as u64);
        }
        Ok(self.pool_of(buf_id)?.pager.arena_addr())
    }

    /// `buf_id`'s pool per-slot byte stride — the multiplier the paged kernels apply to the LUT
    /// slot index (see [`Self::arena_addr`]).
    pub fn slot_bytes(&self, buf_id: usize) -> Result<usize> {
        Ok(self.pool_of(buf_id)?.slot_bytes)
    }

    /// [`Self::arena`]'s LUT twin.
    pub fn lut(&self, buf_id: usize) -> Result<&dyn Buffer> {
        Ok(self.pool_of(buf_id)?.pager.lut_buffer())
    }

    /// Aggregate stats across every pool of `role` (the pool split is a capacity detail; the
    /// hit/miss story reads per role).
    pub fn stats(&self, role: Role) -> PagerStats {
        let mut agg = PagerStats::default();
        for p in self.pools.iter().filter(|p| p.role == role) {
            let s = p.pager.stats();
            agg.hits += s.hits;
            agg.misses += s.misses;
            agg.evictions += s.evictions;
        }
        agg
    }

    /// `paging.stats` (`INFR_PAGER_STATS=1`): print each pool's hit/miss/eviction counters. Called
    /// after generation finishes (see the CLI's bench/run/serve exit paths) — cheap enough to
    /// always compute, only printed when asked.
    pub fn print_stats_if_enabled(&self) {
        if !self.print_stats {
            return;
        }
        for p in &self.pools {
            let s = p.pager.stats();
            tracing::info!(
                "[moe pager] {}/{:.1}MB: {} slots={}",
                p.role.name(),
                p.slot_bytes as f64 / 1e6,
                stats_suffix(&s),
                p.pager.n_slots(),
            );
        }
    }

    /// Per-pool `(role, VRAM residency)` counters, in pool order.
    pub fn pool_stats(&self) -> Vec<(Role, PagerStats)> {
        self.pools
            .iter()
            .map(|p| (p.role, p.pager.stats()))
            .collect()
    }
}

/// `VulkanBackend::moe_pager`'s field type — a `Mutex` since decode touches the LRU and prefill
/// changes the shared arena interpretation from `execute_static` (`&VulkanBackend`).
pub type MoePagerCell = Mutex<Option<MoePagerSession>>;

// ─── Dense layer-streaming session ─────────────────────────────────────────────────────────────
//
// The MoE session above is demand-driven (routing is GPU-decided, residency resolves per touch);
// dense streaming is the SCHEDULE-driven policy `infr_core::pager`'s module doc names: a dense
// forward visits layers in one fixed order every pass, so the host knows every "miss" in advance
// and needs NO readbacks, NO LUT hop and NO paged kernel twins at all. One block = one per-layer
// weight tensor GROUP exactly as the seam uploads it (a fused qkv or gate_up concat is one
// block; split tensors are one block each) — every dense kernel already reads its weight from a
// `w_off` ELEMENT offset (the stacked-MoE-tensor convention), so a streamed dispatch computes the
// resident slot's base BYTE address (`arena_addr + slot * slot_bytes`, 64-bit — see
// `GpuPager::arena_addr`/`DensePagerSession::stage`) and rides the op's own `w_off` on top as a
// within-slot element offset, exactly like the resident path's binding + offset.
// Pools are keyed per (dtype, padded byte stride) tensor class — same reasoning as the MoE
// per-(role, slot_bytes) pools (fixed slot offsets require uniform strides; mixed-precision GGUFs
// bump a subset of layers' tensors to a wider format).
//
// Rejected alternatives (design notes for the seam this replaces):
//   - Descriptor-level (buffer, offset) rebinding: `Recorder::bind_descriptors` binds
//     `(buffer, 0, WHOLE_SIZE)` through ~seventy dispatch helpers — threading a per-binding
//     offset through every signature is a much bigger, riskier diff than reusing the `w_off`
//     element offset the kernels already take, and buys nothing (same descriptor write count).
//   - `-DPAGED` LUT twins of the dense kernels (the MoE route): pointless indirection — the host
//     knows the slot at record time, so the offset can be baked directly; a LUT hop would add a
//     device dependency for information the host already has.
//   - Embeddings / lm_head / norms / biases stay RESIDENT: norms and biases are consumed by ops
//     with no weight-offset support and are tiny (a few KB/layer); token_embd/lm_head are read at
//     every token edge — streaming lm_head would add its full bytes to every token's PCIe bill
//     with zero locality to exploit, a strict loss.

/// The tier a streamed dense block's bytes come from when VRAM misses.
pub enum DenseBytes {
    /// One or more consecutive zero-copy views into the GGUF mmap, in upload order — a fused
    /// qkv/gate_up block lists its component tensors so the concat never materializes in host RAM.
    /// The fits-in-host-RAM fast path, and what every model took before the host tier existed.
    Mmap(Vec<Arc<dyn AsRef<[u8]> + Send + Sync>>),
    /// The pool's host DRAM tier ([`DensePoolSpec::host`]), under this block's own `block_id` —
    /// the model does not fit host RAM either, so its bytes are read from the model file into a
    /// bounded arena instead of being left to the OS page cache.
    Host,
}

/// Where one streamed dense block's bytes live ([`DenseBytes`]), plus the block's schedule id
/// within its pool (ascending layer order — the cyclic-sweep key
/// `infr_core::pager::Pager::schedule` expects). The id keys BOTH tiers: a pool's host pager holds
/// exactly the same block set under the same ids, so one number locates a block in either.
pub struct DenseSource {
    pub bytes: DenseBytes,
    pub block_id: u32,
}

/// One dense pool's fixed layout: every block in it shares `slot_bytes` (the PADDED stride —
/// a multiple of 4 (u32 arena) AND of the pool dtype's block byte size, so a slot base is always
/// a whole number of quant blocks). The arena is a `bufferDeviceAddress` buffer read by 64-bit
/// pointer (see [`DensePagerSession`]), so `n_slots` is bounded only by the VRAM budget share (and
/// the seam's floor) — there is NO per-arena `maxStorageBufferRange` cap and NO u32 element-reach
/// cap (a slot's base byte address is computed in 64-bit; the op's `w_off` element offset rides on
/// top within the kernel). Contrast the resident/SSBO path, which those two caps DID bind.
pub struct DensePoolSpec {
    pub slot_bytes: usize,
    pub n_slots: usize,
    pub n_blocks: usize,
    /// This pool's tier BELOW VRAM, or `None` to read every miss from the mmap (the fast path).
    ///
    /// One host pager per pool rather than one per model: a pool is already exactly a block-size
    /// class, which is the uniform-slot shape [`HostPager`] needs, so the two tiers agree on both
    /// the block set and its ids with no mapping table between them.
    pub host: Option<Arc<HostPager>>,
}

struct DensePool {
    spec: DensePoolSpec,
    pager: GpuPager,
}

/// Layout for [`DensePagerSession::new`] — like [`MoePagerLayout`], sized up front so the session
/// exists (and `Backend::dense_paged` answers truthy) BEFORE the seam's weight-load closure binds
/// the first placeholder.
pub struct DensePagerLayout {
    pub pools: Vec<DensePoolSpec>,
    /// Pinned upload ring total bytes (independently fenced regions); `0` = [`ring_bytes`]'s
    /// floor. Each half is floored at the largest pool slot so one miss always fits.
    pub ring_bytes: usize,
}

/// One model's whole dense layer-streaming session: per-(dtype, stride) arena pools + the shared
/// pinned upload ring. Same ownership story as [`MoePagerSession`] (lives on the `VulkanBackend`
/// handle, `None` for every non-streamed model — zero cost on the resident path). A model is
/// either MoE-paged or dense-streamed, never both (the seam errors on the mixed case).
pub struct DensePagerSession {
    pools: Vec<DensePool>,
    /// `buffer_identity(placeholder)` -> (pool index, source) for every streamed block. A
    /// resident tensor's buffer is never registered here — the adapter's lookup misses and the
    /// op lowers through the ordinary resident path.
    sources: HashMap<usize, (usize, DenseSource)>,
    ring: Box<dyn Buffer>,
    ring_half_bytes: usize,
    ring_slots: usize,
    print_stats: bool,
}

impl DensePagerSession {
    pub fn new(vk: &VulkanBackend, layout: DensePagerLayout) -> Result<Self> {
        // The streamed kernels read the arena by 64-bit device address (native_weight_addr.glsl), so
        // BDA is required. It is probed and hard-errored globally at init (lib.rs, `caps()
        // .buffer_device_address`); assert here so a future refactor that lands a dense session on a
        // BDA-less device fails loudly rather than allocating an un-addressable arena.
        debug_assert!(
            vk.caps().buffer_device_address,
            "dense streaming needs bufferDeviceAddress (BDA arena)"
        );
        let mut pools = Vec::with_capacity(layout.pools.len());
        let mut max_slot = 4usize;
        for spec in layout.pools {
            max_slot = max_slot.max(spec.slot_bytes);
            pools.push(DensePool {
                // Dense-streaming pool: the arena is a `bufferDeviceAddress` buffer the streamed
                // kernels read through a 64-bit pointer, so it may span as much VRAM as the
                // budget allows — no `maxStorageBufferRange` cap (the pre-BDA ~4 GiB-per-pool
                // ceiling this lifts), no u32 element-reach cap.
                pager: GpuPager::new(vk, spec.n_blocks, spec.n_slots, spec.slot_bytes)?,
                spec,
            });
        }
        let ring_total = if layout.ring_bytes > 0 {
            layout.ring_bytes
        } else {
            ring_bytes(0, vk.cfg().paging.ring)
        };
        // Each half must hold the largest slot or `stage` could never make progress on that pool.
        let ring_slots = vk.cfg().paging.ring_slots.clamp(2, 8);
        let ring_half_bytes = ring_region_bytes(ring_total, ring_slots, max_slot);
        let ring = vk.alloc_uninit(ring_slots * ring_half_bytes, BufferUsage::Staging)?;
        Ok(Self {
            pools,
            sources: HashMap::new(),
            ring,
            ring_half_bytes,
            ring_slots,
            print_stats: vk.cfg().paging.stats,
        })
    }

    /// Register one streamed block — called from the seam's weight-load closure (once per
    /// streamed weight group) instead of uploading it. `pool` indexes [`DensePagerLayout::pools`]
    /// (the seam enumerates the layout and the registrations from the same plan, so a mismatch is
    /// a seam bug — validated loudly here).
    pub fn register(&mut self, pool: usize, buf_id: usize, source: DenseSource) -> Result<()> {
        let p = self
            .pools
            .get(pool)
            .ok_or_else(|| be(format!("dense pager: pool index {pool} out of range")))?;
        let total: usize = match &source.bytes {
            DenseBytes::Mmap(segments) => segments.iter().map(|s| expert_bytes(s).len()).sum(),
            // Asking the host tier for the size doubles as the check that the seam registered the
            // block there first — a `Host` source whose block is unknown below would otherwise
            // fail only later, mid-generation, on the first miss.
            DenseBytes::Host => {
                let host = p.spec.host.as_ref().ok_or_else(|| {
                    be(format!(
                        "dense pager: pool {pool} has no host tier for host-backed block {}",
                        source.block_id
                    ))
                })?;
                host.block_bytes(source.block_id).ok_or_else(|| {
                    be(format!(
                        "dense pager: block {} is not registered with pool {pool}'s host tier",
                        source.block_id
                    ))
                })?
            }
        };
        if total > p.spec.slot_bytes {
            return Err(be(format!(
                "dense pager: block bytes ({total}) exceed pool {pool}'s slot stride ({})",
                p.spec.slot_bytes
            )));
        }
        if source.block_id as usize >= p.spec.n_blocks {
            return Err(be(format!(
                "dense pager: block id {} out of range for pool {pool} ({} blocks)",
                source.block_id, p.spec.n_blocks
            )));
        }
        self.sources.insert(buf_id, (pool, source));
        Ok(())
    }

    /// Whether `buf_id` (see [`buffer_identity`]) is a registered streamed block — the adapter's
    /// per-`Op::Linear` dispatch check.
    pub fn is_streamed(&self, buf_id: usize) -> bool {
        self.sources.contains_key(&buf_id)
    }

    /// Ensure `buf_id`'s block is resident, staging a miss through `rec`-recorded ring→arena
    /// copies at `half_base + *cursor`. Returns the resident slot's arena base BYTE address (the
    /// streamed dispatch sets `w_addr` to it and adds the op's own `w_off` element offset on top —
    /// see native_weight_addr.glsl and [`crate::recorder::Recorder::linear_native_at`]), or
    /// `None` when the current ring half can't hold the miss — the caller rotates the ring
    /// (pipelined submit) and re-calls. The address is computed in 64-bit, so no arena size
    /// overflows it (the u32 element-reach the SSBO path needed is gone). Residency rides the exact
    /// cyclic-sweep policy (`infr_core::pager::Pager::schedule`); one block = one touch batch (the
    /// epoch guard protects it across the caller's rotations).
    pub fn stage(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        half_base: usize,
        cursor: &mut usize,
        buf_id: usize,
    ) -> Result<Option<u64>> {
        let Self {
            pools,
            sources,
            ring,
            ring_half_bytes,
            ..
        } = self;
        let (pool_idx, src) = sources
            .get(&buf_id)
            .ok_or_else(|| be("dense pager: stage on an unregistered buffer"))?;
        let pool = &mut pools[*pool_idx];
        let id = src.block_id;
        let needs_slot = !pool.pager.is_resident(id);
        if needs_slot && *cursor + pool.spec.slot_bytes > *ring_half_bytes {
            return Ok(None); // half full — caller rotates and re-calls
        }
        pool.pager.begin_batch();
        let acquire_t0 = (needs_slot && pager_profile::active()).then(std::time::Instant::now);
        let ring_off = half_base + *cursor;
        if let Some(t0) = acquire_t0 {
            pager_profile::record_staging_acquire(t0.elapsed());
        }
        // Pass the source through by reference — `schedule_staged` derefs mmap segments via
        // `expert_bytes` and pins a host block in place, so neither tier materializes a copy here.
        let (slot, consumed) = pool.pager.schedule_staged(
            rec,
            ring.as_ref(),
            ring_off,
            id,
            &src.bytes,
            pool.spec.host.as_deref(),
        )?;
        *cursor += consumed;
        // Slot base BYTE address = arena base + slot * slot_bytes, in 64-bit (the BDA arena's
        // `arena_addr()`; the streamed kernel dereferences this pointer). No cap: the multiply and
        // the address are 64-bit, so an arena of any size the VRAM budget allows is addressable.
        let addr = pool.pager.arena_addr() + slot as u64 * pool.spec.slot_bytes as u64;
        Ok(Some(addr))
    }

    pub fn ring_half_bytes(&self) -> usize {
        self.ring_half_bytes
    }

    pub fn ring_slots(&self) -> usize {
        self.ring_slots
    }

    /// Per-pool `(VRAM residency, host tier)` counters, in pool order.
    ///
    /// The host half is what separates a VRAM miss the DRAM tier absorbed from one that reached the
    /// disk — the two are indistinguishable in every other number this session reports, and a
    /// three-tier path where one of those cases never runs looks exactly like one where it works.
    pub fn pool_stats(&self) -> Vec<(PagerStats, Option<infr_core::hostpager::HostPagerStats>)> {
        self.pools
            .iter()
            .map(|p| (p.pager.stats(), p.spec.host.as_ref().map(|h| h.stats())))
            .collect()
    }

    /// `paging.stats` (`INFR_PAGER_STATS=1`): per-pool hit/miss/eviction counters (cyclic-sweep
    /// hit rate = `(n_slots-1) / n_blocks` per pass at steady state — the honest expectation to
    /// check against).
    pub fn print_stats_if_enabled(&self) {
        if !self.print_stats {
            return;
        }
        for (i, p) in self.pools.iter().enumerate() {
            let s = p.pager.stats();
            tracing::info!(
                "[dense pager] pool{i}/{:.1}MB: {} slots={}/{}",
                p.spec.slot_bytes as f64 / 1e6,
                stats_suffix(&s),
                p.spec.n_slots,
                p.spec.n_blocks,
            );
            // The tier below, when there is one. Its READS are the line that matters: a VRAM miss
            // that the host tier also missed is what actually touched the disk, and nothing else
            // reported here distinguishes those two from each other.
            if let Some(h) = &p.spec.host {
                let hs = h.stats();
                tracing::info!(
                    "[dense pager]   host{i}: {} slots={} reads={} ({} streamed past the arena) \
                     {:.2}GB from disk",
                    stats_suffix(&hs.pager),
                    h.n_slots(),
                    hs.reads,
                    hs.streamed,
                    hs.bytes_read as f64 / 1e9,
                );
            }
        }
    }
}

/// `VulkanBackend::dense_pager`'s field type — same locking story as [`MoePagerCell`].
pub type DensePagerCell = Mutex<Option<DensePagerSession>>;

#[cfg(test)]
mod tests {
    use super::*;

    // ── #4: GpuPager::new dimension validation returns Err (not panic) on bad input ──────────────
    #[test]
    fn validate_pager_dims_rejects_zero_slots() {
        assert!(validate_pager_dims(0, 64).is_err());
    }

    #[test]
    fn validate_pager_dims_rejects_misaligned_slot_bytes() {
        assert!(validate_pager_dims(4, 3).is_err());
        assert!(validate_pager_dims(4, 6).is_err());
    }

    #[test]
    fn validate_pager_dims_accepts_valid() {
        assert!(validate_pager_dims(1, 4).is_ok());
        assert!(validate_pager_dims(238, 13 << 20).is_ok());
    }

    #[test]
    fn ring_regions_are_aligned_without_exceeding_the_shared_budget() {
        let total = 3584 * 1024 * 1024usize;
        for slots in 2..=8 {
            let region = ring_region_bytes(total, slots, 900_003);
            assert_eq!(region % RING_REGION_ALIGN, 0);
            assert!(region >= 900_003);
            assert!(region * slots <= total);
        }
    }

    #[test]
    fn ring_region_floor_is_aligned_when_one_upload_exceeds_its_share() {
        let region = ring_region_bytes(1024, 8, 513);
        assert_eq!(region, 768);
    }

    #[test]
    fn staging_copy_batch_copies_disjoint_experts_exactly() {
        let sources = [vec![0x11u8; 257], vec![0x7au8; 513], vec![0xe3u8; 129]];
        let offsets = [0usize, 320, 896];
        let mut dst = vec![0u8; 1100];
        let jobs = sources
            .iter()
            .zip(offsets)
            .map(|(src, off)| StagingCopy {
                src: src.as_ptr() as usize,
                dst: unsafe { dst.as_mut_ptr().add(off) } as usize,
                len: src.len(),
            })
            .collect::<Vec<_>>();

        run_staging_copies(&jobs);

        for (src, off) in sources.iter().zip(offsets) {
            assert_eq!(&dst[off..off + src.len()], src);
        }
        assert!(dst[257..320].iter().all(|&b| b == 0));
    }

    // ── #5: record_placement / apply_placement LUT bookkeeping is byte-identical to the old
    //        inline evict-then-insert blocks (unit-tested on a plain mirror, no GPU) ──────────────
    #[test]
    fn apply_placement_insert_no_eviction() {
        let mut lut = vec![NOT_RESIDENT; 8];
        apply_placement(&mut lut, 3, 5, None);
        assert_eq!(lut[3], 5);
        // every other entry untouched
        for (i, &v) in lut.iter().enumerate() {
            if i != 3 {
                assert_eq!(v, NOT_RESIDENT);
            }
        }
    }

    #[test]
    fn apply_placement_evict_then_insert() {
        let mut lut = vec![NOT_RESIDENT; 8];
        // block 2 already resident in slot 5
        apply_placement(&mut lut, 2, 5, None);
        // block 6 moves into slot 5, evicting block 2 (the old occupant of that slot)
        apply_placement(&mut lut, 6, 5, Some(2));
        assert_eq!(
            lut[2], NOT_RESIDENT,
            "evicted block must clear to NOT_RESIDENT"
        );
        assert_eq!(lut[6], 5, "new block records the reused slot index");
    }

    #[test]
    fn apply_placement_insert_evict_order_matters_for_self_reuse() {
        // If a block were (pathologically) evicting itself, the insert must win — evict clears
        // first, then insert writes. Guards the ordering the old inline blocks had.
        let mut lut = vec![NOT_RESIDENT; 4];
        apply_placement(&mut lut, 1, 7, Some(1));
        assert_eq!(lut[1], 7);
    }

    #[test]
    fn apply_placement_ignores_out_of_range_ids() {
        // Mirrors the old `get_mut(..)` guards: an id/evicted past the mirror end is a no-op, not
        // a panic (an out-of-pool layer's block is never asked for, but stay total).
        let mut lut = vec![NOT_RESIDENT; 2];
        apply_placement(&mut lut, 99, 3, Some(88));
        assert_eq!(lut, vec![NOT_RESIDENT; 2]);
    }
}
