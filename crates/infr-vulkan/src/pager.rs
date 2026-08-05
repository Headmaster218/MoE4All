//! GPU-resident paged weight cache: wraps `infr_core::pager::Pager`'s host-side LRU bookkeeping
//! with a fixed-slot VRAM arena, a small host-writable/GPU-readable LUT buffer, and upload
//! machinery through a caller-supplied REUSED pinned staging buffer (validated by
//! `tests/bandwidth_probe.rs` — a fresh staging buffer per call roughly halves throughput; see
//! that test's `fresh` vs `combined` columns. On this box the device-copy phase itself is nearly
//! free — ReBAR puts the staging buffer in device-local host-visible VRAM, so the bottleneck is
//! the host memcpy into it, not the subsequent `vkCmdCopyBuffer`).
//!
//! # Design (block-agnostic core, MoE plugs in today)
//! [`GpuPager`] only knows about uniform `slot_bytes`-sized blocks keyed by an opaque
//! `infr_core::pager::BlockId` — it has no idea a block is "an expert". The MoE integration
//! (`infr-llama`'s seam / this crate's `adapter.rs`) packs a `BlockId` from `(layer, role,
//! expert_id)` and calls [`GpuPager::ensure_resident`] with that block's mmap'd tensor bytes
//! before dispatching the id-indexed GEMV/GEMM through the LUT hop (the `PAGED` branch in
//! `shaders/native_gemv_id.comp` / `native_gemv_id_multi.comp`: `slot = lut[ids[slot]]`, scaled
//! onto the arena's 64-bit device address as `arena_addr + slot * slot_bytes` — see the `lut_host`
//! field's doc and `shaders/native_weight_addr.glsl`). A FUTURE dense layer-streaming policy
//! (NOT implemented here — see the task doc) would reuse this exact struct with `BlockId =
//! layer_idx`, `slot_bytes` = one layer's weight size, and a schedule-driven (not LRU) `touch`
//! order (a dense decode visits layers in a fixed known order, so it can exact-prefetch layer
//! `l+1` while `l` runs) — nothing in the arena/LUT/upload core below assumes MoE or LRU.
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
use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use infr_core::backend::{Buffer, BufferUsage};
use infr_core::error::Result;
use infr_core::hostpager::HostPager;
use infr_core::pager::{BlockId, Insert, Pager, PagerStats, Resolution, NOT_RESIDENT};
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
    arena: Box<dyn Buffer>,
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
        let lut_dev = vk.alloc_uninit(n_blocks.max(1) * 4, BufferUsage::Staging)?;
        let lut_host = vec![NOT_RESIDENT; n_blocks.max(1)];
        // Seed the device LUT with the same all-absent state (arena/LUT start coherent).
        vk.upload(lut_dev.as_ref(), bytemuck::cast_slice(&lut_host))?;
        Ok(Self {
            pager: Pager::new(n_slots),
            slot_bytes,
            arena,
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
        // `scan`: full-set sweep (batched prefill's touch-all) → the scan-resistant cold-end
        // policy; otherwise classic LRU (decode's routed-only touches). See
        // `infr_core::pager::Pager::touch_cold`.
        let resolution = if scan {
            self.pager.touch_cold(id)
        } else {
            self.pager.touch(id)
        };
        match resolution {
            Resolution::Hit { .. } => Ok(0),
            Resolution::Miss { slot, evicted } => {
                let base = as_vk_buf(ring)?
                    .mapped_ptr()
                    .ok_or_else(|| be("pager staging ring is not persistently mapped"))?;
                par_copy_to_mapped(bytes, unsafe { base.add(ring_off) });
                rec.copy(
                    ring,
                    ring_off,
                    self.arena.as_ref(),
                    slot as usize * self.slot_bytes,
                    self.slot_bytes,
                );
                self.record_placement(id, slot, evicted);
                Ok(self.slot_bytes)
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
        let resolution = if scan {
            self.pager.touch_cold(id)
        } else {
            self.pager.touch(id)
        };
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
                    slot as usize * self.slot_bytes,
                    self.slot_bytes,
                );
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
        match self.pager.schedule(id) {
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
                        let mut off = ring_off;
                        for s in segments {
                            let seg = expert_bytes(s);
                            par_copy_to_mapped(seg, unsafe { base.add(off) });
                            off += seg.len();
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
                    slot as usize * self.slot_bytes,
                    total.next_multiple_of(4),
                );
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
        match self.pager.touch(id) {
            Resolution::Hit { slot } => Ok(slot),
            Resolution::Miss { slot, evicted } => {
                vk.upload(staging, bytes)?;
                copy_into_slot(vk, staging, self.arena.as_ref(), slot, self.slot_bytes)?;
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

/// Device-to-device copy of `len` bytes from `src[0..len]` into `dst[slot*len .. (slot+1)*len]` —
/// the pager's slot placement, which the shared `Backend::copy_buffer` can't express (it always
/// copies `[0, bytes)` on both sides). Internal to this crate: raw `ash` calls mirroring
/// `VulkanBackend::upload`'s device-copy branch exactly, just with a nonzero destination offset.
fn copy_into_slot(
    vk: &VulkanBackend,
    src: &dyn Buffer,
    dst: &dyn Buffer,
    slot: u32,
    len: usize,
) -> Result<()> {
    let (s, d) = (as_vk_buf(src)?, as_vk_buf(dst)?);
    let (sb, db) = (s.buffer, d.buffer);
    let dst_offset = slot as u64 * len as u64;
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
// bound weight BUFFER's identity to where its layer's expert bytes live in the mmap'd GGUF, and
// the one persistent staging buffer every pool's uploads share.
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

/// Which tier one paged layer's per-role expert bank is read from.
///
/// Either way, "expert `e` is the `e`-th equal-size contiguous slice" holds for every GGUF MoE bank
/// in this codebase (`Op::MoeFfn`'s doc), so `stride_bytes` locates any expert with no
/// quant-format-specific math — the two arms differ only in whether that slice is reached through
/// the mapping or read from the file.
pub enum ExpertBytes {
    /// One zero-copy view of the whole per-layer bank; expert `e` is the `e`-th equal-size slice.
    Mmap(Arc<dyn AsRef<[u8]> + Send + Sync>),
    /// The pool's host DRAM tier ([`MoePoolSpec::host`]), which holds ONE BLOCK PER EXPERT under
    /// the same global id the arena uses (`layer_base + local_id`). The bank is never viewed whole
    /// on this path — an expert is read from the model file on its own, which is what lets a model
    /// far larger than host RAM route through a bounded arena.
    Host,
}

/// Where one paged layer's per-role expert bank lives ([`ExpertBytes`]), plus the byte stride of
/// ONE expert within it. The mmap arm keeps the bank alive via `Arc` — see `infr_gguf::TensorBytes`,
/// which that trait object mirrors without infr-vulkan taking a dependency on infr-gguf.
pub struct ExpertSource {
    pub bytes: ExpertBytes,
    pub stride_bytes: usize,
    /// This layer's offset into the role's shared LUT/arena block-id space
    /// (`layer_index * n_expert`) — turns a per-layer LOCAL expert id (what the router/top-k
    /// produces, `0..n_expert`) into a GLOBAL `BlockId` unique across every paged layer of this
    /// role, so one `Pager`/LUT can hold experts from many layers at once.
    pub layer_base: u32,
}

/// One arena pool: every block in it shares `slot_bytes` (see the section doc above for why the
/// pool key is `(role, slot_bytes)`, not the role alone).
struct Pool {
    role: Role,
    slot_bytes: usize,
    pager: GpuPager,
    /// This pool's tier BELOW VRAM, holding one block per expert (see [`ExpertBytes::Host`]), or
    /// `None` to read every miss from the mmap.
    host: Option<Arc<HostPager>>,
}

/// One model's whole paged-MoE session: the `(role, slot_bytes)` arena pools + the shared
/// persistent staging buffer their uploads reuse (the bandwidth probe's headline finding — see
/// `pager.rs`'s module doc and `tests/bandwidth_probe.rs`). Lives on the `VulkanBackend` HANDLE
/// (NOT `VulkanShared` — the session's buffers hold `Arc<VulkanShared>` clones, and parking it on
/// the shared state made an Arc cycle that leaked the device's whole VRAM footprint until process
/// exit; see the `moe_pager` field doc in lib.rs) for as long as the backend that loaded the
/// paged model lives (`VulkanBackend::init_moe_pager`); `None` for every non-paged model — zero
/// cost, zero behavior change on the common (fits-in-VRAM) path.
pub struct MoePagerSession {
    pools: Vec<Pool>,
    /// `buffer_identity(placeholder)` -> (role, pool index, this layer's expert source), for
    /// every PAGED `_exps` tensor. A non-paged layer's gate/up/down buffer is never registered
    /// here — the adapter's lookup simply misses and falls through to the ordinary
    /// resident-weight path.
    sources: HashMap<usize, (Role, usize, ExpertSource)>,
    staging: Box<dyn Buffer>,
    /// Pinned staging RING for the recorded-upload path ([`GpuPager::touch_staged`]): two
    /// fence-rotated halves of [`Self::ring_half_bytes`] each, so the CPU stages the next
    /// segment's misses while the GPU executes the previous one (see
    /// `adapter::execute_paged_moe`'s rotation). Sized by [`MoePagerLayout::ring_bytes`].
    ring: Box<dyn Buffer>,
    ring_half_bytes: usize,
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
    /// Reusable output buffer for [`Self::touch_role`]'s global-id list — cleared and refilled per
    /// call so the demand path allocates no per-touch `Vec`.
    global_scratch: Vec<u32>,
}

/// One pool's spec in [`MoePagerLayout`]: slot counts are INDEPENDENT per pool. Each pool's arena
/// is a `bufferDeviceAddress` buffer (`48ad9c1`) addressed by 64-bit pointer — no per-arena
/// `maxStorageBufferRange` ceiling — but per-pool sizing
/// still matters because of unequal per-expert sizes (Scout: gate/up 13.8 MB, down 18 MB): a
/// shared slot count is dragged down to fit the LARGEST pool's per-slot bytes within the VRAM
/// budget and strands budget the smaller pools could have used as real hit rate (Scout: uniform
/// 238 slots everywhere left ~6 GB of a 19 GB budget unused; per-pool sizing gives gate/up 312
/// each). Each pool has its own LRU/LUT and `touch_role` resolves pools independently, so unequal
/// counts are correctness-neutral — a pool with fewer slots just misses more often. Computed by
/// the caller (budget-driven count, then per-pool split — see `seam::mod`'s placement policy).
pub struct MoePoolSpec {
    pub role: Role,
    pub slot_bytes: usize,
    pub n_slots: usize,
    /// This pool's host DRAM tier, or `None` to keep every miss on the mmap (the fast path).
    /// Registered with one block per (layer, expert) under the same global id the arena uses, so
    /// the two tiers need no mapping between them — see [`ExpertBytes::Host`].
    pub host: Option<Arc<HostPager>>,
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
    /// Total bytes for the pinned upload ring (two fence-rotated halves — see
    /// [`MoePagerSession`]'s `ring` field). `0` picks the default
    /// ([`ring_bytes`]); either way each half is floored at the largest pool slot so one
    /// miss always fits. The seam's budget math subtracts this before splitting arena shares.
    pub ring_bytes: usize,
}

/// Upload-ring sizing policy — pure budget arithmetic, so it lives in the shared seam
/// ([`infr_core::pager::ring_bytes`], which owns the doc and the boundary tests). Re-exported
/// under this crate's old path because the ring it sizes is a Vulkan buffer pair and every call
/// site here reads better next to them. The `paging.ring` override comes off the backend's
/// `Config` (`INFR_PAGER_RING`), so the caller passes it in.
pub use infr_core::pager::ring_bytes;

impl MoePagerSession {
    pub fn new(vk: &VulkanBackend, layout: MoePagerLayout) -> Result<Self> {
        let mut pools = Vec::with_capacity(layout.pools.len());
        let mut staging_bytes = 4usize;
        for spec in &layout.pools {
            pools.push(Pool {
                role: spec.role,
                slot_bytes: spec.slot_bytes,
                // MoE pools are pointer-addressed (`bufferDeviceAddress`) — no per-arena SSBO cap.
                pager: GpuPager::new(vk, layout.n_blocks, spec.n_slots, spec.slot_bytes)?,
                host: spec.host.clone(),
            });
            staging_bytes = staging_bytes.max(spec.slot_bytes);
        }
        let staging = vk.alloc_uninit(staging_bytes, BufferUsage::Staging)?;
        // Each ring half must hold the largest slot, or `touch_staged` could never make progress
        // on that pool (the adapter rotates halves when one fills; a slot bigger than a half
        // would fit in neither).
        let ring_total = if layout.ring_bytes > 0 {
            layout.ring_bytes
        } else {
            // 0 budget → the clamp floor; the configured `paging.ring` override still wins.
            ring_bytes(0, vk.cfg().paging.ring)
        };
        let ring_half_bytes = (ring_total / 2).max(staging_bytes);
        let ring = vk.alloc_uninit(2 * ring_half_bytes, BufferUsage::Staging)?;
        // One graph's windows = paged layers x roles x n_expert words (Scout: 48 x 3 x 16 = 2.3k)
        // — 64k words (256 KiB) leaves an order of magnitude of headroom; `lut_window` hard-errors
        // on overflow rather than wrapping into a region an in-flight segment may still read.
        let tape_words = 64 * 1024;
        let tape = vk.alloc_uninit(tape_words * 4, BufferUsage::Staging)?;
        Ok(Self {
            pools,
            sources: HashMap::new(),
            staging,
            ring,
            ring_half_bytes,
            tape,
            tape_words,
            print_stats: vk.cfg().paging.stats,
            global_scratch: Vec::new(),
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
        if matches!(source.bytes, ExpertBytes::Host) {
            let host = self.pools[pool].host.as_ref().ok_or_else(|| {
                be(format!(
                    "moe pager: pool ({role:?}, {} B) has no host tier for a host-backed bank",
                    source.stride_bytes,
                ))
            })?;
            for e in 0..n_expert as u32 {
                let id = source.layer_base + e;
                match host.block_bytes(id) {
                    Some(n) if n == source.stride_bytes => {}
                    other => {
                        return Err(be(format!(
                            "moe pager: expert block {id} is {other:?} bytes in the host tier, \
                             expected {}",
                            source.stride_bytes
                        )))
                    }
                }
            }
        }
        self.sources.insert(buf_id, (role, pool, source));
        Ok(())
    }

    /// Whether `buf_id` (see [`buffer_identity`]) is a registered paged tensor of `role` — the
    /// adapter's per-`MoeFfn` dispatch check.
    pub fn is_paged(&self, role: Role, buf_id: usize) -> bool {
        self.sources.get(&buf_id).is_some_and(|(r, ..)| *r == role)
    }

    /// Resolve residency for every id in `local_ids` (this token's routed experts, LOCAL to the
    /// layer) against `buf_id`'s pool, uploading misses through the shared staging buffer and
    /// flushing the LUT once. Returns the GLOBAL ids (`layer_base + local_id`) the paged GEMV
    /// must read instead of `local_ids` — see [`ExpertSource::layer_base`].
    pub fn touch_role(
        &mut self,
        vk: &VulkanBackend,
        role: Role,
        buf_id: usize,
        local_ids: &[u32],
    ) -> Result<&[u32]> {
        let (r, pool, src) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: touch on an unregistered buffer"))?;
        debug_assert_eq!(*r, role, "touch_role: role/buffer mismatch");
        let stride = src.stride_bytes;
        let layer_base = src.layer_base;
        let bank = match &src.bytes {
            ExpertBytes::Mmap(a) => Some(expert_bytes(a)),
            ExpertBytes::Host => None,
        };
        // Destructure the pool: the pager is taken mutably while the tier below is read, and
        // indexing twice would borrow the whole `pools` vector each time.
        let Pool { pager, host, .. } = &mut self.pools[*pool];
        // Reuse a scratch Vec across calls instead of allocating one per touch — this is the
        // steady-state demand path (per layer per token). NOTE: each miss still does one
        // synchronous `one_shot` submit (inside `ensure_resident`); batching those into a single
        // recorded submission would materially change the recorded stream — the RING/`stage_role`
        // path exists precisely for that — so the demand path keeps the per-miss copy and only the
        // per-call allocation is removed here.
        let global = &mut self.global_scratch;
        global.clear();
        global.reserve(local_ids.len());
        for &lid in local_ids {
            let id = layer_base + lid;
            match bank {
                Some(bytes) => {
                    let off = lid as usize * stride;
                    let slice = bytes.get(off..off + stride).ok_or_else(|| {
                        be("moe pager: expert id out of range for this layer's bank")
                    })?;
                    pager.ensure_resident(vk, self.staging.as_ref(), id, slice)?;
                }
                // Tier below: routing is unpredictable, so this caches under recency like the
                // arena above it does on the demand path (`docs/disk-streaming-plan.md` §3.6).
                None => {
                    let host = host
                        .as_ref()
                        .ok_or_else(|| be(format!("moe pager: expert {id} has no host tier")))?;
                    let pin = host.pin(id, Insert::Mru)?;
                    pager.ensure_resident(vk, self.staging.as_ref(), id, &pin)?;
                }
            }
            global.push(id);
        }
        pager.flush_lut(vk)?;
        Ok(&self.global_scratch)
    }

    /// The shared upload ring / its per-half capacity (see the `ring` field's doc). The CURSOR
    /// into it lives with the adapter's per-execute stream state, not here.
    pub fn ring(&self) -> &dyn Buffer {
        self.ring.as_ref()
    }

    pub fn ring_half_bytes(&self) -> usize {
        self.ring_half_bytes
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
        for e in 0..n_expert as u32 {
            let r = pager.pager.touch(layer_base + e);
            debug_assert!(
                matches!(r, Resolution::Hit { .. }),
                "touch_all_hits on a non-resident block (all_resident gate violated)"
            );
        }
        Ok(())
    }

    /// Open a touch batch on `buf_id`'s pool — call once per (layer, role) residency resolution,
    /// BEFORE the first [`Self::stage_role`] call of that batch (rotations re-call `stage_role`
    /// WITHIN the same batch; the epoch protection must span them).
    pub fn begin_batch(&mut self, buf_id: usize) -> Result<()> {
        let (_, pool, _) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("moe pager: begin_batch on an unregistered buffer"))?;
        self.pools[*pool].pager.begin_batch();
        Ok(())
    }

    /// Stage `local_ids`' residency for `buf_id`'s layer through `rec`-recorded ring→arena
    /// copies: hits are marked MRU, misses memcpy into the ring at `half_base + *cursor` and
    /// record the slot copy ([`GpuPager::touch_staged`]). Stops when the current ring half can't
    /// hold the next miss and returns how many ids were FULLY staged — the caller rotates the
    /// ring (submitting the recorder, fencing the half) and re-calls with the remainder; an
    /// expert's bytes are never split across a rotation. Progress is guaranteed: a half holds at
    /// least one slot of every pool (asserted at construction).
    ///
    /// Within-batch eviction safety (`infr_core::pager::Pager`'s invariant) holds across
    /// rotations: a rotation performs no touches, and every id staged earlier in this batch is
    /// MRU-protected from the batch's later touches exactly as in the one-shot path.
    /// `scan` selects the residency policy: `true` = the touch-all full-set sweep (batched
    /// prefill) → scan-resistant cold-end insertion; `false` = classic LRU (decode's routed-only
    /// readback path). See `infr_core::pager::Pager::touch_cold`.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_role(
        &mut self,
        rec: &crate::recorder::Recorder<'_>,
        half_base: usize,
        cursor: &mut usize,
        buf_id: usize,
        local_ids: &[u32],
        scan: bool,
    ) -> Result<usize> {
        let (pool_idx, stride, layer_base, bytes_arc) = {
            let (_, pool, src) = self
                .sources
                .get(&buf_id)
                .ok_or_else(|| be("moe pager: stage on an unregistered buffer"))?;
            let arc = match &src.bytes {
                ExpertBytes::Mmap(a) => Some(Arc::clone(a)),
                ExpertBytes::Host => None,
            };
            (*pool, src.stride_bytes, src.layer_base, arc)
        };
        let bank = bytes_arc.as_ref().map(expert_bytes);
        // Disjoint field borrows (the pool mutably, the ring by ref) — destructure once.
        let Self {
            pools,
            ring,
            ring_half_bytes,
            ..
        } = self;
        let Pool { pager, host, .. } = &mut pools[pool_idx];
        let half_bytes = *ring_half_bytes;
        debug_assert!(
            half_bytes >= pager.slot_bytes(),
            "ring half smaller than a slot (construction floor violated)"
        );
        for (i, &lid) in local_ids.iter().enumerate() {
            let id = layer_base + lid;
            if !pager.is_resident(id) && *cursor + pager.slot_bytes() > half_bytes {
                return Ok(i); // half full — caller rotates and continues from here
            }
            let ring_off = half_base + *cursor;
            *cursor += match bank {
                Some(bytes) => {
                    let off = lid as usize * stride;
                    let slice = bytes.get(off..off + stride).ok_or_else(|| {
                        be("moe pager: expert id out of range for this layer's bank")
                    })?;
                    pager.touch_staged(rec, ring.as_ref(), ring_off, id, slice, scan)?
                }
                // Tier below, under the SAME policy the arena above uses for this batch: a
                // full-set prefill sweep inserts cold, a routed decode touch inserts MRU.
                None => {
                    let host = host
                        .as_ref()
                        .ok_or_else(|| be(format!("moe pager: expert {id} has no host tier")))?;
                    if host.caches() {
                        let insert = if scan { Insert::Cold } else { Insert::Mru };
                        let pin = host.pin(id, insert)?;
                        pager.touch_staged(rec, ring.as_ref(), ring_off, id, &pin, scan)?
                    } else {
                        // Unified memory: nothing is cached below, so read the expert straight
                        // into the ring rather than through an arena that does not exist.
                        pager.touch_staged_read(rec, ring.as_ref(), ring_off, id, host, scan)?
                    }
                }
            };
        }
        Ok(local_ids.len())
    }

    /// Freeze `buf_id`'s layer LUT window — `n_expert` slot indices starting at its `layer_base`,
    /// copied from the pool's host mirror into the tape at `*tape_cursor` — and return the tape
    /// word offset the layer's dispatches pass as `lut_base` (`lut[base + local_id]`). Must be
    /// called AFTER every `stage_role` call for that (layer, role) batch completed (the
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
            // The tier below, when there is one. Its READS are the line that matters: an arena miss
            // this absorbed never touched the disk, and nothing else here tells those two apart.
            if let Some(h) = &p.host {
                let hs = h.stats();
                tracing::info!(
                    "[moe pager]   host/{}: {} slots={} reads={} {:.2}GB from disk",
                    p.role.name(),
                    stats_suffix(&hs.pager),
                    h.n_slots(),
                    hs.reads,
                    hs.bytes_read as f64 / 1e9,
                );
            }
        }
    }

    /// Per-pool `(role, VRAM residency, host tier)` counters, in pool order — the MoE twin of
    /// [`DensePagerSession::pool_stats`], and for the same reason: a tier transition that never
    /// runs is indistinguishable from one that works without these.
    pub fn pool_stats(
        &self,
    ) -> Vec<(
        Role,
        PagerStats,
        Option<infr_core::hostpager::HostPagerStats>,
    )> {
        self.pools
            .iter()
            .map(|p| (p.role, p.pager.stats(), p.host.as_ref().map(|h| h.stats())))
            .collect()
    }
}

/// `VulkanBackend::moe_pager`'s field type — a `Mutex` since `touch_role` mutates the LRU/arena and
/// the adapter calls it from `execute_static` (`&VulkanBackend`, not `&mut`).
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
    /// Pinned upload ring total bytes (two fence-rotated halves); `0` = [`ring_bytes`]'s
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
        let ring_half_bytes = (ring_total / 2).max(max_slot);
        let ring = vk.alloc_uninit(2 * ring_half_bytes, BufferUsage::Staging)?;
        Ok(Self {
            pools,
            sources: HashMap::new(),
            ring,
            ring_half_bytes,
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
        if !pool.pager.is_resident(id) && *cursor + pool.spec.slot_bytes > *ring_half_bytes {
            return Ok(None); // half full — caller rotates and re-calls
        }
        pool.pager.begin_batch();
        // Pass the source through by reference — `schedule_staged` derefs mmap segments via
        // `expert_bytes` and pins a host block in place, so neither tier materializes a copy here.
        let (slot, consumed) = pool.pager.schedule_staged(
            rec,
            ring.as_ref(),
            half_base + *cursor,
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
