//! ROCm/HIP paged MoE expert cache (Slice 33) — the HIP twin of `infr_vulkan::pager`'s
//! `MoePagerSession`, kept deliberately simpler because the ROCm `Op::MoeFfn` executor already
//! ROUTES ON THE HOST (it reads the router logits back and picks the top-k experts in Rust before
//! dispatching any expert GEMV — see `exec.rs`). That host readback means the pager needs none of
//! the Vulkan version's device-LUT / tape / fenced-ring machinery: the host already knows exactly
//! which experts a token wants, so it can page each one in and hand the expert GEMV a raw device
//! pointer to the slot it landed in.
//!
//! # Design
//! Expert weight banks stay in HOST memory (a zero-copy `Arc` view into the mmap'd GGUF — never
//! uploaded to VRAM). Per `(role, per-expert byte size)` there is one arena POOL: a contiguous
//! VRAM buffer of `n_slots` uniform `slot_bytes`-sized slots, plus an `infr_core::pager::Pager`
//! LRU that maps a global expert `BlockId` (`layer_base + local_id`) to a slot. On a miss the
//! selected expert's `slot_bytes` of RAW quant bytes are copied H2D into a free/evicted slot; the
//! existing native `moe_ffn_expert_*` / int8 `moe_*_i8_*` kernels then decode that slot in place,
//! exactly as they decode a resident bank — so a MoE model whose expert banks exceed VRAM fits in
//! a slot budget a fraction of their total size, at the quant footprint (NOT the ~3.5× f16 cache).
//!
//! # Why `(role, slot_bytes)` pools and not one per role
//! Every block sharing an arena must have the same byte size (fixed slot offsets). Uniform-dtype
//! roles yield one pool; a mixed-dtype role (unsloth-dynamic quants bump a subset of layers'
//! banks to a wider format — Qwen3-30B down mixes Q4_K/Q6_K) splits into one pool per byte size.
//! A fused gate_up bank pages under `Role::Gate` as a double-width slot (the model then has no
//! `Role::Up` pool). Every pool shares the SAME global block-id space (`layer * n_expert +
//! local`), so a pool simply never resolves ids of layers that live in another pool.
//!
//! # Eviction / within-batch safety
//! Classic LRU (`infr_core::pager::Pager::touch`) with the per-`(layer, role)` batch epoch: the
//! executor opens a batch per MoeFfn op before touching that layer's experts, so all experts a
//! single op reads are eviction-protected from each other. The seam sizes every pool with at
//! least `n_expert` slots (the batched-prefill worst case — one op may touch every expert of a
//! layer), so the sizing floor the `Pager` requires always holds.
//!
//! # Overlap (Slice 36) — why a copy stream + pinned ring + per-slot events
//! Slice 33 issued every miss's H2D `hipMemcpyAsync` on the SAME (compute) stream the expert GEMV
//! then ran on, and straight out of the GGUF mmap (PAGEABLE host memory). Both are slow: a
//! pageable async copy degenerates to a synchronous staging copy inside the driver, and even a
//! true async one on the compute stream serializes copy→GEMV→copy→GEMV with zero overlap. This
//! module now pipelines page-ins. A dedicated **copy stream** carries the H2D fills, decoupled
//! from compute. Each miss first `memcpy`s the expert's bytes out of the mmap into a **pinned**
//! (page-locked) ring region (a true async DMA needs a pinned source), then `hipMemcpyAsync`s that
//! region into the arena slot on the copy stream and records a per-fill **event**. The compute
//! stream `hipStreamWaitEvent`s that event before the expert GEMV, so the GEMV waits ONLY on its
//! own fill — expert i's GEMV overlaps expert i+1's copy, and a resident (hit) expert runs
//! immediately (compute-stream FIFO orders a later same-op hit after the earlier miss's wait, so a
//! hit needs no wait of its own).
//!
//! The pinned ring / event pool are reused per MoeFfn op ([`RocmMoePager::begin_paged_op`] resets
//! the in-flight cursor). Reuse is safe because the executor's router-logit readback
//! `hipStreamSynchronize`s the compute stream once per layer BEFORE the expert loop — draining
//! every prior op's waits (and therefore its copies) — and an op that pages more than
//! `MAX_INFLIGHT` misses drains the compute stream mid-loop before wrapping. Missing pinned memory
//! / copy stream (allocation failure) falls back to the Slice-33 in-line compute-stream copy —
//! correct, just slow.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use infr_core::error::Result;
use infr_core::pager::{Pager, PagerStats, Resolution};

use crate::backend::RocmBuffer;
use crate::ffi::{self, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};

/// Terse local shorthand for the shared backend-error constructor.
use infr_core::error::backend as be;

/// One paged expert role. A FUSED gate_up bank registers under `Gate` (double-width slot); a
/// fused model then has no `Up` sources. A role with mixed per-expert byte sizes across layers
/// spans several pools — the `(role, slot_bytes)` pair, not the role alone, names a pool.
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

/// Stable identity of a registered paged tensor — the placeholder buffer's device pointer, which
/// `hipMalloc` keeps fixed for the buffer's whole lifetime (the model's `SeamWeights` never frees
/// it until the session ends). The executor recovers the same value from the buffer bound at the
/// `gate_exps`/`up_exps`/`down_exps` tensor and looks the source up by it.
pub fn buffer_identity(b: &dyn infr_core::backend::Buffer) -> usize {
    let rb = b
        .as_any()
        .downcast_ref::<RocmBuffer>()
        .expect("rocm pager: buffer_identity on a non-RocmBuffer");
    rb.ptr as usize
}

/// Force the deref-to-trait-object FIRST so the inner `AsRef<[u8]>::as_ref` (not `Arc`'s own
/// `AsRef`) is the resolved impl — same guard as `infr_vulkan::pager::expert_bytes`.
fn expert_bytes(arc: &Arc<dyn AsRef<[u8]> + Send + Sync>) -> &[u8] {
    let inner: &(dyn AsRef<[u8]> + Send + Sync) = &**arc;
    inner.as_ref()
}

/// Where one paged layer's whole per-role expert bank lives: a zero-copy `Arc` view into the GGUF
/// mmap, the byte stride of ONE expert within it, and this layer's base into the role's global
/// block-id space (`layer_index * n_expert`).
pub struct ExpertSource {
    pub bytes: Arc<dyn AsRef<[u8]> + Send + Sync>,
    pub stride_bytes: usize,
    pub layer_base: u32,
}

/// One pool's spec: slot counts are independent per pool. Computed by the seam (budget-driven,
/// floored at `n_expert` so one op's full expert set is always simultaneously resident).
pub struct MoePoolSpec {
    pub role: Role,
    pub slot_bytes: usize,
    pub n_slots: usize,
}

/// Fixed layout for [`RocmMoePager::new`] — sizes every arena up front, before any tensor is
/// registered (the seam installs the session, then registers each paged `_exps` tensor as it
/// walks the weights).
pub struct MoePagerLayout {
    /// Total distinct experts nameable per pool's block-id space = `n_paged_layers * n_expert`.
    pub n_blocks: usize,
    pub pools: Vec<MoePoolSpec>,
}

/// One arena pool: a VRAM slot arena + its LRU. Every block in it shares `slot_bytes`.
struct Pool {
    role: Role,
    slot_bytes: usize,
    n_slots: usize,
    /// `n_slots * slot_bytes` device buffer of uniform slots.
    arena: RocmBuffer,
    pager: Pager,
}

impl Pool {
    /// Device pointer to `slot`'s base within the arena.
    fn slot_ptr(&self, slot: u32) -> *mut c_void {
        unsafe { (self.arena.ptr as *mut u8).add(slot as usize * self.slot_bytes) as *mut c_void }
    }
}

/// Max async page-in copies kept in flight per MoeFfn op before the pager drains the compute
/// stream and recycles the pinned ring / event pool. Decode routes `n_used * 3` (gate/up/down)
/// misses per layer at most (Qwen3-30B: 8*3 = 24), so this never bites the decode path; only a
/// giant batched-prefill layer (up to `n_expert * 3` distinct misses) wraps, and then a mid-loop
/// `hipStreamSynchronize` keeps the ring/event reuse correct at a modest cost. Also caps the
/// pinned footprint at `MAX_INFLIGHT * max_slot_bytes`.
const MAX_INFLIGHT: usize = 64;

/// A pinned (page-locked, device-mapped) host ring of `MAX_INFLIGHT` uniform regions, each big
/// enough for the largest pool's slot, plus one HIP event per region. The overlap engine: a miss
/// stages its bytes mmap→region[i], DMAs region[i]→arena slot on the copy stream, and records
/// event[i]; the compute stream waits event[i] before the GEMV. `None` if pinned allocation or the
/// copy-stream create failed at construction (the pager then falls back to the in-line copy).
struct CopyEngine {
    /// Dedicated H2D copy stream (distinct from the compute stream the GEMVs run on).
    copy_stream: ffi::hipStream_t,
    /// Page-locked host staging (`hipHostMalloc`) — freed via `host_ptr` in `Drop`.
    pinned: RocmBuffer,
    /// Byte size of one ring region (>= every pool's `slot_bytes`).
    region_bytes: usize,
    /// One ordering event per ring region.
    events: [ffi::hipEvent_t; MAX_INFLIGHT],
    /// Misses issued so far in the CURRENT op (index into the ring / event pool). Reset by
    /// [`RocmMoePager::begin_paged_op`]; wraps (with a compute drain) past `MAX_INFLIGHT`.
    inflight: usize,
}

/// One model's whole paged-MoE session: the `(role, slot_bytes)` arena pools + the map from a
/// bound placeholder buffer's identity to its expert source. Lives on the `RocmBackend`; `None`
/// for every non-paged model (zero cost, zero behavior change on the resident path).
pub struct RocmMoePager {
    pools: Vec<Pool>,
    /// `buffer_identity(placeholder)` -> (role, pool index, this layer's expert source).
    sources: HashMap<usize, (Role, usize, ExpertSource)>,
    /// Compute stream (GEMVs run here; also where cross-stream waits are enqueued).
    stream: ffi::hipStream_t,
    /// Copy-stream + pinned-ring overlap engine (`None` = in-line fallback; see [`CopyEngine`]).
    copy: Option<CopyEngine>,
    print_stats: bool,
}

// The pager owns device buffers + an opaque HIP stream handle — Send/Sync like `RocmBackend`.
unsafe impl Send for RocmMoePager {}
unsafe impl Sync for RocmMoePager {}

impl RocmMoePager {
    pub fn new(layout: MoePagerLayout, stream: ffi::hipStream_t) -> Result<Self> {
        let mut pools = Vec::with_capacity(layout.pools.len());
        let mut max_slot = 0usize;
        for spec in &layout.pools {
            if spec.n_slots == 0 {
                return Err(be("rocm moe pager: a pool needs at least one slot"));
            }
            // Zero-init (calloc contract): a slot is always fully written before its first read
            // (the miss copy fills exactly `slot_bytes`), so this is belt-and-suspenders, paid
            // once at load.
            let arena = RocmBuffer::try_alloc(spec.n_slots * spec.slot_bytes, stream)?;
            pools.push(Pool {
                role: spec.role,
                slot_bytes: spec.slot_bytes,
                n_slots: spec.n_slots,
                arena,
                pager: Pager::new(spec.n_slots),
            });
            max_slot = max_slot.max(spec.slot_bytes);
        }
        // Best-effort overlap engine; a failure to pin host memory or make a copy stream degrades
        // to the in-line copy path, never an error (paging still works, just serial).
        let copy = CopyEngine::try_new(max_slot);
        Ok(Self {
            pools,
            sources: HashMap::new(),
            stream,
            copy,
            print_stats: std::env::var_os("INFR_PAGER_STATS").is_some(),
        })
    }

    /// Reset the overlap engine's in-flight cursor at the start of one MoeFfn (layer) op — call
    /// once, alongside the per-pool [`Self::begin_batch`]es and BEFORE the first
    /// [`Self::ensure_slot`]. The ring regions / events it recycles are all free by the time they
    /// are actually reused: the executor `hipStreamSynchronize`s the compute stream (router-logit
    /// readback) between this call and the expert loop, draining every prior op's copies. No-op on
    /// the in-line fallback.
    pub fn begin_paged_op(&mut self) {
        if let Some(c) = self.copy.as_mut() {
            c.inflight = 0;
        }
    }

    /// Register one paged layer's `role` tensor. Picks the pool by `(role, source.stride_bytes)`;
    /// errors if the layout has no matching pool (a seam sizing bug — the layout enumeration and
    /// this registration must derive the slot size from the same tensor bytes).
    pub fn register(&mut self, role: Role, buf_id: usize, source: ExpertSource) -> Result<()> {
        let pool = self
            .pools
            .iter()
            .position(|p| p.role == role && p.slot_bytes == source.stride_bytes)
            .ok_or_else(|| {
                be(format!(
                    "rocm moe pager: no ({:?}, {} B/expert) pool in the layout for this tensor",
                    role, source.stride_bytes,
                ))
            })?;
        self.sources.insert(buf_id, (role, pool, source));
        Ok(())
    }

    /// Whether `buf_id` is a registered paged tensor of `role` — the executor's per-`MoeFfn`
    /// check before diverting to the paged pointer path.
    pub fn is_paged(&self, role: Role, buf_id: usize) -> bool {
        self.sources.get(&buf_id).is_some_and(|(r, ..)| *r == role)
    }

    /// Open a touch batch on `buf_id`'s pool — call once per (layer, role) MoeFfn op, BEFORE the
    /// first [`Self::ensure_slot`] of that op, so every expert the op reads is eviction-protected
    /// from the op's own later touches (the `Pager` within-batch invariant).
    pub fn begin_batch(&mut self, buf_id: usize) -> Result<()> {
        let (_, pool, _) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("rocm moe pager: begin_batch on an unregistered buffer"))?;
        self.pools[*pool].pager.begin_batch();
        Ok(())
    }

    /// Ensure the `local_id`-th expert of `buf_id`'s layer is resident and return the device
    /// pointer to its slot. On a miss the expert's `slot_bytes` of raw quant bytes are paged H2D
    /// into the freed/evicted slot on the dedicated COPY stream, and the COMPUTE stream is made to
    /// wait (via a recorded event) on exactly that fill — so the expert GEMV the caller dispatches
    /// next overlaps the NEXT expert's copy (see the module doc). A miss never overwrites a slot an
    /// in-flight GEMV of THIS op still reads (the pool holds ≥ `n_expert` slots, so no expert
    /// touched earlier in the op is evicted). Without the overlap engine, the copy runs in-line on
    /// the compute stream (the Slice-33 behavior).
    pub fn ensure_slot(&mut self, role: Role, buf_id: usize, local_id: u32) -> Result<*mut c_void> {
        let (r, pool_idx, src) = self
            .sources
            .get(&buf_id)
            .ok_or_else(|| be("rocm moe pager: ensure_slot on an unregistered buffer"))?;
        debug_assert_eq!(*r, role, "ensure_slot: role/buffer mismatch");
        let pool_idx = *pool_idx;
        let stride = src.stride_bytes;
        let global = src.layer_base + local_id;
        let off = local_id as usize * stride;
        // Borrow the source bytes before taking the pool mutably (disjoint borrows via the map).
        let slice_ptr = {
            let bytes = expert_bytes(&src.bytes);
            let slice = bytes.get(off..off + stride).ok_or_else(|| {
                be("rocm moe pager: expert id out of range for this layer's bank")
            })?;
            slice.as_ptr()
        };
        let stream = self.stream;
        let pool = &mut self.pools[pool_idx];
        debug_assert_eq!(stride, pool.slot_bytes, "expert stride != pool slot size");
        match pool.pager.touch(global) {
            // A hit's slot is already filled by an EARLIER touch; on the overlap path that touch
            // enqueued its wait on the compute stream (FIFO), so this hit's later GEMV is already
            // ordered after the fill — no wait needed here.
            Resolution::Hit { slot } => Ok(pool.slot_ptr(slot)),
            Resolution::Miss { slot, .. } => {
                let dst = pool.slot_ptr(slot);
                match self.copy.as_mut() {
                    Some(engine) => engine.fill(stream, dst, slice_ptr, stride),
                    None => Self::fill_inline(stream, dst, slice_ptr, stride),
                }?;
                Ok(dst)
            }
        }
    }

    /// Slice-33 fallback: a stream-ordered async H2D directly on the compute stream (no overlap).
    fn fill_inline(
        stream: ffi::hipStream_t,
        dst: *mut c_void,
        src: *const u8,
        stride: usize,
    ) -> Result<()> {
        let rc = unsafe {
            ffi::hipMemcpyAsync(
                dst,
                src as *const c_void,
                stride,
                HIP_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        if rc != HIP_SUCCESS {
            return Err(be(format!(
                "rocm moe pager: hipMemcpyAsync H2D slot: rc={rc}"
            )));
        }
        Ok(())
    }

    /// `INFR_PAGER_STATS=1`: print each pool's hit/miss/eviction counters. Called after
    /// generation finishes.
    pub fn print_stats_if_enabled(&self) {
        if !self.print_stats {
            return;
        }
        for p in &self.pools {
            let s: PagerStats = p.pager.stats();
            eprintln!(
                "[rocm moe pager] {}/{:.1}MB: hits={} misses={} evictions={} hit_rate={:.3} \
                 slots={}",
                p.role.name(),
                p.slot_bytes as f64 / 1e6,
                s.hits,
                s.misses,
                s.evictions,
                s.hit_rate(),
                p.n_slots,
            );
        }
    }
}

impl CopyEngine {
    /// Build the overlap engine: a copy stream, a pinned ring of `MAX_INFLIGHT` regions of
    /// `region_bytes` (>= the largest slot), and one event per region. Returns `None` (in-line
    /// fallback) if any HIP call fails — paging must not hard-error just because host memory can't
    /// be pinned.
    fn try_new(max_slot_bytes: usize) -> Option<Self> {
        // A/B lever: force the Slice-33 in-line compute-stream copy (no overlap) for measurement.
        if std::env::var_os("INFR_ROCM_PAGER_NOOVERLAP").is_some() {
            return None;
        }
        let region_bytes = max_slot_bytes.max(4);
        let mut copy_stream: ffi::hipStream_t = std::ptr::null_mut();
        if unsafe { ffi::hipStreamCreate(&mut copy_stream) } != HIP_SUCCESS {
            return None;
        }
        // Pinned host staging (page-locked → a true async DMA source). `hipHostMalloc` via the
        // shared spill allocator; not zero-inited (each region is fully overwritten before its DMA).
        let pinned = match RocmBuffer::try_alloc_host(region_bytes * MAX_INFLIGHT, false) {
            Ok(b) => b,
            Err(_) => {
                unsafe { ffi::hipStreamDestroy(copy_stream) };
                return None;
            }
        };
        let mut events = [std::ptr::null_mut(); MAX_INFLIGHT];
        for (i, ev) in events.iter_mut().enumerate() {
            if unsafe { ffi::hipEventCreateWithFlags(ev, ffi::HIP_EVENT_DISABLE_TIMING) }
                != HIP_SUCCESS
            {
                // Roll back the events created so far + the stream/pinned buffer.
                for e in events.iter().take(i) {
                    unsafe { ffi::hipEventDestroy(*e) };
                }
                unsafe { ffi::hipStreamDestroy(copy_stream) };
                return None;
            }
        }
        Some(Self {
            copy_stream,
            pinned,
            region_bytes,
            events,
            inflight: 0,
        })
    }

    /// Stage + DMA one miss with overlap: `memcpy` the expert bytes into the next pinned region,
    /// `hipMemcpyAsync` that region into the arena slot on the copy stream, record the region's
    /// event, and make the compute `stream` wait on it (so the caller's GEMV overlaps later
    /// copies). On the `MAX_INFLIGHT`-th miss of an op the compute stream is drained first — that
    /// completes every earlier wait (and therefore every earlier copy), freeing the whole ring for
    /// reuse (see [`RocmMoePager::begin_paged_op`]).
    fn fill(
        &mut self,
        stream: ffi::hipStream_t,
        dst: *mut c_void,
        src: *const u8,
        stride: usize,
    ) -> Result<()> {
        debug_assert!(
            stride <= self.region_bytes,
            "expert stride exceeds pinned region"
        );
        if self.inflight >= MAX_INFLIGHT {
            // Wrap: drain the compute stream so every in-flight fill's wait (hence its copy) has
            // completed before we overwrite region 0 again. Rare — only huge prefill layers.
            if unsafe { ffi::hipStreamSynchronize(stream) } != HIP_SUCCESS {
                return Err(be(
                    "rocm moe pager: hipStreamSynchronize (ring wrap) failed",
                ));
            }
            self.inflight = 0;
        }
        let i = self.inflight;
        let region = unsafe { (self.pinned.host_ptr as *mut u8).add(i * self.region_bytes) };
        // mmap → pinned (parallel; the single-thread copy is the staging bottleneck, exactly as in
        // `infr_vulkan::pager::par_copy_to_mapped`).
        let src_slice = unsafe { std::slice::from_raw_parts(src, stride) };
        par_copy_to_pinned(src_slice, region);
        let ev = self.events[i];
        let rc = unsafe {
            ffi::hipMemcpyAsync(
                dst,
                region as *const c_void,
                stride,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.copy_stream,
            )
        };
        if rc != HIP_SUCCESS {
            return Err(be(format!(
                "rocm moe pager: hipMemcpyAsync H2D slot (copy stream): rc={rc}"
            )));
        }
        if unsafe { ffi::hipEventRecord(ev, self.copy_stream) } != HIP_SUCCESS {
            return Err(be("rocm moe pager: hipEventRecord (copy stream) failed"));
        }
        // Compute stream waits ONLY on this fill — the GEMV enqueued next overlaps later copies.
        if unsafe { ffi::hipStreamWaitEvent(stream, ev, 0) } != HIP_SUCCESS {
            return Err(be(
                "rocm moe pager: hipStreamWaitEvent (compute stream) failed",
            ));
        }
        self.inflight += 1;
        Ok(())
    }
}

impl Drop for CopyEngine {
    fn drop(&mut self) {
        // Drain outstanding copies before tearing the stream/pinned buffer down.
        unsafe { ffi::hipStreamSynchronize(self.copy_stream) };
        for &ev in &self.events {
            unsafe { ffi::hipEventDestroy(ev) };
        }
        unsafe { ffi::hipStreamDestroy(self.copy_stream) };
        // `self.pinned` (RocmBuffer) frees its `hipHostMalloc` allocation in its own Drop.
    }
}

/// Parallel `memcpy` of one expert's raw quant bytes from the (pageable) GGUF mmap into a pinned
/// staging region. A single-thread copy of a multi-MB expert is the staging bottleneck (see
/// `infr_vulkan::pager::par_copy_to_mapped`); chunking across the rayon pool recovers the host DRAM
/// bandwidth so the copy overlaps the previous expert's DMA + GEMV rather than serializing them.
fn par_copy_to_pinned(src: &[u8], dst: *mut u8) {
    use rayon::prelude::*;
    const CHUNK: usize = 4 << 20; // 4 MiB
    if src.len() <= CHUNK {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let dst_addr = dst as usize; // Send-able; each chunk writes a disjoint range.
    src.par_chunks(CHUNK).enumerate().for_each(|(i, c)| unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), (dst_addr + i * CHUNK) as *mut u8, c.len());
    });
}
