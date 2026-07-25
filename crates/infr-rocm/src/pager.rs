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

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use infr_core::error::{Error, Result};
use infr_core::pager::{Pager, PagerStats, Resolution};

use crate::backend::RocmBuffer;
use crate::ffi::{self, HIP_MEMCPY_HOST_TO_DEVICE, HIP_SUCCESS};

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

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

/// One model's whole paged-MoE session: the `(role, slot_bytes)` arena pools + the map from a
/// bound placeholder buffer's identity to its expert source. Lives on the `RocmBackend`; `None`
/// for every non-paged model (zero cost, zero behavior change on the resident path).
pub struct RocmMoePager {
    pools: Vec<Pool>,
    /// `buffer_identity(placeholder)` -> (role, pool index, this layer's expert source).
    sources: HashMap<usize, (Role, usize, ExpertSource)>,
    stream: ffi::hipStream_t,
    print_stats: bool,
}

// The pager owns device buffers + an opaque HIP stream handle — Send/Sync like `RocmBackend`.
unsafe impl Send for RocmMoePager {}
unsafe impl Sync for RocmMoePager {}

impl RocmMoePager {
    pub fn new(layout: MoePagerLayout, stream: ffi::hipStream_t) -> Result<Self> {
        let mut pools = Vec::with_capacity(layout.pools.len());
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
        }
        Ok(Self {
            pools,
            sources: HashMap::new(),
            stream,
            print_stats: std::env::var_os("INFR_PAGER_STATS").is_some(),
        })
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
    /// pointer to its slot. On a miss the expert's `slot_bytes` of raw quant bytes are copied H2D
    /// into the freed/evicted slot with a stream-ordered async copy — enqueued on the SAME stream
    /// the caller then dispatches the expert GEMV on, so the copy always completes before the
    /// kernel reads it, and never overwrites a slot an in-flight GEMV of THIS op still reads (the
    /// pool holds ≥ `n_expert` slots, so no expert touched earlier in the op is evicted).
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
        let pool = &mut self.pools[pool_idx];
        debug_assert_eq!(stride, pool.slot_bytes, "expert stride != pool slot size");
        match pool.pager.touch(global) {
            Resolution::Hit { slot } => Ok(pool.slot_ptr(slot)),
            Resolution::Miss { slot, .. } => {
                let dst = pool.slot_ptr(slot);
                let rc = unsafe {
                    ffi::hipMemcpyAsync(
                        dst,
                        slice_ptr as *const c_void,
                        stride,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                        self.stream,
                    )
                };
                if rc != HIP_SUCCESS {
                    return Err(be(format!(
                        "rocm moe pager: hipMemcpyAsync H2D slot: rc={rc}"
                    )));
                }
                Ok(dst)
            }
        }
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
