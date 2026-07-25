//! CPU reference backend — a correctness-first interpreter of the backend-agnostic
//! [`infr_core`] compute [`Graph`]. Projection matmuls and attention use rayon for multi-core
//! parallelism; QK/PV inner loops use an 8-accumulator dot for AVX autovectorization.
//! Weights are read **zero-copy from the GGUF mmap** (no `memcpy`, no owned RAM): the bulk
//! projection weights (`Op::Linear`) are dequantized one row at a time straight from the mapping
//! inside the dot, so 12B / MoE models cost only their on-disk size in page cache. Only the tiny
//! norm weights are dequant-cached; the model writes (KV / conv / recurrent state, per-step IO) use
//! small owned buffers. It exists to (a) run every model without a GPU and (b) serve as the oracle
//! the GPU backends are validated against.
#![allow(clippy::needless_range_loop)]

pub mod kvquant;
mod pool;
pub mod turbo;

mod kernels;
mod moe;
mod quant;
mod repack;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, GraphPlan, Plan};
use infr_core::error::Result;
use infr_core::exec::Provision;
use infr_core::graph::{AttnMask, Graph, MoeGating, Op};
use infr_core::tensor::{DType, TensorId};
use infr_gguf::dequant::dequant_block;
use infr_gguf::TensorBytes;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use kernels::{
    act_fn, dot, dot_bf16, dot_f16, vec_dot_iq1m, vec_dot_iq1m_batch, vec_dot_iq1s,
    vec_dot_iq1s_batch, vec_dot_iq2s, vec_dot_iq2s_batch, vec_dot_iq2xs, vec_dot_iq2xs_batch,
    vec_dot_iq2xxs, vec_dot_iq2xxs_batch, vec_dot_iq3s, vec_dot_iq3s_batch, vec_dot_iq3xxs,
    vec_dot_iq3xxs_batch, vec_dot_iq4nl_32_batch, vec_dot_iq4xs, vec_dot_iq4xs_batch,
    vec_dot_mxfp4_32_batch, vec_dot_nvfp4_batch, vec_dot_q2_0_batch, vec_dot_q2k,
    vec_dot_q2k_batch, vec_dot_q3k, vec_dot_q3k_batch, vec_dot_q4_0_32_batch,
    vec_dot_q4_1_32_batch, vec_dot_q4k, vec_dot_q4k_batch, vec_dot_q4k_batch2, vec_dot_q4k_batch8,
    vec_dot_q5_0_32_batch, vec_dot_q5_1_32_batch, vec_dot_q5k, vec_dot_q5k_batch, vec_dot_q6k,
    vec_dot_q6k_batch, vec_dot_q8_0, vec_dot_q8_0_32_batch, vec_dot_q8_0_batch, vec_dot_tq1_0,
    vec_dot_tq1_0_batch, vec_dot_tq2_0, vec_dot_tq2_0_batch,
};
use moe::{expert_acts_kind, expert_gemm_range, ActsKind, ExpertActs};
use quant::{quantize_q8, quantize_q8_32, Q8x32, Q8};
#[cfg(target_arch = "x86_64")]
use repack::{q4k_pack, q6k_gemm_group, q6k_pack, Q6kPack};
use repack::{Q4kPack, Repack6CacheState, RepackCacheState};

/// Dequantized-weight cache: `(buffer address, byte-len, dtype) -> Arc<f32>`. See `weight_cache`.
type WeightCache = HashMap<(usize, usize, DType), Arc<Vec<f32>>>;

// ─── Q8_0 integer dot kernels ─────────────────────────────────────────────────
//
// Q8_0 weight layout: 34 bytes / 32 elements.  Bytes 0..2 = f16 scale `d`; bytes 2..34 = i8 qs.
// Activation comes in as a `Q8` super-block (256 elems), so one super-block covers 8 Q8_0 weight
// blocks.  Since both weight and activation are i8, we use the llama.cpp sign trick:
// `maddubs(abs(qw), sign(qw)·qx)` = `Σ qw[i]·qx[i]` without overflow into i16.

// ─── Q5_K integer dot kernels ─────────────────────────────────────────────────
//
// Q5_K block layout (176 bytes / 256 elems):
//   [f16 d][f16 dmin][scales[12]][qh[32]][ql[128]]
// q5 = (ql_nibble) | (((qh[l] >> bit) & 1) << 4)  ∈ 0..31  (UNSIGNED → maddubs works directly)
// Dot formula: d·sc·Σ(q5·qx) − dmin·m·Σqx  — identical structure to Q4_K.
// `q8.bsums` provides Σqx per 32-elem sub-block (precomputed in quantize_q8).

// ─── Batched dot kernels (prefill: m > 1) ────────────────────────────────────
//
// Each `vec_dot_qXk_batch(row, q8s, in_f, out)` is equivalent to calling
// `vec_dot_qXk(row, &q8s[r], in_f)` for every r, but decodes the weight row
// ONCE and loops over the m token activations with the pre-decoded data.
//
// Bit-identity guarantee: the per-token f32 result equals the single-token
// kernel exactly (integer dots have no rounding, same accumulation grouping).

// ── Q4_K batch ────────────────────────────────────────────────────────────────

// ── Q4_K 2-row tiled batch ────────────────────────────────────────────────────
//
// Process TWO output rows simultaneously so the Q8 activation data (loaded from
// L3 cache) is reused for both dots instead of loaded twice. This halves the L3
// bandwidth for Q8 reads which is the dominant bottleneck during large-batch prefill.
//
// `out_a` and `out_b` receive the dots for row `row_a` and `row_b` respectively.
// Bit-identical: each `out_x[r]` equals `vec_dot_q4k(row_x, &q8s[r], in_f)`.

// ── Q4_K 8-row tiled batch ────────────────────────────────────────────────────
//
// Process EIGHT output rows simultaneously: the Q8 activation zmm is loaded ONCE
// per (block, nibble-pair) and reused across all 8 row dots. This is 4× less
// activation traffic than the 2-row path and 8× less than the single-row path.
//
// `outs[i][r]` == `vec_dot_q4k(rows[i], &q8s[r], in_f)` — bit-identical to the
// single-token kernel (same per-block accumulation order; tiling only changes which
// rows are computed together, not the per-(row,token) arithmetic).

// ── Q6_K batch ────────────────────────────────────────────────────────────────

// ── Q8_0 batch ────────────────────────────────────────────────────────────────

// ── Q5_K batch ────────────────────────────────────────────────────────────────

// ─── Native 32-block int8 dot kernels (Q8_0 / Q5_0 at their OWN block size) ──────────────────────
//
// The K-quant/Q8_0 batch kernels above all group activations into 256-element super-blocks (`Q8`),
// which requires `in_f % 256 == 0` — true for most projections but NOT MoE's `down` projection,
// whose `in_f = n_ff_exp` can be any multiple of 32 (e.g. DiffusionGemma's 704 = 22×32). Q8_0/Q5_0's
// own native block is 32 elements, so this activation quantizes at THAT granularity instead —
// scalar only (no SIMD; this path is memory/allocation-bound, not compute-bound, so a plain scalar
// loop over int8 bytes already beats dequantizing the whole row to f32 first).

/// A host buffer. Weights are **mapped** — a zero-copy [`TensorBytes`] view straight into the GGUF
/// mmap (read-only, no `memcpy`, no owned RAM). Everything the model writes (KV / conv / recurrent
/// state, per-step IO) is **owned** — a plain byte vec behind a `Mutex` (so `&dyn Buffer` stays
/// `Send + Sync` and writes are safe). `&dyn Buffer` reads go through [`CpuBuffer::read`].
pub enum CpuBuffer {
    Owned(Mutex<Vec<u8>>),
    Mapped(TensorBytes),
}

/// A uniform read view over either storage (a `MutexGuard` for owned, the slice for mapped); both
/// deref to `[u8]`.
enum CpuRead<'a> {
    Owned(std::sync::MutexGuard<'a, Vec<u8>>),
    Mapped(&'a TensorBytes),
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl std::ops::Deref for CpuRead<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            CpuRead::Owned(g) => g,
            CpuRead::Mapped(t) => t,
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl CpuBuffer {
    /// Read view of the bytes (zero-copy for mapped weights; mutex guard for owned buffers).
    fn read(&self) -> CpuRead<'_> {
        match self {
            CpuBuffer::Owned(m) => CpuRead::Owned(m.lock().unwrap()),
            CpuBuffer::Mapped(t) => CpuRead::Mapped(t),
        }
    }
    /// Mutable owned storage; panics for mapped (read-only) weights.
    fn owned(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        match self {
            CpuBuffer::Owned(m) => m.lock().unwrap(),
            CpuBuffer::Mapped(_) => {
                panic!("cpu backend: write to a mapped (read-only) weight buffer")
            }
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Buffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        match self {
            CpuBuffer::Owned(m) => m.lock().unwrap().len(),
            CpuBuffer::Mapped(t) => t.len(),
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Default)]
pub struct CpuBackend {
    /// Dequantized-weight cache keyed by the bound buffer's `(address, byte-len, dtype)` (weights
    /// are bound the same every step, so dequant once and reuse). Only the small norm weights
    /// (`RmsNorm` / `QkNorm`) land here — the large `Op::Linear` weights are streamed row-by-row
    /// instead (see that arm), so this never holds the whole model in f32. Keying on len+dtype (not
    /// the raw address alone) guards a reused `CpuBackend` (serve model reload): a freed weight's
    /// address can be reallocated to a *different* weight, and a bare-address key would then hand
    /// back the previous model's f32 — the len/dtype discriminates the two.
    weight_cache: Mutex<WeightCache>,
    /// (layer, expert)-granular repack cache for the interleaved-x8 Q4_K GEMM ([`Q4kPack`]):
    /// keyed by the expert weight slice's (address, length) — stable for the mmap'd/upload-once
    /// weight buffers this backend binds (same lifetime argument as `weight_cache`). ggml pays
    /// its `block_q4_Kx8` repack once at LOAD; this pays it once per (expert, session) instead
    /// of once per CALL. Byte-budgeted (`INFR_CPU_REPACK_MB`, default 4096): over budget, packs
    /// are built transient and not inserted. The `usize` is the current cached-bytes total.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    repack_cache: Mutex<RepackCacheState>,
    /// Q6_K sibling of `repack_cache` (same keying and budget env; separate accounting) — holds
    /// e.g. the tied Q6_K lm_head's ~740 MB pack, built once per session.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    repack6_cache: Mutex<Repack6CacheState>,
    /// Persistent spin-pool for the op interpreter's parallel loops (see `pool.rs`, threadpool
    /// restructure phase 2). Built on first use so backends constructed for tests/tiny work
    /// never spawn threads. MoeFfn's nested per-expert fan-out stays on rayon (phase 3).
    pool: std::sync::OnceLock<pool::SpinPool>,
    /// REFERENCE mode (see [`CpuBackend::reference`]): decode every quantized weight to f32 and
    /// take an f32 dot, instead of the production int8-quantized-activation kernels. Off by
    /// default — production CPU inference keeps the fast (and llama.cpp-equivalent) int8 path.
    reference: bool,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// A backend in REFERENCE mode: every quantized `Op::Linear` (dense and MoE-expert) decodes
    /// its weight to f32 and takes a plain f32 dot, so the only error left is the weight's own
    /// quantization. The production path instead quantizes the ACTIVATIONS to int8 (the
    /// llama.cpp `vec_dot_*_q8_K` regime) — same arithmetic ggml uses, and ~2-3 orders of
    /// magnitude cheaper, but it carries ~4e-3 relative error on every dtype. That error is
    /// invisible at Q4_K+ yet flips greedy tokens at Q2_K, which is why a GPU-vs-CPU token
    /// comparison must use THIS mode as its oracle rather than the production kernels (see
    /// `gpu_seam_matches_cpu_qwen3_q2k`): the GPU's f32 dequant GEMV is ~1e-7 against the host
    /// decode, so the int8 CPU leg was the drifting one.
    ///
    /// Reference mode is for oracles and triage only — it is far slower and it is NOT what the
    /// CPU goldens pin (`cpu_golden_qwen3` = the production int8 path).
    pub fn reference() -> Self {
        Self {
            reference: true,
            ..Self::default()
        }
    }

    fn pool(&self) -> &pool::SpinPool {
        self.pool.get_or_init(pool::SpinPool::new)
    }

    /// Fetch-or-build the [`Q4kPack`] for one weight bank slice (see `repack_cache`'s doc).
    /// SAFETY-adjacent note: callers only reach this from the VNNI ilv path, which implies AVX2
    /// for the pack expansion.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn q4k_pack_for(&self, w: &[u8], in_f: usize, out_f: usize) -> Arc<Q4kPack> {
        let key = (w.as_ptr() as usize, w.len());
        if let Some(p) = self.repack_cache.lock().unwrap().0.get(&key) {
            return p.clone();
        }
        let pack = Arc::new(unsafe { q4k_pack(w, in_f, out_f) });
        let budget_mb: usize = std::env::var("INFR_CPU_REPACK_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let bytes = pack.bytes();
        // Re-check under the lock: another thread may have inserted `key` between our miss and here.
        let mut guard = self.repack_cache.lock().unwrap();
        repack::cache_insert_if_absent(&mut guard, key, pack, bytes, budget_mb * 1024 * 1024)
    }

    /// Fetch-or-build the [`Q6kPack`] for one weight bank slice — `q4k_pack_for`'s Q6_K sibling.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn q6k_pack_for(&self, w: &[u8], in_f: usize, out_f: usize) -> Arc<Q6kPack> {
        let key = (w.as_ptr() as usize, w.len());
        if let Some(p) = self.repack6_cache.lock().unwrap().0.get(&key) {
            return p.clone();
        }
        let pack = Arc::new(q6k_pack(w, in_f, out_f));
        let budget_mb: usize = std::env::var("INFR_CPU_REPACK_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let bytes = pack.bytes();
        // Re-check under the lock: another thread may have inserted `key` between our miss and here.
        let mut guard = self.repack6_cache.lock().unwrap();
        repack::cache_insert_if_absent(&mut guard, key, pack, bytes, budget_mb * 1024 * 1024)
    }

    /// Wrap a zero-copy GGUF mmap view as a read-only weight buffer (no allocation, no `memcpy`).
    pub fn map_weight(&self, bytes: TensorBytes) -> Box<dyn Buffer> {
        Box::new(CpuBuffer::Mapped(bytes))
    }

    /// Fetch-or-build a dequantized weight from `weight_cache` under its `(addr, len, dtype)` key.
    /// The key's len+dtype ensure a reload that re-uses a freed weight's address can't collide
    /// with the stale entry (see `weight_cache`'s doc). `bytes` is dequantized per `key.2` on a miss.
    pub(crate) fn cached_weight(&self, key: (usize, usize, DType), bytes: &[u8]) -> Arc<Vec<f32>> {
        if let Some(w) = self.weight_cache.lock().unwrap().get(&key) {
            return w.clone();
        }
        let w = Arc::new(bytes_to_f32(bytes, key.2));
        self.weight_cache.lock().unwrap().insert(key, w.clone());
        w
    }
}

/// Reinterpret raw buffer bytes as `f32` values per `dtype` (dequantizing quant/f16/bf16, widening
/// integer position tensors). The universal "read a tensor's value on the host".
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        DType::I32 => bytemuck::cast_slice::<u8, i32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DType::U32 => bytemuck::cast_slice::<u8, u32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        // F16 / Bf16 / all quant + codebook types go through the shared host dequant.
        other => dequant_block(other, bytes).expect("cpu backend: host dequant"),
    }
}

/// Borrow two *distinct* elements of a slice as `(&a, &mut b)` — used by `Op::Copy`/`CopyStrided`
/// to read the source in place instead of cloning the whole tensor. Panics if `a == b` (callers
/// handle the aliasing case separately with a temp).
fn borrow_two_mut<T>(v: &mut [T], a: usize, b: usize) -> (&T, &mut T) {
    assert_ne!(a, b, "borrow_two_mut requires distinct indices");
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        (&lo[a], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(a);
        (&hi[0], &mut lo[b])
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn cpu_buf(b: &dyn Buffer) -> &CpuBuffer {
    b.as_any()
        .downcast_ref::<CpuBuffer>()
        .expect("cpu backend: buffer is not a CpuBuffer (mixed backends?)")
}

/// Dequantize the first `need` elements of a Q8_0-block buffer (34 B / 32 elems, y = d*q).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dequant_prefix_q8_0(bytes: &[u8], need: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(need);
    for b in 0..need.div_ceil(32) {
        let off = b * 34;
        let d = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        for i in 0..32 {
            if out.len() == need {
                break;
            }
            out.push(d * (bytes[off + 2 + i] as i8) as f32);
        }
    }
    out
}

/// Device-side stochastic sampling reference (see `Op::Sample`) — IDENTICAL order of operations to
/// the host `sample_logits` (top-k select desc, softmax(temp), nucleus, CDF walk) with the uniform
/// draw `uu` factored out. `top_k == 0` is the "disable top-k" sentinel → NO truncation (`k == len`);
/// any `top_k >= 1` truncates to `min(top_k, len)`. Returns the sampled token index.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_token(logits: &[f32], uu: f32, top_k: usize, temp: f32, top_p: f32) -> u32 {
    // top_k == 0 disables truncation; otherwise cap at the vocab length. Guarding here keeps the
    // `select_nth_unstable_by(k - 1)` below from underflowing when top_k == 0 (k would be 0).
    let k = if top_k == 0 {
        logits.len()
    } else {
        top_k.min(logits.len())
    };
    let cmp = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    if k < logits.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp);
    let maxl = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - maxl) / temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let mut cum = 0.0;
    let mut cutoff = probs.len();
    for (j, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = j + 1;
            break;
        }
    }
    let total: f32 = probs[..cutoff].iter().sum();
    let r = uu * total;
    let mut tok = idx[cutoff - 1] as u32;
    let mut acc = 0.0;
    for j in 0..cutoff {
        acc += probs[j];
        if r <= acc {
            tok = idx[j] as u32;
            break;
        }
    }
    tok
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: "cpu-reference".into(),
            // The scalar interpreter has no GPU-kernel tier choice; these per-type feature flags
            // don't gate any CPU op handler (they select Vulkan/Metal shaders). f16/i8 math is
            // available in the interpreter; the matrix/dot-primitive shapes are N-A → None/off.
            f16: true,
            coopmat_f16: None,
            f8: false,
            coopmat_f8: None,
            i8: true,
            i8_dot: false,
            coopmat_i8: None,
            bf16: false,
            coopmat_bf16: None,
            subgroup_min: 0,
            subgroup_max: 0,
            sg_pref: 0, // no subgroup pinning on the scalar interpreter
            vendor_intel: false,
            // Device-class fields: there is no GPU here, so no watchdog and nothing to bound.
            // (`integrated` means "an iGPU whose submits must stay under a TDR" — NOT "shares
            // system memory", which is what `unified_memory` below already says.)
            integrated: false,
            compute_units: 0,
            buffer_device_address: false, // CPU interpreter has no device-address concept
            max_shared_memory_bytes: u32::MAX, // scalar interpreter: no shared-memory tiling
            unified_memory: true,
            // The interpreter reads the baked `pos`/`kv_len` from the graph ops, so the decode graph
            // must be rebuilt per token — no record-once replay.
            decode_replay: false,
            combined_gu: false,
            embed_gather: true,
            gpu_sample: true,
            argmax_rows: true,
            argmax_prob: true,
            // The interpreter implements `Op::GatedRmsNorm` (below) for direct-op parity tests,
            // but leaves the capability false so the runner keeps emitting the split
            // `QkNorm`→`GatedAct` pair here — no GPU barrier to save on a scalar interpreter, and
            // it keeps the CPU oracle's op sequence matching the pre-fusion baseline.
            gated_rmsnorm: false,
            // The interpreter's WriteKv/Attention honor ring KV caches (row = pos % cap_rows), so
            // the runner may window-size SWA layers' caches here.
            kv_swa_ring: true,
        }
    }

    fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        Ok(Box::new(CpuBuffer::Owned(Mutex::new(vec![
            0u8;
            bytes.max(4)
        ]))))
    }

    fn alloc_uninit(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Debug: poison with 0xFF (= NaN as f32) so a read-before-write surfaces loudly in the CPU
        // tests/oracle instead of silently working. Release: the Vec is zeroed anyway (no CPU perf
        // win to skip it), so stay safe.
        let fill = if cfg!(debug_assertions) { 0xFFu8 } else { 0u8 };
        Ok(Box::new(CpuBuffer::Owned(Mutex::new(vec![
            fill;
            bytes.max(4)
        ]))))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let mut d = cpu_buf(dst).owned();
        d[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let s = cpu_buf(src).read();
        dst.copy_from_slice(&s[..dst.len()]);
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(GraphPlan::boxed(graph))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<GraphPlan>()
            .expect("cpu backend: plan is not a GraphPlan")
            .graph;

        // f32 working store for every Input/Internal/Output handle (weights are read on demand:
        // norms via the small dequant cache, `Op::Linear` weights streamed row-by-row).
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); g.tensors.len()];
        // KV-cache tensors (the `cache` of `WriteKv`, the `k_cache`/`v_cache` of `Attention`) are
        // accessed straight from their bound buffers — `WriteKv` writes one row, `Attention` reads
        // `kv_len` rows. They're sized for the WHOLE context (`max_ctx`), so loading them into `vals`
        // (and writing them back) each token would cost O(max_ctx) memory traffic per token instead of
        // O(kv_len) — catastrophic at a large `max_new`. Skip the round-trip for them. Which tensors
        // are written in place is graph semantics, computed once by `Graph::in_place_inputs`; the
        // per-kind classification is the shared `infr_core::exec::Provision` (metal's interpreter
        // makes the identical calls).
        for (id, decl, p) in infr_core::exec::provisions(g) {
            let i = id.0 as usize;
            match p {
                Provision::Zero => vals[i] = vec![0f32; decl.desc.numel()],
                Provision::Load => {
                    let buf = bindings.get(id).expect("cpu backend: unbound Input");
                    let bytes = cpu_buf(buf).read();
                    vals[i] = bytes_to_f32(&bytes, decl.desc.dtype);
                }
                // A Weight (lazily dequantized in `weight()`) or an in-place KV Input.
                Provision::Skip => {}
            }
        }

        // Fetch a (cached) dequantized weight.
        let weight = |id: TensorId| -> Arc<Vec<f32>> {
            let buf = bindings.get(id).expect("cpu backend: unbound Weight");
            let addr = cpu_buf(buf) as *const CpuBuffer as usize;
            let key = (addr, buf.len_bytes(), g.desc(id).dtype);
            if let Some(w) = self.weight_cache.lock().unwrap().get(&key) {
                return w.clone();
            }
            let bytes = cpu_buf(buf).read();
            self.cached_weight(key, &bytes)
        };

        let prof_ops = std::env::var("INFR_PROF_OPS").is_ok();
        let mut op_times: HashMap<&'static str, f64> = HashMap::new();
        for op in &g.ops {
            let __t0 = if prof_ops {
                Some(std::time::Instant::now())
            } else {
                None
            };
            match *op {
                Op::RmsNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    dim,
                    eps,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    // Rows are independent (no cross-row reduction) — spin-pool over rows, same
                    // per-row math and order as before (bit-identical, just distributed).
                    let mut out = vec![0f32; rows * dim];
                    self.pool().for_chunks_mut(&mut out, dim, 4, &|r, orow| {
                        let xrow = &xs[r * dim..r * dim + dim];
                        let ss: f32 = (0..dim).map(|i| xrow[i] * xrow[i]).sum::<f32>() / dim as f32;
                        let s = 1.0 / (ss + eps).sqrt();
                        for i in 0..dim {
                            orow[i] = xrow[i] * s * ws[i];
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::RmsNormAdd {
                    x,
                    weight: w,
                    dst,
                    rows: _rows,
                    dim,
                    eps,
                } => {
                    let dim = dim as usize;
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let mut dst_vals = vals[dst.0 as usize].clone();
                    self.pool()
                        .for_chunks_mut(&mut dst_vals, dim, 4, &|r, drow| {
                            let xrow = &xs[r * dim..r * dim + dim];
                            let ss: f32 =
                                (0..dim).map(|i| xrow[i] * xrow[i]).sum::<f32>() / dim as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..dim {
                                drow[i] += xrow[i] * s * ws[i];
                            }
                        });
                    vals[dst.0 as usize] = dst_vals;
                }
                Op::Softmax {
                    x,
                    dst,
                    rows,
                    dim,
                    scale,
                    scale_buf,
                } => {
                    let (rows, dim) = (rows as usize, dim as usize);
                    // Perf (DiffusionGemma denoise, Vulkan) — see `Op::Softmax::scale_buf`'s doc.
                    // CPU never builds a graph with this set (the CPU denoise path keeps its host
                    // self-conditioning), but implements it generically rather than panicking.
                    let scale = match scale_buf {
                        Some(sb) => vals[sb.0 as usize][0],
                        None => scale,
                    };
                    let xs = &vals[x.0 as usize];
                    let mut out = vec![0f32; rows * dim];
                    self.pool().for_chunks_mut(&mut out, dim, 1, &|r, o| {
                        let row = &xs[r * dim..r * dim + dim];
                        let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
                        let mut denom = 0f32;
                        for (dst_v, &v) in o.iter_mut().zip(row) {
                            let e = (v * scale - mx).exp();
                            *dst_v = e;
                            denom += e;
                        }
                        let inv = 1.0 / denom;
                        for dst_v in o.iter_mut() {
                            *dst_v *= inv;
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::QkNorm {
                    x,
                    weight: w,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    eps,
                    ..
                } => {
                    let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                    let xs = &vals[x.0 as usize];
                    let ws = weight(w);
                    let mut out = vec![0f32; rows * nh * hd];
                    for r in 0..rows {
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..hd {
                                out[b + i] = xs[b + i] * s * ws[i];
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate). Reduction
                // structure is IDENTICAL to the `QkNorm` arm above (bit-stable normalization); the
                // gate multiply is the extra elementwise step `GatedAct(Silu)` would otherwise do
                // as a second op.
                Op::GatedRmsNorm {
                    x,
                    weight: w,
                    gate,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    eps,
                } => {
                    let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                    let xs = &vals[x.0 as usize];
                    let zs = &vals[gate.0 as usize];
                    let ws = weight(w);
                    let mut out = vec![0f32; rows * nh * hd];
                    for r in 0..rows {
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            let ss: f32 =
                                (0..hd).map(|i| xs[b + i] * xs[b + i]).sum::<f32>() / hd as f32;
                            let s = 1.0 / (ss + eps).sqrt();
                            for i in 0..hd {
                                let normed = xs[b + i] * s * ws[i];
                                let zv = zs[b + i];
                                let silu = zv / (1.0 + (-zv).exp());
                                out[b + i] = normed * silu;
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Linear {
                    x,
                    weight: w,
                    dst,
                    m,
                    in_f,
                    out_f,
                    w_off,
                } => {
                    let (m, in_f, out_f) = (m as usize, in_f as usize, out_f as usize);
                    let xs = &vals[x.0 as usize];
                    // Stream the (row-major [out_f, in_f]) weight one row at a time straight from the
                    // mmap, dequantizing inside the dot — no full f32 materialization. GGUF rows are
                    // block-aligned, so each row is an equal `bytes/out_f` slice. Output rows are
                    // independent → fan out over the 32 cores with rayon.
                    let buf = bindings.get(w).expect("cpu backend: unbound Weight");
                    let bytes = cpu_buf(buf).read();
                    let dt = g.desc(w).dtype;
                    // `w_off` (elements, row-aligned) selects a projection's rows inside a
                    // CONCATENATED weight (fused QKV): total rows = declared numel / in_f.
                    let total_rows = g.desc(w).numel() / in_f;
                    let bpr = bytes.len() / total_rows; // bytes per weight row
                    let row0 = w_off as usize / in_f;
                    let wbytes: &[u8] = &bytes[row0 * bpr..(row0 + out_f) * bpr];
                    let mut out = vec![0f32; m * out_f];
                    // REFERENCE mode (see `CpuBackend::reference`): decode the weight row to f32 and
                    // take a plain f32 dot, so the ONLY error is the weight's own quantization —
                    // the int8-activation kernels below carry ~4e-3 relative error, which flips
                    // greedy tokens at Q2_K. The float dtypes already take an exact dot, so they
                    // stay on their (identical, faster) production arms.
                    if self.reference && !matches!(dt, DType::F32 | DType::F16 | DType::Bf16) {
                        let mut out_t = vec![0f32; out_f * m];
                        self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                            let row = &wbytes[o * bpr..o * bpr + bpr];
                            let wf = bytes_to_f32(row, dt);
                            for r in 0..m {
                                chunk[r] = dot(&wf, &xs[r * in_f..r * in_f + in_f]);
                            }
                        });
                        self.pool().for_chunks_mut(&mut out, out_f, 1, &|r, orow| {
                            for (o, dst) in orow.iter_mut().enumerate() {
                                *dst = out_t[o * m + r];
                            }
                        });
                        vals[dst.0 as usize] = out;
                        continue;
                    }
                    // One token (decode) is the hot path. Dispatch on the weight dtype to the fastest
                    // per-row kernel: integer Q8×Q4_K/Q6_K dots (quantize the activation once), direct
                    // f16/bf16/f32 dots, else fall back to dequant-to-f32 + dot. All fan out over rows.
                    if m == 1 {
                        let xrow = &xs[..in_f];
                        // Q8_0 is a NATIVE 32-block weight, so its `in_f` can be a multiple of 32
                        // but not 256 (e.g. gemma3-1b's 1152-dim projections). The 256-superblock
                        // Q8 activation path silently drops the sub-256 tail, so route those to the
                        // native 32-block Q8x32 kernel instead. Every other quant routed through Q8
                        // has a 256-element weight block, so its `in_f` is always 256-aligned.
                        let q8_0_blk32 = dt == DType::Q8_0 && !in_f.is_multiple_of(256);
                        let q8 = (matches!(
                            dt,
                            DType::Q4K
                                | DType::Q6K
                                | DType::Q8_0
                                | DType::Q5K
                                | DType::Iq4Xs
                                | DType::Iq2S
                                | DType::Iq2Xs
                                | DType::Iq2Xxs
                                | DType::Iq1S
                                | DType::Iq1M
                                | DType::Iq3S
                                | DType::Iq3Xxs
                                | DType::Q2K
                                | DType::Q3K
                                | DType::Tq2_0
                                | DType::Tq1_0
                        ) && !q8_0_blk32)
                            .then(|| quantize_q8(xrow));
                        // Q4_0/Q4_1/IQ4_NL (and sub-256 Q8_0) use the native-32-block int8
                        // activation (Q8x32), not the 256-superblock Q8; quantize the single
                        // activation row once.
                        let q8_32: Option<[Q8x32; 1]> = (matches!(
                            dt,
                            DType::Q4_0
                                | DType::Q4_1
                                | DType::Q5_1
                                | DType::Iq4Nl
                                | DType::Mxfp4
                                | DType::Nvfp4
                                | DType::Q2_0
                        ) || q8_0_blk32)
                            .then(|| [quantize_q8_32(xrow)]);
                        // Spin-pool, 16 output rows per claimed task (decode's per-row dot is
                        // ~µs-scale; per-row claims would be all cursor contention).
                        self.pool().for_chunks_mut(&mut out, 1, 16, &|o, dst_o| {
                            let row = &wbytes[o * bpr..o * bpr + bpr];
                            dst_o[0] = match dt {
                                DType::Q4K => vec_dot_q4k(row, q8.as_ref().unwrap(), in_f),
                                DType::Q6K => vec_dot_q6k(row, q8.as_ref().unwrap(), in_f),
                                DType::Q8_0 => {
                                    if q8_0_blk32 {
                                        vec_dot_q8_0_32_batch(
                                            row,
                                            q8_32.as_ref().unwrap(),
                                            in_f,
                                            dst_o,
                                        );
                                        return;
                                    }
                                    vec_dot_q8_0(row, q8.as_ref().unwrap(), in_f)
                                }
                                DType::Q5K => vec_dot_q5k(row, q8.as_ref().unwrap(), in_f),
                                DType::Q2K => vec_dot_q2k(row, q8.as_ref().unwrap(), in_f),
                                DType::Q3K => vec_dot_q3k(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq4Xs => vec_dot_iq4xs(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq2S => vec_dot_iq2s(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq2Xs => vec_dot_iq2xs(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq2Xxs => vec_dot_iq2xxs(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq1S => vec_dot_iq1s(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq1M => vec_dot_iq1m(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq3S => vec_dot_iq3s(row, q8.as_ref().unwrap(), in_f),
                                DType::Iq3Xxs => vec_dot_iq3xxs(row, q8.as_ref().unwrap(), in_f),
                                DType::Tq2_0 => vec_dot_tq2_0(row, q8.as_ref().unwrap(), in_f),
                                DType::Tq1_0 => vec_dot_tq1_0(row, q8.as_ref().unwrap(), in_f),
                                DType::Q4_0 => {
                                    vec_dot_q4_0_32_batch(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q4_1 => {
                                    vec_dot_q4_1_32_batch(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q5_1 => {
                                    vec_dot_q5_1_32_batch(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Iq4Nl => {
                                    vec_dot_iq4nl_32_batch(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Mxfp4 => {
                                    vec_dot_mxfp4_32_batch(
                                        row,
                                        q8_32.as_ref().unwrap(),
                                        in_f,
                                        dst_o,
                                    );
                                    return;
                                }
                                DType::Q2_0 => {
                                    vec_dot_q2_0_batch(row, q8_32.as_ref().unwrap(), in_f, dst_o);
                                    return;
                                }
                                DType::Nvfp4 => {
                                    vec_dot_nvfp4_batch(row, q8_32.as_ref().unwrap(), in_f, dst_o);
                                    return;
                                }
                                DType::F32 => dot(bytemuck::cast_slice(row), xrow),
                                DType::F16 => dot_f16(row, xrow),
                                DType::Bf16 => dot_bf16(row, xrow),
                                _ => dot(&bytes_to_f32(row, dt), xrow),
                            };
                        });
                    } else {
                        // PREFILL (m > 1): parallelize over output rows (one weight row per task).
                        // For quant types, use the batched dot kernels: the weight row is decoded
                        // ONCE per output row (inside the batch fn), then the integer dot is
                        // repeated across all m token activations — amortising the expensive
                        // nibble/bit unpacking that the single-token path was redoing m times.
                        //
                        // Layout: out[r * out_f + o].  We accumulate into a transposed buffer
                        // out_t[o * m + r] (contiguous in o-major order) so each parallel chunk
                        // owns a contiguous slice of m floats, then scatter into out at the end.
                        // Parallel: at m=256 canvas rows this collect was ~0.7 ms of SERIAL work
                        // per Linear (31 threads idle) — rows are independent, order preserved by
                        // the indexed collect, bit-identical.
                        // Q8_0 is a native 32-block weight: a sub-256 `in_f` (e.g. gemma3-1b's
                        // 1152-dim projections) must take the 32-block Q8x32 path below, not the
                        // 256-superblock Q8 batch kernel (which drops the sub-256 tail). Skip
                        // building the wasteful 256-block activation in that case.
                        let q8_0_blk32 = dt == DType::Q8_0 && !in_f.is_multiple_of(256);
                        let q8s: Vec<Q8> = if !q8_0_blk32
                            && matches!(
                                dt,
                                DType::Q4K
                                    | DType::Q6K
                                    | DType::Q8_0
                                    | DType::Q5K
                                    | DType::Iq4Xs
                                    | DType::Iq2S
                                    | DType::Iq2Xs
                                    | DType::Iq2Xxs
                                    | DType::Iq1S
                                    | DType::Iq1M
                                    | DType::Iq3S
                                    | DType::Iq3Xxs
                                    | DType::Q2K
                                    | DType::Q3K
                                    | DType::Tq2_0
                                    | DType::Tq1_0
                            ) {
                            self.pool()
                                .collect(m, &|r| quantize_q8(&xs[r * in_f..r * in_f + in_f]))
                        } else {
                            Vec::new()
                        };
                        let mut out_t = vec![0f32; out_f * m];
                        // For Q4_K, use 8-row tiling: each rayon task handles 8 consecutive
                        // output rows and loads the Q8 activation zmm ONCE per (block, nibble-pair),
                        // reusing it across all 8 weight rows. This is 4× less activation traffic
                        // than the 2-row path and 8× less than the single-row path. Remainder rows
                        // (out_f % 8) fall through to the 2-row tile then the 1-row batch.
                        if dt == DType::Q4K && out_f >= 8 {
                            let groups8 = out_f / 8;
                            let rem = out_f % 8;
                            let (g8_t, rest_t) = out_t.split_at_mut(groups8 * 8 * m);
                            // 8-row groups across the spin-pool.
                            self.pool().for_chunks_mut(g8_t, 8 * m, 1, &|g, dc| {
                                let o = g * 8;
                                let (r0, rest) = dc.split_at_mut(m);
                                let (r1, rest) = rest.split_at_mut(m);
                                let (r2, rest) = rest.split_at_mut(m);
                                let (r3, rest) = rest.split_at_mut(m);
                                let (r4, rest) = rest.split_at_mut(m);
                                let (r5, rest) = rest.split_at_mut(m);
                                let (r6, r7) = rest.split_at_mut(m);
                                vec_dot_q4k_batch8(
                                    [
                                        &wbytes[o * bpr..o * bpr + bpr],
                                        &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr],
                                        &wbytes[(o + 2) * bpr..(o + 2) * bpr + bpr],
                                        &wbytes[(o + 3) * bpr..(o + 3) * bpr + bpr],
                                        &wbytes[(o + 4) * bpr..(o + 4) * bpr + bpr],
                                        &wbytes[(o + 5) * bpr..(o + 5) * bpr + bpr],
                                        &wbytes[(o + 6) * bpr..(o + 6) * bpr + bpr],
                                        &wbytes[(o + 7) * bpr..(o + 7) * bpr + bpr],
                                    ],
                                    &q8s,
                                    in_f,
                                    [r0, r1, r2, r3, r4, r5, r6, r7],
                                );
                            });
                            // Remainder: up to 7 rows → 2-row pairs, then at most 1 odd tail.
                            let pairs_rem = rem / 2;
                            let (g2_t, odd_t) = rest_t.split_at_mut(pairs_rem * 2 * m);
                            if pairs_rem > 0 {
                                self.pool().for_chunks_mut(g2_t, 2 * m, 1, &|pair, dc| {
                                    let o = groups8 * 8 + pair * 2;
                                    let (chunk_a, chunk_b) = dc.split_at_mut(m);
                                    let row_a = &wbytes[o * bpr..o * bpr + bpr];
                                    let row_b = &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr];
                                    vec_dot_q4k_batch2(row_a, row_b, &q8s, in_f, chunk_a, chunk_b);
                                });
                            }
                            if rem % 2 != 0 {
                                let o = out_f - 1;
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q4k_batch(row, &q8s, in_f, odd_t);
                            }
                        } else if dt == DType::Q4K && out_f >= 2 {
                            // Small out_f < 8: fall back to 2-row tile.
                            let pairs = out_f / 2;
                            let (even_t, odd_t) = out_t.split_at_mut(pairs * 2 * m);
                            self.pool().for_chunks_mut(even_t, 2 * m, 1, &|pair, dc| {
                                let o = pair * 2;
                                let (chunk_a, chunk_b) = dc.split_at_mut(m);
                                let row_a = &wbytes[o * bpr..o * bpr + bpr];
                                let row_b = &wbytes[(o + 1) * bpr..(o + 1) * bpr + bpr];
                                vec_dot_q4k_batch2(row_a, row_b, &q8s, in_f, chunk_a, chunk_b);
                            });
                            if out_f % 2 != 0 {
                                let o = out_f - 1;
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q4k_batch(row, &q8s, in_f, odd_t);
                            }
                        } else if dt == DType::Q6K && out_f >= 8 && in_f.is_multiple_of(256) && {
                            // `is_x86_feature_detected!` fails to COMPILE on aarch64 even behind
                            // a runtime `cfg!` — it needs a real #[cfg] block (macOS CI trap).
                            #[cfg(target_arch = "x86_64")]
                            {
                                is_x86_feature_detected!("avx512bw")
                                    && is_x86_feature_detected!("avx512vnni")
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                false
                            }
                        } {
                            // Q6_K on the interleaved-x8 pack (the tied Q6_K lm_head is a 189
                            // GMAC GEMM every DG denoise step): 8 weight rows per activation
                            // pass instead of 1 — same activation-reuse trick Q4_K got.
                            #[cfg(target_arch = "x86_64")]
                            {
                                let pack = self.q6k_pack_for(wbytes, in_f, out_f);
                                let groups8 = out_f / 8;
                                let rem = out_f % 8;
                                let (g8_t, rest_t) = out_t.split_at_mut(groups8 * 8 * m);
                                self.pool().for_chunks_mut(g8_t, 8 * m, 1, &|g, dc| {
                                    // SAFETY: VNNI dispatch checked in the branch condition.
                                    unsafe { q6k_gemm_group(&pack.groups[g], pack.nb, &q8s, dc) };
                                });
                                if rem > 0 {
                                    self.pool().for_chunks_mut(rest_t, m, 1, &|i, chunk| {
                                        let o = groups8 * 8 + i;
                                        let row = &wbytes[o * bpr..o * bpr + bpr];
                                        vec_dot_q6k_batch(row, &q8s, in_f, chunk);
                                    });
                                }
                            }
                        } else if q8_0_blk32 {
                            // Q8_0 weight with a sub-256 `in_f` (multiple of 32 but not 256): the
                            // 256-superblock Q8 batch kernel would truncate the tail, so use the
                            // native 32-block Q8x32-activation batch kernel instead. `in_f % 32 == 0`
                            // always holds for a Q8_0 (32-block) weight.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q8_0_32_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Q5_0 && in_f.is_multiple_of(32) {
                            // Dense-layer Q5_0 (DG stores 16 of its 30 dense ffn_down weights as
                            // Q5_0): reuse the MoE path's Q8x32-activation batch kernel. This
                            // dtype previously fell through to the dequant+f32-dot fallback below
                            // (~11% of every DG denoise step in the samply profile) — the switch
                            // puts it on the same int8-quantized-activation regime every other
                            // quantized dtype in this dispatch already uses.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q5_0_32_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Q5_1 && in_f.is_multiple_of(32) {
                            // Q5_1 (affine sibling of Q5_0: `y = d·q5 + m`, 5-bit code + f16 `m`
                            // min, block stride 24): previously fell through to the dequant+f32
                            // fallback below. Reuse the Q8x32-activation batch kernel with the
                            // affine per-block form.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q5_1_32_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Q4_0 && in_f.is_multiple_of(32) {
                            // Q4_0 (same 32-block int8-activation regime as Q5_0, minus the 5th
                            // bit and with offset 8): previously fell through to the dequant+f32
                            // fallback below. Reuse the Q8x32-activation batch kernel.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q4_0_32_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Q4_1 && in_f.is_multiple_of(32) {
                            // Q4_1 (affine sibling of Q4_0: `y = d·q4 + m`, block stride 20):
                            // previously fell through to the dequant+f32 fallback below. Reuse the
                            // Q8x32-activation batch kernel with the affine per-block form.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q4_1_32_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Iq4Nl && in_f.is_multiple_of(32) {
                            // IQ4_NL (flat 32-block codebook cousin of IQ4_XS: ONE f16 `d` per
                            // 32-weight block, signed KVALUES codebook, no super-block scales):
                            // previously fell through to the dequant+f32 fallback below. Reuse the
                            // Q8x32-activation batch kernel with the codebook signed-dot.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_iq4nl_32_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Mxfp4 && in_f.is_multiple_of(32) {
                            // MXFP4 (flat 32-block codebook, structurally identical to IQ4_NL but
                            // with a 1-byte E8M0 exponent scale `d = 2^(e−128)` per block and the
                            // KVALUES_MXFP4 codebook): previously fell through to the dequant+f32
                            // fallback below. Reuse the Q8x32-activation batch kernel.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_mxfp4_32_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Q2_0 && in_f.is_multiple_of(64) {
                            // Q2_0 ("Bonsai ternary", 2-bit linear-offset quant on a 64-weight
                            // block): previously fell through to the dequant+f32 fallback below.
                            // Reuse the Q8x32-activation batch kernel — each 64-weight block maps to
                            // two consecutive 32-element activation sub-blocks. Guarded on
                            // `is_multiple_of(64)` (the block size) so the sub-block count is even.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_q2_0_batch(row, &q8s32, in_f, chunk);
                            });
                        } else if dt == DType::Nvfp4 && in_f.is_multiple_of(64) {
                            // NVFP4 (E2M1 codebook + four per-16-element UE4M3 sub-block scales on a
                            // 64-weight block): previously fell through to the dequant+f32 fallback
                            // below. Reuse the Q8x32-activation batch kernel — each 64-weight block
                            // maps to two consecutive 32-element activation sub-blocks, and each of
                            // those spans two NVFP4 sub-blocks with distinct weight scales. Guarded on
                            // `is_multiple_of(64)` (the block size) so the sub-block count is even.
                            let q8s32: Vec<Q8x32> = self
                                .pool()
                                .collect(m, &|r| quantize_q8_32(&xs[r * in_f..r * in_f + in_f]));
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                vec_dot_nvfp4_batch(row, &q8s32, in_f, chunk);
                            });
                        } else {
                            // Grain 8: at lm_head shape this loop is 262k one-row chunks; per-row
                            // cursor claims (or rayon's per-item splitting) are real overhead.
                            self.pool().for_chunks_mut(&mut out_t, m, 8, &|o, chunk| {
                                let row = &wbytes[o * bpr..o * bpr + bpr];
                                match dt {
                                    DType::Q4K => vec_dot_q4k_batch(row, &q8s, in_f, chunk),
                                    DType::Q6K => vec_dot_q6k_batch(row, &q8s, in_f, chunk),
                                    DType::Q8_0 => vec_dot_q8_0_batch(row, &q8s, in_f, chunk),
                                    DType::Q5K => vec_dot_q5k_batch(row, &q8s, in_f, chunk),
                                    DType::Q2K => vec_dot_q2k_batch(row, &q8s, in_f, chunk),
                                    DType::Q3K => vec_dot_q3k_batch(row, &q8s, in_f, chunk),
                                    DType::Iq4Xs => vec_dot_iq4xs_batch(row, &q8s, in_f, chunk),
                                    DType::Iq2S => vec_dot_iq2s_batch(row, &q8s, in_f, chunk),
                                    DType::Iq2Xs => vec_dot_iq2xs_batch(row, &q8s, in_f, chunk),
                                    DType::Iq2Xxs => vec_dot_iq2xxs_batch(row, &q8s, in_f, chunk),
                                    DType::Iq1S => vec_dot_iq1s_batch(row, &q8s, in_f, chunk),
                                    DType::Iq1M => vec_dot_iq1m_batch(row, &q8s, in_f, chunk),
                                    DType::Iq3S => vec_dot_iq3s_batch(row, &q8s, in_f, chunk),
                                    DType::Iq3Xxs => vec_dot_iq3xxs_batch(row, &q8s, in_f, chunk),
                                    DType::Tq2_0 => vec_dot_tq2_0_batch(row, &q8s, in_f, chunk),
                                    DType::Tq1_0 => vec_dot_tq1_0_batch(row, &q8s, in_f, chunk),
                                    DType::F32 => {
                                        let w32: &[f32] = bytemuck::cast_slice(row);
                                        for r in 0..m {
                                            chunk[r] = dot(w32, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                    DType::F16 => {
                                        for r in 0..m {
                                            chunk[r] = dot_f16(row, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                    DType::Bf16 => {
                                        for r in 0..m {
                                            chunk[r] =
                                                dot_bf16(row, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                    _ => {
                                        // Dequant the weight row ONCE, reuse across all m tokens.
                                        let wf = bytes_to_f32(row, dt);
                                        for r in 0..m {
                                            chunk[r] = dot(&wf, &xs[r * in_f..r * in_f + in_f]);
                                        }
                                    }
                                }
                            });
                        }
                        // Transpose out_t[o * m + r] → out[r * out_f + o], parallel over the m output
                        // rows (each gathers its out_f values from the o-major temp). The serial
                        // version was ~20% of the matvec at large out_f × m.
                        self.pool().for_chunks_mut(&mut out, out_f, 1, &|r, orow| {
                            for (o, dst) in orow.iter_mut().enumerate() {
                                *dst = out_t[o * m + r];
                            }
                        });
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Rope {
                    x,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    freq_factors,
                    ..
                } => {
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    // Borrow the inputs (no clone) — `out` is the single owned copy we mutate and
                    // then move into `vals[dst]`; the borrows end before that write (NLL). If
                    // dst == x this still holds: `out` is an independent Vec.
                    let xs = &vals[x.0 as usize];
                    let pos = &vals[positions.0 as usize];
                    let ff = freq_factors.map(|f| &vals[f.0 as usize]);
                    let mut out = xs.clone(); // dims beyond rope_dim pass through unchanged
                                              // Op::Rope is the no-qk-norm (llama-family) rotation: INTERLEAVED pairs
                                              // (2p, 2p+1) — llama.cpp's ROPE_TYPE_NORM, matching the Vulkan `rope` kernel
                                              // and the bespoke fused attn_in. (QkNormRope is the NEOX split-half rotation
                                              // used by qwen/gemma; the two styles are NOT interchangeable.)
                    let hf = rd / 2;
                    for r in 0..rows {
                        let p0 = pos[r];
                        for h in 0..nh {
                            let b = (r * nh + h) * hd;
                            for p in 0..hf {
                                let (i0, i1) = (2 * p, 2 * p + 1);
                                let mut ang = p0 * theta.powf(-2.0 * p as f32 / rd as f32);
                                if let Some(ff) = &ff {
                                    ang /= ff[p];
                                }
                                let (s, c) = (ang.sin(), ang.cos());
                                let a = xs[b + i0];
                                let bb = xs[b + i1];
                                out[b + i0] = a * c - bb * s;
                                out[b + i1] = a * s + bb * c;
                            }
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::QkNormRope {
                    x,
                    weight: w,
                    positions,
                    dst,
                    rows,
                    n_head,
                    head_dim,
                    rope_dim,
                    theta,
                    eps,
                    freq_factors,
                    x_stride,
                    ..
                } => {
                    let (rows, nh, hd, rd) = (
                        rows as usize,
                        n_head as usize,
                        head_dim as usize,
                        rope_dim as usize,
                    );
                    let raw = &vals[x.0 as usize];
                    let x_stride = x_stride as usize;
                    // If input is interleaved (stride > 0), extract packed query data per row.
                    let xs: Vec<f32> = if x_stride > 0 {
                        let mut packed = vec![0f32; rows * nh * hd];
                        for r in 0..rows {
                            let row_base = r * x_stride;
                            for h in 0..nh {
                                let src = row_base + h * 2 * hd; // head h query starts here
                                let dst = (r * nh + h) * hd;
                                packed[dst..dst + hd].copy_from_slice(&raw[src..src + hd]);
                            }
                        }
                        packed
                    } else {
                        raw.clone()
                    };
                    let xs = &xs;
                    let ws = weight(w);
                    let pos = &vals[positions.0 as usize];
                    let ff = freq_factors.map(|f| vals[f.0 as usize].clone());
                    let mut out = vec![0f32; rows * nh * hd];
                    let hf = rd / 2;
                    // Parallel over the m rows. Within a row the RoPE angles depend only on the
                    // position (not the head), so precompute (cos,sin) per rope index ONCE per row and
                    // reuse across all heads — the powf/sin/cos were the bulk and were redone nh×.
                    self.pool()
                        .for_chunks_mut(&mut out, nh * hd, 1, &|r, orow| {
                            let p0 = pos[r];
                            let cs: Vec<(f32, f32)> = (0..hf)
                                .map(|p| {
                                    let mut ang = p0 * theta.powf(-2.0 * p as f32 / rd as f32);
                                    if let Some(ff) = &ff {
                                        ang /= ff[p];
                                    }
                                    (ang.cos(), ang.sin())
                                })
                                .collect();
                            let xr = &xs[r * nh * hd..r * nh * hd + nh * hd];
                            for h in 0..nh {
                                let b = h * hd;
                                let ss: f32 =
                                    (0..hd).map(|i| xr[b + i] * xr[b + i]).sum::<f32>() / hd as f32;
                                let s = 1.0 / (ss + eps).sqrt();
                                for i in 0..hd {
                                    orow[b + i] = xr[b + i] * s * ws[i];
                                }
                                for p in 0..hf {
                                    let (i0, i1) = (p, p + hf);
                                    let (c, sn) = cs[p];
                                    let a = orow[b + i0];
                                    let bb = orow[b + i1];
                                    orow[b + i0] = a * c - bb * sn;
                                    orow[b + i1] = a * sn + bb * c;
                                }
                            }
                        });
                    vals[dst.0 as usize] = out;
                }
                Op::WriteKv {
                    src,
                    cache,
                    rows,
                    row_stride,
                    pos,
                } => {
                    let (rows, rs, pos) = (rows as usize, row_stride as usize, pos as usize);
                    let s = &vals[src.0 as usize];
                    // Write the new row(s) straight into the persistent KV buffer — only `rows` rows
                    // touched, not the whole `max_ctx`-sized cache. The cache dtype (f16 to match the
                    // GPU and halve memory, or f32) is read from the graph; cast on write.
                    let buf = bindings.get(cache).expect("cpu backend: unbound KV cache");
                    let mut d = cpu_buf(buf).owned();
                    // SWA ring cache: position p lands at row p % cap_rows — the ring only ever
                    // recycles rows whose positions the sliding-window mask already excludes (see
                    // the seam runner's ring sizing), so this is exact. A write crossing the wrap
                    // splits into two contiguous segments `(src_row, dst_row, n_rows)`; on a
                    // full-context cache pos < cap_rows and segment 0 is the old single write.
                    let cap_rows = g.desc(cache).numel() / rs.max(1);
                    let pos_r = if cap_rows > 0 { pos % cap_rows } else { pos };
                    let segs: [(usize, usize, usize); 2] =
                        if cap_rows > 0 && pos_r + rows > cap_rows {
                            let r1 = cap_rows - pos_r;
                            [(0, pos_r, r1), (r1, 0, rows - r1)]
                        } else {
                            [(0, pos_r, rows), (0, 0, 0)]
                        };
                    for &(sr, dr, nr) in segs.iter().filter(|&&(_, _, nr)| nr > 0) {
                        let s = &s[sr * rs..sr * rs + nr * rs];
                        let base = dr * rs;
                        let n = nr * rs;
                        match g.desc(cache).dtype {
                            DType::F16 => {
                                let df: &mut [u16] = bytemuck::cast_slice_mut(&mut d);
                                for i in 0..n {
                                    df[base + i] = half::f16::from_f32(s[i]).to_bits();
                                }
                            }
                            DType::Q8_0 => {
                                // Q8_0 blocks (34 B / 32 elems): d = amax/127 (stored f16), q =
                                // round(x/d) via `round_ties_even`. NOTE: llama.cpp's
                                // quantize_row_q8_0 uses `roundf` (half-away-from-zero) and the GPU
                                // `store_q8.comp` uses GLSL `round()` (halfway is impl-defined), so
                                // this backend deliberately differs on exact-halfway activations.
                                // That is intentional and safe: no path pins the CPU KV Q8 bytes
                                // bit-for-bit — the CPU/GPU KV Q8 tests are all coherence/tolerance
                                // checks (kv_q8_roundtrip err<0.02, gpu_seam_kv_q8_coherent), never
                                // exact-byte goldens — and halfway ties are vanishingly rare, so the
                                // <0.5 LSB difference never changes a decoded token.
                                // `base`/`n` are element counts and rows are 32-aligned (the runner
                                // gates on it), so blocks never straddle a write.
                                debug_assert!(base % 32 == 0 && n % 32 == 0);
                                for b in 0..n / 32 {
                                    let src32 = &s[b * 32..b * 32 + 32];
                                    let amax = src32.iter().fold(0f32, |m, &v| m.max(v.abs()));
                                    let dq = amax / 127.0;
                                    let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
                                    let off = (base / 32 + b) * 34;
                                    let dh = half::f16::from_f32(dq).to_bits().to_le_bytes();
                                    d[off] = dh[0];
                                    d[off + 1] = dh[1];
                                    for (i, &v) in src32.iter().enumerate() {
                                        d[off + 2 + i] =
                                            (v * id).round_ties_even() as i32 as i8 as u8;
                                    }
                                }
                            }
                            DType::Bf16 => {
                                let df: &mut [u16] = bytemuck::cast_slice_mut(&mut d);
                                for i in 0..n {
                                    df[base + i] = half::bf16::from_f32(s[i]).to_bits();
                                }
                            }
                            dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4) => {
                                // TurboQuant: each 128-elem group (a head_dim slice) → one block
                                // (L2-norm + WHT + 2/3/4-bit PolarQuant). base/n are 128-aligned (the
                                // runner gates head_dim%128), so blocks never straddle a write.
                                debug_assert!(base % 128 == 0 && n % 128 == 0);
                                let bb = crate::turbo::block_bytes(dt);
                                let blk0 = base / 128;
                                for b in 0..n / 128 {
                                    let off = (blk0 + b) * bb;
                                    crate::turbo::quantize_block(
                                        dt,
                                        &s[b * 128..b * 128 + 128],
                                        &mut d[off..off + bb],
                                    );
                                }
                            }
                            dt if crate::kvquant::supported(dt) => {
                                // Mainline low-bit KV quants (q4_0/q4_1/q5_0/q5_1/iq4_nl): quantize the
                                // f32 activations into 32-elem blocks. base/n are 32-aligned (kv_align_ok).
                                debug_assert!(base % 32 == 0 && n % 32 == 0);
                                let bb = infr_gguf::nbytes(dt, 32);
                                let off = (base / 32) * bb;
                                crate::kvquant::quantize_row(
                                    dt,
                                    &s[..n],
                                    &mut d[off..off + n / 32 * bb],
                                );
                            }
                            _ => {
                                let df: &mut [f32] = bytemuck::cast_slice_mut(&mut d);
                                df[base..base + n].copy_from_slice(&s[..n]);
                            }
                        }
                    }
                }
                Op::Attention {
                    q,
                    k_cache,
                    v_cache,
                    dst,
                    rows,
                    kv_len,
                    n_head,
                    n_kv,
                    head_dim,
                    scale,
                    mask,
                    pos,
                } => {
                    let (rows, kv_len, nh, nkv, hd) = (
                        rows as usize,
                        kv_len as usize,
                        n_head as usize,
                        n_kv as usize,
                        head_dim as usize,
                    );
                    let qs = &vals[q.0 as usize];
                    // K/V live in their persistent buffers (f32); borrow them — attention reads only
                    // the first `kv_len` rows, never the whole `max_ctx` cache.
                    let kbuf = bindings.get(k_cache).expect("cpu backend: unbound k_cache");
                    let vbuf = bindings.get(v_cache).expect("cpu backend: unbound v_cache");
                    let kguard = cpu_buf(kbuf).read();
                    let vguard = cpu_buf(vbuf).read();
                    // Materialize the valid KV rows as f32. K and V pick their cache dtype
                    // INDEPENDENTLY (f16 matches the GPU's f16 KV; Q8_0 blocks dequant via
                    // y = d*q) — the inner dot then runs in f32 either way.
                    //
                    // SWA ring cache: the cache holds only `cap_rows` rows (window + ubatch — see
                    // the seam runner's ring sizing) and position j lives at row j % cap_rows.
                    // The ring only ever recycles rows whose positions the sliding-window mask
                    // (`lo`) already excludes, so the mapping is exact; on a full-context cache
                    // kv_len <= cap_rows and everything below is the old identity indexing.
                    let cap_rows = g.desc(k_cache).numel() / (nkv * hd).max(1);
                    let need = kv_len.min(cap_rows) * nkv * hd;
                    let deq = |b: &[u8], dt: DType| -> Vec<f32> {
                        match dt {
                            DType::F16 => bytemuck::cast_slice::<u8, u16>(b)[..need]
                                .iter()
                                .map(|&x| half::f16::from_bits(x).to_f32())
                                .collect(),
                            DType::Q8_0 => crate::dequant_prefix_q8_0(b, need),
                            // TurboQuant blocks store the WHT-rotated values; dequant + inverse WHT
                            // recovers the original domain so the f32 SDPA below runs unchanged.
                            dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4) => {
                                crate::turbo::dequant_prefix_orig(dt, b, need)
                            }
                            // bf16 + mainline low-bit quants: dequant the block-aligned prefix via the
                            // shared GGUF dequant (only the valid `kv_len` rows, not the whole cache).
                            DType::Bf16
                            | DType::Q4_0
                            | DType::Q4_1
                            | DType::Q5_0
                            | DType::Q5_1
                            | DType::Iq4Nl => {
                                let pb = infr_gguf::nbytes(dt, need);
                                infr_gguf::dequant::dequant_block(dt, &b[..pb])
                                    .expect("cpu backend: KV dequant")
                            }
                            _ => bytemuck::cast_slice::<u8, f32>(b)[..need].to_vec(),
                        }
                    };
                    let ks = deq(&kguard, g.desc(k_cache).dtype);
                    let vs = deq(&vguard, g.desc(v_cache).dtype);
                    let group = nh / nkv;
                    // `Causal`/`SlidingWindow` clip the causal END at `abs+1` (per-row, from
                    // `pos`); `Canvas` (DiffusionGemma denoise — see `AttnMask::Canvas`'s doc)
                    // ignores `pos` entirely and gives every row the SAME fixed bidirectional
                    // range `[lo, kv_len)`.
                    let (window, canvas_lo) = match mask {
                        AttnMask::Causal => (0usize, None),
                        AttnMask::SlidingWindow(w) => (w, None),
                        AttnMask::Canvas { lo } => (0usize, Some(lo)),
                    };
                    let mut out = vec![0f32; rows * nh * hd];
                    // Each (ti, h) pair writes exactly one hd-sized output slice with no
                    // cross-iteration deps → embarrassingly parallel.  Chunk index i = ti*nh+h.
                    self.pool().for_chunks_mut(&mut out, hd, 2, &|i, ob_slice| {
                        let ti = i / nh;
                        let h = i % nh;
                        let kvh = h / group;
                        let qb = (ti * nh + h) * hd;
                        let abs = pos as usize + ti; // absolute position of this query
                        let (lo, hi) = match canvas_lo {
                            // bidirectional: every row attends the same fixed [lo, kv_len).
                            Some(clo) => (clo, kv_len),
                            // causal (± SWA): [lo, abs] — SWA clips lo to abs-window+1.
                            None => {
                                let lo = if window > 0 && abs + 1 > window {
                                    abs + 1 - window
                                } else {
                                    0
                                };
                                (lo, abs + 1)
                            }
                        };
                        let n_keys = hi - lo;
                        let mut sc = vec![0f32; n_keys];
                        let mut mx = f32::NEG_INFINITY;
                        let qrow = &qs[qb..qb + hd];
                        for (jj, scj) in sc.iter_mut().enumerate() {
                            let j = lo + jj;
                            let jr = if cap_rows > 0 { j % cap_rows } else { j };
                            let kb = (jr * nkv + kvh) * hd;
                            // 8-accumulator SIMD dot (was a serial per-element f32 chain — kept
                            // scalar in an earlier campaign only because the reassociation
                            // flipped a golden hash; the numerics policy now allows it).
                            *scj = dot(qrow, &ks[kb..kb + hd]) * scale;
                            mx = mx.max(*scj);
                        }
                        let mut l = 0f32;
                        for &s in &sc {
                            l += (s - mx).exp();
                        }
                        for (jj, &s) in sc.iter().enumerate() {
                            let j = lo + jj;
                            let jr = if cap_rows > 0 { j % cap_rows } else { j };
                            let p = (s - mx).exp() / l;
                            let vb = (jr * nkv + kvh) * hd;
                            // Independent lanes — vectorizes to FMA.
                            for (o, &v) in ob_slice.iter_mut().zip(&vs[vb..vb + hd]) {
                                *o = v.mul_add(p, *o);
                            }
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::GatedAct {
                    gate,
                    up,
                    dst,
                    rows,
                    nff,
                    act,
                    up_off,
                    up_stride,
                    gate_stride,
                    gate_block_width,
                    ..
                } => {
                    let (rows, nff, up_off, up_stride, gate_stride, gate_block_width) = (
                        rows as usize,
                        nff as usize,
                        up_off as usize,
                        up_stride as usize,
                        gate_stride as usize,
                        gate_block_width as usize,
                    );
                    let gs = &vals[gate.0 as usize];
                    let us = &vals[up.0 as usize];
                    let mut out = vec![0f32; rows * nff];
                    self.pool().for_chunks_mut(&mut out, nff, 1, &|r, orow| {
                        let gb = if gate_block_width > 0 || gate_stride > 0 {
                            r * gate_stride
                        } else {
                            r * nff
                        };
                        let ub = if up_stride == 0 {
                            r * nff + up_off
                        } else {
                            r * up_stride + up_off
                        };
                        for i in 0..nff {
                            let gi = if gate_block_width > 0 {
                                let headw = gate_block_width / 2;
                                let head = i / headw;
                                let off = i % headw;
                                gb + head * gate_block_width + headw + off
                            } else {
                                gb + i
                            };
                            orow[i] = act_fn(act, gs[gi]) * us[ub + i];
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::GatedActFused {
                    gu,
                    dst,
                    rows,
                    nff,
                    act,
                } => {
                    // Combined [rows, 2*nff] gate|up buffer: gate half first, up half second. Rows
                    // are independent — spin-pool over rows, bit-identical to the serial version.
                    let (rows, nff) = (rows as usize, nff as usize);
                    let gus = &vals[gu.0 as usize];
                    let mut out = vec![0f32; rows * nff];
                    self.pool().for_chunks_mut(&mut out, nff, 1, &|r, orow| {
                        let gb = r * 2 * nff;
                        for i in 0..nff {
                            orow[i] = act_fn(act, gus[gb + i]) * gus[gb + nff + i];
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::Add { a, b, dst, n } => {
                    let n = n as usize;
                    // Elementwise with no cross-element dependency — chunked spin-pool, bit-
                    // identical. (The oldest form CLONED the whole `a` vector, then added serially:
                    // a ~0.7 MB memcpy + a one-thread loop per Add while 31 threads slept.)
                    let av = &vals[a.0 as usize];
                    let bv = &vals[b.0 as usize];
                    let mut out = vec![0f32; n];
                    self.pool().for_chunks_mut(&mut out, 4096, 4, &|c, oc| {
                        let base = c * 4096;
                        for (i, o) in oc.iter_mut().enumerate() {
                            *o = av[base + i] + bv[base + i];
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                // Broadcast bias: add the length-`n` `bias` to each of `rows` rows (Qwen2 q/k/v).
                Op::AddBias {
                    x,
                    bias,
                    dst,
                    rows,
                    n,
                } => {
                    let (rows, n) = (rows as usize, n as usize);
                    let xs = &vals[x.0 as usize];
                    let bv = weight(bias); // bias is a bound weight, not an activation
                                           // Rows independent — spin-pool over rows, bit-identical (no whole-input clone).
                    let mut out = vec![0f32; rows * n];
                    self.pool().for_chunks_mut(&mut out, n, 1, &|r, orow| {
                        for (c, o) in orow.iter_mut().enumerate() {
                            *o = xs[r * n + c] + bv[c];
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::Scale { x, dst, s, n } => {
                    let n = n as usize;
                    let xs = &vals[x.0 as usize];
                    // Elementwise — chunked spin-pool, bit-identical (no whole-input clone).
                    let mut out = vec![0f32; n];
                    self.pool().for_chunks_mut(&mut out, 4096, 4, &|c, oc| {
                        let base = c * 4096;
                        for (i, o) in oc.iter_mut().enumerate() {
                            *o = xs[base + i] * s;
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                // Broadcast multiply: the length-`n` `vec` scales every one of `rows` rows
                // (diffusion-gemma's router input scale — the multiplicative twin of `AddBias`).
                Op::MulVec {
                    x,
                    vec: vecid,
                    dst,
                    rows,
                    n,
                } => {
                    let (rows, n) = (rows as usize, n as usize);
                    let xs = &vals[x.0 as usize];
                    let vv = weight(vecid); // vec is a bound weight, not an activation
                                            // Rows independent — spin-pool over rows, bit-identical (no whole-input clone).
                    let mut out = vec![0f32; rows * n];
                    self.pool().for_chunks_mut(&mut out, n, 1, &|r, orow| {
                        for (c, o) in orow.iter_mut().enumerate() {
                            *o = xs[r * n + c] * vv[c];
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                // qwen35moe shared-expert combine: `moe` (routed-MoE output) + sigmoid(`gate[r]`) *
                // `shexp` (the shared expert's own dense FFN output), row-broadcast — see the
                // `Op::MoeSharedExpertAdd` doc. Rows independent — spin-pool over rows, mirrors
                // `Op::MulVec` above (bit-identical, no whole-input clone).
                Op::MoeSharedExpertAdd {
                    moe,
                    shexp,
                    gate,
                    dst,
                    rows,
                    n,
                } => {
                    let (rows, n) = (rows as usize, n as usize);
                    let mv = &vals[moe.0 as usize];
                    let sv = &vals[shexp.0 as usize];
                    let gv = &vals[gate.0 as usize]; // [rows] raw pre-sigmoid logits
                    let mut out = vec![0f32; rows * n];
                    self.pool().for_chunks_mut(&mut out, n, 1, &|r, orow| {
                        let g = 1.0 / (1.0 + (-gv[r]).exp());
                        for (c, o) in orow.iter_mut().enumerate() {
                            *o = mv[r * n + c] + g * sv[r * n + c];
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::Softcap { x, dst, cap, n } => {
                    let n = n as usize;
                    let xs = &vals[x.0 as usize];
                    // Pure elementwise map (no cross-element dependency) — safe to fan out over
                    // rayon with ZERO numeric change. At the lm_head's shape (256 rows × 262144
                    // vocab = ~67M `tanh` calls) this was the single most expensive scalar loop in
                    // the interpreter outside `Op::Linear`/`Op::MoeFfn`.
                    // Chunked (not per-element zip — 67M one-element items is pure plumbing).
                    let mut out = vec![0f32; n];
                    self.pool().for_chunks_mut(&mut out, 4096, 4, &|c, oc| {
                        let base = c * 4096;
                        for (i, o) in oc.iter_mut().enumerate() {
                            *o = cap * (xs[base + i] / cap).tanh();
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::EmbedGather {
                    ids,
                    table,
                    dst,
                    rows,
                    ne,
                    scale,
                } => {
                    // Dequantize the selected rows straight from the raw (mmap'd) table bytes —
                    // identical math to the load-time dequant that used to build the host f32
                    // table, so outputs are bit-equal to the old host-embed path.
                    let (rows, ne) = (rows as usize, ne as usize);
                    let buf = bindings.get(table).expect("cpu backend: unbound Weight");
                    let bytes = cpu_buf(buf).read();
                    let dt = g.desc(table).dtype;
                    let total_rows = g.desc(table).numel() / ne;
                    let bpr = bytes.len() / total_rows;
                    let ids_v = vals[ids.0 as usize].clone();
                    let mut out = vec![0f32; rows * ne];
                    for r in 0..rows {
                        let tok = ids_v[r] as usize;
                        let row = bytes_to_f32(&bytes[tok * bpr..(tok + 1) * bpr], dt);
                        for (o, v) in out[r * ne..(r + 1) * ne].iter_mut().zip(&row) {
                            *o = v * scale;
                        }
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::Sample {
                    x,
                    u,
                    dst,
                    n,
                    top_k,
                    temp,
                    top_p,
                } => {
                    // Device-side stochastic sampling — see `sample_token` (top_k == 0 = no
                    // truncation; the uniform draw is factored out into the 1-float `u` input).
                    let logits = &vals[x.0 as usize][..n as usize];
                    let uu = vals[u.0 as usize][0];
                    let tok = sample_token(logits, uu, top_k as usize, temp, top_p);
                    vals[dst.0 as usize] = vec![f32::from_bits(tok)];
                }
                Op::Argmax { x, dst, n, rows } => {
                    // Greedy device-side sampling: strict `>` keeps the lowest index on ties —
                    // identical to the host sampler's argmax (per row). Each id is stored as a
                    // u32 bit-pattern in the f32 slot (the graph's only tensor dtype). rows == 1
                    // is the decode loop; rows > 1 is the MTP verify accept (issue #31).
                    let n = n as usize;
                    let mut out = Vec::with_capacity(rows as usize);
                    for r in 0..rows as usize {
                        let xs = &vals[x.0 as usize][r * n..(r + 1) * n];
                        let mut best = f32::NEG_INFINITY;
                        let mut bi = 0u32;
                        for (i, &v) in xs.iter().enumerate() {
                            if v > best {
                                best = v;
                                bi = i as u32;
                            }
                        }
                        out.push(f32::from_bits(bi));
                    }
                    vals[dst.0 as usize] = out;
                }
                Op::ArgmaxProb {
                    x,
                    dst_id,
                    dst_prob,
                    n,
                } => {
                    // Exact reference: this MUST match `infr_llama::mtp::top1_softmax` bit-for-bit
                    // (same strict-`>` argmax scan, then the SAME left-to-right `sum_j
                    // exp(x[j]-max)` accumulation order) — the CPU backend is the golden oracle the
                    // GPU two-stage reduction is checked against (`mtp_head_cpu_vulkan_parity`).
                    let n = n as usize;
                    let xs = &vals[x.0 as usize][0..n];
                    let mut best = f32::NEG_INFINITY;
                    let mut bi = 0u32;
                    for (i, &v) in xs.iter().enumerate() {
                        if v > best {
                            best = v;
                            bi = i as u32;
                        }
                    }
                    let z: f32 = xs.iter().map(|&v| (v - best).exp()).sum();
                    vals[dst_id.0 as usize] = vec![f32::from_bits(bi)];
                    vals[dst_prob.0 as usize] = vec![1.0 / z];
                }
                Op::Copy {
                    src,
                    src_off,
                    dst,
                    dst_off,
                    n,
                } => {
                    let (so, dof, n) = (src_off as usize, dst_off as usize, n as usize);
                    if src.0 == dst.0 {
                        // Aliasing copy within one tensor: a temp is required (can't borrow the
                        // same Vec mutably and immutably). Clone only the read window.
                        let s = vals[src.0 as usize][so..so + n].to_vec();
                        vals[dst.0 as usize][dof..dof + n].copy_from_slice(&s);
                    } else {
                        let (s, d) = borrow_two_mut(&mut vals, src.0 as usize, dst.0 as usize);
                        d[dof..dof + n].copy_from_slice(&s[so..so + n]);
                    }
                }
                Op::CopyStrided {
                    src,
                    src_off,
                    src_stride,
                    dst,
                    dst_off,
                    dst_stride,
                    rows,
                    n,
                } => {
                    let (so, ss, dof, ds, n) = (
                        src_off as usize,
                        src_stride as usize,
                        dst_off as usize,
                        dst_stride as usize,
                        n as usize,
                    );
                    if src.0 == dst.0 {
                        // Aliasing strided copy: clone the source rows (can't hold &s and &mut d on
                        // one Vec). Rare — kept correct rather than fast.
                        let s = vals[src.0 as usize].clone();
                        let d = &mut vals[dst.0 as usize];
                        for r in 0..rows as usize {
                            d[dof + r * ds..dof + r * ds + n]
                                .copy_from_slice(&s[so + r * ss..so + r * ss + n]);
                        }
                    } else {
                        let (s, d) = borrow_two_mut(&mut vals, src.0 as usize, dst.0 as usize);
                        for r in 0..rows as usize {
                            d[dof + r * ds..dof + r * ds + n]
                                .copy_from_slice(&s[so + r * ss..so + r * ss + n]);
                        }
                    }
                }
                Op::MoeFfn {
                    x,
                    router_x,
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                    down_scale,
                    fused_gate_up,
                    dst,
                    ne,
                    n_expert,
                    n_used,
                    n_ff_exp,
                    scale,
                    act,
                    gating,
                    norm_w,
                    weight_before,
                    ep_band: _, // EP is a Vulkan-only path; the CPU reference always runs full-expert
                } => {
                    let (ne, n_expert, n_used, nffx) = (
                        ne as usize,
                        n_expert as usize,
                        n_used as usize,
                        n_ff_exp as usize,
                    );
                    let xs = vals[x.0 as usize].clone();
                    // `router_x` is usually the SAME tensor as `x` (qwen3moe); diffusion-gemma binds
                    // a differently-normalized row (see the `Op::MoeFfn` doc). Clone independently —
                    // it may legitimately be a different handle with its own row layout.
                    let rxs = vals[router_x.0 as usize].clone();
                    // `x` may hold several rows (the seam's batched prefill): route + run the
                    // expert FFN independently per row — the reference semantics for the GPU
                    // adapter's GPU-routed batched form.
                    let rows = xs.len() / ne;
                    let rbuf = bindings.get(router).expect("cpu backend: unbound router");
                    let gbuf = bindings
                        .get(gate_exps)
                        .expect("cpu backend: unbound gate_exps");
                    let dbuf = bindings
                        .get(down_exps)
                        .expect("cpu backend: unbound down_exps");
                    let rbytes = cpu_buf(rbuf).read();
                    let gb = cpu_buf(gbuf).read();
                    let db = cpu_buf(dbuf).read();
                    let rdt = g.desc(router).dtype;
                    let gdt = g.desc(gate_exps).dtype;
                    let ddt = g.desc(down_exps).dtype;
                    // Fused: `gate_exps` holds BOTH roles ([ne, 2*n_ff_exp, n_expert], gate rows
                    // first); split gets its own separate up_exps/up buffer.
                    let (ub, udt) = if fused_gate_up {
                        (None, gdt)
                    } else {
                        let ubuf = bindings.get(up_exps).expect("cpu backend: unbound up_exps");
                        (Some(cpu_buf(ubuf).read()), g.desc(up_exps).dtype)
                    };
                    let gst = gb.len() / n_expert;
                    let ust = ub.as_ref().map(|b| b.len() / n_expert);
                    let dst_ = db.len() / n_expert;
                    let dscale = down_scale.map(&weight); // per-expert scale [n_expert], if any

                    // ── Router: ONE batched matvec over every row (router_x is tiny — [n_expert, ne]
                    // — so this alone replaces what used to be `rows` separate re-dequants of it),
                    // then a per-row softmax + top-`n_used` selection (independent per row, so
                    // parallel over rows is safe).
                    // Whole-call gate — see expert_matvec_batch's param doc. REFERENCE mode
                    // (`CpuBackend::reference`) forces every bank onto `ActsKind::Raw`, i.e. the
                    // f32-activation path decode (rows == 1) already takes, so an oracle run has
                    // no int8-activation error anywhere.
                    let int8_ok = rows >= 2 && !self.reference;
                    let pool = self.pool();
                    // Router GEMM on the pool: serial it was 12-20ms/layer at 512 rows (134M MAC)
                    // — long enough that spinning pool workers SMT-throttled it, which is where
                    // qwen3moe's phase-3 prefill regression hid (per-op MoeFfn time had IMPROVED).
                    let logits_all: Vec<f32> = {
                        let r_kind = expert_acts_kind(rdt, ne, int8_ok);
                        let rq8: Vec<Q8> = if r_kind == ActsKind::Super {
                            pool.collect(rows, &|r| quantize_q8(&rxs[r * ne..r * ne + ne]))
                        } else {
                            Vec::new()
                        };
                        let rq832: Vec<Q8x32> = if r_kind == ActsKind::Blk32 {
                            pool.collect(rows, &|r| quantize_q8_32(&rxs[r * ne..r * ne + ne]))
                        } else {
                            Vec::new()
                        };
                        let racts = match r_kind {
                            ActsKind::Super => ExpertActs::Super(&rq8),
                            ActsKind::Blk32 => ExpertActs::Blk32(&rq832),
                            ActsKind::Raw => ExpertActs::Raw(&rxs),
                        };
                        let mut lt = vec![0f32; n_expert * rows]; // o-major
                        pool.for_chunks_mut(&mut lt, 8 * rows, 1, &|c, dst| {
                            let o0 = c * 8;
                            let o1 = (o0 + 8).min(n_expert);
                            expert_gemm_range(
                                &rbytes, rdt, ne, n_expert, &racts, None, o0, o1, dst,
                            );
                        });
                        let mut logits = vec![0f32; rows * n_expert];
                        pool.for_chunks_mut(&mut logits, n_expert, 4, &|r, lrow| {
                            for (o, dst) in lrow.iter_mut().enumerate() {
                                *dst = lt[o * rows + r];
                            }
                        });
                        logits
                    };
                    struct RouteRow {
                        /// Selected experts, sorted by DESCENDING router prob — the same order the
                        /// original per-row loop accumulated in.
                        idx: Vec<usize>,
                        /// Final per-expert weight (renormalized top-k prob × `scale`), aligned to `idx`.
                        w: Vec<f32>,
                    }
                    let routes: Vec<RouteRow> = pool.collect(rows, &|r| {
                        let logits = &logits_all[r * n_expert..r * n_expert + n_expert];
                        // Router probabilities: softmax over all experts (qwen3moe/…) or a per-expert
                        // sigmoid (llama4). Both are monotone in the logit, so the top-`n_used`
                        // selection is identical either way (llama4 selects by raw logits — same set).
                        let probs: Vec<f32> = match gating {
                            MoeGating::Softmax => {
                                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                let mut p: Vec<f32> =
                                    logits.iter().map(|&v| (v - maxl).exp()).collect();
                                let psum: f32 = p.iter().sum();
                                for v in p.iter_mut() {
                                    *v /= psum;
                                }
                                p
                            }
                            MoeGating::Sigmoid => {
                                logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
                            }
                        };
                        let mut idx: Vec<usize> = (0..n_expert).collect();
                        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                        idx.truncate(n_used);
                        // `norm_w`: renormalize the selected weights to sum to 1 before scaling
                        // (softmax MoE); llama4 uses the raw sigmoid prob × scale (no renorm).
                        let w: Vec<f32> = if norm_w {
                            let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                            idx.iter().map(|&e| probs[e] / wsum * scale).collect()
                        } else {
                            idx.iter().map(|&e| probs[e] * scale).collect()
                        };
                        RouteRow { idx, w }
                    });

                    // ── Phase 3 (threadpool restructure): the per-expert rayon fan-out became a
                    // STAGED pipeline of flat task lists on the spin-pool — the same design the
                    // Vulkan backend's single-dispatch-per-stage batched MoE uses. Work is
                    // reordered, never re-derived: every kernel sees the same bytes in the same
                    // per-output order as the old per-expert calls, so outputs stay bit-identical
                    // (the (row, rank) bookkeeping below replays the original per-row summation
                    // order exactly like the old `row_slots` scatter did).
                    //
                    // pair = one (row → expert) routing, grouped contiguously by expert:
                    let mut buckets: Vec<(usize, usize, usize)> = Vec::new(); // (expert, p0, count)
                    let mut pair_row: Vec<usize> = Vec::new();
                    let mut pair_bucket: Vec<u32> = Vec::new();
                    let mut pair_local: Vec<u32> = Vec::new();
                    // llama4 `weight_before_ffn`: the routing weight per (row→expert) pair, folded
                    // into that pair's gate/up activations (stage C) instead of the output (stage E).
                    let mut pair_w: Vec<f32> = Vec::new();
                    // slot_pair[r][rank] = flat pair index — stage E's replay map.
                    let mut slot_pair: Vec<Vec<usize>> = routes
                        .iter()
                        .map(|rr| vec![usize::MAX; rr.idx.len()])
                        .collect();
                    {
                        let mut expert_tasks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_expert];
                        for (r, rr) in routes.iter().enumerate() {
                            for (rank, &e) in rr.idx.iter().enumerate() {
                                expert_tasks[e].push((r, rank));
                            }
                        }
                        for (e, tasks) in expert_tasks.iter().enumerate() {
                            if tasks.is_empty() {
                                continue;
                            }
                            let p0 = pair_row.len();
                            for (i, &(r, rank)) in tasks.iter().enumerate() {
                                slot_pair[r][rank] = pair_row.len();
                                pair_row.push(r);
                                pair_bucket.push(buckets.len() as u32);
                                pair_local.push(i as u32);
                                pair_w.push(routes[r].w[rank]);
                            }
                            buckets.push((e, p0, pair_row.len() - p0));
                        }
                    }
                    let n_pairs = pair_row.len();
                    // `INFR_MOE_COUNTS_DEBUG=1`: print the REAL per-expert row-count distribution
                    // for this chunk/layer (this CPU interpreter's top-k routing is bit-identical
                    // to the GPU's, so this is a host-side stand-in for reading the GPU's own
                    // counts/offsets buffers without a mid-graph readback). Used to check whether
                    // `matmul_mmq_experts`' BM=32/64 row-tile heuristic (recorder.rs,
                    // MOE_EXPERT_SMALL_TILE_AVG_ROWS) — picked against a MEAN-BALANCED synthetic
                    // distribution — still holds up against real (possibly skewed) routing.
                    // `INFR_MOE_COUNTS_DUMP=1` additionally dumps the raw per-expert counts array
                    // (for feeding into an isolated GPU tile-choice bench, e.g.
                    // `moe_expert_row_tile_bench_real_skew` in infr-vulkan's gemm_bench.rs).
                    if std::env::var("INFR_MOE_COUNTS_DEBUG").is_ok() {
                        let mut counts = vec![0usize; n_expert];
                        for &(e, _, c) in buckets.iter() {
                            counts[e] = c;
                        }
                        let mut sorted = counts.clone();
                        sorted.sort_unstable();
                        let sum: usize = counts.iter().sum();
                        let max = *sorted.last().unwrap_or(&0);
                        let min = *sorted.first().unwrap_or(&0);
                        let mean = sum as f64 / n_expert as f64;
                        let p50 = sorted[n_expert / 2];
                        let p90 = sorted[(n_expert * 9) / 10];
                        let nz = counts.iter().filter(|&&c| c > 0).count();
                        let buckets_hist = [0usize, 4, 8, 16, 24, 32, 48, 64, usize::MAX];
                        let mut hist = vec![0usize; buckets_hist.len() - 1];
                        for &c in counts.iter() {
                            for w in 0..buckets_hist.len() - 1 {
                                if c >= buckets_hist[w] && c < buckets_hist[w + 1] {
                                    hist[w] += 1;
                                    break;
                                }
                            }
                        }
                        let waste = |bm: usize| -> usize {
                            counts
                                .iter()
                                .map(|&c| c.div_ceil(bm).max(1) * bm - c)
                                .sum::<usize>()
                        };
                        let real_tiles = |bm: usize| -> usize {
                            counts.iter().map(|&c| c.div_ceil(bm)).sum::<usize>()
                        };
                        // per-expert OPTIMAL tile from {16,32,64} (min waste, tie -> larger BM
                        // for fewer real tiles / less fixed per-workgroup overhead):
                        let opt_waste: usize = counts
                            .iter()
                            .map(|&c| {
                                [16usize, 32, 64]
                                    .iter()
                                    .map(|&bm| c.div_ceil(bm).max(1) * bm - c)
                                    .min()
                                    .unwrap()
                            })
                            .sum();
                        eprintln!(
                            "MOE_COUNTS rows={rows} n_expert={n_expert} n_used={n_used} \
                             n_pairs={n_pairs} nonzero={nz} min={min} max={max} mean={mean:.2} \
                             p50={p50} p90={p90} hist(0-3,4-7,8-15,16-23,24-31,32-47,48-63,64+)={hist:?} \
                             waste16={} waste32={} waste64={} waste_opt={opt_waste} \
                             tiles16={} tiles32={} tiles64={}",
                            waste(16),
                            waste(32),
                            waste(64),
                            real_tiles(16),
                            real_tiles(32),
                            real_tiles(64),
                        );
                        if std::env::var("INFR_MOE_COUNTS_DUMP").is_ok() {
                            eprintln!("MOE_COUNTS_RAW {counts:?}");
                        }
                    }
                    let out_gu = if fused_gate_up { 2 * nffx } else { nffx };
                    // o-major output offsets per bucket for the gate_up / down GEMM stages.
                    let mut gu_off = vec![0usize; buckets.len() + 1];
                    let mut d_off = vec![0usize; buckets.len() + 1];
                    for (b, &(_, _, count)) in buckets.iter().enumerate() {
                        gu_off[b + 1] = gu_off[b] + out_gu * count;
                        d_off[b + 1] = d_off[b] + ne * count;
                    }
                    // 8-aligned o-chunks (64 rows) per bucket — the flat GEMM task list. Dynamic
                    // claiming spreads a straggler expert's chunks over every idle thread, which
                    // replaces both the old per-expert fan-out AND its nested straggler split.
                    let gemm_tasks = |out_f: usize| -> Vec<(u32, u32, u32)> {
                        let mut t = Vec::new();
                        for b in 0..buckets.len() as u32 {
                            let mut o = 0u32;
                            while (o as usize) < out_f {
                                let o1 = (o + 64).min(out_f as u32);
                                t.push((b, o, o1));
                                o = o1;
                            }
                        }
                        t
                    };

                    // ── Stage A: gate/up activations, quantized ONCE per distinct row (the old
                    // per-expert gather re-quantized a row for every expert it routed to), then
                    // cloned per pair so each bucket owns a contiguous slice.
                    let t_a = std::time::Instant::now();
                    let g_kind = expert_acts_kind(gdt, ne, int8_ok);
                    let u_kind = if fused_gate_up {
                        g_kind
                    } else {
                        expert_acts_kind(udt, ne, int8_ok)
                    };
                    let need = |k: ActsKind| g_kind == k || u_kind == k;
                    let q8_pairs: Vec<Q8> = if need(ActsKind::Super) {
                        let q8_rows: Vec<Q8> =
                            pool.collect(rows, &|r| quantize_q8(&xs[r * ne..r * ne + ne]));
                        pool.collect(n_pairs, &|p| q8_rows[pair_row[p]].clone())
                    } else {
                        Vec::new()
                    };
                    let q832_pairs: Vec<Q8x32> = if need(ActsKind::Blk32) {
                        let q_rows: Vec<Q8x32> =
                            pool.collect(rows, &|r| quantize_q8_32(&xs[r * ne..r * ne + ne]));
                        pool.collect(n_pairs, &|p| q_rows[pair_row[p]].clone())
                    } else {
                        Vec::new()
                    };
                    let xin_pairs: Vec<f32> = if need(ActsKind::Raw) {
                        let mut v = vec![0f32; n_pairs * ne];
                        pool.for_chunks_mut(&mut v, ne, 4, &|p, dstrow| {
                            dstrow.copy_from_slice(&xs[pair_row[p] * ne..pair_row[p] * ne + ne]);
                        });
                        v
                    } else {
                        Vec::new()
                    };
                    let acts_for = |k: ActsKind, b: usize| -> ExpertActs {
                        let (_, p0, count) = buckets[b];
                        match k {
                            ActsKind::Super => ExpertActs::Super(&q8_pairs[p0..p0 + count]),
                            ActsKind::Blk32 => ExpertActs::Blk32(&q832_pairs[p0..p0 + count]),
                            ActsKind::Raw => {
                                ExpertActs::Raw(&xin_pairs[p0 * ne..(p0 + count) * ne])
                            }
                        }
                    };
                    let t_gather = t_a.elapsed();

                    // ── Stage B: gate_up GEMMs, o-major per bucket. Cached ilv packs (fused Q4_K
                    // + VNNI only, exactly the old gate) are fetched per bucket up front.
                    let t_b = std::time::Instant::now();
                    #[cfg(target_arch = "x86_64")]
                    let gu_packs: Vec<Option<std::sync::Arc<Q4kPack>>> = {
                        let packable = int8_ok
                            && fused_gate_up
                            && gdt == DType::Q4K
                            && ne.is_multiple_of(256)
                            && (2 * nffx).is_multiple_of(8)
                            && is_x86_feature_detected!("avx512bw")
                            && is_x86_feature_detected!("avx512vnni");
                        if packable {
                            pool.collect(buckets.len(), &|b| {
                                let e = buckets[b].0;
                                Some(self.q4k_pack_for(&gb[e * gst..(e + 1) * gst], ne, 2 * nffx))
                            })
                        } else {
                            vec![None; buckets.len()]
                        }
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let gu_packs: Vec<Option<std::sync::Arc<Q4kPack>>> = vec![None; buckets.len()];
                    let mut gu_all = vec![0f32; gu_off[buckets.len()]];
                    let mut up_all: Vec<f32> = if fused_gate_up {
                        Vec::new()
                    } else {
                        vec![0f32; gu_off[buckets.len()]]
                    };
                    {
                        let tasks = gemm_tasks(out_gu);
                        let gu_ptr = pool::SendPtr::new(gu_all.as_mut_ptr());
                        pool.run(tasks.len(), &|t| {
                            let (b, o0, o1) = tasks[t];
                            let (b, o0, o1) = (b as usize, o0 as usize, o1 as usize);
                            let (e, _, count) = buckets[b];
                            // SAFETY: (bucket, o-range) slices of gu_all are disjoint per task.
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(
                                    gu_ptr.get().add(gu_off[b] + o0 * count),
                                    (o1 - o0) * count,
                                )
                            };
                            expert_gemm_range(
                                &gb[e * gst..(e + 1) * gst],
                                gdt,
                                ne,
                                out_gu,
                                &acts_for(g_kind, b),
                                gu_packs[b].as_deref(),
                                o0,
                                o1,
                                dst,
                            );
                        });
                        if !fused_gate_up {
                            let ub = ub.as_ref().expect("split gate/up: up_exps missing");
                            let ust = ust.expect("split gate/up: up stride missing");
                            let up_ptr = pool::SendPtr::new(up_all.as_mut_ptr());
                            pool.run(tasks.len(), &|t| {
                                let (b, o0, o1) = tasks[t];
                                let (b, o0, o1) = (b as usize, o0 as usize, o1 as usize);
                                let (e, _, count) = buckets[b];
                                // SAFETY: disjoint (bucket, o-range) slices, as above.
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        up_ptr.get().add(gu_off[b] + o0 * count),
                                        (o1 - o0) * count,
                                    )
                                };
                                expert_gemm_range(
                                    &ub[e * ust..(e + 1) * ust],
                                    udt,
                                    ne,
                                    nffx,
                                    &acts_for(u_kind, b),
                                    None,
                                    o0,
                                    o1,
                                    dst,
                                );
                            });
                        }
                    }
                    let t_gate_up = t_b.elapsed();

                    // ── Stage C: gated activation per pair (strided o-major reads — the same
                    // values the old un-transpose+split produced), quantized straight into the
                    // representation the DOWN bank's dtype wants.
                    let t_c = std::time::Instant::now();
                    let d_kind = expert_acts_kind(ddt, nffx, int8_ok);
                    let act_row_of = |p: usize| -> Vec<f32> {
                        let b = pair_bucket[p] as usize;
                        let i = pair_local[p] as usize;
                        let count = buckets[b].2;
                        // llama4 weight-before-FFN: scale gate & up by the pair's routing weight
                        // BEFORE the activation. Exact (`gate`/`up` are linear, so `gate(w·x) =
                        // w·gate(x)`): `silu(w·gate)·(w·up)` == applying `w` to the FFN input.
                        let wp = if weight_before { pair_w[p] } else { 1.0 };
                        let mut row = vec![0f32; nffx];
                        if fused_gate_up {
                            let gu = &gu_all[gu_off[b]..gu_off[b] + 2 * nffx * count];
                            for (f, v) in row.iter_mut().enumerate() {
                                *v = act_fn(act, wp * gu[f * count + i])
                                    * (wp * gu[(nffx + f) * count + i]);
                            }
                        } else {
                            let gt = &gu_all[gu_off[b]..gu_off[b] + nffx * count];
                            let ut = &up_all[gu_off[b]..gu_off[b] + nffx * count];
                            for (f, v) in row.iter_mut().enumerate() {
                                *v = act_fn(act, wp * gt[f * count + i]) * (wp * ut[f * count + i]);
                            }
                        }
                        row
                    };
                    let (dq8_pairs, dq832_pairs, dact_pairs): (Vec<Q8>, Vec<Q8x32>, Vec<f32>) =
                        match d_kind {
                            ActsKind::Super => (
                                pool.collect(n_pairs, &|p| quantize_q8(&act_row_of(p))),
                                Vec::new(),
                                Vec::new(),
                            ),
                            ActsKind::Blk32 => (
                                Vec::new(),
                                pool.collect(n_pairs, &|p| quantize_q8_32(&act_row_of(p))),
                                Vec::new(),
                            ),
                            ActsKind::Raw => {
                                let mut v = vec![0f32; n_pairs * nffx];
                                pool.for_chunks_mut(&mut v, nffx, 1, &|p, dstrow| {
                                    dstrow.copy_from_slice(&act_row_of(p));
                                });
                                (Vec::new(), Vec::new(), v)
                            }
                        };
                    let dacts_for = |b: usize| -> ExpertActs {
                        let (_, p0, count) = buckets[b];
                        match d_kind {
                            ActsKind::Super => ExpertActs::Super(&dq8_pairs[p0..p0 + count]),
                            ActsKind::Blk32 => ExpertActs::Blk32(&dq832_pairs[p0..p0 + count]),
                            ActsKind::Raw => {
                                ExpertActs::Raw(&dact_pairs[p0 * nffx..(p0 + count) * nffx])
                            }
                        }
                    };
                    let t_act = t_c.elapsed();

                    // ── Stage D: down GEMMs (o-major), then per-pair un-transpose to row-major
                    // with the per-expert `down_scale` folded in (`y = s * raw`, the same
                    // scale-then-weight order the old per-expert loop applied).
                    let t_d = std::time::Instant::now();
                    let mut down_all = vec![0f32; d_off[buckets.len()]];
                    {
                        let tasks = gemm_tasks(ne);
                        let d_ptr = pool::SendPtr::new(down_all.as_mut_ptr());
                        pool.run(tasks.len(), &|t| {
                            let (b, o0, o1) = tasks[t];
                            let (b, o0, o1) = (b as usize, o0 as usize, o1 as usize);
                            let (e, _, count) = buckets[b];
                            // SAFETY: disjoint (bucket, o-range) slices, as above.
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(
                                    d_ptr.get().add(d_off[b] + o0 * count),
                                    (o1 - o0) * count,
                                )
                            };
                            expert_gemm_range(
                                &db[e * dst_..(e + 1) * dst_],
                                ddt,
                                nffx,
                                ne,
                                &dacts_for(b),
                                None,
                                o0,
                                o1,
                                dst,
                            );
                        });
                    }
                    let mut y_pairs = vec![0f32; n_pairs * ne];
                    pool.for_chunks_mut(&mut y_pairs, ne, 1, &|p, yrow| {
                        let b = pair_bucket[p] as usize;
                        let i = pair_local[p] as usize;
                        let (e, _, count) = buckets[b];
                        let dt_slice = &down_all[d_off[b]..d_off[b] + ne * count];
                        match &dscale {
                            Some(ds) => {
                                let s_e = ds[e];
                                for (d, y) in yrow.iter_mut().enumerate() {
                                    *y = dt_slice[d * count + i] * s_e;
                                }
                            }
                            None => {
                                for (d, y) in yrow.iter_mut().enumerate() {
                                    *y = dt_slice[d * count + i];
                                }
                            }
                        }
                    });
                    let t_down = t_d.elapsed();
                    if prof_ops {
                        let (mut bmin, mut bmax, mut single) = (usize::MAX, 0usize, 0usize);
                        for &(_, _, c) in &buckets {
                            bmin = bmin.min(c);
                            bmax = bmax.max(c);
                            single += (c == 1) as usize;
                        }
                        eprintln!(
                            "[prof-moe] gather {:.2}ms gate_up {:.2}ms act {:.2}ms down {:.2}ms  bucket[min={},max={}] active={} single={}",
                            t_gather.as_secs_f64() * 1e3,
                            t_gate_up.as_secs_f64() * 1e3,
                            t_act.as_secs_f64() * 1e3,
                            t_down.as_secs_f64() * 1e3,
                            bmin,
                            bmax,
                            buckets.len(),
                            single,
                        );
                    }

                    // ── Stage E: accumulate each row's expert contributions in the SAME order
                    // the original per-row loop used (rank order = idx sorted by descending
                    // router prob) — bit-identical to the old row_slots scatter+accumulate.
                    let mut out = vec![0f32; rows * ne];
                    pool.for_chunks_mut(&mut out, ne, 1, &|r, orow| {
                        let rr = &routes[r];
                        for (rank, &p) in slot_pair[r].iter().enumerate() {
                            debug_assert_ne!(p, usize::MAX);
                            // weight_before already folded the routing weight into stage C.
                            let w_e = if weight_before { 1.0 } else { rr.w[rank] };
                            let y = &y_pairs[p * ne..p * ne + ne];
                            for (o, &yv) in orow.iter_mut().zip(y) {
                                *o += w_e * yv;
                            }
                        }
                    });
                    vals[dst.0 as usize] = out;
                }
                Op::Conv1dSilu {
                    x,
                    weight: w,
                    state,
                    dst,
                    rows,
                    channels,
                    kernel,
                } => {
                    let (rr, cc, kk) = (rows as usize, channels as usize, kernel as usize);
                    let xs = vals[x.0 as usize].clone(); // [rows, channels]
                    let ws = weight(w); // [channels, kernel] row-major (per-channel kernel)
                                        // Gate (perf only — both paths are bit-identical): only pay the pool spin-up on
                                        // real prefill. Decode (rr==1) and tiny batches keep the serial shift-register
                                        // loop, mirroring the Vulkan/Metal `rr >= (kk-1).max(2)` gate.
                    if rr >= (kk - 1).max(2) {
                        // Parallel path: reformulate over the virtual `[state ‖ x]` sequence so every
                        // output row is independent (see `conv1d_silu`). Snapshot state up front, run
                        // the pool, then write the rebuilt history back.
                        let st = &mut vals[state.0 as usize];
                        let out = conv1d_silu(self.pool(), &xs, &ws, st, rr, cc, kk);
                        vals[dst.0 as usize] = out;
                    } else {
                        let st = &mut vals[state.0 as usize]; // [(kernel-1), channels], oldest row first
                        let mut out = vec![0f32; rr * cc];
                        // Process the rows in sequence, carrying the rolling history across tokens.
                        for t in 0..rr {
                            let xt = &xs[t * cc..t * cc + cc];
                            for ch in 0..cc {
                                // window = [history rows.. , current x]; tap j uses weight[ch*kk + j].
                                let mut acc = 0f32;
                                for j in 0..kk - 1 {
                                    acc += st[j * cc + ch] * ws[ch * kk + j];
                                }
                                acc += xt[ch] * ws[ch * kk + (kk - 1)];
                                out[t * cc + ch] = acc / (1.0 + (-acc).exp()); // silu
                            }
                            // shift history (drop oldest, append raw x).
                            for j in 0..kk.saturating_sub(2) {
                                for ch in 0..cc {
                                    st[j * cc + ch] = st[(j + 1) * cc + ch];
                                }
                            }
                            if kk >= 2 {
                                for ch in 0..cc {
                                    st[(kk - 2) * cc + ch] = xt[ch];
                                }
                            }
                        }
                        vals[dst.0 as usize] = out;
                    }
                }
                Op::DeltaNet {
                    q,
                    k,
                    v,
                    b,
                    a,
                    a_coef,
                    dt_bias,
                    state,
                    dst,
                    rows,
                    n_vhead,
                    n_khead,
                    head_k,
                    head_v,
                    eps,
                    src_stride,
                    ..
                } => {
                    let (rr, nv, nk, kd, vd) = (
                        rows as usize,
                        n_vhead as usize,
                        n_khead as usize,
                        head_k as usize,
                        head_v as usize,
                    );
                    let qf_raw = vals[q.0 as usize].clone();
                    let src_stride = src_stride as usize;
                    // For strided DeltaNet, q/k/v share one buffer; extract packed arrays. Only
                    // `qf_raw` is read on that path — defer the k/v clones to the non-strided branch
                    // that actually consumes them.
                    let (qf, kf, vf) = if src_stride > 0 {
                        let mut qv = vec![0f32; rr * nk * kd];
                        let mut kv = vec![0f32; rr * nk * kd];
                        let mut vv = vec![0f32; rr * nv * vd];
                        for t in 0..rr {
                            let row = t * src_stride;
                            qv[t * nk * kd..(t + 1) * nk * kd]
                                .copy_from_slice(&qf_raw[row..row + nk * kd]);
                            kv[t * nk * kd..(t + 1) * nk * kd]
                                .copy_from_slice(&qf_raw[row + nk * kd..row + 2 * nk * kd]);
                            vv[t * nv * vd..(t + 1) * nv * vd].copy_from_slice(
                                &qf_raw[row + 2 * nk * kd..row + 2 * nk * kd + nv * vd],
                            );
                        }
                        (qv, kv, vv)
                    } else {
                        let kf_raw = vals[k.0 as usize].clone();
                        let vf_raw = vals[v.0 as usize].clone();
                        (qf_raw, kf_raw, vf_raw)
                    };
                    // [rows, nk*kd], [rows, nk*kd], [rows, nv*vd]
                    let bf = vals[b.0 as usize].clone(); // [rows, nv]
                    let af = vals[a.0 as usize].clone();
                    let acoef = weight(a_coef);
                    let dtb = weight(dt_bias);
                    let st = &mut vals[state.0 as usize]; // [nv, kd, vd]
                                                          // The token scan is inherently sequential (state S carries across tokens), but
                                                          // value heads are fully independent — parallelize the per-head t-scan (see
                                                          // `deltanet_scan`). Bit-identical to the old serial nest: same float ops, order.
                    let out = deltanet_scan(
                        self.pool(),
                        &qf,
                        &kf,
                        &vf,
                        &bf,
                        &af,
                        &acoef,
                        &dtb,
                        st,
                        rr,
                        nv,
                        nk,
                        kd,
                        vd,
                        eps,
                    );
                    vals[dst.0 as usize] = out;
                }
            }
            if let Some(t0) = __t0 {
                *op_times.entry(op.kind()).or_insert(0.0) += t0.elapsed().as_secs_f64();
            }
        }
        if prof_ops {
            let mut v: Vec<_> = op_times.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let tot: f64 = v.iter().map(|(_, t)| t).sum();
            eprintln!("[prof-ops] execute totals ({:.1} ms):", tot * 1000.0);
            for (k, t) in v {
                eprintln!("  {k:12} {:7.2} ms  {:5.1}%", t * 1000.0, t / tot * 100.0);
            }
        }

        // Write back the buffers the model reads after execute: Outputs (logits) and mutated f32
        // Inputs (conv/recurrent state). KV caches were written in place by `WriteKv`, so they're
        // skipped — no full-cache copy. Weights are read-only; positions are I32, unchanged. The
        // predicate is the shared `infr_core::exec::writes_back` (metal/rocm select the same set).
        for id in infr_core::exec::write_back_targets(g) {
            if let Some(buf) = bindings.get(id) {
                let mut d = cpu_buf(buf).owned();
                d.copy_from_slice(bytemuck::cast_slice(&vals[id.0 as usize]));
            }
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

/// Depthwise causal conv (fixed per-channel kernel of width `kk`) followed by SiLU, over a rolling
/// history `st` carried across token batches. Returns the `[rows, channels]` output and REBUILDS
/// `st` in place to the trailing `km1 = kk-1` history rows.
///
/// Parallel reformulation (the whole trick): the conv is depthwise + causal with a fixed kernel, so
/// every output reads only the VIRTUAL sequence `[state ‖ x]` and all `rr*cc` outputs are
/// independent — the row loop parallelizes with no cross-row state. With
/// `virtual[i][ch] = state_init[i*cc+ch]` for `i < km1` else `xs[(i-km1)*cc+ch]`,
/// `out[t*cc+ch] = silu(Σ_{j=0}^{kk-1} virtual[t+j][ch] * ws[ch*kk+j])`, and the new history is the
/// trailing `km1` virtual rows: `new_state[j*cc+ch] = virtual[rr+j][ch]`.
///
/// Bit-identity vs the serial shift-register loop: at serial step `t` the shifted `st[j]` equals
/// `virtual[t+j]` (`j in 0..km1`) and the current `xt` equals `virtual[t+km1]`, so accumulating the
/// taps in the SAME ascending `j` order (0..kk) reproduces the serial float accumulation
/// float-for-float. The state rebuild is a pure copy (no arithmetic), so it is trivially identical.
/// Handles `kk == 1` (km1=0, no history: `out[t]=silu(xs[t]*w[0])`) and `rr < km1` (the rebuild
/// reads leftover old state via the `i < km1` branch of `virtual`).
fn conv1d_silu(
    pool: &pool::SpinPool,
    xs: &[f32],
    ws: &[f32],
    st: &mut [f32],
    rows: usize,
    channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let (rr, cc, kk) = (rows, channels, kernel);
    let km1 = kk - 1;
    // Snapshot the incoming history so the parallel closure can read it immutably while we rebuild
    // `st` afterward.
    let state_init = st.to_vec();
    let virt = |i: usize, ch: usize| -> f32 {
        if i < km1 {
            state_init[i * cc + ch]
        } else {
            xs[(i - km1) * cc + ch]
        }
    };
    let mut out = vec![0f32; rr * cc];
    // Chunk `out` by output ROW (chunk == cc → chunk index `t` is the row index). Pure scheduling,
    // math unchanged → bit-identical; grain of 4 matches the sibling row-parallel ops.
    pool.for_chunks_mut(&mut out, cc, 4, &|t, orow| {
        for ch in 0..cc {
            // Taps summed in ASCENDING j (0..kk) to match the serial accumulation order exactly.
            let mut acc = 0f32;
            for j in 0..kk {
                acc += virt(t + j, ch) * ws[ch * kk + j];
            }
            orow[ch] = acc / (1.0 + (-acc).exp()); // silu
        }
    });
    // Rebuild the rolling history = trailing km1 virtual rows (pure copy, no arithmetic).
    for j in 0..km1 {
        for ch in 0..cc {
            st[j * cc + ch] = virt(rr + j, ch);
        }
    }
    out
}

/// Parallel DeltaNet scan: ONE pool task per value head. The outer token scan is inherently
/// sequential (state S carries across tokens), but heads are fully independent — head `h` touches
/// only its own state slice `st[h*kd*vd..]`, writes only its own output columns, and otherwise
/// READS shared inputs (`kh_idx = h % nk` is a read index) — so the whole per-head t-scan runs as
/// one task. Returns `out` [rows, nv*vd]; mutates `st` [nv, kd, vd] in place. Bit-identical to the
/// old serial nest: per head the float ops (l2 norm, normalize, beta, softplus/decay, S*=decay,
/// kv=kᵀS, delta, S+=k⊗delta, out=qᵀS) run in the SAME order — heads never interacted, so
/// cross-head order never mattered, and `pool.collect` is order-preserving pure scheduling.
#[allow(clippy::too_many_arguments)]
fn deltanet_scan(
    pool: &pool::SpinPool,
    qf: &[f32], // [rows, nk*kd]
    kf: &[f32], // [rows, nk*kd]
    vf: &[f32], // [rows, nv*vd]
    bf: &[f32], // [rows, nv]
    af: &[f32], // [rows, nv]
    acoef: &[f32],
    dtb: &[f32],
    st: &mut [f32], // [nv, kd, vd]
    rr: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    eps: f32,
) -> Vec<f32> {
    let qscale = 1.0 / (kd as f32).sqrt();
    let l2 = |slice: &[f32]| -> f32 { (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt() };
    // Read the incoming state immutably inside the closure; `st` is only touched mutably AFTER
    // `collect` returns (NLL ends this shared borrow there), so two tasks never `&mut` the state.
    let st_ro: &[f32] = st;
    // One task per value head: snapshot the head's state, run its full sequential t-scan into a
    // local output buffer, and return both for the serial gather below.
    let per_head: Vec<(Vec<f32>, Vec<f32>)> = pool.collect(nv, &|h| {
        let mut sh = st_ro[h * kd * vd..(h + 1) * kd * vd].to_vec(); // [kd, vd]
        let mut oh_all = vec![0f32; rr * vd];
        // GQA: q/k heads TILED to nv value heads → v-head h uses q/k head h % nk.
        let kh_idx = h % nk;
        // Sequential scan over the rows, carrying this head's state S across tokens.
        for t in 0..rr {
            let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
            let mut qh = qf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
            let mut kh = kf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
            let vh = &vf[vb + h * vd..vb + h * vd + vd];
            let qn = l2(&qh);
            let kn = l2(&kh);
            for x in qh.iter_mut() {
                *x = *x / qn * qscale;
            }
            for x in kh.iter_mut() {
                *x /= kn;
            }
            let beta = 1.0 / (1.0 + (-bf[bb + h]).exp());
            // softplus(a + dt_bias), then g = a_coef * softplus (≤ 0); decay = exp(g).
            let sp = {
                let z = af[bb + h] + dtb[h];
                z.max(0.0) + (-z.abs()).exp().ln_1p()
            };
            let decay = (acoef[h] * sp).exp();
            for x in sh.iter_mut() {
                *x *= decay;
            }
            // kv = kᵀS  [vd]
            let mut kv = vec![0f32; vd];
            for kk in 0..kd {
                let kkv = kh[kk];
                let row = &sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    kv[d] += kkv * row[d];
                }
            }
            // delta = (v - kv)*beta ; S += k ⊗ delta
            let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
            for kk in 0..kd {
                let kkv = kh[kk];
                let row = &mut sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    row[d] += kkv * delta[d];
                }
            }
            // out = qᵀS  [vd]
            let oh = &mut oh_all[t * vd..t * vd + vd];
            for kk in 0..kd {
                let qv = qh[kk];
                let row = &sh[kk * vd..kk * vd + vd];
                for d in 0..vd {
                    oh[d] += qv * row[d];
                }
            }
        }
        (oh_all, sh)
    });
    // Serial gather: copy each head's final state back, and scatter its per-row output columns.
    let mut out = vec![0f32; rr * nv * vd];
    for (h, (oh_all, sh)) in per_head.into_iter().enumerate() {
        st[h * kd * vd..(h + 1) * kd * vd].copy_from_slice(&sh);
        for t in 0..rr {
            let dst = t * nv * vd + h * vd;
            out[dst..dst + vd].copy_from_slice(&oh_all[t * vd..t * vd + vd]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Naive serial reference: a byte-for-byte copy of the ORIGINAL shift-register conv loop. The
    // parallel `conv1d_silu` reformulation must reproduce both `out` and the rebuilt `state` exactly.
    fn conv1d_silu_ref(
        xs: &[f32],
        ws: &[f32],
        st: &mut [f32],
        rr: usize,
        cc: usize,
        kk: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; rr * cc];
        for t in 0..rr {
            let xt = &xs[t * cc..t * cc + cc];
            for ch in 0..cc {
                let mut acc = 0f32;
                for j in 0..kk - 1 {
                    acc += st[j * cc + ch] * ws[ch * kk + j];
                }
                acc += xt[ch] * ws[ch * kk + (kk - 1)];
                out[t * cc + ch] = acc / (1.0 + (-acc).exp());
            }
            for j in 0..kk.saturating_sub(2) {
                for ch in 0..cc {
                    st[j * cc + ch] = st[(j + 1) * cc + ch];
                }
            }
            if kk >= 2 {
                for ch in 0..cc {
                    st[(kk - 2) * cc + ch] = xt[ch];
                }
            }
        }
        out
    }

    // The parallel `conv1d_silu` reformulation must be BIT-IDENTICAL (exact f32 equality) to the
    // naive serial reference for BOTH the output and the rebuilt state, across a matrix of
    // (rows, kernel, channels) that spans: real-prefill sizes, decode (rr==1), rr == kk-1, a case
    // with rr < kk-1 (rebuild reads leftover old state), kk==1 (no history), kk==2, small channel
    // counts, and a large channel count that forces the pool's multi-chunk path.
    #[test]
    fn conv1d_silu_bit_identical_to_serial() {
        // Deterministic pseudo-random pattern (no RNG): a cheap hashed ramp in [-1, 1).
        let pat = |seed: usize, i: usize| -> f32 {
            let h = (i
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503)))
                & 0xffff;
            (h as f32 / 32768.0) - 1.0
        };
        let be = CpuBackend::new();
        let pool = be.pool();
        for &(rr, kk) in &[
            (8usize, 4usize),
            (1, 4),
            (3, 4),
            (16, 4),
            (5, 1),
            (2, 2),
            (2, 4),
        ] {
            for &cc in &[40usize, 512] {
                let km1 = kk - 1;
                let xs: Vec<f32> = (0..rr * cc).map(|i| pat(1, i)).collect();
                let ws: Vec<f32> = (0..cc * kk).map(|i| pat(2, i)).collect();
                let st0: Vec<f32> = (0..km1 * cc).map(|i| pat(3, i)).collect();

                let mut st_ref = st0.clone();
                let out_ref = conv1d_silu_ref(&xs, &ws, &mut st_ref, rr, cc, kk);

                let mut st_par = st0.clone();
                let out_par = conv1d_silu(pool, &xs, &ws, &mut st_par, rr, cc, kk);

                assert_eq!(
                    out_par, out_ref,
                    "output mismatch at rr={rr} kk={kk} cc={cc}"
                );
                assert_eq!(
                    st_par, st_ref,
                    "rebuilt state mismatch at rr={rr} kk={kk} cc={cc}"
                );
            }
        }
    }

    // Naive serial reference: a byte-for-byte copy of the ORIGINAL DeltaNet scan nest (loop order
    // `for t { for h }`). The parallel `deltanet_scan` (loop order `for h (parallel) { for t }`)
    // must reproduce both `out` and the mutated state EXACTLY.
    #[allow(clippy::too_many_arguments)]
    fn deltanet_scan_ref(
        qf: &[f32],
        kf: &[f32],
        vf: &[f32],
        bf: &[f32],
        af: &[f32],
        acoef: &[f32],
        dtb: &[f32],
        st: &mut [f32],
        rr: usize,
        nv: usize,
        nk: usize,
        kd: usize,
        vd: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; rr * nv * vd];
        let qscale = 1.0 / (kd as f32).sqrt();
        let l2 = |slice: &[f32]| -> f32 { (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt() };
        for t in 0..rr {
            let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
            for h in 0..nv {
                let kh_idx = h % nk;
                let mut qh = qf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                let mut kh = kf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                let vh = &vf[vb + h * vd..vb + h * vd + vd];
                let qn = l2(&qh);
                let kn = l2(&kh);
                for x in qh.iter_mut() {
                    *x = *x / qn * qscale;
                }
                for x in kh.iter_mut() {
                    *x /= kn;
                }
                let beta = 1.0 / (1.0 + (-bf[bb + h]).exp());
                let sp = {
                    let z = af[bb + h] + dtb[h];
                    z.max(0.0) + (-z.abs()).exp().ln_1p()
                };
                let decay = (acoef[h] * sp).exp();
                let sh = &mut st[h * kd * vd..(h + 1) * kd * vd];
                for x in sh.iter_mut() {
                    *x *= decay;
                }
                let mut kv = vec![0f32; vd];
                for kk in 0..kd {
                    let kkv = kh[kk];
                    let row = &sh[kk * vd..kk * vd + vd];
                    for d in 0..vd {
                        kv[d] += kkv * row[d];
                    }
                }
                let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
                for kk in 0..kd {
                    let kkv = kh[kk];
                    let row = &mut sh[kk * vd..kk * vd + vd];
                    for d in 0..vd {
                        row[d] += kkv * delta[d];
                    }
                }
                let oh = &mut out[vb + h * vd..vb + h * vd + vd];
                for kk in 0..kd {
                    let qv = qh[kk];
                    let row = &sh[kk * vd..kk * vd + vd];
                    for d in 0..vd {
                        oh[d] += qv * row[d];
                    }
                }
            }
        }
        out
    }

    // The head-parallel `deltanet_scan` must be BIT-IDENTICAL (exact f32 equality) to the naive
    // serial reference for BOTH the output and the mutated state, across a matrix of
    // (rows, n_vhead, n_khead, head_k, head_v) that spans: decode (rr==1) and prefill (rr>1),
    // GQA (nk < nv) and nk==nv, small dims, and a large multi-head case that forces the pool's
    // multi-task path (nv=16, kd=vd=128).
    #[test]
    fn deltanet_scan_bit_identical_to_serial() {
        // Deterministic pseudo-random pattern (no RNG): a cheap hashed ramp in [-1, 1).
        let pat = |seed: usize, i: usize| -> f32 {
            let h = (i
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503)))
                & 0xffff;
            (h as f32 / 32768.0) - 1.0
        };
        let be = CpuBackend::new();
        let pool = be.pool();
        let eps = 1e-6f32;
        for &(rr, nv, nk, kd, vd) in &[
            (1usize, 8usize, 2usize, 16usize, 16usize), // decode, GQA
            (4, 8, 2, 16, 16),                          // prefill, GQA
            (7, 16, 4, 32, 32),                         // prefill, GQA
            (3, 4, 4, 16, 16),                          // nk == nv
            (1, 4, 4, 32, 32),                          // decode, nk == nv
            (5, 16, 16, 128, 128),                      // large, multi-task, nk == nv
            (6, 16, 4, 128, 128),                       // large, multi-task, GQA
        ] {
            let qf: Vec<f32> = (0..rr * nk * kd).map(|i| pat(1, i)).collect();
            let kf: Vec<f32> = (0..rr * nk * kd).map(|i| pat(2, i)).collect();
            let vf: Vec<f32> = (0..rr * nv * vd).map(|i| pat(3, i)).collect();
            let bf: Vec<f32> = (0..rr * nv).map(|i| pat(4, i)).collect();
            let af: Vec<f32> = (0..rr * nv).map(|i| pat(5, i)).collect();
            let acoef: Vec<f32> = (0..nv).map(|i| pat(6, i)).collect();
            let dtb: Vec<f32> = (0..nv).map(|i| pat(7, i)).collect();
            let st0: Vec<f32> = (0..nv * kd * vd).map(|i| pat(8, i)).collect();

            let mut st_ref = st0.clone();
            let out_ref = deltanet_scan_ref(
                &qf,
                &kf,
                &vf,
                &bf,
                &af,
                &acoef,
                &dtb,
                &mut st_ref,
                rr,
                nv,
                nk,
                kd,
                vd,
                eps,
            );

            let mut st_par = st0.clone();
            let out_par = deltanet_scan(
                pool,
                &qf,
                &kf,
                &vf,
                &bf,
                &af,
                &acoef,
                &dtb,
                &mut st_par,
                rr,
                nv,
                nk,
                kd,
                vd,
                eps,
            );

            assert_eq!(
                out_par, out_ref,
                "output mismatch at rr={rr} nv={nv} nk={nk} kd={kd} vd={vd}"
            );
            assert_eq!(
                st_par, st_ref,
                "state mismatch at rr={rr} nv={nv} nk={nk} kd={kd} vd={vd}"
            );
        }
    }

    // #1: `sample_token` must not panic when top_k == 0 (the "disable top-k" sentinel) and must
    // return a valid in-range token. Regression for the `select_nth_unstable_by(k - 1)` underflow.
    #[test]
    fn sample_top_k_zero_no_panic() {
        let logits = [0.1f32, 3.0, 0.2, 2.0, 0.05];
        // top_k == 0 => no truncation. Greedy-ish draw (u≈0) should land on the argmax (index 1).
        let tok = sample_token(&logits, 0.0, 0, 1.0, 1.0);
        assert!((tok as usize) < logits.len(), "token {tok} out of range");
        assert_eq!(tok, 1, "u≈0 with full softmax picks the max-logit index");
        // u≈1 walks the CDF to the last (sorted) entry — still a valid index.
        let tok_hi = sample_token(&logits, 0.999_999, 0, 1.0, 1.0);
        assert!((tok_hi as usize) < logits.len());
    }

    // #1: top_k == 0 (no truncation) and top_k == len must agree — both consider every logit.
    #[test]
    fn sample_top_k_zero_equals_full_k() {
        let logits = [1.0f32, 5.0, 2.0, 4.0, 3.0, 0.5];
        for &u in &[0.0f32, 0.25, 0.5, 0.75, 0.999] {
            let a = sample_token(&logits, u, 0, 0.8, 0.95);
            let b = sample_token(&logits, u, logits.len(), 0.8, 0.95);
            assert_eq!(a, b, "top_k==0 must match top_k==len at u={u}");
        }
    }

    // #1: a normal top_k >= 1 still truncates and returns a valid token (path unchanged).
    #[test]
    fn sample_top_k_truncates() {
        let logits = [0.0f32, 9.0, 1.0, 8.0, 2.0];
        let tok = sample_token(&logits, 0.0, 2, 1.0, 1.0);
        // Only indices {1,3} survive top_k=2; u≈0 picks the larger (index 1).
        assert_eq!(tok, 1);
    }

    // #2: `cached_weight` keys on (addr, len, dtype). Same key returns the CACHED value (even if the
    // bytes changed — proves it dedups); a key differing in len or dtype MISSES and dequants fresh
    // (proves a reused address can't collide with a stale, differently-shaped weight).
    #[test]
    fn cached_weight_keys_on_addr_len_dtype() {
        let be = CpuBackend::new();
        let a = [1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let key = (0xDEAD_BEEF, a.len(), DType::F32);
        let w1 = be.cached_weight(key, &a);
        assert_eq!(&**w1, &[1.0, 2.0]);

        // Same key, DIFFERENT bytes => returns the cached f32 (not re-dequantized).
        let b = [9.0f32, 9.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let w2 = be.cached_weight(key, &b);
        assert_eq!(&**w2, &[1.0, 2.0], "same key must hit the cache");

        // Same address+len but a DIFFERENT dtype => distinct key => fresh dequant (no collision).
        let c: Vec<u8> = [3i32, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        let key_i32 = (0xDEAD_BEEF, c.len(), DType::I32);
        let w3 = be.cached_weight(key_i32, &c);
        assert_eq!(
            &**w3,
            &[3.0, 4.0],
            "different dtype must miss and dequant fresh"
        );

        // Same address+dtype but a DIFFERENT byte-len => distinct key => fresh (simulates a reload
        // that reused a freed weight's address for a longer weight).
        let d = [7.0f32, 8.0, 9.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>();
        let key_len = (0xDEAD_BEEF, d.len(), DType::F32);
        let w4 = be.cached_weight(key_len, &d);
        assert_eq!(
            &**w4,
            &[7.0, 8.0, 9.0],
            "different len must miss and dequant fresh"
        );
    }
}
