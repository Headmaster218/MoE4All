//! Tensor parallelism (dense) — Megatron-style intra-op weight sharding across several physical
//! Vulkan devices, so a single autoregressive stream's weight-GEMV (decode's ~77% bottleneck) is
//! SPLIT across the devices instead of replicated. A DIFFERENT decomposition from the layer-split
//! [`crate::PipelineBackend`] (which places whole layers on whole devices and hands the residual
//! across once per boundary): TP shards EACH layer's weight matrices and reduces per layer.
//!
//! # Sharding scheme (comms-minimal — ONE all-reduce per attention + ONE per FFN, per layer)
//!
//! * **Attention** — the q/k/v projections are **column-parallel**: device `r` owns a contiguous
//!   slice of the attention HEADS (rows `[r·qrow/W, (r+1)·qrow/W)` of `attn_q.weight`, and the
//!   matching kv-head slice of `attn_k`/`attn_v`), so it computes its own heads' Q/K/V, its own
//!   RoPE/attention, and holds its own heads' KV (KV is sharded by head — no replication). The
//!   output projection `attn_output.weight` is **row-parallel** (device `r` owns input-columns
//!   `[r·qrow/W, …)`), so each device produces a PARTIAL `[tokens, n_embd]` sum → **all-reduce** →
//!   the full attention output, identical on every device.
//! * **FFN** — gate/up are **column-parallel** (each device owns a slice of the intermediate dim),
//!   down is **row-parallel** (partial `[tokens, n_embd]`) → **all-reduce**.
//! * **Everything else** (embeddings, norms, residual add, rope tables, the LM head) is REPLICATED
//!   and bit-identical on every device: the residual stream is the same on all devices after each
//!   all-reduce, so replicated ops need no communication.
//!
//! # How it stays invisible above the [`Backend`] seam
//!
//! Exactly like [`crate::PipelineBackend`], the model runner (`generate_dense_backend`) drives this
//! as ONE backend. The runner builds the ordinary single-device graph (global dims); at
//! [`TensorParallelBackend::execute`] the full graph is LOWERED once (cached in the plan) into a
//! per-rank sharded graph (sharded op dims + resized scratch decls) and split into SEGMENTS at each
//! row-parallel projection. Each segment runs on every rank; between segments the boundary partial
//! is all-reduced. The lowering derives each op's shard treatment from the DEVICE ROLE of its bound
//! weight (column/row/replicated — attached by the TP binder) forward-propagated through the
//! activation tensors, so no op-label heuristics are needed.
//!
//! # The all-reduce (the optimization-critical piece)
//!
//! Data crosses devices via the host-LESS P2P dma-buf transport ([`crate::VulkanBackend::p2p_export`]
//! / [`p2p_import`](crate::VulkanBackend::p2p_import)), never a host bounce. Cross-device ordering is
//! by `VK_KHR_external_semaphore_fd` when the device pair supports it (device B's read waits on
//! device A's GPU-side signal with NO host round-trip — see [`crate::tp_allreduce`]); a device pair
//! that can't import an external semaphore falls back to the host-fence (`queue_wait_idle`) path,
//! which is reported as the per-layer host-stall perf gap. Two sync points per layer (one after
//! attn-O, one after FFN-down); the reduction is a fixed device-order sum so it is deterministic
//! run-to-run.
//!
//! # Scope (v1, correctness-first)
//!
//! Dense attention models only (Qwen3/Llama/Gemma-dense); MoE / qwen35 DeltaNet / gemma-E2B /
//! diffusion-gemma are rejected up front (they need per-op sharding beyond this slice). The
//! GPU-resident fast paths (decode replay / embed-gather / gpu-sample / argmax) AND the fused
//! qkv / gate-up weight concatenations are turned OFF via [`capabilities`], so the runner takes the
//! classic host-embed + host-sample static path over SEPARATE q/k/v/gate/up projections — each a
//! cleanly sliceable weight + op. Generalizes to any world size `W` that divides `n_head`, `n_kv`
//! and `n_ff`; the all-reduce data path is `W`-general, only its ring/tree schedule optimization is
//! deferred (v1 does an all-to-all-ish exchange).

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan};
use infr_core::error::Result;
use infr_core::graph::{Graph, Op, TensorKind};
use infr_core::tensor::{DType, TensorDesc, TensorId};

use crate::tp_allreduce::AllReduce;
use crate::{be, VulkanBackend};

/// Device role of a tensor-parallel weight, attached by the TP binder (`tensor_parallel_binder`) so
/// the lowering knows, per weight, how the projection that reads it is sharded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpRole {
    /// Column-parallel: each rank owns a contiguous slice of the projection's OUTPUT rows
    /// (`out_f/W`). q/k/v/gate/up. The projection's output activation becomes SHARDED.
    Column,
    /// Row-parallel: each rank owns a slice of the projection's INPUT columns (`in_f/W`), consuming
    /// a sharded input and producing a PARTIAL full-width sum. attn_output / ffn_down. Followed by
    /// an all-reduce.
    Row,
    /// Replicated: the full tensor lives on every rank, identical (norms, biases, embeddings, the
    /// LM head). A projection reading it is replicated (Rep in → Rep out).
    Replicated,
}

/// How a [`TpBuffer`]'s per-rank buffers relate to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TpKind {
    /// Identical copy on every rank (replicated weights, the residual/positions IO, logits).
    Replica,
    /// A weight sliced per rank (`bufs[r]` is rank `r`'s column/row slice — DIFFERENT bytes).
    Weight(TpRole),
    /// A per-rank KV shard (`bufs[r]` holds rank `r`'s heads' KV — different bytes, same layout).
    Kv,
}

/// A buffer owned by a [`TensorParallelBackend`]: `bufs[r]` is the buffer on rank `r`. Replicas hold
/// identical bytes; weight/KV shards hold each rank's own slice. Always unwrapped to the per-rank
/// inner `VkBuffer` before it reaches a sub-backend.
pub struct TpBuffer {
    kind: TpKind,
    bufs: Vec<Box<dyn Buffer>>,
}

impl TpBuffer {
    /// A replicated buffer (identical on every rank).
    pub fn replica(bufs: Vec<Box<dyn Buffer>>) -> Box<dyn Buffer> {
        Box::new(TpBuffer {
            kind: TpKind::Replica,
            bufs,
        })
    }

    /// A per-rank sliced weight with the given device role.
    pub fn weight(role: TpRole, bufs: Vec<Box<dyn Buffer>>) -> Box<dyn Buffer> {
        Box::new(TpBuffer {
            kind: TpKind::Weight(role),
            bufs,
        })
    }

    fn on(&self, r: usize) -> Result<&dyn Buffer> {
        self.bufs
            .get(r)
            .map(|b| b.as_ref())
            .ok_or_else(|| be(format!("tp: no buffer for rank {r}")))
    }
}

impl Buffer for TpBuffer {
    /// The LOGICAL tensor size (see [`Buffer::len_bytes`]), NOT rank 0's slice: a sharded buffer
    /// reports the SUM over its per-rank buffers, so a caller above the seam sees the same byte
    /// count it would on a single device. Only [`TensorParallelBackend`] itself divides by the
    /// world, and only in [`TensorParallelBackend::shard_bytes`].
    ///
    /// This USED to return `bufs[0].len_bytes()` (the per-rank slice), which broke both callers
    /// that mix buffer-derived and config-derived byte counts: `SeamKv::mtp_snapshot_delta` fed
    /// `len_bytes()` back into `alloc(KvCache)`, which sharded the ALREADY-sharded size into
    /// `full/W²` buffers, and then copied `full/W` bytes into them. Summing here is what makes the
    /// alloc/copy round trip (`shard_bytes(tp.len_bytes()) == per-rank size`) hold.
    ///
    /// The sum — rather than `bufs[0].len_bytes() * W` — so a ragged split (if one is ever allowed)
    /// reports what the ranks actually hold instead of an idealized multiple.
    fn len_bytes(&self) -> usize {
        match self.kind {
            // A replica's per-rank buffer IS the whole logical tensor (every rank holds all of it).
            TpKind::Replica => self.bufs[0].len_bytes(),
            TpKind::Kv | TpKind::Weight(_) => self.bufs.iter().map(|b| b.len_bytes()).sum(),
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device_addr(&self) -> Option<u64> {
        // A weight/KV shard's per-rank inner buffer may carry a BDA, but a TpBuffer is never itself
        // addressed as one contiguous thing — the executor always resolves to a per-rank inner.
        None
    }
}

fn as_tp(buf: &dyn Buffer) -> Result<&TpBuffer> {
    buf.as_any()
        .downcast_ref::<TpBuffer>()
        .ok_or_else(|| be("tp: a buffer bound to a TensorParallelBackend was not a TpBuffer"))
}

/// Shard state of an ACTIVATION tensor as the lowering forward-propagates through the op list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shard {
    /// Full dims, identical on every rank (the residual stream, norms' output, logits).
    Rep,
    /// Split along its feature dim by `W` (q/k/v/attn = by head; gate/up/act = by intermediate dim;
    /// KV = by head). Each rank holds `feat/W`.
    Sharded,
    /// Full dims but a PARTIAL sum — the output of a row-parallel projection, pending all-reduce.
    Partial,
}

/// One contiguous run of ops that runs on every rank, plus (except for the last) the boundary tensor
/// to all-reduce after it.
struct Segment {
    /// One compiled plan per rank for this segment's subgraph (identical structure, run per device).
    plans: Vec<Box<dyn Plan>>,
    /// Tensors this segment's ops read/write that must be bound (weights, KV, IO, boundary sub).
    needed: Vec<TensorId>,
    /// The boundary (Partial→Rep) tensor produced at the end of this segment, or `None` for the last.
    reduce: Option<TensorId>,
}

/// The lowered + partitioned form of a graph, built once per plan on first execute.
struct Prepared {
    /// The lowered per-rank graph (dims sharded, scratch decls resized, boundary tensors promoted to
    /// Input). Structurally identical across ranks; only bound weight/KV bytes differ per rank.
    lowered: Graph,
    segments: Vec<Segment>,
    /// Boundary tensors promoted to bound Inputs: `tid -> per-rank persistent buffer`.
    boundary: HashMap<TensorId, Vec<Box<dyn Buffer>>>,
    /// The cross-device all-reduce transport, set up over the (single) boundary tensor size.
    allreduce: AllReduce,
}

/// A [`TensorParallelBackend`] plan: the full graph + its lazily-built lowering.
pub struct TpPlan {
    graph: Graph,
    prepared: Mutex<Option<Prepared>>,
}

impl Plan for TpPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Multi-device tensor-parallel (dense) backend. Owns `W` real Vulkan devices (ranks) and presents
/// one [`Backend`] to the runner; each layer's weight matmuls are sharded across the ranks with a
/// P2P all-reduce per attention + per FFN.
pub struct TensorParallelBackend {
    ranks: Vec<VulkanBackend>,
    /// Try the host-less external-memory (dma-buf) all-reduce transport; falls back per pair.
    use_p2p: bool,
}

impl TensorParallelBackend {
    /// Build a `W`-way tensor-parallel backend over `ranks`. `n_head`/`n_kv`/`n_ff` are the model's
    /// (global) attention-head, kv-head and FFN-intermediate counts — each MUST be divisible by
    /// `W = ranks.len()` for a clean per-head / per-intermediate split.
    pub fn new(
        ranks: Vec<VulkanBackend>,
        n_head: usize,
        n_kv: usize,
        n_ff: usize,
        use_p2p: bool,
    ) -> Result<Self> {
        if ranks.is_empty() {
            return Err(be("tp: needs at least one device"));
        }
        let w = ranks.len();
        for (name, v) in [("n_head", n_head), ("n_kv", n_kv), ("n_ff", n_ff)] {
            if !v.is_multiple_of(w) {
                return Err(be(format!(
                    "tp: {name}={v} is not divisible by the {w}-way tensor-parallel world — TP needs \
                     n_head, n_kv and n_ff all divisible by the device count"
                )));
            }
        }
        Ok(Self { ranks, use_p2p })
    }

    pub fn world(&self) -> usize {
        self.ranks.len()
    }

    /// The real backend for rank `r` — the binder allocs/uploads each per-rank weight slice through it.
    pub fn rank(&self, r: usize) -> &VulkanBackend {
        &self.ranks[r]
    }

    /// Per-rank device names, for the placement report and the per-rank-shard confirmation.
    pub fn device_names(&self) -> Vec<String> {
        self.ranks
            .iter()
            .map(|b| b.capabilities().name.clone())
            .collect()
    }

    /// Replicate: run `f` on every rank, collecting one buffer per rank into a [`TpBuffer`] replica.
    fn replicate<F>(&self, f: F) -> Result<Box<dyn Buffer>>
    where
        F: Fn(&VulkanBackend) -> Result<Box<dyn Buffer>>,
    {
        let mut bufs = Vec::with_capacity(self.ranks.len());
        for b in &self.ranks {
            bufs.push(f(b)?);
        }
        Ok(TpBuffer::replica(bufs))
    }

    /// The device role attached to a bound weight, or [`TpRole::Replicated`] for anything that isn't
    /// a per-rank-sliced weight (a replica, a KV shard, an IO tensor).
    fn weight_role(buf: &dyn Buffer) -> TpRole {
        match as_tp(buf) {
            Ok(tp) => match tp.kind {
                TpKind::Weight(role) => role,
                _ => TpRole::Replicated,
            },
            Err(_) => TpRole::Replicated,
        }
    }

    /// Turn a LOGICAL KV byte count (what a caller above the seam names — see
    /// [`Buffer::len_bytes`]) into the per-rank byte count. THE one place the world divides a byte
    /// count: `alloc`, `alloc_uninit` and `copy_buffer` all route through it, so the size a KV
    /// buffer is allocated at and the length later copied into it cannot disagree by construction.
    /// (They did: the two alloc arms each carried their own inline copy of this check while
    /// `copy_buffer` forwarded `bytes` to every rank unsharded, a `W`-fold overrun per rank.)
    ///
    /// KV is sharded by head: each rank stores 1/W of every row (its heads' K/V), which is exactly
    /// `bytes/W` of a `[ctx, kvrow]` cache since `kvrow` is divisible by W. Non-divisible is an
    /// ERROR, never a truncation — the case that reaches it is a quantized KV whose per-row padding
    /// is not aligned to the head split, where `bytes/W != kv_fmt_bytes(fmt, elems/W)` and no
    /// division is right.
    ///
    /// The arithmetic lives in the free [`shard_kv_bytes`] so it is testable without `W` physical
    /// devices (a `TensorParallelBackend` cannot be built without them, and TP needs >= 2 GPUs, so
    /// none of this runs in CI otherwise).
    fn shard_bytes(&self, bytes: usize) -> Result<usize> {
        shard_kv_bytes(bytes, self.world())
    }

    /// Whether a bound tensor is a per-rank KV shard (its activation state starts SHARDED).
    fn is_kv(buf: &dyn Buffer) -> bool {
        matches!(as_tp(buf).map(|t| t.kind), Ok(TpKind::Kv))
    }

    /// Lower `full` (the runner's global-dim graph) into the per-rank sharded graph + segment
    /// partition + boundary/all-reduce transport. Called once per plan (needs the bindings — hence
    /// each weight's device role — which are known only at execute).
    fn prepare(&self, full: &Graph, bindings: &Bindings) -> Result<Prepared> {
        let w = self.world();
        let n_ops = full.ops.len();

        // ── forward-propagate each tensor's shard state + record dim rewrites ──────────────────
        // Seed: bound tensors take their state from their TpBuffer kind (KV shard ⇒ Sharded, else
        // Rep). Unbound activations get Rep and are refined as ops write them.
        let mut state: HashMap<TensorId, Shard> = HashMap::new();
        for (i, decl) in full.tensors.iter().enumerate() {
            let tid = TensorId(i as u32);
            let s = match decl.kind {
                TensorKind::Input | TensorKind::Weight => match bindings.get(tid) {
                    Some(b) if Self::is_kv(b) => Shard::Sharded,
                    _ => Shard::Rep,
                },
                _ => Shard::Rep,
            };
            state.insert(tid, s);
        }

        // The lowered graph starts as a clone; we rewrite op dims + shrink Sharded scratch decls.
        let mut lowered = full.clone();
        // Boundary tensors (a row-parallel Linear's dst): reduced across ranks, promoted to Input.
        let mut boundary: HashSet<TensorId> = HashSet::new();
        // op indices AFTER which a segment cut + all-reduce happens (the row-parallel Linears).
        let mut cuts: Vec<usize> = Vec::new();

        let wu = w as u32;
        for i in 0..n_ops {
            // Read the branch operands + their input shard states up front (immutable), then take a
            // fresh `&mut lowered.ops[i]` for the dim rewrite, then record the output state. Keeping
            // the reads/writes disjoint avoids aliasing `state` while mutating it.
            match lowered.ops[i].clone() {
                Op::Linear { x, weight, dst, .. } => {
                    let sx = shard_of(&state, x);
                    let role = match bindings.get(weight) {
                        Some(b) => Self::weight_role(b),
                        None => TpRole::Replicated,
                    };
                    match role {
                        TpRole::Column => {
                            require(
                                sx == Shard::Rep,
                                i,
                                "column-parallel Linear needs a replicated input",
                            )?;
                            if let Op::Linear { out_f, w_off, .. } = &mut lowered.ops[i] {
                                require(
                                    *w_off == 0,
                                    i,
                                    "column-parallel Linear must be unfused (w_off==0) under TP",
                                )?;
                                *out_f = shard_dim(*out_f, wu, i, "column-parallel out_f")?;
                            }
                            state.insert(dst, Shard::Sharded);
                        }
                        TpRole::Row => {
                            require(
                                sx == Shard::Sharded,
                                i,
                                "row-parallel Linear needs a sharded input",
                            )?;
                            if let Op::Linear { in_f, .. } = &mut lowered.ops[i] {
                                *in_f = shard_dim(*in_f, wu, i, "row-parallel in_f")?;
                            }
                            state.insert(dst, Shard::Rep); // Rep AFTER the all-reduce recorded below
                            boundary.insert(dst);
                            cuts.push(i);
                        }
                        TpRole::Replicated => {
                            require(
                                sx == Shard::Rep,
                                i,
                                "replicated Linear needs a replicated input",
                            )?;
                            state.insert(dst, Shard::Rep);
                        }
                    }
                }
                Op::RmsNorm { x, dst, .. } | Op::RmsNormAdd { x, dst, .. } => {
                    require(
                        shard_of(&state, x) == Shard::Rep,
                        i,
                        "RmsNorm on a sharded tensor is unsupported under TP",
                    )?;
                    state.insert(dst, Shard::Rep);
                }
                Op::QkNorm { x, dst, .. }
                | Op::QkNormRope { x, dst, .. }
                | Op::Rope { x, dst, .. }
                | Op::GatedRmsNorm { x, dst, .. } => {
                    let sx = shard_of(&state, x);
                    if sx == Shard::Sharded {
                        shard_n_head(&mut lowered.ops[i], wu, i)?;
                    }
                    state.insert(dst, sx);
                }
                Op::WriteKv { src, cache, .. } => {
                    if shard_of(&state, src) == Shard::Sharded {
                        if let Op::WriteKv { row_stride, .. } = &mut lowered.ops[i] {
                            *row_stride = shard_dim(*row_stride, wu, i, "WriteKv row_stride")?;
                        }
                    }
                    // The cache tensor is a per-rank KV shard (state already Sharded from binding).
                    state.insert(cache, Shard::Sharded);
                }
                Op::Attention { q, dst, .. } => {
                    let sq = shard_of(&state, q);
                    if sq == Shard::Sharded {
                        if let Op::Attention { n_head, n_kv, .. } = &mut lowered.ops[i] {
                            *n_head = shard_dim(*n_head, wu, i, "Attention n_head")?;
                            *n_kv = shard_dim(*n_kv, wu, i, "Attention n_kv")?;
                        }
                    }
                    state.insert(dst, sq);
                }
                Op::GatedAct { gate, dst, .. } => {
                    let sg = shard_of(&state, gate);
                    if sg == Shard::Sharded {
                        if let Op::GatedAct { nff, .. } = &mut lowered.ops[i] {
                            *nff = shard_dim(*nff, wu, i, "GatedAct nff")?;
                        }
                    }
                    state.insert(dst, sg);
                }
                Op::GatedActFused { gu, dst, .. } => {
                    let sg = shard_of(&state, gu);
                    if sg == Shard::Sharded {
                        if let Op::GatedActFused { nff, .. } = &mut lowered.ops[i] {
                            *nff = shard_dim(*nff, wu, i, "GatedActFused nff")?;
                        }
                    }
                    state.insert(dst, sg);
                }
                Op::Add { a, b, dst, .. } => {
                    require(
                        shard_of(&state, a) != Shard::Partial
                            && shard_of(&state, b) != Shard::Partial,
                        i,
                        "residual Add on a partial (un-all-reduced) tensor",
                    )?;
                    state.insert(dst, Shard::Rep);
                }
                Op::AddBias { x, dst, .. }
                | Op::Scale { x, dst, .. }
                | Op::MulVec { x, dst, .. }
                | Op::Softcap { x, dst, .. }
                | Op::Softmax { x, dst, .. }
                | Op::Copy { src: x, dst, .. }
                | Op::CopyStrided { src: x, dst, .. } => {
                    require(
                        shard_of(&state, x) == Shard::Rep,
                        i,
                        "elementwise/copy op on a sharded tensor is unsupported under TP",
                    )?;
                    state.insert(dst, Shard::Rep);
                }
                // LM-head tail sampling ops (the greedy device argmax / device sample): the logits
                // they read are Rep (the LM head is replicated), so they run identically on every
                // rank and their id/prob outputs are Rep (the runner reads rank 0's).
                Op::Argmax { x, dst, .. } | Op::Sample { x, dst, .. } => {
                    require(
                        shard_of(&state, x) == Shard::Rep,
                        i,
                        "argmax/sample on a sharded tensor is unsupported under TP",
                    )?;
                    state.insert(dst, Shard::Rep);
                }
                Op::ArgmaxProb {
                    x,
                    dst_id,
                    dst_prob,
                    ..
                } => {
                    require(
                        shard_of(&state, x) == Shard::Rep,
                        i,
                        "argmax-prob on a sharded tensor is unsupported under TP",
                    )?;
                    state.insert(dst_id, Shard::Rep);
                    state.insert(dst_prob, Shard::Rep);
                }
                Op::EmbedGather { dst, .. } => {
                    // embed_gather is disabled via caps; a replicated gather if it ever appears.
                    state.insert(dst, Shard::Rep);
                }
                other => {
                    return Err(be(format!(
                        "tp: op {i} ({}) is not supported by tensor parallelism (dense attention \
                         models only — MoE / DeltaNet / conv are rejected)",
                        other.kind()
                    )));
                }
            }
        }

        // ── resize sharded scratch + weight + KV decls, promote boundary tensors to Input ──────
        // A sharded scratch tensor's buffer is `numel/W` (the rank's feature slice); a column/row
        // weight's bound buffer is `numel/W` (the rank's weight slice); a per-rank KV shard's bound
        // buffer is `numel/W` (the rank's heads' KV — `alloc(KvCache)` splits `bytes/W`). Shrink all
        // three decls to match the buffers the binder/backend actually provide, so the sub-backend's
        // size math agrees AND any KV-capacity/overflow guard reading `desc.numel()`/`row_stride`
        // sees the true per-rank capacity (not W× it).
        for (i, decl) in lowered.tensors.iter_mut().enumerate() {
            let tid = TensorId(i as u32);
            if boundary.contains(&tid) {
                // The boundary sub is Rep-sized (full n_embd) but must be a bound persistent buffer
                // (it crosses segment/execute boundaries and is written by the all-reduce), so it is
                // promoted from Internal to Input; we supply its buffer.
                decl.kind = TensorKind::Input;
                continue;
            }
            let sharded_weight = decl.kind == TensorKind::Weight
                && matches!(
                    bindings.get(tid).map(Self::weight_role),
                    Some(TpRole::Column | TpRole::Row)
                );
            let sharded_scratch =
                decl.kind == TensorKind::Internal && state.get(&tid) == Some(&Shard::Sharded);
            // A per-rank KV shard (bound TpBuffer of kind Kv) — its decl was left full-width.
            let sharded_kv = bindings.get(tid).map(Self::is_kv).unwrap_or(false);
            if sharded_weight || sharded_scratch || sharded_kv {
                let n = decl.desc.numel();
                if !n.is_multiple_of(w) {
                    return Err(be(format!(
                        "tp: sharded tensor {i} numel={n} not divisible by world {w}"
                    )));
                }
                decl.desc = TensorDesc::new(vec![n / w], decl.desc.dtype);
            }
        }

        // ── build segments (cut AFTER each row-parallel Linear) + compile per rank ─────────────
        let mut bounds: Vec<usize> = cuts.iter().map(|&c| c + 1).collect();
        if bounds.last() != Some(&n_ops) {
            bounds.push(n_ops);
        }
        let mut segments = Vec::with_capacity(bounds.len());
        let mut lo = 0usize;
        for (si, &hi) in bounds.iter().enumerate() {
            let mut sub = lowered.clone();
            sub.ops = lowered.ops[lo..hi].to_vec();
            let written: HashSet<TensorId> = sub.ops.iter().flat_map(|op| op.io().1).collect();
            sub.outputs = lowered
                .outputs
                .iter()
                .copied()
                .filter(|t| written.contains(t))
                .collect();
            sub.no_decode_replay = true;
            let mut plans = Vec::with_capacity(w);
            for r in 0..w {
                plans.push(self.ranks[r].compile(&sub)?);
            }
            // Tensors this segment must have bound.
            let mut needed: HashSet<TensorId> = HashSet::new();
            for op in &sub.ops {
                let (rd, wr) = op.io();
                needed.extend(rd);
                needed.extend(wr);
            }
            // The reduce boundary for this segment is the cut op's dst (None for the last segment).
            let reduce = if si + 1 < bounds.len() {
                let cut_op = hi - 1;
                match &lowered.ops[cut_op] {
                    Op::Linear { dst, .. } => Some(*dst),
                    _ => None,
                }
            } else {
                None
            };
            segments.push(Segment {
                plans,
                needed: needed.into_iter().collect(),
                reduce,
            });
            lo = hi;
        }

        // ── boundary persistent buffers (one per rank) + the all-reduce transport ──────────────
        // Every boundary tensor is the SAME `sub` reused across layers with identical size, so one
        // set of per-rank buffers + one all-reduce transport serves them all. The all-reduce compiles
        // ONE reduce plan and `reduce` requires `elems == self.elems` exactly, so all boundaries MUST
        // be the same byte size (asserted here, not silently `max`-ed) and the same dtype.
        let mut boundary_bufs: HashMap<TensorId, Vec<Box<dyn Buffer>>> = HashMap::new();
        let mut sizes: Vec<usize> = Vec::with_capacity(boundary.len());
        let mut reduce_dtype = DType::F32;
        for &tid in &boundary {
            let desc = lowered.desc(tid);
            reduce_dtype = desc.dtype;
            let bytes = desc.numel() * crate::tp_allreduce::dtype_bytes(desc.dtype);
            sizes.push(bytes);
            let mut bufs = Vec::with_capacity(w);
            for r in 0..w {
                bufs.push(self.ranks[r].alloc(bytes, BufferUsage::Activations)?);
            }
            boundary_bufs.insert(tid, bufs);
        }
        let reduce_bytes = crate::tp_allreduce::uniform_boundary_bytes(&sizes)?;
        let allreduce = AllReduce::new(&self.ranks, reduce_bytes, reduce_dtype, self.use_p2p)?;

        Ok(Prepared {
            lowered,
            segments,
            boundary: boundary_bufs,
            allreduce,
        })
    }
}

/// The shard state of a tensor (defaults to Rep for a tensor no op has written yet).
fn shard_of(state: &HashMap<TensorId, Shard>, t: TensorId) -> Shard {
    *state.get(&t).unwrap_or(&Shard::Rep)
}

/// Divide the `n_head` field of a per-head op (QkNorm / QkNormRope / Rope / GatedRmsNorm) by the
/// world — the shared "shard a head-parallel op" rewrite.
fn shard_n_head(op: &mut Op, w: u32, i: usize) -> Result<()> {
    let nh = match op {
        Op::QkNorm { n_head, .. }
        | Op::QkNormRope { n_head, .. }
        | Op::Rope { n_head, .. }
        | Op::GatedRmsNorm { n_head, .. } => n_head,
        _ => return Err(be(format!("tp: op {i}: shard_n_head on a non-head op"))),
    };
    *nh = shard_dim(*nh, w, i, "n_head")?;
    Ok(())
}

/// The pure core of [`TensorParallelBackend::shard_bytes`] — logical KV byte count → per-rank byte
/// count over a `w`-way world. Split out from the method purely so it can be unit-tested without
/// opening `w` physical Vulkan devices.
fn shard_kv_bytes(bytes: usize, w: usize) -> Result<usize> {
    if !bytes.is_multiple_of(w) {
        return Err(be(format!(
            "tp: KV cache byte count {bytes} not divisible by world {w} (a quantized KV padding \
             not aligned to the head split) — use the default f16 KV under TP"
        )));
    }
    Ok(bytes / w)
}

/// Divide a dim field by the world, erroring (not truncating) on a non-divisible dim.
fn shard_dim(v: u32, w: u32, op: usize, what: &str) -> Result<u32> {
    if !v.is_multiple_of(w) {
        return Err(be(format!(
            "tp: op {op}: {what}={v} not divisible by world {w}"
        )));
    }
    Ok(v / w)
}

fn require(cond: bool, op: usize, msg: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(be(format!("tp: op {op}: {msg}")))
    }
}

impl Backend for TensorParallelBackend {
    fn name(&self) -> &str {
        "vulkan-tensor-parallel"
    }

    fn capabilities(&self) -> Capabilities {
        // Rank 0's caps with every fast path the TP lowering does not model turned OFF: the runner
        // then takes the classic host-embed + host-sample static path over SEPARATE q/k/v/gate/up
        // projections (combined_gu=false forces fuse_qkv AND fuse_gu off — each projection is a
        // cleanly sliceable weight + op), which is the only shape TP shards.
        let mut c = self.ranks[0].capabilities();
        c.decode_replay = false;
        c.embed_gather = false;
        c.gpu_sample = false;
        c.argmax_rows = false;
        c.argmax_prob = false;
        c.combined_gu = false;
        c.gated_rmsnorm = false;
        c
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        match usage {
            BufferUsage::KvCache => {
                // `bytes` is the LOGICAL (full-width) cache size; shard_bytes takes it to this
                // rank's heads' slice. The resulting TpBuffer reports the logical size again from
                // len_bytes(), so a caller that re-allocs from it does NOT shard a second time.
                let w = self.world();
                let per = self.shard_bytes(bytes)?;
                let mut bufs = Vec::with_capacity(w);
                for r in 0..w {
                    bufs.push(self.ranks[r].alloc(per, usage)?);
                }
                Ok(Box::new(TpBuffer {
                    kind: TpKind::Kv,
                    bufs,
                }))
            }
            // Weights bypass alloc via the binder; everything else (Staging hidden/positions,
            // Activations, Readback logits) is replicated identically to every rank.
            _ => self.replicate(|b| b.alloc(bytes, usage)),
        }
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        match usage {
            BufferUsage::KvCache => {
                let w = self.world();
                let per = self.shard_bytes(bytes)?;
                let mut bufs = Vec::with_capacity(w);
                for r in 0..w {
                    bufs.push(self.ranks[r].alloc_uninit(per, usage)?);
                }
                Ok(Box::new(TpBuffer {
                    kind: TpKind::Kv,
                    bufs,
                }))
            }
            _ => self.replicate(|b| b.alloc_uninit(bytes, usage)),
        }
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let tp = as_tp(dst)?;
        match tp.kind {
            // Replica / IO: the same bytes to every rank.
            TpKind::Replica => {
                for (b, buf) in self.ranks.iter().zip(&tp.bufs) {
                    b.upload(buf.as_ref(), src)?;
                }
                Ok(())
            }
            // A KV / weight shard is filled by the binder or the kernels, never host-uploaded whole.
            _ => Err(be("tp: host upload of a sharded buffer is not supported")),
        }
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let tp = as_tp(src)?;
        // Replicated buffers (logits) are identical on every rank — read rank 0.
        self.ranks[0].download(tp.on(0)?, dst)
    }

    /// `bytes` is a LOGICAL length (see [`Buffer::len_bytes`]) — the caller names the full-width
    /// tensor, whether it read the count off a buffer (`SeamKv::mtp_snapshot_delta`) or computed it
    /// from the model config (`SeamKv::seed_from`, `p * n_kv * head_dim` — full, unsharded
    /// geometry). Only the config-derived case forces the issue: there is no buffer to ask, so the
    /// sharding MUST happen here. It did not, and every rank was handed a full-width length for a
    /// `full/W`-sized buffer — a `W`-fold overrun, today an error from `check_extent` rather than
    /// silent corruption of whatever VRAM followed the cache.
    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        let (s, d) = (as_tp(src)?, as_tp(dst)?);
        if s.bufs.len() != d.bufs.len() {
            return Err(be("tp: copy_buffer between mismatched TpBuffers"));
        }
        // A replica→shard (or shard→replica) copy would need a scatter/gather, not a per-rank
        // memcpy, and the two sides' per-rank lengths differ — reject instead of copying the wrong
        // number of bytes onto each rank.
        if s.kind != d.kind {
            return Err(be(format!(
                "tp: copy_buffer between TpBuffers of different kinds ({:?} → {:?}) — a copy \
                 across shard layouts is not a per-rank copy",
                s.kind, d.kind
            )));
        }
        let per_rank = match s.kind {
            // Every rank holds the whole tensor: the logical length IS each rank's length.
            TpKind::Replica => bytes,
            // Each rank holds its heads' 1/W slice, so each rank copies 1/W of the logical length.
            TpKind::Kv => self.shard_bytes(bytes)?,
            // No caller copies a weight today, and the divisor is ROLE-dependent (a column-parallel
            // weight splits its output rows, a row-parallel one its input columns — a prefix copy
            // of `bytes` logical bytes is a different per-rank range in each case, and for a
            // row-parallel weight is not even contiguous per rank). Refusing beats guessing.
            TpKind::Weight(role) => {
                return Err(be(format!(
                    "tp: copy_buffer of a {role:?}-parallel weight shard is not supported — the \
                     per-rank byte range of a logical prefix depends on the weight's device role, \
                     so there is no single correct divisor; weights are filled by the binder"
                )))
            }
        };
        // Same-layout copy on every rank (KV fork/seed within a rank, replica→replica).
        for (i, b) in self.ranks.iter().enumerate() {
            b.copy_buffer(s.on(i)?, d.on(i)?, per_rank)?;
        }
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(Box::new(TpPlan {
            graph: graph.clone(),
            prepared: Mutex::new(None),
        }))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let tp = plan
            .as_any()
            .downcast_ref::<TpPlan>()
            .ok_or_else(|| be("tp: execute got a non-tp plan"))?;
        let mut guard = tp.prepared.lock().expect("tp plan poisoned");
        if guard.is_none() {
            *guard = Some(self.prepare(&tp.graph, bindings)?);
        }
        let prep = guard.as_ref().expect("prepared just set");
        let w = self.world();

        for seg in &prep.segments {
            // Run this segment on every rank (independent devices).
            for r in 0..w {
                let mut sub = Bindings::new();
                for &t in &seg.needed {
                    if let Some(bufs) = prep.boundary.get(&t) {
                        sub.bind(t, bufs[r].as_ref());
                    } else if let Some(buf) = bindings.get(t) {
                        sub.bind(t, as_tp(buf)?.on(r)?);
                    }
                    // (unbound = a resized Internal scratch the rank allocates itself)
                }
                self.ranks[r].execute(seg.plans[r].as_ref(), &sub)?;
            }
            // All-reduce the boundary partial across ranks (sum), so the next segment's residual Add
            // reads the full sum identically on every rank.
            if let Some(tid) = seg.reduce {
                let bufs = prep
                    .boundary
                    .get(&tid)
                    .ok_or_else(|| be("tp: reduce tensor has no boundary buffer"))?;
                let elems = prep.lowered.desc(tid).numel();
                prep.allreduce.reduce(&self.ranks, bufs, elems)?;
            }
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        for b in &self.ranks {
            b.sync()?;
        }
        Ok(())
    }

    fn kv_overflow_report(&self) {
        infr_core::backend::kv_overflow_report_each(&self.ranks);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device-free stand-in for a per-rank inner buffer. TP needs >= 2 physical GPUs, so the real
    /// path can never run in CI; the byte accounting these tests pin is pure, so it is exercised
    /// over stubs instead.
    struct StubBuf(usize);

    impl Buffer for StubBuf {
        fn len_bytes(&self) -> usize {
            self.0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn device_addr(&self) -> Option<u64> {
            None
        }
    }

    fn bufs(lens: &[usize]) -> Vec<Box<dyn Buffer>> {
        lens.iter()
            .map(|&n| Box::new(StubBuf(n)) as Box<dyn Buffer>)
            .collect()
    }

    #[test]
    fn len_bytes_is_the_logical_size_replica_is_one_buf_shards_sum() {
        // A replica: every rank holds the WHOLE tensor, so the logical size is one buffer's size —
        // NOT 4 * 1024. Summing a replica would report W× the tensor.
        let rep = TpBuffer {
            kind: TpKind::Replica,
            bufs: bufs(&[1024, 1024, 1024, 1024]),
        };
        assert_eq!(rep.len_bytes(), 1024);

        // A KV shard: each rank holds its heads' slice, so the logical size is the SUM. Reporting
        // bufs[0] here is the bug — the caller then re-shards an already-sharded size.
        let kv = TpBuffer {
            kind: TpKind::Kv,
            bufs: bufs(&[256, 256, 256, 256]),
        };
        assert_eq!(kv.len_bytes(), 1024);

        // Same for a sliced weight, whichever device role it carries.
        for role in [TpRole::Column, TpRole::Row] {
            let wt = TpBuffer {
                kind: TpKind::Weight(role),
                bufs: bufs(&[300, 300]),
            };
            assert_eq!(wt.len_bytes(), 600);
        }
    }

    #[test]
    fn sharded_len_bytes_sums_rather_than_multiplying_rank_zero() {
        // Sum, not `bufs[0] * W`: a ragged split must report what the ranks actually hold.
        let kv = TpBuffer {
            kind: TpKind::Kv,
            bufs: bufs(&[100, 200, 300]),
        };
        assert_eq!(kv.len_bytes(), 600);
    }

    #[test]
    fn shard_bytes_divides_and_rejects_an_unaligned_quantized_kv_size() {
        assert_eq!(shard_kv_bytes(4096, 2).expect("divisible"), 2048);
        assert_eq!(shard_kv_bytes(4096, 4).expect("divisible"), 1024);
        assert_eq!(shard_kv_bytes(0, 4).expect("zero is divisible"), 0);
        assert_eq!(shard_kv_bytes(777, 1).expect("world 1 divides all"), 777);

        // Not divisible ⇒ error, never a truncating divide. The message must keep naming the case
        // that reaches it, since the fix is a KV-format change and not anything about the model.
        let e = shard_kv_bytes(4098, 4).expect_err("4098 is not a multiple of 4");
        let msg = e.to_string();
        assert!(msg.contains("4098"), "{msg}");
        assert!(msg.contains("quantized KV padding"), "{msg}");
    }

    #[test]
    fn kv_alloc_copy_round_trip_reshards_exactly_once() {
        // THE invariant that makes the bug impossible to reintroduce, and the one
        // `SeamKv::mtp_snapshot_delta` got wrong: it reads `len_bytes()` off a KV buffer and hands
        // it straight back to `alloc(KvCache)` / `copy_buffer`, both of which shard. Feeding a
        // sharded buffer's logical size back through the shard must land on the per-rank size
        // again — not on `full/W²`.
        for w in [1usize, 2, 4, 8] {
            let per = 512usize;
            let kv = TpBuffer {
                kind: TpKind::Kv,
                bufs: bufs(&vec![per; w]),
            };
            assert_eq!(kv.len_bytes(), per * w);
            assert_eq!(shard_kv_bytes(kv.len_bytes(), w).expect("divisible"), per);
        }
    }
}
