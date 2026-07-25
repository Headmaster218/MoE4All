//! Expert parallelism (MoE) — distribute a routed-expert model's EXPERTS across several physical
//! Vulkan devices so each device stores + computes only its own contiguous band of experts (a
//! capacity split + parallel expert compute), the per-token routing runs replicated on every rank,
//! and one all-reduce per MoE layer combines each rank's partial expert output into the full
//! weighted top-k sum. A DIFFERENT decomposition from dense tensor parallelism ([`crate::tp`],
//! which shards each dense weight matrix) and layer-split ([`crate::PipelineBackend`], which places
//! whole layers on whole devices): EP shards the STACKED EXPERT BANKS by expert and reduces once
//! per `Op::MoeFfn`.
//!
//! # Sharding scheme (comms-minimal — ONE all-reduce per MoE layer, ZERO dispatch traffic)
//!
//! * **Experts** — the stacked banks `ffn_{gate,up,down}_exps` are split by EXPERT: device `r` of a
//!   `W`-way world owns the contiguous band `[r·E/W, (r+1)·E/W)` (needs `W | E`). The binder uploads
//!   only that band to rank `r` (capacity split — each device holds `E/W` experts' banks).
//! * **Router + everything else** (embeddings, attention, norms, the router GEMV + top-k, the
//!   residual stream, the LM head, KV) is REPLICATED and bit-identical on every rank. Because the
//!   router weight is replicated, every rank computes the IDENTICAL global top-k selection + the
//!   normalized routing weights — so all ranks agree, deterministically, on which expert every token
//!   wants and with what weight.
//! * **Per-band expert compute** — the [`Op::MoeFfn`] on rank `r` carries `ep_band = Some((r·E/W,
//!   E/W))`: it still routes over the FULL `E` (replicated router/top-k) but a `moe_ep_band_remap`
//!   right after top-k rewrites the selected global ids into rank `r`'s LOCAL shard indices
//!   (out-of-band → id 0, weight 0), so the expert GEMV/GEMM stages read only rank `r`'s bank and
//!   the weighted combine drops the assignments rank `r` doesn't own. Rank `r`'s MoE output `dst` is
//!   thus a PARTIAL `[tokens, n_embd]` — the weighted sum over just its band's experts.
//! * **Combine** — the partials are summed across ranks by an **all-reduce** ([`crate::tp_allreduce`],
//!   reused verbatim from TP): host-less P2P dma-buf data path with `VK_KHR_external_semaphore_fd`
//!   GPU-side ordering when the device pair supports it, else host-fence, else host-bounce. For each
//!   (token, slot) exactly one rank has a non-zero weight (the band owning that expert) and every
//!   other rank contributes a hard 0 (exact in f32), so `Σ_r dst_r` reproduces the single-device
//!   weighted top-k sum — differing only by reassociation of the cross-rank add (token-identical,
//!   exactly like TP). ONE all-reduce per MoE layer; the residual hidden state is replicated on
//!   every rank (attention/norms run replicated), so there is NO separate dispatch of activations —
//!   only the combine crosses the bus.
//!
//! # How it stays invisible above the [`Backend`] seam
//!
//! Exactly like [`crate::tp`] / [`crate::PipelineBackend`], the model runner (`generate_dense_backend`)
//! drives this as ONE backend over the ordinary single-device graph. At [`ExpertParallelBackend::execute`]
//! the graph is LOWERED once (cached in the plan) into per-rank graphs (expert-bank decls shrunk to
//! the band, each `Op::MoeFfn`'s `ep_band` set to the rank's band) and partitioned into SEGMENTS cut
//! after each `Op::MoeFfn`. Each segment runs on every rank; after a segment the MoE `dst` partial is
//! all-reduced so the following residual `Add` reads the full sum identically on every rank.
//!
//! # Scope (v1, correctness-first)
//!
//! One MoE arch (qwen3moe: split gate/up, softmax gating, `norm_w`, no shared expert), `W`-general as
//! long as `W | n_expert`, experts RESIDENT (no pager) on each rank. Deferred: shared-expert
//! placement (qwen35moe / llama4 `MoeSharedExpertAdd`), the paged expert path, DeltaNet-MoE /
//! gemma4-MoE / Scout, and compute/comms overlap. The GPU-resident decode-replay / embed-gather /
//! gpu-sample fast paths are turned OFF via [`capabilities`] so the runner takes the static
//! host-embed + host-sample per-token `execute` path (keeps the residual `hidden` a bound Input the
//! segment executor hands through) — the same lever pipeline/TP use.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan};
use infr_core::error::Result;
use infr_core::graph::{Graph, Op, TensorKind};
use infr_core::tensor::{TensorDesc, TensorId};

use crate::tp_allreduce::AllReduce;
use crate::{be, VulkanBackend};

/// A buffer owned by an [`ExpertParallelBackend`]: `bufs[r]` is rank `r`'s buffer. A replica holds
/// identical bytes on every rank (dense weights, router, norms, embeddings, KV, the residual/IO,
/// logits); an expert-bank shard holds each rank's own expert band (different bytes, same per-expert
/// layout). The executor always unwraps to the per-rank inner `VkBuffer` before it reaches a
/// sub-backend.
pub struct EpBuffer {
    bufs: Vec<Box<dyn Buffer>>,
}

impl EpBuffer {
    /// Wrap one buffer per rank (a replica set, or an expert-bank shard set — the wrapper is the
    /// same; only the bytes the binder uploaded differ).
    pub fn wrap(bufs: Vec<Box<dyn Buffer>>) -> Box<dyn Buffer> {
        Box::new(EpBuffer { bufs })
    }

    fn on(&self, r: usize) -> Result<&dyn Buffer> {
        self.bufs
            .get(r)
            .map(|b| b.as_ref())
            .ok_or_else(|| be(format!("ep: no buffer for rank {r}")))
    }
}

impl Buffer for EpBuffer {
    fn len_bytes(&self) -> usize {
        self.bufs[0].len_bytes()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device_addr(&self) -> Option<u64> {
        // An EpBuffer is never itself addressed as one contiguous thing — the executor resolves to a
        // per-rank inner buffer (which does carry its own BDA for the arena reads).
        None
    }
}

fn as_ep(buf: &dyn Buffer) -> Result<&EpBuffer> {
    buf.as_any()
        .downcast_ref::<EpBuffer>()
        .ok_or_else(|| be("ep: a buffer bound to an ExpertParallelBackend was not an EpBuffer"))
}

/// One contiguous run of ops that runs on every rank, plus (except for the last) the MoE `dst`
/// boundary to all-reduce after it.
struct Segment {
    /// One compiled plan per rank (identical structure; the `Op::MoeFfn` in it carries the rank's
    /// `ep_band`, and the bound expert banks are the rank's shard, so the compiled bytes differ).
    plans: Vec<Box<dyn Plan>>,
    /// Tensors this segment's ops read/write that must be bound.
    needed: Vec<TensorId>,
    /// The MoE `dst` partial produced at the end of this segment (all-reduced), or `None` for the last.
    reduce: Option<TensorId>,
}

/// The lowered + partitioned form of a graph, built once per plan on first execute.
struct Prepared {
    /// The lowered graph (expert-bank decls shrunk to the per-rank band). Structurally the boundary
    /// tensors are still Internal — the MoE `dst` is written+read within a step, so unlike TP's
    /// cross-segment residual it needs no promotion; the executor binds the same per-rank buffer the
    /// runner allocated. Kept for `desc()` lookups.
    lowered: Graph,
    segments: Vec<Segment>,
    /// The MoE `dst` boundary tensors, promoted from Internal to bound Input: `tid -> per-rank
    /// persistent buffer`. The MoE `dst` (`sub`) is written by a segment's final `Op::MoeFfn` and
    /// read by the NEXT segment's residual `Add`, so it must be a persistent bound buffer that
    /// survives across the cut (and is the buffer the all-reduce sums in place) — exactly like TP's
    /// row-parallel boundary. Every layer's MoE `dst` is the same reused scratch tid, so one per-rank
    /// buffer set serves them all.
    boundary: HashMap<TensorId, Vec<Box<dyn Buffer>>>,
    /// The cross-device all-reduce transport, over the (single, reused) MoE `dst` size.
    allreduce: AllReduce,
}

/// An [`ExpertParallelBackend`] plan: the full graph + its lazily-built lowering.
pub struct EpPlan {
    graph: Graph,
    prepared: Mutex<Option<Prepared>>,
}

impl Plan for EpPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Multi-device expert-parallel (MoE) backend. Owns `W` real Vulkan devices (ranks) and presents one
/// [`Backend`] to the runner; each MoE layer's experts are split across the ranks with a P2P
/// all-reduce per `Op::MoeFfn`.
pub struct ExpertParallelBackend {
    ranks: Vec<VulkanBackend>,
    n_expert: usize,
    /// Try the host-less external-memory (dma-buf) all-reduce transport; falls back per pair.
    use_p2p: bool,
}

impl ExpertParallelBackend {
    /// Build a `W`-way expert-parallel backend over `ranks`. `n_expert` is the model's global expert
    /// count and MUST be divisible by `W = ranks.len()` for a clean per-expert band split.
    pub fn new(ranks: Vec<VulkanBackend>, n_expert: usize, use_p2p: bool) -> Result<Self> {
        if ranks.is_empty() {
            return Err(be("ep: needs at least one device"));
        }
        let w = ranks.len();
        if !n_expert.is_multiple_of(w) {
            return Err(be(format!(
                "ep: n_expert={n_expert} is not divisible by the {w}-way expert-parallel world — \
                 EP needs n_expert divisible by the device count for a contiguous per-device band"
            )));
        }
        Ok(Self {
            ranks,
            n_expert,
            use_p2p,
        })
    }

    pub fn world(&self) -> usize {
        self.ranks.len()
    }

    /// Experts per device (the band width).
    pub fn experts_per_device(&self) -> usize {
        self.n_expert / self.ranks.len()
    }

    /// The real backend for rank `r` — the binder allocs/uploads each per-rank expert shard through it.
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

    /// Replicate: run `f` on every rank, collecting one buffer per rank into an [`EpBuffer`].
    fn replicate<F>(&self, f: F) -> Result<Box<dyn Buffer>>
    where
        F: Fn(&VulkanBackend) -> Result<Box<dyn Buffer>>,
    {
        let mut bufs = Vec::with_capacity(self.ranks.len());
        for b in &self.ranks {
            bufs.push(f(b)?);
        }
        Ok(EpBuffer::wrap(bufs))
    }

    /// Lower `full` (the runner's global-dim graph) into per-rank segments + the all-reduce transport.
    /// Called once per plan (the first execute). Only [`Op::MoeFfn`] ops are sharded (their `dst`
    /// becomes a per-rank partial reduced right after); everything else runs replicated + identical.
    fn prepare(&self, full: &Graph) -> Result<Prepared> {
        let w = self.world();
        let n_ops = full.ops.len();

        // ── shrink the expert-bank tensor decls to the per-rank band ──────────────────────────────
        // The bound expert-bank buffer on each rank holds only E/W experts (the binder uploaded just
        // the band), so the Weight decl must match or the sub-backend's size math disagrees. Router
        // and every other weight stay full (replicated). `up_exps == gate_exps` when fused, so a set
        // dedupes the shrink.
        let mut lowered = full.clone();
        let mut bank_tids: HashSet<TensorId> = HashSet::new();
        let mut n_moe = 0usize;
        for op in &full.ops {
            if let Op::MoeFfn {
                gate_exps,
                up_exps,
                down_exps,
                n_expert,
                ..
            } = op
            {
                n_moe += 1;
                if (*n_expert as usize) != self.n_expert {
                    return Err(be(format!(
                        "ep: Op::MoeFfn n_expert={n_expert} disagrees with the backend's \
                         n_expert={} (all MoE layers must share the same expert count)",
                        self.n_expert
                    )));
                }
                if !(*n_expert as usize).is_multiple_of(w) {
                    return Err(be(format!(
                        "ep: Op::MoeFfn n_expert={n_expert} not divisible by world {w}"
                    )));
                }
                bank_tids.insert(*gate_exps);
                bank_tids.insert(*up_exps);
                bank_tids.insert(*down_exps);
            }
        }
        if n_moe == 0 {
            return Err(be(
                "ep: the graph has no Op::MoeFfn — expert parallelism needs a routed-expert (MoE) \
                 model",
            ));
        }
        for &tid in &bank_tids {
            let decl = &mut lowered.tensors[tid.0 as usize];
            let n = decl.desc.numel();
            if !n.is_multiple_of(w) {
                return Err(be(format!(
                    "ep: expert bank {} numel={n} not divisible by world {w}",
                    tid.0
                )));
            }
            decl.desc = TensorDesc::new(vec![n / w], decl.desc.dtype);
        }

        // ── segment cuts: after each Op::MoeFfn (its dst partial is all-reduced) ───────────────────
        let cuts: Vec<usize> = (0..n_ops)
            .filter(|&i| matches!(full.ops[i], Op::MoeFfn { .. }))
            .collect();
        let mut bounds: Vec<usize> = cuts.iter().map(|&c| c + 1).collect();
        if bounds.last() != Some(&n_ops) {
            bounds.push(n_ops);
        }

        // ── promote the MoE `dst` boundaries (Internal → Input) so they persist across the cut ─────
        // Each MoeFfn `dst` is written at a segment's end and read by the next segment's residual
        // Add, and is the buffer the all-reduce sums, so it must be a bound persistent buffer (the
        // runner leaves it Internal — a per-segment scratch would neither carry the value nor be
        // addressable by the reduce). Same reused scratch tid across all layers → one buffer set.
        let mut boundary: HashSet<TensorId> = HashSet::new();
        for &c in &cuts {
            if let Op::MoeFfn { dst, .. } = &lowered.ops[c] {
                boundary.insert(*dst);
            }
        }
        for &tid in &boundary {
            lowered.tensors[tid.0 as usize].kind = TensorKind::Input;
        }

        // ── per-rank sub-graphs (each MoeFfn's ep_band set to the rank's band) + compile ──────────
        let nl = self.n_expert / w; // experts per band
        let mut segments = Vec::with_capacity(bounds.len());
        let mut sizes: Vec<usize> = Vec::new();
        let mut reduce_dtype = infr_core::tensor::DType::F32;
        let mut lo = 0usize;
        for (si, &hi) in bounds.iter().enumerate() {
            // The reduce boundary for this segment is the cut op's (MoeFfn) dst — None for the last.
            let reduce = if si + 1 < bounds.len() {
                match &lowered.ops[hi - 1] {
                    Op::MoeFfn { dst, .. } => Some(*dst),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(tid) = reduce {
                let desc = lowered.desc(tid);
                reduce_dtype = desc.dtype;
                sizes.push(desc.numel() * crate::tp_allreduce::dtype_bytes(desc.dtype));
            }
            // Ops this segment needs bound.
            let mut needed: HashSet<TensorId> = HashSet::new();
            for op in &lowered.ops[lo..hi] {
                let (rd, wr) = op.io();
                needed.extend(rd);
                needed.extend(wr);
            }
            // One compiled plan per rank: clone the segment, set the rank's ep_band on its MoeFfn.
            let mut plans = Vec::with_capacity(w);
            for r in 0..w {
                let mut sub = lowered.clone();
                sub.ops = lowered.ops[lo..hi].to_vec();
                for op in &mut sub.ops {
                    if let Op::MoeFfn { ep_band, .. } = op {
                        *ep_band = Some(((r * nl) as u32, nl as u32));
                    }
                }
                let written: HashSet<TensorId> = sub.ops.iter().flat_map(|op| op.io().1).collect();
                sub.outputs = lowered
                    .outputs
                    .iter()
                    .copied()
                    .filter(|t| written.contains(t))
                    .collect();
                sub.no_decode_replay = true;
                plans.push(self.ranks[r].compile(&sub)?);
            }
            segments.push(Segment {
                plans,
                needed: needed.into_iter().collect(),
                reduce,
            });
            lo = hi;
        }

        // ── per-rank persistent boundary buffers (one set per distinct MoE `dst` tid) ──────────────
        // All MoE `dst` boundaries share the reused scratch tid + size; the single reduce transport
        // requires them equal (asserted), not silently `max`-ed.
        let mut boundary_bufs: HashMap<TensorId, Vec<Box<dyn Buffer>>> = HashMap::new();
        for &tid in &boundary {
            let bytes = lowered.desc(tid).numel()
                * crate::tp_allreduce::dtype_bytes(lowered.desc(tid).dtype);
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

impl Backend for ExpertParallelBackend {
    fn name(&self) -> &str {
        "vulkan-expert-parallel"
    }

    fn capabilities(&self) -> Capabilities {
        // Rank 0's caps with the GPU-resident fast paths the EP segment executor does not model
        // turned OFF: the runner then takes the classic host-embed + host-sample static per-token
        // `execute` path (keeps the residual `hidden` a bound Input the executor hands through). The
        // attention qkv/dense fuses are LEFT ON — attention is replicated (not sharded) under EP, so
        // fused projections run identically on every rank.
        let mut c = self.ranks[0].capabilities();
        c.decode_replay = false;
        c.embed_gather = false;
        c.gpu_sample = false;
        c.argmax_rows = false;
        c.argmax_prob = false;
        c
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        // Every allocation is REPLICATED to every rank: KV (each rank keeps the full KV — attention
        // is replicated, not head-sharded like TP), the Staging hidden/positions, Activations
        // (incl. the MoE `dst` partial buffer, same full [tokens, n_embd] size on every rank),
        // Readback logits, and the weightless ones/rope buffers. Expert banks bypass alloc via the
        // binder.
        self.replicate(|b| b.alloc(bytes, usage))
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        self.replicate(|b| b.alloc_uninit(bytes, usage))
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let ep = as_ep(dst)?;
        // Replica / IO: the same bytes to every rank. (Expert shards are uploaded by the binder
        // straight to each rank's backend, never through this whole-EpBuffer path.)
        for (b, buf) in self.ranks.iter().zip(&ep.bufs) {
            b.upload(buf.as_ref(), src)?;
        }
        Ok(())
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        // Replicated buffers (logits) are identical on every rank — read rank 0.
        self.ranks[0].download(as_ep(src)?.on(0)?, dst)
    }

    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        let (s, d) = (as_ep(src)?, as_ep(dst)?);
        if s.bufs.len() != d.bufs.len() {
            return Err(be("ep: copy_buffer between mismatched EpBuffers"));
        }
        // Same-layout copy on every rank (KV fork/seed within a rank, replica→replica).
        for (i, b) in self.ranks.iter().enumerate() {
            b.copy_buffer(s.on(i)?, d.on(i)?, bytes)?;
        }
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(Box::new(EpPlan {
            graph: graph.clone(),
            prepared: Mutex::new(None),
        }))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let ep = plan
            .as_any()
            .downcast_ref::<EpPlan>()
            .ok_or_else(|| be("ep: execute got a non-ep plan"))?;
        let mut guard = ep.prepared.lock().expect("ep plan poisoned");
        if guard.is_none() {
            *guard = Some(self.prepare(&ep.graph)?);
        }
        let prep = guard.as_ref().expect("prepared just set");
        let w = self.world();

        for seg in &prep.segments {
            // Run this segment on every rank (independent devices).
            for r in 0..w {
                let mut sub = Bindings::new();
                for &t in &seg.needed {
                    if let Some(bufs) = prep.boundary.get(&t) {
                        // A promoted MoE `dst` boundary: this backend owns its per-rank buffer.
                        sub.bind(t, bufs[r].as_ref());
                    } else if let Some(buf) = bindings.get(t) {
                        sub.bind(t, as_ep(buf)?.on(r)?);
                    }
                    // (unbound = an Internal scratch the rank allocates itself)
                }
                self.ranks[r].execute(seg.plans[r].as_ref(), &sub)?;
            }
            // All-reduce the MoE `dst` partial across ranks (sum) → the full weighted top-k output,
            // identical on every rank, so the following residual Add reads the same value everywhere.
            if let Some(tid) = seg.reduce {
                let bufs = prep
                    .boundary
                    .get(&tid)
                    .ok_or_else(|| be("ep: reduce tensor has no boundary buffer"))?;
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
