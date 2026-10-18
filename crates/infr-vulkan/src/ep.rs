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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EpKind {
    /// Identical bytes on every rank.
    Replica,
    /// Each rank owns a contiguous band of experts from a stacked bank.
    Expert,
    /// Each rank owns a contiguous input-column band for a row-parallel Linear.
    Row,
}

pub struct EpBuffer {
    bufs: Vec<Box<dyn Buffer>>,
    kind: EpKind,
}

impl EpBuffer {
    /// Wrap one buffer per rank (a replica set, or an expert-bank shard set — the wrapper is the
    /// same; only the bytes the binder uploaded differ).
    pub fn wrap(bufs: Vec<Box<dyn Buffer>>) -> Box<dyn Buffer> {
        Box::new(EpBuffer {
            bufs,
            kind: EpKind::Replica,
        })
    }

    /// Wrap an expert-bank band set: each rank's buffer holds only its expert band.
    pub fn wrap_experts(bufs: Vec<Box<dyn Buffer>>) -> Box<dyn Buffer> {
        Box::new(EpBuffer {
            bufs,
            kind: EpKind::Expert,
        })
    }

    /// Wrap a row-parallel weight shard set: each rank owns an input-column band.
    pub fn wrap_rows(bufs: Vec<Box<dyn Buffer>>) -> Box<dyn Buffer> {
        Box::new(EpBuffer {
            bufs,
            kind: EpKind::Row,
        })
    }

    fn on(&self, r: usize) -> Result<&dyn Buffer> {
        self.bufs
            .get(r)
            .map(|b| b.as_ref())
            .ok_or_else(|| be(format!("ep: no buffer for rank {r}")))
    }

    fn kind(&self) -> EpKind {
        self.kind
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

fn ep_kind(buf: &dyn Buffer) -> Option<EpKind> {
    buf.as_any()
        .downcast_ref::<EpBuffer>()
        .map(EpBuffer::kind)
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
    /// Reduce boundaries promoted from Internal to bound Input: `tid -> per-rank persistent buffer`.
    /// A boundary is written by a segment's final sharded op and read by the next segment, so it
    /// must survive across the cut and is the buffer the all-reduce sums in place.
    boundary: HashMap<TensorId, Vec<Box<dyn Buffer>>>,
    /// The cross-device all-reduce transport. Boundaries are padded to the largest boundary width;
    /// every `reduce` call uses this same element count, so one compiled transport serves every cut.
    allreduce: AllReduce,
    /// Elements in the padded all-reduce transport.
    reduce_elems: usize,
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
    /// Called once per plan (the first execute). Expert banks are sharded by expert; weights bound
    /// as `EpKind::Row` are sharded by input columns and their full-width partial output is reduced
    /// right after the op. Everything else runs replicated.
    fn prepare(&self, full: &Graph, bindings: &Bindings) -> Result<Prepared> {
        let w = self.world();
        let n_ops = full.ops.len();

        let mut lowered = full.clone();
        let mut bank_tids: HashSet<TensorId> = HashSet::new();
        let mut row_tids: HashSet<TensorId> = HashSet::new();
        let mut cuts: Vec<usize> = Vec::new();
        let mut cut_dsts: Vec<TensorId> = Vec::new();
        let mut boundary: HashSet<TensorId> = HashSet::new();
        let mut n_moe = 0usize;
        for (i, op) in full.ops.iter().enumerate() {
            match op {
                Op::MoeFfn {
                    gate_exps,
                    up_exps,
                    down_exps,
                    n_expert,
                    dst,
                    ..
                } => {
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
                    cuts.push(i);
                    cut_dsts.push(*dst);
                    boundary.insert(*dst);
                }
                Op::Linear {
                    weight,
                    dst,
                    w_off,
                    in_f,
                    ..
                } if bindings
                    .get(*weight)
                    .is_some_and(|buf| ep_kind(buf) == Some(EpKind::Row)) =>
                {
                    if *w_off != 0 {
                        return Err(be(format!(
                            "ep: row-parallel Linear {i} has w_off={w_off}; fused weight groups \
                             must be disabled for row-parallel dense execution"
                        )));
                    }
                    if !(*in_f as usize).is_multiple_of(w) {
                        return Err(be(format!(
                            "ep: row-parallel Linear {i} in_f={in_f} not divisible by world {w}"
                        )));
                    }
                    row_tids.insert(*weight);
                    if let Op::Linear { in_f, .. } = &mut lowered.ops[i] {
                        *in_f = (*in_f) / w as u32;
                    }
                    cuts.push(i);
                    cut_dsts.push(*dst);
                    boundary.insert(*dst);
                }
                _ => {}
            }
        }
        if n_moe == 0 {
            return Err(be(
                "ep: the graph has no Op::MoeFfn — expert parallelism needs a routed-expert (MoE) \
                 model",
            ));
        }
        if cuts.is_empty() {
            return Err(be("ep: the graph has no sharded op"));
        }

        // ── shrink the sharded Weight decls to the per-rank buffer size ───────────────────────────
        // Expert banks hold E/W experts; row-parallel weights hold in_f/W input columns. Router,
        // norms, embeddings and the LM head stay full (replicated).
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
        for &tid in &row_tids {
            let decl = &mut lowered.tensors[tid.0 as usize];
            let n = decl.desc.numel();
            if !n.is_multiple_of(w) {
                return Err(be(format!(
                    "ep: row-parallel weight {} numel={n} not divisible by world {w}",
                    tid.0
                )));
            }
            decl.desc = TensorDesc::new(vec![n / w], decl.desc.dtype);
        }

        // ── segment cuts: after every sharded op (its dst partial is all-reduced) ────────────────
        let mut bounds: Vec<usize> = cuts.iter().map(|&c| c + 1).collect();
        if bounds.last() != Some(&n_ops) {
            bounds.push(n_ops);
        }

        // ── promote the reduce boundaries (Internal → Input) so they persist across the cut ───────
        // A boundary is written by a segment's final sharded op and read by the next segment, and is
        // the buffer the all-reduce sums in place, so it must be a bound persistent buffer.
        for &tid in &boundary {
            lowered.tensors[tid.0 as usize].kind = TensorKind::Input;
        }

        // ── per-rank sub-graphs (each MoeFfn's ep_band set to the rank's band) + compile ──────────
        let nl = self.n_expert / w; // experts per band
        let mut segments = Vec::with_capacity(bounds.len());
        let mut sizes: Vec<usize> = Vec::new();
        let reduce_dtype = infr_core::tensor::DType::F32;
        let mut lo = 0usize;
        for (si, &hi) in bounds.iter().enumerate() {
            let reduce = (si + 1 < bounds.len()).then(|| cut_dsts[si]);
            if let Some(tid) = reduce {
                let desc = lowered.desc(tid);
                if desc.dtype != reduce_dtype {
                    return Err(be(format!(
                        "ep: reduce boundary {tid:?} has dtype {:?}, expected F32",
                        desc.dtype
                    )));
                }
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

        // ── per-rank persistent boundary buffers (one set per distinct reduce tid) ────────────────
        // The transport has one fixed padded width. Boundary buffers are allocated at that width;
        // the logical tensor occupies a prefix and the zeroed padding participates harmlessly.
        let reduce_bytes = sizes.iter().copied().max().unwrap_or(0);
        let reduce_bytes = reduce_bytes
            .checked_add(3)
            .map(|bytes| bytes / 4 * 4)
            .ok_or_else(|| be("ep: reduce boundary size overflow"))?;
        let mut boundary_bufs: HashMap<TensorId, Vec<Box<dyn Buffer>>> = HashMap::new();
        for &tid in &boundary {
            let mut bufs = Vec::with_capacity(w);
            for r in 0..w {
                bufs.push(
                    self.ranks[r].alloc(reduce_bytes, BufferUsage::Activations)?,
                );
            }
            boundary_bufs.insert(tid, bufs);
        }

        let allreduce = AllReduce::new(&self.ranks, reduce_bytes, reduce_dtype, self.use_p2p)?;
        let reduce_elems = reduce_bytes / crate::tp_allreduce::dtype_bytes(reduce_dtype);
        Ok(Prepared {
            lowered,
            segments,
            boundary: boundary_bufs,
            allreduce,
            reduce_elems,
        })
    }
}

impl Backend for ExpertParallelBackend {
    fn name(&self) -> &str {
        "vulkan-expert-parallel"
    }

    fn backend_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn capabilities(&self) -> Capabilities {
        // Rank 0's caps with decode replay/sample off (the EP segment executor does not model those
        // one-submit forms). Embedding gather and row argmax ARE modeled: every segment runs the op
        // on each rank, and both outputs are replicated, so Qwen3.8 MTP VERIFY can use the same
        // runner path as the single-device backend. Combined gate/up (and qkv) fusions are disabled
        // because the row-parallel binder needs the separate projection tensors to slice their
        // input-column bands.
        let mut c = self.ranks[0].capabilities();
        c.decode_replay = false;
        c.embed_gather = true;
        c.gpu_sample = false;
        c.argmax_rows = true;
        c.argmax_prob = false;
        c.combined_gu = false;
        // The segment executor hands `hidden` between ranks through its own buffers; the caller's
        // is only read at the start of a step.
        c.graph_input_inplace = false;
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
            *guard = Some(self.prepare(&ep.graph, bindings)?);
        }
        let prep = guard.as_ref().expect("prepared just set");
        let w = self.world();

        for seg in &prep.segments {
            // Run this segment on every rank CONCURRENTLY. The old serial loop made the two GPUs
            // take turns for every segment; all-reduce then paid the full round trip on top.
            // Each rank owns its own device, queue and backend, so the submissions are independent;
            // the all-reduce after the join provides the cross-rank ordering.
            std::thread::scope(|scope| -> Result<()> {
                let mut workers = Vec::with_capacity(w);
                for r in 0..w {
                    let rank = &self.ranks[r];
                    let plan = &seg.plans[r];
                    let needed = &seg.needed;
                    let boundary = &prep.boundary;
                    workers.push(scope.spawn(move || -> Result<()> {
                        let mut sub = Bindings::new();
                        for &t in needed {
                            if let Some(bufs) = boundary.get(&t) {
                                // A promoted MoE `dst` boundary: this backend owns its per-rank
                                // buffer.
                                sub.bind(t, bufs[r].as_ref());
                            } else if let Some(buf) = bindings.get(t) {
                                sub.bind(t, as_ep(buf)?.on(r)?);
                            }
                            // (unbound = an Internal scratch the rank allocates itself)
                        }
                        rank.execute(plan.as_ref(), &sub)
                    }));
                }
                let mut first_err = None;
                for worker in workers {
                    match worker.join() {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => {
                            first_err.get_or_insert(error);
                        }
                        Err(_) => {
                            return Err(be("ep: rank worker panicked"));
                        }
                    }
                }
                match first_err {
                    Some(error) => Err(error),
                    None => Ok(()),
                }
            })?;
            // All-reduce the MoE `dst` partial across ranks (sum) → the full weighted top-k output,
            // identical on every rank, so the following residual Add reads the same value everywhere.
            if let Some(tid) = seg.reduce {
                let bufs = prep
                    .boundary
                    .get(&tid)
                    .ok_or_else(|| be("ep: reduce tensor has no boundary buffer"))?;
                prep.allreduce
                    .reduce(&self.ranks, bufs, prep.reduce_elems)?;
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
