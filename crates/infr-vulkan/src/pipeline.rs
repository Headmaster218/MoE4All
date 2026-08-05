//! Pipeline (layer-split) multi-GPU: run ONE model's transformer layers split across several
//! physical Vulkan devices so a model too big for one device's VRAM can run, handing the hidden
//! state across the device boundary.
//!
//! # What this is
//!
//! [`PipelineBackend`] implements [`infr_core::backend::Backend`] by owning N real
//! [`VulkanBackend`]s (one per physical device) and transparently SPLITTING each compiled graph by
//! layer: layers `[0..k)` run on device A, `[k..N)` on device B (generalized to a device list).
//! Weights and KV for a layer live on the device that runs it; the residual hidden state is handed
//! device A→B at the split (host-less P2P over dma-buf when available, host-bounce otherwise). The
//! model runner (`generate_dense_backend`) is UNCHANGED — it drives this exactly like a single
//! backend; the split is invisible above the `Backend` seam.
//!
//! # How placement works (no runner change)
//!
//! * **Weights** are placed by tensor NAME through a device-aware binder the caller builds (see
//!   `infr_llama::seam::pipeline_binder`): `blk.{l}.*` → device of layer `l`, `output*`/final norm
//!   → last device. Each returns a single-device [`PipelineBuffer`], so an op that reads it is
//!   pinned to that device.
//! * **KV** (`alloc(KvCache)`) is placed per-layer by the strictly-ordered `(k,v)`-pair alloc
//!   counter — the runner allocs `kbufs[l]` then `vbufs[l]` for `l = 0..n_layer`, so alloc pair `l`
//!   is layer `l`.
//! * **IO** — the host-embedded `hidden` residual, `positions`, the rope table, the weightless
//!   ones-vectors (`Staging`/`Weights`/`Activations`) — is REPLICATED to every device (each device
//!   reads its own copy). The residual is the one replicated buffer the ops WRITE, so it is the
//!   cut tensor handed across at each boundary; read-only replicas (positions/rope) need no handoff.
//! * **Logits** (`alloc(Readback)`) live on the last device (the LM head runs there); the runner's
//!   `download(logits)` reads that device.
//!
//! At [`PipelineBackend::execute`] each op's device is INFERRED from the device its bound
//! weight/KV operands live on (ground truth, no layer-label heuristics); pure activation ops
//! (residual `Add`, `Copy`, `Scale`) inherit the previous op's device. The ops partition into
//! contiguous per-device segments; each segment is compiled + run on its device, and the cut
//! tensor(s) are transferred between segments with a host-fence sync.
//!
//! # Scope (v1, correctness-first)
//!
//! `capabilities()` reports the fancy GPU-resident features OFF (`embed_gather`/`decode_replay`/
//! `gpu_sample`/argmax), so the runner takes the classic host-embed + host-sample + static
//! per-token `execute` path — which keeps `hidden` a bound Input (handoff-able) and every step a
//! plain `execute`. Dense attention models only; MoE-paged / dense-streaming are disabled. The
//! split output is BIT-IDENTICAL to the same model run single-device: identical ops, identical
//! per-device kernels (the same [`VulkanBackend`] code on each device), only the boundary residual
//! crosses via a value-preserving copy.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan};
use infr_core::error::Result;
use infr_core::graph::Graph;
use infr_core::tensor::TensorId;

use crate::{be, P2pHandleType, VulkanBackend};

/// A buffer owned by a [`PipelineBackend`]: either resident on ONE device (weights, KV, logits) or
/// REPLICATED across every device (the residual, positions, rope table, ones-vectors). The inner
/// buffers are real `VkBuffer`s — the executor always unwraps to the per-device inner buffer before
/// handing anything to a sub-backend (a `PipelineBuffer` must never reach `VulkanBackend`'s own
/// methods, which downcast to `VkBuffer`).
pub struct PipelineBuffer {
    /// `Single(dev)` ⇒ `bufs` has one entry, resident on device `dev`. `None` ⇒ `bufs[i]` is the
    /// replica on device `i` (len == number of devices).
    device: Option<usize>,
    bufs: Vec<Box<dyn Buffer>>,
}

impl PipelineBuffer {
    /// A single-device buffer (weights/KV/logits).
    pub fn single(device: usize, buf: Box<dyn Buffer>) -> Box<dyn Buffer> {
        Box::new(PipelineBuffer {
            device: Some(device),
            bufs: vec![buf],
        })
    }

    /// The inner buffer as seen on device `dev`: the sole buffer for a single-device handle (must
    /// match `dev`), else the `dev`-th replica.
    fn on(&self, dev: usize) -> Result<&dyn Buffer> {
        match self.device {
            Some(d) => {
                if d != dev {
                    return Err(be(format!(
                        "pipeline: a device-{d} buffer is needed on device {dev} \
                         (cross-device operand that is not a handoff tensor)"
                    )));
                }
                Ok(self.bufs[0].as_ref())
            }
            None => self
                .bufs
                .get(dev)
                .map(|b| b.as_ref())
                .ok_or_else(|| be(format!("pipeline: no replica for device {dev}"))),
        }
    }

    /// The single resident device, or `None` for a replicated (device-agnostic) buffer.
    fn single_device(&self) -> Option<usize> {
        self.device
    }

    fn is_replicated(&self) -> bool {
        self.device.is_none()
    }
}

impl Buffer for PipelineBuffer {
    fn len_bytes(&self) -> usize {
        self.bufs[0].len_bytes()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device_addr(&self) -> Option<u64> {
        // Only single-device buffers carry a BDA (weights/KV); a replica set has no single address.
        match self.device {
            Some(_) => self.bufs[0].device_addr(),
            None => None,
        }
    }
}

/// Downcast a `&dyn Buffer` bound by the runner to the pipeline's own wrapper.
fn as_pipe(buf: &dyn Buffer) -> Result<&PipelineBuffer> {
    buf.as_any()
        .downcast_ref::<PipelineBuffer>()
        .ok_or_else(|| be("pipeline: a buffer bound to a PipelineBackend was not a PipelineBuffer"))
}

/// For each tensor, the device index of the LAST op (in graph order) that wrote it, given each op's
/// written tensors paired with the device it runs on. The residual handoff must copy a cut tensor
/// from the device that ACTUALLY last wrote it — not blindly from the previous segment's device — so
/// a replicated op-written tensor that was produced, then skipped a segment, is handed off from its
/// true producer (else the consumer reads stale bytes on that device's replica).
fn last_writer_devices(op_writes: &[(Vec<TensorId>, usize)]) -> HashMap<TensorId, usize> {
    let mut m = HashMap::new();
    for (writes, dev) in op_writes {
        for &t in writes {
            m.insert(t, *dev);
        }
    }
    m
}

/// The residual-handoff transport between two devices. Built once (per plan boundary) and reused
/// every step: the tiny `[tokens × n_embd]` residual is copied producer→consumer here.
enum Handoff {
    /// Host-less P2P: `export` is a device-local exportable buffer on the PRODUCER; `imported`
    /// aliases it on the CONSUMER. Each step copies producer-replica → export, then imported →
    /// consumer-replica (a real PCIe read), byte-for-byte.
    P2p {
        export: crate::P2pExport,
        imported: Box<dyn Buffer>,
    },
    /// Host-bounce fallback: producer→host RAM→consumer.
    Host { scratch: Mutex<Vec<u8>> },
}

/// One contiguous run of ops assigned to a single device, plus the boundary handoff INTO it.
struct Segment {
    device: usize,
    plan: Box<dyn Plan>,
    /// Ops in this segment, by index into the full graph — used to build the per-device outputs.
    op_range: (usize, usize),
    /// Cut tensors to hand off into this segment before running: `(tensor, producer_device,
    /// handoff)`. `producer_device` is the device that LAST wrote the tensor (its true source), not
    /// necessarily the immediately-previous segment's device.
    cut: Vec<(TensorId, usize, Handoff)>,
}

/// The compiled + partitioned form of a graph, cached in the plan on first execute (bindings — and
/// thus each operand's device — are known only then).
struct Prepared {
    segments: Vec<Segment>,
}

/// A [`PipelineBackend`] plan: the graph, plus the lazily-built [`Prepared`] partition.
pub struct PipelinePlan {
    graph: Graph,
    prepared: Mutex<Option<Prepared>>,
}

impl Plan for PipelinePlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Multi-device pipeline (layer-split) backend. Owns N real Vulkan devices and a per-layer device
/// assignment; presents one [`Backend`] to the runner.
pub struct PipelineBackend {
    backends: Vec<VulkanBackend>,
    /// `layer_device[l]` = the device index that runs layer `l`. Non-decreasing (a forward
    /// pipeline crosses each device boundary once).
    layer_device: Vec<usize>,
    /// Next `(k,v)`-pair index for a `KvCache` alloc (placement by layer order).
    kv_alloc_pairs: AtomicUsize,
    /// Handoff transport preference: try P2P (dma-buf) first, else host-bounce.
    use_p2p: bool,
}

impl PipelineBackend {
    /// Build a pipeline over `backends` with a per-layer device assignment `layer_device` (one
    /// entry per model layer, values indexing `backends`, non-decreasing). `use_p2p` requests the
    /// host-less dma-buf handoff (silently falls back to host-bounce per boundary if unsupported).
    pub fn new(
        backends: Vec<VulkanBackend>,
        layer_device: Vec<usize>,
        use_p2p: bool,
    ) -> Result<Self> {
        if backends.is_empty() {
            return Err(be("pipeline: needs at least one device"));
        }
        if layer_device.iter().any(|&d| d >= backends.len()) {
            return Err(be("pipeline: layer_device index out of range"));
        }
        if layer_device.windows(2).any(|w| w[1] < w[0]) {
            return Err(be(
                "pipeline: layer_device must be non-decreasing (forward pipeline)",
            ));
        }
        Ok(Self {
            backends,
            layer_device,
            kv_alloc_pairs: AtomicUsize::new(0),
            use_p2p,
        })
    }

    /// A balanced N-way split of `n_layer` layers across `backends.len()` devices (device `i` gets
    /// a contiguous, near-equal slice). The natural default when the caller gives no explicit map.
    pub fn balanced_layer_map(n_layer: usize, n_dev: usize) -> Vec<usize> {
        let n_dev = n_dev.max(1);
        (0..n_layer)
            .map(|l| (l * n_dev / n_layer.max(1)).min(n_dev - 1))
            .collect()
    }

    /// The device that runs layer `l`.
    pub fn layer_device(&self, l: usize) -> usize {
        self.layer_device[l.min(self.layer_device.len().saturating_sub(1))]
    }

    /// The last device (holds the final norm + LM head + logits).
    pub fn last_device(&self) -> usize {
        self.backends.len() - 1
    }

    pub fn n_devices(&self) -> usize {
        self.backends.len()
    }

    /// The device a weight named `name` is placed on: `blk.{l}.*` → layer `l`'s device;
    /// `output*` / `token_embd` (the tied LM head) → the last device; anything else → device 0.
    pub fn device_for_weight(&self, name: &str) -> usize {
        if let Some(l) = name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
        {
            return self.layer_device(l);
        }
        if name.starts_with("output") || name.starts_with("token_embd") {
            return self.last_device();
        }
        0
    }

    /// The real backend for device `d` — the binder allocs/uploads a per-layer weight through it.
    pub fn backend(&self, d: usize) -> &VulkanBackend {
        &self.backends[d]
    }

    /// Per-device VRAM/name introspection: `(device_name, index)` for each device, so a caller can
    /// print the placement table and confirm each half landed on its own GPU.
    pub fn device_names(&self) -> Vec<String> {
        self.backends
            .iter()
            .map(|b| b.capabilities().name.clone())
            .collect()
    }

    /// Replicate: run `f` on every device's backend, collecting one buffer per device.
    fn replicate<F>(&self, f: F) -> Result<Box<dyn Buffer>>
    where
        F: Fn(&VulkanBackend) -> Result<Box<dyn Buffer>>,
    {
        let mut bufs = Vec::with_capacity(self.backends.len());
        for b in &self.backends {
            bufs.push(f(b)?);
        }
        Ok(Box::new(PipelineBuffer { device: None, bufs }))
    }

    /// Build the per-device partition of `graph` under `bindings` (called once per plan).
    fn prepare(&self, graph: &Graph, bindings: &Bindings) -> Result<Prepared> {
        let n_ops = graph.ops.len();
        // ── op → device inference ────────────────────────────────────────────────────────────
        // An op runs on the device its single-device (weight/KV) operand lives on; a pure
        // activation op with no such operand inherits the previous op's device.
        let mut op_dev = vec![0usize; n_ops];
        let mut cur = 0usize;
        for (i, op) in graph.ops.iter().enumerate() {
            let (reads, writes) = op.io();
            let mut pinned: Option<usize> = None;
            for t in reads.iter().chain(writes.iter()) {
                if let Some(buf) = bindings.get(*t) {
                    if let Some(d) = as_pipe(buf)?.single_device() {
                        match pinned {
                            None => pinned = Some(d),
                            Some(p) if p != d => {
                                return Err(be(format!(
                                    "pipeline: op {} ({}) reads operands on two devices ({p} and \
                                     {d}) — a weight/KV tensor is misplaced for a layer split",
                                    i,
                                    op.kind()
                                )));
                            }
                            _ => {}
                        }
                    }
                }
            }
            if let Some(d) = pinned {
                cur = d;
            }
            op_dev[i] = cur;
        }
        if op_dev.windows(2).any(|w| w[1] < w[0]) {
            return Err(be(
                "pipeline: op device assignment is not monotonic — the graph is not a clean \
                 forward layer-split (an op on a later device feeds one on an earlier device)",
            ));
        }

        // ── contiguous segments ──────────────────────────────────────────────────────────────
        let mut segments: Vec<Segment> = Vec::new();
        let mut start = 0usize;
        while start < n_ops {
            let d = op_dev[start];
            let mut end = start + 1;
            while end < n_ops && op_dev[end] == d {
                end += 1;
            }
            // Sub-graph for this segment: the SAME tensor decls (so TensorIds match the bindings),
            // this segment's ops, and only the outputs it actually writes.
            let mut sub = graph.clone();
            sub.ops = graph.ops[start..end].to_vec();
            let written: std::collections::HashSet<TensorId> =
                sub.ops.iter().flat_map(|op| op.io().1).collect();
            sub.outputs = graph
                .outputs
                .iter()
                .copied()
                .filter(|t| written.contains(t))
                .collect();
            // The static per-execute path (no record-once replay) — pipeline forces it via caps.
            sub.no_decode_replay = true;
            let plan = self.backends[d].compile(&sub)?;

            // ── cut tensors into this segment ────────────────────────────────────────────────
            // A REPLICATED bound tensor written by an earlier op and read by an op in THIS segment
            // must be handed off from the device that LAST wrote it (its true producer, tracked via
            // `last_writer_devices`) — NOT blindly from the previous segment's device, which is wrong
            // for any replicated op-written tensor produced, then skipped a segment, then read later.
            let cut = if !segments.is_empty() {
                // (tensor's last-writer device) over all ops before this segment.
                let op_writes: Vec<(Vec<TensorId>, usize)> = graph.ops[..start]
                    .iter()
                    .enumerate()
                    .map(|(i, op)| (op.io().1, op_dev[i]))
                    .collect();
                let last_writer = last_writer_devices(&op_writes);
                // tensors read in this segment
                let mut read_here: std::collections::HashSet<TensorId> =
                    std::collections::HashSet::new();
                for op in &graph.ops[start..end] {
                    for t in op.io().0 {
                        read_here.insert(t);
                    }
                }
                let mut cuts = Vec::new();
                for &t in &read_here {
                    let Some(&producer) = last_writer.get(&t) else {
                        continue; // not written before this segment (a fresh input / read-only)
                    };
                    let Some(buf) = bindings.get(t) else {
                        continue;
                    };
                    let pb = as_pipe(buf)?;
                    if !pb.is_replicated() {
                        continue; // single-device tensors don't cross (they're layer-local)
                    }
                    if producer == d {
                        continue; // already resident on this device's replica (no handoff needed)
                    }
                    let bytes = pb.len_bytes();
                    let ho = self.build_handoff(producer, d, bytes)?;
                    cuts.push((t, producer, ho));
                }
                cuts
            } else {
                Vec::new()
            };

            segments.push(Segment {
                device: d,
                plan,
                op_range: (start, end),
                cut,
            });
            start = end;
        }
        Ok(Prepared { segments })
    }

    /// Set up a residual handoff producer→consumer: P2P dma-buf when both devices support it, else
    /// host-bounce.
    fn build_handoff(&self, producer: usize, consumer: usize, bytes: usize) -> Result<Handoff> {
        if self.use_p2p
            && self.backends[producer].p2p_supported(P2pHandleType::DmaBuf)
            && self.backends[consumer].p2p_supported(P2pHandleType::DmaBuf)
        {
            // Try the host-less path; a per-pair rejection (some driver/direction combos refuse a
            // cross-device import) is not fatal — fall back to host-bounce.
            match self.backends[producer].p2p_export(bytes, P2pHandleType::DmaBuf) {
                Ok(export) => match self.backends[consumer].p2p_import(&export) {
                    Ok(imported) => return Ok(Handoff::P2p { export, imported }),
                    Err(e) => tracing::warn!(
                        "pipeline: P2P import Vulkan{producer}→Vulkan{consumer} rejected \
                         ({e}); using host-bounce handoff"
                    ),
                },
                Err(e) => tracing::warn!(
                    "pipeline: P2P export on Vulkan{producer} failed ({e}); host-bounce handoff"
                ),
            }
        }
        Ok(Handoff::Host {
            scratch: Mutex::new(vec![0u8; bytes]),
        })
    }

    /// Move the residual for cut tensor `t` from `producer` to `consumer` (the replicas live in the
    /// bound [`PipelineBuffer`]). The producer's segment has already run + synced.
    fn transfer(
        &self,
        t: TensorId,
        producer: usize,
        consumer: usize,
        ho: &Handoff,
        bindings: &Bindings,
    ) -> Result<()> {
        let pb = as_pipe(
            bindings
                .get(t)
                .ok_or_else(|| be("pipeline: cut tensor not bound at transfer"))?,
        )?;
        let src = pb.on(producer)?;
        let dst = pb.on(consumer)?;
        let bytes = pb.len_bytes();
        match ho {
            Handoff::P2p { export, imported } => {
                // producer replica → shared export (producer-local), RELEASING the export to
                // QUEUE_FAMILY_EXTERNAL; then ACQUIRE the imported alias from EXTERNAL and read it →
                // consumer replica (the PCIe read). The QF ownership transfer satisfies the EXCLUSIVE
                // external buffer's cross-family requirement (the queue_wait_idle fence alone does
                // not); each helper submits + drains + frees its own command buffer.
                self.backends[producer].p2p_publish_copy(src, export.buffer(), bytes)?;
                self.backends[consumer].p2p_gather_copy(imported.as_ref(), dst, bytes)?;
            }
            Handoff::Host { scratch } => {
                let mut host = scratch.lock().expect("handoff scratch poisoned");
                self.backends[producer].download(src, &mut host)?;
                self.backends[consumer].upload(dst, &host)?;
            }
        }
        Ok(())
    }
}

impl Backend for PipelineBackend {
    fn name(&self) -> &str {
        "vulkan-pipeline"
    }

    fn capabilities(&self) -> Capabilities {
        // Device 0's caps, with the GPU-resident fast paths turned OFF so the runner takes the
        // classic host-embed + host-sample + static per-token execute path — the one the pipeline
        // supports (keeps `hidden` a bound Input, every step a plain `execute`). This is the sole
        // lever that keeps `generate_dense_backend` unmodified.
        let mut c = self.backends[0].capabilities();
        c.decode_replay = false;
        c.embed_gather = false;
        c.gpu_sample = false;
        c.argmax_rows = false;
        c.argmax_prob = false;
        // The residual stream crosses devices between stages, so the caller's buffer is not what
        // the later layers wrote.
        c.graph_input_inplace = false;
        c
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        match usage {
            BufferUsage::KvCache => {
                // Placement by layer order: the runner allocs kbufs[l] then vbufs[l] for
                // l = 0..n_layer, so the pair index IS the layer.
                let pair = self.kv_alloc_pairs.fetch_add(1, Ordering::SeqCst) / 2;
                let d = self.layer_device(pair);
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc(bytes, usage)?,
                ))
            }
            BufferUsage::Readback => {
                let d = self.last_device();
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc(bytes, usage)?,
                ))
            }
            // Weights (ones-vectors / rope come through here — per-layer weights bypass alloc via
            // the binder), Staging (hidden/positions), Activations, HostWeights: replicate.
            _ => self.replicate(|b| b.alloc(bytes, usage)),
        }
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        match usage {
            BufferUsage::KvCache => {
                let pair = self.kv_alloc_pairs.fetch_add(1, Ordering::SeqCst) / 2;
                let d = self.layer_device(pair);
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc_uninit(bytes, usage)?,
                ))
            }
            BufferUsage::Readback => {
                let d = self.last_device();
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc_uninit(bytes, usage)?,
                ))
            }
            _ => self.replicate(|b| b.alloc_uninit(bytes, usage)),
        }
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        let pb = as_pipe(dst)?;
        match pb.single_device() {
            Some(d) => self.backends[d].upload(pb.bufs[0].as_ref(), src),
            None => {
                for (d, b) in self.backends.iter().zip(&pb.bufs) {
                    d.upload(b.as_ref(), src)?;
                }
                Ok(())
            }
        }
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        let pb = as_pipe(src)?;
        // Single-device: read its device. Replicated: read device 0 (the residual/positions are
        // identical across replicas after a step, and no caller downloads a replicated buffer on a
        // hot path).
        let d = pb.single_device().unwrap_or(0);
        self.backends[d].download(pb.on(d)?, dst)
    }

    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        let (s, d) = (as_pipe(src)?, as_pipe(dst)?);
        // Both must share a device (KV fork/seed copies are same-layer → same device). A common
        // single device, or a replica set: copy on every shared device.
        match (s.single_device(), d.single_device()) {
            (Some(sd), Some(dd)) if sd == dd => {
                self.backends[sd].copy_buffer(s.bufs[0].as_ref(), d.bufs[0].as_ref(), bytes)
            }
            (None, None) => {
                for (i, b) in self.backends.iter().enumerate() {
                    b.copy_buffer(s.on(i)?, d.on(i)?, bytes)?;
                }
                Ok(())
            }
            _ => Err(be(
                "pipeline: copy_buffer across devices is not supported (cross-device KV copy)",
            )),
        }
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        Ok(Box::new(PipelinePlan {
            graph: graph.clone(),
            prepared: Mutex::new(None),
        }))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let pp = plan
            .as_any()
            .downcast_ref::<PipelinePlan>()
            .ok_or_else(|| be("pipeline: execute got a non-pipeline plan"))?;
        let mut guard = pp.prepared.lock().expect("pipeline plan poisoned");
        if guard.is_none() {
            *guard = Some(self.prepare(&pp.graph, bindings)?);
        }
        let prep = guard.as_ref().expect("prepared just set");

        for seg in &prep.segments {
            // Hand each cut tensor off from the device that LAST wrote it (already run + synced —
            // op_dev is monotonic, so every producer's segment precedes this one) into this segment.
            for (t, producer, ho) in &seg.cut {
                self.transfer(*t, *producer, seg.device, ho, bindings)?;
            }
            // Per-device bindings: resolve every bound tensor to this device's buffer/replica.
            let d = seg.device;
            let mut sub = Bindings::new();
            let (lo, hi) = seg.op_range;
            let mut needed: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
            for op in &pp.graph.ops[lo..hi] {
                let (r, w) = op.io();
                needed.extend(r);
                needed.extend(w);
            }
            for t in &needed {
                if let Some(buf) = bindings.get(*t) {
                    sub.bind(*t, as_pipe(buf)?.on(d)?);
                }
            }
            self.backends[d].execute(seg.plan.as_ref(), &sub)?;
            self.backends[d].sync()?;
        }
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        for b in &self.backends {
            b.sync()?;
        }
        Ok(())
    }

    fn kv_overflow_report(&self) {
        infr_core::backend::kv_overflow_report_each(&self.backends);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_writer_is_the_most_recent_producer_device() {
        let t = |n: u32| TensorId(n);
        // Op sequence (writes, device):
        //   op0 writes t10 on device 0
        //   op1 writes t11 on device 0
        //   op2 writes t10 on device 1   (re-writes t10 — device 1 is now its last writer)
        //   op3 writes t12 on device 1
        let op_writes = vec![
            (vec![t(10)], 0usize),
            (vec![t(11)], 0usize),
            (vec![t(10)], 1usize),
            (vec![t(12)], 1usize),
        ];
        let m = last_writer_devices(&op_writes);
        assert_eq!(m.get(&t(10)), Some(&1)); // last writer wins (device 1), NOT the first (0)
        assert_eq!(m.get(&t(11)), Some(&0));
        assert_eq!(m.get(&t(12)), Some(&1));
        assert_eq!(m.get(&t(99)), None); // never written
    }

    #[test]
    fn last_writer_handoff_source_is_producer_not_prev_segment() {
        // The bug this guards: a tensor written on device 0, skipped over the device-1 segment, then
        // read on device 2 must hand off from device 0 (its producer), not device 1 (prev segment).
        let t = |n: u32| TensorId(n);
        let op_writes = vec![
            (vec![t(1)], 0usize), // producer of t1 = device 0
            (vec![t(2)], 1usize), // a device-1 op that does NOT touch t1
        ];
        let m = last_writer_devices(&op_writes);
        assert_eq!(m.get(&t(1)), Some(&0));
    }
}
