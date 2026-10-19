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

thread_local! {
    static HEAD_ALLOC_SCOPE: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

fn in_head_alloc_scope() -> bool {
    HEAD_ALLOC_SCOPE.with(std::cell::Cell::get)
}

/// RAII scope that marks buffers allocated by the detached MTP head. Trunk runtime/scratch
/// allocations outside this scope stay on the trunk devices; head allocations replicate to (or
/// pin on) the dedicated head backend.
pub struct HeadAllocScope {
    previous: bool,
}

impl Drop for HeadAllocScope {
    fn drop(&mut self) {
        HEAD_ALLOC_SCOPE.with(|scope| scope.set(self.previous));
    }
}

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, InFlight, Plan};
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
    /// For a REPLICATED buffer: the device that last wrote it (see [`PipelineBuffer::read_device`]).
    /// `NO_WRITER` means every replica holds the caller's value (fresh `alloc`/`upload`), so device
    /// 0 is as good as any. A layer split writes a replicated tensor on ONE lane only — the
    /// residual/hidden is written by whichever segment owns the last op touching it, logits by the
    /// tail head — so a `download` that guessed device 0 would read a never-written replica.
    last_writer: AtomicUsize,
}

/// Sentinel for [`PipelineBuffer::last_writer`]: no segment has written this buffer yet.
const NO_WRITER: usize = usize::MAX;

impl PipelineBuffer {
    /// A single-device buffer (weights/KV/logits).
    pub fn single(device: usize, buf: Box<dyn Buffer>) -> Box<dyn Buffer> {
        Box::new(PipelineBuffer {
            device: Some(device),
            bufs: vec![buf],
            last_writer: AtomicUsize::new(NO_WRITER),
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

    /// Record that device `dev` just wrote this buffer, so a later [`Self::read_device`] reads the
    /// replica that actually holds the result. Only called for caller-bound tensors; a segment's
    /// internal scratch is per-plan storage and never crosses a segment.
    fn mark_written(&self, dev: usize) {
        self.last_writer.store(dev, Ordering::Release);
    }

    /// The device a [`PipelineBackend::download`] must read this buffer from: the single resident
    /// device for a weight/KV/logits handle, else the replica a segment actually wrote.
    fn read_device(&self) -> usize {
        match self.device {
            Some(d) => d,
            None => match self.last_writer.load(Ordering::Acquire) {
                NO_WRITER => 0,
                d => d,
            },
        }
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

/// Which half of a cut tensor's handoff a per-device pass performs. The producer's device thread
/// PUBLISHes (producer replica → export / host scratch); the consumer's device thread GATHERs
/// (export / host scratch → consumer replica). Splitting the two halves by device is what lets one
/// thread own each device's queue: no thread ever submits to a device it does not own.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cuts {
    /// This device's segments are being run by this thread: move the cuts into them.
    Move,
    /// The cuts were already moved (or belong to a segment this device does not run): skip.
    Skip,
}

/// One-shot host-side handshake between the device-0 and device-1 threads of
/// [`PipelineBackend::execute_pipelined`]. `signal` releases every `wait`; `abort` releases them
/// with `false` so a failing device-0 pass can never leave the worker blocked on `join`.
struct Gate {
    state: Mutex<(bool, bool)>,
    cv: std::sync::Condvar,
}

impl Gate {
    fn new() -> Self {
        Self {
            state: Mutex::new((false, false)),
            cv: std::sync::Condvar::new(),
        }
    }

    fn signal(&self) {
        let mut s = self.state.lock().expect("pipeline gate poisoned");
        s.0 = true;
        self.cv.notify_all();
    }

    fn abort(&self) {
        let mut s = self.state.lock().expect("pipeline gate poisoned");
        s.1 = true;
        self.cv.notify_all();
    }

    /// Blocks until released. `false` = the device-0 pass failed; do no work.
    fn wait(&self) -> bool {
        let mut s = self.state.lock().expect("pipeline gate poisoned");
        while !s.0 && !s.1 {
            s = self.cv.wait(s).expect("pipeline gate poisoned");
        }
        s.0
    }
}

struct OverlapTiming {
    base: std::time::Instant,
    marks: Mutex<Vec<(String, f64)>>,
}

impl OverlapTiming {
    fn new() -> Self {
        Self {
            base: std::time::Instant::now(),
            marks: Mutex::new(Vec::new()),
        }
    }

    fn mark(&self, name: impl Into<String>) {
        self.marks.lock().expect("overlap timing poisoned").push((
            name.into(),
            self.base.elapsed().as_secs_f64() * 1e3,
        ));
    }

    fn log(&self, deferred: bool, stages: usize) {
        let marks = self.marks.lock().expect("overlap timing poisoned");
        let body = marks
            .iter()
            .map(|(name, ms)| format!("{name}={ms:.1}"))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::info!(
            "[pipe overlap] deferred={deferred} stages={stages} {body}"
        );
    }
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
    /// Optional extra backend used only by the detached MTP head. It lives on the same physical
    /// GPU as the stage-0 lane, but owns a separate Vulkan device so its hot-expert cache gets an
    /// independent unified VRAM arena.
    head_device: Option<usize>,
    /// `layer_device[l]` = the device index that runs layer `l`. Non-decreasing (a forward
    /// pipeline crosses each device boundary once).
    layer_device: Vec<usize>,
    /// Next `(k,v)`-pair index for a `KvCache` alloc (placement by layer order).
    kv_alloc_pairs: AtomicUsize,
    /// Handoff transport preference: try P2P (dma-buf) first, else host-bounce.
    use_p2p: bool,
    /// Replicate EVERY `BufferUsage::KvCache` allocation on all devices instead of placing it on
    /// its layer's device. Needed whenever the model's tensors with that usage are NOT one `(k,v)`
    /// pair per layer: hybrid SSM/DeltaNet models add recurrent conv/state buffers and hyper-
    /// connection models add a per-layer history buffer, all under `KvCache` — which breaks the
    /// `kv_alloc_pairs` layer-order heuristic and leaves a buffer shared ACROSS layers pinned to a
    /// single device ("op 6 (Conv1dSilu) reads operands on two devices"). Replicating is safe: each
    /// layer's op touches only its own replica, KV fork/seed copies every shared device, and the
    /// segment cut logic hands a replica across a boundary when a later stage needs it.
    /// Ceiling: doubles KV VRAM — fine at modest ctx, not for a max-ctx dense pipeline.
    replicate_kv: bool,
}

impl PipelineBackend {
    /// Build a pipeline over `backends` with a per-layer device assignment `layer_device` (one
    /// entry per model layer, values indexing `backends`, non-decreasing). `use_p2p` requests the
    /// host-less dma-buf handoff (silently falls back to host-bounce per boundary if unsupported).
    pub fn new(
        backends: Vec<VulkanBackend>,
        layer_device: Vec<usize>,
        use_p2p: bool,
        replicate_kv: bool,
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
            head_device: None,
            layer_device,
            kv_alloc_pairs: AtomicUsize::new(0),
            use_p2p,
            replicate_kv,
        })
    }

    /// Attach the dedicated MTP-head backend at the end of the backend list and return its
    /// pipeline device index.
    pub fn attach_head_backend(&mut self, backend: VulkanBackend) -> usize {
        let device = self.backends.len();
        self.backends.push(backend);
        self.head_device = Some(device);
        device
    }

    pub fn head_device(&self) -> Option<usize> {
        self.head_device
    }

    pub fn head_backend(&self) -> Option<&VulkanBackend> {
        self.head_device.map(|device| &self.backends[device])
    }

    pub fn enter_head_alloc_scope(&self) -> HeadAllocScope {
        let previous = HEAD_ALLOC_SCOPE.with(|scope| scope.replace(true));
        HeadAllocScope { previous }
    }

    fn trunk_device_count(&self) -> usize {
        self.head_device.unwrap_or(self.backends.len())
    }

    /// A balanced N-way split of `n_layer` layers across `backends.len()` devices (device `i` gets
    /// a contiguous, near-equal slice). The natural default when the caller gives no explicit map.
    pub fn balanced_layer_map(n_layer: usize, n_dev: usize) -> Vec<usize> {
        let n_dev = n_dev.max(1);
        (0..n_layer)
            .map(|l| (l * n_dev / n_layer.max(1)).min(n_dev - 1))
            .collect()
    }

    /// Like [`Self::balanced_layer_map`], but balanced by WEIGHT BYTES instead of layer count.
    ///
    /// A count split assumes layers are the same size. They are not: in one mixed-imatrix quant the
    /// per-layer weight bytes differ by up to ~1.7x (they are sized by each layer's `ffn_*_exps`
    /// dtypes, which the quantizer picks per layer), and the LAST device alone also carries the
    /// final norm + tied LM head + `token_embd`. Measured on Qwen3.8-Flash-Next IQ3_XXS (48 layers,
    /// 2x24 GiB): the even 24/24 split put 24.7 GiB on the tail device against 19.4 GiB on the head
    /// — over the 23.73 GiB heap before any KV — which is why `INFR_PIPELINE` OOM'd while binding
    /// `blk.47.ffn_down_exps.weight`. Closing each band at its byte share gives 27/21 = 22.14/21.94.
    ///
    /// The split is EXACT — it minimises the heaviest device (binary-searched "split array largest
    /// sum"), not a nearest-to-share approximation; measured max 22.14 GiB against 24.7 for 24/24.
    /// The map stays contiguous + monotonic (each device owns one contiguous band), and every
    /// device keeps at least one layer. `layer_bytes[l]` is layer `l`'s resident weight bytes;
    /// `head_bytes` / `tail_bytes` are the non-layer weights the binder pins to device 0 / the last
    /// device (see [`Self::device_for_weight`]); the tail is charged to the last device's share.
    pub fn byte_balanced_layer_map(
        n_layer: usize,
        n_dev: usize,
        layer_bytes: &[u64],
        head_bytes: u64,
        tail_bytes: u64,
    ) -> Vec<usize> {
        let n_dev = n_dev.max(1);
        if n_dev < 2 || layer_bytes.len() < n_layer || n_layer < n_dev {
            return Self::balanced_layer_map(n_layer, n_dev);
        }
        // `head_bytes` rides in device 0's first band; `tail_bytes` rides on the LAST device.
        let mut lb = layer_bytes[..n_layer].to_vec();
        lb[0] = lb[0].saturating_add(head_bytes);
        let total: u128 = lb.iter().map(|&b| b as u128).sum::<u128>() + tail_bytes as u128;
        // EXACT min-max: binary-search the lightest feasible heaviest-device weight, packing each
        // trial greedily (fill a band as far as it goes, so the last — and tightest, since it also
        // carries the tail — band gets the least). "Split array largest sum", solved properly: a
        // nearest-to-share boundary guess is NOT optimal once the tail is in play.
        let feasible = |m: u128| -> bool {
            let mut used = 1usize;
            let mut acc = 0u128;
            for &b in &lb {
                // The last device's band also carries `tail_bytes`.
                let cap = if used == n_dev { m.saturating_sub(tail_bytes as u128) } else { m };
                if b as u128 > cap {
                    return false;
                }
                if acc + b as u128 > cap {
                    used += 1;
                    if used > n_dev {
                        return false;
                    }
                    acc = 0;
                }
                acc += b as u128;
            }
            true
        };
        let (mut lo, mut hi) = (
            lb.iter().copied().max().unwrap_or(0) as u128,
            total,
        );
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if feasible(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let m = lo;
        // Replay the same greedy at the optimum to materialise the map.
        let mut map = Vec::with_capacity(n_layer);
        let mut d = 0usize;
        let mut acc = 0u128;
        for &b in &lb {
            let cap = if d + 1 == n_dev { m.saturating_sub(tail_bytes as u128) } else { m };
            if acc > 0 && acc + b as u128 > cap {
                d += 1;
                acc = 0;
            }
            map.push(d);
            acc += b as u128;
        }
        debug_assert_eq!(map.len(), n_layer);
        map
    }

    /// [`Self::byte_balanced_layer_map`] against **per-device capacity** instead of one uniform
    /// share.
    ///
    /// The two (or N) cards a layer split runs on are not equally available: one may be driving a
    /// display, and every card loses the VRAM guard's headroom plus driver overhead. A uniform byte
    /// split then plans a band the display card cannot hold, and the load dies on the LAST weight of
    /// that band with a couple hundred MiB of headroom left (`blk.26.ffn_down_exps.weight` on a
    /// 27/21 split of Qwen3.8-Flash-Next IQ3_XXS with a 7900 XTX driving a Wayland session:
    /// 22.14 GiB of weights against ~22.0 GiB of allocatable room).
    ///
    /// The objective is the same min-max, over each band's RATIO to its own device's capacity:
    /// binary-search the smallest ratio `r` (fixed point, `SCALE`) at which a greedy
    /// device-by-device packing keeps every band within `r * caps[device]`. `caps[i]` is what the
    /// caller decided device `i` may spend on weights (its `VramInfo::alloc_room` minus whatever
    /// runtime/arena reserve that caller keeps back), so this function never has to guess what the
    /// allocator will do.
    ///
    /// Returns the uniform-capacity split when the capacities are degenerate (a single device, an
    /// empty capacity, a short `layer_bytes`) — and when even a full-capacity packing cannot hold
    /// the model, which the caller is expected to treat as "this model does not fit resident".
    pub fn byte_balanced_layer_map_caps(
        n_layer: usize,
        layer_bytes: &[u64],
        head_bytes: u64,
        tail_bytes: u64,
        caps: &[u64],
    ) -> Vec<usize> {
        let n_dev = caps.len();
        if n_dev < 2
            || layer_bytes.len() < n_layer
            || n_layer < n_dev
            || caps.iter().any(|&cap| cap == 0)
        {
            return Self::balanced_layer_map(n_layer, n_dev.max(1));
        }
        // `head_bytes` rides in device 0's first band; `tail_bytes` rides on the LAST device.
        let mut lb = layer_bytes[..n_layer].to_vec();
        lb[0] = lb[0].saturating_add(head_bytes);
        const SCALE: u128 = 1 << 20;
        let cap_at = |dev: usize, r: u128| -> u128 { (r * caps[dev] as u128) / SCALE };
        // Greedy packing is optimal for feasibility here: filling an earlier device as far as it
        // goes can only leave less on the later ones.
        let feasible = |r: u128| -> bool {
            if tail_bytes as u128 > cap_at(n_dev - 1, r) {
                return false;
            }
            let mut used = 1usize;
            let mut acc = 0u128;
            for &b in &lb {
                let dev = used - 1;
                let cap = cap_at(dev, r)
                    .saturating_sub(if used == n_dev { tail_bytes as u128 } else { 0 });
                if b as u128 > cap {
                    return false;
                }
                if acc + b as u128 > cap {
                    used += 1;
                    if used > n_dev {
                        return false;
                    }
                    acc = 0;
                }
                acc += b as u128;
            }
            true
        };
        if !feasible(SCALE) {
            // Even at 100% of every device's capacity the bands do not fit: report the uniform
            // split so the caller's own fit/banner path keeps its existing meaning.
            return Self::byte_balanced_layer_map(n_layer, n_dev, layer_bytes, head_bytes, tail_bytes);
        }
        let (mut lo, mut hi) = (0u128, SCALE);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if feasible(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let r = lo;
        let mut map = Vec::with_capacity(n_layer);
        let mut dev = 0usize;
        let mut acc = 0u128;
        for &b in &lb {
            let cap = cap_at(dev, r)
                .saturating_sub(if dev + 1 == n_dev { tail_bytes as u128 } else { 0 });
            if acc > 0 && acc + b as u128 > cap {
                dev += 1;
                acc = 0;
            }
            map.push(dev);
            acc += b as u128;
        }
        debug_assert_eq!(map.len(), n_layer);
        map
    }

    /// Bytes each device's band may occupy for the given map, `head`/`tail` included — the number a
    /// placement decision has to compare against that device's capacity.
    pub fn layer_map_band_bytes(
        layer_device: &[usize],
        n_dev: usize,
        layer_bytes: &[u64],
        head_bytes: u64,
        tail_bytes: u64,
    ) -> Vec<u64> {
        let mut bands = vec![0u64; n_dev];
        for (layer, &bytes) in layer_bytes.iter().enumerate() {
            if let Some(&dev) = layer_device.get(layer) {
                if dev < n_dev {
                    bands[dev] = bands[dev].saturating_add(bytes + if layer == 0 { head_bytes } else { 0 });
                }
            }
        }
        if let Some(last) = bands.last_mut() {
            *last = last.saturating_add(tail_bytes);
        }
        bands
    }

    /// The device that runs layer `l`.
    pub fn layer_device(&self, l: usize) -> usize {
        self.layer_device[l.min(self.layer_device.len().saturating_sub(1))]
    }

    /// Contiguous model-layer range owned by each device. The layer map is non-decreasing, so a
    /// device's range is half-open and may be empty when a caller supplied a sparse map.
    pub fn layer_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut starts = vec![None; self.backends.len()];
        let mut ends = vec![0usize; self.backends.len()];
        for (layer, &device) in self.layer_device.iter().enumerate() {
            if starts[device].is_none() {
                starts[device] = Some(layer);
            }
            ends[device] = layer + 1;
        }
        starts
            .into_iter()
            .zip(ends)
            .map(|(start, end)| start.unwrap_or(end)..end)
            .collect()
    }

    /// The last device (holds the final norm + LM head + logits).
    pub fn last_device(&self) -> usize {
        self.trunk_device_count() - 1
    }

    /// The first device (holds the trunk's leading layers and, in MTP stage-0 mode, the draft
    /// head). Kept symmetric with [`Self::last_device`] for the head binder.
    pub fn first_device(&self) -> usize {
        0
    }

    pub fn n_devices(&self) -> usize {
        self.trunk_device_count()
    }

    /// The device a weight named `name` is placed on: `blk.{l}.*` → layer `l`'s device;
    /// `output*` (the tied LM head) → the last device; `token_embd` and anything else → device 0
    /// (the graph's first op, `EmbedGather`, reads `token_embd`, so it must live on the head
    /// device or the device assignment runs backwards and the monotonic check rejects the split).
    pub fn device_for_weight(&self, name: &str) -> usize {
        if let Some(l) = name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
        {
            if l >= self.layer_device.len() {
                if let Some(head) = self.head_device {
                    return head;
                }
            }
            return self.layer_device(l);
        }
        if name.starts_with("nextn.") {
            if let Some(head) = self.head_device {
                return head;
            }
        }
        if name.starts_with("output") {
            return self.last_device();
        }
        0
    }

    /// The real backend for device `d` — the binder allocs/uploads a per-layer weight through it.
    pub fn backend(&self, d: usize) -> &VulkanBackend {
        &self.backends[d]
    }

    /// Upload the same host bytes to every device and return one replicated weight handle.
    ///
    /// This is for read-only tables whose consumers do not carry a layer identity of their own:
    /// `token_embd` is read by the trunk's first `Op::EmbedGather` on the head device and by the
    /// detached MTP head on the tail device. A single-device binding would pin the MTP gather back
    /// to the head and make an unrolled draft graph jump `tail -> head -> tail`, which the pipeline
    /// deliberately rejects. A replicated handle keeps the table available on both lanes while the
    /// surrounding weight operands still decide where each op runs.
    pub fn upload_replicated(&self, bytes: &[u8], usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        let mut bufs = Vec::with_capacity(self.backends.len());
        for backend in &self.backends {
            let buf = backend.alloc_uninit(bytes.len(), usage)?;
            backend.upload(buf.as_ref(), bytes)?;
            bufs.push(buf);
        }
        Ok(Box::new(PipelineBuffer {
            device: None,
            bufs,
            last_writer: AtomicUsize::new(NO_WRITER),
        }))
    }

    /// Allocate a KV buffer pinned to one device without advancing the layer-pair cursor.
    ///
    /// MTP's detached head owns a KV cache before the trunk's layer-ordered KV allocation starts.
    /// That cache is read/written by head-layer ops pinned to device 0. Sending it through
    /// [`Backend::alloc`]'s `KvCache` arm would consume pair 0 and shift every trunk layer's KV to
    /// the NEXT layer's device; pinning it here keeps the trunk's `l -> device` mapping exact.
    pub fn alloc_kv_on(&self, device: usize, bytes: usize) -> Result<Box<dyn Buffer>> {
        let backend = self
            .backends
            .get(device)
            .ok_or_else(|| be(format!("pipeline: no backend for device {device}")))?;
        Ok(PipelineBuffer::single(
            device,
            backend.alloc(bytes, BufferUsage::KvCache)?,
        ))
    }

    /// Allocate a regular device-local buffer on one backend, bypassing the unified expert arena.
    /// The detached MTP head's KV cache uses this so it cannot consume the hot-expert cache's
    /// frozen corridors.
    pub fn alloc_dedicated_on(
        &self,
        device: usize,
        bytes: usize,
        usage: BufferUsage,
    ) -> Result<Box<dyn Buffer>> {
        let backend = self
            .backends
            .get(device)
            .ok_or_else(|| be(format!("pipeline: no backend for device {device}")))?;
        Ok(PipelineBuffer::single(device, backend.alloc(bytes, usage)?))
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
        Ok(Box::new(PipelineBuffer {
            device: None,
            bufs,
            last_writer: AtomicUsize::new(NO_WRITER),
        }))
    }

    /// Replicate across the TRUNK backends only. The extra MTP-head backend never runs a trunk
    /// layer, so replicating the hybrid model's large shared KV/state buffers onto it wastes its
    /// dedicated hot-expert arena and can make the head cache unplaceable.
    fn replicate_trunk<F>(&self, f: F) -> Result<Box<dyn Buffer>>
    where
        F: Fn(&VulkanBackend) -> Result<Box<dyn Buffer>>,
    {
        let trunk = self.trunk_device_count();
        let mut bufs = Vec::with_capacity(trunk);
        for b in &self.backends[..trunk] {
            bufs.push(f(b)?);
        }
        Ok(Box::new(PipelineBuffer {
            device: None,
            bufs,
            last_writer: AtomicUsize::new(NO_WRITER),
        }))
    }

    /// Build the per-device partition of `graph` under `bindings` (called once per plan).
    fn prepare(&self, graph: &Graph, bindings: &Bindings) -> Result<Prepared> {
        let n_ops = graph.ops.len();
        // ── op → device inference ────────────────────────────────────────────────────────────
        // An op runs on the device its single-device (weight/KV) operand lives on; a pure
        // activation op with no such operand inherits the previous op's device.
        let mut op_dev = vec![0usize; n_ops];
        // `Some(d)` ⇒ this op itself reads a single-device (weight/KV) operand on `d`. A `None` op
        // has no natural lane: the walk below hands it `cur`, which is arbitrary AT THE FRONT of a
        // graph (nothing pinned `cur` yet, so it still holds the initial 0).
        let mut op_pin: Vec<Option<usize>> = vec![None; n_ops];
        let mut cur = 0usize;
        let debug_pipe = std::env::var_os("INFR_DEBUG_PIPE_GRAPH").is_some();
        // Per-tensor device ledger for the diagnostic below: the FIRST single-device placement seen
        // for each tensor, so a conflict can name both sides instead of just the op.
        let mut tensor_dev: std::collections::HashMap<TensorId, (usize, String)> =
            std::collections::HashMap::new();
        for (i, op) in graph.ops.iter().enumerate() {
            let (reads, writes) = op.io();
            let mut pinned: Option<usize> = None;
            for t in reads.iter().chain(writes.iter()) {
                // Independent-row (`Graph::independent_rows`) state is bound as one buffer PER LANE,
                // so `get` returns `None` for it. Every lane's copy of a given layer's state lives
                // on that layer's device — that is what makes a batched decode a single segment per
                // device instead of a per-lane scatter — so the op's device is that shared device,
                // and a disagreement means the lanes were placed across the split (reject loudly
                // rather than run with a wrong pin).
                if let Some(rows) = bindings.get_rows(*t) {
                    let mut lanes: Option<usize> = None;
                    for buf in rows {
                        if let Some(d) = as_pipe(*buf)?.single_device() {
                            match lanes {
                                None => lanes = Some(d),
                                Some(p) if p != d => {
                                    return Err(be(format!(
                                        "pipeline: op {} ({}) has independent-row state on two \
                                         devices ({p} and {d}): a batched decode needs every lane's \
                                         copy of a layer's state on that layer's device",
                                        i,
                                        op.kind()
                                    )));
                                }
                                _ => {}
                            }
                        }
                    }
                    if let Some(d) = lanes {
                        tensor_dev.entry(*t).or_insert((d, op.kind().to_string()));
                        match pinned {
                            None => pinned = Some(d),
                            Some(p) if p != d => {
                                return Err(be(format!(
                                    "pipeline: op {i} ({}) pins {p} and {d} across its operands",
                                    op.kind()
                                )));
                            }
                            _ => {}
                        }
                    }
                } else if let Some(buf) = bindings.get(*t) {
                    if let Some(d) = as_pipe(buf)?.single_device() {
                        tensor_dev.entry(*t).or_insert((d, op.kind().to_string()));
                        match pinned {
                            None => pinned = Some(d),
                            Some(p) if p != d => {
                                let a = tensor_dev.get(t).map(|(ad, _)| *ad).unwrap_or(d);
                                return Err(be(format!(
                                    "pipeline: op {} ({}) reads operands on two devices ({p} and \
                                     {d}) — a weight/KV tensor is misplaced for a layer split \
                                     [tensor {t:?} is on {d}, the op's pin was {p}, other side \
                                     {a}]",
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
            op_pin[i] = pinned;
            op_dev[i] = cur;
            if debug_pipe {
                eprintln!(
                    "[pipe-graph] op {i:4} dev={cur} {}{}",
                    op.kind(),
                    if reads.is_empty() {
                        String::new()
                    } else {
                        format!(" reads={:?}", reads)
                    }
                );
            }
        }
        // A LEADING run of purely activation ops has no anchor of its own, so the forward walk pins
        // it to device 0 by default. That is wrong for a detached-head graph (MTP's draft/catch
        // graphs start at an `EmbedGather` whose table is replicated, so the gather has no pin while
        // every op after it is pinned to the tail): the head's first segment then runs on device 0
        // while its consumer segment runs on the tail, and the tensor in between is INTERNAL
        // scratch — per-segment storage that is never bound, so `prepare`'s cut list skips it and
        // the consumer reads a never-written buffer (silently wrong logits, zeros on the logits
        // download). Anchor the leading run to the device of the first op that IS pinned, which is
        // where the rest of the graph — and therefore the whole detached head — actually runs.
        if let Some(first_pin) = op_pin.iter().position(|p| p.is_some()) {
            let d = op_pin[first_pin].expect("position() found a pinned op");
            if first_pin > 0 && d != 0 {
                op_dev[..first_pin].fill(d);
            }
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
            // The static per-execute path (no record-once replay) — pipeline forces it here. A
            // measured attempt to inherit the producer's flag instead changed nothing (an MoE decode
            // graph is ineligible for the tape by design — see `decode_eligible`), so the pipeline's
            // per-token cost is NOT a replay-eligibility problem; see the doc's cost section.
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
                    // Independent-row state bound per lane: this pipeline has no per-lane handoff, so
                    // a layer-split graph must place every lane's copy of a layer's state on that
                    // layer's device. Silently skipping the handoff here would let the consumer read
                    // a replica nobody wrote.
                    if bindings.get(t).is_none() && bindings.get_rows(t).is_some() {
                        return Err(be(format!(
                            "pipeline: independent-row state {t:?} produced on device {producer} is \
                             read across the device split — a batched decode must keep every lane's \
                             copy of a layer's state on that layer's device"
                        )));
                    }
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
        self.publish_cut(t, producer, ho, bindings)?;
        self.gather_cut(t, consumer, ho, bindings)
    }

    /// Producer half of [`Self::transfer`]: make the producer replica readable by the consumer.
    /// Runs on the PRODUCER's device thread (see [`Self::execute_pipelined`]).
    fn publish_cut(
        &self,
        t: TensorId,
        producer: usize,
        ho: &Handoff,
        bindings: &Bindings,
    ) -> Result<()> {
        let pb = as_pipe(
            bindings
                .get(t)
                .ok_or_else(|| be("pipeline: cut tensor not bound at publish"))?,
        )?;
        let src = pb.on(producer)?;
        let bytes = pb.len_bytes();
        match ho {
            // producer replica → shared export (producer-local), RELEASING the export to
            // QUEUE_FAMILY_EXTERNAL; the consumer's `gather_cut` then ACQUIREs the imported alias
            // from EXTERNAL and reads it. The QF ownership transfer satisfies the EXCLUSIVE external
            // buffer's cross-family requirement (the queue_wait_idle fence alone does not); each
            // helper submits + drains + frees its own command buffer.
            Handoff::P2p { export, .. } => {
                self.backends[producer].p2p_publish_copy(src, export.buffer(), bytes)
            }
            Handoff::Host { scratch } => {
                let mut host = scratch.lock().expect("handoff scratch poisoned");
                self.backends[producer].download(src, &mut host)
            }
        }
    }

    /// Consumer half of [`Self::transfer`]: pull the published bytes into the consumer's replica.
    /// Runs on the CONSUMER's device thread.
    fn gather_cut(
        &self,
        t: TensorId,
        consumer: usize,
        ho: &Handoff,
        bindings: &Bindings,
    ) -> Result<()> {
        let pb = as_pipe(
            bindings
                .get(t)
                .ok_or_else(|| be("pipeline: cut tensor not bound at gather"))?,
        )?;
        let dst = pb.on(consumer)?;
        let bytes = pb.len_bytes();
        match ho {
            Handoff::P2p { imported, .. } => {
                self.backends[consumer].p2p_gather_copy(imported.as_ref(), dst, bytes)
            }
            Handoff::Host { scratch } => {
                let host = scratch.lock().expect("handoff scratch poisoned");
                self.backends[consumer].upload(dst, &host)
            }
        }
    }

    /// Lock `plan`'s cached per-device split (building it on first use) and hand it to `f`.
    fn with_prep<R>(
        &self,
        plan: &dyn Plan,
        bindings: &Bindings,
        f: impl FnOnce(&PipelinePlan, &Prepared) -> Result<R>,
    ) -> Result<R> {
        let pp = plan
            .as_any()
            .downcast_ref::<PipelinePlan>()
            .ok_or_else(|| be("pipeline: execute got a non-pipeline plan"))?;
        let mut guard = pp.prepared.lock().expect("pipeline plan poisoned");
        if guard.is_none() {
            *guard = Some(self.prepare(&pp.graph, bindings)?);
        }
        f(pp, guard.as_ref().expect("prepared just set"))
    }

    /// Resolve one segment's tensors to the replicas on its own device. The returned bindings
    /// borrow the caller's buffers and therefore must outlive the segment submission.
    fn bind_segment<'a>(
        &self,
        pp: &PipelinePlan,
        seg: &Segment,
        bindings: &'a Bindings<'a>,
    ) -> Result<(Bindings<'a>, Vec<TensorId>)> {
        let d = seg.device;
        let mut sub = Bindings::new();
        let (lo, hi) = seg.op_range;
        let mut needed: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
        let mut written_here: Vec<TensorId> = Vec::new();
        for op in &pp.graph.ops[lo..hi] {
            let (r, w) = op.io();
            needed.extend(r);
            written_here.extend(w.iter().copied());
            needed.extend(w);
        }
        for t in &needed {
            if let Some(buf) = bindings.get(*t) {
                sub.bind(*t, as_pipe(buf)?.on(d)?);
            } else if let Some(rows) = bindings.get_rows(*t) {
                let mut lanes: Vec<&dyn Buffer> = Vec::with_capacity(rows.len());
                for buf in rows {
                    lanes.push(as_pipe(*buf)?.on(d)?);
                }
                sub.bind_rows(*t, lanes);
            }
        }
        Ok((sub, written_here))
    }

    /// Execute one segment on its own device with that device's sub-bindings.
    fn run_segment(&self, pp: &PipelinePlan, seg: &Segment, bindings: &Bindings) -> Result<()> {
        let time_it = std::env::var_os("INFR_PIPE_TIME").is_some();
        let t_bind = std::time::Instant::now();
        let d = seg.device;
        // Device identity for the timeline: `Query::new` snapshots the thread-local context when
        // the recorder opens, and the per-segment spans below inherit the same guard.
        let _timeline_device = infr_core::timeline::device(d);
        let (lo, hi) = seg.op_range;
        let (sub, written_here) = self.bind_segment(pp, seg, bindings)?;
        let bind_ms = t_bind.elapsed().as_secs_f64() * 1e3;
        let t_exec = std::time::Instant::now();
        self.backends[d].execute(seg.plan.as_ref(), &sub)?;
        let exec_ms = t_exec.elapsed().as_secs_f64() * 1e3;
        let t_sync = std::time::Instant::now();
        // Kept even though the producer's `execute` already drained (the static path's
        // `queue_wait_idle`): the pipeline's own segment order is what makes the cut handoff legal,
        // and an experiment that skipped BOTH the drain and this wait device-lost the GPU
        // (see the doc's cost section — freeing/reusing the transients needs the GPU finished).
        self.backends[d].sync()?;
        for t in &written_here {
            if let Some(buf) = bindings.get(*t) {
                as_pipe(buf)?.mark_written(d);
            }
        }
        if time_it {
            tracing::info!(
                "[pipe seg] dev={d} ops={} bind={bind_ms:.1}ms exec={exec_ms:.1}ms sync={:.1}ms",
                hi - lo,
                t_sync.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(())
    }

    fn run_segment_nowait<'a>(
        &'a self,
        pp: &PipelinePlan,
        seg: &Segment,
        bindings: &Bindings,
    ) -> Result<PipelineSegmentFlight<'a>> {
        let d = seg.device;
        let _timeline_device = infr_core::timeline::device(d);
        let (sub, written_here) = self.bind_segment(pp, seg, bindings)?;
        let inflight = self.backends[d].execute_nowait(seg.plan.as_ref(), &sub)?;
        // Submission is queued; the command buffer references session-owned buffers, so the
        // per-device `sub` can drop now. Mark the replica before any caller can download it (the
        // caller must wait the returned handle first).
        for t in &written_here {
            if let Some(buf) = bindings.get(*t) {
                as_pipe(buf)?.mark_written(d);
            }
        }
        Ok(PipelineSegmentFlight { inflight })
    }

    /// Run `device`'s segments of `prep`, in order. `Cuts::Move` first gathers their cut tensors
    /// (this is the CONSUMER's thread; the producer's thread published them already).
    fn run_device_segments(
        &self,
        pp: &PipelinePlan,
        prep: &Prepared,
        bindings: &Bindings,
        device: usize,
        cuts: Cuts,
    ) -> Result<()> {
        for seg in prep.segments.iter().filter(|s| s.device == device) {
            if cuts == Cuts::Move {
                for (t, _, ho) in &seg.cut {
                    self.gather_cut(*t, device, ho, bindings)?;
                }
            }
            self.run_segment(pp, seg, bindings)?;
        }
        Ok(())
    }

    /// Producer half of every cut tensor belonging to a segment on a device OTHER than `from`:
    /// the publish a deferred device-1 pass will gather on its next call. One copy per cut tensor.
    fn publish_remote_cuts(&self, prep: &Prepared, bindings: &Bindings, from: usize) -> Result<()> {
        for seg in prep.segments.iter().filter(|s| s.device != from) {
            for (t, producer, ho) in &seg.cut {
                self.publish_cut(*t, *producer, ho, bindings)?;
            }
        }
        Ok(())
    }
}

struct PipelineSegmentFlight<'a> {
    inflight: Option<Box<dyn infr_core::backend::InFlight + 'a>>,
}

impl infr_core::backend::InFlight for PipelineSegmentFlight<'_> {
    fn wait(self: Box<Self>) -> Result<()> {
        let this = *self;
        if let Some(inflight) = this.inflight {
            inflight.wait()?;
        }
        Ok(())
    }

    fn is_complete(&self) -> Result<bool> {
        self.inflight
            .as_ref()
            .map_or(Ok(true), |inflight| inflight.is_complete())
    }
}

struct PipelineInFlight<'a> {
    flights: Vec<PipelineSegmentFlight<'a>>,
}

impl infr_core::backend::InFlight for PipelineInFlight<'_> {
    fn wait(self: Box<Self>) -> Result<()> {
        let this = *self;
        for flight in this.flights {
            Box::new(flight).wait()?;
        }
        Ok(())
    }

    fn is_complete(&self) -> Result<bool> {
        for flight in &self.flights {
            if !flight.is_complete()? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Backend for PipelineBackend {
    fn name(&self) -> &str {
        "vulkan-pipeline"
    }

    /// Broadcast the single-device session-allocation policy to every lane: the batched seam drives
    /// its backend as `dyn Backend`, and each lane's unified arena has to hold its own
    /// session-lifetime allocations open for the same window.
    fn defer_session_finalization(&self, deferred: bool) {
        for backend in &self.backends {
            backend.defer_session_finalization(deferred);
        }
    }

    fn finish_deferred_session_allocations(&self) -> Result<()> {
        for backend in &self.backends {
            backend.finish_deferred_session_allocations()?;
        }
        Ok(())
    }

    fn release_primary_runtime_after_cohort_shrink(&self) {
        for backend in &self.backends {
            backend.release_primary_runtime_after_cohort_shrink();
        }
    }

    fn release_moe_load_reservation(&self) {
        for backend in &self.backends {
            backend.release_moe_load_reservation();
        }
    }

    fn print_moe_pager_stats(&self) {
        for backend in &self.backends {
            backend.print_moe_pager_stats();
        }
    }

    fn print_dense_pager_stats(&self) {
        for backend in &self.backends {
            backend.print_dense_pager_stats();
        }
    }

    fn moe_paged(&self) -> bool {
        self.backends[..self.trunk_device_count()]
            .iter()
            .any(VulkanBackend::moe_paged)
    }

    fn backend_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn capabilities(&self) -> Capabilities {
        // Device 0's caps, with the GPU-resident fast paths turned OFF so the runner takes the
        // classic host-embed + host-sample + static per-token execute path — the one the pipeline
        // supports (keeps `hidden` a bound Input, every step a plain `execute`). This is the sole
        // lever that keeps `generate_dense_backend` unmodified.
        let mut c = self.backends[0].capabilities();
        c.decode_replay = false;
        // LOCAL TRIAL (revert before proposing upstream): MTP VERIFY hard-requires the GPU embed
        // gather (`seam/runner.rs` ~9421 bails otherwise for qwen4exp) and the pipeline DOES
        // support it — `execute` forwards every op in a segment unfiltered, so an EmbedGather node
        // runs on whichever lane owns the table exactly like any other op. Only this one cap is
        // re-enabled: `gpu_sample`/`argmax_*`/`decode_replay` stay off, so each step remains a
        // plain `execute` and `generate_dense_backend` is still untouched.
        c.embed_gather = true;
        c.gpu_sample = false;
        // A/B: MTP VERIFY's rows>1 argmax reads back m u32s instead of m*vocab f32 logits.
        // Leave the default path untouched unless the caller explicitly opts in.
        c.argmax_rows = std::env::var_os("INFR_PIPE_ARGMAX_ROWS").is_some();
        c.argmax_prob = false;
        // The residual stream crosses devices between stages, so the caller's buffer is not what
        // the later layers wrote.
        c.graph_input_inplace = false;
        c
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        match usage {
            BufferUsage::KvCache
                if self.replicate_kv && bytes >= 4 * 1024 * 1024 && !in_head_alloc_scope() =>
            {
                self.replicate_trunk(|b| b.alloc(bytes, usage))
            }
            BufferUsage::KvCache if self.replicate_kv => self.replicate(|b| b.alloc(bytes, usage)),
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
            BufferUsage::RecurrentState if self.replicate_kv => {
                self.replicate(|b| b.alloc(bytes, usage))
            }
            BufferUsage::RecurrentState => {
                let pair = self.kv_alloc_pairs.fetch_add(1, Ordering::SeqCst) / 2;
                let d = self.layer_device(pair);
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc(bytes, usage)?,
                ))
            }
            BufferUsage::Readback => {
                let d = if in_head_alloc_scope() {
                    self.head_device.unwrap_or_else(|| self.last_device())
                } else {
                    self.last_device()
                };
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc(bytes, usage)?,
                ))
            }
            // Weights (ones-vectors / rope come through here — per-layer weights bypass alloc via
            // the binder), Staging (hidden/positions), Activations, HostWeights: replicate.
            _ if in_head_alloc_scope() || bytes <= 8 * 1024 * 1024 => {
                self.replicate(|b| b.alloc(bytes, usage))
            }
            _ => self.replicate_trunk(|b| b.alloc(bytes, usage)),
        }
    }

    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        match usage {
            BufferUsage::KvCache
                if self.replicate_kv && bytes >= 4 * 1024 * 1024 && !in_head_alloc_scope() =>
            {
                self.replicate_trunk(|b| b.alloc_uninit(bytes, usage))
            }
            BufferUsage::KvCache if self.replicate_kv => {
                self.replicate(|b| b.alloc_uninit(bytes, usage))
            }
            BufferUsage::KvCache => {
                let pair = self.kv_alloc_pairs.fetch_add(1, Ordering::SeqCst) / 2;
                let d = self.layer_device(pair);
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc_uninit(bytes, usage)?,
                ))
            }
            BufferUsage::RecurrentState if self.replicate_kv => {
                self.replicate(|b| b.alloc_uninit(bytes, usage))
            }
            BufferUsage::RecurrentState => {
                let pair = self.kv_alloc_pairs.fetch_add(1, Ordering::SeqCst) / 2;
                let d = self.layer_device(pair);
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc_uninit(bytes, usage)?,
                ))
            }
            BufferUsage::Readback => {
                let d = if in_head_alloc_scope() {
                    self.head_device.unwrap_or_else(|| self.last_device())
                } else {
                    self.last_device()
                };
                Ok(PipelineBuffer::single(
                    d,
                    self.backends[d].alloc_uninit(bytes, usage)?,
                ))
            }
            _ if in_head_alloc_scope() || bytes <= 8 * 1024 * 1024 => {
                self.replicate(|b| b.alloc_uninit(bytes, usage))
            }
            _ => self.replicate_trunk(|b| b.alloc_uninit(bytes, usage)),
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
        // Single-device: read its device. Replicated: read the replica a segment actually wrote —
        // NOT blindly device 0, which is only correct for buffers whose replicas happen to agree
        // (positions/rope) or that no segment has written yet.
        let d = pb.read_device();
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
                let shared = s.bufs.len().min(d.bufs.len());
                for (i, b) in self.backends.iter().take(shared).enumerate() {
                    b.copy_buffer(s.on(i)?, d.on(i)?, bytes)?;
                }
                Ok(())
            }
            _ => Err(be(
                "pipeline: copy_buffer across devices is not supported (cross-device KV copy)",
            )),
        }
    }

    /// Batch of independent same-shape copies (the MTP checkpoint's per-layer KV snapshot). The
    /// trait default issues each copy as its own `copy_buffer`, and a `VulkanBackend::copy_buffer`
    /// is a one-shot submit + full queue drain — so a per-layer snapshot of a 30-layer model costs
    /// ~60 submits and ~60 queue round trips PER DECODE CYCLE. Grouping by device hands each
    /// sub-backend the whole batch, which records it into ONE command buffer and submits once.
    fn copy_buffers(&self, copies: &[(&dyn Buffer, &dyn Buffer, usize)]) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let mut per_device: Vec<Vec<(&dyn Buffer, &dyn Buffer, usize)>> = (0..self.backends.len())
            .map(|_| Vec::with_capacity(copies.len() / self.backends.len() + 1))
            .collect();
        for &(src, dst, bytes) in copies {
            let (s, d) = (as_pipe(src)?, as_pipe(dst)?);
            match (s.single_device(), d.single_device()) {
                (Some(sd), Some(dd)) if sd == dd => {
                    per_device[sd].push((s.bufs[0].as_ref(), d.bufs[0].as_ref(), bytes));
                }
                // A replicated pair (recurrent/HC state) is copied on every lane.
                (None, None) => {
                    let shared = s.bufs.len().min(d.bufs.len());
                    for (dev, lane) in per_device.iter_mut().take(shared).enumerate() {
                        lane.push((s.on(dev)?, d.on(dev)?, bytes));
                    }
                }
                _ => {
                    return Err(be(
                        "pipeline: copy_buffers across devices is not supported (cross-device KV copy)",
                    ))
                }
            }
        }
        for (backend, batch) in self.backends.iter().zip(&per_device) {
            if !batch.is_empty() {
                backend.copy_buffers(batch)?;
            }
        }
        Ok(())
    }

    fn copy_buffer_ranges(
        &self,
        copies: &[(&dyn Buffer, usize, &dyn Buffer, usize, usize)],
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let (first_src, _, first_dst, _, _) = copies[0];
        let first_src = as_pipe(first_src)?;
        let first_dst = as_pipe(first_dst)?;
        match (first_src.single_device(), first_dst.single_device()) {
            (Some(sd), Some(dd)) if sd == dd => {
                let mut prepared = Vec::with_capacity(copies.len());
                for &(src, src_offset, dst, dst_offset, bytes) in copies {
                    let (s, d) = (as_pipe(src)?, as_pipe(dst)?);
                    if s.single_device() != Some(sd) || d.single_device() != Some(dd) {
                        return Err(be(
                            "pipeline: copy_buffer_ranges spans devices for a single-device batch",
                        ));
                    }
                    prepared.push((
                        s.bufs[0].as_ref(),
                        src_offset,
                        d.bufs[0].as_ref(),
                        dst_offset,
                        bytes,
                    ));
                }
                self.backends[sd].copy_buffer_ranges(&prepared)
            }
            (None, None) => {
                // Recurrent/HC state is replicated on every lane. Batch all ranges per lane so
                // each device pays one submit + wait instead of one host round trip per range.
                let mut prepared: Vec<Vec<(&dyn Buffer, usize, &dyn Buffer, usize, usize)>> =
                    (0..self.backends.len()).map(|_| Vec::with_capacity(copies.len())).collect();
                for &(src, src_offset, dst, dst_offset, bytes) in copies {
                    let (s, d) = (as_pipe(src)?, as_pipe(dst)?);
                    if !s.is_replicated() || !d.is_replicated() {
                        return Err(be(
                            "pipeline: copy_buffer_ranges mixes replicated and single-device buffers",
                        ));
                    }
                    let shared = s.bufs.len().min(d.bufs.len());
                    for (dev, lane) in prepared.iter_mut().take(shared).enumerate() {
                        lane.push((
                            s.on(dev)?,
                            src_offset,
                            d.on(dev)?,
                            dst_offset,
                            bytes,
                        ));
                    }
                }
                for (backend, lane) in self.backends.iter().zip(&prepared) {
                    backend.copy_buffer_ranges(lane)?;
                }
                Ok(())
            }
            _ => Err(be(
                "pipeline: copy_buffer_ranges across devices is not supported",
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
        // `INFR_PIPE_TIME`: where a forward's wall time actually goes (per-segment execute+sync vs
        // the host-staged cut transfers). The pipeline has no other timing hook, and a single
        // forward's 145 ms vs ~5 ms/row of GPU work needs one.
        let time_it = std::env::var_os("INFR_PIPE_TIME").is_some();
        let t_all = std::time::Instant::now();
        if time_it {
            tracing::info!("[pipe path] plain-execute");
        }
        let mut seg_ms: Vec<f64> = Vec::new();
        let mut cut_ms: Vec<f64> = Vec::new();
        self.with_prep(plan, bindings, |pp, prep| {
            for seg in &prep.segments {
                // Hand each cut tensor off from the device that LAST wrote it (already run + synced —
                // op_dev is monotonic, so every producer's segment precedes this one) into this segment.
                let t_cut = std::time::Instant::now();
                for (t, producer, ho) in &seg.cut {
                    self.transfer(*t, *producer, seg.device, ho, bindings)?;
                }
                let cut = t_cut.elapsed().as_secs_f64() * 1e3;
                let t_seg = std::time::Instant::now();
                self.run_segment(pp, seg, bindings)?;
                if time_it {
                    cut_ms.push(cut);
                    seg_ms.push(t_seg.elapsed().as_secs_f64() * 1e3);
                }
            }
            Ok(())
        })?;
        if time_it {
            let fmt = |v: &[f64]| {
                v.iter()
                    .map(|x| format!("{x:.1}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            tracing::info!(
                "[pipe time] total={:.1}ms segs=[{}] cuts=[{}]",
                t_all.elapsed().as_secs_f64() * 1e3,
                fmt(&seg_ms),
                fmt(&cut_ms)
            );
        }
        Ok(())
    }

    fn execute_nowait<'a>(
        &'a self,
        plan: &dyn Plan,
        bindings: &Bindings,
    ) -> Result<Option<Box<dyn infr_core::backend::InFlight + 'a>>> {
        if self.backends.len() != 2 {
            self.execute(plan, bindings)?;
            return Ok(None);
        }
        let flights = self.with_prep(plan, bindings, |pp, prep| {
            let mut current: Vec<PipelineSegmentFlight<'a>> = Vec::new();
            let mut previous_device: Option<usize> = None;
            for device in 0..2usize {
                let segments: Vec<&Segment> = prep
                    .segments
                    .iter()
                    .filter(|segment| segment.device == device)
                    .collect();
                if segments.is_empty() {
                    continue;
                }
                // A later device's cut reads the producer's replica. Wait the producer's handles
                // first; this is the only dependency point between the two halves.
                if previous_device.is_some() {
                    for flight in current.drain(..) {
                        Box::new(flight).wait()?;
                    }
                }
                for segment in segments {
                    for (tensor, producer, handoff) in &segment.cut {
                        self.transfer(*tensor, *producer, segment.device, handoff, bindings)?;
                    }
                    current.push(self.run_segment_nowait(pp, segment, bindings)?);
                }
                previous_device = Some(device);
            }
            Ok(current)
        })?;
        if flights.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Box::new(PipelineInFlight { flights })))
        }
    }

    fn prepare_plan(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        // Compile the per-device subgraphs (and build cut handoffs) without submitting work.
        self.with_prep(plan, bindings, |_pp, _prep| Ok(()))
    }

    /// See [`Backend::execute_pipelined`]: two devices, one thread each, so the device-0 halves of
    /// this call's stages run while the deferred plan's device-1 half runs on the other card —
    /// instead of the two cards taking turns (each is idle ~half of every chunk, because layer L+1
    /// depends on layer L and a layer split puts the halves on different cards).
    ///
    /// Caller invariants: a plan's device-0 half is independent of every EARLIER plan's device-1
    /// half (true for consecutive prefill chunks of one sequence: only the residual handoff and the
    /// KV cache cross chunks, the handoff in the plan's own cached buffers and the KV at disjoint
    /// token positions), and each in-flight chunk binds its OWN buffer set (the deferred half reads
    /// its chunk's buffers, which the next chunk must not reuse).
    fn execute_pipelined(
        &self,
        deferred: Option<(&dyn Plan, &Bindings)>,
        stages: &[(&dyn Plan, &Bindings)],
        after_dev0: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        if self.backends.len() != 2 {
            // Not the 2-way layer split this overlap is written for: the generic serial walk.
            return Backend::execute_pipelined(self, deferred, stages, after_dev0);
        }
        if std::env::var_os("INFR_PIPE_TIME").is_some() {
            tracing::info!(
                "[pipe path] pipelined deferred={} stages={}",
                deferred.is_some(),
                stages.len()
            );
        }
        let timing = std::env::var_os("INFR_PIPE_TIME").is_some().then(OverlapTiming::new);
        if let Some(timing) = &timing {
            timing.mark("call");
        }
        let gate = Gate::new();
        // The attribution context (decode step / phase) is thread-local, so the device-1 worker
        // would otherwise submit everything unattributed. Snapshot it here and adopt it there.
        let inherited = infr_core::timeline::current_context();
        std::thread::scope(|scope| -> Result<()> {
            // Device 1: gather + run the deferred plan's segments, released by the gate. The
            // publishes it consumes were done by the PREVIOUS call (deferred.dev0 ran there) and by
            // this call's publish pass below (see `publish_remote_cuts`).
            let worker = scope.spawn(|| -> Result<()> {
                let _context = infr_core::timeline::adopt(inherited);
                let Some((plan, bindings)) = deferred else {
                    return Ok(());
                };
                if let Some(timing) = &timing {
                    timing.mark("dev1_wait");
                }
                if !gate.wait() {
                    return Ok(()); // device-0 pass failed: run nothing
                }
                if let Some(timing) = &timing {
                    timing.mark("dev1_go");
                }
                self.with_prep(plan, bindings, |pp, prep| {
                    self.run_device_segments(pp, prep, bindings, 1, Cuts::Move)
                })
                .map(|()| {
                    if let Some(timing) = &timing {
                        timing.mark("dev1_done");
                    }
                })
            });
            // Device 0 (this thread): publish the deferred plan's cuts, release device 1, then run
            // every stage's device-0 half — publishing the cuts its own deferred device-1 half will
            // gather on the NEXT call.
            let main = (|| -> Result<()> {
                if let Some((plan, bindings)) = deferred {
                    self.with_prep(plan, bindings, |_pp, prep| {
                        self.publish_remote_cuts(prep, bindings, 0)
                    })?;
                }
                if let Some(timing) = &timing {
                    timing.mark("dev1_published");
                }
                gate.signal();
                if let Some(timing) = &timing {
                    timing.mark("dev1_released");
                }
                for (i, (plan, bindings)) in stages.iter().enumerate() {
                    self.with_prep(*plan, *bindings, |pp, prep| {
                        self.run_device_segments(pp, prep, bindings, 0, Cuts::Skip)?;
                        self.publish_remote_cuts(prep, bindings, 0)
                    })?;
                    after_dev0(i)?;
                    if let Some(timing) = &timing {
                        timing.mark(format!("stage{i}_done"));
                    }
                }
                Ok(())
            })();
            let result = match main {
                Ok(()) => worker
                    .join()
                    .map_err(|_| be("pipeline: device-1 worker panicked"))?,
                Err(error) => {
                    // Release the worker with a failure so `join` cannot hang on a gate that will
                    // never be signaled.
                    gate.abort();
                    let _ = worker.join();
                    Err(error)
                }
            };
            if let Some(timing) = &timing {
                timing.mark("joined");
            }
            result
        })
        .map(|result| {
            if let Some(timing) = &timing {
                timing.log(deferred.is_some(), stages.len());
            }
            result
        })
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

    // ── segmented (lazily committed) KV ─────────────────────────────────────────
    // Replicated exactly like the flat KV path above (`replicate_kv`: Qwen4's recurrent/HC state
    // under `KvCache` usage needs every device to hold its own copy), and for the same reason the
    // segmented layout has no layer index to route by. Sessions that do not replicate keep the
    // flat path unchanged, which is what `Ok(None)` says.
    //
    // Each device keeps its OWN arena reservation, so a 32K-token segment is claimed per device
    // and the whole set stays addressable from the plan (`prepare` binds the `dev`-th replica).

    fn segmented_kv_available(&self) -> bool {
        self.backends.iter().all(|b| b.segmented_kv_available())
    }

    fn alloc_segmented_kv(
        &self,
        spec: infr_core::backend::SegmentedKvSpec,
    ) -> Result<Option<Box<dyn Buffer>>> {
        if !self.replicate_kv {
            return Ok(None);
        }
        let mut bufs = Vec::with_capacity(self.backends.len());
        for b in &self.backends {
            match b.alloc_segmented_kv(spec)? {
                Some(buf) => bufs.push(buf),
                None => return Ok(None),
            }
        }
        Ok(Some(Box::new(PipelineBuffer {
            device: None,
            bufs,
            last_writer: AtomicUsize::new(NO_WRITER),
        })))
    }

    fn ensure_segmented_kv(&self, buffer: &dyn Buffer, segments: usize) -> Result<()> {
        let pipe = as_pipe(buffer)?;
        for (dev, b) in self.backends[..self.trunk_device_count()]
            .iter()
            .enumerate()
        {
            b.ensure_segmented_kv(pipe.on(dev)?, segments)?;
        }
        Ok(())
    }

    fn ensure_segmented_kv_batch(&self, buffers: &[&dyn Buffer], segments: usize) -> Result<()> {
        // One transaction PER DEVICE over the whole plane set, so a context boundary can never
        // leave device 0 grown and device 1 not.
        let trunk = self.trunk_device_count();
        let mut groups: Vec<Vec<&dyn Buffer>> = (0..trunk)
            .map(|_| Vec::with_capacity(buffers.len()))
            .collect();
        for &buffer in buffers {
            let pipe = as_pipe(buffer)?;
            for (dev, group) in groups.iter_mut().enumerate() {
                group.push(pipe.on(dev)?);
            }
        }
        for (b, group) in self.backends[..trunk].iter().zip(&groups) {
            b.ensure_segmented_kv_batch(group, segments)?;
        }
        Ok(())
    }

    fn clear_segmented_kv(&self, buffer: &dyn Buffer) -> Result<()> {
        let pipe = as_pipe(buffer)?;
        for (dev, b) in self.backends[..self.trunk_device_count()]
            .iter()
            .enumerate()
        {
            b.clear_segmented_kv(pipe.on(dev)?)?;
        }
        Ok(())
    }

    fn buffer_committed_bytes(&self, buffer: &dyn Buffer) -> Result<usize> {
        let pipe = as_pipe(buffer)?;
        // Replicas grow together, so any one answers; the smallest is the safe one — a divergence
        // must never let the session read a row that is not physically backed on some device.
        let mut committed = usize::MAX;
        for (dev, b) in self.backends[..self.trunk_device_count()]
            .iter()
            .enumerate()
        {
            committed = committed.min(b.buffer_committed_bytes(pipe.on(dev)?)?);
        }
        Ok(committed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte-weighted split must beat (or match) every other 2-device cut on the heavier device,
    /// and stay contiguous. Pattern = the real Qwen3.8 IQ3_XXS shape: ~0.75 GiB layers early,
    /// ~1.10 GiB late, plus a 0.67 GiB head/norm/embd burden only the last device carries.
    #[test]
    fn byte_balanced_split_minimises_the_heavier_device() {
        let bytes: Vec<u64> = (0..48)
            .map(|l| if l < 28 { 805_306_368 } else { 1_181_116_006 })
            .collect();
        let tail = 719_323_136u64;
        let map = PipelineBackend::byte_balanced_layer_map(48, 2, &bytes, 0, tail);
        let cut = map.iter().position(|&d| d == 1).expect("two bands");
        assert!(map[..cut].iter().all(|&d| d == 0), "contiguous + monotonic: {map:?}");
        let (a, b) = (
            bytes[..cut].iter().sum::<u64>(),
            bytes[cut..].iter().sum::<u64>() + tail,
        );
        let best = (1..48)
            .map(|c| {
                bytes[..c].iter().sum::<u64>().max(bytes[c..].iter().sum::<u64>() + tail)
            })
            .min()
            .unwrap();
        assert_eq!(a.max(b), best, "split {a}/{b} is not the byte optimum {best}");

        // Same property on the general N-device path (brute-force every pair of cuts).
        let map3 = PipelineBackend::byte_balanced_layer_map(48, 3, &bytes, 0, tail);
        let (c1, c2) = (
            map3.iter().position(|&d| d == 1).expect("band 1"),
            map3.iter().position(|&d| d == 2).expect("band 2"),
        );
        assert!(map3[..c1].iter().all(|&d| d == 0) && map3[c1..c2].iter().all(|&d| d == 1));
        let band = |s: usize, e: usize| bytes[s..e].iter().sum::<u64>();
        let mine = band(0, c1).max(band(c1, c2)).max(band(c2, 48) + tail);
        let best3 = (1..47)
            .flat_map(|i| (i + 1..48).map(move |j| (i, j)))
            .map(|(i, j)| band(0, i).max(band(i, j)).max(band(j, 48) + tail))
            .min()
            .unwrap();
        assert_eq!(mine, best3, "3-way split {mine} is not the byte optimum {best3}");
    }

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
