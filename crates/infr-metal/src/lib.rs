// Apple-only: the `metal`/`objc` crates link the Objective-C runtime and don't build off macOS.
// The whole crate compiles to nothing elsewhere (its consumers gate their use to macOS too), so the
// Linux-only CI still builds the workspace.
#![cfg(target_os = "macos")]
//! Reference Metal compute backend — a correctness-first implementation of the [`Backend`] seam
//! (the same one `infr-cpu` and `infr-vulkan` implement) on Apple's Metal API.
//!
//! Priorities are *correctness and clarity* first, then not being needlessly slow. Each op's
//! arithmetic runs in a small Metal compute kernel over f32 `MTLBuffer`s, following `infr-cpu`'s
//! dataflow closely so it stays in numeric parity (see `tests/parity.rs`).
//!
//! To avoid a CPU↔GPU round-trip per op, graph tensors stay *resident on the device*: a per-forward
//! executor (`exec::Resident`) tracks whether each tensor's current value is on the host or the
//! device, encodes consecutive GPU ops into a single command buffer (Metal hazard-tracks the
//! barriers), and only syncs to the host at a host-side op or the final write-back. Quantized
//! `Op::Linear` weights are kept in the compact factored form (`dequant_factored`: bit-packed
//! 4/6/8-bit codes + one i16 `(sc, m)` per 16 elems + one f16 `(d, dmin)` per quant block) and
//! decoded inline by the `linear_quik*` kernels, so the kernels stay format-agnostic and never blow
//! a quant weight up to f32. `INFR_PROF_OPS=1` prints a per-op / GPU-wall breakdown on drop;
//! `=2` isolates per-op GPU wall by flushing after each op (distorts totals); `=3` samples
//! stage-boundary GPU timestamps per op inside ONE command buffer — true in-context per-op GPU
//! time (the decode tape is disabled so every token's ops are walked and attributed).
//!
//! Not bit-for-bit identical to the CPU everywhere: quantized `Op::Linear` reconstructs the exact
//! `dequant` value but dots in f32, whereas the CPU path quantizes the *activation* to Q8 and uses
//! integer dots — so this backend is actually the slightly more accurate of the two. Faster matvec
//! kernels (GEMV occupancy / fusion) are future work.

use std::sync::Arc;

use infr_core::backend::{Backend, Bindings, BufferUsage, Capabilities, GraphPlan, Plan};
use infr_core::config::{Config, MetalCfg};
use infr_core::error::Result;
use infr_core::graph::Graph;
use metal::{Buffer as MtlBuffer, CommandQueue, Device};

mod exec;
mod idcache;
mod pcache;
mod pcache_blob;
mod profile;
mod shaders;

use idcache::{BufId, IdCache};
pub use shaders::{msl_source, PipelineCacheStats};

/// Terse local shorthand for the shared backend-error constructor.
use infr_core::error::backend as be;

/// A device buffer. On Apple Silicon `StorageModeShared` memory is CPU-visible, so `upload`/
/// `download` are plain `memcpy`s against [`MtlBuffer::contents`].
pub struct MetalBuffer {
    raw: MtlBuffer,
    len: usize,
    /// **Allocation identity.** A process-unique serial stamped on every allocation and NEVER
    /// reused. `raw.contents()` is an ADDRESS, not an identity: releasing an `MTLBuffer` returns
    /// its pages to Metal's allocator and a later `newBuffer` hands the same pointer back for
    /// entirely different contents. Anything that memoizes state derived from a buffer's *bytes*
    /// (the [`weight_cache`](MetalBackend::weight_cache) /
    /// [`qui_cache`](MetalBackend::qui_cache)) must therefore compare uids, not pointers — see
    /// [`idcache`].
    uid: u64,
}

impl MetalBuffer {
    /// How this buffer identifies itself to the derived-state caches: recycled slot + unrecycled
    /// identity. See [`idcache::BufId`].
    fn id(&self) -> BufId {
        BufId {
            addr: self.raw.contents() as usize,
            len: self.len,
            uid: self.uid,
        }
    }
}

// MTLBuffer is documented thread-safe for the create/read/write use here; the raw pointer in the
// metal-rs wrapper makes it neither Send nor Sync by default.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl infr_core::backend::Buffer for MetalBuffer {
    fn len_bytes(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Default)]
struct ProfileGate {
    suppressions: std::sync::atomic::AtomicUsize,
}

impl ProfileGate {
    /// Both suppression mechanisms have to clear: this backend's own RAII counter
    /// ([`suppress`](Self::suppress), used by the Metal bench setup) AND the process-wide flag the
    /// other backends' warmup paths set (`infr_llama::with_profiling_suppressed`). They existed
    /// independently — the shared helper's doc said "the Metal path deliberately does NOT use it" —
    /// which meant a bench that suppressed through the shared helper still got Metal's untimed
    /// warmup folded into its profile.
    fn enabled(&self, configured: bool) -> bool {
        configured
            && !infr_core::prof::suppressed()
            && self.suppressions.load(std::sync::atomic::Ordering::Relaxed) == 0
    }

    fn suppress(&self) -> ProfileSuppression<'_> {
        self.suppressions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ProfileSuppression { gate: self }
    }
}

struct ProfileSuppression<'a> {
    gate: &'a ProfileGate,
}

impl Drop for ProfileSuppression<'_> {
    fn drop(&mut self) {
        self.gate
            .suppressions
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    pipelines: shaders::Pipelines,
    /// Dequantized-weight cache: bound weight buffer → device f32 buffer. Weights are bound the
    /// same every step, so a quantized weight is dequantized once and reused.
    ///
    /// Keyed on the buffer's IDENTITY, not its address (see [`idcache`]): this used to be a
    /// `HashMap<contents() as usize, _>`, and a released `MTLBuffer`'s `contents()` pointer comes
    /// straight back out of Metal's allocator for the next allocation — so any backend that saw a
    /// weight buffer freed and a DIFFERENT weight land on its address was served the first
    /// weight's dequant. Silently wrong weights, no error anywhere.
    weight_cache: std::sync::Mutex<IdCache<std::sync::Arc<MtlBuffer>>>,
    /// Native-quant weight cache (same keying and lifetime as `weight_cache`): a quantized weight
    /// kept in its compact factored form — bit-packed 4/6/8-bit codes + i16 (sc, m) per 16 elems +
    /// f16 (d, dmin) per quant block — that the `linear_quik*` kernels decode inline. ~6-8 bpw vs
    /// f32's 32, and reconstructs the exact same value.
    qui_cache: std::sync::Mutex<IdCache<std::sync::Arc<exec::QuiWeight>>>,
    /// Active weight-load progress bar (see [`Backend::weight_progress`]): every
    /// `BufferUsage::Weights`/`HostWeights` allocation advances it (the funnel lives in `alloc`,
    /// so no loader can forget a tensor).
    weight_pb: std::sync::Arc<std::sync::Mutex<Option<indicatif::ProgressBar>>>,
    /// Opt-in execution profiler, on with the cross-backend `prof.ops` (`INFR_PROF_OPS`).
    /// Resolved-once fields so the hot path skips the `Instant` calls when off.
    ///
    /// `flush_per_op` (`prof.metal_device_time=flush`) additionally flushes after each op to
    /// attribute GPU wall per op — costs the batching, so it's an analysis mode, not the fast path.
    pub(crate) profiling: bool,
    pub(crate) flush_per_op: bool,
    /// GPU-counter profiling (`prof.metal_device_time=counters`): per-op GPU time from
    /// stage-boundary timestamp samples — one encoder per op inside ONE command buffer, so the
    /// numbers are in-context (no per-op flush distortion). `None` when the device lacks
    /// stage-boundary sampling or the timestamp counter set.
    pub(crate) counter_set: Option<metal::CounterSet>,
    profile_gate: ProfileGate,
    /// One (cpu_ns, gpu_ticks) correlation taken at init — with a second one at resolve time,
    /// the ratio converts GPU-clock ticks to nanoseconds (the domains drift only with clock
    /// rate changes; a long baseline keeps the estimate stable).
    pub(crate) ts_base: (u64, u64),
    pub(crate) prof: std::sync::Mutex<profile::Profile>,
    /// The recorded decode tape for the seam's record-once replay (`Capabilities::decode_replay`);
    /// single slot, same single-generation lifetime as the weight caches (one backend per
    /// generation, bindings stable across the decode loop).
    pub(crate) replay: std::sync::Mutex<Option<exec::Tape>>,
    /// Op-scratch buffers reused across ops/executes (keyed by (f32 count, tag) — distinct tags
    /// for same-size buffers alive in one op). Reuse across layers is safe: the batch's hazard
    /// tracking orders each layer's writes after the previous layer's reads.
    pub(crate) scratch:
        std::sync::Mutex<std::collections::HashMap<(usize, u8), std::sync::Arc<metal::Buffer>>>,
    /// The engine configuration this backend reads its knobs from — HANDED IN by the caller
    /// ([`MetalBackend::new_with`], S6), held for the backend's whole life, and borrowed (never
    /// cloned) at every read site, including the per-forward exec walk (`docs/config-plan.md`
    /// R4/R6). The 15 `INFR_METAL_NO_*` kill-switches, `INFR_METAL_LMHEAD_MRV`,
    /// `INFR_METAL_NODELTA`, `INFR_METAL_NOMOE` and the two profiler keys all come off it.
    cfg: Arc<Config>,
}

// MTLDevice / MTLCommandQueue are documented thread-safe; the pipeline states are immutable after
// creation. The metal-rs wrappers are !Send/!Sync only because they hold raw obj-c pointers.
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// Borrowed engine configuration — a REFERENCE, never a clone (`docs/config-plan.md` R6).
    ///
    /// `pub` so `infr-llama`'s seam and this crate's own probes read the knobs off the backend they
    /// already hold instead of growing a second env-sourced config.
    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// The Metal kernel-tier slice of [`cfg`](Self::cfg) — what `exec.rs` reads per op.
    pub(crate) fn metal(&self) -> &MetalCfg {
        &self.cfg.kernels.metal
    }

    /// What the persisted pipeline cache did this run: MSL-compile time, pipeline-creation time,
    /// archive hits vs misses, and where the blob lives. See [`PipelineCacheStats`].
    ///
    /// `pub` because it is the only way to MEASURE this backend's startup cost from a Mac — the
    /// dev box has no Metal device, so `tests/pcache.rs` and anyone benchmarking a launch read the
    /// numbers here rather than off a log line.
    pub fn pipeline_cache_stats(&self) -> PipelineCacheStats {
        self.pipelines.cache_stats()
    }

    /// Default constructor for callers with no [`Config`] to hand in — this crate's own GPU
    /// tests/probes and external library users. Resolves `Default` < environment once and forwards
    /// to [`new_with`](Self::new_with). Fallible for the same reason `VulkanBackend::new` is (S5a):
    /// the five LOUD keys are `Config`-sourced, so a layer error must not be swallowed.
    pub fn new() -> Result<Self> {
        let layer = infr_core::config::ConfigLayer::env().map_err(|e| be(e.to_string()))?;
        Self::new_with(Arc::new(Config::load_from_layers(&[layer])))
    }

    /// **The real constructor (S6).** Build a backend reading every knob — the profiler level, the
    /// 15 kernel kill-switches, the DeltaNet/MoE escape hatches, the lm_head mrv ceiling — from the
    /// `cfg` the caller hands in rather than the process environment.
    pub fn new_with(cfg: Arc<Config>) -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| be("no Metal device found"))?;
        let counter_set = cfg
            .prof
            .metal_counters()
            .then(|| {
                if !device
                    .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
                {
                    tracing::warn!(
                        "[infr-metal] prof.metal_device_time=counters: no stage-boundary counter \
                         sampling on this device — falling back to encode-only profiling"
                    );
                    return None;
                }
                let set = device
                    .counter_sets()
                    .into_iter()
                    .find(|cs| cs.name() == "timestamp");
                if set.is_none() {
                    tracing::warn!(
                        "[infr-metal] PROFILE=3: no timestamp counter set — falling back"
                    );
                }
                set
            })
            .flatten();
        let ts_base = {
            let (mut c, mut g) = (0u64, 0u64);
            device.sample_timestamps(&mut c, &mut g);
            (c, g)
        };
        let queue = device.new_command_queue();
        let pipelines = shaders::Pipelines::build(&device, &cfg)?;
        Ok(Self {
            device,
            queue,
            pipelines,
            weight_cache: std::sync::Mutex::new(IdCache::default()),
            qui_cache: std::sync::Mutex::new(IdCache::default()),
            weight_pb: std::sync::Arc::new(std::sync::Mutex::new(None)),
            // The one cross-backend knob (`prof.ops` / `INFR_PROF_OPS`). This backend used to
            // answer only to `INFR_METAL_PROFILE`, so the cross-backend profiling command line
            // silently did nothing here.
            profiling: cfg.prof.ops,
            flush_per_op: cfg.prof.metal_flush_per_op(),
            counter_set,
            profile_gate: ProfileGate::default(),
            ts_base,
            prof: std::sync::Mutex::new(profile::Profile::default()),
            replay: std::sync::Mutex::new(None),
            scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
            cfg,
        })
    }

    /// Temporarily exclude work from the configured profiler. Benchmark setup uses this so
    /// pipeline warmup and depth materialization do not contaminate measured forwards.
    pub fn suppress_profiling(&self) -> impl Drop + '_ {
        self.profile_gate.suppress()
    }

    pub(crate) fn profiling_enabled(&self) -> bool {
        self.profile_gate.enabled(self.profiling)
    }

    pub(crate) fn active_counter_set(&self) -> Option<&metal::CounterSetRef> {
        if self.profiling_enabled() {
            self.counter_set.as_deref()
        } else {
            None
        }
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        if self.profiling {
            self.prof.lock().unwrap().print_summary();
        }
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn weight_progress(
        &self,
        total_bytes: Option<u64>,
    ) -> Box<dyn infr_core::backend::ProgressScope> {
        let pb = infr_core::progress::bar(
            total_bytes,
            "loading weights",
            infr_core::progress::Unit::Bytes,
        );
        *self.weight_pb.lock().unwrap() = Some(pb);
        /// RAII scope over the shared bar cell: dropping finishes and clears the display.
        struct Scope(std::sync::Arc<std::sync::Mutex<Option<indicatif::ProgressBar>>>);
        impl Drop for Scope {
            fn drop(&mut self) {
                if let Some(pb) = self.0.lock().unwrap().take() {
                    pb.finish_and_clear();
                }
            }
        }
        impl infr_core::backend::ProgressScope for Scope {}
        Box::new(Scope(std::sync::Arc::clone(&self.weight_pb)))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            name: self.device.name().to_string(),
            // Metal simdgroup_matrix / dp4a tiers aren't wired through these Vulkan-oriented flags
            // yet; the Metal exec arms select their own kernels. f16/i8 math is available; the
            // matrix/dot-primitive shapes are N-A → None/off.
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
            sg_pref: 0, // Vulkan-only shader-pick field; Metal exec arms pick their own kernels
            // Apple GPUs ARE integrated in the memory sense (`unified_memory` below says so), but
            // `integrated` here means "submits must stay under a GPU watchdog TDR" — the Vulkan/
            // amdgpu `gfx`-ring reset this guards against. Metal has no equivalent for compute and
            // an Apple GPU is nowhere near the ~2-CU class the smaller prefill chunk is calibrated
            // for, so leave it false and keep Metal's chunk exactly as tuned.
            integrated: false,
            compute_units: 0,
            buffer_device_address: false, // Metal backend has no Vulkan buffer-device-address path
            // Metal's per-threadgroup memory limit (MTLDevice.maxThreadgroupMemoryLength) — the
            // analogue of Vulkan's maxComputeSharedMemorySize (typically 32 KB, 64 KB on Apple GPUs).
            max_shared_memory_bytes: self.device.max_threadgroup_memory_length() as u32,
            unified_memory: self.device.has_unified_memory(),
            // Eligible decode graphs are recorded once as a flat dispatch tape and re-encoded
            // per token, with position-dependent ops on dynamic-pos kernels that read the bound
            // positions buffer (see `exec::Tape`). The engine may compile the decode graph once
            // and re-execute it across the whole decode loop.
            decode_replay: true,
            // The reference backend keeps the separate gate/up FFN form (no GatedActFused
            // lowering); the runner's combined-gu upload stays Vulkan-only.
            // One fused [2*nff, ne] gate+up Linear + GatedActFused per FFN — one dispatch and
            // one contiguous weight stream instead of two.
            combined_gu: true,
            // Op::EmbedGather runs on-device (embed_gather.metal, DEC16_* native decode) for
            // F16/BF16/Q8_0/Q4_0/Q5_0/Q4_K/Q6_K/IQ4_NL/IQ4_XS token_embd tables; the runner
            // additionally format-gates via the shared embed_gather_supported list (see the
            // exec arm for the dtypes outside the Metal set, which error loudly).
            embed_gather: true,
            gpu_sample: true,
            // argmax_f32 is single-row (whole-buffer scan, no row offset); the MTP verify
            // accept keeps the host-logits path on Metal — see the Op::Argmax exec arm.
            argmax_rows: false,
            // No fused argmax+prob kernel yet (issue #33 follow-up, Vulkan-only so far) — the MTP
            // draft loop keeps the host `top1_softmax` logits-download path on Metal. See the
            // Op::ArgmaxProb exec arm's Unsupported note.
            argmax_prob: false,
            // One 32-lane per-head reduction with the SiLU gate folded into its store pass.
            gated_rmsnorm: true,
            // Metal's KV kernels index rows directly by position — no ring mapping; the runner
            // keeps full-context KV allocations for SWA layers here.
            kv_swa_ring: false,
            // Same write-back set as the CPU reference (`infr_core::exec::writes_back`): a mutated
            // f32 `Input` is copied back into its bound buffer at the end of the call.
            graph_input_inplace: true,
        }
    }

    fn alloc(
        &self,
        bytes: usize,
        usage: BufferUsage,
    ) -> Result<Box<dyn infr_core::backend::Buffer>> {
        let len = bytes.max(4);
        let raw = self
            .device
            .new_buffer(len as u64, metal::MTLResourceOptions::StorageModeShared);
        // calloc contract: MTL buffers are not guaranteed zeroed; memset the (host-visible) contents.
        unsafe { std::ptr::write_bytes(raw.contents() as *mut u8, 0u8, len) };
        // Advance the weight-load progress bar (if a scope is open) — the single funnel every
        // weight upload passes through (mirrors the Vulkan backend).
        if matches!(usage, BufferUsage::Weights | BufferUsage::HostWeights) {
            if let Some(pb) = self.weight_pb.lock().unwrap().as_ref() {
                pb.inc(bytes as u64);
            }
        }
        Ok(Box::new(MetalBuffer {
            raw,
            len,
            uid: idcache::next_buffer_uid(),
        }))
    }

    fn alloc_uninit(
        &self,
        bytes: usize,
        _usage: BufferUsage,
    ) -> Result<Box<dyn infr_core::backend::Buffer>> {
        // Opt-out: skip the zero-fill. Debug builds poison with 0xFF (= NaN as f32) so a misuse
        // (read-before-write) surfaces loudly in tests instead of relying on lucky zeros.
        let len = bytes.max(4);
        let raw = self
            .device
            .new_buffer(len as u64, metal::MTLResourceOptions::StorageModeShared);
        #[cfg(debug_assertions)]
        unsafe {
            std::ptr::write_bytes(raw.contents() as *mut u8, 0xFFu8, len)
        };
        Ok(Box::new(MetalBuffer {
            raw,
            len,
            uid: idcache::next_buffer_uid(),
        }))
    }

    fn upload(&self, dst: &dyn infr_core::backend::Buffer, src: &[u8]) -> Result<()> {
        let b = metal_buf(dst);
        if src.len() > b.len {
            return Err(be(format!("upload: src {} > buffer {}", src.len(), b.len)));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), b.raw.contents() as *mut u8, src.len());
        }
        Ok(())
    }

    fn download(&self, src: &dyn infr_core::backend::Buffer, dst: &mut [u8]) -> Result<()> {
        let b = metal_buf(src);
        if dst.len() > b.len {
            return Err(be(format!(
                "download: dst {} > buffer {}",
                dst.len(),
                b.len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                b.raw.contents() as *const u8,
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        Ok(())
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        // Invalidate any recorded decode-replay tape: it belongs to the PREVIOUSLY compiled
        // plan, and the tape only matches on a (op-sequence, bound-buffer-address) fingerprint —
        // which can COLLIDE across independent compile+execute calls once the allocator reuses a
        // freed buffer's address, replaying a structurally-stale tape (observed: the MTP head
        // forward, which recompiles a decode-shaped graph every draft step with fresh IO buffers,
        // intermittently got a garbage/zeroed replay of an earlier forward). The seam's decode
        // loop — the ONLY intended replay user — compiles its plan ONCE and executes it per
        // token, so its tape survives and still replays; anything that recompiles gets a clean
        // slate and records afresh.
        *self.replay.lock().unwrap() = None;
        Ok(GraphPlan::boxed(graph))
    }

    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        self.execute_graph(plan, bindings)
    }

    fn execute_chain(
        &self,
        plan: &dyn Plan,
        bindings: &Bindings,
        n: usize,
    ) -> Result<Option<Vec<u32>>> {
        self.execute_graph_chain(plan, bindings, n)
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

fn metal_buf(b: &dyn infr_core::backend::Buffer) -> &MetalBuffer {
    b.as_any()
        .downcast_ref::<MetalBuffer>()
        .expect("metal backend: buffer is not a MetalBuffer (mixed backends?)")
}

#[cfg(test)]
mod tests {
    use super::ProfileGate;

    #[test]
    fn profile_gate_restores_nested_suppression() {
        let gate = ProfileGate::default();
        assert!(gate.enabled(true));

        {
            let _outer = gate.suppress();
            assert!(!gate.enabled(true));
            {
                let _inner = gate.suppress();
                assert!(!gate.enabled(true));
            }
            assert!(!gate.enabled(true));
        }

        assert!(gate.enabled(true));
        assert!(!gate.enabled(false));
    }
}
