//! Vulkan adapter for the agnostic `infr_core::Backend` seam: lower a `Graph` of composite ops onto
//! the existing fused Recorder kernels, recorded into one command buffer. Mirrors the CPU
//! interpreter (`infr-llama::seam`) op-for-op, but executes on the GPU — so the SAME model
//! `Graph` runs on either backend. Built incrementally; ops not yet mapped return an error.

use crate::linear::native_dense_supported;
use crate::recorder::Recorder;
use crate::{be, VulkanBackend};
use infr_core::backend::{Bindings, Buffer, BufferUsage, Plan};
use infr_core::error::{Error, Result};
use infr_core::graph::{Activation, AttnMask, Dsv4CacheFormat, Graph, Op, TensorKind};
use infr_core::shutdown::shutdown_requested;
use infr_core::{Backend, TensorId};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::recorder::RecordedCmd;

/// Vulkan-specific compiled plan. Carries the cloned graph (like the old `GraphPlan`) plus, for an
/// eligible single-token decode graph, a lazily-built record-once replay (see [`DecodeReplay`]). An
/// ineligible graph (prefill batch, gemma/E2B/MoE/qwen35 decode, …) re-records every `execute` — the
/// unchanged static path.
pub(crate) struct VkDecodePlan {
    pub graph: Graph,
    /// True iff the graph is a qwen3-style decode the params-driven `_dyn` kernels can replay (see
    /// [`decode_eligible`]). Computed once at compile time.
    eligible: bool,
    /// Built on the first eligible `execute` and replayed on every one after (interior mutability so
    /// `execute` can stay `&dyn Plan`). `None` until then; always `None` for an ineligible graph.
    replay: Mutex<Option<DecodeReplay>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VkDecodePlan {
    fn boxed(be_: &VulkanBackend, graph: &Graph) -> Box<dyn Plan> {
        Box::new(VkDecodePlan {
            graph: graph.clone(),
            eligible: decode_eligible(be_, graph),
            replay: Mutex::new(None),
        })
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Plan for VkDecodePlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A recorded, replayable decode command buffer plus the persistent buffers its descriptor sets
/// reference. The recording reads `pos`/`kv_len` from `params` (the `_dyn` kernels) and the
/// `attention_kv_dyn` dispatch is fixed regardless of `kv_len`, so ONE recording stays valid across
/// the whole decode — only `params` (and the runner's per-token embedding/position uploads) change.
struct DecodeReplay {
    /// Persistent `Internal` scratch (activations), allocated ONCE. The recorded descriptor sets bind
    /// these, so re-allocating per token would leave the recording pointing at freed buffers.
    scratch: Vec<Option<Box<dyn Buffer>>>,
    /// `[pos, kv_len]` (u32×2) SSBO the `_dyn` kernels read. When `self_advancing`, a
    /// `params_advance` dispatch recorded FIRST increments it on the device every replay
    /// (initialized to `[pos0-1, pos0]` at record time) — the adapter never touches it again.
    params: Box<dyn Buffer>,
    /// The recording starts with the device-side params increment; `execute` skips the host
    /// `read_pos0` + params upload. INFR_NO_GPU_POS=1 at record time forces the old host path.
    self_advancing: bool,
    /// Chained-decode id ring (64 × u32, host-visible): the recording's trailing `id_log`
    /// dispatch writes `ring[pos & 63] = sampled id` each iteration; `execute_chain` reads the
    /// whole chunk back in one go. `Some` only when the graph ends in Argmax/Sample AND
    /// `self_advancing`.
    ring: Option<Box<dyn Buffer>>,
    /// The `positions` Input tensor — `execute` downloads element 0 to learn `pos` for `params`.
    positions: TensorId,
    recorded: RecordedCmd,
    /// Transient GEMM/attention scratch the recording references. Empty for an eligible decode (all
    /// ops are GEMV/silu/dyn kernels with no transient), held anyway so a stray one can't dangle.
    _transient: Vec<Box<dyn Buffer>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn compile(be_: &VulkanBackend, graph: &Graph) -> Result<Box<dyn Plan>> {
    // Eligibility (record-once replay vs per-execute static recording) is a pure function of the
    // graph shape and the backend's `kernels.vulkan.no_replay`, so decide it here, once. `execute`
    // builds the replay lazily on first run.
    Ok(VkDecodePlan::boxed(be_, graph))
}

/// Record-once replay applies to a single-token decode graph the `_dyn` kernels cover: every
/// `Attention` is `rows==1` with a replayable KV dtype (f16, or coupled Q8_0); RoPE rides on
/// `QkNormRope` or an f16-out standalone `Rope` (no ff there). SWA windows/scale/Softcap/MoeFfn/
/// QkNorm and the qwen35 recurrent ops (`Conv1dSilu`/`DeltaNet`) are all replay-safe. Anything
/// else (prefill batches, dequant-prepass KV dtypes, f32 in-place Rope) falls back to the static
/// per-execute path.
/// Standard-GGUF-block low-bit KV quants that ride the dequant→f16 prepass path (not native reads).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_kv_quant(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, Q4_0 | Q4_1 | Q5_0 | Q5_1 | Iq4Nl)
}

/// Dense non-f16 KV caches (f32/bf16): stored via a cast-store, read via a cast→f16 prepass.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_kv_dense_alt(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, F32 | Bf16)
}

/// TurboQuant KV caches (WHT-rotated): quantizing WriteKv + dequant→f16 prepass.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_turbo(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, Turbo2 | Turbo3 | Turbo4)
}

/// Any KV cache dtype that rides the dequant/cast → f16 prepass (not f16 or native-Q8).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_kv_prepass(dt: infr_core::DType) -> bool {
    is_kv_quant(dt) || is_kv_dense_alt(dt) || is_turbo(dt)
}

/// Lever 1 (kv-decode-perf-levers #1): mainline low-bit block quants the DECODE split-K kernel can
/// dequant INLINE via a coalesced block-amortized decoder (`attn_partial`'s `dqv4`,
/// `attn_partial_ml_kernel`) — so decode reads the compact cache directly instead of a dequant→f16
/// prepass. A strict subset of the `is_kv_quant` prepass set: exactly the formats with an inline
/// decoder. iq4_nl (codebook gather, awkward to coalesce), bf16/f32/turbo keep the prepass.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn is_kv_inline_decode(dt: infr_core::DType) -> bool {
    use infr_core::DType::*;
    matches!(dt, Q4_0 | Q4_1 | Q5_0 | Q5_1)
}

/// `Op::MoeFfn` small-m fast-path threshold: at or below this many rows (tokens in one forward),
/// use the per-active-expert id-indexed GEMV path instead of the batched whole-bank-streaming GEMM
/// (see the two paths' docs at the `Op::MoeFfn` match arm).
///
/// Measured on Qwen3-30B-A3B-Q4_K_M / RDNA3 (7900 XTX), forcing each path via `INFR_MOE_SMALL_M`
/// (`=64` / `=0`) at `-d 4096` so both sides run the SAME row count (small path t/s vs batched
/// path t/s):
///
/// | rows (`-p`) |    small |  batched | small path wins by |
/// |------------:|---------:|---------:|--------------------|
/// |           2 |  114.4   |   72.7   | +57%                |
/// |           4 |  158.1   |  125.0   | +26%                |
/// |           8 |  224.4   |  189.6   | +18%                |
/// |          16 |  275.8   |  286.3   | -4% (batched ahead) |
/// |          32 |  326.0   |  414.0   | -21% (batched ahead)|
/// |          64 |  438.8   |  839.4   | -48% (batched ahead)|
///
/// The crossover sits between rows=8 and rows=16 (batched amortizes its per-active-expert weight
/// read across more of the ~n_used·rows assignments sharing an expert as rows grows; the small
/// path re-reads a shared expert's weight once per occurrence, so it degrades faster). `8` sits
/// right at the edge of the small path's win region with headroom to spare.
///
/// `INFR_MOE_SMALL_M` overrides the threshold for experimentation, clamped to the knob's `max`:
/// the small path's GEMV dispatches one workgroup PER (row, slot, out-column) with no row-tile
/// batching, so at the seam's normal prefill chunk size (`INFR_UBATCH`, 1024 rows by default) an
/// UNCLAMPED override turns every prefill chunk into a multi-second single dispatch — this isn't
/// just slow, it trips the amdgpu ring watchdog and device-losts the whole process (reproduced:
/// `INFR_MOE_SMALL_M=100000` on a 1024-row chunk hangs the GPU; see `runner.rs`'s UBATCH doc for
/// the same failure class with an unchunked prefill).
const MOE_SMALL_M: infr_core::tier::EnvRows = infr_core::tier::EnvRows {
    env: "INFR_MOE_SMALL_M",
    default: 8,
    min: 0,
    max: 64,
};

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn moe_small_m_threshold(be_: &VulkanBackend) -> usize {
    MOE_SMALL_M.clamped(be_.cfg().kernels.vulkan.moe_small_m)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn decode_eligible(be_: &VulkanBackend, graph: &Graph) -> bool {
    // `kernels.vulkan.no_replay` (`INFR_SEAM_NO_REPLAY`) forces the static per-execute path
    // (INFR_PROF_OPS timestamps work there; the replay path can't report them). §6.12: ONE field read
    // `is_ok()` here and `!no_replay` in the llama runner — both halves now come off the same
    // `Config` (the runner's moved in S4, this one in S5b).
    if be_.cfg().kernels.vulkan.no_replay {
        return false;
    }
    // Producer opt-out (see `Graph::no_decode_replay`): the replay tape's `_dyn` kernels are only
    // reassociation-equivalent to the static recording, and some consumers (DiffusionGemma's
    // entropy-bound denoise, which chaotically amplifies a 1-ULP KV delta on the prefill frontier
    // row into different committed text) need the static path's exact arithmetic every execute.
    if graph.no_decode_replay {
        return false;
    }
    let mut has_rope = false;
    let mut has_attn = false;
    // Record-once replay drives ALL pos-dependent ops (QkNormRope/WriteKv/Attention) from ONE shared
    // `params` position (a single-token decode: every layer sits at the same absolute position). A
    // graph whose attention ops sit at DIFFERENT positions (e.g. the MTP self-chaining draft, which
    // unrolls n_steps single-token forwards at consecutive positions p, p+1, … into one graph) is
    // NOT a single-token decode — replaying it would collapse every step onto the same KV row + RoPE
    // angle + kv_len. Force those onto the static per-op-position path.
    let mut attn_kv_len: Option<u32> = None;
    for op in &graph.ops {
        match op {
            // MLA attention (DeepSeek V2/V3) uses its own non-standard kernel — not eligible for
            // the split-K record-once replay pipeline. Nor is the V3.2 lightning indexer, which
            // bakes its causal bound from the `pos` push constant at record time (there is no
            // params-driven `_dyn` twin), so a replayed tape would select keys for the position it
            // was recorded at.
            Op::Mla { .. }
            | Op::LightningIndexer { .. }
            | Op::Dsv4Compress { .. }
            | Op::Dsv4CacheWrite { .. }
            | Op::Dsv4Indexer { .. }
            | Op::Dsv4Gather { .. } => return false,
            // Any mask (SWA windows ride push constants + the window-aware prologue) and any
            // scale (gemma4 uses 1.0) — both are baked per-layer into the recorded dispatch.
            // hd%4 ≤ 512 keeps every layer on the self-chunking split path or the scalar
            // fallback, both of which take scale/window.
            Op::Attention {
                rows,
                head_dim,
                k_cache,
                v_cache,
                kv_len,
                sinks,
                ..
            } => {
                has_attn = true;
                if *rows != 1 || *head_dim % 4 != 0 || *head_dim > 512 {
                    return false;
                }
                // Attention sinks have ONE kernel (`attention_kv.comp`'s -DSINKS static build) and
                // no params-driven `_dyn` twin, so a replayed tape would run the sink-free split-K
                // family instead. Force static.
                if sinks.is_some() {
                    return false;
                }
                // Multi-position graph (see the comment above `attn_kv_len`): different kv_len across
                // attention ops ⟹ not a single-token decode ⟹ ineligible for record-once replay.
                match attn_kv_len {
                    Some(prev) if prev != *kv_len => return false,
                    _ => attn_kv_len = Some(*kv_len),
                }
                // A COUPLED Q8_0 KV cache (K==V==Q8) replays: store_q8_dyn writes the row at
                // params' pos and the planar-Q8 dyn attention kernels (attention_kv_dyn_q8 /
                // attn_partial_dynac_q8) read it. (The historic "store_q8_dyn mis-decodes under
                // replay" was actually the Attention lowering passing the ROW capacity where the
                // Q8 kernels expect the ELEMENT capacity `cap` — the planar scales-region base —
                // see the `cap_rows` fix at the Dynamic Attention branch.) A DECOUPLED Q8 side
                // still forces static: the dyn kernels' q8 variant dequants BOTH sides, and the
                // per-side mixed read only exists on the static attn_partial/attention_kv path.
                // The dequant→f16 prepass dtypes (low-bit block quants, f32/bf16, turbo) force
                // static too — their WriteKv/prepass have no dyn kernels.
                let (kdt, vdt) = (graph.desc(*k_cache).dtype, graph.desc(*v_cache).dtype);
                let k_q8 = matches!(kdt, infr_core::DType::Q8_0);
                let v_q8 = matches!(vdt, infr_core::DType::Q8_0);
                if k_q8 != v_q8 || is_kv_prepass(kdt) || is_kv_prepass(vdt) {
                    return false;
                }
            }
            // freq_factors (gemma4 proportional RoPE) binds via qk_norm_rope_dyn_ff.
            Op::QkNormRope { .. } => has_rope = true,
            // Standalone llama Rope replays via rope_f16_dyn — needs the f16-out builder shape
            // (the f32 in-place form has no dyn kernel) and no freq_factors.
            Op::Rope {
                dst, freq_factors, ..
            } => {
                if freq_factors.is_some()
                    || !matches!(graph.desc(*dst).dtype, infr_core::DType::F16)
                {
                    return false;
                }
                has_rope = true;
            }
            // MoeFfn is REPLAY-SAFE: router GEMV + GPU-side top-k + id-indexed expert GEMVs are
            // all push-constant/pos-independent, and its scratch is plan-held. QkNorm (gemma4
            // V-norm) and Softcap are pos-independent elementwise — replay-safe as recorded.
            // Conv1dSilu/DeltaNet (qwen35 recurrent state) are replay-safe too: decode (rows==1)
            // takes the sequential kernels, which are pure in-place RMW of the PERSISTENT state
            // bindings (kbufs/vbufs, stable across the decode loop) with no pos push constant —
            // the recorded tape re-reads the live state every replay, and the tape's leading
            // global barrier orders iteration i's state write before i+1's read under replay_n.
            // GatedRmsNorm (qwen35 DeltaNet z-gate) is pos-independent elementwise like QkNorm —
            // same replay-safety argument.
            Op::MoeFfn { .. }
            | Op::QkNorm { .. }
            | Op::GatedRmsNorm { .. }
            | Op::Softcap { .. }
            | Op::Conv1dSilu { .. }
            | Op::DeltaNet { .. }
            | Op::Kda { .. }
            | Op::HeadwiseSigmoidMul { .. } => {}
            _ => {}
        }
    }
    has_rope && has_attn
}

/// Read `positions[0]` for the record-once decode. The runner writes the position into a host-visible
/// (CpuToGpu) `Staging` buffer via a mapped memcpy each token and no GPU op writes it, so read it
/// straight from the mapping — zero submit/sync. (The generic `download` on a CpuToGpu buffer would
/// allocate a staging buffer + one_shot copy + `queue_wait_idle` EVERY token; on the hot decode loop
/// that per-token full sync is the dominant residual cost.) Falls back to `download` if unmapped.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn read_pos0(be_: &VulkanBackend, buf: &dyn Buffer) -> Result<u32> {
    if let Some(vb) = buf.as_any().downcast_ref::<crate::VkBuffer>() {
        if let Some(ptr) = vb.mapped_ptr() {
            let mut b = [0u8; 4];
            // Safety: the buffer is ≥4 bytes (positions has ≥1 i32) and persistently mapped.
            unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, b.as_mut_ptr(), 4) };
            return Ok(i32::from_le_bytes(b) as u32);
        }
    }
    let mut b = [0u8; 4];
    be_.download(buf, &mut b)?;
    Ok(i32::from_le_bytes(b) as u32)
}

/// Resolve a graph tensor to its device buffer: `Internal` from the scratch, everything else
/// (`Input`/`Weight`/`Output`) from the model-provided `Bindings`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn resolve<'a>(
    scratch: &'a [Option<Box<dyn Buffer>>],
    bindings: &'a Bindings,
    id: TensorId,
) -> Result<&'a dyn Buffer> {
    match &scratch[id.0 as usize] {
        Some(b) => Ok(b.as_ref()),
        None => bindings
            .get(id)
            .ok_or_else(|| be(format!("vulkan adapter: unbound tensor {}", id.0))),
    }
}

/// Allocate the `Internal` scratch (activations) for `graph`. The leading (row) dim is padded to a
/// multiple of 64 so the prefill GEMM / flash kernels — which write ceil(rows/64)*64 output rows —
/// write DIRECTLY into these buffers (no padded temp + copy). Padding rows are never read (downstream
/// ops touch only the real `rows`, and row-major layout keeps element (r<rows, c) at the same index).
/// [`alloc_scratch`]'s result: the per-tensor `Internal` scratch, indexed by `TensorId`.
type ScratchSet = Vec<Option<Box<dyn Buffer>>>;

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn alloc_scratch(be_: &VulkanBackend, graph: &Graph) -> Result<ScratchSet> {
    let mut scratch: Vec<Option<Box<dyn Buffer>>> =
        (0..graph.tensors.len()).map(|_| None).collect();
    // Batch the whole scratch set into ONE zero-init submit (`alloc_zeroed_batch`): the naive
    // per-tensor `alloc` pays a one-shot submit + queue_wait_idle per Internal (~70 x ~35us =
    // ~2.5ms per execute on a 7900 XTX) — the dominant HOST cost of a small-m prefill step.
    let mut idx: Vec<usize> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    for (i, decl) in graph.tensors.iter().enumerate() {
        if matches!(decl.kind, TensorKind::Internal) {
            let numel = decl.desc.numel();
            let padded = match decl.desc.shape.first() {
                Some(&rows) if rows > 0 => rows.div_ceil(64) * 64 * (numel / rows),
                _ => numel,
            };
            let bytes = decl
                .desc
                .dtype
                .dense_bytes(padded)
                .ok_or_else(|| be("vulkan adapter: internal tensor must be a dense dtype"))?;
            idx.push(i);
            sizes.push(bytes.max(4));
        }
    }
    let bufs = be_.alloc_zeroed_batch(&sizes, BufferUsage::Activations)?;
    for (i, buf) in idx.into_iter().zip(bufs) {
        scratch[i] = Some(buf);
    }
    Ok(scratch)
}

/// Peephole: fuse `QkNormRope(k → k16)` + `WriteKv(k16 → cache, pos)` into a single Qk-norm+RoPE that
/// writes directly into the KV cache with the row offset baked in (matches the production Recorder).
/// This drops the intermediate f16 scratch, the copy dispatch, and one pipeline barrier per layer.
///
/// The intermediate tensor (k16) has ONE TensorId reused across ALL layers' QkNormRope ops, so the
/// map is keyed by OP INDEX (not TensorId): each layer's pair maps to its own KV-cache buffer.
/// Returns (fused: op index of the QkNormRope → (cache, row offset `pos`); skip: absorbed WriteKv ops).
/// Per-execute transient-scratch pool: the SAME (tag, bytes) key across ops returns the SAME
/// buffer. Layers are serialized by dataflow and the recorder's hazard tracking turns each
/// rewrite into an ordinary WAR/WAW barrier (the bespoke path shares its split scratch across
/// layers the same way), so per-tag reuse is safe — and it cuts the held transient VRAM from
/// O(n_layer) to O(1) buffers per tag. Without it, an 8B p8000 prefill held 36 layers × ~1GB of
/// flash-attention partials (≈38 GB) and took the device down.
type ScratchPool = HashMap<(&'static str, usize), Box<dyn Buffer>>;

/// mmv (int8 dp4a decode GEMV) size gate for the m≥3 small-m PREFILL mrow path: below this
/// weight-element count the dequant GEMV is already so short (<~10us) that the extra quant_q8
/// dispatch's fixed bubble (~2-3us on a 7900 XTX) eats the kernel saving. Probe data
/// (gemv_vs_mmv, 7900 XTX): 4096x24576 gate+up −38%, 12288x4096 Q6_K down −5.5% (= 48M elements,
/// the smallest clear payer), 4096x4096 o −5% kernel-only but a whole-model LOSS on
/// dispatch-bound decodes (gemma3-1b −2.3%, qwen3moe −6% with no gate).
const MMV_MIN_ELEMS: usize = 48 << 20;

/// The multi-row GEMV (`mrow`) bands of the dense `Linear` m-ladder, in selection order. Only the
/// SHAPE arithmetic lives in the shared [`infr_core::tier`] policy; the capability half (an mrow
/// build for the dtype, `caps.i8_dot`, the int8 dtype sets, `INFR_NO_MROW`/`INFR_NO_MROW16`) stays
/// at the `Op::Linear` arm, which is where the measured rationale for each number is written up.
///
/// - [`MROW_BAND`] (m = 2..=8): the small multi-row batch — spec-decode verify rows, a short
///   chat-turn suffix prefill. m=5..8 gates on `out_f <= 8192` (past that the GEMM's tile grid
///   fills the device and wins).
/// - [`MROW16_BAND`] (m = 9..=16): the int8-only extension for an MTP verify batch carrying
///   committed rows on top of the drafts. Same wide-n cutoff; NO dequant fallback, hence its
///   extra up-front capability gate.
const MROW_BANDS: [infr_core::tier::MultiRowBand; 2] = [
    infr_core::tier::MultiRowBand {
        min_m: 2,
        max_m: 8,
        narrow_max_m: 4,
        wide_out_f_max: 8192,
        out_f_max: usize::MAX,
    },
    infr_core::tier::MultiRowBand {
        min_m: 9,
        max_m: 16,
        narrow_max_m: 0,
        wide_out_f_max: 8192,
        out_f_max: usize::MAX,
    },
];
/// Index of the m = 2..=8 band in [`MROW_BANDS`].
const MROW_BAND: usize = 0;
/// Index of the m = 9..=16 int8 band in [`MROW_BANDS`].
const MROW16_BAND: usize = 1;

/// The m=1 DECODE int8 mmv is DEFAULT-OFF (opt in with `INFR_MMV_DECODE=1`). The word-parallel
/// K-quant `dqblk` (whole-u32 decode, commit 51a35c8) sped up the SCALAR dequant GEMV enough that
/// it now beats the int8 mmv at m=1 across the board — measured `INFR_NO_MMV=1` (scalar) vs mmv,
/// tg128 r=3 on a 7900 XTX: Qwen3-8B 127.3 vs 123.0 (+3.5%), qwen35-4B 142.5 vs 139.4 (+2.2%),
/// 0.6B 667 vs 662 (+0.8%), 8B tg64@d4096 113.7 vs 110.4 (+3.0%). So decode routes to the scalar
/// GEMV; the mmv kernels stay reachable for A/B + future re-tuning. (The m≥3 mrow PREFILL mmv is
/// UNAFFECTED — it still wins: E2B pp4@d4096 loses 5.3% under `INFR_NO_MMV`.)
///
/// §10.2: the two knobs are DIFFERENT grammars and interact here — `INFR_MMV_DECODE` is `presence`
/// (`mmv_decode`, default `false`), `INFR_NO_MMV` is `presence-inv` (`mmv`, default `true`), and the
/// site is `mmv_decode && mmv`.
fn mmv_decode_enabled(vk: &infr_core::config::VulkanCfg) -> bool {
    vk.mmv_decode && vk.mmv
}

/// Multi-warp int8 dp4a decode GEMV route (`native_mmv_mw.comp`, llama's mul_mat_vec_q block:
/// warp-per-row subgroupAdd, WARPS warps/block) for wave32-native GPUs (NVIDIA/Intel — on Intel
/// the warps are pinned SG=16 via `caps.sg_pref`). There the AMD-tuned scalar-dequant decode GEMV
/// runs memory-LATENCY-starved (~30 GB/s of 616 on an RTX 2080 Ti) — a per-op dead end for the
/// scalar path, since it is the f32 dequant ALU + the `v[32]` register pressure (not the
/// reduction) that cap it (a scalar multi-warp variant measured SLOWER than the tree). Only dp4a
/// (raw int8 blocks, no f32 dequant) breaks the ceiling: Qwen3-1.7B Q4_K_M tg128 17.9 → 61.7 t/s
/// isolated (3.4x), 0.11x → ~0.5x of llama.cpp.
///
/// Per-(dtype, vendor) decode-int8 DEFAULT policy — a llama.cpp-style table
/// (`ggml_vk_should_use_mmvq`, `ggml-vulkan.cpp:8877`), but each entry is a number MEASURED on
/// infr's own kernels/hardware, not assumed from llama.cpp's table (their dp4a mmvq kernel and
/// infr's mmv_mw are different code with different per-dispatch overhead, so a win on one engine
/// doesn't imply a win on the other):
///
/// - **Intel**: Q4_K/Q6_K/Q2_K/Q3_K all measured wins (unchanged from before this table existed —
///   `mmv_mw_parity`'s host-reference tests cover the numerics; the Intel A770 throughput deltas
///   predate this doc, see git blame).
/// - **AMD RDNA3/RADV** (this box, `llama-bench` oracle b9957): **Q2_K** measured
///   `infr bench Qwen3-14B-GGUF:Q2_K -p0 -n64` 86.4 → 104.0 t/s (**+20.4%**) — DEFAULT-ON. **Q4_K**
///   was measured a regression at the {4,8}-warps/block shape (78.5 → 71.0 t/s, **−9.6%**) and
///   stayed OFF. Root-caused and the THROUGHPUT half FIXED (README footnote 3): (1) the m=1
///   `dpsub` unpacked Q4_K quants byte-at-a-time instead of the word-parallel nibble-mask
///   `native_mmv_mrow.comp` already used (+5.8%, bit-identical, `mmv_mw_parity`); (2) a
///   dispatch-shape sweep over `INFR_MMV_MW_WARPS` ∈ {1,2,4,8,16} found WARPS=1 (llama.cpp's
///   `rm_kq_int=1` shape — one output row per workgroup, single-subgroup reduce) the clear winner:
///   `Qwen3-14B-GGUF:Q4_K_M` tg64 78.3 → 80.1 t/s (**+2.3%**), tg128 78.0 → 79.8 (+2.3%), tg64@d4096
///   68.2 → 69.3 (+1.6%) — int8 Q4_K now genuinely beats infr's own f32 path in isolation.
///
///   **NOW FLIPPED ON by default on AMD** (was off): turning it on used to break
///   `mtp_spec_matches_target_only_greedy` (`crates/infr-llama/tests/cpu_backend.rs`) — the m=1
///   decode stream (`mmv_mw`) and the m>=3 verify stream (`mrow`, unconditionally int8 for Q4_K
///   already — see `mrow_int8_dtype_ok`) disagreed on the occasional greedy token even though BOTH
///   were int8. Root cause: the two kernels were different code (warp-per-row subgroupAdd vs
///   row-tile accumulation) that both quantized activations to int8 and both dotted the same
///   integers per sub-block, but the cross-sub-block SUMMATION ORDER differed — the same
///   reassociation class `mmv_mw_parity` tolerates at 5e-3 for throughput purposes, but wide enough
///   to occasionally flip a greedy argmax. "Both streams int8" is necessary but not sufficient for
///   token-identity — it also needs the two kernels bit-identical at the same position. THE FIX:
///   AMD's m=1 decode no longer dispatches `native_mmv_mw.comp` at all for these dtypes — it
///   dispatches [`crate::Recorder::linear_mmv_mrow`] with `rows=1`, i.e. the literal same
///   shader/reduction the m>=3 verify tier uses (see [`unified_mmv_row1`] and
///   `native_mmv_mrow.comp`'s header). A row's math depends only on its own activation slice, never
///   on `rows`, so decode and verify are bit-identical by construction — proved by
///   `mmv_row1_bit_identical` (`tests/mmv_row1_bit_identical.rs`). Intel is UNCHANGED (still
///   `native_mmv_mw.comp`, SG=16-pinned, WARPS-tuned) — no Intel GPU in this validation
///   environment, so its already-shipped, already-tuned kernel is left alone rather than swapped
///   blind; see [`unified_mmv_row1`]. **Q6_K is now ON** (`de987d7`) — the word-parallel `wdec`
///   rewrite turned its decode loss into a win, and the `mtp_spec_matches_target_only_greedy`
///   failure that used to block it no longer gates anything: the MTP speculative path is PARKED
///   (`infr_llama::mtp::mtp_enabled` returns false). **Q3_K stays OFF** — not for MTP reasons but
///   for a real coherence cliff on mixed GGUFs; see the dedicated note in the `None` arm below.
///
/// Symmetry (MTP-verify only): this policy set is ALSO what gates every dtype's entry into the
/// m>=3 int8 `mrow` kernel for an MTP-VERIFY batch (`Graph::mtp_verify`, see
/// [`mrow_int8_dtype_ok`]'s `verify` arm) — a dtype added here is int8 in BOTH the decode stream
/// and the verify stream, or in NEITHER. That is the invariant whose violation broke Q5_K
/// token-identity (decode int8, verify f32-exact, or vice versa: the occasional greedy argmax
/// flips between the spec and non-spec streams). ORDINARY prefill is NOT gated by this set at all
/// — see [`mrow_int8_prefill_dtypes`], which every dtype below is also a member of (this decode
/// set is always a subset of the prefill set: prefill has no bit-identity partner to protect, so
/// it never needs to be MORE conservative than decode). Concretely on AMD today: Q2_K/Q4_K/Q6_K/
/// Q4_0/Q5_0/Q5_1/IQ4_NL int8 in both decode AND verify (the unified kernel makes them
/// bit-identical, not just both-int8 — see [`unified_mmv_row1`]); Q3_K/Q5_K/IQ4_XS/Q8_0/Q4_1
/// f32-exact in both decode AND verify, while ALL of them take the int8 `mrow` kernel at ordinary
/// prefill (every integer dtype is a measured prefill win — see [`mrow_int8_prefill_dtypes`]).
/// NOTE: the MTP-verify half of this symmetry is currently DORMANT, not load-bearing — the
/// speculative path is parked (`infr_llama::mtp::mtp_enabled`), so nothing dispatches a verify
/// batch today. The invariant and its unit guard are kept green so re-enabling MTP is a policy
/// question, not an archaeology exercise.
///
/// `INFR_MMV_MW=1` force-enables EVERY dtype with an int8 decode arm (all 11) on ANY vendor for
/// A/B measurement (of the DECODE + MTP-verify tier only — ordinary prefill is
/// already unconditionally on for these dtypes and unaffected by this env); `INFR_MMV_MW=0`
/// force-off everywhere (decode + verify only; prefill is instead controlled by the general
/// `INFR_NO_MMV`/`INFR_NO_MROW` kill switches, same as any other mrow dtype). Both flow through
/// [`mmv_int8_decode_dtypes`], so the A/B escapes stay symmetric too. `INFR_MMV_MW_WARPS` ∈
/// {1,2,4,8,16} only affects Intel now (AMD's decode tier no longer reads WARPS — see
/// [`unified_mmv_row1`]); {1,2,16} are Q4_K-only dispatch-shape sweep builds (see
/// `native_mmv_mw_kernel_name`), not shipped for Q6_K/Q2_K/Q3_K/Q5_K.
///
/// **Q5_K** — a NEW int8 arm this session; it previously had no int8 kernel in EITHER stream, which
/// is why the historical Q5_K attempt (verify-int8 / decode-f32-exact) broke MTP token-identity.
/// Both `native_mmv_mrow.comp` (FMT_Q5K, the unified AMD decode+verify kernel) and
/// `native_mmv_mw.comp` (FMT_Q5K, Intel's decode kernel) now have one, mirroring Q4_K's
/// word-parallel wdec plus a 5th-bit `qh` plane. AMD MEASUREMENTS (Qwen3-14B-Q5_K_M, r=5/r=3, this
/// box, post-unification): decode `-p0 -n64` **66.8 int8 vs 67.8 f32 — a small LOSS, -1.4%**
/// (reproducible across 3 alternating runs); prefill `pp4@d4096` **188 int8 vs 130 f32 — +45%**.
/// That is the same loses-at-decode, wins-big-at-prefill split Q6_K and IQ4_XS already show
/// (per-dispatch activation-quantize overhead is dead weight at m=1, amortized hard at m>=3). So
/// Q5_K is **NOT** in either vendor's DECODE default set (no decode win to justify it) — but its
/// ordinary-prefill win IS banked, unconditionally, via [`mrow_int8_prefill_dtypes`]: the
/// MTP-verify/decode split (`Graph::mtp_verify`) is exactly the "clean way to bank it" this doc
/// used to defer to, and it is now in place. MTP verify still runs Q5_K f32-exact (matching
/// decode), so token-identity holds; only ordinary prefill takes the win.
fn mmv_int8_decode_dtypes(
    _caps: &infr_core::backend::Capabilities,
    vk: &infr_core::config::VulkanCfg,
) -> &'static [infr_core::DType] {
    use infr_core::DType::{Iq4Nl, Q2K, Q3K, Q4K, Q4_0, Q4_1, Q5K, Q5_0, Q5_1, Q6K, Q8_0};
    // §10.3: `INFR_MMV_MW` is TRI-state — `mmv_mw: Option<bool>`, where `Some(false)` is the exact
    // string `"0"`, `Some(true)` any other value, and `None` (unset) the unified default below.
    match vk.mmv_mw {
        Some(false) => &[], // force-off everywhere
        // Explicit opt-in: every dtype with an int8 decode arm, any vendor (the A/B measurement
        // escape). The legacy 32-block set is included so its decode tier is measurable without a
        // rebuild — none of them are on the DEFAULT set (see the `None` arm).
        Some(true) => &[Q4K, Q6K, Q2K, Q3K, Q5K, Q8_0, Q4_0, Q5_0, Q4_1, Q5_1, Iq4Nl],
        // Unified default: the safe intersection of the old per-vendor sets — every dtype here is a
        // measured win on AMD (7900 XTX, Qwen3-14B). Unmeasured on Intel/NVIDIA but the kernels exist
        // for all vendors (unified rows=1 path) and `INFR_MMV_MW=1`/`INFR_MMV_MW=0` override for A/B.
        //
        // Q3_K is OFF: coherence cliff on mixed GGUFs (gpu_seam_matches_cpu_qwen3_q2k — Q2_K files
        // carry Q3_K tensors, flipping them to int8 on a 0.6B model degrades output).
        // Q5_K is OFF: decode -1.4% (noise) on AMD, prefill +45% is banked via mrow_int8_prefill_dtypes.
        // Q8_0 is OFF: -4.2% decode loss, structural (8-bit weight already fits dp4a, no ALU to save).
        // Q4_1 is OFF: ±0% wash on AMD decode.
        None => &[Q2K, Q4K, Q6K, Q4_0, Q5_0, Q5_1, Iq4Nl],
    }
}

/// Per-dtype int8 PREFILL policy for the m>=3 ORDINARY (non-MTP-verify) batched forward — the
/// chunked batched-prefill path and the small-multi-row shapes (spec-decode-shaped verify batches
/// aside, see [`mrow_int8_dtype_ok`]'s `verify` arm). Ordinary prefill has no partner dispatch it
/// must bit-match (see `infr_core::graph::Graph::mtp_verify`'s doc) — it is one path through the
/// model, so it is free to take whichever kernel measures fastest, independent of the m=1 decode
/// policy ([`mmv_int8_decode_dtypes`]) entirely. Every dtype here is a MEASURED win on
/// Qwen3-14B (AMD RDNA3/RADV, `pp4@d4096`, r=3-5): Q6_K +67% (137.9 vs 82.8) at the time this table
/// was written — since improved further to ~183-184 vs 72.8 f32 (+~150%) by the FMT_Q6K
/// word-parallel `wdec` rewrite (see the dedicated note in `mmv_int8_decode_dtypes`'s `None` arm;
/// same numerics, bit-identical, just less unpack ALU per element), IQ4_XS +81% (155.1
/// vs 85.6), Q5_K +45% (188 vs 130), Q3_K +29% (211 vs 161) — Q2_K/Q4_K's wins predate this table
/// (footnote 3). Q4_K/Q6_K/IQ4_XS were already unconditional mrow dtypes before this split (this
/// is not a behavior change for them at prefill); Q2_K/Q3_K/Q5_K are newly unconditional here
/// (previously tied to the decode set, which is what kept Q3_K/Q5_K's prefill win unreachable by
/// default).
///
/// The LEGACY 32-BLOCK set (Q8_0, Q4_0, Q5_0, Q4_1, Q5_1, IQ4_NL) joined this list wholesale — six
/// dtypes that had a dp4a *GEMM* (`native_gemm_mmq_*`) but no dp4a *GEMV*, so every non-k-quant
/// integer file fell to the f32 dequant path at decode AND at small-m. All six MEASURED a large
/// ordinary-prefill win on Qwen3-14B (AMD RDNA3/RADV, `pp4@d4096`, r=3, int8 vs the shipping f32
/// dequant mrow — `INFR_NO_MMV=1`, which for these dtypes is exactly the pre-change behavior):
/// **Q4_0 +66.9%** (139.1 → 232.1), **Q5_0 +64.0%** (130.7 → 214.4), **Q5_1 +42.2%** (145.3 →
/// 206.6), **Q4_1 +32.9%** (152.2 → 202.3), **Q8_0 +28.8%** (127.5 → 164.2), **IQ4_NL +20.7%**
/// (132.2 → 159.5). Four of them ALSO win at decode and are on the decode default; Q8_0 and Q4_1
/// are prefill-ONLY (they lose / wash at m=1 — see [`mmv_int8_decode_dtypes`]'s legacy note for
/// why, it is structural at 8 bits, not a fixable kernel wart). This split is the whole point of
/// having two policies: a dtype that loses decode still banks its prefill win here.
///
/// Q3_K's accuracy was ISOLATED before this default shipped, not assumed: an earlier attempt to
/// flip Q3_K default-on tied decode AND mrow together (the pre-split policy) and broke
/// `gpu_seam_matches_cpu_qwen3_q2k` into a divergent/degenerate generation (unsloth's Q2_K GGUF is
/// MIXED and carries Q3_K tensors, so the flip moved a 0.6B model's layers to int8 where
/// accumulated quant error is worst). This session bisected which side of the split caused it by
/// running the SAME test three ways: (1) PREFILL-int8-only (this set has Q3_K, decode tier
/// untouched — Q3_K stays OFF `mmv_int8_decode_dtypes`) — **coherent, matches the CPU oracle
/// token-for-token**; (2) DECODE-int8-only (`INFR_MMV_MW=1` with Q3_K temporarily pulled out of
/// this prefill set) — **reproduces the exact same divergent generation** as (3); (3) both-on
/// (`INFR_MMV_MW=1` at head, unmodified) — the original historical failure. (1) alone stayed
/// coherent; (2) alone reproduced the failure bit-for-bit identically to (3) — conclusively
/// isolating the cliff to the DECODE tier, not this one. Ordinary prefill's int8 win for Q3_K is
/// therefore safe to ship unconditionally; Q3_K's DECODE default stays off (its own accuracy
/// question is still open — this only clears prefill).
fn mrow_int8_prefill_dtypes(dt: infr_core::DType) -> bool {
    use infr_core::DType::{Iq4Nl, Iq4Xs, Q2K, Q3K, Q4K, Q4_0, Q4_1, Q5K, Q5_0, Q5_1, Q6K, Q8_0};
    matches!(
        dt,
        Q2K | Q3K | Q4K | Q5K | Q6K | Iq4Xs | Q8_0 | Q4_0 | Q5_0 | Q4_1 | Q5_1 | Iq4Nl
    )
}

/// Does `dt` take the int8 dp4a `mrow` kernel for the m>=3 batched forward? `verify` selects which
/// of the two independent policies gates it (see `infr_core::graph::Graph::mtp_verify`'s doc for
/// why they must differ):
///
/// - **MTP-verify batch** (`verify == true`): gated to the EXACT SAME dtype set as the m=1 decode
///   tier ([`mmv_int8_decode_dtypes`]) on this (vendor, env) combination — no exceptions, not even
///   for Q4_K/Q6_K/IQ4_XS. Verify's greedy output must bit-match plain decode at the same position
///   (`mtp_spec_matches_target_only_greedy`'s contract); letting a dtype run int8 in verify while
///   decode stays f32-exact (or vice versa) is the exact Q5_K token-divergence bug class. On AMD,
///   where decode dispatches the unified `linear_mmv_mrow(rows=1)` kernel (see
///   [`unified_mmv_row1`]), a dtype on the decode set is bit-identical in both streams by
///   construction, not merely both-int8.
/// - **Ordinary prefill** (`verify == false`): [`mrow_int8_prefill_dtypes`] — every measured win,
///   independent of the decode policy.
///
/// This closes a real bug: Q6_K and IQ4_XS used to take this kernel UNCONDITIONALLY at m>=3
/// (predating this split), while their m=1 decode is f32-exact and neither is ever on the decode
/// set (IQ4_XS has no int8 decode arm at all; Q6_K's decode is off on AMD by default) — so an MTP
/// verify batch on either dtype could flip a greedy token vs plain decode. After this split, their
/// MTP verify lands on the f32-exact path (matching decode) while ordinary prefill keeps the win.
fn mrow_int8_dtype_ok(
    caps: &infr_core::backend::Capabilities,
    vk: &infr_core::config::VulkanCfg,
    dt: infr_core::DType,
    verify: bool,
) -> bool {
    if verify {
        mmv_int8_decode_dtypes(caps, vk).contains(&dt)
    } else {
        mrow_int8_prefill_dtypes(dt)
    }
}

/// Does the m=1 decode tier dispatch through the unified `linear_mmv_mrow(rows=1)` kernel (the
/// SAME reduction as the m>=3 verify tier — bit-identical at the same position by construction,
/// see `native_mmv_mrow.comp`'s header) instead of the legacy `native_mmv_mw.comp`
/// (warp-per-row, reassociation-tolerant only)?
///
/// YES, always, for every vendor and every dtype `mmv_mw_choice` gates on — the unified kernel
/// exists for all dtypes, is bit-identical by construction (proved by `mmv_row1_bit_identical`),
/// and paths not covered by `mmv_mw_choice` (Intel's legacy set, any future dtype) fall through
/// the same gate. The Intel-specific carve-out was removed as part of the capability-first
/// architecture: new hardware needs no vendor quirk here.
fn unified_mmv_row1(_caps: &infr_core::backend::Capabilities) -> bool {
    true
}

fn mmv_mw_choice(
    caps: &infr_core::backend::Capabilities,
    vk: &infr_core::config::VulkanCfg,
    dt: infr_core::DType,
    in_f: usize,
    out_f: usize,
) -> Option<u32> {
    if !caps.i8_dot || !mmv_int8_decode_dtypes(caps, vk).contains(&dt) {
        return None;
    }
    // Skip tiny GEMVs where per-dispatch overhead dominates (k/v projections); the projection band
    // and lm_head are all well above 2M elements.
    if !in_f.is_multiple_of(32) || in_f * out_f < (2usize << 20) {
        return None;
    }
    // Default WARPS is per-dtype: Q4_K defaults to 1 (llama.cpp's rm_kq_int=1 shape — one output
    // row per workgroup / single-subgroup reduce, the dispatch-shape sweep winner on AMD). Every
    // other dtype defaults to 8. Q4_K warps=1 beats {4,8} and beats the f32 path. Overrideable
    // via INFR_MMV_MW_WARPS for A/B.
    let default_warps = if dt == infr_core::DType::Q4K {
        1u32
    } else {
        8u32
    };
    let warps = vk
        .mmv_mw_warps
        .and_then(|w| u32::try_from(w).ok())
        .unwrap_or(default_warps);
    // SPIR-V existence gate: unified rows=1 path probes the mrow variant for every dtype. All
    // dtypes in the unified decode set have an mrow build; the mmv_mw route is retired (Intel
    // specific dtypes Q3_K/Q5_K that had only mmv_mw builds are no longer in the default set,
    // and INFR_MMV_MW=1 still force-opt-in them — they just fall through here as None since
    // they lack mrow builds).
    let have_spv =
        crate::gemm::native_mmv_mrow_variant_kernel_name(dt, in_f < 2048, true, false).is_some();
    have_spv.then_some(warps)
}

/// Get-or-alloc the pool buffer for (tag, bytes); returns the map key so callers can hold several
/// pool buffers at once via immutable indexing (`pool[&k].as_ref()`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn pooled(
    pool: &mut ScratchPool,
    be_: &VulkanBackend,
    tag: &'static str,
    bytes: usize,
) -> Result<(&'static str, usize)> {
    let key = (tag, bytes.max(4));
    if let std::collections::hash_map::Entry::Vacant(e) = pool.entry(key) {
        // alloc_uninit: every pooled buffer is fully written before read within each op.
        e.insert(be_.alloc_uninit(key.1, BufferUsage::Activations)?);
    }
    Ok(key)
}

/// Small-m MoE scratch handle: a pooled `(tag, bytes)` key (the default — rides the per-execute
/// `pool`, so ALL MoE layers share one buffer per tag and the alloc happens at most once), or an
/// owned per-layer zero-filled `Backend::alloc` under the `INFR_NO_MOE_SM_POOL=1` escape. Resolve
/// with [`SmB::get`] at each use site (immutable pool indexing, same discipline as the batched
/// arm's `pool[&k]`); the owned variant is drained into `transient` via [`SmB::into_transient`] so
/// it outlives `rec.finish()` (pooled buffers are moved into `transient` at the end of the loop).
enum SmB {
    Pool((&'static str, usize)),
    Own(Box<dyn Buffer>),
}

impl SmB {
    fn get<'a>(&'a self, pool: &'a ScratchPool) -> &'a dyn Buffer {
        match self {
            SmB::Pool(k) => pool[k].as_ref(),
            SmB::Own(b) => b.as_ref(),
        }
    }

    /// Move an owned (escape-path) buffer into `transient` so the recording can't reference a freed
    /// buffer. Pooled buffers are owned by `pool` (drained into `transient` at the end of the loop),
    /// so `Pool` is a no-op here.
    fn into_transient(self, transient: &mut Vec<Box<dyn Buffer>>) {
        if let SmB::Own(b) = self {
            transient.push(b);
        }
    }
}

/// Allocate one small-m MoE scratch buffer of `elems` f32/u32 elements: pooled uninit by default
/// (every buffer is fully written before it is read within the op — see the small-m arm's scratch
/// doc for the per-buffer argument), or a fresh calloc-contract `Backend::alloc` when `no_pool`
/// (the `INFR_NO_MOE_SM_POOL=1` A/B escape — each device-local zero-fill costs a one-shot submit +
/// queue_wait_idle, the exact per-layer overhead pooling kills).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn sm_buf(
    pool: &mut ScratchPool,
    be_: &VulkanBackend,
    no_pool: bool,
    tag: &'static str,
    elems: usize,
) -> Result<SmB> {
    if no_pool {
        Ok(SmB::Own(
            be_.alloc((elems * 4).max(4), BufferUsage::Activations)?,
        ))
    } else {
        Ok(SmB::Pool(pooled(pool, be_, tag, elems * 4)?))
    }
}

/// Per-execute Dynamic-attention shared state: ONE attn_live prologue + ONE pm/pl/pacc split
/// scratch set per distinct (nh, hd, chunk, n_chunks, window) attention shape (see `lower_op`'s
/// `dyn_args`). Uniform models (qwen3) have exactly one; gemma alternates SWA/global layers (and
/// gemma4-12b alternates hd 256/512), so the contexts live in a small Vec looked up by key —
/// same-key layers share one prologue dispatch and one scratch set.
struct DynAttnCtx {
    nh: usize,
    chunk: usize,
    n_chunks: usize,
    hd: usize,
    window: usize,
    args: Box<dyn Buffer>,
    pm: Box<dyn Buffer>,
    pl: Box<dyn Buffer>,
    pacc: Box<dyn Buffer>,
}

/// Weight-dtype predicate for the shared `Linear→Add` residual fusion (see
/// [`infr_core::fusion::plan_fusions`]): the dense native-block formats plus f16 — exactly the
/// fused-residual GEMV (`linear_add_native` / `linear_add`) coverage this adapter records.
fn linear_add_weight_ok(dt: infr_core::DType) -> bool {
    native_dense_supported(dt) || matches!(dt, infr_core::DType::F16)
}
static LINEAR_ADD_WEIGHT_OK: fn(infr_core::DType) -> bool = linear_add_weight_ok;

/// This adapter's [`FusionCfg`](infr_core::fusion::FusionCfg): the `Linear→Add` residual fusion
/// (`INFR_NO_FUSE_ADD` hatch) and the `Rope/QkNormRope→WriteKv` KV-write fusion. `plan_fusions`
/// returns the same `(fused, skip)` the two former private peepholes did.
fn fusion_cfg(be_: &VulkanBackend) -> infr_core::fusion::FusionCfg<'static> {
    infr_core::fusion::FusionCfg {
        linear_add: Some(infr_core::fusion::LinearAddCfg {
            weight_ok: &LINEAR_ADD_WEIGHT_OK,
            // `INFR_NO_FUSE_ADD` (config `kernels.vulkan.fuse_add`, positive polarity): PRESENCE
            // of the env key — including `INFR_NO_FUSE_ADD=0` — turns the fold off.
            enabled: be_.cfg().kernels.vulkan.fuse_add,
        }),
        rmsnorm_linear: None,
        // F1c's `MoeFfn → Add` fold is un-planned on every live backend: this adapter's MoE lowering accumulates
        // the experts through its own `moe_accumulate`, which has no residual epilogue, so the
        // pattern stays un-planned here and Vulkan dispatches the standalone `Add` exactly as
        // before.
        moe_add: None,
        kv_write: true,
    }
}

/// How pos-dependent ops (`QkNormRope`, `WriteKv`, `Attention`) are lowered:
/// - `Static`: the classic per-execute recording — the pos is a push constant (read from the
///   `positions` tensor up front into `rope_pos`), the graph's baked `pos`/`kv_len` drive the rest.
/// - `Dynamic`: the record-once path — the `_dyn` kernels read pos/kv_len from the `params` SSBO, so
///   the SAME recording replays across tokens. Only reachable for a [`decode_eligible`] graph.
enum RopeMode<'a> {
    Static(&'a HashMap<u32, usize>),
    Dynamic(&'a dyn Buffer),
}

/// `attn_combine.comp` indexes `shared float wexp[ATTN_MAX_CHUNKS]` by chunk, so a split-K
/// dispatch with a larger chunk COUNT is an OOB shared-memory write. This is the device fact the
/// shared [`infr_core::tier::cap_chunk_count`] / [`infr_core::tier::baked_chunk`] policy is
/// parameterized by.
const ATTN_MAX_CHUNKS: usize = 1024;

/// Flash-decoding split-K chunk policy: ~32 chunks/head, each 64..512 keys, floor-rounded (the
/// measured Vulkan `attn_partial` shape — see [`infr_core::tier::ChunkRounding`] for why the
/// rounding is config and not a shared constant).
const ATTN_SPLIT: infr_core::tier::AttnSplitCfg = infr_core::tier::AttnSplitCfg {
    target_chunks: 32,
    min_chunk: 64,
    max_chunk: 512,
    rounding: infr_core::tier::ChunkRounding::Down,
};

// Probe the wider coupled-Q8 hd256 decode kernel independently from every f16/prefill tier.
// The specialized shader owns a 1024-score slab; all other paths retain ATTN_SPLIT's 512 cap.
const Q8_HD256_DECODE_SPLIT: infr_core::tier::AttnSplitCfg = infr_core::tier::AttnSplitCfg {
    target_chunks: 32,
    min_chunk: 64,
    max_chunk: 1024,
    rounding: infr_core::tier::ChunkRounding::Down,
};

/// Canvas-mask (`AttnMask::Canvas`) chunk DIVISOR: the canvas span is cut into ~this many chunks
/// rather than sized adaptively (the fixed `lo` override makes the span short and known). The
/// floor of 1 keeps the `kv_len.div_ceil(n)` below from dividing by zero.
const CANVAS_CHUNK_N: infr_core::tier::EnvRows = infr_core::tier::EnvRows {
    env: "INFR_CANVAS_CHUNK_N",
    default: 3,
    min: 1,
    max: usize::MAX,
};

/// Cap the static split-K attention chunk COUNT so `n_chunks = span.div_ceil(chunk) <= 1024` (see
/// [`ATTN_MAX_CHUNKS`]). Raises the chunk floor to `span.div_ceil(1024)`; only bites when
/// `span > 512*1024` (≈524k keys, reachable under `INFR_KV_OVERFLOW` huge ctx). Below that it
/// returns `chunk` unchanged, so the recorded dispatch is byte-identical for every realistic span.
fn split_k_chunk_count_cap(span: usize, chunk: usize) -> usize {
    infr_core::tier::cap_chunk_count(span, chunk, ATTN_MAX_CHUNKS)
}

/// Narrow-n deep-k split-K count for the coopmat-warp prefill GEMM (see the resident native-dense
/// arm's split-K comment in `lower_op`): a plain tile's grid underfills the device for narrow-n
/// (`n=1024` → 64 wgs on a 96-wg part), so split k across enough extra workgroups to fill, with a
/// fixed-order reduce. Returns 1 (direct tile, no partials round-trip) for wide/filled shapes.
/// Shared by the resident native-quant arm, the F16 split-K arm, and the streamed twin so their
/// tile decisions can't drift.
fn split_k_plan(m: usize, in_f: usize, out_f: usize) -> usize {
    let narrow_grid = m.div_ceil(64) * (out_f / 128).max(1);
    if out_f.is_multiple_of(128) && in_f >= 1024 && narrow_grid < 128 {
        (256 / narrow_grid).next_power_of_two().clamp(1, 8)
    } else {
        1
    }
}

/// The coopmat-warp prefill GEMM tile selection shared by the resident native-dense `Op::Linear`
/// arm and its arena-addressed streamed twin (`streamed_prefill_gemm`): the A_GLOBAL f16-A cast,
/// the narrow-n split-K (`split_k_plan`), or the direct tile — reading the weight from `w_addr`
/// (the resident weight's BDA, or the streamed pool-arena base). Callers pass the padded `out`
/// (see `with_padded_dst`). Extracted so the two paths can't drift ("must track exactly").
#[allow(clippy::too_many_arguments)]
fn native_warp_gemm(
    be_: &VulkanBackend,
    dt: infr_core::DType,
    w_addr: u64,
    w_off: usize,
    xb: &dyn Buffer,
    out: &dyn Buffer,
    m: usize,
    in_f: usize,
    out_f: usize,
    rec: &Recorder<'_>,
    pool: &mut ScratchPool,
) -> Result<()> {
    // A_GLOBAL: convert A to f16 ONCE (a cheap cast-copy — the warp kernels rounded A to f16 in
    // their staging loop anyway, so numerics are identical) and let the warptiles coopMatLoad it
    // straight from global. Dropping the As stage shrinks LDS to Bs-only → higher occupancy.
    // INFR_GEMM_DIRECT_A=1: skip the store_f16 pass — the _direct kernel variants read f32
    // activations directly and cast to f16 in As staging (saves one dispatch + barrier per GEMM
    // at the cost of reintroducing As shared memory).
    let direct_a = be_.cfg().kernels.vulkan.gemm_direct_a;
    let use_direct = direct_a
        && out_f.is_multiple_of(128)
        && in_f.is_multiple_of(32)
        && crate::gemm::native_gemm_warp_n128_direct_kernel_name(dt).is_some();
    let use_ag = !use_direct
        && out_f.is_multiple_of(128)
        && in_f.is_multiple_of(32)
        && crate::gemm::native_gemm_warp_ag_kernel_name(dt).is_some()
        && be_.cfg().kernels.vulkan.gemm_warp;
    let a16 = if use_ag {
        let mpad = m.div_ceil(64) * 64;
        let key = pooled(pool, be_, "lin_a16", mpad * in_f * 2)?;
        rec.store_f16(xb, pool[&key].as_ref(), m * in_f, 0);
        Some(key)
    } else {
        None
    };
    let splits = split_k_plan(m, in_f, out_f);
    if splits > 1 && crate::gemm::native_gemm_warp_sk_kernel_name(dt).is_some() {
        let mpad = m.div_ceil(64) * 64;
        let pk = pooled(pool, be_, "splitk_part", splits * mpad * out_f * 4)?;
        let a: &dyn Buffer = match &a16 {
            Some(k16) => pool[k16].as_ref(),
            None => xb,
        };
        rec.matmul_native_splitk(
            dt,
            a,
            w_addr,
            w_off,
            pool[&pk].as_ref(),
            out,
            m,
            in_f,
            out_f,
            splits,
            a16.is_some(),
        );
    } else if use_direct {
        rec.matmul_native_f32a(dt, xb, w_addr, w_off, out, m, in_f, out_f);
    } else if let Some(k16) = &a16 {
        rec.matmul_native_f16a(dt, pool[k16].as_ref(), w_addr, w_off, out, m, in_f, out_f);
    } else {
        rec.matmul_native_off(dt, xb, w_addr, w_off, out, m, in_f, out_f);
    }
    Ok(())
}

/// The padded-dst dance shared by every tiled prefill GEMM tier (`is_gemm`, the non-coopmat
/// `nc_mmq`/`nc_fma` tier, and the streamed twin): the GEMM writes `ceil(m/64)*64` rows, so an
/// Internal (row-padded) `dst` is written direct, while a non-Internal `dst` (e.g. the lm_head
/// `logits` Output) gets a padded temp the `record` closure fills, then a copy of the `m` real
/// rows back into `y`. Extracted so the three tiers stay byte-identical.
fn with_padded_dst<F>(
    be_: &VulkanBackend,
    graph: &Graph,
    rec: &Recorder<'_>,
    dst: &TensorId,
    y: &dyn Buffer,
    m: usize,
    out_f: usize,
    transient: &mut Vec<Box<dyn Buffer>>,
    record: F,
) -> Result<()>
where
    F: FnOnce(&dyn Buffer) -> Result<()>,
{
    let dst_internal = matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal);
    let eb = graph.desc(*dst).dtype.dense_bytes(1).unwrap_or(4);
    // alloc_uninit: the GEMM writes all mpad rows before the copy reads m of them.
    let tmp = if dst_internal {
        None
    } else {
        let mpad = m.div_ceil(64) * 64;
        Some(be_.alloc_uninit((mpad * out_f * eb).max(4), BufferUsage::Activations)?)
    };
    let out: &dyn Buffer = match &tmp {
        Some(t) => t.as_ref(),
        None => y,
    };
    record(out)?;
    if let Some(t) = tmp {
        rec.copy(t.as_ref(), 0, y, 0, m * out_f * eb);
        transient.push(t);
    }
    Ok(())
}

/// Capacity of `mla.comp`'s `shared float sq[]` — the absorbed+roped query, `kv_lora_rank +
/// qk_rope_dim` wide.
///
/// MIRRORS `#define MLA_MAX_KEY_LEN` in `shaders/mla.comp`. The shader cannot reject a dispatch
/// that does not fit, so the `Op::Mla` arm of [`lower_op`] checks it here; `mla_shader_bounds_match_host`
/// parses the shader source and fails if the two ever drift.
pub(crate) const MLA_MAX_KEY_LEN: u32 = 576;

/// Capacity of `mla.comp`'s `shared float sv[]` — the V accumulator, `kv_lora_rank` wide.
/// MIRRORS `#define MLA_MAX_KV_LORA_RANK` in `shaders/mla.comp`; see [`MLA_MAX_KEY_LEN`].
pub(crate) const MLA_MAX_KV_LORA_RANK: u32 = 512;

/// Lower ONE graph op into the recorder. Shared by the static (`execute_static`) and record-once
/// (`record_decode_replay`) paths — only the three pos-dependent ops branch on `mode`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn lower_op(
    be_: &VulkanBackend,
    graph: &Graph,
    op_idx: usize,
    op: &Op,
    rec: &Recorder<'_>,
    scratch: &[Option<Box<dyn Buffer>>],
    bindings: &Bindings,
    fused_kv_write: &HashMap<usize, (TensorId, usize)>,
    fused_add: &HashMap<usize, (TensorId, TensorId)>,
    mode: &RopeMode,
    transient: &mut Vec<Box<dyn Buffer>>,
    // Per-execute Dynamic-attention contexts: the prologue args AND the pm/pl/pacc split scratch,
    // shared by every attention op of the same shape (layers are serialized by dataflow, so one
    // set per shape suffices — the bespoke path shares its split scratch the same way; per-layer
    // sets cost 28x the VRAM and added run-to-run placement variance). A uniform model (qwen3)
    // holds one entry; gemma's SWA/global alternation (and gemma4-12b's hd 256/512) holds a few.
    dyn_args: &mut Vec<DynAttnCtx>,
    pool: &mut ScratchPool,
    // Multi-row mmv quantized-activation memo: `Some((x, m, in_f))` when the pooled `mmvr_*`
    // scratch already holds `quant_q8(x)` for that shape. Consecutive same-x Linears (fused-QKV
    // offset GEMVs, gate+up) then skip the requant — which not only drops ~40% of the quant
    // dispatches but, more importantly, removes the WAR hazard (quant rewrites qa → barrier
    // against the previous mmv's read) that would otherwise serialize the sibling GEMVs.
    // `take()`n at the top of every call: ONLY the mmv-mrow branch restores it, so any other op
    // in between (which may rewrite `x` — e.g. next layer's RmsNorm into the same scratch id)
    // invalidates the memo by construction.
    mmv_memo: &mut Option<(TensorId, usize, usize)>,
    // Dense layer streaming (see `pager::DensePagerSession`): `Some(arena_addr)` when this op is an
    // `Op::Linear` whose weight is a streamed block — `arena_addr` is the resident slot's arena base
    // BYTE address. The pool arena is a `bufferDeviceAddress` buffer read purely by 64-bit pointer
    // (NOT a bound SSBO), so the ~4 GiB `maxStorageBufferRange` cap is gone entirely (no descriptor
    // binds it). The op is lowered by an m-split that MIRRORS the resident selection, just
    // arena-addressed: a genuine prefill chunk (`streamed_gemm_applies`) routes through the SAME
    // coopmat-warp GEMM tile the resident path would pick (`streamed_prefill_gemm` → the -DSTREAMED
    // twins of native_gemm/native_gemm_warp), while decode/small-m keep the single streamed GEMV
    // (`linear_native_at`) — the arena-addressed kernels are the ONLY ones a >4 GiB pool ever
    // dispatches, which is what makes dropping the caps safe. `w_off` (a fused-QKV slice offset,
    // usually 0) rides on top as a within-slot element offset in both. The ring→arena copy hazard is
    // ordered explicitly by `arena_stream_barrier` at the staging site (`stage_dense_linear`) — the
    // SAME slot/RAW barrier already covers the prefill dispatch. Every other op — and every
    // non-streamed model — passes `None` (zero change). Only the `Op::Linear` arm reads it; the
    // seam's placement guarantees a streamed weight's dtype rides the offset-capable native paths
    // (`native_dense_supported`) and never the fused-residual peephole (filtered by `execute_static`
    // before the loop).
    wsub: Option<u64>,
) -> Result<()> {
    let memo_prev = mmv_memo.take();
    let r = |id: TensorId| resolve(scratch, bindings, id);
    match op {
        Op::RmsNorm {
            x,
            weight,
            dst,
            rows,
            dim,
            eps,
        } => {
            rec.rmsnorm(
                r(*x)?,
                r(*weight)?,
                r(*dst)?,
                *rows as usize,
                *dim as usize,
                *eps,
            );
        }
        Op::RmsNormAdd {
            x,
            weight,
            dst,
            rows,
            dim,
            eps,
        } => {
            rec.rmsnorm_add(
                r(*x)?,
                r(*weight)?,
                r(*dst)?,
                *rows as usize,
                *dim as usize,
                *eps,
            );
        }
        // Mean-centred LayerNorm (deepseek32's `indexer_k_norm`) — the family's one non-RMS norm.
        Op::LayerNorm {
            x,
            weight,
            bias,
            dst,
            rows,
            dim,
            eps,
        } => {
            rec.layernorm(
                r(*x)?,
                r(*weight)?,
                r(*bias)?,
                r(*dst)?,
                *rows as usize,
                *dim as usize,
                *eps,
            );
        }
        // Row-wise softmax over `dim` columns (diffusion-gemma's in-graph self-conditioning — see
        // docs/diffusion-gemma.md's Phase-B and the reference's `dg_canvas_embed`). `scale_buf`
        // (Some only on the DiffusionGemma denoise SC path — see its doc in `infr_core::graph`)
        // reads the scale from a device buffer instead of the push-constant `scale` field, so this
        // cached/replayed plan can vary the softmax temperature every step without rebuilding.
        Op::Softmax {
            x,
            dst,
            rows,
            dim,
            scale,
            scale_buf,
        } => match scale_buf {
            Some(sb) => rec.softmax_dyn(r(*x)?, r(*sb)?, r(*dst)?, *rows as usize, *dim as usize),
            None => rec.softmax(r(*x)?, r(*dst)?, *rows as usize, *dim as usize, *scale),
        },
        // `dst = x · Wᵀ` — dispatch by weight dtype AND row count. Decode (m=1) uses the native
        // GEMV (or f16 GEMV). Prefill (m>1) with a native-quant weight uses the TILED coopmat GEMM
        // `matmul_native` (decode each weight element ONCE, reuse across the 64-row tile) instead
        // of the GEMV (which re-reads every weight row per output row) — the prefill perf win.
        Op::Linear {
            x,
            weight,
            dst,
            m,
            in_f,
            out_f,
            w_off,
        } => {
            let (m, in_f, out_f, w_off) = (
                *m as usize,
                *in_f as usize,
                *out_f as usize,
                *w_off as usize,
            );
            // INFR_PROF_OPS sub-attribution (no-op otherwise): the vocab-sized GEMMs (lm_head
            // logits, out_f = vocab; DiffusionGemma's SC soft-embedding, in_f = vocab) are 10-100x
            // the FLOPs of a per-layer projection but land in the same "matmul_proj" bucket,
            // which made the bucket un-actionable during the DG slice-7 comparative attribution
            // (llama.cpp's perf logger itemizes per shape; ours didn't). 65536 cleanly separates
            // every real vocab (gemma 262144, qwen 151936) from every hidden/FFN width.
            if out_f >= 65536 {
                rec.label_next("lin_vocab_out");
            } else if in_f >= 65536 {
                rec.label_next("lin_vocab_in");
            }
            let (w, xb, y) = (r(*weight)?, r(*x)?, r(*dst)?);
            let dt = graph.desc(*weight).dtype;
            // Dense layer streaming: the weight lives in a `bufferDeviceAddress` arena pool (see
            // the `wsub` param doc). The op reads the arena by 64-bit address (`arena_addr`), with
            // the op's own `w_off` (a fused-qkv slice offset, usually 0) riding on top as a
            // within-slot element offset. A genuine prefill chunk routes to the SAME coopmat-warp
            // GEMM tile the resident path would pick (arena-addressed) — the prefill perf win;
            // decode/small-m keep the single streamed GEMV. Both dispatch ONLY arena-addressed
            // kernels, never an unconverted SSBO one, which is what lets the seam drop the pool caps.
            if let Some(arena_addr) = wsub {
                // Placement guarantees an offset-capable native dtype (the f16/f32 fallback arms
                // take no weight offset) — a violation here would read the wrong arena bytes.
                if !native_dense_supported(dt) {
                    return Err(be("vulkan adapter: streamed Linear weight of a dtype \
                                   without offset-capable kernels"));
                }
                if streamed_gemm_applies(be_, dt, m, in_f, out_f) {
                    // The tiled prefill GEMM can't be output-row chunked (see the resident-GEMM
                    // guard above) — a breaching streamed lm_head at m>16 fails loudly here.
                    if crate::recorder::bda_weight_breaches(
                        dt,
                        in_f,
                        out_f,
                        &be_.cfg().kernels.vulkan,
                    ) {
                        return Err(be(format!(
                            "vulkan adapter: streamed output-projection GEMM (m={m}, in_f={in_f}, \
                             out_f={out_f}, {dt:?}) reads a >= 2^32-element weight — chunked \
                             dispatch covers only m=1 decode and the per-row GEMV (issue #77)."
                        )));
                    }
                    streamed_prefill_gemm(
                        be_, graph, dt, arena_addr, w_off, xb, y, dst, m, in_f, out_f, rec, pool,
                        transient,
                    )?;
                } else {
                    // GEMV path — chunk-covered inside `linear_native_at` for a breaching weight.
                    rec.linear_native_at(dt, arena_addr, w_off, xb, y, m, in_f, out_f);
                }
                return Ok(());
            }
            // `w_off` on a NON-native (float) weight. The native quant/bf16 kernels take the offset
            // as a push field; the float fallbacks have no such field and index the weight from
            // element 0 — so an unhandled `w_off` there reads rows 0..out_f and is silently wrong.
            //
            // F32 is handled: every float GEMV in this backend already addresses its weight by a
            // 64-bit `bufferDeviceAddress` (`Recorder::linear_f32` is `linear_f32_at` at
            // `w.device_addr()`), so a ROW-ALIGNED slice is just a shifted base pointer, which is
            // what the `w_off != 0` arm at the F32 dispatch below passes. DeepSeek V4's grouped
            // output projection is the caller (`w_off = g * o_lora_rank * in_f`); an f32 `wo_a` is
            // what the synthetic V4 fixture writes, and what the CPU-vs-Vulkan parity test runs.
            //
            // F16 now follows the same shifted-address design. Nomic-BERT's fused QKV projection
            // exercises both its m=1 GEMV and m>1 coopmat GEMM paths; the parity test below pins
            // both against a host reference.
            // F16 kernels do not carry an element-offset field. Convert a row-aligned offset to a
            // shifted BDA once and feed the existing arena-addressed entry points. Four-byte
            // alignment covers the packed-u32 readers used by coopmat and the no-f16 fallback.
            let f16_w_addr = if w_off != 0 && dt == infr_core::DType::F16 {
                if w_off % in_f != 0 {
                    return Err(be(format!(
                        "vulkan adapter: F16 Linear w_off {w_off} is not a whole multiple of \
                         in_f {in_f}"
                    )));
                }
                let byte_off = w_off
                    .checked_mul(2)
                    .ok_or_else(|| be("vulkan adapter: F16 Linear byte offset overflow"))?;
                if byte_off % 4 != 0 {
                    return Err(be(format!(
                        "vulkan adapter: F16 Linear byte offset {byte_off} is not u32 aligned"
                    )));
                }
                let base = w.device_addr().ok_or_else(|| {
                    be("vulkan adapter: F16 Linear with w_off needs a BDA weight address")
                })?;
                Some(
                    base.checked_add(byte_off as u64)
                        .ok_or_else(|| be("vulkan adapter: F16 Linear device address overflow"))?,
                )
            } else {
                None
            };
            if w_off != 0
                && !native_dense_supported(dt)
                && dt != infr_core::DType::F32
                && dt != infr_core::DType::F16
            {
                return Err(be(format!(
                    "vulkan adapter: Linear w_off on a {dt:?} weight — only the native (quant / \
                     bf16) kernels and the f16/f32 paths take a weight offset"
                )));
            }
            // Fused Linear+Add (decode residual): one GEMV with the residual added in-kernel —
            // see linear_add_peephole. `y` (the Linear's scratch dst) is never written.
            if let Some((residual, final_dst)) = fused_add.get(&op_idx) {
                // Streamed weights are filtered OUT of `fused_add` before the op loop (the fused
                // kernels bake a zero weight offset); reaching here with a substitution means the
                // filter was bypassed — fail loudly, slot 0's `elem_base == 0` would otherwise
                // slip through the w_off check below.
                if wsub.is_some() {
                    return Err(be(
                        "vulkan adapter: streamed Linear reached the fused-add path",
                    ));
                }
                if w_off != 0 {
                    return Err(be("vulkan adapter: Linear w_off with fused residual"));
                }
                let (rr, yf) = (r(*residual)?, r(*final_dst)?);
                if native_dense_supported(dt) {
                    // Multi-warp dp4a decode GEMV (wave32-native GPUs) takes precedence — see
                    // `mmv_mw_choice`. AMD returns None here and falls to the scalar/old-mmv path.
                    if let Some(warps) =
                        mmv_mw_choice(be_.caps(), &be_.cfg().kernels.vulkan, dt, in_f, out_f)
                    {
                        // FUSE_QUANT opt-in: inline activation quantization, skip the
                        // separate quant_q8 dispatch.
                        if be_.cfg().kernels.vulkan.mmv_fuse_quant {
                            rec.linear_mmv_mrow_fuseq(dt, w, 0, xb, Some(rr), yf, 1, in_f, out_f);
                            return Ok(());
                        }
                        let nblk = in_f / 32;
                        let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                        let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                        let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            1,
                            in_f,
                        );
                        if unified_mmv_row1(be_.caps()) {
                            // Decode (m=1) dispatches the SAME kernel/reduction as the m>=3 verify
                            // tier (`linear_mmv_mrow`, rows=1) — see `unified_mmv_row1`'s doc. This
                            // is the fix for the mmv_mw/mrow reassociation gap (README footnote 3):
                            // bit-identical to `mmv_mw_choice`'s m>=3 twin at the same position.
                            rec.linear_mmv_mrow(
                                dt,
                                w,
                                0,
                                pool[&qa].as_ref(),
                                pool[&dact].as_ref(),
                                pool[&sact].as_ref(),
                                Some(rr),
                                yf,
                                1,
                                in_f,
                                out_f,
                            );
                        } else {
                            rec.linear_mmv_mw(
                                dt,
                                warps,
                                w,
                                0,
                                pool[&qa].as_ref(),
                                pool[&dact].as_ref(),
                                pool[&sact].as_ref(),
                                Some(rr),
                                yf,
                                in_f,
                                out_f,
                            );
                        }
                    } else if mmv_decode_enabled(&be_.cfg().kernels.vulkan)
                        && be_.caps().i8_dot
                        && in_f % 32 == 0
                        && in_f * out_f >= MMV_MIN_ELEMS
                        && crate::gemm::native_mmv_kernel_name(dt, true).is_some()
                    {
                        let nblk = in_f / 32;
                        let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                        let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                        let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            1,
                            in_f,
                        );
                        rec.linear_add_mmv(
                            dt,
                            w,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            rr,
                            yf,
                            in_f,
                            out_f,
                        );
                    } else {
                        rec.linear_add_native(dt, w, xb, rr, yf, m, in_f, out_f);
                    }
                } else {
                    rec.linear_add(w, xb, rr, yf, m, in_f, out_f);
                }
                return Ok(());
            }
            // Prefill (m>1): a TILED GEMM writes ceil(m/64)*64 rows DIRECTLY into `y` (Internal
            // buffers are row-padded to 64 up front, so no temp/copy). Needs n%64==0, k%32==0.
            //  • Q4_K → mmq (dp4a int8): quantize activations once, integer matmul on the raw
            //    blocks (no per-GEMM weight dequant) — the u4 prefill default.
            //  • other native quants → coopmat `matmul_native` (in-shader dequant).
            //  • f16 (float weights are uploaded as f16) → f16 coopmat `matmul_proj`.
            // Decode (m=1) and non-tileable shapes fall through to the GEMV.
            // Small multi-row batches (m = 2..8: spec-decode verify rows, a short chat-turn
            // suffix prefill) take the multi-row GEMV first — the single-M-tile coopmat GEMM
            // launches only n/64 workgroups (underfills the GPU: measured 51-182 GB/s effective
            // weight stream vs the GEMV class's 292-651 on a 7900 XTX), and the plain GEMV
            // re-streams the weight per row. 7900 XTX, 8B Q4_K shapes: m=2 is 5.6-8.3x the GEMM
            // route, m=4 1.4-4.1x, m=8 1.7-2.6x — EXCEPT very wide n at m>=5 (gate+up n=24576:
            // 384 tiles fill the GPU and the mrow's per-thread m*32-FMA inner loop turns
            // ALU-bound), so m=5..8 gates on out_f <= 8192 (<=128 tiles = the GEMM-underfill
            // regime). Writes exactly m rows (no padded-dst dance). Formats without an mrow
            // build fall through to the GEMM; INFR_NO_MROW forces the old route (A/B).
            // The int8 dp4a mrow's own gates (see the block comment below) — hoisted so the
            // 9..=16 tier can require them UP FRONT: unlike m<=8 there is no dequant-mrow
            // fallback above 8 rows, so entering the tier is only legal when the int8 kernel is
            // actually dispatchable.
            let mmv_gate = be_.caps().i8_dot
                && in_f * out_f >= 8 << 20
                && out_f < 65536
                && crate::gemm::native_mmv_mrow_kernel_name(dt).is_some()
                && mrow_int8_dtype_ok(be_.caps(), &be_.cfg().kernels.vulkan, dt, graph.mtp_verify)
                && be_.cfg().kernels.vulkan.mmv;
            // rows 9..=16 int8 mrow tier (INFR_NO_MROW16 A/B escape): the MTP spec-verify batch
            // once its rollback window carries a few committed rows on top of the n_max drafts.
            // These shapes previously fell through to the split-K coopmat tile, which streams the
            // same weight at 2-4x the per-row cost at this m (9B mtp128 measured: q4k down-proj
            // m=11 sk_ag 148us vs mmvr m=7 67.5us; q6k 194us). Same out_f <= 8192 wide-n cutoff
            // as m=5..8 (the GEMM tiles win on gate_up-wide shapes), same formats as the m<=8
            // int8 tier (others keep the GEMM route — no dequant mrow above 8 rows).
            // The two bands' SHAPE arithmetic lives in `MROW_BANDS` (shared policy); the
            // capability half — kernel builds, dtype sets, caps, env hatches — stays here.
            let mrow_tier = infr_core::tier::linear_tier(m, out_f, &MROW_BANDS);
            let in_band = |b: usize| mrow_tier == infr_core::tier::LinearTier::MultiRow { band: b };
            let mrow16 = in_band(MROW16_BAND)
                && mmv_gate
                && crate::gemm::native_mmv_mrow_m16_kernel_name(dt).is_some()
                && be_.cfg().kernels.vulkan.mrow16;
            if (in_band(MROW_BAND) || mrow16)
                && in_f % 32 == 0
                && crate::gemm::native_mrow_kernel_name(dt).is_some()
                && be_.cfg().kernels.vulkan.mrow
            {
                // Int8 dp4a multi-row GEMV: quantize the m activation rows once (`quant_q8`), then
                // integer-dot the raw weight blocks against ALL rows — the dequant mrow's
                // per-sub-block scalar byte-extract dequant is the m-batch GEMV cost on ALU-bound
                // shapes (E2B pp4@d4096: GEMV class 21.0 → 17.0us/op, pp4 366 → 383 t/s). Dtype
                // eligibility is `mrow_int8_dtype_ok(caps, dt, graph.mtp_verify)`: an MTP-verify
                // batch (`Graph::mtp_verify`) is gated to the SAME dtype set as the m=1 decode
                // tier (`mmv_int8_decode_dtypes`) — int8 in both streams or in neither, never
                // split (that split IS the Q5_K token-divergence bug; `mmv_mrow_symmetric_q2k_q3k`
                // proves the two kernels agree once both are on). Ordinary prefill instead reads
                // `mrow_int8_prefill_dtypes` — every measured win, independent of the decode set.
                // Gates:
                //  • m >= 3: m=2 is the single-head MTP spec-verify shape, whose accept loop
                //    holds a hard output-identical-to-target-greedy bar (`mtp_spec_matches_
                //    target_only_greedy`) — int8-quantized activations perturb the verify logits
                //    enough to risk argmax flips vs the m=1 decode GEMV, so that shape keeps the
                //    f32-dequant mrow.
                //  • weight >= 8M elements: below that the kernel is dispatch-latency-floor
                //    bound (the int8 math saves nothing) and the extra quant_q8 bubble is a pure
                //    loss — qwen3-0.6b (all projections < 8M) measured pp4@d4096 934 → 854 t/s
                //    ungated, and E2B's own win came from its gu/down (9.4-18.9M elems).
                //  • out_f < 65536: the vocab-sized lm_head GEMV is pure weight-bandwidth (the
                //    dequant ALU hides fully behind the stream) — the int8 form measured a hair
                //    SLOWER there (qwen3-0.6b pp4@d4096 931 vs 916 with lm_head included).
                // Same INFR_NO_MMV escape as the m=1 mmv (A/B).
                if m >= 3 && mmv_gate {
                    let nblk = in_f / 32;
                    let qa = pooled(pool, be_, "mmvr_qa", m * in_f)?;
                    let dact = pooled(pool, be_, "mmvr_dact", m * nblk * 2)?;
                    let sact = pooled(pool, be_, "mmvr_sact", m * nblk * 2)?;
                    // Same x already quantized by the immediately-preceding Linear(s)? Reuse it
                    // (see `mmv_memo`'s doc — the take()/restore protocol guarantees nothing
                    // rewrote x or the pooled qa in between).
                    if memo_prev != Some((*x, m, in_f)) {
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            m,
                            in_f,
                        );
                    }
                    *mmv_memo = Some((*x, m, in_f));
                    rec.linear_mmv_mrow(
                        dt,
                        w,
                        w_off,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        None,
                        y,
                        m,
                        in_f,
                        out_f,
                    );
                    return Ok(());
                }
                rec.linear_native_mrow(dt, w, w_off, xb, y, m, in_f, out_f);
                return Ok(());
            }
            let gemm_ok = m > 1 && out_f % 64 == 0 && in_f % 32 == 0;
            // The whole GEMM tier below (matmul_native*/matmul_proj, the coopmat tiled prefill
            // GEMM) dispatches ONLY coopmat SPIR-V — RADV still executes it on hardware without
            // the feature (silent, no fault), so gating is required, not optional (issue: coopmat
            // dispatch on a device that lacks VK_KHR_cooperative_matrix segfaults on some
            // drivers). `!caps.f16_coopmat` routes to the NON-COOPMAT GEMM tier below (`nc_mmq`/
            // `nc_fma` — real tiled GEMMs with no coopmat SPIR-V), and only past that to the
            // scalar `else` arms (`linear_native`/`linear`/`linear_f32` — the SAME scalar
            // dequant-in-shader kernels the m==1 decode GEMV already uses, generalized to
            // `rows=m`; they dispatch `rows*out_f` (or row-tiled) workgroups with no upper bound
            // on `rows`, so they are a drop-in, just without any weight-reuse-across-rows win).
            // Bit-identical to before when caps are all true.
            // ── 8x8x16 coopmat prefill GEMM tier (Intel Arc/ANV XMX; `caps.f16_coopmat_8x8()` is
            // Some only when the device enumerates the 8x8x16 f16 shape AND `INFR_CM_8X8=1` —
            // NEVER on RADV, which enumerates 16x16x16 and keeps the full tier below untouched).
            // Only `native_gemm_warp`'s `_cm8` builds exist at this shape: one fixed
            // NARROW_N+BM32 tile (BN=128, BK=64 → n%128, k%64) over the hot k-quants + Q8_0.
            // Anything it can't cover (dtype without a `_cm8` build, non-dividing shape) keeps
            // falling to the nc_mmq/nc_fma tier — the Arc default this knob A/Bs against.
            let cm8_ok = gemm_ok
                && be_.caps().f16_coopmat_8x8()
                && out_f % 128 == 0
                && in_f % 64 == 0
                && crate::gemm::native_gemm_warp_cm8_build_spv(dt).is_some();
            let is_gemm = cm8_ok
                || (gemm_ok
                    && be_.caps().f16_coopmat()
                    && (native_dense_supported(dt) || matches!(dt, infr_core::DType::F16)));
            // ── NON-COOPMAT prefill GEMM tier (never taken when `caps.f16_coopmat` — zero effect
            // on coopmat-capable defaults). The Intel Arc A770 (Mesa ANV) enumerates f16 coopmat
            // only at M=8,N=8,K=16 — `caps.f16_coopmat()` is false there (see lib.rs) — but HAS
            // packed int8 dot, and prefill GEMMs previously fell all the way to the per-row
            // scalar GEMVs (the field-measured 10-30x prefill gap; decode was already at parity).
            // This tier remains the Arc DEFAULT; the opt-in `cm8_ok` XMX tier above takes only
            // the Linears it covers, and only under INFR_CM_8X8=1.
            // Two arms, same padded-dst convention as `is_gemm`:
            //  • `nc_mmq`: weight dtype in `infr_core::tensor::MOE_MMQ_DTYPES` + `caps.i8_dot` →
            //    the DENSE dp4a mmq GEMM family (`matmul_mmq` — the same shader code the batched
            //    MoE expert path already gates on `i8_dot` alone, base-grid build), fed by the
            //    same `quant_q8` activation prepass the coopmat tier's Q4_K-mmq arm uses.
            //  • `nc_fma`: f16/bf16/f32 weights (no integer codes to dp4a) → the shared-memory
            //    fma warptile (`matmul_fma`, native_gemm_fma.comp — no subgroup ops, no f16
            //    exts; bf16 reads stay native, no f32 upconvert).
            // Exercisable on coopmat hardware via INFR_NO_COOPMAT=1 (lib.rs's force-disable test
            // knob — drops the device feature itself, a faithful simulation). INFR_NO_MMQ_FALLBACK=1
            // disables the WHOLE tier (both arms) for A/B against the scalar floor.
            let nc_tier =
                gemm_ok && !be_.caps().f16_coopmat() && be_.cfg().kernels.vulkan.mmq_fallback;
            let nc_mmq = nc_tier && be_.caps().i8_dot && infr_core::tensor::moe_mmq_ok(dt);
            let nc_fma =
                nc_tier && !nc_mmq && crate::gemm::native_gemm_fma_kernel_name(dt).is_some();
            // A `>= 2^32`-element weight (a big-vocab lm_head/embed) can only take the CHUNKED GEMV
            // (decode m=1, or the per-input-row GEMV) — the tiled coopmat/nc GEMMs below decode a
            // 64-row weight tile that can't be split into single input rows, so a breach on this
            // path (a multi-row lm_head: MTP speculative verify or all-position logits) must fail
            // loudly, never wrap. Issue #77: covered classes chunk; everything else stays loud.
            if (is_gemm || nc_mmq || nc_fma)
                && crate::recorder::bda_weight_breaches(dt, in_f, out_f, &be_.cfg().kernels.vulkan)
            {
                return Err(be(format!(
                    "vulkan adapter: output-projection tiled GEMM (m={m}, in_f={in_f}, \
                     out_f={out_f}, {dt:?}) reads a >= 2^32-element weight — chunked dispatch \
                     covers only m=1 decode and the per-row GEMV (issue #77). This is a multi-row \
                     lm_head (speculative verify / all-position logits) on a big-vocab model."
                )));
            }
            if is_gemm {
                // GEMM writes ceil(m/64)*64 rows; `with_padded_dst` writes an Internal (row-padded)
                // `dst` direct and a non-Internal dst (the lm_head `logits` Output) via a padded
                // temp + copy of the m real rows. Same dance as the nc tier and the streamed twin.
                with_padded_dst(be_, graph, rec, dst, y, m, out_f, transient, |out| {
                    // Q4_K: the 8-warp warptile coopmat GEMM (native_gemm_warp, in-shader dequant)
                    // beats mmq dp4a at prefill shapes (161 vs 417µs on [512,1024]×[1024,6144] — the
                    // wide tile amortizes staging; RADV can't use int8 WMMA). mmq stays for shapes NO
                    // warp tile can cover — n%128≠0 (the NARROW_N tile covers n%128; gemma3-1b's
                    // ne=1152 projections sat on scalar mmq at ~10 TF under the old n%256 gate).
                    // INFR_NO_MMQ also skips mmq for A/B.
                    let warp_ok = out_f % 128 == 0
                        && crate::gemm::native_gemm_warp_kernel_name(dt).is_some()
                        && be_.cfg().kernels.vulkan.gemm_warp;
                    // int8 cooperative-matrix (WMMA) prefill GEMM — MEASUREMENT path, Q8_0 only
                    // (crates/infr-vulkan/shaders/native_gemm_i8cm_q8_0.comp). `caps.i8_coopmat` is
                    // hardware detection only (see its doc); this dispatch ALSO requires
                    // `INFR_I8_COOPMAT=1` (default off) because int8 coopmat hung the GPU on an older
                    // Mesa despite enumerating fine there too (commit ad82a77) — the toggle keeps that
                    // regression class opt-in until a Mesa-version-gated default is warranted. Default
                    // behavior (unset) is completely unaffected: this is a new, additive branch ahead
                    // of the existing Q4_K-mmq / warp-coopmat / off-tier arms below, which are
                    // untouched.
                    // fp8 (E4M3) cooperative-matrix (WMMA) prefill GEMM — Q8_0 only
                    // (crates/infr-vulkan/shaders/native_gemm_f8cm_q8_0.comp). `caps.f8_coopmat` is
                    // hardware enumeration only (RDNA4/Navi44 confirmed, see lib.rs `has_f8_coopmat`);
                    // this dispatch ALSO requires `INFR_F8_COOPMAT=1` (default off) because — unlike
                    // `i8cm_ok` below — this kernel hasn't been run/measured on real fp8-coopmat
                    // hardware at all yet (this dev box has none), so the opt-in stays until an RDNA4
                    // pass validates correctness. Checked AHEAD of `i8cm_ok`: both are Q8_0-only
                    // measurement tiers, so when a caller opts into fp8 it takes priority. Default
                    // behavior (unset) is completely unaffected — new, additive branch ahead of the
                    // i8cm / Q4_K-mmq / warp-coopmat / off-tier arms below, which are untouched.
                    //
                    // Shape gate mirrors `warp_ok`'s f16 warptile pick: WIDE (BN=256, BK=32) needs
                    // out_f%256==0 && in_f%32==0; NARROW_N (BN=128, BK=64) needs out_f%128==0 &&
                    // in_f%64==0 for shapes the wide tile can't cover (n%128 not n%256 — see
                    // native_gemm_f8cm_q8_0.comp's -DNARROW_N). Neither divides (e.g. out_f%128!=0, or
                    // out_f%128==0 with in_f%64!=0 so narrow's BK=64 can't stage either) -> f8cm_ok is
                    // false and the shape falls through to i8cm/mmq/native below, unaffected.
                    let f8_wide = out_f % 256 == 0 && in_f % 32 == 0;
                    let f8_narrow = !f8_wide && out_f % 128 == 0 && in_f % 64 == 0;
                    let f8cm_ok = matches!(dt, infr_core::DType::Q8_0)
                        && be_.caps().f8_coopmat()
                        && (f8_wide || f8_narrow)
                        && be_.cfg().kernels.vulkan.f8_coopmat;
                    // `i8_coopmat_ready` is `caps.i8_coopmat()` AND the `INFR_I8_COOPMAT=1` opt-in
                    // AND this driver having PASSED the accumulator-layout known-answer probe at
                    // init (see `VulkanBackend::verify_i8_coopmat_layout`): the kernel's
                    // in-fragment descale reads accumulator elements at a mapping that is fixed per
                    // implementation, so enumeration alone does not make it safe to dispatch.
                    let i8cm_ok = matches!(dt, infr_core::DType::Q8_0) && be_.i8_coopmat_ready();
                    // NATIVE bf16 cooperative-matrix (WMMA) prefill GEMM — the `-DBF16CM` build of the
                    // SAME production kernel (crates/infr-vulkan/shaders/native_gemm_warp.comp) that
                    // `native_gemm_warp_bf16` (the f16-clamped path below) already uses; only the
                    // coopmat A/B operand type differs, so it should match that kernel's speed while
                    // keeping bf16's full exponent range. `caps.bf16_coopmat` is hardware enumeration
                    // only (RDNA4/Navi44 confirmed, see lib.rs `has_bf16_coopmat`); this dispatch ALSO
                    // requires `INFR_BF16_COOPMAT=1` (default off) because this variant hasn't been
                    // run/measured on real bf16-coopmat hardware at all yet (this dev box has none), so
                    // the opt-in stays until an RDNA4 pass validates correctness. Default behavior
                    // (unset) is completely unaffected: Bf16 keeps routing through the existing
                    // `native_dense_supported` arm below (`native_gemm_warp_bf16`, the f16-clamped
                    // path), byte-identical to before this branch existed. Shape gate mirrors
                    // `f8_wide`/`f8_narrow`: WIDE (BN=256, BK=32) needs out_f%256==0 && in_f%32==0;
                    // NARROW_N (BN=128, BK=64) needs out_f%128==0 && in_f%64==0.
                    let bf16cm_wide = out_f % 256 == 0 && in_f % 32 == 0;
                    let bf16cm_narrow = !bf16cm_wide && out_f % 128 == 0 && in_f % 64 == 0;
                    let bf16cm_ok = matches!(dt, infr_core::DType::Bf16)
                        && be_.caps().bf16_coopmat()
                        && (bf16cm_wide || bf16cm_narrow)
                        && be_.cfg().kernels.vulkan.bf16_coopmat;
                    if cm8_ok {
                        // 8x8x16 `_cm8` warptile (see the `cm8_ok` doc above). First in the chain:
                        // when it's true, `caps.f16_coopmat()` is false (the shapes are mutually
                        // exclusive), so every 16x16x16 branch below would mis-dispatch — none of
                        // their SPIR-V exists at the device's fragment shape.
                        rec.matmul_native_cm8(dt, xb, w, w_off, out, m, in_f, out_f);
                    } else if f8cm_ok {
                        // E4M3's range is tiny (max normal 448) — unscaled f32 activations overflow to
                        // inf/NaN on cast, which is what produced garbage output on the first RDNA4
                        // run. `quant_f8_row` pre-scales activations into E4M3's range (one amax/scale
                        // per row) before the coopmat GEMM, which descales the output by that same
                        // per-row scale in its epilogue. Q8_0 weights are unscaled (see
                        // native_gemm_f8cm_q8_0.comp doc — their post-dequant magnitude is expected to
                        // already fit E4M3).
                        let qa = pooled(pool, be_, "f8cm_qa", m * in_f)?;
                        let srow = pooled(pool, be_, "f8cm_srow", m * 4)?;
                        rec.quant_f8_row(xb, pool[&qa].as_ref(), pool[&srow].as_ref(), m, in_f);
                        // INFR_F8_PREPACK=1 (requires INFR_F8_COOPMAT=1 too, since it only takes
                        // effect inside this arm): bakes the Q8_0 block scale into an E4M3 weight
                        // buffer ONCE via `repack_q8_to_f8`, then the GEMM's Bs staging reads that
                        // buffer DIRECTLY — no in-shader dqblk. Isolates whether removing the
                        // dequant-ALU bottleneck (the SAME cost f16 pays; fp8-dqblk measured 0.73x
                        // f16 on RDNA4) lets fp8's 2x WMMA rate win. The E4M3 buffer is pooled
                        // per-shape scratch (like every other tier here) and re-repacked on EVERY
                        // forward call — a real deployment would cache the repack at load time, not
                        // redo it per Linear; this is a per-op profiling isolation path, not the
                        // proposed production shape. Unset (default): unchanged dqblk path below,
                        // byte-identical to before this branch existed.
                        if be_.cfg().kernels.vulkan.f8_prepack {
                            let f8_w8 = pooled(pool, be_, "f8_w8", out_f * in_f)?;
                            rec.repack_q8_to_f8(w, w_off, pool[&f8_w8].as_ref(), out_f, in_f);
                            rec.matmul_f8cm_q8_0_prepacked(
                                pool[&qa].as_ref(),
                                pool[&srow].as_ref(),
                                pool[&f8_w8].as_ref(),
                                out,
                                m,
                                in_f,
                                out_f,
                            );
                        } else {
                            rec.matmul_f8cm_q8_0(
                                pool[&qa].as_ref(),
                                pool[&srow].as_ref(),
                                w,
                                w_off,
                                out,
                                m,
                                in_f,
                                out_f,
                            );
                        }
                    } else if i8cm_ok {
                        // "Idea 2" measurement (INFR_I8_ROW_SCALE=1, requires INFR_I8_COOPMAT=1 too):
                        // whole-row (block-invariant) activation scale instead of quant_q8's
                        // per-32-block scale — see native_gemm_i8cm_q8_0.comp #ifdef ROW_SCALE /
                        // quant_q8_row.comp. Separate pool tag (different buffer size/layout) and a
                        // separate kernel pair, so this can be measured and reverted independently of
                        // the i8cm baseline path above.
                        if be_.cfg().kernels.vulkan.i8_row_scale {
                            let qa = pooled(pool, be_, "i8cm_qa_row", m * in_f)?;
                            let dact_row = pooled(pool, be_, "i8cm_dact_row", m * 2)?;
                            rec.quant_q8_row(
                                xb,
                                pool[&qa].as_ref(),
                                pool[&dact_row].as_ref(),
                                m,
                                in_f,
                            );
                            rec.matmul_i8cm_q8_0_rowscale(
                                pool[&qa].as_ref(),
                                pool[&dact_row].as_ref(),
                                w,
                                w_off,
                                out,
                                m,
                                in_f,
                                out_f,
                            );
                        } else {
                            let nblk = in_f / 32;
                            let qa = pooled(pool, be_, "mmq_qa", m * in_f)?;
                            let dact = pooled(pool, be_, "mmq_dact", m * nblk * 2)?;
                            let sact = pooled(pool, be_, "mmq_sact", m * nblk * 2)?;
                            rec.quant_q8(
                                xb,
                                pool[&qa].as_ref(),
                                pool[&dact].as_ref(),
                                pool[&sact].as_ref(),
                                m,
                                in_f,
                            );
                            rec.matmul_i8cm_q8_0(
                                pool[&qa].as_ref(),
                                pool[&dact].as_ref(),
                                w,
                                w_off,
                                out,
                                m,
                                in_f,
                                out_f,
                            );
                        }
                    } else if bf16cm_ok {
                        rec.matmul_bf16cm(xb, w, w_off, out, m, in_f, out_f);
                    } else if matches!(dt, infr_core::DType::Q4K)
                        && !warp_ok
                        && be_.caps().i8_dot
                        && be_.cfg().kernels.vulkan.mmq
                    {
                        // mmq (dp4a int8): quantize activations once, integer matmul on raw blocks.
                        // Scratch is pooled — every same-shape Linear in the graph reuses one set.
                        let nblk = in_f / 32;
                        let qa = pooled(pool, be_, "mmq_qa", m * in_f)?;
                        let dact = pooled(pool, be_, "mmq_dact", m * nblk * 2)?;
                        let sact = pooled(pool, be_, "mmq_sact", m * nblk * 2)?;
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            m,
                            in_f,
                        );
                        rec.matmul_mmq_q4k(
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            w,
                            w_off,
                            out,
                            m,
                            in_f,
                            out_f,
                        );
                    } else if native_dense_supported(dt) {
                        // A_GLOBAL f16-A cast / narrow-n split-K / direct tile — the SAME tile selection
                        // as the streamed twin (`native_warp_gemm`), reading the resident weight by its
                        // BDA. Occupancy note: dropping the As stage shrinks LDS to Bs-only → ~1.5x on
                        // the 8B prefill shapes (o proj 29→44 TF). Pool pad rows may hold stale garbage;
                        // GEMM rows are independent, and dst pad rows are never read.
                        let w_addr = w
                            .device_addr()
                            .expect("resident-BDA weight: dense Linear requires a u64 BDA address");
                        native_warp_gemm(
                            be_, dt, w_addr, w_off, xb, out, m, in_f, out_f, rec, pool,
                        )?;
                    } else {
                        // F16 deep-k narrow-n → SPLIT-K warptile (DG slice-7 comparative
                        // attribution): the SC soft-embedding GEMM (m=canvas=256, k=vocab=262144,
                        // n=ne=2816, f16 weight) measured 58ms/step at ~6.5 TFLOPS on the legacy
                        // `gemm_proj` BN=64 tile below (28% of the whole denoise step) vs llama.cpp's
                        // 14.1ms @ 27 TFLOPS for the IDENTICAL shape — the single biggest
                        // infr-vs-fork per-stage delta. Same narrow-grid split-K policy as the
                        // native-quant branch above (out_f%128, deep k, grid < 128 wgs); the F16
                        // decode is exact (see native_decode.glsl FMT_F16), so numerics match the
                        // f16 coopmat route up to accumulation order. Everything that doesn't hit
                        // the split-K window keeps the `matmul_proj` route unchanged.
                        // Same splits policy as the quant branch above (`split_k_plan`). Probed deeper
                        // splits at the SC shape (splits=4/8/16 via a temporary env override): the op
                        // shaved 30.0 -> 25.8-28.4ms but dg-step stayed flat (1100-1107, within noise),
                        // so the shared formula stays — no bespoke tuning knob.
                        let splits = split_k_plan(m, in_f, out_f);
                        if matches!(dt, infr_core::DType::F16)
                            && splits > 1
                            && crate::gemm::native_gemm_warp_sk_kernel_name(dt).is_some()
                            && be_.cfg().kernels.vulkan.gemm_warp
                        {
                            let mpad = m.div_ceil(64) * 64;
                            let pk = pooled(pool, be_, "splitk_part", splits * mpad * out_f * 4)?;
                            let w_addr = w.device_addr().expect(
                            "resident-BDA weight: F16 split-K Linear requires a u64 BDA address",
                        );
                            rec.matmul_native_splitk(
                                dt,
                                xb,
                                w_addr,
                                w_off,
                                pool[&pk].as_ref(),
                                out,
                                m,
                                in_f,
                                out_f,
                                splits,
                                false,
                            );
                        } else if let Some(w_addr) = f16_w_addr {
                            rec.matmul_proj_at(xb, w_addr, out, m, in_f, out_f);
                        } else {
                            // f16 coopmat GEMM. `matmul_proj` internally forks on `wq.device_addr()`
                            // (see its recorder doc) — no threading needed here.
                            rec.matmul_proj(xb, w, out, m, in_f, out_f);
                        }
                    }
                    Ok(())
                })?;
            } else if nc_mmq || nc_fma {
                // Non-coopmat prefill GEMM tier (see the `nc_tier` doc above). Same padded-dst
                // dance as `is_gemm` (`with_padded_dst`): the GEMM writes ceil(m/64)*64 rows.
                with_padded_dst(be_, graph, rec, dst, y, m, out_f, transient, |out| {
                    if nc_mmq {
                        // Same pooled scratch tags/sizes as the coopmat tier's Q4_K-mmq arm — the
                        // two arms are mutually exclusive per device, so the tags never collide.
                        let nblk = in_f / 32;
                        let qa = pooled(pool, be_, "mmq_qa", m * in_f)?;
                        let dact = pooled(pool, be_, "mmq_dact", m * nblk * 2)?;
                        let sact = pooled(pool, be_, "mmq_sact", m * nblk * 2)?;
                        rec.quant_q8(
                            xb,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            m,
                            in_f,
                        );
                        rec.matmul_mmq(
                            dt,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            w,
                            w_off,
                            out,
                            m,
                            in_f,
                            out_f,
                        );
                    } else {
                        rec.matmul_fma(dt, xb, w, w_off, out, m, in_f, out_f);
                    }
                    Ok(())
                })?;
            } else if native_dense_supported(dt) {
                // Decode (m=1) on int-dot-capable K-quants → mmv (see the fused-add branch above).
                // Wave32-native GPUs take the multi-warp dp4a route first (`mmv_mw_choice`); AMD
                // falls through to the scalar GEMV (default) or the old mmv (INFR_MMV_DECODE=1).
                // A `>= 2^32`-element lm_head is chunk-covered on the native dequant GEMV
                // (`linear_native`, the `else` below) but NOT on this multi-warp int8 dp4a mmv
                // route — route a breaching weight to the chunked path rather than wrap its u32
                // index (issue #77). Only lm_head/embed can breach; the tier skipped here matters
                // only for a 256k-vocab model's single vocab GEMV.
                let mw = if m == 1
                    && !crate::recorder::bda_weight_breaches(
                        dt,
                        in_f,
                        out_f,
                        &be_.cfg().kernels.vulkan,
                    ) {
                    mmv_mw_choice(be_.caps(), &be_.cfg().kernels.vulkan, dt, in_f, out_f)
                } else {
                    None
                };
                if let Some(warps) = mw {
                    // FUSE_QUANT opt-in: inline activation quantization, skip the
                    // separate quant_q8 dispatch.
                    if be_.cfg().kernels.vulkan.mmv_fuse_quant {
                        rec.linear_mmv_mrow_fuseq(dt, w, w_off, xb, None, y, 1, in_f, out_f);
                        return Ok(());
                    }
                    let nblk = in_f / 32;
                    let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                    let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                    let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                    rec.quant_q8(
                        xb,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        1,
                        in_f,
                    );
                    if unified_mmv_row1(be_.caps()) {
                        // Same kernel/reduction as the m>=3 verify tier, dispatched at rows=1 — see
                        // the fused-add branch above and `unified_mmv_row1`'s doc.
                        rec.linear_mmv_mrow(
                            dt,
                            w,
                            w_off,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            None,
                            y,
                            1,
                            in_f,
                            out_f,
                        );
                    } else {
                        rec.linear_mmv_mw(
                            dt,
                            warps,
                            w,
                            w_off,
                            pool[&qa].as_ref(),
                            pool[&dact].as_ref(),
                            pool[&sact].as_ref(),
                            None,
                            y,
                            in_f,
                            out_f,
                        );
                    }
                } else if m == 1
                    && mmv_decode_enabled(&be_.cfg().kernels.vulkan)
                    && be_.caps().i8_dot
                    && in_f % 32 == 0
                    && in_f * out_f >= MMV_MIN_ELEMS
                    && crate::gemm::native_mmv_kernel_name(dt, false).is_some()
                    // A `>= 2^32`-element lm_head is chunk-covered on the native dequant GEMV
                    // (`linear_native`, below) but NOT on this int8 dp4a mmv tier — route a
                    // breaching weight to the chunked path rather than wrap its u32 index (issue
                    // #77). Only lm_head/embed can breach, and the perf tier they skip here matters
                    // only for a 256k-vocab model's single vocab GEMV.
                    && !crate::recorder::bda_weight_breaches(dt, in_f, out_f, &be_.cfg().kernels.vulkan)
                {
                    let nblk = in_f / 32;
                    let qa = pooled(pool, be_, "mmv_qa", in_f)?;
                    let dact = pooled(pool, be_, "mmv_dact", nblk * 2)?;
                    let sact = pooled(pool, be_, "mmv_sact", nblk * 2)?;
                    rec.quant_q8(
                        xb,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        1,
                        in_f,
                    );
                    rec.linear_mmv(
                        dt,
                        w,
                        w_off,
                        pool[&qa].as_ref(),
                        pool[&dact].as_ref(),
                        pool[&sact].as_ref(),
                        y,
                        in_f,
                        out_f,
                    );
                } else {
                    rec.linear_native(dt, w, w_off, xb, y, m, in_f, out_f);
                }
            } else if matches!(dt, infr_core::DType::F32) {
                // Full-precision projection weight (gemma4 E2B per-layer inp_gate/proj, DeepSeek
                // V4's grouped output projection): the seam uploads native dtype, and the f16 GEMV
                // would read the f32 bytes as f16 garbage.
                if w_off == 0 {
                    rec.linear_f32(w, xb, y, m, in_f, out_f);
                } else {
                    // Row-aligned slice of the weight (`w_off % in_f == 0`, `Op::Linear::w_off`'s
                    // contract) expressed as a shifted base ADDRESS, because `linear_f32r.comp` has
                    // no offset field. Alignment: the vec4 variant declares a 16-byte-aligned
                    // buffer_reference and is only chosen when `in_f % 4 == 0`, and `w_off` is a
                    // whole multiple of `in_f`, so `w_off * 4` bytes is a multiple of 16 exactly
                    // when that variant can run. The scalar variants need 4, which any f32 element
                    // offset satisfies. Asserted rather than assumed — a violation would read
                    // misaligned and is not otherwise detectable.
                    if w_off % in_f != 0 {
                        return Err(be(format!(
                            "vulkan adapter: Linear w_off {w_off} is not a whole multiple of \
                             in_f {in_f} — the f32 GEMV addresses a row-aligned weight slice"
                        )));
                    }
                    let base = w.device_addr().ok_or_else(|| {
                        be("vulkan adapter: f32 Linear with w_off needs a BDA weight address")
                    })?;
                    rec.linear_f32_at(base + (w_off * 4) as u64, xb, y, m, in_f, out_f);
                }
            } else if be_.caps().f16 {
                if let Some(w_addr) = f16_w_addr {
                    rec.linear_at(w_addr, xb, y, m, in_f, out_f);
                } else {
                    rec.linear(w, xb, y, m, in_f, out_f);
                }
            } else {
                // No shaderFloat16 (implies no coopmat too — f16 is a coopmat prerequisite):
                // linear_f16.comp's `float16_t` SSBO read needs the Float16 SPIR-V capability the
                // device lacks. linear_f16_noext.comp is the same dispatch shape reading the f16
                // weight buffer as packed u32 words and unpacking via the CORE `unpackHalf2x16`
                // builtin instead — correctness-first, not perf-tuned (no row-tiling).
                if let Some(w_addr) = f16_w_addr {
                    rec.linear_f16_noext_at(w_addr, xb, y, m, in_f, out_f);
                } else {
                    rec.linear_f16_noext(w, xb, y, m, in_f, out_f);
                }
            }
        }
        Op::Add { a, b, dst, n } => rec.add(r(*a)?, r(*b)?, r(*dst)?, *n as usize),
        Op::AddBias {
            x,
            bias,
            dst,
            rows,
            n,
        } => rec.add_bias(r(*x)?, r(*bias)?, r(*dst)?, *rows as usize, *n as usize),
        Op::MulVec {
            x,
            vec,
            dst,
            rows,
            n,
        } => rec.mul_vec(r(*x)?, r(*vec)?, r(*dst)?, *rows as usize, *n as usize),
        Op::HeadwiseSigmoidMul {
            x,
            gate,
            dst,
            rows,
            n_head,
            head_dim,
        } => rec.headwise_sigmoid_mul(
            r(*x)?,
            r(*gate)?,
            r(*dst)?,
            *rows as usize,
            *n_head as usize,
            *head_dim as usize,
        ),
        Op::MoeSharedExpertAdd {
            moe,
            shexp,
            gate,
            dst,
            rows,
            n,
        } => rec.moe_shared_expert_add(
            r(*moe)?,
            r(*shexp)?,
            r(*gate)?,
            r(*dst)?,
            *rows as usize,
            *n as usize,
        ),
        Op::Scale { x, dst, s, n } => {
            let n = *n as usize;
            // recorder `scale` is in place on its buffer; copy x→dst first if they differ.
            if x != dst {
                let eb = graph.desc(*dst).dtype.dense_bytes(1).unwrap_or(4);
                rec.copy(r(*x)?, 0, r(*dst)?, 0, n * eb);
            }
            rec.scale(r(*dst)?, *s, n);
        }
        Op::Copy {
            src,
            src_off,
            dst,
            dst_off,
            n,
        } => {
            let (sdt, ddt) = (graph.desc(*src).dtype, graph.desc(*dst).dtype);
            if matches!(sdt, infr_core::DType::F32) && matches!(ddt, infr_core::DType::F16) {
                // Cross-dtype cast-copy (llama4's NoPE Q/K → the f16 rope scratch, unroped/
                // unnormed on those layers — see runner.rs): a raw byte copy would corrupt values
                // (f32 is 2x f16's width), so cast element-wise instead (reuses WriteKv's f32→f16
                // kernel). `store_f16` has no source offset — every in-tree caller of a cross-dtype
                // Copy extracts a whole row range (`src_off == 0`). A structurally-valid IR with a
                // nonzero source offset is unsupported here, not a bug: return a recoverable error
                // (the seam falls back) rather than aborting the process like the sibling shape guards.
                if *src_off != 0 {
                    return Err(be(
                        "vulkan adapter: cross-dtype Copy needs src_off==0 (store_f16 has no source offset)",
                    ));
                }
                rec.store_f16(r(*src)?, r(*dst)?, *n as usize, *dst_off as usize);
            } else {
                // IR offsets/counts are in ELEMENTS; the recorder copy takes BYTES.
                let eb = sdt.dense_bytes(1).unwrap_or(4);
                rec.copy(
                    r(*src)?,
                    *src_off as usize * eb,
                    r(*dst)?,
                    *dst_off as usize * eb,
                    *n as usize * eb,
                );
            }
        }
        // Batched strided copy = `rows` buffer-copy regions (cheap vkCmdCopyBuffer). Splits a
        // batched interleaved buffer (conv q|k|v) into packed per-row slices.
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
            let eb = graph.desc(*src).dtype.dense_bytes(1).unwrap_or(4);
            let (rows_, n_) = (*rows as usize, *n as usize);
            let (so, ss, do_, ds) = (
                *src_off as usize,
                *src_stride as usize,
                *dst_off as usize,
                *dst_stride as usize,
            );
            // One compute dispatch when everything is u32-word aligned (f32 always; f16 when the
            // element counts are even) — the per-row copy loop recorded `rows` vkCmdCopyBuffer +
            // hazard checks per split op (thousands per prefill chunk), dwarfing the bytes moved.
            let word_ok = [so * eb, ss * eb, do_ * eb, ds * eb, n_ * eb]
                .iter()
                .all(|b| b % 4 == 0);
            if word_ok {
                rec.copy_strided(
                    r(*src)?,
                    r(*dst)?,
                    rows_,
                    n_ * eb / 4,
                    so * eb / 4,
                    ss * eb / 4,
                    do_ * eb / 4,
                    ds * eb / 4,
                );
            } else {
                for row in 0..rows_ {
                    rec.copy(
                        r(*src)?,
                        (so + row * ss) * eb,
                        r(*dst)?,
                        (do_ + row * ds) * eb,
                        n_ * eb,
                    );
                }
            }
        }
        // Gated FFN activation: `act(gate) * up[+up_off]`. up_off (E2B per-layer slice) only
        // arises with Gelu (gemma); silu/sigmoid are always up_off==0. up_stride enables per-row
        // strided reads from a wider buffer (eliminates the CopyStrided dispatch per layer).
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
            swiglu_clamp,
            ..
        } => {
            let n = *rows as usize * *nff as usize;
            let (g_, u_, y) = (r(*gate)?, r(*up)?, r(*dst)?);
            let eb = graph.desc(*up).dtype.dense_bytes(1).unwrap_or(4);
            match act {
                Activation::Silu => {
                    // `silu_mul` reads the gate contiguously, so a strided gate (unlike the
                    // Sigmoid arm's `mul_sigmoid`, which honors gate_stride/gate_block_width)
                    // would be computed silently wrong — guard it like the up_off/up_stride cases.
                    if *up_off != 0 || *up_stride != 0 {
                        return Err(be(
                            "vulkan adapter: GatedAct Silu up_off/stride!=0 unsupported",
                        ));
                    }
                    if *gate_stride != 0 || *gate_block_width != 0 {
                        return Err(be(
                            "vulkan adapter: GatedAct Silu gate_stride/gate_block_width!=0 unsupported",
                        ));
                    }
                    rec.silu_mul(g_, u_, y, n, *swiglu_clamp);
                }
                Activation::Sigmoid => {
                    if *up_off != 0 || *up_stride != 0 {
                        return Err(be(
                            "vulkan adapter: GatedAct Sigmoid up_off/stride!=0 unsupported",
                        ));
                    }
                    if swiglu_clamp.is_some() {
                        return Err(be(
                            "vulkan adapter: GatedAct Sigmoid swiglu_clamp unsupported",
                        ));
                    }
                    rec.mul_sigmoid(
                        u_,
                        g_,
                        y,
                        n,
                        *gate_stride as usize,
                        *nff as usize,
                        *gate_block_width as usize,
                    );
                }
                Activation::Gelu => {
                    // `gelu_mul_off` is the strided form, which pushes the clamp words as zero —
                    // no GELU caller clamps (V4 is SiLU), so this stays a refusal rather than a
                    // second clamped kernel variant nothing exercises.
                    if swiglu_clamp.is_some() {
                        return Err(be("vulkan adapter: GatedAct Gelu swiglu_clamp unsupported"));
                    }
                    rec.gelu_mul_off(
                        g_,
                        u_,
                        *up_off as usize * eb,
                        *up_stride as usize * eb,
                        *nff as usize,
                        *gate_stride as usize * eb,
                        *nff as usize,
                        (*gate_block_width as usize) * eb,
                        y,
                        n,
                    );
                }
            }
        }
        // Combined [rows, 2*nff] gate|up buffer — the bespoke path's silu/gelu_mul_fused shape
        // (one GEMV/GEMM produced the whole gu, this reads both halves per row).
        Op::GatedActFused {
            gu,
            dst,
            rows,
            nff,
            act,
            swiglu_clamp,
        } => {
            let (rows, nff) = (*rows as usize, *nff as usize);
            let (gu_, y) = (r(*gu)?, r(*dst)?);
            match act {
                Activation::Silu => rec.silu_mul_fused(gu_, y, rows, nff, *swiglu_clamp),
                Activation::Gelu => rec.gelu_mul_fused(gu_, y, rows, nff, *swiglu_clamp),
                Activation::Sigmoid => {
                    return Err(be("vulkan adapter: GatedActFused Sigmoid unsupported"))
                }
            }
        }
        // Append a row into the persistent KV cache at row `pos`. store_f16 casts f32→f16 (the
        // common case: V / f32 K); an already-f16 source is a straight copy. In Dynamic mode the
        // write offset (pos*n) comes from `params` instead of the baked `pos`.
        Op::WriteKv {
            src,
            cache,
            rows,
            row_stride,
            pos,
        } => {
            let (rows, rs, pos) = (*rows as usize, *row_stride as usize, *pos as usize);
            let n = rows * rs;
            let (s, c) = (r(*src)?, r(*cache)?);
            // Q8_0 cache: quantize the row(s) into 34 B/32-elem blocks. For a Q8 cache the K-rope
            // peephole is disabled, so the K WriteKv (f16 staging) reaches here alongside the f32 V.
            let cache_dt = graph.desc(*cache).dtype;
            let cache_q8 = matches!(cache_dt, infr_core::DType::Q8_0);
            let src_f16 = matches!(graph.desc(*src).dtype, infr_core::DType::F16);
            // Planar scales region begins at byte `cap` = total cache elements.
            let cap = graph.desc(*cache).numel();
            // SWA ring cache: the write for position p lands at row p % cap_rows — the ring only
            // recycles rows whose positions the window mask already excludes (see the runner's
            // ring sizing). A batched prefill write crossing the wrap splits into two contiguous
            // segments `(src_row, dst_row, n_rows)`; a full-context cache never wraps (pos <
            // cap_rows), so segment 0 is exactly the old single write there. Only the f16 and Q8
            // cache formats can be ring-sized (the runner's gate); the low-bit/dense-alt/turbo
            // arms below keep the plain `pos * rs` offset, which is identical for their always-
            // full-context caches.
            let cap_rows = cap / rs.max(1);
            let pos_r = if cap_rows > 0 { pos % cap_rows } else { pos };
            let segs: [(usize, usize, usize); 2] = if cap_rows > 0 && pos_r + rows > cap_rows {
                let r1 = cap_rows - pos_r;
                [(0, pos_r, r1), (r1, 0, rows - r1)]
            } else {
                [(0, pos_r, rows), (0, 0, 0)]
            };
            // #74: the store kernels write the DEST cache `c` (always a KvCache, so it always exposes
            // a device address) by pointer. store_q8/quant_kv/store_kv_dense are BDA-only (slice 5) —
            // they read `c.device_addr()` internally; `c` still stays bound at its write slot so the
            // store→(later attention)read barrier survives. store_f16 keeps its bound/BDA fork below
            // (it also has a non-KV scratch caller), so `c_addr` is still computed for it.
            let c_addr = c.device_addr();
            match mode {
                RopeMode::Static(_) if cache_q8 => {
                    for &(sr, dr, nr) in segs.iter().filter(|&&(_, _, nr)| nr > 0) {
                        rec.store_q8(s, c, nr * rs, dr * rs, cap, src_f16, sr * rs);
                    }
                }
                RopeMode::Dynamic(params) if cache_q8 => {
                    rec.store_q8_dyn(s, *params, c, n, cap, src_f16)
                }
                // Mainline low-bit KV quants: quantize into standard GGUF blocks (static only — a
                // quantized KV cache forces static decode, so a Dynamic WriteKv never reaches here).
                RopeMode::Static(_) if is_kv_quant(cache_dt) => {
                    rec.quant_kv(cache_dt, s, c, n, pos * rs, src_f16)
                }
                // Dense f32/bf16 cache: cast-store the row (also static-only).
                RopeMode::Static(_) if is_kv_dense_alt(cache_dt) => {
                    rec.store_kv_dense(cache_dt, s, c, n, pos * rs, src_f16)
                }
                // TurboQuant cache: WHT-quantize the row (static-only). NOT BDA-migrated (#74 slice 3
                // deferred turbo — its dequant read is byte-granular, outside the u32-word/f16 read
                // family; experimental, not in the gate suite).
                RopeMode::Static(_) if is_turbo(cache_dt) => {
                    rec.quant_turbo(cache_dt, s, c, n, pos * rs, src_f16)
                }
                RopeMode::Static(_) => {
                    for &(sr, dr, nr) in segs.iter().filter(|&&(_, _, nr)| nr > 0) {
                        match graph.desc(*src).dtype {
                            infr_core::DType::F16 => {
                                rec.copy(s, sr * rs * 2, c, dr * rs * 2, nr * rs * 2)
                            }
                            _ => match c_addr {
                                Some(a) => rec.store_f16_off_at(s, c, a, nr * rs, dr * rs, sr * rs),
                                None => rec.store_f16_off(s, c, nr * rs, dr * rs, sr * rs),
                            },
                        }
                    }
                }
                RopeMode::Dynamic(params) => match graph.desc(*src).dtype {
                    // The only f16 WriteKv (K) is always fused into the QkNormRope and skipped, so a
                    // standalone f16 WriteKv shouldn't reach the record-once path.
                    infr_core::DType::F16 => {
                        return Err(be("vulkan adapter: dynamic decode f16 WriteKv unexpected"))
                    }
                    _ => match c_addr {
                        Some(a) => rec.store_f16_dyn_at(s, *params, c, a, n, cap_rows),
                        None => rec.store_f16_dyn(s, *params, c, n, cap_rows),
                    },
                },
            }
        }
        Op::Dsv4Compress {
            values,
            scores,
            ape,
            norm,
            freq_factors,
            state,
            dst,
            state_values_off,
            state_scores_off,
            pos,
            ratio,
            dim,
            rope_dim,
            theta,
            eps,
            overlap,
        } => {
            let ff = match freq_factors {
                Some(id) => Some(r(*id)?),
                None => None,
            };
            rec.dsv4_compress(
                r(*values)?,
                r(*scores)?,
                r(*ape)?,
                r(*norm)?,
                ff,
                r(*state)?,
                r(*dst)?,
                *state_values_off,
                *state_scores_off,
                *pos,
                *ratio,
                *dim,
                *rope_dim,
                *theta,
                *eps,
                *overlap,
            );
        }
        Op::Dsv4CacheWrite {
            src,
            cache,
            rows,
            row,
            cache_off,
            format,
        } => {
            rec.dsv4_cache_write(
                r(*src)?,
                r(*cache)?,
                *rows,
                *row,
                *cache_off,
                matches!(format, Dsv4CacheFormat::Mxfp4),
            );
        }
        Op::Dsv4Indexer {
            q,
            k_cache,
            weights,
            dst,
            cache_off,
            kv_len,
            n_head,
            head_dim,
            top_k,
            scale,
        } => {
            if *head_dim != 128 || *n_head > 64 || *top_k == 0 || *top_k > *kv_len {
                return Err(be(format!(
                    "vulkan Op::Dsv4Indexer requires head_dim=128, n_head<=64 and 0<top_k<=kv_len; \
                     got head_dim={head_dim} n_head={n_head} top_k={top_k} kv_len={kv_len}"
                )));
            }
            let sk = pooled(pool, be_, "dsv4_indexer_scores", *kv_len as usize * 4)?;
            rec.dsv4_indexer(
                r(*q)?,
                r(*k_cache)?,
                r(*weights)?,
                pool[&sk].as_ref(),
                r(*dst)?,
                *cache_off,
                *kv_len,
                *n_head,
                *head_dim,
                *top_k,
                *scale,
            );
        }
        Op::Dsv4Gather {
            raw_cache,
            comp_cache,
            indices,
            dst,
            comp_off,
            pos,
            selected,
            head_dim,
            raw_window,
            ..
        } => {
            if *head_dim != 512 || *raw_window == 0 {
                return Err(be(format!(
                    "vulkan Op::Dsv4Gather requires head_dim=512 and a nonzero raw ring; got \
                     head_dim={head_dim} raw_window={raw_window}"
                )));
            }
            let dummy = pooled(pool, be_, "dsv4_gather_indices_dummy", 4)?;
            let ib = match indices {
                Some(id) => r(*id)?,
                None => pool[&dummy].as_ref(),
            };
            rec.dsv4_gather(
                r(*raw_cache)?,
                r(*comp_cache)?,
                ib,
                r(*dst)?,
                *comp_off,
                *pos,
                *selected,
                *head_dim,
                *raw_window,
                indices.is_some(),
            );
        }
        // Fused per-head RMSNorm + RoPE. Peephole (see `kv_write_peephole`): a QkNormRope whose dst
        // feeds an immediately-following WriteKv is redirected to write the KV cache directly at row
        // `pos` (its WriteKv is skipped). Static uses the pos push constant + `out_base=pos`; Dynamic
        // uses `qk_norm_rope_dyn` (pos from `params`, `out_base_mul` 1 for the cache write, 0 for Q).
        Op::QkNormRope {
            x,
            weight,
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
            // `fused` carries the WRITE ROW (already `pos % cap_rows` for an SWA ring cache —
            // see `kv_write_peephole`); `kcap` is the fused cache's row capacity for the DYNAMIC
            // path, where the row must be derived from the live params pos in-kernel (the ring
            // modulo rides the same params channel as the pos itself — never a baked constant).
            let (out_buf, fused, kcap) = if let Some(&(cache, pos)) = fused_kv_write.get(&op_idx) {
                let row = (*n_head as usize) * (*head_dim as usize);
                (r(cache)?, Some(pos), graph.desc(cache).numel() / row.max(1))
            } else {
                (r(*dst)?, None, 0)
            };
            // #74: fork to the BDA path ONLY when the fused K write lands in the KV cache (which
            // exposes a device address); the un-fused Q-scratch write (Activations, no address) stays
            // on the bound build. `out_buf` stays bound at its write slot in the `_at` method for the
            // store→read barrier. Bit-identical to the bound dispatch.
            let y_addr = out_buf.device_addr();
            match mode {
                RopeMode::Static(rope_pos) => {
                    let ff = match freq_factors {
                        Some(f) => Some(r(*f)?),
                        None => None,
                    };
                    if *x_stride > 0 && ff.is_none() {
                        // Interleaved q+g buffer: read query with stride, skip CopyStrided per head.
                        match y_addr {
                            Some(a) => rec.qk_norm_rope_interleaved_at(
                                r(*x)?,
                                r(*weight)?,
                                out_buf,
                                a,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                rope_pos[&positions.0],
                                fused.unwrap_or(0),
                                *eps,
                                *x_stride as usize,
                            ),
                            None => rec.qk_norm_rope_interleaved(
                                r(*x)?,
                                r(*weight)?,
                                out_buf,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                rope_pos[&positions.0],
                                fused.unwrap_or(0),
                                *eps,
                                *x_stride as usize,
                            ),
                        }
                    } else {
                        match y_addr {
                            Some(a) => rec.qk_norm_rope_at(
                                r(*x)?,
                                r(*weight)?,
                                out_buf,
                                a,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                rope_pos[&positions.0],
                                fused.unwrap_or(0),
                                *eps,
                                ff,
                            ),
                            None => rec.qk_norm_rope(
                                r(*x)?,
                                r(*weight)?,
                                out_buf,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                rope_pos[&positions.0],
                                fused.unwrap_or(0),
                                *eps,
                                ff,
                            ),
                        }
                    } // x_stride > 0
                }
                RopeMode::Dynamic(params) => {
                    // gemma4 proportional RoPE (full-attention layers): the ff divisors bind via
                    // the `qk_norm_rope_dyn_ff` variant — pos still comes from `params`.
                    let ff = match freq_factors {
                        Some(f) => Some(r(*f)?),
                        None => None,
                    };
                    // `out_base_mul` is the 0/1 multiplier the shader scales by pos (then internally
                    // by nheads*hd): 1 → write cache row pos, 0 → write row 0 of the Q scratch.
                    let out_base_mul = usize::from(fused.is_some());
                    if *x_stride > 0 && ff.is_none() {
                        match y_addr {
                            Some(a) => rec.qk_norm_rope_interleaved_dyn_at(
                                r(*x)?,
                                r(*weight)?,
                                *params,
                                out_buf,
                                a,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                out_base_mul,
                                *eps,
                                *x_stride as usize,
                            ),
                            None => rec.qk_norm_rope_interleaved_dyn(
                                r(*x)?,
                                r(*weight)?,
                                *params,
                                out_buf,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                out_base_mul,
                                *eps,
                                *x_stride as usize,
                            ),
                        }
                    } else {
                        let kc = if out_base_mul > 0 { kcap } else { 0 };
                        match y_addr {
                            Some(a) => rec.qk_norm_rope_dyn_at(
                                r(*x)?,
                                r(*weight)?,
                                *params,
                                ff,
                                out_buf,
                                a,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                out_base_mul,
                                *eps,
                                kc,
                            ),
                            None => rec.qk_norm_rope_dyn(
                                r(*x)?,
                                r(*weight)?,
                                *params,
                                ff,
                                out_buf,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                out_base_mul,
                                *eps,
                                kc,
                            ),
                        }
                    } // x_stride > 0
                }
            }
        }
        // Standalone RoPE (llama: no q/k-norm; INTERLEAVED pairs — llama.cpp's ROPE_TYPE_NORM).
        // An f16 dst takes the same fused shapes as QkNormRope: the kv_write peephole redirects
        // the K write into the cache, and Dynamic mode replays via `rope_f16_dyn` (pos from
        // `params`). The legacy f32 in-place form stays static-only. No freq_factors kernel.
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
            neox,
            backward,
            ..
        } => {
            let f16_out = matches!(graph.desc(*dst).dtype, infr_core::DType::F16);
            // YaRN freq_factors (deepseek2 k_pe) are only supported on the STATIC f32 in-place
            // path (`rec.rope` → the `rope_ff` kernel build). The f16-out forms (`rope_f16`,
            // fused KV write) and the record-once `_dyn` variants have no ff kernel — reject
            // rather than silently drop the divisors.
            if freq_factors.is_some() && (f16_out || !matches!(mode, RopeMode::Static(_))) {
                return Err(be(
                    "vulkan adapter: standalone Rope with freq_factors unsupported on this path",
                ));
            }
            // Same for NEOX (deepseek32's indexer): only the static f32 builds carry `-DNEOX`.
            // Dropping it silently would rotate the wrong element pairs — fluent nonsense, no
            // error — so refuse instead.
            if *neox && (f16_out || !matches!(mode, RopeMode::Static(_))) {
                return Err(be(
                    "vulkan adapter: standalone NEOX Rope unsupported on this path (f16-out / \
                     record-once decode have no -DNEOX kernel build)",
                ));
            }
            // Same again for BACKWARD (deepseek4's attention-output de-rope, `Op::Rope::backward`):
            // only the static f32 NORM builds carry `-DBACKWARD`. A dropped sign is the same class
            // of silent-wrong-numbers as a dropped pairing, so refuse.
            if *backward && (f16_out || *neox || !matches!(mode, RopeMode::Static(_))) {
                return Err(be(
                    "vulkan adapter: backward (de-)Rope unsupported on this path (f16-out / NEOX / \
                     record-once decode have no -DBACKWARD kernel build)",
                ));
            }
            let ff = freq_factors.map(&r).transpose()?;
            let (out_buf, fused) = if let Some(&(cache, pos)) = fused_kv_write.get(&op_idx) {
                (r(cache)?, Some(pos))
            } else {
                (r(*dst)?, None)
            };
            // #74: fork to the BDA path when the fused K write lands in the KV cache (device address
            // present); the un-fused Q-scratch write stays bound. `out_buf` stays bound at its write
            // slot in the `_at` method for the store→read barrier.
            let y_addr = out_buf.device_addr();
            match mode {
                RopeMode::Static(rope_pos) => {
                    if f16_out {
                        match y_addr {
                            Some(a) => rec.rope_f16_at(
                                r(*x)?,
                                out_buf,
                                a,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                rope_pos[&positions.0],
                                fused.unwrap_or(0),
                            ),
                            None => rec.rope_f16(
                                r(*x)?,
                                out_buf,
                                *rows as usize,
                                *n_head as usize,
                                *head_dim as usize,
                                *rope_dim as usize,
                                *theta,
                                rope_pos[&positions.0],
                                fused.unwrap_or(0),
                            ),
                        }
                    } else {
                        rec.rope(
                            r(*x)?,
                            r(*dst)?,
                            *rows as usize,
                            *n_head as usize,
                            *head_dim as usize,
                            *rope_dim as usize,
                            *theta,
                            rope_pos[&positions.0],
                            ff,
                            *neox,
                            *backward,
                        );
                    }
                }
                RopeMode::Dynamic(params) => {
                    if !f16_out {
                        return Err(be(
                            "vulkan adapter: dynamic decode f32 in-place Rope unsupported",
                        ));
                    }
                    match y_addr {
                        Some(a) => rec.rope_f16_dyn_at(
                            r(*x)?,
                            *params,
                            out_buf,
                            a,
                            *rows as usize,
                            *n_head as usize,
                            *head_dim as usize,
                            *rope_dim as usize,
                            *theta,
                            usize::from(fused.is_some()),
                        ),
                        None => rec.rope_f16_dyn(
                            r(*x)?,
                            *params,
                            out_buf,
                            *rows as usize,
                            *n_head as usize,
                            *head_dim as usize,
                            *rope_dim as usize,
                            *theta,
                            usize::from(fused.is_some()),
                        ),
                    }
                }
            }
        }
        // GQA scaled-dot-product attention over the f16 KV cache. Dynamic mode (decode, rows==1,
        // causal, 1/√hd) uses `attention_kv_dyn` (pos_offset + kv_len from `params`). Static mode
        // keeps the full lowering: FlashAttention-2 for prefill, `attention_kv` otherwise.
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
            sinks,
        } => {
            let (rows, kv_len, nh, nkv, hd, pos) = (
                *rows as usize,
                *kv_len as usize,
                *n_head as usize,
                *n_kv as usize,
                *head_dim as usize,
                *pos as usize,
            );
            // Planar Q8_0 KV cache, chosen PER-SIDE (K and V independent). The coalesced f16
            // flash/split kernels can't read Q8, so a Q8 side routes decode through the native-Q8
            // split (attn_partial_{k,v}q8) and prefill through a dequant→f16 prepass. A DECOUPLED
            // Q8 side forces STATIC decode (see decode_eligible); the Dynamic branch below sees
            // only coupled K==V==Q8 (or no Q8 at all).
            let k_q8 = matches!(graph.desc(*k_cache).dtype, infr_core::DType::Q8_0);
            let v_q8 = matches!(graph.desc(*v_cache).dtype, infr_core::DType::Q8_0);
            let kv_q8 = k_q8 && v_q8; // coupled (the Dynamic-branch kernels' q8 variant)
                                      // Planar Q8 scales region base = total cache elements (K and V caches share numel).
            let cap = graph.desc(*k_cache).numel();
            // Row capacity of the bound cache. An SWA layer's cache may be a RING of fewer rows
            // than the live kv_len (window + ubatch, see the runner's ring sizing): position j
            // then lives at row j % att_cap_rows — the scalar/split kernels derive the same
            // mapping from their `cap` push constant; the coopmat tile kernels (flash/nonfa) and
            // the rows-batched mrows tier don't, so they're gated back to the split path once
            // kv_len (padded, for nonfa's 256-row tiles) outruns the rows actually present. On a
            // full-context cache kv_len <= att_cap_rows always, so none of these gates move.
            let att_cap_rows = cap / (nkv * hd).max(1);
            // Per-head attention sinks (deepseek4, `Op::Attention::sinks`) live in exactly ONE
            // kernel — `attention_kv.comp`'s -DSINKS build. Nothing else in the tier ladder below
            // (flash, non-FA coopmat, split-K, the Q8/BDA/params twins) knows about them, and a
            // tier that quietly ignored the sink would produce a correct-looking softmax over the
            // wrong denominator. So a sinks op takes the scalar path unconditionally, and the
            // shapes that path cannot serve are refused here rather than silently mis-served.
            // `decode_eligible` already forces sinks graphs off the record-once replay tape.
            if let Some(sk) = sinks {
                // The scalar sinks build reads `q` and both caches as f16 (the seam's own
                // producer→consumer dtype flow). Every other KV dtype reaches the tier ladder
                // through the dequant→f16 prepass below, which this early return skips — so
                // require f16 outright rather than hand the kernel bytes it will misread.
                let f16 = |t: infr_core::tensor::TensorId| {
                    matches!(graph.desc(t).dtype, infr_core::DType::F16)
                };
                if !f16(*k_cache) || !f16(*v_cache) {
                    return Err(be(
                        "vulkan adapter: Op::Attention with sinks needs an f16 K/V cache — the \
                         sinks build (attention_kv_sinks) has no quant read and no dequant prepass",
                    ));
                }
                let window = match mask {
                    AttnMask::Causal => 0,
                    AttnMask::SlidingWindow(w) => *w,
                    AttnMask::Canvas { .. } => {
                        return Err(be(
                            "vulkan adapter: Op::Attention with sinks under AttnMask::Canvas has \
                             no kernel — the scalar sinks path carries no bidirectional `lo`",
                        ));
                    }
                };
                rec.attention_kv_sinks(
                    r(*q)?,
                    r(*k_cache)?,
                    r(*v_cache)?,
                    r(*sk)?,
                    r(*dst)?,
                    rows,
                    kv_len,
                    nh,
                    nkv,
                    hd,
                    pos,
                    window,
                    *scale,
                    cap,
                );
                return Ok(());
            }
            if let RopeMode::Dynamic(params) = mode {
                // Eligibility guarantees rows==1. Scale rides a push constant (gemma4 uses 1.0;
                // 0.0 → kernel default 1/√hd) and SWA windows ride the window-aware prologue +
                // partial kernel, so gemma-family decode replays too.
                let window = match mask {
                    AttnMask::Causal => 0usize,
                    AttnMask::SlidingWindow(w) => *w,
                    // Record-once decode replay is rows==1 causal-only (DiffusionGemma canvas
                    // denoise is a rows=C static forward, never eligible for this path) — never
                    // actually reached with Canvas; 0 is an arbitrary but harmless placeholder.
                    AttnMask::Canvas { .. } => 0usize,
                };
                let def_scale = (*scale - 1.0 / (hd as f32).sqrt()).abs() < 1e-6;
                let pscale = if def_scale { 0.0 } else { *scale };
                // Flash-decoding split-K: the scalar attention_kv_dyn launches only nh workgroups
                // and rescans the whole cache per token — decode slowed ~4x from kv 100→600 on the
                // seam replay path. `chunk`/`n_chunks` are push constants (dispatch structure), so
                // bake the WORST CASE from the bound cache's capacity: chunks past the live kv_len
                // (read from `params` at replay) produce zero-weight partials (m=-3e38, l=0) the
                // combine ignores. attn_combine's shared array caps n_chunks at 512 → scale chunk
                // with capacity. Scratch is plan-held (`transient`), so replay reuses it.
                // NOTE `cap_rows` (row capacity, chunk sizing) vs `cap` (ELEMENT capacity, the
                // planar-Q8 scales-region base push constant): shadowing the latter with the
                // former here is exactly the bug that garbled Q8 KV under record-once replay.
                let cap_rows = graph.desc(*k_cache).numel() / (nkv * hd);
                // Baked MIN chunk (the policy's `min_chunk`, rising only when the capacity would
                // exceed the ATTN_MAX_CHUNKS scratch/combine cap); the kernel derives the effective
                // adaptive chunk from the live kv_len each token, so one recorded plan serves every
                // depth.
                let chunk =
                    infr_core::tier::baked_chunk(cap_rows, ATTN_MAX_CHUNKS, ATTN_SPLIT.min_chunk);
                let n_chunks = infr_core::tier::n_chunks(cap_rows, chunk);
                if hd % 4 == 0 && hd <= 512 && n_chunks > 1 {
                    // ONE prologue + ONE scratch set per (nh, hd, chunk, n_chunks, window) key:
                    // the first Dynamic attention op of a shape records/allocates; every later
                    // same-shape layer reuses (kv_len is per-token, and the layers' attention ops
                    // are serialized by dataflow — pm/pl/pacc hazards are ordinary RAW/WAW
                    // barriers). Gemma alternates SWA/global (and gemma4-12b hd 256/512), so a
                    // handful of keys coexist per execute.
                    let key = |c: &DynAttnCtx| {
                        c.nh == nh
                            && c.chunk == chunk
                            && c.n_chunks == n_chunks
                            && c.hd == hd
                            && c.window == window
                    };
                    if !dyn_args.iter().any(key) {
                        let args = be_.alloc_uninit(16, BufferUsage::Activations)?;
                        rec.attn_live_prologue(*params, args.as_ref(), nh, chunk, window);
                        dyn_args.push(DynAttnCtx {
                            nh,
                            chunk,
                            n_chunks,
                            hd,
                            window,
                            args,
                            pm: be_.alloc_uninit(nh * n_chunks * 4, BufferUsage::Activations)?,
                            pl: be_.alloc_uninit(nh * n_chunks * 4, BufferUsage::Activations)?,
                            pacc: be_
                                .alloc_uninit(nh * n_chunks * hd * 4, BufferUsage::Activations)?,
                        });
                    }
                    let ctx = dyn_args.iter().find(|c| key(c)).unwrap();
                    // KV u64/BDA (#74): fork the HOT record-once decode split-K read to the pointer
                    // twin when both KV caches expose a device address (they always do — this is the
                    // resident decode path, KV is never the dequant scratch here). The caches persist
                    // across replay, so their addresses are stable in the baked push. Bit-identical to
                    // the bound dispatch (kept dual-arm: attn_partial's .comp also serves prefill).
                    let kb = r(*k_cache)?;
                    let vb = r(*v_cache)?;
                    match (kb.device_addr(), vb.device_addr()) {
                        (Some(ka), Some(va)) => rec.attention_kv_split_dynac_at(
                            r(*q)?,
                            kb,
                            vb,
                            ka,
                            va,
                            r(*dst)?,
                            ctx.pm.as_ref(),
                            ctx.pl.as_ref(),
                            ctx.pacc.as_ref(),
                            *params,
                            ctx.args.as_ref(),
                            nh,
                            nkv,
                            hd,
                            chunk,
                            n_chunks,
                            pscale,
                            window,
                            kv_q8,
                            cap,
                        ),
                        _ => rec.attention_kv_split_dynac(
                            r(*q)?,
                            kb,
                            vb,
                            r(*dst)?,
                            ctx.pm.as_ref(),
                            ctx.pl.as_ref(),
                            ctx.pacc.as_ref(),
                            *params,
                            ctx.args.as_ref(),
                            nh,
                            nkv,
                            hd,
                            chunk,
                            n_chunks,
                            pscale,
                            window,
                            kv_q8,
                            cap,
                        ),
                    }
                } else {
                    // KV u64/BDA (#74): fork the record-once decode read to the pointer twin when both
                    // KV buffers expose a device address (they always do on decode). The caches persist
                    // across the replay, so their addresses are stable in the push. Kept dual-arm.
                    let kb = r(*k_cache)?;
                    let vb = r(*v_cache)?;
                    match (kb.device_addr(), vb.device_addr()) {
                        (Some(ka), Some(va)) => rec.attention_kv_dyn_at(
                            r(*q)?,
                            kb,
                            vb,
                            ka,
                            va,
                            *params,
                            r(*dst)?,
                            1,
                            nh,
                            nkv,
                            hd,
                            pscale,
                            window,
                            kv_q8,
                            cap,
                        ),
                        _ => rec.attention_kv_dyn(
                            r(*q)?,
                            kb,
                            vb,
                            *params,
                            r(*dst)?,
                            1,
                            nh,
                            nkv,
                            hd,
                            pscale,
                            window,
                            kv_q8,
                            cap,
                        ),
                    }
                }
            } else {
                // Dequant a quantized KV side → a transient f16 scratch, then run the f16 attention on
                // it. Q8 only dequants on PREFILL (rows>1; decode reads Q8 natively via the coalesced
                // split/scalar kernels — the scalar O(kv²) path TDR-hangs at depth). The mainline
                // low-bit quants (q4_0/…/iq4_nl) have no native read, so they dequant on EVERY pass
                // (prefill + decode). The persistent cache stays quantized (footprint preserved).
                let kdt = graph.desc(*k_cache).dtype;
                let vdt = graph.desc(*v_cache).dtype;
                // Lever 1 (kv-decode-perf-levers #1): DECODE (rows==1) of a mainline low-bit KV cache
                // reads the compact block format INLINE in `attn_partial` (no dequant→f16 prepass,
                // no f16 scratch — the ¼-traffic win at long context). Gate it to the case the
                // split-K decode path definitely handles inline: rows==1, K==V one inline-decode
                // format, hd%4≤512, and the split-K tier is the one that fires (this predicate is
                // EXACTLY `split_ok` at rows==1 for a full-context causal/SWA cache — see below).
                // Conservatively keep the prepass for ring / capacity-limited caches (`ring_past` /
                // `cap_short`) and for `hd` the split path rejects — those still run correctly (if
                // slower) through the f16 scratch. PREFILL (rows>1) always keeps the prepass (coopmat
                // needs f16; small-m suffix split reads the f16 scratch) — this only touches decode.
                let dec_swa_window = match mask {
                    AttnMask::SlidingWindow(w) => *w,
                    _ => 0usize,
                };
                let dec_swa_base = if dec_swa_window > 0 {
                    (pos + 1).saturating_sub(dec_swa_window)
                } else {
                    0
                };
                let dec_span = kv_len - dec_swa_base;
                let dec_chunk = infr_core::tier::adaptive_chunk(dec_span, &ATTN_SPLIT);
                // Only exclude a RING cache (SWA past its wrap) — its RROW-wrapped inline read is
                // correct but out of the kv_addr_parity coverage, so keep the prepass there for now.
                // (`cap_short` — the nonfa 256-row PREFILL tile-pad guard — is irrelevant to the
                // rows==1 split-K decode read, which reads exactly kv_len keys ≤ att_cap_rows; gating
                // on it here just disabled inline whenever the cache was sized tightly.)
                let dec_ring_past = dec_swa_window > 0 && kv_len > att_cap_rows;
                let decode_inline = rows == 1
                    && hd % 4 == 0
                    && hd <= 512
                    && !dec_ring_past
                    && dec_span > dec_chunk
                    && kdt == vdt
                    && is_kv_inline_decode(kdt)
                    // OPT-IN (default OFF). MEASURED: the per-element inline `dq()` decode is a
                    // growing DECODE REGRESSION vs the dequant→f16 prepass on RDNA3 (Qwen3-8B, Q4_0
                    // KV, 7900 XTX: d1024 0.86x, d4096 0.72x, d8192 0.61x). `dq()` per element does
                    // UNCOALESCED scalar block reads (kv_word → global_load_b32 per word) and
                    // RE-DECODES the block f16 scale for every element, so the KV-path ALU/latency
                    // outweighs the ¼-byte-traffic saving — the prepass's one coalesced dequant pass
                    // + coalesced f16 attention read is faster. A real win needs a BLOCK-AMORTIZED
                    // coalesced vec4 decoder (scale-once, b128 nibble reads), not per-element dq().
                    // Kept behind INFR_KV_INLINE=1 for that follow-up + A/B; the shader arm, builds,
                    // recorder plumbing and bit-identity are all in place and gate-clean.
                    && be_.cfg().kv.inline_decode;
                // Lever 2 (INFR_FLASH_DEQUANT, kv-decode-perf-levers): dequant-in-flash. When the
                // coopmat flash-prefill geometry holds AND the KV cache is a supported quant format
                // (K==V ∈ {Q4_0,Q4_1,Q5_0,Q5_1,Q8_0}), SKIP the dequant_kv_f16 prepass and let
                // attn_flash_partial's staged builds decode the GGUF block cache in the QK/PV stage
                // (native_decode `dqblk`, coalesced scale-once). `flash_geom` is exactly flash_ok's
                // geometry minus the quant-eff check; with the prepass skipped and q8-eff forced off
                // below, flash_ok reduces to `flash_geom`, so the flash arm is guaranteed taken.
                // Read ONCE and share: `flash_geom` (below) and `nc_fa_ok` (further down) both floor
                // on this row count and MUST agree — two reads risked a divergence trap.
                let flash_min_rows: usize = be_.cfg().kernels.vulkan.flash_min_rows;
                let flash_hd = hd == 128
                    || (hd == 256
                        && be_.max_shared_memory_bytes() >= crate::FLASH_HD256_BM16_SHARED);
                let flash_geom = (rows >= 64 || (rows >= flash_min_rows && kv_len >= 8192))
                    && flash_hd
                    // The established hd128 shaders directly read a padded final BN=64 tile, so
                    // require that padded width to fit the cache. The hd256 BM16 shader instead
                    // stages only its capacity-short final tile with explicit bounds checks, hence
                    // it is safe when the cache ends at an arbitrary row (notably synthetic d100k).
                    && (if hd == 256 {
                        kv_len <= att_cap_rows
                    } else {
                        kv_len.div_ceil(64) * 64 <= att_cap_rows
                    })
                    && matches!(mask, AttnMask::Causal)
                    && (*scale - 1.0 / (hd as f32).sqrt()).abs() < 1e-6
                    && be_.caps().f16_coopmat()
                    && matches!(graph.tensors[q.0 as usize].kind, TensorKind::Internal)
                    && matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal);
                let flash_deq_fmt = if flash_geom
                    && hd == 128
                    && kdt == vdt
                    // GGUF-block low-bit set only (native_decode dqblk substrate). Q8_0 (planar
                    // cache) and iq4_nl (codebook) stay on the dequant_kv_f16 prepass — see
                    // attn_flash_partial.comp / build.rs.
                    && matches!(
                        kdt,
                        infr_core::DType::Q4_0
                            | infr_core::DType::Q4_1
                            | infr_core::DType::Q5_0
                            | infr_core::DType::Q5_1
                    )
                    && be_.cfg().kernels.vulkan.flash_dequant
                {
                    Some(kdt)
                } else {
                    None
                };
                let deq_k = flash_deq_fmt.is_none()
                    && ((k_q8 && rows > 1) || (is_kv_prepass(kdt) && !decode_inline));
                let deq_v = flash_deq_fmt.is_none()
                    && ((v_q8 && rows > 1) || (is_kv_prepass(vdt) && !decode_inline));
                // Ring cache: only att_cap_rows rows exist — dequant the WHOLE ring (element i →
                // element i keeps the ring layout, so the f16 scratch is read with the same
                // row-modulo mapping as the cache itself). Identity on full-context caches.
                let ne = kv_len.min(att_cap_rows) * nkv * hd;
                // Expand a KV side into the f16 scratch: native Q8 (planar), f32 cast (store_f16),
                // else the shared quant/bf16 dequant (dequant_kv_f16).
                // #74: the dequant READERS read the SOURCE cache (`r(*k_cache)?`/`r(*v_cache)?`, always
                // a KvCache with a device address) by pointer. dequant_q8_f16/dequant_kv_f16 are BDA-only
                // (slice 5) — they read `src.device_addr()` internally; `src` still stays bound at the
                // read slot so the store→dequant-read barrier survives. The f32 cast (store_f16) writes
                // the f16 SCRATCH `sc` (Activations, no address) → bound; turbo dequant is deferred/bound.
                let kc_key = if deq_k {
                    let k = pooled(pool, be_, "kvdeq_k", ne * 2)?;
                    let sc = pool[&k].as_ref();
                    let src = r(*k_cache)?;
                    if k_q8 {
                        rec.dequant_q8_f16(src, sc, ne, cap);
                    } else if matches!(kdt, infr_core::DType::F32) {
                        rec.store_f16(src, sc, ne, 0);
                    } else if is_turbo(kdt) {
                        rec.dequant_turbo_f16(kdt, src, sc, ne);
                    } else {
                        rec.dequant_kv_f16(kdt, src, sc, ne);
                    }
                    Some(k)
                } else {
                    None
                };
                let vc_key = if deq_v {
                    let k = pooled(pool, be_, "kvdeq_v", ne * 2)?;
                    let sc = pool[&k].as_ref();
                    let src = r(*v_cache)?;
                    if v_q8 {
                        rec.dequant_q8_f16(src, sc, ne, cap);
                    } else if matches!(vdt, infr_core::DType::F32) {
                        rec.store_f16(src, sc, ne, 0);
                    } else if is_turbo(vdt) {
                        rec.dequant_turbo_f16(vdt, src, sc, ne);
                    } else {
                        rec.dequant_kv_f16(vdt, src, sc, ne);
                    }
                    Some(k)
                } else {
                    None
                };
                // A dequanted side reads the f16 scratch; native Q8 read only when not dequanted.
                // Dequant-in-flash reads the quant cache DIRECTLY (neither f16 scratch nor native-Q8
                // planar read), so force q8-eff off — flash_ok then reduces to `flash_geom`.
                let k_q8_eff = flash_deq_fmt.is_none() && k_q8 && !deq_k;
                let v_q8_eff = flash_deq_fmt.is_none() && v_q8 && !deq_v;
                // Prefill, causal, hd==128, standard 1/√hd scale: FlashAttention-2 (split-K
                // online softmax, no materialized [m,kv] scores). The flash kernel hardcodes
                // 1/√hd and reads/writes ceil(rows/64)*64 q/dst rows, so guard the scale and
                // require 64-row-padded buffers (Internal). Flash IS the K-tile shape — K/V
                // streamed once per row-tile, padded rows only waste ALU — so at depth it beats
                // the per-row split kernel well below a full 64-row tile. Measured crossover
                // (0.6B, 7900 XTX): flash wins from rows>=24 at kv>=8192 (pp24@8k +13%,
                // pp32@16k +55%) but LOSES to split at kv=4096 up through rows=32 — hence the
                // two-tier floor. INFR_FLASH_MIN_ROWS overrides the deep-kv tier (A/B); it is read
                // once above (`flash_min_rows`) and shared with `flash_geom`.
                // attn_flash*/attn_qk*/attn_pv* are ALL coopmat SPIR-V (see the module doc / audit
                // for this fallback ladder) — `flash_ok`/`nonfa_ok` additionally require
                // `caps.f16_coopmat` so a device without the feature never gets one
                // dispatched (RADV silently executes coopmat SPIR-V even with the feature bit
                // off, so "it ran" isn't proof; other drivers fault). Gated false, both fall
                // through to `split_ok`/the final `else` — `attention_kv_split`/`attention_kv`,
                // already the scalar decode/short-prefill path, dispatch with no coopmat and no
                // row-count ceiling, so they're a correct (if slower) fallback for ANY `rows`.
                let flash_ok = flash_geom && !(k_q8_eff || v_q8_eff);
                // Coopmat prefill shmem-staging mode (Levers 2 & 5). Dequant wins over the pure-f16
                // stage; both need the flash geometry. INFR_FLASH_STAGE stages the f16 KV (cache, or
                // the prepass f16 scratch on a quant model) into shmem — the reuse experiment.
                let flash_stage = if let Some(dt) = flash_deq_fmt {
                    crate::recorder::FlashStage::Dequant(dt)
                } else if flash_geom && hd == 128 && be_.cfg().kernels.vulkan.flash_stage {
                    crate::recorder::FlashStage::Stage
                } else {
                    crate::recorder::FlashStage::Off
                };
                // #74 slice 4 resurrection (INFR_KV_COOPMAT_BDA=1, opt-in, default OFF): read the KV
                // cache in the coopmat flash prefill by 64-bit device address instead of bound SSBOs.
                // Only meaningful on the Off (f16) path — the Stage/Dequant builds have no BDA arm — so
                // gate it there. On RADV/RDNA3 this REGRESSES prefill (~0.8x, no saddr from a
                // buffer_reference coopmat base), so it stays off by default; kept for other silicon.
                let kv_coopmat_bda = hd == 128
                    && matches!(flash_stage, crate::recorder::FlashStage::Off)
                    && be_.cfg().kv.coopmat_bda;
                // Prefill at hd≠128 (qwen35/gemma hd=256): the non-FA coopmat pipeline
                // (attn_qk → softmax → attn_pv) is hd-general and ~an order faster than the scalar
                // attention_kv. Needs 64-row-padded q/dst (Internal buffers are row-padded), so
                // require both to be Internal.
                // DiffusionGemma canvas denoise (docs/diffusion-gemma.md): bidirectional, fixed
                // `[lo, kv_len)` reach per row — neither the flash kernel (Causal-only, already
                // excluded above) nor `attention_prefill_nonfa`'s per-row causal-end window
                // understand that shape, so gate BOTH off it and force the split-K path below
                // (which DOES carry a `lo` override) even though rows=C(=256) would otherwise
                // pick one of these tiers. Perf pass deferred (Phase 4).
                let canvas_lo = match mask {
                    AttnMask::Canvas { lo } => Some(*lo),
                    _ => None,
                };
                let nonfa_ok = !flash_ok
                    && canvas_lo.is_none()
                    && rows >= 64
                    // attn_qk reads K tiles up to ceil(kv_len/256)*256 rows (masked in softmax) —
                    // a ring cache past its wrap has neither those rows nor a direct position →
                    // row mapping the tile loads could use, so fall to the split path (which has
                    // the row-modulo mapping). Full-context caches only hit this within 256 rows
                    // of their very end, where the split path is equally correct.
                    && kv_len.div_ceil(256) * 256 <= att_cap_rows
                    && hd % 64 == 0
                    && hd <= 512
                    && !(k_q8_eff || v_q8_eff)
                    && be_.caps().f16_coopmat()
                    && matches!(graph.tensors[q.0 as usize].kind, TensorKind::Internal)
                    && matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal);
                // NON-COOPMAT flash tier (`attn_nc_fa.comp`, the attention companion of the
                // `nc_mmq`/`nc_fma` GEMM tier): flash/nonfa above are coopmat SPIR-V, so a device
                // without the feature (Intel Arc/ANV) previously ran ALL prefill attention on the
                // scalar per-row `attention_kv` (31% of knob pp512 on Qwen3-14B) or, for rows<64 /
                // ring-past shapes, the split-K path. This shared-memory fma tile (no subgroup
                // ops, ≤54 KB shared) takes the flash tier's row floor; unlike flash/nonfa it
                // handles SWA windows AND ring caches (attn_partial's `cap` row-modulo mapping),
                // so ring-past prefill rides it too. f16 KV only — quantized KV was already
                // dequanted to the f16 ring-layout scratch above at rows>1 (`k/v_q8_eff` false by
                // construction), and the gate keeps the exclusion explicit. No Internal-buffer
                // requirement: the kernel guards pad-row reads/writes itself. Exercisable on
                // coopmat hardware via INFR_NO_COOPMAT=1; INFR_NO_NC_FA=1 disables the arm (A/B
                // against the scalar/split floor).
                let nc_fa_ok = !flash_ok
                    && !nonfa_ok
                    && !be_.caps().f16_coopmat()
                    && (rows >= 64 || (rows >= flash_min_rows && kv_len >= 8192))
                    && canvas_lo.is_none()
                    && hd % 4 == 0
                    && hd <= 512
                    && !(k_q8_eff || v_q8_eff)
                    && be_.cfg().kernels.vulkan.nc_fa;
                // Decode (rows==1) AND small-m suffix prefill (rows 2..63, below the flash/nonfa
                // floor): flash-decoding split-K — each (row, head)'s KV range splits across ~32
                // chunks of workgroups instead of the scalar attention_kv's rows*nh (= nh at
                // decode, ~16 workgroups on a 96-CU GPU; the SHORT TURN suffix shape measured
                // ~2.5ms/layer scalar = 97% of the forward at d4096). attn_partial handles any
                // hd%4==0 ≤ 512 (hd=128 fast path, general path above), any scale (push constant;
                // gemma4 uses 1.0), SWA windows, and per-row causal ends (row r attends
                // ≤ pos + r). Native Q8 reads are decode-only (rows>1 already dequanted to the
                // f16 scratch above, so k/v_q8_eff are false there by construction).
                // Rows-BATCHED split tier (rows 12..flash-floor at deep kv): one workgroup per
                // (head, chunk) streams K/V once per 4-row group through attn_partial_mrows_c256
                // (7KB LDS). Measured wins over the per-row grid: pp12/16/20@d16384
                // 741->799/822->924/882->1012 t/s, pp16@8k a wash — below rows=12 or kv=8192
                // the per-row grid's extra workgroups fill the DRAM queue better than the
                // bandwidth saving pays. hd<=128 (one q vec4 per lane per row); q8 never reaches
                // rows>1 (dequanted above). INFR_MROWS_ATTN=1 forces the batched tier on (tests /
                // A/B), INFR_NO_MROWS_ATTN forces it off.
                //
                // The pair is ONE asymmetric tri-state field (`kernels.vulkan.mrows_attn`):
                // `Some(false)` = the `NO_` key, which wins unconditionally; `Some(true)` = the
                // opt-in, which bypasses the rows/kv_len heuristic; `None` = let the heuristic
                // decide. Reproduces the old `is_err()`/`is_ok()` pair exactly.
                let mrows_attn = be_.cfg().kernels.vulkan.mrows_attn;
                let batched_attn = rows >= 2
                    && hd <= 128
                    && kv_len <= att_cap_rows // the mrows kernel has no ring row mapping
                    && canvas_lo.is_none()
                    && !(k_q8_eff || v_q8_eff)
                    && mrows_attn != Some(false)
                    && ((rows >= 12 && kv_len >= 8192) || mrows_attn == Some(true));
                // The batched kernel stages chunk scores in 4KB of LDS → chunk 256; the per-row
                // grid keeps the adaptive ~32-chunks policy.
                //
                // Canvas (DiffusionGemma denoise, slice 7 comparative-attribution against the
                // fork's fused flash-attn oracle): the ordinary `kv_len/32` policy is tuned for
                // DEEP decode contexts (splitting a multi-thousand kv_len across ~32 workgroups
                // keeps each one short); at canvas's shallow kv_len (prompt + canvas, a few
                // hundred) it still floors to 64, forcing 5 chunks at kv_len=283 —
                // `attention_kv_split`'s `pm`/`pl`/`pacc` partials are [rows(=canvas), nh,
                // n_chunks, hd], so every extra chunk is 256(canvas)×nh more partial-softmax
                // writes AND combine-side reads, pure overhead vs. a genuine single-pass flash
                // kernel (which the canvas mask shape can't use — see `canvas_lo` above).
                //
                // Probed the chunk COUNT directly (`INFR_CANVAS_CHUNK_N`, kv_len=283, Vulkan0):
                // n=2 (the `split_ok`-minimum, `kv_len>chunk` strictly) REGRESSED — attn_partial
                // 11.7k->13.8us (occupancy loss: workgroups halve from 5 to 2, so each one's
                // longer serial K/V loop outweighs the partial-buffer saving) even though
                // attn_combine dropped (1.36k->0.84k). n=3 was the sweet spot: attn_partial
                // ~11.2k (flat vs baseline) + attn_combine ~1.0k (-27%) = net -6% on the
                // attn_partial+combine pair vs the old 64-floor's n=5. n=4 landed between the two
                // (small net win, less than n=3). Default 3; overridable for future re-tuning at
                // other canvas lengths.
                // SWA chunk-grid base: a sliding-window layer's rows attend only the union span
                // `[max(0, pos+1-window), kv_len)` (row r's window start only moves UP with r), so
                // chunk THAT span instead of `[0, kv_len)` — at pp4@d4096 a gemma SWA layer
                // (window 512) drops from 33 chunks (28 of them empty: launched, barriered,
                // zero-partial-written, combine-read) to ~9. The split-K partial pass is
                // fixed-cost-per-workgroup bound at small m (E2B attn_partial scaled exactly with
                // nh·n_chunks·rows across depths), so empty chunks cost like full ones.
                // `attn_partial.comp`'s static branch derives the SAME base from its existing
                // pos/window push constants (no layout change); the rows-batched mrows kernel and
                // Canvas don't, so they keep the full-range grid (base 0).
                let swa_window = match mask {
                    AttnMask::SlidingWindow(w) => *w,
                    _ => 0,
                };
                let swa_base = if swa_window > 0 && !batched_attn && canvas_lo.is_none() {
                    (pos + 1).saturating_sub(swa_window)
                } else {
                    0
                };
                let span = kv_len - swa_base;
                // SWA ring layer past its wrap (kv_len outran the ring's rows): the coopmat
                // prefill tiers were gated off above (no ring row mapping in their tile loads),
                // so the split-K path must take EVERY row count here — the scalar attention_kv
                // fallback at prefill row counts (1024 rows x deep kv) is a guaranteed TDR.
                let ring_past = swa_window > 0 && kv_len > att_cap_rows;
                // Capacity-limited prefill on a FULL-CONTEXT (non-ring) cache: `nonfa_ok` above
                // pads its K-tile read up to a whole 256-row tile (`kv_len.div_ceil(256)*256`),
                // masked in softmax but still a real read — unsafe once that padded width exceeds
                // the cache's declared row capacity, exactly like `ring_past` but without an SWA
                // window. A session's `max_ctx` is sized with only a small slack above the actual
                // prompt length (room for generation, not for 256-row alignment), and the ordinary
                // chunked dense-prefill loop (ubatch-sized chunks) rarely lands a chunk boundary
                // there — but MTP's single un-chunked whole-prompt VERIFY (`generate_mtp_spec_core`'s
                // prime step) is ONE dispatch at rows == kv_len == the whole prompt, so it can park
                // kv_len within one tile-pad of `max_ctx`'s slack (observed: Qwen3.5-4B-MTP bench at
                // d>=3584, kv_len=3591 padding to 3840 against a 3637-row cache — device-lost TDR
                // when this fell through to the scalar `attention_kv` fallback below). Route it to
                // the split-K path instead: `attn_partial`'s `cap` push constant already carries a
                // correct (identity, since kv_len < att_cap_rows always holds) row mapping.
                let cap_short = kv_len.div_ceil(256) * 256 > att_cap_rows;
                let chunk = if canvas_lo.is_some() && kv_len >= 2 {
                    let n = CANVAS_CHUNK_N.clamped(be_.cfg().kernels.vulkan.canvas_chunk_n);
                    kv_len.div_ceil(n).min(kv_len - 1).max(1)
                } else if batched_attn {
                    256
                } else if (ring_past || cap_short) && rows >= 64 {
                    // Large-rows ring/capacity-limited prefill: the pm/pl/pacc partials are [rows,
                    // nh, n_chunks, hd] — the ordinary ~32-chunk policy would balloon them (1024
                    // rows x 32 chunks x hd 256 ≈ 1 GB), and the span is already bounded (window +
                    // rows for a ring, kv_len itself here), so a few big chunks keep the scratch
                    // ~100s of MB with plenty of workgroups (nh * n_chunks * rows).
                    512
                } else if rows == 1
                    && k_q8_eff
                    && v_q8_eff
                    && hd == 256
                    && be_.cfg().kernels.vulkan.q8_decode_chunk1024
                {
                    infr_core::tier::adaptive_chunk(span, &Q8_HD256_DECODE_SPLIT)
                } else {
                    infr_core::tier::adaptive_chunk(span, &ATTN_SPLIT)
                };
                // Bound the chunk COUNT, not just its size: attn_combine.comp indexes its
                // `shared float wexp[1024]` by chunk, so n_chunks (= span.div_ceil(chunk), computed
                // in the `split_ok` arm below) must stay <= 1024. Raising the chunk floor to
                // span.div_ceil(1024) caps n_chunks at 1024. This only bites above ~524k keys
                // (span > 512*1024, reachable under INFR_KV_OVERFLOW huge ctx); for every realistic
                // span span.div_ceil(1024) <= the branch value above, so `chunk` — and the recorded
                // dispatch — is byte-identical. (The Dynamic path caps count the same way, ~2531.)
                let chunk = split_k_chunk_count_cap(span, chunk);
                // Canvas forces the split-K tier regardless of row count (see `canvas_lo` above) —
                // `attn_partial` carries the fixed `lo` override this mask needs; flash/nonfa don't.
                let split_ok = (rows < 64 || canvas_lo.is_some() || ring_past || cap_short)
                    && span > chunk
                    && hd % 4 == 0
                    && hd <= 512
                    && (rows == 1 || !(k_q8_eff || v_q8_eff));
                if flash_ok {
                    let mpad = rows.div_ceil(64) * 64;
                    // Pooled split partials (fully written before the combine reads them) — one
                    // set serves every layer instead of n_layer live copies (~1GB each at 8B p8k).
                    let po = pooled(pool, be_, "flash_po", 8 * mpad * nh * hd * 4)?;
                    let pm = pooled(pool, be_, "flash_pm", 8 * mpad * nh * 4)?;
                    let pl = pooled(pool, be_, "flash_pl", 8 * mpad * nh * 4)?;
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    // Opt-in coopmat-prefill BDA (INFR_KV_COOPMAT_BDA): fork to the `_bda` builds only
                    // when the flag is set AND both KV buffers carry a device address (KvCache always
                    // does since #74). kc/vc stay bound (inert) so the store→read hazard edge holds.
                    let kv_addr = match (kv_coopmat_bda, kcb.device_addr(), vcb.device_addr()) {
                        (true, Some(ka), Some(va)) => Some((ka, va)),
                        _ => None,
                    };
                    if hd == 256 {
                        rec.attention_prefill_flash_hd256(
                            r(*q)?,
                            kcb,
                            vcb,
                            r(*dst)?,
                            pool[&po].as_ref(),
                            pool[&pm].as_ref(),
                            pool[&pl].as_ref(),
                            rows,
                            kv_len,
                            nh,
                            nkv,
                            pos,
                            k_q8 && v_q8 && kv_len >= 65_536,
                            kv_addr,
                        );
                    } else {
                        rec.attention_prefill_flash(
                            r(*q)?,
                            kcb,
                            vcb,
                            r(*dst)?,
                            pool[&po].as_ref(),
                            pool[&pm].as_ref(),
                            pool[&pl].as_ref(),
                            rows,
                            kv_len,
                            nh,
                            nkv,
                            hd,
                            pos,
                            flash_stage,
                            kv_addr,
                        );
                    }
                } else if nonfa_ok {
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                        AttnMask::Canvas { .. } => unreachable!("nonfa_ok excludes Canvas"),
                    };
                    let mpad = rows.div_ceil(64) * 64;
                    let kv_pad = kv_len.div_ceil(256) * 256;
                    // Pooled scores scratch [nh, mpad, kv_pad] f16 + split-K PV partials (≤8
                    // splits) f32 — ~80MB per attention op, fully written before read (attn_qk
                    // fills every [mpad, kv_pad] row; PV partials are written per split before
                    // the reduce), and one set serves every same-shape layer.
                    let s = pooled(pool, be_, "nonfa_s", nh * mpad * kv_pad * 2)?;
                    let pv = pooled(pool, be_, "nonfa_pv", 8 * mpad * nh * hd * 4)?;
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    rec.attention_prefill_nonfa(
                        r(*q)?,
                        kcb,
                        vcb,
                        r(*dst)?,
                        pool[&s].as_ref(),
                        pool[&pv].as_ref(),
                        rows,
                        kv_len,
                        nh,
                        nkv,
                        hd,
                        pos,
                        window,
                        *scale,
                    );
                } else if nc_fa_ok {
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                        AttnMask::Canvas { .. } => unreachable!("nc_fa_ok excludes Canvas"),
                    };
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    rec.attention_prefill_nc_fa(
                        r(*q)?,
                        kcb,
                        vcb,
                        r(*dst)?,
                        rows,
                        kv_len,
                        nh,
                        nkv,
                        hd,
                        pos,
                        window,
                        *scale,
                        cap,
                    );
                } else if split_ok {
                    // `canvas_lo` (computed above) rides a SEPARATE param into
                    // `attention_kv_split` — `window` stays the ordinary causal/SWA value (0 for
                    // Canvas; unused by the shader when `canvas_lo` is set).
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                        AttnMask::Canvas { .. } => 0,
                    };
                    let n_chunks = span.div_ceil(chunk);
                    // Scratch scales with rows (rows*nh partial planes); rows==1 keeps the old
                    // decode sizes, so the hot decode pool entries are unchanged.
                    let pm = pooled(pool, be_, "split_pm", rows * nh * n_chunks * 4)?;
                    let pl = pooled(pool, be_, "split_pl", rows * nh * n_chunks * 4)?;
                    let pacc = pooled(pool, be_, "split_pacc", rows * nh * n_chunks * hd * 4)?;
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    // KV u64/BDA (#74): fork the split-K partial read to the pointer twin when both KV
                    // buffers expose a device address. `kcb`/`vcb` MAY be the dequant scratch (rows>1
                    // quantized KV prefill) — those are Activations with no address and fall through to
                    // the bound arm, which is why this kernel stays dual-arm. Bit-identical either way.
                    // Lever 1: mainline low-bit KV decode → the INLINE-decode build (`kv_ml`), which
                    // reads the compact cache directly (no f16 scratch was allocated: `decode_inline`
                    // drove `deq_k`/`deq_v` false, so `kcb`/`vcb` ARE the quant cache with a device
                    // address → the `(Some, Some)` arm). `decode_inline` ⟹ `split_ok` here, so this
                    // is the arm that fires. None for f16/Q8/prefill.
                    let kv_ml = if decode_inline { Some(kdt) } else { None };
                    match (kcb.device_addr(), vcb.device_addr()) {
                        (Some(ka), Some(va)) => rec.attention_kv_split_at(
                            r(*q)?,
                            kcb,
                            vcb,
                            ka,
                            va,
                            r(*dst)?,
                            pool[&pm].as_ref(),
                            pool[&pl].as_ref(),
                            pool[&pacc].as_ref(),
                            rows,
                            pos,
                            kv_len,
                            nh,
                            nkv,
                            hd,
                            chunk,
                            n_chunks,
                            *scale,
                            window,
                            canvas_lo,
                            k_q8_eff,
                            v_q8_eff,
                            cap,
                            batched_attn,
                            kv_ml,
                        ),
                        _ => rec.attention_kv_split(
                            r(*q)?,
                            kcb,
                            vcb,
                            r(*dst)?,
                            pool[&pm].as_ref(),
                            pool[&pl].as_ref(),
                            pool[&pacc].as_ref(),
                            rows,
                            pos,
                            kv_len,
                            nh,
                            nkv,
                            hd,
                            chunk,
                            n_chunks,
                            *scale,
                            window,
                            canvas_lo,
                            k_q8_eff,
                            v_q8_eff,
                            cap,
                            batched_attn,
                        ),
                    }
                } else {
                    let window = match mask {
                        AttnMask::Causal => 0,
                        AttnMask::SlidingWindow(w) => *w,
                        AttnMask::Canvas { .. } => {
                            return Err(be(
                                "vulkan adapter: AttnMask::Canvas requires the split-K attention \
                                 path (split_ok) — the scalar attention_kv fallback doesn't carry \
                                 a bidirectional `lo` override",
                            ));
                        }
                    };
                    let kcb = match &kc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*k_cache)?,
                    };
                    let vcb = match &vc_key {
                        Some(k) => pool[k].as_ref(),
                        None => r(*v_cache)?,
                    };
                    // KV u64/BDA (#74): fork to the pointer-read twin when both KV buffers expose a
                    // device address (KvCache always does; the dequant scratch may not). Bit-identical
                    // to the bound dispatch below, which is why this kernel stays dual-arm.
                    match (kcb.device_addr(), vcb.device_addr()) {
                        (Some(ka), Some(va)) => rec.attention_kv_at(
                            r(*q)?,
                            kcb,
                            vcb,
                            ka,
                            va,
                            r(*dst)?,
                            rows,
                            kv_len,
                            nh,
                            nkv,
                            hd,
                            pos,
                            window,
                            *scale,
                            k_q8_eff,
                            v_q8_eff,
                            cap,
                        ),
                        _ => rec.attention_kv(
                            r(*q)?,
                            kcb,
                            vcb,
                            r(*dst)?,
                            rows,
                            kv_len,
                            nh,
                            nkv,
                            hd,
                            pos,
                            window,
                            *scale,
                            k_q8_eff,
                            v_q8_eff,
                            cap,
                        ),
                    }
                }
            }
        }
        // Per-head RMSNorm == rmsnorm over rows*n_head rows of head_dim (gemma4's weightless
        // V-norm passes a ones weight → out = x/rms; llama4's post-rope weightless Q/K L2-norm
        // runs in-place on the f16 rope scratch — `x`'s dtype picks the f16 or f32 kernel, `w`
        // (the ones-vector weight) stays f32 either way). `weight: None` (deepseek4's Q norm) is
        // the genuinely weightless build with no `w` binding at all.
        Op::QkNorm {
            x,
            weight,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
            ..
        } => {
            let (rows_n, dim) = ((*rows * *n_head) as usize, *head_dim as usize);
            let f16 = matches!(graph.desc(*x).dtype, infr_core::DType::F16);
            match (weight, f16) {
                (Some(w), true) => rec.rmsnorm_f16(r(*x)?, r(*w)?, r(*dst)?, rows_n, dim, *eps),
                (Some(w), false) => rec.rmsnorm(r(*x)?, r(*w)?, r(*dst)?, rows_n, dim, *eps),
                (None, false) => rec.rmsnorm_nw(r(*x)?, r(*dst)?, rows_n, dim, *eps),
                // No -DNO_WEIGHT -DF16IO build: deepseek4's weightless Q norm runs on the f32
                // wq_b output, and a variant nothing dispatches is a variant nothing tests.
                (None, true) => {
                    return Err(be(
                        "vulkan adapter: Op::QkNorm { weight: None } over an f16 x has no kernel \
                         — the weightless per-head RMSNorm is built f32-only (rmsnorm_nw)",
                    ));
                }
            }
        }
        // Fused per-head RMSNorm + SiLU gate multiply (qwen35 DeltaNet z-gate) — one dispatch
        // instead of QkNorm→GatedAct's two, removing the read-after-write barrier between them.
        Op::GatedRmsNorm {
            x,
            weight,
            gate,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
        } => {
            rec.rmsnorm_gate(
                r(*x)?,
                r(*weight)?,
                r(*gate)?,
                r(*dst)?,
                (*rows * *n_head) as usize,
                *head_dim as usize,
                *eps,
            );
        }
        // Qwen3-Next SSM: depthwise causal conv + SiLU (rolling conv `state` mutated in place).
        Op::Conv1dSilu {
            x,
            weight,
            state,
            dst,
            rows,
            channels,
            kernel,
        } => {
            // Batch (rows ≥ kconv-1): all rows·cc outputs in parallel + a history rebuild pass,
            // instead of the token-serial history walk. Decode keeps the sequential kernel.
            let cv = if *rows as usize >= (*kernel as usize).saturating_sub(1).max(2) {
                Recorder::conv1d_silu_batch
            } else {
                Recorder::conv1d_silu
            };
            cv(
                rec,
                r(*x)?,
                r(*weight)?,
                r(*state)?,
                r(*dst)?,
                *rows as usize,
                *channels as usize,
                *kernel as usize,
            );
        }
        // Qwen3-Next gated DeltaNet recurrence (persistent `state` S mutated in place).
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
            ..
        } => {
            // Prefill AND multi-row batches (rows ≥ 2): the chunkwise delta rule processes up to 32
            // tokens per state traversal (matmuls + a triangular solve) instead of the token-serial
            // recurrence — the difference between rows and ⌈rows/32⌉ sequential state sweeps. The
            // partial-chunk case (rows < 32, one chunk) is the SAME shader path prefill already uses
            // for its tail chunk (c = min(32, rows-base) is masked throughout deltanet_chunked.comp /
            // deltanet_prep.comp), so this isn't a new code path — just a lower row count triggering
            // it. Threshold was 32 (i.e. only prefill); MTP verify runs a batched trunk forward over
            // only m≈2-19 rows (committed_suffix ++ drafted, prefix-diffed) and was stranded on the
            // sequential kernel below 32, capping the whole spec-decode cycle (verify was 68-69% of
            // it). Measured crossover (gemm_bench.rs microbench, qwen35 dims nv=nk=16 kd=vd=128,
            // isolated dispatch cost): chunked_split beats sequential at EVERY row count from 1..32
            // (2.0x at rows=1 up to 8.2x at rows=32; rows=2..8, the MTP-verify range, is 3.2-4.4x).
            // rows=1 stays on the sequential kernel anyway — this is plain single-token decode, out
            // of scope here (decode has its own hot path; the extra transient-buffer allocs below
            // aren't free at decode's much higher call frequency, unlike prep+scan's per-call cost
            // which the microbench doesn't include either but easily amortizes at rows≥2). The
            // real-workload win: MTP 4B, `deltanet` total dropped 116.3ms→~28ms (prep+gates+scan) at
            // an identical (bit-exact) accept pattern across a 128-token run. The default is the
            // SPLIT form (parallel prep/gates passes + a light state-coupled scan; the monolithic
            // kernel duplicated the prep work per column block). INFR_NO_DN_CHUNK forces sequential
            // (A/B, also the m==1-decode behavior); INFR_NO_DN_SPLIT keeps the chunked math but in
            // the monolithic kernel (A/B).
            let (rows_, nv_, nk_, kd_, vd_) = (
                *rows as usize,
                *n_vhead as usize,
                *n_khead as usize,
                *head_k as usize,
                *head_v as usize,
            );
            let chunked = rows_ >= 2 && be_.cfg().kernels.vulkan.dn_chunk;
            // DEFAULT prefill path: the token-serial scan with the state column register-resident
            // (norm + gates + seq). The chunked delta rule was believed to win by doing ⌈rows/32⌉
            // state sweeps instead of `rows`, but counted out it does NOT save arithmetic (~420M vs
            // ~402M FMA per layer at Ornith dims: the chunk form trades state sweeps for the
            // triangular solve + D/Dq contractions) — it only shortens the dependency chain, and it
            // pays with LDS state, unroll-blocking runtime trip counts, ~96 workgroup barriers, and
            // only nv·(vd/8)=256 workgroups (~2.7/CU on a 96-CU part → nothing to hide latency).
            // The serial form keeps state in registers, has zero barriers (single-subgroup
            // workgroups; the kd-contractions are subgroupAdd), and launches nv·vd=2048 workgroups.
            // Measured on Ornith-35B pp512: deltanet_scan 31.7ms → deltanet_seq 6.0ms.
            // Needs kd == 128 (RPL=4 register shards) — anything else falls back to chunked below.
            // INFR_DN_CHUNK_SCAN=1 forces the old chunked-split path (A/B).
            //
            // NOTE no f16_coopmat gate: coopmat was only ever needed by deltanet_prep's D/Dq
            // contractions, and this path never forms them. So DeltaNet prefill no longer needs
            // coopmat at all — non-coopmat GPUs get the fast path too instead of being routed to
            // the monolithic chunked kernel.
            if chunked
                && kd_ == 128
                && vd_.is_multiple_of(crate::recorder::DN_SEQ_NCOL)
                && be_.cfg().kernels.vulkan.dn_chunk_scan
                && be_.cfg().kernels.vulkan.dn_split
            {
                // alloc_uninit: every slot the scan reads is written by norm/gates first.
                let kn =
                    be_.alloc_uninit((rows_ * nk_ * kd_ * 4).max(4), BufferUsage::Activations)?;
                let qn =
                    be_.alloc_uninit((rows_ * nk_ * kd_ * 4).max(4), BufferUsage::Activations)?;
                let bet = be_.alloc_uninit((rows_ * nv_ * 4).max(4), BufferUsage::Activations)?;
                let dec = be_.alloc_uninit((rows_ * nv_ * 4).max(4), BufferUsage::Activations)?;
                rec.deltanet_seq_split(
                    r(*q)?,
                    r(*k)?,
                    r(*v)?,
                    r(*b)?,
                    r(*a)?,
                    r(*a_coef)?,
                    r(*dt_bias)?,
                    r(*state)?,
                    r(*dst)?,
                    kn.as_ref(),
                    qn.as_ref(),
                    bet.as_ref(),
                    dec.as_ref(),
                    rows_,
                    nv_,
                    nk_,
                    kd_,
                    vd_,
                    *eps,
                );
                transient.extend([kn, qn, bet, dec]);
                return Ok(());
            }
            // deltanet_chunked_split's prep pass (deltanet_prep.comp) is the ONLY DeltaNet shader
            // using coopmat (the D/Dq dot matrices); deltanet_chunked (monolithic) and deltanet
            // (sequential) are both scalar. `!caps.f16_coopmat` routes to the still-chunked
            // (still fast, just not split-prep) `deltanet_chunked` kernel below instead of the
            // sequential one — chunked math doesn't require coopmat, only this particular prep
            // kernel's implementation does.
            if chunked && be_.caps().f16_coopmat() && be_.cfg().kernels.vulkan.dn_split {
                let nchunk = rows_.div_ceil(32);
                // alloc_uninit: every slot the scan reads is written by prep/gates first.
                let kn =
                    be_.alloc_uninit((rows_ * nk_ * kd_ * 4).max(4), BufferUsage::Activations)?;
                let qn =
                    be_.alloc_uninit((rows_ * nk_ * kd_ * 4).max(4), BufferUsage::Activations)?;
                let dk =
                    be_.alloc_uninit((nchunk * nk_ * 1024 * 4).max(4), BufferUsage::Activations)?;
                let dq =
                    be_.alloc_uninit((nchunk * nk_ * 1024 * 4).max(4), BufferUsage::Activations)?;
                let bg =
                    be_.alloc_uninit((nchunk * nv_ * 32 * 4).max(4), BufferUsage::Activations)?;
                let gg =
                    be_.alloc_uninit((nchunk * nv_ * 32 * 4).max(4), BufferUsage::Activations)?;
                rec.deltanet_chunked_split(
                    r(*q)?,
                    r(*k)?,
                    r(*v)?,
                    r(*b)?,
                    r(*a)?,
                    r(*a_coef)?,
                    r(*dt_bias)?,
                    r(*state)?,
                    r(*dst)?,
                    kn.as_ref(),
                    qn.as_ref(),
                    dk.as_ref(),
                    dq.as_ref(),
                    bg.as_ref(),
                    gg.as_ref(),
                    rows_,
                    nv_,
                    nk_,
                    kd_,
                    vd_,
                    *eps,
                );
                transient.extend([kn, qn, dk, dq, bg, gg]);
            } else {
                // Strided DeltaNet: when q==k==v (all same source buffer), derive stride from
                // dimensions: 2*nk*kd + nv*vd.  The runner only forms this graph for Vulkan
                // single-token decode; the equality check keeps malformed/custom graphs on the
                // packed fallback.
                let strided = *q == *k && *k == *v && be_.cfg().kernels.vulkan.delta_strided;
                if strided {
                    let stride = 2 * nk_ * kd_ + nv_ * vd_;
                    rec.deltanet_strided(
                        r(*q)?,
                        r(*k)?,
                        r(*v)?,
                        r(*b)?,
                        r(*a)?,
                        r(*a_coef)?,
                        r(*dt_bias)?,
                        r(*state)?,
                        r(*dst)?,
                        rows_,
                        nv_,
                        nk_,
                        kd_,
                        vd_,
                        *eps,
                        stride,
                    );
                } else {
                    let dn = if chunked {
                        Recorder::deltanet_chunked
                    } else {
                        Recorder::deltanet
                    };
                    dn(
                        rec,
                        r(*q)?,
                        r(*k)?,
                        r(*v)?,
                        r(*b)?,
                        r(*a)?,
                        r(*a_coef)?,
                        r(*dt_bias)?,
                        r(*state)?,
                        r(*dst)?,
                        rows_,
                        nv_,
                        nk_,
                        kd_,
                        vd_,
                        *eps,
                    );
                } // if *src_stride > 0
            }
        }
        Op::Kda {
            qkv,
            forget,
            beta,
            a,
            dt_bias,
            state,
            dst,
            rows,
            n_head,
            head_dim,
            eps,
            lower_bound,
        } => rec.kda(
            r(*qkv)?,
            r(*forget)?,
            r(*beta)?,
            r(*a)?,
            r(*dt_bias)?,
            r(*state)?,
            r(*dst)?,
            *rows as usize,
            *n_head as usize,
            *head_dim as usize,
            *eps,
            *lower_bound,
        ),
        // Elementwise gemma logit softcap `y = cap·tanh(x/cap)` (in-place safe).
        Op::Softcap { x, dst, cap, n } => {
            rec.softcap(r(*x)?, r(*dst)?, *cap, *n as usize);
        }
        Op::EmbedGather {
            ids,
            table,
            dst,
            rows,
            ne,
            scale,
        } => {
            let dt = graph.desc(*table).dtype;
            rec.embed_gather(
                dt,
                r(*table)?,
                r(*ids)?,
                r(*dst)?,
                *rows as usize,
                *ne as usize,
                *scale,
            );
        }
        Op::GatherI32 {
            ids,
            table,
            dst,
            rows,
            ne,
        } => {
            let dt = graph.desc(*table).dtype;
            if dt != infr_core::DType::I32 {
                return Err(be(format!(
                    "vulkan adapter: Op::GatherI32 copies integers verbatim — its table must be \
                     I32, got {dt:?}"
                )));
            }
            rec.gather_i32(r(*table)?, r(*ids)?, r(*dst)?, *rows as usize, *ne as usize);
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
            let cand = pooled(pool, be_, "sample_cand", 2 * 256 * *top_k as usize * 4)?;
            match mode {
                // Record-once/self-advancing path (single-shot `execute` OR chained
                // `execute_chain` — the same recording serves both): `u` is a 64-slot ring keyed
                // by the self-advancing `params[0]`, matching `id_log`'s ring geometry.
                RopeMode::Dynamic(params) => {
                    rec.sample_topk_chain(
                        r(*x)?,
                        pool[&cand].as_ref(),
                        r(*u)?,
                        *params,
                        r(*dst)?,
                        *n as usize,
                        *top_k as usize,
                        *temp,
                        *top_p,
                    );
                }
                // Classic per-execute recording: `u` is the plain 1-float buffer.
                RopeMode::Static(_) => {
                    rec.sample_topk(
                        r(*x)?,
                        pool[&cand].as_ref(),
                        r(*u)?,
                        r(*dst)?,
                        *n as usize,
                        *top_k as usize,
                        *temp,
                        *top_p,
                    );
                }
            }
        }
        Op::Argmax { x, dst, n, rows } => {
            let part = pooled(pool, be_, "argmax_part", 512 * 4)?;
            rec.argmax(
                r(*x)?,
                pool[&part].as_ref(),
                r(*dst)?,
                *n as usize,
                *rows as usize,
            );
        }
        Op::ArgmaxProb {
            x,
            dst_id,
            dst_prob,
            n,
        } => {
            // 3*256 f32 scratch: (max, idx-bits, sum_exp) triples — see argmax_prob.comp.
            let part = pooled(pool, be_, "argmax_prob_part", 768 * 4)?;
            rec.argmax_prob(
                r(*x)?,
                pool[&part].as_ref(),
                r(*dst_id)?,
                r(*dst_prob)?,
                *n as usize,
            );
        }
        // MoE FFN (single token): router GEMV → GPU-resident top-k (gating-dependent weighting,
        // ×scale) → fused multi-slot expert SwiGLU (gate/up share the row, down reads each slot's
        // act) → weighted accumulate. Mirrors the production GPU-resident decode path
        // (transformer.rs) and the CPU `Op::MoeFfn` interpreter. Expert banks must use an
        // id-native quant format.
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
            ep_band,
            exp_probs_b,
            n_expert_groups,
            n_expert_groups_used,
            swiglu_clamp,
            expert_ids,
        } => {
            // Router gating (softmax vs sigmoid) and renormalization are `moe_topk` push-constant
            // flags — see its shader doc. `weight_before` (llama4: the routing weight scales the
            // expert INPUT) is folded in as a post-GEMM/GEMV, pre-activation scale of the gate/up
            // outputs (`moe_weight_scale`, exact since gate/up are linear — see its shader doc),
            // with the corresponding accumulate/reduce stage told `prescaled` so it doesn't also
            // apply the weight to the (already-weighted) down output.
            let gating_u32 = match gating {
                infr_core::graph::MoeGating::Softmax => 0u32,
                infr_core::graph::MoeGating::Sigmoid => 1u32,
                infr_core::graph::MoeGating::SqrtSoftplus => 2u32,
            };
            // exp_probs_b: resolve if present, else bind a dummy zero buffer (the shader ignores
            // it when `has_bias == 0`). One float is enough — the shader reads only indices
            // < n_expert, which push constants clamp before the bias buffer is touched.
            let bias_buf: &dyn Buffer = if let Some(epb) = exp_probs_b {
                r(*epb)?
            } else {
                &*be_.alloc_uninit(4, BufferUsage::Weights)?
            };
            let has_bias = exp_probs_b.is_some();
            // DeepSeek V4 hash routing: the selection arrives pre-gathered, so `moe_topk` copies
            // it in and skips its whole selection phase (see `moe_topk.comp`). Same dummy-bind
            // shape as `bias_buf` when absent. The two SELECTION-only inputs have nothing left to
            // select on a hash layer, so refuse rather than ignore them.
            let hash_buf: &dyn Buffer = if let Some(hid) = expert_ids {
                r(*hid)?
            } else {
                &*be_.alloc_uninit(4, BufferUsage::Activations)?
            };
            let hash = expert_ids.is_some();
            if hash && (has_bias || *n_expert_groups > 1) {
                return Err(be(
                    "vulkan adapter: MoeFfn expert_ids (hash routing) supplies the selection — \
                     exp_probs_b and group routing have nothing to select",
                ));
            }
            if swiglu_clamp.is_some() && (matches!(act, Activation::Sigmoid) || *weight_before) {
                return Err(be(
                    "vulkan adapter: MoeFfn swiglu_clamp needs a gated SiLU/GELU with \
                     output-side weighting",
                ));
            }
            let (ne, n_expert, n_used, nff) = (
                *ne as usize,
                *n_expert as usize,
                *n_used as usize,
                *n_ff_exp as usize,
            );
            // Expert-parallel (multi-GPU EP) band: under EP this rank's bound expert banks
            // (`gate_exps`/`up_exps`/`down_exps`) hold ONLY experts `[ep_lo, ep_lo+n_expert_local)`
            // of the global `n_expert`. The router GEMV + `moe_topk` below still run over the FULL
            // `n_expert` (the router weight is replicated, so every rank makes the identical global
            // top-k selection); a `moe_ep_band_remap` right after top-k rewrites the selected ids
            // into this rank's LOCAL shard indices (out-of-band → id 0, weight 0) so the expert
            // GEMV/GEMM stages index the local bank (`n_expert_local` distinct experts) and the
            // weighted combine drops the assignments this rank doesn't own. The EP backend then
            // all-reduces every rank's partial `dst`. `None` (single-device) ⇒ `n_expert_local ==
            // n_expert`, no remap ⇒ byte-identical to before this field existed.
            let (ep_lo, n_expert_local) =
                ep_band.map_or((0u32, n_expert), |(b, nl)| (b, nl as usize));
            // Fused gate_up_exps stores BOTH roles per expert ([ne, 2*nff, n_expert]) — the id-native
            // dtype check below reads `up_exps` too, but the call site binds it to the SAME handle as
            // `gate_exps` when fused (never separately read), so the dtype check still holds.
            // `down_exps` is ALWAYS width-`nff` per expert (unaffected by a fused gate/up) — its own
            // stride, used by the per-token path below (the batched path never sees `fused_gate_up`,
            // guarded above, so reusing `stride` there for gate/up/down alike stays correct).
            let stride = if *fused_gate_up { 2 * nff } else { nff } * ne;
            let down_stride = nff * ne;
            let (gdt, udt, ddt) = (
                graph.desc(*gate_exps).dtype,
                graph.desc(*up_exps).dtype,
                graph.desc(*down_exps).dtype,
            );
            if !(crate::linear::native_id_kernel_name(gdt).is_some()
                && crate::linear::native_id_kernel_name(udt).is_some()
                && crate::linear::native_id_kernel_name(ddt).is_some())
            {
                return Err(be(
                    "vulkan adapter: MoeFfn expert banks need an id-native quant format",
                ));
            }
            let rows = graph.desc(*x).numel() / ne;
            // ── BATCHED MoE FFN (rows > 1: the seam's prefill chunks AND DiffusionGemma's canvas
            // denoise): GPU-resident expert routing (top-k → bucket count/scan/scatter, all
            // on-GPU) + a prologue that writes per-expert INDIRECT dispatch args from the counts —
            // so the whole expert loop records with NO host readback (the bespoke path downloads
            // counts mid-graph to size its GEMMs; indirect dispatch replaces that). Gate/up AND
            // down each independently accept Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q5_1/Q2_K/Q3_K (split OR
            // fused gate/up; Q5_0 is what the shipped diffusiongemma-26B-A4B-it-GGUF's down banks
            // use; Q5_1 is what the shipped gemma-4-26B-A4B-it-GGUF's down banks use (29/30
            // layers); Q5_K/Q6_K is what unsloth-dynamic Qwen3.6-MoE quants mix into most layers'
            // gate/up/down banks; Q2_K/Q3_K is Llama-4-Scout's shipped gate/up (Q2_K) and down
            // (Q3_K) — every dtype routes through the SAME dtype-generic `matmul_mmq_experts` dp4a
            // kernel table, so there's no per-role dtype restriction beyond that shared set —
            // MXFP4/NVFP4 joined it via the IQ4_NL signed-codebook treatment, IQ2_S/IQ3_S (the
            // UD-IQ3_S expert pair) via shared-LUT grid staging. The remaining non-mmq formats
            // (IQ1_*/IQ2_XXS/IQ2_XS/IQ3_XXS, ternary, floats — see `MOE_MMQ_DTYPES`'s DELIBERATE
            // EXCLUSIONS doc) route through the per-token path below regardless of role.
            //
            // Fused gate_up (diffusion-gemma): native block formats can't be split at an
            // arbitrary row offset without block-alignment gymnastics (Q4_K's superblocks straddle
            // 256-element runs), so instead of two GEMMs at different `w_off`s we run ONE GEMM
            // over the whole [ne, 2*nff] expert slice (`gu_width` below) and split gate/up in the
            // activation kernel — `gelu_mul_fused`/`silu_mul_fused` already implement exactly that
            // split (gate half first, up half second per row), reused unchanged from the per-token
            // path. Split gate/up (qwen3moe): unchanged two-GEMM shape.
            //
            // Gated on `rows > moe_small_m_threshold()` (not `rows > 1`): tiny prefill chunks (e.g.
            // `pp4`) take the small-m fast path below instead — see its doc for why.
            // ALSO gated on `caps.i8_dot`: the batched path's expert GEMM (`matmul_mmq_experts`,
            // below) is dp4a int8 ONLY — no coopmat, no scalar sibling. A device lacking packed
            // integer dot product falls through to the small-m path unconditionally instead (its
            // `linear_native_id_multi` id-indexed GEMVs are plain dequant-in-shader, no dp4a) —
            // correct for any `rows`, just without the batched path's cross-token weight-bank
            // reuse (a real perf cost for large-batch MoE prefill on such hardware).
            if rows > moe_small_m_threshold(be_) && be_.caps().i8_dot {
                // `matmul_mmq_experts` is dtype-generic (dp4a mmq kernels exist for every dtype in
                // `infr_core::tensor::MOE_MMQ_DTYPES` — that's why `down_ok` already covered the
                // wider set) and role-agnostic (gate/up/down all call the SAME function, just with
                // a different weight handle and stride) — so gate/up get the SAME coverage as
                // down. The uncovered grid i-quants (IQ1_*/IQ2_XXS/IQ2_XS/IQ3_XXS), ternary (TQ*),
                // and float banks have no dp4a-mmq kernel (deliberate — see `MOE_MMQ_DTYPES`'s
                // EXCLUSIONS doc) and stay out; those experts keep the per-token path below.
                //
                // `MOE_MMQ_DTYPES` is the SINGLE SOURCE OF TRUTH this gate and infr-llama/seam/
                // runner.rs's `moe_mmq_ok` both derive from — see its doc for why a mismatch here
                // either silently falls back to per-token prefill (this gate stricter) or compiles
                // a batched graph this adapter then rejects (that gate stricter), and
                // `moe_mmq_drift_test` (this crate's test suite) for the guard that catches drift.
                let mmq_ok = infr_core::tensor::moe_mmq_ok;
                let down_ok = mmq_ok(ddt);
                let act_ok = if *fused_gate_up {
                    matches!(act, Activation::Silu | Activation::Gelu)
                } else {
                    // qwen3moe/llama4 (the split-gate_up batched callers) ship SiLU only; a non-
                    // fused GELU batched kernel doesn't exist (no caller needs it today).
                    matches!(act, Activation::Silu)
                };
                if !(mmq_ok(gdt) && mmq_ok(udt) && down_ok && act_ok) {
                    return Err(be(format!(
                        "vulkan adapter: batched MoeFfn needs a dtype in \
                         infr_core::tensor::MOE_MMQ_DTYPES for gate/up/down \
                         (+ SiLU, or GELU when fused_gate_up) \
                         (got gate={gdt:?} up={udt:?} \
                         down={ddt:?} act={act:?} fused={fused_gate_up})"
                    )));
                }
                let alu = |n: usize| be_.alloc_uninit(n.max(4), BufferUsage::Activations);
                let n_pairs = rows * n_used;
                let xb = r(*x)?;
                let yb = r(*dst)?;
                // diffusion-gemma's router reads a DIFFERENTLY normalized/scaled row than the
                // experts (see the `Op::MoeFfn` doc); qwen3moe binds the same handle as `x`.
                let rxb = r(*router_x)?;

                // ── SINGLE-DISPATCH-PER-STAGE pipeline over the PACKED bucket layout. The old
                // shape ran every stage per expert (8-way waves of indirect dispatches): ~1050
                // dispatches and ~110 barriers per layer, and at pp512 the launch/serialization
                // overhead — not GEMM math — was ~60% of MoE prefill GPU time (quant 25%,
                // gather+scatter+silu 34%). Instead every stage runs ONCE over all n_pairs
                // packed rows: the expert GEMMs put the expert id on gl_WorkGroupID.y
                // (segment = offsets[e]..+counts[e], worst-case row tiles exit immediately),
                // gather fuses into the activation quant, and a per-token reduce over `inv_pos`
                // (assignment → bucket slot, fixed slot order — deterministic, no atomics)
                // replaces the KWAY per-set dst copies + adds. ~13 dispatches per layer, no
                // indirect args, no host readback.
                //
                // The GEMM As stage reads up to 63 rows past a segment end (garbage, results
                // discarded) — pad the packed row dimension so the LAST expert's overread stays
                // in-bounds.
                let npad = n_pairs.div_ceil(64) * 64 + 64;
                let logits = alu(rows * n_expert * 4)?;
                let ids = alu(n_pairs * 4)?;
                let wts = alu(n_pairs * 4)?;
                // Bucket arrays index the LOCAL expert shard under EP (ids are remapped to
                // `[0, n_expert_local)` right after top-k); `n_expert_local == n_expert` off EP.
                // Device-zeroed by `rec.zero(counts,…)` below (bucket_count accumulates), so the
                // non-blocking `alu` alloc is correct — the blocking zeroing alloc would add one
                // pointless host `queue_wait_idle` per MoE layer for a buffer re-zeroed on-GPU anyway.
                let counts = alu(n_expert_local * 4)?;
                let offsets = alu(n_expert_local * 4)?;
                let fill = alu(n_expert_local * 4)?;
                let bucket_rows = alu(n_pairs * 4)?;
                let bucket_wts = alu(n_pairs * 4)?;
                let inv_pos = alu(n_pairs * 4)?;

                // Packed scratch, POOLED — one set serves every MoE layer in the graph. `gu_width`
                // is the fused gate|up GEMM's output width (2*nff) vs split's (nff); `ue` (the
                // split path's separate up-projection buffer) is unused/unallocated when fused.
                let gu_width = if *fused_gate_up { 2 * nff } else { nff };
                let qa = pooled(pool, be_, "moe_qa", npad * ne)?;
                let qda = pooled(pool, be_, "moe_qda", npad * (ne / 32) * 2)?;
                let qsa = pooled(pool, be_, "moe_qsa", npad * (ne / 32) * 2)?;
                let ge = pooled(pool, be_, "moe_ge", npad * gu_width * 4)?;
                let ue = if *fused_gate_up {
                    None
                } else {
                    Some(pooled(pool, be_, "moe_ue", npad * nff * 4)?)
                };
                let ae = pooled(pool, be_, "moe_ae", npad * nff * 4)?;
                let dqa = pooled(pool, be_, "moe_dqa", npad * nff)?;
                let dda = pooled(pool, be_, "moe_dda", npad * (nff / 32) * 2)?;
                let dsa = pooled(pool, be_, "moe_dsa", npad * (nff / 32) * 2)?;
                let ye = pooled(pool, be_, "moe_ye", npad * ne * 4)?;

                // Router logits for all rows (on `router_x`, NOT `x`), then GPU routing.
                let rdt = graph.desc(*router).dtype;
                let rw = r(*router)?;
                if native_dense_supported(rdt) {
                    rec.linear_native(rdt, rw, 0, rxb, logits.as_ref(), rows, ne, n_expert);
                } else if matches!(rdt, infr_core::DType::F32) {
                    rec.linear_f32(rw, rxb, logits.as_ref(), rows, ne, n_expert);
                } else {
                    rec.linear(rw, rxb, logits.as_ref(), rows, ne, n_expert);
                }
                rec.moe_topk(
                    logits.as_ref(),
                    ids.as_ref(),
                    wts.as_ref(),
                    bias_buf,
                    rows,
                    n_expert,
                    n_used,
                    *scale,
                    gating_u32,
                    *norm_w,
                    has_bias,
                    *n_expert_groups,
                    *n_expert_groups_used,
                    hash_buf,
                    hash,
                );
                // EP: rewrite the global top-k ids into this rank's local shard indices before the
                // bucket sort (out-of-band → id 0, weight 0 — those assignments bucket harmlessly
                // and the weighted combine drops them). No-op off EP.
                if ep_band.is_some() {
                    rec.moe_ep_band_remap(
                        ids.as_ref(),
                        wts.as_ref(),
                        n_pairs,
                        ep_lo,
                        ep_lo + n_expert_local as u32,
                    );
                }
                rec.zero(counts.as_ref(), n_expert_local);
                // Clear the scatter's `fill` counters here (independent of `counts`) so the parallel
                // clear overlaps the count/scan rather than riding the 1-lane serial scan.
                rec.zero(fill.as_ref(), n_expert_local);
                rec.moe_bucket_count(ids.as_ref(), counts.as_ref(), n_pairs);
                rec.moe_bucket_scan(counts.as_ref(), offsets.as_ref(), n_expert_local);
                // Per-expert down_scale (diffusion-gemma) is baked into `bucket_wts` HERE (the
                // scatter already has the expert id in hand to index it) rather than as a separate
                // post-GEMM pass — `moe_scatter_reduce` then needs no changes at all, and the
                // scale multiply is exactly equivalent to `moe_accumulate_scaled`'s per-token
                // semantics since it's linear in the down output.
                let dsb: Option<&dyn Buffer> = match down_scale {
                    Some(ds) => Some(r(*ds)?),
                    None => None,
                };
                rec.moe_bucket_scatter(
                    ids.as_ref(),
                    wts.as_ref(),
                    offsets.as_ref(),
                    fill.as_ref(),
                    bucket_rows.as_ref(),
                    bucket_wts.as_ref(),
                    inv_pos.as_ref(),
                    dsb,
                    n_pairs,
                    n_used,
                );

                let (gw, dw) = (r(*gate_exps)?, r(*down_exps)?);
                // Gather+quant all assignments into the packed layout in one pass.
                rec.quant_q8_gather(
                    xb,
                    bucket_rows.as_ref(),
                    pool[&qa].as_ref(),
                    pool[&qda].as_ref(),
                    pool[&qsa].as_ref(),
                    n_pairs,
                    ne,
                );
                // `sact` (the activation's per-block min-correction sums) is only READ by the
                // dtypes in `infr_core::tensor::MOE_MMQ_SACT_DTYPES` (Q4_K/Q5_K/Q5_1/Q4_1) — the
                // rest of `MOE_MMQ_DTYPES` are symmetric (no min term), and Q2_K (also
                // min-carrying) self-computes its own narrower Σx in-shader instead (its 16-elem
                // sub-block is half `sact`'s 32-elem granularity — see
                // `native_gemm_mmq_q2_k.comp`'s doc). Same split `down_needs_sact` already uses
                // below. Mirrored per-role here since gate/up can each independently be any
                // `MOE_MMQ_DTYPES` member.
                let gate_needs_sact = infr_core::tensor::moe_mmq_needs_sact(gdt);
                rec.matmul_mmq_experts(
                    gdt,
                    "expert_gateup",
                    pool[&qa].as_ref(),
                    pool[&qda].as_ref(),
                    gate_needs_sact.then(|| pool[&qsa].as_ref()),
                    gw,
                    0,
                    stride,
                    counts.as_ref(),
                    offsets.as_ref(),
                    pool[&ge].as_ref(),
                    rows,
                    ne,
                    gu_width,
                    n_expert_local,
                    n_used,
                );
                if let Some(ue) = ue {
                    // Split (qwen3moe, llama4): the up GEMM reads the same quantized activations
                    // and writes its own buffer — disjoint from the gate GEMM, no barrier needed.
                    let up_needs_sact = infr_core::tensor::moe_mmq_needs_sact(udt);
                    rec.suppress_sync(true);
                    rec.matmul_mmq_experts(
                        udt,
                        "expert_gateup",
                        pool[&qa].as_ref(),
                        pool[&qda].as_ref(),
                        up_needs_sact.then(|| pool[&qsa].as_ref()),
                        r(*up_exps)?,
                        0,
                        stride,
                        counts.as_ref(),
                        offsets.as_ref(),
                        pool[&ue].as_ref(),
                        rows,
                        ne,
                        nff,
                        n_expert_local,
                        n_used,
                    );
                    rec.suppress_sync(false);
                    // llama4 weight-before-FFN: scale the gate/up GEMM outputs by the packed
                    // per-pair routing weight (`bucket_wts`, indexed the same way `ge`/`ue` are
                    // packed) BEFORE the activation — exact-equivalent to the CPU's input-side
                    // fold (see `moe_weight_scale`'s doc). `moe_scatter_reduce` below is told
                    // `prescaled` so it doesn't apply the weight a second time.
                    if *weight_before {
                        rec.moe_weight_scale(pool[&ge].as_ref(), bucket_wts.as_ref(), n_pairs, nff);
                        rec.moe_weight_scale(pool[&ue].as_ref(), bucket_wts.as_ref(), n_pairs, nff);
                    }
                    rec.silu_mul(
                        pool[&ge].as_ref(),
                        pool[&ue].as_ref(),
                        pool[&ae].as_ref(),
                        n_pairs * nff,
                        *swiglu_clamp,
                    );
                } else {
                    // Fused: `ge` already holds [n_pairs, 2*nff] (gate half first, up half second
                    // per row) from the single wide GEMM above.
                    if *weight_before {
                        rec.moe_weight_scale(
                            pool[&ge].as_ref(),
                            bucket_wts.as_ref(),
                            n_pairs,
                            gu_width,
                        );
                    }
                    match act {
                        Activation::Silu => rec.silu_mul_fused(
                            pool[&ge].as_ref(),
                            pool[&ae].as_ref(),
                            n_pairs,
                            nff,
                            *swiglu_clamp,
                        ),
                        Activation::Gelu => rec.gelu_mul_fused(
                            pool[&ge].as_ref(),
                            pool[&ae].as_ref(),
                            n_pairs,
                            nff,
                            *swiglu_clamp,
                        ),
                        Activation::Sigmoid => {
                            return Err(be(
                                "vulkan adapter: fused_gate_up batched MoeFfn Sigmoid unsupported",
                            ))
                        }
                    }
                }
                rec.quant_q8(
                    pool[&ae].as_ref(),
                    pool[&dqa].as_ref(),
                    pool[&dda].as_ref(),
                    pool[&dsa].as_ref(),
                    n_pairs,
                    nff,
                );
                // The min-carrying down formats (Q4_K/Q5_K's K-quant min, Q5_1/Q4_1's legacy min)
                // bind `sact` — the rest of `MOE_MMQ_DTYPES` are symmetric (no min), and Q2_K
                // self-computes its own min term in-shader (see the gate/up `_needs_sact` comment
                // above), same as the per-token path's dtype-gated `sact` use.
                let down_needs_sact = infr_core::tensor::moe_mmq_needs_sact(ddt);
                rec.matmul_mmq_experts(
                    ddt,
                    "expert_down",
                    pool[&dqa].as_ref(),
                    pool[&dda].as_ref(),
                    down_needs_sact.then(|| pool[&dsa].as_ref()),
                    dw,
                    0,
                    down_stride,
                    counts.as_ref(),
                    offsets.as_ref(),
                    pool[&ye].as_ref(),
                    rows,
                    nff,
                    ne,
                    n_expert_local,
                    n_used,
                );
                rec.moe_scatter_reduce(
                    pool[&ye].as_ref(),
                    bucket_wts.as_ref(),
                    inv_pos.as_ref(),
                    yb,
                    rows,
                    ne,
                    n_used,
                    *weight_before,
                );
                transient.extend([
                    logits,
                    ids,
                    wts,
                    counts,
                    offsets,
                    fill,
                    bucket_rows,
                    bucket_wts,
                    inv_pos,
                ]);
                return Ok(());
            }
            // ── SMALL-m fast path (`rows` ≤ `moe_small_m_threshold()`, covering decode's rows==1
            // AND tiny prefill chunks like `pp4`): the batched path above pays a near-fixed
            // per-ACTIVE-expert cost — its tiled GEMM streams that expert's FULL weight bank no
            // matter how few rows use it (inherent to any approach — the weight has to be read
            // once if the expert is used at all), but ALSO pays quant_q8_gather + bucket
            // count/scan/scatter and a 64-row-tile GEMM that computes 60+ garbage rows when only
            // 1-8 are real. For a handful of tokens that overhead dominates. Instead: one dispatch
            // per stage computes ALL `rows*n_used` (token, expert) pairs' id-indexed GEMVs
            // directly against the native block format (no int8 quant pass, no bucket sort) —
            // `linear_native_id_multi`'s `rows` widening (see its doc) flattens `(row, slot)` into
            // the same indexing the decode path always used for its single token. Trade-off: an
            // expert used by more than one of the `rows` tokens gets its weight bank re-read once
            // per occurrence (no shared BM=64-row-tile reuse like the batched path gets), so this
            // loses once `rows` grows enough for cross-token expert overlap to matter — hence the
            // threshold (measured; see `moe_small_m_threshold`'s doc).
            //
            // Per-layer scratch rides the per-execute `pool` (same (tag, bytes) key across layers
            // → same buffer; the recorder's hazard tracking serializes the reuse, exactly like the
            // batched arm's packed scratch above), resolved at each use site via `SmB::get(pool)`.
            // Every one of these 7 buffers is FULLY WRITTEN before it is read within this op, so
            // `pooled`'s alloc_uninit is safe (no zero-init needed): the router GEMV writes all
            // `rows*n_expert` logits; `moe_topk` writes all `n_slots = rows*n_used` ids/wts
            // (n_slots is EXACT — every top-k slot is a real assignment, no padding/unrouted rows
            // like the batched path's `npad`); the id-GEMVs write every `n_slots` row of gate/up,
            // act, and y; and `dst` (a separate scratch tensor, not one of these) is explicitly
            // `rec.zero`'d before the accumulate. The previous shape alloc'd 7 FRESH zero-filled
            // buffers per MoE layer per execute — `Backend::alloc`'s calloc contract fills each
            // device-local buffer through a one-shot submit + queue_wait_idle (~27us each), and at
            // qwen3moe's 48 MoE layers that was ~340 fence-waited submits of pure host stall
            // (decode never sees this — record-once replay allocs the pool once).
            // `INFR_NO_MOE_SM_POOL=1` restores the per-layer zeroed allocs (A/B correctness oracle).
            let no_pool = be_.cfg().kernels.vulkan.no_moe_sm_pool;
            let n_slots = rows * n_used;
            let logits = sm_buf(pool, be_, no_pool, "moe_sm_logits", rows * n_expert)?;
            let ids = sm_buf(pool, be_, no_pool, "moe_sm_ids", n_slots)?;
            let wts = sm_buf(pool, be_, no_pool, "moe_sm_wts", n_slots)?;
            // Fused: one [n_slots, 2*nff] gate|up buffer (gate half first, up half second per row —
            // `Op::GatedActFused`'s convention); split: separate [n_slots, nff] gate/up buffers.
            let gubuf = if *fused_gate_up {
                Some(sm_buf(pool, be_, no_pool, "moe_sm_gu", n_slots * 2 * nff)?)
            } else {
                None
            };
            let gbuf = if *fused_gate_up {
                None
            } else {
                Some(sm_buf(pool, be_, no_pool, "moe_sm_g", n_slots * nff)?)
            };
            let ubuf = if *fused_gate_up {
                None
            } else {
                Some(sm_buf(pool, be_, no_pool, "moe_sm_u", n_slots * nff)?)
            };
            let abuf = sm_buf(pool, be_, no_pool, "moe_sm_a", n_slots * nff)?;
            let ybuf = sm_buf(pool, be_, no_pool, "moe_sm_y", n_slots * ne)?;
            // All `sm_buf` allocations are done — from here `pool` is only indexed immutably (via
            // `SmB::get`), so multiple pooled buffers can be resolved at once (the batched arm's
            // `pool[&k]` discipline).
            let xb = r(*x)?;
            // diffusion-gemma's router reads a DIFFERENTLY normalized/scaled row than the experts
            // (see the `Op::MoeFfn` doc); qwen3moe binds the same handle as `x`.
            let rxb = r(*router_x)?;

            // Router logits over all experts, all `rows` tokens in one dispatch.
            let rdt = graph.desc(*router).dtype;
            let rw = r(*router)?;
            if native_dense_supported(rdt) {
                rec.linear_native(rdt, rw, 0, rxb, logits.get(pool), rows, ne, n_expert);
            } else if matches!(rdt, infr_core::DType::F32) {
                // qwen3moe ships the router (ffn_gate_inp) as F32 — the f16 GEMV would read its
                // bytes as f16 garbage and route to arbitrary experts.
                rec.linear_f32(rw, rxb, logits.get(pool), rows, ne, n_expert);
            } else {
                rec.linear(rw, rxb, logits.get(pool), rows, ne, n_expert);
            }
            // Top-`n_used` per token (gating-dependent weighting), weights pre-scaled by `scale`.
            rec.moe_topk(
                logits.get(pool),
                ids.get(pool),
                wts.get(pool),
                bias_buf,
                rows,
                n_expert,
                n_used,
                *scale,
                gating_u32,
                *norm_w,
                has_bias,
                *n_expert_groups,
                *n_expert_groups_used,
                hash_buf,
                hash,
            );
            // EP: rewrite the global top-k ids into this rank's local shard indices (out-of-band →
            // id 0, weight 0). The id-indexed GEMVs below then read the local bank; the weighted
            // `moe_accumulate` drops the zero-weight (out-of-band) slots. No-op off EP.
            if ep_band.is_some() {
                rec.moe_ep_band_remap(
                    ids.get(pool),
                    wts.get(pool),
                    n_slots,
                    ep_lo,
                    ep_lo + n_expert_local as u32,
                );
            }
            let n_act = n_slots * nff;
            if let Some(gubuf) = &gubuf {
                // Fused per-role expert GEMV: ONE dispatch over all rows' [ne, 2*nff] expert slices.
                rec.linear_native_id_multi(
                    gdt,
                    r(*gate_exps)?,
                    ids.get(pool),
                    n_used,
                    stride,
                    xb,
                    false,
                    gubuf.get(pool),
                    ne,
                    2 * nff,
                    rows,
                );
                // llama4 weight-before-FFN: scale the fused gate|up GEMV output by the per-slot
                // routing weight (`wts`, one weight per (token,slot) = one row of `gubuf`) BEFORE
                // the activation — see `moe_weight_scale`'s doc. `moe_accumulate` below is told
                // `prescaled` so it doesn't apply the weight a second time.
                if *weight_before {
                    rec.moe_weight_scale(gubuf.get(pool), wts.get(pool), n_slots, 2 * nff);
                }
                match act {
                    Activation::Silu => rec.silu_mul_fused(
                        gubuf.get(pool),
                        abuf.get(pool),
                        n_slots,
                        nff,
                        *swiglu_clamp,
                    ),
                    Activation::Gelu => rec.gelu_mul_fused(
                        gubuf.get(pool),
                        abuf.get(pool),
                        n_slots,
                        nff,
                        *swiglu_clamp,
                    ),
                    Activation::Sigmoid => {
                        return Err(be(
                            "vulkan adapter: fused_gate_up MoeFfn Sigmoid unsupported",
                        ))
                    }
                }
            } else {
                // Split per-role expert GEMVs: gate/up read the shared row; down reads each slot's act.
                let (gbuf, ubuf) = (gbuf.as_ref().unwrap(), ubuf.as_ref().unwrap());
                rec.linear_native_id_multi(
                    gdt,
                    r(*gate_exps)?,
                    ids.get(pool),
                    n_used,
                    stride,
                    xb,
                    false,
                    gbuf.get(pool),
                    ne,
                    nff,
                    rows,
                );
                rec.linear_native_id_multi(
                    udt,
                    r(*up_exps)?,
                    ids.get(pool),
                    n_used,
                    stride,
                    xb,
                    false,
                    ubuf.get(pool),
                    ne,
                    nff,
                    rows,
                );
                if *weight_before {
                    rec.moe_weight_scale(gbuf.get(pool), wts.get(pool), n_slots, nff);
                    rec.moe_weight_scale(ubuf.get(pool), wts.get(pool), n_slots, nff);
                }
                match act {
                    Activation::Silu => rec.silu_mul(
                        gbuf.get(pool),
                        ubuf.get(pool),
                        abuf.get(pool),
                        n_act,
                        *swiglu_clamp,
                    ),
                    Activation::Sigmoid => rec.mul_sigmoid(
                        gbuf.get(pool),
                        ubuf.get(pool),
                        abuf.get(pool),
                        n_act,
                        0,
                        0,
                        0,
                    ),
                    Activation::Gelu => rec.gelu_mul_off(
                        gbuf.get(pool),
                        ubuf.get(pool),
                        0,
                        0,
                        nff,
                        0,
                        nff,
                        0,
                        abuf.get(pool),
                        n_act,
                    ),
                }
            }
            rec.linear_native_id_multi(
                ddt,
                r(*down_exps)?,
                ids.get(pool),
                n_used,
                down_stride,
                abuf.get(pool),
                true,
                ybuf.get(pool),
                nff,
                ne,
                rows,
            );
            // `Op::MoeFfn` dst is the pure FFN output (residual Add is a separate op), but
            // moe_accumulate(_scaled) ADDs — zero dst first.
            let dstb = r(*dst)?;
            rec.zero(dstb, rows * ne);
            match down_scale {
                Some(ds) => rec.moe_accumulate_scaled(
                    ybuf.get(pool),
                    wts.get(pool),
                    ids.get(pool),
                    r(*ds)?,
                    dstb,
                    ne,
                    n_used,
                    rows,
                ),
                None => rec.moe_accumulate(
                    ybuf.get(pool),
                    wts.get(pool),
                    dstb,
                    ne,
                    n_used,
                    rows,
                    *weight_before,
                ),
            }
            // Under the `no_pool` escape these are owned per-layer boxes — drain them into
            // `transient` so the recording outlives them; pooled buffers are no-ops here (owned by
            // `pool`, itself drained into `transient` at the end of the op loop).
            for b in [logits, ids, wts, abuf, ybuf] {
                b.into_transient(transient);
            }
            for b in [gubuf, gbuf, ubuf].into_iter().flatten() {
                b.into_transient(transient);
            }
        }
        Op::Mla {
            q,
            k_cache,
            wk_b,
            wv_b,
            dst,
            rows,
            kv_len,
            n_head,
            q_head_dim,
            kv_lora_rank,
            qk_nope_dim,
            qk_rope_dim,
            v_head_dim,
            scale,
            mask,
            pos,
            theta,
            freq_factors,
            key_bias,
        } => {
            // `mla.comp` holds the absorbed query and the V accumulator in FIXED-size shared
            // arrays, so an over-wide latent would write past them (shared memory is not bounds
            // checked — the symptom is corrupted neighbours or a lost device, not a fault). The
            // kernel cannot refuse the work; this is the last place that can, and it still has the
            // dims to name.
            if *kv_lora_rank + *qk_rope_dim > MLA_MAX_KEY_LEN
                || *kv_lora_rank > MLA_MAX_KV_LORA_RANK
            {
                return Err(be(format!(
                    "vulkan Op::Mla: dims exceed mla.comp's fixed shared arrays — kv_lora_rank \
                     {kv_lora_rank} + qk_rope_dim {qk_rope_dim} = key_len {} (max \
                     {MLA_MAX_KEY_LEN}), kv_lora_rank max {MLA_MAX_KV_LORA_RANK}",
                    *kv_lora_rank + *qk_rope_dim
                )));
            }
            let key_len = (*kv_lora_rank + *qk_rope_dim) as usize;
            let cap = graph.desc(*k_cache).numel();
            let cache_cap_rows = (cap / key_len.max(1)) as u32;
            // `canvas_lo` rides its own push-constant field rather than sharing `window`: the
            // shader's causal arm reads `window` too (a causal mask with a window narrows `lo`),
            // so one repurposed slot could not carry both meanings.
            let (mask_type, window, canvas_lo) = match mask {
                AttnMask::Causal => (0u32, 0u32, 0u32),
                AttnMask::SlidingWindow(w) => (1u32, *w as u32, 0u32),
                AttnMask::Canvas { lo } => (2u32, 0u32, *lo as u32),
            };
            let ff = freq_factors.map(&r).transpose()?;
            let kbias = key_bias.map(&r).transpose()?;
            rec.mla(
                r(*q)?,
                r(*k_cache)?,
                r(*wk_b)?,
                r(*wv_b)?,
                r(*dst)?,
                *rows,
                *kv_len,
                *n_head,
                *q_head_dim,
                *kv_lora_rank,
                *qk_nope_dim,
                *qk_rope_dim,
                *v_head_dim,
                *scale,
                *pos,
                mask_type,
                window,
                *theta,
                cache_cap_rows,
                canvas_lo,
                ff,
                kbias,
            );
        }
        Op::LightningIndexer {
            q,
            k_cache,
            weights,
            dst,
            rows,
            kv_len,
            n_head,
            head_dim,
            top_k,
            scale,
            pos,
        } => {
            // Every output slot must name a key that exists. llama.cpp clamps at graph-build time
            // (`n_top_k = min(indexer_score->ne[0], n_indexer_top_k)`); the kernel cannot refuse
            // the work, so refuse it here with the dims to name rather than emit a duplicated or
            // sentinel index.
            if top_k > kv_len {
                return Err(be(format!(
                    "vulkan Op::LightningIndexer: top_k {top_k} exceeds kv_len {kv_len} — there \
                     are not that many keys to name"
                )));
            }
            // The shader reads the indexer cache as u32-packed f16 PAIRS, so a row must not start
            // mid-word.
            if !head_dim.is_multiple_of(2) {
                return Err(be(format!(
                    "vulkan Op::LightningIndexer: head_dim {head_dim} is odd — the f16 cache is \
                     read as u32-packed pairs, which needs an even row width"
                )));
            }
            // Key `j` is row `j` — no ring fold (see `Op::LightningIndexer`'s doc: causal masking
            // makes position 0 eligible for every query, so a wrapped cache has already lost a key
            // this op must score). Refuse one rather than read a row that holds another position.
            let cap_rows = (graph.desc(*k_cache).numel() / (*head_dim as usize).max(1)) as u32;
            if cap_rows < *kv_len {
                return Err(be(format!(
                    "vulkan Op::LightningIndexer: indexer cache holds {cap_rows} rows but kv_len \
                     is {kv_len} — a wrapped indexer cache has already lost keys this op must score"
                )));
            }
            // Score scratch, the `[n_kv, n_tokens]` tensor llama.cpp materializes too. Pooled: one
            // buffer per (rows, kv_len) shape shared by every indexer layer in the graph, since the
            // layers are serialized by dataflow. Fully written by phase 1 before phase 2 reads it.
            let sk = pooled(
                pool,
                be_,
                "lidx_scores",
                (*rows as usize) * (*kv_len as usize) * 4,
            )?;
            rec.lightning_indexer(
                r(*q)?,
                r(*k_cache)?,
                r(*weights)?,
                pool[&sk].as_ref(),
                r(*dst)?,
                *rows,
                *kv_len,
                *n_head,
                *head_dim,
                *top_k,
                *scale,
                *pos,
            );
        }
        Op::TopkMask {
            idx,
            dst,
            rows,
            kv_len,
            top_k,
        } => {
            // Every index the scatter writes must land inside the row. The kernel cannot refuse
            // the dispatch and an out-of-range write into an SSBO is undefined, so refuse here —
            // this is the same bound `Op::LightningIndexer` enforces on its own output.
            if top_k > kv_len {
                return Err(be(format!(
                    "vulkan Op::TopkMask: top_k {top_k} exceeds kv_len {kv_len} — the indices \
                     cannot all name keys inside the mask row"
                )));
            }
            rec.topk_mask(r(*idx)?, r(*dst)?, *rows, *kv_len, *top_k);
        }
        Op::HyperConnectMix {
            mixes,
            scale,
            base,
            pre,
            gates,
            rows,
            hc,
            eps,
            n_iter,
        } => {
            hyper_check_hc(*hc, "vulkan Op::HyperConnectMix")?;
            if *n_iter < 1 {
                return Err(be(format!(
                    "vulkan Op::HyperConnectMix: n_iter {n_iter} — Sinkhorn runs at least one \
                     normalisation over src"
                )));
            }
            let g = match gates {
                Some(infr_core::graph::HyperGates { post, comb }) => Some((r(*post)?, r(*comb)?)),
                None => None,
            };
            rec.hyper_mix(
                r(*mixes)?,
                r(*scale)?,
                r(*base)?,
                r(*pre)?,
                g,
                *rows,
                *hc,
                *eps,
                *n_iter,
            );
        }
        Op::HyperConnectPre {
            x,
            weights,
            dst,
            rows,
            hc,
            n_embd,
        } => {
            hyper_check_hc(*hc, "vulkan Op::HyperConnectPre")?;
            rec.hyper_pre(r(*x)?, r(*weights)?, r(*dst)?, *rows, *hc, *n_embd);
        }
        Op::HyperConnectPost {
            x,
            residual,
            post,
            comb,
            dst,
            rows,
            hc,
            n_embd,
        } => {
            hyper_check_hc(*hc, "vulkan Op::HyperConnectPost")?;
            rec.hyper_post(
                r(*x)?,
                r(*residual)?,
                r(*post)?,
                r(*comb)?,
                r(*dst)?,
                *rows,
                *hc,
                *n_embd,
            );
        }
    }
    Ok(())
}

/// Refuse a stream count wider than the fixed-size private array `hyper_mix.comp` holds a token's
/// mixing matrix in. The kernel cannot bounds-check itself — an out-of-range private-array write
/// is undefined — so the refusal has to happen here, before the dispatch.
fn hyper_check_hc(hc: u32, what: &str) -> Result<()> {
    let max = infr_core::graph::HYPER_CONNECT_MAX_MULT;
    if hc < 1 || hc > max {
        return Err(be(format!(
            "{what}: hc_mult {hc} outside the supported 1..={max} (every shipped DeepSeek V4 \
             config uses 4; hyper_mix.comp holds a token's whole hc*hc matrix in a private array \
             of HC_MAX*HC_MAX floats)"
        )));
    }
    Ok(())
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn execute(be_: &VulkanBackend, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
    let plan = plan
        .as_any()
        .downcast_ref::<VkDecodePlan>()
        .ok_or_else(|| be("vulkan adapter: plan is not a VkDecodePlan"))?;
    // A paged model forces the static per-execute path regardless of `plan.eligible`: paged
    // execution needs a host readback between a MoE layer's router/top-k and its expert GEMVs
    // (see `execute_static`'s paged branch), which the record-once replay tape — recorded ONCE
    // with no host round-trip built in — can't express. `Backend::moe_paged`'s doc has the full
    // rationale; the seam's `dyn_replay` gate mirrors this so a paged model never even BUILDS the
    // (then-unused) replay plan, but this check is what actually guarantees correctness.
    // Dense layer streaming forces the static path for a different reason than MoE (no readback
    // exists — layer order is deterministic — but the per-token ring staging + per-slot weight
    // offsets can't live inside a record-once tape whose descriptor sets and push constants are
    // frozen at record time).
    if !plan.eligible || be_.moe_paged() || be_.dense_paged() {
        return execute_static(be_, &plan.graph, bindings);
    }

    // Record-once replay: build the recording on the first execute (cache miss), then only refresh
    // `params` from `positions[0]` and resubmit. The runner reuses the SAME bound buffers across the
    // whole decode loop, so the recorded descriptor sets stay valid.
    let mut guard = plan.replay.lock().unwrap();
    if guard.is_none() {
        *guard = Some(record_decode_replay(be_, &plan.graph, bindings)?);
    }
    let replay = guard.as_ref().unwrap();
    if !replay.self_advancing {
        // Host fallback (INFR_NO_GPU_POS): pos from positions[0] (decode rows=1); kv_len = pos+1.
        let pos = read_pos0(be_, resolve(&replay.scratch, bindings, replay.positions)?)?;
        let kv_len = pos + 1;
        let mut pbytes = [0u8; 8];
        pbytes[0..4].copy_from_slice(&pos.to_le_bytes());
        pbytes[4..8].copy_from_slice(&kv_len.to_le_bytes());
        be_.upload(replay.params.as_ref(), &pbytes)?;
    }
    replay.recorded.replay().map_err(|e| be(e.to_string()))?;
    Ok(())
}

/// Chained decode (`Backend::execute_chain`): replay the record-once recording `n` times in ONE
/// submission — the sampled id flows device-side (the runner bound the sampler output and the
/// embed-gather ids input to one buffer), params self-advance, and the trailing `id_log` writes
/// each iteration's id into the ring. Returns the n ids read from the ring, or `None` when the
/// plan can't chain (ineligible graph, host-pos fallback, no device sampler, or n out of the
/// ring's range) — the caller falls back to per-token `execute`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn execute_chain(
    be_: &VulkanBackend,
    plan: &dyn Plan,
    bindings: &Bindings,
    n: usize,
) -> Result<Option<Vec<u32>>> {
    let Some(plan) = plan.as_any().downcast_ref::<VkDecodePlan>() else {
        return Ok(None);
    };
    if !plan.eligible || n == 0 || n > 64 || be_.moe_paged() || be_.dense_paged() {
        return Ok(None);
    }
    let mut guard = plan.replay.lock().unwrap();
    if guard.is_none() {
        *guard = Some(record_decode_replay(be_, &plan.graph, bindings)?);
    }
    let replay = guard.as_ref().unwrap();
    let (true, Some(ring)) = (replay.self_advancing, replay.ring.as_ref()) else {
        return Ok(None);
    };
    // A chain is ONE submit of `n` back-to-back decode steps, and the GPU hang watchdog is armed
    // per submit — so the chain length is bounded by the same measured budget as the forward-pass
    // splitter. DECLINE the chain rather than shortening it: the caller draws one sampling uniform
    // per chained step BEFORE calling (`Backend::max_decode_chain` is what it clamps with), so a
    // backend that quietly returned fewer ids than it was asked for would leave the RNG stream
    // advanced past the tokens it produced. Returning None hands the caller back to its per-token
    // path, which re-draws for the position it actually reaches.
    //
    // Normally unreachable — the caller's own clamp already collapses the chain to 1 on any
    // splitting device — but the cap is re-tuned from measurement (`observe_forward`) and can flip
    // under a concurrent request between that clamp and this call.
    if n > replay.recorded.max_chain() {
        return Ok(None);
    }
    // The chunk decodes positions p0+1 ..= p0+n (params[0] = the last decoded position).
    let p0 = read_pos0(be_, replay.params.as_ref())?;
    replay.recorded.replay_n(n).map_err(|e| be(e.to_string()))?;
    let mut rbytes = vec![0u8; 64 * 4];
    be_.download(ring.as_ref(), &mut rbytes)?;
    let ring_u32: &[u32] = bytemuck::cast_slice(&rbytes);
    let ids = (1..=n as u32)
        // `p0` can be the `-1i32 as u32` sentinel from `read_pos0` (no prior decode yet); the `&
        // 63` mask below is applied AFTER the add, so wrapping is the arithmetically correct sum
        // — only the non-wrapping `+` panics in debug on the sentinel.
        .map(|i| ring_u32[(p0.wrapping_add(i) & 63) as usize])
        .collect();
    Ok(Some(ids))
}

/// Build the record-once decode replay: persistent scratch + params SSBO, one recording via the
/// `_dyn` (params-driven) kernels for the pos-dependent ops. Only called for a [`decode_eligible`]
/// graph. The returned recording is replayed (not submitted here) — the caller updates `params` and
/// calls `replay()`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn record_decode_replay(
    be_: &VulkanBackend,
    graph: &Graph,
    bindings: &Bindings,
) -> Result<DecodeReplay> {
    let scratch = alloc_scratch(be_, graph)?;
    // `[pos, kv_len]` — Staging (host-visible mapped) so per-token `upload` is a plain memcpy.
    let params = be_.alloc(8, BufferUsage::Staging)?;
    let positions = graph
        .ops
        .iter()
        .find_map(|op| match op {
            Op::QkNormRope { positions, .. } | Op::Rope { positions, .. } => Some(*positions),
            _ => None,
        })
        .ok_or_else(|| be("vulkan adapter: eligible decode has no positions tensor"))?;
    let plan = infr_core::fusion::plan_fusions(graph, &fusion_cfg(be_));
    let fused_kv_write = plan.kv_write;
    let fused_add = plan.linear_add;
    let mut skip_op = plan.skip;

    let mut transient: Vec<Box<dyn Buffer>> = Vec::new();
    let mut dyn_args: Vec<DynAttnCtx> = Vec::new();
    let mut pool: ScratchPool = HashMap::new();
    // Watchdog splitter (mirrors `execute_static`): the GPU hang watchdog is armed per SUBMIT, so
    // recording the whole decode into ONE command buffer makes it one indivisible watchdog job —
    // fine on a discrete GPU (tens of ms) and fatal on a slow integrated one, where a big-model
    // decode step in a single submit exceeds the ~2 s TDR budget and hard-lasts the device. `cap`
    // (0 = unlimited, the discrete default) bounds the dispatches per segment; each segment becomes
    // a SEPARATE submit at replay (`RecordedCmd::replay`) so the watchdog sees several short jobs
    // instead of one long one. This preserves the `_dyn`/params/ring decode semantics exactly — the
    // identical dispatch stream is just distributed across command buffers, with a seeded global
    // barrier at each continuation segment's head carrying the cross-segment ordering.
    let cap = be_.submit_dispatch_cap();
    let mut segments: Vec<crate::recorder::RecordedSegment> = Vec::new();
    let mut rec = be_.recorder_persistent()?;
    // Device-side position stream: seed params to [pos0-1, pos0] and record a one-thread
    // increment FIRST — every replay self-advances and the host never writes pos/params again
    // (no dyn kernel reads the `positions` buffer; they all read params). INFR_NO_GPU_POS=1
    // keeps the per-replay host read_pos0 + params upload instead (A/B).
    let self_advancing = be_.cfg().kernels.vulkan.gpu_pos;
    if self_advancing {
        let pos0 = read_pos0(be_, resolve(&scratch, bindings, positions)?)?;
        let mut pbytes = [0u8; 8];
        pbytes[0..4].copy_from_slice(&pos0.wrapping_sub(1).to_le_bytes());
        pbytes[4..8].copy_from_slice(&pos0.to_le_bytes());
        be_.upload(params.as_ref(), &pbytes)?;
        // Cross-iteration fence for the chained decode (replay_n submits this recording n times
        // back-to-back): orders iteration i's tail writes (sampled id, id_log) before iteration
        // i+1's head reads (params_advance, EmbedGather). One extra barrier per token — noise.
        rec.global_barrier();
        rec.params_advance(params.as_ref());
    }
    let mode = RopeMode::Dynamic(params.as_ref());
    let mut mmv_memo: Option<(TensorId, usize, usize)> = None;
    for (op_idx, op) in graph.ops.iter().enumerate() {
        if skip_op.contains(&op_idx) {
            continue;
        }
        // Split BETWEEN ops, never inside one (a single op can lower to several dispatches sharing
        // transient scratch whose ordering is per-recorder): once the current segment reaches the
        // cap, close it and open a fresh persistent recorder. Its leading `seed_barrier` seeds the
        // cross-segment RAW/WAR ordering that per-recorder hazard tracking (which starts empty)
        // can't otherwise see — every layer reads the residual stream a prior segment wrote. cap ==
        // 0 (discrete) never trips this — ONE segment, byte-identical to the record-once fast path.
        if cap > 0 && rec.dispatches() >= cap {
            let fresh = be_.recorder_persistent()?;
            fresh.seed_barrier();
            segments.push(
                std::mem::replace(&mut rec, fresh)
                    .end_segment()
                    .map_err(|e| be(e.to_string()))?,
            );
        }
        // Peephole: fuse Op::Linear (f32, m<=4) + Op::GatedAct (Gelu, strided up) into one
        // e2b_gate dispatch for E2B per-layer inp_gate projections.
        if let Op::Linear {
            x,
            weight,
            dst,
            m,
            in_f,
            out_f,
            ..
        } = op
        {
            if graph.desc(*weight).dtype == infr_core::DType::F32
                && *m <= 4
                && op_idx + 1 < graph.ops.len()
            {
                if let Op::GatedAct {
                    gate: g_gate,
                    up: g_up,
                    dst: g_dst,
                    act: Activation::Gelu,
                    up_off: g_off,
                    up_stride: g_stride,
                    ..
                } = &graph.ops[op_idx + 1]
                {
                    if *g_gate == *dst && *g_dst == *dst && *g_stride > 0 {
                        let (w_, x_, up_, y_) = (
                            resolve(&scratch, bindings, *weight)?,
                            resolve(&scratch, bindings, *x)?,
                            resolve(&scratch, bindings, *g_up)?,
                            resolve(&scratch, bindings, *dst)?,
                        );
                        rec.e2b_gate(
                            w_,
                            x_,
                            up_,
                            *g_off as usize,
                            *g_stride as usize,
                            y_,
                            *m as usize,
                            *in_f as usize,
                            *out_f as usize,
                        );
                        skip_op.insert(op_idx + 1);
                        continue;
                    }
                }
            }
        }
        lower_op(
            be_,
            graph,
            op_idx,
            op,
            &rec,
            &scratch,
            bindings,
            &fused_kv_write,
            &fused_add,
            &mode,
            &mut transient,
            &mut dyn_args,
            &mut pool,
            &mut mmv_memo,
            None,
        )?;
    }
    for c in dyn_args.drain(..) {
        transient.extend([c.args, c.pm, c.pl, c.pacc]);
    }
    transient.extend(pool.into_values());
    // Chained decode: when the graph ends in a device-side sampler (Argmax/Sample), log each
    // iteration's id into a host-visible ring so `execute_chain` reads the whole chunk in one go.
    let ring = if self_advancing {
        graph
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Argmax { dst, .. } | Op::Sample { dst, .. } => Some(*dst),
                _ => None,
            })
            .map(|dst| -> Result<Box<dyn Buffer>> {
                let rb = be_.alloc(64 * 4, BufferUsage::Readback)?;
                rec.id_log(
                    params.as_ref(),
                    resolve(&scratch, bindings, dst)?,
                    rb.as_ref(),
                );
                Ok(rb)
            })
            .transpose()?
    } else {
        None
    };
    // Close the final segment (holds the trailing `id_log`) and assemble the recording. A discrete
    // GPU (cap == 0) never split, so `segments` is exactly this one — a single-segment `RecordedCmd`
    // that replays in one submit, unchanged.
    segments.push(rec.end_segment().map_err(|e| be(e.to_string()))?);
    let recorded = crate::recorder::RecordedCmd::from_segments(be_, segments);
    Ok(DecodeReplay {
        scratch,
        params,
        self_advancing,
        ring,
        positions,
        recorded,
        _transient: transient,
    })
}

/// Tear down the segment being recorded when an op fails to lower, and return the error to
/// propagate.
///
/// `Recorder` has no `Drop` (see `Recorder::free_transient`), so a bare `return Err(..)` out of the
/// op loop leaves that segment's descriptor pools alive and `vkDestroyDevice` reports them —
/// VUID-vkDestroyDevice-device-05137, which is what the validation layer prints if this is skipped.
/// The partial recording is waste (the forward is being abandoned), so it is discarded rather than
/// submitted, exactly as the shutdown abort in the loop does. Already-submitted segments need
/// nothing here: `PendingSegment`'s `Drop` drains and frees them.
///
/// A teardown failure is folded into the message instead of replacing it — the original error is
/// why the forward stopped, and it is the one worth reading.
fn abort_segment(rec: Option<Recorder<'_>>, e: Error) -> Error {
    match rec.map(|partial| partial.discard()) {
        Some(Err(de)) => be(format!("{e} (partial segment teardown also failed: {de})")),
        _ => e,
    }
}

/// Per-execute static recording: allocate `Internal` scratch fresh, record every op via `lower_op`
/// (Static mode — pos as a push constant read from `positions[0]`), submit + wait. Used for prefill
/// batches and every ineligible decode (gemma/E2B/MoE/qwen35).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn execute_static(be_: &VulkanBackend, graph: &Graph, bindings: &Bindings) -> Result<()> {
    let scratch = alloc_scratch(be_, graph)?;

    // RoPE position: the static `qk_norm_rope`/`rope` kernels take a scalar `rope_pos`, but the IR
    // carries a `positions` i32 tensor. Read `positions[0]` (decode rows=1, or the start of a
    // consecutive-prefill run) up front — `read_pos0` reads host-visible (mapped) positions
    // directly (the seam always binds Staging there) and only falls back to the syncing
    // `download` (a one-shot submit + wait) for a device-local buffer.
    let mut rope_pos: HashMap<u32, usize> = HashMap::new();
    for op in &graph.ops {
        let pid = match op {
            Op::Rope { positions, .. } | Op::QkNormRope { positions, .. } => Some(*positions),
            _ => None,
        };
        if let Some(pid) = pid {
            if let std::collections::hash_map::Entry::Vacant(e) = rope_pos.entry(pid.0) {
                e.insert(read_pos0(be_, resolve(&scratch, bindings, pid)?)? as usize);
            }
        }
    }

    let mut plan = infr_core::fusion::plan_fusions(graph, &fusion_cfg(be_));
    // Dense layer streaming: the fused Linear+Add kernels bake a ZERO weight offset, so un-fuse
    // any pair whose Linear weight is a streamed block — the standalone Add op runs instead
    // (bit-identical math: the fused kernel only moves the same exact add in-kernel).
    if be_.dense_paged() {
        let guard = be_.dense_pager().lock().unwrap();
        if let Some(sess) = guard.as_ref() {
            let mut unfuse: Vec<usize> = Vec::new();
            for &idx in plan.linear_add.keys() {
                if let Op::Linear { weight, .. } = &graph.ops[idx] {
                    let w = resolve(&scratch, bindings, *weight)?;
                    if sess.is_streamed(crate::pager::buffer_identity(w)) {
                        unfuse.push(idx);
                    }
                }
            }
            for idx in unfuse {
                plan.linear_add.remove(&idx);
                plan.skip.remove(&(idx + 1));
            }
        }
    }
    let fused_kv_write = plan.kv_write;
    let fused_add = plan.linear_add;
    let mut skip_op = plan.skip;
    // Strict decode-only Qwen peephole. Only suppress the six dense shared-expert ops when the
    // routed bank is actually backed by the Vulkan pager; resident/CPU/other-model graphs keep the
    // original op stream. `PagedMoeShared` is Copy and contains only graph handles.
    let mut paged_moe_shared = HashMap::<usize, PagedMoeShared>::new();
    for op_idx in 0..graph.ops.len() {
        let Some(shared) = paged_moe_shared_at(graph, op_idx) else {
            continue;
        };
        let Op::MoeFfn { gate_exps, .. } = &graph.ops[op_idx] else {
            unreachable!();
        };
        let gbuf = resolve(&scratch, bindings, *gate_exps)?;
        let paged = be_.moe_pager().lock().unwrap().as_ref().is_some_and(|s| {
            s.is_paged(
                crate::pager::Role::Gate,
                crate::pager::buffer_identity(gbuf),
            )
        });
        if paged {
            paged_moe_shared.insert(op_idx, shared);
            skip_op.extend(op_idx + 1..op_idx + 7);
        }
    }

    // Transient buffers allocated inside the op loop (GEMM/attention/MoE scratch) must outlive the
    // recorder — hold them here so they drop only after `rec.finish()` submits.
    let mut transient: Vec<Box<dyn Buffer>> = Vec::new();

    // `rec` is an `Option` (not a plain binding) because a PAGED `MoeFfn` op may hand the ambient
    // segment off mid-graph: `execute_paged_moe` records the whole MoE op INLINE into it on the
    // no-readback paths, and only rotates (pipelined `finish_nowait`, ring-half backpressure) or
    // blocks (the small-m split's router readback) when residency actually demands it — see
    // `PagedStream`'s doc. Every non-paged graph (the overwhelming common case) never touches any
    // of this and records exactly like before — one recorder, one submit.
    let mut rec = Some(be_.recorder()?);
    let mode = RopeMode::Static(&rope_pos);
    let mut dyn_args: Vec<DynAttnCtx> = Vec::new();
    let mut pool: ScratchPool = HashMap::new();
    let mut mmv_memo: Option<(TensorId, usize, usize)> = None;
    let mut pstream = PagedStream::default();

    // ── submit splitter ───────────────────────────────────────────────────────────────────────
    // The GPU hang watchdog is armed per SUBMIT, so recording the whole forward into ONE command
    // buffer makes the whole forward one indivisible watchdog job. That is fine on a discrete GPU
    // (tens of ms) and fatal on a slow integrated one, where a Qwen3-8B prefill chunk is ~2.05 s
    // of GPU against a device that kills a job at ~2.06 s. `cap` (0 = unlimited, the discrete
    // default) bounds the dispatches per command buffer; segments are submitted WITHOUT waiting
    // and run back-to-back on the queue, so the GPU sees the same uninterrupted stream of work —
    // only the watchdog's view changes, from one long job to several short ones.
    let cap = be_.submit_dispatch_cap();
    let pager_prof = infr_core::pager_profile::active();
    let submit_before = pager_prof.then(infr_core::pager_profile::queue_submit_count);
    let t_forward = std::time::Instant::now();
    let mut submitted_dispatches = 0usize;
    /// In-flight segments allowed at once. Each one pins a command buffer plus its descriptor
    /// pools until the GPU is done reading them, and the devices that split are exactly the
    /// memory-tight ones — letting every segment of a forward pile up (6+ on a Qwen3-8B prefill
    /// chunk) is what made RADV report `Not enough memory for command submission` on the 2-CU
    /// iGPU under the validation layer. Two keeps the pipeline full — the CPU records segment
    /// N+1 while the GPU chews N — at a bounded, constant cost.
    const MAX_INFLIGHT: usize = 2;
    // Submitted-but-unwaited segments, oldest first. They hold command buffers/descriptor pools
    // the GPU is still reading, so they must outlive every buffer they reference — `transient`,
    // `pool` and the scratch arena all live to the end of this function, and the drain below
    // happens before any of them drop.
    let mut segments: std::collections::VecDeque<crate::recorder::PendingSegment> =
        std::collections::VecDeque::new();
    // Graph order with the fused-away indices elided — the shared walk definition
    // (`infr_core::exec::live_ops`). Only the WALK is shared: everything between ops below (the
    // submit splitter, the shutdown poll, the paged-MoE hand-off, the dense-streaming stage) is
    // device-specific and stays here, so this path keeps its own loop body rather than an
    // `OpDispatch`. `record_decode_replay` can't even use this much — its skip set is MUTATED
    // mid-walk by the E2B `Linear`+`GatedAct` peephole.
    for (op_idx, op) in infr_core::exec::live_ops(&graph.ops, &skip_op) {
        // ── shutdown (SIGINT/SIGTERM) ─────────────────────────────────────────────────────────
        // Polled HERE — at the op/submit boundary INSIDE the forward — and not merely between
        // tokens, because on the devices that split (§ the splitter above) a single forward is
        // tens of seconds of GPU: a token-boundary-only check would make Ctrl-C during an iGPU
        // prefill sit for a whole forward before it took effect, which is exactly the window in
        // which a impatient second signal (or a `timeout` SIGKILL) kills the process mid-submit
        // and wedges the device. Stopping at an op boundary bounds the wait by ONE segment.
        //
        // A submitted command buffer CANNOT be cancelled, so this is a "stop recording", never a
        // "stop waiting". Two halves:
        //   * the segment being RECORDED right now is not submitted at all — `discard` frees it
        //     (its dispatches are waste: this forward is being abandoned, and handing the driver
        //     more work is the last thing we want while trying to stop);
        //   * everything ALREADY submitted is drained to its fence, exactly as the success path
        //     does. That is the irreducible part of the wait, and it is why the in-flight window
        //     (`MAX_INFLIGHT`) is the real latency bound.
        if shutdown_requested() {
            let partial = rec.take().expect("segment always Some between ops");
            partial.discard().map_err(|e| be(e.to_string()))?;
            for seg in segments {
                seg.wait().map_err(|e| be(e.to_string()))?;
            }
            pstream.drain()?;
            pstream.finish_prefill_uploads(be_)?;
            // `transient`, `pool`, `dyn_args` and the scratch arena drop AFTER these waits (they
            // are locals of this fn), so nothing the GPU was reading is freed under it.
            return Err(Error::Aborted);
        }
        // Split BETWEEN ops, never inside one: a single op can lower to several dispatches that
        // share transient scratch, and the hazard tracking that orders them is per-recorder.
        // Crossing the cap therefore closes the segment at the next op boundary.
        if cap > 0 && rec.as_ref().is_some_and(|r| r.dispatches() >= cap) {
            let seg = rec
                .take()
                .expect("segment always Some between ops")
                .finish_nowait()
                .map_err(|e| be(e.to_string()))?;
            submitted_dispatches += seg.dispatches();
            segments.push_back(seg);
            // Retire the oldest in-flight segment once the window is full. Queue execution is
            // FIFO, so by the time we are recording segment N+2 the GPU has almost always already
            // finished N — in steady state this fence is already signaled and the wait is free.
            while segments.len() > MAX_INFLIGHT {
                let old = segments.pop_front().expect("len > MAX_INFLIGHT >= 0");
                old.wait().map_err(|e| be(e.to_string()))?;
            }
            let fresh = be_.recorder()?;
            // Hazard tracking is per-recorder and starts empty, so cross-segment RAW/WAR ordering
            // (every layer reads the residual stream the previous segment wrote) is invisible to
            // it and MUST be seeded explicitly. `vkCmdPipelineBarrier`'s scope spans submission
            // order on the queue, so this one barrier orders the whole segment after everything
            // already submitted.
            fresh.seed_barrier();
            rec = Some(fresh);
        }
        if let Op::MoeFfn { gate_exps, .. } = op {
            let gbuf = resolve(&scratch, bindings, *gate_exps)?;
            let paged = be_.moe_pager().lock().unwrap().as_ref().is_some_and(|s| {
                s.is_paged(
                    crate::pager::Role::Gate,
                    crate::pager::buffer_identity(gbuf),
                )
            });
            if paged {
                execute_paged_moe(
                    be_,
                    graph,
                    op,
                    &scratch,
                    bindings,
                    &mut pool,
                    &mut rec,
                    &mut pstream,
                    paged_moe_shared.get(&op_idx).copied(),
                )?;
                continue;
            }
        }
        // Dense layer streaming: a `Op::Linear` whose weight buffer is a registered streamed
        // block stages its residency (recorded ring→arena copy, pipelined via the same
        // `PagedStream` rotation the MoE path uses — the CPU's memcpys for later layers overlap
        // the GPU's execution of already-submitted segments), then lowers through the ORDINARY
        // Linear arm with the pool arena + slot element base substituted for the placeholder
        // (see `lower_op`'s `wsub` param). No readbacks anywhere: layer order is deterministic,
        // so every miss is known the moment the op is reached.
        if be_.dense_paged() {
            if let Op::Linear { weight, .. } = op {
                let wid = crate::pager::buffer_identity(resolve(&scratch, bindings, *weight)?);
                let streamed = be_
                    .dense_pager()
                    .lock()
                    .unwrap()
                    .as_ref()
                    .is_some_and(|s| s.is_streamed(wid));
                if streamed {
                    let arena_addr = stage_dense_linear(be_, &mut rec, &mut pstream, wid)?;
                    if let Err(e) = lower_op(
                        be_,
                        graph,
                        op_idx,
                        op,
                        rec.as_ref().expect("segment always Some between ops"),
                        &scratch,
                        bindings,
                        &fused_kv_write,
                        &fused_add,
                        &mode,
                        &mut transient,
                        &mut dyn_args,
                        &mut pool,
                        &mut mmv_memo,
                        Some(arena_addr),
                    ) {
                        return Err(abort_segment(rec.take(), e));
                    }
                    continue;
                }
            }
        }
        if let Err(e) = lower_op(
            be_,
            graph,
            op_idx,
            op,
            rec.as_ref().expect("segment always Some between ops"),
            &scratch,
            bindings,
            &fused_kv_write,
            &fused_add,
            &mode,
            &mut transient,
            &mut dyn_args,
            &mut pool,
            &mut mmv_memo,
            None,
        ) {
            return Err(abort_segment(rec.take(), e));
        }
    }
    for c in dyn_args.drain(..) {
        transient.extend([c.args, c.pm, c.pl, c.pacc]);
    }
    transient.extend(pool.into_values());
    let last = rec.take().expect("segment always Some at loop end");
    submitted_dispatches += last.dispatches();
    last.finish().map_err(|e| be(e.to_string()))?;
    // The blocking finish above already waited the queue idle — draining just releases the
    // pipelined segments' command buffers/fences (every buffer they referenced outlives this
    // call: pooled scratch in `transient`, ring/tape/arenas on the session).
    for seg in segments {
        seg.wait().map_err(|e| be(e.to_string()))?;
    }
    pstream.drain()?;
    pstream.finish_prefill_uploads(be_)?;
    // Feed this forward back into the splitter: `finish` waited the queue idle, so the elapsed
    // time now covers every segment's GPU execution. See `VulkanBackend::observe_forward`.
    be_.observe_forward(t_forward.elapsed(), submitted_dispatches);
    if let Some(before) = submit_before {
        let submits = infr_core::pager_profile::queue_submit_count().saturating_sub(before);
        infr_core::pager_profile::record_splitter_forward(
            cap,
            be_.submit_dispatch_cap(),
            submitted_dispatches,
            submits,
        );
    }
    Ok(())
}

/// Get-or-alloc a pooled buffer with an explicit `usage` (unlike [`pooled`], which is always
/// `Activations`) — the paged MoE router-ids scratch needs `Staging` (host-visible, so the
/// small-m split path's readback is a plain mapped memcpy with no extra submit).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn pooled_usage(
    pool: &mut ScratchPool,
    be_: &VulkanBackend,
    tag: &'static str,
    bytes: usize,
    usage: BufferUsage,
) -> Result<(&'static str, usize)> {
    let key = (tag, bytes.max(4));
    if let std::collections::hash_map::Entry::Vacant(e) = pool.entry(key) {
        e.insert(be_.alloc_uninit(key.1, usage)?);
    }
    Ok(key)
}

/// Host side of paged execution, per `execute_static` call: Dense streaming owns the ring cursor,
/// MoE owns the LUT cursor plus in-flight compute, and both end fully drained. The full MoE payload
/// is CPU-only; runtime transfers are direct writes into the mapped ReBAR arena.
///
/// Ring rotation (`rotate_stream`): when the current ring half can't hold a miss, the ambient
/// recorder is submitted WITHOUT waiting (`Recorder::finish_nowait`) and staging continues into
/// the other half — the CPU's expert memcpys overlap the GPU's execution of the segment just
/// submitted. Before reusing a half, its previous segment's fence is waited (`pending`), which
/// is the whole ring-region-lifetime story: a region is never rewritten until the recording that
/// read it has fully executed.
struct PrefillUploadCommand {
    job: crate::pager::PrefillCopyJob,
    /// The layer that previously owned this lane. The uploader may overwrite the lane only after
    /// this GPU fence signals. Initial free-lane fills have no dependency.
    after: Option<crate::recorder::PendingSegment>,
}

struct PrefillUploadCompletion {
    buf_id: usize,
    result: std::result::Result<(), String>,
}

struct PrefillUploader {
    tx: std::sync::mpsc::Sender<Option<PrefillUploadCommand>>,
    rx: std::sync::mpsc::Receiver<PrefillUploadCompletion>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PrefillUploader {
    fn new() -> Self {
        let (tx, jobs) = std::sync::mpsc::channel::<Option<PrefillUploadCommand>>();
        let (done, rx) = std::sync::mpsc::channel::<PrefillUploadCompletion>();
        let thread = std::thread::Builder::new()
            .name("infr-moe-prefill-upload".to_owned())
            .spawn(move || {
                while let Ok(Some(command)) = jobs.recv() {
                    let buf_id = command.job.buf_id();
                    let result = command
                        .after
                        .map_or(Ok(()), |segment| segment.wait().map_err(|e| e.to_string()));
                    let result =
                        result.and_then(|()| command.job.execute().map_err(|e| e.to_string()));
                    if done
                        .send(PrefillUploadCompletion { buf_id, result })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("spawn Prefill uploader");
        Self {
            tx,
            rx,
            thread: Some(thread),
        }
    }

    fn enqueue(&self, command: PrefillUploadCommand) -> Result<()> {
        self.tx
            .send(Some(command))
            .map_err(|_| be("Prefill uploader stopped unexpectedly"))
    }

    fn finish(&mut self) {
        let _ = self.tx.send(None);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for PrefillUploader {
    fn drop(&mut self) {
        self.finish();
    }
}

#[derive(Default)]
struct PagedStream {
    /// Per-ring-region in-flight segment (the one whose recorded copies read that region).
    pending: Vec<Option<crate::recorder::PendingSegment>>,
    /// Which ring region `cursor` allocates from.
    half: usize,
    /// Bytes used in the current region.
    cursor: usize,
    /// Words used in the session's LUT tape (reset only after full drains — `sync_stream`).
    tape_cursor: usize,
    /// Prefill compute segments that do not consume a staging-ring region.
    compute_pending: std::collections::VecDeque<crate::recorder::PendingSegment>,
    /// Dedicated producer for whole-layer Host -> mapped-ReBAR copies. It owns the layer fence
    /// while waiting for a ring lane to become reusable, so upload progress is independent from
    /// graph recording and from whether the active layer is Attention or DeltaNet.
    prefill_uploader: Option<PrefillUploader>,
}

impl PagedStream {
    fn ensure_slots(&mut self, slots: usize) {
        debug_assert!(slots >= 2);
        if self.pending.is_empty() {
            self.pending.resize_with(slots, || None);
        } else {
            debug_assert_eq!(self.pending.len(), slots);
        }
    }

    fn drain(&mut self) -> Result<()> {
        for p in &mut self.pending {
            if let Some(s) = p.take() {
                s.wait().map_err(|e| be(e.to_string()))?;
            }
        }
        while let Some(s) = self.compute_pending.pop_front() {
            s.wait().map_err(|e| be(e.to_string()))?;
        }
        Ok(())
    }

    fn apply_prefill_completion(
        be_: &VulkanBackend,
        completion: PrefillUploadCompletion,
    ) -> Result<()> {
        completion
            .result
            .map_err(|e| be(format!("async Prefill upload dependency failed: {e}")))?;
        be_.moe_pager()
            .lock()
            .unwrap()
            .as_mut()
            .expect("paged execution requires a session")
            .complete_prefill_layer_cpu(completion.buf_id)
    }

    fn poll_prefill_uploads(&mut self, be_: &VulkanBackend) -> Result<()> {
        let Some(uploader) = self.prefill_uploader.as_ref() else {
            return Ok(());
        };
        while let Ok(completion) = uploader.rx.try_recv() {
            Self::apply_prefill_completion(be_, completion)?;
        }
        Ok(())
    }

    /// Returns false when no async job owns this layer, allowing the caller's cold synchronous
    /// path to seed the first current layer. A pending layer blocks only at actual consumption.
    fn wait_prefill_layer(&mut self, be_: &VulkanBackend, buf_id: usize) -> Result<bool> {
        loop {
            self.poll_prefill_uploads(be_)?;
            let (ready, pending) = {
                let guard = be_.moe_pager().lock().unwrap();
                let sess = guard.as_ref().expect("paged execution requires a session");
                (
                    sess.layer_bank_current(buf_id)?,
                    sess.layer_bank_pending(buf_id)?,
                )
            };
            if ready {
                return Ok(true);
            }
            if !pending {
                return Ok(false);
            }
            // Reuse the existing blocked-wait counter: in layer-stream mode there is no dense
            // staging-ring acquisition, so this is exactly the consumer time lost because the
            // async producer had not filled the needed lane yet. Profiling disabled means only
            // the cheap `start()` feature check remains.
            let wait_t0 = infr_core::pager_profile::start();
            let completion = self
                .prefill_uploader
                .as_ref()
                .ok_or_else(|| be("Prefill layer is pending without an uploader"))?
                .rx
                .recv()
                .map_err(|_| be("Prefill uploader disconnected before layer became ready"))?;
            if let Some(elapsed) = infr_core::pager_profile::elapsed(wait_t0) {
                infr_core::pager_profile::record_staging_wait(elapsed);
            }
            Self::apply_prefill_completion(be_, completion)?;
        }
    }

    fn enqueue_prefill_upload(
        &mut self,
        job: crate::pager::PrefillCopyJob,
        after: Option<crate::recorder::PendingSegment>,
    ) -> Result<()> {
        self.prefill_uploader
            .get_or_insert_with(PrefillUploader::new)
            .enqueue(PrefillUploadCommand { job, after })
    }

    fn finish_prefill_uploads(&mut self, be_: &VulkanBackend) -> Result<()> {
        let Some(mut uploader) = self.prefill_uploader.take() else {
            return Ok(());
        };
        uploader.finish();
        while let Ok(completion) = uploader.rx.try_recv() {
            Self::apply_prefill_completion(be_, completion)?;
        }
        Ok(())
    }
}

/// Submit the ambient segment without waiting and rotate to the other ring half (fencing out its
/// previous occupant) — the pipelined flush `PagedStream`'s doc describes. The fresh recorder
/// opens with `seed_barrier`: hazard tracking is per-recorder, so ordering against the in-flight
/// segment (pooled-scratch reuse, arena-slot rewrites vs its reads) must be seeded explicitly.
fn rotate_stream<'a>(
    be_: &'a VulkanBackend,
    rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
) -> Result<()> {
    let acquire_t0 = infr_core::pager_profile::start();
    let seg = rec
        .take()
        .expect("segment always Some between ops")
        .finish_nowait()
        .map_err(|e| be(e.to_string()))?;
    debug_assert!(!ps.pending.is_empty());
    ps.pending[ps.half] = Some(seg);
    ps.half = (ps.half + 1) % ps.pending.len();
    ps.cursor = 0;
    if let Some(prev) = ps.pending[ps.half].take() {
        let wait_t0 = infr_core::pager_profile::start();
        prev.wait().map_err(|e| be(e.to_string()))?;
        if let Some(elapsed) = infr_core::pager_profile::elapsed(wait_t0) {
            infr_core::pager_profile::record_staging_wait(elapsed);
        }
    }
    let fresh = be_.recorder()?;
    fresh.seed_barrier();
    *rec = Some(fresh);
    if let Some(elapsed) = infr_core::pager_profile::elapsed(acquire_t0) {
        infr_core::pager_profile::record_staging_acquire(elapsed);
    }
    Ok(())
}

/// Retain a submitted Prefill segment that has no future ring replacement to guard. This is only
/// the tail of the model; steady-state segments move to the uploader together with the refill job
/// whose destination becomes writable when that segment's fence signals.
fn retain_prefill_compute(
    ps: &mut PagedStream,
    seg: crate::recorder::PendingSegment,
) -> Result<()> {
    ps.compute_pending.push_back(seg);
    while ps.compute_pending.len() > 2 {
        let old = ps.compute_pending.pop_front().expect("len > 2");
        old.wait().map_err(|e| be(e.to_string()))?;
    }
    Ok(())
}

// The small-m Decode Down-overlap path also submits the Gate/Up work without waiting, then
// observes whether it is still live while the missing Down weights are pushed. Keep that probe
// independent from the Prefill layer-ring producer: Decode may use either the dense staging ring
// or the general compute queue, but it never transfers ownership of its segment to the Prefill
// uploader.
#[derive(Clone, Copy)]
enum PrefetchComputeProbe {
    Ring(usize),
    General,
}

fn prefetch_compute_live(ps: &PagedStream, probe: PrefetchComputeProbe) -> Result<bool> {
    let segment = match probe {
        PrefetchComputeProbe::Ring(slot) => ps.pending.get(slot).and_then(Option::as_ref),
        PrefetchComputeProbe::General => ps.compute_pending.back(),
    };
    segment
        .map(|seg| seg.is_complete().map(|complete| !complete))
        .unwrap_or(Ok(false))
}

fn submit_prefill_compute<'a>(
    rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
) -> Result<PrefetchComputeProbe> {
    let seg = rec
        .take()
        .expect("segment always Some between ops")
        .finish_nowait()
        .map_err(|e| be(e.to_string()))?;
    if ps.cursor > 0 {
        let slot = ps.half;
        debug_assert!(ps.pending[ps.half].is_none());
        ps.pending[ps.half] = Some(seg);
        ps.half = (ps.half + 1) % ps.pending.len();
        ps.cursor = 0;
        Ok(PrefetchComputeProbe::Ring(slot))
    } else {
        ps.compute_pending.push_back(seg);
        while ps.compute_pending.len() > 2 {
            let old = ps.compute_pending.pop_front().expect("len > 2");
            old.wait().map_err(|e| be(e.to_string()))?;
        }
        Ok(PrefetchComputeProbe::General)
    }
}

/// Blocking submit of the ambient segment (queue idle on return) + full stream drain — the ONE
/// remaining host sync on the paged path, paid only by a small-m layer whose expert set is not
/// fully resident (its router ids must be read back before its GEMVs can be staged). Both ring
/// halves and the whole tape are reusable afterwards.
fn sync_stream<'a>(
    be_: &'a VulkanBackend,
    rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
) -> Result<()> {
    let sync_t0 = infr_core::pager_profile::start();
    rec.take()
        .expect("segment always Some between ops")
        .finish()
        .map_err(|e| be(e.to_string()))?;
    ps.drain()?; // fences already signaled (queue idle) — releases their transient objects
    ps.cursor = 0;
    ps.tape_cursor = 0;
    *rec = Some(be_.recorder()?);
    if let Some(elapsed) = infr_core::pager_profile::elapsed(sync_t0) {
        infr_core::pager_profile::record_paging_sync_wait(elapsed);
    }
    Ok(())
}

/// Does a streamed dense `Op::Linear` at `(dt, m, in_f, out_f)` route to the arena-addressed
/// coopmat-warp prefill GEMM (`streamed_prefill_gemm`), or fall back to the streamed GEMV? Mirrors
/// the RESIDENT `is_gemm && native_dense_supported` gate so the streamed route tracks the resident
/// route exactly — just arena-addressed. A genuine prefill chunk (m>16, past the resident mrow
/// window) with a coopmat-warp-eligible native quant takes the GEMM; decode/small-m/verify keep the
/// GEMV (the task's "small-m keeps the GEMV"). The one native-quant shape the resident path routes
/// to the dp4a `mmq` GEMM instead of the coopmat-warp arm — Q4_K with out_f%128!=0 (no warp tile) —
/// is NOT arena-converted (its shader reads the weight outside native_decode's NW chokepoint), so it
/// too keeps the GEMV; every other native quant reaches the coopmat-warp arm `streamed_prefill_gemm`
/// mirrors. Also mirrors the resident `warp_ok` gate's `INFR_NO_GEMM_WARP` escape hatch: with it
/// set, the resident path drops out of the coopmat-warp arm entirely (falling to mmq/off-tier), so
/// the streamed route must fall back to the GEMV too — otherwise a Q4_K out_f%128==0 shape would
/// diverge (resident: mmq: streamed: still the coopmat GEMM twin), which can panic the twin-SPV
/// lookup under that debug knob.
fn streamed_gemm_applies(
    be_: &VulkanBackend,
    dt: infr_core::DType,
    m: usize,
    in_f: usize,
    out_f: usize,
) -> bool {
    m > 16
        && out_f.is_multiple_of(64)
        && in_f.is_multiple_of(32)
        && be_.caps().f16_coopmat()
        && native_dense_supported(dt)
        && be_.cfg().kernels.vulkan.gemm_warp
        && !(matches!(dt, infr_core::DType::Q4K)
            && !out_f.is_multiple_of(128)
            && be_.caps().i8_dot
            && be_.cfg().kernels.vulkan.mmq)
}

/// Arena-addressed twin of the resident `Op::Linear` `is_gemm && native_dense_supported` coopmat-
/// warp prefill arm (see the `is_gemm` block in `lower_op`): the SAME tile selection (A_GLOBAL
/// f16-A cast, narrow-n split-K, or the direct tile) and the SAME padded-dst dance, but the weight
/// is read from the pool arena by 64-bit address (`arena_addr`) — the recorder methods pass the
/// address to whatever tile they pick and bind the activation as the binding-1 filler, so the
/// arena is NEVER a bound descriptor. `w_off` (a fused-QKV slice offset)
/// rides on top as a within-slot element offset. The ring→arena copy is already ordered before this
/// dispatch by `stage_dense_linear`'s RAW `arena_stream_barrier` (same slot, same segment).
#[allow(clippy::too_many_arguments)]
fn streamed_prefill_gemm(
    be_: &VulkanBackend,
    graph: &Graph,
    dt: infr_core::DType,
    arena_addr: u64,
    w_off: usize,
    xb: &dyn Buffer,
    y: &dyn Buffer,
    dst: &TensorId,
    m: usize,
    in_f: usize,
    out_f: usize,
    rec: &Recorder<'_>,
    pool: &mut ScratchPool,
    transient: &mut Vec<Box<dyn Buffer>>,
) -> Result<()> {
    // Padded-dst dance + tile selection are shared with the resident native-dense arm (the whole
    // point of this twin is that they track EXACTLY) — read from `arena_addr` instead of a bound
    // weight. `with_padded_dst` handles the ceil(m/64)*64 row padding; `native_warp_gemm` picks the
    // A_GLOBAL / split-K / direct tile.
    with_padded_dst(be_, graph, rec, dst, y, m, out_f, transient, |out| {
        native_warp_gemm(
            be_, dt, arena_addr, w_off, xb, out, m, in_f, out_f, rec, pool,
        )
    })
}

/// Ensure a streamed dense Linear weight (`wid` — see `pager::buffer_identity`) is resident,
/// staging a miss through the session ring into the ambient segment and rotating the stream
/// (pipelined `finish_nowait` + fenced ring-half swap, exactly `rotate_stream`'s contract) when
/// the current half can't hold it. Returns the resident slot's arena base BYTE address for the
/// streamed dispatch (see `DensePagerSession::stage`). Progress is guaranteed: a ring half holds
/// at least the largest pool slot (asserted at session construction).
fn stage_dense_linear<'a>(
    be_: &'a VulkanBackend,
    rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
    wid: usize,
) -> Result<u64> {
    loop {
        // WAR: order any prior streamed dispatch's arena read (a 64-bit pointer read, invisible to
        // the buffer hazard tracker) before the ring→arena copy this `stage` may record into a
        // re-used slot. Emitted before the copy; harmless on a hit (no copy recorded).
        rec.as_ref()
            .expect("segment always Some between ops")
            .arena_stream_barrier();
        let staged = {
            let mut guard = be_.dense_pager().lock().unwrap();
            let sess = guard
                .as_mut()
                .expect("dense streamed execution requires a session");
            ps.ensure_slots(sess.ring_slots());
            let half_base = ps.half * sess.ring_half_bytes();
            sess.stage(
                rec.as_ref().expect("segment always Some between ops"),
                half_base,
                &mut ps.cursor,
                wid,
            )?
        };
        match staged {
            Some(arena_addr) => {
                // RAW: order the ring→arena copy just recorded before the streamed dispatch that
                // reads the slot by pointer (see `Recorder::arena_stream_barrier`).
                rec.as_ref()
                    .expect("segment always Some between ops")
                    .arena_stream_barrier();
                return Ok(arena_addr);
            }
            None => rotate_stream(be_, rec, ps)?,
        }
    }
}

/// CPU-push `ids` (layer-LOCAL) of `buf_id`'s role from the unique host store into final ReBAR LRU
/// slots, then freeze the layer's LUT window. `ids` empty means residency is already guaranteed.
fn stage_and_window<'a>(
    be_: &'a VulkanBackend,
    rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
    buf_id: usize,
    ids: &[u32],
    n_expert: usize,
    scan: bool,
    batch_open: bool,
) -> Result<u32> {
    // One epoch batch per (layer, role): the epoch, not LRU position, protects this batch's
    // earlier ids from its later misses on the cold-insert path.
    if !batch_open {
        let mut guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_mut().expect("paged execution requires a session");
        sess.begin_batch(buf_id)?;
    }
    if !ids.is_empty() {
        let push = {
            let mut guard = be_.moe_pager().lock().unwrap();
            let sess = guard.as_mut().expect("paged execution requires a session");
            sess.push_role_cpu(buf_id, ids, scan)?
        };
        if let Some(recorder) = rec.as_ref() {
            push.record(recorder)?;
        } else {
            // Full-Host-Store Down overlap intentionally has no open recorder here: Gate/Up is
            // already executing while the host writes Down. Dropping preserves that established
            // CPU-copy window instead of serializing Down DMA behind the submitted compute.
            drop(push);
        }
    }
    let mut guard = be_.moe_pager().lock().unwrap();
    let sess = guard.as_mut().expect("paged execution requires a session");
    sess.lut_window(&mut ps.tape_cursor, buf_id, n_expert)
}

/// Whole-layer Prefill twin of [`stage_and_window`]. The first role encountered pushes exactly one
/// load-time-contiguous layer; the other role windows reuse it. No expert pager/LRU is touched.
fn stage_layer_and_window<'a>(
    be_: &'a VulkanBackend,
    _rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
    buf_id: usize,
    n_expert: usize,
) -> Result<u32> {
    {
        let mut guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_mut().expect("paged execution requires a session");
        sess.enter_prefill_layer()?;
    }
    let already_current = ps.wait_prefill_layer(be_, buf_id)?;
    let mut guard = be_.moe_pager().lock().unwrap();
    let sess = guard.as_mut().expect("paged execution requires a session");
    if !already_current {
        sess.push_prefill_layer_cpu(buf_id)?;
    }
    sess.layer_lut_window(&mut ps.tape_cursor, buf_id, n_expert)
}

fn run_prefill_job_sync(be_: &VulkanBackend, job: crate::pager::PrefillCopyJob) -> Result<()> {
    let buf_id = job.buf_id();
    job.execute()?;
    be_.moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .expect("paged execution requires a session")
        .complete_prefill_layer_cpu(buf_id)
}

/// Close every Prefill layer into its own asynchronous segment and hand its fence to a dedicated
/// producer. Initial free lanes are filled immediately. Thereafter each completed GPU layer frees
/// exactly its modulo lane and triggers the next layer at distance `lane_count`; the producer keeps
/// draining that queue regardless of whether Attention or DeltaNet is active. The graph thread
/// waits only if it catches the producer at the actual point of consumption.
fn prefetch_next_moe_layer<'a>(
    be_: &'a VulkanBackend,
    rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
    current_gate_id: usize,
) -> Result<()> {
    ps.poll_prefill_uploads(be_)?;
    // Dense streaming can attach staging-ring lifetime to this same segment. That uncommon mixed
    // path keeps a synchronous correctness fallback; ordinary paged MoE has cursor=0 and uses the
    // fully asynchronous worker below.
    let synchronous = ps.cursor > 0;
    if synchronous {
        ps.finish_prefill_uploads(be_)?;
    }

    let (initial, replacement) = {
        let mut guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_mut().expect("paged execution requires a session");
        sess.enter_prefill_layer()?;
        sess.prefill_successors(current_gate_id)?
    };
    let (initial_jobs, replacement_job) = {
        let mut guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_mut().expect("paged execution requires a session");
        let mut jobs = Vec::with_capacity(initial.len());
        for next in initial {
            if let Some(job) = sess.prepare_prefill_layer_cpu(next)? {
                jobs.push(job);
            }
        }
        let replacement = replacement
            .map(|next| sess.prepare_prefill_layer_cpu(next))
            .transpose()?
            .flatten();
        (jobs, replacement)
    };

    let segment = rec
        .take()
        .expect("segment always Some between ops")
        .finish_nowait()
        .map_err(|e| be(e.to_string()))?;
    if synchronous {
        segment.wait().map_err(|e| be(e.to_string()))?;
        ps.cursor = 0;
        for job in initial_jobs {
            run_prefill_job_sync(be_, job)?;
        }
        if let Some(job) = replacement_job {
            run_prefill_job_sync(be_, job)?;
        }
    } else {
        for job in initial_jobs {
            ps.enqueue_prefill_upload(job, None)?;
        }
        if let Some(job) = replacement_job {
            ps.enqueue_prefill_upload(job, Some(segment))?;
        } else {
            retain_prefill_compute(ps, segment)?;
        }
    }

    let compute = be_.recorder()?;
    compute.seed_barrier();
    *rec = Some(compute);
    Ok(())
}

/// Execute ONE paged `Op::MoeFfn`, recording INLINE into the ambient segment wherever residency
/// allows — the host-orchestration rework that removed the old per-layer
/// submit→readback→touch/upload→submit cadence (~4 full-pipeline syncs per layer per ubatch
/// chunk; INFR_PROF_OPS measured the GPU ~89% idle on Scout pp512 under it). Three paths:
///
///   - BATCHED chunks (`rows·n_used >= 3·n_expert`): the router readback is dropped entirely —
///     routing that wide touches every expert of the layer with overwhelming probability
///     (P(expert unrouted) < 1e-8 at the 3x bound), so all `n_expert` are staged up front and
///     the GPU-side bucket count/scan/scatter pipeline needs no host decision at all. The whole
///     chunk records inline. Layer mode uses a completion-driven whole-layer upload ring;
///     expert mode directly copies the required expert offsets into LRU slots.
///   - Small-m (decode) layers whose full expert set is resident: `touch_all_hits` (LRU upkeep,
///     no residency/LUT mutation) + a frozen LUT window, recorded inline — zero host syncs.
///   - Small-m layers with any absent expert: the ONE remaining blocking sync (`sync_stream`) to
///     read the router's ids — a mapped memcpy, the ids buffer is `Staging` — then routed-only
///     staging into the fresh ambient segment (which stays open for the next layer).
///
/// Every path reads layer-LOCAL expert ids against a frozen per-(layer, role) LUT window in the
/// session's tape (`lut[window + local_id]` — see `MoePagerSession::lut_window`), never the live
/// pool LUT, which later layers' staging keeps mutating while earlier recorded work is still in
/// flight. The arena itself is BDA-addressed (`48ad9c1`): the paged dispatches deref it by 64-bit
/// pointer (`native_weight_addr.glsl`) and deliberately never bind it as a descriptor (binding 0
/// takes a small `lut` filler instead), so the generic buffer hazard tracker (`Recorder::sync`)
/// never sees the read and cannot order it against the direct transfer writes on its own.
/// Ordering is instead explicit, mirroring the dense streamer's `arena_stream_barrier` (`36bcbf5`):
/// one WAR `arena_stream_barrier()` before this op's staging begins (orders every prior arena read
/// — this layer's or an earlier one's — before any copy that may overwrite a slot still being
/// read) and one RAW `arena_stream_barrier()` after all of this op's staging completes and before
/// the first dispatch that reads the arena (`matmul_mmq_experts_paged` batched /
/// `linear_native_id_multi_paged` small-m). A rotation's `seed_barrier` additionally covers the
/// seeded recorders cover the cross-segment Prefill dependency; the two `arena_stream_barrier`
/// calls here cover copies recorded in the ambient segment.
///
/// Only reached from `execute_static`'s loop, and only for a `MoeFfn` whose `gate_exps` buffer is
/// registered in the backend's `MoePagerSession` (see `pager.rs`'s module doc for why that check —
/// not a graph-level flag — is the paging signal). Mirrors the resident arms' math exactly
/// (batched bucket-scatter mmq pipeline / small-m id-GEMV path), including the fused gate_up
/// shape: a fused `ffn_gate_up_exps` bank (gemma-4 MoE / DiffusionGemma) is registered under
/// `Role::Gate` with a double-width ([ne, 2*nff]) slot, so its GEMV/GEMM reads the same
/// arena+window hop with `out_f = 2*nff` and the activation splits gate|up per row exactly like
/// the resident fused arms. `rows`/`n_used` are general (not hardcoded to llama4's rows=1/
/// n_used=1).
#[derive(Clone, Copy)]
struct PagedMoeShared {
    gate_inp: TensorId,
    gate: TensorId,
    wgate: TensorId,
    wup: TensorId,
    wdown: TensorId,
    dst: TensorId,
}

/// Recognize the exact Qwen sigmoid-gated shared-expert tail emitted by `seam::runner`. This is a
/// backend peephole rather than a graph semantic change: other backends and prefill retain the
/// ordinary dense chain, while Vulkan decode can place the shared expert in routed slot 8.
fn paged_moe_shared_at(graph: &Graph, op_idx: usize) -> Option<PagedMoeShared> {
    // Validation/escape hatch: permits same-binary A/B and an immediate fallback if a future
    // model reuses the exact graph shape but violates an unstated packing invariant.
    if std::env::var_os("INFR_NO_MOE_SHARED_SLOT").is_some() {
        return None;
    }
    let Op::MoeFfn {
        x,
        gate_exps,
        up_exps,
        down_exps,
        down_scale: None,
        fused_gate_up: false,
        dst: moe_dst,
        ne,
        n_used,
        n_ff_exp,
        act: Activation::Silu,
        weight_before: false,
        swiglu_clamp: None,
        ..
    } = graph.ops.get(op_idx)?
    else {
        return None;
    };
    if graph.desc(*x).numel() != *ne as usize || *n_used >= 31 {
        return None;
    }
    let supported_routed = |id: TensorId| {
        matches!(
            graph.desc(id).dtype,
            infr_core::DType::Q5K | infr_core::DType::Q6K | infr_core::DType::Iq4Xs
        )
    };
    if !supported_routed(*gate_exps) || !supported_routed(*up_exps) || !supported_routed(*down_exps)
    {
        return None;
    }
    let [gate_op, g_op, u_op, act_op, d_op, add_op] = graph.ops.get(op_idx + 1..op_idx + 7)? else {
        return None;
    };
    let Op::Linear {
        x: gate_x,
        weight: gate_inp,
        dst: gate,
        m: 1,
        in_f: gate_in,
        out_f: 1,
        w_off: 0,
    } = gate_op
    else {
        return None;
    };
    let Op::Linear {
        x: gx,
        weight: wgate,
        dst: gbuf,
        m: 1,
        in_f: gin,
        out_f: gout,
        w_off: 0,
    } = g_op
    else {
        return None;
    };
    let Op::Linear {
        x: ux,
        weight: wup,
        dst: ubuf,
        m: 1,
        in_f: uin,
        out_f: uout,
        w_off: 0,
    } = u_op
    else {
        return None;
    };
    let Op::GatedAct {
        gate: ag,
        up: au,
        dst: abuf,
        rows: 1,
        nff,
        act: Activation::Silu,
        up_off: 0,
        up_stride: 0,
        gate_stride: 0,
        gate_block_width: 0,
        swiglu_clamp: None,
    } = act_op
    else {
        return None;
    };
    let Op::Linear {
        x: dx,
        weight: wdown,
        dst: d_out,
        m: 1,
        in_f: din,
        out_f: dout,
        w_off: 0,
    } = d_op
    else {
        return None;
    };
    let Op::MoeSharedExpertAdd {
        moe,
        shexp,
        gate: add_gate,
        dst,
        rows: 1,
        n,
    } = add_op
    else {
        return None;
    };
    let ne = *ne;
    let nff_exp = *n_ff_exp;
    if *gate_x != *x
        || *gx != *x
        || *ux != *x
        || *gate_in != ne
        || *gin != ne
        || *uin != ne
        || *gout != nff_exp
        || *uout != nff_exp
        || *ag != *gbuf
        || *au != *ubuf
        || *nff != nff_exp
        || *dx != *abuf
        || *din != nff_exp
        || *dout != ne
        || *moe != *moe_dst
        || *shexp != *d_out
        || *add_gate != *gate
        || *n != ne
        || graph.desc(*gate_inp).dtype != infr_core::DType::F32
        || graph.desc(*wgate).dtype != infr_core::DType::Q8_0
        || graph.desc(*wup).dtype != infr_core::DType::Q8_0
        || graph.desc(*wdown).dtype != infr_core::DType::Q8_0
    {
        return None;
    }
    Some(PagedMoeShared {
        gate_inp: *gate_inp,
        gate: *gate,
        wgate: *wgate,
        wup: *wup,
        wdown: *wdown,
        dst: *dst,
    })
}

#[allow(clippy::too_many_arguments)]
fn linear_paged_maybe_shared(
    rec: &Recorder<'_>,
    dtype: infr_core::DType,
    arena_addr: u64,
    slot_bytes: u32,
    lut: &dyn Buffer,
    ids: &dyn Buffer,
    n_used: usize,
    lut_base: usize,
    shared: Option<&dyn Buffer>,
    x: &dyn Buffer,
    x_per_slot: bool,
    y: &dyn Buffer,
    in_f: usize,
    out_f: usize,
    rows: usize,
    active_mask: u32,
) {
    if let Some(shared) = shared {
        debug_assert_eq!(rows, 1);
        rec.linear_native_id_multi_paged_shared(
            dtype,
            arena_addr,
            slot_bytes,
            lut,
            ids,
            n_used,
            lut_base,
            shared,
            x,
            x_per_slot,
            y,
            in_f,
            out_f,
            active_mask,
        );
    } else {
        rec.linear_native_id_multi_paged(
            dtype,
            arena_addr,
            slot_bytes,
            lut,
            ids,
            n_used,
            lut_base,
            x,
            x_per_slot,
            y,
            in_f,
            out_f,
            rows,
            active_mask,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn execute_paged_moe<'a>(
    be_: &'a VulkanBackend,
    graph: &Graph,
    op: &Op,
    scratch: &[Option<Box<dyn Buffer>>],
    bindings: &Bindings,
    pool: &mut ScratchPool,
    rec: &mut Option<Recorder<'a>>,
    ps: &mut PagedStream,
    shared: Option<PagedMoeShared>,
) -> Result<()> {
    use crate::pager::buffer_identity;
    let Op::MoeFfn {
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
        exp_probs_b,
        n_expert_groups,
        n_expert_groups_used,
        swiglu_clamp,
        expert_ids,
        ..
    } = op
    else {
        unreachable!("execute_paged_moe only ever called for an Op::MoeFfn");
    };
    let gating_u32 = match gating {
        infr_core::graph::MoeGating::Softmax => 0u32,
        infr_core::graph::MoeGating::Sigmoid => 1u32,
        infr_core::graph::MoeGating::SqrtSoftplus => 2u32,
    };
    let (ne, n_expert, n_used, nff) = (
        *ne as usize,
        *n_expert as usize,
        *n_used as usize,
        *n_ff_exp as usize,
    );
    // Fused gate_up: the expert slice (and thus the GEMV/GEMM output width) is [ne, 2*nff] —
    // mirrors the resident arms' `gu_width` (`stride` is unused by the paged kernels themselves,
    // the LUT bakes the base; it still shapes the pooled scratch below).
    let gu_width = if *fused_gate_up { 2 * nff } else { nff };
    let (gdt, udt, ddt) = (
        graph.desc(*gate_exps).dtype,
        graph.desc(*up_exps).dtype,
        graph.desc(*down_exps).dtype,
    );
    let rows = graph.desc(*x).numel() / ne;
    let n_slots = rows * n_used;
    let physical_used = n_used + usize::from(shared.is_some());
    let physical_slots = rows * physical_used;
    let routed_mask = if n_used == u32::BITS as usize {
        u32::MAX
    } else {
        (1u32 << n_used) - 1
    };
    let shared_mask = shared.map_or(0, |_| 1u32 << n_used);
    let all_active_mask = routed_mask | shared_mask;
    let r = |id: TensorId| resolve(scratch, bindings, id);
    let shared_wgate = shared.map(|s| r(s.wgate)).transpose()?;
    let shared_wup = shared.map(|s| r(s.wup)).transpose()?;
    let shared_wdown = shared.map(|s| r(s.wdown)).transpose()?;

    // Dummy bias buffer when exp_probs_b is absent (the shader ignores it when has_bias==0).
    let bias_buf: &dyn Buffer = if let Some(epb) = exp_probs_b {
        r(*epb)?
    } else {
        &*be_.alloc_uninit(4, BufferUsage::Weights)?
    };
    let has_bias = exp_probs_b.is_some();
    // Hash routing changes only expert selection; the paged executor consumes the resulting IDs
    // through the same residency and arena path as router top-k. Reuse the resident moe_topk hash
    // mode so V4's early hash layers remain pageable without a second routing implementation.
    let hash_dummy = be_.alloc_uninit(4, BufferUsage::Activations)?;
    let hash_buf: &dyn Buffer = match expert_ids {
        Some(id) => r(*id)?,
        None => hash_dummy.as_ref(),
    };
    let hash = expert_ids.is_some();
    if swiglu_clamp.is_some() && (matches!(act, Activation::Sigmoid) || *weight_before) {
        return Err(be(
            "vulkan adapter: paged MoeFfn swiglu_clamp needs a gated SiLU/GELU with \
             output-side weighting",
        ));
    }

    // ── Router GEMV + top-k, recorded into the AMBIENT segment (no dedicated submit). `ids` is
    // `Staging` (ReBAR device-local host-visible): the ONE remaining readback — the small-m
    // split path below — is then a mapped memcpy after `sync_stream`, no one-shot copy submit
    // and no fresh readback allocation per layer.
    let logits = pooled(pool, be_, "moe_paged_logits", rows * n_expert * 4)?;
    let ids_key = pooled_usage(
        pool,
        be_,
        "moe_paged_ids",
        n_slots * 4,
        BufferUsage::Staging,
    )?;
    let wts = pooled(pool, be_, "moe_paged_wts", n_slots * 4)?;
    {
        let rc = rec.as_ref().expect("segment always Some between ops");
        let rxb = r(*router_x)?;
        let rw = r(*router)?;
        let rdt = graph.desc(*router).dtype;
        if native_dense_supported(rdt) {
            rc.linear_native(rdt, rw, 0, rxb, pool[&logits].as_ref(), rows, ne, n_expert);
        } else if matches!(rdt, infr_core::DType::F32) {
            rc.linear_f32(rw, rxb, pool[&logits].as_ref(), rows, ne, n_expert);
        } else {
            rc.linear(rw, rxb, pool[&logits].as_ref(), rows, ne, n_expert);
        }
        rc.moe_topk(
            pool[&logits].as_ref(),
            pool[&ids_key].as_ref(),
            pool[&wts].as_ref(),
            bias_buf,
            rows,
            n_expert,
            n_used,
            *scale,
            gating_u32,
            *norm_w,
            has_bias,
            *n_expert_groups,
            *n_expert_groups_used,
            hash_buf,
            hash,
        );
        if let Some(shared) = shared {
            // One scalar raw gate logit per row. It is independent of top-k and remains a tiny
            // dense F32 projection; only the three large shared matrices join the routed slots.
            rc.linear_f32(r(shared.gate_inp)?, r(*x)?, r(shared.gate)?, rows, ne, 1);
        }
    }

    let gate_buf = r(*gate_exps)?;
    let up_buf = r(*up_exps)?;
    let down_buf = r(*down_exps)?;
    let (gate_id, up_id, down_id) = (
        buffer_identity(gate_buf),
        buffer_identity(up_buf),
        buffer_identity(down_buf),
    );

    // ── Residency: pick the staging strategy (see the fn doc), ending with every needed expert
    // resident-at-execution and a frozen per-role LUT window in the session tape.
    let touch_all = rows * n_used >= 3 * n_expert;
    let layer_stream = touch_all && be_.cfg().paging.moe_layer_stream;
    // Layer-LOCAL ids to stage; empty = residency already guaranteed (all-resident inline path).
    let mut stage_ids: Vec<u32> = Vec::new();
    let mut stream_synced_for_cpu_push = false;
    if layer_stream {
        // Whole-layer staging below bypasses expert-level LRU bookkeeping entirely.
    } else if touch_all {
        stage_ids = (0..n_expert as u32).collect();
    } else {
        // A preceding prefill may have reinterpreted the arena as contiguous layer storage.
        // The prior execute drained before returning, so it is safe to invalidate its mappings.
        {
            let mut guard = be_.moe_pager().lock().unwrap();
            let changed = guard
                .as_mut()
                .expect("paged execution requires a session")
                .enter_decode();
            if changed && infr_core::pager_profile::active() {
                eprintln!(
                    "[moe-prefill] arena switched to decode-LRU: rows={rows} n_used={n_used} n_expert={n_expert}"
                );
            }
        }
        let inline_ok = {
            let guard = be_.moe_pager().lock().unwrap();
            let sess = guard
                .as_ref()
                .expect("execute_static only reaches here when be_.moe_paged() is true");
            sess.all_resident(gate_id, n_expert)
                // Fused gate_up: `up_exps` is the SAME handle as `gate_exps` (never separately
                // read, per `Op::MoeFfn`'s doc) and the bank was registered once under
                // `Role::Gate` — there is no Up source to check or touch.
                && (*fused_gate_up || sess.all_resident(up_id, n_expert))
                && sess.all_resident(down_id, n_expert)
        };
        if inline_ok {
            let mut guard = be_.moe_pager().lock().unwrap();
            let sess = guard.as_mut().expect("checked above");
            sess.touch_all_hits(gate_id, n_expert)?;
            if !*fused_gate_up {
                sess.touch_all_hits(up_id, n_expert)?;
            }
            sess.touch_all_hits(down_id, n_expert)?;
        } else {
            // Split: the router's routing is host-unknown and some routed expert may be absent —
            // block once, read the ids off the mapped Staging buffer, stage exactly the routed
            // set into the fresh ambient segment (which then stays open for the layers after).
            sync_stream(be_, rec, ps)?;
            stream_synced_for_cpu_push = true;
            let mut id_bytes = vec![0u8; n_slots * 4];
            be_.download(pool[&ids_key].as_ref(), &mut id_bytes)
                .map_err(|e| be(e.to_string()))?;
            stage_ids = bytemuck::cast_slice(&id_bytes).to_vec();
        }
    }
    // CPU writes are not commands in the ambient recorder. The non-layer compatibility path must
    // drain earlier arena readers before overwriting LRU slots. Normal Prefill uses fenced dynamic
    // lanes; Decode already synced above in order to read router ids.
    if !layer_stream && !stage_ids.is_empty() && !stream_synced_for_cpu_push {
        sync_stream(be_, rec, ps)?;
    }
    let mut active_mask = all_active_mask;
    let mut shared_batch_preopened = false;
    // Decode-only hit-first trial: when some complete Gate/Up/Down triplets are already resident,
    // launch those slots before blocking on the remaining host promotions. The original slot
    // order is retained through a kernel mask, so router weights and the final accumulation stay
    // byte-for-byte in their ordinary layout. Keep the pager epoch open across both halves: miss
    // insertion may then use any cold slot except the hit weights still being read by the GPU.
    if !layer_stream
        && rows == 1
        && !*fused_gate_up
        && !*weight_before
        && stage_ids.len() == n_used
        && n_used <= u32::BITS as usize
    {
        let (hit_mask, bounded) = {
            let guard = be_.moe_pager().lock().unwrap();
            let sess = guard.as_ref().expect("paged execution requires a session");
            (
                sess.routed_roles_resident_mask(&[gate_id, up_id, down_id], stage_ids.as_slice())?,
                sess.role_uses_bounded_host_tier(gate_id)?,
            )
        };
        // Without a shared expert, an empty hit set still has no useful first-stage work. With
        // one, shared-only is useful work and overlaps the all-miss host promotion as requested.
        if bounded && hit_mask != routed_mask && (hit_mask != 0 || shared.is_some()) {
            let mut hit_ids = Vec::with_capacity(n_used);
            let mut miss_ids = Vec::with_capacity(n_used);
            for (slot, &expert) in stage_ids.iter().enumerate() {
                if hit_mask & (1u32 << slot) != 0 {
                    hit_ids.push(expert);
                } else {
                    miss_ids.push(expert);
                }
            }
            let shared = {
                let mut guard = be_.moe_pager().lock().unwrap();
                let sess = guard.as_mut().expect("paged execution requires a session");
                let shared = sess.begin_shared_batch(&[gate_id, up_id, down_id])?;
                if shared && !hit_ids.is_empty() {
                    sess.push_roles_cpu(
                        &[
                            (gate_id, hit_ids.as_slice()),
                            (up_id, hit_ids.as_slice()),
                            (down_id, hit_ids.as_slice()),
                        ],
                        false,
                    )?;
                }
                shared
            };
            if shared {
                let gate_hit_w =
                    stage_and_window(be_, rec, ps, gate_id, &[], n_expert, false, true)?;
                let up_hit_w = stage_and_window(be_, rec, ps, up_id, &[], n_expert, false, true)?;
                let down_hit_w =
                    stage_and_window(be_, rec, ps, down_id, &[], n_expert, false, true)?;
                let gbuf = pooled(pool, be_, "moe_paged_g", physical_slots * gu_width * 4)?;
                let ubuf = pooled(pool, be_, "moe_paged_u", physical_slots * nff * 4)?;
                let abuf = pooled(pool, be_, "moe_paged_a", physical_slots * nff * 4)?;
                let ybuf = pooled(pool, be_, "moe_paged_y", physical_slots * ne * 4)?;
                let rec2 = rec.as_ref().expect("segment always Some between ops");
                rec2.zero(pool[&gbuf].as_ref(), physical_slots * gu_width);
                rec2.zero(pool[&ubuf].as_ref(), physical_slots * nff);
                rec2.zero(pool[&ybuf].as_ref(), physical_slots * ne);
                rec2.arena_stream_barrier();
                let xb = r(*x)?;
                let first_mask = hit_mask | shared_mask;
                {
                    let guard = be_.moe_pager().lock().unwrap();
                    let sess = guard.as_ref().expect("checked above");
                    linear_paged_maybe_shared(
                        rec2,
                        gdt,
                        sess.arena_addr(gate_id)?,
                        sess.slot_bytes(gate_id)? as u32,
                        sess.tape(),
                        pool[&ids_key].as_ref(),
                        n_used,
                        gate_hit_w as usize,
                        shared_wgate,
                        xb,
                        false,
                        pool[&gbuf].as_ref(),
                        ne,
                        gu_width,
                        rows,
                        first_mask,
                    );
                    linear_paged_maybe_shared(
                        rec2,
                        udt,
                        sess.arena_addr(up_id)?,
                        sess.slot_bytes(up_id)? as u32,
                        sess.tape(),
                        pool[&ids_key].as_ref(),
                        n_used,
                        up_hit_w as usize,
                        shared_wup,
                        xb,
                        false,
                        pool[&ubuf].as_ref(),
                        ne,
                        nff,
                        rows,
                        first_mask,
                    );
                }
                let n_act = physical_slots * nff;
                match act {
                    Activation::Silu => rec2.silu_mul(
                        pool[&gbuf].as_ref(),
                        pool[&ubuf].as_ref(),
                        pool[&abuf].as_ref(),
                        n_act,
                        *swiglu_clamp,
                    ),
                    Activation::Sigmoid => rec2.mul_sigmoid(
                        pool[&gbuf].as_ref(),
                        pool[&ubuf].as_ref(),
                        pool[&abuf].as_ref(),
                        n_act,
                        0,
                        0,
                        0,
                    ),
                    Activation::Gelu => rec2.gelu_mul_off(
                        pool[&gbuf].as_ref(),
                        pool[&ubuf].as_ref(),
                        0,
                        0,
                        nff,
                        0,
                        nff,
                        0,
                        pool[&abuf].as_ref(),
                        n_act,
                    ),
                }
                {
                    let guard = be_.moe_pager().lock().unwrap();
                    let sess = guard.as_ref().expect("checked above");
                    linear_paged_maybe_shared(
                        rec2,
                        ddt,
                        sess.arena_addr(down_id)?,
                        sess.slot_bytes(down_id)? as u32,
                        sess.tape(),
                        pool[&ids_key].as_ref(),
                        n_used,
                        down_hit_w as usize,
                        shared_wdown,
                        pool[&abuf].as_ref(),
                        true,
                        pool[&ybuf].as_ref(),
                        nff,
                        ne,
                        rows,
                        first_mask,
                    );
                }
                submit_prefill_compute(rec, ps)?;
                let fresh = be_.recorder()?;
                fresh.seed_barrier();
                fresh.arena_stream_barrier();
                *rec = Some(fresh);
                stage_ids = miss_ids;
                active_mask = routed_mask ^ hit_mask;
                shared_batch_preopened = true;
            }
        }
    }
    // WAR: order every arena read recorded so far (this layer's prior roles, or an earlier
    // layer's dispatch still in the ambient segment — a raw pointer deref, invisible to the buffer
    // hazard tracker) before this op's direct copies below, which may overwrite a slot that read
    // still targets. One call covers the whole-layer copy or all three roles' expert copies;
    // harmless when `stage_ids` is empty.
    rec.as_ref()
        .expect("segment always Some between ops")
        .arena_stream_barrier();
    let (shared_batch, bounded_host_batch) = if shared_batch_preopened {
        (true, true)
    } else if !layer_stream && !stage_ids.is_empty() && !*fused_gate_up {
        let mut guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_mut().expect("paged execution requires a session");
        let shared = sess.begin_shared_batch(&[gate_id, up_id, down_id])?;
        let bounded = shared && sess.role_uses_bounded_host_tier(gate_id)?;
        (shared, bounded)
    } else {
        (false, false)
    };
    // On one shared size pool, resolving all three roles before moving bytes gives the host tier
    // enough independent work to saturate SSD/RAM/ReBAR. Keep it restricted to the bounded tier:
    // the complete Host Store has no parallel promotion work here, so batching Down would only
    // discard its useful overlap with Gate/Up compute on Qwen/Ling-style models.
    let roles_batched = bounded_host_batch;
    if roles_batched {
        let push = {
            let mut guard = be_.moe_pager().lock().unwrap();
            guard
                .as_mut()
                .expect("paged execution requires a session")
                .push_roles_cpu(
                    &[
                        (gate_id, stage_ids.as_slice()),
                        (up_id, stage_ids.as_slice()),
                        (down_id, stage_ids.as_slice()),
                    ],
                    touch_all,
                )?
        };
        push.record(rec.as_ref().expect("segment always Some between ops"))?;
    }
    let role_stage_ids = if roles_batched {
        &[][..]
    } else {
        stage_ids.as_slice()
    };
    let gate_w = if layer_stream {
        stage_layer_and_window(be_, rec, ps, gate_id, n_expert)?
    } else {
        stage_and_window(
            be_,
            rec,
            ps,
            gate_id,
            role_stage_ids,
            n_expert,
            touch_all,
            shared_batch,
        )?
    };
    let up_w = if *fused_gate_up {
        gate_w // never dispatched (no Up GEMV on the fused shape) — placeholder
    } else if layer_stream {
        stage_layer_and_window(be_, rec, ps, up_id, n_expert)?
    } else {
        stage_and_window(
            be_,
            rec,
            ps,
            up_id,
            role_stage_ids,
            n_expert,
            touch_all,
            shared_batch,
        )?
    };
    let down_has_miss = if roles_batched {
        false
    } else if !layer_stream && !stage_ids.is_empty() {
        let guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_ref().expect("paged execution requires a session");
        !sess.routed_all_resident(down_id, &stage_ids)?
    } else {
        false
    };
    let overlap_down = !layer_stream
        && !stage_ids.is_empty()
        && down_has_miss
        && rows <= moe_small_m_threshold(be_)
        && be_.prefers_decode_down_overlap();
    let down_w = if overlap_down {
        None
    } else if layer_stream {
        Some(stage_layer_and_window(be_, rec, ps, down_id, n_expert)?)
    } else {
        Some(stage_and_window(
            be_,
            rec,
            ps,
            down_id,
            role_stage_ids,
            n_expert,
            touch_all,
            shared_batch,
        )?)
    };
    // RAW: make every direct HOST write to the mapped arena visible before the first dispatch
    // below reads it by pointer. Covers both batched and small-m paged expert kernels.
    rec.as_ref()
        .expect("segment always Some between ops")
        .arena_stream_barrier();

    // ── BATCHED arm (rows > threshold, Scout's chunked prefill): the same GPU-resident
    // bucket count/scan/scatter → dp4a mmq expert-GEMM pipeline as the resident batched arm
    // (`lower_op`'s `Op::MoeFfn`), with the GEMMs reading the pager arena through the frozen
    // tape window (`matmul_mmq_experts_paged`) instead of a resident bank — recorded INLINE
    // into the ambient segment. Every ROUTED expert is resident-at-execution (the staging above
    // covers all `n_expert` on the touch-all path, or exactly the readback's routed set — up to
    // `n_expert` distinct simultaneously, which the pager budget floor in `seam::mod` guarantees
    // fits); buckets with count 0 never read the window. Gated on the exact dtype set that has
    // `_xpg` kernel builds (`infr_core::tensor::MOE_MMQ_PAGED_DTYPES` — the FULL
    // `MOE_MMQ_DTYPES` set, mirror checked by `moe_mmq_drift_test`) + activation + dp4a
    // support; anything else stays on the id-GEMV arm below, which is shape-general.
    {
        let paged_mmq_ok = infr_core::tensor::moe_paged_mmq_ok;
        let act_ok = if *fused_gate_up {
            // Fused callers ship GeGLU (gemma-4 MoE / DiffusionGemma) or SwiGLU — same set the
            // resident fused arm accepts.
            matches!(act, Activation::Silu | Activation::Gelu)
        } else {
            matches!(act, Activation::Silu)
        };
        if rows > moe_small_m_threshold(be_)
            && be_.caps().i8_dot
            && act_ok
            && paged_mmq_ok(gdt)
            && paged_mmq_ok(udt)
            && paged_mmq_ok(ddt)
        {
            let n_pairs = n_slots;
            // The GEMM As stage reads up to 63 rows past a segment end — pad the packed row
            // dimension so the LAST expert's overread stays in-bounds (the resident arm's npad).
            let npad = n_pairs.div_ceil(64) * 64 + 64;
            let counts = pooled(pool, be_, "moe_pgb_counts", n_expert * 4)?;
            let offsets = pooled(pool, be_, "moe_pgb_offsets", n_expert * 4)?;
            let fill = pooled(pool, be_, "moe_pgb_fill", n_expert * 4)?;
            let bucket_rows = pooled(pool, be_, "moe_pgb_brows", n_pairs * 4)?;
            let bucket_wts = pooled(pool, be_, "moe_pgb_bwts", n_pairs * 4)?;
            let inv_pos = pooled(pool, be_, "moe_pgb_ipos", n_pairs * 4)?;
            let qa = pooled(pool, be_, "moe_pgb_qa", npad * ne)?;
            let qda = pooled(pool, be_, "moe_pgb_qda", npad * (ne / 32) * 2)?;
            let qsa = pooled(pool, be_, "moe_pgb_qsa", npad * (ne / 32) * 2)?;
            // Fused: `ge` holds the single wide [n_pairs, 2*nff] gate|up GEMM output (the
            // resident batched fused arm's shape); `ue` is unused/unallocated.
            let ge = pooled(pool, be_, "moe_pgb_ge", npad * gu_width * 4)?;
            let ue = if *fused_gate_up {
                None
            } else {
                Some(pooled(pool, be_, "moe_pgb_ue", npad * nff * 4)?)
            };
            let ae = pooled(pool, be_, "moe_pgb_ae", npad * nff * 4)?;
            let dqa = pooled(pool, be_, "moe_pgb_dqa", npad * nff)?;
            let dda = pooled(pool, be_, "moe_pgb_dda", npad * (nff / 32) * 2)?;
            let dsa = pooled(pool, be_, "moe_pgb_dsa", npad * (nff / 32) * 2)?;
            let ye = pooled(pool, be_, "moe_pgb_ye", npad * ne * 4)?;

            let rec2 = rec.as_ref().expect("segment always Some between ops");
            let xb = r(*x)?;
            rec2.zero(pool[&counts].as_ref(), n_expert);
            // Clear the scatter's `fill` counters here (independent of `counts`) so the parallel
            // clear overlaps the count/scan rather than riding the 1-lane serial scan.
            rec2.zero(pool[&fill].as_ref(), n_expert);
            rec2.moe_bucket_count(pool[&ids_key].as_ref(), pool[&counts].as_ref(), n_pairs);
            rec2.moe_bucket_scan(pool[&counts].as_ref(), pool[&offsets].as_ref(), n_expert);
            let dsb: Option<&dyn Buffer> = match down_scale {
                Some(ds) => Some(r(*ds)?),
                None => None,
            };
            rec2.moe_bucket_scatter(
                pool[&ids_key].as_ref(),
                pool[&wts].as_ref(),
                pool[&offsets].as_ref(),
                pool[&fill].as_ref(),
                pool[&bucket_rows].as_ref(),
                pool[&bucket_wts].as_ref(),
                pool[&inv_pos].as_ref(),
                dsb,
                n_pairs,
                n_used,
            );
            rec2.quant_q8_gather(
                xb,
                pool[&bucket_rows].as_ref(),
                pool[&qa].as_ref(),
                pool[&qda].as_ref(),
                pool[&qsa].as_ref(),
                n_pairs,
                ne,
            );
            {
                let guard = be_.moe_pager().lock().unwrap();
                let sess = guard.as_ref().expect("checked above");
                // Same `MOE_MMQ_SACT_DTYPES` split as the resident arm (gate/up can each
                // independently be any `paged_mmq_ok` member).
                let gate_needs_sact = infr_core::tensor::moe_mmq_needs_sact(gdt);
                rec2.matmul_mmq_experts_paged(
                    gdt,
                    "expert_gateup",
                    pool[&qa].as_ref(),
                    pool[&qda].as_ref(),
                    gate_needs_sact.then(|| pool[&qsa].as_ref()),
                    sess.arena_addr(gate_id)?,
                    sess.slot_bytes(gate_id)? as u32,
                    sess.tape(),
                    gate_w as usize,
                    pool[&counts].as_ref(),
                    pool[&offsets].as_ref(),
                    pool[&ge].as_ref(),
                    rows,
                    ne,
                    gu_width,
                    n_expert,
                    n_used,
                );
                if let Some(ue) = &ue {
                    // The up GEMM reads the same quantized activations and writes its own buffer —
                    // disjoint from the gate GEMM, no barrier needed (resident arm's pattern).
                    rec2.suppress_sync(true);
                    let up_needs_sact = infr_core::tensor::moe_mmq_needs_sact(udt);
                    rec2.matmul_mmq_experts_paged(
                        udt,
                        "expert_gateup",
                        pool[&qa].as_ref(),
                        pool[&qda].as_ref(),
                        up_needs_sact.then(|| pool[&qsa].as_ref()),
                        sess.arena_addr(up_id)?,
                        sess.slot_bytes(up_id)? as u32,
                        sess.tape(),
                        up_w as usize,
                        pool[&counts].as_ref(),
                        pool[&offsets].as_ref(),
                        pool[ue].as_ref(),
                        rows,
                        ne,
                        nff,
                        n_expert,
                        n_used,
                    );
                    rec2.suppress_sync(false);
                }
            }
            if *weight_before {
                rec2.moe_weight_scale(
                    pool[&ge].as_ref(),
                    pool[&bucket_wts].as_ref(),
                    n_pairs,
                    gu_width,
                );
                if let Some(ue) = &ue {
                    rec2.moe_weight_scale(
                        pool[ue].as_ref(),
                        pool[&bucket_wts].as_ref(),
                        n_pairs,
                        nff,
                    );
                }
            }
            match &ue {
                // Split: gate and up are separate [n_pairs, nff] buffers.
                Some(ue) => rec2.silu_mul(
                    pool[&ge].as_ref(),
                    pool[ue].as_ref(),
                    pool[&ae].as_ref(),
                    n_pairs * nff,
                    *swiglu_clamp,
                ),
                // Fused: `ge` already holds [n_pairs, 2*nff] (gate half first, up half second per
                // row) from the single wide GEMM above — the resident batched fused arm's shape.
                None => match act {
                    Activation::Silu => rec2.silu_mul_fused(
                        pool[&ge].as_ref(),
                        pool[&ae].as_ref(),
                        n_pairs,
                        nff,
                        *swiglu_clamp,
                    ),
                    Activation::Gelu => rec2.gelu_mul_fused(
                        pool[&ge].as_ref(),
                        pool[&ae].as_ref(),
                        n_pairs,
                        nff,
                        *swiglu_clamp,
                    ),
                    Activation::Sigmoid => unreachable!("act_ok gate above excludes Sigmoid"),
                },
            }
            rec2.quant_q8(
                pool[&ae].as_ref(),
                pool[&dqa].as_ref(),
                pool[&dda].as_ref(),
                pool[&dsa].as_ref(),
                n_pairs,
                nff,
            );
            {
                let guard = be_.moe_pager().lock().unwrap();
                let sess = guard.as_ref().expect("checked above");
                let down_needs_sact = infr_core::tensor::moe_mmq_needs_sact(ddt);
                rec2.matmul_mmq_experts_paged(
                    ddt,
                    "expert_down",
                    pool[&dqa].as_ref(),
                    pool[&dda].as_ref(),
                    down_needs_sact.then(|| pool[&dsa].as_ref()),
                    sess.arena_addr(down_id)?,
                    sess.slot_bytes(down_id)? as u32,
                    sess.tape(),
                    down_w.expect("batched path stages Down before dispatch") as usize,
                    pool[&counts].as_ref(),
                    pool[&offsets].as_ref(),
                    pool[&ye].as_ref(),
                    rows,
                    nff,
                    ne,
                    n_expert,
                    n_used,
                );
            }
            rec2.moe_scatter_reduce(
                pool[&ye].as_ref(),
                pool[&bucket_wts].as_ref(),
                pool[&inv_pos].as_ref(),
                r(*dst)?,
                rows,
                ne,
                n_used,
                *weight_before,
            );
            if layer_stream {
                prefetch_next_moe_layer(be_, rec, ps, gate_id)?;
            }
            return Ok(()); // recorded inline — the ambient segment stays open
        }
    }

    // ── Small-m id-GEMV arm: the paged expert GEMVs (arena + frozen tape window, LOCAL ids)
    // through the rest of this MoeFfn's math — activation, down-projection, weighted accumulate —
    // exactly mirroring the non-paged small-m arm (including its fused-gate_up shape: one
    // double-width GEMV into a gate|up buffer, split by the fused activation kernel). Recorded
    // inline into the ambient segment like the batched arm above.
    let gbuf = pooled(pool, be_, "moe_paged_g", physical_slots * gu_width * 4)?;
    let ubuf = if *fused_gate_up {
        None
    } else {
        Some(pooled(pool, be_, "moe_paged_u", physical_slots * nff * 4)?)
    };
    let abuf = pooled(pool, be_, "moe_paged_a", physical_slots * nff * 4)?;
    let ybuf = pooled(pool, be_, "moe_paged_y", physical_slots * ne * 4)?;
    let n_act = physical_slots * nff;
    let rec2 = rec.as_ref().expect("segment always Some between ops");
    let xb = r(*x)?;
    {
        let guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_ref().expect("checked above");
        linear_paged_maybe_shared(
            rec2,
            gdt,
            sess.arena_addr(gate_id)?,
            sess.slot_bytes(gate_id)? as u32,
            sess.tape(),
            pool[&ids_key].as_ref(),
            n_used,
            gate_w as usize,
            shared_wgate,
            xb,
            false,
            pool[&gbuf].as_ref(),
            ne,
            gu_width,
            rows,
            active_mask,
        );
        if let Some(ubuf) = &ubuf {
            linear_paged_maybe_shared(
                rec2,
                udt,
                sess.arena_addr(up_id)?,
                sess.slot_bytes(up_id)? as u32,
                sess.tape(),
                pool[&ids_key].as_ref(),
                n_used,
                up_w as usize,
                shared_wup,
                xb,
                false,
                pool[ubuf].as_ref(),
                ne,
                nff,
                rows,
                active_mask,
            );
        }
    }
    if *weight_before {
        rec2.moe_weight_scale(pool[&gbuf].as_ref(), pool[&wts].as_ref(), n_slots, gu_width);
        if let Some(ubuf) = &ubuf {
            rec2.moe_weight_scale(pool[ubuf].as_ref(), pool[&wts].as_ref(), n_slots, nff);
        }
    }
    match &ubuf {
        Some(ubuf) => match act {
            Activation::Silu => rec2.silu_mul(
                pool[&gbuf].as_ref(),
                pool[ubuf].as_ref(),
                pool[&abuf].as_ref(),
                n_act,
                *swiglu_clamp,
            ),
            Activation::Sigmoid => rec2.mul_sigmoid(
                pool[&gbuf].as_ref(),
                pool[ubuf].as_ref(),
                pool[&abuf].as_ref(),
                n_act,
                0,
                0,
                0,
            ),
            Activation::Gelu => rec2.gelu_mul_off(
                pool[&gbuf].as_ref(),
                pool[ubuf].as_ref(),
                0,
                0,
                nff,
                0,
                nff,
                0,
                pool[&abuf].as_ref(),
                n_act,
            ),
        },
        // Fused: `gbuf` is [n_slots, 2*nff] gate|up rows; the fused activation kernels split it
        // (gate half first, up half second per row — `Op::GatedActFused`'s convention), exactly
        // like the resident small-m fused arm.
        None => match act {
            Activation::Silu => rec2.silu_mul_fused(
                pool[&gbuf].as_ref(),
                pool[&abuf].as_ref(),
                n_slots,
                nff,
                *swiglu_clamp,
            ),
            Activation::Gelu => rec2.gelu_mul_fused(
                pool[&gbuf].as_ref(),
                pool[&abuf].as_ref(),
                n_slots,
                nff,
                *swiglu_clamp,
            ),
            Activation::Sigmoid => {
                return Err(be(
                    "vulkan adapter: fused_gate_up paged MoeFfn Sigmoid unsupported",
                ))
            }
        },
    }
    let down_w = if let Some(window) = down_w {
        window
    } else {
        // Gate/Up and activation do not read the Down role pool. Launch them first, then direct-
        // push Down misses while that independent GPU work is live. The fresh recorder is queue-
        // ordered after the first segment and its seed/barrier publishes the Host writes before
        // the Down GEMV. The next layer's ordinary sync drains this pending segment and tape.
        let probe = submit_prefill_compute(rec, ps)?;
        let profile = infr_core::pager_profile::active();
        let compute_live_at_start = profile && prefetch_compute_live(ps, probe)?;
        let window = stage_and_window(
            be_,
            rec,
            ps,
            down_id,
            &stage_ids,
            n_expert,
            touch_all,
            shared_batch,
        )?;
        if profile {
            infr_core::pager_profile::record_prefetch_window(
                compute_live_at_start,
                prefetch_compute_live(ps, probe)?,
            );
        }
        let fresh = be_.recorder()?;
        fresh.seed_barrier();
        fresh.arena_stream_barrier();
        *rec = Some(fresh);
        window
    };
    let rec2 = rec.as_ref().expect("segment always Some between ops");
    {
        let guard = be_.moe_pager().lock().unwrap();
        let sess = guard.as_ref().expect("checked above");
        linear_paged_maybe_shared(
            rec2,
            ddt,
            sess.arena_addr(down_id)?,
            sess.slot_bytes(down_id)? as u32,
            sess.tape(),
            pool[&ids_key].as_ref(),
            n_used,
            down_w as usize,
            shared_wdown,
            pool[&abuf].as_ref(),
            true,
            pool[&ybuf].as_ref(),
            nff,
            ne,
            rows,
            active_mask,
        );
    }
    let dstb = match shared {
        Some(shared) => r(shared.dst)?,
        None => r(*dst)?,
    };
    if let Some(shared) = shared {
        rec2.moe_accumulate_shared(
            pool[&ybuf].as_ref(),
            pool[&wts].as_ref(),
            r(shared.gate)?,
            dstb,
            ne,
            n_used,
            rows,
        );
    } else {
        rec2.zero(dstb, rows * ne);
        match down_scale {
            Some(ds) => rec2.moe_accumulate_scaled(
                pool[&ybuf].as_ref(),
                pool[&wts].as_ref(),
                // `down_scale[expert_id]` indexes by the LOCAL id (the layer's own [n_expert] scale
                // array) — the same local-ids buffer the windowed GEMVs read.
                pool[&ids_key].as_ref(),
                r(*ds)?,
                dstb,
                ne,
                n_used,
                rows,
            ),
            None => rec2.moe_accumulate(
                pool[&ybuf].as_ref(),
                pool[&wts].as_ref(),
                dstb,
                ne,
                n_used,
                rows,
                *weight_before,
            ),
        }
    }
    Ok(()) // recorded inline — the ambient segment stays open
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::graph::Graph;
    use infr_core::tensor::TensorDesc;
    use infr_core::DType;

    /// Read a `#define <name> <integer>` back out of GLSL source. Panics when the define is gone —
    /// a shader that no longer states its bound is exactly the drift this guards.
    fn glsl_define_u32(src: &str, name: &str) -> u32 {
        for line in src.lines() {
            let Some(rest) = line.trim_start().strip_prefix("#define ") else {
                continue;
            };
            let mut it = rest.split_whitespace();
            if it.next() == Some(name) {
                let v = it
                    .next()
                    .unwrap_or_else(|| panic!("`#define {name}` has no value"));
                return v
                    .parse()
                    .unwrap_or_else(|e| panic!("`#define {name} {v}` is not a u32: {e}"));
            }
        }
        panic!("`#define {name}` not found in the shader source");
    }

    /// `MLA_MAX_KEY_LEN` / `MLA_MAX_KV_LORA_RANK` are the host's copy of `mla.comp`'s shared-array
    /// sizes, and nothing in the toolchain ties the two together — the shader could be re-sized
    /// and the `Op::Mla` guard would go on enforcing the old number (rejecting work the kernel now
    /// fits, or, the dangerous direction, admitting dims that overrun `sq`/`sv`). Read the numbers
    /// back out of the shader text and compare, and check the arrays are still declared THROUGH the
    /// defines rather than with bare literals that this test could not see.
    #[test]
    fn mla_shader_bounds_match_host() {
        let src = include_str!("../shaders/mla.comp");
        assert_eq!(
            glsl_define_u32(src, "MLA_MAX_KEY_LEN"),
            MLA_MAX_KEY_LEN,
            "mla.comp's sq[] capacity drifted from the host guard"
        );
        assert_eq!(
            glsl_define_u32(src, "MLA_MAX_KV_LORA_RANK"),
            MLA_MAX_KV_LORA_RANK,
            "mla.comp's sv[] capacity drifted from the host guard"
        );
        assert!(
            src.contains("shared float sq[MLA_MAX_KEY_LEN];"),
            "mla.comp's sq[] is no longer sized by MLA_MAX_KEY_LEN"
        );
        assert!(
            src.contains("shared float sv[MLA_MAX_KV_LORA_RANK];"),
            "mla.comp's sv[] is no longer sized by MLA_MAX_KV_LORA_RANK"
        );
    }

    /// Same drift guard, one axis over: the `-DMLA_SG` builds size their cross-subgroup scratch as
    /// `128 / MLA_SG_WIDTH` and sum exactly that many slots, while the HOST is what actually pins
    /// the pipeline's `requiredSubgroupSize` (`gemm::MLA_SG_WIDTH`, passed to `kernel_sg` by
    /// `Recorder::mla`). Let the two drift apart and every score comes out summed over the wrong
    /// number of subgroups — a wrong number, not a crash, and every shape check still passes.
    #[test]
    fn mla_sg_width_matches_host() {
        let src = include_str!("../shaders/mla.comp");
        assert_eq!(
            glsl_define_u32(src, "MLA_SG_WIDTH"),
            crate::gemm::MLA_SG_WIDTH,
            "mla.comp's MLA_SG_WIDTH drifted from the width Recorder::mla pins"
        );
        assert!(
            src.contains("shared float sgred[2 * MLA_SG_KB * MLA_NSG];")
                && src.contains("#define MLA_NSG (128 / MLA_SG_WIDTH)"),
            "mla.comp's sgred[] is no longer sized through MLA_SG_WIDTH"
        );
    }

    /// The `-DMLA_SG` tier is capability-gated at the point of use, not just by the backend's
    /// device-level wave32 refusal: a device whose subgroup range excludes the pinned width must
    /// get the scalar two-pass builds, whose eight-way selector must in turn never hand a `sg`
    /// name a scalar build's SPIR-V (kernels are cached by NAME, so a mismatch would poison the
    /// cache entry for the whole process).
    #[test]
    fn mla_sg_tier_degrades_without_the_pinnable_width() {
        let mut caps = infr_core::backend::Capabilities::default();
        let vk = infr_core::config::VulkanCfg::default();
        assert!(vk.mla_sg, "the subgroup MLA tier is on by default");

        caps.subgroup_min = 32;
        caps.subgroup_max = 64;
        assert!(
            crate::gemm::mla_sg_tier(&vk, &caps),
            "a device that can pin 32 must take the subgroup tier"
        );

        // Subgroup-size control absent (`(0, 0)`), and a range that simply excludes the width.
        for (lo, hi) in [(0, 0), (64, 64), (8, 16)] {
            caps.subgroup_min = lo;
            caps.subgroup_max = hi;
            assert!(
                !crate::gemm::mla_sg_tier(&vk, &caps),
                "subgroup range [{lo}, {hi}] excludes {} — must degrade to the scalar builds",
                crate::gemm::MLA_SG_WIDTH
            );
        }

        // And the env knob overrides a capable device.
        caps.subgroup_min = 32;
        caps.subgroup_max = 64;
        let mut off = vk.clone();
        off.mla_sg = false;
        assert!(
            !crate::gemm::mla_sg_tier(&off, &caps),
            "INFR_NO_MLA_SG must force the scalar builds"
        );

        // Every (tier, ff, bias) build is distinct, and the `sg` axis only ever moves the name
        // onto an `mla_sg` one.
        let mut names = std::collections::HashSet::new();
        let mut modules = std::collections::HashSet::new();
        for sg in [false, true] {
            for ff in [false, true] {
                for bias in [false, true] {
                    let (name, spv) = crate::gemm::mla_kernel(sg, ff, bias);
                    assert_eq!(
                        name.starts_with("mla_sg"),
                        sg,
                        "{name} does not name the tier it was asked for"
                    );
                    assert!(!spv.is_empty(), "{name} has no SPIR-V");
                    assert!(names.insert(name), "duplicate MLA kernel name {name}");
                    assert!(
                        modules.insert(spv),
                        "{name} shares another build's SPIR-V — the -D variants collapsed"
                    );
                }
            }
        }
    }

    /// B45 guard: `mla.comp` keeps its whole working state in FIXED-size shared arrays, so an
    /// `Op::Mla` whose latent does not fit must be refused BEFORE dispatch — a shared-memory
    /// overrun has no diagnostic (corrupted neighbours or a lost device, not a fault). Both halves
    /// of the condition get a case, and the boundary dims — `key_len` exactly `MLA_MAX_KEY_LEN` —
    /// must still RUN: a guard that rejected everything would satisfy the negative cases alone.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn mla_oversized_dims_are_refused() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        // One token, one head, a 2-wide nope/value part so only the latent dims vary.
        let (rows, n_head, qk_nope, v_head_dim) = (1usize, 1usize, 2usize, 2usize);
        let run = |kv_lora: usize, qk_rope: usize| -> Result<()> {
            let key_len = kv_lora + qk_rope;
            let q_head_dim = qk_nope + qk_rope;
            let mut g = Graph::new();
            let q = g.input(TensorDesc::new(vec![rows, n_head, q_head_dim], DType::F32));
            let k_cache = g.input(TensorDesc::new(vec![rows, key_len], DType::F16));
            let wk_b = g.weight(TensorDesc::new(vec![n_head, kv_lora, qk_nope], DType::F32));
            let wv_b = g.weight(TensorDesc::new(
                vec![n_head, kv_lora, v_head_dim],
                DType::F32,
            ));
            let dst = g.output(TensorDesc::new(vec![rows, n_head, v_head_dim], DType::F32));
            g.push(Op::Mla {
                q,
                k_cache,
                wk_b,
                wv_b,
                dst,
                rows: rows as u32,
                kv_len: rows as u32,
                n_head: n_head as u32,
                q_head_dim: q_head_dim as u32,
                kv_lora_rank: kv_lora as u32,
                qk_nope_dim: qk_nope as u32,
                qk_rope_dim: qk_rope as u32,
                v_head_dim: v_head_dim as u32,
                scale: 1.0,
                mask: AttnMask::Causal,
                pos: 0,
                theta: 10000.0,
                freq_factors: None,
                key_bias: None,
            });
            // Zeroed buffers of the exact shapes — the boundary case really dispatches over them.
            let qb = be_.alloc(rows * n_head * q_head_dim * 4, BufferUsage::Activations)?;
            let kb = be_.alloc(rows * key_len * 2, BufferUsage::Activations)?;
            let wkb = be_.alloc(n_head * kv_lora * qk_nope * 4, BufferUsage::Weights)?;
            let wvb = be_.alloc(n_head * kv_lora * v_head_dim * 4, BufferUsage::Weights)?;
            let db = be_.alloc(rows * n_head * v_head_dim * 4, BufferUsage::Activations)?;
            let plan = be_.compile(&g)?;
            let mut bind = Bindings::new();
            bind.bind(q, qb.as_ref());
            bind.bind(k_cache, kb.as_ref());
            bind.bind(wk_b, wkb.as_ref());
            bind.bind(wv_b, wvb.as_ref());
            bind.bind(dst, db.as_ref());
            be_.execute(plan.as_ref(), &bind)
        };
        // Positive control at the exact boundary: kv_lora_rank fills `sv[]` and key_len fills
        // `sq[]` to the last element.
        let rope = (MLA_MAX_KEY_LEN - MLA_MAX_KV_LORA_RANK) as usize;
        run(MLA_MAX_KV_LORA_RANK as usize, rope).expect("boundary dims must still dispatch");
        // kv_lora_rank past `sv[]` (and, with it, key_len past `sq[]`).
        let e = run(1024, 64).unwrap_err().to_string();
        assert!(
            e.contains("kv_lora_rank 1024") && e.contains("key_len 1088"),
            "oversized kv_lora_rank: unexpected error text: {e}"
        );
        // key_len past `sq[]` while kv_lora_rank alone still fits `sv[]`.
        let e = run(MLA_MAX_KV_LORA_RANK as usize, 128)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("key_len 640"),
            "oversized key_len: unexpected error text: {e}"
        );
    }

    /// The OLD static split-K chunk formula, spelled out inline (the else-branch decode/prefill
    /// policy, before the count cap and before it moved to `infr_core::tier`). Kept as the
    /// byte-identity reference for BOTH the shared `adaptive_chunk` policy and the count cap: a
    /// re-tune of `ATTN_SPLIT` must face this literal.
    fn old_static_chunk(span: usize) -> usize {
        (span / 32).clamp(64, 512)
    }

    /// #2 regression guard: the chunk-COUNT cap must NOT change `chunk` (and thus `n_chunks`, the
    /// recorded dispatch) for any realistic span — it may only kick in above ~524k keys. AND it
    /// must actually bound `n_chunks <= 1024` for the huge spans reachable under INFR_KV_OVERFLOW,
    /// across every static branch's base chunk (the else formula, the ring/cap 512, the batched
    /// 256). attn_combine.comp indexes `shared float wexp[1024]` by chunk, so >1024 is an OOB write.
    #[test]
    fn split_k_chunk_count_cap_is_identity_for_realistic_spans() {
        // Identity up to and including 512*1024 (the exact boundary where the else formula's
        // chunk=512 gives n_chunks=1024). Step keeps the test fast while sweeping every regime
        // (the clamp floors at 64, the /32 ramp, the 512 ceiling), plus every exact boundary.
        let boundary = 512 * 1024;
        let mut spans: Vec<usize> = (1..=boundary).step_by(97).collect();
        spans.extend([
            1,
            2,
            63,
            64,
            2047,
            2048,
            16383,
            16384,
            boundary - 1,
            boundary,
        ]);
        for span in spans {
            let old = old_static_chunk(span);
            // Structural: the shared policy configured with `ATTN_SPLIT` must reproduce the old
            // inline formula exactly, at every span.
            assert_eq!(
                infr_core::tier::adaptive_chunk(span, &ATTN_SPLIT),
                old,
                "ATTN_SPLIT drifted from the inline formula at span {span}"
            );
            let capped = split_k_chunk_count_cap(span, old);
            assert_eq!(
                old, capped,
                "cap changed chunk for realistic span {span}: {old} -> {capped}"
            );
            assert!(
                span.div_ceil(capped) <= 1024,
                "n_chunks OOB at span {span}: {}",
                span.div_ceil(capped)
            );
        }
    }

    #[test]
    fn split_k_chunk_count_cap_bounds_huge_spans() {
        // Past the boundary the cap must engage: n_chunks stays <= 1024 for every static base
        // chunk (else formula, ring/cap 512, batched 256) up to a few million keys.
        for span in (1..=8_000_000).step_by(7919) {
            for base in [old_static_chunk(span), 512usize, 256usize] {
                let capped = split_k_chunk_count_cap(span, base);
                let n_chunks = span.div_ceil(capped);
                assert!(
                    n_chunks <= 1024,
                    "n_chunks {n_chunks} > 1024 at span {span}, base {base}"
                );
            }
        }
    }

    /// #6: the extracted `split_k_plan` must reproduce the copy-pasted narrow-grid/splits formula
    /// exactly (it fed the resident native-quant arm, the F16 split-K arm, and the streamed twin).
    #[test]
    fn split_k_plan_matches_inline_formula() {
        let inline = |m: usize, in_f: usize, out_f: usize| -> usize {
            let narrow_grid = m.div_ceil(64) * (out_f / 128).max(1);
            if out_f.is_multiple_of(128) && in_f >= 1024 && narrow_grid < 128 {
                (256 / narrow_grid).next_power_of_two().clamp(1, 8)
            } else {
                1
            }
        };
        for &m in &[1usize, 16, 17, 64, 128, 256, 512, 1024] {
            for &in_f in &[256usize, 512, 1024, 2048, 4096, 6144, 262144] {
                for &out_f in &[128usize, 256, 512, 1024, 1152, 2048, 2816, 4096, 6144, 8192] {
                    assert_eq!(
                        split_k_plan(m, in_f, out_f),
                        inline(m, in_f, out_f),
                        "split_k_plan mismatch at m={m} in_f={in_f} out_f={out_f}"
                    );
                }
            }
        }
    }

    /// The mrow m-ladder moved to the shared `infr_core::tier` policy (candidate F): the two
    /// configured bands must reproduce the OLD inline row/width expressions exactly, for every
    /// (m, out_f) either could see. A band edit that changes a routing decision fails here.
    #[test]
    fn mrow_bands_match_inline_formula() {
        use infr_core::tier::{linear_tier, LinearTier};
        // The pre-extraction expressions, verbatim.
        let old_mrow = |m: usize, out_f: usize| {
            (2..=4).contains(&m) || ((5..=8).contains(&m) && out_f <= 8192)
        };
        let old_mrow16 = |m: usize, out_f: usize| (9..=16).contains(&m) && out_f <= 8192;
        for m in 0usize..=24 {
            for &out_f in &[
                1usize, 64, 4096, 8191, 8192, 8193, 9216, 24576, 65536, 151_936,
            ] {
                let tier = linear_tier(m, out_f, &MROW_BANDS);
                assert_eq!(
                    tier == LinearTier::MultiRow { band: MROW_BAND },
                    old_mrow(m, out_f),
                    "mrow band m={m} out_f={out_f}"
                );
                assert_eq!(
                    tier == LinearTier::MultiRow { band: MROW16_BAND },
                    old_mrow16(m, out_f),
                    "mrow16 band m={m} out_f={out_f}"
                );
                // m == 1 is the decode GEMV on every width — never a band.
                assert_eq!(m == 1, tier == LinearTier::Decode, "decode m={m}");
            }
        }
    }

    /// The tier knobs' shipping numbers, asserted on the tables and on `Config::default()` — the
    /// two spellings of the same default, which MUST agree (the value now arrives from the config,
    /// the clamp range from the table). A default/ceiling edit has to face this.
    #[test]
    fn tier_knob_defaults_and_clamps() {
        let d = infr_core::config::Config::default();
        // `INFR_MOE_SMALL_M`: measured crossover 8, hard ceiling 64 (past it a prefill chunk
        // becomes one multi-second dispatch and trips the amdgpu watchdog).
        assert_eq!(MOE_SMALL_M.env, "INFR_MOE_SMALL_M");
        assert_eq!(MOE_SMALL_M.default, 8);
        assert_eq!(d.kernels.vulkan.moe_small_m, MOE_SMALL_M.default);
        assert_eq!((MOE_SMALL_M.min, MOE_SMALL_M.max), (0, 64));
        // A config value past the ceiling is clamped at the read site, exactly as the env override
        // always was — `INFR_MOE_SMALL_M=100000` must not reach a dispatch.
        assert_eq!(MOE_SMALL_M.clamped(100_000), 64);
        // `INFR_CANVAS_CHUNK_N`: a divisor, so it floors at 1 rather than capping.
        assert_eq!(CANVAS_CHUNK_N.env, "INFR_CANVAS_CHUNK_N");
        assert_eq!(CANVAS_CHUNK_N.default, 3);
        assert_eq!(d.kernels.vulkan.canvas_chunk_n, CANVAS_CHUNK_N.default);
        assert_eq!(CANVAS_CHUNK_N.min, 1);
        assert_eq!(CANVAS_CHUNK_N.clamped(0), 1);
        // The fusion hatch is a POSITIVE field: fused by default, `INFR_NO_FUSE_ADD` clears it.
        assert!(d.kernels.vulkan.fuse_add);
        // The split-K policy the attention lowering is configured with.
        assert_eq!(ATTN_SPLIT.target_chunks, 32);
        assert_eq!((ATTN_SPLIT.min_chunk, ATTN_SPLIT.max_chunk), (64, 512));
        assert_eq!(ATTN_SPLIT.rounding, infr_core::tier::ChunkRounding::Down);
        assert_eq!(ATTN_MAX_CHUNKS, 1024); // attn_combine.comp's shared wexp[1024]
    }

    /// #3: a strided-gate `Silu` (`gate_stride != 0`) is shape-legal but `silu_mul` reads the gate
    /// contiguously — it must ERROR (recoverable), not silently compute the wrong thing. Executing
    /// the graph must return `Err`, never succeed. (GPU-gated: reaches the guard via `lower_op`.)
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn gated_act_silu_strided_gate_errors() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, nff) = (1usize, 4usize);
        let mut g = Graph::new();
        let gate = g.input(TensorDesc::new(vec![rows, nff], DType::F32));
        let up = g.input(TensorDesc::new(vec![rows, nff], DType::F32));
        let dst = g.output(TensorDesc::new(vec![rows, nff], DType::F32));
        g.push(Op::GatedAct {
            gate,
            up,
            dst,
            rows: rows as u32,
            nff: nff as u32,
            act: Activation::Silu,
            up_off: 0,
            up_stride: 0,
            gate_stride: (2 * nff) as u32, // strided gate — unsupported by silu_mul
            gate_block_width: 0,
            swiglu_clamp: None,
        });
        let gb = be_.alloc(rows * nff * 4, BufferUsage::Activations).unwrap();
        let ub = be_.alloc(rows * nff * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(rows * nff * 4, BufferUsage::Activations).unwrap();
        let mut bind = Bindings::new();
        bind.bind(gate, gb.as_ref());
        bind.bind(up, ub.as_ref());
        bind.bind(dst, yb.as_ref());
        let res = be_
            .compile(&g)
            .and_then(|plan| be_.execute(plan.as_ref(), &bind));
        assert!(
            res.is_err(),
            "strided-gate Silu must return Err (not compute silently-wrong / panic)"
        );
    }

    /// #4: a cross-dtype (F32→F16) `Op::Copy` with `src_off != 0` is structurally valid but
    /// unsupported (`store_f16` has no source offset). It must return a recoverable `Err` — the old
    /// `assert_eq!` aborted the whole process. (GPU-gated: reaches the guard via `lower_op`.)
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn cross_dtype_copy_src_off_errors() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let mut g = Graph::new();
        let src = g.input(TensorDesc::new(vec![4usize], DType::F32));
        let dst = g.output(TensorDesc::new(vec![2usize], DType::F16));
        g.push(Op::Copy {
            src,
            src_off: 1, // nonzero source offset — unsupported for the cast-copy
            dst,
            dst_off: 0,
            n: 2,
        });
        let sb = be_.alloc(4 * 4, BufferUsage::Activations).unwrap();
        let db = be_.alloc(2 * 2, BufferUsage::Activations).unwrap();
        let mut bind = Bindings::new();
        bind.bind(src, sb.as_ref());
        bind.bind(dst, db.as_ref());
        let res = be_
            .compile(&g)
            .and_then(|plan| be_.execute(plan.as_ref(), &bind));
        assert!(
            res.is_err(),
            "cross-dtype Copy with src_off!=0 must return Err (not panic the process)"
        );
    }

    /// THE symmetry invariant (the Q5_K token-divergence bug class, guarded at the policy level):
    /// for EVERY dtype, the m=1 DECODE tier (`mmv_int8_decode_dtypes`) and the m>=3 MTP-VERIFY
    /// tier (`mrow_int8_dtype_ok(.., verify: true)`) must agree on EVERY (vendor × env)
    /// combination — int8 in both streams or in neither. A dtype that is int8 in one stream and
    /// f32-exact in the other flips the occasional greedy argmax between the spec and non-spec
    /// streams (this is precisely how the Q5_K attempt broke `mtp_spec_matches_target_only_greedy`,
    /// and how the pre-split code's unconditional Q4_K/Q6_K/IQ4_XS mrow tier could have too — see
    /// `mrow_int8_dtype_ok`'s doc).
    ///
    /// This is the NEW, stronger invariant post-split: unlike the old test, NO dtype is exempt —
    /// `verify` no longer has an unconditional arm for Q4_K/Q6_K/IQ4_XS. ORDINARY prefill (verify:
    /// false) is intentionally NOT covered by this loop: it has no bit-identity partner, so it is
    /// checked separately below against `mrow_int8_prefill_dtypes` instead.
    ///
    /// EVERY dtype with an int8 mrow build must be listed in `POLICY_DTYPES` — the k-quants, IQ4_XS,
    /// and the legacy 32-block family.
    const POLICY_DTYPES: [DType; 12] = [
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::Iq4Xs,
        DType::Q8_0,
        DType::Q4_0,
        DType::Q5_0,
        DType::Q4_1,
        DType::Q5_1,
        DType::Iq4Nl,
    ];

    #[test]
    fn int8_decode_and_mrow_tiers_agree_on_policy_dtypes() {
        let amd = infr_core::backend::Capabilities {
            i8_dot: true,
            ..Default::default()
        };
        let intel = infr_core::backend::Capabilities {
            i8_dot: true,
            ..Default::default()
        };
        // `kernels.vulkan.mmv_mw`, the tri-state (§10.3): `None` = unset (the shipping default),
        // `Some(true)` = `INFR_MMV_MW=1` (any non-"0" value), `Some(false)` = `INFR_MMV_MW=0`.
        // A VALUE per sweep step since S5b — no process env, no guard, no serialization needed.
        let vkcfg = |mmv_mw: Option<bool>| infr_core::config::VulkanCfg {
            mmv_mw,
            ..Default::default()
        };
        for mmv_mw in [None, Some(true), Some(false)] {
            let vk = vkcfg(mmv_mw);
            for (caps, vendor) in [(&amd, "amd"), (&intel, "intel")] {
                for dt in POLICY_DTYPES {
                    let decode_int8 = mmv_int8_decode_dtypes(caps, &vk).contains(&dt);
                    let verify_int8 = mrow_int8_dtype_ok(caps, &vk, dt, true);
                    assert_eq!(
                        decode_int8, verify_int8,
                        "{vendor} mmv_mw={mmv_mw:?} {dt:?}: decode int8={decode_int8} but \
                         MTP-verify int8={verify_int8} — ASYMMETRIC (the Q5_K bug class)"
                    );
                    // Ordinary prefill is NEVER more conservative than MTP-verify (the decode set
                    // is always a subset of the prefill set) — prefill has no partner to protect,
                    // so it can only be equal or more permissive.
                    let prefill_int8 = mrow_int8_dtype_ok(caps, &vk, dt, false);
                    assert!(
                        prefill_int8 || !verify_int8,
                        "{vendor} mmv_mw={mmv_mw:?} {dt:?}: verify int8={verify_int8} but \
                         prefill int8={prefill_int8} — prefill is stricter than verify, backwards"
                    );
                }
            }
        }
        // The rest of the assertions are on the SHIPPING default (`mmv_mw: None`).
        let vk = vkcfg(None);
        // The shipping AMD default, spelled out so a policy edit has to face it: Q2_K int8 in BOTH
        // decode and verify (the measured +20% win), Q3_K f32-exact in both. Q3_K's DECODE stays
        // off despite the accuracy isolation (see `mrow_int8_prefill_dtypes`'s doc) finding the
        // cliff was decode-side, not prefill-side — decode was never proven safe, just not (yet)
        // re-attempted, so it stays conservatively off pending its own measurement.
        // Q3_K: OFF on every vendor — coherence cliff on mixed GGUFs (see
        // mmv_int8_decode_dtypes). Intel previously had it default-on; the unified set
        // drops it for safety. INFR_MMV_MW=1 still enables it for A/B.
        assert!(!mmv_int8_decode_dtypes(&amd, &vk).contains(&DType::Q3K));
        assert!(!mrow_int8_dtype_ok(&amd, &vk, DType::Q3K, true));
        assert!(!mmv_int8_decode_dtypes(&intel, &vk).contains(&DType::Q3K));
        // Q4_K is ON in the unified decode default: the throughput win (README footnote 3) is real.
        assert!(mmv_int8_decode_dtypes(&amd, &vk).contains(&DType::Q4K));
        assert!(mrow_int8_dtype_ok(&amd, &vk, DType::Q4K, true));
        assert!(mmv_int8_decode_dtypes(&intel, &vk).contains(&DType::Q4K));
        assert!(unified_mmv_row1(&amd), "rows=1 unified path must be on");
        // Q2_K is ON in the unified decode default (+20.4% measured win on AMD).
        assert!(mmv_int8_decode_dtypes(&amd, &vk).contains(&DType::Q2K));
        assert!(mrow_int8_dtype_ok(&amd, &vk, DType::Q2K, true));
        assert!(mmv_int8_decode_dtypes(&intel, &vk).contains(&DType::Q2K));
        // Q5_K: decode measured a small LOSS on AMD (66.8 int8 vs 67.8 f32, -1.4%; see
        // mmv_int8_decode_dtypes's doc) — OFF the decode/verify default on every vendor.
        // Its ordinary-prefill win (+45%) IS banked below regardless.
        assert!(!mmv_int8_decode_dtypes(&amd, &vk).contains(&DType::Q5K));
        assert!(!mrow_int8_dtype_ok(&amd, &vk, DType::Q5K, true));
        // Q6_K: decode-ON since the word-parallel `wdec` rewrite (see `mmv_int8_decode_dtypes`).
        // Unified across all vendors.
        assert!(mmv_int8_decode_dtypes(&amd, &vk).contains(&DType::Q6K));
        assert!(mmv_int8_decode_dtypes(&intel, &vk).contains(&DType::Q6K));
        assert!(!mrow_int8_dtype_ok(&amd, &vk, DType::Iq4Xs, true));
        // The legacy 32-block set's decode default is a MEASURED split (see
        // `mmv_int8_decode_dtypes`): Q4_0/Q5_0/Q5_1/IQ4_NL win at m=1 and are ON; Q8_0 LOSES
        // (−4.2%) and Q4_1 washes, so both are decode-OFF and prefill-ONLY. Unified across all
        // vendors — spelled out so a "finish the set" edit has to face the numbers.
        for caps in [&amd, &intel] {
            for dt in [DType::Q4_0, DType::Q5_0, DType::Q5_1, DType::Iq4Nl] {
                assert!(
                    mmv_int8_decode_dtypes(caps, &vk).contains(&dt),
                    "{dt:?} decode"
                );
            }
            for dt in [DType::Q8_0, DType::Q4_1] {
                assert!(
                    !mmv_int8_decode_dtypes(caps, &vk).contains(&dt),
                    "{dt:?}: decode measured a loss/wash — prefill-only, do not flip without a number"
                );
                assert!(mrow_int8_prefill_dtypes(dt), "{dt:?} prefill win must stay");
            }
        }
        // Every dtype's ORDINARY PREFILL is unconditionally on regardless of the decode/verify
        // policy above — this is the actual point of the split: prefill has no partner to protect.
        // Unified across all vendors.
        for caps in [&amd, &intel] {
            for dt in POLICY_DTYPES {
                assert!(
                    mrow_int8_dtype_ok(caps, &vk, dt, false),
                    "{dt:?} ordinary-prefill mrow must stay on"
                );
            }
        }
        // Intel now takes the same unified rows=1 path (unconditional, part of the
        // capability-first architecture).
        assert!(unified_mmv_row1(&intel));
    }

    /// The drift guard the SSOT lists promise (`infr_core::tensor::MOE_MMQ_DTYPES`'s doc): every
    /// dtype claimed by the batched-mmq family must ALSO have the small-m id-GEMV kernels its
    /// per-token/decode fallback needs (a format on the fast path but not the slow one would
    /// panic at decode time), the paged set must have its paged twins AND mirror the mmq set in
    /// full (the pager is the sole MoE offload mechanism — an mmq dtype missing a paged build
    /// would silently fall back to the far slower id-GEMV prefill segment when paged), and the
    /// subset/sact relations must hold. Pure name-table checks — the per-format GPU parity tests
    /// in this module exercise the actual `matmul_mmq_experts(_paged)` dispatch arms (whose `_ =>
    /// unreachable!` is the runtime backstop for a listed-but-unwired format).
    #[test]
    fn moe_mmq_drift_test() {
        use infr_core::tensor::{MOE_MMQ_DTYPES, MOE_MMQ_PAGED_DTYPES, MOE_MMQ_SACT_DTYPES};
        for &dt in MOE_MMQ_DTYPES {
            assert!(
                crate::linear::native_id_kernel_name(dt).is_some(),
                "{dt:?} is mmq-covered but has no small-m id-GEMV kernel (decode fallback)"
            );
            assert!(
                crate::linear::native_idm_kernel_name(dt).is_some(),
                "{dt:?} is mmq-covered but has no multi-slot idm-GEMV kernel (decode fallback)"
            );
            assert!(
                infr_core::tensor::moe_paged_mmq_ok(dt),
                "{dt:?} is mmq-covered but has no paged (_xpg) batched build — paged prefill \
                 would silently degrade to the id-GEMV segment"
            );
        }
        for &dt in MOE_MMQ_PAGED_DTYPES {
            assert!(
                infr_core::tensor::moe_mmq_ok(dt),
                "{dt:?} is paged-mmq but not in MOE_MMQ_DTYPES (subset violated)"
            );
            assert!(
                crate::linear::native_id_paged_kernel_name(dt).is_some()
                    && crate::linear::native_idm_paged_kernel_name(dt).is_some(),
                "{dt:?} is paged-mmq but lacks a paged id/idm-GEMV kernel (paged decode fallback)"
            );
        }
        for &dt in MOE_MMQ_SACT_DTYPES {
            assert!(
                infr_core::tensor::moe_mmq_ok(dt),
                "{dt:?} is sact-classified but not in MOE_MMQ_DTYPES (subset violated)"
            );
        }
    }

    /// Prove the adapter machinery end-to-end (compile → bind → execute → download): a one-op
    /// `RmsNorm` graph run through the Vulkan seam must match a host reference. (Milestone #2: a
    /// small graph runs on Vulkan.)
    ///
    /// On the SHARED harness (`infr_testkit::run_graph`) — see the `linear_graph_matches_host`
    /// doc for what the conversion buys and what is left to follow up.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn rmsnorm_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, dim, eps) = (2usize, 8usize, 1e-6f32);
        let x: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.1 - 0.4).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 * 0.05).collect();
        // host reference rmsnorm
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let b = r * dim;
            let ss = (0..dim).map(|i| x[b + i] * x[b + i]).sum::<f32>() / dim as f32;
            let s = 1.0 / (ss + eps).sqrt();
            for i in 0..dim {
                want[b + i] = x[b + i] * s * w[i];
            }
        }
        // build the graph
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![dim], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::RmsNorm {
            x: xi,
            weight: wi,
            dst: yi,
            rows: rows as u32,
            dim: dim as u32,
            eps,
        });
        let got = infr_testkit::run_graph(
            &be_,
            &g,
            &[
                (xi, infr_testkit::f32_bytes(&x)),
                (wi, infr_testkit::f32_bytes(&w)),
            ],
            yi,
            rows * dim,
        );
        for i in 0..rows * dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "rmsnorm mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `GatedRmsNorm` graph (fused per-head RMSNorm + SiLU gate, qwen35 DeltaNet z-gate)
    /// through the seam must match `rmsnorm(x,w) * silu(z)` computed in two host passes — i.e.
    /// the SAME thing the split `QkNorm`→`GatedAct` pair produces. Multi-head (n_head=2) shape so
    /// the per-head reduction boundary is exercised, not just a single-row rmsnorm.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn gated_rmsnorm_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, n_head, head_dim, eps) = (2usize, 2usize, 8usize, 1e-6f32);
        let dim = n_head * head_dim;
        let x: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.1 - 0.4).collect();
        let w: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32 * 0.05).collect();
        let z: Vec<f32> = (0..rows * dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        // host reference: per-head rmsnorm, then elementwise silu(z) gate — the split-op semantics.
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            for h in 0..n_head {
                let b = (r * n_head + h) * head_dim;
                let ss = (0..head_dim).map(|i| x[b + i] * x[b + i]).sum::<f32>() / head_dim as f32;
                let s = 1.0 / (ss + eps).sqrt();
                for i in 0..head_dim {
                    want[b + i] = x[b + i] * s * w[i] * silu(z[b + i]);
                }
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![head_dim], DType::F32));
        let zi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::GatedRmsNorm {
            x: xi,
            weight: wi,
            gate: zi,
            dst: yi,
            rows: rows as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            eps,
        });
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(head_dim * 4, BufferUsage::Weights).unwrap();
        let zb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        be_.upload(zb.as_ref(), bytemuck::cast_slice(&z)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(zi, zb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; rows * dim];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..rows * dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "gated_rmsnorm mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `Linear` graph (f16 weight, 1-row GEMV) through the seam must match a host matvec.
    ///
    /// **On the shared harness** (backend-unification candidate H): the graph build stays here —
    /// it IS what the test is about — but the bind/alloc/upload/compile/execute/download ladder and
    /// the host oracle now come from `infr-testkit`, which cpu/metal drive the same way. Same
    /// shapes, same reference, same tolerance; ~20 lines of boilerplate gone.
    ///
    /// Follow-up: the other ~70 `*_matches_host` tests in this crate can move the same way, but
    /// several (the `moe_ffn_*` family, the pager/mmq integration tests) build multi-op graphs with
    /// several read-back tensors, which needs a multi-output `run_graph` the harness does not have
    /// yet. Converted here + `rmsnorm_graph_matches_host` + `gated_act_silu_matches_host` to prove
    /// the shape, deliberately NOT mass-rewritten.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn linear_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (in_f, out_f) = (16usize, 4usize);
        let x: Vec<f32> = (0..in_f).map(|i| i as f32 * 0.1 - 0.8).collect();
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.03).sin()).collect();
        let wf16 = infr_testkit::f16_bytes(&w);
        // Host reference uses the same f16-rounded weight the GPU reads — i.e. the shared decode
        // oracle over the SAME bytes, then the shared reference GEMV.
        let wq = infr_testkit::dequant_oracle(DType::F16, &wf16);
        let want = infr_testkit::ref_linear(&x, &wq, 1, in_f, out_f);
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, in_f], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F16));
        let yi = g.output(TensorDesc::new(vec![1, out_f], DType::F32));
        g.push(Op::Linear {
            x: xi,
            weight: wi,
            dst: yi,
            m: 1,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        let got = infr_testkit::run_graph(
            &be_,
            &g,
            &[(xi, infr_testkit::f32_bytes(&x)), (wi, wf16)],
            yi,
            out_f,
        );
        for o in 0..out_f {
            assert!(
                (got[o] - want[o]).abs() < 1e-2,
                "linear mismatch at {o}: got {} want {}",
                got[o],
                want[o]
            );
        }
    }

    /// A fused F16 weight can expose a row-aligned projection through `Linear::w_off` without
    /// copying the slice into a second buffer. Cover both scalar GEMV and tiled GEMM dispatch.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn f16_linear_offset_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return;
        };
        let (in_f, out_f) = (32usize, 64usize);
        for rows in [1usize, 17] {
            let x: Vec<f32> = (0..rows * in_f)
                .map(|i| (i as f32 * 0.071).sin() * 0.25)
                .collect();
            let stacked: Vec<f32> = (0..3 * out_f * in_f)
                .map(|i| (i as f32 * 0.013 + 0.3).cos() * 0.2)
                .collect();
            let wf16 = infr_testkit::f16_bytes(&stacked);
            let wq = infr_testkit::dequant_oracle(DType::F16, &wf16);
            let w_off = out_f * in_f;
            let want =
                infr_testkit::ref_linear(&x, &wq[w_off..w_off + out_f * in_f], rows, in_f, out_f);
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, in_f], DType::F32));
            let wi = g.weight(TensorDesc::new(vec![3 * out_f, in_f], DType::F16));
            let yi = g.output(TensorDesc::new(vec![rows, out_f], DType::F32));
            g.push(Op::Linear {
                x: xi,
                weight: wi,
                dst: yi,
                m: rows as u32,
                in_f: in_f as u32,
                out_f: out_f as u32,
                w_off: w_off as u32,
            });
            let got = infr_testkit::run_graph(
                &be_,
                &g,
                &[(xi, infr_testkit::f32_bytes(&x)), (wi, wf16)],
                yi,
                rows * out_f,
            );
            for (i, (&got, &want)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (got - want).abs() < 2e-2,
                    "rows={rows} offset linear mismatch at {i}: got {got} want {want}"
                );
            }
        }
    }

    /// A one-op `GatedAct` (SwiGLU) graph through the seam must match a host loop.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn gated_act_silu_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let nff = 8usize;
        let gate: Vec<f32> = (0..nff).map(|i| i as f32 * 0.2 - 0.7).collect();
        let up: Vec<f32> = (0..nff).map(|i| 1.0 - i as f32 * 0.1).collect();
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let want: Vec<f32> = (0..nff).map(|i| silu(gate[i]) * up[i]).collect();
        let mut g = Graph::new();
        let gi = g.input(TensorDesc::new(vec![1, nff], DType::F32));
        let ui = g.input(TensorDesc::new(vec![1, nff], DType::F32));
        let yi = g.output(TensorDesc::new(vec![1, nff], DType::F32));
        g.push(Op::GatedAct {
            gate: gi,
            up: ui,
            dst: yi,
            rows: 1,
            nff: nff as u32,
            act: Activation::Silu,
            up_off: 0,
            gate_stride: 0,
            up_stride: 0,
            gate_block_width: 0,
            swiglu_clamp: None,
        });
        let got = infr_testkit::run_graph(
            &be_,
            &g,
            &[
                (gi, infr_testkit::f32_bytes(&gate)),
                (ui, infr_testkit::f32_bytes(&up)),
            ],
            yi,
            nff,
        );
        for i in 0..nff {
            assert!(
                (got[i] - want[i]).abs() < 1e-3,
                "gated_act mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `QkNormRope` graph (per-head RMSNorm + RoPE, f32 in → f16 out, positions tensor) must
    /// match a host reference (f16 tolerance).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn qk_norm_rope_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, hd, pos) = (2usize, 8usize, 3usize);
        let (eps, theta, rope_dim) = (1e-6f32, 10000.0f32, 8usize);
        let x: Vec<f32> = (0..nh * hd).map(|i| i as f32 * 0.1 - 0.5).collect();
        let w: Vec<f32> = (0..hd).map(|i| 1.0 + i as f32 * 0.05).collect();
        // host reference: per-head rmsnorm (×w) then split-half NEOX rope
        let mut want = vec![0f32; nh * hd];
        let hf = rope_dim / 2;
        for h in 0..nh {
            let b = h * hd;
            let ss = (0..hd).map(|i| x[b + i] * x[b + i]).sum::<f32>() / hd as f32;
            let s = 1.0 / (ss + eps).sqrt();
            let nrm: Vec<f32> = (0..hd).map(|i| x[b + i] * s * w[i]).collect();
            want[b..b + hd].copy_from_slice(&nrm);
            for p in 0..hf {
                let ang = pos as f32 * theta.powf(-2.0 * p as f32 / rope_dim as f32);
                let (sn, c) = (ang.sin(), ang.cos());
                want[b + p] = nrm[p] * c - nrm[p + hf] * sn;
                want[b + p + hf] = nrm[p] * sn + nrm[p + hf] * c;
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, nh, hd], DType::F32));
        let wi = g.weight(TensorDesc::new(vec![hd], DType::F32));
        let pi = g.input(TensorDesc::new(vec![1], DType::I32));
        let yi = g.output(TensorDesc::new(vec![1, nh, hd], DType::F16));
        g.push(Op::QkNormRope {
            x: xi,
            weight: wi,
            positions: pi,
            dst: yi,
            rows: 1,
            n_head: nh as u32,
            head_dim: hd as u32,
            rope_dim: rope_dim as u32,
            theta,
            eps,
            freq_factors: None,
            x_stride: 0,
        });
        let xb = be_.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
        let wb = be_.alloc(hd * 4, BufferUsage::Weights).unwrap();
        let pb = be_.alloc(4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        be_.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        be_.upload(pb.as_ref(), &(pos as i32).to_le_bytes())
            .unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(wi, wb.as_ref());
        bind.bind(pi, pb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut y16 = vec![0u8; nh * hd * 2];
        be_.download(yb.as_ref(), &mut y16).unwrap();
        let got: Vec<f32> = y16
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 2e-2,
                "qk_norm_rope mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `Op::QkNorm` graph with F16 x/dst (llama4's post-rope weightless Q/K L2-norm,
    /// in-place on the f16 rope scratch — `x == dst`, exactly the shape `seam/runner.rs` emits on
    /// rope layers) must match a host per-head RMSNorm computed on the SAME f16-rounded input.
    /// Multi-head (n_head=2) exercises the per-head reduction boundary — the op doesn't
    /// distinguish Q from K, just a head count, so this also covers K's `nkv`-headed call.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn qk_norm_f16_inplace_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, hd, eps) = (2usize, 8usize, 1e-6f32);
        let to_f16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let deq = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let x_f32: Vec<f32> = (0..nh * hd).map(|i| i as f32 * 0.1 - 0.5).collect();
        let w: Vec<f32> = (0..hd).map(|i| 1.0 + i as f32 * 0.05).collect(); // NOT ones — general
        let xb16 = to_f16(&x_f32);
        let x = deq(&xb16); // the f16-rounded input the GPU shader actually reads
                            // host reference: per-head rmsnorm(x) * w, on the SAME f16-rounded input.
        let mut want = vec![0f32; nh * hd];
        for h in 0..nh {
            let b = h * hd;
            let ss = (0..hd).map(|i| x[b + i] * x[b + i]).sum::<f32>() / hd as f32;
            let s = 1.0 / (ss + eps).sqrt();
            for i in 0..hd {
                want[b + i] = x[b + i] * s * w[i];
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, nh, hd], DType::F16));
        let wi = g.weight(TensorDesc::new(vec![hd], DType::F32));
        g.push(Op::QkNorm {
            x: xi,
            weight: Some(wi),
            dst: xi, // in-place — matches the runner's q16/k16 usage exactly
            rows: 1,
            n_head: nh as u32,
            head_dim: hd as u32,
            eps,
            x_stride: 0,
        });
        let xbuf = be_.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
        let wbuf = be_.alloc(hd * 4, BufferUsage::Weights).unwrap();
        be_.upload(xbuf.as_ref(), &xb16).unwrap();
        be_.upload(wbuf.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xbuf.as_ref());
        bind.bind(wi, wbuf.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got16 = vec![0u8; nh * hd * 2];
        be_.download(xbuf.as_ref(), &mut got16).unwrap();
        let got = deq(&got16);
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 2e-2,
                "qk_norm f16 in-place mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `Op::Copy` graph with a cross-dtype (f32 src → f16 dst) shape — llama4's NoPE
    /// (global) attention layers cast the raw Q/K projection into the f16 rope scratch this way
    /// (`seam/runner.rs`, `nope` branch) — must match the f32 values cast to f16, NOT a raw byte
    /// copy (which would corrupt every value: f32 is 2x f16's width). Same-dtype `Copy` (the
    /// common case elsewhere) is exercised implicitly by every other graph test that uses it.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn copy_f32_to_f16_cast_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let n = 37usize; // not a multiple of the dispatch's local size — exercises the tail
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin() * 3.0).collect();
        let want: Vec<f32> = x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![n], DType::F32));
        let yi = g.output(TensorDesc::new(vec![n], DType::F16));
        g.push(Op::Copy {
            src: xi,
            src_off: 0,
            dst: yi,
            dst_off: 0,
            n: n as u32,
        });
        let xb = be_.alloc(n * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(n * 2, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got16 = vec![0u8; n * 2];
        be_.download(yb.as_ref(), &mut got16).unwrap();
        let got: Vec<f32> = got16
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        for i in 0..n {
            assert_eq!(
                got[i], want[i],
                "copy f32->f16 cast mismatch at {i}: got {} want {}",
                got[i], want[i]
            );
        }
    }

    /// A one-op `Attention` graph (GQA, causal, f16 q + f16 KV) must match a host softmax-attention.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn attention_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, nkv, hd, pos) = (2usize, 1usize, 8usize, 2usize);
        let kv_len = pos + 1; // causal: query at `pos` attends keys 0..=pos
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let to_f16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let deq = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let q: Vec<f32> = (0..nh * hd).map(|i| (i as f32 * 0.07).sin()).collect();
        let k: Vec<f32> = (0..kv_len * nkv * hd)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..kv_len * nkv * hd)
            .map(|i| i as f32 * 0.01 - 0.1)
            .collect();
        let (qf, kf, vf) = (to_f16(&q), to_f16(&k), to_f16(&v));
        let (qd, kd, vd) = (deq(&qf), deq(&kf), deq(&vf)); // host uses the same f16-rounded values
                                                           // host GQA softmax attention
        let mut want = vec![0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / group;
            let mut sc = vec![0f32; kv_len];
            let mut mx = f32::NEG_INFINITY;
            for (j, scj) in sc.iter_mut().enumerate() {
                let d: f32 = (0..hd)
                    .map(|x| qd[h * hd + x] * kd[(j * nkv + kvh) * hd + x])
                    .sum();
                *scj = d * scale;
                mx = mx.max(*scj);
            }
            let l: f32 = sc.iter().map(|s| (s - mx).exp()).sum();
            for (j, &s) in sc.iter().enumerate() {
                let p = (s - mx).exp() / l;
                for x in 0..hd {
                    want[h * hd + x] += p * vd[(j * nkv + kvh) * hd + x];
                }
            }
        }
        let mut g = Graph::new();
        let qi = g.input(TensorDesc::new(vec![1, nh, hd], DType::F16));
        let ki = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let vi = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let yi = g.output(TensorDesc::new(vec![1, nh, hd], DType::F32));
        g.push(Op::Attention {
            q: qi,
            k_cache: ki,
            v_cache: vi,
            dst: yi,
            rows: 1,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale,
            mask: AttnMask::Causal,
            pos: pos as u32,
            sinks: None,
        });
        let qb = be_.alloc(qf.len(), BufferUsage::Activations).unwrap();
        let kb = be_.alloc(kf.len(), BufferUsage::Activations).unwrap();
        let vb = be_.alloc(vf.len(), BufferUsage::Activations).unwrap();
        let yb = be_.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
        be_.upload(qb.as_ref(), &qf).unwrap();
        be_.upload(kb.as_ref(), &kf).unwrap();
        be_.upload(vb.as_ref(), &vf).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(qi, qb.as_ref());
        bind.bind(ki, kb.as_ref());
        bind.bind(vi, vb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; nh * hd];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..nh * hd {
            assert!(
                (got[i] - want[i]).abs() < 2e-2,
                "attention mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// A one-op `Softmax` graph (row-wise, with a non-trivial `scale`) must match a host softmax.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn softmax_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, dim, scale) = (3usize, 300usize, 1.3f32);
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.037).sin() * 5.0)
            .collect();
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let b = r * dim;
            let row = &x[b..b + dim];
            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
            let mut denom = 0f32;
            for (w, &v) in want[b..b + dim].iter_mut().zip(row) {
                *w = (v * scale - mx).exp();
                denom += *w;
            }
            for w in want[b..b + dim].iter_mut() {
                *w /= denom;
            }
        }
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::Softmax {
            x: xi,
            dst: yi,
            rows: rows as u32,
            dim: dim as u32,
            scale,
            scale_buf: None,
        });
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; rows * dim];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..rows * dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-4,
                "softmax mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// `Op::Softmax` with `scale_buf: Some(_)` (the DiffusionGemma denoise perf path — see its doc)
    /// must match the SAME host softmax as `scale`, with the scale sourced from a bound 1-element
    /// buffer instead of a push constant. Also checks that a plan built ONCE and re-`execute`d with
    /// a DIFFERENT `scale_buf` value each time picks up the new scale — the whole point of the
    /// mechanism (a cached/replayed plan varying its softmax temperature without a rebuild).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn softmax_dyn_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (rows, dim) = (3usize, 300usize);
        let x: Vec<f32> = (0..rows * dim)
            .map(|i| (i as f32 * 0.037).sin() * 5.0)
            .collect();
        let host_softmax = |scale: f32| -> Vec<f32> {
            let mut want = vec![0f32; rows * dim];
            for r in 0..rows {
                let b = r * dim;
                let row = &x[b..b + dim];
                let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
                let mut denom = 0f32;
                for (w, &v) in want[b..b + dim].iter_mut().zip(row) {
                    *w = (v * scale - mx).exp();
                    denom += *w;
                }
                for w in want[b..b + dim].iter_mut() {
                    *w /= denom;
                }
            }
            want
        };
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
        let si = g.input(TensorDesc::new(vec![1], DType::F32));
        let yi = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
        g.push(Op::Softmax {
            x: xi,
            dst: yi,
            rows: rows as u32,
            dim: dim as u32,
            scale: 1.0, // ignored — scale_buf takes over
            scale_buf: Some(si),
        });
        let xb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        let sb = be_.alloc(4, BufferUsage::Staging).unwrap();
        let yb = be_.alloc(rows * dim * 4, BufferUsage::Activations).unwrap();
        be_.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(si, sb.as_ref());
        bind.bind(yi, yb.as_ref());
        // Same compiled plan, replayed with two DIFFERENT scale_buf values — exactly the
        // DiffusionGemma denoise loop's per-step usage (see `crates/infr-llama/src/seam/
        // runner.rs`'s denoise call site).
        for &scale in &[1.3f32, 0.42f32] {
            be_.upload(sb.as_ref(), &scale.to_le_bytes()).unwrap();
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * dim];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            let want = host_softmax(scale);
            for i in 0..rows * dim {
                assert!(
                    (got[i] - want[i]).abs() < 1e-4,
                    "softmax_dyn mismatch at scale={scale} i={i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// DiffusionGemma Phase-B: the FULL in-graph self-conditioning subgraph (softmax → soft-embed
    /// matmul → scale → rmsnorm → gate/up linears → gated-GELU → down linear — the exact op
    /// sequence `seam.rs`'s `build` emits for a SC-on denoise step) at small synthetic dims,
    /// compared against a host reference computed against the SAME f16-rounded weights (isolating
    /// the graph WIRING/layout from f16 rounding noise, like `linear_graph_matches_host`). This is
    /// "one SC denoise step" in miniature — the real model's SC block is this identical op
    /// sequence at production sizes (ne/vocab/nff/canvas_length), just with GGUF-loaded weights.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn diffusion_gemma_sc_subgraph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (cc, vocab, ne, nff) = (4usize, 512usize, 128usize, 256usize);
        let eps = 1e-6f32;
        let scale = 1.0f32; // production always premultiplies temp_inv on the host (see seam.rs)

        let to_f16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let deq = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };

        let sc_logits: Vec<f32> = (0..cc * vocab)
            .map(|i| (i as f32 * 0.0131).sin() * 4.0)
            .collect();
        let embt: Vec<f32> = (0..ne * vocab)
            .map(|i| (i as f32 * 0.0007).cos() * 0.3)
            .collect();
        let gate_w: Vec<f32> = (0..nff * ne)
            .map(|i| (i as f32 * 0.0011).sin() * 0.2)
            .collect();
        let up_w: Vec<f32> = (0..nff * ne)
            .map(|i| (i as f32 * 0.0017).cos() * 0.2)
            .collect();
        let down_w: Vec<f32> = (0..ne * nff)
            .map(|i| (i as f32 * 0.0013).sin() * 0.2)
            .collect();
        let pre_norm: Vec<f32> = (0..ne).map(|i| 1.0 + i as f32 * 0.01).collect();

        // Host uses the SAME f16-rounded weights the GPU reads (isolates wiring, not rounding).
        let (embt16, gate16, up16, down16) = (
            to_f16(&embt),
            to_f16(&gate_w),
            to_f16(&up_w),
            to_f16(&down_w),
        );
        let (embq, gateq, upq, downq) = (deq(&embt16), deq(&gate16), deq(&up16), deq(&down16));

        let mut want = vec![0f32; cc * ne];
        for r in 0..cc {
            let row = &sc_logits[r * vocab..(r + 1) * vocab];
            let mx = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v * scale));
            let mut probs = vec![0f32; vocab];
            let mut denom = 0f32;
            for (p, &v) in probs.iter_mut().zip(row) {
                *p = (v * scale - mx).exp();
                denom += *p;
            }
            for p in probs.iter_mut() {
                *p /= denom;
            }
            // soft = probs @ embT — embq is [ne, vocab] row-major (Op::Linear's weight layout).
            let mut soft = vec![0f32; ne];
            for (e, s) in soft.iter_mut().enumerate() {
                let erow = &embq[e * vocab..(e + 1) * vocab];
                *s = probs.iter().zip(erow).map(|(&p, &w)| p * w).sum::<f32>() * (ne as f32).sqrt();
            }
            let ms: f32 = soft.iter().map(|&x| x * x).sum::<f32>() / ne as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            let normed: Vec<f32> = soft
                .iter()
                .zip(&pre_norm)
                .map(|(&x, &w)| x * inv * w)
                .collect();
            let mut gv = vec![0f32; nff];
            let mut uv = vec![0f32; nff];
            for f in 0..nff {
                let grow = &gateq[f * ne..(f + 1) * ne];
                let urow = &upq[f * ne..(f + 1) * ne];
                gv[f] = grow.iter().zip(&normed).map(|(&w, &x)| w * x).sum();
                uv[f] = urow.iter().zip(&normed).map(|(&w, &x)| w * x).sum();
            }
            let act: Vec<f32> = gv
                .iter()
                .zip(&uv)
                .map(|(&gd, &ud)| {
                    let gelu =
                        0.5 * gd * (1.0 + (0.797_884_6 * (gd + 0.044715 * gd * gd * gd)).tanh());
                    gelu * ud
                })
                .collect();
            for (e, w_row) in want[r * ne..(r + 1) * ne].iter_mut().enumerate() {
                let drow = &downq[e * nff..e * nff + nff];
                *w_row = drow.iter().zip(&act).map(|(&w, &a)| w * a).sum();
            }
        }

        let mut g = Graph::new();
        let logits_i = g.input(TensorDesc::new(vec![cc, vocab], DType::F32));
        let embt_i = g.weight(TensorDesc::new(vec![ne, vocab], DType::F16));
        let pre_norm_i = g.weight(TensorDesc::new(vec![ne], DType::F32));
        let gate_i = g.weight(TensorDesc::new(vec![nff, ne], DType::F16));
        let up_i = g.weight(TensorDesc::new(vec![nff, ne], DType::F16));
        let down_i = g.weight(TensorDesc::new(vec![ne, nff], DType::F16));
        let sig_i = g.output(TensorDesc::new(vec![cc, ne], DType::F32));

        let probs_i = g.internal(TensorDesc::new(vec![cc, vocab], DType::F32));
        g.push(Op::Softmax {
            x: logits_i,
            dst: probs_i,
            rows: cc as u32,
            dim: vocab as u32,
            scale,
            scale_buf: None,
        });
        let soft_i = g.internal(TensorDesc::new(vec![cc, ne], DType::F32));
        g.push(Op::Linear {
            x: probs_i,
            weight: embt_i,
            dst: soft_i,
            m: cc as u32,
            in_f: vocab as u32,
            out_f: ne as u32,
            w_off: 0,
        });
        g.push(Op::Scale {
            x: soft_i,
            dst: soft_i,
            s: (ne as f32).sqrt(),
            n: (cc * ne) as u32,
        });
        let normed_i = g.internal(TensorDesc::new(vec![cc, ne], DType::F32));
        g.push(Op::RmsNorm {
            x: soft_i,
            weight: pre_norm_i,
            dst: normed_i,
            rows: cc as u32,
            dim: ne as u32,
            eps,
        });
        let g_i = g.internal(TensorDesc::new(vec![cc, nff], DType::F32));
        let u_i = g.internal(TensorDesc::new(vec![cc, nff], DType::F32));
        g.push(Op::Linear {
            x: normed_i,
            weight: gate_i,
            dst: g_i,
            m: cc as u32,
            in_f: ne as u32,
            out_f: nff as u32,
            w_off: 0,
        });
        g.push(Op::Linear {
            x: normed_i,
            weight: up_i,
            dst: u_i,
            m: cc as u32,
            in_f: ne as u32,
            out_f: nff as u32,
            w_off: 0,
        });
        let act_i = g.internal(TensorDesc::new(vec![cc, nff], DType::F32));
        g.push(Op::GatedAct {
            gate: g_i,
            up: u_i,
            dst: act_i,
            rows: cc as u32,
            nff: nff as u32,
            act: Activation::Gelu,
            up_off: 0,
            gate_stride: 0,
            up_stride: 0,
            gate_block_width: 0,
            swiglu_clamp: None,
        });
        g.push(Op::Linear {
            x: act_i,
            weight: down_i,
            dst: sig_i,
            m: cc as u32,
            in_f: nff as u32,
            out_f: ne as u32,
            w_off: 0,
        });

        let logits_b = be_.alloc(cc * vocab * 4, BufferUsage::Activations).unwrap();
        let embt_b = be_.alloc(embt16.len(), BufferUsage::Weights).unwrap();
        let pre_norm_b = be_.alloc(ne * 4, BufferUsage::Weights).unwrap();
        let gate_b = be_.alloc(gate16.len(), BufferUsage::Weights).unwrap();
        let up_b = be_.alloc(up16.len(), BufferUsage::Weights).unwrap();
        let down_b = be_.alloc(down16.len(), BufferUsage::Weights).unwrap();
        let sig_b = be_.alloc(cc * ne * 4, BufferUsage::Activations).unwrap();
        be_.upload(logits_b.as_ref(), bytemuck::cast_slice(&sc_logits))
            .unwrap();
        be_.upload(embt_b.as_ref(), &embt16).unwrap();
        be_.upload(pre_norm_b.as_ref(), bytemuck::cast_slice(&pre_norm))
            .unwrap();
        be_.upload(gate_b.as_ref(), &gate16).unwrap();
        be_.upload(up_b.as_ref(), &up16).unwrap();
        be_.upload(down_b.as_ref(), &down16).unwrap();

        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(logits_i, logits_b.as_ref());
        bind.bind(embt_i, embt_b.as_ref());
        bind.bind(pre_norm_i, pre_norm_b.as_ref());
        bind.bind(gate_i, gate_b.as_ref());
        bind.bind(up_i, up_b.as_ref());
        bind.bind(down_i, down_b.as_ref());
        bind.bind(sig_i, sig_b.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; cc * ne];
        be_.download(sig_b.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();

        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (&x, &y) in a.iter().zip(b) {
                dot += x as f64 * y as f64;
                na += x as f64 * x as f64;
                nb += y as f64 * y as f64;
            }
            dot / (na.sqrt() * nb.sqrt())
        };
        for r in 0..cc {
            let (w, g_) = (&want[r * ne..(r + 1) * ne], &got[r * ne..(r + 1) * ne]);
            let c = cos(w, g_);
            println!("diffusion_gemma_sc_subgraph_matches_host: row {r} cosine={c:.5}");
            assert!(c > 0.99, "row {r}: SC subgraph cosine too low: {c}");
        }
    }

    /// Small-m attention through the split-K path (rows 2..63, kv_len > chunk — the SHORT TURN
    /// suffix-prefill shape): every row's output must match a host oracle with the PER-ROW causal
    /// bound (row r attends keys 0..=pos+r), full-causal and sliding-window. rows=17 also covers
    /// the 9..63 band; kv_len=300 > chunk (=64) forces the split route, not the scalar fallback.
    #[test]
    fn attention_small_m_split_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (nh, nkv, hd) = (4usize, 2usize, 64usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let to_f16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
                .collect()
        };
        let deq = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        // kv=300 exercises the per-row split grid; kv=8300 with rows >= 12 exercises the
        // rows-BATCHED tier (attn_partial_mrows_c256) — including a short last row-group (17 =
        // 4+4+4+4+1) and a windowed case. (Flash needs Internal q/dst, so raw-input graphs like
        // these always take the split family.)
        for (rows, window, kv_len) in [
            (2usize, 0usize, 300usize),
            (5, 0, 300),
            (8, 0, 300),
            (17, 0, 300),
            (4, 128, 300),
            (17, 0, 8300),
            (40, 0, 8300),
            (13, 256, 8300),
        ] {
            let pos = kv_len - rows; // the suffix occupies the last `rows` positions
            let q: Vec<f32> = (0..rows * nh * hd)
                .map(|i| (i as f32 * 0.03).sin())
                .collect();
            let k: Vec<f32> = (0..kv_len * nkv * hd)
                .map(|i| (i as f32 * 0.011).cos())
                .collect();
            let v: Vec<f32> = (0..kv_len * nkv * hd)
                .map(|i| ((i % 173) as f32) * 0.011 - 0.9)
                .collect();
            let (qf, kf, vf) = (to_f16(&q), to_f16(&k), to_f16(&v));
            let (qd, kd, vd) = (deq(&qf), deq(&kf), deq(&vf));
            // host oracle: per-row causal end pos+r+1, window lo = max(0, end - window)
            let mut want = vec![0f32; rows * nh * hd];
            for r0 in 0..rows {
                let end = pos + r0 + 1;
                let lo = if window > 0 && end > window {
                    end - window
                } else {
                    0
                };
                for h in 0..nh {
                    let kvh = h / group;
                    let mut sc = vec![0f32; end - lo];
                    let mut mx = f32::NEG_INFINITY;
                    for (jj, scj) in sc.iter_mut().enumerate() {
                        let j = lo + jj;
                        let d: f32 = (0..hd)
                            .map(|x| qd[(r0 * nh + h) * hd + x] * kd[(j * nkv + kvh) * hd + x])
                            .sum();
                        *scj = d * scale;
                        mx = mx.max(*scj);
                    }
                    let l: f32 = sc.iter().map(|s| (s - mx).exp()).sum();
                    for (jj, &s) in sc.iter().enumerate() {
                        let j = lo + jj;
                        let p = (s - mx).exp() / l;
                        for x in 0..hd {
                            want[(r0 * nh + h) * hd + x] += p * vd[(j * nkv + kvh) * hd + x];
                        }
                    }
                }
            }
            let mut g = Graph::new();
            let qi = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F16));
            let ki = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
            let vi = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
            let yi = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
            g.push(Op::Attention {
                q: qi,
                k_cache: ki,
                v_cache: vi,
                dst: yi,
                rows: rows as u32,
                kv_len: kv_len as u32,
                n_head: nh as u32,
                n_kv: nkv as u32,
                head_dim: hd as u32,
                scale,
                mask: if window > 0 {
                    AttnMask::SlidingWindow(window)
                } else {
                    AttnMask::Causal
                },
                pos: pos as u32,
                sinks: None,
            });
            let qb = be_.alloc(qf.len(), BufferUsage::Activations).unwrap();
            let kb = be_.alloc(kf.len(), BufferUsage::Activations).unwrap();
            let vb = be_.alloc(vf.len(), BufferUsage::Activations).unwrap();
            let yb = be_
                .alloc(rows * nh * hd * 4, BufferUsage::Activations)
                .unwrap();
            be_.upload(qb.as_ref(), &qf).unwrap();
            be_.upload(kb.as_ref(), &kf).unwrap();
            be_.upload(vb.as_ref(), &vf).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(qi, qb.as_ref());
            bind.bind(ki, kb.as_ref());
            bind.bind(vi, vb.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * nh * hd];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * nh * hd {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "small-m attention mismatch rows={rows} window={window} at {i} \
                     (row {} head {}): got {} want {}",
                    i / (nh * hd),
                    (i / hd) % nh,
                    got[i],
                    want[i]
                );
            }
        }
    }

    // ---- Q8_0 helpers (block=32: f16 scale + 32×int8 = 34 bytes) so a MoE test can bind id-native
    // expert banks and the host reference dequants the SAME rounded values the GPU shader does. ----
    fn q8_0(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 34);
        for blk in x.chunks(32) {
            let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for &v in blk {
                out.push(((v * id).round().clamp(-127.0, 127.0) as i8) as u8);
            }
        }
        out
    }
    fn deq_q8(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 34 * 32);
        for blk in bytes.chunks(34) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            for i in 0..32 {
                out.push((blk[2 + i] as i8 as f32) * d);
            }
        }
        out
    }

    // ---- Q4_K helpers (block=256: f16 d + f16 dmin + 12-byte packed 6-bit scale/min ×8 sub-blocks
    // + 128 nibbles = 144 bytes) — mirrors `native_gemm_mmq_q4k`'s decode exactly (including the
    // `get_scale_min_k4`-style 6-bit pack/unpack and the (sub_even low nibble, sub_odd high nibble)
    // interleave), so the batched-fused isolation test can synthesize a real Q4_K gate_up bank and
    // have the host reference dequant the SAME rounded values the GPU shader reads. Not a
    // rate-distortion-optimal quantizer (min/max per sub-block, not llama.cpp's search) — only
    // internal round-trip consistency matters for this test.
    fn q4k(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 256 * 144);
        for blk in x.chunks(256) {
            let mut sc = [0u32; 8];
            let mut mn = [0u32; 8];
            let mut sub_lo = [0f32; 8]; // per-sub-block min
            let mut sub_sc = [0f32; 8]; // per-sub-block (max-min)/15
            for (j, sub) in blk.chunks(32).enumerate() {
                let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                sub_lo[j] = lo;
                sub_sc[j] = ((hi - lo) / 15.0).max(1e-8);
            }
            let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 63.0;
            let dmin = sub_lo
                .iter()
                .cloned()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(1e-8)
                / 63.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let idmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
            for j in 0..8 {
                sc[j] = ((sub_sc[j] * id).round() as i32).clamp(0, 63) as u32;
                mn[j] = ((sub_lo[j].abs() * idmin).round() as i32).clamp(0, 63) as u32;
            }
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            // Pack the 8 (sc,mn) pairs into 12 bytes (get_scale_min_k4 convention).
            let mut scales = [0u8; 12];
            for k in 0..4 {
                scales[k] = (sc[k] & 0x3F) as u8 | (((sc[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[4 + k] = (mn[k] & 0x3F) as u8 | (((mn[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[8 + k] = (sc[4 + k] & 0xF) as u8 | (((mn[4 + k] & 0xF) as u8) << 4);
            }
            out.extend_from_slice(&scales);
            // Quantize each sub-block's 32 elements against ITS recovered (d*sc, dmin*mn) — the
            // SAME values the GPU decode reconstructs — then pack (sub_even low nibble, sub_odd
            // high nibble) per 32-byte pair-region.
            let mut q = [[0u8; 32]; 8];
            for j in 0..8 {
                let scale = d * sc[j] as f32;
                let min = dmin * mn[j] as f32;
                let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for (l, &v) in blk[j * 32..j * 32 + 32].iter().enumerate() {
                    q[j][l] = (((v - min) * iscale).round() as i32).clamp(0, 15) as u8;
                }
            }
            let mut qs = [0u8; 128];
            for pair in 0..4 {
                let (lo, hi) = (&q[2 * pair], &q[2 * pair + 1]);
                for l in 0..32 {
                    qs[pair * 32 + l] = (lo[l] & 0xF) | (hi[l] << 4);
                }
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_q4k(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 144 * 256);
        for blk in bytes.chunks(144) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            for j in 0..8u32 {
                let (sc, mn) = if j < 4 {
                    (
                        (scales[j as usize] & 0x3F) as u32,
                        (scales[j as usize + 4] & 0x3F) as u32,
                    )
                } else {
                    let i2 = j as usize - 4;
                    (
                        (scales[j as usize + 4] as u32 & 0xF) | (((scales[i2] as u32) >> 6) << 4),
                        (scales[j as usize + 4] as u32 >> 4)
                            | (((scales[i2 + 4] as u32) >> 6) << 4),
                    )
                };
                let scale = d * sc as f32;
                let min = dmin * mn as f32;
                let pair = (j / 2) as usize;
                let lo = (j % 2) == 0;
                for l in 0..32 {
                    let byte = qs[pair * 32 + l];
                    let nib = if lo { byte & 0xF } else { byte >> 4 };
                    out.push(scale * nib as f32 - min);
                }
            }
        }
        out
    }

    /// A one-op `MoeFfn` graph (Q8_0 router + stacked experts, split gate/up, unscaled — qwen3moe's
    /// exact shape) through the seam must match a host reference that mirrors the CPU `Op::MoeFfn`
    /// interpreter on the SAME q8-rounded weights: router softmax → top-`n_used` → per-expert
    /// SwiGLU → weighted (×scale) accumulate. Runs rows=1 (decode) and rows=4 (a small prefill
    /// chunk, e.g. `pp4`) — both ≤ `moe_small_m_threshold()`, so `gate`/`up` being Q8_0 (not
    /// Q4_K) doesn't matter here: this exercises only the small-m fast path either way, covering
    /// `linear_native_id_multi`'s multi-row widening and (unlike
    /// `moe_ffn_batched_fused_scaled_matches_host`, which only exercises the SCALED accumulate at
    /// multi-row) the plain `moe_accumulate`'s `rows` generalization. Q8_0 gate/up IS also
    /// eligible for the BATCHED path above threshold since the `matmul_mmq_experts` dtype
    /// widening — see `moe_ffn_batched_split_q5k_gate_up_matches_host` for that coverage.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_graph_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        // ne/nff = 32 is the native-block K floor, not an arbitrary small number: the expert
        // id-GEMV decodes whole 32-element sub-blocks (`nsub = in_f/32`), and the recorder's
        // `assert_native_k` refuses anything below or off that grid. See
        // `id_gemv_multi_rejects_sub_block_in_f` (infr-vulkan tests) for what the unguarded
        // dispatch used to return instead.
        let (ne, n_expert, n_used, nff) = (32usize, 4usize, 2usize, 32usize);
        let scale = 1.3f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5; // deterministic weight/act filler
                                                               // Distinct per-expert router rows so top-k selection is unambiguous.
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        let gate: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.017)).collect();
        let up: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.023)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q8_0(&gate), q8_0(&up), q8_0(&down));
        // Host reference uses the dequantized (q8-rounded) weights — same values the GPU reads.
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q8(&gq), deq_q8(&uq), deq_q8(&dq));
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let silu = |z: f32| z / (1.0 + (-z).exp());

        for &rows in &[1usize, 4usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                // router logits → softmax over all experts
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], xr);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], xr);
                            silu(g) * u
                        })
                        .collect();
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] += w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actv);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gb = mk(&gq, BufferUsage::Weights);
            let ub = mk(&uq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 3e-3,
                    "moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Expert parallelism (`crate::ep::ExpertParallelBackend`): a 2-device EXPERT split of the SAME
    /// tiny qwen3moe-shaped `MoeFfn` as `moe_ffn_graph_matches_host` must match BOTH the host
    /// reference AND the 1-device (world=1) EP run — token/logit-identical to reduction-order
    /// tolerance. This is the correctness proof for the EP mechanism (per-band expert compute + the
    /// `moe_ep_band_remap` local-id rewrite + the P2P/all-reduce combine) that runs on TINY synthetic
    /// experts, so it exercises the whole 2-device path WITHOUT the memory pressure of a real 30B MoE
    /// on this box's weak iGPU (which device-losts on the amdgpu submission-memory limit replicating a
    /// 30B model — a hardware limit, not an EP-logic one; see `expert_parallel_matches_single_device`
    /// in infr-llama for the full-model integration test). Covers rows=1/4 (the small-m id-GEMV
    /// expert path) AND rows=16 (the batched dp4a `matmul_mmq_experts` path — its `n_expert_local`
    /// bucket dims + bucket-0 drop of out-of-band assignments). `n_expert=4`, world=2 → 2 experts on
    /// each device.
    #[test]
    #[ignore = "requires TWO Vulkan GPUs: run with --include-ignored on a multi-GPU box"]
    fn ep_matches_single_device_synthetic() {
        let Ok(devs) = VulkanBackend::enumerate_devices_from_env() else {
            return; // no Vulkan — self-skip
        };
        if devs.len() < 2 {
            eprintln!("skip: ep_matches_single_device_synthetic needs >=2 Vulkan devices");
            return;
        }
        let (d0, d1) = (devs[0].index, devs[1].index);
        // ne=256 so rows=16 exercises the batched dp4a `matmul_mmq_experts` path (it needs the wider
        // GEMM shape — the tiny ne=32 of `moe_ffn_graph_matches_host` only ever hits the small-m
        // id-GEMV path); rows=1/4 stay on the small-m path. n_expert=4, world=2 → 2 experts/device.
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 256usize);
        let scale = 1.3f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        // Distinct per-expert router rows so top-k selection is unambiguous (the two devices must
        // agree bit-for-bit on which experts each token routes to — the whole EP premise).
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        let gate: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.017)).collect();
        let up: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.023)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q8_0(&gate), q8_0(&up), q8_0(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q8(&gq), deq_q8(&uq), deq_q8(&dq));
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let silu = |z: f32| z / (1.0 + (-z).exp());

        // Run the tiny MoE through an ExpertParallelBackend over `devices` (world=1 = the identity
        // reference; world=2 = the real expert split). Weights: gate/up/down SHARDED by expert
        // across the ranks, router/x REPLICATED; the MoE `dst` is an internal `sub` (all-reduced),
        // copied to the graph output. Returns the downloaded output.
        let run_ep = |devices: &[usize], x: &[f32], rows: usize| -> Vec<f32> {
            let world = devices.len();
            let mut backends = Vec::with_capacity(world);
            for &idx in devices {
                backends.push(VulkanBackend::new_on(idx).expect("vulkan new_on"));
            }
            let ep = crate::ep::ExpertParallelBackend::new(backends, n_expert, true)
                .expect("ep backend");
            // Shard a stacked expert bank by expert across the ranks (contiguous per-expert slices).
            let shard = |full: &[u8]| -> Box<dyn crate::Buffer> {
                let stride = full.len() / n_expert;
                let nl = n_expert / world;
                let bufs: Vec<Box<dyn crate::Buffer>> = (0..world)
                    .map(|r| {
                        let slice = &full[r * nl * stride..(r + 1) * nl * stride];
                        let b = ep
                            .rank(r)
                            .alloc_uninit(slice.len(), BufferUsage::Weights)
                            .unwrap();
                        ep.rank(r).upload(b.as_ref(), slice).unwrap();
                        b
                    })
                    .collect();
                crate::ep::EpBuffer::wrap(bufs)
            };
            // Replicate a buffer (same bytes on every rank) through the EP backend's alloc.
            let repl = |bytes: &[u8], usage| {
                let b = ep.alloc(bytes.len(), usage).unwrap();
                ep.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
            let sub = g.internal(TensorDesc::new(vec![rows, ne], DType::F32));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
                dst: sub,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None, // set per-rank by the EP lowering
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            g.push(Op::Copy {
                src: sub,
                src_off: 0,
                dst: yi,
                dst_off: 0,
                n: (rows * ne) as u32,
            });
            let xb = repl(bytemuck::cast_slice(x), BufferUsage::Activations);
            let rb = repl(&rq, BufferUsage::Weights);
            let gb = shard(&gq);
            let ub = shard(&uq);
            let db = shard(&dq);
            let yb = ep.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = ep.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            ep.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            ep.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            got
        };

        for &rows in &[1usize, 4usize, 16usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            // Host reference (same as moe_ffn_graph_matches_host): full-expert weighted top-k sum.
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], xr);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], xr);
                            silu(g) * u
                        })
                        .collect();
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] += w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actv);
                    }
                }
            }

            let got1 = run_ep(&[d0], &x, rows); // world=1: single-device (identity band, no reduce)
            let got2 = run_ep(&[d0, d1], &x, rows); // world=2: real 2-device expert split
                                                    // THE EP correctness bar: the 2-device expert split reproduces the single-device MoE
                                                    // (same runner path; world=1 is the identity lowering). Holds for BOTH the small-m
                                                    // id-GEMV path (rows=1/4) and the batched dp4a path (rows=16), so it validates the
                                                    // per-band compute + `moe_ep_band_remap` local-id rewrite + the P2P/all-reduce combine
                                                    // on both. Reduction-order tolerance (the cross-rank sum reassociates the weighted
                                                    // accumulate).
            for i in 0..rows * ne {
                assert!(
                    (got2[i] - got1[i]).abs() < 3e-3,
                    "ep world=2 (expert split across two GPUs) diverged from world=1 (single-device) \
                     rows={rows} at {i}: 2-dev {} vs 1-dev {} — the band remap / expert shard / \
                     all-reduce is not reproducing the single-device weighted expert sum",
                    got2[i],
                    got1[i]
                );
            }
            // For the small-m id-GEMV path (rows <= moe_small_m_threshold()) the underlying expert
            // math is bit-accurate to the host reference too, so pin the ACTUAL values — not just
            // self-consistency. (The batched dp4a path's Q8_0 gate/up carries its own quant slack vs
            // this exact-arithmetic host reference — a property of that kernel, orthogonal to EP,
            // which single-vs-single above already isolates; the batched EP correctness is the
            // world-2-vs-world-1 identity asserted for all rows above.)
            if rows <= 8 {
                for i in 0..rows * ne {
                    assert!(
                        (got1[i] - want[i]).abs() < 3e-3,
                        "ep small-m mismatch vs host rows={rows} at {i}: got {} want {}",
                        got1[i],
                        want[i]
                    );
                }
            }
            eprintln!(
                "[ep-synth] rows={rows}: 2-device expert split matches single-device \
                 (max |Δ| 2dev-vs-1dev={:.2e})",
                (0..rows * ne)
                    .map(|i| (got2[i] - got1[i]).abs())
                    .fold(0f32, f32::max)
            );
        }
        assert_ne!(
            devs[0].name, devs[1].name,
            "the two devices must be distinct physical GPUs"
        );
    }

    /// diffusion-gemma's `MoeFfn` shape: a SEPARATE `router_x` (not `x`), a FUSED `gate_up_exps`
    /// tensor (gate rows first, up rows second per expert), and a per-expert `down_scale`. Same
    /// host-reference-vs-seam structure as `moe_ffn_graph_matches_host`, extended for the three
    /// new fields (see `docs/diffusion-gemma.md` and the `Op::MoeFfn` doc comment).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_fused_scaled_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (32usize, 4usize, 2usize, 32usize);
        let scale = 1.0f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5; // deterministic weight/act filler
        let x: Vec<f32> = (0..ne).map(|i| f(i, 0.11) + 0.05).collect();
        // A DIFFERENT router input row — if the seam mistakenly routed on `x`, the top-k pick
        // (and hence the whole output) would diverge from this host reference.
        let rx: Vec<f32> = (0..ne).map(|i| f(i, 0.19) - 0.03).collect();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // Fused gate|up: per-expert [2*nff, ne], gate rows first, up rows second.
        let gate_up: Vec<f32> = (0..n_expert * 2 * nff * ne).map(|i| f(i, 0.017)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let down_scale: Vec<f32> = (0..n_expert).map(|e| 0.7 + 0.1 * e as f32).collect();
        let (rq, guq, dq) = (q8_0(&router), q8_0(&gate_up), q8_0(&down));
        let (rd, gud, dd) = (deq_q8(&rq), deq_q8(&guq), deq_q8(&dq));
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let silu = |z: f32| z / (1.0 + (-z).exp());
        // router logits over rx (NOT x) → softmax over all experts
        let logits: Vec<f32> = (0..n_expert)
            .map(|e| dot(&rd[e * ne..(e + 1) * ne], &rx))
            .collect();
        let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
        let psum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= psum);
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(n_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
        let mut want = vec![0f32; ne];
        for &e in &idx {
            let gus = e * 2 * nff * ne;
            let ds = e * ne * nff;
            let actv: Vec<f32> = (0..nff)
                .map(|j| {
                    let g = dot(&gud[gus + j * ne..gus + (j + 1) * ne], &x);
                    let u = dot(&gud[gus + (nff + j) * ne..gus + (nff + j + 1) * ne], &x);
                    silu(g) * u
                })
                .collect();
            let w_e = probs[e] / wsum * scale * down_scale[e];
            for (i, wi) in want.iter_mut().enumerate() {
                *wi += w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actv);
            }
        }
        // graph
        let mut g = Graph::new();
        let xi = g.input(TensorDesc::new(vec![1, ne], DType::F32));
        let rxi = g.input(TensorDesc::new(vec![1, ne], DType::F32));
        let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
        let gui = g.weight(TensorDesc::new(vec![n_expert, 2 * nff, ne], DType::Q8_0));
        let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
        let dsi = g.weight(TensorDesc::new(vec![n_expert], DType::F32));
        let yi = g.output(TensorDesc::new(vec![1, ne], DType::F32));
        g.push(Op::MoeFfn {
            x: xi,
            router_x: rxi,
            router: ri,
            gate_exps: gui,
            up_exps: gui, // fused: same handle, never read
            down_exps: di,
            down_scale: Some(dsi),
            fused_gate_up: true,
            dst: yi,
            ne: ne as u32,
            n_expert: n_expert as u32,
            n_used: n_used as u32,
            n_ff_exp: nff as u32,
            scale,
            act: Activation::Silu,
            gating: infr_core::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
            swiglu_clamp: None,
            expert_ids: None,
        });
        let mk = |bytes: &[u8], usage| {
            let b = be_.alloc(bytes.len(), usage).unwrap();
            be_.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
        let rxb = mk(bytemuck::cast_slice(&rx), BufferUsage::Activations);
        let rb = mk(&rq, BufferUsage::Weights);
        let gub = mk(&guq, BufferUsage::Weights);
        let db = mk(&dq, BufferUsage::Weights);
        let dsb = mk(bytemuck::cast_slice(&down_scale), BufferUsage::Weights);
        let yb = be_.alloc(ne * 4, BufferUsage::Activations).unwrap();
        let plan = be_.compile(&g).unwrap();
        let mut bind = Bindings::new();
        bind.bind(xi, xb.as_ref());
        bind.bind(rxi, rxb.as_ref());
        bind.bind(ri, rb.as_ref());
        bind.bind(gui, gub.as_ref());
        bind.bind(di, db.as_ref());
        bind.bind(dsi, dsb.as_ref());
        bind.bind(yi, yb.as_ref());
        be_.execute(plan.as_ref(), &bind).unwrap();
        let mut got = vec![0f32; ne];
        be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
            .unwrap();
        for i in 0..ne {
            assert!(
                (got[i] - want[i]).abs() < 3e-3,
                "moe_ffn (fused+scaled) mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    /// llama4's exact routing shape on the SMALL-m path: sigmoid gating, NO top-k renorm, top-1
    /// (n_used=1), and weight-before-FFN (the routing weight scales the expert INPUT — folded
    /// here into the gate/up GEMV outputs via `moe_weight_scale`, BEFORE the activation; see its
    /// doc). Host reference mirrors the CPU `Op::MoeFfn` interpreter's llama4 arm bit-for-bit
    /// (sigmoid prob, no renorm, weight applied to gate/up pre-activation, not re-applied at the
    /// output). Runs rows=1 (decode) and rows=4 (small prefill chunk) — both
    /// ≤ `moe_small_m_threshold()`, so this exercises ONLY the small-m fast path; the batched-path
    /// counterpart is `moe_ffn_llama4_batched_matches_host`.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_llama4_small_m_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (32usize, 4usize, 1usize, 32usize);
        let scale = 1.3f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        let gate: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.017)).collect();
        let up: Vec<f32> = (0..n_expert * nff * ne).map(|i| f(i, 0.023)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q8_0(&gate), q8_0(&up), q8_0(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q8(&gq), deq_q8(&uq), deq_q8(&dq));
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
        let silu = |z: f32| z / (1.0 + (-z).exp());

        for &rows in &[1usize, 4usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                // top-1 by raw logit — sigmoid is monotone, so this matches top-1-by-sigmoid-prob.
                let e = (0..n_expert)
                    .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap())
                    .unwrap();
                let w = sigmoid(logits[e]) * scale; // no renorm: raw prob × scale
                let gs = e * nff * ne;
                let ds = e * ne * nff;
                let actv: Vec<f32> = (0..nff)
                    .map(|j| {
                        let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], xr);
                        let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], xr);
                        // weight-before: w scales gate/up BEFORE the activation.
                        silu(w * g) * (w * u)
                    })
                    .collect();
                for i in 0..ne {
                    want[t * ne + i] = dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actv);
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Sigmoid,
                norm_w: false,
                weight_before: true,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gb = mk(&gq, BufferUsage::Weights);
            let ub = mk(&uq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 3e-3,
                    "llama4 small-m moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// llama4's exact routing shape on the BATCHED path (large `rows`, id-native mmq experts):
    /// sigmoid gating, no renorm, top-1, weight-before-FFN. Twin of
    /// `moe_ffn_llama4_small_m_matches_host` — same math, but exercises the packed
    /// bucket-scatter → `moe_weight_scale` (post-GEMM, pre-activation) →
    /// `moe_scatter_reduce(prescaled)` sequence the batched path uses instead of the small-m
    /// path's per-slot GEMV + `moe_accumulate(prescaled)`. No shipped llama4 GGUF reaches this
    /// path today (Scout's Q2_K expert banks have no dp4a-mmq kernel — see the `llama4` arch
    /// note), but a future non-codebook-quant llama4 checkpoint would.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_llama4_batched_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 1usize, 256usize);
        let scale = 1.3f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
        let silu = |z: f32| z / (1.0 + (-z).exp());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // 0.3x amplitude — see `moe_ffn_batched_split_q5k_down_matches_host`'s rationale (256-term
        // dots need damping to stay in the int8-activation-quant's well-conditioned range).
        let gate: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.017) * 0.3)
            .collect();
        let up: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.023) * 0.3)
            .collect();
        let down: Vec<f32> = (0..n_expert * ne * nff)
            .map(|i| f(i, 0.029) * 0.3)
            .collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q4k(&gate), q4k(&up), q8_0(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q4k(&gq), deq_q4k(&uq), deq_q8(&dq));

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let e = (0..n_expert)
                    .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap())
                    .unwrap();
                let w = sigmoid(logits[e]) * scale;
                let gs = e * nff * ne;
                let ds = e * ne * nff;
                // The batched path reads int8-quantized activations (quant_q8_gather for gate/up,
                // quant_q8 for the down input) — round-trip both so the reference matches.
                let xrq = deq_q8(&q8_0(xr));
                let actv: Vec<f32> = (0..nff)
                    .map(|j| {
                        let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], &xrq);
                        let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], &xrq);
                        silu(w * g) * (w * u)
                    })
                    .collect();
                let actvq = deq_q8(&q8_0(&actv));
                for i in 0..ne {
                    want[t * ne + i] = dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Sigmoid,
                norm_w: false,
                weight_before: true,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gb = mk(&gq, BufferUsage::Weights);
            let ub = mk(&uq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "llama4 batched moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    // ---- Q5_K helpers (block=256: Q4_K's f16 d + f16 dmin + 12-byte packed 6-bit scale/min plus
    // a 32-byte qh high-bit plane + 128 nibbles = 176 bytes) — mirrors `native_gemm_mmq_q5k`'s
    // decode exactly (qh bit `sub` of byte `l` supplies quant bit 4), so the batched split-gate
    // isolation test below can synthesize a real Q5_K down bank and have the host reference
    // dequant the SAME rounded values the GPU shader reads. Same internal-round-trip-only caveat
    // as the Q4_K helpers above.
    fn q5k(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 256 * 176);
        for blk in x.chunks(256) {
            let mut sc = [0u32; 8];
            let mut mn = [0u32; 8];
            let mut sub_lo = [0f32; 8]; // per-sub-block min
            let mut sub_sc = [0f32; 8]; // per-sub-block (max-min)/31
            for (j, sub) in blk.chunks(32).enumerate() {
                let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                sub_lo[j] = lo;
                sub_sc[j] = ((hi - lo) / 31.0).max(1e-8);
            }
            let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 63.0;
            let dmin = sub_lo
                .iter()
                .cloned()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(1e-8)
                / 63.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let idmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
            for j in 0..8 {
                sc[j] = ((sub_sc[j] * id).round() as i32).clamp(0, 63) as u32;
                mn[j] = ((sub_lo[j].abs() * idmin).round() as i32).clamp(0, 63) as u32;
            }
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            // Pack the 8 (sc,mn) pairs into 12 bytes (get_scale_min_k4 convention).
            let mut scales = [0u8; 12];
            for k in 0..4 {
                scales[k] = (sc[k] & 0x3F) as u8 | (((sc[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[4 + k] = (mn[k] & 0x3F) as u8 | (((mn[4 + k] >> 4) & 0x3) as u8) << 6;
                scales[8 + k] = (sc[4 + k] & 0xF) as u8 | (((mn[4 + k] & 0xF) as u8) << 4);
            }
            out.extend_from_slice(&scales);
            // Quantize to 5 bits against the recovered (d*sc, dmin*mn); low nibble → qs (sub_even
            // low, sub_odd high per 32-byte pair-region), bit 4 → qh bit `sub`.
            let mut q = [[0u8; 32]; 8];
            for j in 0..8 {
                let scale = d * sc[j] as f32;
                let min = dmin * mn[j] as f32;
                let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for (l, &v) in blk[j * 32..j * 32 + 32].iter().enumerate() {
                    q[j][l] = (((v - min) * iscale).round() as i32).clamp(0, 31) as u8;
                }
            }
            let mut qh = [0u8; 32];
            for (j, qj) in q.iter().enumerate() {
                for (l, &qv) in qj.iter().enumerate() {
                    qh[l] |= ((qv >> 4) & 1) << j;
                }
            }
            out.extend_from_slice(&qh);
            let mut qs = [0u8; 128];
            for pair in 0..4 {
                let (lo, hi) = (&q[2 * pair], &q[2 * pair + 1]);
                for l in 0..32 {
                    qs[pair * 32 + l] = (lo[l] & 0xF) | ((hi[l] & 0xF) << 4);
                }
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_q5k(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 176 * 256);
        for blk in bytes.chunks(176) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let scales = &blk[4..16];
            let qh = &blk[16..48];
            let qs = &blk[48..176];
            for j in 0..8u32 {
                let (sc, mn) = if j < 4 {
                    (
                        (scales[j as usize] & 0x3F) as u32,
                        (scales[j as usize + 4] & 0x3F) as u32,
                    )
                } else {
                    let i2 = j as usize - 4;
                    (
                        (scales[j as usize + 4] as u32 & 0xF) | (((scales[i2] as u32) >> 6) << 4),
                        (scales[j as usize + 4] as u32 >> 4)
                            | (((scales[i2 + 4] as u32) >> 6) << 4),
                    )
                };
                let scale = d * sc as f32;
                let min = dmin * mn as f32;
                let pair = (j / 2) as usize;
                let lo = (j % 2) == 0;
                for l in 0..32 {
                    let byte = qs[pair * 32 + l];
                    let nib = if lo { byte & 0xF } else { byte >> 4 };
                    let hb = (qh[l] >> j) & 1;
                    out.push(scale * (nib | (hb << 4)) as f32 - min);
                }
            }
        }
        out
    }

    // ---- Q5_1 helpers (block=32: f16 d + f16 m + 4-byte qh + 16-byte qs) — mirrors
    // `native_gemm_mmq_q5_1`'s decode exactly: min-carrying like Q4_K/Q5_K (`w = d*q + m`, PLUS
    // convention, not Q5_K's stored-negated minus) but with NO superblock sub-scale — one d/m pair
    // per 32-element block, ggml's legacy Q4_1-family layout (gemma-4-26B-A4B-it-GGUF's shipped
    // MoE down-projection format). Same internal-round-trip-only caveat as the Q4_K/Q5_K helpers.
    fn q5_1(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 24);
        for blk in x.chunks(32) {
            let lo = blk.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = blk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let d = ((hi - lo) / 31.0).max(1e-8);
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(lo).to_le_bytes());
            let q: Vec<u8> = blk
                .iter()
                .map(|&v| (((v - lo) * id).round() as i32).clamp(0, 31) as u8)
                .collect();
            let mut qh = 0u32;
            for (l, &qv) in q.iter().enumerate() {
                qh |= ((qv as u32 >> 4) & 1) << l;
            }
            out.extend_from_slice(&qh.to_le_bytes());
            let mut qs = [0u8; 16];
            for l in 0..16 {
                qs[l] = (q[l] & 0xF) | ((q[l + 16] & 0xF) << 4);
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_q5_1(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 24 * 32);
        for blk in bytes.chunks(24) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let m = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
            let qs = &blk[8..24];
            let mut code = [0u8; 32];
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                code[j] = ((qs[j] as u32 & 0xF) | xh0) as u8;
                code[j + 16] = ((qs[j] as u32 >> 4) | xh1) as u8;
            }
            out.extend(code.iter().map(|&c| d * c as f32 + m));
        }
        out
    }

    // ---- Q4_0 helpers (block=18: f16 d + 16-byte qs) — mirrors `native_gemm_mmq_q4_0`'s decode
    // exactly: symmetric (no min, no highbit) `w = d*(q4-8)`, q4 stored as a plain nibble (unlike
    // Q5_0 there is no `qh` 5th-bit field). Same internal-round-trip-only caveat as the other
    // quant helpers here (self-consistent test-data encoder, not a bit-exact ggml quantizer).
    fn q4_0(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 18);
        for blk in x.chunks(32) {
            let amax = blk.iter().cloned().fold(0f32, |m, v| m.max(v.abs()));
            let d = (amax / 7.0).max(1e-8);
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            let q: Vec<u8> = blk
                .iter()
                .map(|&v| ((v * id).round() as i32 + 8).clamp(0, 15) as u8)
                .collect();
            let mut qs = [0u8; 16];
            for l in 0..16 {
                qs[l] = (q[l] & 0xF) | ((q[l + 16] & 0xF) << 4);
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_q4_0(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 18 * 32);
        for blk in bytes.chunks(18) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let qs = &blk[2..18];
            let mut code = [0u8; 32];
            for j in 0..16 {
                code[j] = qs[j] & 0xF;
                code[j + 16] = qs[j] >> 4;
            }
            out.extend(code.iter().map(|&c| d * (c as f32 - 8.0)));
        }
        out
    }

    // ---- Q4_1 helpers (block=20: f16 d + f16 m + 16-byte qs) — mirrors `native_gemm_mmq_q4_1`'s
    // decode exactly: min-carrying like Q5_1 (`w = d*q4 + m`, PLUS convention), no highbit field.
    fn q4_1(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 20);
        for blk in x.chunks(32) {
            let lo = blk.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = blk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let d = ((hi - lo) / 15.0).max(1e-8);
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(lo).to_le_bytes());
            let q: Vec<u8> = blk
                .iter()
                .map(|&v| (((v - lo) * id).round() as i32).clamp(0, 15) as u8)
                .collect();
            let mut qs = [0u8; 16];
            for l in 0..16 {
                qs[l] = (q[l] & 0xF) | ((q[l + 16] & 0xF) << 4);
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_q4_1(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 20 * 32);
        for blk in bytes.chunks(20) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let m = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let qs = &blk[4..20];
            let mut code = [0u8; 32];
            for j in 0..16 {
                code[j] = qs[j] & 0xF;
                code[j + 16] = qs[j] >> 4;
            }
            out.extend(code.iter().map(|&c| d * c as f32 + m));
        }
        out
    }

    /// IQ4_NL/IQ4_XS's 16-entry signed codebook (ggml-common.h `kvalues_iq4nl`) — same table the
    /// GPU shaders' `kv_iq4nl` uses, duplicated here for the host reference/test-data encoder.
    const KVALUES_IQ4NL_TEST: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    fn nearest_iq4nl(v: f32) -> u8 {
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for (i, &k) in KVALUES_IQ4NL_TEST.iter().enumerate() {
            let dd = (v - k as f32).abs();
            if dd < bd {
                bd = dd;
                best = i;
            }
        }
        best as u8
    }

    // ---- IQ4_NL helpers (block=18: f16 d + 16-byte qs, SAME layout as Q4_0) — mirrors
    // `native_gemm_mmq_iq4_nl`'s decode: codebook (not affine), symmetric — the looked-up table
    // value IS the signed dp4a operand, no min/centering.
    fn iq4_nl(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 18);
        for blk in x.chunks(32) {
            let amax = blk.iter().cloned().fold(0f32, |m, v| m.max(v.abs()));
            let d = (amax / 113.0).max(1e-8);
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            let q: Vec<u8> = blk.iter().map(|&v| nearest_iq4nl(v * id)).collect();
            let mut qs = [0u8; 16];
            for l in 0..16 {
                qs[l] = (q[l] & 0xF) | ((q[l + 16] & 0xF) << 4);
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_iq4_nl(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 18 * 32);
        for blk in bytes.chunks(18) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let qs = &blk[2..18];
            let mut code = [0u8; 32];
            for j in 0..16 {
                code[j] = qs[j] & 0xF;
                code[j + 16] = qs[j] >> 4;
            }
            out.extend(
                code.iter()
                    .map(|&c| d * KVALUES_IQ4NL_TEST[c as usize] as f32),
            );
        }
        out
    }

    // ---- IQ4_XS helpers (block=136: f16 d + u16 scales_h + u8 scales_l[4] + u8 qs[128], 8
    // sub-blocks of 32 elements) — mirrors `native_gemm_mmq_iq4_xs`'s decode: codebook +
    // Q4_K-shaped superblock, symmetric (`ls-32` is signed, no separate min).
    fn iq4_xs(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 256 * 136);
        for blk in x.chunks(256) {
            let mut sub_amax = [0f32; 8];
            for (si, sub) in blk.chunks(32).enumerate() {
                sub_amax[si] = sub.iter().cloned().fold(0f32, |m, v| m.max(v.abs()));
            }
            let dmax = sub_amax.iter().cloned().fold(1e-8f32, f32::max);
            let d = dmax / 113.0 / 31.0; // covers the largest sub-block at ls-32==31
            let mut ls = [0u32; 8];
            let mut codes = [0u8; 256];
            for (si, sub) in blk.chunks(32).enumerate() {
                let target_dl = sub_amax[si] / 113.0;
                let ls_signed = (target_dl / d).round().clamp(-32.0, 31.0) as i32;
                ls[si] = (ls_signed + 32) as u32;
                let dl = d * ls_signed as f32;
                let idl = if dl.abs() > 1e-12 { 1.0 / dl } else { 0.0 };
                for (l, &v) in sub.iter().enumerate() {
                    codes[si * 32 + l] = nearest_iq4nl(v * idl);
                }
            }
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            let mut scales_h: u16 = 0;
            for (si, &l) in ls.iter().enumerate() {
                scales_h |= (((l >> 4) & 3) as u16) << (2 * si);
            }
            out.extend_from_slice(&scales_h.to_le_bytes());
            let mut scales_l = [0u8; 4];
            for si in 0..8 {
                scales_l[si / 2] |= ((ls[si] & 0xF) as u8) << (4 * (si % 2));
            }
            out.extend_from_slice(&scales_l);
            let mut qs = [0u8; 128];
            for si in 0..8 {
                for l in 0..16 {
                    qs[si * 16 + l] =
                        (codes[si * 32 + l] & 0xF) | ((codes[si * 32 + l + 16] & 0xF) << 4);
                }
            }
            out.extend_from_slice(&qs);
        }
        out
    }
    fn deq_iq4_xs(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 136 * 256);
        for blk in bytes.chunks(136) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            let scales_l = &blk[4..8];
            let qs = &blk[8..136];
            let mut vals = [0f32; 256];
            for si in 0..8usize {
                let lo = ((scales_l[si / 2] >> (4 * (si % 2))) & 0xF) as u32;
                let hi = ((scales_h >> (2 * si)) & 3) as u32;
                let ls = lo | (hi << 4);
                let dl = d * (ls as i32 - 32) as f32;
                for l in 0..16 {
                    let byte = qs[si * 16 + l];
                    vals[si * 32 + l] = dl * KVALUES_IQ4NL_TEST[(byte & 0xF) as usize] as f32;
                    vals[si * 32 + l + 16] = dl * KVALUES_IQ4NL_TEST[(byte >> 4) as usize] as f32;
                }
            }
            out.extend_from_slice(&vals);
        }
        out
    }

    // ---- Q2_K helpers (block=256: 16-byte scales[16] (low nibble=4-bit scale, high nibble=4-bit
    // min) + 64-byte qs (2-bit codes) + f16 d + f16 dmin = 84 bytes) — mirrors
    // `native_gemm_mmq_q2_k`'s decode exactly: min-carrying (like Q4_K/Q5_K, MINUS convention:
    // `y = d*(sc&0xF)*q2 - dmin*(sc>>4)`) but with 16-element sub-block granularity (one whole
    // byte per sub-block, no 6-bit packing) — Llama-4-Scout's shipped gate/up MoE format. The
    // byte/shift mapping (`base(si) = 32*(si>>3) + 16*(si&1)`, `shift(si) = 2*((si&7)>>1)`) is
    // ground-truthed against infr-gguf's dequant.rs Q2_K reference (and the shader's own doc).
    // Same internal-round-trip-only caveat as the Q4_K/Q5_K helpers (this is a self-consistent
    // encoder for test data, not a bit-exact ggml quantizer — only the DECODER must match the GPU
    // shader exactly).
    fn q2_k(x: &[f32]) -> Vec<u8> {
        let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
        let shift = |si: usize| 2 * ((si & 7) >> 1);
        let mut out = Vec::with_capacity(x.len() / 256 * 84);
        for blk in x.chunks(256) {
            let mut sub_lo = [0f32; 16];
            let mut sub_sc = [0f32; 16];
            for (si, sub) in blk.chunks(16).enumerate() {
                let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                sub_lo[si] = lo;
                sub_sc[si] = ((hi - lo) / 3.0).max(1e-8);
            }
            let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 15.0;
            let dmin = sub_lo
                .iter()
                .cloned()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(1e-8)
                / 15.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let idmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
            let mut scales = [0u8; 16];
            let mut sc = [0u32; 16];
            let mut mn = [0u32; 16];
            for si in 0..16 {
                sc[si] = ((sub_sc[si] * id).round() as i32).clamp(0, 15) as u32;
                mn[si] = ((sub_lo[si].abs() * idmin).round() as i32).clamp(0, 15) as u32;
                scales[si] = (sc[si] | (mn[si] << 4)) as u8;
            }
            out.extend_from_slice(&scales);
            let mut qs = [0u8; 64];
            for si in 0..16 {
                let scale = d * sc[si] as f32;
                let min = dmin * mn[si] as f32;
                let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                let b = base(si);
                let sh = shift(si);
                for l in 0..16 {
                    let v = blk[si * 16 + l];
                    let q = (((v - min) * iscale).round() as i32).clamp(0, 3) as u8;
                    qs[b + l] |= q << sh;
                }
            }
            out.extend_from_slice(&qs);
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
        }
        out
    }
    fn deq_q2_k(bytes: &[u8]) -> Vec<f32> {
        let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
        let shift = |si: usize| 2 * ((si & 7) >> 1);
        let mut out = Vec::with_capacity(bytes.len() / 84 * 256);
        for blk in bytes.chunks(84) {
            let scales = &blk[0..16];
            let qs = &blk[16..80];
            let d = half::f16::from_le_bytes([blk[80], blk[81]]).to_f32();
            let dmin = half::f16::from_le_bytes([blk[82], blk[83]]).to_f32();
            for (si, &scb) in scales.iter().enumerate() {
                let scale = d * (scb & 0xF) as f32;
                let min = dmin * (scb >> 4) as f32;
                let b = base(si);
                let sh = shift(si);
                for l in 0..16 {
                    let q = (qs[b + l] >> sh) & 3;
                    out.push(scale * q as f32 - min);
                }
            }
        }
        out
    }

    // ---- Q3_K helpers (block=256: 32-byte hmask + 64-byte qs (2-bit) + 12-byte packed 6-bit
    // scales + f16 d = 110 bytes) — mirrors `native_gemm_mmq_q3_k`'s decode exactly: SYMMETRIC (no
    // min, unlike Q2_K/Q4_K/Q5_K), `y = d*(sc6-32)*(q3u-4)` with q3u = 2 low bits from `qs` OR'd
    // with a high bit from `hmask` — Llama-4-Scout's shipped down-projection MoE format. Same
    // byte/shift mapping as Q2_K for the low 2 bits; `hmask` bit position cycles
    // `4*(si>>3) + ((si&7)>>1)` within byte `16*(si&1)+l`. Scale packing follows llama.cpp's
    // 4-word bit-interleave (`sc3` in the shader); this encoder inverts it directly (no need for
    // the CPU reference's SIMD-vectorized reconstruction — one scalar per sub-block suffices for
    // test-data synthesis). Same internal-round-trip-only caveat as the other K-quant helpers.
    fn q3_k(x: &[f32]) -> Vec<u8> {
        let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
        let shift = |si: usize| 2 * ((si & 7) >> 1);
        let hbit = |si: usize| 4 * (si >> 3) + ((si & 7) >> 1);
        let mut out = Vec::with_capacity(x.len() / 256 * 110);
        for blk in x.chunks(256) {
            let mut sub_sc = [0f32; 16]; // per-sub-block (max abs)/3, sign-agnostic
            for (si, sub) in blk.chunks(16).enumerate() {
                let amax = sub.iter().cloned().fold(0f32, |m, v| m.max(v.abs()));
                sub_sc[si] = (amax / 3.0).max(1e-8);
            }
            let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 31.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            // signed 6-bit scale in [-32, 31]; keep it positive here (>=1) since a real 3-bit
            // symmetric quantizer never needs a negative overall sub-block scale for this
            // synthetic (non-adversarial) test data.
            let mut s6 = [0i32; 16];
            for si in 0..16 {
                s6[si] = ((sub_sc[si] * id).round() as i32).clamp(1, 31);
            }
            // Pack s6 (stored as sc6+32, i.e. 33..63) into the 12-byte scales_raw using the SAME
            // byte-wise scheme `sc3()` decodes (inverse of that function).
            let mut sr = [0u8; 12];
            for (si, &s) in s6.iter().enumerate() {
                let val = (s + 32) as u8; // 0..63
                let k = si >> 2;
                let bi = si & 3;
                let lo4 = val & 0xF;
                let hi2 = (val >> 4) & 3;
                match k {
                    0 => {
                        sr[bi] |= lo4;
                        sr[8 + bi] |= hi2;
                    }
                    1 => {
                        sr[4 + bi] |= lo4;
                        sr[8 + bi] |= hi2 << 2;
                    }
                    2 => {
                        sr[bi] |= lo4 << 4;
                        sr[8 + bi] |= hi2 << 4;
                    }
                    _ => {
                        sr[4 + bi] |= lo4 << 4;
                        sr[8 + bi] |= hi2 << 6;
                    }
                }
            }
            out.extend_from_slice(&[0u8; 32]); // hmask placeholder, filled below
            let hmask_start = out.len() - 32;
            let mut qs = [0u8; 64];
            for si in 0..16 {
                let scale = d * s6[si] as f32;
                let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                let b = base(si);
                let sh = shift(si);
                let hb = hbit(si);
                for l in 0..16 {
                    let v = blk[si * 16 + l];
                    let q3u = (((v * iscale) + 4.0).round() as i32).clamp(0, 7) as u8;
                    qs[b + l] |= (q3u & 3) << sh;
                    if q3u & 4 != 0 {
                        let hbyte_idx = hmask_start + 16 * (si & 1) + l;
                        out[hbyte_idx] |= 1 << hb;
                    }
                }
            }
            out.extend_from_slice(&qs);
            out.extend_from_slice(&sr);
            out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        }
        out
    }
    fn deq_q3_k(bytes: &[u8]) -> Vec<f32> {
        let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
        let shift = |si: usize| 2 * ((si & 7) >> 1);
        let hbit = |si: usize| 4 * (si >> 3) + ((si & 7) >> 1);
        let sc3 = |sr: &[u8], si: usize| -> i32 {
            let k = si >> 2;
            let bi = si & 3;
            let a = sr[bi] as u32;
            let b = sr[4 + bi] as u32;
            let c = sr[8 + bi] as u32;
            let val = match k {
                0 => (a & 0xF) | ((c & 3) << 4),
                1 => (b & 0xF) | (((c >> 2) & 3) << 4),
                2 => ((a >> 4) & 0xF) | (((c >> 4) & 3) << 4),
                _ => ((b >> 4) & 0xF) | (((c >> 6) & 3) << 4),
            };
            val as i32 - 32
        };
        let mut out = Vec::with_capacity(bytes.len() / 110 * 256);
        for blk in bytes.chunks(110) {
            let hmask = &blk[0..32];
            let qs = &blk[32..96];
            let sr = &blk[96..108];
            let d = half::f16::from_le_bytes([blk[108], blk[109]]).to_f32();
            for si in 0..16usize {
                let s = sc3(sr, si);
                let scale = d * s as f32;
                let b = base(si);
                let sh = shift(si);
                let hb = hbit(si);
                for l in 0..16 {
                    let low2 = (qs[b + l] >> sh) & 3;
                    let high = (hmask[16 * (si & 1) + l] >> hb) & 1;
                    let q3u = low2 | (high << 2);
                    out.push(scale * (q3u as f32 - 4.0));
                }
            }
        }
        out
    }

    /// The `rows>moe_small_m_threshold()` (batched, GPU-resident routing) twin of
    /// `moe_ffn_fused_scaled_matches_host`: diffusion-gemma's ACTUAL production dtypes (fused Q4_K
    /// gate_up, not Q8_0 — exercises `matmul_mmq_experts`'s Q4_K path with `gu_width = 2*nff`) +
    /// Q8_0 down (exercises the new `native_gemm_mmq_q8_0_xp` kernel) + per-expert `down_scale`
    /// (exercises the `moe_bucket_scatter_scaled` dscale-into-bucket_wts path) + a separate
    /// `router_x` (exercises the batched routing prologue reading `router_x`, not `x`) + GELU
    /// (gemma's activation, not qwen3moe's SiLU). `ne=256` is the minimum Q4_K superblock width
    /// (256-element blocks); the real model's ne=2816=11×256. Runs rows=5 (below the small-m
    /// threshold — exercises the id-indexed GEMV fast path's multi-row widening, `Op::MoeFfn`'s
    /// OTHER new-code branch introduced alongside it), rows=9 (above threshold, a ragged non-
    /// 64-aligned row count — exercises the batched GEMM's row-tile overread/clip path, the
    /// scenario rows=5 used to cover before the small-m threshold existed), and rows=256
    /// (diffusion-gemma's canvas width). The host reference's int8-activation rounding models the
    /// BATCHED path's `quant_q8`/`quant_q8_gather` step exactly (rows=9, rows=256); for rows=5 the
    /// small-m path skips that quantization (reads `x`/`actv` at full f32 precision via
    /// `linear_native_id_multi`), so the reference only bounds the small-m result from above (its
    /// error is quantization noise being ADDED to the reference, not present in `got`) — still
    /// covered by the shared tolerance below, which was sized for the two-quantization-layer case.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_fused_scaled_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 32usize);
        let scale = 1.1f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let gelu = |z: f32| 0.5 * z * (1.0 + (0.797_884_6 * (z + 0.044715 * z * z * z)).tanh());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // Fused gate|up: per-expert [2*nff, ne], gate rows first, up rows second.
        let gate_up: Vec<f32> = (0..n_expert * 2 * nff * ne).map(|i| f(i, 0.017)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let down_scale: Vec<f32> = (0..n_expert).map(|e| 0.7 + 0.1 * e as f32).collect();
        let (rq, guq, dq) = (q8_0(&router), q4k(&gate_up), q8_0(&down));
        let (rd, gud, dd) = (deq_q8(&rq), deq_q4k(&guq), deq_q8(&dq));
        let dsb_bytes = bytemuck::cast_slice(&down_scale).to_vec();

        for &rows in &[5usize, 9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            // A DIFFERENT router input row — if the seam mistakenly routed on `x`, the top-k pick
            // (and hence the whole output) would diverge from this host reference.
            let rx: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.19) - 0.03).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let rxr = &rx[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], rxr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                // The batched path's expert GEMMs read int8-quantized activations (`quant_q8`/
                // `quant_q8_gather`, symmetric per-32-block — SAME scheme as the `q8_0` weight
                // helper above), NOT full-precision `x` — unlike the per-token path's
                // `linear_native_id_multi` GEMV, which dequants weights against f32 x directly.
                // Round-trip `xr` (gate/up input) and `actv` (down input) through it so the host
                // reference matches what the GPU shader actually reads.
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gus = e * 2 * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gud[gus + j * ne..gus + (j + 1) * ne], &xrq);
                            let u = dot(&gud[gus + (nff + j) * ne..gus + (nff + j + 1) * ne], &xrq);
                            gelu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale * down_scale[e];
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let rxi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gui = g.weight(TensorDesc::new(vec![n_expert, 2 * nff, ne], DType::Q4K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q8_0));
            let dsi = g.weight(TensorDesc::new(vec![n_expert], DType::F32));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: rxi,
                router: ri,
                gate_exps: gui,
                up_exps: gui, // fused: same handle, never read
                down_exps: di,
                down_scale: Some(dsi),
                fused_gate_up: true,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Gelu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rxb = mk(bytemuck::cast_slice(&rx), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gub = mk(&guq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let dsb = mk(&dsb_bytes, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(rxi, rxb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gui, gub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(dsi, dsb.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            // Tolerance is looser than the single-quantization-layer precedent tests' 3e-3: this
            // path stacks TWO lossy layers the per-token tests never exercise — Q4_K (4-bit, ~10x
            // coarser than Q8_0) gate/up weights AND int8-quantized activations (`quant_q8`/
            // `quant_q8_gather`) on ne=256 (8x the precedent tests' ne=32) — verified in isolation
            // (`matmul_mmq_experts` alone, skewed non-64-aligned expert counts) at ~0.008 max
            // absolute error on comparable data; this loop's ~0.01-0.02 (measured empirically)
            // compounds that through GELU + a second (down) quantization. 2e-2 stays two orders of
            // magnitude below a wrong-dtype/stride/scale/routing bug (which corrupts a whole
            // expert's O(1) contribution, not a couple percent of it).
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched fused-scaled moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Batched split-gate MoeFfn with a Q5_K down bank — qwen35moe's unsloth-dynamic (UD) quant
    /// shape (Q4_K gate/up + Q5_K down on most layers, SiLU, router_x == x, no down_scale).
    /// Exercises the `native_gemm_mmq_q5k_xp` kernel (min-carrying like Q4_K → binds `sact`)
    /// through the full GPU-resident routing pipeline at rows=9 (ragged, non-64-aligned row
    /// tiles) and rows=256. Host reference structure and int8-activation rounding follow
    /// `moe_ffn_batched_fused_scaled_matches_host` (see its tolerance rationale).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_split_q5k_down_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 256usize);
        let scale = 1.0f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let silu = |z: f32| z / (1.0 + (-z).exp());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // 0.3x weight amplitude vs the precedent tests: this shape's dots run over 256 terms at
        // BOTH stages (Q5_K's 256-element superblock forces nff=256, vs the fused test's nff=32),
        // and full-amplitude weights push SwiGLU activations into a dynamic range where the
        // per-32-block int8 activation quant (f16-rounded scale + f16-rounded `sact` block sums —
        // neither modeled by the host reference) costs ~0.05-0.19 absolute; damped, both stages
        // stay well-conditioned and the shared 2e-2 tolerance holds with margin.
        let gate: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.017) * 0.3)
            .collect();
        let up: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.023) * 0.3)
            .collect();
        let down: Vec<f32> = (0..n_expert * ne * nff)
            .map(|i| f(i, 0.029) * 0.3)
            .collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q4k(&gate), q4k(&up), q5k(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q4k(&gq), deq_q4k(&uq), deq_q5k(&dq));

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                // The batched path reads int8-quantized activations (quant_q8_gather for gate/up,
                // quant_q8 for the down input) — round-trip both so the reference matches.
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            silu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q5K));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gb = mk(&gq, BufferUsage::Weights);
            let ub = mk(&uq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched split q5k-down moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Batched split-gate MoeFfn with NON-Q4_K gate/up — the shape this test targets is exactly
    /// what the batched path rejected before the `matmul_mmq_experts` dtype widening (adapter.rs
    /// `mmq_ok`): gate=Q5_K (min-carrying → binds `sact`, same kernel family as Q4_K),
    /// up=Q8_0 (symmetric → NO `sact`), down=Q4_K. Deliberately gives gate and up DIFFERENT
    /// dtypes with DIFFERENT `sact` requirements (unlike `moe_ffn_batched_split_q5k_down_matches_host`,
    /// where gate/up share one dtype) — this is the case that would silently corrupt output (or
    /// hit a binding-count mismatch) if `gate_needs_sact`/`up_needs_sact` weren't threaded
    /// per-role and independently from `down_needs_sact`. Host reference structure and
    /// int8-activation rounding follow `moe_ffn_batched_fused_scaled_matches_host`.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_split_q5k_gate_up_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 256usize);
        let scale = 1.0f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let silu = |z: f32| z / (1.0 + (-z).exp());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // 0.3x amplitude — same rationale as `moe_ffn_batched_split_q5k_down_matches_host`
        // (256-term dots at both stages need damping to stay in the int8-activation-quant's
        // well-conditioned range).
        let gate: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.017) * 0.3)
            .collect();
        let up: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.023) * 0.3)
            .collect();
        let down: Vec<f32> = (0..n_expert * ne * nff)
            .map(|i| f(i, 0.029) * 0.3)
            .collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q5k(&gate), q8_0(&up), q4k(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q5k(&gq), deq_q8(&uq), deq_q4k(&dq));

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            silu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q5K));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q8_0));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q4K));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gb = mk(&gq, BufferUsage::Weights);
            let ub = mk(&uq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched split q5k/q8_0 gate-up moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Batched fused-gate_up MoeFfn with a Q5_1 down bank — gemma-4-26B-A4B-it-GGUF's shipped
    /// shape (Q4_K fused gate_up + Q5_1 down on 29/30 layers, GELU, per-expert `down_scale`,
    /// separate `router_x`). Same structure as `moe_ffn_batched_fused_scaled_matches_host` (which
    /// covers this exact shape with a Q8_0 down bank) with the down dtype swapped to Q5_1 —
    /// exercises the new `native_gemm_mmq_q5_1_xp` kernel's min-term (`w = d*q + m`, PLUS
    /// convention, no superblock) through the full GPU-resident batched routing pipeline at
    /// rows=9 (ragged, non-64-aligned row tiles) and rows=256.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_fused_q5_1_down_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 32usize);
        let scale = 1.1f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let gelu = |z: f32| 0.5 * z * (1.0 + (0.797_884_6 * (z + 0.044715 * z * z * z)).tanh());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        // Fused gate|up: per-expert [2*nff, ne], gate rows first, up rows second.
        let gate_up: Vec<f32> = (0..n_expert * 2 * nff * ne).map(|i| f(i, 0.017)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let down_scale: Vec<f32> = (0..n_expert).map(|e| 0.7 + 0.1 * e as f32).collect();
        let (rq, guq, dq) = (q8_0(&router), q4k(&gate_up), q5_1(&down));
        let (rd, gud, dd) = (deq_q8(&rq), deq_q4k(&guq), deq_q5_1(&dq));
        let dsb_bytes = bytemuck::cast_slice(&down_scale).to_vec();

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let rx: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.19) - 0.03).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let rxr = &rx[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], rxr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gus = e * 2 * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gud[gus + j * ne..gus + (j + 1) * ne], &xrq);
                            let u = dot(&gud[gus + (nff + j) * ne..gus + (nff + j + 1) * ne], &xrq);
                            gelu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale * down_scale[e];
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let rxi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gui = g.weight(TensorDesc::new(vec![n_expert, 2 * nff, ne], DType::Q4K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q5_1));
            let dsi = g.weight(TensorDesc::new(vec![n_expert], DType::F32));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: rxi,
                router: ri,
                gate_exps: gui,
                up_exps: gui, // fused: same handle, never read
                down_exps: di,
                down_scale: Some(dsi),
                fused_gate_up: true,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Gelu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rxb = mk(bytemuck::cast_slice(&rx), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gub = mk(&guq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let dsb = mk(&dsb_bytes, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(rxi, rxb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gui, gub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(dsi, dsb.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            // Same tolerance rationale as `moe_ffn_batched_fused_scaled_matches_host`: two lossy
            // layers (Q4_K gate/up + int8-quantized activations) stacked through GELU + a second
            // (Q5_1) down quantization.
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched fused q5_1-down moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Shared body for the new-format batched-fused-down parity tests below: same shape as
    /// `moe_ffn_batched_fused_q5_1_down_matches_host` (Q4_K fused gate_up, GELU, per-expert
    /// `down_scale`, separate `router_x`) with the down dtype/quantizer swapped in — proves each
    /// new mmq kernel (Q4_0/Q4_1/IQ4_NL/IQ4_XS) through the full GPU-resident batched routing
    /// pipeline at rows=9 (ragged, non-64-aligned row tiles) and rows=256.
    fn fused_down_parity_check(
        down_dtype: DType,
        quant_down: fn(&[f32]) -> Vec<u8>,
        dequant_down: fn(&[u8]) -> Vec<f32>,
        label: &str,
    ) {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 32usize);
        let scale = 1.1f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let gelu = |z: f32| 0.5 * z * (1.0 + (0.797_884_6 * (z + 0.044715 * z * z * z)).tanh());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        let gate_up: Vec<f32> = (0..n_expert * 2 * nff * ne).map(|i| f(i, 0.017)).collect();
        let down: Vec<f32> = (0..n_expert * ne * nff).map(|i| f(i, 0.029)).collect();
        let down_scale: Vec<f32> = (0..n_expert).map(|e| 0.7 + 0.1 * e as f32).collect();
        let (rq, guq, dq) = (q8_0(&router), q4k(&gate_up), quant_down(&down));
        let (rd, gud, dd) = (deq_q8(&rq), deq_q4k(&guq), dequant_down(&dq));
        let dsb_bytes = bytemuck::cast_slice(&down_scale).to_vec();

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let rx: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.19) - 0.03).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let rxr = &rx[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], rxr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gus = e * 2 * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gud[gus + j * ne..gus + (j + 1) * ne], &xrq);
                            let u = dot(&gud[gus + (nff + j) * ne..gus + (nff + j + 1) * ne], &xrq);
                            gelu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale * down_scale[e];
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let rxi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gui = g.weight(TensorDesc::new(vec![n_expert, 2 * nff, ne], DType::Q4K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], down_dtype));
            let dsi = g.weight(TensorDesc::new(vec![n_expert], DType::F32));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: rxi,
                router: ri,
                gate_exps: gui,
                up_exps: gui, // fused: same handle, never read
                down_exps: di,
                down_scale: Some(dsi),
                fused_gate_up: true,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Gelu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rxb = mk(bytemuck::cast_slice(&rx), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gub = mk(&guq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let dsb = mk(&dsb_bytes, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(rxi, rxb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gui, gub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(dsi, dsb.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched fused {label}-down moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// Q4_0 as the down bank: symmetric trivial family member — exercises
    /// `native_gemm_mmq_q4_0_xp`/`_xp32` and confirms `mmq_ok`/`down_needs_sact` (false) handle it.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_fused_q4_0_down_matches_host() {
        fused_down_parity_check(DType::Q4_0, q4_0, deq_q4_0, "q4_0");
    }

    /// Q4_1 as the down bank: min-carrying (Q5_1's pattern minus the highbit) — exercises
    /// `native_gemm_mmq_q4_1_xp`/`_xp32`'s `sact` binding and confirms `down_needs_sact` (true).
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_fused_q4_1_down_matches_host() {
        fused_down_parity_check(DType::Q4_1, q4_1, deq_q4_1, "q4_1");
    }

    /// IQ4_NL as the down bank: codebook, symmetric — exercises
    /// `native_gemm_mmq_iq4_nl_xp`/`_xp32`'s `kv_iq4nl` lookup-then-dp4a path.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_fused_iq4_nl_down_matches_host() {
        fused_down_parity_check(DType::Iq4Nl, iq4_nl, deq_iq4_nl, "iq4_nl");
    }

    /// IQ4_XS as the down bank: codebook + Q4_K-shaped superblock, symmetric — exercises
    /// `native_gemm_mmq_iq4_xs_xp`/`_xp32`'s sub-block `ls-32` scale + `kv_iq4nl` lookup. Real-model
    /// relevance: unsloth's Qwen3.6-35B-A3B-UD-IQ3_S GGUF mixes IQ4_XS into most gate/up banks.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_fused_iq4_xs_down_matches_host() {
        fused_down_parity_check(DType::Iq4Xs, iq4_xs, deq_iq4_xs, "iq4_xs");
    }

    /// Batched split-gate MoeFfn in Llama-4-Scout's actual shipped shape: gate=Q2_K, up=Q2_K
    /// (min-carrying, but its 16-elem sub-block is HALF the activation's 32-elem `sact`
    /// granularity, so it self-computes its own Σx in-shader instead of binding `sact` — see
    /// `native_gemm_mmq_q2_k.comp`'s doc; exercises that kernel for BOTH gate/up roles at once),
    /// down=Q3_K (symmetric, no min at all, no `sact` — exercises the new
    /// `native_gemm_mmq_q3_k_xp` kernel and confirms `mmq_ok`/`down_needs_sact` handle it
    /// correctly). SiLU, split gate/up (llama4's `dual_moe()` is false), separate `router_x`
    /// unused (bound to `x`, matching qwen3moe/llama4's convention). Host reference structure
    /// mirrors `moe_ffn_batched_split_q5k_gate_up_matches_host`; 0.3x amplitude damping for the
    /// same int8-activation-quant conditioning reason. An earlier version of this kernel had a
    /// real bug here, caught by an isolated single-expert diagnostic during development: Q2_K's
    /// min term tried to reuse the shared 32-wide `sact` buffer, which silently mixed in the
    /// wrong half's activation sum — fixed by computing the 16-wide sum in-shader from the
    /// already-staged int8 codes instead.
    #[test]
    #[ignore = "requires a Vulkan-capable GPU"]
    fn moe_ffn_batched_split_q2_k_q3_k_scout_shape_matches_host() {
        let Ok(be_) = VulkanBackend::new() else {
            return; // no GPU — self-skip
        };
        let (ne, n_expert, n_used, nff) = (256usize, 4usize, 2usize, 256usize);
        let scale = 1.0f32;
        let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.5;
        let silu = |z: f32| z / (1.0 + (-z).exp());
        let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
        let router: Vec<f32> = (0..n_expert * ne)
            .map(|i| f(i, 0.037) + (i / ne) as f32 * 0.15)
            .collect();
        let gate: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.017) * 0.3)
            .collect();
        let up: Vec<f32> = (0..n_expert * nff * ne)
            .map(|i| f(i, 0.023) * 0.3)
            .collect();
        let down: Vec<f32> = (0..n_expert * ne * nff)
            .map(|i| f(i, 0.029) * 0.3)
            .collect();
        let (rq, gq, uq, dq) = (q8_0(&router), q2_k(&gate), q2_k(&up), q3_k(&down));
        let (rd, gd, ud, dd) = (deq_q8(&rq), deq_q2_k(&gq), deq_q2_k(&uq), deq_q3_k(&dq));

        for &rows in &[9usize, 256usize] {
            let x: Vec<f32> = (0..rows * ne).map(|i| f(i, 0.11) + 0.05).collect();
            let mut want = vec![0f32; rows * ne];
            for t in 0..rows {
                let xr = &x[t * ne..(t + 1) * ne];
                let logits: Vec<f32> = (0..n_expert)
                    .map(|e| dot(&rd[e * ne..(e + 1) * ne], xr))
                    .collect();
                let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                let psum: f32 = probs.iter().sum();
                probs.iter_mut().for_each(|p| *p /= psum);
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                idx.truncate(n_used);
                let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                let xrq = deq_q8(&q8_0(xr));
                for &e in &idx {
                    let gs = e * nff * ne;
                    let ds = e * ne * nff;
                    let actv: Vec<f32> = (0..nff)
                        .map(|j| {
                            let g = dot(&gd[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            let u = dot(&ud[gs + j * ne..gs + (j + 1) * ne], &xrq);
                            silu(g) * u
                        })
                        .collect();
                    let actvq = deq_q8(&q8_0(&actv));
                    let w_e = probs[e] / wsum * scale;
                    for i in 0..ne {
                        want[t * ne + i] +=
                            w_e * dot(&dd[ds + i * nff..ds + (i + 1) * nff], &actvq);
                    }
                }
            }
            // graph
            let mut g = Graph::new();
            let xi = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
            let ri = g.weight(TensorDesc::new(vec![n_expert, ne], DType::Q8_0));
            let gi = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q2K));
            let ui = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q2K));
            let di = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q3K));
            let yi = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
            g.push(Op::MoeFfn {
                x: xi,
                router_x: xi,
                router: ri,
                gate_exps: gi,
                up_exps: ui,
                down_exps: di,
                down_scale: None,
                fused_gate_up: false,
                dst: yi,
                ne: ne as u32,
                n_expert: n_expert as u32,
                n_used: n_used as u32,
                n_ff_exp: nff as u32,
                scale,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            let mk = |bytes: &[u8], usage| {
                let b = be_.alloc(bytes.len(), usage).unwrap();
                be_.upload(b.as_ref(), bytes).unwrap();
                b
            };
            let xb = mk(bytemuck::cast_slice(&x), BufferUsage::Activations);
            let rb = mk(&rq, BufferUsage::Weights);
            let gb = mk(&gq, BufferUsage::Weights);
            let ub = mk(&uq, BufferUsage::Weights);
            let db = mk(&dq, BufferUsage::Weights);
            let yb = be_.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
            let plan = be_.compile(&g).unwrap();
            let mut bind = Bindings::new();
            bind.bind(xi, xb.as_ref());
            bind.bind(ri, rb.as_ref());
            bind.bind(gi, gb.as_ref());
            bind.bind(ui, ub.as_ref());
            bind.bind(di, db.as_ref());
            bind.bind(yi, yb.as_ref());
            be_.execute(plan.as_ref(), &bind).unwrap();
            let mut got = vec![0f32; rows * ne];
            be_.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut got))
                .unwrap();
            for i in 0..rows * ne {
                assert!(
                    (got[i] - want[i]).abs() < 2e-2,
                    "batched split q2_k/q3_k (Scout shape) moe_ffn mismatch rows={rows} at {i}: got {} want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}
