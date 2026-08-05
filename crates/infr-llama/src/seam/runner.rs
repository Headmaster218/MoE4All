//! Backend-generic dense decode runner: builds the agnostic decode [`Graph`] per token/batch and
//! drives it through a [`Backend`]. This is the giant `generate_dense_backend` — the single
//! forward every entry point in `super` (CPU/Vulkan/Metal, one-shot/session/verify/denoise) funnels
//! through. Pure-move split of `seam.rs` — see `super` for the module overview.
use super::sc::{
    build_sc_embt, diffusion_self_cond, DenoiseCache, DenoiseReq, EbReduced, SelfCondWeights,
};
use super::weights::{
    AttnW, DeltaW, FfnW, LayerW, MixerW, MlaW, MoeSharedW, SeamKv, SeamWeights, SessionStable,
};
use super::{common_prefix_len, e2b_ipl_rows, kv_fmt_bytes, kv_forces_static, BindWeight, WBytes};
use crate::seam::TokenEmbd;
use crate::{Config, EngineConfig, GenStats, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::WeightSource;
use infr_gguf::Gguf;

/// Combined gate+up FFN upload decision (one GEMV/GEMM + GatedActFused instead of two Linears +
/// GatedAct). Requires the backend to opt in (`Capabilities::combined_gu` — Vulkan; the CPU keeps
/// zero-copy separate tensors) AND every dense layer's gate/up to share a dtype (the concat is
/// one [2*nff, ne] tensor). Extracted from the runner's inline form so the seam's dense
/// layer-streaming plan enumerates blocks EXACTLY as `wload` uploads them (a drift between the
/// two would register a block whose bytes don't match the graph's handle — caught loudly at pool
/// registration, but the shared decision removes the drift class entirely). NOT gated on
/// `c.moe.is_none()`: a pure MoE arch has no `ffn_gate.weight` at all, so the `.all()` is false
/// for it regardless; see the original comment block at the call site for the DiffusionGemma
/// rationale.
pub(crate) fn fuse_gu_decision(combined_gu: bool, g: &Gguf, c: &Config) -> bool {
    combined_gu
        && (0..c.n_layer).all(|l| {
            let dt = |s: &str| {
                let name = format!("blk.{l}.{s}");
                g.tensors().iter().find(|t| t.name == name).map(|t| t.dtype)
            };
            dt("ffn_gate.weight").is_some() && dt("ffn_gate.weight") == dt("ffn_up.weight")
        })
}

/// Combined QKV upload decision (one wide prefill GEMM; decode keeps three offset GEMVs via
/// `Op::Linear.w_off`). Needs every layer to own all three projections in ONE native-supported
/// dtype, uniform dims, and a backend that opted into combined weights — mixed-precision GGUFs
/// (llama.cpp's Q4_K_M bumps attn_v to Q6_K on alternating layers) fail the uniform-dtype gate
/// and keep the split form. `kernels.qkv_fuse` (`INFR_NO_QKV_FUSE`, inverted) forces the split form
/// for A/B. Shared with the seam's dense-streaming plan — see [`fuse_gu_decision`].
pub(crate) fn fuse_qkv_decision(
    combined_gu: bool,
    g: &Gguf,
    c: &Config,
    ec: &EngineConfig,
) -> bool {
    combined_gu
        && ec.kernels.qkv_fuse
        && (0..c.n_layer).all(|l| {
            let dt = |s: &str| {
                let name = format!("blk.{l}.{s}");
                g.tensors().iter().find(|t| t.name == name).map(|t| t.dtype)
            };
            let q = dt("attn_q.weight");
            q.is_some()
                && q == dt("attn_k.weight")
                && q == dt("attn_v.weight")
                && q.is_some_and(|d| {
                    infr_vulkan::linear::native_dense_supported(d) && d != DType::F16
                })
                && c.layer_head_dim(l) == c.head_dim
                && c.layer_n_kv(l) == c.n_kv
                && c.has_own_kv(l)
        })
}

/// Range-check externally-supplied token ids against the vocabulary BEFORE they index the
/// embedding table (`tok as usize * n_embd`). An out-of-vocab id would otherwise slice the table
/// out of bounds and panic; surface a clean error instead. Only caller-supplied ids (prompt,
/// denoise canvas) can be arbitrary — generated ids are always sampling/grammar-bounded to
/// `< vocab`, so the decode loop never needs this.
fn validate_token_ids(ids: &[u32], vocab: usize) -> AResult<()> {
    if let Some(&bad) = ids.iter().find(|&&t| t as usize >= vocab) {
        return Err(anyhow!(
            "token id {bad} out of range for vocab size {vocab}"
        ));
    }
    Ok(())
}

/// Which prompt/generated tokens' KV rows are materialized (resident) after a generation turn —
/// the token list recorded in `SeamKv::cached` for the next turn's prefix diff.
///
/// `cur` is the fed-token stream (`prompt` followed by the generated tokens; `cur[pos]` is the
/// token fed at absolute sequence position `pos`). `last_written` is the highest position whose
/// KV row was actually WRITTEN by a kept token this turn — `None` when nothing was fed (an empty
/// prompt, or a `max_new == 0` single-token prompt whose only token is the un-fed frontier).
///
/// Excluded by construction: the final sampled token (pushed to `out` but never fed back, so its
/// KV row does not exist), the `max_new == 0` frontier token (the decode loop breaks before
/// feeding it), and any grammar-forced tokens queued past the frontier that a break left un-fed.
/// Recording any of those would make the next turn's prefix reuse (`common_prefix_len == plen`,
/// `start = plen`) attend a stale/zero KV row and corrupt the output. Kept identical to the old
/// `prompt ++ out[..out.len()-1]` teardown on every run where that was already correct.
fn resident_after_gen(cur: &[u32], last_written: Option<usize>) -> Vec<u32> {
    match last_written {
        Some(p) => cur[..(p + 1).min(cur.len())].to_vec(),
        None => Vec::new(),
    }
}

/// Bind the per-layer IO + weights shared by EVERY decode/prefill/verify/denoise execution: the
/// gemma4 `rope_freqs` Input (when present), each layer's K/V cache pair, and the flat weight list
/// (declaration == upload order). Extracted so the four bind sites (denoise, verify, record-once,
/// per-token) can never drift — a forgotten bind here (e.g. `rope_freqs`) is a live unbound-Input
/// panic at execute. The caller still binds the per-site pieces (hidden/tok_ids, positions,
/// logits, sampling/h-tap/ipl/SC handles).
#[allow(clippy::type_complexity)]
fn bind_layer_io<'a>(
    b: &mut Bindings<'a>,
    h: &DecodeHandles,
    n_layer: usize,
    rf_buf: &'a Option<(Box<dyn Buffer>, usize)>,
    kbufs: &'a [Box<dyn Buffer>],
    vbufs: &'a [Box<dyn Buffer>],
    wbufs: &'a [Box<dyn Buffer>],
) {
    if let (Some(rid), Some((rb, _))) = (h.rope_freqs, rf_buf) {
        b.bind(rid, rb.as_ref());
    }
    for l in 0..n_layer {
        b.bind(h.k_cache[l], kbufs[l].as_ref());
        b.bind(h.v_cache[l], vbufs[l].as_ref());
    }
    for (i, wid) in h.weights.iter().enumerate() {
        b.bind(*wid, wbufs[i].as_ref());
    }
}

/// Compute the [`SessionStable`] derivations — the per-layer tensor scans + real `load_tensor_dequant`s
/// that are pure in `(backend caps, gguf, config, env)`. Run ONCE at cold session init (the result
/// is stashed in `SeamKv` and reused via `Arc` on warm calls / forks) instead of every request.
fn session_stable(
    be: &dyn Backend,
    g: &Gguf,
    c: &Config,
    ec: &EngineConfig,
) -> AResult<SessionStable> {
    // Capabilities are a per-backend invariant; query ONCE (each call clones an owned struct with a
    // heap `String name`) and read fields off the cached copy — this function otherwise queried the
    // backend 8× per build.
    let caps = be.capabilities();
    // Per-layer presence of an explicit V projection. gemma4 full-attention layers omit it (V = the
    // raw K projection); every layer of every other model has one.
    let has_wv: Vec<bool> = (0..c.n_layer)
        .map(|l| {
            g.tensors()
                .iter()
                .any(|t| t.name == format!("blk.{l}.attn_v.weight"))
        })
        .collect();
    // gemma4 per-layer output scale (`layer_output_scale.weight`, a single scalar multiplying the
    // layer output before the next layer). Read host-side; applied as an `Op::Scale`. diffusion-
    // gemma ships TWO per-layer scalars (encoder for the prompt, decoder for the canvas); Phase 1
    // is the encoder-only causal prefill, so it reads `enc_layer_output_scale` — the decoder
    // scalar is unused until the canvas denoise graph (Phase 2+).
    let out_scale_name = if c.diffusion_gemma {
        "enc_layer_output_scale"
    } else {
        "layer_output_scale"
    };
    let out_scale: Vec<Option<f32>> = (0..c.n_layer)
        .map(|l| {
            let name = format!("blk.{l}.{out_scale_name}.weight");
            if g.tensors().iter().any(|t| t.name == name) {
                crate::load_tensor_dequant(g, &name)
                    .ok()
                    .and_then(|(v, _)| v.first().copied())
            } else {
                None
            }
        })
        .collect();
    // diffusion-gemma's DECODER per-layer scalar (`layer_output_scale`, the canvas-denoise twin
    // of `out_scale`'s encoder-named array above) — read unconditionally alongside it (both are
    // tiny [1]-tensors, negligible host cost) so the denoise graph (`build`'s `denoise` flag) can
    // select it without re-deriving the name. `None`/empty for every non-diffusion-gemma model
    // (never read there).
    let dec_out_scale: Vec<Option<f32>> = (0..c.n_layer)
        .map(|l| {
            let name = format!("blk.{l}.layer_output_scale.weight");
            if g.tensors().iter().any(|t| t.name == name) {
                crate::load_tensor_dequant(g, &name)
                    .ok()
                    .and_then(|(v, _)| v.first().copied())
            } else {
                None
            }
        })
        .collect();
    // gemma4 proportional-RoPE frequency divisors (`rope_freqs.weight`, `[rope_dim/2]`): applied on
    // full-attention layers only (SWA layers use plain RoPE). Bound as a per-step f32 Input.
    let rope_freqs: Option<Vec<f32>> =
        if c.gemma4 && g.tensors().iter().any(|t| t.name == "rope_freqs.weight") {
            Some(crate::load_tensor_dequant(g, "rope_freqs.weight").map(|(v, _)| v)?)
        } else {
            None
        };
    // Combined gate+up FFN weights (one GEMV/GEMM + GatedActFused instead of two Linears +
    // GatedAct — the bespoke path's fused-gu shape, ~1 dispatch/layer off the decode hot loop).
    // Requires the backend to opt in (Vulkan; the CPU keeps zero-copy separate tensors) AND every
    // dense layer's gate/up to share a dtype (the concat is one [2*nff, ne] tensor). The decision
    // is global so the upload order and `build`'s handle declarations always agree. NOT gated on
    // `c.moe.is_none()`: a pure MoE arch (qwen3moe) has no `ffn_gate.weight`/`ffn_up.weight` tensors
    // at all, so the `.all()` below is false for it regardless; diffusion-gemma DOES carry both (its
    // dense "shared expert" branch, separate from the MoE bank) and its dense n_ff=2112 out_f clears
    // neither warp-tile gate on its own (%256 nor %128) so it fell to the slower mmq path — fused
    // out_f=2*2112=4224 clears %128. See `FfnW::DiffusionMoe::fused_gu`.
    let fuse_gu = fuse_gu_decision(caps.combined_gu, g, c);
    // Combined QKV: one [qrow+2·kvrow, ne] weight → prefill runs ONE wide GEMM (the separate
    // q/k/v GEMMs are narrow-n and underfill a big GPU — the pp512 sweep's dominant cost), and
    // decode keeps three offset GEMVs into the same buffer (`Op::Linear.w_off`), so its dispatch
    // count is unchanged. Needs every layer to own all three projections in ONE native-supported
    // dtype (gemma4's V-less full layers keep the split form), uniform dims (the offsets are
    // baked once), and a backend that opted into combined weights.
    // NOTE: llama.cpp's Q4_K_M etc. bump attn_v to Q6_K on alternating layers, so mixed-precision
    // GGUFs (e.g. Qwen3-8B Q4_K_M: v = 18×Q4K + 18×Q6K) fail the uniform-dtype gate and keep the
    // split form. INFR_NO_QKV_FUSE forces the split form for A/B (default unset = fuse; the split
    // form is bit-identical — same dots, same fixed-order sums).
    let fuse_qkv = fuse_qkv_decision(caps.combined_gu, g, c, ec);
    // Batched-prefill eligibility for MoE: every layer's expert banks must have a dp4a-mmq kernel
    // (`MOE_MMQ_DTYPES`) — see the decode_start call site's comment for the full rationale.
    let moe_mmq_ok = |d: Option<DType>| d.is_some_and(infr_core::tensor::moe_mmq_ok);
    let moe_batched_ok = c.moe.is_some() && {
        let dt = |n: String| g.tensors().iter().find(|t| t.name == n).map(|t| t.dtype);
        if c.dual_moe() {
            (0..c.n_layer).all(|l| {
                moe_mmq_ok(dt(format!("blk.{l}.ffn_gate_up_exps.weight")))
                    && moe_mmq_ok(dt(format!("blk.{l}.ffn_down_exps.weight")))
            })
        } else {
            (0..c.n_layer).all(|l| {
                moe_mmq_ok(dt(format!("blk.{l}.ffn_gate_exps.weight")))
                    && moe_mmq_ok(dt(format!("blk.{l}.ffn_up_exps.weight")))
                    && moe_mmq_ok(dt(format!("blk.{l}.ffn_down_exps.weight")))
            })
        }
    };
    Ok(SessionStable {
        has_wv,
        out_scale,
        dec_out_scale,
        rope_freqs,
        fuse_gu,
        fuse_qkv,
        moe_batched_ok,
    })
}

/// Handles into one freshly-built decode graph that the driver re-binds each step.
pub(super) struct DecodeHandles {
    hidden: TensorId,
    positions: TensorId,
    rope_freqs: Option<TensorId>, // gemma4 proportional-RoPE divisors (full-attention layers)
    // gemma4 E2B host-gathered per-layer TOKEN embedding rows `[n_layer*npl]` — the graph Input
    // the driver binds `ipl_buf` to; the GPU prologue turns this into the layer loop's actual
    // per-layer input vector (see `per_layer_inp` inside `build`).
    pl_tok_in: Option<TensorId>,
    // Phase-B perf: DiffusionGemma in-graph self-conditioning inputs/weight — `Some` only when
    // `build` was called with `gpu_sc: Some(true)` (see `build`'s doc). `sc_logits` is the
    // per-step Input (host-premultiplied previous canvas logits); `sc_embt` is the one-time
    // device weight bound from `SeamKv::sc_embt`, NOT from the ordinary `weights` upload loop.
    sc_logits: Option<TensorId>,
    sc_embt: Option<TensorId>,
    // Vulkan-only perf (`dyn_sc_scale`'s doc on `build`): the 1-element Input the SC subgraph's
    // `Op::Softmax` reads its scale from instead of a baked constant. `Some` only when `build` was
    // called with `gpu_sc: Some(true)` AND `dyn_sc_scale: true`; `None` otherwise (Metal's SC
    // subgraph, and every non-SC build) — mirrors `sc_logits`/`sc_embt`'s gating.
    temp_inv: Option<TensorId>,
    // `None` for headless builds (`logits_rows == 0` — the batched-prefill chunks, whose logits
    // nothing consumes); `Some` everywhere else.
    logits: Option<TensorId>,
    // MTP Phase 1 (issue #33, docs/mtp.md): the LM-head INPUT — the same rows `logits` was
    // computed from, one op earlier (post-`output_norm`, pre-`w_lm`). `Some` only when `build`
    // was called with `h_tap: true`; `None` for every ordinary caller (no extra op, no extra
    // download). This is the primitive Phase 2's MTP head needs (`h_p` in `docs/mtp.md`'s forward
    // pseudocode) — Phase 1 only exposes the tap, no head graph reads it yet.
    h_out: Option<TensorId>,
    // GPU embed gather (`use_ids` on `build`): the I32 token-id Input the driver binds instead
    // of uploading embedded f32 rows into `hidden` (which is then an Internal fed by the
    // in-graph `Op::EmbedGather`). `None` on host-embed builds.
    tok_ids: Option<TensorId>,
    // GPU stochastic sampling (`gpu_sample` on `build`): the 1-float Input holding this step's
    // host-drawn uniform (see `Op::Sample`). `None` on greedy/host-sampled builds.
    u_in: Option<TensorId>,
    // GPU-resident greedy sampling (`gpu_argmax` on `build`): the `Op::Argmax` output — one u32
    // token id (as an f32-slot bit-pattern). The decode loop reads THIS back (4 bytes) instead of
    // the `[vocab]` logits. `None` when the build didn't append the op (sampling temp > 0, a
    // grammar constraint, or a multi-row logits build).
    tok_id: Option<TensorId>,
    k_cache: Vec<TensorId>,
    v_cache: Vec<TensorId>,
    weights: Vec<TensorId>, // flat, in declaration == upload order
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_backend(
    be: &dyn Backend,
    bind_weight: &BindWeight,
    g: &Gguf,
    cfg: &Config,
    // The ENGINE configuration this whole forward reads its knobs from — `kv.*`, `spec.*`,
    // `sampling.*`, `device.ubatch*`, `kernels.{qkv_fuse,gated_rmsnorm}` and the `prof.*`
    // diagnostics. BORROWED for the call (R6): every value it feeds is hoisted into a local
    // ABOVE the decode loop, so a token costs no field walk and certainly no clone.
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    mut on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    mut constraint: Option<&mut crate::grammar::Constraint>,
    verify: Option<&mut Vec<f32>>,
    // GPU-resident MTP verify accept (issue #31): when `Some` alongside `verify`, the VERIFY
    // branch appends a per-row `Op::Argmax` to the batched forward and downloads the m u32
    // greedy ids into this (leaving `verify`'s logits vec EMPTY — only m×4 bytes cross the bus,
    // not m×vocab×4). Falls back to the full-logits download (this vec left empty, `verify`
    // filled as before) when the backend lacks `Capabilities::argmax_rows`, when a grammar
    // constraint needs host logits, or under INFR_NO_GPU_ARGMAX / INFR_NO_GPU_MTP_ACCEPT (A/B).
    // The caller must handle both shapes (`ids.is_empty()` = host path). `None` everywhere but
    // the MTP driver's `run_verify`.
    verify_ids: Option<&mut Vec<u32>>,
    // Phase-1 DiffusionGemma validation hook: captures the LAST prompt token's raw logits (the
    // per-token loop's first `is_decode` row, i.e. the causal-prefill result) without disturbing
    // the sampled continuation. `None` everywhere else. Unlike `verify` (a batched m-row forward,
    // MoE-incompatible — see its guard below) this rides the existing rows==1 per-token loop, so
    // it works for MoE/diffusion-gemma models too.
    mut logits_out: Option<&mut Vec<f32>>,
    // MTP Phase 1 (issue #33, docs/mtp.md): captures the LM-head INPUT rows (post-`output_norm`,
    // pre-`w_lm` — `DecodeHandles::h_out`'s doc) for the SAME row(s) `logits_out`/`verify` came
    // from: `[ne]` for the per-token decode loop's frontier row, `[m * ne]` for speculative
    // VERIFY's `m` rows. `None` everywhere else (no extra op, no extra download — see `h_tap`'s
    // doc on `build`). This is Phase 2's MTP driver primitive (`h_p` in `docs/mtp.md`); Phase 1
    // only exposes it for validation (`lm_head(h_row) == logits_row`).
    mut h_out: Option<&mut Vec<f32>>,
    // Phase-2 DiffusionGemma canvas denoise (see `DenoiseReq`'s doc). `None` everywhere else.
    denoise_req: Option<DenoiseReq>,
    // The in-flight SEQUENCE's own state (`infr serve`): its sampling overrides, its stop-sequence
    // abort latch, and its turn on the GPU baton.
    //
    // Explicitly per-SEQUENCE, and deliberately LAST so that adding it could not silently reshuffle
    // the `Option`-heavy tail above (a misplaced bare `None` would still typecheck). This used to be
    // a `thread_local!`, which was only sound while one generation owned one thread; N concurrent
    // sequences make that wrong by construction — see `crate::sampling::RequestCtx`.
    //
    // `None` on every non-serve path (run / bench / every test / both goldens / MTP): sampling then
    // resolves purely from the env, no abort latch is polled, and no gate is taken — byte-for-byte
    // the pre-existing behavior.
    req: Option<&crate::sampling::RequestCtx>,
) -> AResult<(Vec<u32>, GenStats)> {
    let c = cfg;
    // Backend capabilities are a per-backend invariant; query ONCE (each call clones an owned
    // struct with a heap `String name`) and read fields off the cached copy below.
    let caps = be.capabilities();
    let (ne, nh) = (c.n_embd, c.n_head);
    // gemma4: per-layer SWA/full dims differ; size shared scratch + KV by the max over layers.
    let max_hd = c.max_head_dim();
    let max_kvrow = c.max_n_kv() * max_hd;
    let max_qrow = nh * max_hd;
    // DeepSeek2 MLA: per-layer dims are uniform (no SWA/full variation), but the max calc is for
    // uniform models anyway.
    let mla_key_len = if c.deepseek2 {
        c.kv_lora_rank + c.qk_rope_dim
    } else {
        0
    };
    let mla_qhead = if c.deepseek2 {
        c.head_k_mla + c.qk_rope_dim
    } else {
        0
    };
    let mla_qrow = nh * mla_qhead;
    let nff = c.n_ff; // max FFN width
    let gemma = c.gemma;
    let gemma4 = c.gemma4;
    let qk_norm = c.qk_norm;
    let act = if gemma {
        Activation::Gelu
    } else {
        Activation::Silu
    };
    // gemma4 E2B (gemma3n): per-layer input embeddings + KV-layer sharing.
    let e2b = c.n_embd_per_layer > 0;
    let npl = c.n_embd_per_layer;

    // Session-stable derivations (per-layer tensor scans + real dequants, all pure in
    // `(caps, g, c, env)`): computed ONCE at cold init, stashed in `SeamKv`, and reused via `Arc`
    // on every warm call / fork instead of re-running the O(n_layer × n_tensors) scans + the
    // gemma4/diffusion `load_tensor_dequant`s per request. `has_wv`/`out_scale`/`dec_out_scale`/
    // `rope_freqs`/`fuse_gu`/`fuse_qkv`/`moe_batched_ok` all live here — see `SessionStable`.
    let stable: std::sync::Arc<SessionStable> = match state.as_ref() {
        Some(kv) => std::sync::Arc::clone(&kv.stable),
        None => std::sync::Arc::new(session_stable(be, g, c, ec)?),
    };
    // Local views into the session-stable derivations (the code below reads these under their
    // original names; `rope_freqs` is read directly as `stable.rope_freqs` at its two sites).
    let has_wv = &stable.has_wv;
    let out_scale = &stable.out_scale;
    let dec_out_scale = &stable.dec_out_scale;
    let fuse_gu = stable.fuse_gu;
    let fuse_qkv = stable.fuse_qkv;
    let moe_batched_ok = stable.moe_batched_ok;

    // qwen35 DeltaNet silu-gated RMSNorm fusion (decode op-fusion campaign): QkNorm's per-head
    // rmsnorm write is immediately read-after-write by the z-gate GatedAct — a real barrier on
    // backends that track hazards (Vulkan). `Op::GatedRmsNorm` collapses the pair into one
    // dispatch with bit-identical math (same reduction, elementwise gate multiply added on
    // store). `kernels.gated_rmsnorm` (`INFR_NO_GATED_RMSNORM`, inverted) forces the split form
    // for A/B (default = fuse).
    let fuse_gated_rmsnorm = caps.gated_rmsnorm && ec.kernels.gated_rmsnorm;

    // GPU embed gather (Op::EmbedGather, task #28): the host feeds token IDS (4 bytes each) and
    // the device gathers+dequantizes the embedding rows from the resident quantized table —
    // decode and prefill stop uploading f32 embedding rows entirely (a 512-token prefill chunk
    // was 4*n_embd*512 = ~8 MB of host-embedded f32; now it's 2 KB of ids). Tied-lm_head models
    // reuse the already-uploaded lm_head buffer (same tensor); untied models upload token_embd
    // once more (extra VRAM = its on-disk size). INFR_NO_GPU_EMBED forces the host path (A/B).
    let untied_lm = g.tensors().iter().any(|t| t.name == "output.weight");
    let gpu_embed = caps.embed_gather
        && c.n_embd.is_multiple_of(32)
        && g.tensors()
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .is_some_and(|t| infr_vulkan::linear::embed_gather_supported(t.dtype))
        && ec.spec.gpu_embed;
    // gemma4-E2B: gather the per-layer TOKEN embedding rows on-device too (the same
    // Op::EmbedGather, table = per_layer_token_embd, scale = sqrt(npl)) — the last host-side
    // per-token gather. Costs the quantized table's on-disk size in VRAM (uploaded once);
    // unlocks the chained decode for E2B (its host ipl gather was the stale-rows blocker).
    let gpu_ple = gpu_embed
        && e2b
        && (c.n_layer * npl).is_multiple_of(32)
        && g.tensors()
            .iter()
            .find(|t| t.name == "per_layer_token_embd.weight")
            .is_some_and(|t| infr_vulkan::linear::embed_gather_supported(t.dtype));

    // KV cache dtype, chosen PER-SIDE (K and V independent, like llama's --cache-type-k /
    // --cache-type-v). Q8_0 stores 34 bytes / 32 elems — half the f16 footprint and bandwidth.
    //   INFR_KV_TYPE_K / INFR_KV_TYPE_V ∈ {f16, q8_0}  (per-side override)
    //   INFR_KV_Q8=1                                    legacy alias: any side not otherwise set → q8_0
    // Per-side KV dtype, chosen from INFR_KV_TYPE_K/V (llama's --cache-type-k/-v). The graph decl
    // carries the dtype and the env is stable for the process, so a warm session and its rebuilt
    // graphs always agree. Gates: Q8_0 needs each layer's KV row (n_kv*head_dim) 32-block-aligned and
    // a backend with the Q8 read/write (cpu/vulkan/metal). TurboQuant (turbo2/3/4) is WHT-rotated,
    // 128-elem blocks = head_dim slices, so it needs head_dim%128; it runs on cpu/vulkan/metal —
    // natively on CPU, through a dequant→f16 prepass on both GPUs. The mainline low-bit quants
    // (q4_0/q4_1/q5_0/q5_1/iq4_nl) + f32/bf16 run on those same three backends; the block quants
    // need 32-alignment. All of these are footprint knobs, not speed knobs — a prepass format
    // re-expands the whole prefix every token.
    // The SAME 32-block gate the placement estimator applies before pinning an auto-q8 cache
    // (`crate::seam::kv_q8_layout_ok`) — one function, so a pinned q8 can never be gated back to
    // f16 here against an estimate that priced it at q8.
    let kv_align_ok = crate::seam::kv_q8_layout_ok(c);
    let kv_q8_backend = matches!(be.name(), "metal" | "cpu" | "vulkan");
    // TurboQuant (turbo2/3/4): CPU + Vulkan + Metal (both GPUs use a dequant→f16 prepass); needs
    // head_dim % 128 (a WHT group is a 128-elem head_dim slice).
    let kv_turbo_ok = matches!(be.name(), "cpu" | "vulkan" | "metal")
        && (0..c.n_layer).all(|l| c.layer_head_dim(l).is_multiple_of(128));
    // Mainline low-bit block quants (q4_0/…/iq4_nl): CPU + Vulkan + Metal; need 32-block alignment.
    let blk_ok = matches!(be.name(), "cpu" | "vulkan" | "metal") && kv_align_ok;
    // Dense f32/bf16 KV: CPU + Vulkan + Metal. Vulkan/Metal store dense; f32 reads natively (its
    // own f32 attention), bf16 reads via a cast→f16 prepass.
    let dense_ok = matches!(be.name(), "cpu" | "vulkan" | "metal");
    // The requested format's NAME is parsed by the shared grammar (`infr_core::budget`, the same
    // table the placement estimator prices with); only the per-format CAPABILITY gates below are
    // the runner's, since they turn on this backend and this model's layout. A request that fails
    // its gate — or a name nothing recognizes — falls through to the default ladder exactly as
    // the old inline match did.
    // `want` is the config's already-parsed dtype (`budget::parse_kv_dtype`, applied once in the
    // env/file/CLI layer). A name nothing recognizes yields `None` here AND leaves `*_specified`
    // true, which is exactly today's split (§11 decision 8): it falls through to the ladder below
    // while still having suppressed auto-q8 up in the placement.
    let parse_kv_fmt = |want: Option<DType>| -> DType {
        match want {
            Some(dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4)) if kv_turbo_ok => dt,
            Some(DType::Q8_0) if kv_align_ok && kv_q8_backend => DType::Q8_0,
            Some(dt @ (DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Iq4Nl))
                if blk_ok =>
            {
                dt
            }
            Some(dt @ (DType::Bf16 | DType::F32)) if dense_ok => dt,
            Some(DType::F16) => DType::F16,
            // unset/unknown/gated-out → legacy `kv.force_q8` alias (both sides q8) or f16.
            _ if ec.kv.force_q8 && kv_align_ok && kv_q8_backend => DType::Q8_0,
            // Placement-pinned auto-q8 (see `crate::seam::PlacementPins`): the Vulkan placement
            // chose a q8 cache to stay resident / keep the default ctx. Vulkan-gated — the pin
            // is a Vulkan placement decision and must not leak into a CPU/Metal session (e.g.
            // the CPU oracle of a parity test) running in the same process. The pin is only ever
            // set when `kv_env_unset()` and `kv_q8_layout_ok` held, so the alignment/backend
            // gates here re-check what already passed; the env being unset means no earlier arm
            // can shadow this one.
            _ if crate::seam::kv_auto_q8() && be.name() == "vulkan" && kv_align_ok => DType::Q8_0,
            _ => DType::F16,
        }
    };
    let mut k_fmt = parse_kv_fmt(ec.kv.type_k);
    let mut v_fmt = parse_kv_fmt(ec.kv.type_v);
    // Metal's Q8 and F32 KV use native-read attention that reads BOTH sides as one dtype, so a
    // mixed request with q8/f32 on one side would misread the other — clamp those to coupled f16.
    // The prepass formats (block quants / bf16 / turbo) expand each side to its own f16 scratch, so
    // they compose freely with each other and with a native-f16 side (per-side, like Vulkan/CPU).
    if be.name() == "metal" && k_fmt != v_fmt {
        let native_read = |dt| matches!(dt, DType::Q8_0 | DType::F32);
        if native_read(k_fmt) || native_read(v_fmt) {
            k_fmt = DType::F16;
            v_fmt = DType::F16;
        }
    }

    // SWA ring KV: window layers allocate `min(want_ctx, window + ubatch)` rows and the backend
    // writes/reads position p at row `p % rows` (see `crate::seam::kv_rows` for the sizing and
    // the correctness argument, and `kv_ring_wanted` for the model/env gates). Requires the
    // backend's ring semantics (Vulkan + CPU; Metal indexes rows by position) and f16/q8 caches
    // on BOTH sides (the prepass formats keep full-context caches — documented scope gate).
    // Everything downstream (alloc, graph decl, rewind guard, fork/seed) derives from this ONE
    // flag; the env set is stable for the process, so warm sessions always recompute the same
    // value their buffers were sized with.
    let kv_ring = caps.kv_swa_ring
        && crate::seam::kv_ring_wanted(c, ec)
        && matches!(k_fmt, DType::F16 | DType::Q8_0)
        && matches!(v_fmt, DType::F16 | DType::Q8_0);

    // ── one-time session init: weights, KV cache, per-step IO (skipped when `state` is warm) ──
    if state.is_none() {
        // Every backend call below (alloc, upload) records into the shared Vulkan command pool, so
        // it must hold the GPU baton like any other step — see `StepGate`'s "correctness" note.
        // `infr serve` pre-forks its slots at startup, so a REQUEST never lands here; this is the
        // lazy `infr run` / CPU / Metal path, plus that startup fork itself.
        let _gp = req.and_then(|r| r.gate_pass());
        // Weight-load progress: opened HERE — the single weight-upload funnel every runner path
        // (CPU/Vulkan/Metal × one-shot/session/bench/serve) goes through — so no entry point can
        // load without it. The ticking lives in each backend's `alloc` (Weights/HostWeights);
        // backends without a display return a no-op scope. Guard drops when the init block ends.
        let fp = crate::weights::weight_footprint(g);
        let _weight_pb = be.weight_progress(Some(fp.dense + fp.expert));
        // ── upload weights in their NATIVE GGUF dtype (no host pre-dequant — the backend dequants
        //    lazily in `bytes_to_f32`, so a quant weight occupies ~quant size, not 8× f32). `wspecs`
        //    records each (dtype, numel) so `build` can declare the handle with the matching dtype; its
        //    order MUST equal the `g.weight()` order in `build` below. ──────────────────────────────────
        let mut wbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut wspecs: Vec<(DType, usize)> = Vec::new();
        // Load one weight (zero-copy mmap slice — no alloc, no memcpy) or CONCATENATE several into
        // one owned buffer (the combined gate+up upload; same dtype, row-major concat of [nff, ne]
        // tensors = a valid [k*nff, ne] tensor). Records the native dtype + element count so
        // `build` declares the handle to match.
        // NEOX→NORM row permute (qwen2, `Config::permute_qk_neox`): qwen2's GGUF keeps attn_q/attn_k
        // in the HF rotate-half order (the converter only permutes llama-arch), but the no-qknorm
        // path's `Op::Rope` is the INTERLEAVED rotation. Reordering each head's rows at load —
        // new[2p] = old[p], new[2p+1] = old[p + rd/2], dims past rope_dim pass through — makes NORM
        // rope over the permuted projections equal NEOX over the originals (llama.cpp's convert-time
        // permute), with no kernel variant on any backend. Row reorder is quant-block-safe (blocks
        // run along the input dim, whole rows move). Returns the head count for a q/k tensor, or
        // None (no permute).
        let qk_perm_heads = |name: &str| -> Option<usize> {
            if !c.permute_qk_neox {
                return None;
            }
            if name.ends_with("attn_q.weight") || name.ends_with("attn_q.bias") {
                Some(c.n_head)
            } else if name.ends_with("attn_k.weight") || name.ends_with("attn_k.bias") {
                Some(c.n_kv)
            } else {
                None
            }
        };
        let permute_rows = |src: &[u8], heads: usize, row_b: usize| -> Vec<u8> {
            let (hd, rd) = (c.head_dim, c.rope_dim);
            let mut out = vec![0u8; src.len()];
            for h in 0..heads {
                for j in 0..hd {
                    let sj = if j < rd {
                        if j % 2 == 0 {
                            j / 2
                        } else {
                            j / 2 + rd / 2
                        }
                    } else {
                        j
                    };
                    let (d, s) = ((h * hd + j) * row_b, (h * hd + sj) * row_b);
                    out[d..d + row_b].copy_from_slice(&src[s..s + row_b]);
                }
            }
            out
        };
        let mut wload = |names: &[&str]| -> AResult<()> {
            let info = |name: &str| {
                g.tensors()
                    .iter()
                    .find(|t| t.name == name)
                    .cloned()
                    .ok_or_else(|| anyhow!("tensor not found: {name}"))
            };
            // Bytes-per-row for the permute: a weight row is `n_embd` elements of the tensor's
            // dtype (block-aligned); a bias "row" is one f32.
            let row_bytes = |name: &str, dt: DType| -> usize {
                if name.ends_with(".bias") {
                    4
                } else {
                    infr_gguf::nbytes(dt, c.n_embd)
                }
            };
            let (bytes, dt, numel) = if info(names[0])?.dtype == DType::I2S {
                // BitNet i2_s carries a per-TENSOR trailing f32 scale and an interleaved 128-group
                // packing (see `DType::I2S`) — neither composes with the per-row weight streaming
                // the seam does (permute_rows / row_bytes / the CPU Op::Linear all assume one quant
                // block per weight row, and the GPU has no i2_s kernel). Host-dequant to f16 ONCE
                // here; every downstream stage then treats it as a plain f16 weight. The f16
                // footprint (numel*2) is exactly what `weights::tensor_resident_bytes` already
                // prices for a non-`native_dense_supported` dtype, so the VRAM budget stays honest.
                let mut cat: Vec<u8> = Vec::new();
                let mut numel = 0usize;
                for name in names {
                    let i = info(name)?;
                    if i.dtype != DType::I2S {
                        return Err(anyhow!("wload concat dtype mismatch: {names:?}"));
                    }
                    numel += i.shape.iter().product::<usize>();
                    let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
                    let f32v = crate::dequant_block(DType::I2S, &tb).map_err(|e| anyhow!("{e}"))?;
                    let f16: Vec<u8> = f32v
                        .iter()
                        .flat_map(|&x| infr_gguf::dequant::f32_to_f16_sat(x).to_le_bytes())
                        .collect();
                    match qk_perm_heads(name) {
                        Some(heads) => cat.extend_from_slice(&permute_rows(
                            &f16,
                            heads,
                            row_bytes(name, DType::F16),
                        )),
                        None => cat.extend_from_slice(&f16),
                    }
                }
                (WBytes::Owned(cat), DType::F16, numel)
            } else if let [name] = names {
                let i = info(name)?;
                let tb = g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
                let numel = i.shape.iter().product();
                match qk_perm_heads(name) {
                    Some(heads) => {
                        let rb = row_bytes(name, i.dtype);
                        (WBytes::Owned(permute_rows(&tb, heads, rb)), i.dtype, numel)
                    }
                    None => (WBytes::Mmap(tb), i.dtype, numel),
                }
            } else {
                // A fused group (qkv, gate+up). Its components are handed over as VIEWS, not as a
                // concatenated buffer: a binder that pages or streams the group registers their
                // file ranges and never wants the bytes, and materializing here would build — and
                // fault in — a multi-MB copy per group only to drop it. `WBytes::materialize` joins
                // them for the binders that do read bytes.
                //
                // A permuted component (qwen2 q/k) is a load-time REWRITE with no on-disk form, so
                // any group containing one falls back to the owned concat.
                let mut parts = Vec::with_capacity(names.len());
                let mut numel = 0usize;
                let mut permuted = false;
                let dt = info(names[0])?.dtype;
                for name in names {
                    let i = info(name)?;
                    if i.dtype != dt {
                        return Err(anyhow!("wload concat dtype mismatch: {names:?}"));
                    }
                    numel += i.shape.iter().product::<usize>();
                    parts.push(g.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?);
                    permuted |= qk_perm_heads(name).is_some();
                }
                if !permuted {
                    (WBytes::Concat(parts), dt, numel)
                } else {
                    let mut cat = Vec::new();
                    for (name, tb) in names.iter().zip(&parts) {
                        match qk_perm_heads(name) {
                            Some(heads) => {
                                cat.extend_from_slice(&permute_rows(tb, heads, row_bytes(name, dt)))
                            }
                            None => cat.extend_from_slice(tb),
                        }
                    }
                    (WBytes::Owned(cat), dt, numel)
                }
            };
            // bind_weight returns the EFFECTIVE dtype the buffer holds (the GPU binder may convert float
            // weights to f16), so the graph declares the handle to match what the backend will read.
            let (buf, eff_dt) = bind_weight(names[0], bytes, dt, numel)?;
            wbufs.push(buf);
            wspecs.push((eff_dt, numel));
            Ok(())
        };
        for l in 0..c.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            // qwen35 gated-DeltaNet linear-attention layer (see docs/qwen35.md): a wholly different
            // mixer, no q/k/v/qk_norm/attn_output/bias at all. `false` for every non-qwen35 model.
            let is_delta = c.qwen35 && !c.is_qwen35_attn_layer(l);
            let is_mla = c.deepseek2;
            wload(&[&p("attn_norm.weight")])?;
            if is_mla {
                // MLA: wq_a → q_a_norm → wq_b (or wq for lite), wkv_a_mqa, kv_a_norm, wk_b, wv_b, wo.
                if !c.is_lite {
                    wload(&[&p("attn_q_a.weight")])?;
                    wload(&[&p("attn_q_a_norm.weight")])?;
                }
                wload(&[&p(if c.is_lite {
                    "attn_q.weight"
                } else {
                    "attn_q_b.weight"
                })])?;
                wload(&[&p("attn_kv_a_mqa.weight")])?;
                wload(&[&p("attn_kv_a_norm.weight")])?;
                wload(&[&p("attn_k_b.weight")])?;
                wload(&[&p("attn_v_b.weight")])?;
                wload(&[&p("attn_output.weight")])?;
            } else if is_delta {
                wload(&[&p("attn_qkv.weight")])?;
                wload(&[&p("attn_gate.weight")])?;
                wload(&[&p("ssm_conv1d.weight")])?;
                wload(&[&p("ssm_alpha.weight")])?;
                wload(&[&p("ssm_beta.weight")])?;
                wload(&[&p("ssm_a")])?;
                wload(&[&p("ssm_dt.bias")])?;
                wload(&[&p("ssm_norm.weight")])?;
                wload(&[&p("ssm_out.weight")])?;
            } else if fuse_qkv {
                wload(&[
                    &p("attn_q.weight"),
                    &p("attn_k.weight"),
                    &p("attn_v.weight"),
                ])?;
            } else {
                wload(&[&p("attn_q.weight")])?;
                wload(&[&p("attn_k.weight")])?;
                if has_wv[l] {
                    wload(&[&p("attn_v.weight")])?;
                }
            }
            // Qwen2/2.5 q/k/v projection biases (small f32 [out_f] vectors). Loaded AFTER the q/k/v
            // weights so the upload order matches the `wpush` order below.
            if c.qkv_bias {
                wload(&[&p("attn_q.bias")])?;
                wload(&[&p("attn_k.bias")])?;
                wload(&[&p("attn_v.bias")])?;
            }
            if qk_norm && !is_delta && !is_mla {
                wload(&[&p("attn_q_norm.weight")])?;
                wload(&[&p("attn_k_norm.weight")])?;
            }
            if !is_delta && !is_mla {
                wload(&[&p("attn_output.weight")])?;
            }
            // bitnet SubLN: the attention-output RMSNorm sits BETWEEN the attention op and `wo` in
            // the graph, but load order is arbitrary as long as `wpush` mirrors it — kept right
            // after `attn_output` (its logical neighbor). `[n_embd]`, resident (small f32 norm).
            if c.sub_norm && !is_delta {
                wload(&[&p("attn_sub_norm.weight")])?;
            }
            if gemma {
                wload(&[&p("post_attention_norm.weight")])?;
            }
            // qwen35 names its post-mixer/pre-FFN norm `post_attention_norm.weight` on BOTH layer
            // kinds (not `ffn_norm.weight`) — same role (`lw.ffn_norm`), different tensor name.
            let ffn_norm_name = if c.qwen35 {
                "post_attention_norm.weight"
            } else {
                "ffn_norm.weight"
            };
            wload(&[&p(ffn_norm_name)])?;
            if c.dual_moe() {
                // Dual FFN: dense GeGLU (n_ff=2112) ∥ 128-expert MoE (fused gate_up_exps + a
                // per-expert down scale), summed — see docs/diffusion-gemma.md's FFN wiring. Shared
                // by diffusion-gemma and the autoregressive gemma4 MoE (26B-A4B); identical tensors.
                // `fuse_gu`: one concatenated [2*nff,ne] gate+up tensor (see the comment at its
                // definition) instead of two separate n_ff=2112 tensors.
                if fuse_gu {
                    wload(&[&p("ffn_gate.weight"), &p("ffn_up.weight")])?;
                } else {
                    wload(&[&p("ffn_gate.weight")])?;
                    wload(&[&p("ffn_up.weight")])?;
                }
                wload(&[&p("ffn_down.weight")])?;
                wload(&[&p("post_ffw_norm_1.weight")])?;
                wload(&[&p("pre_ffw_norm_2.weight")])?;
                wload(&[&p("ffn_gate_inp.weight")])?;
                wload(&[&p("ffn_gate_inp.scale")])?;
                wload(&[&p("ffn_gate_up_exps.weight")])?;
                wload(&[&p("ffn_down_exps.weight")])?;
                wload(&[&p("ffn_down_exps.scale")])?;
                wload(&[&p("post_ffw_norm_2.weight")])?;
            } else if c.moe.is_some() && c.is_moe_layer(l) {
                // qwen3moe / qwen35moe / llama4 (MoE layer): router + stacked per-expert gate/up/
                // down banks. (llama4 interleaves dense layers on `moe_interleave_step`; those fall
                // through to the dense branches below — Scout's step==1 makes every layer MoE.)
                wload(&[&p("ffn_gate_inp.weight")])?;
                wload(&[&p("ffn_gate_exps.weight")])?;
                wload(&[&p("ffn_up_exps.weight")])?;
                wload(&[&p("ffn_down_exps.weight")])?;
                if c.shexp_ff > 0 {
                    // Shared expert (qwen35moe / llama4): a dense SwiGLU FFN alongside the routed
                    // bank. qwen35moe gates it by a per-token sigmoid (`ffn_gate_inp_shexp`); llama4
                    // has no gate tensor and sums it in plain — see `FfnW::Moe`'s `shexp` field.
                    if c.shexp_gated {
                        wload(&[&p("ffn_gate_inp_shexp.weight")])?;
                    }
                    wload(&[&p("ffn_gate_shexp.weight")])?;
                    wload(&[&p("ffn_up_shexp.weight")])?;
                    wload(&[&p("ffn_down_shexp.weight")])?;
                }
            } else if fuse_gu {
                wload(&[&p("ffn_gate.weight"), &p("ffn_up.weight")])?;
                wload(&[&p("ffn_down.weight")])?;
            } else {
                wload(&[&p("ffn_gate.weight")])?;
                wload(&[&p("ffn_up.weight")])?;
                wload(&[&p("ffn_down.weight")])?;
            }
            // bitnet SubLN: the FFN-intermediate RMSNorm (`[n_ff]`) applied BEFORE `ffn_down` in the
            // graph; loaded here to mirror the `wpush` order below. Only bitnet is dense-gated AND
            // `sub_norm`, so this never fires on a MoE/dual-FFN layer.
            if c.sub_norm {
                wload(&[&p("ffn_sub_norm.weight")])?;
            }
            if gemma {
                wload(&[&p("post_ffw_norm.weight")])?;
            }
            if e2b {
                // gemma4 E2B per-layer input-embedding application weights.
                wload(&[&p("inp_gate.weight")])?;
                wload(&[&p("proj.weight")])?;
                wload(&[&p("post_norm.weight")])?;
            }
        }
        // Globals: output_norm, lm_head. lm_head = `output.weight`, or (tied) the quantized
        // `token_embd.weight` mapped from the mmap and dequantized per-row by `Op::Linear` — same f32
        // values as the host `token_embd`, but zero-copy.
        wload(&["output_norm.weight"])?;
        if g.tensors().iter().any(|t| t.name == "output.weight") {
            wload(&["output.weight"])?;
        } else {
            wload(&["token_embd.weight"])?;
        }
        // GPU embed gather: untied models upload the quantized token_embd as one extra weight
        // slot (tied models reuse the lm_head slot above — same tensor). Order-sensitive:
        // mirrors `build`'s `w_embd` wpush, right after `w_lm`.
        if gpu_embed && untied_lm {
            wload(&["token_embd.weight"])?;
        }
        // gemma4-E2B on-device per-layer gather: the (large, quantized) per-layer token
        // embedding table. Mirrors `build`'s `w_ple` wpush.
        if gpu_ple {
            wload(&["per_layer_token_embd.weight"])?;
        }
        // diffusion-gemma: top-level self-conditioning gated MLP. LOADED (occupies a weight-buffer
        // slot like any other tensor) but NOT READ by any Op this phase — Phase 1 is the
        // encoder-only causal prefill, which runs with self-conditioning permanently off (see
        // docs/diffusion-gemma.md); the canvas denoise graph (Phase 2+) is the first reader.
        // Loaded BEFORE the e2b block (mutually exclusive with it — no model is both) so the
        // `debug_assert_eq!` below, which indexes `wspecs` directly, isn't straddled by a later
        // `wload` call (the closure's mutable borrow of `wspecs` would conflict with that read).
        if c.diffusion_gemma {
            wload(&["self_cond_pre_norm.weight"])?;
            wload(&["self_cond_gate.weight"])?;
            wload(&["self_cond_up.weight"])?;
            wload(&["self_cond_down.weight"])?;
        }
        // gemma4 E2B: the per-layer input-embedding projection weights, native-uploaded like any
        // other weight (model_proj stays bf16 — the seam's native bf16 GEMV/GEMM reads it directly;
        // proj_norm is f32). The GPU graph prologue (in `build`, below) runs the GEMV + RMSNorm that
        // used to be a host loop. Declared here (in upload order) — `build` pushes the matching
        // handles right after `w_lm`/before `v_ones`.
        if e2b {
            wload(&["per_layer_model_proj.weight"])?;
            wload(&["per_layer_proj_norm.weight"])?;
            // Sanity-check the two uploads landed with the shapes the GPU prologue assumes
            // (model_proj is `[n_layer*npl, n_embd]`, proj_norm is `[npl]`).
            debug_assert_eq!(wspecs[wspecs.len() - 2].1, c.n_layer * npl * ne);
            debug_assert_eq!(wspecs[wspecs.len() - 1].1, npl);
        }
        // gemma4 weightless per-head V-norm = `QkNorm` with a unit weight (out = x/rms). One ones-vector
        // of the max head dim serves every layer (a narrower layer reads its leading prefix).
        if gemma4 {
            let ones = vec![1.0f32; max_hd];
            let b = be
                .alloc(ones.len() * 4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(b.as_ref(), bytemuck::cast_slice(&ones))
                .map_err(|e| anyhow!("{e}"))?;
            wbufs.push(b);
            wspecs.push((DType::F32, max_hd));
        }
        // dual-FFN MoE (diffusion-gemma / gemma4 26B-A4B): weightless FULL-WIDTH (ne-wide) RMSNorm
        // for the MoE router's own input (`rmsnorm_noscale(attn_out)`, see the graph-build wiring) —
        // a SEPARATE ones-vector from `v_ones` above (that one's per-HEAD width `max_hd`; this is
        // the whole residual width).
        if c.dual_moe() {
            let ones = vec![1.0f32; ne];
            let b = be
                .alloc(ones.len() * 4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(b.as_ref(), bytemuck::cast_slice(&ones))
                .map_err(|e| anyhow!("{e}"))?;
            wbufs.push(b);
            wspecs.push((DType::F32, ne));
        }
        // llama4 `Llama4TextL2Norm`: a WEIGHTLESS per-head RMS/L2-norm on Q and K after rope (rope
        // layers only) = `QkNorm` with a unit weight. One `head_dim`-wide ones-vector serves every
        // rope layer (see the matching `qk_ones` handle in `build`).
        if c.kq_l2norm {
            let ones = vec![1.0f32; c.head_dim];
            let b = be
                .alloc(ones.len() * 4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(b.as_ref(), bytemuck::cast_slice(&ones))
                .map_err(|e| anyhow!("{e}"))?;
            wbufs.push(b);
            wspecs.push((DType::F32, c.head_dim));
        }

        // ── re-decide the context against what the device says is LEFT ───────────────────────
        // Every weight this session will hold is resident by now — including the arena block tails
        // and the driver's own memory that no pre-load footprint prices — so the budget query here
        // is a MEASUREMENT where the clamp that sized `want_ctx` could only estimate. Shrinks only,
        // and never a context the user pinned. See `reclamp_ctx_to_live_room`.
        let want_ctx = crate::seam::reclamp_ctx_to_live_room(be, c, ec, want_ctx, k_fmt, v_fmt);

        // ── persistent KV cache buffers, sized per-layer (gemma4 SWA layers are narrower) and
        //    per-side (K and V pick their dtype independently) ────────────────────────────────
        // qwen35 DeltaNet layers have NO KV cache: `kbufs[l]`/`vbufs[l]` instead hold that layer's
        // conv-history state (`[(d_conv-1), conv_channels]` f32) and DeltaNet recurrent state
        // (`[n_vhead, head_k, head_v]` f32) — fixed-size (NOT `want_ctx`-scaled) and always f32
        // regardless of the session's chosen KV dtype (see `MixerW::DeltaNet` / the `build` closure).
        // DeepSeek2 MLA layers have ONE k_cache (key_length = kv_lora_rank + qk_rope_dim wide) per
        // token; V is an aliased prefix view — no separate v_cache.
        let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
        for l in 0..c.n_layer {
            if c.qwen35 && !c.is_qwen35_attn_layer(l) {
                let conv_elems = (c.ssm_d_conv - 1) * c.q35_conv_channels();
                let s_elems = c.q35_num_v_heads() * c.q35_head_k_dim() * c.q35_head_v_dim();
                kbufs.push(
                    be.alloc(conv_elems * 4, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                vbufs.push(
                    be.alloc(s_elems * 4, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                continue;
            }
            // DeepSeek2 MLA: ONE k_cache row (key_length = kv_lora_rank + qk_rope_dim) per token;
            // V is an aliased prefix view — no separate v_cache. vbufs[l] is a zero-size placeholder.
            if c.deepseek2 {
                let key_len = c.kv_lora_rank + c.qk_rope_dim;
                let rows_l = crate::seam::kv_rows(c, l, want_ctx, kv_ring, ec);
                kbufs.push(
                    be.alloc(kv_fmt_bytes(k_fmt, rows_l * key_len), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                vbufs.push(
                    be.alloc(4, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                continue;
            }
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            // SWA ring: window layers hold only `window + ubatch` rows (the ring recycles rows
            // the window mask already excludes — `crate::seam::kv_rows` has the argument).
            let rows_l = crate::seam::kv_rows(c, l, want_ctx, kv_ring, ec);
            kbufs.push(
                be.alloc(kv_fmt_bytes(k_fmt, rows_l * kvrow_l), BufferUsage::KvCache)
                    .map_err(|e| anyhow!("{e}"))?,
            );
            vbufs.push(
                be.alloc(kv_fmt_bytes(v_fmt, rows_l * kvrow_l), BufferUsage::KvCache)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }
        // VRAM-first KV overflow (`INFR_KV_OVERFLOW`): now that every per-layer/per-side KV buffer
        // is placed, let the backend log the resident-vs-spilled split once. No-op with the flag off.
        be.kv_overflow_report();

        // ── per-step IO buffers ────────────────────────────────────────────────────────
        let hidden_buf = be
            .alloc(ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let pos_buf = be
            .alloc(4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let rf_buf = match &stable.rope_freqs {
            Some(rf) => {
                let b = be
                    .alloc(rf.len() * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?;
                be.upload(b.as_ref(), bytemuck::cast_slice(rf))
                    .map_err(|e| anyhow!("{e}"))?;
                Some((b, rf.len()))
            }
            None => None,
        };
        // gemma4 E2B per-(token,layer) input vector `[n_layer*npl]`, recomputed + re-uploaded each step.
        // (Host path only — `gpu_ple` gathers it on-device from the resident table.)
        let ipl_buf = if e2b && !gpu_ple {
            Some(
                be.alloc(c.n_layer * npl * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let logits_buf = be
            .alloc(c.vocab * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        *state = Some(SeamKv {
            weights: std::sync::Arc::new(SeamWeights {
                wbufs,
                wspecs,
                rf_buf,
            }),
            stable: std::sync::Arc::clone(&stable),
            kbufs,
            vbufs,
            k_fmt,
            v_fmt,
            hidden_buf,
            pos_buf,
            ipl_buf,
            logits_buf,
            max_ctx: want_ctx,
            kv_ring,
            cached: Vec::new(),
            denoise_cache: None,
            self_cond_w: None,
            sc_embt: None,
            sc_ping: None,
            sc_ping_write: 0,
            sc_temp_inv_buf: None,
            mtp_delta_ckpt: None,
        });
    }
    let SeamKv {
        weights,
        // The session-stable derivations are already read via the `stable` local above (cloned
        // from this same Arc, or freshly computed on the cold path).
        stable: _,
        kbufs,
        vbufs,
        k_fmt: _,
        v_fmt: _,
        hidden_buf,
        pos_buf,
        ipl_buf,
        logits_buf,
        max_ctx,
        cached,
        denoise_cache,
        self_cond_w,
        sc_embt,
        sc_ping,
        sc_ping_write,
        sc_temp_inv_buf,
        mtp_delta_ckpt: _,
        // The env-derived local `kv_ring` above is this same value on every call (stable env),
        // so the struct field is only read by fork/seed (which have no backend caps at hand).
        kv_ring: _,
    } = state.as_mut().expect("seam state just initialized");
    let SeamWeights {
        wbufs,
        wspecs,
        rf_buf,
    } = weights.as_ref();
    let max_ctx = *max_ctx;
    if prompt.len() + max_new + 1 > max_ctx {
        return Err(anyhow!(
            "prompt {} + gen {} exceeds the session KV capacity {max_ctx}",
            prompt.len(),
            max_new
        ));
    }
    // Ordinary (non-denoise) generation needs a non-empty prompt (the `.min(prompt.len()-1)`
    // prefix-diff below would underflow on empty, and there'd be nothing to sample the first
    // token from), and every prompt id indexes the embedding table directly — range-check both
    // once here so an out-of-vocab id is a clean error, not an OOB slice panic. Denoise carries
    // no prompt (it validates its canvas separately below).
    if denoise_req.is_none() {
        if prompt.is_empty() {
            return Err(anyhow!("empty prompt: nothing to generate from"));
        }
        validate_token_ids(prompt, c.vocab)?;
    }
    // Phase-2 DiffusionGemma denoise: capture the prompt length BEFORE the ordinary prefix-diff
    // logic below runs (a denoise call's `prompt`/`max_new` are empty/0 — see `DenoiseReq`'s
    // caller — so `start`/`cached` are left untouched: the `if denoise_req.is_some()` guard just
    // below makes both a no-op). `P` for the denoise graph is this, NOT `start`.
    let denoise_p = cached.len();
    // ChatSession-style prefix reuse: KV rows 0..start are already materialized for `cached`'s
    // shared prefix — prefill only the suffix. Always leave ≥1 prompt token to process so the
    // first generated token samples from fresh logits.
    //
    // qwen35's gated-DeltaNet recurrent state is an APPEND-ONLY summary, not a per-position cache —
    // it can't rewind to an arbitrary shared prefix the way a real KV cache can. So a turn reuses it
    // ONLY when `prompt` exactly EXTENDS `cached` (mirrors the old seam's `SeamState` rule); anything
    // else (divergent prompt, identical resend, first-ever call) zero-resets every DeltaNet layer's
    // conv/S state and re-prefills from scratch. Dense/attention models keep the generic
    // longest-common-prefix diff.
    let start = if denoise_req.is_some() {
        // No-op: a denoise call never touches `cached` (it isn't part of the prompt/generation
        // token stream) — `cached.truncate(start)` below is then a truncate-to-current-length.
        cached.len()
    } else if c.qwen35 {
        let pfx = common_prefix_len(cached, prompt);
        if pfx == cached.len() && pfx < prompt.len() {
            pfx
        } else {
            // Non-extending prompt (divergent / identical resend / first-ever call): the append-only
            // recurrent state can't rewind to an arbitrary prefix, so zero every DeltaNet layer's
            // conv/S state and re-prefill from scratch. ALWAYS zero — a stateful backend's warmup
            // generation (CPU is a no-op, but Vulkan/Metal run a throwaway "Hi") dirties these
            // persistent buffers while leaving `cached` empty, so an `is_empty` guard here would
            // wrongly skip the reset and the first real prompt would inherit the warmup's state.
            let conv_elems = (c.ssm_d_conv - 1) * c.q35_conv_channels();
            let s_elems = c.q35_num_v_heads() * c.q35_head_k_dim() * c.q35_head_v_dim();
            for l in 0..c.n_layer {
                if !c.is_qwen35_attn_layer(l) {
                    be.upload(
                        kbufs[l].as_ref(),
                        bytemuck::cast_slice(&vec![0f32; conv_elems]),
                    )
                    .map_err(|e| anyhow!("{e}"))?;
                    be.upload(
                        vbufs[l].as_ref(),
                        bytemuck::cast_slice(&vec![0f32; s_elems]),
                    )
                    .map_err(|e| anyhow!("{e}"))?;
                }
            }
            cached.clear();
            0
        }
    } else {
        // `saturating_sub(1)` guards the empty-prompt underflow defensively; the early guard
        // above already rejects an empty non-denoise prompt, so for any real call this is the
        // plain `prompt.len() - 1` (leave ≥1 prompt token to sample the first logits from).
        common_prefix_len(cached, prompt).min(prompt.len().saturating_sub(1))
    };
    // SWA ring rewind guard: a ring layer RETAINS only its last `rows_l` positions — rows for
    // positions older than `cached.len() - rows_l` were recycled by newer writes. Re-prefilling
    // from `start` attends positions `[start - window, start)` on window layers, so a rewind
    // deeper than the ring's retained tail would read recycled rows; fall back to a full
    // re-prefill (start = 0 — correctness over reuse; the full cache never hits this). Extending
    // turns (start == cached.len()) and the ≤1-token `.min(prompt.len()-1)` rewind stay safe:
    // rows_l - window >= ubatch >= 1.
    let start = if kv_ring && start > 0 && start < cached.len() {
        let safe = (0..c.n_layer)
            .filter(|&l| c.is_swa_layer(l))
            .map(|l| crate::seam::kv_rows(c, l, max_ctx, true, ec))
            .filter(|&rows_l| rows_l < max_ctx) // a never-wrapping layer can't lose positions
            .all(|rows_l| {
                let live_from = cached.len().saturating_sub(rows_l);
                start.saturating_sub(c.swa_window) >= live_from
            });
        if safe {
            start
        } else {
            0
        }
    } else {
        start
    };
    cached.truncate(start);

    // Build a forward graph for `batch` tokens starting at absolute position `start_pos`.
    // `batch = 1` is the normal decode path; `batch > 1` is the batched-prefill path.
    // Scratch tensors scale by `batch`; the LM head runs on the last `logits_rows` tokens —
    // 1 everywhere except speculative VERIFY, which needs the distribution after every
    // candidate (logits output = [logits_rows, vocab], logits_rows ∈ {0, 1, batch}).
    // `logits_rows == 0` builds a HEADLESS graph (no logits Output, no LM-head tail at all) —
    // the batched-prefill chunks, whose logits nothing consumes (task #27).
    // `denoise`: build the DiffusionGemma canvas-denoise variant of this layer stack instead of
    // the ordinary causal forward — see docs/diffusion-gemma.md's "Seam extensions". `batch` is the
    // canvas length C, `start_pos` the prompt length P (unchanged meaning: WriteKv still lands at
    // row P, Attention's kv_len is still `start_pos+batch` = P+C, positions are still bound
    // per-row P..P+C-1 by the caller) — ONLY the attention mask and the per-layer output scalar
    // change. Never true for any existing caller (all pass `false`).
    // `gpu_sc`: Phase-B/D perf, DiffusionGemma in-graph self-conditioning (see
    // docs/diffusion-gemma.md's Phase-B and the reference's `dg_canvas_embed`) — `None` for every
    // ordinary caller (CPU denoise included: it keeps the Phase-A host `diffusion_self_cond`
    // path, so `hidden` is already the fully-baked residual). `Some(sc_on)` from the Vulkan AND
    // Metal denoise call sites: `hidden` there holds the RAW scaled canvas embedding (no SC add, no
    // norm) and this flag additionally emits the SC subgraph (`sc_on == true`) and/or the
    // weightless canvas-embed post-norm (always, when `Some`) INSIDE the graph. Baked into the
    // cached plan (the `(cc,p,sc_on)` key — see `DenoiseCache`) rather than a runtime gate, so the
    // compiled graph never branches at execute time.
    // `h_tap` (MTP Phase 1, issue #33): also expose the LM-head input as a second graph Output
    // (`DecodeHandles::h_out`) — see that field's doc. `false` for every existing call site;
    // `true` only from the decode loop / speculative-VERIFY call sites below when the caller
    // passed `h_out: Some(_)` to `generate_dense_backend`.
    // `dyn_sc_scale` (Vulkan-only perf, docs/diffusion-gemma.md's Phase-B "sc round-trip"
    // elimination): only meaningful when `gpu_sc == Some(true)`. `true` declares an EXTRA 1-element
    // Input (`DecodeHandles::temp_inv`) and wires it as the SC subgraph's `Op::Softmax::scale_buf`
    // instead of baking `scale: 1.0` — the denoise call site uploads a 4-byte scalar there each
    // step instead of premultiplying + reuploading the whole `[batch, vocab]` `sc_logits_in`.
    // `false` for every other call site (Metal keeps the original host-premultiply path).
    // Sampling: greedy unless INFR_TEMP is set (the CLI sets chat defaults for run/serve; the
    // golden/parity tests pin INFR_TEMP=0 or leave it unset). Defined BEFORE `build` — the
    // GPU-sampling ops bake the sampler config into the graph.
    let sampler = crate::sampling::Sampler::resolve(req, &ec.sampling);
    let mut rng = crate::sampling::resolve_seed(req, &ec.sampling);
    // Repetition penalties (`infr serve`'s presence/frequency/repeat request fields). `None` for
    // every non-serve caller AND for any request that leaves them at their neutral defaults — see
    // `Penalties::resolve`. When active they must MUTATE the logits row, so they force the host
    // sampling path below (exactly like a grammar constraint does).
    let mut penalties = crate::sampling::Penalties::resolve(req);
    let penalize = penalties.is_some();
    // GPU-resident greedy sampling (`Op::Argmax` appended to the decode graph): only the 4-byte
    // token id is read back per step — the [vocab] logits stay in VRAM (downloaded only for the
    // one-time `logits_out` hook). Grammar-constrained decodes need host logits every step
    // (llguidance masking). INFR_NO_GPU_ARGMAX forces the host path (A/B).
    let gpu_argmax = (sampler.temp <= 0.0 || sampler.top_k == 1)
        && constraint.is_none()
        && !penalize
        && ec.spec.gpu_argmax;
    // GPU-resident stochastic sampling (`Op::Sample`): temperature + top-k + top-p ON the device,
    // inverse-CDF'd with a host-drawn uniform uploaded as 4 bytes/token — the host consumes the
    // SAME xorshift stream as the host sampler (`next_uniform`), so the two paths are
    // distribution-identical. Gate mirrors the kernel bound (2..=SAMPLE_KMAX); top_k == 0
    // (full-vocab) and grammar-constrained decodes keep the host path. INFR_NO_GPU_SAMPLE = A/B.
    let gpu_sample = !gpu_argmax
        && sampler.temp > 0.0
        && (2..=infr_vulkan::Recorder::SAMPLE_KMAX).contains(&sampler.top_k)
        && constraint.is_none()
        && !penalize
        && caps.gpu_sample
        && ec.spec.gpu_sample;

    let build = |batch: usize,
                 start_pos: usize,
                 logits_rows: usize,
                 denoise: bool,
                 gpu_sc: Option<bool>,
                 dyn_sc_scale: bool,
                 h_tap: bool,
                 gpu_argmax: bool,
                 gpu_sample: bool,
                 // build the graph input as TOKEN IDS + an in-graph EmbedGather (gpu_embed
                 // callers) instead of the host-embedded f32 `hidden` rows.
                 use_ids: bool,
                 // Sets `Graph::mtp_verify` — see that field's doc. `true` from ONLY the
                 // speculative-VERIFY call site below (this fn's `verify` param is `Some`);
                 // `false` from every other caller (decode loop, batched prefill, DG denoise).
                 mtp_verify: bool,
                 // LAYER SPAN: emit the ops for `span` only, and carry the residual stream in a
                 // caller-owned buffer instead of graph scratch — `hidden` becomes an `Input` the
                 // caller binds and the ops mutate in place, so a span that is not the whole model
                 // can be chained with the spans around it (`Capabilities::graph_input_inplace` is
                 // what makes that chaining real; the seam's layer-major prefill is the caller).
                 // `None` = the whole model with `hidden` as scratch — every other call site, and
                 // byte-identical to what they built before the parameter existed.
                 span: Option<std::ops::Range<usize>>|
     -> (Graph, DecodeHandles) {
        let (l_first, l_end) = match &span {
            Some(r) => (r.start, r.end),
            None => (0, c.n_layer),
        };
        // A partial span's last layer is not `output_norm`'s input, so an LM head over it would be
        // reading a half-computed residual stream. Every span caller is headless by construction
        // (batched prefill); this is the guard that keeps it that way.
        assert!(
            l_end == c.n_layer || logits_rows == 0,
            "a partial layer span cannot carry the LM head (logits_rows={logits_rows})"
        );
        // gemma4-E2B's layer loop reads `per_layer_inp`, which the PROLOGUE computes — a span that
        // skips the prologue has no way to hand it over (it is a whole `[batch, n_layer*npl]`
        // tensor, not part of the residual stream). E2B prefills token-by-token today and never
        // asks for a span; reject it here rather than emit a graph with an unbound read.
        assert!(
            l_first == 0 || !e2b,
            "gemma4-E2B cannot start a layer span past layer 0 (per_layer_inp is prologue-built)"
        );
        let mut g = Graph::new();
        g.mtp_verify = mtp_verify;
        // DiffusionGemma: force the per-execute STATIC path for every graph of this model (see
        // `Graph::no_decode_replay`). The record-once replay's `_dyn` kernels agree with the
        // static recording only to float-reassociation noise; the entropy-bound denoise loop
        // chaotically amplifies that noise on the ONE committed-prefix KV row the decode loop
        // writes per prefill call (the frontier token) into different accepted tokens — default
        // and `INFR_SEAM_NO_REPLAY=1` runs produced different text. Only the rows==1 decode
        // graph is ever replay-eligible anyway (batched prefill/denoise are rows>1), and DG has
        // no autoregressive decode loop — its per-prefill decode is exactly one token — so this
        // costs nothing while making both modes bit-identical.
        g.no_decode_replay = c.diffusion_gemma;
        let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
        // KV cache dtype: f16 by default (halves memory vs f32, tightens CPU↔GPU parity); Q8_0
        // per-side when the runner enabled it (see `k_fmt`/`v_fmt` at the cache alloc). ONLY the
        // persistent caches take this dtype — the roped q16/k16 staging stays f16
        // (`qk_norm_rope`/`rope_f16` write f16; a Q8_0 decl there would lie to any backend that
        // trusts it, and the Vulkan kv-write peephole fuses on the f16 decl).
        let kd = |n: usize| TensorDesc::new(vec![n], k_fmt);
        let vd = |n: usize| TensorDesc::new(vec![n], v_fmt);
        let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
        // GPU embed gather: `hidden` becomes an Internal computed by the Op::EmbedGather pushed
        // just before the first layer op (after the table weight handle is declared) — unless a
        // layer span asked for it in a caller-owned buffer, where the gather writes the Input.
        let (hidden, tok_ids) = if use_ids {
            let ids = g.input(TensorDesc::new(vec![batch], DType::I32));
            let h = if span.is_some() {
                g.input(f32d(batch * ne))
            } else {
                g.internal(f32d(batch * ne))
            };
            (h, Some(ids))
        } else {
            (g.input(f32d(batch * ne)), None)
        };
        let positions = g.input(TensorDesc::new(vec![batch], DType::I32));
        let rope_freqs = rf_buf.as_ref().map(|(_, n)| g.input(f32d(*n)));
        // gemma4 E2B per-(token,layer) TOKEN embedding rows `[batch, n_layer*npl]` — host-gathered
        // + dequanted (the big `per_layer_token_embd` table stays off-VRAM, gathered per token).
        // The full `per_layer_inp` consumed by the layer loop is computed from this on the GPU
        // (model_proj GEMV + RMSNorm), further down, once its weights are declared.
        // On-device per-layer gather (`gpu_ple` + a token-ids build): the rows become an
        // Internal filled by an Op::EmbedGather from the resident table — `pl_gathered` marks
        // that the driver must NOT bind it (DecodeHandles.pl_tok_in goes out as None).
        let pl_gathered = use_ids && gpu_ple;
        let pl_tok_in = if e2b {
            Some(if pl_gathered {
                g.internal(f32d(batch * c.n_layer * npl))
            } else {
                g.input(f32d(batch * c.n_layer * npl))
            })
        } else {
            None
        };
        // Phase-B perf: previous-step canvas logits `[batch, vocab]`, premultiplied by temp_inv on
        // the HOST before upload (keeps `Op::Softmax`'s `scale` a constant 1.0 across steps whose
        // temp_inv legitimately changes, so this same plan replays — see the denoise call site).
        let sc_logits_in = if gpu_sc == Some(true) {
            Some(g.input(f32d(batch * c.vocab)))
        } else {
            None
        };
        // Vulkan-only perf: the SC softmax's per-step temperature divisor, read from a 4-byte
        // device buffer instead of baked into the plan (see `dyn_sc_scale`'s doc above and
        // `Op::Softmax::scale_buf`). `None` keeps the SC subgraph's Softmax at the original
        // `scale: 1.0` (host-premultiplied `sc_logits_in` — Metal's path).
        let temp_inv_id = if gpu_sc == Some(true) && dyn_sc_scale {
            Some(g.input(f32d(1)))
        } else {
            None
        };
        // qwen35 DeltaNet layers have no KV cache — `k_cache[l]`/`v_cache[l]` instead declare that
        // layer's conv-state / DeltaNet-S-state Inputs (see the matching alloc in
        // `generate_dense_backend` and `MixerW::DeltaNet`'s use of them below).
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for l in 0..c.n_layer {
            if c.qwen35 && !c.is_qwen35_attn_layer(l) {
                let conv_elems = (c.ssm_d_conv - 1) * c.q35_conv_channels();
                let s_elems = c.q35_num_v_heads() * c.q35_head_k_dim() * c.q35_head_v_dim();
                k_cache.push(g.input(f32d(conv_elems)));
                v_cache.push(g.input(f32d(s_elems)));
                continue;
            }
            // DeepSeek2 MLA: ONE k_cache row (key_length wide); v_cache is a zero-size placeholder.
            if c.deepseek2 {
                let key_len = c.kv_lora_rank + c.qk_rope_dim;
                let rows_l = crate::seam::kv_rows(c, l, max_ctx, kv_ring, ec);
                k_cache.push(g.input(kd(rows_l * key_len)));
                v_cache.push(g.input(vd(4))); // placeholder, V aliased from K
                continue;
            }
            let kvrow_l = c.layer_n_kv(l) * c.layer_head_dim(l);
            // Declared rows MUST equal the allocation above: every backend derives the ring's
            // row capacity from this declared element count (row = pos % (numel / row_width)).
            let rows_l = crate::seam::kv_rows(c, l, max_ctx, kv_ring, ec);
            k_cache.push(g.input(kd(rows_l * kvrow_l)));
            v_cache.push(g.input(vd(rows_l * kvrow_l)));
        }

        // Weights — declared in the SAME order as the upload loop, pulling (dtype, numel) from
        // `wspecs` so each handle carries its native GGUF dtype (the backend dequants on read).
        // `wpush` records the handle in the flat `weights` list (for binding) and returns it.
        let mut weights: Vec<TensorId> = Vec::new();
        let mut wi = 0usize;
        let mut wpush = |g: &mut Graph, weights: &mut Vec<TensorId>| -> TensorId {
            let (dt, n) = wspecs[wi];
            wi += 1;
            let id = g.weight(TensorDesc::new(vec![n], dt));
            weights.push(id);
            id
        };
        let mut lw: Vec<LayerW> = Vec::new();
        for l in 0..c.n_layer {
            let attn_norm = wpush(&mut g, &mut weights);
            // qwen35 gated-DeltaNet layer: 9 mixer weights, no q/k/v/qk_norm/bias/wo at all (mirrors
            // the `wload` skip above). `is_delta` is `false` for every non-qwen35 model.
            let is_delta = c.qwen35 && !c.is_qwen35_attn_layer(l);
            let is_mla = c.deepseek2;
            let mixer = if is_mla {
                let (wq_a, q_a_norm) = if c.is_lite {
                    (None, None)
                } else {
                    (
                        Some(wpush(&mut g, &mut weights)),
                        Some(wpush(&mut g, &mut weights)),
                    )
                };
                let wq_b = wpush(&mut g, &mut weights);
                let wkv_a_mqa = wpush(&mut g, &mut weights);
                let kv_a_norm = wpush(&mut g, &mut weights);
                let wk_b = wpush(&mut g, &mut weights);
                let wv_b = wpush(&mut g, &mut weights);
                let wo = wpush(&mut g, &mut weights);
                MixerW::Mla(MlaW {
                    wq_a,
                    q_a_norm,
                    wq_b,
                    wkv_a_mqa,
                    kv_a_norm,
                    wk_b,
                    wv_b,
                    wo,
                })
            } else if is_delta {
                let qkv = wpush(&mut g, &mut weights);
                let gate = wpush(&mut g, &mut weights);
                let conv1d = wpush(&mut g, &mut weights);
                let alpha = wpush(&mut g, &mut weights);
                let beta = wpush(&mut g, &mut weights);
                let ssm_a = wpush(&mut g, &mut weights);
                let dt_bias = wpush(&mut g, &mut weights);
                let ssm_norm = wpush(&mut g, &mut weights);
                let out = wpush(&mut g, &mut weights);
                MixerW::DeltaNet(DeltaW {
                    qkv,
                    gate,
                    conv1d,
                    alpha,
                    beta,
                    ssm_a,
                    dt_bias,
                    ssm_norm,
                    out,
                })
            } else {
                // Fused QKV: ONE concatenated weight handle serves q/k/v (the builder bakes each
                // projection's `w_off` slice); split form declares three.
                let (wq, wk, wv) = if fuse_qkv {
                    let wqkv = wpush(&mut g, &mut weights);
                    (wqkv, wqkv, Some(wqkv))
                } else {
                    let wq = wpush(&mut g, &mut weights);
                    let wk = wpush(&mut g, &mut weights);
                    let wv = if has_wv[l] {
                        Some(wpush(&mut g, &mut weights))
                    } else {
                        None
                    };
                    (wq, wk, wv)
                };
                // Qwen2 q/k/v biases — pushed here to match the `wload` order (after the q/k/v
                // weights, before qk_norm). Always three separate handles (they add to the SPLIT
                // q/k/v buffers, independent of whether the weights were fused).
                let (qb, kb, vb) = if c.qkv_bias {
                    (
                        Some(wpush(&mut g, &mut weights)),
                        Some(wpush(&mut g, &mut weights)),
                        Some(wpush(&mut g, &mut weights)),
                    )
                } else {
                    (None, None, None)
                };
                let (q_norm, k_norm) = if qk_norm {
                    (
                        Some(wpush(&mut g, &mut weights)),
                        Some(wpush(&mut g, &mut weights)),
                    )
                } else {
                    (None, None)
                };
                let wo = wpush(&mut g, &mut weights);
                MixerW::Attn(AttnW {
                    wq,
                    wk,
                    wv,
                    qb,
                    kb,
                    vb,
                    q_norm,
                    k_norm,
                    wo,
                })
            };
            // bitnet SubLN attention-output norm — mirrors the `wload` push right after
            // `attn_output.weight` (loaded under the same `c.sub_norm && !is_delta` gate; bitnet
            // has no DeltaNet layers, so `is_delta` is always false there).
            let attn_sub_norm = if c.sub_norm && !is_delta {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let post_attn = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let ffn_norm = wpush(&mut g, &mut weights);
            let ffn = if c.dual_moe() {
                let d_gate = wpush(&mut g, &mut weights);
                // fused: `d_up` is the SAME handle as `d_gate` (one concatenated upload, see the
                // matching `wload` above) — never separately read; mirrors `FfnW::Moe`'s
                // `up_exps: gate_up_exps` pattern for the same reason.
                let d_up = if fuse_gu {
                    d_gate
                } else {
                    wpush(&mut g, &mut weights)
                };
                FfnW::DiffusionMoe {
                    d_gate,
                    d_up,
                    fused_gu: fuse_gu,
                    d_down: wpush(&mut g, &mut weights),
                    d_post_norm: wpush(&mut g, &mut weights),
                    m_pre_norm: wpush(&mut g, &mut weights),
                    router: wpush(&mut g, &mut weights),
                    router_scale: wpush(&mut g, &mut weights),
                    gate_up_exps: wpush(&mut g, &mut weights),
                    down_exps: wpush(&mut g, &mut weights),
                    down_scale: wpush(&mut g, &mut weights),
                    m_post_norm: wpush(&mut g, &mut weights),
                }
            } else if c.moe.is_some() && c.is_moe_layer(l) {
                FfnW::Moe {
                    router: wpush(&mut g, &mut weights),
                    gate_exps: wpush(&mut g, &mut weights),
                    up_exps: wpush(&mut g, &mut weights),
                    down_exps: wpush(&mut g, &mut weights),
                    shexp: if c.shexp_ff > 0 {
                        // Order MUST mirror the `wload` above: `gate_inp` (qwen35moe only) precedes
                        // the gate/up/down. llama4 has no gate tensor (`gate_inp = None`).
                        let gate_inp = if c.shexp_gated {
                            Some(wpush(&mut g, &mut weights))
                        } else {
                            None
                        };
                        Some(MoeSharedW {
                            gate_inp,
                            wgate: wpush(&mut g, &mut weights),
                            wup: wpush(&mut g, &mut weights),
                            wdown: wpush(&mut g, &mut weights),
                        })
                    } else {
                        None
                    },
                }
            } else if fuse_gu {
                FfnW::DenseFused {
                    wgu: wpush(&mut g, &mut weights),
                    wdown: wpush(&mut g, &mut weights),
                }
            } else {
                FfnW::Dense {
                    wgate: wpush(&mut g, &mut weights),
                    wup: wpush(&mut g, &mut weights),
                    wdown: wpush(&mut g, &mut weights),
                }
            };
            // bitnet SubLN FFN-intermediate norm — mirrors the `wload` push right after `ffn_down`.
            let ffn_sub_norm = if c.sub_norm {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let post_ffw = if gemma {
                Some(wpush(&mut g, &mut weights))
            } else {
                None
            };
            let (pl_inp_gate, pl_proj, pl_post_norm) = if e2b {
                (
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                    Some(wpush(&mut g, &mut weights)),
                )
            } else {
                (None, None, None)
            };
            lw.push(LayerW {
                attn_norm,
                mixer,
                attn_sub_norm,
                post_attn,
                ffn_norm,
                ffn,
                ffn_sub_norm,
                post_ffw,
                pl_inp_gate,
                pl_proj,
                pl_post_norm,
            });
        }
        let w_out_norm = wpush(&mut g, &mut weights);
        let w_lm = wpush(&mut g, &mut weights);
        // GPU embed gather table: tied-lm_head models read the w_lm slot (same tensor); untied
        // models declare the extra upload here (mirrors the wload order).
        // Declarations MIRROR THE UPLOADS (generation-gated `gpu_embed`/`gpu_ple`, NOT the
        // per-build `use_ids`): wpush consumes wspecs sequentially, so a use_ids=false build
        // (MTP verify, DG denoise) must still declare every uploaded slot or every later
        // weight handle binds one buffer off.
        let w_embd = if gpu_embed && untied_lm {
            Some(wpush(&mut g, &mut weights))
        } else if gpu_embed {
            Some(w_lm) // tied: the lm_head slot IS the token_embd table
        } else {
            None
        };
        let w_ple = if gpu_ple {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        // diffusion-gemma: self-conditioning gated-MLP handles — declared to match `wload`'s
        // upload order (right after lm_head, before the e2b projection weights). Read by the
        // in-graph SC subgraph below when `gpu_sc == Some(true)`; harmlessly unread otherwise
        // (every other build, including the CPU denoise path, which computes SC on the
        // host — see `diffusion_self_cond`).
        let (sc_pre_norm_id, sc_gate_id, sc_up_id, sc_down_id) = if c.diffusion_gemma {
            (
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
            )
        } else {
            (None, None, None, None)
        };
        // Phase-B perf: the SC soft-embedding weight — `token_embd` dequantized + TRANSPOSED to
        // f16 `[n_embd, n_vocab]` (row e holds embedding dim e across every vocab token; the
        // reference's `sc_embT` / `dg_ensure_sc_embT`). NOT a GGUF tensor (`wpush` doesn't cover
        // it) — built once on the host from the already-dequantized `token_embd` and bound
        // separately by the denoise call site (see `SeamKv::sc_embt`).
        let sc_embt_id = if gpu_sc == Some(true) {
            Some(g.weight(TensorDesc::new(vec![c.vocab * ne], DType::F16)))
        } else {
            None
        };
        // gemma4 E2B per-layer input-embedding projection weights — declared here to match the
        // `wload` upload order (right after lm_head/self_cond, before the gemma4 V-norm ones-vector).
        let (mp_w, pn_w) = if e2b {
            (
                Some(wpush(&mut g, &mut weights)),
                Some(wpush(&mut g, &mut weights)),
            )
        } else {
            (None, None)
        };
        let v_ones = if gemma4 {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        // dual-FFN MoE (diffusion-gemma / gemma4 26B-A4B): weightless full-width (ne) RMSNorm
        // ones-vector for the MoE router's own input — see the matching upload in
        // `generate_dense_backend`'s init block.
        let router_ones = if c.dual_moe() {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        // llama4 weightless per-head Q/K L2-norm ones-vector (`head_dim`-wide) — upload order
        // matches the `if c.kq_l2norm` block above.
        let qk_ones = if c.kq_l2norm {
            Some(wpush(&mut g, &mut weights))
        } else {
            None
        };
        // `logits_rows == 0` (task #27): a HEADLESS graph — the chunked batched-prefill path,
        // whose per-chunk logits nothing ever consumes (the sampler reads the decode loop's own
        // fresh logits for the LAST prompt token; earlier rows' logits were always discarded).
        // No logits Output is declared and the whole LM-head tail (output_norm RmsNorm over the
        // chunk, last-row Copy, vocab-wide Linear, Softcap, sampling ops) is skipped — the
        // graph's effect is purely its KV writes.
        let logits = (logits_rows > 0).then(|| g.output(f32d(c.vocab * logits_rows)));

        // scratch (sized to the per-layer max × batch; ops reallocate dst, so these are upper bounds)
        let hn = g.internal(f32d(batch * ne));
        let q = g.internal(f32d(batch * max_qrow));
        let k = g.internal(f32d(batch * max_kvrow));
        let v = g.internal(f32d(batch * max_kvrow));
        // QkNorm+RoPE writes f16 (the GPU `qk_norm_rope` is f32-in→f16-out, can't be in place; the GPU
        // attention reads f16 q). q16/k16 hold the f16 normed+roped q/k for the q/k-norm (qwen3/gemma)
        // path; the llama RoPE-only path stays in f32 q/k. Free on the CPU (its store is f32 regardless).
        let q16 = g.internal(f16d(batch * max_qrow));
        let k16 = g.internal(f16d(batch * max_kvrow));
        let attn = g.internal(f32d(batch * max_qrow));
        // DeepSeek2 MLA scratch: mla_q (f32 query with [nope|rope] per head, roped by the kernel),
        // mla_k16 (f16 K row for WriteKv), mla_kv_cmpr (f32 latent for norm), mla_rope (f32 k_pe).
        let mla_q = if mla_qrow > 0 {
            Some(g.internal(f32d(batch * mla_qrow)))
        } else {
            None
        };
        let mla_k16 = if mla_key_len > 0 {
            Some(g.internal(f16d(batch * mla_key_len)))
        } else {
            None
        };
        let mla_kv_cmpr = if c.deepseek2 {
            Some(g.internal(f32d(batch * c.kv_lora_rank)))
        } else {
            None
        };
        let mla_rope = if c.deepseek2 {
            Some(g.internal(f32d(batch * c.qk_rope_dim)))
        } else {
            None
        };
        // Fused-QKV prefill staging: the wide GEMM writes [batch, qrow+2·kvrow] here, then three
        // CopyStrided ops split it into q/k/v. Decode (batch==1) skips it (offset GEMVs).
        let qkvbuf = if fuse_qkv && batch > 1 {
            Some(g.internal(f32d(batch * (max_qrow + 2 * max_kvrow))))
        } else {
            None
        };
        // Separate gate/up scratch, or one combined [batch, 2*nff] gu buffer when fused — declare
        // only the shape in use (Internal buffers are allocated by the backend even if never read).
        let (gbuf, ubuf, gubuf) = if fuse_gu {
            let gu = g.internal(f32d(batch * 2 * nff));
            (gu, gu, gu)
        } else {
            let gb = g.internal(f32d(batch * nff));
            let ub = g.internal(f32d(batch * nff));
            (gb, ub, gb)
        };
        let actbuf = g.internal(f32d(batch * nff));
        let sub = g.internal(f32d(batch * ne));
        // E2B per-layer embed scratch: gate `[npl]` and projected `[ne]`.
        let plg = g.internal(f32d(batch * npl.max(1)));
        let plp = g.internal(f32d(batch * ne));

        // diffusion-gemma dual-FFN scratch (see docs/diffusion-gemma.md's FFN wiring): the dense
        // branch's own output (`d_out`, before summing with the MoE branch), the router's own
        // input row (`router_tmp` — a DIFFERENT normalization of `attn_out` than either FFN
        // branch reads), the MoE branch's input (`moe_in`) and raw output (`moe_out`). Harmlessly
        // allocated (but unused) on every other arch, like the E2B/qwen35 scratch above.
        let d_out = g.internal(f32d(batch * ne));
        let router_tmp = g.internal(f32d(batch * ne));
        let moe_in = g.internal(f32d(batch * ne));
        let moe_out = g.internal(f32d(batch * ne));
        // qwen35moe shared-expert gate scratch: one raw (pre-sigmoid) logit per token, the
        // `Op::Linear(out_f=1)` output that `Op::MoeSharedExpertAdd` sigmoids — see its doc.
        // Harmlessly allocated (but unused) on every other arch, like the scratch above.
        let shexp_gate = g.internal(f32d(batch));

        // qwen35 attention out-gate scratch (the interleaved q+gate trap — see docs/qwen35.md):
        // `qg` holds the RAW `attn_q` projection (`[batch, nh*2*hd]`, q and gate interleaved per
        // head); `gate_a` holds the split-out gate, packed like `q` (`[batch, nh*hd]`), consumed by
        // the post-attention `GatedAct(Sigmoid)`. Unused (but harmlessly allocated) on every other
        // arch, exactly like the E2B scratch above.
        let qg = g.internal(f32d(batch * max_qrow * 2));
        let _gate_a = g.internal(f32d(batch * max_qrow));

        // qwen35 gated-DeltaNet mixer scratch (see docs/qwen35.md), reused across every DeltaNet
        // layer exactly like `hn`/`sub` above (qwen35's SSM dims are uniform across layers, unlike
        // gemma4's per-layer varying attention dims). `.max(1)`-guarded so a non-qwen35 model (every
        // q35_* dim is 0) still gets a valid, harmlessly-tiny allocation.
        let q35_cc = c.q35_conv_channels();
        let q35_di = c.ssm_d_inner;
        let q35_nk = c.q35_num_k_heads();
        let q35_kd = c.q35_head_k_dim();
        let q35_nv = c.q35_num_v_heads();
        let q35_vd = c.q35_head_v_dim();
        let q35_keydim = q35_nk * q35_kd;
        let dn_qkvbuf = g.internal(f32d(batch * q35_cc.max(1)));
        let dn_zbuf = g.internal(f32d(batch * q35_di.max(1)));
        let dn_convout = g.internal(f32d(batch * q35_cc.max(1)));
        let dn_qbuf = g.internal(f32d(batch * q35_keydim.max(1)));
        let dn_kbuf = g.internal(f32d(batch * q35_keydim.max(1)));
        let dn_vbuf = g.internal(f32d(batch * (q35_nv * q35_vd).max(1)));
        let dn_bbuf = g.internal(f32d(batch * q35_nv.max(1)));
        let dn_abuf = g.internal(f32d(batch * q35_nv.max(1)));
        let dn_out = g.internal(f32d(batch * (q35_nv * q35_vd).max(1)));

        let eps = c.rms_eps;

        // Everything from here to the layer loop is the PROLOGUE: it produces the layer stack's
        // input from the token ids, so it belongs to the span that starts at layer 0. A later span
        // reads the residual stream the earlier one left in the bound `hidden` buffer instead.
        let prologue = l_first == 0;
        // GPU embed gather: materialize `hidden` from the token ids ON the device — the first op
        // of the graph, so every consumer below is unchanged. Bakes Gemma's sqrt(n_embd) scale.
        if let (Some(ids), Some(tbl), true) = (tok_ids, w_embd, prologue) {
            g.push(Op::EmbedGather {
                ids,
                table: tbl,
                dst: hidden,
                rows: batch as u32,
                ne: ne as u32,
                scale: if gemma { (ne as f32).sqrt() } else { 1.0 },
            });
        }
        // gemma4-E2B: the per-layer token rows from the resident table — same gather, same ids,
        // scale = sqrt(npl) (mirrors the host `e2b_ipl_rows`).
        if pl_gathered && prologue {
            if let (Some(ids), Some(tbl), Some(dst)) = (tok_ids, w_ple, pl_tok_in) {
                g.push(Op::EmbedGather {
                    ids,
                    table: tbl,
                    dst,
                    rows: batch as u32,
                    ne: (c.n_layer * npl) as u32,
                    scale: (npl as f32).sqrt(),
                });
            }
        }

        // gemma4 E2B prologue: compute the full per-(token,layer) input vector `per_layer_inp`
        // ([batch, n_layer*npl]) that the layer loop below consumes, on the GPU — matches
        // llama.cpp's split (host: gather+dequant the per-layer token embedding row; GPU: the
        // model_proj GEMV + RMSNorm + combine). `hidden` here is already the scaled token
        // embedding (`emb = token_embd[tok] * embed_scale`), so it IS the `emb` the host version
        // used to dot against `model_proj`.
        let per_layer_inp =
            if let (Some(mp_w), Some(pn_w), Some(pl_tok_in)) = (mp_w, pn_w, pl_tok_in) {
                let nlnpl = c.n_layer * npl;
                let acc = g.internal(f32d(batch * nlnpl));
                g.push(Op::Linear {
                    x: hidden,
                    weight: mp_w,
                    dst: acc,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: nlnpl as u32,
                    w_off: 0,
                });
                g.push(Op::Scale {
                    x: acc,
                    dst: acc,
                    s: 1.0 / (ne as f32).sqrt(),
                    n: (batch * nlnpl) as u32,
                });
                let normed = g.internal(f32d(batch * nlnpl));
                g.push(Op::RmsNorm {
                    x: acc,
                    weight: pn_w,
                    dst: normed,
                    rows: (batch * c.n_layer) as u32,
                    dim: npl as u32,
                    eps,
                });
                let ipl = g.internal(f32d(batch * nlnpl));
                g.push(Op::Add {
                    a: normed,
                    b: pl_tok_in,
                    dst: ipl,
                    n: (batch * nlnpl) as u32,
                });
                g.push(Op::Scale {
                    x: ipl,
                    dst: ipl,
                    s: 1.0 / 2f32.sqrt(),
                    n: (batch * nlnpl) as u32,
                });
                Some(ipl)
            } else {
                None
            };

        // Phase-B perf: DiffusionGemma in-graph canvas embedding (ported from the reference's
        // `dg_canvas_embed` in diffusion-gemma.cpp — see docs/diffusion-gemma.md's Phase-B).
        // `hidden` at this point holds ONLY the raw scaled canvas embedding
        // (`embed(tok)·√n_embd`, no SC add, no norm — the host caller uploads exactly that
        // instead of the Phase-A fully-baked residual). Runs BEFORE the layer loop below, which
        // reads/mutates `hidden` in place exactly as every other caller's — no change needed
        // there.
        if let (Some(sc_on), true) = (gpu_sc, prologue) {
            if sc_on {
                let sc_logits_in =
                    sc_logits_in.expect("gpu_sc(true) plan always declares sc_logits_in");
                let sc_embt = sc_embt_id.expect("gpu_sc(true) plan always declares sc_embt_id");
                let (sc_pre_norm_id, sc_gate_id, sc_up_id, sc_down_id) = (
                    sc_pre_norm_id.expect("diffusion-gemma always declares sc_pre_norm_id"),
                    sc_gate_id.expect("diffusion-gemma always declares sc_gate_id"),
                    sc_up_id.expect("diffusion-gemma always declares sc_up_id"),
                    sc_down_id.expect("diffusion-gemma always declares sc_down_id"),
                );
                let vocab = c.vocab;
                // probs = softmax(sc_logits * scale). `dyn_sc_scale`: `scale_buf` reads temp_inv
                // from a device buffer per step (Vulkan) — `scale: 1.0` below is then ignored.
                // Otherwise temp_inv was already applied on the host, so `scale: 1.0` is the real
                // value (see `sc_logits_in`'s doc / Metal's path).
                let probs = g.internal(f32d(batch * vocab));
                g.push(Op::Softmax {
                    x: sc_logits_in,
                    dst: probs,
                    rows: batch as u32,
                    dim: vocab as u32,
                    scale: 1.0,
                    scale_buf: temp_inv_id,
                });
                // soft = (probs @ sc_embT) * sqrt(n_embd) — sc_embT is [n_embd, n_vocab]
                // (Op::Linear's `weight: [out_f, in_f]` convention), so this is exactly the
                // reference's `ggml_mul_mat(sc_embT, probs)`.
                let soft = g.internal(f32d(batch * ne));
                g.push(Op::Linear {
                    x: probs,
                    weight: sc_embt,
                    dst: soft,
                    m: batch as u32,
                    in_f: vocab as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                g.push(Op::Scale {
                    x: soft,
                    dst: soft,
                    s: (ne as f32).sqrt(),
                    n: (batch * ne) as u32,
                });
                // sc_pre_norm: a NORMAL (weighted) rmsnorm — unlike the canvas embedding's
                // weightless one below.
                let sc_normed = g.internal(f32d(batch * ne));
                g.push(Op::RmsNorm {
                    x: soft,
                    weight: sc_pre_norm_id,
                    dst: sc_normed,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
                // Gated-GELU MLP: down(gelu_tanh(gate·normed) * (up·normed)).
                let sc_g = g.internal(f32d(batch * nff));
                let sc_u = g.internal(f32d(batch * nff));
                g.push(Op::Linear {
                    x: sc_normed,
                    weight: sc_gate_id,
                    dst: sc_g,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: nff as u32,
                    w_off: 0,
                });
                g.push(Op::Linear {
                    x: sc_normed,
                    weight: sc_up_id,
                    dst: sc_u,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: nff as u32,
                    w_off: 0,
                });
                let sc_act = g.internal(f32d(batch * nff));
                g.push(Op::GatedAct {
                    gate: sc_g,
                    up: sc_u,
                    dst: sc_act,
                    rows: batch as u32,
                    nff: nff as u32,
                    act: Activation::Gelu,
                    up_off: 0,
                    up_stride: 0,
                    gate_stride: 0,
                    gate_block_width: 0,
                });
                let sc_sig = g.internal(f32d(batch * ne));
                g.push(Op::Linear {
                    x: sc_act,
                    weight: sc_down_id,
                    dst: sc_sig,
                    m: batch as u32,
                    in_f: nff as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                g.push(Op::Add {
                    a: hidden,
                    b: sc_sig,
                    dst: hidden,
                    n: (batch * ne) as u32,
                });
            }
            // weightless canvas-embed post-norm (no scale weight — matches `dg_canvas_embed`
            // exactly). Reuses `router_ones` (the same ne-wide ones vector diffusion-gemma
            // already declares for the MoE router's weightless norm) instead of a new weight.
            let ones =
                router_ones.expect("diffusion-gemma gpu_sc build always declares router_ones");
            g.push(Op::RmsNorm {
                x: hidden,
                weight: ones,
                dst: hidden,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
        }

        for (l, lw) in lw.iter().enumerate().take(l_end).skip(l_first) {
            // Per-layer dims (gemma4 SWA vs full; uniform for every other model).
            let hd = c.layer_head_dim(l);
            let nkv = c.layer_n_kv(l);
            let kvrow = nkv * hd;
            let qrow = nh * hd;
            let nff_l = c.layer_n_ff(l);
            let theta = c.layer_rope_theta(l); // gemma dual-rope (SWA 1e4 / full 1e6); uniform else
            let rope_dim = c.layer_rope_dim(l);
            let swa = gemma && c.is_swa_layer(l);
            // llama4 iRoPE: NoPE (global) layers skip rope entirely; rope (local) layers apply a
            // weightless per-head L2-norm to Q/K AFTER rope (`Llama4TextL2Norm`). `l2norm` is the
            // ones-vector handle on llama4 rope layers, `None` on NoPE layers and every other model.
            let nope = c.is_nope_layer(l);
            let l2norm = if nope { None } else { qk_ones };
            // DiffusionGemma canvas denoise (docs/diffusion-gemma.md): every canvas query attends
            // the SAME fixed bidirectional range `[lo, kv_len)` — `lo = 0` on full-attention
            // layers (every prompt + canvas key visible), `lo = max(0, P-(n_swa-1))` on SWA
            // layers (only the last `n_swa-1` prompt positions, but every canvas key — canvas
            // keys live in `[P, kv_len)` ⊆ `[lo, kv_len)` on both layer types since `lo <= P`).
            // `start_pos` IS `P` here (the denoise batch starts right after the cached prompt).
            let mask = if denoise {
                let lo = if swa {
                    start_pos.saturating_sub(c.swa_window.saturating_sub(1))
                } else {
                    0
                };
                AttnMask::Canvas { lo }
            } else if swa {
                AttnMask::SlidingWindow(c.swa_window)
            } else {
                AttnMask::Causal
            };
            // gemma4: attn scale 1.0 (QK-norm controls magnitude); everyone else 1/√hd.
            let scale = if gemma4 {
                1.0
            } else {
                1.0 / (hd as f32).sqrt()
            };
            // gemma4 proportional-RoPE applies only on full-attention layers.
            let layer_ff = if gemma4 && !swa { rope_freqs } else { None };
            // attn input norm
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.attn_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            // gemma4 E2B KV-layer sharing: shared layers compute Q only and attend to an earlier
            // layer's cache. `own_kv`/`kv_src` are `true`/`l` for every layer of a non-sharing model.
            let own_kv = c.has_own_kv(l);
            let kv_src = c.kv_src_layer(l);
            if let MixerW::DeltaNet(dw) = &lw.mixer {
                // gated-DeltaNet linear attention (see docs/qwen35.md) — no KV cache; the
                // recurrent state lives in `k_cache[l]`/`v_cache[l]` (repurposed as
                // conv_state/s_state, see the matching alloc in `generate_dense_backend`).
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.qkv,
                    dst: dn_qkvbuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_cc as u32,
                    w_off: 0,
                });
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.gate,
                    dst: dn_zbuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_di as u32,
                    w_off: 0,
                });
                g.push(Op::Conv1dSilu {
                    x: dn_qkvbuf,
                    weight: dw.conv1d,
                    state: k_cache[l],
                    dst: dn_convout,
                    rows: batch as u32,
                    channels: q35_cc as u32,
                    kernel: c.ssm_d_conv as u32,
                });
                // Strided DeltaNet: when INFR_DELTA_STRIDED=1 and batch==1 (decode only —
                // the chunked prefill path doesn't support stride), q/k/v read from conv_out,
                // skipping 3 CopyStrided dispatches per layer.
                let delta_strided = batch == 1 && ec.kernels.vulkan.delta_strided;
                if !delta_strided {
                    g.push(Op::CopyStrided {
                        src: dn_convout,
                        src_off: 0,
                        src_stride: q35_cc as u32,
                        dst: dn_qbuf,
                        dst_off: 0,
                        dst_stride: q35_keydim as u32,
                        rows: batch as u32,
                        n: q35_keydim as u32,
                    });
                    g.push(Op::CopyStrided {
                        src: dn_convout,
                        src_off: q35_keydim as u32,
                        src_stride: q35_cc as u32,
                        dst: dn_kbuf,
                        dst_off: 0,
                        dst_stride: q35_keydim as u32,
                        rows: batch as u32,
                        n: q35_keydim as u32,
                    });
                    g.push(Op::CopyStrided {
                        src: dn_convout,
                        src_off: (2 * q35_keydim) as u32,
                        src_stride: q35_cc as u32,
                        dst: dn_vbuf,
                        dst_off: 0,
                        dst_stride: (q35_nv * q35_vd) as u32,
                        rows: batch as u32,
                        n: (q35_nv * q35_vd) as u32,
                    });
                }
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.beta,
                    dst: dn_bbuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_nv as u32,
                    w_off: 0,
                });
                g.push(Op::Linear {
                    x: hn,
                    weight: dw.alpha,
                    dst: dn_abuf,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: q35_nv as u32,
                    w_off: 0,
                });
                let (q_src, k_src, v_src) = if delta_strided {
                    (dn_convout, dn_convout, dn_convout)
                } else {
                    (dn_qbuf, dn_kbuf, dn_vbuf)
                };
                g.push(Op::DeltaNet {
                    q: q_src,
                    k: k_src,
                    v: v_src,
                    b: dn_bbuf,
                    a: dn_abuf,
                    a_coef: dw.ssm_a,
                    dt_bias: dw.dt_bias,
                    state: v_cache[l],
                    dst: dn_out,
                    rows: batch as u32,
                    n_vhead: q35_nv as u32,
                    n_khead: q35_nk as u32,
                    head_k: q35_kd as u32,
                    head_v: q35_vd as u32,
                    eps: 1e-6,
                    src_stride: 0,
                });
                // silu-gated RMSNorm per v-head: rmsnorm(out, ssm_norm) then * silu(z). Fused into
                // ONE dispatch when the backend supports it (see `fuse_gated_rmsnorm`'s doc) — the
                // split form's GatedAct reads QkNorm's freshly-written `dn_out`, a real
                // read-after-write hazard the fusion removes.
                if fuse_gated_rmsnorm {
                    g.push(Op::GatedRmsNorm {
                        x: dn_out,
                        weight: dw.ssm_norm,
                        gate: dn_zbuf,
                        dst: dn_out,
                        rows: batch as u32,
                        n_head: q35_nv as u32,
                        head_dim: q35_vd as u32,
                        eps,
                    });
                } else {
                    g.push(Op::QkNorm {
                        x: dn_out,
                        weight: dw.ssm_norm,
                        dst: dn_out,
                        rows: batch as u32,
                        n_head: q35_nv as u32,
                        head_dim: q35_vd as u32,
                        eps,
                        x_stride: 0,
                    });
                    g.push(Op::GatedAct {
                        gate: dn_zbuf,
                        up: dn_out,
                        dst: dn_out,
                        rows: batch as u32,
                        nff: (q35_nv * q35_vd) as u32,
                        act: Activation::Silu,
                        up_off: 0,
                        up_stride: 0,
                        gate_stride: 0,
                        gate_block_width: 0,
                    });
                }
                g.push(Op::Linear {
                    x: dn_out,
                    weight: dw.out,
                    dst: sub,
                    m: batch as u32,
                    in_f: q35_di as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                // DeltaNet's residual contribution is already in `sub` — skip the attention-only
                // code below (query/key/value projections, RoPE, Attention, o-proj) entirely.
            } else if let MixerW::Mla(mw) = &lw.mixer {
                // DeepSeek V2+ MLA — absorbed form. Scratch tensors declared above.
                let mla_q = mla_q.expect("deepseek2 model without mla_q scratch");
                let mla_k16 = mla_k16.expect("deepseek2 model without mla_k16 scratch");
                let kv_cmpr = mla_kv_cmpr.expect("deepseek2 model without mla_kv_cmpr scratch");
                let kv_rope = mla_rope.expect("deepseek2 model without mla_rope scratch");
                let key_len = (c.kv_lora_rank + c.qk_rope_dim) as u32;
                let qk_nope = c.head_k_mla as u32;
                let qk_rope = c.qk_rope_dim as u32;
                let q_head_dim = qk_nope + qk_rope;
                let qrow = (c.n_head as u32) * q_head_dim;
                let kv_lora = c.kv_lora_rank as u32;
                let v_hd = c.v_head_dim as u32;

                // Q: wq_a (opt) → RMSNorm → wq_b (or wq for lite).
                if let (Some(wq_a), Some(qan)) = (mw.wq_a, mw.q_a_norm) {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wq_a,
                        dst: attn,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: c.q_lora_rank as u32,
                        w_off: 0,
                    });
                    g.push(Op::RmsNorm {
                        x: attn,
                        weight: qan,
                        dst: attn,
                        rows: batch as u32,
                        dim: c.q_lora_rank as u32,
                        eps,
                    });
                    g.push(Op::Linear {
                        x: attn,
                        weight: mw.wq_b,
                        dst: mla_q,
                        m: batch as u32,
                        in_f: c.q_lora_rank as u32,
                        out_f: qrow,
                        w_off: 0,
                    });
                } else {
                    g.push(Op::Linear {
                        x: hn,
                        weight: mw.wq_b,
                        dst: mla_q,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: qrow,
                        w_off: 0,
                    });
                }

                // KV: wkv_a_mqa → mla_k16 (f16). Split into kv_cmpr and k_pe, norm+rope, reassemble.
                g.push(Op::Linear {
                    x: hn,
                    weight: mw.wkv_a_mqa,
                    dst: mla_k16,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: key_len,
                    w_off: 0,
                });
                // Extract kv_cmpr (first kv_lora columns), norm it.
                g.push(Op::CopyStrided {
                    src: mla_k16,
                    src_off: 0,
                    src_stride: key_len,
                    dst: kv_cmpr,
                    dst_off: 0,
                    dst_stride: kv_lora,
                    rows: batch as u32,
                    n: kv_lora,
                });
                g.push(Op::RmsNorm {
                    x: kv_cmpr,
                    weight: mw.kv_a_norm,
                    dst: kv_cmpr,
                    rows: batch as u32,
                    dim: kv_lora,
                    eps,
                });
                // Copy normed kv_cmpr back.
                g.push(Op::CopyStrided {
                    src: kv_cmpr,
                    src_off: 0,
                    src_stride: kv_lora,
                    dst: mla_k16,
                    dst_off: 0,
                    dst_stride: key_len,
                    rows: batch as u32,
                    n: kv_lora,
                });
                // Extract k_pe (last qk_rope columns), rope it.
                g.push(Op::CopyStrided {
                    src: mla_k16,
                    src_off: kv_lora,
                    src_stride: key_len,
                    dst: kv_rope,
                    dst_off: 0,
                    dst_stride: qk_rope,
                    rows: batch as u32,
                    n: qk_rope,
                });
                g.push(Op::Rope {
                    x: kv_rope,
                    positions,
                    dst: kv_rope,
                    rows: batch as u32,
                    n_head: 1,
                    head_dim: qk_rope,
                    rope_dim: qk_rope,
                    theta,
                    freq_factors: None,
                    x_stride: 0,
                });
                // Copy roped k_pe back.
                g.push(Op::CopyStrided {
                    src: kv_rope,
                    src_off: 0,
                    src_stride: qk_rope,
                    dst: mla_k16,
                    dst_off: kv_lora,
                    dst_stride: key_len,
                    rows: batch as u32,
                    n: qk_rope,
                });
                // WriteKv.
                g.push(Op::WriteKv {
                    src: mla_k16,
                    cache: k_cache[l],
                    rows: batch as u32,
                    row_stride: key_len,
                    pos: start_pos as u32,
                });

                // MLA attention (the kernel handles q_pe rope internally).
                let mla_scale = 1.0 / ((qk_nope + qk_rope) as f32).sqrt();
                g.push(Op::Mla {
                    q: mla_q,
                    k_cache: k_cache[l],
                    wk_b: mw.wk_b,
                    wv_b: mw.wv_b,
                    dst: attn,
                    rows: batch as u32,
                    kv_len: (start_pos + batch) as u32,
                    n_head: c.n_head as u32,
                    q_head_dim,
                    kv_lora_rank: kv_lora,
                    qk_nope_dim: qk_nope,
                    qk_rope_dim: qk_rope,
                    v_head_dim: v_hd,
                    scale: mla_scale,
                    mask,
                    pos: start_pos as u32,
                });
                // wo projection, residual add.
                g.push(Op::Linear {
                    x: attn,
                    weight: mw.wo,
                    dst: sub,
                    m: batch as u32,
                    in_f: (c.n_head as u32) * v_hd,
                    out_f: ne as u32,
                    w_off: 0,
                });
                g.push(Op::Add {
                    a: hidden,
                    b: sub,
                    dst: hidden,
                    n: (batch * ne) as u32,
                });
                // Skip the standard attention code below (q/k/v, RoPE, Attn, o-proj).
                // Continue to FFN.
            } else {
                let MixerW::Attn(aw) = &lw.mixer else {
                    unreachable!("qwen35 DeltaNet handled above")
                };
                if let Some(qkv) = qkvbuf {
                    // Fused QKV (prefill): ONE wide GEMM over the concatenated weight — the separate
                    // q/k/v GEMMs are narrow-n and underfill the GPU — then split rows into q/k/v.
                    let stride = (qrow + 2 * kvrow) as u32;
                    g.push(Op::Linear {
                        x: hn,
                        weight: aw.wq,
                        dst: qkv,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: stride,
                        w_off: 0,
                    });
                    for (dst, off, n) in [
                        (q, 0u32, qrow as u32),
                        (k, qrow as u32, kvrow as u32),
                        (v, (qrow + kvrow) as u32, kvrow as u32),
                    ] {
                        g.push(Op::CopyStrided {
                            src: qkv,
                            src_off: off,
                            src_stride: stride,
                            dst,
                            dst_off: 0,
                            dst_stride: n,
                            rows: batch as u32,
                            n,
                        });
                    }
                } else if fuse_qkv {
                    // Fused QKV (decode): three offset GEMVs into the concatenated weight — the same
                    // dispatch count as the split form, no staging copies.
                    for (dst, off, n) in [
                        (q, 0usize, qrow),
                        (k, qrow * ne, kvrow),
                        (v, (qrow + kvrow) * ne, kvrow),
                    ] {
                        g.push(Op::Linear {
                            x: hn,
                            weight: aw.wq,
                            dst,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: n as u32,
                            w_off: off as u32,
                        });
                    }
                } else if c.attn_out_gate {
                    // qwen35 attention layers pack q + a SIGMOID output gate INTERLEAVED per head in
                    // `attn_q` (`[h0 q(hd) | h0 gate(hd) | h1 q | h1 gate | …]`, NOT two contiguous
                    // blocks — see docs/qwen35.md). Project into `qg` (width 2*qrow) then split each
                    // head's two halves into the packed `q` / `gate_a` scratch.
                    g.push(Op::Linear {
                        x: hn,
                        weight: aw.wq,
                        dst: qg,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: (qrow * 2) as u32,
                        w_off: 0,
                    });
                    // qwen35 attn_out_gate: QkNormRope reads q from qg with stride,
                    // GatedAct(sigmoid) reads gate from qg with stride.
                    // No CopyStrided needed — both ops consume qg directly.
                } else {
                    g.push(Op::Linear {
                        x: hn,
                        weight: aw.wq,
                        dst: q,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: qrow as u32,
                        w_off: 0,
                    });
                }
                // Qwen2 q-bias: `q += qb` after the projection (all three projection paths converge on
                // `q` here), before RoPE. `Wx + b`.
                if let Some(qb) = aw.qb {
                    g.push(Op::AddBias {
                        x: q,
                        bias: qb,
                        dst: q,
                        rows: batch as u32,
                        n: qrow as u32,
                    });
                }
                if own_kv {
                    if !fuse_qkv {
                        g.push(Op::Linear {
                            x: hn,
                            weight: aw.wk,
                            dst: k,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: kvrow as u32,
                            w_off: 0,
                        });
                        // V projection, or (gemma4 full layers) V = the raw K projection, copied BEFORE
                        // K is QK-normed + RoPE'd.
                        match aw.wv {
                            Some(wv) => g.push(Op::Linear {
                                x: hn,
                                weight: wv,
                                dst: v,
                                m: batch as u32,
                                in_f: ne as u32,
                                out_f: kvrow as u32,
                                w_off: 0,
                            }),
                            None => g.push(Op::Copy {
                                src: k,
                                src_off: 0,
                                dst: v,
                                dst_off: 0,
                                n: (batch * kvrow) as u32,
                            }),
                        }
                    }
                    // Qwen2 k/v-bias: `k += kb`, `v += vb` after the projections (here q/k/v are all
                    // materialized in every path — fused prefill/decode projected k/v above, the split
                    // form just did), BEFORE the K RoPE and the V-norm/WriteKv. Emitted before the K
                    // QkNormRope so that op stays adjacent to its WriteKv (see below).
                    if let Some(kb) = aw.kb {
                        g.push(Op::AddBias {
                            x: k,
                            bias: kb,
                            dst: k,
                            rows: batch as u32,
                            n: kvrow as u32,
                        });
                    }
                    if let Some(vb) = aw.vb {
                        g.push(Op::AddBias {
                            x: v,
                            bias: vb,
                            dst: v,
                            rows: batch as u32,
                            n: kvrow as u32,
                        });
                    }
                    // gemma4 weightless per-head RMSNorm on V (= x/rms) before caching. Emitted BEFORE
                    // the K QkNormRope so that op stays ADJACENT to its WriteKv — the Vulkan adapter's
                    // kv_write_peephole only fuses an immediately-following pair, and the record-once
                    // decode path REQUIRES the K write fused (a standalone f16 WriteKv has no dyn
                    // kernel). V only depends on the raw K projection, so the order is free.
                    if let Some(ones) = v_ones {
                        g.push(Op::QkNorm {
                            x: v,
                            weight: ones,
                            dst: v,
                            rows: batch as u32,
                            n_head: nkv as u32,
                            head_dim: hd as u32,
                            eps,
                            x_stride: 0,
                        });
                    }
                    // K: fused QkNorm+RoPE (qwen3/gemma) → f16 `k16`, else RoPE alone (llama) in-place f32.
                    let k_write = match aw.k_norm {
                        Some(kn) => {
                            g.push(Op::QkNormRope {
                                x: k,
                                weight: kn,
                                positions,
                                dst: k16,
                                rows: batch as u32,
                                n_head: nkv as u32,
                                head_dim: hd as u32,
                                rope_dim: rope_dim as u32,
                                theta,
                                eps,
                                freq_factors: layer_ff,
                                x_stride: 0,
                            });
                            k16
                        }
                        None if nope => {
                            // llama4 NoPE (global) layer: NO rope, NO L2-norm — the reference
                            // caches the RAW K projection. `Op::Copy` is a value-preserving cast
                            // here (CPU stores every buffer as f32 regardless of declared dtype,
                            // so this is a no-op there; Vulkan's WriteKv/attention kernels require
                            // the f16 scratch, and the adapter's `Op::Copy` lowering casts f32→f16
                            // instead of a raw byte copy when src/dst dtypes differ).
                            g.push(Op::Copy {
                                src: k,
                                src_off: 0,
                                dst: k16,
                                dst_off: 0,
                                n: (batch * nkv * hd) as u32,
                            });
                            k16
                        }
                        None => {
                            // llama (no k-norm): interleaved RoPE straight to the f16 scratch — the
                            // same fused shape as the qk-norm path, so the Vulkan peephole redirects
                            // the write into the KV cache and the decode replays via rope_f16_dyn.
                            g.push(Op::Rope {
                                x: k,
                                positions,
                                dst: k16,
                                rows: batch as u32,
                                n_head: nkv as u32,
                                head_dim: hd as u32,
                                rope_dim: rope_dim as u32,
                                theta,
                                freq_factors: layer_ff,
                                x_stride: 0,
                            });
                            // llama4 rope layer: weightless per-head L2-norm on the roped K.
                            if let Some(ones) = l2norm {
                                g.push(Op::QkNorm {
                                    x: k16,
                                    weight: ones,
                                    dst: k16,
                                    rows: batch as u32,
                                    n_head: nkv as u32,
                                    head_dim: hd as u32,
                                    eps,
                                    x_stride: 0,
                                });
                            }
                            k16
                        }
                    };
                    g.push(Op::WriteKv {
                        src: k_write,
                        cache: k_cache[l],
                        rows: batch as u32,
                        row_stride: kvrow as u32,
                        pos: start_pos as u32,
                    });
                    g.push(Op::WriteKv {
                        src: v,
                        cache: v_cache[l],
                        rows: batch as u32,
                        row_stride: kvrow as u32,
                        pos: start_pos as u32,
                    });
                }
                // Q: fused QkNorm+RoPE (qwen3/gemma) → f16 `q16`, else RoPE alone (llama) in-place f32.
                let q_attn = match aw.q_norm {
                    Some(qn) => {
                        // Interleaved q+g buffer: skip CopyStrided, QkNormRope reads from qg with stride.
                        let (q_src, q_stride) = if c.attn_out_gate {
                            (qg, (nh * 2 * hd) as u32)
                        } else {
                            (q, 0)
                        };
                        g.push(Op::QkNormRope {
                            x: q_src,
                            weight: qn,
                            positions,
                            dst: q16,
                            rows: batch as u32,
                            n_head: nh as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            eps,
                            freq_factors: layer_ff,
                            x_stride: q_stride,
                        });
                        q16
                    }
                    None if nope => {
                        // llama4 NoPE (global) layer: Q is UNROPED — cast the raw f32 projection
                        // into q16 the same way K does above (`Op::Copy`, value-preserving; see
                        // its doc there). The reference multiplies Q by an attention-temperature
                        // scale here; that scale is EXACTLY 1.0 below the 8192-token chunk size,
                        // so it is a no-op in infr's CPU-testable regime — see the `llama4` arch
                        // note for the long-context follow-up.
                        g.push(Op::Copy {
                            src: q,
                            src_off: 0,
                            dst: q16,
                            dst_off: 0,
                            n: (batch * nh * hd) as u32,
                        });
                        q16
                    }
                    None => {
                        // llama: Q roped to the f16 scratch (the attention kernels read f16 q).
                        g.push(Op::Rope {
                            x: q,
                            positions,
                            dst: q16,
                            rows: batch as u32,
                            n_head: nh as u32,
                            head_dim: hd as u32,
                            rope_dim: rope_dim as u32,
                            theta,
                            freq_factors: layer_ff,
                            x_stride: 0,
                        });
                        // llama4 rope layer: weightless per-head L2-norm on the roped Q.
                        if let Some(ones) = l2norm {
                            g.push(Op::QkNorm {
                                x: q16,
                                weight: ones,
                                dst: q16,
                                rows: batch as u32,
                                n_head: nh as u32,
                                head_dim: hd as u32,
                                eps,
                                x_stride: 0,
                            });
                        }
                        q16
                    }
                };
                g.push(Op::Attention {
                    q: q_attn,
                    k_cache: k_cache[kv_src],
                    v_cache: v_cache[kv_src],
                    dst: attn,
                    rows: batch as u32,
                    kv_len: (start_pos + batch) as u32,
                    n_head: nh as u32,
                    n_kv: nkv as u32,
                    head_dim: hd as u32,
                    scale,
                    mask,
                    pos: start_pos as u32,
                });
                // qwen35: per-head SIGMOID output gate applied to the attention output BEFORE the
                // o-projection (`gate_a` was split out of the interleaved `attn_q` projection above).
                if c.attn_out_gate {
                    g.push(Op::GatedAct {
                        gate: qg, // read gate directly from interleaved q+g buffer
                        up: attn,
                        dst: attn,
                        rows: batch as u32,
                        nff: qrow as u32,
                        act: Activation::Sigmoid,
                        up_off: 0,
                        up_stride: 0,
                        gate_stride: (nh * 2 * hd) as u32, // per-row stride in qg
                        gate_block_width: (2 * hd) as u32, // query+gate block per head
                    });
                }
                // bitnet SubLN: RMSNorm the concatenated-heads attention output (`attn`, width
                // `qrow` = n_head*head_dim = n_embd) BEFORE the o-projection — matches llama.cpp's
                // `build_bitnet` (`attn_sub_norm` applied to `cur` right before `build_lora_mm(wo)`).
                if let Some(asn) = lw.attn_sub_norm {
                    g.push(Op::RmsNorm {
                        x: attn,
                        weight: asn,
                        dst: attn,
                        rows: batch as u32,
                        dim: qrow as u32,
                        eps,
                    });
                }
                g.push(Op::Linear {
                    x: attn,
                    weight: aw.wo,
                    dst: sub,
                    m: batch as u32,
                    in_f: qrow as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
            } // else (MixerW::Attn) — matches the `if let MixerW::DeltaNet` above
              // gemma sandwich: post-attention norm on the sublayer output BEFORE the residual add.
            if let Some(pa) = lw.post_attn {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pa,
                    dst: sub,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
            });
            // ffn
            g.push(Op::RmsNorm {
                x: hidden,
                weight: lw.ffn_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            match lw.ffn {
                FfnW::DenseFused { wgu, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgu,
                        dst: gubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: (2 * nff_l) as u32,
                        w_off: 0,
                    });
                    g.push(Op::GatedActFused {
                        gu: gubuf,
                        dst: actbuf,
                        rows: batch as u32,
                        nff: nff_l as u32,
                        act,
                    });
                    // bitnet SubLN: RMSNorm the FFN intermediate (`actbuf`, width `nff_l`) BEFORE
                    // the down projection — matches llama.cpp `build_bitnet`'s `ffn_sub_norm`.
                    if let Some(fsn) = lw.ffn_sub_norm {
                        g.push(Op::RmsNorm {
                            x: actbuf,
                            weight: fsn,
                            dst: actbuf,
                            rows: batch as u32,
                            dim: nff_l as u32,
                            eps,
                        });
                    }
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                        w_off: 0,
                    });
                }
                FfnW::Dense { wgate, wup, wdown } => {
                    g.push(Op::Linear {
                        x: hn,
                        weight: wgate,
                        dst: gbuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
                    });
                    g.push(Op::Linear {
                        x: hn,
                        weight: wup,
                        dst: ubuf,
                        m: batch as u32,
                        in_f: ne as u32,
                        out_f: nff_l as u32,
                        w_off: 0,
                    });
                    g.push(Op::GatedAct {
                        gate: gbuf,
                        up: ubuf,
                        dst: actbuf,
                        rows: batch as u32,
                        nff: nff_l as u32,
                        act,
                        up_off: 0,
                        up_stride: 0,
                        gate_stride: 0,
                        gate_block_width: 0,
                    });
                    // bitnet SubLN: RMSNorm the FFN intermediate BEFORE the down projection (see the
                    // `FfnW::DenseFused` arm above; same `ffn_sub_norm` from llama.cpp `build_bitnet`).
                    if let Some(fsn) = lw.ffn_sub_norm {
                        g.push(Op::RmsNorm {
                            x: actbuf,
                            weight: fsn,
                            dst: actbuf,
                            rows: batch as u32,
                            dim: nff_l as u32,
                            eps,
                        });
                    }
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: wdown,
                        dst: sub,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                        w_off: 0,
                    });
                }
                FfnW::Moe {
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                    shexp,
                } => {
                    let mc = c.moe.expect("moe layer without MoeConfig");
                    // With a shared expert, the routed branch lands in `moe_out` and
                    // `Op::MoeSharedExpertAdd` combines it into `sub` below; with none (qwen3moe)
                    // it writes `sub` directly, unchanged from before this arm grew a `shexp` field.
                    let moe_dst = if shexp.is_some() { moe_out } else { sub };
                    g.push(Op::MoeFfn {
                        x: hn,
                        router_x: hn, // qwen3moe/qwen35moe: router reads the SAME normed input as the experts
                        router,
                        gate_exps,
                        up_exps,
                        down_exps,
                        down_scale: None,
                        fused_gate_up: false,
                        dst: moe_dst,
                        ne: ne as u32,
                        n_expert: mc.n_expert as u32,
                        n_used: mc.n_used as u32,
                        n_ff_exp: mc.n_ff_exp as u32,
                        scale: mc.scale,
                        act, // qwen3moe/qwen35moe/llama4: SwiGLU (act == Silu)
                        gating: mc.gating,
                        norm_w: mc.norm_w,
                        weight_before: mc.weight_before,
                        ep_band: None, // set per-rank by ExpertParallelBackend's lowering
                    });
                    if let Some(MoeSharedW {
                        gate_inp,
                        wgate,
                        wup,
                        wdown,
                    }) = shexp
                    {
                        // Shared expert (qwen35moe / llama4): a dense SwiGLU FFN on the SAME input
                        // (`hn`) as the routed bank, summed with the routed output. Width is
                        // `shexp_ff` — NOT `nff_l` (llama4's dense `feed_forward_length` is WIDER
                        // than a shared expert; qwen35moe's `nff_l` happens to equal `shexp_ff`).
                        // `fuse_gu` is always false here (no dense `ffn_gate.weight` to fuse), so
                        // `gbuf`/`ubuf`/`actbuf` are the plain scratch (sized to `n_ff >= shexp_ff`),
                        // exactly like `FfnW::Dense` above.
                        let sff = c.shexp_ff as u32;
                        // qwen35moe only: per-token sigmoid gate logit (`Op::Linear` out_f=1).
                        // llama4's shared expert has no gate — it's summed in plain (`Op::Add`).
                        if let Some(gi) = gate_inp {
                            g.push(Op::Linear {
                                x: hn,
                                weight: gi,
                                dst: shexp_gate,
                                m: batch as u32,
                                in_f: ne as u32,
                                out_f: 1,
                                w_off: 0,
                            });
                        }
                        g.push(Op::Linear {
                            x: hn,
                            weight: wgate,
                            dst: gbuf,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: sff,
                            w_off: 0,
                        });
                        g.push(Op::Linear {
                            x: hn,
                            weight: wup,
                            dst: ubuf,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: sff,
                            w_off: 0,
                        });
                        g.push(Op::GatedAct {
                            gate: gbuf,
                            up: ubuf,
                            dst: actbuf,
                            rows: batch as u32,
                            nff: sff,
                            act,
                            up_off: 0,
                            up_stride: 0,
                            gate_stride: 0,
                            gate_block_width: 0,
                        });
                        g.push(Op::Linear {
                            x: actbuf,
                            weight: wdown,
                            dst: d_out,
                            m: batch as u32,
                            in_f: sff,
                            out_f: ne as u32,
                            w_off: 0,
                        });
                        if gate_inp.is_some() {
                            // qwen35moe: `moe_out + sigmoid(gate) · shexp`.
                            g.push(Op::MoeSharedExpertAdd {
                                moe: moe_out,
                                shexp: d_out,
                                gate: shexp_gate,
                                dst: sub,
                                rows: batch as u32,
                                n: ne as u32,
                            });
                        } else {
                            // llama4: `moe_out + shexp` (plain sum, no gate).
                            g.push(Op::Add {
                                a: moe_out,
                                b: d_out,
                                dst: sub,
                                n: (batch * ne) as u32,
                            });
                        }
                    }
                }
                FfnW::DiffusionMoe {
                    d_gate,
                    d_up,
                    fused_gu,
                    d_down,
                    d_post_norm,
                    m_pre_norm,
                    router,
                    router_scale,
                    gate_up_exps,
                    down_exps,
                    down_scale,
                    m_post_norm,
                } => {
                    let mc = c.moe.expect("diffusion-gemma layer without MoeConfig");
                    // Dense branch (the "shared expert"): GELU-par gate/up/down on `hn` (already
                    // ffn_norm(attn_out) from above), then its own post-norm. `act` is Gelu here —
                    // gemma implies it (see the `act` computation above). `fused_gu`: one wide
                    // [ne -> 2*nff] GEMM + `GatedActFused` instead of two n_ff=2112 GEMMs that
                    // clear no warp-tile gate on their own (see `fuse_gu`'s definition comment).
                    if fused_gu {
                        g.push(Op::Linear {
                            x: hn,
                            weight: d_gate,
                            dst: gubuf,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: (2 * nff_l) as u32,
                            w_off: 0,
                        });
                        g.push(Op::GatedActFused {
                            gu: gubuf,
                            dst: actbuf,
                            rows: batch as u32,
                            nff: nff_l as u32,
                            act,
                        });
                    } else {
                        g.push(Op::Linear {
                            x: hn,
                            weight: d_gate,
                            dst: gbuf,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: nff_l as u32,
                            w_off: 0,
                        });
                        g.push(Op::Linear {
                            x: hn,
                            weight: d_up,
                            dst: ubuf,
                            m: batch as u32,
                            in_f: ne as u32,
                            out_f: nff_l as u32,
                            w_off: 0,
                        });
                        g.push(Op::GatedAct {
                            gate: gbuf,
                            up: ubuf,
                            dst: actbuf,
                            rows: batch as u32,
                            nff: nff_l as u32,
                            act,
                            up_off: 0,
                            up_stride: 0,
                            gate_stride: 0,
                            gate_block_width: 0,
                        });
                    }
                    g.push(Op::Linear {
                        x: actbuf,
                        weight: d_down,
                        dst: d_out,
                        m: batch as u32,
                        in_f: nff_l as u32,
                        out_f: ne as u32,
                        w_off: 0,
                    });
                    g.push(Op::RmsNorm {
                        x: d_out,
                        weight: d_post_norm,
                        dst: d_out,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    // Router's OWN input: rmsnorm_noscale(attn_out) · 1/√ne · ffn_gate_inp.scale —
                    // reads the UNNORMED post-attention residual `hidden`, NOT `hn` (neither FFN
                    // branch's normed input). `router_ones` is the weightless full-width RMSNorm
                    // (see its upload next to `v_ones`).
                    let ones = router_ones.expect("diffusion-gemma layer without router_ones");
                    g.push(Op::RmsNorm {
                        x: hidden,
                        weight: ones,
                        dst: router_tmp,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    g.push(Op::Scale {
                        x: router_tmp,
                        dst: router_tmp,
                        s: 1.0 / (ne as f32).sqrt(),
                        n: (batch * ne) as u32,
                    });
                    g.push(Op::MulVec {
                        x: router_tmp,
                        vec: router_scale,
                        dst: router_tmp,
                        rows: batch as u32,
                        n: ne as u32,
                    });
                    // MoE branch input: pre_ffw_norm_2(attn_out) — also reads `hidden`, a THIRD
                    // independent normalization of the same residual.
                    g.push(Op::RmsNorm {
                        x: hidden,
                        weight: m_pre_norm,
                        dst: moe_in,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    g.push(Op::MoeFfn {
                        x: moe_in,
                        router_x: router_tmp,
                        router,
                        gate_exps: gate_up_exps,
                        up_exps: gate_up_exps, // fused: same handle as gate_exps, never read
                        down_exps,
                        down_scale: Some(down_scale),
                        fused_gate_up: true,
                        dst: moe_out,
                        ne: ne as u32,
                        n_expert: mc.n_expert as u32,
                        n_used: mc.n_used as u32,
                        n_ff_exp: mc.n_ff_exp as u32,
                        scale: mc.scale,
                        act,
                        gating: mc.gating,
                        norm_w: mc.norm_w,
                        weight_before: mc.weight_before,
                        ep_band: None, // diffusion-gemma is not an EP arch (Vulkan single-device)
                    });
                    g.push(Op::RmsNorm {
                        x: moe_out,
                        weight: m_post_norm,
                        dst: moe_out,
                        rows: batch as u32,
                        dim: ne as u32,
                        eps,
                    });
                    // out = post_ffw_norm(dense + moe) + attn_out — the sum lands in `sub`; the
                    // shared `post_ffw_norm` (`lw.post_ffw`, generic below) and residual add are
                    // the SAME code every gemma layer already runs.
                    g.push(Op::Add {
                        a: d_out,
                        b: moe_out,
                        dst: sub,
                        n: (batch * ne) as u32,
                    });
                }
            }
            if let Some(pf) = lw.post_ffw {
                g.push(Op::RmsNorm {
                    x: sub,
                    weight: pf,
                    dst: sub,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            g.push(Op::Add {
                a: hidden,
                b: sub,
                dst: hidden,
                n: (batch * ne) as u32,
            });
            // gemma4 E2B per-layer input embedding (gemma3n): mix this layer's input vector into
            // `hidden` after the FFN residual. `g = gelu(inp_gate·hidden) * inp_per_layer[l]`,
            // `p = post_norm(proj·g)`, `hidden += p`.
            if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(ipl)) =
                (lw.pl_inp_gate, lw.pl_proj, lw.pl_post_norm, per_layer_inp)
            {
                g.push(Op::Linear {
                    x: hidden,
                    weight: gate_w,
                    dst: plg,
                    m: batch as u32,
                    in_f: ne as u32,
                    out_f: npl as u32,
                    w_off: 0,
                });
                // gelu(plg) * ipl[r, l*npl .. l*npl+npl] — read the layer-l slice directly from
                // the [batch, n_layer*npl] buffer via per-row stride, without a CopyStrided dispatch.
                g.push(Op::GatedAct {
                    gate: plg,
                    up: ipl,
                    dst: plg,
                    rows: batch as u32,
                    nff: npl as u32,
                    act: Activation::Gelu,
                    up_off: (l * npl) as u32,
                    up_stride: (c.n_layer * npl) as u32,
                    gate_stride: 0,
                    gate_block_width: 0,
                });
                g.push(Op::Linear {
                    x: plg,
                    weight: proj_w,
                    dst: plp,
                    m: batch as u32,
                    in_f: npl as u32,
                    out_f: ne as u32,
                    w_off: 0,
                });
                // fused RMSNorm + Add: hidden += rmsnorm(plp, post_norm)
                g.push(Op::RmsNormAdd {
                    x: plp,
                    weight: post_norm,
                    dst: hidden,
                    rows: batch as u32,
                    dim: ne as u32,
                    eps,
                });
            }
            // gemma4: scale the whole layer output by the per-layer scalar before the next layer.
            // DiffusionGemma denoise reads the DECODER scalar (`layer_output_scale`) instead of
            // the encoder one baked into `out_scale` for every other diffusion-gemma phase (the
            // causal prompt prefill) — see docs/diffusion-gemma.md.
            let layer_scale = if denoise {
                dec_out_scale[l]
            } else {
                out_scale[l]
            };
            if let Some(s) = layer_scale {
                g.push(Op::Scale {
                    x: hidden,
                    dst: hidden,
                    s,
                    n: (batch * ne) as u32,
                });
            }
        }
        // LM-head tail — skipped entirely for headless builds (`logits_rows == 0`, the batched
        // prefill chunks: see `logits`' declaration above).
        let (h_out, tok_id, u_in) = if let Some(logits) = logits {
            g.push(Op::RmsNorm {
                x: hidden,
                weight: w_out_norm,
                dst: hn,
                rows: batch as u32,
                dim: ne as u32,
                eps,
            });
            // For batch > 1 with logits_rows == 1: the LM head runs only on the LAST token's
            // hidden state — extract it via Op::Copy before the projection so the logits output is
            // [vocab]. Speculative verify passes logits_rows == batch and runs the head over every
            // row instead (no Copy).
            let lm_in = if batch > 1 && logits_rows == 1 {
                let hn_last = g.internal(f32d(ne));
                g.push(Op::Copy {
                    src: hn,
                    src_off: ((batch - 1) * ne) as u32,
                    dst: hn_last,
                    dst_off: 0,
                    n: ne as u32,
                });
                hn_last
            } else {
                hn
            };
            // MTP Phase 1 (issue #33): `lm_in` IS the tap target — exactly the rows `logits` is about
            // to be computed from, one op earlier (the reference's `res->t_h_nextn`, captured right
            // after `output_norm` in `qwen35.cpp`). A plain Copy into a fresh Output, so this never
            // disturbs `lm_in`'s existing consumer (the `Op::Linear` below).
            let h_out = if h_tap {
                let ho = g.output(f32d(ne * logits_rows));
                g.push(Op::Copy {
                    src: lm_in,
                    src_off: 0,
                    dst: ho,
                    dst_off: 0,
                    n: (ne * logits_rows) as u32,
                });
                Some(ho)
            } else {
                None
            };
            g.push(Op::Linear {
                x: lm_in,
                weight: w_lm,
                dst: logits,
                m: logits_rows as u32,
                in_f: ne as u32,
                out_f: c.vocab as u32,
                w_off: 0,
            });
            if c.final_softcap > 0.0 {
                g.push(Op::Softcap {
                    x: logits,
                    dst: logits,
                    cap: c.final_softcap,
                    n: (c.vocab * logits_rows) as u32,
                });
            }
            // GPU-resident sampling: pick the token ON the device so only the 4-byte id crosses
            // back to the host (the [vocab] logits stay in VRAM). Appended last so it reads the
            // final (softcapped) logits. Greedy = Op::Argmax; stochastic = Op::Sample with the
            // host-drawn uniform read from the 1-float `u_in` Input. `logits_rows > 1` with
            // `gpu_argmax` is the MTP speculative-verify accept (issue #31): one per-row argmax,
            // m ids read back instead of m×vocab logits.
            let (tok_id, u_in) = if gpu_argmax {
                let tid = g.output(f32d(logits_rows));
                g.push(Op::Argmax {
                    x: logits,
                    dst: tid,
                    n: c.vocab as u32,
                    rows: logits_rows as u32,
                });
                (Some(tid), None)
            } else if gpu_sample && logits_rows == 1 {
                let uin = g.input(f32d(1));
                let tid = g.output(f32d(1));
                g.push(Op::Sample {
                    x: logits,
                    u: uin,
                    dst: tid,
                    n: c.vocab as u32,
                    top_k: sampler.top_k as u32,
                    temp: sampler.temp,
                    top_p: sampler.top_p,
                });
                (Some(tid), Some(uin))
            } else {
                (None, None)
            };
            (h_out, tok_id, u_in)
        } else {
            (None, None, None)
        };
        (
            g,
            DecodeHandles {
                hidden,
                positions,
                rope_freqs,
                pl_tok_in: if pl_gathered { None } else { pl_tok_in },
                sc_logits: sc_logits_in,
                sc_embt: sc_embt_id,
                temp_inv: temp_inv_id,
                logits,
                h_out,
                tok_ids,
                u_in,
                tok_id,
                k_cache,
                v_cache,
                weights,
            },
        )
    };

    // ── Phase-2 DiffusionGemma canvas denoise (see `DenoiseReq`'s doc) ───────────────────────
    // ONE forward over the C canvas rows, reusing the session's already-prefilled prompt KV
    // (rows 0..P, P = `denoise_p`). Mirrors the VERIFY early-return below (batched multi-row
    // forward, LM head on every row) but with the canvas embedding/mask/decoder-scalar wiring.
    if let Some(req) = denoise_req {
        if !c.diffusion_gemma {
            return Err(anyhow!(
                "canvas denoise forward: diffusion-gemma models only"
            ));
        }
        // Phase-A/B perf: per-step timing, gated on INFR_PROF_STAGES=1 (stderr, one line/step).
        // Phase A found `sc` was ~85% of every step (the host SC matvec); Phase B moved that
        // in-graph on Vulkan, so `sc` now reports only the (cheap) host prep — embed gather, and
        // the temp_inv premultiply on the gpu_sc path — while `exec` absorbs the SC math itself.
        let time_diffusion = ec.prof.stages;
        let canvas = req.canvas_tokens;
        // Canvas ids index the embedding table (`tok * n_embd`) below — range-check once so an
        // out-of-vocab id is a clean error, not an OOB slice panic.
        validate_token_ids(canvas, c.vocab)?;
        let cc = canvas.len();
        let p = denoise_p;
        if p + cc > max_ctx {
            return Err(anyhow!(
                "denoise: prompt {p} + canvas {cc} exceeds the session KV capacity {max_ctx}"
            ));
        }
        // Phase-B perf: in-graph self-conditioning on Vulkan (see docs/diffusion-gemma.md's
        // Phase-B and the reference's `dg_canvas_embed`) — Phase D widened this to Metal too:
        // `Op::Softmax`'s wide kernel handles the [C, vocab] shape unmodified (a plain grid-stride
        // loop, no row/dim limit) and `sc_embT`'s `DType::F16` weight already flows through
        // Metal's ordinary non-quant `Op::Linear` path (`weight_buf` dequant-caches it to f32 —
        // functionally correct, just not the dedicated native-f16 GEMV a quant weight would get;
        // see `weight_buf`'s VRAM-budget guard for the failure mode if it doesn't fit). CPU alone
        // keeps the Phase-A host path (`diffusion_self_cond` + host weightless norm) below.
        let gpu_sc = matches!(be.name(), "vulkan" | "metal");
        let sc_on = req.sc_logits.is_some();
        // The plan shape only varies with SC on the gpu_sc path (CPU's graph never changes;
        // `sc_on` there is purely a host-side input difference) — see `DenoiseCache::sc`'s doc.
        let plan_sc = gpu_sc && sc_on;
        // Perf (Vulkan only — docs/diffusion-gemma.md's Phase-B "sc round-trip" elimination): a
        // session-persistent ping-pong pair of GPU buffers (`SeamKv::sc_ping`) stands in for
        // `DenoiseCache`'s per-plan `logits_buf`/`sc_logits_buf`. The previous call's LM-head
        // output is ALREADY resident in one of the pair (it's the very buffer that call
        // downloaded from), so this call's self-conditioning softmax reads it directly instead of
        // the host premultiplying and reuploading the whole `[cc, vocab]` array — see `sc_ping`'s
        // doc and `dyn_sc_scale`'s doc on `build`. Metal keeps the original host-premultiply path
        // (unverified hardware, out of scope here); CPU never reaches `plan_sc` at all.
        let use_ping = be.name() == "vulkan";
        let dyn_sc = plan_sc && use_ping;
        if use_ping && sc_ping.is_none() {
            let bytes = cc * c.vocab * 4;
            *sc_ping = Some([
                be.alloc(bytes, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
                be.alloc(bytes, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            ]);
            *sc_temp_inv_buf = Some(
                be.alloc(4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            );
            if plan_sc {
                // Defensive seed: this is the FIRST denoise call ever on this slot and it ALREADY
                // wants self-conditioning, so the ping slot we're about to read from is freshly
                // zero-initialized, not real data. Every actual caller (`diffusion_generate`'s
                // `denoise_block`) resets `sc_logits` to `None` at the start of every block, so
                // production traffic never takes this branch past the very first call ever — seed
                // it once from the host slice the caller gave us so a hypothetical caller that
                // DOESN'T reset stays correct too.
                let read_idx = 1 - *sc_ping_write;
                be.upload(
                    sc_ping.as_ref().expect("just allocated")[read_idx].as_ref(),
                    bytemuck::cast_slice(req.sc_logits.expect("plan_sc implies sc_on")),
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }

        let t_sc0 = std::time::Instant::now();
        // 1. Canvas embedding: e = embed(tok)·√n_embd. diffusion-gemma is always gemma-family, so
        // `embed_scale` is always √n_embd — computed locally (the "── drive ──" section below
        // defines its own copy for the ordinary decode loop, unreached by this early return). On
        // the gpu_sc path this is ALL of `hidden_host` — the SC add + weightless norm run IN-GRAPH
        // instead (see `build`'s SC subgraph); on CPU it's completed below exactly as Phase A.
        let embed_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
        let mut hidden_host: Vec<f32> = Vec::with_capacity(cc * ne);
        let token_embd = token_embd.get()?; // host embed gather → materialize the table
        for &tok in canvas {
            let base = tok as usize * ne;
            hidden_host.extend(token_embd[base..base + ne].iter().map(|&x| x * embed_scale));
        }
        // Per-step host-premultiplied previous canvas logits for the Metal `gpu_sc` path — `Some`
        // only when `plan_sc && !dyn_sc` (populated below); declared here so it outlives the
        // upload further down. Vulkan's `dyn_sc` path never populates this — see `use_ping`'s doc.
        let mut sc_logits_host: Option<Vec<f32>> = None;
        if let Some(sc_logits) = req.sc_logits {
            // Perf slice 3: on the Vulkan `dyn_sc` ping path the VALUES here are never read (the
            // previous step's raw logits are already GPU-resident in `sc_ping` — see `use_ping`'s
            // doc), so `sc_logits` may be a placeholder slice when the GPU reducer produced the
            // previous step's outcome (no full `[C,vocab]` host buffer to hand back). Only
            // enforce the real shape where the values actually get read below.
            if !dyn_sc && sc_logits.len() != cc * c.vocab {
                return Err(anyhow!(
                    "denoise: sc_logits length {} != {cc}*{} (canvas rows * vocab)",
                    sc_logits.len(),
                    c.vocab
                ));
            }
            if gpu_sc {
                if !dyn_sc {
                    // Premultiply by temp_inv on the HOST (one pass over cc*vocab floats,
                    // threaded) so the in-graph `Op::Softmax`'s `scale` stays a CONSTANT 1.0
                    // across steps whose temp_inv legitimately changes — see `sc_logits_in`'s doc
                    // in `build`. Metal only now: Vulkan's `dyn_sc` path skips this entirely (the
                    // previous step's RAW logits are already GPU-resident in `sc_ping` and the
                    // scale rides `sc_temp_inv_buf` instead — see the bind section below).
                    use rayon::prelude::*;
                    let mut scaled = vec![0f32; sc_logits.len()];
                    scaled
                        .par_iter_mut()
                        .zip(sc_logits.par_iter())
                        .for_each(|(d, &s)| *d = s * req.temp_inv);
                    sc_logits_host = Some(scaled);
                }
            } else {
                // Phase-A host path (CPU only now), unchanged.
                // Phase-A perf: dequantize the self-cond MLP weights ONCE per session, not once
                // per call — `diffusion_self_cond` used to re-run four `load_tensor_dequant`s
                // every step.
                if self_cond_w.is_none() {
                    let (pre_norm, _) = crate::load_tensor_dequant(g, "self_cond_pre_norm.weight")?;
                    let (gate_w, _) = crate::load_tensor_dequant(g, "self_cond_gate.weight")?; // [nff, ne]
                    let (up_w, _) = crate::load_tensor_dequant(g, "self_cond_up.weight")?; // [nff, ne]
                    let (down_w, _) = crate::load_tensor_dequant(g, "self_cond_down.weight")?; // [ne, nff]
                                                                                               // One-time f16 conversion of the embedding table (see `SelfCondWeights::emb16`).
                    let mut emb16 = vec![0u16; token_embd.len()];
                    {
                        use rayon::prelude::*;
                        emb16
                            .par_chunks_mut(1 << 16)
                            .zip(token_embd.par_chunks(1 << 16))
                            .for_each(|(dst, src)| {
                                for (d, &v) in dst.iter_mut().zip(src) {
                                    *d = half::f16::from_f32(v).to_bits();
                                }
                            });
                    }
                    *self_cond_w = Some(std::sync::Arc::new(SelfCondWeights {
                        pre_norm,
                        gate_w,
                        up_w,
                        down_w,
                        emb16,
                    }));
                }
                let scw = self_cond_w.as_ref().expect("just populated above");
                let sc_sig = diffusion_self_cond(scw, c, sc_logits, req.temp_inv, cc)?;
                for (h, s) in hidden_host.iter_mut().zip(sc_sig.iter()) {
                    *h += s;
                }
            }
        }
        if !gpu_sc {
            // Phase-A host weightless canvas-embed norm (CPU only now — the gpu_sc path applies
            // this IN-GRAPH for both the sc-on and no-sc plans, see `build`).
            for row in hidden_host.chunks_mut(ne) {
                let ms: f32 = row.iter().map(|&x| x * x).sum::<f32>() / ne as f32;
                let inv = 1.0 / (ms + c.rms_eps).sqrt();
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        }
        let sc_secs = t_sc0.elapsed().as_secs_f64();
        let dn_positions: Vec<i32> = (p as i32..(p + cc) as i32).collect();

        // Phase-A/B perf: cache the compiled plan + its staging buffers across denoise() calls,
        // keyed by (cc, p, sc) — see `DenoiseCache`'s doc. A hit skips `build`+`compile`+N `alloc`s
        // entirely; a miss (first call, a block boundary, a resized canvas, or an SC on/off
        // transition on the gpu_sc path) rebuilds once and the NEXT call on this key hits.
        let t_build0 = std::time::Instant::now();
        let stale = match denoise_cache {
            Some(dcache) => dcache.cc != cc || dcache.p != p || dcache.sc != plan_sc,
            None => true,
        };
        if stale {
            // 2/3/4. Per-layer forward: the decoder-scalar / Canvas-mask denoise variant of
            // `build`; 5. logits over ALL C rows (logits_rows = cc).
            let (dg, dh) = build(
                cc,
                p,
                cc,
                true,
                if gpu_sc { Some(plan_sc) } else { None },
                dyn_sc,
                false, // MTP h-tap: diffusion-gemma denoise never taps
                false, // gpu_argmax: denoise samples via the EB reducer, not Op::Argmax
                false, // gpu_sample: same
                false, // use_ids: the canvas rows are soft-embeds, not token ids
                false, // mtp_verify: DG denoise is never an MTP-verify batch
                None,  // span: the whole model in one graph
            );
            let plan = be.compile(&dg).map_err(|e| anyhow!("{e}"))?;
            let hidden_buf = be
                .alloc(cc * ne * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            let pos_buf = be
                .alloc(cc * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            // Vulkan: the output lives in the session-level `sc_ping` pair instead (see its doc)
            // — no per-plan logits_buf to allocate. Metal/CPU keep the original per-plan buffer.
            let logits_buf = if use_ping {
                None
            } else {
                Some(
                    be.alloc(cc * c.vocab * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            };
            let sc_logits_buf = if plan_sc && !dyn_sc {
                Some(
                    be.alloc(cc * c.vocab * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            };
            *denoise_cache = Some(DenoiseCache {
                cc,
                p,
                sc: plan_sc,
                plan,
                dh,
                hidden_buf,
                pos_buf,
                logits_buf,
                sc_logits_buf,
            });
        }
        let build_secs = t_build0.elapsed().as_secs_f64();

        // Phase-B perf: ensure the one-time SC soft-embedding weight (Vulkan + SC only) — lazy,
        // built ONCE per session (shared across forked slots — see `SeamKv::sc_embt`) from the
        // already-dequantized `token_embd`.
        if plan_sc && sc_embt.is_none() {
            let t_embt0 = std::time::Instant::now();
            *sc_embt = Some(build_sc_embt(be, token_embd, ne, c.vocab)?);
            tracing::info!(
                "[diffusion denoise] built the SC soft-embedding weight ({:.0} MB) in {:.2}s",
                (ne * c.vocab * 2) as f64 / 1e6,
                t_embt0.elapsed().as_secs_f64()
            );
        }

        let dcache = denoise_cache.as_ref().expect("just ensured present above");

        be.upload(
            dcache.hidden_buf.as_ref(),
            bytemuck::cast_slice(&hidden_host),
        )
        .map_err(|e| anyhow!("{e}"))?;
        be.upload(dcache.pos_buf.as_ref(), bytemuck::cast_slice(&dn_positions))
            .map_err(|e| anyhow!("{e}"))?;
        let mut db = Bindings::new();
        db.bind(dcache.dh.hidden, dcache.hidden_buf.as_ref());
        db.bind(dcache.dh.positions, dcache.pos_buf.as_ref());
        bind_layer_io(
            &mut db,
            &dcache.dh,
            c.n_layer,
            rf_buf,
            &kbufs[..],
            &vbufs[..],
            &wbufs[..],
        );
        if plan_sc {
            if dyn_sc {
                // Vulkan perf: the SC input is the OTHER ping slot (this call's write target is
                // the opposite one — see `sc_ping`'s doc) — already GPU-resident, no upload. Only
                // the 4-byte temp_inv scalar moves host->device this step.
                let ping = sc_ping.as_ref().expect("allocated above");
                let read_idx = 1 - *sc_ping_write;
                let temp_inv_buf = sc_temp_inv_buf.as_ref().expect("allocated above");
                be.upload(temp_inv_buf.as_ref(), &req.temp_inv.to_le_bytes())
                    .map_err(|e| anyhow!("{e}"))?;
                db.bind(
                    dcache
                        .dh
                        .sc_logits
                        .expect("plan_sc plan declares sc_logits"),
                    ping[read_idx].as_ref(),
                );
                db.bind(
                    dcache.dh.temp_inv.expect("dyn_sc plan declares temp_inv"),
                    temp_inv_buf.as_ref(),
                );
            } else {
                // Metal: original host-premultiply-and-upload path, unchanged.
                let sc_logits_host = sc_logits_host
                    .as_ref()
                    .expect("plan_sc && !dyn_sc implies sc_logits_host is Some");
                let sc_logits_buf = dcache
                    .sc_logits_buf
                    .as_ref()
                    .expect("plan_sc && !dyn_sc plan always allocates sc_logits_buf");
                be.upload(sc_logits_buf.as_ref(), bytemuck::cast_slice(sc_logits_host))
                    .map_err(|e| anyhow!("{e}"))?;
                db.bind(
                    dcache
                        .dh
                        .sc_logits
                        .expect("plan_sc plan declares sc_logits"),
                    sc_logits_buf.as_ref(),
                );
            }
            db.bind(
                dcache.dh.sc_embt.expect("plan_sc plan declares sc_embt"),
                sc_embt.as_ref().expect("ensured present above").as_ref(),
            );
        }
        // Vulkan: the ping slot THIS call writes into — the opposite of the one just bound above
        // as SC input (when `plan_sc`), or simply the current write slot on a non-SC step. Metal/
        // CPU keep the per-plan `logits_buf`.
        let logits_out_buf: &dyn Buffer = if use_ping {
            sc_ping.as_ref().expect("allocated above")[*sc_ping_write].as_ref()
        } else {
            dcache
                .logits_buf
                .as_ref()
                .expect("non-ping path always allocates logits_buf")
                .as_ref()
        };
        db.bind(
            dcache.dh.logits.expect("denoise build has logits"),
            logits_out_buf,
        );
        let t_exec0 = std::time::Instant::now();
        be.execute(dcache.plan.as_ref(), &db)
            .map_err(|e| anyhow!("{e}"))?;
        let exec_secs = t_exec0.elapsed().as_secs_f64();

        let t_dl0 = std::time::Instant::now();
        // Perf slice 3 (docs/diffusion-gemma.md): try the GPU entropy-bound sampler reducer on
        // THIS step's freshly-written logits before falling back to the full `[cc, vocab]`
        // download — see `EbReduced`'s doc. `req.u` is `None` for CPU/Metal (they never reach
        // this branch's Vulkan-only `use_ping` path anyway) and for Vulkan callers that opt out.
        let mut reduced_now: Option<EbReduced> = None;
        if let Some(u_host) = req.u {
            if u_host.len() != cc {
                return Err(anyhow!(
                    "denoise: u length {} != {cc} (canvas rows)",
                    u_host.len()
                ));
            }
            let u_buf = be
                .alloc(cc * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(u_buf.as_ref(), bytemuck::cast_slice(u_host))
                .map_err(|e| anyhow!("{e}"))?;
            let argmax_buf = be
                .alloc(cc * 4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?;
            let entropy_buf = be
                .alloc(cc * 4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?;
            let sampled_buf = be
                .alloc(cc * 4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?;
            let ok = be
                .eb_sample_reduce(
                    logits_out_buf,
                    u_buf.as_ref(),
                    cc,
                    c.vocab,
                    req.sample_temp_inv,
                    argmax_buf.as_ref(),
                    entropy_buf.as_ref(),
                    sampled_buf.as_ref(),
                )
                .map_err(|e| anyhow!("{e}"))?;
            if ok {
                let mut argmax = vec![0u32; cc];
                be.download(argmax_buf.as_ref(), bytemuck::cast_slice_mut(&mut argmax))
                    .map_err(|e| anyhow!("{e}"))?;
                let mut entropy = vec![0f32; cc];
                be.download(entropy_buf.as_ref(), bytemuck::cast_slice_mut(&mut entropy))
                    .map_err(|e| anyhow!("{e}"))?;
                let mut sampled = vec![0u32; cc];
                be.download(sampled_buf.as_ref(), bytemuck::cast_slice_mut(&mut sampled))
                    .map_err(|e| anyhow!("{e}"))?;
                reduced_now = Some(EbReduced {
                    argmax,
                    entropy,
                    sampled,
                });
            }
        }
        if reduced_now.is_none() {
            req.out_logits.resize(cc * c.vocab, 0.0);
            be.download(logits_out_buf, bytemuck::cast_slice_mut(req.out_logits))
                .map_err(|e| anyhow!("{e}"))?;
        }
        *req.reduced = reduced_now;
        let dl_secs = t_dl0.elapsed().as_secs_f64();
        // Vulkan: flip which ping slot is "write" vs "read" for the NEXT call — this call's output
        // (just downloaded above) becomes the next call's self-conditioning input, already
        // GPU-resident (see `sc_ping`'s doc).
        if use_ping {
            *sc_ping_write = 1 - *sc_ping_write;
        }
        if time_diffusion {
            tracing::info!(
                "[diffusion denoise] sc={sc_secs:.3}s build={build_secs:.3}s exec={exec_secs:.3}s dl={dl_secs:.3}s total={:.3}s",
                sc_secs + build_secs + exec_secs + dl_secs,
            );
        }
        // `cached`/`start` were left untouched above (the canvas isn't part of the prompt/gen
        // token stream) — the prompt-KV rows 0..P stay exactly as the prior prefill call left
        // them, so the NEXT denoise call (same or different canvas) re-overwrites rows P..P+C
        // again, and a later real prefill still resumes from P.
        return Ok((
            Vec::new(),
            GenStats {
                n_prompt: 0,
                prompt_secs: sc_secs + build_secs + exec_secs + dl_secs,
                n_gen: 0,
                decode_secs: 0.0,
            },
        ));
    }

    // ── speculative VERIFY ──────────────────────────────────────────────────────────
    // One batched forward over the un-cached suffix with the LM head on EVERY row: returns
    // [m, vocab] logits (the distribution after each suffix token) and generates nothing.
    // The suffix-prefill contract doubles as the accept/rollback mechanism: the caller
    // truncates its committed token list and the next call's prefix diff overwrites the
    // stale KV rows. Dense non-E2B models only (mirrors the batched-prefill guard).
    if let Some(out_logits) = verify {
        if c.moe.is_some() || ple.is_some() {
            return Err(anyhow!("speculative verify: dense non-E2B models only"));
        }
        let vf_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
        let m = prompt.len() - start;
        let mut vf_hidden: Vec<f32> = Vec::with_capacity(m * ne);
        let token_embd = token_embd.get()?; // host embed gather → materialize the table
        for &tok in &prompt[start..] {
            let base = tok as usize * ne;
            vf_hidden.extend(token_embd[base..base + ne].iter().map(|&x| x * vf_scale));
        }
        let vf_positions: Vec<i32> = (start as i32..(start + m) as i32).collect();
        let vf_hidden_buf = be
            .alloc(m * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let vf_pos_buf = be
            .alloc(m * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let vf_logits_buf = be
            .alloc(m * c.vocab * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(vf_hidden_buf.as_ref(), bytemuck::cast_slice(&vf_hidden))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(vf_pos_buf.as_ref(), bytemuck::cast_slice(&vf_positions))
            .map_err(|e| anyhow!("{e}"))?;
        // MTP Phase 1 (issue #33): VERIFY already runs the LM head on every one of the `m` rows —
        // exactly the rows the MTP catch-up driver needs `h` for (docs/mtp.md's `process()`).
        // `h_tap` piggybacks on the SAME graph/execute, just an extra Output + download.
        let want_h = h_out.is_some();
        // Phase-4 MTP profiling (issue #33, INFR_PROF_STAGES=1): split VERIFY's own wall time into
        // graph-build / plan-compile / execute / download, and report `m` (the rows actually
        // reprocessed) + whether this call is a FULL reprefill (`start == 0` with a nonempty
        // history behind it, i.e. the qwen35 no-rewind fallback fired) vs the cheap incremental
        // suffix-only path. This is the number the MTP perf pass profiles before touching any
        // code — see mtp.rs's `generate_mtp_spec_vulkan_timed` doc on the no-rewind cost.
        let time_verify = ec.prof.stages;
        let full_reprefill = start == 0 && m > 1;
        // GPU-resident verify accept (issue #31, task #31): per-row Op::Argmax appended to the
        // batched forward — m u32 ids read back instead of the m×vocab f32 logits. Host-logits
        // fallback: grammar constraints (llguidance needs full logits), backends without the
        // multi-row kernel (Metal), and the A/B escapes (INFR_NO_GPU_ARGMAX covers all device
        // argmax; INFR_NO_GPU_MTP_ACCEPT narrows to just this path).
        let gpu_verify_ids = verify_ids.is_some()
            && constraint.is_none()
            && caps.argmax_rows
            && ec.spec.gpu_argmax
            && ec.spec.gpu_mtp_accept;
        let t_vbuild0 = std::time::Instant::now();
        let (vg, vh) = build(
            m,
            start,
            m,
            false,
            None,
            false,
            want_h,
            gpu_verify_ids,
            false,
            false,
            true, // mtp_verify: this IS the speculative-VERIFY batched forward
            None, // span: the whole model in one graph
        );
        let vbuild_secs = t_vbuild0.elapsed().as_secs_f64();
        let t_vcompile0 = std::time::Instant::now();
        let vplan = be.compile(&vg).map_err(|e| anyhow!("{e}"))?;
        let vcompile_secs = t_vcompile0.elapsed().as_secs_f64();
        let vf_h_buf = if want_h {
            Some(
                be.alloc(m * ne * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let mut vb = Bindings::new();
        vb.bind(vh.hidden, vf_hidden_buf.as_ref());
        vb.bind(vh.positions, vf_pos_buf.as_ref());
        bind_layer_io(
            &mut vb,
            &vh,
            c.n_layer,
            rf_buf,
            &kbufs[..],
            &vbufs[..],
            &wbufs[..],
        );
        vb.bind(
            vh.logits.expect("verify build has logits"),
            vf_logits_buf.as_ref(),
        );
        // The m-slot id output (gpu_verify_ids builds only) — 4 bytes/row readback.
        let vf_ids_buf = if gpu_verify_ids {
            Some(
                be.alloc(m * 4, BufferUsage::Readback)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        if let (Some(tid), Some(ib)) = (vh.tok_id, &vf_ids_buf) {
            vb.bind(tid, ib.as_ref());
        }
        if let (Some(ho), Some(hb)) = (vh.h_out, &vf_h_buf) {
            vb.bind(ho, hb.as_ref());
        }
        let t0 = std::time::Instant::now();
        be.execute(vplan.as_ref(), &vb)
            .map_err(|e| anyhow!("{e}"))?;
        let vexec_secs = t0.elapsed().as_secs_f64();
        let t_vdl0 = std::time::Instant::now();
        if let (Some(out_ids), Some(ib)) = (verify_ids, &vf_ids_buf) {
            // GPU accept path: m u32 ids down, the m×vocab logits stay in VRAM (`out_logits`
            // deliberately left EMPTY — the caller keys the fallback off that).
            out_ids.resize(m, 0);
            be.download(ib.as_ref(), bytemuck::cast_slice_mut(out_ids))
                .map_err(|e| anyhow!("{e}"))?;
        } else {
            out_logits.resize(m * c.vocab, 0.0);
            be.download(vf_logits_buf.as_ref(), bytemuck::cast_slice_mut(out_logits))
                .map_err(|e| anyhow!("{e}"))?;
        }
        if let (Some(out), Some(hb)) = (h_out.take(), &vf_h_buf) {
            out.resize(m * ne, 0.0);
            be.download(hb.as_ref(), bytemuck::cast_slice_mut(out))
                .map_err(|e| anyhow!("{e}"))?;
        }
        let vdl_secs = t_vdl0.elapsed().as_secs_f64();
        if time_verify {
            tracing::info!(
                "[mtp verify] m={m} start={start} full_reprefill={full_reprefill} \
                 build={:.1}ms compile={:.1}ms exec={:.1}ms dl={:.1}ms total={:.1}ms",
                vbuild_secs * 1e3,
                vcompile_secs * 1e3,
                vexec_secs * 1e3,
                vdl_secs * 1e3,
                (vbuild_secs + vcompile_secs + vexec_secs + vdl_secs) * 1e3,
            );
        }
        cached.extend_from_slice(&prompt[start..]);
        return Ok((
            Vec::new(),
            GenStats {
                n_prompt: m,
                prompt_secs: t0.elapsed().as_secs_f64(),
                n_gen: 0,
                decode_secs: 0.0,
            },
        ));
    }

    // ── drive ───────────────────────────────────────────────────────────────────────
    // The per-call decode IO buffers. These are `be.alloc`s, and on Vulkan an `alloc` zero-fills
    // through a one-shot command buffer — i.e. it RECORDS, so it needs the baton exactly like a
    // step does (see `StepGate`: the command pool is externally synchronised, and the backend hands
    // its handle out from under the mutex). Scoped so the baton is released before the prefill loop
    // below, which takes its own per-chunk turn.
    let (tok_id_buf, dec_ids_buf, u_buf) = {
        let _gp = req.and_then(|r| r.gate_pass());
        let tok_id_buf = be
            .alloc(4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        // GPU embed gather: the decode loop's 4-byte token-id input (replaces the n_embd*4 host
        // embed + hidden upload when `gpu_embed`).
        let dec_ids_buf = be
            .alloc(4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        // GPU stochastic sampling: the host-drawn uniform(s) for `Op::Sample`'s `u` Input — a 64-slot
        // ring (mirrors the chained id-log ring), indexed by `pos & 63` in BOTH the per-token and
        // chained paths so a record-once recording can be replayed either way (see adapter.rs
        // `Recorder::sample_topk_chain`). Sized 64*4 unconditionally: the same buffer is bound whether
        // or not this decode ever actually chains.
        let u_buf = be
            .alloc(64 * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        (tok_id_buf, dec_ids_buf, u_buf)
    };
    // Host-side mirror of `u_buf`'s 64 slots. `Backend::upload` has no partial-buffer/offset
    // form, so setting one slot re-uploads the whole 256 bytes from this mirror — negligible cost.
    // Positions monotonically increase and are consumed before their slot wraps (mod 64), so no
    // read ever observes a stale draw from 64 tokens ago within the active window.
    let mut u_ring_host = [0f32; 64];
    // Chained decode: when the graph both GATHERS from an id input and SAMPLES an id output,
    // bind them to the SAME buffer — within one iteration the gather reads before the sampler
    // writes, and across chained iterations the sampler's id feeds the next gather directly
    // on-device. Per-token mode is unaffected (the host re-uploads the fed id every step and
    // reads the sampled one back from the same slot).
    let id_out: &dyn Buffer = if gpu_embed {
        dec_ids_buf.as_ref()
    } else {
        tok_id_buf.as_ref()
    };
    let embed_scale = if gemma { (ne as f32).sqrt() } else { 1.0 };
    let mut out = Vec::new();
    let mut cur = prompt.to_vec();
    let mut logits = vec![0f32; c.vocab];
    // INFR_PROF_STAGES=1: report prompt-ingest + decode tok/s to stderr (CPU perf iteration).
    let prof = ec.prof.stages;
    let mut prompt_t = std::time::Duration::ZERO;
    let mut decode_t = std::time::Duration::ZERO;
    let mut decode_n = 0usize;
    // `prof.stages` (INFR_PROF_STAGES): split decode per-token wall time into host setup (build
    // graph + compile + bind) vs execute (record + submit + GPU + wait) to guide the
    // record-once-replay decision. Hoisted here, ABOVE the loop — the old read was a `getenv` on
    // EVERY decode step (R6/§10.9).
    let prof_dec = ec.prof.stages;
    let mut dec_setup = std::time::Duration::ZERO;
    let mut dec_exec = std::time::Duration::ZERO;

    // ── batched prefill (dense + adapter-covered MoE; non-E2B models only) ────────────────────
    // Process all-but-the-last prompt tokens in a single graph execution: each Op::Linear runs
    // m=(N-1) activations against every weight row in parallel (O(out_f) rayon tasks, N-1 dots
    // each), reading each weight row ONCE and reusing it across all tokens. This fills the KV
    // cache for positions 0..N-2. The last prompt token is left for the normal decode loop so
    // that the "decode" stats (tok/s) remain meaningful and the first generated token is sampled
    // in the canonical way.
    //
    // Guard: E2B/gemma4 requires a per-(token,layer) host-side input vector that is computed in
    // the per-step loop, so it falls through to the original token-by-token loop below unchanged.
    // Batched MoE prefill needs the adapter's GPU-routed expert path: gate/up AND down each
    // independently in `infr_core::tensor::MOE_MMQ_DTYPES` (split gate/up, what
    // qwen3moe/qwen35moe/llama4 ship, or fused gate_up, diffusion-gemma's/gemma-4-MoE's
    // `ffn_gate_up_exps`) — Q5_0 is what the shipped diffusiongemma-26B-A4B-it-GGUF's down banks
    // use; Q5_1 is what the shipped gemma-4-26B-A4B-it-GGUF's down banks use (29/30 layers);
    // unsloth-dynamic Qwen3.6-MoE (UD) quants mix Q5_K/Q6_K/IQ4_XS into gate/up/down banks across
    // layers; Q2_K/Q3_K is Llama-4-Scout's shipped gate/up (Q2_K) and down (Q3_K); IQ2_S/IQ3_S is
    // the UD-IQ3_S file's expert pair (grid-codebook mmq via shared-LUT staging). The remaining
    // codebook quants (IQ1_*/IQ2_XXS/IQ2_XS/IQ3_XXS/TQ*) have no dp4a-mmq kernel (no shipped MoE
    // GGUF uses them for expert banks — see MOE_MMQ_DTYPES's EXCLUSIONS) and keep the per-token
    // loop.
    // `MOE_MMQ_DTYPES` is the SINGLE SOURCE OF TRUTH this closure and the Vulkan adapter's batched
    // `Op::MoeFfn` gate (its `mmq_ok`) both derive from — a mismatch either silently falls back to
    // per-token prefill or compiles a graph the adapter rejects; `moe_mmq_drift_test` (in
    // infr-vulkan, since only that crate links both dtype sets at test time) guards it. NOTE:
    // accepting Q2_K/Q3_K here also flips paged models (Scout: 37GB Q2_K/Q3_K experts on a 24GB
    // card) onto the batched-chunk `Op::MoeFfn` construction — the Vulkan adapter's paged-buffer
    // interception (`execute_static`, ahead of `lower_op`'s batched/small-m split) routes every
    // paged MoeFfn through `execute_paged_moe`, whose own batched arm runs the same
    // bucket-scatter → dp4a mmq expert-GEMM pipeline against the pager arena
    // (`matmul_mmq_experts_paged`): one residency readback per layer per ubatch chunk instead of
    // one per token, with the mmq GEMM's cross-token expert-bank reuse. (Routing the batched
    // chunk through the paged id-GEMV instead was measured SLOWER than per-token — 14.7 vs
    // 27.6 t/s pp512 — the giant uncoalesced multi-row GEMV loses more to cache-hostile weight
    // re-reads than it saves in readbacks.)
    // `moe_batched_ok` (this `MOE_MMQ_DTYPES` eligibility scan) is a session-stable derivation —
    // computed once in `session_stable` and read here off `stable`.
    let decode_start = if prompt.len() - start > 2 && (c.moe.is_none() || moe_batched_ok) {
        // Batch-prefill the un-cached suffix, all but the last prompt token (positions
        // start..plen-1; rows 0..start are reused from the session cache) — in UBATCH CHUNKS.
        // One giant graph would scale the internal activation/attention scratch with the whole
        // prompt (an 8B p8000 prefill built a multi-second single submission whose tail work
        // tripped the amdgpu ring watchdog → device lost) and bakes a multi-second unpreemptible
        // submit; fixed-size chunks bound both, exactly like the bespoke path's ubatches.
        // Shared reader (`crate::seam::ubatch_rows`): the SWA ring sizing must cover exactly
        // this chunk height — see `kv_rows`' correctness bound.
        // Prefill vs in-flight decodes (`infr serve --parallel N`): the chunk is the unit of GPU
        // ownership, so a chunk is exactly how long a newly-admitted request's prefill can stall
        // everyone else's decode. The default ubatch (INFR_UBATCH, 1024 rows) is ~100ms+ of
        // unpreemptible GPU on a 14B — enough to visibly hitch 3 other streams. Under a gate we
        // therefore cap the chunk (INFR_UBATCH_PARALLEL, default 256 rows), yield the baton between
        // chunks, and let the round-robin interleave prefill chunks with the other sequences' decode
        // steps. This is llama.cpp's "chunked prefill interleaved with decode" without the shared
        // batch: same starvation bound, no batching win. A sole request (`req` None, or `-np 1`)
        // keeps the full 1024-row chunk — prefill throughput is UNCHANGED there.
        let ubatch: usize = if req.is_some_and(crate::sampling::RequestCtx::shares_gpu) {
            crate::seam::ubatch_rows(ec).min(crate::seam::ubatch_rows_parallel(ec))
        } else {
            crate::seam::ubatch_rows(ec)
        };
        let pf_end = prompt.len() - 1;
        // ── prefill work list: (layer span, chunk), in EXECUTION order ───────────────────────
        // Chunk-major is one whole-model span per chunk, so every chunk drags the entire weight
        // set past the pager again — free when the weights are resident, and the whole prefill
        // bill when they stream. Layer-major inverts the nesting: every chunk passes through
        // layer L before any chunk reaches L+1, so the model is swept ONCE per prompt at the same
        // chunk-sized dispatches (a taller chunk reaches the same I/O and bakes a submit long
        // enough to trip the GPU watchdog — see the chunk comment above). See
        // `crate::seam::layer_major_prefill` for the gate and the measurement behind it.
        //
        // Correctness is the same causal argument either way: a chunk's attention reads its own
        // layer's KV rows for positions below its own, and chunks run in ASCENDING order inside
        // each layer, so every earlier position is already written when a chunk reaches it. The
        // SWA ring is untouched by the reorder — its bound is per layer and per dispatch
        // ("window + one chunk", see `kv_rows`), and each layer's ring still sees exactly the same
        // ascending sequence of writes it saw chunk-major.
        let layer_major = crate::seam::layer_major_prefill(ec, &caps, be.dense_paged(), !e2b);
        let spans: Vec<std::ops::Range<usize>> = if layer_major {
            (0..c.n_layer).map(|l| l..l + 1).collect()
        } else {
            std::iter::once(0..c.n_layer).collect()
        };
        let chunks: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            let mut cs = start;
            while cs < pf_end {
                let ce = (cs + ubatch).min(pf_end);
                v.push((cs, ce));
                cs = ce;
            }
            v
        };
        // A chunk's uploaded inputs and its residual stream, materialized on first use and
        // dropped after the chunk's LAST span. Chunk-major (one span) therefore holds exactly one
        // chunk's buffers at a time, as it always did; layer-major holds every chunk's, which is
        // the activation cost `crate::seam::layer_major_act_bytes` prices.
        struct PfChunk {
            m: usize,
            /// Token ids (gpu_embed) or the host-embedded f32 rows.
            input: Box<dyn Buffer>,
            /// The residual stream, when `input` holds ids and cannot serve as one.
            resid: Option<Box<dyn Buffer>>,
            pos: Box<dyn Buffer>,
            /// gemma4-E2B per-layer token rows.
            ipl: Option<Box<dyn Buffer>>,
        }
        let mut live: Vec<Option<PfChunk>> = (0..chunks.len()).map(|_| None).collect();
        for (si, span) in spans.iter().enumerate() {
            for (ci, &(cstart, cend)) in chunks.iter().enumerate() {
                // Shutdown (SIGINT/SIGTERM) or a per-request abort: do not START another chunk.
                // The chunk that was already in flight is not cut off — the backend drains it and
                // returns `Error::Aborted` from `be.execute` below, which lands here as the same
                // bail. Nothing useful was produced (a half-filled KV cache is not a generation),
                // so this is an error and not a partial success; the CLI turns the latched signal
                // into the conventional 130/143 exit status regardless of what this call returns.
                if crate::sampling::abort_requested(req) {
                    anyhow::bail!("aborted: shutdown requested");
                }
                // One dispatch = one turn on the GPU. Dropped at the end of the iteration, handing
                // the baton to whichever sequence has been waiting longest.
                let _gp = req.and_then(|r| r.gate_pass());
                let pf_m = cend - cstart;
                if live[ci].is_none() {
                    // GPU embed gather: upload the chunk's token IDS (4*pf_m bytes) — the graph's
                    // Op::EmbedGather dequantizes the rows on-device. Host-embed fallback keeps
                    // the old f32 rows upload (4*n_embd*pf_m bytes).
                    let input = if gpu_embed {
                        let ids: Vec<i32> =
                            prompt[cstart..cend].iter().map(|&t| t as i32).collect();
                        let b = be
                            .alloc(pf_m * 4, BufferUsage::Staging)
                            .map_err(|e| anyhow!("{e}"))?;
                        be.upload(b.as_ref(), bytemuck::cast_slice(&ids))
                            .map_err(|e| anyhow!("{e}"))?;
                        b
                    } else {
                        let mut pf_hidden: Vec<f32> = Vec::with_capacity(pf_m * ne);
                        let token_embd = token_embd.get()?; // host embed gather → materialize
                        for &tok in &prompt[cstart..cend] {
                            let base = tok as usize * ne;
                            pf_hidden.extend(
                                token_embd[base..base + ne].iter().map(|&x| x * embed_scale),
                            );
                        }
                        let b = be
                            .alloc(pf_m * ne * 4, BufferUsage::Staging)
                            .map_err(|e| anyhow!("{e}"))?;
                        be.upload(b.as_ref(), bytemuck::cast_slice(&pf_hidden))
                            .map_err(|e| anyhow!("{e}"))?;
                        b
                    };
                    // The residual stream. A layer-span build carries `hidden` in a CALLER-owned
                    // buffer rather than graph scratch, so the chunk owns one: on the host-embed
                    // path the uploaded rows already ARE the layer stack's input and `input`
                    // serves, while the gpu_embed path uploads ids there and the in-graph gather
                    // needs somewhere to write. Sized EXACTLY like the `[batch, n_embd]` handle it
                    // binds — the interpreters' write-back is a length-checked `copy_from_slice`
                    // against the declared numel, and the host-embed path has always bound this
                    // shape, so nothing writes past it.
                    let resid = if gpu_embed {
                        Some(
                            be.alloc(pf_m * ne * 4, BufferUsage::Activations)
                                .map_err(|e| anyhow!("{e}"))?,
                        )
                    } else {
                        None
                    };
                    // Absolute positions [cstart, ..., cend-1].
                    let pf_positions: Vec<i32> = (cstart as i32..cend as i32).collect();
                    let pos = be
                        .alloc(pf_m * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?;
                    be.upload(pos.as_ref(), bytemuck::cast_slice(&pf_positions))
                        .map_err(|e| anyhow!("{e}"))?;
                    // gemma4 E2B: the chunk's per-layer TOKEN embedding rows (gather+dequant only
                    // — the model_proj GEMV/RMSNorm/combine run as GPU graph ops in the `build`
                    // prologue).
                    let ipl = if let (Some(ple), false) = (ple, gpu_ple) {
                        let rows = e2b_ipl_rows(g, ple, &prompt[cstart..cend])?;
                        let b = be
                            .alloc(rows.len() * 4, BufferUsage::Staging)
                            .map_err(|e| anyhow!("{e}"))?;
                        be.upload(b.as_ref(), bytemuck::cast_slice(&rows))
                            .map_err(|e| anyhow!("{e}"))?;
                        Some(b)
                    } else {
                        None
                    };
                    live[ci] = Some(PfChunk {
                        m: pf_m,
                        input,
                        resid,
                        pos,
                        ipl,
                    });
                }
                let ch = live[ci]
                    .as_ref()
                    .expect("the chunk's buffers were just materialized");
                let pf_t0 = std::time::Instant::now();
                // HEADLESS build (`logits_rows == 0`, task #27): nothing ever consumes a prefill
                // chunk's logits — the decode loop below feeds the LAST prompt token itself and
                // samples from its own fresh logits — so the LM-head tail (whole-chunk
                // output_norm, last-row Copy, vocab-wide Linear, Softcap) is skipped per chunk. On
                // a 262k-vocab model that drops a vocab×n_embd GEMV + a [chunk, n_embd] RmsNorm
                // per chunk.
                // MTP h-tap gap (Phase 2 TODO, docs/mtp.md): the chunked BATCHED-PREFILL path
                // never taps `h`. The MTP catch-up driver needs `h` for EVERY prefill row; wiring
                // that requires this path to carry `logits_rows == pf_m` on demand, which Phase 2
                // will add alongside the actual head forward.
                let (pf_g, pf_h) = build(
                    ch.m,
                    cstart,
                    0,
                    false,
                    None,
                    false,
                    false,
                    false,
                    false,
                    // The token-id input + in-graph gather belong to the span that STARTS the
                    // stack; a later span reads the residual stream that one left behind.
                    gpu_embed && span.start == 0,
                    false, // mtp_verify: ordinary chunked prefill, not MTP verify
                    Some(span.clone()),
                );
                let t_build = pf_t0.elapsed();
                let pf_plan = be.compile(&pf_g).map_err(|e| anyhow!("{e}"))?;
                let t_compile = pf_t0.elapsed();
                let mut pf_b = Bindings::new();
                if let Some(ids) = pf_h.tok_ids {
                    pf_b.bind(ids, ch.input.as_ref());
                }
                pf_b.bind(
                    pf_h.hidden,
                    ch.resid.as_deref().unwrap_or(ch.input.as_ref()),
                );
                pf_b.bind(pf_h.positions, ch.pos.as_ref());
                if let (Some(pid), Some(ib)) = (pf_h.pl_tok_in, &ch.ipl) {
                    pf_b.bind(pid, ib.as_ref());
                }
                // gemma4's proportional-RoPE divisors + the K/V caches + the weights
                // (`bind_layer_io`): without the `rope_freqs` bind the batched graph has a live
                // unbound Input and panics.
                bind_layer_io(
                    &mut pf_b,
                    &pf_h,
                    c.n_layer,
                    rf_buf,
                    &kbufs[..],
                    &vbufs[..],
                    &wbufs[..],
                );
                debug_assert!(
                    pf_h.logits.is_none(),
                    "headless prefill build has no logits"
                );
                be.execute(pf_plan.as_ref(), &pf_b)
                    .map_err(|e| anyhow!("{e}"))?;
                // INFR_PROF_STAGES: split the per-dispatch prefill wall time into host graph
                // build, plan compile, and execute (record + submit + GPU) — where a small-batch
                // chunk's fixed cost lives decides whether to attack recording or kernels. `l` is
                // the layer span, the whole model unless this is a layer-major prefill.
                if ec.prof.stages {
                    tracing::info!(
                        "[pf prof] m={} l={}..{} build={:.1}ms compile={:.1}ms execute={:.1}ms",
                        ch.m,
                        span.start,
                        span.end,
                        t_build.as_secs_f64() * 1e3,
                        (t_compile - t_build).as_secs_f64() * 1e3,
                        (pf_t0.elapsed() - t_compile).as_secs_f64() * 1e3,
                    );
                }
                prompt_t += pf_t0.elapsed();
                // Last span for this chunk: its uploads and its residual stream are dead.
                if si + 1 == spans.len() {
                    live[ci] = None;
                }
            }
        }

        // KV rows are now filled through position plen-2; the last prompt token is handled by
        // the decode loop below (writes its KV, produces the logits the first sample uses).
        pf_end
    } else {
        start // fall through to per-token loop for MoE / E2B / short suffixes
    };

    // Record-once decode: for an eligible decode on a backend that supports replay (the Vulkan
    // seam), build+compile+bind ONE plan here and reuse it across the whole decode loop. The
    // adapter records the graph once and replays it per token, reading `pos` from the bound
    // positions buffer + a params SSBO — so the baked pos=0 here is irrelevant, and the per-token
    // host cost drops to just the emb/pos (+ E2B ipl) uploads. The gate mirrors the adapter's
    // graph eligibility: every dense arch replays — qk-norm (qwen3), the gemma family (SWA
    // windows + scale via push constants, freq_factors via qk_norm_rope_dyn_ff, V-norm/Softcap/
    // Scale are pos-independent), llama (f16-out interleaved Rope via rope_f16_dyn), MoE. Backends
    // without `decode_replay` (CPU interpreter, which reads the baked `pos`) and every ineligible
    // model keep rebuilding + recompiling per token below.
    // INFR_SEAM_NO_REPLAY=1 forces per-token rebuild (the adapter's static path) — slower, but
    // INFR_PROF_OPS per-op GPU timestamps work there (the replay path can't report them).
    // This gate MUST stay a strict subset of the adapter's `decode_eligible` — the plan below
    // bakes pos=0/kv_len=1, which is only correct when the adapter replays it (dyn kernels read
    // the live pos/kv_len); an ineligible graph would silently run the static path with the baked
    // values. Hence the per-layer head-dim mirror of the adapter's Attention check.
    // llama (no qk-norm) replays too — its f16-out Rope has a dyn kernel — but only without
    // freq_factors (the standalone Rope kernel has no ff binding; gemma4's ff rides QkNormRope).
    // A paged MoE model (see `Backend::moe_paged`'s doc) forces the static per-execute path: the
    // adapter's own `execute`/`execute_chain` already refuse to replay one (belt-and-suspenders,
    // the actual correctness guarantee), but skipping `dyn_replay` here too avoids building the
    // (then-never-used) replay plan's persistent scratch/self-advancing params machinery at all.
    // Dense layer streaming forces static per-execute decode for the same replay-can't-express-it
    // reason (per-token ring staging + per-slot weight offsets — see `Backend::dense_paged`).
    let dyn_replay = caps.decode_replay
        && !be.moe_paged()
        && !be.dense_paged()
        && !ec.kernels.vulkan.no_replay
        // DiffusionGemma graphs opt out of the replay tape entirely (`Graph::no_decode_replay`,
        // set in `build` above — the adapter's `decode_eligible` rejects them, this mirror just
        // skips building the then-unused replay plan). Keeps this gate a strict subset of the
        // adapter's eligibility.
        && !c.diffusion_gemma
        && (qk_norm || stable.rope_freqs.is_none())
        // Quantized/dense-alt KV caches force the per-execute STATIC decode (see the adapter's
        // `decode_eligible`: the low-bit block quants / bf16 / f32 / turbo ride a dequant→f16
        // prepass with a standalone WriteKv that has no dyn kernel) — EXCEPT coupled Q8_0
        // (K==V==Q8), which replays natively (store_q8_dyn write + the planar-Q8 dyn attention
        // read). A DECOUPLED Q8 side still forces static (the dyn q8 kernels dequant both sides).
        // Must mirror the adapter's rejection so this gate stays a strict subset — else the loop
        // bakes pos=0 for a static run.
        && ((k_fmt == DType::Q8_0 && v_fmt == DType::Q8_0)
            || (!kv_forces_static(k_fmt) && !kv_forces_static(v_fmt)))
        && (0..c.n_layer)
            .all(|l| c.layer_head_dim(l).is_multiple_of(4) && c.layer_head_dim(l) <= 512)
        // MTP h-tap (Phase 1, issue #33): the replay tape binds a FIXED set of tensors once: an
        // h-tap request changes the graph shape (an extra Output + Copy) per-call based on
        // whether THIS position is the one being sampled, which the static replay tape can't
        // express. Force the ordinary per-token rebuild path below when a caller wants the tap —
        // slower, but this is a validation-only hook (see `h_out`'s doc), never a hot path.
        && h_out.is_none();
    let ro = if dyn_replay {
        // `compile` builds pipelines and records the replay tape into the shared command pool —
        // GPU work, so it takes a turn like any step.
        let _gp = req.and_then(|r| r.gate_pass());
        let (g, h) = build(
            1, 0, 1, false, None, false, false, gpu_argmax, gpu_sample, gpu_embed,
            false, // mtp_verify: ordinary per-token decode, not MTP verify
            None,  // span: the whole model in one graph
        );
        let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
        let mut b = Bindings::new();
        match h.tok_ids {
            Some(ids) => {
                b.bind(ids, dec_ids_buf.as_ref());
            }
            None => {
                b.bind(h.hidden, hidden_buf.as_ref());
            }
        }
        b.bind(h.positions, pos_buf.as_ref());
        if let (Some(pid), Some(ib)) = (h.pl_tok_in, &ipl_buf) {
            b.bind(pid, ib.as_ref());
        }
        bind_layer_io(
            &mut b,
            &h,
            c.n_layer,
            rf_buf,
            &kbufs[..],
            &vbufs[..],
            &wbufs[..],
        );
        b.bind(
            h.logits.expect("decode build has logits"),
            logits_buf.as_ref(),
        );
        if let Some(tid) = h.tok_id {
            b.bind(tid, id_out);
        }
        if let Some(uin) = h.u_in {
            b.bind(uin, u_buf.as_ref());
        }
        Some((plan, b))
    } else {
        None
    };

    // INFR_IGNORE_EOS=1 (benchmarks): decode the full requested count — a model that emits EOS
    // instantly on a dummy context (gemma at depth) otherwise "finishes" 64 tokens in one step
    // and the reported tok/s is fiction. llama-bench ignores EOS the same way.
    let ignore_eos = ec.sampling.ignore_eos;
    // Chained decode (Vulkan): run N decode iterations in ONE submission — the sampled id feeds
    // the next iteration's embed gather on-device (shared `id_out` slot), params self-advance,
    // and the N ids come back from the replay's id ring in one readback. Falls back to the
    // per-token path whenever any step needs host work (grammar, logits_out, temp sampling's
    // per-step uniform) or the backend declines. INFR_DECODE_CHAIN sets N (default 8, 0/1 off).
    let chain_n: usize = ec.spec.decode_chain;
    // gemma4-E2B can't chain: its per-layer token-embedding rows (`ipl_buf`) are host-gathered
    // per FED token — chained iterations 2..n would read the first token's stale rows. Lifting
    // this needs the per-layer table resident + gathered on-device (task #28 follow-up).
    // Temperature sampling chains too (`gpu_sample`): `Op::Sample`'s `u` ring lets the SAME
    // recording replay chained — see adapter.rs `Recorder::sample_topk_chain`. The `INFR_NO_GPU_POS`
    // exclusion matters only here (not for `gpu_argmax`): a decline mid-flight would have already
    // drawn+uploaded this chunk's uniforms from the host RNG stream, desyncing it from the
    // per-token fallback's draws for the SAME tokens (argmax draws nothing, so it's harmless there).
    let can_chain = gpu_embed
        && (gpu_argmax || gpu_sample)
        && ro.is_some()
        && chain_n >= 2
        && ipl_buf.is_none()
        && ec.kernels.vulkan.gpu_pos;
    let end = prompt.len() + max_new;
    let mut pos = decode_start;
    // Highest absolute position whose KV row was actually WRITTEN by a kept token this call — the
    // single source of truth for the teardown's `cached` (see `resident_after_gen`). Seeded to the
    // last already-resident position: rows `0..decode_start` are live from the session-cache reuse
    // (`start`) plus the batched prefill (which fills through `plen-2`), so the highest is
    // `decode_start - 1` (`None` when nothing is resident yet). Bumped after every executed step.
    let mut last_written: Option<usize> = decode_start.checked_sub(1);
    while pos < end {
        // `max_new == 0` (prefill-only: bench pp, session cache warming) must still FEED the
        // prompt: models without a batched-prefill path (MoE with non-Q4_K expert banks, E2B
        // short suffixes) do their entire prefill in this loop — breaking before the prompt is
        // consumed skips their KV fill and reports a zero prompt time (bench pp printed 512e9
        // t/s for qwen35moe UD quants). Only break once every prompt position but the frontier
        // has been processed (the frontier token stays un-fed at max_new == 0, matching the
        // batched-prefill path's plen-1 contract).
        if out.len() >= max_new && pos + 1 >= prompt.len() {
            break;
        }
        // Shutdown (SIGINT/SIGTERM), polled at the TOP of the loop as well as at the existing
        // bottom-of-loop stop checks. The bottom checks only run on a step that SAMPLED, so the
        // per-token PROMPT-feed steps this loop also does (MoE / gemma-E2B / short suffixes, which
        // have no batched-prefill path) had no stop check at all — a Ctrl-C during one of those
        // prefills would have fed the whole prompt, one submit at a time, before anything noticed.
        // Whatever partial output was already streamed stands; the caller reports it.
        if crate::sampling::abort_requested(req) {
            break;
        }
        // ONE turn on the GPU per loop iteration — a single decode step, or one chained chunk of
        // `chain_n` steps (the `can_chain` branch below submits them as one recording). Dropped at
        // the end of the iteration (including on `continue`, `break`, and `?`), handing the baton
        // to the longest-waiting sequence. THIS is the token-granularity round-robin: no request
        // can be head-of-line blocked behind another's whole generation, only behind one step of it.
        // `req` None (run/bench/tests) constructs nothing.
        let _gp = req.and_then(|r| r.gate_pass());
        if can_chain && pos + 1 >= prompt.len() && pos + 1 == cur.len() && logits_out.is_none() {
            // Clamped by the backend's watchdog budget too: a chain is one submit of `n` decode
            // steps, and on a slow device that submit has to stay short (see
            // `Backend::max_decode_chain`). Clamped HERE, before the per-step sampling uniforms
            // below are drawn, so the RNG stream advances by exactly the steps we run.
            let n = chain_n
                .min(max_new - out.len())
                .min(64)
                .min(be.max_decode_chain());
            if n >= 2 {
                let step_t0 = std::time::Instant::now();
                // Seed the shared id slot with the token to feed (the previous chunk's last
                // sampled id is already there device-side, but forced/first tokens aren't) and
                // the position (read ONCE, at replay-record time — the chain may be the very
                // first decode step, before the per-token path ever wrote pos_buf).
                be.upload(
                    dec_ids_buf.as_ref(),
                    bytemuck::cast_slice(&[cur[pos] as i32]),
                )
                .map_err(|e| anyhow!("{e}"))?;
                be.upload(pos_buf.as_ref(), bytemuck::cast_slice(&[pos as i32]))
                    .map_err(|e| anyhow!("{e}"))?;
                // GPU stochastic sampling: draw this chunk's `n` uniforms from the SAME xorshift
                // stream the per-token path consumes (one per frontier token, in order) and seed
                // the ring slots the chained replay will read. Replay `j` (1-indexed within this
                // chunk) feeds the token at sequence position `pos+j-1` (replay 1 feeds THIS
                // iteration's frontier token, `cur[pos]`, just uploaded above) and its
                // `params_advance` lands on the adapter's `p0+j` where `p0 = pos-1` (the runner's
                // `pos` tracks one AHEAD of the device's pre-call `params[0]`, matching
                // `execute_chain`'s own `p0+1..=p0+n` accounting) — so replay `j`'s ring slot is
                // `(p0+j)&63 = (pos+j-1)&63`, i.e. slots `pos .. pos+n-1` for `i in 0..n`.
                if gpu_sample {
                    for i in 0..n {
                        u_ring_host[(pos + i) & 63] = crate::sampling::next_uniform(&mut rng);
                    }
                    be.upload(u_buf.as_ref(), bytemuck::cast_slice(&u_ring_host))
                        .map_err(|e| anyhow!("{e}"))?;
                }
                let (plan, b) = ro.as_ref().expect("can_chain implies record-once");
                if let Some(ids) = be
                    .execute_chain(plan.as_ref(), b, n)
                    .map_err(|e| anyhow!("{e}"))?
                {
                    let mut stop = false;
                    let mut fed = 0usize;
                    for &id in &ids {
                        out.push(id);
                        decode_n += 1;
                        fed += 1;
                        let is_eos = !ignore_eos && (c.eos_ids.contains(&id) || id == c.eos);
                        if is_eos {
                            stop = true;
                            break;
                        }
                        on_token(id);
                        cur.push(id);
                        // `abort_requested` was NOT polled here before (pre-existing bug, reported):
                        // the chained path is the DEFAULT decode fast path (greedy + gpu_embed), so
                        // an `infr serve` request with a `stop` sequence kept generating all the way
                        // to `max_tokens` after its stop had already fired. The client's TEXT was
                        // right (StopMatcher suppresses emission after a hit) but the server burned
                        // the whole budget on tokens nobody would ever see — and, now, held a slot
                        // other requests were queued behind. Same shape as the EOS/max_new checks.
                        if out.len() >= max_new || crate::sampling::abort_requested(req) {
                            stop = true;
                            break;
                        }
                    }
                    decode_t += step_t0.elapsed();
                    // The chain fed positions `pos..pos+fed-1` with kept tokens (`cur[pos]` then
                    // each accepted sampled id); the final sampled id was pushed but never fed, so
                    // its row isn't written — `pos + fed - 1` is the last materialized position.
                    // Matches the old `prompt ++ out[..out.len()-1]` teardown on every chain case.
                    if fed > 0 {
                        last_written = Some(pos + fed - 1);
                    }
                    if stop {
                        break;
                    }
                    pos += fed;
                    continue;
                }
                // Backend declined (e.g. adapter fell back to static) — per-token path below.
            }
        }
        let step_t0 = std::time::Instant::now();
        let tok = cur[pos] as usize;
        if gpu_embed {
            // GPU embed gather: feed the 4-byte token id — the device dequantizes the row.
            be.upload(dec_ids_buf.as_ref(), bytemuck::cast_slice(&[tok as i32]))
                .map_err(|e| anyhow!("{e}"))?;
        } else {
            // embed (gemma scales by √n_embd; qwen3/llama identity). At the identity scale the
            // table slice is already the row to upload — hand it straight to the backend rather
            // than allocating a throwaway `Vec<f32>` per token to copy it.
            let row = &token_embd.get()?[tok * ne..tok * ne + ne];
            if embed_scale == 1.0 {
                be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(row))
                    .map_err(|e| anyhow!("{e}"))?;
            } else {
                let emb: Vec<f32> = row.iter().map(|&x| x * embed_scale).collect();
                be.upload(hidden_buf.as_ref(), bytemuck::cast_slice(&emb))
                    .map_err(|e| anyhow!("{e}"))?;
            }
        }
        be.upload(pos_buf.as_ref(), bytemuck::cast_slice(&[pos as i32]))
            .map_err(|e| anyhow!("{e}"))?;

        // gemma4 E2B host ipl path: this token's per-layer TOKEN embedding row (gather+dequant
        // only). `ipl_buf` is None under `gpu_ple` — the graph gathers on-device.
        if let (Some(ple), Some(ipl_buf)) = (ple, &ipl_buf) {
            let ipl = e2b_ipl_rows(g, ple, &[tok as u32])?;
            be.upload(ipl_buf.as_ref(), bytemuck::cast_slice(&ipl))
                .map_err(|e| anyhow!("{e}"))?;
        }

        // Only sample once we're past the prompt (decode position = last prompt token onward).
        let is_decode = pos + 1 >= prompt.len();
        // GPU stochastic sampling: draw this step's uniform from the SAME xorshift stream the
        // host sampler would consume. Record-once (`ro.is_some()`, `RopeMode::Dynamic`) seeds the
        // ring slot `pos & 63` — the in-graph `Op::Sample` reads `u_buf[params[0] & 63]`, matching
        // the chained fast-path's geometry above, since this iteration's dispatch advances
        // `params[0]` to `pos` whether it's replayed singly here or folded into an
        // `execute_chain` chunk (see `Recorder::sample_topk_chain`). The classic per-execute
        // rebuild (`ro` is `None`, `RopeMode::Static`) has no ring — its kernel always reads
        // `u_buf[0]`, so pin the write there. Only frontier rows sample, matching the host path's
        // rng consumption exactly.
        if gpu_sample && is_decode && pos + 1 == cur.len() {
            let slot = if ro.is_some() { pos & 63 } else { 0 };
            u_ring_host[slot] = crate::sampling::next_uniform(&mut rng);
            be.upload(u_buf.as_ref(), bytemuck::cast_slice(&u_ring_host))
                .map_err(|e| anyhow!("{e}"))?;
        }
        // Sample only at the FRONTIER (this position's token is the newest one fed). A constrained
        // step can emit several deterministically-forced tokens at once — they're queued onto
        // `cur` and the following iterations just feed them (no sampling) until the frontier.
        let at_frontier = pos + 1 == cur.len();
        // MTP Phase 1 h-tap (issue #33): only the frontier row is ever sampled/downloaded as
        // logits — same row `h_out` (when requested) captures. `dyn_replay` already excludes
        // `h_out.is_some()` (see its doc), so this is always the rebuild (`else`) branch below.
        let want_h = h_out.is_some() && is_decode && at_frontier;
        let h_tap_buf = if want_h {
            Some(
                be.alloc(ne * 4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let t_setup = std::time::Instant::now();
        let (setup_el, exec_el);
        if let Some((plan, b)) = &ro {
            // Record-once path: reuse the single compiled plan + bindings (no per-token rebuild).
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        } else {
            let (g, h) = build(
                1, pos, 1, false, None, false, want_h, gpu_argmax, gpu_sample, gpu_embed,
                false, // mtp_verify: ordinary per-token decode, not MTP verify
                None,  // span: the whole model in one graph
            );
            let plan = be.compile(&g).map_err(|e| anyhow!("{e}"))?;
            let mut b = Bindings::new();
            match h.tok_ids {
                Some(ids) => {
                    b.bind(ids, dec_ids_buf.as_ref());
                }
                None => {
                    b.bind(h.hidden, hidden_buf.as_ref());
                }
            }
            b.bind(h.positions, pos_buf.as_ref());
            if let (Some(pid), Some(ib)) = (h.pl_tok_in, &ipl_buf) {
                b.bind(pid, ib.as_ref());
            }
            bind_layer_io(
                &mut b,
                &h,
                c.n_layer,
                rf_buf,
                &kbufs[..],
                &vbufs[..],
                &wbufs[..],
            );
            b.bind(
                h.logits.expect("decode build has logits"),
                logits_buf.as_ref(),
            );
            if let Some(tid) = h.tok_id {
                b.bind(tid, id_out);
            }
            if let Some(uin) = h.u_in {
                b.bind(uin, u_buf.as_ref());
            }
            if let (Some(ho), Some(hb)) = (h.h_out, &h_tap_buf) {
                b.bind(ho, hb.as_ref());
            }
            setup_el = t_setup.elapsed();
            let t_exec = std::time::Instant::now();
            be.execute(plan.as_ref(), &b).map_err(|e| anyhow!("{e}"))?;
            exec_el = t_exec.elapsed();
        }
        // This step wrote `cur[pos]`'s KV row (a prompt token, a fed generated token, or a fed
        // grammar-forced token — all kept). The final sampled token and any forced tokens queued
        // past the frontier are pushed AFTER this and never re-enter the loop, so they stay
        // excluded from `last_written` — the fix for the `max_new==0` frontier and the
        // constrained-break unfed-forced-token cache-corruption cases.
        last_written = Some(pos);
        if prof_dec && pos + 1 >= prompt.len() {
            dec_setup += setup_el;
            dec_exec += exec_el;
        }

        if is_decode && at_frontier {
            // GPU-sampled paths (argmax or stochastic): skip the [vocab] logits download
            // entirely — the sampled id is read below (4 bytes). The one-time `logits_out` hook
            // still wants the full row.
            if !(gpu_argmax || gpu_sample) || logits_out.is_some() {
                be.download(logits_buf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                    .map_err(|e| anyhow!("{e}"))?;
            }
            // Phase-1 DiffusionGemma validation hook (see the param doc): this is the FIRST
            // is_decode row — the causal prefill's last-token logits — captured before sampling
            // touches `logits` (grammar-constrained steps overwrite it in place below).
            if let Some(out) = logits_out.take() {
                *out = logits.clone();
            }
            // MTP Phase 1 h-tap (see `want_h` above): same row, one op earlier than `logits`.
            if let (Some(out), Some(hb)) = (h_out.take(), &h_tap_buf) {
                let mut hrow = vec![0f32; ne];
                be.download(hb.as_ref(), bytemuck::cast_slice_mut(&mut hrow))
                    .map_err(|e| anyhow!("{e}"))?;
                *out = hrow;
            }
            if let Some(cst) = constraint.as_deref_mut() {
                // Grammar-forced span (serve's tool_choice "required"/named): the shared
                // llguidance step. Empty step ⇒ the constrained span ended.
                let (step, done) = crate::grammar::constrained_step(
                    cst,
                    &mut logits,
                    &c.eos_ids,
                    sampler,
                    &mut rng,
                )
                .map_err(|e| anyhow!("{e}"))?;
                decode_t += step_t0.elapsed();
                if step.is_empty() {
                    break;
                }
                for &t in &step {
                    out.push(t);
                    on_token(t);
                    cur.push(t);
                    decode_n += 1;
                }
                if done || out.len() >= max_new || crate::sampling::abort_requested(req) {
                    break;
                }
            } else {
                let next = if gpu_argmax || gpu_sample {
                    // Device-side sampling (Op::Argmax / Op::Sample): read back the 4-byte id.
                    let mut idb = [0u8; 4];
                    be.download(id_out, &mut idb).map_err(|e| anyhow!("{e}"))?;
                    u32::from_le_bytes(idb)
                } else {
                    // Serve-only: repetition penalties patch the row in place (no-op allocation-wise
                    // — `penalties` is `None` on every other path, so this is one branch per token).
                    if let Some(p) = penalties.as_ref() {
                        p.apply(&mut logits);
                    }
                    crate::sampling::sample_logits(&logits, sampler, &mut rng)
                };
                if let Some(p) = penalties.as_mut() {
                    p.observe(next);
                }
                let is_eos = !ignore_eos && (c.eos_ids.contains(&next) || next == c.eos);
                out.push(next);
                decode_t += step_t0.elapsed();
                decode_n += 1;
                if !is_eos {
                    on_token(next); // stream the token (EOS is not emitted)
                }
                // `on_token` -> the server's stop-sequence matcher may have latched an abort (a stop
                // string completed inside this token's text). One relaxed atomic load per token,
                // and not even that when `req` is None.
                if is_eos || out.len() >= max_new || crate::sampling::abort_requested(req) {
                    break;
                }
                cur.push(next);
            }
        } else if is_decode {
            // feeding a queued forced token — its KV write is the whole point of this step
            decode_t += step_t0.elapsed();
        } else {
            prompt_t += step_t0.elapsed();
        }
        pos += 1;
    }
    // TEARDOWN IS GPU WORK TOO. The record-once replay plan owns a Vulkan command buffer, and
    // dropping it runs `RecordedCmd::drop` -> `vkQueueWaitIdle` + `vkFreeCommandBuffers`. Left to
    // fall out of scope at the end of this function it would do that OUTSIDE the baton — and N
    // sequences finishing at once would then hit the queue and the command pool concurrently, which
    // is exactly the "externally synchronised" rule Vulkan states for both. The validation layer
    // caught this (UNASSIGNED-Threading-MultipleThreads-Write on VkQueue) even though every
    // *stepping* call site was already gated: the leak was in the destructor, not the hot loop.
    //
    // So take one last turn and drop it deliberately. (The prefill chunk plans are declared INSIDE
    // their gated scope, so they already drop before that scope's baton is released — drop order is
    // reverse-declaration and `_gp` is declared first.)
    {
        let _gp = req.and_then(|r| r.gate_pass());
        drop(ro);
    }
    if prof {
        let ts = |d: std::time::Duration, n: usize| {
            if d.as_secs_f64() > 0.0 {
                n as f64 / d.as_secs_f64()
            } else {
                0.0
            }
        };
        tracing::info!(
            "[cpu prof] prompt {} tok in {:.2}s ({:.1} tok/s) | decode {} tok in {:.2}s ({:.2} tok/s)",
            prompt.len(),
            prompt_t.as_secs_f64(),
            ts(prompt_t, prompt.len()),
            decode_n,
            decode_t.as_secs_f64(),
            ts(decode_t, decode_n),
        );
    }
    if prof_dec && decode_n > 0 {
        tracing::info!(
            "[dec prof] {} decode tok | setup(build+compile+bind) {:.3}ms/tok | exec(record+submit+gpu) {:.3}ms/tok",
            decode_n,
            dec_setup.as_secs_f64() * 1e3 / decode_n as f64,
            dec_exec.as_secs_f64() * 1e3 / decode_n as f64,
        );
    }
    // Record what the KV cache now holds for the next turn's prefix diff, straight from the
    // per-step `last_written` bookkeeping (`resident_after_gen`): exactly the tokens whose KV rows
    // were actually written by a kept token. This excludes the final sampled token (pushed to
    // `out` but never fed back), the `max_new == 0` un-fed frontier (the loop breaks before
    // feeding it — recording it corrupted the next "session cache warming" turn), and any
    // grammar-forced tokens queued past the frontier that a break left un-fed. On every run where
    // the old `prompt ++ out[..out.len()-1]` was already correct this is byte-identical to it.
    *cached = resident_after_gen(&cur, last_written);
    // The activation reserve is a PREDICTION, and this is the one place the predicted quantity can
    // be compared with what happened: the backend's high-water mark of concurrently-live
    // activation bytes. Nothing else in the tree observes it, so without this a reserve wrong by a
    // factor stays invisible until it fails an allocation on some other model. Reported at WARN
    // because an over-run means every context this model advertises was sized against a number
    // that does not hold.
    if let Some(peak) = be.activation_peak() {
        let rows = crate::seam::ubatch_rows(ec);
        // Both halves of what a prefill holds live, so the comparison stays honest in either
        // order: the per-chunk scratch, plus — layer-major only — the whole prompt's residual
        // stream, which is exactly what the placement budgets were told to leave room for.
        let reserved = crate::seam::dense_act_reserve_at(c, max_ctx, rows).saturating_add(
            if crate::seam::layer_major_prefill(ec, &caps, be.dense_paged(), !e2b) {
                crate::seam::layer_major_act_bytes(c, max_ctx, rows)
            } else {
                0
            },
        );
        // RUST_LOG=debug turns this into the measurement the reserve is re-fit against; the WARN
        // below is what a user sees when the prediction was wrong in the direction that hurts.
        tracing::debug!(
            activation_peak = peak,
            reserved,
            prefill_chunk = rows,
            ctx = max_ctx,
            "activations peaked at {:.0} MiB against a {:.0} MiB reserve",
            peak as f64 / (1u64 << 20) as f64,
            reserved as f64 / (1u64 << 20) as f64,
        );
        if peak > reserved && crate::seam::claim_act_over_reserve_report() {
            tracing::warn!(
                activation_peak = peak,
                reserved,
                prefill_chunk = rows,
                ctx = max_ctx,
                "activation reserve too low: {:.0} MiB of activations were live at once against a \
                 {:.0} MiB reserve — every context sized with this reserve is optimistic by the \
                 difference (see `dense_act_reserve_at`)",
                peak as f64 / (1u64 << 20) as f64,
                reserved as f64 / (1u64 << 20) as f64,
            );
        }
    }
    let stats = GenStats {
        // The tokens actually PREFILLED this call (the un-cached suffix) — the TTFT-honest count.
        n_prompt: prompt.len() - start,
        prompt_secs: prompt_t.as_secs_f64(),
        n_gen: decode_n,
        decode_secs: decode_t.as_secs_f64(),
    };
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::{resident_after_gen, validate_token_ids};

    // ── resident_after_gen: which tokens' KV rows are recorded as materialized ────────────────
    //
    // The runner tracks `last_written` = the highest position whose KV row was actually written
    // by a kept token. These tests pin the slicing decision for the scenarios the audit flagged.

    #[test]
    fn resident_max_new_zero_excludes_unfed_frontier() {
        // max_new == 0: every prompt token is fed EXCEPT the frontier (pos = plen-1), whose KV
        // row is never written (the loop breaks before feeding it). cur == prompt; the last
        // written position is plen-2. Must record prompt[..plen-1], NOT the full prompt (the old
        // `*cached = prompt.to_vec()` bug that corrupted the next session-cache-warming turn).
        let prompt = [10u32, 11, 12, 13];
        let plen = prompt.len();
        let cached = resident_after_gen(&prompt, Some(plen - 2));
        assert_eq!(cached, &prompt[..plen - 1]);
    }

    #[test]
    fn resident_single_token_prompt_max_new_zero_is_empty() {
        // A 1-token prompt with max_new == 0: the sole token is the un-fed frontier, nothing is
        // written this call. `last_written` is None → empty (prompt[..0]).
        let prompt = [42u32];
        assert!(resident_after_gen(&prompt, None).is_empty());
    }

    #[test]
    fn resident_normal_gen_excludes_last_sampled() {
        // A normal decode: the last sampled token is pushed to `out` but never fed, so
        // cur == prompt ++ out[..out.len()-1] and last_written == cur.len()-1. Result must equal
        // the old `prompt ++ out[..out.len()-1]` exactly (no behavior change on correct runs).
        let prompt = [1u32, 2, 3];
        let out = [4u32, 5, 6]; // 6 is the final sampled token, never fed back
        let mut cur = prompt.to_vec();
        cur.extend_from_slice(&out[..out.len() - 1]);
        let last = cur.len() - 1;
        let cached = resident_after_gen(&cur, Some(last));
        let mut expect = prompt.to_vec();
        expect.extend_from_slice(&out[..out.len() - 1]);
        assert_eq!(cached, expect);
    }

    #[test]
    fn resident_constrained_break_excludes_unfed_forced() {
        // A grammar-forced span emitted several tokens at the frontier then broke; none were fed
        // (their KV rows don't exist). cur == prompt ++ prev_fed ++ forced, but last_written
        // points at the frontier (the last fed token), so the forced tokens are excluded.
        let prompt = [1u32, 2];
        let prev_fed = [7u32]; // an earlier accepted+fed generated token
        let forced = [20u32, 21, 22]; // pushed to cur+out at the frontier, then break — never fed
        let mut cur = prompt.to_vec();
        cur.extend_from_slice(&prev_fed);
        let last = cur.len() - 1; // frontier = last fed token's position
        cur.extend_from_slice(&forced);
        let cached = resident_after_gen(&cur, Some(last));
        let mut expect = prompt.to_vec();
        expect.extend_from_slice(&prev_fed);
        assert_eq!(cached, expect);
    }

    #[test]
    fn resident_clamps_to_cur_len() {
        // Defensive: an over-large `last_written` never slices past `cur`.
        assert_eq!(resident_after_gen(&[1, 2, 3], Some(99)), vec![1, 2, 3]);
    }

    // ── validate_token_ids: OOB embedding-table slicing → clean error, not a panic ────────────

    #[test]
    fn validate_token_ids_accepts_in_range() {
        assert!(validate_token_ids(&[0, 1, 2, 99], 100).is_ok());
        assert!(validate_token_ids(&[], 100).is_ok());
    }

    #[test]
    fn validate_token_ids_rejects_out_of_vocab() {
        let err = validate_token_ids(&[0, 1, 100], 100).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn validate_token_ids_rejects_far_out_of_vocab() {
        assert!(validate_token_ids(&[u32::MAX], 100).is_err());
    }
}
