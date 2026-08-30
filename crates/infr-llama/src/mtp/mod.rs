//! MTP (multi-token prediction) head weights + forward for qwen35 (issue #33 — see `docs/mtp.md`).
//!
//! Phase 1 scope: locate + shape-check the tensors the head's 1-layer graph needs (`load_mtp_head`
//! below) — no ops, no forward.
//!
//! Phase 2 scope (this module's second half): the head's 1-layer forward as its own backend-generic
//! `Graph` (`build_mtp_graph`, ported op-for-op from `seam.rs`'s qwen35 full-attention-layer
//! emission — see that function's doc for the exact line citations) plus the two engine-level driver
//! primitives `docs/mtp.md` specifies (`catch_up`/`draft`, porting `common/speculative.cpp`'s
//! `common_speculative_impl_draft_mtp`). NO target-loop integration yet (Phase 3) — `MtpHeadSession`
//! is a standalone session over the head's OWN 1-layer KV, driven by tokens/h-rows the CALLER
//! supplies (from the trunk's `h_out` tap — Phase 1).
//!
//! The head layer sits at GGUF index `blk.{n_layer}` (`Config::n_layer` is already the TRUNK
//! count — see `Config::n_layer_nextn`'s doc), one index past the trunk's last layer, and carries
//! a FULL qwen35 attention-layer tensor set (same names/shapes/interleaved-q+gate layout as a
//! trunk full-attention layer, `docs/qwen35.md`) plus the `nextn.*` bridging tensors.

use anyhow::{anyhow, bail, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::{TensorInfo, WeightSource};
use infr_gguf::Gguf;

mod backends;
pub use backends::*;

/// One resolved-and-shape-checked tensor: its GGUF metadata (name/shape/dtype/offset), kept around
/// so Phase 2's `wload` can re-derive the bytes from `.name` without this module re-reading the
/// file or holding a second copy of the mmap slice.
pub type MtpTensor = TensorInfo;

/// The qwen35 MTP head's tensors (see the module doc). Every required field here EXISTED in the
/// GGUF and had the expected shape at [`load_mtp_head`] time; the three `Option` fields are the
/// ones the reference allows to fall back to the main model's tensors when absent (`docs/mtp.md`'s
/// confirmed dump: the shipped 4B GGUF omits `embed_tokens`/`shared_head_head` — those two fall
/// back — but DOES ship its own `shared_head_norm`).
pub struct MtpHeadWeights {
    /// The GGUF block index of the head layer (`cfg.n_layer`, i.e. immediately after the trunk).
    pub il: usize,
    // ── standard qwen35 full-attention layer tensors (identical shapes to a trunk full-attn
    //    layer at a `(il+1) % full_attn_interval == 0` index — see `Config::is_qwen35_attn_layer`) ──
    pub attn_norm: MtpTensor,
    /// Interleaved q+gate: `[n_embd, head_dim * n_head * 2]` (see `Config::attn_out_gate`).
    pub attn_q: MtpTensor,
    pub attn_k: MtpTensor,
    pub attn_v: MtpTensor,
    pub attn_q_norm: MtpTensor,
    pub attn_k_norm: MtpTensor,
    pub attn_output: MtpTensor,
    pub post_attention_norm: MtpTensor,
    pub ffn_gate: MtpTensor,
    pub ffn_up: MtpTensor,
    pub ffn_down: MtpTensor,
    // ── NextN bridge (the tensors that make this an MTP head, not just another trunk layer) ──
    /// `[2*n_embd, n_embd]`: projects `concat(rmsnorm(embed(t)), rmsnorm(h_target))` down to
    /// `n_embd` before the layer's own attention (see `docs/mtp.md`'s forward pseudocode).
    pub eh_proj: MtpTensor,
    pub enorm: MtpTensor,
    pub hnorm: MtpTensor,
    /// Falls back to the main model's `token_embd.weight` when absent (the shipped 4B GGUF has no
    /// `nextn.embed_tokens` — see `docs/mtp.md`'s confirmed dump).
    pub embed_tokens: Option<MtpTensor>,
    /// Falls back to the main model's (tied) lm_head when absent.
    pub shared_head_head: Option<MtpTensor>,
    /// Falls back to the main model's `output_norm.weight` when absent.
    pub shared_head_norm: Option<MtpTensor>,
}

/// MASTER KILL-SWITCH for the whole MTP self-speculative decode path — currently **DISABLED**.
///
/// Every MTP entry point (the three `chat/{cpu,metal,vulkan}.rs` `INFR_MTP=1` gates, `infr bench`'s
/// `mtp` arm, `infr compare`'s `mtp128` column) routes through this. While it returns `false`:
/// `INFR_MTP=1` is IGNORED with a warning, and a head-bearing GGUF (Qwen3.5-*-MTP) simply runs the
/// ORDINARY non-speculative decode path — the `nextn` head tensors are left unused, which is
/// harmless (they are extra tensors, not a different trunk).
///
/// WHY (2026-07-13): MTP's contract is that its output is token-identical to non-speculative greedy
/// ("pure speedup"). That no longer holds. int8-activation decode kernels — which every fast dtype
/// now uses — carry small per-token rounding noise, and MTP's verify batch and the plain-decode
/// chain it must match are computed at DIFFERENT sequence positions with different KV state. The
/// same noise that plain decode absorbs harmlessly is enough to flip a close-margin greedy argmax
/// across the two streams, so `mtp_spec_matches_target_only_greedy` fails. This is NOT a
/// bit-identity bug (`mmv_row1_bit_identical` passes — decode and verify share one kernel) and NOT
/// an accuracy cliff (all 13 `gpu_seam_matches_cpu_*` pass, output stays coherent). It is inherent
/// quantization noise, and it was BLOCKING the int8 decode tier for Q6_K (+10% decode, +34%
/// prefill) and others. Rather than hold every other dtype's win hostage to a speculative-decode
/// path that was already our slowest row (0.59-0.78x vs llama.cpp), MTP is parked.
///
/// TO RE-ENABLE: return `true` here, then make `mtp_spec_matches_target_only_greedy` pass again.
/// The real fix is an accuracy mitigation (e.g. re-verify in f32 when the top-2 logit margin is
/// below a threshold), NOT faster kernels — see `infr_vulkan`'s `mmv_int8_decode_dtypes` doc.
pub fn mtp_enabled() -> bool {
    false
}

/// The SINGLE MTP opt-in decision the chat backends share (`chat::vulkan`/`chat::metal`/`chat::cpu`
/// `ChatModel`s). Returns `true` only when the user asked (`INFR_MTP=1`), MTP is not parked
/// ([`mtp_enabled`]), AND the GGUF actually ships an MTP head (`cfg.n_layer_nextn > 0`). When
/// `INFR_MTP=1` but MTP is parked it warns once and returns `false` (the head-bearing GGUF still
/// runs — its `nextn` tensors are simply unused). Extracted because the Vulkan and Metal gates had
/// drifted: one warned on the parked path, the other silently ignored the env var.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn should_use_mtp(cfg: &crate::Config, ec: &crate::EngineConfig) -> bool {
    // `spec.mtp` is the EXACT string "1" (§10.4) — `INFR_MTP=true` did nothing before and still
    // does nothing; the env layer owns that comparison now.
    if !ec.spec.mtp {
        return false;
    }
    if !mtp_enabled() {
        tracing::warn!(
            "[infr] INFR_MTP=1 ignored: MTP speculative decode is disabled (known-broken — it \
             no longer matches greedy output under the int8 decode kernels; see README). \
             Running the ordinary decode path."
        );
        return false;
    }
    cfg.n_layer_nextn > 0
}

/// Cheap arch/head check from a resolved GGUF path — no `Config`/tensor validation, just the
/// metadata flag (mirrors `qwen35::is_qwen35`/`diffusion::is_diffusion_gemma`'s "peek without a
/// full load" convention). `infr compare`'s MTP DECODE section and the `--sweep` `mtp` column
/// (issue #33, phase 4 — perf-bottleneck visibility) use this to decide whether a model needs the
/// extra measurement at all, cheaper than `Config::from_gguf` (which validates every other field
/// too) for a check this narrow. Callers must ALSO check [`mtp_enabled`] — while MTP is parked this
/// stays truthful about the FILE (the head really is there), it just must not be acted on.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn has_mtp_head(path: &std::path::Path) -> bool {
    let Ok(g) = Gguf::open(path) else {
        return false;
    };
    let arch = g.metadata().str("general.architecture").unwrap_or("");
    g.metadata()
        .u64(&format!("{arch}.nextn_predict_layers"))
        .unwrap_or(0)
        > 0
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn find<'a>(g: &'a Gguf, name: &str) -> Option<&'a TensorInfo> {
    g.tensors().iter().find(|t| t.name == name)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn require(g: &Gguf, name: &str, want: &[usize]) -> Result<MtpTensor> {
    let t = find(g, name).ok_or_else(|| anyhow!("MTP head: missing tensor {name}"))?;
    if t.shape != want {
        bail!(
            "MTP head: {name} has shape {:?}, expected {:?}",
            t.shape,
            want
        );
    }
    Ok(t.clone())
}

/// Like [`require`] but `None` (not an error) when the tensor is simply absent — the reference's
/// fallback tensors (`nextn.embed_tokens`/`shared_head_head`/`shared_head_norm`). A PRESENT tensor
/// with the wrong shape is still an error (a real corruption, not an intentional omission).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn optional(g: &Gguf, name: &str, want: &[usize]) -> Result<Option<MtpTensor>> {
    match find(g, name) {
        Some(t) if t.shape == want => Ok(Some(t.clone())),
        Some(t) => bail!(
            "MTP head: {name} has shape {:?}, expected {:?}",
            t.shape,
            want
        ),
        None => Ok(None),
    }
}

/// Locate + shape-check the qwen35 MTP head's tensors (see the module doc). Requires
/// `cfg.n_layer_nextn == 1` (Phase 1's only supported case — `Config::from_gguf` already rejects
/// anything else) and `cfg.qwen35`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn load_mtp_head(g: &Gguf, cfg: &crate::Config) -> Result<MtpHeadWeights> {
    if !cfg.qwen35 || cfg.n_layer_nextn != 1 {
        bail!(
            "load_mtp_head: requires a qwen35 GGUF with nextn_predict_layers==1 (got qwen35={}, \
             n_layer_nextn={})",
            cfg.qwen35,
            cfg.n_layer_nextn,
        );
    }
    let il = cfg.n_layer; // the MTP head sits immediately after the trunk (see Config::n_layer_nextn)
    let p = |s: &str| format!("blk.{il}.{s}");
    let ne = cfg.n_embd;
    let qdim = cfg.head_dim * cfg.n_head * 2; // interleaved q+gate, see Config::attn_out_gate
    let kv_dim = cfg.n_kv * cfg.head_dim;
    Ok(MtpHeadWeights {
        il,
        attn_norm: require(g, &p("attn_norm.weight"), &[ne])?,
        attn_q: require(g, &p("attn_q.weight"), &[ne, qdim])?,
        attn_k: require(g, &p("attn_k.weight"), &[ne, kv_dim])?,
        attn_v: require(g, &p("attn_v.weight"), &[ne, kv_dim])?,
        attn_q_norm: require(g, &p("attn_q_norm.weight"), &[cfg.head_dim])?,
        attn_k_norm: require(g, &p("attn_k_norm.weight"), &[cfg.head_dim])?,
        attn_output: require(
            g,
            &p("attn_output.weight"),
            &[cfg.head_dim * cfg.n_head, ne],
        )?,
        post_attention_norm: require(g, &p("post_attention_norm.weight"), &[ne])?,
        ffn_gate: require(g, &p("ffn_gate.weight"), &[ne, cfg.n_ff])?,
        ffn_up: require(g, &p("ffn_up.weight"), &[ne, cfg.n_ff])?,
        ffn_down: require(g, &p("ffn_down.weight"), &[cfg.n_ff, ne])?,
        eh_proj: require(g, &p("nextn.eh_proj.weight"), &[2 * ne, ne])?,
        enorm: require(g, &p("nextn.enorm.weight"), &[ne])?,
        hnorm: require(g, &p("nextn.hnorm.weight"), &[ne])?,
        embed_tokens: optional(g, &p("nextn.embed_tokens.weight"), &[ne, cfg.vocab])?,
        shared_head_head: optional(g, &p("nextn.shared_head_head.weight"), &[ne, cfg.vocab])?,
        shared_head_norm: optional(g, &p("nextn.shared_head_norm.weight"), &[ne])?,
    })
}

// ─── Phase 2: the head forward + the draft primitives (issue #33 — see `docs/mtp.md`) ──────────

/// A weight binder in the shape `seam.rs`'s (module-private) `BindWeight` uses: turns a
/// native-dtype GGUF tensor into a backend buffer + the effective dtype it now holds. Named here
/// (rather than inlined at each call site) purely to keep clippy's `type_complexity` lint quiet —
/// see [`upload_mtp_head_bufs`] / [`MtpHeadSession::build`].
type BindWeightFn<'a> =
    dyn Fn(&str, crate::seam::WBytes, DType, usize) -> Result<(Box<dyn Buffer>, DType)> + 'a;

/// [`upload_mtp_head_bufs`]'s return shape: the 16 uploaded weight buffers, index-parallel with
/// their (effective dtype, element count) — the pair `build_mtp_graph` needs to declare each
/// handle. Named purely to keep clippy's `type_complexity` lint quiet.
type WBufs = (Vec<Box<dyn Buffer>>, Vec<(DType, usize)>);

/// Resolve a fallback tensor from the MAIN model (the reference's `layer.nextn.X ? layer.nextn.X :
/// model.Y` — `qwen35.cpp:624-638`) when the head's own `nextn.*` tensor is absent.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn main_output_norm(g: &Gguf, ne: usize) -> Result<MtpTensor> {
    require(g, "output_norm.weight", &[ne])
}

/// The tied/untied lm_head fallback (`qwen35.cpp:636`'s `layer.nextn.shared_head_head ? ... :
/// model.output`), mirroring `seam.rs`'s own tied-weight rule (`wload(&["output.weight"])`
/// vs the `token_embd.weight` fallback right above it in that file).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn main_lm_head(g: &Gguf, ne: usize, vocab: usize) -> Result<MtpTensor> {
    if find(g, "output.weight").is_some() {
        require(g, "output.weight", &[ne, vocab])
    } else {
        require(g, "token_embd.weight", &[ne, vocab])
    }
}

/// The main model's `token_embd.weight` fallback (`qwen35.cpp`'s `layer.nextn.embed_tokens ? ... :
/// model.tok_embd`) for [`build_embed_chain_buf`]'s device-side table upload, when the head has no
/// own `nextn.embed_tokens` — mirrors [`main_lm_head`]'s fallback shape.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn main_token_embd(g: &Gguf, ne: usize, vocab: usize) -> Result<MtpTensor> {
    require(g, "token_embd.weight", &[ne, vocab])
}

/// Resolve the head's OWN embedding table (`nextn.embed_tokens`, dequantized) when the GGUF ships
/// one — `None` when it doesn't (the shipped 4B GGUF: `docs/mtp.md`'s confirmed dump), in which
/// case the caller passes the main model's already-dequantized `token_embd` straight to
/// [`MtpHeadSession::new_cpu`]/[`new_vulkan`] instead.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn resolve_own_embed_table(g: &Gguf, head: &MtpHeadWeights) -> Result<Option<Vec<f32>>> {
    match &head.embed_tokens {
        Some(t) => Ok(Some(crate::load_tensor_dequant(g, &t.name)?.0)),
        None => Ok(None),
    }
}

/// Upload the head's 16 graph weights (attention-layer set + NextN bridge, in `wpush` order —
/// MUST match [`build_mtp_graph`]'s `wpush` calls 1:1) through the caller's `bind_weight` — the
/// SAME binder shape `seam.rs`'s `BindWeight` uses (zero-copy mmap on CPU, padded upload on
/// Vulkan — see `MtpHeadSession::new_cpu`/`new_vulkan`), so the head's tensors land in memory
/// exactly like the trunk's do. Falls back to the main model's `output_norm`/tied lm_head for the
/// two optional NextN tensors that are absent (`docs/mtp.md`'s confirmed dump — `shared_head_norm`
/// IS present in the shipped 4B GGUF, so only the lm_head fallback fires there in practice).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn upload_mtp_head_bufs(
    be: &dyn Backend,
    bind_weight: &BindWeightFn,
    g: &Gguf,
    cfg: &crate::Config,
    head: &MtpHeadWeights,
) -> Result<WBufs> {
    let _ = be; // kept for symmetry with seam's wload closures; binder itself owns `be`
    let mut bufs = Vec::with_capacity(16);
    let mut specs = Vec::with_capacity(16);
    let mut push = |t: &MtpTensor| -> Result<()> {
        let tb = g.tensor_bytes_arc(&t.name).map_err(|e| anyhow!("{e}"))?;
        let numel: usize = t.shape.iter().product();
        let (buf, dt) = bind_weight(&t.name, crate::seam::WBytes::Mmap(tb), t.dtype, numel)?;
        bufs.push(buf);
        specs.push((dt, numel));
        Ok(())
    };
    push(&head.attn_norm)?;
    push(&head.attn_q)?;
    push(&head.attn_k)?;
    push(&head.attn_v)?;
    push(&head.attn_q_norm)?;
    push(&head.attn_k_norm)?;
    push(&head.attn_output)?;
    push(&head.post_attention_norm)?;
    push(&head.ffn_gate)?;
    push(&head.ffn_up)?;
    push(&head.ffn_down)?;
    push(&head.eh_proj)?;
    push(&head.enorm)?;
    push(&head.hnorm)?;
    match &head.shared_head_norm {
        Some(t) => push(t)?,
        None => push(&main_output_norm(g, cfg.n_embd)?)?,
    }
    match &head.shared_head_head {
        Some(t) => push(t)?,
        None => push(&main_lm_head(g, cfg.n_embd, cfg.vocab)?)?,
    }
    Ok((bufs, specs))
}

/// Resolve + conditionally upload the head's embedding table as a device Weight buffer for
/// [`MtpHeadSession::draft_chain`]'s on-device `Op::EmbedGather` chain (issue #33 follow-up).
/// Source tensor: the head's own `nextn.embed_tokens` when the GGUF ships one, else the main
/// model's `token_embd.weight` (mirrors [`resolve_own_embed_table`]'s fallback — the SAME table
/// [`MtpHeadSession::embed_table`]'s host-side dequantized copy comes from). Uploaded through the
/// caller's `bind_weight` (the SAME per-backend binder the other 16 head weights use — zero-copy
/// mmap on CPU, padded upload on Vulkan), so its device layout matches every other resident weight.
///
/// Gated on `Capabilities::embed_gather && Capabilities::argmax_prob` (both must hold for the
/// chained draft loop's `Op::EmbedGather` + `Op::ArgmaxProb` to run on-device), `cfg.n_embd`
/// 32-aligned, and the table's native dtype passing
/// `infr_vulkan::linear::embed_gather_supported` — the SAME three gates `seam/runner.rs`'s
/// `gpu_embed` flag checks for the trunk's own embed-gather path, reused verbatim here rather than
/// re-deriving a parallel dtype allowlist. Returns `(None, DType::F32)` when any gate fails —
/// `MtpHeadSession::can_draft_chain` reflects the `None` and `draft_chain` refuses to run, falling
/// back to the host-driven [`draft`] loop (Metal today: `argmax_prob == false`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn build_embed_chain_buf(
    be: &dyn Backend,
    bind_weight: &BindWeightFn,
    g: &Gguf,
    cfg: &crate::Config,
    head: &MtpHeadWeights,
) -> Result<(Option<Box<dyn Buffer>>, DType)> {
    let t = match &head.embed_tokens {
        Some(t) => t.clone(),
        None => main_token_embd(g, cfg.n_embd, cfg.vocab)?,
    };
    let caps = be.capabilities();
    // The fused chain feeds each step's device-produced argmax id (`Op::ArgmaxProb`) straight into
    // the next step's device `Op::EmbedGather` — a pure device submit-batching win over the per-step
    // `draft()`. The scalar CPU-reference interpreter can't round-trip that on-device id chain: it
    // stores an argmax id as a BIT-PATTERN f32 (`f32::from_bits`, so the host readback of the id
    // output is byte-correct) but reads `EmbedGather` ids NUMERICALLY (`id as f32`, matching a
    // host-uploaded i32 ids input), so a chained id decodes to garbage. The chain has no perf value
    // on the reference backend anyway (no real submits to batch), so gate it to GPU backends and let
    // CPU take the canonical per-step `draft()` path (which the MTP acceptance oracle wants regardless).
    let gate = caps.name != "cpu-reference"
        && caps.embed_gather
        && caps.argmax_prob
        && cfg.n_embd.is_multiple_of(32)
        && infr_vulkan::linear::embed_gather_supported(t.dtype);
    if !gate {
        return Ok((None, DType::F32));
    }
    let tb = g.tensor_bytes_arc(&t.name).map_err(|e| anyhow!("{e}"))?;
    let numel: usize = t.shape.iter().product();
    let (buf, dt) = bind_weight(&t.name, crate::seam::WBytes::Mmap(tb), t.dtype, numel)?;
    Ok((Some(buf), dt))
}

/// The head graph's tensor handles `MtpHeadSession::forward` binds/reads each call.
struct MtpHandles {
    e_raw: TensorId,
    h_in: TensorId,
    positions: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    /// The 16 weight handles, in [`upload_mtp_head_bufs`]'s push order (index-parallel with
    /// `MtpHeadSession::wbufs`).
    weights: Vec<TensorId>,
    /// The lm_head GEMM's output, or `None` when built with `want_logits == false` (the catch-up
    /// path — AUDIT #2): a KV-only forward omits the lm_head `Op::Linear` + its `[rows*vocab]`
    /// readback entirely, since catch_up discards the logits. When present it's an `Output`
    /// (host-downloadable) tensor for `fuse_prob == false`, or an `Internal` (device-only scratch,
    /// never bound) tensor for `fuse_prob == true` — the fused variant reads it straight into
    /// `Op::ArgmaxProb` and never sends the row to the host at all (see [`build_mtp_graph`]'s doc).
    logits: Option<TensorId>,
    h_mtp: TensorId,
    /// `Some((id, prob))` only when built with `fuse_prob == true` — the two 1-element `Output`s
    /// `Op::ArgmaxProb` writes (issue #33 follow-up: 8 bytes instead of the `[vocab]` logits row).
    id_prob: Option<(TensorId, TensorId)>,
}

/// Build the MTP head's 1-layer forward graph for `rows` `(token, h_target)` pairs starting at
/// absolute KV position `start_pos` (`docs/mtp.md`'s forward pseudocode):
/// ```text
/// e = rmsnorm(embed(t), enorm);  h = rmsnorm(h_target, hnorm)
/// x = eh_proj @ concat([e; h])                    # eh_proj: [2ne, ne] (in=2ne, out=ne)
/// x = qwen35_attention_layer(x, pos)               # own 1-layer KV, causal
/// h_mtp = rmsnorm(x, shared_head_norm)             # fed back on the NEXT draft step
/// logits = lm_head @ h_mtp
/// ```
/// The attention-layer ops (interleaved q+gate split, qk-norm+RoPE, sigmoid out-gate, SwiGLU FFN)
/// are ported op-for-op from `seam.rs`'s qwen35 `c.attn_out_gate` branch (that file's
/// `generate_dense_backend::build` closure, roughly lines 2269-2707: the per-layer attn-input norm,
/// the interleaved qg projection + per-head `CopyStrided` split, k/v projections, `QkNormRope`,
/// `WriteKv`, `Attention`, the post-attention `GatedAct(Sigmoid)` gate, the o-projection, the
/// residual add, the ffn-norm + dense SwiGLU FFN, and the final residual add) — see
/// `qwen35.cpp:556-622` for the reference this ports.
///
/// The `eh_proj` "concat" (`qwen35.cpp:548`'s `ggml_concat(e_norm, h_norm, dim=0)`): with infr's
/// `Op::Linear` weight convention (`[out, in]` row-major — a weight's `in`-dim is the CONTIGUOUS
/// per-row axis), `eh_proj`'s GGUF shape `[2ne, ne]` (in=2ne, out=ne) means row `j`'s `2ne`
/// contiguous elements are `[a_j (ne, dot e) | b_j (ne, dot h)]` — i.e. `ggml_concat(dim=0)` is a
/// COLUMN split of each output row, not a row split. Rather than a strided weight VIEW (the same
/// trick `attn_q`'s interleaved q/gate split already uses, just on the weight side instead of an
/// activation), the least-new-code correct form builds the concatenated ACTIVATION on device with
/// two existing `Op::CopyStrided` (e into `concat[0..ne]`, h into `concat[ne..2ne]`, per row) and
/// then ONE ordinary `Op::Linear` over the full `[2ne,ne]` weight — no new `Op` variant, and it's
/// literally the reference's `ggml_concat` + `mm` in the same order. Proved by
/// `mtp_head_forward_finite`'s finite-logits check plus `h_tap_matches_lm_head`-style consistency
/// implicit in the parity test (a layout bug here would show up as garbage/NaN logits or gross
/// CPU/Vulkan disagreement, not a subtle numeric drift).
/// `fuse_prob` (issue #33 follow-up, de35727's deferred DRAFT-side twin): when `true`, appends
/// `Op::ArgmaxProb` reading the lm_head output straight off the device and returns its two 1-element
/// `Output`s as `MtpHandles::id_prob` instead of exposing the full `[rows*vocab]` logits row as an
/// `Output` — the caller downloads 8 bytes instead of the whole row (and skips the host
/// argmax/softmax scan that used to consume it). Requires `rows == 1` (the draft loop's own shape;
/// `Op::ArgmaxProb` is single-row only, see its doc) — callers with `rows > 1` (catch_up, verify
/// priming) must pass `fuse_prob = false`.
///
/// `want_logits` (AUDIT #2): when `false`, the lm_head `Op::Linear` (a `[rows, vocab]` GEMM over the
/// 151936-token vocab) and its `Output` logits tensor are OMITTED — the graph stops at the KV writes
/// and the `h_mtp` norm. The catch-up path (`catch_up` → `forward(want_logits = false)`) only needs
/// the `WriteKv` ops to re-sync the head's KV; it discards the logits every cycle, so computing and
/// downloading a full vocab-wide row was pure waste. The retained ops (through the `WriteKv`s) are
/// byte-identical to the `want_logits == true` build. `fuse_prob == true` forces logits on (the
/// `Op::ArgmaxProb` reads them).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn build_mtp_graph(
    cfg: &crate::Config,
    wspecs: &[(DType, usize)],
    max_ctx: usize,
    rows: usize,
    start_pos: usize,
    fuse_prob: bool,
    want_logits: bool,
) -> (Graph, MtpHandles) {
    debug_assert!(
        !fuse_prob || rows == 1,
        "build_mtp_graph: fuse_prob requires rows == 1 (Op::ArgmaxProb is single-row only)"
    );
    // fuse_prob needs the logits row to reduce; it always implies want_logits.
    let want_logits = want_logits || fuse_prob;
    let ne = cfg.n_embd;
    let nh = cfg.n_head;
    let nkv = cfg.n_kv;
    let hd = cfg.head_dim;
    let qrow = nh * hd;
    let kvrow = nkv * hd;
    let nff = cfg.n_ff;
    let eps = cfg.rms_eps;
    let theta = cfg.rope_theta; // qwen35: uniform (no gemma dual-rope) — Config::layer_rope_theta
    let rope_dim = cfg.rope_dim; // collapses to this for a non-SWA model; see Config's doc.
    let scale = 1.0 / (hd as f32).sqrt(); // qwen35 isn't gemma4 (no QK-norm-only scale of 1.0)

    let mut g = Graph::new();
    let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
    let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16); // KV dtype: fixed f16 (see this
                                                                // fn's doc on the Phase-3 TODO)
    let i32d = |n: usize| TensorDesc::new(vec![n], DType::I32);

    let e_raw = g.input(f32d(rows * ne));
    let h_in = g.input(f32d(rows * ne));
    let positions = g.input(i32d(rows));
    let k_cache = g.input(f16d(max_ctx * kvrow));
    let v_cache = g.input(f16d(max_ctx * kvrow));

    let mut wi = 0usize;
    let mut wpush = |g: &mut Graph| -> TensorId {
        let (dt, n) = wspecs[wi];
        wi += 1;
        g.weight(TensorDesc::new(vec![n], dt))
    };
    // MUST match `upload_mtp_head_bufs`'s push order exactly.
    let w_attn_norm = wpush(&mut g);
    let w_q = wpush(&mut g);
    let w_k = wpush(&mut g);
    let w_v = wpush(&mut g);
    let w_qn = wpush(&mut g);
    let w_kn = wpush(&mut g);
    let w_o = wpush(&mut g);
    let w_ffn_norm = wpush(&mut g);
    let w_gate = wpush(&mut g);
    let w_up = wpush(&mut g);
    let w_down = wpush(&mut g);
    let w_eh_proj = wpush(&mut g);
    let w_enorm = wpush(&mut g);
    let w_hnorm = wpush(&mut g);
    let w_head_norm = wpush(&mut g);
    let w_lm_head = wpush(&mut g);
    let weights = vec![
        w_attn_norm,
        w_q,
        w_k,
        w_v,
        w_qn,
        w_kn,
        w_o,
        w_ffn_norm,
        w_gate,
        w_up,
        w_down,
        w_eh_proj,
        w_enorm,
        w_hnorm,
        w_head_norm,
        w_lm_head,
    ];

    // `logits`: omitted entirely on the KV-only catch-up path (`want_logits == false` — AUDIT #2).
    // Fused variant: `logits` never leaves the device (Internal scratch, read straight into
    // Op::ArgmaxProb below) — see this fn's doc on `fuse_prob`/`want_logits`.
    let logits = if !want_logits {
        None
    } else if fuse_prob {
        Some(g.internal(f32d(rows * cfg.vocab)))
    } else {
        Some(g.output(f32d(rows * cfg.vocab)))
    };
    let h_mtp = g.output(f32d(rows * ne));
    let id_prob = fuse_prob.then(|| (g.output(f32d(1)), g.output(f32d(1))));

    // scratch
    let e_norm = g.internal(f32d(rows * ne));
    let h_norm = g.internal(f32d(rows * ne));
    let concat = g.internal(f32d(rows * 2 * ne));
    let resid = g.internal(f32d(rows * ne)); // eh_proj output; mutated in place across the residuals
    let hn = g.internal(f32d(rows * ne));
    let qg = g.internal(f32d(rows * 2 * qrow));
    let q = g.internal(f32d(rows * qrow));
    let gate_a = g.internal(f32d(rows * qrow));
    let k = g.internal(f32d(rows * kvrow));
    let v = g.internal(f32d(rows * kvrow));
    let q16 = g.internal(f16d(rows * qrow));
    let k16 = g.internal(f16d(rows * kvrow));
    let attn = g.internal(f32d(rows * qrow));
    let sub = g.internal(f32d(rows * ne));
    let gbuf = g.internal(f32d(rows * nff));
    let ubuf = g.internal(f32d(rows * nff));
    let actbuf = g.internal(f32d(rows * nff));
    let hn2 = g.internal(f32d(rows * ne));
    let layer_out = g.internal(f32d(rows * ne));

    // e = rmsnorm(embed(t), enorm); h = rmsnorm(h_target, hnorm) — qwen35.cpp:542-546.
    g.push(Op::RmsNorm {
        x: e_raw,
        weight: w_enorm,
        dst: e_norm,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    g.push(Op::RmsNorm {
        x: h_in,
        weight: w_hnorm,
        dst: h_norm,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    // concat = [e_norm | h_norm] per row (this fn's doc on the eh_proj layout) — qwen35.cpp:548.
    g.push(Op::CopyStrided {
        src: e_norm,
        src_off: 0,
        src_stride: ne as u32,
        dst: concat,
        dst_off: 0,
        dst_stride: (2 * ne) as u32,
        rows: rows as u32,
        n: ne as u32,
    });
    g.push(Op::CopyStrided {
        src: h_norm,
        src_off: 0,
        src_stride: ne as u32,
        dst: concat,
        dst_off: ne as u32,
        dst_stride: (2 * ne) as u32,
        rows: rows as u32,
        n: ne as u32,
    });
    // resid = eh_proj @ concat — qwen35.cpp:551 (`inpSA` in the reference).
    g.push(Op::Linear {
        x: concat,
        weight: w_eh_proj,
        dst: resid,
        m: rows as u32,
        in_f: (2 * ne) as u32,
        out_f: ne as u32,
        w_off: 0,
    });

    // ── one qwen35 full-attention layer on `resid` (own 1-layer KV) — seam.rs's
    //    `c.attn_out_gate` branch, qwen35.cpp:556-619 ──
    g.push(Op::RmsNorm {
        x: resid,
        weight: w_attn_norm,
        dst: hn,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    // interleaved q+gate projection + per-head split (seam.rs:2464-2493, qwen35.cpp:559-576).
    g.push(Op::Linear {
        x: hn,
        weight: w_q,
        dst: qg,
        m: rows as u32,
        in_f: ne as u32,
        out_f: (qrow * 2) as u32,
        w_off: 0,
    });
    for h in 0..nh {
        g.push(Op::CopyStrided {
            src: qg,
            src_off: (h * 2 * hd) as u32,
            src_stride: (nh * 2 * hd) as u32,
            dst: q,
            dst_off: (h * hd) as u32,
            dst_stride: (nh * hd) as u32,
            rows: rows as u32,
            n: hd as u32,
        });
        g.push(Op::CopyStrided {
            src: qg,
            src_off: (h * 2 * hd + hd) as u32,
            src_stride: (nh * 2 * hd) as u32,
            dst: gate_a,
            dst_off: (h * hd) as u32,
            dst_stride: (nh * hd) as u32,
            rows: rows as u32,
            n: hd as u32,
        });
    }
    g.push(Op::Linear {
        x: hn,
        weight: w_k,
        dst: k,
        m: rows as u32,
        in_f: ne as u32,
        out_f: kvrow as u32,
        w_off: 0,
    });
    g.push(Op::Linear {
        x: hn,
        weight: w_v,
        dst: v,
        m: rows as u32,
        in_f: ne as u32,
        out_f: kvrow as u32,
        w_off: 0,
    });
    // K: fused QkNorm+RoPE (own KV write immediately follows, matching the trunk's fused-write
    // adjacency convention — seam.rs:2587-2636).
    g.push(Op::QkNormRope {
        x: k,
        weight: w_kn,
        positions,
        dst: k16,
        rows: rows as u32,
        n_head: nkv as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta,
        eps,
        freq_factors: None, // gemma4-only; qwen35 never sets this
        x_stride: 0,
    });
    g.push(Op::WriteKv {
        src: k16,
        cache: k_cache,
        rows: rows as u32,
        row_stride: kvrow as u32,
        pos: start_pos as u32,
    });
    g.push(Op::WriteKv {
        src: v,
        cache: v_cache,
        rows: rows as u32,
        row_stride: kvrow as u32,
        pos: start_pos as u32,
    });
    g.push(Op::QkNormRope {
        x: q,
        weight: w_qn,
        positions,
        dst: q16,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta,
        eps,
        freq_factors: None,
        x_stride: 0,
    });
    g.push(Op::Attention {
        q: q16,
        k_cache,
        v_cache,
        dst: attn,
        rows: rows as u32,
        kv_len: (start_pos + rows) as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale,
        mask: AttnMask::Causal,
        pos: start_pos as u32,
        sinks: None,
    });
    // sigmoid out-gate BEFORE the o-projection — seam.rs:2686-2698, qwen35.cpp:602.
    g.push(Op::GatedAct {
        gate: gate_a,
        up: attn,
        dst: attn,
        rows: rows as u32,
        nff: qrow as u32,
        act: Activation::Sigmoid,
        up_off: 0,
        up_stride: 0,
        gate_stride: 0,
        gate_block_width: 0,
        swiglu_clamp: None,
    });
    g.push(Op::Linear {
        x: attn,
        weight: w_o,
        dst: sub,
        m: rows as u32,
        in_f: qrow as u32,
        out_f: ne as u32,
        w_off: 0,
    });
    // attn residual — qwen35.cpp:606 (`cur = add(cur, inpSA)`; `inpSA` IS `resid` here).
    g.push(Op::Add {
        a: resid,
        b: sub,
        dst: resid,
        n: (rows * ne) as u32,
    });
    // ffn — qwen35.cpp:610-619 (SwiGLU; qwen35 names the pre-FFN norm `post_attention_norm`, not
    // `ffn_norm` — see `Config::qwen35`'s doc / `w_ffn_norm` bound to `post_attention_norm.weight`
    // in `upload_mtp_head_bufs`).
    g.push(Op::RmsNorm {
        x: resid,
        weight: w_ffn_norm,
        dst: hn2,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    g.push(Op::Linear {
        x: hn2,
        weight: w_gate,
        dst: gbuf,
        m: rows as u32,
        in_f: ne as u32,
        out_f: nff as u32,
        w_off: 0,
    });
    g.push(Op::Linear {
        x: hn2,
        weight: w_up,
        dst: ubuf,
        m: rows as u32,
        in_f: ne as u32,
        out_f: nff as u32,
        w_off: 0,
    });
    g.push(Op::GatedAct {
        gate: gbuf,
        up: ubuf,
        dst: actbuf,
        rows: rows as u32,
        nff: nff as u32,
        act: Activation::Silu,
        up_off: 0,
        up_stride: 0,
        gate_stride: 0,
        gate_block_width: 0,
        swiglu_clamp: None,
    });
    g.push(Op::Linear {
        x: actbuf,
        weight: w_down,
        dst: sub,
        m: rows as u32,
        in_f: nff as u32,
        out_f: ne as u32,
        w_off: 0,
    });
    // ffn residual — qwen35.cpp:621.
    g.push(Op::Add {
        a: resid,
        b: sub,
        dst: layer_out,
        n: (rows * ne) as u32,
    });

    // h_mtp = rmsnorm(layer_out, shared_head_norm) — qwen35.cpp:624-631 (`res->t_h_nextn`, ALSO
    // fed back as the next draft step's `h_target` when self-chaining — see `draft`'s doc).
    g.push(Op::RmsNorm {
        x: layer_out,
        weight: w_head_norm,
        dst: h_mtp,
        rows: rows as u32,
        dim: ne as u32,
        eps,
    });
    // logits = lm_head @ h_mtp — qwen35.cpp:633-642. Skipped on the KV-only catch-up path
    // (`logits == None`, AUDIT #2): the vocab-wide GEMM + readback is dead work there.
    if let Some(logits) = logits {
        g.push(Op::Linear {
            x: h_mtp,
            weight: w_lm_head,
            dst: logits,
            m: rows as u32,
            in_f: ne as u32,
            out_f: cfg.vocab as u32,
            w_off: 0,
        });
        // Fused draft-loop accept (issue #33 follow-up, de35727's deferred DRAFT-side twin): argmax
        // + softmax top-1 probability computed on-device, straight off the Internal `logits` this
        // Linear just wrote — 8 bytes cross the bus instead of the whole `[vocab]` row. See
        // `Op::ArgmaxProb`'s doc for the tie-break/numerics contract.
        if let Some((id_out, prob_out)) = id_prob {
            g.push(Op::ArgmaxProb {
                x: logits,
                dst_id: id_out,
                dst_prob: prob_out,
                n: cfg.vocab as u32,
            });
        }
    }

    (
        g,
        MtpHandles {
            e_raw,
            h_in,
            positions,
            k_cache,
            v_cache,
            weights,
            logits,
            h_mtp,
            id_prob,
        },
    )
}

/// [`build_mtp_draft_chain_graph`]'s tensor handles: the inputs bound once per `draft_chain` call
/// (`id0`/`h0`/the `n_steps` `pos_s`/the head's own KV/the 16 weights/the embed table), plus the
/// `n_steps` `Op::ArgmaxProb` id [`Output`](TensorId)s the host reads back after the ONE `execute`.
struct MtpChainHandles {
    id0: TensorId,
    h0: TensorId,
    pos_s: Vec<TensorId>,
    k_cache: TensorId,
    v_cache: TensorId,
    /// The 16 weight handles, same push order as [`MtpHandles::weights`].
    weights: Vec<TensorId>,
    /// The head's own/main-model embedding table, uploaded as a device Weight so
    /// [`Op::EmbedGather`] can gather+dequantize each step's drafted token on-device (see
    /// [`build_embed_chain_buf`]).
    w_embd: TensorId,
    /// The `n_steps` drafted-id outputs, index-parallel with the draft position
    /// (`ids[s]` was drafted at absolute KV position `start_pos + s`).
    ids: Vec<TensorId>,
}

/// Build the MTP head's self-chaining greedy draft loop as ONE unrolled `Graph`: `n_steps` copies
/// of [`build_mtp_graph`]'s `rows == 1, fuse_prob == true` body, back-to-back, with the id/`h_mtp`
/// chain dependency threaded ON DEVICE instead of round-tripping through the host between steps
/// (issue #33 follow-up — see [`MtpHeadSession::draft_chain`]'s doc for the correctness argument
/// and the perf motivation). Step `s`'s drafted id feeds step `s+1`'s [`Op::EmbedGather`] (reading
/// the SAME embedding table the trunk gathers from — `w_embd`); step `s`'s `h_mtp` feeds step
/// `s+1`'s `h_in` directly (an `Internal` tensor, never bound/downloaded). `positions` can't be one
/// shared multi-element input the way `build_mtp_graph`'s batched `rows` case uses it: every step
/// here has `rows == 1`, so `QkNormRope`'s `positions[row]` would read element 0 of a shared buffer
/// every step — `n_steps` separate 1-element `pos_s` inputs sidestep that (the caller uploads
/// `[start_pos + s]` into each once, no per-step re-upload once bound).
///
/// The attention-layer SCRATCH tensors (`e_norm`/`h_norm`/`concat`/`resid`/`hn`/`qg`/`q`/`gate_a`/
/// `k`/`v`/`q16`/`k16`/`attn`/`sub`/`gbuf`/`ubuf`/`actbuf`/`hn2`/`layer_out`) are declared ONCE and
/// REUSED across every step, exactly like [`build_mtp_graph`]'s own `resid` (written by one op,
/// read by a later one, written again) — see that function's doc + `Recorder::sync`'s hazard
/// tracker (`infr-vulkan/src/recorder.rs`): ops execute in `graph.ops` push order and a barrier is
/// auto-inserted only when a real RAW/WAR/WAW hazard exists against work since the last one, so
/// reusing one scratch id across `n_steps` sequential step-bodies is exactly as safe as reusing it
/// across the ops WITHIN one step already is today. Per-step tensors that must NOT be shared
/// (`e_raw_s`/`logits_s`/`h_mtp_s`/`id_s`/its prob scratch) get a fresh id every step instead.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
fn build_mtp_draft_chain_graph(
    cfg: &crate::Config,
    wspecs: &[(DType, usize)],
    embed_dtype: DType,
    max_ctx: usize,
    n_steps: usize,
    start_pos: usize,
) -> (Graph, MtpChainHandles) {
    let ne = cfg.n_embd;
    let nh = cfg.n_head;
    let nkv = cfg.n_kv;
    let hd = cfg.head_dim;
    let qrow = nh * hd;
    let kvrow = nkv * hd;
    let nff = cfg.n_ff;
    let eps = cfg.rms_eps;
    let theta = cfg.rope_theta;
    let rope_dim = cfg.rope_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let vocab = cfg.vocab;

    let mut g = Graph::new();
    let f32d = |n: usize| TensorDesc::new(vec![n], DType::F32);
    let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
    let i32d = |n: usize| TensorDesc::new(vec![n], DType::I32);

    let id0 = g.input(i32d(1));
    let h0 = g.input(f32d(ne));
    let k_cache = g.input(f16d(max_ctx * kvrow));
    let v_cache = g.input(f16d(max_ctx * kvrow));

    let mut wi = 0usize;
    let mut wpush = |g: &mut Graph| -> TensorId {
        let (dt, n) = wspecs[wi];
        wi += 1;
        g.weight(TensorDesc::new(vec![n], dt))
    };
    // MUST match `upload_mtp_head_bufs`'s push order exactly (same 16 weights `build_mtp_graph`
    // uses, unrolled `n_steps` times below against these SAME handles).
    let w_attn_norm = wpush(&mut g);
    let w_q = wpush(&mut g);
    let w_k = wpush(&mut g);
    let w_v = wpush(&mut g);
    let w_qn = wpush(&mut g);
    let w_kn = wpush(&mut g);
    let w_o = wpush(&mut g);
    let w_ffn_norm = wpush(&mut g);
    let w_gate = wpush(&mut g);
    let w_up = wpush(&mut g);
    let w_down = wpush(&mut g);
    let w_eh_proj = wpush(&mut g);
    let w_enorm = wpush(&mut g);
    let w_hnorm = wpush(&mut g);
    let w_head_norm = wpush(&mut g);
    let w_lm_head = wpush(&mut g);
    let weights = vec![
        w_attn_norm,
        w_q,
        w_k,
        w_v,
        w_qn,
        w_kn,
        w_o,
        w_ffn_norm,
        w_gate,
        w_up,
        w_down,
        w_eh_proj,
        w_enorm,
        w_hnorm,
        w_head_norm,
        w_lm_head,
    ];
    let w_embd = g.weight(TensorDesc::new(vec![vocab * ne], embed_dtype));

    let mut pos_s_vec = Vec::with_capacity(n_steps);
    let mut ids_out = Vec::with_capacity(n_steps);
    let mut prev_id = id0;
    let mut prev_h = h0;

    for s in 0..n_steps {
        let pos_s = g.input(i32d(1));
        pos_s_vec.push(pos_s);
        let sp = start_pos + s;

        // ── per-step attention-layer scratch (a fresh id set each step; the recorder serializes
        //    the sequential step bodies via ordinary RAW/WAR hazards) ──
        let e_norm = g.internal(f32d(ne));
        let h_norm = g.internal(f32d(ne));
        let concat = g.internal(f32d(2 * ne));
        let resid = g.internal(f32d(ne));
        let hn = g.internal(f32d(ne));
        let qg = g.internal(f32d(2 * qrow));
        let q = g.internal(f32d(qrow));
        let gate_a = g.internal(f32d(qrow));
        let k = g.internal(f32d(kvrow));
        let v = g.internal(f32d(kvrow));
        let q16 = g.internal(f16d(qrow));
        let k16 = g.internal(f16d(kvrow));
        let attn = g.internal(f32d(qrow));
        let sub = g.internal(f32d(ne));
        let gbuf = g.internal(f32d(nff));
        let ubuf = g.internal(f32d(nff));
        let actbuf = g.internal(f32d(nff));
        let hn2 = g.internal(f32d(ne));
        let layer_out = g.internal(f32d(ne));

        // e_raw_s = table[prev_id, :] * 1.0 — qwen35 isn't gemma, no embed scale (see
        // `build_mtp_graph`'s `forward_draft` twin: the host-embed path uses scale 1.0 too).
        let e_raw_s = g.internal(f32d(ne));
        g.push(Op::EmbedGather {
            ids: prev_id,
            table: w_embd,
            dst: e_raw_s,
            rows: 1,
            ne: ne as u32,
            scale: 1.0,
        });
        let h_in_s = prev_h;

        // e = rmsnorm(embed(t), enorm); h = rmsnorm(h_target, hnorm) — qwen35.cpp:542-546.
        g.push(Op::RmsNorm {
            x: e_raw_s,
            weight: w_enorm,
            dst: e_norm,
            rows: 1,
            dim: ne as u32,
            eps,
        });
        g.push(Op::RmsNorm {
            x: h_in_s,
            weight: w_hnorm,
            dst: h_norm,
            rows: 1,
            dim: ne as u32,
            eps,
        });
        // concat = [e_norm | h_norm] — qwen35.cpp:548 (see `build_mtp_graph`'s doc on this layout).
        g.push(Op::CopyStrided {
            src: e_norm,
            src_off: 0,
            src_stride: ne as u32,
            dst: concat,
            dst_off: 0,
            dst_stride: (2 * ne) as u32,
            rows: 1,
            n: ne as u32,
        });
        g.push(Op::CopyStrided {
            src: h_norm,
            src_off: 0,
            src_stride: ne as u32,
            dst: concat,
            dst_off: ne as u32,
            dst_stride: (2 * ne) as u32,
            rows: 1,
            n: ne as u32,
        });
        // resid = eh_proj @ concat — qwen35.cpp:551.
        g.push(Op::Linear {
            x: concat,
            weight: w_eh_proj,
            dst: resid,
            m: 1,
            in_f: (2 * ne) as u32,
            out_f: ne as u32,
            w_off: 0,
        });

        // ── one qwen35 full-attention layer on `resid` (own 1-layer KV) — qwen35.cpp:556-619 ──
        g.push(Op::RmsNorm {
            x: resid,
            weight: w_attn_norm,
            dst: hn,
            rows: 1,
            dim: ne as u32,
            eps,
        });
        g.push(Op::Linear {
            x: hn,
            weight: w_q,
            dst: qg,
            m: 1,
            in_f: ne as u32,
            out_f: (qrow * 2) as u32,
            w_off: 0,
        });
        for h in 0..nh {
            g.push(Op::CopyStrided {
                src: qg,
                src_off: (h * 2 * hd) as u32,
                src_stride: (nh * 2 * hd) as u32,
                dst: q,
                dst_off: (h * hd) as u32,
                dst_stride: (nh * hd) as u32,
                rows: 1,
                n: hd as u32,
            });
            g.push(Op::CopyStrided {
                src: qg,
                src_off: (h * 2 * hd + hd) as u32,
                src_stride: (nh * 2 * hd) as u32,
                dst: gate_a,
                dst_off: (h * hd) as u32,
                dst_stride: (nh * hd) as u32,
                rows: 1,
                n: hd as u32,
            });
        }
        g.push(Op::Linear {
            x: hn,
            weight: w_k,
            dst: k,
            m: 1,
            in_f: ne as u32,
            out_f: kvrow as u32,
            w_off: 0,
        });
        g.push(Op::Linear {
            x: hn,
            weight: w_v,
            dst: v,
            m: 1,
            in_f: ne as u32,
            out_f: kvrow as u32,
            w_off: 0,
        });
        g.push(Op::QkNormRope {
            x: k,
            weight: w_kn,
            positions: pos_s,
            dst: k16,
            rows: 1,
            n_head: nkv as u32,
            head_dim: hd as u32,
            rope_dim: rope_dim as u32,
            theta,
            eps,
            freq_factors: None,
            x_stride: 0,
        });
        g.push(Op::WriteKv {
            src: k16,
            cache: k_cache,
            rows: 1,
            row_stride: kvrow as u32,
            pos: sp as u32,
        });
        g.push(Op::WriteKv {
            src: v,
            cache: v_cache,
            rows: 1,
            row_stride: kvrow as u32,
            pos: sp as u32,
        });
        g.push(Op::QkNormRope {
            x: q,
            weight: w_qn,
            positions: pos_s,
            dst: q16,
            rows: 1,
            n_head: nh as u32,
            head_dim: hd as u32,
            rope_dim: rope_dim as u32,
            theta,
            eps,
            freq_factors: None,
            x_stride: 0,
        });
        g.push(Op::Attention {
            q: q16,
            k_cache,
            v_cache,
            dst: attn,
            rows: 1,
            kv_len: (sp + 1) as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale,
            mask: AttnMask::Causal,
            pos: sp as u32,
            sinks: None,
        });
        g.push(Op::GatedAct {
            gate: gate_a,
            up: attn,
            dst: attn,
            rows: 1,
            nff: qrow as u32,
            act: Activation::Sigmoid,
            up_off: 0,
            up_stride: 0,
            gate_stride: 0,
            gate_block_width: 0,
            swiglu_clamp: None,
        });
        g.push(Op::Linear {
            x: attn,
            weight: w_o,
            dst: sub,
            m: 1,
            in_f: qrow as u32,
            out_f: ne as u32,
            w_off: 0,
        });
        g.push(Op::Add {
            a: resid,
            b: sub,
            dst: resid,
            n: ne as u32,
        });
        g.push(Op::RmsNorm {
            x: resid,
            weight: w_ffn_norm,
            dst: hn2,
            rows: 1,
            dim: ne as u32,
            eps,
        });
        g.push(Op::Linear {
            x: hn2,
            weight: w_gate,
            dst: gbuf,
            m: 1,
            in_f: ne as u32,
            out_f: nff as u32,
            w_off: 0,
        });
        g.push(Op::Linear {
            x: hn2,
            weight: w_up,
            dst: ubuf,
            m: 1,
            in_f: ne as u32,
            out_f: nff as u32,
            w_off: 0,
        });
        g.push(Op::GatedAct {
            gate: gbuf,
            up: ubuf,
            dst: actbuf,
            rows: 1,
            nff: nff as u32,
            act: Activation::Silu,
            up_off: 0,
            up_stride: 0,
            gate_stride: 0,
            gate_block_width: 0,
            swiglu_clamp: None,
        });
        g.push(Op::Linear {
            x: actbuf,
            weight: w_down,
            dst: sub,
            m: 1,
            in_f: nff as u32,
            out_f: ne as u32,
            w_off: 0,
        });
        g.push(Op::Add {
            a: resid,
            b: sub,
            dst: layer_out,
            n: ne as u32,
        });

        // h_mtp_s = rmsnorm(layer_out, shared_head_norm) — fed to step s+1's h_in AS-IS (internal,
        // never bound/downloaded — see this fn's doc).
        let h_mtp_s = g.internal(f32d(ne));
        g.push(Op::RmsNorm {
            x: layer_out,
            weight: w_head_norm,
            dst: h_mtp_s,
            rows: 1,
            dim: ne as u32,
            eps,
        });
        // logits_s = lm_head @ h_mtp_s — Internal, read straight into Op::ArgmaxProb below (never
        // leaves the device, exactly like `build_mtp_graph(fuse_prob=true)`).
        let logits_s = g.internal(f32d(vocab));
        g.push(Op::Linear {
            x: h_mtp_s,
            weight: w_lm_head,
            dst: logits_s,
            m: 1,
            in_f: ne as u32,
            out_f: vocab as u32,
            w_off: 0,
        });

        // id_s: the drafted token, read back by the host AND fed as step s+1's EmbedGather ids
        // (Op::ArgmaxProb writes a raw u32 index; Op::EmbedGather reads its `ids` input as I32 —
        // same bit pattern, see this fn's doc). prob is discarded (greedy: p_min == 0.0 never
        // early-exits, `draft`'s doc / `DEFAULT_P_MIN`).
        let id_s = g.output(i32d(1));
        let prob_scratch_s = g.internal(f32d(1));
        g.push(Op::ArgmaxProb {
            x: logits_s,
            dst_id: id_s,
            dst_prob: prob_scratch_s,
            n: vocab as u32,
        });

        ids_out.push(id_s);
        prev_id = id_s;
        prev_h = h_mtp_s;
    }

    (
        g,
        MtpChainHandles {
            id0,
            h0,
            pos_s: pos_s_vec,
            k_cache,
            v_cache,
            weights,
            w_embd,
            ids: ids_out,
        },
    )
}

/// A live MTP head: the head's OWN uploaded weights + its OWN 1-layer KV (independent of the
/// trunk's — `docs/mtp.md`'s driver section), driven by `forward`/`catch_up`/`draft`. Borrows the
/// backend and the host embedding table for its lifetime (mirrors `seam.rs`'s session
/// structs) — construct via [`new_cpu`](Self::new_cpu) / [`new_vulkan`](Self::new_vulkan).
///
/// Not cached like `DenoiseCache` (`seam.rs`): `Op::Attention::kv_len` and `Op::WriteKv::pos`
/// are BAKED into the graph at build time (there's no dynamic-`kv_len` binding in this IR — see
/// `docs/mtp.md`'s IR), and both change on essentially every `catch_up`/`draft` call (`draft`
/// advances `kv_len` by one every step). A shape-keyed cache would almost always miss, so
/// `forward` rebuilds + recompiles a fresh graph every call instead — exactly the trunk's default
/// per-token rebuild path for callers that don't opt into `decode_replay`. Phase 3 can revisit if
/// head-forward dispatch overhead matters on the hot serving path (e.g. a `kv_len`/`pos` INPUT
/// binding instead of an inline constant, mirroring `Capabilities::decode_replay`).
pub struct MtpHeadSession<'a> {
    be: &'a dyn Backend,
    cfg: crate::Config,
    /// The engine configuration the head reads its own knobs from — today just
    /// `spec.gpu_draft_prob` ([`gpu_draft_prob_enabled`](Self::gpu_draft_prob_enabled)). Borrowed
    /// from the driver, which borrows it from the `SeamModel` (R4/R6).
    ec: &'a crate::EngineConfig,
    /// The table `forward` gathers embedding rows from (host-side, like every other embed-gather
    /// call site on this seam) — the main model's `token_embd`, or [`resolve_own_embed_table`]'s
    /// result when the GGUF ships its own `nextn.embed_tokens`.
    embed_table: &'a [f32],
    wbufs: Vec<Box<dyn Buffer>>,
    wspecs: Vec<(DType, usize)>,
    k_cache: Box<dyn Buffer>,
    v_cache: Box<dyn Buffer>,
    max_ctx: usize,
    /// The head's own/main-model embedding table, uploaded as a device Weight — `Some` only when
    /// the backend + table dtype support on-device `Op::EmbedGather` (see
    /// [`build_embed_chain_buf`]/[`can_draft_chain`](Self::can_draft_chain)). `None` means
    /// [`draft_chain`](Self::draft_chain) is unavailable and callers must fall back to [`draft`].
    embed_buf: Option<Box<dyn Buffer>>,
    /// The embedding table's effective dtype as uploaded (mirrors `wspecs`'s per-weight dtype) —
    /// meaningless when `embed_buf` is `None`.
    embed_dtype: DType,
    /// Phase 3 perf instrumentation (issue #33, `INFR_PROF_STAGES=1` — see
    /// `generate_mtp_spec_vulkan`'s doc on the per-call rebuild cost this is meant to quantify):
    /// cumulative seconds spent building+compiling a fresh [`Graph`] per [`forward`](Self::forward)
    /// call vs executing it, reset by [`take_timing`](Self::take_timing).
    build_secs: f64,
    exec_secs: f64,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<'a> MtpHeadSession<'a> {
    /// Construct over the CPU reference backend (zero-copy mmap weight upload — mirrors
    /// `seam/model.rs`'s `DiffusionGemmaCpuSession` closures).
    pub fn new_cpu(
        cpu_be: &'a infr_cpu::CpuBackend,
        g: &Gguf,
        cfg: &crate::Config,
        ec: &'a crate::EngineConfig,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        Self::build(
            cpu_be,
            &|_name, tb, dt, _n| match tb {
                crate::seam::WBytes::Mmap(tb) => Ok((cpu_be.map_weight(tb), dt)),
                other => {
                    let v = other.materialize();
                    let buf = cpu_be
                        .alloc(v.len().max(1), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    cpu_be
                        .upload(buf.as_ref(), &v)
                        .map_err(|e| anyhow!("{e}"))?;
                    Ok((buf, dt))
                }
            },
            g,
            cfg,
            ec,
            head,
            embed_table,
            max_ctx,
        )
    }

    /// Construct over the Vulkan backend (padded upload — mirrors `seam/model.rs`'s
    /// `DiffusionGemmaVulkanSession` closures).
    pub fn new_vulkan(
        vk: &'a infr_vulkan::VulkanBackend,
        g: &Gguf,
        cfg: &crate::Config,
        ec: &'a crate::EngineConfig,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        Self::build(
            vk,
            &|_name, tb, dt, _n| {
                let bytes = tb.materialize();
                let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
                let buf = vk
                    .alloc(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                vk.upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            g,
            cfg,
            ec,
            head,
            embed_table,
            max_ctx,
        )
    }

    /// Construct over the Metal backend (raw native-dtype upload — mirrors
    /// `seam.rs`'s `generate_dense_metal_session` weight closure).
    #[cfg(target_os = "macos")]
    pub fn new_metal(
        mtl: &'a infr_metal::MetalBackend,
        g: &Gguf,
        cfg: &crate::Config,
        ec: &'a crate::EngineConfig,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        Self::build(
            mtl,
            &|_name, tb, dt, _n| {
                let buf = mtl
                    .alloc(tb.len().max(1), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                mtl.upload(buf.as_ref(), &tb.materialize())
                    .map_err(|e| anyhow!("{e}"))?;
                Ok((buf, dt))
            },
            g,
            cfg,
            ec,
            head,
            embed_table,
            max_ctx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        be: &'a dyn Backend,
        bind_weight: &BindWeightFn,
        g: &Gguf,
        cfg: &crate::Config,
        ec: &'a crate::EngineConfig,
        head: &MtpHeadWeights,
        embed_table: &'a [f32],
        max_ctx: usize,
    ) -> Result<Self> {
        let (wbufs, wspecs) = upload_mtp_head_bufs(be, bind_weight, g, cfg, head)?;
        let kvrow = cfg.n_kv * cfg.head_dim;
        let k_cache = be
            .alloc(max_ctx * kvrow * 2, BufferUsage::KvCache) // f16 KV — 2 bytes/elem
            .map_err(|e| anyhow!("{e}"))?;
        let v_cache = be
            .alloc(max_ctx * kvrow * 2, BufferUsage::KvCache)
            .map_err(|e| anyhow!("{e}"))?;
        let (embed_buf, embed_dtype) = build_embed_chain_buf(be, bind_weight, g, cfg, head)?;
        Ok(Self {
            be,
            cfg: cfg.clone(),
            ec,
            embed_table,
            wbufs,
            wspecs,
            k_cache,
            v_cache,
            max_ctx,
            embed_buf,
            embed_dtype,
            build_secs: 0.0,
            exec_secs: 0.0,
        })
    }

    /// Drain the cumulative build-vs-exec timing since the last call (or construction) —
    /// `generate_mtp_spec_vulkan`'s `INFR_PROF_STAGES=1` breakdown reads this once per spec cycle.
    pub fn take_timing(&mut self) -> (f64, f64) {
        (
            std::mem::take(&mut self.build_secs),
            std::mem::take(&mut self.exec_secs),
        )
    }

    /// One head forward over `rows = tokens.len()` `(token, h_target)` pairs at absolute KV
    /// positions `start_pos..start_pos+rows` (`docs/mtp.md`'s forward pseudocode — see
    /// `build_mtp_graph`'s doc for the op-level port). Returns `(logits [rows*vocab], h_mtp
    /// [rows*ne])`. `WriteKv` OVERWRITES any existing rows at these positions (a re-draft's next
    /// `catch_up`/`draft` at the same positions is expected to do exactly this — `docs/mtp.md`'s
    /// driver section, last paragraph).
    pub fn forward(
        &mut self,
        tokens: &[u32],
        h_rows: &[f32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let rows = tokens.len();
        let ne = self.cfg.n_embd;
        anyhow::ensure!(
            h_rows.len() == rows * ne,
            "MTP head forward: h_rows length {} != rows*ne {}",
            h_rows.len(),
            rows * ne
        );
        anyhow::ensure!(
            start_pos + rows <= self.max_ctx,
            "MTP head forward: KV overflow ({start_pos}+{rows} > max_ctx {})",
            self.max_ctx
        );

        let mut e_raw = Vec::with_capacity(rows * ne);
        for &t in tokens {
            let base = t as usize * ne;
            e_raw.extend_from_slice(&self.embed_table[base..base + ne]);
        }
        let positions: Vec<i32> = (start_pos as i32..(start_pos + rows) as i32).collect();

        let t_build = std::time::Instant::now();
        let (graph, h) = build_mtp_graph(
            &self.cfg,
            &self.wspecs,
            self.max_ctx,
            rows,
            start_pos,
            false,
            want_logits,
        );
        let plan = self.be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
        self.build_secs += t_build.elapsed().as_secs_f64();

        let t_exec = std::time::Instant::now();
        let e_buf = self
            .be
            .alloc(rows * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let h_buf = self
            .be
            .alloc(rows * ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let pos_buf = self
            .be
            .alloc(rows * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        // KV-only catch-up (`want_logits == false`, AUDIT #2): no lm_head output to read back.
        let logits_buf = match h.logits {
            Some(_) => Some(
                self.be
                    .alloc(rows * self.cfg.vocab * 4, BufferUsage::Readback)
                    .map_err(|e| anyhow!("{e}"))?,
            ),
            None => None,
        };
        let hmtp_buf = self
            .be
            .alloc(rows * ne * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;

        self.be
            .upload(e_buf.as_ref(), bytemuck::cast_slice(&e_raw))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(h_buf.as_ref(), bytemuck::cast_slice(h_rows))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(pos_buf.as_ref(), bytemuck::cast_slice(&positions))
            .map_err(|e| anyhow!("{e}"))?;

        let mut b = Bindings::new();
        b.bind(h.e_raw, e_buf.as_ref());
        b.bind(h.h_in, h_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
        b.bind(h.k_cache, self.k_cache.as_ref());
        b.bind(h.v_cache, self.v_cache.as_ref());
        for (i, wid) in h.weights.iter().enumerate() {
            b.bind(*wid, self.wbufs[i].as_ref());
        }
        if let (Some(lid), Some(lbuf)) = (h.logits, logits_buf.as_ref()) {
            b.bind(lid, lbuf.as_ref());
        }
        b.bind(h.h_mtp, hmtp_buf.as_ref());

        self.be
            .execute(plan.as_ref(), &b)
            .map_err(|e| anyhow!("{e}"))?;

        let mut logits = Vec::new();
        if let Some(lbuf) = logits_buf.as_ref() {
            logits = vec![0f32; rows * self.cfg.vocab];
            self.be
                .download(lbuf.as_ref(), bytemuck::cast_slice_mut(&mut logits))
                .map_err(|e| anyhow!("{e}"))?;
        }
        let mut h_mtp = vec![0f32; rows * ne];
        self.be
            .download(hmtp_buf.as_ref(), bytemuck::cast_slice_mut(&mut h_mtp))
            .map_err(|e| anyhow!("{e}"))?;
        self.exec_secs += t_exec.elapsed().as_secs_f64();
        Ok((logits, h_mtp))
    }

    /// One SELF-CHAINING draft step (`draft`'s inner loop body): forward `tok`/`h_row` at `pos`
    /// and return `(id, prob, h_mtp)` — the drafted token id, its softmax top-1 probability, and
    /// the head's own hidden output to feed the NEXT step. Issue #33 follow-up to de35727's
    /// GPU-resident VERIFY accept: when the backend advertises `Capabilities::argmax_prob` (and
    /// `INFR_NO_GPU_DRAFT_PROB` isn't set), builds the FUSED graph (`build_mtp_graph`'s
    /// `fuse_prob`) so only 8 bytes read back instead of the whole `[vocab]` logits row — the
    /// profiler-measured dominant per-step draft cost was the host `top1_softmax` scan this
    /// replaces (~650-700us on a 151936-token vocab), not the download bytes themselves
    /// (~25-50us), so the fused reduce kills both in one pass. Falls back to
    /// [`forward`](Self::forward) + [`top1_softmax`] otherwise (Metal, or the A/B escape) —
    /// bit-identical results either way (both compute the SAME argmax; `DEFAULT_P_MIN == 0.0`
    /// means the `prob` value alone can never flip a `prob < p_min` decision, so the two paths'
    /// accept/reject/token stream is provably identical on the shipped default — see
    /// `Op::ArgmaxProb`'s doc on the reduction-order caveat for a future nonzero `p_min`).
    fn forward_draft(
        &mut self,
        tok: u32,
        h_row: &[f32],
        pos: usize,
    ) -> Result<(u32, f32, Vec<f32>)> {
        if !self.gpu_draft_prob_enabled() {
            let (logits, h_mtp) = self.forward(&[tok], h_row, pos, true)?;
            let (id, p) = top1_softmax(&logits);
            return Ok((id, p, h_mtp));
        }

        let ne = self.cfg.n_embd;
        anyhow::ensure!(
            h_row.len() == ne,
            "MTP head forward_draft: h_row length {} != ne {ne}",
            h_row.len()
        );
        anyhow::ensure!(
            pos < self.max_ctx,
            "MTP head forward_draft: KV overflow ({pos}+1 > max_ctx {})",
            self.max_ctx
        );

        let base = tok as usize * ne;
        let e_raw = &self.embed_table[base..base + ne];
        let positions = [pos as i32];

        let t_build = std::time::Instant::now();
        let (graph, h) = build_mtp_graph(&self.cfg, &self.wspecs, self.max_ctx, 1, pos, true, true);
        let plan = self.be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
        self.build_secs += t_build.elapsed().as_secs_f64();

        let t_exec = std::time::Instant::now();
        let e_buf = self
            .be
            .alloc(ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let h_buf = self
            .be
            .alloc(ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let pos_buf = self
            .be
            .alloc(4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let hmtp_buf = self
            .be
            .alloc(ne * 4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let id_buf = self
            .be
            .alloc(4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;
        let prob_buf = self
            .be
            .alloc(4, BufferUsage::Readback)
            .map_err(|e| anyhow!("{e}"))?;

        self.be
            .upload(e_buf.as_ref(), bytemuck::cast_slice(e_raw))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(h_buf.as_ref(), bytemuck::cast_slice(h_row))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(pos_buf.as_ref(), bytemuck::cast_slice(&positions))
            .map_err(|e| anyhow!("{e}"))?;

        let (id_out, prob_out) = h
            .id_prob
            .expect("build_mtp_graph(fuse_prob=true) always sets id_prob");

        let mut b = Bindings::new();
        b.bind(h.e_raw, e_buf.as_ref());
        b.bind(h.h_in, h_buf.as_ref());
        b.bind(h.positions, pos_buf.as_ref());
        b.bind(h.k_cache, self.k_cache.as_ref());
        b.bind(h.v_cache, self.v_cache.as_ref());
        for (i, wid) in h.weights.iter().enumerate() {
            b.bind(*wid, self.wbufs[i].as_ref());
        }
        // NOTE: `h.logits` is Internal in the fused graph (never bound — see build_mtp_graph's
        // doc); only h_mtp + the two 8-byte accept outputs cross the bus.
        b.bind(h.h_mtp, hmtp_buf.as_ref());
        b.bind(id_out, id_buf.as_ref());
        b.bind(prob_out, prob_buf.as_ref());

        self.be
            .execute(plan.as_ref(), &b)
            .map_err(|e| anyhow!("{e}"))?;

        let mut h_mtp = vec![0f32; ne];
        self.be
            .download(hmtp_buf.as_ref(), bytemuck::cast_slice_mut(&mut h_mtp))
            .map_err(|e| anyhow!("{e}"))?;
        // Device-side accept ids/probs (mirrors the decode loop's Op::Argmax/Op::Sample readback
        // in seam/runner.rs): raw 4-byte little-endian id, plain f32 prob.
        let mut idb = [0u8; 4];
        self.be
            .download(id_buf.as_ref(), &mut idb)
            .map_err(|e| anyhow!("{e}"))?;
        let id = u32::from_le_bytes(idb);
        let mut pb = [0u8; 4];
        self.be
            .download(prob_buf.as_ref(), &mut pb)
            .map_err(|e| anyhow!("{e}"))?;
        let prob = f32::from_le_bytes(pb);
        self.exec_secs += t_exec.elapsed().as_secs_f64();

        Ok((id, prob, h_mtp))
    }

    /// Whether [`forward_draft`](Self::forward_draft) should build the fused `Op::ArgmaxProb`
    /// graph: the backend must advertise the capability, and the `spec.gpu_draft_prob` A/B escape
    /// (`INFR_NO_GPU_DRAFT_PROB`, inverted — byte-identity verification against the host
    /// `top1_softmax` path) must be on.
    fn gpu_draft_prob_enabled(&self) -> bool {
        self.be.capabilities().argmax_prob && self.ec.spec.gpu_draft_prob
    }

    /// Whether [`draft_chain`](Self::draft_chain) can run: the head's embedding table uploaded as
    /// a device Weight (see [`build_embed_chain_buf`]) — `None` when the backend/dtype doesn't
    /// support on-device `Op::EmbedGather` for this table (Metal today, or an unsupported quant
    /// format). Callers gate the fused-chain fast path on this and fall back to [`draft`].
    pub fn can_draft_chain(&self) -> bool {
        self.embed_buf.is_some()
    }

    /// The self-chained greedy draft loop (issue #33 follow-up to `forward_draft`'s per-step
    /// fusion): unrolls `n_steps` copies of the head forward into ONE [`Graph`]
    /// ([`build_mtp_draft_chain_graph`]) — id/`h_mtp` chained device-side via
    /// `Op::EmbedGather`/internal tensors instead of the old `draft()`'s `n_steps` sequential
    /// submit→wait→readback round-trips — compiles + executes it ONCE, and downloads the
    /// `n_steps` drafted ids in one pass. Measured baseline: `n_max = 6` round-trips cost ~9ms/
    /// cycle on Qwen3.5-9B, ~0.9ms/step of which is pure submit/wait/readback/alloc overhead (not
    /// compute) — fusing to one submission is expected to roughly halve draft wall time.
    ///
    /// ## Correctness
    /// `crate::seam::model::spec_accept` (the VERIFY step every caller of `draft`/`draft_chain`
    /// runs the candidates through) only ever COMMITS a token the trunk's own greedy argmax
    /// independently confirms — the draft candidates are a proposal, never committed directly.
    /// That means the draft head's numerics can only ever affect α (how many of its guesses the
    /// trunk happens to agree with), NEVER which tokens land in the output stream: the committed
    /// stream is provably trunk-greedy regardless of whether the draft ran as `n_steps` separate
    /// calls or one fused chain. The fixed `n_steps`-long unroll (no early exit mid-chain) is valid
    /// because greedy drafting uses `DEFAULT_P_MIN == 0.0` and a softmax top-1 probability is
    /// always `> 0.0`, so `draft`'s `if p < p_min { break }` NEVER fires in practice — every greedy
    /// draft call already runs the full `n_max` steps today, so unrolling exactly that many steps
    /// changes nothing observable. Finally, `Op::EmbedGather` here reads the SAME embedding table
    /// (`w_embd`, uploaded from the identical GGUF tensor `build_embed_chain_buf` resolves) the
    /// host-side `forward_draft`/`draft` path reads via `embed_table` — same bytes, same dequant —
    /// so α should match or improve, never regress from a table mismatch.
    ///
    /// Requires [`can_draft_chain`](Self::can_draft_chain); panics (via `expect`) if called
    /// without checking it first — callers MUST gate on `can_draft_chain()` (see
    /// `generate_mtp_spec_core`'s greedy branch).
    pub fn draft_chain(
        &mut self,
        id_last: u32,
        pending_h: &[f32],
        n_past: usize,
        n_steps: usize,
    ) -> Result<Vec<u32>> {
        let ne = self.cfg.n_embd;
        anyhow::ensure!(
            pending_h.len() == ne,
            "MTP head draft_chain: pending_h length {} != ne {ne}",
            pending_h.len()
        );
        anyhow::ensure!(
            n_past + n_steps <= self.max_ctx,
            "MTP head draft_chain: KV overflow ({n_past}+{n_steps} > max_ctx {})",
            self.max_ctx
        );
        anyhow::ensure!(n_steps > 0, "MTP head draft_chain: n_steps must be > 0");

        let t_build = std::time::Instant::now();
        let (graph, h) = build_mtp_draft_chain_graph(
            &self.cfg,
            &self.wspecs,
            self.embed_dtype,
            self.max_ctx,
            n_steps,
            n_past,
        );
        let plan = self.be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
        self.build_secs += t_build.elapsed().as_secs_f64();

        let t_exec = std::time::Instant::now();
        let id0_buf = self
            .be
            .alloc(4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let h0_buf = self
            .be
            .alloc(ne * 4, BufferUsage::Staging)
            .map_err(|e| anyhow!("{e}"))?;
        let mut pos_bufs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            pos_bufs.push(
                self.be
                    .alloc(4, BufferUsage::Staging)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }
        let mut id_bufs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            id_bufs.push(
                self.be
                    .alloc(4, BufferUsage::Readback)
                    .map_err(|e| anyhow!("{e}"))?,
            );
        }

        self.be
            .upload(id0_buf.as_ref(), bytemuck::cast_slice(&[id_last as i32]))
            .map_err(|e| anyhow!("{e}"))?;
        self.be
            .upload(h0_buf.as_ref(), bytemuck::cast_slice(pending_h))
            .map_err(|e| anyhow!("{e}"))?;
        for (s, pos_buf) in pos_bufs.iter().enumerate() {
            let p = (n_past + s) as i32;
            self.be
                .upload(pos_buf.as_ref(), bytemuck::cast_slice(&[p]))
                .map_err(|e| anyhow!("{e}"))?;
        }

        let embed_buf = self
            .embed_buf
            .as_ref()
            .expect("draft_chain requires can_draft_chain() to be checked by the caller")
            .as_ref();

        let mut b = Bindings::new();
        b.bind(h.id0, id0_buf.as_ref());
        b.bind(h.h0, h0_buf.as_ref());
        b.bind(h.k_cache, self.k_cache.as_ref());
        b.bind(h.v_cache, self.v_cache.as_ref());
        for (i, wid) in h.weights.iter().enumerate() {
            b.bind(*wid, self.wbufs[i].as_ref());
        }
        b.bind(h.w_embd, embed_buf);
        for (pos_id, pos_buf) in h.pos_s.iter().zip(pos_bufs.iter()) {
            b.bind(*pos_id, pos_buf.as_ref());
        }
        for (id_id, id_buf) in h.ids.iter().zip(id_bufs.iter()) {
            b.bind(*id_id, id_buf.as_ref());
        }

        self.be
            .execute(plan.as_ref(), &b)
            .map_err(|e| anyhow!("{e}"))?;

        let mut out = Vec::with_capacity(n_steps);
        for buf in &id_bufs {
            let mut idb = [0u8; 4];
            self.be
                .download(buf.as_ref(), &mut idb)
                .map_err(|e| anyhow!("{e}"))?;
            out.push(u32::from_le_bytes(idb));
        }
        self.exec_secs += t_exec.elapsed().as_secs_f64();

        Ok(out)
    }
}

/// llama.cpp's default `--spec-draft-p-min` (`common/common.h:329`'s
/// `common_params_speculative_draft::p_min = 0.0f`) — i.e. `n_max` alone bounds a draft run unless
/// the caller raises it. Matches `docs/mtp.md`'s captured oracle run (no `--spec-draft-p-min` flag
/// was passed for the 2.0x number).
pub const DEFAULT_P_MIN: f32 = 0.0;

/// The MTP catch-up hook (`docs/mtp.md`'s `process()`, `speculative.cpp:1354-1470`'s single-head
/// branch): re-syncs the head's own KV to the target's committed rows after EVERY target
/// prefill/decode ubatch. `tokens[i]` pairs with `h_rows[i]` — the CALLER does the "shift `h` right
/// by one + splice in the previous call's `pending_h`" (`speculative.cpp:1396-1417`'s `h_tgt`
/// memcpy + `set_h`), so this primitive stays a plain "forward these `R` `(token, h)` pairs at
/// `start_pos`, discard the logits" call with no special-casing of the first row. The head's own KV
/// rows `start_pos..start_pos+tokens.len()` are (over)written — see `forward`'s doc on the
/// overwrite semantics a re-draft relies on. Runs the KV-only forward (`want_logits == false`,
/// AUDIT #2): only the `WriteKv` ops matter here, so the lm_head `[rows*vocab]` GEMM + its readback
/// are omitted — catch_up discarded that logits row every call.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn catch_up(
    sess: &mut MtpHeadSession,
    tokens: &[u32],
    h_rows: &[f32],
    start_pos: usize,
) -> Result<()> {
    sess.forward(tokens, h_rows, start_pos, false)?;
    Ok(())
}

/// The MTP draft loop (`docs/mtp.md`'s `draft()`, `speculative.cpp:1472-1621`'s single-head, non-
/// chained, non-mem-shared branch): self-chaining greedy decode over the head's OWN KV, ONE row per
/// step. Starts at `(id_last, pending_h)` at absolute position `n_past` (the reference's
/// `dp.n_past`/`dp.id_last` — the first UNCOMMITTED position and the last committed token); each
/// accepted step re-feeds `(sampled_id, h_mtp)` — the head's OWN hidden output, NOT the target's —
/// at the next position (`speculative.cpp:1542`'s `h_row = llama_get_embeddings_nextn_ith(ctx_dft,
/// ...)`), so the head's KV grows by exactly the rows drafted. Stops at the first row whose top-1
/// probability is `< p_min` (`speculative.cpp:1556`) or once `n_max` tokens are drafted
/// (`speculative.cpp:1570`) — plain greedy argmax either way (temp=0 throughout `docs/mtp.md`'s
/// validated oracle run; the reference's draft sampler is `top_k=10` but only ever reads
/// `data[0]`). Returns `(token, top1_prob)` pairs so a caller can inspect confidence without a
/// second pass; re-drafting from the same `n_past` OVERWRITES the head's KV at those positions
/// (`forward`'s doc) — the caller doesn't need to reset anything first.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn draft(
    sess: &mut MtpHeadSession,
    id_last: u32,
    pending_h: &[f32],
    n_past: usize,
    p_min: f32,
    n_max: usize,
) -> Result<Vec<(u32, f32)>> {
    let mut result = Vec::with_capacity(n_max);
    let mut tok = id_last;
    let mut h = pending_h.to_vec();
    for pos in n_past..n_past + n_max {
        // Issue #33 follow-up (de35727's deferred DRAFT-side twin): `forward_draft` builds the
        // fused `Op::ArgmaxProb` graph when the backend supports it (8-byte readback instead of
        // the whole `[vocab]` logits row + a host argmax/softmax scan), falling back to `forward`
        // + `top1_softmax` otherwise — same `(id, p)` either way, see that method's doc.
        let (id, p, h_mtp) = sess.forward_draft(tok, &h, pos)?;
        if p < p_min {
            break;
        }
        result.push((id, p));
        tok = id;
        h = h_mtp; // self-chain: the head's OWN h_mtp feeds the next draft step
    }
    Ok(result)
}

/// A truncated (top-k/top-p, temperature-scaled) distribution: `(vocab id, normalized prob)` pairs
/// summing to 1 — [`sampling::truncated_dist`](crate::sampling::truncated_dist)'s return shape,
/// named here purely to keep clippy's `type_complexity` lint quiet on [`draft_stochastic`]'s
/// signature.
type TruncDist = Vec<(u32, f32)>;

/// Temperature-aware draft loop — [`draft`]'s stochastic sibling, used when `INFR_TEMP > 0` (see
/// `crate::seam::model::spec_accept_stochastic`'s doc for why greedy-only MTP degenerates on
/// thinking models). Self-chaining SAMPLE (not argmax) decode over the head's own KV, one row per
/// step: each step calls [`MtpHeadSession::forward`] (the full-logits path — NOT
/// [`MtpHeadSession::forward_draft`]'s fused GPU `Op::ArgmaxProb`, which only surfaces a top-1
/// id+prob and can't support the accept rule's full-distribution ratio/residual test), truncates
/// the head's logits with `sampling::truncated_dist` at the SAME `sampler` config the caller's
/// target verify uses, and draws the drafted id from that truncated distribution via the shared
/// `rng` stream. Returns `(id, q_i)` pairs — the drafted id AND its full truncated proposal
/// distribution `q_i`, since the caller's accept rule needs `q_i(x)` for arbitrary `x` (the
/// residual-resample case), not just the drafted token's own probability.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn draft_stochastic(
    sess: &mut MtpHeadSession,
    id_last: u32,
    pending_h: &[f32],
    n_past: usize,
    n_max: usize,
    sampler: crate::sampling::Sampler,
    rng: &mut u64,
) -> Result<Vec<(u32, TruncDist)>> {
    let mut result = Vec::with_capacity(n_max);
    let mut tok = id_last;
    let mut h = pending_h.to_vec();
    for pos in n_past..n_past + n_max {
        let (logits, h_mtp) = sess.forward(&[tok], &h, pos, true)?;
        let dist = crate::sampling::truncated_dist(&logits, sampler);
        let id = crate::sampling::sample_from_dist(&dist, rng);
        result.push((id, dist));
        tok = id;
        h = h_mtp; // self-chain: the head's OWN h_mtp feeds the next draft step
    }
    Ok(result)
}

/// Greedy top-1 token id + its softmax probability over one `[vocab]` logits row
/// (`speculative.cpp`'s `common_sampler_sample`/`cur_p->data[0]`): since only the winning
/// candidate's OWN probability mass is needed (not the runner-up ids a real top-k sampler would
/// keep), argmax + `1/Σexp(logit - max)` is exactly `softmax(logits)[argmax]` without materializing
/// the full distribution.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn top1_softmax(logits: &[f32]) -> (u32, f32) {
    let (mut amax, mut m) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > m {
            m = v;
            amax = i;
        }
    }
    let z: f32 = logits.iter().map(|&v| (v - m).exp()).sum();
    (amax as u32, 1.0 / z)
}

// ─── Phase 3: the MTP self-speculative generation loop + run/serve wiring (issue #33) ───────────

/// llama.cpp's oracle run used `--spec-draft-n-max 6` (`docs/mtp.md`) — the max candidates drafted
/// per cycle before a batched verify.
pub const DEFAULT_N_MAX: usize = 6;

/// One batched VERIFY forward (`seam.rs`'s VERIFY branch) over `tokens`' un-cached suffix,
/// on a persistent Vulkan session, ALSO capturing `h` for every returned row — the primitive
/// [`generate_mtp_spec_vulkan`] needs every cycle (`crate::seam::model::SeamModel`'s own
/// `verify_dense_*_with_h` helpers all use a FRESH one-shot session; this one threads `state`
/// across calls so the trunk's KV/DeltaNet state persists cycle to cycle — the whole point of
/// self-speculative decoding). Returns `(argmax ids [m], h [m*n_embd])`, `m = tokens.len() -
/// (tokens already cached in `state`)`.
#[allow(clippy::too_many_arguments)]
/// One TARGET-trunk verify forward over `tokens`, returning (per-row greedy argmax ids `[m]`,
/// all-row hidden states `[m*n_embd]`) — the hidden rows are the extra tap the MTP head
/// consumes. Backend-generic: the trunk graph is `generate_dense_backend` (which already takes
/// `be` + a bind closure), so this drives Vulkan, Metal, or CPU by the caller's binder.
///
/// GPU-resident accept (issue #31): the runner's VERIFY branch appends a per-row `Op::Argmax`
/// when the backend supports it (`Capabilities::argmax_rows`; A/B escape via INFR_NO_GPU_ARGMAX
/// or INFR_NO_GPU_MTP_ACCEPT) so only m×4 id bytes cross the bus, not the m×vocab f32 logits.
/// When the device path is off (Metal, env A/B), the runner fills `logits` instead (ids left
/// empty) and THIS function does the exact host argmax per row — either way the caller sees the
/// same `[m]` id vector, bit-identical (both paths argmax the same f32 logits with strict-`>`
/// lowest-index tie-break).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn run_verify(
    be: &dyn Backend,
    bind: &BindWeightFn,
    g: &Gguf,
    cfg: &crate::Config,
    ec: &crate::EngineConfig,
    token_embd: crate::seam::TokenEmbd<'_>,
    tokens: &[u32],
    state: &mut Option<crate::seam::SeamKv>,
    max_ctx: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut logits = Vec::new();
    let mut ids = Vec::new();
    let mut h = Vec::new();
    crate::seam::generate_dense_backend(
        be,
        bind,
        g,
        cfg,
        ec,
        token_embd,
        None,
        tokens,
        0,
        |_| {},
        state,
        max_ctx,
        None,
        Some(&mut logits),
        Some(&mut ids),
        None,
        Some(&mut h),
        None,
        None,
        None,
    )?;
    if ids.is_empty() {
        // Host fallback: the runner downloaded the m×vocab logits instead (see this fn's doc).
        let m = h.len() / cfg.n_embd;
        anyhow::ensure!(
            logits.len() == m * cfg.vocab,
            "run_verify: host-logits fallback expected {}*{} logits, got {}",
            m,
            cfg.vocab,
            logits.len()
        );
        ids = (0..m)
            .map(|j| argmax_row(&logits[j * cfg.vocab..(j + 1) * cfg.vocab]))
            .collect();
    }
    Ok((ids, h))
}

/// Full-distribution VERIFY forward — the temperature-aware MTP accept rule's twin of
/// [`run_verify`]. Identical batched trunk forward, except it passes `verify_ids: None` to
/// `generate_dense_backend`, which disables the GPU-resident argmax-only accept path (see that
/// param's doc) so the runner ALWAYS downloads the full `[m, vocab]` logits instead of `m` argmax
/// ids. `run_verify`'s fused id-only GPU accept can't support
/// `crate::seam::model::spec_accept_stochastic`'s importance-ratio test (it needs each row's full
/// truncated distribution, not just the argmax id), so temp>0 pays the full `m*vocab*4`-byte
/// download every verify cycle instead of `m*4` bytes — the perf tradeoff noted on
/// [`generate_mtp_spec_core`]'s doc. Returns `(logits [m*vocab], h [m*n_embd])`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn run_verify_full(
    be: &dyn Backend,
    bind: &BindWeightFn,
    g: &Gguf,
    cfg: &crate::Config,
    ec: &crate::EngineConfig,
    token_embd: crate::seam::TokenEmbd<'_>,
    tokens: &[u32],
    state: &mut Option<crate::seam::SeamKv>,
    max_ctx: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut logits = Vec::new();
    let mut h = Vec::new();
    crate::seam::generate_dense_backend(
        be,
        bind,
        g,
        cfg,
        ec,
        token_embd,
        None,
        tokens,
        0,
        |_| {},
        state,
        max_ctx,
        None,
        Some(&mut logits),
        None, // verify_ids: None forces the full-logits download path (see this fn's doc)
        None,
        Some(&mut h),
        None,
        None,
        None,
    )?;
    Ok((logits, h))
}

/// Re-primes the trunk's KV/DeltaNet state over `tokens`' un-cached suffix WITHOUT paying
/// [`run_verify`]/[`run_verify_full`]'s full-batch lm_head cost — the reprime call site
/// (`mtp_reprime`'s doc on [`generate_mtp_spec_core`]) only ever consumes the LAST row's id (or
/// full distribution) and `h`, never the other `mr-1` rows [`run_verify`] computes the vocab-wide
/// `Op::Linear` over. This drives the ORDINARY (non-VERIFY) per-token decode path instead —
/// `generate_dense_backend`'s `logits_rows == 1` lm_head tail already extracts just the frontier
/// row via a `Copy` before the `Linear` (`runner.rs`'s `lm_in` selection), so the wasted rows are
/// simply never computed, not computed-then-discarded. `max_new = 1` runs the model's own
/// greedy/stochastic sampling for that one row: greedy reads back `ids[0]` (the same
/// `Op::Argmax`-with-strict-`>`-tie-break every ordinary decode step already uses — bit-identical
/// to [`run_verify`]'s per-row argmax by construction, since it is the SAME kernel path); passing
/// `logits_out` additionally captures the row's raw logits (needed for the stochastic accept
/// rule's truncated distribution) without disturbing the sample. Using the exact path plain
/// (non-MTP) decode already runs for this shape is deliberate: it cannot introduce a NEW
/// precision asymmetry between the spec and non-spec streams (unlike, say, routing this row
/// through a small-batch GEMV kernel plain decode never touches — an int8-quantized-activation
/// mrow tier for Q5_K was tried as a *different* MTP verify lever earlier in this same campaign
/// and rejected for exactly that reason: it made MTP's own verify batch numerically diverge from
/// this fn's single-row path, occasionally flipping a greedy token). Returns `(id, logits or
/// empty, h[ne])` — `logits` is only populated when `want_logits` (the stochastic flavor).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn run_prime_last(
    be: &dyn Backend,
    bind: &BindWeightFn,
    g: &Gguf,
    cfg: &crate::Config,
    ec: &crate::EngineConfig,
    token_embd: crate::seam::TokenEmbd<'_>,
    tokens: &[u32],
    state: &mut Option<crate::seam::SeamKv>,
    max_ctx: usize,
    want_logits: bool,
) -> Result<(u32, Vec<f32>, Vec<f32>)> {
    let mut logits = Vec::new();
    let mut h = Vec::new();
    let (ids, _) = crate::seam::generate_dense_backend(
        be,
        bind,
        g,
        cfg,
        ec,
        token_embd,
        None,
        tokens,
        1,
        |_| {},
        state,
        max_ctx,
        None,
        None,
        None,
        want_logits.then_some(&mut logits),
        Some(&mut h),
        None,
        None,
        None,
    )?;
    anyhow::ensure!(
        ids.len() == 1 && h.len() == cfg.n_embd,
        "run_prime_last: expected 1 id and {} h floats, got {} ids, {} h floats",
        cfg.n_embd,
        ids.len(),
        h.len()
    );
    Ok((ids[0], logits, h))
}

/// Cumulative per-phase wall time + accept-rate counters over one [`generate_mtp_spec_vulkan_timed`]
/// run (issue #33, phase 4 — `infr bench`/`infr compare`'s perf-bottleneck visibility pass: this is
/// the struct return `docs/mtp.md`'s `INFR_PROF_STAGES` per-cycle `eprintln!`s are refactored to feed,
/// instead of the caller scraping stderr). `draft_secs`/`verify_secs`/`catchup_secs` sum every
/// cycle's three timed sections (the SAME three the `[mtp cycle N]` debug line prints); the one-time
/// prompt-prime VERIFY forward is deliberately excluded — it's already `GenStats::prompt_secs`, not
/// part of the steady-state decode this breakdown characterizes.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtpTiming {
    pub draft_secs: f64,
    pub verify_secs: f64,
    pub catchup_secs: f64,
    pub total_drafted: usize,
    pub total_accepted: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl MtpTiming {
    /// Fold another run's counters in (bench averages several reps — see `infr-cli`'s
    /// `bench_mtp_tg`).
    pub fn add(&mut self, other: &MtpTiming) {
        self.draft_secs += other.draft_secs;
        self.verify_secs += other.verify_secs;
        self.catchup_secs += other.catchup_secs;
        self.total_drafted += other.total_drafted;
        self.total_accepted += other.total_accepted;
    }

    /// Draft acceptance rate (`accepted / drafted`) — llama.cpp's `alpha`.
    pub fn alpha(&self) -> f64 {
        self.total_accepted as f64 / self.total_drafted.max(1) as f64
    }

    /// `(draft%, verify%, catchup%)` of the three phases' summed wall time — the breakdown that
    /// tells the next perf pass where the net-negative time actually goes.
    pub fn phase_shares(&self) -> (f64, f64, f64) {
        let total = self.draft_secs + self.verify_secs + self.catchup_secs;
        if total <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (
            100.0 * self.draft_secs / total,
            100.0 * self.verify_secs / total,
            100.0 * self.catchup_secs / total,
        )
    }
}

/// The base row of the NON-leading verify branch (`generate_mtp_spec_core`'s cycle-2+ path): the
/// verify feed is `committed ++ cand`, and after a re-prefill the trunk returns `m` rows whose LAST
/// `cand.len() + 1` are the real leading row (position `n_past-1`, re-confirming `cand[0]`) followed
/// by the `cand.len()` draft rows — so the leading row sits at `base = m - (cand.len() + 1)`.
///
/// AUDIT #4: guards the `usize` subtraction. The `debug_assert_eq!(m, cand.len())` on the LEADING
/// branches never covers this one; a state edge with `m < cand.len() + 1` would wrap to a huge index
/// and panic on the subsequent slice. Return an `Err` instead of underflowing.
fn nonleading_base(m: usize, cand_len: usize) -> Result<usize> {
    anyhow::ensure!(
        m > cand_len,
        "mtp verify: got {m} row(s) but the non-leading branch needs at least cand.len()+1 = {} \
         (leading row + {cand_len} draft rows)",
        cand_len + 1
    );
    Ok(m - (cand_len + 1))
}

/// The cycle-1 (and post-reprime) VIRTUAL leading row (`generate_mtp_spec_core`'s doc + `backends.rs`'s
/// "The `+1` leading row" section): the row that re-confirms/predicts a cycle's `cand[0]`, absent from
/// the verify feed because the prime/reprime already cached it, spliced in so the SAME accept/catch-up
/// code handles cycle 1 like every later cycle.
///
/// AUDIT #5: bundling `h` with its prediction into one `Option<Leading>` enforces the present-or-absent
/// lock-step the three former `leading_{h,id,dist}` Options had to maintain by hand — `h` is ALWAYS
/// present when a leading row exists, and EXACTLY one of the greedy `Id` / stochastic `Dist`
/// predictions is (matching the `stochastic` flag), instead of two independently-`Option`al fields a
/// future edit could desync.
struct Leading {
    /// The leading row's target hidden (position `n_past-1`) — row 0 of `catchup_h_all`.
    h: Vec<f32>,
    /// The leading row's prediction, in the flavor the accept rule for this run consumes.
    pred: LeadingPred,
}

/// The leading row's prediction — greedy hands back just the argmax id (GPU-resident verify, issue
/// #31); the stochastic accept rule needs the full truncated proposal distribution instead.
enum LeadingPred {
    Id(u32),
    Dist(Vec<(u32, f32)>),
}

/// Greedy argmax over one `[vocab]` logits row (unlike [`top1_softmax`], no probability needed —
/// `spec_accept`/the verify-round check only reads the winning id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn argmax_row(row: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

/// The backend-generic MTP draft-verify-catchup loop — the shared body of the `_vulkan`/`_metal`
/// drivers. The trunk verify runs through `run_verify(be, bind, …)`; the draft head is the
/// caller-built [`MtpHeadSession`] (bound to the same `be`). Everything between — draft, accept,
/// catch-up, streaming — is backend-agnostic.
///
/// Sampling: reads `Sampler::from_cfg(&ec.sampling)` (`sampling.temp`/`top_k`/`top_p`) like the
/// non-MTP seam decode loop. `temp <= 0` takes the ORIGINAL greedy fast path — `draft` +
/// `crate::seam::model::spec_accept` against the trunk's GPU-resident argmax, bit-identical to
/// before this doc comment was added (pinned by `mtp_spec_matches_target_only_greedy`). `temp > 0`
/// takes the stochastic path — `draft_stochastic` + `crate::seam::model::spec_accept_stochastic`,
/// the standard speculative-sampling acceptance rule (Leviathan & Kalman 2023 / Chen et al. 2023),
/// which provably preserves the TARGET's sampling distribution instead of forcing pure greedy.
/// This is the fix for Qwen3.5/3.6 (thinking models) degenerating under MTP's old greedy-only
/// accept rule — see `docs/mtp.md` / issue #33's follow-up. Perf tradeoff: `temp > 0` can't use
/// either GPU-resident id-only fast path (verify's `Op::Argmax` or draft's fused
/// `Op::ArgmaxProb`) — the ratio/residual test needs each row's FULL truncated distribution, not
/// just an argmax id — so it downloads the full `[vocab]` logits row on every draft step AND every
/// verify row instead of 4 bytes each. Correctness over speed: the greedy path stays the fast one.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn generate_mtp_spec_core(
    be: &dyn Backend,
    bind: &BindWeightFn,
    model: &crate::SeamModel,
    head_sess: &mut MtpHeadSession,
    max_ctx: usize,
    prompt: &str,
    max_new: usize,
    mut on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, MtpTiming)> {
    let cfg = model.config();
    let ec = model.engine_cfg();
    let ne = cfg.n_embd;
    let n_max = DEFAULT_N_MAX;
    let time_mtp = ec.prof.stages;
    // qwen35 DeltaNet spec-decode rollback (default ON; A/B escape via INFR_NO_MTP_CKPT): snapshot
    // the trunk's DeltaNet recurrent state at each clean committed boundary so a partial-accept
    // cycle rolls back to it and re-prefills only the short accepted suffix, instead of qwen35's
    // default full re-prefill-from-zero on the divergent feed (its append-only recurrent state
    // can't rewind by cache truncation — see `SeamKv::mtp_snapshot_delta` and the `c.qwen35` branch
    // in `generate_dense_backend`'s `start` computation). Token-identical either way (a pure decode
    // speedup): restoring an exact state snapshot + re-prefilling the same committed tokens yields
    // the same recurrence as prefilling them from zero.
    let mtp_ckpt = ec.spec.mtp_ckpt;
    // Bounded rollback window (INFR_NO_MTP_REPRIME=1 A/B escape): the snapshot above only advances
    // on FULL-accept cycles, so at low accept rates (alpha≈0.5 on repetitive/dummy prompts —
    // full-accept probability ≈ alpha^n_max ≈ 2%) the boundary stalls and each cycle's verify
    // re-prefills a suffix that GROWS by (accepted+1) per cycle: measured verify m climbing 7→36
    // on the 9B at alpha=0.51, which pushes every projection GEMM off the small-m fast kernels
    // (mrow/BM16 gate at m<=8/m<=16) onto larger tiles at 2-4x the per-row cost. The fix
    // (`reprime_pending` below): when the NEXT verify's row count would leave the fast-tile
    // regime, re-prefill the committed suffix as its own pass right after this cycle commits —
    // the identical rows the next verify would have re-run anyway — snapshot the now-clean
    // boundary, and carry the pass's last row as the next cycle's leading row (the existing
    // cycle-1 mechanism). THRESHOLDED, not every-cycle: small-m GEMMs are weight-bandwidth-bound,
    // so splitting one m=13 verify into a m=7 reprime + m=6 verify streams the whole trunk (and
    // the vocab head) TWICE — measured a 4.6% mtp128 LOSS on the 4B when reprimed on every
    // partial cycle vs +22% on the 9B whose window actually blew out. Repriming only when
    // window + n_max + 1 > MTP_REPRIME_MAX_M keeps typical cycles single-stream and caps the
    // verify at the BM16 tile's m<=16 gate (`DENSE_SMALL_TILE_MAX_M16`).
    let mtp_reprime = mtp_ckpt && ec.spec.mtp_reprime;
    const MTP_REPRIME_MAX_M: usize = 16;
    let ignore_eos = ec.sampling.ignore_eos;
    let hit_eos = |t: u32| !ignore_eos && (cfg.eos_ids.contains(&t) || t == cfg.eos);
    // Sampling config (see this fn's doc): temp<=0 keeps the original bit-identical greedy path;
    // temp>0 (the CLI's chat default, INFR_TEMP=0.6) takes the stochastic accept rule.
    let sampler = crate::sampling::Sampler::from_cfg(&ec.sampling);
    let mut rng = crate::sampling::seed_rng(&ec.sampling);
    let stochastic = sampler.temp > 0.0;

    let prompt_tokens = model.encode(prompt)?;
    anyhow::ensure!(!prompt_tokens.is_empty(), "generate_mtp_spec: empty prompt");
    let p = prompt_tokens.len();

    let mut trunk_state: Option<crate::seam::SeamKv> = None;

    // ── prime: one VERIFY over the whole prompt, then catch the head up over it ──────────────
    let t_prime = std::time::Instant::now();
    // Greedy needs only the prime verify's per-row argmax ids (GPU-resident, `run_verify`); the
    // stochastic path needs the FULL logits (to build cycle 1's leading truncated distribution
    // below) — `run_verify_full`'s doc.
    let (ids0, logits0, h_rows0) = if stochastic {
        let (logits0, h_rows0) = run_verify_full(
            be,
            bind,
            model.gguf(),
            cfg,
            ec,
            model.embd(),
            &prompt_tokens,
            &mut trunk_state,
            max_ctx,
        )?;
        (Vec::new(), logits0, h_rows0)
    } else {
        let (ids0, h_rows0) = run_verify(
            be,
            bind,
            model.gguf(),
            cfg,
            ec,
            model.embd(),
            &prompt_tokens,
            &mut trunk_state,
            max_ctx,
        )?;
        (ids0, Vec::new(), h_rows0)
    };
    let prime_verify_secs = t_prime.elapsed().as_secs_f64();
    anyhow::ensure!(
        h_rows0.len() == p * ne,
        "prime verify: h_rows0 length {} != {p}*{ne}",
        h_rows0.len()
    );

    let mut shifted_h = vec![0f32; p * ne];
    if p > 1 {
        shifted_h[ne..].copy_from_slice(&h_rows0[..(p - 1) * ne]);
    }
    catch_up(head_sess, &prompt_tokens, &shifted_h, 0)?;

    // First clean committed boundary: the trunk's DeltaNet state now covers the whole prompt (the
    // prime verify left `cached == prompt_tokens`). Snapshot it so cycle 1's partial-accept case
    // (if any) can roll back here instead of re-prefilling from zero.
    if mtp_ckpt {
        if let Some(st) = trunk_state.as_mut() {
            st.mtp_snapshot_delta(be, cfg)?;
        }
    }
    // Committed-token length of the last snapshot — the rollback window `committed.len() -
    // boundary` is what the next verify re-prefills after a partial accept (`mtp_reprime`'s doc).
    let mut boundary = p;

    let mut committed = prompt_tokens.clone();
    let mut id_last = prompt_tokens[p - 1];
    let mut pending_h = h_rows0[(p - 1) * ne..].to_vec();
    let mut n_past = p;

    // Cycle 1's virtual leading row (see this fn's doc + `Leading`'s doc) — `None` from cycle 2 on,
    // once consumed. Greedy carries the row's ARGMAX ID (the GPU-resident verify accept, issue #31,
    // hands back just the ids); the stochastic path carries the row's full truncated distribution
    // instead, built from the prime verify's full logits row. AUDIT #5: one `Option<Leading>` keeps
    // `h` and the prediction present-or-absent in lock-step.
    let mut leading: Option<Leading> = Some(Leading {
        h: pending_h.clone(),
        pred: if stochastic {
            LeadingPred::Dist(crate::sampling::truncated_dist(
                &logits0[(p - 1) * cfg.vocab..p * cfg.vocab],
                sampler,
            ))
        } else {
            LeadingPred::Id(ids0[p - 1])
        },
    });

    let mut acc: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let mut out: Vec<u32> = Vec::new();
    let mut cycle = 0usize;
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;
    // Phase 4 (issue #33): the SAME per-cycle sections `INFR_PROF_STAGES`'s `eprintln!` below already
    // times, summed across the whole run — this is what `MtpTiming`'s return threads out.
    let mut sum_draft_secs = 0.0f64;
    let mut sum_verify_secs = 0.0f64;
    let mut sum_catchup_secs = 0.0f64;
    let t_decode = std::time::Instant::now();

    'cycles: while out.len() < max_new {
        let n_max_round = n_max.min(max_new - out.len());
        if n_max_round == 0 {
            break;
        }
        cycle += 1;

        let t_draft = std::time::Instant::now();
        // cand: the drafted token ids, either flavor. q_dists: the stochastic flavor's per-step
        // truncated proposal distributions (empty/unused on the greedy path).
        let (cand, q_dists): (Vec<u32>, Vec<Vec<(u32, f32)>>) = if stochastic {
            let drafted = draft_stochastic(
                head_sess,
                id_last,
                &pending_h,
                n_past,
                n_max_round,
                sampler,
                &mut rng,
            )?;
            let cand = drafted.iter().map(|(id, _)| *id).collect();
            let q_dists = drafted.into_iter().map(|(_, q)| q).collect();
            (cand, q_dists)
        } else if head_sess.can_draft_chain() && ec.spec.mtp_draft_chain {
            // Fused single-submission chain (issue #33 follow-up — see `draft_chain`'s doc for the
            // correctness argument): replaces `n_max_round` sequential submit→wait→readback
            // round-trips with ONE compile+execute+download. `INFR_NO_MTP_DRAFT_CHAIN=1` A/B
            // escape keeps the old per-step `draft()` path for comparison.
            let cand = head_sess.draft_chain(id_last, &pending_h, n_past, n_max_round)?;
            (cand, Vec::new())
        } else {
            let drafted = draft(
                head_sess,
                id_last,
                &pending_h,
                n_past,
                DEFAULT_P_MIN,
                n_max_round,
            )?;
            (drafted.iter().map(|&(id, _)| id).collect(), Vec::new())
        };
        let draft_secs = t_draft.elapsed().as_secs_f64();

        let mut feed = committed.clone();
        feed.extend_from_slice(&cand);
        let t_verify = std::time::Instant::now();
        // vids: greedy flavor's per-row argmax ids. vlogits: stochastic flavor's full [m*vocab]
        // logits (needed to build each row's truncated target distribution below).
        let (vids, vlogits, h_rows) = if stochastic {
            let (vlogits, h_rows) = run_verify_full(
                be,
                bind,
                model.gguf(),
                cfg,
                ec,
                model.embd(),
                &feed,
                &mut trunk_state,
                max_ctx,
            )?;
            (Vec::new(), vlogits, h_rows)
        } else {
            let (vids, h_rows) = run_verify(
                be,
                bind,
                model.gguf(),
                cfg,
                ec,
                model.embd(),
                &feed,
                &mut trunk_state,
                max_ctx,
            )?;
            (vids, Vec::new(), h_rows)
        };
        let verify_secs = t_verify.elapsed().as_secs_f64();
        let m = h_rows.len() / ne;

        // Consume the leading row ONCE (AUDIT #5: `Leading` bundles `h` + prediction in lock-step).
        // The accept rule below reads the prediction; the catch-up block reads `h` — split the taken
        // value so each gets its half and the state auto-clears for cycle 2+ (`None` until reprime).
        let (leading_h, leading_pred) = match leading.take() {
            Some(Leading { h, pred }) => (Some(h), Some(pred)),
            None => (None, None),
        };
        let (accepted, next_tok) = if stochastic {
            let vocab = cfg.vocab;
            let p_rows: Vec<Vec<(u32, f32)>> = if let Some(LeadingPred::Dist(ld)) = leading_pred {
                debug_assert_eq!(m, cand.len(), "cycle 1 verify carries no leading row");
                let mut v = Vec::with_capacity(m + 1);
                v.push(ld);
                v.extend((0..m).map(|j| {
                    crate::sampling::truncated_dist(&vlogits[j * vocab..(j + 1) * vocab], sampler)
                }));
                v
            } else {
                let base = nonleading_base(m, cand.len())?;
                (base..m)
                    .map(|j| {
                        crate::sampling::truncated_dist(
                            &vlogits[j * vocab..(j + 1) * vocab],
                            sampler,
                        )
                    })
                    .collect()
            };
            crate::seam::model::spec_accept_stochastic(&cand, &q_dists, &p_rows, &mut rng)
        } else {
            let varg: Vec<u32> = if let Some(LeadingPred::Id(lid)) = leading_pred {
                debug_assert_eq!(m, cand.len(), "cycle 1 verify carries no leading row");
                std::iter::once(lid).chain(vids.iter().copied()).collect()
            } else {
                let base = nonleading_base(m, cand.len())?;
                vids[base..].to_vec()
            };
            crate::seam::model::spec_accept(&cand, &varg)
        };
        total_drafted += cand.len();
        total_accepted += accepted;

        // DeltaNet rollback bookkeeping (see `mtp_ckpt`'s doc). A fully-accepted cycle leaves the
        // trunk's recurrent state on a clean committed prefix (`cached == committed ++ cand`, every
        // drafted row committed) — snapshot it as the new rollback point. A partial/zero-accept
        // cycle left the state polluted by the rejected drafts — restore the last clean snapshot so
        // the NEXT verify's prefix-diff re-prefills only the short accepted suffix from there rather
        // than triggering qwen35's full state reset. `h_rows`/`varg` were already read above, so the
        // rollback only affects the trunk state the next cycle consumes.
        let mut restored = false;
        if mtp_ckpt {
            if let Some(st) = trunk_state.as_mut() {
                if accepted == cand.len() {
                    st.mtp_snapshot_delta(be, cfg)?;
                    // The state covers `feed` = committed ++ cand (next_tok never went through
                    // the trunk this cycle) — that's the snapshot's committed length.
                    boundary = feed.len();
                } else {
                    st.mtp_restore_delta(be)?;
                    // A clean boundary may need re-establishing at the END of this cycle (after
                    // the accepted tokens + next_tok are committed) — see `mtp_reprime`'s doc.
                    restored = true;
                }
            }
        }

        // The (accepted+1) rows of `h` to catch the head's KV up to — row 0 is the virtual/real
        // leading row, rows 1.. are this cycle's own verify output (see this fn's doc).
        let catchup_h_all: Vec<f32> = if let Some(lh) = leading_h {
            let mut v = lh;
            v.extend_from_slice(&h_rows);
            v
        } else {
            let base = nonleading_base(m, cand.len())?;
            h_rows[base * ne..].to_vec()
        };
        let mut catchup_tokens: Vec<u32> = cand[..accepted].to_vec();
        catchup_tokens.push(next_tok);
        let catchup_h = &catchup_h_all[..(accepted + 1) * ne];

        let t_catchup = std::time::Instant::now();
        // AUDIT #1: catch up at `n_past`, NOT `n_past + 1`. The draft appended its rows at absolute
        // positions `n_past + s` (`draft`/`forward_draft`, matching `speculative.cpp`'s
        // `dp.n_past + i`), so the committed tokens `[cand[..accepted] | next_tok]` occupy positions
        // `n_past .. n_past + accepted` and MUST overwrite the head's KV at exactly those rows —
        // `start_pos = n_past`. `catchup_h[0]` (= the leading row's h, target hidden at position
        // `n_past-1`) pairs with token position `n_past` under the head's `(t_pos, h_{pos-1})`
        // convention (see prime's `catch_up(..., 0)` with `shifted_h[i] = h_{i-1}`). The old `+1`
        // stored every (token, h) pair one row too high (wrong RoPE position + a stale h) and left
        // the draft's own stale row at `n_past` un-rewritten, corrupting the KV the NEXT cycle's
        // attention reads (an acceptance-rate bug; token-identity is VERIFY-guarded either way).
        catch_up(head_sess, &catchup_tokens, catchup_h, n_past)?;
        let catchup_secs = t_catchup.elapsed().as_secs_f64();
        // AUDIT #3: `pending_h` here is `h_{n_past+accepted-1}` — the target hidden that PREDICTED
        // `next_tok`, i.e. one step behind `next_tok`'s OWN hidden `h_{n_past+accepted}` that the
        // convention wants for the next draft's row-0 (position `n_past_new-1`). `next_tok` is
        // deferred through the trunk (it "never went through the trunk this cycle"), so its own
        // hidden is NOT computed here — the row at position `n_past+accepted` was fed the REJECTED
        // draft token, not `next_tok`. This one-step-stale handoff is the best available on the
        // common path; the reprime branch below, which DOES re-decode `next_tok`, corrects it.
        pending_h = catchup_h[accepted * ne..].to_vec();
        sum_draft_secs += draft_secs;
        sum_verify_secs += verify_secs;
        sum_catchup_secs += catchup_secs;

        let (build_secs, exec_secs) = head_sess.take_timing();
        if time_mtp {
            tracing::info!(
                "[mtp cycle {cycle}] drafted={} accepted={accepted} \
                 draft={:.1}ms verify={:.1}ms catchup={:.1}ms build={:.1}ms exec={:.1}ms",
                cand.len(),
                draft_secs * 1e3,
                verify_secs * 1e3,
                catchup_secs * 1e3,
                build_secs * 1e3,
                exec_secs * 1e3,
            );
        }

        let mut stop = false;
        for &c in &cand[..accepted] {
            let eos_hit = hit_eos(c);
            out.push(c);
            committed.push(c);
            if !eos_hit {
                crate::stream_token(model.tokenizer(), &mut acc, &mut printed, c, &mut on_piece);
            }
            if eos_hit || out.len() >= max_new {
                stop = true;
                break;
            }
        }
        if !stop {
            let eos_hit = hit_eos(next_tok);
            out.push(next_tok);
            committed.push(next_tok);
            if !eos_hit {
                crate::stream_token(
                    model.tokenizer(),
                    &mut acc,
                    &mut printed,
                    next_tok,
                    &mut on_piece,
                );
            }
            if eos_hit || out.len() >= max_new {
                stop = true;
            }
        }
        if stop {
            break 'cycles;
        }

        id_last = next_tok;
        n_past = committed.len();

        // Partial-accept boundary re-prime (`mtp_reprime`'s doc): when the rollback window has
        // grown enough that the NEXT verify would leave the small-m fast-tile regime, prefill the
        // committed suffix now (prefix-diff from the just-restored snapshot — the SAME rows the
        // next verify would otherwise re-run inside its own pass), snapshot the clean state, and
        // hand the last row to the next cycle as its leading row exactly like the prompt prime
        // does for cycle 1. Timed into the catchup bucket: it's committed-boundary maintenance,
        // not verify work. Uses [`run_prime_last`], NOT [`run_verify`]/[`run_verify_full`]: only
        // the LAST row's id/logits/h is ever read below (it becomes the next cycle's [`Leading`] +
        // `pending_h` — AUDIT #3), so a full VERIFY's per-row lm_head here would compute up to
        // `MTP_REPRIME_MAX_M`-1 rows
        // of vocab-wide GEMM that are immediately thrown away — see `run_prime_last`'s doc.
        if restored && mtp_reprime && (committed.len() - boundary) + n_max > MTP_REPRIME_MAX_M {
            let t_reprime = std::time::Instant::now();
            let (rid, rlogits, rh) = run_prime_last(
                be,
                bind,
                model.gguf(),
                cfg,
                ec,
                model.embd(),
                &committed,
                &mut trunk_state,
                max_ctx,
                stochastic,
            )?;
            let pred = if stochastic {
                anyhow::ensure!(
                    rlogits.len() == cfg.vocab,
                    "mtp reprime: expected {} logits, got {}",
                    cfg.vocab,
                    rlogits.len()
                );
                LeadingPred::Dist(crate::sampling::truncated_dist(&rlogits, sampler))
            } else {
                LeadingPred::Id(rid)
            };
            // AUDIT #3: the reprime pass re-decoded `next_tok` (committed's last row), so `rh` is
            // `next_tok`'s OWN hidden `h_{n_past_new-1}` — exactly the row-0 `h` the next draft wants
            // (position `n_past_new`, convention `(t_pos, h_{pos-1})`). Refresh `pending_h` with it,
            // replacing the one-step-stale predicting-hidden the common-path handoff left above. (On
            // the non-reprime path `next_tok`'s own hidden isn't computed until the next verify, so
            // the stale handoff stands there — see the `pending_h` comment after `catch_up`.)
            pending_h = rh.clone();
            leading = Some(Leading { h: rh, pred });
            if let Some(st) = trunk_state.as_mut() {
                st.mtp_snapshot_delta(be, cfg)?;
            }
            boundary = committed.len();
            sum_catchup_secs += t_reprime.elapsed().as_secs_f64();
        }
    }

    let timing = MtpTiming {
        draft_secs: sum_draft_secs,
        verify_secs: sum_verify_secs,
        catchup_secs: sum_catchup_secs,
        total_drafted,
        total_accepted,
    };
    if time_mtp {
        let (dp, vp, cp) = timing.phase_shares();
        tracing::info!(
            "[mtp summary] {cycle} cycles, {total_accepted}/{total_drafted} accepted (alpha={:.3}), {} tokens generated, phase share: draft {dp:.0}% verify {vp:.0}% catchup {cp:.0}%",
            timing.alpha(),
            out.len()
        );
    }

    Ok((
        crate::GenStats {
            n_prompt: p,
            n_cached: 0,
            prompt_secs: prime_verify_secs,
            n_gen: out.len(),
            decode_secs: t_decode.elapsed().as_secs_f64(),
        },
        timing,
    ))
}

#[cfg(test)]
mod tests {
    use super::nonleading_base;

    /// AUDIT #4: the non-leading `base = m - (cand.len()+1)` subtraction must return an `Err` (not
    /// wrap `usize` and panic on the subsequent slice) when the trunk hands back too few rows.
    #[test]
    fn nonleading_base_guards_underflow() {
        // Normal cycle-2+ shape: m = cand.len() + 1 (leading row + the drafted rows).
        assert_eq!(nonleading_base(7, 6).unwrap(), 0);
        // Re-prefilled window: extra committed rows precede the leading row.
        assert_eq!(nonleading_base(13, 6).unwrap(), 6);
        assert_eq!(nonleading_base(1, 0).unwrap(), 0);
        // Underflow edge (m < cand.len()+1) -> Err, NOT a wrapped huge index / slice panic.
        assert!(nonleading_base(6, 6).is_err());
        assert!(nonleading_base(0, 0).is_err());
        assert!(nonleading_base(3, 6).is_err());
    }
}
