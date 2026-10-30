//! Per-layer weight handles + the persistent seam session state ([`SeamKv`]/[`SeamWeights`]).
//! Pure-move split of `seam.rs` — see `super` for the module overview.
use super::common_prefix_len;
use super::sc::{DenoiseCache, SelfCondWeights};
use super::segmented_kv::{PlaneKind, SegmentedKvLayout, KV_GROW_ROWS};
use crate::Config;
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::tensor::{DType, TensorId};

/// FFN weight handles: a dense gated FFN, a qwen3moe routed-expert bank (router + stacked
/// per-expert gate/up/down), or diffusion-gemma's dual FFN (dense ∥ MoE, summed).
pub(super) enum FfnW {
    Dense {
        wgate: TensorId,
        wup: TensorId,
        wdown: TensorId,
    },
    /// Combined gate+up weight `[2*nff, ne]` (one GEMV/GEMM + `GatedActFused`); see `fuse_gu`.
    DenseFused { wgu: TensorId, wdown: TensorId },
    Moe {
        router: TensorId,
        gate_exps: TensorId,
        up_exps: TensorId,
        down_exps: TensorId,
        /// `true` when `gate_exps` is the fused `[2*n_ff_exp, n_embd]` bank and `up_exps` aliases
        /// it. Qwen3.8 uses the same layout already supported by the pager's MoE op.
        fused_gate_up: bool,
        /// Shared expert (qwen35moe / llama4): `Some` when `Config::shexp_ff > 0` — a dense SwiGLU
        /// branch run on the SAME input as the routed bank and summed with its output. qwen35moe
        /// gates it by a per-token sigmoid (`Op::MoeSharedExpertAdd`, `gate_inp = Some`); llama4
        /// sums it in PLAIN (`Op::Add`, `gate_inp = None` — `Config::shexp_gated == false`). `None`
        /// for qwen3moe (no shared expert).
        shexp: Option<MoeSharedW>,
        /// DeepSeek V2+: per-layer router bias `[n_expert]` added to logits for selection only
        /// (the unbias'd probs are still used for routing weights). `None` = no bias.
        exp_probs_b: Option<TensorId>,
        /// DeepSeek V4 HASH-routed layer: `ffn_gate_tid2eid.weight`, the I32
        /// `[n_expert_used, n_vocab]` token-id → expert-id table. `Op::GatherI32` turns it plus the
        /// graph's token-id Input into the `[batch, n_expert_used]` selection
        /// `Op::MoeFfn::expert_ids` consumes.
        ///
        /// Mutually exclusive with `exp_probs_b` — `deepseek4.cpp` creates exactly one of them per
        /// layer, and they occupy the SAME slot of the upload order (see `wload`'s `c.deepseek4`
        /// arm and the matching `wpush`). `None` on every other arch and on a bias-routed V4 layer.
        tid2eid: Option<TensorId>,
    },
    /// diffusion-gemma's per-layer dual FFN: a dense GeGLU branch (the "shared expert") ∥ a
    /// 128-expert MoE branch (fused `gate_up_exps` + per-expert `down_exps` scale), summed and
    /// sandwich-normed. See the FFN wiring in `docs/diffusion-gemma.md`. `LayerW::ffn_norm` is the
    /// dense branch's INPUT norm and `LayerW::post_ffw` the shared FINAL norm (both reused as-is —
    /// every gemma model already carries them); the fields below are the pieces unique to the
    /// dual-FFN block.
    DiffusionMoe {
        d_gate: TensorId,
        /// Equal to `d_gate` (same handle, never separately read) when `fused_gu` — the concat
        /// mirrors `DenseFused`'s `wgu`, just kept on the `DiffusionMoe` shape since this branch's
        /// down-projection/router/expert fields don't otherwise fit `FfnW::DenseFused`.
        d_up: TensorId,
        /// `d_gate`/`d_up` are ONE concatenated `[2*nff, ne]` weight (see `fuse_gu` in `runner.rs`);
        /// the dense branch issues one wide `Op::Linear` + `Op::GatedActFused` instead of two
        /// `Op::Linear` + `Op::GatedAct` — out_f=2112 clears neither warp-tile gate (`%256`/`%128`)
        /// on its own so it fell to the slower `mmq` path; fused out_f=4224 clears `%128`.
        fused_gu: bool,
        d_down: TensorId,
        /// `post_ffw_norm_1`: dense branch output norm (before summing with the MoE branch).
        d_post_norm: TensorId,
        /// `pre_ffw_norm_2`: MoE branch's own input norm, applied to `attn_out` (the UNNORMED
        /// post-attention residual — a separate parallel read from the dense branch's `ffn_norm`).
        m_pre_norm: TensorId,
        /// `ffn_gate_inp.weight`: router logits projection.
        router: TensorId,
        /// `ffn_gate_inp.scale` `[ne]`: elementwise scale on the router's OWN input (the weightless
        /// rmsnorm of `attn_out`, further scaled by `1/√ne` — see the graph-build wiring).
        router_scale: TensorId,
        /// `ffn_gate_up_exps.weight`, fused `[ne, 2*n_ff_exp, n_expert]`.
        gate_up_exps: TensorId,
        down_exps: TensorId,
        /// `ffn_down_exps.scale` `[n_expert]`: per-expert scale on the down-projection output.
        down_scale: TensorId,
        /// `post_ffw_norm_2`: MoE branch output norm (before summing with the dense branch).
        m_post_norm: TensorId,
    },
}

impl FfnW {
    pub(super) fn has_shared_expert(&self) -> bool {
        matches!(self, Self::Moe { shexp: Some(_), .. })
    }
}

/// qwen35moe (Qwen3.6 MoE) Qwen2-MoE-style shared-expert weights (see `FfnW::Moe`'s `shexp`
/// field): a dense SwiGLU FFN run on the same input as the routed bank, gated by a per-token
/// sigmoid on `gate_inp`'s (scalar) output. `Copy` (all `TensorId` fields) so `FfnW::Moe` stays
/// matchable-by-value through a `&LayerW`, exactly like every other all-`TensorId` `FfnW` variant.
#[derive(Clone, Copy)]
pub(super) struct MoeSharedW {
    /// `ffn_gate_inp_shexp.weight` `[ne]`: projects the FFN input to ONE raw (pre-sigmoid) gate
    /// logit per token (`Op::Linear` with `out_f=1`). `Some` for the sigmoid-gated qwen35moe
    /// shared expert; `None` for llama4 (its shared expert is summed in plain — no gate tensor).
    pub(super) gate_inp: Option<TensorId>,
    pub(super) wgate: TensorId,
    pub(super) wup: TensorId,
    pub(super) wdown: TensorId,
}

/// Attention-mixer weights (the classic transformer token mixer: QKV projections + output;
/// q/k-norm optional, `wv` absent on gemma4 full-attention layers which reuse the raw K
/// projection as V). A future phase adds a DeltaNet variant (qwen35's linear-attention mixer),
/// so everything attention-specific lives here and everything layer-generic (norms, FFN,
/// per-layer embeddings) stays on [`LayerW`].
pub(super) struct AttnW {
    pub(super) wq: TensorId,
    pub(super) wk: TensorId,
    pub(super) wv: Option<TensorId>,
    // Qwen2/2.5 q/k/v projection biases (`Config::qkv_bias`); `None` on every bias-free arch.
    pub(super) qb: Option<TensorId>,
    pub(super) kb: Option<TensorId>,
    pub(super) vb: Option<TensorId>,
    pub(super) q_norm: Option<TensorId>,
    pub(super) k_norm: Option<TensorId>,
    pub(super) wo: TensorId,
    /// Qwen3.8 QSA block-indexer weights. Present only on its full-attention layers.
    pub(super) qsa: Option<QsaW>,
}

#[derive(Clone, Copy)]
pub(super) struct QsaW {
    pub(super) k_norm: TensorId,
    pub(super) k_proj: TensorId,
    pub(super) q_norm: TensorId,
    pub(super) q_proj: TensorId,
    pub(super) fused_prepare: bool,
}

/// qwen35 gated-DeltaNet linear-attention mixer weights (see `docs/qwen35.md`). Unlike `AttnW` this
/// mixer owns no KV cache — its recurrent state (a rolling conv history + the DeltaNet `S` matrix)
/// is session state, held in the SAME `kbufs`/`vbufs` slots a KV-caching layer would use (see
/// `SeamKv` and the state-buffer alloc in `generate_dense_backend`).
pub(super) struct DeltaW {
    pub(super) qkv: TensorId,
    pub(super) gate: TensorId,
    pub(super) conv1d: TensorId,
    pub(super) alpha: TensorId,
    pub(super) beta: TensorId,
    pub(super) ssm_a: TensorId,
    pub(super) dt_bias: TensorId,
    pub(super) ssm_norm: TensorId,
    pub(super) out: TensorId,
}

/// Ling KDA Q/K/V projections. Most GGUFs use one dtype for all three and keep the single fused
/// projection. Ling 3.0 Tiny Q4_K_M stores Q/K as Q4_K but V as Q6_K, so that model keeps a fused
/// Q/K bank plus a separate V bank and packs their activations before the causal convolution.
#[derive(Clone, Copy)]
pub(super) enum KdaQkvW {
    Fused(TensorId),
    QkV { qk: TensorId, v: TensorId },
}

/// Ling KDA mixer weights. The causal-conv banks are concatenated once at load, matching the
/// packed activation consumed by [`infr_core::Op::Kda`].
pub(super) struct KdaW {
    pub(super) qkv: KdaQkvW,
    pub(super) conv: TensorId,
    pub(super) forget: TensorId,
    /// glm5next's SECOND-stage forget projection `[rank, n_head * kda_head_dim]` — the forget gate
    /// is two-stage low-rank (`ssm_f_a` then `ssm_f_b`), where Ling carries one stage. `Some`
    /// exactly when `Config::glm5next`; pushed right after `forget` to mirror the upload order.
    pub(super) forget_b: Option<TensorId>,
    pub(super) beta: TensorId,
    pub(super) a: TensorId,
    pub(super) dt_bias: TensorId,
    pub(super) norm: TensorId,
    pub(super) gate: TensorId,
    /// glm5next's SECOND-stage gate projection, the `ssm_g_b` twin of [`Self::forget_b`].
    pub(super) gate_b: Option<TensorId>,
    pub(super) out: TensorId,
}

/// DeepSeek V2+ MLA (Multi-head Latent Attention) mixer weights — absorbed form. The KV cache holds
/// ONE compressed row per token (`key_length = kv_lora_rank + qk_rope_dim`); V is an aliased prefix —
/// no separate V cache. See `docs/deepseek.md` § Stage 2.
///
/// `deepseek32` (V3.2) uses this same mixer plus the five per-layer lightning-indexer tensors on
/// [`IndexerW`] — see `docs/deepseek.md` § Stage 3.
pub(super) struct MlaW {
    /// Q low-rank input projection `[n_embd, q_lora_rank]` (absent in lite models).
    pub(super) wq_a: Option<TensorId>,
    /// RMSNorm on the q_lora_rank-dimensional intermediate, between wq_a and wq_b.
    pub(super) q_a_norm: Option<TensorId>,
    /// Q low-rank output `[q_lora_rank, n_head * head_k_mla]` (`wq` for lite: `[n_embd, ...]`).
    pub(super) wq_b: TensorId,
    /// Combined KV compression + rope projection `[n_embd, kv_lora_rank + qk_rope_dim]`.
    pub(super) wkv_a_mqa: TensorId,
    /// RMSNorm on the KV latent (`kv_lora_rank`-wide) AFTER the split from k_pe.
    pub(super) kv_a_norm: TensorId,
    /// Absorption weight, per-head `[qk_nope_dim, kv_lora_rank]` as stored in the GGUF (attn_k_b) — wk_b[h]ᵀ maps q_nope to latent.
    pub(super) wk_b: TensorId,
    /// Output weight, per-head `[kv_lora_rank, v_head_dim]` as stored in the GGUF (attn_v_b) — applied AFTER the KQV product.
    pub(super) wv_b: TensorId,
    /// Output projection `[n_head * v_head_dim, n_embd]`.
    pub(super) wo: TensorId,
    /// Ling's one-scalar-per-head sigmoid output gate; absent on DeepSeek MLA.
    pub(super) gate: Option<TensorId>,
    /// deepseek32's lightning indexer. `Some` exactly when `Config::deepseek32`; `None` for every
    /// `deepseek2` model, which attends every causally-eligible key. The emit arm reads this
    /// `Option` rather than the config flag, so a `deepseek32` model whose indexer weights were not
    /// captured cannot fall through to the deepseek2 graph — it fails.
    pub(super) indexer: Option<IndexerW>,
}

/// DeepSeek V3.2's per-layer lightning-indexer weights — the five tensors V3.2 adds to
/// `deepseek2`'s absorbed MLA, on EVERY layer (dense-lead included: `deepseek32.cpp` creates them
/// outside the dense/MoE branch). See `docs/deepseek.md` § "The lightning indexer".
pub(super) struct IndexerW {
    /// `indexer.k_norm.weight` `[indexer_head_size]` — the gain of a mean-centred LayerNorm
    /// (`Op::LayerNorm`), the only non-RMS norm anywhere in the DeepSeek family.
    pub(super) k_norm: TensorId,
    /// `indexer.k_norm.bias` `[indexer_head_size]` — that LayerNorm's bias. Not optional: the
    /// reference always loads it, and an RMSNorm-shaped port would silently drop it.
    pub(super) k_norm_b: TensorId,
    /// `indexer.proj.weight` `[n_embd, indexer_n_head]` — projects the layer input to ONE weight
    /// per indexer head per token (`w` in the score formula), applied to the attn-normed input.
    pub(super) proj: TensorId,
    /// `indexer.attn_k.weight` `[n_embd, indexer_head_size]` — the single shared indexer KEY head
    /// (MQA: one key row per token, dotted against every indexer query head).
    pub(super) attn_k: TensorId,
    /// `indexer.attn_q_b.weight` `[q_lora_rank, indexer_n_head * indexer_head_size]` — the indexer
    /// queries, read off the SAME normed low-rank `qr` that `wq_b` consumes.
    pub(super) attn_q_b: TensorId,
    /// glm5next's key-pooling compressor APE `[kpool, key_length]` — the learned per-slot position
    /// embedding of the kpool-4 pool (`glm5next.cpp`'s `indexer_compressor_ape`). `Some` exactly
    /// when `Config::glm5next`; V3.2 scores every token directly and has no pool.
    pub(super) compressor_ape: Option<TensorId>,
    /// glm5next's key-pooling gate `[n_embd, key_length]` — projects the token input to pool the
    /// kpool consecutive key rows with (`indexer_compressor_gate`). Same `glm5next`-only rule.
    pub(super) compressor_gate: Option<TensorId>,
}

/// DeepSeek V4's attention-mixer weights shared by all three compression tiers.
/// See `docs/deepseek.md` § Stage 4.
///
/// V4 is NOT MLA: there is no `kv_lora_rank`, no `wk_b`/`wv_b`, and one MQA KV head serves every
/// query head. What it keeps from deepseek2 is the Q-LoRA triple; everything after it is its own.
///
/// Ratio-4 and ratio-128 layers additionally carry [`Dsv4CompressedW`]; ratio-0 leaves that field
/// empty and uses only the sliding-window cache.
pub(super) struct Dsv4W {
    /// `attn_sinks.weight` `[n_head]` — one learned logit per query head, joining the softmax MAX
    /// and DENOMINATOR and never the numerator (`Op::Attention::sinks`).
    pub(super) sinks: TensorId,
    /// Q LoRA: `attn_q_a` `[n_embd, q_lora_rank]` → `attn_q_a_norm` → `attn_q_b`
    /// `[q_lora_rank, n_head * head_dim]`. V4 has no "lite" variant — `q_lora_rank` is mandatory.
    pub(super) wq_a: TensorId,
    pub(super) q_a_norm: TensorId,
    pub(super) wq_b: TensorId,
    /// `attn_kv.weight` `[n_embd, head_dim]` — the SINGLE MQA key/value head shared by every query
    /// head. Not a per-head bank, and not deepseek2's `wkv_a_mqa` (there is no latent to compress).
    pub(super) wkv: TensorId,
    /// `attn_kv_a_norm.weight` `[head_dim]` — RMSNorm over the whole KV row. Shares its on-disk
    /// name with deepseek2's `attn_kv_a_norm` (`docs/deepseek.md` open question 8).
    pub(super) wkv_norm: TensorId,
    /// Grouped low-rank output projection, replacing `attn_output`:
    /// `attn_output_a` `[n_head*head_dim / o_group_count, o_lora_rank * o_group_count]` read as
    /// `{hd_g, o_lora_rank, o_group_count}` — group `g` is rows `[g*o_lora_rank, (g+1)*o_lora_rank)`
    /// (selected by `Op::Linear::w_off`) over input columns `[g*hd_g, (g+1)*hd_g)` — then
    /// `attn_output_b` `[o_group_count*o_lora_rank, n_embd]`.
    pub(super) wo_a: TensorId,
    pub(super) wo_b: TensorId,
}

/// One DeepSeek V4 softmax-pooling compressor. Ratio-4 layers own two of these (attention CSA and
/// LID); ratio-128 layers own only the attention HCA compressor.
pub(super) struct Dsv4CompressorW {
    pub(super) wkv: TensorId,
    pub(super) wgate: TensorId,
    pub(super) ape: TensorId,
    pub(super) norm: TensorId,
}

/// Ratio-4-only lightning indexer weights and its independent compressor.
pub(super) struct Dsv4IndexerW {
    pub(super) proj: TensorId,
    pub(super) q_b: TensorId,
    pub(super) compressor: Dsv4CompressorW,
}

pub(super) struct Dsv4CompressedW {
    pub(super) attention: Dsv4CompressorW,
    pub(super) indexer: Option<Dsv4IndexerW>,
}

/// One hyper-connection block's `(fn, base, scale)` triple. `w_fn` is the mixing matmul's weight
/// (`[hc_mult*n_embd, (2+hc_mult)*hc_mult]` wrapping a sublayer, `[hc_mult*n_embd, hc_mult]` at the
/// model head); `base`/`scale` are the per-chunk affine offsets/slopes `Op::HyperConnectMix` reads.
#[derive(Clone, Copy)]
pub(super) struct HcTriple {
    pub(super) w_fn: TensorId,
    pub(super) base: TensorId,
    pub(super) scale: TensorId,
}

/// The two per-layer hyper-connection triples: one wrapping the attention sublayer, one the FFN.
/// Both are unconditional on every V4 layer (`hc_attn_*` / `hc_ffn_*`) — and on every glm5next
/// trunk layer, which reuses the same tensor names and the (fn, base, scale) triple shape.
pub(super) struct LayerHcW {
    pub(super) attn: HcTriple,
    pub(super) ffn: HcTriple,
}

/// Qwen3.8's low-rank gated residual module. Unlike DeepSeek V4 HC, this is grouped RMSNorm,
/// down/SiLU/up elementwise gates, and an optional per-stream injection projection.
#[derive(Clone, Copy)]
pub(super) struct QwenHcW {
    pub(super) norm: TensorId,
    pub(super) down: TensorId,
    pub(super) up: TensorId,
    pub(super) inject: Option<TensorId>,
}

pub(super) struct QwenLayerHcW {
    pub(super) attn: QwenHcW,
    pub(super) ffn: QwenHcW,
}

pub(super) struct QwenPleW {
    pub(super) key: TensorId,
    pub(super) value: TensorId,
    pub(super) norm_key: TensorId,
    pub(super) norm_query: TensorId,
    pub(super) norm_conv: TensorId,
    /// Dilation-expanded depthwise kernel (`ple_conv_kernel=4`, ngram dilation 3 -> kernel 10).
    pub(super) conv: TensorId,
}

/// The layer's token mixer: classic attention, qwen35 gated-DeltaNet, DeepSeek MLA, or DeepSeek V4.
pub(super) enum MixerW {
    Attn(AttnW),
    DeltaNet(DeltaW),
    Kda(KdaW),
    Mla(MlaW),
    Dsv4(Dsv4W),
}

/// Per-layer weight handles captured while building one decode graph (sandwich norms optional).
/// The order they're declared in MUST match the upload order so `weights[i]` binds to `wbufs[i]`.
pub(super) struct LayerW {
    pub(super) attn_norm: TensorId, // the mixer INPUT norm (applies to any mixer type)
    pub(super) mixer: MixerW,
    /// DeepSeek V4's two per-layer hyper-connection triples. `Some` exactly when the mixer is
    /// [`MixerW::Dsv4`]; `None` for every other arch. They are NOT mixer weights — one of them
    /// wraps the FFN sublayer — which is why they sit on `LayerW` beside the norms.
    pub(super) hc: Option<LayerHcW>,
    pub(super) qwen_hc: Option<QwenLayerHcW>,
    pub(super) ple: Option<QwenPleW>,
    /// V4 ratio-4/128 compressor handles, declared after the layer's HC weights to mirror GGUF
    /// upload order. `None` for ratio-0 and every non-V4 layer.
    pub(super) dsv4_compressed: Option<Dsv4CompressedW>,
    /// bitnet (BitNet b1.58) SubLN: RMSNorm on the concatenated-heads attention output BEFORE the
    /// o-projection (`AttnW::wo`). `Some` only when `Config::sub_norm` (bitnet); `None` elsewhere.
    pub(super) attn_sub_norm: Option<TensorId>,
    pub(super) post_attn: Option<TensorId>,
    pub(super) ffn_norm: TensorId,
    pub(super) ffn: FfnW,
    /// bitnet SubLN: RMSNorm on the FFN intermediate (`[n_ff]`) BEFORE `ffn_down`. `Some` only for
    /// bitnet; `None` elsewhere.
    pub(super) ffn_sub_norm: Option<TensorId>,
    pub(super) post_ffw: Option<TensorId>,
    // gemma4 E2B per-layer input embedding: inp_gate, proj, post_norm.
    pub(super) pl_inp_gate: Option<TensorId>,
    pub(super) pl_proj: Option<TensorId>,
    pub(super) pl_post_norm: Option<TensorId>,
}

/// Session-stable derivations that are pure in `(backend caps, gguf, config, env)` and therefore
/// identical on every (warm) call for a given session — computed ONCE at cold init and reused
/// (via `Arc`) on warm calls and forks instead of re-running the per-layer tensor scans / real
/// `load_tensor_dequant`s every request. See `runner::session_stable`.
pub(crate) struct SessionStable {
    /// Per-layer presence of an explicit V projection (gemma4 full-attention layers omit it).
    pub(super) has_wv: Vec<bool>,
    /// gemma4 per-layer output scalar (`layer_output_scale` / `enc_layer_output_scale`), dequanted.
    pub(super) out_scale: Vec<Option<f32>>,
    /// diffusion-gemma DECODER per-layer output scalar (`layer_output_scale`), dequanted.
    pub(super) dec_out_scale: Vec<Option<f32>>,
    /// gemma4 proportional-RoPE frequency divisors (`rope_freqs.weight`), dequanted.
    pub(super) rope_freqs: Option<Vec<f32>>,
    /// DeepSeek V2+ YaRN per-pair frequency divisors (`qk_rope_dim/2` floats), computed from
    /// `rope_scaling_factor`/`n_ctx_train`/`rope_theta` — see `runner::session_stable`.
    pub(super) yarn_ff: Option<Vec<f32>>,
    /// Combined gate+up FFN upload decision.
    pub(super) fuse_gu: bool,
    /// Combined QKV upload decision.
    pub(super) fuse_qkv: bool,
    /// Experimental combined QSA query/key projection and prepare decision.
    pub(super) fuse_qsa_prepare: bool,
    /// Adjacent F32 DeltaNet alpha/beta projections execute as one backend dispatch.
    pub(super) fuse_delta_ab: bool,
    pub(super) fuse_qwen_hc_down_inject: bool,
    /// Per-layer Ling KDA QKV layout: true for one fused bank, false for QK + V.
    pub(super) kda_qkv_fused: Vec<bool>,
    /// Whether the MoE expert banks all have a dp4a-mmq kernel (batched-prefill eligibility).
    pub(super) moe_batched_ok: bool,
    /// Whether the MoE expert banks are IQ-grid dtypes the CPU Super batch kernels run at ANY
    /// row count (glm5next: Iq2Xxs gate/up + Iq2S/Iq3Xxs down). The `MOE_MMQ_DTYPES` scan above
    /// is the Vulkan dp4a-mmq story and carries no Iq2Xxs kernel, which stranded every glm5next
    /// prefill on one token per submit; the CPU backend's `expert_acts_kind` Super path (the same
    /// `vec_dot_iq*_batch` kernels `Op::Linear` runs for these formats) batches them instead.
    /// CPU-only gate — see the `batched_prefill_ok` call site in `runner`.
    pub(super) moe_super_batched_ok: bool,
}

pub(crate) struct SeamKv {
    /// The uploaded weights, SHARED across slots (Arc): forking a new conversation slot costs
    /// only its KV + IO buffers, never a re-upload.
    pub(super) weights: std::sync::Arc<SeamWeights>,
    /// Session-stable pure derivations (see [`SessionStable`]) — shared across warm calls + forks.
    pub(super) stable: std::sync::Arc<SessionStable>,
    pub(super) kbufs: Vec<Box<dyn Buffer>>,
    pub(super) vbufs: Vec<Box<dyn Buffer>>,
    /// Qwen3.8 QSA raw index-key cache. Full-attention layers own one F16 row per token;
    /// recurrent layers have no entry.
    pub(super) qsa_kbufs: Vec<Option<Box<dyn Buffer>>>,
    /// Qwen3.8 QSA final block-key cache. Each complete compressed block owns one F32 row.
    pub(super) qsa_cbufs: Vec<Option<Box<dyn Buffer>>>,
    /// Lazily committed Qwen KV state. The graph still sees the full logical context, while this
    /// tracks how many 32K-token physical segments are currently backed by the unified arena.
    pub(super) segmented_kv: SegmentedKvState,
    /// KV cache element dtypes, chosen per-side (K and V independent). Fork/seed reuse them so a
    /// forked slot sizes + copies its buffers to match this slot's layout.
    pub(super) k_fmt: DType,
    pub(super) v_fmt: DType,
    pub(super) hidden_buf: Box<dyn Buffer>,
    pub(super) pos_buf: Box<dyn Buffer>,
    pub(super) ipl_buf: Option<Box<dyn Buffer>>,
    pub(super) logits_buf: Box<dyn Buffer>,
    /// Qwen3.8's four-stream residual carried across the layer-0/PLE split execution.
    pub(super) qwen_wide_buf: Option<Box<dyn Buffer>>,
    /// Host-gathered PLE rows uploaded after layer 0 finishes.
    pub(super) ple_embd_buf: Option<Box<dyn Buffer>>,
    /// Persistent PLE dilated-convolution history (9 x hc*n_embd f32 on the released model).
    pub(super) ple_state_buf: Option<Box<dyn Buffer>>,
    /// Fixed Qwen3.8 MTP VERIFY IO. A single slot needs four rows; parallel serving reserves the
    /// whole four-rows-per-lane cohort so whichever slot becomes the primary can own the graph.
    /// These buffers are allocated before the deferred Vulkan pager measures the remaining room,
    /// then replayed by every speculative cycle.
    pub(super) mtp_verify_bufs: Option<QwenMtpVerifyBuffers>,
    /// The context this slot's KV cache was ACTUALLY allocated for. Usually the `want_ctx` the
    /// caller asked for; smaller when the cold init's live-room re-clamp shrank it (see
    /// `crate::seam::reclamp_ctx_to_live_room`), which is why callers holding a `want_ctx` of
    /// their own read it back from here ([`SeamKv::max_ctx`]) once the first generation returns.
    pub(super) max_ctx: usize,
    /// Whether this session's SWA layers were allocated as window-sized RINGS (see
    /// `crate::seam::kv_rows`): fork must size its buffers identically, and seed must respect
    /// that a wrapped ring no longer retains the early prefix rows a seed would copy.
    pub(super) kv_ring: bool,
    /// Token ids whose KV rows are materialized (prompt + generated of the last turn).
    pub(super) cached: Vec<u32>,
    /// Phase-A perf: DiffusionGemma canvas-denoise plan + staging buffers, `None` for every
    /// non-diffusion-gemma caller (never populated). Reset to `None` whenever the (cc, p) key
    /// changes (see `DenoiseCache`).
    pub(super) denoise_cache: Option<DenoiseCache>,
    /// Phase-A perf: DiffusionGemma self-conditioning MLP weights, dequantized lazily on the first
    /// denoise call with self-conditioning ON. `Arc` so `fork()` shares it with forked conversation
    /// slots for free (a pure function of the model, not per-conversation state).
    pub(super) self_cond_w: Option<std::sync::Arc<SelfCondWeights>>,
    /// Phase-B/D perf: the in-graph SC soft-embedding weight (`token_embd` dequantized + transposed
    /// to f16 `[n_embd, n_vocab]`, ~1.4 GiB — see the reference's `dg_ensure_sc_embT` and
    /// `build_sc_embt`), built lazily on the FIRST Vulkan/Metal denoise call with SC on. `None` for
    /// CPU (it never sets it) and for every non-diffusion-gemma caller. `Arc` so `fork()`
    /// shares it with forked conversation slots for free — mirrors `self_cond_w`.
    pub(super) sc_embt: Option<std::sync::Arc<dyn Buffer>>,
    /// Perf (Vulkan only — see docs/diffusion-gemma.md's Phase-B "sc round-trip" elimination):
    /// ping-pong pair of persistent `[cc*vocab]` device buffers backing the denoise loop's canvas
    /// logits, so the previous step's raw output is already GPU-resident for this step's
    /// self-conditioning softmax input — no host download+reupload. Session-lifetime: `cc`/vocab
    /// are fixed for the whole session, so this pair survives every `denoise_cache` rebuild
    /// (block boundaries, the sc off→on plan-shape transition). `None` until the first Vulkan
    /// denoise call; stays `None` forever on Metal/CPU (they keep the original per-plan-owned
    /// `DenoiseCache::logits_buf`/`sc_logits_buf`).
    pub(super) sc_ping: Option<[Box<dyn Buffer>; 2]>,
    /// Which `sc_ping` slot the NEXT denoise call's LM-head output lands in (flips every call —
    /// the OTHER slot holds the value to read as that call's self-conditioning input, already
    /// GPU-resident from the call that wrote it).
    pub(super) sc_ping_write: usize,
    /// 4-byte device scalar holding the CURRENT call's self-conditioning `temp_inv`, read by the
    /// dynamic-scale softmax (`Op::Softmax::scale_buf`) instead of a per-step host premultiply of
    /// the whole `[cc, vocab]` logits buffer. Lazily allocated alongside `sc_ping`.
    pub(super) sc_temp_inv_buf: Option<Box<dyn Buffer>>,
    /// MTP spec-decode rollback checkpoint (`mtp_snapshot_delta`/`mtp_restore_delta`): a device-
    /// resident copy of every qwen35 DeltaNet layer's recurrent state at the last CLEAN committed
    /// boundary, plus the cached-token length there. Lets a partial-accept cycle roll the trunk's
    /// draft-polluted state back to a committed prefix and re-prefill only the short accepted suffix,
    /// instead of qwen35's default full re-prefill-from-zero (its append-only DeltaNet state can't
    /// rewind by cache truncation the way a per-position KV cache can — see the `c.qwen35` branch in
    /// `generate_dense_backend`'s `start` computation). `None` for every non-MTP caller.
    pub(super) mtp_delta_ckpt: Option<MtpDeltaCkpt>,
    /// Two conversation checkpoints for append-only recurrent mixers: the reusable agent prefix
    /// and the prefix immediately before the latest user message. Attention/QSA rows remain in
    /// their ordinary cache; only the fixed recurrent/PLE summaries are duplicated here.
    pub(super) turn_recurrent_ckpts: [Option<TurnRecurrentCkpt>; super::TURN_CHECKPOINT_COUNT],
    /// Additional serve slots whose persistent device allocations were materialized beside slot
    /// 0 before the unified arena consumed the measured remainder. Startup drains this vector
    /// immediately; ordinary one-shot/session paths always keep it empty.
    pub(super) preallocated_siblings: Vec<SeamKv>,
}

pub(super) struct QwenMtpVerifyBuffers {
    pub(super) ids: Box<dyn Buffer>,
    pub(super) positions: Box<dyn Buffer>,
    pub(super) positions4: Box<dyn Buffer>,
    pub(super) hidden: Box<dyn Buffer>,
    pub(super) wide: Box<dyn Buffer>,
    pub(super) ple: Box<dyn Buffer>,
    pub(super) logits: Box<dyn Buffer>,
    pub(super) h_out: Box<dyn Buffer>,
    pub(super) out_ids: Box<dyn Buffer>,
}

impl QwenMtpVerifyBuffers {
    pub(super) fn allocate(be: &dyn Backend, cfg: &Config) -> AResult<Self> {
        Self::allocate_rows(
            be,
            cfg,
            crate::mtp::DRAFT_TOKENS * super::placement_slots() as usize,
        )
    }

    fn rows(&self) -> usize {
        self.ids.len_bytes() / std::mem::size_of::<u32>()
    }

    pub(super) fn allocate_rows(be: &dyn Backend, cfg: &Config, rows: usize) -> AResult<Self> {
        let h_width = super::target_verify_hidden_width(cfg);
        let ple_row = super::target_verify_ple_row(cfg);
        let alloc = |bytes, usage| be.alloc(bytes, usage).map_err(|e| anyhow!("{e}"));
        Ok(Self {
            ids: alloc(rows * 4, BufferUsage::Staging)?,
            positions: alloc(rows * 4, BufferUsage::Staging)?,
            positions4: alloc(rows * 4 * 4, BufferUsage::Staging)?,
            hidden: be
                .alloc_uninit(rows * cfg.n_embd * 4, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
            wide: be
                .alloc_uninit(rows * h_width * 4, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
            ple: alloc((rows * ple_row * 4).max(4), BufferUsage::Staging)?,
            logits: alloc(rows * cfg.vocab * 4, BufferUsage::Staging)?,
            h_out: alloc(rows * h_width * 4, BufferUsage::Readback)?,
            out_ids: alloc(rows * 4, BufferUsage::Readback)?,
        })
    }
}

#[derive(Default)]
pub(super) struct SegmentedKvState {
    pub(super) enabled: bool,
    pub(super) committed_tokens: usize,
}

impl SegmentedKvState {
    pub(super) fn enabled() -> Self {
        Self {
            enabled: true,
            committed_tokens: 0,
        }
    }

    /// Commit every per-token plane through `tokens`. The fast path is one comparison; allocation
    /// and address-table updates happen only when a call crosses a 32K-token boundary.
    pub(super) fn ensure_depth(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        max_ctx: usize,
        k_fmt: DType,
        v_fmt: DType,
        kbufs: &[Box<dyn Buffer>],
        vbufs: &[Box<dyn Buffer>],
        qsa_kbufs: &[Option<Box<dyn Buffer>>],
        qsa_cbufs: &[Option<Box<dyn Buffer>>],
        tokens: usize,
    ) -> AResult<()> {
        if !self.enabled || tokens <= self.committed_tokens {
            return Ok(());
        }
        if tokens > max_ctx {
            return Err(anyhow!(
                "KV depth {tokens} exceeds the session capacity {max_ctx}"
            ));
        }
        let layout = SegmentedKvLayout::for_qwen(cfg, max_ctx, k_fmt, v_fmt)
            .ok_or_else(|| anyhow!("segmented KV enabled for a non-Qwen session"))?;
        let segments = layout.segments_for_tokens(tokens);
        let buffers: Vec<&dyn Buffer> = layout
            .planes
            .iter()
            .map(|plane| -> AResult<&dyn Buffer> {
                Ok(match plane.kind {
                    PlaneKind::K => kbufs[plane.layer].as_ref(),
                    PlaneKind::V => vbufs[plane.layer].as_ref(),
                    PlaneKind::QsaRaw => qsa_kbufs[plane.layer].as_deref().ok_or_else(|| {
                        anyhow!("missing QSA raw cache for layer {}", plane.layer)
                    })?,
                    PlaneKind::QsaBlock => qsa_cbufs[plane.layer].as_deref().ok_or_else(|| {
                        anyhow!("missing QSA block cache for layer {}", plane.layer)
                    })?,
                })
            })
            .collect::<AResult<_>>()?;
        be.ensure_segmented_kv_batch(&buffers, segments)
            .map_err(|e| anyhow!("commit segmented KV growth transaction: {e}"))?;
        self.committed_tokens = (segments * KV_GROW_ROWS).min(max_ctx);
        tracing::info!(
            requested_tokens = tokens,
            committed_tokens = self.committed_tokens,
            segments,
            "expanded dynamic KV cache"
        );
        Ok(())
    }

    fn clear(
        &self,
        be: &dyn Backend,
        cfg: &Config,
        kbufs: &[Box<dyn Buffer>],
        vbufs: &[Box<dyn Buffer>],
        qsa_kbufs: &[Option<Box<dyn Buffer>>],
        qsa_cbufs: &[Option<Box<dyn Buffer>>],
    ) -> AResult<()> {
        if !self.enabled {
            return Ok(());
        }
        for layer in 0..cfg.n_layer {
            if !cfg.is_recurrent_layer(layer) {
                be.clear_segmented_kv(kbufs[layer].as_ref())
                    .map_err(|e| anyhow!("clear segmented K cache at layer {layer}: {e}"))?;
                be.clear_segmented_kv(vbufs[layer].as_ref())
                    .map_err(|e| anyhow!("clear segmented V cache at layer {layer}: {e}"))?;
            }
            for (name, buffer) in [
                ("QSA raw", qsa_kbufs[layer].as_deref()),
                ("QSA block", qsa_cbufs[layer].as_deref()),
            ] {
                if let Some(buffer) = buffer {
                    be.clear_segmented_kv(buffer).map_err(|e| {
                        anyhow!("clear segmented {name} cache at layer {layer}: {e}")
                    })?;
                }
            }
        }
        Ok(())
    }
}

pub(super) fn alloc_segmented_plane(
    be: &dyn Backend,
    layout: &SegmentedKvLayout,
    layer: usize,
    kind: PlaneKind,
) -> AResult<Box<dyn Buffer>> {
    let plane = layout
        .plane(layer, kind)
        .ok_or_else(|| anyhow!("missing segmented KV geometry for layer {layer} {kind:?}"))?;
    be.alloc_segmented_kv(plane.spec(layout.max_ctx))
        .map_err(|e| anyhow!("allocate segmented KV layer {layer} {kind:?}: {e}"))?
        .ok_or_else(|| anyhow!("segmented KV support disappeared while allocating the session"))
}

/// The device-resident DeltaNet-state snapshot backing [`SeamKv::mtp_snapshot_delta`] — one
/// conv-state + one S-state buffer per qwen35 DeltaNet layer (parallel to `layers`), plus the
/// cached-token length the snapshot corresponds to. Qwen3.8 MTP allocates it before the dynamic
/// KV/expert pool is sized and reuses it every cycle. Lookup allocates only row traces until an
/// explicit snapshot requests a base; the fallback constructor remains lazy.
/// Append-only Attention uses an empty owner: only frontier/trace metadata, no device buffers.
pub(super) struct MtpDeltaCkpt {
    pub(super) kbufs: Vec<Box<dyn Buffer>>,
    pub(super) vbufs: Vec<Box<dyn Buffer>>,
    pub(super) ple_state: Option<Box<dyn Buffer>>,
    /// Row-major recurrent snapshots written as part of the target VERIFY scan. Each buffer is
    /// `trace_rows` consecutive copies of its corresponding live state.
    pub(super) trace_kbufs: Vec<Box<dyn Buffer>>,
    pub(super) trace_vbufs: Vec<Box<dyn Buffer>>,
    pub(super) trace_ple_state: Option<Box<dyn Buffer>>,
    /// The layer indices (into `SeamKv::kbufs`/`vbufs`) that are DeltaNet mixers.
    pub(super) layers: Vec<usize>,
    pub(super) trace_rows: usize,
    /// Forks preserve startup policy, even after a Lookup owner lazily acquires a base.
    eager_base: bool,
    base_allocated: bool,
    armed_verify_rows: Option<usize>,
    cached_len: usize,
    /// `cached_len` may instead be a Lookup trace base, with no corresponding base-buffer copy.
    clean_base_valid: bool,
    /// Lookup owns a positive-row transaction, never a zero-row checkpoint.
    lookup_verify_rows: Option<usize>,
}

impl MtpDeltaCkpt {
    fn empty(trace_rows: usize, eager_base: bool) -> Self {
        Self {
            kbufs: Vec::new(),
            vbufs: Vec::new(),
            ple_state: None,
            trace_kbufs: Vec::new(),
            trace_vbufs: Vec::new(),
            trace_ple_state: None,
            layers: Vec::new(),
            trace_rows,
            eager_base,
            base_allocated: eager_base,
            armed_verify_rows: None,
            cached_len: 0,
            clean_base_valid: false,
            lookup_verify_rows: None,
        }
    }

    fn invalidate(&mut self) {
        self.armed_verify_rows = None;
        self.clean_base_valid = false;
        self.lookup_verify_rows = None;
    }

    fn arm(&mut self, rows: usize) -> AResult<()> {
        anyhow::ensure!(
            rows > 0 && rows - 1 <= self.trace_rows,
            "target Verify requested {rows} rows, intermediate capacity is {}",
            self.trace_rows
        );
        self.armed_verify_rows = Some(rows);
        Ok(())
    }

    fn validate(&self, rows: usize, cached_len: usize) -> AResult<()> {
        anyhow::ensure!(
            (2..=crate::mtp::DRAFT_TOKENS).contains(&rows)
                && self.armed_verify_rows == Some(rows)
                && self.trace_rows >= rows - 1
                && self.cached_len == cached_len,
            "target Verify trace must be armed at its committed frontier for {rows} rows"
        );
        Ok(())
    }

    fn take_trace(&mut self) -> Option<(usize, usize)> {
        let rows = self.armed_verify_rows.take()?;
        Some((rows, rows - 1))
    }

    fn restored_len(&self, accepted: usize) -> AResult<usize> {
        anyhow::ensure!(
            accepted <= self.trace_rows,
            "target Verify restore requested {accepted} rows, capacity is {}",
            self.trace_rows
        );
        self.cached_len
            .checked_add(accepted)
            .ok_or_else(|| anyhow!("target Verify restore length overflows"))
    }

    fn allocate_for_fork(
        be: &dyn Backend,
        cfg: &Config,
        src_k: &[Box<dyn Buffer>],
        src_v: &[Box<dyn Buffer>],
        src_ple: Option<&dyn Buffer>,
        trace_rows: usize,
        eager_base: bool,
    ) -> AResult<Option<Self>> {
        Self::allocate_sized(
            be,
            cfg,
            |layer| Some(src_k[layer].len_bytes()),
            |layer| Some(src_v[layer].len_bytes()),
            src_ple.map(Buffer::len_bytes),
            trace_rows,
            eager_base,
        )
    }

    pub(super) fn allocate_before_dynamic_kv(
        be: &dyn Backend,
        cfg: &Config,
        src_k: &[Option<Box<dyn Buffer>>],
        src_v: &[Option<Box<dyn Buffer>>],
        src_ple: Option<&dyn Buffer>,
        trace_rows: usize,
        eager_base: bool,
    ) -> AResult<Option<Self>> {
        Self::allocate_sized(
            be,
            cfg,
            |layer| src_k[layer].as_ref().map(|buffer| buffer.len_bytes()),
            |layer| src_v[layer].as_ref().map(|buffer| buffer.len_bytes()),
            src_ple.map(Buffer::len_bytes),
            trace_rows,
            eager_base,
        )
    }

    fn allocate_sized(
        be: &dyn Backend,
        cfg: &Config,
        mut k_len: impl FnMut(usize) -> Option<usize>,
        mut v_len: impl FnMut(usize) -> Option<usize>,
        ple_len: Option<usize>,
        trace_rows: usize,
        eager_base: bool,
    ) -> AResult<Option<Self>> {
        let layers: Vec<usize> = (0..cfg.n_layer)
            .filter(|&layer| cfg.is_recurrent_layer(layer))
            .collect();
        if layers.is_empty() && ple_len.is_none() {
            return Ok(
                super::target_verify_supported(cfg).then(|| Self::empty(trace_rows, eager_base))
            );
        }
        let mut kbufs = Vec::with_capacity(layers.len());
        let mut vbufs = Vec::with_capacity(layers.len());
        let mut trace_kbufs = Vec::with_capacity(layers.len());
        let mut trace_vbufs = Vec::with_capacity(layers.len());
        for &layer in &layers {
            let kb = k_len(layer).ok_or_else(|| {
                anyhow!("MTP checkpoint layer {layer} has no fixed K/state allocation")
            })?;
            let vb = v_len(layer).ok_or_else(|| {
                anyhow!("MTP checkpoint layer {layer} has no fixed V/state allocation")
            })?;
            if eager_base {
                kbufs.push(
                    be.alloc(kb.max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                vbufs.push(
                    be.alloc(vb.max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
            }
            if trace_rows > 0 {
                trace_kbufs.push(
                    be.alloc(
                        kb.checked_mul(trace_rows)
                            .ok_or_else(|| anyhow!("MTP conv trace allocation overflow"))?
                            .max(1),
                        BufferUsage::KvCache,
                    )
                    .map_err(|e| anyhow!("{e}"))?,
                );
                trace_vbufs.push(
                    be.alloc(
                        vb.checked_mul(trace_rows)
                            .ok_or_else(|| anyhow!("MTP DeltaNet trace allocation overflow"))?
                            .max(1),
                        BufferUsage::KvCache,
                    )
                    .map_err(|e| anyhow!("{e}"))?,
                );
            }
        }
        let ple_state = ple_len
            .filter(|_| eager_base)
            .map(|bytes| {
                be.alloc(bytes.max(1), BufferUsage::KvCache)
                    .map_err(|e| anyhow!("{e}"))
            })
            .transpose()?;
        let trace_ple_state = if trace_rows > 0 {
            ple_len
                .map(|bytes| {
                    be.alloc(
                        bytes
                            .checked_mul(trace_rows)
                            .ok_or_else(|| anyhow!("MTP PLE trace allocation overflow"))?
                            .max(1),
                        BufferUsage::KvCache,
                    )
                    .map_err(|e| anyhow!("{e}"))
                })
                .transpose()?
        } else {
            None
        };
        Ok(Some(Self {
            kbufs,
            vbufs,
            ple_state,
            trace_kbufs,
            trace_vbufs,
            trace_ple_state,
            layers,
            trace_rows,
            eager_base,
            base_allocated: eager_base,
            armed_verify_rows: None,
            cached_len: 0,
            clean_base_valid: false,
            lookup_verify_rows: None,
        }))
    }
}

/// Device-resident recurrent state at one stable rendered conversation boundary. Buffers are
/// allocated once per slot and reused; `copied` lets layer-major prefill capture each recurrent
/// layer exactly when that layer reaches the boundary without any per-operation allocation.
pub(super) struct TurnRecurrentCkpt {
    pub(super) kbufs: Vec<Box<dyn Buffer>>,
    pub(super) vbufs: Vec<Box<dyn Buffer>>,
    /// Qwen3.8's model-level PLE convolution history is recurrent state too. Keeping it beside
    /// the per-layer DeltaNet snapshots makes one stable conversation boundary self-contained.
    pub(super) ple_state: Option<Box<dyn Buffer>>,
    pub(super) layers: Vec<usize>,
    pub(super) tokens: Vec<u32>,
    pub(super) copied: Vec<bool>,
    pub(super) ple_copied: bool,
    pub(super) valid: bool,
}

fn checkpoint_extension_start(checkpoint: &[u32], prompt: &[u32]) -> Option<usize> {
    (!checkpoint.is_empty() && checkpoint.len() < prompt.len() && prompt.starts_with(checkpoint))
        .then_some(checkpoint.len())
}

impl TurnRecurrentCkpt {
    pub(super) fn invalidate(&mut self) {
        self.valid = false;
        self.tokens.clear();
        self.copied.fill(false);
        self.ple_copied = false;
    }

    /// Start replacing the rolling checkpoint with `tokens`. The existing device buffers are
    /// retained; validity is published only after every recurrent layer has been copied.
    pub(super) fn begin(
        slot: &mut Option<Self>,
        be: &dyn Backend,
        cfg: &Config,
        src_k: &[Box<dyn Buffer>],
        src_v: &[Box<dyn Buffer>],
        src_ple: Option<&dyn Buffer>,
        tokens: &[u32],
    ) -> AResult<()> {
        Self::begin_sized(
            slot,
            be,
            cfg,
            |layer| Some(src_k[layer].len_bytes()),
            |layer| Some(src_v[layer].len_bytes()),
            src_ple.map(Buffer::len_bytes),
            tokens,
        )
    }

    /// Cold-load variant used while segmented KV planes are intentionally still unbound. Every
    /// recurrent layer is already a concrete fixed allocation; dynamic planes remain `None` and
    /// are never inspected because they are not checkpoint state.
    pub(super) fn begin_before_dynamic_kv(
        slot: &mut Option<Self>,
        be: &dyn Backend,
        cfg: &Config,
        src_k: &[Option<Box<dyn Buffer>>],
        src_v: &[Option<Box<dyn Buffer>>],
        src_ple: Option<&dyn Buffer>,
        tokens: &[u32],
    ) -> AResult<()> {
        Self::begin_sized(
            slot,
            be,
            cfg,
            |layer| src_k[layer].as_ref().map(|buffer| buffer.len_bytes()),
            |layer| src_v[layer].as_ref().map(|buffer| buffer.len_bytes()),
            src_ple.map(Buffer::len_bytes),
            tokens,
        )
    }

    fn begin_sized(
        slot: &mut Option<Self>,
        be: &dyn Backend,
        cfg: &Config,
        mut k_len: impl FnMut(usize) -> Option<usize>,
        mut v_len: impl FnMut(usize) -> Option<usize>,
        ple_len: Option<usize>,
        tokens: &[u32],
    ) -> AResult<()> {
        if slot.is_none() {
            let layers: Vec<usize> = (0..cfg.n_layer)
                .filter(|&l| cfg.is_recurrent_layer(l))
                .collect();
            if layers.is_empty() {
                return Ok(());
            }
            let mut kbufs = Vec::with_capacity(layers.len());
            let mut vbufs = Vec::with_capacity(layers.len());
            for &l in &layers {
                let kb = k_len(l).ok_or_else(|| {
                    anyhow!("recurrent checkpoint layer {l} has no fixed K/state allocation")
                })?;
                let vb = v_len(l).ok_or_else(|| {
                    anyhow!("recurrent checkpoint layer {l} has no fixed V/state allocation")
                })?;
                kbufs.push(
                    be.alloc(kb.max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                vbufs.push(
                    be.alloc(vb.max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
            }
            let ple_state = ple_len
                .map(|bytes| {
                    be.alloc(bytes.max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))
                })
                .transpose()?;
            let copied = vec![false; layers.len()];
            *slot = Some(Self {
                kbufs,
                vbufs,
                ple_state,
                layers,
                tokens: Vec::new(),
                copied,
                ple_copied: false,
                valid: false,
            });
        }
        let ck = slot.as_mut().expect("checkpoint was just allocated");
        if ck.ple_state.is_some() != ple_len.is_some() {
            return Err(anyhow!(
                "stable recurrent checkpoint PLE shape changed within one session"
            ));
        }
        ck.tokens.clear();
        ck.tokens.extend_from_slice(tokens);
        ck.copied.fill(false);
        ck.ple_copied = ple_len.is_none();
        ck.valid = false;
        Ok(())
    }

    /// Capture one recurrent layer. Used by layer-major prefill at the precise boundary.
    pub(super) fn snapshot_layer(
        &mut self,
        be: &dyn Backend,
        src_k: &[Box<dyn Buffer>],
        src_v: &[Box<dyn Buffer>],
        layer: usize,
    ) -> AResult<()> {
        let Some(i) = self.layers.iter().position(|&l| l == layer) else {
            return Ok(());
        };
        self.snapshot_index(be, src_k, src_v, i)
    }

    fn snapshot_index(
        &mut self,
        be: &dyn Backend,
        src_k: &[Box<dyn Buffer>],
        src_v: &[Box<dyn Buffer>],
        i: usize,
    ) -> AResult<()> {
        if self.copied[i] {
            return Ok(());
        }
        let layer = self.layers[i];
        let copies = [
            (
                src_k[layer].as_ref(),
                self.kbufs[i].as_ref(),
                src_k[layer].len_bytes(),
            ),
            (
                src_v[layer].as_ref(),
                self.vbufs[i].as_ref(),
                src_v[layer].len_bytes(),
            ),
        ];
        be.copy_buffers(&copies).map_err(|e| anyhow!("{e}"))?;
        self.copied[i] = true;
        self.refresh_valid();
        Ok(())
    }

    /// Capture Qwen3.8's PLE history after the span containing its state update reaches the stable
    /// boundary. Other recurrent architectures carry no PLE buffer and are already complete.
    pub(super) fn snapshot_ple(
        &mut self,
        be: &dyn Backend,
        src: Option<&dyn Buffer>,
    ) -> AResult<()> {
        match (self.ple_state.as_ref(), src) {
            (Some(dst), Some(src)) if !self.ple_copied => {
                be.copy_buffer(src, dst.as_ref(), src.len_bytes())
                    .map_err(|e| anyhow!("{e}"))?;
                self.ple_copied = true;
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(anyhow!(
                    "stable recurrent checkpoint PLE source differs from its allocation"
                ));
            }
            _ => {}
        }
        self.refresh_valid();
        Ok(())
    }

    fn refresh_valid(&mut self) {
        self.valid =
            !self.tokens.is_empty() && self.copied.iter().all(|&done| done) && self.ple_copied;
    }

    /// Capture every recurrent layer. Used by chunk-major prefill and the per-token path.
    pub(super) fn snapshot_all(
        &mut self,
        be: &dyn Backend,
        src_k: &[Box<dyn Buffer>],
        src_v: &[Box<dyn Buffer>],
        src_ple: Option<&dyn Buffer>,
    ) -> AResult<()> {
        if self.ple_state.is_some() != src_ple.is_some() {
            return Err(anyhow!(
                "stable recurrent checkpoint PLE source differs from its allocation"
            ));
        }

        let mut copies =
            Vec::with_capacity(self.layers.len() * 2 + usize::from(self.ple_state.is_some()));
        for (i, &layer) in self.layers.iter().enumerate() {
            if self.copied[i] {
                continue;
            }
            copies.push((
                src_k[layer].as_ref(),
                self.kbufs[i].as_ref(),
                src_k[layer].len_bytes(),
            ));
            copies.push((
                src_v[layer].as_ref(),
                self.vbufs[i].as_ref(),
                src_v[layer].len_bytes(),
            ));
        }
        if !self.ple_copied {
            if let (Some(src), Some(dst)) = (src_ple, self.ple_state.as_deref()) {
                copies.push((src, dst, src.len_bytes()));
            }
        }
        be.copy_buffers(&copies).map_err(|e| anyhow!("{e}"))?;
        drop(copies);

        self.copied.fill(true);
        self.ple_copied = true;
        self.refresh_valid();
        Ok(())
    }
}

/// The upload-once half of a [`SeamKv`]: weight buffers + their declared (dtype, numel) specs and
/// the rope_freqs constant. Shared across conversation slots via `Arc`.
pub(crate) struct SeamWeights {
    pub(super) wbufs: Vec<Box<dyn Buffer>>,
    pub(super) wspecs: Vec<(DType, usize)>,
    pub(super) rf_buf: Option<(Box<dyn Buffer>, usize)>,
    /// DeepSeek V2+ YaRN per-pair frequency divisors (`yarn_ff`), uploaded once like `rf_buf`.
    pub(super) yff_buf: Option<(Box<dyn Buffer>, usize)>,
    /// Per-layer: whether `ffn_exp_probs_b.weight` was loaded (DeepSeek V3+ router bias).
    pub(super) layer_has_epb: Vec<bool>,
    pub(super) layer_fused_experts: Vec<bool>,
    pub(super) ple_worker: Option<std::sync::Arc<super::ple::PleWorker>>,
    /// Main-model tables borrowed by Qwen3.8's detached shared MTP head. Indices point into
    /// `wbufs`/`wspecs`; the sidecar deliberately omits both tensors.
    pub(super) mtp_token_embd_index: usize,
    pub(super) mtp_lm_head_index: usize,
}

type MtpSharedWeight<'a> = (&'a dyn Buffer, DType, usize);
type MtpSharedWeights<'a> = (MtpSharedWeight<'a>, MtpSharedWeight<'a>);

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SeamKv {
    pub(crate) fn mtp_shared_weights(&self) -> MtpSharedWeights<'_> {
        let wi = &self.weights;
        let emb = wi.mtp_token_embd_index;
        let lm = wi.mtp_lm_head_index;
        (
            (wi.wbufs[emb].as_ref(), wi.wspecs[emb].0, wi.wspecs[emb].1),
            (wi.wbufs[lm].as_ref(), wi.wspecs[lm].0, wi.wspecs[lm].1),
        )
    }

    pub(crate) fn take_preallocated_siblings(&mut self) -> Vec<SeamKv> {
        std::mem::take(&mut self.preallocated_siblings)
    }

    /// The context this slot's KV cache was allocated for — the AUTHORITY on a session's window,
    /// because the cold init may have re-clamped the caller's `want_ctx` against the device's live
    /// free memory (`crate::seam::reclamp_ctx_to_live_room`). A caller that keeps its own copy
    /// (`DenseVulkanSession::max_ctx`, `ParallelSeam::max_ctx`) refreshes it from here after the
    /// first generation, so what it advertises is what was allocated.
    pub(crate) fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Whether this slot's allocated KV caches are coupled Q8_0 on both K and V sides.
    pub(crate) fn kv_q8(&self) -> bool {
        self.k_fmt == DType::Q8_0 && self.v_fmt == DType::Q8_0
    }

    /// Allocated cache formats, without cloning token history or accessing device buffers.
    pub(crate) fn kv_formats(&self) -> (DType, DType) {
        (self.k_fmt, self.v_fmt)
    }

    /// Longest common prefix of this slot's materialized tokens and `prompt` — the slot-selection
    /// score for multi-conversation serve.
    pub(crate) fn prefix_score(&self, prompt: &[u32]) -> usize {
        common_prefix_len(&self.cached, prompt)
    }

    /// Longest state this slot can continue for `prompt`: either its live materialized state or
    /// the rolling recurrent checkpoint. The live-state rule remains exactly the existing one;
    /// a checkpoint is usable only for a strict extension of its complete token prefix.
    pub(crate) fn continuation_prefix_len(&self, prompt: &[u32]) -> Option<usize> {
        let live_score = self.prefix_score(prompt);
        let live = (live_score > 0
            && (live_score == self.cached.len() || live_score == prompt.len()))
        .then_some(live_score);
        let checkpoint = self
            .turn_recurrent_ckpts
            .iter()
            .flatten()
            .filter_map(|ck| {
                ck.valid
                    .then(|| checkpoint_extension_start(&ck.tokens, prompt))
                    .flatten()
            })
            .max();
        live.into_iter().chain(checkpoint).max()
    }

    /// Longest prefix this slot can copy into another slot without replaying model work. Hybrid
    /// recurrent models can only export their live frontier or a captured turn checkpoint; an
    /// arbitrary token LCP has no matching recurrent/PLE summary.
    pub(crate) fn seedable_prefix_len(&self, cfg: &Config, prompt: &[u32]) -> usize {
        if cfg.qwen35 || cfg.bailingmoe3 || cfg.glm5next || cfg.deepseek4 {
            return 0;
        }
        if !cfg.qwen4exp {
            return self.prefix_score(prompt);
        }
        let live = (!self.cached.is_empty() && prompt.starts_with(&self.cached))
            .then_some(self.cached.len());
        let checkpoint = self
            .turn_recurrent_ckpts
            .iter()
            .flatten()
            .filter(|checkpoint| {
                checkpoint.valid
                    && self.cached.starts_with(&checkpoint.tokens)
                    && prompt.starts_with(&checkpoint.tokens)
            })
            .map(|checkpoint| checkpoint.tokens.len())
            .max();
        live.into_iter().chain(checkpoint).max().unwrap_or(0)
    }

    /// Forget the materialized tokens WITHOUT dropping weights or buffers: the next call
    /// re-prefills from position 0 into the same session. Bench reps use this so each rep
    /// measures a full prefill while weights/pipelines/repack caches stay warm.
    /// (cfg-gated with its only caller, the Metal bench session — dead code on other targets.)
    #[cfg(target_os = "macos")]
    pub(crate) fn reset_tokens(&mut self) {
        self.cached.clear();
        self.invalidate_turn_checkpoint();
        self.invalidate_delta_checkpoint();
    }

    /// Number of token ids materialized in this slot's KV cache.
    pub(crate) fn cached_len(&self) -> usize {
        self.cached.len()
    }

    pub(crate) fn cached_tokens(&self) -> &[u32] {
        &self.cached
    }

    /// Forget the materialized tokens (the KV rows become dead; the next prompt prefills from
    /// row 0). Used to discard a warmup generation without dropping the slot's buffers.
    pub(crate) fn reset(&mut self) {
        self.cached.clear();
        self.invalidate_turn_checkpoint();
        self.invalidate_delta_checkpoint();
    }

    fn invalidate_delta_checkpoint(&mut self) {
        if let Some(ck) = self.mtp_delta_ckpt.as_mut() {
            ck.invalidate();
        }
    }

    /// Commit this slot's segmented planes through `tokens`. The ordinary runner invokes the same
    /// state method through its local macro; layer-synchronous decode needs it once per lane.
    pub(super) fn ensure_segmented_depth(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        tokens: usize,
    ) -> AResult<()> {
        self.segmented_kv.ensure_depth(
            be,
            cfg,
            self.max_ctx,
            self.k_fmt,
            self.v_fmt,
            &self.kbufs,
            &self.vbufs,
            &self.qsa_kbufs,
            &self.qsa_cbufs,
            tokens,
        )
    }

    /// Benchmark-only synthetic context: mark the first `tokens.len()` positions as resident without
    /// running the model over them. KV buffers are zero-initialized at allocation; for qwen35's
    /// fixed recurrent state, clear the small state buffers here so repeated synthetic reps start
    /// from deterministic data too.
    pub(crate) fn set_synthetic_cached(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        tokens: Vec<u32>,
    ) -> AResult<()> {
        if tokens.len() > self.max_ctx {
            return Err(anyhow!(
                "synthetic depth {} exceeds the session KV capacity {}",
                tokens.len(),
                self.max_ctx
            ));
        }
        self.invalidate_delta_checkpoint();
        self.segmented_kv.ensure_depth(
            be,
            cfg,
            self.max_ctx,
            self.k_fmt,
            self.v_fmt,
            &self.kbufs,
            &self.vbufs,
            &self.qsa_kbufs,
            &self.qsa_cbufs,
            tokens.len(),
        )?;
        self.segmented_kv.clear(
            be,
            cfg,
            &self.kbufs,
            &self.vbufs,
            &self.qsa_kbufs,
            &self.qsa_cbufs,
        )?;
        if cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3 || cfg.glm5next {
            let conv_elems = (cfg.ssm_d_conv - 1) * cfg.recurrent_conv_channels();
            let s_elems = cfg.recurrent_state_elems();
            let conv_zero = vec![0f32; conv_elems];
            let s_zero = vec![0f32; s_elems];
            for l in 0..cfg.n_layer {
                if cfg.is_recurrent_layer(l) {
                    be.upload(self.kbufs[l].as_ref(), bytemuck::cast_slice(&conv_zero))
                        .map_err(|e| anyhow!("{e}"))?;
                    be.upload(self.vbufs[l].as_ref(), bytemuck::cast_slice(&s_zero))
                        .map_err(|e| anyhow!("{e}"))?;
                }
            }
            if let Some(state) = &self.ple_state_buf {
                let zeros = vec![0u8; state.len_bytes()];
                be.upload(state.as_ref(), &zeros)
                    .map_err(|e| anyhow!("{e}"))?;
            }
            if cfg.qwen4exp && !self.segmented_kv.enabled {
                for cache in self.qsa_kbufs.iter().chain(&self.qsa_cbufs).flatten() {
                    let zeros = vec![0u8; cache.len_bytes()];
                    be.upload(cache.as_ref(), &zeros)
                        .map_err(|e| anyhow!("{e}"))?;
                }
            }
        } else if cfg.deepseek4 {
            // Synthetic depth represents deterministic zero history. V4's compressor rings and
            // compressed caches survive ordinary token resets, so clear both packed state buffers
            // explicitly between benchmark reps; otherwise a prior rep's partial block would feed
            // the first new block at the manufactured depth.
            let max_bytes = self
                .kbufs
                .iter()
                .chain(self.vbufs.iter())
                .map(|b| b.len_bytes())
                .max()
                .unwrap_or(4);
            let zeros = vec![0u8; max_bytes];
            for b in self.kbufs.iter().chain(self.vbufs.iter()) {
                be.upload(b.as_ref(), &zeros[..b.len_bytes()])
                    .map_err(|e| anyhow!("{e}"))?;
            }
        }
        self.cached = tokens;
        self.invalidate_turn_checkpoint();
        Ok(())
    }

    fn invalidate_turn_checkpoint(&mut self) {
        for ck in self.turn_recurrent_ckpts.iter_mut().flatten() {
            ck.invalidate();
        }
    }

    /// Restore a rolling recurrent checkpoint when `prompt` strictly extends it. A mismatch is
    /// the ordinary `Option` fallback, not an error: the runner then performs its unchanged zero
    /// reset + full prefill. Copies stay entirely on the backend/device.
    pub(crate) fn restore_turn_recurrent(
        &mut self,
        be: &dyn Backend,
        prompt: &[u32],
    ) -> AResult<Option<usize>> {
        let selected = self
            .turn_recurrent_ckpts
            .iter()
            .enumerate()
            .filter_map(|(index, checkpoint)| {
                let checkpoint = checkpoint.as_ref()?;
                checkpoint
                    .valid
                    .then(|| checkpoint_extension_start(&checkpoint.tokens, prompt))
                    .flatten()
                    .map(|len| (index, len))
            })
            .max_by_key(|&(_, len)| len);
        let Some((selected, len)) = selected else {
            return Ok(None);
        };
        self.invalidate_delta_checkpoint();

        // A divergent branch overwrites every attention/QSA row after the selected boundary.
        // Any checkpoint from the old branch that is not itself a prefix of this prompt would
        // otherwise retain a valid recurrent summary beside invalid per-token cache rows.
        for (index, checkpoint) in self.turn_recurrent_ckpts.iter_mut().enumerate() {
            if index != selected {
                if let Some(checkpoint) = checkpoint.as_mut() {
                    if checkpoint.valid && !prompt.starts_with(&checkpoint.tokens) {
                        checkpoint.invalidate();
                    }
                }
            }
        }
        let ck = self.turn_recurrent_ckpts[selected]
            .as_ref()
            .expect("selected recurrent checkpoint exists");
        let mut copies =
            Vec::with_capacity(ck.layers.len() * 2 + usize::from(ck.ple_state.is_some()));
        for (i, &l) in ck.layers.iter().enumerate() {
            copies.push((
                ck.kbufs[i].as_ref(),
                self.kbufs[l].as_ref(),
                ck.kbufs[i].len_bytes(),
            ));
            copies.push((
                ck.vbufs[i].as_ref(),
                self.vbufs[l].as_ref(),
                ck.vbufs[i].len_bytes(),
            ));
        }
        match (ck.ple_state.as_ref(), self.ple_state_buf.as_deref()) {
            (Some(src), Some(dst)) => copies.push((src.as_ref(), dst, src.len_bytes())),
            (Some(_), None) | (None, Some(_)) => {
                return Err(anyhow!(
                    "stable recurrent checkpoint PLE target differs from its snapshot"
                ));
            }
            (None, None) => {}
        }
        be.copy_buffers(&copies).map_err(|e| anyhow!("{e}"))?;
        drop(copies);
        self.cached.clone_from(&ck.tokens);
        Ok(Some(len))
    }

    /// Fork a fresh conversation slot: same (Arc-shared) weights, its own zero KV + IO buffers.
    /// Snapshot the qwen35 DeltaNet recurrent state (every DeltaNet layer's conv + S buffers) plus
    /// the current `cached` length into the device-resident [`MtpDeltaCkpt`] (allocated once on the
    /// first call). The MTP spec-decode loop calls this at a CLEAN committed boundary (after the
    /// prime prefill and after every fully-accepted cycle) so a later partial-accept cycle can roll
    /// back to it via [`mtp_restore_delta`]. Attention-only targets record just the cached length.
    /// Device state uses buffer copies, never a host bounce; empty owners need no backend call.
    pub(crate) fn mtp_snapshot_delta(&mut self, be: &dyn Backend, cfg: &Config) -> AResult<()> {
        if self.mtp_delta_ckpt.is_none() {
            self.mtp_delta_ckpt = MtpDeltaCkpt::allocate_sized(
                be,
                cfg,
                |layer| Some(self.kbufs[layer].len_bytes()),
                |layer| Some(self.vbufs[layer].len_bytes()),
                self.ple_state_buf.as_deref().map(Buffer::len_bytes),
                0,
                true,
            )?;
        }
        if self.mtp_delta_ckpt.is_none() {
            return Ok(());
        }
        let ck = self.mtp_delta_ckpt.as_mut().expect("just ensured Some");
        anyhow::ensure!(
            ck.lookup_verify_rows.is_none(),
            "cannot snapshot an unsettled Lookup Verify"
        );
        // A failed copy must not publish a partially refreshed clean checkpoint.
        ck.clean_base_valid = false;
        if !ck.base_allocated {
            // Publish all base buffers together. A failed allocation drops the temporary owner,
            // leaving the row traces and live state intact for a later retry.
            // Late capacity is not reserved: backend budget checks may reject this allocation.
            let base = MtpDeltaCkpt::allocate_sized(
                be,
                cfg,
                |layer| Some(self.kbufs[layer].len_bytes()),
                |layer| Some(self.vbufs[layer].len_bytes()),
                self.ple_state_buf.as_deref().map(Buffer::len_bytes),
                0,
                true,
            )?
            .ok_or_else(|| anyhow!("target snapshot has no checkpoint runtime"))?;
            let ck = self.mtp_delta_ckpt.as_mut().expect("just ensured Some");
            ck.kbufs = base.kbufs;
            ck.vbufs = base.vbufs;
            ck.ple_state = base.ple_state;
            ck.base_allocated = true;
        }
        let cached_len = self.cached.len();
        let ck = self.mtp_delta_ckpt.as_ref().expect("just ensured Some");
        let mut copies =
            Vec::with_capacity(ck.layers.len() * 2 + usize::from(ck.ple_state.is_some()));
        for (i, &l) in ck.layers.iter().enumerate() {
            copies.push((
                self.kbufs[l].as_ref(),
                ck.kbufs[i].as_ref(),
                self.kbufs[l].len_bytes(),
            ));
            copies.push((
                self.vbufs[l].as_ref(),
                ck.vbufs[i].as_ref(),
                self.vbufs[l].len_bytes(),
            ));
        }
        if let (Some(src), Some(dst)) = (self.ple_state_buf.as_deref(), ck.ple_state.as_deref()) {
            copies.push((src, dst, src.len_bytes()));
        }
        if !copies.is_empty() {
            be.copy_buffers(&copies).map_err(|e| anyhow!("{e}"))?;
        }
        let ck = self.mtp_delta_ckpt.as_mut().expect("just ensured Some");
        ck.cached_len = cached_len;
        ck.clean_base_valid = true;
        Ok(())
    }

    /// Arm preallocated row traces at the actual live frontier, without copying a base snapshot.
    /// Lookup must retain its already-emitted pending input, even when delivery is cancelled.
    pub(crate) fn lookup_arm_delta_trace(&mut self, base: usize, rows: usize) -> AResult<()> {
        anyhow::ensure!(base == self.cached.len(), "Lookup trace frontier changed");
        let end = base
            .checked_add(rows)
            .ok_or_else(|| anyhow!("Lookup trace frontier overflows"))?;
        anyhow::ensure!(
            (2..=crate::mtp::DRAFT_TOKENS).contains(&rows) && end <= self.max_ctx,
            "Lookup trace width or frontier exceeds capacity"
        );
        let ck = self
            .mtp_delta_ckpt
            .as_mut()
            .ok_or_else(|| anyhow!("Lookup trace has no preallocated runtime"))?;
        anyhow::ensure!(
            ck.lookup_verify_rows.is_none() && ck.armed_verify_rows.is_none(),
            "Lookup trace already has an outstanding Verify"
        );
        ck.arm(rows)?;
        ck.cached_len = base;
        ck.clean_base_valid = false;
        ck.lookup_verify_rows = Some(rows);
        Ok(())
    }

    /// Retain a positive prefix of a completed Lookup Verify. The final row is already live.
    /// Only intermediate rows require a trace copy; no clean-base snapshot is made here.
    pub(crate) fn lookup_settle_delta_trace(
        &mut self,
        be: &dyn Backend,
        keep_inputs: usize,
    ) -> AResult<()> {
        let ck = self
            .mtp_delta_ckpt
            .as_ref()
            .ok_or_else(|| anyhow!("Lookup settle has no trace runtime"))?;
        let rows = ck
            .lookup_verify_rows
            .ok_or_else(|| anyhow!("Lookup settle has no outstanding Verify"))?;
        anyhow::ensure!(
            (1..=rows).contains(&keep_inputs)
                && ck.armed_verify_rows.is_none()
                && ck.cached_len.checked_add(rows) == Some(self.cached.len()),
            "Lookup settle requires a completed Verify and a positive retained prefix"
        );
        if keep_inputs < rows {
            if let Err(error) = self.mtp_restore_delta_row(be, keep_inputs) {
                // A backend may have copied only some layers: neither retry nor reuse is safe.
                self.reset();
                return Err(error);
            }
        }
        self.mtp_delta_ckpt
            .as_mut()
            .expect("Lookup trace runtime exists")
            .lookup_verify_rows = None;
        Ok(())
    }

    /// Arm the next target VERIFY to write its intermediate recurrent states. Row zero already has
    /// the clean checkpoint and a fully accepted batch leaves the live state intact, so an `n`-row
    /// VERIFY needs storage for only `n - 1` rows.
    pub(crate) fn mtp_arm_delta_trace(&mut self, rows: usize) -> AResult<()> {
        let ck = self
            .mtp_delta_ckpt
            .as_mut()
            .ok_or_else(|| anyhow!("MTP recurrent trace requested without a checkpoint runtime"))?;
        anyhow::ensure!(
            ck.clean_base_valid && ck.lookup_verify_rows.is_none(),
            "MTP trace requires a clean snapshot and no unsettled Lookup Verify"
        );
        ck.arm(rows)
    }

    pub(super) fn mtp_validate_delta_trace(&self, rows: usize) -> AResult<()> {
        let ck = self
            .mtp_delta_ckpt
            .as_ref()
            .ok_or_else(|| anyhow!("target Verify has no recurrent checkpoint runtime"))?;
        ck.validate(rows, self.cached.len())
    }

    pub(super) fn mtp_take_delta_trace(&mut self) -> Option<(usize, usize)> {
        let ck = self.mtp_delta_ckpt.as_mut()?;
        ck.take_trace()
    }

    /// Restore the DeltaNet state captured by the last [`mtp_snapshot_delta`] and truncate `cached`
    /// back to the snapshot's token length — the MTP loop's rollback after a partial-accept cycle
    /// (drops the rejected drafts the verify forward absorbed into the recurrent state). A no-op
    /// when no checkpoint runtime exists. An allocated but invalid snapshot is rejected.
    pub(crate) fn mtp_restore_delta(&mut self, be: &dyn Backend) -> AResult<()> {
        let Some(ck) = self.mtp_delta_ckpt.as_ref() else {
            return Ok(());
        };
        anyhow::ensure!(
            ck.clean_base_valid && ck.lookup_verify_rows.is_none(),
            "target zero-row restore has no clean snapshot"
        );
        let mut copies =
            Vec::with_capacity(ck.layers.len() * 2 + usize::from(ck.ple_state.is_some()));
        for (i, &l) in ck.layers.iter().enumerate() {
            copies.push((
                ck.kbufs[i].as_ref(),
                self.kbufs[l].as_ref(),
                ck.kbufs[i].len_bytes(),
            ));
            copies.push((
                ck.vbufs[i].as_ref(),
                self.vbufs[l].as_ref(),
                ck.vbufs[i].len_bytes(),
            ));
        }
        if let (Some(src), Some(dst)) = (ck.ple_state.as_deref(), self.ple_state_buf.as_deref()) {
            copies.push((src, dst, src.len_bytes()));
        }
        if !copies.is_empty() {
            be.copy_buffers(&copies).map_err(|e| anyhow!("{e}"))?;
        }
        self.cached.truncate(ck.cached_len);
        Ok(())
    }

    /// Restore the recurrent state immediately after `accepted` VERIFY rows. Row zero requires a
    /// valid clean checkpoint; positive rows come from the snapshots emitted inside the
    /// recurrent kernels during that VERIFY. This avoids replaying accepted target tokens.
    pub(crate) fn mtp_restore_delta_row(
        &mut self,
        be: &dyn Backend,
        accepted: usize,
    ) -> AResult<()> {
        if accepted == 0 {
            return self.mtp_restore_delta(be);
        }
        let ck = self
            .mtp_delta_ckpt
            .as_ref()
            .ok_or_else(|| anyhow!("MTP recurrent row restore has no checkpoint runtime"))?;
        if let Some(rows) = ck.lookup_verify_rows {
            anyhow::ensure!(
                accepted < rows
                    && ck.armed_verify_rows.is_none()
                    && ck.cached_len.checked_add(rows) == Some(self.cached.len()),
                "Lookup row restore requires its completed Verify frontier"
            );
        } else {
            anyhow::ensure!(
                ck.clean_base_valid,
                "target row restore has no valid trace base"
            );
        }
        let restored_len = ck.restored_len(accepted)?;
        let row = accepted - 1;
        let mut copies =
            Vec::with_capacity(ck.layers.len() * 2 + usize::from(ck.trace_ple_state.is_some()));
        for (i, &layer) in ck.layers.iter().enumerate() {
            let kb = self.kbufs[layer].len_bytes();
            let vb = self.vbufs[layer].len_bytes();
            copies.push((
                ck.trace_kbufs[i].as_ref(),
                row * kb,
                self.kbufs[layer].as_ref(),
                0,
                kb,
            ));
            copies.push((
                ck.trace_vbufs[i].as_ref(),
                row * vb,
                self.vbufs[layer].as_ref(),
                0,
                vb,
            ));
        }
        if let (Some(src), Some(dst)) =
            (ck.trace_ple_state.as_deref(), self.ple_state_buf.as_deref())
        {
            let bytes = dst.len_bytes();
            copies.push((src, row * bytes, dst, 0, bytes));
        }
        if !copies.is_empty() {
            be.copy_buffer_ranges(&copies)
                .map_err(|e| anyhow!("restore MTP recurrent row {row}: {e}"))?;
        }
        self.cached.truncate(restored_len);
        Ok(())
    }

    pub(crate) fn fork(
        &self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &crate::EngineConfig,
    ) -> AResult<SeamKv> {
        let e2b = self.ipl_buf.is_some();
        let npl = cfg.n_embd_per_layer.max(1);
        let mut kbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut vbufs: Vec<Box<dyn Buffer>> = Vec::new();
        let mut qsa_kbufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
        let mut qsa_cbufs: Vec<Option<Box<dyn Buffer>>> = Vec::new();
        let segmented_layout = self.segmented_kv.enabled.then(|| {
            SegmentedKvLayout::for_qwen(cfg, self.max_ctx, self.k_fmt, self.v_fmt)
                .expect("a segmented Qwen slot keeps its Qwen geometry when forked")
        });
        for l in 0..cfg.n_layer {
            let (k_bytes, v_bytes) = crate::seam::layer_state_bytes(
                cfg,
                l,
                self.max_ctx,
                self.kv_ring,
                crate::seam::ubatch_rows(ec),
                self.k_fmt,
                self.v_fmt,
            );
            kbufs.push(
                if let Some(layout) = segmented_layout
                    .as_ref()
                    .filter(|layout| layout.plane(l, PlaneKind::K).is_some())
                {
                    alloc_segmented_plane(be, layout, l, PlaneKind::K)?
                } else {
                    be.alloc(k_bytes, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?
                },
            );
            vbufs.push(
                if let Some(layout) = segmented_layout
                    .as_ref()
                    .filter(|layout| layout.plane(l, PlaneKind::V).is_some())
                {
                    alloc_segmented_plane(be, layout, l, PlaneKind::V)?
                } else {
                    be.alloc(v_bytes, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?
                },
            );
            let qsa_bytes = crate::seam::qsa_raw_cache_bytes(cfg, l, self.max_ctx);
            qsa_kbufs.push(if qsa_bytes > 0 {
                Some(if let Some(layout) = segmented_layout.as_ref() {
                    alloc_segmented_plane(be, layout, l, PlaneKind::QsaRaw)?
                } else {
                    be.alloc(qsa_bytes, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?
                })
            } else {
                None
            });
            let qsa_comp_bytes = crate::seam::qsa_block_cache_bytes(cfg, l, self.max_ctx);
            qsa_cbufs.push(if qsa_comp_bytes > 0 {
                Some(if let Some(layout) = segmented_layout.as_ref() {
                    alloc_segmented_plane(be, layout, l, PlaneKind::QsaBlock)?
                } else {
                    be.alloc(qsa_comp_bytes, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?
                })
            } else {
                None
            });
        }
        // A fork is only usable if its buffers have the SAME geometry as the slot it forked from,
        // and the source's own sizes are right here — so check them instead of trusting two call
        // sites to derive the same width. They did not: the MLA row width was missing from this
        // one, which allocated a third of what the kernels index (docs/backlog.md B41).
        for l in 0..cfg.n_layer {
            for (side, forked, src) in [
                ("k", kbufs[l].len_bytes(), self.kbufs[l].len_bytes()),
                ("v", vbufs[l].len_bytes(), self.vbufs[l].len_bytes()),
            ] {
                if forked != src {
                    return Err(anyhow!(
                        "forked KV slot geometry differs from its source at layer {l} ({side}): \
                         {forked} bytes vs {src} — the fork and the original allocation disagree \
                         about this layer's cache row"
                    ));
                }
            }
            match (&qsa_kbufs[l], &self.qsa_kbufs[l]) {
                (Some(forked), Some(src)) if forked.len_bytes() == src.len_bytes() => {}
                (None, None) => {}
                (forked, src) => {
                    return Err(anyhow!(
                        "forked QSA cache geometry differs from its source at layer {l}: {:?} vs {:?}",
                        forked.as_ref().map(|b| b.len_bytes()),
                        src.as_ref().map(|b| b.len_bytes())
                    ));
                }
            }
            match (&qsa_cbufs[l], &self.qsa_cbufs[l]) {
                (Some(forked), Some(src)) if forked.len_bytes() == src.len_bytes() => {}
                (None, None) => {}
                (forked, src) => {
                    return Err(anyhow!(
                        "forked QSA block cache geometry differs from its source at layer {l}: {:?} vs {:?}",
                        forked.as_ref().map(|b| b.len_bytes()),
                        src.as_ref().map(|b| b.len_bytes())
                    ));
                }
            }
        }
        let ple_state_buf = if self.ple_state_buf.is_some() {
            let hist = (cfg.ple_conv_kernel - 1) * cfg.ple_ngram_size;
            Some(
                be.alloc(hist * cfg.hc_mult * cfg.n_embd * 4, BufferUsage::KvCache)
                    .map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            None
        };
        let mut turn_recurrent_ckpts = std::array::from_fn(|_| None);
        if cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3 || cfg.glm5next {
            for checkpoint in &mut turn_recurrent_ckpts {
                TurnRecurrentCkpt::begin(
                    checkpoint,
                    be,
                    cfg,
                    &kbufs,
                    &vbufs,
                    ple_state_buf.as_deref(),
                    &[],
                )?;
            }
        }
        let mtp_delta_ckpt = if let Some(source) = self.mtp_delta_ckpt.as_ref() {
            MtpDeltaCkpt::allocate_for_fork(
                be,
                cfg,
                &kbufs,
                &vbufs,
                ple_state_buf.as_deref(),
                crate::mtp::DRAFT_TOKENS.saturating_sub(1),
                source.eager_base,
            )?
        } else {
            None
        };
        let mtp_verify_bufs = self
            .mtp_verify_bufs
            .as_ref()
            // Parallel MTP sorts a visual lane first so that lane owns the MRoPE bindings. That
            // lane is not necessarily slot 0, therefore every fork must preserve the root slot's
            // full-cohort capacity rather than falling back to the per-lane four-row default.
            .map(|source| QwenMtpVerifyBuffers::allocate_rows(be, cfg, source.rows()))
            .transpose()?;
        Ok(SeamKv {
            weights: std::sync::Arc::clone(&self.weights),
            stable: std::sync::Arc::clone(&self.stable),
            kbufs,
            vbufs,
            qsa_kbufs,
            qsa_cbufs,
            segmented_kv: if self.segmented_kv.enabled {
                SegmentedKvState::enabled()
            } else {
                SegmentedKvState::default()
            },
            k_fmt: self.k_fmt,
            v_fmt: self.v_fmt,
            hidden_buf: be
                .alloc(cfg.n_embd * 4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
            pos_buf: be
                .alloc(4, BufferUsage::Staging)
                .map_err(|e| anyhow!("{e}"))?,
            ipl_buf: if e2b {
                Some(
                    be.alloc(cfg.n_layer * npl * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            },
            logits_buf: be
                .alloc(cfg.vocab * 4, BufferUsage::Readback)
                .map_err(|e| anyhow!("{e}"))?,
            qwen_wide_buf: if cfg.qwen4exp {
                Some(
                    be.alloc(cfg.hc_mult * cfg.n_embd * 4, BufferUsage::Activations)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            },
            ple_embd_buf: if self.ple_embd_buf.is_some() {
                let heads = (cfg.ple_ngram_size - 1) * cfg.ple_heads_per_ngram;
                Some(
                    be.alloc(cfg.ple_head_dim * heads * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            },
            ple_state_buf,
            mtp_verify_bufs,
            max_ctx: self.max_ctx,
            kv_ring: self.kv_ring,
            cached: Vec::new(),
            // The forked slot's KV/weight buffers are new objects, so a cached plan's bindings
            // (which point at the OLD slot's buffers) don't carry over — rebuild lazily on this
            // slot's first denoise call. `self_cond_w`/`sc_embt` are model-derived (not
            // buffer-derived, and `sc_embt` lives on the SAME shared backend/device as `self`), so
            // they DO carry over (cheap Arc clone, skips a redundant dequant/rebuild).
            denoise_cache: None,
            self_cond_w: self.self_cond_w.clone(),
            sc_embt: self.sc_embt.clone(),
            // `sc_ping`'s buffers are per-slot device state (bound in the forked slot's own
            // graph executions), unlike the model-derived `self_cond_w`/`sc_embt` above — rebuild
            // lazily on this slot's first Vulkan denoise call, exactly like `denoise_cache`.
            sc_ping: None,
            sc_ping_write: 0,
            sc_temp_inv_buf: None,
            mtp_delta_ckpt,
            turn_recurrent_ckpts,
            preallocated_siblings: Vec::new(),
        })
    }

    /// Seed this slot's KV cache with the first `p` rows of `src`'s (the shared conversation
    /// prefix — e.g. the system prompt) via a device-side buffer copy, so the new conversation
    /// skips re-prefilling it. `p` must be ≤ src's materialized length.
    ///
    /// qwen35: a no-op. The gated-DeltaNet recurrent state is a single fixed-size summary of
    /// EVERY token fed so far — there's no "first `p` tokens' worth" of it to slice out and copy
    /// the way a real per-position KV cache allows (see `docs/qwen35.md` and the no-rewind rule in
    /// `generate_dense_backend`). Leaving `self.cached` empty (this slot's `fork()` already zeroed
    /// its state) is the CORRECT fallback: the next call on this slot fully re-prefills, exactly
    /// like the single-slot session's divergent-prompt reset.
    pub(crate) fn seed_from(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &crate::EngineConfig,
        src: &SeamKv,
        p: usize,
    ) -> AResult<usize> {
        if cfg.qwen35 || cfg.bailingmoe3 || cfg.glm5next || cfg.deepseek4 {
            return Ok(0);
        }
        if cfg.qwen4exp {
            return self.seed_qwen4_from(be, cfg, ec, src, p);
        }
        let p = p.min(src.cached.len()).min(self.max_ctx);
        if p == 0 {
            return Ok(0);
        }
        // SWA ring caches: positions [0, p) sit at rows [0, p) ONLY while the source hasn't
        // wrapped (cached_len <= ring rows) — a wrapped ring recycled exactly those early rows,
        // so the plain prefix copy below would seed stale data. Skipping the seed is the CORRECT
        // fallback (the slot just re-prefills the shared prefix); only cross-conversation KV
        // reuse on long conversations is lost. (Seeding a wrapped source's window TAIL would be
        // exact too, but needs two-segment copies per side per layer + a tail-only `cached`
        // semantics — deferred until serve traffic shows it matters.)
        if self.kv_ring {
            let wrapped = (0..cfg.n_layer)
                .filter(|&l| cfg.is_swa_layer(l))
                .map(|l| crate::seam::kv_rows(cfg, l, self.max_ctx, true, ec))
                .any(|rows_l| src.cached.len() > rows_l);
            if wrapped {
                return Ok(0);
            }
        }
        self.invalidate_delta_checkpoint();
        let mut copies = Vec::new();
        for l in 0..cfg.n_layer {
            // One prefix position is `k_row`/`v_row` elements per side — MLA's compressed row is
            // three times `n_kv * head_dim` wide and has no V side at all, so both widths come
            // from the shared `crate::seam::kv_row_elems` (docs/backlog.md B41).
            let (k_row, v_row) = crate::seam::kv_row_elems(cfg, l);
            copies.push((
                src.kbufs[l].as_ref(),
                self.kbufs[l].as_ref(),
                self.k_fmt,
                p * k_row,
            ));
            // `v_row == 0`: this arch caches no V (MLA) and `vbufs[l]` is the placeholder — there
            // is nothing to seed.
            if v_row > 0 {
                copies.push((
                    src.vbufs[l].as_ref(),
                    self.vbufs[l].as_ref(),
                    self.v_fmt,
                    p * v_row,
                ));
            }
        }
        be.copy_kv_prefixes(&copies).map_err(|e| anyhow!("{e}"))?;
        self.cached = src.cached[..p].to_vec();
        Ok(p)
    }

    /// Benchmark setup copies a complete live hybrid frontier, never an arbitrary rewind.
    /// Keep the normal serving policy (which rejects Qwen3.5 cross-slot seeds) unchanged.
    pub(crate) fn seed_benchmark_from(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &crate::EngineConfig,
        src: &SeamKv,
    ) -> AResult<usize> {
        let tokens = src.cached.len();
        if cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3 {
            return self.seed_qwen4_from(be, cfg, ec, src, tokens);
        }
        // Compressed and windowed caches carry more state than a linear token prefix.
        // Their non-segmented buffers can be copied whole at this exact live frontier.
        if cfg.deepseek4 || src.kv_ring {
            self.invalidate_delta_checkpoint();
            let copies = src
                .kbufs
                .iter()
                .chain(&src.vbufs)
                .zip(self.kbufs.iter().chain(&self.vbufs))
                .map(|(source, target)| (source.as_ref(), target.as_ref(), source.len_bytes()))
                .collect::<Vec<_>>();
            be.copy_buffers(&copies)
                .map_err(|e| anyhow!("benchmark live-state copy: {e}"))?;
            self.cached.clone_from(&src.cached);
            self.invalidate_turn_checkpoint();
            return Ok(tokens);
        }
        self.seed_from(be, cfg, ec, src, tokens)
    }

    fn seed_qwen4_from(
        &mut self,
        be: &dyn Backend,
        cfg: &Config,
        ec: &crate::EngineConfig,
        src: &SeamKv,
        requested: usize,
    ) -> AResult<usize> {
        let requested = requested.min(src.cached.len()).min(self.max_ctx);
        let live = (requested == src.cached.len() && requested != 0).then_some((requested, None));
        let checkpoint = src
            .turn_recurrent_ckpts
            .iter()
            .enumerate()
            .filter_map(|(index, checkpoint)| {
                let checkpoint = checkpoint.as_ref()?;
                (checkpoint.valid
                    && !checkpoint.tokens.is_empty()
                    && checkpoint.tokens.len() <= requested
                    && src.cached.starts_with(&checkpoint.tokens))
                .then_some((checkpoint.tokens.len(), Some(index)))
            })
            .max_by_key(|&(tokens, _)| tokens);
        let Some((tokens, recurrent_checkpoint)) = live
            .into_iter()
            .chain(checkpoint)
            .max_by_key(|&(tokens, _)| tokens)
        else {
            return Ok(0);
        };

        if self.kv_ring {
            let wrapped = (0..cfg.n_layer)
                .filter(|&layer| cfg.is_swa_layer(layer))
                .map(|layer| crate::seam::kv_rows(cfg, layer, self.max_ctx, true, ec))
                .any(|rows| src.cached.len() > rows);
            if wrapped {
                return Ok(0);
            }
        }

        self.invalidate_delta_checkpoint();
        self.ensure_segmented_depth(be, cfg, tokens)?;
        let mut copies: Vec<(&dyn Buffer, usize, &dyn Buffer, usize, usize)> = Vec::new();
        let mut kv_copies = Vec::new();
        for layer in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(layer) {
                let (src_k, src_v) = if let Some(index) = recurrent_checkpoint {
                    let checkpoint = src.turn_recurrent_ckpts[index]
                        .as_ref()
                        .expect("selected Qwen3.8 checkpoint exists");
                    let position = checkpoint
                        .layers
                        .iter()
                        .position(|&candidate| candidate == layer)
                        .ok_or_else(|| {
                            anyhow!("Qwen3.8 checkpoint has no recurrent layer {layer}")
                        })?;
                    (
                        checkpoint.kbufs[position].as_ref(),
                        checkpoint.vbufs[position].as_ref(),
                    )
                } else {
                    (src.kbufs[layer].as_ref(), src.vbufs[layer].as_ref())
                };
                copies.push((src_k, 0, self.kbufs[layer].as_ref(), 0, src_k.len_bytes()));
                copies.push((src_v, 0, self.vbufs[layer].as_ref(), 0, src_v.len_bytes()));
                continue;
            }

            let (k_row, v_row) = crate::seam::kv_row_elems(cfg, layer);
            if k_row != 0 {
                kv_copies.push((
                    src.kbufs[layer].as_ref(),
                    self.kbufs[layer].as_ref(),
                    self.k_fmt,
                    tokens * k_row,
                ));
            }
            if v_row != 0 {
                kv_copies.push((
                    src.vbufs[layer].as_ref(),
                    self.vbufs[layer].as_ref(),
                    self.v_fmt,
                    tokens * v_row,
                ));
            }
            if let (Some(src_raw), Some(dst_raw)) = (
                src.qsa_kbufs[layer].as_deref(),
                self.qsa_kbufs[layer].as_deref(),
            ) {
                let bytes = tokens
                    .saturating_mul(cfg.indexer_head_size)
                    .saturating_mul(2);
                if bytes != 0 {
                    copies.push((src_raw, 0, dst_raw, 0, bytes));
                }
            }
            if let (Some(src_blocks), Some(dst_blocks)) = (
                src.qsa_cbufs[layer].as_deref(),
                self.qsa_cbufs[layer].as_deref(),
            ) {
                let rows = tokens / cfg.layer_compress_ratio(layer).max(1);
                let bytes = rows.saturating_mul(cfg.indexer_head_size).saturating_mul(4);
                if bytes != 0 {
                    copies.push((src_blocks, 0, dst_blocks, 0, bytes));
                }
            }
        }
        if let (Some(src_ple), Some(dst_ple)) = (
            recurrent_checkpoint
                .and_then(|index| src.turn_recurrent_ckpts[index].as_ref())
                .and_then(|checkpoint| checkpoint.ple_state.as_deref())
                .or(src.ple_state_buf.as_deref()),
            self.ple_state_buf.as_deref(),
        ) {
            copies.push((src_ple, 0, dst_ple, 0, src_ple.len_bytes()));
        }
        be.copy_buffer_ranges(&copies)
            .map_err(|error| anyhow!("seed Qwen3.8 live state: {error}"))?;
        be.copy_kv_prefixes(&kv_copies)
            .map_err(|error| anyhow!("seed KV prefix planes: {error}"))?;
        drop(copies);

        for checkpoint in self.turn_recurrent_ckpts.iter_mut().flatten() {
            checkpoint.invalidate();
        }
        let selected_tokens = &src.cached[..tokens];
        for index in 0..super::TURN_CHECKPOINT_COUNT {
            let Some(source) = src.turn_recurrent_ckpts[index]
                .as_ref()
                .filter(|checkpoint| {
                    checkpoint.valid && selected_tokens.starts_with(&checkpoint.tokens)
                })
            else {
                continue;
            };
            if self.turn_recurrent_ckpts[index].is_none() {
                TurnRecurrentCkpt::begin(
                    &mut self.turn_recurrent_ckpts[index],
                    be,
                    cfg,
                    &self.kbufs,
                    &self.vbufs,
                    self.ple_state_buf.as_deref(),
                    &source.tokens,
                )?;
            }
            let target = self.turn_recurrent_ckpts[index]
                .as_mut()
                .expect("Qwen3.8 fork has recurrent checkpoint storage");
            let mut checkpoint_copies = Vec::with_capacity(
                source.layers.len() * 2 + usize::from(source.ple_state.is_some()),
            );
            for (position, &layer) in source.layers.iter().enumerate() {
                let target_position = target
                    .layers
                    .iter()
                    .position(|&candidate| candidate == layer)
                    .ok_or_else(|| anyhow!("Qwen3.8 seed target has no recurrent layer {layer}"))?;
                checkpoint_copies.push((
                    source.kbufs[position].as_ref(),
                    target.kbufs[target_position].as_ref(),
                    source.kbufs[position].len_bytes(),
                ));
                checkpoint_copies.push((
                    source.vbufs[position].as_ref(),
                    target.vbufs[target_position].as_ref(),
                    source.vbufs[position].len_bytes(),
                ));
            }
            if let (Some(source), Some(target)) =
                (source.ple_state.as_deref(), target.ple_state.as_deref())
            {
                checkpoint_copies.push((source, target, source.len_bytes()));
            }
            be.copy_buffers(&checkpoint_copies)
                .map_err(|error| anyhow!("seed Qwen3.8 turn checkpoint {index}: {error}"))?;
            target.tokens.clone_from(&source.tokens);
            target.copied.fill(true);
            target.ple_copied = true;
            target.valid = true;
        }
        self.cached.clear();
        self.cached.extend_from_slice(selected_tokens);
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checkpoint_extension_start, FfnW, MoeSharedW, MtpDeltaCkpt, SeamKv, SeamWeights,
        SegmentedKvState, SessionStable, TurnRecurrentCkpt,
    };
    use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Capabilities, Plan};
    use infr_core::error::{Error as BackendError, Result as BackendResult};
    use infr_core::graph::Graph;
    use infr_core::tensor::DType;
    use infr_core::tensor::TensorId;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingRollbackBackend {
        inner: infr_cpu::CpuBackend,
        allocations: AtomicUsize,
        allocated_bytes: AtomicUsize,
        fail_allocation_at: AtomicUsize,
        base_copies: AtomicUsize,
        row_copies: AtomicUsize,
        fail_snapshot: AtomicBool,
        fail_restore: AtomicBool,
    }

    impl CountingRollbackBackend {
        fn new() -> Self {
            Self {
                inner: infr_cpu::CpuBackend::new_with(Arc::new(crate::EngineConfig::default())),
                allocations: AtomicUsize::new(0),
                allocated_bytes: AtomicUsize::new(0),
                fail_allocation_at: AtomicUsize::new(0),
                base_copies: AtomicUsize::new(0),
                row_copies: AtomicUsize::new(0),
                fail_snapshot: AtomicBool::new(false),
                fail_restore: AtomicBool::new(false),
            }
        }

        fn counts(&self) -> (usize, usize, usize) {
            (
                self.allocations.load(Ordering::Relaxed),
                self.base_copies.load(Ordering::Relaxed),
                self.row_copies.load(Ordering::Relaxed),
            )
        }
    }

    impl Backend for CountingRollbackBackend {
        fn name(&self) -> &str {
            "counting-rollback-cpu"
        }
        fn capabilities(&self) -> Capabilities {
            self.inner.capabilities()
        }
        fn alloc(&self, bytes: usize, usage: BufferUsage) -> BackendResult<Box<dyn Buffer>> {
            let allocation = self.allocations.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_allocation_at.load(Ordering::Relaxed) == allocation {
                return Err(BackendError::backend(
                    "injected checkpoint allocation failure",
                ));
            }
            let buffer = self.inner.alloc(bytes, usage)?;
            self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
            Ok(buffer)
        }
        fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> BackendResult<()> {
            self.inner.upload(dst, src)
        }
        fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> BackendResult<()> {
            self.inner.download(src, dst)
        }
        fn copy_buffer(
            &self,
            src: &dyn Buffer,
            dst: &dyn Buffer,
            bytes: usize,
        ) -> BackendResult<()> {
            self.base_copies.fetch_add(1, Ordering::Relaxed);
            self.inner.copy_buffer(src, dst, bytes)
        }
        fn copy_buffers(&self, copies: &[(&dyn Buffer, &dyn Buffer, usize)]) -> BackendResult<()> {
            self.base_copies.fetch_add(1, Ordering::Relaxed);
            if self.fail_snapshot.swap(false, Ordering::Relaxed) {
                self.inner.copy_buffers(&copies[..1])?;
                return Err(BackendError::backend("injected partial snapshot failure"));
            }
            self.inner.copy_buffers(copies)
        }
        fn copy_buffer_ranges(
            &self,
            copies: &[(&dyn Buffer, usize, &dyn Buffer, usize, usize)],
        ) -> BackendResult<()> {
            self.row_copies.fetch_add(1, Ordering::Relaxed);
            if self.fail_restore.swap(false, Ordering::Relaxed) {
                self.inner.copy_buffer_ranges(&copies[..1])?;
                return Err(BackendError::backend(
                    "injected partial row restore failure",
                ));
            }
            self.inner.copy_buffer_ranges(copies)
        }
        fn compile(&self, graph: &Graph) -> BackendResult<Box<dyn Plan>> {
            self.inner.compile(graph)
        }
        fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> BackendResult<()> {
            self.inner.execute(plan, bindings)
        }
        fn sync(&self) -> BackendResult<()> {
            self.inner.sync()
        }
    }

    fn rollback_pattern(
        bytes: usize,
        salt: usize,
        layer: usize,
        row: usize,
        role: usize,
    ) -> Vec<u8> {
        let tag = 1000 * salt + 100 * layer + 7 * row + role;
        (0..bytes).map(|i| ((tag + 17 * i) % 251) as u8).collect()
    }

    fn rollback_buffer(be: &dyn Backend, bytes: usize, tag: usize) -> Box<dyn Buffer> {
        let buffer = be.alloc(bytes, BufferUsage::KvCache).unwrap();
        be.upload(buffer.as_ref(), &rollback_pattern(bytes, tag, 0, 0, 0))
            .unwrap();
        buffer
    }

    fn rollback_read(be: &dyn Backend, buffer: &dyn Buffer) -> Vec<u8> {
        let mut bytes = vec![0; buffer.len_bytes()];
        be.download(buffer, &mut bytes).unwrap();
        bytes
    }

    // Minimal live slot for the production snapshot/restore methods. No graph or model weights.
    fn rollback_slot(
        be: &dyn Backend,
        gdn: bool,
        ple: bool,
        salt: usize,
    ) -> (crate::Config, SeamKv) {
        rollback_slot_with_base(be, gdn, ple, salt, true)
    }

    fn lookup_rollback_slot(
        be: &dyn Backend,
        gdn: bool,
        ple: bool,
        salt: usize,
    ) -> (crate::Config, SeamKv) {
        rollback_slot_with_base(be, gdn, ple, salt, false)
    }

    fn rollback_slot_with_base(
        be: &dyn Backend,
        gdn: bool,
        ple: bool,
        salt: usize,
        eager_base: bool,
    ) -> (crate::Config, SeamKv) {
        let cfg = crate::Config {
            qwen4exp: gdn || ple,
            n_layer: 4,
            n_layer_kv_from_start: 4,
            n_embd: 32,
            hc_mult: 1,
            n_head: 1,
            n_kv: 1,
            head_dim: 32,
            recurrent_layers: vec![gdn, false, gdn, false],
            ple_layers: vec![false, ple, false, false],
            ple_ngram_size: 2,
            ple_conv_kernel: 2,
            ple_heads_per_ngram: 1,
            ple_head_dim: 4,
            ssm_d_conv: 3,
            ssm_d_state: 128,
            ssm_d_inner: 2,
            ssm_n_group: 1,
            ssm_dt_rank: 1,
            ..Default::default()
        };
        let kbufs = (0..cfg.n_layer)
            .map(|l| {
                let words = if cfg.is_recurrent_layer(l) {
                    (cfg.ssm_d_conv - 1) * cfg.recurrent_conv_channels()
                } else {
                    16 * cfg.n_kv * cfg.head_dim
                };
                rollback_buffer(be, words * 4, salt + l)
            })
            .collect::<Vec<_>>();
        let vbufs = (0..cfg.n_layer)
            .map(|l| {
                let words = if cfg.is_recurrent_layer(l) {
                    cfg.recurrent_state_elems()
                } else {
                    16 * cfg.n_kv * cfg.head_dim
                };
                rollback_buffer(be, words * 4, salt + l + 4)
            })
            .collect::<Vec<_>>();
        let ple_state = ple.then(|| rollback_buffer(be, 2 * cfg.n_embd * 4, salt + 8));
        let checkpoint = MtpDeltaCkpt::allocate_for_fork(
            be,
            &cfg,
            &kbufs,
            &vbufs,
            ple_state.as_deref(),
            3,
            eager_base,
        )
        .unwrap();
        let slot = SeamKv {
            weights: Arc::new(SeamWeights {
                wbufs: Vec::new(),
                wspecs: Vec::new(),
                rf_buf: None,
                yff_buf: None,
                layer_has_epb: Vec::new(),
                layer_fused_experts: Vec::new(),
                ple_worker: None,
                mtp_token_embd_index: 0,
                mtp_lm_head_index: 0,
            }),
            stable: Arc::new(SessionStable {
                has_wv: Vec::new(),
                out_scale: Vec::new(),
                dec_out_scale: Vec::new(),
                rope_freqs: None,
                yarn_ff: None,
                fuse_gu: false,
                fuse_qkv: false,
                fuse_qsa_prepare: false,
                fuse_delta_ab: false,
                fuse_qwen_hc_down_inject: false,
                kda_qkv_fused: Vec::new(),
                moe_batched_ok: false,
            }),
            kbufs,
            vbufs,
            // Sentinel planes prove rollback does not overwrite append-only QSA storage.
            qsa_kbufs: (0..4)
                .map(|l| Some(rollback_buffer(be, 96, salt + l + 10)))
                .collect(),
            qsa_cbufs: (0..4)
                .map(|l| Some(rollback_buffer(be, 48, salt + l + 14)))
                .collect(),
            segmented_kv: SegmentedKvState::default(),
            k_fmt: DType::F32,
            v_fmt: DType::F32,
            hidden_buf: rollback_buffer(be, 4, 20),
            pos_buf: rollback_buffer(be, 4, 21),
            ipl_buf: None,
            logits_buf: rollback_buffer(be, 4, 22),
            qwen_wide_buf: None,
            ple_embd_buf: None,
            ple_state_buf: ple_state,
            mtp_verify_bufs: None,
            max_ctx: 16,
            kv_ring: false,
            cached: vec![10, 20, 30, 40],
            denoise_cache: None,
            self_cond_w: None,
            sc_embt: None,
            sc_ping: None,
            sc_ping_write: 0,
            sc_temp_inv_buf: None,
            mtp_delta_ckpt: checkpoint,
            turn_recurrent_ckpts: std::array::from_fn(|_| None),
            preallocated_siblings: Vec::new(),
        };
        (cfg, slot)
    }

    fn rollback_state(be: &dyn Backend, slot: &SeamKv) -> Vec<Vec<u8>> {
        slot.kbufs
            .iter()
            .chain(&slot.vbufs)
            .chain(slot.qsa_kbufs.iter().flatten())
            .chain(slot.qsa_cbufs.iter().flatten())
            .map(|buffer| rollback_read(be, buffer.as_ref()))
            .chain(
                slot.ple_state_buf
                    .iter()
                    .map(|buffer| rollback_read(be, buffer.as_ref())),
            )
            .collect()
    }

    // CPU execution ignores state_trace outputs. Stage synthetic writer results here to test
    // real backend copies/offsets/slot isolation, without claiming GPU trace-writer parity.
    fn stage_rollback_rows(be: &dyn Backend, slot: &mut SeamKv, width: usize, salt: usize) {
        slot.mtp_arm_delta_trace(width).unwrap();
        stage_armed_rollback_rows(be, slot, width, salt);
    }

    fn stage_armed_rollback_rows(be: &dyn Backend, slot: &mut SeamKv, width: usize, salt: usize) {
        slot.mtp_validate_delta_trace(width).unwrap();
        assert_eq!(slot.mtp_take_delta_trace(), Some((width, width - 1)));
        let checkpoint = slot.mtp_delta_ckpt.as_ref().unwrap();
        for (ordinal, &layer) in checkpoint.layers.iter().enumerate() {
            for (live, trace, role) in [
                (
                    slot.kbufs[layer].as_ref(),
                    checkpoint.trace_kbufs[ordinal].as_ref(),
                    1,
                ),
                (
                    slot.vbufs[layer].as_ref(),
                    checkpoint.trace_vbufs[ordinal].as_ref(),
                    2,
                ),
            ] {
                let bytes = live.len_bytes();
                let mut rows = vec![0xDD; trace.len_bytes()];
                for row in 1..width {
                    rows[(row - 1) * bytes..row * bytes]
                        .copy_from_slice(&rollback_pattern(bytes, salt, layer, row, role));
                }
                be.upload(trace, &rows).unwrap();
                be.upload(live, &rollback_pattern(bytes, salt, layer, width, role))
                    .unwrap();
            }
        }
        if let (Some(live), Some(trace)) = (
            slot.ple_state_buf.as_deref(),
            checkpoint.trace_ple_state.as_deref(),
        ) {
            let bytes = live.len_bytes();
            let mut rows = vec![0xDD; trace.len_bytes()];
            for row in 1..width {
                rows[(row - 1) * bytes..row * bytes]
                    .copy_from_slice(&rollback_pattern(bytes, salt, 4, row, 3));
            }
            be.upload(trace, &rows).unwrap();
            be.upload(live, &rollback_pattern(bytes, salt, 4, width, 3))
                .unwrap();
        }
        slot.cached.extend((0..width).map(|row| 100 + row as u32));
    }

    fn rollback_base_bytes(cfg: &crate::Config, slot: &SeamKv) -> usize {
        (0..cfg.n_layer)
            .filter(|&layer| cfg.is_recurrent_layer(layer))
            .map(|layer| slot.kbufs[layer].len_bytes() + slot.vbufs[layer].len_bytes())
            .sum::<usize>()
            + slot.ple_state_buf.as_deref().map_or(0, Buffer::len_bytes)
    }

    #[test]
    fn cpu_lookup_checkpoint_startup_allocates_three_copies_mtp_four() {
        for (gdn, ple) in [(false, false), (true, false), (false, true), (true, true)] {
            let (cfg, source) = rollback_slot(&infr_cpu::CpuBackend::new(), gdn, ple, 1);
            let base_bytes = rollback_base_bytes(&cfg, &source);
            let kbufs: Vec<_> = source.kbufs.into_iter().map(Some).collect();
            let vbufs: Vec<_> = source.vbufs.into_iter().map(Some).collect();
            for eager_base in [false, true] {
                let be = CountingRollbackBackend::new();
                let checkpoint = MtpDeltaCkpt::allocate_before_dynamic_kv(
                    &be,
                    &cfg,
                    &kbufs,
                    &vbufs,
                    source.ple_state_buf.as_deref(),
                    3,
                    eager_base,
                )
                .unwrap()
                .unwrap();
                assert_eq!(
                    be.allocated_bytes.load(Ordering::Relaxed),
                    base_bytes * (3 + usize::from(eager_base))
                );
                assert_eq!(checkpoint.base_allocated, eager_base);
                assert_eq!(checkpoint.eager_base, eager_base);
                assert!(!checkpoint.clean_base_valid);
                assert_eq!(checkpoint.trace_rows, 3);
                if !eager_base {
                    assert!(checkpoint.kbufs.is_empty() && checkpoint.vbufs.is_empty());
                    assert!(checkpoint.ple_state.is_none());
                }
                assert_eq!((be.counts().1, be.counts().2), (0, 0));
            }
        }
    }

    #[test]
    fn cpu_lookup_lazy_base_snapshot_allocates_once_and_preserves_zero_restore() {
        for (gdn, ple) in [(false, false), (true, false), (false, true), (true, true)] {
            for eager_base in [false, true] {
                let be = CountingRollbackBackend::new();
                let (cfg, mut slot) = rollback_slot_with_base(&be, gdn, ple, 1, eager_base);
                let base_bytes = rollback_base_bytes(&cfg, &slot);
                let bytes = be.allocated_bytes.load(Ordering::Relaxed);
                let tokens = slot.cached.clone();
                let state = rollback_state(&be, &slot);
                assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
                slot.mtp_snapshot_delta(&be, &cfg).unwrap();
                assert_eq!(
                    be.allocated_bytes.load(Ordering::Relaxed) - bytes,
                    if eager_base { 0 } else { base_bytes }
                );
                let checkpoint = slot.mtp_delta_ckpt.as_ref().unwrap();
                assert!(checkpoint.base_allocated && checkpoint.clean_base_valid);
                assert_eq!(checkpoint.eager_base, eager_base);
                let allocations = be.counts().0;
                slot.mtp_snapshot_delta(&be, &cfg).unwrap();
                stage_rollback_rows(&be, &mut slot, 4, 2);
                slot.mtp_restore_delta_row(&be, 0).unwrap();
                assert_eq!(slot.cached, tokens);
                assert_eq!(rollback_state(&be, &slot), state);
                assert_eq!(be.counts().0, allocations);
            }
        }
    }

    #[test]
    fn cpu_lookup_lazy_base_allocation_failure_is_atomic_and_retryable() {
        // Two recurrent layers own K/S pairs; PLE owns the fifth base buffer.
        for failed_buffer in 1..=5 {
            let be = CountingRollbackBackend::new();
            let (cfg, mut slot) = lookup_rollback_slot(&be, true, true, 1);
            let state = rollback_state(&be, &slot);
            let tokens = slot.cached.clone();
            be.fail_allocation_at
                .store(be.counts().0 + failed_buffer, Ordering::Relaxed);
            assert!(slot.mtp_snapshot_delta(&be, &cfg).is_err());
            let checkpoint = slot.mtp_delta_ckpt.as_ref().unwrap();
            assert!(!checkpoint.base_allocated && !checkpoint.clean_base_valid);
            assert!(checkpoint.kbufs.is_empty() && checkpoint.vbufs.is_empty());
            assert!(checkpoint.ple_state.is_none());
            assert_eq!(checkpoint.trace_rows, 3);
            let counts = be.counts();
            assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
            assert_eq!(be.counts(), counts);
            assert_eq!(slot.cached, tokens);
            assert_eq!(rollback_state(&be, &slot), state);

            let bytes = be.allocated_bytes.load(Ordering::Relaxed);
            slot.mtp_snapshot_delta(&be, &cfg).unwrap();
            assert_eq!(
                be.allocated_bytes.load(Ordering::Relaxed) - bytes,
                rollback_base_bytes(&cfg, &slot)
            );
            stage_rollback_rows(&be, &mut slot, 3, 2);
            slot.mtp_restore_delta_row(&be, 0).unwrap();
            assert_eq!(slot.cached, tokens);
            assert_eq!(rollback_state(&be, &slot), state);
        }
    }

    #[test]
    fn cpu_lookup_lazy_base_copy_failure_reuses_buffers_on_retry() {
        let be = CountingRollbackBackend::new();
        let (cfg, mut slot) = lookup_rollback_slot(&be, true, true, 1);
        let state = rollback_state(&be, &slot);
        let tokens = slot.cached.clone();
        let bytes = be.allocated_bytes.load(Ordering::Relaxed);
        be.fail_snapshot.store(true, Ordering::Relaxed);
        assert!(slot.mtp_snapshot_delta(&be, &cfg).is_err());
        assert_eq!(
            be.allocated_bytes.load(Ordering::Relaxed) - bytes,
            rollback_base_bytes(&cfg, &slot)
        );
        let checkpoint = slot.mtp_delta_ckpt.as_ref().unwrap();
        assert!(checkpoint.base_allocated && !checkpoint.clean_base_valid);
        let counts = be.counts();
        assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
        assert_eq!(be.counts(), counts);
        assert_eq!(slot.cached, tokens);
        assert_eq!(rollback_state(&be, &slot), state);

        slot.mtp_snapshot_delta(&be, &cfg).unwrap();
        assert_eq!(be.counts().0, counts.0);
        stage_rollback_rows(&be, &mut slot, 4, 2);
        slot.mtp_restore_delta_row(&be, 0).unwrap();
        assert_eq!(slot.cached, tokens);
        assert_eq!(rollback_state(&be, &slot), state);
    }

    #[test]
    fn cpu_lookup_forks_keep_startup_policy_after_lazy_handoff() {
        for eager_base in [false, true] {
            let be = CountingRollbackBackend::new();
            let (mut cfg, mut slot) = rollback_slot_with_base(&be, true, true, 1, eager_base);
            cfg.vocab = 32;
            // The rollback fixture has artificial QSA sentinels, but this tiny model declares
            // no index-key planes. Real fork geometry checks must see that same layout.
            slot.qsa_kbufs = (0..cfg.n_layer).map(|_| None).collect();
            slot.qsa_cbufs = (0..cfg.n_layer).map(|_| None).collect();
            for snapshot_first in [false, true] {
                if snapshot_first {
                    slot.mtp_snapshot_delta(&be, &cfg).unwrap();
                }
                let sibling = slot
                    .fork(&be, &cfg, &crate::EngineConfig::default())
                    .unwrap();
                let checkpoint = sibling.mtp_delta_ckpt.as_ref().unwrap();
                assert_eq!(checkpoint.eager_base, eager_base);
                assert_eq!(checkpoint.base_allocated, eager_base);
                assert!(!checkpoint.clean_base_valid);
                assert_eq!(checkpoint.trace_rows, 3);
                let base_bytes = checkpoint
                    .kbufs
                    .iter()
                    .chain(&checkpoint.vbufs)
                    .map(|buffer| buffer.len_bytes())
                    .sum::<usize>()
                    + checkpoint.ple_state.as_deref().map_or(0, Buffer::len_bytes);
                assert_eq!(
                    base_bytes,
                    if eager_base {
                        rollback_base_bytes(&cfg, &sibling)
                    } else {
                        0
                    }
                );
                assert!(sibling.cached.is_empty());
            }
        }
    }

    #[test]
    fn cpu_lookup_flashnext_unused_bases_are_236077056_bytes_for_two_slots() {
        let cfg = crate::Config {
            qwen4exp: true,
            n_layer: 48,
            n_embd: 2560,
            hc_mult: 4,
            recurrent_layers: (0..48).map(|layer| (layer + 1) % 4 != 0).collect(),
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_d_inner: 6144,
            ssm_n_group: 16,
            ssm_dt_rank: 48,
            ple_conv_kernel: 4,
            ple_ngram_size: 3,
            ..Default::default()
        };
        assert_eq!(
            crate::seam::recurrent_checkpoint_bytes(&cfg) * 2,
            236_077_056
        );
    }

    #[test]
    fn cpu_lookup_trace_arming_and_positive_settle_never_copy_a_base() {
        for (gdn, ple) in [(false, false), (true, false), (false, true), (true, true)] {
            for width in 2..=4 {
                for kept in 1..=width {
                    let be = CountingRollbackBackend::new();
                    let (cfg, mut slot) = lookup_rollback_slot(&be, gdn, ple, 1);
                    let initial = rollback_state(&be, &slot);
                    let counts = be.counts();
                    assert_eq!((counts.1, counts.2), (0, 0));
                    assert!(slot.mtp_restore_delta(&be).is_err());
                    slot.lookup_arm_delta_trace(4, width).unwrap();
                    assert_eq!(be.counts(), counts, "arming must not copy or allocate");
                    assert_eq!(rollback_state(&be, &slot), initial);
                    assert_eq!(slot.cached, [10, 20, 30, 40]);
                    assert!(slot.mtp_restore_delta(&be).is_err());
                    assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
                    assert!(slot.mtp_restore_delta_row(&be, 1).is_err());
                    assert!(slot.lookup_settle_delta_trace(&be, 1).is_err());
                    assert!(slot.mtp_arm_delta_trace(width).is_err());
                    assert!(slot.mtp_snapshot_delta(&be, &cfg).is_err());
                    assert_eq!(be.counts(), counts);

                    stage_armed_rollback_rows(&be, &mut slot, width, 1);
                    let complete = rollback_state(&be, &slot);
                    let complete_tokens = slot.cached.clone();
                    assert!(slot.lookup_settle_delta_trace(&be, 0).is_err());
                    assert!(slot.lookup_settle_delta_trace(&be, width + 1).is_err());
                    assert!(slot.mtp_restore_delta_row(&be, width).is_err());
                    assert!(slot.mtp_snapshot_delta(&be, &cfg).is_err());
                    assert_eq!(be.counts(), counts);
                    assert_eq!(rollback_state(&be, &slot), complete);
                    assert_eq!(slot.cached, complete_tokens);
                    slot.lookup_settle_delta_trace(&be, kept).unwrap();
                    assert_eq!(
                        be.counts(),
                        (counts.0, 0, usize::from(kept < width && (gdn || ple)))
                    );
                    assert_eq!(slot.cached, complete_tokens[..4 + kept]);
                    for layer in 0..4 {
                        for (buffer, role, base_index) in [
                            (slot.kbufs[layer].as_ref(), 1, layer),
                            (slot.vbufs[layer].as_ref(), 2, 4 + layer),
                        ] {
                            let expected = if cfg.is_recurrent_layer(layer) {
                                rollback_pattern(buffer.len_bytes(), 1, layer, kept, role)
                            } else {
                                initial[base_index].clone()
                            };
                            assert_eq!(rollback_read(&be, buffer), expected);
                        }
                    }
                    let settled = rollback_state(&be, &slot);
                    assert_eq!(settled[8..16], initial[8..16]);
                    if let Some(buffer) = slot.ple_state_buf.as_deref() {
                        assert_eq!(
                            rollback_read(&be, buffer),
                            rollback_pattern(buffer.len_bytes(), 1, 4, kept, 3)
                        );
                    }
                    let counts = be.counts();
                    assert!(slot.mtp_restore_delta(&be).is_err());
                    assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
                    assert!(slot.mtp_restore_delta_row(&be, 1).is_err());
                    assert!(slot.lookup_settle_delta_trace(&be, kept).is_err());
                    assert_eq!(be.counts(), counts);
                    assert_eq!(rollback_state(&be, &slot), settled);
                }
            }
        }
    }

    #[test]
    fn cpu_lookup_trace_retries_and_rebases_after_cycles_ordinary_and_context_reset() {
        let be = CountingRollbackBackend::new();
        let (cfg, mut slot) = lookup_rollback_slot(&be, true, true, 1);
        slot.max_ctx = 32;
        let counts = be.counts();
        assert!(slot.lookup_arm_delta_trace(3, 4).is_err());
        assert!(slot.lookup_arm_delta_trace(4, 1).is_err());
        assert!(slot.lookup_arm_delta_trace(4, 5).is_err());
        slot.max_ctx = 5;
        assert!(slot.lookup_arm_delta_trace(4, 2).is_err());
        slot.max_ctx = 32;
        assert_eq!(be.counts(), counts);
        let mut row_copies = 0;
        for (width, kept) in [(4, 4), (3, 1), (2, 2)] {
            let base = slot.cached_len();
            slot.lookup_arm_delta_trace(base, width).unwrap();
            assert!(slot.lookup_arm_delta_trace(base, width).is_err());
            assert_eq!(slot.mtp_delta_ckpt.as_ref().unwrap().cached_len, base);
            stage_armed_rollback_rows(&be, &mut slot, width, 1);
            // A consumed trace alone is not completion. A short frontier must be rejected,
            // without poisoning the valid transaction's later retry at its complete frontier.
            let last = slot.cached.pop().unwrap();
            assert!(slot.lookup_settle_delta_trace(&be, kept).is_err());
            slot.cached.push(last);
            slot.lookup_settle_delta_trace(&be, kept).unwrap();
            row_copies += usize::from(kept < width);
            assert_eq!(slot.cached_len(), base + kept);
            assert_eq!(be.counts(), (counts.0, 0, row_copies));
            assert!(slot.lookup_arm_delta_trace(base, width).is_err());
        }

        // Ordinary execution has advanced the real live state, with no retained clean snapshot.
        slot.cached.push(777);
        let ordinary_base = slot.cached_len();
        slot.lookup_arm_delta_trace(ordinary_base, 4).unwrap();
        stage_armed_rollback_rows(&be, &mut slot, 4, 2);
        slot.lookup_settle_delta_trace(&be, 1).unwrap(); // cancelled delivery retains pending
        assert_eq!(slot.cached_len(), ordinary_base + 1);
        assert_eq!(be.counts(), (counts.0, 0, row_copies + 1));

        slot.set_synthetic_cached(&be, &cfg, vec![5; 4]).unwrap();
        assert!(slot.lookup_settle_delta_trace(&be, 1).is_err());
        slot.lookup_arm_delta_trace(4, 2).unwrap();
        stage_armed_rollback_rows(&be, &mut slot, 2, 3);
        slot.lookup_settle_delta_trace(&be, 2).unwrap();
        slot.reset();
        assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
        assert!(slot.lookup_settle_delta_trace(&be, 1).is_err());
        slot.set_synthetic_cached(&be, &cfg, vec![6; 2]).unwrap();
        assert!(slot.lookup_arm_delta_trace(4, 2).is_err());
        slot.lookup_arm_delta_trace(2, 2).unwrap();
        stage_armed_rollback_rows(&be, &mut slot, 2, 4);
        slot.lookup_settle_delta_trace(&be, 2).unwrap();
        assert_eq!(be.counts(), (counts.0, 0, row_copies + 1));
    }

    #[test]
    fn cpu_lookup_handoff_requires_real_snapshot_and_preserves_mtp_zero_restore() {
        let be = CountingRollbackBackend::new();
        let (cfg, mut slot) = lookup_rollback_slot(&be, true, true, 1);
        slot.mtp_snapshot_delta(&be, &cfg).unwrap();
        let counts = be.counts();
        assert_eq!((counts.1, counts.2), (1, 0));
        slot.lookup_arm_delta_trace(4, 4).unwrap();
        assert_eq!(be.counts(), counts);
        assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
        stage_armed_rollback_rows(&be, &mut slot, 4, 2);
        slot.lookup_settle_delta_trace(&be, 1).unwrap();
        assert!(slot.mtp_restore_delta(&be).is_err());
        assert!(slot.mtp_arm_delta_trace(4).is_err());
        assert_eq!(be.counts(), (counts.0, 1, 1));

        let tokens = slot.cached.clone();
        let state = rollback_state(&be, &slot);
        slot.mtp_snapshot_delta(&be, &cfg).unwrap();
        stage_rollback_rows(&be, &mut slot, 4, 3);
        slot.mtp_restore_delta_row(&be, 0).unwrap();
        assert_eq!(slot.cached, tokens);
        assert_eq!(rollback_state(&be, &slot), state);
        assert_eq!(be.counts(), (counts.0, 3, 1));
        slot.mtp_restore_delta(&be).unwrap(); // an actual clean snapshot remains reusable
        assert_eq!(slot.cached, tokens);
        assert_eq!(be.counts(), (counts.0, 4, 1));
    }

    #[test]
    fn cpu_lookup_failed_verify_or_partial_restore_invalidates_frontiers() {
        let be = CountingRollbackBackend::new();
        let (cfg, mut slot) = lookup_rollback_slot(&be, true, true, 1);
        slot.lookup_arm_delta_trace(4, 4).unwrap();
        slot.mtp_take_delta_trace().unwrap();
        slot.cached.push(100); // failed Verify may have materialized only some inputs
        assert!(slot.lookup_settle_delta_trace(&be, 1).is_err());
        slot.reset(); // same reset as try_lookup_cycle's error path
        assert_eq!((be.counts().1, be.counts().2), (0, 0));
        assert!(slot.lookup_settle_delta_trace(&be, 1).is_err());
        assert!(slot.mtp_restore_delta(&be).is_err());

        slot.set_synthetic_cached(&be, &cfg, vec![7; 4]).unwrap();
        slot.lookup_arm_delta_trace(4, 4).unwrap();
        stage_armed_rollback_rows(&be, &mut slot, 4, 2);
        be.fail_restore.store(true, Ordering::Relaxed);
        assert!(slot.lookup_settle_delta_trace(&be, 1).is_err());
        assert_eq!(slot.cached_len(), 0);
        assert_eq!(slot.mtp_take_delta_trace(), None);
        let counts = be.counts();
        assert_eq!((counts.1, counts.2), (0, 1));
        assert!(slot.lookup_settle_delta_trace(&be, 1).is_err());
        assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
        assert!(slot.mtp_restore_delta_row(&be, 1).is_err());
        assert_eq!(be.counts(), counts);

        slot.set_synthetic_cached(&be, &cfg, vec![8; 3]).unwrap();
        slot.lookup_arm_delta_trace(3, 3).unwrap();
        stage_armed_rollback_rows(&be, &mut slot, 3, 4);
        slot.lookup_settle_delta_trace(&be, 2).unwrap();
        assert_eq!(slot.cached_len(), 5);
        assert_eq!((be.counts().1, be.counts().2), (0, 2));
    }

    #[test]
    fn cpu_lookup_failed_snapshot_cannot_reenable_zero_restore() {
        let be = CountingRollbackBackend::new();
        let (cfg, mut slot) = lookup_rollback_slot(&be, true, true, 1);
        slot.mtp_snapshot_delta(&be, &cfg).unwrap();
        slot.lookup_arm_delta_trace(4, 2).unwrap();
        stage_armed_rollback_rows(&be, &mut slot, 2, 2);
        slot.lookup_settle_delta_trace(&be, 2).unwrap();
        be.fail_snapshot.store(true, Ordering::Relaxed);
        assert!(slot.mtp_snapshot_delta(&be, &cfg).is_err());
        let counts = be.counts();
        let state = rollback_state(&be, &slot);
        let tokens = slot.cached.clone();
        assert!(slot.mtp_restore_delta(&be).is_err());
        assert!(slot.mtp_restore_delta_row(&be, 0).is_err());
        assert_eq!(be.counts(), counts);
        assert_eq!(rollback_state(&be, &slot), state);
        assert_eq!(slot.cached, tokens);
        slot.mtp_snapshot_delta(&be, &cfg).unwrap();
        stage_rollback_rows(&be, &mut slot, 3, 3);
        slot.mtp_restore_delta_row(&be, 0).unwrap();
        assert_eq!(rollback_state(&be, &slot), state);
        assert_eq!(slot.cached, tokens);
    }

    #[test]
    fn cpu_verify_row_rollback_restores_bytes_for_every_width_prefix_and_participant() {
        let be = infr_cpu::CpuBackend::new_with(Arc::new(crate::EngineConfig::default()));
        for (gdn, ple) in [(false, false), (true, false), (false, true), (true, true)] {
            for width in 2..=4 {
                for kept in 0..=width {
                    let (cfg, mut slot) = rollback_slot(&be, gdn, ple, 1);
                    let (_, sibling) = rollback_slot(&be, gdn, ple, 9);
                    let sibling_before = rollback_state(&be, &sibling);
                    let base = rollback_state(&be, &slot);
                    slot.mtp_snapshot_delta(&be, &cfg).unwrap();
                    stage_rollback_rows(&be, &mut slot, width, 1);
                    if kept < width {
                        slot.mtp_restore_delta_row(&be, kept).unwrap();
                    }
                    assert_eq!(
                        slot.cached,
                        [10, 20, 30, 40]
                            .into_iter()
                            .chain((0..kept).map(|row| 100 + row as u32))
                            .collect::<Vec<_>>()
                    );
                    for layer in 0..4 {
                        for (buffer, role, base_index) in [
                            (slot.kbufs[layer].as_ref(), 1, layer),
                            (slot.vbufs[layer].as_ref(), 2, 4 + layer),
                        ] {
                            let expected = if cfg.is_recurrent_layer(layer) && kept > 0 {
                                rollback_pattern(buffer.len_bytes(), 1, layer, kept, role)
                            } else {
                                base[base_index].clone()
                            };
                            assert_eq!(rollback_read(&be, buffer), expected,
                                "gdn={gdn} ple={ple} width={width} kept={kept} layer={layer} role={role}");
                        }
                    }
                    let after = rollback_state(&be, &slot);
                    assert_eq!(
                        after[8..16],
                        base[8..16],
                        "QSA planes must stay append-only"
                    );
                    if let Some(buffer) = slot.ple_state_buf.as_deref() {
                        let expected = if kept == 0 {
                            base[16].clone()
                        } else {
                            rollback_pattern(buffer.len_bytes(), 1, 4, kept, 3)
                        };
                        assert_eq!(rollback_read(&be, buffer), expected);
                    }
                    let before_error = rollback_state(&be, &slot);
                    let tokens_before_error = slot.cached.clone();
                    assert!(slot.mtp_restore_delta_row(&be, 5).is_err());
                    assert_eq!(rollback_state(&be, &slot), before_error);
                    assert_eq!(slot.cached, tokens_before_error);

                    // The next clean checkpoint must describe the retained prefix, not the
                    // original base or rejected final live state.
                    slot.mtp_snapshot_delta(&be, &cfg).unwrap();
                    for &layer in &slot.mtp_delta_ckpt.as_ref().unwrap().layers {
                        for buffer in [&slot.kbufs[layer], &slot.vbufs[layer]] {
                            be.upload(buffer.as_ref(), &vec![0xEE; buffer.len_bytes()])
                                .unwrap();
                        }
                    }
                    if let Some(buffer) = slot.ple_state_buf.as_deref() {
                        be.upload(buffer, &vec![0xEE; buffer.len_bytes()]).unwrap();
                    }
                    slot.cached.push(999);
                    slot.mtp_restore_delta(&be).unwrap();
                    assert_eq!(slot.cached, tokens_before_error);
                    assert_eq!(rollback_state(&be, &slot), after);
                    assert_eq!(rollback_state(&be, &sibling), sibling_before);
                    assert_eq!(sibling.cached, [10, 20, 30, 40]);
                }
            }
        }
    }

    #[test]
    fn attention_checkpoint_allocation_needs_no_device_state_or_source_buffers() {
        let cfg = crate::Config {
            n_layer: 2,
            n_layer_kv_from_start: 2,
            n_embd: 256,
            ..Default::default()
        };
        let checkpoint = MtpDeltaCkpt::allocate_sized(
            &infr_cpu::CpuBackend::new(),
            &cfg,
            |_| panic!("append-only attention must not query recurrent K state"),
            |_| panic!("append-only attention must not query recurrent V state"),
            None,
            3,
            true,
        )
        .unwrap()
        .expect("attention Verify still owns a frontier checkpoint");
        assert!(checkpoint.layers.is_empty());
        assert!(checkpoint.kbufs.is_empty() && checkpoint.vbufs.is_empty());
        assert!(checkpoint.trace_kbufs.is_empty() && checkpoint.trace_vbufs.is_empty());
        assert!(checkpoint.ple_state.is_none() && checkpoint.trace_ple_state.is_none());
        assert_eq!(checkpoint.cached_len, 0);
        assert_eq!(checkpoint.trace_rows, 3);
    }

    #[test]
    fn empty_checkpoint_tracks_arming_and_every_retained_frontier_without_buffers() {
        for width in 2..=4 {
            let mut checkpoint = MtpDeltaCkpt::empty(3, true);
            checkpoint.cached_len = 100;
            checkpoint.arm(width).unwrap();
            checkpoint.validate(width, 100).unwrap();
            assert!(checkpoint.validate(width, 101).is_err());
            for other_width in 2..=4 {
                assert_eq!(
                    checkpoint.validate(other_width, 100).is_ok(),
                    other_width == width
                );
            }
            assert_eq!(checkpoint.take_trace(), Some((width, width - 1)));
            assert_eq!(checkpoint.take_trace(), None);
            assert!(checkpoint.validate(width, 100).is_err());
            for kept in 0..width {
                let mut tokens = vec![0u32; 100 + width];
                tokens.truncate(checkpoint.restored_len(kept).unwrap());
                assert_eq!(tokens.len(), 100 + kept);
            }
            assert!(checkpoint.restored_len(4).is_err());
            assert!(checkpoint.arm(0).is_err());
            assert!(checkpoint.arm(5).is_err());
        }
        let mut checkpoint = MtpDeltaCkpt::empty(3, true);
        checkpoint.cached_len = usize::MAX;
        assert!(checkpoint.restored_len(1).is_err());
    }

    fn moe_ffn(shexp: Option<MoeSharedW>) -> FfnW {
        let t = TensorId(0);
        FfnW::Moe {
            router: t,
            gate_exps: t,
            up_exps: t,
            down_exps: t,
            fused_gate_up: true,
            shexp,
            exp_probs_b: None,
            tid2eid: None,
        }
    }

    #[test]
    fn shared_expert_requires_separate_dense_and_routed_outputs() {
        assert!(!moe_ffn(None).has_shared_expert());
        assert!(moe_ffn(Some(MoeSharedW {
            gate_inp: Some(TensorId(1)),
            wgate: TensorId(2),
            wup: TensorId(3),
            wdown: TensorId(4),
        }))
        .has_shared_expert());
    }

    #[test]
    fn recurrent_checkpoint_requires_a_nonempty_strict_extension() {
        assert_eq!(
            checkpoint_extension_start(&[10, 20], &[10, 20, 30]),
            Some(2)
        );
        assert_eq!(checkpoint_extension_start(&[], &[10]), None);
        assert_eq!(checkpoint_extension_start(&[10, 20], &[10, 20]), None);
        assert_eq!(checkpoint_extension_start(&[10, 20], &[10, 99, 30]), None);
    }

    #[test]
    fn invalidating_recurrent_checkpoint_clears_all_publish_metadata() {
        let mut checkpoint = TurnRecurrentCkpt {
            kbufs: Vec::new(),
            vbufs: Vec::new(),
            ple_state: None,
            layers: Vec::new(),
            tokens: vec![10, 20, 30],
            copied: vec![true, true],
            ple_copied: true,
            valid: true,
        };

        checkpoint.invalidate();

        assert!(!checkpoint.valid);
        assert!(checkpoint.tokens.is_empty());
        assert_eq!(checkpoint.copied, [false, false]);
        assert!(!checkpoint.ple_copied);
    }
}
