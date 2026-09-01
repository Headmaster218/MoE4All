//! Per-layer weight handles + the persistent seam session state ([`SeamKv`]/[`SeamWeights`]).
//! Pure-move split of `seam.rs` — see `super` for the module overview.
use super::sc::{DenoiseCache, SelfCondWeights};
use super::segmented_kv::{PlaneKind, SegmentedKvLayout, KV_GROW_ROWS};
use super::{common_prefix_len, kv_fmt_bytes};
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

/// Ling KDA mixer weights. Q/K/V projection and causal-conv banks are concatenated once at load,
/// matching the packed activation consumed by [`infr_core::Op::Kda`].
pub(super) struct KdaW {
    pub(super) qkv: TensorId,
    pub(super) conv: TensorId,
    pub(super) forget: TensorId,
    pub(super) beta: TensorId,
    pub(super) a: TensorId,
    pub(super) dt_bias: TensorId,
    pub(super) norm: TensorId,
    pub(super) gate: TensorId,
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
/// Both are unconditional on every V4 layer (`hc_attn_*` / `hc_ffn_*`).
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
    /// Whether the MoE expert banks all have a dp4a-mmq kernel (batched-prefill eligibility).
    pub(super) moe_batched_ok: bool,
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
    /// to f16 `[n_embd, n_vocab]`, ~1.4 GB — see the reference's `dg_ensure_sc_embT` and
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
    /// Rolling conversation checkpoint for append-only recurrent mixers. Unlike the MTP rollback
    /// checkpoint above, this snapshot is taken at the stable rendered-history boundary BEFORE
    /// the assistant generation prompt. The next turn can therefore restore that exact state and
    /// prefill only the prior visible answer plus the new user turn, even when chat-history
    /// normalization makes the newly rendered prompt diverge from `cached`.
    pub(super) turn_recurrent_ckpt: Option<TurnRecurrentCkpt>,
}

#[derive(Default)]
pub(super) struct SegmentedKvState {
    pub(super) enabled: bool,
    committed_tokens: usize,
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
/// cached-token length the snapshot corresponds to. Allocated once (lazily) and reused every cycle.
pub(super) struct MtpDeltaCkpt {
    kbufs: Vec<Box<dyn Buffer>>,
    vbufs: Vec<Box<dyn Buffer>>,
    /// The layer indices (into `SeamKv::kbufs`/`vbufs`) that are DeltaNet mixers.
    layers: Vec<usize>,
    cached_len: usize,
}

/// Device-resident recurrent state at one stable rendered conversation boundary. Buffers are
/// allocated once per slot and reused; `copied` lets layer-major prefill capture each recurrent
/// layer exactly when that layer reaches the boundary without any per-operation allocation.
pub(super) struct TurnRecurrentCkpt {
    kbufs: Vec<Box<dyn Buffer>>,
    vbufs: Vec<Box<dyn Buffer>>,
    layers: Vec<usize>,
    tokens: Vec<u32>,
    copied: Vec<bool>,
    valid: bool,
}

fn checkpoint_extension_start(checkpoint: &[u32], prompt: &[u32]) -> Option<usize> {
    (!checkpoint.is_empty() && checkpoint.len() < prompt.len() && prompt.starts_with(checkpoint))
        .then_some(checkpoint.len())
}

impl TurnRecurrentCkpt {
    /// Start replacing the rolling checkpoint with `tokens`. The existing device buffers are
    /// retained; validity is published only after every recurrent layer has been copied.
    pub(super) fn begin(
        slot: &mut Option<Self>,
        be: &dyn Backend,
        cfg: &Config,
        src_k: &[Box<dyn Buffer>],
        src_v: &[Box<dyn Buffer>],
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
                kbufs.push(
                    be.alloc(src_k[l].len_bytes().max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                vbufs.push(
                    be.alloc(src_v[l].len_bytes().max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
            }
            let copied = vec![false; layers.len()];
            *slot = Some(Self {
                kbufs,
                vbufs,
                layers,
                tokens: Vec::new(),
                copied,
                valid: false,
            });
        }
        let ck = slot.as_mut().expect("checkpoint was just allocated");
        ck.tokens.clear();
        ck.tokens.extend_from_slice(tokens);
        ck.copied.fill(false);
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
        be.copy_buffer(
            src_k[layer].as_ref(),
            self.kbufs[i].as_ref(),
            src_k[layer].len_bytes(),
        )
        .map_err(|e| anyhow!("{e}"))?;
        be.copy_buffer(
            src_v[layer].as_ref(),
            self.vbufs[i].as_ref(),
            src_v[layer].len_bytes(),
        )
        .map_err(|e| anyhow!("{e}"))?;
        self.copied[i] = true;
        self.valid = !self.tokens.is_empty() && self.copied.iter().all(|&done| done);
        Ok(())
    }

    /// Capture every recurrent layer. Used by chunk-major prefill and the per-token path.
    pub(super) fn snapshot_all(
        &mut self,
        be: &dyn Backend,
        src_k: &[Box<dyn Buffer>],
        src_v: &[Box<dyn Buffer>],
    ) -> AResult<()> {
        for i in 0..self.layers.len() {
            self.snapshot_index(be, src_k, src_v, i)?;
        }
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
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SeamKv {
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
        let checkpoint = self.turn_recurrent_ckpt.as_ref().and_then(|ck| {
            ck.valid
                .then(|| checkpoint_extension_start(&ck.tokens, prompt))
                .flatten()
        });
        live.into_iter().chain(checkpoint).max()
    }

    /// Forget the materialized tokens WITHOUT dropping weights or buffers: the next call
    /// re-prefills from position 0 into the same session. Bench reps use this so each rep
    /// measures a full prefill while weights/pipelines/repack caches stay warm.
    /// (cfg-gated with its only caller, the Metal bench session — dead code on other targets.)
    #[cfg(target_os = "macos")]
    pub(crate) fn reset_tokens(&mut self) {
        self.cached.clear();
        self.invalidate_turn_checkpoint();
    }

    /// Number of token ids materialized in this slot's KV cache.
    pub(crate) fn cached_len(&self) -> usize {
        self.cached.len()
    }

    /// Forget the materialized tokens (the KV rows become dead; the next prompt prefills from
    /// row 0). Used to discard a warmup generation without dropping the slot's buffers.
    pub(crate) fn reset(&mut self) {
        self.cached.clear();
        self.invalidate_turn_checkpoint();
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
        if cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3 {
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
        if let Some(ck) = self.turn_recurrent_ckpt.as_mut() {
            ck.valid = false;
            ck.tokens.clear();
            ck.copied.fill(false);
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
        let Some(ck) = self.turn_recurrent_ckpt.as_ref() else {
            return Ok(None);
        };
        let Some(len) = ck
            .valid
            .then(|| checkpoint_extension_start(&ck.tokens, prompt))
            .flatten()
        else {
            return Ok(None);
        };
        for (i, &l) in ck.layers.iter().enumerate() {
            be.copy_buffer(
                ck.kbufs[i].as_ref(),
                self.kbufs[l].as_ref(),
                ck.kbufs[i].len_bytes(),
            )
            .map_err(|e| anyhow!("{e}"))?;
            be.copy_buffer(
                ck.vbufs[i].as_ref(),
                self.vbufs[l].as_ref(),
                ck.vbufs[i].len_bytes(),
            )
            .map_err(|e| anyhow!("{e}"))?;
        }
        self.cached.clone_from(&ck.tokens);
        Ok(Some(len))
    }

    /// Fork a fresh conversation slot: same (Arc-shared) weights, its own zero KV + IO buffers.
    /// Snapshot the qwen35 DeltaNet recurrent state (every DeltaNet layer's conv + S buffers) plus
    /// the current `cached` length into the device-resident [`MtpDeltaCkpt`] (allocated once on the
    /// first call). The MTP spec-decode loop calls this at a CLEAN committed boundary (after the
    /// prime prefill and after every fully-accepted cycle) so a later partial-accept cycle can roll
    /// back to it via [`mtp_restore_delta`]. A no-op on a non-qwen35 model (no DeltaNet layers). The
    /// snapshot is a pure device→device buffer copy (`Backend::copy_buffer`), never a host bounce.
    pub(crate) fn mtp_snapshot_delta(&mut self, be: &dyn Backend, cfg: &Config) -> AResult<()> {
        if self.mtp_delta_ckpt.is_none() {
            let layers: Vec<usize> = (0..cfg.n_layer)
                .filter(|&l| cfg.qwen35 && !cfg.is_qwen35_attn_layer(l))
                .collect();
            if layers.is_empty() {
                return Ok(());
            }
            let mut kbufs = Vec::with_capacity(layers.len());
            let mut vbufs = Vec::with_capacity(layers.len());
            for &l in &layers {
                kbufs.push(
                    be.alloc(self.kbufs[l].len_bytes().max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
                vbufs.push(
                    be.alloc(self.vbufs[l].len_bytes().max(1), BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                );
            }
            self.mtp_delta_ckpt = Some(MtpDeltaCkpt {
                kbufs,
                vbufs,
                layers,
                cached_len: 0,
            });
        }
        let cached_len = self.cached.len();
        let ck = self.mtp_delta_ckpt.as_ref().expect("just ensured Some");
        for (i, &l) in ck.layers.iter().enumerate() {
            be.copy_buffer(
                self.kbufs[l].as_ref(),
                ck.kbufs[i].as_ref(),
                self.kbufs[l].len_bytes(),
            )
            .map_err(|e| anyhow!("{e}"))?;
            be.copy_buffer(
                self.vbufs[l].as_ref(),
                ck.vbufs[i].as_ref(),
                self.vbufs[l].len_bytes(),
            )
            .map_err(|e| anyhow!("{e}"))?;
        }
        self.mtp_delta_ckpt
            .as_mut()
            .expect("just ensured Some")
            .cached_len = cached_len;
        Ok(())
    }

    /// Restore the DeltaNet state captured by the last [`mtp_snapshot_delta`] and truncate `cached`
    /// back to the snapshot's token length — the MTP loop's rollback after a partial-accept cycle
    /// (drops the rejected drafts the verify forward absorbed into the recurrent state). A no-op
    /// when no snapshot has been taken yet.
    pub(crate) fn mtp_restore_delta(&mut self, be: &dyn Backend) -> AResult<()> {
        let Some(ck) = self.mtp_delta_ckpt.as_ref() else {
            return Ok(());
        };
        for (i, &l) in ck.layers.iter().enumerate() {
            be.copy_buffer(
                ck.kbufs[i].as_ref(),
                self.kbufs[l].as_ref(),
                ck.kbufs[i].len_bytes(),
            )
            .map_err(|e| anyhow!("{e}"))?;
            be.copy_buffer(
                ck.vbufs[i].as_ref(),
                self.vbufs[l].as_ref(),
                ck.vbufs[i].len_bytes(),
            )
            .map_err(|e| anyhow!("{e}"))?;
        }
        self.cached.truncate(ck.cached_len);
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
            ple_embd_buf: if cfg.qwen4exp {
                let heads = (cfg.ple_ngram_size - 1) * cfg.ple_heads_per_ngram;
                Some(
                    be.alloc(cfg.ple_head_dim * heads * 4, BufferUsage::Staging)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            },
            ple_state_buf: if cfg.qwen4exp {
                let hist = (cfg.ple_conv_kernel - 1) * cfg.ple_ngram_size;
                Some(
                    be.alloc(hist * cfg.hc_mult * cfg.n_embd * 4, BufferUsage::KvCache)
                        .map_err(|e| anyhow!("{e}"))?,
                )
            } else {
                None
            },
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
            mtp_delta_ckpt: None,
            turn_recurrent_ckpt: None,
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
    ) -> AResult<()> {
        if cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3 || cfg.deepseek4 {
            return Ok(());
        }
        let p = p.min(src.cached.len()).min(self.max_ctx);
        if p == 0 {
            return Ok(());
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
                return Ok(());
            }
        }
        for l in 0..cfg.n_layer {
            // One prefix position is `k_row`/`v_row` elements per side — MLA's compressed row is
            // three times `n_kv * head_dim` wide and has no V side at all, so both widths come
            // from the shared `crate::seam::kv_row_elems` (docs/backlog.md B41).
            let (k_row, v_row) = crate::seam::kv_row_elems(cfg, l);
            be.copy_buffer(
                src.kbufs[l].as_ref(),
                self.kbufs[l].as_ref(),
                kv_fmt_bytes(self.k_fmt, p * k_row),
            )
            .map_err(|e| anyhow!("{e}"))?;
            // `v_row == 0`: this arch caches no V (MLA) and `vbufs[l]` is the placeholder — there
            // is nothing to seed.
            if v_row > 0 {
                be.copy_buffer(
                    src.vbufs[l].as_ref(),
                    self.vbufs[l].as_ref(),
                    kv_fmt_bytes(self.v_fmt, p * v_row),
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }
        self.cached = src.cached[..p].to_vec();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::checkpoint_extension_start;

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
}
