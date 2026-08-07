//! Model hyper-parameters parsed from GGUF metadata. Mechanically split out of `lib.rs`.
use crate::{meta_f64, meta_u64, MoeConfig};
use anyhow::{bail, Context, Result};
use infr_core::loader::MetaValue;
use infr_core::WeightSource;
use infr_gguf::Gguf;

/// `Default` is an ALL-ZERO placeholder, not a runnable model: [`Config::from_gguf`] is the only
/// source of a real one. It exists so a test can name the two or three fields the arithmetic it is
/// checking actually reads (`Config { n_layer: 48, n_kv: 8, ..Default::default() }`) instead of
/// spelling out sixty irrelevant ones — see `seam_helper_tests`' KV-fit cases.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub n_embd: usize,
    /// Dense FFN inner width. For models with a uniform FFN this is the width every layer uses; for
    /// gemma4 E2B (per-layer FFN array) it's the MAX over layers, used to size shared FFN scratch.
    pub n_ff: usize,
    /// Per-layer FFN inner width. gemma4 E2B stores `feed_forward_length` as an array (most 6144, the
    /// late layers 12288); every other model is uniform (all entries equal `n_ff`).
    pub n_ff_layers: Vec<usize>,
    /// gemma4 gemma3n-style per-layer input embeddings: the width of each layer's extra input vector
    /// (`embedding_length_per_layer_input`, 256 for E2B). `0` = the model has no per-layer embeddings.
    pub n_embd_per_layer: usize,
    /// gemma4 E2B KV sharing (gemma3n): only the first `n_layer_kv_from_start` layers compute + cache
    /// their own K/V; later layers reuse an earlier layer's cache (SWA→`from_start-2`, full→`-1`).
    /// Equal to `n_layer` (every layer owns its KV) for models without sharing.
    pub n_layer_kv_from_start: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub vocab: usize,
    pub eos: u32,
    /// All tokens that end generation (the GGUF eos plus `<|im_end|>` / `<|endoftext|>` when present
    /// in the vocab). A chat model can emit any of these; stopping only on `eos` lets it ramble.
    pub eos_ids: Vec<u32>,
    /// Qwen3-style per-head RMSNorm on Q and K before RoPE.
    pub qk_norm: bool,
    /// Qwen2/2.5 add a learned bias to the q/k/v projections (`Wx + b`); Qwen3 dropped them. o-proj
    /// and FFN stay bias-free. `true` only for `arch == "qwen2"`.
    pub qkv_bias: bool,
    /// qwen2 needs the NEOX (rotate-half) RoPE but rides the no-qknorm path, whose `Op::Rope` is the
    /// INTERLEAVED (NORM) rotation — llama-arch GGUFs get that layout from the converter's q/k row
    /// permute, qwen2's GGUF stays in HF order. Rather than a NEOX kernel variant on every backend,
    /// the loader applies the same permute (new\[2p\]=old\[p\], new\[2p+1\]=old\[p+rd/2\] per head) to
    /// attn_q/attn_k rows + biases: NORM rope over permuted rows == NEOX over the originals, and q·k
    /// dots are permutation-invariant (K cached permuted, Q permuted identically; V untouched).
    /// qwen3/gemma rotate NEOX inside QkNormRope directly — no permute.
    pub permute_qk_neox: bool,
    /// gemma family: scale input embeddings by √n_embd, sandwich norms (post-attn / post-ffw RMSNorm
    /// before the residual add), and a GeGLU (GELU) FFN instead of SwiGLU.
    pub gemma: bool,
    /// gemma4: adds per-layer heterogeneous head dims (the `*_swa` fields), a weightless RMSNorm on V,
    /// attention scale 1.0 (no 1/√d — QK-norm handles magnitude), a final logit softcap, and
    /// proportional RoPE (freq_factors) on the full-attention layers. `true` for gemma4 AND
    /// diffusion-gemma (its backbone is gemma4's, verbatim — see `docs/diffusion-gemma.md`); use
    /// `diffusion_gemma` below to gate the DIFFERENCES (dual FFN, encoder/decoder output scalars).
    pub gemma4: bool,
    /// diffusion-gemma (block text-diffusion MoE on the gemma4 backbone): `true` only for
    /// `arch == "diffusion-gemma"`. Gates the dual per-layer FFN (dense GeGLU ∥ 128-expert MoE,
    /// summed) and the encoder-scalar tensor name (`enc_layer_output_scale` vs gemma4's
    /// `layer_output_scale`); everything else reuses the `gemma4` backbone gate above.
    pub diffusion_gemma: bool,
    /// gemma4 MoE (26B-A4B): `true` for a plain `gemma4` arch that carries routed-expert tensors.
    /// Shares diffusion-gemma's dual-FFN wiring (`FfnW::DiffusionMoe`) — gate the FFN build with
    /// [`Config::dual_moe`], NOT `diffusion_gemma`, so both variants take it — but runs the standard
    /// autoregressive decode (no canvas/denoise). `false` for gemma4 dense (E2B/E4B).
    pub gemma4_moe: bool,
    /// diffusion-gemma: the canvas (denoise-target) length — `[prompt | canvas]` splits at
    /// `n_tokens - canvas_length`. Phase 1 (prompt-only causal prefill) doesn't slice the canvas
    /// off, but the field is parsed now so later phases don't need to touch `Config` again. `0`
    /// for every non-diffusion-gemma model.
    pub canvas_length: usize,
    /// diffusion-gemma: the vocab id used to pad an unfinished canvas block (`tokenizer.ggml.
    /// mask_token_id`). Unused until the denoise decode loop (Phase 3). `0` for every other model.
    pub mask_token_id: u32,
    /// diffusion-gemma: the entropy-bound sampler's parameters (Phase 3 — see
    /// `docs/diffusion-gemma.md`'s "Decode loop" section and `diffusion_generate_entropy_bound` in
    /// the reference `examples/diffusion/diffusion.cpp`). Parsed from `diffusion.eb_*` GGUF
    /// metadata with the reference's own fallbacks (`diffusion-cli.cpp`'s `meta_f`/`meta_i`
    /// defaults) when a key is absent. All zero/unused for every non-diffusion-gemma model.
    pub eb_max_steps: usize,
    pub eb_t_min: f32,
    pub eb_t_max: f32,
    pub eb_entropy_bound: f32,
    pub eb_stability_threshold: usize,
    pub eb_confidence_threshold: f32,
    /// Per-layer dims for the SWA (local) layers when they differ from the full (global) layers
    /// (gemma4). Equal to `head_dim` / `n_kv` / `rope_dim` for uniform-dim models.
    pub head_dim_swa: usize,
    pub n_kv_swa: usize,
    pub rope_dim_swa: usize,
    /// Final logit softcap (gemma2/gemma4): `logits = cap * tanh(logits / cap)`. `0` = no softcap.
    pub final_softcap: f32,
    /// Sliding-window attention size (gemma); `0` = full causal attention everywhere. SWA layers
    /// only attend to the last `swa_window` keys.
    pub swa_window: usize,
    /// SWA layer pattern (gemma): every `swa_pattern`-th layer uses FULL attention, the rest SWA.
    /// `0`/`1` = no pattern. llama.cpp `set_swa_pattern(p)`: layer `il` is full iff `(il+1) % p == 0`.
    pub swa_pattern: usize,
    /// RoPE base for the SWA (local) layers (gemma3 dual-rope): SWA layers use this, full layers use
    /// `rope_theta`. Defaults to 10000 (llama.cpp's `rope_freq_base_train_swa` default) when gemma's
    /// GGUF omits an explicit `rope.freq_base_swa`. Equal to `rope_theta` for non-SWA models.
    pub swa_rope_theta: f32,
    /// MoE config (qwen3moe): `Some` enables the routed-expert FFN. `None` = dense FFN.
    pub moe: Option<MoeConfig>,
    /// The model's trained/default maximum context length (`<arch>.context_length`). Used as the
    /// default KV-cache size when the caller doesn't request a custom context (overridable). Falls
    /// back to 8192 if the GGUF omits it.
    pub n_ctx_train: usize,
    /// qwen35 (Qwen3.5/3.6 gated-DeltaNet hybrid): `true` only for `arch == "qwen35"`. Gates every
    /// field below plus the `MixerW::DeltaNet` mixer branch in `seam`'s layer loop — the same
    /// shared transformer skeleton every other architecture runs through.
    pub qwen35: bool,
    /// qwen35: attention layers sit at `i` where `(i+1) % full_attn_interval == 0`; every other
    /// layer is gated-DeltaNet linear attention. `0` for every non-qwen35 model (never read).
    pub full_attn_interval: usize,
    /// qwen35 SSM (gated-DeltaNet) dims — see `docs/qwen35.md`. `0` for non-qwen35 models.
    pub ssm_d_conv: usize,
    pub ssm_d_state: usize,
    pub ssm_d_inner: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    /// qwen35 sectioned RoPE (`rope.dimension_sections`, e.g. `[11,11,10,0]`). Parsed for parity
    /// with `qwen35::Cfg`, but NOT applied differently from plain NEOX rope: with every section
    /// sharing the same 1-D position id, a sectioned rotation over
    /// `rope_dim` collapses to the standard `QkNormRope`, so `layer_rope_dim`/`layer_rope_theta`
    /// alone drive qwen35's rope emission. `[0;4]` for non-qwen35 models.
    pub rope_sections: [u32; 4],
    /// qwen35 attention layers pack `q` and an output SIGMOID gate INTERLEAVED per head in
    /// `attn_q` (`[h0 q(hd) | h0 gate(hd) | h1 q | h1 gate | …]`, NOT two contiguous blocks) —
    /// the one real trap in `docs/qwen35.md`. `true` only for qwen35.
    pub attn_out_gate: bool,
    /// MTP/NextN (Qwen3.5/3.6, issue #33 — see `docs/mtp.md`): the number of extra decoder blocks
    /// `{arch}.block_count` includes BEYOND the trunk (`{arch}.nextn_predict_layers`). `n_layer`
    /// above is already the TRUNK count (`block_count - n_layer_nextn`) — every existing field/
    /// helper on this `Config` keeps working unmodified. `0` for every model without an MTP head
    /// (which is every arch except qwen35 today). The MTP head layer itself sits at GGUF index
    /// `blk.{n_layer}` — Phase 1 only parses this; the head weights load separately (see
    /// `crate::mtp`) and Phase 2 emits its forward.
    pub n_layer_nextn: usize,
    /// qwen35moe (Qwen3.6 MoE) shared-expert FFN inner width (`expert_shared_feed_forward_length`)
    /// — a Qwen2-MoE-style DENSE SwiGLU branch run on the same per-layer input as the routed MoE
    /// bank, gated by a per-token sigmoid (`ffn_gate_inp_shexp`) and summed with the routed
    /// output (see `FfnW::Moe`'s `shexp` field in `seam::weights` and its graph wiring in
    /// `seam::runner`). `0` = no shared expert — every non-qwen35moe model, and any qwen35moe
    /// model without `ffn_*_shexp` tensors.
    pub shexp_ff: usize,
    /// llama4 (Scout etc.): `true` only for `arch == "llama4"`. Gates every field below plus the
    /// per-layer NoPE / weightless-L2-norm attention wiring and the sigmoid-gated MoE in `seam`.
    pub llama4: bool,
    /// llama4 `interleave_moe_layer_step`: layer `il` is a MoE layer iff `(il+1) % step == 0`
    /// (`step == 1` on Scout → every layer). Other layers are dense FFN. `0` for every non-llama4
    /// model (whose MoE archs — qwen3moe etc. — are MoE on ALL layers; see [`Config::is_moe_layer`]).
    pub moe_interleave_step: usize,
    /// llama4 iRoPE `no_rope_layer_step`: layer `il` SKIPS rope (NoPE / global attention) iff
    /// `(il+1) % step == 0` (4 on Scout — pattern "3 local-rope, 1 global-NoPE"). `0` = every layer
    /// uses rope (every non-llama4 model, and a llama4 checkpoint with `attention.sliding_window==0`).
    pub no_rope_step: usize,
    /// llama4 `Llama4TextL2Norm`: a WEIGHTLESS per-head RMS/L2-norm applied to Q and K AFTER rope,
    /// on the rope (non-NoPE) layers only. `use_kq_norm = n_expert != 128` in the reference (`true`
    /// for Scout 16E). `false` for every non-llama4 model.
    pub kq_l2norm: bool,
    /// Whether the MoE shared expert is GATED by a per-token sigmoid (qwen35moe, Qwen2-MoE style)
    /// or summed in PLAIN (llama4). Only meaningful when `shexp_ff > 0`. `false` for llama4.
    pub shexp_gated: bool,
    /// bitnet (BitNet b1.58) SubLN: `true` only for `arch == "bitnet"`. Loads two extra per-layer
    /// RMSNorm weights (`attn_sub_norm` `[n_embd]`, `ffn_sub_norm` `[n_ff]`) and inserts them in the
    /// seam's block graph — `attn_sub_norm` on the concatenated-heads attention output BEFORE the
    /// o-projection (`wo`), and `ffn_sub_norm` on the FFN intermediate BEFORE `ffn_down`. Confirmed
    /// against llama.cpp's `build_bitnet` (`src/models/bitnet.cpp`). `false` for every other arch
    /// (their block graphs stay byte-identical).
    pub sub_norm: bool,
    /// DeepSeek V1 (and later): `true` only for `arch == "deepseek"`. Gates the MoE with
    /// no-renormalization softmax gating, threshold-mode `is_moe_layer`, and an ungated
    /// shared expert (summed in plain, like llama4). See docs/deepseek.md § Stage 1.
    pub deepseek: bool,
    /// DeepSeek: the first `n_layer_dense_lead` layers use a DENSE FFN; the rest are MoE
    /// (unless the model has no MoE — `Config::moe` is `None` — in which case every layer
    /// is dense regardless). `0` = every layer MoE (not used today). Non-DeepSeek models
    /// ignore this field.
    pub n_layer_dense_lead: usize,
    /// DeepSeek V2+: `true` only for `arch == "deepseek2"`. Gates MLA attention (absorbed form with
    /// asymmetric K/V dims + single K row), YaRN rope scaling, and group-limited MoE routing with
    /// router bias correction. See docs/deepseek.md § Stage 2.
    pub deepseek2: bool,
    /// DeepSeek V2+ MLA: Q LoRA rank (wq_a output dim, q_a_norm dim, wq_b input dim). `0` when the
    /// "lite" variant carries a direct `wq` instead.
    pub q_lora_rank: usize,
    /// DeepSeek V2+ MLA: KV LoRA rank (wkv_a_mqa latent dim, attn_kv_a_norm dim, wk_b/wv_b dim-0).
    pub kv_lora_rank: usize,
    /// DeepSeek V2+ MLA: RoPE dimension on Q and K (q_pe/k_pe width = rope.dimension_count).
    pub qk_rope_dim: usize,
    /// DeepSeek V2+ MLA: MLA key length = kv_lora_rank + qk_rope_dim (576 for V3). The width of
    /// the single cached KV row per token.
    pub key_length: usize,
    /// DeepSeek V2+ MLA: per-head Q nope dim (128 for V3). Derived from wq_b output shape /
    /// n_head — NOT a GGUF key. `0` for non-DeepSeek2 models.
    pub head_k_mla: usize,
    /// DeepSeek V2+ MLA: per-head V output dim (128 for V3). Derived from wv_b shape — NOT a GGUF
    /// key. `0` for non-DeepSeek2 models.
    pub v_head_dim: usize,
    /// DeepSeek V2+ MoE: gating function enum (1=softmax, 2=sigmoid, 3=softmax-on-weights,
    /// 4=sqrt-softplus). Parsed from `expert_gating_func`, mapped to `MoeConfig::gating`.
    /// `0` for non-DeepSeek2 models (and V1, which hardcodes softmax).
    pub expert_gating_func: u8,
    /// DeepSeek V2+ MoE: group-limited routing — number of expert groups. `0` = no grouping.
    pub n_expert_groups: usize,
    /// DeepSeek V2+ MoE: number of groups selected per routing decision.
    pub n_expert_groups_used: usize,
    /// DeepSeek V2+ YaRN: the `rope.scaling.yarn_log_multiplier` divided by 0.1 on load (the
    /// converter writes `0.1 * mscale_all_dim`; the loader divides it back). `0.0` = no YaRN.
    pub rope_yarn_log_mul: f32,
    /// DeepSeek V2+ YaRN: `rope.scaling.type == "yarn"` (the GGUF declares it). Gates the
    /// per-dimension frequency ramp (`freq_scale` interpolation) applied to the q_pe/k_pe rope
    /// divisors and the constant mscale² attention boost — see `docs/deepseek.md` § YaRN.
    /// llama.cpp activates the full ramp whenever this flag is set (`yarn_ext_factor = 1.0` in
    /// `llama-context.cpp`), regardless of the current context length.
    pub rope_scaling_yarn: bool,
    /// DeepSeek V2+ YaRN: `rope.scaling.factor` — the long-context interpolate factor (40 for
    /// V2-Lite). `1.0` when the GGUF omits it. The per-pair divisor is `freq_scale = 1/factor`
    /// ramped toward 1.0 below `corr_dim` (see `docs/deepseek.md`).
    pub rope_scaling_factor: f32,
    /// DeepSeek V2+ YaRN: `rope.scaling.original_context_length` — the context the model was
    /// TRAINED at, the `n_ctx_orig` of the corr_dim ramp (4096 for V2-Lite). Defaults to
    /// `context_length` when the GGUF omits it (llama.cpp's `n_ctx_orig_yarn` fallback). NOT
    /// `n_ctx_train`: this GGUF sets `context_length = 163840` while the yarn ramp must use 4096.
    pub rope_scaling_orig_ctx: usize,
    /// DeepSeek V2+ "lite" variant: `wq` present ⇒ lite (no wq_a/q_a_norm/wq_b). Detected by
    /// tensor presence, not GGUF key.
    pub is_lite: bool,
}

fn positive_model_dimension(key: &str, value: u64) -> Result<usize> {
    let value = value as usize;
    if value == 0 {
        bail!("{key} must be positive");
    }
    Ok(value)
}

fn positive_array_model_dimension(
    values: Option<&[MetaValue]>,
    key: &str,
    index: usize,
) -> Result<Option<usize>> {
    let Some(value) = values
        .and_then(|values| values.get(index))
        .and_then(MetaValue::as_u64)
    else {
        return Ok(None);
    };
    positive_model_dimension(&format!("{key}[{index}]"), value).map(Some)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Config {
    /// Whether layer `il` uses sliding-window (vs full) attention. gemma interleaves SWA with full
    /// attention on a fixed period; non-gemma models are always full.
    pub fn is_swa_layer(&self, il: usize) -> bool {
        self.swa_window > 0 && self.swa_pattern > 1 && !(il + 1).is_multiple_of(self.swa_pattern)
    }

    /// RoPE base for layer `il`: gemma3 SWA (local) layers use the smaller `swa_rope_theta`, full
    /// (global) layers use `rope_theta`. Non-gemma models return `rope_theta` for every layer.
    pub fn layer_rope_theta(&self, il: usize) -> f32 {
        if self.is_swa_layer(il) {
            self.swa_rope_theta
        } else {
            self.rope_theta
        }
    }

    /// Head dim for layer `il`. gemma4 SWA layers are narrower than full layers; uniform elsewhere.
    pub fn layer_head_dim(&self, il: usize) -> usize {
        if self.is_swa_layer(il) {
            self.head_dim_swa
        } else {
            self.head_dim
        }
    }

    /// KV-head count for layer `il` (gemma4 SWA vs full GQA grouping; uniform elsewhere).
    pub fn layer_n_kv(&self, il: usize) -> usize {
        if self.is_swa_layer(il) {
            self.n_kv_swa
        } else {
            self.n_kv
        }
    }

    /// FFN inner width for layer `il`. gemma4 E2B's late layers are wider (12288 vs 6144); uniform
    /// (`n_ff`) for every other model.
    pub fn layer_n_ff(&self, il: usize) -> usize {
        self.n_ff_layers.get(il).copied().unwrap_or(self.n_ff)
    }

    /// Whether layer `il` computes + caches its own K/V. gemma4 E2B's later layers (`il >=
    /// n_layer_kv_from_start`) reuse an earlier layer's cache instead. `true` for every layer of a
    /// non-sharing model.
    pub fn has_own_kv(&self, il: usize) -> bool {
        il < self.n_layer_kv_from_start
    }

    /// The cache layer whose K/V layer `il` attends to. For an own-KV layer that's `il` itself; for a
    /// gemma4 E2B shared layer it's `n_layer_kv_from_start - (2 if SWA else 1)` (matching llama.cpp's
    /// gemma3n/gemma4 reuse: SWA shared layers reuse the last own SWA layer, full the last own full).
    pub fn kv_src_layer(&self, il: usize) -> usize {
        if self.has_own_kv(il) {
            il
        } else {
            self.n_layer_kv_from_start - if self.is_swa_layer(il) { 2 } else { 1 }
        }
    }

    /// RoPE rotation dim for layer `il` (gemma4 SWA vs full; uniform elsewhere).
    pub fn layer_rope_dim(&self, il: usize) -> usize {
        if self.is_swa_layer(il) {
            self.rope_dim_swa
        } else {
            self.rope_dim
        }
    }

    /// The largest per-layer head_dim / n_kv across all layers — used to size shared activation and
    /// KV scratch that's reused across layers of differing width (gemma4).
    pub fn max_head_dim(&self) -> usize {
        self.head_dim.max(self.head_dim_swa)
    }
    pub fn max_n_kv(&self) -> usize {
        self.n_kv.max(self.n_kv_swa)
    }

    /// Whether the per-layer FFN is the dual GeGLU ∥ MoE block (`FfnW::DiffusionMoe`): both
    /// diffusion-gemma and the autoregressive gemma4 MoE (26B-A4B) share that FFN wiring. Gate
    /// FFN structure on this, NOT `diffusion_gemma` (which additionally implies canvas/denoise).
    pub fn dual_moe(&self) -> bool {
        self.diffusion_gemma || self.gemma4_moe
    }

    /// Whether layer `il` uses the routed-expert (MoE) FFN. For every MoE arch except llama4 and
    /// deepseek that's EVERY layer (`moe_interleave_step == 0`). llama4 interleaves MoE with dense
    /// layers on a fixed step (`(il+1) % step == 0`). DeepSeek uses a threshold: the first
    /// `n_layer_dense_lead` layers are dense, the rest are MoE. `false` for dense models
    /// (`self.moe` is `None`).
    pub fn is_moe_layer(&self, il: usize) -> bool {
        self.moe.is_some()
            && if self.deepseek || self.deepseek2 {
                il >= self.n_layer_dense_lead
            } else if self.moe_interleave_step > 0 {
                (il + 1).is_multiple_of(self.moe_interleave_step)
            } else {
                true
            }
    }

    /// llama4 iRoPE: whether layer `il` SKIPS rope (a NoPE / global-attention layer). `false` for
    /// every non-llama4 model (`no_rope_step == 0`).
    pub fn is_nope_layer(&self, il: usize) -> bool {
        self.no_rope_step > 0 && (il + 1).is_multiple_of(self.no_rope_step)
    }

    /// qwen35: whether layer `il` is one of the FULL-attention layers (vs gated-DeltaNet linear
    /// attention). `false` for every non-qwen35 model. Mirrors the old seam's `Cfg::is_attn_layer`.
    pub fn is_qwen35_attn_layer(&self, il: usize) -> bool {
        self.qwen35
            && self.full_attn_interval > 0
            && (il + 1).is_multiple_of(self.full_attn_interval)
    }
    /// qwen35 gated-DeltaNet derived dims (see `docs/qwen35.md`) — ports of the old seam's `Cfg`
    /// helpers of the same name, now living on the shared `Config`.
    pub fn q35_num_k_heads(&self) -> usize {
        self.ssm_n_group
    }
    pub fn q35_num_v_heads(&self) -> usize {
        self.ssm_dt_rank
    }
    pub fn q35_head_k_dim(&self) -> usize {
        self.ssm_d_state
    }
    pub fn q35_head_v_dim(&self) -> usize {
        self.ssm_d_inner / self.ssm_dt_rank.max(1)
    }
    pub fn q35_conv_channels(&self) -> usize {
        self.ssm_d_inner + 2 * self.ssm_n_group * self.ssm_d_state
    }

    /// Parse the model config purely from GGUF metadata + tensor shapes — no GPU/Vulkan, no weight
    /// upload. The single source of truth for both the GPU loader ([`Llama::load_opt`]) and the
    /// CPU-only loader ([`SeamModel::load`]). `eos_ids` holds only the GGUF `eos` here; chat-end
    /// markers (`<|im_end|>` …) are appended once a tokenizer exists (see [`add_chat_eos`]).
    pub fn from_gguf(g: &Gguf) -> Result<Config> {
        let arch = g
            .metadata()
            .str("general.architecture")
            .unwrap_or("")
            .to_string();
        let qk_norm = match arch.as_str() {
            // llama4 has no LEARNED q/k-norm weights; its weightless per-head L2-norm on Q/K is
            // applied AFTER rope (see `kq_l2norm` below), not via the qwen3-style QkNormRope.
            // bitnet (BitNet b1.58) is the llama skeleton + SubLN (two extra RMSNorms, gated by
            // `sub_norm` below) — no learned q/k-norm, like llama/qwen2.
            crate::arch::LLAMA
            | crate::arch::LLAMA4
            | crate::arch::QWEN2
            | crate::arch::DEEPSEEK
            | crate::arch::DEEPSEEK2
            | crate::arch::BITNET
            | crate::arch::BITNET_B158 => false,
            crate::arch::QWEN3
            | crate::arch::QWEN3_MOE
            | crate::arch::GEMMA3
            | crate::arch::GEMMA4
            | crate::arch::DIFFUSION_GEMMA => true,
            // qwen35's full-attention layers are qk-normed like qwen3/gemma (see `docs/qwen35.md`).
            // Kept as its own match arm (rather than folded into `arch::TRANSFORMER` above) because
            // `is_qwen35` gates a handful of qwen35-only fields below it; the error message renders
            // both lists so the supported set can't drift from the match arms above. `QWEN35_MOE`
            // (Qwen3.6 MoE) shares the exact same attention/qk-norm shape as dense `QWEN35` — only
            // its FFN differs (see the `moe`/`shexp_ff` parses below).
            crate::arch::QWEN35 | crate::arch::QWEN35_MOE => true,
            other => bail!(
                "infr-llama supports architecture={} (plus {}|{}), got {other:?}",
                crate::arch::TRANSFORMER.join("|"),
                crate::arch::QWEN35,
                crate::arch::QWEN35_MOE,
            ),
        };
        // Qwen2/2.5 bias their q/k/v projections (Qwen3 removed them); every other supported arch is
        // bias-free on attention. They also keep the HF rotate-half q/k row order (see the
        // `permute_qk_neox` field doc).
        let qkv_bias = arch == crate::arch::QWEN2;
        // bitnet, like qwen2, uses NEOX rope (llama.cpp maps LLM_ARCH_BITNET to LLAMA_ROPE_TYPE_NEOX)
        // and its GGUF keeps attn_q/attn_k in HF rotate-half order — permute the rows at load so the
        // no-qknorm interleaved `Op::Rope` reproduces NEOX (see the `permute_qk_neox` field doc).
        let permute_qk_neox = arch == crate::arch::QWEN2 || crate::arch::is_bitnet(&arch);
        // bitnet (BitNet b1.58) SubLN: two extra RMSNorms per layer — `attn_sub_norm` on the
        // concatenated-heads attention output BEFORE the o-projection, and `ffn_sub_norm` on the
        // FFN intermediate BEFORE `ffn_down`. Gates both the extra weight loads and the graph ops.
        let sub_norm = crate::arch::is_bitnet(&arch);
        // DeepSeek V1: the llama skeleton (NORM/interleaved rope, no bias, no qk-norm) plus a
        // softmax-gated MoE (no top-k renormalization, shared expert summed in plain). See
        // docs/deepseek.md § Stage 1. All the divergent semantics are HARDCODED in the reference
        // loader (`src/models/deepseek.cpp`), not metadata-driven.
        let deepseek = arch == crate::arch::DEEPSEEK;
        let deepseek2 = arch == crate::arch::DEEPSEEK2;
        // llama4 (Scout etc.): shares the llama attention skeleton (NORM/interleaved rope, no bias,
        // converter-permuted q/k) but adds a 16-expert sigmoid top-1 MoE + iRoPE (per-layer NoPE) +
        // a weightless post-rope Q/K L2-norm. All the divergent semantics are HARDCODED for `llama4`
        // in the reference loader (`models/llama4.cpp`), not metadata-driven.
        let llama4 = arch == crate::arch::LLAMA4;
        let diffusion_gemma = arch == crate::arch::DIFFUSION_GEMMA;
        // diffusion-gemma's backbone IS gemma4's (heterogeneous per-layer dims, V-norm,
        // freq_factors, softcap) verbatim — see docs/diffusion-gemma.md — so it folds into the same
        // gate as every other gemma4-shared parse below. `diffusion_gemma` gates only what's
        // actually different (dual FFN, canvas/mask fields, encoder-scalar tensor name).
        let gemma4 = arch == crate::arch::GEMMA4 || diffusion_gemma;
        let gemma = arch == crate::arch::GEMMA3 || gemma4;
        // gemma4 MoE (26B-A4B): the same gemma4 backbone with a routed-expert FFN — structurally
        // identical to diffusion-gemma's dual FFN (dense GeGLU ∥ 128-expert MoE, fused `gate_up_exps`
        // + per-expert `down_exps` scale + router-input rmsnorm/scale, summed), minus the diffusion
        // decode. Detected by expert tensors on a plain `gemma4` arch (dense E2B/E4B carry none).
        // Gates the `FfnW::DiffusionMoe` FFN build via `dual_moe()`; every other gemma4-shared field
        // (attention, softcap, `layer_output_scale`) is unchanged.
        let gemma4_moe =
            arch == crate::arch::GEMMA4 && meta_u64(g, &format!("{arch}.expert_count")).is_some();
        // qwen35moe (Qwen3.6 MoE) runs the identical DeltaNet/full-attention skeleton as dense
        // qwen35 — `qwen35` gates every shared field below; `qwen35_moe` alone gates the FFN
        // shape (routed-expert bank + shared expert vs plain dense).
        let qwen35_moe = arch == crate::arch::QWEN35_MOE;
        let qwen35 = arch == crate::arch::QWEN35 || qwen35_moe;
        let mk = |k: &str| format!("{arch}.{k}");
        // DeepSeek2-specific metadata: MLA dims, MoE gating function, YaRN (everything EXCEPT
        // "lite" detection, which needs n_layer — added after `n_layer` below).
        let (q_lora_rank, kv_lora_rank, key_length_mla, value_length_mla, qk_rope_dim) =
            if deepseek2 {
                let qlr = meta_u64(g, &mk("attention.q_lora_rank")).unwrap_or(0) as usize;
                let kvlr = meta_u64(g, &mk("attention.kv_lora_rank"))
                    .context("deepseek2.attention.kv_lora_rank")?
                    as usize;
                let klen = meta_u64(g, &mk("attention.key_length_mla"))
                    .unwrap_or((kvlr + 64) as u64) as usize;
                let vlen =
                    meta_u64(g, &mk("attention.value_length_mla")).unwrap_or(kvlr as u64) as usize;
                let qkr = meta_u64(g, &mk("rope.dimension_count")).unwrap_or(64) as usize;
                (qlr, kvlr, klen, vlen, qkr)
            } else {
                (0, 0, 0, 0, 0)
            };
        let (expert_gating_func, expert_weights_norm, rope_yarn_log_mul) = if deepseek2 {
            let gf = meta_u64(g, &mk("expert_gating_func")).unwrap_or(0) as u8;
            let norm = g
                .metadata()
                .get(&mk("expert_weights_norm"))
                .and_then(|v| match v {
                    MetaValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(false);
            let mut ylm =
                meta_f64(g, &mk("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0) as f32;
            if ylm != 0.0 {
                ylm /= 0.1_f32; // convert-script fix
            }
            (gf, norm, ylm)
        } else {
            (0, false, 0.0)
        };
        // DeepSeek V2+ YaRN: `rope.scaling.type` (string; `"yarn"` enables the full frequency
        // ramp), `rope.scaling.factor` (the interpolate factor; llama.cpp defaults it to 1.0 when
        // the GGUF omits it), and `rope.scaling.original_context_length` (the corr_dim ramp's
        // `n_ctx_orig`; falls back to `context_length` like llama.cpp's `n_ctx_orig_yarn`).
        let (rope_scaling_yarn, rope_scaling_factor, rope_scaling_orig_ctx) = if deepseek2 {
            let yarn = g
                .metadata()
                .get(&mk("rope.scaling.type"))
                .and_then(MetaValue::as_str)
                .is_some_and(|s| s == "yarn");
            let factor = meta_f64(g, &mk("rope.scaling.factor")).unwrap_or(1.0) as f32;
            let orig_ctx = meta_u64(g, &mk("rope.scaling.original_context_length"))
                .or_else(|| meta_u64(g, &mk("context_length")))
                .unwrap_or(8192) as usize;
            (yarn, factor, orig_ctx)
        } else {
            (false, 1.0, 0)
        };
        // Derived MLA dims from tensor shapes (NOT GGUF keys).
        let (head_k_mla, v_head_dim) = if deepseek2 {
            let hk_mla = key_length_mla
                .checked_sub(kv_lora_rank + qk_rope_dim)
                .unwrap_or(128);
            let vhd = value_length_mla.checked_sub(kv_lora_rank).unwrap_or(128);
            (hk_mla.max(1), vhd.max(1))
        } else {
            (0, 0)
        };
        let n_expert_groups = if deepseek2 {
            meta_u64(g, &mk("expert_group_count")).unwrap_or(0) as usize
        } else {
            0
        };
        let n_expert_groups_used = if deepseek2 {
            meta_u64(g, &mk("expert_group_used_count")).unwrap_or(0) as usize
        } else {
            0
        };
        let n_layer_all = meta_u64(g, &mk("block_count")).context("block_count")? as usize;
        // MTP/NextN (Qwen3.5/3.6, issue #33 — see docs/mtp.md): `{arch}.nextn_predict_layers`
        // extra decoder block(s) appended AFTER the trunk — `block_count` INCLUDES them. Ported
        // from the reference loader's `hparams.n_layer_nextn`
        // (`llama.cpp/src/models/qwen35.cpp::load_arch_hparams`). The confirmed 4B MTP GGUF sets
        // `block_count=33`/`nextn_predict_layers=1`: the trunk is 32 layers and `blk.32` is the
        // MTP head — without this split, `n_layer` below would include it and the trunk layer
        // loop (`seam.rs`'s `wload`) would misclassify `blk.32` as a gated-DeltaNet layer
        // (`(32+1) % full_attn_interval != 0`) and fail on missing `ssm_*` tensors.
        let n_layer_nextn = meta_u64(g, &mk("nextn_predict_layers")).unwrap_or(0) as usize;
        if n_layer_nextn > 0 && arch != crate::arch::QWEN35 && arch != crate::arch::QWEN35_MOE {
            bail!(
                "{arch}.nextn_predict_layers is only supported on arch={}|{} (MTP/NextN); got \
                 nextn_predict_layers={n_layer_nextn} on arch={arch:?}",
                crate::arch::QWEN35,
                crate::arch::QWEN35_MOE,
            );
        }
        // The reference caps qwen35 at a single MTP block (`GGML_ASSERT(hparams.n_layer_nextn ==
        // 1)` in `graph_mtp`'s ctor) — mirrored here so an unsupported wider value fails loudly
        // instead of silently misreading the tensor layout.
        if n_layer_nextn > 1 {
            bail!(
                "qwen35 MTP: nextn_predict_layers={n_layer_nextn} > 1 not supported (the \
                 reference implementation caps at a single MTP block — see docs/mtp.md)",
            );
        }
        if n_layer_nextn >= n_layer_all {
            bail!(
                "qwen35 MTP: nextn_predict_layers={n_layer_nextn} must be < block_count={n_layer_all}",
            );
        }
        let n_layer = n_layer_all - n_layer_nextn;
        // DeepSeek2 "lite" detection (tensor-presence test + heuristic fallback). Must run after
        // n_layer is known.
        let is_lite = deepseek2 && {
            g.tensors().iter().any(|t| t.name == "blk.0.attn_q.weight")
                || n_layer == 27
                || n_layer == 26
        };
        let n_embd = meta_u64(g, &mk("embedding_length")).context("embedding_length")? as usize;
        let n_head = positive_model_dimension(
            &mk("attention.head_count"),
            meta_u64(g, &mk("attention.head_count")).context("head_count")?,
        )?;
        // DeepSeek2 MLA: n_kv is always 1 (single key head shared by all query heads).
        let n_kv = if deepseek2 {
            1
        } else {
            match meta_u64(g, &mk("attention.head_count_kv")) {
                Some(value) => positive_model_dimension(&mk("attention.head_count_kv"), value)?,
                None => n_head,
            }
        };
        let n_ff_layers: Vec<usize> = if let Some(arr) = g
            .metadata()
            .get(&mk("feed_forward_length"))
            .and_then(MetaValue::as_arr)
        {
            arr.iter()
                .filter_map(MetaValue::as_u64)
                .map(|v| v as usize)
                .collect()
        } else if let Some(ff) = meta_u64(g, &mk("feed_forward_length")) {
            vec![ff as usize; n_layer]
        } else if qwen35_moe {
            // qwen35moe (Qwen3.6 MoE, routed-expert FFN on every layer) carries NO top-level
            // `feed_forward_length` — there's no dense `ffn_*` tensor anywhere in the GGUF, only
            // the routed bank (`ffn_*_exps`, sized by `expert_feed_forward_length` — parsed into
            // `MoeConfig` below) and the optional Qwen2-MoE-style shared expert (`ffn_*_shexp`,
            // sized by `expert_shared_feed_forward_length` — see `shexp_ff`). `n_ff`/`n_ff_layers`
            // only size the shared dense-FFN-SHAPED scratch (`gbuf`/`ubuf`/`actbuf` in
            // `seam::runner`), which the shared-expert branch reuses at ITS width; 0 (no scratch
            // needed) if the model has no shared expert either.
            let shexp = meta_u64(g, &mk("expert_shared_feed_forward_length")).unwrap_or(0) as usize;
            vec![shexp; n_layer]
        } else {
            bail!("{arch}.feed_forward_length missing (and not an array)");
        };
        let n_ff = n_ff_layers.iter().copied().max().unwrap_or(0);
        // diffusion-gemma's MoE shape (128 experts / 8 used, softmax gating, no extra routed-weight
        // scale) is parsed identically to qwen3moe's — only the FFN it feeds differs (dual FFN:
        // dense ∥ MoE, summed, vs qwen3moe's MoE-only), which is a `seam` graph-build detail.
        // qwen35moe's routed bank is parsed the SAME way too (same softmax + top-k-renormalize
        // gating, confirmed against llama.cpp's `qwen35moe.cpp::build_layer_ffn` — hardcoded
        // `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`/`norm_w=true`, identical to qwen3moe); only its
        // Qwen2-MoE-style SHARED expert (`shexp_ff` below) is genuinely new.
        let moe = if arch == crate::arch::QWEN3_MOE
            || diffusion_gemma
            || qwen35_moe
            || gemma4_moe
            || llama4
            || deepseek
            || deepseek2
        {
            let n_expert = meta_u64(g, &mk("expert_count")).context("expert_count")? as usize;
            let n_used =
                meta_u64(g, &mk("expert_used_count")).context("expert_used_count")? as usize;
            let n_ff_exp = meta_u64(g, &mk("expert_feed_forward_length"))
                .map(|v| v as usize)
                .unwrap_or(n_ff / n_used.max(1));
            // llama4 hardcodes SIGMOID gating, NO top-k renormalization, and weight-before-FFN
            // (`models/llama4.cpp::build_arch_graph` → `build_moe_ffn(.., norm_w=false, ..,
            // LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID)` with `weight_before_ffn = arch == LLAMA4`).
            // Its routed-weight scale is `expert_weights_scale` (absent on Scout → 1.0).
            // DeepSeek V1 hardcodes SOFTMAX gating, NO top-k renormalization, weight-after-FFN,
            // and scale from `expert_weights_scale` (default 0) — matching the reference's
            // `build_deepseek` (`src/models/deepseek.cpp`). Every other MoE arch here is softmax
            // + renormalized top-k, output-weighted, scale 1.0.
            let (gating, norm_w, weight_before, scale) = if llama4 {
                (
                    infr_core::graph::MoeGating::Sigmoid,
                    false,
                    true,
                    meta_f64(g, &mk("expert_weights_scale")).unwrap_or(1.0) as f32,
                )
            } else if deepseek2 {
                let gating = match expert_gating_func {
                    1 => infr_core::graph::MoeGating::Softmax,
                    2 => infr_core::graph::MoeGating::Sigmoid,
                    4 => infr_core::graph::MoeGating::SqrtSoftplus,
                    _ => infr_core::graph::MoeGating::Softmax, // type 0/3 default to softmax
                };
                (
                    gating,
                    expert_weights_norm,
                    false, // weight-before-FFN: always false for deepseek2
                    meta_f64(g, &mk("expert_weights_scale")).unwrap_or(0.0) as f32,
                )
            } else if deepseek {
                (
                    infr_core::graph::MoeGating::Softmax,
                    false,
                    false,
                    meta_f64(g, &mk("expert_weights_scale")).unwrap_or(0.0) as f32,
                )
            } else {
                (infr_core::graph::MoeGating::Softmax, true, false, 1.0)
            };
            Some(MoeConfig {
                n_expert,
                n_used,
                n_ff_exp,
                scale,
                gating,
                norm_w,
                weight_before,
                n_expert_groups: n_expert_groups as u32,
                n_expert_groups_used: n_expert_groups_used as u32,
            })
        } else {
            None
        };
        // The model's trained context length (its default max context). Default 8192 if absent.
        let n_ctx_train = meta_u64(g, &mk("context_length")).unwrap_or(8192) as usize;
        let head_dim = if deepseek2 {
            key_length_mla
        } else {
            meta_u64(g, &mk("attention.key_length")).unwrap_or((n_embd / n_head) as u64) as usize
        };
        let rope_dim = if deepseek2 {
            qk_rope_dim
        } else {
            meta_u64(g, &mk("rope.dimension_count")).unwrap_or(head_dim as u64) as usize
        };
        let rope_theta = g
            .metadata()
            .get(&mk("rope.freq_base"))
            .and_then(|v| match v {
                MetaValue::F64(f) => Some(*f as f32),
                MetaValue::U64(u) => Some(*u as f32),
                _ => None,
            })
            // qwen35's GGUF sets this explicitly (1e7); the fallback only matters if it's absent.
            .unwrap_or(if qwen35 { 1e7 } else { 10000.0 });
        let rms_eps = g
            .metadata()
            .get(&mk("attention.layer_norm_rms_epsilon"))
            .and_then(|v| match v {
                MetaValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            // qwen35's old seam (`Cfg::from_gguf`) defaults this to 1e-6, not the generic 1e-5.
            .unwrap_or(if qwen35 { 1e-6 } else { 1e-5 });
        let swa_window = if gemma {
            meta_u64(g, &mk("attention.sliding_window")).unwrap_or(0) as usize
        } else {
            0
        };
        let swa_pattern = if swa_window == 0 {
            0
        } else if let Some(arr) = g
            .metadata()
            .get(&mk("attention.sliding_window_pattern"))
            .and_then(MetaValue::as_arr)
        {
            arr.iter()
                .position(|v| matches!(v, MetaValue::Bool(false)))
                .map(|i| i + 1)
                .unwrap_or(6)
        } else {
            meta_u64(g, &mk("attention.sliding_window_pattern")).unwrap_or(6) as usize
        };
        let swa_rope_theta = if swa_window > 0 {
            g.metadata()
                .get(&mk("rope.freq_base_swa"))
                .and_then(|v| match v {
                    MetaValue::F64(f) => Some(*f as f32),
                    MetaValue::U64(u) => Some(*u as f32),
                    _ => None,
                })
                .unwrap_or(10000.0)
        } else {
            rope_theta
        };
        let (head_dim, n_kv, rope_dim, head_dim_swa, n_kv_swa, rope_dim_swa) = if gemma4 {
            let kv_key = mk("attention.head_count_kv");
            let hk = g.metadata().get(&kv_key).and_then(MetaValue::as_arr);
            let full_idx = swa_pattern.saturating_sub(1);
            let hd_full =
                meta_u64(g, &mk("attention.key_length")).unwrap_or(head_dim as u64) as usize;
            let hd_swa =
                meta_u64(g, &mk("attention.key_length_swa")).unwrap_or(hd_full as u64) as usize;
            let rd_full =
                meta_u64(g, &mk("rope.dimension_count")).unwrap_or(hd_full as u64) as usize;
            let rd_swa =
                meta_u64(g, &mk("rope.dimension_count_swa")).unwrap_or(hd_swa as u64) as usize;
            (
                hd_full,
                positive_array_model_dimension(hk, &kv_key, full_idx)?.unwrap_or(n_kv),
                rd_full,
                hd_swa,
                positive_array_model_dimension(hk, &kv_key, 0)?.unwrap_or(n_kv),
                rd_swa,
            )
        } else {
            (head_dim, n_kv, rope_dim, head_dim, n_kv, rope_dim)
        };
        let final_softcap = if gemma4 {
            g.metadata()
                .get(&mk("final_logit_softcapping"))
                .and_then(MetaValue::as_f64)
                .unwrap_or(0.0) as f32
        } else {
            0.0
        };
        let n_embd_per_layer = if gemma4 {
            meta_u64(g, &mk("embedding_length_per_layer_input")).unwrap_or(0) as usize
        } else {
            0
        };
        let n_layer_kv_from_start = if gemma4 {
            let shared = meta_u64(g, &mk("attention.shared_kv_layers")).unwrap_or(0) as usize;
            n_layer.saturating_sub(shared)
        } else {
            n_layer
        };
        let eos = meta_u64(g, "tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;
        // vocab = token_embd rows (GGUF shape `[n_embd, vocab]`) — read from the tensor header, no load.
        let vocab = g
            .tensors()
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.last().copied())
            .context("token_embd.weight shape")?;
        // qwen35 (gated-DeltaNet hybrid) extras — ports of the old seam's `qwen35::Cfg::from_gguf`.
        // These metadata keys sit directly under `qwen35.*` (NOT `qwen35.attention.*`), matching the
        // old seam's parser exactly.
        let full_attn_interval = if qwen35 {
            meta_u64(g, &mk("full_attention_interval")).unwrap_or(4) as usize
        } else {
            0
        };
        let (ssm_d_conv, ssm_d_state, ssm_d_inner, ssm_n_group, ssm_dt_rank) = if qwen35 {
            (
                meta_u64(g, &mk("ssm.conv_kernel")).context("qwen35 ssm.conv_kernel")? as usize,
                meta_u64(g, &mk("ssm.state_size")).context("qwen35 ssm.state_size")? as usize,
                meta_u64(g, &mk("ssm.inner_size")).context("qwen35 ssm.inner_size")? as usize,
                meta_u64(g, &mk("ssm.group_count")).context("qwen35 ssm.group_count")? as usize,
                meta_u64(g, &mk("ssm.time_step_rank")).context("qwen35 ssm.time_step_rank")?
                    as usize,
            )
        } else {
            (0, 0, 0, 0, 0)
        };
        let rope_sections: [u32; 4] = if qwen35 {
            let mut s = [0u32; 4];
            if let Some(arr) = g
                .metadata()
                .get(&mk("rope.dimension_sections"))
                .and_then(MetaValue::as_arr)
            {
                for (i, v) in arr.iter().take(4).enumerate() {
                    s[i] = v.as_u64().unwrap_or(0) as u32;
                }
            }
            s
        } else {
            [0u32; 4]
        };
        let attn_out_gate = qwen35;
        // DeepSeek: first N layers are dense FFN, the rest are MoE.
        let n_layer_dense_lead = if deepseek || deepseek2 {
            meta_u64(g, &mk("leading_dense_block_count")).unwrap_or(0) as usize
        } else {
            0
        };
        // Shared-expert FFN width. qwen35moe reads `expert_shared_feed_forward_length`; llama4 has
        // NO such key — its shared expert is the SAME width as a routed expert
        // (`models/llama4.cpp`: `n_ff_shexp = n_ff_exp`). DeepSeek's shared expert is
        // `n_ff_exp * n_expert_shared` wide (llama.cpp fuses `n_expert_shared` experts into one
        // wider branch — see docs/deepseek.md § Stage 1). `0` = no shared expert.
        let shexp_ff = if qwen35_moe {
            meta_u64(g, &mk("expert_shared_feed_forward_length")).unwrap_or(0) as usize
        } else if llama4 {
            moe.map(|m| m.n_ff_exp).unwrap_or(0)
        } else if deepseek || deepseek2 {
            let n_shared = meta_u64(g, &mk("expert_shared_count")).unwrap_or(0) as usize;
            moe.map(|m| m.n_ff_exp * n_shared).unwrap_or(0)
        } else {
            0
        };
        // llama4 shared expert is summed in PLAIN (no per-token gate); qwen35moe gates by sigmoid.
        // DeepSeek's shared expert is also plain-summed (no gate).
        let shexp_gated = qwen35_moe;
        // llama4 iRoPE + MoE interleave. `interleave_moe_layer_step` defaults to 1 (every layer MoE,
        // as on Scout). The NoPE step: the reference forces the chunked-attention branch whenever
        // `attention.sliding_window` is absent or nonzero, in which case `n_no_rope_layer_step`
        // stays its default 4; only an explicit `sliding_window == 0` disables NoPE (all-rope). The
        // attention temperature params (floor 8192, scale 0.1, offset 1.0) and the 8192 chunk size
        // are likewise hardcoded there — both are exact no-ops below 8192 tokens, so infr's CPU
        // path (which can't reach that length on a 109B model) leaves them unimplemented; see the
        // arch doc.
        let (moe_interleave_step, no_rope_step, kq_l2norm) = if llama4 {
            let step = meta_u64(g, &mk("interleave_moe_layer_step")).unwrap_or(1) as usize;
            let no_rope = match meta_u64(g, &mk("attention.sliding_window")) {
                Some(0) => 0, // all-rope (swa_type NONE)
                _ => 4,       // chunked branch → default no_rope_layer_step
            };
            // `use_kq_norm = type != LLM_TYPE_17B_128E` (i.e. n_expert != 128).
            let l2 = moe.map(|m| m.n_expert != 128).unwrap_or(true);
            (step, no_rope, l2)
        } else {
            (0, 0, false)
        };
        // diffusion-gemma's canvas/mask keys sit at the TOP level (`diffusion.*` /
        // `tokenizer.ggml.*`), not namespaced under `{arch}.*` like everything else above —
        // matches the reference loader (`diffusion-gemma.cpp: load_arch_hparams`) verbatim.
        let canvas_length = if diffusion_gemma {
            let cl =
                meta_u64(g, "diffusion.canvas_length").context("diffusion.canvas_length")? as usize;
            if cl == 0 {
                bail!("DiffusionGemma requires a positive diffusion.canvas_length");
            }
            cl
        } else {
            0
        };
        let mask_token_id = if diffusion_gemma {
            meta_u64(g, "tokenizer.ggml.mask_token_id").unwrap_or(4) as u32
        } else {
            0
        };
        // Entropy-bound sampler params (`diffusion.eb_*`), fallbacks matching
        // `diffusion-cli.cpp`'s `meta_f`/`meta_i` defaults exactly.
        let (
            eb_max_steps,
            eb_t_min,
            eb_t_max,
            eb_entropy_bound,
            eb_stability_threshold,
            eb_confidence_threshold,
        ) = if diffusion_gemma {
            (
                meta_u64(g, "diffusion.eb_max_steps").unwrap_or(48) as usize,
                meta_f64(g, "diffusion.eb_t_min").unwrap_or(0.4) as f32,
                meta_f64(g, "diffusion.eb_t_max").unwrap_or(0.8) as f32,
                meta_f64(g, "diffusion.eb_entropy_bound").unwrap_or(0.1) as f32,
                meta_u64(g, "diffusion.eb_stability_threshold").unwrap_or(1) as usize,
                meta_f64(g, "diffusion.eb_confidence_threshold").unwrap_or(0.005) as f32,
            )
        } else {
            (0, 0.0, 0.0, 0.0, 0, 0.0)
        };
        Ok(Config {
            n_layer,
            n_head,
            n_kv,
            n_embd,
            n_ff,
            n_ff_layers,
            n_embd_per_layer,
            n_layer_kv_from_start,
            head_dim,
            rope_dim,
            rope_theta,
            rms_eps,
            vocab,
            eos,
            eos_ids: vec![eos],
            qk_norm,
            qkv_bias,
            permute_qk_neox,
            gemma,
            gemma4,
            diffusion_gemma,
            gemma4_moe,
            canvas_length,
            mask_token_id,
            eb_max_steps,
            eb_t_min,
            eb_t_max,
            eb_entropy_bound,
            eb_stability_threshold,
            eb_confidence_threshold,
            head_dim_swa,
            n_kv_swa,
            rope_dim_swa,
            final_softcap,
            swa_window,
            swa_pattern,
            swa_rope_theta,
            moe,
            n_ctx_train,
            qwen35,
            full_attn_interval,
            ssm_d_conv,
            ssm_d_state,
            ssm_d_inner,
            ssm_n_group,
            ssm_dt_rank,
            rope_sections,
            attn_out_gate,
            n_layer_nextn,
            shexp_ff,
            llama4,
            moe_interleave_step,
            no_rope_step,
            kq_l2norm,
            shexp_gated,
            sub_norm,
            deepseek,
            n_layer_dense_lead,
            deepseek2,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_dim,
            key_length: key_length_mla,
            head_k_mla,
            v_head_dim,
            expert_gating_func,
            n_expert_groups,
            n_expert_groups_used,
            rope_yarn_log_mul,
            rope_scaling_yarn,
            rope_scaling_factor,
            rope_scaling_orig_ctx,
            is_lite,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_str(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_str(bytes, key);
        push_u32(bytes, 4); // GGUF_TYPE_UINT32
        push_u32(bytes, value);
    }

    fn gemma4_zero_kv_heads_fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747); // GGUF magic
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1); // tensor count
        push_u64(&mut bytes, 6); // metadata count

        push_str(&mut bytes, "general.architecture");
        push_u32(&mut bytes, 8); // GGUF_TYPE_STRING
        push_str(&mut bytes, "gemma4");
        push_u32_metadata(&mut bytes, "gemma4.block_count", 1);
        push_u32_metadata(&mut bytes, "gemma4.embedding_length", 8);
        push_u32_metadata(&mut bytes, "gemma4.attention.head_count", 8);
        push_u32_metadata(&mut bytes, "gemma4.feed_forward_length", 16);

        push_str(&mut bytes, "gemma4.attention.head_count_kv");
        push_u32(&mut bytes, 9); // GGUF_TYPE_ARRAY
        push_u32(&mut bytes, 4); // GGUF_TYPE_UINT32
        push_u64(&mut bytes, 1);
        push_u32(&mut bytes, 0);

        push_str(&mut bytes, "token_embd.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 8);
        push_u64(&mut bytes, 1);
        push_u32(&mut bytes, 0); // F32
        push_u64(&mut bytes, 0);
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0; 32]);
        bytes
    }

    #[test]
    fn model_dimensions_must_be_positive() {
        assert!(positive_model_dimension("test.head_count", 0).is_err());
        assert_eq!(positive_model_dimension("test.head_count", 1).unwrap(), 1);
    }

    #[test]
    fn gemma4_array_kv_heads_must_be_positive() {
        let path = std::env::temp_dir().join(format!(
            "infr-llama-gemma4-zero-kv-heads-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, gemma4_zero_kv_heads_fixture()).expect("write GGUF fixture");
        let gguf = Gguf::open(&path).expect("open GGUF fixture");
        std::fs::remove_file(path).expect("remove GGUF fixture");

        let Err(err) = Config::from_gguf(&gguf) else {
            panic!("zero KV heads must fail");
        };
        assert_eq!(
            err.to_string(),
            "gemma4.attention.head_count_kv[0] must be positive"
        );
    }
}
