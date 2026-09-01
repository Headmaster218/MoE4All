//! Model hyper-parameters parsed from GGUF metadata. Mechanically split out of `lib.rs`.
use crate::{meta_f64, meta_u64, MoeConfig};
use anyhow::{bail, Context, Result};
use infr_core::graph::QSA_MAX_TOP_BLOCKS;
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
    /// Qwen3.8 Flash Next (`qwen4exp`): Qwen-style recurrent/full attention wrapped in four
    /// low-rank gated residual streams, plus PLE and a routed/shared-expert MoE.
    pub qwen4exp: bool,
    /// Ling 3.0 Flash (`bailingmoe3`): hybrid KDA/MLA mixer and DeepSeek-style MoE.
    pub bailingmoe3: bool,
    /// Per-layer mixer selector for `bailingmoe3`: true is MLA, false is KDA. Empty elsewhere.
    pub bailing_mla_layers: Vec<bool>,
    /// Ling KDA per-head Q/K/V width and lower bound used by its log-decay gate.
    pub kda_head_dim: usize,
    pub kda_gate_lower_bound: f32,
    /// qwen35: attention layers sit at `i` where `(i+1) % full_attn_interval == 0`; every other
    /// layer is gated-DeltaNet linear attention. `0` for every non-qwen35 model (never read).
    pub full_attn_interval: usize,
    /// Explicit per-layer recurrent selector used by qwen4exp GGUFs. Empty on other arches.
    pub recurrent_layers: Vec<bool>,
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
    /// DeepSeek V2+: `true` for `arch == "deepseek2"` AND for `arch == "deepseek32"` (V3.2 is the
    /// same skeleton plus an indexer — see [`Config::deepseek32`]). Gates MLA attention (absorbed
    /// form with asymmetric K/V dims + single K row), YaRN rope scaling, and group-limited MoE
    /// routing with router bias correction. See docs/deepseek.md § Stage 2.
    pub deepseek2: bool,
    /// DeepSeek V3.2: `true` only for `arch == "deepseek32"`, which also sets
    /// [`Config::deepseek2`] (every MLA/KV/MoE gate is shared verbatim). This flag alone gates what
    /// V3.2 adds: the three `indexer_*` hyperparameters below, the five per-layer lightning-indexer
    /// tensors the loader consumes, and the keys the reference reads UNCONDITIONALLY where
    /// `deepseek2` tolerates their absence (`attention.q_lora_rank`, `expert_gating_func`) plus its
    /// refusal of a non-MLA file. See docs/deepseek.md § Stage 3.
    pub deepseek32: bool,
    /// DeepSeek V4: `true` only for `arch == "deepseek4"`, and it sets NEITHER [`Config::deepseek2`]
    /// nor [`Config::deepseek32`] — V4 is not MLA (no `kv_lora_rank`, no `wk_b`/`wv_b`), so every
    /// gate those two drive is wrong for it. What V4 shares with the family is the MoE block, the
    /// shared expert, the Q-LoRA projection, the norms and the generic rope/embedding plumbing;
    /// everything else is new and lives on the `deepseek4`-only fields below. See
    /// docs/deepseek.md § Stage 4.
    ///
    /// **Every V4 layer is sliding-window** — `deepseek4.cpp::load_arch_hparams` ends with
    /// `set_swa_pattern(0)`, which marks every layer SWA rather than interleaving. [`Config::
    /// is_swa_layer`] special-cases this arch for that reason: `swa_pattern` encodes "every p-th
    /// layer is FULL", and V4 has no full layers to point at.
    ///
    /// The seam emits all three V4 attention tiers and keeps their compressed caches in the
    /// architecture-specific mixed-FP8/MXFP4 layout.
    pub deepseek4: bool,
    /// DeepSeek V4 (`{arch}.attention.compress_ratios`): the per-layer compression ratio, and the
    /// MASTER per-layer switch — which caches a layer keeps and which tensors it carries both hang
    /// off it. Only `{0, 4, 128}` are accepted (`deepseek4.cpp::load_arch_tensors` throws on
    /// anything else): `0` is a pure sliding-window layer with no compressor at all, `4` adds the
    /// CSA compressor plus the lightning indexer, `128` adds the HCA compressor alone.
    ///
    /// qwen4exp reuses the same key with a DIFFERENT meaning: it is the QSA BLOCK factor, so
    /// `{0, 4}` only (`0` = no block compression, `4` = pool every 4 indexer keys into one scored
    /// block key). `0` is mandatory on recurrent layers. On full-attention layers an all-zero
    /// converter defect is restored to the official ratio 4; mixed 0/4 metadata is rejected. Read
    /// through [`Config::layer_compress_ratio`], which is the accessor every consumer should use.
    /// Empty for every model that is neither `deepseek4` nor qwen4exp.
    pub compress_ratios: Vec<usize>,
    /// DeepSeek V4 (`{arch}.swiglu_clamp_exp`): per-layer clamp on the routed experts' SwiGLU. The
    /// GGUF may carry ONE value (broadcast to every layer) or a full array, which is why this is a
    /// `Vec` even for a uniform model — same shape as [`Config::n_ff_layers`]. Empty for every
    /// non-`deepseek4` model. **V4 clamps the gate PRE-SiLU** where every other arch clamps
    /// post-SiLU; the graph slice owns that difference.
    pub swiglu_clamp_exp: Vec<f32>,
    /// DeepSeek V4 (`{arch}.swiglu_clamp_shexp`): the same clamp for the SHARED expert. The
    /// reference falls back to [`Config::swiglu_clamp_exp`] when the key is absent, so these two are
    /// equal on a GGUF that declares only the routed one. Empty for every non-`deepseek4` model.
    pub swiglu_clamp_shexp: Vec<f32>,
    /// DeepSeek V4 (`{arch}.hash_layer_count`): the first this-many layers are HASH-routed — they
    /// carry a `ffn_gate_tid2eid` token-id→expert-id table INSTEAD of the `exp_probs_b` router bias
    /// every later layer carries. `0` for every non-`deepseek4` model.
    pub hash_layer_count: usize,
    /// DeepSeek V4 (`{arch}.attention.output_group_count`): how many groups the low-rank output
    /// projection splits the concatenated heads into (`wo_a` is
    /// `[n_head * head_dim / this, o_lora_rank * this]`). `0` for every non-`deepseek4` model.
    pub o_group_count: usize,
    /// DeepSeek V4 (`{arch}.attention.output_lora_rank`): the rank of that output projection's
    /// bottleneck, per group. `0` for every non-`deepseek4` model.
    pub o_lora_rank: usize,
    /// DeepSeek V4 (`{arch}.attention.compress_rope_freq_base`): the rope base the COMPRESSED
    /// (ratio 4 / 128) layers use, with YaRN. Ratio-0 layers use plain unscaled rope at
    /// [`Config::rope_theta`]. `0.0` for every non-`deepseek4` model.
    pub compress_rope_theta: f32,
    /// DeepSeek V4 (`{arch}.hyper_connection.count`): how many parallel residual streams the
    /// hyper-connection block carries. Every per-layer `hc_*` tensor and the model-level
    /// `output_hc_*` triple are shaped by it. `0` for every non-`deepseek4` model.
    pub hc_mult: usize,
    /// Qwen3.8's low-rank gated-residual bottleneck. DeepSeek V4 uses full-rank Sinkhorn HC and
    /// leaves this zero.
    pub hc_low_rank: usize,
    /// DeepSeek V4 (`{arch}.hyper_connection.sinkhorn_iterations`): how many Sinkhorn normalisation
    /// rounds make the stream-mixing matrix approximately doubly stochastic. `0` for every
    /// non-`deepseek4` model.
    pub hc_sinkhorn_iters: usize,
    /// DeepSeek V4 (`{arch}.hyper_connection.epsilon`): the epsilon added inside that Sinkhorn
    /// normalisation. `0.0` for every non-`deepseek4` model.
    pub hc_eps: f32,
    /// Qwen3.8 PLE metadata. The enormous `per_layer_token_embd` tensor remains mmap-backed;
    /// these fields are enough to hash and fetch its rows without materializing the table.
    pub ple_layers: Vec<bool>,
    pub ple_ngram_size: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_head_dim: usize,
    pub ple_conv_kernel: usize,
    pub ple_eos: u32,
    pub ple_layer_multipliers: Vec<u64>,
    pub ple_head_offsets: Vec<u64>,
    pub ple_head_vocab_sizes: Vec<u64>,
    /// DeepSeek lightning indexer: number of indexer QUERY heads
    /// (`{arch}.attention.indexer.head_count`). One KEY head is shared by all of them (MQA) on V3.2;
    /// on V4 the keys come from the compressor instead. `0` for every model that is neither
    /// `deepseek32` nor `deepseek4`.
    pub indexer_n_head: usize,
    /// DeepSeek lightning indexer: per-head key/query width
    /// (`{arch}.attention.indexer.key_length`), the width of V3.2's `indexer.k_norm` and of one
    /// indexer KV-cache row. `0` for every model that is neither `deepseek32` nor `deepseek4`.
    pub indexer_head_size: usize,
    /// DeepSeek lightning indexer: how many keys the indexer's top-k keeps for the real attention
    /// (`{arch}.attention.indexer.top_k`). V4 counts COMPRESSED BLOCKS here, not tokens, because its
    /// indexer keys come from the compressor. `0` for every model that is neither `deepseek32` nor
    /// `deepseek4`.
    pub indexer_top_k: usize,
    /// llama.cpp's `hparams.f_norm_eps` — the epsilon of a non-RMS LayerNorm, DISTINCT from
    /// [`Config::rms_eps`] (`f_norm_rms_eps`, read from the GGUF). The lightning indexer's
    /// `k_norm` is the only mean-centred norm anywhere in this family, and `deepseek32.cpp`
    /// HARDCODES its epsilon rather than reading a key. `0.0` for every other arch — none of them
    /// emits `Op::LayerNorm` at all.
    pub norm_eps: f32,
    /// DeepSeek V2+ MLA: Q LoRA rank (wq_a output dim, q_a_norm dim, wq_b input dim). `0` when the
    /// "lite" variant carries a direct `wq` instead.
    pub q_lora_rank: usize,
    /// DeepSeek V2+ MLA: KV LoRA rank (wkv_a_mqa latent dim, attn_kv_a_norm dim, wk_b/wv_b dim-0).
    pub kv_lora_rank: usize,
    /// DeepSeek V2+ MLA: RoPE dimension on Q and K (q_pe/k_pe width = rope.dimension_count).
    pub qk_rope_dim: usize,
    /// DeepSeek V2+ MLA: the `attention.key_length_mla` GGUF key verbatim — llama.cpp's
    /// `n_embd_head_k_mla`, the FULL per-head key width (192 on V2-Lite), which is also what
    /// [`Config::head_dim`] carries for these models.
    ///
    /// NOT the cached-row width: that is `kv_lora_rank + qk_rope_dim` (576 on V2-Lite), which
    /// `seam::kv_row_elems` and `seam::runner`'s MLA arm each derive from those two fields. Nothing
    /// reads this field today.
    pub key_length: usize,
    /// DeepSeek V2+ MLA: llama.cpp's `n_embd_head_qk_nope` — the per-head NOPE (non-rotary) width
    /// of Q and of `wk_b`'s dim-0, 128 for V2-Lite and V3. Derived as
    /// `attention.key_length_mla - rope.dimension_count`, matching `src/models/deepseek2.cpp`.
    /// Despite the name this is NOT llama.cpp's `n_embd_head_k_mla` (the FULL per-head key width,
    /// 192 for V2-Lite, which is the `attention.key_length_mla` key itself). `0` for
    /// non-DeepSeek2 models.
    pub head_k_mla: usize,
    /// DeepSeek V2+ MLA: per-head V output dim (128 for V2-Lite and V3) — llama.cpp's
    /// `n_embd_head_v_mla`, which is the `attention.value_length_mla` GGUF key used as-is
    /// (`llama_hparams::n_embd_head_v_mla`). `0` for non-DeepSeek2 models.
    pub v_head_dim: usize,
    /// DeepSeek V2+ MoE: gating function enum (1=softmax, 2=sigmoid, 3=softmax-on-weights,
    /// 4=sqrt-softplus). Parsed from `expert_gating_func`, mapped to `MoeConfig::gating` — 3 has
    /// no `MoeGating` and is REFUSED at load rather than approximated by 1.
    /// `0` (the key absent, softmax by llama.cpp's compatibility rule) for non-DeepSeek2 models
    /// (and V1, which hardcodes softmax).
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
    /// DeepSeek V2+ YaRN: `rope.scaling.attn_factor` — llama.cpp's `hparams.rope_attn_factor`,
    /// folded into `cparams.yarn_attn_factor` (`llama-context.cpp`) BEFORE the deepseek2 builder
    /// recovers `attn_factor_org` from it, so it lands inside the squared mscale of the MLA score
    /// scale. `1.0` (llama.cpp's default) when the GGUF omits it.
    pub rope_attn_factor: f32,
    /// DeepSeek V2+ YaRN: `rope.scaling.yarn_beta_fast` — the ramp's high-frequency corner, the
    /// `beta_fast` of `ggml_rope_yarn_corr_dims`'s START dim. Defaults to llama.cpp's
    /// `llama_hparams::yarn_beta_fast`.
    pub rope_yarn_beta_fast: f32,
    /// DeepSeek V2+ YaRN: `rope.scaling.yarn_beta_slow` — the ramp's low-frequency corner, the
    /// `beta_slow` of `ggml_rope_yarn_corr_dims`'s END dim. Defaults to llama.cpp's
    /// `llama_hparams::yarn_beta_slow`.
    pub rope_yarn_beta_slow: f32,
    /// DeepSeek V2+ "lite" variant: `wq` present ⇒ lite (no wq_a/q_a_norm/wq_b). Detected by
    /// tensor presence, not GGUF key.
    pub is_lite: bool,
}

/// The QSA block ratio Qwen3.8-Flash-Next is trained with (`indexer_compress_ratio: 4` in the
/// model's `config.json`), and the ONLY value the all-zero `attention.compress_ratios` repair is
/// allowed to restore. It is a model structure parameter, not a kernel budget knob: guessing a
/// ratio that merely happens to fit [`QSA_MAX_TOP_BLOCKS`] would let a damaged file load and then
/// decode nonsense, so the repair never infers 1/2/8 — it either restores 4 or refuses to load.
const QWEN38_QSA_BLOCK_RATIO: usize = 4;

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
    ///
    /// deepseek4 is neither: `deepseek4.cpp::load_arch_hparams` calls `set_swa_pattern(0)`, whose
    /// `n_pattern == 0 || …` arm marks EVERY layer sliding-window. `swa_pattern` cannot express
    /// that — it names the period at which a layer is FULL, and V4 has no full layers — so the arch
    /// is checked directly. Its long-range recall comes from the compressed caches instead
    /// ([`Config::layer_compress_ratio`]), not from a wider window on some layers.
    pub fn is_swa_layer(&self, il: usize) -> bool {
        if self.deepseek4 {
            return self.swa_window > 0;
        }
        self.swa_window > 0 && self.swa_pattern > 1 && !(il + 1).is_multiple_of(self.swa_pattern)
    }

    /// The per-layer compression ratio (`{arch}.attention.compress_ratios[il]`) — for deepseek4
    /// the master switch over which caches and which tensors layer `il` has (one of `{0, 4, 128}`,
    /// where `0` means pure sliding window with no compressor), and for qwen4exp the QSA block
    /// factor (`0` = no block compression, `4` = one scored block key per 4 indexer keys).
    /// `0` for every layer of every other arch.
    pub fn layer_compress_ratio(&self, il: usize) -> usize {
        self.compress_ratios.get(il).copied().unwrap_or(0)
    }

    /// deepseek4: whether layer `il` is HASH-routed — it carries a `ffn_gate_tid2eid` token-id →
    /// expert-id table and NO `exp_probs_b` router bias. `false` for every layer of every other
    /// arch (`hash_layer_count == 0`).
    pub fn is_hash_moe_layer(&self, il: usize) -> bool {
        il < self.hash_layer_count
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
    /// layers on a fixed step (`(il+1) % step == 0`). DeepSeek V1/V2/V3.2 use a threshold: the first
    /// `n_layer_dense_lead` layers are dense, the rest are MoE — V4 has no dense-lead layers at all
    /// and rides the every-layer branch. `false` for dense models (`self.moe` is `None`).
    pub fn is_moe_layer(&self, il: usize) -> bool {
        self.moe.is_some()
            && if self.deepseek || self.deepseek2 || self.bailingmoe3 {
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

    /// Whether a Qwen hybrid layer is full attention rather than recurrent. The qwen35 helper is
    /// retained for its existing callers; qwen4exp uses an explicit per-layer vector when present.
    pub fn is_qwen_hybrid_attn_layer(&self, il: usize) -> bool {
        self.is_qwen35_attn_layer(il)
            || (self.qwen4exp
                && !self.recurrent_layers.get(il).copied().unwrap_or_else(|| {
                    self.full_attn_interval > 0 && !(il + 1).is_multiple_of(self.full_attn_interval)
                }))
    }

    pub fn is_ple_layer(&self, il: usize) -> bool {
        self.qwen4exp && self.ple_layers.get(il).copied().unwrap_or(false)
    }

    /// Whether Ling layer `il` uses MLA rather than KDA.
    pub fn is_bailing_mla_layer(&self, il: usize) -> bool {
        self.bailingmoe3 && self.bailing_mla_layers.get(il).copied().unwrap_or(false)
    }

    pub fn is_mla_layer(&self, il: usize) -> bool {
        self.deepseek2 || self.is_bailing_mla_layer(il)
    }

    pub fn is_recurrent_layer(&self, il: usize) -> bool {
        (self.qwen35 && !self.is_qwen35_attn_layer(il))
            || (self.qwen4exp && self.recurrent_layers.get(il).copied().unwrap_or(false))
            || (self.bailingmoe3 && !self.is_bailing_mla_layer(il))
    }

    pub fn recurrent_conv_channels(&self) -> usize {
        if self.bailingmoe3 {
            3 * self.n_head * self.kda_head_dim
        } else {
            self.q35_conv_channels()
        }
    }

    pub fn recurrent_state_elems(&self) -> usize {
        if self.bailingmoe3 {
            self.n_head * self.kda_head_dim * self.kda_head_dim
        } else {
            self.q35_num_v_heads() * self.q35_head_k_dim() * self.q35_head_v_dim()
        }
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
            | crate::arch::DEEPSEEK32
            | crate::arch::DEEPSEEK4
            | crate::arch::BAILINGMOE3
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
            crate::arch::QWEN35 | crate::arch::QWEN35_MOE | crate::arch::QWEN4EXP => true,
            other => bail!(
                "infr-llama supports architecture={} (plus {}|{}|{}), got {other:?}",
                crate::arch::TRANSFORMER.join("|"),
                crate::arch::QWEN35,
                crate::arch::QWEN35_MOE,
                crate::arch::QWEN4EXP,
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
        // DeepSeek V3.2 IS deepseek2 plus the lightning indexer: the same absorbed MLA, the same
        // one-compressed-row KV geometry, the same group-limited MoE with router bias, the same
        // dense-lead threshold and the same YaRN. So `deepseek2` — which is what every one of those
        // gates reads, here and in `seam` — widens to cover it, exactly as `gemma4` widens over
        // `diffusion-gemma` and `qwen35` over `qwen35moe` below. `deepseek32` alone gates the parts
        // that genuinely differ (mandatory MLA / q_lora_rank / expert_gating_func, the indexer).
        let deepseek32 = arch == crate::arch::DEEPSEEK32;
        let deepseek2 = arch == crate::arch::DEEPSEEK2 || deepseek32;
        // DeepSeek V4 does NOT widen `deepseek2`, and that is the whole point: V4 is not MLA. It
        // has no `kv_lora_rank`, no `wk_b`/`wv_b` and no compressed-KV row, so every `deepseek2`
        // gate — the MLA mixer, `kv_row_elems`' 576-wide row, the f16-only cache, the group-limited
        // router — is wrong for it. Sharing is limited to the MoE block, the shared expert, the
        // Q-LoRA projection and generic plumbing, each of which is picked up by name below.
        let deepseek4 = arch == crate::arch::DEEPSEEK4;
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
        let qwen4exp = arch == crate::arch::QWEN4EXP;
        let bailingmoe3 = arch == crate::arch::BAILINGMOE3;
        let mla_arch = deepseek2 || bailingmoe3;
        let mk = |k: &str| format!("{arch}.{k}");
        // DeepSeek2-specific metadata: MLA dims, MoE gating function, YaRN (everything EXCEPT
        // "lite" detection, which needs n_layer — added after `n_layer` below).
        let (q_lora_rank, kv_lora_rank, key_length_mla, value_length_mla, qk_rope_dim) = if mla_arch
        {
            if deepseek32 {
                // `deepseek32.cpp::load_arch_tensors` opens with
                // `if (!hparams.is_mla()) throw "DEEPSEEK32 architecture requires MLA"`, and
                // `llama_hparams::is_mla()` is "both MLA head-length keys are non-zero" (they
                // default to 0 when absent). V3.2 has NO unabsorbed fallback at all, so say
                // that, rather than letting the generic deepseek2 message below describe a
                // derivation this arch never had.
                for key in ["attention.key_length_mla", "attention.value_length_mla"] {
                    if meta_u64(g, &mk(key)).unwrap_or(0) == 0 {
                        bail!(
                            "deepseek32 architecture requires MLA: {} is missing or zero",
                            mk(key)
                        );
                    }
                }
            }
            // `deepseek32.cpp` reads `q_lora_rank` unconditionally — V3.2 has no "lite"
            // variant carrying a direct `wq`, so a file without the key is not a lite model,
            // it is a broken one.
            let qlr = if deepseek32 {
                meta_u64(g, &mk("attention.q_lora_rank"))
                    .context("deepseek32.attention.q_lora_rank")? as usize
            } else {
                meta_u64(g, &mk("attention.q_lora_rank")).unwrap_or(0) as usize
            };
            let kvlr = meta_u64(g, &mk("attention.kv_lora_rank"))
                .with_context(|| format!("{}.attention.kv_lora_rank", arch))?
                as usize;
            // No default for either MLA length: `head_k_mla`/`v_head_dim` below are derived
            // from them and nothing downstream can detect a wrong guess — the attention is
            // simply mis-shaped. llama.cpp routes a GGUF that carries neither key through
            // `is_mla() == false` (`llama_hparams::is_mla`), the unabsorbed `wkv_b` path,
            // which infr does not implement; refuse the file instead.
            let key_mla = mk("attention.key_length_mla");
            let Some(klen) = meta_u64(g, &key_mla) else {
                bail!("{key_mla} missing: deepseek2 MLA head dims cannot be derived");
            };
            let value_mla = mk("attention.value_length_mla");
            let Some(vlen) = meta_u64(g, &value_mla) else {
                bail!("{value_mla} missing: deepseek2 MLA head dims cannot be derived");
            };
            let qkr = meta_u64(g, &mk("rope.dimension_count")).unwrap_or(64) as usize;
            (qlr, kvlr, klen as usize, vlen as usize, qkr)
        } else {
            (0, 0, 0, 0, 0)
        };
        let (expert_gating_func, expert_weights_norm, rope_yarn_log_mul) =
            if deepseek2 || deepseek4 || bailingmoe3 {
                // deepseek2 tolerates the key's absence (llama.cpp documents 0 as "softmax", the
                // pre-`expert_gating_func` V2/V2.5 files); `deepseek32.cpp` and `deepseek4.cpp` read it
                // with no such fallback, and neither arch's real value is softmax — defaulting it would
                // silently re-route.
                let gf = if deepseek32 || deepseek4 {
                    let key = mk("expert_gating_func");
                    meta_u64(g, &key).with_context(|| format!("{key} missing"))? as u8
                } else {
                    meta_u64(g, &mk("expert_gating_func")).unwrap_or(0) as u8
                };
                let norm = g
                    .metadata()
                    .get(&mk("expert_weights_norm"))
                    .and_then(|v| match v {
                        MetaValue::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false);
                // `yarn_log_multiplier` is the input to deepseek2's MLA score mscale. V4's three
                // attention call sites all use a plain `1/√head_dim` — none of stage 2's mscale² games
                // — so the key stays unread there rather than feeding an arithmetic V4 does not have.
                let mut ylm = if deepseek2 {
                    meta_f64(g, &mk("rope.scaling.yarn_log_multiplier")).unwrap_or(0.0) as f32
                } else {
                    0.0
                };
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
        // `rope.scaling.attn_factor` and the two `yarn_beta_*` ramp corners come along here: all
        // three feed the same YaRN arithmetic (`seam::runner`'s `yarn_ff` ramp and `mla_scale`),
        // and their defaults are llama.cpp's `llama_hparams` initializers.
        let (
            rope_scaling_yarn,
            rope_scaling_factor,
            rope_scaling_orig_ctx,
            rope_attn_factor,
            rope_yarn_beta_fast,
            rope_yarn_beta_slow,
        ) = if deepseek2 || deepseek4 {
            let yarn = g
                .metadata()
                .get(&mk("rope.scaling.type"))
                .and_then(MetaValue::as_str)
                .is_some_and(|s| s == "yarn");
            let factor = meta_f64(g, &mk("rope.scaling.factor")).unwrap_or(1.0) as f32;
            let orig_ctx = meta_u64(g, &mk("rope.scaling.original_context_length"))
                .or_else(|| meta_u64(g, &mk("context_length")))
                .unwrap_or(8192) as usize;
            let attn_factor = meta_f64(g, &mk("rope.scaling.attn_factor")).unwrap_or(1.0) as f32;
            let beta_fast = meta_f64(g, &mk("rope.scaling.yarn_beta_fast")).unwrap_or(32.0) as f32;
            let beta_slow = meta_f64(g, &mk("rope.scaling.yarn_beta_slow")).unwrap_or(1.0) as f32;
            (yarn, factor, orig_ctx, attn_factor, beta_fast, beta_slow)
        } else {
            (false, 1.0, 0, 1.0, 32.0, 1.0)
        };
        // MLA per-head dims, derived exactly as the reference does (`src/models/deepseek2.cpp`
        // and `llama_hparams::n_embd_head_v_mla`): the NOPE width is the full per-head key width
        // minus the rotary part, and the V width is the GGUF key verbatim.
        let (head_k_mla, v_head_dim) = if mla_arch {
            // Mirrors `GGML_ASSERT(n_embd_head_qk_nope >= 1)`. Clamping instead would hand the
            // MLA kernels a one-wide NOPE dim and produce garbage with no error.
            let Some(qk_nope) = key_length_mla
                .checked_sub(qk_rope_dim)
                .filter(|nope| *nope > 0)
            else {
                bail!(
                    "deepseek2 MLA: attention.key_length_mla ({key_length_mla}) must exceed \
                     rope.dimension_count ({qk_rope_dim})"
                );
            };
            if value_length_mla == 0 {
                bail!("deepseek2 MLA: attention.value_length_mla must be positive");
            }
            (qk_nope, value_length_mla)
        } else {
            (0, 0)
        };
        let n_expert_groups = if deepseek2 || bailingmoe3 {
            meta_u64(g, &mk("expert_group_count")).unwrap_or(0) as usize
        } else {
            0
        };
        let n_expert_groups_used = if deepseek2 || bailingmoe3 {
            meta_u64(g, &mk("expert_group_used_count")).unwrap_or(0) as usize
        } else {
            0
        };
        // DeepSeek lightning indexer (V3.2 and V4 both). All three keys are REQUIRED (both
        // `load_arch_hparams` read them with a plain `get_key`), and all three are dimensions the
        // per-layer indexer tensors are shaped by — a zero would mis-shape them with nothing
        // downstream able to notice. V4's indexer differs structurally (no `indexer.attn_k`, no
        // `indexer.k_norm` — its keys come from the compressor) but is described by the same three
        // numbers.
        let (indexer_n_head, indexer_head_size, indexer_top_k) =
            if deepseek32 || deepseek4 || qwen4exp {
                let ix = |k: &str| -> Result<usize> {
                    let key = mk(k);
                    let v = meta_u64(g, &key).with_context(|| format!("{key} missing"))?;
                    positive_model_dimension(&key, v)
                };
                (
                    ix("attention.indexer.head_count")?,
                    ix("attention.indexer.key_length")?,
                    ix("attention.indexer.top_k")?,
                )
            } else {
                (0, 0, 0)
            };
        // The indexer's `k_norm` is a real (mean-centred) LayerNorm and `deepseek32.cpp` hardcodes
        // `hparams.f_norm_eps = 1e-6` for it — a SEPARATE epsilon from `f_norm_rms_eps` (`rms_eps`
        // below), which stays whatever the GGUF declares.
        let norm_eps = if deepseek32 { 1e-6 } else { 0.0 };
        let n_layer_all = meta_u64(g, &mk("block_count")).context("block_count")? as usize;
        // MTP/NextN (Qwen3.5/3.6, issue #33 — see docs/mtp.md): `{arch}.nextn_predict_layers`
        // extra decoder block(s) appended AFTER the trunk — `block_count` INCLUDES them. Ported
        // from the reference loader's `hparams.n_layer_nextn`
        // (`llama.cpp/src/models/qwen35.cpp::load_arch_hparams`). The confirmed 4B MTP GGUF sets
        // `block_count=33`/`nextn_predict_layers=1`: the trunk is 32 layers and `blk.32` is the
        // MTP head — without this split, `n_layer` below would include it and the trunk layer
        // loop (`seam.rs`'s `wload`) would misclassify `blk.32` as a gated-DeltaNet layer
        // (`(32+1) % full_attn_interval != 0`) and fail on missing `ssm_*` tensors.
        let declared_nextn = meta_u64(g, &mk("nextn_predict_layers")).unwrap_or(0) as usize;
        // The public Ling GGUF declares one NextN layer but contains no NextN/MTP tensors: all 42
        // `blk.*` entries are ordinary KDA/MLA trunk layers. Treat the metadata as stale. Should a
        // future file actually carry an MTP tensor set, fail explicitly until that head is wired.
        let bailing_has_nextn_tensors = bailingmoe3
            && g.tensors()
                .iter()
                .any(|t| t.name.starts_with("nextn.") || t.name.contains(".nextn."));
        if bailing_has_nextn_tensors {
            bail!("bailingmoe3 NextN/MTP tensors are not supported yet");
        }
        let n_layer_nextn = if bailingmoe3 { 0 } else { declared_nextn };
        // The bail exists because a NextN block that is NOT split off the trunk gets fed to the
        // per-layer loop as if it were an ordinary block, and then either fails on missing tensors
        // or (worse) misreads the ones it finds. It is not "MTP is unimplemented": qwen35 splits
        // and USES the block, and `deepseek32` splits and SKIPS it, which is exactly what
        // `deepseek32.cpp::load_arch_tensors` does with its `i >= n_layer` → `TENSOR_SKIP` arm. So
        // the arch list here is "arches that handle the split", and an arch missing from it still
        // fails loudly rather than silently.
        if n_layer_nextn > 0
            && arch != crate::arch::QWEN35
            && arch != crate::arch::QWEN35_MOE
            && !deepseek32
        {
            bail!(
                "{arch}.nextn_predict_layers is only supported on arch={}|{}|{} (MTP/NextN); got \
                 nextn_predict_layers={n_layer_nextn} on arch={arch:?}",
                crate::arch::QWEN35,
                crate::arch::QWEN35_MOE,
                crate::arch::DEEPSEEK32,
            );
        }
        // The reference caps qwen35 at a single MTP block (`GGML_ASSERT(hparams.n_layer_nextn ==
        // 1)` in `graph_mtp`'s ctor) — mirrored here so an unsupported wider value fails loudly
        // instead of silently misreading the tensor layout. deepseek32 has no such cap: it never
        // builds an MTP graph, so any count simply means that many trailing blocks are skipped
        // (`deepseek32.cpp` asserts only `n_layer_nextn < n_layer_all`, which is the next check).
        if n_layer_nextn > 1 && !deepseek32 {
            bail!(
                "qwen35 MTP: nextn_predict_layers={n_layer_nextn} > 1 not supported (the \
                 reference implementation caps at a single MTP block — see docs/mtp.md)",
            );
        }
        if n_layer_nextn >= n_layer_all {
            bail!(
                "{arch} NextN: nextn_predict_layers={n_layer_nextn} must be < block_count={n_layer_all}",
            );
        }
        let n_layer = n_layer_all - n_layer_nextn;
        // ── DeepSeek V4 hyperparameters (`deepseek4.cpp::load_arch_hparams`) ──────────────────
        // Everything here is read with a plain `get_key` in the reference, so absence is a broken
        // file rather than an older revision to default around: V4 shipped with all of them.
        let req_u64 = |k: &str| -> Result<u64> {
            let key = mk(k);
            meta_u64(g, &key).with_context(|| format!("{key} missing"))
        };
        let req_dim = |k: &str| -> Result<usize> { positive_model_dimension(&mk(k), req_u64(k)?) };
        let req_f32 = |k: &str| -> Result<f32> {
            let key = mk(k);
            meta_f64(g, &key)
                .map(|v| v as f32)
                .with_context(|| format!("{key} missing"))
        };
        // Widths that shape a tensor (`hc_dim = hc_mult * n_embd`, `wo_a`'s group split, `wo_b`'s
        // `o_groups * o_lora_rank` input) are held to positive; a zero would divide by zero or
        // silently collapse a dimension. The remaining three are counts and an epsilon, where zero
        // is a legal — if degenerate — value, so they are only required to be PRESENT.
        let (o_group_count, o_lora_rank, hc_mult, hc_low_rank, hc_sinkhorn_iters, hash_layer_count) =
            if deepseek4 {
                (
                    req_dim("attention.output_group_count")?,
                    req_dim("attention.output_lora_rank")?,
                    req_dim("hyper_connection.count")?,
                    0,
                    req_u64("hyper_connection.sinkhorn_iterations")? as usize,
                    req_u64("hash_layer_count")? as usize,
                )
            } else if qwen4exp {
                (
                    0,
                    0,
                    req_dim("hyper_connection.count")?,
                    req_dim("hyper_connection.low_rank")?,
                    0,
                    0,
                )
            } else {
                (0, 0, 0, 0, 0, 0)
            };
        let (compress_rope_theta, hc_eps) = if deepseek4 {
            (
                req_f32("attention.compress_rope_freq_base")?,
                req_f32("hyper_connection.epsilon")?,
            )
        } else {
            (0.0, 0.0)
        };
        // `compress_ratios` is the master per-layer switch (see `Config::compress_ratios`). The
        // reference reads the array length FIRST and throws when it is shorter than `block_count`,
        // because `load_arch_tensors` then indexes `[il]` for every layer; the extra entries a
        // longer array carries are simply unused, which is why this takes a prefix rather than
        // demanding an exact length.
        let compress_ratios: Vec<usize> = if deepseek4 || qwen4exp {
            let key = mk("attention.compress_ratios");
            let arr = g
                .metadata()
                .get(&key)
                .and_then(MetaValue::as_arr)
                .with_context(|| format!("{key} missing (or not an array)"))?;
            if arr.len() < n_layer {
                bail!(
                    "{key} is shorter than block_count: {} entries for {n_layer} layers",
                    arr.len()
                );
            }
            arr[..n_layer]
                .iter()
                .enumerate()
                .map(|(il, v)| {
                    let r = v.as_u64().unwrap_or(u64::MAX) as usize;
                    // `deepseek4.cpp::load_arch_tensors`: "DeepSeek-V4 loader only supports
                    // compression ratios 0, 4, and 128". An unknown ratio is not a wider variant of
                    // a known one — it decides which tensors the layer HAS — so refuse rather than
                    // round it to the nearest tier.
                    let valid = if deepseek4 {
                        matches!(r, 0 | 4 | 128)
                    } else {
                        matches!(r, 0 | 4)
                    };
                    if !valid {
                        if deepseek4 {
                            bail!("{key}[{il}] = {v:?} is not one of 0, 4, 128");
                        }
                        bail!("{key}[{il}] = {v:?} is not one of 0, 4");
                    }
                    Ok(r)
                })
                .collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        // `get_key_or_arr(.., n_layer)` in the reference: ONE scalar broadcast to every layer, or a
        // per-layer array. `swiglu_clamp_shexp` is the same key with `required = 0` and falls back
        // to the routed clamp when absent, which is what a GGUF declaring only `_exp` means.
        let clamp_arr = |k: &str| -> Option<Vec<f32>> {
            let v = g.metadata().get(&mk(k))?;
            match v {
                MetaValue::Arr(a) => Some(
                    a.iter()
                        .map(|e| e.as_f64().unwrap_or(0.0) as f32)
                        .take(n_layer)
                        .collect(),
                ),
                other => other.as_f64().map(|f| vec![f as f32; n_layer]),
            }
        };
        let (swiglu_clamp_exp, swiglu_clamp_shexp) = if deepseek4 || bailingmoe3 {
            let key = mk("swiglu_clamp_exp");
            let exp = clamp_arr("swiglu_clamp_exp").with_context(|| format!("{key} missing"))?;
            if exp.len() < n_layer {
                bail!("{key} has {} entries for {n_layer} layers", exp.len());
            }
            let shexp = clamp_arr("swiglu_clamp_shexp").unwrap_or_else(|| exp.clone());
            if shexp.len() < n_layer {
                bail!(
                    "{} has {} entries for {n_layer} layers",
                    mk("swiglu_clamp_shexp"),
                    shexp.len()
                );
            }
            (exp, shexp)
        } else {
            (Vec::new(), Vec::new())
        };
        if deepseek4 {
            // `deepseek4.cpp::load_arch_hparams` throws on anything but sqrt-softplus. That is not
            // a placeholder for "the others are untested": the router this arch was trained with is
            // the one that must run, and every other value here names a DIFFERENT function.
            const SQRT_SOFTPLUS: u8 = 4;
            if expert_gating_func != SQRT_SOFTPLUS {
                bail!(
                    "deepseek4 requires sqrt-softplus MoE scoring ({}={SQRT_SOFTPLUS}); got {}",
                    mk("expert_gating_func"),
                    expert_gating_func
                );
            }
            // Keys the reference requires that the SHARED parses further down would otherwise
            // default silently — and each default is wrong in a way nothing downstream can detect:
            // `expert_weights_scale` falls back to 0, which zeroes every routed expert's
            // contribution, `expert_feed_forward_length` falls back to `n_ff / n_used`, and
            // `attention.layer_norm_rms_epsilon` falls back to the generic 1e-5.
            for key in [
                "attention.layer_norm_rms_epsilon",
                "expert_feed_forward_length",
                "expert_shared_count",
                "expert_weights_scale",
                "expert_weights_norm",
            ] {
                let key = mk(key);
                if g.metadata().get(&key).is_none() {
                    bail!("{key} missing");
                }
            }
        }
        // V4 keeps deepseek2's Q-LoRA shape (`wq_a → q_a_norm → wq_b`) and nothing else of MLA, so
        // it reads `q_lora_rank` here rather than through the MLA tuple above — which is gated on
        // `deepseek2` and also parses `kv_lora_rank` and the two MLA head lengths, none of which
        // exist in a V4 file. Mandatory, like `deepseek32`'s: there is no lite variant.
        let q_lora_rank = if deepseek4 {
            req_dim("attention.q_lora_rank")?
        } else {
            q_lora_rank
        };
        // DeepSeek2 "lite" detection: the direct `wq` is present instead of the
        // wq_a/q_a_norm/wq_b LoRA triple — a tensor-presence test, per docs/deepseek.md § Stage 2.
        // llama.cpp reaches the same conclusion from a layer-count table
        // (`deepseek2.cpp::load_arch_hparams`), which misclassifies any other model with the same
        // depth; the tensors say it directly.
        // V3.2 is never lite — `deepseek32.cpp` reads `q_lora_rank` unconditionally and its tensor
        // loader has no `attn_q` arm at all — so the presence test does not run for it. Without
        // this a V3.2 file that happened to carry a `blk.0.attn_q.weight` would drop the whole
        // wq_a/q_a_norm/wq_b path the model actually has.
        let is_lite =
            mla_arch && !deepseek32 && g.tensors().iter().any(|t| t.name == "blk.0.attn_q.weight");
        let n_embd = meta_u64(g, &mk("embedding_length")).context("embedding_length")? as usize;
        let n_head = positive_model_dimension(
            &mk("attention.head_count"),
            meta_u64(g, &mk("attention.head_count")).context("head_count")?,
        )?;
        // DeepSeek2 MLA: n_kv is always 1 (single key head shared by all query heads).
        let n_kv = if mla_arch {
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
        } else if qwen35_moe || qwen4exp {
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
        } else if deepseek4 {
            // V4 has no dense FFN on any layer — no dense lead, every layer routed — so its GGUF
            // need not carry `feed_forward_length` at all, and `load_arch_hparams` never reads one.
            // As for qwen35moe above, `n_ff`/`n_ff_layers` only size the dense-FFN-SHAPED scratch
            // the shared-expert branch reuses at ITS width, which for DeepSeek is
            // `expert_feed_forward_length * expert_shared_count` (see `shexp_ff` below). Both keys
            // were required in the deepseek4 block above, so neither fallback here can fire.
            let n_ff_exp = meta_u64(g, &mk("expert_feed_forward_length")).unwrap_or(0) as usize;
            let n_shared = meta_u64(g, &mk("expert_shared_count")).unwrap_or(0) as usize;
            vec![n_ff_exp * n_shared; n_layer]
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
            || qwen4exp
            || gemma4_moe
            || llama4
            || deepseek
            || deepseek2
            || deepseek4
            || bailingmoe3
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
            } else if deepseek2 || deepseek4 || bailingmoe3 {
                let gating = match expert_gating_func {
                    // 0 = the key is absent, which llama.cpp documents as softmax for the old
                    // V2/V2.5 GGUFs written before `expert_gating_func` existed
                    // (`deepseek2.cpp::load_arch_hparams`).
                    0 | 1 => infr_core::graph::MoeGating::Softmax,
                    2 => infr_core::graph::MoeGating::Sigmoid,
                    4 => infr_core::graph::MoeGating::SqrtSoftplus,
                    // 3 is softmax-over-SELECTED-weights, a different function from 1 — there is
                    // no `MoeGating` for it, and falling back to plain softmax would silently
                    // change the routing rather than fail.
                    other => bail!("{arch}.expert_gating_func = {other} is not supported"),
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
        let head_dim = if mla_arch {
            key_length_mla
        } else {
            meta_u64(g, &mk("attention.key_length")).unwrap_or((n_embd / n_head) as u64) as usize
        };
        let rope_dim = if mla_arch {
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
            .unwrap_or(if qwen35 || qwen4exp { 1e7 } else { 10000.0 });
        let rms_eps = g
            .metadata()
            .get(&mk("attention.layer_norm_rms_epsilon"))
            .and_then(|v| match v {
                MetaValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            // qwen35's old seam (`Cfg::from_gguf`) defaults this to 1e-6, not the generic 1e-5.
            .unwrap_or(if qwen35 || qwen4exp { 1e-6 } else { 1e-5 });
        // deepseek4 reads `attention.sliding_window` with a plain `get_key` — its raw attention is
        // sliding-window on EVERY layer, so the window is not optional decoration there.
        let swa_window = if deepseek4 {
            req_dim("attention.sliding_window")?
        } else if gemma {
            meta_u64(g, &mk("attention.sliding_window")).unwrap_or(0) as usize
        } else {
            0
        };
        // deepseek4 keeps `swa_pattern` at 0: the field means "every p-th layer is FULL", and V4
        // has no full layers (`set_swa_pattern(0)`). `Config::is_swa_layer` special-cases the arch
        // instead — reading `attention.sliding_window_pattern` here would invent a period the file
        // does not declare and make five layers in six look full.
        let swa_pattern = if swa_window == 0 || deepseek4 {
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
        // gemma3's SWA layers rope at a SMALLER base than its full layers, which is what this
        // field is for. deepseek4's sliding-window layers do not: a ratio-0 layer ropes at the
        // plain `rope_theta`, and the compressed tiers use `compress_rope_theta` (a separate field)
        // — so V4 keeps them equal rather than picking up gemma's 10000 fallback.
        let swa_rope_theta = if swa_window > 0 && !deepseek4 {
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
        let full_attn_interval = if qwen35 || qwen4exp {
            meta_u64(g, &mk("full_attention_interval")).unwrap_or(4) as usize
        } else {
            0
        };
        let recurrent_layers = if qwen4exp {
            let key = mk("attention.recurrent_layers");
            if let Some(values) = g.metadata().get(&key).and_then(MetaValue::as_arr) {
                if values.len() < n_layer {
                    bail!("{key} has {} entries for {n_layer} layers", values.len());
                }
                values[..n_layer]
                    .iter()
                    .enumerate()
                    .map(|(il, v)| match v {
                        MetaValue::Bool(b) => Ok(*b),
                        _ => v
                            .as_u64()
                            .map(|n| n != 0)
                            .with_context(|| format!("{key}[{il}] is not bool/integer")),
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                (0..n_layer)
                    .map(|il| !(il + 1).is_multiple_of(full_attn_interval.max(1)))
                    .collect()
            }
        } else {
            Vec::new()
        };
        let (kda_head_dim, kda_gate_lower_bound) = if bailingmoe3 {
            (
                meta_u64(g, &mk("kda.head_dim")).context("bailingmoe3.kda.head_dim")? as usize,
                meta_f64(g, &mk("kda.gate_lower_bound"))
                    .context("bailingmoe3.kda.gate_lower_bound")? as f32,
            )
        } else {
            (0, 0.0)
        };
        let (ssm_d_conv, ssm_d_state, ssm_d_inner, ssm_n_group, ssm_dt_rank) = if qwen35 || qwen4exp
        {
            (
                meta_u64(g, &mk("ssm.conv_kernel"))
                    .with_context(|| format!("{arch} ssm.conv_kernel"))? as usize,
                meta_u64(g, &mk("ssm.state_size"))
                    .with_context(|| format!("{arch} ssm.state_size"))? as usize,
                meta_u64(g, &mk("ssm.inner_size"))
                    .with_context(|| format!("{arch} ssm.inner_size"))? as usize,
                meta_u64(g, &mk("ssm.group_count"))
                    .with_context(|| format!("{arch} ssm.group_count"))? as usize,
                meta_u64(g, &mk("ssm.time_step_rank"))
                    .with_context(|| format!("{arch} ssm.time_step_rank"))?
                    as usize,
            )
        } else if bailingmoe3 {
            (
                meta_u64(g, &mk("ssm.conv_kernel")).context("bailingmoe3.ssm.conv_kernel")?
                    as usize,
                kda_head_dim,
                n_head * kda_head_dim,
                n_head,
                n_head,
            )
        } else {
            (0, 0, 0, 0, 0)
        };
        let rope_sections: [u32; 4] = if qwen35 || qwen4exp {
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
        let attn_out_gate = qwen35 || qwen4exp;
        // DeepSeek: first N layers are dense FFN, the rest are MoE.
        let n_layer_dense_lead = if deepseek || deepseek2 || bailingmoe3 {
            meta_u64(g, &mk("leading_dense_block_count")).unwrap_or(0) as usize
        } else {
            0
        };
        // Shared-expert FFN width. qwen35moe reads `expert_shared_feed_forward_length`; llama4 has
        // NO such key — its shared expert is the SAME width as a routed expert
        // (`models/llama4.cpp`: `n_ff_shexp = n_ff_exp`). DeepSeek's shared expert is
        // `n_ff_exp * n_expert_shared` wide (llama.cpp fuses `n_expert_shared` experts into one
        // wider branch — see docs/deepseek.md § Stage 1). `0` = no shared expert.
        let shexp_ff = if qwen35_moe || qwen4exp {
            meta_u64(g, &mk("expert_shared_feed_forward_length")).unwrap_or(0) as usize
        } else if llama4 {
            moe.map(|m| m.n_ff_exp).unwrap_or(0)
        } else if deepseek || deepseek2 || deepseek4 || bailingmoe3 {
            let n_shared = meta_u64(g, &mk("expert_shared_count")).unwrap_or(0) as usize;
            moe.map(|m| m.n_ff_exp * n_shared).unwrap_or(0)
        } else {
            0
        };
        // llama4 shared expert is summed in PLAIN (no per-token gate); qwen35moe gates by sigmoid.
        // DeepSeek's shared expert is also plain-summed (no gate).
        let shexp_gated = qwen35_moe || qwen4exp;
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
        let bailing_mla_layers = if bailingmoe3 {
            let key = mk("attention.head_count_kv");
            let values = g
                .metadata()
                .get(&key)
                .and_then(MetaValue::as_arr)
                .with_context(|| format!("{key} missing (or not an array)"))?;
            if values.len() < n_layer {
                bail!("{key} has {} entries for {n_layer} layers", values.len());
            }
            values[..n_layer]
                .iter()
                .enumerate()
                .map(|(il, v)| match v.as_u64() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    other => bail!("{key}[{il}] must be 0 (KDA) or 1 (MLA), got {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let (
            ple_layers,
            ple_ngram_size,
            ple_heads_per_ngram,
            ple_head_dim,
            ple_conv_kernel,
            ple_eos,
            ple_layer_multipliers,
            ple_head_offsets,
            ple_head_vocab_sizes,
        ) = if qwen4exp {
            let arr_u64 = |suffix: &str| -> Result<Vec<u64>> {
                let key = mk(suffix);
                g.metadata()
                    .get(&key)
                    .and_then(MetaValue::as_arr)
                    .with_context(|| format!("{key} missing (or not an array)"))?
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_u64()
                            .with_context(|| format!("{key}[{i}] is not a non-negative integer"))
                    })
                    .collect()
            };
            let mut layers = vec![false; n_layer];
            for il in arr_u64("ple.layers")? {
                let il = usize::try_from(il).context("qwen4exp.ple.layers index overflow")?;
                if il >= n_layer {
                    bail!(
                        "{}.ple.layers contains {il}, but block_count is {n_layer}",
                        arch
                    );
                }
                layers[il] = true;
            }
            let ng = req_dim("ple.ngram_size")?;
            let hpg = req_dim("ple.heads_per_ngram")?;
            let row = req_dim("embedding_length_per_layer_input")?;
            let kernel = req_dim("ple.conv_kernel")?;
            let eos = req_u64("ple.eos_token_id")? as u32;
            let multipliers = arr_u64("ple.layer_multipliers")?;
            let offsets = arr_u64("ple.head_offsets")?;
            let sizes = arr_u64("ple.head_vocab_sizes")?;
            let n_heads = ng.saturating_sub(1).saturating_mul(hpg);
            if ng < 2 || multipliers.len() < ng {
                bail!(
                    "qwen4exp PLE needs ngram_size >= 2 and one multiplier per position; got \
                     ngram_size={ng}, multipliers={}",
                    multipliers.len()
                );
            }
            if offsets.len() != n_heads || sizes.len() != n_heads || sizes.contains(&0) {
                bail!(
                    "qwen4exp PLE head metadata mismatch: expected {n_heads} non-zero heads, got \
                     offsets={} sizes={}",
                    offsets.len(),
                    sizes.len()
                );
            }
            (
                layers,
                ng,
                hpg,
                row,
                kernel,
                eos,
                multipliers,
                offsets,
                sizes,
            )
        } else {
            (
                Vec::new(),
                0,
                0,
                0,
                0,
                0,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };
        // Converter-defect repair, qwen4exp only. Published Qwen3.8-Flash-Next quants can carry an
        // ALL-ZERO `attention.compress_ratios` even though the model compresses: the model's
        // `config.json` sets `indexer_compress_ratio: 4`, the indexer's own weights ARE there on
        // every full-attention layer, and its `indexer_top_k` of 2048 TOKENS is only expressible as
        // `2048 / 4 = 512` blocks — exactly `QSA_MAX_TOP_BLOCKS` — so the file's own numbers are
        // self-contradictory and the zeros are a defect rather than a design.
        //
        // It restores the official ratio 4 and NOTHING else. The ratio is a model structure
        // parameter, not a kernel budget knob, so deriving whatever happens to fit the block budget
        // (1/2/8 …) would let a damaged file load and then decode wrong. When the restored ratio
        // does not fit either, refuse to load instead of silently changing the model.
        //
        // This runs only when EVERY full-attention layer reads 0. Anything the file does declare
        // still wins, and a 0/nonzero MIX is deliberately left untouched so the strict per-layer
        // check below rejects it: the runner carries a single global QSA block ratio and cannot
        // express per-layer differences.
        let compress_ratios = if qwen4exp
            && !recurrent_layers
                .iter()
                .zip(&compress_ratios)
                .any(|(&recurrent, &ratio)| !recurrent && ratio != 0)
        {
            if indexer_top_k == 0
                || !indexer_top_k.is_multiple_of(QWEN38_QSA_BLOCK_RATIO)
                || indexer_top_k / QWEN38_QSA_BLOCK_RATIO > QSA_MAX_TOP_BLOCKS as usize
            {
                bail!(
                    "{arch}.attention.compress_ratios is 0 on every full-attention layer and \
                     cannot be restored to the Qwen3.8 block ratio {QWEN38_QSA_BLOCK_RATIO}: \
                     attention.indexer.top_k {indexer_top_k} must be a nonzero multiple of \
                     {QWEN38_QSA_BLOCK_RATIO} leaving at most {QSA_MAX_TOP_BLOCKS} blocks"
                );
            }
            let blocks = indexer_top_k / QWEN38_QSA_BLOCK_RATIO;
            tracing::warn!(
                "{arch}.attention.compress_ratios is 0 on every full-attention layer; treating \
                 that as converter-defect metadata and restoring the Qwen3.8 block ratio \
                 {QWEN38_QSA_BLOCK_RATIO} (indexer top_k {indexer_top_k} = {blocks} blocks)"
            );
            recurrent_layers
                .iter()
                .map(
                    |&recurrent| {
                        if recurrent {
                            0
                        } else {
                            QWEN38_QSA_BLOCK_RATIO
                        }
                    },
                )
                .collect()
        } else {
            compress_ratios
        };
        if qwen4exp {
            for (name, value) in [
                ("ssm.conv_kernel", ssm_d_conv),
                ("ssm.state_size", ssm_d_state),
                ("ssm.inner_size", ssm_d_inner),
                ("ssm.group_count", ssm_n_group),
                ("ssm.time_step_rank", ssm_dt_rank),
            ] {
                if value == 0 {
                    bail!("{}.{} must be positive", arch, name);
                }
            }
            if recurrent_layers.len() != n_layer {
                bail!(
                    "qwen4exp recurrent-layer map has {} entries for {n_layer} layers",
                    recurrent_layers.len()
                );
            }
            if !recurrent_layers.iter().any(|&v| v) || !recurrent_layers.iter().any(|&v| !v) {
                bail!("qwen4exp requires both recurrent and full-attention layers");
            }
            if !ssm_dt_rank.is_multiple_of(ssm_n_group) {
                bail!(
                    "qwen4exp ssm.time_step_rank ({ssm_dt_rank}) must be divisible by \
                     ssm.group_count ({ssm_n_group})"
                );
            }
            if !ssm_d_inner.is_multiple_of(ssm_dt_rank) {
                bail!(
                    "qwen4exp ssm.inner_size ({ssm_d_inner}) must be divisible by \
                     ssm.time_step_rank ({ssm_dt_rank})"
                );
            }
            let head_v_dim = ssm_d_inner / ssm_dt_rank;
            if head_v_dim != ssm_d_state {
                bail!(
                    "qwen4exp gated-DeltaNet requires equal K/V head dimensions; state_size is \
                     {ssm_d_state}, but inner_size/time_step_rank is {head_v_dim}"
                );
            }
            // `compress_ratios[il]` is qwen4exp's QSA BLOCK factor: the indexer mean-pools
            // `ratio` raw indexer keys into one scored block key, so `indexer_top_k` TOKENS means
            // `indexer_top_k / ratio` BLOCKS. A recurrent layer has no indexer at all and must
            // read 0; a full-attention layer MUST compress, with a ratio that divides
            // `indexer_top_k`. A trailing 0 on a full-attention layer is only reachable as part of
            // a 0/nonzero mix, which the repair above leaves alone on purpose: one global ratio
            // cannot describe it, so it is an error rather than a guess.
            for (il, (&recurrent, &ratio)) in
                recurrent_layers.iter().zip(&compress_ratios).enumerate()
            {
                if recurrent && ratio != 0 {
                    bail!(
                        "qwen4exp attention.compress_ratios[{il}] is {ratio} on a recurrent \
                        layer; recurrent layers must use ratio 0"
                    );
                }
                if !recurrent && ratio == 0 {
                    bail!(
                        "qwen4exp attention.compress_ratios[{il}] is 0 on a full-attention layer \
                         while another full-attention layer compresses; this runner uses a single \
                         global QSA block ratio and cannot express per-layer differences"
                    );
                }
                if !recurrent && (indexer_top_k == 0 || !indexer_top_k.is_multiple_of(ratio)) {
                    bail!(
                        "qwen4exp full-attention layer {il} has compression ratio {ratio}, which \
                         does not divide indexer top_k {indexer_top_k}"
                    );
                }
            }
            let ple_impl: Vec<_> = ple_layers
                .iter()
                .enumerate()
                .filter_map(|(il, &enabled)| enabled.then_some(il))
                .collect();
            if ple_impl.as_slice() != [1] {
                bail!(
                    "qwen4exp v1 expects the released model's single PLE implementation at \
                     zero-based layer 1; got {ple_impl:?}"
                );
            }
            let ple_width = ple_ngram_size
                .checked_sub(1)
                .and_then(|n| n.checked_mul(ple_heads_per_ngram))
                .and_then(|n| n.checked_mul(ple_head_dim))
                .context("qwen4exp PLE flattened width overflow")?;
            if ple_width != n_embd {
                bail!(
                    "qwen4exp PLE flattened width is {ple_width}, but embedding_length is \
                     {n_embd}"
                );
            }
            hc_mult
                .checked_mul(n_embd)
                .context("qwen4exp hyper-connection width overflow")?;
        }
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
            qwen4exp,
            bailingmoe3,
            bailing_mla_layers,
            kda_head_dim,
            kda_gate_lower_bound,
            full_attn_interval,
            recurrent_layers,
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
            deepseek32,
            deepseek4,
            compress_ratios,
            swiglu_clamp_exp,
            swiglu_clamp_shexp,
            hash_layer_count,
            o_group_count,
            o_lora_rank,
            compress_rope_theta,
            hc_mult,
            hc_low_rank,
            hc_sinkhorn_iters,
            hc_eps,
            ple_layers,
            ple_ngram_size,
            ple_heads_per_ngram,
            ple_head_dim,
            ple_conv_kernel,
            ple_eos,
            ple_layer_multipliers,
            ple_head_offsets,
            ple_head_vocab_sizes,
            indexer_n_head,
            indexer_head_size,
            indexer_top_k,
            norm_eps,
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
            rope_attn_factor,
            rope_yarn_beta_fast,
            rope_yarn_beta_slow,
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

    fn push_u32_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[u32]) {
        push_str(bytes, key);
        push_u32(bytes, 9); // GGUF_TYPE_ARRAY
        push_u32(bytes, 4); // GGUF_TYPE_UINT32
        push_u64(bytes, values.len() as u64);
        for &value in values {
            push_u32(bytes, value);
        }
    }

    fn push_u64_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[u64]) {
        push_str(bytes, key);
        push_u32(bytes, 9); // GGUF_TYPE_ARRAY
        push_u32(bytes, 10); // GGUF_TYPE_UINT64
        push_u64(bytes, values.len() as u64);
        for &value in values {
            push_u64(bytes, value);
        }
    }

    /// Minimal `deepseek2` GGUF carrying exactly `keys` (each prefixed with `deepseek2.`) plus a
    /// single dummy tensor — enough for `Config::from_gguf` to reach the MLA and MoE parses.
    fn deepseek2_fixture(keys: &[(&str, u32)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747); // GGUF magic
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1); // tensor count
        push_u64(&mut bytes, 1 + keys.len() as u64);

        push_str(&mut bytes, "general.architecture");
        push_u32(&mut bytes, 8); // GGUF_TYPE_STRING
        push_str(&mut bytes, "deepseek2");
        for (key, value) in keys {
            push_u32_metadata(&mut bytes, &format!("deepseek2.{key}"), *value);
        }

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

    /// Write a fixture to a uniquely-named temp file, parse it, and clean up.
    fn config_from_fixture(name: &str, bytes: &[u8]) -> Result<Config> {
        let path =
            std::env::temp_dir().join(format!("infr-llama-{name}-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).expect("write GGUF fixture");
        let gguf = Gguf::open(&path).expect("open GGUF fixture");
        let parsed = Config::from_gguf(&gguf);
        std::fs::remove_file(path).expect("remove GGUF fixture");
        parsed
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

    /// Four-layer scale model of the released Qwen3.8 layout. It deliberately omits
    /// `attention.recurrent_layers`, as the current converter does, to exercise the
    /// `full_attention_interval` fallback — pass `1` to make every layer full attention, `2` for
    /// the 2 GDN + 2 full-attention split the mixed-ratio tests need.
    fn qwen4exp_fixture(
        ple_row_dim: u32,
        indexer_top_k: u32,
        full_attention_interval: u32,
        compress_ratios: &[u32],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747); // GGUF magic
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1); // tensor count
        push_u64(&mut bytes, 34); // metadata count

        push_str(&mut bytes, "general.architecture");
        push_u32(&mut bytes, 8); // GGUF_TYPE_STRING
        push_str(&mut bytes, "qwen4exp");
        for (key, value) in [
            ("block_count", 4),
            ("embedding_length", 32),
            ("attention.head_count", 4),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 8),
            ("rope.dimension_count", 2),
            ("context_length", 64),
            ("expert_count", 8),
            ("expert_used_count", 2),
            ("expert_feed_forward_length", 16),
            ("expert_shared_feed_forward_length", 16),
            ("full_attention_interval", full_attention_interval),
            ("ssm.conv_kernel", 4),
            ("ssm.state_size", 4),
            ("ssm.inner_size", 16),
            ("ssm.group_count", 2),
            ("ssm.time_step_rank", 4),
            ("hyper_connection.count", 4),
            ("hyper_connection.low_rank", 8),
            ("attention.indexer.head_count", 2),
            ("attention.indexer.key_length", 8),
            ("attention.indexer.top_k", indexer_top_k),
            ("ple.ngram_size", 3),
            ("ple.heads_per_ngram", 2),
            ("embedding_length_per_layer_input", ple_row_dim),
            ("ple.conv_kernel", 4),
            ("ple.eos_token_id", 2),
        ] {
            push_u32_metadata(&mut bytes, &format!("qwen4exp.{key}"), value);
        }
        push_u32_array_metadata(
            &mut bytes,
            "qwen4exp.attention.compress_ratios",
            compress_ratios,
        );
        push_u32_array_metadata(
            &mut bytes,
            "qwen4exp.rope.dimension_sections",
            &[1, 1, 0, 0],
        );
        push_u32_array_metadata(&mut bytes, "qwen4exp.ple.layers", &[1]);
        push_u64_array_metadata(
            &mut bytes,
            "qwen4exp.ple.layer_multipliers",
            &[u32::MAX as u64 + 17, 11, 13],
        );
        push_u64_array_metadata(&mut bytes, "qwen4exp.ple.head_offsets", &[0, 100, 200, 300]);
        push_u64_array_metadata(
            &mut bytes,
            "qwen4exp.ple.head_vocab_sizes",
            &[97, 89, 83, 79],
        );

        push_str(&mut bytes, "token_embd.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 32);
        push_u64(&mut bytes, 8);
        push_u32(&mut bytes, 0); // F32
        push_u64(&mut bytes, 0);
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0; 32 * 8 * 4]);
        bytes
    }

    #[test]
    fn model_dimensions_must_be_positive() {
        assert!(positive_model_dimension("test.head_count", 0).is_err());
        assert_eq!(positive_model_dimension("test.head_count", 1).unwrap(), 1);
    }

    #[test]
    fn qwen4exp_interval_and_ple_geometry_match_the_released_layout() {
        let cfg = config_from_fixture("qwen4exp", &qwen4exp_fixture(8, 16, 4, &[0, 0, 0, 4]))
            .expect("scaled qwen4exp fixture must parse");
        assert!(cfg.qwen4exp);
        assert_eq!(cfg.recurrent_layers, [true, true, true, false]);
        assert_eq!(cfg.compress_ratios, [0, 0, 0, 4]);
        assert_eq!(cfg.ple_layers, [false, true, false, false]);
        assert_eq!(cfg.ple_layer_multipliers[0], u32::MAX as u64 + 17);
        assert_eq!(cfg.q35_head_k_dim(), 4);
        assert_eq!(cfg.q35_head_v_dim(), 4);
    }

    /// The released Qwen3.8-Flash-Next quants declare `compress_ratios` as 48 zeros while their
    /// `indexer_top_k` is 2048 TOKENS — uncompressed that is 2048 blocks, past the kernels'
    /// `QSA_MAX_TOP_BLOCKS`, so the file's own numbers are self-contradictory and the ratio has to
    /// be recovered (the model's `config.json` confirms `indexer_compress_ratio: 4`). The repair
    /// restores exactly that 4 — never a ratio derived from the block budget.
    #[test]
    fn qwen4exp_all_zero_compress_ratios_are_restored_to_the_qwen38_ratio() {
        let cfg = config_from_fixture(
            "qwen4exp-zeros",
            &qwen4exp_fixture(8, 2048, 4, &[0, 0, 0, 0]),
        )
        .expect("an all-zero compress_ratios array must parse");
        assert_eq!(
            cfg.compress_ratios,
            [0, 0, 0, 4],
            "2048 tokens / 4 = 512 blocks, the largest budget the QSA kernels accept"
        );
        assert_eq!(cfg.layer_compress_ratio(3), 4);
        // The recurrent layers stay uncompressed whatever the repair decides.
        assert_eq!(cfg.layer_compress_ratio(0), 0);
    }

    /// A smaller `top_k` still restores 4 rather than the 1 the block budget alone would allow:
    /// the ratio is a model structure parameter, so it must not be inferred from the budget.
    #[test]
    fn qwen4exp_all_zero_compress_ratios_restore_four_not_the_budget_ratio() {
        let cfg = config_from_fixture(
            "qwen4exp-zeros-small",
            &qwen4exp_fixture(8, 16, 4, &[0, 0, 0, 0]),
        )
        .expect("a small top_k leaves room for ratio 1, but 4 is the trained value");
        assert_eq!(cfg.compress_ratios, [0, 0, 0, 4]);
    }

    /// The repair is a last resort, not a blanket rewrite: when restoring 4 cannot work the file
    /// has to be rejected instead of silently loaded with a made-up geometry.
    #[test]
    fn qwen4exp_refuses_all_zero_compress_ratios_that_ratio_four_cannot_express() {
        // 4096 / 4 = 1024 blocks, past the kernels' 512-block ceiling.
        let Err(err) = config_from_fixture(
            "qwen4exp-zeros-oversized",
            &qwen4exp_fixture(8, 4096, 4, &[0, 0, 0, 0]),
        ) else {
            panic!("all-zero compress_ratios with an oversized indexer top_k must fail");
        };
        assert!(
            err.to_string()
                .contains("cannot be restored to the Qwen3.8 block ratio 4"),
            "unexpected error: {err}"
        );
        // 1002 is not a multiple of 4, so `selected * ratio + tail` could not be exact.
        let Err(err) = config_from_fixture(
            "qwen4exp-zeros-indivisible",
            &qwen4exp_fixture(8, 1002, 4, &[0, 0, 0, 0]),
        ) else {
            panic!("all-zero compress_ratios with a top_k that 4 does not divide must fail");
        };
        assert!(
            err.to_string()
                .contains("cannot be restored to the Qwen3.8 block ratio 4"),
            "unexpected error: {err}"
        );
    }

    /// A 0/4 mix is NOT repairable: this runner carries one global QSA block ratio, so loading such
    /// a file would run every layer with the wrong geometry.
    #[test]
    fn qwen4exp_refuses_a_mix_of_zero_and_nonzero_compress_ratios() {
        // `full_attention_interval` 2 makes layers 1 and 3 full attention; layer 3 compresses and
        // layer 1 does not, which no single global ratio can describe.
        let Err(err) = config_from_fixture(
            "qwen4exp-mixed-ratios",
            &qwen4exp_fixture(8, 2048, 2, &[0, 0, 0, 4]),
        ) else {
            panic!("a 0/4 compress_ratios mix must fail");
        };
        assert!(
            err.to_string().contains(
                "is 0 on a full-attention layer while another full-attention layer compresses"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn qwen4exp_refuses_a_ple_width_that_does_not_flatten_to_hidden_size() {
        let Err(err) = config_from_fixture(
            "qwen4exp-bad-ple",
            &qwen4exp_fixture(7, 16, 4, &[0, 0, 0, 4]),
        ) else {
            panic!("mismatched qwen4exp PLE width parsed successfully");
        };
        assert!(err.to_string().contains("PLE flattened width"));
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

    /// A deepseek2 GGUF that parses, with V2-Lite's real MLA geometry. `block_count` is 27 — the
    /// depth llama.cpp's heuristic reads as "lite" — while the fixture carries no
    /// `blk.0.attn_q.weight`, so `is_lite` must stay false.
    const DEEPSEEK2_BASE_KEYS: &[(&str, u32)] = &[
        ("attention.kv_lora_rank", 512),
        ("attention.key_length_mla", 192),
        ("attention.value_length_mla", 128),
        ("rope.dimension_count", 64),
        ("block_count", 27),
        ("embedding_length", 2048),
        ("attention.head_count", 16),
        ("feed_forward_length", 10944),
        ("expert_count", 64),
        ("expert_used_count", 6),
    ];

    fn deepseek2_keys_with(extra: &[(&str, u32)]) -> Vec<(&'static str, u32)> {
        let mut keys = DEEPSEEK2_BASE_KEYS.to_vec();
        for (key, value) in extra {
            match keys.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = *value,
                None => panic!("{key} is not a base key; add it to DEEPSEEK2_BASE_KEYS"),
            }
        }
        keys
    }

    /// B40: the MLA head dims come from the reference's derivation — `key_length_mla` minus the
    /// rope width for the NOPE dim, `value_length_mla` verbatim for V.
    #[test]
    fn deepseek2_mla_head_dims_match_reference() {
        let cfg = config_from_fixture("ds2-mla-dims", &deepseek2_fixture(DEEPSEEK2_BASE_KEYS))
            .expect("base deepseek2 fixture must parse");
        assert_eq!(cfg.head_k_mla, 192 - 64, "qk_nope = key_length_mla - n_rot");
        assert_eq!(cfg.v_head_dim, 128, "v_head_dim = value_length_mla");
        assert!(!cfg.is_lite, "27 layers alone must not imply lite");
    }

    #[test]
    fn deepseek2_missing_key_length_mla_is_refused() {
        let keys: Vec<_> = DEEPSEEK2_BASE_KEYS
            .iter()
            .filter(|(k, _)| *k != "attention.key_length_mla")
            .copied()
            .collect();
        let Err(err) = config_from_fixture("ds2-no-klen", &deepseek2_fixture(&keys)) else {
            panic!("a deepseek2 GGUF without key_length_mla must fail");
        };
        assert_eq!(
            err.to_string(),
            "deepseek2.attention.key_length_mla missing: deepseek2 MLA head dims cannot be derived"
        );
    }

    #[test]
    fn deepseek2_missing_value_length_mla_is_refused() {
        let keys: Vec<_> = DEEPSEEK2_BASE_KEYS
            .iter()
            .filter(|(k, _)| *k != "attention.value_length_mla")
            .copied()
            .collect();
        let Err(err) = config_from_fixture("ds2-no-vlen", &deepseek2_fixture(&keys)) else {
            panic!("a deepseek2 GGUF without value_length_mla must fail");
        };
        assert_eq!(
            err.to_string(),
            "deepseek2.attention.value_length_mla missing: deepseek2 MLA head dims cannot be derived"
        );
    }

    /// Mirrors `GGML_ASSERT(n_embd_head_qk_nope >= 1)`: an all-rotary key width leaves no NOPE
    /// dim, which the old code clamped to 1 and ran anyway.
    #[test]
    fn deepseek2_zero_qk_nope_is_refused() {
        let keys = deepseek2_keys_with(&[("attention.key_length_mla", 64)]);
        let Err(err) = config_from_fixture("ds2-zero-nope", &deepseek2_fixture(&keys)) else {
            panic!("key_length_mla == rope.dimension_count must fail");
        };
        assert_eq!(
            err.to_string(),
            "deepseek2 MLA: attention.key_length_mla (64) must exceed rope.dimension_count (64)"
        );
    }

    #[test]
    fn deepseek2_zero_v_head_dim_is_refused() {
        let keys = deepseek2_keys_with(&[("attention.value_length_mla", 0)]);
        let Err(err) = config_from_fixture("ds2-zero-vhd", &deepseek2_fixture(&keys)) else {
            panic!("value_length_mla == 0 must fail");
        };
        assert_eq!(
            err.to_string(),
            "deepseek2 MLA: attention.value_length_mla must be positive"
        );
    }

    /// B44: gating 3 is softmax-over-selected-weights, which infr has no `MoeGating` for — it
    /// must refuse rather than quietly route as plain softmax. 0 (key absent) stays softmax.
    #[test]
    fn deepseek2_unsupported_expert_gating_func_is_refused() {
        let mut keys = DEEPSEEK2_BASE_KEYS.to_vec();
        keys.push(("expert_gating_func", 3));
        let Err(err) = config_from_fixture("ds2-gating3", &deepseek2_fixture(&keys)) else {
            panic!("expert_gating_func = 3 must fail");
        };
        assert_eq!(
            err.to_string(),
            "deepseek2.expert_gating_func = 3 is not supported"
        );

        let cfg = config_from_fixture("ds2-gating0", &deepseek2_fixture(DEEPSEEK2_BASE_KEYS))
            .expect("absent expert_gating_func must still parse");
        assert_eq!(
            cfg.moe.expect("deepseek2 fixture has experts").gating,
            infr_core::graph::MoeGating::Softmax,
        );
    }

    /// B44: the YaRN knobs the loader used to drop. Absent keys must land on llama.cpp's own
    /// `llama_hparams` defaults, present ones must be read.
    #[test]
    fn deepseek2_yarn_knobs_are_read() {
        let cfg = config_from_fixture("ds2-yarn-default", &deepseek2_fixture(DEEPSEEK2_BASE_KEYS))
            .expect("base deepseek2 fixture must parse");
        assert_eq!(cfg.rope_attn_factor, 1.0);
        assert_eq!(cfg.rope_yarn_beta_fast, 32.0);
        assert_eq!(cfg.rope_yarn_beta_slow, 1.0);

        let mut keys = DEEPSEEK2_BASE_KEYS.to_vec();
        keys.push(("rope.scaling.yarn_beta_fast", 16));
        keys.push(("rope.scaling.yarn_beta_slow", 2));
        keys.push(("rope.scaling.attn_factor", 3));
        let cfg = config_from_fixture("ds2-yarn-set", &deepseek2_fixture(&keys))
            .expect("deepseek2 fixture with YaRN knobs must parse");
        assert_eq!(cfg.rope_yarn_beta_fast, 16.0);
        assert_eq!(cfg.rope_yarn_beta_slow, 2.0);
        assert_eq!(cfg.rope_attn_factor, 3.0);
    }
}
