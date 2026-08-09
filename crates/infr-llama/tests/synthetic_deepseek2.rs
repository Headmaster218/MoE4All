//! Synthetic-GGUF harness: a tiny but structurally complete `deepseek2` model, built in memory,
//! written to a temp file, and driven through the REAL loader (`Config::from_gguf`) and the REAL
//! seam — on CPU unconditionally, and on Vulkan behind `#[ignore]`.
//!
//! # Why this exists
//!
//! `docs/deepseek.md` § "Why this order": stages 3 (`deepseek32`/V3.2) and 4 (`deepseek4`/V4) have
//! **no model small enough to develop against** — V3.2 is 671B and V4-Flash's smallest quant is
//! 82.5 GB — so "stages 1–2 must leave behind MLA and MoE-routing pieces that are independently
//! tested". `docs/backlog.md` B46 names the two pieces stage 3 inherits verbatim that no test on a
//! real model reaches: **group-limited routing** (`n_expert_groups > 1`) and the **`exp_probs_b`
//! router bias**. V2-Lite — the only DeepSeek small enough to run here — ships
//! `expert_group_count = 1` and carries no bias tensor, so both branches were exercised only by
//! `seam_op_parity.rs`'s op-level `moe_groups_bias_parity`, never through a model load.
//!
//! This file closes that: the routing metadata and the bias tensor come off DISK, through
//! `Config::from_gguf` and the `wload`/`wpush`/emit lockstep in `seam/runner.rs`, into a real
//! prefill on both backends.
//!
//! # Adding an architecture
//!
//! Everything above [`mla_model`] is arch-agnostic: [`Meta`] (the GGUF value types this harness
//! writes), [`TensorSpec`] + [`Fill`] (name, ggml-order shape, deterministic values),
//! [`SyntheticModel`] (the whole file as data) and its writer, and [`TempGguf`].
//!
//! Stage 3 (`deepseek32`/V3.2) took that offer and found it could reuse more: V3.2 IS deepseek2's
//! absorbed MLA plus a lightning indexer, so rather than a second model function listing the same
//! thirty tensors, [`mla_model`] gained an `arch` parameter (every model key is `{arch}.`-prefixed)
//! and an optional [`IndexerDims`]. `synthetic_deepseek32_is_deepseek2_plus_the_indexer` then
//! ASSERTS the containment the sharing assumes. A future arch that is not a DeepSeek-MLA variant
//! should write its own function instead — the harness below [`TempGguf`] is still what it reuses.
//!
//! The fill is seeded by the tensor NAME, not by its position in the file, which is what makes the
//! differential tests below valid: adding or removing `exp_probs_b.bias` changes that tensor and
//! nothing else, so two variants differ only in the thing under test.
//!
//! ```text
//! cargo test -p infr-llama --test synthetic_deepseek2
//! cargo test -p infr-llama --test synthetic_deepseek2 -- --include-ignored   # + Vulkan
//! ```

use infr_core::WeightSource;
use std::path::{Path, PathBuf};

// ─── GGUF metadata values ────────────────────────────────────────────────────────

/// A GGUF metadata value, restricted to the type tags this harness writes. Tags are the ggml
/// `gguf_type` enum (see `infr-gguf`'s `read_meta_value_at_depth`).
#[derive(Clone, Debug, PartialEq)]
enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// An array of STRINGs — the vocab and merge lists.
    StrArr(Vec<String>),
}

fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// A GGUF string: u64 byte length, then the UTF-8 bytes (no NUL).
fn push_gguf_str(b: &mut Vec<u8>, s: &str) {
    push_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

impl Meta {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Meta::U32(v) => {
                push_u32(b, 4);
                push_u32(b, *v);
            }
            Meta::F32(v) => {
                push_u32(b, 6);
                b.extend_from_slice(&v.to_le_bytes());
            }
            Meta::Bool(v) => {
                push_u32(b, 7);
                b.push(u8::from(*v));
            }
            Meta::Str(s) => {
                push_u32(b, 8);
                push_gguf_str(b, s);
            }
            Meta::StrArr(a) => {
                push_u32(b, 9);
                push_u32(b, 8); // element type: STRING
                push_u64(b, a.len() as u64);
                for s in a {
                    push_gguf_str(b, s);
                }
            }
        }
    }
}

// ─── deterministic tensor values ─────────────────────────────────────────────────

/// Stable FNV-1a-64 — the per-tensor RNG seed. Not `DefaultHasher`, which is not stable across
/// toolchains and would make the "same file every run" promise a lie on the next compiler.
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// SplitMix64 finalizer — the whole RNG. Stateless, so element `i` of a tensor depends only on
/// `(name, i)` and never on how many tensors came before it.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic value in `[-1, 1)` for element `i` of the tensor seeded by `seed`.
fn uniform(seed: u64, i: usize) -> f32 {
    let h = splitmix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // 24 bits → an exactly-representable f32 in [0, 1).
    ((h >> 40) as f32) / f32::from(1u16 << 12) / f32::from(1u16 << 12) * 2.0 - 1.0
}

/// How a tensor's values are produced.
#[derive(Clone, Debug)]
enum Fill {
    /// Pseudo-random in `[-amp, amp]`.
    Rand(f32),
    /// `1 + 0.1·u` — an RMSNorm gain, kept near 1 so activations stay in a sane range.
    Gain,
    /// Verbatim values — the router bias, whose exact numbers are the point (see [`FORCE_BIAS`]).
    Exact(Vec<f32>),
}

/// One tensor of the synthetic model. `shape` is in GGUF/ggml order — `ne0` (the fastest,
/// in-features dimension) FIRST, matching the shape column of `docs/deepseek.md` § Stage 2's tensor
/// table and what `Config::from_gguf` reads off `token_embd.weight`.
#[derive(Clone, Debug)]
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    fill: Fill,
}

impl TensorSpec {
    fn new(name: impl Into<String>, shape: Vec<usize>, fill: Fill) -> Self {
        Self {
            name: name.into(),
            shape,
            fill,
        }
    }

    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn values(&self) -> Vec<f32> {
        let seed = fnv1a64(&self.name);
        match &self.fill {
            Fill::Rand(amp) => (0..self.numel()).map(|i| uniform(seed, i) * amp).collect(),
            Fill::Gain => (0..self.numel())
                .map(|i| 1.0 + 0.1 * uniform(seed, i))
                .collect(),
            Fill::Exact(v) => {
                assert_eq!(v.len(), self.numel(), "{}: Exact fill length", self.name);
                v.clone()
            }
        }
    }
}

// ─── the file ────────────────────────────────────────────────────────────────────

/// GGUF's default `general.alignment`, which this harness does not override.
const ALIGN: usize = 32;

/// A whole synthetic GGUF as data: the metadata KVs in write order, and the tensors. Arch-agnostic
/// — see the module doc's "Adding an architecture".
struct SyntheticModel {
    meta: Vec<(String, Meta)>,
    tensors: Vec<TensorSpec>,
}

impl SyntheticModel {
    /// Serialize to GGUF v3 bytes. Every tensor is F32 (ggml type 0): the seam host-dequants
    /// `attn_k_b`/`attn_v_b` anyway (`seam/runner.rs`'s `wload`), and a quantized fixture would test
    /// the dequantizers rather than the routing this file is about.
    fn to_gguf_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        push_u32(&mut b, 0x4655_4747); // "GGUF"
        push_u32(&mut b, 3); // version
        push_u64(&mut b, self.tensors.len() as u64);
        push_u64(&mut b, self.meta.len() as u64);
        for (k, v) in &self.meta {
            push_gguf_str(&mut b, k);
            v.write(&mut b);
        }
        // Tensor directory. Offsets are relative to the data region and each is ALIGN-aligned.
        let mut offset = 0usize;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            offsets.push(offset);
            offset += (t.numel() * 4).div_ceil(ALIGN) * ALIGN;
        }
        for (t, off) in self.tensors.iter().zip(&offsets) {
            push_gguf_str(&mut b, &t.name);
            push_u32(&mut b, t.shape.len() as u32);
            for d in &t.shape {
                push_u64(&mut b, *d as u64);
            }
            push_u32(&mut b, 0); // GGML_TYPE_F32
            push_u64(&mut b, *off as u64);
        }
        while !b.len().is_multiple_of(ALIGN) {
            b.push(0);
        }
        let data_start = b.len();
        b.resize(data_start + offset, 0);
        for (t, off) in self.tensors.iter().zip(&offsets) {
            let base = data_start + off;
            for (i, v) in t.values().iter().enumerate() {
                b[base + i * 4..base + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }
}

/// A synthetic GGUF on disk, removed when the test drops it. `tempfile` is not a dev-dependency of
/// this crate, so this follows `config.rs`'s own fixture pattern (`std::env::temp_dir()` + an
/// explicit unlink) rather than adding one; the counter keeps concurrently-running tests in this
/// binary from colliding on a name.
struct TempGguf(PathBuf);

impl TempGguf {
    fn write(tag: &str, model: &SyntheticModel) -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("infr-synth-{tag}-{}-{n}.gguf", std::process::id()));
        std::fs::write(&path, model.to_gguf_bytes()).expect("write synthetic GGUF");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ─── the deepseek2 model description ─────────────────────────────────────────────

/// Every dimension of the synthetic `deepseek2` model. The MLA relationships `config.rs` derives
/// are spelled out on the fields that bind them; violating one is refused by the loader, which
/// `deepseek2_mla_head_dims_match_reference` (config.rs) already pins.
#[derive(Clone, Debug)]
struct Ds2Dims {
    n_layer: usize,
    /// `nextn_predict_layers` — trailing NextN/MTP blocks that `block_count` INCLUDES and that the
    /// trunk loop must not walk. `0` for every deepseek2 case (V2 ships none).
    n_layer_nextn: usize,
    /// `attention.layer_norm_rms_epsilon`. Deliberately NOT `deepseek32`'s hardcoded LayerNorm
    /// epsilon, so `Config::norm_eps` and `Config::rms_eps` can be told apart.
    rms_eps: f32,
    /// `leading_dense_block_count` — layers `< this` run a plain dense SwiGLU at `n_ff`; the rest
    /// are MoE. DeepSeek's "first N dense, rest MoE" threshold, not a periodic step.
    n_dense_lead: usize,
    n_embd: usize,
    n_head: usize,
    n_ff: usize,
    vocab: usize,
    /// Non-zero ⇒ the `wq_a → q_a_norm → wq_b` LoRA query path. V2-Lite is the LITE variant
    /// (direct `attn_q`), so this is the branch a real model has never exercised here — and it is
    /// the only one stage 3 has (`deepseek32` makes `q_lora_rank` mandatory).
    q_lora_rank: usize,
    kv_lora_rank: usize,
    /// `rope.dimension_count` — the decoupled-rope width, shared by `q_pe` and the single `k_pe`.
    qk_rope_dim: usize,
    /// `attention.key_length_mla`; `head_k_mla` (the NOPE width) is this MINUS `qk_rope_dim`.
    key_length_mla: usize,
    /// `attention.value_length_mla`, which IS `v_head_dim`.
    value_length_mla: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    /// `expert_shared_count`; the shared expert's width is `n_ff_exp * this` (llama.cpp fuses the
    /// shared experts into one wider branch).
    n_expert_shared: usize,
    /// `expert_group_count` / `expert_group_used_count`. `n_expert` must divide by the count.
    n_groups: usize,
    n_groups_used: usize,
    /// `blk.*.exp_probs_b.bias` — omitted entirely when `None`, like every V2 GGUF.
    exp_probs_b: Option<Vec<f32>>,
}

impl Ds2Dims {
    /// The NOPE width, derived exactly as `config.rs` does.
    fn qk_nope(&self) -> usize {
        self.key_length_mla - self.qk_rope_dim
    }

    /// The cache row width: ONE row per token per layer, `[latent | rope]`. V is a prefix VIEW of
    /// it, never a second cache.
    fn kv_row(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_dim
    }

    fn shexp_ff(&self) -> usize {
        self.n_ff_exp * self.n_expert_shared
    }
}

/// The `deepseek32` lightning indexer's three hyperparameters — the only metadata V3.2 adds to
/// [`Ds2Dims`]. See `docs/deepseek.md` § Stage 3.
#[derive(Clone, Debug)]
struct IndexerDims {
    /// `attention.indexer.head_count` — the indexer's QUERY heads. One key head serves all of them.
    n_head: usize,
    /// `attention.indexer.key_length` — the per-head key/query width, and the width of `k_norm`.
    head_size: usize,
    /// `attention.indexer.top_k` — how many keys survive to the real attention.
    top_k: usize,
}

/// The synthetic model's dimensions. Small, but every MLA relationship holds and every width the
/// Vulkan kernels constrain is legal: the expert id-GEMV decodes 32-element sub-blocks, so `n_embd`
/// and `n_ff_exp` are ≥ 32 and multiples of 32; `mla.comp` packs the f16 KV row two-per-`uint`, so
/// `kv_row()` is even. `qk_nope`, `kv_lora_rank` and `value_length_mla` are deliberately three
/// DIFFERENT numbers, so `wk_b` `[qk_nope, kv_lora, n_head]` and `wv_b` `[kv_lora, v_head, n_head]`
/// have different shapes — swapping them (the classic MLA porting bug, `docs/deepseek.md` § Stage
/// 2) cannot go unnoticed the way it would if the dims coincided as they do on V2-Lite/V3.
fn ds2_dims(n_groups: usize, n_groups_used: usize, exp_probs_b: Option<Vec<f32>>) -> Ds2Dims {
    Ds2Dims {
        n_layer: 3,
        n_layer_nextn: 0,
        rms_eps: 1e-6,
        n_dense_lead: 1,
        n_embd: 64,
        n_head: 2,
        n_ff: 64,
        vocab: 64,
        q_lora_rank: 32,
        kv_lora_rank: 32,
        qk_rope_dim: 16,
        key_length_mla: 64, // ⇒ qk_nope = 48
        value_length_mla: 48,
        n_expert: N_EXPERT,
        n_used: N_USED,
        n_ff_exp: 32,
        n_expert_shared: 1,
        n_groups,
        n_groups_used,
        exp_probs_b,
    }
}

/// Build the whole GGUF description for an absorbed-MLA DeepSeek model of `d` under `arch`
/// (`deepseek2` or `deepseek32` — every model key is `{arch}.`-prefixed, which is the whole reason
/// the arch string is a parameter rather than a literal).
///
/// `indexer` is `Some` only for `deepseek32`: it adds the three `attention.indexer.*` keys and five
/// per-layer tensors and changes NOTHING else, because V3.2 genuinely is V2's absorbed MLA plus the
/// lightning indexer. Keeping one builder is what makes "the deepseek32 model is the deepseek2 one
/// plus the indexer" a property the file can assert rather than a claim in a comment.
fn mla_model(arch: &str, d: &Ds2Dims, indexer: Option<&IndexerDims>) -> SyntheticModel {
    assert!(
        d.n_expert.is_multiple_of(d.n_groups.max(1)),
        "expert_group_count must divide expert_count"
    );
    let u = |k: &str, v: usize| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f32| (format!("{arch}.{k}"), Meta::F32(v));
    let mut meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str(arch.to_string()),
        ),
        // `block_count` COUNTS the NextN blocks; the trunk is `block_count - nextn_predict_layers`.
        u("block_count", d.n_layer + d.n_layer_nextn),
        u("embedding_length", d.n_embd),
        u("feed_forward_length", d.n_ff),
        u("attention.head_count", d.n_head),
        u("context_length", 256),
        f("attention.layer_norm_rms_epsilon", d.rms_eps),
        // MLA geometry.
        u("attention.q_lora_rank", d.q_lora_rank),
        u("attention.kv_lora_rank", d.kv_lora_rank),
        u("attention.key_length_mla", d.key_length_mla),
        u("attention.value_length_mla", d.value_length_mla),
        u("rope.dimension_count", d.qk_rope_dim),
        f("rope.freq_base", 10000.0),
        // YaRN, in the shape the V2-Lite GGUF declares it (`docs/deepseek.md` open question 6):
        // `type = yarn` is what makes llama.cpp run the FULL ramp at every context length, and the
        // convert script writes `0.1 * mscale_all_dim`, which the loader divides back out.
        (
            format!("{arch}.rope.scaling.type"),
            Meta::Str("yarn".to_string()),
        ),
        f("rope.scaling.factor", 40.0),
        u("rope.scaling.original_context_length", 128),
        f("rope.scaling.yarn_log_multiplier", 0.0707),
        // MoE.
        u("expert_count", d.n_expert),
        u("expert_used_count", d.n_used),
        u("expert_feed_forward_length", d.n_ff_exp),
        u("expert_shared_count", d.n_expert_shared),
        u("leading_dense_block_count", d.n_dense_lead),
        u("expert_gating_func", 2), // sigmoid — V3's scoring_func
        (format!("{arch}.expert_weights_norm"), Meta::Bool(true)),
        f("expert_weights_scale", 2.5),
        u("expert_group_count", d.n_groups),
        u("expert_group_used_count", d.n_groups_used),
        // The minimum tokenizer `build_tokenizer` accepts: a gpt2-model vocab. `.merges` may be
        // empty (nothing here encodes text — the tests hand token ids straight to the seam), but
        // the key must exist or the build fails. `.token_type` is optional and omitted.
        (
            "tokenizer.ggml.model".to_string(),
            Meta::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Meta::StrArr((0..d.vocab).map(|i| format!("t{i}")).collect()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Meta::StrArr(Vec::new()),
        ),
        ("tokenizer.ggml.eos_token_id".to_string(), Meta::U32(2)),
    ];
    if d.n_layer_nextn > 0 {
        meta.push(u("nextn_predict_layers", d.n_layer_nextn));
    }
    if let Some(ix) = indexer {
        meta.push(u("attention.indexer.head_count", ix.n_head));
        meta.push(u("attention.indexer.key_length", ix.head_size));
        meta.push(u("attention.indexer.top_k", ix.top_k));
    }
    meta.sort_by(|a, b| a.0.cmp(&b.0));

    let w = Fill::Rand(0.25);
    let mut tensors = vec![
        TensorSpec::new("token_embd.weight", vec![d.n_embd, d.vocab], w.clone()),
        TensorSpec::new("output_norm.weight", vec![d.n_embd], Fill::Gain),
        TensorSpec::new("output.weight", vec![d.n_embd, d.vocab], w.clone()),
    ];
    for l in 0..d.n_layer {
        let p = |s: &str| format!("blk.{l}.{s}");
        tensors.push(TensorSpec::new(
            p("attn_norm.weight"),
            vec![d.n_embd],
            Fill::Gain,
        ));
        // MLA, non-lite: the LoRA query path plus the compressed KV path.
        tensors.push(TensorSpec::new(
            p("attn_q_a.weight"),
            vec![d.n_embd, d.q_lora_rank],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_q_a_norm.weight"),
            vec![d.q_lora_rank],
            Fill::Gain,
        ));
        tensors.push(TensorSpec::new(
            p("attn_q_b.weight"),
            vec![d.q_lora_rank, d.n_head * d.key_length_mla],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_kv_a_mqa.weight"),
            vec![d.n_embd, d.kv_row()],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_kv_a_norm.weight"),
            vec![d.kv_lora_rank],
            Fill::Gain,
        ));
        // `wk_b` is TRANSPOSED relative to the HF weight and `wv_b` is not — the conversion script
        // calls `.transpose(1, 2)` on `k_b` only (docs/deepseek.md § Stage 2).
        tensors.push(TensorSpec::new(
            p("attn_k_b.weight"),
            vec![d.qk_nope(), d.kv_lora_rank, d.n_head],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_v_b.weight"),
            vec![d.kv_lora_rank, d.value_length_mla, d.n_head],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("attn_output.weight"),
            vec![d.n_head * d.value_length_mla, d.n_embd],
            w.clone(),
        ));
        // The lightning indexer sits on EVERY layer, dense-lead included — `deepseek32.cpp` creates
        // these five outside the dense/MoE branch. `k_norm` carries a weight AND a bias under one
        // GGUF name because it is a mean-centred LayerNorm, not an RMSNorm.
        if let Some(ix) = indexer {
            tensors.push(TensorSpec::new(
                p("indexer.k_norm.weight"),
                vec![ix.head_size],
                Fill::Gain,
            ));
            tensors.push(TensorSpec::new(
                p("indexer.k_norm.bias"),
                vec![ix.head_size],
                Fill::Rand(0.1),
            ));
            tensors.push(TensorSpec::new(
                p("indexer.proj.weight"),
                vec![d.n_embd, ix.n_head],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer.attn_k.weight"),
                vec![d.n_embd, ix.head_size],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("indexer.attn_q_b.weight"),
                vec![d.q_lora_rank, ix.n_head * ix.head_size],
                w.clone(),
            ));
        }
        tensors.push(TensorSpec::new(
            p("ffn_norm.weight"),
            vec![d.n_embd],
            Fill::Gain,
        ));
        if l < d.n_dense_lead {
            tensors.push(TensorSpec::new(
                p("ffn_gate.weight"),
                vec![d.n_embd, d.n_ff],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("ffn_up.weight"),
                vec![d.n_embd, d.n_ff],
                w.clone(),
            ));
            tensors.push(TensorSpec::new(
                p("ffn_down.weight"),
                vec![d.n_ff, d.n_embd],
                w.clone(),
            ));
            continue;
        }
        tensors.push(TensorSpec::new(
            p("ffn_gate_inp.weight"),
            vec![d.n_embd, d.n_expert],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_gate_exps.weight"),
            vec![d.n_embd, d.n_ff_exp, d.n_expert],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_up_exps.weight"),
            vec![d.n_embd, d.n_ff_exp, d.n_expert],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_down_exps.weight"),
            vec![d.n_ff_exp, d.n_embd, d.n_expert],
            w.clone(),
        ));
        if let Some(bias) = &d.exp_probs_b {
            tensors.push(TensorSpec::new(
                p("exp_probs_b.bias"),
                vec![d.n_expert],
                Fill::Exact(bias.clone()),
            ));
        }
        tensors.push(TensorSpec::new(
            p("ffn_gate_shexp.weight"),
            vec![d.n_embd, d.shexp_ff()],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_up_shexp.weight"),
            vec![d.n_embd, d.shexp_ff()],
            w.clone(),
        ));
        tensors.push(TensorSpec::new(
            p("ffn_down_shexp.weight"),
            vec![d.shexp_ff(), d.n_embd],
            w.clone(),
        ));
    }
    SyntheticModel { meta, tensors }
}

/// [`mla_model`] as `deepseek2` — the arch every test above this line uses.
fn deepseek2_model(d: &Ds2Dims) -> SyntheticModel {
    mla_model(infr_llama::arch::DEEPSEEK2, d, None)
}

// ─── the routing cases ───────────────────────────────────────────────────────────

const N_EXPERT: usize = 16;
const N_USED: usize = 2;
const N_GROUPS: usize = 4;
const N_GROUPS_USED: usize = 2;

/// The router bias that FORCES group-limited routing to disagree with a flat top-k, for every
/// token, whatever the router logits are.
///
/// Gating is sigmoid, so every unbiased prob `p` is strictly inside `(0, 1)`; the selection score
/// is `p + bias` (bias affects SELECTION only). With four groups of four:
///
/// | group | experts | biased scores                | group score = top-2 sum |
/// | ----: | ------- | ---------------------------- | ----------------------- |
/// |     0 | 0–3     | e0 ∈ (9,10), rest ∈ (0,1)    | (9, 11)                 |
/// |     1 | 4–7     | e4,e5 ∈ (8,9), rest ∈ (0,1)  | (16, 18)                |
/// |     2 | 8–11    | e8,e9 ∈ (7,8), rest ∈ (0,1)  | (14, 16)                |
/// |     3 | 12–15   | all ∈ (0,1)                  | (0, 2)                  |
///
/// The four ranges are disjoint, so the group order is `1 > 2 > 0 > 3` unconditionally. The top
/// `expert_group_used_count = 2` groups are therefore `{1, 2}` — and group 0, which holds the
/// single highest-scoring expert in the whole layer, is MASKED OUT. So:
///
/// * grouped `top-2` = `{4, 5}` (their scores dominate e8/e9 and everything unbiased),
/// * flat `top-2` = `{0, 4 or 5}`.
///
/// **Expert 0 is routed if and only if group routing is off.** That is what makes
/// `synthetic_deepseek2_group_routing_changes_output` a test of the group mask rather than of the
/// weather: it cannot pass with `n_expert_groups = 1`, and it cannot pass if the mask is dropped.
///
/// It also pins the routing SET across backends, so the CPU-vs-Vulkan check measures arithmetic
/// rather than a routing near-tie flipping.
const FORCE_BIAS: [f32; N_EXPERT] = [
    9.0, 0.0, 0.0, 0.0, //
    8.0, 8.0, 0.0, 0.0, //
    7.0, 7.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0,
];

/// A bias that is the SAME on every expert. It cannot change any selection — a flat top-k order is
/// invariant under a uniform shift, and every group's top-2 sum shifts by exactly `2×` the same
/// constant — so a correct implementation, which reads the returned weights from the UNBIASED
/// probs, must produce byte-identical logits to a model with no bias tensor at all. An
/// implementation that weights from the biased probs cannot: `(p+c)/Σ(p+c) ≠ p/Σp`.
const UNIFORM_BIAS: f32 = 0.25;

/// The prompt every case is prefilled with: raw token ids, no tokenizer round-trip (the synthetic
/// vocab spells nothing).
const PROMPT: &[u32] = &[3, 11, 5, 40, 27, 8, 61, 14];

fn force_bias() -> Vec<f32> {
    FORCE_BIAS.to_vec()
}

/// The canonical model: V3's routing shape (group-limited routing ON, a non-uniform router bias) at
/// toy dimensions.
fn grouped_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias())))
}

/// The canonical model with group routing DISABLED, exactly as a V2-era GGUF declares it
/// (`expert_group_count = 1`). Byte-identical in every other respect, bias included.
fn flat_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(1, 1, Some(force_bias())))
}

/// The canonical model with a UNIFORM bias — see [`UNIFORM_BIAS`].
fn uniform_bias_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(
        N_GROUPS,
        N_GROUPS_USED,
        Some(vec![UNIFORM_BIAS; N_EXPERT]),
    ))
}

/// The canonical model with NO `exp_probs_b` tensor at all — what V2-Lite ships.
fn no_bias_model() -> SyntheticModel {
    deepseek2_model(&ds2_dims(N_GROUPS, N_GROUPS_USED, None))
}

// ─── running them ────────────────────────────────────────────────────────────────

/// Serialize the Vulkan tests against each other: each `prefill_logits_vulkan` opens its own
/// device session, and cargo runs a binary's tests in parallel. Mirrors `cpu_backend.rs`'s
/// `test_serial_lock`, and poison-tolerant for the same reason.
fn gpu_serial_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn load(tmp: &TempGguf) -> infr_llama::SeamModel {
    infr_llama::SeamModel::load_with(
        tmp.path(),
        None,
        std::sync::Arc::new(infr_llama::EngineConfig::default()),
    )
    .expect("synthetic model load")
}

/// Prefill `PROMPT` on the CPU reference backend and return the last row's logits.
fn cpu_logits(tag: &str, model: &SyntheticModel) -> Vec<f32> {
    let tmp = TempGguf::write(tag, model);
    load(&tmp).prefill_logits_cpu(PROMPT).expect("cpu prefill")
}

/// [`cpu_logits`]'s Vulkan twin.
fn vulkan_logits(tag: &str, model: &SyntheticModel) -> Vec<f32> {
    let tmp = TempGguf::write(tag, model);
    load(&tmp)
        .prefill_logits_vulkan(PROMPT)
        .expect("vulkan prefill")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vector lengths");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / v.len() as f64).sqrt() as f32
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    dot / (na * nb)
}

/// Assert the two runs routed differently: a difference that is a real fraction of the signal, not
/// float noise on an identical computation.
fn assert_routing_differs(what: &str, a: &[f32], b: &[f32]) {
    let d = max_abs_diff(a, b);
    let scale = rms(a).max(rms(b));
    println!("{what}: max|Δ| = {d:e}  (logit rms {scale:e})");
    assert!(
        d > 0.01 * scale,
        "{what}: the two runs are indistinguishable (max|Δ| = {d:e}, logit rms {scale:e}) — the \
         routing path under test did not execute"
    );
}

// ─── tests ───────────────────────────────────────────────────────────────────────

/// Every variant's tensors as `(name, values)`, for comparing two variants' weights directly.
fn weights_of(m: &SyntheticModel) -> Vec<(String, Vec<f32>)> {
    m.tensors
        .iter()
        .map(|t| (t.name.clone(), t.values()))
        .collect()
}

/// The premise every differential test below rests on: two variants differ in exactly the thing
/// under test and in nothing else. If the fill ever became order- or run-dependent — a `HashMap`
/// walk, a clock seed — `grouped vs flat` would still "differ", but for the wrong reason, and
/// nothing else here would notice.
#[test]
fn synthetic_deepseek2_variants_differ_only_in_the_thing_under_test() {
    // Same description twice ⇒ byte-identical files.
    assert_eq!(
        grouped_model().to_gguf_bytes(),
        grouped_model().to_gguf_bytes(),
        "the synthetic GGUF is not reproducible"
    );

    // Grouped vs flat: identical tensors end to end; only the group metadata moves.
    assert_eq!(
        weights_of(&grouped_model()),
        weights_of(&flat_model()),
        "grouped and flat must share every weight — only expert_group_count differs"
    );
    let (gm, fm) = (grouped_model(), flat_model());
    let diff: Vec<_> = gm
        .meta
        .iter()
        .zip(&fm.meta)
        .filter(|(a, b)| a != b)
        .map(|(a, _)| a.0.clone())
        .collect();
    assert_eq!(
        diff,
        vec![
            "deepseek2.expert_group_count".to_string(),
            "deepseek2.expert_group_used_count".to_string()
        ],
        "grouped and flat must differ in the group metadata and nothing else"
    );

    // Uniform-bias vs no-bias: same metadata, and the ONLY tensor difference is the bias itself.
    let (um, nm) = (uniform_bias_model(), no_bias_model());
    assert_eq!(um.meta, nm.meta, "the bias is a TENSOR, not metadata");
    let (uw, nw) = (weights_of(&um), weights_of(&nm));
    let extra: Vec<_> = uw
        .iter()
        .filter(|(n, _)| !nw.iter().any(|(m, _)| m == n))
        .map(|(n, _)| n.clone())
        .collect();
    assert_eq!(
        extra,
        vec![
            "blk.1.exp_probs_b.bias".to_string(),
            "blk.2.exp_probs_b.bias".to_string()
        ],
        "the no-bias variant must be the uniform-bias one minus exactly the bias tensors"
    );
    for (n, v) in &nw {
        let (_, u) = uw.iter().find(|(m, _)| m == n).expect("shared tensor");
        assert_eq!(u, v, "{n} must be identical across the two bias variants");
    }
}

/// The harness itself: the bytes it writes must survive `Gguf::open` and be read back by the REAL
/// loader as the model that was described. Without this, every test below could be asserting about
/// a file whose dims silently drifted from the ones the comments reason over.
#[test]
fn synthetic_deepseek2_round_trips_through_the_real_loader() {
    let d = ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias()));
    let tmp = TempGguf::write("roundtrip", &deepseek2_model(&d));
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");

    // Tensor directory: names, shapes and dtypes are what was written.
    let find = |name: &str| {
        g.tensors()
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} missing from the synthetic GGUF"))
    };
    assert_eq!(find("token_embd.weight").shape, vec![d.n_embd, d.vocab]);
    assert_eq!(
        find("blk.1.attn_k_b.weight").shape,
        vec![d.qk_nope(), d.kv_lora_rank, d.n_head],
        "wk_b is [qk_nope, kv_lora, n_head] on disk"
    );
    assert_eq!(
        find("blk.1.attn_v_b.weight").shape,
        vec![d.kv_lora_rank, d.value_length_mla, d.n_head],
        "wv_b is [kv_lora, v_head_dim, n_head] on disk — NOT wk_b's orientation"
    );
    assert_eq!(find("blk.1.exp_probs_b.bias").shape, vec![d.n_expert]);
    assert!(
        !g.tensors()
            .iter()
            .any(|t| t.name == "blk.0.exp_probs_b.bias"),
        "layer 0 is dense-lead: no router, no bias"
    );
    assert!(
        !g.tensors().iter().any(|t| t.name == "blk.0.attn_q.weight"),
        "the synthetic model is NON-lite (wq_a/q_a_norm/wq_b), the branch V2-Lite never takes"
    );
    assert_eq!(find("blk.0.attn_k_b.weight").dtype, infr_core::DType::F32);

    // The bias tensor's VALUES survive the write — the whole forcing argument rests on them.
    let raw = g
        .tensor_bytes_arc("blk.2.exp_probs_b.bias")
        .expect("bias bytes");
    let got: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, FORCE_BIAS.to_vec());

    // And the loader derives the config the tests reason about.
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert!(cfg.deepseek2);
    assert!(!cfg.is_lite);
    assert_eq!(cfg.n_layer, d.n_layer);
    assert_eq!(cfg.vocab, d.vocab);
    assert_eq!(cfg.head_k_mla, d.qk_nope());
    assert_eq!(cfg.v_head_dim, d.value_length_mla);
    assert_eq!(cfg.kv_lora_rank, d.kv_lora_rank);
    assert_eq!(cfg.qk_rope_dim, d.qk_rope_dim);
    assert_eq!(cfg.head_dim, d.key_length_mla);
    assert_eq!(cfg.n_kv, 1, "MLA caches one key head for every query head");
    assert_eq!(cfg.q_lora_rank, d.q_lora_rank);
    assert_eq!(cfg.n_layer_dense_lead, d.n_dense_lead);
    assert!(!cfg.is_moe_layer(0), "layer 0 is the dense lead");
    assert!(cfg.is_moe_layer(1) && cfg.is_moe_layer(2));
    assert_eq!(cfg.shexp_ff, d.shexp_ff());
    assert!(cfg.rope_scaling_yarn);
    // The convert script writes `0.1 * mscale_all_dim`; the loader divides it back out.
    assert!((cfg.rope_yarn_log_mul - 0.707).abs() < 1e-4);
    let moe = cfg.moe.expect("the synthetic model has experts");
    assert_eq!(moe.n_expert, N_EXPERT);
    assert_eq!(moe.n_used, N_USED);
    assert_eq!(moe.n_expert_groups, N_GROUPS as u32);
    assert_eq!(moe.n_expert_groups_used, N_GROUPS_USED as u32);
    assert_eq!(moe.gating, infr_core::graph::MoeGating::Sigmoid);
    assert!(moe.norm_w);
    assert_eq!(moe.scale, 2.5);
}

/// The whole model runs: MLA (non-lite LoRA query path, YaRN ramp, compressed single-row KV) plus a
/// dense-lead layer and two group-routed MoE layers, end to end on the CPU reference backend.
#[test]
fn synthetic_deepseek2_cpu_prefill_is_finite() {
    let logits = cpu_logits("finite", &grouped_model());
    assert_eq!(logits.len(), 64, "logits are [vocab]");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite logit in the synthetic deepseek2 CPU prefill: {logits:?}"
    );
    assert!(rms(&logits) > 1e-3, "logits are degenerate: {logits:?}");
}

/// **Group-limited routing executes and changes the answer.** The same weights, the same bias, the
/// same prompt — only `expert_group_count`/`expert_group_used_count` differ. By [`FORCE_BIAS`]'s
/// construction the grouped run routes `{4, 5}` and the flat run routes `{0, 4|5}` for every token,
/// so the outputs cannot agree unless the group mask never ran.
#[test]
fn synthetic_deepseek2_group_routing_changes_output() {
    let grouped = cpu_logits("grouped", &grouped_model());
    let flat = cpu_logits("flat", &flat_model());
    assert_routing_differs("cpu grouped vs flat", &grouped, &flat);
}

/// **The router bias participates in selection.** Group routing is on in both runs; only the bias
/// tensor differs (present-and-forcing vs absent). With the bias, `{4, 5}` are routed on every
/// token; without it the router's own probs decide.
///
/// This is also the only proof that the LOAD path ran: `wload`'s `exp_probs_b` arm is
/// presence-gated (`layer_has_epb`), so a tensor it failed to find would leave `exp_probs_b: None`
/// on the op and make this run identical to the no-bias one — which is exactly the failure this
/// asserts against.
#[test]
fn synthetic_deepseek2_router_bias_changes_output() {
    let biased = cpu_logits("biased", &grouped_model());
    let unbiased = cpu_logits("unbiased", &no_bias_model());
    assert_routing_differs("cpu forcing-bias vs no-bias", &biased, &unbiased);
}

/// **The bias affects SELECTION only; the weights come from the unbiased probs.** A uniform bias
/// cannot move any selection (see [`UNIFORM_BIAS`]), so the logits must match a no-bias model
/// EXACTLY. They do not if the returned weight is read from the biased probs — that is the one
/// perturbation this case exists to catch, and the reason it asserts equality rather than
/// difference.
#[test]
fn synthetic_deepseek2_uniform_router_bias_is_a_no_op() {
    let uniform = cpu_logits("uniform-bias", &uniform_bias_model());
    let none = cpu_logits("no-bias", &no_bias_model());
    let d = max_abs_diff(&uniform, &none);
    println!("cpu uniform-bias vs no-bias: max|Δ| = {d:e}");
    assert_eq!(
        d, 0.0,
        "a uniform router bias changed the output — the returned expert weights are being read \
         from the BIASED probs instead of the unbiased ones"
    );
}

// ─── deepseek32 (V3.2) ───────────────────────────────────────────────────────────
//
// Stage 3's LOAD path. There is no V3.2 GGUF anywhere — the model is 671B — so this is the only
// place the arch is exercised at all: `Config::from_gguf`'s `deepseek32` branch, and `wload`'s
// five extra per-layer tensors, both driven off a real file through the real loader. The graph is
// a later slice, so every run below ends at the seam's explicit refusal rather than in logits.

/// V3.2's indexer at toy dimensions. `head_size` is DELIBERATELY not `qk_rope_dim` and `n_head` is
/// not `n_head`: the indexer's four shaped tensors then have four distinct shapes, so a loader that
/// reached for the MLA head's dims instead of the indexer's could not still fit.
fn ds32_indexer() -> IndexerDims {
    IndexerDims {
        n_head: 4,
        head_size: 24,
        top_k: 5,
    }
}

/// The canonical `deepseek32` dims: the grouped/biased deepseek2 model, with a DIFFERENT
/// `rms_eps` from the 1e-6 `deepseek32.cpp` hardcodes for the indexer's LayerNorm, so
/// `Config::rms_eps` and `Config::norm_eps` cannot be confused for one another.
fn ds32_dims() -> Ds2Dims {
    Ds2Dims {
        rms_eps: 1e-5,
        ..ds2_dims(N_GROUPS, N_GROUPS_USED, Some(force_bias()))
    }
}

fn deepseek32_model() -> SyntheticModel {
    mla_model(
        infr_llama::arch::DEEPSEEK32,
        &ds32_dims(),
        Some(&ds32_indexer()),
    )
}

/// The five per-layer tensor names V3.2 adds, as they appear on disk (`llama-arch.cpp`'s
/// `LLM_TENSOR_INDEXER_*` format strings).
const INDEXER_TENSORS: [&str; 5] = [
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.proj.weight",
    "indexer.attn_k.weight",
    "indexer.attn_q_b.weight",
];

/// `m` minus one tensor. The count check is the point: a typo in `name` would otherwise remove
/// nothing and leave a "differential" test comparing a model with itself.
fn without_tensor(mut m: SyntheticModel, name: &str) -> SyntheticModel {
    let before = m.tensors.len();
    m.tensors.retain(|t| t.name != name);
    assert_eq!(
        m.tensors.len() + 1,
        before,
        "{name} was not in the model — nothing was removed"
    );
    m
}

/// [`without_tensor`]'s metadata twin.
fn without_meta(mut m: SyntheticModel, key: &str) -> SyntheticModel {
    let before = m.meta.len();
    m.meta.retain(|(k, _)| k != key);
    assert_eq!(
        m.meta.len() + 1,
        before,
        "{key} was not in the model — nothing was removed"
    );
    m
}

/// `Config::from_gguf`'s error for `m`, as a full `{:#}` chain. Panics if the file parses.
fn config_err(tag: &str, m: &SyntheticModel) -> String {
    let tmp = TempGguf::write(tag, m);
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let err = infr_llama::Config::from_gguf(&g).expect_err("this fixture must be refused");
    format!("{err:#}")
}

/// The error a CPU prefill of `m` fails with, as a full `{:#}` chain. Every `deepseek32` fixture
/// fails — a complete one at the graph build, an incomplete one inside `wload` — and WHICH of the
/// two is the assertion.
fn prefill_err(tag: &str, m: &SyntheticModel) -> String {
    let tmp = TempGguf::write(tag, m);
    let model = load(&tmp);
    let err = model
        .prefill_logits_cpu(PROMPT)
        .expect_err("deepseek32 cannot generate yet — this must not return logits");
    format!("{err:#}")
}

/// The message the seam refuses a `deepseek32` graph build with. Every load-path test below routes
/// through a prefill, so this is also the marker for "the load got all the way through".
const GRAPH_REFUSAL: &str = "arch=deepseek32 (DeepSeek V3.2) loads but cannot generate yet";

/// Every gate boolean and every derived dim of a `deepseek32` config, including where the
/// `deepseek2`-only gates land. `deepseek2` is TRUE here on purpose: MLA, the one-compressed-row KV
/// geometry, the group-limited MoE and the dense-lead threshold are all shared verbatim, so they
/// read one flag. `deepseek32` gates only what V3.2 adds.
#[test]
fn synthetic_deepseek32_config_gates() {
    let d = ds32_dims();
    let ix = ds32_indexer();
    let tmp = TempGguf::write("ds32-config", &deepseek32_model());
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");

    assert!(cfg.deepseek32, "deepseek32 gate");
    assert!(
        cfg.deepseek2,
        "deepseek32 must ALSO set deepseek2 — that flag is what gates MLA, the compressed KV row, \
         the MoE shape and the dense-lead threshold, all of which V3.2 shares verbatim"
    );
    assert!(!cfg.deepseek, "the V1 gate must stay false");
    assert!(
        !cfg.is_lite,
        "V3.2 has no lite variant — the LoRA query path is the only one it has"
    );
    assert!(!cfg.qk_norm, "no learned q/k-norm");
    assert!(!cfg.qkv_bias, "no attention biases");
    assert!(!cfg.permute_qk_neox);
    assert!(!cfg.sub_norm);
    assert!(!cfg.llama4);
    assert!(!cfg.qwen35);
    assert!(!cfg.gemma && !cfg.gemma4);
    assert!(!cfg.shexp_gated, "DeepSeek's shared expert is summed plain");

    // Indexer hparams, and the LayerNorm epsilon that is NOT the RMSNorm one.
    assert_eq!(cfg.indexer_n_head, ix.n_head);
    assert_eq!(cfg.indexer_head_size, ix.head_size);
    assert_eq!(cfg.indexer_top_k, ix.top_k);
    assert_eq!(cfg.norm_eps, 1e-6, "deepseek32.cpp hardcodes f_norm_eps");
    assert_eq!(cfg.rms_eps, d.rms_eps, "f_norm_rms_eps comes off the GGUF");
    assert_ne!(
        cfg.norm_eps, cfg.rms_eps,
        "the indexer's LayerNorm epsilon and the RMSNorm epsilon are separate hparams"
    );

    // MLA geometry, derived by the same code deepseek2 uses.
    assert_eq!(cfg.n_layer, d.n_layer);
    assert_eq!(cfg.vocab, d.vocab);
    assert_eq!(cfg.n_kv, 1, "MLA caches one key head for every query head");
    assert_eq!(cfg.head_dim, d.key_length_mla);
    assert_eq!(cfg.head_k_mla, d.qk_nope());
    assert_eq!(cfg.v_head_dim, d.value_length_mla);
    assert_eq!(cfg.kv_lora_rank, d.kv_lora_rank);
    assert_eq!(cfg.qk_rope_dim, d.qk_rope_dim);
    assert_eq!(cfg.q_lora_rank, d.q_lora_rank);
    assert_eq!(cfg.key_length, d.key_length_mla);

    // MoE, dense lead, YaRN — every one of them a `deepseek2` code path.
    assert_eq!(cfg.n_layer_dense_lead, d.n_dense_lead);
    assert!(!cfg.is_moe_layer(0), "layer 0 is the dense lead");
    assert!(cfg.is_moe_layer(1) && cfg.is_moe_layer(2));
    assert_eq!(cfg.shexp_ff, d.shexp_ff());
    assert!(cfg.rope_scaling_yarn);
    assert!((cfg.rope_yarn_log_mul - 0.707).abs() < 1e-4);
    let moe = cfg.moe.expect("deepseek32 is a MoE arch");
    assert_eq!(moe.gating, infr_core::graph::MoeGating::Sigmoid);
    assert_eq!(moe.n_expert_groups, N_GROUPS as u32);
    assert_eq!(moe.n_expert_groups_used, N_GROUPS_USED as u32);

    // And the flag really is arch-keyed: the deepseek2 twin leaves every V3.2 field at zero.
    let tmp2 = TempGguf::write("ds2-not-32", &grouped_model());
    let g2 = infr_gguf::Gguf::open(tmp2.path()).expect("open synthetic GGUF");
    let cfg2 = infr_llama::Config::from_gguf(&g2).expect("Config::from_gguf");
    assert!(cfg2.deepseek2 && !cfg2.deepseek32);
    assert_eq!(
        (
            cfg2.indexer_n_head,
            cfg2.indexer_head_size,
            cfg2.indexer_top_k
        ),
        (0, 0, 0)
    );
    assert_eq!(cfg2.norm_eps, 0.0, "deepseek2 emits no LayerNorm");
}

/// **The weight loader consumes all five indexer tensors, on every layer.**
///
/// A complete model reaches the graph-build refusal, which is only possible once `wload` has walked
/// every layer without complaint. Remove any ONE of the five from any layer and the SAME run stops
/// earlier, inside `wload`, naming the tensor it wanted — so a loader that quietly ignored these
/// files' extra tensors would fail this test on all ten of its cases.
#[test]
fn synthetic_deepseek32_load_consumes_every_indexer_tensor() {
    let complete = prefill_err("ds32-complete", &deepseek32_model());
    assert!(
        complete.contains(GRAPH_REFUSAL),
        "a complete deepseek32 model must load fully and stop at the graph build, got: {complete}"
    );

    // Layer 0 is the DENSE-lead layer and layer 2 is a MoE layer: the indexer is unconditional, so
    // both must demand all five.
    for l in [0usize, 2] {
        for suffix in INDEXER_TENSORS {
            let name = format!("blk.{l}.{suffix}");
            let err = prefill_err(
                &format!("ds32-no-{}-{l}", suffix.replace('.', "-")),
                &without_tensor(deepseek32_model(), &name),
            );
            println!("deepseek32 without {name}: {err}");
            assert!(
                err.contains(&format!("tensor not found: {name}")),
                "removing {name} must fail the weight load naming it, got: {err}"
            );
        }
    }
}

/// A misnamed tensor is the same failure as a missing one, and this is the case that would catch a
/// loader asking for llama.cpp's ENUM name (`indexer_k_norm`) instead of the on-disk one
/// (`indexer.k_norm`). Renaming is remove-plus-add, so the file stays otherwise complete.
#[test]
fn synthetic_deepseek32_misnamed_indexer_tensor_is_refused() {
    let mut m = deepseek32_model();
    let ix = ds32_indexer();
    m = without_tensor(m, "blk.1.indexer.proj.weight");
    m.tensors.push(TensorSpec::new(
        "blk.1.indexer_proj.weight",
        vec![ds32_dims().n_embd, ix.n_head],
        Fill::Rand(0.25),
    ));
    let err = prefill_err("ds32-misnamed", &m);
    println!("deepseek32 with a misnamed indexer.proj: {err}");
    assert!(
        err.contains("tensor not found: blk.1.indexer.proj.weight"),
        "a misnamed indexer tensor must be refused, got: {err}"
    );
}

/// **MLA is mandatory.** `deepseek32.cpp::load_arch_tensors` opens with
/// `if (!hparams.is_mla()) throw "DEEPSEEK32 architecture requires MLA"`, and `is_mla()` is exactly
/// "both MLA head-length keys are non-zero". There is no unabsorbed fallback for V3.2 at all.
#[test]
fn synthetic_deepseek32_requires_mla() {
    for key in ["attention.key_length_mla", "attention.value_length_mla"] {
        let full = format!("deepseek32.{key}");
        let err = config_err(
            &format!("ds32-no-{}", key.replace('.', "-")),
            &without_meta(deepseek32_model(), &full),
        );
        println!("deepseek32 without {full}: {err}");
        assert_eq!(
            err,
            format!("deepseek32 architecture requires MLA: {full} is missing or zero")
        );
    }
}

/// The keys `deepseek32.cpp` reads UNCONDITIONALLY where `deepseek2.cpp` tolerates their absence.
/// Both defaults would be silently wrong here: a missing `q_lora_rank` would read as the lite
/// variant V3.2 does not have, and a missing `expert_gating_func` would route softmax where V3.2 is
/// sigmoid.
#[test]
fn synthetic_deepseek32_mandatory_keys_are_required() {
    for key in [
        "attention.q_lora_rank",
        "expert_gating_func",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
    ] {
        let full = format!("deepseek32.{key}");
        let err = config_err(
            &format!("ds32-nokey-{}", key.replace('.', "-")),
            &without_meta(deepseek32_model(), &full),
        );
        println!("deepseek32 without {full}: {err}");
        assert!(
            err.contains(&full),
            "a deepseek32 GGUF without {full} must be refused naming it, got: {err}"
        );
    }
}

/// **NextN/MTP blocks are split off and skipped**, which is what `deepseek32.cpp` does with its
/// `i >= n_layer` → `TENSOR_SKIP` arm. `block_count` counts them, so the trunk is the difference;
/// the fixture carries no `blk.3` tensors at all and must still load, proving nothing walks them.
#[test]
fn synthetic_deepseek32_nextn_blocks_are_skipped() {
    let d = Ds2Dims {
        n_layer_nextn: 1,
        ..ds32_dims()
    };
    let m = mla_model(infr_llama::arch::DEEPSEEK32, &d, Some(&ds32_indexer()));
    assert!(
        !m.tensors.iter().any(|t| t.name.starts_with("blk.3.")),
        "the fixture must carry no NextN-block tensors — that is what makes the skip observable"
    );
    let tmp = TempGguf::write("ds32-nextn", &m);
    let g = infr_gguf::Gguf::open(tmp.path()).expect("open synthetic GGUF");
    let cfg = infr_llama::Config::from_gguf(&g).expect("Config::from_gguf");
    assert_eq!(cfg.n_layer_nextn, 1);
    assert_eq!(cfg.n_layer, d.n_layer, "the trunk excludes the NextN block");

    let err = prefill_err("ds32-nextn-load", &m);
    assert!(
        err.contains(GRAPH_REFUSAL),
        "the NextN fixture must load fully and stop at the graph build, got: {err}"
    );
}

/// The graph build refuses `deepseek32` with a message that says WHY, rather than panicking or —
/// far worse — emitting the deepseek2 graph and returning dense-attention logits that look fine.
#[test]
fn synthetic_deepseek32_graph_build_is_refused_clearly() {
    let err = prefill_err("ds32-refusal", &deepseek32_model());
    println!("deepseek32 graph build: {err}");
    assert_eq!(
        err,
        "arch=deepseek32 (DeepSeek V3.2) loads but cannot generate yet: its lightning indexer is \
         not implemented, and emitting the deepseek2 MLA graph without the indexer's top-k key \
         selection would produce silently wrong output. See docs/deepseek.md § Stage 3."
    );
}

/// The premise the load tests rest on: the deepseek32 fixture IS the deepseek2 one plus the
/// indexer. If the two builders drifted, "removing an indexer tensor breaks the load" could pass
/// for a reason that has nothing to do with the indexer.
#[test]
fn synthetic_deepseek32_is_deepseek2_plus_the_indexer() {
    let d = ds32_dims();
    let ds2 = mla_model(infr_llama::arch::DEEPSEEK2, &d, None);
    let ds32 = deepseek32_model();

    let extra: Vec<String> = ds32
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !ds2.tensors.iter().any(|t| &t.name == n))
        .collect();
    let mut want: Vec<String> = Vec::new();
    for l in 0..d.n_layer {
        for suffix in INDEXER_TENSORS {
            want.push(format!("blk.{l}.{suffix}"));
        }
    }
    assert_eq!(
        extra, want,
        "deepseek32 must add exactly the five indexer tensors, on every layer"
    );

    let strip = |m: &SyntheticModel| -> Vec<(String, Meta)> {
        m.meta
            .iter()
            .filter(|(k, _)| !k.contains("indexer"))
            .map(|(k, v)| (k.replacen("deepseek32.", "deepseek2.", 1), v.clone()))
            .collect()
    };
    assert_eq!(
        strip(&ds32),
        strip(&ds2)
            .into_iter()
            .map(|(k, v)| if k == "general.architecture" {
                (k, Meta::Str("deepseek32".to_string()))
            } else {
                (k, v)
            })
            .collect::<Vec<_>>(),
        "apart from the arch string and the three indexer keys, the two models' metadata is equal"
    );
}

// ─── Vulkan ──────────────────────────────────────────────────────────────────────

/// CPU vs Vulkan on the same synthetic file — the differential oracle. [`FORCE_BIAS`] pins the
/// routed set identically on both backends, so any divergence here is arithmetic (f16 KV, f16
/// weights, a different reduction order), not a routing near-tie.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek2_matches_cpu() {
    let _lk = gpu_serial_lock();
    let model = grouped_model();
    let cpu = cpu_logits("gpu-parity-cpu", &model);
    let gpu = vulkan_logits("gpu-parity-vk", &model);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "non-finite logit in the Vulkan prefill: {gpu:?}"
    );
    let cos = cosine(&cpu, &gpu);
    println!("synthetic deepseek2 cpu-vs-vulkan cosine = {cos}");
    let (cpu_top, gpu_top) = (argmax(&cpu), argmax(&gpu));
    println!("cpu argmax = {cpu_top}, vulkan argmax = {gpu_top}");
    assert_eq!(cpu_top, gpu_top, "CPU and Vulkan disagree on the top token");
    // Tighter than the real-model seam tests' `> 0.5` on purpose: those run 27 layers of Q4_K
    // weights and tolerate routing near-ties flipping, while here the routed set is pinned and the
    // whole model is three f32 layers, so the two backends agree to ~1e-13. A threshold anywhere
    // near the observed value would be a driver-noise tripwire; this one still fails on any real
    // kernel divergence.
    assert!(
        cos > 0.9999,
        "CPU/Vulkan logits diverged on the synthetic deepseek2 model: cosine = {cos}"
    );
}

/// The CPU routing cases, repeated on Vulkan: `moe_topk.comp`'s group-mask and bias branches have
/// their own implementation of every rule the CPU arm implements, so proving them on one backend
/// proves nothing about the other.
#[test]
#[ignore = "requires a Vulkan GPU: run with --include-ignored on a GPU box"]
fn gpu_synthetic_deepseek2_routing_paths_execute() {
    let _lk = gpu_serial_lock();
    let grouped = vulkan_logits("vk-grouped", &grouped_model());
    let flat = vulkan_logits("vk-flat", &flat_model());
    assert_routing_differs("vulkan grouped vs flat", &grouped, &flat);

    let unbiased = vulkan_logits("vk-unbiased", &no_bias_model());
    assert_routing_differs("vulkan forcing-bias vs no-bias", &grouped, &unbiased);

    let uniform = vulkan_logits("vk-uniform-bias", &uniform_bias_model());
    let d = max_abs_diff(&uniform, &unbiased);
    println!("vulkan uniform-bias vs no-bias: max|Δ| = {d:e}");
    assert_eq!(
        d, 0.0,
        "a uniform router bias changed the Vulkan output — moe_topk.comp is weighting from the \
         BIASED probs instead of the unbiased ones"
    );
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}
