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
//! Everything above [`deepseek2_model`] is arch-agnostic: [`Meta`] (the GGUF value types this
//! harness writes), [`TensorSpec`] + [`Fill`] (name, ggml-order shape, deterministic values),
//! [`SyntheticModel`] (the whole file as data) and its writer, and [`TempGguf`]. A later stage adds
//! ONE function — its own `deepseek32_model(&Ds32Dims) -> SyntheticModel` — listing that arch's
//! metadata keys and tensor names/shapes, and reuses the rest unchanged.
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

/// Build the whole GGUF description for a `deepseek2` model of `d`. Metadata keys are the ones
/// `Config::from_gguf`'s `deepseek2` branch reads; tensor names are the ones `seam/runner.rs`'s
/// `wload` MLA/MoE arms ask for.
fn deepseek2_model(d: &Ds2Dims) -> SyntheticModel {
    assert!(
        d.n_expert.is_multiple_of(d.n_groups.max(1)),
        "expert_group_count must divide expert_count"
    );
    let u = |k: &str, v: usize| (format!("deepseek2.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f32| (format!("deepseek2.{k}"), Meta::F32(v));
    let mut meta = vec![
        (
            "general.architecture".to_string(),
            Meta::Str("deepseek2".to_string()),
        ),
        u("block_count", d.n_layer),
        u("embedding_length", d.n_embd),
        u("feed_forward_length", d.n_ff),
        u("attention.head_count", d.n_head),
        u("context_length", 256),
        f("attention.layer_norm_rms_epsilon", 1e-6),
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
            "deepseek2.rope.scaling.type".to_string(),
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
        (
            "deepseek2.expert_weights_norm".to_string(),
            Meta::Bool(true),
        ),
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
