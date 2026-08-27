//! Backend-agnostic compute IR.
//!
//! The model layer builds a [`Graph`] — an explicit, ordered list of semantic [`Op`]s over
//! typed [`TensorId`] handles — and a [`crate::backend::Backend`] compiles + executes it
//! however it likes (Vulkan SPIR-V, CPU loops, Metal MSL). See docs/plan.md
//! "The backend abstraction".
//!
//! ## Why an op-list, not a pure DAG
//!
//! The real transformer forward is imperative: it reuses scratch buffers, RoPEs in place, and
//! writes K/V into a persistent cache at a running offset. A pure SSA DAG can't express those
//! aliasing/stateful writes cleanly, so [`Graph`] is an **ordered list** of ops, each naming the
//! tensor handles it reads and the handle it writes (`dst`). Two ops may legally write the same
//! handle (in-place / scratch reuse) — order is significant, exactly like a command buffer.
//!
//! ## Composite ops
//!
//! Ops are *composite/semantic* (e.g. [`Op::Attention`], [`Op::QkNorm`]) rather than scalar
//! primitives, so a GPU backend can map each one straight to a hand-fused kernel (no perf loss)
//! while a CPU backend runs a plain loop. A future backend may either implement the composites
//! directly or add a lowering pass that decomposes them into primitives.

use crate::tensor::{TensorDesc, TensorId};

/// Attention masking mode. SWA layers (Gemma) mask beyond a sliding window; the rest are causal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMask {
    /// Causal full attention (every position attends to all earlier positions).
    Causal,
    /// Causal sliding-window attention with the given window size (in tokens).
    SlidingWindow(usize),
    /// DiffusionGemma canvas denoise mask (bidirectional, NOT causal — see
    /// `docs/diffusion-gemma.md`'s "Seam extensions" and the reference
    /// `llm_graph_input_attn_diffusion_decode::set_input` in `diffusion-gemma.cpp`): EVERY query
    /// row attends the SAME fixed range `[lo, kv_len)` regardless of its own row index — `pos`/
    /// per-row causal bounds are ignored entirely. `lo = 0` on full-attention layers (every
    /// prompt + canvas key visible); `lo = max(0, P - (n_swa-1))` on SWA layers (only the last
    /// `n_swa-1` prompt positions, but EVERY canvas key: canvas keys live in `[P, kv_len)` ⊆
    /// `[lo, kv_len)` on both layer types, since `lo <= P`). The caller (graph builder) computes
    /// `lo` from the prompt length `P` and the layer's SWA window; this variant only carries the
    /// already-resolved value.
    Canvas { lo: usize },
}

/// Activation used by the gated FFN (`act(gate) * up`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// SwiGLU: `silu(gate) * up` (Llama / Qwen).
    Silu,
    /// GeGLU: `gelu_tanh(gate) * up` (Gemma).
    Gelu,
    /// `sigmoid(gate) * up` (qwen35 output gate / silu-gated-RMSNorm uses Silu instead).
    Sigmoid,
}

/// Largest per-layer SwiGLU clamp limit that still means "this layer does NOT clamp". llama.cpp
/// gates both of its clamp sites on `limit > 1e-6` (`llm_graph_context::build_ffn` and
/// `build_moe_ffn`, `LLM_FFN_SILU` arms), so a per-layer entry at or below this is the DISABLED
/// state — not a request to squash the whole FFN into `[-limit, +limit]`, which is what a literal
/// `clamp(x, -0.0, 0.0)` on a layer whose configured limit is `0.0` would do.
pub const SWIGLU_CLAMP_MIN: f32 = 1e-6;

/// Resolve a per-layer clamp limit (DeepSeek V4's `swiglu_clamp_exp[il]` / `swiglu_clamp_shexp[il]`)
/// into the `swiglu_clamp` field [`Op::GatedAct`], [`Op::GatedActFused`] and [`Op::MoeFfn`] carry.
///
/// This is the ONE place llama.cpp's `limit > 1e-6` gate lives: a caller hands over the raw
/// per-layer array entry and gets `None` for the layers that do not clamp, so no backend has to
/// re-derive the threshold and no caller can pass `Some(0.0)` by reading the array directly.
pub fn swiglu_clamp(limit: f32) -> Option<f32> {
    (limit > SWIGLU_CLAMP_MIN).then_some(limit)
}

/// Router gating function for a routed-expert FFN ([`Op::MoeFfn`]) — how the per-expert logits
/// (`router · x`) become the selection scores + per-expert weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeGating {
    /// `probs = softmax(logits)` over all experts (qwen3moe / qwen35moe / diffusion-gemma).
    Softmax,
    /// `probs = sigmoid(logits)` per expert (llama4). Selection order is unchanged by the monotone
    /// sigmoid, so top-k picks the same experts as by raw logits.
    Sigmoid,
    /// `probs = sqrt(softplus(logits))` per expert (deepseek2/3/4 with `expert_gating_func=4`).
    /// Monotone in the logit like sigmoid; top-k picks the same experts as by raw logits.
    SqrtSoftplus,
}

/// Largest `hc_mult` (parallel residual-stream count) any backend accepts for DeepSeek V4's
/// hyper-connection ops. Every shipped V4 config uses 4 and llama.cpp `GGML_ASSERT`s exactly that;
/// this is the next power of two above it, chosen so a token's whole `hc × hc` mixing matrix
/// (`HYPER_CONNECT_MAX_MULT²` = 64 f32) fits in a fixed-size per-thread scratch array on every
/// backend — Sinkhorn iterates over the whole matrix, so it has to be materialised somewhere.
/// [`Op::HyperConnectMix`], [`Op::HyperConnectPre`] and [`Op::HyperConnectPost`] refuse anything
/// wider on the HOST, before a kernel that would read past that array is ever dispatched.
pub const HYPER_CONNECT_MAX_MULT: u32 = 8;

/// The two extra outputs [`Op::HyperConnectMix`] produces when it wraps a SUBLAYER, absent in the
/// model-head form. One struct rather than two `Option` fields because llama.cpp's `build_hc_pre`
/// writes both or neither: separate `Option`s would let a caller ask for a `comb` with no `post`,
/// a state no backend could act on and every one would have to reject at run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HyperGates {
    /// Per-stream output gates `[rows, hc]` f32 — [`Op::HyperConnectPost`]'s `post`.
    pub post: TensorId,
    /// Sinkhorn-normalised mixing matrix `[rows, hc, hc]` f32, flat `dst + hc*src` within a token
    /// — [`Op::HyperConnectPost`]'s `comb`.
    pub comb: TensorId,
}

/// Persistent cache encodings used by DeepSeek V4's specialized cache writer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dsv4CacheFormat {
    /// 512 values: 448 OCP E4M3 bytes, 64 BF16 values, and 8 UE8M0 scale bytes.
    Fp8Kv,
    /// 128 values: 64 packed E2M1 bytes and 4 UE8M0 scale bytes.
    Mxfp4,
}

/// How a tensor handle is provisioned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorKind {
    /// Per-step input bound at execute time via [`Bindings`] (e.g. the embedded hidden state,
    /// position ids, the KV cache). The backend does NOT allocate these.
    Input,
    /// Model weight bound from the loader via [`Bindings`]. Read-only.
    Weight,
    /// Backend-allocated scratch / activation, lives for the duration of one execute.
    Internal,
    /// An [`Internal`](TensorKind::Internal) tensor whose final value is read back by the caller
    /// (collected into [`Bindings::outputs`] after execute).
    Output,
}

/// Declaration of a tensor handle: its shape/dtype and how it's provisioned.
#[derive(Clone, Debug)]
pub struct TensorDecl {
    pub desc: TensorDesc,
    pub kind: TensorKind,
    /// Optional debug label (op/tensor name) for profiling + error messages.
    pub label: Option<String>,
}

/// Semantic ops. Each names the handles it reads plus the `dst` it writes. Grow as models need.
///
/// Dimensions that aren't derivable from the operand descs are carried inline (e.g. `n_head`,
/// `head_dim`) so a backend can execute an op without re-deriving layout from shapes.
#[derive(Clone, Debug)]
pub enum Op {
    /// `dst = rmsnorm(x) * weight`, normalizing over the last `dim` of each of `rows` rows.
    /// A weightless RMSNorm (Gemma V-norm) sets `weight` to a ones tensor.
    RmsNorm {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    /// `dst += rmsnorm(x) * weight`: normalize `x` per row, then add to `dst` in-place.
    /// Eliminates the separate RmsNorm + Add dispatch pair (E2B per-layer projection tail).
    RmsNormAdd {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    /// `dst = ((x - mean) / sqrt(var + eps)) * weight + bias`, normalizing over the last `dim` of
    /// each of `rows` rows — the MEAN-CENTRED LayerNorm, llama.cpp's `LLM_NORM` (`ggml_norm` then
    /// `ggml_mul(mw)` then `ggml_add(mb)`, see `llm_graph_context::build_norm`), as opposed to
    /// [`Op::RmsNorm`]'s `LLM_NORM_RMS`. Two details that are silent precision bugs if got wrong,
    /// both read off `ggml_compute_forward_norm_f32`: `var` is the BIASED estimator (`Σ(x-mean)²`
    /// divided by `dim`, not `dim-1`), and `eps` is added to the variance INSIDE the sqrt.
    ///
    /// deepseek32's `indexer_k_norm` is the only mean-centred norm anywhere in the DeepSeek family
    /// (every other one there is RMS); it always carries a bias, so `bias` is not optional.
    LayerNorm {
        x: TensorId,
        weight: TensorId,
        bias: TensorId,
        dst: TensorId,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    /// `dst[m, out_f] = x[m, in_f] · weightᵀ`. `weight` may be any (quantized) dtype; the backend
    /// dispatches the kernel (GEMV/GEMM/MMQ on GPU, dequant+matvec on CPU).
    Linear {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        m: u32,
        in_f: u32,
        out_f: u32,
        /// ELEMENT offset into `weight` where this projection's rows start (0 = whole tensor).
        /// Lets several projections share one concatenated weight upload (fused QKV): prefill
        /// runs ONE wide GEMM over the whole tensor while decode keeps per-projection GEMVs into
        /// its slices. Must be row-aligned (`w_off % in_f == 0`) and block-aligned for quants.
        w_off: u32,
    },
    /// Row-wise softmax: `dst[r, :] = softmax(x[r, :] * scale)` over `dim` columns, `rows` rows.
    /// diffusion-gemma's in-graph self-conditioning (see `docs/diffusion-gemma.md`'s Phase-B and
    /// the reference's `dg_canvas_embed`): softmaxes the previous step's canvas logits over the
    /// FULL vocab before the soft-embedding matmul. `scale` is baked in so a temperature that
    /// changes per call doesn't need a separate `Scale` op ahead of this one — production code
    /// pre-multiplies on the host instead (keeps the compiled plan static across steps) and
    /// passes `scale: 1.0`, but the op itself is general.
    Softmax {
        x: TensorId,
        dst: TensorId,
        rows: u32,
        dim: u32,
        scale: f32,
        /// Perf (DiffusionGemma denoise, Vulkan — see `crates/infr-llama/src/seam/runner.rs`'s
        /// denoise call site): when `Some`, the backend reads ONE f32 from this tensor's bound
        /// buffer at dispatch time and uses THAT as the scale instead of the compile-time `scale`
        /// field above (which is then ignored). Lets a cached/replayed plan vary the softmax
        /// temperature every call — a tiny per-step 4-byte upload — instead of re-baking `scale`
        /// into the plan (which would force a rebuild) or pre-multiplying the whole row host-side.
        /// `None` (every other call site) keeps the exact prior behavior: `scale` is a plain
        /// compile-time constant, unmodified.
        scale_buf: Option<TensorId>,
    },
    /// Per-head RMSNorm of `x` (`rows × n_head × head_dim`) with a per-`head_dim` `weight`
    /// (Qwen3 / Gemma Q-norm and K-norm). In place when `dst == x`.
    QkNorm {
        x: TensorId,
        /// Per-`head_dim` scale, or `None` for a genuinely WEIGHTLESS per-head RMSNorm —
        /// `dst = x / sqrt(mean_head(x²) + eps)` with no multiply at all.
        ///
        /// `None` is DeepSeek V4's Q norm: after `wq_b` it reshapes to `[head_dim, n_head,
        /// n_tokens]` and calls bare `ggml_rms_norm` (`deepseek4.cpp`'s `build_attention`, the
        /// `q_norm` callback) — there is no `attn_q_norm` tensor in the file to bind.
        ///
        /// A ones-vector weight would compute the same numbers ([`Op::RmsNorm`]'s doc records that
        /// convention, and gemma4's V-norm/llama4's L2-norm use it), but it is a fake operand: it
        /// costs a real `head_dim`-float allocation and upload per graph, and it makes a reader
        /// look for the weight in the GGUF. `Option` says what the reference says.
        weight: Option<TensorId>,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
        /// Per-row stride in `x`. 0 = packed (stride = n_head * head_dim). Non-zero when
        /// reading from an interleaved buffer (e.g. qwen35's q+g layout) to skip a
        /// per-head CopyStrided dispatch.
        x_stride: u32,
    },
    /// Fused per-head RMSNorm + SiLU gate multiply: `QkNorm` immediately followed by an
    /// `Op::GatedAct` (`Activation::Silu`) consuming QkNorm's own output (qwen35's DeltaNet
    /// silu-gated RMSNorm — see docs/qwen35.md). One pass: for each of `rows * n_head` heads,
    /// `dst[i] = (x[i] * rms_scale * weight[i]) * silu(gate[i])` where `rms_scale =
    /// 1/sqrt(mean_head(x^2) + eps)` and `i` ranges over the head's `head_dim` elements. `gate` is
    /// a same-shape `[rows, n_head*head_dim]` buffer, indexed by the SAME flat element position as
    /// `x` (not a separate per-head layout). In place when `dst == x`.
    ///
    /// Exists because `GatedAct` reading `QkNorm`'s freshly-written output is a real
    /// read-after-write hazard (a pipeline barrier on GPU backends) — fusing the two into one
    /// dispatch removes it. The rmsnorm reduction is bit-identical to standalone `QkNorm`; the
    /// gate multiply is pure elementwise (no reassociation of the reduction). Backends without a
    /// fused kernel advertise `Capabilities::gated_rmsnorm == false`; the runner keeps emitting
    /// the split `QkNorm` → `GatedAct` pair for them.
    GatedRmsNorm {
        x: TensorId,
        weight: TensorId,
        gate: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    },
    /// RoPE over the first `rope_dim` of each head; dims past `rope_dim` pass through unrotated.
    /// `positions` is an i32 tensor of length `rows`. `freq_factors`, if present, divides per-pair
    /// angles (Gemma proportional RoPE, DeepSeek's YaRN ramp).
    Rope {
        x: TensorId,
        positions: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        freq_factors: Option<TensorId>,
        /// Per-row stride in `x`. 0 = packed (stride = n_head * head_dim).
        x_stride: u32,
        /// Which two elements a rotation pair is made of — llama.cpp's rope TYPE, and the one thing
        /// about RoPE that produces fluent nonsense rather than an error when it is wrong (the
        /// angles, the widths and the pass-through tail are all identical either way).
        ///
        /// * `false` — **NORM** (`LLAMA_ROPE_TYPE_NORM`): interleaved consecutive pairs
        ///   `(2p, 2p+1)` share the angle for pair `p`. The llama family, and DeepSeek's main
        ///   q_pe/k_pe rope.
        /// * `true` — **NEOX** (`LLAMA_ROPE_TYPE_NEOX`): split-half pairs `(p, p + rope_dim/2)`.
        ///   The same pairing [`Op::QkNormRope`] applies unconditionally (qwen/gemma), and the
        ///   pairing deepseek32's lightning indexer hardcodes for `indexer_q`/`indexer_k` while the
        ///   MLA rope beside it stays NORM (see `docs/deepseek.md` § Stage 3).
        ///
        /// `Config::permute_qk_neox` is a DIFFERENT mechanism for a different problem: it rewrites
        /// a whole model's `attn_q`/`attn_k` ROWS at load so that a NORM rope reproduces NEOX for a
        /// GGUF that stayed in HF rotate-half order. It is model-wide and keyed on tensor names, so
        /// it cannot express "this one projection ropes NEOX while the model's main rope is NORM".
        neox: bool,
        /// Rotate BACKWARDS by the position — llama.cpp's `ggml_rope_ext_back`
        /// (`GGML_OP_ROPE_BACK`), which DeepSeek V4 applies to the rope slice of the ATTENTION
        /// OUTPUT before the grouped output projection (`deepseek4.cpp`'s `attn_derope`). Nothing
        /// else in the family de-ropes.
        ///
        /// It is the SAME kernel with one sign flipped. `ggml_compute_forward_rope_back` calls
        /// `ggml_compute_forward_rope_flt(..., forward = false)`, whose only use of that flag is
        /// `sin_sign = forward ? 1 : -1`, applied as `cache[i0 + 1] *= sin_sign` in
        /// `ggml_rope_cache_init` — `cos` untouched, `sin` negated, i.e. the TRANSPOSE of the 2×2
        /// rotation. Same angles, same `freq_factors` division, same pairing, same pass-through
        /// tail: `dst[i0] = a·cos + b·sin`, `dst[i1] = −a·sin + b·cos`.
        ///
        /// The transpose is the INVERSE only when the rotation is orthonormal, and ggml's is not in
        /// general: `rope_yarn` multiplies BOTH `cos` and `sin` by `mscale` (the `attn_factor`, with
        /// YaRN's `1 + 0.1·ln(1/freq_scale)` correction folded in when `ext_factor != 0`), and
        /// `sin_sign` does not invert that — so ggml's forward-then-back scales by `mscale²`, not 1.
        /// V4 cancels it exactly: `dsv4_rope_attn_factor` returns `1/(1 + 0.1·ln(1/freq_scale))`
        /// when `ext_factor != 0` and `1.0` otherwise, making the effective `mscale` 1 at every V4
        /// rope call site. [`Op::Rope`] carries no magnitude scale at all (YaRN reaches it only as
        /// `freq_factors`, a per-pair ANGLE divisor), so `backward` here is an exact inverse of the
        /// forward rope with the same fields — the property `rope_back_inverts_rope_forward`
        /// asserts.
        backward: bool,
    },
    /// Fused per-head RMSNorm + NEOX RoPE — `QkNorm` immediately followed by `Rope` on the same
    /// tensor (the common qwen3/gemma q/k case). One pass: each head is rmsnormed (`× weight`) then
    /// its first `rope_dim` rotated, dims beyond `rope_dim` passing through normed. Maps 1:1 to the
    /// GPU's fused `qk_norm_rope` kernel; the CPU runs it as a single loop. Use the standalone
    /// `QkNorm` (gemma4 weightless V-norm, no RoPE) or `Rope` (llama, no q/k-norm) when not both.
    QkNormRope {
        x: TensorId,
        weight: TensorId,
        positions: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        eps: f32,
        freq_factors: Option<TensorId>,
        /// Per-row stride in `x`. 0 = packed (stride = n_head * head_dim).
        x_stride: u32,
    },
    /// Append `src` (`rows × row_stride`) into the persistent KV `cache` starting at row `pos`,
    /// casting to the cache dtype (typically f16). Stateful write — order matters.
    WriteKv {
        src: TensorId,
        cache: TensorId,
        rows: u32,
        row_stride: u32,
        pos: u32,
    },
    /// DeepSeek V4 compressor state update and, on a block boundary, per-channel softmax pooling.
    /// `state` is an opaque byte-addressed cache buffer; the two offsets name its f32 value and
    /// score rings. The projected rows are f32 and have width `dim` (HCA) or `2*dim` (overlapping
    /// CSA/LID). The op always persists the current row and writes `dst[dim]` only when
    /// `(pos + 1) % ratio == 0`.
    Dsv4Compress {
        values: TensorId,
        scores: TensorId,
        ape: TensorId,
        norm: TensorId,
        freq_factors: Option<TensorId>,
        state: TensorId,
        dst: TensorId,
        state_values_off: u32,
        state_scores_off: u32,
        pos: u32,
        ratio: u32,
        dim: u32,
        rope_dim: u32,
        theta: f32,
        eps: f32,
        overlap: bool,
    },
    /// Quantize f32 rows into one of DeepSeek V4's two cache formats. `Fp8Kv` is the official
    /// 512-wide mixed row (448 E4M3 + 64 BF16 + eight UE8M0 scale bytes); `Mxfp4` is the
    /// indexer's 128-wide E2M1 cache (64 packed bytes + four UE8M0 scales).
    Dsv4CacheWrite {
        src: TensorId,
        cache: TensorId,
        rows: u32,
        row: u32,
        cache_off: u32,
        format: Dsv4CacheFormat,
    },
    /// DeepSeek V4's ratio-4 block indexer over MXFP4 query/key rows. `q` holds `n_head` rows and
    /// `k_cache` holds `kv_len` rows at `cache_off`; both use the 68-byte official indexer row.
    Dsv4Indexer {
        q: TensorId,
        k_cache: TensorId,
        weights: TensorId,
        dst: TensorId,
        cache_off: u32,
        kv_len: u32,
        n_head: u32,
        head_dim: u32,
        top_k: u32,
        scale: f32,
    },
    /// Gather the live V4 raw-SWA rows plus compressed rows into one packed f16 K=V scratch.
    /// `indices` selects compressed rows for CSA; `None` gathers the complete visible prefix.
    /// The following ordinary `Op::Attention` owns the shared softmax and attention sink.
    Dsv4Gather {
        raw_cache: TensorId,
        comp_cache: TensorId,
        indices: Option<TensorId>,
        dst: TensorId,
        comp_off: u32,
        pos: u32,
        visible: u32,
        selected: u32,
        head_dim: u32,
        raw_window: u32,
    },
    /// Qwen3.8 QSA block selector. `k_cache` stores one raw (unnormalised, unroped) F16 index-key
    /// row per token. Complete `ratio`-token blocks are mean-pooled, RMS-normalised and roped at
    /// their first token position before scoring against the already normalised/roped query heads.
    /// `dst` contains the selected block indices in chronological order.
    QsaIndexer {
        q: TensorId,
        k_cache: TensorId,
        k_norm: TensorId,
        dst: TensorId,
        /// Number of consecutive query rows. Row `r` sees
        /// `kv_len - rows + r + 1` cached tokens.
        rows: u32,
        kv_len: u32,
        n_head: u32,
        head_dim: u32,
        top_blocks: u32,
        ratio: u32,
        rope_dim: u32,
        theta: f32,
        eps: f32,
        scale: f32,
    },
    /// Gather QSA-selected complete blocks plus the always-visible incomplete tail from the
    /// ordinary full-attention K/V caches into packed F16 scratch consumed by `Op::Attention`.
    QsaGather {
        k_cache: TensorId,
        v_cache: TensorId,
        indices: TensorId,
        k_dst: TensorId,
        v_dst: TensorId,
        selected_blocks: u32,
        complete_blocks: u32,
        tail: u32,
        ratio: u32,
        row_elems: u32,
    },
    /// Batched Qwen3.8 QSA attention over row-specific selected blocks. Unlike
    /// [`Op::QsaGather`], which materializes one compact K/V prefix for one query, this op reads
    /// each row's chronological block-index slice directly and applies attention to that row's
    /// selected complete blocks plus its own incomplete causal tail. `kv_len` is the visible
    /// length of the final row; row `r` sees `kv_len - rows + r + 1` tokens.
    QsaBatchAttention {
        q: TensorId,
        k_cache: TensorId,
        v_cache: TensorId,
        indices: TensorId,
        dst: TensorId,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
        top_blocks: u32,
        ratio: u32,
        scale: f32,
    },
    /// Scaled-dot-product attention. `q` is `rows × n_head × head_dim`; `k_cache`/`v_cache` hold
    /// `kv_len` rows of `n_kv × head_dim`. GQA when `n_head > n_kv`. `dst` is `rows × n_head ×
    /// head_dim`. `pos` is the absolute position of the first query row (for masking).
    Attention {
        q: TensorId,
        k_cache: TensorId,
        v_cache: TensorId,
        dst: TensorId,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
        scale: f32,
        mask: AttnMask,
        pos: u32,
        /// Optional per-head ATTENTION SINK logits `[n_head]`, f32 — one extra logit per head that
        /// joins the softmax denominator and contributes NO value. DeepSeek V4's `attn_sinks`
        /// (`deepseek4.cpp` passes `layer.attn_sinks` into `build_attn` at all three of its
        /// attention call sites); nothing else in this codebase has sinks.
        ///
        /// The arithmetic is `ggml_compute_forward_soft_max_f32`'s `src2` handling, verbatim — it is
        /// two lines, and both matter:
        ///
        /// ```text
        /// m = max_j(score[j])                 // score = q·K[j]*scale (+ mask), masked keys only
        /// m = max(m, sink[h])                 // the sink JOINS THE MAX
        /// l = Σ_j exp(score[j] - m)
        /// l = l + exp(sink[h] - m)            // and the DENOMINATOR
        /// dst = Σ_j (exp(score[j] - m) / l) * V[j]
        /// ```
        ///
        /// Two ways to get it wrong that still produce plausible numbers. Leaving the sink out of the
        /// MAX is a pure precision bug — it is algebraically identical while `sink[h] <= max_j
        /// score[j]`, and only overflows `exp` once a sink dominates, which is exactly the regime a
        /// sink is FOR. Including the sink in the NUMERATOR (giving it a value row) still yields a
        /// row summing to 1 and reads like a softmax; it is a different function, and it is what
        /// `attention_sinks_are_denominator_only` is written to catch.
        ///
        /// `sink[h]` is the RAW weight value: not multiplied by `scale` (which has already been
        /// applied to the scores), not touched by the ALiBi slope, not masked. A head whose sink is
        /// far below its running max costs essentially nothing (`exp(sink - m) → 0`); a head whose
        /// sink dominates suppresses every real key's weight toward 0, which is the "attend to
        /// nothing" escape valve.
        ///
        /// `None` for every arch in this codebase today, and on that path each backend runs the code
        /// it ran before this field existed — no current model's numerics can move.
        sinks: Option<TensorId>,
    },
    /// Multi-head Latent Attention (DeepSeek V2/V3). Absorbed form: one compressed K row per token
    /// (V is an aliased prefix view), `wk_b` absorbs Q nope into the latent space before the dot
    /// product, and `wv_b` maps the attention output back to per-head space. See `docs/deepseek.md`.
    Mla {
        /// Query: `rows × n_head × q_head_dim` (`[nope|rope]` per head, q_pe already rope'd).
        q: TensorId,
        /// KV cache: `kv_len × key_length` (key_length = kv_lora_rank + qk_rope_dim, 576 for V3).
        /// One row per token. V is the first `kv_lora_rank` columns — aliased, no separate cache.
        k_cache: TensorId,
        /// Absorption weight: `[n_head, kv_lora_rank, qk_nope_dim]`. Per head `wk_b[h]ᵀ` maps
        /// each head's q_nope (128) into the latent space (512) before dotting with K.
        wk_b: TensorId,
        /// Output weight: `[n_head, kv_lora_rank, v_head_dim]`. Applied AFTER the KQV product per
        /// head to map the attention output (512-wide) back to per-head output dim (128).
        wv_b: TensorId,
        dst: TensorId,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        q_head_dim: u32,
        kv_lora_rank: u32,
        qk_nope_dim: u32,
        qk_rope_dim: u32,
        v_head_dim: u32,
        scale: f32,
        mask: AttnMask,
        pos: u32,
        /// RoPE base frequency for the internal q_pe rope (DeepSeek V2/V3 use 10000.0).
        theta: f32,
        /// Optional per-pair frequency divisors for the internal q_pe rope (YaRN long-context
        /// ramp, `qk_rope_dim/2` floats) — same semantic as `Op::Rope`'s `freq_factors`: the
        /// rotation angle is DIVIDED by `ff[p]` for pair p. `None` = plain RoPE.
        freq_factors: Option<TensorId>,
        /// Optional additive per-(query row, key) score bias `[rows, kv_len]`, f32, added to the
        /// scaled score `q·K[j] * scale` BEFORE the softmax max — i.e. `Σ` in llama.cpp's
        /// `ggml_soft_max_ext(kq, kq_mask, ...)` terms, the mask summand.
        ///
        /// deepseek32's only caller: the lightning indexer's top-k selection, expanded by
        /// [`Op::TopkMask`] into `0` on the selected keys and `-inf` everywhere else. llama.cpp
        /// does exactly this — it adds the top-k mask to the ordinary causal mask and runs DENSE
        /// attention over the full `n_kv`, realising none of the FLOP saving (see `docs/deepseek.md`
        /// § "How top-k feeds attention"). Indexing is by KEY POSITION `j`, not by the cache row
        /// `j % cap_rows`, so a ring cache does not shift it.
        ///
        /// `None` for deepseek2 (V2/V2-Lite/V3/V3.1), which attends every causally-eligible key —
        /// and on that path every backend takes the same code it did before this field existed, so
        /// V2-Lite's numerics cannot move.
        key_bias: Option<TensorId>,
    },
    /// Expand `Op::LightningIndexer`'s `[rows, top_k]` i32 key indices into the additive
    /// `[rows, kv_len]` f32 score mask `Op::Mla::key_bias` consumes: `0.0` at every selected key,
    /// `f32::NEG_INFINITY` everywhere else.
    ///
    /// This is the whole of "how top-k feeds attention" (`docs/deepseek.md`): llama.cpp materialises
    /// the same `[n_kv, n_tokens]` mask and adds it to the causal one, so the numerics are faithful
    /// and the FLOP saving is not taken. Keeping the expansion in its own op (rather than having the
    /// indexer emit a mask directly) is what leaves a later gather open as a pure optimisation.
    ///
    /// Cost, stated because it is the reason a gather exists at all: `rows * kv_len` f32 of scratch
    /// and one extra f32 load per key in the MLA inner loop. The alternative — membership-testing
    /// the index list inside the MLA kernel — needs no scratch but scans `top_k` (≈2048 on V3.2)
    /// per (row, head, key), which is worse by three orders of magnitude on the inner loop.
    ///
    /// `-inf` (not a large finite negative) is safe because at least one key is always selected:
    /// `top_k >= 1` and the indexer's own order ranks every causally-eligible key above every
    /// ineligible one, so each row's softmax max is finite and `exp(-inf - max) == 0` exactly.
    /// Indices are read as u32 bit patterns out of the i32 `idx` tensor — [`Op::Argmax`]'s carrier
    /// convention, which is what `Op::LightningIndexer` writes.
    TopkMask {
        /// Selected key indices `[rows, top_k]`, i32 — `Op::LightningIndexer`'s `dst`.
        idx: TensorId,
        /// Additive mask `[rows, kv_len]`, f32.
        dst: TensorId,
        rows: u32,
        kv_len: u32,
        top_k: u32,
    },
    /// DeepSeek V4's **hyper-connection mixing coefficients** — the whole of `deepseek4.cpp`'s
    /// `build_hc_pre` except its final collapse, i.e. llama.cpp's `ggml_dsv4_hc_comb` plus the
    /// `pre`/`post` gate arithmetic that reference leaves as elementwise views.
    ///
    /// V4 widens the residual stream to `hc` parallel copies and wraps every sublayer
    /// `pre → sublayer → post` (`docs/deepseek.md` § "Sinkhorn hyper-connections"). ONE matmul over
    /// the RMS-normed flattened streams produces `mixes` `[rows, (2 + hc)*hc]`; this op turns that
    /// into the three coefficient sets the wrap needs. Per token `t`:
    ///
    /// ```text
    /// pre[t, h]           = sigmoid(mixes[t, h]                * scale[0] + base[h])              + eps
    /// post[t, h]          = sigmoid(mixes[t, hc + h]           * scale[1] + base[hc + h])         * 2
    /// logits[t, dst, src] =         mixes[t, 2hc + dst + hc*src]* scale[2] + base[2hc + dst + hc*src]
    /// comb[t]             = sinkhorn(logits[t])
    /// ```
    ///
    /// with `sinkhorn` making the `hc × hc` matrix approximately doubly stochastic, so no stream's
    /// mass blows up or vanishes with depth:
    ///
    /// ```text
    /// m = softmax over dst        (independently for each src column)
    /// m = m + eps
    /// norm_src(m)                                     // 1 column normalisation, THEN
    /// for _ in 1..n_iter { norm_dst(m); norm_src(m) } // n_iter-1 (row, column) rounds
    ///
    /// norm_src: for each dst, m[dst, :] /= (Σ_src m[dst, src] + eps)
    /// norm_dst: for each src, m[:, src] /= (Σ_dst m[dst, src] + eps)
    /// ```
    ///
    /// Five things that still RUN when got wrong, all read off `build_hc_sinkhorn` and
    /// `ggml_dsv4_hc_comb`'s header comment rather than off prose:
    ///
    /// * **The `comb` chunk starts at `2*hc` and its flat index is `dst + hc*src`.** `dst` is the
    ///   FAST axis. `src + hc*dst` transposes the mixing matrix, and since Sinkhorn's fixed point is
    ///   near-symmetric the output stays plausible — only a value check catches it.
    /// * **The iteration is ASYMMETRIC**: `n_iter` normalisations over `src`, `n_iter - 1` over
    ///   `dst`. The last one is over `src`, which is why `Σ_src` comes out closer to 1 than `Σ_dst`.
    /// * **`eps` is added in THREE places inside Sinkhorn** — once to every element after the
    ///   softmax, and once to each of the two sums before they divide — plus a FOURTH on `pre`.
    ///   None of them is a guard against a zero that cannot happen: the post-softmax `+eps` shifts
    ///   the fixed point the iteration converges to, so dropping it moves the answer.
    /// * **`pre` is `sigmoid` then `+ eps`; `post` is `sigmoid` then `× 2`.** Different tails on
    ///   adjacent, equal-width chunks of the same tensor.
    /// * **llama.cpp's lambda names are inverted relative to its own `dst`/`src` vocabulary.** Its
    ///   `norm_cols` sums over `src` (`ggml_sum_rows` of the PERMUTED matrix) and its `norm_rows`
    ///   sums over `dst` (`ggml_sum_rows` of the matrix as laid out, whose fast axis is `dst`).
    ///   `norm_src`/`norm_dst` above are named for the axis each one actually reduces.
    ///
    /// `gates` is `None` for the **model head** (`build_hc_head`), which collapses the streams one
    /// last time before `output_norm` and has no sublayer to wrap. That is not a separate op: the
    /// head's `output_hc_fn` is `{hc_dim, hc}` instead of `{hc_dim, (2+hc)*hc}`, so its `mixes` is
    /// exactly the `pre` chunk, and its `output_hc_scale` `{1}` / `output_hc_base` `{hc}` are read
    /// at the SAME indices (`scale[0]`, `base[0..hc]`) the wrapping form reads them at. Same
    /// arithmetic, narrower inputs — a second op would have been a copy of the `pre` line.
    ///
    /// `hc` is 4 in every shipped V4 config (llama.cpp `GGML_ASSERT`s it); every backend here
    /// accepts `1..=`[`HYPER_CONNECT_MAX_MULT`] and refuses anything wider rather than read past a
    /// fixed-size per-token scratch array. `n_iter >= 1` is required — `n_iter = 0` would still run
    /// one `norm_src` in the reference (its loop starts at 1), which is a shape no config asks for
    /// and not worth reproducing.
    HyperConnectMix {
        /// `[rows, (2 + hc)*hc]` f32 — the mixing matmul's output. `[rows, hc]` when `gates` is
        /// `None` (the head form).
        mixes: TensorId,
        /// `[3]` f32 — the per-chunk affine slopes. `[1]` when `gates` is `None`.
        scale: TensorId,
        /// `[(2 + hc)*hc]` f32 — the per-element affine offsets, sliced at `0`, `hc` and `2*hc`.
        /// `[hc]` when `gates` is `None`.
        base: TensorId,
        /// Stream-collapse weights `[rows, hc]` f32 — [`Op::HyperConnectPre`]'s `weights`.
        pre: TensorId,
        /// The wrapping form's two extra outputs; `None` for the model head (see above).
        gates: Option<HyperGates>,
        /// Token count.
        rows: u32,
        /// Number of parallel residual streams (`hc_mult`).
        hc: u32,
        eps: f32,
        /// Sinkhorn iteration count. Ignored when `gates` is `None` (nothing to normalise).
        n_iter: u32,
    },
    /// DeepSeek V4 hyper-connection **stream collapse** — llama.cpp's `ggml_dsv4_hc_pre`, the
    /// three-argument `build_hc_pre` overload. Weighted sum of the `hc` parallel residual streams
    /// down to the single vector a sublayer (or the model head) consumes:
    ///
    /// ```text
    /// dst[t, i] = Σ_h x[t, h, i] * weights[t, h]
    /// ```
    ///
    /// `weights` is [`Op::HyperConnectMix`]'s `pre` — strictly positive (a sigmoid plus `eps`), so
    /// this is a convex-ish blend rather than a signed combination, but nothing here depends on
    /// that.
    HyperConnectPre {
        /// The widened residual `[rows, hc, n_embd]` f32.
        x: TensorId,
        /// Per-stream collapse weights `[rows, hc]` f32.
        weights: TensorId,
        /// `[rows, n_embd]` f32.
        dst: TensorId,
        rows: u32,
        hc: u32,
        n_embd: u32,
    },
    /// DeepSeek V4 hyper-connection **stream re-expansion** — llama.cpp's `ggml_dsv4_hc_post`, the
    /// non-fused `build_hc_post`. Writes the sublayer's single output back across all `hc` streams
    /// while mixing the streams among themselves, replacing `x = x + f(x)`:
    ///
    /// ```text
    /// dst[t, dst_h, i] = x[t, i] * post[t, dst_h] + Σ_src residual[t, src, i] * comb[t, dst_h, src]
    /// ```
    ///
    /// `residual` is the widened stream as it entered the wrap (NOT the collapsed vector), and
    /// `comb`'s per-token flat index is `dst_h + hc*src` — see [`Op::HyperConnectMix`] on why that
    /// transpose is invisible to anything but a value check.
    HyperConnectPost {
        /// The sublayer's output `[rows, n_embd]` f32.
        x: TensorId,
        /// The widened residual as it entered the wrap `[rows, hc, n_embd]` f32.
        residual: TensorId,
        /// Per-stream output gates `[rows, hc]` f32.
        post: TensorId,
        /// Sinkhorn-normalised mixing matrix `[rows, hc, hc]` f32, `dst_h + hc*src` per token.
        comb: TensorId,
        /// The new widened residual `[rows, hc, n_embd]` f32.
        dst: TensorId,
        rows: u32,
        hc: u32,
        n_embd: u32,
    },
    /// DeepSeek V3.2's **lightning indexer** (`deepseek32.cpp`'s `// lightning indexer` block, the
    /// non-fused branch): a cheap per-(query row, key) relevance score whose top-`top_k` keys are
    /// the only ones that layer's real MLA attention may then see. Per query row `t` at absolute
    /// position `abs = pos + t`:
    ///
    /// ```text
    /// score[t, j] = Σ_h  (w[t, h] * scale) * ReLU( q[t, h] · k[j] )     for j <= abs
    /// dst[t, :]   = the top_k key positions j by (score DESC, j ASC)
    /// ```
    ///
    /// Four things the reference pins down that a plausible-looking variant gets silently wrong:
    ///
    /// * **One key head shared by every indexer query head** (MQA): `k_cache` holds ONE
    ///   `head_dim`-wide row per token, dotted against all `n_head` query heads. (llama.cpp's
    ///   `indexer_k` is `{head_size, 1, n_tokens}` against `indexer_q`'s `{head_size,
    ///   n_indexer_head, n_tokens}`.)
    /// * **The ReLU is INSIDE the head-weighted sum**, applied to each head's raw dot before the
    ///   `w` multiply — `ggml_relu` then `ggml_mul` then `ggml_sum_rows`. A ReLU on the summed
    ///   score instead would only clamp negatives and would leave the ordering of the positive
    ///   scores untouched, so it passes any test whose winners all score positive.
    /// * **`scale` multiplies the per-head WEIGHT, never the score.** It is the `1/sqrt(head_dim *
    ///   n_head)` normaliser, which llama.cpp folds into `indexer_weights` with a `ggml_scale`
    ///   ("pre-scale weights to avoid scaling operations on huge indexer_score tensor") rather than
    ///   scaling the `[n_kv, rows]` score tensor. Same value, different rounding, so it is carried
    ///   here as an op field applied to `w[t, h]` instead of being left to the caller. Note this is
    ///   the one part of the arithmetic **no test can guard through this op's output**: a positive
    ///   uniform factor is order-preserving, so dropping it or moving it onto the score selects the
    ///   same keys (`lightning_indexer_scale_cannot_change_the_selection` asserts exactly that
    ///   invariance). It is kept faithful so the intermediate SCORES stay comparable with the
    ///   reference during bring-up.
    /// * **The head sum runs `h` ASCENDING** (`ggml_sum_rows` over the contiguous head axis), which
    ///   is the accumulation order every backend here reproduces.
    ///
    /// What the CALLER owes (deepseek32's traps, see `docs/deepseek.md` § "The lightning indexer"):
    /// `q` arrives ALREADY ASSEMBLED and ALREADY ROPE'D — this op does no rope of its own (unlike
    /// [`Op::Mla`], which ropes its q_pe internally). The indexer head is laid out `[rope | nope]`,
    /// the OPPOSITE of the MLA head, and its rope is NEOX where the main rope is NORM; both are the
    /// graph builder's problem, and by the time the head reaches here it is just `head_dim` floats.
    /// The **Hadamard rotation** llama.cpp applies to `q` and `k` is deliberately NOT ported: it is
    /// one orthogonal transform applied identically to both sides, so it preserves every dot
    /// product and exists only for quantisation friendliness — nothing to reproduce in an
    /// unquantised port, and the caller must not apply it to one side alone.
    ///
    /// `dst` holds `rows * top_k` **i32 key indices** (the shape `ggml_top_k` produces), written as
    /// raw i32 words on a device and as u32 bit patterns in the host interpreter's f32 slots —
    /// the same carrier convention as [`Op::Argmax`]'s token id. Each index is a key position in
    /// `[0, kv_len)`, which is also its cache ROW: unlike [`Op::Mla`], this op does NOT fold `j %
    /// cap_rows`, and every backend refuses a cache with fewer than `kv_len` rows. A wrapped ring
    /// cannot arise here and could not be served if it did — masking is causal only, so position 0
    /// is eligible for every query row, and a ring that had wrapped would have overwritten row 0
    /// with a later position before this op ever ran. (V3.2's indexer has no sliding window; the
    /// caches that wrap in this codebase are the SWA ones.) Emitting indices rather than a `-inf`
    /// mask keeps both the mask expansion and a later gather open to the consumer (see
    /// `docs/deepseek.md` § "How top-k feeds attention"): llama.cpp itself expands them back into a
    /// mask and runs dense attention, realising none of the FLOP saving.
    ///
    /// **Ordering is total and identical on every backend**: by score descending, ties broken
    /// toward the LOWER key index. Keys past the causal bound (`j > abs`) rank below every eligible
    /// key and among themselves by ascending index, so when fewer than `top_k` keys are eligible
    /// the tail fills with the lowest ineligible positions. That short case is llama.cpp's too —
    /// it clamps only against `n_kv` (`n_top_k = min(indexer_score->ne[0], n_indexer_top_k)`) and
    /// never against the causal count, so `ggml_top_k`'s `std::partial_sort` returns masked
    /// `-INFINITY` entries in the tail; their order there is genuinely unspecified (it is not a
    /// stable sort, and the kernel then swaps `dst[0]`/`dst[1]` "to emphasize that the order is not
    /// important"), which is harmless only because the consumer re-applies the causal mask. This op
    /// picks the deterministic member of that family rather than inheriting the ambiguity.
    ///
    /// `top_k <= kv_len` is a precondition (the caller clamps, as llama.cpp's `min` does); each
    /// backend refuses a larger `top_k` rather than naming keys that do not exist. Masking is
    /// causal only — V3.2's indexer has no sliding window, so there is no [`AttnMask`] here.
    LightningIndexer {
        /// Indexer queries `[rows, n_head, head_dim]`, f32, rope already applied.
        q: TensorId,
        /// Indexer key cache `[cap_rows, head_dim]` — ONE row per token (MQA). `cap_rows = numel /
        /// head_dim` must be at least `kv_len`; key `j` is row `j` (see above on why there is no
        /// ring fold here).
        k_cache: TensorId,
        /// Per-head indexer weights `w` `[rows, n_head]`, f32 — `indexer_proj · x`, UNSCALED (this
        /// op applies `scale`).
        weights: TensorId,
        /// Selected key indices `[rows, top_k]`, i32.
        dst: TensorId,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        head_dim: u32,
        top_k: u32,
        /// `1/sqrt(head_dim * n_head)` — multiplies `w[t, h]`, NOT the score (see above).
        scale: f32,
        /// Absolute position of the first query row (the causal bound is `pos + t`).
        pos: u32,
    },
    /// Gated FFN activation: `dst[r,i] = act(gate[r,i]) * up[r, i + up_off]` (`rows × nff`). `gate`
    /// and `up` are separate handles (a backend may fuse them into one buffer internally). `up_off`
    /// shifts the `up` read by a whole-element offset so a layer-major slice of a bigger buffer can
    /// be consumed in place (Gemma E2B per-layer embedding); 0 for the normal case. `up_stride`
    /// is the per-row stride of the `up` buffer when it's embedded in a wider row-major tensor;
    /// 0 means the rows are tightly packed (stride = nff).
    /// `gate_stride` is the same for the `gate` buffer — used when gate data is strided in a wider
    /// interleaved buffer (e.g. qwen35's q+g layout where query and gate share rows).
    /// `gate_block_width` (>0) means the gate is interleaved in blocks of this width, with each
    /// block containing query+gate pairs (qwen35: block_width = 2*hd, gate at offset hd within block).
    GatedAct {
        gate: TensorId,
        up: TensorId,
        dst: TensorId,
        rows: u32,
        nff: u32,
        act: Activation,
        up_off: u32,
        up_stride: u32,
        gate_stride: u32,
        gate_block_width: u32,
        /// DeepSeek V4's per-layer SwiGLU clamp — `swiglu_clamp_shexp[il]` on this (dense /
        /// shared-expert) path, resolved through [`swiglu_clamp`] so `None` is the disabled state.
        ///
        /// `Some(limit)` replaces `act(gate) * up` with
        /// **`act(min(gate, limit)) * clamp(up, -limit, limit)`**: `up` is clamped SYMMETRICALLY,
        /// the gate ONE-SIDED (upper bound only, `ggml_clamp(gate, -INFINITY, limit)`), and the
        /// gate clamp happens **BEFORE** the activation. Every other arch llama.cpp clamps
        /// (gpt-oss) clamps the gate AFTER its SiLU; V4 is the exception, and `deepseek4.cpp` is
        /// the only reason this field exists, so the post-activation order is not representable
        /// here — the moment an arch needing it lands, that is when the mode flag earns its place.
        /// Read off `llm_graph_context::build_ffn`'s `LLM_FFN_SILU` arm
        /// (`arch == LLM_ARCH_DEEPSEEK4` branch) in llama.cpp's `src/llama-graph.cpp`.
        ///
        /// The orders are NOT interchangeable: SiLU is non-monotone below zero and is not the
        /// identity, so a gate above the limit and a gate below zero both move. Only
        /// [`Activation::Silu`] and [`Activation::Gelu`] carry it (the two the reference's gated
        /// arms use); a clamped [`Activation::Sigmoid`] has no caller and every backend refuses it.
        /// `None` for every existing arch, so no current model's numerics move.
        swiglu_clamp: Option<f32>,
    },
    /// Gated FFN activation over a COMBINED `gu` buffer `[rows, 2*nff]` (gate half first, up half
    /// second per row): `dst[r,i] = act(gu[r,i]) * gu[r, nff+i]`. Produced when the runner
    /// concatenates the gate+up weights into one `[2*nff, ne]` tensor so the FFN input projection
    /// is a single GEMV/GEMM instead of two (see `Capabilities::combined_gu`).
    GatedActFused {
        gu: TensorId,
        dst: TensorId,
        rows: u32,
        nff: u32,
        act: Activation,
        /// Per-layer SwiGLU clamp, same arithmetic and same constraints as
        /// [`Op::GatedAct::swiglu_clamp`] — see there.
        swiglu_clamp: Option<f32>,
    },
    /// `dst[i] = a[i] + b[i]` (residual add). In place when `dst == a`.
    Add {
        a: TensorId,
        b: TensorId,
        dst: TensorId,
        n: u32,
    },
    /// Broadcast bias add: `dst[r*n + c] = x[r*n + c] + bias[c]` for `r` in `0..rows`, `c` in
    /// `0..n`. `bias` is a length-`n` vector added to every one of the `rows` rows (a projection's
    /// `Wx + b`). Qwen2/2.5 bias their q/k/v projections; the seam has no bias otherwise. In place
    /// when `dst == x`.
    AddBias {
        x: TensorId,
        bias: TensorId,
        dst: TensorId,
        rows: u32,
        n: u32,
    },
    /// `dst[i] = x[i] * s` (Gemma per-layer output scale, embedding scale).
    Scale {
        x: TensorId,
        dst: TensorId,
        s: f32,
        n: u32,
    },
    /// Elementwise `dst[i] = silu(x[i] * scale)`. Qwen3.8's low-rank gated-residual bottleneck
    /// divides by the number of residual streams before SiLU.
    Silu {
        x: TensorId,
        dst: TensorId,
        n: u32,
        scale: f32,
    },
    /// Qwen3.8 stream collapse after its low-rank gate projection. `x` and `gate` are
    /// `[rows, hc, n_embd]`; gate contains raw logits. The output is
    /// `mean_h(x[r,h,d] * sigmoid(gate[r,h,d]))`.
    QwenHcMix {
        x: TensorId,
        gate: TensorId,
        dst: TensorId,
        rows: u32,
        hc: u32,
        n_embd: u32,
    },
    /// Qwen3.8 residual injection. `residual`/`dst` are `[rows,hc,n_embd]`, `block` is
    /// `[rows,n_embd]`, and `gate` is `[rows,hc]` raw logits. Per element:
    /// `dst = residual + block * 2*sigmoid(gate/hc)`.
    QwenHcInject {
        residual: TensorId,
        block: TensorId,
        gate: TensorId,
        dst: TensorId,
        rows: u32,
        hc: u32,
        n_embd: u32,
    },
    /// Qwen3.8 PLE query-dependent value gate. `key` and `query` are grouped-normalized
    /// `[rows,hc,n_embd]`; `value` is `[rows,n_embd]`. For each stream, reduce
    /// `dot(key,query)/sqrt(n_embd)`, apply signed sqrt then sigmoid, and broadcast-multiply the
    /// value into `dst[rows,hc,n_embd]`.
    QwenPleGate {
        key: TensorId,
        query: TensorId,
        value: TensorId,
        dst: TensorId,
        rows: u32,
        hc: u32,
        n_embd: u32,
    },
    /// Broadcast elementwise multiply: `dst[r*n+c] = x[r*n+c] * vec[c]` for `r` in `0..rows`, `c`
    /// in `0..n` — the multiplicative twin of [`Op::AddBias`]. `vec` is a length-`n` weight
    /// (diffusion-gemma's router input scale `ffn_gate_inp.scale`, applied to the router's
    /// rmsnorm-noscale'd input before the router `Linear`; see `docs/diffusion-gemma.md`).
    MulVec {
        x: TensorId,
        vec: TensorId,
        dst: TensorId,
        rows: u32,
        n: u32,
    },
    /// Head-wise sigmoid gate: `dst[r,h,d] = x[r,h,d] * sigmoid(gate[r,h])`. Ling's gated MLA
    /// produces one scalar per attention head, then broadcasts it over that head's V width before
    /// the output projection. In place when `dst == x`.
    HeadwiseSigmoidMul {
        x: TensorId,
        gate: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
    },
    /// `dst[i] = cap * tanh(x[i] / cap)` (Gemma final-logit softcap).
    Softcap {
        x: TensorId,
        dst: TensorId,
        cap: f32,
        n: u32,
    },
    /// `dst[r] = argmax(x[r*n..(r+1)*n])` for `rows` rows, each id a u32 bit-pattern in an f32
    /// tensor slot. Greedy sampling on the device: the generated token id(s) are the ONLY thing
    /// that crosses back to the host (4 bytes/row), not the `[rows, vocab]` logits. Strict `>`
    /// keeps the lowest index on ties (per row), matching the host-side sampler. `rows == 1` is
    /// the decode-loop shape; `rows > 1` is the MTP speculative-verify accept (issue #31) —
    /// small m, so backends may run m sequential single-row reductions (a whole-vocab
    /// single-workgroup scan measured SLOWER than the download it replaced). Backends without
    /// multi-row support advertise `Capabilities::argmax_rows == false`; the runner gates.
    Argmax {
        x: TensorId,
        dst: TensorId,
        n: u32,
        rows: u32,
    },
    /// Fused single-row argmax + softmax top-1 probability (MTP draft-loop accept, issue #33
    /// follow-up to `Op::Argmax`'s VERIFY-side fusion, de35727): `dst_id[0] = argmax(x[0..n])`
    /// (u32 bit-pattern, strict `>` lowest-index tie-break — identical rule to [`Op::Argmax`]) and
    /// `dst_prob[0] = softmax(x[0..n])[dst_id[0]] = 1 / sum_j exp(x[j] - x[dst_id[0]])`. Replaces
    /// the MTP self-chaining draft loop's per-step full `[vocab]` logits download + host
    /// `argmax`/`exp`-sum scan with an 8-byte readback — the host scan (not the download bytes)
    /// was the measured dominant cost (~650-700us/step on a 151936-vocab head, vs ~25-50us for the
    /// download itself). Single row only (the draft loop self-chains one token at a time; unlike
    /// `Op::Argmax` there's no multi-row MTP-verify use case here — verify's accept is `Op::Argmax`
    /// with `rows = m`, a DIFFERENT accept rule that doesn't need a probability at all). Backends
    /// implement this as a two-stage reduction (256-way slice-parallel partials -> one-workgroup
    /// merge), the SAME shape `Op::Argmax` uses and for the SAME reason (a single-workgroup
    /// whole-vocab scan measured slower than the download it replaces — see that op's doc); the
    /// merge combines each stage's `(max, argmax, sum_exp)` triple via the standard online-softmax
    /// rule (`new_sum = a.sum*exp(a.max-new_max) + b.sum*exp(b.max-new_max)`), which is NOT
    /// guaranteed bit-identical to the host's strictly-sequential `sum_j exp(x[j]-max)` (parallel
    /// reduction reorders the float additions) — harmless in practice because
    /// `mtp::DEFAULT_P_MIN == 0.0` and a softmax top-1 probability is always `> 0.0`, so the
    /// `prob < p_min` accept/reject decision the caller makes from `dst_prob` can never flip on
    /// the default path regardless of reduction order; a future non-zero `p_min` near a genuine
    /// `prob` value would need the caller to tolerate the same ULP-level slack the two-stage
    /// `Op::Argmax` already accepts for ties. Backends without a device kernel advertise
    /// `Capabilities::argmax_prob == false`; the caller keeps the host logits-download path.
    ArgmaxProb {
        x: TensorId,
        dst_id: TensorId,
        dst_prob: TensorId,
        n: u32,
    },
    /// Device-side stochastic sampling: `dst[0] = sample(x[0..n])` (u32 id bit-pattern in the f32
    /// slot) via temperature + top-k + top-p, inverse-CDF'd with the uniform draw read from the
    /// 1-float `u` Input — the host draws u (4 bytes/token) and reads back only the id, the
    /// `[vocab]` logits never leave the device. Same order of operations as the host sampler
    /// (top-k select desc → softmax(temp) → nucleus cutoff → CDF walk), so the same `u` picks the
    /// same token. `top_k` must be `2..=64` (backend kernel bound); the runner gates.
    Sample {
        x: TensorId,
        u: TensorId,
        dst: TensorId,
        n: u32,
        top_k: u32,
        temp: f32,
        top_p: f32,
    },
    /// Gather + dequantize embedding rows: `dst[r, :] = table[ids[r], :] * scale` for `rows`
    /// rows of `ne` elements. `ids` is an I32 input holding token ids; `table` is the (quantized)
    /// `token_embd` Weight; `scale` bakes Gemma's sqrt(n_embd) embed scaling. Lets the host feed
    /// TOKEN IDS instead of dequantized f32 embedding rows — the model's input stream stays
    /// 4 bytes/token end to end.
    EmbedGather {
        ids: TensorId,
        table: TensorId,
        dst: TensorId,
        rows: u32,
        ne: u32,
        scale: f32,
    },
    /// Gather rows of an **I32 lookup table**: `dst[r, :] = table[ids[r], :]` for `rows` rows of
    /// `ne` integers, values copied verbatim. `ids` and `dst` are [`crate::DType::I32`] handles
    /// holding plain integers (the convention [`Op::EmbedGather`]'s `ids` uses); `table` is an I32
    /// Weight.
    ///
    /// This is `ggml_get_rows` on an integer tensor — llama.cpp preserves I32 there rather than
    /// dequantizing. Its one caller is DeepSeek V4's hash-routed MoE:
    /// `selected_experts = ggml_get_rows(layer.ffn_gate_tid2eid, inp_tokens)` produces the
    /// `[rows, n_expert_used]` selection [`Op::MoeFfn::expert_ids`] consumes, so `ids` is the
    /// graph's token-id Input and `ne` is `n_expert_used`.
    ///
    /// Deliberately NOT a mode of [`Op::EmbedGather`], which dequantizes a quantized row through the
    /// per-format block decoders into f32 and scales it: this op has no dtype ladder, no scale, and
    /// an integer destination. It also has no 32-element block structure to walk, which is what
    /// makes `EmbedGather` unusable here — an `n_expert_used`-wide row (6 or 8) is narrower than one
    /// sub-block.
    GatherI32 {
        ids: TensorId,
        table: TensorId,
        dst: TensorId,
        rows: u32,
        ne: u32,
    },
    /// Copy `n` elements `src[src_off..] -> dst[dst_off..]` (extract last row, gather a slice).
    Copy {
        src: TensorId,
        src_off: u32,
        dst: TensorId,
        dst_off: u32,
        n: u32,
    },
    /// Batched strided copy: for `rows` rows, copy `n` elements
    /// `src[src_off + r*src_stride ..] -> dst[dst_off + r*dst_stride ..]`. Used to split a batched
    /// `[rows, cc]` interleaved buffer (e.g. conv output q|k|v) into packed `[rows, n]` slices in one
    /// op. `Copy` is the rows=1 special case.
    CopyStrided {
        src: TensorId,
        src_off: u32,
        src_stride: u32,
        dst: TensorId,
        dst_off: u32,
        dst_stride: u32,
        rows: u32,
        n: u32,
    },
    /// Mixture-of-experts FFN for a single token row (qwen3moe; diffusion-gemma's MoE branch — see
    /// `docs/diffusion-gemma.md`). The router (`Linear` of `router_x[ne] → n_expert`) is softmaxed,
    /// the top-`n_used` experts selected, their softmax weights renormalized, and each runs a gated
    /// FFN on `x` (`act(gate·x) * (up·x)`, then `down·`); the outputs are summed weighted by the
    /// renormalized weights × `scale` into `dst[ne]` (the residual contribution).
    /// `gate_exps`/`up_exps`/`down_exps` are the stacked per-expert weights — expert `e` is the `e`-th
    /// equal byte slice (gate/up are `[n_ff_exp, ne]`, down is `[ne, n_ff_exp]` row-major).
    MoeFfn {
        x: TensorId,
        /// The router's own input row — usually the SAME tensor as `x` (qwen3moe: the router reads
        /// whatever normed input feeds the experts). diffusion-gemma's router reads a DIFFERENTLY
        /// normalized/scaled row of the same residual (`rmsnorm_noscale(attn_out)/√ne ·
        /// ffn_gate_inp.scale`, built with `Op::RmsNorm` + `Op::Scale` + `Op::MulVec` upstream), so
        /// it's a separate handle rather than reusing `x`.
        router_x: TensorId,
        router: TensorId,
        gate_exps: TensorId,
        /// Ignored when `fused_gate_up` is set (the call site passes the same handle as
        /// `gate_exps` — never read).
        up_exps: TensorId,
        down_exps: TensorId,
        /// Per-expert scale on the selected expert's DOWN-projection output BEFORE the weighted
        /// sum (diffusion-gemma `ffn_down_exps.scale[n_expert]`, one f32 per expert). `None` = no
        /// scale (qwen3moe).
        down_scale: Option<TensorId>,
        /// `gate_exps` holds gate AND up FUSED into one `[ne, 2*n_ff_exp, n_expert]` tensor (gate
        /// rows first, up rows second — the same "gate half, up half" per-expert-slice convention
        /// as `Op::GatedActFused`'s combined `gu` buffer). `up_exps` is unused when `true`
        /// (diffusion-gemma's `ffn_gate_up_exps`); `false` = separate `gate_exps`/`up_exps` tensors
        /// (qwen3moe).
        fused_gate_up: bool,
        dst: TensorId,
        ne: u32,
        n_expert: u32,
        n_used: u32,
        n_ff_exp: u32,
        scale: f32,
        act: Activation,
        /// DeepSeek V4's per-layer SwiGLU clamp for the ROUTED experts — `swiglu_clamp_exp[il]`,
        /// resolved through [`swiglu_clamp`]. Identical arithmetic to
        /// [`Op::GatedAct::swiglu_clamp`] (see there for the pre-activation / one-sided-gate
        /// details), applied per selected expert to that expert's own gate/up projections, i.e.
        /// between the gate/up GEMMs and the down GEMM. llama.cpp reads it from a different
        /// hparam array than the shared expert's, so the two clamps can differ within a layer.
        swiglu_clamp: Option<f32>,
        /// Router gating (softmax over experts vs per-expert sigmoid). `Softmax` for
        /// qwen3moe/qwen35moe/diffusion-gemma; `Sigmoid` for llama4.
        gating: MoeGating,
        /// Renormalize the selected top-k expert weights to sum to 1 before scaling
        /// (`w[e] = probs[e] / Σprobs · scale`). `true` for softmax MoE (the reference
        /// `norm_w`); `false` for llama4 (top-1, weight = `sigmoid(logit) · scale`, no renorm).
        norm_w: bool,
        /// Apply the per-expert routing weight to the expert INPUT (before the gate/up projections
        /// and activation) rather than to the expert OUTPUT. `true` only for llama4 (its
        /// `weight_before_ffn`); the two differ through the SiLU nonlinearity. Folded into the
        /// gate/up activations on CPU (`silu(w·gate)·(w·up)`), exact since gate/up are linear.
        weight_before: bool,
        /// Expert-parallel (multi-GPU EP) band: `Some((base, n_local))` means the bound expert banks
        /// (`gate_exps`/`up_exps`/`down_exps`) hold ONLY this rank's contiguous expert shard
        /// `[base, base+n_local)` (of the global `n_expert`), so the op routes GLOBALLY (full
        /// `router`/`n_expert` top-k, replicated across ranks) but computes only its owned experts —
        /// the assignments to other ranks' experts are dropped (weight 0). The producing MoE output
        /// (`dst`) is then a PARTIAL that the EP backend all-reduces (sums) across ranks to the full
        /// weighted top-k output. `None` (the DEFAULT) = ordinary single-device MoE over all
        /// `n_expert` experts, byte-identical to before this field existed. Set only by
        /// `infr_vulkan::ExpertParallelBackend`'s per-rank graph lowering; every model builder and
        /// the CPU/Metal reference interpreters leave it `None` (EP is a Vulkan-only path).
        ep_band: Option<(u32, u32)>,
        /// DeepSeek V2+: per-layer router bias `[n_expert]` added to logits for SELECTION only;
        /// the unbias'd probs are still used for the per-expert routing weights. `None` = no bias.
        /// Must be `None` when [`Op::MoeFfn::expert_ids`] is set — there is no selection to bias.
        exp_probs_b: Option<TensorId>,
        /// DeepSeek V3+: group-limited routing — number of expert groups (0 = no grouping). Must be
        /// `0`/`1` when [`Op::MoeFfn::expert_ids`] is set (same reason as `exp_probs_b`).
        n_expert_groups: u32,
        /// DeepSeek V3+: number of groups selected per routing decision.
        n_expert_groups_used: u32,
        /// DeepSeek V4 **hash-routed MoE** (its first `hash_layer_count` layers): the selected
        /// expert ids `[rows, n_used]`, i32, ALREADY gathered by the caller — llama.cpp's
        /// `selected_experts = ggml_get_rows(layer.ffn_gate_tid2eid, inp_tokens)`, a
        /// `{n_expert_used, n_vocab}` lookup table indexed by TOKEN ID (`deepseek4.cpp`'s MoE
        /// block), handed to `build_moe_ffn` as its `selected_experts_in`. The tensor is an
        /// [`crate::DType::I32`] handle whose values are read as PLAIN INTEGERS — the convention
        /// [`Op::EmbedGather`]'s `ids` uses, not [`Op::Argmax`]'s bit-pattern-in-an-f32-slot
        /// carrier (that one exists for handles a backend declares f32).
        ///
        /// What supplying them changes, read off `build_moe_ffn` (`src/llama-graph.cpp`): **only
        /// the selection**. The router matmul still runs, the [`MoeGating`] function still turns
        /// its logits into `probs`, and the routing WEIGHTS are still
        /// `ggml_get_rows(probs, selected_experts)` — i.e. the router's own probability at each
        /// hash-chosen expert, then `norm_w` and `scale` exactly as on a top-k layer. They are NOT
        /// uniform. What is bypassed is `ggml_argsort_top_k` and everything feeding only it:
        /// `exp_probs_b` and the group masking build `selection_probs`, which nothing reads once
        /// `selected_experts_in` is non-null (llama.cpp passes `exp_probs_b = nullptr` on a hash
        /// layer for the same reason). Backends must SKIP that work rather than compute and
        /// overwrite it, and refuse `exp_probs_b`/`n_expert_groups > 1` alongside these ids.
        ///
        /// `None` = ordinary top-k routing, the path every other arch and every non-hash V4 layer
        /// takes, byte-identical to before this field existed.
        expert_ids: Option<TensorId>,
    },
    /// Depthwise causal 1-D conv over `channels` followed by SiLU (qwen35 gated DeltaNet).
    /// Processes `rows` tokens sequentially, carrying the rolling history in `state` across rows and
    /// leaving it updated after the last row. `x`/`dst` are `[rows, channels]`; `weight` is the
    /// per-channel kernel `[channels, kernel]`; `state` is the rolling `[(kernel-1), channels]`
    /// history (oldest row first). Per token: `dst[ch] = silu(Σ_{j<kernel-1} state[j,ch]·w[ch,j] +
    /// x[ch]·w[ch,K-1])`, then history shifts (drop oldest, append raw `x`). `rows=1` = one token.
    Conv1dSilu {
        x: TensorId,
        weight: TensorId,
        state: TensorId,
        dst: TensorId,
        rows: u32,
        channels: u32,
        kernel: u32,
    },
    /// Gated-DeltaNet linear-attention recurrence step (qwen35), one token. Per VALUE head:
    /// L2-normalize `q`,`k`; scale `q` by `1/√head_k`; `beta = sigmoid(b)`, `decay =
    /// exp(a_coef·softplus(a + dt_bias))`; update the persistent state `S[head_k, head_v]`: `S *=
    /// decay`, `delta = (v − Sᵀk)·beta`, `S += k⊗delta`; `dst = Sᵀq`. GQA linear attention: `n_vhead`
    /// value heads share `n_khead` query/key heads in contiguous groups of `n_vhead/n_khead` — value
    /// head `h` uses q/k head `h/(n_vhead/n_khead)`. `q`/`k` are `[n_khead·head_k]`, `v`/`dst` are
    /// `[n_vhead·head_v]`, `b`/`a` are `[n_vhead]`, `a_coef`/`dt_bias` are weights `[n_vhead]`,
    /// `state` is `[n_vhead·head_k·head_v]` (mutated in place). Processes `rows` tokens sequentially,
    /// carrying `state` across rows (and leaving it updated after the last). `q`/`k` are
    /// `[rows, n_khead·head_k]`, `v`/`dst` are `[rows, n_vhead·head_v]`, `b`/`a` are `[rows, n_vhead]`.
    /// `rows=1` = one token.
    DeltaNet {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        b: TensorId,
        a: TensorId,
        a_coef: TensorId,
        dt_bias: TensorId,
        state: TensorId,
        dst: TensorId,
        rows: u32,
        n_vhead: u32,
        n_khead: u32,
        head_k: u32,
        head_v: u32,
        eps: f32,
        /// When >0, q/k/v are slices of a single source buffer with per-row stride.
        /// q at offset 0, k at n_khead*head_k, v at 2*n_khead*head_k within each row.
        /// Eliminates 3 CopyStrided dispatches per DeltaNet layer (qwen35).
        src_stride: u32,
    },
    /// Ling KDA recurrent delta-rule attention. `qkv` is `[rows, 3*n_head*head_dim]` with packed
    /// Q, K, V blocks per row after projection + causal conv + SiLU. `forget` is
    /// `[rows, n_head*head_dim]`, `beta` is `[rows, n_head]`, `a` is the converted `exp(A_log)`
    /// weight `[n_head]`, and `dt_bias` is `[n_head*head_dim]`. Per token/head, Q and K are L2
    /// normalized, Q is scaled by `1/sqrt(head_dim)`, and each state row receives its own decay:
    /// `decay[k] = exp(lower_bound * sigmoid(a[h] * (forget[k] + dt_bias[k])))`. Then
    /// `delta = (v - k^T S) * sigmoid(beta)`, `S += outer(k, delta)`, `dst = q^T S`.
    Kda {
        qkv: TensorId,
        forget: TensorId,
        beta: TensorId,
        a: TensorId,
        dt_bias: TensorId,
        state: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
        lower_bound: f32,
    },
    /// qwen35moe Qwen2-MoE-style shared-expert combine: `dst[r,c] = moe[r,c] + sigmoid(gate[r]) *
    /// shexp[r,c]` for `rows` rows of `n` elements. `moe` is the routed-MoE branch's output
    /// (`Op::MoeFfn`'s `dst`); `shexp` is the shared expert's own dense SwiGLU FFN output (a
    /// plain `Linear`→`GatedAct`→`Linear` on the SAME input, run alongside the routed branch);
    /// `gate` holds ONE raw (pre-sigmoid) logit per row — the output of a `Linear` with
    /// `out_f=1` against `ffn_gate_inp_shexp`. Fuses the per-token sigmoid gate + broadcast
    /// multiply + residual add into one op (the shared-expert twin of `GatedActFused`).
    MoeSharedExpertAdd {
        moe: TensorId,
        shexp: TensorId,
        gate: TensorId,
        dst: TensorId,
        rows: u32,
        n: u32,
    },
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Op {
    /// The op's variant name — used by backends for per-op profiling / error messages so the
    /// mapping lives in ONE place (was duplicated as `op_kind`/`op_name` in each backend).
    pub fn kind(&self) -> &'static str {
        match self {
            Op::RmsNorm { .. } => "RmsNorm",
            Op::RmsNormAdd { .. } => "RmsNormAdd",
            Op::LayerNorm { .. } => "LayerNorm",
            Op::Softmax { .. } => "Softmax",
            Op::Linear { .. } => "Linear",
            Op::QkNorm { .. } => "QkNorm",
            Op::GatedRmsNorm { .. } => "GatedRmsNorm",
            Op::Rope { .. } => "Rope",
            Op::QkNormRope { .. } => "QkNormRope",
            Op::WriteKv { .. } => "WriteKv",
            Op::Dsv4Compress { .. } => "Dsv4Compress",
            Op::Dsv4CacheWrite { .. } => "Dsv4CacheWrite",
            Op::Dsv4Indexer { .. } => "Dsv4Indexer",
            Op::Dsv4Gather { .. } => "Dsv4Gather",
            Op::QsaIndexer { .. } => "QsaIndexer",
            Op::QsaGather { .. } => "QsaGather",
            Op::QsaBatchAttention { .. } => "QsaBatchAttention",
            Op::Attention { .. } => "Attention",
            Op::Mla { .. } => "Mla",
            Op::LightningIndexer { .. } => "LightningIndexer",
            Op::TopkMask { .. } => "TopkMask",
            Op::HyperConnectMix { .. } => "HyperConnectMix",
            Op::HyperConnectPre { .. } => "HyperConnectPre",
            Op::HyperConnectPost { .. } => "HyperConnectPost",
            Op::GatedAct { .. } => "GatedAct",
            Op::GatedActFused { .. } => "GatedActFused",
            Op::Add { .. } => "Add",
            Op::AddBias { .. } => "AddBias",
            Op::Scale { .. } => "Scale",
            Op::Silu { .. } => "Silu",
            Op::QwenHcMix { .. } => "QwenHcMix",
            Op::QwenHcInject { .. } => "QwenHcInject",
            Op::QwenPleGate { .. } => "QwenPleGate",
            Op::MulVec { .. } => "MulVec",
            Op::HeadwiseSigmoidMul { .. } => "HeadwiseSigmoidMul",
            Op::Softcap { .. } => "Softcap",
            Op::Argmax { .. } => "Argmax",
            Op::ArgmaxProb { .. } => "ArgmaxProb",
            Op::Sample { .. } => "Sample",
            Op::EmbedGather { .. } => "EmbedGather",
            Op::GatherI32 { .. } => "GatherI32",
            Op::Copy { .. } => "Copy",
            Op::CopyStrided { .. } => "CopyStrided",
            Op::MoeFfn { .. } => "MoeFfn",
            Op::Conv1dSilu { .. } => "Conv1dSilu",
            Op::DeltaNet { .. } => "DeltaNet",
            Op::Kda { .. } => "Kda",
            Op::MoeSharedExpertAdd { .. } => "MoeSharedExpertAdd",
        }
    }

    /// The tensor handles this op READS and the ones it WRITES, as `(reads, writes)`.
    ///
    /// Used by the multi-device pipeline executor (`infr-vulkan`'s `PipelineBackend`) to infer,
    /// from the DEVICE each bound operand lives on, which physical device an op runs on — and to
    /// detect the cross-device "cut" tensors (a handle written by an op on device A and read by an
    /// op on device B) that must be handed off at the layer-split boundary. An IN-PLACE update
    /// (`dst == x`, a `+=`, or a stateful `state`/`cache` write) appears in BOTH lists: it is read
    /// and written. Kept exhaustive (no `_` arm) so a new [`Op`] variant forces a decision here.
    pub fn io(&self) -> (Vec<TensorId>, Vec<TensorId>) {
        match *self {
            Op::RmsNorm { x, weight, dst, .. } => (vec![x, weight], vec![dst]),
            Op::RmsNormAdd { x, weight, dst, .. } => (vec![x, weight, dst], vec![dst]),
            Op::LayerNorm {
                x,
                weight,
                bias,
                dst,
                ..
            } => (vec![x, weight, bias], vec![dst]),
            Op::Softmax {
                x, dst, scale_buf, ..
            } => {
                let mut r = vec![x];
                r.extend(scale_buf);
                (r, vec![dst])
            }
            Op::Linear { x, weight, dst, .. } => (vec![x, weight], vec![dst]),
            Op::QkNorm { x, weight, dst, .. } => {
                let mut r = vec![x];
                r.extend(weight);
                (r, vec![dst])
            }
            Op::GatedRmsNorm {
                x,
                weight,
                gate,
                dst,
                ..
            } => (vec![x, weight, gate], vec![dst]),
            Op::Rope {
                x,
                positions,
                dst,
                freq_factors,
                ..
            } => {
                let mut r = vec![x, positions];
                r.extend(freq_factors);
                (r, vec![dst])
            }
            Op::QkNormRope {
                x,
                weight,
                positions,
                dst,
                freq_factors,
                ..
            } => {
                let mut r = vec![x, weight, positions];
                r.extend(freq_factors);
                (r, vec![dst])
            }
            Op::WriteKv { src, cache, .. } => (vec![src, cache], vec![cache]),
            Op::Dsv4Compress {
                values,
                scores,
                ape,
                norm,
                freq_factors,
                state,
                dst,
                ..
            } => {
                let mut r = vec![values, scores, ape, norm, state];
                r.extend(freq_factors);
                (r, vec![state, dst])
            }
            Op::Dsv4CacheWrite { src, cache, .. } => (vec![src, cache], vec![cache]),
            Op::Dsv4Indexer {
                q,
                k_cache,
                weights,
                dst,
                ..
            } => (vec![q, k_cache, weights], vec![dst]),
            Op::Dsv4Gather {
                raw_cache,
                comp_cache,
                indices,
                dst,
                ..
            } => {
                let mut r = vec![raw_cache, comp_cache];
                r.extend(indices);
                (r, vec![dst])
            }
            Op::QsaIndexer {
                q,
                k_cache,
                k_norm,
                dst,
                ..
            } => (vec![q, k_cache, k_norm], vec![dst]),
            Op::QsaGather {
                k_cache,
                v_cache,
                indices,
                k_dst,
                v_dst,
                ..
            } => (vec![k_cache, v_cache, indices], vec![k_dst, v_dst]),
            Op::QsaBatchAttention {
                q,
                k_cache,
                v_cache,
                indices,
                dst,
                ..
            } => (vec![q, k_cache, v_cache, indices], vec![dst]),
            Op::Attention {
                q,
                k_cache,
                v_cache,
                dst,
                sinks,
                ..
            } => {
                let mut r = vec![q, k_cache, v_cache];
                r.extend(sinks);
                (r, vec![dst])
            }
            Op::Mla {
                q,
                k_cache,
                wk_b,
                wv_b,
                dst,
                key_bias,
                ..
            } => {
                let mut r = vec![q, k_cache, wk_b, wv_b];
                r.extend(key_bias);
                (r, vec![dst])
            }
            Op::LightningIndexer {
                q,
                k_cache,
                weights,
                dst,
                ..
            } => (vec![q, k_cache, weights], vec![dst]),
            Op::TopkMask { idx, dst, .. } => (vec![idx], vec![dst]),
            Op::HyperConnectMix {
                mixes,
                scale,
                base,
                pre,
                gates,
                ..
            } => {
                let mut w = vec![pre];
                if let Some(HyperGates { post, comb }) = gates {
                    w.push(post);
                    w.push(comb);
                }
                (vec![mixes, scale, base], w)
            }
            Op::HyperConnectPre {
                x, weights, dst, ..
            } => (vec![x, weights], vec![dst]),
            Op::HyperConnectPost {
                x,
                residual,
                post,
                comb,
                dst,
                ..
            } => (vec![x, residual, post, comb], vec![dst]),
            Op::GatedAct { gate, up, dst, .. } => (vec![gate, up], vec![dst]),
            Op::GatedActFused { gu, dst, .. } => (vec![gu], vec![dst]),
            Op::Add { a, b, dst, .. } => (vec![a, b], vec![dst]),
            Op::AddBias { x, bias, dst, .. } => (vec![x, bias], vec![dst]),
            Op::Scale { x, dst, .. } => (vec![x], vec![dst]),
            Op::Silu { x, dst, .. } => (vec![x], vec![dst]),
            Op::QwenHcMix { x, gate, dst, .. } => (vec![x, gate], vec![dst]),
            Op::QwenHcInject {
                residual,
                block,
                gate,
                dst,
                ..
            } => (vec![residual, block, gate], vec![dst]),
            Op::QwenPleGate {
                key,
                query,
                value,
                dst,
                ..
            } => (vec![key, query, value], vec![dst]),
            Op::MulVec { x, vec: v, dst, .. } => (vec![x, v], vec![dst]),
            Op::HeadwiseSigmoidMul { x, gate, dst, .. } => (vec![x, gate], vec![dst]),
            Op::Softcap { x, dst, .. } => (vec![x], vec![dst]),
            Op::Argmax { x, dst, .. } => (vec![x], vec![dst]),
            Op::ArgmaxProb {
                x,
                dst_id,
                dst_prob,
                ..
            } => (vec![x], vec![dst_id, dst_prob]),
            Op::Sample { x, u, dst, .. } => (vec![x, u], vec![dst]),
            Op::EmbedGather {
                ids, table, dst, ..
            } => (vec![ids, table], vec![dst]),
            Op::GatherI32 {
                ids, table, dst, ..
            } => (vec![ids, table], vec![dst]),
            Op::Copy { src, dst, .. } => (vec![src], vec![dst]),
            Op::CopyStrided { src, dst, .. } => (vec![src], vec![dst]),
            Op::MoeFfn {
                x,
                router_x,
                router,
                gate_exps,
                up_exps,
                down_exps,
                down_scale,
                expert_ids,
                dst,
                ..
            } => {
                let mut r = vec![x, router_x, router, gate_exps, up_exps, down_exps];
                r.extend(down_scale);
                r.extend(expert_ids);
                (r, vec![dst])
            }
            Op::Conv1dSilu {
                x,
                weight,
                state,
                dst,
                ..
            } => (vec![x, weight, state], vec![state, dst]),
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
                ..
            } => (
                vec![q, k, v, b, a, a_coef, dt_bias, state],
                vec![state, dst],
            ),
            Op::Kda {
                qkv,
                forget,
                beta,
                a,
                dt_bias,
                state,
                dst,
                ..
            } => (vec![qkv, forget, beta, a, dt_bias, state], vec![state, dst]),
            Op::MoeSharedExpertAdd {
                moe,
                shexp,
                gate,
                dst,
                ..
            } => (vec![moe, shexp, gate], vec![dst]),
        }
    }
}

/// An ordered op-list over declared tensor handles. Node index in `tensors` == [`TensorId`].
#[derive(Clone, Default)]
pub struct Graph {
    pub tensors: Vec<TensorDecl>,
    pub ops: Vec<Op>,
    pub inputs: Vec<TensorId>,
    pub weights: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    /// Producer-set opt-out of the Vulkan record-once decode replay: `true` forces the
    /// per-execute STATIC path even for an otherwise replay-eligible single-token decode.
    ///
    /// The replay tape lowers the pos-dependent ops through a DIFFERENT kernel family (the
    /// params-driven `_dyn` kernels, with worst-case-capacity split-K chunking) than the static
    /// recording — same math, different float-accumulation order, so the two paths agree only to
    /// reassociation-level noise (~1 f16 ULP on the KV row a decode writes). Autoregressive
    /// decode tolerates that (greedy/top-k sampling is robust to sub-ULP logit noise), but
    /// DiffusionGemma's entropy-bound denoise loop is chaotic in it: the committed-prefix KV row
    /// the seam's decode loop writes (the prefill frontier token) seeds EVERY canvas row's
    /// attention, and a 128-expert top-8 MoE amplifies a ~1e-3 f16 KV delta into flipped
    /// argmax/acceptance decisions — replay-mode text visibly diverges from the static path the
    /// CPU reference/goldens validate. The seam sets this on diffusion-gemma graphs so both
    /// execution modes run the SAME (static) kernels bit-identically; everything else keeps the
    /// replay fast path. See `infr-vulkan`'s `decode_eligible`.
    pub no_decode_replay: bool,
    /// Set `true` ONLY for an MTP-verify batched forward — the trunk's speculative VERIFY pass
    /// (`crates/infr-llama/src/mtp/mod.rs`'s `run_verify`/`run_verify_full`, driven through
    /// `generate_dense_backend`'s `verify` branch). `false` (the `Graph::default()`/`Graph::new()`
    /// value) for every other graph this seam builds: the per-token decode loop, the chunked
    /// ordinary batched-prefill path, and the DiffusionGemma canvas denoise.
    ///
    /// Why this exists: MTP verify's greedy output must bit-match plain (non-speculative) decode
    /// at the same position — that's `mtp_spec_matches_target_only_greedy`'s whole contract, and
    /// the historical Q5_K bug (README footnote 2) was exactly this bit-identity breaking. Ordinary
    /// prefill has no such partner dispatch to agree with: it's one path through the model, free to
    /// take whichever kernel measures fastest. Before this flag, both were just "m>=3 batched
    /// forward" to the kernel-selection code, so a dtype's int8 `mrow` tier could only be unlocked
    /// for prefill by ALSO unlocking it for MTP verify — which is what broke token-identity. See
    /// `infr_vulkan::adapter`'s `mrow_int8_dtype_ok` for the consumer.
    pub mtp_verify: bool,
    /// Memoized [`Self::in_place_inputs`] — a graph invariant (which KV-cache `Input`s the ops
    /// mutate in place), computed lazily on first query and reused. `execute` calls it PER TOKEN;
    /// without this it re-scanned every op and re-allocated a `HashSet` each call. Interior-mutable
    /// (`OnceLock`) so it fills through a shared `&Graph` (the plan holds the graph immutably).
    /// Not part of the graph's identity — cloning carries a filled cache along if present, and an
    /// empty one refills once on the clone's first query.
    in_place_cache: std::sync::OnceLock<std::collections::HashSet<TensorId>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// The `Input` tensors written IN PLACE by the graph's ops (the KV cache: `WriteKv`'s `cache` and
    /// `Attention`'s `k_cache`/`v_cache`). This is pure graph semantics, so it lives here rather than
    /// being rediscovered per backend: an eager-load backend (like the CPU interpreter) skips loading
    /// these into its working store + skips writing them back (they're mutated directly), avoiding
    /// O(max_ctx) copies per step.
    ///
    /// The set is a graph INVARIANT (the ops never change after compile), so it is computed once and
    /// MEMOIZED — `execute` queries it per token and must not re-scan every op / re-alloc a
    /// `HashSet` each call. Returns a borrow of the cached set.
    pub fn in_place_inputs(&self) -> &std::collections::HashSet<TensorId> {
        self.in_place_cache.get_or_init(|| {
            let mut set = std::collections::HashSet::new();
            for op in &self.ops {
                match op {
                    Op::WriteKv { cache, .. } => {
                        set.insert(*cache);
                    }
                    Op::Dsv4Compress { state, .. } => {
                        set.insert(*state);
                    }
                    Op::Dsv4CacheWrite { cache, .. } => {
                        set.insert(*cache);
                    }
                    Op::Dsv4Indexer { k_cache, .. } => {
                        set.insert(*k_cache);
                    }
                    Op::Dsv4Gather {
                        raw_cache,
                        comp_cache,
                        ..
                    } => {
                        set.insert(*raw_cache);
                        set.insert(*comp_cache);
                    }
                    Op::QsaIndexer { k_cache, .. } => {
                        set.insert(*k_cache);
                    }
                    Op::QsaGather {
                        k_cache, v_cache, ..
                    } => {
                        set.insert(*k_cache);
                        set.insert(*v_cache);
                    }
                    Op::QsaBatchAttention {
                        k_cache, v_cache, ..
                    } => {
                        set.insert(*k_cache);
                        set.insert(*v_cache);
                    }
                    Op::Attention {
                        k_cache, v_cache, ..
                    } => {
                        set.insert(*k_cache);
                        set.insert(*v_cache);
                    }
                    // Both DeepSeek ops read a one-row-per-token cache the same way: straight out
                    // of its BOUND buffer at the op's own dtype, never through the interpreter's
                    // f32 working store. `Skip` is what keeps the store from mirroring (and, for
                    // an f32-declared cache, writing back) a buffer no op here ever wrote.
                    Op::Mla { k_cache, .. } | Op::LightningIndexer { k_cache, .. } => {
                        set.insert(*k_cache);
                    }
                    _ => {}
                }
            }
            set
        })
    }

    fn decl(&mut self, desc: TensorDesc, kind: TensorKind) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorDecl {
            desc,
            kind,
            label: None,
        });
        id
    }

    /// Declare a per-step input (bound at execute time).
    pub fn input(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Input);
        self.inputs.push(id);
        id
    }

    /// Declare a model weight (bound from the loader).
    pub fn weight(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Weight);
        self.weights.push(id);
        id
    }

    /// Declare backend-allocated scratch.
    pub fn internal(&mut self, desc: TensorDesc) -> TensorId {
        self.decl(desc, TensorKind::Internal)
    }

    /// Declare a read-back output.
    pub fn output(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Output);
        self.outputs.push(id);
        id
    }

    /// Attach a debug label to a tensor handle.
    pub fn label(&mut self, id: TensorId, label: impl Into<String>) -> TensorId {
        self.tensors[id.0 as usize].label = Some(label.into());
        id
    }

    /// Append an op to the list.
    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn desc(&self, id: TensorId) -> &TensorDesc {
        &self.tensors[id.0 as usize].desc
    }

    pub fn kind(&self, id: TensorId) -> TensorKind {
        self.tensors[id.0 as usize].kind
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, TensorDesc};

    /// `Op::io` reports the exact read/write handles — the contract the multi-device pipeline
    /// executor infers each op's device and cut tensors from.
    #[test]
    fn op_io_reads_and_writes() {
        let t = |n: u32| TensorId(n);
        // Linear reads x + weight, writes dst.
        let lin = Op::Linear {
            x: t(0),
            weight: t(1),
            dst: t(2),
            m: 1,
            in_f: 4,
            out_f: 4,
            w_off: 0,
        };
        assert_eq!(lin.io(), (vec![t(0), t(1)], vec![t(2)]));

        // In-place residual Add: reads a + b, writes dst (== a here).
        let add = Op::Add {
            a: t(2),
            b: t(3),
            dst: t(2),
            n: 4,
        };
        assert_eq!(add.io(), (vec![t(2), t(3)], vec![t(2)]));

        // WriteKv: the KV cache is BOTH read and written (stateful append pins the op's device).
        let wk = Op::WriteKv {
            src: t(5),
            cache: t(6),
            rows: 1,
            row_stride: 8,
            pos: 0,
        };
        assert_eq!(wk.io(), (vec![t(5), t(6)], vec![t(6)]));

        // Attention reads q + both caches, writes dst (caches are read-only here — WriteKv wrote them).
        let attn = Op::Attention {
            q: t(7),
            k_cache: t(6),
            v_cache: t(8),
            dst: t(9),
            rows: 1,
            kv_len: 1,
            n_head: 1,
            n_kv: 1,
            head_dim: 8,
            scale: 1.0,
            mask: AttnMask::Causal,
            pos: 0,
            sinks: None,
        };
        assert_eq!(attn.io(), (vec![t(7), t(6), t(8)], vec![t(9)]));
    }

    /// Optional read operands (rope `freq_factors`) appear only when present.
    #[test]
    fn op_io_optional_operands() {
        let t = |n: u32| TensorId(n);
        let rope = |ff: Option<TensorId>| Op::Rope {
            x: t(0),
            positions: t(1),
            dst: t(0),
            rows: 1,
            n_head: 1,
            head_dim: 8,
            rope_dim: 8,
            theta: 1e4,
            freq_factors: ff,
            x_stride: 0,
            neox: false,
            backward: false,
        };
        assert_eq!(rope(None).io(), (vec![t(0), t(1)], vec![t(0)]));
        assert_eq!(rope(Some(t(2))).io(), (vec![t(0), t(1), t(2)], vec![t(0)]));
        // `Op::QkNorm`'s weight and `Op::Attention`'s sinks are the same shape of optional read.
        let qkn = |w: Option<TensorId>| Op::QkNorm {
            x: t(0),
            weight: w,
            dst: t(0),
            rows: 1,
            n_head: 2,
            head_dim: 4,
            eps: 1e-6,
            x_stride: 0,
        };
        assert_eq!(
            qkn(None).io(),
            (vec![t(0)], vec![t(0)]),
            "V4's weightless Q norm reads no weight at all"
        );
        assert_eq!(qkn(Some(t(1))).io(), (vec![t(0), t(1)], vec![t(0)]));
        let attn = |sk: Option<TensorId>| Op::Attention {
            q: t(0),
            k_cache: t(1),
            v_cache: t(2),
            dst: t(3),
            rows: 1,
            kv_len: 1,
            n_head: 1,
            n_kv: 1,
            head_dim: 8,
            scale: 1.0,
            mask: AttnMask::Causal,
            pos: 0,
            sinks: sk,
        };
        assert_eq!(
            attn(None).io(),
            (vec![t(0), t(1), t(2)], vec![t(3)]),
            "every existing arch (sinks None) reads exactly what it always did"
        );
        assert_eq!(
            attn(Some(t(4))).io(),
            (vec![t(0), t(1), t(2), t(4)], vec![t(3)])
        );
        // `Op::Mla`'s optional top-k score mask is the same shape of optional read operand.
        let mla = |kb: Option<TensorId>| Op::Mla {
            q: t(0),
            k_cache: t(1),
            wk_b: t(2),
            wv_b: t(3),
            dst: t(4),
            rows: 1,
            kv_len: 1,
            n_head: 1,
            q_head_dim: 8,
            kv_lora_rank: 4,
            qk_nope_dim: 4,
            qk_rope_dim: 4,
            v_head_dim: 4,
            scale: 1.0,
            mask: AttnMask::Causal,
            pos: 0,
            theta: 1e4,
            freq_factors: None,
            key_bias: kb,
        };
        assert_eq!(
            mla(None).io(),
            (vec![t(0), t(1), t(2), t(3)], vec![t(4)]),
            "deepseek2 (key_bias None) reads exactly what it always did"
        );
        assert_eq!(
            mla(Some(t(5))).io(),
            (vec![t(0), t(1), t(2), t(3), t(5)], vec![t(4)])
        );
        // A minimal graph round-trips a declared handle's kind (keeps the tensor imports live).
        let mut g = Graph::new();
        let w = g.weight(TensorDesc::new(vec![4], DType::F32));
        assert_eq!(g.kind(w), TensorKind::Weight);
    }

    /// `in_place_inputs` must report exactly the KV-cache handles the ops mutate in place
    /// (`WriteKv`'s `cache`, `Attention`'s `k_cache`/`v_cache`) — the set the CPU/Metal `execute`
    /// use to skip the O(max_ctx) working-store round-trip — and must be MEMOIZED (computed once,
    /// the same set handed back per token) rather than rescanned/re-allocated per call.
    #[test]
    fn in_place_inputs_is_the_kv_set_and_memoized() {
        let t = |n: u32| TensorId(n);
        let mut g = Graph::new();
        g.push(Op::WriteKv {
            src: t(5),
            cache: t(6),
            rows: 1,
            row_stride: 8,
            pos: 0,
        });
        g.push(Op::Attention {
            q: t(7),
            k_cache: t(6),
            v_cache: t(8),
            dst: t(9),
            rows: 1,
            kv_len: 1,
            n_head: 1,
            n_kv: 1,
            head_dim: 8,
            scale: 1.0,
            mask: AttnMask::Causal,
            pos: 0,
            sinks: None,
        });
        // A non-KV op must NOT contribute any in-place input.
        g.push(Op::Add {
            a: t(9),
            b: t(9),
            dst: t(9),
            n: 8,
        });

        let want: std::collections::HashSet<TensorId> = [t(6), t(8)].into_iter().collect();
        let first = g.in_place_inputs();
        assert_eq!(first, &want);

        // Memoized: a second query returns the SAME cached set (same address), never recomputed.
        let first_ptr = first as *const _;
        let second = g.in_place_inputs();
        assert_eq!(second, &want);
        assert_eq!(
            second as *const _, first_ptr,
            "set was recomputed, not cached"
        );

        // A clone with an already-filled cache carries the same set (byte-identical semantics).
        assert_eq!(g.clone().in_place_inputs(), &want);
    }
}
