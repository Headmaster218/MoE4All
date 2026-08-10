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
        weight: TensorId,
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
    /// NEOX RoPE over the first `rope_dim` of each head. `positions` is an i32 tensor of length
    /// `rows`. `freq_factors`, if present, divides per-pair angles (Gemma proportional RoPE).
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
        exp_probs_b: Option<TensorId>,
        /// DeepSeek V3+: group-limited routing — number of expert groups (0 = no grouping).
        n_expert_groups: u32,
        /// DeepSeek V3+: number of groups selected per routing decision.
        n_expert_groups_used: u32,
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
            Op::Attention { .. } => "Attention",
            Op::Mla { .. } => "Mla",
            Op::LightningIndexer { .. } => "LightningIndexer",
            Op::GatedAct { .. } => "GatedAct",
            Op::GatedActFused { .. } => "GatedActFused",
            Op::Add { .. } => "Add",
            Op::AddBias { .. } => "AddBias",
            Op::Scale { .. } => "Scale",
            Op::MulVec { .. } => "MulVec",
            Op::Softcap { .. } => "Softcap",
            Op::Argmax { .. } => "Argmax",
            Op::ArgmaxProb { .. } => "ArgmaxProb",
            Op::Sample { .. } => "Sample",
            Op::EmbedGather { .. } => "EmbedGather",
            Op::Copy { .. } => "Copy",
            Op::CopyStrided { .. } => "CopyStrided",
            Op::MoeFfn { .. } => "MoeFfn",
            Op::Conv1dSilu { .. } => "Conv1dSilu",
            Op::DeltaNet { .. } => "DeltaNet",
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
            Op::QkNorm { x, weight, dst, .. } => (vec![x, weight], vec![dst]),
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
            Op::Attention {
                q,
                k_cache,
                v_cache,
                dst,
                ..
            } => (vec![q, k_cache, v_cache], vec![dst]),
            Op::Mla {
                q,
                k_cache,
                wk_b,
                wv_b,
                dst,
                ..
            } => (vec![q, k_cache, wk_b, wv_b], vec![dst]),
            Op::LightningIndexer {
                q,
                k_cache,
                weights,
                dst,
                ..
            } => (vec![q, k_cache, weights], vec![dst]),
            Op::GatedAct { gate, up, dst, .. } => (vec![gate, up], vec![dst]),
            Op::GatedActFused { gu, dst, .. } => (vec![gu], vec![dst]),
            Op::Add { a, b, dst, .. } => (vec![a, b], vec![dst]),
            Op::AddBias { x, bias, dst, .. } => (vec![x, bias], vec![dst]),
            Op::Scale { x, dst, .. } => (vec![x], vec![dst]),
            Op::MulVec { x, vec: v, dst, .. } => (vec![x, v], vec![dst]),
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
                dst,
                ..
            } => {
                let mut r = vec![x, router_x, router, gate_exps, up_exps, down_exps];
                r.extend(down_scale);
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
        };
        assert_eq!(rope(None).io(), (vec![t(0), t(1)], vec![t(0)]));
        assert_eq!(rope(Some(t(2))).io(), (vec![t(0), t(1), t(2)], vec![t(0)]));
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
