# Qwen3.5 / Qwen3.6 (`qwen35`) support

Both `unsloth/Qwen3.5-*` and `unsloth/Qwen3.6-*` GGUFs declare
`general.architecture = qwen35`: a hybrid of **gated DeltaNet linear-attention**
layers and **gated full-attention** layers, with SwiGLU (dense for ≤1B; MoE for
larger) and sectioned RoPE. Our engine is a pure transformer, so this is a
net-new architecture family. (Qwen3.6 has no ≤1B variant; smallest is 27B. Start
with `unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` — already pulled.)

> **Not Qwen3-Next.** llama.cpp's `qwen3next` is a _sibling_ arch in the same
> DeltaNet family, but its V heads broadcast differently — Qwen3-Next in blocks
> (`[k0_v0, k0_v1, k1_v2, k1_v3]`), Qwen3.5 interleaved
> (`[k0_v0, k1_v1, k0_v2, k1_v3]`, the `h % n_khead` tiling infr implements). A
> `qwen3next` GGUF through this code would be silently wrong; the arch gate
> rejects it on purpose. `qwen35moe` is likewise its own arch, unsupported until
> the expert FFN lands.

## 0.8B config (metadata)

- block_count=24, embedding_length=1024, context_length=262144
- full_attention_interval=4 → attention layers at i where (i+1)%4==0 → **layers
  3,7,11,15,19,23** (6 attention); the other **18 are linear (gated DeltaNet)**.
- ssm: conv_kernel=4, state_size=128, inner_size=2048, group_count=16,
  time_step_rank=16
- attention: head_count=8, head_count_kv=2, key_length=value_length=256
- rope: dimension_count=64, dimension_sections=[11,11,10,0], freq_base=1e7
- rms_eps=1e-6

### Derived dims (llama.cpp reuses SSM hparam fields)

- head_k_dim = state_size = 128 ; num_k_heads = group_count = 16 → key_dim =
  2048
- num_v_heads = time_step_rank = 16 ; d_inner = inner_size = 2048 → head_v_dim =
  128
- conv_channels = d_inner + 2*group_count*state_size = 2048 + 4096 = **6144** (=
  attn_qkv out)

## Linear layer (gated DeltaNet) — per layer tensors

- attn_norm[1024] (input RMSNorm)
- attn_qkv[1024,6144] → q(2048)+k(2048)+v(2048) (NO z here)
- attn_gate[1024,2048] → z (the output gate)
- ssm_conv1d[4,6144] depthwise causal conv over the 6144 qkv channels
- ssm_beta[1024,16] → b ; ssm_alpha[1024,16] → a (per v-head)
- ssm_a[16] = −exp(A_log) ; ssm_dt.bias[16]
- ssm_norm[128] (RMSNorm over head_v_dim)
- ssm_out[2048,1024]
- ffn_norm? (post_attention_norm[1024]) + ffn_gate/up[1024,3584] +
  ffn_down[3584,1024]

### Forward (per token, decode/recurrent form — matches HF `fused_recurrent_gated_delta_rule`)

1. x = rmsnorm(hidden, attn_norm)
2. qkv = x @ attn_qkv → [6144]; z = x @ attn_gate → [2048]
3. qkv = silu(causal_conv1d(qkv, ssm_conv1d, k=4)) (per-channel depthwise; state
   = last k-1 cols)
4. split qkv → q[16×128], k[16×128], v[16×128]
5. b = x@ssm_beta[16]; a = x@ssm_alpha[16]; beta = sigmoid(b); g = ssm_a \*
   softplus(a + ssm_dt.bias) (g ≤ 0, per v-head)
6. per head h (q,k L2-normed, scale=1/sqrt(128)): S = S _ exp(g_h) (S:
   [head_k_dim=128, head_v_dim=128]) kv = kᵀS ; delta = (v − kv) _ beta_h ; S +=
   k ⊗ delta ; out_h = qᵀS (out [128])
7. out = silu_gated_rmsnorm(out, ssm_norm, gate=z): rmsnorm(out)\*silu(z) (per
   head_v_dim)
8. y = out @ ssm_out → [1024]; hidden += y
9. h2 = rmsnorm(hidden, post_attention_norm); hidden += swiglu_ffn(h2) (note:
   confirm whether attn_norm/ffn ordering uses pre+post norms — see graph
   build_layer)

## Attention layer — per layer tensors

- attn_norm[1024]; attn_q[1024,4096]=q(8×256)+gate(2048);
  attn_k[1024,512]=2×256; attn_v[1024,512]
- attn_q_norm[256], attn_k_norm[256] (per-head RMSNorm); attn_output[2048,1024]
- ffn\_\* same as above

### Forward: standard GQA softmax attention, head_dim=256, 8 q / 2 kv heads, sectioned RoPE

(dimension_sections [11,11,10,0]); output gated by sigmoid(gate) before
attn_output.

## State caches (decode)

- conv state: per linear layer, [conv_channels=6144, k-1=3] rolling window.
- recurrent state: per linear layer, [num_v_heads=16, head_k_dim=128,
  head_v_dim=128] f32.
- KV cache: only for the 6 attention layers (head_dim 256).

## Status

- **UNIFIED (phases 1–4 of the seam merge, done): qwen35 runs on the SHARED
  transformer runner** (`seam::generate_dense_backend`) via `MixerW::DeltaNet` —
  same run/serve/bench paths, SlotPool multi-slot serve, and every shared-path
  optimization (mrow GEMV, small-m attention tiers). FASTER than the old bespoke
  seam it replaced: pp512 12.2k→14.7k t/s, tg64@d4096 228→243 on the 0.8B.
  DeltaNet layers keep conv/S state in the session's per-layer buffers (fixed
  f32, no KV); the session reuses state only on EXACT prompt extension
  (recurrent state can't rewind — anything else zero-resets and re-prefills).
  Record-once decode replay is gated off for qwen35 (the tape isn't audited for
  recurrent-state bindings; static per-token decode matches the unified CPU
  oracle).
- Phase 4 (issue #30): the old hand-written qwen35-only seam that used to live
  in `qwen35.rs` (proven token-identical to the unified path during the cutover,
  then reachable only via a temporary escape hatch for one release) has been
  DELETED — the unified path is the only one now. `qwen35.rs` today holds just
  the raw metadata parse (`Cfg`), arch detection (`is_qwen35`), and the
  chat-template renderer.
- **CPU reference: DONE & correct** (`Config::from_gguf`'s qwen35 fields +
  `MixerW::DeltaNet` in `crates/infr-llama/src/seam/`). The one real bug was the
  attention q/gate split: `attn_q` packs query+gate INTERLEAVED PER HEAD
  `[h0 q | h0 gate | h1 q | h1 gate | …]`, not two contiguous blocks.

## GPU situation (corrected)

ggml's **Vulkan** backend does NOT implement the SSM ops (gated*delta_net,
ssm_conv), but ggml's scheduler **falls back to CPU per-unsupported-op**, so
`llama-cli -ngl 99 -dev Vulkan0` RUNS qwen35 as a CPU+GPU **hybrid** (not an
error). (ggml's CUDA backend \_does* implement them → full GPU on NVIDIA.)
Benchmark (Qwen3.5-0.8B-Q4, 7900 XTX, llama.cpp):

| test  | pure CPU `-ngl 0` | Vulkan hybrid `-ngl 99` | speedup |
| ----- | ----------------- | ----------------------- | ------- |
| pp512 | 4208 t/s          | 18054 t/s               | 4.3×    |
| tg64  | 33.5 t/s          | 387 t/s                 | 11.5×   |

The SSM recurrence on CPU is NOT the bottleneck once matmuls are on GPU.

## Build plan (revised)

1. ✅ CPU reference forward (correct vs llama.cpp / HF) — locks all math above.
2. ✅ GGUF loader (arch=qwen35, both layer types, dequant) + `infr run`
   dispatch.
3. **Hybrid GPU path (recommended next):** SSM recurrence + causal conv on CPU;
   run all matmuls/FFN/ gated-attention through our Vulkan backend (head_dim=256
   attention needs the non-flash path or a hd-generalized kernel). Per the
   benchmark this captures ~most of the GPU win with NO SSM Vulkan kernels.
   Needs: keep weights on GPU (quantized), CPU↔GPU handoff per linear layer for
   the SSM step.
4. (Optional, large) Full-GPU SSM Vulkan kernels: gated-delta scan + ssm_conv +
   silu-gated rmsnorm, to remove the CPU handoff. Bigger effort; only worth it
   if the hybrid handoff dominates.
5. CPU-reference speedups regardless: rayon-parallel `matvec`, then quantized
   matvec.

## Vision (image input)

`qwen35` and `qwen35moe` accept images when a matching `--mmproj` (projector type
`qwen3vl_merger`) is supplied alongside the text GGUF. This works on Vulkan and
CPU; Metal has no `Op::QkNormMrope` and will reject an image turn.

Image spans use **M-RoPE** (llama.cpp `LLAMA_ROPE_TYPE_IMROPE`): each token gets a
4-plane `(T, H, W, E)` position instead of a single index, and the per-head
QK-norm is fused with the rotation in `Op::QkNormMrope`
(`crates/infr-vulkan/shaders/qk_norm_rope_mrope.comp`). The plane a frequency pair
rotates by comes from `rope.dimension_sections` — `[11,11,10,0]` for the qwen35
family, i.e. 11 T pairs, 11 H, 10 W, none for E, out of `rope_dim/2 = 32`.

Two properties make this safe for the hybrid skeleton:

- **The DeltaNet layers ignore positions entirely.** `Op::Conv1dSilu` and
  `Op::DeltaNet` carry no position operand — token order reaches them only through
  the causal conv's rolling state and the append-only recurrent state. M-RoPE is
  consumed only by the full-attention layers, i.e. those where
  `(il+1) % full_attention_interval == 0`.
- **Text-only requests are unaffected.** A row whose T, H and W are equal is
  exactly representable by 1-D rope, and `mrope_rows_are_plain_rope` detects that
  and emits the ordinary `Op::QkNormRope` instead. So generated tokens after an
  image — and every request without an image — take the original kernel.

Caveats: batch-cooperative visual decode (the `--parallel N>1` scheduler) remains
`qwen4exp`-only, so a `qwen35moe` image turn runs on the single-slot runner. The
recurrent state cannot rewind, so an image prompt must arrive as part of one
prefill (zero state) or an exact extension.

## References (read for math, reimplement — both MIT)

- HF transformers `modeling_qwen3_next` (recurrence: fused_recurrent +
  chunk_gated_delta_rule)
- llama.cpp `src/models/qwen3next.cpp` (GGUF tensor→role mapping, ggml_ssm_conv
  usage)
- llama.cpp `ggml_mrope_cache_init` (`is_imrope` branch) for the M-RoPE plane
  cascade above
