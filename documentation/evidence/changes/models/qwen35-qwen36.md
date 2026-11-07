---
kind: change-history
status: historical
scope: qwen35-qwen36
preserved_in_tag: release-0.9.0
---

# Qwen3.5 / Qwen3.6（`qwen35`）支持

> 历史架构开发记录，包含早期未实现阶段的原话；0.9.0 的支持范围见[模型能力矩阵](../../../reference/model-capabilities.md)。

早期小型 dense GGUF 使用 `general.architecture = qwen35`；MoE GGUF 使用独立的 `qwen35moe`。
该家族由 **gated DeltaNet linear-attention** 层和 **gated
full-attention** 层混合组成，并使用 SwiGLU（≤1B 为 dense，更大的模型为 MoE）和分段 RoPE。
我们的引擎是纯 transformer，因此这是一个全新的架构系列。（Qwen3.6 没有 ≤1B 版本，最小为
27B。可从 `unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-Q4_K_M.gguf` 开始；该模型已拉取。）

> **不要与 Qwen3-Next 混淆。** llama.cpp 的 `qwen3next` 是同属 DeltaNet 家族的一个近缘架构，
> 但 V head 的广播方式不同：Qwen3-Next 按 block 排列（`[k0_v0, k0_v1, k1_v2, k1_v3]`），
> Qwen3.5 则交错排列（`[k0_v0, k1_v1, k0_v2, k1_v3]`，对应 infr 实现的 `h % n_khead`
> tiling）。若用这段代码加载 `qwen3next` GGUF，结果会静默错误，因此架构检查会有意拒绝它。
> `qwen35moe` 是独立架构。早期 expert FFN 尚未接入时不支持；这句话不再描述 0.9.0 的能力。

## 0.8B 配置（元数据）

- block_count=24, embedding_length=1024, context_length=262144
- full_attention_interval=4 → attention layers at i where (i+1)%4==0 → **layers
  3,7,11,15,19,23** (6 attention); the other **18 are linear (gated DeltaNet)**.
- ssm: conv_kernel=4, state_size=128, inner_size=2048, group_count=16,
  time_step_rank=16
- attention: head_count=8, head_count_kv=2, key_length=value_length=256
- rope: dimension_count=64, dimension_sections=[11,11,10,0], freq_base=1e7
- rms_eps=1e-6

### 派生维度（llama.cpp 复用 SSM hparam 字段）

- head_k_dim = state_size = 128 ; num_k_heads = group_count = 16 → key_dim =
  2048
- num_v_heads = time_step_rank = 16 ; d_inner = inner_size = 2048 → head_v_dim =
  128
- conv_channels = d_inner + 2*group_count*state_size = 2048 + 4096 = **6144** (=
  attn_qkv out)

## 线性层（门控 DeltaNet）：每层张量

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

### 前向过程（每个 token，解码/循环形式：匹配 HF `fused_recurrent_gated_delta_rule`）

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

## 注意力层：每层张量

- attn_norm[1024]; attn_q[1024,4096]=q(8×256)+gate(2048);
  attn_k[1024,512]=2×256; attn_v[1024,512]
- attn_q_norm[256], attn_k_norm[256] (per-head RMSNorm); attn_output[2048,1024]
- ffn\_\* same as above

### 前向过程：标准 GQA softmax 注意力，head_dim=256，8 个 q / 2 个 kv 头，分段 RoPE

(dimension_sections [11,11,10,0]); output gated by sigmoid(gate) before
attn_output.

## 状态缓存（解码）

- conv state: per linear layer, [conv_channels=6144, k-1=3] rolling window.
- recurrent state: per linear layer, [num_v_heads=16, head_k_dim=128,
  head_v_dim=128] f32.
- KV cache: only for the 6 attention layers (head_dim 256).

## 状态

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

## GPU 情况（已修正）

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

## 构建计划（已修订）

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

## 参考资料（用于理解数学和重新实现：均为 MIT）

- HF transformers `modeling_qwen3_next` (recurrence: fused_recurrent +
  chunk_gated_delta_rule)
- llama.cpp `src/models/qwen3next.cpp` (GGUF tensor→role mapping, ggml_ssm_conv
  usage)
