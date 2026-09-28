---
kind: change-history
status: historical
scope: qwen35-mtp
preserved_in_tag: release-0.9.0
---

# MTP（多 token 预测）推测解码：qwen35 单头

> 历史实现与方案记录。Qwen3.5/3.6 的旧单头 MTP 当前停用；0.9.0 可用的 Qwen3.8 路径见[四 token MTP](../../../architecture/runtime/qwen38-mtp.md)。

问题 #33。参考：llama.cpp master（`--spec-type draft-mtp`，于 2026-05-16 从 PR #22673 合并）：
`common/speculative.cpp` 的 `common_speculative_impl_draft_mtp`（驱动程序）及 `src/models/qwen35.cpp` 的
`graph_mtp`（头图）。模型：`unsloth/Qwen3.5-4B-MTP-GGUF:UD-Q4_K_XL`。`nextn.*` 头张量嵌入主 GGUF 中（qwen35 没有同级文件；`--mtp`/`mtp-*.gguf` 同级下载流程用于其他架构系列）。llama.cpp 报告约 1.5-2 倍生成加速。

> **当前状态：Qwen3.5/3.6 的 MTP speculative decode 已停用。** `infr_llama::mtp::mtp_enabled()` 是这组旧单头实现的总开关，当前返回 `false`；设置 `INFR_MTP=1` 不会启用该路径，而是告警后使用普通 Decode。下面的实现细节、公式和测试说明作为历史设计记录，不代表当前可用能力。Qwen3.8 使用独立的 Qwen4 MTP runtime，不受这个总开关控制，见 [Qwen3.8 MTP](../../../architecture/runtime/qwen38-mtp.md)。

## 头的精确定义（qwen35：一个 MTP 层，`n_layer_nextn = 1`）

张量位于 `blk.{n_layer}.nextn.*`（主干**之后**的层索引），同一索引上还有一整套标准 qwen35 注意力层张量：

- `nextn.eh_proj [2*ne, ne]`, `nextn.enorm [ne]`, `nextn.hnorm [ne]`
- `nextn.embed_tokens [ne, vocab]`（可选：回退到主 tok_embd）
- `nextn.shared_head_head [ne, vocab]` + `nextn.shared_head_norm [ne]`（可选：回退到主 output/output_norm）
- `attn_norm/attn_q (interleaved q+gate!)/attn_k/attn_v/attn_q_norm/attn_k_norm/ attn_output/attn_post_norm/ffn_gate/ffn_up/ffn_down`
  ：infr 的统一运行器已为 qwen35 全注意力层执行的**精确**层形状（按头交错的 q/gate 切分、sigmoid 输出门、qk-norm、单位置 m-rope 分段 ≡ NEOX、SwiGLU）。

单个草稿行的前向过程 `(token t_{p+1}, target hidden h_p, position p+1)`：

```
e = rmsnorm(embed(t_{p+1}), enorm)        # embed from nextn.embed_tokens (or main)
h = rmsnorm(h_p, hnorm)                   # h_p = target's POST-output_norm hidden at p
x = eh_proj @ concat([e; h])              # [2ne] -> [ne]
x = qwen35_attention_layer(x, pos=p+1)    # own KV cache; standard causal attention
h_mtp = rmsnorm(x, shared_head_norm || output_norm)   # ALSO fed back when chaining drafts
logits = (shared_head_head || lm_head) @ h_mtp
```

**`h_p` 是目标模型在 `output_norm` **之后**的隐藏状态**：恰好是运行器已具体化的 lm_head 输入（参考实现中的 `res->t_h_nextn`；`llama_get_embeddings_nextn` 暂存 API 按批处理行读取它）。

## 驱动流程（单头模式：`n_mtp_layers==1`，不共享 KV，也不串接多个头）

Two hooks around the ordinary target decode:

1. **`process(batch)`：追赶目标模型状态；每次目标模型 Prefill/Decode ubatch 后都执行。**
   将同一批 token 再送入 MTP 层，但把 h 输入向右错开一位（`embd[i] = h_tgt[i-1]`，
   `embd[first] = pending_h`，其中 `pending_h` 来自上一批），确保 MTP 自己的 KV 覆盖所有已
   提交的位置。之后保存 `pending_h = h_tgt[last]`。（因此 MTP 需要目标模型每一行 prompt/verify
   的 h，而不只是采样行。）
2. **`draft(id_last, n_past)`：**在 `pos = n_past` 处输入 `(id_last, pending_h)`，对 MTP
   logits 做 greedy sampling。循环时只追加新 token（MTP KV 随之增长），并将上一 draft 行中
   MTP 头自身的 `h_mtp` 作为下一次的 `h`，由 MTP 头在 draft 阶段自行串接。若最高概率小于
   `p_min`（只保留高置信度草稿），或已生成 `n_max` 个 token，则停止（llama.cpp 示例：
   `--spec-draft-n-max 6`）。

验证阶段使用常规 spec verify：目标模型一次性批量前向处理整段 draft，并接受最长匹配前缀。
`infr` 已有这条路径（`spec_accept` 多行 verify）。随后对 verify batch 调用 `process()`，即可
自然地将 MTP KV 重新同步到实际接受的 token；draft 区域对应的 KV 行会在相同位置直接覆盖。

## infr 接入计划

- **阶段 1：提取 hidden state 并加载权重。**将 `{arch}.nextn_predict_layers` 和
  `blk.{n_layer}.nextn.*`/额外层张量解析到 Config/weights（仅 qwen35，并像参照实现一样强制
  `n_layer_nextn==1`）；为 `generate_dense_backend` 增加可选输出，导出 `output_norm` 之后的
  hidden rows（类似 `logits_out` 的 hook，提前一个 op，大小为 rows × ne 个 f32）。验收：张量可
  正确加载；同一次前向中 `lm_head(h_row) == logits_row`。
- **阶段 2：MTP 头前向与 draft 循环。**通过统一 runner 将 MTP 层作为单层 graph 执行（独立的
  单层 KV buffers；原样复用 qwen35 attention 生成逻辑），并在引擎层实现追赶、`pending_h` 与
  draft 循环（`p_min`/`n_max`）。支持 CPU + Vulkan。验收：固定 prompt、greedy 条件下，draft
  token/prob 与 llama.cpp 参照实现一致，且 CPU==Vulkan。
- **阶段 3：接入 spec 与用户入口。**让 run/serve 使用已有 verify 机制；维持 spec 与仅目标模型
  greedy 输出等价这一既有测试保证；加入 INFR 开关，并通过 benchmark 比较接受率和 tok/s，补齐
  文档并更新本文状态。

后续方向：Gemma4 mem-shared 模式（不单独分配 KV，使用一个 graph）、qwen35moe MTP 头（架构本身
已实现，但 MTP 头尚未接入）以及 chained-head 模型（step35）。

## Oracle commands (llama.cpp master, Vulkan build at

`~/Projects/mxaddict/llama.cpp/build/bin`)

```bash
llama-cli -m Qwen3.5-4B-MTP-UD-Q4_K_XL.gguf -ngl 99 --spec-type draft-mtp \
  --spec-draft-n-max 6 -p "..." -n 64 --temp 0   # MTP on
llama-cli -m ... -n 64 --temp 0                   # baseline, same binary
```

数据采集于 2026-07-05（CPU 构建；机器上没有 Vulkan headers，因此重点看相对提升）：提示词为
“What is the capital of France?”
`-n 48 --temp 0 --single-turn`:

| mode                | generation |
| ------------------- | ---------- |
| baseline (no spec)  | 20.5 t/s   |
| `draft-mtp` n_max=6 | 41.0 t/s   |

**生成速度提升 2.0 倍，输出逐字节一致**（两次运行的 diff 只有加载动画帧不同）。参照实现满足
spec ≡ target-greedy 不变量，这也是 infr 实现必须达到的标准。

## Confirmed GGUF facts (Qwen3.5-4B-MTP UD-Q4_K_XL dump)

- `qwen35.block_count = 33` **INCLUDES the MTP layer** (32 trunk + 1 head at
  `blk.32`); `qwen35.nextn_predict_layers = 1`. **infr's Config today would
  treat blk.32 as a DeltaNet layer** ((32+1)%4 ≠ 0) and fail on missing ssm
  tensors — Phase 1 must set trunk
  `n_layer = block_count − nextn_predict_layers` and stash the head layer.
- blk.32 = full qwen35 attention layer (interleaved q+gate `attn_q [2560,8192]`,
  4 kv × hd 256, q/k norms, `attn_output [4096,2560]`, post_attention_norm,
  SwiGLU ffn 9216) + `nextn.eh_proj [5120,2560] Q8_0`,
  `nextn.enorm/hnorm/ shared_head_norm [2560] F32`. NO `nextn.embed_tokens` /
  `shared_head_head` → main tok_embd + tied lm_head fallbacks are the live path.
- 4B trunk: ne=2560, full-attn every 4th layer (3,7,…,31), DeltaNet elsewhere
  (ssm inner 4096, ts_rank 32 — larger than the 0.8B but same shape family).

---

## 停用原因与范围

**Qwen3.5/3.6 MTP self-speculative decode is DISABLED** (`infr_llama::mtp::mtp_enabled` is
the single kill-switch, and carries the full rationale). `INFR_MTP=1` is ignored
with a warning, and the MTP-head GGUFs (Qwen3.5-\*-MTP) run the **ordinary**
decode path — their `nextn` tensors are simply unused, which is harmless. Those
models are otherwise fully supported; only the speculative path is off.

Why: MTP's contract was that its output is **token-identical to non-speculative
greedy** — a pure speedup, not a quality trade. That no longer holds. The
int8-activation decode kernels every fast dtype now uses carry small per-token
rounding noise, and MTP's verify batch and the plain-decode chain it must match
are computed at **different sequence positions with different KV state**. The
same noise plain decode absorbs harmlessly is enough to flip a close-margin
greedy argmax between the two streams, so `mtp_spec_matches_target_only_greedy`
fails. Notably this is **not** a bit-identity bug (`mmv_row1_bit_identical`
passes — decode and verify share one kernel) and **not** an accuracy cliff (all
13 `gpu_seam_matches_cpu_*` pass; output stays coherent).

That guarantee was holding the rest of the engine hostage: it blocked Q6_K's
int8 decode tier (+10% decode, +34% prefill) on a speculative path that was
already our slowest row (0.59–0.78× vs llama.cpp). So MTP is parked and the
kernel wins ship. The identity test is `#[ignore]`d, **not weakened** — the
assertion is correct; re-enabling MTP means making it pass again, which needs an
accuracy mitigation (e.g. re-verify in f32 when the top-2 logit margin is
tight), not faster kernels.
