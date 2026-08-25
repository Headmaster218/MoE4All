# DeltaNet 与 KDA Recurrent Kernels

[首页](../README.md) / [Kernels](README.md) / DeltaNet & KDA

## 两者不能混为一个算子

Qwen3.5/3.6 使用 gated DeltaNet；Ling 3.0 Flash 使用 KDA。它们都是每层维护 recurrent
state 的 linear-attention 类 mixer，但 projection、gate、decay 和 state update 语义不同。
本 fork 为两者保留不同 graph op 和 Vulkan shader。

## Qwen gated DeltaNet

### Strided Decode

早期 Q/K/V 等子块通过 `CopyStrided` 拆成多个 temporary。`447cd50` 让 Decode kernel 直接
从 packed convout 以 offset/stride 读取，消除 supported path 的中间 copy dispatch。

端到端深场景收益约 0～2%，不是 headline，但它：

- 减少每层小 dispatch；
- 降低 temporary allocation/recording；
- 为后续 recorder reuse 和固定气泡优化提供更干净链路；
- 保留 unsupported shape fallback。

### 状态生命周期

DeltaNet state 与 KV 一样属于 session persistent state，但不是 KV tensor。empty cache、
新 session 或 prefix reset 时必须同时重置。`1accc8c` 修复这一点，避免 122B 长输出退化为
重复符号/异常序列。

### Attention layer pattern

Qwen metadata 给出 `full_attn_interval`。Attention layer 更新 KV，其他层更新 DeltaNet
state。planner 根据真实 pattern 分别计算 KV/persistent bytes，不把所有层都按 Attention
KV 或所有层都按 recurrent state 估算。

## Ling KDA

输入：

- `qkv`: packed `[rows, 3*n_head*head_dim]`；
- `forget`: 每 channel forgetting logits；
- `beta`: 每 head update gate；
- `a = exp(A_log)`；
- `dt_bias`；
- state `S[h, k, v]`。

每 token/head：

```text
q, k = l2_normalize(q, k)
q = q / sqrt(head_dim)
decay[k] = exp(lower_bound * sigmoid(a[h] * (forget[k] + dt_bias[k])))
prediction = kᵀ S
delta = (v - prediction) * sigmoid(beta[h])
S = decay ⊙ S + outer(k, delta)
out = qᵀ S
```

CPU reference 和 Vulkan `kda.comp` 同时实现。CPU 不是性能目标，而是 parity 与边界语义的
oracle。

## Headwise gate 小算子

Ling 还需要 headwise sigmoid multiply 等 elementwise 图节点。对这类“很小但很多”的算子，
优化方向通常是融合和减少 dispatch，而非单个 shader 吞吐。DeepSeek HyperConnection 也
展示了同样规律：拆成更多 dispatch 即使每个更并行，端到端也可能更慢。

## Prefill 与 Decode

- Decode：rows=1，state update latency 和 dispatch 固定成本敏感；strided/fused path 重要。
- Prefill：rows 大，按 chunk 扫描 token；要保证 state 的因果顺序，不能把 token 维任意
  GEMM 化。
- phase 切换只改变 Expert cache layout，不可清掉 recurrent state，除非语义上开始新
  session。

## 显存 accounting

KDA state bytes 来自 `n_head * head_dim * head_dim`；Qwen DeltaNet 则由 value heads、K/V
state dims 和 conv history 共同决定。`persistent-state sizing` 与实际 allocation 共用 helper，
避免 GUI/plan 只认识 KV。

## 仍可优化的方向

- 将 projection/conv/gate/KDA 的更多小 dispatch 融合，但必须用 per-op trace 证明固定成本；
- 为特定 rows/head_dim 做 subgroup specialization；
- Prefill chunk scan 的共享内存与 occupancy；
- 将 state load/store 与相邻 norm/residual 合并。

这些方向没有在当前阶段给出未经实测的收益数字。

---

[Kernels](README.md) · [Ling](../models/ling3-flash.md) ·
[Prefill/Decode](../architecture/prefill-decode.md)
