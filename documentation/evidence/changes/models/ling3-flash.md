---
kind: change-history
status: historical
scope: ling3-flash-and-kda
period: 2026-08-23..2026-08-25
evidence_level: commit-and-historical-sample
---

# Ling 3.0 Flash 与 KDA 接入

Ling 3.0 Flash 是早期派生版本中验证另一种 hybrid recurrent/MLA 模型复用 Vulkan graph、专家 Pager 和 bounded RAM/SSD 路径的案例。`d3f1af5` 阶段接入 `bailingmoe3` 配置、权重布局、CPU reference、Vulkan 算子和 runner state。阶段模型为 42 层 hybrid trunk、前两层 dense，其余含 grouped routed MoE 与 shared expert；KDA/MLA 层由 metadata 指定，不能套用 Qwen 的固定层间隔。公开 GGUF 曾带有陈旧 NextN metadata 却没有对应权重，加载时以真实 NextN tensor 为准，不把普通 trunk 误认作 MTP。

## 算子与状态边界

Ling KDA 与 Qwen gated DeltaNet 都保留跨 token 的 state，但 projection、gate、decay 和更新顺序不同，不是同一个 graph op。历史 KDA 每个 token/head 的核心关系是：

```text
q, k = l2_normalize(q, k)
q = q / sqrt(head_dim)
decay[k] = exp(lower_bound * sigmoid(exp(A_log[h]) * (forget[k] + dt_bias[k])))
prediction = k^T S
delta = (v - prediction) * sigmoid(beta[h])
S = decay * S + outer(k, delta)
out = q^T S
```

CPU reference 与 Vulkan KDA 路径相互校验。KDA state 规模由 `n_head * kda_head_dim * kda_head_dim` 决定；它与 Attention KV 一样属于 session 持久状态，不能在普通 Prefill→Decode 切换时清空。MLA 层则不应预留普通完整 Attention score-matrix scratch：`9f4b6fe` 修正此估算，将空间归还给专家缓存。历史 Qwen DeltaNet 路径中，`447cd50` 允许 Decode 从 packed convout 以 stride/offset 直接读取，减少小 copy dispatch；此优化不等于 Ling KDA 的数学实现。混合模型清空 KV 时未同时重置 recurrent state 曾导致 122B 重复输出，见[冷追踪](../../benchmarks/2026-08-24-qwen35-122b-cold-trace.md)。

## 结果与限制

阶段连续 Decode 约 36 tok/s 只是一条缺完整命令与重复日志的历史样本，不能扩写成 0.9.0 多上下文 benchmark。Ling 尚缺 Qwen 35B/122B 那样的完整 route trace 与矩阵；当时也没有 Ling NextN/MTP 图，Prefill 的 SSD→RAM lookahead 还不是完整异步流水线。后来优先复用的成果是 bounded inclusive RAM/SSD、full-RAM 分流、shared/resident UGD、ordered trace 和 Host DMA，而非宣称单个 KDA tile 已解决整个性能问题。当前支持范围见[模型能力矩阵](../../../reference/model-capabilities.md)。
