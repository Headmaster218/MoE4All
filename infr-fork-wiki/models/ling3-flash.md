# Ling 3.0 Flash

[首页](../README.md) / [模型](README.md) / Ling 3.0 Flash

## 在 fork 中的意义

Ling 3.0 Flash 是第一个证明“不同于 Qwen DeltaNet 的混合 recurrent/MLA 架构，也能复用
现有 Pager 与 Vulkan graph”的大模型。接入提交 `d3f1af5` 一次性覆盖配置解析、权重布局、
CPU reference、Vulkan kernel、runner 状态和 tokenizer 适配。

它也承担了第一个重要边界验证：expert payload 可以大于 VRAM，并在 bounded RAM/SSD 下
端到端生成，不要求完整 mmap/Host Store 常驻 RAM。

## 架构差异

- GGUF architecture：`bailingmoe3`。
- 42 层 hybrid trunk。
- 每层由 metadata 决定是 KDA 还是 MLA，而不是固定 `1:3` 的 Qwen模式。
- 前两个 dense lead layers，后续使用 grouped routed MoE 和 shared expert。
- KDA 是独立 recurrent operator；MLA/MoE 在 tensor layout 相同时复用 DeepSeek 路径。
- public GGUF 声明过 stale NextN metadata，但没有相应 tensor；加载器只在真实 NextN tensor
  出现时拒绝，避免把 trunk layer 误判成 MTP。

## 新增 KDA 算子

每 token/head：

1. Q/K 归一化，Q 乘 `1/sqrt(head_dim)`；
2. 由 `A_log`、forget、`dt_bias` 和 lower bound 形成 per-channel decay；
3. `delta = (v - kᵀS) * sigmoid(beta)`；
4. `S += outer(k, delta)`；
5. 输出 `qᵀS`。

CPU reference 与 Vulkan `kda.comp` 同时加入，避免只有 GPU 能跑却无可比正确性基线。
另外加入 headwise sigmoid multiply 等小算子，完成真实图拼接。

## 状态与显存规划

KDA recurrent state 的形状来自模型 metadata：

```text
n_head * kda_head_dim * kda_head_dim
```

MLA layer 则不应被当作普通 Attention score-matrix scratch。`9f4b6fe` 修正了显存 planner，
避免为 MLA 错留通用 Attention reserve，从而把空间还给 Expert cache。

## 性能状态

历史连续 Decode 体验约 **36 tok/s**。这是阶段运行记录，不是当前 Wiki 内可重新校验的
严格 A/B 日志，因此标记为“历史样本”。它证明模型规模本身不必然导致 DeepSeek V4 那样
的 4 tok/s；差异更多来自 active expert 字节量、量化 kernel 和 SSD miss 工作集。

Ling 的性能专项后来让位于 DeepSeek V4 和 Qwen 122B。对后续工作最有价值的不是继续抠
一个 KDA tile，而是继承：

- bounded RAM/SSD 基础设施；
- full-RAM 与 bounded 路径的明确分流；
- shared/resident UGD fusion；
- ordered route trace；
- Host DMA。

## 与 Qwen/DeepSeek 的关系

| 组件 | 来源/策略 |
|---|---|
| KDA | 本 fork 新算子，不能用 Qwen DeltaNet 冒充 |
| MLA | 复用匹配的 DeepSeek MLA 图与 kernel |
| grouped MoE | 复用 DeepSeek 风格 routing 语义 |
| Expert Pager | 使用通用按尺寸 global pool |
| RAM/SSD | 使用 bounded inclusive tier |
| shared expert | 可使用通用 mixed-quant fusion |

## 当前限制

- 没有为 Ling 的 NextN/MTP tensor 实现执行图；若未来 GGUF 真包含它，应显式适配。
- 缺少一份像 Qwen 35B/122B 那样完整的 route trace 和矩阵；36 tok/s 不应扩写成跨场景结论。
- Prefill 的 SSD→RAM lookahead 尚不是完整异步流水线。

---

[模型索引](README.md) · [DeltaNet/KDA](../kernels/deltanet-kda.md) ·
[RAM/SSD cache](../architecture/ram-ssd-cache.md)
