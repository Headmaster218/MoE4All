---
kind: change-history
status: historical
period: 2026-08-15..2026-08-25
baseline: upstream/main..311ed4c
evidence_level: commit-and-historical-measurement
---

# 2026 年 8 月运行时架构演化

本页按决策变化整理早期派生版本的 89 个提交，而不是描述 `release-0.9.0` 的全部现状。当前实现以[系统总览](../../architecture/system-overview.md)和[资源生命周期](../../architecture/memory/runtime-resource-lifecycle.md)为准。合并基点为 `d7f320e7b8936fd6e1860115c5dd579c4572a27f`，本段终点为 `311ed4c`。实验数值只对应各自的历史条件。

| 日期 | 当时引入的机制 | 随后修正的假设 |
|---|---|---|
| 8 月 15 日 | Windows native、可用 RAM 探测、可选 Pager profiling、O(1) LRU、合成上下文深度 | 先使 100K-250K 深度 A/B 可重复，合成深度不等于真实长 Prefill |
| 8 月 17-18 日 | hd256 FlashAttention、DeltaNet、量化专家边界、Q8 KV Decode、recorder 复用、多槽上传 | Q8 节省约 46.9% KV bytes，但当时 200K Attention 约慢 29%；内核和 Pager 要分开归因。Q8 Decode 系列在该阶段 200K 累计约 +25.7% |
| 8 月 19 日 | layer-major Host Store、CPU→ReBAR、Prefill 整层 A/B lane、Down 重叠 | Prefill 顺序过层，Decode 按路由稀疏取专家，不能强行共用 expert 级 LRU 粒度 |
| 8 月 20 日 | 从六个 `(role,size)` 池转为按尺寸全局池、异步 Prefill ring、总 VRAM budget | 固定 role quota 的 8:7/成对淘汰方案退化；Prefill 的容量上限与 Decode 不同。详见[深度矩阵](../benchmarks/2026-08-20-qwen36-apex-matrix.md) |
| 8 月 21-22 日 | 原生 Embedding、共享 Vulkan arena、persistent/runtime 统一记账 | 初版 Embedding 权重仍长期驻留；后改成按请求/空闲期释放，把显存还给 Expert filler |
| 8 月 23 日 | RAM/SSD 第三级、Ling/DeepSeek V4、批量 Host 提升 | 初版排他式 RAM/SSD 后改为 inclusive shadow；VRAM→RAM 回写约 44 MB/s，不适合作 miss 热路径 |
| 8 月 24 日 | full-RAM 与 bounded 两路 host backing、弹性 arena、有序 trace、共享专家融合 | Qwen 122B 的 recurrent state 必须随 KV 清空；复杂 UG→D 按 tier 分支经微基准否决，见[122B 追踪](../benchmarks/2026-08-24-qwen35-122b-cold-trace.md)与[微基准](../benchmarks/2026-08-25-moe-pager-microbench.md) |
| 8 月 25 日 | `VK_EXT_external_memory_host` 原地导入和按池比例分配 import 额度 | 驱动只导入约 29 GiB，未导入尾段继续 CPU push；122B 热 tg256 从 19.2 到 23.2 tok/s，不替代冷长追踪 |

## 关键设计转折

1. **同一 VRAM，两个访问形态。** Prefill 用整层 Host Store、常驻层和异步 layer ring；Decode 用按尺寸全局 expert LRU。不是两份完整专家缓存。批次中的已访问专家受 epoch 保护，不能被后续 Down miss 立即淘汰。
2. **从专家缓存预算到总显存预算。** 固定权重、KV、运行时峰值与安全 margin 进入统一规划；否则 Prefill 容易峰值超额，而 Decode 又浪费空闲 runtime reserve。
3. **从各自 malloc 到弹性 arena。** LLM、Embedding、Vision、临时 runtime 与 Expert 竞争同一分片逻辑空间。大连续 claim 采用 cold-window eviction 和高地址分配，而非昂贵的全量显存压缩整理。
   当时 18.79 GB arena 的初始 tail 仅 720896 bytes（0.00384%），说明不能只凭整 GiB 的预算口径判断实际 slot packing。
4. **从 full-RAM 到有界 inclusive RAM/SSD。** 35B 的 23.57 GB 专家载荷可全进 RAM；Ling、DeepSeek V4、122B 则要求按需 SSD miss 和 RAM shadow。SSD/GGUF 是只读最终来源，淘汰无需 GPU→RAM 回写。
5. **从 CPU push 到可回退 Host DMA。** 单线程 CPU push 约 8.8 GB/s，批量/多线程约 14-19 GB/s；足够大负载的 Vulkan copy 微基准约 23-25 GiB/s。真实单专家提升还受固定提交成本和队列依赖限制。

## 提交索引

以下按阶段列出原 89 个增量提交。查行为应看具体提交及 0.9.0 源码，不能把阶段中间形态当作最终设计。

| 阶段 | 提交 |
|---|---|
| Windows 与测量（6） | `d9bd5a9` `1038b5d` `ebf5b79` `8c49710` `898ff91` `16fbee2` |
| Qwen 深上下文（9） | `dbc51fe` `9bef28d` `95b8ffa` `447cd50` `276d9c8` `0ffdefd` `3c9523a` `46c0b88` `a73d43a` |
| Q8 与 Pager 流水线（12） | `ff69e83` `5a33e58` `02e0bfb` `d72f60f` `e6b6137` `633638b` `84fb844` `0b37574` `15eb5f7` `b8cbc52` `6afab3a` `3651e29` |
| Host Store 与 MoE（9） | `ce97e4f` `354c0c3` `e603efe` `dd43c45` `2d57e7b` `132b824` `5a1faeb` `301a620` `951ffa3` |
| 全局池与预算（7） | `d7be656` `637f23f` `c90577d` `c517290` `0a1aa30` `3722b8a` `742c500` |
| Embedding 起点（5） | `949d6b2` `9a53907` `bb2cb95` `5ec490b` `5b0c4de` |
| 原生 Embedding 与共享 VRAM（15） | `b88c3cd` `4da3e16` `3c44a9d` `e7fec2f` `deb24c3` `b16222d` `a90d285` `fd7e3d6` `4bb8b3b` `1ef73d6` `5aeb58b` `02250a3` `c944e04` `bc4fc93` `35da757` |
| 三级缓存与新模型（14） | `ebcbd3b` `d3f1af5` `2f65486` `7d8b0aa` `9f4b6fe` `b007730` `4343c03` `99e6e40` `082e1eb` `f3af338` `074e388` `f6ca35c` `495fc9a` `f2fb30f` |
| 弹性池、追踪与调度（8） | `7935cf8` `66b66ab` `31d9883` `df428ad` `7eb9f0d` `1accc8c` `3d38515` `0c86ce7` |
| Host DMA 与阶段记录（4） | `2bd5469` `497460a` `81724ae` `311ed4c` |

早期反例和复测门槛见[否决实验](2026-08-rejected-experiments.md)，后续 Qwen3.8 与 0.9.0 的变化各见对应 campaign 和 release changelog。
