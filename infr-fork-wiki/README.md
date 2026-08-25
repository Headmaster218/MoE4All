# INFR 大模型异构推理 Fork Wiki

> 这是一套独立于原 `infr` 仓库的工程 Wiki，只记录本 fork 在
> `upstream/main..HEAD` 范围内新增、修改、验证和放弃的工作。

## 先看结论

这个 fork 的主线不是单独写出一个更快的算子，而是把消费级 AMD GPU 上的超大 MoE
推理逐步改造成一个可测量、可分页、可跨 VRAM/RAM/SSD 运行的系统：

```text
Windows/RDNA3 可运行
        ↓
长上下文 Attention 与 DeltaNet 优化
        ↓
专家分页、全局缓存池与 Prefill/Decode 分治
        ↓
统一显存预算与弹性 VRAM
        ↓
Embedding 与 LLM 共用同一物理显存池
        ↓
RAM/SSD 三级缓存、路由 trace 与离线模拟
        ↓
Ling、DeepSeek V4、Qwen 122B 超大 MoE 验证
        ↓
Host RAM 原地 Vulkan DMA，122B Decode 达到 23.2 tok/s
```

代表性实测结果：

| 模型 / 工作负载 | 条件摘要 | 结果 |
|---|---|---:|
| Qwen3.6-35B-A3B Balanced，Decode，250K synthetic depth | Q8 KV，8 GiB expert-cache 设置 | **41.2 tok/s** |
| Qwen3.6-35B-A3B Balanced，Prefill 4096，250K depth | Q8 KV，8 GiB expert-cache 设置 | **477.9 tok/s** |
| Qwen3.6-35B-A3B Balanced，Prefill 4096，d0 | Q8 KV，8 GiB expert-cache 设置 | **2855.6 tok/s** |
| Ling 3.0 Flash | 历史连续 Decode 体验样本 | **约 36 tok/s** |
| DeepSeek V4 Flash | 12.67 GiB VRAM expert + 40 GiB bounded RAM/SSD | **约 4.3 tok/s** |
| Qwen3.5-122B-A10B Quality，tg256 | 13.97 GiB GPU expert arena，45 GiB RAM，Host DMA | **23.2 tok/s** |

这些数字不是跨模型排行榜。模型、量化、上下文、缓存预算、是否稳态、是否开启 profiler
都不同。使用前先阅读[基准方法与数字口径](reference/benchmark-method.md)。

## 按模型阅读

- [Qwen3.6-35B-A3B：完整优化主线](models/qwen36-35b.md)
- [Qwen3.5-122B-A10B：三级缓存、trace、shared fusion 与 Host DMA](models/qwen35-122b.md)
- [Ling 3.0 Flash：KDA/MLA 混合架构接入](models/ling3-flash.md)
- [DeepSeek V4 Flash：正确性接入、性能优化与收尾](models/deepseek-v4-flash.md)
- [Nomic Embedding：从受管 worker 到原生引擎和统一显存](models/nomic-embedding.md)
- [模型索引](models/README.md)

## 按技术阅读

| 方向 | 文档 |
|---|---|
| 架构总演化 | [从六池到统一弹性三级缓存](overview/architecture-evolution.md) |
| 总显存与 RAM 预算 | [预算规划与生命周期](architecture/memory-budget.md) |
| 统一 VRAM | [物理分片、逻辑连续、低高地址分配](architecture/unified-vram.md) |
| Expert Pager | [O(1) LRU、epoch 保护、全局槽位](architecture/expert-pager.md) |
| RAM/SSD | [full-RAM 与 bounded inclusive RAM/SSD](architecture/ram-ssd-cache.md) |
| RAM→VRAM | [ReBAR CPU push 与 Host DMA](architecture/host-dma.md) |
| Prefill / Decode | [两条执行链及其不同目标](architecture/prefill-decode.md) |
| U/G/D 与 shared | [MoE 调度、融合与实测决策](architecture/moe-scheduling.md) |
| Attention | [hd256、Q8 KV 与深上下文](kernels/attention-hd256-q8.md) |
| DeltaNet / KDA | [循环状态算子](kernels/deltanet-kda.md) |
| 量化 MoE | [Q5/Q6/IQ4/MXFP4 解码](kernels/quantized-moe.md) |
| DeepSeek 特化 | [FP8 KV、MXFP4 indexer、HyperConnection](kernels/deepseek-v4.md) |
| GUI / 服务 | [浏览器控制面与 worker 生命周期](product/browser-gui.md) |
| Embedding API | [原生接口、并发门与资源回收](product/embedding-api.md) |

## 按实验阅读

- [成功与失败实验总索引](experiments/README.md)
- [Qwen 35B 优化 campaign](experiments/qwen36-campaign.md)
- [缓存策略：六池、双池、比例淘汰与 plain LRU](experiments/cache-policy.md)
- [路由 trace、离线 replay 与模拟边界](experiments/trace-simulation.md)
- [被否决或暂缓的方向](experiments/rejected-experiments.md)

## 查数据与溯源

- [成果总表](overview/results.md)
- [完整时间线](overview/timeline.md)
- [Qwen 35B 性能矩阵](reference/qwen36-matrix.md)
- [Qwen 122B trace 与缓存数据](reference/qwen122-trace.md)
- [DeepSeek V4 缓存与模拟数据](reference/deepseek-v4-data.md)
- [MoE 计算、RAM/SSD 搬运微基准](reference/moe-schedule-microbench.md)
- [完整 fork commit map](reference/commit-map.md)
- [原始证据索引](reference/evidence-index.md)
- [术语表](reference/glossary.md)

## Wiki 边界

本 Wiki 不替上游补写完整手册。上游已有模型、通用 Vulkan/CPU/Metal backend、配置系统、
基础 runner 等，只在解释本 fork 的改动时作为背景出现。详细范围规则见
[项目范围与证据规则](overview/scope.md)。

---

[项目范围](overview/scope.md) · [架构演化](overview/architecture-evolution.md) ·
[时间线](overview/timeline.md) · [结果总表](overview/results.md)
