# 架构索引

[首页](../README.md) / 架构

本目录只描述这个 fork 新增或重构的系统机制。

| 层次 | 页面 | 核心问题 |
|---|---|---|
| 规划 | [Memory budget](memory-budget.md) | 如何从模型真实 layout 得到 VRAM/RAM 分配 |
| VRAM | [统一弹性 VRAM](unified-vram.md) | 多物理 shard 如何成为一个逻辑 arena |
| VRAM cache | [Expert Pager](expert-pager.md) | slot、LRU、epoch、淘汰与恢复 |
| Host/SSD | [RAM/SSD cache](ram-ssd-cache.md) | full-RAM 与 bounded inclusive 两条路径 |
| 传输 | [Host DMA](host-dma.md) | CPU push、Vulkan DMA、import ceiling 与 fallback |
| 执行 | [Prefill / Decode](prefill-decode.md) | 为什么两阶段必须不同调度 |
| MoE | [U/G/D 与 shared scheduling](moe-scheduling.md) | 哪些计算先做、哪些搬运合并 |

总演化过程见[架构怎样逐步演化](../overview/architecture-evolution.md)。

---

[首页](../README.md) · [模型索引](../models/README.md) · [实验索引](../experiments/README.md)
