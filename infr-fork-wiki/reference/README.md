# Reference 与原始数据索引

[首页](../README.md) / Reference

| 页面 | 内容 |
|---|---|
| [Benchmark method](benchmark-method.md) | 数字口径、A/B、warmup、synthetic depth |
| [Qwen 35B matrix](qwen36-matrix.md) | Q8/F16 × depth × cache 的完整阶段矩阵 |
| [Qwen 122B trace](qwen122-trace.md) | 2K ordered trace、hit、SSD traffic、shared/DMA AB |
| [DeepSeek V4 data](deepseek-v4-data.md) | 最终 cache state 与容量模拟 |
| [MoE schedule microbench](moe-schedule-microbench.md) | 2～9 expert compute、1～8 expert transfer、split cost |
| [Commit map](commit-map.md) | 89 个 fork commits 的阶段映射 |
| [Evidence index](evidence-index.md) | 原 `infr` 中的日志/report/trace 路径与用途 |
| [Glossary](glossary.md) | 项目专用术语 |

Reference 保存“事实层”，架构/模型页负责解释。若两处摘要冲突，以更接近原始 artifact 的
Reference 页面为准，再修正文档。

---

[首页](../README.md) · [实验索引](../experiments/README.md)
