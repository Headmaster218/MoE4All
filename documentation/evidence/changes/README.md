---
kind: index
status: current
scope: change-history
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 改进与变化记录

这里记录某一阶段为何选择、修正或否决一种实现。日期、提交和设备只约束该阶段；**当前行为以[架构](../../architecture/README.md)为准**。`preserved_in_tag` 只表示该技术记录存在于 tag 时，不表示正文每一阶段都由 tag 实测验证；历史计划中的“当前”不能作为 0.9.0 能力断言。

## 演化与反例

- [2026 年 8 月运行时演化与提交索引](2026-08-runtime-evolution.md)
- [早期否决实验与后来变化](2026-08-rejected-experiments.md)

## 历史技术过程

- 内存：[统一内存的目标与阶段](memory/unified-memory.md)、[三级专家分页演化](memory/tiered-weight-paging.md)
- 模型：[Qwen3.5/3.6](models/qwen35-qwen36.md)、[Qwen3.5 MTP](models/qwen35-mtp.md)、[DeepSeek 系列](models/deepseek-family.md)、[DiffusionGemma](models/diffusion-gemma.md)
- 后端：[Metal](backends/metal.md)、[iGPU](backends/integrated-gpu.md)
- 产品：[浏览器控制面演化](product/browser-control-plane.md)

有精确硬件数字的专项继续见[优化专项](../campaigns/README.md)和[基准测试](../benchmarks/README.md)；事故原因见[事故记录](../incidents/README.md)。
