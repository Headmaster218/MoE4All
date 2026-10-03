---
kind: index
status: current
scope: evidence
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 工程证据

## 类型

- [基准测试](benchmarks/README.md)：固定提交、硬件、模型和配置下的运行结果。
- [优化专项](campaigns/README.md)：基线、性能分析、实验、落地与结项。
- [改进与变化记录](changes/README.md)：实现演化、被替代的设计及否决方向。
- [验收](acceptance/README.md)：功能验收记录。
- [事故](incidents/README.md)：症状、根因、被破坏的不变量和回归保护。
- [审查](audits/README.md)：固定时间点的代码或平台审查。

原始 benchmark 数据、完整日志、trace 和原始总结统一保存在本机 Git 忽略的 `benchmark-data/`；本目录只保存提炼后的结果、方法、限制和必要的校验信息。该本机目录不随仓库发布。

## 证据等级

- `measured`：有明确命令与结果的硬件运行。
- `historical-sample`：真实运行但原始证据不完整。
- `simulation`：基于 trace 或成本模型。
- `theoretical`：基于字节量、带宽或依赖关系推导。

不同等级不能在同一表中不加说明地比较。

当前代码架构统一以 `release-0.9.0` 为准；单次运行记录仍各自绑定工件中的精确源码 commit，不因文档基线而改写。
