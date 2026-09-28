---
kind: index
status: current
scope: incidents
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 事故

这里保存已经定位的用户可见故障：复现、影响、根因、被破坏的不变量、修复和回归测试。

已整理：

- [INC-20260925：冻结的专家 LUT 槽位导致重复输出](INC-20260925-frozen-expert-lut-slot.md)
- [INC-20260926：DSH 裁剪暴露缺失的递归 checkpoint](INC-20260926-dsh-prune-missing-recurrent-checkpoint.md)

待从历史提交继续补录：

- Qwen3.8 循环状态跨轮恢复；
- 不可能的并行预填充临时工作区预留；
- 统一显存走廊碎片化。
