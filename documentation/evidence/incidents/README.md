---
kind: index
status: current
scope: incidents
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 事故

- [INC-20261009：idle KV spill 阻塞 Decode](INC-20261009-idle-kv-spill-stall.md)
- [INC-20261009：辅助 Vulkan command pool 生命周期](INC-20261009-vulkan-aux-command-pool.md)
- [INC-20261010：mixed batch sampling suffix 行错位](INC-20261010-mixed-batch-sampling-suffix.md)

这里保存已经定位的用户可见故障：复现、影响、根因、被破坏的不变量、修复和回归测试。

已整理：

- [INC-20260925：冻结的专家 LUT 槽位导致重复输出](INC-20260925-frozen-expert-lut-slot.md)
- [INC-20260926：DSH 裁剪暴露缺失的递归 checkpoint](INC-20260926-dsh-prune-missing-recurrent-checkpoint.md)

待从历史提交继续补录：

- Qwen3.8 循环状态跨轮恢复；
- 不可能的并行预填充临时工作区预留；
- 统一显存走廊碎片化。
