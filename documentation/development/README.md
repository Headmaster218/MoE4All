---
kind: index
status: current
scope: development
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 开发手册

- [新增 GGUF 模型家族](model-porting.md)
- [从源码构建](building-from-source.md)：工具链与 glslc 版本要求、二进制可移植性、GPU 测试与 GTT 注意。

## 性能

- [Benchmarking 与 profiling](performance/benchmarking.md)
- [优化方法和已知误区](performance/optimization-playbook.md)

## 验证与发布

旧的 Windows 固定资源矩阵及其发版门禁手册已退役，不再作为当前版本的验证入口。
提示词规划器 `__test-plan-prompt` 和测试资源模拟能力仍保留，供其他基准测试使用。

不可变的运行结果和 campaign 放在 [工程证据](../evidence/README.md)，不要把新的数字继续追加进方法论页面。
