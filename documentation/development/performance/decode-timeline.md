---
kind: engineering
status: current
scope: decode-timeline
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# Decode timeline 与测量口径

## 有界采集

0.10.0 的 `prof.timeline_path` 启用异步、有界 Decode timeline collector；默认不输出。相关参数为 `prof.timeline_skip_steps=64`、`timeline_steps=16`、`timeline_stride=128`、`timeline_windows=12`、`timeline_lanes=0`。确认自身 workload 有足够 step 覆盖采样窗口；参数名与默认值见[配置](../../reference/configuration.md)。

collector 利用已有同步点记录 Main/Upload 队列及 mapped expert copy spans，避免给每一个 op 加同步。异步采集不等于零开销；10 月专项测得 profiler 每 step 约 2.13..3.59 ms 的额外成本，clean end-to-end 和 timeline 数字不能混算。

## 分析规则

- step wall、GPU Main、Upload、host prepare、mapped copy 和 encode 可能重叠，不能直接相加。
- 不用多个独立 P50 的相减估算可节省时间；检查同一个 step 的时间区间和因果依赖。
- isolated kernel 加速、DMA 带宽及减少 dispatch 都不自动证明模型速度更快。
- collector 只解释实际采样 shape、设备与路径，不从 ordinary Decode 推算 MTP VERIFY。
- 普通服务 1.5 秒仪表盘/5 秒日志是 interval throughput，API final timings 是全程；SSE frame 数不是 token 数。
- 多 lane aggregate 需共同活跃窗口；单独 lane 的平均 tok/s 求和可能包含错开的时段。
- Prefill 的 chunk 插值峰值不是瞬时 kernel 速度；前中后段按 wall time 还是 token progress 必须注明。

## 可重跑策略

显式固定 commit、EXE SHA、硬件/driver、实际 ubatch/ring lanes、KV 与 host tier、sampling、MTP/CPU miss 开关、前缀复用、请求顺序、并发及退出方式。先 correctness，再 clean matched ABBA，最后用采样解释瓶颈，完整规则见[优化手册](optimization-playbook.md)。

日志/JSON/trace 留在忽略的 `benchmark-data/`；新测试缓存必须在根 `test-kv-cache/`，统一滚动 30 GiB，遵守[缓存规则](../release/long-context-validation.md#测试缓存管理)。文档只保存基线、方法、结论、关键数据和反例，不依赖本机 raw 路径作为唯一事实来源。
