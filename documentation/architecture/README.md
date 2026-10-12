---
kind: index
status: current
scope: architecture
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 当前架构

本页只索引 `release-0.10.0` tag 已核对的机制。过去如何演进见[变化记录](../evidence/changes/README.md)，运行数字见[工程证据](../evidence/README.md)，未落地方案见[路线图](../roadmap/README.md)。

## 当前机制

- [CPU expert miss](runtime/cpu-miss-offload.md)：默认关闭的实验 scalar 支路。
- [Agent 产品边界](product/agent-distribution.md)：DSH 生态复用与数据/进程隔离。

- [系统总览](system-overview.md)
- [代码库地图](codebase-map.md)
- [模型执行路径](models/runtime-families.md)
- [并发调度](runtime/parallel-scheduler.md)
- [Qwen3.8 MTP](runtime/qwen38-mtp.md)
- [运行时资源生命周期](memory/runtime-resource-lifecycle.md)
- [服务与冷 KV 会话](services/server-and-session-cache.md)
- [Vision 与 Embedding](services/vision-and-embedding.md)

模型、后端、量化和上下文组合的可用范围另见[模型能力矩阵](../reference/model-capabilities.md)，不能只由架构名推断。
