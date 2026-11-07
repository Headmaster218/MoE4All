---
kind: index
status: current
scope: architecture
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 当前架构

本页只索引 `release-0.9.0` tag 已核对的机制。过去如何演进见[变化记录](../evidence/changes/README.md)，运行数字见[工程证据](../evidence/README.md)，未落地方案见[路线图](../roadmap/README.md)。

## 当前机制

- [系统总览](system-overview.md)
- [代码库地图](codebase-map.md)
- [模型执行路径](models/runtime-families.md)
- [并发调度](runtime/parallel-scheduler.md)
- [Qwen3.8 四 token MTP](runtime/qwen38-mtp.md)
- [运行时资源生命周期](memory/runtime-resource-lifecycle.md)
- [服务与冷 KV 会话](services/server-and-session-cache.md)
- [Vision 与 Embedding](services/vision-and-embedding.md)

模型、后端、量化和上下文组合的可用范围另见[模型能力矩阵](../reference/model-capabilities.md)，不能只由架构名推断。
