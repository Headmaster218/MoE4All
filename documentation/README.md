---
kind: index
status: current
scope: repository-documentation
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---
# MoE4All 文档中心

这里是 MoE4All 的统一文档目录。根目录的 [中文 README](../README.md)、
[English README](../README_EN.md) 和 [CHANGELOG](../CHANGELOG.md) 继续作为项目与发布入口；
其余用户指南、参考资料、当前架构、工程手册、性能证据、变化记录、决策和路线图统一放在本目录。

当前源码、架构和接口事实固定在 tag `release-0.10.0`（`7b4a9411e9c7524f8dc6ae2603c62d5f338190dc`）。本轮按该 tag 核对源码、0.9.0 以来的提交和已有实测记录；核验日期为 2026-10-11，没有新跑推理。运行记录保留各自的提交与二进制，tag 之后的未提交性能代码和 Agent preview 不计入当前事实。完整覆盖范围见[0.10.0 文档核对审计](evidence/audits/2026-10-11-documentation-0.10.0.md)。

## 按任务阅读

| 我想做什么                             | 入口                              |
| -------------------------------------- | --------------------------------- |
| 下载并启动发布版                       | [用户指南](guide/README.md)        |
| 查配置、能力和内核支持                 | [参考手册](reference/README.md)    |
| 理解运行时、内存、调度和模型实现       | [当前架构](architecture/README.md) |
| 做性能优化、验证或发版                 | [开发手册](development/README.md)  |
| 查某次基准测试、优化过程、验收或审计   | [工程证据](evidence/README.md)     |
| 查为何采用某项设计                     | [架构决策](decisions/README.md)    |
| 查尚未完成的工作                       | [路线图](roadmap/README.md)        |
| 查实现如何演变、哪些实验被否决         | [变化记录](evidence/changes/README.md) |

## 当前事实源

以下页面是当前事实入口：

- [系统总览](architecture/system-overview.md)
- [代码库地图](architecture/codebase-map.md)
- [模型执行路径](architecture/models/runtime-families.md)
- [并发调度](architecture/runtime/parallel-scheduler.md)
- [CPU miss](architecture/runtime/cpu-miss-offload.md)
- [Agent 产品边界](architecture/product/agent-distribution.md)
- [Qwen3.8 MTP](architecture/runtime/qwen38-mtp.md)
- [运行时资源生命周期](architecture/memory/runtime-resource-lifecycle.md)
- [服务与冷 KV 会话](architecture/services/server-and-session-cache.md)
- [Vision 与 Embedding](architecture/services/vision-and-embedding.md)
- [模型能力矩阵](reference/model-capabilities.md)
- [API 使用](guide/serving/api-quickstart.md)

0.10.0 源码已包含品牌化 DSH 与八个维护插件的 gitlink、隔离 Agent 初始化和 Web 启动脚本。它们是开发入口；引擎 tag 不等于正式 Agent EXE 发行。见[Agent 产品边界](architecture/product/agent-distribution.md)、[Agent 使用与构建](guide/agent-development.md)及[剩余路线图](roadmap/agent-distribution.md)。

本版变化与实测入口：[0.9.0 → 0.10.0 变化记录](evidence/changes/2026-10-release-0.10.0.md)、[发布前性能与稳定性矩阵](evidence/benchmarks/2026-10-09-release-0.10.0-matrix.md)、[Decode 与 CPU miss 专项](evidence/campaigns/qwen38/2026-10-08-decode-and-cpu-miss.md)。

带日期的记录保留当时的论证和数据，不自动代表当前 `HEAD`。已核实机制与变化过程分开导航；原分支的旧文件不作为本目录的第二事实源。

## 内容规则

技术术语按[文档术语表](reference/terminology.md)统一。正文以中文为主，Prefill、Decode、hit、miss、bubble 等术语及代码、接口、日志、专名保留原文。历史记录的基线和测量日期不随当前版本号改写。

1. 当前机制写在 `architecture/`，当时为何这样选写在 `decisions/`。
2. 一次运行的数字写在 `evidence/benchmarks/`，不可回写成无基线的“当前结果”。
3. 优化过程和失败实验写在 `evidence/campaigns/`。
4. 事故根因写在 `evidence/incidents/`，用户排错步骤写在 `guide/`。
5. 未完成事项写在 `roadmap/`，不混入当前架构正文。
6. 被替代的实现写入变化记录；无需在本目录保存原样旧文档，Git 历史仍可恢复。

详细规则见 [文档架构](decisions/documentation-architecture.md)。
写新页面时可使用[文档模板](_templates/README.md)。
