---
kind: index
status: current
scope: repository-documentation
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---
# MoE4All 文档中心

这里是 MoE4All 的统一文档目录。根目录的 [中文 README](../README.md)、
[English README](../README_EN.md) 和 [CHANGELOG](../CHANGELOG.md) 继续作为项目与发布入口；
其余用户指南、参考资料、当前架构、工程手册、性能证据、变化记录、决策和路线图统一放在本目录。

当前源码与架构事实固定在 tag `release-0.9.0`（`ed62393068679573afe94a1472454efe7eae0f15`）。tag 之后的 Agent/DSH 提交与另一源码提交上的新 benchmark 均单独标注，不回写为该 tag 的实现事实。

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
- [Qwen3.8 MTP](architecture/runtime/qwen38-mtp.md)
- [运行时资源生命周期](architecture/memory/runtime-resource-lifecycle.md)
- [服务与冷 KV 会话](architecture/services/server-and-session-cache.md)
- [Vision 与 Embedding](architecture/services/vision-and-embedding.md)
- [模型能力矩阵](reference/model-capabilities.md)
- [API 使用](guide/serving/api-quickstart.md)

Agent 发行版仍是提案，且来源于 tag 之后的开发历史，见[Agent 发行版边界](roadmap/agent-distribution.md)和[基线审计](evidence/audits/2026-09-28-release-0.9.0-baseline-and-new-commits.md)。

带日期的记录保留当时的论证和数据，不自动代表当前 `HEAD`。已核实机制与变化过程分开导航；原分支的旧文件不作为本目录的第二事实源。

## 内容规则

技术术语按[文档术语表](reference/terminology.md)统一。当前事实页、新增指南和导航以中文为主；代码、接口、专名保留原文。部分历史长文仍有英文正文，页首说明其时间和适用范围。

1. 当前机制写在 `architecture/`，当时为何这样选写在 `decisions/`。
2. 一次运行的数字写在 `evidence/benchmarks/`，不可回写成无基线的“当前结果”。
3. 优化过程和失败实验写在 `evidence/campaigns/`。
4. 事故根因写在 `evidence/incidents/`，用户排错步骤写在 `guide/`。
5. 未完成事项写在 `roadmap/`，不混入当前架构正文。
6. 被替代的实现写入变化记录；无需在本目录保存原样旧文档，Git 历史仍可恢复。

详细规则见 [文档架构](decisions/documentation-architecture.md)。
写新页面时可使用[文档模板](_templates/README.md)。
