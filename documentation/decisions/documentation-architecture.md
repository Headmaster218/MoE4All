---
kind: decision
status: accepted
scope: repository-documentation
date: 2026-09-26
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 文档架构

## 决策

`documentation/` 是 MoE4All 唯一的完整文档库。仓库根目录只保留项目级入口：

- `README.md`
- `README_EN.md`
- `CHANGELOG.md`

用户指南、参考手册、当前架构、开发流程、性能证据、变化记录、设计决策和路线图都属于
`documentation/`。crate README 只允许保留构建该 crate 所必需的极短入口，详细内容必须链接回本目录。

## 目标

这套结构同时服务四类读者：

1. 使用发布版、API 或命令行的用户。
2. 调试模型、显存、分页和并发行为的维护者。
3. 做性能实验、正确性验证和发版验收的开发者。
4. 维护未来 Agent、Harness、插件和 Engine Manager 的产品开发者。

目录按读者任务和内容生命周期组织，不按某一次重构或某个作者的工作笔记组织。

## 目录职责

| 目录 | 内容 | 更新方式 |
|---|---|---|
| `guide/` | 用户怎样完成任务 | 随用户流程更新 |
| `reference/` | 参数、API、能力矩阵、内核覆盖 | 随接口和能力更新 |
| `architecture/` | 当前系统怎样工作、边界和不变量 | 随实现行为更新 |
| `development/` | 如何做基准测试、验证、扩展和发版 | 随工程流程更新 |
| `evidence/` | 基准测试、优化专项、变化记录、验收、事故、审计和产物 | 追加新记录，避免覆写历史 |
| `decisions/` | 重要设计选择及其理由 | 新建 ADR 或明确标记为已被取代 |
| `roadmap/` | 尚未落地的方向和研究队列 | 完成后移出或关闭 |
| `_templates/` | 可复用的文档模板 | 随文档制度更新 |

本目录不保留原样旧文档副本。已经提交过的旧版本可从对应旧分支或 Git 提交读取；这里仅维护按主题重写的事实和历史证据。未提交的本机草稿不具备这种恢复路径。

2026-09-26 的整理草案曾建议保留 `archive/` 原样副本，并为每次 benchmark 提交 tracked manifest。最终选用不留原样归档、原始日志放本机忽略目录的方式；因此旧草案的清单和流程不能当作已经执行的制度。必要的 checksum、固定条件与证据限制仍应写入相应 evidence 页面。未提交的本机草稿不能假定可从 Git 恢复，删除前需完成逐篇覆盖审计。

## 事实层级

同一主题出现冲突时，按以下顺序判断：

1. 当前源码、测试和可复现运行结果。
2. 标记为 `current` 且 `verified_commit` 足够新的架构或参考页。
3. 带固定提交、模型、配置和硬件信息的 evidence。
4. 带日期与来源的历史变化记录。
5. roadmap、proposal 和尚未验证的推导。

历史记录可以解释当时发生了什么，但不能覆盖 current 页面描述的现状。

## 内容边界

### 指南与参考手册

Guide 以任务为中心，例如启动服务、调用接口、处理图片或控制思考模式。Reference 以可查字段为中心，例如配置键、
支持矩阵和 kernel 能力。Guide 可以链接 Reference，但不复制完整字段表。

### 架构与设计决策

Architecture 描述当前机制、所有权、数据流和不变量。Decision 记录为什么选择该方案、替代方案和后果。
一个决策被新方案替代后，旧 ADR 保留并标记 `superseded`，当前架构页只描述生效方案。

### 工程证据

工程证据是可追溯记录，不是长期事实页：

- `benchmarks/` 保存固定条件下的测量结果。
- `campaigns/` 保存基线、性能分析、假设、实验、失败方向和结项记录。
- `acceptance/` 保存功能验收。
- `incidents/` 保存用户症状、根因、被破坏的不变量、修复和回归保护。
- `audits/` 保存固定时间点的审查。
- 本机 Git 忽略的 `benchmark-data/` 保存原始产物、日志和 trace；`evidence/` 只保存提炼结果、来源说明及必要的 checksum。

新的数字必须说明 evidence 等级：`measured`、`historical-sample`、`simulation` 或 `theoretical`。

### Roadmap 与变化记录

尚未实现的设计不得写成当前能力。完成后应把稳定机制写入架构文档，把测量过程写入工程证据，并关闭对应
路线图条目。被替代方案、原因及负面结果放在 `evidence/changes/`，不回填为当前架构。

## 元数据

新建的权威页面使用简短 front matter：

```yaml
---
kind: architecture
status: current
scope: parallel-runtime
last_verified: 2026-09-26
verified_commit: COMMIT
---
```

常用状态：

- `current`：当前事实源。
- `draft`：正在形成，不能作为稳定契约。
- `historical`：固定时间点的记录。
- `mixed`：过渡期间尚未完全拆分的材料，不作为长期目标状态。
- `proposed`：尚未接受的设计。
- `superseded`：已被新决策取代。

Evidence 还应记录日期、提交、模型、量化、硬件、配置、命令和原始产物位置。

## 链接规则

1. 文档间使用 `documentation/` 内的相对 Markdown 链接。
2. 不引用已删除的文档路径，也不把目录外的副本描述为权威原文。
3. 源码可以作为实现证据；链接应落到稳定模块或符号附近，避免依赖易漂移的行号。
4. 外部论文、上游实现和 issue 必须说明其角色，不能让外部链接代替本项目行为说明。
5. 当前页与 evidence 互链：当前页给结论，evidence 给测量和历史过程。
6. 历史记录中的旧命令和源码路径需注明对应版本，不能伪装成当前使用入口。

## 命名规则

- 当前主题使用稳定名，例如 `parallel-scheduler.md`、`configuration.md`。
- 一次性记录使用日期前缀，例如 `2026-09-24-qwen38-mtp-20k.md`。
- 事故使用 `INC-YYYYMMDD-slug.md`。
- ADR 使用 `ADR-NNNN-slug.md`。
- 避免无范围的 `plan.md`、`results.md`、`report.md` 和 `notes.md`。

## 产品扩展

智能体发行版继续沿用同一结构：Harness、智能体、工具、审批、插件和 Web UI 的当前机制进入
`architecture/product/`；Engine Manager 与 worker 协议进入 `architecture/services/`；插件兼容与安全验证进入
`development/` 和 `evidence/`。不为 Agent 另建一套平行文档树。

## 维护门槛

文档变更至少检查：

1. 所有相对链接都能解析。
2. 新文件能从分类索引到达。
3. current 页面包含核验日期和提交。
4. Benchmark 没有遗漏模型、量化、硬件、上下文、并发和采样条件。
5. 行为变化同时更新 Architecture/Reference；只有过程变化时更新 Evidence。
6. 根 README 只摘要并链接，不复制详细事实表。

模板位于 [`_templates/`](../_templates/)。
