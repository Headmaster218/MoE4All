---
kind: audit
status: completed-with-limitations
audit_date: 2026-09-28
baseline_tag: release-0.9.0
baseline_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 0.9.0 文档归并与去重审计

## 决策与范围

`docs/release-0.9.0` 从 `release-0.9.0` tag 建立。`documentation/` 是唯一完整文档目录；根 README、README_EN 与 CHANGELOG 是发行入口。tag 旧 `docs/`、`infr-fork-wiki/`、根性能报告、GUI README 在本整理分支从原路径移除，**不在新目录保留逐字归档副本**。原文可从 tag 或旧分支恢复。主工作区及其未提交改动不属于本分支整理范围。

tag 的相关 Git tree 为 85 项：三个根入口继续保留，其余 82 项在删除前曾逐一与 `release-0.9.0` blob ID 做 hash 核对，82/82 一致。这证明来源清单和恢复路径，并不证明每段文字都应在新体系逐字出现。2026-08-25 Wiki 的 41 篇中文阶段材料也按主题吸收，原样目录不保留。

## 内容落点

| 来源主题 | 当前使用位置 | 被替代的实现、测量或限制 |
|---|---|---|
| 参数、启动、API、模型与 kernel | `guide/`、`reference/` | 旧版本的配置口径和模型开发阶段在 `evidence/changes/` |
| 运行时、模型、服务、并发、MTP、显存 | `architecture/` 的 `current` 页面 | 六池→全局池、Prefill/Decode 分治、Host DMA、旧单头 MTP 等在[变化记录](../changes/README.md) |
| Qwen3.6 35B 与 Qwen3.5 122B | 当前机制见模型/内存页 | [深度缓存矩阵](../benchmarks/2026-08-20-qwen36-apex-matrix.md)、[122B 冷追踪](../benchmarks/2026-08-24-qwen35-122b-cold-trace.md)、[MoE 微基准](../benchmarks/2026-08-25-moe-pager-microbench.md)保存当时数字与配置 |
| Qwen3.8、DeepSeek V4、iGPU/Metal | 当前支持范围见能力矩阵 | benchmark、campaign、事故、变化记录区分成功、回退与尚未验收的组合 |
| Agent/DSH、训练及其他未交付方向 | 不写作 tag 当前能力 | `roadmap/` 标为提案；tag 后提交另见[基线审计](2026-09-28-release-0.9.0-baseline-and-new-commits.md) |
| 发版与性能方法 | `development/` | 固定运行的数字保留在 `evidence/benchmarks/`，不混入方法页 |

Wiki 中独有的长 commit/hash 和小数数值，排除旧目录后与新体系做 token 交叉检查，缺项为 0。82 份 tag 来源的首轮检查只发现两处旧索引文件名中的日期写法 `20260924`；对应的 20K MTP benchmark 已按新命名收入[基准测试](../benchmarks/2026-09-24-qwen38-mtp-20k.md)，不是测量缺失。说明这一改名后，比较时统一千位分隔符，长 hash 和小数数值无缺项。该检查能发现漏掉的提交和历史数字，**不能证明语义逐句等价**。迁移中重点人工核对了 122B 冷/热口径、Qwen3.6 的合成深度限制、未采用的缓存和内核策略，以及曾被标为暂缓而后来落地的动态 KV。

## 版本与证据边界

- 架构、配置和产品能力按 tag `ed62393068679573afe94a1472454efe7eae0f15` 说明。2026-09-27 系统 benchmark 自述的 `1f53db6b5db84e7eceb0a1c7254eee573c19b70c` 不是该 tag 的祖先；报告是独立二进制上的实测，不是 0.9.0 tag 的性能验收。
- 历史数据保留其硬件、模型、配置、提交和证据等级。缺少命令、重复次数或原始日志时标 `historical-sample`；模拟和理论值不混充硬件实测。
- ignored trace、含完整 prompt/reasoning/output 的逐轮 JSON 与本机绝对路径不纳入文档库；仅保留可读报告、脱敏指标与必要的产物校验信息。
- 本次只处理文档和性能/稳定性资料，没有运行 Rust 构建、测试或 GPU benchmark，也没有修改运行时代码。源码和配置注释中仍可能有旧 `docs/` 字面路径，超出本次修改范围。

清除旧目录并移出 benchmark 原件后复查：`documentation/archive/` 不存在；目录外的 Markdown/Mermaid/文档产物只有根 README、README_EN、CHANGELOG。87 篇 Markdown（含三个根入口）的 222 条本地链接无断链，首页可达全部非根页面。原始报告、DOCX、指标 JSON 和追踪 ZIP 转入本机 Git 忽略的 `benchmark-data/`；`documentation/` 仅保留提炼结论、证据限制与必要校验值。`architecture/` 下的 Markdown 均标为 `current`，提案与历史变化不在此目录。Git 变更范围仅文档、旧文档删除和三个根入口；Rust 代码未改。此次扫描仍不包含原始 Git 版本中的旧链接，因为旧文档只通过 Git 历史恢复。
