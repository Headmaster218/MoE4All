---
kind: audit
status: completed-with-limitations
audit_date: 2026-09-29
baseline_tag: release-0.9.0
baseline_commit: ed62393068679573afe94a1472454efe7eae0f15
scope: local-100-draft-coverage
---

# 旧草稿逐篇覆盖审计

## 范围与判断

来源是主工作区 `documentation/` 的 99 篇未跟踪 Markdown，加上 `docs/documentation-architecture-plan.md`，共 100 篇。它们不是该主分支上的 Git blob，不能假设删除后可从 Git 恢复。目标是 `docs/release-0.9.0` 的统一 `documentation/`，当前事实核对基线为 tag `ed62393068679573afe94a1472454efe7eae0f15`。本审计只处理这 100 篇草稿，不声称审完仓库全部历史版本或本机忽略的原始 trace。

核对的单位是可复用的信息：当前机制、历史实验条件与原始结果、失败和回退原因、提交/产物溯源、未落地方案的状态。旧索引链接、逐字段落、过时命令或草案章节号不要求保留；发现旧推算与原始样本冲突时保留样本并写明纠正。`current` 页面不得借用 tag 之后的 harness 提交证明 0.9.0 行为。

| 来源类别 | 篇数 | 处理规则 |
| --- | ---: | --- |
| 与新库同相对路径 | 42 | 逐篇比较正文，按 tag 更新元数据、链接和事实；路径不等于逐字相同 |
| 迁至历史变化或 roadmap | 10 | 逐篇检查新位置的正文和状态 |
| 旧归档 | 45 | 按主题进入当前机制、历史证据或设计取舍，不留下原样 archive |
| 旧 artifact 索引、旧覆盖审计、目录外整理草案 | 3 | 校验值归入实验页；审计与方案被后续审计/决策吸收或明确取代 |
| **合计** | **100** | 仅在下述例外处理完、链接和可达性通过后清理旧工作区 |

## 十篇改路径的草稿

| 旧 `documentation/` 相对路径 | 新主位置 |
| --- | --- |
| `architecture/backends/integrated-gpu.md` | `evidence/changes/backends/integrated-gpu.md` |
| `architecture/backends/metal.md` | `evidence/changes/backends/metal.md` |
| `architecture/memory/tiered-weight-paging.md` | `evidence/changes/memory/tiered-weight-paging.md` |
| `architecture/memory/unified-memory.md` | `evidence/changes/memory/unified-memory.md` |
| `architecture/models/deepseek-family.md` | `evidence/changes/models/deepseek-family.md` |
| `architecture/models/diffusion-gemma.md` | `evidence/changes/models/diffusion-gemma.md` |
| `architecture/models/qwen35-qwen36.md` | `evidence/changes/models/qwen35-qwen36.md` |
| `architecture/product/agent-distribution.md` | `roadmap/agent-distribution.md` |
| `architecture/product/browser-control-plane.md` | `evidence/changes/product/browser-control-plane.md` |
| `architecture/runtime/qwen35-mtp.md` | `evidence/changes/models/qwen35-mtp.md` |

旧 Agent 草稿把后续 DSH/插件 submodule 计入较早 checkpoint；tag 本身不包含这些来源。新页只把它们当作 tag 后开发，并将可执行的产品边界归入 proposal。旧 Qwen3.5/3.6 文案中的“尚未支持”也不能覆盖 0.9.0 的[模型能力矩阵](../../reference/model-capabilities.md)。

## 四十五篇归档的主题去向

下表的路径均相对于旧或新 `documentation/`；右列是主要承载，不表示一个旧页面只能进入一个新页面。旧 wiki 的 41 篇是 2026-08-25 阶段材料，历史结果不升级为 tag 当前实测。

| 旧 `archive/` 相对路径 | 新主位置或处理 |
| --- | --- |
| `README.md` | `README.md`、本审计；旧 archive 使用说明不再适用 |
| `retired-indexes/docs-index-2026-09-24.md` | `README.md`、各类索引；仅导航已换路径 |
| `retired-indexes/performance-index-2026-09-24.md` | `evidence/README.md`、`evidence/benchmarks/README.md` |
| `retired-plans/project-plan-2026-08-05.md` | `evidence/changes/2026-08-05-project-baseline.md`、`development/model-porting.md` |
| `wiki-2026-08-25/architecture/expert-pager.md` | `architecture/memory/runtime-resource-lifecycle.md`、`evidence/changes/2026-08-runtime-evolution.md` |
| `wiki-2026-08-25/architecture/host-dma.md` | `architecture/memory/runtime-resource-lifecycle.md`、`evidence/benchmarks/2026-08-25-moe-pager-microbench.md` |
| `wiki-2026-08-25/architecture/memory-budget.md` | `architecture/memory/runtime-resource-lifecycle.md`、`reference/configuration.md` |
| `wiki-2026-08-25/architecture/moe-scheduling.md` | `evidence/benchmarks/2026-08-25-moe-pager-microbench.md`、`architecture/runtime/parallel-scheduler.md` |
| `wiki-2026-08-25/architecture/prefill-decode.md` | `architecture/runtime/parallel-scheduler.md`、Qwen3.6 campaign |
| `wiki-2026-08-25/architecture/ram-ssd-cache.md` | `architecture/memory/runtime-resource-lifecycle.md`、DeepSeek V4 campaign |
| `wiki-2026-08-25/architecture/README.md` | `architecture/README.md`、`evidence/changes/README.md` |
| `wiki-2026-08-25/architecture/unified-vram.md` | `architecture/memory/runtime-resource-lifecycle.md`、`evidence/acceptance/2026-08-24-elastic-unified-vram.md` |
| `wiki-2026-08-25/experiments/cache-policy.md` | `evidence/campaigns/qwen36/2026-08-20-two-pool-decode-cache.md`、`evidence/changes/2026-08-rejected-experiments.md` |
| `wiki-2026-08-25/experiments/qwen36-campaign.md` | `evidence/campaigns/qwen36/2026-08-19-rx7900xtx-optimization-history.md` |
| `wiki-2026-08-25/experiments/README.md` | `evidence/campaigns/README.md` |
| `wiki-2026-08-25/experiments/rejected-experiments.md` | `evidence/changes/2026-08-rejected-experiments.md` |
| `wiki-2026-08-25/experiments/trace-simulation.md` | `evidence/benchmarks/2026-08-24-qwen35-122b-cold-trace.md`、DeepSeek V4 campaign |
| `wiki-2026-08-25/kernels/attention-hd256-q8.md` | Qwen3.6 campaign、`evidence/changes/2026-08-runtime-evolution.md` |
| `wiki-2026-08-25/kernels/deepseek-v4.md` | `evidence/changes/models/deepseek-family.md`、DeepSeek V4 campaign |
| `wiki-2026-08-25/kernels/deltanet-kda.md` | `evidence/changes/models/ling3-flash.md`、`evidence/changes/models/qwen35-qwen36.md` |
| `wiki-2026-08-25/kernels/quantized-moe.md` | `reference/kernel-capabilities.md`、Qwen3.6 campaign |
| `wiki-2026-08-25/kernels/README.md` | `reference/kernel-capabilities.md` |
| `wiki-2026-08-25/models/deepseek-v4-flash.md` | `evidence/campaigns/deepseek-v4/2026-08-24-rx7900xtx-closeout.md` |
| `wiki-2026-08-25/models/ling3-flash.md` | `evidence/changes/models/ling3-flash.md` |
| `wiki-2026-08-25/models/nomic-embedding.md` | `architecture/services/vision-and-embedding.md`、`evidence/acceptance/2026-08-22-unified-vram.md` |
| `wiki-2026-08-25/models/qwen35-122b.md` | `evidence/benchmarks/2026-08-24-qwen35-122b-cold-trace.md` |
| `wiki-2026-08-25/models/qwen36-35b.md` | `evidence/benchmarks/2026-08-20-qwen36-apex-matrix.md`、Qwen3.6 campaign |
| `wiki-2026-08-25/models/README.md` | `reference/model-capabilities.md`、`evidence/benchmarks/README.md` |
| `wiki-2026-08-25/overview/architecture-evolution.md` | `evidence/changes/2026-08-runtime-evolution.md` |
| `wiki-2026-08-25/overview/results.md` | `evidence/benchmarks/`、`evidence/acceptance/` 与对应 campaign |
| `wiki-2026-08-25/overview/scope.md` | `decisions/documentation-architecture.md`、`development/performance/benchmarking.md` |
| `wiki-2026-08-25/overview/timeline.md` | `evidence/changes/2026-08-runtime-evolution.md` |
| `wiki-2026-08-25/product/browser-gui.md` | `evidence/changes/product/browser-control-plane.md` |
| `wiki-2026-08-25/product/embedding-api.md` | `architecture/services/vision-and-embedding.md`、`evidence/acceptance/` |
| `wiki-2026-08-25/product/README.md` | `guide/README.md`、`architecture/services/server-and-session-cache.md` |
| `wiki-2026-08-25/README.md` | `README.md` 的新导航，不保留旧首页 |
| `wiki-2026-08-25/reference/benchmark-method.md` | `development/performance/benchmarking.md` |
| `wiki-2026-08-25/reference/commit-map.md` | `evidence/changes/2026-08-runtime-evolution.md` 的 89 提交索引 |
| `wiki-2026-08-25/reference/deepseek-v4-data.md` | `evidence/campaigns/deepseek-v4/2026-08-24-rx7900xtx-closeout.md` |
| `wiki-2026-08-25/reference/evidence-index.md` | 各 benchmark/acceptance 的指标与校验值；原始路径不作为现行地址 |
| `wiki-2026-08-25/reference/glossary.md` | `reference/terminology.md` |
| `wiki-2026-08-25/reference/moe-schedule-microbench.md` | `evidence/benchmarks/2026-08-25-moe-pager-microbench.md` |
| `wiki-2026-08-25/reference/qwen122-trace.md` | `evidence/benchmarks/2026-08-24-qwen35-122b-cold-trace.md` |
| `wiki-2026-08-25/reference/qwen36-matrix.md` | `evidence/benchmarks/2026-08-20-qwen36-apex-matrix.md` |
| `wiki-2026-08-25/reference/README.md` | `reference/README.md`、`evidence/README.md` |

## 三篇特别来源与纠正

- `evidence/artifacts/README.md` 的 DeepSeek ZIP 大小、SHA-256 已在[DeepSeek closeout](../campaigns/deepseek-v4/2026-08-24-rx7900xtx-closeout.md)；原始 ZIP 不随文档发布。
- `evidence/audits/2026-09-26-documentation-source-coverage.md` 固定较早 `ee492e2e`，且因保留 41 篇 archive 原样副本而得到 token 零缺项；它的来源主题和负面实验结论由[0.9.0 归并审计](2026-09-28-documentation-consolidation.md)与本审计接续，不能把旧零缺项沿用到删除归档之后。
- 目录外 `docs/documentation-architecture-plan.md` 是 `abddb273` 时的未实施方案。读者、文档类型、权威来源和维护门槛由[已采纳决策](../../decisions/documentation-architecture.md)承接；“保留 archive 原样副本”“每次 benchmark 提交 tracked manifest”等旧建议已明确改选。其 5.2、12.4 等小数是章节号，不是丢失的性能数据。

补遗包括：122B CSV/ZIP 历史 SHA-256、19.2→23.2 的阶段 20.8%、MoE 微基准 2.66 s 编译开销、早期 hd256 200K 2.57 倍、Arc A770 三次 30.15–30.28、社区反馈评论标识、Ling KDA 语义、GUI 停机/日志修复，以及模型引用与共享 HF 缓存。旧 122B shared-fusion 与 35B guard 的百分比上界和原始样本不符，已在[122B 追踪](../benchmarks/2026-08-24-qwen35-122b-cold-trace.md)按原始数值更正；“冷 11.2 对热 23.2 提速 107%”是旧方法页列出的错误示例，不是性能结论。

## 自动检查与限制

对 100 篇来源与新 `documentation/` 全树扫描，统一数字中的千位分隔符、小数尾零和大小写：可解释的旧短 commit 前缀由完整 commit 覆盖；没有未处理的旧长 SHA-256、历史小数或百分比。仅剩旧整理草案的章节号、旧文件名日期写法和旧路径标识。另逐篇核对 45 篇旧归档的章节主题、关键机制、失败结论与历史口径；同路径 42 篇和迁移的 10 篇复查状态、正文变化与链接。token 相同不证明其上下文相同，本审计不承诺逐字翻译，也不把缺失原始命令/trace 说成可复现测量。

文件级映射复查为同路径 42、迁移 10、归档 45、特别来源 2 加目录外方案 1，100/100 有去向，未发现漏列文件。新 `documentation/` 有 87 篇 Markdown，从总索引可达 87/87；扫描 214 条本地文件链接，断链 0；九篇当前架构 Markdown 均标为 `current`，`verified_commit` 均为上述 0.9.0 tag commit。两个页内锚点也与目标标题对应。脚本检查后又人工复核了新补入的历史数字、校验值、Ling/KDA 与 GUI 边界。此结果允许清理主工作区的未跟踪草稿；它不是 GPU 运行验收或逐句语义等价证明。本机 `benchmark-data/` 与其他代码改动不在清理范围。
