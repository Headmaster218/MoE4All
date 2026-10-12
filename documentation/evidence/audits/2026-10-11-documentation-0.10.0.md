---
kind: audit
status: historical
scope: documentation-0.10.0
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# 0.10.0 文档全量更新核对

## 固定范围

当前事实固定 `release-0.10.0` = `7b4a9411e9c7524f8dc6ae2603c62d5f338190dc`，不跟随另一个工作区不断变化的HEAD。基于tag新建 `docs/release-0.10.0` 文档分支，未修改Rust/shader/scripts/子模块指针，未启动后端或新跑GPU。

原有91篇canonical Markdown均纳入扫描；当前指南、参考、架构、工程方法和索引按tag更新，8/9月历史数字及失败现场保留原基线。新增17篇页面补当前CPU miss/Agent、辅助服务指南、timeline、发布矩阵、线上对账、优化取舍、事故与版本覆盖。原harness四份说明收敛为两份短入口，inventory/local-changes的实质内容重组入迁移记录，没有另保留原样归档。

## 覆盖与事实源

| 范围 | 本轮结果 | 主要定位 |
| --- | --- | --- |
| 98个0.9→0.10提交，含merge | 全部纳入分组索引；机制按正文解释，不重复算收益 | [变化记录](../changes/2026-10-release-0.10.0.md) |
| scheduler/MTP/resource | corrected rows/horizon/cutoff/defaultcohort、livePrefill/currentdepth、两类服务路径 | [架构](../../architecture/README.md) |
| Config/wizard | field/default/显式覆盖、Xe1、4096 ladder、CPU实验配方及模型默认 | [配置](../../reference/configuration.md) |
| 12-cell/72-response发布前矩阵 | 全数据、metric、commit/EXE、有限稳定性、aux失败/修复 | [矩阵](../benchmarks/2026-10-09-release-0.10.0-matrix.md) |
| Decode/CPU miss | E1..E16采用/否决、ABBA、CPU基本持平、重叠时间线 | [专项](../campaigns/qwen38/2026-10-08-decode-and-cpu-miss.md) |
| 早期MTP/Prefill | head/window/probe、ready panic、原型失败、DMA与ubatch对照 | [专项](../campaigns/qwen38/2026-10-06-prefill-and-mtp.md) |
| 在线服务回退 | binary身份不确定、spill停顿、idleKV压缩ring、真实不同prompt差距 | [对账](../benchmarks/2026-10-09-live-service-reconciliation.md) |
| correctness | shared pool/workspace、mixed sampling suffix、未知hostalloc只增强诊断 | [事故](../incidents/README.md) |
| Linux/图片发行 | singleSH、glibc/nativecodecs/有界HEIF、CI与真实GPU验收分开 | [Linux指南](../../guide/linux-build-and-run.md) |
| Agent/DSH | 固定gitlink、隔离初始化、生产迁移工具与干净产品工具区分 | [产品](../../architecture/product/agent-distribution.md) |
| 旧来源与改造 | 来源版本、commit、package差异、插件迁移与未纳入候选 | [迁移](../changes/product/dsh-source-and-local-migration.md) |

## 确认的主要纠偏

1. 四行VERIFY包含一个leading token，通常仅三个新候选；并非预测四个新token。
2. Qwen3.8仍支持MTP，不能把Qwen3.5旧搁置结论套到全部MTP。
3. head32768 horizon与ParallelSeam49152target cutoff不同，传统单请求不读取cutoff。
4. CPUmiss默认关闭且shape/量化/完整RAM受限；代码支持1..3不等于各上限已测出收益。
5. arena按固定资源实际分配后的余量，不按预估；ringlanes、activework、configuredslots是三个指标。
6. 每个请求/仪表盘interval、API全程、共同活跃窗口和SSEframe不混用。
7. Vision/Embedding partial结果在crash前不能算PASS；mixedsampling与frozenLUT是不同事故。
8. tag包含Agent源码基础，不包含后续preview成品的发包与验收。
9. Linux/unit/包夹具通过不等于真实GPU/自动更新已验收；本轮没有查询线上CI状态。
10. 新测试统一根test-kv-cache/与30GiB，不延用历史48GiB临时quota，也不触碰生产kv-sessions。

## 证据限制

原始benchmark/log/trace仍在本机忽略的benchmark-data，不复制进公开文档，不发布私人IP、模型绝对路径、credential或用户对话全文。提炼记录保存可解释的数据与hash，但有限screen不替代长期soak。原有48h审计与codec-size记录按原日期继续保留，不随版本号伪造新运行。

root README/README_EN/CHANGELOG仍为发布入口；这次不替代用户的release文案。tag中的CHANGELOG未覆盖完整0.10.0，版本差异另用全提交索引核对。当前事实页无第二份旧基线；历史页出现0.9.0表示当时来源而非漏更新。

## 文档静态门禁

检查当前页frontmatter基线、相对链接及导航可达性、配置路径/关键默认值、版本提交覆盖、Markdown fence和git diff whitespace。未运行模型，不将静态核对写成新增性能或稳定性验收。

## 静态核对结果

- 108 篇 canonical Markdown（原91篇保留、新增17篇）；全部从文档首页经导航可达。
- 333 个相对文件/目录链接与本地标题锚点有效，Markdown fenced blocks 成对；108 篇页面全部可达。
- 43 篇 current 页均固定 release-0.10.0 及 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc；混合方法与历史页使用各自状态，不盲改实测基线。
- Config 宏声明展开的261叶字段已与完整默认值表逐项核对；新增运行时表的路径存在，MTP/CPUmiss/timeline关键默认与向导注入分别核对。
- 98 个范围内提交全部被索引且无重复、无遗漏；原91篇历史文件均保留，没有删除原有技术记录。
- harness inventory/local-changes 已在迁移页覆盖来源commit/版本、10项行为与候选边界，原位置仅保留必要入口README，不保存重复归档。
- git diff --check 通过；改动范围只有 documentation 与 harness Markdown。原性能工作区未被修改；无推理、KV cache 或 GPU 操作。

这是一轮文档与来源核验，不是新性能跑分。后续部署应在 final artifact 上按对应门禁重跑。
