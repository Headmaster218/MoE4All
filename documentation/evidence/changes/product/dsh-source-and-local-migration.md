---
kind: change-record
status: historical
scope: dsh-provenance
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# DSH 来源与本地改造迁移

这是2026-09-23建立Agent基础时观察到的生产安装来源与行为清单，不是当前依赖锁；当前gitlink见[产品架构](../../../architecture/product/agent-distribution.md)。本文集中原harness inventory/local-changes中的信息，不携带用户数据、地址和凭据。

## 来源版本

| 组件 | 安装版本 | 上游提交与注意事项 |
| --- | --- | --- |
| DSH | 0.1.1-rc.2 | `b150a551b8d465e31e418e1b2eaf5e79bbb7d28e`；安装Harness本体未发现overlay |
| Desktop | 0.6.3 | `e589bf688624871d953f8c58418f913c407dfc7e`；root package仍0.1.1，artifact/tag为0.6.3，Harness依赖0.1.1-rc.2 |
| dshmarket | 1.45.1 | `1664caec99219b4902f1686e2e34614815d38346`；无overlay |
| Brave | 1.0.0 | `10799a8c19ec6d672789257be609750515dbb513`；profile选择该搜索provider |
| TTS | 0.3.1 | `ff26f361df77605599896fa2ba76eb875eddc1d2`；以安装lockfile而非同名tag为准 |
| Remote Web UI | 0.3.17 | `46f1616f26525c7b8a782968f02fa97fde2c5790` |
| Mneme | 0.7.29 | `0b3e99f6ff4948ab4b823914a0a16f4411d25a4c`；nested dsh-mneme才是发布package，root仍0.7.16 |
| EasyRewrite | 2.4.1 | `8bfa03c946e81e6159d0c839ffb7a02c6c52c158`；无overlay |
| Cron | 0.2.7 | `f7a57a64ac26e325ddacc59cd9368e6844657347`；无overlay |
| 候选 uploader | 0.4.0 | `63d474b2a3aea6b880a6c7f36aa7a1d0680a38a6`；未启用、未纳入当前submodule |

各维护仓库从准确source checkpoint导入，不导入完整上游history；各自 `UPSTREAM.md` 保存来源、release/commit、license与attribution，根gitlink固定组合。Desktop shell和uploader没有当前维护子模块，不能列成内置已交付插件。

## 行为迁移与边界

| ID | 行为 | 迁移状态/边界 |
| --- | --- | --- |
| DESKTOP-PORT | Desktop固定1633，启动检查占用 | 未迁移，隔离Web使用独立端口；不能照搬为产品默认 |
| REMOTE-EVENTS | 配对/授权门控转发events.mux/host，使审批和提问到手机 | 维护Remote子模块已迁移，有contract/rewrite测试记录 |
| REMOTE-TRANSPORT | 旧DSH缺少预期transport，createApiClient fallback | 兼容adapter；统一transport契约后再考虑移除 |
| MNEME-V2 | 单调用逐轮distill2、cluster驱动dream3、engine/reasoning设置 | 维护Mneme已迁移；保存功能而非生产记忆内容 |
| MNEME-EMBED | native Embedding替代未用的HF transformers依赖 | profile明确stub override；未来可换native provider契约 |
| PROFILE-POLICY | Brave、Mneme v2、title no-reasoning route | portable patch保留；模型特化仍要发布审查 |
| UPLOAD-PATH | 控制字符、ADS、保留设备名防护；delete仅uploads直接子项 | 旧v1候选保留安全设计，未进当前内置组件；另有same-origin、upload cap、keep-forever默认 |
| REMOTE-ENDPOINT | 私人公网地址和Windows port proxy | 只属用户配置/迁移输入，绝不作默认 |
| TITLE-DIAG | 临时title proxy/探针/stream诊断 | 测试用途，不进入生产routing；log不发布 |
| LOCAL-TOOLS | ASR/media、受控Bilibili评论流程及个人skills | 单独审查提炼为可选插件；不导入cookie、队列、corpus、secret或个人workspace |

历史 `distribution/profile/cordis.patch.yml` 保存行为覆盖，profile/package.json保存插件组合；HF stub保留当时安装策略。源码快照排除session、记忆库、附件、log、credential、remote publicBaseUrl。默认profile不是证明成品已验收；legacy DevHarness复制设置/凭据行为和干净Agent初始化必须区分。发行与迁移剩余工作见[路线图](../../../roadmap/agent-distribution.md)。
