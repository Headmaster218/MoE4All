---
kind: proposal
status: proposed
scope: agent-product
updated: 2026-09-30
---

# MoE4All Agent 发行版

`release-0.9.0` 是引擎版本，不包含 Agent 发行版。本页描述 tag 之后确认的产品方向；
不能把开发分支已有的源码、可运行的隔离开发环境和面向用户的完整安装包视为同一完成状态。
旧的独立 Product Host、Engine Manager 和新控制协议方案不再是当前目标。

## 产品定位

MoE4All Agent 以 DSH 为主体：保留其 Agent、会话、工具、审批、插件、profile 和 Web UI 机制，
将用户可见的产品名称与视觉标识改为 MoE4All，并提供一组由本项目维护的默认插件。
用户仍通过 DSH 原有的插件安装与 profile 机制管理插件，不需要第二套 Agent 框架或插件运行时。
内部 `@deepseek-ai/dsh-*` 包名、CLI 命令和 `DSH_HOME` 等兼容标识不要求为了品牌化而批量改名。

独立的 `infr` 引擎继续提供推理 API。`dsh-llm-moe4all` 是 DSH 插件，负责引擎连接、模型发现、
可选的本地引擎安装与启停，以及相关设置界面。其他定制插件仍是独立仓库，按正常 DSH 插件方式
安装到默认 profile；它们的内部开发与 Agent 主仓库的文档和发行整合分开进行。
Agent 的消息与工具记录仍由 DSH 保存，引擎的 KV cache 只是可丢弃的加速状态；需要 Embedding 的插件
按现有 Provider/Embedding API 配置，不要求新建统一控制层。

```text
MoE4All 品牌的 DSH（Web；Desktop 尚未接入）
  ├─ DSH 原有 Agent、会话、工具、审批与插件机制
  ├─ MoE4All 默认 profile 和维护的插件包
  └─ dsh-llm-moe4all 插件 ──> 独立 infr serve 进程
```

这里的引擎仍是单独进程，但不因此新增一个独立产品层或要求先重构 Rust crates。
如果插件启动了本地引擎，只能管理它自己启动的进程；外部已有的服务按连接方式使用。

## 已有基础与缺口

`feat/agent-distribution-v2` 已通过 Git submodule 固定 DSH 源码和维护的插件源码。
`harness/distribution/profile/` 记录默认插件组合及不含凭据的 portable 配置；
`harness/scripts/` 可构建 DSH、打包插件并在隔离的开发 profile 中运行 Web 版。
这些脚本会使用本机安装的 DSH 配置建立开发环境，不是可直接交付给其他用户的安装程序。

当前 DSH fork 基本保留上游行为，Web 标题、PWA 名称和 CLI 帮助仍显示 DSH/DeepSeek Harness；
用户可见的品牌替换尚未完成。Desktop 源码未纳入当前维护树，正式 Agent 打包、安装与更新流程
也尚未完成。发行版 profile 中列出的插件是目标组合，不等于已经制成并验收的用户安装包。

## 下一步范围

1. 完成用户可见的 MoE4All 名称、图标、启动入口和默认 profile；保留必要的内部兼容标识。
2. 从固定的 DSH 与插件版本构建可安装发行物，按 DSH 现有机制安装或更新插件，并验证首次启动。
3. 打通引擎安装、模型选择、连接或可选启动、首轮对话及失败重试；保留独立引擎用法。
4. 如需 Desktop，再接入其源码与打包流程；不要将当前 Web 开发入口描述为已完成的桌面发行版。

发布前要验证上述用户路径和既有引擎行为，但不以新一轮性能优化或引擎架构重写为前提。
源码与历史提交的归属见[发行基线审计](../evidence/audits/2026-09-28-release-0.9.0-baseline-and-new-commits.md)。
用户凭据、会话、记忆库、附件、日志与机器地址不得进入源码或默认发行 profile。
