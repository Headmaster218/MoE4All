---
kind: proposal
status: proposed
scope: agent-product
proposal_commit: 6441d296
updated: 2026-09-28
---

# Agent 发行版边界

这是 tag 之后提出的目标产品架构，不表示 `release-0.9.0` 已交付 Agent。该 tag 不包含 DSH fork、插件 Git submodule
或隔离 harness；相关源码后来出现在 `feat/agent-distribution-v2` 的提交历史中，详见[发行基线审计](../evidence/audits/2026-09-28-release-0.9.0-baseline-and-new-commits.md)。
该方案基于 2026-09-23 对 `9085a78`、本机 DSH 安装包和脱敏配置结构的调研；当时只读代码和配置，不启动模型，也未读取凭据或会话内容。早期“直接纳入 DSH 子树”的建议已由后续独立仓库/submodule 边界取代，不应把原建议当作最终源码布局。

## 产品目标

MoE4All Agent 发行版包含：

- 受维护的 DSH 派生 harness 和 Web UI；
- 选定插件与发行 profile；
- Product Host 与 Engine Manager；
- 独立 infr worker；
- 可选桌面 shell；
- 无头和浏览器入口。

## 进程边界

```text
Desktop shell (optional)
  -> Product Host / DSH-derived harness
       -> agents, sessions, tools, approvals, plugins, Web UI
       -> Engine Manager
            -> infr worker
                 -> models, KV, MTP, scheduler, unified GPU resources
```

Engine Manager 是产品模式下工作进程生命周期的唯一所有者。Electron、插件和 GUI 不得分别启动同一模型并争抢 GPU。

## 两个协议面

- **推理数据面**：兼容 OpenAI 的聊天、嵌入、模型列表。
- **引擎控制面**：协议版本、工作进程实例、检查/估算、加载/就绪、能力、资源快照、
  槽位/队列、排空和关闭。

控制面不能通过解析人类可读日志实现。日志仍写 stderr；结构化协议应带请求 ID 和工作进程实例 ID。

## 状态所有权

| 状态 | 所有者 |
|---|---|
| Agent、消息、工具、审批、任务、记忆 | Harness/Product Host |
| 模型、KV、MTP、并发槽位、GPU 池 | infr 工作进程 |
| 工作进程启停、模型切换、准入策略 | Engine Manager |
| 用户配置、插件选择、产品设置 | Product Host |

## 源代码与插件来源

发行版维护三层：

1. 上游 DSH/Desktop 机制层，固定 commit 和来源。
2. MoE4All 产品修改层，保留差异动机和测试。
3. 第三方插件层，固定版本、权限和本地覆盖层。

后续 Agent 开发分支通过固定 submodule 引入 DSH fork 与选定插件；检出完整源码仍需初始化 submodule。源码存在、开发
harness 可启动与产品集成完成是不同状态，文档和发行说明必须分别描述。

用户凭据、会话、记忆库、附件、日志和机器地址不进入源码树。插件声明不是安全沙箱；需要隔离的工具走受约束子进程、
MCP 或 OS sandbox。

## 本机改造带来的产品要求

调研时的 DSH Desktop `0.6.3` 搭载 `@deepseek-ai/dsh 0.1.1-rc.2`，安装目录含约 197 个 `@deepseek-ai` 包；这不是需要照搬或已经启用的插件数量。profile 中见到记忆、搜索、定时、远程 Web UI、TTS 等第三方组合。发现的本地改造包括固定监听端口、远程审批/事件路由、旧 transport fallback、上传插件文件名与删除授权补丁，以及用外部 Embedding 服务替换重复的本地 ONNX 依赖。这些是源码迁移的需求和测试线索，不是 0.9.0 发行版能力。

早期 profile 端口覆盖曾与桌面 watchdog 等待端口不一致；正式产品必须让 launcher、健康检查、UI 和远程入口读取同一配置。上传插件当时不在启用 bundles 清单中，不能因找到工作区 diff 就宣称已启用或已完成安全验收。旧 remote-web-ui 的 manifest 要求比本机 harness 更高的 dsh 版本，不能长期依赖注入 fallback 维持协议兼容。记忆插件的 Embedding 后端应是显式可选依赖，而非静默 stub。

## 与当前 Rust 整理的关系

Agent 接入不要求先完成全仓 crate 重写。优先让 `infr-engine` 承接稳定生成/能力契约，保持
`infr-server` 为协议适配，逐步拆清 `infr-llama` 内部模型/会话/调度器/MTP 边界。
