---
kind: architecture
status: current
scope: agent-product
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# Agent 产品与兼容边界

## 产品结构

MoE4All Agent 复用 DSH 的 Agent、会话、工具、审批、插件与 profile，只替换可见品牌并提供维护插件；不新建第二套框架。推理仍由独立 `infr serve` 提供。0.10.0 纳入的是源码组合与隔离 Web 开发入口，不是正式 Agent EXE 安装包。

```text
MoE4All 品牌 DSH + 独立 DSH_HOME / workspace / port
  -> 默认 profile 与维护插件
  -> dsh-llm-moe4all
       -> connect: 外部 infr API
       -> 插件拥有的本地引擎进程（可选）
```

内部 `@deepseek-ai/dsh-*`、`DSH_HOME` 等兼容名保留以复用插件生态；保留变量名不等于共享变量值。连接已有引擎不获得该进程的生命周期所有权，不能把“连接”变成重启/停止生产后端。

## tag 中的固定组合

根 `.gitmodules` 全部指向 Headmaster218 维护仓库；gitlink 是版本事实，不以本机子模块工作树替代。

| 组件 | gitlink |
| --- | --- |
| DSH | `79bab8b84216fda929963f7d5248649feb240280` |
| MoE4All provider | `41e137e29ca4ca4e0f24c482ca3cd22ba9e3ea79` |
| 市场 | `ea381411bc360492371860f2d85070e11fe48d84` |
| Mneme | `01ace38811e3b8b17413076dc7326e820c419500` |
| EasyRewrite | `36fe6c186cbff19f476998f54f2b16dd1ca924d7` |
| 定时任务 | `b4cb40a6c24ea5e51eb5760f74b7fd90484b48c6` |
| TTS | `ad88d32ac5c0d09aba4b5dd19d956d3602664285` |
| 手机/Remote Web UI | `323c98ba3559d7d976353e6e12ab3bf5f32384ce` |
| Brave 搜索 | `3b91a69766f6e4efbeeed64ffef709b26e211d3c` |

## 数据与运行隔离

`Initialize-Agent.ps1` 默认在 `harness/agent-state/home` 建立新 profile、workspace 与 agents，默认 provider 为 MoE4All、endpoint 为 `http://127.0.0.1:1234/v1`、mode 为 connect，禁用默认 DeepSeek。已有隔离 profile 会保留；模型发现失败不应迫使用户启动引擎。初始化拒绝生产 `APPDATA/dsh-desktop/harness`、`~/.dsh` 等既有数据位置与不安全 reparse 路径。

`Run-Agent.ps1` 使用独立 `DSH_HOME`/`DSH_AGENTS_HOME`，默认 loopback Web 端口 17373，结束后恢复父进程环境；共享外部 workspace 需明确 `AllowSharedWorkspace`。这让开发入口与生产 DSH 分开，但还不能代替最终安装包的同机双运行验收。

旧 `Build-DevHarness.ps1` 是本地迁移开发工具：会复制设置、凭据、patch 和 workspace registry，刻意不复制 sessions、attachments、projection cache、node_modules 或远端配对状态；workspace 记录仍指向真实目录。不要用它制作公共发行 profile，干净构建使用 Agent 入口。

## 发布边界

tag 有 `Build-Agent.ps1`、`Initialize-Agent.ps1` 和 `Run-Agent.ps1`。后续分支的 preview EXE builder、完整应用更新器和浏览器测试结果不属于本 tag。默认 profile 中特定模型的 title/distill 覆盖仍需公开发行审查；“无凭据快照”不等于所有模型/机器都适用。

用户要求的整包更新确认、第三方服务主动开启、引擎/手机首次引导及快照迁移是[交付目标](../../roadmap/agent-distribution.md)。源码与改造出处见[迁移记录](../../evidence/changes/product/dsh-source-and-local-migration.md)。不将个人 IP、凭据、会话和记忆库作为发行默认值。
