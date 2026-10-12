---
kind: guide
status: current
scope: agent-development
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# Agent 源码开发入口

这不是正式 Agent EXE 安装说明。先获取根仓库固定的 DSH 与八个插件 submodule，按[源码构建](../development/building-from-source.md)准备环境。

```powershell
git submodule update --init --recursive
powershell.exe -NoProfile -ExecutionPolicy Bypass -File harness/scripts/Build-Agent.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File harness/scripts/Run-Agent.ps1
```

默认 Web 地址为 `http://127.0.0.1:17373`，独立状态在 `harness/agent-state/`。默认连接已有 `http://127.0.0.1:1234/v1`，不应自动启后端；不运行本地引擎也能构建/检查 UI。首次初始化可单独使用 `Initialize-Agent.ps1`，现有配置会保留。

保持 MoE4All 与生产 DSH 的 home、agents、port 独立。不要将生产 profile 目录交给 Agent 脚本；外部 workspace 分享须主动授权。不要在发布构建中使用会拷贝生产凭据的旧 DevHarness 初始化路径。

`harness/1-Build-or-Initialize.cmd` 与 `2-Start.cmd` 是早期 DevHarness 两入口，等价于 Build/Run-DevHarness，并非干净公共发行入口。`-RefreshConfig` 会主动刷新生产设置到隔离开发环境，普通增量构建不会覆盖既有配置。

内置插件源码由 gitlink 固定，不能从市场最新版临时拼装后仍宣称同一发布基线。成品验收需确认 loader 依赖到位、并发双应用隔离、默认模型提供商、手机审批、定时任务及升级确认流程。更多边界见[当前产品架构](../architecture/product/agent-distribution.md)。
