---
kind: architecture
status: current
scope: codebase
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 代码库地图

`release-0.10.0` 源码树仍有 17 个 Rust crate，另外固定了 DSH 与八个插件的 Git submodule。下面按实际职责归类。

| 包 | 当前职责 | 依赖边界说明 |
|---|---|---|
| `infr-core` | 后端/图/张量、配置、预算、分页器、主机内存层、资源跟踪与有界 timeline | 不放具体模型或产品 Agent 状态 |
| `infr-gguf` | GGUF 元数据、张量和分片读取 | 格式层，不决定执行调度 |
| `infr-hub` | 模型解析、下载、缓存和分片 | 不加载模型运行时 |
| `infr-chat` | 聊天模板、消息、工具、流式增量 | 不是 Agent 会话数据库 |
| `infr-llama` | 多架构模型、权重、会话、KV、采样、并发调度器、MTP | 当前最大职责集中区 |
| `infr-vulkan` | Vulkan 后端、内核降级、记录/提交、传输、分页器、统一 arena 与可选 CPU miss 协调 | 物理执行与内存后端 |
| `infr-cpu` | CPU 参考后端、优化内核及私有任务池 | 正确性预言机和无 GPU 路径 |
| `infr-metal` | Apple Metal 后端 | 平台独立后端，需真实硬件验收 |
| `infr-server` | HTTP DTO、路由、鉴权、SSE、请求管理 | 不应拥有 GPU 具体实现 |
| `infr-engine` | 当前主要重导出聊天契约 | 此表只描述 tag 中的现状 |
| `infr-cli` | 命令、运行时装配、基准测试、终端呈现 | 当前不是薄入口，后续可内部拆模块 |
| `infr-embedding` | 原生/兼容嵌入引擎 | 通过资源合同与 LLM 共用显存 |
| `infr-vision` | 投影器、图像解码/预处理、HEIC/AVIF 动态库入口和 Vulkan 执行 | 请求级权重驻留 |
| `infr-gui` | 浏览器控制面、目录、估算、下载、工作进程生命周期 | 现有产品能力来源，不只是前端 |
| `infr-prof` | 构建期插桩宏 | 独立过程宏边界 |
| `infr-prof-rt` | 性能分析运行时 | 与宏分离 |
| `infr-testkit` | 共享测试支持 | 隔离仅开发依赖和环依赖 |

## 当前依赖方向

```text
CLI / GUI
  |
  +--> server / engine contracts
  +--> model runtime and auxiliary engines
             |
             +--> core + gguf
             +--> CPU / Vulkan / Metal

```

## Rust 树之外

- `Start-INFR-Wizard.cmd`、`scripts/infr-wizard.ps1`：Windows 向导与引擎更新。
- `Start-INFR-Wizard-Linux.sh`：单文件 Linux 向导与更新；发包工具见 `scripts/linux_release.py`。
- `scripts/build-image-codecs.py`：固定版本的原生图片解码库与源码附件构建。
- `harness/dsh`、`harness/plugins/*`：品牌化 DSH 与维护插件，版本由主仓库 gitlink 固定。
- `harness/scripts/Build-Agent.ps1`、`Initialize-Agent.ps1`、`Run-Agent.ps1`：独立状态的 Agent Web 开发入口。
- `harness/scripts/Build-DevHarness.ps1`、`Run-DevHarness.ps1`：从已有本机 DSH 设置建立开发测试环境，与干净 Agent 初始化用途不同。

CPU miss 从 Vulkan 路由任务调用 `infr-cpu::task_pool`；它没有把整个模型迁移到 CPU 后端。边界见[CPU miss](runtime/cpu-miss-offload.md)。Agent 与引擎使用进程/API 边界，见[产品架构](product/agent-distribution.md)。
