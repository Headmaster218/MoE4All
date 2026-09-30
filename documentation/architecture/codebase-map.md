---
kind: architecture
status: current
scope: codebase
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 代码库地图

`release-0.9.0` 源码树有 17 个 crate。下面按实际职责归类。

| 包 | 当前职责 | 依赖边界说明 |
|---|---|---|
| `infr-core` | 后端/图/张量、配置、预算、分页器、主机内存层、资源跟踪 | 不放具体模型或产品 Agent 状态 |
| `infr-gguf` | GGUF 元数据、张量和分片读取 | 格式层，不决定执行调度 |
| `infr-hub` | 模型解析、下载、缓存和分片 | 不加载模型运行时 |
| `infr-chat` | 聊天模板、消息、工具、流式增量 | 不是 Agent 会话数据库 |
| `infr-llama` | 多架构模型、权重、会话、KV、采样、并发调度器、MTP | 当前最大职责集中区 |
| `infr-vulkan` | Vulkan 后端、内核降级、记录/提交、传输、分页器、统一 arena | 物理执行与内存后端 |
| `infr-cpu` | CPU 参考后端与优化内核 | 正确性预言机和无 GPU 路径 |
| `infr-metal` | Apple Metal 后端 | 平台独立后端，需真实硬件验收 |
| `infr-server` | HTTP DTO、路由、鉴权、SSE、请求管理 | 不应拥有 GPU 具体实现 |
| `infr-engine` | 当前主要重导出聊天契约 | 此表只描述 tag 中的现状 |
| `infr-cli` | 命令、运行时装配、基准测试、终端呈现 | 当前不是薄入口，后续可内部拆模块 |
| `infr-embedding` | 原生/兼容嵌入引擎 | 通过资源合同与 LLM 共用显存 |
| `infr-vision` | 投影器元数据、预处理和 Vulkan 执行 | 请求级权重驻留 |
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

`release-0.9.0` 源码树是引擎发行版，不包含后续 DSH/插件 submodule。较新的 harness 工作只作为 tag 后历史记录，见[发行基线审计](../evidence/audits/2026-09-28-release-0.9.0-baseline-and-new-commits.md)，不计入此代码地图。

后续 Agent/DSH 与 crate 边界调整属于[发行版设计提案](../roadmap/agent-distribution.md)，不属于本页的现状依赖图。
