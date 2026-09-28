---
kind: architecture
status: current
scope: system
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 系统总览

## 产品形态

MoE4All 当前以独立 Rust 推理 worker 为核心，提供四类入口：

- Windows Wizard 组合启动参数；
- `infr run` 终端对话；
- `infr serve` / `infr multi` OpenAI-compatible 服务；
- `infr bench` / `infr compare` 性能与参考对照。

`infr-gui` 是现有浏览器控制面，负责目录、估算、下载和 worker 生命周期。Agent/DSH harness 不属于
`release-0.9.0` tag；后续开发分支加入了相关源码和开发环境，但不能据此描述为 0.9.0 已交付能力。版本边界见
[提交审计](../evidence/audits/2026-09-28-release-0.9.0-baseline-and-new-commits.md)。未来产品边界另见[Agent 发行版提案](../roadmap/agent-distribution.md)。

## 请求路径

```text
CLI / Wizard / HTTP
        |
        v
chat template + request validation
        |
        v
model admission / scheduler
        |
        +--> Prefill
        +--> Decode
        +--> MTP draft + batched VERIFY
        |
        v
backend-neutral Graph
        |
        +--> Vulkan
        +--> CPU reference
        +--> Metal
```

HTTP DTO、鉴权、SSE 和错误映射属于 `infr-server`；0.9.0 同时提供 Chat Completions 与无状态 Responses API。模板、工具和 reasoning/content 分流属于 `infr-chat`。
模型、session、scheduler 与 MTP 主要仍在 `infr-llama`。CLI 负责把这些模块装配成实际命令。Vulkan `serve` 通常由
`ParallelSeam` 统一调度；仅单 slot、纯文本且没有 Vision/Embedding 的 Qwen3.8 MTP serve 保留传统单请求路径。
并发 token group 的 lane 顺序优先多模态 lane，再优先剩余 Prefill 较长的 lane，最后以 slot 稳定排序；这是共享图形状
与请求推进的调度顺序，不代表长 Prefill 期间会插入 Decode。

## 模型与会话

模型级固定权重跨请求共享。每个并发 slot 有独立的 KV、recurrent state、采样和会话前缀。
Qwen3.8 compatible rows 可按层批量执行，但位置、QSA selection、PLE/recurrent state 和输出仍分别维护。Qwen3.8 concurrent
MTP 共享 target/head 权重、每 slot 独立 MTP state，当前最多两个 slot；请求不符合 greedy-compatible 条件时回退普通 Decode。

冷 KV session cache 是引擎状态优化，不是聊天记录数据库。恢复失败最多导致重新 Prefill，不应丢失 Agent 或用户会话事实。

## 资源层

启动顺序是架构约束：

1. 在 `ParallelSeam` 路径加载并真实分配固定权重和固定 runtime 资源（含启用时的 MTP sidecar/runtime）。
2. 查询设备真实剩余空间。
3. 建立统一 VRAM arena。
4. 用 Expert filler 填充未被 KV、runtime、Prefill、Vision 或 Embedding 占用的空间。

RAM 可承载完整 host store 或 bounded inclusive cache；SSD/GGUF 是 immutable expert 权重的最终真源。
Host DMA、ReBAR CPU push 或 staged upload 由硬件能力和冻结的 transfer plan 选择。

## 关键不变量

- 长 Prefill 独占其 ring 和专家重建阶段，完成后再恢复 Decode。
- 每个 slot 的 KV/recurrent/MTP state 独立。
- frozen LUT window 可引用的物理 Expert slot 在相关 command stream drain 前不能移动。
- 高优先级 runtime claim 发布前必须完成 victim/move/目录更新，失败时完整回滚。
- 用户会话事实属于上层产品；引擎 session state 可丢弃并重建。

## 继续阅读

- [代码库地图](codebase-map.md)
- [并发调度](runtime/parallel-scheduler.md)
- [运行时资源生命周期](memory/runtime-resource-lifecycle.md)
- [服务与冷 KV 会话](services/server-and-session-cache.md)
- [Agent 发行版边界](../roadmap/agent-distribution.md)
