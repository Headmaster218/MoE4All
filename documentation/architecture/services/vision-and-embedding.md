---
kind: architecture
status: current
scope: auxiliary-engines
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 视觉与嵌入

## 共同资源边界

Vision 和 Embedding 有独立模型逻辑，但在 Vulkan 模式下可从运行中的 LLM backend 派生 client，并使用同一个 unified
VRAM pool。它们不能绕过资源管理器再建立一份不可见的 GPU 模型栈。

## 视觉

当前原生路径面向 Qwen3-VL-style CLIP projector 的受限子集：

- 解析 mmproj GGUF；
- 解码 data URI 或 base64 image；
- resize、patch 与位置预处理；
- 请求级载入 projector weights；
- 生成 embedding 并注入文本 runtime；
- image batch 完成后释放临时 VRAM。

不支持的 projector/deepstack 或远程 HTTP image URL 应明确失败，避免静默降级和 SSRF 面。

## 嵌入

原生 Embedding engine 与 llama.cpp compatibility worker 共享一个上层 contract。原生 Vulkan 模式可以：

- 第一次请求从 GGUF/SSD 加载权重；
- 在 idle timeout 内复用；
- 超时后释放权重并恢复 Expert filler；
- 通过 `/v1/embeddings` 返回 normalized vectors 和 token usage。

## 并发原则

- Active LLM、Vision 和 Embedding request 都通过统一 owner/lease 保护资源。
- 辅助模型不能回收正在执行的 LLM/MTP runtime range 或 frozen Expert slot。
未来 Agent 插件如何接入 Embedding 由[Agent 发行版提案](../../roadmap/agent-distribution.md)定义；本页只描述 0.9.0 引擎的共享显存行为。
