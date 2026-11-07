---
kind: reference
status: current
scope: model-capabilities
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 模型与组合能力

此表描述 `release-0.9.0` tag 中的主要能力，不表示任意 GGUF 转换都已验证。实际可用性还取决于元数据、张量形状、量化、聊天模板、后端和当前配置。

| 模型系列 | GGUF 架构 | 主要执行路径 | 特殊能力或限制 |
|---|---|---|---|
| Llama / Llama 4 | `llama`、`llama4` | 稠密 / MoE | Llama 4 的部分路由语义仍可能限制 GPU 执行路径 |
| Qwen2/2.5/3 | `qwen2`、`qwen3`、`qwen3moe` | 稠密 / MoE | 支持 Vulkan、CPU 参照实现及部分 Metal 执行路径 |
| Qwen3.5/3.6 | `qwen35`、`qwen35moe` | DeltaNet、完整注意力、MoE | 分段状态与 KV；旧版 Qwen3.5 MTP 状态见专题历史页 |
| Qwen3.8 Flash Next | `qwen4exp` | QSA、PLE、DeltaNet、Hyper-Connection、分页式 MoE | Vulkan 文本推理、并发、视觉、四 token MTP |
| Gemma 3/4 | `gemma3`、`gemma4` | 稠密、MoE、E2B | 具体组合取决于后端能力 |
| Ling 3 Flash / Ling 3.0 Tiny | `bailingmoe3` | KDA、门控 MLA、MoE | 包含 Ling 3.0 Tiny Q4_K_M 支持；面向 RAM/SSD 分层分页的大模型执行路径 |
| DeepSeek | `deepseek`、`deepseek2`、`deepseek32`、`deepseek4` | MLA、索引器、Hyper-Connection、MoE | 各代差异较大，详见模型专题 |
| DiffusionGemma | `diffusion-gemma` | 块扩散 | 执行方式不同于常规自回归解码 |
| GGUF 嵌入模型 | 受支持的嵌入架构 | 原生 CPU / Vulkan | 可独立提供服务，也可与 LLM 共用统一显存 |

## Qwen3.8 组合

| 能力 | 当前基线 |
|---|---|
| 普通单路解码 | 支持 |
| 普通多槽位 | 支持持久工作线程与兼容行批处理 |
| 四 token MTP | 支持 Qwen3.8 辅助模型及批量 VERIFY |
| 多槽位 MTP | 支持两槽位验证与批量解码；更多槽位仍需单独验证 |
| 视觉 | 支持 Qwen3-VL 风格投影器的受限子集，仅 Vulkan |
| 视觉 + MTP | 0.9.0 服务端支持组合运行；具体模型、输入和资源压力仍需按目标发行版矩阵验收 |
| 嵌入共存 | 支持按需借用统一显存；与高压 MTP/并发组合应单独验收 |
| 冷 KV 会话缓存 | 动态分段 Q8 KV 下可选 |

面向用户的开关和发行版承诺应以对应版本的 README、CHANGELOG 和验收记录为准。
