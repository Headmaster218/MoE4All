---
kind: architecture
status: current
scope: model-runtime-families
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 模型执行路径

## 从 GGUF 到执行图

`infr-gguf` 读取分片和张量；`infr-llama/src/arch.rs` 保存 GGUF 的 `general.architecture` 名称，`config.rs` 解析各架构元数据。模型加载时，`seam/weights.rs` 按架构绑定权重，`seam/runner.rs` 以共享算子及架构分支构造执行图。图的 `Op`、`Graph` 和后端接口定义在 `infr-core`，CPU、Vulkan 和 Metal 实现各自的执行与分配。

```text
GGUF 架构名与元数据
  -> Config + 权重绑定
  -> Seam runner 构图
  -> Backend 执行
  -> 每会话 KV / recurrent state / sampler
```

架构名称只决定该模型可进入哪条构图路径；能加载某个 GGUF 不等于该量化、后端或上下文组合已通过数值与长时运行验收。[能力矩阵](../../reference/model-capabilities.md)给使用范围，[历史测试](../../evidence/benchmarks/README.md)给特定硬件上的证据。

## 主要家族

| GGUF 架构 | 在共享运行器上的差异 | 需要特别核对的状态 |
| --- | --- | --- |
| `llama`、`qwen2`、`qwen3`、`qwen3moe` | 标准注意力、RoPE 与稠密或专家 FFN | KV、路由专家 |
| `qwen35`、`qwen35moe` | DeltaNet 与完整注意力混合；MoE 变体有路由及 shared expert | KV 加循环状态；不能当作 `qwen3next` |
| `qwen4exp` | Qwen3.8 Flash Next 的 QSA、PLE、多路残差与 MoE | KV、QSA 选择、循环状态；MTP 另有 head 状态 |
| `bailingmoe3` | Ling 3.0 的 KDA/MLA 与专家层 | KDA 状态、压缩注意力缓存 |
| `deepseek2`、`deepseek32`、`deepseek4` | MLA、索引器或 V4 压缩注意力与 Hyper-Connection 按架构分别构图 | 缓存几何与量化布局不能互相套用 |
| `gemma3`、`gemma4`、`diffusion-gemma` | 滑动注意力、异构层或块扩散 | DiffusionGemma 不使用普通逐 token Decode 语义 |

`deepseek32` 和 `deepseek4` 在 `seam/runner.rs` 均有权重绑定及图发射路径。`arch.rs` 中关于这两者“只有 LOAD、建图拒绝”的注释是早期阶段遗留；判断 0.9.0 的实际路径应看构图实现及对应的[DeepSeek V4 历史结项](../../evidence/campaigns/deepseek-v4/2026-08-24-rx7900xtx-closeout.md)，不能单凭旧注释。

## 请求与后端边界

`infr-chat` 用模型自带 chat template 渲染消息。终端交互由 `infr-llama/src/chat/` 的共享对话层组织，HTTP 请求经 `infr-server` 进入引擎。`ParallelSeam` 负责并发槽位、Prefill/Decode 排程；MTP 的草稿、验证与回退由 `infr-llama/src/mtp/` 及调度器协调。每个槽位的 KV、循环状态和采样状态独立，模型权重共享。

Vulkan 有统一 VRAM arena、专家分页及辅助模型共存路径；CPU 可作为数值参照；Metal 有独立硬件和内核覆盖。是否支持某个组合，需要同时检查执行图、后端 kernel、资源预算和实测记录，不能由架构名推出。
