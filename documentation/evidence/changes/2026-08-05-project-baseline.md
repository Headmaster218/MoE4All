---
kind: change-history
status: historical
scope: early-engine-plan
source_date: 2026-08-05
evidence_level: dated-plan-and-source-history
---

# 2026 年 8 月初的引擎计划与后续取舍

早期项目计划同时包含 2026-08-05 的代码树快照和后续候选方向，不能整篇当作 `release-0.9.0` 的架构。它提出 Vulkan 优先、CPU reference 和原生 Metal 共用语义 `Backend`/`Graph`/`Op` 边界，模型与 HTTP 服务不依赖具体 GPU API。最初 DiffusionGemma MVP 完成前，自回归模型家族已经先落地；当时的模型来源由独立仓库设想收敛为本地 GGUF 与共享 Hugging Face Hub 缓存，Ollama 注册表客户端不再是路径。Qwen3.5 的旧 MTP 虽有实现，阶段内暂停启用；它不能替代[0.9.0 的 Qwen3.8 MTP](../../architecture/runtime/qwen38-mtp.md)。

模型接入当时形成的顺序是检查真实 GGUF 元数据与张量目录、对照固定版本的参考 builder、区分仅元数据/权重形状/新算子三类差异、先做 CPU 数值路径，再补 Vulkan/Metal parity 和端到端验收。此流程仍有用，现行模块与验收门槛以[模型接入指南](../../development/model-porting.md)为准；旧计划里的本机绝对路径、源文件行号和旧索引不作为执行命令。早期可见的里程碑包括共享缓存拉取、GGUF 加载、CPU/Vulkan/Metal 后端、流式 `run`/`serve`、DiffusionGemma 去噪与工具回合；性能迭代没有“完成”状态。

当时的候选顺序是先 DeepSeek MLA（可从小模型分阶段验收），再考虑较接近现有 MoE FFN 的 GLM-4.7-Flash / Ernie 4.5，最后考虑需要 Mamba2 SSM、参考实现也较弱的 Nemotron-Nano/H。这是旧投资假设，不表示后三者已被 0.9.0 支持或仍按原顺序排期。`VK_NV_cooperative_vector` 曾被列为 Decode GEMV 研究线索，但阶段记录称 RADV/Mesa 26.1.4 未公开该扩展，未形成实测优化；未来重启应重新探测驱动而非沿用旧结论。safetensors 在该快照未实现，不应从当年的“待办”推断今天的格式能力。

持续有效的风险是量化 matmul 对 CPU 参照的一致性、图抽象不能强迫逐 op 同步，以及三个后端对同一 op 的覆盖不能只靠一个后端的 golden。当前能力与源码职责分别以[模型能力矩阵](../../reference/model-capabilities.md)和[代码库地图](../../architecture/codebase-map.md)为准；早期计划的 crate 数量、目录图和候选模型列表不覆盖它们。
