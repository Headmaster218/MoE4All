---
kind: development-guide
status: current
scope: gguf-model-porting
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 接入新的 GGUF 模型家族

0.9.0 的模型家族不是运行时插件：GGUF 元数据进入 `infr-llama` 的 `Config`，再由权重绑定与共享 runner 选择图分支。先建立可检验的 CPU 数值路径，再处理 Vulkan/Metal；否则构图错误与 shader 错误难以分辨。当前边界见[模型执行路径](../architecture/models/runtime-families.md)。

1. **检查真实 GGUF。** 导出元数据和张量目录，记录 `general.architecture`、键名、dtype、shape、chat template 和特殊 token。与最接近的已支持家族比较，不根据模型名称猜架构字符串。
2. **固定参照实现。** 阅读匹配版本的 llama.cpp 模型 builder 与转换器，将差异分为“仅元数据”“权重名称/形状”或“需要新算子”。参考实现、GGUF 文件与转换器版本都写入验收记录。
3. **解析并绑定。** 在 `crates/infr-llama/src/arch.rs`、`config.rs` 注册和解析架构；在 `seam/weights.rs`、`seam/runner.rs` 绑定逐层权重。其他模型不得无条件读取新张量。
4. **先构 CPU 图。** 优先复用现有 `Op`。确需新 op 时，在 `infr-core` 声明完整的读写依赖，在 CPU 后端实现，再为 Vulkan/Metal 增加对等实现；缺少后端必须明确拒绝，不能静默跳过。
5. **验证 token 边界。** 模板和 tokenizer 来自 GGUF，但预分词器、EOS/回合结束 token、采样默认值仍要核对。连续续写、重复符号和“数值有限但文本异常”都不能用一个通过的 hash 掩盖。
6. **由小到大验收。** 先用固定输入对比 CPU logits/top-k，再测 CPU 文本金样、CPU/GPU 同模型 parity，最后做真实端到端生成、长上下文和 benchmark。MoE 路由接近并列时，CPU f32 与 GPU f16 可出现合法差异，应同时看输出及数值容差。

新架构还需检查后端 kernel、显存预算、KV/recurrent state 生命周期和流式传输；“能 LOAD”不等于“能端到端运行”。验收后更新[模型能力矩阵](../reference/model-capabilities.md)、根 README 和 CHANGELOG，并把失败实验留在对应 campaign。早期 DeepSeek 分阶段接入的取舍见[变化记录](../evidence/changes/models/deepseek-family.md)。
