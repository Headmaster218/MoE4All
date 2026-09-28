# infr-train：LLM 训练支持（提案）

本提案基于截至 2026 年 7 月 6 日对接口层和后端代码的评估，以及对 Rust 训练生态的调研。目前尚未实现相关功能；本文记录设计判断和分阶段计划。

**核心判断：** 在当前工作区新增一个与现有 crate 同级的 `infr-train`，采用 llm.c 风格的手写反向传播，并复用 `infr-core` 的接口层。初期不引入通用的、基于计算记录的自动微分系统，也不依赖外部训练框架。首先实现 **CPU 后端上的 LoRA 微调**，再扩展到 Vulkan 上的 QLoRA。

---

## 为什么采用这种设计

### 代码评估结果

infr 目前专注于前向计算和推理解码。代码中没有梯度、反向传播或优化器实现；各后端的矩阵乘法也只计算 `Y = X·Wᵀ`，训练所需的转置方向计算（`dX = dY·W`、`dW = dYᵀ·X`）尚未实现。

现有接口层仍然适合作为训练能力的基础：

- IR（`infr-core::graph`）是由语义明确的复合操作组成的有序列表，并使用强类型张量句柄。反向传播可以逆序遍历计算图并发出梯度操作。这与 llm.c 手写反向传播的组织方式相似，而 infr 的计算图已与具体后端解耦。
- **原地别名既合法也很常见**，例如原地 RoPE、KV 写入和临时缓冲区复用；`Graph::in_place_inputs()` 正是为此存在。直接对现有计算图应用朴素的反向模式自动微分会遇到别名问题。因此，训练图构建器必须为每个操作显式生成反向操作，并避免覆盖反向传播仍需使用的激活值。
- CPU 后端可以作为 **参考实现**：它支持当前实现的架构和量化格式，并可直接读取 GGUF mmap 数据而无需复制。可先在此实现和验证反向内核，再用相同的模式验证 GPU 后端；推理端已经采用了类似做法。
- 可以直接复用现有组件：`infr-gguf` 的读取功能、`infr-hub`、分词器，以及 `Backend`/`Bindings`/`DType` 等接口。未来每个前向内核也都需要对应的反向内核。

为什么选择新增同级 crate，而不是把训练并入推理实现或拆成独立仓库：

- **不放进 `infr-llama`：** 前向路径针对推理进行了优化，包括只处理单 token 的解码图（n=1）、定量驻留的权重、原地别名和 KV 增长。把训练直接并入会让两套执行路径相互牵连，也会增加维持 CPU/GPU 逐 token 结果一致性的难度。
- **不单独建仓库：** 训练需要与共享组件同步演进，例如在 `infr-core` 中新增 `Op` 变体、为各后端实现反向内核，以及为 `infr-gguf` 增加写入器。若分散在多个仓库，跨仓库协调会持续增加维护成本。

### 生态调研结果

- **手写反向传播已有成功先例。** llm.c 在单个节点上复现 GPT-2 1.6B，速度比当时的 PyTorch 快约 7%。其关键做法包括约 20 个手写反向函数、融合内核和预分配内存区域，不依赖通用自动微分系统。Rust 移植项目（`llm.rs`、`llm.rust`）也印证了这种实现规模（数千行代码）以及接近 C 的性能。
- **llama.cpp 的训练实现提供了警示。** `finetune` 和 `train-text-from-scratch` 因计算图重构后失效而被移除（PR #8669）：它们把通用训练图接入推理引擎，却没有持续集成中的反向操作测试，最终难以维护。替代方案 `ggml_opt`（PR #10544）仍处于早期阶段。经验是：训练操作的范围要小，并为每个操作检查梯度；达不到这点就不要发布。
- **现有框架都不完全合适。** burn 是较认真探索 LLM 训练的 Rust 框架，支持 CubeCL Vulkan/Metal 后端和 coopmat 矩阵乘法；但 burn-lm 训练仍是未发布的 alpha 版本。采用它意味着引入第二套张量系统，并重复实现加载器和 GGUF 转换。candle 的量化张量在不使用 autograd 时仍比 PyTorch 慢约 4 倍，而且没有 QLoRA 路径。tch-rs 依赖 libtorch，也不支持 Vulkan。可以参考它们在 SPIR-V 协作矩阵和自动调优分块方面的经验，但不直接采用其训练栈。
- **QLoRA 是更现实的显存起点。** 对 7B 模型，4 位基础权重约需 3.6 GB，适配器约 30 MB，AdamW 状态约 60 MB（r=8–16），激活值约 2–4 GB；启用检查点后，总需求约 **7–9 GB**，可在 12 GB 消费级 GPU 上尝试。冻结的量化基础权重不计算梯度，因此可避免 dequant→grad→requant 的开销，也绕开了 llama.cpp 训练实现遇到的大部分问题。

---

## 需要新增的能力

1. **反向操作**（约 20 种，作为 `infr-core` 中的新 `Op` 变体）：矩阵乘法的两种转置方向（`dX = dY·W`、`dW = dYᵀ·X`）、rmsnorm-bwd、rope-bwd、attention-bwd（反向 Flash Attention 需要 softmax 统计量，而当前前向计算不会保存这些数据）、swiglu/gate-act-bwd、融合 softmax-交叉熵，以及 embedding 分散累加。每种操作都先提供 CPU 参考实现。
2. **支持批量和完整序列的前向计算图。** CPU 接口层目前只支持解码（n=1）；训练需要 `[batch, seq]` 形状，并且必须 **保留激活值**。当前 `Internal` 暂存区在 `execute` 过程中分配，执行结束后即释放。训练图需要把激活值声明为可持续保留的输出。
3. **激活检查点。** 这是训练相较于推理最关键的能力之一：反向传播时重新计算各层块的激活值。没有检查点，7B 模型的激活内存需求难以接受；使用检查点后预计约为 2–4 GB。
4. **优化器：** 使用 AdamW，并为 LoRA 适配器维护 f32 主权重及 f32 的 m/v 状态（7B 模型、r=16 时约 60 MB，开销较小）；同时支持梯度范数裁剪。
5. **检查点写入器：** `infr-gguf` 目前只支持读取，需要增加写入能力，以 GGUF LoRA 和/或 SafeTensors 格式导出适配器，并保存可恢复的训练状态。
6. **梯度检查工具，不可省略。** 每种操作都要有有限差分测试，并进行端到端 PyTorch 一致性验证（数据和初始化相同，损失曲线应相符）。静默的梯度错误是手写反向传播最容易漏掉的问题。

---

## 阶段计划

| 阶段 | 可交付成果 |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0** | 新增 `infr-train` crate；让 CPU 接口层支持批量完整序列前向计算；为 `Linear`/`RmsNorm`/`Attention`/`GatedAct` 实现反向操作；提供有限差分梯度检查工具。 |
| **P1** | 在 CPU 上为小型 llama/qwen3 模型实现端到端 LoRA 微调；验证损失明显下降、适配器可导出并重新加载到 `infr run`，且损失曲线与 PyTorch 一致。 |
| **P2** | 实现 Vulkan 反向内核：coopmat 矩阵乘法强制使用 **f32 累加**（即使 f16 累加适用于推理，在训练中也可能导致结果偏离）；通过 CAS 或私有化归约实现分散累加（浮点原子操作的跨厂商支持并不完整）。目标是在 12 GB 显卡上运行 7B QLoRA。 |
| **P3** | （可选）小模型预训练：使用 bf16 和损失缩放，并融合 softmax-xent。llm.c 的结果表明，在单节点上训练约 1.6B 模型是可行的。 |

**明确暂缓：** 通用的、基于计算记录的自动微分。只有当某个架构系列的反向操作超出手写实现的维护能力时，才考虑增加一个轻量记录器；即便如此，也只用于逐操作持续集成测试，并遵循 llama.cpp 的经验。全参数微调（为所有基础权重计算梯度、执行 dequant→grad→requant，并维护模型规模的优化器状态）属于单独的后续决策，不在本提案范围内。

---

## 代码中的锚点

- `crates/infr-core/src/graph.rs` — 在 `Op` 中增加梯度操作；训练图构建器放在其附近。
- `crates/infr-core/src/backend.rs` — 原样复用 `alloc`/`compile`/`execute` 接口；将激活值改为持久绑定。
- `crates/infr-cpu/src/lib.rs` — CPU 反向内核的参考实现位置。
- `crates/infr-llama/src/seam/{model,runner}.rs` — 前向解码图构建器，可供训练图构建器参考。
- `crates/infr-gguf/` — 复用加载器；新增写入器。
- `crates/infr-hub/`、分词器 — 原样复用。

## 来源

- llm.c：<https://github.com/karpathy/llm.c> — GPT-2 1.6B 复现：<https://github.com/karpathy/llm.c/discussions/677>
- Rust 移植：<https://github.com/ToJen/llm.rs>、<https://github.com/Steboss/llm.rust>、<https://github.com/yijunyu/llm.rs>
- llama.cpp 训练器移除：<https://github.com/ggml-org/llama.cpp/pull/8669>；`ggml_opt` 训练：<https://github.com/ggerganov/llama.cpp/pull/10544>、<https://github.com/ggml-org/llama.cpp/pull/13105>；检查点 RFC：<https://github.com/ggml-org/llama.cpp/issues/15442>
- QLoRA：<https://arxiv.org/pdf/2305.14314>；7B 显存配置：<https://kaitchup.substack.com/p/mistral-7b-recipes-for-fine-tuning>、<https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/>
- burn / burn-lm：<https://burn.dev/blog/>、<https://burn.dev/blog/burn-lm-announcement/>、<https://github.com/tracel-ai/burn>
- candle 训练缺口：<https://github.com/huggingface/candle/issues/1383>、<https://github.com/huggingface/candle/issues/3052>；LoRA：<https://github.com/EricLBuehler/candle-lora>
- Vulkan 与 CUDA 训练性能对比：<https://github.com/ggml-org/llama.cpp/issues/17273>；WebGPU LLM 性能：<https://arxiv.org/pdf/2605.20706>
