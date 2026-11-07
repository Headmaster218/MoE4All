---
kind: development-guide
status: mixed
scope: benchmarking-methods
source_checkpoint: release-0.9.0
---

# 基准测试与性能剖析

`infr bench` 与 `llama-bench` 的 `-p`/`-n`/`-d`/`-r` 标志对应，因此两者可直接比较。管线会在模型加载时编译并首次触及 GPU 状态（`Llama::warmup`），所以计时测量的是计算而非一次性设置。**请一次只运行一个基准测试**，并发 GPU 工作会扭曲结果。

```bash
M='unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M'   # MoE 性能测试目标

# Prefill（pp = n_prompt/time）与 Decode（tg = n_gen/time）：
infr bench "$M" -p 2048 -n 0 -r 3       # Prefill 2048 个 token
infr bench "$M" -p 8000 -n 0 -r 2       # 在指定上下文深度下测 Prefill
infr bench "$M" -p 0 -n 64 -r 3         # Decode 64 个 token
infr bench "$M" -p 0 -n 64 -d 2048      # 在 2048 上下文深度下测 Decode（-d 只预热，不计时）
```

## 数字口径与可比性

`ppN` 是 N token Prefill，`tgN` 是 N token Decode；`dK` 是测试前已有的 KV 深度。`--synthetic-depth` 会构造并初始化测试用 KV，让 Attention 扫描指定深度，却没有真正执行此前 K token 的文本 Prefill，也不能验证长对话质量或真实专家路由。报告 `cache` 大小时须说明它是旧专家缓存限制还是总 VRAM budget；命中率须注明 GPU hit、RAM 条件 hit 或其他分母。GB/s 与 GiB/s 不可混写。

比较代码变更时至少固定 GGUF/量化、KV 类型、上下文与深度、VRAM/RAM 预算、ubatch、Prefill/Decode token 数、提交上限、预热与 profiler 状态；优先交错 A-B-B-A，并报告逐次样本及离散程度。区分新进程冷启动、benchmark 预热、连续会话和稳态。逐算子时间戳、有序 Pager CSV 与 RGP 捕获用于归因，最终 tok/s 应在关闭插桩后复测。原始记录还应包含提交或二进制 hash、GPU/驱动、完整命令及产物位置；缺项时只标历史样本，不补造可复现性。

例如，122B 冷进程 2K 的 11.2 tok/s 与热缓存 tg256 的 23.2 tok/s 不是补丁 A/B，不能据此宣称 107% 提速；35B 的 14 GiB 纯 Decode 数字也不能证明 250K 对话可安全运行。模拟的槽位命中、理论 `bytes / bandwidth` 和硬件实测必须分别标注。

使用 `INFR_PROF_OPS=1` **剖析**逐算子 GPU 时间（时间戳查询）。每次派发都会自动加时间戳并以**内核名称**标记（另有 `expert_gateup`/`expert_down` 等少数角色覆盖），无需手动标记。它会为每次提交输出一个区块，并在进程退出时输出一份汇总的 `INFR_PROF_OPS GPU report`（各内核总计、次数、平均值，以及所有计时提交中的 GPU 百分比；预热运行不做剖析）。添加 `INFR_PROF_OP_SHAPES=1` 可获得按形状细分的 GEMV/GEMM 桶（`mmvr:m4:1536x24576`）。解码重放带不携带时间戳；使用 `INFR_SEAM_NO_REPLAY=1` 剖析解码。详见 [`playbook.md`](optimization-playbook.md)。

```bash
INFR_PROF_OPS=1 infr bench "$M" -p 2048 -n 0 -r 1 2>&1 | tail -30   # 查看退出时的汇总结果
```

**验证 Vulkan 工作**：任何触及 `infr-vulkan`（内核、记录器、适配器、分页器）的变更，必须在 Khronos 验证层下运行其 GPU 测试和至少一次端到端生成，并在合入前修复它报告的每个错误和警告（标准是验证层无输出，不是“它产生了正确 token”；健壮访问读取、缺少屏障和绑定范围溢出都可能返回看似合理的垃圾结果，而不是崩溃）：

```bash
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation cargo test -p infr-vulkan -- --ignored
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation infr run "$M" "smoke prompt"
```

该层由 `vulkan-validation-layers` 包提供。它会明显降低 GPU 工作速度；请用于正确性检查，绝不要放入计时基准测试中。

**与 llama.cpp 比较**：`infr compare` 会以匹配标志在编程代理形态的工作负载（预填充、指定深度的解码、完整轮次）上调用 `infr bench` 和系统 `llama-bench`。`--ctx` 以逗号分隔：

```bash
infr compare "$M" --ctx 8000,16000 --gen 256 --turn 2048,256 --reps 2
```

**DiffusionGemma** 尚未合入上游 `llama-bench` 支持，因此 `infr compare` / `infr compare --sweep` 会将 `arch=diffusion-gemma` 模型交给另一套对比工具：参考分支中的 `llama-diffusion-cli`（`~/Projects/mxaddict/llama.cpp-dg`，按 `INFR_LLAMA_DIFFUSION_CLI`、`PATH`、该分支的 `build-vulkan`/`build` 目录顺序查找；具体优先级与 PATH 回退注意事项见 `ModelBench::llama_diffusion_cli_path`）。它不会输出常见的 pp/tg 矩阵，而是输出两行：`dg-step`（单个去噪步骤内的并行 tok/s 比值；双方去噪步数不同但都受熵约束，因此该数值更适合公平比较）以及 `dg-e2e`（端到端 tok/s，仅供参考；双方各自的去噪步数都会计入，以体现总耗时差异）。详见 [DiffusionGemma 架构](../../evidence/changes/models/diffusion-gemma.md)。

有用的调节项：`--temp` / `--top-k` / `--top-p`（采样；`--temp 0` → 贪心）、`--max-new`、`--ctx`，或者 `sampling.*` / `device.*` 配置路径，或其 `INFR_*` 对应项。参见[配置参考](../../reference/configuration.md)。

**MoE 专家放置**：专家权重库可装入 VRAM 时保持常驻（零配置、零变更）；否则每层通过按剩余 VRAM 定尺寸的 VRAM 常驻 LRU 专家缓存（`infr_vulkan::pager`）进行分页。无论是否可装入，`INFR_CACHE=<size>` 都会以该预算强制每层经过分页器（适用于测试，或为更大上下文释放 VRAM）。所有权重库形状都会分页：拆分 gate/up（llama4/Qwen3-MoE/Qwen3.6-MoE）、融合 gate_up（DiffusionGemma、Gemma-4 MoE，每位专家占一个双宽槽位）和混合 dtype 角色（unsloth-dynamic 量化会将部分层的权重库提升为更宽 K 量化；每种专家字节大小一个逻辑内存区池，在兼容角色间共享）。`INFR_PAGER_STATS=1` 输出每个池的命中/未命中/驱逐计数。

如需限制整个进程，请改用 `INFR_VRAM_BUDGET=<size>`。它同时涵盖常驻权重、KV、运行时工作区和分页内存区；`INFR_VRAM_RESERVE=<size>` 会在 Vulkan 内置保护之上额外保留物理 VRAM。旧版 `INFR_CACHE` 覆盖仍会强制分页，但会被限制为此统一计划的余量。MoE 预填充期间，整层流式加载环只从解码专家内存区借用所需的最冷连续范围。不重叠的热门专家条目会跨越预填充→解码转换，借用的槽位无需第二次分配即可回归普通 LRU。

**稠密层流式加载**：大于 VRAM 的稠密模型会通过同一分页 VRAM 机制流式加载每层投影权重（注意力 q/k/v/o + FFN gate/up/down，即加载器上传的同一融合 qkv/gate_up 组），但由调度驱动而非 LRU：稠密前向按固定顺序访问各层，因此常驻使用精确的循环扫描策略（Belady 一致性：稳定常驻前缀加每池一个周转槽位），且任何位置都**没有读回**（每次“未命中”均可预知；未命中会经由与 MoE 路径相同的流水线半栅栏暂存环中已记录的环→内存区拷贝，因此后续层的 CPU memcpy 与先前层的 GPU 执行重叠）。流式派发是普通稠密内核，以槽位元素偏移量（`w_off` 约定）读取池内存区；没有内核变体，因此流式输出与常驻运行 token 一致。嵌入、lm_head、归一化和偏置保持常驻（lm_head 在每个 token 边界读取，流式加载会将其完整字节数计入每个 token 的 PCIe 成本，而没有任何可利用局部性）。放置自动完成（全部装入时常驻，零变更）；`INFR_CACHE=<size>` 会以该预算强制流式加载。合理预期是：预填充在整个批次中摊销上传（Qwen3-14B Q8_0、约 15.7 GB、`INFR_CACHE=8g` 时：pp512 987 t/s，相对常驻 1505 为 0.66×）；解码没有可利用局部性，因此上限为每 token 的 PCIe_bw ÷ overflow_bytes，这是物理限制而非缺陷（相同设置：每 token 重新上传约 7.0 GB ÷ 约 22 GB/s ≈ 3.1 t/s 上限，实测 3.1 t/s；在约 45% 溢出时 CPU 后端为 4.4 t/s，所以只有溢出较小时流式加载才胜过 CPU；该机器的实测交叉点约为模型四分之一溢出）。MoE 模型若其稠密部分也无法装入，则不在范围内并会明确报错。

**大小语法**：`paging.cache` / `INFR_CACHE`、`device.vram_budget` / `INFR_VRAM_BUDGET`、`device.vram_reserve` / `INFR_VRAM_RESERVE` 和 `device.ctx` / `INFR_CTX` / `--ctx` 共用一套取值语法（`infr_core::parse_size`）：纯数字表示基准单位（内存预算为字节，`INFR_CTX` 为 token），`k`/`m`/`g`/`t` 后缀按 1024 缩放（`INFR_VRAM_BUDGET=23g`、`INFR_CTX=256k`）。`%` 的总量/保留预算相对设备总内存解析；`INFR_CACHE=%` 保持其历史可用 VRAM 基数；上下文百分比在 Vulkan 上使用空闲 VRAM KV 容量（在 CPU/Metal 聊天路径上使用训练上下文）。

**常驻 BDA 权重内存区**：始终启用，也是唯一的权重路径。它将每次权重分配路由到单一 `bufferDeviceAddress` 内存区，让内核通过 64 位设备地址而非每张量 SSBO 描述符绑定读取权重。稠密投影权重和 MoE 专家权重库经由 `-DSTREAMED` 内核孪生体读取，子张量经子范围描述符绑定读取，分页专家缓存可在其上保持不变地组合。该寻址在整个模型集合中与已退役的 u32-SSBO 描述符路径逐位一致（稠密、MoE、qwen35/DeltaNet、DiffusionGemma 和分页 Scout 专家；由 `gpu_seam` 黄金值和流式一致性测试套件证明），并且在 RDNA3（7900 XTX）的稠密和 qwen3-MoE 路径上运行速度不低于该路径。
