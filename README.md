# MoE4All

**让 A 卡用户也能在本地运行大型 MoE AI**
*Making huge MoE LLMs accessible to AMD users.*

[下载最新版 Windows 程序](https://github.com/Headmaster218/MoE4All/releases/latest) |
[快速开始](GETTING_STARTED.md) |
[English](README_EN.md) |
[技术文档](https://github.com/Headmaster218/MoE4All/blob/main/docs/README.md)

MoE4All 是一个面向 AMD 显卡和 Windows 11 的本地大模型运行项目。它让
MoE 模型的专家权重按需在显存、内存和 SSD 之间流动，因此模型不必全部塞进
显存，也能在消费级 A 卡上运行。

无需理解分页、KV Cache 或 Vulkan。下载发布包，准备好 GGUF 模型，
双击启动向导并选择“自动配置”，就可以开始聊天。

> 当前主要开发和实测平台是原生 Windows 11、AMD Radeon RX 7900 XTX 和
> Vulkan。其他 Vulkan GPU 可能可用，但不是 MoE4All 当前的重点验证平台。

## 最新进展：完整支持 Qwen3.8-Flash-Next

Qwen3.8-Flash-Next 已经通过 MoE4All 在消费级 **AMD Radeon RX 7900 XTX**
上稳定生成正常内容，简单数学和推理问题也能够正确回答。Q2_K_XL 和 IQ4_XS
量化均已完成 Windows 11 实测，并可在**受限进程 RAM 预算**下从 SSD 分页运行。

当前 `qwen4exp` 文本路径已经完整接入发布模型所需的四流 Hyper-Connection、
Gated DeltaNet/全注意力混合层、SSD 支持的 PLE、分页 MoE，以及 **QSA 稀疏注意力**。
QSA 会维护独立的 F16 index-key cache，选择完整历史块并保留未完成的 causal tail，
以覆盖长上下文。主 K/V Cache 支持 Q8_0；QSA index-key cache 保持 F16，并独立
计价和分配。

下表来自 RX 7900 XTX、Vulkan0、Q8 K/V 和 40 GiB bounded RAM 下的历史稳定实测，
未发生 Windows 系统分页。Decode 使用 `tg128`，Prefill 使用 `pp1024`，ubatch 为
1024；128K/250K 通过 synthetic depth 构造真实 KV 长度。表内为三次平均值。

| Context depth | Q2_K_XL decode | Q2_K_XL prefill | IQ4_XS decode | IQ4_XS prefill |
|---:|---:|---:|---:|---:|
| 0 | **29.45 tok/s** | **155.16 tok/s** | **16.85 tok/s** | **244.68 tok/s** |
| 128K | **26.23 tok/s** | **170.44 tok/s** | **14.15 tok/s** | **250.55 tok/s** |
| 250K | **22.82 tok/s** | **152.27 tok/s** | **15.26 tok/s** | **239.00 tok/s** |

更多历史稳定成绩与测试条件见
[Windows 本地大模型代表性性能记录](docs/perf/windows-local-model-matrix-20260829.md)。

Q2 与 IQ4_XS 都通过了三轮真实 API 对话，能够保持校验码、完成跨轮算术并总结
先前内容。

QSA 当前使用保持 score/index 精确顺序的 radix top-k，并已接入 batched QSA/PLE
Prefill。Decode 仍明显受专家 RAM/SSD 覆盖率影响，仍有继续优化空间。

## 三步开始

### 1. 下载

打开 [最新 Release](https://github.com/Headmaster218/MoE4All/releases/latest)，
下载 `MoE4All-Windows-x86_64-v*.zip` 并完整解压。

发布包已经包含 `infr.exe` 和中英双语启动向导。运行发布版不需要安装 Rust、
Visual Studio 或 Vulkan SDK，只需要正常的 64 位 AMD 显卡驱动及其 Vulkan
运行时。

### 2. 准备模型

模型需要是当前支持架构的 GGUF 文件，发布包不包含模型。分片 GGUF 的所有
分片必须放在同一个目录；加载器可以从其中任意一片找到完整模型。

建议第一次先用小模型确认环境，再尝试几十到上百 GB 的大型 MoE。模型下载、
分片和量化选择见[快速开始](GETTING_STARTED.md)。

### 3. 启动

双击：

```text
Start-INFR-Wizard.cmd
```

选择终端聊天、OpenAI 兼容 API 或性能测试，然后输入或拖入 GGUF 路径。
普通用户建议使用“自动配置”：MoE4All 会探测 GPU、可用显存和系统内存，并
自动规划 KV Cache、运行时空间和专家缓存。

向导启动时会用很短的网络请求检查 GitHub Release；发现新版本时只显示下载
链接，不会自动修改程序。断网不会阻止启动。

## 它能做什么

- **让大模型跨显存运行**：按需使用 VRAM、RAM 和 SSD，不要求整个 MoE
  模型常驻显存。
- **原生 AMD Vulkan**：Windows 下不依赖 CUDA，也不要求通过 WSL 启动。
- **直接聊天**：终端中保持上下文进行多轮对话，可选择模型默认、开启或关闭
  思考模式。
- **兼容现有客户端**：提供 OpenAI 兼容的聊天与 Embedding API。
- **长上下文**：支持量化 KV Cache、KV 溢出和长上下文性能测试。
- **可测量、可调试**：内置 prefill/decode benchmark、synthetic depth 和分页
  统计工具。

## 需要什么

| 项目 | 说明 |
|---|---|
| 操作系统 | 主要测试于 64 位 Windows 11 |
| GPU | 当前重点支持 AMD Vulkan；显存越大，可常驻的权重和 KV 越多 |
| 驱动 | 安装 AMD 官方稳定驱动，`infr.exe devices` 应能看到显卡 |
| 内存 | 大型 MoE 会利用系统 RAM；模型远大于 RAM 时可继续从 SSD 分页 |
| 存储 | 建议高速本地 SSD，并为模型全部分片留足空间 |
| 模型 | 自备受支持架构的 GGUF，单文件或完整分片组 |

能运行多大的模型取决于模型量化、固定权重、上下文、显存、RAM 和 SSD。MoE4All
的目标是尽量利用现有硬件，而不是承诺任意模型都能在任意 A 卡上达到同样速度。

## 实测结果

以下代表值来自 RX 7900 XTX 24 GiB、Ryzen 5 5600X、64 GiB DDR4、Windows 11
主机的历史稳定测试。已排除进入系统分页的资源压力诊断结果；各行测试日期、模型量化、
上下文和缓存条件不同，不能直接横向比较。

| 模型与负载 | 关键条件 | 结果 |
|---|---|---:|
| Qwen3.6-35B-A3B，250K synthetic depth 后 decode | Q8 K/V，生成 1,000 token | **41.2 tok/s** |
| Qwen3.6-35B-A3B，250K 后 prefill 4,096 | Q8 K/V | **477.9 tok/s** |
| Qwen3.6-35B-A3B，depth 0 prefill 4,096 | Q8 K/V | **2,855.6 tok/s** |
| Qwen3.5-122B-A10B，depth 0 decode | F16 K/V，45 GiB bounded RAM，3 次重复 | **23.2 tok/s** |
| Qwen3.8-Flash-Next Q2_K_XL，250K 后 decode | Q8 K/V，40 GiB bounded RAM，tg128，3 次平均 | **22.82 tok/s** |
| Qwen3.8-Flash-Next Q2_K_XL，250K 后 prefill 1,024 | Q8 K/V，40 GiB bounded RAM，3 次平均 | **152.27 tok/s** |
| Qwen3.8-Flash-Next IQ4_XS，250K 后 decode | Q8 K/V，40 GiB bounded RAM，tg128，3 次平均 | **15.26 tok/s** |
| Qwen3.8-Flash-Next IQ4_XS，250K 后 prefill 1,024 | Q8 K/V，40 GiB bounded RAM，3 次平均 | **239.00 tok/s** |

完整条件和优化历史见：

- [Windows 本地大模型代表性性能记录](docs/perf/windows-local-model-matrix-20260829.md)
- [Qwen3.6 RX 7900 XTX 优化记录](https://github.com/Headmaster218/MoE4All/blob/main/docs/perf/qwen36-rx7900xtx-optimization-history-20260819.md)
- [统一显存验收记录](https://github.com/Headmaster218/MoE4All/blob/main/docs/unified-vram-elastic-acceptance-20260824.md)
- [DeepSeek V4 Flash 收尾记录](https://github.com/Headmaster218/MoE4All/blob/main/docs/perf/deepseek-v4-flash-rx7900xtx-closeout-20260824.md)

## 当前模型支持

| 模型家族 | GGUF 架构 | 状态 |
|---|---|---|
| Llama、Llama 4 | `llama`、`llama4` | Dense 与 MoE Vulkan 推理 |
| Qwen2 / Qwen2.5 / Qwen3 | `qwen2`、`qwen3`、`qwen3moe` | Dense 与 Qwen3 MoE |
| Qwen3.5 / Qwen3.6 | `qwen35`、`qwen35moe` | Gated DeltaNet、Attention 与分页 MoE |
| Qwen3.8 Flash Next | `qwen4exp` | Hyper-Connection、Gated DeltaNet、PLE、QSA 与分页 MoE Vulkan 文本推理 |
| Gemma 3 / Gemma 4 | `gemma3`、`gemma4` | Dense、MoE 与 E2B 变体 |
| Ling 3.0 Flash | `bailingmoe3` | KDA、gated MLA、512 experts 与 RAM/SSD 分页 |
| DeepSeek V4 Flash | `deepseek4` | FP8 KV、MXFP4 indexer cache 与分页 MoE |
| DiffusionGemma | `diffusion-gemma` | 文本扩散推理 |
| Embedding GGUF | 受支持的 Embedding 架构 | 原生 CPU/Vulkan OpenAI Embedding API |

同一架构上的微调模型通常可以直接复用现有实现，但 GGUF metadata、量化格式和
chat template 仍必须完整。项目不会仅凭模型名称假定兼容。

## 三种日常用法

### 终端聊天

启动向导中选择“实时终端对话”，或者直接运行：

```powershell
.\infr.exe run 'D:\Models\model.gguf'
```

### OpenAI 兼容 API

```powershell
.\infr.exe serve --addr 127.0.0.1:8080 'D:\Models\model.gguf'
```

API Base URL 为 `http://127.0.0.1:8080/v1`。对局域网开放前请配置 API key，
不要把无鉴权服务直接暴露到公网。

### 性能测试

```powershell
# Prefill
.\infr.exe bench -p 1024 -n 0 -r 1 'D:\Models\model.gguf'

# Decode
.\infr.exe bench -p 0 -n 128 -r 1 'D:\Models\model.gguf'
```

更多参数和“已有长上下文后再测”的 synthetic depth 用法见
[快速开始](GETTING_STARTED.md)。

## 为什么能跑超过显存的模型

大型 MoE 每个 token 通常只激活全部专家中的一小部分。MoE4All 不要求所有
专家一直留在 GPU，而是维护三级存储：

```text
SSD 上的完整 GGUF
        ↓
完整 Host store 或有上限的 RAM cache
        ↓
弹性 GPU expert cache
        ↓
AMD Vulkan 计算
```

常用专家尽量留在显存，RAM 作为更大的热数据层，剩余内容继续由 SSD 提供。
显存中的模型固定部分、KV Cache、运行时 scratch 和专家缓存由统一预算协调，
prefill 与 decode 切换时可以重新分配弹性空间。

更深入的实现说明在
[技术文档索引](https://github.com/Headmaster218/MoE4All/blob/main/docs/README.md) 和
[MoE4All Wiki](https://github.com/Headmaster218/MoE4All/blob/main/infr-fork-wiki/README.md)。

## 当前限制

- 首次加载大型模型或 RAM cache 尚未预热时，SSD 读取可能让开头几轮明显变慢。
- 大模型 prefill 的 SSD 到 RAM 异步预读仍有优化空间。
- Host DMA 受 Windows 驱动可导入内存范围影响，未导入区域会自动回退到 CPU
  写入 ReBAR。
- 自动预算以“尽量可靠启动”为目标，不保证对每台机器都是最高吞吐配置。
- 当前 Windows 发布包以命令行向导为主；浏览器 GUI 仍属于源码开发入口。

## 项目与署名

MoE4All 由 John / [Headmaster218](https://github.com/Headmaster218) 维护。
项目基于 kryptic.sh 的 Pure-Rust、Vulkan-first 推理引擎
[infr](https://github.com/kryptic-sh/infr)。上游项目的原始说明请直接阅读
[infr README](https://github.com/kryptic-sh/infr#readme)，不再在本仓库中重复
嵌入。

本项目的架构决策、性能调查和验收由维护者主导，并广泛使用 AI coding agents
辅助 Rust、Vulkan、测试和文档工作。

MoE4All 的修改与整体发行采用 [Apache License 2.0](LICENSE)。继承自上游
infr 的代码保留其原始 [MIT License](LICENSE-MIT) 和版权声明；详细归属见
[NOTICE](NOTICE)。同时保留两个许可证文件是为了分别说明本项目与上游代码的
许可来源，并不表示用户必须在二者中二选一。
