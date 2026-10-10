# MoE4All

**让游戏显卡跑起远超显存容量的大模型。AMD、NVIDIA、Intel，均已跑通。**

Windows 免安装，下载约 14 MiB。下载 GGUF、选择自动配置，即可本地聊天，
或通过 OpenAI 兼容接口接入现有客户端。显存、内存与 SSD 协同加载 MoE 专家权重。

**0.10.0 最快分段速度参考**：RX 7900 XTX 24 GiB + Ryzen 5 5600X + 64 GiB DDR4，自动激进配置。
Flash-Next 4.27bpw 在 30K 下单路 Prefill 为 **1,414.3 tok/s**，三路 Decode 为 **65.5 tok/s**；
35B Balanced 在 30K 下单路 Decode 为 **63.6 tok/s**。
MTP / CPU miss 关闭，配置三个槽位。以下取 2026-10-09 实测前 / 中 / 后三段中最快一段的平均速度，单位 tok/s；统计口径见[实测结果](#实测结果)。

| 模型 | 同时生成路数 | 30K Prefill  | 30K Decode  | 150K Prefill  | 150K Decode  |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 1 | 1,414.3 | **45.3** | 1,358.2 | **41.5** |
| Flash-Next 4.27bpw | 2 | 1,335.0 | **63.3** | 1,353.4 | **55.7** |
| Flash-Next 4.27bpw | 3 | 1,302.4 | **65.5** | 1,303.6 | **46.7** |
| 35B Balanced | 1 | 3,254.9 | **63.6** | 1,829.5 | **36.0** |

<sub>各统计区间按时间三等分，取三段中最高的平均速度。Decode 分段值为三轮均值；Prefill 分段速度由 chunk 完成进度插值估算。</sub>

[下载已发布的 Windows 版本](https://github.com/Headmaster218/MoE4All/releases/latest) |
[快速使用](#快速使用) |
[实测结果](#实测结果) |
[社区实测](#社区实测) |
[English](README_EN.md) |
[技术文档](documentation/README.md)

## 快速使用

### 1. 下载程序

打开 [MoE4All Releases](https://github.com/Headmaster218/MoE4All/releases)，
下载对应版本的 `MoE4All-Windows-x86_64-v*.zip`。本页面向 **0.10.0**，
性能表的测试条件与来源见下方实测结果。


### 2. 解压

将 ZIP **完整解压**到一个目录，例如 `D:\MoE4All`。

### 3. 下载 GGUF 模型

推荐以下模型与组件，模型单独下载到本地 SSD。

| 模型 / 组件 | 下载链接 | 文件与用途 |
| --- | --- | --- |
| **Qwen3.6 35B 本体** | [下载 APEX-I-Balanced](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true) | `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`；35B 下载这一个文件即可 |
| **Flash-Next 主模型** | [下载 AD-4.27bpw-Q4_K_M-M64 全部分片](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64) | 该目录下的 **33 个 GGUF 分片**，全部放在同一文件夹 |
| **Flash-Next 视觉** | [下载 F16 视觉文件](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true) | `mmproj-Qwen3.8-Flash-Next-F16.gguf`；图片理解时加载 |
| **Flash-Next MTP** | [下载 shared Q4_K_M MTP 头](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf?download=true) | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`；可选的文本加速组件 |

Flash-Next 启动时选择第一片：`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`。
视觉与 MTP 文件可放在主模型目录，在向导中按用途选择。

### 4. 运行

1. 双击解压目录中的 **`Start-INFR-Wizard.cmd`**。
2. 选择“终端聊天”或“OpenAI 兼容 API”，将主模型 GGUF 拖入窗口，按 Enter。
3. 按需选择 MTP、视觉和 Embedding，选择运行设备及自动配置档位：**激进性能**是本页实测使用的档位；**保守**适合首次试运行或后台程序较多时使用。
4. 设置并发、上下文和 API，确认启动。复现实测时，配置三个槽位，上下文设置 `163840`，关闭 MTP 与实验性 CPU 计算。



自动档默认使用 Q8 K/V，显存、内存、专家缓存与 Ubatch 由引擎规划。
35B 加载本体即可聊天。Flash-Next 的可选组件如下：

- **文本 MTP 加速**：启用“Qwen3.8 MTP 加速”，选择上表的 MTP 头，使用 greedy（`temperature=0`）。
  双槽机会式 MTP 在仅一路活跃时使用 MTP，两路同时 Decode 时自动回退普通批量 Decode。
- **图片理解**：选择 API 模式，启用“视觉图片理解”，选择上表的 F16 视觉文件；可与 MTP 组合。

组合能力和限制见[模型能力矩阵](documentation/reference/model-capabilities.md)。API 默认地址为 `http://127.0.0.1:8080/v1`；请求示例见[API 使用](documentation/guide/serving/api-quickstart.md)，完整配置项见[配置参考](documentation/reference/configuration.md)。

## 从源码构建（Linux）

Linux x86_64 实验性发行包与 Windows 同版发包，欢迎试用反馈；首次 CI 验证完成前仍建议从源码构建。
发布页提供 `.tar.gz` 后可直接解压，运行包内 `Start-INFR-Wizard-Linux.sh`，无需 Rust 或 glslc。
以下源码安装命令适用于 Ubuntu 26.04；
其他发行版请先准备 shaderc 2025 或更新的 `glslc`：

```sh
sudo apt-get update && sudo apt-get install -y glslc
git clone https://github.com/Headmaster218/MoE4All.git && cd MoE4All
cargo build --release --locked -p infr-cli
./Start-INFR-Wizard-Linux.sh      # 交互式启动；加 --dry-run 只打印命令
```

需要 Rust 1.97.1（`rust-toolchain.toml` 会自动安装）与 shaderc 2025 及以上的
`glslc`。完整指南见 [在 Linux 上构建并运行](documentation/guide/linux-build-and-run.md)，
工具链与平台细节见 [从源码构建](documentation/development/building-from-source.md)。

## 实测结果

### 0.10.0：30K / 150K 实测

测试于 **2026-10-09**，使用 0.10.0 通用 x86-64 发布构建，代码提交 `53afa86f3`。
硬件为 RX 7900 XTX 24 GiB、Ryzen 5 5600X、64 GiB DDR4、Windows 11；
模型为 `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64` 与 `Qwen3.6-35B-A3B-APEX-I-Balanced`。
使用自动激进配置、Q8 K/V、`ctx=163840`，配置三个槽位，分别同时占用 1 / 2 / 3 槽；
**MTP 与 CPU miss 均关闭**，采样与思考使用模型默认。视觉与 Embedding API 已启用，
下表使用纯文本请求，各路为不同的长技术背景加普通问题。

每组重新启动服务，首轮全冷 Prefill，随后两轮复用 KV；每轮每路生成 768 token。
速度单位均为 tok/s。Prefill 为首次进度采样至最后一路完成 Prefill 的组吞吐，
不含首次采样前的启动时间，也不是各路速度相加。Decode 为所有路同时 Decode 区间的三轮均值，
不把等待其他请求 Prefill 的时间算成解码降速；“每路约”为总均速除以路数。
峰值为 1.5 秒滚动窗口最大值；Prefill 的峰值与分段速度由 chunk 完成进度插值估算，
不代表精确的瞬时 GPU 速度。

| 模型 | 每路输入 | 同时生成路数 | 冷 Prefill 均速 / 峰值 | Decode 总均速 / 峰值 | Decode 每路约 |
| --- | --- | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 30K | 1 | 1,280.4 / 1,439.4 | **42.8** / 47.7 | 42.8 |
| Flash-Next 4.27bpw | 30K | 2 | 1,288.2 / 1,445.2 | **60.7** / 71.3 | 30.3 |
| Flash-Next 4.27bpw | 30K | 3 | 1,283.9 / 1,437.0 | **64.0** / 79.4 | 21.3 |
| Flash-Next 4.27bpw | 150K | 1 | 1,307.5 / 1,444.6 | **39.4** / 44.7 | 39.4 |
| Flash-Next 4.27bpw | 150K | 2 | 1,301.9 / 1,446.3 | **52.8** / 63.6 | 26.4 |
| Flash-Next 4.27bpw | 150K | 3 | 1,228.3 / 1,434.2 | **45.4** / 62.5 | 15.1 |
| 35B Balanced | 30K | 1 | 2,579.9 / 3,824.8 | **63.2** / 65.4 | 63.2 |
| 35B Balanced | 150K | 1 | 1,164.9 / 3,981.7 | **35.9** / 36.7 | 35.9 |

前 / 中 / 后段按各自统计区间的时间三等分；Decode 分段值同样是三轮均值。

| 模型 | 每路输入 | 同时生成路数 | Prefill 前 / 中 / 后 | Decode 总速前 / 中 / 后 |
| --- | --- | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 30K | 1 | 1,225.1 / 1,414.3 / 1,201.9 | 40.8 / 42.2 / 45.3 |
| Flash-Next 4.27bpw | 30K | 2 | 1,315.6 / 1,214.0 / 1,335.0 | 58.5 / 63.3 / 60.2 |
| Flash-Next 4.27bpw | 30K | 3 | 1,302.4 / 1,285.8 / 1,263.3 | 65.5 / 62.6 / 63.8 |
| Flash-Next 4.27bpw | 150K | 1 | 1,358.2 / 1,345.7 / 1,218.8 | 36.3 / 40.4 / 41.5 |
| Flash-Next 4.27bpw | 150K | 2 | 1,353.4 / 1,281.4 / 1,270.8 | 50.3 / 55.7 / 52.4 |
| Flash-Next 4.27bpw | 150K | 3 | 1,303.6 / 1,263.0 / 1,118.3 | 44.0 / 46.7 / 45.4 |
| 35B Balanced | 30K | 1 | 3,254.9 / 2,797.3 / 1,687.6 | 63.6 / 62.6 / 63.4 |
| 35B Balanced | 150K | 1 | 1,829.5 / 968.9 / 696.4 | 35.8 / 36.0 / 35.8 |

Flash-Next 在 150K 下三路总速低于双路，并发收益不是线性的。
本轮完整矩阵共 36 轮、72 路生成，无请求失败，后两轮均命中预期 KVlash-Next 测试期间 Windows 提交额度余量较小，也可能影响复测速度。
## 社区实测

**AMD、NVIDIA RTX 和 Intel Arc 均已有用户实测，具体配置与速度如下。**


| GPU / CPU 与内存 | 模型与条件 | Prefill | Decode | 版本与来源 |
| --- | --- | ---: | ---: | --- |
| **AMD R9700 + Ryzen 9 9950X3D + 48GB RAM** | Flash-Next，量化与上下文未注明 | **800–1,100 tok/s** | **20–50 tok/s** | v0.10.0，用户反馈 |
| **Intel Arc A770 16GB + 64GB DDR4-3200** | Flash-Next，三次测试，量化与上下文未注明 | 未提供 | **17.44 / 17.30 / 17.45 tok/s** | v0.10.0，用户反馈 |
| **AMD RX 7700 XT 12GB + 64GB RAM** | Ornith 1.5 35B-A3B，`serve`，131K 上下文容量 | 未提供 | **41.5–42.7 tok/s** | v0.5.2 社区基线，[PR #21](https://github.com/Headmaster218/MoE4All/pull/21) |
| **Intel Arc A770 16GB + 64GB DDR4-3200** | Ornith 1.5 35B Q4_K_M，F16 KV，96K 上下文容量，REAL 负载三次测试 | 未提供 | **30.15–30.28 tok/s** | v0.6.0-beta.1，[Issue #41](https://github.com/Headmaster218/MoE4All/issues/41) |
| **NVIDIA RTX 3090 Ti + 64GB RAM** | 社区成功运行反馈；原评论未注明模型、量化和上下文 | 未提供 | **约 29 tok/s** | 版本未注明，[B站用户反馈](https://www.bilibili.com/video/BV1ALha63Eyd/)（rpid `318146261056`） |

社区速度保留用户原始口径，未独立复测；未说明是否为均值或峰值，不能直接与上方最快分段速度表比较。


欢迎在 [Discussions](https://github.com/Headmaster218/MoE4All/discussions) 分享成功配置，
或通过 [Issues](https://github.com/Headmaster218/MoE4All/issues) 报告问题。
附上 GPU/显存、RAM、系统与驱动、程序版本、模型量化、上下文、自动档位或启动命令，
以及 Prefill/Decode 速度，就能帮助更多相同硬件的用户复现。

## 它能做什么

- **让大模型跨显存运行**：按需使用 VRAM、RAM 和 SSD，协同缓存与加载 MoE 专家权重。
- **Vulkan 推理**：在 Windows 下直接通过显卡驱动执行 Vulkan 计算；
  AMD 主力验证，NVIDIA 与 Intel 已有社区实测。
- **直接聊天**：终端中保持上下文进行多轮对话，可选择模型默认、开启或关闭
  思考模式。
- **兼容现有客户端**：提供 OpenAI 兼容的聊天与 Embedding API。
- **并发服务**：多个独立 K/V 槽可处理先后到达、上下文长度不同的请求。
- **会话持久化**：可选的 SSD 缓存能转存空闲文本 K/V，并在服务重启后恢复。
- **长上下文**：支持量化 KV Cache、KV 溢出和长上下文性能测试。
- **Qwen3.8 MTP**：可选的单路与双槽机会式投机解码，收益取决于草稿接受率，
  使用方式见[快速使用](#快速使用)。
- **实验性 CPU miss 计算**：默认关闭，支持 AVX2 + FMA3 CPU；Ryzen 5 5600X 使用
  4 核处理 1 miss 时，端到端速度与 GPU 路径基本相同，更强 CPU 可能提速，需本机实测。
- **可测量、可调试**：内置 prefill/decode benchmark、synthetic depth 和分页
  统计工具。


## 当前模型支持

| 模型家族                | GGUF 架构                          | 状态                                                                  |
| ----------------------- | ---------------------------------- | --------------------------------------------------------------------- |
| Llama、Llama 4          | `llama`、`llama4`              | Dense 与 MoE Vulkan 推理                                              |
| Qwen2 / Qwen2.5 / Qwen3 | `qwen2`、`qwen3`、`qwen3moe` | Dense 与 Qwen3 MoE                                                    |
| Qwen3.5 / Qwen3.6       | `qwen35`、`qwen35moe`          | Gated DeltaNet、Attention 与分页 MoE                                  |
| Qwen3.8 Flash Next      | `qwen4exp`                       | Vulkan 文本与视觉推理、并发生成、Hyper-Connection、DeltaNet、PLE、QSA 与分页 MoE |
| Gemma 3 / Gemma 4       | `gemma3`、`gemma4`             | Dense、MoE 与 E2B 变体                                                |
| Ling 3.0 Flash          | `bailingmoe3`                    | KDA、gated MLA、512 experts 与 RAM/SSD 分页                           |
| DeepSeek V4 Flash       | `deepseek4`                      | FP8 KV、MXFP4 indexer cache 与分页 MoE                                |
| DiffusionGemma          | `diffusion-gemma`                | 文本扩散推理                                                          |
| Embedding GGUF          | 受支持的 Embedding 架构            | 原生 CPU/Vulkan OpenAI Embedding API                                  |

同一架构上的微调模型通常可以复用现有实现。兼容性取决于 GGUF metadata、
量化格式与 chat template，模型文件应包含完整的推理元数据。


## 为什么能跑超过显存的模型

大型 MoE 每个 token 通常只激活全部专家中的一小部分。MoE4All 维护三级存储：

```text
SSD 上的完整 GGUF
        ↓
完整 Host store 或有上限的 RAM cache
        ↓
弹性 GPU expert cache
        ↓
Vulkan GPU 计算
```

常用专家尽量留在显存，RAM 作为更大的热数据层，剩余内容继续由 SSD 提供。
显存中的模型固定部分、KV Cache、运行时 scratch 和专家缓存由统一预算协调，
prefill 与 decode 切换时可以重新分配弹性空间。

更深入的实现说明见[技术文档索引](documentation/README.md)；历史优化和取舍见[变化记录](documentation/evidence/changes/README.md)。


## 项目与署名

MoE4All 由 John / [Headmaster218](https://github.com/Headmaster218) 维护。
项目基于 kryptic.sh 的 Pure-Rust、Vulkan-first 推理引擎
[infr](https://github.com/kryptic-sh/infr)。上游项目的原始说明请直接阅读
[infr README](https://github.com/kryptic-sh/infr#readme)。

本项目的架构决策、性能调查和验收由维护者主导，并广泛使用 AI coding agents
辅助 Rust、Vulkan、测试和文档工作。

MoE4All 的修改与整体发行采用 [Apache License 2.0](LICENSE)。继承自上游
infr 的代码保留其原始 [MIT License](LICENSE-MIT) 和版权声明；详细归属见
[NOTICE](NOTICE)。两个许可证文件分别记录本项目与上游代码的许可来源。
