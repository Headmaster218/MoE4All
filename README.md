# MoE4All

**让游戏显卡跑起远超显存容量的大模型。AMD、NVIDIA、Intel，均已跑通。**

Windows、Linux 免安装，约 17 MiB 即下即用；Apple Silicon 同步提供实验性 Metal 发行包。选择推荐的模型或自备模型，自动配置参数，即可立即开始聊天，
或通过 OpenAI 兼容API接入现有客户端。显存、内存与 SSD 三级流动，协同加载权重运行。

**1.0.0 性能参考**：Win 11 + RX 7900 XTX 24 GiB + Ryzen 5 5600X + 64 GiB DDR4，
自动激进配置，Q8 K/V，关闭 MTP 与 CPU miss。速度单位均为 tok/s。

| 模型 | 30K 单路 Prefill | 30K Decode 总均速 x1 / x2 / x3 | 150K 单路 Prefill | 150K Decode 总均速 x1 / x2 / x3 |
| --- | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | **1,414.75** | **50.44 / 66.19 / 71.20** | **1,445.50** | **44.16 / 49.50 / 45.47** |
| Qwen3.6 35B Balanced | **2,749.06** | **80.49 / 118.34 / 110.78** | **1,169.40** | **41.91 / 50.87 / 49.27** |

<sub>Decode 为 1 / 2 / 3 路同时生成时的总平均吞吐；Prefill 为单路完整区间平均速度。完整条件见 [1.0.0 实测](#10030k--150k-平均性能2026-10-14)。</sub>

**同机 Strata v0.1.44.1 参考**：使用其自动配置、不启用 MTP；`x2` 为两路总吞吐。

| 模型 | 30K 单路 Prefill | 30K Decode x1 / x2 | 150K 单路 Prefill | 150K Decode x1 |
| --- | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 576.15 | 30.17 / 48.14 | 650.01 | 36.52 |
| Qwen3.6 35B Balanced | 1,325.32 | 19.37 / 未支持 | 436.51 | 11.00 |

<sub>这是同一台机器上的方向性对照，兼容转换和测试差异见 [Strata 对照说明](#strata-v01441-同机对照2026-10-15)。</sub>

[下载已发布版本](https://github.com/Headmaster218/MoE4All/releases/latest) |
[快速使用](#快速使用) |
[实测结果](#实测结果) |
[社区实测](#社区实测) |
[English](README_EN.md) |
[技术文档](documentation/README.md) |
[低权重专家跳过实验](documentation/evidence/campaigns/qwen38/2026-10-11-low-weight-expert-omission.md) |
[专家缓存与多卡配比](documentation/evidence/benchmarks/2026-10-11-flashnext-expert-cache-capacity.md)

## 快速使用

### 1. 下载程序

打开 [MoE4All Releases](https://github.com/Headmaster218/MoE4All/releases)，
Windows 下载 `MoE4All-Windows-x86_64-v*.zip`，Linux x86_64 下载对应的
`.tar.gz` 包；Apple Silicon 可试用 `MoE4All-macOS-arm64-v*.tar.gz` 实验包。


### 2. 解压

将发行包**完整解压**到一个目录，例如 Windows 的 `D:\MoE4All`。

### 3. 运行

1. Windows 双击 **`Start-INFR-Wizard.cmd`**；Linux 运行 `bash Start-INFR-Wizard-Linux.sh`；Apple Silicon 运行 `./Start-INFR-Wizard-macOS.sh`。
2. 选择“终端聊天”或“OpenAI 兼容 API”。模型列表先显示本地已保存模型，下面的 `D` 是下载入口；进入后可自行选择一个或多个主模型，Windows/Linux 还可选择视觉或 Embedding 组件。下载完成后会回到本地模型列表，不会替用户默认选定某个模型。Windows 也可将已有 GGUF 拖入窗口后按 Enter。
3. 选择运行设备与自动配置档位。**激进性能**是本页实测使用的档位；**保守**适合首次试运行或后台程序较多时使用。
4. Windows/Linux API 模式可按需启用视觉和 Embedding。macOS Metal 实验版目前为单槽文本服务，不提供视觉与 Embedding。设置上下文和 API 后确认启动。

三个发行版的自动下载都会显示进度，失败重试并从 `.part` 文件续传，文件统一保存在安装目录的 `models`。Linux 向导不依赖 Python 或 jq；macOS 向导只依赖系统 `zsh`、`curl` 和标准 Apple 工具，无需 Homebrew、Python、jq 或 Rust。自动档默认使用 Q8 K/V，由引擎规划可用资源。终端聊天当前为纯文本；图片请通过 Windows/Linux 上启用视觉的 OpenAI 兼容 API 发送。

macOS 实验包使用临时签名，尚未 Apple 公证。若 Gatekeeper 拦截首次运行，请在终端对解压目录执行 `xattr -dr com.apple.quarantine /path/to/MoE4All-macOS-arm64-v*`，再启动向导。

组合能力和限制见[模型能力矩阵](documentation/reference/model-capabilities.md)。API 默认地址为 `http://127.0.0.1:8080/v1`；请求示例见[API 使用](documentation/guide/serving/api-quickstart.md)，完整配置项见[配置参考](documentation/reference/configuration.md)。

## 推荐模型与下载

Windows 与 Linux 启动向导可从统一下载菜单选择下列主模型和组件；macOS Metal 实验版当前只列出两个主模型。也可使用链接自行下载到本地 SSD，自动下载统一保存到解压目录的 `models` 文件夹。

| 模型 / 组件 | 下载链接 | 文件与用途 |
| --- | --- | --- |
| **Qwen3.6 35B 本体** | [下载 APEX-I-Balanced](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true) | `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`；35B 下载这一个文件即可 |
| **Flash-Next 主模型** | [下载 AD-4.27bpw-Q4_K_M-M64 全部分片](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64) | 该目录下的 **33 个 GGUF 分片**，全部放在同一文件夹；启动时选择第一片 |
| **Flash-Next 视觉** | [下载 F16 视觉文件](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true) | `mmproj-Qwen3.8-Flash-Next-F16.gguf`；API 图片理解时加载 |
| **Qwen3 Embedding 0.6B** | [下载 Q8_0 GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true) | `Qwen3-Embedding-0.6B-Q8_0.gguf`；提供 `/v1/embeddings` |

Flash-Next 第一片为 `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`。视觉文件会自动放到同一模型目录，便于向导发现。


## 实测结果

### 1.0.0：30K / 150K 平均性能（2026-10-14）

测试硬件为 RX 7900 XTX 24 GiB、Ryzen 5 5600X、64 GiB DDR4、Windows 11。
使用自动激进配置、Q8 K/V、`ctx=163840`，关闭 MTP 与 CPU miss。Decode 分别同时
占用 1 / 2 / 3 个槽位，数值为所有活跃路合计的完整区间平均吞吐；Prefill 为单路平均速度。

| 模型 | 输入长度 | 单路 Prefill | Decode x1 总均速 | Decode x2 总均速 | Decode x3 总均速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next 4.27bpw | 30K | **1,414.75** | **50.44** | **66.19** | **71.20** |
| Flash-Next 4.27bpw | 150K | **1,445.50** | **44.16** | **49.50** | **45.47** |
| Qwen3.6 35B Balanced | 30K | **2,749.06** | **80.49** | **118.34** | **110.78** |
| Qwen3.6 35B Balanced | 150K | **1,169.40** | **41.91** | **50.87** | **49.27** |

### Strata v0.1.44.1 同机对照（2026-10-15）

本次额外在同一台 RX 7900 XTX / Ryzen 5 5600X / 64 GiB Windows 主机上运行 Strata 官方 v0.1.44.1 HIP 发行版。使用 `ctx=163840`、INT8 K/V，以及 Strata 的自动 expert-cache 和 Prefill 策略；关闭 prompt cache，不使用 MTP。每项先预热，再记录完整 Prefill 和较长 Decode 区间。

| 模型 | 输入长度 | 单路 Prefill | Decode x1 均速 | Decode x2 总均速 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| Flash-Next 4.27bpw | 30K | 576.15 | 30.17 | 48.14 | 双路为同步重叠区间总吞吐 |
| Flash-Next 4.27bpw | 150K | 650.01 | 36.52 | 未测试 | Decode 前 / 中 / 后为 21.28 / 40.88 / 45.71 |
| Qwen3.6 35B Balanced | 30K | 1,325.32 | 19.37 | 当前构建不支持 | Prefill 为预热后三轮中位数；Decode 范围 18.51–21.61 |
| Qwen3.6 35B Balanced | 150K | 436.51 | 11.00 | 当前构建不支持 | 一次完整 Prefill，随后生成 512 token |

为让模型被该版本 Strata 接受，Flash-Next 的 363 个固定投影张量转换为 BF16；35B 补充了兼容 metadata，并将 60 个很小的 GDN alpha/beta 状态张量从 Q6_K 转为 F32。两者的 routed expert 量化均未改变，原模型文件也未修改，但固定权重已不再逐位一致，因此这些数字只能作为方向性对照。

Strata 自动模式为完整专家分配约 47.17 GiB Host arena；Flash-Next 单路另使用约 13 GiB GPU expert cache，miss 可在 CPU 计算或经 PCIe 搬运。测试时 Windows 总提交量接近 95 GiB，且 Strata 没有套用 MoE4All 的 48 GiB 进程总 RAM 预算。其混合 CPU/GPU 路径在更强 CPU 上可能表现不同。两边测试 prompt、固定权重格式、内存策略和统计脚本并不完全一致，不能把差值归因于某一个 kernel 或单项设计。

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
- **并发服务**：多个独立 K/V 槽可处理先后到达、上下文长度不同的请求；
  Qwen3.5 / Qwen3.6 MoE 可将同时活跃的文本请求合入层同步批量 Decode。
- **会话持久化**：可选的 SSD 缓存能转存空闲文本 K/V，并在服务重启后恢复。
- **长上下文**：支持量化 KV Cache、KV 溢出和长上下文性能测试。
- **Qwen3.8 MTP**：可选的单路与双槽机会式投机解码，收益取决于草稿接受率，
  当前保留为手动 CLI / 配置功能，不在 Windows 发行版启动向导中提供。
- **实验性 CPU miss 计算**：默认关闭，支持 AVX2 + FMA3 CPU；Ryzen 5 5600X 使用
  4 核处理 1 miss 时，端到端速度与 GPU 路径基本相同，更强 CPU 可能提速，需本机实测。
- **可测量、可调试**：内置 prefill/decode benchmark、synthetic depth 和分页
  统计工具。


## 当前模型支持

| 模型家族                | GGUF 架构                          | 状态                                                                  |
| ----------------------- | ---------------------------------- | --------------------------------------------------------------------- |
| Llama、Llama 4          | `llama`、`llama4`              | Dense 与 MoE Vulkan 推理                                              |
| Qwen2 / Qwen2.5 / Qwen3 | `qwen2`、`qwen3`、`qwen3moe` | Dense 与 Qwen3 MoE                                                    |
| Qwen3.5 / Qwen3.6       | `qwen35`、`qwen35moe`          | Gated DeltaNet、Attention、分页 MoE 与层同步并发生成                   |
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

专家缓存留多少显存才够、多张不同 PCIe 带宽的显卡该怎样分层？参见 [FlashNext 专家缓存与多卡配比](documentation/evidence/benchmarks/2026-10-11-flashnext-expert-cache-capacity.md)：包含真实访问序列的 0.5～48 GiB 离线回放曲线、连续分段指数函数，以及基于曲线的多卡层数分配公式和适用限制。


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
