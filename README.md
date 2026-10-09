# MoE4All

**让游戏显卡跑起远超显存容量的大模型。AMD、NVIDIA、Intel，均已跑通。**

Windows 免安装，下载包约 14 MiB。下载 GGUF、选择自动配置，即可本地聊天，
或通过 OpenAI 兼容接口接入现有客户端。显存、内存与 SSD 协同加载 MoE 专家权重。

**0.10.0 性能参考**：RX 7900 XTX 24 GiB + Ryzen 5 5600X + 64 GiB DDR4，自动激进配置。
Qwen3.8-Flash-Next · AD-4.27bpw-Q4_K_M-M64，MTP / CPU miss 关闭，三个槽位。
以下为 2026-10-09 测试记录，单位 tok/s；测试来源与统计口径见[实测结果](#实测结果)。

| 同时生成路数 | 30K Prefill | 30K Decode 总速 | 150K Prefill | 150K Decode 总速 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,093 | **42.8** | 752 | **39.5** |
| 2 | 960 | **59.7** | 958 | **48.1** |
| 3 | 1,033 | **70.2** | 1,049 | **40.4** |

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

推荐以下模型与组件，模型单独下载到本地 SSD。上方性能表使用 Flash-Next 主模型，不启用 MTP。

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

已有保存设置时，可选择直接按上次设置启动；启动前会先检查更新，默认不更新。

自动档默认使用 Q8 K/V，显存、内存、专家缓存与 Ubatch 由引擎规划。
35B 加载本体即可聊天。Flash-Next 的可选组件如下：

- **文本 MTP 加速**：启用“Qwen3.8 MTP 加速”，选择上表的 MTP 头，使用 greedy（`temperature=0`）。
  双槽机会式 MTP 在仅一路活跃时使用 MTP，两路同时 Decode 时自动回退普通批量 Decode。
- **图片理解**：选择 API 模式，启用“视觉图片理解”，选择上表的 F16 视觉文件；可与 MTP 组合。

组合能力和限制见[模型能力矩阵](documentation/reference/model-capabilities.md)。API 默认地址为 `http://127.0.0.1:8080/v1`；请求示例见[API 使用](documentation/guide/serving/api-quickstart.md)，完整配置项见[配置参考](documentation/reference/configuration.md)。

## 实测结果

### 0.10.0：Flash-Next，30K / 150K 与 1 / 2 / 3 路并发

本组来自升级到 0.10.0 前的 **0.9.0 本地服务**，不是 0.10.0 重跑结果。
硬件为 RX 7900 XTX 24 GiB、Ryzen 5 5600X、64 GiB DDR4、Windows 11；
模型为 `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`。
使用自动激进配置、Q8 K/V、`ctx=163840`，配置三个槽位，分别同时占用 1 / 2 / 3 槽；
**MTP 与 CPU miss 均关闭**，greedy、关闭思考，各路使用不同的长技术背景提示词。

速度单位均为 tok/s。Prefill 为首轮所有请求均未命中 KV 时的组吞吐：总输入 token
除以最后一路完成 Prefill 的组阶段耗时，包含调度和等待，不是各路速度相加。
Decode 为随后复用 KV 的三轮均值，每轮各路生成 512 token；总速是各路 API 最终
`predicted_per_second` 之和，**不是终端瞬时值**。多个大 Prefill 按顺序完成，然后并发 Decode。

| 每路输入 | 同时生成路数 | 冷 Prefill 总速 | Decode 总速 | Decode 每路约 |
| --- | ---: | ---: | ---: | ---: |
| 30K | 1 | 1,093 | **42.8** | 42.8 |
| 30K | 2 | 960 | **59.7** | 29.9 |
| 30K | 3 | 1,033 | **70.2** | 23.4 |
| 150K | 1 | 752 | **39.5** | 39.5 |
| 150K | 2 | 958 | **48.1** | 24.0 |
| 150K | 3 | 1,049 | **40.4** | 13.5 |

服务未在各组间重启，结果也反映了空闲槽的 KV 驻留和 SSD 写回影响：
150K 单路首次只有一条专家 Prefill lane，释放其他空闲槽后，另组完整 Prefill 达到
**1,313–1,316 tok/s**。30K 双路记录含一次约 1 秒暂停；150K 三路 Decode 三轮为
**37.6–43.1 tok/s**。这些是有限轮次的服务实测，不是隔离 kernel 跑分或长期稳定性保证。

<details>
<summary>相同与不同提示词对并发 Decode 的影响</summary>

另两组对照都使用 greedy、关闭思考，各取三轮 KV 复用后的均值。
不同提示词组使用三个技术主题；相同提示词组的完整输入、seed 和生成结果均相同。
以下仍为 Decode **总速**，单位 tok/s，不应把相同输出的最佳吞吐当作任意 Agent 任务速度。

| 每路输入 | 同时生成路数 | 不同提示词 | 完全相同提示词 |
| --- | ---: | ---: | ---: |
| 30K | 2 | 61.5 | **69.9** |
| 30K | 3 | 69.5 | **84.9** |
| 150K | 2 | 48.6 | **59.8** |
| 150K | 3 | 44.5 | **64.4** |

这是独立的提示词对照组，与上表的全冷 Prefill 组不混算；相同提示词也不代表并发首轮
一定只需做一次 Prefill。

</details>

## 社区实测

**AMD、NVIDIA RTX 和 Intel Arc 均已有用户实测，具体配置与速度如下。**


| GPU 与内存 | 模型与条件 | 生成速度 | 版本与来源 |
| --- | --- | ---: | --- |
| **AMD RX 7700 XT 12GB + 64GB RAM** | Ornith 1.5 35B-A3B，`serve`，131K 上下文容量 | **41.5–42.7 tok/s** | v0.5.2 社区基线，[PR #21](https://github.com/Headmaster218/MoE4All/pull/21) |
| **Intel Arc A770 16GB + 64GB DDR4-3200** | Ornith 1.5 35B Q4_K_M，F16 KV，96K 上下文容量，REAL 负载三次测试 | **30.15–30.28 tok/s** | v0.6.0-beta.1，[Issue #41](https://github.com/Headmaster218/MoE4All/issues/41) |
| **NVIDIA RTX 3090 Ti + 64GB RAM** | 社区成功运行反馈；原评论未注明模型、量化和上下文 | **约 29 tok/s** | 版本未注明，[B站用户反馈](https://www.bilibili.com/video/BV1ALha63Eyd/)（rpid `318146261056`） |


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
