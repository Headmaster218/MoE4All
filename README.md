# MoE4All

**让游戏显卡跑起远超显存容量的大模型。AMD、NVIDIA、Intel，均已跑通。**

Windows 免安装，程序 13 MiB。下载 GGUF、选择自动配置，即可本地聊天，
或通过 OpenAI 兼容接口接入现有客户端。显存、内存与 SSD 协同加载 MoE 专家权重。

**0.8.0 发布实测**：RX 7900 XTX 24 GiB + 64 GiB DDR4，自动配置：激进性能。
下表为实测生成速度，单位 tok/s；完整配置与分段结果见[实测结果](#实测结果)。

| 主推模型与量化 | MTP | 20K 输入 | 150K 输入 |
| --- | --- | ---: | ---: |
| **Qwen3.6-35B-A3B · APEX-I-Balanced** | 关闭 | **65.6** | **39.9** |
| **Qwen3.8-Flash-Next · AD-4.27bpw-Q4_K_M-M64** | 开启 | **60.2** | **48.1** |

[下载已发布的 Windows 版本](https://github.com/Headmaster218/MoE4All/releases/latest) |
[快速使用](#快速使用) |
[实测结果](#实测结果) |
[社区实测](#社区实测) |
[English](README_EN.md) |
[技术文档](documentation/README.md)

## 快速使用

### 1. 下载程序

打开 [MoE4All Releases](https://github.com/Headmaster218/MoE4All/releases)，
下载对应版本的 `MoE4All-Windows-x86_64-v*.zip`。本页速度表是 **0.8.0** 历史实测；当前文档与源码事实以 **0.9.0** tag 为准。


### 2. 解压

将 ZIP **完整解压**到一个目录，例如 `D:\MoE4All`。

### 3. 下载 GGUF 模型

上方 0.8.0 历史实测使用以下两种量化；复现这些数字时请使用对应文件。模型单独下载到本地 SSD。

| 模型 / 组件 | 下载链接 | 文件与用途 |
| --- | --- | --- |
| **Qwen3.6 35B 本体** | [下载 APEX-I-Balanced](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/resolve/main/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf?download=true) | `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`；35B 下载这一个文件即可 |
| **Flash-Next 主模型** | [下载 AD-4.27bpw-Q4_K_M-M64 全部分片](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/tree/main/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64) | 该目录下的 **33 个 GGUF 分片**，全部放在同一文件夹 |
| **Flash-Next 视觉** | [下载 F16 视觉文件](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/resolve/main/mmproj-Qwen3.8-Flash-Next-F16.gguf?download=true) | `mmproj-Qwen3.8-Flash-Next-F16.gguf`；图片理解时加载 |
| **Flash-Next MTP** | [下载 shared Q4_K_M MTP 头](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf?download=true) | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`；本页实测使用的文本加速组件 |

Flash-Next 启动时选择第一片：`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`。
视觉与 MTP 文件可放在主模型目录，在向导中按用途选择。

### 4. 运行

1. 双击解压目录中的 **`Start-INFR-Wizard.cmd`**。
2. 选择“终端聊天”或“OpenAI 兼容 API”，将主模型 GGUF 拖入窗口，按 Enter。
3. 选择自动配置档位：**激进性能**是本页实测使用的档位；**保守**适合首次试运行或后台程序较多时使用。
4. 上下文留空由引擎自动确定，确认启动。复现实测时，20K 输入设置 `32768`，150K 输入设置 `163840`。

自动档默认使用 Q8 K/V，显存、内存、专家缓存与 Ubatch 由引擎规划。
35B 加载本体即可聊天。以下 Flash-Next 选项用于复现上方 **0.8.0 历史实测**：

- **文本 MTP 加速**：启用“Qwen3.8 MTP 单路加速”，选择上表的 MTP 头，验证宽度选 `4`。
  使用 greedy（`temperature=0`）和单会话生成。
- **图片理解**：选择 API 模式，MTP 选择关闭，启用“视觉图片理解”，选择上表的 F16 视觉文件。

0.9.0 已支持服务端 Vision 与 MTP 组合，组合能力和限制见[模型能力矩阵](documentation/reference/model-capabilities.md)。API 默认地址为 `http://127.0.0.1:8080/v1`；请求示例见[API 使用](documentation/guide/serving/api-quickstart.md)，完整配置项见[配置参考](documentation/reference/configuration.md)。

## 实测结果

### 0.8.0：Qwen3.8-Flash-Next，20K 与 150K 输入对照

两组都使用**自动配置：激进性能**。测试指定上下文、Q8 K/V 和采样方式，
其余资源由引擎规划，最终自动选用 `ubatch=4096`。

- **硬件与系统**：RX 7900 XTX 24 GiB、Ryzen 5 5600X、64 GiB DDR4、Windows 11。
- **模型**：`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`；MTP 使用匹配的
  `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`。
- **共同设置**：Q8 K/V、greedy、关闭思考、单路生成、自动激进资源策略。
- **上下文容量**：20K 输入使用 32K 容量；150K 输入使用 `ctx=163840`。

Prefill 为处理输入的速度，Decode 为全程生成速度，单位均为 tok/s。

| 测试负载 | 模式 | 20K Prefill | 20K Decode | 150K Prefill | 150K Decode |
| --- | --- | ---: | ---: | ---: | ---: |
| 复杂问题 | MTP | 937 | **39.9** | 766 | **33.4** |
| 复杂问题 | 无 MTP | 1,034 | **35.0** | 889 | **30.8** |
| 简单问题（高接受率） | MTP | 936 | **60.2** | 769 | **48.1** |
| 简单问题（高接受率） | 无 MTP | 1,035 | **37.7** | 888 | **32.4** |

- **长输入下仍保持生成速度**：从 20K 增至 150K，各组全程 Decode 保留约 **80%–88%**。
- **MTP 收益随内容变化**：复杂问题在 20K / 150K 时分别提速约 **14.0% / 8.4%**；
  简单问题分别约 **59.7% / 48.5%**。
- **输入与输出分别计时**：本组 MTP 的 Prefill 开销有所增加；整次请求耗时由输入处理
  与输出生成两部分共同决定。

<details>
<summary>展开完整分段结果、MTP 接受率与长上下文变化</summary>

以下速度单位均为 tok/s。中段指生成 token 的 25%–62.5%，后段为 62.5%–100%；
Alpha 为 MTP 草稿接受率。

| 输入规模 | 模式与负载 | Prefill | Decode 全程 | 中段 | 后段 | Alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20K | MTP 简单问题 | 936 | 60.2 | 61.4 | 61.3 | 0.994 |
| 20K | MTP 复杂问题 | 937 | 39.9 | 42.0 | 39.7 | 0.594 |
| 20K | 无 MTP 简单问题 | 1,035 | 37.7 | 38.6 | 37.8 | — |
| 20K | 无 MTP 复杂问题 | 1,034 | 35.0 | 35.1 | 36.8 | — |
| 150K | MTP 简单问题 | 769 | 48.1 | 49.2 | 46.8 | 0.979 |
| 150K | MTP 复杂问题 | 766 | 33.4 | 35.5 | 34.3 | 0.642 |
| 150K | 无 MTP 简单问题 | 888 | 32.4 | 32.0 | 32.9 | — |
| 150K | 无 MTP 复杂问题 | 889 | 30.8 | 31.4 | 31.2 | — |

实际输入长度：20K 简单问题 / 复杂问题为 **19,988 / 20,019** token；
150K 简单问题 / 复杂问题为 **149,849 / 149,857** token。

| 模式与负载 | 150K 相对 20K 的 Prefill 变化 | 全程 Decode 变化 |
| --- | ---: | ---: |
| MTP 简单问题 | -17.8% | -20.1% |
| MTP 复杂问题 | -18.2% | -16.3% |
| 无 MTP 简单问题 | -14.2% | -14.1% |
| 无 MTP 复杂问题 | -14.0% | -12.0% |

[20K 测试条件、结果与分析](documentation/evidence/benchmarks/2026-09-24-qwen38-mtp-20k.md)。

</details>

### 0.8.0：Qwen3.6 35B · 无 MTP

**复杂问题在 20K 输入下生成 59.1 tok/s，150K 输入下生成 39.0 tok/s。**

- **模型**：`Qwen3.6-35B-A3B-APEX-I-Balanced`，不启用 MTP。
- **硬件与系统**：与上组相同，RX 7900 XTX 24 GiB、Ryzen 5 5600X、64 GiB DDR4、Windows 11。
- **测试条件**：同样使用自动配置的激进性能策略、Q8 K/V、greedy、关闭思考、单路生成。
  20K 输入使用 32K 上下文容量，150K 输入使用 `ctx=163840`。

Prompt 为实际输入 token 数；Prefill 与 Decode 的单位均为 tok/s。
中段、后段分别为生成 token 的 25%–62.5%、62.5%–100%。

| 输入规模 / 任务 | Prompt | Prefill | Decode 全程 | 中段 | 后段 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20K 简单问题 | 20,041 | 2,484 | **65.6** | 66.7 | 66.8 |
| 20K 复杂问题 | 20,049 | 2,421 | **59.1** | 59.5 | 62.6 |
| 150K 简单问题 | 149,849 | 1,135 | **39.9** | 40.3 | 39.9 |
| 150K 复杂问题 | 149,857 | 1,140 | **39.0** | 39.6 | 39.8 |

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
- **Qwen3.8 MTP（0.8.0 预览）**：可选的单路投机解码，收益取决于草稿接受率，
  使用方式见[快速使用](#快速使用)。
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
