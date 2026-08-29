# Windows 本地大模型性能矩阵（2026-08-29）

[返回主 README](../../README.md)

本文记录 MoE4All 在同一台原生 Windows 11 主机上运行五个本地 MoE 模型的
性能取向资源矩阵。测试覆盖两档显存与进程 RAM 总预算，以及 0、128K、250K
三种 context depth。表中同时保留实际运行 ubatch 和明确的容量边界。

## 测试环境

| 项目 | 配置 |
|---|---|
| 日期 | 2026-08-29（Asia/Shanghai） |
| infr commit | `d261a7e9532168c0b5eb69aca3015c3c3147bf4c` |
| 操作系统 | Windows 11 专业版，build 26200 |
| GPU | AMD Radeon RX 7900 XTX 24 GiB，Vulkan0 |
| 驱动 | AMD `32.0.23033.1002` |
| CPU | AMD Ryzen 5 5600X，6 核 12 线程；测试使用 10 个 CPU 线程 |
| 内存 | 64 GiB DDR4 |
| D 盘 | Samsung MZVL21T0HCLR-00B00，NVMe |
| G 盘 | Acer SSD FA100 512GB，NVMe |

## 模型

| 简称 | 模型与量化 | 约占用 | 磁盘 | KV |
|---|---|---:|---:|---|
| Qwen3.5 122B | Qwen3.5-122B-A10B APEX-I-Quality | 72.3 GiB | G: | Q8_0 K/V |
| Qwen3.6 35B | Qwen3.6-35B-A3B APEX-I-Balanced | 23.9 GiB | G: | Q8_0 K/V |
| Ling 3 Flash | Ling-3.0-Flash Q5_K_M | 83.3 GiB | G: | F16 K/V |
| Qwen3.8 Q2 | Qwen3.8-Flash-Next Q2_K_XL | 73.5 GiB | D: | Q8_0 K/V |
| Qwen3.8 IQ4 | Qwen3.8-Flash-Next IQ4_XS | 87.3 GiB | D: | Q8_0 K/V |

磁盘列只记录模型实际位置。模型架构、量化和大小不同，不能用跨模型吞吐差异
反推 D/G 盘的纯磁盘性能。

## 方法

- 16GB+32GB 档设置 `device.vram_budget=14g`、`device.ram_budget=22g`；
  24GB+64GB 档设置 `device.vram_budget=20g`、`device.ram_budget=50g`。
- `device.ram_budget` 是 infr 的进程 RAM 规划目标，不是 Windows Job Object 的硬
  Working Set 上限。GGUF 映射页、驱动分配和加载期临时页可能让任务管理器读数更高。
- Decode 使用 `-p 0 -n 64 -r 3`。Prefill 使用 `-p <ubatch> -n 0 -r 3`，每次只
  prefill 一个 ubatch；表中括号记录 infr 最终实际使用的 ubatch。
- benchmark 使用四组固定数学提示词：第一组做不计时的同形状预热，后三组正式测量。
  表中 tok/s 是 CLI 对三次正式结果给出的汇总值。
- Depth 0 使用 `--ctx 8k`；128K 使用 `--synthetic-depth 131072 --ctx 136k`；
  250K 使用 `--synthetic-depth 256000 --ctx 256k`。
- 设置 `paging.host_dma=true`、`kv.overflow=false`，并清除调用者环境中的 `INFR_*`
  变量。除 Ling 外使用 Q8_0 K/V；Ling 当前使用 F16 K/V。
- Decode 和非 Ling prefill 显式禁用 submit splitter。Ling prefill 固定 `split/16`
  以控制 Windows TDR 风险；失败时逐级降低 ubatch，表中只记录稳定完成的配置。

`CAP` 表示即使降至本轮最小 ubatch，剩余显存仍低于一个完整专家 streaming lane
所需的最小工作集。

## 16GB VRAM + 32GB RAM 档

配置预算为 14 GiB VRAM、22 GiB 进程 RAM。

| 模型 | Depth | Decode tok/s（u） | Prefill tok/s（u） | KV |
|---|---:|---:|---:|---|
| Qwen3.5 122B | 0 | 3.4（256） | 19.3（512） | Q8_0 |
| Qwen3.5 122B | 128K | 6.0（64） | 4.1（64） | Q8_0 |
| Qwen3.5 122B | 250K | CAP | CAP | Q8_0 |
| Qwen3.6 35B | 0 | 56.1（256） | 1,576.6（2048） | Q8_0 |
| Qwen3.6 35B | 128K | 51.9（256） | 802.4（2048） | Q8_0 |
| Qwen3.6 35B | 250K | 42.7（128） | 413.5（1024） | Q8_0 |
| Ling 3 Flash | 0 | 4.7（256） | 28.6（1024） | F16 |
| Ling 3 Flash | 128K | 0.6（256） | 7.5（128） | F16 |
| Ling 3 Flash | 250K | 0.3（128） | 4.6（64） | F16 |
| Qwen3.8 Q2 | 0 | 15.7（256） | 329.2（1024） | Q8_0 |
| Qwen3.8 Q2 | 128K | 14.5（256） | 274.0（2048） | Q8_0 |
| Qwen3.8 Q2 | 250K | 12.5（128） | 171.1（1024） | Q8_0 |
| Qwen3.8 IQ4 | 0 | 11.2（256） | 137.5（1024） | Q8_0 |
| Qwen3.8 IQ4 | 128K | 9.3（256） | 66.8（512） | Q8_0 |
| Qwen3.8 IQ4 | 250K | CAP | CAP | Q8_0 |

## 24GB VRAM + 64GB RAM 档

配置预算为 20 GiB VRAM、50 GiB 进程 RAM。

| 模型 | Depth | Decode tok/s（u） | Prefill tok/s（u） | KV |
|---|---:|---:|---:|---|
| Qwen3.5 122B | 0 | 5.8（512） | 209.6（2048） | Q8_0 |
| Qwen3.5 122B | 128K | 9.9（512） | 98.4（1024） | Q8_0 |
| Qwen3.5 122B | 250K | 7.3（256） | 48.4（512） | Q8_0 |
| Qwen3.6 35B | 0 | 84.7（512） | 3,208.8（4096） | Q8_0 |
| Qwen3.6 35B | 128K | 62.3（512） | 841.7（4096） | Q8_0 |
| Qwen3.6 35B | 250K | 50.9（256） | 460.1（2048） | Q8_0 |
| Ling 3 Flash | 0 | 5.3（512） | 105.8（2048） | F16 |
| Ling 3 Flash | 128K | 0.7（512） | 9.8（128） | F16 |
| Ling 3 Flash | 250K | 0.3（256） | 5.2（64） | F16 |
| Qwen3.8 Q2 | 0 | 17.3（512） | 684.0（4096） | Q8_0 |
| Qwen3.8 Q2 | 128K | 20.3（512） | 513.9（4096） | Q8_0 |
| Qwen3.8 Q2 | 250K | 23.0（256） | 424.4（2048） | Q8_0 |
| Qwen3.8 IQ4 | 0 | 15.6（512） | 468.5（4096） | Q8_0 |
| Qwen3.8 IQ4 | 128K | 15.6（512） | 294.1（1024） | Q8_0 |
| Qwen3.8 IQ4 | 250K | 16.5（256） | 311.0（2048） | Q8_0 |

## 容量与稳定性

- Qwen3.5 122B 在 14 GiB VRAM、250K 下只剩约 3,215 MiB 专家缓存，低于一个
  完整 prefill 专家层所需的约 4,698 MiB，因此 decode/prefill 都记为 `CAP`。
- Qwen3.8 IQ4_XS 在 14 GiB VRAM、250K 下只剩约 2,095 MiB，低于约 2,838 MiB
  的完整专家层最小工作集，因此 decode/prefill 都记为 `CAP`。
- Ling 的 128K prefill 需要降至 ubatch 128，250K 需要降至 ubatch 64；更大值即使
  使用 `split/16` 仍可能触发 Vulkan device-lost。表内均为降档后稳定完成的结果。
- 低资源 Qwen3.8 Q2 的 depth 0 prefill 在 ubatch 2048 遇到统一 VRAM 连续窗口
  限制，降至 1024 后完成。IQ4 的部分配置也由统一 arena 自动降低实际 ubatch。
- 50 GiB RAM 档中，Windows AMD 驱动通常只能将约 29--31 GiB host RAM 导入
  Vulkan DMA；剩余 RAM 自动回退到 CPU ReBAR copy，不会失去 RAM cache 功能。

## 如何解读

- Qwen3.6 35B 最适合更大显存档：depth 0 decode 从 56.1 提升到 84.7 tok/s，
  prefill 通过更大 ubatch 从 1,576.6 提升到 3,208.8 tok/s。
- Qwen3.5 122B 的低 RAM 档仍大量依赖 SSD。增加到 50 GiB 进程 RAM 后，depth 0
  prefill 从 19.3 提升到 209.6 tok/s，但不同提示词的专家路由使 decode 波动较大。
- Ling 的 depth 0 prefill 会从更多 RAM 和更大 ubatch 获益；128K/250K decode
  几乎不随 expert cache 增加而改善，当前主要受 F16 MLA/KV 长上下文扫描限制。
- Qwen3.8 Q2 在低资源档比 IQ4_XS 更快，也能在 14 GiB VRAM 下运行 250K；IQ4_XS
  需要 20 GiB VRAM 才能跨过 250K 的专家最小工作集边界。
- Synthetic depth 构造真实 KV allocation、访问长度和 allocator 状态，但 KV 内容
  没有语言意义，会改变后续 activation、专家路由和 cache 热度。因此长深度结果适合
  衡量执行成本与容量，不应被解读为“真实长对话一定比 depth 0 更快”。

这些数字是性能取向工程快照。三组不同提示词有意保留了专家路由和冷数据差异；若要
判断小于约 5% 的内核变化，仍应固定单一 workload、冷热状态并交替运行取中位数。
