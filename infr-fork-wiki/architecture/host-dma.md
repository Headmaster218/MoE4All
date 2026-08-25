# RAM → VRAM：ReBAR Push 与 Vulkan Host DMA

[首页](../README.md) / [架构](README.md) / Host DMA

## 为什么 RAM→VRAM 成为最大项

GPU miss 不论 RAM hit 还是 SSD hit，最终都要把完整 Expert bytes 写入 VRAM。SSD miss 的
数据流是 SSD→RAM→VRAM，所以 Host→VRAM 计数包含全部 GPU misses；SSD 只包含 RAM 也
miss 的子集。

DeepSeek V4 每 token：

- Host→ReBAR：1.053 GiB；
- SSD demand：0.404 GiB。

因此“SSD 慢”成立，但不能忽略更大的 RAM→VRAM 总流量。

## 路径 1：CPU 直接写 mapped ReBAR

统一 GPU arena 是 `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`，CPU 可直接把 Host Store
复制到最终 slot。优点：

- 无 staging mirror；
- 地址和 copy 流程简单；
- 可以与 GPU compute 并发；
- import/DMA 不可用时始终正确。

缺点：mapped VRAM 是 write-combined PCIe memory，单线程实测约 8.8 GB/s；并行批量路径
通常约 14～19 GB/s。35B 早期 200K Prefill 17.63 GiB push 约 19.18 GB/s、0.99 s。

## 路径 2：普通 RAM 原地导入 Vulkan

使用 `VK_EXT_external_memory_host`：

1. Host Pager 以满足 `minImportedHostPointerAlignment` 的方式分配 RAM；
2. 同一 host pointer 创建 imported Vulkan buffer；
3. RAM ownership 仍属于 Host Pager，Vulkan object 只提供 alias/view；
4. `vkCmdCopyBuffer` 将多个 region 搬到 unified VRAM slots；
5. cache 释放前先销毁 import view，避免悬挂。

关键点：没有第二份 45 GiB staging cache，SSD 仍直接填原来的 RAM slots。

## 微基准上限

专用 Vulkan queue matrix（payload 为 1～8 个约 5.06 MiB unit）：

| Payload | H2D universal | H2D compute | H2D transfer-only | D2H transfer-only |
|---:|---:|---:|---:|---:|
| 5.06 MiB | 18.9 | 20.3 | 19.5 | 19.9 GiB/s |
| 20.25 MiB | 24.0 | 23.1 | 23.9 | 23.0 GiB/s |
| 40.50 MiB | 25.0 | 25.1 | 24.9 | 25.1 GiB/s |

单方向大 payload 约 25 GiB/s，比生产 CPU push 14～19 GiB/s 有约 30%～70% 潜力。

## “全双工”测到了什么

同一测试的 H2D+D2H aggregate 在不同 queue 组合下约 20～26.5 GiB/s，而不是理想的
约 50 GiB/s。AMD 文档中的硬件双向能力不等于 WDDM/Vulkan 当前 workload 自动双满：

- copy engines/queue family 可能仍共享调度或链路瓶颈；
- host memory read/write 方向与 cache/coherency 会互相影响；
- 小 batch 固定成本明显；
- queue submit 并发不保证物理 DMA engines 真正独立。

所以当前收益重点是提高 H2D，不依赖同步 D2H write-back 才成立。

## 29 GiB import ceiling

该 Windows 驱动只能累计 import 约 29 GiB ordinary host memory。即使先分配 16 GiB ReBAR，
仍可达到相同 ceiling，说明它不是简单的“VRAM+Host 共 32 GiB aperture”。

生产策略：

- 尽量 import；
- 驱动返回 limit 后停止尝试；
- 剩余 RAM arena 保留 CPU push；
- 服务仍正常加载和运行。

## 为什么按 pool 比例分配 import

122B 三个 pool 大小和访问频率不同。first-come 可能得到类似 `[pool0 很多, pool1 少量,
pool2 0]`，使某类量化几乎永远走慢路径。最终每次选择当前 `imported/total` 比例最低的
arena，平局优先大 arena。

45 GiB RAM 运行中导入 28.99 GiB：

| Pool coverage | 比例 |
|---|---:|
| Pool 0 | 62.6% |
| Pool 1 | 67.7% |
| Pool 2 | 64.9% |

tg256 从 first-come 22.4 提到 proportional 23.2 tok/s（+3.6%）。

## 当前生产限制

- DMA source 只有 imported prefix；每个 pool 后缀仍 CPU push。
- copy 当前与 compute 的队列编排仍可继续研究，尚无稳定 separate-transfer overlap 策略。
- GPU→RAM 的直接 CPU read 约 44 MB/s，不能用于 victim write-back。
- 纯 D2H Vulkan DMA 微基准可快，不代表现有 Pager 已具备安全、无阻塞、带 ownership 的
  background eviction pipeline。
- copy 优先级没有一个可简单设置为“永远高于 compute”的通用 Vulkan API；正确方法是
  queue/submit/timeline 设计和足够大的 batch，而不是 CPU thread priority 幻想。

---

[架构索引](README.md) · [RAM/SSD cache](ram-ssd-cache.md) ·
[Qwen 122B](../models/qwen35-122b.md)
