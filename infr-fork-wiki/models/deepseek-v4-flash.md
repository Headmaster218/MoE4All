# DeepSeek V4 Flash

[首页](../README.md) / [模型](README.md) / DeepSeek V4 Flash

## 目标与最终结论

DeepSeek V4 Flash 的接入目标分为四段：MXFP4 GGUF 主模型、无 DSpark 正确运行、FP8 KV
和 FP4 indexer/压缩缓存，最后才是 replay/Prefill/Decode 性能。

前三段完成，模型可以在 native graph 端到端生成。性能 campaign 最终收尾在约
**4.3 Decode tok/s**（12.67 GiB VRAM expert + 40 GiB bounded RAM/SSD）。17～25 tok/s
目标在这套存储层级下不现实：dominant cost 是 expert miss traffic，不是少一个快 5× 的
kernel。

## 新增模型语义

- FP8 KV cache。
- MXFP4 indexer cache 与对应 gather/compress/top-k/write ops。
- raw sliding-window + CSA/HCA/LID compressed cache 状态机。
- HyperConnection / Sinkhorn residual mixing。
- MXFP4 paged experts。
- V4 特定 cache plan、rope/indexer layout 与边界语义。
- DSpark speculative module 未实现。

这些内容详见 [DeepSeek V4 kernel 与状态机](../kernels/deepseek-v4.md)。

## 最终缓存实测

| 项目 | 值 |
|---|---:|
| Expert payload | 147.17 GB，33,024 blocks |
| 单个 MXFP4 role block | 4.25 MiB |
| VRAM expert arena | 12.67 GiB，3,052 blocks |
| Inclusive RAM | 40 GiB，9,637 blocks |
| GPU hit | 67.21% |
| RAM conditional hit | 61.62% |
| VRAM-or-RAM combined hit | 87.41% |
| SSD demand | 51.75 GiB / 128 tokens |
| SSD demand/token | 0.404 GiB |
| Host→ReBAR traffic/token | 1.053 GiB |

每 token 访问 774 blocks。即使 SSD 只负责 GPU+RAM 都 miss 的部分，RAM→VRAM promotion
仍服务全部 GPU miss，所以 Host→VRAM 字节量比 SSD demand 更大。

## 为什么保留 inclusive shadow

曾考虑减少 RAM 中与 VRAM 重复的 shadow，把更多独占专家放进 RAM。问题是 VRAM victim
若无 RAM 副本，就必须：

- 从 GPU 回写到 RAM；或
- 未来需要时重新从 SSD 读。

实测 mapped ReBAR 的 GPU→RAM 读回只有约 44 MB/s，完全不能进入每 token 关键路径。
丢弃 shadow 的 A/B 也从 4.0 降到 3.9 tok/s，RAM conditional hit 61.6% 降到 60.6%。
因此保留 inclusive shadow，用 RAM 容量换取 metadata-only VRAM eviction。

## 保留的优化

| 改动 | 结果 | 为何保留 |
|---|---:|---|
| Decode HyperConnection Sinkhorn | cache-hot 约 8.0 → 8.7 tok/s | 针对一 token/hc=4，减少串行小算子 |
| Windows concurrent block reads | 单独 fanout 不显著 | 是 request-level batching 的必要基础 |
| Concurrent host promotions | 支持 RAM/SSD 并发 fill | bounded tier 通用能力 |
| Gate+Up+Down batch | copy calls 11,008 → 5,504；push 5.9 → 6.9 GB/s | 固定成本减半 |
| Complete-block MXFP4 Decode | 3.9 → 4.3 tok/s | scale/address work 在 tile 内共享 |
| F32 vec4 GEMV | 79.5 → 23.7 ms，约 3.35× | 通用一行 F32 linear 获益 |

F32 GEMV 提速没有让端到端超过 4.3，反而是“已经不是 kernel bottleneck”的直接证据。

## 否决的优化

- unlimited submit：6.8 vs 6.9 tok/s，且其他运行会退步；保留 split/16。
- DP4A MXFP4：后期可比运行 5.9～6.3，劣于 complete-block decode。
- F16 HyperConnection temporary：7.3～7.5，F32 是 8.2～8.3。
- 四 dispatch HyperConnection：并行收益被 dispatch cost 抵消。
- non-temporal ReBAR write：2.7～2.8 且不稳定。
- 8-way request fanout：3.7 vs 3.8，线程/请求固定开销过高。
- 丢弃 inclusive shadow：负收益且破坏便宜 eviction。

## trace 模拟告诉了什么

模型：

```text
token_ms = 72.57
         + 0.2396 * GPU_miss_blocks
         + 1.0170 * RAM_miss_blocks
```

在该 128-token trace 上，RAM 60 GiB 已覆盖 trace 中出现的约 59 GiB working set，因此
模拟速度平台约 7.5～7.8 tok/s；这不代表 60 GiB 能装下 147.17 GB 完整 payload。更长或
不同对话会访问 trace 未见专家。

IQ3_M 只按 block bytes ×0.873 做 size-effect 估计，60 GiB 平台约 8.2～8.5 tok/s；它
不是 IQ3_M kernel benchmark。

## 对其他模型的影响控制

V4 ops 只有 V4 graph 会发出。通用改动包括 resident-BDA allocator、Windows block I/O、
bounded Host Pager、MXFP4 Decode 和 F32 GEMV。`f6ca35c` 曾误让 full-RAM Qwen/Ling 失去
Down overlap，`7935cf8` 将 all-role batch 限定回 bounded tier，保护其他模型。

## 收尾决定

- 保留正确性支持和 route trace；
- 保留 inclusive full shadow；
- 13.47 GiB 是已成功加载的最大 expert arena，14.5/15 GiB 仅是模拟点；
- 没有新的存储配置或更长代表 trace 前，不继续小 kernel sweep；
- 将工程重心转回 Ling/Qwen 体系。

---

[模型索引](README.md) · [DeepSeek 数据](../reference/deepseek-v4-data.md) ·
[失败实验](../experiments/rejected-experiments.md)
