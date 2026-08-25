# hd256 Attention 与 Q8 KV

[首页](../README.md) / [Kernels](README.md) / Attention

## 原始问题

Qwen 35B 的 Attention head dim 为 256。早期深上下文 Prefill 会回落到大 score matrix
路径，显存和带宽成本随 query×KV 增长；Q8 KV 虽省 46.9% bytes，但 Decode inline
dequant 在 200K 比 F16 Attention 慢约 29%。

两者分别需要 Prefill 和 Decode 专用设计，不能用“Q8 更小所以自然更快”代替 profiling。

## hd256 FlashAttention Prefill

### BM16 路径

`dbc51fe` 新增 hd256 BM16 FlashAttention 和 combine，避免物化完整 score matrix，并满足
Windows 32 KiB shared-memory 限制。

早期代表收益（Balanced pp512）：

| Depth | 之前 | hd256 FA 阶段 | 变化 |
|---|---:|---:|---:|
| 100K | 约 164 | 约 320 tok/s | 约 1.95× |
| 200K | 约 89 | 约 229 tok/s | 约 2.57× |

### Activation reserve

score matrix 消失后，旧 planner 仍预留过多 scratch。实测 peak 约 540 MiB，最终 reserve
548 MiB。此改动主要返还可用 VRAM/context，不宣称 kernel tok/s 提升。

### Register-O

`a73d43a` 将 output accumulator 留在寄存器，减少 score/output 中间流量。Balanced 200K
Prefill 226.8 → 275.2 tok/s（+21.3%）；三模型 200K 提升约 20～23%。

### 输出四 lane

`02e0bfb` 恢复 output dimension occupancy：Balanced pp512 约：

- d0：487 → 509；
- 100K：367 → 417；
- 200K：279 → 300 tok/s。

## Split-K 为什么没有强制默认

isolated kernel 在 100K 下 2 splits 从 71.4 降到 65.3 ms（快 8.5%）；但 200K auto 与
4 splits 为 124.1 vs 124.4 ms，持平。跨模型 whole-run 还受 Expert Pager 噪声影响，Q4
路径不稳定/退步。

`flash_splits` 是全局开关，不知道 depth/device/model。最终保留 auto policy 和显式 override，
不把一个 100K isolated winner 设为所有场景默认。

## Q8 KV Decode

### 初始归因

3 模型、40-token profiled runs：

| Depth | F16 `attn_decode_hd256` | Q8 `attn_partial_q8_bda` | Q8/F16 |
|---:|---:|---:|---:|
| 100K | 406.8 ms | 535.0 ms | 131.5% |
| 200K | 807.5 ms | 1.04 s | 128.8% |

Q8 write path总计不到约 1 ms；Regression 在读/解码/Attention 内核，不是写 cache。

### 累积优化序列

| 改动 | 200K 代表结果 | 机制 |
|---|---:|---|
| Q8 专用 hd256 path | 17.1 → 18.4（早期基线） | scale/block-aware |
| 按 quant block cluster QK | 31.55 → 33.95，+7.6% | scale/code 复用，depth split 32→8 |
| LS64→128 | 34.0 → 34.7，+2.1% | 更高 workgroup occupancy |
| LS128→256 | 34.35 → 35.0，+1.9% | 更多 wave parallelism |
| packed fp16 QK | 35.1 → 36.35，+3.6% | 降低寄存器/带宽压力 |
| packed value dequant | 36.3 → 37.2，+2.5% | PV 侧流量/转换减少 |
| chunk 1024 | 37.3 → 39.65，+6.3% | combine 减半，前置并行已足够 |

完整序列约 31.55 → 39.65（+25.7%）。关键教训：chunk 1024 在前面的 D8/LS/QK/PV
改造之前曾是 -3%；优化的收益依赖 occupancy 与并行结构，不能孤立移植最终参数。

## 失败路径及原因

### Direct planar-Q8 Prefill FA

100K -6.3%，200K -8.9%。GQA/query tiles 重复解码同一 compact KV，而旧的一次性 Q8→F16
prepass 只占 device time 约 0.62%。未来只有在多个 GQA heads/tiles 共享 dequant 后才值得。

### Scale-once shared decode

31.45 → 29.9（-4.9%）。减少 scale load 却降低 independent lane work/occupancy。

### 更少 combine tiles

ntile4 39.3、ntile2 38.3、ntile1 35.4。少重复 exp/max 不等于更快，parallelism 的损失
更大。

### 更大 chunk

1536 持平，2048 -4.1%，4096 -7.7%。combine 更少，但 pass-1 serial QK/PV 变长，GPU
无法用其他 waves 隐藏延迟。

### f16 PV/raw scale/packed unpack

分别退步约 1.2%、2.3%、10.8%。更窄表示或“看起来更整齐”的 unpack 并不保证编译器生成
更好的 RDNA3 指令/寄存器布局。

## 深上下文最终判断

- Q8 的核心价值是 KV byte saving + 允许更多 Expert cache，不只是 Attention kernel 本身。
- 250K Prefill 的长 FA window 已接近 memory-unit bound，小的 host 优化百分比会下降。
- Decode 仍同时受 Attention depth 和 Expert miss 影响；必须把相同 cache/depth/KV 的 A/B
放在一起。

---

[Kernels](README.md) · [Qwen 35B](../models/qwen36-35b.md) ·
[完整 campaign](../experiments/qwen36-campaign.md)
