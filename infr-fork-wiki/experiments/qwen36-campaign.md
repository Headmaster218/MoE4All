# Qwen 35B Optimization Campaign

[首页](../README.md) / [实验](README.md) / Qwen 35B campaign

## 实验纪律

- synthetic depth 直接 materialize KV/allocator state，不生成前 250K tokens；
- Prefill 与 Decode 分开看，不假设一个改动同时帮助两者；
- 只有相同 model/quant/KV/cache/depth/batch/profiler 的 A/B 才计算百分比；
- 短 1-rep 只诊断，最终偏好 3/5 reps 或 A-B-B-A；
- profiler 默认关闭，避免观测本身改变 hot path。

## 基础设施

| Commit | 改动 | 价值 |
|---|---|---|
| `ebf5b79` | opt-in Pager profiler | 能区分 lookup/victim/copy/submit/wait/LRU |
| `8c49710` | O(1) LRU promotion | 消除 hit 扫描；后续 profile promotion scan=0 |
| `898ff91` | KV format reporting fix | 防止 F16/Q8 benchmark mislabeled |
| `16fbee2` | synthetic context depth | 让 100K/200K/250K 成为可重复日常测试 |

## Attention 与 recurrent mixer

| Commit | 改动 | 结果/原因 |
|---|---|---|
| `dbc51fe` | hd256 BM16 FA Prefill | pp512 100K 约 164→320，200K 89→229 |
| `9bef28d` | 回收 score matrix reserve | peak 约 540 MiB，reserve 548 MiB；返还容量 |
| `447cd50` | strided DeltaNet Decode | 深场景约 0～2%；去掉 CopyStrided 小 dispatch |
| `276d9c8` | IQ4_NL partial-tile mask | correctness，不制造性能 claim |
| `0ffdefd` | subgroup paged quant Decode | Q4 近持平，IQ4_NL 约 +1% |
| `46c0b88` | hd256 Q8 Decode | 200K 多模型约 +8%～10% 早期提升 |
| `a73d43a` | register-O FA | Balanced 200K 226.8→275.2，+21.3% |
| `02e0bfb` | Prefill output 4 lanes | 100K 约 367→417，200K 279→300 |

## Decode 固定气泡

`ff69e83` 复用 transient recorder buffers/descriptors/resources。这个改动与执行链重构一起，
让 Balanced 100K 达 44.6、200K 32.8 tok/s，早期仅 low-20s。profiling 显示 Decode 不是
device-wide compute/bandwidth 满载，而是大量小 dispatch、CPU submit 和同步间隙。

`5a33e58` 又证明“名字更专用的 kernel”不一定更快：RDNA3 generic hd256 在 100K 56.2
vs specialized 54.0，200K 42.8 vs 41.9，最终选择 generic。

## Q8 Decode 累积优化

| Commit | 技术 | 代表变化 |
|---|---|---:|
| `d72f60f` | QK 按 Q8 block clustering | +7.6% |
| `e6b6137` | LS64→128 | +2.1% |
| `633638b` | LS128→256 / more waves | +1.9% |
| `84fb844` | packed fp16 QK | +3.6% |
| `0b37574` | packed Q8 value dequant | +2.5% |
| `15eb5f7` | chunk 1024 | +6.3% |

完整序列 200K 约 31.55→39.65（+25.7%）。不是把这些百分比简单相加；每一步基于前一步
binary。

## Pager 与 Host feed

| Commit | 改动 | 机制/结果 |
|---|---|---|
| `b8cbc52` | configurable upload ring slots | CPU prep/upload/GPU work 可重叠；4/8 slot 有效，3/5 很差 |
| `6afab3a` | cold victim scan batch | 解决此前平均 114～128 victim steps |
| `3651e29` | expert staging copy batch | 5 GiB、200K pp512 达 412.3；d0/16 GiB 达 823.1 |
| `ce97e4f` | layer-major Host Store + A/B | 架构有价值，但 GPU-pull d0 2672.5→2438.9，200K 522.3→492.6 |
| `354c0c3` | CPU direct ReBAR push | d0 2672.5→2977.8（+11.4%）；200K 492.6→515.5 |
| `2d57e7b` | Down upload overlap | full-RAM 中 UG compute 与 D copy overlap |
| `5a1faeb` | skip needless Down submits | Down resident 时不建立无效 wait/submit |

Host Store 的第一次实现退步但没有立即推翻架构，因为它建立了 contiguous layer layout；
后续 CPU push 去掉 GPU-visible Host/staging mirror 后，才兑现 d0 收益。这是“架构中间态可
退步，但必须有明确下一步机制”的例子。

## Prefill ring

静态 A/B 两 lane 的问题：

- DeltaNet 很快，旧 copy 未完时又释放新槽；
- Attention 很慢，可能释放多个槽，但 fixed state machine 补一个就停。

`637f23f` 改为异步 refill：计算开始/结束只发送 slot availability，refill worker 只要有
空槽就继续搬未来 layer。Qwen 约 1 Attention + 3 DeltaNet，5 slots 是良好几何；实际 lane
数仍受 VRAM budget 限制。

大矩阵显示同 depth 下 6～14 GiB Prefill 吞吐多数差异约 1%，说明 ring 基本消除了 Expert
cache 容量对 Prefill 的影响，长 context 主要由 Attention 决定。

## 里程碑总结

匹配条件下 `16fbee2`→`02e0bfb`：

| 操作 | d0 | 100K | 200K |
|---|---:|---:|---:|
| Prefill | +16.5% | +153.7% | +236.5% |
| Decode | +129.0% | +132.7% | +83.6% |

后续工作没有继续维持同样倍数，因为瓶颈从错误路径/巨大 bubble 转移到真实 Attention 和
RAM→VRAM 字节量。优化不是“越做百分比越大”，而是逐层暴露下一瓶颈。

---

[实验索引](README.md) · [Qwen 模型页](../models/qwen36-35b.md) ·
[失败实验](rejected-experiments.md)
