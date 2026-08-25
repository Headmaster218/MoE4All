# 被否决或暂缓的实验

[首页](../README.md) / [实验](README.md) / Rejected experiments

## 阅读规则

“否决”表示在记录条件下不值得进入默认 hot path，不表示永远不可能。只有当硬件、模型
shape、队列架构或前置优化发生明确变化时，才应该重新打开。

## Attention / Q8

| 实验 | 结果 | 决策依据 |
|---|---:|---|
| Submit cap 64～768 | 最好 +0.34%/0.47% | 小于 run drift；discrete GPU 保持 unlimited + TDR guard |
| Adaptive warmup | 28.6→41.4→60.0，仍 25～28% spread | 暖的是 cache state，不是收敛；回退 |
| 强制 FA split-K | 100K isolated +8.5%，200K/跨模型不稳 | 保留 auto + override |
| hd128 cooperative-matrix BDA | 约 0.8× | scalar address/load 低效，保持 opt-in |
| Prefill KV BDA hd256 | 100K +0.3%，200K 略负 | 噪声/负，不切默认 |
| Direct planar-Q8 Prefill | 100K -6.3%，200K -8.9% | GQA tiles 重复 dequant |
| Q8 scale-once shared | -4.9% | occupancy 损失大于 scale load 节省 |
| combine tile 4→2→1 | 39.3→38.3→35.4 | parallelism 下降 |
| chunk 1536/2048/4096 | flat / -4.1% / -7.7% | pass-1 serial latency 变长 |
| combine-128 specialization | -12.2% | output/head parallelism 不足 |
| parallel-32 combine | 34.95→35.0 | 无端到端意义 |
| raw-f16 scale | -1.2% | conversion/register 行为更差 |
| f16 PV weight | -2.3% | 没映射到更快 instruction path |
| packed Q8 unpack rewrite | -10.8% | shuffle/unpack 和寄存器更差 |

## Pager / Prefill / Cache

| 实验 | 结果 | 决策依据 |
|---|---:|---|
| Ring slots 3/5 | 约 289/284 tok/s；2/4 约 344/376 | slot 数不单调，allocation geometry/lane reuse 重要 |
| Smaller host copy chunks | 4096B 390.2；512B 323.2；256B 289.4 | task/copy fixed overhead 爆炸 |
| Layer mode 本身 | pp512 415 vs 418～420 | 价值是连续 transfer，不是粒度本身 |
| 第三 ReBAR lane | 未做 | A/B 已耗约 660 MiB；先证明收益 |
| 8:7 UG/Down + paired eviction | 41.50→37.65，miss +5.72% | 固定 quota 破坏冷热 |
| Down weight 7 | 40.65→40.55 | 短测假收益未通过长 confirmation |
| Down weight 5 | 39.10，miss 114,750 | 明显 churn |
| 复杂 UG→D tier branch | 模拟多慢 0.1～3.1 ms | compute window 太短，second submit 太贵 |

## DeepSeek V4

| 实验 | 结果 | 决策依据 |
|---|---:|---|
| Wider MXFP4 dqblk | 6.8 vs 6.9 | 无收益 |
| Unlimited submit | 6.8 vs 6.9，其他 run 退步 | 保持 split/16 |
| DP4A MXFP4 | 后期 5.9～6.3 | 劣于 complete-block decoder |
| F16 HyperConnection temp | 7.3～7.5 vs F32 8.2～8.3 | 负收益 |
| HyperConnection 4 dispatch | 8.2 vs 8.3～8.4 | dispatch 抵消并行 |
| Non-temporal ReBAR writes | 2.7～2.8，不稳 | 不保留 |
| File handle fanout alone | 2.7～2.9 | 缺 request batching |
| 8-way request fanout | 3.7 vs 3.8 | fixed/thread overhead |
| Drop inclusive shadow | 3.9 vs 4.0；RAM hit 60.6 vs 61.6% | 负收益且 victim recovery 变贵 |
| Flat outer parallel copy | 3.9 vs 3.9 | 中性 |

## 显存/缓存设想

### VRAM compaction

曾设想淘汰冷 Experts 后把物理末尾 Experts 搬到前面，制造连续 Embedding/Vision range。
最终用 cold contiguous window eviction + high-address allocator，避免 VRAM→RAM 44 MB/s
回读和 LUT 全量重定位。除非 future allocator 证明 window eviction 无法满足需求，不重开。

### Dynamic KV

理论上 session 前期可让未使用的 KV 容量变成 Expert cache，随 context 增长再收回。收益是
短对话更高 hit；代价是 KV 地址/graph 稳定性、迁移、OOM 边界和恢复复杂度。当前 KV 约
2～3 GiB（250K/Q8）且固定预算已可控，暂缓。

### Exclusive RAM hierarchy

容量利用看似更高，但无可用 VRAM victim write-back。只有未来 Vulkan D2H DMA 能在真实
计算窗口后台稳定完成、且 ownership/queue 不引入等待，才重新评估 shadow 比例。

### SSD router prefetch

当前没有能在 router 结果前准确预测下一 token experts 的通用信号。盲目预取可能污染 RAM
与 VRAM。应先用长 trace 做 precision/pollution 模拟。

## 如何重开一项实验

必须同时写出：

1. 哪个前置条件已经变化；
2. 原失败机制为何不再成立；
3. 一个隔离微基准；
4. 一个端到端 A/B；
5. correctness/other-model guard；
6. 失败时能完整回退的边界。

---

[实验索引](README.md) · [Benchmark method](../reference/benchmark-method.md) ·
[证据索引](../reference/evidence-index.md)
