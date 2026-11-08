

---

## Round 56：GPU 时间轴碎片化 —— 每层 2.39 段、每段间空 1.34ms

### 数据

```
decode 12 tok @ 6.2 tok/s
span = 4.41s
main_intervals = 1149        <- GPU 作业段数
main_busy      = 2.86s
outside        = 1.54s       <- 段与段之间的空档
```

推导：

| 量 | 值 |
|---|---|
| 每 token 段数 | 1149 / 12 = **96** |
| 每层段数 | 1149 / (12×40) = **2.39** |
| 平均每段 busy | **2.49 ms** |
| 平均每段空档 | **1.34 ms** |
| 空档 / busy | **0.54** |

**GPU 每干 2.5ms 就要停 1.3ms。**

### 已排除的解释

`record_lifetime = 400.6ms`，只占 span 的 9%，**不足以解释 1.54s 空档**。
所以不是"CPU 记录太慢"这一个简单答案。

### 另一个可疑点

`dedicated DMA: gpu_timed_submits=0 gpu_timed_bytes=0`、`dma_busy=0ns`。

**DMA 完全不在 GPU 时间轴里统计。** 如果 decode 期间真的有 DMA 在传专家，
那段传输时间不会出现在 `main_busy`，而会以"空档"的形式算进 `outside`。

**这正好解释了 1.54s 空档的可能来源：GPU 在等 DMA 传完，但计时器看不到。**

### 下一步（可 A/B）

如果是 DMA 在占空档，做法是把传输和计算**重叠**起来（`dma_hidden` 目前是 0.0%）。
`INFR_PAGER_RING_SLOTS` 控制 staging ring 槽数——槽多了才可能重叠。

如果槽数已经足够而仍未重叠，那就是同步结构的问题，不是资源问题。

### 说明

这轮我没能让 `INFR_PROF_OPS` 工作（`infr_profile` feature 没编进二进制），
并且判断它即使能用也只说明"忙在哪"，**答不了"为什么闲"**——所以没有为它专门编译。

