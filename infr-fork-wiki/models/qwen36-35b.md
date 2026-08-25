# Qwen3.6-35B-A3B APEX-I Balanced

[首页](../README.md) / [模型](README.md) / Qwen3.6-35B

## 它为何是整个 fork 的主线

Qwen3.6-35B-A3B 同时包含 Attention、gated DeltaNet、shared expert 和 routed MoE，且
Balanced GGUF 中 expert matrix 存在 Q5_K/Q6_K 两种物理尺寸。它足够大，能暴露 host
搬运和显存命中率问题；又能把完整 23.57 GB routed-expert payload 放进 64 GiB RAM，
允许先把 SSD 变量排除，专注于 GPU kernel、调度和 RAM→VRAM。

后来的全局池、Prefill layer ring、统一 VRAM、full-RAM backing、shared-expert fusion 和
Host DMA 都首先在这条模型线上形成或得到兼容性保护。

## 模型访问形态

- 40 个 expert layer；每层激活 8 个 routed experts，另有 shared expert。
- Attention 与 gated DeltaNet 混合；Attention 约每四层出现一次。
- Balanced expert payload 共 23,571,988,480 bytes。
- 10 层主要为 Q6_K expert matrix，30 层主要为 Q5_K；三种 role 是 Gate/Up/Down。
- 每轮 router 结果使 Gate/Up/Down 具备相同 expert identity，但三块矩阵仍可独立驻留。

## 最终执行结构

### Prefill

```text
顺序进入 layer N
  ├─ Attention 或 DeltaNet/KDA 类 mixer 在 GPU 计算
  ├─ 当前整层专家从 resident slot / ring lane 计算
  └─ lane 一释放，异步 refill 后续整层
```

Host Store 在加载时已是 `Layer → Role → Expert`，因此不再在运行时 GGUF reread、gather、
reorder 或 pack。Prefill 使用 whole-layer 连续搬运，不再对每个 token 做 expert LRU 决策。

### Decode

```text
router → 本轮 8 routed + 1 shared
       → resident/shared UGD 先形成计算 batch
       → 所有 miss 作为 UGD 一次批量晋升
       → miss batch 计算并累加
```

Decode 使用按物理 block size 的全局 slot pool 和 O(1) LRU。当前 router batch 由 epoch
保护，后续 Down/miss 不会淘汰本轮已需要的 Gate/Up/Down。

## 性能演化

### 早期基线到第一里程碑

| 操作 | Depth | `16fbee2` | `02e0bfb` | 变化 |
|---|---:|---:|---:|---:|
| Prefill 512 | 0 | 437.3 | 509.4 | +16.5% |
| Prefill 512 | 100K | 164.3 | 416.8 | +153.7% |
| Prefill 512 | 200K | 89.1 | 299.8 | +236.5% |
| Decode | 0 | 13.8 | 31.6 | +129.0% |
| Decode | 100K | 20.5 | 47.7 | +132.7% |
| Decode | 200K | 18.3 | 33.6 | +83.6% |

深 Prefill 的主因是 hd256 FlashAttention + register-O；Decode 的最大步来自 recorder 资源
复用和同步气泡收缩，不是单个 attention tile。

### Q8 KV

Q8_0 每 32 元素 34 bytes，相比 F16 的 64 bytes 省 46.9%。最初 200K Q8 Decode 比 F16
慢约 9～10%，profiling 证明瓶颈在 inline dequant 的 Attention，而不是 KV write 或 Pager。

block clustering、LS128/256、更高 wave parallelism、packed fp16 QK/PV 和 chunk 1024
依次落地后，200K 从约 31.55 到 39.65 tok/s，累计 +25.7%。最终大矩阵中 Q8 也因 KV
更小而允许保留更多 Expert cache。

### 最终矩阵代表点

| 工作负载 | 结果 |
|---|---:|
| Q8 Prefill 4096，d0，8 GiB 设置 | 2855.6 tok/s（统一 VRAM验收） |
| Q8 Prefill 4096，250K，8 GiB 设置 | 477.9 tok/s |
| Q8 Decode 1000，250K，8 GiB 设置 | 41.2 tok/s |
| Q8 Decode 1000，250K，10 GiB 设置 | 43.7 tok/s（历史大矩阵） |
| Q8 Decode 1000，250K，14 GiB decode-only | 48.1 tok/s |

注意 14 GiB decode-only 不代表同一服务可安全 Prefill；250K Q8 的 Prefill 安全上限在该
矩阵中是 10 GiB。完整表见 [Qwen 35B matrix](../reference/qwen36-matrix.md)。

## 哪些设计被模型否决

### Gate+Up 绑定和 Down 保留比例

直觉是“Down 可与 Up/Gate 计算重叠，所以少留 Down、多留完整 UG 对”。实际结果：

- 8:7 + paired eviction：37.65 vs 六池 41.50 tok/s；miss +5.72%。
- 只给 Down soft cap，权重 7：1000 token 40.55 vs plain 40.65，仍略负。
- 权重 5：500 token 已降到 39.10，并出现明显 Down churn。

原因不是“Down 完全没有 overlap”，而是固定配额会为冷 UG 淘汰热 Down；UG 与 D 分段还
增加 submit/hand-off。真实冷热比人工 role 配额更重要。

### 先算 Hit 专家

35B 上 shared/resident fusion 的 ABBA：

- off：85.7、86.5 tok/s；
- on：90.0、90.6 tok/s。

说明把 shared 作为第 9 个等价 expert 融入 resident batch 有稳定价值。但进一步把 U/G/D
拆成多段并没有价值：35B 2～9 expert 的完整 UGD 只有约 19～44 µs，而 1 个 RAM UGD
promotion 已是 136～164 µs，计算窗不足以藏住一次完整搬运。

## 当前瓶颈判断

- 短上下文/大 Prefill：host feed、per-layer window 与固定编排开销仍重要。
- 长上下文 Prefill：Attention 已占大头；只继续抠 memcpy 很难获得同等百分比收益。
- Decode：在 full-RAM 下主要看 Expert hit rate、RAM→VRAM 和每 token 固定提交气泡。
- 当 Expert cache 足够大，Attention depth 才再次成为主导。

## 相关文档

- [完整优化 campaign](../experiments/qwen36-campaign.md)
- [全局池与缓存策略](../experiments/cache-policy.md)
- [Prefill/Decode 执行链](../architecture/prefill-decode.md)
- [Attention/Q8 kernel](../kernels/attention-hd256-q8.md)
- [MoE 调度微基准](../reference/moe-schedule-microbench.md)

---

[模型索引](README.md) · [Qwen 122B](qwen35-122b.md) · [结果总表](../overview/results.md)
