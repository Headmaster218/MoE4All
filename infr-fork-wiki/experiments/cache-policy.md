# Expert Cache Policy 实验

[首页](../README.md) / [实验](README.md) / Cache policy

## 设计问题

35B Balanced 每层激活 8 experts。直觉认为 Up+Gate 必须成对才有立即计算价值，而 Down
可以和 UG compute overlap，因此希望用两个 Down slot 换一对 UG，目标占比约：

```text
(Up + Gate) : Down = 16 : 7
等价按 pair 看为 8 : 7
```

同时物理上有 Q5/Q6 两种 block size，旧实现又分为 6 个 `(role,size)` 固定池。实验拆成
两个问题：

1. 是否该做全局 size pool；
2. 是否该在 global pool 内强制 role ratio/paired eviction。

## 旧六池

优点：slot 简单、role 不混；缺点：layer/role 固定映射到某个 pool，空闲容量不能跨 role
复用。它并不是真正的统一 cache storage。

## Q5/Q6 全局池

最终几何（7 GiB expert cache）：

| Logical pool | Global slots | Physical arenas |
|---|---:|---|
| Q5_K / 0.7 MiB | 5,637 | 3.00 GiB + 803.7 MiB |
| Q6_K / 0.9 MiB | 4,013 | 3.00 GiB + 220.7 MiB |

arena boundary 对 LRU/free-slot 不可见，shader LUT 存绝对地址。batch epoch 防止 Down miss
淘汰本轮 Gate/Up。

## 第一轮：8:7 + paired eviction

200K、Q8 KV、7 GiB、1000 Decode：

| Variant | Runs | Mean | Final misses | 对六池 miss |
|---|---:|---:|---:|---:|
| 旧六池 | 41.7, 41.3 | 41.50 | 283,067 | baseline |
| Global + 8:7 + paired UG | 37.6, 37.7 | 37.65 | 299,264 | +5.72% |
| Global + plain LRU | 39.2, 40.6 | 39.90 | 283,428 | +0.13% |

8:7 确认回退。Global plain LRU 的 miss 与旧六池几乎相同，固定端到端差约 0～1%，但获得
真正全局 capacity/reuse。

## 第二轮：只限制 Down

为排除“paired UG 本身害了结果”，保持三 role 独立，只给 Down soft occupancy cap。

500-token：

| Down:Gate:Up weight | Runs | Mean | Misses |
|---|---:|---:|---:|
| 8:8:8 | 40.9, 40.8 | 40.85 | 106,101 |
| 7:8:8 | 41.0, 41.3 | 41.15 | 106,382 |
| 6:8:8 | 40.9, 40.7 | 40.80 | 108,368 |
| 5:8:8 | 38.7, 39.5 | 39.10 | 114,750 |

权重 7 的短测看似 +0.3 tok/s，于是做 1000-token crossed confirmation：

| Weight | Runs | Mean | Misses |
|---|---:|---:|---:|
| 8 | 40.6, 40.7 | 40.65 | 283,428 |
| 7 | 40.6, 40.5 | 40.55 | 283,858 |

短测收益消失，最终略负；ratio 代码从 hot path 删除。

## 为什么直觉失效

### UG 成对并不等于应该强制留 UG

Gate/Up 的访问 identity 和热度相同，实际 global LRU 已会让同轮两者同时提升、受 epoch
保护。额外 paired eviction 会让一次 victim decision 淘汰两个 block，放大 churn。

### Down overlap 不是“Down miss 免费”

只有 promotion 完全落在足够长的 UG compute window 中才免费。35B routed UG compute
约 9～22 µs，一个 RAM D promotion 已 45～431 µs；再加第二次 submit/hand-off，无法全藏。

### 固定 role quota 看不到真实冷热

为了满足 8:7，它可能淘汰一个热门 Down，只保留更冷的 UG pair。下一次命中又付完整 Down
搬运。真实 route sequence 比静态角色价值更重要。

### Run drift 会制造短样本假收益

interleaved confirmation 中同一阶段 global/old 分别 40.8/40.9，较低阶段 38.4/38.8。
中间 global 37.8 的 miss counters 与其他 run bit-for-bit 一致，随后 old binary 也跌到 38.8，
说明系统/GPU state 漂移，而不是 LRU 随机。

## 最终决策

- 按物理 block size 建 global pool；
- role 独立 addressable/evictable；
- 当前 router batch 统一 epoch 保护；
- plain global LRU；
- 不保留 8:7/size weight/paired eviction 参数；
- 如果以后有更长 route trace，先离线模拟，再改在线热路径。

---

[实验索引](README.md) · [Expert Pager](../architecture/expert-pager.md) ·
[MoE 调度](../architecture/moe-scheduling.md)
