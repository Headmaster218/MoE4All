# RAM / SSD Expert Cache

[首页](../README.md) / [架构](README.md) / RAM/SSD cache

## 两条明确路径

### Full-RAM

条件：RAM budget 足以装下完整 routed-expert payload。

```text
GGUF load → one complete layer-major Host Store → VRAM expert cache
```

- 运行期不需要 SSD demand read；
- Prefill 和 Decode 共享同一份物理 Host payload；
- Prefill 按整层连续读取；Decode 按 expert offset 读取；
- 35B payload 为 23.57 GB，典型使用此路径；
- full-RAM Down overlap 可保留，不应被 bounded-tier 的 all-role batch 覆盖。

### Bounded inclusive RAM/SSD

条件：完整 expert payload 大于 RAM budget。

```text
SSD-backed GGUF blocks
       ↓ RAM miss
bounded RAM slots
       ↓ GPU miss
VRAM expert slots
```

- startup 按每个 pool/layer 的相同比例均匀 preload，而非先填满前几层；
- VRAM resident block 若 RAM capacity 允许，保留 immutable RAM shadow；
- SSD miss 读入可复用 aligned RAM slot，再 promotion 到 VRAM；
- RAM slot 也由 LRU/ownership 管理，但不通过 GPU 回写补 victim。

## 为什么是 inclusive，而非严格 exclusive

如果 VRAM 12 GiB + RAM 40 GiB 完全 exclusive，理论上能覆盖 52 GiB distinct experts；
inclusive 则有约 12 GiB 重复。直觉上 exclusive 命中率更高，但它要求 VRAM victim 的
bytes 有去处。

可选动作：

1. VRAM→RAM 回写：实测约 44 MB/s，不可用；
2. 丢弃 victim，未来从 SSD 重读：会把每次 GPU cache churn 变成潜在 SSD traffic；
3. 为所有 VRAM block 保留 clean RAM shadow：eviction 只更新 metadata。

DeepSeek A/B 也显示 drop shadow 从 4.0 降到 3.9 tok/s，conditional RAM hit 61.6% 降到
60.6%。因此当前选择 inclusive。

## “无回写”不等于随机填充

RAM 中有 GPU victim shadow 时，只需让 VRAM slot 指向新 Expert；原 victim 的 RAM copy
保持不变。若 RAM slot 要被 SSD 新 block 覆盖，选择 RAM LRU victim，并确保：

- 被覆盖 block 若仍在 VRAM，VRAM copy 继续有效，但以后 eviction 后需 SSD 恢复；
- 更保守的策略优先淘汰不在 VRAM、最冷的 RAM block；
- 不把随机 SSD Expert 填入空位，预取必须由 route/policy 驱动。

当前实现的重点是 bounded inclusive cache 与成批 promotion，不是复杂 SLRU/ARC。

## SSD 并发

Windows positioned read 需要独立 file handles 才能真的并发。`082e1eb` 为 independent
pieces 建 handle，`f3af338` 在 request 级别批量 host promotions。

微基准：

| 122B pool | 1 expert SSD UGD | 8 experts SSD UGD |
|---|---:|---:|
| IQ4_XS 1.7 MB | 1.689 ms | 9.514 ms |
| Q5_K 2.2 MB | 2.753 ms | 14.236 ms |
| Q6_K 2.6 MB | 2.553 ms | 15.240 ms |

拟合带宽约 3.69～4.29 GiB/s，另有 0.31～1.08 ms 固定成本。把多个 miss 合成 request
很重要；单独增加 handle fanout 而没有上层 batching 曾无明显收益。

## RAM 命中与 SSD 命中的数据流

### GPU miss / RAM hit

```text
lookup → 选 GPU victim → RAM bytes → VRAM slot → LUT update → compute
```

不需要移动 RAM slot，也不需要将 GPU victim 回写。

### GPU miss / RAM miss

```text
lookup → 选 RAM victim + GPU victim
       → SSD read 到 RAM slot
       → RAM slot promotion 到 GPU slot
       → 两级 metadata update
       → compute
```

SSD→RAM 与 RAM→VRAM 不是同一段物理搬运；但端到端 Pager 统计中，SSD miss 的 block 最终
同样要经过 RAM→VRAM，因此 Host→VRAM traffic 包含所有 GPU miss，通常比 SSD traffic 大。

## 自动 full-RAM 判定

如果 routed payload 能被 RAM budget 完整覆盖，planner 直接选择 full-RAM，不建立一个
“容量刚好等于全部 blocks 的 bounded cache”再保留 SSD 逻辑。这样：

- 热路径不查 SSD tier；
- Prefill 可使用 layer-major Host Store；
- Down overlap 使用 full-RAM 专用路径；
- 日志明确显示 `host=full-RAM`。

## 当前缺口

- 大模型 Prefill 还没有完整的 SSD→RAM 多层 lookahead；
- bounded RAM 的预取策略主要是均匀 preload + demand，尚未基于语义预测未来 router；
- 双向 PCIe/后台 write-back 没有成为生产策略，因为 D2H copy 与 RAM ownership 仍需更完整
  pipeline；
- 更复杂 admission/SLRU 必须先由长 trace 模拟证明，而不是直接加入热路径。

---

[架构索引](README.md) · [Host DMA](host-dma.md) ·
[Trace/模拟](../experiments/trace-simulation.md)
