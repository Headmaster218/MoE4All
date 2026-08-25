# Prefill 与 Decode：两条不同执行链

[首页](../README.md) / [架构](README.md) / Prefill & Decode

## 为什么必须分开

Prefill 和 Decode 共享权重，却不是同一 workload：

| 维度 | Prefill | Decode |
|---|---|---|
| token rows | 数百～数千 | 通常 1 |
| 层访问 | 顺序完整遍历 | 顺序完整遍历，但 expert identity 每 token 变化 |
| Expert 计算 | 大 GEMM/MMQ，更易覆盖搬运 | 小 GEMV，固定 submit/同步占比高 |
| Attention | 随 depth 增长，长窗口占主导 | 每 token 扫历史 KV |
| 最佳 cache 粒度 | whole layer/ring | expert role block/LRU |
| 主要目标 | 吞吐与连续预取 | 延迟、hit rate、减少 bubble |

把同一个 LRU 策略强加给二者，会同时损害连续传输和跨 token 热度。

## Prefill 时间线

以 Qwen hybrid 的相邻 DeltaNet / Attention layer 为例：

```text
时间向下

CPU/Pager                             GPU
────────────────────────────────────────────────────────────
确认当前 layer 已在 ring lane         执行 Layer N DeltaNet 前处理
若有空 lane，立即开始补后续 layer  ─┐ DeltaNet/KDA recurrent compute
                                  │ routed MoE whole-layer compute
当前 layer 完成，lane 归还          │ residual/norm
                                  └→ Layer N+1 Attention QKV
继续填任意空 lane                  ─── FlashAttention / KV write
                                     routed MoE whole-layer compute
                                     residual/norm
```

ring 不是死板的 A/B“本层只补下一层”。只要某个 lane 因 layer 完成而空闲，refill worker
就继续补队列中的后续 layer；Attention 很长时可以连续补多个空槽，DeltaNet 很短时也不会
因为上一个 copy 未完就丢失新释放槽。

## 为什么 5-slot 合理但不是硬编码常数

Qwen 约 1 个 Attention 后跟 3 个 DeltaNet。直觉上 current + 后续 4 层共 5 slots 可覆盖
一个周期。实现受两层约束：

1. 根据模型 mixer pattern 得到理想 prefetch depth；
2. 根据实际 elastic VRAM 只建立放得下的 lane 数。

若模型全 Attention，就退化为“显存能放几层就几层”，不假设 1:3。异步 refill 又使槽数
不需要严格等于周期长度才能工作。

## Decode 时间线

```text
时间向下

CPU/Pager                             GPU
────────────────────────────────────────────────────────────
等待 router 结果可见                  Layer N mixer/router compute
lookup 8 routed × U/G/D + shared
建立 protected batch epoch

批量晋升所有 miss UGD  ───────────┐   shared + resident UGD compute
RAM hit: RAM→VRAM                  │   （两者并行）
RAM miss: SSD→RAM→VRAM             │
                                  └→  等 promotion fence/submit dependency
                                      miss UGD batch compute
                                      accumulate + residual

开始 Layer N+1 mixer/router
上一层 slots 离开 batch epoch
```

在 full-RAM 的旧细粒度路径中，曾让 Up/Gate 计算与 Down copy overlap；bounded RAM/SSD 则
倾向一次批量 UGD。最终微基准证明再根据 RAM/SSD 动态切 D 粒度的复杂分支收益不足，统一为
resident-first + all-miss UGD。

## Attention 与 DeltaNet 的不同状态

- Attention layer 更新 KV，并在 Decode 扫描已有 context；
- DeltaNet/KDA layer 更新 recurrent state，不使用同样的 KV 访问；
- empty-cache/session reset 必须同时重置 KV position 和 recurrent state；
- synthetic depth 真实分配/初始化 KV 并推进 allocator state，但不会重放有语义的历史。

## phase 切换

Prefill → Decode：

1. 停止 layer refill；
2. 等待在途 ring work；
3. 清理 whole-layer residency metadata；
4. 用同一物理 arena 重建 expert slot pools/LRU；
5. 尽可能保留不冲突的热 Expert bytes，但不强求旧 layout identity。

早期设计曾考虑每次切换全清 Expert cache；最终统一预算与 loan/restore 使“只清理必要空间、
优先冷项”成为原则，减少 Decode 重新预热。

## 性能边界

- 35B d0 Prefill GPU 曾只有约 38% 3D busy，host feed/submit 是主问题；
- 200K Prefill 的长 FA window 可达约 92% Memory Unit busy，Attention 已限制收益；
- Decode 早期选定窗口 GPU busy 约 19%，说明固定 bubble 比纯 PCIe 带宽更显著；
- 122B/DeepSeek 在大量 GPU miss 下，RAM→VRAM 再次成为最大可优化项。

因此“GPU 是否 100%”没有统一答案，必须按 phase、depth、cache hit 和 trace 分层。

---

[架构索引](README.md) · [MoE 调度](moe-scheduling.md) ·
[Attention](../kernels/attention-hd256-q8.md)
