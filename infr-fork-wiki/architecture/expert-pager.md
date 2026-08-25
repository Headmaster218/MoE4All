# Expert Pager 与全局 LRU

[首页](../README.md) / [架构](README.md) / Expert Pager

## 基本对象

Pager 管理的是一个“expert role matrix block”，而不是整个 expert 三件套。identity 包含：

```text
pool / layer / expert / role(Gate|Up|Down) / bytes
```

Gate、Up、Down 可以独立 resident/evict，这对不同量化、full-RAM Down overlap 和 bounded
tier batch 都更通用；但 router batch 会同时保护其当轮需要的所有 roles。

## 为什么按尺寸建池

不同 GGUF dtype/shape 的 block byte size 不同。固定 slot pool 避免 variable-sized Expert
在热路径中做通用堆分配；同一尺寸的任意 layer/role 可互换。

- 35B Balanced：Q5_K、Q6_K 两个逻辑池。
- Qwen 122B Quality：约 1.7/2.2/2.6 MB 三个实际 pool。
- DeepSeek V4 MXFP4：4.25 MiB role block。

“统一 VRAM”不等于强行用一个最大 slot 装所有尺寸，那会产生内部碎片；统一发生在 arena
与全局 range/accounting，Expert 热路径仍可有多个 size class。

## O(1) LRU

早期 promotion 会扫描链表。`8c49710` 改为 intrusive doubly-linked list：

- hit：从当前位置摘下并移到 MRU，O(1)；
- free slot：从 pool free list 直接取；
- victim：从 LRU tail 开始，跳过受保护/不可淘汰项；
- eviction：更新 LUT、owner、generation 和链表，O(1) 加少量保护检查。

Global Q5/Q6 profile 对相同 960,000 lookup：

| Layout | Lookup | Eviction | Victim scan |
|---|---:|---:|---:|
| 旧六池 | 370.8 ms | 73.9 ms | 1.0 step avg |
| Global Q5/Q6 | 284.7 ms | 62.4 ms | 1.0 step avg |

更大的 global list 没有成为 CPU 瓶颈。

## 为什么不用“层时间”或 token timestamp

如果第 1 层和第 40 层按 CPU wall time 更新，单 token 内第 1 层天然比第 40 层“老”，可能
造成层序偏置。曾讨论 token-based epoch，但最终不必引入复杂 aging 公式：

- intrusive LRU 反映真实访问顺序；
- 每个 router batch 用统一 epoch/protection 表示“本轮都热”；
- victim selection 跳过当前 batch，而不是把 40 层强行赋同一数值再排序；
- route trace 可以离线验证层命中范围，避免先上复杂策略。

122B trace 的 layer hit 从约 41.2% 到 88.5% 不等，主要是路由/尺寸/容量差异，不能只靠
抹平层时间解决。

## Batch epoch 保护

解析一轮 router 结果前打开 batch epoch：

1. lookup Gate/Up/Down；
2. 所有 touched/resolved blocks 标为本 epoch protected；
3. miss promotion 选择 victim 时不得淘汰本轮已触碰 block；
4. 等整轮 routed set 完成后关闭 epoch。

因此“需要 Down 时 Gate/Up 已经变热”不仅依赖 MRU 顺序，还有显式保护。后续 miss 无法
在同轮把前面已准备好的 Expert 淘汰。

## Cold touch 与 scan resistance

并非所有访问都等价。startup preload、prefetch 或一次性整层扫描若全部提升到 MRU，会
污染长期 Decode 热度。Pager 保留 cold-touch/epoch 语义，使预取项进入可用状态但不必
伪装成最近真实命中。具体策略通过 trace 验证，不按模型名硬编码。

## Phase 切换

Prefill 和 Decode 对同一物理 arena 使用不同布局。切换时清理并重建 residency metadata，
不是让 Decode 继承 Prefill 整层 ring 的 slot identity。这保证 Decode 可以使用完整 arena
做 expert-level LRU。

## 失败的 policy bias

8:7 Gate+Up/Down、paired Gate+Up eviction、Down weight 5/6/7 都没有保留。理由不是代码
实现困难，而是实际 miss/throughput 证明 global plain LRU 更稳。详见
[缓存策略实验](../experiments/cache-policy.md)。

---

[架构索引](README.md) · [统一 VRAM](unified-vram.md) ·
[RAM/SSD cache](ram-ssd-cache.md)
