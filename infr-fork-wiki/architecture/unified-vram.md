# 统一弹性 VRAM

[首页](../README.md) / [架构](README.md) / Unified VRAM

## “统一”指什么

统一不要求一块物理上连续的 18 GiB Vulkan allocation。Windows AMD 驱动对单个 mapped
ReBAR allocation 有现实上限，因此 arena 可以由多个 shard 组成。统一的含义是：

- 一个全局逻辑地址/范围管理器；
- 所有 shard 的 free range、slot、ownership 和 accounting 统一；
- 任意 layer/role 可使用同尺寸 pool 的任意 slot；
- auxiliary allocation 可以跨原本的 Expert cache 边界借空间；
- GUI/日志报告一个总预算，而不是多份互相不知情的 allocator。

```text
logical arena
┌──────────────────────────────────────────────────────┐
│ shard 0 │ shard 1 │ shard 2 │ ... │ shard N         │
└──────────────────────────────────────────────────────┘
physical Vulkan allocations, independently mapped
```

shader LUT 保存 resident block 的最终 64-bit GPU address，因此执行不依赖“layer 固定属于
arena X”。

## 低地址 / 高地址双向生长

```text
低地址                                                        高地址
┌─────────────────────────────────────────────────────────────────┐
│ Expert slots →              free/loan window        ← variable │
└─────────────────────────────────────────────────────────────────┘
```

- Expert slot 尺寸离散、数量多、生命周期跨 token，从低地址增长。
- LLM scratch、Embedding/Vision/Draft 权重和 runtime 尺寸不规则，从高地址增长。

这让 variable allocation 获得自然连续区，同时不会要求 Embedding tensor“恰好拆成若干专家
slot”。

## 没有连续空间时如何处理

最初设想是先淘汰总量足够的冷 Expert，再把物理末尾 Expert 搬到前面做 compaction。最终
没有这样做，因为 VRAM→RAM 回读极慢，且移动 resident slot 会修改所有正在使用的地址。

实际算法：

1. 从高地址边界向低地址寻找能够形成目标长度的 contiguous Expert window；
2. 选择该窗口内代价最低/最冷且可淘汰 slots；
3. 等待 in-flight read lease，释放窗口并记录 generation/slot identity；
4. variable allocation 直接占用这段原地址；
5. 生命周期结束后释放；
6. Pager 在后续需要时从 Host/SSD 原位恢复 slots。

这是“淘汰以制造连续窗口”，不是“搬运现存 Expert 做碎片压缩”。

## Slot loan 与 generation

借用时不能只记“这里曾经有 Expert A”。在借用期间，同一 logical Expert 可能因别的访问
被装入新位置，旧 restore 不能覆盖新 generation。恢复条件至少包含：

- slot 当前 owner/generation 仍匹配；
- Expert 没有在其他 slot 更新为更晚 residency；
- range 已从 auxiliary owner 归还；
- 当前 batch 没有保护该 slot。

这样可以避免 Embedding 完成后用陈旧 metadata 恢复出重复或错误地址。

## 执行锁

统一物理 arena 意味着两个 execution graph 不能完全无协调地提交：

- LLM GPU work 引用 Expert slots 时持 read lease；
- auxiliary graph 可先记录不会改变地址的部分；
- 需要分配/驱逐时取 write lease，等待所有相关 LLM work 完成；
- allocation 结束后释放 write lease，才允许新 LLM submit 使用更新后的 LUT。

第一版在同线程持 read 后尝试升级 write，造成首个 Embedding 请求死锁。最终规则明确区分
recording 与 allocation/execution 生命周期。

## 物理分片限制

35B 双池阶段已经验证：Q5 logical pool 可由 `3.00 GiB + 803.7 MiB` 支撑，Q6 由
`3.00 GiB + 220.7 MiB` 支撑，arena boundary 对 LRU 不可见。后来统一 arena 在 20 GiB
总预算下使用多个约 2 GiB shard，仍保持全局 range management。

## 实际空间利用率

Qwen 35B + Nomic，20 GiB budget：

| 状态 | Expert | Auxiliary | Free tail |
|---|---:|---:|---:|
| 启动后 | 18,789,572,608 B | 0 | 720,896 B |
| Embedding weights resident | 减少约 351 slots | 273,530,880 B + runtime | 22,568,960 B |
| Embedding 释放且 Chat 恢复后 | 18,789,433,344 B | 340,224 B LLM runtime | 519,936 B |

最大碎片比例约 0.120%，最终 tail 0.00277%。这满足“不要为统一管理浪费大量 cache”的
验收目标。

## 不进入 arena 的对象

- fixed dense weights；
- persistent KV；
- recurrent state；
- Vulkan 驱动自身和不能安全重定位的长期对象。

把所有东西都叫“统一内存”会模糊生命周期。当前统一的是具有可替换/可借用语义的弹性区，
不是强制整个 GPU 只做一次 giant malloc。

---

[架构索引](README.md) · [Memory budget](memory-budget.md) ·
[Expert Pager](expert-pager.md) · [Nomic 验收](../models/nomic-embedding.md)
