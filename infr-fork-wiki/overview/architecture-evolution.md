# 架构怎样逐步演化

[首页](../README.md) / 总览 / 架构演化

## 不是一开始就设计出了最终形态

最终架构来自几个连续被真实模型推翻的假设：

1. 单个模型可以只靠 VRAM expert cache；
2. 六个固定 role/size 池已经足够；
3. Prefill 和 Decode 可以使用同一套 expert 粒度 LRU；
4. Host RAM 总能放下全部专家；
5. Embedding 很小，单独 malloc 就行；
6. CPU push 到 ReBAR 已经接近 PCIe 上限。

每个假设在 35B、Ling、DeepSeek V4 或 122B 上遇到边界，系统才演化到下一层。

## 阶段 1：固定六池

最初按 `(Gate/Up/Down, Q5/Q6)` 分成六个池。优点是 slot 大小固定、实现直接；缺点是
每个 layer/role 只能在自己的池里竞争。某池有空位也无法救另一个池，逻辑容量被切碎。

为了判断“Down 是否可以更激进淘汰”，试过 8:7 保留、Gate+Up 成对淘汰、Down soft cap。
最终完整测试证明这些规则会无视真实冷热，增加 miss；最简单的 global LRU 更稳。

详见[缓存策略实验](../experiments/cache-policy.md)。

## 阶段 2：按尺寸的逻辑全局池

35B Balanced 只有 Q5_K/Q6_K 两类专家矩阵，因此变成两个逻辑池：

```text
Q5 logical pool ─┬─ physical arena 0 (<3 GiB)
                 └─ physical arena 1
Q6 logical pool ─┬─ physical arena 0 (<3 GiB)
                 └─ physical arena 1
```

物理显存仍因 Windows AMD 映射限制拆分，但 free slot、LRU、eviction 和 LUT 都是全局的。
任意 layer/role 能使用同尺寸池的任意 slot。Router 本轮访问用 batch epoch 保护，Down miss
不会淘汰本轮已触碰的 Gate/Up。

## 阶段 3：Prefill/Decode 分治

观察访问形态后发现：

- Prefill 会顺序执行全部层，专家选择虽稀疏，但大 batch 下把整层搬来最容易形成连续传输；
- Decode 每 token 每层只访问 router 选中的专家，需要保留跨 token 热度。

于是同一物理 cache 在 phase 切换时重建布局：

- Prefill：layer-major Host Store + resident whole layers + 异步 layer ring；
- Decode：expert-level global LRU。

这不是维护两份完整缓存；是对相同物理空间在不同阶段采用不同解释。

## 阶段 4：总 VRAM budget

原先用户给的是“Expert Cache 大小”，KV、fixed weights、runtime reserve 另外计算，容易出现：

- Prefill 峰值超预算；
- Decode 时大量 runtime reserve 空着，却不能变成专家槽；
- GUI 估算与真实分配不一致。

改造后用户指定总 VRAM budget。模型 metadata 和实际 tensor layout 先形成 memory plan，再把
固定区、persistent state、弹性区与安全 margin 对齐。KV 仍是 persistent，不参加逐请求淘汰；
activation/runtime 则进入弹性区。

## 阶段 5：统一弹性 VRAM

LLM Expert、LLM activation、Embedding weights/runtime 进入同一 logical arena：

```text
low address                                            high address
┌───────────────────────────┬────────────────────────────────────┐
│ Expert slots →            │ ← LLM / Embedding / Vision / Draft│
└───────────────────────────┴────────────────────────────────────┘
```

大块临时申请无法直接塞进离散 expert hole 时，Pager 淘汰一个连续的冷 expert window，形成
高地址连续范围；借用结束后按 generation 原位恢复。这样没有必要把现存 Expert 在 VRAM
中做昂贵 compaction。

Embedding 最初被当作 persistent allocation；最终改为请求时加载，执行和下载完成后释放，
防止一个偶尔调用的模型永久吃掉 Expert 命中率。

## 阶段 6：从二级到三级缓存

35B 的 23.57 GB expert payload 能进 RAM，于是形成 full-RAM Host Store。Ling、DeepSeek V4、
Qwen 122B 使“全进 RAM”失效，加入 bounded RAM/SSD：

```text
SSD GGUF / block I/O
        ↓ miss fill
bounded inclusive RAM cache
        ↓ promotion
elastic VRAM expert cache
```

RAM 采用 inclusive shadow：VRAM 中的 Expert 尽量保留 RAM 副本。虽然重复占用一部分 RAM，
却让 VRAM eviction 只改 metadata。实测 VRAM→RAM 只有约 44 MB/s，不能作为关键路径回写。

## 阶段 7：Host RAM 原地 DMA

原 CPU push 单线程约 8.8 GB/s，多线程/批处理约 14～19 GB/s；专用 Vulkan copy 微基准在
足够大 payload 下可达约 25 GB/s。最终用 `VK_EXT_external_memory_host` 把现有 RAM cache
原位导入，不再创建第二份 staging cache。

驱动只能导入约 29 GiB，因此不是“全有或全无”：按 expert pool 比例分配 import 覆盖，
未导入部分继续走旧 CPU push。122B tg256 从 shared-fusion 后的约 19.2 提到 23.2 tok/s。

## 最终不变量

- fixed dense weights 与 KV/persistent recurrent state 不参与逐请求动态淘汰；
- Expert、Embedding/Vision/Draft 权重和临时 runtime 可在弹性区竞争；
- 物理 arena 可分片，逻辑 slot 管理必须全局；
- 当前 router batch 所需 Expert 受 epoch/generation 保护；
- full-RAM 模式不走 SSD；bounded 模式不假装 RAM 中存在完整 Host Store；
- VRAM victim 不依赖慢速 GPU→RAM 回写；
- import/DMA 失败必须回退到正确的 CPU push，而不是加载失败。

---

[首页](../README.md) · [统一 VRAM](../architecture/unified-vram.md) ·
[三级缓存](../architecture/ram-ssd-cache.md) · [Host DMA](../architecture/host-dma.md)
