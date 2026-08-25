# 术语表

[首页](../README.md) / [Reference](README.md) / Glossary

| 术语 | 本 Wiki 中的含义 |
|---|---|
| Arena | 一段统一管理的逻辑显存范围，可由多个物理 Vulkan allocations 组成 |
| Shard | arena 的一个物理 allocation/backing fragment |
| Slot | 固定 size-class 中可放一个 Expert role matrix 的位置 |
| Pool | 相同物理 block size/dtype geometry 的 slot 集合；逻辑管理可跨 shard |
| Expert block | 一个 routed Expert 的一个 role matrix（Gate、Up 或 Down） |
| U/G/D | Up、Gate、Down；`UGD` 表示完整 Expert FFN |
| Shared expert | 每层必算、不由 router top-k 选择的 Expert |
| Routed expert | router 按 token 选择的稀疏 Experts |
| Resident | block 当前在 VRAM 中且 LUT 有效 |
| Promotion | 从 RAM/SSD 层将 block 变为 VRAM resident |
| Eviction | 释放 VRAM/RAM slot 的 residency；不一定复制 bytes |
| Shadow | VRAM block 在 RAM 中保留的 clean immutable 副本 |
| Inclusive cache | VRAM 内容允许/尽量同时存在于 RAM，层级有重复 |
| Exclusive cache | 一个 block 理想上只存在一层；容量高但 victim 需要迁移/重读 |
| Full-RAM Host Store | 完整 routed-expert payload 在 RAM 中一次性、layer-major 存储 |
| Bounded RAM | RAM 只缓存 expert payload 的一部分，其余 SSD-backed |
| ReBAR | CPU 可映射 device-local VRAM 的 PCIe BAR；此处用于 CPU direct push |
| Host DMA | ordinary RAM 经 `VK_EXT_external_memory_host` 原位 import 后由 Vulkan copy |
| BDA | Buffer Device Address；shader 使用 64-bit GPU address 访问 resident block |
| LRU | Least Recently Used；本 fork 使用 intrusive O(1) promotion |
| Batch epoch | 一轮 router 解析期间保护所有已触碰 required blocks 的生命周期标记 |
| Cold touch | 让预取/扫描 block 可用但不把它当作强 MRU 的访问语义 |
| Prefill | 一次处理多个新 prompt tokens 的阶段 |
| Decode | 通常每步生成一个 token 的阶段 |
| Synthetic depth | 不重放前缀，直接创建目标 KV 深度用于性能测试 |
| Pager profile | aggregate lookup/copy/submit/wait counters；不同于 ordered trace |
| Ordered trace | 每个 block access 的完整执行顺序，可用于 exact replay |
| Conditional RAM hit | 只在 GPU miss 样本中计算的 RAM hit rate |
| Combined hit | VRAM hit +（GPU miss 且 RAM hit）占总 access 的比例 |
| Layer ring | Prefill 的 whole-layer streaming slots；空槽异步补未来 layers |
| Loan | auxiliary allocation 临时占用 cold Expert window，后续按 generation 恢复 |
| Persistent state | KV、DeltaNet/KDA recurrent state 等跨 token 保持的 session 数据 |
| Runtime scratch | 只在一次 phase/request/graph execution 内存在的临时 tensor |
| Sinkhorn | DeepSeek V4 HyperConnection 中近似双随机矩阵归一化 |
| CSA/HCA/LID | DeepSeek V4 的 compressed attention/indexer cache 类型 |
| MXFP4 | DeepSeek V4 使用的 4-bit block floating weight/cache format |
| tgN / ppN | Decode N tokens / Prefill N tokens 的 benchmark shorthand |
| dK | synthetic/existing KV depth K |

---

[Reference](README.md) · [首页](../README.md)
