# Fork Commit Map

[首页](../README.md) / [Reference](README.md) / Commit map

范围：`upstream/main..311ed4c`，merge-base `d7f320e7`，共 89 commits。作者字段不作为
范围过滤；下表只记录 fork 增量。

## 2026-08-15：Windows 与测量基础（6）

| Commit | 内容 | 主题 |
|---|---|---|
| `d9bd5a9` | Windows native runtime compatibility | Windows |
| `1038b5d` | Windows host available-memory detection | Budget/Pager |
| `ebf5b79` | opt-in Pager profiling；submit cap fix | Measurement |
| `8c49710` | O(1) LRU promotion | Pager |
| `898ff91` | benchmark KV format reporting fix | Measurement |
| `16fbee2` | synthetic context depth | Benchmark |

## 2026-08-17：Qwen 深上下文基础（9）

| Commit | 内容 | 主题 |
|---|---|---|
| `dbc51fe` | hd256 FlashAttention Prefill | Attention |
| `9bef28d` | reclaim hd256 activation reserve | VRAM plan |
| `95b8ffa` | record hd256 tuning decision | Docs |
| `447cd50` | strided DeltaNet Decode | DeltaNet |
| `276d9c8` | IQ4_NL partial expert tile mask | Correctness |
| `0ffdefd` | subgroup expert Decode for paged quant | MoE kernel |
| `3c9523a` | Q8 KV Decode bottleneck profile | Measurement |
| `46c0b88` | specialized hd256 Q8 Decode | Attention |
| `a73d43a` | register-O FlashAttention Prefill | Attention |

## 2026-08-18：Q8 kernel、recorder 与 Pager pipeline（12）

| Commit | 内容 | 主题 |
|---|---|---|
| `ff69e83` | recycle transient recorder resources | Decode bubble |
| `5a33e58` | prefer generic hd256 Decode on RDNA3 | Kernel selection |
| `02e0bfb` | split hd256 Prefill output across 4 lanes | Attention |
| `d72f60f` | cluster Q8 QK by quant block | Q8 Decode |
| `e6b6137` | widen Q8 hd256 workgroups | Q8 Decode |
| `633638b` | expose more Q8 Decode wave parallelism | Q8 Decode |
| `84fb844` | packed fp16 Q8 QK | Q8 Decode |
| `0b37574` | packed Q8 value dequant | Q8 Decode |
| `15eb5f7` | Q8 hd256 chunk 1024 | Q8 Decode |
| `b8cbc52` | configurable multi-slot Pager upload pipeline | Pager |
| `6afab3a` | amortize cold-batch victim scans | Pager |
| `3651e29` | batch paged expert staging copies | Pager/transfer |

## 2026-08-19：Host Store、Down overlap、cache policy（9）

| Commit | 内容 | 主题 |
|---|---|---|
| `ce97e4f` | layer-major MoE Host Store | Prefill |
| `354c0c3` | CPU push Host Store into ReBAR pools | Transfer |
| `e603efe` | Qwen optimization history | Docs |
| `dd43c45` | wider deep Q8 Prefill splits | Attention |
| `2d57e7b` | overlap paged MoE Down uploads | MoE scheduling |
| `132b824` | halve deep hd256 Q8 attention rereads | Attention |
| `5a1faeb` | skip needless Down submits | MoE scheduling |
| `301a620` | reuse K tiles across deep Q8 QK rows | Attention |
| `951ffa3` | size-aware balanced cache experiment | Cache policy |

## 2026-08-20：全局池、Prefill ring、总预算、GUI（7）

| Commit | 内容 | 主题 |
|---|---|---|
| `d7be656` | unify Q5/Q6 logical MoE pools | Global pool |
| `637f23f` | asynchronously refill Prefill layer ring | Prefill |
| `c90577d` | final performance matrix/report | Docs |
| `c517290` | remove role-sharded six pools | Cleanup |
| `0a1aa30` | unified total VRAM budget；preserve Decode cache | Budget |
| `3722b8a` | GUI supervision primitives | Product |
| `742c500` | server-hosted browser control plane | Product |

## 2026-08-21：Embedding service 与 memory fix 起点（5）

| Commit | 内容 | 主题 |
|---|---|---|
| `949d6b2` | managed Embedding serving | Embedding |
| `9a53907` | GUI start path fix | Product |
| `bb2cb95` | share persistent-state sizing | Budget |
| `5ec490b` | align state allocation and accounting | Budget |
| `5b0c4de` | centralize total-VRAM accounting | Budget |

## 2026-08-22：Native Embedding 与统一 VRAM（15）

| Commit | 内容 | 主题 |
|---|---|---|
| `b88c3cd` | right-size phase workspace reserve | Budget |
| `4da3e16` | normalize GUI logs and live throughput | Product fix |
| `3c44a9d` | llama.cpp Embedding parity harness | Validation |
| `e7fec2f` | Embedding engine boundary | Architecture |
| `deb24c3` | native BERT WordPiece tokenizer | Embedding |
| `b16222d` | attention over graph-local KV | Correctness |
| `a90d285` | F16 linear weight offsets | Kernel/Embedding |
| `fd7e3d6` | Nomic BERT on native backends | Embedding |
| `4bb8b3b` | native `/v1/embeddings` | Product |
| `1ef73d6` | loan Expert cache slots | Unified VRAM |
| `5aeb58b` | unified VRAM range allocator | Unified VRAM |
| `02250a3` | shared VRAM physical shards | Unified VRAM |
| `c944e04` | loan unified VRAM to auxiliary engines | Unified VRAM |
| `bc4fc93` | native Embedding shares LLM arena | Unified VRAM |
| `35da757` | finalize unified VRAM sharing | Fix/acceptance |

## 2026-08-23：三级缓存、Ling、DeepSeek V4（14）

| Commit | 内容 | 主题 |
|---|---|---|
| `ebcbd3b` | exclusive RAM/SSD Expert tier | Tiering |
| `d3f1af5` | Ling 3.0 Flash on Vulkan | Model |
| `2f65486` | evenly preload bounded MoE RAM | Tiering |
| `7d8b0aa` | retain inclusive RAM shadows | Tiering |
| `9f4b6fe` | exclude MLA from attention scratch reserve | Budget fix |
| `b007730` | DeepSeek V4 FP8 KV + MXFP4 indexer | Model |
| `4343c03` | keep auto Expert cache within VRAM | V4 fix |
| `99e6e40` | parallel Decode HyperConnection Sinkhorn | V4 kernel |
| `082e1eb` | parallel Windows SSD piece reads | I/O |
| `f3af338` | batch concurrent Host promotions | Pager/I/O |
| `074e388` | batch Gate+Up promotions | MoE scheduling |
| `f6ca35c` | batch all Decode roles | MoE scheduling |
| `495fc9a` | MXFP4 block decode once per tile | V4 kernel |
| `f2fb30f` | vectorized Decode F32 GEMV | Kernel |

## 2026-08-24：收尾、弹性池、trace 与 122B scheduling（8）

| Commit | 内容 | 主题 |
|---|---|---|
| `7935cf8` | preserve full-RAM Down overlap | Compatibility |
| `66b66ab` | DeepSeek V4 campaign closeout | Docs |
| `31d9883` | explicit full-RAM Expert backing | Tiering |
| `df428ad` | fully elastic LLM/Embedding arena | Unified VRAM |
| `7eb9f0d` | ordered Decode access trace | Measurement |
| `1accc8c` | reset recurrent state after empty cache | Correctness |
| `3d38515` | overlap resident Experts with miss promotion | Scheduling |
| `0c86ce7` | fuse shared expert into paged MoE | Scheduling/kernel |

## 2026-08-25：Host DMA 与阶段文档（4）

| Commit | 内容 | 主题 |
|---|---|---|
| `2bd5469` | upload Host cache with Vulkan DMA | Transfer |
| `497460a` | distribute finite Host imports proportionally | Transfer |
| `81724ae` | first fork README | Docs |
| `311ed4c` | refresh large-MoE fork overview | Docs |

## 按成果寻找提交

- Qwen Attention：`dbc51fe`～`15eb5f7`、`dd43c45`、`132b824`、`301a620`。
- Pager/Host feed：`b8cbc52`～`354c0c3`、`d7be656`、`637f23f`。
- 统一 VRAM/Embedding：`1ef73d6`～`35da757`、`df428ad`。
- 三级缓存：`ebcbd3b`、`2f65486`、`7d8b0aa`、`31d9883`。
- DeepSeek V4：`b007730`、`4343c03`、`99e6e40`、`495fc9a`、`f2fb30f`。
- 122B scheduling/transfer：`7eb9f0d`、`3d38515`、`0c86ce7`、`2bd5469`、`497460a`。

---

[Reference](README.md) · [时间线](../overview/timeline.md) · [Evidence](evidence-index.md)
