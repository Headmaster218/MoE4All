# DeepSeek V4 专用算子与状态机

[首页](../README.md) / [Kernels](README.md) / DeepSeek V4

## 接入边界

已实现：native graph、FP8 KV、compressed attention cache、MXFP4 indexer、MXFP4 paged
experts、HyperConnection/Sinkhorn。未实现 DSpark speculative module。

## FP8 KV 与 compressed caches

V4 每层由 `compress_ratio` 选择：

| Ratio | 模式 | Cache |
|---:|---|---|
| 0 | pure sliding window | raw SWA |
| 4 | CSA + lightning indexer | raw SWA + CSA + LID + 两组 compressor state |
| 128 | HCA | raw SWA + HCA + compressor state |

cache inventory 不只是一个 KV tensor：raw SWA、CSA、HCA、LID，以及各 compressor 的
`kv`/`score` state。FP8/quant dtype、padding 和 state F32 需要分别计入 planner。

## Compressor state 的四个关键边界

### 1. Partial block 不可见

```text
n_visible = floor((pos + 1) / ratio)
```

只有 `(pos+1) % ratio == 0` 才 commit compressed row。Prefill 末尾不完整 block 继续靠 raw
SWA recall，不能提前 flush 半成品。

### 2. `n_kv == 0` 改变图

没有任何 completed block 时，不只是 mask 全 `-inf`，而是 compressed branch 不进入图，
执行 pure raw attention。短 Prefill 首先命中这个 shape case。

### 3. Padded `n_kv`

graph shape 将最大 visible rows pad 到 256；每 token mask 仍只放行自己的 `n_visible`。
“已 commit 但对该 token 不可见”和“从未写入的 padding”都必须被 mask。

### 4. CSA dummy write

非 block boundary 的 CSA Decode 为保持 graph shape，会写 cache 最后一行垃圾数据，但它
永远在 visible range 外。HCA 没有同样 fallback。照搬一个统一 commit rule 会静默出错。

## Indexer 纠正

- V4 indexer Rope 是 NORM，不是 V3.2 的 NEOX。
- head layout 是 `[nope | rope]`。
- indexer key 来自 compressor，没有 `indexer_attn_k/indexer_k_norm`。
- top-k 针对 compressed blocks，而不是 raw tokens。

这些错误都可能“能跑但输出错”，因此先写 CPU/reference parity，再做 Vulkan specialization。

## HyperConnection / Sinkhorn

residual stream 扩为 `hc` copies。每 sublayer 的 mixes 产生 pre、post 和 `hc×hc` comb；comb
经近似 doubly-stochastic Sinkhorn 归一化后混合 streams。

Decode rows=1、hc=4 的通用小 op 串行开销明显。`99e6e40` 将 gate arithmetic 与 Sinkhorn
组合到专用 kernel，短 cache-hot A-B-B-A 约 8.0 → 8.7 tok/s。

曾尝试：

- F16 temporary：7.3～7.5，F32 8.2～8.3；
- 分四个 dispatch：8.2 vs 8.3～8.4；

都说明在小矩阵上少 dispatch/少 round trip 比降低 temporary bytes 更重要。

## MXFP4 Indexer ops

新增 Vulkan ops/shaders：

- cache write；
- compress；
- gather；
- indexer score；
- indexer top-k。

对应 CPU 实现用于 correctness。与 Expert MXFP4 GEMV 不同，indexer state 有自己的 shape、
quant cache layout 和 top-k 语义，不能只复用 MoE decoder。

## 为什么 kernel 优化没有救端到端

F32 `16384x24` GEMV 从 79.5 降到 23.7 ms（688 dispatch total），但 endpoint 保持 4.3
tok/s。最终 128-token trace 每 token 仍需 1.053 GiB Host→VRAM 和 0.404 GiB SSD。

这正是 profiling 的价值：kernel 局部 3.35× 后，Amdahl 定律暴露出 storage/paging 已占
绝大多数 token wall time。继续做另一个 2× 小 kernel 不会达到 17 tok/s。

## 兼容隔离

V4-specific graph 才发出这些 ops。通用 F32 GEMV、MXFP4 decoder 和 bounded Pager 有 shape/
dtype guard；full-RAM Qwen/Ling 的 Down overlap 曾被 bounded all-role batch 误伤，后来用
host-backing mode 显式分流修复。

---

[Kernels](README.md) · [DeepSeek 模型结论](../models/deepseek-v4-flash.md) ·
[DeepSeek 数据](../reference/deepseek-v4-data.md)
