# Kernel 与图算子索引

[首页](../README.md) / Kernels

| 页面 | 内容 |
|---|---|
| [hd256 Attention 与 Q8 KV](attention-hd256-q8.md) | Prefill FA、Decode Q8、深上下文调优 |
| [DeltaNet 与 KDA](deltanet-kda.md) | Qwen recurrent layer、Ling KDA、状态生命周期 |
| [量化 MoE](quantized-moe.md) | Q5/Q6/IQ4/MXFP4 直接计算、mixed batch 与 pager LUT |
| [DeepSeek V4 专用算子](deepseek-v4.md) | FP8 KV、压缩 cache、indexer、HyperConnection |

Kernel 页面只解释本 fork 新增/修改的路径和实验结论，不重复上游通用 GEMM/graph 手册。

---

[首页](../README.md) · [架构索引](../architecture/README.md)
