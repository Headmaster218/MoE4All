# 量化 MoE Expert Decode

[首页](../README.md) / [Kernels](README.md) / Quantized MoE

## 权重不是先完整解压再计算

Q4/Q5/Q6/IQ/MXFP4 expert weights 在 GPU 中保持量化 block format。shader 在 tile 内加载
scale/code，边解码边进行 dot/GEMV/MMQ，不会先把整个 matrix 膨胀为 F16 再计算。

因此：

- 更小量化减少 VRAM slot bytes 和 PCIe traffic；
- kernel 效率取决于 block layout、解码指令、复用和 occupancy；
- “bit 更少”不保证 compute 更快；122B IQ4_XS 就比 Q5/Q6 kernel 慢。

## Paged address

每个 resident block 的 LUT 给出最终 64-bit BDA/SSBO address。multi-expert kernel 接收
descriptor 列表：

```text
{ address, dtype, rows/cols, stride, routing weight, output slot }
```

物理 arena shard 和 layer/role 不出现在数学 kernel 的 placement 假设中。这样同一 batch
可跨 shard、跨 dtype 计算。

## Partial tile correctness

`276d9c8` 为 IQ4_NL ragged expert tile 增加 mask，防止 matrix 维度不是 tile 整数倍时读取
或累加 padding 外数据。它属于正确性/未来模型兼容，不宣称全模型性能提升。

## Subgroup Expert Decode

`0ffdefd` 为 paged quant formats 增加 subgroup path。100K A/B：APEX/Q4_K_XL 基本持平，
IQ4_NL_XL 约 +1%。保留是因为 supported format kernel 更合理且有 fallback，但不把它列为
主要 headline。

## Shared + Routed mixed quant

35B Balanced 有 Q5/Q6 routed expert；122B 还出现 IQ4_XS，shared 可能是 Q8。融合不是统一
转成一个 dtype，而是 batch descriptor 按 item 选择 decoder：

- routing 和 output accumulate 统一；
- weight load/dequant 分支按 dtype specialization；
- 相同 dtype/shape 尽量聚类，避免 divergence；
- shared 作为 always-on item 与 resident routed batch 一起提交。

端到端 shared fusion 对 35B 约 +5%，对 122B 约 +12%。

## MXFP4 complete-block Decode

早期同一个 MXFP4 block 的 scale/address work 在 tile 内重复。`495fc9a` 改成每 GEMV tile
完整解码一次 block 并共享中间值，DeepSeek final tg128 3.9 → 4.3 tok/s。

这项优化只作用 MXFP4；Q5/Q6/IQ/F16 不进入该代码。

## 122B kernel 时间揭示的问题

完整 UGD（shared included）：

| Pool | 2 experts | 9 experts |
|---|---:|---:|
| IQ4_XS | 34.5 µs | 120.0 µs |
| Q5_K | 26.2 µs | 68.9 µs |
| Q6_K | 29.5 µs | 87.9 µs |

IQ4_XS bytes 更小，却因 decoder/packing/指令效率更低而最慢。未来优化优先级应是：

1. profile IQ4_XS tile 内解码与寄存器压力；
2. 检查 wave occupancy、dp4a/packed op 映射；
3. 保持 parity 后再改变 block-sharing；
4. 端到端必须同时看 RAM→VRAM 是否已主导。

即使 IQ4_XS kernel 快一倍，若 token time 大头是 promotion，整体收益也不会一倍。

## Prefill/Decode kernel 选择

- Decode rows=1，使用 GEMV/subgroup path，固定 launch 和 dequant 占比高；
- Prefill rows 大，使用 MMQ/GEMM，weight block 可在更多 rows 间复用；
- 同一 dtype 的最佳 tile 不必相同；
- Pager 只负责提供地址，不应该让 Prefill/Decode 被迫共用同一个计算 geometry。

---

[Kernels](README.md) · [MoE 调度](../architecture/moe-scheduling.md) ·
[微基准](../reference/moe-schedule-microbench.md)
