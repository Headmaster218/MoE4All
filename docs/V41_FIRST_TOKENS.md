

---

## Round 65：带宽和算力都不是瓶颈，瓶颈在 kernel 自身的效率

### 两个上限都远没到

| 资源 | 本机能力 | decode 实际 | 利用率 |
|---|---|---|---|
| 带宽（CpuToGpu 实测） | **112.74 GB/s** | 6.6 GB/s | **5.9%** |
| 算力（保守估算峰值） | ~10 TFLOPS | **0.105 TFLOPS** | **1.05%** |

计算量核验：40 层 × top-6 × `3×5120×2304` MAC = **8.49 G MAC/token**，
6.2 tok/s 即 0.105 TFLOPS。

### 但 GPU 确实在忙

`main_busy` 折算到 decode 阶段约 107 ms/token（161 ms 的 66%）。

**所以不是"在等"，是 GPU 里在跑效率很低的东西。**

### 一个被排除的可能优化

`recorder.rs:463` 的注释提到：消掉 `grid_init()` 的 codebook 暂存后，
`native_idm_iq2s` 从 49.8ms → 23.3ms（约 50ms/505ms decode 的收益）。

**但这条对 IQ1_M 不适用**：`native_gemv_id_multi.comp` 里
`grid_init` 出现次数为 **0**，IQ1_M 的 GEMV 本来就不做 LDS 暂存。

另外 `Iq1M` 也不在 SG 名单里（名单是 `Q6K | Q5K | Iq3S`），走的是 NR=1。

### 剩下的瓶颈是什么

三个候选，都需要读 kernel 或重写：

1. **每 workgroup 只算 1 个输出**（`native_idm_*_paged` 的固有结构）
2. **网格解码本身的开销**（记忆里 2026-10-17 的结论：decode m=1 时网格解码占 78%）
3. 每节点的 dispatch / barrier 开销

**这不是调参能解决的，是 kernel 结构问题。**

### 诚实判断

- prefill：25 tok/s（512 时 27），**2.1 倍已拿到**
- decode：6.2 tok/s，**九个调参假设全否，且带宽/算力都不是瓶颈**
- 再往上需要**改 kernel 结构**（重写 GEMV 或换 MMQ 路径处理 m=1），
  风险和体量都超出"调优"的范围

