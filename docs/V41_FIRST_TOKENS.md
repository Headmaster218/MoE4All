

---

## Round 32：根因找到了 —— paged MMQ 白名单漏掉 IQ1_M

### 完整因果链

1. `native_idm_iq1m_paged` 标在 **`native_gemv_id_multi`** 上（`build.rs:2545`）——
   **它是一个 GEMV，不是 GEMM。** 每个 workgroup 只算 **1 个输出**，沿 K=5120 协作归约。

2. GEMV 的算术强度：`5120 MAC / 1000 字节 = 5.1 MAC/B`。
   对比 MMQ 的 tile 形态（64×64 输出算一次），强度高 64 倍。
   **GEMV 天生喂不满 dp4a，这是 0.26 TFLOPS 的来源。**

3. 仓库**已经有** `native_gemm_mmq_iq1_m.comp`（MMQ 版），而且
   `DType::Iq1M` **已经在 `MOE_MMQ_DTYPES` 里**：

   > `IQ1_M: added once native_gemm_mmq_iq1_m landed the per-8-element sign-keyed delta
   > correction (two dp4a accumulators per 32-block, four self-computed per-l activation sums).`

4. **但 paged 路径用的是另一个白名单 `MOE_MMQ_PAGED_DTYPES`，里面没有 `Iq1M`。**

```
MOE_MMQ_DTYPES        : ... Iq2Xs, Iq1M, Iq3S, Iq3Xxs, Mxfp4, Nvfp4, Q2_0
MOE_MMQ_PAGED_DTYPES  : ... Iq2Xs,        Iq3S, Iq3Xxs, Mxfp4, Nvfp4, Q2_0
                                    ^^^^ 缺 Iq1M
```

### 这就是那 38.8 秒

V4.1 的 gate/up 有 **16/40 层是 IQ1_M**（Round 11 普查）。
这些层的专家矩阵乘因为 paged 白名单不含 IQ1_M，**全部退回到 GEMV 路径**——
单 workgroup 单输出、算术强度 5.1 MAC/B、实测 0.26 TFLOPS。

### 这不是架构限制

`native_gemm_mmq_iq1_m.comp` 已经实现了 IQ1_M 的 MMQ（含 ±0.125 delta 的 sign-keyed 校正、
每 32-block 两个 dp4a 累加器）。**它只是没有被 paged 路径接受。**

### 状态

无代码改动。PR #183：64 commits。

### 下一步（明确且收益可估）

把 `DType::Iq1M` 加进 `MOE_MMQ_PAGED_DTYPES`，验证：
1. `synthetic_deepseek2` 仍绿
2. 实跑 V4.1，看 `native_idm_iq1m_paged` 是否消失、被 `native_gemm_mmq_iq1_m` 取代
3. 正确性：输出是否与 GEMV 路径一致（**这是关键，不能只看速度**）
4. 速度：38.8 秒能降到多少

**风险**：paged 白名单当初排除 IQ1_M 可能有原因（比如 paged 布局下 sact 校正在 64-bit
BDA 寻址上有问题）。**必须先查清楚为什么当初没加**，或者直接在小样本上对比两条路径的数值。

