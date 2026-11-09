

---

## Round 120：找到额外开销 —— **MoE 在 prefill 里根本没批处理，逐 token 走 1 行 GEMV**

### 用户的判断是对的

我先前把 6 t/s 归因于「PCIe 链路上限」，并说 27.7 t/s 物理不可能。
**这个结论错了**，错在把「观测到的带宽」当成了「带宽上限」。
用户指出：Vulkan 图灵完备，达不到就是还有预期外开销。

### 决定性探针

**① 速度与工作量完全无关（bench 实测）**

| 配置 | 吞吐 |
|---|---|
| pp32 | 5.9 t/s |
| pp128 | 6.1 t/s |
| pp512 | 6.0 t/s |
| pp512, ub=32 | 6.0 t/s |
| pp512, ub=128 | 6.0 t/s |
| pp512, ub=4096 | 6.0 t/s |

**16 倍工作量、128 倍 ubatch 差异，吞吐恒定在 6.0。** 这不是带宽受限的特征。

**② 门控探针（`INFR_MMQ_PROBE`）**

```
57848  rows=1 thr=8 rows_ok=FALSE dtypes全OK(Iq1M/Iq1M/Iq1M)
37188  rows=1 thr=8 rows_ok=FALSE Iq2Xxs/Iq2Xxs/Iq1M
20660  rows=1 thr=8 rows_ok=FALSE Iq2Xs/Iq2Xs/Iq1M
```

**每一条都是 `rows=1`，`rows_ok=false`。**

**③ 形状探针（决定性）**

```
MMQ_SHAPE ne=5120 numel=5120 rows=1
```

**prefill 32 个 token，MoE 输入张量却只有 `numel=5120 = ne`，即 1 行。**

### 根因

`paged_mmq_selected` 的门控里有 **`rows > small_m_threshold(8)`**。
MoE 的 `rows = graph.desc(x).numel() / ne`，而 prefill 时 **MoE 被逐 token 调用了 32 次、每次 1 行**，
所以 `rows=1` 永远小于 8，**永远走不到 MMQ 快路径**，落在 `native_idm_iq1m_paged` 这个
**1-output-per-workgroup 的 GEMV** 上（commit `1224cf8` 注释实测 **0.26 TFLOPS**）。

**这解释全部现象：**

| 观测 | 解释 |
|---|---|
| pp32 = pp512 = 6.0 t/s | MoE 逐 token 处理，batch 维度完全没进入 |
| ubatch 32/128/4096 全无影响 | 改的是外层分块，MoE 内部照样逐行 |
| `native_idm_iq1m_paged` 31432 次 × 880µs = 27.7s (39%) | 正是这些 1 行 GEMV |
| `copy_strided` 453120 次 | 每行都要搬一次专家数据 |

### 排除掉的测量假象

`INFR_PROF_OPS` 默认把 **`zero_fill` 报成 22.4s (31.7%)**。
用 `INFR_STAMP_AFTER_BARRIER=1` 重测后它**完全消失**，时间转移到
`native_idm_iq1m_paged`（7.09s → 27.7s）。**zero_fill 的耗时里绝大部分是「等待之前的计算完成」**，
代码注释早已写明这是 ~160× 的误归因。

### 量级与下一步

| | GEMV（当前） | MMQ（应有） |
|---|---|---|
| 算术强度 | 1 行/次，0.26 TFLOPS | prefill 批量，int8 峰值 40+ TFLOPS |
| 行利用率 (pp512) | 1/512 | 满 |

**下一步：让 prefill 的 MoE 按 batch 行数一次发射，而不是逐 token 循环。**
这才是真正的额外开销。

