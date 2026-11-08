

---

## Round 73：对照实验证明引擎没问题，问题只在 V4.1

### 用同一个引擎跑一个已知正常的模型

`qwen2.5-1.5b-q4_k_m`（1.1 GB），同一 prompt `2+2=`，同样 greedy：

**输出：`2+2=4`** ✅

### 这个对照排除了什么

| 环节 | 结论 |
|---|---|
| Vulkan 后端 | ✅ 正常 |
| 采样 / greedy 路径 | ✅ 正常 |
| chat template | ✅ 正常 |
| `infr run` CLI | ✅ 正常 |
| **V4.1 模型的推理** | ❌ **错误** |

**所以问题被完全限制在 V4.1 这条实现路径上**，不是引擎或工具链。

### 补充：V4.1 的专用开关只有三个

```
INFR_DSV41_BATCHED_PREFILL
INFR_DSV41_SHARD1
INFR_ENGRAM_TABLE
```

**没有关闭 engram / dspark 的总开关**，所以不能靠"关掉某个组件"来二分定位。

### 现在的诊断状态

问题精确定位到 V4.1 的 forward 实现，候选仍是 Round 72 那三个：

1. **IQ1_M 专家 dequant 错误**（`grid_mmq_parity` 只做 GPU-vs-host，发现不了）
2. **34 个复用层的 source-stream 路径**
3. **engram / dspark 接线**

### 最有价值的下一步

`qwen2.5-1.5b` 的正确性说明**引擎的通用算子是对的**。
V4.1 独有的东西是：IQ1_M experts、复用层、engram、dspark、hc_mult 残差。

**逐个关闭/替换这些独有组件，就能二分定位。** 需要代码改动来加开关，
但至少现在每次 A/B 都有 `2+2=4` 这个客观判据，不再是盲测。

