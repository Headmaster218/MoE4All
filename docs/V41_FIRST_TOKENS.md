
---

## Round 190：找到同引擎、不同精度的对照条件 —— `dsv41-mixed-mxfp4-l0-19`

### 之前我说"本机没有高精度权重"，是错的（第 13 次未穷尽检查）

`--pack-info` 直接读出的结果：

```
40 layers x 384 experts, 6 used; dim 5120, ff 2304, vocab 129280
expert tensors by type:  IQ1_M 18   IQ2_XS 14   IQ2_XXS 28   MXFP4 60
dense types:             BF16 F32 F8_E4M3 F8_E8M0 Q5_K Q6_K Q8_0
engram layers: 1,14
files: 5, all present and long enough
```

**第 0-19 层专家是 MXFP4（18.8 MB/expert），其余是 IQ1_M / IQ2_XXS / IQ2_XS（7.7-10.2 MB）。**

**⇒ 这是一份**部分高精度**的权重，且 `strata-dsv41` 能直接读（`format gguf`）。**

### 为什么它有价值

**这是本会话第一次拿到「同一个引擎、不同专家精度」的对照条件：**

| 对照 | 专家精度 | 能回答什么 |
|---|---|---|
| `pack-q2k-all`（已测） | 全部 IQ2_XXS | ❌ `Germany is` → `Paris`（Round 189） |
| `dsv41-mixed-mxfp4-l0-19`（待测） | 前 20 层 MXFP4 | ？ |

**如果 MXFP4 前 20 层能让 strata 答对，就直接证明专家精度是根因。**
**如果仍不行，说明还有别的问题（后 20 层的低精度，或非专家路径）。**

**两种结果都能推进判断，所以这个实验值得做。**

### 已确认的旁证

`dense layers.1.engram.embed.{weight,scale}` 和 `layers.14` 是 **F8_E4M3 / F8_E8M0**，
而本会话用的独立 engram 表是 MXFP8 —— **两种表示不同**。这可能是 engram 行为差异的一个来源，
但需要单独验证。

