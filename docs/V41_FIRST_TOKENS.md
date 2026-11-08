

---

## Round 71：决定性发现 —— batched 路径的输出是错的

### 对照实验

同一个 prompt（`The capital of France is`）、同样 `--max-new 8 --temp 0`：

| 路径 | 输出 | 复现 |
|---|---|---|
| `INFR_DSV41_BATCHED_PREFILL=1` | `We need answer. Need answer. We` | **两次一致** |
| 默认 per-token | `. Wait, the answer is 100` | **两次一致** |

**两条路径生成完全不同的文本。**

### 这意味着什么

**batched 路径不只是"跨 chunk 会挂"——它在单 chunk 内的数值本身就不对。**

`hc_mult` 残差问题比注释描述的更严重：
不是"chunk 边界传不过去"，而是**这条路径从来就没有算对过**。

### 这直接推翻了之前对 prefill 成果的解读

| 之前的说法 | 现在的事实 |
|---|---|
| 「prefill 27.5 t/s，2.3 倍」 | **数值错误，这个速度没有意义** |
| 「4K/8K 在 batched 路径下有效」 | **不可作为正确性依据** |
| 「16K 只是跨 chunk 传不过」 | **单 chunk 就已经错了** |

**27.5 t/s 是一个算错东西的速度。**

### 那么 IQ1_M 那个修复还有效吗

**有效，但需要重新界定范围。**

那个 LUT 绑定缺陷是**真实的**（全零 → 非零），它让模型从"完全跑不出 token"变成"能跑出 token"。
但：
- 在 batched 路径上，正确性被 `hc_mult` 问题掩盖
- 在 per-token 路径上，那个修复的作用是真实的

**需要重新测：per-token 路径上的 prefill 速率，才是产品口径的、可信的数字。**
（Round 70 测到 per-token 4K 是 <1.37 tok/s，那个数字现在反而更重要了。）

### 也解释了模型输出质量差

`We need answer. Need answer. We` 和 `. Wait, the answer is 100`
**都不是对"The capital of France is"的合理回答**。per-token 至少语法通顺。

两者都不是好答案，但 batched 明显更差——这与"batched 路径算错"一致。

### 结论

**本分支真正站得住的成果只有：**
1. `stray LUT binding` 缺陷修复（真实、可验证、有单测）
2. `ape: Option` + 复用层处理（让模型能加载并出 token）

**prefill 提速 2.3 倍这个结论必须撤回**——它是在一条算错数值的路径上测的。

