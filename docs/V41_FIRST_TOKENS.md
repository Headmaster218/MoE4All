

---

## Round 72：问题比"batched 路径算错"更深 —— 两条路径都答不出来

### 用一个更简单的问题重测

`2+2=`，`--max-new 6 --temp 0`：

| 路径 | 输出 |
|---|---|
| per-token（默认） | `We need answer. The question` |
| batched | `We need answer. Need answer. We` |

**两条路径都没有回答 2+2。**

### 这改变了 Round 71 的结论

Round 71 我看到两条路径输出不同，判断"batched 算错、per-token 是对的"。

**这个判断不成立。** 两条路径都在输出同类的模板套话，**没有任何一条在真正回答问题**。

所以问题不在"哪条路径算错"，而在**更底层：模型本身没有正确工作**。

### 三个候选根因

1. **IQ1_M 专家的 dequant 仍然错** —— 从"全零"变成了"非零但错误"，
   而 `grid_mmq_parity` 只对照 host 侧的**同一套误解码**，所以测不出来
2. **34 个复用层的 source-stream 复用路径没接对**
3. V4.1 特有组件（engram / dspark）未正确接线

### 一个检验路径被排除了

尝试用 `dsv41-engram-iq4nl`（IQ4_NL，非 IQ1_M）做对照模型，**加载失败**：

```
infr-llama supports architecture=...|deepseek|..., got "deepseek41-engram"
```

**本机有两个不同的 V4.1 架构变体**：`deepseek41`（当前测的）和
`deepseek41-engram`（未支持）。所以没有现成的同架构对照模型。

### 这意味着什么

`grid_mmq_parity` 的 `iq1m_expert_gemm_matches_host` 通过，**只证明 GPU 与 host
实现一致**——如果 host 侧的 IQ1_M 解码本身是错的，两边一起错，测试照样通过。

**这是一个结构性盲点**：本项目没有任何一个测试对照了"真实模型的外部正确答案"。

### 下一步该做的

需要一个**独立的、外部可信**的正确性参照。可行选项：
- 用 llama.cpp（若支持 V4.1）跑同一模型，对比输出
- 或用同一模型的已知正确输出作为 golden
- 若都没有，只能逐层 dump 中间激活与参考实现比对

**在此之前，任何速度数字都没有意义。**

