

---

## Round 83：engram 缺的是两层，不是一层

### 上一轮我以为只需要"接上 Op"

Round 82 的计划是：把 `wkv`/`k`/`q` 三个投影接进 graph。当时假设"输入 embedding 已经在某处备好了"。

**这个假设是错的。**

### `crate::engram` 在推理路径里零调用

```
$ grep -rn "engram_map_for_layer|dequant_mxfp8_row|engram::" --include=*.rs crates/infr-llama/src/     | grep -v "^crates/infr-llama/src/engram.rs"
（空）
```

`engram.rs` 里那几个函数——`row_indices`（完整实现了 n-gram hash：token_map 映射、
multipliers 加权、prime 取模）、`dequant_mxfp8_row`（E4M3 码 + E8M0 指数反量化）、
`engram_map_for_layer`——**一个都没被 `runner.rs` 调用过**。

（`ple.rs` 里有 `ple_row_indices_into`，但那是 qwen4exp 的 PLE，与 engram 无关。）

### 所以 engram 缺两层

| 层 | 状态 |
|---|---|
| **host 侧 gather**（按 token 算 n-gram 行号，从 189 GB 表里读并反量化） | **写好了，从未被调用** |
| **graph 侧计算**（三个投影 + QwenPleGate + 残差加回） | **权重上传了，字段未读** |

Round 80 我用 `rustc` 的 `never read` 定位了第二层。**第一层没有编译器能提醒**——
`engram.rs` 是个有测试的模块，测试通过、函数是 `pub`，看起来完全正常。

### 这解释了为什么它一直没被发现

- 配置校验通过（`config.rs:1320-1361` 有完整校验，说明作者知道格式）
- 权重加载成功（`runner.rs:2766`）
- 模块单测全绿（`engram.rs` 自己的测试）
- **唯独主流程没调用它**

每一处局部检查都通过，缺的是**它们之间的连线**。

### 修复的实际范围

现在清楚了，比 Round 81 估计的大，比 Round 81 开头估计的小：

1. **host 侧**：在 forward 之前，对层 1/14 的每个 token 调 `row_indices` +
   从 mmap 读行 + `dequant_mxfp8_row`，得到 `[n_embd, hc*nt]`（每 token 每副本一行）
2. **上传**：把这块作为 graph Input 交给 backend（需要一个 input buffer）
3. **graph 侧**：三个投影 + `QwenPleGate` + 残差累加（Round 81 已确认 Op 齐备）
4. **接残差**：`hcr[0]` 在 6300 之后、`hc_mix` 之前

### 第 1 步是真正的工作量

189 GB 的表，每 token 每层要读 `hc_mult` 行（4 行 × n_embd × key_len）。
**这是随机读，不走 DMA 流水。** 而且 `engram-mxfp8.gguf` 是独立文件，
需要 mmap 或预读。

**这一步的性能会直接吃掉不少 decode 速度**——但正确性优先。

### 下一轮

先看 `engram.rs` 里 `EngramMap` 的接口能否直接产出 `[n_embd, hc*nt]` 的矩形，
以及 `dequant_mxfp8_row` 的成本。**如果 gather 本身开销过大，可能需要先确认
engram 对 `2+2=` 这类短 prompt 是否真的必需**（也许短 prompt 下 gate 值极小）。

