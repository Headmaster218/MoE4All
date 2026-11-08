

---

## Round 82：engram 插入点的精确定位

### 层循环结构（`runner.rs`）

| 行 | 内容 |
|---|---|
| 6179-6190 | **循环前**：`hidden` 复制到 `hcr[0]` 的 `hc_mult` 份（残差流初始化） |
| **6216** | `for (l, lw) in lw.iter().enumerate().take(l_end).skip(l_first)` |
| 6217-6300 | 循环体开头：`if let Some(pw) = &lw.ple { ... }`（qwen4exp 专用） |
| 6301+ | `// Per-layer dims` |
| 6398 | `hc_wrap_pre(attn, hcr[0], hn)` —— attention 子层起点 |
| 8297 | `hc_wrap_pre(ffn, hcr[1], hn)` —— FFN 子层起点 |

### 参考实现的对应位置

```cpp
for (il = 0; il < n_layer; ++il) {
    if (engram_index(il) >= 0) {
        inpL = build_engram(model, inpL, build_inp_engram(model, il), il);   // ← 作用于 inpL
    }
    ...
    build_hc_mixes(inpL, hc_attn_fn, ...);        // 之后的 hc 混合读 inpL
```

**`inpL` 就是残差流，对应本项目的 `hcr[0]`。**
所以 engram 要在**每层循环体开头、`hc_mix`/`hc_wrap_pre` 之前**，作用于 `hcr[0]`。

### 插入方案

在 `6216` 循环体内、PLE 分支之后（`6300` 的 `}` 之后）加一段：

```rust
// DeepSeek V4.1 engram associative memory (deepseek41.cpp:382 build_engram). Injected at the
// TOP of the layer, before the hyper-connection mixes read the stream.
if let Some(ew) = &lw.engram {
    // 1) one projection: hc keys + one shared value  [hc*n_embd + n_embd, batch]
    g.push(Op::Linear { x: engram_emb, weight: ew.wkv, dst: engram_kv,
                        m: batch, in_f: ne, out_f: (hcw + ne), w_off: 0 });
    // 2) grouped RMSNorm per (token, hc copy) over n_embd, then the per-copy scale
    //    key slice  = engram_kv[0 .. hc*n_embd)      value slice = engram_kv[hc*n_embd ..)
    // 3) Op::QwenPleGate { key, query, value, dst: engram_out, rows: batch*hc, hc, n_embd }
    // 4) hcr[0] += engram_out
}
```

### 三个还需要解决的技术点

1. **`grouped_norm`**：参考是 `reshape[n_embd, hc, nt] → rms_norm → reshape → mul(w)`。
   本项目 `Op::RmsNorm` 的作用维度和能否表达"按 n_embd 分组"需要确认——
   `QwenHcNorm` 可能就是干这个的（qwen4exp 的 PLE 路径用了它）。
2. **`engram_emb`**：engram 的输入是 **host 侧 gather 出来的 embedding**（`crate::engram` 的
   n-gram hash 查找），不是 token embedding。需要确认它作为 graph Input 如何注入
   （`engram.rs` 里有 `generate` 侧的 gather，但要接到 forward 里）。
3. **`hcr[0] += engram_out`**：需要原地累加。现有 `Op` 里有没有 add-into-residual 的形式，
   还是要一个新 buffer + Copy。

### 这一轮没动手的原因

前两个技术点（特别是 `engram_emb` 这个 graph Input 的接法）没确认就写代码，
会写出编译不过或语义错误的东西。

**下一轮先确认这两点，再动手。** 这比写了一堆错代码再回滚划算。

