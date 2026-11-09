

---

## Round 114：**engram 第一次真正参与计算** —— A/B 对照证明

### 用 backtrace 终结猜测

在 `engram_apply` 里打 backtrace，得到 V4.1 的**真实执行路径**：

```
0: closure#48  engram_apply    runner.rs:5999
1: closure#73  层循环           runner.rs:6426
2: closure#290                runner.rs:13540   ← prepare_decode_plan
3: closure#291                runner.rs:13570
4: closure#293                runner.rs:13896   ← execute_decode
```

**`batch=1`，两层（layer=1、layer=14）各被调用一次。**

**V4.1 是一个 token 一个 token 地解码**，不走我之前改的批量 prefill 路径。
这和 Round 95 的发现完全一致——**round 95 的判断从头到尾都是对的**。

### 正确的接入点

每步注入 token id 的地方（`13742` 的 `be.upload(dec_ids_buf, ...)` 旁）：

```rust
if let Some(table) = weights.engram_table.as_ref() {
    let history: Vec<u32> = prompt[..pos.min(prompt.len())].to_vec();
    let batch_toks = [tok as u32];
    for (layer, dst) in engram_emb_buf.iter() {
        crate::engram::upload_layer(be, c, table, *layer, &history, &batch_toks, dst)?;
    }
}
```

**`pos` 就是 `tok` 的绝对位置，`prompt[..pos]` 正是前驱历史。**

**`engram_emb` 在这个作用域不可见（rustc: E0425）**，但 `engram_emb_buf` 可见，
**且 buffer 自带层号** ⇒ 遍历 buffer 即可，不需要 graph handle。

### A/B 对照（决定性证据）

| 配置 | `2+2=` 输出 |
|---|---|
| **无 engram** | `We need answer. The question is "2+2". So 2+` |
| **有 engram** | `**  **  **  **  **  **  **  ** ` |

**两者完全不同。engram 数据确实进入了计算路径。**

### 但答案还是不对

输出从"语法通顺但答错"变成了**重复的 markdown 符号**。
**engram 现在有影响，但影响的方向不对。**

最可能的原因：**upload 的 rows 布局与图期望的不一致**。
参考实现是 `reshape(key_length*n_cols, n_tokens)`——token-major、
**每个 token 内 24 个 bucket 连续**。而 `gather_rows` 返回 bucket-major，
`gather_for_layer` 按 token 顺序拼 bucket——**方向可能反了**。

### 一个意外收获

加载器自己确认了 engram 元数据完整：

```
GGUF: skipped 4 redundant engram lookup table(s) (engram_{token_map,primes,offsets,multipliers}.weight,
integer dtypes, duplicated verbatim by the deepseek41.engram.* metadata arrays)
```

### 下一步

**核对 rows 的内存布局**：`engram_emb` 声明是
`f32d(batch * n_cols * key_length)`，而 `engram_apply` 把它当 `x` 传给 `Op::Linear`。
需要确认 `row_indices` 生成的顺序与 `gather_rows` 的输出拼接，是否与参考的
`(key_length*n_cols) × n_tokens` 一致。

