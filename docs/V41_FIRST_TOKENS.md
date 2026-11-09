

---

## Round 110：定位 upload 的可行位置 —— `kv` 与 buffer 不在同一作用域

### 已排除的位置

| 位置 | `be`/`cfg` | `engram_emb_buf` | `state`/`kv` | 结论 |
|---|---|---|---|---|
| `build` 闭包内 | ✅ | ❌ | ❌ | **不行** |
| `build` 调用点（如 `9516`） | ✅ | ❌ | ❌ | 不行 |
| 7 个 bind 点（如 `9562`/`9643`） | ✅ | ✅ | ❌ | **`kv` 不可见** |

**根因**：`engram_emb_buf` 属于 `SeamKv`，在 `build` 之后才建立；
而 bind 点处 `kv` 已被解构成 `kbufs`/`vbufs`/`engram_emb_buf` 等局部变量，
`state`/`kv` 本身不再可见。

### 但有一条出路

**token 历史不必从 `kv` 取。** 在 bind 点处这些是可见的：

```rust
for range in &ranges { ... }                    // 9449
let batch = row_start;                           // 9458
parallel.prompts[lane][range.clone()]            // 9481
```

**prefill 阶段 token 就是 `ranges` + `parallel.prompts`**，decode 阶段 token 在当前 step 变量里。

**所以 `upload_layer` 的 `history` 参数可以就地构造**，不需要 `cached_tokens()`。

### 下一步

在 **7 个 bind 点之前**（buffer 已可见处）插入：

```rust
for &(layer, dst) in &engram_emb_buf {
    engram::upload_layer(be, c, table, layer, history, batch_tokens, dst.as_ref())?;
}
```

其中 `history` / `batch_tokens` 由该点的 `ranges`/`prompts`/step 变量就地拼出。
**`table` 从 `weights.engram_table` 取**——需要确认 bind 点能否访问 `weights`。

**这是下一轮的第一件事：确认 `weights` 在 bind 点的可见性。**

