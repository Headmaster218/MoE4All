

---

## Round 140：**抓到批量 prefill 的 KV 只写第一行** —— 那条断言在 release 里被编译掉了

### 发现

`runner.rs:7147` 本来就有一条断言：

```rust
debug_assert_eq!(batch, 1, "DeepSeek V4 currently builds scalar forwards");
g.push(Op::Dsv4CacheWrite {
    src: d4_kv, cache: k_cache[l],
    rows: 1,                              // ← 硬编码 1
    row:  (start_pos % layout4.raw_rows) as u32,   // ← 单行
```

**`rows: 1` 与 `row:` 都是单行的。** 所以 `batch > 1` 时，只有第一行被写进 KV cache。

而这条守卫是 `debug_assert_eq!` —— **在 release 构建里被完全编译掉**，
而 release 是我们唯一能跑这个 130 GB 模型的构建。于是它一直在静默发生。

### 实测

把断言包进 `INFR_V4_BATCH_CHECK` 让它在 release 生效：

| 路径 | 结果 |
|---|---|
| 默认（逐 token） | 断言**不触发**，`batch == 1` ✅ |
| **`INFR_DSV41_BATCHED_PREFILL=1`** | **panic**：`assertion left == right failed: DeepSeek V4 currently builds scalar forwards` |

### 结论一：4.9 倍提速那条路径是**带数值错误**的

`INFR_DSV41_BATCHED_PREFILL` 让 `batch > 1` 进入一段只写一行的代码，
**KV cache 其余行是垃圾**。之前 Round 123 观察到「批量路径输出错误」现在有了机制解释。

**所以 Round 123 把这条路径默认打开是错的**（Round 125 已回退为 opt-in，现在知道了确切原因）。
**在修好 Dsv4CacheWrite 的批量写入之前，这条路径的 4.9 倍不能算可用收益。**

### 结论二：这**不是**默认路径答错的原因

默认路径实测 `batch == 1`，断言不触发，但它同样答错。
⇒ **存在两个独立缺陷：**

- **缺陷 A**：批量 prefill 的 `Dsv4CacheWrite` 只写第一行（本轮发现，机制清楚）
- **缺陷 B**：默认逐 token 路径也答错（更早就在，尚未定位）

### 下一步

缺陷 A 有明确的修法：让 `Dsv4CacheWrite` 按 `batch` 写多行（`rows: batch`，逐行 `row`）。
修完再看批量路径的输出是否变正确——如果变正确，还能反过来帮助孤立缺陷 B。

