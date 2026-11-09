

---

## Round 141：把单行守卫变成 release 常驻，而不是硬修

### 为什么不是「把 rows: 1 改成 batch」

读了整段才发现，**不止 KV 写入是单行的**：

```rust
if rd > 0 {
    rope_tail(&mut g, d4_kv, d4_rkv, 1, hd4, rd, theta4, ff4, false);
    //                                  ^ batch 位置传的是常量 1
}
g.push(Op::Dsv4CacheWrite {
    rows: 1,                                     // 只写一行
    row: (start_pos % layout4.raw_rows) as u32,  // 单行索引
```

**RoPE 也只旋转第 1 行。** 所以 batch > 1 时，其余行会带着未旋转的 K、
且 KV cache 里根本没有它们的任何东西，一起进 attention。

只把 `rows` 改成 `batch` 会让「KV 有了但 RoPE 没做」，**错误会变得更隐蔽**。

### 实际改动

把 `debug_assert_eq!` 换成**无条件 `assert_eq!`**，并写清原因：

```
DeepSeek V4's attention block is single-row: KV write and rope both handle only row 0,
so batch > 1 would silently write one KV row and leave the rest unrotated
(this guard must stay in release builds)
```

### 验证

| 路径 | 结果 |
|---|---|
| 默认（逐 token） | 0 次 panic ✅ |
| `INFR_DSV41_BATCHED_PREFILL=1` | panic，信息准确 ✅ |

**这条路径以前静默产生垃圾 KV，现在会立刻失败。** 这是纯收益：
任何未来误把 batch > 1 送进来的人，会当场看到原因，而不是拿到一个能跑但答错的模型。

### 对「4.9 倍提速」的影响

**要在 V4 上真正启用批量 prefill，需要把整段 attention block 改成逐行的**
（KV 写、RoPE、以及这一块里其他隐含 batch==1 的地方），不是一行补丁。
在那之前，`INFR_DSV41_BATCHED_PREFILL` 必须保持 opt-in。

### 缺陷 B 仍未定位

默认路径 `batch == 1`、守卫通过、**但依然答错**。
所以 V4.1 的答错问题是独立的，不在这条批量路径上。

