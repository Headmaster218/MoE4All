

---

## Round 106：发现设计缺陷 —— 两个 engram 层共用一个 input 张量

### 证据链

**① `row_indices` 按 `slot` 切片**（`engram.rs:211-213`）：

```rust
let base = self.slot * self.n_gram;
let mult = &self.multipliers[base..base + self.n_gram];
...
let bucket_base = self.slot * self.n_bucket();
```

`self.slot` 是**层在 engram 列表中的下标**（层 1 → slot 0，层 14 → slot 1）。
`multipliers` / `primes` / `offsets` 全部按 `slot` 切片。

**⇒ 层 1 和层 14 对同一个 token 算出的 rows 完全不同。**

**② 我们只分配了一个张量**（`runner.rs:4879`）：

```rust
let span_has_engram = c.deepseek41 && (l_first..l_end).any(|l| c.has_engram(l));
let engram_emb = span_has_engram.then(|| g.input(f32d(...)));   // 单个
```

**③ `engram_apply` 两层都用同一个 `emb`**（`:5966`）：

```rust
let emb = engram_emb.expect("an engram layer needs the gathered engram input");
```

**④ 参考实现是每层一个**（`deepseek41.cpp:840`）：

```cpp
if (pmodel.engram_index(il) >= 0) {
    inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
}
```

`build_inp_engram(model, il)` 带层号，**每层生成自己的 `rows` 张量**。

### 结论

**当前设计下，层 14 会读到层 1 的 rows。** 即使把 gather 接上、上传正确的值，第二层仍是错的。

**这也是为什么必须先把 gather 接上才能暴露这个问题**——在此之前 buffer 是空的，两层读到的都是垃圾，
错得一样，看不出差异。

### 需要的改动

| 项 | 现状 | 目标 |
|---|---|---|
| `DecodeHandles::engram_emb` | `Option<TensorId>` | `Vec<(u32, TensorId)>` 按层 |
| `PendingSeamSlot::engram_emb_buf` | `Option<Box<dyn Buffer>>` | `Vec<Box<dyn Buffer>>` 按层 |
| `SeamKv::engram_emb_buf` | 同上 | 同上 |
| `bind_layer_io` | 绑一个 | 按 `h` 所属层绑对应的 |
| `engram_apply` | 用共享 `emb` | 用本层的 `emb` |

**影响 20 处引用、2 个分配点、跨 `runner.rs`/`weights.rs`。**

### 这一轮为什么没动手

改动跨文件且耦合紧（bind 链路刚在 Round 100 才理清）。
**半途而废的重构风险高于收益**——尤其它不能部分生效：只改声明不改 bind 会 unbound，
只改 bind 不改分配会越界。

**先把缺陷记下来，下次有完整时间块一次性改完。**

