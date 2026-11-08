

---

## Round 81（续）：修复量比预估小得多 —— `QwenPleGate` 就是 engram 的算子

### 上一节我算错了工作量

我数出"缺 6 个算子"（`sum_rows`/`sqrt`/`abs`/`clamp`/`sgn`/`repeat`），
据此判断"一天量级"。

**这 6 个全都已经在同一个现成 Op 里：`Op::QwenPleGate`。**

### 逐项对照

参考实现 `build_engram` 的 gate：

```cpp
s    = sum_rows(k * q) / sqrt(n_embd)
mag  = sqrt(clamp(abs(s), 1e-6, inf))
gate = sigmoid(sgn(s) * mag)
out  = x + repeat(value, hc) * gate
```

`Op::QwenPleGate` 的 CPU 实现（`infr-cpu/src/lib.rs:3268-3292`）：

```rust
let score  = dot / (ne as f32).sqrt();                       // s
let signed = if score > 0.0      {  score.abs().max(1e-6).sqrt() }
             else if score < 0.0 { -score.abs().max(1e-6).sqrt() }
             else                { 0.0 };                     // sqrt(clamp(abs))，带号
let scale  = 1.0 / (1.0 + (-signed).exp());                  // sigmoid
row[d]     = vs[r * ne + d] * scale;                         // repeat(value) * gate
```

**数学上完全一致。**

Vulkan 后端 `shaders/qwen_ple_gate.comp` 同样逐行对上：

```glsl
float score = sums[0] / sqrt(float(pc.n_embd));
float signed_root = score > 0.0 ? sqrt(max(abs(score), 1e-6))
                  : score < 0.0 ? -sqrt(max(abs(score), 1e-6)) : 0.0;
sums[0] = 1.0 / (1.0 + exp(-signed_root));
```

### 而且签名也吻合

```rust
Op::QwenPleGate { key, query, value, dst, rows, hc, n_embd }
```

- `key` / `query` —— engram 的 `grouped_norm(key, engram_k)` / `grouped_norm(x, engram_q)`
- `value` —— `kv` 的共享 value 段
- `rows` = `hc * nt`（每个 (token, hc 副本) 一行）
- `hc` / `n_embd` —— 广播与归约维度

**engram 需要的输入，这个 Op 一个不缺。**

### 所以修复是

1. 在 `engram_wkv` 上做一次 `Op::Linear`，得到 `[hc*n_embd + n_embd, nt]`
2. `key` 取 `[0, hc*n_embd)`，`value` 取 `[hc*n_embd, +n_embd)`（`w_off` 已支持）
3. 对 `key` 和 `x` 各做一次分组 RMSNorm 并乘上 `engram_k`/`engram_q`
4. 推一个 `Op::QwenPleGate`
5. 结果加回残差

**全部用现有 Op，不需要新 shader，不需要新 enum 变体。**

### 修正后的判断

| 之前 | 现在 |
|---|---|
| 缺 6 个算子，一天量级 | **全部现成，一轮可能做完** |
| 新 shader 正确性无法验证 | **复用已验证的 shader**（它有 fixture） |

### 唯一还需要确认的

`QwenPleGate` 的 `value` 广播语义：CPU 里是 `vs[r * ne + d]`（`r = rh / hc`），
即**每 hc 行共用同一个 value**。这正是 engram 要的（"the value is shared across the
copies, only the gate differs"）。

**语义吻合。**

### 下一步

下一轮直接动手：在 `runner.rs` 的层循环顶部、`hc_mix` 之前，插入 engram 的
5 步发射。改完编译、跑 `synthetic_deepseek2`、再用 `check-v41-answers.sh` 验。

