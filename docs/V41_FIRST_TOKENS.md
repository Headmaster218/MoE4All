

---

## Round 85：撤回上一轮，维度链实际完全吻合

### 我上一轮读反了 ggml 的维度顺序

Round 84 我说"`wkv` 输出 6144，注释说 25600，对不上"。

**错的是我。** ggml 的 `dims` 是 `[ne0, ne1]`，**`ne0` 是最快变化维（输入侧）**：

```
blk.1.engram_wkv.weight  dims = [6144, 25600]
  ne0 = 6144    <- 输入（gathered embedding 宽度）
  ne1 = 25600   <- 输出 = n_embd * (hc_mult + 1)
```

**25600 = 5120 × 5，与 `weights.rs:311` 的注释完全一致。注释是对的。**

### 完整参数链（全部从文件验证）

| 参数 | 值 | 验证依据 |
|---|---|---|
| `engram.head_count` | **8** | 直接读出 |
| `engram.max_ngram_size` | **4** | 直接读出 |
| `engram.key_length` | **256** | 直接读出（= `engram_head_dim`） |
| `n_bucket` | **24** | `multipliers=[8]` = 2 层 × 4；`primes=[48]` = 2 × 24 |
| `n_cols` | **24** | `(4-1) × 8` |
| **gathered emb 宽** | **6144** | `key_length(256) × n_cols(24)` ✓ **= wkv ne0** |
| **wkv 输出** | **25600** | `n_embd(5120) × (hc_mult+1)(5)` ✓ **= wkv ne1** |

**四路对账全部闭合。**

### 于是 engram 前向的真实形状

```rust
// 1. gather: 每 token 24 行，每行 256 字节 → reshape [6144, nt]
emb: [6144, nt]

// 2. 一次投影
kv = wkv @ emb                          // [25600, nt]
key   = kv[0       .. 20480]            // hc_mult*n_embd = 4×5120
value = kv[20480   .. 25600]            // n_embd = 5120

// 3. 分组归一化（每 (token, hc 副本) 在 n_embd 上）
k = grouped_norm(key, engram_k)         // engram_k [5120, 4]
q = grouped_norm(x,   engram_q)         // x 是残差流前 n_embd 部分

// 4. gate（QwenPleGate 干的正是这个）
g = sigmoid(sgn(dot(k,q)/sqrt(n_embd)) * sqrt(clamp(abs(·),1e-6,inf)))

// 5. 加回
x += repeat(value, hc) * g
```

**这个形状与我 Round 81 设计的 `QwenPleGate` 接法是吻合的**——当时依据注释，而注释是对的。

### Round 84 那一条"注释与实现不符"作废

我在日志里写过"`weights.rs:311` 的注释与张量不符，Round 81 的方案建立在注释之上、不能实施"。

**那句话是错的**，起因是我把 ggml 的 `[ne0, ne1]` 当成 `[rows, cols]` 读了。

### 教训

**ggml 的 dims 顺序与直觉相反**：`[25600, 6144]` 才是我原本以为的"25600 行 × 6144 列"。
读张量形状时，`dims[0]` 是**输入/最快变化维**，不是行数。

这条已经错了一次，代价是整整一轮（Round 84）的结论作废。

