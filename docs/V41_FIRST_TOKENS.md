

---

## Round 84：engram 维度链从张量反推出来，闭合了

### 实测的三组张量

```
blk.1.engram_wkv.weight  [6144,  25600]     F32? → 投影
blk.1.engram_k.weight    [5120,      4]     → 每 hc 副本一个 scale
blk.1.engram_q.weight    [5120,      4]     → 同上
engram 表 blk.1.engram_embd.weight  [384006168, 256]   1 B/elem (E4M3)
          blk.1.engram_embd.scale   [384006168,   8]   1 B/32 elem (E8M0)
```

### 维度对账

| 项 | 值 | 依据 |
|---|---|---|
| `wkv` 输入 | **25600** | `n_embd(5120) × (hc_mult+1)(5)` ✓ |
| `wkv` 输出 | **6144** | `hc_mult×n_embd(20480)`… **等等，不是 6144** |

**这里有个问题：`wkv` 输出是 6144，不是 6144 = ?**

`hc_mult*n_embd + n_embd = 5*5120 = 25600`，但实际输出只有 **6144**。

**6144 = 5120 + 1024**。所以不是 `hc*n_embd + n_embd`。

### 重新看 `EngramW` 的注释

`weights.rs:311-313` 说：

> *"`wkv` is ONE projection split into two offset GEMVs: rows `[0, hc_mult*n_embd)` are the
> per-copy keys and `[hc_mult*n_embd, hc_mult*n_embd + n_embd)` the shared value block"*

按这个描述输出应该是 **25600**，但实际是 **6144**。

**注释与张量不符。**

### 那么 6144 是什么

`6144 = 1024 + 5120`。可能是：
- `key_len(256) × n_head(?)` + `n_embd`
- 或 `1024 = 256 × 4` = `key_length × hc_mult`

**如果 `1024 = key_length(256) × hc_mult(4)`，那么 key 部分每副本 256 宽，value 部分是 n_embd(5120)。**

但那与 `Op::QwenPleGate` 需要的 `key`(=`grouped_norm` 后宽度 `hc*n_embd`) 和 `query`(=`n_embd`) 又对不上。

### 结论：注释和实现可能与真实模型不符

`EngramW` 的文档说 `wkv` 输出 `hc_mult*n_embd + n_embd = 25600`，实测是 **6144**。

**差得很远。** 这说明 `EngramW` 的结构、乃至 Round 81 设计的 `QwenPleGate` 接法，
都是按注释而不是按真实张量形状设计的。

### 这不是坏消息

**这正是为什么"接上就完事"的想法是错的**——真实形状与我依据注释推的形状不一致。

### 下一步

按实测形状重新推导 engram 的前向：
- `emb`（gather 出的 `[25600, nt]`）→ `wkv` → `[6144, nt]`
- 这个 6144 如何拆成 key / value？
- `engram_k`/`engram_q` 是 `[5120, 4]`，4 = hc_mult，说明 **归一化作用在 n_embd 上、每副本一个 scale**

**先算清楚 6144 怎么拆，再谈接 Op。**

