

---

## Round 75：三条源码线索逐一排除，以及我自己的一次误判

### ① 43 项 vs 40 层：不是偏移问题

`config.rs:1170` 是 `arr[..n_layer]`——**截取前 40 项**。
多出的 3 项被直接丢弃，索引不存在偏移。

### ② 角色语义：代码是对的

`config.rs:1181-1195` 把语义写得很清楚：

```
0 = pure sliding-window layer; reads no compressed stream at all.
1 = a compressed SOURCE that pools one token per row (unpooled).
2 = a compressed layer pooling two tokens per row, OR a reuse layer
    reading an earlier source's stream — the source map decides which.
```

**所以 `visible = (pos+1)/ratio4` 对 role=2 除以 2 是正确语义**，不是"拿角色当比例"。
Round 74 我把它当成缺陷，**这个判断撤回**。

### ③ 我犯了一个更基础的错误：只读了一个分片

Round 75 中我解析第一个 gguf 发现"只有 42 个张量、没有 attn_compressor_kv"，
于是推断"所有权判定全落空"。**这是错的。**

模型是**四分片**（`-00001-of-00004` 等），第一片只有专家权重和 engram 表。
把四片合起来：

| 分片 | 张量数 |
|---|---|
| 00001 | 42 |
| 00002 | 36 |
| 00003 | 34 |
| 00004 | **896** |
| **合计** | **1008** |

（`config.rs` 注释里那句 "0 of 1008 tensors" 正是这个总数。）

用全集重新验证属主：

```
owns attn_compressor_kv : [2, 8, 14, 20]            ← 与 config 声明一致
owns indexer.attn_q_b   : [2,8,14,20,24,28,32,36]   ← 与 config 声明一致
```

**和代码推断完全吻合。** 所有权判定没有问题，`owns_compressor` 会正确地为 4 个 source 层
返回 true、为其余 36 层返回 false。

### 本轮净结果：三条线索全部排除

| 线索 | 结论 |
|---|---|
| 43 vs 40 偏移 | 不存在，`arr[..n_layer]` |
| 角色当比例 | 语义正确，撤回该判断 |
| compressor 属主 | 与文件完全一致 |

### 剩下的

`hc_mult` 残差 与 IQ1_M 专家 dequant 仍是候选，但都需要外部参照才能判定。
**本轮没有找到新缺陷。**

### 教训

**多分片 gguf 必须先合并元数据再下结论。** 我用第一片 42 个张量推断出了两个错误结论
（"compressor 不存在"、"所有权全落空"），幸好验证时自己发现了。

