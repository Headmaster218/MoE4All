
---

## Round 172：缓冲宽度全面扫描 —— 没有再发现同类问题，但找到批量路径的第二个缺陷

### 方法

列出 `runner.rs` 里全部约 90 处 `g.internal(f32d(...))` 分配，逐个与消费它的 op 的写入量对照。

### 结果一：默认路径（batch == 1）没有第二处越界

`hcr`、`hc_pre_lag`、`engram_k_normed`、`engram_q_normed`、`engram_gated`、`q`、`k`、`v`、`attn`、`gu`、`sub`
等全部与内核写入量一致。**Round 170 修的 `hn` 是唯一一处。**

### 结果二：批量 prefill 路径有一整批不带 `batch` 的缓冲

```
5810: d4_comp_values = (2 * c.head_dim)           ← 应为 batch * ...
5811: d4_comp_scores = (2 * c.head_dim)
5812: d4_comp        = c.head_dim
5813: d4_lid_values  = (2 * c.indexer_head_size)
5814: d4_lid_scores  = (2 * c.indexer_head_size)
5815: d4_lid         = c.indexer_head_size
5816: d4_ix_q        = (c.indexer_n_head * c.indexer_head_size)
5823: d4_ix_w        = c.indexer_n_head
```

`Op::Dsv4Compress` / `Op::Dsv4Indexer` 在 `batch > 1` 时会写 `batch × ...`，
而这些缓冲只按单 token 分配 —— **与 `hn` 完全相同的越界模式。**

**但这不影响默认路径**（`batch == 1` 时 `batch * X == X`，尺寸恰好），
而且 `INFR_DSV41_BATCHED_PREFILL` 已经是 opt-in，批量入口有 release 常驻断言拦截。

**这解释了批量 prefill 为什么"数值不可信"** —— 不只是 Round 140 发现的"只写一行 KV"，
这些缓冲也会越界。批量路径要真正可用，这两处都得修。

### 当前状态

| 项 | 结果 |
|---|---|
| `hn` 越界（60 KiB/层） | ✅ 已修（Round 170），置信度 6.31%→17.18%，输入敏感性恢复 |
| 默认路径还有同类越界？ | ❌ 扫描后确认没有 |
| 批量路径的缓冲越界 | ⬜ 已知，待批量路径立项时一起修 |
| `The capital of France is` → `294` | ❌ 仍错，剩余缺陷不属于"缓冲越界"类 |

**⇒ 剩下的问题需要换一类假设。** 缓冲维度这条线已经走完。

