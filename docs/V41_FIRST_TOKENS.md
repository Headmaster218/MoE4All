

---

## Round 69：挂起的原因找到了 —— 是我自己开的 bring-up 开关

### `runner.rs:12370` 的注释直接写了答案

```
// MEASURED, 2026-11-08: enabling it for V4.1 is NOT viable today. ... a 2032-token
// prompt loses the DEVICE ... this is a reproducible hang in the batched V4.1 graph,
// not a card problem. ... Until the hc_mult-width residual is carried across a chunked
// span properly, the family stays on the per-token path, which is correct and merely
// slower.
```

**根因**：V4.1 的 `hc_mult` 宽度残差**没有在分块 span 之间正确传递**。

batched prefill 对每个 chunk 构建整模型图（`span = None`），残差跨 chunk 丢失，
所以 **batch > 1 的 V4 图从来没有正确执行过**。

**而 `INFR_DSV41_BATCHED_PREFILL` 是我在前面几十轮里一直开着的。**

### 关掉它之后

16K 不再挂起：进程 CPU 时间持续增长（2:49 → 8:25），读字节 97.7 → 126.3 GB，
说明它在**真的干活**，只是走 per-token 路径很慢。

**但跑了 50 分钟仍没完成**（被 3000s timeout 截断）。

### 所以 16K 的真相

| 路径 | 结果 |
|---|---|
| `INFR_DSV41_BATCHED_PREFILL=1`（我一直在用） | **~20 分钟后 device lost** |
| 默认（per-token） | **不挂，但 50 分钟算不完** |

**两条路都不可用，但原因不同**：一条是图构造缺陷，一条是纯速度。

### 这个教训很重

**我在前面所有 prefill 测量里都开着这个 bring-up 开关**，
而注释明写它是 opt-in 的 bring-up 实验、当前不可用。

- 4K / 8K 在 batched 路径下能跑通（27.5 / 26.3 t/s），所以那部分数据有效
- 但 16K+ 的失败，**直接原因是我用了不该默认启用的路径**

正确做法：**默认路径（per-token）才是产品行为**，它不挂但很慢；
batched 路径是实验性的，只在能跑通的规模上才有意义。

### 修正后的可用性

| 规模 | batched（实验） | per-token（默认） |
|---|---|---|
| 4K | 27.5 t/s ✅ | 未测（历史上 ~6-7 t/s） |
| 8K | 26.3 t/s ✅ | 未测 |
| 16K | ❌ device lost | ❌ 50+ 分钟未完 |

