

---

## Round 38：读 IQ1_M mmq shader，排除了三项，剩 `wbase` 一处可疑

### 已读并确认无误的部分

| 项 | 位置 | 结论 |
|---|---|---|
| `d_bits` 组装 | 第 160-161 行 | 与 `infr-gguf/dequant.rs` 的 Iq1M 分支**逐位相同** |
| `rb()` 字节寻址 | 第 101-102 行 | 与 IQ2_S 的 `rb()` **完全相同**（`nw[byte >> 2]`） |
| 屏障结构 | 第 202 行 + 第 250 行 | **正确**：202 行是「写完 As/Bs/Bdl1/Bdl2/Bneg 后」的屏障，250 行是循环回边前的屏障，写后读与读后写都被覆盖 |
| `dl1/dl2` 尺度解码 | 第 168-169 行 | `scl >> 6*(ib%2)` 与 `>> 6*(ib%2)+3`，与 host 一致 |
| delta 符号 | 第 170-178 行 | `qh0&0x08 → l=0`、`qh0&0x80 → l=1`，与 host 一致 |

### 剩下最可疑的一处：`wbase` 的语义

```glsl
uint gstart = wbase + (wgCol + cc) * pc.k + sub * BLK;
uint bd = (gstart / 256u) * IQ1M_BLOCK_BYTES;
```

resident 分支里 `wbase = pc.w_base + ex * pc.m`。

**`pc.m` 在 resident 下传的是什么？** 在 STREAMED 分支里它被明确注释为「per-expert BYTE stride」，
而这里同一行代码把 `ex * pc.m` 加到**元素偏移** `wbase` 上。两者语义不同则地址必错。
这是我下一步要确认的地方。

### 本轮没有结论

读了 shader，排除三项，定位到一处可疑但**未证实**。没有代码改动，工作树干净。

PR #183：70 commits。

### 教训

这一轮又是「读代码试图找 bug」。前几轮已经证明这种做法效率低：
Round 33-37 用**测量**（对照 dtype、真实数据、另一条路径）五轮走到了确定结论；
而读代码一项一项排除，慢且无法证明。

**下一轮应该测而不是读**：把 `wbase` 的语义在两处打日志对比，或者写一个最小用例——
`n_expert=1`、`ex=0`，此时 `ex * pc.m = 0`，若结果正确则说明问题就在 `ex * pc.m` 这一项。

