

---

## Round 56（续）：A/B 否定 —— staging ring 槽数无关

`INFR_PAGER_RING_SLOTS` 2（默认）vs 8：

```
              slots=2 (默认)      slots=8
decode        6.2 tok/s           6.2 tok/s
main_intervals 1149               1149
main_busy     2.86s               2.86s
outside       1.54s               1.55s
dma_hidden    0.0%                0.0%
staging ring  acquisitions=0      acquisitions=0
```

**三项完全一致。**

### 决定性的一点

**`staging ring: acquisitions=0` —— 这个 ring 从头到尾一次都没被用过。**

所以 1.54s 的空档**不是** staging ring 造成的，增加槽位自然也没有意义。

### 这条线索关闭

第 6 个被 A/B 否掉的假设。

### 现在剩下的候选

空档 = GPU 两次 `main_intervals` 之间的时间。已知：
- 不是 CPU 记录（只占 9%）
- 不是 staging ring（从未使用）
- 不是全队列同步（Round 55 已否）
- DMA 计时为 0，无法直接观测

**下一个该查的是 `main_intervals` 为什么被切成 1149 段、每层 2.39 段。**
如果能把每层的段数降到 1，即使空档不变，也能省掉一半的切换。

