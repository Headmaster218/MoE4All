

---

## Round 58：区间是什么 —— 每次带时间戳的 submit

### 代码事实

`main_intervals` 的唯一记录点是 `lib.rs:1864`：

```rust
infr_core::pager_profile::record_device_interval(
    DeviceIntervalKind::MainQueue, ticks[0], ticks[1], ...)
```

它读的是 **`SubmitTimingToken` 的一对时间戳查询**（`get_query_pool_results` 的
`ticks[0]` / `ticks[1]`，`WAIT` 模式）。

### 于是

**1149 个区间 = 1149 次带时间戳的 submit。**

而 `main_gaps = 0` 说明这 1149 个 submit 的 GPU 区间彼此相接、互不重叠——
**每个 submit 内部是连续执行的**（`avg_dispatches_per_submit = 23.3` 个 dispatch 一口气跑完）。

### 所以空档在哪

**空档在 submit 与 submit 之间，不在 dispatch 与 dispatch 之间。**

`submits = 1684` 而 `main_intervals = 1149`，说明有 535 次 submit 不带时间戳查询，
所以只有 1149 次可观测——但两者都是"submit 之间的间隙"。

### 这个结论把优化目标锁定了

不要去合并 kernel、不要去减少 dispatch（那 23 个 dispatch 本来就是连续的）。
**要去减少 submit 的次数。**

`submit splitter` 行显示每次 forward 里 `cap_end = split/16`，即被切成 16 段。
如果能减少这个切分，submit 数下降，空档数也随之下降。

### 数据校验

1149 次 submit、span 4.41s → **平均每次 submit 间隔 3.84ms**
其中 GPU 忙 2.49ms、空 1.35ms —— 与实际测到的两个数吻合。

### 下一步

找 `cap_end = split/16` 那个 cap 是从哪来的、能否 A/B 调大。

