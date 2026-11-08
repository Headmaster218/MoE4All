

---

## Round 59（续）：对账失败 —— 空档其实全部已归因

### 我先前的错误

上一节我手算「已知 host 开销 503ms，空档 1550ms，剩 1047ms 无解释」。

**这个算法漏了一项：`sync_wait_ns`。**

`host_accounted` 的定义（`runner.rs:14264`）是：

```rust
backend_setup + command_recorder_acquire + command_record + queue_submit + sync_wait
```

**`sync_wait` 就在里面**，而它正是 3.88s 那个大头。我没把它算进去。

### 加上 `unattributed` 指标后的实测

```
execute_wall  = 4.27s
attributed    = 4.25s
unattributed  = 26.4ms     <- 0.6%
```

**未归因时间只有 0.6%，几乎为零。**

### 结论

**空档不是"CPU 在干别的事"，而是 CPU 在 `sync_wait` 里等 GPU。**

所以"给这 1047ms 插采样器看它在干嘛"这个计划本身就不成立——
那 1047ms 并不存在，是我算错的。

### 这意味着什么

八轮下来，所有证据现在指向同一个结论：

- GPU 忙 2.86s 是**真的在算**
- GPU 闲 1.55s 时，**CPU 在等它**（`sync_wait` 已计入）
- host 侧没有隐藏的浪费（0.6%）

**decode 6.2 tok/s 就是这套计算+换入的真实成本，而不是某个环节的浪费。**

要更快，只有两条路：
1. **减少 GPU 的工作量**（更少的专家、更少的换入）
2. **让 CPU 不等**（异步流水，让下一个 token 的准备与当前计算重叠）

第 2 条才是 `hit_windows`/`overlap` 那些指标本来要衡量的事——而它们实测是 0。

### 保留的产出

`unattributed` 这个指标本身是有价值的：它把"CPU 时间有没有漏账"变成了一个恒等式，
以后任何 host 侧改动都能立刻看到有没有凭空多出时间。

