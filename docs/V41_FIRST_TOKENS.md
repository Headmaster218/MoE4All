

---

## Round 44：两个否定结果 —— 不是 debug 变体的问题，也不是 resident_nb

### 做了什么

上一轮的结论是「新增的 `_xp_dbg` 变体没有到达设备」。这一轮绕开它：
**把探针无条件加进生产 `_xp` kernel**（同一 shader，去掉 `-DIQ1M_DEBUG`），
并把 `INFR_IQ1M_MMQ_DEBUG` 的 kernel 覆盖整段删掉。

### 结果一：magic 仍然不出现

```
MAGIC ABSENT (saw 0) - the probe never executed; nothing else is readable
```

探针现在位于 IQ2_S 能正常工作的**同一条生产路径**上。所以：
**"新变体没注册" 这个解释被排除。**

### 结果二：`resident_nb: 8 → 6` 也不是原因

IQ1_M 的 `resident_nb` 是 8，而 EXPERT_GRID 构建的 shader 只有 6 个 binding、
`bufs` 也只给 6 个——看起来像个明显的类型错误（注释写的是 "8 activation blocks"，
那是 k 循环次数不是 binding 数）。改成 6 之后：

```
MAGIC ABSENT (saw 0) - ...
Iq1M xp mismatch at 0: got 0 want -0.05855285
```

**完全没变。**

### 两个改动都已撤回

工作树回到 `90b42b0`，干净。没有提交代码。

### 现在的状态

**探针在生产 kernel 里都不执行**，这意味着 `matmul_mmq_experts` 对 IQ1_M 的 dispatch
要么没有发出、要么被丢弃。IQ2_S 走同一函数却能工作，所以差异只在参数。

还没排除的：（a）`push_size=24` 与 IQ1_M 的 PC 布局不符导致 pipeline 建失败；
（b）`gx`/网格维度退化为 0；（c）`kernel()` 对该 name 建 pipeline 时静默失败。

### 下一步该换个层级

连续四轮（41–44）都卡在「让插桩跑起来」，而没有向前推进缺陷本身。
**下一轮不该继续修插桩**，而是：
在 `matmul_mmq_experts` 里直接打印它实际用的 `name` / `nb` / `gx` / `n_expert` / `push`
（Rust 侧 `eprintln!`，不需要 GPU 执行就能看到），先确认 dispatch 参数是否正确。
**Rust 侧的日志比 shader 侧的探针便宜得多。**

