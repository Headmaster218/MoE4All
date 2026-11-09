

---

## Round 100：**跑通了** —— unbound tensor 报错消失，模型首次完整执行

### 修复

把 engram 绑定加进 `bind_layer_io` 函数体内部。

**这个函数的文档注释早就写明了该怎么做**（`runner.rs:983`）：

> *"Extracted so the four bind sites (denoise, verify, record-once, per-token) can never drift —
> **a forgotten bind here (e.g. `rope_freqs`) is a live unbound-Input panic at execute**."*

**engram 正是属于这一类：所有执行路径共用这个函数，漏在这里就是 execute 时 unbound。**

### 运行时结果（首次无报错）

```
$ infr run ... "Hi"
[prefill 32 tok @ 6 tok/s (4949 ms) | decode 1 tok @ 6.4 tok/s]
```

**这是本会话第一次完整跑完 prefill + decode 而不崩溃。**

### 但答案仍然错

```
$ infr run ... "2+2=" --max-new 8
"We need answer. The question is "
```

### 为什么现在能跑但仍答错

| 之前 | 现在 |
|---|---|
| `op 67 Linear failed: unbound tensor 3` | 能跑完，输出 token |
| 崩在 execute | 但 engram 的输入内容是**未初始化的 buffer** |

**`engram_emb_buf` 已绑定但从未被填充**——`engram::gather_for_layer`（`dba67e8` 写的）**还没有任何调用点**。

所以现在的状态是：**引擎不再崩，但 engram 贡献的是垃圾数据，模型输出也就没有意义。**

### 这一轮的进展是实质性的

**从"完全跑不起来"到"跑起来但答错"**，这是两个不同阶段的问题：
- 前者是**结构缺陷**（绑定遗漏）—— 已修复
- 后者是**数据未填充** —— 下一步

### 下一步

在 forward 之前调用 `engram::gather_for_layer`，用 `cached` tokens + 本批 token 拼出 `prev` 窗口，然后 `be.upload(engram_emb_buf, ...)`。

**这是最后一步。** 做完之后 `2+2=` 才有机会答对。

