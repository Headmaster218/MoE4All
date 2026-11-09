

---

## Round 115：找到并修复 history 的真 bug —— decode 阶段窗口曾是静态的

### 探针直接照出问题

在 `gather_for_layer` 里打印 `tokens` 与 `prev`：

```
tokens=[23166] prev=[[2, 2, 2]]         ← prefill 第 1 个，无前驱，正确
tokens=[60170] prev=[[2, 2, 23166]]    ← prefill 第 2 个，正确
tokens=[0]     prev=[[2, 2, 2]]        ← **decode 第 1 个，前驱全部退化为 pad**
```

### 根因

我写的是 `history = prompt[..pos.min(prompt.len())]`。

**prefill 阶段正确**（`pos` 在 prompt 内）。
**decode 阶段 `pos` 超过 `prompt.len()`** ⇒ 切片被 clamp 成整个 prompt ⇒ `tok` 是生成的
token，而它的前驱窗口是对一个**停止增长的序列**算的。

**⇒ 整个 decode 阶段（V4.1 几乎所有步都在这里）的 engram gate 都在读 padding 上下文。**

### 修复

```rust
let mut history: Vec<u32> = prompt.to_vec();
history.extend(out.iter().copied());
```

序列 = **prompt + 本轮已生成的 token**。`tok` 本身是即将喂进去的那个，
所以正确地不在两个切片里。

### 修复后（窗口开始滑动）

```
tokens=[23166] prev=[[2, 2, 23166]]       ← 有前驱了
tokens=[27526] prev=[[2, 23166, 27526]]   ← 窗口在滑动
```

### 顺带确认布局是对的

探针还显示：**两层用同一个 token、同一个前驱窗口，只有 row index 不同**——
这正是 `EngramMap::slot` 应有的行为。

而且布局链是对的：`gather_for_layer` 按 token 拼 bucket（token-major）、
`gather_rows` 每个 row 展成 `key_length` 个 float ⇒ **`[token][bucket][key]`**，
与参考的 `reshape(key_length*n_cols, n_tokens)` 一致（ggml 的 `ne0` 是最快变化维）。

**上一轮怀疑的布局问题不存在。**

### 但答案还不完整

输出是 `0`（单字），不是 `4`。方向对了，但还差一点。

### 下一步

`0` 至少是个数字。可能还差：**engram 是否应只在前缀 token 上生效**、
或 **`rel`/`wkv` 的列切分偏移**。下一轮逐项核对。

