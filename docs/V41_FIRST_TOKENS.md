

---

## Round 119：找到并修复「窗口冻结」的第二个 bug —— out/cur 在 decode 路径上都不写

### 探针证据

真实 turn 里 `tok` 在变，`prev` 却纹丝不动：

```
tok=27526  prev=[2, 23166, 27526]      ← prefill 最后一步，正常
tok=0      prev=[31, 128804, 128821]   ← decode 第 1 步
tok=128799 prev=[31, 128804, 128821]   ← 同一个
tok=77666  prev=[31, 128804, 128821]   ← 同一个
tok=288    prev=[31, 128804, 128821]   ← 同一个
```

再打 `history.len()` 确认：

```
ENGRAM_HIST pos=0 prompt=35 out=0 hist=35
ENGRAM_HIST pos=6 prompt=35 out=0 hist=35   ← out 始终为 0
```

### 根因

**`out` 和 `cur` 在 record-once decode 路径上都不被追加。** 两者都停在 prompt 长度，
所以 `history` 永远是 prompt，每一步的前驱窗口完全相同。

**warmup 掩盖了这个 bug**：warmup 走的是会追加 `out` 的路径，所以第一版修复在 warmup 上
看起来是对的。**这正是 Round 118 那个教训的延伸——观测对象选错，结论就错。**

### 修复

在**注入点本身**记账，那里才是 token 真正交给 forward 的地方：

```rust
let mut engram_seen: Vec<u32> = Vec::new();   // 闭包外，可变捕获
...
let mut history: Vec<u32> = prompt.to_vec();
history.extend(engram_seen.iter().copied());
// ... gather + upload ...
engram_seen.push(tok as u32);                 // 本步 gather 之后再记，避免看到自己
```

### 修复后：窗口正常推进（输出形态改变）

| engram | `2+2=` 输出 |
|---|---|
| **有** | `**  **  **  ** ` |
| **无** | `We need answer. The question is "` |

**两者仍然不同，且都错。** 但与修复前相比，有 engram 一侧从
`** 0.0.0`（Round 117 观测）变成了纯 `**` 重复 —— 窗口推进确实改变了模型行为。

### 诚实评估

**又修好了一个真 bug，但答案依然不对。**

engram 这条链现在：参与计算 ✅、数学正确 ✅、数据健康 ✅、
**窗口正确推进 ✅**（本轮新达成）。但 `2+2=` 仍不是 `4`。

**⇒ 剩余问题不在 engram。** 无 engram 时模型同样答错，说明是更上游的问题
（本 goal 里的 compressor/APE 那条线仍未处理）。

### 状态

| 项 | 值 |
|---|---|
| 工作区 | 干净 |
| 编译 | ✅ |
| 本地提交 | **92 个** |
| prefill | 6 tok/s |
| decode | 5.7–6.4 tok/s |

