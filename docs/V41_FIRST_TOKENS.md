
---

## Round 185：engram 哈希路径与 C++ 参考逐行核对 —— 等价，不是缺陷源

### 核对的三段

**1. `row_indices()` vs `deepseek41.cpp:337-348`**

| 步 | C++ | infr | 一致 |
|---|---|---|---|
| ctx[0] | `map_of(ubatch->token[i])` | `map_of(current)` | ✅ |
| prev 取值 | `prev[n_prev - s]` | `prev[prev.len() - s]` | ✅ |
| blocked 传播 | `blocked = blocked \|\| t < 0` | 同 | ✅ |
| rolling | `ctx[0]*mult[0]` 再 `^=` 逐项 | 同 | ✅ |
| bucket | `b = (s-1)*n_heads + h` | 同 | ✅ |
| row | `rolling % prime[b] + offset[b]` | 同 | ✅ |

**2. `engram_windows()` 的窗口顺序**

`row_indices` 需要 `prev[len-s]` = 最近；`engram_windows` 构造 `win[0]=最远 … win[len-1]=最近`，
与 C++ 的 "prev[] is oldest-first" 完全对应。**✅**

**3. 生产调用链**

`runner.rs:10463` → `upload_layer` → `gather_layer` → `engram_windows` + `gather_for_layer`，
history/batch 的划分（`feeds[..start]` 与 `feeds[start..end]`）逻辑正确。**✅**

### 结论

**engram 的哈希、窗口、取行三段都与参考实现等价。**
**⇒ Round 183 观察到的"置零 engram 后置信度提升 3.3 倍"不是哈希错误造成的。**

### 那它是什么

**最可能的解释是：engram 注入的是**训练时学到的分布统计**，而这份权重（IQ2_XXS 量化）
在 engram 相关的两个层上量化误差特别大。**——但这只是假设，没有证据。**

**另一种可能：engram 的 `engram_k`/`engram_q` 归一化参数在 GGUF 里被量化过，
而 `QwenHcNorm` 把它当作精确 f32 使用。**——同样只是假设。**

### 方法记录（第 11 次）

**这一轮我先假设"`row_indices` 的 ctx 顺序可疑"，然后**读了 C++ 参考**，发现完全一致。**
**教训：怀疑某个算法实现时，先读参考实现再下判断，而不是凭注释推断。**

### 累计

| 子系统 | 状态 |
|---|---|
| engram 缓冲越界 | ✅ 已修 |
| engram 哈希/窗口/取行 | ✅ 与 C++ 参考等价 |
| engram 注入效果 | ❌ 置零后改善 3.3 倍（原因不明） |
| 其它子系统 | 详见上方排除表 |

