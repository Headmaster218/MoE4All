

---

## Round 80：找到了 —— engram 权重上传了，但从未参与计算

### 编译器给出的证据

```
$ cargo check -p infr-llama
warning: fields `wkv`, `k`, and `q` are never read
  --> crates/infr-llama/src/seam/weights.rs:315
   |
315| pub(super) struct EngramW {
```

**`EngramW` 的三个字段 `wkv` / `k` / `q` 从未被读取。**

### 这不是我读代码的推测，是 rustc 的静态结论

| 检查 | 结果 |
|---|---|
| `grep -rn "EngramW" crates/` | 只有定义、构造、字段声明三处 |
| `grep -rn "\.engram"` 全部消费点 | **零** |
| `infr-core/src/graph.rs` 有 Engram Op 吗 | **没有** |
| `cargo check` | **`never read`** |

### 而代码在别处是"支持 engram"的样子

| 位置 | 行为 |
|---|---|
| `config.rs:551` `has_engram()` | 判定层 1、14 有 engram |
| `config.rs:1320-1361` | **一整套配置校验**（head count、2-gram、key length、prime 非零、multipliers 长度） |
| `runner.rs:2766` | 加载 `engram_wkv/k/q.weight` |
| `runner.rs:5323` | 构造 `EngramW` 放进 `LayerW` |
| `weights.rs:309-313` | 注释说"这三个是唯一上传的 engram 权重" |
| **计算** | **不存在** |

**配置校验、权重加载、结构体、注释全都写好了——唯独没写用它算。**

这正是 Round 79 收敛出的那类故障：**组件是对的，但它没有被调用。**

### 为什么这足以解释全部现象

`engram_layer_ids = [1, 14]`，层 1 是**第二层**。

残差流逐层累积——**第 1 层注入缺失，后面 39 层全部偏离**。

模型不是"某个模块算错"，而是**从第 1 层起就走在一条与训练时不同的残差轨迹上**，
最终输出只能是语法通顺但没有内容的套话（`We need answer. The question`）。

这与我 Round 73 的排除理由（"只有 2 层"）正好相反——**恰恰因为它在第 1 层**。

### 未确认的部分

1. **engram 是否"必须"**——有没有可能缺了它只是精度下降，不至于答错 `2+2=`？
   参考实现 `deepseek41.cpp` 是把它当必经路径算的（`engram_index(il) >= 0` 就注入）。
2. **补上它有多大工作量**——需要：host 侧 hash（`crate::engram` 已有）、
   gather（已有）、三个投影的 Op 序列（`wkv` 拆两次 GEMV + `k`/`q` 的 norm 缩放）。
   **`Op::Linear` 的 `w_off` 已经支持拆分**（注释提到与 V4 分组输出投影同一技巧）。

### 下一步

按 `deepseek41.cpp` 的 engram 注入段，把三个投影接进 graph。
预计是几十行 Op 序列，不是重写。

**但第一步先确认参考实现里 engram 注入的确切位置和顺序**
（`deepseek41.cpp` 的 `engram_index(i) >= 0` 分支）。

