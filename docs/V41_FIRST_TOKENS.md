

---

## Round 81：engram 算法已完整读出，但修复是大工程

### 参考实现的 `build_engram`（`deepseek41.cpp:382`）

注入点在**每层主循环最开头**，`build_hc_mixes` 之前：

```cpp
if (pmodel.engram_index(il) >= 0) {
    inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
}
```

算法：

```cpp
// 1. 一次投影产出 hc 个 key + 一个共享 value
kv    = engram_wkv @ emb                      // [hc_dim + n_embd, nt]
key   = kv[0        .. hc_dim]                // 每 hc 副本一个 key
value = kv[hc_dim   .. hc_dim + n_embd]       // 所有副本共享

// 2. 分组 RMSNorm：按 (token, hc 副本) 在 n_embd 上归一，不是跨副本联合
k = grouped_norm(key, engram_k)               // reshape [n_embd, hc, nt] → rms_norm → 乘 w
q = grouped_norm(x,   engram_q)

// 3. 打分
s = sum_rows(k * q) * (1/sqrt(n_embd))

// 4. 符号平方根后 sigmoid（对应训练 kernel）
mag  = sqrt(clamp(abs(s), 1e-6, inf))
gate = sigmoid(sgn(s) * mag)

// 5. value 广播到 hc 份，乘 gate，加回残差
v = repeat_4d(value, n_embd, hc, nt, 1)
return x + v * gate
```

### 本项目缺的算子

| 步骤 | 需要的 Op | 本项目 |
|---|---|---|
| 投影 | `Linear` | ✅（`w_off` 支持拆分） |
| 分组 RMSNorm | `RmsNorm` | ✅ |
| 元素乘 | — | 有 `MulVec`，但**广播语义不同** |
| `sum_rows` | 按行归约 | ❌ **无** |
| `sqrt` | — | ❌ **无** |
| `abs` | — | ❌ **无** |
| `clamp` | — | ❌ **无** |
| `sigmoid` | — | ✅（`Sigmoid` 存在） |
| `sgn` | — | ❌ **无**（`QwenPleGate` 里可能内联） |
| `repeat_4d` | 广播 | ❌ **无** |

**缺 6 个基础算子。**

### 这意味着什么

**修复不是"接上三个投影"，而是新增 6 个 Op，每个都要：**
1. `infr-core/src/graph.rs` 加 enum 变体 + 依赖/输出声明
2. CPU 后端实现
3. **Vulkan 后端实现（要写 shader）**
4. `synthetic_deepseek2` fixture 覆盖

**这是一天量级的工程，不是一轮能完成的。**

而且每加一个 shader 就要编译、要在 GPU 上验证正确性——
**在模型本身还答不对题的现在，新 kernel 的正确性无法用 `2+2=4` 来判**（它本来就错）。

### 但这仍然是本会话最大的进展

| 之前 | 现在 |
|---|---|
| 不知道哪里错 | **知道 engram 没被调用** |
| 四个候选全排除 | **根因定位到具体组件** |
| 只能猜 | **有完整算法和参考实现** |

### 我这一轮不做的事

**不在这一轮动手写那 6 个算子。** 理由：
- 需要 GPU shader，每步都要编译验证，一轮做不完
- 做完之前模型仍然错，无法验证新算子是否写对
- 半成品提交会污染这个已经 45 个提交的分支

**更合理的做法**：先只加 CPU 后端实现（不需要 shader、不需要 GPU），
用现有 `synthetic_deepseek2` fixture 验证算子正确性；Vulkan 版本单独一轮做。

### 下一步

先看 `QwenPleGate` —— 它的结构与 engram gate 极像（同样的 sigmoid(sgn*sqrt(clamp(abs)))），
**如果它已经有现成实现，engram 可以复用它的后端**。

