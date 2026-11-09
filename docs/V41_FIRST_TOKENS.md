

---

## Round 133：引擎通用路径是好的 —— 缺陷只在 V4.1 专有代码里

### 对照组实验（终于做了）

同一个 `infr` 二进制、同一台设备 `Vulkan1`、同样走 Vulkan seam：

| 模型 | prompt | 输出 | 判定 |
|---|---|---|---|
| **qwen2.5-0.5b** | `A B C D E F G` | `I'm sorry, but I don't understand what` | ✅ **正常** |
| **DeepSeek-V4.1** | `A B C D` | `-` | ❌ |

qwen 同时给出 `prefill 249 t/s / decode 127.6 t/s`，速度也正常。

**⇒ 引擎的通用推理路径是健康的。** 问题被限制在 V4.1 这条 arch 专有分支上。

### 这条排除为什么重要

我前面几轮反复怀疑「输出层映射」「lm_head 绑定」这类**通用**环节。
对照组直接否定了这个方向 —— 通用环节若坏了，qwen 也会错。

**缺陷范围现在只剩 V4.1 专有代码：**
1. **engram**（两层注入；引擎自己警告不设表会「contribute garbage」）
2. **压缩注意力**（`kv_source_layer_ids=[2,8,14,20]` 四个源层 + 34 个复用层）
3. **超连接 `hyper_connection`**（`hc` 份残差副本 + 混合）
4. **MoE 路由**（含 hash routing `ffn_gate_tid2eid`）

### 同时撤销 Round 129 的无效排除

```
infr: INFR_ENGRAM_TABLE is unset; V4.1 engram will contribute garbage
```

引擎明确警告：不提供表就会注入垃圾。所以 Round 129 用「不设 engram 仍答错」来排除 engram
**是无效对照** —— 那不是「关掉 engram」，是「用垃圾 engram」。**engram 重新列为嫌疑。**

### 症状汇总（供后续定位）

- `Once upon a time` → `. Wait, I need to produce the final answer. The answer is 100 ... 10`
  —— **语法完整、逻辑自洽、主题与输入毫无关系**
- 所有输出坍缩到 `. Wait, the answer is X` 模板
- logits 健康（熵 0.001–0.103，top1 98–99.99%）
- `A B C D` 接不出 `E`，`1 2 3` 接不出 `4`
- 偶尔「对」（`Bonjour`、`8`、`2`），但更像高频关联碰巧命中

**这些共同指向：模型看到的上下文表示被系统性偏移了，而偏移来源在 V4.1 专有路径。**

