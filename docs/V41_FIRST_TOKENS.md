

---

## Round 131：模型能答对翻译，却把「法国首都」答成 100 —— 指向输出层映射

### 关键对照数据（`temp=0`，无 engram）

| prompt | 输出 | 判定 |
|---|---|---|
| `Translate hello to French:` | `. Wait, the answer is "Bonjour".` | ✅ **正确** |
| `What are the first five primes?` | `We need answer. Need interpret. Need likely ask` | ❌ |
| `The capital of France is` | `. Wait, the answer is 100. So` | ❌ **荒谬** |
| `5+3=` | `We need answer. The question is "8". Wait,` | ⚠️ 答案对，表述乱 |
| `1+1=` | `. Wait, the answer is 2` | ✅ |
| `Name three colors:` | `. Wait, I need to output in the final answer.` | ❌ |
| `France`（单词续写） | `. The user wants me to translate the` | ❌ |
| `Paris`（单词续写） | `. The question is: "What is` | ❌ |

### 结论一：模型没有全盘失效

它能答对 `Bonjour`、`8`、`2`。**知识和能力都在。**

### 结论二：输出形式被锁死在一个模板

几乎每条都以 `. Wait, the answer is X` 收尾。**模板固定，X 才变化。**
「法国首都是 100」——**算术题的答案串到了知识题上**。

### 结论三：warmup 不是污染源（已 A/B 排除）

`INFR_NO_WARMUP=1` 跳过抛掷式的 `"Hi"` 预热轮：

```
A: 有 warmup  →  <think>. Wait, the answer is 100. So
B: 无 warmup   →  . Wait, the answer is 100. So
```

**上下文完全干净时，模型仍然说「法国首都是 100」。** 唯一差别是 A 有 `<think>`、B 没有。
⇒ `reset_kv` 是干净的，warmup 无罪。

### 当前最强假设：输出层权重映射错位

症状自洽地指向「语言模型形式正确、内容错位」：

1. 形式完全符合 LM（会写 `. Wait, the answer is`）
2. 内容系统性错位（`100` 出现在知识题里）
3. logits 高置信（Round 130：entropy 0.001–0.103，top1 99%+）——
   错位的映射本身也是自洽的，所以不会出现低置信
4. 换 prompt 输出模板相同

**具体怀疑：`lm_head` 与 `token_embd` 的绑定（tied embedding）挂反或挂错。**
DeepSeek 系列常用权重绑定；若 GGUF 分开存而引擎取了错的张量，
就会得到「语法对、语义错」的输出。

### 下一步

核对 `lm_head.weight` 与 `token_embd.weight`：
1. 模型里两者是否都存在、形状是否一致
2. 引擎实际挂的是哪一个
3. 如果模型声明了 tying，引擎有没有遵守

