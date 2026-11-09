
---

## Round 182：`INFR_PROMPT_OVERRIDE` —— 第一次做到严格同输入

### 加的钩子

`INFR_PROMPT_OVERRIDE=<path>`：用该文件的内容**完全替换** chat template 生成的结果。

加它的原因：`infr` 的模板无法产生参考引擎那种 `<|Assistant|>` 干净结尾
（默认追加 `<|think>`，`--no-think` 追加 `</think>`），所以此前所有"同 prompt"的对比都不成立。

### 第一次严格对齐的对比

```bash
printf '<|begin▁of▁sentence|><|System|>You are a helpful assistant.<|User|>What is the capital of France? Answer in one word.<|Assistant|>' > /tmp/q2.txt
llama-tokenize -f /tmp/q2.txt --ids
```

两边编码结果**逐 token 相同**：

```
[0, 128799, 3476, 477, 260, 11502, 22896, 16, 128803, 3085, 344, 270, 6102, 294, 8760,
 33, 9361, 295, 834, 2004, 16, 128804]
```

输出：

| | 结果 |
|---|---|
| **参考** `strata-dsv41` | 生成 `271, 5, 223, 19, 16, 10793`（连贯） |
| **moe4all** | top5 = `201:8.49%, 344:4.04%, 304:3.47%, 16:3.04%, 734:2.67%`，熵 6.405 |

**⇒ 同样的 token、同样的权重，一个能回答，一个不能。缺陷现在是严格证明的。**

（唯一的交集是 token `16`，moe4all 排第 4，参考在第 5 个生成位置 —— 不足以说明任何东西。）

### 这个钩子的意义

**它把"输入对齐"这个此前反复出错的环节变成了机械可验证的步骤：**
dump prompt → tokenize → 逐 token 比对 → 两边跑。

**在此之前，本会话有 10 次因为输入未真正对齐而得出错误结论。这个钩子是那些错误的根源修复。**

### 下一步

有了严格对齐，可以做真正有意义的实验：
- 逐层残差流（仍受 `Internal` 张量不可绑定的限制）
- 或者在**已对齐的这组输入**上，逐个开关做 A/B（engram / hc / kv 精度），看哪个能把
  `201` 推向 `271`

