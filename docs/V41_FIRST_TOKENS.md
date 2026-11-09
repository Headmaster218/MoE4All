

---

## Round 118：追 tokenizer —— 两次错判，最后发现是 warmup 干扰

### 错误判断 1：token 23166 = "Hi" ⇒ prompt 被替换

我用 `/data/models/dsv41-official-meta/tokenizer.json` 查 `23166` 得到 `'Hi'`，
据此认为 prompt 被错误替换。

**错在**：该 `tokenizer.json` 的 id 空间与 GGUF 内嵌的 tokenizer **不一致**
（它把 `'2'` 映射到 20，而实际喂入的是 23166）。

### 错误判断 2：三个不同 prompt 都 `len=1` ⇒ MESSAGE 没传到模型

我又测了三个不同 prompt，全部得到 `ENGRAM_PROMPT len=1 ids=[23166]`，
据此认为 CLI 丢弃了消息。

**错在**：没有意识到**每个进程都会先跑一次 warmup**。

### 真相（`ENGRAM_TURN` 探针）

```
ENGRAM_TURN prompt="Hi"                          ← warmup
ENGRAM_PROMPT len=1 ids=[23166]

ENGRAM_TURN prompt="...2+2=..."                  ← 真正的用户 turn
ENGRAM_PROMPT len=35 ids=[0, 128799, 77666, ...] ← 35 token，完全正常
```

**`"Hi"` 是引擎的 warmup 调用。** 我看到的每个 `len=1` 都是 warmup，不是我的 prompt。

**⇒ 真实 prompt 有 35 个 token，tokenizer 和 CLI 都正常。**

### 方法教训（第三次同类）

| 轮次 | 我凭什么下结论 | 错在哪 |
|---|---|---|
| Round 111-113 | 分支代码看起来对 | 没跑 runtime 证据 |
| **Round 118a** | 一个**不同来源**的 tokenizer.json | 没验证两个 tokenizer 是否同源 |
| **Round 118b** | 三次观测都一致 | 没意识到采样的是**同一个 warmup**，而非三次调用 |

**共同模式：证据的「一致性」不等于「正确性」** —— 三次相同观测若都来自同一个非目标对象，
一致性反而会强化错误结论。

### 现在确知的事实

| 项 | 状态 |
|---|---|
| prompt 分词 | ✅ 35 token，正常 |
| CLI 参数传递 | ✅ 正常 |
| 引擎 warmup | ✅ 存在，输出 `"Hi"` |
| **engram 在真实 turn 中的行为** | **待重新验证** |

**之前 engram 的 A/B 结论可能被 warmup 污染过**——需要重跑，只统计真实 turn 的 `ENGRAM_FEED`。

