
---

## Round 155：engram 注入确实在左右结果；IQ4_NL 表确认损坏

### 开关 A/B（对齐行 `mine[2]`）

| 配置 | top5 | 熵 |
|---|---|---|
| 基线（`engram-mxfp8.gguf`） | 666:6.7%, 2619:3.4%, 201:3.3%, 29:3.3%, 223:2.8% | 6.411 |
| `INFR_ENGRAM_ZERO=1` | **2581:78.4%, 671:4.5%**, 50249:2.7%, 43:1.2%, 16:0.8% | **1.556** |
| 参考 `strata-dsv41` | 271:9.6%, 943:9.4%, 28:6.3% | ~6.0 |

**两个事实：**

1. **engram 零注入让 `671` 出现了** —— 而 `671` 正是参考引擎生成序列里紧跟 `271` 的那个 token，
   也正是 prompt 中间那段的首个 token。**说明 engram 注入正在把结果推离参考方向。**
2. **但零注入同时把熵从 6.41 压到 1.56**（单 token 78%），这同样是错误的 ——
   **不是"要不要 engram"，而是"engram 注入了什么"。**

`INFR_NO_QKV_FUSE` 与 `INFR_NO_GATED_RMSNORM` 的结果**与基线逐位相同**，这不是证据：
前者代码里明确写了 split 形式是 bit-identical（同点积、同固定顺序求和），本来就不该有差异。

### IQ4_NL 表：确认损坏，但**不能因此断定 mxfp8 是错的**

模型 GGUF 里写着：

```
deepseek41.engram.file = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
```

而本机那张 `engram-iq4nl.gguf`（17.9 GB）加载时报：

```
'blk.1.engram_embd.weight' occupies bytes 224..55296888416, past the 17901748448 bytes of the file
```

**文件被截断**（声称 55.3 GB，实际 17.9 GB），不可用。

**但我不能仅凭元数据里一个文件名就断定 `engram-mxfp8.gguf` 用错了** ——
它就在模型目录 `/data/models/dsv41-uncensored-iq2xxs/` 内，202 GB，体量与全精度 engram 相符，
而 17.9 GB 的 IQ4_NL 明显更小且已损坏。元数据里的名字更可能只是原始发布名。

### 下一步

需要确认 `engram-mxfp8.gguf` 是否真的是这个 checkpoint 的配套表：
比对它内部张量与 GGUF 里 `deepseek41.engram.*` 声明的形状/层号是否一致。
这比继续换表更有意义 —— 一个能加载但内容不匹配的表，正是"语言能力完好、语义全错"的完美解释。

