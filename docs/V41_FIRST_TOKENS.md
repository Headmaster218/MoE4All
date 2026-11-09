

---

## Round 125：**根因找到 —— engram 表格式不匹配，模型声明的表我们从未加载过**

### 决定性证据链

**① 模型自己声明了它需要哪个表**

直接读 GGUF 头（`deepseek41.engram.file`，gguf type 8 = string）：

```
deepseek41.engram.file = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
```

**② 本机确实有那个表**

```
/data/models/dsv41-engram-iq4nl/engram-iq4nl.gguf   17.9 GB
/home/dyy/engram-mxfp8/engram-mxfp8.gguf           202.8 GB   ← 我们一直在用这个
```

**③ 换成正确的表，引擎直接拒绝**

```
Error: unsupported: engram table tensor has ggml type 20;
this reader decodes only type 32 (E4M3 codes) and 34 (E8M0 scales)
```

**④ 代码里的注释印证**

`engram.rs:44-45`：

```
blk.N.engram_embd.weight  ggml type 32  [rows, 256]  E4M3 codes
blk.N.engram_embd.scale   ggml type 34  [rows, 8]    E8M0 exponents
```

`engram.rs:100-102` 明确写着这是「**this converter's private E4M3 code-stream**」——
即某个私有转换器的产物，**不是**模型公布的标准 IQ4_NL 格式。

### 结论

| 项 | 值 |
|---|---|
| 模型声明的表 | **IQ4_NL，ggml type 20，17.9 GB** |
| `engram.rs` 支持的 | **MXFP8 私有流，type 32/34，202 GB** |
| 此前所有测试用的 | MXFP8 |
| **后果** | **engram 两层读到完全错误的数据 ⇒ 答非所问** |

**这解释了为什么「模型收到正确的 `2+2=` 却答错」——错不在 prompt、不在 YOCO、不在注意力，
而在 engram 层用了错的表。**

### 为什么之前所有 engram 验证都「通过」了

我逐个核对过数学（gate 公式、偏移、广播）、数据（数值 O(1)、无 NaN、两层不同）——
**这些都对，但它们验证的是「代码与 MXFP8 表一致」，而不是「MXFP8 表是本模型该用的表」。**

数学正确 + 数据源错误 = 结果错误。这是一个我没有设想过失败方式。

### 下一步

**必须给 `engram.rs` 加 IQ4_NL（ggml type 20）解码支持**，让它能读模型声明的那个表。
标准 IQ4_NL 是 32 元素一块、每块 18 字节（2 字节 f16 scale + 16 字节 4-bit code），
比 MXFP8 简单得多。

