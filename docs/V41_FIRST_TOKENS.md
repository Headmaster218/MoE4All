

---

## Round 73（续）：纠正 —— `engram-iq4nl.gguf` 不是模型

上一节提到"用 `dsv41-engram-iq4nl` 做 IQ4_NL 对照"。**这个说法是错的。**

核实文件头：**只有 2 个张量、1 个元数据项**。

`engram.rs:38` 的文档写明：`deepseek41-engram` 是 **engram 表**的架构名，
布局是每层两个张量（`engram_embd.weight` + `engram_embd.scale`），
是主模型旁边的**辅助表**，不是可推理的模型。

**所以 `dsv41-engram-iq4nl/engram-iq4nl.gguf` 是一个 14 GB 的 engram 表，
不是模型。** 拿它当对照是我读错了目录名。

### 本机没有非 IQ1_M 的 V4.1 模型

盘点 `/data/models/dsv41-*`：

| 目录 | 内容 |
|---|---|
| `dsv41-uncensored-iq2xxs` | **真正的模型**（4 分片，含 IQ1_M 专家） |
| `dsv41-engram-iq4nl` | engram 表（2 张量），非模型 |
| `dsv41-engram-stub` | engram 桩（4 KB） |
| `dsv41-fp8-*` / `dsv41-i8test` | 只有 5 MB 的 lora，无模型 |

**要绕过 IQ1_M 做对照，需要另外下载一个非 IQ1_M 的 V4.1 量化版本。**

### 但对照实验的价值不受影响

`qwen2.5-1.5b` → `2+2=4` 这个结论仍然成立且重要：
**引擎正确，只有 V4.1 错。** 只是"V4.1 内部哪个组件错"还定位不了。

### 当前的准确定位

```
引擎（Vulkan/采样/template/CLI）   ✅ 由 qwen2.5-1.5b 证明
V4.1 forward                        ❌ 输出错误
  ├─ IQ1_M 专家 dequant             ?  
  ├─ 34 复用层 source-stream        ?
  ├─ engram / dspark                ?
  └─ hc_mult 残差                   ?
```

**四个候选，无法在本机用现有模型二分。**

