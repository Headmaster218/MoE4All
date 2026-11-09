
---

## Round 196：撤回 Round 195 —— "bad gguf" 不是权重损坏，是老版加载器不支持

### 我上一轮错在哪

Round 195 我读到 `auto-test-result.txt` 里的 `RESULT: bad gguf`，就宣布根因是"权重文件损坏"。

**但同一个目录里还有一个 `auto-fix-test.log`，日志写得很清楚：**

```
E llama_model_load: error loading model: error loading model hyperparameters:
    key not found in model: deepseek41.engram.head_count
E llama_server: exiting due to model loading error
```

**⇒ 失败原因是**那个 llama.cpp 版本不认识 `deepseek41.engram.head_count` 这个键**，
不是权重内容损坏。**

**反证：`infr` 能正常读同一个文件，并且打印出 `deepseek41.engram.head_count = 8`。**

### 所以结论撤回

**权重没有"损坏"的证据。Round 195 的结论作废。**

**正确的表述是：那个测试记录反映的是 **llama.cpp 版本兼容问题**，不反映权重质量。**

### 第 15 次同类错误

**看到 `RESULT: bad gguf` 的字面就下结论，没读同目录的日志。**

**而这些日志就在旁边。这和第 195 轮"没读 auto-test-result"是同一个错误的镜像版本。**

### 现在的真实状态

| 项 | 状态 |
|---|---|
| 权重是否损坏 | **未知**（没有证据） |
| llama.cpp 测试失败原因 | 版本不支持新键（已确认） |
| 两个引擎在同一权重上退化 | 事实，但原因未明 |
| `RESULT: bad gguf` | **不是权重损坏的证据** |

**⇒ 目标仍未达成。上一轮的"根因"是错的。**

