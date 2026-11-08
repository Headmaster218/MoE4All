

---

## Round 86：engram 注入的五点改动清单

### 现成的范例：`ple_embd`

宿主侧数据进入 graph 的机制已经存在，`ple_embd` 就是范例，三步：

```rust
// 1. handle（builder 的结构体里声明）
ple_embd: Option<TensorId>,
// 2. alloc
let ple_embd_buf = Some(be.alloc(..., BufferUsage::Activations)?);
// 3. bind
if let (Some(id), Some(buf)) = (h.ple_embd, ple_embd_buf) { b.bind(id, buf.as_ref()); }
```

### engram 要动的五处

| # | 位置 | 内容 |
|---|---|---|
| 1 | builder handle 结构体 | 加 `engram_emb: Option<TensorId>` |
| 2 | `runner.rs:1732` 旁 | `alloc(n_cols(24) * key_length(256) * nt * 4)` |
| 3 | `runner.rs:1030` 旁 | `bind(h.engram_emb, buf)` |
| 4 | 每步 forward 前 | 调 `crate::engram` 算 rows、gather、`dequant_mxfp8_row` |
| 5 | `runner.rs:6300` 后 | `Linear(wkv)` → 切 key/value → `RmsNorm` + `engram_k/q` → `QwenPleGate` → 累加进 `hcr[0]` |

### 数据量

| 场景 | 每步传输 |
|---|---|
| decode `nt=1` | **24 KiB** |
| prefill `nt=256` | **6.0 MiB** |

**decode 的 24 KiB 可忽略；prefill 的 6 MiB 会明显拖慢**（当前 prefill 27 tok/s，
按流式带宽算每步多传 6 MiB 需要额外时间）。

### 但有一个前提必须先确认

**engram 表是独立文件（189 GB），运行时需要 mmap 或 open+seek。**

问题：`run` 命令目前只接收主 gguf 路径，**没有 engram 表路径的参数**。
`engram.file = DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` 是元数据里的**相对文件名**，
与 189 GB 的 `engram-mxfp8.gguf` 不一致。

**所以第一件事是确认 engram 表如何定位**：
- 是否已有 CLI 参数 / 环境变量（`INFR_ENGRAM_TABLE` 在 `engram.rs` 的测试里出现过）
- 还是需要在主 gguf 同目录按约定查找

**这个不确定的话，改动 1-5 写完了也跑不起来。**

### 下一步

先查 `INFR_ENGRAM_TABLE` 的实际含义和加载路径，确认表如何打开。
这是编译前必须打通的最后一环。

