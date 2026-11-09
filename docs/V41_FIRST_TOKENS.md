

---

## Round 101：找到数据流的断点 —— engram 表从未在主流程加载过

### 问题

`engram_emb_buf` 已分配、已绑定，但**内容从未被写入**，所以 gate 跑在垃圾数据上。

### 追查结果

| 环节 | 状态 |
|---|---|
| `EngramTable::open`（mmap 202 GB 表） | **只有测试调用，主流程 0 次** |
| `engram_map_for_layer` | 只在测试里 |
| `gather_for_layer`（我写的） | **0 个调用点** |
| `engram_emb_buf` 的 `be.upload` | **不存在** |

**整条 host 侧数据流一次都没接上过。**

### 表文件确认存在

```
/home/dyy/engram-mxfp8/engram-mxfp8.gguf              202,758,032,752 字节
/data/models/dsv41-uncensored-iq2xxs/engram-mxfp8.gguf  同目录副本
```

**`EngramTable::open` 只做 mmap + `advise(Random)`，不读入内存**——
注释明确写着：*"Rows are gathered in hash order, which has no locality. This is the opposite call
from the weight mmap's `WantNeed`: here, populating the mapping would fault in 202 GB."*

**所以接入它没有内存代价，是安全的。**

### 接入点还没定

decode 的 `b1` 构造在 qwen4exp 分支内（`ple_buf` 那句 `qwen4exp session has no PLE staging buffer`
是硬性要求），V4.1 不走那里。

需要找到 prefill 和 decode 的**最外层公共入口**，在那里：
1. 按层算出 `tokens` 与 `prev` 窗口
2. `gather_for_layer` → `Vec<f32>`
3. `be.upload(engram_emb_buf, ...)`

### 注意：engram 有 2 层，但 buffer 只有 1 个

`engram_layer_ids = [1, 14]`，而 `engram_emb_buf` 是单个 buffer。
若两层共用，必须在**每层 forward 之前**分别上传——这不是一次上传能解决的。

**这一点需要先确认设计意图**：语言模型的两层 engram 是否真的共享同一个 input 张量。
如果不是，`engram_emb` 需要按层拆分。

