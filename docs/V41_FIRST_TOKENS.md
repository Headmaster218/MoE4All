

---

## Round 103：**推翻 Round 102 的结论** —— host gather 就是这个项目的设计

### 我上一轮错在哪

Round 102 我对照 llama.cpp 的 `deepseek41.cpp` 得出「传进图的应该是 I32 行下标」，
并断言我们的 `f32d(...)` 声明是错的。

**我没查这个项目自己的设计说明。**

### 铁证（`runner.rs:2794-2798`）

V4.1 engram 权重加载处的注释：

> *"The 32 GiB `engram_embd` table is **NOT an uploaded weight**: its rows are
> **gathered on the host** from the mmap (`crate::engram`) and **arrive as a graph Input**,
> so it never appears here."*

**这个项目刻意选择 host 侧 gather。** 理由显然：202 GB 的表不能进显存。

**所以：**
- `f32d(batch * n_cols * key_length)` 声明 —— **对的**
- `gather_for_layer` 返回 `Vec<f32>` —— **对的**
- 不需要 `Op::GetRows`，不需要改 I32

### 真正缺的仍然是那三件事

Round 101 的结论依然成立：

| 项 | 状态 |
|---|---|
| `EngramTable::open` | 主流程零调用 |
| `gather_for_layer` | 零调用点 |
| `be.upload(engram_emb_buf)` | 不存在 |

**只是「调用它们并上传」，不是「重新设计数据格式」。**

### 教训

**在判定「实现错了」之前，先查本项目自己的设计注释，不要只对照上游。**

上游是 llama.cpp（ggml 图），本项目是自研 Vulkan 引擎——架构取舍本来就可以不同，
而且这个项目**在代码里写明了理由**。

**Round 102 差点让我把一个正确的设计改错**，那会是几轮的浪费。

