

---

## Round 102：找到根本性设计差异 —— 传进图的应该是 I32 行下标，不是 F32 值

### 参考实现的做法（`deepseek41.cpp:351-378`）

```cpp
inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols * n_tokens);
ggml_tensor * emb = ggml_get_rows(ctx0, model.layers[il].engram_embd, rows);
emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
```

**传进图的是 `I32` 行下标；`ggml_get_rows` 在【图内】完成 gather。**

### 我们的实现

```
engram_emb_buf = g.input(f32d(batch * n_cols * key_length))   // F32 值
```

**host 侧 gather 出 F32 浮点，再上传。**

### 这就是 `gather_for_layer` 至今零调用的真正原因

**不是忘了调用，是方向错了。** 我写的 `gather_for_layer` 返回 `Vec<f32>`（已解量化），
而架构上应该传的是 **`Vec<i32>` 行下标**。

### 而且这解释了 202 GB 表为什么能工作

`engram_embd` 在**模型权重里**（`model.layers[il].engram_embd`），由图内的 `get_rows` 读取。
**GPU 直接按行下标取行**，不需要 host 搬运。

我先前把 `EngramTable`（那个 202 GB 的 mmap 文件）当成了必须 host 侧读取的数据源——
**但它其实是 GPU 侧的权重表，参考实现根本不在 host 侧 gather。**

### 正确方向

| 项 | 错误做法（当前） | 正确做法 |
|---|---|---|
| 传给图的内容 | `F32` 已解量化的值 | **`I32` 行下标** |
| gather 位置 | host | **图内 `get_rows`** |
| buffer 大小 | `batch*n_cols*key_length*4` | **`batch*n_cols*4`** |
| 表读取者 | host mmap | **GPU 权重** |

### 两层是否共用 buffer

参考实现**每层创建独立的 input**（`build_inp_engram(model, il)` 每层调用一次），
但每层传的是**自己那层的行下标**。

**在 infra 的单一 input 张量设计下，两层不能共用同一步的数据**——
要么按层拆分 input，要么每层 forward 前分别上传。

**参考实现选择了按层独立**，这与我上一轮的猜测一致。

### 下一步

1. 把 `engram_emb` 的声明从 `f32d(...)` 改成 `i32d(batch * n_cols)`
2. host 侧只算**行下标**（`EngramMap::row_indices` 已有），不 dequant
3. 图内加 `Op::GetRows`（若已有则复用）取 `engram_embd`
4. `gather_for_layer` 改造成只返回下标

**这是一个不小的改动，但它是对的方向。**

