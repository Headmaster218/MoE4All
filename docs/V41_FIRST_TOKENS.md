
---

## Round 156：engram 表核对通过 —— 这条线走完了，正式排除

把 `engram-mxfp8.gguf` 打开对比：

| 检查项 | 结果 |
|---|---|
| 表内张量 | `blk.1.engram_embd.weight/.scale`、`blk.14.engram_embd.weight/.scale` |
| GGUF 声明 `engram_layer_ids` | `1, 14` ✅ |
| 引擎查找名 | `"blk.{layer_id}.engram_embd.weight"` ✅ |
| 格式 | ggml type 32 权重 / type 34 缩放，引擎只支持这两种 ✅ |
| 体量 | 202.76 GB，2 层全精度 ✅ |

**四项全对。`engram-mxfp8.gguf` 就是这个 checkpoint 的配套表。**

元数据里的 `deepseek41.engram.file = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"` 只是**原始发布名**，
本机那份同名 IQ4_NL 文件（17.9 GB）是损坏的副本，与当前用的是否为同一张表无关。

### 那 Round 155 的 `ENGRAM_ZERO` 现象怎么解释

**说明 engram 确实在参与计算**，关掉必然改变输出。
它"露出"了 `671`（参考也有的 token）**只是巧合**，不是"更接近正确" ——
同时熵从 6.411 塌到 1.556（单 token 78%），这明显是错的。

### 累计已排除（全部有运行证据）

引擎通用路径 · prompt 构造 · YOCO · APE · warmup · 超连接滞后 · 配置 24 项 ·
采样参数 · logits 尺度/形状 · 批量 prefill（已定位+守卫）· **engram 表本身** ·
QKV 融合/拆分（设计上逐位等价）

**剩下未排除的主要嫌疑：压缩注意力（compressed attention / kv_source 复用）、hash MoE 路由。**

