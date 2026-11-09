

---

## Round 126：IQ4_NL 表的结构已探明 —— 与 MXFP8 是**转置关系**

### 两个表的实际张量声明

| 表 | `blk.1.engram_embd.weight` shape | ggml type | 字节 |
|---|---|---|---|
| **MXFP8**（此前一直用） | `[384006168, 256]` | 32 (E4M3) | 98.3 GB |
| **IQ4_NL**（模型声明） | `[256, 384006168]` | 20 (IQ4_NL) | 55.3 GB |

**行数完全相同（384006168 = 两层合计），但 `ne0`/`ne1` 位置对调。**

GGUF 约定 `ne0` 是最快变化维：

- **MXFP8**：`ne0 = rows(384006168)`，`ne1 = 256` —— 行是外层，每行 256 个 E4M3 码，
  另有独立 `blk.N.engram_embd.scale` 张量 `[384006168, 8]`（E8M0，8 个指数覆盖 256 元素）。
- **IQ4_NL**：`ne0 = 256`，`ne1 = rows(384006168)` —— 标准 ggml 声明，
  每「行」是一个 256 元素的 embedding，**scale 内嵌在块里**（每 32 元素一个 f16 d）。

**⇒ 这不只是量化格式不同，是两种张量布局约定。** 用 MXFP8 的读法去读 IQ4_NL 表，
或者反过来，读到的都是完全错位的数据。

### MXFP8 表的每层行数

```
blk.1.engram_embd.weight   [384006168, 256] type 32
blk.1.engram_embd.scale    [384006168, 8]   type 34
blk.14.engram_embd.weight  [384016682, 256] type 32
blk.14.engram_embd.scale   [384016682, 8]   type 34
```

两层行数略不同（384006168 / 384016682），与 `engram_vocab_size` 的分配方案相符。

### 需要做的事

1. `ggml_type_layout(20)` → `(18, 32)`：32 元素一块、每块 18 字节（**已加**）。
2. 行/列索引要按表类型分支：type 20 用 `shape[1]` 作行数、`shape[0]` 作 key_length；
   type 32 相反。
3. 新增 `dequant_iq4_nl_row`：每 32 元素读 2 字节 f16 d + 16 字节 4-bit code，
   查 16 项 codebook。
4. type 20 没有独立 scale 张量，`tensor_pair` 要能返回单张量。

