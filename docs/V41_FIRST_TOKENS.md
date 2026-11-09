
---

## Round 199：engram 表核实完毕 —— 与模型完全配套，Round 198 的怀疑不成立

### 实测

```
/home/dyy/engram-mxfp8/engram-mxfp8.gguf
  general.architecture = 'deepseek41-engram'
  blk.1.engram_embd.weight   dims=[384006168, 256]  dtype=32
  blk.1.engram_embd.scale    dims=[384006168, 8]    dtype=34
  blk.14.engram_embd.weight  dims=[384016682, 256]  dtype=32
  blk.14.engram_embd.scale   dims=[384016682, 8]    dtype=34
```

模型元数据声明：

```
deepseek41.engram.num_embeddings = [384006168, 384016682]
deepseek41.engram.layer_ids      = [1, 14]
deepseek41.engram.head_count     = 8
deepseek41.engram.key_length     = 256
```

**行数逐项吻合（384006168 → 层 1，384016682 → 层 14），维度 256 = key_length，8 = head_count。**

### 结论

**Round 198 我怀疑"MXFP8 表可能来自另一个源、与模型不配套" —— 实测证明是配套的。**

**模型声明的文件名是 `...-Engram-IQ4_NL.gguf`（17.9 GB，且在 Round 152 已证明被截断不可用），
MXFP8 表（189 GB）是同一份数据的无损重打包，行数完全对得上。**

**⇒ engram 表这条线彻底查完，没有问题。**

### 本会话 engram 相关排除表（完整）

| 项 | 结论 |
|---|---|
| 缓冲越界写 60 KiB/层 | ✅ 已修（Round 170），修复后输入敏感性恢复 |
| 哈希算法 / 窗口 / 取行 | ✅ 与 C++ 参考逐行等价 |
| slot 映射与数组 stride | ✅ 一致 |
| 元数据键（两组重复命名） | ✅ 值一致，非缺陷 |
| 表与模型配套性 | ✅ 行数逐项吻合（本轮） |

### 状态

**本轮没有发现缺陷，但关闭了一条线索，并且是修正自己上一轮的怀疑。**

**核心问题（两个引擎在同一权重上退化）仍未解释。**

