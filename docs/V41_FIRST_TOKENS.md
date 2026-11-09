
---

## Round 200：主 GGUF + 独立 engram 表的设计与索引，全部自洽

### 主分片的张量清单

```
shard1: tensors=42, 其中 engram 相关 4 个
  engram_token_map.weight    [129280]   I16
  engram_primes.weight       [24, 2]    I32
  engram_offsets.weight      [24, 2]    I32
  engram_multipliers.weight  [4, 2]     I32
```

**主 GGUF 只有**索引元数据**，189 GB 的 embedding 表在独立文件里。`bench-iq2.log` 里
`blk.1.engram_embd.weight is missing` 正是这个设计的结果——单独加载主文件必然缺表。**

### 元数据数组与代码期望的长度

```
deepseek41.engram.primes      len=48   = 2 slots × 24 buckets
deepseek41.engram.offsets     len=48
deepseek41.engram.multipliers len=8    = 2 slots × 4 ngram
```

`n_bucket = (4-1) × 8 = 24`，与 `primes` 的第一维 `[24, 2]` 一致。

### 数值结构自洽

```
primes[0..3]  = [16000057, 16000079, 16000081]
offsets[0..3] = [0, 16000057, 32000136]

offsets 每段跳 ~1600 万 = engram_vocab_size
num_embeddings[0] = 384006168 = 24 × 16000257 ≈ 24 个 bucket × 1600 万行
```

**每个 bucket 占 1600 万行，24 个 bucket 共 3.84 亿行 —— 与表的第一维完全对上。**

### 结论

**engram 的索引设计（token_map / primes / offsets / multipliers）、行数分配、
元数据与外部表的一致性，逐项自洽。**

**⇒ engram 不是缺陷来源。这是本会话第 4 次独立确认。**

### 状态

**核心问题（两个引擎在同一权重上退化）仍未解释。**

**已排除的完整清单（含本轮）：**
缓冲越界（已修一条）· 哈希算法 · 窗口顺序 · slot 映射 · 元数据键冲突 ·
表与模型配套性 · 索引数组长度与数值结构 · 专家量化（MXFP4 局部替换无效）·
prompt 三种格式 · chat template · 注意力结构 · 输出投影 · 配置键

