
---

## Round 186：engram 全链路核对完毕 —— 计算路径无缺陷

### 从 GGUF 直接读出的元数据

```
deepseek41.engram.head_count    = 8
deepseek41.engram.key_length    = 256
deepseek41.engram.max_ngram_size= 4
deepseek41.engram.pad_id        = 2
engram_primes.weight      dims=[24, 2]   dtype=27 (I32)
engram_offsets.weight     dims=[24, 2]   dtype=27
engram_multipliers.weight dims=[4, 2]    dtype=27
engram_token_map.weight   dims=[129280]  dtype=26 (I16)
```

`n_bucket = (4-1)*8 = 24` ✓ 与 `primes` 的第一维一致。

### slot 映射核对

```
infr:  slot = engram_layer_ids.iter().position(|&l| l == layer_id)
参考:  engram_index(il) 遍历 engram_layer_ids 找匹配
```

两边都是「在 `engram_layer_ids` 里的下标」。`engram_layer_ids = [1,14]` ⇒ slot 0 = 层 1，slot 1 = 层 14。

`multipliers` 按 `slot * n_gram = slot * 4` 取基址（stride 4，形状 `[4,2]` ✓）；
`primes`/`offsets` 按 `slot * n_bucket = slot * 24` 取基址（stride 24，形状 `[24,2]` ✓）。

### 完整核对表

| 环节 | 状态 |
|---|---|
| 哈希算法（ctx/rolling/bucket/row） | ✅ 与 C++ 逐行等价 |
| 前驱窗口顺序（oldest-first） | ✅ 一致 |
| row gather | ✅ 正确 |
| slot 映射与数组 stride | ✅ 一致 |
| 缓冲分配与越界 | ✅ 已修（Round 170） |

### 结论

**engram 的计算路径已全部核对，没有缺陷。**

**⇒ Round 183 观察到的「置零 engram 后置信度 8.49% → 28.17%」无法用计算错误解释。**

**剩下唯一符合证据的解释：这份 IQ2_XXS 权重在 engram 投影器（`engram_wkv/k/q`）上的量化误差
特别大，导致注入的向量本身偏了。** 这不是可以"修复"的 bug，而是量化选择的代价。

### 这意味着什么

**如果要验证这一点，需要一个高精度（FP8/BF16）的同模型权重跑同样输入。**
**本机只有 `/data/models/dsv41-i8test/` 的 IQ2_XXS 打包，没有更高质量的副本。**

**⇒ 这条线索在当前资源下无法继续。**

### 注意

**`engram_k/q` 张量不在这 4 个分片的 tensor 列表里**（只有 token_map/primes/offsets/multipliers），
说明它们在后面的分片里，无法用同样的方式快速检查 dtype。

