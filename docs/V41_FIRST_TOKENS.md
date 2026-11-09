

---

## Round 143：配置全项核对通过（加了 `INFR_CFG_DUMP`），配置类错误排除

### 背景

`/data/models/dsv41-i8test/pack/ds4.txt` 是一份**权威的模型配置快照**（45 行，与 `deepseek4.cpp`
同源），比逐个字段读代码可靠。于是加 `INFR_CFG_DUMP`，把引擎解析出来的配置按同样的键名打出来，
两边直接 diff。

### 核对结果：**没有一项错**

| 项 | 权威 | 引擎 |
|---|---|---|
| vocab_size | 129280 | 129280 |
| hidden_size | 5120 | 5120 |
| num_hidden_layers | 40 | 40 |
| head_dim | 512 | 512 |
| qk_rope_head_dim | 64 | 64 |
| hc_mult | 4 | 4 |
| hc_sinkhorn_iters | 20 | 20 |
| index_head_dim | 128 | 128 |
| index_topk | 512 | 512 |
| engram_layer_ids | 1,14 | [1, 14] |
| engram_n_head / max_ngram / key_length | 8 / 4 / 256 | 8 / 4 / 256 |
| n_routed_experts | 384 | 384 |
| num_experts_per_tok | 6 | 6 |
| moe_intermediate_size | 2304 | 2304 |
| routed_scaling_factor | 1.5 | 1.5 |
| rms_norm_eps | 9.9999997e-21 | 1e-20 |
| rope_theta / compress_rope_theta | 10000 / 160000 | 10000 / 160000 |
| rope_scaling.factor | 16.0 | 16 |
| sliding_window | 128 | 128 |
| swiglu_limit | 10.0 | 10.0 ×40 |
| compress_ratios | 43 项，roles `0,0,2*18,1*20` | 相同（引擎裁到 40） |
| kv_source_layer_ids | 2,8,14,20 | 展开图与之一致 |
| index_source_layer_ids | 2,8,14,20,24,28,32,36 | 8 个源，是 KV 源的超集 ✓ |

两处曾被我误判为错，都是**我自己 dump 错了字段**：
`rms_norm_eps` 我一开始打了 `norm_eps`（那是非 RMS LayerNorm 的，V4 不用），
`rope_scaling_factor` 我一开始漏打了。补上后两者都正确。

`rope_yarn_log_mul = 0` 也不是问题：GGUF 里确实没有 `yarn_log_multiplier` 这个键，
而 DeepSeek 的 mscale 由 `rope.scaling.factor`（已正确读为 16）决定，频率缩放走 factor。
在这台机器上只有 512 token 的上下文，YaRN 的 ramp 根本没被触发。

### 结论

**配置类错误排除。** 缺陷只能在计算图 / kernel 的数值实现里。

`INFR_CFG_DUMP` 留着：以后再遇到「怀疑配置读错」，一条命令就能把 40 项打全，不必逐字段读代码。

