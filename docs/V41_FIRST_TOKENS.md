

---

## Round 144：**参考基准终于有了** —— 本机的 `strata-dsv41` 用同一份 GGUF，还能 `--dump-logits`

### 前面十几轮为什么卡住

一直缺「同一份模型文件 + 能正确运行的引擎 + 可导出的中间值」这三样同时具备的参照：

| 候选 | 为什么不行 |
|---|---|
| llama.cpp（`build-sycl/bin/llama-cli`） | 报 `blk.1.engram_embd.weight is missing` —— 这个 GGUF 不含 engram 张量，只有元数据 |
| DeepInfer | Python 实现，130 GB 模型跑不动 |
| Mach X（`auto-test-result.txt`） | FP8 safetensors，量化格式与代码路径都不同 |
| qwen2.5 | 不同架构，只能证明引擎通用路径健康 |

### 找到了：`/data/strata-new/build-sycl-jit/strata-dsv41`

线索来自 `/data/models/dsv41-i8test/bench/bench-ab.sh`：

```bash
ENGINE=/data/strata-new/build-sycl-jit/strata-dsv41
PACK=/data/models/dsv41-i8test/pack-q2k-all     # 指向同一份 IQ2_XXS
```

需要 `source /opt/intel/oneapi/setvars.sh`（否则缺 `libmkl_sycl_blas.so.6`）。

### 实测：跑通了，且支持 `--dump-logits`

```
prompt: 989 tokens in 28234 ms (35.03 tok/s, windows of 16)
455,4787,13769,46012,54994,1060,270,41638,6397,14,305,270
decode: 12 tokens in 631 ms (17.42 tok/s)
```

`/tmp/ref-logits.bin` 511 MB —— **逐 token 的 logits 参考值**。

### 这改变了什么

从现在起可以做**真正的数值比对**，而不是靠「范数看起来异常」这种间接判断：

1. 拿同一段 prompt token ids 跑 moe4all
2. 比对两者在**第一个 token 的 logits**
3. 若第一个 token 就分叉 ⇒ 单次前向就有错，可以逐层二分
4. 若前 N 个 token 一致后才分叉 ⇒ 错误在增量/缓存路径

### 参考输出（已存档）

```
455,4787,13769,46012,54994,1060,270,41638,6397,14,305,270
```

同一份 `--ids @/data/models/dsv41-iq2xxs-pack-real/bench/1k.ids`（989 tokens）。

