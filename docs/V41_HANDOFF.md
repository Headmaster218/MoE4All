# DeepSeek-V4.1 on moe4all(Vulkan) — 答错问题交接

**状态：未解决。** 本文档记录已验证的事实、可复用的工具、以及下一步该做什么。

---

## 一分钟摘要

同一个 GGUF、同样的输入 token，两个引擎预测出**完全不同的下一个 token**：

| 输入 | 参考 `strata-dsv41` | moe4all |
|---|---|---|
| `Hi`（5 token） | `304:63.6%` | `2619:6.3%` |
| `What are the first five primes?` | `271:9.6%, 943:9.4%` | `666:6.7%, 2619:3.4%` |

**置信度差一个量级**（63.6% vs 6.3%）。这是「信息丢失」而不是「数值放大」。

**在只有 5 个 token 的输入上就已经分叉** —— 所有缓存类、累积类机制都被排除。

---

## 关键：参考基准就在本机

之前十几轮一直以为没有参照。实际上有：

```bash
source /opt/intel/oneapi/setvars.sh      # 否则缺 libmkl_sycl_blas.so.6

/data/strata-new/build-sycl-jit/strata-dsv41 \
  --pack /data/models/dsv41-i8test/pack-q2k-all \
  --gpus 0 --experts 6 --dspark off --vision off --threads on \
  --vram-experts 0.3 --mmap-experts auto --ssd-cache 0 \
  --n 4 --prefill-chunk 8192 \
  --ids "0,128803,23166,128804,128821" \
  --dump-logits /tmp/ref.bin
```

`/data/models/dsv41-i8test/pack-q2k-all/` 是一组指向**同一个 IQ2_XXS** 的软链接。

**已知限制**：`--serve` 模式会 core dump；没有逐层张量导出，只有 `--dump-logits`。

---

## 输入对齐方法（这是关键的一步）

两边输入格式不同（参考要裸 ids，moe4all 自己套 chat template），所以必须先对齐：

```bash
# 1. 让 moe4all 打印它实际喂给模型的 prompt 字符串
INFR_PROMPT_DUMP=1 ./target/release/infr run "$MODEL" "Hi" --max-new 1 --dev Vulkan1
#    -> /tmp/turn-prompt.txt

# 2. 用同一个 tokenizer 编码
./llama-cpp-dsv41/build-cpu/bin/llama-tokenize \
  -m "$MODEL" -f /tmp/turn-prompt.txt --ids

# 3. 把结果喂给参考引擎（见上面的 --ids）
```

---

## 可用的诊断工具（全部已提交）

| 环境变量 | 作用 |
|---|---|
| `INFR_CFG_DUMP=1` | 打印引擎解析出的全部配置（与 `ds4.txt` 同键名，可 diff） |
| `INFR_LOGIT_PROBE=1` | 每步打印 top5 与熵 |
| `INFR_DUMP_LOGITS=<path>` | 把完整 logits 行写成 raw f32，**与参考 `--dump-logits` 格式逐字节一致** |
| `INFR_PROMPT_DUMP=1` | 把喂给模型的精确 prompt 字符串写到 `/tmp/turn-prompt.txt` |
| `INFR_ENGRAM_ZERO=1` | 跳过 engram 注入（真零对照） |
| `INFR_HASH_ID_PROBE=1` | hash 路由 token id 探针（本 checkpoint 不会触发，原因见代码注释） |
| `INFR_V4_BATCH_CHECK` | 已移除：单行守卫现在是 release 常驻 `assert!` |

**logits 文件格式**：`129280 × 4` 字节一行，一行一个位置，无文件头。

---

## 已排除（全部有运行证据）

| 项 | 证据 |
|---|---|
| 引擎通用路径 | qwen2.5-0.5b 在同一二进制/设备上正常 |
| prompt 构造 | `PROMPT_DUMP` 显示措辞正确 |
| **配置 24 项** | `INFR_CFG_DUMP` 与 `ds4.txt` 逐项 diff，无差异 |
| YOCO / n_prefill_layer | 默认 off，只影响可见域 |
| APE | V4.1 无该张量，全零表等价 |
| warmup | `INFR_NO_WARMUP` A/B 输出几乎一致 |
| **超连接滞后** | `INFR_HC_LAG_CHECK` release 断言通过 |
| **engram 表** | 表内层号 `1,14` 与声明一致、张量名一致、格式一致、202 GB 全精度 |
| **logits 尺度** | 对齐行 std 比值 **1.161**，不是 1.5 ⇒ `routed_scaling_factor` 未重复应用 |
| **采样参数** | temp 0/1 两条路径都错 |
| **hash 路由** | `hash_layer_count = 0`，本文件根本没有 |
| **长序列累积** | 5/31/121 字符三档熵 6.34/6.41/6.46，无单调恶化 |
| **压缩状态累积** | 5 token 输入上就分叉，压缩机制来不及生效 |
| QKV 融合/拆分 | 设计上逐位等价 |

## 已定位并修复的**另一个** bug

`Dsv4CacheWrite` 在 V4 注意力路径上硬编码 `rows: 1` + 单个 `row`，而 `rope_tail` 也只旋转第 1 行。
批量 prefill（`INFR_DSV41_BATCHED_PREFILL`）会走进这里，**静默写坏 KV**。

- 那个 4.9× 的 prefill 提速因此**数值不可信**，已改回 opt-in
- 守卫从 `debug_assert` 改成 **release 常驻 `assert!`**，现在任何 batch>1 都会立刻失败并说明原因
- **这不是默认路径答错的原因**（默认路径 batch == 1，守卫通过）

---

## 下一步（按性价比排序）

### 1. 逐层残差流观测 —— 最直接

在 `runner.rs` 每层构造处（`hcr[0]` 是层输入的残差流）加一个把范数写到文件的 probe。
5 token 的输入下计算量极小，且不引入任何缓存干扰。

**看什么**：残差流范数是否在某一层突变、是否单调增长、层间增量是否与原版 DeepSeek 的典型量级相符
（同架构 V3/V2 的每层增量通常在个位数百分比）。

### 2. 检查 attn 输出量级

V4 的注意力是 `build_attn_mha(q, k_all, k_all, …)`（V 就是 K）。如果 `k_all` 的归一化
（`k_norm`）没生效，attention 输出会整体偏小/偏大，症状正是「信息传不过去」。

### 3. 与 llama.cpp 的 dsv41 分支对照源码

本机有 `llama-cpp-dsv41/` 源码树。它跑不起来（缺 engram 张量），但**源码可读**。
重点对照 `deepseek4.cpp` 里 attention 与 hyper-connection 的部分。

---

## 方法教训（这个会话里交的学费）

1. **先测对方自己的抖动，再判断差异是否有意义。** 参考引擎两次运行 max|Δlogit|=5.70、会换位，
   所以只能做粗粒度比对。
2. **一次 dump 可能有多行，确认哪一行对应目标位置。** 我拿 warmup 行当了生成行，浪费了好几轮。
3. **加探针前确认代码路径有被执行的结构前提**（如 `hash_layer_count > 0`），不能只确认能编译。
4. **`debug_assert` 在 release 里是死的**，而 release 是唯一能跑这个模型的构建。要证据就得用 `assert!`。
5. **比较前必须对齐输入**。两个引擎的输入格式不同，这是十几轮无效对比的根源。
