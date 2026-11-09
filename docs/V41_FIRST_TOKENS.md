

---

## Round 139：超连接滞后调度排除（release 实测），嫌疑只剩压缩注意力 / hash 路由

### 做法

`runner.rs:6592` 本来就有一条针对 HC 滞后的 `debug_assert!`：

```
"deepseek41 layer {l}: the attention collapse reads lag slot [0] before layer {}'s
 FFN sublayer wrote it — the HC lag schedule is broken, and every layer but the first
 would silently consume a stale mix"
```

**但它只在 debug 构建生效，而我们唯一能跑这个模型的构建是 release。**
读完代码「觉得对」不算证据。所以把它包进 `INFR_HC_LAG_CHECK`，让 `assert!` 在 release 也执行。

### 实测

```
INFR_HC_LAG_CHECK=1 infr run ... "A B C D"
→ [exit=0]，无 panic
```

**⇒ V4.1 的「HC 系数滞后一个子层」调度是正确的。** 这是运行证据，不是阅读判断。

### 嫌疑范围

| # | 嫌疑 | 状态 |
|---|---|---|
| 1 | 超连接滞后 | ✅ **排除**（本轮）|
| 2 | 压缩注意力（`kv_source_layer_ids=[2,8,14,20]` + 34 复用层） | 待查 |
| 3 | MoE hash 路由（`ffn_gate_tid2eid`） | 待查 |

### 已排除清单（累计，全部有运行证据）

- 引擎通用路径：qwen2.5-0.5b 在同一二进制上输出正常
- prompt 构造：模型确实收到 `2+2=`
- YOCO / `n_prefill_layer`：DeepInfer 默认 `off`
- APE：V4.1 无该张量
- warmup：`INFR_NO_WARMUP` A/B 输出几乎一致
- engram：`INFR_ENGRAM_ZERO` 零注入仍错
- logits 崩溃：熵 0.001–0.103，top1 98–99.99%，分布健康
- **超连接滞后：`INFR_HC_LAG_CHECK` release 断言通过**

### 仍未解决的硬约束

需要一个**可运行的参考实现**才能做数值比对。llama.cpp 报 `blk.1.engram_embd.weight is missing`
（该 GGUF 不含 engram 张量，只有元数据），DeepInfer 是 Python 实现跑不动 130 GB。

