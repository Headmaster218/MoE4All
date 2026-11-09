

---

## Round 130：logits 探针终于命中 —— 模型是「确信地」答错

### 探针位置（前几轮绕圈的教训）

单请求走**单流 decode**，它在 `runner.rs` 的 `if is_decode && at_frontier` 分支里：

```rust
if !(gpu_argmax || gpu_sample) || logits_out.is_some() {
    be.download(logits_buf, cast_slice_mut(&mut logits))?;   // ← 真正的出口
```

**关键**：默认 `temp=0` 走 GPU argmax，**logits 根本不下回主机**，所以前三次把探针加在
`logits_out_buf`、并行采样循环、`req.out_logits` 上全都落空。

命中条件：`INFR_LOGIT_PROBE=1 INFR_NO_GPU_ARGMAX=1 INFR_NO_GPU_SAMPLE=1`，
逼它走 host 采样路径。

**教训：加探针前先确认那条代码路径会被执行，不要对着候选点逐个试。**

### 实测数据

`What are the first five primes?`，无 engram：

```
max=32.8237 min=-19.7105 entropy=0.001 top5=[1309:0.9999, 477:0.0000, ...]  → "We"
max=26.9747 min=-15.4691 entropy=0.018 top5=[3287:0.9979, 304:0.0015, ...]  → " need"
max=24.0455 min=-15.2429 entropy=0.103 top5=[16:0.9878, 4888:0.0033, ...]    → " answer"
```

（前两行 `entropy=3.342 / 4.560` 是 warmup 那一轮 `"Hi"` 的，不算数。）

### 结论：**不是崩溃，是确信地错**

| 指标 | 值 | 说明 |
|---|---|---|
| softmax 熵 | **0.001 – 0.103** | 极集中 |
| top-1 概率 | **99.99% – 98.78%** | 几乎确定 |
| top-2 及以后 | 全部 < 0.3% | 完全分化 |

**分布是健康的、尖锐的、稳定的。** 模型非常确定自己应该接着说
`We need answer. Need interpret. Need likely ask. Need answer.` ——
一种自我重复的泛化话术，而不是针对 `What are the first five primes?` 的回答。

### 这排除了什么

- ❌ logits 崩溃 / NaN / 平坦
- ❌ 数值不稳定
- ❌ 采样 bug（而且 `temp=0` 是确定性的）
- ❌ RoPE 配置错误 —— 引擎的 `role==0 → 10000 无 YaRN`、`role 1/2 → 160000 带 YaRN`
  与 `config.rs:1126-1146` 注释记录的参考实现一致

### 剩下的是什么

模型**确信**它在回答一个泛化问题，而不是这个具体问题。
⇒ 上下文表示（attention 输出 / KV 内容）没有把「质数」这个具体信息带下去，
但残差流仍然稳定，所以输出层照常高置信度地生成套话。

**下一步应该 dump 中间层**：第 1 层 attention 输出、第 1 层 MoE 输出，
看残差流里到底有没有 prompt 的信息。

