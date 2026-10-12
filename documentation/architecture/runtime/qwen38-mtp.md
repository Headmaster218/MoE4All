---
kind: architecture
status: current
scope: qwen38-mtp
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# Qwen3.8 MTP 与机会式调度

## 周期与状态

Qwen3.8 的自推测路径用轻量 MTP head 产生候选，再由 target 一次批量 VERIFY。并发调度器的 `DRAFT_TOKENS=4` 是 **VERIFY 总行数**：一个已确定的 leading token 加最多三个 draft 候选，不是四个新候选。传统单请求路径将 `spec.k` 限制在 2..4，末轮还受 generation/context 剩余容量限制。alpha 只统计 draft 候选。

```text
committed target/head frontier
  -> leading token + draft candidates
  -> target batched VERIFY
  -> accept prefix / reject suffix
  -> commit accepted state and restore the frontier
```

拒绝时恢复 accepted frontier 对应的 checkpoint，不能重用未接受后缀，也不能从头重算整个草稿。target recurrent state、head KV、pending hidden、checkpoints 与接受前沿按 slot 独立；固定权重共享。VERIFY 引用的专家 LUT 和 runtime claim 在 command stream drain 前不能移动。

## 固定资源与 Prefill

`ParallelSeam` 在统一 arena 前真实分配 target/head 固定权重及各 slot 的 MTP runtime，再查询剩余 VRAM 并建立 arena；不能拿预估余量替代真实分配。传统单请求路径在首个 MTP 请求时懒初始化 runtime，两者不能混写成相同的启动顺序。

`spec.mtp_context=32768` 是 head 的滑动历史窗口，限制 head KV 驻留及追赶成本；target 仍保留完整上下文。它不是用户 context 上限，也不同于下面的 target 深度门槛。MTP Prefill 现在跟随普通自动 ubatch 策略，不再隐含下调到 2048。固定驻留、hidden tap 和 head catch-up 仍有开销，不能承诺两者 Prefill 必然同速。

## ParallelSeam 的门槛

- `spec.mtp=false` 默认关闭；启用后只对 greedy-compatible 请求使用，普通采样回退 ordinary Decode。
- `spec.mtp_max_context=49152` 是 target 上下文深度 cutoff，0 表示禁用 cutoff。这个参数用于并发调度器；传统单请求 `generate_vulkan` 不读取它。
- `spec.mtp_concurrent=false` 默认不跑多活跃 lane 的 MTP；有其他工作时走普通 batch，跳过 prime 或使既有 head 失活。不能承诺 lane 减少后自动重新 prime 所有 head。
- 显式开启并发 MTP 后，用 8 个 probe cycle 和至少 80% 的接受率判断是否继续；head 未全部 ready 或收益不足时回退 ordinary Decode。
- Vulkan serve 的 MTP draft 配置最多两个 KV slot；不是只在 `mtp_concurrent=true` 时才有限制。
- 单 slot、纯文本、没有辅助 Vision/Embedding 时保留传统串行 MTP 服务路径；需要共享 arena 的组合走 `ParallelSeam`。

多模态 lane 的 mRoPE/KV identity、hidden 和 checkpoint 必须保持一致。普通 token group 优先多模态 lane，再按剩余 Prefill 长度降序及 slot 排序；普通 Decode 同步阶段沿用该顺序。长 Prefill 的 ring 阶段不插入 Decode；ring lane 数不等于请求并发数。

## 核验与性能解释

覆盖 0/partial/all acceptance、EOS、context 上限、32K segment、QSA boundary、后缀回退以及跨 slot 隔离。Qwen3.5/3.6 的历史 MTP 数值问题及搁置结论不能套用为 Qwen3.8 当前能力。

性能必须同时写出路径、实际是否启用 MTP、head horizon、target cutoff、VERIFY 行数、alpha、上下文及普通 Decode 对照。150K 下仅写“启用 MTP”不足以证明调度器正在跑 VERIFY。早期 [20K](../../evidence/benchmarks/2026-09-24-qwen38-mtp-20k.md) 和 [150K](../../evidence/benchmarks/2026-09-24-qwen38-mtp-150k.md) 数字保留原基线；新策略的过程见 [0.10.0 Prefill/MTP 专项](../../evidence/campaigns/qwen38/2026-10-06-prefill-and-mtp.md)。

主要源码：`crates/infr-llama/src/parallel.rs`、`seam/runner.rs`、`mtp/mod.rs` 和 `crates/infr-core/src/config/mod.rs`。
