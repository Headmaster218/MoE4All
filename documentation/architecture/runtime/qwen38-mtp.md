---
kind: architecture
status: current
scope: qwen38-mtp
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# Qwen3.8 四个 token 的 MTP

## 目标

Qwen3.8 MTP 是自推测解码：轻量 MTP 头每轮最多预测四个后续 token，目标模型用一次批量 VERIFY 检查候选行，
然后提交已接受前缀。有效预测数会受剩余 context 和 generation 容量限制，末轮不保证始终有四行。接受率高时减少目标模型解码次数；接受率低时收益受 VERIFY 和状态维护开销限制。

## 驻留

并发服务路径（`ParallelSeam`）在统一显存 arena 建立前固定分配 MTP sidecar 权重和各 slot 的 MTP runtime：

1. 主模型与 MTP 固定资源先真实申请。
2. 查询 Vulkan 设备真实剩余空间。
3. 用剩余空间建立统一显存 arena。

因此专家缓存会按测得的余量缩小。传统单请求路径则在首个 MTP 请求时
懒初始化 MTP runtime；不要把并发服务的启动期资源顺序套用到这条路径。两条路径都不能用预估值替代真实显存分配和余量查询。

## 一个周期

```text
committed target state
        |
        v
MTP head drafts up to 4 tokens
        |
        v
target batched VERIFY (up to 4 rows in one forward)
        |
        +--> accept prefix
        +--> reject suffix
        |
        v
restore/commit target and head frontier
```

拒绝时保留已接受位置对应的前状态，不从头重算整个草稿。VERIFY 写入的推测后缀必须能回退到已接受前沿。

## Prefill

目标提示词 prime 仍是主要预填充工作。MTP 头需要追赶隐藏行和自己的状态。历史 20K 测量表明，
目标模型特殊 prime 路径、隐藏行往返和 MTP 固定驻留都可能造成预填充差距；这些是优化专项问题，
不改变 MTP 的状态语义。

## 并发与多模态

当前实现边界：

- 两个槽位的 Qwen3.8 草稿 VERIFY；
- 并发槽位的批量 MTP 解码；
- 多模态推测服务；
- 长上下文生成容量限制。

并发 MTP 最多支持两个 slot；只对 greedy-compatible 请求启用，不满足采样条件的请求回退普通 Decode。单 slot、纯文本且
没有辅助 Vision/Embedding 服务时，serve 保留传统单请求 MTP 路径；需要共享 Vulkan unified arena 的组合则走 `ParallelSeam`。

普通并发 token group 的 lane 排序优先多模态 lane，再按剩余 Prefill 长度降序、slot 序号排序；并发 MTP 的普通 Decode
同步阶段沿用该顺序。该排序不改变长 Prefill 独占 ring、期间不插入 Decode 的规则。

固定 target/head 权重共享，每个 slot 的 head KV、pending hidden、target checkpoints 和接受 frontier 必须独立。
Vision 插入后的 mRoPE/KV identity 也属于 slot state，不能用文本 slot 的普通前缀身份替代。

## 正确性门槛

- Greedy 高接受率 workload 应与 ordinary target output 一致。
- 0/partial/all acceptance 都必须覆盖。
- VERIFY 跨 32K segment、QSA boundary、EOS 和最大 context 时状态一致。
- 拒绝后下一 cycle 从 accepted frontier 继续，不复用未接受 suffix。
- frozen expert LUT 和 runtime claim 不能在 VERIFY command stream drain 前移动被引用 slot。

## 性能解释

MTP tok/s 必须同时报告 alpha、VERIFY 宽度和普通 Decode 基线。高 alpha 的数数任务不能代表普通问题；全程平均也不能替代
中段和后段。20K 对照见 [Benchmark](../../evidence/benchmarks/2026-09-24-qwen38-mtp-20k.md)，调查过程见
[Qwen3.8 campaigns](../../evidence/campaigns/qwen38/2026-09-22-mtp-investigation.md)。
