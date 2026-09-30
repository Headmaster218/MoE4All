---
kind: architecture
status: current
scope: parallel-runtime
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 并发调度

## 所有权

`ParallelSeam` 是 `infr serve --parallel N` 背后的持久计算工作进程。请求在线程侧完成协议校验和准备后进入
同一个模型调度器，不由任意一个请求线程临时充当批处理领导者。

每个槽位独立拥有：

- KV 与动态段；
- DeltaNet/PLE 等循环状态；
- 采样、停止条件和输出进度；
- MTP 目标/头状态（启用时）。

模型固定权重、Vulkan 后端、专家分页器和可兼容的运行时工作区跨槽位共享。

## Prefill

- 短的未缓存提示词可以沿解码 LRU 图以教师强制行前进。
- 长预填充使用配置的 ubatch 和独占预填充 ring。
- 长预填充期间不在中间插入解码；这是 ring-buffer 与专家重建的设计约束，不是尚未实现的优化。
- 预填充结束并恢复解码 arena 后，就绪行才重新进入批量解码。
- 共享 token group 的 lane 按多模态优先、剩余 Prefill 较长优先、slot 序号稳定排序；这决定共享执行中的 lane 顺序，
  不会打断长 Prefill 独占 ring 的约束。

## Decode

兼容的 Qwen3.8 行按层同步批量执行。稠密/稀疏 QSA 可以形成兼容子组，但每个序列保持独立位置、
KV、所选块和循环状态。无法进入该路径的架构或受约束生成使用已有交错门。

调度器需要处理：

- 同时 ready 的 rows；
- 先后到达的请求；
- cohort 缩小、EOS 和取消；
- 某 slot 进入长 Prefill 时其他 Decode 的等待；
- 冷 KV restore 和 prefix reuse。

## Context 与资源

默认 context 由全部 `N` 个 slot state 加一份共享 runtime workspace 一起定价。提高 `--parallel` 可能降低每 slot
自动 context，但不应让原本能启动的默认配置在运行中突然 OOM。显式 `--ctx` 则按每 slot 固定容量计价，无法容纳时在
启动或分配边界明确失败。

## MTP

MTP concurrency 不是让每个请求建立一份模型。Target/MTP 固定权重共享，每个 slot 保留自己的 draft、VERIFY、
accepted-prefix 和 checkpoint state。兼容 slot 可进入 batched target VERIFY/Decode；普通和 MTP work 的公平性仍由同一
worker 控制。

当前 Qwen3.8 concurrent MTP 上限为两个 slot，并仅对 greedy-compatible 请求启用；不兼容请求走普通 Decode。MTP sidecar
与各 slot runtime 在 `ParallelSeam` 初始化阶段分配。传统单请求、无 Vision/Embedding 的 serve 路径不经过此调度器，MTP runtime
在首次 MTP 请求时初始化。长 Prefill 仍独占 Prefill ring，期间不插入 Decode。

## 不变量与验证

- 不在长 Prefill ring 使用期间插入普通 Decode。
- 一个 slot 的位置、KV 或 recurrent checkpoint 不能被另一 slot 覆盖。
- batch 中提前 EOS、取消或拒绝 draft 的 row 不得改变其他 row 的 state。
- throughput 评估同时看 aggregate、per-slot、中后段、TTFT 和 text-gap tail latency。
- 并发 Benchmark 必须包含先后到达和不同上下文长度，不能只测同步短请求。

历史测量和优化过程见 [Qwen3.8 concurrency campaign](../../evidence/campaigns/qwen38/2026-09-23-concurrency.md)。
