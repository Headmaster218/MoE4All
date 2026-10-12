---
kind: architecture
status: current
scope: parallel-runtime
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
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
- 长 Prefill 使用启动后最终选定的 ubatch，再按本次事务深度和 live arena 选择可容纳的 rows / ring lanes。流水线 lanes 是专家上传/计算缓冲通道，不是并发请求数。
- Qwen3.8 的长 Prefill 逐个完成请求，正在执行的 slot 按目标深度增长 KV；尚未执行的请求不提前提交完整 KV。未提交 KV 预留和本轮未使用的 QSA scratch 余量可临时借给 ring。
- 长预填充期间不在中间插入解码；这是 ring-buffer 与专家重建的设计约束，不是尚未实现的优化。
- 预填充结束并恢复解码 arena 后，就绪行才重新进入批量解码。
- 共享 token group 的 lane 按多模态优先、剩余 Prefill 较长优先、slot 序号稳定排序；这决定共享执行中的 lane 顺序，
  不会打断长 Prefill 独占 ring 的约束。

## Decode

兼容的 Qwen3.8 行按层同步批量执行。稠密/稀疏 QSA 可以形成兼容子组，但每个序列保持独立位置、
KV、所选块和循环状态。无法进入该路径的架构或受约束生成使用已有交错门。

该持久批处理 worker 的启动门控是 `qwen4exp`；其他 Vulkan 架构仍可经 `ParallelSeam` 提供多槽位和交错执行，但不能推断都有 Qwen3.8 的同步 batch 或长 Prefill 排程。发布前 Balanced 35B 三路日志曾显示多个长 prompt 交替推进 chunk，见[发布矩阵](../../evidence/benchmarks/2026-10-09-release-0.10.0-matrix.md)。

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

当前 Qwen3.8 MTP 服务仍最多两个配置 slot，仅对 greedy-compatible 请求启用；不兼容请求走普通 Decode。默认关闭并发 VERIFY（`spec.mtp_concurrent=false`）：多个 active work 使用普通批量路径，并跳过不需要的 head prime。显式开启并发 MTP 后，每 8 个 VERIFY cycle 检查草稿接受率，80% 为收益选择门槛；接受率不足或 head 未全部就绪时回到普通批量 Decode。Target context 超过 `spec.mtp_max_context=49152` 时调度器停用 MTP；设为 0 才关闭该 cutoff。这些收益门控不替代输出与状态正确性检查。MTP sidecar
与各 slot runtime 在 `ParallelSeam` 初始化阶段分配。传统单请求、无 Vision/Embedding 的 serve 路径不经过此调度器，MTP runtime
在首次 MTP 请求时初始化。长 Prefill 仍独占 Prefill ring，期间不插入 Decode。

## 不变量与验证

- 不在长 Prefill ring 使用期间插入普通 Decode。
- 一个 slot 的位置、KV 或 recurrent checkpoint 不能被另一 slot 覆盖。
- batch 中提前 EOS、取消或拒绝 draft 的 row 不得改变其他 row 的 state。
- throughput 评估同时看 aggregate、per-slot、中后段、TTFT 和 text-gap tail latency。
- 并发 Benchmark 必须包含先后到达和不同上下文长度，不能只测同步短请求。

历史测量和优化过程见 [Qwen3.8 concurrency campaign](../../evidence/campaigns/qwen38/2026-09-23-concurrency.md)。
