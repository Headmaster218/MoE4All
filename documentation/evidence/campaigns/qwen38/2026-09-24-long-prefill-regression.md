---
kind: campaign
status: historical
scope: qwen38-long-prefill-regression
last_updated: 2026-09-26
verified_commit: ee492e2e7e2c177e4cf6700ca64ac72a412988f4
---

# Qwen3.8 长 prompt Prefill 速度回退调查

## 现象与证据

2026-09-24 的 0.8.0 服务日志中，一次 Qwen3.8 请求报告 Prefill 约 **414 tok/s**；后续长期观察约为 **600 tok/s**。用户指出此前相近服务能更快，并将回退范围缩小到 `e3192ae082a426ff4976857e106f5e9635a57f7f` 之后的若干提交。后续明确：服务使用两个并发 slot，但两次大型 Prefill 不会同时执行，必须等当前 Prefill 完成后才切换 Decode/下一次 Prefill。

该日志记录：

- `parallel=2`，`ctx=163840`，请求含 83 条 messages、`prompt_chars=222073`；实际 prompt token 数未在该行给出。
- MTP 关闭；Wizard 参数为 `ubatch=3072`、RAM budget `48g`、Q8 K/V，并选择 Vulkan1 RX 7900 XTX。
- placement 日志称保留请求的 3072-row Prefill chunk；prefill pager 为 `target_lanes=4 actual_lanes=1`、`resident_layers=0/48`、`ring_bytes=1192755200`。
- 同一启动计划显示主 arena 实测 16.67 GiB，48 个专家层均分页；prefill 时使用独占 ring。
- README 中 150K Qwen3.8 普通 Decode benchmark 的 Prefill 为 888–889 tok/s，20K 为 1,034–1,035 tok/s。但模型 prompt、运行命令、commit 与工作状态并未证明和服务日志相同，因此这些数值只能说明量级差异，不能构成严格回归 A/B。

## 提交范围与候选解释

`e3192ae` 到 checkpoint 之间的提交中，可能影响资源放置或 Prefill 执行形状的变化包括：

| Commit | 提交说明 | 与现象的关系 / 当前证据边界 |
|---|---|---|
| `2cc94991` | `perf: raise automatic prefill ubatch defaults` | 服务显式传入 3072，启动日志也称保留该值；默认值变化不足以单独解释本次速度。 |
| `1b45d40e` | `fix: freeze automatic RAM budgets at startup` | 日志配置是显式 48 GiB RAM budget，不足以证明该预算变化是根因。 |
| `de2e2ee7` | `perf: stream Qwen3.8 MTP prompt priming` | 此请求 MTP 关闭，不能直接归因于 MTP prime 路径。 |
| `30dbaa8e` | `fix(prefill): avoid impossible parallel scratch reservation` | 修改并行 Prefill scratch reservation，属于值得 A/B 的候选；日志中的 `actual_lanes=1` 与目标 4 lanes 的差距显示执行形状受限，但尚未证明由该 commit 引入。 |
| `6532b460` | `fix(vulkan): retry MoE arena placement at lower ubatches` | 可能改变资源不足时的 placement 行为；本次日志记载所选 3072 被保留，未显示发生降档，不能据此归因。 |

## 当前判断

已确认的不是“某个 commit 导致回退”，而是本次服务真实 Prefill lane 数只有目标值的四分之一。它是需要解释的强相关观测，不等于单独证明因果；`resident_layers=0/48` 和 ring 配置也不能独立说明 GPU 空闲或带宽利用不足。`414` 与长期约 `600` 的差异还可能包含请求内容、运行阶段、测量窗口及系统状态差异。

因此该回退的根因在 checkpoint 时仍**未定论**。没有同一 prompt、同一配置、同一机器状态下，对 `e3192ae` 与后续候选 commit 交替构建运行的对照数据；不能把 MTP、RAM、scratch reservation 或自动 ubatch 任一项写成已证实根因。大型 Prefill 串行切换是设计约束，不是这次回退的修复方向。

## 需要的验证

1. 使用相同模型文件、完整 prompt 与 tokenized prompt hash，固定 `ctx=163840`、`ubatch=3072`、RAM/VRAM、Q8 KV、并发 2、MTP 关闭；每个版本至少交替重复三次。
2. 分别测 `e3192ae`、`30dbaa8e` 及 checkpoint；记录最终 ubatch、target/actual lanes、ring bytes、实际 token 数、Prefill wall time 和 GPU/CPU 资源状态。
3. 不同时发起两个大型 Prefill；严格保持用户确认的调度前提。分开报告冷启动/首块和稳态 chunk，不用单次日志替代长期均值。
4. 只有复现 commit 间稳定差异后，才把性能变化归给特定改动；若表现随 `actual_lanes` 而非 commit 变化，再隔离 placement 与 scratch 预算。

## 结论状态

保留为未结项性能回退记录。这里没有运行 GPU benchmark，也没有宣称根因已定位；补齐受控 A/B 后追加结果，不覆盖本次观察。
