---
kind: benchmark
status: current
scope: v5 final stability baseline
run_date: 2026-10-19
commit: v5-final-stable
evidence: measured
---

# 2026-10-19 v5-final-stable 稳定性验证

**结论**：`v5-final-stable` 的运行时代码回到 `v4-pair-pipeline`。MTP
双窗口实验不再采用；默认单发路径仍使用串行 MTP，默认 VERIFY 宽度为 `k=3`，Qwen3.8
48 层模型的自动层切分为 `26/22`。

## 回退边界

- `crates/infr-llama/src/mtp/qwen4.rs`、`seam/mod.rs`、`seam/runner.rs` 和
  `seam/weights.rs` 已恢复到 `v4-pair-pipeline` 的运行时代码。
- 未提交的 `RecurrentState` VRAM-first 放置改动一并撤掉。该项在 32K/256K、`-u256`、
  双 lane 原始工作点下额外占住约 36 MiB，使激活池 `moe_ye`（50.6 MiB）撞上 VRAM
  守门，因此不适合作为生产稳定基线。
- 运行时代码不再包含本轮双窗口 MTP 的 `MtpWindow`、双 front trace bank 和
  `draft_following_window` 逻辑；`v5-final-stable` 与 `v4-pair-pipeline` 的运行时差异
  仅限本轮验证文档和已有探针文件。

## 默认配置确认

- 层切分：`INFR_PIPELINE_SPLIT` 未设置时，启动日志为
  `balanced_head=26 preferred_head=26`，最终放置为 `layers [0..=25]` 和
  `layers [26..=47]`。显式 `INFR_PIPELINE_SPLIT=26,22` 仍可覆盖。
- VERIFY 宽度：Qwen3.8 的配置默认 `spec.k=6` 会采用实测偏好 3。串行 MTP 日志中每轮为
  `candidate=[2 ids]`、`verify=[3 ids]`。

## 验证结果

### 32K/256K 双 lane 并发

配置：`INFR_PIPELINE=Vulkan0,Vulkan2`、`INFR_PIPELINE_OVERLAP=1`、
`INFR_KV_OVERFLOW_VRAM_MB=0`、`--parallel 2 -u256`，32K 输入、32 token 输出，流式请求。

- 两条 lane 均完成流式输出，没有请求失败。
- cohort ratio：**1.7479**。
- 单请求 decode：2935.5 ms；并发 cohort decode：3359.0 ms。
- `v4-pair-pipeline` 原始记录为 1.776-1.784，本次结果处于同一工作点和正常波动范围内。

### 串行 MTP 与会话连续性

配置：`INFR_MTP=1`、Q8_0 MTP sidecar、`INFR_PIPELINE=Vulkan0,Vulkan2`、
`--parallel 1 --ctx 16384 --temp 0`。

- 连续两次相同流式请求均正常结束。
- 两次输出文本逐字节一致，没有观察到 KV 污染或跨请求状态泄漏。
- MTP cycle 日志为 `candidate=[13, 2972]`、`verify=[13, 2972, 54]`，确认默认
  `k=3`。
- 本次 24-token 短跑不是吞吐基准；其中期 acceptance 下降，不能替代长上下文 MTP
  吞吐记录。

## 结论与限制

本记录证明的是回退后的生产稳定性、默认切分和 `k=3` 生效，不代表新的吞吐纪录。
32K/256K 双 lane 是稳定性与回归验证；串行 MTP 的短输出测试用于状态与流式正确性。
所有原始日志和 JSON 位于本机工作目录，不随 tag 发布。
