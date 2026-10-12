---
kind: incident
status: historical
scope: mixed-batch-sampling
recorded: 2026-10-11
source_checkpoint: release-0.10.0
---

# 混合 batch 采样行错位

## 复现与影响

2026-10-10 隔离 A/B：两条正在 Decode 的 lane 加一条带32个 uncached token、约758 cached tokens 的短继续请求。仅一路 Decode 的 two-lane control 通过；两路 Decode+短 Prefill 的四次 mixed/math trial 中，旧版八条继续输出全部错误，修复版八条均正确并生成384 tokens。旧输出常在18..20 tokens后早停/串词，一条 math trial 变成重复“CH”。短上下文也能触发，不要求跨32K或刚好2048边界。

control ORCHID/MARBLE 单独输出 old/fixed 均正确且完全相同。mixed-0 短请求有一次连贯的拒绝重复任务，原报告单列 task-following exception，不将模型拒绝混作行路由失败。

## 根因与修复

混合 batch 的 hidden 行布局与最终 sampling suffix 没有使用同一明确行定位。将错误行送入 LM head 会使一个请求采样其他位置的状态，即使 KV/routing 本身合法也会输出异常。

`ef4d972b2` 在 `crates/infr-llama/src/seam/runner.rs` 修正 mixed batch 的 sampling suffix 投影并加入行边界检查/测试。它与[冻结 expert LUT 事故](INC-20260925-frozen-expert-lut-slot.md)是不同根因，不能因为都重复输出就合并归因。

## 隔离证据

runtime base 为 `5b3a7a65afe1c17e185336a559265c2e929ce6ea`，比对423文件仅 runner.rs 改动。old EXE SHA256 `a69d42a569d33c804e4a178369b6e05fa56602129e12b5c5e00c121bb5a019ca`，fixed `132042cfde0a5c4127d446dd257e6e25ec36d416174c2e104a6284c26f6b316b`；rowRoutingChecksPassed=true。这是固定 runtime 的 correctness A/B，不是版本性能对照。

deployment smoke 记录OK；部分后台重启曾出现 host pager VirtualAlloc 失败，最终从 byte-identical 已测副本恢复，**分配失败原因未诊断**。后续 `1d343805b` 增加 host allocation 诊断，不把它写成已修复未知 allocation 根因。原始 comparison 在忽略的 mixed-head-ab artifact，本文不发布生产 IP/请求。
