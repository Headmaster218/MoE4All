---
kind: architecture
status: current
scope: vulkan-cpu-miss
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 可选 CPU expert miss

## 定位与入口

这是 Vulkan paged-MoE 的实验 miss 支路，不是 CPU 后端，也不是把所有专家改到 CPU。默认 `kernels.vulkan.cpu_miss_threads=0` 关闭；不符合条件时保留原 GPU 路径。实现分布在 `crates/infr-vulkan/src/cpu_miss.rs、ops.rs 与 pager.rs`、`crates/infr-cpu/src/task_pool.rs` 与 `crates/infr-core/src/config/mod.rs`。

成立条件包括 x86_64 AVX2+FMA3、完整常驻 RAM expert store、普通单行 Decode、非第 0 层、指定量化与 shape。当前目标 shape 为输入 2560、中间宽度 640；Gate/Up 为相同的 IQ2_S 或 IQ3_S，Down 为 IQ4_NL；IQ3 需额外 opt-in。多行 batch、Prefill 和 MTP VERIFY 不使用此路径。

## 工作与同步

router 完成后区分 GPU hit 与 miss。满足 miss 上限时，将整个缺失集合交给私有 CPU task pool，同时保留 GPU hit 计算；不能靠删掉超限专家来满足上限。上限支持 1..3 个 miss，默认 1。CPU 结果合并遵守原 routing 权重和输出语义；专家仍按正常策略晋升 GPU cache，CPU 支路不取消后续 promotion。

私有 pool 不是共享 Rayon pool。Windows 可固定到物理核并优先快速核，在必要前保留两个快速核给主线程等工作；线程数受实际物理核能力限制。其他平台不等于复制 Windows affinity 策略。忙等、唤醒、host result 和 park 会改变 overlap 与功耗，不能只看孤立 FFN 耗时。

## 默认值与证据

Config 的 push、host_result、token_park 等开关默认 false，spin 默认 16384；向导主动启用 CPU miss 时设置一组不同的实验参数，不能把向导配方当全局默认。

[专项记录](../../evidence/campaigns/qwen38/2026-10-08-decode-and-cpu-miss.md)的最终 CPU 端到端 screen 与 GPU 基本持平；因此默认关闭，不宣称速度提升。该专项早期只测一个 miss，代码后来支持 1..3 不等于两/三个 miss 已获性能验收。另一个同阶段 GPU router-to-hit 改进有独立 A/B，不能归功于 CPU miss。

用户操作见[CPU miss 指南](../../guide/cpu-miss.md)，参数见[配置](../../reference/configuration.md)。
