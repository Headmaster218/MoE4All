---
kind: campaign
status: historical-experimental
scope: MoE4All Qwen3.8 IQ3_XXS Strata-parity campaign
run_date: 2026-10-21
evidence: measured
---

# MoE4All Strata-Parity Campaign

## English

This directory collects the engineering reports behind PR #106, the experimental dual-GPU
256K serving path for Qwen3.8-Flash-Next IQ3_XXS.

The campaign investigated:

- correctness of long-context IQ3_XXS serving;
- dual-GPU layer splitting and cross-device state handoff;
- dynamic KV growth and host fallback at 256K context;
- two-slot serving and two-lane decode scheduling;
- Qwen recurrent-state and PLE-history placement;
- QSA decode-kernel cost and safe optimizations;
- capacity and performance limits on 2x RX 7900 XTX with 62 GiB host RAM.

The final experimental point is `v7-qsa-tuned`. It is evidence for one model, one quantization
and one AMD RADV hardware pair. It is not yet a universal production configuration.

The benchmark harness is semi-automatic: one command can start and stop the server, generate a
fixed prompt, issue concurrent requests and write logs/JSON, but model paths, configuration,
devices and split settings are still supplied manually. It is not a fully automatic CI benchmark.

Some reports contain machine-local paths from the original investigation. Model weights, local
`radv-test.toml` files and some raw traces are intentionally not included in this repository.

### Report Index

- [01-project-freeze.md](01-project-freeze.md): final campaign status, best-tested versions and
  explicit limitations.
- [02-m0-closeout.md](02-m0-closeout.md): M0 baseline, correctness closure, concurrency, MTP,
  capacity and microbenchmark findings.
- [03-iq3-correctness-closeout.md](03-iq3-correctness-closeout.md): IQ3_XXS correctness root-cause
  investigation and the temporary decode baseline.
- [04-iq3-budget-vulkan0.md](04-iq3-budget-vulkan0.md): capacity budget, Vulkan0 prefill candidate
  and the evidence behind device-placement decisions.
- [05-strata-parity-plan.md](05-strata-parity-plan.md): full investigation plan, baselines,
  go/no-go gates and deferred architecture work.

## 中文

本目录收集 PR #106 背后的工程报告。该 PR 为 Qwen3.8-Flash-Next IQ3_XXS 增加实验性双卡
256K 服务路径。

本次工程调查覆盖：

- IQ3_XXS 长上下文服务的正确性；
- 双卡层切分和跨卡状态交接；
- 256K 上下文下的动态 KV 增长与主机回退；
- 双 slot 服务和双 lane 解码调度；
- Qwen recurrent state 与 PLE history 的显存放置；
- QSA decode kernel 的开销定位和安全优化；
- 2 张 RX 7900 XTX、62 GiB 主机内存下的容量与性能边界。

最终实验版本是 `v7-qsa-tuned`。这些数据只对一套模型、一种量化和一套 AMD RADV 硬件成立，
还不是通用生产配置。

当前 benchmark 是半自动基准：一条命令可以启动和关闭服务、生成固定 prompt、发起并发请求并
写出日志/JSON，但模型路径、配置、设备和 split 仍需手动指定，尚未接入全自动 CI。

部分报告保留了调查时的本机路径。模型权重、本机 `radv-test.toml` 和部分原始 trace 因体积或
环境依赖没有放入仓库。

### 报告索引

- [01-project-freeze.md](01-project-freeze.md)：最终工程状态、已验证的最好版本和明确限制。
- [02-m0-closeout.md](02-m0-closeout.md)：M0 基线、正确性闭环、并发、MTP、容量和微基准结论。
- [03-iq3-correctness-closeout.md](03-iq3-correctness-closeout.md)：IQ3_XXS 正确性根因调查和临时
  decode 基线。
- [04-iq3-budget-vulkan0.md](04-iq3-budget-vulkan0.md)：容量预算、Vulkan0 prefill 候选和设备
  放置依据。
- [05-strata-parity-plan.md](05-strata-parity-plan.md)：完整调查计划、基线、go/no-go 门槛和暂缓的
  架构工作。
