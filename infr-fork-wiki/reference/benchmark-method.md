# Benchmark 方法与数字口径

[首页](../README.md) / [Reference](README.md) / Benchmark method

## 测试机

阶段主机：

| Component | 配置 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX，24 GiB |
| Backend | Windows 11，Vulkan，RDNA3 |
| CPU | Ryzen 5 5600X |
| RAM | 64 GiB DDR4 |
| Storage | 本地 SSD |

不同日期的桌面进程、GPU power/clock、SSD cache 和服务器负载会造成漂移，因此 interleaved
A/B 比跨小时绝对值更可信。

## 常用指标

- `ppN`：Prefill N tokens；
- `tgN`：Decode/generate N tokens；
- `dK`：已有 KV depth；
- `cache G`：需确认是旧 expert-cache 参数还是新 total VRAM budget；
- hit rate：需注明 GPU total、RAM conditional 还是 VRAM-or-RAM combined；
- bandwidth：GB/s 与 GiB/s 不混写。

## Synthetic depth

`--synthetic-depth`：

- 真实分配并初始化所需 KV buffers；
- 推进 runner/allocator position，使 Attention 扫描目标 history length；
- 不执行生成此前 100K/250K tokens 的完整 Prefill；
- deterministic KV 只用于性能，不代表有语义的历史。

因此它可以比较 depth cost，却不能用来验证长文本回答质量或真实 router history。

## Warmup

默认 benchmark discarded warm repetition 会改变：

- shader/pipeline cache；
- Expert GPU/RAM residency；
- WDDM clock/power state；
- filesystem cache；
- recurrent/KV state（若 reset 语义有 bug）。

报告必须标明 cold process、benchmark warmup、continuous session 或 steady-state。自适应 warmup
曾让吞吐从 28.6→41.4→60.0 仍不收敛，因为它在改变 cache working set，因此被回退。

## A/B 最低要求

匹配：

- 同模型文件和 quant；
- 同 commit 除目标 patch；
- 同 KV dtype、context、synthetic depth；
- 同 total VRAM/RAM budget 和 pool geometry；
- 同 prefill/decode tokens、ubatch、submit cap；
- 同 profiler/trace 开关；
- 尽量 A-B-B-A 或交错执行。

若其中任何一项变化，数字可以并列展示，但不直接计算“代码提升”。

## Profiler 对性能的影响

- Pager aggregate profiler 较轻，但仍可能影响短 Decode；
- per-op GPU timestamp 增加查询/同步；
- ordered trace 写数百万 CSV rows，不用于最终 throughput；
- RGP capture 会显著改变 wall time，只做归因。

正确用法：profile 找瓶颈，关闭 profile 做 endpoint A/B。

## 实测、模拟、理论

### 实测

有 binary、命令条件、日志/CSV；仍要防 run drift。

### 模拟

route trace exact replay + 实测成本模型。报告 slot/miss 可以很精确，但 time prediction 受
overlap 和 calibration 限制。

### 理论

例如 `bytes / bandwidth` 或 Amdahl 估计。应列出 bytes、带宽、固定成本、可 overlap 窗口。

## 常见误用

- 用 122B cold 2K 的 11.2 与 tg256 DMA 的 23.2 宣称一个 patch +107%；
- 用 35B 14 GiB decode-only 48.1 当作 250K 对话服务安全设置；
- 将 DeepSeek 60 GiB trace plateau 当成 60 GiB 能装完整 147 GB payload；
- 将 IQ3_M 0.873 size simulation 当成真实 kernel benchmark；
- 把 AIDA64 25.2 GB/s 大块 copy 当作每个 1～8 expert promotion 都能达到；
- 将 GPU 3D utilization 当作 compute occupancy 的唯一指标。

## 推荐记录模板

```text
Date / commit / binary hash
Model exact path + GGUF shard/quant
GPU/driver/backend
VRAM budget / RAM budget / KV dtype / context
Workload: pp/tg/depth/ubatch/reps/warmup
Profiler/trace/env overrides
Raw samples + mean/range/spread
Cache geometry + hit/miss bytes
Conclusion + rejected alternative + artifact paths
```

---

[Reference](README.md) · [Evidence index](evidence-index.md) ·
[Trace/simulation](../experiments/trace-simulation.md)
