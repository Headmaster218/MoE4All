---
kind: guide
status: current
scope: cpu-miss
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# 实验 CPU miss 使用

普通 GPU 推理不需要开启此功能。它要求专家完整装入 RAM，且只覆盖特定 Qwen shape/量化的单行 Decode；具体边界见[架构](../architecture/runtime/cpu-miss-offload.md)。

Windows 向导在选择 CPU miss 后询问计算物理核与 miss 上限。混合架构会区分快速/效率物理核；不要照搬其他机器的线程数。手动 CLI 的最小配置为：

```powershell
.\infr.exe serve --set kernels.vulkan.cpu_miss_threads=4 --set kernels.vulkan.cpu_miss_max=1 <model.gguf>
```

这是示例而不是建议所有机器用四核。0 表示关闭线程功能；上限范围 1..3。直接 CLI 不自动加入向导的实验配方。向导还会设置 push、host_result、token_park 为 true、spin 为 262144；Config 默认则为 false/16384。实验时显式记录这些参数。

比较要在同一二进制、相同 KV/cache 状态、sampling、上下文和请求下做 GPU/CPU/GPU 或 ABBA，并保留输出哈希和中后段速度。不要用孤立 FFN 微基准替代 end-to-end，也不要把 MTP VERIFY 或多行并发当作 CPU offload 已生效的证据。

当前结果是基本持平而非稳定加速，默认保持关闭。配置失效、资源不足或 shape 不符合时回退 GPU。RAM 分配失败的诊断应结合物理内存、Windows commit headroom 和连续分配错误，不能只看进程工作集。
