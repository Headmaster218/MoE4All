# 成果总表

[首页](../README.md) / 总览 / 成果

## 工程成果

| 方向 | fork 前/早期问题 | 最终阶段性结果 | 状态 |
|---|---|---|---|
| Windows | 运行、内存检测与脚本偏向非 Windows | Windows 11 + Vulkan 原生工作流，自动主机内存检测 | 已落地 |
| 深上下文 Attention | hd256 走大 score matrix；Q8 decode 慢 | hd256 FA、register-O、Q8 专用 decode kernel | 已落地 |
| Decode 气泡 | recorder/资源反复创建、同步过多 | 复用资源、流水化上传、减少提交等待 | 已落地 |
| Prefill 专家调度 | expert LRU 粒度造成编排与搬运碎片 | layer-major Host Store + 异步 5-slot layer ring | 已落地 |
| Decode 缓存 | 六个固定 `(role,size)` 池互相不能借槽 | 按物理尺寸的全局逻辑池，全局 LRU | 已落地 |
| 显存预算 | expert cache、runtime、Embedding 各自预留 | 同一弹性 VRAM arena，低/高地址双向生长 | 已落地 |
| Embedding | 最初依赖独立 llama.cpp worker | Nomic BERT 原生 CPU/Vulkan，和 LLM 共用 arena | 已落地 |
| 三级缓存 | 假设专家全部能进 RAM | full-RAM 与 bounded inclusive RAM/SSD 两条明确路径 | 已落地 |
| 超大 MoE | 64 GiB 主机无法承载完整专家 payload | Ling、DSV4、Qwen 122B 均能 SSD-backed 运行 | 已落地 |
| RAM→VRAM | CPU 直接写 ReBAR 约 14–19 GB/s | 部分 Host RAM 原地 Vulkan DMA，fallback 保留 | 已落地 |
| 可观测性 | 难以区分算子、分页、等待、SSD miss | profiler、ordered trace、离线 replay、微基准 | 已落地 |
| 产品化 | 依赖手动命令管理服务 | 浏览器 GUI、模型配置、启停切换、日志和速度 | 已落地 |

## 性能主线

### Qwen3.6-35B-A3B Balanced

同配置的早期 `16fbee2` 到 `02e0bfb`：

| 操作 | Depth | 早期 | 阶段里程碑 | 提升 |
|---|---:|---:|---:|---:|
| Prefill 512 | 0 | 437.3 | 509.4 | +16.5% |
| Prefill 512 | 100K | 164.3 | 416.8 | +153.7% |
| Prefill 512 | 200K | 89.1 | 299.8 | +236.5% |
| Decode | 0 | 13.8 | 31.6 | +129.0% |
| Decode | 100K | 20.5 | 47.7 | +132.7% |
| Decode | 200K | 18.3 | 33.6 | +83.6% |

之后 Q8 KV 专用 kernel 序列又在 200K 从约 31.55 提到 39.65 tok/s，累计
**+25.7%**；统一 VRAM 验收在 Q8/250K/8 GiB 设置下达到 41.2 tok/s。完整过程见
[Qwen 35B campaign](../experiments/qwen36-campaign.md)。

### DeepSeek V4 Flash

| 优化 | 局部或端到端结果 | 结论 |
|---|---:|---|
| Decode HyperConnection Sinkhorn | cache-hot 约 8.0 → 8.7 tok/s | 保留 |
| Gate+Up+Down bounded-tier batch | push 约 5.9 → 6.9 GB/s；3.9 → 4.0 tok/s | 保留并限制作用域 |
| MXFP4 complete-block decode | 3.9 → 4.3 tok/s | 保留 |
| F32 vec4 GEMV | `16384x24` 79.5 → 23.7 ms，约 3.35× | kernel 赢，端到端被 paging 吞没 |

最终 40 GiB bounded RAM/SSD 配置仍约 4.3 tok/s。瓶颈是每 token 约 1.053 GiB
Host→VRAM 流量和 0.404 GiB SSD demand，而不是还缺一个 5× kernel。见
[DeepSeek 收尾](../models/deepseek-v4-flash.md)。

### Qwen3.5-122B-A10B

| 阶段 | 实测结果 | 条件/解释 |
|---|---:|---|
| 2K 冷启动 trace | 11.2 tok/s | 15.53 GB GPU pool、51 GiB RAM、包含大量 SSD miss |
| Shared expert 融合 off | 17.0 / 17.3 tok/s | tg256、3 reps |
| Shared expert 融合 on | 19.2 / 19.2 tok/s | 同条件，约 +11.5%～12.9% |
| Host DMA，first-come import | 22.4 tok/s | 45 GiB bounded RAM |
| Host DMA，按池比例 import | **23.2 tok/s** | 28.99/45 GiB 成功 import，较 22.4 再 +3.6% |

11.2 与 23.2 不是单变量 A/B：前者是冷启动到 2K 的真实 route trace，后者是短基准。
它们分别回答“真实 SSD-backed 长运行怎样”和“新数据通路的上限改善多少”。

## 正确性与资源验收

- Nomic Embedding 对 llama.cpp oracle 的 cosine similarity 为
  `0.999955788`～`0.999974766`。
- 20 GiB 总 VRAM 预算下，弹性 arena 为 18,790,293,504 bytes；初始不可用尾部仅
  720,896 bytes（0.00384%）。
- Embedding 临时加载 260.86 MiB 权重和约 1.5 MiB runtime，完成后两类占用均回到 0。
- Chat + Embedding 并发请求通过统一执行门串行冲突访问，无死锁、无 stale Expert。
- Qwen 122B ordered trace 共 2,441,088 次 block access，无 sequence discontinuity、无
  backward call id。

## 没有被包装成“成功”的工作

比例保留 Down、paired Gate+Up eviction、强制 FlashAttention split、直接 planar-Q8
Prefill、丢弃 inclusive shadow、F16 HyperConnection temporary、过多并发 fanout 等均在
实测中无收益或退步，详见[失败实验](../experiments/rejected-experiments.md)。

---

[首页](../README.md) · [完整数据](../reference/README.md) · [架构演化](architecture-evolution.md)
