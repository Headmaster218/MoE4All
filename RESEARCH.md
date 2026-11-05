---
kind: index
status: current
scope: research-and-performance-models
last_verified: 2026-11-06
---

# 理论研究与性能方法

这里汇总 MoE4All 开发过程中已在本仓库公开记录的分析模型、实验方法与设计判断。每项保留摘要，并链接到完整推导、数据和实现记录；原文中的日期、模型及硬件条件仍然有效。

## 1. 专家缓存容量与命中率

**问题：多给专家缓存一些显存，能少搬多少专家？**

对真实任务的专家访问序列进行 LRU 离线回放，得到缓存有效容量与 Miss 率之间的非线性关系，再用连续分段指数函数压缩曲线。在这份 FlashNext 序列中，16 GiB 缓存的 Miss 率约为 11.96%，24 GiB 约为 3.86%。这里的容量是专家实际可用空间，不是显卡总容量。

这份研究适合分析 KV、MTP 和临时工作区挤占缓存的代价，也可作为资源分配模型的输入。它是固定访问序列的容量回放，不是所有模型的通用命中率公式。

[完整曲线、拟合公式与误差](documentation/evidence/benchmarks/2026-10-11-flashnext-expert-cache-capacity.md) · [机器可读参数](documentation/evidence/benchmarks/assets/2026-10-11-flashnext-expert-cache/fit-summary.json)

## 2. 异构设备如何分配模型层

**问题：两张显存或带宽不同的卡，应该各承担多少层？**

分层同时决定每张卡的工作量和每层能分到的缓存。慢传输设备可以少分一些层，提高其专家命中率；快设备则承担更多层。应联合考虑有效专家缓存 `Vᵢ`、实际块传输带宽 `Bᵢ` 和分层比例 `rᵢ`，最小化预测搬运时间，而不是只按显存容量或带宽比例切分。

现有推导给出基于 Miss 曲线的目标函数及整数层数求解方法。这属于部署规划模型，尚不等同于加入计算、跨卡同步和拓扑竞争后的完整多卡实测。

[多卡层数配比公式与算例](documentation/evidence/benchmarks/2026-10-11-flashnext-expert-cache-capacity.md#多卡层数配比)

## 3. 看关键路径，而不是把耗时直接相加

**问题：一个模块更快了，为什么整轮速度没有变？**

计算、DMA、CPU 上传与准备工作可以重叠。真正影响 token 时间的是没有被其他工作覆盖的部分；减少并行路径上的耗时，可能暂时没有端到端收益。反过来，额外同步、资源争用或缓存挤压，也可能让一个单独更快的模块拖慢整轮。

优化准入因此分层进行：先保证数值和状态正确，再验证单元与真实模块收益，最后检查匹配条件下的端到端是否回退。模块有收益、端到端持平的优化可以保留。

[GPU 驻留与交接专项记录](documentation/evidence/campaigns/qwen38/2026-10-11-gpu-resident-handoff.md) · [计算与传输的条件分析](documentation/evidence/campaigns/qwen38/2026-10-23-cpu-expert-offload.md)

## 4. Prefill 的 ubatch、lane 与传输重叠

**问题：更大的 ubatch、更多的 lane，为什么不总是更快？**

ubatch 影响一次处理的 token 数、激活与运行时空间，以及需要加载的专家集合；lane 提供专家加载和 GPU 计算之间的流水缓冲。两者共同消耗显存，也共同决定传输能否被计算覆盖。

应比较真实专家集合与搬运量、计算效率、可用缓冲及物理布局，而不只比较 token 行数。带宽受限的整轮测试可能掩盖 kernel 收益，因此需要把纯计算、模块流水与端到端结果分开。

[Prefill 性能回退专项](documentation/evidence/campaigns/qwen38/2026-09-24-long-prefill-regression.md) · [运行时资源生命周期](documentation/architecture/memory/runtime-resource-lifecycle.md)

## 5. CPU 算专家，还是传给 GPU？

**问题：CPU 能否在 GPU 计算命中专家时，把 Miss 专家顺便算完？**

判断标准是完整专家 FFN 的耗时，加上输入、结果与同步成本，能否胜过搬运权重后由 GPU 计算。单看峰值算力、某个点积或顺序内存读取带宽，都不能回答这个问题。

已有研究区分了算子实现、CPU 指令与依赖模型、DMA 预算和真实集成结果。在已测条件下，CPU Miss 计算没有形成足以默认启用的稳定收益；更强 CPU 的纸面算例仍需实机验证。

[CPU 专家计算的模型、历史测试与边界](documentation/evidence/campaigns/qwen38/2026-10-23-cpu-expert-offload.md)

## 6. MTP 的收益来自哪里

**问题：一次预测多个 token，为什么仍可能比普通 Decode 慢？**

收益取决于每轮实际接受的 token 数，以及 draft、批量 verify、状态恢复和 catch-up 的总成本；还要计入 MTP 权重及 KV 对专家缓存的占用。接受率高，并不自动意味着周期更短。

应分别测量草稿生成、批量验证与状态恢复，再比较相同模型和输入条件下的完整周期。短、长上下文需要分别记录，不能只凭接受率或草稿头大小预测提速。

[MTP 瓶颈专项](documentation/evidence/campaigns/qwen38/2026-09-23-mtp-bottlenecks.md) · [20K 实测](documentation/evidence/benchmarks/2026-09-24-qwen38-mtp-20k.md) · [150K 实测](documentation/evidence/benchmarks/2026-09-24-qwen38-mtp-150k.md)

---

[返回项目首页](README.md) · [文档中心](documentation/README.md)
