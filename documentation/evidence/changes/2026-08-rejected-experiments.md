---
kind: change-history
status: historical
period: 2026-08
evidence_level: historical-measurement
---

# 早期否决实验与后来变化

“否决”只表示当时模型、设备和二进制下未达到默认 hot path 的收益，不是永久禁令。本表保留回退的幅度与原因；数值不可跨实验直接排序。后来实现发生变化时在最右列说明，避免把旧提案当成 0.9.0 当前状态。

| 范围 | 尝试与结果 | 当时决定 / 后来变化 |
|---|---|---|
| Attention/Q8 | submit cap 64-768 最好只提高 0.34%/0.47%；自适应预热虽出现 28.6→41.4→60.0，波动仍有 25%-28% | 小于漂移或不能收敛，不作为默认收益 |
| Attention/Q8 | 强制 FA split-K 在单一 100K 场景 +8.5%，200K 和跨模型不稳；hd128 cooperative matrix BDA 约 0.8x | 保留自动选择与手动覆盖 |
| Attention/Q8 | hd256 Prefill KV BDA 在 100K +0.3%、200K 略退；planar-Q8 Prefill 在 100K/200K 分别 -6.3%/-8.9% | GQA 下反量化重复，均不设默认 |
| Attention/Q8 | Q8 shared scale 一次加载 -4.9%；combine 4→2→1 对应 39.3→38.3→35.4；1536/2048/4096 块对应持平/-4.1%/-7.7% | 省字节不一定抵过占用率和串行阶段成本 |
| Attention/Q8 | combine-128 -12.2%；parallel-32 combine 34.95→35.0；F16 scale -1.2%；F16 PV -2.3%；重写 Q8 unpack -10.8% | 并行度或指令路径不如原实现 |
| Pager/Prefill | ring slots 3/5 约 289/284 tok/s，2/4 约 344/376；复制块 4096B 为 390.2、512B 为 323.2、256B 为 289.4 | 槽位与小块并非越多越快；第三 ReBAR lane 因 A/B 已耗约 660 MiB 未先做 |
| Pager/Prefill | 单纯 layer 模式 pp512 约 415，对照 418-420 | 获益来自连续传输与重叠，不是“改粒度”本身 |
| 缓存策略 | UG/Down 8:7 加成对淘汰 41.50→37.65、miss +5.72%；Down weight 7 在长测 40.65→40.55；weight 5 为 39.10、miss 114750 | 固定角色 quota 破坏真实冷热，回退到全局 LRU |
| MoE 调度 | UG→D 分支模拟通常慢 0.1-3.1 ms | UG 计算窗口短于搬运和第二次提交，保留完整 UGD 批次；详见[微基准](../benchmarks/2026-08-25-moe-pager-microbench.md) |
| DeepSeek V4 | 加宽 MXFP4 dqblk 6.8 对照 6.9；不限提交 6.8 对照 6.9；DP4A 后期 5.9-6.3；F16 HyperConnection 临时区 7.3-7.5 对照 F32 8.2-8.3；拆 4 次调度 8.2 对照 8.3-8.4 | 内核吞吐、占用率或调度成本抵消理论节省，见[V4 结项](../campaigns/deepseek-v4/2026-08-24-rx7900xtx-closeout.md) |
| DeepSeek V4 I/O | 非时序 ReBAR 2.7-2.8 且不稳；只增文件句柄并行 2.7-2.9；8 路请求并行 3.7 对照 3.8；去掉 inclusive shadow 3.9 对照 4.0，RAM hit 60.6% 对照 61.6%；外层并行复制 3.9 对照 3.9 | 保留请求级批量和 inclusive shadow，避免额外线程/恢复代价 |

另外三个设计选择不能停留在早期提案状态：VRAM 全量 compaction 后来被 cold contiguous window eviction 取代；排他式 RAM 层级被 inclusive shadow 取代，因为当时 VRAM→RAM 仅约 44 MB/s；**动态 KV 已在后续版本实现**，不能继续按 8 月“暂缓”描述。SSD router 盲预取则需要先在有序 route trace 上验证 precision、污染与队列窗口，不能从空闲带宽直接推断收益。用 MXFP4 route trace 推断 IQ3_M 时，`87.3%` 的块字节比例只表示同一访问序列下的容量效应，不证明改变量化后仍访问相同专家。

重开任何旧实验，应说明变了哪个前提、旧失败机制为何不再成立，并给出隔离微基准、端到端 A/B、正确性与跨模型回归、可完整回退的实现边界。静态模拟只能筛策略，不能证明 GPU/SSD 时序重叠。
