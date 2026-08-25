# 实验与决策索引

[首页](../README.md) / 实验

这里记录的不只是最终保留代码，也包括真实测过但被否决的方向。

| 页面 | 主题 |
|---|---|
| [Qwen 35B optimization campaign](qwen36-campaign.md) | 从 synthetic baseline 到 kernel/Host Store/Prefill ring |
| [缓存策略实验](cache-policy.md) | 六池、全局池、8:7、Down soft cap、plain LRU |
| [Trace 与模拟](trace-simulation.md) | ordered route、replay、成本模型、适用边界 |
| [失败/暂缓实验](rejected-experiments.md) | 不应无证据重复投入的方向 |

## 保留实验记录的原因

性能工程很容易在不同 run state、cache geometry 或 profiler mode 下重复发现“假收益”。本
目录要求每项写明：

- 单变量是什么；
- workload/模型/量化/depth/cache；
- 观察值；
- 机制解释；
- 保留、回退还是等待什么条件再做。

数字使用规则见[benchmark method](../reference/benchmark-method.md)。

---

[首页](../README.md) · [模型索引](../models/README.md) · [证据索引](../reference/evidence-index.md)
