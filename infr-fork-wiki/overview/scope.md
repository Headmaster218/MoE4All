# 项目范围与证据规则

[首页](../README.md) / 总览 / 项目范围

## 记录什么

Git 范围以原仓库 `upstream/main` 与当前 fork `HEAD` 的差集为准：

```text
upstream: kryptic-sh/infr
origin:   Headmaster218/infr
merge-base: d7f320e7b8936fd6e1860115c5dd579c4572a27f
fork commits: 89
first fork commit: d9bd5a9 (2026-08-15)
```

作者名不是范围边界。`John(Desktop)` 与 `Zhuohang Wu` 在这 89 个提交中的工作都属于本轮
fork 演化，均纳入 Wiki。

纳入内容：

- Windows/RDNA3 适配和测量设施；
- Qwen 35B、Ling、DeepSeek V4、Qwen 122B、Nomic Embedding 的新增适配；
- 长上下文 Attention、DeltaNet、量化 MoE kernel 与调度优化；
- Expert Pager、统一显存、RAM/SSD 三级缓存、Host DMA；
- GUI、服务 supervision、Embedding API；
- 为上述工作做过的 benchmark、trace、模拟、失败实验和架构决策。

不纳入内容：

- `upstream/main` 已经存在且本 fork 未改变的通用功能；
- 为上游所有模型写一份完整用户手册；
- 未经记录支持的性能宣传；
- 仅存在于设想、从未实现或验证的能力，除非明确标记为“规划/暂缓”。

## 如何描述上游基础

必要时可以写一句“复用了上游的某能力”，但正文关注增量。例如：

> 上游已经提供 Vulkan runner。本 fork 新增 Windows 原生路径、分页专家数据通路和针对
> RDNA3 的 kernel/调度优化。

不会把前半句扩写成上游架构教程，也不会将上游能力计入本 fork 成果。

## 数字的四种标签

| 标签 | 含义 | 可以怎样使用 |
|---|---|---|
| **实测** | 有明确命令/日志/结果文件的硬件运行 | 可作同条件 A/B；仍需写清条件 |
| **历史样本** | 对话或阶段记录中的真实运行，但原始日志不完整 | 可证明量级，不能伪装成严格 A/B |
| **模拟** | 使用 route trace 和测得成本模型回放 | 用于找甜点；不能当硬件验收 |
| **理论** | 依据字节量、带宽、依赖关系推导 | 用于解释方向；必须陈述假设 |

### 不允许的比较

- 不把 `tg128`、`tg256`、连续对话的数字直接相减；
- 不把短 cache-hot 诊断和 SSD-backed 稳态当作同一场景；
- 不把 profiler-on 和 profiler-off 当作纯代码 A/B；
- 不把 Q8 KV 与 F16 KV、不同 cache、不同 depth 的数字宣称为单变量提升；
- 不把模拟表中的 14.5/15 GiB 当作已成功加载配置。

## 文档组织原则

1. 模型页面回答“这台模型最终怎样跑起来、瓶颈是什么”。
2. 架构页面回答“机制怎样工作、生命周期和不变量是什么”。
3. 实验页面回答“哪些尝试成功/失败，为什么”。
4. Reference 页面保存可复查的数据、提交和原始路径。
5. 同一事实只设一个详细主页面，其他页面通过相对链接引用。

## 当前快照

本 Wiki 第一版基于 2026-08-25 的 `main`，范围头为 `311ed4c`。后续新增工作应在
[commit map](../reference/commit-map.md) 和相关主题页面同时补充。

---

[返回首页](../README.md) · [时间线](timeline.md) · [证据索引](../reference/evidence-index.md)
