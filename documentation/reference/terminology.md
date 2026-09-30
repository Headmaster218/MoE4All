---
kind: reference
status: current
scope: documentation-terminology
last_verified: 2026-09-28
verified_commit: ed62393068679573afe94a1472454efe7eae0f15
---

# 文档术语表

本文统一 `documentation/` 的中文叙述风格和技术术语。文档以中文说明为主；业内常用、表达更准确自然的技术词可直接保留英文，例如 Prefill、Decode、GPU、VRAM、RAM、SSD、hit/miss、bubble、cache、kernel、submit。源代码、API、配置键、命令、日志原文、模型名和提交说明中的英文标识始终按原样保留。不要为了“全中文”而生硬翻译术语，也不要因此保留整段普通英文说明。

| 英文术语 | 统一译法 | 使用说明 |
|---|---|---|
| Agent | 智能体 | 产品名称或代码标识中保留原文；首次说明可写“Agent（智能体）”。 |
| arena | 内存区 | 表示统一管理的连续逻辑显存范围；代码标识、日志原文保留 `arena`。 |
| shard | 物理分片 | 一个 arena 可跨多个 Vulkan allocation/backing fragment；逻辑地址连续不要求物理单块连续。 |
| slot | 槽位 | 分配器或缓存中可容纳资源的固定位置。 |
| pool | 池 | 相同块大小与 dtype 几何的 slot 集合；逻辑 pool 可跨多个 shard。 |
| runtime | 运行时 | 指执行环境或运行时资源；Rust 运行库另按上下文说明。 |
| scratch | 临时区 / 临时缓冲区 | 按具体对象选择；避免译作“草稿”。 |
| Prefill | Prefill | 模型处理提示词 token 的阶段；正文和性能指标中保留业内常用写法。 |
| Decode | Decode | 生成 token 的阶段；正文和性能指标中保留业内常用写法。 |
| pager | 分页器 | 管理专家权重在显存、RAM、SSD 间分页的组件。 |
| expert | 专家 | MoE 模型中的专家；`expert cache` 译为“专家缓存”。 |
| expert block / UGD | 专家块 / 完整 UGD | block 可指 Gate、Up、Down 中一个角色矩阵；UGD 指一个专家 FFN 的完整三角色计算。 |
| resident | 常驻 | 表示资源当前驻留在某一级内存中。 |
| shadow | 影子副本 | 包容式缓存中，同一只读专家块在 RAM 保留的 clean 副本。 |
| miss / hit | miss / hit | 缓存术语；需要解释时可写“缓存 miss（未命中）”或“缓存 hit（命中）”。 |
| promotion | 提升 | 指将资源载入更高层级；首次出现可说明“提升（载入并设为常驻）”。 |
| eviction | 淘汰 | 从缓存或内存层移除资源。 |
| batch epoch | 批次保护期 | Router 当前批次所需专家在本轮完成前不可作为 victim。 |
| cold touch | 冷触碰 | 让预取块可用但不把它提升为最热 MRU。 |
| host | 主机 | 与 GPU/device 相对；host memory 译为“主机内存”。 |
| ReBAR / Host DMA | ReBAR / Host DMA | 前者是 CPU 可映射的 device-local VRAM；后者将普通 RAM 原位导入 Vulkan 后由 GPU copy。两者不是同一传输路径。 |
| BDA | Buffer Device Address | shader 使用的 GPU 地址；在正文首次出现时写出全称。 |
| LRU | 最近最少使用策略 | 历史 Pager 的 hit promotion 可为 O(1)；不能把 LRU 与固定角色配额混同。 |
| lane | 通道 | 表示并行执行或缓冲通道。 |
| trace | 执行追踪 | 指有序记录的运行访问过程；避免单独用“轨迹”表示数据文件。 |
| ordered trace / Pager profile | 有序追踪 / 分页器汇总 | 前者保留每次 block access 的顺序，可回放；后者主要汇总 lookup、copy、submit、wait，不能替代顺序记录。 |
| synthetic depth | 合成深度 | 直接构造目标 KV 深度，不包含真实前缀 Prefill；历史深上下文测量必须标注。 |
| conditional RAM hit / combined hit | RAM 条件命中 / 综合命中 | 前者分母仅为 GPU miss；后者分母为全部访问，计入 GPU hit 与其后的 RAM hit。 |
| benchmark | 基准测试 | 结果或方法均使用此译法。 |
| campaign | 优化专项 | 指有目标、有基线和结项记录的一组工程工作。 |
| acceptance | 验收 | 功能或性能达到约定条件的检查记录。 |
| harness | 测试工具 / 测试框架 | 按实际规模选择；不机械音译。 |
| oracle | 参照实现 | 用于正确性或性能对比的实现。 |
| seam | 接口层 / 适配边界 | 按句意选用；不译作“接缝”。 |
| autograd | 自动微分 | `tape autograd` 可写“基于计算记录的自动微分”。 |
| parity | 一致性 | `parity test` 译为“一致性测试”。 |
| warmup | 预热 | 基准测试或运行时准备阶段。 |
| fallback | 回退路径 | 指主路径不可用时采用的替代执行方式。 |
| unified VRAM | 统一显存 | 统一分配器管理的显存资源。 |
| bounded RAM | 有界 RAM | 用户配置上限内的 RAM 缓存层。 |
| full-RAM Host Store | 完整 RAM 专家存储 | 路由专家载荷一次性按 layer-major 布局放入 RAM；不同于有界缓存。 |
| layer ring / loan | 整层 ring / 临时借用 | 前者为 Prefill 异步流送整层；后者让高优先级辅助资源暂占冷专家窗口，释放后按 generation 恢复。 |
| persistent state | 持久状态 | 会话跨 token 保留的 KV 与 DeltaNet/KDA recurrent state；不是只指 KV。 |
| Sinkhorn / CSA / HCA / LID | 专有算子与缓存名 | DeepSeek V4 的 HyperConnection 归一化和压缩注意力/索引缓存，具体含义按模型阶段说明。 |
| tgN / ppN / dK | benchmark 缩写 | Decode N token、Prefill N token、已有或合成的 K 级 KV 深度；必须说明是否真实 Prefill。 |
| inclusive / exclusive cache | 包容式 / 排他式缓存 | 多级缓存语义。 |
| shared expert | 共享专家 | 按模型结构每层参与计算的共享专家。 |
| routed expert | 路由专家 | 由路由器按 token 选择的专家。 |
| throughput | 吞吐量 | token 生成速度可直接写“速度”并保留 `tok/s` 单位。 |
| bubble | bubble | 指设备执行链中的空转间隙；必要时首次写作“执行 bubble（空转间隙）”。 |
| profiler / profiling | 性能分析器 / 性能分析 | 具体工具名和配置键仍保留原文。 |

允许保留的英文还包括自然融入中文句子的业内术语和通用缩写。章节标题、表格标签、方法说明和结论等普通叙述应使用中文；如原文是一条必须逐字引用的日志、提交主题或数据，应保留原文并明确其引用属性。
