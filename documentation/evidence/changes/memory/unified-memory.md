---
kind: change-history
status: historical
scope: tiered-unified-memory
preserved_in_tag: release-0.9.0
---

# 分层统一内存与专家缓存架构

> 历史目标设计，包含截至 0.6.0 的实现观察和后续构想；0.9.0 的资源分配顺序及不变量见[运行时资源生命周期](../../../architecture/memory/runtime-resource-lifecycle.md)。本文不逐段代表发布版现状。

> 状态：目标架构与实现约束。0.6.0 已具备物理后端解耦、会话级传输路线、弹性 VRAM
> owner 和分段 KV 等基础；专家空间整理、RAM 同构管理与预测预取按本文继续演进。

对应图：[unified-memory-architecture.mmd](assets/unified-memory-architecture.mmd)

## 核心原则

1. 上层只描述计算依赖和资源需求，不选择 ReBAR、DMA、staging 或 CPU copy。
2. 内存管理器只管理逻辑地址、所有权、生命周期和容量，不依赖某一种 Vulkan heap。
3. 物理后端在模型加载前探测能力并冻结传输路线；运行时只按来源和目的位置执行既定路线。
4. KV、runtime、Prefill、Embedding 和 Vision 是高优先级 owner；Expert 是填满剩余空间、可被回收的最低优先级缓存。
5. Expert 权重不可变，SSD/GGUF 是最终真源。所有缓存副本都是 clean，淘汰不承担正确性回写义务。

## 分层

| 层 | 职责 | 不应关心 |
|---|---|---|
| 业务调度层 | Decode、Prefill、Embedding、Vision 的阶段、依赖和并行窗口 | 具体内存类型与搬运 API |
| Pager / 缓存层 | 命中判定、各尺寸类 LRU、专家空间回收、RAM/VRAM/SSD 晋升 | Vulkan heap 和映射方式 |
| 统一内存层 | 分段逻辑空间、owner lease、事务式 claim/release、地址目录 | 模型算子含义 |
| 物理后端层 | VRAM/RAM shard、BDA、ReBAR、staging、host import、DMA、barrier | LRU 与业务策略 |

计算与搬运是两条独立时间线。调度层可以同时提交“计算当前已就绪专家”和“准备缺失专家”，但二者通过显式依赖点汇合。

## 专家比例与回收粒度

### 规划比例

“Expert Cluster（专家簇）”只表示不同尺寸类 slot 的目标比例，不是必须原子分配、回收的物理对象。

一个专家簇由若干尺寸类的 slot 按固定配方组成：

```text
cluster_recipe = { class_0: n0, class_1: n1, ... }
```

配方来自模型的 expert block 组成和 cache planner，不在全局硬编码某个比例。物理布局按该比例平滑交错，避免 KV、ring 或 runtime 总是集中切掉某一个尺寸类。边界处允许少量比例偏差，不为凑齐整簇浪费显存或引入跨 shard 逻辑。

### LRU 与物理空间是两个维度

- 每个 tier、每个尺寸类各有一条 LRU：`VRAM_LRU[class]`、`RAM_LRU[class]`。
- LRU 决定同尺寸候选块的冷热；逻辑 extent 决定哪些物理 slot 挡住本次 claim。
- 清理 ring 等区域时按实际字节和 alignment 规划，不向上取整到完整专家簇。
- 被覆盖的冷块直接淘汰；挡路的热块用 D2D 搬到区域外同尺寸的空闲或冷 slot。
- 搬迁只修改 `slot -> physical extent/address` 目录，不改变热块的 LRU 次序。
- 每个尺寸类始终保留自己的 safety floor；最终比例允许随实际 claim 小幅漂移。
- 只有全部目标 extents 腾空后，统一内存层才发布新的 owner lease。

因此，一个专家 slot 是实际搬迁和淘汰单位；“簇”只是让总体容量保持合理比例的规划概念。同一专家的 U/G/D 也不要求在物理上连续。

## 两级与三级缓存

### VRAM

VRAM 是计算驻留层。底层可以由多个物理 shard 组成，上层看到的是有序的分段逻辑空间，而不是强求一个巨大连续 `VkBuffer`。

```text
低地址                                                        高地址
[ 动态 KV / recurrent state ][ Expert filler / transient ring ][ runtime phase arena ]
```

- **低地址**：KV、QSA/PLE index state 和其他会话持久状态按 32K segment 向上增长。
- **中间**：平时由交错的专家 slot 填满；大 Prefill、Embedding、Vision 需要时按实际空间回收为 transient ring/corridor。
- **高地址**：当前业务一次性 claim 足以覆盖峰值的 runtime phase arena，阶段结束整体释放。
- 未被高优先级 owner 使用的空间立即回到 Expert filler。

“低/中/高”表示稳定的逻辑顺序。单个 extent 不跨物理 allocation；大区域可由多个 extent/bank 组成。

### RAM

RAM 尽量缓存 Expert，不保存第二套 runtime/KV。容量足以容纳全部 Expert 时无需发生 RAM 淘汰；容量受限时，按尺寸类维护 RAM LRU，并沿用同一目标比例做容量规划。

### SSD

GGUF 是完整、不可变的最终真源。RAM miss 从 SSD 读入 RAM slot；必要时也可经 staging 直接晋升到 VRAM。因为权重只读，VRAM/RAM 淘汰均可直接丢弃，后续重新从下一级取得。

## 业务路径

### 解码

1. Router 得到本层实际专家集合。
2. 各尺寸类 LRU 将实际 hit 提升到 MRU；本 dispatch 的活跃集合进入短生命周期 epoch，不能成为本批 victim。
3. GPU 先计算 shared expert 与已驻留 hit；搬运线同时把 miss 从 RAM/SSD 晋升到 VRAM 的冷 slot。
4. 传输完成并发布地址后，GPU 继续计算 miss expert，随后汇合。

不保留跨阶段的永久 pin。短生命周期 epoch 只是防止仍在执行的地址被覆盖，不能用“已经移到 LRU 头部”替代这个正确性约束。

### 小型预填充

专家集合较小时沿用 Decode 路径，避免建立整层 ring 的固定成本。

### 大型预填充

1. 在安全点按实际字节回收足够空间，挡路热专家 D2D 搬迁，建立整层 streaming ring。
2. 按层流水传输和计算，不为复用少量离散 resident expert 增加复杂分支。
3. 阶段结束释放 ring，并从 RAM 批量恢复所需的温专家集合。

KV 在长 Prefill 中仍可跨 32K 边界增长；KV claim 与 ring claim 必须来自预先规划的不冲突逻辑区间，或在同一安全点完成事务切换。

### 嵌入与视觉

- Embedding：请求到来时回收所需 transient 区，SSD 按需加载；输出下载后即可释放。
- Vision：同一请求的全部图片处理完成后释放权重和 runtime；已生成的 LLM 输入表示必须先复制到会话稳定存储。
- 四类业务共享执行门和 owner 协议，不能在仍有 in-flight 地址时改变区域归属。

## 预测预取接口

预测预取属于调度策略，不进入内存后端：

- 计算第 `n` 层 Router 时，可额外产生第 `n+1` 层预测集合。
- 已在 VRAM 的预测块不提升 LRU；缺失块只占用冷端可替换 slot。
- 在下一层真实 Router 完成前持续提交可完成的搬运，此后停止新增任务；已经提交的 copy 不做危险的中途取消。
- 真实命中的预测块才提升到 MRU；误预测块保留在冷端，之后自然淘汰。

该接口允许以后接 FATE 或训练型 predictor，而不改变缓存、地址和传输抽象。

## 事务与不变量

每次 KV 增长、ring 建立、runtime 扩张或辅助模型装载都遵循：

1. 进入 GPU 安全点，停止相关 uploader，并确认旧命令不再引用待改地址。
2. 计算目标 extents、各尺寸类 victim 和搬迁目的地；先验证 dispatch floor 与容量下限。
3. 必要时 D2D 搬迁挡路热块，完成后一次性更新物理目录。
4. 创建 owner lease，最后原子发布地址表；发布前失败必须完整回滚。
5. release 后将空闲 slot 恢复给 Expert cache，不要求恢复原物理位置或精确比例。

必须始终成立：

- 每个尺寸类都保留最宽单批 dispatch 所需的安全 floor。
- 未完成传输的 slot 不能标记为 resident；同一 block 的并发晋升必须合并。
- 搬迁不改变 LRU 语义，预取不冒充真实访问。
- 统计使用实际 committed/resident 字节，所有预算与日志统一使用 GiB/MiB/KiB。
- 传输路线由会话能力计划冻结，但具体 source/destination tier 仍在运行时查询。

## 实施顺序

1. 完成 VRAM 可移动 slot 目录、按字节 claim、D2D 挡路搬迁和事务发布。
2. 将 KV、Prefill ring、runtime、Embedding、Vision 全部收敛到同一 owner API。
3. 为 bounded RAM 建立同构的尺寸类 LRU、软比例容量规划和 SSD 晋升。
4. 最后接预测预取、异步队列与更积极的传输/计算重叠；这些只改变调度，不改变内存所有权模型。
