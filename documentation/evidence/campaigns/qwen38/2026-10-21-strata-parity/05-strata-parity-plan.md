# MoE4All 追赶 Strata 架构改造计划书 1.1

状态：项目已封存；IQ3_XXS 正确性 G0 和 Vulkan0 prefill 候选已打标签；当前没有可交付 decode G0；日常使用转 Strata  
日期：2026-10-20  
建议分支：`perf/strata-parity`  
建议底座：MoE4All `main`，基准提交 `fb7c578`  
实际仓库：`/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All`  
实际硬件：2 x RX 7900 XTX、Intel i7-10700K、64 GiB RAM  
计划硬件偏差：原计划写 Ryzen 5 5600X，实际为 i7-10700K，CPU 微基准不能直接外推  
目标模型：SC117 Qwen3.8-Flash-Next IQ3_XXS，2 个分片  
历史诊断模型：Q4_K_M-M64，仅保留旧证据，不再作为 G0、目标或验收依据  
冻结二进制：`infr 1.0.3`，SHA-256 `8ba3f1e4dbe85557607f0c86f0220097331c5dbce27193bb3df6359abd18e81e`  

## 0. M0 收尾时间盒

开始时间：2026-10-20  
时间盒：最多 2 个工作日或 2 轮完整实验，以先到者为准。  
本轮只允许两项硬验证：

1. M0-12：在相同 Ubuntu、RADV、i7-10700K、Vulkan2、发布模型和固定配置下构建并跑 `fabf3e614`，对跑 4K 和 30K。
2. 30K 预填瓶颈限时定位：只检查最可能的 2 到 3 个固定开销假设。

执行规则：

- 两项完成或时间盒到期后，强制冻结本机 G0，不再扩展 M0。
- 冻结状态写为“正确性基线通过，性能基线按时间盒冻结”。
- 如果确认 1.0.3 相对 `fabf3e614` 存在预填回归，立即停止时间盒，先修回归，不进入 M1a。
- 如果无法构建、无法对跑或不能解释 6.8 倍差距，写“证据不足”，按时间盒到期冻结并进入 M1a。

## 1. 这份计划书怎么用

这份文件不是一次性提案，而是后续实施的主计划。每个阶段完成后，都在本文更新：

- 当前状态：未开始、进行中、通过、暂停、否决。
- 实际提交号和二进制哈希。
- 实测命令、模型、上下文、并发数和输出哈希。
- 证据文件路径。
- 与上一阶段的对比结果。
- 如果被否决，写清原因是性能、正确性、稳定性还是工程成本。

后续每次继续工作前，先看“当前入口任务”，不要从旧聊天记录重新推断方向。

### 当前进度总览

| 阶段 | 状态 | 当前任务 | 最后更新 | 证据 |
| --- | --- | --- | --- | --- |
| M0 基线 | 目标重写，未通过 | IQ3_XXS 单路 30K 仍分叉，必须先修正确性 | 2026-10-20 | `moe4all-m0-execution-report.md` |
| M0-7 容量 | 不通过 | 全专家锁页方案不可行 | 2026-10-20 | `m0-7-budget.jsonl` |
| M0-8 CPU 专家 | 有条件通过 | 只证明单行概念 | 2026-10-20 | `m0-8-cpu-experts.log` |
| M0-9 双卡通信 | 有条件通过 | 只传小 residual | 2026-10-20 | `m0-9-dual-gpu.log` |
| M0-10 回滚下界 | 有条件通过 | 还要做真实窗口回滚 | 2026-10-20 | `m0-10-rollback.log` |
| M0-11 口径复核 | 通过 | G0 必须重写 | 2026-10-20 | `m0-11-scope-review.md` |
| M0-12 旧版对跑 | 通过 | fabf3e614 与 1.0.3 性能一致 | 2026-10-20 | `fabf3e614-30k-r3-warm1/` |
| M1a 容量兜底 | 已实现，实机 fallback 未触发 | 提交 `2cbd0ed`，仅受限 pool 单测覆盖 | 2026-10-20 | `benchmark-data/artifacts/strata-parity-m1a-20261020/` |
| M1b 热点 KV | 已启动 | `8791f7d` 配置/选择；`91c014c` segment host/VRAM 分流 | 2026-10-20 | `benchmark-data/artifacts/strata-parity-m1b-20261020/` |
| M2 专家并行 | 已暂停 | CPU GEMM 否决；Q2_0 质量门不通过 | 2026-10-20 | `m2-p2-q2-quality.md` |
| M3 窗口流水 | 未开始 | 无 | - | 无 |
| M4 动态预测 | 未开始 | 无 | - | 无 |
| M5 输入和稳定 | 当前入口 | 先做预填停顿/输入阶段可测改造 | 2026-10-20 | 待建立 |

### 关键决策记录

| 编号 | 决策 | 状态 | 结论 |
| --- | --- | --- | --- |
| D0 | 是否以 1.0 稳定路径为新架构底座 | 建议接受 | 是，旧实验分支只作为来源 |
| D1 | KV 页大小、缓存比例和淘汰策略 | 待 M1b | 未决定 |
| D2 | CPU 与显卡同时计算专家是否足以继续窗口流水 | 待 M2 | 未决定 |
| D3 | 窗口流水是否达到 70 token/s 且正确 | 待 M3 | 未决定 |
| D4 | 动态预测是否进入默认路径 | 待 M4 | 未决定 |
| D5 | 30K 贪心输出不复现是否阻断所有性能优化 | 已定位 | 发布模型单路通过；分叉限定到 IQ3_XXS |
| D6 | 是否允许“全专家锁页 + 2 x 256K KV”原方案 | 已决定 | 不通过，62 GiB 物理内存装不下 |
| D7 | CPU 专家是否继续 | 有条件继续 | 先补多行、真实模型对拍和输出一致性 |

## 2. 执行摘要

结论：可以追到 Strata 的主要效果，但必须改执行架构，不能继续只调参数或只做单项融合。

现实目标：

- 32K 到 200K 单路稳定输出达到 45 到 50 token/s。
- 32K 到 200K 输入速度达到 2,000 token/s 左右。
- 短聊天稳定达到 70 到 85 token/s。
- 代码等可预测场景争取达到 100 token/s 以上。
- 256K 可以加载、推理和并发，不因显存差几 MiB 直接失败。
- 所有优化默认保持近似固定输出的正确性；贪心模式下输出哈希必须与基线一致。

不作为硬目标：

- 不承诺普通聊天都达到 130 到 145 token/s。
- 不把 Strata 作为后端或外层代理。
- 不一次性移植 Strata 的全部细节。
- 不牺牲现有 1.0 发布路径的稳定性。

130 到 145 token/s 是 Strata 在特定代码输入、短上下文和调优配置下的峰值。当前更稳定的 Strata 长上下文结果约为 47 到 50 token/s，所以本计划以稳定长上下文性能为主要目标，以短上下文峰值为第二阶段目标。

## 3. 历史基准

### 3.1 Strata 观测值

| 场景 | 结果 | 备注 |
| --- | ---: | --- |
| 32K 输入 | 2,191 token/s | 全新长提示 |
| 64K 输入 | 2,253 token/s | 全新长提示 |
| 200K 输入 | 2,140 token/s | 全新长提示 |
| 32K 稳定输出 | 47.8 token/s | 输出 128 token |
| 64K 稳定输出 | 49.5 token/s | 输出 128 token |
| 200K 稳定输出 | 46.7 token/s | 输出 128 token |
| 短聊天输出 | 57 到 75 token/s | 一次运行中位数 61.6 |
| 代码调优峰值 | 96 到 145 token/s | 依赖输入、预测命中和配置 |

Strata 也不是完全稳定。一次运行中的 32K 新提示测试因为显存只剩 644 MiB，等待 GPU 60 秒后触发停机；后续另一组 32K、64K、200K 测试全部完成。因此本计划必须同时追性能和稳定性。

### 3.2 MoE4All 1.0 稳定路径

| 模型和场景 | 输入 | 输出 |
| --- | ---: | ---: |
| Flash-Next 30K 单路 | 1,415 token/s | 50.4 token/s |
| Flash-Next 150K 单路 | 1,446 token/s | 44.2 token/s |
| Qwen3.6-35B 30K 单路 | 2,749 token/s | 80.5 token/s |
| Qwen3.6-35B 150K 单路 | 1,169 token/s | 41.9 token/s |

这些是 1.0 发布路径的成绩。它们说明 MoE4All 在长上下文输出上已经接近 Strata，主要差距在输入速度和部分高并发、可预测场景。

注意：上述 1.0 发布数据来自单张 RX 7900 XTX。Strata 的 32K 到 200K 长上下文记录来自双卡配置。两者只能用于判断差距方向，不能直接当作严格同硬件对照。本计划的新架构目标以双卡主机为主，同时要求单卡稳定路径不退化。

### 3.3 实验双卡 256K 路径

历史实验记录中出现过更低的数字：

- 端到端输出约 11.3 token/s。
- 修正 KV 额度后，bench 输出约 16.7 token/s。
- 输入约 683 token/s。

这组数字不能代表 1.0 稳定路径，但能说明旧实验分支的页式专家执行、KV 放置和解码图复用还没有成熟。

### 3.4 对照限制

以下差异使 Strata 与 MoE4All 不能做严格逐位对比：

- 固定投影权重格式不同。
- 主机内存预算和专家驻留策略不同。
- 部分 Strata 数字来自不同运行实例和配置。
- 提示、预热、统计区间和并发方式不完全一致。

因此本文把 Strata 当作硬件能力和架构方向参考，不把它当作必须逐项复刻的唯一标准。

### 3.5 非发布模型的早期探测结果

本节数据来自 `Qwen3.8-Flash-Next-IQ3_XXS`，不是 1.0 发布的 `AD-4.27bpw-Q4_K_M-M64`。保留它们只用于说明模型格式差异，不能作为 G0 性能基线。

冻结条件：

- 分支：`perf/strata-parity`
- 提交：`fb7c5784908e8cfe6754d32826a70ffa4ed902db`
- 二进制：`8ba3f1e4dbe85557607f0c86f0220097331c5dbce27193bb3df6359abd18e81e`
- 运行设备：`Vulkan2`，RX 7900 XTX
- KV：Q8，自动放置
- 预测：关闭
- 专家：1.0 自动 GPU 热专家缓存，不是全专家常驻

| 场景 | 预填 | 稳态输出 | 首个 token | 进程内存峰值 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4K 普通聊天，单路 | 180.7 token/s | 6.5 token/s | 22.3 s | 49.6 GB | 完成 |
| 4K 代码，单路 | 195.8 token/s | 7.0 token/s | 20.3 s | 49.6 GB | 完成 |
| 4K 唯一文本，KV 主机 | 183.4 token/s | 5.0 token/s | 21.7 s | 49.7 GB | 完成，慢于显存 |
| 4K 唯一文本，KV 显存 | 191.1 token/s | 7.4 token/s | 21.0 s | 49.6 GB | 完成 |
| 4K 固定输入，三轮 | 181.3/251.2/251.2 token/s | 7.2/35.5/49.4 token/s | 22.0/15.7/15.7 s | 49.8 GB | 输出哈希稳定 |
| 30K 唯一文本，三轮 | 229/241/241 token/s | 10.4/15.5/15.4 token/s | 131/124/124 s | 49.7 GB | 完成，冷启动明显慢 |
| 30K 固定输入，三轮 | 230/242/242 token/s | 10.2/16.7/15.8 token/s | 130/123/123 s | 49.3 GB | 输出哈希不稳定 |
| 150K 唯一文本 | 到 4K 时仅约 61 token/s | 未完成 | 未完成 | 未完成 | 估算单轮超过 40 分钟，停止 |
| 256K 单路 | 未运行 | 未运行 | 未运行 | 未运行 | 容量门先失败 |

该模型的发现：

1. 4K 固定输入连续三轮输出哈希一致；30K 固定输入同样参数、同 seed、`temperature=0`，输出从第 56 个字符后分叉，三次哈希不同。
2. `--parallel 2` 在 4K 即失败，错误为 `parallel prefill requires Vulkan GPU embedding`。
3. MTP K=3 在启动时失败，错误为 `Qwen3.8 MTP verify requires the backend's GPU embedding gather`。
4. 强制 `paging.cache=2GiB` 无法放置预填块；当前可用路径是自动专家缓存，而不是“全专家显卡常驻”。
5. 该模型的 token embedding dtype 是 `Iq3S`，当前 GPU embed-gather 不支持它，因此并发和 MTP 会因为 `gpu_embed=false` 失败。

### 3.6 1.0 发布模型最小基线

模型：`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`，33 个分片，88.034 GiB。

协议：

- 单卡 `Vulkan2`
- 单路
- 贪心，`temperature=0`，`seed=1`
- MTP 关闭
- Q8 KV
- KV overflow 关闭
- 专家使用 1.0 自动 GPU 热专家缓存
- 固定 `ubatch=4096`、线程 10
- 丢弃 1 轮预热，随后测量 3 轮
- 页面缓存保持预热，不主动清空
- GPU 性能档位未能固定，驱动当前为 `auto`，且设置 high 需要 sudo

| 场景 | 预热轮 | 测量三轮 | 输出 | TTFT | 哈希 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4K，单路，32 token | 151.2 | 214.1/214.9/214.7 | 40.08/39.97/40.04 | 18.4/18.3/18.3 s | 稳定 |
| 30K，单路，32 token | 198.3 | 207.4/207.5/207.5 | 42.62/42.70/42.72 | 144.3/144.2/144.2 s | 稳定 |
| 4K，两路并发 | 49.3/49.3 | 54.2/54.2 | 12.8/12.8 | 72.9/72.8 s | 两路各稳定，但跨路不同 |

结论：

- 30K 正确性在发布模型、单路、固定协议下通过。
- 早期 30K 分叉属于 IQ3_XXS 模型路径，不是 1.0 发布模型的通用基线。
- 预填约 207 token/s，和发布记录的 1,414.75 token/s 差距很大，但发布记录使用 Windows、Ryzen 5 5600X、1.0.0 二进制和不同输出统计窗口，不能直接判定回归。
- 两路并发可以启动，但相同输入的两路输出哈希不同，并发确定性仍需作为正确性风险单独处理。

## 4. 目标和非目标

### 4.1 主要目标

1. 建立同一模型、同一硬件、同一提示、同一统计口径的基线：1.0 稳定路径、旧实验路径、Strata 参考。
2. 把原始 KV 改为内存保存完整内容、显卡保存热点页、按 QSA 选择和近期访问动态换入。
3. 把专家改为显卡算热点、CPU 同时算未命中的专家，不再只等数据搬进显卡。
4. 把两张卡改成相邻两批输出同时推进，减少一半时间空闲。
5. 把预测方式改成按场景动态选择，不再固定开或关 MTP。
6. 把输入阶段的专家搬运、注意力和大块提交重叠起来。
7. 完成后把性能提升合并到默认路径，保留一键回退开关。

必须达到：

- 30K 固定输入的贪心输出哈希连续三次一致。
- 两路并发能启动并完成请求，跨路确定性策略必须明确。
- MTP K=3 需要兼容的 sidecar 和 embedding gather；没有工件时明确标为不支持，不阻塞基线。
- 150K 单路能够在可接受时间内完成，并给出真实预填和输出速度。
- 256K 容量方案在 62 GiB 物理内存内成立，不能依赖不存在的内存。

争取达到：

- 32K 到 200K 单路稳定输出 45 到 50 token/s。
- 32K 到 200K 输入速度 2,000 token/s 左右。
- 短聊天 70 到 85 token/s。
- 可预测代码场景 100 token/s 以上。

### 4.2 正确性目标

- 贪心输出哈希与串行基线一致。
- 页式 KV 在单路和双路下结果一致。
- CPU 专家计算结果与显卡路径一致。
- 窗口预测失败时能完整回滚，不污染下一个窗口。
- 所有小数值误差必须能解释来自量化或顺序变化，不能靠容差掩盖错误。
- M1a/M1b 的正确性门：每个 lane 单独重复时输出可复现。
- 跨 lane 相同输入 bit-identical 是独立优化项，不阻塞 M1a/M1b；当前发布模型两路已完成但跨 lane 哈希不同。
- 不得通过默认关闭并发或降低单 lane 正确性要求来获得跨 lane 一致。

### 4.3 非目标

- 不在第一阶段改模型结构。
- 不在第一阶段支持所有模型。
- 不在第一阶段重写全部 GPU 内核。
- 不为了速度默认开启会导致输出不稳定的 MTP 或专家跳过。
- 不删除现有稳定路径。

## 5. 总体架构

目标架构由四个面组成：

### 5.1 内存面

完整 KV 放在主机内存，显卡只保留热点 KV 页。热点判定来自：

- QSA 当前选择的块。
- 最近若干输出步重复选择的块。
- 最近的尾部 token。
- 每层固定需要保留的注意力锚点。

显卡和主机不再要求“全部专家保存在锁页内存”。目标是：

- 热专家常驻显卡或锁页主机。
- 冷专家分页或从 mmap/SSD 流式读取。
- 非专家权重保持可分页，不要求全量常驻。

### 5.2 计算面

每个 MoE 层尽量做到：

- 显卡计算命中的专家。
- CPU 同时计算未命中的专家。
- 独立搬运线程准备下一层需要的数据。
- 当前层只等待当前层必须的数据，不等待下一层。

### 5.3 调度面

解码阶段维护两个相邻窗口：

- 窗口 N 正常验证。
- 窗口 N+1 使用预测 token 提前跑第一段。
- 窗口 N 完成后比较预测前缀。
- 预测正确就保留窗口 N+1 的前半段。
- 预测错误就回滚循环状态并重建窗口 N+1。

### 5.4 预测面

预测控制器按场景选择：

- 不预测。
- MTP 预测。
- 后缀或文本复制预测。

选择依据是预计每次提交能真正产生多少 token，以及生成相同 token 需要多少时间。

### 5.5 CPU/GPU 跨路径输出一致性策略

优先级从高到低：

1. CPU 和 GPU 都按固定 expert id 顺序合并，禁止并行浮点累加导致顺序漂移。
2. 单专家输出先做 bit-exact 对拍；量化内核确实无法 bit-exact 时，记录最大绝对误差和相对误差。
3. logits/top-k 容差只能作为诊断，不能单独作为生产放行条件。
4. 生产门要求固定 greedy 语料上的输出哈希一致；如果做不到，必须默认关闭 CPU/GPU 混算。
5. 如果只能在 top-1 margin 足够大时保证一致，则加入 margin 回退：接近分叉时停止 CPU 分支并走稳定路径。

## 6. 阶段和里程碑

### M0：建立可信基线

目标：在开始改动前冻结可重复的对比环境。

任务：

| 任务 | 交付物 | 验收 |
| --- | --- | --- |
| M0-1 | 从 `fb7c578` 创建 `perf/strata-parity` | 分支干净，提交号记录 |
| M0-2 | 固定模型、量化、驱动、系统、配置 | 清单写入证据目录 |
| M0-3 | 建立 32K、64K、150K、200K、256K 测试矩阵 | 单路和双路结果完整 |
| M0-4 | 固定提示和输出长度 | 输出哈希可重复 |
| M0-5 | 记录显存、内存、交换区和命中率 | 每轮都有完整数据 |
| M0-6 | 复现 1.0 稳定路径数据 | 30K 和 150K 与发布数据同档 |
| M0-7 | 256K 双会话 + 全专家锁页预算 | 给出可复算预算表 |
| M0-8 | CPU 专家计算与 GPU 搬运微基准 | 给出 CPU/GPU 成本和重叠判断 |
| M0-9 | 双卡层交接和小状态传输 | 给出通信成本是否吃掉收益 |
| M0-10 | 窗口快照、恢复和 1,000 次失败回滚 | 给出回滚下界和停止条件 |
| M0-11 | 1.0 发布数据口径与模型分片复核 | 明确是否为同一 A/B |
| M0-12 | `fabf3e614` 同机 4K/30K 对跑 | 判断 1.0.3 是否预填回归 |

证据要求：

- 二进制哈希。
- 完整启动命令。
- 原始日志。
- 每轮 JSON 结果。
- 输出文本哈希。
- 显存和内存峰值。

退出条件：

- 同一配置连续三轮的主要速度波动不超过 5%。
- 固定输入连续三轮输出哈希稳定。
- M0-7 到 M0-10 都有通过、不通过或证据不足的明确结论。

M0-7 容量预算，按 1.0 发布模型 `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64` 重新计算：

| 项目 | 大小 |
| --- | ---: |
| 完整模型分片 | 88.034 GiB |
| routed 专家张量 | 47.168 GiB |
| 非专家张量 | 40.855 GiB |
| 256K 单会话 KV | 4.642 GiB |
| 256K 双会话 KV | 9.285 GiB |
| 47.168 GiB 专家 + 9.285 GiB KV | 56.453 GiB |
| 再加完整非专家张量的最坏情况 | 97.528 GiB |

上一版 M0 报告使用的是 IQ3_XXS 模型，专家预算为 39.966 GiB，不是 1.0 发布模型。新的容量结论必须以 47.168 GiB 专家池为准。

1.0 实际驻留策略是：47.168 GiB routed 专家进入主机侧 store，非专家权重主要走 mmap/页缓存，显卡保留自动热专家缓存。发布模型 30K 最小基线实测进程峰值约 55.6 GB（约 51.8 GiB），其中已经包含约 47.17 GiB 专家和约 0.87 GiB 单 slot KV。

四列预算：

| 类别 | 内容 | 256K 双会话建议 |
| --- | --- | ---: |
| 必须常驻 | KV、运行时、热专家工作集 | KV 9.285 GiB + 运行时约 5.2 GiB + 热专家 16 到 24 GiB |
| 可分页 | 其余专家权重 | 23 到 31 GiB，按热度和 CPU/GPU 需求换入 |
| 可流式 | 非专家权重页、冷专家 | 不全部常驻，基于 mmap/SSD/流式路径 |
| 预留余量 | OS、驱动、碎片、GC、突发 scratch | 至少 8 GiB |

结论：全专家锁页、完整非专家权重常驻和 2 x 256K KV 同时存在时，需要约 97.53 GiB，超过 62 GiB 物理内存和 48 GiB 计划预算。原方案不通过。

预算表不再使用“全部专家保存在锁页内存”。改为“热专家锁页 + 冷专家分页/流式 + 非专家权重流式”。

M0-8 微基准：

- i7-10700K、8 个 CPU 工作线程。该 CPU 与计划中的 Ryzen 5 5600X 不同，结论只对当前主机成立。
- IQ3_XXS 合成工作集：约 1.75 MiB/专家，p50 114.7 微秒，p90 121.5 微秒，约 8,722 专家/秒。
- 1.0 发布模型真实权重探针：layer 4 单 miss p50 约 125.9 微秒，layer 0 单 miss p50 约 144.6 微秒；2 miss 约 193 到 246 微秒。
- 同一机器 1 MiB 主机到显卡搬运约 1.04 毫秒，64 MiB 约 53.8 毫秒。
- 四行 CPU miss 被 `geometry` 明确拒绝；测试 `m0_four_row_cpu_miss_is_explicitly_unsupported` 通过。
- 结论：单行 CPU 专家值得继续；四行 MTP 仍不可用，真实输出 parity 和 CPU/GPU 合并一致性尚未完成。

四行约束位置：`crates/infr-vulkan/src/cpu_miss.rs` 的 `geometry` 要求 `rows == 1`，并只允许 1 到 3 个 miss。

最小可行方案评估：

- 短期方案：对四行分别提交单行任务，保持固定 expert id 和 row 顺序合并；会增加调度开销，但可先验证正确性。
- 正式方案：扩展 Worker 输入/输出为多行 batch，复用同一 expert 权重读取，减少重复访存。
- 本轮不实现；列为 M2 前置任务。结论只对 i7-10700K 成立，Ryzen 5 5600X 未复测。

M0-9 微基准：

- Vulkan0 和 Vulkan2 都支持 dma-buf P2P。
- 64 KiB P2P 约 0.87 GiB/s，单次约 72 微秒；主机中转约 151 微秒。
- 发布模型宽 residual 为 `(1 + hc_mult) * n_embd * 4`，单行 51.2 KiB，四行约 204.8 KiB。
- 256 KiB P2P 约 1.88 GiB/s，单次约 137 微秒；主机中转约 278 微秒。
- 1 MiB P2P 约 2.68 GiB/s；64 MiB P2P 约 3.07 GiB/s。
- 64 MiB 主机中转约 53.8 毫秒，64 MiB P2P 约 20.8 毫秒。
- 30K 单路输出约 42.7 token/s，单 token 约 23.4 毫秒；一次 205 KiB P2P 边界传输约占 0.6%。
- 每窗口只在层切分边界传 1 次 residual，不是每层传一次。
- 结论：边界只传小 residual 时远低于 20% 层时间阈值；绝不能在每个窗口传完整 KV 或完整循环状态。

M0-10 回滚下界：

- 循环状态总量约 118,038,528 字节。
- 1,000 次设备内快照 + 恢复总耗时约 680 毫秒，即每轮 0.680 毫秒。
- 1,000 次后再污染并恢复，字节完全一致，`restored_exactly=true`。
- 结论：纯设备复制不是主要成本；真实窗口回滚中的 KV 重写、调度同步和状态污染仍未测，必须补端到端回滚。

M0-11 口径复核：

- 发布记录：`infr 1.0.0`，运行源 `fabf3e614`，Windows 11，Ryzen 5 5600X，单卡 RX 7900 XTX，Flash-Next AD-4.27bpw-Q4_K_M-M64，33 分片，512 token，1 预热 + 3 测量轮。
- 当前：`infr 1.0.3`，提交 `fb7c578`，Ubuntu 22.04/RADV，i7-10700K，单卡 Vulkan2，同一发布模型，32 token，1 预热 + 3 测量轮。
- 结论：不是同一 A/B。G0 必须重写为“1.0.3 / Linux / i7 / 本机驱动”的基线，不能把当前数字直接称为 1.0 回归。
- 严格回归对比需要另建 `fabf3e614` 独立二进制，并用相同模型、prompt、上下文、输出长度和 harness 对跑。

M0-12 同机对跑：

- 旧源码：`fabf3e614a56efed66dfaa89d4fac3a10ec0c40b`
- 构建：默认 release，`cargo build --release -p infr-cli --bin infr`
- 二进制显示版本：`infr 0.11.0`，SHA-256 `82750cb29f46fc319dc2d768391c94b8f6e98a776fe99ab8265e05dfc14e32eb`
- 相同 Ubuntu、RADV、i7-10700K、Vulkan2、发布模型、1 预热 + 3 测量、32 token。
- 4K 旧版：预填约 213.4，输出约 39.2。
- 30K 旧版：预填 207.42/207.45/207.45，输出 42.70/42.71/42.38。
- 30K 当前 1.0.3：预填 207.36/207.48/207.47，输出 42.62/42.70/42.72。
- 两者的 30K 哈希均为 `bfccea41ad5424cee5ab0fc28542d7c82f591383f8902619e0c45b4b57b2f76f`。
- 判定：没有发现 1.0.3 预填回归。约 207 token/s 是本机环境/发布口径差异下的共同结果，不是 `fb7c578` 引入。
- 因此 G0 冻结为本机 1.0.3 口径，发布 1.0.0 数据仅作历史参考。

30K 预填瓶颈限时结论：

- 假设 A：`fb7c578` 引入 1.0.3 回归。结果：**排除**。`fabf3e614` 和 1.0.3 都是约 207 token/s。
- 假设 B：固定 per-chunk 开销。结果：**支持**。服务统计显示 GPU 活跃窗口约 815 token/s，但相邻 4K chunk 间存在数秒到十余秒的 `prefill_tps=0` 间隔，导致完整请求平均只有约 207 token/s。
- 假设 C：Windows 1.0.0 发布配置与 Linux/RADV 当前环境不同。结果：**证据不足**，但这是剩下的主要环境差异；当前时间盒不扩展到 Windows 对照。
- 未能在时间盒内把 per-chunk 间隔进一步拆成 CPU 准备、页缓存、同步或提交成本。按规则冻结，不再延迟 M1a。

M1b 前置补测：

- 修正 profiling 开关为 `INFR_PROF_STAGES=1`。
- 30K 单路逐 chunk：build 0.1-0.2 ms，compile 0.1 ms；4K chunk execute 约 17.95 秒。
- `device.submit_dispatches=0` 的 4K A/B execute 约 18.18 秒，和默认路径相同，排除 submit 切分粒度主因。
  - 停顿类别：不在 chunk 调度、graph build、compile 或 submit splitter；已定位在 `execute` 内部。
  - 修正后的 op-walk 计时显示：4K 最后一个 4,029-token chunk 为 `total=18.15s, paged_moe=16.15s, lower_op=0.23s, other=1.76s`；30K 稳态 4,096-token chunk 的 paged MoE 约 15.5-17.5 s。
  - 4K paged profile 进一步显示：48 层共搬运 45,225 MiB，copy/record 口径 16.74 s，staging wait 11.81 s；有效搬运率约 2.8 GiB/s。
  - 该层满足 `touch_all`，实际搬运量接近模型全部 routed expert 权重（47.17 GiB），因此重复专家去重不是第一收益点。
  - 同步上传 A/B：异步 165.1 token/s，同步 132.6 token/s，输出哈希相同；异步上传器有价值，但仍有 11.81 s consumer wait。
  - 独立 H2D/staging 微基准：1/2/4/8 条流、256 KiB/1 MiB/4 MiB/40 MiB 块全部稳定在约 2.6 GiB/s，增加流数不扩展；mmap 与普通 malloc 源结果一致。
  - 推算：45 GiB 在当前路径约 17 秒；4K chunk 要支撑 2,500 token/s，只有 1.6 秒预算，等效需要约 28 GiB/s。扩大 ring/lane 不是充分解，必须改变专家驻留或计算位置。
  - 同一运行的 recorder 明细中，dispatch 编码、descriptor 绑定和 barrier 都只有毫秒级；因此不能再说主因是 CPU dispatch 记录。
  - 结论：**当前 prefill 主项是 paged expert MoE 的专家搬运/暂存路径**；CPU 命令编码不是主项。
  - 证据：`benchmark-data/artifacts/strata-parity-m1b-20261020/opwalk-classification.md`、`prefill-opwalk-4k/`、`prefill-opwalk-30k/`、`prefill-paged-profile-4k/`、`prefill-paged-sync-4k/`、`prefill-bandwidth-probe.log`。
- 因此 150K 继续禁用；先决定专家驻留/搬运路线，再谈解禁。

### M1a：紧急容量兜底和并发前置门

目标：先让现有路径在容量边界和两路并发上不再直接失败，不引入淘汰算法。

任务：

| 任务 | 主要代码 | 交付物 |
| --- | --- | --- |
| M1a-1 | `crates/infr-llama/src/seam/session_state.rs` | 分段 KV 增长失败时退回主机，而不是终止会话 |
| M1a-2 | `crates/infr-vulkan/src/lib.rs` | 增长事务预留不足时返回可恢复错误 |
| M1a-3 | 两路并发确定性 | 明确相同输入跨 lane 哈希不同是否接受；不接受则加入 lane 级确定性策略 |
| M1a-4 | MTP 前置 | 提供兼容 sidecar；对 IQ3_S embedding 补 GPU gather，或明确排除该模型 |
| M1a-5 | 30K 输出复现 | 发布模型已通过；保留 IQ3_XXS 差异作为模型格式限制 |

M1a 停止条件：

- 发布模型 30K 固定输入仍不能连续三次哈希一致：停止所有性能优化，优先修正确性。
- 两路并发跨 lane 不确定策略未决定：不进入 M2 并发实验。
- MTP 前置未解决：M3/M4 不能以 MTP 为默认路径。
- KV 增长仍会直接杀死会话：不进入 M1b。

M1a 验收：

- 两个并发会话 32K 增长成功。
- 单路贪心输出哈希与本机 G0 一致。
- 记录显存、内存、交换区和主机回退次数。
- 跨 lane bit-identical 不作为 M1a 门禁，但必须继续记录。

M1a 实施结果：

- 提交：`2cbd0ed fix(kv): fall back segmented growth to host memory`
- 二进制：`7bdbbd1be88b6146b585549140c3263a71c6dc3f307f1a188532489ea677e656`
- VRAM 预留/claim 失败时，新增长批次整体落到主机内存，后续增长保持 host-only。
- 已有 VRAM 段不移动，不增加淘汰或替换算法。
- 强制受限 pool 单测通过：TVRAM KV 保持 2 MiB，4 MiB 新段进入主机，段表地址非零。
- 两路 39,835-token 请求均跨过 32K，提交到 65,536 段并完成；未出现 VRAM 预算终止。
- 两路 1-token 输出哈希相同：`983987033f0e117011e531dc33ad9bb15290bba41a414d830fb5cbdbcda2ff17`。
- 单路 G0 prompt 回归哈希仍为 `100f0e2e0808507cbb95ff0ce53e7481f3e83de86202eeebd536e7398b5e9eac`。
- 实机两路 40K 未触发 host fallback（VRAM 足够），fallback 路径由强制受限 pool 单测覆盖。
- 额外尝试两路 4K、每 slot 262K 最大布局：两路完成，但未打印 host fallback；预填降至约 6.8 token/s。该场景证明最大布局预留会造成严重退化，但仍未实机覆盖 fallback。
- M1a 遗留缺口：真实服务路径的 host fallback 尚未强制触发；只完成真实 GPU 受限 pool 单测覆盖。

M1 并行前置任务，MTP sidecar：

- 发布模型目录缺少 MTP sidecar，这是**数据工件缺失**；当前运行时的 MTP 验证实现存在。
- IQ3_XXS 有 sidecar，但 `token_embd.weight=Iq3S` 缺少 GPU embed-gather，这是**实现覆盖缺失**。
- 获取路径：取得与发布模型同一 tokenizer/vocab 和层几何的官方 MTP head，验证 `n_layer_nextn`、tensor shapes 与 sidecar 加载。
- 生成路径：从匹配的 MTP 训练头/转换工具生成 sidecar，完成 greedy parity 后才能使用。
- 责任任务：M1p-MTP，独立于 M1a；MTP 不进入 M0 G0。
- 若短期无法取得 sidecar，M3/M4 必须准备无 MTP 的 lookup/无预测退化方案。

### M1b：热点 KV

目标：在容量兜底通过后，解决 KV 全放内存太慢、全放显存容易失败的问题。

任务：

| 任务 | 主要代码 | 交付物 |
| --- | --- | --- |
| M1b-1 | `crates/infr-vulkan/src/lib.rs` | KV 页表支持主机页和显卡页 |
| M1b-2 | `crates/infr-llama/src/seam/weights.rs` | 完整 KV 主机备份和显卡热页视图 |
| M1b-3 | `crates/infr-llama/src/seam/session_state.rs` | 会话增长按页提交并支持主机回退 |
| M1b-4 | QSA gather 路径 | 使用当前 QSA top-k 触发提前换入 |
| M1b-5 | KV 管理器 | 支持按层统计访问次数和命中率 |
| M1b-6 | 配置层 | 支持固定比例、自动比例和关闭热点缓存 |

策略：

- 先测试显卡缓存比例 0%、10%、20%、30%、40%、50%、75%、100%。
- 同时测试页大小 128、256、512 token。
- 默认值取命中率曲线出现明显拐点的位置。
- 两个并发会话不能各自假设拥有 50% 显存，必须共享总预算。

M1b 性能比较口径：

- 只相对本机 1.0.3 G0 和同机“KV 全主机”路径。
- 不相对 Windows 1.0.0 发布数据。
- 200K 目标为相对同机全主机路径至少 2 倍；未达到时必须给出命中率、PCIe 字节和等待时间。
- 单路贪心输出哈希必须与本机 G0 一致。

M1b 停止条件：

- 30K 或 150K 固定输入输出哈希不稳定：禁止启用热点缓存为默认。
- 两路会话在 32K 增长失败：先回 M1a。
- 热点缓存相对全主机路径不足 1.5 倍：重新评估页大小和收益，不进入默认。

决策点 D1：选择页大小、默认缓存比例、淘汰策略，并决定是否继续进入 M2。

M1b 已启动的范围：

- `KvHotCacheMode::{Off, Fixed, Auto}`。
- `INFR_KV_HOT_CACHE`、`INFR_KV_HOT_CACHE_RATIO`、`INFR_KV_HOT_CACHE_PAGE_TOKENS`。
- 按累计访问分数选择热点页；同分按 page id 稳定排序。
- 默认 `off`，不改变 1.0 路径。

M1b 首个提交：

- `8791f7d feat(kv): add deterministic hot page selection policy`
- 二进制：`4df74bdb71c8093f6f51f2d7a4697d14e1d248f70d77eee3e9dc43d2a09ac61c`
- `infr-core` 热点页测试通过，44 个 config-layer 测试通过。

M1b 第二个提交：

- `91c014c feat(kv): split segmented kv into hot and host tiers`
- 在现有 32K segment 地址表上增加 host/VRAM 分流，不引入第二套 shader 表 ABI。
- `off` 保持原 1.0 路径；`fixed` 按比例把稳定前缀 segment 放 VRAM，其余 segment 落主机；`auto` 仍保持冷启动，避免悄悄等同 `fixed`。
- 真实 Vulkan2 定向单测：4 个 1 MiB segment、`fixed=0.5` 时为 2 VRAM + 2 host；已有 host fallback 单测也通过。
- 尚未完成：canonical host backup、QSA top-k 触发、热页动态换入、命中率/读取字节/等待时间。

M1b 尚未实现：

- 完整 KV 主机备份。
- QSA top-k 到 hot-page 的运行时接线。
- VRAM 热页镜像和换入。
- 命中率、每 token host→VRAM 字节数和等待时间。
- 200K 相对本机 G0 的 2 倍门禁。

### M2：CPU 和显卡同时计算专家

目标：去掉“等专家搬进显卡再算”的主要停顿。

M2 P0 预算结论（2026-10-20）：

- 实测 host→staging 并行带宽：1/2/4/8 流，256 KiB 到 40 MiB 块，全部约 `2.6 GiB/s`；增加流数没有扩展。
- 拓扑复核：Vulkan0 走 CPU 根端口 `Gen3 x8`，Vulkan2 走 `Gen3 x4`；同一 probe 在 Vulkan0 约 `5.2 GiB/s`。当前 G0 的 Vulkan2 约为 Vulkan0 的一半，4K 单路探索性预填为 `247.3` 对 `165.1 token/s`。
- 精确 G0 prompt 三轮：Vulkan0 在 4K 为 `410.9/412.5/412.4` 预填、`25.9/26.6/26.4` 输出；30K 为 `268.4/268.5/269.2` 预填、`37.6/37.7/37.9` 输出，4K/30K 哈希均分别与 Vulkan2 G0 完全一致。Vulkan0 通过正确性和稳定性门，但 decode 明显低于 Vulkan2，不能直接整体替换 G0。
- 当前 4K paged expert 路径每 chunk 搬运约 45 GiB，和 `2.6 GiB/s` 相乘正好落在此前测得的 16-18 秒。
- 4K `2,500 token/s` 的预填预算只有约 1.6 秒；即使 45 GiB 不变，也需要约 `28 GiB/s`，不是当前路径能通过调 ring/lane 达到的。
- 两卡 48 GiB 标称显存无法同时容纳 47.168 GiB routed experts、非专家权重、KV、运行时和余量；全专家 VRAM 常驻不通过。
- 16 GiB 热专家 + 31.168 GiB 冷专家 + 256K 双会话预算约 53.65 GiB，可以落在 62 GiB 内存内（前提是非专家继续流式），但按 2.6 GiB/s 仍有约 12 秒/chunk 的残余搬运，达不到目标。
- 改用 Vulkan0 的 5.2 GiB/s 后，同样 31.168 GiB 冷专家约 6.0 秒/chunk，仍高于 1.6 秒目标。
- 实际 dry-run 复核后，Q4→Q2 不能按假设 2 倍计算：`Q2_K` 输出约 74.91 GiB（3.64 BPW），640 宽的 `ffn_down_exps` 回退 q4_0；最接近 2-bit 的 `Q2_0` 输出约 66.46 GiB（3.23 BPW），expert tensors 从 47.168 GiB 降到 31.641 GiB，只有约 `1.491x` 压缩，不是 2x。
- Q2_0 预算：16 GiB 热 expert 后仍有 `15.641 GiB` 冷 expert，按 5.2 GiB/s 约 `3.01 s/chunk`，仍超 1.6 秒；只压冷 expert（保留 16 GiB Q4 热）约 `20.9 GiB`，约 `4.02 s/chunk`，更差。
- 全部 Q2_0 expert 常驻两卡名义 VRAM：`31.641 + 9.285 KV + 5.2 runtime = 46.126 GiB`，名义 48 GiB 仅余约 `1.874 GiB`；非专家 40.855 GiB 必须完全走主机分页/流式，且现有引擎没有双卡 expert 驻留实现。这是研究候选，不是已验证路径。
- 主机完整常驻预算若把非专家也算进去：`16 GiB 热 + 15.641 GiB 冷 + 40.855 GiB 非专家 + 9.285 GiB KV + 5.2 GiB runtime + 8 GiB reserve = 95.0 GiB`（即使 hot 只算一次），与 62 GiB 相差约 33 GiB；所以非专家必须流式，不能把“分页”当成免费。
- 因而唯一可能成立的方向是“Q2_0 全 expert 驻留两卡 VRAM + 非专家严格分页/流式”，但它只剩约 1.874 GiB VRAM 余量，且尚未实现；Q2 质量门和完整实际工作集测量都必须先过。
- Q2_0 实际质量门不通过：4K 输出是 `RIC landmadaRIC...` 重复乱码，30K 输出是 `land land... RIC...` 重复乱码；4K/30K 预填虽为 `329.52/538.20 token/s`，但 decode 为 `18.09/19.71 token/s`，文本不可用。详见 `m2-p2-q2-quality.md`。
- 因此 M2 的首个实现方向不能只是扩大 staging；必须减少跨 host 边界的 expert 字节，或把一部分计算放到多卡/重压缩路径。CPU 行串行路径已在 P1 正式否决。
- 证据：`benchmark-data/artifacts/strata-parity-m1b-20261020/m2-p0-budget.md`、`prefill-bandwidth-probe.log`、`opwalk-classification.md`。
- 拓扑证据：`benchmark-data/artifacts/strata-parity-m1b-20261020/pcie-topology.md`、`prefill-bandwidth-probe-vulkan0.log`。

M2 P1 CPU 多行诊断（2026-10-20）：

- 发布模型 layer 4 的真实权重，i7-10700K 8 线程，诊断为 `1/2/4` rows × `1/2/4` misses。
- `rows=4, misses=4` p50 约 `1.328 ms`，折算当前行串行实现约 `83 µs/行×专家`。
- 按 4K chunk 的 47 层、512 experts、约 78.7 rows/expert 推算，当前实现约 `157 s/chunk`；理想化 4 倍权重复用仍约 `39 s/chunk`，远高于 4 秒停止线。
- **正式否决：批量多行 CPU GEMM 不再作为候选或原型方向。** 当前行串行实测约 `157 s/chunk`；即使假设批量 GEMM 有 `2-3x` 提升，仍是 `52-79 s/chunk`，相对 `1.6 s` 预算差约 `33-49x`（按 4x 理想化仍有 `39 s`，差约 `24x`）。这是量级不成立，不是实现难度问题。
- 不实现批量多行 CPU GEMM 原型；M2 的唯一入口改为 Q2 质量门 + 完整预算 go/no-go。
- 这不是 M2 实现提交，默认路径未改变。
- 证据：`benchmark-data/artifacts/strata-parity-m1b-20261020/m2-cpu-p1.md`、`m2-cpu-rows-probe.log`。

任务：

| 任务 | 主要代码 | 交付物 |
| --- | --- | --- |
| M2-P2.1 | Q2_0 量化工具/模型 | 30K 固定 prompt 质量门；hash 只记录分叉，不以 bit-identical 单独否决 |
| M2-P2.2 | 预算表 | 16 GiB 热 expert + 15.641 GiB Q2_0 残余 + 非专家/ KV / runtime 的 62 GiB 主机与 48 GiB VRAM 双预算 |
| M2-1 | `crates/infr-vulkan/src/pager.rs` | 仅在 P2 两项通过后，锁页主机专家池和显卡热点池 |
| M2-2 | `crates/infr-vulkan/src/cpu_miss.rs` | **否决，不执行**；行串行和批量 CPU GEMM 都不在当前路线 |
| M2-3 | Vulkan MoE 适配层 | 仅在 P2 通过后，多卡/重压缩执行需要时再评估 |
| M2-4 | `crates/infr-llama/src/seam/runner.rs` | 仅在 P2 通过后，结果合并方案另行设计 |
| M2-5 | 预测预取 | 仅在 P2 通过后 |
| M2-6 | 遥测 | GPU 命中、搬运、等待和回退 |

硬性要求：

- 不能启用已知会打飞地址的原地主机读取路径。
- Q2 质量门不能要求 hash 与 Q4 G0 完全相同；Q2 必然改变贪心路径。hash 只记录分叉位置，使用固定 prompt 输出对比、格式/事实约束和人工可解释性作为质量判据；决定性哈希不匹配本身不单独否决。
- 完整预算必须同时核算 62 GiB 主机内存和 48 GiB 名义显存，不能只算其中一边。
- 任何优化结果必须与 Vulkan2 G0 哈希/性能口径对拍；Vulkan0 只作 prefill-lane 候选。

中间验收：

- 150K 单路输出达到 30 token/s 以上。
- CPU 和 GPU 的重叠时间占总层时间 70% 以上。
- 输出哈希与未重叠路径一致。

M2 停止条件：

- Q2 30K 贪心哈希不可接受：压缩路线死，M2 暂停。
- 完整预算任一项装不下，或 5.2 GiB/s 下完整 4K 仍无法接近 1.6 s：M2 暂停，转 M1b 或 M5。
- 若 Q2 和预算都通过，才允许提出多卡驻留/压缩原型；CPU GEMM 仍不复活。
- 150K 未达到 30 token/s：先修专家执行，不进入 M3。
- CPU 路径在真实模型上仍无法运行诊断：证据不足，不能扩到生产。

决策点 D2：如果 150K 仍低于 30 token/s，暂停窗口流水，先继续解决专家执行；如果通过，进入 M3。

### M3：双卡窗口流水

目标：让两张卡同时处理相邻两批输出，追到 70 到 85 token/s。

任务：

| 任务 | 主要代码 | 交付物 |
| --- | --- | --- |
| M3-1 | `crates/infr-vulkan/src/pipeline.rs` | 持久化的两窗口执行句柄 |
| M3-2 | `crates/infr-llama/src/seam/runner.rs` | 验证分支可跨调用保留计划、绑定和临时缓冲 |
| M3-3 | `crates/infr-llama/src/seam/weights.rs` | 两套 KV 和循环状态的快照、恢复和提交 |
| M3-4 | `crates/infr-llama/src/mtp/qwen4.rs` | 构造下一窗口预测，并处理前缀命中或失败 |
| M3-5 | 调度层 | 窗口 N 的显卡 1 半段与窗口 N+1 的显卡 0 半段并行 |
| M3-6 | 遥测 | 记录预测命中率、回滚次数、回滚时间和两卡空闲时间 |

第一版限制：

- 只支持 Qwen3.8 MTP。
- 只支持双卡 Vulkan 层切分。
- 第一版固定 K=3。
- 第一版不接后缀预测。
- 任何不支持的形状自动回到串行路径。

验收：

- 短上下文输出达到 70 token/s 以上。
- 贪心输出哈希与串行 MTP 路径一致。
- 预测失败 1,000 次不污染后续状态。
- 回滚成本低于预测成功节省的时间。

M3 停止条件：

- 30K 或 150K 固定输入不能保持哈希一致：停止窗口流水。
- 单窗口边界通信超过 1 毫秒或每次搬完整状态：重新设计，不继续。
- 1,000 次失败回滚出现状态污染：停止并回滚实验分支。
- 窗口流水不能超过当前稳定路径：保留实验开关，不进入默认。

决策点 D3：窗口流水通过后再做动态预测；否则保留实验分支，回到 M2 或重新评估双卡结构。

### M4：动态预测和文本复制

目标：在可预测场景提高接受长度，在不可预测场景不要变慢。

任务：

| 任务 | 主要代码 | 交付物 |
| --- | --- | --- |
| M4-1 | 预测控制器 | 根据预计收益选择不预测、MTP 或复制预测 |
| M4-2 | 后缀预测 | 对代码、复制、重复上下文启用 |
| M4-3 | K 值调节 | 对 K=2、3、4、6、8 做场景化对比 |
| M4-4 | 最低匹配阈值 | 低匹配时不启动昂贵窗口 |
| M4-5 | 统计 | 记录每种策略的接受率、每周期 token 和实际耗时 |

验收：

- 普通聊天不低于 70 token/s。
- 代码类场景达到 100 token/s 以上。
- 不可预测文本启用预测后不允许明显慢于关闭状态。
- 预测策略切换不影响输出结果。

M4 停止条件：

- MTP 前置仍缺失 GPU embedding gather：停止 MTP 默认路径。
- 动态预测在普通聊天中低于预测关闭速度：只保留显式实验开关。
- 任何预测策略导致固定输入哈希变化：不进入默认。

### M5：输入阶段、内核和 256K 稳定

目标：追平 Strata 的输入速度并保证长上下文完整可用。

任务：

| 任务 | 主要代码 | 交付物 |
| --- | --- | --- |
| M5-1 | 预填调度 | 大块输入时重叠专家搬运和注意力 |
| M5-2 | QSA union | 预填每块只读紧凑热平面，过散时回退 |
| M5-3 | 提交路径 | 复用解码计划和命令缓冲，减少每步准备时间 |
| M5-4 | 内存预算 | 权重、KV 热页、专家热页、临时缓冲统一分配 |
| M5-5 | 256K | 两个会话和三种上下文深度的稳定性测试 |

验收：

- 32K、64K、150K、200K 输入达到 2,000 token/s 左右。
- 输出达到 45 到 50 token/s。
- 256K 单路和双路完成请求，没有看门狗停机。
- 关闭新路径后，1.0 原有性能不下降。

M5 停止条件：

- 256K 双会话容量仍超过物理内存：不宣称支持，降级为文档化限制。
- 150K 不能稳定完成请求：不进入发布。
- 默认路径不能一键关闭：不合并默认。

一键回退要求：

- 新架构总开关建议命名为 `INFR_STRATA_PARITY`，默认关闭，全部实验通过后再考虑切换。
- 总开关关闭时，模型加载、KV 放置、专家执行、MTP 和解码调度必须回到 1.0 行为。
- 子开关可以独立关闭热点 KV、CPU 专家、窗口流水和动态预测。

### G0 冻结值

冻结口径：

- 模型：`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`，33 分片
- 本机：Ubuntu 22.04、RADV、i7-10700K、Vulkan2
- 单路、贪心、KV VRAM、MTP 关闭、CPU miss 关闭
- UBATCH 4096、线程 10
- 1 轮丢弃预热、3 轮测量、32 token
- 页面缓存保持预热，GPU 性能档位 `auto`

冻结值：

| 场景 | 预填 | 输出 | TTFT | 哈希 |
| --- | ---: | ---: | ---: | --- |
| 4K | 约 214 | 约 40.0 | 约 18.3 s | `100f0e2e0808507cbb95ff0ce53e7481f3e83de86202eeebd536e7398b5e9eac` |
| 30K | 约 207 | 约 42.7 | 约 144.2 s | `bfccea41ad5424cee5ab0fc28542d7c82f591383f8902619e0c45b4b57b2f76f` |

这不是 Windows 1.0.0 发布口径，不能直接称为 1.0 回归。150K 和 256K 在预填瓶颈定位前不重建。

硬件拓扑候选（尚未整体替换 G0）：Vulkan0 的上游是 Gen3 x8，Vulkan2 的上游是 Gen3 x4；同一 staging probe 分别约 5.2 和 2.6 GiB/s。精确 G0 prompt 的 Vulkan0 三轮已通过哈希和波动门，但 4K decode 约 `26.3`、30K decode 约 `37.8`，低于 Vulkan2 的 `40.0/42.7`。因此 Vulkan0 记录为 prefill-lane 候选，Vulkan2 继续作为 decode 参考；任何按阶段切换设备的方案必须单独设计、验证并保留回退。

## 7. 性能验收门

| 门 | 场景 | 目标 | 必须同时满足 |
| --- | --- | ---: | --- |
| G0a | 4K 正确性 | 固定输入三轮输出一致 | 已通过：发布模型哈希一致 |
| G0b | 30K 正确性 | 固定输入三轮输出一致 | 已通过：发布模型哈希一致 |
| G0c | 150K 可执行 | 单轮完成并记录真实速度 | 未通过：尚未在发布模型上完成 |
| G0d | 两路并发 | 两个请求完成 | 启动和完成通过，跨 lane 相同输入哈希不一致 |
| G0e | MTP K=3 | 兼容 sidecar 下完成 | 证据不足：发布模型无 sidecar；IQ3_S 缺 embed gather |
| G1 | KV 热缓存 | 200K 达到全主机 2 倍 | 双会话不失败，哈希一致 |
| G2 | 专家并行 | 150K 达到 30 | CPU/GPU 重叠 70% |
| G3 | 窗口流水 | 短上下文 70 | 回滚正确，输出哈希一致 |
| G4 | 动态预测 | 代码 100 | 普通聊天不低于 70 |
| G5 | 最终稳定 | 32K 到 200K 45 到 50 | 输入 2,000，256K 可用 |

任何阶段如果不能通过正确性门，不能进入下一阶段。

## 8. 基准矩阵

每个阶段至少执行以下矩阵：

| 维度 | 取值 |
| --- | --- |
| 上下文 | 4K，32K，64K，150K，200K，256K |
| 并发 | 1，2，可选 3 |
| 生成长度 | 128，256 |
| 提示类型 | 普通聊天，代码，重复文本，唯一文本 |
| KV | 全主机，全显存，自动，热点缓存 |
| 专家 | 常驻显卡，当前分页，CPU 与显卡同时 |
| 预测 | 关，MTP K=3，动态 |

M0 收尾后的最小矩阵先缩为：

1. 4K 单路，固定预热协议，三轮。
2. 30K 单路，固定预热协议，三轮。
3. 4K 两路并发，验证跨 lane 确定性策略。
4. 以上可信后，才允许跑 150K 单路。
5. 150K 预填瓶颈定位后，才允许跑 256K 和双会话。

证据禁用：

- `baseline-150k-r3-v2/` 是主动中止的不完整运行，不得作为 150K 性能或正确性证据。
- IQ3_XXS 的 30K 分叉只用于格式诊断，不得作为发布模型 G0。
- GPU 频率未能固定：当前为 `auto`，需要记录每轮频率波动；不能把微小速度差归因于代码。

每次必须记录：

- 输入速度。
- 首个 token 时间。
- 稳态输出速度。
- 总墙钟时间。
- 显存和内存峰值。
- 交换区使用量。
- KV 命中率和主机读取字节。
- 专家命中率和主机读取字节。
- 两卡忙碌时间。
- 预测接受率、命中率和回滚时间。
- 输出哈希。

## 9. 代码改动地图

| 区域 | 主要文件 | 主要职责 |
| --- | --- | --- |
| KV 分配和页表 | `crates/infr-vulkan/src/lib.rs` | 显存页、主机页、动态增长 |
| KV 会话状态 | `crates/infr-llama/src/seam/session_state.rs` | 增长、恢复、清理 |
| KV 权重绑定 | `crates/infr-llama/src/seam/weights.rs` | 单会话和双会话缓冲 |
| QSA 访问 | `crates/infr-vulkan/src/adapter.rs` | top-k、gather、预取信号 |
| 专家页管理 | `crates/infr-vulkan/src/pager.rs` | 命中、未命中、搬运、淘汰 |
| CPU 专家 | `crates/infr-vulkan/src/cpu_miss.rs` | 多行计算和结果合并 |
| 双卡流水 | `crates/infr-vulkan/src/pipeline.rs` | 层切分、跨卡交接、重叠执行 |
| 解码调度 | `crates/infr-llama/src/seam/runner.rs` | 计划、绑定、提交、回滚 |
| MTP | `crates/infr-llama/src/mtp/qwen4.rs` | 草稿、验证、循环状态 |
| 配置 | `crates/infr-core/src/config/mod.rs` | 新路径开关和默认值 |
| 基准记录 | `scripts/` 和 `benchmark-data/` | 可重复测试和原始证据 |

## 10. 风险和对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| 显存抖动 | 50% 缓存比例反而变慢 | 扫命中率曲线，使用共享预算和滞后淘汰 |
| CPU 太慢 | CPU 专家拖住 GPU | 只把未命中专家交 CPU，限制并行任务数，按实测调整 |
| 状态污染 | 预测失败后输出变化 | 每次窗口后校验哈希，失败立即回滚 |
| 平台差异 | Linux 快、Windows 慢或相反 | 每阶段至少在目标平台执行完整矩阵 |
| 复杂度失控 | 新旧路径难以维护 | 全部新路径用独立开关，先保证稳定路径不变 |
| 统计口径漂移 | 数字好看但不可比 | 固定模型、提示、长度、并发和统计区间 |
| 主机内存不足 | 锁页专家层分配失败 | 允许专家池按预算缩小，并保留分页路径 |

## 11. 后续工作协议

每次实施遵循同一流程：

1. 从本文选择一个任务编号。
2. 记录当前提交和二进制哈希。
3. 先跑基线，再跑候选。
4. 保存原始日志和结果 JSON。
5. 对比速度、命中率、内存和输出哈希。
6. 通过后更新本文状态和证据路径。
7. 失败时写清否决原因，避免未来重复试验。

完成任务的定义：

- 代码已提交。
- 单元测试和相关集成测试通过。
- 至少一次端到端 A/B 完成。
- 输出正确性有证据。
- 性能目标达到或明确说明未达到。
- 文档和计划表已更新。

## 12. 当前入口任务

项目冻结：2026-10-21。除非满足 `moe4all-project-freeze-20261021.md` 的解冻条件，否则不继续修改引擎。

当前入口任务：

1. 冻结 Iq3S 正确性 G0：`hit_first=false`，30K 128-token 两轮 hash `49f384ee...`，只用于 correctness，不用于性能。
2. 冻结“无可交付 decode 性能 G0”的结论：small-m static 128-token 为 `0.34/0.60 tok/s`；static batched 128-token 两轮 hash 一致 `f7c0f88b...`，但只有 `0.72/1.25 tok/s`；hit-first 快但不正确。
3. Vulkan0 prefill 迁移已完成：30K、32 token、丢弃预热、3 轮为 `401.34/403.79/402.99 tok/s`，hash `400972cd...` 三轮一致。
4. 驻留和压缩预算已完成，见 `moe4all-iq3-budget-vulkan0-20261021.md`。24 GiB hot expert 是最平衡档，但按当前效率预计 prefill 只能到约 1,000 tok/s；32 GiB hot 也只有约 2,021 tok/s，且显存余量不足。
5. `P-hit-first-determinism` 已在独立分支 `perf/hit-first-determinism` 启动。首个实现提交 `2d771ef test(moe): probe unmasked Iq3S resident all-hit` 未通过 hidden 一致性，但已证明问题不只在初次 all-hit 标志。
6. M2 实现继续受 decode/正确性门约束；M2 驻留预算、路由统计和容量诊断不受约束，可以继续。
7. `dca2cc4` 的 `decode_gpu_resident_iq3s_hit_first=true` 只作诊断 opt-in，永不默认开启。

P-hit-first 硬时间盒：

- 最长 2 天，不进入 M2 实现。
- 成功门：30K 128-token 两轮 hidden 一致；decode `>20 tok/s`。
- 成功后冻结 G0-decode，再进入 M2 最小实现：24 GiB hot expert + 冷专家分页，不做 CPU miss。
- 失败门：2 天内不能同时满足，则暂停 P-hit-first，转 M2 最小实现；M2 目标降级为“prefill 从约 401 提到约 1000”，decode 暂用 `0.34/0.60` 兜底，并明确端到端不可交付。

禁止：

- 不得把 hit-first 的性能参考数据写成 IQ3_XXS G0。
- 不得在 128-token 正确性未过的路径上进入热点 KV、专家并行、窗口流水或预测架构。
- 不得用前 8 个 token 的稳定结果替代完整 128-token 门。

旧 Q4 的 G0、Vulkan0 412 token/s、Q2 质量门和 2500 推算全部只作历史诊断，不进入当前门禁。`token_embd.weight=Iq3S` 的 GPU gather 仍是独立前置实现任务，不与本次 hit-first 正确性修复混做。

## 13. IQ3_XXS 30K 静默分叉根因记录

结论：

- 根因不在 embedding、KV、MoE 路由、gate 权重、expert 权重内容或 small-m IQ3S 内核本身。
- 根因位于 GPU-resident decode 的 hit-first 异步路径。同一输入、同一 top-10 expert、同一 gate 权重、全命中且 slot 内容采样 hash 一致时，hit-first 仍会让第 2 层 `ffn_raw` 在第一轮与后续轮之间分叉。
- 只关闭 `decode_gpu_resident_hit_first`，保留 resident controller、fast-LUT 和 small-m，就可以让两轮 30K 前缀的 1152 个 hidden hash 与文本 hash 全部一致。

关键 A/B：

| 路径 | 两轮 hidden | 文本 hash | 热态 decode |
| --- | --- | --- | --- |
| resident hit-first 默认 | 第 2 层 `ffn_raw` 从第 1 个 decode 位置起不同 | 不稳定 | 基准约 9-16 token/s，但输出错误 |
| resident，仅 `hit_first=false` | 1152/1152 一致 | 一致 `13a3fc68...` | 约 `7.4 tok/s`，第一轮约 `0.2 tok/s` |
| resident，conservative 关闭多项 | 1152/1152 一致 | 一致 | 约 `8.8 tok/s` |
| static small-m | 1152/1152 一致 | 一致 | 热态约 `7.3 tok/s`，第一轮约 `0.2 tok/s` |
| static batched MMQ | 8-token 前缀一致；128-token 自身两轮稳定 | 自身一致 `f7c0f88b...`，与 small-m G0 不同 | 128-token `0.72/1.25 tok/s` |

实现提交：

- `2296f6b fix(moe): disable hit-first for Iq3S resident decode`
- 二进制 SHA-256：`d013924694273ae5bd21b0ca25cea275666969e4cebcefcf2c6a64d76e65a264`
- 默认 resident graph 日志应显示 `hit_first=false`，但其他量化格式保持原默认。

证据目录：

- `/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All/benchmark-data/artifacts/strata-parity-iq3-correctness-20261020/30k-layer6-moe-slotsample-trace-r2/`
- `.../30k-smallm-static-slots-ab-r2/`
- `.../30k-batched-moe-static-prefix-ab-r2/`
- `.../30k-resident-fastlut-no-hitfirst-prefix-ab-r2/`
- `.../30k-resident-no-hitfirst-only-prefix-ab-r2/`
- `.../30k-iq3s-no-hitfirst-default-r2/`

未完成门：

- 默认 `hit_first=false` 的完整 30K 128-token 两轮已跑完：两轮输出 hash 均为 `49f384eef7391a151c5a3d9e9521cbcb8d5e85cf790db152bdf8d273b8e33eb2`，1152 个 hidden hash 全一致。但 decode 只有 `0.34/0.60 tok/s`，不能作为性能 G0。
- static batched MMQ 完整 128-token 两轮已跑完：hash 均为 `f7c0f88bcab8b72ba9cc45a83fc98bc5934d28af7fd5b70bd6ca36aa4b648fc2`，decode 为 `0.72/1.25 tok/s`。它自身稳定，但 hash 与 small-m G0 `49f384ee...` 不同，不能算同一正确性路径，也不能作为 `G0-decode-temp`。
- 因此结论是“IQ3_XXS 已有正确性 G0，但没有可交付的 decode 性能 G0”；M0 性能门不能标通过。
- `baseline-150k-r3-v2/` 仍禁用，不作为证据。

Hash 口径：

- Vulkan0 `400972cd...` 来自 `decode_gpu_resident=false` 正确路径，但只是 32-token 路径 hash，不能与 128-token G0 直接比较。
- static batched `f7c0f88b...` 与 small-m `49f384ee...` 不同；没有外部质量参考时，不能把 batched 数值差异视为同一正确路径。

新增证据：

- `/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All/benchmark-data/artifacts/strata-parity-iq3-correctness-20261020/30k-smallm-static-g0-r2-128/`
- `/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All/benchmark-data/artifacts/strata-parity-iq3-correctness-20261020/30k-batched-moe-static-prefix-ab-r2/`

当前最终二进制：

- HEAD：`dca2cc4 perf(moe): expose Iq3S hit-first opt-in`
- SHA-256：`b4ed21345dc1089d03bbc69dcc0e5eb6a8fb5d7776131b4b66d7ffa8383409fe`
- 默认 `hit_first=false`；显式 opt-in `kernels.vulkan.decode_gpu_resident_iq3s_hit_first=true` 只用于诊断，不得进入默认路径。

## 14. 2026-10-21 补充结果

### static batched 128-token

- 30K、128 token、两轮
- 输出 hash：两轮均为 `f7c0f88bcab8b72ba9cc45a83fc98bc5934d28af7fd5b70bd6ca36aa4b648fc2`
- prefill：`232.62/237.39 tok/s`
- decode：`0.72/1.25 tok/s`
- 判定：稳定，但不可交付；不能作为 `G0-decode-temp`
- 该 hash 与 static small-m 的 `49f384ee...` 不同，只能作为该数值路径自己的稳定 hash

### Vulkan0 prefill

- 30K、32 token、丢弃 1 轮预热、3 轮
- prefill：`401.34/403.79/402.99 tok/s`
- 波动：约 `0.61%`
- 输出 hash：三轮 `400972cd0814cfa69c6e207fef95e28cca0f14ecd0f968abcec6d2162b2f3bfe`
- decode：`31.98/31.89/32.05 tok/s`，仅作短跑参考
- Vulkan0 可继续作为 prefill-lane 候选

### 驻留预算

完整数据和计算见：

`/home/qiankun/Documents/Codex/2026-10-19/new-chat/outputs/moe4all-iq3-budget-vulkan0-20261021.md`

关键结论：

- 62 GiB 主机内存下，256K 双会话 + 全部 IQ3 expert 主机常驻需要约 `62.451 GiB`，已越过 62 GiB。
- 24 GiB hot expert 的主机预算约 `38.451 GiB`，剩余约 `23.549 GiB` 给非专家页缓存和余量。
- 5.2 GiB/s 纯搬运上限：16 GiB hot 约 888 token/s；24 GiB hot 约 1,334 token/s；32 GiB hot 约 2,675 token/s。
- 用当前 Vulkan0 约 75.6% 效率修正：16/24/32 GiB hot 约为 671/1,007/2,021 token/s。
- 因此 IQ3_XXS 不能只靠 hot/cold 流式达到 2,500 token/s，必须减少跨边界字节或增加 CPU/第二卡计算。

## 15. 参考记录

- `/home/qiankun/Documents/Codex/2026-10-16/new-chat-2/work/STRATA_LESSONS.md`
- `/home/qiankun/Documents/Codex/2026-10-16/new-chat-2/work/ARCH_MTP_WINDOW.md`
- `/home/qiankun/Documents/Codex/2026-10-16/new-chat-2/work/STATUS_MOE4ALL.md`
- `/home/qiankun/Documents/Codex/2026-10-18/new-chat/outputs/moe4all-256k-recipe.md`
- `/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/outputs/strata-speed-2026-10-19.md`
- `/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/outputs/strata-context-speed-32k-64k-200k.md`
- `/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All/README.md`
- `/home/qiankun/Documents/Codex/2026-10-19/oe4all-github/work/MoE4All/benchmark-data/artifacts/strata-parity-m0-20261019/`
