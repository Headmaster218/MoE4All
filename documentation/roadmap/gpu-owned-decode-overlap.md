---
kind: roadmap
status: partially-validated
scope: ordinary-decode-gpu-ownership-and-overlap
last_verified: 2026-10-28
verified_commit: 5f4d21dc0
---
# 普通 Decode：GPU 所有权与重叠优化

## 目标与规则

进入普通 Decode 时，将专家驻留表与 LRU 所有权交给 GPU；连续 token 保持所有权，
只在 token 边界发生工况或布局变化时交还 CPU。执行中保留三类必要交互：
与 L0 并行准备的 PLE、GPU 无法自行搬运时的指定专家上传、token 结果与边界控制。
CPU 优先解除 GPU 阻塞，其余工作与 GPU 计算重叠。

- 架构与数据容量覆盖 1～8 路；本机原生实测最多 4 路，8 路以单元／模块验证，不强改显存预算。
- 先 35B，再带 PLE 的 FlashNext；本轮不接 MTP，不改缓存容量或省略专家。
- 保护 KV 扩容、换槽、Prefill、视觉／Embedding、取消和失败时的统一显存租约。
- 目标为整批前向相比原版：35B 约省 1.15ms，FlashNext 最高约省 2.5ms。
  模块收益不能直接相加或乘层数，trace 成本不能当收益。
- 按小单元→真实模块→必要无 trace 原生 A/B 验证；异常再扩大，不机械重跑完整矩阵。
  模块有收益且端到端无实质回退才保留；更大架构改造可明确 pending，继续其余项目。

原版基准：`b95bebee5dabf02976ae0e281f55852addceb3b6`。
本地分支：`feat/gpu-owned-decode`。未推送、发版或替换生产 exe。

## 实施顺序与结论

| 顺序 | 工作 | 验收条件 | 结果 |
| --- | --- | --- | --- |
| A | 贯穿 1～8 路：批量发布／ACK、3～8 路 router／owner、recovery | 数值、FIFO、取消及 fallback 正确；多路不退 | 已保留；原生有收益，但 Flash 四路仅在原版波动内 |
| B | 完整 hit FFN 与 GPU 自主搬运重叠 | 全程重叠而非内部 barrier 提前 join；模块更快 | 两种局部方案回退，拒绝；跨队列／持久 FFN pending |
| C | GU 先搬、Down 后搬，保留完整 hit；CPU 上传不拆两次 | 相同搬运参数下模块获益，必要原生无回退 | 仅严格 Q5 单路形状保留；其他形状／路数回退原调度 |
| D | CPU miss 首尾：尽早上传、旧 SKIP／准备与上传重叠 | 目标、源租约、ACK／event 代次正确 | 成功路径保留并通过组合测试；真实 DMA wait-error 仍未闭合 |
| E | 首尾 gap：前移准备、延后非关键工作、批量交付 | 不越过采样、EOS、取消、工况切换边界 | 已保留并默认开启；非所有工况达到预期 |

## A：多路兼容

普通 Decode 先发送所有非 EOS 路的 token，再收齐所有成功发送路的 ACK。
某路取消、断连或回调 panic 不得漏收其他已发送路的 ACK。MTP、Lookup、Prefill 保留原回调路径。

1／2 路原融合 router 不变；3～8 路新增独立 WG256／Fixed32 路径，受设备、形状和路由语义证明保护。
不支持的路由、设备或容量走原有 fallback，不通过识别模型名放宽门限。

- 17 项交付 CPU 回归；3～8 路路由、重复专家、非有限值、拒绝、FIFO／压紧、prefix1～128 通过。
- bounded checked-prefix 与独立 pool-SG：3／4 路 mixed 模块省约 17～51us，全 hit 基本持平。
- non-TopK recovery 的四路 used8／10 模块约省 5.18／3.38us；没有顺带扩大普通 admission。
- trace mask 与容量按 MAX_DECODE_ROWS，8×16 超 WG 容量仍完整回退。
- 当前 LRU 是时间戳 FIFO 日志，非 prev／next 链表；热路径已并行追加，不在每次 hot 操作扫描整条历史日志。

热点归因 v104：相同融合 kernel，四路 used10 的 true-route 均值 48.200us，
保留 oracle ID 的 owner-only 为 31.705us；差值约 16.495us，配对离散较大。
不能将全部时间归为 LRU，也不能乘层数推算整机收益。

随后 v105 尝试在同一 private float[16] 中提前计算 selection score：
3～8 路 external 精确性及六组 GPU 测试通过，1／2 路六个相关 SPV byte-identical；
但八组 3／4 路配对模块全部回退约 6～33us。因此不改生产 shader，只保留测试接线与外部证据。
不凭静态指令数继续扩大原生测试，也不夹带新链表策略。

## B：完整 hit 与搬运

现有 bounded CU copy 可与 hit Gate／Up 重叠，但后续普通 COMPUTE execution barrier 会 join copy。
仅缩小 buffer barrier 范围不能证明它只等待某几个 dispatch。

已做并拒绝：

- 单 kernel 完整 hit FFN：每专家工作组过少、Down 分片重复 GU，实际 Q5 模块约 14→301us，
  Flash 约 29→553us。消掉 barrier 不能抵消计算并行度损失。
- KHR indirect copy：真实量化 1～8 路、合法 dummy 记录、地址提示均正确；
  连续 frame 仍比现有 partial overlap 慢约 5%～10%，不进入生产扩展初始化。

进一步完整重叠需要独立 compute queue 或持久／协作 FFN，涉及执行模型与同步保证。
本轮不删除关键 barrier 强行“完成”；该项架构性 pending，测试与失败证据保留。

## C：GPU role 流水

同一次 admission／LRU／journal 事务：phase1 搬 GU，phase2 搬 Down 与计算 miss GU，
最后发布整个专家 READY。无 mapped GU job 时 Down 提前搬；CPU 上传始终整批。

只在同 recorder／命令／resource epoch、固定 hit suffix、scratch 不别名及来源不变的证明下，
省去第二阶段入口同步；证明不全走原入口，不提供通用跳 barrier 接口。

公平单专家三-role miss 模块：

| 形状 | 原调度 | 候选 | 结论 |
| --- | ---: | ---: | --- |
| 35B，单路 | 约 110us | 104～109us | 小收益 |
| 35B，双路 | 约 142us | 140～143us | 基本持平 |
| Flash，单路 | 约 149us | 158～164us | 回退 |
| Flash，双路 | 约 224us | 249～252us | 回退 |

比较统一 copy 宽度／组数，包含 fresh scratch、毒化 miss 权重、candidate-only replay、
跳过 GU 的负对照、fused／原生 GU 以及保守／no-entry 四组，不由旧数据制造假收益。

因此生产仅保留：G/U 均 Q5K、NE=2048、NFF=512、used=8、单路，
并保留设备、来源、几何、后缀、epoch、别名全部门限；不是模型名特判。
其他量化／形状／路数保持原调度。`kernels.vulkan.decode_gpu_resident_role_pipeline=false` 可关闭。

v102 同 EXE 开／关：30K 7.362125／7.385410ms（约省23us）；
150K 12.910154／12.907149ms（约慢3us，持平）。
保留依据是模块收益及必要端到端无实质回退，不称为稳定的大幅整机提升。

实际 adapter／prepared、2／4／8 路与 Flash fallback、mixed status5 的 GPU-readable 和
CPU-only miss、延迟 host-ready 上传、ACT／输出及同实例下一请求／交接均通过。
v103 混合测试仅补测试代码，生产字节与冻结 v102 CLI 相同。

## D：CPU miss 关键路径与生命周期

分别统计 notice→上传开始、上传本身、上传完成→miss 开算，不将带宽当作全部耗时。

已保留：
1. 只收集 GPU 实际 miss ID，保留完整 count／mask／ID 校验。
2. 同一不可变上传 plan 内缓存目标映射与来源分类，不跨资源租约复用。
3. producer 已成功等待并退休 DMA 后，consumer 不重复等待；仍保留 GPU acquire timeline。
4. 有界复用上传 worker：observer 可并行处理旧 SKIP；任何非 SKIP 前先等上传退休。
5. Drop 等已接受任务结束，不持 pager 锁等线程；共享 ACK 覆盖双方。
   原 upload wait→旧 ACK drain→event reset→host-ready signal 顺序不删。

成功 DMA 的 Shared→transfer slot→Buffer→Shared 引用环已修：
只在成功 timeline wait、实际等待该 value 的 compute fence 或成功 counter 证明后退休，
锁外释放 Arc；不增加每 fence 的 Vulkan query／queue-idle。
10 项 CPU 与3项真实 GPU 测试验证字节、取消 drain 及 Weak 释放。

**D3 未解决**：真实 DMA timeline wait 失败时，allocation Arc 不等于源 block 不可复用。
现有 bounded RAM shadow pin 的寿命尚未证明覆盖未退休 DMA；新 worker 不接 bounded 来源，
不能把它全部归因于 worker，也不能用 mock wait-error 通过宣称真实安全恢复。
需同时审计独立来源 pin、目标 slot、owner handoff 与未退休提交的错误状态，
不能只改 drop 顺序或增加重试。详见本轮 `d3-retirement-audit.md`。

第二次审计确认目标端也缺保护：`discard_cpu_miss` 可 evict，handoff 将 COPYING slot
放回 free，`DeviceTransferTarget` 只持 Buffer Arc而无 slot reservation。
因此 source-only 补丁不足以允许错误后继续复用；需覆盖 owner／Prefill／统一池的目标租约，
以及无法证明退休时的失败隔离与退场政策。该生命周期改造明确 pending，不悄悄新增 abort。

## E：首尾与状态切换

- 前移不变资源验证、布局及安全预录；下一步须 ACK 后才执行。
- 按同 owner／来源 bank 的精确证书，预录至首个需 CPU 服务层的完整现有 group；
  全 GPU-readable 保留原范围，证明未知则回退。不改执行窗口、rolling、inflight、credits。
- 当前 token miss 不自动取消已完成 NEXT。成功 scope-end cancel 试验未获益，拒绝；
  保留失败／取消时的协作撤销。旧录包7～24ms不等于 scope join，后者仅约0.5～2.4us。
- disabled mask 改为完整切片精确比较；16384-slot equal 微测约8.67→0.60us。
  保留长度／几何／代次，未采样或只比较 hash。
- 五项普通调度默认 true，仍可显式 false：command_ahead、early_submit、resident、
  copy_overlap、host_ready；能力和工况回退继续有效。
- MTP 保持旧路；Lookup 不提前录制；Verify 保留串行复制，本轮不扩大 speculative 支持。

547 项 Vulkan CPU、302 项 Core、17 项交付通过；
短预录包接管、early-submit、未 signal host-wait 的错误／取消、同实例重试、
KV／辅助资源交接与非普通图隔离 GPU 回归通过。
这些不是全部设备／工况的穷举证明。

## 无 trace 端到端结果

统一参数：VRAM23g、RAM48g、ubatch3072、Q8 KV，XTX/Vulkan1；真实30K／150K输入，
256-token warmup 后测256 tokens，温度0、相同 seed／输入／缓存；不手动覆盖五项默认调度。
表内是整批一轮前向，不除以并发数；原版三个重复全部保留。

| 工况 | 原版 ms/轮 | 候选 ms/轮 | 净省 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 35B 30K 单路，v102 | 8.212886 | 7.362125 | 0.851ms | 有收益，未达1.15ms |
| 35B 150K 单路，v102 | 13.704018 | 12.910154 | 0.794ms | 有收益，未达1.15ms |
| 35B 30K 四路，v102 | 20.956608 | 19.535288 | 1.421ms | 有收益 |
| Flash 30K 单路，v100 | 20.764510 | 19.612793 | 1.152ms | 有收益，未达2.5ms |
| Flash 30K 双路，v100 | 31.392296 | 30.323131 | 1.069ms | 有收益，未达2.5ms |
| Flash 150K 双路，v100 | 37.419088 | 36.247959 | 1.171ms | 有收益，未达2.5ms |
| Flash 30K 四路，v102 | 48.138715～49.058874 | 48.562564 | -0.424～+0.496ms | 原版波动内，无稳定收益 |

全部配对输出和 SSE 分块一致。引擎 step 在交付／ACK 前结束；
另核对完全相同分块的 SSE 间隔，text delta 不冒充 token ID。
旧版本双路数据另保留，不当作最新 exe 重测结果；prefill 进度偶尔0不等于挂死。

当前目标**未全部达到**。Flash 四路曾从 v85 52.20ms、v94 50.26ms 改到 v102 48.56ms，
已消除先前明显回退，但不能把相对旧候选的改善全当相对原版的收益。

## 本地提交与证据

| 提交 | 内容 |
| --- | --- |
| `0e0d51708` | GPU 连续所有权底座 |
| `5ad870bc5` | 批量交付／ACK |
| `9a76a948a` | 精确 mask 切片比较 |
| `86b8d179d` | 成功 DMA 资源退休 |
| `f4f6386ba` | 设备下限与 MTP 隔离 |
| `21f023bdc` | ManyRow partial／pool-SG |
| `153ea63eb` | 仅 recovery 的多路融合 |
| `e429fb6ad` | 构建元数据检查误报 |
| `5f79aba99` | test-only 保留被否决 B/C 实验 |
| `970c79ddd` | 上传 helper、source-certificate 预录及组合回归 |
| `a0095839f` | 严格 Q5 单路 role 流水 |
| `39b118ece` | 已验普通调度默认开启 |
| `cb60c3d69` | 路由／hot-owner 归因测试 |
| `5f4d21dc0` | current／external 候选精确与配对测试，评分 hoist 被否决 |

任务 ID：`20261025-130100-gpu-owned-decode-c741`。
生成物统一在 `D:\AIinfr\infr\local-work`：
`evidence/infr/<task-id>` 保存 manifest、输入／exe身份、原始 log 和失败方案；
`results/infr/<task-id>` 保存 `roadmap-native-summary.json/md`、
`v103-v104-qualification.md`、`v105-score-hoist.md` 与 D3 审计。
详细本机实验账本：`upstream-research/decode-overlap-roadmap.md`。

共用通用 x86-64／static-CRT 构建缓存，冻结 exe 再测；原生最多四路，模块保留八路。
超过1核或512MiB的工作独占 `D:\cpuandram.lock`，自身进程退出后释放，不碰他人锁。
磁盘规范唯一来源：`D:\AIinfr\disk_usage_rule.md`。
