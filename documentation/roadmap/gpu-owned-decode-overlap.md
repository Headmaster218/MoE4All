---
kind: roadmap
status: partially-validated
scope: ordinary-decode-gpu-ownership-and-overlap
last_verified: 2026-10-28
verified_commit: 434037b39
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
| B | 完整 hit FFN 与 GPU 自主搬运重叠 | 包含最终归约的模块更快；必要原生无回退 | 多方案及参数复测完成；event 相对既有 split 回退，删除其运行时；保留默认关闭的 IQ3 四路计算拓展 |
| C | GU 先搬、Down 后搬，保留完整 hit；CPU 上传不拆两次 | 相同搬运参数下模块获益，必要原生无回退 | 严格 Q5 单路原方案保留；新 mixed dispatch／预分类无额外收益 |
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

上述否决只针对旧实现，不等于局部融合无法获益。新一轮按 NFF 分块，每块只计算一次
Gate／Up→SiLU，再生成 Down 部分和，避免每个 Down 输出 tile 重算 GU。
该部分可与原整专家 CU copy 并行；join 后单独归约。主要 Down 运算已前移，
但最终 H 输出仍依赖 join，不能称为完全无同步，也没有取消必要 barrier。

### 新 split-NFF 试验（v117）

- 先覆盖 1／2／4／8 路模块、shared／无 shared、all-hit／mapped miss、非法几何、
  stale W0、别名及 prepared replay，再选实际 engine 形状。
- 早期串行归约在小幅输入下通过，但 512 倍输入放大后超出原定误差门限；
  v115 的 6K 原生结果也有 3 路文本不同，因此其速度不作为最终合格证据。
- 修正为原生 Down 的 SG32 归约树；128 输出合并读取、带 padding 的 LDS 转置，
  不用 atomic 求和、不放宽门限。最终模块和实际 adapter 强数值测试 max_abs=0，
  包括三种重合／不重合路由、ID>9、真实 mapped miss、accepted prepared replay、
  新 H partial 写入、M partial 保持毒值、guard 与资源代次不变。
- Q5 shared 四路最终模块仍回退（约91.33→108.37us／154.86→176.50us），
  不加入 Q5 新运行路径。旧串行版 Q5 单路 miss 模块的小收益不能冒充最终合格结果。

保留 `kernels.vulkan.decode_gpu_resident_splitk_hit`，**默认 false**，仅普通独立四路、
G/U=IQ2S、Down=IQ4NL、NE2560／NFF640／top10、F32／SiLU，且全部能力与语义证明通过时可用。
不是模型名特判；Flash 当前权重的36个 IQ2 层可入选，12个 IQ3 层保持原路径。
1／2／3／5～8 路、其他量化／形状、MTP／Verify／Prefill 以及设备不支持均回退。
shared expert 保持 engine 原有 standalone dense tail，不把合成 shared 模块当真实基线。

chunk32／local128／Fixed32，需要 BASIC／ARITHMETIC／SHUFFLE 和完整 subgroup；
归约使用16,512字节 LDS。新增8,192,000字节（7.8125MiB）统一 phase scratch，
所有层复用，在 owner／LUT capture **之前**预留，录制中只查已存在 pool key，不再扩池。
同设备、范围、对齐、LUT window、非别名、命令及 epoch 校验继续有效；
CPU 上传、LRU 所有权、miss recovery 与合并规则不改。准备包失效仍走现有回退。

该窄路径已有一次无 trace 30K×4 配对收益，见下表；150K、其他设备尚未原生确认，
不据此默认开启或宣称所有路数受益。完整最终 H 与 copy 独立的后续试验见下节，
不能将相对原生 FFN 的收益重复计为相对 split 路径的额外收益。

### 完整 H 再试（v118～v128）

本轮只搬完整专家，**不拆 GU／Down 传输**。模块覆盖 Q5K 与 Flash IQ2S／IQ3S，
1／2／4 路、全命中和一个整专家 miss；放大输入、原误差门限、毒值与 guard 均保留。
Q5 合成 shared 模块还补测了真实引擎的独立 shared-tail 口径，不能互相代替。
原生测试只对模块通过的候选做必要对照，不扩展完整上下文矩阵。

| 方案 | 迭代与结果 | 决定 |
| --- | --- | --- |
| 完整 FFN 单 WG／输出分块 | WG128／256、tile1～32，再试显式展开和 WG512／1024；重复 GU 与并行度损失仍大，部分大 WG Q5 强数值检查失败 | 不进入运行时 |
| 紧凑 GU＋SiLU、原生 Down、精确 event | NR2／4／8，LDS／buffer 码表，再合并 event 重置；Q5 真实非 shared 单／双路全 hit 仍约多0.4／0.8us，不能只拿合成 shared 的小收益宣称通过 | 不进入运行时 |
| 同 family 辅助计算队列 | 保留原生 H；再将16阶段压入两次 submit、减少自等待／信号、先提交辅助队列；组合成本仍高约130～220us／阶段 | 仅留显式 cfg(test) 实验，不改生产队列 |
| split partial／reduce＋event | 可以令完整 H 归约不再等待整专家 copy；数值与真实 adapter 正确，但相对既有 split 没有增量收益 | 撤掉实验运行时，只留测试与原始证据 |

辅助计算队列仍用 CU 做复制，**不是硬件 DMA 队列**。queue count 不等于硬件并发能力；
模块时间包含提交范围内的信号、等待和最终 M，不能只报 H kernel 的时间。

关键的同条件增量对照（v127，四路、batch event reset，单位 us）：

| 量化 | 全 hit：既有 split → event | 一个整专家 miss：既有 split → event |
| --- | ---: | ---: |
| IQ2S／IQ4NL | 242.81 → 244.15 | 327.98 → 336.38 |
| IQ3S／IQ4NL | 246.13 → 248.23 | 294.80 → 336.07 |

此前相对原生 FFN 的约70us收益，主要是 split 计算本身，不是 event 的额外收益。
既有 copy-first 调度已经能用 partial 计算覆盖不少搬运；提前末尾归约的收益有限，
producer-first 排序又不利于尽早发起复制。这里不将模块数据冒充精确硬件 overlap 时间线。
event 的执行范围依据 [Vulkan 同步规范](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-events)，
合法缩小依赖范围也不保证端到端更快。

v126 曾接入真实运行时，覆盖 IQ2／IQ3、三种跨路路由、prepared replay、
CPU／GPU 混合来源、下一 token 与所有权交接，weighted max_abs=0；但无 trace 的
30K×4 同 EXE A/B/A 为47.4672／48.3654／46.9082ms，event 比两次旧路径均值慢1.1777ms。
因此最终不保留它的 event 资源、重置跟踪和新执行控制；`pager_gpu_copy.rs` 与
`recorder.rs` 恢复原样。源码快照、失败及数值正确的版本都保留为证据。

### IQ3 计算路径拓展（v128）

只扩展已有 chunk32 partial→整专家 copy join→原生树 reduce，不改变同步策略。
新增 `kernels.vulkan.decode_gpu_resident_splitk_iq3_hit`，默认 false，仍需
`decode_gpu_resident_splitk_hit=true`；仅 IQ3S／IQ3S／IQ4NL、普通独立四路、
NE2560／NFF640／top10 及原能力／语义证明均成立时可用。
图级预检传递实际 dtype，不再硬编码 IQ2；所有层复用已有7.8125MiB scratch，
不在冻结后扩池。35B、单／双路及其他不合格形状保留旧路径。

同条件模块对照（原生 FFN → classic split；包含整专家搬运、H／M 计算及最终归约，单位 us）：

| IQ3 路数 | 全 hit | 一个整专家 miss | 决定 |
| --- | ---: | ---: | --- |
| 1 | 108.22 → 106.62 | 152.35 → 188.75 | miss 回退，不入选 |
| 2 | 174.13 → 148.71 | 203.88 → 239.56 | miss 回退，不入选 |
| 4 | 320.10 → 246.95 | 391.81 → 299.58 | 两种模式六组配对均更快，保留窄路径 |

v128 的552项 Vulkan、303项 Core、17项交付 CPU 测试及相关控制测试通过。
真实 IQ2／IQ3 跨路／prepared／混合 CPU＋GPU recovery／下一 token／handoff 验证通过，
weighted max_abs=0，H partial 新写、M partial 保留毒值、guard／资源代次正确。

无 trace 原生只做必要的 FlashNext 4.27bpw、30K×4 同 EXE 对照，
固定 VRAM23g／RAM48g／ubatch3072／Q8，不开 MTP，256 token 预热＋每路128 token 测量。
原 IQ2 split 始终开启，单独切换 IQ3 拓展；role transfer split 两侧均关闭。

| 顺序 | IQ3 拓展 | 每批前向 ms | 聚合 tok/s |
| --- | --- | ---: | ---: |
| 1 | 关闭 | 46.886 | 85.31 |
| 2 | 开启 | 46.420 | 86.17 |
| 3 | 关闭 | 48.260 | 82.88 |
| 4 | 开启 | 46.892 | 85.30 |

均值47.573 → 46.656ms，约省0.917ms（1.93%时间）；各次输入与四路输出 hash 均相同。
两侧均为24,787个 arena 专家块、15.86GiB arena、31.02／47.17GiB导入覆盖。
关闭状态本身波动约1.37ms，不能把均值当作稳定的普遍增益；150K和其他设备未重新做原生确认。
因此只保留显式、默认关闭的四路试验，不放宽路数、设备能力或默认策略。
最终没有新增运行时 event、队列、所有权状态或 CPU 同步。

原始证据和汇总均在 `local-work/{evidence,results}/infr/20261025-130100-gpu-owned-decode-c741/`：
`hitfirst-v3-final.md`、`hitfirst-v3-joint-analysis.json`、`splitk-iq3-native-v128-30000.json`，
源码／成品对应关系见 `hitfirst-v128-final-equivalence.json`。失败原型和反例不删除。
本轮本地提交：`25a30a35f` 保留多方案对照测试；`434037b39` 仅保留上述 IQ3 窄路径。

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

本轮再测 mixed dispatch：独立 workgroup 搬 Down 与算 miss GU，另测 GPU 提前／延后
job 分类，计入分类 launch 和 barrier。比较同宽256×8／256×32及原1024×8控制，
通过1／2／4／8路数值、status5／零 mask、别名及 stale W0 拒绝测试。
同宽 Q5 mixed 略慢（约106.1→107.3us），Flash 相比原调度回退；
分类再增加开销，未超过已保留流水。只提交测试与 shader 证据，不注册新运行路径。

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

551 项 Vulkan CPU、302 项 Core、17 项交付通过；
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

### v117 split-K 独立配对

以下为同一冻结 v117 EXE 开关比较，不是对旧原版重测；参数同上，
每路 warmup256 tokens、正式128 tokens，四路不同普通思考问题。计时无 trace，
表内整批时长取各路 API engine token duration 的均值，不是单路延迟或纯 GPU 时间。

| 工况 | 关闭 ms/轮 | 开启 ms/轮 | 净省 | 聚合 tok/s |
| --- | ---: | ---: | ---: | ---: |
| FlashNext 4.27bpw，30K×4 | 48.465510 | 46.627893 | 1.837617ms／3.79% | 82.53→85.79 |

四路各128 tokens完整结束，4／4最终文本一致，输入与成品身份相同。
只有一次成对原生测试，不作为统计稳定性结论；不与旧256-token测量的收益直接相加。
源文件 hash 逐项匹配冻结成品；SHA256：
`13BFCEAC60DF4DD91661710B0555DAD4DD69B8E0F61C821CCB2DA7FABA3632E8`。

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
| `498739daa` | split-NFF hit FFN、原生归约树及模块对照 |
| `85e506a27` | mixed Down-copy／miss-GU 与预分类对照，未进入运行路径 |
| `040aae67f` | 默认关闭的四路 IQ2S split-K 接入、池预留和实际 adapter 强数值回归 |

任务 ID：`20261025-130100-gpu-owned-decode-c741`。
生成物统一在 `D:\AIinfr\infr\local-work`：
`evidence/infr/<task-id>` 保存 manifest、输入／exe身份、原始 log 和失败方案；
`results/infr/<task-id>` 保存 `roadmap-native-summary.json/md`、
`v103-v104-qualification.md`、`v105-score-hoist.md`、D3 审计、
`fusion-v2-checkpoint.md`、`fusion-v2-results.json/md`、`fusion-v2-final.md` 和
`splitk-native-v117-30000.json`；`fusion-runtime-v117-equivalence.json` 保存最终源码与成品核对。
详细本机实验账本：`upstream-research/decode-overlap-roadmap.md`。

共用通用 x86-64／static-CRT 构建缓存，冻结 exe 再测；原生最多四路，模块保留八路。
超过1核或512MiB的工作独占 `D:\cpuandram.lock`，自身进程退出后释放，不碰他人锁。
磁盘规范唯一来源：`D:\AIinfr\disk_usage_rule.md`。
