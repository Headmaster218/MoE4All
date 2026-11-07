# Qwen3.8 MTP 调研与实现路径

后续报告：[优化后的测量与优先级](2026-09-23-mtp-bottlenecks.md)。后续报告修正了下文对
queue 计时的解释，并记录了受控的 hidden-readback 实验。

## 实现更新

第一轮优化现已实现。固定 benchmark 条件保持不变：24 GiB VRAM、48 GiB RAM、ubatch 3072、
parallel ubatch 256、context 4096、greedy sampling，以及相同的 64-token 后续文本。

| 实现版本 | Decode |
| --- | ---: |
| Ordinary target | 29.9 tok/s |
| Initial four-token MTP | 12.0 tok/s |
| Pending-target protocol, fixed buffers and small-row MoE work | 22.9 tok/s |
| Fixed three-row VERIFY and width-sized draft | 24.7 tok/s |
| Multi-row PLE gather parallelism, run 1 | 26.8 tok/s |
| Multi-row PLE gather parallelism, run 2 | 26.4 tok/s |
| Production path without pager profiling | 26.0 tok/s |

本轮生成的 64-token 后续文本已与普通 greedy Decode 逐字节一致。这仍只是范围有限的正确性检查，
不能替代多 prompt 和边界测试。

A production-shaped 的 256-token 比较为两条路径使用了相同的较长 prompt。MTP 为 31.9 tok/s，
普通 Decode 为 33.7 tok/s。两个输出文件的 SHA-256 均为
`421f724070183a6324adb00621d6e7285ea1844e05577c844cde407f4f66d10a`.
因此，较长测试中 MTP 达到普通 Decode 速度的约 95%，但尚未证明有性能提升，也未达到 40 tok/s 目标。

已实现的改动：

1. Target prediction 提前一个 token 保存。每次 VERIFY 从已知的 target token 开始，因此至少有一行
   会提交；发生拒绝时可直接恢复到已接受的 recurrent row。独立的 target correction pass 和
   MTP-head catch-up pass 均已移除。
2. Target VERIFY 输入/输出 buffer 改为固定分配，并在 unified expert arena 最终确定前申请。
   PLE 行准备就绪后，VERIFY 使用一次 target graph 执行。
3. Decode scratch 保留有界的 draft/VERIFY 交替拓扑。多行因果 VERIFY 支持 shared-expert fusion
   和逐行 hit-first mask。后续[单路 Decode 调研](2026-09-23-single-decode.md)确认，shared-slot
   fusion 并不覆盖本模型实际执行的 IQ2_S/IQ4_NL/IQ3_S 格式，因此这些 bank 当前并未启用该融合。
4. Draft 长度随 VERIFY 宽度调整。最后一个 draft step 仍会写入 MTP KV，但跳过不再使用的 HC head、
   vocabulary projection、argmax 和 readback。
5. 多行 PLE batch 使用四线程 gather pool。每次 64-token 测试的 PLE 工作量从约 249 ms 降到
   80～83 ms，暴露出的 wait 从约 243 ms 降到 74～76 ms。
6. `spec.k` 是上限。Qwen3.8 最多选择三行 VERIFY：固定宽度测试中，两行是 25.9 tok/s，三行
   26.4～26.8，四行 24.9。由拒绝情况驱动的 2/3/4 自适应策略为 25.5 tok/s，已移除。

本轮测得但未保留的方向：

- Forcing the small-row MMQ path measured 19.9 tok/s.
- Extending next-layer expert prefetch to multi-row VERIFY measured 21.3 tok/s;
  nearly all predicted candidates were already resident.
- Disabling the submit splitter reduced submissions from 3970 to 3874 but left
  throughput unchanged at 26.5 tok/s.

剩余 target 开销中包含结构性同步边界。一个代表性的三行测试在 2.84 秒时间跨度内，main-queue
timestamp 覆盖 1.40 秒；每个 target layer、每个 VERIFY cycle 都有一次 paging sync，32 个 cycle
共等待 1,536 次，累计 1.36 秒。0.99 秒的 recorder lifetime 也包含主机准备工作，而不只是
command encoding。这些测量彼此重叠：paging sync 会等待之前的 GPU 计算和数据传输，queue 覆盖
范围也不包含未计时的 DMA。它们**不能**用来推导纯计算下限，也不能证明一定能达到 40 tok/s。
后续报告测算了每 cycle 必须减少的耗时，并先确认较小的优化机会，再建议采用 GPU 常驻的 routed-hit
调度。

## 范围和基线

本调查基于 `perf/qwen38-24g48g-ub3000` 分支的 `901d332`，只加入诊断改动，没有修改推理算法
或 kernel。

- GPU：AMD Radeon RX 7900 XTX。Vulkan 枚举顺序可能随进程变化，应按日志中的设备名称识别，不要
  假定 Vulkan index 永远不变。
- Target: `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`.
- Head: `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
- RAM/VRAM/ubatch/parallel ubatch：48 GiB / 24 GiB / 3072 / 256。
- 最终配对测试：context capacity 4096，prompt 为 `北京是什么？`（15 个 prompt token），输出
  64 个 token，temperature 0，不启用 thinking，每次只运行一个请求。
- 开启 `INFR_PAGER_PROFILE=1`、`RUST_LOG=info`；关闭 `INFR_PROF_STAGES` 和逐 op profiling。
- 较早的诊断测试使用 context 512，因此有效 ubatch 被限制为 512；它不代表完整的 3072-ubatch
  内存配置。

| 最终配对测试 | Decode |
| --- | ---: |
| Ordinary | 29.9 tok/s |
| MTP, four candidates and one four-row VERIFY | 12.0 tok/s |

这些都是短上下文单次运行，不代表长上下文性能。生成的后续文本不同，因此这是相同请求条件下的
比较，不是 token 序列完全一致的 microbenchmark。

原始日志是本地忽略文件，位于 `target/mtp-research-{async,baseline}`，后缀为 `.err.log` 和
`.out.log`。更早的同步诊断日志为 `target/mtp-research-profile.err.log`。

## 测得成本

异步 MTP 测试运行了 27 个 draft/VERIFY cycle，并进行了 26 次 correction。下表各 scope 的平均值
不包含 warmup；prompt 初始化也不在这些统计范围内。

| 阶段 | 墙钟耗时（ms/次） | Backend scratch setup（ms/次） |
| --- | ---: | ---: |
| Four-token draft | 12.86 | 5.88 |
| Four-row VERIFY | 107.45 | 11.48 |
| Head KV catch-up | 21.28 | 5.03 |
| Restore accepted recurrent state | 0.38 | 0 |
| One-token target correction | 47.13 | 11.14 |
| New baseline snapshot | 0.46 | 0 |

VERIFY 平均有 136 次 queue submission、55.05 ms 已记录的主机同步等待、86.41 MiB host memcpy/
ReBAR push，以及 208.47 MiB 专用 DMA payload。Head catch-up 没有分页专家流量，但 command
recording/lowering 仍记录到 15.77 ms。Lowering 包含 lazy workspace allocation，不代表 shader
执行时间。

整轮 profiling 的 scratch setup：MTP 为 909.5 ms，普通模式为 59.8 ms。这些总数包含 prompt
执行，而且 target row 数不同；分析 MTP 各阶段时应使用上方逐 scope 计数。Backend setup/record/
sync 计时彼此重叠，**不能**当作独立墙钟耗时相加。

MTP profiling 测试中，PLE 剩余 wait 只有 23.7 微秒。Host-tier read 和 mmap fallback 均为 0。
这次测试不支持将 ngram 或 SSD read 视作主要的暴露停顿来源。本设备/运行条件下没有完整 DMA
GPU 计时（`gpu_timed_submits=0`）；报告的 DMA GPU 时间为 0 不代表 DMA 没有成本。

按 64 / 27 = 每 cycle 输出 2.37 个 token、普通模式 29.9 tok/s 计算，当前接受模式要打平，完整
cycle 平均需低于约 79 ms。单 VERIFY 目前就已超过这一预算。修复 allocation 有价值，但不足以
单独证明 MTP 能提速。

## 已确认的架构问题

1. 固定 MTP runtime 的资源所有权尚不完整。Head 权重、KV 和输入 buffers 会预先分配，但 unified
   arena 建立后，graph 内部资源和逐 op scratch 仍经由通用 executor 分配。
   `Qwen4MtpSession::{draft4,catch_up}` 每次调用都会重建 graph。
2. `RuntimePhaseArena` 只保留当前 topology 和一个暂存 topology。Draft graph 包含 MoE，即使 head
   expert 已常驻，也会进入同一个 backend 全局 cache，与 target layer-0、剩余层和 correction
   graph 竞争。Catch-up 不包含 MoE，因此 `paged_static_phase` 返回 None，scratch/pool 每次调用时
   都是局部资源。
3. Qwen VERIFY runner 每次都会单独分配 input、position、hidden、PLE、logits 和 result buffer，
   没有复用固定的四行执行对象。Position 和 KV length 已写入 graph operation，直接沿用旧 plan
   会导致错误。
4. VERIFY 缺少普通 Decode 中的重要 MoE 优化。Shared-expert fusion 只支持一行；hit-first 按需
   promotion 只支持一行或相互独立的序列行；四行因果 speculative row 两者都不满足。模型的
   expert-prefetch hint 也只在 batch=1 时生成；本次测量中 prefetch 未启用，因此不能宣称扩展
   hint 路径已有收益。
5. Candidate 0 会与已知的 `target_prediction` 比较，但检查发生在完整 VERIFY 之后。即使已知不匹配，
   仍然会执行四行 target forward、捕获 trace，再恢复基线状态。分阶段诊断的 27 个 cycle 中，有
   7 个属于这种情况。
6. 每次部分接受都会立即执行单独的 target correction forward。Prefix-state restore 已去掉对已接受
   token 的重放，但没有去掉这次额外的完整模型前向。状态恢复现在很便宜，correction forward
   及其 workspace churn 仍然昂贵。
7. Prompt priming 会将整个 prompt 经由全行 VERIFY 处理，包括分配/计算 `m * vocab` logits，尽管
   最终只使用最后一个 prediction。与普通 Prefill 不同，这条路径不会按 ubatch 对 target forward
   分块。15-token 测试无法暴露长 prompt 下的时间和内存风险。

相关代码入口：

- `crates/infr-llama/src/mtp/qwen4.rs`: head graphs, fixed resources, cycle protocol.
- `crates/infr-llama/src/seam/runner.rs`: Qwen VERIFY, prompt ingestion and correction.
- `crates/infr-vulkan/src/adapter.rs`: `RuntimePhaseArena`, `execute_static_inner`,
  `paged_static_phase`, `paged_moe_shared_at`, `execute_paged_moe`.
- `crates/infr-llama/src/seam/weights.rs`: recurrent snapshots and accepted-row restore.

## 实现顺序

### 0. 建立正确性门槛

MTP 与普通 greedy 输出仍会在以 `创新中心` 结尾的短语之后发生分歧。State-trace 改动并未引入
这一差异，但这**不代表**它只是无害的数值噪声。当前仍不应默认启用 MTP。

增加 same-prefix teacher-forced 比较：保存相同的初始状态，先批量运行 4 个已知输入 token，再恢复
状态并逐个运行相同 token。比较每行 raw logits、argmax margin、hidden state 和 recurrent state，
然后定位首次产生差异的 layer。另需将恢复后的 0～4 长度 prefix 与逐 token 参照状态比较，并覆盖
PLE 和 QSA boundary。还要测试输出组中途遇到 EOS 的情况：当前 inner-loop 的 break 不会终止外层
generation loop。

### 1. 使小型 MTP 运行时常驻

保持四 token draft 和四行 VERIFY 策略不变。MTP 应显式拥有 draft、target VERIFY layer-0/main、
correction 和 head catch-up 在 1～4 行 shape 下所需的 workspace。按现有内存设计，在最后一次
实时 VRAM 查询和 unified-pool 创建前分配所需固定 buffers。Graph tensor 和逐 op scratch 都要保留；
仅缓存 `compile()` 不够，因为当前 `compile()` 只是包装/克隆 graph，并不会重新编译 shader。

应由调用方拥有执行 workspace，或在 backend 中使用显式 workspace identity；不要使用无界 topology
cache，也不要复制一份独立 model/backend。依赖 position 的字段必须继续更新。保留普通大 Prefill
arena 的释放规则，不要将所有 Prefill buffer 都改成永久常驻。

复用四行 IO 和 logits buffer；在可行时让 hidden-state handoff 保持在设备端，CPU 只需读回接受结果
ID 和实际输出 token。删除未使用的 catch-up scratch 声明。第一阶段验收条件是稳态不再 allocation，
且 expert residency 稳定；目标是每 cycle workspace 准备时间低于 5 ms，再测真实端到端收益。

### 2. 为因果 VERIFY 提供优化的小行 MoE 调度

将 hit-first mask 和 shared-expert fusion 扩展到最多四个因果行的局部 MoE operation。虽然
attention 和 recurrent state 有因果关系，同一层中的 routing/FFN 行彼此独立。先统一 staging
所需 expert 集合，在 miss 传输期间执行 resident slot；保持每行原有 router slot/reduction 顺序，
最后补齐缺失 slot。

不要在 VERIFY graph 上设置 `Graph::independent_rows`，否则会改变 attention/recurrent 语义。
也不要为了强制走 Prefill MMQ 路径而盲目降低全局 small-m threshold；必须在实际模型 shape 上
测量其分桶开销和数值行为。

验证 paged/resident 等价性、混合量化类型、shared slot 和所有接受 prefix。也要测试普通单/双 slot
Decode，因为 executor 是共享的。Vulkan validation 应在计时测试之外运行。只有完成这套调度的测量后，
才用逐 kernel profiling 选择特定的四行 dense/LM-head/DeltaNet 优化；本次调查尚未分离这些项目各自
的贡献。

### 3. 移除可证明浪费的周期和传递

较小的独立改动：启动 trace/VERIFY 前先检查 candidate 0。若不匹配，就用已知 target token 让 head
追上状态，并只执行必要的 correction。不要截断有效 trunk state，也不要恢复从未生成过的 trace。
增加强制首次拒绝时的 token 等价测试。此项可与步骤 1 并行开发，不改变四 token 策略。

较大的后续改动：将 correction token 或全部接受时的 bonus token 作为 pending target input 带入下次
VERIFY，避免立即执行一次仅用于 correction 的前向。这需要显式的 pending-token 状态机。若仍保留
4 个实际 draft candidate，就需要验证 5 个输入；若 VERIFY 仍只有 4 行，则只能保留 3 个 speculative
row。应按要求单独决定这项策略，不要悄悄改变当前四 token 基线。

### 4. 完成长上下文集成并调优策略

复用现有 Prefill 路径对 prompt priming 分块；按有界 chunk 提取 hidden state 供 head catch-up 使用，
并且只计算 frontier 所需的 logits。测试 QSA、segmented-KV boundary 附近的场景，以及 max-new 限制、
EOS 和取消请求。之后比较 2/3/4 candidate 策略、confidence gate 和有收益时的回退；依据每 cycle
实际输出 token 数与墙钟时间评估，不能只看聚合接受率。测试回退路径时应保持 MTP head 常驻：卸载
head 会改变 target expert-cache budget，属于另一种比较条件。

## 测量注意事项

- 启用 `prof.stages` 或 `prof.ops` 后，`Recorder::finish_nowait` 会变成阻塞操作。生产形态的
  overlap 应使用 scoped pager counter；逐 op timestamp 只用于分析 kernel，不能据此宣称端到端收益。
- 本次测得 MTP expert arena 为 14.10 GiB，普通模式为 16.31 GiB。固定 head/state 确实消耗 VRAM，
  这是合理成本，测试中必须保留。
- 先前日志中的 allocation retry 来自显式的剩余预算保护；单凭此现象不能证明 Vulkan 碎片化或
  head 放置错误。
- GPU occupancy/power 本身无法区分 dispatch、allocation、传输和计算瓶颈。Main-queue timestamp
  不包含未计时 DMA，不能视为完整的设备利用率测量。
- 宣称性能提升前，应在多个上下文深度和多个 prompt 上重复测试，每次至少生成 256 个 token。保持
  RAM/VRAM/ubatch 基线不变，并保留原始输出以验证正确性。
