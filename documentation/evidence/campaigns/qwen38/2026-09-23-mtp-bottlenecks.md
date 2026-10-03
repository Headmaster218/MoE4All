# Qwen3.8 MTP：第一次优化后的瓶颈

> 实现后更新：本文保留原始诊断基线。当前结果以及最终保留或否决的改动，见
> [实现后的后续记录](#已实现的后续工作)。

后续记录：[普通单序列 Decode 测量](2026-09-23-single-decode.md)，其中还修正了 MTP 也会受影响的
shared-expert fusion 格式覆盖范围。

## 范围和可复现性

基线：`perf/qwen38-24g48g-ub3000` 分支上的 `72df13d`，初始 worktree 干净。最初的诊断只在
`INFR_PAGER_PROFILE` 下记录 cycle、接受情况和 VERIFY 细节，没有启用会阻塞执行的
`INFR_PROF_STAGES` 路径。后续章节记录最终保留的优化。

- AMD Radeon RX 7900 XTX；应按枚举出的设备名称选择，不要写死设备索引。
- Target: `D:\AILMStudioModels\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
- MTP head：`G:\Qwen3.8-Flash-Next-UD-Q2_K_XL\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`。
- VRAM / RAM / ubatch / parallel ubatch：**24 GiB / 48 GiB / 3072 / 256**。
- Context capacity 为 4096，temperature 为 0，seed 为 1，不启用 thinking，生成 256 个 token。
- 固定 runtime 支持 **2 到 4 行 VERIFY**。`spec.k` 控制此范围，不再有隐藏的三行上限；默认值
  对应 4 行，即 1 个待处理的 target token 加 3 个 speculative token。
- History prompt 与前一项 256-token benchmark 相同，是一个 24-token 的北京历史、文化和地理
  问题。另一个 38-token prompt 要求实现 Rust bounded LRU cache，并解释复杂度和测试。
- 这些是短上下文、单请求测量结果，不代表长上下文或并发服务性能。

本地被忽略的产物位于 `target/`：`mtp-profile-run.cjs`、
`mtp-profile-analyze.cjs`, `mtp-op-analyze.cjs`, and each run's `.meta.json`,
`.err.log`, `.out.log` and, for MTP profiles, `.analysis.json`.
运行脚本会清除继承来的 `INFR_*` 实验参数、记录命令行参数，并直接捕获 UTF-8 输出，避免
PowerShell 对 stderr 再包装。

在 workspace 根目录下运行的示例：

```powershell
node target/mtp-profile-run.cjs mtp-current-history-1 mtp pager history 256
node target/mtp-profile-run.cjs ordinary-current-history-off ordinary off history 256
node target/mtp-profile-analyze.cjs mtp-current-history-1
```

## 吞吐和验收

| 测试项 | Profiling | Decode tok/s |
| --- | --- | ---: |
| Ordinary, history | off | 31.7 |
| MTP, history | off, first / repeated | 30.6 / 30.2 |
| Ordinary, history | pager | 30.2 |
| MTP, history | pager | 31.3 |
| MTP, Rust code | pager | 31.1 |
| MTP, Rust code | off | 30.3 |
| Ordinary, Rust code | off | 30.0 |
| MTP, hidden Readback experiment, history | pager | 32.2 |
| MTP, hidden Readback experiment, history | off | 34.1 |

不要混用不同 profiling 模式来宣称性能提升。这些都是有明显波动的单次运行；旧的 31.9/33.7
数据属于历史结果，不是本轮配对基线。普通 Decode、MTP、重复运行和 Readback 实验生成的
history 后续文本逐字节一致，其 UTF-8 SHA-256 为：
`99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`.

| MTP pager 测试 | cycle 数 | 接受的 draft token 数 | 接受 0 / 1 / 2 个 draft 的次数 | 每 cycle 输出 token 数 |
| --- | ---: | ---: | --- | ---: |
| History | 108 | 149/216 = 69.0% | 21 / 25 / 62 | 2.370 |
| Rust code | 104 | 153/208 = 73.6% | 15 / 25 / 64 | 2.462 |

最后一个 cycle 可能提交多于 `max_new` 实际输出数量的行。估算吞吐时应使用 256 除以 cycle
数，而不是 `(cycles + accepted_drafts)/cycles`。所有基线与 Readback history 测试的接受模式相同。

Rust 代码样例与普通 greedy 输出**不一致**：开头的 cache 定义和几条注释相同，之后 MTP 输出
`But VecDeque`，普通 Decode 输出 `But Vec`，后续文本随之分叉。因此 history 的 hash 一致并不能
证明整体正确。代码 prompt 的吞吐比较也不是基于相同 token 序列。默认启用 MTP 前，必须在相同
prefix 下通过 teacher-forced 方式比较批量/逐 token logits 和循环状态；不能假设差异只是无害的
数值噪声。
关闭 profiling 后重复运行 MTP，得到的代码样例 hash 与 pager 测试相同
（`c5abdeec2bf38d000dffc28aed8952fb8f031a0119db4b656eedad60bc566312`），但与普通模式的
`2c9aad1c2e4cfc870d4d730a04198e45c7bbdc15e7bbe9490ba466c84926e1bd`。
因此，输出差异不能简单归因于启用了诊断日志。

## 可加的墙钟时间分解

按 history pager 测试计算的每 cycle 耗时；不包含 prompt priming。

| 环节 | 平均耗时（ms） | 说明 |
| --- | ---: | --- |
| Draft | 3.46 | 包含常驻的 MTP head |
| Target VERIFY execution | 63.61 | GPU 计算及主机侧 paging/orchestration |
| VERIFY hidden/ID readback | 5.04 | execution 完成后发生 |
| PLE 准备等待与上传 | 2.41 | 暴露出来的 wait，不是异步工作的总耗时 |
| VERIFY graph 构建与 bindings | 0.21 | 不包含 shader 编译 |
| VERIFY 输入准备 | 0.01 | 复用固定 buffers |
| 其他 VERIFY wrapper 工作 | 0.12 | 与完整 VERIFY 范围的差值 |
| Restore 与 snapshot 记录 | 0.70 | 已避免重放 target token |

完整 VERIFY 平均耗时 **71.40 ms**，约占一个 cycle 的 94%。代码 prompt 的 VERIFY 平均为
74.78 ms，其中 execution 为 67.01 ms，readback 为 5.09 ms。当前主要优化方向既不是 draft，也
不是 restore。Scratch setup 每次 VERIFY 的中位数为 0.03 ms；0.55 ms 的平均值包含首次 shape
准备。继续做通用 graph/scratch 缓存不是当前优先项。

按每 cycle 输出 2.370 个 token 计算，要达到 40 tok/s，**每个完整 cycle 必须不超过
59.26 ms**。当前 cycle 约为 75.7 ms。在其他成本不变时，VERIFY 必须从 71.4 ms 降至约
**55 ms**，减少约 16 ms（23%）。代码 prompt 对应的 VERIFY 预算约为 57 ms。这是目标预算，
并非对可达速度的预测。

即使完全消除 draft，每次 history 测试也只节省约 0.37 秒，单靠这一项无法达到 40 tok/s。

## 已确认的低风险机会：隐藏状态回读

`seam/runner.rs` 将固定的小行数 `h_out` 和较大的 prompt fallback 都分配为
`BufferUsage::Staging`。Vulkan 将其映射为 `CpuToGpu`，而 `Readback` 对应 `GpuToCpu`。
`VulkanBackend::download` 会用 `copy_nonoverlapping` 直接读取映射 buffer，不会通过额外的
cached staging copy 来补救不合适的内存放置。

一次只改两行的 A/B，仅将这些 hidden-output allocation 改为 `Readback`：

- Readback **5.044 -> 0.016 ms/cycle**; 544.78 -> 1.73 ms over 108 cycles.
- IDs, acceptance distribution, expert misses and transfer bytes unchanged.
- Full VERIFY **71.40 -> 69.21 ms** in the profiled samples: execution itself
  varied from 63.61 to 66.32 ms, so not all of the isolated saving appeared as
  end-to-end improvement in that pair.
- 关闭 profiling：基线 30.6、实验 34.1、恢复基线后 30.2 tok/s。结果有希望，但目前只有一次
  实验样本，不能据此承诺稳定的百分比提升。

上面记录的是诊断阶段结束时的状态。现在，固定四行 hidden buffer 及其较大 fallback 在源码中
都保留了 Readback 放置方式。这些固定 buffer 仍在查询实时空闲 VRAM、创建统一专家 arena 之前
分配。

这一发现与 Vulkan 的警告一致：读取 uncached/write-combined 的 host-visible memory 可能很慢。
参见
[Vulkan memory specification](https://docs.vulkan.org/spec/latest/chapters/memory.html).
本机的优化优先级是由实测结果决定的，而不只是依据这条通用警告。即使应用此改动，固定资源的
分配顺序也必须继续早于实时空闲 VRAM 查询和统一 arena 的创建。

下一步应让已接受的 hidden row 留在 GPU 上，并传递或复制到 `draft_h`。只有 3 个接受结果的 ID
需要读回 CPU。现有的 `Backend::copy_buffer_ranges` 可以选取已接受的行；最好把复制并入已有
submission，而不是增加一次阻塞提交。这是 MTP/runner 的局部接口调整，不需要新增模型、backend
或 pager。

## 专家计算和分页边界

另一次 64-token、`INFR_PROF_OPS=1` 测试用于识别 kernel。该模式会关闭正常的异步 overlap，并
输出大量逐次提交报告；其中 5.8 tok/s **不是生产吞吐数据**。这轮诊断期间还进行了 CPU 编译，
因此不能用其绝对墙钟时间或 device milliseconds 预测生产环境收益。Phase marker 可在分析
VERIFY kernel 分布时排除 prompt 和 draft：

| VERIFY timestamp 分类 | 占已插桩 VERIFY device interval 的比例 |
| --- | ---: |
| `native_idm_iq2s_paged` | 38.0% |
| `native_idm_iq4nl_paged` | 14.2% |
| `native_idm_iq3s_paged` | 4.7% |
| `deltanet_seq_trace` | 7.3% |
| 三行 dense projection | 数个 1～4% 的分类 |
| Target vocabulary projection | 2.2% |

三个 routed-expert 分类合计约占本次诊断 device time 的 **57%**。因此，目标模型并非只是在
空等专家权重传输；专家计算和 dequantization 本身也是显著热点。

在重新设计 pager 之前，有两项明确的代码级优化机会：

1. `native_gemv_id_multi.comp` and its `_sg` twin call `GRID_INIT` **before**
   checking row/slot masks. Hit-first and miss-second dispatches therefore both
   initialize IQ codebooks for groups that immediately return. Prototype a uniform
   workgroup-level mask/bounds decision before initialization; preserve valid
   barrier control flow and prove that active groups' arithmetic is unchanged.
   The later ordinary-decode investigation confirmed that shared-slot fusion
   currently supports Q5_K/Q6_K/IQ4_XS, not this model's IQ2_S/IQ4_NL/IQ3_S banks.
   Skipping IQ initialization for a shared Q8 slot is therefore an additional
   requirement for a future format extension, not a current measured hotspot.
2. The current small-row grid independently handles each `(token row, expert
   slot, output row)`. Repeated experts across speculative rows do not explicitly
   share weight decode. Prototype a 2-4-row expert microkernel or persistent
   read-only codebook buffer, starting with IQ2_S. Benchmark actual quant/shape
   combinations and register/LDS pressure. Do not force generic MMQ or globally
   enable the SG route: previous measurements and `native_id_sg_choice` document
   regressions for these low-output IQ2_S shapes.

结构性边界仍然存在：`execute_paged_moe` 检查的是整层 expert bank 是否常驻，而不只是本次选中
的专家。否则 `sync_stream` 会提交并 drain 先前工作、在 CPU 上读取 router ID，然后生成 hit mask
并准备 hit/miss 执行。每次 VERIFY 有 48 个这样的边界。hit-first 已经让 miss 准备与常驻专家
计算 overlap，因此再增加通用 prefetch 开关只会重复已有工作。

History VERIFY 平均每次有 107 次 queue submission，每个 cycle 在 paging-sync scope 内累计
36.06 ms。但这 36.06 ms 包含先前计算、queue submission、drain 和 recorder 获取，**不等于专家
I/O 空等了 36 ms**，不能直接当成可追回的时间扣除。仅用 fence 替代 queue idle，仍然存在
router→CPU→expert 的依赖。参见
[Khronos synchronization example](https://docs.vulkan.org/samples/latest/samples/performance/wait_idle/README.html)
了解 drain 整个 queue 与等待特定 work 之间的区别。

真正的 overlap 设计需要 GPU 可读的 residency map 和 GPU hit mask、提前发出的 router readback
完成信号、在 CPU promotion 完成前提交常驻专家 work、每次 submission 固定的 LUT，以及在读者
完成前 pin 住对应 slot。不能只凭 router fence 就重置当前 LUT tape 或覆盖 LRU cell：目前只有
完整 drain 能保证它们的生命周期。该改动涉及 Vulkan pager/executor，而不是模型因果计算图；应在
较小改动完成后，作为独立的正确性与性能里程碑处理。

## 传输和其他次要成本

- 47.17 GiB 的 routed expert 全部放入 host store；没有记录到运行时 expert SSD 或 host-tier
  miss 读取。但这不代表独立的 PLE mmap/dequant 工作没有成本。
- 只有 31.02 GiB 的 host expert memory 被导入用于 direct DMA。History VERIFY 每个 cycle 通过
  DMA 传输 116.33 MiB，通过 host/ReBAR push 传输 54.58 MiB；后者占用 3.82 ms CPU 时间。代码
  prompt 对应为 147.69 MiB、85.16 MiB 和 5.51 ms。这些操作与其他工作重叠，不能简单相加为停顿。
- 包含 prime 在内的 8.73 秒时间段中，main-queue timestamp 覆盖 4.54 秒。本轮 DMA 没有 GPU
  timestamp。未覆盖区间不能精确代表硬件空闲时间，这个比例也不能证明 40 tok/s 的上限。
- 本轮测得的统一 arena：MTP 为 14.59 GiB，普通模式为 16.92 GiB。MTP 专家缓存较小是固定资源
  带来的真实代价。后续比较普通回退路径时，应让 MTP head 继续常驻，而不能只与未加载 head 的
  全新进程比较。
- PLE 每 cycle 暴露出的 2.4 ms 是次一级优化目标。可在输入已知时更早异步提交，但不要为了消除
  当前 wait 引入更多 execute boundary。
- Recurrent trace 写入比 restore 更值得关注：`deltanet_seq_trace` 是可测得的 kernel 分类，
  但 prefix state 只需记录 `n-1` 个。保留对已接受 prefix 的直接恢复，不要重新引入 replay。

## 建议的交付顺序

| 优先级 | 工作项 | 验收条件 |
| --- | --- | --- |
| P0 | 修正 hidden readback 放置，再实现 GPU hidden handoff | Readback 低于 0.1 ms；接受结果和输出一致；重复配对测试确认端到端收益 |
| P1 | IQ codebook staging 前执行 mask；另行扩展 shared-slot 格式覆盖 | mask 后专家输出逐位一致；普通/MTP 回归测试通过；关闭 profiling 仍有收益 |
| P2 | 针对 IQ2_S/IQ4_NL 优化小行数 kernel，并复用权重/codebook | 特定 shape 的 microbench 有收益，且完整 cycle 变快，而非只有 kernel 变快 |
| P3 | GPU routed-hit 调度与细粒度 pager 生命周期管理 | 减少暴露的 router boundary，且无过期 LUT/slot 竞态或行语义变化 |
| P4 | PLE overlap、trace 带宽、有界宽度/回退策略 | MTP head 常驻时按实际输出 token 和墙钟时间评估；成本变化后重测宽度 |

每个优先级都必须先满足正确性门槛：使用相同输入 token prefix 和 logit margin，定位 Rust 样例
首次分歧的位置；再将所有 accepted-prefix recurrent/PLE state 与逐 token 执行结果比较。

不要假定各项收益可以独立相加。按当前接受模式，近期目标是先稳定达到普通 Decode 的速度，
再将 history 样例的完整 cycle 降至 59 ms 以内。每完成一个里程碑都要重新测量。

每项改动落地后，都应在关闭 profiling 的情况下交替运行 ordinary/current/new，至少重复三次；
保留 token ID、输出和 cache budget。覆盖拒绝 prefix、全部接受的 cycle、EOS/max-new 截断、混合
量化、shared slot、QSA boundary，以及普通的一槽/两槽 Decode。Vulkan validation 必须在计时运行
之外执行。长上下文服务仍需单独验收：MTP prompt-prime 路径仍会生成所有行的 hidden/logits，且
没有使用普通的分块 Prefill，因此短 prompt 测量无法证明其内存和耗时风险已解决。

## 已实现的后续工作

最终保留的 MTP 改动如下：

- 将 VERIFY hidden output 放入 `BufferUsage::Readback`。实测 readback 从每 cycle 约 5 ms 降至
  约 0.012 ms。
- 复用在 unified-pool 空闲空间查询前分配的固定 ID、位置、hidden、wide residual、PLE、结果及
  recurrent trace buffers。
- 应用普通路径中的 IQ mask-before-codebook 改动。多行 A/B 中，新 shared-slot IQ fusion 为
  36.8 tok/s，而现有 dense shared 路径为 37.8 tok/s，因此只在单行场景保留该融合。
- 增加 IQ2_S 多输出行 tree variants，并通过 `kernels.vulkan.gemv.id_grid_nr` 默认选择 NR=8。
  在 64-token op 诊断中，IQ2_S VERIFY
  device time fell from 706.9 to 384.7 ms, or 45.6%, with identical output.
- 在 PLE 异步 gather 进行期间执行 target layer 0，再将固定的 hidden/wide state 传给第 1～47 层。
  可通过 `spec.mtp_ple_overlap` 控制，也可用 `INFR_NO_MTP_PLE_OVERLAP=1` 关闭。
- 移除隐藏的三行策略上限：现在可通过 `spec.k=4` 使用预分配的四行 runtime。

IQ3_S NR2/4/8 实验已否决。在相同的 108-cycle 接受率 profile 下，它将 VERIFY 平均耗时从约
54.5 ms 增至 57.0 ms，因此只保留已验证的
因此只保留已验证的 IQ2_S variants。

### 最终测量

所有结果均使用 24-token history prompt、256 个输出 token、24/48 GiB 预算、ubatch 3072、
parallel ubatch 256、temperature 0 和 seed 1。最终输出
SHA-256 未变，仍为
`99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`.

| 最终模式 | Decode tok/s | cycle 数 | Draft 接受情况 |
| --- | ---: | ---: | ---: |
| Ordinary decode | **37.1** | 256 forwards | n/a |
| MTP, three VERIFY rows (`spec.k=3`) | **40.8** | 108 | 149/216 = 69.0% |
| MTP, default four VERIFY rows | **40.5** | 96 | 162/288 = 56.2% |

默认四行结果比最终普通 Decode 快约 9.2%；三行快约 10.0%。四行虽然减少了 cycle 数，但每个
cycle 的 VERIFY 和 draft 成本也随之增加，因此在这个 prompt 上两种策略基本持平。宽度应保持可
配置，并按墙钟时间内实际输出的 token 数选择，而不是只看原始接受率。

三行 pager-profile A/B 中，layer-0 PLE overlap 令暴露的 PLE 从每 cycle 2.249 ms 降至
0.831 ms；但由于增加了 graph/submit，VERIFY backend execution 从 54.014 ms 升至 54.439 ms。
最终 VERIFY 墙钟时间从
56.606 ms 降至 55.810 ms/cycle，开启 profiling 时的吞吐从 39.1 升至 39.6 tok/s。

最低目标，即 MTP 追平优化后的普通 Decode，已经达到。理想的 20% 提升约对应 44.5 tok/s，目前
尚未达到。剩余主要瓶颈在 target VERIFY pager/executor：每 cycle 约有 29～33 ms 落在 paging
同步和 queue completion 中，每 cycle 约提交 108～113 次。这些计数包含真实 GPU 工作，不能
简单视为 CPU 空闲时间并消除。下一个可信的里程碑是实现 GPU 常驻的 routed-hit mask、固定每次
submission 的 LUT，并 pin 住 pager slot，使常驻专家计算和 promotion 能 overlap，避免当前完整的
router-to-CPU drain。

## 实现提交溯源

本 campaign 的实现变化可沿以下提交追溯。提交说明和触及文件用于确认代码落点；吞吐、耗时和否决结论仍以本页记录的具体测试条件为准。

| Commit | 阶段/变更 | 记录中的对应结果 |
|---|---|---|
| `30c60903` (`feat(qwen38): add fixed four-token MTP decode`) | 加入固定四 token MTP decode 基础路径 | 原始 MTP profiling/baseline 的实现起点 |
| `901d3323` (`perf(qwen38): restore accepted MTP prefix states`) | 保留并恢复 accepted-prefix 状态，避免重算已接受前缀 | 本文对 rollback/correction 成本的阶段划分 |
| `72df13d5` (`perf(qwen38): streamline MTP verification`) | 精简 VERIFY 路径 | 初始 bottleneck profile 的代码基线 |
| `96886a18` (`perf(qwen38): optimize decode and MTP verify`) | hidden output/readback、固定 VERIFY buffers、IQ 路径与 PLE 改动的主要实现批次 | 本文“已实现的后续工作”及其 A/B/拒绝实验 |

注意：阶段记录最初以 `72df13d` binary/worktree 为基线；优化结果不应反向标成 `72df13d` 的成绩。若结果摘要未提供运行 binary SHA 或精确 commit，本页不会补推一个确定的 benchmark commit。
