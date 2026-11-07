# Qwen3.8 普通单序列解码：测得的瓶颈

> 实现后更新：下文保留基线分析以便追溯。优化结果和当前状态记录在
> [已实现的优化轮次](#已实现的优化工作)。

## 范围和复现

本报告基于 `perf/qwen38-24g48g-ub3000` 分支上的 `72df13d`，接续[MTP 瓶颈调查](2026-09-23-mtp-bottlenecks.md)。
测试时关闭 MTP，且没有加载 MTP head。测量的是普通 Decode，不是 MTP 固定资源仍常驻时的普通
回退路径，也不是并发服务。

- CPU：Ryzen 5 5600X；GPU：Radeon RX 7900 XTX。应按枚举出的 GPU 名称选择，因为 Vulkan 设备
  index 可能随进程变化。
- Model: `D:\AILMStudioModels\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
- 固定 VRAM / RAM / ubatch / parallel ubatch：**24 GiB / 48 GiB / 3072 / 256**。
- Temperature 为 0，seed 为 1，不启用 thinking；每轮使用全新进程和请求。
- 实际 prompt 长度为 24、1,347 和 9,368 个 token。前两项 context capacity 为 4,096，最后一项
  为 16,384。Capacity 不等于实际上下文长度。
- 固定资源真实分配后，普通模式的 expert arena 约为 16.92 GiB。47.17 GiB routed expert
  全部放入 host store，其中 31.02 GiB 被导入用于 direct DMA。Decode 期间没有发生 expert
  host-tier 或 mmap fallback 读取。

在 `crates/infr-llama/src/seam/runner.rs` 中加入仅用于诊断的 Decode phase marker、阶段计数器
和逐层耗时差值，由 Qwen3.8 条件及 `INFR_PAGER_PROFILE` 控制。它们排除 Prefill，也不会启用有
阻塞影响的 `INFR_PROF_STAGES` 路径。没有改动推理算法、kernel 或调度策略；此前未提交的 MTP
诊断和报告均予以保留。

本地被忽略的产物位于 `target/`，包括 `mtp-profile-run.cjs`、`single-decode-analyze.cjs`、
`single-decode-compare.cjs`，以及每轮测试对应的 `.meta.json`、`.err.log`、`.out.log` 和
`.analysis.json` 文件。运行脚本会清除继承来的 `INFR_*` 参数，并记录完整 prompt、环境和命令行
参数。以下命令从 workspace 根目录运行：

```powershell
node target/mtp-profile-run.cjs single-decode-history-off-1 ordinary off history 256
node target/mtp-profile-run.cjs single-decode-history-pager ordinary pager history 256
node target/mtp-profile-run.cjs single-decode-context-pager ordinary pager context 256
node target/mtp-profile-run.cjs single-decode-deep-pager ordinary pager deep 256 target/release/infr.exe 16384
node target/mtp-profile-run.cjs single-decode-history-ops ordinary ops history 64
node target/mtp-profile-run.cjs single-decode-deep-ops ordinary ops deep 64 target/release/infr.exe 16384
node target/single-decode-analyze.cjs single-decode-history-pager
```

`history` 与 MTP 报告使用相同的北京历史、文化和地理 prompt。`context` 和 `deep` 会先重复结构化
的 engine 记录，再提出基准测试设计问题。两者生成的后续内容不同，因此运行结果差异不能单独归因
于上下文长度。

## 吞吐基线

| `single-decode-` 后的产物标签 | Prompt token 数 | 输出 token 数 | Profiling | Decode tok/s |
| --- | ---: | ---: | --- | ---: |
| `history-off-1` | 24 | 256 | off | 35.0 |
| `history-off-2` | 24 | 256 | off | 32.2 |
| `history-512-off` | 24 | 512 | off | 32.0 |
| `history-pager` | 24 | 256 | async pager | 33.3 |
| `context-pager` | 1,347 | 256 | async pager | 31.3 |
| `deep-pager` | 9,368 | 256 | async pager | 30.0 |

两次生成 256 token 且未开启 profiling 的 history 输出，与 pager 输出完全相同，SHA-256 为
`99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`。第一次 off 测试早于新增的
单路 Decode 诊断 hook；之后的测试均使用重新构建的诊断版 binary。这不是优化 A/B。结果波动明显，
应报告为 **32.2～35.0 tok/s**，而非稳定的 35 tok/s 基线。512-token 样本也说明，预热后不能
假定生成得越久速度就越快；expert 选择和 cache churn 会随生成 token 而变化。

逐 op 诊断测试只有 3.3/3.2 tok/s，因为它们会在 submission 边界同步并输出报告，破坏正常
overlap；不能用这些速度或 kernel 的绝对毫秒数预测生产性能。新增逐 op 诊断期间没有并行执行
编译。

## 普通 token 墙钟时间去向

仅统计 Decode 阶段，单位为每个输出 token 的平均毫秒数：

| 阶段 | 24-token prompt | 1,347-token prompt | 9,368-token prompt |
| --- | ---: | ---: | ---: |
| Decode 总墙钟时间 | 30.062 | 31.921 | 33.380 |
| Target execution：主机 setup/record/submit 与 GPU | 28.363 | 29.958 | 31.621 |
| execution 之外暴露的 PLE wait 和 upload | 1.270 | 1.505 | 1.284 |
| Graph 构建、compile wrapper 和 bindings（不含 PLE） | 0.299 | 0.316 | 0.329 |
| Greedy token ID readback | 0.0064 | 0.0065 | 0.0067 |
| 其他 runner 工作（包括诊断日志） | 0.123 | 0.135 | 0.139 |

除总时间外，表中各项可以相加。Backend setup/recording 已计入 execution。Layer 0 在 execution
中每 token 约需 1.14～1.18 ms，并与异步 PLE job overlap。PLE worker 总耗时为每 token 2.30～
2.68 ms；只有暴露出来的 wait 计入表中可相加的部分。

普通 greedy Decode 只下载一个 4-byte ID，不会请求 MTP hidden-output buffer。因此 MTP 报告中
约 5 ms/cycle 的 hidden-readback 问题**不适用于此处**。LM-head 的 GPU 计算是另一项较小开销。

冷启动转换的影响不可忽略：deep-context 下第一次前向 execution 为 289 ms，而短 prompt 约为
44 ms。deep 测试最初 64 次前向平均为 38.35 ms；之后每个 64-token 窗口分别为 30.90、31.97 和
31.72 ms。Pipeline
在将这次尖峰归因于某个原因之前，还需分别追踪不同 variants、Prefill 到 Decode 的资源转换及
expert-cache 状态。首 token latency 和稳定态 Decode 应分开报告，不要悄悄排除冷启动开销。

## 设备热点，而不只是专家传输

单独运行 64-token op 诊断，并只统计 Decode marker 之间的数据。下列比例基于插桩后的 GPU
operator interval，**不是 token 墙钟时间占比**：

| Kernel 分类 | 短 prompt | 9,368-token prompt |
| --- | ---: | ---: |
| Routed experts | 41.5% | 40.4% |
| Dense projection（含未融合的 shared-expert projection） | 38.4% | 35.9% |
| Attention 与 QSA | 1.2% | 5.1% |
| Recurrent/DeltaNet/conv | 2.9% | 2.7% |
| Vocabulary projection | 3.4% | 3.0% |
| Routing、normalization 与其他 operator | 12.6% | 12.9% |

短 prompt 下，各 expert kernel 分类占比为 IQ2_S 28.0%、IQ4_NL 9.5%、IQ3_S 4.0%。占比较高的
`gemv_streamed:m1` shape 包括 `2560x10240`（6.5%）、`6144x2560`（5.5%）、`320x10240`
（4.8%）、`2560x6144`（4.5%）和 `10240x320`（3.8%）。`moe_topk_sg` 另占 3.8%。固定
projection 值得单独开优化专项；与 MTP VERIFY 相比，普通 Decode 受 expert 计算主导的程度低得多。

deep 测试确实执行了 `qsa_indexer_topk`，因此并非只是 KV capacity 很大但 QSA 分支未启用。在
本次测得的深度下，Attention/QSA 仍不是首要优化目标；但这不能说明它在 32k～163k 时的成本。

### 新确认的共享专家覆盖缺口

`infr-vulkan/src/adapter.rs::paged_moe_shared_at` 目前只支持 Q5_K、Q6_K 或 IQ4_XS 格式的
routed weight，且 gate/up/down bank 必须全部受支持。它还要求 graph shape 兼容，shared weight
为 Q8_0。`infr-vulkan/src/gemm.rs::native_idm_paged_shared_build_spv` 中的 SPIR-V builder 也有
相同的格式限制。本模型实际执行的 routed bank 是 IQ2_S/IQ4_NL/IQ3_S，诊断中没有出现融合
shared-slot 的 expert kernel。

因此，现有 multi-row shared-slot 实现对这个量化模型**并未启用**，普通 Decode 和 MTP 都如此。
支持这三种格式有机会在一次 expert dispatch 中合并更多工作，并省去独立的 shared-path launch；
但这只是尚未测量的机会，不是确定收益。需要增加实际 shader variants，并验证混合 bank 格式、
Q8 shared packing、mask 和行数限制；仅放宽格式识别条件是不正确的。

### 具体的 IQ 内核浪费

`native_gemv_id_multi.comp` 及其 `_sg` 变体会先执行 `GRID_INIT`，再检查 workgroup 统一的边界和
mask。在 hit-first/miss-second dispatch 中，无效 group 可能先初始化 IQ codebook，随后立即返回。
应把符合条件的统一检查提前到初始化之前，同时确保 barrier 参与规则正确。普通 Decode 每个
token 的 48 层中约有 18.4～24.0 个 hit-first window；不能直接用拆分更细的 MTP 路径推算其收益。

之后可针对实际使用的 IQ2_S/IQ4_NL/IQ3_S shape，评估常驻只读 codebook buffer 和 output-row
tiling。普通 Decode 的 `m=1` 无法利用 MTP 方案中的跨 token 权重复用。在获得实测前，保留现有
SG 排除规则：`native_id_sg_choice` 显示相关小输出 shape 上 IQ2_S 表现不佳。还要注意
`infr-vulkan/build.rs::gen_grids`：之前直接使用动态索引的常量数组曾导致 RADV 严重寄存器溢出，
不能用这种数组替代 LDS staging 走捷径。

## 主机调度、PLE 和传输

这些计数会与 execution 及彼此重叠，**不能相加**：

| 每 token 的 Decode 计数 | 短 prompt | 1,347 tokens | 9,368 tokens |
| --- | ---: | ---: | ---: |
| Paging synchronization calls | 48 | 48 | 48 |
| Paging synchronization scope, ms | 18.98 | 19.37 | 19.94 |
| Queue submissions | 81.64 | 90.81 | 90.23 |
| CPU time inside queue submission, ms | 1.51 | 1.78 | 1.78 |
| Recorder lifetime, ms | 6.73 | 7.94 | 9.13 |
| Backend setup, ms | 0.96 | 0.69 | 0.59 |
| Timestamped main-queue intervals, ms | 17.30 | 17.48 | 17.77 |
| Host/ReBAR expert push, MiB | 18.50 | 32.47 | 31.73 |
| CPU time inside host/ReBAR push, ms | 1.42 | 2.11 | 2.10 |
| Direct-DMA expert payload, MiB | 39.94 | 59.78 | 57.46 |
| Expert-role lookup hit rate | 93.79% | 90.26% | 90.61% |

Hit rate 统计的是 gate/up/down role lookup，不代表独立 expert 命中率，也不代表模型权重常驻比例。
Recorder lifetime 包含主机准备和复制，不只有 Vulkan command encoding。Paging wait 也包含此前
提交的 GEMV/attention/router 工作。这些测试没有 DMA 的 GPU timestamp。因此，无论 19 ms 的
paging 时间，还是约 30 ms 墙钟时间与 17 ms main-queue 覆盖时间的差值，都不能直接视为可追回的
空闲时间。GPU 利用率和功耗偏低与工作碎片化相符，但单凭这两项无法定位受限资源或确定吞吐上限。

### PLE 仍暴露超过一毫秒

`infr-llama/src/seam/ple.rs` 仅在 gather group 至少为 256，或独立/多行 batch 足够大时才启用
并行化。普通单行的 16-group 任务不满足该条件，尽管已有 persistent worker pool。可尝试为单行
gather 设置有界的 2/4-worker 并行，并缓存 hot row，目标是将暴露出来的 PLE 时间降至 0.3 ms
以下。此数值是测试目标，不是已测结果。应分别测量 worker dispatch 成本、CPU 争用和 mmap page
fault；现有计时器无法区分存储停顿与 gather/dequant 工作。Cache 和 worker buffer 必须纳入约定
的 RAM 预算。

### DMA 覆盖不完整

本次测试中，按从 0 开始编号的第 0～3 层和第 37～47 层使用 host/ReBAR push；第 4～36 层使用
direct DMA。首层 miss 尤其值得关注，因为 layer 0 同时决定 PLE overlap window。可针对未覆盖的
热点层测试按需加权的 host import 放置策略，或设置有界的 DMA staging ring。Staging ring 会增加
一次 host copy 和 submission，因此必须通过配对 benchmark 证明它优于当前 hit-first CPU-copy
overlap，而不能只看 DMA 字节数增加。31.02 GiB 的导入结果并不意味着该驱动可以直接导入全部
47.17 GiB。

传输设计遵循 [Khronos memory guide](https://docs.vulkan.org/guide/latest/memory_allocation.html)
对独立显存与 host staging 的区分。不过，该指南本身不能证明现有 ReBAR 路径能因此获益。

### 结构性依赖仍是 CPU 路由

对于未能完全常驻的 bank，`adapter.rs::sync_stream` 会 drain 之前的 GPU work、在 CPU 上读取
router ID，并准备 residency mask/LUT。Hit-first 已经让 miss 准备与常驻 expert work overlap。
即使某一层恰好全部命中，这 48 个边界仍然存在。

[Khronos synchronization sample](https://docs.vulkan.org/samples/latest/samples/performance/wait_idle/README.html)
说明 queue-wide idle wait 为什么比 fence 粒度更粗。但在这里，仅更换 wait 原语并不能消除
router 到 CPU 的依赖。真正的下一步需要 GPU residency map/hit mask、提前发出的 router readback
信号、按 submission 版本化的 LUT，以及在所有 reader 完成前 pin 住 slot。目前完整 drain 负责
保护 LUT tape 重置和 LRU 覆盖的生命周期；允许跨边界 overlap 前，必须先保留这些保证。

`runner.rs` 针对该架构明确关闭 `dyn_replay`：先执行 layer 0，等待 PLE，再执行剩余层。这是两次
backend execute，而非两次 queue submission。外层 compile wrapper 不会重复编译 shader，整个
build/bind 范围也只有约 0.3 ms/token。通用 graph caching 不是主要优化方向。未来可评估复用
command segment，但必须正确更新 position、recurrent/KV/QSA state 和 BDA/LUT metadata。直接
启用现有 full-graph replay 并不安全。

## 可行的交付计划

| 顺序 | 工作项及负责层 | 验收条件 |
| --- | --- | --- |
| 1 | Vulkan IQ mask-before-codebook initialization | 活跃 expert 结果一致；barrier 控制流有效；关闭 profiling 后重复测得端到端收益 |
| 2 | 将 shared-slot shader/recognizer 扩展到 IQ2_S/IQ4_NL/IQ3_S | 确认融合路径实际启用；比较启用/关闭时的输出和 dispatch 数；普通与 MTP 测试通过 |
| 3 | 调优固定 HC/dense GEMV shape 与相邻融合；独立测试有界单行 PLE | 特定 shape 的 microbench 有收益且 token 墙钟时间下降；PLE 暴露 wait 低于 0.3 ms，且不增加内存压力 |
| 4 | 改善未覆盖热点层的 DMA 放置/staging；缓存不变的 backend layout/scan metadata | 降低暴露的传输/CPU 成本，而不只是减少分配或增加 DMA；维持 24/48 GiB 预算 |
| 5 | GPU hit routing、细粒度同步和可复用 command segment | 减少 CPU routing 停顿，且 LUT/slot/KV/recurrent 生命周期验证通过；作为独立架构里程碑 |

第 1～3 项不改变模型 graph 或 unified-pool allocation 架构，主要调整 kernel、backend 识别逻辑
或 PLE worker 策略。第 4 项在现有预算内调整放置。第 5 项则是较大的 pager/executor 调度改动。
小行数 backend 优化也可能帮助 MTP，但 MTP 行复用和单行 PLE 仍需分别测量。

32.0～35.0 tok/s 对应的未插桩 token 耗时为 31.25～28.57 ms。达到 40 tok/s 需要降至
25 ms/token，即每 token 节省约 **3.6～6.3 ms（13～20%）**。仅消除所有暴露出来的 PLE 时间
仍不足以填平差距，还需要两个主要计算分类和主机编排共同改善。这些测量指出了可行方向，但尚
不能证明一定能达到 40 tok/s。更深上下文的优化预算，必须先通过独立、关闭 profiling 的重复
测试确定。

每项改动都应在关闭 profiling 后，以相同模型、prompt、token 数、设备和内存预算交替测试基线与
新版本，至少重复三次。保留输出 hash 以及首 token/稳定态分布；生成内容发生分歧时，使用相同
prefix replay 隔离 kernel 性能。测试范围包括 24/1.3k/9.4k 上下文，之后扩展到 32k 以上、QSA
阈值、混合量化 bank、shared-slot mask、普通单/双 slot 以及 MTP accepted-prefix state。Vulkan
validation 应在计时测试之外运行。全局 pager snapshot 适用于这类隔离测试，不适合并发场景下的
单请求归因。

## 已实现的优化工作

以下改动经配对测量后予以保留：

- 在两个 multi-slot native GEMV shader 中，将 workgroup 统一的 bounds 和 active-mask 检查移到
  `GRID_INIT` 之前。shared Q8 expert slot 也会跳过 routed IQ codebook 初始化。
- 增加 IQ2_S、IQ3_S 和 IQ4_NL 的 shared-expert slot shader 支持，并用于普通单行 Decode。对于
  多行 MTP，新 IQ 格式仍使用原有 dense shared-expert 路径，因为在本模型测量中融合 slot 更慢。
- 让现有 persistent PLE worker pool 处理普通单行的 16-group gather。由
  `kernels.ple_single_parallel` 配置，可通过 `INFR_NO_PLE_SINGLE_PAR=1` 关闭。

以下配对测量使用相同的 24-token prompt、256-token 输出、修改前保存的 binary 和输出 SHA-256：

| 构建版本 / 功能组合 | Decode tok/s |
| --- | ---: |
| Saved pre-change binary, repeats | 34.2 / 34.1 |
| New shader checks only; PLE and shared slot disabled | 34.4 |
| Shader checks plus parallel PLE; shared slot disabled | 35.9 |
| All ordinary optimizations, repeats | 37.1 / 36.1 |
| Final release binary | **37.1** |

最终输出 SHA-256 仍为
`99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`.
相较配对基线 34.1～34.2 tok/s，37.1 tok/s 提升了 8.5～8.8%。

在开启 pager profiling 的 A/B 中，暴露出来的 PLE 时间从约 1.27 降至 0.030 ms/token；PLE
worker 总时间则从约 2.30 降至 0.88 ms/token。
普通 Decode 剩余开销仍主要来自 routed expert、固定 dense projection，以及 router/pager 执行
边界。这些实现没有改变固定资源的分配顺序或 unified-pool budget 查询。

本轮实现的验证结果：release CLI 构建成功，配置测试通过，输出一致性保留，格式化和 diff 检查
通过。链接器仍输出原有的 LIBCMT 默认库警告。

### 实现提交溯源

性能 A/B 的保存 binary 与最终提交不是同一概念；上面的 34.1–34.2 到 37.1 tok/s 是对保存 binary 与优化构建的配对测量。checkpoint 中与实现对应的提交为：

| Commit | 实现关联 | 代码范围 |
|---|---|---|
| `96886a18` (`perf(qwen38): optimize decode and MTP verify`) | IQ codebook 初始化前移 mask 检查、shared-slot 识别/执行、MTP hidden/readback 等优化的主要落地提交 | `infr-vulkan` GEMV shader/build/adapter/recorder；`infr-llama` runner/PLE/MTP |
| `ca6cb938` (`perf(qwen38): fan out parallel PLE gathers`) | PLE gather worker fan-out 后续调整 | `infr-llama/src/seam/ple.rs` |

以上 commit 归属来自 checkpoint 的 Git 提交说明和变更文件；本页配对性能表仍是实测证据。除非对应 A/B manifest 明确列出 binary commit/hash，不将最终 37.1 tok/s 简写成某个 commit 的独立、可复现性能保证。
