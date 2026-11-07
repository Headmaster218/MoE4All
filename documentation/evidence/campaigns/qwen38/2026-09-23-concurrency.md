# Qwen3.8 并发服务：吞吐、停顿和后续步骤

## 范围

本报告前半部分测量的是完成单路 Decode 和 MTP 优化后的发行版 binary `ea1f17e`。后续优化调整
了并发调度、按 lane 划分的 QSA 降低策略以及 PLE gather fan-out；最终发行版 binary 的结果在下文
单独报告。

- Binary SHA-256：`8eb6a509233d44a1e1045e2216f1e1363b2bdc674700a006b09e50c9088f78fb`。
- Ryzen 5 5600X、RX 7900 XTX，可见系统 RAM 约 64 GiB。
- Model: `D:\AILMStudioModels\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
- MTP head：`G:\Qwen3.8-Flash-Next-UD-Q2_K_XL\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`。
- 固定 VRAM/RAM/ubatch/parallel-ubatch：**24 GiB / 48 GiB / 3072 / 256**。
- 请求参数为 `--parallel 2`、temperature 0、seed 1、`--no-think`。
- 每个 slot 的 context capacity 为 32,768，并另设 163,840 的对照项。实际 prompt 长度为 507 至
  29,688 个 token；capacity 不等于实际占用的上下文长度。
- 使用重复的合成背景记录，随后提出两种不同的技术写作任务。每个请求生成 1,024 或 1,536 个
  token，cached prompt token 数为 0。B 请求在 A 首次输出文本 6 秒后到达。
- 每次只运行一个 server process，不并行执行 benchmark、模型任务或编译。

运行脚本为 `scripts/bench-concurrency.cjs`，分析脚本为
`scripts/analyze-concurrency.cjs`。本地被忽略的原始日志、prompt、输出、计时、hash 和摘要位于
`target/concurrency-20260923/`。

## 已实现的优化工作

最终发行版 SHA-256：
`133eedf7a5ed69c0bb4904178e0d9a49997b13b4ed705a9261f76f9e14ff5d00`.
模型、prompt、预算和 server 参数均与基线相同。

- 独立的 PLE batch 现在会按互不重叠的 gather group 分配给全部 4 个 gather worker，不再把双 slot
  batch 的 worker 数限制为 2。
- Dense-QSA 和 sparse-QSA Decode lane 保持在同一个 model cohort 中。固定 projection、recurrent
  layer 和 MoE 均以 `m=2` 执行；每条 QSA lane 保留自己的 cache、选中 block 数和 dense-prefix
  identity selection。
- 长 Prefill 的独占阶段保持不变。Decode 仍需等待该 Prefill 完成，以保留现有 ring-buffer 和
  expert 重建约定。

关闭 profiling、每个请求输出 1,024 token；下表统计两个请求同时 Decode 期间的聚合吞吐：

| Prompt token 数 A / B | 基线 | 优化后 | 变化 | 优化后三段速度 |
| --- | ---: | ---: | ---: | --- |
| 507 / 508 | 50.19 | 54.16 | +7.9% | 55.02 / 54.43 / 53.03 |
| 8,187 / 8,188 | 50.37 | 52.73 | +4.7% | 53.38 / 51.98 / 52.83 |
| 507 / 28,668 | 34.50 | 54.34 | +57.5% | 55.74 / 52.82 / 54.46 |

短、中文本的输出 hash 与原始非融合路径按 lane 对比一致。混合 QSA 场景现在可持续达到每 lane
约 27.2 tok/s，不再交替运行单行、每次 16 step 的 cohort。但当后到的长 Prefill 独占 scheduler
时，先到请求仍会按设计暂停输出文本 32.6 秒。

达到 63 tok/s 需要每个双 token step 不超过 31.75 ms。最终短上下文测试为 36.89 ms，仍需减少
约 5.14 ms/step（14%）。更新后的 pager profiling 将暴露的 PLE wait 从每 step 1.57 ms 降到
0.22～0.38 ms，但 profiling 下 backend 仍耗时约 38～39 ms，每 step 约有 107 次 queue
submission、12 ms command recording 和 2.2 ms CPU submit time。剩余差距来自 pager/orchestration，
而不是 PLE。

同一 binary 下被否决的实验可作为边界参考：将 shared expert 融入 paged IQ slot 的效果持平或更差；
关闭 hit-first 后降至 49.3 tok/s；独立行 int8 mrow 没有提高吞吐；`MR=2` 的 Q4_K/Q6_K 构建
出现回退；expert grid 使用 `NR=4` 仅提升约 0～1%，`NR=2` 则回退。这些实验均未保留在生产源码中。

普通 token 数来自 server 的 `GenerationProgress` 和最终 usage，而不是 SSE event 数。重叠区间的
三等分按两个请求同时 Decode 的墙钟时间划分；插值分辨率约为 1 秒。请求平均速度包含 Decode
开始后的暂停时间。SSE timestamp 只用于计算首个文本到达时间和文本间停顿。MTP segment 数来自
单独 pager-profile 测试中的 committed-token cycle 日志。

## 普通并发解码

下表所有数值均为关闭 profiling 时**两个请求合计的 tok/s**。它们统计重叠窗口，不是单个请求
整个生命周期的平均速度，也不包含初始 Prefill。

| 实际 prompt token 数 A / B | 到达顺序 | 前三分之一 | 中间三分之一 | 后三分之一 | 整段重叠区间 |
| --- | --- | ---: | ---: | ---: | ---: |
| 507 / 508 | Together | 51.8 | 51.1 | 47.6 | 50.2 |
| 8,187 / 8,188 | Together | 51.7 | 50.1 | 49.3 | 50.4 |
| 28,667 / 28,668 | Together | 52.5 | 49.9 | 51.8 | 51.4 |
| 3,577 / 29,688 | A, then B | 50.5 | 48.2 | 46.3 | 48.3 |
| 29,687 / 3,578 | A, then B | 54.0 | 53.6 | 51.6 | 53.1 |
| 507 / 28,668 | A, then B | 35.3 | 33.7 | 34.5 | 34.5 |

QSA 模式匹配时，稳定态速度约为每请求 24～27 tok/s。最后一行的 QSA 模式不同，因此失去了双行
batch。这并不意味着“上下文越长 Decode 就一定越慢”：28k/28k 的配对仍达到 51.4 tok/s。由于
生成后续内容和 cache 状态不同，也不能把反向到达顺序下更高的速度单独归因于到达顺序。

163,840 capacity 的对照测试重复了实际长度为 3,577 → 29,688 token 的请求：聚合速度 48.3，
三段分别为 49.1 / 49.0 / 46.9。在本次 Decode 区间内，更大的 capacity 没有复现持续
17～18 tok/s 的聚合速度；但它确实带来了更长的 Prefill 暂停，详见下文。

### 到达停顿远比稳态解码严重

| 请求到达方式 | A 的最长文本停顿 | A 整个请求的 Decode 平均速度 | B 首文本延迟 |
| --- | ---: | ---: | ---: |
| 3.5k -> 29k, 32k capacity | 33.42 s | 16.95 tok/s | 33.79 s |
| 3.5k -> 29k, 160k capacity | 58.72 s | 13.23 tok/s | 59.04 s |
| 29k -> 3.5k, 32k capacity | 6.28 s | 25.77 tok/s | 6.50 s |
| 0.5k -> 28k, 32k capacity | 33.00 s | 11.93 tok/s | 33.97 s |

在 3.5k → 29k 测试中，B 完成后 A 单独恢复到约 33.8～34.1 tok/s。因此较低的请求平均速度不
代表它恢复 Decode 后仍只有 13～17 tok/s。不过，这并不能否定用户最初看到的**实时** 17.5：
本轮测试矩阵仍未复现两个 slot 都已进入 Decode 后持续只有 17.5 tok/s 的情况。

`parallel.rs:1726` 将长 Prefill 作为一个独占事务执行。Scheduler 在返回 Decode 前先调用该事务
（`parallel.rs:2105`），且必须等每条长 Prefill lane 都到达 Decode frontier 才算完成。因此，
现有 Prefill 内部分块之间不会为正在 Decode 的请求提供服务。这是刻意设计的，用来避免在 Prefill
ring chunk 之间重建 Decode LRU；改善公平性时必须保留这项 cache 生命周期保护。

160k 对照测试在全新进程中启动长 sparse Prefill，而 32k 混合场景是在其他测试预热后运行；系统
内存压力也有变化。不能把 33 → 59 秒的全部差异都归因于 context capacity。两个进程的普通
expert arena 都约为 16.67 GiB，但保留的 dynamic/runtime corridor 不同（9.47 vs 14.41 GiB）。

### QSA 拆分损失批处理，而非仅损失公平性

`parallel.rs:1613` classifies each lane by
`indexer_top_k + max(compress_ratios) - 1` (about 2,051 tokens here).
`parallel.rs:2119` partitions the entire forward into dense/sparse groups and
若两组都存在，则设置 16-step quantum；拆分范围不只包括 attention。

因此，0.5k/28k 请求对会轮流执行两个单行 model forward，而不是将两个请求的固定 projection
和 expert 计算合并为 batch。Prefill 完成后，关闭 profiling 测得的单流文本间隔仍约为 0.58 秒。
The independent pager run confirms only `lanes=1` cohorts, predominantly 16
steps, versus a single `lanes=2` cohort for 8k/8k.

较短的 512-output pager split 测试包含冷启动转换，并出现 1.74 秒文本间隔；重叠区间三段速度为
23.0 / 33.4 / 32.7 tok/s，整段重叠吞吐为 29.7。因此不能用这轮 profiler 测试替代更长的、关闭
profiling 的 34.5 基线。预热后的 16-step cohort 每次仍需约 17～23 ms 一次性 setup（约
1.1～1.4 ms/token），但大部分损失来自无法共享双行计算，而不只是 setup 开销。

## 双行时间去向

独立的异步 pager profiling 测试使用 8,187/8,188-token prompt，每个请求输出 512 个 token，
聚合速度为 49.8 tok/s，接近未插桩测试的 50.4。测得的 runner cohort 覆盖 512 个 step、1,024
个输出行，总耗时 20,614.4 ms。

可相加的阶段平均耗时，单位为**每个输出两个 token 的 step**所用毫秒：

| 阶段 | ms |
| --- | ---: |
| Layer 0 execution, overlapping PLE | 1.422 |
| Exposed PLE wait | 1.567 |
| Main execution, layers after layer 0 | 36.296 |
| Front/setup/upload/tail/teardown/unaccounted | 0.978 |
| Total | 40.263 |

其他计数存在嵌套和重叠，不能作为可相加的收益：

| 每个双 token step 的计数 | 数值 |
| --- | ---: |
| Backend execution including layer 0 | 37.715 ms |
| Timestamped main-queue intervals | 22.936 ms |
| Timestamped DMA intervals | 3.765 ms |
| Main/DMA overlap | 2.188 ms |
| GPU interval union | 24.514 ms |
| Backend minus timestamped GPU union | 13.202 ms |
| Recorder lifetime, including host copies/preparation | 11.238 ms |
| Queue submissions / CPU submission time | 106.9 / 2.084 ms |
| Synchronization scope | 22.669 ms |
| Expert-role lookup hit rate | 92.0% |
| Host/ReBAR push / CPU push time | 52.45 MiB / 3.348 ms |
| Direct-DMA expert bytes | 98.74 MiB |
| Host-store misses / expert mmap reads | 0 / 0 |
| Total PLE worker time / exposed wait | 3.081 / 1.565 ms |

47.17 GiB routed expert 全部位于 host store，其中 31.02 GiB 导入用于 direct DMA。GPU 中未命中
的 expert 仍需 promotion，但本轮测试没有等待 expert SSD/mmap fallback 读取。带 timestamp 的
DMA 耗时约有 58% 与 main-queue work overlap。剩余 DMA 时间约为 1.58 ms/step，并不等于所有
expert 相关停顿的总预算。

CPU router readback、residency/LUT 准备和频繁的 submission boundary 仍使执行碎片化。22.7 ms
同步范围内包含真实 GPU work；13.2 ms 未被 timestamp 覆盖的 backend 区间，不能证明是硬件空闲，
也不能认为可以全部消除。本调查测量的是 orchestration 和传输，不能把先前单行 kernel 的占比
直接套用为双行 kernel 占比。

### 一项小而具体的 PLE 后续工作

`seam/ple.rs:427` 将 independent-batch task 数限制为 `spans.len()`。因此两个独立请求只会得到
两个 task chunk，尽管优化后的单行路径使用四线程 persistent pool。应按实际 gather group 数和
有界 worker 数量划分任务，而不是按请求数划分；同时保证各任务写入互不重叠的目标区域。

当前暴露的 wait 是 40.26 ms 中的 1.57 ms。即使全部消除，本 profile 下聚合吞吐也只能从
49.7 提升到 51.7，**理论上限约 4%**。实际实验应以更小的收益为目标，并检查 CPU 争用和
page fault，不要假设它能解决主要瓶颈。

### 内存压力注意事项

普通模式测试期间的系统快照显示，可用物理 RAM 约 4.5 GiB，剩余 system commit 仅 1.1 GiB，
同时存在 page-in 活动。进程 private bytes 约 75.7 GB，working set 为 51.2 GB。专家/RAM 配置
为 48 GiB，并不代表进程的所有分配、driver commitment 和 PLE mapping 都能塞进这 48 GiB。

这些计数无法指出是哪项 allocation 导致 page-in。Pager 仍报告 expert host-store/mmap fallback
读取为零。没有 allocation/page-fault 归因前，不要把所有 OS paging 都称为“expert SSD 等待”，
也不要把冷 Prefill 差异归咎于此。应保留 commit headroom 作为运行保护，并分别报告 staging、
fixed 和 runtime 的主机内存 commitment。

## MTP：并发到达当前被串行化

启用 Vulkan MTP 和 draft head 时，`infr-cli/src/main.rs:4542` 会选择串行的 ChatModel adapter。
启动时会明确警告 `--parallel 2` 被忽略。实际计数为 `active=1 queued=1 kv_slots=1/1`，而不是
两个 Decode slot。这是当前架构限制，不是可以通过调 scheduler 参数解决的问题。

默认 `spec.k=4` 表示 target VERIFY 有四行：一个 pending target token，加最多三个 speculative
token。并不是四个 draft token 再加第五行。

| Prompt / 到达方式 | Profiling | A Decode | B Decode | B 首文本延迟 | 总输出 / 墙钟时间 |
| --- | --- | ---: | ---: | ---: | ---: |
| 507 / 508, together | Off | 36.0 | 35.8 | 35.75 s | 31.8 tok/s |
| 507 / 508, together | Pager | 35.1 | 35.1 | 36.86 s | 31.0 tok/s |
| 1,528 / 1,529, staggered | Off | 35.9 | 38.5 | 42.02 s | 31.8 tok/s |
| 1,528 / 1,529, staggered | Pager | 35.8 | 36.9 | 41.73 s | 31.5 tok/s |

两个 Decode 列分别表示先后执行的单个请求，不能相加。对相同的短 HTTP 到达 workload，普通服务
生成 2,048 个 token 用时 44.35 秒（端到端 46.2 tok/s），MTP 用时 64.37 秒（31.8 tok/s）。
这些端到端数据包含 Prefill 和排队；之前的普通模式 50.2 数值不包含初始 Prefill。

短 prompt 的 pager 测试中，A 的 Decode 三段速度为 35.2 / 33.6 / 35.1，B 为 34.3 / 33.7 /
36.3 tok/s。平均 cycle 数据如下：

| 请求 | Draft 接受率 | 每 cycle 输出数 | Draft | Verify | Catch-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 52.0% | 2.55 | 4.56 ms | 68.34 ms | 0.73 ms |
| B | 54.4% | 2.63 | 4.59 ms | 70.01 ms | 0.71 ms |

1.5k pager 测试跨过 QSA boundary，最终上下文分别达到 3,064 / 2,553 token。三段速度为 A：
**30.2 / 37.7 / 38.7**，B：**35.1 / 35.9 / 38.5** tok/s。VERIFY 平均为 67.64 / 71.76 ms，
接受率为 53.7% / 61.0%，每 cycle 输出 2.61 / 2.82 token。两个输出 hash 均与未插桩的 1.5k
对照一致。因此，3.5k 初始化失败并不代表现有上下文跨过 QSA threshold 后就完全无法 Decode。

主要耗时仍在 VERIFY。这些较长的技术类后续内容没有达到此前 24-token history prompt 的
40.5 tok/s。这里 ordinary 与 MTP 的输出 hash 不同，因此这是 workload 吞吐比较，不是 token
完全一致的正确性 A/B。接受后续数值/kernel 改动前，仍需保留 identical-prefix replay 和
logit-margin 检查。

MTP 目前尚未输出普通路径的 GenerationProgress 更新。Server 的实时计数会退回到文本 delta，并在
结束时与最终 usage 对账；因此实时显示数字并不是该路径可靠的精确 token 计数器。使用界面底部
数据调节 MTP policy 前，需先恢复 progress callback 和 context limit。

### 长提示初始化在解码前失败

| 实测 prompt 长度 | 结果 |
| --- | --- |
| 507/508 | Successful |
| 1,528/1,529 | Successful, including continuation beyond the QSA threshold |
| 3,577/3,578 | Contiguous activation request 3,559,915,520 bytes exceeds largest arena gap 2,147,379,200 bytes |
| 8,187/8,188 | 7.57 GiB staging allocation fails with Vulkan `ERROR_UNKNOWN` |
| 28,667/28,668 | 26.52 GiB staging allocation fails |
| 29,687/29,688 | 27.46 GiB staging allocation fails |

失败的测试没有 Decode 吞吐数据。两种请求到达顺序都尝试过，但请求在首段文本输出前就失败，因而
无法实际触发预定的 stagger delay。Server 在请求报错后仍正常运行。

`mtp/qwen4.rs:1296` 会通过 `run_verify_with_finish` 重置并处理整个 prompt，物化所有行的
hidden/ID，并让 head 追赶完整 prompt。它没有使用普通路径的有界分块 Prefill。
`seam/runner.rs:8116` 在固定 VERIFY buffer 不适用时，还会分配 `m * vocab * 4` 的 staging，
即使 GPU argmax ID 已足够。计算图也会保留很大的 vocabulary 形状中间结果。

3.5k 失败时，各 shard 合计仍有 5.99 GB 空间，但最大单个 gap 只有约 2 GiB。这是 tensor/chunk
shape 限制，并不能证明固定 MTP 权重是在 unified pool 之后才分配的。扩大 arena 或移动 head
reservation 都无法消除 prompt 长度与 vocabulary 维度相乘带来的扩张。应将 target/head prime
分块，正确传递错位 hidden boundary，并且只计算启动生成所需的 frontier vocabulary 结果。

## 实用优化顺序

| 优先级 | 改动 | 预期影响与验证方式 |
| --- | --- | --- |
| 1 | GPU routed-hit mask 与更细粒度的 pager 同步 | 移除 CPU readback/mask 决策；在保留版本化 LUT 和 slot 生命周期的前提下，减少每 step 约 107 次 submission。这是目前唯一测得足以弥补剩余 5.1 ms/step 的方向。 |
| 2 | 在 paged MoE 周围复用 command segment | 在动态 router boundary 之间缓存或 replay shape 稳定的 dense/recurrent command segment；目标是减少实测约 12 ms/step 的 command recording。 |
| 3 | MTP 分块 prime 与精确进度 | 在预算不变的情况下支持 3.5k/8k/29k prompt；避免按 prompt 大小分配 vocabulary staging，采用有界 hidden handoff，并保持 prefix state 与普通路径等价。 |
| 4 | 将逐 slot MTP 接入 ParallelSeam | 实现真正并发的 speculative batch，并独立处理接受/回滚；保留普通 Decode 回退路径。不要在两个 server process 中复制整套模型。 |

本轮已完成双请求 PLE fan-out 和 mixed-QSA shared cohort。长 Prefill 交错执行刻意不作为优化目标：
当前独占阶段用于保护 ring buffer 和 expert 重建，继续保持不变。

Mixed-QSA 改动必须验证 position、compressed-history 边界和 state 隔离。单纯缩短 16-step
quantum 只能让文本输出更平滑，无法恢复 shared expert 的工作复用，还可能增加 cohort 重建开销。

真正支持 MTP 并发需要共享不可变的 target/head 权重，并为每个 slot 独立维护 head KV、pending
token/hidden、accepted-prefix recurrent trace 和取消状态。双 slot、四行 VERIFY 实际包含两组因果
关系各自独立的四行，**不是八个互不相关的行**。应遵循现有分配约定，在查询实时空闲 VRAM 并
分配 unified arena 之前，为各 slot 预留固定资源。应明确限制并发数和宽度，不能静默超额承诺资源。

按当前短测接受情况（每个 slot 每 cycle 约输出 2.55 个 token），未来双 slot batched MTP 每个
cycle 约输出 5.1 个 token。要达到普通模式 50 tok/s 的聚合吞吐，完整 cycle 需控制在约 102 ms；
达到 60 tok/s（提升 20%）则需约 85 ms。当前串行处理两个请求约耗时 147 ms。这只是条件式预算
估算，不是实测的八行结果；batching 可能改变接受率、传输压力和实际成本。宽度及回退策略应按
单位墙钟时间实际输出的 token 数决定，同时比较 MTP head 仍常驻与未加载 head 两种普通 Decode
基线。

普通双行 Decode 的优化后短测为 36.89 ms/step。要达到 63 tok/s，需降至 31.75 ms/step，因此
下一轮还需减少约 5.14 ms（14%）。混合 QSA 测试只有 54.34 tok/s，因此首文本延迟和长 Prefill
独占期间的暂停，仍是与重叠窗口吞吐分开的用户体验问题。

## 复现和限制

```powershell
node scripts/bench-concurrency.cjs ordinary-matrix-off ordinary off
node scripts/bench-concurrency.cjs ordinary-final ordinary off short,medium,split
node scripts/bench-concurrency.cjs mtp-matrix-pager mtp pager short,medium,mixed,reverse,long
node scripts/bench-concurrency.cjs mtp-control-off mtp off short,lowmid
node scripts/bench-concurrency.cjs mtp-lowmid-pager mtp pager lowmid
$env:BENCH_CTX = '163840'
node scripts/bench-concurrency.cjs ordinary-cap160k-off ordinary off mixed
Remove-Item Env:BENCH_CTX
$env:BENCH_TOKENS = '512'
node scripts/bench-concurrency.cjs ordinary-phases-pager ordinary pager medium,split
Remove-Item Env:BENCH_TOKENS
node scripts/analyze-concurrency.cjs ordinary-matrix-off ordinary-final ordinary-cap160k-off ordinary-phases-pager mtp-matrix-pager mtp-control-off mtp-lowmid-pager
```

每个矩阵场景只有一次完整的未插桩运行；注明的场景另有 profiling/control 测试，但这些数据不构成
统计置信区间。生成内容以及冷/热状态均可能不同。实测最大上下文约 31k，尚未跨过下一个 32k
分段边界；本轮未测试 160k 实际占用上下文、4 个以上请求、vision/embedding 共存或取消请求。

本轮改动涉及并发 scheduler、PLE gather 调度和 Vulkan QSA lowering，并新增两个 benchmark 脚本及
本报告。所有测试 server 均已在测试后停止，没有遗留运行中的服务。发行版 binary 已重新构建，
mixed-QSA 纯逻辑测试和真实 Vulkan 测试均通过。

合入优化前，应在关闭 profiling 的条件下交替运行旧版/新版至少三次，保持 prompt 和预算完全相同，
比较重叠区间中后段吞吐、P95/P99 文本间隔、TTFT、内存 commitment 以及输出/state 正确性。覆盖
跨越 QSA boundary、MTP 接受数为 0/部分/全部、EOS、延迟到达、cohort 缩减和对端取消等情况。
GPU validation 应在计时运行之外执行。

## 实现提交溯源

本记录包含普通并发的优化前后测量，也包含当时 MTP 尚未真正并发的限制；后续提交已经改变了该能力边界。代码演进按提交区分如下：

| Commit | 提交说明 | 对本报告结论的影响 |
|---|---|---|
| `d42655d9` (`perf(qwen38): batch mixed QSA decode lanes`) | 混合 QSA decode lane batching | 对应普通并发共享 cohort/混合 QSA 路径；性能需按本报告对应 binary 与 workload 解读 |
| `30dbaa8e` (`fix(prefill): avoid impossible parallel scratch reservation`) | 避免不可能的并行 Prefill scratch reservation | 与 Prefill placement/scratch 有关；不能仅凭提交说明认定为 9 月 24 日 414 tok/s 回退的根因 |
| `cbdc273c` (`feat(mtp): verify Qwen3.8 drafts across two slots`) | 引入双 slot MTP VERIFY | 后续能力实现，不能改写本文当时“到达被串行化”的历史实测 |
| `ca40f961` (`perf(mtp): use batched decode for concurrent slots`) | 并发 slot 使用 batched MTP decode | 将双 slot MTP 从能力接入推进到批量 Decode；本文的原始串行数据仍是历史基线 |
| `7a1830fa` (`fix(vulkan): protect frozen expert LUT slots`) | 保护 frozen expert LUT slots | 并发/VERIFY 正确性修复，不是吞吐优化数字 |

以上提交信息由 checkpoint Git 历史核对；不同提交的结果不能拼成一个未经测量的“累计提速”。
