---
kind: development-guide
status: mixed
scope: performance-optimization-methods
source_checkpoint: release-0.10.0
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
---

# 性能优化方法

方法与工具入口按 0.10.0 核对；正文中的 2026-07/08 测量和实验仅代表其历史条件。0.10.0 的新 timeline 与端到端结果另见[性能分析方法](decode-timeline.md)及[发布前矩阵](../../evidence/benchmarks/2026-10-09-release-0.10.0-matrix.md)。

本页保留历史专项的方法和当时的经验数字；运行新的优化任务时应先按目标设备与当前版本重新建立基线。

本文面向接手性能优化专项的代理（或人类），说明如何让 infr 更快。参考目标是同一 GPU 上的 llama.cpp：我们关心的每个数字都是匹配标志下的**比值**（`infr t/s ÷ llama.cpp t/s`）。专项目标是在所有受支持的模型 × 量化、预填充和解码场景达到 ≥1.0x，且绝不牺牲正确性。

## 硬件争用 — 独占设备访问（请先阅读）

比性能循环本身更重要的唯一操作规则：

> **每次只能有一个进程/代理使用 CPU 或 GPU。** 任何触及计算设备的工作——构建、测试运行、基准测试、`infr run`/`serve` 或性能剖析——都会在**整个持续时间内独占**该设备。无论你是人类、代理，还是分派子代理的编排器，都绝不可并发运行两项这类任务。设备是一把单一互斥锁：获取它、完成、释放，再开始下一项。

为何不可妥协：

- **基准测试只在隔离状态下有效。** 并发设备工作会同时扭曲每个比值的两侧：既有热影响（相邻任务加热的芯片读数会低 2-8%），也有争用影响（共享内存带宽、SM/CU 占用）。在其他设备工作旁测得的“回归”是伪象，不是发现。
- **GPU 可能卡死。** 对 UMA/集成设备，两个大模型同时运行会耗尽 GTT 并使设备卡死；在提交中途终止作业会留下不可杀死的 D 状态任务，持有 GPU 直到重启（见[集成 GPU](../../evidence/changes/backends/integrated-gpu.md)）。串行且耐心地使用设备也是稳定性要求，而不仅是测量要求。

由此导出的规则：

1. **所有设备工作都必须串行。** 基准测试、性能剖析、`infr run`/`serve`、GPU/CPU 测试套件，以及支撑它们的发行构建都一次一个。其范围大于“一次一个基准测试”：与基准测试竞争的编译，或两次测试运行，争用同样严重。
2. **性能/基准测试片段必须严格串行，绝不可同时执行两个。** A/B 变体比较的 A 与 B 也不例外。（这是将循环中的“不要同时执行两个片段”明确化。）
3. **编程/编辑代理可并行（最多 2 个）；设备工作不可。** 可以分派只读取或编辑源码的子代理；但任务一旦编译、测试、执行基准、性能剖析或运行模型，就必须是唯一执行此类工作的任务。委派的**性能代理每次只能运行一个**；委派的编程代理**最多两个**，且二者均不得与另一个代理并发执行基准测试/运行模型。
4. **绝不在后台运行设备作业。** 不要后台执行 `cargo test` / `cargo bench` / `infr run`；请在前台使用充足的 `timeout` 运行，使其不会超出自己的轮次或与下一任务重叠。（后台运行非设备工作没有问题。）
5. **绝不在 GPU 作业提交中途使用 `timeout`/SIGTERM。** 在某些设备上（尤其是集成 AMD），提交期间终止会在 `dma_fence_wait` 中留下不可杀死的 D 状态任务，持有 GPU 并从枚举中消失直到重启（[集成 GPU](../../evidence/changes/backends/integrated-gpu.md)）。为设备运行设置充足的 `timeout`，绝不要将其包装进可能更早超时的工具调用。
6. **手动基准测试之间应冷却。** 连续的 `infr bench` A/B 运行之间间隔 60 秒以上，以便 GPU 回到空闲温度；高温 GPU 会使下一次运行低 2-8%，足以反转本来胜出的变体。（`infr compare --sweep` 在内部串行，但多模型扫描仍会加热芯片；宣布回归前，请在空闲设备上单独复测被标记的行，见“归档扫描”。）
7. **先验证正确性，再做基准测试**：任何计时前都进行一致性 + 黄金值验证（也是循环的第 5 步）。正确性运行同样是设备工作，必须像其他工作一样串行。

## 循环流程

每个性能片段都遵循同一流程。不要跳过步骤，也不要同时执行两个片段。

1. **扫描**：针对 llama.cpp 测量完整模型×量化矩阵，并按差距排序（下文的 `infr compare --sweep`）。旧数据很快过时：任何已合入片段都可能改变所有比值，因此选择目标前先重新测量。
2. **选择最大收益**，而非最有趣的收益。权衡 `差距 × 该配置的使用量`。0.5x 的模型类别优先于 0.9x 的模型；慢路径上的整个量化家族优先于单一形状。
3. **先剖析，再设计**：使用 `INFR_PROF_OPS=1` 获取逐算子设备时间。对瓶颈**类别**（见下方分类）提出假设，在编写修复前用微基准测试验证。专项中有两项经验：所谓“内核数学”瓶颈实际是占用率（LDS 预算），所谓“GEMM”瓶颈实际是派发启动开销；两次都是测量胜过直觉。
4. **修复一个杠杆**：以最小改动解决已测得的瓶颈。
5. **首先验证正确性**：一致性测试套件 + 黄金值（见下文），之后才进行基准测试。
6. **串行执行基准测试**：一次一个 GPU 作业，既测试修改的配置，也测试共享所触及代码路径的配置（一个模型的门控变更可能静默重路由所有其他模型）。
7. **带着数据提交**：在提交信息中记录前后 t/s 和比值，并更新文档中的性能证据（项目记忆只在用户明确要求时修改）。

## 模型来源 — Hugging Face 缓存 + `infr pull`

模型位于 Hugging Face 缓存（`~/.cache/huggingface/hub/`），`infr` 直接读取该目录，无需创建符号链接或复制。有两种方式可将模型放到那里：

```bash
# Pull from huggingface.co/org/repo (picks the most-used quant):
infr pull unsloth/Qwen3-0.6B-GGUF

# Or place a .gguf anywhere in the cache manually:
# ~/.cache/huggingface/hub/models--unsloth--<name>/snapshots/<hash>/model.gguf
```

`infr` 和 `llama-bench` 会正常解析模型路径，因此优化专项目录中的符号链接和直接使用缓存路径都可用：

```bash
infr bench ~/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/<hash>/Qwen3-0.6B-Q4_K_M.gguf -p 512 -n 0 -r 3
```

`scripts/perf-sweep.sh` 接受任何路径；可将一组模型符号链接到单一目录，以保持重复扫描列表简短。

## 基准测试与比较

README 的“基准测试与性能剖析”节包含完整命令参考。简要版本如下：

```bash
# infr and llama-bench take the same -p/-n/-d/-r flags:
infr bench "$M" -p 512 -n 0 -r 3          # pp512 (prefill throughput)
infr bench "$M" -p 0 -n 128 -r 3          # tg128 (decode throughput)
llama-bench -m model.gguf -p 512 -n 0 -fa 1 -r 3

# The whole matrix, ranked worst-gap-first:
infr compare --sweep <models...> --sweep-depth 4096
```

这些规则来自曾经踩过的坑：

- **一次只能进行一个基准测试。** 并发 GPU 工作会扭曲两侧结果。
- **手动运行之间应冷却。** 连续运行多个 `infr bench` 命令时（例如 A/B 测试内核变体），应在两次运行间休眠 60 秒以上，使 GPU 冷却至空闲温度。上次运行导致的 GPU 高温会使下一次数据低 2-8%，足以反转胜出的变体。这适用于手动基准调用；`infr compare --sweep` 已在内部串行，但多模型带来的芯片加热是另一问题（见下方“归档扫描”）。
- **r=2 噪声较大。** r=2 出现的“回归”可能只是方差；反应前请用 r=3+ 重新测量并与历史范围比较。绝不自动回退，应先诊断。
- **固定数量语义。** `infr bench` 设置 `INFR_IGNORE_EOS`，使解码基准测试像 llama-bench 一样恰好生成 n 个 token。若某个数值好得不可能，请检查生成是否提前停止。
- **重新构建的新鲜度。** 实际编辑后若构建工具报告“Finished in 0.06s”，输出可能来自缓存；基准测试前强制执行真实构建。
- 预填充和解码是两台独立机器。预填充变更必须让 tg 保持逐位一致（验证前后 tg128）；解码变更必须让 pp 保持不变。
- **DiffusionGemma 的扫描方式不同。** `arch=diffusion-gemma` 没有上游 `llama-bench` 支持，因此其扫描行改为来自参考分支的 `llama-diffusion-cli`（见 README 的 Compare 节及 [DiffusionGemma 变化记录](../../evidence/changes/models/diffusion-gemma.md)），输出 `dg-step`/`dg-e2e`，而非 pp512/tg128/tg64@d/mtp128。只有 `dg-step`（步骤内并行吞吐量）进入按排名的“最大差距”摘要；`dg-e2e` 仅供参考，因为两个实现的熵约束采样器对相同 `-n` 运行不同的步骤数，原始端到端 tok/s 不是公平比值。

## 归档扫描

`scripts/perf-sweep.sh <models...>`（macOS 选择 `--dev metal`；旧 `INFR_METAL` 已移除）运行扫描，并将原始矩阵归档到本机忽略的 `benchmark-data/target/perf/<utc>-<sha>.txt`。请固定模型列表，使每次提交的比值可比；将提炼后的对照、条件和限制写入 `documentation/evidence/`。

**扫描行不是回归证据。** 多模型扫描会加热芯片；在扫描中途测得的行会低几个百分点（观察到：一行单独为 0.93×，扫描中读为 0.83×，数分钟后单独重测回到 0.91×）。宣布回归前，请在空闲机器上单独复测被标记的行，并比较绝对数字而不仅是比值（llama.cpp 一侧也会受相同热量影响而漂移）。

## 性能剖析

**一个调节项，覆盖所有后端。** `INFR_PROF_OPS=1` 会在 Vulkan、Metal 和 CPU 后端上启用逐算子性能剖析：

```bash
INFR_PROF_OPS=1    infr bench "$M" -p 512 -n 0 -r 1  # per-op device time, any --dev
INFR_PROF_STAGES=1 infr bench ...                    # host-side stage timing
```

`prof.*` 可通过 `--set prof.<name>=<value>` 或 `[prof]` 节设置；下表为具有历史环境别名的字段。0.10.0 新增 timeline 与 CPU miss 样本参数，不自动具有 `INFR_PROF_*` 别名，详见[配置](../../reference/configuration.md)。

| 调节项                | 作用                                               |
| ------------------- | ---------------------------------------------------------- |
| `ops`               | 每个算子的设备性能剖析（主要选项）                       |
| `op_shapes`         | 按形状逐项列出标签，而不是把一个种类折叠在一行中（Vulkan） |
| `stages`            | 主机端阶段计时（见下文）                         |
| `vram`              | 在权重加载后记录正在使用的 VRAM                         |
| `out`               | 同时将退出报告作为 JSON 写入此路径            |
| `diffusion_trace`   | 扩散模型的逐步调度/熵跟踪       |
| `metal_device_time` | `off`/`flush`/`counters`：Metal 获取逐算子设备时间的方式      |
| `metal_debug`       | 额外的 Metal 性能剖析器输出                                |

`stages` 是覆盖每条管线的单一开关：提示词/解码吞吐量、解码设置与执行的划分、每块预填充构建/编译/执行的划分、MTP 验证时间、扩散逐步阶段。只有对应管线运行过才会输出该节。过去这由五个独立调节项控制（`INFR_PROF`、`INFR_PROF_DEC`、`INFR_PROF_PF`、`INFR_MTP_TIME`、`INFR_DIFFUSION_TIME`），因此要回答“实际时钟时间花在何处”，找到第一个开关后还得再发现四个开关。

所有 backend 都会输出同一种表格，标明 backend 名称，并按总耗时排序：

```
[prof:vulkan]     4.91   27.3%       14     350.7  Linear m=128 3072x1024 Q6K
[prof:vulkan]     2.25   12.5%       28      80.2  Linear m=128 1024x6144 Q4K
[prof:vulkan]     1.14    6.4%       28      40.8  Attention rows=128 kv=128 h=16/8 d=128 causal
```

每个后端还会将行汇总为覆盖整个运行的单一进程退出汇总（`== per-op device report: …`）；这也是 `INFR_PROFILE=1` 主机函数表写入的报告，并且是 `INFR_PROF_OUT` 写为 JSON 的同一报告。因此主机节告诉你 GPU 路径**在** `replay_n` 中等待，而设备节告诉你它在等待哪些算子：
现在任意 backend 都能提供这一信息，不再只有 Vulkan 支持。

在 U1–U3 统一之前，这套能力分散在四个独立工具里，并且实际造成了问题：GPU 测试设置
`INFR_PROF_OPS=1` 时**没有任何剖析输出**（只有 CPU backend 读取它，Vulkan 则使用
`INFR_PROF2`）；Metal 表格还会悄悄包含 benchmark 中不计时的 warmup 前向。**因此，若引用此前
commit message 中的 profile，各项占比都被 warmup 稀释了。**统一入口位于
`crates/infr-core/src/prof.rs`；source tripwire 会在 backend 越过该接口时让构建失败。

**标签。**采用“op 类型 + 决定成本的 shape”，只有工作内容相同的 dispatch 才会合并到同一行。
因此，28 层模型会折叠成少数几行；`us/ea` 表示某一种 shape 的成本，而不是把成本可能相差
100 倍的 shape 混成平均值。`Linear` 附带 `m`（区分同一权重上的 Decode GEMV 与 Prefill GEMM），
`Attention` 附带 `kv_len`（完整体现上下文深度），`MoeFfn` 附带 expert geometry；其他算子只显示
op 类型。

**单位不一定都是 device time**，表格会注明具体单位。Metal 的常开模式测量主机侧 _encode_ 墙钟
时间（`[prof:metal] … host-encode`），因为它将多个 op 合并进一个 command buffer，更细粒度的
计时并非免费；若 encode 占比很高，这本身就是诊断结果。只有 `Unit::Device` 行会进入进程退出
汇总，因此不会误将 host-encode 时间与 device time 相加。

### 后端特定说明

- **Vulkan** 按 kernel 名称而非 op 类型计数：一个 op 经 Recorder 降成多个 dispatch，kernel 更细。标签由 `be.kernel("name", …)` 随 ComputeKernel 传递，copy/fill 标为 `copy_buffer`/`zero_fill`；`Recorder::label_next` 可区分共用 MMQ 的 expert_gateup/expert_down，以及 LM head/投影。op_shapes 增加 shape 桶。历史计数曾有绝对 µs overflow 风险，不仅看占比也检查计数类型和范围。record-once replay 的 query 固定，不能逐次测时；用 `--set kernels.vulkan.no_replay=true` 强制重新录制，结果包含不同路径开销。
- **Metal** 的 `--set prof.metal_device_time=counters` 在阶段边界采样 device timestamp，并关闭本次 replay；`flush` 每 op 排空会破坏 batch，而且 replay 不遍历 op 时不能据单个 token 样本推算全程。默认 off 时取得 host encode 而非 GPU执行时间；设备选择用 `--dev MTL0`。
- **CPU** 逐 op 的 host wall 就是 host backend 执行时间，表中作为 `Unit::Device`。
- `infr bench` 各 backend 的 untimed warmup/depth-warm 不纳入正式 profile，warmup 后重置汇总。若出现未请求的 m=1/m=8 Linear，要检查 warmup 是否泄漏。
- 分别看 GEMM、elementwise、attention 与 op count；每 chunk 50K dispatch 本身可能就是瓶颈，不仅比较单个 kernel 耗时。
- 微型探针：`crates/infr-vulkan/tests/`（`gemm_bench`、`decode_gemv_bw`、`small_m_bench`、`bandwidth_probe`），`crates/infr-metal/tests/`（`gemv_bw`、`dispatch_overhead`、`moe_bw`）。其中的链式派发数据会在连续运行间受温度影响而不稳定；应以端到端基准测试作出接受/回退决定。
- 对可疑的 GEMM 形状，将其加入按形状微基准测试（`crates/infr-vulkan/tests/gemm_bench.rs`）并输出 µs + TFLOPS。与内核的已知上限比较：远低于上限的形状属于形状问题（网格/占用率），上限本身过低才是内核微架构问题。

### 构建时自动插桩（`INFR_PROFILE=1`）

全工作区 CPU 侧函数计时：**默认构建中零代码**，且**无需手动修改计时器**：

```bash
INFR_PROFILE=1 cargo build --release -p infr-cli   # instrumented binary
./target/release/infr bench "$M" -p 0 -n 128 -r 3  # report prints at exit
INFR_PROF_OUT=prof.json ./target/release/infr ... # + full table as JSON
```

进程退出时，运行会输出一张合并后按自身耗时排序的表：

```
== INFR_PROFILE report: 200 sites, 1 threads, wall 1.96s (...), accounted self 1.92s ==
        self   self%        total        calls  avg(self)  function
     624.2ms  31.91%      624.2ms           49     12.7ms  infr_vulkan::recorder::RecordedCmd::replay_n
     581.0ms  29.70%      670.0ms            1    581.0ms  infr_gguf::dequant::dequant_unified
       7.7ms   0.39%      728.3ms            4      1.9ms  infr_llama::seam::runner::generate_dense_backend
```

`self` 不包含花费在已插桩被调函数中的时间（`total` 为包含时间；递归仅在最外层帧计入一次包含时间）。累加是线程本地的（热点路径中没有共享状态）；各线程表在报告时合并，因此覆盖 rayon/自旋池线程。

**实现方式。** 热点crate的build.rs将 `INFR_PROFILE=1` 转为 `cfg(infr_profile)`；infr-core、infr-cpu、infr-vulkan、infr-gguf、infr-llama中的顶层fn/impl带 `#[cfg_attr(infr_profile, infr_prof::instrument)]`。未设置环境变量时，该属性会被完全编译掉，默认二进制不包含性能剖析代码、运行时分支或任何额外内容。设置后，`infr-prof` 过程宏会重写项目中的每个函数，使其打开可跨越 `?`、`return` 和 panic 的 RAII 范围（`infr-prof-rt`）。

**覆盖规则——新函数需要什么：**

- 已注解 `impl` 块中的新方法：**无需任何操作**，会自动覆盖。
- 新的顶层 `fn` 或新的 `impl` 块：添加一行 `#[cfg_attr(infr_profile, infr_prof::instrument)]`。
- `#[inline]` / `#[inline(always)]` 函数会被**自动跳过**：将函数声明为内联已表明它小于约 50ns 的探针调用对。非内联且低于 100ns 的叶函数可使用 `#[cfg_attr(infr_profile, infr_prof::skip)]` 退出（见 `infr_gguf::dequant::k4`：每次运行 1.7e9 次调用时，剖析此类叶函数会让 CPU 解码慢 12 倍并淹没报告）。
- `const fn`、`async fn`、`#[naked]`、`#[test]` 函数和闭包会自动跳过；泛型在各实例化间共用一个位置；trait 实现报告为 `<Ty as Trait>::fn`。

**开销（Qwen3-0.6B Q4_K_M、7900 XTX + CPU 后端）：** GPU pp512 26.0k → 25.2k t/s（约 3%），GPU tg128 处于噪声范围（618 对 616），CPU pp128 处于噪声范围（933 对 943），CPU tg32 118 → 105 t/s（约 11%，解码的逐行内核调用每个约为 200ns，因此探针影响可见）。插桩构建用于归因，不用于比值基准测试；绝不可在性能日志中引用插桩数据。

**设计说明（被拒绝的替代方案）。** （B）拒绝通过 build.rs 将 crate 源码用 syn 重写到 OUT_DIR + `include!`：非内联 `mod foo;` 的解析会在 `include!` 下失效，每个嵌套模块都需要路径重写，且 rust-analyzer/增量构建会看到幽灵源码；唯一收益只是每个项目少写一行属性，不值得引入这些脆弱性。（C）rustc 级插桩（`-Z instrument-xray`、mcount 风格钩子）仅支持 nightly，并向外部工具输出，在 stable 上没有 self/child 或逐线程信息；samply 已覆盖“完全没有源代码标记”的场景。（非内联 `mod foo;` 上的过程宏属性也不稳定，因此注解按项目而非按文件。）

**历史替代方向与当前边界。** 早期方案希望用构建期自动插桩替代零散的 `Instant::now()`/eprintln计时；0.10.0的prof.*仍在使用，不能把“以后移除”写成已退役事实。INFR_PROFILE是build.rs输入而非运行时Config。设备逐op计时仍是runtime门控，可不重建而切换；标签由dispatch入口自动传递。两者可组合输出host函数表与GPU aggregate，INFR_PROF_OUT的JSON含sites和gpu：host说明在replay_n等待，device说明所等op。0.10.0还增加独立有界timeline；需要明确同步区间的结构化测量不属于随意加计时器。

**不要再新增另一套逐算子性能剖析器。** 新后端通过调用 `infr_core::prof::enabled` 并将耗时交给 `OpProf`，即可获得全部功能：调节项、标签语法、表格、退出汇总和预热抑制。它唯一需要自行编写的是如何从自身 API 获得耗时，因为该部分确实不同（时间戳查询/计数器样本/主机时钟）。若 `no_backend_feeds_the_aggregate_behind_the_shared_reporter` 越过此边界，构建会失败。

### CPU 性能剖析（samply）

CPU归因优先采样profiler，不为一次调查塞入零散计时器。历史denoise exec在INFR_PROF_OPS打开时6.0s、关闭3.85s，说明eprintln/同步开销会干扰测量。采样能保留函数、行和调用栈，不需要改代码。

可用[samply](https://github.com/mstange/samply)（cargo install samply），通常不需要root；workspace release的line-tables-only提供符号/行号而不改变codegen。下面是原调查环境的命令，使用前确认本机工具和参数。

```bash
# Interactive: records, then opens the Firefox Profiler UI (flame graph,
# per-thread timelines, inverted call tree = self-time ranking).
samply record ./target/release/infr bench "$M" --ngl 0 -p 16 -n 32 -r 1

# Headless (agents, SSH): save the profile, then rank self-time per function.
samply record --save-only -o prof.json.gz -- ./target/release/infr bench ...
scripts/samply-top.py prof.json.gz 30
```

samply-top.py百分比是所有线程的total thread-seconds，不是wall time。32线程中3%的函数如果串行，仍可能主导端到端；accum.rs、crossbeam_epoch、Producer::fold_with较高可能表示fork-join/idle-spin成本。要逐行归因，使用samply load同一prof.json.gz。

samply不提供DRAM带宽、IPC、cache miss等硬件计数器，带宽/计算分类需perf stat等工具；先估算表大小×流送次数的字节量，不以猜测代替计数。Linux发行工具安装命令依发行版而异。现有prof.ops、prof.stages和pager聚合仍可作为快速首轮读数，实际profile有开销，关闭时不等于每种instrumentation都构建期移除。

## 瓶颈分类 — 性能剖析的含义

修复前先分类。每一类有不同杠杆，专项中每类都有可工作的示例：

**1. 完全错误的内核（覆盖/门控缺陷）。** 快速路径存在，但路由门控排除了该形状或 dtype，因此回退到慢路径（标量 GEMV、dp4a mmq、64×64 分块）。这是代码库中成本最低的收益：一行门控修复。示例：在添加 `n % 128` 分块后，warp-GEMM 门控仍要求 `n % 256 == 0`，导致 gemma3-1b 的 ne=1152 投影以约 10 TF 落在标量 mmq 上。每次合入新的内核变体时都应审查门控；其上游的**每一个**资格检查现在都可能已过时。

**2. 网格未填满。** 内核本身正常，但派发未填满 GPU（窄 n、小 m → 工作组少于计算单元）。杠杆：更窄的分块（相同工作产生更多工作组）、split-K（并行化归约维度，但保持固定归约顺序以保证确定性）、将同级 GEMM 融合为一次宽派发（Q/K/V；gate+up）。

**3. 占用率上限。** 网格已填满，但每个 SM/WGP 并发运行的工作组过少，无法隐藏延迟，通常受 LDS 或寄存器预算限制。通过实验诊断：若更大的分块反而失败，受限的是占用率而非计算。杠杆是缩减每工作组资源。示例：从共享内存移除 A 分块（先将 A 转为 f16 一次，`coopMatLoad` 直接从全局内存加载）后，LDS 从 25→20 KB，工作组/WGP 从 2→3，8B GEMM 从约 28 TF 提升到约 44 TF。

**4. 派发/启动开销。** 总 GPU 时间由大量微小派发及其间屏障主导，而非任何算子的计算。迹象包括：逐算子成本与 GEMM 相当的逐元素算子、数以万计的算子计数、small-m 块耗费荒谬的实际时钟时间（固定成本不会随规模缩小）。杠杆：**每个阶段对全部项目只做一次派发**（将项目索引置于一个网格维度，提前退出多余工作组，空工作组几乎免费）、将相邻逐元素阶段融合为一个内核、在固定最坏情况网格 + 提前退出更简单时移除间接参数机制。示例：批量 MoE FFN 从每层约 1050 次派发 + 约 110 个屏障（按专家波次）变为约 13 次单一派发，在零 GEMM 改动下 pp512 从 0.59x → 0.91x。

**5. CPU 受限 / 主机介入。** GPU 在主机执行本可作为着色器完成的工作时处于_空闲_状态：路由决策、图中途读回、逐 token 重建图、迫使提交-等待-重新提交模式的调度。迹象是实际时钟时间远大于 GPU 算子时间总和，或 `INFR_PROF_STAGES` 显示记录/主机时间与 GPU 时间相当。**常规规则是：若算子在 CPU 上运行且能表示为管线中的着色器，就把它移入管线。** 每次主机读回都会让整条管线停顿（提交 + 等待 + 重新提交）；每个可由内核作出的主机侧决策都会让 GPU 挨饿。树中已有的示例包括 GPU 常驻 MoE 路由（top-k、桶计数/扫描/散射均在 GPU 上，旧路径会在图中途下载计数以确定 GEMM 大小）和仅记录一次的解码重放（`_dyn` 内核从设备侧参数缓冲读取 pos/kv_len，因此主机无需每个 token 重新记录）。发现主机循环向 GPU 发送微小提交时，修复几乎总是“让 GPU 计算自己的控制数据”。

**6. 内核微架构。** 仅在排除 1–5 后考虑：内核在完整网格上运行到自身上限，但该上限低于竞争对手。这是修复成本最高的一类（分块形状、流水线、累加器精度），且实验经常失败；相信任何变体前都应在按形状微基准测试上测量它。

按上述顺序处理这些类别：它们大致按每百分点所需工作量排序。

## 正确性不可妥协

只有输出不变时，性能工作才算真实有效。

- **先一致性，后基准测试。** CPU 后端是基准：`cargo test -p infr-llama --release --test cpu_backend`（每个模型家族的黄金值）加 infr-vulkan 算子一致性测试套件（GPU 门控测试使用 `-- --ignored`）。
- **没有证据绝不重新确认黄金值。** 若哈希变化，先捕获前后真实生成（`infr run …`）并验证文本，之后才重新确认，并在提交中说明。优先选择按构造逐位一致的设计：使用固定顺序归约而非原子操作，保持相同的舍入点和累加顺序。本专项几乎每个片段（QKV 融合、split-K、A_GLOBAL、批量 MoE）均经设计保证逐位一致，黄金值从未变动。
- **确定性。** 引入的任何并行归约都必须有固定求和顺序。会影响_数值_（而不只是放置）的原子操作不允许使用。
- **零初始化纪律。** `Backend::alloc` 采用 calloc 风格；只有能证明每个元素均在读取前写入时才使用 `alloc_uninit`。仅当能证明垃圾数据绝不会流入真实输出时，包含垃圾的填充行才可接受（GEMM 行相互独立；应在注释中说明理由）。
- **测试使用 `Config` 配置行为，绝不使用环境变量。** 构建所需的 `Config` 并传入（`VulkanBackend::new_with(cfg)`、`CpuBackend::new_with(cfg)` 等）。测试中设置 `INFR_*` 变量是进程范围的写入，会与二进制中所有其他测试竞争，这正是原始“flaky golden”；现在也没有 `EnvGuard` 可规避它。见 [`config.md`](../../reference/configuration.md)。

## 代码库习惯

- 每次提交前运行全工作区 clippy（`cargo clippy --workspace --all-targets --locked -- -D warnings`）和 `cargo fmt --all`；按 crate 运行 clippy 曾遗漏 CI 失败。
- infr-metal 受 macOS 门控：修改共享类型（`Op` 字段、trait 签名）时，Linux 构建会静默通过。请同时盲改其源码和测试，并让 macOS CI 验证。
- 删除着色器时，应一并删除其 build.rs 条目、`_spv` 访问器和记录器方法，然后确保_干净_构建通过（目标目录中陈旧的 `.spv` 文件可能在本地掩盖缺失条目）。
- 在一份 `.comp` 源文件上通过编译时 define 新建内核变体优于复制粘贴着色器。记录器绑定数组按风险排序：**输入在前，输出在后**；添加只读输入的变体必须重新编号绑定，以保持输出位于末尾。
- 临时缓冲应池化（`pooled(pool, be_, tag, bytes)`），以便图中每个相同形状算子共享一次分配。

## 时间花在何处（优化专项日志，2026-07）

作为校准，以下是各类修复在相对 llama.cpp 的 pp512 比值上产生的收益类型：

| 改动片段                             | 分类             | 结果                                                           |
| --------------------------------- | ----------------- | ---------------------------------------------------------------- |
| 融合 QKV + 窄分块 + split-K | 网格未填满    | 0.6B 0.56 → 0.74x                                                |
| A_GLOBAL（LDS → 占用率）        | 占用率         | 0.6B → 0.92x，8B 0.72 → 0.83x                                    |
| 批量 MoE（派发折叠）   | 派发开销 | MoE 0.59 → 0.91x                                                 |
| warp 门控 n%256 → n%128           | 覆盖门控     | gemma3-1b 0.60 → 0.67x                                           |
| warp-GEMM Q5_K/Q4_0/Q2_K          | 覆盖门控     | Q4_0 0.6B 0.43 → 0.85x，Q5_K 0.42 → 0.61x，Q2_K 14B 0.47 → 0.68x |
| IQ4_XS dqblk + warp-GEMM          | 覆盖门控     | 0.6B pp 0.25 → 0.67x（2.7x），tg 0.39 → 0.68x                     |

**当时未闭环项，使用前重新测量：** 量化断崖大部分已关闭：Q5_K/Q4_0/Q2_K/IQ4_XS接入warptile，K-quant已有dqblk只需接线，IQ4_XS新增32-element dqblk也改善native_gemv。当时最大Q4_K_M差距是Gemma4 E2B Decode（tg@d4096约0.64x、tg128约0.70x），其次qwen35 DeltaNet occupancy/narrow-n split-K、Gemma3-1B窄GEMM（pp0.66x）、宽shape warptile约36TF对照llama约45..50TF，以及Metal parity。这是7/8月专项状态，不代表0.10.0最新排名。

## Coopmat 操作数层级（fp8 / int8 / bf16）— 相对 f16 经测量确认的死路

Vulkan 后端枚举设备接受的 cooperative-matrix 组件类型，并在启动横幅中报告它们（`f16cm / bf16cm / f8cm / i8cm`）。在 RDNA4（RX 9060 XT / Navi 44、Mesa 26.1.4）上四种均存在；在 RDNA3（7900 XTX）上仅有 f16 + int8。很容易想将量化权重 GEMM 路由到“更快”的 8 位单元。**我们已在真实硬件上构建和测量了全部路径，均未胜过 f16 coopmat GEMM。没有原生低位模型时，不要重新开启此方向。**

以下保留原RDNA4/Mesa26.1.4实验，不把旧INFR开关当作0.10.0运行时承诺；当前字段查Config参考。

- **fp8 E4M3 coopmat：** 完整路径按行把activation缩放到±448，使用native_gemm_warp同类tile，再按行descale。当时INFR_F8_COOPMAT/INFR_F8_PREPACK默认off。shader内dequant只有f16的0.73x（同样Q8解码外加scale/epilogue）；预打包E4M3去掉dequant后，1024×6144/m512仍364.5µs对f16 364.3µs，窄shape因缺split-K/A_GLOBAL更差。这些GEMM约17.6TF/s，latency/occupancy限制下更快MAC没带来净收益。
- **int8 coopmat：** 当时INFR_I8_COOPMAT默认off，原始约慢4.8..5.6倍，经shared逐block rescale后仍仅约f16的0.73x。v1没有in-flight逐元素scale；dp4a mmq让每线程拥有accumulator、可后置scale，是不同整数路线。
- **coopmat2逐元素：** VK_NV_cooperative_matrix2/coopMatPerElementNV在RDNA4 RADV需radv_cooperative_matrix2_nv=true。GL_NV_cooperative_matrix2的`void coopMatPerElementNV(out coopmat r, coopmat m, T fn)`支持row/col fragment访问可去shared rescale往返，但probe仍无收益：single-tile2.16对2.24µs属噪声；512×2048×2048、64 rescale blocks/4096workgroups为1239.8对1195.3µs、0.964x。shared成本不是主要瓶颈。保留crates/infr-vulkan/examples/coopmat2_test.rs作为后续Mesa重测工具，不能据barrier更少翻案。

```bash
radv_cooperative_matrix2_nv=true cargo run --release -p infr-vulkan --example coopmat2_test
```

- **bf16 coopmat：** native_gemm_warp的BF16CM变体使用bfloat16_t/CMTYPE，默认SPV字节一致性当时用md5核对。保留bf16指数范围、不clamp到f16的65504，但同shape/non-A_GLOBAL的Qwen3-0.6B BF16/pp512为865.1µs，对f16-clamp769.9µs，宽shape约慢12%、窄约27%。可能是RADV codegen成熟度或硬件率差，未单独证实；因此不能承诺bf16保真且同速。standalone native_gemm_bf16cm已退役。native16-bit bfloat16_t[] SSBO在独立测量中还比uint32双bf16/word+dqblk慢约27%，packed32bit访问对RADV更好，不能把“少转换”当更快。
- **为什么换8-bit操作数没赢：** 当时Vulkan/AMD low-bit float权重仍需先转WMMA类型；ggml-vulkan的MXFP4/NVFP4也先以UE4M3scale×码本解成f16。原调查对比NVIDIA Blackwell block-scaled fp4 MMA（e2m1操作数、UE4M3scale），RDNA4没有对应e2m1 coopmat配置。这个跨硬件判断保留为当时观察，不是已在本项目采用的新内核。若再研究，须用当前native权重格式、设备能力和matched模型验证，而不是重复dequant-bound GEMM的操作数替换。
