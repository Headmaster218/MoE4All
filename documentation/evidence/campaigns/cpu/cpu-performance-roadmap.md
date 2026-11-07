# cpu-perf.md — CPU 后端性能路线图

这是从 CPU 性能审查汇总的 `infr-cpu` 参考后端结论和优先工作清单。按**实现难度由低到高**排序，以便先落地成本低、确定性高的收益。

## 结果快照

目前已落地（除非另有说明均逐位一致；精度翻转分项在验收前会检查与独立 Vulkan int8 路径的连贯且 token 一致）：

| 分项 | 模型 / 量化 | 解码 | 预填充（pp512） |
| ---------------------------------- | ----------------- | ------------ | ----------------- |
| conv1d parallel (`ac9c228`)        | Qwen3.5-9B Q4_K_M | —            | flat (GEMM-bound) |
| mmap madvise（`5ed932a`） | Qwen3.5-9B Q4_K_M | 中性 | 中性 |
| DeltaNet 头并行（`9595bf3`） | Qwen3.5-9B Q4_K_M | 持平 | 67→110 t/s（+63%） |
| native int8 **Q4_0** (`6e7decd`)   | Qwen3-0.6B Q4_0   | +142% (2.4×) | +239% (3.4×)      |
| native int8 **IQ4_XS** (`304dd42`) | Qwen3-0.6B IQ4_XS | +156% (2.6×) | +320% (4.2×)      |
| native int8 **Q2_K** (`1e90613`)   | Qwen3-0.6B Q2_K   | +29%         | +71%              |
| native int8 **Q3_K** (`d559984`)   | Qwen3-0.6B Q3_K_M | +185% (2.9×) | +199% (3.0×)      |
| native int8 **Q4_1** (`f4738a1`)   | Qwen3-0.6B Q4_1   | +138% (2.4×) | +243% (3.4×)      |
| native int8 **IQ4_NL** (`0ef8366`) | Qwen3-0.6B IQ4_NL | +47%         | +253% (3.5×)      |

### 完整 CPU↔Vulkan 原生量化一致性（已由单元测试验证 — 无模型）

CPU 现已为**Vulkan 具有原生 mmq 内核的每种格式**提供原生 int8 内核。下面六项补齐最后缺口；它们都没有可用于端到端连贯性检查的、受支持架构的小型 GGUF（`IQ2_S`/`IQ3_S` 仅大型模型；`Q5_1`/`Q2_0` 为旧格式；`MXFP4`=gpt-oss，`infr-llama` 未实现其架构；`NVFP4` 为前沿格式）。其门槛为**SIMD↔标量逐位一致 + 与精确 `dequant_block` 的容差一致性（量化激活点积采用严格 1e-3）**：数学验证严格，不生成连贯性文本。

| 格式 | 提交 | 说明 |
| --------- | --------- | --------------------------------- |
| `Q5_1` | `e7465ed` | 仿射，Q5_0 5 位 + Q4_1 最小值 |
| `Q2_0` | `3f7c79e` | Bonsai 三元，64 块，2×Q8x32 |
| `MXFP4` | `29ee2e5` | IQ4_NL + E8M0 缩放（gpt-oss） |
| `NVFP4` | `06cc0ef` | MXFP4 码本 + 每 16 个 UE4M3 |
| `IQ2_S` | `8e616c3` | 网格码本，单次展开行 |
| `IQ3_S` | `34fd4f2` | 网格码本，每 32 个缩放 |
| `IQ2_XXS` | `aa07a2f` | 网格，KSIGNS 查找，连贯 ✓ |
| `IQ3_XXS` | `e97a913` | 网格，与 GPU token 一致 ✓ |
| `IQ2_XS` | `7e0fb1d` | 网格，9 位索引（单元测试） |
| `IQ1_S` | `93e3225` | 1 位，`dl·(iprod + delta·asum)` |
| `IQ1_M` | `c70111f` | 1 位，缩放中的 d，每 8 个 delta |

网格/码本集合（`IQ2_XXS`/`IQ2_XS`/`IQ2_S`、`IQ3_XXS`/`IQ3_S`、`IQ1_S`/`IQ1_M`）使用“单次展开行 → 有符号 i8 → 每组整数点积”模式；
每项镜像 `dequant_codebook`，并以相对精确解码的**严格 1e-3** 容差为门槛。连贯性方面：`IQ3_XXS` 与 GPU 原生路径 token 一致；
`IQ2_XXS` 匹配前导 token；1 位 `IQ1_S`/`IQ1_M` 在噪声底部混沌（int8/f32/CPU/GPU 全部发散——1e-3 数学检查是正确性门槛）。

**CPU 现已原生处理每一种权重量化格式**，包括三元 `TQ1_0`（`b7a4201`）/ `TQ2_0`（`6336df3`），将 `(digit−1)` 折入有符号 i8 +
单缩放整数点积。与 Vulkan 原生集合完全一致。

已延期：第 3 项（DeltaNet 克隆——测得约 0.1%，可忽略）。剩余：第 8 项（f16/bf16，低优先级）、第 9 项（受 `perf` 阻塞）、第 10 项（融合）、
以及 IQ4_XS 的 VNNI 批处理。

## 跨后端快速内核覆盖

CPU、Vulkan 和 Metal 现在各自都为 infr 支持的**每一种**权重量化格式（24/24）提供原生快速内核——任何后端均无格式回退到反量化→浮点。
完整矩阵和每后端解码策略已移至单独文档：**`kernels.md`**。（CPU 以三元 `TQ1_0` `6336df3`/`b7a4201` 补齐最后缺口；
TriLM-3.9B TQ2_0 预填充 16.5→131.9 t/s ≈ 8×。）

## 背景：两种模式、两类不同瓶颈

CPU 推理按批大小明显分为两类，每类的缓存/带宽状况不同：

- **解码（`m == 1`）受 DRAM 带宽限制。** 真实模型的权重以 GB 计（`Q4_K` 9B ≈ 5 GB；即使 `Q2_K` 0.6B ≈ 180 MB），均**远大于 L3**。
  每个权重每 token 读取一次，从 RAM 连续流式读取。在顺序流上，硬件预取器已使内存控制器饱和，因此可扩展解码的唯一杠杆是**更少的流式字节**（原生量化）
  加上 **TLB** 缓解（大页）。此处的软件权重预取无收益——硬件预取器已预测该流。
- **预填充（`m > 1`）受计算 + 缓存复用限制。** 权重仍会流式读取，但每个权重行跨 `m` 个激活列复用，因此保持激活分块常驻 L1/L2 才是杠杆。
  这是分块/分块尺寸/融合产生收益之处。

### 参考硬件（开发机）

AMD Ryzen 9 9950X3D（Zen 5、3D V-Cache）：**128 MiB L3**、16 MiB L2（1 MB/核）、768 KiB L1d（48 KB/核）、16 核 / 32 线程、
**1 个 NUMA 节点**。ISA：AVX-512 F/BW/VL/DQ/CD、**AVX-512-VNNI**、AVX-512-BF16、AVX-VNNI、F16C、3DNow-prefetch。
大容量 X3D L3 有助于预填充（整层激活 + KV 保持热数据）；它**不能**挽救解码（权重仍远大于 128 MB）。

## 已完成的工作（不要重复）

- **权重原生 mmap。** `Op::Linear` 每次从 mmap 直接流式读取一个行主序 GGUF 权重行——RAM 中不实体化 f32。
- 常见 k-quant 的**int8 量化激活 VNNI 点积**——**Q4_K、Q5_K、Q6_K、Q8_0、Q5_0**——具有标量→AVX2→AVX-512BW→VNNI 内核及最多
  **8 行缓存分块**（激活加载一次，跨 8 个权重行复用）。这已是“原生格式、有损但快速、缓存友好”策略；这也是 Q4_K_M 模型已很紧凑的原因。
- 在虚拟 `[state‖x]` 序列上**并行化预填充 conv1d**（`ac9c228`）。逐位一致；孤立内核约 7.3×，但端到端持平（conv1d 不到受 GEMM 限制预填充的 1%）。

## 瓶颈排序（列表为何如此排列）

1. **Fewer bytes (native quant coverage)** — dominant decode lever.
2. **Hugepages / madvise on the weight mmap** — real TLB win on the GB stream.
3. **Op fusion** — cuts intermediate DRAM round-trips.
4. **Prefill tile tuning** to the X3D topology — real but measure-first.
5. **Software prefetch** — micro-opt, usually a wash. Not a strategy.

---

## 工作清单（低 → 高难度）

每个分项：数学不变处（并行化、精确算子融合）采用 TDD、逐位一致；数学变化处（int8 激活量化有损）采用容差一致性 + 经认可的 golden 重新认证。
一次只做一个分项；基准测试前先验证正确性。

**重新认证纪律：**只有新输出经**验证正确/连贯**后，golden 才可重新认证——绝不盲目接受 diff。对于精度翻转，这意味着确认模型仍生成合理连贯的文本
（将短生成与 f32/GPU 路径比较），并且 CPU 结果在容差内匹配 GPU int8 结果。若 golden diff 以看似乱码的方式改变产生的 token，
那是 bug，而非精度翻转。

### 1. 权重 mmap `madvise` + THP 提示 — _简单_

- **内容：**在权重 mmap 上向内核提示：`MADV_HUGEPAGE`（2 MB 页面减少数 GB 顺序读取上的 dTLB 页表遍历）、`MADV_SEQUENTIAL` /
  `MADV_WILLNEED`（按实际消费方式偏置预读）。
- **原因：**4 KB 页面上的大于 L3 顺序 mmap 读取会重击 dTLB；TLB 是解码流中仍有余量的“预测”结构。大页是在带宽现实下仍能成立的、
  最接近“帮助 CPU 预加载下一区域”的方式。
- **影响：**小到中等的解码 + 权重加载收益；低风险。
- **精度：**无（纯内存提示）。逐位一致。
- **状态：**已完成（`5ed932a`）。`WillNeed` + Linux `HugePage`，尽力而为，不使用 `Sequential`。开发机测得**中性**（热页缓存；THP 在文件支持映射上常无操作）：
  9B Q4_K_M tg64 10.2→10.1，pp512 67.4→67.1（噪声）。零风险地保留用于冷加载 / 启用 THP 的文件系统 / 超 RAM 情形；
  严格 A/B 需要 `perf` 计数器或冷缓存（见“测量”）。

### 2. DeltaNet 头并行 — _简单至中等_

- **内容：**`Op::DeltaNet` 执行串行单线程扫描。外层 `for t` 天然串行（状态跨 token 传递），但跨 value head 的内层 `for h in 0..n_vhead`
  循环**完全独立**——每个 head 拥有不相交的 `state[h*kd*vd..]` 切片、自身 out 切片，且仅读取共享输入。按 head 并行：每个 head 任务在
  自身状态副本上运行完整 `t` 扫描（`pool.collect`），随后写回 state + out。
- **原因：**DeltaNet 是**约 75% Qwen3.5 层**的线性注意力路径（每第 4 层才有完整注意力）——不同于 conv1d，是主要 CPU 成本。16 个头（9B）
  使主导注意力算子最高可有 16 路并行。
- **影响：**预期对预填充**和**解码有真实收益。
- **精度：**逐位一致（每头浮点顺序相同；状态重建为复制）。
- **状态：**已完成（`9595bf3`）。`deltanet_scan` 辅助函数，每个头一个池任务。逐位一致（同等测试、精确 f32）。
  **Qwen3.5-9B Q4_K_M 预填充 pp512 67.3→109.8 t/s（+63%）**；解码持平（10.3→10.4，行数=1 时受 DRAM 限制）。
  这是目前该专项最大的预填充收益。

### 3. 消除 DeltaNet 输入克隆 — _中等_

- **内容：**DeltaNet 分支每个算子都 `.clone()` 整个 `q/k/v` 缓冲区（`[rows, heads·dim]`），纯粹为避开借用检查器（state 需要 `&mut vals`，
  而 q/k/v 需要 `&vals`）。引入不相交 `vals` 访问器（拆出一个 `&mut` 索引，其余借用 `&`）以去除克隆。相同模式在其他算子中复现
  （conv1d 也有克隆），因此访问器可复用。
- **原因：**预填充时，这些克隆对每个 DeltaNet 层是约 100 万浮点数 × 3 的纯分配 + 复制流量。
- **影响：**中等预填充收益；消除分配器压力。
- **精度：**逐位一致。
- **状态：**已延期——测得可忽略。9B 有 18 个 DeltaNet 层 × 约 12 MB（q/k/v）克隆 = 每次 512 token 预填充约 216 MB memcpy，
  约为 4.7 s 预填充的 **~0.1%**；临时分配立即释放（无 RSS 顾虑），且解码为 rr=1（无可克隆内容）。
  不值得承担不相交 `vals` 访问器复杂度 / 不安全借用拆分。仅当性能分析指出 DeltaNet 分配，或因其他原因进行更宽泛的 `vals` 访问器重构时再考虑。

### 4. 原生 int8 点积：**Q4_0** — _中等_

- **内容：**`Q4_0` 当前回退到 `bytes_to_f32` 反量化 + f32 点积（缓慢的通用回退）。添加原生 int8 激活内核（标量/AVX2/AVX-512BW/VNNI +
  batch/batch8），并接入 `m==1` 与 `m>1` 调度，镜像 `Q8_0` 和 GPU 原生 Q4_0 内核。
- **原因：**Q4_0 无处不在；GPU 已有原生内核。它是未覆盖格式中第一个且最简单的。
- **影响：**对 Q4_0 模型（解码 + 预填充）影响大；消除 f32 扇出。
- **精度：**int8 激活量化有损 → 这会**改变 Q4_0 的 CPU 参考输出**。以 f32 参考的容差一致性测试约束误差；Q4_0 gpu_seam golden 是经认可的
  **精度翻转重新认证**（`--include-ignored`），新 CPU 路径应匹配 GPU int8 结果，而非旧 f32。
- **状态：**已完成（`6e7decd`）。`vec_dot_q4_0_32_batch`（标量 + AVX2 + VNNI）从 Q5_0 克隆（18 字节块、偏移 8、无第 5 位）；
  复用 `Q8x32`。已接入解码 + 预填充（此前解码没有 Q4_0 内核）。**没有 golden 改变**，无需重新认证——CPU 贪心输出连贯，
  且**与独立 Vulkan int8 路径 token 一致**（“……是 **Paris**。”）。SIMD 与标量预言机逐位一致；与全精度反量化容差一致。
  **Qwen3-0.6B Q4_0 CPU：解码 28.7→69.6 t/s（+142%），预填充 128.7→435.9 t/s（+239%）。**

### 5. 原生 int8 点积：**IQ4_XS** — _中等_

- 与 Q4_0 相同的处理，适用于常见小模型格式 IQ4_XS（本地 Qwen3-0.6B 有一个）。GPU 参考存在（quant-cliff-warp）。
- **精度：**按第 4 项进行精度翻转重新认证。
- **状态：**已完成（`304dd42`）。`vec_dot_iq4xs` / `_batch`（标量 + AVX2 + AVX-512BW 单 token）以 Q6_K 为模型，
  但采用 `KVALUES_IQ4NL` 码本 `pshufb` 查找和 Q8_0 的 abs/sign 有符号点积技巧。输出连贯且与 Vulkan int8 token 一致（“……是 **Paris**”）；
  无 golden 改变；SIMD 与标量逐位一致。**Qwen3-0.6B IQ4_XS CPU：解码 37.8→96.6 t/s（+156%），预填充 129.7→544.8 t/s（+320%）。**
  后续：尚无 AVX-512-VNNI **batch** 变体（batch 运行 AVX2）——`dpbusd` batch 路径可在 VNNI 主机上进一步提升预填充（此机有 `avx512_vnni`）。

### 6. 原生 int8 点积：**Q2_K、Q3_K** — _中等至高_

- 具有打包缩放的 K-quant 超级块格式；解码工作多于 Q4_0，但属于相同 int8 激活机制。每种一个分项。
- **精度：**按 dtype 进行精度翻转重新认证。
- **状态：**Q2_K **已完成**（`1e90613`）。`vec_dot_q2k` / `_batch`（标量 + AVX2 - AVX-512BW + VNNI）以仿射 Q4_K 为模型；
  2 位码、每 16 个子块，通过现有 `q8.bsums16` 做最小值校正。输出连贯且与 Vulkan int8 token 一致（“……是 Paris。”）；无 golden 改变；
  SIMD 与标量逐位一致。**Qwen3-0.6B Q2_K CPU：解码 25.2→32.5 t/s（+29%），预填充 127.9→218.5 t/s（+71%）。**
  Q3_K **已完成**（`d559984`）：`vec_dot_q3k` / `_batch`（标量 + AVX2 - AVX-512BW + VNNI），以 Q6_K 的有符号路径为模型，偏移 32→4；
  通过辅助位混排得到 6 位缩放，通过 qs + hmask 位平面得到 3 位码，采用 `−4·bsums16` 校正。输出连贯且与 Vulkan int8 token 一致
  （“……**Paris**”）；SIMD 与标量逐位一致。**Qwen3-0.6B Q3_K_M CPU：解码 35.4→100.9 t/s（+185%），预填充 198.2→592.8 t/s（+199%）。**
  至此整个**K-quant 系列均为原生实现**（Q2_K/Q3_K/Q4_K/Q5_K/Q6_K）。

### 7. 原生 int8 点积：旧式仿射 + IQ 码本 + 网格系列 — _中等 → 高_

- 剩余未覆盖格式，按难度递增排列。
- **精度：**每种 dtype 的精度翻转；验收前相对 Vulkan int8 路径验证连贯性。
- **状态：**
  - `Q4_1` **已完成**（`f4738a1`）：Q4_0 的仿射克隆（`y = d·q4 + m`、`as·(d·iprod + m·bsum)`）。与 Vulkan 输出连贯/token 一致。
    **Qwen3-0.6B Q4_1：解码 28.7→68.2 t/s（+138%），预填充 129.2→442.6 t/s（+243%）。**
  - `IQ4_NL` **已完成**（`0ef8366`）：IQ4_XS 的扁平 32 块码本表亲（`KVALUES_IQ4NL` pshufb + abs/sign 点积、`Q8x32` 激活）。
    输出连贯/token 一致。**Qwen3-0.6B IQ4_NL：解码 34.3→50.5 t/s（+47%），预填充 124.5→439.6 t/s（+253%）。**
  - `Q5_1` **已完成**（`e7465ed`）、`Q2_0` **已完成**（`3f7c79e`）、`MXFP4` **已完成**（`29ee2e5`）、`NVFP4` **已完成**（`06cc0ef`）、
    `IQ2_S` **已完成**（`8e616c3`）、`IQ3_S` **已完成**（`34fd4f2`），以及完整网格/码本集合 `IQ2_XXS`（`aa07a2f`）、
    `IQ2_XS`（`7e0fb1d`）、`IQ3_XXS`（`e97a913`）、`IQ1_S`（`93e3225`）、`IQ1_M`（`c70111f`）——均**已完成**
    （Vulkan 已有这些原生内核；这补齐 CPU 侧）。见顶部“完整一致性”章节和覆盖审计。网格内核单次将网格行展开为有符号 i8
    （标量 gather + `apply_signs`），随后复用 IQ4_XS 每子块缩放 × 整数点积，并在批次中摊销。

### 8. f16 / bf16 native AVX-512-FP16/BF16 dot — _medium_

- **What:** f16/bf16 weights already read native 2-byte (bandwidth already
  minimal), but the dot accumulates in f32 after widening. Add a native
  AVX-512-FP16 / AVX-512-BF16 dot to cut the arithmetic.
- **Why:** compute-only win; the bandwidth is already optimal, so this is
  smaller than the quant slices — do it after the quant gap is closed.
- **Impact:** modest, prefill-leaning.
- **Precision:** changes accumulation precision → tolerance-parity + re-bless if
  the f16/bf16 goldens move.
- **Status:** TODO

### 9. Prefill tile-size tuning to the X3D topology — _medium–high (measure-first)_

- **What:** tune the prefill GEMM tile (rows × `m` block) so the activation tile
  stays resident in L1/L2 (48 KB / 1 MB) while weights stream; exploit the 128
  MB L3 for layer-resident activations + KV.
- **Why:** prefill is the cache-reuse regime; current tiling (8-row) is a fixed
  heuristic, not topology-aware.
- **Impact:** prefill win, hardware-dependent.
- **Precision:** bit-identical (scheduling/tiling only).
- **Gate:** needs `perf stat` (LLC / dTLB / backend-stall) to confirm we have
  cache-miss slack before investing. **`perf` is not installed on the dev box.**
- **Status:** TODO (blocked on measurement)

### 10. Op fusion (RMSNorm→Linear, gate/up, residual-add) — _high (structural)_

- **What:** fuse adjacent ops in the Graph/IR so intermediate activation vectors
  never round-trip to DRAM (stay in L1/registers). Some fusion exists
  (`GatedActFused`, `RmsNormAdd`); extend to norm→linear and residual chains.
- **Why:** the real "keep it in cache" lever in both regimes — cuts memory
  traffic, which is what helps when bandwidth-bound.
- **Impact:** moderate–large, broad.
- **Precision:** bit-identical if the fused ops compute the same values in the
  same order; verify per fusion.
- **Status:** TODO

---

## 测量（第 9 项的前置条件，且全程有用）

"Should help in theory" gets verified with counters, not intuition. `perf` is
**not installed** on the dev box; installing it (or an equivalent that reads
`LLC-load-misses`, `dTLB-load-misses`, `stalled-cycles-backend`) lets us
classify each stall as DRAM-bound (→ only fewer bytes helps), TLB-bound (→ #1),
or cache-miss slack (→ #9). For hotspot attribution use `samply` (never ad-hoc
timers); for A/B throughput use `infr bench --dev cpu` / `infr compare`.

## 软件预取 — 明确降低优先级

Explicit `_mm_prefetch` of weights is a micro-opt, not a strategy: the HW
prefetcher already predicts the sequential weight stream, and a mistuned
prefetch distance evicts useful lines. Only revisit if `perf` shows
latency-bound (not bandwidth-bound) stalls on an _irregular_ access pattern.
