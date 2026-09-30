# 性能审查 — Vulkan 后端，多厂商（Intel / NVIDIA / AMD）

2026-07-31

> **部分内容已被取代，请先阅读（注于 2026-08-11）。** 发现 #2
> （“厂商检测应改为能力检测”）以及基于它的架构说明和表格，描述了 `Capabilities` 上的
> `vendor_intel` 标志与四项按厂商路由的决策。**该标志已不存在**：`vendor_intel` 在
> `crates/` 中已无任何出现位置，`Capabilities` 没有厂商字段，且 `adapter.rs` 中
> `unified_mmv_row1` 的注释将其移除记录为有意为之（“新硬件不需要此处的厂商特例”）。
> 该建议已落实；请把 #2、“架构说明”章节和“统一默认值的预期形态”表视为当时决策的记录，
> 而不是当前代码树的描述。
>
> 本文件中的每个行号均早于该变更及若干其他切片，因此请按符号重新定位，而不要信任行号。
>
> 2026-08-11 的硬件能力审计针对当前 `HEAD` 重新推导检测清单，发现本审查未考虑的一点：
> 能力优先设计信任设备枚举的内容，而 llama.cpp 记录了两个错误报告协作矩阵支持的驱动。其余内容见
> `backlog.md` § B-HWDET-DRIVERID 以及 § B-HWDET-LIMITS / § B-HWDET-I8CM-FRAGLAYOUT。

## 范围

Vulkan 实现（`crates/infr-vulkan/`），重点考察 Intel Arc（ANV）、NVIDIA（专有驱动 / NVK）
与 AMD（RADV）之间不同的按厂商内核路由、GEMM/闪存注意力调度和功能门控决策。

覆盖内容：适配器路由（`adapter.rs`）、能力探测（`lib.rs`）、记录器调度热点路径（`recorder.rs`）、
GEMM 内核解析（`gemm.rs`）、着色器构建矩阵（`build.rs`）和管线缓存（`pcache.rs`）。

本轮未覆盖（不在范围内）：CPU 后端、Metal 后端、GGUF 反量化内部实现、模型图编译、主机端
seam/runner，以及非 Vulkan 的配置/分析基础设施。任何需要通过分析确认的发现均标为
**需要测量**。

---

## 发现（按预估影响排序）

### 1. Intel Arc XMX coopmat 受显式启用门控，默认使用 nc_mmq/nc_fma/nc_fa 非 coopmat 层级

- `crates/infr-vulkan/src/adapter.rs:1470`（`cm8_ok` 门控）
- `crates/infr-vulkan/src/lib.rs:1596`（显式启用设计注释）
- `crates/infr-vulkan/src/lib.rs:3499` (`select_coopmat_shape` — 8×8×16 only
  位于 `allow_8x8x16` 下）

**现象：** Intel Arc A770（Mesa ANV）仅枚举 8×8×16 形状的 f16 协作矩阵，而非生产用的
16×16×16。除非设置 `INFR_CM_8X8=1`，`select_coopmat_shape` 的偏好阶梯会为 8×8×16
返回 `None`。未设置时，`caps.f16_coopmat()` 为 `false`，适配器会将全部预填充 GEMM
路由至非 coopmat 层级：

- `nc_mmq`：用于量化权重（k-quants、Q8_0）的 dp4a `matmul_mmq`
- `nc_fma`：用于 f16/bf16/f32 权重的共享内存 fma `matmul_fma`，无子组操作、无 f16 ALU
  （`native_gemm_fma.comp:2899`）
- `nc_fa`：共享内存 fma 闪存注意力（`attn_nc_fa.comp`），固定 bm=32 磁贴、无子组操作、
  共享内存不超过 54 KB

`adapter.rs:1483` 的注释称，在此层级出现前预填充存在“现场测得的 10-30 倍”差距（回退路径是
逐行标量 GEMV）。nc 层级缩小了相对于标量的差距，但 XMX coopmat 路径（`native_gemm_warp` 的
`_cm8` 构建）才是真正的 Intel 张量核心路径，且仍然**仅可显式启用**。

**为何是热点路径：** Intel Arc 上的每次预填充前向传播都会经过它。未设置 `INFR_CM_8X8=1` 的
A770 用户在运行 Qwen3-8B 预填充时，承担的是 dp4a/fma GEMM 成本而非 XMX 张量核心成本。
infr 从未测量 Intel 硬件上的 nc_mmq 与 cm8 warp GEMM 差距，显式启用门控阻止了这项测量。

**为何受门控：** 注释引用“Alchemist coopmat 是 llama.cpp 所记录的回归”（adapter.rs:3501）。
但 Mesa ANV 已在 Mesa 24.0 合入 `VK_KHR_cooperative_matrix` 支持（2024 年第一季度；参见
[Phoronix](https://www.phoronix.com/news/Intel-ANV-Cooperative-Matrix)）。该回归可能已过时：
在当前 Mesa ANV（至少 24.2，本项目推荐的最低 Mesa 版本）上重新测量，可能使其改为默认开启。

**修复：** 在 Mesa ≥24.2 的 Intel Arc A770 上测量 `INFR_CM_8X8=1`：

```bash
INFR_CM_8X8=1 infr bench model.gguf -p 512 -n 0 -r 3
```

若 pp512 吞吐以可测量幅度胜过 nc_mmq 默认值，且一致性测试套件通过（GPU 门控测试使用
`cargo test -p infr-vulkan --release -- --ignored`），则改变默认值：Mesa 版本达到已知良好
发行版时移除 Intel 的显式启用门控，或将其设为默认开启并提供 `INFR_NO_CM_8X8` 退出开关。
`native_gemm_warp_cm8_build_spv` 函数已以 `n%128 && k%64` 为门控，无法覆盖的形状仍会回退至
nc_mmq/nc_fma，因此开启它是增量变更，并非全量切换。

**风险：** 若原始 llama.cpp 回归仍可在当前 Mesa 上复现，则保持显式启用。调查成本是一次 A770
基准测试会话。

---

### 2. 厂商检测应改为能力检测：`vendor_intel` 门控的是策略而非硬件

- `crates/infr-vulkan/src/lib.rs:1928`（`vendor_intel` 探测）
- `crates/infr-vulkan/src/adapter.rs:531`（每厂商的 dtype 集合）
- `crates/infr-vulkan/src/adapter.rs:726`（每厂商的内核路由）
- `crates/infr-vulkan/src/adapter.rs:749`（每厂商的 WARPS 默认值）

**现象：** `vendor_intel`（由 `vendor_id == 0x8086` 探测）驱动四项决策，它们的能力可替代性各不相同：

| 使用位置 | 门控内容 | 能否由能力替代？ |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lib.rs:1935` — `sg_pref` 默认值 | Intel 为 16，其他为 32 | **可以。** `subgroup_min <= 16` 等价：能固定 16 且最小值不超过 16 的设备取 16；wave32 设备（min=32）本就无法固定 16，回退为 32。从条件中移除 `vendor_intel &&`。 |
| `adapter.rs:531` — 解码 int8 dtype 集合 | Intel：{Q4K,Q6K,Q2K,Q3K}。AMD：{Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} | **部分可以。** Intel 集合排除了旧式 32-block dtype（Q4_0、Q5_0、Q5_1、IQ4_NL），原因仅是它们从未在 Intel 上测量；内核存在，只是未做基准测试。AMD 集合因混合 GGUF 的一致性断崖而排除 Q3_K（`gpu_seam_matches_cpu_qwen3_q2k` 失败）。能力无法表达“Q3_K 会在该 GPU 的内存/缓存层级上出错”。AMD 集合作为统一默认值（即安全交集）可在各处工作；`INFR_MMV_MW=1` 已为 A/B 启用所有 dtype。 |
| `adapter.rs:726` — `unified_mmv_row1` | `!vendor_intel`：Intel 使用旧 `native_mmv_mw.comp`，AMD 使用统一 `native_mmv_mrow.comp` rows=1 | **可以。** 所有 dtype 的内核均存在于所有厂商。该分支仅因统一后未重新测量 Intel 而存在。单一统一路径可工作，内核按构造逐位一致。 |
| `adapter.rs:749` — mmv WARPS 默认值 | Intel/所有非 Q4K 为 8，AMD Q4_K 为 1（扫描胜者） | **部分可以。** Q4_K 的 warps=1 形状只在 AMD 上经扫描调优。Intel 的 warps=8 是“已发布、已调优”的形状。可在所有位置将 Q4_K 默认设为 warps=1（llama.cpp 的 `rm_kq_int=1` 形状，即每个工作组一行输出），但没有 Intel 测量时这是盲改。 |

**架构原则：** 硬件所*声明*的每项决策——协作矩阵形状、共享内存预算、子组范围、缓冲区设备地址、
shaderFloat16——均已按能力门控。上方四处 `vendor_intel` 使用是仅有的按厂商身份决定的地方，其中三处
可由能力或统一默认值替代。

**修复（优先顺序）：**

1. **以 `subgroup_min` 替换 `sg_pref` 厂商门控**（简单，不需测量）：从 `lib.rs:1935` 删除
   `vendor_intel &&`。若 `subgroup_min <= 16` 且可固定 16，则默认取 16。这是纯能力决策：
   未来配备 SIMD8/SIMD16 EU 的非 Intel GPU 会自动获得正确默认值。

2. **令 `unified_mmv_row1` 无条件启用**（低风险）：统一内核适用于每个 dtype，按构造逐位一致，
   且已由 `mmv_row1_bit_identical` 证明。排除 Intel 的唯一原因是“此验证环境没有 Intel GPU”。
   将其开启；若 Intel 解码吞吐回归（可能性低，相同数学、不同内核），已有 `INFR_MMV_MW=0` 退出开关。

3. **将解码 int8 dtype 集合统一为 AMD 默认值**（安全交集）：`&[Q2K, Q4K, Q6K, Q4_0, Q5_0, Q5_1, Iq4Nl]`。
   Intel 失去 Q3_K（未测量但该处默认启用的收益），获得 Q4_0/Q5_0/Q5_1/IQ4_NL（Intel 未测量、
   AMD 已测得收益）。全部均可由 `INFR_MMV_MW=1` / `INFR_MMV_MW=0` 覆盖。AMD 上 Q3_K 的
   一致性断崖正是其不能进入统一集合的原因。

4. **移除 WARPS 厂商分支**：在所有位置为 Q4_K 默认设为 `1`（llama.cpp 的 `rm_kq_int=1` 形状），
   其他 dtype 为 `8`。不再由厂商决定，而是由 dtype 决定。可由 `INFR_MMV_MW_WARPS` 覆盖。

   最终状态：从 `Capabilities` 完全移除 `vendor_intel`。`sg_pref` 由能力驱动；解码策略是一张
   统一的逐 dtype 表；内核路由无条件执行。新硬件无需新厂商标志，能力和 `INFR_*` 开关覆盖测量需求。

---

### 3. NVIDIA 闪存注意力的磁贴尺寸为 AMD 的一半：占用率与磁贴效率的权衡

- `crates/infr-vulkan/src/recorder.rs:4426-4442` (bm=64 → bm=32 for sub-64 KB
  共享内存）
- `crates/infr-vulkan/src/recorder.rs:4651-4658` (BR=128 → BR=64 for sub-64 KB
  共享内存）

**现象：** NVIDIA GPU 暴露 `maxComputeSharedMemorySize = 48 KB`（RADV 为 64 KB）。
闪存 warp 内核的 bm=64 磁贴需要 58112 B（约 57 KB），所以 NVIDIA 回退至 bm=32（29056 B）。
寄存器 O 内核的 BR=128 磁贴需要 58880 B，因此 NVIDIA 回退至 BR=64（29440 B）。

较小磁贴意味着相同行数下 **2 倍工作组**。工作组彼此独立（split-K reduce 无跨 WG 同步），因此额外的
WG 能更充分填满 GPU；但每个 WG 只做一半工作，每 WG 开销（屏障、共享内存初始化）也会支付两次。

**为何重要：** 在 512 行 × 16 头 × 深 KV 的预填充中，warp 路径在 AMD 上发射
`(512/64)*16 = 128` 个 WG，在 NVIDIA 上发射 `(512/32)*16 = 256` 个 WG。256 个 WG 很可能
能充分填满 RTX 4090 的 128 个 SM；但在较小 NVIDIA GPU（RTX 4060，24 个 SM）上，256 个 WG 意味着
约 10 个 wave，仍无问题。真正的问题是每 WG 开销是否主导了计算。

**能否修复？** 共享内存预算是硬件常数，内核磁贴必须容纳其中。选项如下：

1. **不同的内核设计：** 使用较少共享内存暂存的寄存器密集变体（例如以更小分块流式传输 K/V、
   重新计算 Q 磁贴）。这是第 6 类（内核微架构），构建和测量成本高。
2. **接受该磁贴：** bm=32 已针对 NVIDIA 测量并发布。`limits_probe.rs` 示例正是为调试此项而存在。

**建议：****需要测量。** 在 NVIDIA GPU 上用 `INFR_PROF_OPS=1` 分析 pp512 注意力时间。若注意力
少于预填充墙钟时间的 20%，磁贴差距不是切入点。若超过 40%，且 bm=32 的 WG 数量明显未填满 SM 阵列，
则应开始界定寄存器 O 重设计。

---

### 4. Intel `nc_fa` 闪存注意力使用固定 bm=32 磁贴（无更大构建）

- `crates/infr-vulkan/src/gemm.rs:1008-1024` (`attn_nc_fa_spv` — fixed bm)
- `crates/infr-vulkan/src/recorder.rs:4579-4623` (`attention_prefill_nc_fa`)

**现象：** `attn_nc_fa` 内核（nc_mmq/nc_fma 的非 coopmat 闪存注意力配套内核）在 hd≤256 时
使用硬编码 `bm=32` 磁贴，在 hd≤512 时使用 `bm=16`。与同时具有 bm=64 和 bm=32 变体的 coopmat
flash-warp 内核不同，它没有 bm=64 构建。

这意味着即使 Intel Arc 具有至少 64 KB 共享内存（Alchemist 与 RADV 一样具有 64 KB），nc*fa 内核
也无法使用更大磁贴。coopmat flash-warp 内核*可以*在 Intel 上使用 bm=64，但仅在 `INFR_CM_8X8=1`
下，该设置也会门控 GEMM 层级。

**修复：** 若 Intel cm8 coopmat 保持显式启用（发现 #1），为 `attn_nc_fa` 新增 bm=64 构建
（以 `max_shared_memory >= 64*FLASH_SHARED_PER_ROW` 门控）可让 Intel Arc 预填充注意力使用更大磁贴。
`recorder.rs:4439` 现有磁贴选择逻辑已可用，只需 nc_fa 内核的 bm=64 SPIR-V 构建。**成本：** 一个新
着色器变体、一项新 `build.rs` 条目和约 30 行 `gemm.rs`。

---

### 5. `with_padded_dst` 为每个非 Internal GEMM 输出分配临时缓冲区

- `crates/infr-vulkan/src/adapter.rs:1044`

**现象：** 每个写入非 Internal 张量（例如 lm_head `logits` Output）的分块 GEMM（coopmat、
nc*mmq、nc_fma）均通过 `be\*.alloc_uninit` 分配一个 `ceil(m/64)*64` 行临时缓冲区，填充后再复制
回 `m` 个实际行。lm_head 路径每次前向传播发生一次该分配；中间层产生 Internal 张量并跳过复制。

**影响：** 低。lm_head 是前向传播末尾附近的一个操作，分配大小为 `vocab_size * n_embd * dtype_bytes`
的缓冲区。对于 Qwen3-8B（vocab 152064、n_embd 4096、f32 输出），约为 2.5 GB；但
`with_padded_dst` 填充至 `ceil(152064/64)*64 = 152064` 行（恰为整数倍，无填充），因此临时区恰为
输出大小。该分配与输出缓冲区本身规模相当，且每次前向传播只发生一次。

**修复：** 可在 `ScratchPool` 中按形状池化该缓冲区（增加类似 `"lin_pad_dst"` 的标签）。但 lm_head
输出在预填充（m>1）和解码（m=1）之间改变形状，且该池本就是每次执行独立的。每次前向仅一次分配，
不值得引入此复杂度。

---

### 6. 每次 execute_static 调用均重建 `rope_pos` HashMap

- `crates/infr-vulkan/src/adapter.rs:4666-4677`

**现象：** 在操作循环之前，`execute_static` 扫描所有图操作，以构建“位置张量 → rope 位置”的
`HashMap<u32, usize>`。复杂度为 O(ops)，每个唯一位置张量调用一次 `read_pos0`（每次前向通常为 1 次）。

**影响：** 可忽略。操作数小于 1000，扫描每次前向仅发生一次，解码重放路径
（`execute` → `record_decode_replay`）完全跳过它。`read_pos0` 调用只从主机可见缓冲区读取一个 u32，
无需提交/等待。

---

### 7. 链式解码 `replay_n` 每条链分配 `vec![seg.cmd; n]`

- `crates/infr-vulkan/src/recorder.rs:9103`

**现象：** `execute_chain` 在一次提交中重放 `n` 份已记录解码命令缓冲区。它构建
`vec![seg.cmd; n]`，即一个包含 `n` 个 `vk::CommandBuffer` 句柄的 `Vec`。

**影响：** 可忽略。`n ≤ 64`（受 `max_decode_chain` 限制），因此分配不超过 512 字节。链式解码是热点
路径（每个 token 批次），但此分配远小于其提交的 GPU 工作量。

---

## Architectural note: vendor detection vs capability detection

`vendor_intel` (`caps.vendor_intel`, probed from `vendor_id == 0x8086`) is used
in four places. None of them gate whether a kernel _can_ run — they gate which
kernel _should_ run by default, based on empirical throughput measurements on
specific hardware.

The principle: capability detection answers "can this GPU run this kernel?"
Vendor detection answers "should this GPU run this kernel by default?" The
second question is better answered by a single unified default table with
`INFR_*` knobs for per-dtype/per-shape A/B measurement on any hardware.

**Why capability-first is more robust:**

- A future non-Intel SIMD8 GPU (e.g., a new ARM Mali with subgroup_min=8) would
  get `sg_pref=32` under vendor detection (not Intel → 32), but should get 16.
  Under `subgroup_min <= 16` it gets 16 automatically.
- A future NVIDIA GPU with 64 KB shared memory would get bm=32 flash tiles under
  vendor detection (if we keyed bm on vendor), but under
  `max_shared_memory_bytes >= 58112` it gets bm=64 automatically — which is what
  the existing flash tile selection already does.
- A future Intel dGPU with subgroup_min=32 would get `sg_pref=16` under vendor
  detection (Intel → 16), which is wrong. Under `subgroup_min <= 16` it gets 32
  automatically.

**What stays vendor-agnostic (already capability-driven):**

- Cooperative matrix shape selection (`select_coopmat_shape`): enumerates device
  configs, picks 16×16×16 if present, else 8×8×16 if opt-in.
- Flash attention tile size: `max_shared_memory_bytes` → bm=64 vs bm=32.
- ShaderFloat16, shaderInt8, subgroup-size-control: all probed from device
  features.
- Pipeline cache: keyed per `(vendor_id, device_id)` — but this is a disk cache
  namespace, not a kernel routing decision, and the Vulkan driver's own
  `pipelineCacheUUID` already encodes device identity.

**What the unified defaults would look like:**

| Decision              | Current (vendor-split)                                             | Proposed (capability)                                               |
| --------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| `sg_pref`             | `vendor_intel && subgroup_min <= 16` → 16, else 32                 | `subgroup_min <= 16` → 16, else 32                                  |
| `unified_mmv_row1`    | `!vendor_intel`                                                    | `true` (unconditional)                                              |
| Decode int8 dtype set | Intel: {Q4K,Q6K,Q2K,Q3K}. AMD: {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} | {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} (AMD's set = safe intersection) |
| mmv WARPS default     | Intel/all-non-Q4K → 8, AMD Q4_K → 1                                | Q4_K → 1, everything else → 8                                       |

All four are overrideable via `INFR_SG`, `INFR_MMV_MW`, `INFR_MMV_MW_WARPS`.

---

## Coverage

**Traced paths (verified call-frequency):**

| Path                                                         | Frequency                    | Traced? |
| ------------------------------------------------------------ | ---------------------------- | ------- |
| Decode record-once replay (`execute` → `replay`)             | Per token                    | ✓       |
| Chained decode (`execute_chain` → `replay_n`)                | Per n-token batch            | ✓       |
| Static prefill (`execute_static` → `lower_op` loop)          | Per forward                  | ✓       |
| Coopmat GEMM routing (`is_gemm`, warp_ok, split-K)           | Per Linear op in prefill     | ✓       |
| Non-coopmat GEMM routing (`nc_mmq`, `nc_fma`)                | Per Linear on Intel/!coopmat | ✓       |
| Flash attention routing (`flash_ok`, `nonfa_ok`, `nc_fa_ok`) | Per Attention op             | ✓       |
| Decode GEMV routing (`mmv_mw_choice`, `unified_mmv_row1`)    | Per token × Linear           | ✓       |
| Capabilities probe (`VulkanBackend::new`)                    | Once at init                 | ✓       |
| Pipeline cache (`pcache.rs`)                                 | Once at init + periodic      | ✓       |

**Not traced (known gaps):**

- **MoE dispatch overhead** (paged expert GEMMs, router/top-k readback): The
  adapter's `MoeFfn` handling splits into paged vs batched paths. Not traced in
  this pass — the perf.md campaign log already covers the batched-MoE
  dispatch-collapse win (class 4, pp512 0.59→0.91×).

- **Dense layer streaming** (`streamed_prefill_gemm`): The adapter's
  `INFR_DENSE_PAGE` path for models too large to fit in VRAM. Not traced.

- **Canvas/DiffusionGemma**: The `AttnMask::Canvas` path in attention routing.
  Not traced — perf.md notes it's benchmarked differently (dg-step vs pp/tg).

- **Metal parity**: Linux builds compile Metal sources blind. Any type changes
  in shared types (`Capabilities` fields, `Op` variants) only get verified on
  macOS CI.

**Needs measurement (cannot settle without profiling):**

1. Intel Arc `INFR_CM_8X8=1` vs nc_mmq prefill throughput (finding #1).
2. NVIDIA flash attention bm=32 vs theoretical bm=64 wall-time share (finding
   #3).
3. Impact of unifying decode int8 dtype defaults on Intel (Q3_K loss,
   Q4_0/Q5_0/Q5_1/IQ4_NL gain) and on NVIDIA (currently inherits AMD's set —
   same as the proposed unified default) (finding #2).
4. `nc_fa` bm=64 build benefit on Intel Arc (finding #4).
5. The `nc_mmq` vs `nc_fma` throughput ratio on Intel Arc — are both arms tuned?

---

## Multi-vendor sanity checks

### Intel Arc (ANV)

| Check                                                  | Status                                                                               |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `sg_pref = 16` (SIMD8 EUs)                             | ✓ (`vendor_intel && subgroup_min <= 16` — replaceable by `subgroup_min <= 16` alone) |
| `sg_pref` falls back to 32 when 16 unpinnable          | ✓                                                                                    |
| Decode GEMV: `native_mmv_mw.comp` (SG=16, WARPS-tuned) | ✓ (`unified_mmv_row1 = false` for Intel — replaceable by unconditional `true`)       |
| Decode int8 dtypes: {Q4K,Q6K,Q2K,Q3K}                  | ✓ (`mmv_int8_decode_dtypes` Intel arm — would change under unified default)          |
| Coopmat: `f16_coopmat()` = false (8×8×16 only)         | ✓ (correct; cm8 is opt-in)                                                           |
| Non-coopmat GEMM tier: `nc_mmq` + `nc_fma`             | ✓ (default — see finding #1)                                                         |
| Non-coopmat flash: `nc_fa` (`attn_nc_fa.comp`)         | ✓                                                                                    |
| Pipeline cache: keyed per `(vendor_id, device_id)`     | ✓                                                                                    |

### NVIDIA (proprietary / NVK)

| Check                                                                  | Status                                          |
| ---------------------------------------------------------------------- | ----------------------------------------------- |
| `sg_pref = 32` (warp = 32)                                             | ✓ (capability: `subgroup_min > 16` → 32)        |
| Decode GEMV: `unified_mmv_row1` (mrow, bit-identical)                  | ✓                                               |
| Decode int8 dtypes: inherits AMD's {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} | ⚠ Unmeasured — same as proposed unified default |
| Coopmat: `f16_coopmat()` = true (16×16×16)                             | ✓ (Turing+)                                     |
| Flash bm=32 (48 KB shared → 29056 B fits, 58112 B doesn't)             | ✓ (capability: `max_shared_memory_bytes`)       |
| Flash BR=64 (48 KB shared → 29440 B fits, 58880 B doesn't)             | ✓ (capability: `max_shared_memory_bytes`)       |
| Flash warp path available                                              | ✓ (bm=32 build exists; `recorder.rs:4448`)      |
| Shared memory tile selection: `max_shared_memory_bytes()` check        | ✓                                               |
| Pipeline cache: keyed per `(vendor_id, device_id)`                     | ✓                                               |

### AMD (RADV)

| Check                                                   | Status                                    |
| ------------------------------------------------------- | ----------------------------------------- |
| `sg_pref = 32` (wave32)                                 | ✓ (capability: `subgroup_min == 32` → 32) |
| Decode GEMV: `unified_mmv_row1` (mrow, bit-identical)   | ✓                                         |
| Decode int8 dtypes: {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} | ✓ (measured on 7900 XTX)                  |
| Coopmat: `f16_coopmat()` = true (16×16×16)              | ✓ (RDNA3+)                                |
| Flash bm=64 (64 KB shared → 58112 B fits)               | ✓ (capability: `max_shared_memory_bytes`) |
| Flash BR=128 (64 KB shared → 58880 B fits)              | ✓ (capability: `max_shared_memory_bytes`) |
| Flash warp path available (bm=64 + bm=32 builds)        | ✓                                         |
| Compute unit count probe (shader engine count)          | ✓ (`VK_AMD_shader_core_properties`)       |
| Integrated GPU detection + chunk sizing                 | ✓                                         |
| Pipeline cache: keyed per `(vendor_id, device_id)`      | ✓                                         |

---

## Summary

**Highest ROI, ready to act:**

1. **Measure Intel Arc `INFR_CM_8X8=1` on current Mesa ANV.** If the
   llama.cpp-documented regression is fixed in Mesa ≥24.2, flipping this to
   default-on gives Intel Arc users the XMX tensor-core GEMM path — the single
   largest lever for Intel prefill throughput.

2. **Remove `vendor_intel` in favor of capability detection.** Four-step plan:
   (a) replace `sg_pref` vendor gate with `subgroup_min <= 16`; (b) make
   `unified_mmv_row1` unconditional; (c) unify the decode int8 dtype set to
   AMD's safe default; (d) drop the WARPS vendor split. End state: zero vendor
   flags in `Capabilities`, every routing decision keyed off capabilities the
   device declares, `INFR_*` knobs for A/B measurement on any hardware. New GPUs
   need no new vendor flags.

3. **Profile NVIDIA flash attention tile overhead.** If bm=32 attention is a
   significant share of NVIDIA prefill wall time, a register-O redesign that
   fits in 48 KB shared could be the next class-6 lever. Until profiled, accept
   the current bm=32 as correct.

**Lower priority, but worth tracking:**

4. Add bm=64 build of `attn_nc_fa` for Intel Arc (if cm8 stays opt-in).
5. Measure the unified decode int8 dtype defaults on Intel and NVIDIA. Requires
   access to those GPUs.
