---
kind: change-history
status: historical
scope: integrated-gpu
preserved_in_tag: release-0.9.0
---

# iGPU 支持：集成 GPU 正确性优化专项

> 历史专项记录。下文设备编号、驱动和阶段状态属于当时机器及测试窗口，不是 0.9.0 所有 iGPU 的通用配置。

在集成 GPU 上运行 infr 的优化专项日志（本机的 AMD Ryzen 9 9950X3D
iGPU / RADV RAPHAEL_MENDOCINO，以及通常所指的 Intel iGPU / Strix Halo 级 APU
目标类别）。内容从工作笔记汇总而来；保留了长期有效的发现和根因轨迹。已删去时点性的运行状态（机器卡死
告警、每次重启的操作步骤）。

## 状态：第 1 阶段（正确性）已完成

README 表中所有能放下的模型都能在 iGPU 上加载并生成连贯文本；seam 套件在**非 coopmat 层为 25/25**。UMA 支持已落地
（`4771cda`）。剩余 iGPU 工作**仅为第 2 阶段（性能）**，见文末。

- **第 1 阶段：正确性优先。** README 性能表中的每个模型都能在 iGPU 上运行：加载、生成连贯输出并通过测试。已完成。
- **第 2 阶段：性能。** 在 iGPU 上超越 llama.cpp Vulkan。尚未开始；唯一阻塞因素是优先级。

## 硬件（本机）

- `GPU1` = **AMD Ryzen 9 9950X3D 集成 GPU（RADV RAPHAEL_MENDOCINO）**，RDNA2，使用
  `--dev Vulkan1`。
- `GPU0` = RX 7900 XTX (RDNA3)：其余一切均针对这张独立显卡调优。

已知 iGPU 事实（`vulkaninfo`）；每一项都是本次调查需要证实或排除的潜在故障来源：

| 属性                         | iGPU (Vulkan1) | dGPU (Vulkan0) | 结论                                                                                      |
| ---------------------------- | -------------- | -------------- | ------------------------------------------------------------------------------------------ |
| `subgroupSize`               | 基准 64        | 基准 64        | 两者均为 `minSubgroupSize=32`；sg32 固定值正常，**不是问题**                               |
| 专用显存                     | ~2 GB carveout | 24 GB          | carveout 不是上限，模型位于 GTT（下文）                                                    |
| 系统 RAM                     | 共享 DDR       | —              | **关键杠杆**（UMA，见下文）                                                                |
| `maxMemoryAllocationSize`    | ~4 GiB         | ~4 GiB         | 相同，并非 iGPU 特有                                                                       |
| `maxStorageBufferRange`      | ~4 GiB         | ~4 GiB         | 超过 4 GiB 的 SSBO 可能静默读为 0；本次未观察到                                             |
| `maxComputeSharedMemorySize` | 64 KB          | 64 KB          | 相同                                                                                       |
| 协作矩阵                     | **缺失**       | 存在           | RDNA2 没有 coopmat（`f16cm:n i8cm:n`）→ 回退层；影响性能，不影响正确性                     |

## UMA 洞见（可靠的交付结果）

在 APU 上，“VRAM” carveout 不是真正的边界，它们是同一物理 DDR。host-visible/GTT 中的权重以与 carveout 相同的带宽读取。目标不是把模型挤进 2 GB，而是让放置逻辑不依赖 carveout，并让模型位于系统 RAM。

此 iGPU 上的堆表：

    heap[0]  10.73 GiB   (no flags)
    heap[1]  21.47 GiB   DEVICE_LOCAL      <- 所有 GpuOnly buffer 都放在这里
    type[0]  heap=1  DEVICE_LOCAL
    type[2]  heap=0  HOST_VISIBLE|HOST_COHERENT
    type[3]  heap=1  DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT

- `10.73 + 21.47 = 32.20 GiB` 恰好等于 `vram_total`（2 GiB carveout）+
  `gtt_total`（30.20 GiB）。**RADV 将“DEVICE_LOCAL”堆合成为（carveout + GTT）的约 2/3。**
  它既不是 VRAM 也不是 carveout，且**不受硬性限制**：从“21.47 GiB”堆分配的 41 GiB 全部成功，
  并落在 GTT 中。该设备上的 infr 权重本来就在系统 RAM 中。
- dGPU 表面上也有相同的双堆形态，但超额提交会经由 PCIe 溢出（真正的带宽悬崖）。**因此仅限 device-local 的防护对独显至关重要，必须保留。**这种不对称性正是整个设计。
- 仅扩大预算还不够：gpu-allocator 会将 `GpuOnly` 映射到第一个 DEVICE_LOCAL 类型，且不会回退，
  因此整个堆可能被静默超额订阅，直到提交因“Not enough memory for command submission”失败。
  溢出资源必须实际**放置**到其他内存类型（`probe_uma_overflow_type` + spill），不能只统计容量。

已在真实 iGPU（`--dev Vulkan1`）上验证：Gemma-4-31B UD-Q5_K_XL 可以加载（显示 UNIFIED MEMORY
横幅，32.20 GiB 预算，可放入 20.37 GiB 权重）并连贯 Decode。速度虽慢（2 CU 上 Prefill 约
3 t/s、Decode 约 0.4 t/s），但结果正确。

## 主导本专项的阻塞项：每次提交的看门狗

**GPU 挂起看门狗按每次提交启动，而 infr 将整个前向过程记录在一个命令缓冲区中。**在 2-CU iGPU 上，一个 Qwen3-8B 预填充块是单个任务约 2.05 秒的 GPU 工作，而设备会在约 2.06 秒终止任务，余量约 1%，因此会间歇性地随机挂起（15 次长运行约失败 3 次）。

诊断轨迹：

- 通过时间线关联锁定：内核在提交后 2.06 秒重置 ring；重置会强制信号 fence，因此 `vkQueueWaitIdle` 返回 _success_。被终止的块报告看似合理的约 2046 ms，进程仅在**下一次**提交时死亡。这就是故障从不指向自身的原因。`INFR_PROF_OPS` 时间戳确认整个窗口 GPU 始终繁忙：是任务过长，**不是**着色器挂起。
- **行数是错误的调节项。**提交时间随行数几乎不变（一次前向过程是 757 次 dispatch + 684 个 barrier + 一次权重扫描，均不会随行数缩小）：8→1163 ms、16→1199、32→1861、64→2047、128→2048。将行数削减 16 倍仅带来 1.76 倍收益。这就是早期 128 行预填充上限无法修复它的原因。

**修复（`0ea6600`，修正 `0c14f26`）：限制每次提交的 dispatch 数**，并将前向过程切为多个连续命令缓冲区（工作相同，使用 `finish_nowait`；看门狗只会看到短任务）。根据设备类别设定初值（**独显不设限，dGPU 路径不变**），再按每个前向过程实测的每 dispatch 成本重新调优，使其在未调查硬件上也成立。`INFR_SUBMIT_DISPATCHES` 可覆盖。运行器中链式解码（`replay_n`）由同一预算限制。**llama.cpp 也采用同样做法**（`max_nodes_per_submit`，加上按 CU 缩放的 FLOP 预算；这是更一般的边界，也是已记录的升级路径）。

证据：10/10 次长运行干净通过（此前 15 次约失败 3 次），gemma-3-12b 为 3/3，**dGPU 未变化**（pp512 3562.7→3547.1，tg128 143.5→143.1 = 噪声），gpu_seam 25/25。

**无效结果：已有证据证伪，请勿重新开启：**wave64 代码生成（`RADV_PERFTEST=cswave32` 仍 3/3 挂起）；VRAM 超额提交/驱逐（一次运行仅 0.48 MiB `amdgpu_bo_move`）；时钟降频（挂起期间 SCLK 固定 2200 MHz）；仅计算队列族（使其成为确定性问题，`comp` ring 预算更紧）。`RADV_DEBUG=hang` 是**红鲱鱼**：syncshaders 将 IB 串行化，故任务本身突破预算，只会“命名”2 秒时正在运行的着色器。

### 稠密解码路由错误（追查上述问题时发现）

一个分支曾将稠密单 token 解码路由到 `execute_static`（预填充/分页路径），以避开不可切分的 replay tape。这跳过了解码专用的 `record_decode_replay`（由 `_dyn` 参数驱动的位置内核 + 自推进 ring）⇒ 错误 logits ⇒ 重复 token 垃圾输出（`1.1.1…`、`))))))`）。使用 `INFR_SUBMIT_DISPATCHES=64` 可在 dGPU 上复现，gemma-4 **和** qwen3-8b 均受影响，因而并非 Gemma 特有，而是破坏所有稠密解码。正确修复是让**replay 路径本身可切分**（`RecordedCmd → Vec<RecordedSegment>`，分开提交）且保留解码内核；已随 UMA 改动落地（`4771cda`）。

## 调查：`--dev Vulkan1` 上的完整 README 表

提示词“法国的首都是哪里？请用一句简短的话回答。”，`INFR_MAX_NEW=40`。横幅：
`INTEGRATED (cu:2) — prefill chunk 128 rows, forward split every 128 dispatches`.

| #   | 模型                  | 量化       | 可加载 | 连贯     | 说明                                       |
| --- | --------------------- | ---------- | ----- | -------- | ------------------------------------------ |
| 1   | Gemma-4-26B-A4B (MoE) | UD-Q4_K_M  | 是    | 是       | 分页器 30/30 分页；tg 2.3 t/s              |
| 2   | Qwen3.6-27B           | Q4_K_M     | 是    | 是       | 稠密常驻；tg 1.0 t/s（最慢）                |
| 3   | Qwen3-30B-A3B (MoE)   | Q4_K_M     | 是    | 是       | 分页器 48/48；ctx 限制 40960→24893         |
| 4   | Gemma-4-31B           | UD-Q5_K_XL | 是¹   | 是¹      | ¹使用 UMA 改动；预 UMA 时 VRAM 防护阻止加载 |
| 5   | Ornith-1.0-35B        | Q4_K_M     | 是    | 是       | DeltaNet；tg 2.2 t/s                       |
| 6   | Qwen3.6-35B-A3B (MoE) | UD-IQ3_S   | 是    | 是       | 常驻 12.68 GiB；tg 3.7 t/s                 |
| 7   | Qwen3.6-35B-A3B (MoE) | UD-Q4_K_M  | 是    | 是       | tg 1.8 t/s                                 |
| 8   | DiffusionGemma-26B    | Q4_K_M     | 是    | 是²      | ²DG seam 测试修复后（`feb61b5`）            |
| 9   | Ternary-Bonsai-1.7B   | Q2_0_g64   | yes   | yes      |                                            |
| 10  | Ternary-Bonsai-4B     | Q2_0_g64   | yes   | yes      |                                            |
| 11  | Ternary-Bonsai-8B     | Q2_0_g64   | yes   | yes      |                                            |
| 12  | Ternary-Bonsai-1.7B   | 原始 Q2_0  | 否    | —        | 加载器；与设备无关的已知限制               |
| 13  | Ternary-Bonsai-1.7B   | PQ2_0      | 否    | —        | `unsupported: ggml type 142`（未实现）     |

### 已解决的故障类别

- **DiffusionGemma non-coopmat (`feb61b5`) — was an over-strict TEST, not a
  kernel bug.** The DG prefill last-token top-5 overlap assert tripped on a
  near-tie (whole-vocab cosine 0.80 nc / 0.811 coopmat — textbook int8<f16<f32
  laddering). Fix: gate on the distribution (`overlap || cos > 0.78`, keep hard
  `cos > 0.7`), mirroring the sibling `_denoise` check; tighten `_denoise` floor
  0.7→0.75 (measured healthy min 0.789). gpu_seam now 25/25 on BOTH tiers.
- **`--dev` was decorative (`c05b526`).** `let _ = dev;` — the backend hardcoded
  "first discrete GPU", so `infr bench --dev Vulkan1` silently benched the dGPU.
  Now real, with a hard error on an unknown device.
- **Vacuous-green test harness (`0a353c6`).** Seam tests reached the GPU via
  `gpu_available()` → `VulkanBackend::new().is_ok()`, so a failing device made
  them SKIP SILENTLY and report "passed" (`INFR_DEV=Vulkan9` → "1 passed" in
  0.02 s). Now a hard failure when `INFR_DEV` is set explicitly.

### 与设备无关，而非 iGPU 错误

- Bonsai plain `Q2_0` (34B/128 elems) vs `Q2_0_g64` (18B/64) declare the
  **identical `ggml type 42` with no metadata discriminator**, so upstream made
  them indistinguishable — infr hardcodes g64 (`infr-gguf/src/lib.rs`). README
  pins `:Q2_0_g64`. `PQ2_0` is ggml type 142, not implemented.
- Qwen3-0.6B-Q2_K degenerates into repetition on the dGPU too = model quality.

## 运维规则（付出代价后得出的结论）

**NEVER let a `timeout`/SIGTERM kill `infr` mid-GPU-submit on the iGPU.** It
leaves an unkillable D-state task in `dma_fence_wait` holding the device, and
RADV then drops the iGPU from enumeration until reboot
(`no such Vulkan device`). Give iGPU runs a generous `timeout` (they are 30-60×
slower than the dGPU — a long prompt + decode is minutes) and never wrap them in
a tool call that can time out first. GPU runs are **serial**; on UMA, two big
models concurrently = GTT exhaustion = wedge.

## 第 2 阶段（性能）：剩余项及顺序

1. **No-coopmat prefill tier is the whole gap** (pp 11-200 tok/s vs thousands on
   the dGPU). RDNA2 has no cooperative matrix; prefill falls to the
   scalar/f16-warp ladder. This is the dominant Phase-2 lever.
2. **UMA placement follow-ups:** `unified_memory` was hardcoded `false` in the
   Vulkan caps; the host→staging-ring→device copy is a pointless DDR→DDR pass on
   an APU; the VRAM guard should treat GTT as evictable system RAM. (Landed in
   the UMA slice; re-audit for headroom.)
3. **Revisit `INFR_UBATCH` on the iGPU.** Rows barely affect submit time now
   that submits are duration-bounded, so the 128-row prefill cap is costing
   throughput — do the FLOP-aware budget (llama.cpp's shape) before raising it.
4. Class-3 wart: default-ctx clamp collapsing to 1024 on many models.
