# rocm-plan.md — native ROCm/HIP backend: Vulkan feature-parity tracker

The `infr-rocm` backend is **built and correct**: every supported model and
quant format runs coherently on AMD GPUs via ROCm/HIP, gated token-for-token
against the CPU reference (Part A complete). It also has real perf (native/int8
decode, WMMA prefill, split-KV attention, MoE fast paths, decode fusion) and all
three Vulkan-style memory-paging modes.

This plan now tracks **reaching feature parity with the Vulkan backend** —
infr's most heavily tuned GPU path, which is competitive with (and sometimes
beats) llama.cpp HIP on the same silicon. The goal, in one line:

> **Every fast-kernel path Vulkan has, for every supported model × quant; the
> same memory handling; and the same multi-GPU support — ported to ROCm wherever
> it measurably benefits ROCm.**

The "wherever it benefits" caveat is load-bearing: RDNA3/HIP specifics make a
few Vulkan tricks moot on ROCm (see "Tricks that may not port" below) — each
lever is adopted only if it measurably wins, verified against `llama.cpp` HIP
(`~/Projects/mxaddict/llama.cpp/build-hip/bin/llama-bench`,
`-sm none -mg 0 -fa 1`). The process/perf discipline is `docs/perf.md`
(exclusive device access, profile-first, biggest-gap-first, goldens never move
except sanctioned flips).

Architecture reference (how the backend is wired — the `Backend` seam, the
`INFR_DEV=rocm` path, kernel authoring, the `be.name()` gates) is in the code:
`crates/infr-rocm/src/{backend,exec,kernels,pager,weight_pager,ffi}.rs`, wired
through `crates/infr-llama/src/{chat/rocm.rs,seam/}`. The cross-backend seam
(`Backend` trait, `Op`/`Graph` IR, `dequant_block`, `iquant_grids`, the LRU
`Pager`) is in `infr-core`/`infr-gguf` and shared by all backends.

## Delivered so far (do not redo)

- **Part A correctness** — all archs (llama, qwen2/3, gemma3/4/E2B, qwen35
  DeltaNet, qwen3moe, BitNet i2_s) coherent; all 24 weight quant formats
  parity-validated; a `rocm_seam` gpu_seam gate (9 models, token-for-token/hash
  vs CPU). Goldens locked.
- **Decode** — native in-kernel block decode + int8-activation dp4a GEMV for
  **Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1**; grid-underfill fixed
  across attention/WriteKv/GatedAct/RmsNorm/Argmax/QkNormRope; RmsNorm→Linear +
  Linear→Add fusion. ~1.9 → ~130 t/s (Qwen3-0.6B Q4_K_M, ~60× over naive, ~0.3×
  llama.cpp).
- **Prefill** — int8 **WMMA** matrix-core GEMM (RM×CN register tiling +
  software-pipelined Q4_K); ~4500 t/s (~0.2× llama.cpp).
- **Attention** — split-KV / flash-decoding (10.6× at depth), Causal/SWA/Canvas.
- **DeltaNet** — chunked/parallel prefill (88×) + column-parallel decode.
- **MoE** — int8 dp4a experts ({q80,q2k,q3k,q4k,q5k,q6k,q40,q41,q51}, gate/up ×
  down) + **GPU-side top-k routing** (device-routed, no host readback).
- **Memory paging** — all three Vulkan modes: MoE expert LRU cache (host→VRAM,
  copy-stream overlap), KV-cache overflow to host (`INFR_KV_OVERFLOW`),
  dense-weight prefetch ring. 30B MoE, 27 GiB KV contexts, and >VRAM dense
  models run on a 24 GB card.

---

# The parity roadmap

Each item: the gap (Vulkan has it, ROCm lacks it), the Vulkan reference, the
ROCm task. Ordered roughly by value. Every kernel lands with a parity test vs
the CPU reference before its capability flag flips (the rule that kept the blind
kernels correct).

## 1. Quant coverage — fast kernels for ALL 24 formats × all paths

**The biggest correctness-of-perf gap.** ROCm has native/int8 fast kernels for
**10** formats (Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/**Q4_0**/**Q4_1**/Q5_0/**Q5_1**);
every other quant falls back to the slow `dequant→f16` GEMV (256 threads — the
pathology that made gemma-3's Q5_0 0.04× before it was covered). Vulkan is
native on **all 24** + floats, with a full `native_id`/`native_idm` MoE-GEMV
family (`crates/infr-vulkan/src/linear.rs:136-254`, `gemm.rs`).

- ✅ **R1 — Q5_K LANDED.** `deq_q5k` native decode (+ `linear_q5k`/`embed_q5k`/
  `deqf16_q5k`), `linear_i8_q5k`/`i8acc_q5k` int8 dp4a GEMV, the
  `wmma_i8_q5k_{1x1,2x1,2x2}` WMMA prefill tier, and Q5_K MoE experts (the
  `{q80,q4k,q5k,q6k}²` cross product, host-routed + device-routed). Measured on
  the RX 7900 XTX with Qwen3-0.6B Q5_K_M: decode **14.6 → 125.2 t/s** (8.6×),
  prefill **1363 → 4133 t/s** (3.0×); Q4_K_M unmoved (126.6 → 127.1 tg, 4403 →
  4420 pp). Q5_K stays on the plain WMMA tier — the Slice-27 software-pipelined
  and Slice-28 cooperative kernels remain Q4_K-only (the pipe prefetch buffer
  would have to carry `qh` as well as the nibbles, and the coop family is a
  measured regression on gfx1100 anyway).
- ✅ **R2 — Q2_K + Q3_K LANDED.** `deq_q2k`/`deq_q3k` native decode (+
  `linear_*`/`embed_*`/`deqf16_*`), `linear_i8_*`/`i8acc_*` int8 dp4a GEMV, the
  `wmma_i8_{q2k,q3k}_{1x1,2x1,2x2}` WMMA prefill tier, and Q2_K/Q3_K MoE experts
  over the full `{q80,q2k,q3k,q4k,q5k,q6k}²` cross product (host-routed +
  device-routed) — so **llama4-Scout's Q2_K/Q3_K expert banks** now go through
  the fast expert pager. Measured on the RX 7900 XTX with Qwen3-0.6B: Q2_K
  decode **12.8 → 137.6 t/s** (10.8×), prefill **1245 → 4163 t/s** (3.3×);
  Q3_K_M decode **20.5 → 133.1 t/s** (6.5×), prefill **1773 → 4331 t/s** (2.4×);
  Q4_K_M control unmoved (127.1 → 126.7 tg, 4412 → 4445 pp). Both stay on the
  plain WMMA tier for the same reason Q5_K does (the Slice-27 pipe and Slice-28
  coop kernels are Q4_K-only). The 6×6 MoE cross product costs no measurable
  hiprtc time (backend init + a 1-token bench is unchanged at ~0.46 s wall).
- ✅ **R3 — Q4_0 + Q4_1 + Q5_1 LANDED.** The legacy 32-element round quants join
  Q5_0 on every fast path: `deq_q40`/`deq_q41`/`deq_q51` native decode (each
  with its `linear_q40`, `embed_q40` and `deqf16_q40` instantiations), the
  `linear_i8_q40` / `i8acc_q40` int8 dp4a GEMV, the `wmma_i8_q40_2x1` WMMA
  prefill tier in all three tiles, and Q4_0/Q4_1/Q5_1 MoE experts. One shared
  `GEN_WMMA_R32` body covers all three WMMA formats — they differ only in `BPB`,
  `HASMIN` and `FIVEBIT`. Q4_1 and Q5_1 are the first AFFINE 32-block formats:
  the offset is a per-block f16 min `m` rather than a constant multiple of `d`,
  so the int8 ones-dot is weighted by each block's own `m`. Measured on the RX
  7900 XTX with Qwen3-0.6B: Q4_0 decode **12.8 → 138.9 t/s** (10.9×), prefill
  **1235 → 4885 t/s** (4.0×); Q4_1 decode **12.8 → 138.8 t/s** (10.8×), prefill
  **1226 → 4889 t/s** (4.0×); Q4_K_M control unmoved (125.7 → 126.9 tg, 4431 →
  4441 pp). No Q5_1 GGUF is cached on the box, so Q5_1 rests on the parity tests
  (shared decode sweep at both m-tiers, int8 GEMV, WMMA, EmbedGather, MoE
  expert). All three stay on the plain WMMA tier for the same reason
  Q5_K/Q2_K/Q3_K do (the Slice-27 pipe and Slice-28 coop kernels are Q4_K-only).
- **Extend native decode GEMV + int8 dp4a + WMMA prefill** to the remaining ~14:
  `IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, TQ1_0, TQ2_0, Q2_0, MXFP4, NVFP4`
  (+ `Bf16` weights). Priority by real usage: **IQ4_XS/IQ4_NL** (common), then
  the exotic IQ/fp4/ternary. Reuse `infr_gguf::dequant`/`iquant_grids` for
  bit-faithful decode; mirror the DEC-per-block pattern in `kernels.rs`.
- **MoE experts beyond the current set** — extend `moe_ffn_expert*`/`moe_*_i8*`
  to the remaining formats. The **escape hatch is now taken** (R3 measured it):
  the Phase-3 `moe_ffn_expert_<gu>_<dn>` cross product is no longer complete
  over `moe_native_fmt`. Going 6×6 → 9×9 (81 pairs/macro) cost **+1.1 s of COLD
  hiprtc** (4.31 → 6.27 s for backend init + a 1-token bench with
  `~/.cache/comgr` cleared; warm startup is ~0.48 s in every variant), so
  `moe_expert_kernel` and its routed twin now return `Option` and only the **48
  reachable** pairs are instantiated — `{q80,q2k,q3k,q4k,q5k,q6k}²` plus
  `{q40,q41,q51} × {q40,q41,q51,q80}` (5.44 s cold). Absent pairs fall back to
  the dequant→f16 `moe_ffn_expert` path, which costs nothing real: those kernels
  only run under `INFR_ROCM_NO_I8` (the default int8 expert path uses the
  per-FORMAT `moe_gate_up_act_i8_<gu>`/`moe_down_i8_<dn>` kernels, still total
  over `moe_native_fmt`), and that switch's comparand IS the f16 path. When
  adding a format, extend `MOE_EXPERT_PAIRS` (exec.rs test module) with only the
  pairs a real GGUF can produce; `moe_expert_pair_tables_agree` pins both
  mappers to it.
- **The `native_id`/`native_idm`/paged-id MoE GEMV family** — Vulkan's
  id-indexed decode GEMVs for resident + paged small-m MoE. ROCm's expert path
  is the batched/routed kernels; add the id-GEMV tier for the low-m regime.

## 2. Attention parity

ROCm has split-KV **decode** attention and a single-wave scalar **prefill**;
Vulkan has the full set (`crates/infr-vulkan/src/recorder.rs:4251-6559`).

- **WMMA flash PREFILL** (`attention_prefill_flash`/`_reg`/`_nc_fa` analogue) —
  move prefill attention onto the matrix cores (currently scalar).
- **Dequant-in-flash for quantized KV** — read Q4_0/Q4_1/Q5_0/Q5_1 KV inside the
  attention kernel (Vulkan `recorder.rs:4520`), avoiding a dequant prepass.
- **KV-cache quant store** — q8_0 KV (`store_q8_dyn`), TurboQuant KV
  (`turbo2/3/4`, `recorder.rs:5328-5353`), block-quant KV. Thread `"rocm"` into
  the seam KV gates (`kv_q8_backend`/`kv_turbo_ok`/`blk_ok`,
  `seam/runner.rs:432-439`).

## 3. Fusion breadth

ROCm has RmsNorm→Linear + Linear→Add peepholes (Slice 32); Vulkan has more
(`adapter.rs`):

- **`GatedActFused`** — SiLU/GeGLU gate×up in one kernel (Vulkan
  `adapter.rs:2115`; ROCm has a fused gate-up in MoE but not the general
  `combined_gu` capability for dense FFN).
- **Fused per-head QkNormRope + KV-cache write** (`kv_write_peephole`,
  `adapter.rs:876`).
- **`Op::RmsNormAdd`** fused norm+residual as a runner-emitted op
  (`adapter.rs:1134`).

## 4. Device-side sampling

ROCm reads logits back to the host and samples in Rust. Vulkan samples on-GPU
(`recorder.rs:7401-7693`): `argmax`, `argmax_prob`, two-stage `sample_topk` (+
chained `_dyn`), and `dg_eb_sample` (diffusion entropy-bound, logits stay in
VRAM). Port these + flip the `gpu_sample`/`argmax_rows`/`argmax_prob`/
`eb_sample_reduce` caps — this also unblocks MTP speculative decode and the
DiffusionGemma in-graph sampler on ROCm.

## 5. Decode-replay tape (record-once decode) — evaluate first

Vulkan's `RecordedCmd`/`replay_n`/`_dyn`-param record-once decode
(`recorder.rs:9049-9169`) submits the whole decode graph n times in one submit.
**But a HIP-graph probe (Slice 31) showed graph replay does NOT cut ROCm's
launch floor** — most of the ~10 µs/dispatch is real per-kernel GPU work, not
launch overhead. So the record-once tape's benefit on ROCm is **uncertain**;
probe/measure before building. (GPU-side MoE routing, Slice 38, removed the
per-layer host sync that would otherwise block a tape.)

## 6. DeltaNet variants

ROCm has chunked prefill + column-parallel decode. Vulkan also has
`deltanet_chunked_split`, `deltanet_seq_split`, and strided variants
(`recorder.rs:6643-6960`) for different shape/occupancy regimes — port the ones
that measurably help qwen35 at real dims.

## 7. Memory-handling parity (paging done; the rest)

Paging (experts/KV/weights) is **done and prefetch-optimized**. Remaining Vulkan
memory machinery, adopt where it helps:

- **BDA arena addressing** (`alloc_arena_bda`, `Backing::BdaSub`, `lib.rs:2528`)
  — HIP raw device pointers may make this unnecessary; **evaluate** whether an
  arena+offset model buys anything over per-buffer `hipMalloc`.
- **Staging ring** for weight uploads (`StagingRing`, `lib.rs:262`).
- **ReBAR** — host-writable device-local VRAM (`Backing::Vram` mapped,
  `lib.rs:429`) for cheaper H2D.
- **UMA / integrated** overflow-spill + the default-ctx clamp ladder — relevant
  for AMD APUs (the RDNA2 iGPU is best-effort, per the memory notes).
- **VRAM budget guard** refinements (`check_vram_budget`/`vram_budget_fits`,
  `lib.rs:2295-2346`) — coordinate the paging budgets with a proper guard.

## 8. Multi-GPU — the largest gap (entirely absent on ROCm)

ROCm is **single-device only** (`RocmBackend::new(device)`). Vulkan has a full
multi-GPU stack across `tp.rs`/`ep.rs`/`pipeline.rs`/`p2p.rs`/`tp_sem.rs`/
`tp_allreduce.rs`. Port each with its HIP equivalent:

- **Multi-device sessions** — one `RocmBackend` per device, `--dev rocm:N`
  selection, a device pool (mirror `parallel.rs`/the seam session pool).
- **Tensor parallelism** (Megatron-style) — column-parallel q/k/v+gate/up,
  row-parallel O+down, one all-reduce per attention and per FFN; KV sharded by
  head (`tp.rs`).
- **Expert parallelism** — stacked expert banks split into per-rank bands
  (`ep_band`), replicated router/top-k, one all-reduce per `Op::MoeFfn`
  (`ep.rs`, `moe_ep_band_remap`).
- **Pipeline parallelism** — split the graph by layer across devices, hand
  hidden state across the boundary (`pipeline.rs`).
- **P2P / peer memory** — HIP peer access (`hipDeviceEnablePeerAccess`,
  `hipMemcpyPeer`) or IPC handles (`hipIpcGetMemHandle`/`hipIpcOpenMemHandle`)
  for host-less cross-device buffer sharing (Vulkan external-memory dma-buf,
  `p2p.rs`).
- **External semaphores** — HIP `hipExternalSemaphore` (or stream events across
  contexts) for cross-device GPU-side sync without a host round-trip (Vulkan
  `VK_KHR_external_semaphore_fd`, `tp_sem.rs`); host-fence fallback.
- **All-reduce / collectives** — **RCCL** (ROCm's NCCL) is the natural fit for
  the TP/EP all-reduce, or a manual fixed-order P2P sum mirroring
  `tp_allreduce.rs` (deterministic reduction for bit-stable goldens).

## 9. Perf endgame — ≥1.0× per model × quant

**Measured baseline (RX 7900 XTX / gfx1100, Q4_K_M, r=3, resident; `infr` t/s ÷
`llama.cpp` HIP t/s — `llama-bench -sm none -mg 0 -fa 1`).** This is the target
to beat, captured after the current campaign:

| model                   | pp512 (infr / llama = ratio) | tg128 (ratio)           |
| ----------------------- | ---------------------------- | ----------------------- |
| Qwen3-0.6B              | 3975 / 22822 = **0.17×**     | 72 / 383 = **0.19×**    |
| gemma-3-1b              | 4755 / 18855 = **0.25×**     | 69 / 275 = **0.25×**    |
| Qwen3.5-0.8B (DeltaNet) | 1198 / 17909 = **0.067×**    | 12.6 / 305 = **0.041×** |
| Llama-3.2-1B            | 3674 / 17643 = **0.21×**     | 51 / 450 = **0.11×**    |
| Qwen3-30B-A3B (MoE)     | 104 / 2878 = **0.036×**      | 33 / 141 = **0.23×**    |

Started far worse: DeltaNet prefill was 0.0007× (1350× gap) and gemma-3 0.04×
before the model-specific catastrophes were fixed; decode climbed ~60× over the
naive baseline. The remaining broad gap is the GEMM/GEMV kernel engineering.

Close it to llama.cpp HIP. The plateau analysis (Slices 25–28) identified the
remaining prefill lever as a **cooperative-LDS, async-pipelined int8 mmq**
(decode-once weight-tile reuse + double-buffered LDS to hide the decode→WMMA
chain — the bit-faithful cooperative kernels landed opt-in as its foundation).
Decode needs an **mmvq-style GEMV** tuning pass. Endgame: `infr compare --sweep`
the full model×quant matrix vs `llama.cpp` HIP, biggest-gap-first, to ≥1.0×.
This is the hardest, most-uncertain work — matching a mature backend's kernel
engineering.

---

## Tricks that may NOT port (measure before adopting)

Confirmed or suspected non-wins on RDNA3/HIP — do not port blindly:

- **HIP graphs / record-once tape** — probed negative (Slice 31): ROCm's
  per-dispatch cost is real GPU work, not launch overhead graphs can hide.
- **BDA arena addressing** — HIP raw pointers may make it redundant (evaluate).
- **rocBLAS f16 prefill** — measured _worse_ end-to-end than int8 WMMA (dequant
  tax + VRAM blowup, Slice 26); kept opt-in only.
- **Cooperative decode-once GEMM (non-pipelined)** — regressed vs the
  single-wave baseline (occupancy, Slice 28); only wins once async-pipelined.

## Parity checklist

- [ ] **Quant coverage** — native decode + int8 + WMMA-prefill for all 24
      formats; MoE experts for all formats; `native_id`/`idm` MoE GEMV family
- [ ] **Attention** — WMMA flash prefill; dequant-in-flash; q8_0/Turbo/block KV
      quant
- [ ] **Fusion** — GatedActFused (dense), QkNormRope+KV-write, RmsNormAdd
- [ ] **Device-side sampling** — argmax / sample_topk / eb_sample (unblocks
      MTP + DG)
- [ ] **Decode-replay tape** — evaluate benefit, build if it wins
- [ ] **DeltaNet** — split/strided variants where they help
- [ ] **Memory** — BDA/staging/ReBAR/UMA/VRAM-guard where beneficial
- [ ] **Multi-GPU** — device pool, tensor-parallel, expert-parallel,
      pipeline-parallel, P2P, external semaphores, all-reduce (RCCL)
- [ ] **Perf ≥1.0×** — async-pipelined mmq prefill + mmvq decode; sweep vs
      llama.cpp HIP

## Cross-backend note

Much of the non-kernel logic being ported here (op routing, peephole fusion,
KV-cache management, capability tiering, paging orchestration, sampling) is
parallel across the CPU/Vulkan/Metal/ROCm backends. A companion effort —
`docs/backend-unification-plan.md` — audits that duplication and plans a shared
seam so parity work is done once, not four times. Prefer landing a shared
implementation there over a fourth per-backend copy when the logic is
device-agnostic.
