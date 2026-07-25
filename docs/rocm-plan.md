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
  **Q8_0/Q4_K/Q6_K/Q5_0**; grid-underfill fixed across attention/WriteKv/
  GatedAct/RmsNorm/Argmax/QkNormRope; RmsNorm→Linear + Linear→Add fusion. ~1.9 →
  ~130 t/s (Qwen3-0.6B Q4_K_M, ~60× over naive, ~0.3× llama.cpp).
- **Prefill** — int8 **WMMA** matrix-core GEMM (RM×CN register tiling +
  software-pipelined Q4_K); ~4500 t/s (~0.2× llama.cpp).
- **Attention** — split-KV / flash-decoding (10.6× at depth), Causal/SWA/Canvas.
- **DeltaNet** — chunked/parallel prefill (88×) + column-parallel decode.
- **MoE** — int8 dp4a experts ({q80,q4k,q6k}²) + **GPU-side top-k routing**
  (device-routed, no host readback).
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
only **4** formats (Q8_0/Q4_K/Q6_K/Q5_0); every other quant falls back to the
slow `dequant→f16` GEMV (256 threads — the pathology that made gemma-3's Q5_0
0.04× before it was covered). Vulkan is native on **all 24** + floats, with a
full `native_id`/`native_idm` MoE-GEMV family
(`crates/infr-vulkan/src/linear.rs:136-254`, `gemm.rs`).

- **Extend native decode GEMV + int8 dp4a + WMMA prefill** to the remaining ~20:
  `Q4_0, Q4_1, Q5_1, Q2_K, Q3_K, Q5_K, IQ4_NL, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, TQ1_0, TQ2_0, Q2_0, MXFP4, NVFP4`
  (+ `Bf16` weights). Priority by real usage: **Q5_K** (Q5_K_M builds),
  **Q2_K/Q3_K** (small quants + llama4-Scout's experts), **Q4_0/IQ4_XS/IQ4_NL**
  (common), then the exotic IQ/fp4/ternary. Reuse
  `infr_gguf::dequant`/`iquant_grids` for bit-faithful decode; mirror the
  DEC-per-block pattern in `kernels.rs`.
- **MoE experts beyond {q80,q4k,q6k}²** — extend `moe_ffn_expert*`/`moe_*_i8*`
  to every format, so **llama4-Scout (Q2_K/Q3_K experts, 37 GB)** finally runs
  through the (already-fast) expert pager, and Q5_K MoE builds work.
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

Close the resident-perf gap to llama.cpp HIP (currently ~0.2–0.3×). The plateau
analysis (Slices 25–28) identified the remaining prefill lever as a
**cooperative-LDS, async-pipelined int8 mmq** (decode-once weight-tile reuse +
double-buffered LDS to hide the decode→WMMA chain — the bit-faithful cooperative
kernels landed opt-in as its foundation). Decode needs an **mmvq-style GEMV**
tuning pass. Endgame: `infr compare --sweep` the full model×quant matrix vs
`llama.cpp` HIP, biggest-gap-first, to ≥1.0×. This is the hardest,
most-uncertain work — matching a mature backend's kernel engineering.

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
