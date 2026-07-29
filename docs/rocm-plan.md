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

## Where we stand (read this first)

**RX 7900 XTX / gfx1100, at `06f6088`.** `infr` t/s ÷ `llama.cpp` HIP t/s.
Oracle is the LOCAL build at
`~/Projects/mxaddict/llama.cpp/build-hip/bin/llama-bench -sm none -mg 0 -fa 1` —
**not** `/usr/bin/llama-bench`, which is broken on this box
(`undefined symbol: ggml_dsv4_hc_post`, a mismatched `libllama`/`libggml`).

| model               | pp512 (infr / llama)      | tg128 (d0)            | tg128 @ d4096         |
| ------------------- | ------------------------- | --------------------- | --------------------- |
| Qwen3-0.6B Q4_K_M   | 14212 / 21714 = **0.66×** | 305 / 384 = **0.79×** | 174 / 307 = **0.57×** |
| Qwen3-0.6B Q6_K     | — / 22033 = **—**         | — / 366 = **—**       | —                     |
| Qwen3-30B-A3B (MoE) | 784 / 2905 = **0.27×**    | — / 141 = **—**       | —                     |

infr-Vulkan baseline (same model, same GPU): pp512 31525, tg128 689,
tg128@d4096 473. Vulkan delivers **2.2×** our prefill and **2.3×** our decode
throughput.

Started at 0.17× / 0.19× (dense) and 0.036× / 0.23× (MoE); before that, DeltaNet
prefill was 0.0007×. gemma-3, DeltaNet and Llama-3.2 rows have not been
re-measured recently and should not be quoted.

### Per-op profile (pp512, Qwen3-0.6B Q4_K_M, 7900 XTX)

| op                           | infr-ROCm | Vulkan  | ratio |
| ---------------------------- | --------- | ------- | ----- |
| Attention (flash)            | 288.0 µs  | 90.4 µs | 3.2×  |
| Linear Q4K 1024×6144 gate/up | 253.6 µs  | 65.8 µs | 3.9×  |
| Linear Q6K 3072×1024 down    | 284.3 µs  | 44.6 µs | 6.4×  |
| Linear Q4K 1024×2048 q/k/v   | 116.7 µs  | 31.1 µs | 3.8×  |
| Linear Q4K 1024×1024         | 63.1 µs   | 31.1 µs | 2.0×  |
| QkNormRope                   | 25.4 µs   | 5.1 µs  | 5.0×  |
| RmsNorm                      | 13.1 µs   | 6.0 µs  | 2.2×  |
| WriteKv                      | 10.6 µs   | 4.6 µs  | 2.3×  |

Started at 0.17× / 0.19× (dense) and 0.036× / 0.23× (MoE); before that, DeltaNet
prefill was 0.0007×. gemma-3, DeltaNet and Llama-3.2 rows have not been
re-measured recently and should not be quoted.

### Vulkan-perf parity audit: techniques not yet ported

A comprehensive audit of every kernel family Vulkan ships that ROCm does not —
or ships in a weaker form. Ordered roughly by value. Each item names the Vulkan
reference (shader or recorder function) and the gap.

#### A. KV-cache quantization (VRAM + bandwidth)

| technique                  | Vulkan ref                                 | ROCm state                                                  |
| -------------------------- | ------------------------------------------ | ----------------------------------------------------------- |
| **store_q8** (Q8_0 KV)     | `recorder.rs:5162` / `store_q8.comp`       | missing — only f16 WriteKv; Q8_0 read path ready            |
| **store_turbo**            | `recorder.rs:5298` / `quant_turbo.comp`    | missing                                                     |
| **store_kv_dense**         | `recorder.rs:5345`                         | missing                                                     |
| **dequant_kv_f16 prepass** | `recorder.rs:5377` / `dequant_kv_f16.comp` | missing — no way to read quantized KV in attention          |
| **q8_0/Turbo/blk seam**    | `runner.rs:450-457`                        | ✅ `kv_q8_backend` threaded; `kv_turbo_ok`/`blk_ok` not yet |

**Impact**: Q8_0 halves KV VRAM (doubles usable context length). Q4_0 quarters
it. Vulkan gates these behind seam flags that ROCm currently fails — the runner
downgrades ROCm to f16 KV regardless of config.

#### B. Inline KV decode in attention (skip dequant prepass)

| technique                      | Vulkan ref                                           | ROCm state                                                                    |
| ------------------------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Q8_0 in attn_partial**       | `attn_partial.comp:136-158` (-DKQ8/-DVQ8)            | ✅ in `attention_prefill_flash` (q8kv_decode); needs store_q8 to produce data |
| **Q4_0/Q4_1/Q5_0/Q5_1 inline** | `attn_partial.comp:69-121` (-DKMAINLINE/-DVMAINLINE) | missing — skip prepass, decode block scale once per block                     |
| **Flash-stage Dequant**        | `recorder.rs:4399` / `FlashStage::Dequant(dt)`       | missing — routes through staged builds                                        |
| **Flash-warp Dequant**         | `attn_flash_warp.comp`                               | no WMMA flash kernel yet                                                      |

**Impact**: Vulkan's inline decode avoids a full f16-scratch write+read for
every quantized KV cache read — a 2× bandwidth savings at decode depth where
attention is K/V-read-bound. Combined with Q8_0 store, this halves decode
attention's global memory traffic (half the bytes, no prepass round-trip).

#### C. WMMA/matrix-core attention for prefill AND decode

| technique                                | Vulkan ref                                     | ROCm state                                        |
| ---------------------------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **attn_flash_warp** (coopmat)            | `attn_flash_warp.comp` (BM=64, BN=64, 8 warps) | scalar placeholder `attention_prefill_flash_wmma` |
| **attn_qk_warp** (coopmat Q·K)           | `attn_qk_warp.comp`                            | missing                                           |
| **attn_pv_warp** (coopmat P·V)           | `attn_pv_warp.comp`                            | missing                                           |
| **attn_flash_partial** (split-K+coopmat) | `attn_flash_partial.comp`                      | missing — decode uses scalar lane-per-key P7      |
| **attn_softmax** (cluster softmax)       | `attn_softmax.comp`                            | missing — uses scalar wave_allmax                 |

**Impact**: 3.2× gap on prefill attention (288 vs 90 µs). Vulkan uses WMMA for
both Q·K _and_ P·V, with 4× wider tiles (BM=64 vs br=16). The decode at depth is
also attention-bound — a coopmat split-K partial+combine would match Vulkan's
decode attention speed.

#### D. F16-activation GEMM (A_GLOBAL family)

| technique                        | Vulkan ref                                              | ROCm state                                           |
| -------------------------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| **matmul_native_f16a** (n128_ag) | `recorder.rs:2110` / `native_gemm_warp.comp` -DA_GLOBAL | missing — ROCm uses int8 quantized activations       |
| **BM=32/16 small-m tiles**       | `recorder.rs:70-101`                                    | missing — only 2×1/2×2 RM×CN tiles                   |
| **Split-K for narrow-N GEMM**    | `recorder.rs:2209` / `matmul_native_splitk`             | implicit via WMMA tile grid, but no explicit split-K |
| **BF16 coopmat**                 | `recorder.rs:2531` / `native_gemm_warp.comp` -DBF16CM   | missing                                              |
| **FMA fallback (no coopmat)**    | `recorder.rs:2905` / `native_gemm_fma.comp`             | ROCm has scalar dequant→f16 fallback                 |

**Impact**: Vulkan's A_GLOBAL path pre-converts f32 activations to f16 via
`store_f16`, then the GEMM reads f16 activations directly from global (no
staging). This halves the BM×BK LDS for activations (to zero), letting 3
workgroups fit per CU instead of 2. Combined with the BM=64 tile, this is why
Vulkan's Q4_K gate/up GEMM is 3.9× faster (65.8 vs 253.6 µs).

#### E. MMQ (decode-once-reuse) all-expert MoE GEMM

| technique                    | Vulkan ref                                    | ROCm state                                            |
| ---------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| **matmul_mmq_experts**       | `recorder.rs:7867` / `native_gemm_mmq_*.comp` | missing — ROCm uses id-multi GEMV (serial per-expert) |
| **matmul_mmq_experts_paged** | `recorder.rs:7975`                            | missing                                               |
| **moe_scatter_reduce**       | `recorder.rs:8048`                            | missing — needed for MMQ output scatter               |

**Impact**: Vulkan's MMQ decodes each expert's weight ONCE into LDS, then all
tokens (rows) that hit that expert share the decode — the llama.cpp threadblock
pattern. ROCm's id-multi GEMV re-decodes weights for every expert × token,
wasting compute. This is the largest MoE prefill lever.

#### F. QkNormRope + WriteKv (full peephole)

| technique                       | Vulkan ref                                                | ROCm state                                 |
| ------------------------------- | --------------------------------------------------------- | ------------------------------------------ |
| **qk_norm_rope_interleaved_at** | `recorder.rs:5466` (fused QK norm + rope + K write + BDA) | F1d only fused QkNormRope→WriteKv (K half) |
| V-write (Linear→WriteKv)        | `recorder.rs` write_kv peephole                           | missing — V goes through separate WriteKv  |

**Impact**: Vulkan's interleaved shader merges Q/K norm, RoPE, K cache store AND
sends the address as a BDA pointer — all in one dispatch. ROCm's F1d only folds
the K half (QkNormRope→WriteKv). The V-write is still a separate dispatch per
layer.

#### G. DeltaNet variants

| technique                  | Vulkan ref         | ROCm state |
| -------------------------- | ------------------ | ---------- |
| **deltanet_chunked_split** | `recorder.rs:6801` | missing    |
| **deltanet_seq_split**     | `recorder.rs:6928` | missing    |
| **deltanet_strided**       | `recorder.rs:6671` | missing    |

**Impact**: Vulkan's split/strided variants fill the GPU better at different
depth×state-size ratios. ROCm only has the basic chunked prefill +
column-parallel decode.

#### H. Model-specific fusion (gemma4/E2B)

| technique                | Vulkan ref                              | ROCm state |
| ------------------------ | --------------------------------------- | ---------- |
| **e2b_gate**             | `recorder.rs:1831` / `e2b_gate.comp`    | missing    |
| **mul_sigmoid** (gemma4) | `recorder.rs:7203` / `mul_sigmoid.comp` | missing    |

**Impact**: `e2b_gate` is a fused Linear(f32)+GatedAct(gelu, stride) for E2B
models. `mul_sigmoid` is gemma4's per-layer output gate. Both are small
capability items — the models work via separate op dispatch, but each dispatch
is 2-3 µs of device turnaround on ROCm.

#### I. DiffusionGemma in-graph sampler

| technique            | Vulkan ref                               | ROCm state                                  |
| -------------------- | ---------------------------------------- | ------------------------------------------- |
| **dg_eb_sample**     | `recorder.rs:7661` / `dg_eb_sample.comp` | missing — DiffusionGemma uses host fallback |
| **eb_sample_reduce** | `Backend` trait default                  | not implemented                             |

**Impact**: DiffusionGemma's denoise loop does a per-canvas-row entropy-bound
sampler — argmax, entropy, CDF-sample. On Vulkan this stays in-graph (no host
round-trip). On ROCm it falls back to full host download+compute, costing one
PCIe round-trip per denoise step.

#### J. MMQ per-format GEMMs (wider set than WMMA)

Vulkan ships 18 MMQ format specializations (`native_gemm_mmq_*.comp`): q2k, q3k,
q4k, q5k, q6k, q8_0, q4_0, q4_1, q5_0, q5_1, iq2_s, iq3_s, iq4_nl, iq4_xs, q2_0,
mxfp4, nvfp4. ROCm's WMMA family covers these but only as single-wave scalar
WMMA — not the multi-warp threadblock LDS-shared decode-once pattern Vulkan's
MMQ uses. For formats other than Q4_K (which has the Slice-27 pipe), ROCm's
re-decode-every-row overhead is the full Vulkan gap.

#### K. Memory machinery (probed negative or low-priority)

| technique                | Vulkan ref                | ROCm state                                           |
| ------------------------ | ------------------------- | ---------------------------------------------------- |
| HIP graphs / replay tape | probed twice: -36%        | **DO NOT PORT** — per-dispatch cost is real GPU work |
| BDA arena addressing     | `alloc_arena_bda`         | HIP raw pointers likely make this redundant          |
| Staging ring             | `StagingRing`             | missing — lower priority                             |
| ReBAR                    | `Backing::Vram`           | evaluate on APU hardware                             |
| rocBLAS f16 prefill      | measured worse (Slice 26) | **DO NOT PORT** — dequant tax + VRAM blowup          |
| Cooperative decode-once  | regressed (Slice 28)      | only wins with double-buffered async pipeline        |

### Ranked next work (top 5, by estimated value)

1. **KV-cache quantization (Q8_0)** — `store_q8` kernel + seam gate threading.
   Halves KV VRAM, unlocks double context length. `qk_norm_rope` already writes
   f16 K, so only V needs the store kernel.
2. **f16 A_GLOBAL GEMM** — pre-convert activations to f16, eliminate the int8
   quant/dequant tax on the GEMM path. This is Vulkan's largest
   single-multiplier advantage (2-4× on every GEMM). Requires `store_f16` for
   activations + a new WMMA kernel that reads f16 A fragments directly from
   global.
3. **WMMA flash attention (intrinsics)** — replace the scalar placeholder with
   real `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`. The kernel body and
   dispatch are already in place.
4. **Inline KV decode in attention** — Q8_0 (and later Q4_0/Q4_1/Q5_0/Q5_1)
   decode inside the flash loop. Skips the dequant prepass round-trip.
5. **MMQ all-expert MoE GEMM** — decode-once-reuse for MoE expert weights,
   matching the llama.cpp MMQ threadblock pattern already shipped on Vulkan in
   `matmul_mmq_experts`.

### Measurement traps that have already cost this campaign time

- **`INFR_PROF_OPS` overhead scales with DISPATCH COUNT.** 0.3% on MoE `pp512`
  (48 spans) but **37% on dense `tg128`** (~79 000 spans; 134.5 vs 212.2 t/s).
  Use it to RANK ops; never size a total against wall time on a dispatch-heavy
  workload, and never quote a profiled t/s. To size ONE decode op, skip it and
  diff two clean runs.
- **A cold page cache on the 17 GB MoE model reads ~40% low.** `pp512` is ~787
  now; if you see ~211 or ~173 that is a cold cache, not a finding. P2 built a
  whole false conclusion on this before it was caught. **Always discard the
  first run of a burst** — thermal boost on small models, page cache on big
  ones.
- **The oracle drifts ~2% between sessions**, and its `pp512` on small models
  has ±25–27% spread even at `-r 5`. Measure both sides in one sitting; treat
  `pp512` ratios as approximate and `tg128` ratios (±1) as solid.
- **Every slice brief so far has been wrong** — 6 for 6. See the perf log below;
  "profile before designing" is not ceremony here.

## Delivered so far (do not redo)

- **Part A correctness** — all archs (llama, qwen2/3, gemma3/4/E2B, qwen35
  DeltaNet, qwen3moe, BitNet i2_s) coherent; all 24 weight quant formats
  parity-validated; a `rocm_seam` gpu_seam gate (9 models, token-for-token/hash
  vs CPU). Goldens locked.
- **Decode** — native in-kernel block decode + int8-activation dp4a GEMV for
  **all 24 weight quants**
  (Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/TQ1_0/TQ2_0/Q2_0/MXFP4/NVFP4
  — nothing quantized is left on the `dequant→f16` fallback, which now serves
  only the dense float dtypes F32/BF16); grid-underfill fixed across
  attention/WriteKv/GatedAct/RmsNorm/Argmax/QkNormRope; RmsNorm→Linear +
  Linear→Add fusion. ~1.9 → **213 t/s** (Qwen3-0.6B Q4_K_M — ~110× over naive,
  0.56× llama.cpp; see the table at the top for current ratios).
- **Prefill** — int8 **WMMA** matrix-core GEMM (RM×CN register tiling +
  software-pipelined Q4_K) plus P1's tiled flash prefill; ~4500 → **13931 t/s**
  (0.69× llama.cpp).
- **Attention** — split-KV / flash-decoding (10.6× at depth), Causal/SWA/Canvas.
- **DeltaNet** — chunked/parallel prefill (88×) + column-parallel decode.
- **MoE** — int8 dp4a experts
  ({q80,q2k,q3k,q4k,q5k,q6k,q40,q41,q51,iq4nl,iq4xs,iq2xxs,iq2xs,iq2s,iq3xxs,iq3s,iq1s,iq1m,tq10,tq20,q20,mxfp4,nvfp4}
  — every weight quant except Q5_0, which no shipped GGUF uses for expert banks;
  gate/up × down) + **GPU-side top-k routing** (device-routed, no host
  readback) + the **id-indexed multi-slot expert GEMV** (R8: all `rows × n_used`
  slots in ONE dispatch per stage instead of a serialized per-expert host loop —
  Qwen3-30B-A3B `pp512` 104 → 254 t/s, `tg64` 35 → 42, bit-identical to the loop
  it replaces) and **P2**'s bucket-sorted batched twin. With P3 that model's
  `pp512` now reads **787 t/s**.
- **Memory paging** — all three Vulkan modes: MoE expert LRU cache (host→VRAM,
  copy-stream overlap), KV-cache overflow to host (`INFR_KV_OVERFLOW`),
  dense-weight prefetch ring. 30B MoE, 27 GiB KV contexts, and >VRAM dense
  models run on a 24 GB card.
- **Module cache (RC)** — the hiprtc code object is persisted to
  `~/.cache/infr/rocm-module-<arch>.bin` (`infr_core::kernel_cache`, shared with
  Vulkan's pipeline cache; `kernels.rocm.module_cache`, default on) and reloaded
  with `hipModuleLoadData`. **The ~9.2 s cold `hiprtcCompileProgram` is no
  longer tied to comgr's cache**: wiping `~/.cache/comgr` used to cost 9.2 s on
  the next launch and now costs **0.47 s**. Warm-cache launch 0.49 → 0.475 s
  (`Pipelines::build` itself 33 ms → 13 ms). It is NOT a licence to grow the
  kernel set for free: a `kernels.rs` edit changes the key
  (`FNV(hip_source())`), so the first run after any edit still pays the full
  cold compile — the "cold hiprtc budget" notes below stand, they are now a
  per-EDIT cost rather than a per-comgr-eviction one.

---

# The parity roadmap

Each item: the gap (Vulkan has it, ROCm lacks it), the Vulkan reference, the
ROCm task. Ordered roughly by value. Every kernel lands with a parity test vs
the CPU reference before its capability flag flips (the rule that kept the blind
kernels correct).

## 1. Quant coverage — COMPLETE (24/24)

Native in-kernel block decode + int8-activation dp4a GEMV + WMMA prefill for
**all 24 weight quant formats**, MoE expert banks for every format a shipped
GGUF packs them with (all but Q5_0), and the id-indexed multi-slot expert GEMV
(**R8**) plus its bucket-sorted batched twin (**P2**). Nothing quantized is left
on the `dequant→f16` fallback, which now serves only F32/BF16.

Slices R1–R8 did this work and are finished; the blow-by-blow was pruned. What
survives is the part that is still load-bearing when writing a kernel:

- **Super-block alignment decides whether you can use wide loads.** Q4_K (144 B)
  and Q5_K (176 B) are multiples of 16, so a `uint4` cast is legal. **Q6_K's
  210-byte super-block is only 2-byte aligned** and a `uint4` cast there is UB.
  Use `__builtin_memcpy`, which states the align-1 contract honestly and **still
  lowers to a single `global_load_b128` on gfx11** — verified in ISA with
  `hipcc --offload-arch=gfx1100 -S`. This one fact was worth ~2× twice (P3, P4).
- **The IQ/FP4 codebooks are generated from `infr_core::iquant_grids`**, the
  same tables the CPU dequant reads, so kernels stay bit-exact by construction
  rather than by hand-transcribed tables. MXFP4 and NVFP4 index
  `KVALUES_MXFP4`/E2M1 the same way.
- **The cold-hiprtc budget is per-EDIT, not per-launch.** The module cache
  (`kernels.rocm.module_cache`, `infr_core::kernel_cache`) persists the code
  object, but any `kernels.rs` edit changes the key (`FNV(hip_source())`) and
  pays the full ~9.2 s compile once. Adding kernel families is not free.
- **MoE format routing** lives in `moe_native_fmt` / `MOE_EXPERT_PAIRS`
  (`exec.rs`); paged MoE deliberately keeps the per-expert loop for its
  copy/compute overlap.

## 2. Attention parity

ROCm has split-KV **decode** attention and, since **P1**, a tiled flash
**prefill**; Vulkan has the full set
(`crates/infr-vulkan/src/recorder.rs:4251-6559`).

### P1 — tiled flash prefill attention (landed)

The single-wave scalar prefill is gone. `attention_prefill_flash` gives a
WORKGROUP a tile of `br` consecutive query rows of one head, streams the kv
range through LDS in `bc`-key tiles, and runs a one-pass online softmax.
`kernels.rocm.attn_flash` routes back to the plain kernel (the A/B control its
parity cases need). What the old kernel did, and why each of those is now gone:

| the plain kernel                                                         | the flash kernel                                                                                                                                                                        |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| one wave per (query row, head), so K and V are re-read once **per row**  | once per query **TILE** — `br = 16` at head_dim 128, so 16× less global traffic                                                                                                         |
| **two** passes over the whole kv range (a max pass, then exp/accumulate) | one pass, online softmax with an `exp(m_old − m_new)` rescale                                                                                                                           |
| evaluates the q·k dot for masked keys and throws the score away          | the key range `[j_lo, j_hi)` is clamped per workgroup from the tile's own position span — the masked half of a causal score matrix, and everything below a SWA window, is never visited |
| a 5-`shfl` butterfly **per key** to reduce the dot                       | LANE PER KEY for scores (no cross-lane reduction at all) and lane per DIM pair for P·V (one `__shfl` broadcast per key)                                                                 |
| 2-byte `global_load_ushort` per element                                  | `uint4` (8 halves) tile staging — measured as the kernel's single biggest cost before it was vectorized                                                                                 |

**Measured, in isolation** (the shipping shapes, µs per layer; the standalone
harness is the F4 playbook — build the kernel against a driver and time it away
from the model):

| shape (rows × kv, head_dim, n_head)  | plain | flash    |           |
| ------------------------------------ | ----- | -------- | --------- |
| 512×512, d=128, h=16 (Qwen3-0.6B)    | 2643  | **279**  | **9.5×**  |
| 2048×2048, d=128, h=16               | 41397 | **3652** | **11.3×** |
| 512×512, d=256, h=4 (gemma-3-1b)     | 1823  | **273**  | **6.7×**  |
| 512×512, d=128, h=32 (Qwen3-30B-A3B) | 5975  | **477**  | **12.5×** |

**Tiling policy** (`attn_flash_tiling`, unit-tested). `bc` is 32 keys — one per
lane — and drops to 16 for head dims above 128; `nw` is then the widest
workgroup whose LDS tile fits 32 KiB, so two workgroups stay co-resident per CU.
`br = nw · ATTN_FLASH_QPW` is exactly the factor by which global K/V traffic
falls, so widest-that-fits IS the policy. `head_dim % 32 != 0` or `> 256` keeps
the plain kernel. Swept: `QPW` 2 → 279 µs, 4 → 313, 8 → 441 (the accumulator is
register state, and occupancy falls faster than the K reuse pays); `(nw, bc)`
over a 5 × 3 grid, with `(8, 32)` the winner at every point on the diagonal.

**The correctness lesson, which cost this slice its longest detour.** The flash
kernel gives ONE lane the whole `q·k` dot where the plain kernel splits it over
32 lanes and butterfly-reduces. Written the obvious way — a serial sum — it is
~1e-7 off, **passes every tolerance gate in `parity.rs` including the CPU-oracle
comparison at 2048 keys**, and still flips a near-tie argmax fourteen tokens
into the Q8_0 seam run ("I know" → "I remember"). The fix is to rebuild the
reference's tree in registers: `g[t]` is lane `t`'s partial, and the reduction
pairs `t` with `t+16`, then `t+8`, … which is exactly `__shfl_xor` at off=16, 8,
…. It costs ~4% and it is what keeps the goldens.

It is still not BIT-equality, and that is a property of the reference: the plain
kernel evaluates the same dot TWICE (max pass, accumulate pass), and nothing
makes the compiler round two separately-scheduled copies identically — its own
softmax weight for a single-key row is not exactly 1. So the gate is stated as
what is measurable: inside one key tile, ≤6% of outputs differ from the plain
kernel at all, where a serial dot leaves 25–41%
(`attention_prefill_flash_keeps_the_plain_kernels_score_tree`).

**Correctness evidence.** Four new parity cases + one tiling unit test
(infr-rocm 152 → 156). vs the CPU reference at rel ≤ 1.3e-7 on six shapes —
512×512 GQA, **2048×2048 causal**, **2048×2048 SWA(384) at head_dim 256**,
`rows`/`kv` not multiples of the tile (300×300), a non-zero `pos` (40×160 at
pos=120), and the `Canvas` mask; plus the plain-kernel A/B on all six, and a
fallback case (`head_dim = 20`) proving an untileable shape still answers.
`rocm_seam` **9/9 in BOTH release and debug** with the qwen3 golden
`0xfd63781ea3bfa785` unmoved; the shared decode-parity sweep unchanged;
`ROCM_TOL` untouched. Debug matters here for the same reason it did in F5 — the
`Attention` `dst` is drawn with `uninit_dev`, and the flash arm has to write
every element of it (it does: `nw` waves × `QPW` rows cover `br`, and the `c`
blocks cover `head_dim` in lane pairs).

**Still open on this axis:**

- **WMMA.** P1 is a scalar-ALU flash kernel, not a matrix-core one. The profile
  says that was the right call for THIS step — the plain kernel was losing to
  redundant memory traffic, not to arithmetic throughput, and 2.6 GB/layer of
  re-read for 2 MB of distinct cache is not something matrix cores fix. Now that
  the traffic is gone, the ~1 GFLOP/layer of actual math is the floor, and WMMA
  is what would approach it: at 279 µs/layer the kernel is still ~5× off its own
  LDS-throughput bound.
- **Dequant-in-flash for quantized KV** — read Q4_0/Q4_1/Q5_0/Q5_1 KV inside the
  attention kernel (Vulkan `recorder.rs:4520`), avoiding a dequant prepass.
- **KV-cache quant store** — q8_0 KV (`store_q8_dyn`), TurboQuant KV
  (`turbo2/3/4`, `recorder.rs:5328-5353`), block-quant KV. Thread `"rocm"` into
  the seam KV gates (`kv_q8_backend`/`kv_turbo_ok`/`blk_ok`,
  `seam/runner.rs:432-439`).

## 3. Fusion breadth

ROCm has RmsNorm→Linear + Linear→Add peepholes (Slice 32), the **F1b**
sibling-GEMV activation-quant memo, the **F1c** MoE folds (RmsNorm→MoeFfn,
MoeFfn→Add) and the **F1d** K-write fold (QkNormRope→WriteKv) — see §9's perf
log for what each taught — plus, since **F1**, the capability-gated fusions its
executor already implemented but the "start with NOTHING fused" bring-up dial in
`backend.rs`'s `capabilities()` had never let the seam emit. Each was proved
against the CPU reference first (`crates/infr-rocm/tests/parity.rs`, the "F1
fusion gate" section) and then measured on a 7900 XTX:

| capability      | state | evidence                                                                                                                                                                                                                                                                                                |
| --------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `combined_gu`   | ON    | one `[2*nff, ne]` FFN GEMV + `GatedActFused`. Qwen3-0.6B Q4_K_M tg128 126.3 → **137.3 t/s (+8.7%)**, pp512 +0.7%. Golden unmoved.                                                                                                                                                                       |
| `gated_rmsnorm` | ON    | fused per-head RMSNorm×SiLU-gate, **bit-identical** to the `QkNorm`→`GatedAct` pair (max_err 0.0). Qwen3.5-0.8B pp512 +1.3%, tg a wash.                                                                                                                                                                 |
| `argmax_rows`   | ON    | multi-row `Op::Argmax` is id-for-id with the CPU at a 151936 vocab, ties included. No perf change today — it gates only the MTP verify accept, and no ROCm `MtpHeadSession` exists yet.                                                                                                                 |
| `embed_gather`  | OFF   | works, but the native `embed_*` decode f16-rounds each element (`fin`) where the host embed path is exact f32 — ~2.5e-4 relative on the **Q6_K** `token_embd` a Q4*K_M GGUF ships, which moves the qwen3 golden. Also −1.2% tg. Needs an exact-f32 `deq*\*` sibling before it is even worth re-pricing. |

Kernel launches per decode token, before → after: Qwen3-0.6B **563 → 507**,
Qwen3.5-0.8B **561 → 495**, Qwen3-30B-A3B **1059 → 1059** (a pure-MoE arch has
no dense `ffn_gate`/`ffn_up` to concatenate).

### Next on this axis after F1d

Dense decode is 423/token: `linear_i8_q4k` 140 (5 real GEMVs/layer),
`rmsnorm_quant_i8_32` 57, `qk_norm_rope` 56, `quant_i8_32` 56, `linear_i8_q6k`
29, `attention` 28, `gated_act` 28, `write_kv` 28, `argmax` 1. MoE is 819/token
with `quant_i8_32` 96 and `write_kv` 48.

The next-largest item that is not the GEMV itself is **`quant_i8_32` — 56 dense
/ 96 MoE, two per layer** — and the conclusion is unchanged from F1c: neither is
sibling-redundant (o*proj's row comes from `attention`, down*proj's from
`gated_act`, the MoE `h` from `moe_gate_up_act_i8_idm**`), so killing them needs
an int8-emitting epilogue on the **producing** kernel, not a peephole — the same
new-kernel work the V write needs (see the F1d note in §9's perf log), on the
same set of kernels. After that the remaining `write_kv` ×28/48 (V) is the next
fusable count.

## 4. Device-side sampling

ROCm samples greedily in-graph already (`Op::Argmax`, rows 1 and — since F1 —
rows > 1); what is missing is the stochastic and diffusion half. Vulkan
(`recorder.rs:7401-7693`) also has `argmax_prob`, two-stage `sample_topk` (+
chained `_dyn`), and `dg_eb_sample` (diffusion entropy-bound, logits stay in
VRAM). Port these + flip the `gpu_sample`/`argmax_prob`/`eb_sample_reduce` caps
— this also unblocks MTP speculative decode and the DiffusionGemma in-graph
sampler on ROCm.

Do NOT expect throughput from the readback itself: F1 measured a temp-0.6 decode
(which downloads the whole `[vocab]` logits row every token — 608 KB for Qwen3)
against a greedy one that reads back 4 bytes, and the two are indistinguishable
(137.0 vs 136.6 t/s). At ~25 GB/s over PCIe that transfer is ~24 µs against a
7.3 ms token. The win here is capability, not bandwidth.

## 5. Decode-replay tape (record-once decode) — evaluate first

Vulkan's `RecordedCmd`/`replay_n`/`_dyn`-param record-once decode
(`recorder.rs:9049-9169`) submits the whole decode graph n times in one submit.
**But a HIP-graph probe (Slice 31) showed graph replay does NOT cut ROCm's
launch floor** — most of the ~10 µs/dispatch is real per-kernel GPU work, not
launch overhead.

**P6 closed this question with a direct measurement: a record-once tape is worth
~0.4%.** `INFR_PROF_STAGES=1` prices the per-token host rebuild this tape would
eliminate — graph build + compile + bind — at **0.021 ms against a 4.98 ms
`execute`**. Separately, phase timers put the host op-walk at 0.41 ms while the
host then WAITS 4.50 ms, so the host already runs ahead of the GPU and the tape
would remove time that is not on the critical path. The remaining out-of-kernel
cost is device-side dispatch turnaround (~2.6 µs each, measured in situ by
null-kernel injection), which a tape does not touch. **Do not build it for
throughput.** (GPU-side MoE routing, Slice 38, removed the per-layer host sync
that would otherwise block a tape, so it stays cheap to build if wanted for
another reason.)

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

Current ratios and the ranked next work are at the **top** of this file ("Where
we stand"), because they are what a reader needs first. This section is the
history.

### The perf log — what each slice actually taught

Compressed from the full write-ups, which were pruned — each entry is the number
and the transferable lesson, and the commit is there so `git show <sha>` still
has the full reasoning. The code is the authority on mechanism.

| slice | commit    | result                                                                               |
| ----- | --------- | ------------------------------------------------------------------------------------ |
| F1    | `702a57d` | enable built-but-disabled fusions; +5.4% (not the claimed 8.7%)                      |
| F1b   | `15642b7` | quantize a decode activation row once per sibling GEMV group                         |
| F1c   | `48d37e4` | MoE residual + router-norm folds, 96 of 963 launches                                 |
| F1d   | `24aa5f0` | f16-out `qk_norm_rope` writes K straight into the cache                              |
| F4    | `bfc18c0` | 128-bit weight loads + 2 rows/wave in the int8 decode GEMV                           |
| F5    | `4a88a23` | stop zeroing scratch the kernel fully overwrites                                     |
| P1    | `601bf73` | tiled flash prefill attention — **2.6× dense pp512, 4.2× pp2048**                    |
| P2    | `5987581` | MoE bucket-sort by expert, re-read 32×→4× — only **1.13×**                           |
| P3    | `0cd7432` | branchless dword-wide `i8acc_q6k` — **1.97× MoE pp512**                              |
| P4    | `00bf77f` | same fix in the DENSE Q6_K kernels — **2.05× Q6_K pp, 1.34× mainline Q4_K_M decode** |
| P6    | `f796557` | batched-prefetch decode attention — 1.10× d0, **1.24× @d4096**                       |
| P7    | —         | one-pass online-softmax + one-key-per-lane split-KV — **1.16× d0, 1.48× @d4096**     |
| P7b   | —         | vectorised `copy_strided` (float4 per row) — **Q6_K pp512 1.51×**                    |
| P7c   | —         | multi-block two-pass argmax — **8.6×** per disch, decode +2.1%                       |
| P7e   | —         | MoE PF4 block-prefetch gate/up — **-14.7%**, register pressure                       |
| P7f   | —         | MoE CN=2 column tiling Q4_K gate/up — **+12.9%** MoE pp512 (699→790)                 |
| P7h   | —         | WMMA RM=4 tile (dense), KICK=2 ILP (attention flash) — both **flat/regress**, VGPR   |

**The briefs were wrong every time, and how they were wrong is the pattern:**

- **F4** — the premise was that decode was bandwidth-starved. It was not: the
  kernel hit **933 GB/s, 97% of peak**. The aggregate looked slow because it was
  diluted by per-dispatch cost. Measure the kernel, not the aggregate.
- **P1** — the plan ranked WMMA first. The profile said the scalar kernel
  re-read K/V ~1000× (~2.6 GB per layer for 2 MB of distinct cache) against ~1
  GFLOP of math. Matrix cores don't fix traffic; tiling does. WMMA dropped to
  third.
- **P2** — predicted ~10× from removing 8× of expert-bank traffic. Delivered
  **1.13×**, because the expert GEMV was never bandwidth-bound (~6% of peak
  after). A well-evidenced negative about R8's whole premise.
- **P3** — named one defect (a per-lane-divergent 4-way branch). There were
  **two**, and the unnamed one was bigger: 16 scalar `global_load_u8` per 16
  codes, which are **address-divergent across lanes**, so the real cost is L1
  line-requests, not instruction count.
- **P4** — called the branch fix mechanical. In `GEN_WMMA_Q6K` that branch
  derives from a **loop counter**, so it is wave-uniform and worth nothing; the
  win was again the loads. The slice nearly reported a fix that wasn't one.
- **P6** — said decode was launch-bound with 28% out-of-kernel. Out-of-kernel
  was 16%, host graph rebuild was 0.021 ms of a 4.98 ms token, and the real
  answer was that attention is half the token.
- **P7d** — predicted ~4% decode win from fusing Q+K QkNormRope (56→28
  dispatches). Delivered **-1.6% regression**: the dispatch savings (~73 µs)
  were offset by register pressure from doing both Q and K in one wave.
  Per-dispatch cost is real GPU work (Slice 31, P6), not launch overhead.
- **P7e** — predicted PF=4 block prefetch would hide MoE GEMV memory latency.
  Delivered **-14.7% regression**: staging 4 super-blocks' weights into
  registers (32 uint4 per lane) caused register spilling to scratch. With 3.1M
  waves per dispatch, the hardware scheduler already hides latency — the
  bottleneck is wave count, not per-wave pipeline depth.
- **P7h** — predicted RM=4 WMMA tiling would improve weight reuse for wide-N
  GEMMs. Delivered **-33% regression**: 4× the row accumulators (32 VGPR) plus
  weight-decoding temporaries hit the spill threshold. The existing 2×1/2×2
  auto-tier is at the register-occupancy optimum. KICK=2 ILP in the attention
  flash kernel was flat — doubling the per-lane accumulator pool (32→64) spares
  too much VGPR for marginal LDS-latency hiding.

**Correctness lessons worth keeping:**

- **P1's near-tie argmax.** A plain serial dot sat ~1e-7 from the CPU oracle,
  passed every tolerance gate including at 2048 keys, and still flipped a
  near-tie argmax fourteen tokens into a seam run. The fix was rebuilding the
  reduction TREE in registers to match the shuffle order. Bit-equality is
  unattainable there (the reference evaluates the same dot twice and nothing
  forces identical rounding), so the gate is a measurable property instead:
  inside one key tile ≤6% of outputs differ, where a serial dot leaves 25–41%.
- **P6: decode attention pass 2 must RE-DERIVE each score, not reuse pass 1's.**
  Caching them is +4% and WRONG — the reference computes the score twice, LLVM
  contracts the two copies differently, `max` comes from one and `expf(s-max)`
  from the other. Reusing one is _more_ self-consistent and therefore differs.
- **A register staging array needs a compile-time subscript** or LLVM sinks it
  to scratch (592 B/lane) and the win evaporates. P6: 195.7 base / 212.8
  registers / 193.1 with the LDS workaround.
- **F1d: V is NOT the K-write peephole's to take.** V's producer is a `Linear`,
  a `Copy`, or an in-place `AddBias`/`QkNorm` — none is a rope, and the in-place
  ones have no output pointer to redirect. Absorbing `Linear → WriteKv` is a
  different rewrite: an f16-cache-store epilogue on every int8-decode GEMV entry
  point, mutually exclusive with the fused-residual epilogue those kernels
  already carry, and it must decline the prefill WMMA arm. That is a slice, not
  a peephole. The K half was worth +2.6% of `tg128`, so V is the same order —
  enough to justify measuring, not enough to justify guessing.

## Tricks that may NOT port (measure before adopting)

Confirmed or suspected non-wins on RDNA3/HIP — do not port blindly:

- **HIP graphs / record-once tape** — probed negative twice. Slice 31: ROCm's
  per-dispatch cost is real GPU work, not launch overhead graphs can hide. P6:
  the host-side rebuild a tape would remove is 0.021 ms of a 4.98 ms token
  (0.4%), and the host already runs ahead of the device.
- **BDA arena addressing** — HIP raw pointers may make it redundant (evaluate).
- **rocBLAS f16 prefill** — measured _worse_ end-to-end than int8 WMMA (dequant
  tax + VRAM blowup, Slice 26); kept opt-in only.
- **Cooperative decode-once GEMM (non-pipelined)** — regressed vs the
  single-wave baseline (occupancy, Slice 28); only wins once async-pipelined.

## Parity checklist

- [x] **Quant coverage** — ✅ native decode + int8 + WMMA-prefill for all 24
      formats (**24/24 after R7**), ✅ MoE experts for every format a GGUF packs
      expert banks with, ✅ the id-indexed multi-slot MoE expert GEMV (**R8**,
      total over `moe_native_fmt`) and ✅ its bucket-sorted batched twin
      (**P2**, total over the same set, bit-identical). Paged MoE deliberately
      keeps the per-expert loop for its copy/compute overlap.
- [ ] **Attention** — ✅ tiled flash prefill (**P1**, 6.7–12.5× on the kernel,
      goldens unmoved); ✅ batched-prefetch decode attention (**P6**, bit-exact
      against `attn_pf=false`); ✅ **one-pass online-softmax +
      one-key-per-lane** split-KV (**P7**, 1.16× d0 / 1.48× @d4096, golden
      unmoved); ✅ **tuned split-KV chunking** (tgt_chunks=64, d0 +19%); ✅
      **vectorised `copy_strided`** (**P7b**, 30× kernel speedup, Q6_K pp512
      1.51×); ✅ **WMMA placeholder kernel** (br=64 bc=64, f16 Q conversion,
      opt-in). Still open: WMMA f16 intrinsics for Q·K + P·V;
      Q4_0/Q4_1/Q5_0/Q5_1 inline KV decode; `store_q8` kernel for Q8_0 KV write;
      split-KV coopmat decode (attn_flash_partial)
- [x] **Quant unpack** — ✅ branchless dword-wide Q6_K in the MoE expert decode
      (**P3**) and the two dense kernels (**P4**); `__builtin_memcpy` is the
      align-1 idiom that still emits `global_load_b128`
- [x] **Fusion** — ✅ GatedActFused (Slice 32), ✅ QkNormRope→KV-write (F1d), ✅
      RmsNormAdd (Slice 32), ✅ GatedRmsNorm (F1), ✅ Conv1dSilu, ✅
      RmsNorm→(Linear|MoeFfn) (Slice 32), ✅ MoeFfn→Add (F1c), ✅ Combined
      gate/up weight upload; ✅ **WMMA 2×2 tile threshold lowered** to N≥1024
      (+2.2% pp512). Q+K dispatch fusion probed negative (P7d, -1.6%). Missing
      vs Vulkan: `e2b_gate` (E2B only), `mul_sigmoid` (gemma4), **V-write
      peephole** (Linear→WriteKv epilogue on GEMV), **f16 A_GLOBAL GEMM** (skip
      int8 quant/dequant, pre-convert activations to f16)
- [x] **Device-side sampling** — ✅ argmax (**P7c**, 8.6×); ✅ **`Op::Sample`**
      (two-stage radix-select + nucleus/CDF); ✅ **`Op::ArgmaxProb`** (two-stage
      online-softmax reduction). Missing: `dg_eb_sample` (DiffusionGemma
      entropy-bound sampler), `eb_sample_reduce` (Backend trait method)
- [x] **Decode-replay tape** — evaluated (P6): worth ~0.4%, NOT built
- [ ] **KV-cache quantization** — missing: `store_q8`/`store_q8_dyn` (Q8_0 KV
      write), `store_turbo` (TurboQuant), `store_kv_dense`; ✅ seam gate
      `"rocm"` threaded into `kv_q8_backend`; `kv_turbo_ok`/`blk_ok` not yet
- [ ] **f16 A_GLOBAL + wide-tile GEMM** — missing: pre-convert activations to
      f16 via `store_f16`; WMMA kernel reading f16 A fragments from global (no
      As staging, 3 WGs/CU instead of 2); BM=64/32/16 row-tile ladder for
      small-m batched prefill
- [ ] **WMMA flash attention** — kernel body in place (scalar placeholder),
      needs `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` for Q·K + P·V
- [ ] **Inline KV decode** — ✅ Q8_0 planar decode in flash attention K+V
      staging loop (`q8kv_decode`, committed); remaining: Q4_0/Q4_1/Q5_0/Q5_1
      block-amortized inline decoders (`dqv4` pattern from Vulkan's
      `attn_partial.comp`)
- [ ] **MMQ all-expert MoE GEMM** — decode-once-reuse threadblock pattern
      (`matmul_mmq_experts`) replacing per-expert × per-token scalar GEMV
- [ ] **DeltaNet** — split/strided variants where they help
- [ ] **Memory** — BDA/staging/ReBAR/UMA/VRAM-guard where beneficial
- [ ] **Multi-GPU** — device pool, tensor-parallel, expert-parallel,
      pipeline-parallel, P2P, external semaphores, all-reduce (RCCL)
- [ ] **Perf ≥1.0×** — remaining 2.2× prefill / 2.3× decode gap to Vulkan
      primarily from three families: f16 A_GLOBAL GEMM (~2-4×), WMMA attention
      (~3.2×), and MMQ all-expert MoE (~3-5× for MoE models)

## Cross-backend note

Much of the non-kernel logic being ported here (op routing, peephole fusion,
KV-cache management, capability tiering, paging orchestration, sampling) is
parallel across the CPU/Vulkan/Metal/ROCm backends. A companion effort —
`docs/backend-unification-plan.md` — audits that duplication and plans a shared
seam so parity work is done once, not four times. Prefer landing a shared
implementation there over a fourth per-backend copy when the logic is
device-agnostic.
