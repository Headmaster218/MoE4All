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
  **22 of the 24 weight quants**
  (Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/TQ1_0/TQ2_0/Q2_0
  — only MXFP4/NVFP4 remain on the `dequant→f16` fallback); grid-underfill fixed
  across attention/WriteKv/GatedAct/RmsNorm/Argmax/QkNormRope; RmsNorm→Linear +
  Linear→Add fusion. ~1.9 → ~130 t/s (Qwen3-0.6B Q4_K_M, ~60× over naive, ~0.3×
  llama.cpp).
- **Prefill** — int8 **WMMA** matrix-core GEMM (RM×CN register tiling +
  software-pipelined Q4_K); ~4500 t/s (~0.2× llama.cpp).
- **Attention** — split-KV / flash-decoding (10.6× at depth), Causal/SWA/Canvas.
- **DeltaNet** — chunked/parallel prefill (88×) + column-parallel decode.
- **MoE** — int8 dp4a experts
  ({q80,q2k,q3k,q4k,q5k,q6k,q40,q41,q51,iq4nl,iq4xs,iq2xxs,iq2xs,iq2s,iq3xxs,iq3s,iq1s,iq1m,tq10,tq20,q20},
  gate/up × down) + **GPU-side top-k routing** (device-routed, no host
  readback).
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

## 1. Quant coverage — fast kernels for ALL 24 formats × all paths

**Nearly closed.** ROCm has native/int8 fast kernels for **22** formats
(Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/IQ2_XXS/IQ2_XS/
IQ2_S/IQ3_XXS/IQ3_S/**IQ1_S**/**IQ1_M**/**TQ1_0**/**TQ2_0**/**Q2_0**); only
**MXFP4** and **NVFP4** still fall back to the slow `dequant→f16` GEMV (256
threads — the pathology that made gemma-3's Q5_0 0.04× before it was covered).
Vulkan is native on **all 24** + floats, with a full `native_id`/`native_idm`
MoE-GEMV family (`crates/infr-vulkan/src/linear.rs:136-254`, `gemm.rs`).

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
- ✅ **R4 — IQ4_XS + IQ4_NL LANDED.** The first CODEBOOK formats on the ROCm
  fast paths: the 4-bit field is an index into the fixed 16-entry signed
  `kvalues_iq4nl` table, not a linear quant level, so the decoded value is one
  multiply (`scale · KV[idx]`) with NO offset — and the int8 tier therefore
  feeds the TABLE VALUE straight into dp4a and carries no
  ones-dot/min-correction term at all (the affine formats' second integer dot is
  simply absent). Landed: `deq_iq4nl`/`deq_iq4xs` native decode with their
  `infr_rocm::kernels::linear_iq4xs`, `infr_rocm::kernels::embed_iq4xs` and
  `infr_rocm::kernels::deqf16_iq4xs` instantiations; the
  `infr_rocm::kernels::linear_i8_iq4xs` / `i8acc_iq4xs` int8 dp4a GEMV; the
  `wmma_i8_iq4nl_*` / `wmma_i8_iq4xs_*` WMMA prefill tier in all three tiles
  from ONE shared `GEN_WMMA_IQ4` body (`XS` selects the scale source); and IQ4
  MoE experts. **The codebook is generated into the HIP module from
  `infr_gguf::dequant::KVALUES_IQ4NL`** — packed 4 signed bytes per `u32`, read
  by a `kv_iq4nl` word-select — rather than re-typed or uploaded as a constant
  buffer, so the device table cannot drift from the decode oracle and the gather
  stays pure ALU (a unit test unpacks the emitted words back to the host const).
  Measured on the RX 7900 XTX with Qwen3-0.6B: IQ4_XS decode **12.8 → 134.1
  t/s** (10.5×), prefill **1243 → 4505 t/s** (3.6×); IQ4_NL decode **12.8 →
  135.4 t/s** (10.6×), prefill **1231 → 4486 t/s** (3.6×); Q4_K_M control
  unmoved (126.6 → 125.9 tg, 4452 → 4455 pp). Both stay on the plain WMMA tier
  for the same reason Q5_K/Q2_K/Q3_K/the R3 round quants do (the Slice-27 pipe
  and Slice-28 coop kernels are Q4_K-only) — and for IQ4 the pipe would be
  prefetching the wrong thing anyway: the codebook gather is ALU-bound, not
  DRAM-bound (Vulkan's bytes-vs-speed sweep measured IQ4_XS at 4.25 bpw running
  1.55-2.1× SLOWER per dispatch than Q4_K's 4.5 bpw at matched shapes).
- ✅ **R5 — IQ2_XXS + IQ2_XS + IQ2_S + IQ3_XXS + IQ3_S LANDED.** The GRID
  (codebook-of-vectors) family — the last structurally-new decode shape. One
  axis past R4's codebook: the stored code indexes a table of packed signed-byte
  VECTORS (8 bytes per entry for the IQ2 grids, 4 for the IQ3 ones) and a
  separately packed sign bit negates each element, so a decoded value is
  `db · gv · sign` — still with NO offset, so the int8 tier feeds the (already
  signed) grid byte straight into dp4a with no ones-dot term, exactly as R4's
  codebook formats do. All five are 256-element super-blocks walked as 8
  sub-blocks × 4 groups × 8 elements; they differ only in the grid index width
  (8/9/10 bits), the sign source (`ksigns[7b]` for IQ2*XXS/IQ2_XS/IQ3_XXS, raw
  sign BYTES for IQ2_S/IQ3_S) and whether the 32-element block carries one scale
  or two (IQ2_XS/IQ2_S put a 4-bit magnitude on each half). **The grids are
  generated into the HIP module from `infr_core::iquant_grids`**
  (`iquant_grid_src`) — 17.1 KiB across six tables, parsed back by a unit test
  and required to BE the host statics. A device buffer was re-weighed at this
  size and rejected: it would need an extra pointer parameter on every IQ2/IQ3
  kernel, including the ones whose signatures come from macros shared with the
  affine formats, and it buys nothing at run time because on AMDGCN a
  module-scope `__device__ const` array already IS device global memory. (This
  is where HIP differs from Vulkan, which must mirror the same tables into LDS
  by hand — glslang/ACO materialize a dynamically-indexed `const` array into
  per-invocation scratch. There is no such lowering here, so no LDS mirror and
  no `grid_init()` barrier.) Landed per format: `infr_rocm::kernels::deq_iq3xxs`
  native decode with its `infr_rocm::kernels::linear_iq3xxs`,
  `infr_rocm::kernels::embed_iq3xxs` and `infr_rocm::kernels::deqf16_iq3xxs`
  instantiations; the `infr_rocm::kernels::linear_i8_iq3xxs` /
  `infr_rocm::kernels::i8acc_iq3xxs` int8 dp4a GEMV; the
  `infr_rocm::kernels::wmma_i8_iq3xxs_2x1` WMMA prefill tier in all three tiles;
  and grid MoE experts. All five tiers share ONE per-32-block decoder per format
  (`infr_rocm::kernels::wdec_iq3xxs` and its siblings), which is what lets ONE
  int8-GEMV body and ONE WMMA body (R6 renamed them `GEN_LINEAR_I8_WDEC` /
  `GEN_WMMA_WDEC`) serve the whole family and makes the tiers unable to drift.
  The grid entry and sign pattern are fetched once per GROUP OF 8 and peeled in
  registers — the hoisting Vulkan's grid GEMVs also needed; a per-element gather
  re-reads the same entry 8 times. Measured on the RX 7900 XTX with Qwen3-0.6B:
  IQ2_XXS decode **13.0 → 147.5 t/s** (11.3×), prefill **1245 → 4612 t/s**
  (3.7×); IQ3_XXS decode **13.5 → 148.1 t/s** (11.0×), prefill **1248 → 4587
  t/s** (3.7×); Q4_K_M control unmoved (126.7 → 127.3 tg, 4445 → 4464 pp). Both
  beat the Q4_K_M control at decode, which is the expected shape — at 2.1/3.1
  bpw they stream less than half the weight bytes. No IQ2_XS/IQ2_S GGUF exists
  standalone, but the two cached Qwen3-0.6B UD mixes contain all five formats
  between them (UD-IQ3_XXS is IQ3_XXS + IQ3_S + IQ2_S + IQ2_XS + IQ4_XS), so
  every kernel runs in a real coherent generation. All five stay on the plain
  WMMA tier for the same reason every format after Q4_K does — and R4's extra
  argument against the software-pipelined tier applies with MORE force here: a
  fixed-shape prefetch cannot reach the GRID reads, whose addresses are not
  known until the block's own indices have been fetched and unpacked. The decode
  is not purely DRAM-bound either — at 2.06 bpw IQ2_XXS streams under half
  Q4_K's 4.5 bpw of weight bytes yet runs only 1.16× its decode rate. R5 also
  fixed a STALE gate outside the backend: `rocm_moe_pageable`
  (`infr-llama/src/seam/mod.rs`) still listed R0's `{Q8_0, Q4_K, Q6_K}` while
  `moe_native_fmt` had grown to 16 formats, so a paged expert bank in any of the
  other 13 was rejected at LOAD time even though its kernels existed — which is
  why R2's Q2_K/Q3_K work never actually reached llama4-Scout's paged banks.
  With it synced, `Qwen3.6-35B-A3B-UD-IQ3_S` (IQ2_S gate/up, IQ3_S + IQ4_XS
  down) runs coherently on the 24 GB card through the expert pager at 6.6 t/s —
  it previously refused to load at all.
- ✅ **R6 — IQ1_S + IQ1_M + TQ1_0 + TQ2_0 + Q2_0 LANDED.** The ternary / 1-bit
  tail, and with it native coverage of **22 of the 24** weight formats. Two
  families in one slice.

  **IQ1_S / IQ1_M** close the grid family and add the one decode shape nothing
  in R1-R5 had: a per-group fractional **ADDEND**. The element is
  `dl · (gv + delta)` with `delta = ±IQ1S_DELTA` (±0.125) — not the affine
  `d · code + m`, whose offset sits OUTSIDE the code's scale — so the
  element-wise decode gets its own `fina` helper alongside `fin`/`finc`/`fing`.
  Both share the 2048-entry `IQ1S_GRID` at an 11-bit index (the widest in the
  family); IQ1_S carries one scale + one delta sign per 32 elements, IQ1_M
  carries a scale per **16** and a delta sign per **8**, and IQ1_M has no
  standalone `d` field at all (its f16 bits are the top nibbles of the four u16
  scale words — the layout `infr_core::decode_spec::ScaleEnc::Iq1mSplitF16`
  describes). **The int8/WMMA tiers fold the addend into the code**: every IQ1
  grid byte is −1/0/+1 and 0.125 is a power of two, so `dl · (gv + delta)` is
  EXACTLY `(dl · 0.125) · (8·gv ± 1)` — a re-association, not an approximation
  (both halves are exact in binary and `dl` is never near subnormal). That keeps
  the family on the no-ones-dot seam and, crucially, handles IQ1_M's per-8 delta
  sign for free, where a delta-split correction term would have needed a
  ones-dot per GROUP OF 8. `|code| ≤ 9`, well inside the int8 WMMA operand.

  **TQ1_0 / TQ2_0 / Q2_0** are the simplest family in the set: one f16 `d` per
  block, no grid, no codebook, no sign field, no sub-block scales, and a small
  unsigned code with a CONSTANT `−1` offset (`y = (code − 1) · d`). The `−1` is
  folded straight into the stored signed code, so they too carry no ones-dot.
  They differ only in packing: TQ1_0 packs FIVE base-3 digits per byte
  (`digit = ((u8)(byte · 3ⁿ) · 3) >> 8`, with the u8 wrap load-bearing) over a
  three-segment element order; TQ2_0 packs 4 elements per byte at 2 bits over
  two 32-byte chunks × 4 shifts × 32; and **Q2_0 is infr's own format** — the
  only **64-element** block in the covered set, so one activation 32-block is
  HALF a weight block and its `wdec` indexes by `blk>>1` where every other
  format uses `blk>>3`. infr remains the only GPU engine that runs Q2_0.

  Landed per format: `infr_rocm::kernels::deq_iq1s` (and the four siblings)
  bit-faithful device decode with its `infr_rocm::kernels::linear_iq1s`,
  `infr_rocm::kernels::embed_iq1s` and `infr_rocm::kernels::deqf16_iq1s`
  instantiations; the `infr_rocm::kernels::linear_i8_iq1s` /
  `infr_rocm::kernels::i8acc_iq1s` int8 dp4a GEMV; the
  `infr_rocm::kernels::wmma_i8_iq1s_2x1` WMMA prefill tier in all three tiles;
  and IQ1/ternary MoE experts (gate/up, down, host- and device-routed). All ten
  formats on the shared per-32-block decoder seam now go through ONE body per
  tier, renamed for what it actually selects: `GEN_LINEAR_I8_WDEC`,
  `GEN_I8ACC_WDEC` and `GEN_WMMA_WDEC` (was `..._IQG`, which read as "IQ grid"
  and no longer describes the ternary members). The defining property of the
  seam is that the decoded code is ALREADY SIGNED and so no member needs an
  `isum` ones-dot — the grid byte for R5, the ×8 delta fold for IQ1, the folded
  `−1` for ternary. `IQ1S_GRID` is generated into the module alongside R5's five
  tables (`iquant_grid_src`, now 33.1 KiB across seven tables — `g_iq1s` is 16
  KiB of that) and so is `IQ1S_DELTA`, both parsed back by the unit test and
  required to BE the host statics.

  All five stay on the **plain** WMMA tier, same as every format after Q4_K (the
  Slice-27 pipe and Slice-28 coop kernels are Q4_K-only, and coop is a measured
  gfx1100 regression regardless). R5's argument against the pipe holds unchanged
  for IQ1 — the critical-path reads are grid gathers whose addresses are unknown
  until the block's own indices are unpacked, so a fixed-shape prefetch cannot
  reach them. For ternary the argument inverts to the same conclusion: there is
  no table and the whole weight tile is 32-48 bytes per 32 elements at a
  statically known offset, so a prefetch has nothing left to hide.

  Measured on the RX 7900 XTX (warmed, `-p 0 -n 64 -r 2` / `-p 512 -n 0 -r 2`):

  | model                   | format | decode t/s              | prefill t/s             |
  | ----------------------- | ------ | ----------------------- | ----------------------- |
  | Qwen3-0.6B UD-IQ1_S     | IQ1_S  | 18.1 → **149.6** (8.3×) | 1507 → **4586** (3.0×)  |
  | Qwen3-0.6B UD-IQ1_M     | IQ1_M  | 19.8 → **155.8** (7.9×) | 1659 → **4578** (2.8×)  |
  | TriLM-3.9B              | TQ1_0  | 1.5 → **17.6** (11.7×)  | 71.5 → **1240** (17.3×) |
  | TriLM-3.9B              | TQ2_0  | 1.5 → **18.5** (12.3×)  | 69.5 → **1505** (21.6×) |
  | Ternary-Bonsai-8B (g64) | Q2_0   | 0.7 → **97.3** (139×)   | 11.7 → **1016** (87×)   |
  | Qwen3-0.6B Q4_K_M       | —      | 126.9 → 127.6 (control) | 4429 → 4424 (control)   |

  Both IQ1 mixes now BEAT the Q4_K_M control at decode, the expected shape at
  1.7/1.9 bpw. The Q2_0 8B result is the outlier for a structural reason: an 8B
  model at 2.25 bpw needed a ~5.5 GiB `dequant→f16` cache before, which no
  longer exists at all. The TriLM 3.9B decode rate lines up with model size
  against the 0.6B control (127 ÷ 6.5 ≈ 19.5 t/s), so it is size-bound rather
  than kernel-bound.

  End-to-end: the four cached ternary models (bitnet-b1.58-large TQ2_0,
  TriLM-3.9B TQ1_0 and TQ2_0, Ternary-Bonsai-8B Q2_0) all agree with the CPU
  reference on the top-1 next token for "The capital of France is", whole-vocab
  cosine 0.9997-0.99999; Ternary-Bonsai-8B answers "Paris." at `--temp 0`. The
  two IQ1 mixes are degenerate on a 0.6B model at 1.7/1.9 bpw and diverge from
  the CPU reference (cosine ~0.72) — but they did so BEFORE this slice too
  (baseline cosine 0.689/0.714 on the `dequant→f16` path), so R6 slightly
  IMPROVES them and the divergence is model conditioning, not a kernel fault.
  The TQ1_0/TQ2_0 GGUFs are base models with no chat template, so `infr run`
  cannot drive them; the logits comparison above is the substitute (and a
  stronger one).

  Parity: shared decode sweep vs `dequant_block` green at the unchanged 2e-2
  tolerance — IQ1_S 5.3e-3/3.7e-3, IQ1_M 3.8e-3/4.4e-3, TQ1_0 4.4e-3/4.0e-3,
  TQ2_0 3.7e-3/3.9e-3, Q2_0 2.7e-3/3.5e-3 at m=1/m=16, with Q2_K's 7.6e-3 still
  binding. (The ternary three had been ~3e-7 on the f32-exact dequant path;
  landing in the int8 band is itself the evidence the new kernels are running.)
  New int8-GEMV, WMMA, EmbedGather and MoE-expert cases per format. EmbedGather
  is again the load-bearing one and the only element-wise comparison in the
  suite: the ternary formats come in at **exactly 0** error (their codes times
  an f16 `d` are exactly representable), and IQ1_S/IQ1_M at 2.3e-4 against a
  2e-3 bound — a mis-scaled or wrong-signed delta is at most an eighth of one
  term and hides in any dot, but not here. `rocm_seam` 9/9 with the qwen3 golden
  `0xfd63781ea3bfa785` unmoved. infr-rocm 119 (13 unit + 103 parity + 3 shared),
  workspace 550.

- **Extend native decode GEMV + int8 dp4a + WMMA prefill** to the remaining 2:
  `MXFP4, NVFP4` (+ `Bf16` weights). Neither is a codebook-of-vectors; both are
  FP4 microscaling formats (MXFP4's per-32 E8M0 shared exponent, NVFP4's four
  UE4M3 sub-block scales), so the new work is the scale ENCODING rather than the
  code layout, and the signed E2M1 table means they join the no-ones-dot `wdec`
  seam rather than the affine one. Both currently land at ~3e-7 on the f32-exact
  host path, so they are a perf item, not a correctness one. Reuse
  `infr_gguf::dequant` for bit-faithful decode; mirror the per-32-block
  `infr_rocm::kernels::wdec_tq20` + `GEN_LINEAR_I8_WDEC` pattern in
  `kernels.rs`.
- **MoE experts beyond the current set** — extend `moe_ffn_expert*`/`moe_*_i8*`
  to the remaining formats. The **escape hatch is taken** (R3 measured it, R4
  re-measured it): the Phase-3 `moe_ffn_expert_<gu>_<dn>` cross product is not
  complete over `moe_native_fmt`. Going 6×6 → 9×9 (81 pairs/macro) cost **+1.1 s
  of COLD hiprtc**, so `moe_expert_kernel` and its routed twin return `Option`
  and only the **reachable** pairs are instantiated — now **116**:
  `{q80,q2k,q3k,q4k,q5k,q6k}²` (36), `{q40,q41,q51} × {q40,q41,q51,q80}` (12),
  `{iq4nl,iq4xs} × {iq4nl,iq4xs,q4k,q5k,q6k,q80}` (12), `{q2k,q3k} × {iq4nl}`
  (2, llama.cpp's `convert_incompatible_tensor` rewrite),
  `{iq2xxs,iq2xs,iq2s,iq3xxs,iq3s} × {iq2s,iq3xxs,iq3s,iq4nl,iq4xs,q4k,q6k}`
  (35, R5 — grid quants are gate/up banks only, since an IQ ftype always bumps
  `ffn_down` to something wider; the cached `Qwen3.6-35B-A3B-UD-IQ3_S` is
  exactly `("iq2s","iq3s")` + `("iq2s","iq4xs")`),
  `{iq1s,iq1m} × {iq1s,iq1m,iq2xxs,iq2s,iq3s,iq4xs,q4k,q6k}` (16, R6 — IQ1
  quants ARE also a legal `dn`, unlike the grid quants: the `dn` set is read off
  the two cached UD-IQ1 mixes, which leave 18 of 28 `ffn_down` tensors at the
  gate/up type and boost the rest to IQ2_S/IQ3_S, plus the wider bumps a big-MoE
  IQ1 mix reaches) and the three ternary SELF pairs
  `{(tq10,tq10),(tq20,tq20),(q20,q20)}` (3, R6 — TQ1_0/TQ2_0/Q2_0 are
  whole-model conversion targets for a natively ternary checkpoint, not ftype
  mixes, so there is no `ffn_down` bump to model and nothing mixes them with
  another family in either direction). R6's cold-hiprtc re-measurement (backend
  init + a 1-token bench, `~/.cache/comgr` AND `~/.cache/infr/rocm-module-*.bin`
  cleared, 3 reps each): **9.06-9.12 s** at R5's 97 pairs → **10.98-11.27 s**
  once R6's 60 DENSE kernels + the 16 KiB `g_iq1s` table are added at the same
  97 pairs → **11.42-11.56 s** at the shipped 116. So the 19 new pairs (38
  kernels) cost ~0.4 s (~11 ms each) and the dense kernels — the actual feature
  — are ~2.0 s of it; the full 21×21 would have added 325 more cells (~3.6 s)
  for nothing. **Warm-cache startup is unchanged.** Absent pairs fall back to
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
      formats (**22/24 after R6**; MXFP4 + NVFP4 left); MoE experts for all
      formats; `native_id`/`idm` MoE GEMV family
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
