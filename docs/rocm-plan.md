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
  **all 24 weight quants**
  (Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/TQ1_0/TQ2_0/Q2_0/MXFP4/NVFP4
  — nothing quantized is left on the `dequant→f16` fallback, which now serves
  only the dense float dtypes F32/BF16); grid-underfill fixed across
  attention/WriteKv/GatedAct/RmsNorm/Argmax/QkNormRope; RmsNorm→Linear +
  Linear→Add fusion. ~1.9 → ~130 t/s (Qwen3-0.6B Q4_K_M, ~60× over naive, ~0.3×
  llama.cpp).
- **Prefill** — int8 **WMMA** matrix-core GEMM (RM×CN register tiling +
  software-pipelined Q4_K); ~4500 t/s (~0.2× llama.cpp).
- **Attention** — split-KV / flash-decoding (10.6× at depth), Causal/SWA/Canvas.
- **DeltaNet** — chunked/parallel prefill (88×) + column-parallel decode.
- **MoE** — int8 dp4a experts
  ({q80,q2k,q3k,q4k,q5k,q6k,q40,q41,q51,iq4nl,iq4xs,iq2xxs,iq2xs,iq2s,iq3xxs,iq3s,iq1s,iq1m,tq10,tq20,q20,mxfp4,nvfp4}
  — every weight quant except Q5_0, which no shipped GGUF uses for expert banks;
  gate/up × down) + **GPU-side top-k routing** (device-routed, no host
  readback) + the **id-indexed multi-slot expert GEMV** (R8: all `rows × n_used`
  slots in ONE dispatch per stage instead of a serialized per-expert host loop —
  Qwen3-30B-A3B `pp512` 104 → 254 t/s, `tg64` 35 → 42, bit-identical to the loop
  it replaces).
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

**FORMAT COVERAGE COMPLETE (R7).** ROCm has native decode + int8 dp4a GEMV +
WMMA prefill + MoE expert kernels for **all 24** weight quant formats —
`infr_core::decode_spec::WEIGHT_QUANTS` in full. Nothing quantized takes the
`dequant→f16` fallback any longer (the 256-thread GEMV pathology that made
gemma-3's Q5_0 0.04× before it was covered).
`native_decode_is_total_over_every_gguf_weight_dtype` (exec.rs) is the standing
proof: an EXHAUSTIVE `match` over `DType` that forces every variant to be either
natively decoded or an explicitly-reasoned exclusion, so this cannot silently
regress and a new dtype cannot be added without deciding.

**SECTION COMPLETE (R8).** The id-indexed MoE expert GEMV tier landed too, so
every format now has kernels on every tier ROCm has, and the MoE decode path is
no longer a serialized per-expert host loop. See the R8 bullet below for the
tier, its bit-identity gate, and where the remaining MoE prefill gap actually
lives (per-slot weight traffic, not launches).

**What "complete" does NOT mean.** Two things are still open, and neither is a
format gap:

- **The `moe_ffn_expert_<gu>_<dn>` cross product is deliberately partial** (118
  reachable pairs, not 24²) — an A/B-only path, see that bullet.
- **The DENSE FLOAT weights (F32/BF16) still take a host convert→f16 at load.**
  That is a format cast, not a quant decode, and R7 measured that it costs
  nothing at run time (see the Bf16 note under R7) — but the one-time load pass
  is real and F16 pays it identically.

Vulkan reference for this section: `crates/infr-vulkan/src/linear.rs:136-254`,
`gemm.rs`.

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
  (8/9/10 bits), the sign source (`ksigns[7b]` for IQ2\*XXS/IQ2_XS/IQ3_XXS, raw
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

- ✅ **R7 — MXFP4 + NVFP4 LANDED; quant coverage COMPLETE at 24/24.** The FP4
  microscaling pair. Neither is a codebook-of-vectors and neither is a new code
  layout: both index the fixed 16-entry signed **E2M1** table `KVALUES_MXFP4`
  (`{0,±1,±2,±3,±4,±6,±8,±12}`), so the decoded value is R4's one multiply
  `d · KV[idx]` with no offset, the table value IS the signed dp4a operand, and
  both join the no-ones-dot `wdec` seam rather than the affine one — exactly as
  R6's report predicted. **The new thing is the scale ENCODING**, and it is the
  whole slice:

  |       | block           | scale                              |
  | ----- | --------------- | ---------------------------------- |
  | MXFP4 | 17 B / 32 elems | ONE E8M0 byte, `d = 2^(e − 128)`   |
  | NVFP4 | 36 B / 64 elems | FOUR UE4M3 bytes, one per 16 elems |

  **E8M0** is a bare 8-bit exponent — no sign, no mantissa — so the value is a
  pure power of two. Decoded from `infr_gguf::dequant::e8m0_to_fp32_half` (not
  from the OCP spec), including its two-case form: the byte drops into the f32
  exponent field as `e − 1` for `e ≥ 2`, but `e ∈ {0,1}` take the SUBNORMAL bit
  pattern `0x00200000 << e`, where the common-case formula would produce ±inf.
  **UE4M3** is a genuine FP8 (4 exponent bits biased 7, 3 mantissa bits) HALVED:
  `0.5 · 2^(e−7) · (1 + m/8)`, with an `e == 0` subnormal branch and the two
  reserved codes `0x00`/`0x7F` decoding to 0.0. Transcribed from
  `infr_gguf::dequant::ue4m3_to_fp32` case for case; the `·0.5` tail and the
  `0x7F` hole are both part of the oracle. Note this codebase's NVFP4 has **no
  per-tensor second-level scale** — the oracle's block is four per-16 UE4M3
  scales and nothing else.

  **The power-of-two scale makes the int8 re-association EXACT**, which is the
  property R6 exploited for IQ1: `sc · Σ(code·a)` reproduces the oracle's
  `Σ((sc·code)·a)` term for term when `sc·code` consumes no mantissa bits. For
  NVFP4 the same holds for a different reason — a 3-bit-mantissa scale against a
  ≤4-bit code needs at most 7 significand bits, well inside f32's 24.

  NVFP4's 64-element block additionally makes it the only covered format that is
  BOTH a two-activation-blocks-per-header format (Q2_0's `blk>>1` shape) AND
  split-scale (`s0`/`s1` per 16 elements, IQ2_XS's shape). Its nibble split is
  also the one departure from every other nibble format in the file: within a
  16-element sub-block the low nibbles are elements 0..7, not 0..15.

  Landed per format: `infr_rocm::kernels::deq_mxfp4` / `deq_nvfp4` bit-faithful
  device decode (via R4's `finc` — one multiply, no addend) with their
  `linear_mxfp4`, `embed_mxfp4` and `deqf16_mxfp4` instantiations; the
  `linear_i8_mxfp4` / `i8acc_mxfp4` int8 dp4a GEMV;
  `wmma_i8_mxfp4_{1x1,2x1,2x2}` WMMA prefill (plain tier, same reason as every
  format after Q4_K); and fp4 MoE experts (gate/up, down, host- and
  device-routed). All three tiers run off ONE `wdec_mxfp4` / `wdec_nvfp4` per
  format, so they cannot drift. **The E2M1 codebook is generated into the
  module** from `infr_gguf::dequant::KVALUES_MXFP4` beside R4's `KVALUES_IQ4NL`
  — the two now share one emitter (`kv16_codebook_src`), and the unit test
  unpacks BOTH back to their host consts and asserts the two tables differ, so
  neither accessor can be wired to the wrong format. Registered in all five
  routing tables + the MoE name mappers, and `rocm_moe_pageable`
  (`infr-llama/src/seam/mod.rs`) syncs in the same step — the entry that matters
  most, since `gpt-oss` ships its expert banks as MXFP4 and nothing else.

  Measured on the RX 7900 XTX (warmed, `-p 0 -n 64 -r 2` / `-p 512 -n 0 -r 2`):

  | model                   | format | decode t/s             | prefill t/s            |
  | ----------------------- | ------ | ---------------------- | ---------------------- |
  | Llama-3.2-1B pure MXFP4 | MXFP4  | 4.2 → **66.1** (15.7×) | 243 → **3863** (15.9×) |
  | Llama-3.2-1B Q4_K_M     | —      | 53.9 → 53.9 (control)  | 3638 → 3648 (control)  |

  MXFP4 now BEATS the Q4_K_M control at decode (66.1 vs 53.9), the expected
  shape at 4.25 vs 4.58 bpw. **Model provenance matters here**: the only cached
  MXFP4 GGUF is `ggml-org/gpt-oss-20b-MXFP4`, and **infr does not support the
  `gpt-oss` architecture**, so it cannot be run end to end at all. The measured
  model above is `Llama-3.2-1B-Instruct-BF16` locally requantized with
  `llama-quantize --pure … MXFP4_MOE` (all 113 weight tensors MXFP4, 4.25 bpw) —
  a real GGUF on a supported arch, not a synthetic. It answers "Paris" at
  `--temp 0`, IDENTICALLY before and after the slice. **NVFP4 has no GGUF at
  all** — llama.cpp has no NVFP4 ftype yet and nothing is cached — so it rests
  entirely on the parity tests.

  **Bf16 weights: no kernel, by measurement.** A BF16 weight is host-converted
  to f16 once at load and then rides `linear_f16` / the rocBLAS f16 GEMM.
  Measured on `Llama-3.2-1B-Instruct` in BOTH formats (same model, same 2.36
  GB): decode **4.2 t/s BF16 vs 4.2 t/s F16**, prefill **~312 vs ~316 t/s** (3
  reps each, inside noise). So BF16 is already ON the F16 path with no decode
  penalty, and a native `deq_bf16`/`linear_bf16` would have identical memory
  traffic (2 B/elem) and identical arithmetic — there is no int8/WMMA tier for a
  float weight to move to, so it could not be faster. It would also not be
  numerics-neutral: R1-R7 are all bit-faithful to the `dequant→f16` value, and a
  native bf16 path is only bit-faithful if it rounds bf16→f16 anyway, at which
  point it saves nothing but the host pass. The narrowing is close to free in
  practice — f16 has 11 significand bits to bf16's 8, so **bf16→f16 is EXACT for
  every value in f16's normal range**; scanning both cached BF16 GGUFs found max
  |w| = 1.23 (Qwen3-0.6B, 596 M weights) and 1.20 (Llama-3.2-1B, 1.24 G),
  **zero** overflows, and only 1127 / 3372 weights below 2^-24 (≈2e-6 of the
  tensor) that flush to zero. **Flagged for the lead, NOT fixed here:** the one
  real cost is load time — `Llama-3.2-1B` takes 3.53 s (F16) / 3.55 s (BF16) to
  load+decode 1 token vs 0.76 s for Q8_0, because `dequant_weight_or_cache`
  round-trips the dense float weights through f32 on the host. F16 pays exactly
  the same, so the fix is a format-agnostic raw-upload path for the dense
  floats, not a bf16 quant kernel.

  Parity: shared decode sweep vs `dequant_block` green at the unchanged 2e-2
  tolerance — MXFP4 2.4e-3/4.0e-3, NVFP4 6.0e-3/5.3e-3 at m=1/m=16, with Q2_K's
  7.6e-3 still binding. Both had been ~3e-7 on the f32-exact dequant path;
  landing in the int8 band is itself the evidence the new kernels are the ones
  running, and it means all 24 formats are now measured against the SAME
  activation-quant bound with nothing left on the weaker comparison. New
  int8-GEMV, WMMA, EmbedGather and MoE-expert cases per format. **EmbedGather is
  the load-bearing one and more sharply so than in any prior slice**: a
  mis-decoded exponent is a FACTOR-OF-TWO error, which a coarse relative bound
  over a 256-deep dot can swallow on a subset of blocks but an element-wise
  comparison cannot. Both land at **effectively 0** (NVFP4 exactly 0; MXFP4
  1.1e-36 absolute — the subnormal `e = 1` blocks, i.e. proof that branch is
  live rather than ±inf) against a 2e-3 bound. The fp4 block builders in
  `tests/parity.rs` deliberately do NOT use `synth_weight`'s constant scale:
  they overwrite the scale bytes with a varying valid encoding (MXFP4 cycling
  `e ∈ {126..129}` plus `e = 1` every 11th block; NVFP4 four DIFFERENT
  sub-scales per block plus a `0x7F` hole), because a constant scale would make
  both a broken subnormal branch and a broadcast-`s0`-over-`s1` bug invisible.
  Verified by tripwire: replacing `s1` with `s0` in `wdec_nvfp4` moves the int8
  GEMV to **rel 0.66** and fails three cases. `rocm_seam` 9/9 with the qwen3
  golden `0xfd63781ea3bfa785` unmoved. infr-rocm 128 (14 unit + 111 parity + 3
  shared), workspace 551.

- **MoE experts beyond the current set** — extend `moe_ffn_expert*`/`moe_*_i8*`
  to the remaining formats. The **escape hatch is taken** (R3 measured it, R4
  re-measured it): the Phase-3 `moe_ffn_expert_<gu>_<dn>` cross product is not
  complete over `moe_native_fmt`. Going 6×6 → 9×9 (81 pairs/macro) cost **+1.1 s
  of COLD hiprtc**, so `moe_expert_kernel` and its routed twin return `Option`
  and only the **reachable** pairs are instantiated — now **118**:
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
  gate/up type and boost the rest to IQ2*S/IQ3_S, plus the wider bumps a big-MoE
  IQ1 mix reaches) and the three ternary SELF pairs
  `{(tq10,tq10),(tq20,tq20),(q20,q20)}` (3, R6 — TQ1_0/TQ2_0/Q2_0 are
  whole-model conversion targets for a natively ternary checkpoint, not ftype
  mixes, so there is no `ffn_down` bump to model and nothing mixes them with
  another family in either direction) and the two fp4 SELF pairs
  `{(mxfp4,mxfp4),(nvfp4,nvfp4)}` (2, R7 — here the rule is WRITTEN DOWN rather
  than inferred: llama.cpp's `llama_tensor_get_type` handles
  `LLAMA_FTYPE_MOSTLY_MXFP4_MOE` before every other branch as one unconditional
  arm, "MoE tensors (`ne[2] > 1`) → MXFP4, other tensors → Q8_0", with no
  `use_more_bits` and no `ffn_down` bump, so gate/up and down are the same type
  by construction; the cached `ggml-org/gpt-oss-20b-MXFP4` is exactly that — all
  72 `ffn*{gate,up,down}_exps`MXFP4 and every dense tensor Q8_0). R6's
  cold-hiprtc re-measurement (backend init + a 1-token
  bench,`~/.cache/comgr`AND`~/.cache/infr/rocm-module-\*.bin`cleared, 3 reps
  each): **9.06-9.12 s** at R5's 97 pairs → **10.98-11.27 s** once R6's 60 DENSE
  kernels + the 16 KiB`g_iq1s`table are added at the same 97 pairs →
  **11.42-11.56 s** at the shipped 116. So the 19 new pairs (38 kernels) cost
  ~0.4 s (~11 ms each) and the dense kernels — the actual feature — are ~2.0 s
  of it; the full 21×21 would have added 325 more cells (~3.6 s) for nothing. R7
  re-measured the same way: **11.45-12.24 s** at R6's 116 pairs → **13.46-14.00
  s** at the shipped 118, so R7's 28 kernel bodies cost ~**+2.0 s** and its 2
  new pairs are ~0.02 s at R6's measured ~11 ms/cell, i.e. below the noise floor
  — essentially all of the delta is the dense kernels. **Warm-cache startup is
  unchanged** (0.50-0.51 s before and after). Absent pairs fall back to the
  dequant→f16`moe_ffn_expert`path, which costs nothing real: those kernels only
  run under`INFR_ROCM_NO_I8`(the default int8 expert path uses the
  per-FORMAT`moe_gate_up_act_i8_<gu>`/`moe*down_i8*<dn>`kernels, still total
  over`moe_native_fmt`), and that switch's comparand IS the f16 path. When
  adding a format, extend `MOE_EXPERT_PAIRS`(exec.rs test module) with only the
  pairs a real GGUF can produce;`moe_expert_pair_tables_agree` pins both mappers
  to it.
- ✅ **R8 — the id-indexed MULTI-SLOT MoE expert GEMV LANDED**
  (`moe_gate_up_act_i8_idm_*` / `moe_down_i8_idm_*` / `moe_accum_idm`, total
  over `moe_native_fmt`'s 23 formats). **§1 IS NOW CLOSED.**

  **The gap was the MULTI, not the id.** ROCm's Slice-38 `*_routed_*` kernels
  already resolved the expert bank in-kernel from `route_ids[slot]` — ROCm never
  lacked id indexing. What it lacked was a dispatch shape: the executor drove
  them from a host `for row { for k in 0..n_used { … } }` loop, so one MoE layer
  cost `1 + 3·rows·n_used` launches AND ran the selected experts SERIALLY (all
  slots shared one `h` scratch, so expert k+1 could not start until expert k's
  down GEMV retired). At qwen3moe's 48 layers × 8 experts that is ~1150 launches
  per decoded token, each filling a fraction of the device. At `pp512` it is
  ~590 000 launches per chunk, which is why ROCm MoE prefill sat at ~100 t/s
  while the same box does 4500 t/s dense.

  The new kernels take the whole `[rows, n_used]` slot grid in ONE dispatch per
  stage (`blockIdx.y` = Vulkan's flat `slot_global`, `row = slot / n_used`),
  with per-block arithmetic IDENTICAL to the `*_routed_*` twin — same `i8acc_*`
  decode+dot, same `wave_sum32`, same weight fold. Five dispatches per row-chunk
  regardless of `n_used`.
  - **Bit-identical to the tier it replaces**, not merely within tolerance:
    `moe_ffn_id_tier_matches_the_serial_tier_bitwise` runs the same problem at
    `moe_id_rows = 128` and `= 0` (the tier off) and compares `f32::to_bits`.
    That is only possible because the down GEMV does NOT `atomicAdd` into `dst`
    the way the serial kernel does — concurrent slots would make the f32
    summation ORDER nondeterministic and the golden hash would drift between
    runs. It writes `y[n_slots, ne]` and `moe_accum_idm` sums the slots in
    ASCENDING order onto a zero seed, which is exactly the sequence the serial
    loop's atomics produced against a zeroed `dst`.
  - **Addressing: a 64-bit BYTE offset on a 64-bit pointer**,
    `base + (long)expert_id * bstride`, with `bstride` the host-computed
    per-expert byte stride passed as `i64`. This is the Vulkan u64/BDA finding
    transplanted (its `native_gemv_id` STREAMED build had to move to
    `uint64_t(ids[slot]) * uint64_t(stride)` after a u32 element-space multiply
    went coherent-but-wrong past ~102 Scout-sized slots). HIP pointers are
    already 64-bit and `long` is 64-bit on AMDGCN, so the multiply is 64-bit BY
    CONSTRUCTION — but only while `bstride` stays a `long` PARAMETER; narrowed
    to `int` it wraps at 2 GiB, which Scout's 2.7 GiB Q4_K down bank clears on
    its own. `moe_id_multi_strides_are_64_bit` pins the declarations and the
    address expressions against the emitted source, and
    `moe_id_multi_host_strides_exceed_u32_without_wrapping` pins the host
    arithmetic at Scout's shape (expert 127's base = 2 996 305 920 B).
  - **Routing: the id tier takes the RESIDENT int8 expert path at every `m`**,
    and that is a deliberate departure from Vulkan's `moe_small_m = 8`
    crossover, not an oversight. Vulkan's threshold chooses between the id tier
    and a bucket-sorted batched expert GEMM; ROCm HAS NO BATCHED EXPERT PATH, so
    above a threshold it would fall back to the very loop the id tier replaces,
    over the SAME per-slot weight traffic (the serial loop already re-reads each
    routed expert's full bank per `(row, slot)`) — strictly worse at every `m`.
    Measured, `pp512` on Qwen3-30B-A3B Q4_K_M: **104 t/s** with the tier capped
    at Vulkan's 8 rows' worth of behaviour (`moe_id_rows = 0`, i.e. the old
    loop) vs **254 t/s** with the tier on. So `kernels.rocm.moe_id_rows` is a
    SCRATCH bound (rows per dispatch), not a crossover, and it reuses
    `infr_core::tier::EnvRows` for the clamp policy rather than a fourth copy of
    a threshold. Row-chunk sweep at `pp512` (3 reps): 8 → 237.1, 32 → 249.0, 128
    → **253.3**, unchunked → 261.2 t/s. The default is 128 because the remaining
    +3% is not free: at `-p 1024` an unchunked (or 512-row) chunk asks ~50-100
    MiB of pool on top of a 17 GiB weight set and `BufferPool`'s `hipMalloc`
    FAILS — reproduced, and today that aborts the process. A shape-aware 16 MiB
    scratch cap sits under the knob so no `ne`/`n_ff_exp`/ `n_used` combination
    can walk into that through the default (with it, even `moe_id_rows = 1024`
    runs `pp1024` cleanly at 231.3 t/s).
  - **The PAGED path deliberately stays on the per-expert loop.** Vulkan needs a
    device LUT for its paged id-GEMV (`native_gemv_id*.comp`'s `-DPAGED` build
    reads `lut[window + ids[slot]]`, a per-layer window into a slot-index tape)
    precisely because a paged bank's slot index is not the model's expert id and
    its routing happens on the GPU. ROCm's pager routes on the HOST — it must
    know which experts to page in before it can page them — so it already
    resolves each slot's arena pointer in Rust and needs no LUT at all. What it
    would LOSE is Slice 36's copy/compute overlap: the loop exists so expert i's
    GEMV runs while expert i+1's H2D fill is in flight on the copy stream, and
    one collapsed dispatch would serialize every fill ahead of all compute. The
    trade is a page-in (hundreds of µs to ms per expert bank) against ~20
    launches (~50 µs), so the loop wins. Verified unchanged end to end:
    Qwen3.6-35B-A3B-UD-IQ3_S through the pager, 6.5 → 6.7 t/s decode (noise),
    still coherent at `--temp 0`.
  - Scoped OUT, with the reason: the `INFR_ROCM_NO_I8` path and the dequant→f16
    fallback keep the serial loop. Both are A/B comparands whose whole job is to
    be the OTHER path, and neither ships;
    `moe_ffn_serial_tier_still_matches_cpu_with_i8_off` keeps the first one
    honest so it cannot rot into a second untested tier.

  Cold hiprtc (backend init + a 1-token bench, `~/.cache/comgr` AND
  `~/.cache/infr/rocm-module-*.bin` cleared, 3 reps): R7 baseline 14.18-15.58 s
  → R8 (47 new kernels) 16.07-19.47 s, ~+2 s. Warm-cache start UNCHANGED at
  0.49-0.52 s.

  Measured (RX 7900 XTX, `INFR_DEV=rocm`, warmed):

  | shape                                | before | after  |
  | ------------------------------------ | ------ | ------ |
  | Qwen3-30B-A3B Q4_K_M `tg64`          | 35.1   | 42.1   |
  | Qwen3-30B-A3B Q4_K_M `pp512`         | 103.9  | 253.7  |
  | Qwen3-30B-A3B Q4_K_M `pp1024`        | 102.6  | 223.3  |
  | gemma-4-26B-A4B Q4_K_M chat prefill  | 150    | 323    |
  | gemma-4-26B-A4B Q4_K_M chat decode   | 31.1   | 35.4   |
  | Qwen3.6-35B-A3B IQ3_S `tg64` (paged) | 6.5    | 6.7    |
  | Qwen3-0.6B Q4_K_M `tg64` (control)   | 126.5  | 127.0  |
  | Qwen3-0.6B Q4_K_M `pp512` (control)  | 4482.7 | 4475.5 |

  Token-identical at `--temp 0` with the tier on vs off, on both resident MoE
  architectures: Qwen3-30B-A3B over a 37-token generation, and gemma-4-26B-A4B
  (a different gating/activation shape) over 16.

  What this does NOT fix: the id tier removes the launch and serialization
  overhead but not the per-slot weight TRAFFIC. A routed expert's bank is still
  streamed once per `(row, slot)`, so at `pp512` a 128-expert layer re-reads
  each expert ~32×. The bucket-sorted batched expert GEMM that fixes that
  (Vulkan's `moe_scatter_reduce` arm) is a separate slice and is where the
  remaining prefill gap lives.

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

ROCm has RmsNorm→Linear + Linear→Add peepholes (Slice 32), the **F1b**
sibling-GEMV activation-quant memo, the **F1c** MoE folds (RmsNorm→MoeFfn,
MoeFfn→Add) and the **F1d** K-write fold (QkNormRope→WriteKv — all below), plus,
since **F1**, the capability-gated fusions its executor already implemented but
the "start with NOTHING fused" bring-up dial in `backend.rs`'s `capabilities()`
had never let the seam emit. Each was proved against the CPU reference first
(`crates/infr-rocm/tests/parity.rs`, the "F1 fusion gate" section) and then
measured on a 7900 XTX:

| capability      | state | evidence                                                                                                                                                                                                                                                                                                |
| --------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `combined_gu`   | ON    | one `[2*nff, ne]` FFN GEMV + `GatedActFused`. Qwen3-0.6B Q4_K_M tg128 126.3 → **137.3 t/s (+8.7%)**, pp512 +0.7%. Golden unmoved.                                                                                                                                                                       |
| `gated_rmsnorm` | ON    | fused per-head RMSNorm×SiLU-gate, **bit-identical** to the `QkNorm`→`GatedAct` pair (max_err 0.0). Qwen3.5-0.8B pp512 +1.3%, tg a wash.                                                                                                                                                                 |
| `argmax_rows`   | ON    | multi-row `Op::Argmax` is id-for-id with the CPU at a 151936 vocab, ties included. No perf change today — it gates only the MTP verify accept, and no ROCm `MtpHeadSession` exists yet.                                                                                                                 |
| `embed_gather`  | OFF   | works, but the native `embed_*` decode f16-rounds each element (`fin`) where the host embed path is exact f32 — ~2.5e-4 relative on the **Q6_K** `token_embd` a Q4*K_M GGUF ships, which moves the qwen3 golden. Also −1.2% tg. Needs an exact-f32 `deq*\*` sibling before it is even worth re-pricing. |

Kernel launches per decode token, before → after: Qwen3-0.6B **563 → 507**,
Qwen3.5-0.8B **561 → 495**, Qwen3-30B-A3B **1059 → 1059** (a pure-MoE arch has
no dense `ffn_gate`/`ffn_up` to concatenate — see §9's MoE items for its share).

### F1b — the sibling-GEMV activation-quant memo

**`Op::RmsNormAdd` was a false lead and is CLOSED.** F1 named it the largest
remaining dense-decode item; a per-kernel launch histogram says otherwise. The
Slice-32 peepholes already leave the dense qwen3 decode with **zero** standalone
`rmsnorm` and **zero** standalone `add` dispatches — the norm is folded into
each consuming GEMV's quant and the residual into the GEMV epilogue — so there
is no adjacent norm+residual pair left to fuse. The op itself is implemented
(`exec.rs`, `rmsnorm_add`) and already emitted wherever the seam emits it
(gemma4-E2B's per-layer projection tail, `seam/runner.rs`); nothing was gated
off. The "56 per token" was real, but it belonged to the item below.

The actual largest item was **`rmsnorm_quant_i8_32` at 113 of 507** — 4 per
layer where the arch has 2 norms. `q`/`k`/`v` are three separate GEMVs off ONE
input norm, and each re-ran the whole normalize+int8-quantize pass over the
identical row into its own scratch. So the executor now runs the pass ONCE and
rebinds the same `(codes, scales)` for the siblings (`QuantMemo` in `exec.rs`:
keyed on the source row's `TensorId` **and** device pointer, the norm weight,
`eps`, `m` and `in_f`; taken at the top of every `run_op` and republished only
by the int8 GEMV branch, so any op that could rewrite the row invalidates it by
construction, and never republished by a GEMV whose fused-residual epilogue
wrote the very row it quantized). Vulkan has had the same thing since its
`mmv_memo`.

It is **bit-identical**, not merely close: the GEMV binds the exact bytes the
elided pass would have written. `quant_memo_sibling_linears_are_bit_identical`
asserts `==` (not a tolerance) between three sibling projections issued in one
graph and the same three issued one graph at a time, for Q4_K / Q4_0 / Q8_0;
`quant_memo_is_dropped_when_the_gemv_rewrites_its_own_source` pins the
invalidation guard (removing it makes that case ~100 % wrong, not subtly wrong).
All nine `rocm_seam` goldens hold with qwen3 at `0xfd63781ea3bfa785`, and temp-0
output is token-identical before/after on a dense, an MoE and a DeltaNet model.

Kernel launches per decode token, F1 → F1b:

| model                       | total          | `rmsnorm_quant_i8_32` |
| --------------------------- | -------------- | --------------------- |
| Qwen3-0.6B Q4_K_M (dense)   | 507 → **451**  | 113 → 57              |
| Qwen3.5-0.8B Q4_K_M (Delta) | 495 → **447**  | 115 → 67              |
| Qwen3-30B-A3B Q4_K_M (MoE)  | 1059 → **963** | 145 → 49              |

Measured interleaved against an F1 binary, warmed, first burst discarded (the
box drifts over a long session — the pairs matter, the absolutes do not):

- Qwen3-0.6B `tg128`: 11/11 pairs positive. Final binary 121.2/121.0/120.3 →
  **125.3/124.9/125.3 (+3.6 %)**; the earlier stable batch 119.0/119.9/119.6/
  119.7 → 124.1/123.6/123.4/122.0 (+3.2 %); a first, drifting batch 116.3 →
  123.0.
- Qwen3-30B-A3B `tg64`: base 42.1/42.5/42.6 → **44.1/44.1/44.1 (+3.8 %)**.
- `pp512` a wash everywhere (dense 4458.8 → 4469.7; MoE 252.9 → 252.6) — prefill
  is weight-bandwidth-bound, and the pass it elides is per-row.
- Qwen3.5-0.8B `tg128`: **a wash.** A first session read −1.7 % across 6
  forward-and-reversed pairs; a longer 3-way interleave (F1 / F1b / an F1b built
  to keep the pool-draw sequence identical) put all three inside 18.0-18.4, i.e.
  inside this model's ±2 % run-to-run band. Its decode is ~55 ms/token dominated
  by `deltanet_decode`, so 48 fewer small launches is not resolvable there.

### Next on this axis

Dense decode is now 451/token: `linear_i8_q4k` 140 (5 real GEMVs/layer),
`rmsnorm_quant_i8_32` 57, `qk_norm_rope` 56, `quant_i8_32` 56, `write_kv` 56,
`linear_i8_q6k` 29, `attention` 28, `gated_act` 28, `argmax` 1.

- **`write_kv` (56, of which 28 are fusable)** — the largest remaining _fusable_
  item, and the `kv_write_peephole` §3 already names (`adapter.rs:876`). The
  shared pass plans it already (`infr_core::fusion::plan_kv_write`); ROCm's
  `decode_fusion` passes `kv_write: false`. The blocker is the kernel, not the
  plan: the peephole's contract is an **f16** K row written straight into an f16
  cache, and ROCm's `qk_norm_rope` writes a fresh **f32** packed buffer
  (`zero_dev`). Needs an f16-out variant taking the cache pointer + ring row. V
  has no rope to absorb its write, so the ceiling is 28. → **landed in F1d
  below**, at exactly that ceiling.
- **`quant_i8_32` (56) + `gated_act` (28) + `attention` (28)** — the remaining
  quant passes are _not_ sibling-redundant (o_proj's row comes from `attention`,
  down_proj's from `gated_act`); killing them means an int8-emitting epilogue on
  the producing kernel, i.e. new kernels rather than a peephole.

### F1c — the two MoE decode folds

F1b left Qwen3-30B-A3B at 963/token with **96 of those a standalone `rmsnorm`
×48 and `add` ×48** the dense path does not pay. Both are landed, as two new
patterns in the **shared** `infr_core::fusion` pass, each behind a
backend-supplied predicate so Vulkan and Metal are untouched until they opt in
(both pass `moe_ok: None` / `moe_add: None`; their suites are the gate — 216 +
27/27 unmoved).

- **`MoeFfn → Add` (`plan_moe_add`)** — the structural twin of `plan_linear_add`
  (same Internal-dst requirement, same immediately-following-op rule, same
  live-range bound); only the producer differs. ROCm folds the residual into
  `moe_accum_idm`, which now takes an optional `res` pointer and writes
  `dst[i] = res[i] + acc` instead of a standalone `add` over a zeroed scratch.
  **The R8 summation order is untouched**: `acc` is still the ascending-slot
  reduction onto a `0.0f` seed and the residual joins it exactly where the
  elided `add` joined it, so `res + acc` ≡ `add(res, 0.0f + acc)` bit-for-bit —
  `0.0f + acc == acc` for every value `acc` can take, because a sum seeded at
  `+0.0` never rounds to `-0.0`, the one case where adding zero is not the
  identity. Scoped to the R8 tier: the pre-R8 per-slot loop `atomicAdd`s into
  `dst` and is deterministic only because the host serializes the slots, so
  seeding it with the residual would re-associate the sum. A tier that declines
  (paged, `INFR_ROCM_NO_I8`, `moe_id_rows = 0`) **replays** the elided `Add`.
- **`RmsNorm → MoeFfn` (`RmsNormLinearCfg::moe_ok`)** — F1b's note that the norm
  feeds "the F16 router `Linear`" was one step off: the router GEMV lives INSIDE
  `Op::MoeFfn` (`linear_f16`, 48/token), so the norm's only graph consumer is
  the MoE op itself and there is no F16 `Linear` to disqualify. That makes the
  fold reachable without touching the router: `rmsnorm_quant_i8_32` gained an
  optional `xn` output, so ONE pass now produces both the experts' int8 codes
  and the normalized f32 row the router reads — byte for byte what the
  standalone `rmsnorm` wrote, from the same expression. The pair `rmsnorm` +
  `quant_i8_32` becomes one launch. Single-row only (the prefill arm chunks its
  rows, so one normalize pass cannot serve a router GEMV spanning all of them);
  a tier that cannot take it replays the elided `rmsnorm`. The `xn` write is its
  **own** loop rather than a predicated store inside the quantize loop — folded
  in, it cost the dense path (which passes `xn == 0`) ~1 % of decode, measured.

Both are asserted **bit-identical**, not to a tolerance:
`moe_residual_fold_is_bit_identical` (`==` vs the same graph with `fuse_add`
cleared, at `n_used` 1/2/4, residual scaled to the MoE output's own magnitude so
a re-association cannot hide in the rounding),
`moe_residual_fold_declined_tier_replays_the_add` (`moe_id_rows = 0`), and
`rmsnorm_moe_fold_is_bit_identical` (`==` vs `fuse_norm` cleared, with the
router weights scaled DOWN so the softmax stays soft — at the raw scale the
top-1 gate saturates to exactly 1.0 and the test would pass with an `xn` that is
merely close; mutation-checked both ways). Four `infr_core::fusion` unit tests
pin the opt-in, the expert predicate, the live-range bound and the single-row
gate. All nine `rocm_seam` goldens hold with qwen3 at `0xfd63781ea3bfa785`, and
temp-0 output is token-identical before/after on a dense (Qwen3-0.6B), an MoE
(Qwen3-30B-A3B) and a MoE+DeltaNet (Qwen3.6-35B-A3B) model.

Kernel launches per Qwen3-30B-A3B decode token, F1b → F1c (963 → **867**):

| kernel                        | F1b | F1c     |
| ----------------------------- | --- | ------- |
| `linear_i8_q4k`               | 168 | 168     |
| `quant_i8_32`                 | 144 | **96**  |
| `qk_norm_rope`                | 96  | 96      |
| `write_kv`                    | 96  | 96      |
| `rmsnorm_quant_i8_32`         | 49  | **97**  |
| `add`                         | 48  | **0**   |
| `rmsnorm`                     | 48  | **0**   |
| `attention`                   | 48  | 48      |
| `linear_f16` (the MoE router) | 48  | 48      |
| `moe_accum_idm`               | 48  | 48      |
| `moe_gate_up_act_i8_idm_q4k`  | 48  | 48      |
| `moe_topk`                    | 48  | 48      |
| `linear_i8_q6k`               | 25  | 25      |
| `moe_down_i8_idm_q4k/q6k`     | 48  | 48      |
| `argmax`                      | 1   | 1       |
| **total**                     | 963 | **867** |

Measured on a 7900 XTX, interleaved against a binary built from the parent
commit (`15642b7`), warmed, first burst discarded:

- Qwen3-30B-A3B `tg64 -r 3`: base 44.1/44.2/44.1/44.1/44.2 → **46.2/46.1/46.1/
  46.1/46.1 (+4.4 %)**. Back-to-back ×8 each way confirms it with no overlap:
  base 44.1-44.2, F1c 46.1-46.4. (At `-r 2` three of twelve F1c runs dipped to
  39-43 while base never did; at `-r 3`, and in the ×8 back-to-back batches, the
  dips vanish — a short-burst artefact of alternating two processes over a 17
  GiB weight set, not the fold.)
- Qwen3-30B-A3B `pp512`: **a wash** (252.3/252.5/252.3 → 252.7/252.5/252.4) —
  both folds are decode-only by construction.
- Qwen3-0.6B `tg128` (dense control, must be inert): **a wash**
  (124.0/126.7/124.5/125.2 → 125.1/124.7/124.4/125.0).
- Qwen3.6-35B-A3B `tg64`: a wash (7.1 → 7.1). That model is VRAM-bound at ~140
  ms/token; 48 launches are not resolvable there.

### MoE, still open after F1c

Qwen3-30B-A3B is 867/token. The largest remaining MoE-specific item is
**`quant_i8_32` ×96** — 2 per layer (the expert `h` quantize between gate/up and
down, and the o*proj input). Neither is sibling-redundant: `h` is produced by
`moe_gate_up_act_i8_idm*\*`and o_proj's row by`attention`, so killing them means
an int8-emitting epilogue on the producing kernel, not a peephole — the same
conclusion §3 reached for the dense `quant_i8_32`×56. After that,`write_kv`×96
/`qk_norm_rope`×96 are the shared dense item (28 fusable on the dense arch, 48
here, blocked on an f16-out`qk_norm_rope` — **taken by F1d below**), and
`linear_f16` ×48 (the MoE router) is a single small GEMV per layer with nothing
adjacent to absorb it.

### F1d — the K-write peephole

Both F1b and F1c named `write_kv` next, and the shared pass had planned it since
the unification (`infr_core::fusion::plan_kv_write`); ROCm just passed
`kv_write: false`. The blocker was the kernel: the peephole's contract is an f16
K row written straight into an f16 cache, and `qk_norm_rope` wrote a fresh f32
packed scratch. It now takes `(kv, kv_row, kv_stride)` and, when they are set,
casts each rotated element with `__float2half` into the cache row the elided
`write_kv` would have filled. **Byte-identical, not close**: the value
expression is untouched and an f32 register holds the same bits an f32
round-trip through DRAM would have, so the cast sees exactly the input it saw
before. `kv` is a uniform kernel argument, so the un-fused Q path pays one
scalar branch, not per-lane divergence — no measurable cost on the dense
control's `qk_norm_rope` ×56. The elided `WriteKv` also takes with it the K
scratch entirely: its pooled draw, its zeroing `hipMemsetAsync`, and the
round-trip.

`FusionCfg::kv_write` has no per-backend predicate hook (its own gate is fixed:
an Internal f16 rope `dst` feeding an immediately-following `WriteKv` into an
f16 cache), so ROCm applies its coverage as a **post-filter** — `kv_fuse_ok` in
`exec.rs` drops every planned entry this backend cannot reproduce exactly and
**un-skips** its `WriteKv` so the standalone kernel replays. Four gates, each a
way the fused kernel could differ from the pair it replaces:

1. **`Op::QkNormRope` only.** The shared pass also matches an f16-out `Op::Rope`
   (llama's K path), but ROCm's `rope` rotates an f32 buffer in place after a
   DtoD copy — no output pointer to redirect, let alone an f16 one. Those keep
   the split pair (Llama-3.2-1B is token-identical before/after at ~5 k
   context).
2. **The rope must tile the cache row exactly** —
   `row_stride == n_head * head_dim`, same `rows`. The elided kernel copied
   `rows × row_stride` packed elements; the fused grid is `rows × n_head` waves
   of `head_dim`, so any other stride would leave the row's tail unwritten or
   run into the NEXT position's slot.
3. **No ring wrap.** `kv_swa_ring: false` means the seam hands ROCm full-context
   caches and `write_kv` indexes `pos + row` with **no modulo at all**; the
   shared plan hands back `pos % cap_rows` regardless, because Vulkan's ring
   caches need it. The fused kernel writes the **plan's** row and the gate is
   what makes that row equal `pos` — so "the fused variant does exactly what the
   write path does today" is a checked property, not a comment, and the day ROCm
   wants ring semantics this gate is what forces `write_kv` to learn them first.
4. **Live range.** The plan carries no live-range bound for this pattern
   (Vulkan's record-once decode _requires_ its K write fused, so it cannot
   afford one), but redirecting the write leaves the rope's `dst` scratch never
   written — safe only if nothing reads it. Reuses the shared
   `dst_only_read_by_next`, now `pub`.

Six parity cases, all `==` or poison-exact, all at a **non-zero** write row with
every other cache row poisoned — a wrong row corrupts a _different_ position's
attention, which a short greedy check does not see:
`kv_write_fold_is_bit_identical_at_a_nonzero_row` (fold on vs off over the whole
cache, `rows` 1 and 4, `pos` 0/1/5/11/12, plus an assert that the un-fused
control actually filled its row, so the comparison cannot be vacuous),
`kv_write_fold_lands_on_the_row_the_cpu_writes` (the **absolute** row against
the CPU backend, so an offset wrong the same way on both sides of that A/B
cannot survive), and one case per gate:
`kv_write_fold_declines_a_row_the_write_path_would_not_wrap` (a cache declared 8
rows and allocated 16, written at `pos = 11`: row 11 must hold the data and row
3 must still be poison), `kv_write_fold_declines_a_row_stride_it_would_overrun`,
`kv_write_fold_declines_when_the_rope_dst_is_read_again`, and
`kv_write_fold_hatch_leaves_the_standalone_write`. Six mutations, all caught:
dropping `kv_row` from the write offset, dropping the per-head offset, and
dropping each of gates 2/3/4 or the un-skip.

`kernels.rocm.fuse_kv_write` is the hatch (no env key, following `module_cache`
— `--set kernels.rocm.fuse_kv_write=false`); it exists because a byte-identity
claim is only checkable against the un-fused control.

Kernel launches per decode token, F1c → F1d:

| model                       | total         | `write_kv` |
| --------------------------- | ------------- | ---------- |
| Qwen3-0.6B Q4_K_M (dense)   | 451 → **423** | 56 → 28    |
| Qwen3-30B-A3B Q4_K_M (MoE)  | 867 → **819** | 96 → 48    |
| Qwen3.5-0.8B Q4_K_M (Delta) | 447 → **441** | 12 → 6     |

Exactly the K half, per layer, on every arch; nothing else in either histogram
moved. Prefill fuses too (Qwen3-0.6B `pp64` graph 560 → 532) — the fold is
dtype-free and the fused grid is already one wave per (row, head).

Measured on a 7900 XTX, interleaved against a binary built from the parent
commit (`48d37e4`), warmed, first burst discarded, both binaries stripped of the
temporary launch-histogram instrumentation:

- Qwen3-0.6B `tg128 -r 2`: base 124.3/124.6/125.3/124.6 and 124.8/124.0/124.3 →
  **127.4/127.9/128.2/127.8 and 127.7/127.4/127.5 (+2.6 %)**. Two independent
  bursts, non-overlapping bands.
- Qwen3-0.6B `tg64 -d 2048 -r 2` (decode at depth — where a ring-row bug would
  show and a launch saving is diluted by the longer attention): base
  93.2/93.2/92.9/93.1 → **94.2/94.7/94.9/94.5 (+1.6 %)**.
- Qwen3-30B-A3B `tg64 -r 2`: base 46.6/46.7/46.8/46.6 → **47.4/47.5/47.4/47.4
  (+1.6 %)**.
- `pp512` a wash on both: dense 4443.7/4438.0/4452.8/4461.6 →
  4477.1/4435.7/4432.9/4469.9; MoE 252.8/252.7/252.7/252.5 →
  252.8/252.9/252.6/252.3. Prefill is weight-bandwidth-bound; 28 launches out of
  a 532-launch chunk graph are not resolvable.
- Qwen3.5-0.8B `tg128`: a wash (18.4/18.4/18.4 → 18.4/18.4/18.3) — 6 saved
  launches on a 55 ms `deltanet_decode` token.

`rocm_seam` 9/9 with qwen3 at `0xfd63781ea3bfa785` unmoved; `infr-rocm` 151;
temp-0 output token-identical before/after on Qwen3-0.6B, Qwen3-30B-A3B and
Qwen3.5-0.8B at a short prompt, and at ~2.8-5 k tokens of context on Qwen3-0.6B,
Qwen3-30B-A3B, gemma-3-1b (SWA) and Llama-3.2-1B (the `Op::Rope` decline path).
The shared pass's own gate is clean: `infr-vulkan` 216, `gpu_seam` 27/27 — the
only `infr_core::fusion` change is making `dst_only_read_by_next` `pub`, which
alters no plan.

### V is NOT this peephole's to take

The V half of `write_kv` (28 dense / 48 MoE) stays, and deliberately. V's write
has a producer — but not one this pattern can absorb:

- Its immediate producer is `Op::Linear` (the v projection), or `Op::Copy`
  (gemma4 full layers, V = the raw K projection), or an `Op::AddBias` /
  weightless `Op::QkNorm` writing `v` **in place**. None is a rope, and the
  in-place ones have no output pointer to redirect at all.
- Absorbing `Linear → WriteKv` is a **different** rewrite: a fifth shared
  pattern plus an f16-cache-store epilogue on every int8-decode GEMV entry point
  (12 named `linear_i8_*` kernels plus the macro family), mutually exclusive
  with the fused-residual epilogue those same kernels already carry, and it must
  decline the prefill WMMA/rocBLAS arm which does not go through them.

That is a slice, not a peephole, and this one does not speculate about its
result. What it does establish is the price of a launch on this path: 28 dense
launches (plus their scratch draw + memset + round-trip) were worth +2.6 % of
`tg128`, so the V half is worth roughly the same order — enough to justify
_measuring_ the GEMV-epilogue work, not enough to justify guessing at it.

### Next on this axis after F1d

Dense decode is 423/token: `linear_i8_q4k` 140 (5 real GEMVs/layer),
`rmsnorm_quant_i8_32` 57, `qk_norm_rope` 56, `quant_i8_32` 56, `linear_i8_q6k`
29, `attention` 28, `gated_act` 28, `write_kv` 28, `argmax` 1. MoE is 819/token
with `quant_i8_32` 96 and `write_kv` 48.

The next-largest item that is not the GEMV itself is **`quant_i8_32` — 56 dense
/ 96 MoE, two per layer** — and the conclusion is unchanged from F1c: neither is
sibling-redundant (o_proj's row comes from `attention`, down_proj's from
`gated_act`, the MoE `h` from `moe_gate_up_act_i8_idm_*`), so killing them needs
an int8-emitting epilogue on the **producing** kernel, not a peephole — the same
new-kernel work the V write above needs, on the same set of kernels. After that
the remaining `write_kv` ×28/48 (V) is the next fusable count.

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

- [x] **Quant coverage** — ✅ native decode + int8 + WMMA-prefill for all 24
      formats (**24/24 after R7**), ✅ MoE experts for every format a GGUF packs
      expert banks with, and ✅ the id-indexed multi-slot MoE expert GEMV
      (**R8**, total over `moe_native_fmt`). Paged MoE deliberately keeps the
      per-expert loop for its copy/compute overlap — see R8
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
