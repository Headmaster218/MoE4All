# cpu-perf.md — CPU backend performance roadmap

Findings and a prioritized worklist for the `infr-cpu` reference backend,
aggregated from the CPU perf review. Ordered **low → high implementation
difficulty** so we land the cheap, high-certainty wins first.

## Results snapshot

Landed so far (bit-identical unless noted; precision-flip slices
coherence-checked token-identical to the independent Vulkan int8 path before
acceptance):

| Slice                              | Model / quant     | Decode       | Prefill (pp512)   |
| ---------------------------------- | ----------------- | ------------ | ----------------- |
| conv1d parallel (`ac9c228`)        | Qwen3.5-9B Q4_K_M | —            | flat (GEMM-bound) |
| mmap madvise (`5ed932a`)           | Qwen3.5-9B Q4_K_M | neutral      | neutral           |
| DeltaNet head-parallel (`9595bf3`) | Qwen3.5-9B Q4_K_M | flat         | 67→110 t/s (+63%) |
| native int8 **Q4_0** (`6e7decd`)   | Qwen3-0.6B Q4_0   | +142% (2.4×) | +239% (3.4×)      |
| native int8 **IQ4_XS** (`304dd42`) | Qwen3-0.6B IQ4_XS | +156% (2.6×) | +320% (4.2×)      |
| native int8 **Q2_K** (`1e90613`)   | Qwen3-0.6B Q2_K   | +29%         | +71%              |
| native int8 **Q3_K** (`d559984`)   | Qwen3-0.6B Q3_K_M | +185% (2.9×) | +199% (3.0×)      |
| native int8 **Q4_1** (`f4738a1`)   | Qwen3-0.6B Q4_1   | +138% (2.4×) | +243% (3.4×)      |
| native int8 **IQ4_NL** (`0ef8366`) | Qwen3-0.6B IQ4_NL | +47%         | +253% (3.5×)      |

### Full CPU↔Vulkan native-quant parity (unit-test-verified — no models)

CPU now has a native int8 kernel for **every format Vulkan has a native mmq
kernel for**. The six below close the last of the gap; none has a small
supported-arch GGUF for an end-to-end coherence check (`IQ2_S`/`IQ3_S` are
big-model-only; `Q5_1`/`Q2_0` legacy; `MXFP4`=gpt-oss, whose architecture
infr-llama doesn't implement; `NVFP4` bleeding-edge). Gate for these =
**bit-identical SIMD↔ scalar + tolerance-parity to the exact `dequant_block`
(tight 1e-3 vs the quantized-activation dot)** — strong on the math, no
coherence generation.

| Format    | commit    | notes                             |
| --------- | --------- | --------------------------------- |
| `Q5_1`    | `e7465ed` | affine, Q5_0 5-bit + Q4_1 min     |
| `Q2_0`    | `3f7c79e` | Bonsai ternary, 64-block, 2×Q8x32 |
| `MXFP4`   | `29ee2e5` | IQ4_NL + E8M0 scale (gpt-oss)     |
| `NVFP4`   | `06cc0ef` | MXFP4 codebook + per-16 UE4M3     |
| `IQ2_S`   | `8e616c3` | grid-codebook, expand-row-once    |
| `IQ3_S`   | `34fd4f2` | grid-codebook, per-32 scale       |
| `IQ2_XXS` | `aa07a2f` | grid, KSIGNS lookup, coherence ✓  |
| `IQ3_XXS` | `e97a913` | grid, token-identical to GPU ✓    |
| `IQ2_XS`  | `7e0fb1d` | grid, 9-bit index (unit-test)     |
| `IQ1_S`   | `93e3225` | 1-bit, `dl·(iprod + delta·asum)`  |
| `IQ1_M`   | `c70111f` | 1-bit, d-in-scales, per-8 delta   |

The grid/codebook set (`IQ2_XXS`/`IQ2_XS`/`IQ2_S`, `IQ3_XXS`/`IQ3_S`,
`IQ1_S`/`IQ1_M`) uses the expand-row-once → signed-i8 → per-group int dot
pattern; each mirrors `dequant_codebook` and is gated on a **tight 1e-3**
tolerance vs the exact decode. Coherence: `IQ3_XXS` is token-identical to the
GPU native path; `IQ2_XXS` matches leading tokens; the 1-bit `IQ1_S`/`IQ1_M` are
chaotic at the noise floor (int8/f32/CPU/GPU all diverge — the 1e-3 math check
is the correctness gate).

**CPU now natively handles EVERY weight quant format** — including ternary
`TQ1_0` (`b7a4201`) / `TQ2_0` (`6336df3`), fold `(digit−1)` into signed i8 +
single-scale int dot. Full parity with Vulkan's native set.

Deferred: #3 (DeltaNet clones — measured ~0.1%, negligible). Remaining: #8
(f16/bf16, low priority), #9 (blocked on `perf`), #10 (fusion), VNNI batch for
IQ4_XS.

## Cross-backend fast-kernel coverage

CPU, Vulkan, and Metal now each have a native fast kernel for **every** weight
quant format infr supports (24/24) — no format falls back to dequant→float on
any backend. The full matrix and per-backend decode strategy moved to its own
doc: **`docs/kernels.md`**. (CPU closed its last gap with ternary `TQ1_0`
`6336df3`/`b7a4201`; TriLM-3.9B TQ2_0 prefill 16.5→131.9 t/s ≈ 8×.)

## Context: two regimes, two different bottlenecks

CPU inference splits hard by batch size, and the cache/bandwidth story is
different in each:

- **Decode (`m == 1`) is DRAM-bandwidth-bound.** A real model's weights are GBs
  (`Q4_K` 9B ≈ 5 GB; even a `Q2_K` 0.6B ≈ 180 MB) — all **≫ L3**. Every weight
  is read once per token, streamed contiguously from RAM. On a sequential stream
  the hardware prefetcher already saturates the memory controllers, so the only
  lever that scales decode is **fewer bytes streamed** (native quantization)
  plus **TLB** relief (hugepages). Software prefetch of weights is a wash here —
  the HW prefetcher already predicts the stream.
- **Prefill (`m > 1`) is compute + cache-reuse-bound.** Weights still stream,
  but each weight row is reused across `m` activation columns, so keeping the
  activation tile resident in L1/L2 is the lever. This is where blocking / tile
  sizing / fusion pay off.

### Reference hardware (dev box)

AMD Ryzen 9 9950X3D (Zen 5, 3D V-Cache): **128 MiB L3**, 16 MiB L2 (1 MB/core),
768 KiB L1d (48 KB/core), 16 cores / 32 threads, **1 NUMA node**. ISA: AVX-512
F/BW/VL/DQ/CD, **AVX-512-VNNI**, AVX-512-BF16, AVX-VNNI, F16C, 3DNow-prefetch.
The big X3D L3 helps prefill (a whole layer's activations + KV stay hot); it
does **not** rescue decode (weights still ≫ 128 MB).

## What is already done (do not redo)

- **Weights mmap'd native.** `Op::Linear` streams the row-major GGUF weight one
  row at a time straight from the mmap — no f32 materialization in RAM.
- **Int8-quantized-activation VNNI dots** for the common k-quants — **Q4_K,
  Q5_K, Q6_K, Q8_0, Q5_0** — with scalar→AVX2→AVX-512BW→VNNI kernels and up to
  **8-row cache-blocking tiles** (activation loaded once, reused across 8 weight
  rows). This is already the "native format, lossy-but-fast, cache- friendly"
  strategy; it is why a Q4_K_M model is already tight.
- **Prefill conv1d parallelized** over the virtual `[state‖x]` sequence
  (`ac9c228`). Bit-identical; isolated kernel ~7.3× but end-to-end flat (conv1d
  is <1% of GEMM-bound prefill).

## The bottleneck ranking (why the list is ordered as it is)

1. **Fewer bytes (native quant coverage)** — dominant decode lever.
2. **Hugepages / madvise on the weight mmap** — real TLB win on the GB stream.
3. **Op fusion** — cuts intermediate DRAM round-trips.
4. **Prefill tile tuning** to the X3D topology — real but measure-first.
5. **Software prefetch** — micro-opt, usually a wash. Not a strategy.

---

## Worklist (low → high difficulty)

Each slice: TDD, bit-identity where the math is unchanged (parallelization,
fusion of exact ops), tolerance-parity + a sanctioned golden re-bless where the
math changes (int8 activation quant is lossy). One slice at a time; validate
correctness before benching.

**Re-bless discipline:** a golden only gets re-blessed after the new output is
_verified correct/coherent_ — never blind-accept a diff. For a precision flip
that means confirming the model still generates sane, coherent text (compare a
short generation against the f32/GPU path) AND that the CPU result matches the
GPU int8 result within tolerance. A golden diff that changes _which_ tokens are
produced in a way that looks like garbage is a bug, not a precision flip.

### 1. Weight mmap `madvise` + THP hint — _easy_

- **What:** on the weight mmap, advise the kernel: `MADV_HUGEPAGE` (2 MB pages
  cut dTLB page-walks on the multi-GB sequential read), `MADV_SEQUENTIAL` /
  `MADV_WILLNEED` (bias readahead the way we actually consume).
- **Why:** a >L3 sequential mmap read at 4 KB pages hammers the dTLB; the TLB is
  the one "prediction" structure with headroom in the decode stream. Hugepages
  is the closest thing to "help the CPU preload the next region" that survives
  the bandwidth reality.
- **Impact:** small–moderate decode + weight-load win; low risk.
- **Precision:** none (pure memory hint). Bit-identical.
- **Status:** DONE (`5ed932a`). `WillNeed` + Linux `HugePage`, best-effort, not
  `Sequential`. Measured **neutral** on the dev box (warm page cache; THP
  commonly a no-op on file-backed maps): 9B Q4_K_M tg64 10.2→10.1, pp512
  67.4→67.1 (noise). Kept for cold-load / THP-enabled fs / over-RAM cases at
  zero risk; a rigorous A/B needs `perf` counters or cold cache (see
  Measurement).

### 2. DeltaNet head-parallelism — _easy–medium_

- **What:** `Op::DeltaNet` runs a serial single-thread scan. The outer `for t`
  is inherently sequential (state carries across tokens), but the inner
  `for h in 0..n_vhead` loop over value heads is **fully independent** — each
  head owns a disjoint `state[h*kd*vd..]` slice, its own out slice, and reads
  only shared inputs. Parallelize over heads: each head task runs its whole
  `t`-scan on its own state copy (`pool.collect`), then write state + out back.
- **Why:** DeltaNet is the linear-attention path for **~75% of Qwen3.5 layers**
  (full attention only every 4th) — a major CPU cost, unlike conv1d. 16 heads
  (9B) → up to 16-way parallelism on the dominant attention op.
- **Impact:** real prefill **and** decode win expected.
- **Precision:** bit-identical (same per-head float order; state rebuild is a
  copy).
- **Status:** DONE (`9595bf3`). `deltanet_scan` helper, one pool task per head.
  Bit-identical (parity test, exact f32). **Qwen3.5-9B Q4_K_M prefill pp512
  67.3→109.8 t/s (+63%)**; decode flat (10.3→10.4, DRAM-bound at rows=1). The
  big prefill win of the campaign so far.

### 3. DeltaNet input-clone elimination — _medium_

- **What:** the DeltaNet arm `.clone()`s the whole `q/k/v` buffers
  (`[rows, heads·dim]`) every op purely to dodge the borrow checker (state needs
  `&mut vals` while q/k/v need `&vals`). Introduce a disjoint-`vals` accessor
  (split one `&mut` index out, borrow the rest `&`) to drop the clones. The same
  pattern recurs in other ops (conv1d clones too), so the accessor is reusable.
- **Why:** at prefill those clones are ~1M floats × 3 per DeltaNet layer of pure
  allocation + copy traffic.
- **Impact:** moderate prefill win; removes allocator pressure.
- **Precision:** bit-identical.
- **Status:** DEFERRED — measured negligible. 9B has 18 DeltaNet layers × ~12 MB
  (q/k/v) clones = ~216 MB memcpy per 512-tok prefill ≈ **~0.1%** of a ~4.7 s
  prefill; transient allocs are freed immediately (no RSS concern) and decode is
  rr=1 (nothing to clone). Not worth the disjoint-`vals`-accessor complexity /
  unsafe borrow-splitting. Revisit only if a profile flags DeltaNet allocation,
  or fold into a broader `vals` accessor refactor if one is done for another
  reason.

### 4. Native int8 dot: **Q4_0** — _medium_

- **What:** `Q4_0` currently falls to `bytes_to_f32` dequant + f32 dot (the slow
  catch-all fallback). Add native int8-activation kernels
  (scalar/AVX2/AVX-512BW/VNNI + batch/batch8) and wire into both the `m==1` and
  `m>1` dispatch, mirroring `Q8_0` and the GPU's native Q4_0 kernel.
- **Why:** Q4_0 is ubiquitous; the GPU already has a native kernel. First and
  simplest of the uncovered formats.
- **Impact:** large on Q4_0 models (decode + prefill); kills the f32 fan-out.
- **Precision:** int8 activation quant is lossy → this **changes the CPU
  reference output for Q4_0**. Tolerance-parity test vs the f32 reference to
  bound error; the Q4_0 gpu_seam golden is a sanctioned **precision-flip
  re-bless** (`--include-ignored`), and the new CPU path should match the GPU
  int8 result, not the old f32.
- **Status:** DONE (`6e7decd`). `vec_dot_q4_0_32_batch` (scalar + AVX2 + VNNI),
  cloned from Q5_0 (18-byte block, offset 8, no 5th bit); reuses `Q8x32`. Wired
  into decode + prefill (decode had no Q4_0 kernel before). **No golden
  changed** and no re-bless needed — CPU greedy output is coherent and
  **token-identical to the independent Vulkan int8 path** ("…is **Paris**.").
  SIMD bit-identical to the scalar oracle; tolerance-parity vs full-precision
  dequant. **Qwen3-0.6B Q4_0 CPU: decode 28.7→69.6 t/s (+142%), prefill
  128.7→435.9 t/s (+239%).**

### 5. Native int8 dot: **IQ4_XS** — _medium_

- Same treatment as Q4_0, for the common small-model format IQ4_XS (local
  Qwen3-0.6B has one). GPU reference exists (quant-cliff-warp).
- **Precision:** precision-flip re-bless as in #4.
- **Status:** DONE (`304dd42`). `vec_dot_iq4xs` / `_batch` (scalar + AVX2 +
  AVX-512BW single-token), modeled on Q6_K but with a `KVALUES_IQ4NL` codebook
  `pshufb` lookup and Q8_0's abs/sign signed-dot trick. Coherent +
  token-identical to Vulkan int8 ("…is **Paris**"); no golden changed; SIMD
  bit-identical to scalar. **Qwen3-0.6B IQ4_XS CPU: decode 37.8→96.6 t/s
  (+156%), prefill 129.7→544.8 t/s (+320%).** Follow-up: no AVX-512-VNNI
  **batch** variant yet (batch runs AVX2) — a `dpbusd` batch path can lift
  prefill further on VNNI hosts (this box has `avx512_vnni`).

### 6. Native int8 dot: **Q2_K, Q3_K** — _medium–high_

- K-quant super-block formats with packed scales; more decode work than Q4_0 but
  same int8-activation regime. One slice each.
- **Precision:** precision-flip re-bless per dtype.
- **Status:** Q2_K **DONE** (`1e90613`). `vec_dot_q2k` / `_batch` (scalar + AVX2
  - AVX-512BW + VNNI), modeled on affine Q4_K; 2-bit codes, per-16 sub-blocks,
    min-correction via the existing `q8.bsums16`. Coherent + token-identical to
    Vulkan int8 ("…is Paris."); no golden changed; SIMD bit-identical to scalar.
    **Qwen3-0.6B Q2_K CPU: decode 25.2→32.5 t/s (+29%), prefill 127.9→218.5 t/s
    (+71%).** Q3_K **DONE** (`d559984`): `vec_dot_q3k` / `_batch` (scalar + AVX2
    - AVX-512BW + VNNI), modeled on Q6_K's signed path with offset 32→4; 6-bit
      scales via the aux bit-shuffle, 3-bit codes from qs + hmask bit-planes,
      `−4·bsums16` correction. Coherent + token-identical to Vulkan int8
      ("…**Paris**"); SIMD bit-identical to scalar. **Qwen3-0.6B Q3_K_M CPU:
      decode 35.4→100.9 t/s (+185%), prefill 198.2→592.8 t/s (+199%).** With
      this the whole **K-quant family is native** (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K).

### 7. Native int8 dot: legacy affine + IQ codebook + grid family — _medium → high_

- The remaining uncovered formats, in ascending difficulty.
- **Precision:** precision-flip per dtype, coherence-verified vs the Vulkan int8
  path before acceptance.
- **Status:**
  - `Q4_1` **DONE** (`f4738a1`): affine clone of Q4_0 (`y = d·q4 + m`,
    `as·(d·iprod + m·bsum)`). Coherent/token-identical to Vulkan. **Qwen3-0.6B
    Q4_1: decode 28.7→68.2 t/s (+138%), prefill 129.2→442.6 t/s (+243%).**
  - `IQ4_NL` **DONE** (`0ef8366`): flat 32-block codebook cousin of IQ4_XS
    (`KVALUES_IQ4NL` pshufb + abs/sign dot, `Q8x32` activation). Coherent/
    token-identical. **Qwen3-0.6B IQ4_NL: decode 34.3→50.5 t/s (+47%), prefill
    124.5→439.6 t/s (+253%).**
  - `Q5_1` **DONE** (`e7465ed`), `Q2_0` **DONE** (`3f7c79e`), `MXFP4` **DONE**
    (`29ee2e5`), `NVFP4` **DONE** (`06cc0ef`), `IQ2_S` **DONE** (`8e616c3`),
    `IQ3_S` **DONE** (`34fd4f2`), and the full grid/codebook set `IQ2_XXS`
    (`aa07a2f`), `IQ2_XS` (`7e0fb1d`), `IQ3_XXS` (`e97a913`), `IQ1_S`
    (`93e3225`), `IQ1_M` (`c70111f`) — all **DONE** (Vulkan already had native
    kernels for these; this closed the CPU side). See the "Full parity" section
    and the coverage audit at the top. The grid kernels expand the grid row to
    signed i8 once (scalar gather + `apply_signs`) then reuse the IQ4_XS
    per-sub-block scale × int-dot, amortized across the batch.

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

## Measurement (prerequisite for #9, useful throughout)

"Should help in theory" gets verified with counters, not intuition. `perf` is
**not installed** on the dev box; installing it (or an equivalent that reads
`LLC-load-misses`, `dTLB-load-misses`, `stalled-cycles-backend`) lets us
classify each stall as DRAM-bound (→ only fewer bytes helps), TLB-bound (→ #1),
or cache-miss slack (→ #9). For hotspot attribution use `samply` (never ad-hoc
timers); for A/B throughput use `infr bench --dev cpu` / `infr compare`.

## Software prefetch — explicitly deprioritized

Explicit `_mm_prefetch` of weights is a micro-opt, not a strategy: the HW
prefetcher already predicts the sequential weight stream, and a mistuned
prefetch distance evicts useful lines. Only revisit if `perf` shows
latency-bound (not bandwidth-bound) stalls on an _irregular_ access pattern.
