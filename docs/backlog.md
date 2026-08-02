# Backlog

Known work that is deliberately not done, with enough context to pick it up
cold.

Everything here has been triaged: it is either blocked on something, scoped out
of the slice that surfaced it, or waiting on hardware. Items that were merely
_unfinished_ do not belong here — they get done. An item leaves this file when
it lands or when it is withdrawn (with the reason recorded, so it is not
rediscovered).

Provenance tags point at the finding that opened the item:

- `CR-*` — [code-review.md](code-review.md), the 2026-08-01 whole-tree review.

---

## Open

### B1 — `INFR_DN_CHUNK_SCAN` has inverted polarity

**Tag:** CR-N10 · **Blocked on:** a breaking-change sweep

`crates/infr-core/src/config/env.rs`:

```rust
v.dn_chunk_scan = presence_inv(get, "INFR_DN_CHUNK_SCAN");
```

The key is spelled positively but its presence _disables_ the chunked scan — the
only key in that file whose name means the opposite of what it does. It is
R1-frozen (the config campaign pinned existing spellings), so renaming it is a
breaking change for anyone who has it set.

**When the next breaking sweep happens:** rename to `INFR_NO_DN_CHUNK_SCAN`,
which makes it match the `presence_inv` grammar every other `INFR_NO_*` key
uses. No alias — the project's env policy is to drop old spellings cleanly
rather than carry them.

### B2 — 16 unconverted 128-bit SIMD loads

**Tag:** CR-U3 residual · **Blocked on:** nothing; scoped out of its slice

`crates/infr-cpu/src/kernels.rs` still has 16 `_mm_loadu_si128` sites taking the
raw intrinsic. The 171 256/512-bit sites were converted to the bounds-checking
`load256` / `load512` helpers; the 128-bit ones were explicitly out of that
slice's scope.

Same failure mode: a `RangeFrom` slice index bounds only the first byte while
the intrinsic reads sixteen, so the loads are correct by block geometry alone
and a layout change breaks them silently.

**To do:** add a `load128` alongside the existing two (same shape — generic over
the element type, offset in elements, width checked in bytes, `#[inline]` +
`#[target_feature(enable = "sse2")]`) and convert all 16. Note several read a
fixed-size const table (`KVALUES_IQ4NL.as_ptr()`) where the bound is trivially
exact; those are still worth converting so the invariant is uniform.

### B3 — the VNNI kernel family's bounds assertions are unexercised

**Tag:** CR-U3 coverage gap · **Blocked on:** CI hardware

63 of the 171 converted SIMD load sites are in the `*_vnni` kernels. They
dispatch behind `is_x86_feature_detected!("avx512vnni")`, and no development or
CI machine currently has it, so their `debug_assert!`s are compiled but never
executed. The tests _call_ those kernels; the runtime gate skips them.

The other tiers are covered: avx512bw runs natively, and the avx2 tier was
verified by temporarily stubbing the 37 `avx512bw` gates to `false` and
re-running the suite.

**To do:** either add a VNNI-capable CI runner, or install an emulator (Intel
SDE) and add a job that runs `cargo test -p infr-cpu` under it. Until then,
treat the VNNI kernels' bounds as argued-but-untested.

### B4 — tensor-parallel has no automated coverage

**Tag:** CR-C10 · **Blocked on:** CI hardware

`TensorParallelBackend` needs ≥2 physical GPUs, so nothing in CI exercises it.
The C10 byte-accounting bug (KV checkpoints allocated at `full/W²` while the
copy moved `full/W`) went unnoticed because of this, and was only caught when an
unrelated bounds guard turned it into an error.

The fix landed with unit tests on the pure parts — `TpBuffer::len_bytes()` per
kind, `shard_bytes`' divisibility guard, and the
`shard_bytes(len_bytes()) == per-rank` round trip — but the real path is still
unrun.

**To do:** a two-GPU smoke run of `--tensor-parallel` with MTP enabled (which is
what exercises the checkpoint path) would close it. Worth doing once by hand
even without CI.

### B5 — `infr serve` has no rate limiting

**Tag:** CR-S7 (partial) · **Decision:** out of scope for this binary

The per-request wall-clock deadline landed (`serve.request_timeout_secs`), which
bounds how long one request can hold a `--parallel` slot. Rate limiting —
bounding how many requests one client may make — deliberately did not.

**Reasoning:** every operator exposing this to a network already has a reverse
proxy, and that is where connection limits, per-IP quotas and burst control
belong. Reimplementing them in a single-binary inference server means owning a
worse version of solved infrastructure.

**Revisit if:** someone wants to expose `infr serve` directly to untrusted
traffic with no proxy in front. That is a different product decision, not a
missing feature.

### B6 — prefill throughput is not reproducible run-to-run

**Tag:** measured 2026-08-02 · **Blocked on:** nothing; needs a decision on
where to pin the tier

Running the full 35-row sweep twice against the SAME infr binary (`691c0dc`),
with llama.cpp absolutes moving under 1%, the two prefill columns disagree with
themselves:

| column       | mean abs Δ between runs | worst row | rows moving >5% |
| ------------ | ----------------------- | --------- | --------------- |
| `tg128`      | 0.8%                    | 3.0%      | 0 / 35          |
| `tg64@d4096` | 0.7%                    | 2.2%      | 0 / 35          |
| `pp512`      | 6.8%                    | **34.5%** | 10 / 35         |
| `pp4@d4096`  | 7.7%                    | **31.7%** | **19 / 35**     |

Decode is solid; prefill is not. The cause is the documented tier/chunk
nondeterminism — a short prefill can land on a different kernel tier between
runs — and it bites hardest where the prefill is shortest (the sub-2B models)
and on the IQ3_S MoE (+34.5%).

This is not cosmetic. It moved `pp512` from 26/35 wins to 34/35 between two runs
of one binary, and it means a prefill A/B under ~10% cannot be trusted without
repeats. The previous snapshot's "small-model `pp512` cluster" finding turned
out to be this variance, not a real deficit.

**To do:** make the tier/chunk choice deterministic for a given shape, or pin it
when `-r > 1` so a benchmark at least measures one tier consistently. Until then
`infr bench -u/--ubatch <N>` is the workaround, and prefill numbers should be
reported as a median of several runs rather than a single value.

### B7 — decode attention at depth: three designs measured; a mid-depth win found

**Tag:** measured 2026-08-02 · **Blocked on:** nothing; slice 3a (the
specialization) has LANDED — what remains open is width by workgroup count

Note the `tg128` numbers in the table below are the PRE-slice-3a baselines
(`INFR_NO_ATTN_DECODE=1` reproduces them exactly); see "What slice 3a bought" at
the end of this entry for the shipped figures.

The largest remaining gap to llama.cpp is decode at depth, not prefill.
Qwen3-30B-A3B Q4_K_M on a 7900 XTX against `llama-bench c629da5`:

| depth | `pp512` infr/llama  | `tg128` infr/llama |
| ----- | ------------------- | ------------------ |
| 8192  | 1771.7 / 1692.9 t/s | 138.7 / 165.1 t/s  |
| 16384 | 1145.4 / 1174.8     | 106.5 / 140.3      |
| 32768 | 670.3 / 738.3       | **66.9 / 112.1**   |

Prefill holds near parity; decode falls to **0.60×** and is what drags the
`pg8192,512` turn to 0.73× @32k. Per-token cost over 8k→32k (a 2.25 GiB KV
delta) rises **+7.74 ms** for infr against **+2.86 ms** for llama.cpp — an
effective KV rate of ~291 GB/s vs ~787 GB/s.

`attn_partial_bda` is **59% of decode GPU time** at d32768 (177 µs per
layer-token, 3072 dispatches) and `attn_combine` another 8%. It scales exactly
linearly — 44.2 µs @ d8192 → 177 µs @ d32768, 4.0× for 4× the KV, no fixed
overhead.

**The bottleneck is the per-key subgroup reduction, not bandwidth.** The kernel
gives each 32-lane wave one key and reduces its 128-dim dot with a
`subgroupAdd`, so the cross-lane reduction ALU scales with keys×heads. Three
measurements pin it:

- **GQA head-grouping LOSES.** One workgroup per (KV-head, chunk) covering all
  `g = nh/nkv` query heads cuts K/V traffic 8× (537 MB → 67 MB per layer-token,
  the re-read the per-query-head grid pays). Measured **329 µs vs 177 µs**, i.e.
  1.87× SLOWER while moving an eighth of the bytes. Grouping serializes 8
  cross-lane reductions into one wave that previously ran on 8 CUs in parallel;
  the re-read was nearly free, served out of Infinity Cache.
- **It is not workgroup starvation either.** Re-running the grouped kernel at
  matched parallelism (chunk 64 → 2048 workgroups, same as the per-head grid)
  measured **359 µs** — worse still, and `attn_combine` went 24.5 → 146 µs on
  the 8× larger `pacc`.
- **Fewer keys per workgroup also loses.** Halving `ATTN_SPLIT.max_chunk` to 256
  on the UNGROUPED kernel (4096 workgroups) measured **314 µs** vs 177 µs. The
  shipped 512/2048 point is a real optimum between per-workgroup fixed cost and
  latency hiding.

So neither traffic nor occupancy is the lever, and the whole GQA-grouping
approach is **declined** on this kernel structure — do not re-try it as written.
(The experiment was reverted; the tree is unchanged. It did reach bit-identical
parity with `attn_partial`, so the approach was correct, just slower.)

- **The LDS-staged K-tile kernel LOSES too, badly.** This is the design
  `recorder.rs`'s `attention_kv_split_impl` names in its rows-batched comment
  ("per-thread full dots, no cross-lane reductions, which is how llama.cpp wins
  that cell"). Built and measured as an unwired probe
  (`tests/attn_ktile_probe.rs`, `shaders/attn_ktile.comp`), per-dispatch µs at
  `nh=32 nkv=4 hd=128 chunk=512`, reference measured in the same harness:

  | leg                             | d8192 | d32768    |
  | ------------------------------- | ----- | --------- |
  | shipped `attention_kv_split_at` | 69.8  | **183.9** |
  | k-tile, 64-key tile, 17 KB LDS  | 127.7 | 493.9     |
  | k-tile, same but unpadded rows  | 147.3 | 535.4     |
  | k-tile, 128-key tile, 34 KB LDS | 190.1 | 687.2     |
  | k-tile, 64-key half-depth, 9 KB | 127.9 | **381.9** |

  Best config is **2.7× slower** at d32768. It is not an implementation miss:
  the ISA confirms the design did what it set out to do — `attn_partial_bda` has
  54 cross-lane ops, all four k-tile builds have **0**, with ACO fusing the
  f16→f32 conversion into 140 `v_fma_mix_f32` per shader. The reduction is gone
  and it is still 2.7× down, because **the LDS transpose that buys its removal
  costs more than the reduction saved**. The K tile has ZERO data reuse (each
  staged key row is read by exactly one thread), so LDS is pure coalescing
  overhead — every K byte written once and read once — plus an occupancy loss.
  Time is monotone in LDS budget: 34 KB → 687 µs, 17 KB → 494, 9 KB → 382.

So the reduction ALU is real but is not the lever either, and **both the
GQA-grouping and the LDS-K-tile approaches are declined** — do not re-try either
as written. (GQA grouping was reverted; the k-tile survives as an unwired probe
because it is the measurement rig for the next attempt. Both reached agreement
with the reference — GQA bit-identically, k-tile to 9.6e-7 relative.)

**What the oracle actually does.** Rather than guess a third time, read
llama.cpp's own decode path (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
`get_fa_tuning_params_scalar`). For our shape on RDNA3 — hsk=hsv=128, n_rows=1,
f16 KV — it resolves to:

- `path = FA_SCALAR`. Coopmat is **deliberately avoided at decode**: "scalar is
  faster than coopmat when N==1" forces `FA_COOPMAT1/2` → `FA_SCALAR` at
  `n_rows == 1`. This kills the matrix-core / `gqa_ratio` idea previously
  proposed here — the oracle does not use matrix cores for decode.
- `shmem_staging = 0` on AMD (it is set only for NVIDIA). The oracle does
  **not** stage K/V in shared memory on this hardware, independently
  corroborating the k-tile negative above.
- `block_rows = 1`, `block_cols = 64`, `workgroup_size = 128` (4 subgroups of
  32).
- **`d_split = min(min(subgroup_size, 8), D_lsb/4) = 8`.**

`d_split` is the parameter that matters and it is the one thing none of the
three experiments varied. It is the width of the group that cooperates on ONE
key's dot:

| design        | d_split | reduction steps | keys in flight per wave | needs LDS?    |
| ------------- | ------- | --------------- | ----------------------- | ------------- |
| shipped       | 32      | 5               | 1                       | no            |
| k-tile probe  | 1       | 0               | 32                      | yes (sank it) |
| **llama.cpp** | **8**   | **3**           | **4**                   | **no**        |

**The `d_split` sweep was run** (`shaders/attn_partial_dsplit.comp`,
`tests/attn_dsplit_probe.rs` — a parameterized decode-only copy; the grid, chunk
policy, `pm`/`pl`/`pacc` layout and `attn_combine` are all unchanged, so it A/Bs
directly). Width × workgroup size, per-dispatch µs, reference measured in the
same harness. Read the rows at PRODUCTION's chunk: `adaptive_chunk` picks 256 at
d8192 and 512 at d32768, so ch=512-at-d8192 is a configuration production never
dispatches.

| leg                | d8192 ch=256 (1024 wg) | d32768 ch=512 (2048 wg) |
| ------------------ | ---------------------- | ----------------------- |
| shipped reference  | 53.3                   | **179.9**               |
| w=1 wg=64          | 47.3 (1.13×)           | 306.9 (0.59×)           |
| w=2 wg=64          | 41.9 (1.27×)           | 273.4 (0.66×)           |
| **w=4 wg=64**      | **36.6 (1.46×)**       | 212.6 (0.85×)           |
| w=8 wg=64 (llama)  | 37.2 (1.43×)           | 209.8 (0.86×)           |
| w=16 wg=64         | 39.5 (1.35×)           | 201.6 (0.89×)           |
| w=32 wg=64 (ctrl)  | 49.0 (1.09×)           | **170.3 (1.06×)**       |
| w=8 wg=128 (llama) | 50.2 (1.06×)           | 243.1 (0.74×)           |

Two SEPARABLE findings, and the `w=32` control is what separates them:

1. **Specialization is free ~6–9% at every depth, with no algorithmic change.**
   `w=32` reproduces the shipped mapping exactly yet beats it (1.09× / 1.06×),
   because the probe is a decode-only copy — no window/canvas/ring/Q8/hd-256/512
   arms — and allocates **96 VGPRs against `attn_partial_bda`'s 120**
   (`RADV_DEBUG=shaderstats`), so more waves fit per SIMD. Zero spills in all 12
   builds, so this is occupancy, not spill relief.
2. **Narrow width wins only where workgroup parallelism is short**, and the
   effect is monotone in workgroup count: 512 wg → best 2.01×, 1024 wg → 1.46×,
   2048 wg → every width loses. Width substitutes keys-in-flight-per-wave for
   workgroups. At depth the kernel is already at ~3.0 TB/s (537 MB / 180 µs =
   Infinity-Cache rate), and splitting a wave's contiguous 512-byte K read into
   `32/w` separate segments costs more than the shallower reduction saves.

So llama.cpp's `d_split = 8` is right for ITS configuration, not universally —
and **the original B7 target (d32768) remains a negative**: nothing beats the
shipped kernel there except the free specialization.

**What slice 3a bought (LANDED).** `shaders/attn_decode.comp` — the hd=128 f16
BDA decode arm of `attn_partial`, copied statement for statement, with the
window / canvas / SWA-ring / Q8 / mainline-inline / hd-256 / hd-512 / small-m
arms deleted and `sc[1024]` cut to `sc[512]`. Two builds (static and
`-DUSE_PARAMS -DSELF_CHUNK` replay), selected in
`Recorder::attention_kv_split_impl` / `attention_kv_split_dynac_impl`;
`INFR_NO_ATTN_DECODE=1` (`kernels.vulkan.attn_decode`) forces `attn_partial`
back. `RADV_DEBUG=shaderstats`: **96 VGPRs / 3072 B LDS** against
`attn_partial_bda`'s **120 / 5120**, zero spills either way.

Qwen3-30B-A3B Q4_K_M, `infr bench -p 0 -n 128 -r 3`, legs alternated twice:

| depth | ON            | OFF (= the table above) | gain      |
| ----- | ------------- | ----------------------- | --------- |
| 4096  | 169.9 / 169.5 | 163.2 / 162.8           | **1.04×** |
| 8192  | 147.4 / 147.2 | 138.4 / 138.3           | **1.06×** |
| 32768 | 71.9 / 71.7   | 66.5 / 66.4             | **1.08×** |

Independently re-measured on the same box (separate alternated run): 170.4 /
163.1 = 1.045×, 148.1 / 138.5 = 1.069×, 72.3 / 67.1 = 1.078×. The OFF legs land
on the pre-slice baselines recorded at the top of this entry (138.9 and 66.9),
which is what makes the A/B trustworthy — the knob really is reverting to the
old kernel and not just perturbing it.

The output is **BIT-identical** —
`crates/infr-vulkan/tests/attn_decode_parity.rs` asserts raw `f32` bits over
every shape × both call paths, and no `gpu_seam_matches_cpu*` golden moved. B7's
earlier claim that the drift came from `ClusteredReduce` is WRONG, or at least
incomplete: making all five of `attn_decode`'s reductions
`subgroupClusteredAdd(., 32u)` still produced bit-identical output on RADV/RDNA3
(a full-width cluster lowers to the same tree), and so did swapping the runtime
`sqrt(float(pc.hd))` for a constant-folded `sqrt(128.0)`. **The
`attn_partial_dsplit` probe's 9.6e-7 at w=32 is therefore unexplained** — worth
ten minutes before slice (b) leans on that probe's numbers again.

Also: `dsplit_bench`'s "SHIPPED reference" leg now measures `attn_decode`, not
`attn_partial_bda`, because it calls `attention_kv_split_at`. Re-run it under
`INFR_NO_ATTN_DECODE=1` to compare against the old baseline.

**Still open — (b) width by workgroup count.** Choose `w` from `nh * n_chunks`
against the device CU count rather than from depth directly (the monotone
relationship is in workgroups, not kv_len). Needs validating on shapes the probe
never covered — it only tested `nh=32 nkv=4 hd=128`. Note the width sweep's
per-width numbers were measured against the OLD reference, so its ratios now
overstate the remaining headroom by the 4–8% slice 3a already took.

This does NOT close the headline deep-context gap: at d32768 the model is still
~0.64× llama.cpp's tg128. Treat (b) as a mid-depth lever, not a fix for B7's
opening table.

**What slice 3b bought (LANDED) — coverage, not speed.** The two exclusions that
cost real models throughput are gone: **sliding-window** (`-DSWA -DRING`) and
**hd 256 / 512** (`-DDHD4=64/128`). Widened as a FAMILY of build-time
specializations, never one runtime-branching kernel — `attn_decode.comp` now
compiles into 12 builds (3 head dims × causal/SWA × static/replay). `-DSWA`
always ships with `-DRING`: an SWA layer's cache is a ring of
`round64(window + ubatch)` rows, so the build carries `j % rcap`, which is the
identity on a full-context cache and lets the host gate skip any cap reasoning.
The static gate keeps its `pos < cap/(nkv*hd)` row bound for the CAUSAL builds
only (they still assume the identity); the replay gate cannot check it at all
(kv_len is not known at record time), which is exactly why covering the ring was
a precondition for covering SWA there.

`RADV_DEBUG=shaderstats` (fresh `XDG_CACHE_HOME`,
`MESA_SHADER_CACHE_DISABLE=1`), all 12 builds: **96 VGPRs / 3072 B LDS, zero
spills** — identical to slice 3a's and against `attn_partial_bda`'s /
`attn_partial_dynac_bda`'s **120 / 5120** and `attn_partial_nohd_bda`'s **120 /
5120**. So every variant is meaningfully leaner than the arm it replaces; none
was withheld. (Dropping `vsh[32]` for hd 256/512 does not show up — LDS
granularity rounds 2304 B back to 3072.)

`infr bench -p 0 -n 128 -r 3`, legs alternated twice, f16 KV (the bench default
at these depths; note `infr run` at gemma's full 131072 ctx auto-quants KV to
q8_0, and a q8 cache does not take this path at all):

| model                 | depth | ON            | OFF           | gain       |
| --------------------- | ----- | ------------- | ------------- | ---------- |
| gemma-3-12b (hd256)   | 4096  | 86.3 / 86.1   | 85.9 / 85.7   | 1.005×     |
| gemma-3-12b           | 8192  | 83.8 / 83.9   | 83.4 / 83.5   | 1.005×     |
| gemma-4-12b (256/512) | 4096  | 84.9 / 84.8   | 84.2 / 84.3   | 1.008×     |
| gemma-4-12b           | 8192  | 82.3 / 82.3   | 81.4 / 81.5   | **1.010×** |
| Qwen3-30B-A3B (ctrl)  | 4096  | 170.0 / 170.1 | 163.6 / 163.0 | 1.041×     |
| Qwen3-30B-A3B (ctrl)  | 8192  | 147.5 / 147.6 | 138.3 / 138.3 | 1.067×     |

The Qwen control reproduces slice 3a's 1.04× / 1.06× to within noise, so the
shipped hd=128 causal gate was not perturbed.

Independently re-measured on the same box: gemma-3-12b 86.3 / 85.9 @d4096 and
83.8 / 83.5 @d8192; Qwen control 148.4 / 139.0 @d8192 (1.068×). The gemma
profile above was re-run too and confirms both halves of the explanation —
attention pass 1 is 8.8% of decode GPU time, and `attn_decode_hd256_swa` is flat
at 18.1 → 17.8 µs from d4096 to d8192 while `attn_decode_hd256` grows 40.1 →
77.9 µs.

**Judgement: this slice is coverage insurance, not throughput.** Roughly 0.5% on
gemma is barely above measurement noise, and it cost 12 shader builds and ~240
lines of `#ifdef`. It ships because it is bit-identical, uniformly leaner, and
removes a "some layers fast, some not" inconsistency that would confuse the next
profile — not because it moved a number worth quoting.

**The gemma gain is ~0.5–1%, an order of magnitude below Qwen's, and the reason
is share, not kernel quality.** `INFR_PROF_OPS=1 INFR_SEAM_NO_REPLAY=1` on
gemma-3-12b @d4096: `native_mmv_mrow_q4k_m4` is 55% of decode GPU time and the
whole attention pass 1 is 8.7% (`attn_decode_hd256_swa` 5.9% over 40 layers,
`attn_decode_hd256` 2.6–5.7% over 8). Gemma's SWA layers are **window-capped** —
17.8 µs/layer at d4096 AND at d8192 — so 40 of 48 layers do not grow with depth
and the lever cannot grow with it either. Do not expect this to improve at
d32768; the global layers are the only part that scales. gemma-4-12b @d4096 is
the same picture with `attn_decode_hd512` (47.5 µs × 8 global layers) in place
of hd256.

**One measured trap, worth remembering.** The hd 256/512 QK tail loop must keep
`attn_partial`'s redundant `if (r < hd4)` guard AND read `hd4` from the push
constant at runtime. With `hd4` folded to the build-time literal the guard
vanishes, ACO fuses `part + dot(..)` across the terms into one FMA chain, and
the chunk score moves 1 ULP — which `attn_combine`'s `exp(m_c - M)` weight then
turns into 136/2048 non-identical outputs on
`hd256 swa513 kv1500 ring1024 single-key chunk`. The comment in the shader says
so; do not "clean it up".

**Still falling back after 3b** — each would need its own family member:
planar-Q8 and mainline-inline quant KV (this is what gemma runs by DEFAULT at
full context, so widening it is the biggest remaining coverage hole), the
DiffusionGemma canvas mask, `rows > 1` (small-m spec-verify and prefill), the
bound-SSBO (non-BDA) dispatch, `chunk > 512` (only reachable above ~524k keys
under `INFR_KV_OVERFLOW`), head dims other than 128/256/512, and a RING cache on
a `window == 0` layer (unreachable today — only SWA layers are allocated as
rings — and the static gate's row bound rejects it rather than assuming).

### B8 — the activation/scratch reserve is not path-aware

**Tag:** measured 2026-08-02 · **Blocked on:** a design choice between the two
options below

`seam::dense_act_reserve` estimates how much VRAM to hold back for activations.
It is the term that made `kv_fit_ctx_fmt` claim gemma-3-12b's trained window
would not fit at f16 (~11 GiB reserved at the default 1024-row chunk), which
triggered a KV auto-quant that measurement showed was unnecessary. The immediate
inconsistency — reserving for a 1024-row chunk that placement then abandons for
512 — is being fixed separately. **This entry is about the reserve's ACCURACY,
which is a different problem and is not fixed by that.**

Measured ground truth, gemma-3-12b Q4_K_M f16 KV @ctx 131072 with a real 120k
prefill (peak sampled from `/sys/class/drm/card1/device/mem_info_vram_used`):

| term        | MiB        | source                                              |
| ----------- | ---------- | --------------------------------------------------- |
| peak VRAM   | 17 506     | measured                                            |
| weights     | 6 962      | GGUF size on disk                                   |
| KV cache    | ~8 672     | 8 global × 8192 B/tok × 131072 + 40 SWA × 1536 rows |
| **scratch** | **~1 872** | residual                                            |

(The 40/8 SWA/global split is confirmed from a decode profile: 1280
`attn_decode_hd256_swa` dispatches over 32 tokens = 40 layers, 256 = 8.)

The reserve has exactly ONE path branch today:

```rust
let attn_s = if cfg.swa_window == 0 && cfg.max_head_dim() == 128 { 0 } else { /* full want_ctx */ };
```

Two things are wrong with it. It is far too coarse — real dispatch picks between
flash / non-flash-coopmat / `nc_fa` / split-K **per layer**, on hd, mask, row
count, kv length, KV dtype and coopmat capability — and the comment above it
asserts "gemma3-12b: full layers are Causal+hd128 = flash", which a profile
disproves: gemma-3-12b is hd **256** on every layer and takes the pessimistic
branch. Terms a correct accounting has to price, none expressible as a constant:

- **non-flash score tile** — `rows × n_head × kv × 4`; 3.9 GiB at 512×16×120064,
  and **zero** on the flash path.
- **split-K partials** `pm`/`pl`/`pacc` — `[rows, nh, n_chunks, hd]`; the
  adapter's own comment notes ~1 GB at 1024 rows × 32 chunks × hd 256.
- **KV dequant f16 scratch** — only for the prepass formats (q4_0/q4_1/q5_0/
  q5_1/iq4_nl); zero for f16 and for native planar-Q8.
- flash `po`/`pm`/`pl` pools, MoE expert scratch / pager arenas, GEMM output row
  padding, rmsnorm and quantize temporaries.

**Do NOT replace it with a flat pad.** A constant was proposed (KV + 128 MiB)
and withdrawn on the numbers above: 128 MiB under-reserves the measured 1 872
MiB by ~15×, at a 512-row chunk, and scratch scales with both chunk height and
path.

**Two designs, pick one:**

1. **Shared tier predicate + drift test.** Extract the attention-tier selection
   the adapter already performs into one predicate that BOTH the dispatch gate
   and the estimator consume, then price each tier's buffers. Precedent in this
   repo: `infr_core::tensor::MOE_MMQ_DTYPES` as the single source both the
   graph-build and adapter gates derive from, guarded by `moe_mmq_drift_test`.
   Cheaper, but it is still two consumers of one fact and only the drift test
   stops them separating.
2. **Derive from the plan.** The adapter already allocates every scratch buffer
   through a named pool while recording the graph, so a dry-run graph build for
   the intended shape yields the exact total with no duplicated logic and
   nothing to drift. Exact by construction rather than by maintenance. More
   plumbing, and potentially circular — the plan depends on the ctx being sized
   — so it likely needs a fixed-point or a two-pass build.

**Second measurement, and the reason this is now urgent.** With the chunk-ladder
fix landed, `kv_fit_ctx_for` returns the EXACT boundary of
`weights + KV + dense_act_reserve_at <= alloc_room`, so any error in the reserve
is no longer absorbed by slack — it becomes a mid-prefill allocation failure.
gemma-4-31B UD-Q5_K_XL (weights 21 871 930 488 B, 24 GiB XTX) now resolves its
default context to 19 968 tokens at a 128-row chunk, and:

| prefill depth | result                                          | peak MiB |
| ------------- | ----------------------------------------------- | -------- |
| 8 000         | ok, 31.4 t/s                                    | 23 807   |
| 12 000        | ok, 30.8 t/s                                    | 24 038   |
| 16 000        | ok, 30.4 t/s                                    | 24 221   |
| 19 000        | **`VRAM budget exceeded`** on a 4 MiB act alloc | 24 298   |

So the top ~15-20% of the window it hands out could not actually be filled. The
term that runs away is the non-flash score tile, which the estimator prices as
`4 * n_head * ctx_pad` per row ("2 live pools at the final ctx") while the real
pools are keyed by byte size and ACCUMULATE as `kv_len` grows across the ~150+
chunks of a deep prefill. Note the failure tracks the prefill DEPTH reached, not
the context allocated: at margin 1.25 the model advertised a SMALLER context
(17 848) and still died, at d17840, because the depth tested rose with it.
Fixing that term is the specific thing that would make an advertised window
honest.

**An interim margin is in the tree, and it is yours to DELETE.**
`seam::ACT_RESERVE_MARGIN = (3, 2)` multiplies `dense_act_reserve_at`'s result.
It is applied to the ESTIMATED term only — KV bytes stay exact, computed from
geometry and dtype through the runner's own `kv_fmt_bytes` — and it was sized by
running each candidate and prefilling at ~100% of the context that candidate
advertises:

| margin | advertised ctx | prefill at ~100% of it          | peak MiB |
| ------ | -------------- | ------------------------------- | -------- |
| 1.00   | 19 973         | `VRAM budget exceeded` @ d19000 | 24 298   |
| 1.25   | 17 848         | `VRAM budget exceeded` @ d17840 | 24 299   |
| 1.50   | 15 872         | **ok**, 30.2 t/s @ d15864       | 24 142   |

1.50 is the first rung that fills its own window; also verified at d15000 /
d12000 / d8000 (30.3 / 30.6 / 31.2 t/s), and the d15864 peak reproduces
byte-for-byte. Remaining spare at the advertised context is **157 MiB of the 24
299 MiB guard budget (~0.6%)** — thin enough that this entry is the fix, not a
bigger constant.

A path-aware reserve must REPLACE the margin, not stack on it: delete the
constant, its multiply in `dense_act_reserve_at`, and the
`act_reserve_carries_the_interim_margin` test, then re-run the table above and
show the true reserve fills the window without it.

### B9 — audit bare `println!`/`eprintln!` onto `tracing`

**Tag:** raised 2026-08-02 · **Blocked on:** nothing; needs the policy below
agreed before a mechanical sweep

Diagnostics are emitted with bare `eprintln!` across the tree, so they carry no
level, no structured fields, and no filtering — `infr serve` in particular is
near-silent while running. A `tracing` subscriber already exists and covers
every subcommand (`infr-cli`'s `main()`: `tracing_subscriber::fmt()` with
`EnvFilter::try_from_default_env()` defaulting to `info`), so converting is safe
— messages will not vanish. `infr-llama` and `infr-server` already declare
`tracing`; `infr-llama` now uses it for the two KV auto-quant warnings
(`seam::model::clamp_default_ctx` and `vulkan_moe_binder`'s dense rung) and
nothing else, so its remaining 33 production sites are the sweep.

Scope, counted by splitting each `src/` file at its first `#[cfg(test)]`:

| crate        | production | in-test (leave alone) |
| ------------ | ---------- | --------------------- |
| infr-cli     | 64         | 0                     |
| infr-llama   | 33         | 10                    |
| infr-metal   | 13         | 1                     |
| infr-vulkan  | 11         | 55                    |
| infr-prof-rt | 11         | 0                     |
| infr-core    | 8          | 3                     |
| infr-cpu     | 3          | 6                     |
| infr-chat    | 2          | 0                     |
| infr-testkit | 1          | 0                     |
| **total**    | **146**    | **75**                |

Note infr-vulkan looks like the worst offender on a naive grep (66) and is
actually 11 — 55 of them are in-crate test output.

**Policy — this is not a blanket conversion. Four categories:**

- **Diagnostics → `tracing`.** Everything in the library crates (infr-core,
  infr-vulkan, infr-llama, infr-cpu, infr-metal, infr-gguf, infr-hub): a library
  should never write to the process's streams directly. Levels: `warn!` for
  degradations the user should know about (auto-quant, clamps, fallbacks),
  `info!` for lifecycle, `debug!`/`trace!` for the rest.
- **Program OUTPUT stays `println!`.** Generated tokens, `infr bench` result
  tables, `--json` output, `infr devices` listings. This is the CLI's contract
  and is piped by users; routing it through a filterable logger would break it.
- **In-crate test modules stay as they are** (75 sites) — that is test output.
- **`build.rs` stays** (17 sites) — `cargo:rerun-if-changed` etc. are a build
  protocol, not logging.

**Two hard carve-outs, both load-bearing:**

- **The SIGINT/SIGTERM handler must keep raw `write(2)`**
  (`infr-cli/src/main.rs`'s `on_signal`). It is async-signal-safe by
  construction; `tracing` allocates and takes locks, so converting it
  reintroduces exactly the deadlock-against-an-interrupted-`print!` that the
  comment there warns about.
- **The token-streaming path** deliberately avoids taking the stdout lock for
  the same reason — check before touching anything near it.

**To do:** agree the policy, then sweep crate by crate (library crates first,
they are the clear-cut cases), and add a lint or a test that fails on a new bare
`println!`/`eprintln!` outside the sanctioned categories — otherwise this decays
straight back.

### B10a — the serve arrival line reports prompt CHARS, not prompt tokens

**Tag:** raised 2026-08-02 · **Blocked on:** a `ChatGenerator` trait change

B10's request/throughput logging landed, with one part of its spec unmet: the
`request start` line carries `prompt_chars`, not prompt tokens. The tokenizer
lives behind `ChatGenerator`, so at arrival the server genuinely cannot know the
token count — it only learns it from `ChatOutcome` when the generation ends,
which is where the real `prompt_tokens` is logged (`request done`).

Fixing it means a new `ChatGenerator` method
(`count_prompt_tokens(&[ChatMessage])` or similar) implemented by `infr-cli`'s
`SeamGenerator` / `ParallelGenerator`, which then have to render the chat
template a second time — the template render, not just the tokenization, is what
determines the count. Out of scope for the logging slice: it is a trait change
plus duplicated render work on every request, for a number the completion line
already reports accurately a moment later.

### B11 — dense placement still budgets against raw free VRAM, not the guard's ceiling

**Tag:** measured 2026-08-02 · **Blocked on:** nothing; deliberately scoped out
of the KV-fit slice to keep the residency decision from moving

`VulkanBackend::check_vram_budget` enforces
`used + want <= total - GUARD_HEADROOM` (256 MiB), so the largest allocation
that can ever succeed is `vram().available - GUARD_HEADROOM`.
`VulkanBackend::alloc_room()` now returns exactly that, and
`SeamModel::kv_fit_ctx_fmt` budgets against it.

`vulkan_moe_binder`'s dense placement sweeps do NOT — their `fits` closure still
compares `fp.total() + kv_total_at(..) + dense_act_reserve_at(..)` against
`vram.available`, i.e. it may declare a model resident while planning 256 MiB
into memory the allocator will refuse. Same for the streaming `budget_at` and
the MoE expert-placement budget.

Not changed in the KV-fit slice on purpose: tightening the binder's budget can
flip a borderline model from resident to streamed, which is a ~10x decode
regression (gemma-4-31B: 33 t/s resident vs ~3 t/s streamed), so it needs its
own before/after measurement on the tight models rather than riding along.

Live consequence worth knowing before touching it: the two budgets now disagree
by more than the 256 MiB, because `ACT_RESERVE_MARGIN` (B8) widens the reserve
both of them consume while only the fit math also subtracts `GUARD_HEADROOM`. On
gemma-3-12b @131072 that is visible — the fit math validates the context at a
256-row chunk while the binder still goes resident at 512. Safe today (the
binder is the looser of the two, and the run peaks 6.5 GiB under budget), but it
means the two are no longer picking the same rung, which is the property the
shared `ubatch_candidates` ladder exists to give. Fix them together.

### B12 — an explicit `--ctx N` never reaches the refuse rung

**Tag:** raised 2026-08-02 · **Blocked on:** a product decision

`SeamModel::clamp_default_ctx` gained a refuse rung: when neither f16 nor q8_0
can serve even `MIN_SESSION_CTX` tokens it returns an `Err` naming the requested
context, both fits, the KV bytes needed and the free bytes after weights,
instead of handing back an unusable window.

It only ever sees the MODEL-DEFAULT context. A user-supplied `--ctx N` /
`INFR_CTX=N` is taken verbatim — `vulkan_session_on` and `vulkan_slot_ctx`'s
`SizeSpec::Bytes` arm both return it without consulting the fit math — and fails
later at allocation time with the generic VRAM-guard message. That is the case
where "refuse rather than silently degrade" has the most force, and it is the
one not covered.

The narrowness of the default-path rung is also deliberate and should not be
widened by accident: refusing whenever the TRAINED window does not fit would
break every ordinary long-context model on a 24 GiB card (Qwen3-30B-A3B clamps
262144 → ~50k and runs at 148 t/s; gemma-4-31B clamps 262144 → 19 968). Only "no
usable context at all" may refuse.

**To do:** decide whether an explicit oversized `--ctx` should fail early with
the detailed message (a behaviour change on a path documented as "never clamped
— the user asked") or keep failing at the alloc guard. If early: the check must
be provable, i.e. exact KV bytes alone + weights > `alloc_room()`, with no
activation reserve in it, so an over-estimating reserve can never refuse a run
that would have worked.

### B13 — the `+64` rows in every KV footprint estimate is slop, not padding

**Tag:** verified 2026-08-02 · **Blocked on:** nothing; left alone deliberately

`seam::kv_bytes_estimate_fmt` adds `KV_SLOP_ROWS = 64` rows per layer before
sizing each side's buffer, and the comment it inherited described this as
mirroring a pad `SeamKv` allegedly applies. It does not: both allocation sites
(`generate_dense_backend`'s KV loop and `SeamKv::fork`) allocate exactly
`kv_rows(..) * n_kv * head_dim` elements. The 64 rows are a deliberate
conservative margin and nothing more — the doc now says so.

Left in because every placement estimate shares this helper and removing it
would loosen all of them at once, on a model where (see B8) the remaining margin
is already ~1%. Worth revisiting only together with a path-aware reserve.

### B14 — verification gaps from the 2026-08-02 decode-attention and KV-fit slices

**Tag:** raised 2026-08-02 · **Blocked on:** nothing; each is a measurement
someone has to run

Recorded as gaps rather than left implicit. Everything below shipped without the
check named, and in each case the check is cheap — the reason it is missing is
time or hardware, not difficulty.

- **qwen35 (hd 256/512) was never benched against `attn_decode`.** Slice 3b's
  hd-256/512 family is proven CORRECT on it (the bitwise parity suite plus
  `unified_qwen35_gpu_seam_matches_cpu` / `gpu_seam_matches_cpu_qwen35moe`), but
  only gemma-3-12b and gemma-4-12b got a throughput A/B. qwen35 could be faster,
  slower or a wash and nobody has looked.
- **`INFR_NO_ATTN_HD=1` is wired and compiles but has never been executed.** It
  is the A/B escape back to `attn_partial_*_nohd_bda` for hd 256/512. An unrun
  escape hatch is not an escape hatch.
- **`docs/perf/results.md` is stale.** Its decode-at-depth cells are 4–8% better
  than published after slices 3a/3b, and the whole 35-row table predates both.
  It needs a re-sweep — and per B6 the prefill columns need medians of several
  runs, not single values, or the regenerated table will just re-import that
  variance.
- **Metal and CPU are unexercised for the KV-fit change.** Only the Vulkan path
  was measured. The Apple `#[cfg]`-gated code is not even compiled locally, so
  only CI can judge it.
- **`infr serve --parallel N` for N > 1 was not exercised.** `vulkan_slot_ctx`'s
  divide-by-N branch is unchanged by the slice but was only ever run at N=1.
- **The refuse rung's `Err` has never been printed by a real run.** No model on
  this box drives `max(fit_f16, fit_q8)` under `MIN_SESSION_CTX`, so the message
  text — the thing a stuck user actually reads — is untested against a human.
- **The iGPU chunk-ladder filtering is reasoned, not measured.** Filtering
  `ubatch_candidates` to heights below the current one also stops a placement
  sweep raising an integrated GPU's chunk above its watchdog-safe default. That
  argument was never run on an iGPU, and the watchdog is exactly the thing that
  punishes being wrong (see `docs/igpu.md`).
- **gemma at d32768 with `attn_decode` was not measured.** The window-capped
  profile (40 of 48 layers flat at ~17.8 µs from d4096 to d8192) predicts no
  further gain there. That is a prediction, not a measurement.

---

## Withdrawn

Recorded so they are not rediscovered and re-investigated.

### W1 — VRAM guard check-then-act race (CR-N7)

**Claim:** `check_vram_budget` / `vram_budget_fits` read the budget then
allocate with no lock, so two threads under `serve --parallel` could both pass a
budget only one fits in.

**Why it is wrong:** all large allocations happen at startup. Weights load
serially and `ParallelSeam::init_slots` is a sequential loop. The only `alloc`
reachable during serving is the MTP h-tap `Staging` buffer at `n_embd * 4` (~16
KB), which is below the guard's 1 MiB `CHECK_MIN` and skipped by design. There
is no concurrent large allocation to race.

### W2 — GGUF tensor lookup is O(n) per call (CR-N2, perf half)

**Claim:** `Gguf::resolve`'s linear scan costs "a few million string compares"
at load.

**Why it is wrong:** there are 8 `tensor_bytes*` call sites, 5 of them per-layer
— a few hundred thousand short comparisons once, i.e. microseconds. A `HashMap`
index would be complexity for no measurable gain. The _other_ half of N2
(duplicate tensor names silently accepted, first-wins) was real and is fixed.
