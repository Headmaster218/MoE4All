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

**Tag:** measured 2026-08-02 · **Blocked on:** nothing; next step is two
contained slices (ship the specialization; then width by workgroup count)

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

**Next slice.** Two independent pieces, in this order:

- **(a) Ship the specialization.** A decode-only `attn_partial` variant at w=32
  is a ~6–9% win at every depth with the summation order unchanged. Lowest risk
  in the campaign.
- **(b) Width by workgroup count.** Choose `w` from `nh * n_chunks` against the
  device CU count rather than from depth directly (the monotone relationship is
  in workgroups, not kv_len). Needs validating on shapes the probe never covered
  — it only tested `nh=32 nkv=4 hd=128`. Also unmeasured: d2048/d4096, which is
  where the published `tg64@d4096` column sits and where a win would show up in
  the table.

Expected end-to-end, this model: d8192 tg128 +12% (0.84× → ~0.95× vs llama.cpp),
d32768 only +3% (0.60× → ~0.62×). Worth having, but it does NOT close the
headline deep-context gap — treat (a)+(b) as a mid-depth win, not a fix for B7's
opening table.

Note the output is not bit-identical (worst 1.18e-6 relative, 9.6e-7 at w=32 —
glslang emits `ClusteredReduce` where the shipped kernel gets `Reduce`, so ACO
builds a different tree). Landing either piece needs the `gpu_seam_matches_cpu*`
goldens run, and possibly re-blessed.

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
