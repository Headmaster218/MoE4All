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

### B7 — decode attention at depth is ALU-bound, and the fix is a different kernel

**Tag:** measured 2026-08-02 · **Blocked on:** nothing; campaign-sized, needs a
decision to start

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

**What would actually work** is the design `recorder.rs`'s
`attention_kv_split_impl` already names in its rows-batched comment — "the
LDS-staged K-TILE kernel (per-thread full dots, no cross-lane reductions), which
is how llama.cpp wins that cell". Stage a K tile in LDS with coalesced global
reads, then let each lane compute a whole 128-dim dot from LDS, so the
`subgroupAdd` per key disappears entirely. GQA grouping is then worth revisiting
_on top of that_, because with `g` query heads as the M dimension a decode step
becomes an `8×128 @ 128×chunk` GEMM with enough M to use the matrix cores —
which is what llama.cpp's `gqa_ratio` flash-decode path does.

That is a new kernel plus its own parity suite across the `attn_partial` variant
matrix (static / replay / SWA-ring / Q8 / mainline-inline), so it is a campaign,
not a slice.

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
