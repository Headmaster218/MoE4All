# Backlog

Known work that is deliberately not done, with enough context to pick it up
cold.

Everything here has been triaged: it is either blocked on something, scoped out
of the slice that surfaced it, or waiting on hardware. Items that were merely
_unfinished_ do not belong here — they get done. An item leaves this file when
it lands or when it is withdrawn (with the reason recorded, so it is not
rediscovered).

Provenance tags point at the finding that opened the item:

- `CR-*` — the whole-tree correctness reviews. Their report lived at
  `docs/code-review.md` and was **deleted on 2026-08-03, folded into this
  file**: the eight findings of the 2026-08-03 pass were re-verified against the
  code and are B19–B26, and that pass's cleared / hardening / coverage lists are
  B27–B29. The tags on B1–B5 come from the earlier 2026-08-01 pass, whose text
  the file had already stopped carrying (`6ab8b1c` overwrote it with the later
  review). A `CR-*` tag is therefore a historical marker for where an item came
  from, not a link to anything.

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

### B3 — the VNNI kernel family's bounds assertions are unexercised

**Tag:** CR-U3 coverage gap · **Blocked on:** CI hardware

69 of the 187 converted SIMD load sites are in the `*_vnni` kernels. They
dispatch behind `is_x86_feature_detected!("avx512vnni")`, and no development or
CI machine currently has it, so their `debug_assert!`s are compiled but never
executed. The tests _call_ those kernels; the runtime gate skips them.

The other tiers are covered: avx512bw runs natively, and the avx2 tier was
verified by temporarily stubbing the 37 `avx512bw` gates to `false` and
re-running the suite.

The count grew from 63/171 when the 16 remaining `_mm_loadu_si128` sites were
routed through a new `load128` helper: 6 of those 16 landed in this untested set
— two in `vec_dot_q6k_batch_vnni` (the `scales_arr` per-block scale load), two
in `vec_dot_q32_batch8_ilv_vnni` (the Q8_0 bias path's two halves of `blk`), and
two in `vec_dot_nvfp4_batch_vnni` (`w_flat` and `q8.qs`). The other 10 run on
this hardware through the avx2 and avx512bw tiers and are covered by the
per-format parity tests; `load128`'s assertion was shown to fire by widening the
offset at `iq4nl_expand_codes`' code load and watching the iq4nl parity tests
panic.

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

### B6 — prefill reproducibility: the stated cause was WRONG; the real one is fixed

**Tag:** diagnosed + fixed 2026-08-02 · **Blocked on:** nothing; what is left is
the residual on the two smallest models, and it is not tier choice

**The original claim, kept because it is what was disproved.** Running the
35-row sweep twice against the same binary (`691c0dc`) gave `pp512` 6.8% mean /
34.5% worst and `pp4@d4096` 7.7% / 31.7%, against under 1% on both decode
columns, and the entry attributed it to "tier/chunk nondeterminism — a short
prefill can land on a different kernel tier between runs".

**That attribution does not survive measurement.** Three separate checks:

- **The kernels are identical.** `INFR_PROF_OPS=1` over six back-to-back `pp512`
  runs and eight `pp4@d4096` runs of Qwen3-0.6B Q4_K_M produced a byte-identical
  (op name, dispatch count) signature every time — 14 ops / 729 dispatches for
  `pp512`, 449 for `pp4` — while `pp4`'s reported throughput moved 1029.5 →
  1120.4 t/s. Different throughput, same kernels: no tier changed.
- **Nothing feeds a tier decision live VRAM at bench scale.** `adaptive_chunk`
  is a pure function of the KV span. `vulkan_moe_binder` takes ONE `vk.vram()`
  snapshot per load and every budget derives from it, and both rungs that could
  move a chunk (`dense_resident_rung`, the auto-q8 rung) log at WARN when they
  fire. A bench sizes its KV to `depth + p + g + 16` — 528 rows for `pp512` — so
  no model in the sweep is anywhere near a rung boundary. None of these fired.
- **`pp512` is now reproducible, so the 34.5% is not a live property of the
  tree.** Four full sweeps of the five named-worst models at `68a74b2`: `pp512`
  peak-to-peak 0.9% mean / 1.4% worst, `tg128` 0.6/1.1, `tg64@d4096` 1.0/1.7.
  Only `pp4@d4096` moved — 8.4% mean, 20.2% worst, on the IQ3_S MoE.

**What actually moved `pp4@d4096`: the first timed rep was a COLD rep.**
`bench_vulkan`'s untimed warmup was a hardcoded `(8, 2)` turn — 7 prefill rows
and 2 decode steps — which does not cover the shape about to be timed, so the
first TIMED rep paid that shape's one-time costs (its pipeline variants, its
first-touch scratch pools) inside the measured window and `avg_ts` averaged them
in. `INFR_PROF_STAGES=1` on Qwen3.6-35B-A3B UD-IQ3_S `pp4 @ d4096`, per-rep wall
for the timed m=4 chunk across six processes:

| rep | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 |
| --- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1   | 24.6  | 40.1  | 45.6  | 42.3  | 24.1  | 46.9  |
| 2   | 13.6  | 13.6  | 13.6  | 14.1  | 13.8  | 13.5  |
| 3   | 13.5  | 13.5  | 13.9  | 14.1  | 13.3  | 14.0  |

Reps 2-3 never leave 13.3-14.1 ms. Rep 1 costs 1.8-3.5x that and its scatter IS
the cell's variance (`avg_ts` 220.1-251.2, 13.4%). The same effect is only 3% on
`pp512` (155.5 / 153.1 / 150.4 ms) because there the fixed cost is small next to
the work — which is exactly why the two prefill columns disagreed about
reproducibility, and why the SHORTEST metric was the worst offender rather than
the one with the shortest prefill.

**Fixed** by running one untimed warm rep at the measured shape before the timed
ones — the same discarded warmup iteration llama-bench does, and what
`bench_vulkan`'s own doc already claimed. Four sweeps before, four after:

| column       | p2p before (mean / worst) | p2p after (mean / worst) |
| ------------ | ------------------------- | ------------------------ |
| `pp4@d4096`  | 8.4% / **20.2%**          | 4.3% / 7.9%              |
| `pp512`      | 0.9% / 1.4%               | 1.4% / 2.8%              |
| `tg128`      | 0.6% / 1.1%               | 1.4% / 1.9%              |
| `tg64@d4096` | 1.0% / 1.7%               | 1.2% / 1.7%              |

The named worst row collapses: Qwen3.6-35B-A3B UD-IQ3_S `pp4@d4096` went
223/200/246/244 (20.2%) to 293/294/294/290 (**1.4%**). The other three columns
are unchanged within their own noise — the small rises there are between-block
ambient drift, not an effect of the change, which cannot touch a steady-state
rep.

**`pp4@d4096` absolutes MOVED and are not comparable across this change.** The
MoE reads +26% (232 → 290 t/s) because a cold rep is no longer averaged into a
steady-state figure. The re-sweep (B14) must regenerate that column, not diff it
against the old table.

**What is still open, and it is not tier choice.** Qwen3-0.6B and gemma-3-1b
keep ~8% peak-to-peak on `pp4@d4096`. That metric times four tokens: ~3.7 ms
wall per rep of which ~2.8 ms is device time (`INFR_PROF_OPS`), so roughly a
quarter is host-side record/submit/fence and its jitter is what is left. Options
if it ever matters: raise `-r` for that column only, report the median instead
of the mean, or accept that a four-token measurement resolves nothing under
~10%. `infr bench` now prints the per-rep min-max and spread on every line, so
this is visible in the output instead of inferred.

**Two other things this slice found and changed:**

- **`infr compare --sweep` was completely broken at `68a74b2`** and printed
  `ERR` for every cell. `refactor: route library diagnostics through tracing`
  (`a6e9131`) moved the device-probe lines onto `tracing`, and
  `tracing_subscriber::fmt()` defaults to STDOUT — so `infr bench --json`
  emitted five INFO lines ahead of its `[{"avg_ts": ..}]` and
  `ModelBench::infr_json`'s `serde_json::from_slice` failed on all of them. The
  subscriber is now pinned to stderr, and
  `bench_json_line_parses_and_leads_with_avg_ts` guards the shape half (shown
  red by prefixing the line with a log line).
- **The submit splitter can latch on a wall-clock sample.**
  `VulkanBackend::observe_forward` arms `submit_dispatch_cap` on a discrete GPU
  when ONE forward exceeds `SUBMIT_DANGER_NS` (1 s), and the cap only ratchets
  down — so a slow first forward (a cold pipeline build, a loaded host)
  permanently changes the submit structure of every later forward in that
  process while leaving the dispatched kernels byte-identical, i.e. invisible in
  `INFR_PROF_OPS`. It never fired in any run measured here (every line reports
  `submit unlimited`). It now logs at WARN when it arms and `infr bench` reports
  the final cap. **UPDATE 2026-08-03: it is no longer only latent.** The
  `results.md` re-sweep caught it arming reproducibly — see B17.

**Coverage gaps in this diagnosis, stated plainly.** `llama-bench` on this box
is currently broken
(`/usr/lib/libllama.so.0: undefined symbol: ggml_dsv4_hc_post`), so all eight
sweeps ran infr-only with `NA` in the oracle column. The original two sweeps
interleaved ~30 s of full-GPU llama-bench work between every infr cell; that
thermal coupling is absent here and cannot be ruled out as a contributor to the
original `pp512` numbers. The subset was five models (the four sub-2B rows and
the IQ3_S MoE that B6 named), not all 35.

**Withdrawn from the original entry:** "pin the tier when `-r > 1`". There is no
tier to pin — the choice was already a pure function of the shape at bench
scale, and pinning it would have fixed nothing.

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

**Addendum, 2026-08-02 — three B14 measurement gaps closed.** All throughput
figures are `infr bench -p 0 -n 128 -r 3` on the 7900 XTX, legs alternated at
least twice, f16 KV; ON = shipped, OFF = `INFR_NO_ATTN_DECODE=1`. Per-op figures
are `INFR_PROF_OPS=1 INFR_SEAM_NO_REPLAY=1`.

_(1) qwen35, the model slice 3b was never benched on._ **Qwen3.5-9B Q4_K_M** —
all three cached Qwen3.5 GGUFs carry `qwen35.attention.key_length = 256`, so any
of them exercises the `-DDHD4=64` family; the 9B is the largest dense one and is
not the MTP variant, so it has the biggest attention share and no spec-decode
interaction. `attn_decode_hd256` confirmed dispatching, 8× per token — qwen35 is
a DeltaNet hybrid, so only 8 of its 32 layers are full attention.

| depth | ON            | OFF           | gain   |
| ----- | ------------- | ------------- | ------ |
| 4096  | 116.9 / 116.7 | 116.4 / 116.4 | 1.004× |
| 8192  | 113.7 / 113.6 | 112.7 / 112.7 | 1.008× |
| 32768 | 102.9 / 102.5 | 101.5 / 101.4 | 1.011× |

**A wash end-to-end, and for the same reason gemma is: SHARE, not kernel
quality.** Attention pass 1 is 2.93% of decode GPU time at d4096 and 12.42% at
d32768. The KERNEL is in fact the best result the specialization has recorded
anywhere: at d32768 `attn_decode_hd256` is **152.9 µs** against
`attn_partial_bda`'s **167.0 µs**, **1.09×** — and this one needs no modelling,
because every qwen35 attention layer is causal hd 256, so both legs put all 128
dispatches in a single profiler bucket.

_(2) `INFR_NO_ATTN_HD=1` — executed, and it does what it says._ Verified by
KERNEL NAME, not by inference from a throughput number:

- Qwen3.5-9B d4096: `attn_decode_hd256` 33.5 µs × 64 → `attn_partial_nohd_bda`
  **83.1 µs** × 64.
- gemma-3-12b d4096: `attn_decode_hd256` 40.1 µs × 128 + `attn_decode_hd256_swa`
  18.4 µs × 640 (16.9 ms) → one `attn_partial_nohd_bda` **45.3 µs** × 768 (34.8
  ms). BOTH arms fall back, causal and SWA.

Output is correct. Greedy `--temp 0 --max-new 120`, same prompt: on gemma-3-12b
all three legs (shipped / `NO_ATTN_HD` / `NO_ATTN_DECODE`) produce
**byte-identical** text. **But `NO_ATTN_HD` is NOT bit-identical** — on
Qwen3.5-9B it stays coherent and correct while splitting from the shipped output
at generated byte 302 of 396 ("Additionally, if 1 were prime, the Fundamental
Theorem…" vs "Additionally, excluding 1 from the set of primes ensures that the
Fundamental Theorem…"), where the shipped and `NO_ATTN_DECODE` legs are
byte-for- byte equal. Expected — `-DNO_HD_SPEC` deletes the specialized arm and
runs the general runtime-`hd4` loop, a different summation order — but it means
this knob is a DIAGNOSTIC, not a bitwise A/B, and it must not be used as one.

Cost of the hatch (tg128, alternated): Qwen3.5-9B d4096 116.9 / 116.8 → 112.0 /
111.9 (**−4.2%**), d32768 103.0 / 102.9 → 90.9 / 90.9 (**−11.7%**). Those legs
ran WITHOUT `INFR_SEAM_NO_REPLAY`, so the replay gate in
`attention_kv_split_dynac_impl` honors the knob too, not just the static one.

_(3) gemma at d32768 — the prediction's premise held and its conclusion did
not._ f16 KV confirmed two ways: no `kv auto-quant: q8_0` warning fired, and the
profile shows `attn_decode_hd256` / `attn_decode_hd256_swa` rather than any
`attn_partial_*q8*`, so the leg really is on the path.

| depth | ON (gemma-3-12b)          | OFF                       | ratio      |
| ----- | ------------------------- | ------------------------- | ---------- |
| 8192  | 83.9 / 83.8 / 83.6 / 83.6 | 83.3 / 83.3 / 83.1 / 83.0 | 1.006×     |
| 16384 | 79.6 / 79.6               | 78.9 / 79.0               | 1.008×     |
| 32768 | 70.8 / 70.8 / 70.6 / 70.6 | 71.9 / 71.9 / 71.7 / 71.7 | **0.985×** |

**At d32768 the specialization is a 1.5% REGRESSION**, reproducible to 0.1 t/s
across three alternated pairs. See **B15**.

What held: the SWA layers really are window-capped. `attn_decode_hd256_swa` is
**18.4 / 18.4 / 18.5 / 18.1 µs** at d4096 / d8192 / d16384 / d32768 — flat over
an 8× depth range.

What did not: the inference drawn from it. Attention pass 1 goes from **8.9%**
of decode GPU time at d4096 (16.9 ms of 190.4) to **23.8%** at d32768 (54.6 ms
of 229.8), because the 8 global layers scale 40.1 → 335.7 µs. The lever DOES
grow with depth; it just points the other way at the far end.

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

### B9a — the `tracing` sweep's infr-metal half is unverified off macOS

**Tag:** raised 2026-08-02 · **Blocked on:** a CI run, or an Apple machine

B9's sweep landed (bare `println!`/`eprintln!` diagnostics routed onto
`tracing`, guarded by `infr-core/tests/no_bare_print.rs`). Nine of its
conversions are in `infr-metal`, which is `#![cfg(target_os = "macos")]` and so
compiles to an empty lib on Linux: `cargo clippy --all-targets` and `cargo test`
here say **nothing** about them. The sites are `MetalBackend::new`'s two
counter-sampling fallbacks, `ArchiveCache`'s five pipeline-cache warnings, and
`Pipelines::new`'s two `prof.stages` lines. Each is a one-line macro swap
(`eprintln!` → `tracing::warn!`/`tracing::info!`) with the message text
untouched, so the risk is a compile error, not a behaviour change — but only CI
or an Apple build can say so.

`tracing` was added to `crates/infr-metal/Cargo.toml` under plain
`[dependencies]`, not the `cfg(target_os = "macos")` block, deliberately: the
crate body is cfg'd out off-macOS, so a target-gated dep would make the import
resolve only on Apple and hide any mistake even longer.

Also unverified: `infr_metal::profile`'s three-line report table and
`shaders.rs`'s pipeline-cache summary line were KEPT as `eprintln!` and marked
`// print-ok:`, on the same reasoning as `infr_core::prof::OpProf::flush` — a
per-line `tracing` prefix would destroy a column-aligned table. Nobody has seen
that table render since the change; it should not have moved, but it has not
been looked at.

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

### B11 — an explicit `INFR_CACHE=<pct>` still resolves against raw free VRAM

**Tag:** narrowed 2026-08-02 · **Blocked on:** a decision about what the
percentage should MEAN

The placement budgets that infr derives for itself — `vulkan_moe_binder`'s dense
residency predicate, its streaming budget and its MoE expert budget — now take
the allocator's ceiling (`VramInfo::alloc_room`, free minus the VRAM guard's 256
MiB headroom), the same function the context-fit math uses, guarded by
`budgets_agree_with_the_allocator_ceiling` and
`fit_math_and_placement_pick_the_same_rung`.

Both `INFR_CACHE` tiers still resolve a percentage spec against `vram.available`
(`spec.resolve(..)` in the MoE and dense override arms). Left alone
deliberately: that value is the CALLER's budget, the grammar is documented as "a
percentage of the device's AVAILABLE VRAM", and the override exists to force a
placement the auto tiers would not choose — so failing loudly at the alloc guard
is defensible where silently handing back less than asked is not.
`INFR_CACHE=100%` is the case that would trip it. Decide whether the percentage
means "of free VRAM" (today) or "of what can actually be allocated" before
changing it; either way it is one `spec.resolve` argument per arm.

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
- **The tightened placement budgets were only exercised on their RESIDENT
  branch** (the 2026-08-02 B11 slice). gemma-4-31B, gemma-3-12b and
  Qwen3-30B-A3B all stay resident on this box, so `dense_stream_budget_at` and
  `moe_expert_budget`'s `None` arm (dense weights + KV past the ceiling — the
  hard error) were verified only by unit test, never by a run that actually
  streams or pages. A model that does not fit this card would cover both;
  `INFR_CACHE=<size>` forces the streaming path but with the caller's budget,
  not the derived one.

### B15 — `attn_decode` crosses over and LOSES on gemma above d16384

**Tag:** measured 2026-08-02 · **Blocked on:** a decision — gate the
specialization, or accept 1.5% at deep context on gemma

`attn_decode` (B7 slices 3a/3b) is bit-identical to `attn_partial` and leaner on
paper (96 VGPRs / 3072 B LDS against 120 / 5120), and it wins at every depth
measured before this one. On **gemma-3-12b at d32768 it loses 1.5%** (tg128 70.8
/ 70.8 / 70.6 / 70.6 ON against 71.9 / 71.9 / 71.7 / 71.7 OFF).

**Where the time goes.** At d32768 the specialized family costs 128 × 335.7 µs +
640 × 18.1 µs = **54.6 ms** across the same 768 pass-1 dispatches that
`attn_partial_bda` serves in **51.0 ms** — 7.1% slower on attention, which lands
as +1.4% on the device total (229.8 vs 226.6 ms) and shows up as the 1.5%
throughput loss. `attn_combine` is untouched (9.3 ms both legs), so the whole
difference is pass 1.

**The global (non-SWA) layers are the regressor, and the argument needs no
modelling.** Because the SWA kernel is flat in depth, the d8192 → d32768 DELTA
belongs entirely to the 128 global dispatches: ON `attn_decode_hd256` goes 79.2
→ 335.7 µs (**+256.5**), OFF `attn_partial_bda`'s total goes 22.3 → 51.0 ms
(**+224.4 µs** per global dispatch). The specialized kernel scales **14% worse
with depth**. Taking the SWA cost as equal in both legs (18.4 / 18.4 / 18.5 /
18.1 µs, measured on the ON leg) the implied per-dispatch global cost is:

| depth | `attn_decode_hd256` | implied `attn_partial_bda` global | ratio |
| ----- | ------------------- | --------------------------------- | ----- |
| 4096  | 40.1                | ~43                               | 1.08× |
| 8192  | 79.2                | ~82                               | 1.04× |
| 16384 | 153.4               | ~163                              | 1.06× |
| 32768 | 335.7               | ~308                              | 0.92× |

**Crossover is between d16384 and d32768**, and end-to-end tg128 agrees: 1.008×
at d16384, 0.985× at d32768.

**Why gemma and not qwen35, at the same head dim and the same depth.** Both are
`nh=16 hd=256`, so both grids move the same 537 MB per layer-token — but gemma
has `nkv=8` where qwen35 has `nkv=4`, so gemma's UNIQUE KV footprint is 268 MB
against 134 MB and it gets half the re-read reuse out of the 96 MB Infinity
Cache. Achieved rate at d32768: qwen35 537 MB / 152.9 µs = **3.5 TB/s**
(cache-served, and the specialization wins 1.09× there); gemma 537 MB / 335.7 µs
= **1.6 TB/s**, far closer to DRAM. HYPOTHESIS, consistent with all the data but
NOT directly instrumented (no counter was read): the leaner kernel's extra
occupancy — 16 waves/SIMD against 12 — helps while the kernel is
occupancy/latency-limited and hurts once it is streaming-bound, because more
concurrent streams cost locality. Reading RGP/`RADV` memory counters on the two
builds at d32768 would settle it.

**The options.** (a) Leave it: 1.5% on one model at one depth, against 4–7% on
Qwen3-30B-A3B and 1.09× on the qwen35 kernel. (b) Gate `attn_decode` on
something that predicts the regime — the crossover tracks `kv_len * nkv * hd`
against cache size, not depth alone, so a depth threshold would be wrong on
qwen35, which is still gaining at d32768. (c) Chase the occupancy hypothesis and
fix the kernel rather than gate it. Nothing here is urgent; it is recorded so
the next profile does not rediscover it as a mystery.

**Related, not the same thing:** `docs/perf/results.md` used to say "the
decode-at-depth cells are now stale by 4–8%". The 2026-08-03 re-sweep settled
that: the `tg64@d4096` column moved **+0.021× averaged over all 35 rows**, with
the gain concentrated on Qwen and MoE rows (Qwen3-30B-A3B 0.91× → 0.96×,
Qwen3-14B Q4_K_M 0.94× → 1.00×) and nothing measurable on gemma (Gemma-3-12B
1.13× → 1.14×). The doc now says that instead of extrapolating. The d32768
regression this entry is about is beyond the table's depth and unaffected.

### B16 — one decode leg in thirteen was 8.9% low

**Tag:** measured 2026-08-02, cause identified 2026-08-02 · **Blocked on:**
nothing; recorded so it is not mistaken for a real effect

**Very likely the same cold-first-rep effect B6 diagnosed and fixed** — the
suspicion below ("a cold shader/pipeline cache or a first-touch VRAM effect")
was right, and B6 measured it directly: before the fix, rep 1 of a bench cost
1.8-3.5x the steady-state rep and only rep 1. The tell here is the same one: "it
was the FIRST leg run against that model". NOT re-measured on this shape
(gemma-3-12b `tg128 @ d8192` with `INFR_NO_ATTN_DECODE=1`), so this is a strong
inference, not a verified claim — the original 13 legs would have to be re-run
to confirm the 8.9% outlier is gone. Everything below is the original record.

B6 establishes that decode is reproducible to ~1% on this box while prefill is
not. That is right on average and it is not a guarantee. Benching gemma-3-12b
`tg128 @ d8192` with `INFR_NO_ATTN_DECODE=1`, the first leg of the first
sequence measured **75.9 t/s**; three later repeats of the identical command
measured **83.3 / 83.1 / 83.0**, and the ON legs of the same sequences were 83.9
/ 83.8 / 83.6 / 83.6. So one leg in thirteen came in **8.9% low** — enough to
invert a 1% A/B — and it was the FIRST leg run against that model.

Not diagnosed. The obvious suspects are a cold shader/pipeline cache or a
first-touch VRAM effect; nothing else was on the GPU. The practical consequence
is the one B7 already applies: alternate legs at least twice and report every
repeat, never a single value or an average that hides the spread. A single-leg
A/B on this box can be wrong by 9%.

### B17 — the submit splitter arms for real on Qwen3.6-27B at d4096

**Tag:** measured 2026-08-03 (the `results.md` re-sweep) · **Blocked on:** a
decision — soften the trigger, or accept that one table row is measured with a
different submit structure

B6 recorded `VulkanBackend::observe_forward`'s submit splitter as a latent
hazard that had never fired. It fires. Benching **Qwen3.6-27B Q4_K_M** at
`-d 4096`, the untimed depth prime is a single 1633-dispatch forward that takes
**~1.01 s** — just past the 1 s `SUBMIT_DANGER_NS` threshold — so the cap
latches and every later forward in the process, **including the timed ones**,
splits every ~400 dispatches. Reproduced in **3 of 3** processes (caps 392 / 401
/ 403, read from the `submit_cap` field `infr bench --json` emits and the
matching WARN line). It also arms on **Qwen3-30B-A3B** on four of the nine
deep-context legs: cap 269 on `pp512`/`tg128` at `-d 32768`, and 342 / 222 on
`pg8192,512` at d16384 / d32768. It does not arm on the d16384 `pp512`/`tg128`
legs or anywhere at d8192, so the trigger tracks the single longest forward in
the process, not the depth.

**Why it matters.** The dispatched kernels stay byte-identical, so this is
invisible to `INFR_PROF_OPS` and to any golden; only the submit structure
changes. Two consequences:

- `results.md`'s Qwen3.6-27B `tg64@d4096` and `pp4@d4096` cells, and every
  deep-context row past d8192, are measured under a split submit while every
  other cell in the table is not. The doc flags this; the cells are not wrong,
  they are just not the same experiment.
- The trigger is a **wall-clock sample of one forward, at a threshold this model
  sits within 2% of**, so a slightly warmer or busier box would flip the row.
  Reproducing 3/3 here is the luck of that margin, not stability.

**What was NOT done:** no A/B of the split against `INFR_SUBMIT_DISPATCHES=0` on
this model, so the size of the effect on those two cells is **unknown** — it may
be nothing. That measurement is the obvious next step and is cheap.

**Options.** (a) Leave it. (b) Exclude the untimed depth-prime forward from
`observe_forward`'s sampling, which is arguably what the guard always meant —
the watchdog risk is about user-visible forwards, and the prime is one
deliberate bulk operation. (c) Raise the threshold. (b) looks right but changes
iGPU watchdog behaviour, so it is not a one-liner — see `docs/igpu.md`.

### B18 — three rows the 2026-08-03 re-sweep left unexplained

**Tag:** measured 2026-08-03 · **Blocked on:** nothing; each needs a profile

Recorded so the next `results.md` reader does not have to re-derive them. All
three are reproducible, not variance: `pp512` and both decode columns now repeat
to under 3.4% peak-to-peak over four full runs (see the doc's variance box).

- **Gemma-3-1B Q4_K_M `pp512` = 0.95×, and it is real.** Four independent runs
  gave 0.95 / 0.95 / 0.95 / 0.94. The doc used to dismiss the small-model
  `pp512` cluster as "prefill variance, not a real deficit"; that explanation
  died with the warm-rep fix. It is **dtype-specific, not architectural** — the
  same model's Q2_K reads 1.05× and its Q8_0 1.30× on the same column. Nobody
  has profiled the Q4_K prefill path on a 1B gemma.
- **Qwen3-14B Q8_0 `pp4@d4096` = 0.95×** — the table's only `pp4@d4096` loss.
  `results.md` footnote ⁷ used to imply this cell was 1.18×; that figure was the
  legacy-int8-GEMV slice's own before/after A/B, never a table cell, and the doc
  now says so. Why the small-m prefill path loses on Q8_0 specifically, when the
  same slice's +28.8% held on every other legacy format, is unexplained.
- **Llama-3.2-1B `tg64@d4096` = 0.88× (Q8_0) / 0.94× (Q4_K_M)** — the worst cell
  in the table, on the smallest model in it, while both of that model's other
  columns win. Reproducible (0.92–0.94× on Q4_K_M across four runs). An isolated
  small-model decode-at-depth deficit with no named cause.

**Coverage gaps in that sweep, stated plainly.** The Ternary-Bonsai Q2_0 table
(CPU oracle), the Llama-4-Scout pager figures and DiffusionGemma's `dg-e2e` were
**not** re-measured — of the DG pair only `dg-step` was. Metal was not measured
at all. The oracle was a cached `llama-cpp-vulkan-b9833` release build run
through an `LD_LIBRARY_PATH` shim, because the distro `llama-cpp` (b10182) links
ggml 0.17 against an installed `ggml-vulkan` 0.15.3 and **no system llama.cpp
binary runs on this box** (`undefined symbol: ggml_dsv4_hc_post`). The packaging
fix is not ours; the shim lived in session scratch and will not survive, so the
next sweep needs its own working oracle before it starts.

### B21 — `DiffusionGemmaChat` discards `RequestCtx`: no cancellation, no per-request seed

**Tag:** CR-2026-08-03 M3 (verified) · **Blocked on:** threading an abort poll
and a seed through `diffusion_generate`'s block loop

Both `ChatModel` entry points on `DiffusionGemmaChat`
(`crates/infr-llama/src/chat/diffusion.rs`) take `_req` and drop it. The impl's
own doc comment already discloses the gap; what it does not say is how far the
gap reaches now that `infr serve` hosts DG.

`cmd_serve` in `crates/infr-cli/src/main.rs` routes diffusion-gemma to the
SERIALISED path (`is_vulkan = !is_dg && ..`) — one request at a time behind a
Mutex. `run_chat` creates the `RequestCtx` and its `on_piece` calls
`req.abort()` on a stop-sequence hit or a latched `cancel`. On DG none of it
lands:

- **A client disconnect does not stop the generation.** `streaming`'s failing
  `tx.send` latches `cancel`, `run_chat` calls `req.abort()`, and
  `diffusion_generate`'s block loop polls nothing — all `blocks_wanted` blocks
  still run.
- **Neither does the deadline.** `serve.request_timeout_secs` works by latching
  that same `cancel` via `arm_deadline`. B5 says the deadline "bounds how long
  one request can hold a `--parallel` slot"; on DG it bounds nothing, and
  because the DG path is serialised the next request waits behind the whole
  thing.
- **`on_piece` fires once per finished BLOCK**, not per token, so even the poll
  points would be coarse.
- **The per-request seed is ignored.** `GenParams.seed` is a real accepted field
  and `request_sampling` copies it into `RequestSampling.seed`; `generate_impl`
  resolves `self.model.engine_cfg().sampling.seed.unwrap_or(42)` instead, so two
  requests with different seeds produce identical output.

Combined with B20 (a `max_tokens: 1` request generating a whole canvas), the
worst case is a disconnected client leaving the single DG slot busy for
`ceil(max_new / canvas_length)` blocks with nothing able to interrupt it.

### B26 — `matmul_f32` leaks its transient Vulkan handles on every error path

**Tag:** CR-2026-08-03 L3 (verified, and narrower than filed) · **Blocked on:**
nothing; the open question is whether it is worth touching at all

`crates/infr-vulkan/src/matmul.rs`: `VulkanBackend::matmul_f32` creates a shader
module, descriptor-set layout, pipeline layout, pipeline and descriptor pool as
raw handles and destroys them only in the success tail. Every `?` between them
leaks whatever was already created, and so does the explicit "driver returned
VK_SUCCESS with a null pipeline handle" early return, which abandons the shader
module and both layouts. The buffers are fine — `buf_a`/`buf_b`/`buf_c` are
RAII.

**What the review missed, and it is the whole severity story:** the function is
`#[doc(hidden)]` and its own doc says "ONE-SHOT bench/test helper (only callers
are `examples/smoke.rs` and `test_matmul_f32`) … NOT on any production path". A
grep confirms those are the only two callers. So the leak is real code and
unreachable from `infr run` / `serve` / `bench`, and it leaks per failed call in
a process that is about to exit anyway.

Worth recording anyway because the same doc comment's step 5 — "Destroys all
transient Vulkan objects (pool, pipeline, layouts, shader module)" — is a
factual claim that holds only on success, and that is the kind of comment the
next reader trusts without checking.

**If it is fixed:** one `(|| { .. })()` inner closure whose `Err` falls through
to the existing destroy block, rather than five separate `map_err` cleanups.

### B27 — hardening candidates from the 2026-08-03 review

**Tag:** CR-2026-08-03 hardening · **Blocked on:** nothing; none of these is an
established defect and none was verified in the fold-in pass

Kept with the deleted report's framing intact: the review listed these as **"not
established current defects"** — places where a stronger construction would
survive a case nobody has shown to occur. Do not promote one to a bug without
first exhibiting that case.

- `with_profiling_suppressed` restores a process-global boolean only on normal
  return; an RAII nesting counter would survive panics and overlapping scopes.
- `SpinPool::collect` leaks already-initialised values when another task panics;
  unwind cleanup could track and drop initialised slots. (Adjacent to B25, but a
  different mechanism — that one is attribution, this is cleanup.)
- Existing HuggingFace `blobs/<expected_sha>` files are trusted by pathname and
  existence without rehashing; optional verification would catch local cache
  corruption.
- Public `dequant_block` assumes block-sized input. The reviewed GGUF callers
  supply validated slices; an explicit length check would protect direct
  callers.
- The streaming SSE channel is unbounded, so a non-reading client can retain a
  whole completion in memory.

**That last one is NOT B5, and it is already a recorded decision.** B5 declined
RATE limiting — how many requests one client may make — because a reverse proxy
owns that. This is per-stream memory retention on an already-admitted request,
and `streaming`'s own comment in `crates/infr-server/src/lib.rs` argues the
choice explicitly: a bounded channel would push backpressure into `on_delta`,
which runs inside the decode loop while it holds the GPU baton, so one slow
client would stall every other sequence. Retention is bounded by `max_tokens`,
itself capped by `clamp_max_tokens` / `max_tokens_cap`. Treat it as decided
unless someone measures the retention and finds the bound too loose.

### B28 — what the 2026-08-03 review CLEARED

**Tag:** CR-2026-08-03 cleared · **Blocked on:** nothing; recorded so it is not
re-investigated

Investigated at low depth and found sound. These were NOT re-verified in the
fold-in pass — that pass verified the review's FINDINGS, not its clearances — so
treat each as "one reviewer looked and was satisfied", never as tested.

- CLI backend strings such as `vulkanfoo` do not silently select Vulkan device
  0; the downstream numeric parse rejects the suffix.
- GGUF loading validates metadata depth, tensor dimension arithmetic, alignment,
  duplicate names, quantisation block divisibility and mapped-file bounds before
  exposing tensor bytes.
- `StopMatcher` preserves split stop prefixes and UTF-8 boundaries.
- Vulkan upload, download and copy paths validate buffer extents.
- Vulkan external-memory file descriptors transfer ownership on a successful
  import and close the duplicate on failure.
- The Metal derived-buffer cache keys include monotonic allocation identity, so
  a recycled address cannot produce a false cache hit.
- Autoregressive generation handles `max_new == 0` explicitly; the budget defect
  is confined to block diffusion (B20).
- `SpinPool` waits for every worker check-in before releasing an ordinary
  borrowed job; the surviving defect is panic attribution between jobs (B25).
- Pager production callers reject zero slots before construction.
- Dense-runner token ids are validated before embedding lookup on the reviewed
  generation paths.
- Malformed Hermes tool markup is gated out of production streaming and removed
  rather than surfacing as assistant content.

### B29 — what the 2026-08-03 review did NOT cover

**Tag:** CR-2026-08-03 coverage · **Blocked on:** nothing; each line is a gap,
stated as one

The review was explicitly **low depth** and **ran no build and no tests**. It
traced core graph / pager / sampling / pool / backend boundaries; llama's
autoregressive, diffusion, grammar, chat and request-context flow; server
validation, admission, deadlines, streaming and non-streaming responses,
statistics and cancellation; hub selection, shard expansion, cache resolution,
downloads, integrity, symlinks and path validation; GGUF metadata/tensor
validation and host dequant entry points; CLI backend/model parsing and the
serve generation adapters; chat-template, reasoning, stop and tool-call parsing;
profiling aggregation and JSON output; selected Vulkan and Metal allocation,
transfer, cache, synchronisation, dispatch and lifecycle paths; and the testkit.
It did NOT go line by line through:

- SIMD/scalar numerical bodies and architecture-specific branches in
  `crates/infr-cpu/src/kernels.rs` (see also B3).
- Every graph-rewrite combination in `crates/infr-core/src/fusion.rs`.
- The static quantisation tables in `crates/infr-core/src/iquant_grids.rs`.
- Every MTP and per-architecture seam graph formula.
- Most Vulkan recorder, adapter, GEMM, pager, tensor-parallel, expert-routing
  and shader host/ABI combinations (see also B4).
- Most of `crates/infr-metal/src/exec.rs`, and full Metal shader/host ABI
  validation (see also B9a, B14).
- Every quantisation arm in `crates/infr-gguf/src/dequant.rs`.
- Large CLI benchmark and diffusion-specific command paths.
- Every test assertion and example.
- Live Vulkan or Metal execution — nothing ran on a device.
- Platform-gated macOS code, which was read but never compiled.

---

## Withdrawn

Recorded so they are not rediscovered and re-investigated.

**Nothing from the 2026-08-03 review landed here.** All eight of its findings
were re-verified against the code and all eight survived — seven outright
(B19–B25) and one narrowed by a scope the review had missed (B26: the leaking
function is a `#[doc(hidden)]` test/bench helper with no production caller).

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
