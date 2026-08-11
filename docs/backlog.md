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
  code and became B19–B26, all eight of which have since been fixed and deleted
  from here (`git log -S'### B19' -- docs/backlog.md` finds them). That pass's
  cleared / hardening / coverage lists survive as B27–B29. The tags on B1–B5
  come from the earlier 2026-08-01 pass, whose text the file had already stopped
  carrying (`6ab8b1c` overwrote it with the later review). A `CR-*` tag is
  therefore a historical marker for where an item came from, not a link to
  anything.

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

### B6 — a four-token bench column resolves nothing under ~10%

**Tag:** diagnosed + fixed 2026-08-02 · **Blocked on:** nothing; what is left is
a measurement floor nobody intends to chase

The prefill columns' 6.8–34.5% run-to-run spread was **not** tier
nondeterminism, which is what this entry originally claimed: `INFR_PROF_OPS=1`
over six back-to-back runs produced a byte-identical (op name, dispatch count)
signature while throughput moved 1029.5 → 1120.4 t/s, and nothing feeds a tier
decision live VRAM at bench scale. The cause was that `bench_vulkan`'s untimed
warmup was a hardcoded `(8, 2)` turn, so the first TIMED rep of any other shape
paid that shape's one-time costs — pipeline variants, first-touch scratch pools
— inside the measured window. Fixed by warming at the measured shape.
`pp4@d4096` peak-to-peak went 8.4% / 20.2% to 4.3% / 7.9% over four sweeps each
way, and the worst named row (Qwen3.6-35B-A3B UD-IQ3_S) 20.2% → **1.4%**.

**The residual, accepted.** Qwen3-0.6B and gemma-3-1b keep ~8% peak-to-peak on
`pp4@d4096`. That column times four tokens — ~3.7 ms wall per rep of which ~2.8
ms is device time — so roughly a quarter is host-side record/submit/fence and
its jitter is what is left. Options if it ever matters: raise `-r` for that
column only, report the median instead of the mean, or accept that a four-token
measurement resolves nothing under ~10%. `infr bench` prints the per-rep min-max
and spread on every line, so this is visible in the output rather than inferred.

**Methodology that outlived the diagnosis:** alternate A/B legs at least twice
and report every repeat, never a single value or an average that hides the
spread. A single-leg A/B on this box has been wrong by 9% — that outlier (one
leg in thirteen, gemma-3-12b `tg128 @ d8192`, 75.9 against a 83.0–83.3 cluster,
and it was the FIRST leg run against that model) was measured on the pre-fix
binary and never re-run, so it is a reason for the rule and not evidence about
the current tree.

### B7 — decode at depth: two designs declined, one landed, one still open

**Tag:** measured 2026-08-02 · **Blocked on:** nothing; slices 3a and 3b have
LANDED and are deleted from here — what remains open is (b), width by workgroup
count

The largest remaining gap to llama.cpp is decode at depth, not prefill.
Qwen3-30B-A3B Q4_K_M on a 7900 XTX against `llama-bench c629da5`, `tg128`
infr/llama: **138.7 / 165.1** @d8192, **106.5 / 140.3** @d16384, **66.9 /
112.1** @d32768 — 0.60× at depth while `pp512` holds parity. (Those are the
PRE-slice-3a baselines; `INFR_NO_ATTN_DECODE=1` reproduces them.)
`attn_partial_bda` is **59% of decode GPU time** at d32768 and scales exactly
linearly with KV.

**Two designs are DECLINED — do not re-try either as written.** Both reached
agreement with the reference (GQA bit-identically, the k-tile to 9.6e-7), so
they were correct, just slower:

- **GQA head-grouping**, one workgroup per (KV-head, chunk) covering all
  `g = nh/nkv` query heads: cuts K/V traffic 8× (537 → 67 MB per layer-token)
  and measures **329 µs against 177 µs — 1.87× SLOWER**. Grouping serializes 8
  cross-lane reductions into one wave that previously ran on 8 CUs; the re-read
  it eliminates was nearly free out of Infinity Cache. Not starvation either:
  re-run at matched parallelism it measured 359 µs, with `attn_combine` going
  24.5 → 146 µs on the 8× larger `pacc`. Fewer keys per workgroup also loses
  (chunk 256 on the ungrouped kernel: 314 µs). So neither traffic nor occupancy
  is the lever. Reverted; the tree is unchanged.
- **The LDS-staged K-tile** — the design `recorder.rs`'s
  `attention_kv_split_impl` comment names ("per-thread full dots, no cross-lane
  reductions, which is how llama.cpp wins that cell"). Best of four configs is
  **2.7× slower** at d32768 (382 µs against the shipped 184). Not an
  implementation miss: the ISA shows **0 cross-lane ops** against
  `attn_partial_bda`'s 54, so the reduction really is gone — the LDS transpose
  that buys its removal costs more than the reduction saved, because the K tile
  has ZERO data reuse (every byte written once, read once) and time is monotone
  in LDS budget (34 KB → 687 µs, 17 KB → 494, 9 KB → 382). Survives unwired as
  `tests/attn_ktile_probe.rs` + `shaders/attn_ktile.comp` because it is the
  measurement rig for the next attempt.

**What the oracle actually does**, read rather than guessed (`ggml-vulkan.cpp`,
`get_fa_tuning_params_scalar`, our shape on RDNA3): `path = FA_SCALAR` — coopmat
is **deliberately avoided at decode** ("scalar is faster than coopmat when
N==1"), which kills the matrix-core / `gqa_ratio` idea this entry once proposed;
`shmem_staging = 0` on AMD, independently corroborating the k-tile negative;
`block_rows = 1`, `block_cols = 64`, `workgroup_size = 128`; and
**`d_split = 8`** — the width of the group that cooperates on one key's dot, the
one parameter none of the experiments varied.

**The `d_split` sweep** (`shaders/attn_partial_dsplit.comp`,
`tests/attn_dsplit_probe.rs`) produced two SEPARABLE findings, and the `w=32`
control is what separates them:

1. **Specialization is free 6–9% at every depth with no algorithmic change** —
   `w=32` reproduces the shipped mapping exactly yet beats it, because a
   decode-only copy allocates **96 VGPRs against 120** with zero spills either
   way, so more waves fit per SIMD. This is what slices 3a/3b shipped.
2. **Narrow width wins only where workgroup parallelism is short**, monotone in
   workgroup COUNT rather than depth: 512 wg → best 2.01×, 1024 wg → 1.46×, 2048
   wg → every width loses. At depth the kernel is already at ~3.0 TB/s
   (Infinity-Cache rate) and splitting a wave's contiguous 512-byte K read into
   `32/w` segments costs more than the shallower reduction saves. So llama.cpp's
   `d_split = 8` is right for ITS configuration, not universally, and **the
   original B7 target (d32768) remains a negative**.

**Still open — (b) width by workgroup count.** Choose `w` from `nh * n_chunks`
against the device CU count rather than from depth. Needs shapes the probe never
covered — it only tested `nh=32 nkv=4 hd=128` — and its per-width numbers were
measured against the OLD reference, so the ratios overstate the remaining
headroom by the 4–8% slice 3a already took. Treat it as a mid-depth lever: it
does NOT close the headline gap, where the model is still ~0.64× at d32768.
Before leaning on that probe again, spend ten minutes on **the unexplained
9.6e-7 drift at w=32** — `attn_decode` itself is bit-identical, and neither
making every reduction `subgroupClusteredAdd(., 32u)` nor constant-folding
`sqrt(float(pc.hd))` reproduces it. Also note `dsplit_bench`'s "SHIPPED
reference" leg now measures `attn_decode`, so re-run it under
`INFR_NO_ATTN_DECODE=1` to compare against the old baseline.

**Still falling back to `attn_partial`** — each would need its own member of the
`attn_decode` family: planar-Q8 and mainline-inline quant KV (**this is what
gemma runs by DEFAULT at full context, so it is the biggest remaining coverage
hole**), the DiffusionGemma canvas mask, `rows > 1` (small-m spec-verify and
prefill), the bound-SSBO (non-BDA) dispatch, `chunk > 512` (only reachable above
~524k keys under `INFR_KV_OVERFLOW`), head dims other than 128/256/512, and a
RING cache on a `window == 0` layer (unreachable today — only SWA layers are
allocated as rings — and the static gate's row bound rejects it rather than
assuming).

**Two traps the shipped kernels carry, worth reading before touching them.**

- The hd 256/512 QK tail loop must keep `attn_partial`'s redundant
  `if (r < hd4)` guard AND read `hd4` from the push constant at runtime. Folded
  to a build-time literal the guard vanishes, ACO fuses the terms into one FMA
  chain, and the chunk score moves 1 ULP — which `attn_combine`'s `exp(m_c - M)`
  weight turns into 136/2048 non-identical outputs. The shader says so; do not
  "clean it up".
- `INFR_NO_ATTN_HD=1` is a **DIAGNOSTIC, not a bitwise A/B**. It deletes the
  specialized arm for the general runtime-`hd4` loop — a different summation
  order — and on Qwen3.5-9B the output stays coherent and correct while
  splitting from the shipped text at generated byte 302 of 396, where shipped
  and `INFR_NO_ATTN_DECODE` are byte-equal. Verified BY KERNEL NAME that both
  the static and replay gates honour it. Cost: −4.2% at d4096, −11.7% at d32768.

**Why the gains are model-shaped:** 4–7% on Qwen3-30B-A3B but ~0.5–1% on gemma
and a wash on Qwen3.5-9B, because attention SHARE differs (2.9% of decode GPU
time on qwen35 at d4096, 8.7% on gemma-3-12b), not because the kernel is worse —
at d32768 `attn_decode_hd256` beats `attn_partial_bda` 152.9 vs 167.0 µs on
qwen35. Gemma's 40 SWA layers are window-capped (18.4 µs flat from d4096 to
d32768), so most of its layers cannot grow with depth. The one place the
specialization LOSES is gemma above d16384 — see B15.

### B8 — what is still ESTIMATED after the measured-fit slice

**Tag:** re-measured 2026-08-04 · **Blocked on:** nothing; each item below is a
bounded piece of work, and none of them is currently causing a failure

The original entry claimed the reserve's runaway term was the non-flash score
tile, priced as "2 live pools at the final ctx" while the real pools "ACCUMULATE
across the ~150+ chunks of a deep prefill". **Both halves were wrong**, and the
measurement that settled it is now permanent machinery:
`Backend::activation_peak` (a high-water mark of live `Activations` bytes) and
the runner's `activation reserve too low` warning. Each chunk's `execute` drops
its pool before the next builds, and every layer of a chunk shares one `kv_len`,
so ONE tile is live — the model was over-reserving, by 3.5x at a 128-row chunk.

What actually broke the fit, on the reported gemma-4-31B UD-Q5_K_XL case
(reproduced at margin 1.0, `bench -p 0 -n 4 -d 19000`, which died on a 4 MiB
alloc **2 MiB** past the guard budget):

| term            | planner        | actual                         | delta   |
| --------------- | -------------- | ------------------------------ | ------- |
| weights         | 21 871 930 488 | 22 353 012 736 (217 arena blk) | +481 MB |
| KV              | 2 559 590 400  | ~2 504 MB                      | −56 MB  |
| activation peak | 911 343 616    | 262 MB                         | −649 MB |
| driver-side     | 0              | 187 MB after load → 368 at pk  | +368 MB |

So the fix was not a better activation model: it was to stop estimating the
things the device can be ASKED about. `reclamp_ctx_to_live_room` re-decides the
window between the weight upload and the KV allocation, against
`Backend::device_alloc_room`. What is still predicted at that point:

- **The activation reserve**, now re-fit to measured peaks with named MoE and
  DeltaNet terms and a 1.5x pad (`ACT_RESERVE_PAD`, sized by the worst arch
  measured — Qwen3.5-4B-MTP at 1.42x).
- **`POST_KV_DEVICE_RESERVE`** (256 MiB) — the pipelines/descriptors the driver
  builds while recording the first forwards, measured at 181 MiB on the largest
  model here.

**What is left, in the order it matters.**

1. **The per-arch activation algebra is whack-a-mole, and the structural fix is
   known.** Every term in `dense_act_reserve_at` re-derives, in the seam, what
   `runner.rs`'s `build` closure already declares exactly (`g.internal(...)`,
   each `batch * <width>`), plus what the Vulkan adapter's `pooled(...)` sites
   allocate. Fitting it per arch found three misses in one afternoon (MoE expert
   scratch, qwen35's DeltaNet mixer, qwen35's double-width `qg`/`gate_a` pair) —
   each caught by the new warning, none by a test. The fix is to SUM the graph's
   `Internal` tensors instead of modelling them, which needs the graph buildable
   for a shape before the KV cache exists; today `build` is defined after the
   cold-init block and closes over the KV handles. The pooled attention/MoE
   terms would still need the adapter's tier predicate (the old option 1).
2. **The chunk rung is still chosen pre-load, against the light weight
   estimate.** The re-clamp can LOWER it (`repin_ubatch_lower`) when that buys
   context, but placement's resident-vs-stream decision itself is unchanged and
   ~2% optimistic on weights. It self-corrects into a smaller window rather than
   a failure, so this is a context cost, not a correctness one. Whenever the
   reserve moves, re-check `docs/perf/results.md`'s placement box, which names
   the rung a model settles on: gemma-3-12b went 256 → 1024 rows at ctx 131072
   in this slice (780 t/s, was 760), while gemma-4-31B's documented 256 rows at
   d4096 and 1024 rows at `pp512` were re-measured and still hold.
3. **The context is ~1.5k tokens more conservative than the device strictly
   requires.** gemma-4-31B advertises 14 592 (the interim-margin build said 15
   872, and 19 021 was measured to miss by 2 MiB). The gap is
   `POST_KV_DEVICE_RESERVE` + the allocator's own 256 MiB `GUARD_HEADROOM` + the
   reserve's pad + `KV_SLOP_ROWS` (B13). Reclaiming it means measuring the
   driver-side growth on more than one GPU first — 181 MiB is one sample, on one
   RADV version.
4. **Every measurement here is from one 7900 XTX on RADV.** The arena-tail
   percentage, the driver-side growth and the activation peaks are all
   Mesa/RADV/discrete numbers. An Intel or NVIDIA driver could report its budget
   with different granularity, and an iGPU shares the heap with the host.
5. **Metal and CPU are untouched by design**: `device_alloc_room` and
   `activation_peak` default to `None`, so both keep exactly their previous
   behaviour and neither gets the measured clamp. Metal has a working-set query
   that could implement the first.
6. **`infr serve --parallel N` was exercised at N=2** (two 40960-token slots on
   Qwen3-1.7B, a request served) but not at a size where the re-clamp fires, so
   the interaction between a shrunk window and `vulkan_slot_ctx`'s divide-by-N
   is reasoned, not measured.

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
262144 → ~50k and runs at 148 t/s; gemma-4-31B clamps 262144 → 14 592 and fills
it at 30.3 t/s). Only "no usable context at all" may refuse.

**To do:** decide whether an explicit oversized `--ctx` should fail early with
the detailed message (a behaviour change on a path documented as "never clamped
— the user asked") or keep failing at the alloc guard. If early: the check must
be provable, i.e. exact KV bytes alone + weights > `alloc_room()`, with no
activation reserve in it, so an over-estimating reserve can never refuse a run
that would have worked.

**Since 2026-08-04 the explicit path at least says so first.**
`reclamp_ctx_to_live_room` runs on it too and WARNs, naming the window that does
fit the device's measured free memory, before honoring the one that was asked
for. The decision above is now only about whether to escalate that warning to an
error — and the "provable check" it asks for is exactly what that path already
computes.

### B13 — the `+64` rows in every KV footprint estimate is slop, not padding

**Tag:** verified 2026-08-02 · **Blocked on:** nothing; left alone deliberately

`seam::kv_bytes_estimate_fmt` adds `KV_SLOP_ROWS = 64` rows per layer before
sizing each side's buffer, and the comment it inherited described this as
mirroring a pad `SeamKv` allegedly applies. It does not: both allocation sites
(`generate_dense_backend`'s KV loop and `SeamKv::fork`) allocate exactly
`kv_rows(..) * n_kv * head_dim` elements. The 64 rows are a deliberate
conservative margin and nothing more — the doc now says so.

Left in because every placement estimate shares this helper and removing it
would loosen all of them at once. That argument has weakened since: the window a
session actually gets is now re-decided against the device's measured free
memory (B8), so these rows are no longer a cushion against an over-optimistic
plan — they are context the fit hands back for nothing. Removing them is worth a
measurement now, not just a revisit.

### B14 — verification gaps from the 2026-08-02 decode-attention and KV-fit slices

**Tag:** raised 2026-08-02 · **Blocked on:** nothing; each is a measurement
someone has to run

Recorded as gaps rather than left implicit. Everything below shipped without the
check named, and in each case the check is cheap — the reason it is missing is
time or hardware, not difficulty.

- **Metal and CPU are unexercised for the KV-fit change.** Only the Vulkan path
  was measured. The Apple `#[cfg]`-gated code does not compile locally at all,
  so CI is the only thing that judges it — and CI does: the
  `cargo test (macOS / Metal)` and `cargo check (infr-metal, Apple target)` jobs
  are green. That settles compilation, not behaviour; nobody has run the fit
  math on an Apple device or on the CPU backend. The 2026-08-04 measured
  re-clamp does not change that either way: both backends return `None` from
  `Backend::device_alloc_room`, so they keep the estimate-only path unchanged
  (B8).
- **`infr serve --parallel N` is exercised at N=2 only, and not at a size where
  the window moves.** Two 40960-token slots on Qwen3-1.7B served a request
  (2026-08-04), which covers the fork path but not `vulkan_slot_ctx`'s
  divide-by-N against a window the post-load re-clamp shrank — the engine now
  reads its advertised window back from slot 0 after the warmup, and no run has
  put those two together.
- **The refuse rung's `Err` has never been printed by a real run.** No model on
  this box drives `max(fit_f16, fit_q8)` under `MIN_SESSION_CTX`, so the message
  text — the thing a stuck user actually reads — is untested against a human.
- **The iGPU chunk-ladder filtering is reasoned, not measured.** Filtering
  `ubatch_candidates` to heights below the current one also stops a placement
  sweep raising an integrated GPU's chunk above its watchdog-safe default. That
  argument was never run on an iGPU, and the watchdog is exactly the thing that
  punishes being wrong (see `docs/igpu.md`). `repin_ubatch_lower` (B8's measured
  re-clamp) refuses to RAISE a height for the same reason, and is equally
  unmeasured there.
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
the process, not the depth. Also observed 2026-08-04 on **gemma-3-12b Q4_K_M at
`-p 131056`** (cap 133) — a whole-window prefill is long enough to latch it, so
any deep-prefill measurement on that model is split too.

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

### B21a — the DG abort poll is wired but has never fired in a real run

**Tag:** CR-2026-08-03 M3 residual · **Blocked on:** nothing; it is a test that
needs a live DG serve request

`DiffusionGemmaChat` now threads `RequestCtx` into `diffusion_generate`, which
polls the abort latch at each BLOCK boundary, and the per-request seed is
resolved by `resolve_seed` (unit-tested). What is not tested is the poll
actually stopping anything: `diffusion_generate`'s loop needs a loaded
DiffusionGemma model, so no unit test can drive it, and the wiring was verified
by reading rather than by cancelling a real request.

Two things to check by hand against `infr serve` hosting a DG model: a client
disconnect mid-generation stops at the next block instead of running every
remaining block, and `serve.request_timeout_secs` does the same. Both latch the
same flag, so one test covers the mechanism.

Also still true, and by construction: a block is the finest granularity
available. `denoise_block` runs a whole canvas to completion, so a cancelled DG
turn stops at a block boundary, not immediately.

### B30 — the GGUF weight mmap trusts the file, and cannot enforce it

**Tag:** PR#90 · **Blocked on:** nothing outstanding; detection SHIPPED, the
preventing half is deliberately not attempted

`Gguf::open` maps the model file and hands out `&[u8]` slices into it for the
mapping's whole life. Nothing stops another process writing or truncating that
file: a write mutates memory Rust believes is frozen, and a truncation turns a
resident page into `SIGBUS` on next touch. That invariant is stated on `open`
and remains **UNENFORCED** — `infr_gguf::watch::WeightWatch` notices the breach,
it does not prevent it. Statting the HELD DESCRIPTOR rather than the path is the
load-bearing choice and `a_rename_into_place_is_not_a_change` pins it:
`infr pull` renames into place, which leaves a live mapping on the old inode
perfectly intact, so a path-stat would cry wolf on the one file-replacing
operation infr itself performs.

**Known boundaries of the detection, none worth closing on current evidence:**

- A same-length in-place write whose mtime is then restored is invisible. The
  only alternative is hashing gigabytes per check.
- `serve` checks per REQUEST, so a change landing mid-request is caught by the
  next one, after that response has already streamed. Post-checking would not
  un-send it.
- `WeightWatch::open` is a second `open` beside `Gguf::open`, so a rename
  landing exactly between them leaves the watch on the new inode and the mapping
  on the old — a missed detection, never a false one, two syscalls wide.

**Considered and rejected: copying the file into an anonymous mapping.** PR #90
did exactly that and it works, but it is not affordable. Measured on a 16 GiB
Qwen3.6-27B, warm page cache, two reps each: warm load 1.87 s → 10.5 s (5.6x,
re-opening almost exactly the gap `model-load-time` closed), and 14 GiB of
evictable page cache became 20.2 GiB of anonymous RSS. Anonymous pages are
swap-only, so a model larger than RAM goes from slow to unrunnable — the Llama-4
Scout blob on this host is 47.5 GiB against 60 GiB of RAM. Reverted in
`5ba6b3f`; the same PR's tensor byte-count overflow check was kept.

**Considered and rejected: an advisory `flock`.** `infr` cannot corrupt its own
mapping in the first place — `pull.rs` downloads to a temp and renames, nothing
anywhere opens a blob for writing — so a lock has no in-house writer to conflict
with. And it does not bind the writer that actually matters:
`cp new.gguf live.gguf` opens the destination `O_TRUNC` and takes no lock, nor
does any editor, nor llama.cpp. Reusable machinery exists (`FileLock` in
`pull.rs`) if this is ever revisited, and it should be taken SHARED — exclusive
would stop two `infr` processes sharing one model. Windows is not the obstacle
it first appeared: `FileLock` already calls `libc::flock` unconditionally, so
`infr-hub` does not build there today and CI covers only ubuntu-26.04 and
macos-15.

### B31a — what the weekly miri job does NOT cover

**Tag:** PR#90 review residual · **Blocked on:** nothing; recorded so the
coverage claim stays accurate

Miri runs weekly against `SpinPool` and `infr_core::hostpager` via
`.github/workflows/cron.yml` (which carries the reasoning for its flags — two
upstream workarounds are load-bearing and must not be "simplified" away). That
covers `collect`'s raw base pointer, its `set_len` over uninitialized slots,
`CollectGuard`'s `drop_in_place` during unwinding, the `Vec::from_raw_parts`
rebuild, and the host pager arena's per-slot raw-pointer slices across threads.
What it does not, and will not:

- **`kernels.rs`** — 168 of `infr-cpu`'s 191 `unsafe` uses are x86 SIMD, which
  miri cannot execute. That unsafe stays unchecked by anything but review.
- **Every FFI crate**, by construction: `infr-vulkan` dlopens `libvulkan`,
  `infr-metal` talks to a real GPU, `infr-gguf` maps a file, `infr-hub` takes an
  `flock`.
- **`infr-core` and `infr-chat` in FULL** — both probed crate-wide, neither
  finished inside the window it was given (10 and 50 minutes; `infr-chat` had
  completed 16 of 58 tests when stopped). Those bounds are "did not finish by",
  not measured durations. Little is lost: `infr-chat` contains no `unsafe`, and
  `infr-core`'s uses outside `hostpager` are three, one being a `libc::kill`
  miri cannot execute. The `hostpager::` filter added later runs in seconds, so
  the crate-wide cost was never the obstacle for the part that matters.

### B33 — the `wildcards` gate in `deny.toml` is off, for a fixable reason

**Tag:** cron slice 2026-08-04 · **Blocked on:** a decision about whether the
workspace crates are ever published

`cargo-deny`'s `wildcards = "deny"` catches a `*` version requirement on a
registry crate — non-reproducible builds, and a semver-major landing with no
diff. It is set to `allow` because this workspace's own members are declared
`{ path = ... }` with no version, which cargo-deny also reads as a wildcard.
`allow-wildcard-paths = true` is the intended escape and does not help: it
applies only to crates marked `publish = false`, and only `infr-testkit` is.

Marking the remaining members `publish = false` turns the gate back on and is
accurate today — everything is at `0.0.0` and nothing is on crates.io. It is
also a statement that these crates will not be published, which is a call to
make on purpose rather than in passing, and `infr-cli` is the one where the
answer might reasonably be "eventually". No registry wildcard exists in the tree
right now (`grep '= "\*"' crates/*/Cargo.toml`), so what is currently missing is
a guard against a future mistake, not cover for a present one.

### B34 — no fuzz targets, and there is an obvious first one

**Tag:** cron slice 2026-08-04 · **Blocked on:** nothing; scoped out of the cron
slice that surfaced it

The sibling repos' `cron.yml` runs `cargo-fuzz`; this one does not, because the
tree has no fuzz targets. hjkl's job guards its absence with
`if [ -d .../fuzz ]`, which is a job that reports "skipping" forever — a check
that cannot fail — so it was left out rather than copied.

`infr_chat::tools::parse_tool_calls` is the target that has already earned it.
It parses model output, which on `infr serve` is steerable by whoever sent the
request, and it has produced one unbounded-allocation hang in that position
(`{a:[}]}`, fixed in `0aa0661`). `delimiter_soup_always_terminates` now covers
every 6-byte body of container punctuation exhaustively, which is a floor rather
than a ceiling — libFuzzer over the same entry point would reach the string,
escape and `\u` paths that the punctuation alphabet cannot.

Adding it means a `fuzz/` crate, a nightly job, and a decision about how long
per target per week.

### B35 — tiered weight paging: phase 4 unbuilt, phase 5 one lever in

**Tag:** design slice 2026-08-04 · **Blocked on:** phase 4 needs Apple hardware
this host does not have

`docs/disk-streaming-plan.md` carries the design and the per-phase verification.
**Phases 0-3 have LANDED** (baseline measured, core `blockio`/`hostpager`/pins,
CPU backend on the DRAM tier, and the Vulkan third tier under BOTH dense
streaming and the paged MoE cache — numbers in `docs/perf/results.md`), and the
tier now **beats mmap on both backends**: CPU 2.06x at a 1.5 GB cap, Vulkan
2.17x on decode at an 8 GB cap with a 7 GB arena (1.41x with a 3 GB one). What
is left:

- **Phase 4, Metal / UMA collapse.** Unbuilt and **unverifiable here**: no Apple
  hardware, and `infr-metal` does not compile on this box. Writing it blind
  would produce code whose only evidence is that it type-checks in CI. Its own
  precondition is the `qui_cache` gate below. The options and their trade-off
  are written out in the plan's §7 as an open question for the user — do not
  re-derive them.
- **Double-caching: CLOSED as a non-problem, and the premise was wrong in both
  directions.** This entry used to say a buffered `pread` halves the tier's
  effective budget, and that `posix_fadvise(DONTNEED)` **cannot** reclaim the
  duplicate because it drops only clean UNMAPPED pages while `Gguf::open` maps
  the whole file. A `mincore` probe refutes both halves. `DONTNEED` DOES reclaim
  mapped-but-untouched pages (65 536 → 0 in the probe); a page is exempt only
  once it is actually faulted into a page table, and this tier never touches
  paged ranges through the mapping — it reads them with `pread`. Only the
  touched case stayed pinned, which is the control. And the reclaim is not
  needed anyway: an anonymous arena already wins page-cache reclaim under a
  cgroup cap, demonstrated by a 7 GB arena under an 8 GB cap running with major
  faults flat at ~1 700 and reading 110 GB against mmap's 232. No
  `O_DIRECT`/`F_NOCACHE` rewrite and no alignment work (plan §3.5) is required.
  **Do not reopen without new evidence** — what looked like a double-caching
  cost was the budget being too small, which is the auto-sizing item in B36.
- **Prefetch is deprioritized, and that reversal is the useful part.** It is
  still unbuilt on every backend (`HostPager::pin` reads synchronously, and on
  Vulkan under the dense/MoE session mutex). It was recorded here as "the
  leading suspect" for the GPU tier being slow. It was not: the run is I/O-bound
  by orders of magnitude — roughly 12.5 GB read per token against tens of
  milliseconds of GPU compute — so hiding a read behind compute has nearly
  nothing to hide it behind. The read was too SLOW, not too LATE, and the
  concurrent reader is what fixed it. Prefetch only becomes interesting once the
  arena is big enough that the tier stops being I/O-bound. Do not build it
  before then.
- **The reader's speedup is Linux/NVMe only.** `FileBlockIo` splits a block
  across `IO_FANOUT` concurrent positioned reads, measured 1.2-1.5 → 2.2 GB/s on
  this box's Samsung 980. Correctness is platform-independent (each read carries
  its own offset), but the SPEEDUP is not: on Windows `seek_read` issues
  `ReadFile` with an `OVERLAPPED` offset and a handle not opened
  `FILE_FLAG_OVERLAPPED` has concurrent operations serialized by the kernel, so
  the fanout may buy nothing until the file is opened for overlapped I/O.
  Untested on Windows and macOS. A rotational disk is also untested and is the
  one case where the concurrency could plausibly HURT (seek interleaving);
  nothing in the code adapts to device type.
- **The rest of phase 5**, still gated on measurements not taken: io_uring only
  if the reader proves queue-depth bound beyond what `IO_FANOUT` concurrent
  `pread`s reach (on this drive they already hit the device ceiling, so there
  may be nothing left), frequency-warmed DRAM for MoE-on-GPU, exclusive
  VRAM/DRAM placement for MoE, and multi-GPU/MTP coverage (TP/EP/pipeline
  binders and MTP's second weight set bypass the tier entirely).

Constraints the remaining phases must handle, recorded so they are not
rediscovered:

- `infr-metal`'s `qui_cache` factored arm copies the transformed weight out and
  retains it unboundedly, keyed by `MTLBuffer::id()`. Correct, but it is a
  second full copy of every touched weight in host RAM, so a paged Metal model
  must gate or budget it (plan §3.4).
- **Host paging is single-process-shaped.** `HostPager`'s exhaustion error names
  `paging.dram`, but nothing sizes the arena against `infr serve --parallel N`:
  the floor is N concurrent working sets and only 1 is priced. A tight budget
  under `--parallel` will surface as that error rather than a deadlock, which is
  the safe failure, but the sizing is still not done. The Vulkan tier is less
  exposed than the CPU one — `DensePagerSession::stage` drops its pin before
  returning, so its floor is one slot per pool regardless of N — while the CPU
  interpreter holds a whole op's pins across the op.
- **The per-pass file-change check is not wired.**
  `FileBlockIo::verify_unchanged` exists and is tested; no caller runs it yet,
  so a model rewritten mid-generation is read as whatever the new bytes are (the
  same exposure as B30, now reachable through explicit reads rather than the
  mapping).

### B36 — paging optimizations found by review, measured but not built

**Tag:** paging review 2026-08-04 · **Blocked on:** nothing; each is scoped out
of the slice that found it

A read of the whole paging path (DISK `blockio`, DRAM `hostpager`, VRAM
`infr-vulkan::pager`, the CPU `paged` pools and the seam placement) against the
counters `INFR_PAGER_STATS` reports. Three items LANDED from it — the concurrent
reader, the admission doorkeeper, and auto-sizing — and only the last leaves a
residue worth tracking; what follows is that plus what was deliberately left.

- **Auto-sizing LANDED; what is left is the platforms it cannot measure.**
  `infr_core::hostmem` sizes the arena from `MemAvailable` floored by the
  tightest cgroup limit, and the tier turns itself on whenever a model does not
  fit. Measured: it picks 7.44 GB under an 8 GB cap and reaches 0.38 t/s against
  the swept best of 0.39. **Linux only.** macOS needs `host_statistics64`'s
  free/inactive/purgeable split and Windows `GlobalMemoryStatusEx`; neither is
  reachable through this workspace's existing dependencies and neither could be
  verified on the machine this was written on, so both answer "unknown" and keep
  the mmap path unless `INFR_DRAM_CACHE` is set by hand. Adding either means
  adding a dependency, which is the user's call.
- **The unified-memory path is unverified on unified hardware.** An iGPU/APU now
  streams `DISK → GPU-accessible RAM` with no host cache
  (`HostPager::stream_only`, selected by `DeviceCaps::unified_memory`). The
  MECHANISM is covered on a discrete GPU by `INFR_DRAM_BYPASS` — the dense leg
  in `dense_tier_parity` content-checks it and the MoE leg in
  `gpu_seam_paged_moe_host_tier_matches_resident` is token-identical, both shown
  to fail when the tier serves a neighbouring block. What is NOT covered is the
  SELECTION and the sizing: that `unified_memory` is actually set on real iGPU
  and APU parts, and that `paging.cache` (the arena above, which on those parts
  comes out of shared RAM) ends up large enough to be worth having — nothing
  currently sizes it against HOST memory the way `paging.dram` now is. Needs an
  APU to answer. Metal is a separate question: it has no pager at all until
  phase 4, so it inherits none of this yet.
- **Layer-major prefill LANDED; what is left is where it does not reach.** The
  chunk loop now runs inside the layer loop for a streamed model
  (`seam::layer_major_prefill`, the `spans`/`chunks` walk in
  `generate_dense_backend`), so a prompt sweeps the weight set once instead of
  once per chunk. Re-measured on the B36 shape — Qwen3-14B Q8_0 / RX 7900 XTX,
  `MemoryMax=8G`, `paging.cache=2g`, `paging.dram=6g`, P=4096, three rounds with
  the arm order permuted and a cold page cache before every run — at the
  1024-row default chunk: 25.27 → 6.31 GB read and 341.9 → 779.9 pp t/s, the
  read volume now exactly a single-chunk prefill's. The residue:
  - **The remaining gap to the single-chunk arm is HOST cost, not I/O.** Same
    6.31 GB, but 779.9 t/s against 1049.6: layer-major builds and compiles a
    graph per (layer, chunk) — 40 x 4 here — where the one-chunk arm builds one.
    `build` re-declares every weight handle for the whole model on each call and
    `alloc_scratch` re-allocates the whole Internal set per execute, so both
    scale with the dispatch count. A per-(layer, batch-shape) plan cache, or a
    span band wider than one layer, is the obvious next lever; neither was
    tried.
  - **The activation reserve is priced at the FULL context, not the prompt.**
    `layer_major_act_bytes` reserves `ctx * n_embd` f32 out of the streaming
    budget (`dense_stream_budget_at`) because those buffers are allocated
    mid-prefill and the arenas are sized before any prompt arrives. A session
    that never fills its window holds that back for nothing — at ctx 32k /
    n_embd 5120 it is 671 MB of arena. Sizing it against a per-request prompt
    length, or reserving in bands, needs the budget to be re-decidable after
    load, which it is not today.
  - **Only the dense Vulkan path.** MoE expert paging, MTP, the qwen35/DeltaNet
    bespoke path, Metal and the CPU backend all keep chunk-major: the gate is
    `Backend::dense_paged`, and the E2B arch is refused by
    `layer_major_prefill`'s `spannable` arm (its `per_layer_inp` is
    prologue-built and a later span cannot see it). That gate is load-bearing,
    not belt-and-braces: E2B DOES take the batched-prefill path, so while the
    only refusal was the `assert!` in `build`, a streamed E2B panicked on an
    ordinary `infr bench <e2b> --set paging.cache=200m`. Covered by
    `gpu_seam_streamed_e2b_stays_chunk_major`. A paged MoE prefill has the same
    `ceil(P/ubatch)` structure and was never swept; whether the expert cache's
    locality makes it the same win is unknown.
  - **Not measured: more than one prompt length, and any model but this one.**
    The sweep is P=4096 on one dense model, one drive, one GPU. The claim that
    the ratio grows with prompt length is arithmetic (`ceil(P/ubatch)` sweeps
    become one), not an observation.
  - **`Capabilities::graph_input_inplace` is answered, not tested, for Metal.**
    It is set true there on the strength of `infr_core::exec::writes_back` being
    shared with the CPU interpreter — read, not run. Nothing on Metal takes the
    layer-major path today (its backend hosts no dense pager), so the flag is
    inert there until something forces it on.

- **`Pager`'s LRU is O(n_slots) per touch.** `mark_mru`, `evict` and
  `take_slot`/`take_slot_opt` all do `lru.iter().position(...)` followed by
  `VecDeque::remove`. The module doc scopes this to "tens to low hundreds" of
  slots and names the intrusive doubly-linked list as the upgrade path, and at
  today's model sizes it is genuinely not worth doing. It stops being true at
  DeepSeek-V4-Flash scale: 256 experts x 43 layers is ~11k blocks per role, and
  an MoE decode step touches ~6 experts x 43 layers x 3 roles per token. Fix it
  before that model, not because of a measurement today.
- **The dense session mutex spans the disk read.** `stage_dense_linear`
  (`infr-vulkan/src/adapter.rs`) holds `be_.dense_pager().lock()` across
  `DensePagerSession::stage`, which reaches `HostPager::fill` and blocks on I/O.
  Irrelevant to `bench` (one sequence), but under `infr serve --parallel N`
  every sequence serializes on every other sequence's disk reads. Related to the
  `--parallel` sizing gap recorded in B35 but a distinct problem: that one is
  about arena capacity, this one is about lock hold time.
- **Checked and CLEARED, so it is not re-investigated:** `plan_slots`'
  proportional split cannot affect bytes read. Total cached bytes equal the
  arena size no matter how slots divide across size classes, because a dense
  pass touches every block exactly once — the split only decides WHICH blocks
  are cached, not how many bytes. Any fully-spending split is equivalent on I/O
  volume.

### B37 — the cheap macOS guard does not guard the crate that keeps breaking

**Tag:** ci coverage · **Blocked on:** a decision about how much cross-compile
setup is worth paying for

`metal-check` in `.github/workflows/ci.yml` runs
`cargo check -p infr-metal --target aarch64-apple-darwin` on a Linux runner, and
its comment sells it as catching "Op-signature drift before the expensive
`test-macos` runner". It does not: every macOS break so far has been in
`infr-llama`'s `#[cfg(target_os = "macos")]` arms, not in `infr-metal`, and that
crate is outside the job's `-p`. `WBytes` replaced the binder's `&[u8]` in
`e657a66d` and left three Metal upload sites uncompilable; `test-macos` was red
for every commit from there to `588653b`, where it was fixed, because nothing
cheaper ever looks at that code.

Widening the `-p` is not free. `infr-llama` pulls `tokenizers`, whose `onig_sys`
and `esaxx-rs` build scripts compile C for the target — verified locally, where
both fail with `unrecognized command-line option '-arch'` — so the job would
need a macOS SDK and stop being the cheap Linux guard it was designed as. The
options are: pay for an SDK (osxcross or similar) on that job; find a
feature/dependency arrangement where the typecheck does not need the C deps; or
accept the gap and rely on `test-macos`, which is honest but leaves every macOS
break a full round trip away.

Worth knowing either way: `rustup target list --installed` claimed
`aarch64-apple-darwin` was present on this machine while
`$(rustc --print sysroot)/lib/rustlib/` did not contain it, so a local
cross-check has to be run against a target confirmed in the sysroot.
`x86_64-apple-darwin` gates identically and was actually installed.

### B38 — doc drift found while rewriting `docs/plan.md` (2026-08-05)

**Tag:** docs · **Blocked on:** nothing; scoped out of the plan.md rewrite,
which deliberately touched only that file

Two things the rewrite surfaced and did not fix, both verified against the tree
on 2026-08-05:

- **The root `README.md` supported-models table has no BitNet rows.**
  `infr_llama::arch::ALL` carries `bitnet` and `bitnet-b1.58` (landed in
  `5b44ef9` and `dbc8431` — llama skeleton + SubLN, TQ2_0 / i2_s ternary
  weights), so the engine accepts two families the README does not advertise. A
  reader picking models off that table concludes they are unsupported. Fix is
  two table rows plus a line in the `Scope` list; the arch consts' own doc
  comments already say what to write.
- **`docs/config-plan.md` was deleted (`3010e45`, campaign complete) and 74
  references to it survive** across `docs/config.md` and code comments in
  `infr-cpu`, `infr-cli`, `infr-vulkan` and `crates/infr-cpu/tests`. Most cite a
  section number (`§10.6`, `R6`, `R4/R6`) as the rationale for a design
  decision, so they are not simply deletable: the reasoning they point at is now
  only in `git log`. Either restore the sections that are still load-bearing
  into `docs/config.md` and repoint, or replace each citation with the reason it
  was standing in for. Count is from
  `grep -rn "config-plan.md" --include=*.md --include=*.rs .`

### B39 — Vulkan MoE id-GEMV silently no-ops for `in_f < 32` (2026-08-07)

**Tag:** vulkan · **Blocked on:** nothing; found while writing the deepseek2 MoE
op-parity tests, which needed ne=32 to make the Vulkan cross-check meaningful

`native_gemv_id_multi.comp` computes `nsub = pc.in_f/32` (integer division), so
an expert bank with `in_f < 32` runs **zero** sub-blocks and the dispatched op
leaves `dst` all-zero — a silent wrong output, no error. The existing adapter
test `moe_ffn_graph_matches_host` deliberately uses ne=32 (the kernels' floor),
and every production model has ne/n_ff ≥ 32, so no real GGUF hits it — the
hazard is synthetic tests and any future small-`ne` arch. Fix options: assert
`in_f >= 32` at adapter dispatch (fail loudly instead of zeros), or add a
sub-block floor (`nsub = max(in_f/32, 1)` with clamped reads) to the kernel. The
seam tests `moe_sqrt_softplus_parity` / `moe_groups_bias_parity` document the
constraint in their comments.

**The DENSE native path has the same hazard (2026-08-10).** Writing
`grouped_output_projection_composes_from_linear_and_copystrided` (deepseek4
slice 2), a plain `Op::Linear` with a bf16 weight at `in_f = 8` returned all
zeros on Vulkan while the CPU interpreter returned the right answer — same
graph, same bindings, no error anywhere. Raising `in_f` to 32 made both agree
exactly. So whatever `native_dense_dtypes` weights dispatch through inherits the
same `in_f/32` floor as the MoE id-GEMV, and a fix should cover both. NOT
root-caused to a specific kernel (the dense path picks among several by
`m`/dtype/tier); the observation is the m=3 bf16 case. The seam test documents
the constraint in its comment and uses block-aligned dims.

### B48 — a failing op leaks its in-flight Vulkan recorder on most error paths (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** nothing; surfaced by the B45
guard, which was the first `lower_op` error path that could fire in a real run

`Recorder` has no `Drop` by design — a segment is finished or explicitly
discarded — so an early `?` out of `execute_static`'s op loop drops it with its
descriptor pools still allocated, and the validation layer reports them as
leaked objects at `vkDestroyDevice`. The two `lower_op` call sites now route
through `abort_segment`, which discards the partial recorder and folds any
teardown error into the message. The other `?` exits inside the same loop —
`resolve`, `execute_paged_moe`, `stage_dense_linear`, `finish_nowait` — still
drop it.

None of them fires in a healthy run, which is why this was invisible until an op
gained a reachable refusal. Fix is mechanical (same `abort_segment` wrapper);
the reason it was not done with B45 is that each site returns from a different
place in the loop and the change wants its own review. Verified live: before the
`abort_segment` fix the guard's own test printed
`VUID-vkDestroyDevice-device-05137 … has 2 leaked objects` under
`VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation`; after it, that run is
clean.

### B49 — the full-softmax MoE weight has no regression test, and cannot have one here (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** hardware — a Vulkan
implementation that preserves f32 subnormals

The defect this entry opened with is FIXED: `moe_topk.comp`'s
softmax-without-renormalization branch summed `exp(logit - mx)` over every
expert while `mx` was the max over the SELECTED ones, so a router bias or group
mask could leave the largest raw logit unselected, overflow the denominator to
`+inf` and zero every weight. It now computes its own max over all experts,
matching the CPU oracle's constant.

What stays open is that **nothing guards it**, and nothing can on this box. A
selected expert's weight is `1 / D` where `D` is the denominator the wrong shift
computes; the bug needs `D` past f32's max (3.4e38) to overflow, so the correct
answer the fixed kernel must produce is always below 2.9e-39 — inside the
subnormal range. Measured on an RX 7900 XTX (RADV, no denorm-preserve execution
mode): a weight of 1.8e-35 comes back exactly, a weight of 5.5e-42 comes back as
`0.0`. The fixed and the broken kernel return the same bytes there, so a test
would be a green light wired to nothing; one was written, measured, and deleted
rather than landed. lavapipe was tried as a denorm-preserving second
implementation and the backend refuses it (it needs a pinnable subgroup size of
32; lavapipe's range is [8, 8]).

To close this, either run it under an implementation that preserves subnormals,
or enable `VK_KHR_shader_float_controls`'s denorm-preserve on the pipeline and
re-run the deleted case. The finding and these numbers are also recorded in the
shader beside the fix.

Unrelated residual in the same branch: the extra serial `mx_all` scan runs on
thread 0 once per token per MoE layer, and V2-Lite does take this branch
(softmax gating, `norm_topk_prob = false`). The cost was not measured.

### B50 — Metal cannot run a DeepSeek MoE layer (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** nothing; it is a missing
kernel feature, not a bug, and no one has asked for DeepSeek on Metal

`infr-metal`'s `Op::MoeFfn` arm implements softmax gating + top-k renorm +
output-weighting and asserts on anything else. V2-Lite ships
`norm_topk_prob = false` and V3 is sigmoid-gated, so both already fail that
assert — DeepSeek MoE layers are CPU + Vulkan only. MLA attention itself IS
implemented on Metal and is unaffected.

The arm also read neither `exp_probs_b` nor the group-routing fields: it
destructured them away with `..`, so softmax + renorm + `expert_group_count > 1`
— a legal `deepseek2` config — passed the assert and then routed with neither
the bias nor the group mask applied, picking the wrong experts with no error.
That combination now asserts too, so the gap is loud rather than silent, but the
underlying feature is still missing.

Closing it means teaching the Metal router the same two extensions
`moe_topk.comp` grew: select on `probs + exp_probs_b` while weighting from the
unbiased probs, and mask all but the top `n_expert_groups_used` groups scored by
their top-2 sum. `moe_topk.comp` is the working reference. Note there is no
Apple hardware on the dev box, so this can only be verified on the macOS CI job.

### B52 — the weight loader validates tensor NAMES, not shapes (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** nothing; family-wide, surfaced
by the deepseek4 load slice

`wload` asks the GGUF for a tensor by name and fails if it is absent, but never
checks its dimensions against what the graph will index it as. Every "the loader
consumes every tensor" test in `tests/synthetic_deepseek2.rs` therefore proves
only that each name was requested — never that it was the right shape. A GGUF
whose `attn_q_b` is the wrong width loads clean and produces garbage.

Two V4-specific instances of the same gap, both found reading
`src/models/deepseek4.cpp`:

- **`output_group_count` divisibility is unchecked.** The reference sizes `wo_a`
  as `n_head * n_embd_head / o_groups` with plain integer division
  (`deepseek4.cpp:97`), so a non-dividing group count silently truncates.
  llama.cpp catches it downstream as a `create_tensor` shape mismatch; `infr`
  would not notice at all.
- **The reference shapes every V4 tensor with `n_embd_head_k()` at the default
  `il = 0`**, and `load_arch_hparams` has already called `set_swa_pattern(0)`,
  so that call returns the SWA head width rather than the full one. They are
  equal only because `llama-model.cpp` defaults `_swa` to `_full` and no V4 GGUF
  declares `attention.key_length_swa`. `infr` reads `attention.key_length`
  directly and so does not inherit this, but a file that declared the SWA key
  would make the two implementations disagree.

Fix is one shared "expected dims" check in `wload` rather than per-arch asserts;
the tests that exist would then gain teeth for free.

### B53 — V4's KV geometry duplicates the V side (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** a one-line change in
`crates/infr-cpu`, which the wiring slice did not own

`seam::kv_row_elems` now has a `deepseek4` branch and it returns
`(head_dim, head_dim)` — one MQA row per side per token. **The V side is a
DUPLICATE of the K side, written by a second `Op::WriteKv` from the same source
row**, and that is not what the arithmetic wants.

V4's raw attention is `build_attn_mha(q, k_all, k_all, …)`: K and V really are
the same rows, so `(head_dim, 0)` — MLA's aliasing, with `kv_side_elems`
supplying the placeholder and `Op::Attention` pointed at one buffer for both
sides — is the correct shape. It is not the shape this codebase can execute. The
CPU backend's `Op::Attention` arm takes `cpu_buf(kbuf).read()` and
`cpu_buf(vbuf).read()` as two simultaneously-live guards, and a KV buffer is
`CpuStore::Owned(Mutex<Vec<u8>>)` — a non-reentrant `std::sync::Mutex`. One id
bound to both sides therefore self-deadlocks on the first V4 attention op. (The
CPU MLA arm and `Op::LightningIndexer` both already take exactly ONE guard,
which is the shape to copy; Vulkan is fine with the aliasing — both bindings are
`readonly` and `Recorder::sync`'s dirty-set tracking de-dupes.)

**To close it:** make the CPU arm take one guard when `k_cache == v_cache` (or
take one guard and dequant twice from it), then flip the branch to
`(head_dim, 0)` and drop the second `Op::WriteKv` in the `MixerW::Dsv4` emit.
Add a CPU parity test that binds one id to both sides — there is none today.
Worth roughly half of a V4 session's KV bytes. Also worth a `k_cache == v_cache`
short-circuit in the Vulkan adapter's dequant prepass, which would otherwise
dequant the same cache twice into two pooled scratch buffers (`kvdeq_k` /
`kvdeq_v`); harmless for V4's f16 cache, live if a quantized V4 KV ever lands.

**The compressed caches and compressor states are still unmodelled**, and cannot
be modelled by this helper: they are per-layer (a ratio-0 layer has none) and
the three compressor states are fixed-size recurrent buffers rather than
per-token rows, so the precedent is `MixerW::DeltaNet`'s conv/S-state
allocation. Sizes are tabulated in `docs/deepseek.md` § Stage 4. Nothing reads
them: `generate_dense_backend` refuses a non-zero ratio before a graph is built.

### B54 — `WeightWatch` watches one file of a shard set (2026-08-10)

**Tag:** multi-shard GGUF slice · **Blocked on:** nothing; scoped out of the
loading slice that surfaced it

`Gguf::open` now loads a whole `gguf-split` set, but `WeightWatch::open` is
still called from `infr-cli` with the single path the user typed, so on a split
model it stamps shard 1 and notices nothing about shards 2..N being replaced
mid-run. The detection it provides is per-inode (see B30), so extending it means
holding one stamp per shard — the shape `FileBlockIo::open_shards` already has,
where every shard's descriptor is stamped and `verify_unchanged` walks them.

The pieces to connect: `Gguf::shards` reports `(path, length)` per shard, which
is what a set-aware `WeightWatch::open` would take instead of a path, and every
`WeightWatch::open` call site in `infr-cli/src/main.rs` passes a path it got
from model resolution rather than from the loaded `Gguf`. Note the streaming
tier is already covered on a split model — `FileBlockIo::open_shards` stamps
every shard and refuses one whose length no longer matches what the weights were
loaded against — so this gap is only about the non-streaming mmap path.

### B57 — `infr pull` ignores the shutdown latch (2026-08-10)

**Tag:** concurrent-pull slice · **Blocked on:** nothing; PRE-EXISTING, left out
of the concurrency slice on scope grounds

The CLI installs `SIGINT`/`SIGTERM` handlers that latch
`infr_core::shutdown::request_shutdown`, and every GPU submit path polls
`shutdown_requested()`. Nothing on the download path does: neither
`download::stream_into`'s read loop nor `pull::fetch_all`'s claim loop. Verified
by observation, not by reading alone — a `SIGTERM` sent to a running
`infr pull unsloth/DeepSeek-V3.2-GGUF:Q2_K` left the process downloading, and it
took `SIGKILL` to stop it.

Nothing is corrupted by that: the partials are append-only, the next run resumes
from `metadata(tmp).len()`, and a real 229 GB pull was in fact killed and
resumed exactly this way. The defect is that the first Ctrl-C appears to do
nothing.

**The fix is three polls now**, all of which keep today's "partial kept for
resume" contract: `fetch_all`'s `while !stop` becomes
`while !stop && !shutdown_requested()` so a fan-out over 236 shards stops
claiming files; `ranged::worker`'s claim loop gets the same, so a 161 GB
single-file pull stops claiming CHUNKS (its sidecar already makes that a clean
stopping point — every completed cell is recorded); and `stream_into` checks per
64 KiB chunk and returns `Error::Aborted`. The last one is why this was not just
done: `Aborted` would have to travel out through `StreamError`, and the caller
must treat it as the KEEP-the-partial case rather than the discard case — a
distinction the current two-variant enum makes by accident of which error it is,
not deliberately. `RangedError` already makes exactly that distinction
deliberately (`Fatal` keeps the partial, `Changed`/`NoRanges` discard it), so
the ranged half is the cheap one.

### B58 — what the concurrent-pull slice did NOT verify (2026-08-10)

**Tag:** concurrent-pull slice coverage · **Blocked on:** nothing; each line is
a gap, stated so the coverage claim stays honest

`hub.pull_jobs` and `pull::fetch_all` are covered by six tests against a local
HTTP origin (`infr-hub/src/testhttp.rs`) — all files land, the bound is asserted
on the peak the SERVER observed, `jobs = 1` stays sequential, a planted partial
resumes under concurrency, a stale `If-Range` restarts rather than splices, a
sha256 mismatch is refused and unlinked, and a failure stops the fan-out — plus
one real 229 GB five-shard pull. Not covered:

- **A concurrency bound above 5 in anger.** SUPERSEDED in part by the ranged
  slice, which ran the default 8 against one object: aggregate 79.9 MB/s, 10.0
  MB/s per connection against 15.6 MB/s for a lone one. So the knee is somewhere
  between 5 and 8 and the AGGREGATE is flat across it (78.7 MB/s at five, 79.9
  at eight) — this host tops out near 80 MB/s whatever the count. What is still
  unmeasured is a bound above 8, and whether the ceiling is the link, the CDN or
  a per-client cap.
- **Two `infr` processes pulling the same model at once.** The per-blob `flock`
  that serialises them is unchanged and still unit-tested
  (`file_lock_is_exclusive`), but no test starts a second process, and the
  cross-process case is now N locks held at once instead of one.
- **A repo with more files than the bound.** `DeepSeek-V3.2-REAP`'s 236 shards
  is the case the bound exists for, and it has only been exercised at 10 files /
  3 workers against a local origin with the progress group HIDDEN. What that
  leaves unknown is the shape of the bar block over a long queue: `indicatif`
  reaps a finished bar's line only once every bar ahead of it has also finished
  (`BarState::drop` marks a zombie; `MultiState::draw` reaps consecutive zombies
  from the head), so files completing out of order could leave the redrawn block
  growing past `pull_jobs` lines. Five shards did not show it. If it turns out
  to matter, the lever is `MultiProgress::remove` / `finish_and_clear` on each
  completed bar, at the cost of the per-shard `✓` line.
- **Windows and macOS.** `download.rs` calls `libc::flock` unconditionally, so
  `infr-hub` does not build on Windows at all (B30 records the same); the
  fan-out itself is `std::thread::scope` and portable. macOS is CI-only.
- **`HF_TOKEN` on a gated repo.** The token now reaches `download_to_blob` from
  inside rather than as a parameter; unchanged in effect, but no gated repo was
  pulled.
- **The progress rendering was read, not eyeballed live.** The five-bar block
  was captured from a `script`-allocated pty and inspected as escape sequences
  (one line per shard, `MultiProgress::suspend` clearing the block around each
  log line), not watched on a real terminal.

### B59 — what the ranged-download slice did NOT verify (2026-08-11)

**Tag:** ranged-pull slice coverage · **Blocked on:** nothing; each line is a
gap, stated so the coverage claim stays honest

Intra-file ranged parallelism (`infr_hub::ranged`, `infr_hub::parts`) is covered
by twelve tests against the local origin (`infr-hub/src/testhttp.rs`) plus
eleven unit tests for the sidecar, the grid, the identity check and the
connection budget — each one shown red before green by breaking what it guards —
and by one real pull of `unsloth/DeepSeek-V3.2-GGUF:UD-TQ1_0`: 161 280 830 528
bytes over 2 404 ranges through two deliberate `SIGKILL`s, 33.6 minutes at 80.0
MB/s end to end, sha256 equal to HF's `lfs.oid`. Not covered:

- **A re-upload of a REAL object mid-download.** The splice guard is exercised
  only against the local origin. In production the guard that actually fires is
  the plan-time comparison of HF's LFS oid (`x-linked-etag`) with the one in the
  sidecar; the per-chunk `If-Range` has never been seen to reject anything,
  because the CDN answered all 2 404 chunk requests `206`.
- **A connection freed mid-chunk.** A range worker only tries to grow the
  fan-out between chunks (`ranged::worker`), so a download already inside its
  last chunk cannot pick up a permit another file just released. Bounded by one
  chunk time (64 MiB, seconds), and not worth a wake-up mechanism until
  something measures it.
- **An `ETag` that differs between CDN edges.** If one ever did, `If-Range`
  would make that chunk come back `200` and the file would restart as a single
  stream — correct, but slow, and nothing would say why beyond one `warn!`.
  Unobserved in a 2 404-chunk pull, so the risk is bounded but not zero.
- **Chunk-level retry.** A failed chunk aborts the whole file (its partial and
  sidecar are kept, so the next run resumes). Retrying that one range in place
  would be strictly better on a flaky link and is not implemented.
- **The 64 MiB chunk size was not swept.** It was picked from the two costs it
  trades (a request per chunk against work lost on interruption) and the real
  pull confirms the request overhead is not visible at 80 MB/s, but 32 or 128
  MiB were never run.
- **`hub.pull_jobs` above 8.** See B58: aggregate throughput is flat between
  five and eight connections on this host (78.7 → 79.9 MB/s), so the interesting
  question is now whether anything is left above 8 — untried.
- **Two processes pulling the same big file.** The per-blob `flock` that
  serialises them is unchanged and still unit-tested, and both modes take it
  from the same `.dl-…lock` name so a mode difference cannot dodge it. No test
  starts a second process.
- **Windows.** `ranged::fetch_chunk` writes with `std::os::unix::fs::FileExt`,
  so it is one more unix-only dependency in a crate that already does not build
  there (B30).
- **A gated repo over ranges.** The bearer token is attached to the probe and to
  every chunk request, but reqwest drops `Authorization` across the cross-origin
  redirect to the CDN (as it should — the CDN URL is pre-signed), and no gated
  repo was pulled.
- **The progress block was read, not eyeballed.** One bar per file is what the
  code does (chunk workers all `inc` the same bar); the real pull ran with
  stderr redirected to a file, where bars are hidden.

### B-DSV4-WIRING — what the V4 graph slice still owes (2026-08-10)

**Tag:** CR-2026-08-09 deepseek · **Blocked on:** nothing

**Slice A (ratio 0) is DONE and generates** — see `docs/deepseek.md` § Stage 4,
"Slice A". What is left:

**Slice B — ratios 4 and 128.** Needs new ops, not just wiring:

- A **per-channel softmax pooling** op (softmax over the block axis, then a
  weighted sum) — no current op does it.
- Persistent **compressor state** buffers with the ring-update and commit plan
  of `dsv4_build_comp_plan`, including the `[persistent | scratch | sentinel]`
  gather layout and the two-contiguous-halves read order for the overlapping
  compressors.
- **`Op::Attention` has no `key_bias` field.** `Op::TopkMask` exists and
  `Op::Mla` consumes it, but the CSA path needs a top-k mask on ordinary
  attention over a concatenated `[raw | compressed]` K, which has nowhere to go
  today.
- **`Op::LightningIndexer`'s contract is written for V3.2's per-token key
  cache.** V4's keys come out of the compressor, so `top_k` counts compressed
  blocks; re-read the op's `k_cache`/`kv_len` meaning before reusing it.
- The per-layer KV geometry those tiers need (§ B53) and the `wpush` handles for
  their tensors, which the ratio-0 arm deliberately does not declare — the
  `assert!` at the top of the build closure is what keeps that honest.

**What slice A left unverified, in its own right:**

- **`batch > 1` has never been EXECUTED for V4 on any backend.** The only V4
  fixture that exists (`crates/infr-llama/tests/synthetic_deepseek2.rs`'s
  `dsv4_model`) writes f32 expert banks, so `moe_batched_ok` is false and the
  chunked batched prefill is unreachable from it — and `generate_dense_backend`
  now excludes V4 from that path outright rather than leave an untested shape
  live. Every V4 graph the tests build is `batch == 1`. Re-enabling it means a
  quantized-expert fixture (or a real GGUF) plus the layer-span question below.
- **A partial layer span is refused** (the assert beside the compress-ratio
  one): the widened residual is `hc_mult` streams wide and a span hands over one
  `n_embd`-wide `hidden` buffer. Layer-major prefill therefore cannot serve V4
  as written; a span-carried widened stream would need `hidden` to be the wide
  buffer for V4 builds.
- **Metal has never been executed** (no Apple hardware). The V4 emit is
  backend-generic, so Metal will take it as soon as its own gaps close — but its
  DEVICE MoE path asserts `MoeGating::Softmax` and V4's gating is mandatory
  `SqrtSoftplus` (§ B-DSV4-HASH), so a V4 MoE layer aborts there first.
- **No real V4 GGUF has been opened.** Every dimension in the fixture came from
  the reference's formulas — see `docs/deepseek.md` § "Open questions" 1.

### B-DSV4 — what the V4 attention primitives do NOT cover yet (2026-08-10)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing; scoped out of the
op-level slice, which was explicitly "add the primitives, emit nothing"

Each of the three new capabilities lives in exactly ONE kernel per backend
rather than across the whole attention/norm/rope tier ladder. That was
deliberate — a tier that quietly ignored a sink or a sign would produce
plausible wrong numbers, and there is no caller yet to justify the fan-out — but
the V4 wiring slice inherits the list. Every refusal below is a loud error, not
a silent fallback.

- **`Op::Attention { sinks }` on Vulkan** runs only `attention_kv.comp`'s
  `-DSINKS` build: f16 K/V, bound descriptors, static recording, causal/SWA.
  Refused: a Q8_0 or any other non-f16 cache (the dequant→f16 prepass sits below
  the early return that routes to the sinks kernel), and `AttnMask::Canvas`.
  `decode_eligible` also returns false for any graph containing a sinks op, so a
  V4 decode loses the record-once replay tape. Nothing routes to flash,
  non-FA-coopmat, split-K or the mrows tier, so a sinks layer runs the scalar
  one-workgroup-per-(row, head) kernel at every depth — the thing `attn_partial`
  exists to avoid. A perf pass means teaching at least `attn_partial` +
  `attn_combine` the sink (it folds cleanly into the combine's final `(m, l)`,
  which is where the split-K partials already merge).
- **`Op::Attention { sinks }` on Metal** runs only `ATTN_SINKS_KERNEL`'s two
  instantiations (`attention_sinks_f32`, `attention_sinks_f16kv`). Refused: a
  decoupled or quantized K/V pair, `AttnMask::Canvas`, and the decode-replay
  tape. Same tier story as Vulkan.
- **`Op::QkNorm { weight: None }` on Vulkan** is `rmsnorm.comp`'s `-DNO_WEIGHT`
  f32 build only; an f16 `x` (llama4's post-rope L2-norm shape) errors. V4's Q
  norm reads the f32 `wq_b` output, so nothing needs the f16 twin yet, and a
  build nothing dispatches is a build nothing tests.
- **`Op::Rope { backward: true }` on Vulkan** is static f32 NORM only
  (`rope_back`, `rope_ff_back`). NEOX+backward, f16-out and the record-once
  `_dyn` path all error. V4's de-rope is NORM on an f32 scratch. Metal's rope is
  one runtime-parameterised kernel, so it carries `backward` on every path
  already.
- **A V4 graph cannot take the record-once decode replay tape at all** — four of
  its ops (`Attention{sinks}`, the three `HyperConnect*`, `Rope{backward}`,
  `QkNorm{weight:None}`) have no `_dyn` twin, so `decode_eligible` is false and
  the seam's mirror gate excludes `c.deepseek4` explicitly. Every V4 decode
  token therefore rebuilds + recompiles its graph. That is a real per-token host
  cost (the thing the tape exists to remove) and the first perf item for this
  arch.
- **`Op::Linear` with `w_off` on an F16 weight is still refused on Vulkan.** F32
  now rides a shifted `bufferDeviceAddress` base (the grouped output
  projection's caller); F16 would need the same at `matmul_proj` / `linear` /
  `linear_f16_noext`, plus a `w_off % 2 == 0` obligation on the two that read
  the weight as packed u32 words. Nothing produces an f16 `w_off` today.
- **gemma4's V-norm and llama4's L2-norm still pass a ones-vector weight** to
  `Op::QkNorm`. They could now pass `None` and drop a per-graph `head_dim`-float
  allocation each; the numbers are bit-identical (`x * s * 1.0` is `x * s` in
  IEEE, which `qknorm_weightless_matches_a_ones_weight` asserts at
  `max_err == 0`). Left alone because it is an edit to `crates/infr-llama/src`,
  which the op-level slice did not own.
- **Metal has never been executed.** No Apple hardware was available; the three
  new kernels (`qknorm_nw_f32`, the `backward` arm of `rope_f32`, and the two
  `ATTN_SINKS_KERNEL` instantiations) typecheck via
  `cargo check -p infr-metal --all-targets --target x86_64-apple-darwin`, and
  MSL is compiled on-device at runtime, so the macOS CI job is their first real
  compile AND their first execution. The three `#[ignore]`d tests
  (`qknorm_weightless_parity`, `rope_backward_parity`, `attention_sinks_parity`)
  are what will report it.

### B-DSV4-HC — what the Sinkhorn hyper-connection ops do NOT cover yet (2026-08-10)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing; scoped out of the
op-level slice, which was explicitly "add the ops, emit nothing"

`Op::HyperConnectMix` / `HyperConnectPre` / `HyperConnectPost` are implemented
and parity-tested on CPU + Vulkan (and typechecked on Metal). What is left:

- **How the widened stream is SEEDED is an assumption, not a transcription.**
  The ratio-0 emit replicates the token embedding across all `hc_mult` streams
  (`Op::CopyStrided` per stream, in the prologue). Neither `docs/deepseek.md` §
  Stage 4 nor this file records what `deepseek4.cpp` actually does there — the
  read slice covered the compressed caches, the attention block and the HC math,
  not the stream initialisation. Replication is what the hyper-connections
  formulation calls for and what makes the head's collapse a partition of unity
  at depth 0, but it has not been checked against the source. **Read
  `deepseek4.cpp`'s `build_arch_graph` prologue and confirm it before trusting a
  real V4 checkpoint's output.** A wrong seed produces plausible logits.
- **The weightless RMSNorm over the flattened `hc*n_embd` row went the
  ones-vector way**, not the `Op::RmsNorm { weight: Option<_> }` way: one
  `hc_mult*n_embd`-wide vector of 1.0 is uploaded per V4 session, matching the
  three ones-vectors gemma4 / dual-MoE / llama4 already upload. Extending
  `Op::RmsNorm` the way `Op::QkNorm` was extended would drop that allocation for
  all four callers and is the tidier end state; it was out of scope here because
  it is a change to every backend.
- **Performance was not considered at all.** Every kernel is the naive shape:
  `hyper_mix` runs ONE THREAD per token with the `hc × hc` matrix in a private
  array (dynamic indexing into a private array is the classic scratch-memory
  spill), and `hyper_pre` / `hyper_post` are one thread per output element with
  a serial `hc`-term loop and no vectorisation. At `hc = 4` the mix op is
  `rows × 24` floats of work, so it is unlikely to matter; `hyper_post` writes
  `rows × hc × n_embd` and is the one to measure first. Nothing here is
  measured.
- **`hc_mult` is capped at `HYPER_CONNECT_MAX_MULT` (8)** by a host check in
  each backend. Raising it means raising `HC_MAX` in `hyper_mix.comp` and
  `elementwise_norms.metal` together — the constant is duplicated in three
  places (Rust, GLSL, MSL) with no compile-time link between them, only the host
  refusal keeping the kernels in range.
- **`Op::HyperConnectMix` writes three outputs.** Vulkan's `dispatch` treats the
  trailing `n_out` bindings as writes and Metal takes an explicit write mask, so
  the hazard tracking is right — but this is the first op in the codebase with
  more than two `dst`s, and no fusion/scheduling pass has been looked at for it.
- **Metal has never been executed.** No Apple hardware was available; the four
  new kernels (`hyper_mix_f32`, `hyper_mix_gates_f32`, `hyper_pre_f32`,
  `hyper_post_f32`) typecheck via
  `cargo check -p infr-metal --all-targets --target x86_64-apple-darwin`, and
  MSL is compiled on-device at runtime, so the macOS CI job is their first real
  compile AND their first execution. The three `#[ignore]`d tests in
  `crates/infr-metal/tests/parity.rs` (`hyper_connect_mix_parity`,
  `hyper_connect_pre_parity`, `hyper_connect_post_parity`) are what will report
  it.
- **`n_iter` is not exercised above 40, and `eps` only at 1e-6 / 1e-2.** The
  real V4 values come from `{arch}.hyper_connection.sinkhorn_iterations` and
  `.epsilon`, neither of which has been read off a real GGUF (see § "Open
  questions" 1 in `docs/deepseek.md` — no V4 file has been dumped).
- **The four eps sites are not all pinned at production eps.** At `eps = 1e-6`
  the over-dst site moves the answer by ~3e-7, below the 1e-5 tolerance the
  backend comparisons run at; only the synthetic `eps = 1e-2` case pins it for
  an f32 kernel. Same for the asymmetric iteration COUNT (~1e-11 at 1e-6). Both
  are pinned in exact arithmetic by `hyper_connect_details_are_load_bearing`. If
  a real V4 GGUF turns out to use a small eps, a backend that dropped the
  over-dst eps would pass every test here.

### B-DSV4-HASH — what the hash-routed MoE / SwiGLU-clamp ops do NOT cover yet (2026-08-10)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing; scoped out of the
op-level slice, which was explicitly "extend the ops, emit nothing"

`Op::MoeFfn::expert_ids`, and `swiglu_clamp` on `Op::GatedAct` /
`Op::GatedActFused` / `Op::MoeFfn`, are implemented and parity-tested on CPU +
Vulkan (real RX 7900 XTX, validation layer clean) and typechecked on Metal. What
is left:

- **The CLAMPS now emit; hash routing does not.** A ratio-0 V4 layer builds a
  real `Op::MoeFfn` with `swiglu_clamp: swiglu_clamp(swiglu_clamp_exp[il])` and
  a shared expert at `swiglu_clamp_shexp[il]`, on CPU and Vulkan. A HASH-ROUTED
  layer is refused by name in `generate_dense_backend`, because of the gather
  below — emitting it with `expert_ids: None` would silently run the router's
  own top-k and pick different experts. That refusal is what the ratio-0
  generating fixture avoids by setting `hash_layer_count = 0`; the canonical
  fixture keeps `hash_layer_count = 2` and pins the refusal.
- **The GATHER that produces `expert_ids` does not exist.** The op takes the ids
  ALREADY gathered; nothing in the tree can produce them from
  `blk.N.ffn_gate_tid2eid.weight` + the token-id input. `Op::EmbedGather` was
  checked and cannot be reused as-is on Vulkan: `embed_gather.comp` iterates
  whole 32-element sub-blocks (`nsub = pc.ne / 32u`), and a tid2eid row is
  `n_expert_used` wide (8 in every V4 config seen), so the loop body never runs
  and the kernel writes NOTHING — a silent all-zero gather, i.e. every token
  routing to expert 0. It also decodes through `native_decode.glsl`, which has
  no I32 format, while llama.cpp's `ggml_get_rows` preserves I32 only for an I32
  source and `ggml_mul_mat_id` asserts its ids are I32 — so a real V4 file's
  table is I32. The wiring slice needs either a small-`ne` I32 path in
  `embed_gather.comp` or a dedicated gather op. `infr`'s own synthetic V4
  fixture (`crates/infr-llama/tests/synthetic_deepseek2.rs`) writes the table as
  a FLOAT weight, so it will not surface this.

  A third option the ratio-0 wiring slice considered and did NOT take: gather on
  the HOST. The token ids are known there, the table is
  `[n_expert_used, n_vocab]`, and the result would be one
  `[batch, n_expert_used]` i32 Input per hash layer per step — no new kernel at
  all. Declined on scope (it needs a host-side dequant of every hash layer's
  table on `SessionStable`, one extra graph Input per hash layer, and a per-step
  upload), and because a small-`ne` I32 `embed_gather` path is the reusable
  answer. Revisit if the kernel work looks worse than the plumbing.

- **`Op::io` still omits `exp_probs_b`.** Pre-existing (it lists `x`,
  `router_x`, `router`, the three banks and `down_scale`); `expert_ids` was
  added, `exp_probs_b` deliberately left alone as out of scope. It matters only
  for the multi-device pipeline executor, which infers each op's device and cut
  tensors from `Op::io` — an `exp_probs_b` on a pipeline boundary would not be
  marked as read. No shipped pipeline config routes a deepseek2/3 MoE across a
  cut, so this has never fired.
- **The Vulkan PAGED MoE path refuses both features.** `execute_paged_moe`
  returns an error for `expert_ids` and for a clamp with
  Sigmoid/`weight_before`. The pager exists for an expert bank too large for
  VRAM (Llama-4-Scout); a V4 hash layer is not that model, and a second untested
  hash path there would have had no caller. If a V4 checkpoint ever needs the
  pager this is the gap.
- **Metal's DEVICE MoE path is only reachable for softmax gating.** Its arm
  asserts `MoeGating::Softmax`, and V4's gating is mandatory `SqrtSoftplus`, so
  a real V4 MoE layer on Metal would abort at that assert before hash routing or
  the clamp mattered. Both are implemented anyway (`moe_topk`'s `hash` flag,
  `gatedact_f32`'s `do_clamp`/`limit`) and `moe_ffn_hash_routing_parity` covers
  the softmax shape, but the sqrt-softplus refusal is the real blocker for V4 on
  Metal and is untouched.
- **Metal has never been executed.** No Apple hardware was available. The
  changed kernels (`moe_topk`, `gatedact_f32`, `gatedactfused_f32`) typecheck
  via `cargo check -p infr-metal --all-targets --target x86_64-apple-darwin`;
  MSL compiles on-device, so the macOS CI job is their first real compile and
  first execution. `gatedact_swiglu_clamp_parity` and
  `moe_ffn_hash_routing_parity` in `crates/infr-metal/tests/parity.rs` are what
  will report it. Note in particular that `GatedActParams` and `GatedParams`
  both GREW by two words — every dispatch site had to be updated to push the
  longer struct, and a missed one reads garbage into `do_clamp`.
- **Vulkan's strided/sigmoid gated forms refuse the clamp.** `gelu_mul_off`
  (gemma4's per-layer-embd gate) and `mul_sigmoid` push the clamp words as zero
  and the adapter returns an error if a clamp reaches them. V4 is SiLU, so
  nothing needs them; a future clamped GELU-with-strides would.
- **Performance was not considered.** The clamp adds a workgroup-uniform branch
  and one `min`/`clamp` per element to `silu_mul`/`silu_mul_fused` — expected to
  be free, not measured. `moe_topk`'s hash branch replaces the whole `n_used`
  reduction with an `n_used`-element copy, so it can only be faster; also not
  measured.

### B-DSV4-METAL-CLIPPY — `cargo clippy --target x86_64-apple-darwin` is red at HEAD (2026-08-10)

**Tag:** metal · lint · **Blocked on:** nothing; pre-existing, not this slice's

`crates/infr-metal/src/exec.rs`'s `Op::Mla` arm has two `manual_map` clippy
errors (the `freq_factors` and `key_bias`
`match … { Some(x) => Some(f(x)), None => None }` pairs), so
`cargo clippy -p infr-metal --all-targets --target x86_64-apple-darwin -- -D warnings`
fails. Confirmed present at `HEAD`
(`git show HEAD:crates/infr-metal/src/exec.rs`), i.e. it came in with the MLA
`key_bias` work, not with the hash-routing slice — left alone on scope grounds.
The workspace clippy run is green because those lines are Apple-only and a Linux
build never compiles them. Worth checking whether CI lints that target at all;
if it does, it has been red since `key_bias` landed.

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
