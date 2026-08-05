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
- **A chunked prefill re-reads the whole model once per chunk.** The prefill
  loop (`infr-llama/src/seam/runner.rs`, the `cstart`/`cend` walk) runs the full
  graph per `ubatch` chunk, so a P-token prompt costs `ceil(P / ubatch)`
  complete weight sweeps. That is invisible when the weights are resident and
  brutal when they stream: at the 1024-row default a 32k prompt is 32 sweeps.
  **Layer-major prefill** — layers outer, chunks inner — reads the model ONCE
  per prefill instead. The cost is holding every chunk's activations at once
  rather than one chunk's: 32k x 4096 x 2 B is ~268 MB, affordable. This is the
  largest unbuilt win for streamed models and it matters most for the
  DeepSeek-class targets this feature exists for. It is a real restructuring of
  the prefill path, not a tweak, and it should be gated on the model actually
  streaming (resident models must keep today's chunk-major order, which is right
  for them).
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

### B27 — hardening candidates from the 2026-08-03 review

**Tag:** CR-2026-08-03 hardening · **Blocked on:** nothing; none of these is an
established defect and none was verified in the fold-in pass

Kept with the deleted report's framing intact: the review listed these as **"not
established current defects"** — places where a stronger construction would
survive a case nobody has shown to occur. Do not promote one to a bug without
first exhibiting that case.

- `with_profiling_suppressed` restores a process-global boolean only on normal
  return; an RAII nesting counter would survive panics and overlapping scopes.
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
  validation (see also B14).
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

### W3 — pin the kernel tier when `-r > 1` (B6's original remedy)

**Claim:** the prefill columns' run-to-run spread came from a short prefill
landing on a different kernel tier between runs, so the tier should be pinned
for a multi-rep bench.

**Why it is wrong:** there is no tier to pin. `INFR_PROF_OPS=1` over six
back-to-back runs gives a byte-identical (op name, dispatch count) signature
while throughput moves 9%, `adaptive_chunk` is a pure function of the KV span,
and every rung that could move a chunk logs at WARN and never fired. The real
cause was a warmup at the wrong shape (B6), and pinning would have fixed
nothing.

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
