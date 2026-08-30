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

**Tag:** raised 2026-08-02 · **Blocked on:** nothing — the product decision was
made 2026-08-12, see "DECIDED" at the end of this entry

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

**DECIDED 2026-08-12 (user).** An explicit `--ctx N` that does not fit **must
try q8_0 KV before doing anything else**: _"when setting `--ctx` that would not
fit in VRAM using k/v cache, have infr auto quant the context to Q8_0 if it will
fit."_ So the explicit path walks the same ladder `clamp_default_ctx` already
has — f16, then q8_0, then refuse — instead of being taken verbatim and failing
later at the alloc guard. Quantizing to serve the window the user asked for is
preferred over clamping the window; refusing stays the last rung.

To implement: route `vulkan_session_on` and `vulkan_slot_ctx`'s
`SizeSpec::Bytes` arm through the fit math rather than returning the value
verbatim. Two constraints from above still bind — the check must be **provable**
(exact KV bytes + weights vs `alloc_room()`, no activation reserve in it), and
the DEFAULT-path refuse rung must not widen (only "no usable context at all" may
refuse, or every ordinary long-context model breaks on a 24 GiB card).

Sequencing note: this is one rung of the ladder in `vram-audit-2026-07-12` (SWA
→ q8 → clamp → stream → KV-overflow last), and the rung ORDER itself is what B62
asks someone to verify.

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
- **Layer-major prefill is now explicit-only.** It originally made the chunk
  loop run inside the layer loop for a streamed model
  (`seam::layer_major_prefill`, the `spans`/`chunks` walk in
  `generate_dense_backend`), so a prompt sweeps the weight set once instead of
  once per chunk. Re-measured on the B36 shape — Qwen3-14B Q8_0 / RX 7900 XTX,
  `MemoryMax=8G`, `paging.cache=2g`, `paging.dram=6g`, P=4096, three rounds with
  the arm order permuted and a cold page cache before every run — at the
  1024-row default chunk: 25.27 → 6.31 GB read and 341.9 → 779.9 pp t/s, the
  read volume now exactly a single-chunk prefill's. A later automatic Qwen3.8
  paged-MoE extension did not generalize: on Q2 pp4096, chunk-major measured
  169.44-205.00 tok/s at ubatch 1024-2048 against layer-major's 11.44-13.34.
  Unset and `paging.layer_major=false` therefore use chunk-major everywhere;
  `paging.layer_major=true` retains the implementation for explicit A/B. The
  original dense experiment's residue remains useful context:
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

### B60 — every infr-vs-llama.cpp cosine in the docs is the wrong metric (2026-08-11)

**Tag:** CR-2026-08-11 · **Blocked on:** nothing; the replacement metric exists
and is in use

`cpu_prefill_matches_llama_debug_dump` scores infr's prefill against llama.cpp's
on the same GGUF over llama.cpp's own token ids. Running it produced a result
that invalidates how this repo has been reporting agreement:

| pair                                         | logit cosine | probability cosine |
| -------------------------------------------- | -----------: | -----------------: |
| V2-Lite, same prompt, **both pick " Paris"** |    **0.774** |             0.9969 |
| Qwen3-0.6B, same prompt                      |       0.9985 |             0.9994 |
| Qwen3-0.6B, **unrelated prompt**             |    **0.851** |             0.0164 |

A **correct** deepseek2 match scores BELOW an **unrelated** Qwen pair. A
whole-vocab logit cosine is dominated by the per-token bias every row of a model
shares, so it cannot separate a match from a mismatch, and the "~0.79–0.91
established range" the docs quoted was measuring that bias rather than
agreement. `docs/deepseek.md`'s YaRN checklist entry is corrected; the
greedy-token identities it reports are real evidence and stand.

Where this still needs sweeping: any other cosine figure quoted as evidence in
`docs/`, and — separately — the CPU-vs-**Vulkan** cosines (`0.9955` for
deepseek2, `0.99999992` for V4). Those compare two runs of the SAME model and
weights, so the shared bias argument is weaker and they may well be fine, but
they have not been re-examined against a probability cosine and should not be
assumed sound just because they are high.

### B61 — the native-block GEMM/MMQ family keeps the unguarded `k/BLK` floor (2026-08-12)

**Tag:** vulkan · **Blocked on:** nothing; a decision about how far the B39
guard should reach, not a discovery

B39 (Vulkan native-block GEMV silently returning zeros below the 32-element
sub-block floor) was fixed by asserting the K invariant at dispatch —
`assert_native_k` in `crates/infr-vulkan/src/recorder.rs`, called from every
terminal native GEMV/gather dispatcher (the `linear_native*`, `linear_mmv*`,
`linear_native_id*` and `embed_gather_at` entry points). The **prefill GEMM /
batched-MoE MMQ** family was checked and deliberately left alone: every
`shaders/native_gemm_mmq_*.comp` opens with `nblk = pc.k / BLK` (plus
`kw = pc.k / 4`), and `native_gemm_i8cm_q8_0.comp` with `nblk = pc.k / BK`, all
of which floor to zero exactly the same way. Their host entry points
(`Recorder::matmul_mmq_at`, `matmul_mmq_experts`, `matmul_mmq_experts_paged`,
`matmul_i8cm_q8_0_at`, …) assert nothing about `k`.

Why it was not swept in with B39: those particular shaders are reachable only
for quantized dtypes (the `native_gemm_mmq_*` kernel tables and
`infr_core::tensor::MOE_MMQ_DTYPES`; `i8cm` is Q8*0 only), whose GGUF block
sizes are 32/64/256 and therefore already divide any legal `k` — so unlike the
GEMV path there is no dtype (BF16/F16/F32) that can even express an off-grid
row. Reaching it needs a hand-built synthetic tensor, which is precisely the
hazard B39 was about, so this is a real gap and not a non-issue. The reason to
stop was scope: each
`matmul*\*` entry carries its own tiling constraints (`gemm.rs`'s `matmul_f16`already asserts`m,n
%64 && k %32`) and several take `k` with different meanings, so a uniform guard
needs its own read of the family rather than a mechanical sweep.

**Not the same defect, checked while here:** `native_gemm.comp` (the coopmat
prefill GEMM, the one path that DOES take BF16) steps
`for (k0 = 0; k0 < pc.k; k0 += BK)`, so it never runs zero iterations — an
off-grid `k` makes its `dqblk` over-read past the row instead. Its tier is
already gated on `in_f % 32 == 0` in `adapter.rs`'s `Op::Linear` arm
(`gemm_ok`), so it is unreachable off-grid today; noted so a future reader does
not mistake it for another instance of the floor-to-zero bug.

To finish: audit the `matmul_*` dispatchers in `recorder.rs` for what each one's
`k` actually indexes, then add `assert_native_k` (or a tile-aware equivalent) at
each. Not verified: whether any existing test drives a `matmul_*` entry with an
off-grid `k` — that is what would turn the sweep from mechanical into a real
change.

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
- **A real V4 GGUF HAS now been opened** (2026-08-12) —
  `ggml-org/DeepSeek-V4-Flash-0731-GGUF:Q2_K_S`, 43 layers, 1328 tensors. Every
  tensor name and every shape `dsv4_model`'s formulas derive, evaluated at the
  real file's hyperparameters, matches the file exactly: no missing tensor, no
  extra tensor, no shape mismatch. `Config::from_gguf` and `wload`'s `is_dsv4`
  arm both run clean and the model reaches the designed compressed-layer
  refusal. The formulas that were only inferred and are now MEASURED:
  `attn_output_a` = `[n_head·key_length/o_groups, o_lora_rank·o_groups]`,
  `attn_output_b` = `[o_groups·o_lora_rank, n_embd]`, `hc_*_fn` =
  `[hc_mult·n_embd, (2+hc_mult)·hc_mult]` with a `[3]` scale, the compressor's
  `coff = 2 iff ratio == 4` channel doubling, and `attn_compressor_norm` staying
  `[key_length]` in BOTH tiers while `_kv`/`_gate`/`_ape` widen with `coff`.
  What is still NOT verified against a real file: any NUMBER the graph computes
  — the refusal fires before a single forward pass, so shapes and names are all
  this proves.
- **The real file's dtype set is wider than the fixture's.** The fixture writes
  f32 for everything but `ffn_gate_tid2eid` (now i32, matching); the real file
  is `{Q2_K: 129, Q8_0: 660, F32: 492, BF16: 43, Q6_K: 1, I32: 3}`. The 43 BF16
  tensors are the per-layer `ffn_gate_inp.weight` routers — a bf16 router has
  never been exercised by any V4 test, and the fixture cannot express one
  without a per-tensor dtype beyond the i32 arm added for the routing table.

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

### B-DSV4-HASH — what hash-routed MoE / the SwiGLU clamp do NOT cover yet (2026-08-13)

**Tag:** deepseek · vulkan · metal · **Blocked on:** nothing

Hash routing now EMITS and generates on CPU + Vulkan: `Op::GatherI32`
(`gather_i32.comp`, and the CPU interpreter arm) reads the token's row of
`blk.N.ffn_gate_tid2eid` into the `[batch, n_expert_used]` selection
`Op::MoeFfn::expert_ids` consumes, and `generate_dense_backend` no longer
refuses a hash-routed layer. See `docs/deepseek.md` § Stage 4, "Slice A2". What
is left:

- **`Op::GatherI32` is REFUSED on Metal**, by name, in `run_op`. Implementing it
  alone would move the failure one op later without making anything work: the
  Metal DEVICE `Op::MoeFfn` path asserts `MoeGating::Softmax` and V4's gating is
  mandatory `SqrtSoftplus`, so a V4 MoE layer cannot run there either way (the
  same blocker this entry has recorded since 2026-08-10). Whoever lifts the
  sqrt-softplus assert owns the gather too — it is one small MSL kernel, or a
  host arm beside `Op::MoeFfn`'s existing host fallback, which already
  implements `expert_ids`.
- **The gather has NO `_dyn` twin and was not considered for the replay tape.**
  It falls through `decode_eligible`'s `_ => {}`, i.e. it is treated as
  replay-safe (it is: pos-independent, and its `ids` input is the same buffer
  `Op::EmbedGather` reads under the chained decode). Nothing exercises that,
  because V4 is excluded from the tape outright for four other reasons (§
  B-DSV4). Re-examine when V4 gets its tape.
- **The gather runs once per hash layer per step, un-fused and unmeasured.** All
  `hash_layer_count` layers gather the SAME token's ids at the same `n_used`
  width, so the whole selection could be one dispatch over a stacked table (or
  cached across layers) instead of one dispatch each. At `hash_layer_count = 3`
  on the shipped file that is 3 dispatches of 6 threads — expected to be noise
  next to the MoE itself, but nothing was profiled.
- **The table is read as ONE bound storage buffer**, not through the
  resident-BDA arena. Fine at `n_expert_used * n_vocab` dwords (3.1 MB on the
  shipped file, 8 MB at a 256k vocab and `n_used = 8`), and `Recorder::vkb`
  asserts the range against the device's real `maxStorageBufferRange` — but a
  future table that tripped that assert would need a `-DSTREAMED` twin, which
  does not exist.
- **The host-gather alternative was reconsidered and declined again.** The token
  ids ARE host-known on every V4 path today (V4 takes neither the record-once
  tape nor the chained decode, so every step goes through the per-token loop
  where the host has just fed `cur[pos]`), so a host gather would have worked
  with no new kernel. Declined because it needs one extra graph Input and one
  extra per-step upload PER HASH LAYER, a host dequant of every table, and
  because it would have to be undone the moment V4 gets a GPU-resident decode —
  where the sampled id never reaches the host. The device gather costs one small
  shader and reuses the ids Input that already exists.
- **Only `batch == 1` has been executed.** `Op::GatherI32` is written for any
  `rows` and the op-level parity test drives `rows = 2`, but the WHOLE-MODEL
  path is per-token: V4 is excluded from batched prefill (§ B-DSV4-WIRING), and
  `build` asserts a hash-routed span cannot be built without the ids Input,
  which the chunked prefill has nowhere to bind. Re-enabling batched V4 prefill
  means giving `PfChunk` an ids buffer that is separate from its host-embedded
  rows.

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
  a real V4 MoE layer on Metal would abort at that assert even with the gather
  in hand. Hash routing and the clamp are implemented there anyway (`moe_topk`'s
  `hash` flag, `gatedact_f32`'s `do_clamp`/`limit`) and
  `moe_ffn_hash_routing_parity` covers the softmax shape; the sqrt-softplus
  refusal is the real blocker for V4 on Metal and is untouched. Lifting it is
  what makes the refused `Op::GatherI32` above worth writing.
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

### B-DSV4-REAL — the shipped V4 file now needs ONLY the compressed tiers (2026-08-13)

**Tag:** deepseek · **Blocked on:** B-DSV4-WIRING slice B

Measured from `ggml-org/DeepSeek-V4-Flash-0731-GGUF:Q2_K_S` (`block_count = 43`,
`hash_layer_count = 3`, `compress_ratios` = `[0, 0, 4, 128, 4, 128, …, 4]` over
the 43 layers, then three surplus zeros): **no layer of the real model is both
ratio-0 and bias-routed.** Layers 0 and 1 are the only ratio-0 layers and both
are hash-routed; layers 2..42 are all compressed (alternating 4/128), and layer
2 is compressed AND hash-routed.

Hash routing landed on 2026-08-13 (§ B-DSV4-HASH), so layers 0 and 1 — 2 of the
43 — are now fully covered, and **the only thing between this checkpoint and a
forward pass is B-DSV4-WIRING slice B**, the compressed (ratio 4/128) tiers.
That is also the only refusal left in `generate_dense_backend` for this file:
`arch=deepseek4 … layer 2 has compress_ratio 4`.

What is NOT established by this entry: any NUMBER the graph computes on a real
file. The refusal still fires before the first forward pass, so what has been
verified against the real GGUF is names, shapes and the two hash-routing
prerequisites (an i32 `blk.{0,1,2}.ffn_gate_tid2eid.weight` of shape
`[6, 129280]` = `[n_expert_used, vocab]`, which is exactly what the new shape
check demands) — nothing about its logits. The hash-routing correctness evidence
is the synthetic fixture and the op-level parity tests, on CPU and Vulkan.

Verified by running `infr run` against the file on the CPU backend (2026-08-12):
config parses, `wload` resolves every tensor, and the refusal is reached. NOT
re-run after the hash slice — the compressed-layer refusal is ordered first and
fires at layer 2 regardless, so the observable would not have moved.

### B-GGUF-META-IGNORED — keys the reference conversion carries that infr never reads (2026-08-12)

**Tag:** gguf · **Blocked on:** a decision on whether to honour them

Two metadata families the real DeepSeek-V4 file declares and infr has no reader
for. Neither is a V4 problem — both are general — and neither is wrong today,
but both are silent.

- **`general.sampling.*`** (`llama-arch.h`'s `LLM_KV_GENERAL_SAMPLING_*`:
  `temp`, `top_k`, `top_p`, `min_p`, `xtc_*`, `penalty_last_n`, `sequence`). The
  V4 file sets `general.sampling.temp = 1.0` and `general.sampling.top_p = 1.0`;
  infr applied its own per-arch defaults (`temp = 0.6, top_k = 20, top_p = 0.95`
  for `deepseek4`) and never looked. `grep -rn "general.sampling" crates/` is
  empty. The question is precedence: a model-declared sampler should presumably
  sit between infr's arch defaults and the user's flags, but nothing implements
  any of the three orderings today.
- **`{arch}.embedding_length_out`** = 16384 on V4 (= `hc_mult · n_embd`, the
  un-collapsed hyper-connection residual). In llama.cpp this feeds
  `llama_hparams::n_embd_out()`, which sizes the EMBEDDINGS output buffer only —
  never the logits path — so ignoring it costs nothing while infr has no
  embeddings endpoint. It becomes wrong the moment one exists for an arch whose
  output width differs from `embedding_length`.

<!-- ── hardware-capability audit, 2026-08-11 ─────────────────────────────────
     Three follow-up slices came out of one read-only audit of what infr detects
     about a GPU versus what llama.cpp detects. The prefixes are the slices:
       B-HWDET-*   detection and gating          (task #16)
       B-NVSHAPE-* NVIDIA kernel shapes          (task #17)
       B-DSHW-*    DeepSeek-specific fast paths  (task #18)
     HARDWARE REALITY FOR EVERY ENTRY BELOW: this box has one RX 7900 XTX
     (RDNA3, RADV) and no NVIDIA, Intel or Apple part. Each entry states its own
     evidence; anything about NVIDIA or Intel BEHAVIOUR is read off llama.cpp's
     source, never observed here. -->

### B-HWDET-VENDOR-RULES — the coopmat driver rules and the arch bucket exist now, and NONE of their non-RADV arms can be run here (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** NVIDIA / Intel / AMD-proprietary
hardware; nothing else

The capability-first design now has exactly one vendor-keyed exception, landed
as `caps::coopmat_trust` + `caps::device_architecture`
(`crates/infr-vulkan/src/caps.rs`), applied in `VulkanBackend::new` as a FILTER
over the enumerated coopmat shape list. Both rules are transcriptions of
llama.cpp's `ggml_vk_khr_cooperative_matrix_support` / `get_device_architecture`
(read at pin 030ebb5): AMD proprietary + AMDVLK are believed only on RDNA3+, and
Intel pre-Xe2 loses the 16x16x16 tier while keeping the 8x8x16 tile that already
sits behind `INFR_CM_8X8=1`.

**What is actually verified:** only the RADV/RDNA3 arm, on this box — the banner
now prints `AmdRdna3/MESA_RADV` and every coopmat tier stayed live
(`f16cm:y i8cm:y`), i.e. the filter is a no-op here. Everything else is
unit-tested over synthetic `DeviceProbe` values only (`caps::tests`), which
proves the RULE, not the PROBE that feeds it: no NVIDIA, Intel or
AMD-proprietary device has ever filled a `DeviceProbe` in this tree.

**The specific things a machine with one of those parts should check first:**

- That `probe_device_facts` actually fills `wavefronts_per_simd` /
  `warps_per_sm` / the packed-dot bits on that driver. A silently-zero field
  misfiles the device as `DeviceArch::Other`, which on an AMD proprietary driver
  means coopmat REFUSED (fail-safe) but on Intel means the 16x16x16 tier refused
  too.
- Whether the Intel `Tile8Only` verdict is the right shape for `infr` at all. It
  was chosen so the `_cm8` `native_gemm_warp` builds stay reachable on Alchemist
  (they are the only kernels at that shape, and the only reason the opt-in
  exists); the alternative — refusing Intel pre-Xe2 coopmat outright, as
  upstream does — would make those builds dead code on every Intel part.
- That an `AMD_PROPRIETARY` refusal is survivable: it drops the device to the
  non-coopmat ladder, which is a large prefill regression, not a failure. There
  is no force-on knob to override the deny-list; adding one is the obvious
  escape hatch if a real device is misjudged.

**Considered and declined:** generalizing the known-answer validation (option 3
of the retired B-HWDET-DRIVERID entry) to the f16/bf16/f8 coopmat tiers. The
int8 one was built because that kernel reads accumulator elements at an
implementation-defined mapping; every other coopmat kernel here moves data in
and out of fragments only through `coopMatLoad`/`coopMatStore` with an explicit
layout enum, so there is no per-element assumption to falsify — a known-answer
GEMM there would test the driver's arithmetic, not infr's assumption, and would
cost init latency on every run.

### B-HWDET-LIMITS — three device limits are still assumed at the spec minimum (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** nothing; the remaining items are
perf-unmeasured or currently unreachable

Two of the five items in the original audit entry are closed:
`maxPushConstantsSize` is queried and every kernel's block is checked against it
(`caps::check_push_constant_size`, called from `ops::try_make_compute_kernel`),
and `maxStorageBufferRange` is queried into `crate::max_storage_buffer_range()`
with `Recorder::vkb`'s bind check promoted from `debug_assert!` to a release
`assert!` against the device's real number. What is left:

- **`maxComputeWorkGroupCount[0]`** → `Recorder::MAX_GROUP_COUNT_X` is still
  pinned at the guaranteed minimum, so every dispatch wider than that pays a 2-D
  split RDNA3 does not need. Reading the real limit is a few lines; whether the
  split costs anything measurable has **never been profiled**, and that
  measurement is the actual blocker — not the code.
- **`maxComputeWorkGroupInvocations` / `maxComputeWorkGroupSize`** → never
  queried; every `local_size_x` is a compile-time constant at or under the
  guaranteed 1024. Safe by spec, not by check. A guard would have to live in the
  build (the sizes are baked into SPIR-V), which is why it was not added with
  the push-constant one.
- **`minStorageBufferOffsetAlignment`** → never queried; 256 (the spec's MAXIMUM
  for this limit, so it over-aligns on every device) is hardcoded in the range
  padding. Correct everywhere, wasteful nowhere that has been measured.

**Not verified:** the new release `assert!` in `vkb` has never been seen to FIRE
— no tensor in any model here comes near this device's 4 GiB
`maxStorageBufferRange`, and the streamed/BDA path is what carries the big ones.
It is a guard for a device that reports a smaller limit, which is not this one.

### B-HWDET-FEATUREBITS — the bf16/f8 feature-bit code is written and cannot be exercised on this box (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** RDNA4-class hardware

`VK_KHR_shader_bfloat16` and `VK_EXT_shader_float8` are no longer gated on the
extension string alone. `crates/infr-vulkan/src/vkext.rs` supplies the two
feature structs ash 0.38 predates (layout and `sType` transcribed from the
system `vulkan_core.h`), `caps.bf16`/`caps.f8` now mean extension AND feature
bit, the coopmat tiers additionally require `shaderBFloat16CooperativeMatrix` /
`shaderFloat8CooperativeMatrix` (the bits the `-DBF16CM` and fp8 kernels'
operand types actually need), and both extension+feature are ENABLED on the
device when their tier is live — which they never were before, so those kernels'
SPIR-V would have violated its VUID on the first RDNA4 run.

**None of that has run.** RDNA3 advertises neither extension, so the query is
skipped and the enables are absent; this box's device-create chain is byte
identical to before. What IS exercised here is the raw `sType`/`pNext` chaining
those structs depend on: `vkext::tests::raw_chaining_matches_ash` hand-rolls
`VkPhysicalDeviceShaderFloat16Int8Features` and asserts the bits match ash's
typed query (passes on the 7900 XTX). So the mechanism is tested and the two
payloads are not.

**First thing to check on an RDNA4 part:** that `create_device` still succeeds
with `INFR_BF16_COOPMAT=1` — enabling a feature the device reports false is a
device create FAILURE, and `ShaderBfloat16Features::enable` only asks for bits
the query returned, which is exactly the code path nobody has run.

### B-HWDET-NO-ARCH-BUCKET — the per-op subgroup-width table is still not representable, and is still a guess (2026-08-11)

**Tag:** vulkan · detection · **Blocked on:** a measurement nobody can take here

The architecture bucket itself now exists (`caps::DeviceArch`, see
B-HWDET-VENDOR-RULES) and has exactly one consumer: coopmat trust. What the
audit listed as the thing a bucket would BUY is still absent and still
unjustified:

- **A per-architecture, per-pipeline subgroup-size table** (upstream's
  `gpu_pipeline_configs` / `rdna1_pipelines` / `rdna2_pipelines`). `infr` has
  two pinned widths — a global 32 and `sg_pref` for one curated decode family —
  so a per-op width cannot be expressed at all. Adding the table is a real
  change to the kernel-cache key, and the evidence for it is upstream's tuning
  of DIFFERENT kernels on hardware that is not here. Still a guess, deliberately
  not built.
- **Wave64-only hardware stays unsupported by rule**: the backend hard-refuses
  any device that cannot pin subgroup 32, before the bucket is ever consulted.
  Recorded as a decision, not a gap. `DeviceArch::AmdGcn` exists only so such a
  part is classified as known-unsupported rather than unknown.

### B-NVSHAPE-COOPMAT2 — `VK_NV_cooperative_matrix2` is gated and researched; the kernel path does not exist (2026-08-11)

**Tag:** vulkan · nvidia · **Blocked on:** NVIDIA hardware; nothing else

The adapter has no coopmat2 kernel — apart from the capability gate described
below, the only coopmat2 code anywhere in `crates/` is
`examples/coopmat2_test.rs`, a standalone probe with its own instance and
device. It is unusually well prepared: its header records the
`coopMatPerElementNV` signature (a `void` with an `out` result, not a
value-returning call), confirms via `spirv-dis` that it lowers to
`OpCooperativeMatrixPerElementOpNV` with no shared memory and no
`OpControlBarrier`, and it benches two epilogue strategies (single-tile and a
full 512×2048×2048 int8 GEMM) to answer whether per-element access removes the
int8 "rescale tax". **It has never been run** — RADV does not expose the
extension and there is no NVIDIA part here. So the question it was built to
answer is still open.

The direct motivation is B-HWDET-I8CM-FRAGLAYOUT: coopmat2's per-element
callback addresses `(row, col)` **portably**, which would retire the empirically
derived fragment mapping entirely on hardware that has it.

**The GATE now exists** (`caps::check_coopmat2_support` over `Coopmat2Probe`,
probed by `probe_coopmat2` in `lib.rs` and reported as `cm2:` in the device
banner). It is a transcription of llama.cpp's — all seven coopmat2 feature bits
plus `bufferDeviceAddress`, then fp16 A/B with fp16 AND fp32 accumulators at
BOTH 128- and 256-invocation workgroup sizes, then `maxDimension >= 512` — and
it fails closed on anything it cannot establish. What is still missing is the
kernel path: nothing consumes an `Ok`.

What has actually been exercised here: the refusal path on lavapipe, which
**does** advertise `VK_NV_cooperative_matrix2` on this box (device `Vulkan2`)
and is refused because `cooperativeMatrixWorkgroupScope` is clear — i.e. the
case that proves an extension-string check would have been wrong. No device here
has ever reached `Ok`, so the ACCEPT side of the gate rests on unit tests alone.
The optional bf16 half of upstream's gate was deliberately not copied: it only
feeds bf16 coopmat2 shaders, and there is no coopmat2 shader of any kind.

`examples/coopmat2_test.rs` deliberately does NOT go through this gate. It uses
only `coopMatPerElementNV`, not flexible dimensions / tensor addressing / block
loads, so the full gate would make it skip on hardware where its own experiment
would run.

### B-NVSHAPE-CORECOUNT — the shader-core count now exists on every vendor; nothing occupancy-driven consumes it (2026-08-11)

**Tag:** vulkan · nvidia · intel · **Blocked on:** NVIDIA/Intel hardware to
validate; a decision on whether the split-K constants should be device-derived
at all

`caps::shader_core_count` now sources the count the way llama.cpp does —
`VK_NV_shader_sm_builtins`' `shaderSMCount` on NVIDIA,
`VK_AMD_shader_core_properties2`' `activeComputeUnitCount` on AMD, and a PCI
device-id table on Intel transcribed verbatim from
`ggml_vk_intel_shader_core_count` — with the old `VK_AMD_shader_core_properties`
(v1) product kept as a fourth fallback so an AMD driver advertising only v1 does
not regress to 0. It lands in the existing `Capabilities::compute_units` (0 =
unknown), so the field name and every consumer are unchanged.

Verified on this box: both AMD devices report the same number through v2 as they
did through the v1 product (RX 7900 XTX 96, Raphael iGPU 2), so preferring v2 is
a no-op here. **The NVIDIA and Intel branches have never executed** — no such
part exists on this machine.

**Consequence nobody has watched run:** the only consumer is
`infr_core::integrated_ubatch_rows` via `infr_llama::seam::default_ubatch_rows`,
which scales the integrated prefill chunk from the count. An Intel iGPU in the
table (0xB080, Panther Lake Xe3 LPG, 12) previously got 0 → the 128-row floor
and would now get 768 rows. That is what the scaling function means, but it is a
6x larger per-submit chunk on a watchdog-sensitive part, unvalidated. If an
Intel iGPU ever hangs in prefill, this is the first thing to look at;
`INFR_UBATCH` overrides it.

**Deliberately NOT wired into split-K or tile selection.** The audit that
produced this entry left `infr`'s split-K selection untraced; it has now been
traced, and the answer is that a core count does not obviously improve it:

- Four independent sites carry a hardcoded "workgroups needed to fill this GPU"
  number — `split_k_plan` (`adapter.rs`, threshold 128 / target 256), the two
  flash split-K formulas and the PV split-K formula (`recorder.rs`, threshold
  1024 / target 2048), plus the `wide_grid >= 128` wide-tile gates and
  `ATTN_SPLIT.target_chunks = 32`.
- Their implied workgroups-per-core ratios are not the same number: 256/96 ≈ 2.7
  for the GEMM, 2048/96 ≈ 21 for attention. So the constants encode per-kernel
  occupancy (waves in flight, register/LDS pressure), not core count, and
  nothing in the probe knows that factor.
- Replacing them with `cores * K` therefore means inventing a `K` per site,
  chosen so that 96 reproduces today's empirically tuned constant, and then
  extrapolating it onto hardware nobody here can measure. That is a guess with a
  silent failure mode, so it was not built.
- Two extra costs if someone does: `split_k_plan` is pinned by a formula
  equivalence test in `adapter.rs`, and the attention chunk policy is
  **duplicated in GLSL** (`attn_live.comp` recomputes it on the GPU), so a
  device-derived target needs a new push constant.

The honest next step is a measurement, not a refactor: a second GPU with a
different core count, running the four sites' current constants against
core-derived ones.

### B-NVSHAPE-MMQ-PER-ARCH — upstream tunes quantized-GEMM tile shape per microarchitecture; infr has one shape (2026-08-11)

**Tag:** nvidia · perf-design · **Blocked on:** nothing to read; everything to
validate (no NVIDIA hardware)

**This is a CUDA finding, and `infr` has no CUDA backend** — it is recorded as a
design reference for how far per-shape tuning can go, not as a portable path.

llama.cpp's CUDA MMQ kernel selects its configuration from a
per-microarchitecture table:
`ggml/src/ggml-cuda/mmq-config-{pascal,ampere,blackwell,cdna,rdna2,rdna3,rdna3-5,rdna4}.cuh`.
Each entry is keyed on `(quant type, J, fallback)` and carries `nthreads`,
`occupancy`, `I`, `J`, an SRAM layout, `K_vram`, and a `stream_k` flag — the
`ggml_cuda_mmq_config` struct in `mmq.cuh`. Read directly. Two rows for the same
quant type show the spread: Ampere runs `nthreads=256, occupancy=1, I=128` with
**`stream_k = true`**, while RDNA3 runs
`nthreads=128/256, occupancy=2, I=64/128` with `stream_k = false`, and the two
enumerate different J granularities. `rows_per_warp()` also branches — 16 on AMD
MFMA/WMMA, 32 on NVIDIA when `J >= 48 && J % 16 == 0`.

Two things worth carrying over independent of CUDA:

- **Stream-K decomposition** is on for NVIDIA and off for RDNA in this table.
  `infr` does not have stream-K at all (it has split-K, which is a different
  decomposition). Whether it would win is unmeasured and unknown.
- The **tile shape is a function of the quant type and the N dimension**, not a
  constant. `infr`'s GEMM shape gates are coarse by comparison (`out_f % 256` /
  `in_f % 32` style divisibility tests choosing WIDE vs NARROW_N).

For the NVIDIA tensor-core shapes themselves: `mma.cuh` issues
`mma.sync.aligned.m16n8k16` and `m16n8k32` for `s8×s8→s32`, falling back to 2×
and 4× `m8n8k16` on Turing (`__CUDA_ARCH__ < GGML_CUDA_CC_AMPERE`); f16 uses
`m16n8k16` with a 2×/4× `m16n8k8` Turing fallback; there is a tf32 `m16n8k8` and
a Blackwell block-scaled `m16n8k64` mxf4/nvf4 path. Note `TURING_MMA_AVAILABLE`
is `!defined(GGML_USE_HIP) && __CUDA_ARCH__ >= 750`, so **none** of this is
reachable on AMD-via-HIP; AMD goes through `AMD_MFMA_AVAILABLE` /
`AMD_WMMA_AVAILABLE` instead. All read directly.

### B-NVSHAPE-AMD-NOT-HERE — CDNA MFMA and RDNA4 WMMA cannot be exercised on this box (2026-08-11)

**Tag:** amd · coverage-gap · **Blocked on:** hardware

Stated as a coverage gap, not a task. This machine's RX 7900 XTX is RDNA3
(gfx1100). From llama.cpp's own architecture constants, read directly in
`ggml/src/ggml-cuda/common.cuh`: RDNA3 is the **minimum for WMMA**, CDNA1 the
minimum for MFMA, RDNA2 the minimum for dp4a. So this box can exercise WMMA and
dp4a but **never** MFMA (CDNA/MI-series only) and never RDNA4's WMMA generation
(RX 9000).

Concretely, that means the following stay unvalidated here no matter how much
work is done: `infr`'s `coopmat_bf16` and `coopmat_f8` tiers, whose field docs
already say they are `None` on all pre-RDNA4 hardware and whose dispatches are
opt-in specifically because "this dev box has none" (read in `adapter.rs`'s
`bf16cm_ok` comment). Any claim that those tiers are correct is currently a
compile-time claim only.

### B-DSHW-MLA-SCALAR-TIER — DeepSeek attention has one subgroup tier now; the occupancy arm is still missing (2026-08-11, re-measured 2026-08-12)

**Tag:** deepseek · vulkan · perf · **Blocked on:** nothing

**Re-measured 2026-08-12** on the forward that adds attention to the residual
ONCE (7e778ee) — every number below is from this box today, RX 7900 XTX / RADV,
`JenniSD/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF`, oracle = `llama-bench` on
llama.cpp's **Vulkan** backend, same GGUF, `build: c629da5 (417)`. `infr` rows
are the means of three reps with the arms ALTERNATED (`INFR_NO_MLA_SG` toggled
between reps, not run in blocks):

| bench       | before (scalar `mla`) | after (`mla_sg`)       | gain  | llama.cpp Vulkan | ratio now |
| ----------- | --------------------- | ---------------------- | ----- | ---------------- | --------- |
| pp512       | 965.8 [963.3–969.7]   | 1035.5 [1030.2–1042.8] | 1.07× | 4470.77 ± 100.32 | 4.3×      |
| pp2048      | 671.3 [670.6–672.0]   | 851.1 [847.2–855.4]    | 1.27× | 4001.94 ± 13.56  | 4.7×      |
| tg64        | 111.6 [111.5–111.8]   | 128.7 [127.9–129.4]    | 1.15× | 225.73 ± 7.17    | 1.8×      |
| tg64 @ 2048 | 6.3 [6.3–6.3]         | 11.9 [11.9–11.9]       | 1.89× | 216.52 ± 2.95    | 18×       |

The pre-7e778ee numbers this entry used to carry (975.5 / 673.4 / 111.2 / 6.3)
are within noise of the "before" column, so the doubled-residual fix moved
throughput by nothing — it removed one `Op::Add` per layer against a kernel that
was 90%+ of the frame.

**What shipped: `mla.comp`'s `-DMLA_SG` builds** (`mla_sg`, `mla_sg_ff`,
`mla_sg_bias`, `mla_sg_ff_bias`), selected by `gemm::mla_sg_tier` and dispatched
through `kernel_sg(.., MLA_SG_WIDTH)`. Two structural changes to the key loop,
nothing else — same dispatch shape, same bindings, same push constants:

- **One pass, not two.** The scalar build walks the key range twice (once for
  the row max, once to exponentiate), computing every `sq · K[j]` dot twice. The
  tier carries a running max and running denominator and rescales `sv` by
  `exp(m_old − m_new)` on the keys that raise the max — the same online softmax
  `attn_flash_partial.comp` already uses.
- **`subgroupAdd` + a batch, not a 7-deep tree per key.** The scalar reduction
  is a 128-lane shared-memory tree costing 9 `barrier()`s per key per pass, 18
  per key in total. The tier scores `MLA_SG_KB` keys per barrier (a
  `subgroupAdd` each, then one ping-ponged cross-subgroup fixup), so it pays ONE
  barrier per 8 keys — and the batch's V contributions sum in a register,
  turning 8 read-modify-writes of `sv` into one.

`MLA_SG_KB` = 8 was picked by sweeping it, everything else held (3 reps each):

| `MLA_SG_KB` | 1     | 2     | 4     | 6     | 8      | 16     |
| ----------- | ----- | ----- | ----- | ----- | ------ | ------ |
| pp2048      | 839.8 | 904.9 | 936.1 | 947.8 | 852.7¹ | 834.9¹ |
| tg64 @ 2048 | 10.7  | 12.1  | 13.2  | —     | 11.8¹  | 11.6¹  |

¹ the 8/16 columns are the SHIPPED (re-reading) form; 1–6 were measured on the
held-K form below, so the two halves of that row are not directly comparable —
they are here for the shape, and the shipped column is the A/B table above.

**Still open, and now the whole remaining gap: occupancy at decode.**
`INFR_PROF_OPS` at `-d 2048` (8 tokens, `INFR_SEAM_NO_REPLAY=1`) puts
`mla_sg_ff` at **94.9%** of decode GPU time — 621.2 ms of 654.4 ms, 2.9 ms per
layer dispatch, down from 5.6 ms and 97.3% before. At prefill (`-p 2048`) it is
**86.8%**, 2.08 s of 2.40 s, 38.5 ms per dispatch, from 50.1 ms and 89.4%. So
the kernel is still the frame, and the shape that has NOT been touched is the
one the original entry named first: **one workgroup per (row, head)**, i.e.
`n_head` = 16 workgroups on a 96-CU part at decode, ~83% of the device idle.
`Op::Attention`'s split-K decomposition (`attn_partial` + `attn_combine`, chunk
policy in `infr_core::tier::adaptive_chunk`, partials pooled as
`split_pm`/`split_pl`/`split_pacc`) is the pattern to copy; `Op::Mla` has no arm
of it. Note prefill CANNOT be an occupancy story — 16384 workgroups there — so a
split-K arm buys the decode rows only.

**Measured and NOT shipped: holding the fetched K across the barrier.** V is the
leading `kv_lora_rank` columns of the same K row, so the V accumulation re-reads
columns the score loop already fetched. Holding them in registers
(`kk[MLA_SG_KB][MLA_SG_VSLOTS]`) was implemented and benched at `MLA_SG_KB` = 8:
pp512 857.2, pp2048 955.3, tg64 130.9, tg64 @ 2048 13.8 — i.e. **+12% pp2048 and
+17% tg64@2048 against the shipped form, but −18% pp512**, a regression against
the kernel it replaces on the short-prompt row. Shipped the form that regresses
nothing and recorded the trade here rather than defending a 12% prefill loss.

The mechanism is NOT understood, and the obvious hypothesis is RULED OUT:
`RADV_DEBUG=shaderstats` reports **96 VGPRs and 0 scratch for BOTH forms** (the
mla build is the largest compute shader in the dump — code size 18548 held vs
14692 re-reading), so occupancy per SIMD is identical and register pressure
cannot be the explanation. Trimming the provably-dead fifth held slot moved
pp512 by 3 t/s (857.2 vs 854), which also says registers are not the lever.
Whoever picks this up: the two forms differ only in the V loop, so a selection
on `rows`/`kv_len` (the way `flash_min_rows` and the `rows >= 64` gates already
work for `Op::Attention`) would take both wins — but find the mechanism first.

**Not tried, and the next cheap idea:** `kread` issues one dword load per column
and throws half of it away — the f16 cache packs two adjacent columns per
`uint`. Giving each lane a PAIR of adjacent columns instead of a stride-128
single would halve the load instruction count outright. `key_len` is already
required to be even (`kread` addresses the row as u32 pairs), so the shape
allows it.

- **`Op::Mla` and `Op::LightningIndexer` each disqualify a graph from
  record-once decode replay — and that costs almost nothing.** The
  decode-eligibility loop in `adapter.rs` returns false on either, for stated
  reasons: MLA uses a non-standard kernel, and the indexer bakes its causal
  bound from the `pos` push constant at record time with no params-driven `_dyn`
  twin, so a replayed tape would select keys for the position it was recorded
  at. So every DeepSeek decode token does rebuild and re-record its graph — but
  `INFR_PROF_STAGES` prices that rebuild at **0.039 ms/tok of an 8.9 ms/tok**
  decode step, 0.44%, both from the same run. A separate `INFR_PROF_OPS` run
  puts GPU at 8.2 ms/tok, which INFERS (across runs, so the profiler's own
  overhead is in it) that record+submit is the only other ~0.7 ms and a replay
  tape's whole ceiling is under a tenth of the step. Do the kernel tiers first;
  this is not the DeepSeek decode problem.

**A third cost was found by measuring, and is FIXED** (`session_stable`'s
`moe_batched_ok` in `seam/runner.rs`): the expert-bank dtype scan ran over EVERY
layer, so DeepSeek's leading dense blocks (`Config::is_moe_layer` false, no
`ffn_*_exps` tensor to look up) read `None`, failed the scan, and set
`batched_prefill_ok` false for every DeepSeek model — the whole prompt was
prefilled one token per submit. Measured before/after, alternating arms: V2-Lite
pp512 39.1 → 975.5 (25×), pp2048 12.3 → 673.4 (54×); `deepseek` V1
(`mradermacher/deepseek-moe-16b-chat-GGUF` Q4_K_M, same dense-lead shape, no
MLA) pp512 229.0 → 4185.0, which is parity with llama.cpp Vulkan's 4320.07 ±
358.49 on that model. Decode is untouched by it (tg64 111.0 → 111.2). The V1
parity is the useful control: the batched MoE path itself is fine, so V2-Lite's
remaining prefill gap is `mla.comp` too.

**A coverage gap in the kernel's own parity test, found while landing the
tier.** `recorder::tests::mla_matches_cpu_reference` is the DeepSeek-SIZED
dispatch (`kv_lora_rank` 512, 16 heads) and it is SATURATED: its synthetic q and
K make the two attended keys' scores differ by ~1e7, so the softmax is a hard
argmax and the output is the winner's V regardless of the scores. Multiplying
every score by 1.01 inside the kernel leaves its `max_err` at exactly 0 — the
test cannot see a score-path defect at all, only a V/absorption/layout one. (It
already carries a comment worrying about this shape for a different reason.)
`mla_ring_and_mask_matches_cpu_reference` DOES catch it (the same perturbation
takes it to `max_err` 1.97e-4 against a 1e-4 floor), so the pair is covered —
but the big-shape test is not the guard its size suggests. Fix would be scaling
its synthetic inputs so the scores land within a few units of each other.

**Calibration, so this is not read as further behind than it is:** llama.cpp's
**Vulkan** backend has no fused indexer either (see B-DSHW-FUSED-REF). `infr` is
behind llama.cpp's CUDA backend here, not its Vulkan one.

### B-DSV2-ACTIVATION-PRECISION — int8 activations are not a DeepSeek defect, and landing f16 for `cfg.deepseek2` was declined (2026-08-11, closed 2026-08-12)

**Tag:** deepseek · reference · **Blocked on:** nothing; this is a decision
record and a ruled-out list, not a defect

This entry used to be B-DSV2-DEGENERATE — "DeepSeek-V2-Lite output decays into
repetition on Vulkan". **That symptom is fixed** and its cause was not
activation precision: `MixerW::Mla` in `seam/runner.rs` pushed its own
`Op::Add{a: hidden, b: sub, dst: hidden}` AND fell through to the shared
post-mixer residual add, so the attention output entered the residual stream
TWICE on every DeepSeek2 layer, on both backends. Removed 2026-08-12; the
generation that named this entry now reads

```
The sky appears blue because of a phenomenon called Rayleigh scattering. When
sunlight enters the Earth's atmosphere, it collides with molecules and small
particles in the air. …
```

on Vulkan (`--temp 0 --max-new 60`, `Why is the sky blue?`), with the CPU run
differing in one word ("process" for "phenomenon") — a near-tie flip, not
degeneration. `gpu_seam_matches_cpu_deepseek2` measures the whole-vocab
CPU/Vulkan cosine at **0.99861** on the 55-token sky paragraph, against the
0.9385 recorded here at 56 tokens before the fix, and that prompt + a 0.99 floor
is now what the test carries.

What is worth keeping is everything that was ruled OUT on the way, each by
measurement, plus one decision.

#### The activation-precision lead is DEAD — do not re-open it

The reading that the Vulkan `Op::Linear` int8 activation prepass (`quant_q8`,
per-32-block symmetric) was the error rested on a layer-0 isolation showing the
int8 route at 6.423e-3 relative L2 against an f64 oracle where the
f32-activation GEMV is 9.081e-8. That isolation is correct about the MATMUL and
wrong about the CONCLUSION. Two independent measurements killed it, both taken
while the doubled residual was still in the tree:

**1. Taking every activation off int8 did not fix the output, and barely moved
the agreement.** Whole-vocab CPU-vs-Vulkan logits cosine on llama.cpp's own 56
token ids for the sky-scattering paragraph, one model load per row:

| activation route                                                     | cos(cpu, vk) | pp512     | tg64  |
| -------------------------------------------------------------------- | -----------: | --------- | ----- |
| int8 `quant_q8` + dp4a mmq (shipping default)                        |       0.9385 | 977.6 t/s | 111.5 |
| f16 (`INFR_NO_MMQ=1` → coopmat GEMM, f16 A staging)                  |       0.9503 | 968.0 t/s | 111.6 |
| f32 (`INFR_NO_COOPMAT=1 INFR_NO_MMQ_FALLBACK=1 INFR_MOE_SMALL_M=64`) |       0.9625 | 708.6 t/s | 111.2 |

All three still degenerated end to end. The perf column is the honest price of
each: f16 costs **1.0% of pp512**, f32 costs **27.5%** — the latter reproducing
the −28% the reverted `exact_linear_acts` flag measured, which is the
corroboration that this row really is the all-f32-activation route.

Coverage was checked rather than assumed. `mmq` off removes the only int8
dense-Linear arm a Q4_K prefill can take on this box (`nc_mmq` needs
`!f16_coopmat()`, which is false on RADV); `moe_small_m = 64` routes the whole
56-row MoE onto `linear_native_id_multi`'s native-block f32 GEMVs instead of
`quant_q8_gather` + `matmul_mmq_experts`; the m=1 decode GEMV was already
`linear_native` on AMD (`mmv_mw_choice` returns `None`, `mmv_decode` defaults
off), which is why tg64 is flat across all three rows. `gemm_warp` off on top of
the f16 row changed the output BIT-IDENTICALLY (cos 0.951823 twice), i.e. no
warp-tile shape was in play to begin with.

**2. llama.cpp's Vulkan backend uses the SAME int8 activation prepass here.**
Read at pin 030ebb5. `ggml_vk_mul_mat_q_f16`
(`ggml/src/ggml-vulkan/ggml-vulkan.cpp:9172`) sets `quantize_y` from
`device->integer_dot_product && src1 is contiguous f32 && (ne11*ne10)%4==0` and
then tries the mmq pipeline FIRST, with no perf heuristic — it is demoted only
if no q8_1 pipeline exists for the dtype, and Q4_K has one. `quantize_q8_1.comp`
is a per-32-block symmetric `d = amax/127` quantizer, and `mul_mmq_funcs.glsl`'s
Q4_K arm is `dotPacked4x8EXT` integer dot. The gate is
`VK_KHR_shader_integer_dot_product` +
`integerDotProduct4x8BitPackedSignedAccelerated`, on by default (kill switch:
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`), which RADV satisfies. Decode is the same
prepass behind a per-vendor `k` threshold (`ggml_vk_should_use_mmvq`: AMD uses
int8 once `k >= 2048`). The one architecture that never gets int8 activations at
prefill is coopmat2 (NVIDIA `VK_NV_cooperative_matrix2`), which force-converts
src1 to f16 at `ggml-vulkan.cpp:9163` and builds no q8_1 pipelines at all.

So **infr's int8-activation default is not a divergence from upstream**: the
coherent reference implementation reduces activations exactly the same way on
exactly this hardware.

#### Also ruled OUT by measurement — do not re-walk these

- **`mla.comp` is not a defect.** Two independent checks. (1) A DeepSeek-sized
  sweep (`n_head` 16, `kv_lora_rank` 512, `qk_nope_dim` 128, `qk_rope_dim` 64,
  `v_head_dim` 128) over `kv_len` 2…2048, with and without the `-DFREQ_FACTORS`
  build, against an f64 reference: max relative error 6e-7…4e-6. (2) The REAL
  layer-0 `Op::Mla` inputs (`q`, the f16 KV rows, `wk_b`, `wv_b`,
  `freq_factors`, `rows`=59, `kv_len`=59) captured out of the CPU backend's own
  arm and replayed through `Recorder::mla`: per-row cosine 1.00000000 against
  the CPU's output, max absolute error 1.8e-7.
- **Not the YaRN trunk scale, not the rope type, not the freq_factors ramp** —
  every constant read out of llama.cpp at the pinned revision and compared to
  what infr dispatches: `kq_scale` 0.11472137 on both,
  `llama_model_rope_type(LLM_ARCH_DEEPSEEK2)` is `LLAMA_ROPE_TYPE_NORM`
  (`neox: false`), and infr's `yarn_ff` divisors reproduce `rope_yarn_ramp`
  exactly (p=0 → 1.0, p=11 → 1.081081, p≥23 → 40.0).
- **Not the in-shader Q4_K WEIGHT dequant.** `native_decode.glsl`'s `dqblk` was
  read out of the device for EVERY element of the real
  `blk.0.attn_kv_a_mqa.weight` (Q4_K, 2048→576) by dispatching
  `native_gemv.comp` with one-hot activations: 1179648 elements, **0 not
  bit-identical** to `infr_gguf::dequant::dequant_block`.
- **Not `Op::EmbedGather`'s dequant of `token_embd`** (Q4_K, 2048×102400): 64
  rows spread across the vocabulary gathered on the GPU, 131072 elements, **0
  not bit-identical**.
- **Not `Op::RmsNorm` on `attn_norm`**: real weight, real embedding rows, eps
  from the file, against an f64 reference — max relative error 2.365e-7
  (256-thread prefill build) / 1.191e-7 (`-DWIDE` decode build).

All three dequant/norm results are guarded by
`crates/infr-vulkan/tests/weight_dequant_parity.rs`, and each assertion was
shown to go red before being trusted.

#### Considered and declined

**Landing the f16 activation route for `cfg.deepseek2`.** It is nearly free
(1.0% of pp512, measured above, vs 27.5% for f32) and its matmul really is more
accurate than int8. Declined: it did not fix the reported bug (the residual
did), its CPU-agreement gain was not even monotone across prompt lengths (worse
than int8 at len 16 and 32, better at 56), and upstream ships int8 on this exact
path. The lever already exists as `INFR_NO_MMQ` if a later slice wants to A/B it
again — nothing needs to be built.

#### Coverage gaps, stated plainly

- `weight_dequant_parity` covers Q4_K only. Every other format's `dqblk` arm in
  `native_decode.glsl` is still compared only against another Vulkan build
  (`weight_addr_parity`), never against the host.
- The f32-activation row above also drops coopmat from the ATTENTION kernels
  (`INFR_NO_COOPMAT` disables the device feature wholesale), so it is not a pure
  activation-precision A/B. Its pp512 matching the earlier `exact_linear_acts`
  measurement to within a point is what makes it trustworthy as one anyway.
- The Metal MLA path was not looked at at all, and the residual fix was verified
  on CPU and Vulkan only. Metal shares the same `seam/runner.rs` emit, so it
  gets the fix by construction, but nothing was RUN there.
- Whether V3/V3.2/V4 were affected is **untested**. They share the `MixerW::Mla`
  arm, so they were, but the 671B V3.2 has no llama.cpp oracle on this box (it
  cannot load there) and `cpu_deepseek_v32_golden` was NOT re-blessed by this
  slice — it is `#[ignore]`d, costs minutes per token through the host pager,
  and will be stale until someone runs it.

### B-DSHW-FUSED-REF — upstream's fused DeepSeek kernels exist on CUDA and SYCL only (2026-08-11)

**Tag:** deepseek · reference · **Blocked on:** nothing; this is a design
reference, not a defect

Enumerated by listing each backend's source directory at the new pin: fused
DeepSeek kernels exist as `lightning-indexer` and `dsv4-hc` under
`ggml/src/ggml-cuda/` and `ggml/src/ggml-sycl/` (the SYCL pair is **new in this
pull**), and **nowhere else** — no Vulkan, no Metal, no CPU equivalent, and the
Vulkan shader directory has no indexer or hyper-connection shader at all.

Useful specifics for anyone porting, read directly from
`ggml_cuda_lightning_indexer_supported`: the fused path is refused unless the
indexer head dim is exactly 128 and the head count is 64 or 32, every non-quant
`q`/`k` stride is 16-byte aligned, and `k`'s type is one of F32, BF16, F16,
Q8_0, Q5_1, Q5_0, Q4_1, Q4_0. `dsv4-hc` exposes three entry points —
`ggml_cuda_op_dsv4_hc_comb`, `_pre`, `_post` — matching the
`build_hc_pre`/`build_hc_post` decomposition `infr` already mirrors.

Worth noting `infr`'s `Op::LightningIndexer` is architecturally **ahead** of
upstream on both backends that have it: its doc records that llama.cpp expands
the selected indices back into a `-inf` mask and runs dense attention,
"realising none of the FLOP saving", whereas `infr` emits the `top_k` indices
and leaves the gather open to the consumer. Nothing here suggests changing that.

### B-DSHW-PULL — the llama.cpp reference moved 139 commits; what was re-verified and what was not (2026-08-11)

**Tag:** deepseek · reference · **Blocked on:** nothing

`~/Projects/mxaddict/llama.cpp` was pulled from `b10218-1-gc629da5`
(`c629da565c80b0b17fac6262acdca4d772e745d8`) to `b10356-2-g030ebb5`
(`030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`) — 139 commits, and all four files
`docs/deepseek.md` names as the source of its maths changed.
`docs/deepseek.md`'s pin line was updated to the new SHA in the same pass.

**Re-verified unchanged** (each by extracting the block from both revisions and
diffing a non-empty extraction, not by reading a diffstat):

- The trunk **YaRN pre-scale** in `deepseek2.cpp`'s `graph::graph` —
  `attn_factor_org`, `mscale`, `kq_scale` — byte-identical.
- The **lightning indexer** in `deepseek32.cpp` — every line of the indexer body
  is unchanged; the file's additions are all MTP.
- The **hyper-connection helpers** in `deepseek4.cpp` — `build_hc_pre`,
  `build_hc_post`, `build_hc_head` — 201 lines compared, identical.
- **`build_moe_ffn`'s router bias** in `llama-graph.cpp` — the bias is still
  added to `probs` (the gated/sigmoid output) and not to `logits`, which is the
  semantics `fa6a7a8` fixed on the Vulkan side.

**Changed, and worth knowing:**

- `build_moe_ffn` changed by **exactly one line**: the asymmetric-gate SwiGLU
  clamp arm, previously `arch == LLM_ARCH_DEEPSEEK4`, now also fires for
  `LLM_ARCH_DFLASH` when `dsv4_hc_mult > 0`. The clamp maths itself
  (`ggml_clamp(gate, -INFINITY, limit)` with a symmetric `[-limit, limit]` on
  `up`) is untouched.
- `deepseek4.cpp` now reads `swiglu_clamp_exp` / `swiglu_clamp_shexp` over
  `n_layer_all` rather than `n_layer()` — an array **length** change to cover
  MTP layers, not a change to the clamp.
- `deepseek32.cpp`'s layer-count→model-type switch moved from `case 62` to
  `case 61`, consistent with it now switching on `hparams.n_layer()` (trunk
  only) rather than the all-layers count.
- **A second copy of the YaRN pre-scale now exists** in `deepseek2.cpp`'s new
  `graph_mtp`, and it divides by `n_embd_head_k_mla` where the trunk copy
  divides by `n_embd_head_k`. Flagged because `docs/deepseek.md` treats the YaRN
  block as having one home; anyone re-deriving it from upstream will now find
  two, and should read the trunk one.
- MTP/NextN support landed for **V3.2 and V4** (`graph_mtp` in both
  `deepseek2.cpp` and `deepseek32.cpp`, plus `load_mtp`/`TENSOR_SKIP` loader
  plumbing). `infr`'s MTP work may want to track it; not assessed here.

**Not checked:** everything else in the 139 commits. The Vulkan backend moved
only three commits in that range (a submission-batching fix,
`GATED_LINEAR_ATTN`, and a `topk_moe` fusion extension), so the
capability-probing findings elsewhere in this group are current; no other
subsystem was diffed.

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

### B62 — is host-RAM KV spill actually the LAST rung? (2026-08-12)

**Tag:** vram · kv · **Blocked on:** nothing; this is a trace someone has to do

User, 2026-08-12: _"Only spill K/V cache to system ram when absolutely
necessary."_

The surface exists — `crates/infr-core/src/config/manifest.rs` carries
`INFR_KV_OVERFLOW` → `kv.overflow`, plus `kv.overflow_vram_mb` and
`kv.overflow_reserve_mb` — and it is documented as VRAM-first. What has **never
been established** is whether the rung ORDER is enforced in code: that q8_0 KV
quantization and context clamping are both attempted, and exhausted, before a
single KV row is placed in host memory.

The intended ladder, from `vram-audit-2026-07-12`: SWA → q8 → clamp → stream →
KV-overflow **last**.

**The task is to trace it and then either prove the order with a test or fix
it** — and to say plainly which of the two it turned out to be. A test is the
point: an ordering that holds today because of how two independent branches
happen to be arranged is one refactor away from silently spilling first, and
spilling is invisible in output. It only shows up as throughput.

Related: B12's DECIDED note makes q8_0 mandatory on the explicit `--ctx` path,
which is the rung immediately above this one.

### B63 — deep-context performance is unmeasured across the whole model matrix (2026-08-12)

**Tag:** perf · coverage-gap · **Blocked on:** nothing

Every performance number this repo records is short-context: pp512 / tg128, or
pp2048 at most (`perf-state-2026-07-20`, the cross-model sweep in
`bench-profile-commands`). **Nothing has ever been measured at agentic-workload
depth.**

User, 2026-08-12: _"benchmark deep context for some models and optimize deep
context, i.e. 35k context prefill + tg512 at depth 35k, so we can see which
parts of infr are slow at those deeper context … I want to be able to run Qwen
3.6 27b locally with larger context for agentic workflow"_, then: _"We should
probably have the same perf sweep for all our supported models."_

**The measurement**, per model: `pp35000`, `tg512 @ depth 35000`, and an
`INFR_PROF_OPS` per-op profile **at depth** — the profile is the deliverable,
not the throughput, because the question is WHICH parts degrade. Score against
llama.cpp's Vulkan backend on the same GGUF wherever it can load it. One size
per family that fits 24 GiB, and always name the quant.

**The matrix** is README's own supported-model table (that table is the
authority; do not re-derive it): `llama`, `llama4`, `qwen2`, `qwen3`,
`qwen3moe`, `gemma3`, `gemma4` (dense / E2B / MoE), `qwen35`, `qwen35moe`,
`diffusion-gemma`, `deepseek`, `deepseek2`/`deepseek32`, and the ternary
`bitnet` / Ternary-Bonsai models.

Two cells are worth more than the rest:

- **SWA models (`gemma3`, and anything with a sliding window) should be FLAT in
  depth by construction** — the window caps the key range, so cost per token
  must not grow with context. If they are not flat, that is a BUG, not a perf
  item, and it is the single most informative measurement in the sweep.
- **`qwen35` / `qwen35moe` are hybrid**: DeltaNet layers are linear in sequence
  length, the attention layers are not, so their depth curve should bend
  differently from every dense model. Nobody has measured it — and this is the
  family the user's Qwen3.6-27B target rides.

**Precedent that this will find something.** DeepSeek V2-Lite decode collapsed
111 → 6.3 t/s between short context and `-d 2048` (34× off llama.cpp), and one
kernel's inner loop was the entire cause — see `B-DSHW-MLA-SCALAR-TIER`, fixed
in fa9dd2c to 11.9 t/s. An identical collapse in any Qwen or Gemma model would
be invisible today, because no measurement in this repo would show it.

Methodology, learned the hard way this session: benches run ONE AT A TIME (a
concurrent GPU process skews everything), arms alternate across reps, and the
spread is reported rather than just the mean.

### B64 — the i32 gathers trust their ids, and robust buffer access is not requested (2026-08-12)

**Tag:** vulkan · soundness · **Blocked on:** nothing; small

`gather_i32.comp` indexes `table[uint(ids[r]) * ne + j]` and `embed_gather.comp`
does the equivalent, neither validating that `ids[r]` is inside the table. Both
document the contract in a comment — the ids are the model's own token ids, and
`generate_dense_backend` refuses a `ffn_gate_tid2eid` whose row count is not the
vocabulary — but nothing enforces it at the boundary the shader reads.

`grep -rn 'robust' crates/infr-vulkan/src/` finds nothing, so
`robustBufferAccess` is not requested and an out-of-range id is an undefined
read rather than a clamped or zeroed one.

This is **pre-existing, not introduced** by `Op::GatherI32` — `embed_gather` has
carried it since it was written, and no path currently produces an out-of-vocab
id. Worth closing anyway, per CLAUDE.md's "enforce a contract, don't document
one": either request `robustBufferAccess` on the device, or clamp in-shader, or
validate the id range host-side where the ids are uploaded. Say which, and why
the other two were not chosen.
