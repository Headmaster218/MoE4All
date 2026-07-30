# perf.md — the performance optimization playbook

How to make infr faster, written for an agent (or human) picking up the perf
campaign. The reference target is llama.cpp on the same GPU: every number we
care about is a **ratio** (`infr t/s ÷ llama.cpp t/s`) on matched flags. The
campaign goal is ≥1.0x on every supported model × quant, prefill and decode,
without ever sacrificing correctness.

## Hardware contention — exclusive device access (READ FIRST)

The single most important operational rule, above the perf loop itself:

> **One process/agent at a time may use the CPU or GPU.** Any work that touches
> a compute device — a build, a test run, a benchmark, an `infr run`/`serve`, a
> profile — holds that device **exclusively for its whole duration**. Never run
> two such tasks concurrently, whether you are a human, an agent, or an
> orchestrator fanning out subagents. The device is a single mutex; take it,
> finish, release, then start the next.

Why it is non-negotiable:

- **Benchmarks are only valid in isolation.** Concurrent device work skews both
  sides of every ratio — thermally (a chip warmed by a neighbouring job reads
  2-8% low) and through contention (shared memory bandwidth, SM/CU occupancy). A
  "regression" measured next to other device work is an artifact, not a finding.
- **The GPU can wedge.** On UMA/integrated parts two big models running at once
  exhaust GTT and wedge the device; a job killed mid-submit leaves an unkillable
  D-state task holding the GPU until reboot (see `docs/igpu.md`). Serial,
  patient device use is also a stability requirement, not only a measurement
  one.

The rules that follow from it:

1. **Serialize ALL device work.** Benches, profiles, `infr run`/`serve`, GPU/CPU
   test suites, and the release builds that feed them — one at a time. This is
   broader than "one benchmark at a time": a compile racing a bench, or two test
   runs, contend just as badly.
2. **Perf/bench slices are strictly serial — never two at once.** Including the
   A and B of an A/B variant comparison. (This is the loop's "do not do two
   slices at once", made explicit.)
3. **Coding/edit agents may parallelize (max 2); device work may not.** You may
   fan out subagents that only read or edit source. But the moment a task
   compiles, tests, benches, profiles, or runs a model it must be the ONLY one
   doing so. Delegated **perf agents run ONE at a time**; delegated coding
   agents **at most two**, and neither may bench/run a model concurrently with
   the other.
4. **Never background a device job.** No background `cargo test` / `cargo bench`
   / `infr run` — run it in the foreground with a generous `timeout` so it
   cannot outlive its turn or overlap the next task. (Backgrounding non-device
   work is fine.)
5. **Never `timeout`/SIGTERM a GPU job mid-submit.** On some devices (integrated
   AMD especially) a kill during a submit leaves an unkillable D-state task in
   `dma_fence_wait` holding the GPU, dropped from enumeration until reboot
   (`docs/igpu.md`). Give device runs a generous `timeout` and never wrap them
   in a tool call that can time out first.
6. **Cool down between manual bench runs.** 60s+ between back-to-back
   `infr bench` A/B runs so the GPU returns to idle temperature — a hot GPU
   depresses the next run 2-8%, enough to reverse a winning variant.
   (`infr compare --sweep` serializes internally, but a multi-model sweep still
   heats the chip; re-probe a flagged row SOLO on an idle device before calling
   a regression — see "Archiving sweeps".)
7. **Validate correctness first, then bench** — parity + goldens before any
   timing (also step 5 of the loop). A correctness run is device work too:
   serialize it like everything else.

## The loop

Every perf slice follows the same shape. Do not skip steps, and do not do two
slices at once.

1. **Sweep** — measure the full model×quant matrix against llama.cpp and rank by
   gap (`infr compare --sweep`, below). Old numbers go stale fast: any landed
   slice can shift every ratio, so re-measure before choosing a target.
2. **Pick the biggest win** — not the most interesting one. Weigh
   `gap × how much anyone runs that config`. A model class at 0.5x beats a model
   at 0.9x. A whole quant family on a slow path beats one shape.
3. **Profile before designing** — `INFR_PROF_OPS=1` per-op device time. Form a
   hypothesis about the bottleneck _class_ (taxonomy below) and verify it with a
   micro-benchmark before writing the fix. Two campaign lessons: a "kernel math"
   hypothesis turned out to be occupancy (LDS budget), and a "GEMM" hypothesis
   turned out to be dispatch-launch overhead. Measurement beat intuition both
   times.
4. **Fix one lever** — the smallest change that addresses the measured
   bottleneck.
5. **Validate correctness FIRST** — parity suites + goldens (below). Only then
   benchmark.
6. **Benchmark serially** — one GPU job at a time, the changed config _and_ the
   configs that share the touched code path (a gate change for one model can
   silently reroute every other model).
7. **Commit with the numbers** — before/after t/s and ratio in the commit
   message. Update the perf log in project memory/docs.

## Model sourcing — Hugging Face cache + `infr pull`

Models live in the Hugging Face cache (`~/.cache/huggingface/hub/`), which
`infr` reads directly — no need to symlink or copy. Two ways to get a model
there:

```bash
# Pull from huggingface.co/org/repo (picks the most-used quant):
infr pull unsloth/Qwen3-0.6B-GGUF

# Or place a .gguf anywhere in the cache manually:
# ~/.cache/huggingface/hub/models--unsloth--<name>/snapshots/<hash>/model.gguf
```

`infr` and `llama-bench` resolve model paths normally, so a symlink in a
per-campaign directory or the path straight from the cache both work:

```bash
infr bench ~/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/<hash>/Qwen3-0.6B-Q4_K_M.gguf -p 512 -n 0 -r 3
```

The `scripts/perf-sweep.sh` accepts any path — symlink a set of models into a
single directory to keep the repeated sweep list short.

## Benchmarking & comparing

The README "Benchmarking & profiling" section has the full command reference.
The short version:

```bash
# infr and llama-bench take the same -p/-n/-d/-r flags:
infr bench "$M" -p 512 -n 0 -r 3          # pp512 (prefill throughput)
infr bench "$M" -p 0 -n 128 -r 3          # tg128 (decode throughput)
llama-bench -m model.gguf -p 512 -n 0 -fa 1 -r 3

# The whole matrix, ranked worst-gap-first:
infr compare --sweep <models...> --sweep-depth 4096
```

Rules that exist because we got burned:

- **One benchmark at a time.** Concurrent GPU work skews both sides.
- **Cool-down between manual runs.** When running multiple `infr bench` commands
  back to back (e.g. A/B testing kernel variants), insert a 60s+ sleep between
  runs so the GPU cools to idle temperature. A hot GPU from a prior run can
  depress the next run's numbers by 2-8% — enough to reverse a winning variant.
  This applies to manual bench calls; `infr compare --sweep` already serializes
  internally but the multi-model chip heating is its own problem (see Archiving
  sweeps below).
- **r=2 is noisy.** A "regression" that appears at r=2 may be variance —
  re-measure with r=3+ and compare against the historical range before reacting.
  Never auto-revert; diagnose first.
- **Fixed-count semantics.** `infr bench` sets `INFR_IGNORE_EOS` so decode
  benches generate exactly n tokens like llama-bench does. If a number looks
  impossibly good, check whether generation stopped early.
- **Rebuild freshness.** If a build tool reports "Finished in 0.06s" after a
  real edit, the output may be cached — force a real build before benching.
- Prefill and decode are separate machines. A prefill change must leave tg
  bit-for-bit alone (verify tg128 before/after); a decode change must leave pp
  alone.
- **DiffusionGemma sweeps differently.** `arch=diffusion-gemma` has no upstream
  `llama-bench` support, so its sweep row comes from the reference fork's
  `llama-diffusion-cli` instead (see the README's Compare section and
  `docs/diffusion-gemma.md`) and prints `dg-step`/`dg-e2e` instead of
  pp512/tg128/tg64@d/mtp128. Only `dg-step` (in-step-parallel throughput) feeds
  the ranked "BIGGEST GAPS" summary — `dg-e2e` is informational because the two
  implementations' entropy-bound samplers run different step counts for the same
  `-n`, so raw end-to-end tok/s isn't a fair ratio.

## Archiving sweeps

`scripts/perf-sweep.sh <models...>` (with `INFR_METAL=1` on macOS) runs the
sweep and archives the matrix under `target/perf/<utc>-<sha>.txt` — keep the
model list fixed so every commit's ratios are a diff away, and paste the matrix
into the PR when a slice lands.

**Sweep rows are not regression evidence.** A multi-model sweep heats the chip;
rows measured mid-sweep run a few percent low (observed: a 0.93× row reading
0.83× in-sweep, back at 0.91× solo minutes later). Before calling a regression,
re-probe the flagged row SOLO on an idle machine and compare absolute numbers,
not just ratios (llama.cpp's side drifts with the same heat).

## Profiling

**One knob, every backend.** `INFR_PROF_OPS=1` turns on per-op profiling on
Vulkan, ROCm, Metal and the CPU backend alike:

```bash
INFR_PROF_OPS=1    infr bench "$M" -p 512 -n 0 -r 1  # per-op device time, any --dev
INFR_PROF_STAGES=1 infr bench ...                    # host-side stage timing
```

Every `prof.*` knob is settable three ways — `INFR_PROF_*` in the environment,
`--set prof.<name>=<value>` on any command, or a `[prof]` section in
`infr.toml`. There are eight, all prefixed `INFR_PROF_`:

| knob                | what it does                                               |
| ------------------- | ---------------------------------------------------------- |
| `ops`               | per-op device profile (the main one)                       |
| `op_shapes`         | itemize labels by shape instead of folding a kind (Vulkan) |
| `stages`            | host-side stage timing (see below)                         |
| `vram`              | live VRAM in use after weight load                         |
| `out`               | also write the exit report as JSON to this path            |
| `diffusion_trace`   | per-step schedule/entropy trace for diffusion models       |
| `metal_device_time` | `off`/`flush`/`counters` — how Metal gets device time      |
| `metal_debug`       | extra Metal profiler output                                |

`stages` is one switch covering every pipeline: prompt/decode throughput, the
decode setup-vs-execute split, the per-chunk prefill build/compile/execute
split, MTP verify time, diffusion per-step phases. Each section prints only if
its pipeline ran. That used to be five separate knobs (`INFR_PROF`,
`INFR_PROF_DEC`, `INFR_PROF_PF`, `INFR_MTP_TIME`, `INFR_DIFFUSION_TIME`), so
"where does the wall time go" meant discovering four more switches after you
found the first.

Every backend prints the same table, tagged with its name, sorted by total time:

```
[prof:rocm]       4.91   27.3%       14     350.7  Linear m=128 3072x1024 Q6K
[prof:rocm]       2.25   12.5%       28      80.2  Linear m=128 1024x6144 Q4K
[prof:rocm]       1.14    6.4%       28      40.8  Attention rows=128 kv=128 h=16/8 d=128 causal
```

and every backend also folds its rows into ONE process-exit aggregate over the
whole run (`== per-op device report: …`), which is the same report the
`INFR_PROFILE=1` host function table prints into and the same one
`INFR_PROF_OUT` writes as JSON. So the host section tells you _that_ the GPU
path waits in `replay_n`; the device section tells you which ops it waited on —
on any backend, now, rather than only on Vulkan.

This was four separate tools until the U1–U3 unification, and the split cost
real work: `INFR_PROF_OPS=1` on a GPU run profiled **nothing** (only the CPU
backend read it — Vulkan answered to `INFR_PROF2`), ROCm's equivalent had no env
var at all and was reachable only through `--set`, and ROCm's and Metal's tables
silently included the bench's untimed warmup forward. **If you are reading a
profile quoted in a commit message from before that, its shares are diluted.**
The seam is `crates/infr-core/src/prof.rs`; a source tripwire fails the build if
a backend reaches past it.

**Labels.** `op kind + the shape that makes the cost`, so two dispatches share a
row exactly when they do the same work — a 28-layer model folds to a handful of
rows and `us/ea` means one shape's cost, not an average over shapes that differ
by 100x. `Linear` carries `m` (which separates decode's GEMV from prefill's GEMM
on the same weight), `Attention` carries `kv_len` (the whole at-depth story),
and `MoeFfn` carries its expert geometry. Everything else is the bare op kind.

**Units are not always device time**, and the table says which it is. Metal's
always-on mode measures host _encode_ wall (`[prof:metal] … host-encode`),
because its ops batch into one command buffer and nothing finer is free — a
large encode share is itself the finding. Only `Unit::Device` rows feed the exit
aggregate, so a host-encode number can never be summed against a device one by
accident.

### Backend-specific notes

- **Vulkan** labels by **kernel name**, not op kind — its walk is a recorder
  that lowers one op into several dispatches, so the kernel is the finer, truer
  unit there. Labels are automatic: the `be.kernel("name", …)` cache key travels
  on `ComputeKernel`, so a new kernel appears in the report by existing. Buffer
  copies and fills stamp as `copy_buffer` / `zero_fill`.
  `Recorder::label_next("…")` overrides one dispatch whose name is ambiguous
  (`expert_gateup` vs `expert_down` share one MMQ kernel; the
  `lin_vocab_out`/`lin_vocab_in` lm_head-vs-projection split).
  `INFR_PROF_OP_SHAPES=1` swaps GEMV/GEMM labels for shape-itemized ones
  (`mmvr:m4:1536x24576`) — Vulkan's equivalent of the shared itemizer. **Trust
  the percentages and per-op relative times; the absolute µs totals can
  overflow.** Decode's record-once replay tape cannot carry per-replay
  timestamps (queries are baked at record time), so the replayed decode path
  reports nothing — set `INFR_SEAM_NO_REPLAY=1` to force the re-recorded path
  when profiling decode.
- **ROCm** brackets each op in a HIP timing-event pair on its single stream. A
  host `Instant` would measure enqueue (the ops queue with no per-op sync), and
  a `hipStreamSynchronize` per op would measure execution but serialize the
  pipeline and add F4's ~2.7 µs launch floor to every sample.

  **Its overhead scales with DISPATCH COUNT, and on decode that is not small.**
  Measured on an RX 7900 XTX: **0.3%** on Qwen3-30B-A3B `pp512` (48 profiled ops
  in the forward — 347.2/346.9 profiled vs 347.9/348.0 clean) but **37%** on
  Qwen3-0.6B `tg128` (~310 dispatches per token, so ~79 000 event pairs over the
  run — 134.5 profiled vs 212.2 clean). The cause is that the collector creates
  and destroys an event pair per span rather than pooling them.

  So: **use it to RANK ops, never to size a total against wall time on a
  dispatch-heavy workload**, and never quote a profiled t/s. If you need the
  size of one op on decode, price it by skipping the op and taking the
  difference in clean runs — that is how P6 established attention was 50% of a
  d0 token and 79% at d4096. Pooling the events would fix this and has not been
  done.

- **Metal** needs a second knob for DEVICE time, because batching means it is
  never free there: `--set prof.metal_device_time=counters` samples
  stage-boundary timestamps — **the only honest per-op device mode** — and
  disables the replay tape for the run. `=flush` flushes after each op instead,
  which costs the batching and, because the tape re-executes decode tokens
  without walking ops, silently attributes a sample of one token. Left `off`
  (the default), `INFR_PROF_OPS` still gives you host encode time. Sweep with
  `--dev MTL0` (llama-bench rejects `Vulkan0` on a Metal box).
- **CPU** measures host wall per op, which for a host backend _is_ execution
  time, so its rows are `Unit::Device` and compare directly against a GPU's.

- `infr bench` excludes its untimed warmup + depth-warm turns from the profile
  on **every** backend (they run suppressed and the exit aggregate is reset
  after warmup), so the report covers exactly the timed reps. A profile whose
  `Linear` rows include an `m=1` or `m=8` shape you did not ask for is warmup
  leaking in.
- Sum GEMM-ish labels vs elementwise labels vs attention. Then look at **op
  counts**: 50k dispatches per chunk is a finding in itself, regardless of what
  any one op costs.
- Micro-probes: `crates/infr-vulkan/tests/` (`gemm_bench`, `decode_gemv_bw`,
  `small_m_bench`, `bandwidth_probe`), `crates/infr-metal/tests/` (`gemv_bw`,
  `dispatch_overhead`, `moe_bw`), `crates/infr-rocm/examples/decode_gemv_bw.rs`.
  Chained-dispatch numbers there are thermally unstable across back-to-back
  runs; trust e2e benches for accept / revert calls.
- For a suspect GEMM shape, add it to the per-shape micro-bench
  (`crates/infr-vulkan/tests/gemm_bench.rs`) and print µs + TFLOPS. Compare
  against the kernel's known ceiling — a shape far below ceiling is a shape
  problem (grid/occupancy), the ceiling itself being too low is a kernel
  micro-arch problem.

### Build-time auto-instrumentation (`INFR_PROFILE=1`)

Whole-workspace CPU-side function timing with **zero code in default builds**
and **zero manual timer edits**:

```bash
INFR_PROFILE=1 cargo build --release -p infr-cli   # instrumented binary
./target/release/infr bench "$M" -p 0 -n 128 -r 3  # report prints at exit
INFR_PROF_OUT=prof.json ./target/release/infr ... # + full table as JSON
```

At process exit the run prints a merged, self-time-sorted table:

```
== INFR_PROFILE report: 200 sites, 1 threads, wall 1.96s (...), accounted self 1.92s ==
        self   self%        total        calls  avg(self)  function
     624.2ms  31.91%      624.2ms           49     12.7ms  infr_vulkan::recorder::RecordedCmd::replay_n
     581.0ms  29.70%      670.0ms            1    581.0ms  infr_gguf::dequant::dequant_unified
       7.7ms   0.39%      728.3ms            4      1.9ms  infr_llama::seam::runner::generate_dense_backend
```

`self` excludes time spent in instrumented callees (`total` is inclusive;
recursion counts inclusive time once at the outermost frame). Accumulation is
thread-local (no shared state on the hot path); per-thread tables merge at
report time, so rayon/spin-pool threads are covered.

**How it works.** Each hot crate's `build.rs` turns `INFR_PROFILE=1` into
`cfg(infr_profile)`; every top-level `fn` and `impl` block in infr-core,
infr-cpu, infr-vulkan, infr-gguf and infr-llama carries
`#[cfg_attr(infr_profile, infr_prof::instrument)]`. Without the env the
attribute is compiled away entirely — the default binary contains no profiling
code, no runtime branch, nothing. With it, the `infr-prof` proc macro rewrites
every function in the item to open an RAII span (`infr-prof-rt`) that survives
`?`, `return` and panics.

**Coverage rules — what a new function needs:**

- new method in an already-annotated `impl` block: **nothing**, covered
  automatically.
- new top-level `fn` or new `impl` block: one line —
  `#[cfg_attr(infr_profile, infr_prof::instrument)]`.
- `#[inline]` / `#[inline(always)]` fns are **auto-skipped**: declaring a fn
  inline already asserts it is smaller than the ~50ns probe pair. Non-inline
  sub-100ns leaves opt out with `#[cfg_attr(infr_profile, infr_prof::skip)]`
  (see `infr_gguf::dequant::k4` — at 1.7e9 calls/run, probing such leaves made
  CPU decode 12x slower and drowned the report).
- `const fn`, `async fn`, `#[naked]`, `#[test]` fns and closures are skipped
  automatically; generics share one site across instantiations; trait impls
  report as `<Ty as Trait>::fn`.

**Overhead (Qwen3-0.6B Q4_K_M, 7900 XTX + CPU backend):** GPU pp512 26.0k →
25.2k t/s (~3%), GPU tg128 within noise (618 vs 616), CPU pp128 within noise
(933 vs 943), CPU tg32 118 → 105 t/s (~11%, decode's per-row kernel calls are
~200ns each so the probe shows). Instrumented builds are for attribution, not
for ratio benchmarking — never quote instrumented numbers in a perf log.

**Design notes (rejected alternatives).** (B) a build.rs syn-rewrite of crate
sources into OUT_DIR + `include!` was rejected: non-inline `mod foo;` resolution
breaks under `include!`, every nested module needs path rewriting, and
rust-analyzer/incremental builds see phantom sources — fragility for the sole
benefit of removing one attribute line per item. (C) rustc-level instrumentation
(`-Z instrument-xray`, mcount-style hooks) is nightly-only and emits to external
tooling with no self/child or per-thread story on stable — samply already covers
the "no source markers at all" niche. (A proc-macro attribute on non-inline
`mod foo;` is also unstable, which is why annotation is per-item, not per-file.)

**Deprecation rule:** do not add new one-off `Instant::now()` accumulators or
config-gated eprintln timers (`prof.prof_dec` / `INFR_PROF_STAGES`-style) —
build with `INFR_PROFILE=1` instead. The existing `prof.*` knobs stay until this
system has proven itself in a few campaigns, then get removed. (`INFR_PROFILE`
itself is a BUILD-time input read by `build.rs`, which is why it is one of the
few `INFR_*` keys that is deliberately not a `Config` field.) Device-side timing
is the same idea at each backend's dispatch chokepoint: `INFR_PROF_OPS` per-op
timing is auto-labeled (no manual stamp calls — see the Profiling section
above), but stays **runtime**-gated, not build-gated: the instrumentation costs
nothing when off, and toggling profiling without a rebuild is worth keeping. The
two compose: run an `INFR_PROFILE=1` build with `INFR_PROF_OPS=1` and the exit
report prints the host function table AND the per-op device aggregate in one
output (`INFR_PROF_OUT` JSON carries both as `"sites"` + `"gpu"`) — the host
section tells you _that_ the GPU path waits in `replay_n`; the device section
tells you which ops it spent that time on.

**And do not add a fifth per-op profiler.** A new backend gets the whole thing —
knob, label grammar, table, exit aggregate, warmup suppression — by calling
`infr_core::prof::enabled` and handing durations to `OpProf`. The only piece it
writes itself is how it obtains a duration from its own API, because that part
genuinely differs (timestamp queries / HIP events / counter samples / host
clock). `no_backend_feeds_the_aggregate_behind_the_shared_reporter` fails the
build if it reaches past the seam.

### CPU profiling (samply)

For CPU-side attribution, use a sampling profiler — **do not add ad-hoc
`Instant::now()` timers to the code**. Hand-rolled timing gets added, skews what
it measures (the eprintln/sync overhead is visible at this scale: a denoise
`exec` read 6.0s with `INFR_PROF_OPS=1` vs 3.85s without), and then has to be
reverted. A profile answers the same question per-function, per-line, with zero
code changes, and keeps stack context the timers throw away.

Tooling: [`samply`](https://github.com/mstange/samply) (`cargo install samply`),
no root needed. `[profile.release] debug = "line-tables-only"` in the workspace
`Cargo.toml` gives it symbols + line numbers; debug info never affects codegen,
so release numbers and the bit-exact goldens are unchanged.

```bash
# Interactive: records, then opens the Firefox Profiler UI (flame graph,
# per-thread timelines, inverted call tree = self-time ranking).
samply record ./target/release/infr bench "$M" --ngl 0 -p 16 -n 32 -r 1

# Headless (agents, SSH): save the profile, then rank self-time per function.
samply record --save-only -o prof.json.gz -- ./target/release/infr bench ...
scripts/samply-top.py prof.json.gz 30
```

Reading `samply-top.py` output: percentages are of **total thread-seconds**
across all threads, not wall time — on 32 threads, a function at 3% of
thread-time can still be the top wall-time lever if it's serial, and rayon
plumbing frames (`accum.rs`, `crossbeam_epoch`, `Producer::fold_with`) showing
up high is itself a finding (fork-join overhead / idle spinning, the
"whole-graph threadpool" class). For per-line attribution inside one hot
function, open the same `prof.json.gz` in the UI (`samply load prof.json.gz`).

What samply does NOT give you: hardware counters (DRAM bandwidth, IPC, cache
misses). When the taxonomy question is "bandwidth-bound or compute-bound?", use
`perf stat` (`sudo pacman -S perf`) — reason about bytes-moved first though; a
back-of-envelope traffic count (table size × times streamed) has called it
correctly every time so far.

The env-gated profiles (`INFR_PROF_OPS=1` per-op totals + the per-stage MoE
breakdown, `INFR_PROF_STAGES=1` per-denoise-step phases) stay useful as cheap
first reads — they're already in-tree, run everywhere, and cost nothing when
off. The rule is only: don't add NEW timer instrumentation when a profile would
answer the question.

## Bottleneck taxonomy — what the profile means

Classify before you fix. Each class has a different lever, and the campaign has
a worked example of each:

**1. Wrong kernel entirely (coverage/gate bugs).** A fast path exists but a
routing gate excludes the shape or dtype, so it falls to a slow fallback (scalar
GEMV, dp4a mmq, the 64×64 tile). Cheapest wins in the codebase — one-line gate
fixes. Example: gemma3-1b's ne=1152 projections sat on scalar mmq at ~10 TF
because the warp-GEMM gate still required `n % 256 == 0` after a `n % 128` tile
had been added. Audit gates whenever a new kernel variant lands: _every_
eligibility check upstream of it is now potentially stale.

**2. Grid underfill.** The kernel is fine but the dispatch doesn't fill the GPU
(narrow n, small m → fewer workgroups than compute units). Levers: narrower
tiles (more workgroups for the same work), split-K (parallelize the reduction
dimension — keep the reduce fixed-order for determinism), fusing sibling GEMMs
into one wide dispatch (Q/K/V; gate+up).

**3. Occupancy ceiling.** The grid is full but each SM/WGP runs too few
workgroups concurrently to hide latency — usually LDS or register budget.
Diagnose by experiment: if bigger tiles _lose_, you're occupancy-bound, not
math-bound. Lever: shrink per-workgroup resources. Example: dropping the A-tile
from shared memory (pre-cast A to f16 once, `coopMatLoad` straight from global)
cut LDS 25→20 KB, lifted 2→3 workgroups/WGP, and took the 8B GEMMs from ~28 to
~44 TF.

**4. Dispatch/launch overhead.** Total GPU time is dominated by many tiny
dispatches and the barriers between them, not by any op's math. Tell-tales:
elementwise ops costing as much per-op as GEMMs; op counts in the tens of
thousands; small-m chunks taking absurd wall time (fixed cost doesn't scale
down). Levers: **one dispatch per stage over all items** (put the item index on
a grid dimension, early-exit surplus workgroups — empty workgroups are nearly
free), fuse adjacent elementwise stages into one kernel, drop indirect-args
machinery when a fixed worst-case grid + early-exit is simpler. Example: the
batched MoE FFN went from ~1050 dispatches + ~110 barriers per layer (per-expert
waves) to ~13 single dispatches — pp512 0.59x → 0.91x with zero GEMM changes.

**5. CPU-bound / host-in-the-loop.** The GPU is _idle_ while the host does work
that could be a shader: routing decisions, readbacks mid-graph, per-token graph
rebuilds, scheduling that forces a submit-wait-resubmit pattern. Tell-tale: wall
time ≫ summed GPU op time, or `INFR_PROF_STAGES` showing record/host time
rivaling GPU time. **The standing rule: if an op runs on the CPU and can be
expressed as a shader in the pipeline, move it into the pipeline.** Every host
readback is a full pipeline stall (submit + wait + resubmit); every host-side
decision that a kernel could make keeps the GPU starved. Worked examples already
in-tree: GPU-resident MoE routing (top-k, bucket count/scan/ scatter all on-GPU
— the old path downloaded counts mid-graph to size its GEMMs), and record-once
decode replay (the `_dyn` kernels read pos/kv_len from a device-side params
buffer so the host doesn't re-record per token). When you find a host loop
feeding the GPU small submits, the fix is almost always "make the GPU compute
its own control data".

**6. Kernel micro-architecture.** Only after 1–5 are excluded: the kernel runs
at its own ceiling on a full grid and the ceiling is below the competition's.
This is the most expensive class to fix (tile shapes, pipelining, accumulator
precision) and experiments frequently lose — measure every variant on the
per-shape micro-bench before believing it.

Work the classes in that order: they're roughly sorted by effort-per-percent.

## Correctness is non-negotiable

Perf work is only real if the output is unchanged.

- **Parity first, bench second.** The CPU backend is the oracle:
  `cargo test -p infr-llama --release --test cpu_backend` (goldens for every
  model family) plus the infr-vulkan op-parity suites (`-- --ignored` for the
  GPU-gated ones).
- **Never re-bless a golden without proof.** If a hash changes, first capture a
  real generation (`infr run …`) before and after and verify the text; only then
  re-bless, and say so in the commit. Prefer designs that are bit-identical by
  construction: fixed-order reductions instead of atomics, same rounding points,
  same accumulation order. Almost every slice in this campaign (QKV fusion,
  split-K, A_GLOBAL, batched MoE) was engineered to be bit-identical — the
  goldens never moved.
- **Determinism.** Any parallel reduction you introduce must have a fixed
  summation order. Atomics that affect _values_ (not just placement) are out.
- **Zero-init discipline.** `Backend::alloc` is calloc-style; `alloc_uninit`
  only when every element is provably written before read. Padding rows that
  hold garbage are fine only when the garbage provably never feeds a real output
  (GEMM rows are independent; document the argument in a comment).
- **Tests configure behaviour with a `Config`, never with the environment.**
  Build the `Config` you want and pass it (`VulkanBackend::new_with(cfg)`,
  `CpuBackend::new_with(cfg)`, …). Setting an `INFR_*` variable from a test is a
  process-wide write that races every other test in the binary — that was the
  original "flaky golden" — and there is no longer an `EnvGuard` to work around
  it. See [`config.md`](config.md).

## Codebase habits

- Full-workspace clippy
  (`cargo clippy --workspace --all-targets --locked -- -D warnings`) and
  `cargo fmt --all` before every commit — per-crate clippy has missed CI
  failures.
- infr-metal is macOS-gated: Linux builds pass silently when you change a shared
  type (`Op` fields, trait signatures). Patch its sources _and tests_ blind, and
  let macOS CI verify.
- Deleting a shader: delete its build.rs entry, its `_spv` accessor, and its
  recorder method together, then make sure a _clean_ build passes (stale `.spv`
  files in the target dir can mask a missing entry locally).
- New kernel variants via compile-time defines on one `.comp` source beat
  copy-pasted shaders. Recorder binding arrays are hazard-ordered: **inputs
  first, outputs last** — a variant that adds a readonly input must renumber
  bindings to keep outputs at the end.
- Scratch buffers: pool them (`pooled(pool, be_, tag, bytes)`) so every
  same-shape op in a graph shares one allocation.

## Where the time went (campaign log, 2026-07)

For calibration — the kind of yield each class of fix produced on pp512 ratios
vs llama.cpp:

| slice                             | class             | result                                                           |
| --------------------------------- | ----------------- | ---------------------------------------------------------------- |
| fused QKV + narrow tile + split-K | grid underfill    | 0.6B 0.56 → 0.74x                                                |
| A_GLOBAL (LDS → occupancy)        | occupancy         | 0.6B → 0.92x, 8B 0.72 → 0.83x                                    |
| batched MoE (dispatch collapse)   | dispatch overhead | MoE 0.59 → 0.91x                                                 |
| warp-gate n%256 → n%128           | coverage gate     | gemma3-1b 0.60 → 0.67x                                           |
| warp-GEMM Q5_K/Q4_0/Q2_K          | coverage gate     | Q4_0 0.6B 0.43 → 0.85x, Q5_K 0.42 → 0.61x, Q2_K 14B 0.47 → 0.68x |
| IQ4_XS dqblk + warp-GEMM          | coverage gate     | 0.6B pp 0.25 → 0.67x (2.7x), tg 0.39 → 0.68x                     |

Known open items (re-sweep before trusting): the quant cliff — **largely
closed** (Q5_K/Q4_0/Q2_K/IQ4_XS now on the warptile; for the K-quants the dqblk
decoders already existed so it was pure wiring, and IQ4_XS just needed a
`dqblk_iq4xs` written — a 32-elem sub-block is exactly one IQ4_XS sub-block, so
the amortized decode also fixed its slow `native_gemv` decode). Remaining,
biggest-gap-first: **gemma-4-E2B decode is now the worst Q4_K_M combo (tg 0.64x
@d4096, 0.70x tg128)**; qwen35 engine (DeltaNet occupancy, narrow-n split-K);
gemma3-1b's narrow-shape GEMM efficiency (pp 0.66x); the warptile's ~36 TF
ceiling vs llama's ~45-50 on wide shapes (class-6, dense pp tails); and Metal
parity for everything above (most fast paths are Vulkan-only; Metal states its
own capabilities, so it degrades gracefully but slowly).

## Coopmat operand tiers (fp8 / int8 / bf16) — a measured dead end vs f16

The Vulkan backend enumerates which cooperative-matrix component types the
device accepts and reports them in the startup banner
(`f16cm / bf16cm / f8cm / i8cm`). On RDNA4 (RX 9060 XT / Navi 44, Mesa 26.1.4)
all four are present; on RDNA3 (7900 XTX) only f16 + int8. It's tempting to
route quantized-weight GEMMs through the "faster" 8-bit units. **We built and
measured all of them on real hardware — none beat the f16 coopmat GEMM. Do not
re-open without a native-low-bit model.**

- **fp8 (E4M3) coopmat GEMM** — full path built + validated on RDNA4 (per-row
  activation range-scaling into E4M3's ±448, warp-tiled to match
  `native_gemm_warp`, per-row descale). Gated opt-in `INFR_F8_COOPMAT=1` (+
  `INFR_F8_PREPACK=1` for the Q8_0→E4M3 prepack path), default-off. Result:
  - fp8-with-in-shader-dequant = **0.73×** f16 (pays the SAME Q8_0 dequant f16
    pays, then adds an activation-scale pass + a descale epilogue).
  - fp8-with-prepacked-E4M3-weights (dequant removed) ties f16 **exactly**
    per-op (1024×6144 @ m512: f16 **364.3µs** vs fp8 **364.5µs**), loses on
    narrow shapes (no split-K/A_GLOBAL variant). So even fully WMMA-bound, RDNA4
    fp8 gives **no speedup** — these GEMMs measure ~17.6 TF/s
    (latency/occupancy- bound, well under peak), so a faster MAC can't help, and
    the identical wide- shape time argues fp8 WMMA ≈ f16 rate here anyway.
- **int8 coopmat GEMM** — also built (`INFR_I8_COOPMAT=1`, default-off):
  4.8-5.6× slower raw, amortized to ~0.73× after a store-to-shared per-block
  rescale epilogue, still loses. Coopmat v1 has no in-flight per-element scale,
  so block-quant integer matmul pays a rescale tax f16-dequant doesn't. **The
  principled integer path is dp4a `mmq`** (each thread owns its accumulator →
  scale-after for free), which is what infr's Q4_K mmq + llama.cpp both use.
- **coopmat2 (`VK_NV_cooperative_matrix2`) per-element — tested, doesn't rescue
  int8.** Coopmat2's `coopMatPerElementNV` gives (row,col)-indexed in-fragment
  element access → apply the per-block rank-1 descale with NO store-to-shared
  round trip (the "rescale tax" above). Present on RDNA4 RADV behind the driconf
  flag `radv_cooperative_matrix2_nv=true` (Mesa 26.1.4); glslc compiles it
  (`GL_NV_cooperative_matrix2`;
  `void coopMatPerElementNV(out coopmat r, coopmat m, T fn)`). The probe
  `crates/infr-vulkan/examples/coopmat2_test.rs` (per-element vs the v1
  store-to-shared epilogue) measures coopmat2 **SLOWER at both scales**:
  single-tile 2.16 vs 2.24µs (1.04×, noise); **full-GEMM 512×2048×2048 (64
  rescale blocks, 4096 wg) 1239.8 vs 1195.3µs = 0.964× (coopmat2 slower)**. So
  removing the rescale tax doesn't help — it was never the bottleneck
  (store-to-shared is cheap on RDNA4). Example KEPT as a re-test tool; **re-run
  it if a future Mesa stabilizes/optimizes coopmat2** (it's currently
  experimental/driconf-gated). Run:
  `radv_cooperative_matrix2_nv=true cargo run --release -p infr-vulkan --example coopmat2_test`.
- **bf16 coopmat GEMM** — `native_gemm_warp.comp` `-DBF16CM` variant
  (`INFR_BF16_COOPMAT=1`, default-off): the PRODUCTION warptile with
  `bfloat16_t` operands instead of `float16_t` (a `CMTYPE` macro; default build
  byte-identical, verified via .spv md5), reading bf16 weights exactly (no f16
  clamp). Its value is **precision faithfulness** — preserves bf16's exponent
  range instead of clamping to f16 (max 65504). **But it is ~12-27% SLOWER than
  the f16-clamp path, and that gap is the bf16 WMMA itself, not a missing
  optimization.** Per-op profiling (RDNA4 Qwen3-0.6B-BF16, pp512, SAME shape +
  SAME non-A_GLOBAL variant): `native_gemm_warp_bf16` (f16) 769.9µs vs
  `native_gemm_warp_bf16cm` (bf16) 865.1µs @ 1024×6144 (+12%; +27% on narrow).
  Identical kernel/staging/ tiling — only the coopmat operand type differs →
  **RDNA4's `bfloat16_t` coopMatMulAdd runs slower than `float16_t`'s** (likely
  RADV codegen immaturity for the newer bf16-coopmat path — re-check on a future
  Mesa — or a real HW rate gap). So bf16-at-f16-speed is NOT achievable here;
  kept opt-in faithful path, ~12-27% slower. (Retired the standalone
  `native_gemm_bf16cm.comp`.) **Memory-access gotcha (measured on the
  standalone):** reading bf16 weights as a native `bfloat16_t[]` SSBO (16-bit
  loads, "no conversion") ran **~27% SLOWER** than the `dqblk` path that reads
  32-bit words (`uint nw[]`, 2 bf16/word) + ALU bitcast — narrow 16-bit loads
  don't coalesce on RADV. Read packed 16-bit weights as 32-bit words, never as a
  native 16-bit array (same reason the f16 GEMM reads `uint nw[]`).
- **Why no 8-bit operand swap wins:** on Vulkan/AMD, low-bit-float weights have
  no native matmul — you always dequant/convert to the WMMA operand type first,
  and RDNA4's fp8/int8 WMMA doesn't out-rate f16 on inference-shaped GEMMs.
  **ggml- vulkan does the same thing**: it has no fp8/fp4 weight matmul —
  MXFP4/NVFP4 dequant to f16 (`ue4m3_to_fp32` scale × `kvalues_mxfp4` codebook)
  then run the normal f16 coopmat. The only genuine low-bit-float _compute_ win
  is NVIDIA **Blackwell's block-scaled fp4 MMA**
  (`mma…kind::mxf4nvf4.block_scale…e2m1.e2m1 …ue4m3` — fp4 operands, fp8 scale
  applied in the tensor core), which RDNA4 does not expose (no `e2m1` coopmat
  config). fp8/int8 coopmat stay as gated-off measurement paths; the win, if a
  low-bit-float model ever needs it, is a native weight format, not an
  operand-type swap on a dequant-bound GEMM.
