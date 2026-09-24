# Qwen3.8 concurrent serving: throughput, stalls and next steps

## Scope

The first half of this report measures the release binary at `ea1f17e`, following
the single-decode and MTP optimization pass. The follow-up optimization pass
changes concurrent scheduling, per-lane QSA lowering and PLE gather fan-out;
its final release binary is reported separately below.

- Binary SHA-256: `8eb6a509233d44a1e1045e2216f1e1363b2bdc674700a006b09e50c9088f78fb`.
- Ryzen 5 5600X, RX 7900 XTX, approximately 64 GiB visible system RAM.
- Model: `D:\AILMStudioModels\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
- MTP head: `G:\Qwen3.8-Flash-Next-UD-Q2_K_XL\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
- Fixed VRAM/RAM/ubatch/parallel-ubatch: **24 GiB / 48 GiB / 3072 / 256**.
- Requested `--parallel 2`, temperature 0, seed 1, `--no-think`.
- Context capacity 32,768 per slot, plus a 163,840-capacity control. Actual
  prompts range from 507 to 29,688 tokens; capacity is not occupied context.
- Synthetic repeated background records followed by two different technical
  writing tasks. Each request generates 1,024 or 1,536 tokens. Cached prompt
  tokens are zero. Staggered arrivals occur six seconds after A's first text.
- One server process at a time. No competing benchmark, model or compilation.

The runner is `scripts/bench-concurrency.cjs`; analysis is
`scripts/analyze-concurrency.cjs`. Local ignored raw logs, prompts, outputs,
timings, hashes and summaries are in `target/concurrency-20260923/`.

## Implemented optimization pass

Final release SHA-256:
`133eedf7a5ed69c0bb4904178e0d9a49997b13b4ed705a9261f76f9e14ff5d00`.
The model, prompts, budgets and server arguments are unchanged from the baseline.

- Independent PLE batches now partition by disjoint gather groups and all four
  gather workers, rather than capping a two-slot batch at two workers.
- Dense- and sparse-QSA decode lanes remain in one model cohort. Fixed projections,
  recurrent layers and MoE run at `m=2`; each QSA lane retains its own cache,
  selected-block count and dense-prefix identity selection.
- The long-prefill exclusive phase is intentionally unchanged. Decode still waits
  until that prefill completes, preserving the existing ring-buffer and expert
  reconstruction contract.

Unprofiled 1,024-token results, aggregate over the interval where both requests
decode:

| Prompt tokens A / B | Baseline | Optimized | Change | Optimized thirds |
| --- | ---: | ---: | ---: | --- |
| 507 / 508 | 50.19 | 54.16 | +7.9% | 55.02 / 54.43 / 53.03 |
| 8,187 / 8,188 | 50.37 | 52.73 | +4.7% | 53.38 / 51.98 / 52.83 |
| 507 / 28,668 | 34.50 | 54.34 | +57.5% | 55.74 / 52.82 / 54.46 |

The short and medium output hashes match the original non-fused path per lane.
The mixed-QSA case now sustains about 27.2 tok/s per lane instead of alternating
one-row 16-step cohorts. Its earlier request still has the designed 32.6-second
text pause while the later long prefill owns the scheduler.

The 63 tok/s target would require at most 31.75 ms per two-token step. The final
short run takes 36.89 ms, leaving about 5.14 ms/step (14%) still to remove.
Updated pager profiling reduced exposed PLE wait from 1.57 to 0.22-0.38 ms/step,
but the backend still spends about 38-39 ms under profiling, with roughly 107
queue submissions, 12 ms of command recording and 2.2 ms of CPU submit time per
step. That remaining gap is a pager/orchestration problem, not a PLE problem.

Rejected same-binary experiments are useful constraints: folding the shared
expert into paged IQ slots was neutral to negative, disabling hit-first fell to
49.3 tok/s, an independent-row int8 mrow did not improve throughput, `MR=2`
Q4_K/Q6_K builds regressed, and expert grid `NR=4` gained only about 0-1% while
`NR=2` regressed. None of those experiments remain in the production source.

Ordinary token counts come from server `GenerationProgress` and final usage,
not SSE event counts. Overlap thirds divide the interval in which both requests
are decoding into equal wall-time windows; interpolation has approximately
one-second resolution. Request averages include pauses after decode starts.
SSE timestamps are used only for first text and inter-text stalls. MTP segment
counts come from committed-token cycle logs in the separate pager-profile runs.

## Ordinary concurrent decode

All values in the following table are **aggregate tok/s across both requests**,
with profiling disabled. These are overlap-window results, not per-request
lifetime averages or averages including initial prefill.

| Actual prompt tokens A / B | Arrival order | Early third | Middle third | Late third | Whole overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| 507 / 508 | Together | 51.8 | 51.1 | 47.6 | 50.2 |
| 8,187 / 8,188 | Together | 51.7 | 50.1 | 49.3 | 50.4 |
| 28,667 / 28,668 | Together | 52.5 | 49.9 | 51.8 | 51.4 |
| 3,577 / 29,688 | A, then B | 50.5 | 48.2 | 46.3 | 48.3 |
| 29,687 / 3,578 | A, then B | 54.0 | 53.6 | 51.6 | 53.1 |
| 507 / 28,668 | A, then B | 35.3 | 33.7 | 34.5 | 34.5 |

At matched QSA modes the steady per-request rate is roughly 24-27 tok/s. The
last row has different QSA modes and loses the two-row batch. This is not simply
"a longer context always decodes slower": the 28k/28k pair still sustains 51.4.
Different continuations/cache states also prevent interpreting the reverse
order's higher rate as a benefit caused solely by arrival order.

The 163,840-capacity control repeats 3,577 -> 29,688 actual prompt tokens:
48.3 aggregate, with thirds 49.1 / 49.0 / 46.9. The larger capacity did not
reproduce a sustained 17-18 aggregate tok/s in this measured decode interval.
It did expose a longer prefill pause, as detailed below.

### Arrival stalls are much worse than steady decode

| Arrival pattern | A's longest text pause | A's whole-request decode average | B first-text latency |
| --- | ---: | ---: | ---: |
| 3.5k -> 29k, 32k capacity | 33.42 s | 16.95 tok/s | 33.79 s |
| 3.5k -> 29k, 160k capacity | 58.72 s | 13.23 tok/s | 59.04 s |
| 29k -> 3.5k, 32k capacity | 6.28 s | 25.77 tok/s | 6.50 s |
| 0.5k -> 28k, 32k capacity | 33.00 s | 11.93 tok/s | 33.97 s |

After B completes in the 3.5k -> 29k test, A recovers to approximately 33.8-34.1
tok/s alone. Thus the low request-average value is not evidence that its
resumed decode remains at 13-17 tok/s. This does not explain away the user's
original **real-time** 17.5 reading: a sustained aggregate 17.5 with both slots
already decoding was not reproduced in this matrix.

`parallel.rs:1726` runs long prefill as one exclusive transaction. The scheduler
calls it before returning to decode (`parallel.rs:2105`), and completion requires
every long-prefill lane to reach its decode frontier. Existing internal prefill
chunks therefore do not give the decoding request service between chunks.
This was deliberately designed to avoid rebuilding Decode LRU between Prefill
ring chunks; fixing fairness must preserve that cache-lifetime protection.

The 160k control starts its long sparse prefill in a fresh process, whereas the
32k mixed case follows other warmed cases. System memory pressure also varies.
Do not attribute the full 33 -> 59 second difference to context capacity.
Ordinary expert arena size was about 16.67 GiB in both processes; the retained
dynamic/runtime corridor differed (9.47 vs 14.41 GiB).

### QSA split loses batching, not just fairness

`parallel.rs:1613` classifies each lane by
`indexer_top_k + max(compress_ratios) - 1` (about 2,051 tokens here).
`parallel.rs:2119` partitions the entire forward into dense/sparse groups and
sets a 16-step quantum when both groups exist. It does not only split attention.

Consequently the 0.5k/28k pair executes two one-row model forwards in turns,
instead of batching the fixed projections and experts for both requests.
Unprofiled per-stream text gaps reach about 0.58 s even after prefill completes.
The independent pager run confirms only `lanes=1` cohorts, predominantly 16
steps, versus a single `lanes=2` cohort for 8k/8k.

The shorter 512-output pager split test has a cold transition and a 1.74 s text
gap: overlap thirds 23.0 / 33.4 / 32.7 tok/s. Its whole overlap is 29.7, so the
profiler run must not replace the longer unprofiled 34.5 baseline. Warm 16-step
cohorts incur roughly 17-23 ms of one-time setup each (about 1.1-1.4 ms/token),
but most of the loss is the absence of two-row work sharing, not setup alone.

## Where the two-row time goes

Independent async pager profiling, 8,187/8,188 prompts and 512 output tokens per
request: 49.8 aggregate tok/s, close to the unprofiled 50.4. The measured runner
cohort covers 512 steps / 1,024 output rows in 20,614.4 ms.

Additive phase averages, milliseconds **per step producing two tokens**:

| Phase | ms |
| --- | ---: |
| Layer 0 execution, overlapping PLE | 1.422 |
| Exposed PLE wait | 1.567 |
| Main execution, layers after layer 0 | 36.296 |
| Front/setup/upload/tail/teardown/unaccounted | 0.978 |
| Total | 40.263 |

Additional counters are nested/overlapping, not additive savings:

| Counter per two-token step | Value |
| --- | ---: |
| Backend execution including layer 0 | 37.715 ms |
| Timestamped main-queue intervals | 22.936 ms |
| Timestamped DMA intervals | 3.765 ms |
| Main/DMA overlap | 2.188 ms |
| GPU interval union | 24.514 ms |
| Backend minus timestamped GPU union | 13.202 ms |
| Recorder lifetime, including host copies/preparation | 11.238 ms |
| Queue submissions / CPU submission time | 106.9 / 2.084 ms |
| Synchronization scope | 22.669 ms |
| Expert-role lookup hit rate | 92.0% |
| Host/ReBAR push / CPU push time | 52.45 MiB / 3.348 ms |
| Direct-DMA expert bytes | 98.74 MiB |
| Host-store misses / expert mmap reads | 0 / 0 |
| Total PLE worker time / exposed wait | 3.081 / 1.565 ms |

All 47.17 GiB routed experts reside in the host store, with 31.02 GiB imported
for direct DMA. Missing GPU experts still require promotion, but these runs
do not wait for expert SSD/mmap fallback reads. Approximately 58% of timestamped
DMA duration overlaps main-queue work. The remaining DMA duration is about
1.58 ms/step, not the complete expert-related stall budget.

CPU router readback, residency/LUT preparation and frequent submission boundaries
still fragment execution. The 22.7 ms synchronization scope contains real GPU
work; the 13.2 ms uncovered backend interval is not proven hardware idle or
wholly removable time. This investigation measures orchestration and transfers;
it does not reuse the previous single-row kernel shares as two-row kernel shares.

### A small, concrete PLE follow-up

`seam/ple.rs:427` caps independent-batch tasks at `spans.len()`. Two independent
requests therefore get only two task chunks, despite the four-thread persistent
pool used by the improved single-row path. Partition by actual gather groups
and bounded worker count, not request count; preserve disjoint destination writes.

The present exposed wait is 1.57 ms of 40.26 ms. Even eliminating all of it
would improve aggregate throughput only from 49.7 to 51.7, about **4% maximum**
in this profile. A realistic experiment should target a smaller gain and check
CPU contention and page faults, rather than assuming it solves the main limit.

### Memory pressure caveat

A system snapshot during ordinary testing showed roughly 4.5 GiB available
physical RAM and only 1.1 GiB remaining system commit, alongside page-in activity.
The process private bytes were approximately 75.7 GB and working set 51.2 GB.
The 48 GiB expert/RAM configuration does not imply that every process allocation,
driver commitment and PLE mapping fits in 48 GiB.

These counters do not identify which allocation caused the page-ins. The pager
still reports zero expert host-store/mmap fallback reads. Do not call all OS
paging "expert SSD waiting", or assign the cold-prefill difference to it without
allocation/page-fault attribution. Retain commit headroom as an operational
guard and expose staging/fixed/runtime host commitment separately.

## MTP: concurrent arrival is currently serialized

`infr-cli/src/main.rs:4542` selects the serialized ChatModel adapter when Vulkan
MTP and its draft head are enabled. Startup explicitly warns that `--parallel 2`
is ignored. Observed counters are `active=1 queued=1 kv_slots=1/1`, not two decode
slots. This is a current architectural limitation, not a scheduler tuning knob.

Default `spec.k=4` means four target VERIFY rows: one pending target token plus
up to three speculative tokens. It is not four draft tokens plus a fifth row.

| Prompts / arrivals | Profiling | A decode | B decode | B first text | Total output / wall |
| --- | --- | ---: | ---: | ---: | ---: |
| 507 / 508, together | Off | 36.0 | 35.8 | 35.75 s | 31.8 tok/s |
| 507 / 508, together | Pager | 35.1 | 35.1 | 36.86 s | 31.0 tok/s |
| 1,528 / 1,529, staggered | Off | 35.9 | 38.5 | 42.02 s | 31.8 tok/s |
| 1,528 / 1,529, staggered | Pager | 35.8 | 36.9 | 41.73 s | 31.5 tok/s |

The two decode columns describe successive single requests. They must not be
added. For the same short HTTP-arrival workload, ordinary serving completes
2,048 tokens in 44.35 s (46.2 end-to-end tok/s), versus MTP's 64.37 s (31.8).
These end-to-end values include prefill and queueing; the earlier 50.2 ordinary
number excludes initial prefill.

In the short pager run, A's decode thirds are 35.2 / 33.6 / 35.1 and B's are
34.3 / 33.7 / 36.3 tok/s. Mean cycles:

| Request | Draft acceptance | Emitted/cycle | Draft | Verify | Catch-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 52.0% | 2.55 | 4.56 ms | 68.34 ms | 0.73 ms |
| B | 54.4% | 2.63 | 4.59 ms | 70.01 ms | 0.71 ms |

The 1.5k pager run continues through the QSA boundary to 3,064 / 2,553 total
context tokens. Its thirds are A: **30.2 / 37.7 / 38.7**, B: **35.1 / 35.9 /
38.5** tok/s. Mean VERIFY is 67.64 / 71.76 ms, acceptance 53.7% / 61.0% and
emitted tokens per cycle 2.61 / 2.82. Both output hashes match the unprofiled
1.5k control. Thus the 3.5k initialization failure is not a blanket inability
to decode once the existing context crosses the QSA threshold.

VERIFY still dominates. These longer technical continuations do not sustain
the previous 24-token history prompt's 40.5 tok/s. Output hashes differ between
ordinary and MTP here, so this is a workload throughput comparison, not a
token-identical correctness A/B. Retain identical-prefix replay/logit-margin
checks before accepting future numerical/kernel changes.

MTP does not yet emit the ordinary GenerationProgress updates. The server's
live counter falls back to text deltas and reconciles final usage; its real-time
number is not a reliable exact token counter for this path. Restore progress
callbacks and context limits before using the footer to tune MTP policy.

### Long prompt initialization fails before decode

| Actual prompt size tested | Result |
| --- | --- |
| 507/508 | Successful |
| 1,528/1,529 | Successful, including continuation beyond the QSA threshold |
| 3,577/3,578 | Contiguous activation request 3,559,915,520 bytes exceeds largest arena gap 2,147,379,200 bytes |
| 8,187/8,188 | 7.57 GiB staging allocation fails with Vulkan `ERROR_UNKNOWN` |
| 28,667/28,668 | 26.52 GiB staging allocation fails |
| 29,687/29,688 | 27.46 GiB staging allocation fails |

No decode throughput exists for the failed cases. Both arrival orders were
attempted, but failures before first text mean the nominal stagger delay cannot
be exercised there. The server survived the request errors.

`mtp/qwen4.rs:1296` resets and primes the entire prompt with
`run_verify_with_finish`, materializes all-row hidden/IDs and catches the head up
on the whole prompt. It does not use ordinary bounded chunked prefill.
`seam/runner.rs:8116` also allocates `m * vocab * 4` staging when fixed VERIFY
buffers do not apply, even when GPU argmax IDs are sufficient. The graph carries
large vocabulary-shaped intermediates too.

For the 3.5k failure, the corridor had 5.99 GB free across shards but no single
gap larger than about 2 GiB. This is a tensor/chunk shape constraint, not proof
that fixed MTP weights were allocated after the unified pool. Enlarging the
arena or moving the head reservation does not remove prompt-by-vocabulary
scaling. Chunk target/head prime, carry the shifted hidden boundary correctly,
and compute only the frontier vocabulary result required to start generation.

## Practical optimization order

| Priority | Change | Expected impact and verification |
| --- | --- | --- |
| 1 | GPU routed-hit masks and finer pager synchronization | Remove the CPU readback/mask decision and reduce the approximately 107 per-step submissions while preserving versioned LUT and slot lifetimes. This is the only measured path large enough to close the remaining 5.1 ms/step. |
| 2 | Reusable command segments around paged MoE | Cache or replay shape-stable dense/recurrent command segments between dynamic router boundaries; target the measured 12 ms/step command-recording scope. |
| 3 | MTP chunked prime and exact progress | Make 3.5k/8k/29k prompts usable within unchanged budgets; no prompt-sized vocabulary staging, bounded hidden handoff, ordinary-equivalent prefix states. |
| 4 | Integrate per-slot MTP into ParallelSeam | True concurrent speculative batches with independent acceptance/rollback; retain ordinary decode fallback. Do not duplicate the whole model in two server processes. |

Two-request PLE fan-out and mixed-QSA shared cohorts are complete in this pass.
Long-prefill interleaving is deliberately not an optimization target: the current
exclusive phase protects the ring buffer and expert reconstruction and remains
unchanged by design.

Mixed-QSA changes must validate positions, compressed-history boundaries and
state isolation. Reducing the 16-step quantum alone smooths text delivery but
does not recover shared expert work and can increase cohort rebuild overhead.

True MTP concurrency needs shared immutable target/head weights, plus per-slot
head KV, pending token/hidden, accepted-prefix recurrent traces and cancellation
state. A two-slot four-row VERIFY has two causal four-row spans, **not eight
independent rows**. Reserve fixed per-slot resources before the real free-VRAM
query and unified arena allocation, following the existing allocation contract.
Bound concurrency/width rather than silently overcommitting resources.

At current short-run acceptance (about 2.55 emitted tokens per slot/cycle), a
future two-slot batched MTP cycle emits about 5.1 tokens. Matching ordinary
50 aggregate tok/s requires the complete batched cycle to stay below about
102 ms; achieving 60 (+20%) requires about 85 ms. Two serialized current cycles
cost about 147 ms. This is a conditional budgeting model, not a measured
eight-row result: batching may change acceptance, transfer pressure and cost.
Select width/fallback from emitted tokens per wall time, comparing ordinary
decode with the MTP head still resident as well as the headless baseline.

For ordinary two-row decode, the optimized short run is 36.89 ms/step. Reaching
63 tok/s permits 31.75 ms/step, so the next pass must save another 5.14 ms or
14%. PLE and mixed-QSA batching are no longer the limiting paths.

## Reproduction and limits

```powershell
node scripts/bench-concurrency.cjs ordinary-matrix-off ordinary off
node scripts/bench-concurrency.cjs ordinary-final ordinary off short,medium,split
node scripts/bench-concurrency.cjs mtp-matrix-pager mtp pager short,medium,mixed,reverse,long
node scripts/bench-concurrency.cjs mtp-control-off mtp off short,lowmid
node scripts/bench-concurrency.cjs mtp-lowmid-pager mtp pager lowmid
$env:BENCH_CTX = '163840'
node scripts/bench-concurrency.cjs ordinary-cap160k-off ordinary off mixed
Remove-Item Env:BENCH_CTX
$env:BENCH_TOKENS = '512'
node scripts/bench-concurrency.cjs ordinary-phases-pager ordinary pager medium,split
Remove-Item Env:BENCH_TOKENS
node scripts/analyze-concurrency.cjs ordinary-matrix-off ordinary-final ordinary-cap160k-off ordinary-phases-pager mtp-matrix-pager mtp-control-off mtp-lowmid-pager
```

Each matrix scenario has one full unprofiled run, with separate profiled/control
runs where stated, not a statistical confidence interval. Generated continuations
and cold/warm state differ. Maximum actual context tested is about 31k, below
the next 32k segment boundary; 160k occupied context, four-plus requests,
vision/embedding coexistence and cancellation were not benchmarked here.

The pass changes the parallel scheduler, PLE gather scheduling and Vulkan QSA
lowering, and adds the two benchmark scripts plus this report. All test servers
were stopped after their runs; no service was left running. The release binary
was rebuilt and the mixed-QSA pure and real-Vulkan tests passed.

Before merging an optimization, alternate old/new unprofiled runs at least three
times, preserve exact prompts/budgets, and compare mid/late overlap throughput,
P95/P99 text gaps, TTFT, memory commitment and output/state correctness. Include
QSA-boundary crossings, zero/partial/all MTP acceptance, EOS, late arrivals,
cohort shrink and cancelled peers. GPU validation belongs outside timed runs.
