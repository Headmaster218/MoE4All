# Qwen3.8 MTP: bottlenecks after the first optimization pass

> Update after implementation: this document keeps the original diagnostic
> baseline. Current results and retained/rejected changes are in
> [Implemented follow-up](#implemented-follow-up).

Follow-up: [ordinary single-sequence decode measurements](qwen38-single-decode-bottlenecks-20260923.md),
including a shared-expert fusion format-coverage correction applicable to MTP.

## Scope and reproducibility

Base: `72df13d` on `perf/qwen38-24g48g-ub3000`. The initial worktree was clean.
The original diagnostic pass only exposed cycle, acceptance and VERIFY-detail
logs under `INFR_PAGER_PROFILE`, without enabling the blocking
`INFR_PROF_STAGES` path. Later sections describe the retained optimizations.

- AMD Radeon RX 7900 XTX; select its enumerated name, not a hard-coded index.
- Target: `D:\AILMStudioModels\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
- Head: `G:\Qwen3.8-Flash-Next-UD-Q2_K_XL\MTP\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
- VRAM / RAM / ubatch / parallel ubatch: **24 GiB / 48 GiB / 3072 / 256**.
- Context capacity 4096, temperature 0, seed 1, no thinking, 256 output tokens.
- The fixed runtime supports **two through four VERIFY rows**. `spec.k` now
  controls that range without a hidden three-row cap; the default maps to four
  rows, one pending target token plus three speculative tokens.
- History prompt is the same 24-token Beijing history/culture/geography request
  used in the preceding 256-token benchmark. A second 38-token prompt requests
  a Rust bounded LRU cache, complexity explanation and tests.
- These are short-context, single-request measurements, not a long-context or
  concurrent-serving performance claim.

Local ignored artifacts are under `target/`: `mtp-profile-run.cjs`,
`mtp-profile-analyze.cjs`, `mtp-op-analyze.cjs`, and each run's `.meta.json`,
`.err.log`, `.out.log` and, for MTP profiles, `.analysis.json`.
The runner clears inherited `INFR_*` experiment knobs, records arguments, and
captures UTF-8 directly without PowerShell stderr wrapping.

Example from the workspace root:

```powershell
node target/mtp-profile-run.cjs mtp-current-history-1 mtp pager history 256
node target/mtp-profile-run.cjs ordinary-current-history-off ordinary off history 256
node target/mtp-profile-analyze.cjs mtp-current-history-1
```

## Throughput and acceptance

| Run | Profiling | Decode tok/s |
| --- | --- | ---: |
| Ordinary, history | off | 31.7 |
| MTP, history | off, first / repeated | 30.6 / 30.2 |
| Ordinary, history | pager | 30.2 |
| MTP, history | pager | 31.3 |
| MTP, Rust code | pager | 31.1 |
| MTP, Rust code | off | 30.3 |
| Ordinary, Rust code | off | 30.0 |
| MTP, hidden Readback experiment, history | pager | 32.2 |
| MTP, hidden Readback experiment, history | off | 34.1 |

Do not mix profiling modes to claim a speedup. These are individual runs with
visible variation; the old 31.9/33.7 numbers remain historical, not this session's
paired baseline. The history continuations are byte-identical across ordinary,
MTP, repeats and the Readback experiment, UTF-8 SHA-256:
`99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`.

| MTP pager run | Cycles | Accepted draft tokens | 0 / 1 / 2 accepted drafts | Emitted tokens/cycle |
| --- | ---: | ---: | --- | ---: |
| History | 108 | 149/216 = 69.0% | 21 / 25 / 62 | 2.370 |
| Rust code | 104 | 153/208 = 73.6% | 15 / 25 / 64 | 2.462 |

The last cycle can commit more rows than `max_new` emits. Use 256 divided by
cycle count for throughput budgeting, not `(cycles + accepted_drafts)/cycles`.
All baseline and Readback history runs have the same acceptance pattern.

The Rust sample does NOT match ordinary greedy: both agree through the initial
cache definition and several comments, then MTP writes `But VecDeque` where
ordinary writes `But Vec`. Subsequent text diverges. The history hash match is
therefore not a general correctness clearance. The code-prompt throughput is
not an identical-token-stream comparison. A same-prefix, teacher-forced
batched/sequential logits and recurrent-state comparison is required before
enabling MTP by default; do not assume the mismatch is harmless numerical noise.
Repeating MTP with profiling off produced the same code-sample hash as its pager
run (`c5abdeec2bf38d000dffc28aed8952fb8f031a0119db4b656eedad60bc566312`),
different from ordinary's
`2c9aad1c2e4cfc870d4d730a04198e45c7bbdc15e7bbe9490ba466c84926e1bd`.
The mismatch is not explained merely by enabling the diagnostic logging.

## Additive wall-time decomposition

Per cycle, using the history pager run. Prompt priming is excluded.

| Component | Mean ms | Interpretation |
| --- | ---: | --- |
| Draft | 3.46 | Includes the resident MTP head |
| Target VERIFY execution | 63.61 | GPU computation plus host paging/orchestration |
| VERIFY hidden/ID readback | 5.04 | After execution has completed |
| PLE preparation wait and upload | 2.41 | Exposed wait, not asynchronous total work |
| VERIFY graph construction and bindings | 0.21 | Not shader compilation |
| VERIFY input preparation | 0.01 | Fixed buffers are reused |
| Other VERIFY wrapper work | 0.12 | Difference to full VERIFY scope |
| Restore and snapshot bookkeeping | 0.70 | Already avoids target-token replay |

Full VERIFY averages **71.40 ms**, about 94% of the cycle. Code prompt VERIFY
averages 74.78 ms, including 67.01 ms execution and 5.09 ms readback. Neither
draft nor restore is currently the main lever. Scratch setup has a 0.03 ms
median per VERIFY; its 0.55 ms mean includes initial shape preparation. Further
generic graph/scratch caching is not the first priority.

At 2.370 emitted tokens/cycle, 40 tok/s permits **59.26 ms per complete cycle**.
The present cycle is approximately 75.7 ms. With the current other costs,
VERIFY must fall from 71.4 to roughly **55 ms**, a reduction of about 16 ms/23%.
For the code prompt the corresponding VERIFY budget is about 57 ms. This is a
target budget, not a predicted achievable speed.

Even eliminating draft entirely would save only about 0.37 s per history run;
it cannot independently deliver 40 tok/s.

## Confirmed low-risk opportunity: hidden readback

`seam/runner.rs` allocates both fixed small-row `h_out` and the larger prompt
fallback as `BufferUsage::Staging`. Vulkan maps this to `CpuToGpu`, while
`Readback` maps to `GpuToCpu`. `VulkanBackend::download` reads mapped buffers
directly with `copy_nonoverlapping`; it does not repair an unsuitable memory
placement with a cached staging copy.

A two-line A/B changed only those hidden-output allocations to `Readback`:

- Readback **5.044 -> 0.016 ms/cycle**; 544.78 -> 1.73 ms over 108 cycles.
- IDs, acceptance distribution, expert misses and transfer bytes unchanged.
- Full VERIFY **71.40 -> 69.21 ms** in the profiled samples: execution itself
  varied from 63.61 to 66.32 ms, so not all of the isolated saving appeared as
  end-to-end improvement in that pair.
- Without profiling: baseline 30.6, experiment 34.1, restored baseline 30.2
  tok/s. This is promising, but still only one experiment sample, not a stable
  percentage-speedup guarantee.

This paragraph described the state at the end of the diagnostic pass. The
Readback placement is now retained in source for both the fixed four-row hidden
buffer and its larger fallback. Those fixed buffers are still allocated before
the live free-VRAM query and unified expert arena.

This finding is consistent with Vulkan's warning that uncached/write-combined
host-visible memory can be very slow to read. See the
[Vulkan memory specification](https://docs.vulkan.org/spec/latest/chapters/memory.html).
The measured result, rather than that general rule alone, establishes the
priority on this machine. Fixed-resource allocation must remain before the live
free-VRAM query and unified arena, including after this change.

Next, keep the accepted hidden row on the GPU and pass/copy it into `draft_h`.
Only the three acceptance IDs need CPU readback. Existing
`Backend::copy_buffer_ranges` can select the accepted row; preferably include
the copy in an existing submission instead of adding another blocking one.
This is a local MTP/runner interface change, not a new model/backend or pager.

## Expert computation and the paging boundary

A separate 64-token `INFR_PROF_OPS=1` run identifies kernels. This mode disables
normal async overlap and emits extensive per-submit reports; its 5.8 tok/s is
NOT a production throughput result. CPU compilation also ran during this
diagnostic, so neither its absolute wall time nor device milliseconds should be
used to predict the production gain. Phase markers let the analysis exclude
prompt and draft from the VERIFY kernel distribution:

| VERIFY timestamp bucket | Share of instrumented VERIFY device intervals |
| --- | ---: |
| `native_idm_iq2s_paged` | 38.0% |
| `native_idm_iq4nl_paged` | 14.2% |
| `native_idm_iq3s_paged` | 4.7% |
| `deltanet_seq_trace` | 7.3% |
| Three-row dense projections | Several 1-4% buckets |
| Target vocabulary projection | 2.2% |

The three routed-expert buckets total about **57%** of this diagnostic device
time. Thus the target is not simply idle waiting for expert bytes; expert
computation/dequantization is a substantial hotspot too.

Two specific code-level opportunities precede a pager redesign:

1. `native_gemv_id_multi.comp` and its `_sg` twin call `GRID_INIT` **before**
   checking row/slot masks. Hit-first and miss-second dispatches therefore both
   initialize IQ codebooks for groups that immediately return. Prototype a uniform
   workgroup-level mask/bounds decision before initialization; preserve valid
   barrier control flow and prove that active groups' arithmetic is unchanged.
   The later ordinary-decode investigation confirmed that shared-slot fusion
   currently supports Q5_K/Q6_K/IQ4_XS, not this model's IQ2_S/IQ4_NL/IQ3_S banks.
   Skipping IQ initialization for a shared Q8 slot is therefore an additional
   requirement for a future format extension, not a current measured hotspot.
2. The current small-row grid independently handles each `(token row, expert
   slot, output row)`. Repeated experts across speculative rows do not explicitly
   share weight decode. Prototype a 2-4-row expert microkernel or persistent
   read-only codebook buffer, starting with IQ2_S. Benchmark actual quant/shape
   combinations and register/LDS pressure. Do not force generic MMQ or globally
   enable the SG route: previous measurements and `native_id_sg_choice` document
   regressions for these low-output IQ2_S shapes.

The structural boundary remains: `execute_paged_moe` tests whether the entire
layer's expert banks are resident, not merely the selected experts. Otherwise
`sync_stream` submits and drains prior work, reads router IDs on the CPU, then
builds hit masks and prepares hit/miss execution. There are 48 such boundaries
per VERIFY. Hit-first already overlaps miss preparation with resident expert
work, so adding another generic prefetch knob would duplicate existing work.

History VERIFY averages 107 total queue submissions and records 36.06 ms inside
paging-sync scopes per cycle. That 36.06 ms includes prior computation, queue
submission, draining and recorder acquisition. It is **not 36 ms of expert-I/O
stall**, and cannot be subtracted as recoverable time. Replacing queue idle with
a fence alone still leaves the router-to-CPU-to-expert dependency. See the
[Khronos synchronization example](https://docs.vulkan.org/samples/latest/samples/performance/wait_idle/README.html)
for the distinction between draining a queue and waiting on specific work.

A real overlap design needs a GPU-readable residency map and GPU hit masks, an
early router-readback completion signal, resident work submitted before CPU
promotion finishes, frozen per-submission LUTs, and slot pinning until readers
complete. Do not reset the current LUT tape or overwrite LRU cells on only a
router fence: the existing full drain currently guarantees their lifetime.
This affects the Vulkan pager/executor, not the causal model graph. It deserves
its own correctness and performance milestone after the smaller changes.

## Transfer and other secondary costs

- All 47.17 GiB routed experts fit in the host store; no runtime expert SSD or
  host-tier miss reads were recorded. This does not mean the separate PLE
  mmap/dequant work is free.
- Only 31.02 GiB of host expert memory was imported for direct DMA. History
  VERIFY moves 116.33 MiB by DMA plus 54.58 MiB by host/ReBAR push per cycle;
  the latter takes 3.82 ms of CPU time. Code prompt values are 147.69 MiB,
  85.16 MiB and 5.51 ms. These overlap other work and are not additive stalls.
- Main-queue timestamp coverage was 4.54 s over an 8.73 s span including prime.
  DMA has no GPU timestamps in this run. Uncovered intervals are not a precise
  hardware-idle measurement, and this ratio does not prove a 40 tok/s bound.
- Measured unified arena: MTP 14.59 GiB, ordinary 16.92 GiB in this session.
  MTP's smaller expert cache is a real fixed-resource cost. Compare a future
  ordinary fallback with the MTP head still resident, not only a fresh process
  without the head.
- PLE's exposed 2.4 ms/cycle is a smaller follow-up target. Seek earlier
  asynchronous submission where inputs are known, without reintroducing more
  execute boundaries than the wait it removes.
- Recurrent trace writes matter more than restore: `deltanet_seq_trace` is a
  measurable kernel bucket, but only `n-1` prefix states are already traced.
  Preserve direct accepted-prefix restoration; do not reintroduce replay.

## Recommended delivery order

| Priority | Work | Acceptance gate |
| --- | --- | --- |
| P0 | Correct hidden readback placement, then GPU hidden handoff | Readback below 0.1 ms; same acceptance/output; repeated paired end-to-end gain |
| P1 | Mask before IQ codebook staging; extend shared-slot format coverage separately | Bit-identical masked expert outputs; ordinary/MTP regression tests; gain with profiling off |
| P2 | Target IQ2_S/IQ4_NL small-row kernels and weight/codebook reuse | Shape-specific microbench plus complete-cycle improvement, not only kernel speed |
| P3 | GPU routed-hit schedule and fine-grained pager lifetimes | Fewer exposed router boundaries without stale LUT/slot races or changed row semantics |
| P4 | PLE overlap, trace bandwidth, bounded width/fallback policy | Select by emitted tokens/wall time with head resident; retest widths after costs change |

Correctness is a gate for every priority: isolate the Rust sample's first
divergence with identical input token prefixes and logit margins, then compare
all accepted-prefix recurrent/PLE states with sequential execution.

Do not assume the savings add independently. At the current acceptance pattern,
the immediate objective is stable ordinary-decode parity, then a complete cycle
at or below 59 ms for the history sample. Re-measure after every milestone.

For each landed change, alternate ordinary/current/new runs at least three
times with profiling off; retain token IDs/output and cache budgets. Cover
rejection prefixes, all-accepted cycles, EOS/max-new truncation, quant mixtures,
shared slots, QSA boundaries and ordinary one/two-slot decode. Run Vulkan
validation outside timed runs. Long-context serving remains a separate gate:
the MTP prompt-prime path still produces all-row hidden/logits and does not use
ordinary chunked prefill, so the short-prompt measurements cannot clear that
memory/time risk.

## Implemented follow-up

The retained MTP work is:

- Put VERIFY hidden output in `BufferUsage::Readback`. Its measured readback is
  about 0.012 ms/cycle instead of roughly 5 ms/cycle.
- Reuse fixed IDs, positions, hidden, wide residual, PLE, result and recurrent
  trace buffers allocated before the unified-pool free-space query.
- Apply the ordinary IQ mask-before-codebook change. Keep the new shared-slot IQ
  fusion for one row only after the multi-row A/B showed 36.8 tok/s enabled and
  37.8 tok/s with the existing dense shared path.
- Add IQ2_S multi-output-row tree variants and select NR=8 by default through
  `kernels.vulkan.gemv.id_grid_nr`. In the 64-token op diagnostic, IQ2_S VERIFY
  device time fell from 706.9 to 384.7 ms, or 45.6%, with identical output.
- Run target layer 0 while asynchronous PLE gathering is in flight, then pass a
  fixed hidden/wide state into layers 1-47. This is controlled by
  `spec.mtp_ple_overlap` and `INFR_NO_MTP_PLE_OVERLAP=1`.
- Remove the hidden three-row policy cap: the preallocated four-row runtime is
  now reachable through `spec.k=4`.

An IQ3_S NR2/4/8 experiment was rejected. With the same 108-cycle acceptance
profile it increased mean VERIFY from about 54.5 to 57.0 ms, so only the proven
IQ2_S variants remain.

### Final measurements

All rows use the 24-token history prompt, 256 output tokens, 24/48 GiB budgets,
ubatch 3072, parallel ubatch 256, temperature 0 and seed 1. The final output
SHA-256 is unchanged at
`99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`.

| Final mode | Decode tok/s | Cycles | Draft acceptance |
| --- | ---: | ---: | ---: |
| Ordinary decode | **37.1** | 256 forwards | n/a |
| MTP, three VERIFY rows (`spec.k=3`) | **40.8** | 108 | 149/216 = 69.0% |
| MTP, default four VERIFY rows | **40.5** | 96 | 162/288 = 56.2% |

The default four-row result is about 9.2% faster than final ordinary decode;
three rows are about 10.0% faster. Four rows reduce cycle count but increase the
per-cycle VERIFY and draft cost enough that the two policies are effectively
tied on this prompt. Keep the width configurable and select on emitted tokens
per wall time rather than raw acceptance.

For the three-row pager-profile A/B, layer-0 PLE overlap changed exposed PLE
from 2.249 to 0.831 ms/cycle, while VERIFY backend execution rose from 54.014
to 54.439 ms/cycle because of the extra graph/submit. Net VERIFY wall fell from
56.606 to 55.810 ms/cycle and profiled throughput rose from 39.1 to 39.6 tok/s.

The minimum objective, MTP parity with optimized ordinary decode, is met. The
ideal 20% uplift would be roughly 44.5 tok/s and is not yet met. The remaining
dominant work is the target VERIFY pager/executor: roughly 29-33 ms/cycle is
inside paging synchronization/queue completion, with around 108-113 submits per
cycle. Those counters include real GPU work and cannot be removed as simple
CPU idle. The next credible milestone is GPU-resident routed-hit masks plus
frozen submission LUTs and pinned pager slots, allowing resident expert work and
promotion to overlap without the current full router-to-CPU drain.
