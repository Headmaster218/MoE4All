# Qwen3.8 MTP investigation and implementation path

Follow-up: [post-optimization measurements and priorities](qwen38-mtp-bottlenecks-20260923.md).
The follow-up corrects the queue-timing interpretation below and includes a
controlled hidden-readback experiment.

## Implementation update

The first optimization pass is now implemented. The fixed benchmark remains
24 GiB VRAM, 48 GiB RAM, ubatch 3072, parallel ubatch 256, context 4096, greedy
sampling and the same 64-token continuation.

| Implementation | Decode |
| --- | ---: |
| Ordinary target | 29.9 tok/s |
| Initial four-token MTP | 12.0 tok/s |
| Pending-target protocol, fixed buffers and small-row MoE work | 22.9 tok/s |
| Fixed three-row VERIFY and width-sized draft | 24.7 tok/s |
| Multi-row PLE gather parallelism, run 1 | 26.8 tok/s |
| Multi-row PLE gather parallelism, run 2 | 26.4 tok/s |
| Production path without pager profiling | 26.0 tok/s |

The generated 64-token continuation is now byte-identical to ordinary greedy
decode in these runs. This is still a narrow correctness gate, not a substitute
for multi-prompt and boundary tests.

A production-shaped 256-token comparison used the same longer prompt for both
paths. MTP measured 31.9 tok/s and ordinary decode measured 33.7 tok/s. Both
output files had SHA-256
`421f724070183a6324adb00621d6e7285ea1844e05577c844cde407f4f66d10a`.
The longer run therefore reaches about 95% of ordinary decode, but does not yet
establish a speedup or the 40 tok/s target.

Implemented changes:

1. The target prediction is carried one token ahead. Each VERIFY starts with
   that known target token, so at least one row commits and a rejection restores
   directly to an accepted recurrent row. The separate target correction pass
   and MTP-head catch-up pass are gone.
2. Target VERIFY input/output buffers are fixed allocations made before the
   unified expert arena is finalized. VERIFY uses one target graph after its PLE
   rows are ready.
3. Decode scratch retains the bounded alternating draft/VERIFY topology family.
   Multi-row causal VERIFY supports shared-expert fusion and per-row hit-first
   masks. The later [single-decode investigation](qwen38-single-decode-bottlenecks-20260923.md)
   confirmed that shared-slot fusion does not cover this model's executed
   IQ2_S/IQ4_NL/IQ3_S formats; the implemented fusion is not active for these banks.
4. Draft length follows VERIFY width. The final draft step still writes MTP KV,
   but skips its unused HC head, vocabulary projection, argmax and readback.
5. Multi-row PLE batches use the four-thread gather pool. PLE work fell from
   about 249 ms to 80-83 ms per 64-token run, and exposed wait fell from about
   243 ms to 74-76 ms.
6. `spec.k` is an upper bound. Qwen3.8 selects at most three VERIFY rows: fixed
   width measurements were 25.9 tok/s for two, 26.4-26.8 for three, and 24.9
   for four. A rejection-driven adaptive 2/3/4 policy measured 25.5 tok/s and
   was removed.

Measured dead ends from this pass:

- Forcing the small-row MMQ path measured 19.9 tok/s.
- Extending next-layer expert prefetch to multi-row VERIFY measured 21.3 tok/s;
  nearly all predicted candidates were already resident.
- Disabling the submit splitter reduced submissions from 3970 to 3874 but left
  throughput unchanged at 26.5 tok/s.

The remaining target cost includes a structural synchronization boundary. A
representative three-row run recorded 1.40 s inside main-queue timestamped
intervals over a 2.84 s span, and one paging sync per target layer and VERIFY
cycle: 1536 waits totaling 1.36 s over 32 cycles. The 0.99 s recorder lifetime
also includes host preparation, not just command encoding. These are overlapping
measurements: paging sync waits for prior GPU computation as well as transfers,
and untimed DMA is absent from the queue coverage. They do NOT establish a
compute-only lower bound or prove 40 tok/s is attainable. The follow-up measures
the required per-cycle reduction and identifies smaller verified opportunities
before recommending a GPU-resident routed-hit schedule.

## Scope and baseline

Investigated `901d332` (`perf/qwen38-24g48g-ub3000`) with diagnostic-only changes.
No inference algorithm or kernel was changed in this investigation.

- GPU: AMD Radeon RX 7900 XTX. Vulkan enumeration changed between processes;
  identify the device by its logged name, not a permanently assumed Vulkan index.
- Target: `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`.
- Head: `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
- RAM/VRAM/ubatch/parallel ubatch: 48 GiB / 24 GiB / 3072 / 256.
- Final paired runs: context capacity 4096, prompt `北京是什么？`, 15 prompt tokens,
  64 output tokens, temperature 0, no thinking, one request at a time.
- `INFR_PAGER_PROFILE=1`, `RUST_LOG=info`; `INFR_PROF_STAGES` and per-op profiling OFF.
- Earlier diagnostic run used context 512, which clamps the effective ubatch to
  512. It is not the full 3072-ubatch memory configuration.

| Final paired run | Decode |
| --- | ---: |
| Ordinary | 29.9 tok/s |
| MTP, four candidates and one four-row VERIFY | 12.0 tok/s |

These are single short-context runs, not a long-context performance claim. The
generated continuations differ, so this is a same-request comparison, not an
identical-token-stream microbenchmark.

Raw logs are local ignored artifacts under `target/mtp-research-{async,baseline}`
with `.err.log` and `.out.log` suffixes. The earlier synchronous diagnostic is
`target/mtp-research-profile.err.log`.

## Measured costs

The asynchronous MTP run had 27 draft/VERIFY cycles and 26 corrections. Scope
averages below exclude warmup; prompt initialization is outside these scopes.

| Scope | Wall ms/call | Backend scratch setup ms/call |
| --- | ---: | ---: |
| Four-token draft | 12.86 | 5.88 |
| Four-row VERIFY | 107.45 | 11.48 |
| Head KV catch-up | 21.28 | 5.03 |
| Restore accepted recurrent state | 0.38 | 0 |
| One-token target correction | 47.13 | 11.14 |
| New baseline snapshot | 0.46 | 0 |

VERIFY averaged 136 queue submissions, 55.05 ms of recorded host sync wait,
86.41 MiB of host memcpy/ReBAR push and 208.47 MiB of dedicated DMA payload.
Head catch-up recorded 15.77 ms inside command recording/lowering despite having
no paged expert traffic. Lowering includes lazy workspace allocation; this is
not a measurement of shader execution time.

Whole profiled-run scratch setup: MTP 909.5 ms versus ordinary 59.8 ms. These
totals include prompt execution and different numbers of target rows; use the
per-scope counters above for MTP phase attribution. Backend setup/record/sync
timers overlap and must NOT be summed as independent wall-time components.

PLE residual wait in the MTP profiled run was only 23.7 microseconds. Host-tier
reads and mmap fallback were zero. This run does not point to ngram or SSD reads
as the dominant exposed stall. Full DMA GPU timing is unavailable on this
device/run (`gpu_timed_submits=0`); zero reported DMA GPU time is not zero cost.

At 64 / 27 = 2.37 emitted tokens per cycle and 29.9 ordinary tok/s, the current
accept pattern needs an average complete cycle below about 79 ms to break even.
VERIFY alone currently exceeds that. Fixing allocation alone is useful but
insufficient to establish an MTP speedup.

## Confirmed architectural issues

1. Fixed MTP runtime ownership is incomplete. Head weights, KV and ingress
   buffers are allocated up front, but graph internals and per-op scratch still
   go through the generic executor after the unified arena exists.
   `Qwen4MtpSession::{draft4,catch_up}` rebuild graphs on every call.
2. `RuntimePhaseArena` retains only the active and one parked graph topology.
   The draft graph contains MoE and enters this same backend-wide cache even
   though its head experts are resident. It competes with target layer-0,
   remaining-layer and correction graph families. Catch-up has no MoE, so
   `paged_static_phase` returns None and its scratch/pool are local to each call.
3. The Qwen VERIFY runner separately allocates input, position, hidden, PLE,
   logits and result buffers each time. It does not reuse a fixed four-row
   execution object. Position and KV length are baked into graph operations;
   blindly retaining an old plan would be incorrect.
4. VERIFY misses important ordinary-decode MoE optimizations. Shared-expert
   fusion requires one row. Hit-first demand promotion requires one row or
   independent-sequence rows. Four causal speculative rows satisfy neither.
   The model's expert-prefetch hints are also generated only for batch=1;
   prefetch was inactive in these measurements, so no measured gain is claimed
   for extending that hint path.
5. Candidate zero is compared to an already-known `target_prediction`, but the
   comparison happens after the full VERIFY. A known failure still spends a
   four-row target forward, captures traces, and then restores the baseline.
   The staged diagnostic had seven such cycles among 27 measured cycles.
6. Every partial acceptance immediately runs a separate target correction
   forward. Prefix-state restoration removed accepted-token replay, but did not
   remove this extra full-model pass. Restoring state is now cheap; the correction
   forward and its workspace churn remain expensive.
7. Prompt priming runs the whole prompt through all-row VERIFY, including an
   `m * vocab` logits allocation/projection although only the final prediction
   is consumed. Unlike ordinary prefill, this path does not chunk the target
   forward by ubatch. The 15-token test does not expose the resulting long-prompt
   time/memory risk.

Code entry points:

- `crates/infr-llama/src/mtp/qwen4.rs`: head graphs, fixed resources, cycle protocol.
- `crates/infr-llama/src/seam/runner.rs`: Qwen VERIFY, prompt ingestion and correction.
- `crates/infr-vulkan/src/adapter.rs`: `RuntimePhaseArena`, `execute_static_inner`,
  `paged_static_phase`, `paged_moe_shared_at`, `execute_paged_moe`.
- `crates/infr-llama/src/seam/weights.rs`: recurrent snapshots and accepted-row restore.

## Implementation order

### 0. Establish the correctness gate

MTP and ordinary greedy still diverge after the phrase ending in `创新中心`.
The state-trace change did not introduce this difference, but that does NOT
prove it is harmless numerical noise. Do not enable MTP by default yet.

Add a same-prefix teacher-forced comparison: snapshot identical initial state,
run four known input tokens batched, restore, then run those exact tokens one at
a time. Compare per-row raw logits, argmax margins, hidden states and recurrent
state, then narrow the first differing layer. Separately compare restored
prefixes 0..4 with sequential reference states, including PLE and QSA boundary
crossings. Test EOS inside an emitted group: the current inner-loop break does
not terminate the outer generation loop.

### 1. Make the small MTP runtime persistent

Keep four-token drafting and four-row verification unchanged. Give MTP explicit
workspace ownership for draft, target VERIFY layer-0/main, correction and head
catch-up shapes 1..4. Allocate the required fixed buffers before the final live
VRAM query and unified-pool creation, as required by the existing memory design.
Retain op scratch as well as graph tensors; merely caching `compile()` is not
enough. `compile()` currently wraps/clones the graph, not recompiles shaders.

Use caller-owned execution workspace or an explicit workspace identity in the
backend, not an unbounded topology cache and not a separate model/backend copy.
Position-dependent fields must remain refreshed. Preserve the large ordinary
prefill arena's release rules rather than making every prefill buffer permanent.

Reuse four-row IO and logits buffers. Keep hidden-state handoff device-resident
where practical; CPU readback only needs acceptance IDs and emitted tokens.
Remove unused catch-up scratch declarations. Treat alloc-free steady state and
stable expert residency as the first acceptance criteria; target under 5 ms of
workspace preparation per cycle, then measure actual end-to-end benefit.

### 2. Give causal VERIFY the optimized small-row MoE schedule

Extend hit-first masks and shared-expert fusion to the local MoE operation for
up to four causal rows. Routing/FFN rows are independent within a layer even
though attention and recurrent state are causal. Stage the union of needed
experts once, execute resident slots while misses transfer, preserve each row's
router slot/reduction order, then complete missing slots.

Do NOT set `Graph::independent_rows` on the VERIFY graph: that would change
attention/recurrent semantics. Do not blindly lower the global small-m threshold
to force the prefill MMQ path; its bucketing overhead and numerical behavior
must be measured at the real model shapes.

Validate paged/resident equivalence, mixed quantization types, shared slots and
all acceptance prefixes. Benchmark ordinary one/two-slot decode too, because
the modified executor is shared. Run Vulkan validation outside timed tests.
Only after this schedule is measured should per-kernel profiling select specific
four-row dense/LM-head/DeltaNet optimizations. Their individual contributions
have not been isolated in this investigation.

### 3. Remove provably wasted cycles and passes

Small independent change: check candidate zero before arming traces/VERIFY.
On mismatch, catch up the head with the known target token and run only the
necessary correction. Do not truncate valid trunk state or restore a trace
that was never produced. Add a forced-first-rejection token-equivalence test.
This can be developed alongside step 1 without changing the four-token policy.

Larger later change: carry the correction or full-accept bonus token as a pending
target input into the next VERIFY, avoiding an immediate correction-only pass.
This needs an explicit pending-token state machine. Keeping four actual draft
candidates would then require five verified inputs; keeping four verification
rows would leave only three speculative rows. Resolve that policy separately,
as requested, rather than silently changing the current four-token baseline.

### 4. Finish long-context integration and tune policy

Chunk prompt priming through the existing prefill path, tap hidden states for
head catch-up in bounded chunks, and compute only the required frontier logits.
Test near QSA and segmented-KV boundaries, max-new limits, EOS and cancellation.
Then compare 2/3/4 candidate policies, confidence gates and profitable fallback
using actual emitted tokens per cycle and wall time, not aggregate acceptance
rate alone. Measure a fallback while the MTP head remains resident: unloading
the head changes the target expert-cache budget and is a different comparison.

## Measurement cautions

- `Recorder::finish_nowait` becomes blocking when `prof.stages` OR `prof.ops`
  is enabled. Use scoped pager counters for production-shaped overlap; use
  per-op timestamps only to investigate kernels, not to claim end-to-end gains.
- MTP's measured expert arena was 14.10 GiB versus ordinary 16.31 GiB. The fixed
  head/state consume real VRAM; this cost is legitimate and must remain in tests.
- The allocation retry in prior logs came from an explicit remaining-budget
  guard. It does not by itself prove Vulkan fragmentation or wrong head placement.
- GPU occupancy/power alone does not distinguish dispatch, allocation, transfer
  and compute limits. Main-queue timestamps omit untimed DMA and are not a
  complete device-utilization measurement.
- Use repeated 256+ token tests at multiple context depths and several prompts
  before claiming a speedup. Keep the RAM/VRAM/ubatch baseline fixed and retain
  raw output for correctness checks.
