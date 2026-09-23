# Qwen3.8 ordinary single-sequence decode: measured bottlenecks

> Update after implementation: the baseline analysis below is retained for
> provenance. The optimization results and current status are recorded in
> [Implemented optimization pass](#implemented-optimization-pass).

## Scope and reproduction

Investigated `72df13d` on `perf/qwen38-24g48g-ub3000`, following the
[MTP investigation](qwen38-mtp-bottlenecks-20260923.md). MTP is disabled and its
head is not loaded. This measures ordinary decode, not the ordinary fallback
with MTP fixed resources still resident, and not concurrent serving.

- CPU: Ryzen 5 5600X; GPU: Radeon RX 7900 XTX. Select the enumerated GPU name,
  since its Vulkan device index can change between processes.
- Model: `D:\AILMStudioModels\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
- Fixed VRAM / RAM / ubatch / parallel ubatch: **24 GiB / 48 GiB / 3072 / 256**.
- Temperature 0, seed 1, no thinking; one fresh process/request per run.
- Actual prompt lengths: 24, 1,347 and 9,368 tokens. Context capacities: 4,096
  for the first two, 16,384 for the last. Capacity is not actual context length.
- Ordinary expert arena: about 16.92 GiB after real fixed-resource allocation.
  All 47.17 GiB of routed experts fit in the host store; 31.02 GiB is imported
  for direct DMA. No expert host-tier or mmap fallback reads occur during decode.

Added diagnostic-only decode phase markers, phase counters and per-layer deltas
to `crates/infr-llama/src/seam/runner.rs`, gated by Qwen3.8 and
`INFR_PAGER_PROFILE`. They exclude prefill and do not enable the blocking
`INFR_PROF_STAGES` path. No inference algorithm, kernel or scheduling policy was
changed. Earlier uncommitted MTP diagnostics and reports were preserved.

Local ignored artifacts under `target/` include `mtp-profile-run.cjs`,
`single-decode-analyze.cjs`, `single-decode-compare.cjs`, and run-specific
`.meta.json`, `.err.log`, `.out.log` and `.analysis.json` files. The runner clears
inherited `INFR_*` knobs and records exact prompts, environment and arguments.
Example commands from the workspace root:

```powershell
node target/mtp-profile-run.cjs single-decode-history-off-1 ordinary off history 256
node target/mtp-profile-run.cjs single-decode-history-pager ordinary pager history 256
node target/mtp-profile-run.cjs single-decode-context-pager ordinary pager context 256
node target/mtp-profile-run.cjs single-decode-deep-pager ordinary pager deep 256 target/release/infr.exe 16384
node target/mtp-profile-run.cjs single-decode-history-ops ordinary ops history 64
node target/mtp-profile-run.cjs single-decode-deep-ops ordinary ops deep 64 target/release/infr.exe 16384
node target/single-decode-analyze.cjs single-decode-history-pager
```

`history` is the same Beijing history/culture/geography prompt as the MTP report.
`context` and `deep` contain repeated structured engine records followed by a
benchmark-design question. They have different continuations: differences
between these runs cannot be attributed to context length alone.

## Throughput baseline

| Artifact label after `single-decode-` | Prompt tokens | Output tokens | Profiling | Decode tok/s |
| --- | ---: | ---: | --- | ---: |
| `history-off-1` | 24 | 256 | off | 35.0 |
| `history-off-2` | 24 | 256 | off | 32.2 |
| `history-512-off` | 24 | 512 | off | 32.0 |
| `history-pager` | 24 | 256 | async pager | 33.3 |
| `context-pager` | 1,347 | 256 | async pager | 31.3 |
| `deep-pager` | 9,368 | 256 | async pager | 30.0 |

The two 256-token unprofiled history outputs and the pager output are identical,
SHA-256 `99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`.
The first off run predates the new single-decode diagnostic hooks; all later
runs use the rebuilt diagnostic binary. This is not an optimization A/B.
Variation is material: report **32.2-35.0 tok/s**, not a stable 35 tok/s baseline.
The 512-token sample also does not support assuming that a longer run must be
faster once warm. Expert selection and cache churn depend on generated tokens.

Per-op diagnostic runs produce only 3.3/3.2 tok/s. They synchronize/report at
submission boundaries and destroy normal overlap; never use those rates or
their absolute kernel milliseconds as production performance predictions.
No compilation ran concurrently with these new per-op diagnostics.

## Where ordinary token wall time goes

Decode-only phase averages, milliseconds per emitted token:

| Phase | 24-token prompt | 1,347-token prompt | 9,368-token prompt |
| --- | ---: | ---: | ---: |
| Total decode wall | 30.062 | 31.921 | 33.380 |
| Target execution: host setup/record/submit plus GPU | 28.363 | 29.958 | 31.621 |
| PLE exposed wait and upload, outside execution | 1.270 | 1.505 | 1.284 |
| Graph construction, compile wrapper and bindings, excluding PLE | 0.299 | 0.316 | 0.329 |
| Greedy token-ID readback | 0.0064 | 0.0065 | 0.0067 |
| Other runner work, including diagnostic logging | 0.123 | 0.135 | 0.139 |

These rows are additive except the total. Backend setup/recording is already
inside execution. Layer 0 takes about 1.14-1.18 ms/token within execution and
overlaps the asynchronous PLE job. PLE's total worker time is 2.30-2.68 ms/token;
only the exposed wait belongs in the additive table.

Ordinary greedy decode downloads a four-byte ID and does not request the MTP
hidden-output buffer. The MTP report's roughly 5 ms/cycle hidden-readback problem
does **not** apply here. LM-head GPU computation is a separate, smaller cost.

Cold transition matters: the first deep-context forward has 289 ms execution,
versus about 44 ms for the short prompt. The deep run's first 64 forwards average
38.35 ms, then 30.90, 31.97 and 31.72 ms in subsequent 64-token windows. Pipeline
variants, prefill-to-decode resource transitions and expert-cache state need
separate tracing before assigning this spike to one cause. Report first-token
latency and steady-state decode separately; do not silently drop cold work.

## Device hotspots, not just expert transfer

Separate 64-token op diagnostics, aggregated only between the decode markers.
Shares are of instrumented GPU operator intervals, **not token wall time**:

| Kernel group | Short prompt | 9,368-token prompt |
| --- | ---: | ---: |
| Routed experts | 41.5% | 40.4% |
| Dense projections, including unfused shared-expert projections | 38.4% | 35.9% |
| Attention and QSA | 1.2% | 5.1% |
| Recurrent/DeltaNet/conv | 2.9% | 2.7% |
| Vocabulary projection | 3.4% | 3.0% |
| Routing, normalization and other operators | 12.6% | 12.9% |

Short-prompt expert buckets are IQ2_S 28.0%, IQ4_NL 9.5%, IQ3_S 4.0%.
Prominent `gemv_streamed:m1` shapes include `2560x10240` (6.5%), `6144x2560`
(5.5%), `320x10240` (4.8%), `2560x6144` (4.5%) and `10240x320` (3.8%).
`moe_topk_sg` is another 3.8%. Fixed projections deserve a dedicated optimization
track; ordinary decode is much less expert-dominated than MTP VERIFY.

The deep run actually executes `qsa_indexer_topk`, so this is not merely a large
KV capacity with an inactive QSA branch. At this measured depth, attention/QSA
is still not the leading target. This does not establish its cost at 32k-163k.

### Newly confirmed shared-expert coverage gap

`infr-vulkan/src/adapter.rs::paged_moe_shared_at` accepts routed weights only in
Q5_K, Q6_K or IQ4_XS; all gate/up/down banks must be supported. It also requires
compatible graph shapes and Q8_0 shared weights. The SPIR-V builders in
`infr-vulkan/src/gemm.rs::native_idm_paged_shared_build_spv` have the same limited
format coverage. This model's executed routed banks are IQ2_S/IQ4_NL/IQ3_S, and
the diagnostic contains no shared-slot fused expert kernel.

Thus the existing multi-row shared-slot implementation is **not active for
this quantized model**, in ordinary decode or MTP. Supporting these three formats
could combine more work in an expert dispatch and remove separate shared-path
launches, but this is an unmeasured opportunity, not a guaranteed gain. Add real
shader variants and validate mixed bank formats, Q8 shared packing, masks and
row limits; merely relaxing the recognizer would be incorrect.

### Concrete IQ kernel waste

`native_gemv_id_multi.comp` and its `_sg` twin run `GRID_INIT` before uniform
workgroup bounds/mask checks. In hit-first/miss-second dispatches, inactive
groups can initialize the IQ codebook and then return. Move eligible uniform
checks ahead of initialization without violating barrier participation.
Ordinary decode opens about 18.4-24.0 hit-first windows out of 48 layers/token;
its gain cannot be inferred directly from the more heavily split MTP path.

Then benchmark a persistent read-only codebook buffer and output-row tiling for
the actual IQ2_S/IQ4_NL/IQ3_S shapes. Ordinary `m=1` cannot exploit the cross-token
weight reuse proposed for MTP. Preserve the existing SG exclusions until measured:
`native_id_sg_choice` records poor IQ2_S results for relevant small-output shapes.
Also heed `infr-vulkan/build.rs::gen_grids`: naive dynamically indexed constant
arrays previously caused extreme RADV register spilling. Replacing LDS staging
with such an array is not an acceptable shortcut.

## Host scheduling, PLE and transfer

These counters overlap execution and each other; they are **not additive**:

| Decode counter per token | Short prompt | 1,347 tokens | 9,368 tokens |
| --- | ---: | ---: | ---: |
| Paging synchronization calls | 48 | 48 | 48 |
| Paging synchronization scope, ms | 18.98 | 19.37 | 19.94 |
| Queue submissions | 81.64 | 90.81 | 90.23 |
| CPU time inside queue submission, ms | 1.51 | 1.78 | 1.78 |
| Recorder lifetime, ms | 6.73 | 7.94 | 9.13 |
| Backend setup, ms | 0.96 | 0.69 | 0.59 |
| Timestamped main-queue intervals, ms | 17.30 | 17.48 | 17.77 |
| Host/ReBAR expert push, MiB | 18.50 | 32.47 | 31.73 |
| CPU time inside host/ReBAR push, ms | 1.42 | 2.11 | 2.10 |
| Direct-DMA expert payload, MiB | 39.94 | 59.78 | 57.46 |
| Expert-role lookup hit rate | 93.79% | 90.26% | 90.61% |

The hit rate counts gate/up/down role lookups, not unique experts or the fraction
of model weights resident. Recorder lifetime includes host preparation/copies,
not only Vulkan command encoding. Paging waits include previously submitted
GEMV/attention/router work. DMA has no GPU timestamps in these runs. Neither
19 ms paging nor the difference between 30 ms wall and 17 ms main-queue coverage
can be treated as recoverable idle time. Low reported GPU utilization/power is
consistent with fragmented work, but does not by itself identify the limiting
resource or establish a throughput ceiling.

### PLE still exposes over one millisecond

`infr-llama/src/seam/ple.rs` parallelizes when there are at least 256 gather groups,
or a sufficiently large independent/multi-row batch. The ordinary single-row
16-group job misses that predicate, despite the existing persistent worker pool.
Experiment with bounded two/four-worker single-row gathers and hot-row caching,
targeting exposed PLE below 0.3 ms. That is a test target, not a measured result.
Measure worker dispatch costs, CPU contention and mmap page faults separately;
the present timers do not distinguish storage stalls from gather/dequant work.
Keep cache and worker buffers inside the agreed RAM budget.

### DMA coverage is partial

Zero-based layers 0-3 and 37-47 use host/ReBAR pushes; layers 4-36 have direct DMA
in these runs. First-layer misses are especially relevant because layer 0 also
defines the PLE overlap window. Test demand-weighted host import placement or a
bounded DMA staging ring for uncovered hot layers. A staging ring adds a host
copy and submissions, so it must beat current hit-first CPU-copy overlap in a
paired benchmark, not only increase DMA bytes. The 31.02 GiB import result is not
evidence that all 47.17 GiB can simply be imported on this driver.

The transfer design follows the distinction between discrete-device memory and
host staging described in the [Khronos memory guide](https://docs.vulkan.org/guide/latest/memory_allocation.html).
That guidance does not establish a gain for this existing ReBAR path.

### The structural dependency is still CPU routing

For incompletely resident banks, `adapter.rs::sync_stream` drains preceding GPU
work, reads router IDs on the CPU and prepares residency masks/LUTs. Hit-first
already overlaps miss preparation with resident expert work. These 48 boundaries
remain even if the selected experts for an individual layer happen to be hits.

The [Khronos synchronization sample](https://docs.vulkan.org/samples/latest/samples/performance/wait_idle/README.html)
explains why queue-wide idle waits are coarser than fences. Here, however, changing
the wait primitive alone does not remove the router-to-CPU dependency. A real
next step needs a GPU residency map/hit masks, early router readback signaling,
versioned per-submission LUTs and pinned slots until all readers finish. The
current full drain protects LUT-tape reset and LRU overwrite; preserve those
lifetimes before allowing work to overlap across the boundary.

`runner.rs` explicitly disables `dyn_replay` for this architecture and executes
layer 0, waits for PLE, then executes the remaining layers. These are two backend
executes, not two queue submissions. The outer compile wrapper is not repeated
shader compilation, and its entire build/bind scope is only about 0.3 ms/token.
Generic graph caching is not the main lever. Reusable command segments could
eventually help, but must update positions, recurrent/KV/QSA state and BDA/LUT
metadata correctly. Simply enabling existing full-graph replay is unsafe.

## Feasible delivery plan

| Order | Work and ownership | Acceptance gate |
| --- | --- | --- |
| 1 | Vulkan IQ mask-before-codebook initialization | Identical active expert results; no invalid barrier flow; repeated unprofiled end-to-end gain |
| 2 | Extend shared-slot shaders/recognizer to IQ2_S/IQ4_NL/IQ3_S | Verify fusion actually activates; compare enabled/disabled output and dispatch counts; ordinary and MTP tests |
| 3 | Tune fixed HC/dense GEMV shapes and adjacent fusion; run bounded single-row PLE experiment independently | Shape-specific microbench plus token-wall reduction; PLE exposed wait target below 0.3 ms without extra memory pressure |
| 4 | Improve uncovered hot-layer DMA placement/staging; cache immutable backend layout/scan metadata | Lower exposed transfer/CPU cost, not just fewer allocations or more DMA; preserve fixed 24/48 GiB budgets |
| 5 | GPU hit routing, fine-grained synchronization and reusable command segments | Fewer CPU routing stalls with validated LUT/slot/KV/recurrent lifetimes; separate structural milestone |

Orders 1-3 leave the model graph and unified-pool allocation architecture intact;
they primarily change kernels, backend recognition or the PLE worker policy.
Order 4 changes placement within existing budgets. Order 5 is the significant
pager/executor scheduling change. Small-row backend improvements can also help
MTP, but MTP row reuse and single-row PLE need separate measurements.

At 32.0-35.0 tok/s the unprofiled token budget is 31.25-28.57 ms. Reaching
40 tok/s requires 25 ms/token, saving roughly **3.6-6.3 ms/token (13-20%)**.
Removing all exposed PLE alone cannot close that gap. The two large compute
groups and host orchestration must contribute. These measurements identify
plausible work, but do not yet prove 40 tok/s attainable. A deeper-context target
requires its own unprofiled repeats before assigning a hard savings budget.

For each change, alternate baseline/new at least three times, profiling off,
same model, prompt, token count, device and memory budgets. Retain output hashes
and first-token/steady-state distributions; use identical-prefix replay to isolate
kernel performance when generated tokens diverge. Cover 24/1.3k/9.4k contexts,
then 32k+, QSA thresholds, mixed quant banks, shared-slot masks, ordinary one/two
slots and MTP accepted-prefix states. Run Vulkan validation outside timed runs.
Global pager snapshots are suitable for these isolated runs, not per-request
attribution under concurrency.

## Implemented optimization pass

The following changes were retained after paired measurements:

- Move workgroup-uniform bounds and active-mask checks before `GRID_INIT` in
  both multi-slot native GEMV shaders. A shared Q8 expert slot also skips the
  routed IQ codebook initialization.
- Add IQ2_S, IQ3_S and IQ4_NL shared-expert slot shader coverage. It is selected
  for ordinary one-row decode. The new IQ formats deliberately keep their
  existing dense shared-expert path for multi-row MTP, where the fused slot was
  slower in the measured model.
- Let the existing persistent PLE worker pool process the 16-group ordinary
  one-row gather. This is configured by `kernels.ple_single_parallel` and can be
  disabled with `INFR_NO_PLE_SINGLE_PAR=1`.

Paired 24-token-prompt, 256-output measurements with the same saved pre-change
binary and output SHA-256 were:

| Build / feature set | Decode tok/s |
| --- | ---: |
| Saved pre-change binary, repeats | 34.2 / 34.1 |
| New shader checks only; PLE and shared slot disabled | 34.4 |
| Shader checks plus parallel PLE; shared slot disabled | 35.9 |
| All ordinary optimizations, repeats | 37.1 / 36.1 |
| Final release binary | **37.1** |

The final output SHA-256 remains
`99368f9f6f184aaad21a757395e56e5b636079a7376596ca18ef5ff3ee367b6e`.
Against the paired 34.1-34.2 baseline, 37.1 tok/s is an 8.5-8.8% improvement.

In the pager-profiled A/B, exposed PLE time fell from about 1.27 to 0.030
ms/token, while total PLE worker time fell from about 2.30 to 0.88 ms/token.
The remaining ordinary cost is again dominated by routed experts, dense fixed
projections and the router/pager execution boundary. The implementation does
not alter fixed-resource ordering or the unified-pool budget query.

Verification for the implementation pass: release CLI builds succeeded,
configuration tests passed, output parity was retained, and formatting/diff
checks passed. The linker emitted the existing LIBCMT default-library warning.
