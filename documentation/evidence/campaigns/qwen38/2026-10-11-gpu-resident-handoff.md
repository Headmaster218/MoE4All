# GPU-resident Decode: four-stage implementation and measurements

## Campaign status

All four ordinary-Decode mechanisms are implemented behind default-off switches:
full-hit GPU continuation, rolling command submission, stopped-miss recovery,
and GPU notice / hit-first overlap. MTP is excluded. Earlier frozen versions
completed matched 30K/150K one-active/two-active matrices; subsequent refinements
must be validated independently. The best completed, uncontaminated 30K single-
active screen is 45.01 versus 42.27 tok/s, not the requested 50 tok/s target.
No original checkout, production executable or production KV cache is modified.
The sections below preserve each version's evidence rather than pooling results.

## Initial microbenchmark scope and provenance

- Date: 2026-10-11. GPU: Radeon RX 7900 XTX, Windows driver 32.0.31019.2002.
- Branch: `perf/gpu-resident-decode`, based on fetched main
  `bcd1ec870a01a4de9bc4d164e52e0452550bb757`.
- This is an experimental branch, not a new released engine version.
- Rust 1.97.1, `cargo test --release -p infr-vulkan --lib --no-run`,
  repository native-CPU flags, optimized test executable.
- Executable: `D:\AIinfr\infr\target\gpu-resident-experiment\release\deps\infr_vulkan-91edf65af9d227b0.exe`.
- Executable SHA256:
  `2d0a0edf7f18231ae238368b9e1738ca3fb9ecc762fa4211134fcbbb8e44e440`.
- Source SHA256:
  `b38f4d0847e81aba4ea4529907cfd238ab2c8404c09a7563113839745484cc25`.
  Hash input is the sorted nine changed implementation paths: config, build script,
  adapter, gemm, lib, pager, recorder, and the two new shaders. Each path is followed
  by one NUL byte and its raw file bytes.
- Artifacts: `benchmark-data/artifacts/gpu-resident-handoff-20261011/`.
  `tracked-changes.patch` plus the two copied shader files reconstruct this source
  against the baseline. Patch SHA256:
  `86104382dd6b57dd721e01ba0104246bafe71f9288b3fbef343b99aad77dc49e`.
- The previously running GPU service was allowed to finish; it was not terminated.
  No KV cache was needed, created, read or cleaned in this campaign.

## Measurement definition

GPU timestamps measure the interval from the end of router/top-k to the ordered
start marker before identical representative downstream compute. Thus the interval
includes CPU readback, checking, command recording, queue scheduling and any expert
copy required for continuation, but excludes the preceding router computation and
subsequent downstream computation.

The CPU-hit microbenchmark reads mapped IDs and checks a mirrored address table.
It does **not** include the complete production Pager LRU, full expert FFN recording,
HC, GDN/QSA or the next layer's frontend. These are mechanism timings, not a rerun
of the historical 200+ us production-layer gap and not model tok/s.

One representative expert payload is 1,971,200 bytes (1.88 MiB): IQ2_S Gate/Up and
IQ4_NL Down for ne=2560, nff=640. The copy benchmark uses one contiguous payload,
not three fragmented production cache destinations. Sources are asserted to be
HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, **without DEVICE_LOCAL**; destination
is mapped DEVICE_LOCAL VRAM. Its tail is cleared before every sample and checked
after continuation, so a stale copy cannot pass validation.

Three runs use normal, reversed and normal case order. Each has 16 warmup iterations
and 96 measured samples per case. Table values pool all 288 measured samples.
Validation layers are disabled for performance runs.

## Results

All times are microseconds.

| Rows | Path | Handoff p50 | Handoff p90 | Whole microbenchmark p50 |
| --- | --- | ---: | ---: | ---: |
| 1 | CPU full-hit continuation | 48.25 | 65.46 | 139.20 |
| 1 | GPU conditional full-hit continuation | 8.56 | 9.80 | 80.90 |
| 1 | GPU miss, CPU push, resume | 195.02 | 225.61 | 277.20 |
| 1 | GPU miss, same-queue DMA, resume | 138.35 | 167.10 | 214.10 |
| 1 | GPU miss, dedicated DMA queue, resume | 190.77 | 223.54 | 269.70 |
| 2 | CPU full-hit continuation | 38.36 | 54.50 | 120.60 |
| 2 | GPU conditional full-hit continuation | 9.36 | 10.80 | 93.90 |
| 2 | GPU miss, CPU push, resume | 196.61 | 218.94 | 288.50 |
| 2 | GPU miss, same-queue DMA, resume | 137.05 | 160.35 | 221.60 |
| 2 | GPU miss, dedicated DMA queue, resume | 196.95 | 233.89 | 293.50 |

Full-hit handoff reduction is 82.26% for one row and 75.60% for two rows.
This establishes a useful local mechanism improvement, not an end-to-end Decode gain.
Dedicated DMA does not win this stopped-miss fixture: there is no hit computation
to overlap its extra submission and timeline dependency. This does not invalidate
the existing production hit-first overlap results.

`final-normal-1.log`, `final-reverse-2.log`, `final-normal-3.log` contain raw
`HANDOFF_SAMPLES` JSON records. Analyze with
`node benchmark-data/analyze-gpu-resident-handoff.cjs`.
Earlier non-`final-*` timing logs are provisional and excluded. In particular,
the first staging-source run could measure VRAM-to-VRAM rather than RAM-to-VRAM.

## Prototype implementation

- Opt-in `kernels.vulkan.decode_gpu_resident=true`; default is false.
- `decode_gpu_resident_layers` defaults to 4, clamped to 1..16.
- Narrow ordinary FlashNext-shaped, one/two token-row path. MTP/verify,
  expert prefetch and CPU-miss computation do not use this prototype.
- Optional VK_EXT_conditional_rendering is feature-probed. Unsupported devices
  retain the existing path. Windows and Linux share the same Vulkan implementation;
  Linux has not been runtime-tested here.
- GPU checks the frozen Gate/Up/Down address windows. Full hits proceed without
  CPU routing readback. The first miss clears the run word; subsequent computation
  and state writes in the recorded chunk are predicated off.
- GPU journals actual routed IDs. CPU replays completed-hit LRU touches in order
  after the chunk drains and before promoting the paused layer.
- The existing allocator and eviction policy are retained. Resident address slots
  are pinned until GPU completion; this is not a GPU-owned linked-list allocator.
- Resume uses the missed router's retained IDs, weights and shared-gate result,
  starting at its FFN, not its frontend. Main KV/GDN state is not recomputed.
- Plain Vulkan copies/fills are not conditional. Inside predicate regions they
  become compute copies/fills, so paused later layers cannot update checkpoints.
- The first version reports a miss at a bounded chunk completion, not via a
  persistent GPU mailbox or a CPU busy-poll loop. New GPU hit-first overlap is deferred.

## Verification and self-review

- infr-core: 239 default tests passed.
- infr-vulkan: 209 default tests passed; 106 GPU/explicit tests ignored by that run.
- Conditional-state GPU fixture: one/two rows, all hit, first/middle/last miss,
  each missing role; skipped copy/fill and exactly-once frontend resume checked.
- Complete two-layer graph fixture: ordinary and GPU-resident modes, one/two rows,
  all hit and either missed layer. Uses real IQ2_S/IQ4_NL expert kernel geometry
  with zero-scaled synthetic weights and in-place frontend counters. This checks
  adapter recovery and checkpoint preservation, not real-model numerical quality.
- Both explicit GPU correctness fixtures passed with Khronos validation enabled.
  Loader messages about stale Epic overlay registrations and duplicated AMD layers
  are host installation warnings, not a clean-loader claim.
- Explicit shader/transfer-write to HOST_READ publication precedes journal reads.
- Self-review removed redundant empty tail submits and moved host LUT visibility
  to one barrier per chunk rather than one global barrier per checked layer.

## Next decision

The local continuation mechanism is worth testing in the full model. Keep it opt-in
until matched ordinary Decode tests establish real benefits and correctness:
two slots with one active, then two active, 30K and 150K, including natural misses.
Choose chunk size from those measurements.

CPU encoding is still done before each recorded chunk; a smaller intra-GPU gap
does not make that host work disappear. LUT snapshot/pinning, journal replay,
recording skipped commands and miss recovery may offset part of the gain.
If these dominate, evaluate retained LUT/control buffers and cached command plans
before extending the prototype to GPU-owned LRU or hit-first mailbox protocols.

## Stage 1 real-model pilot

FlashNext 4.27 bpw, Vulkan1/RX 7900 XTX, two configured slots with one active,
30K ordinary technical prompt, Q8 KV, 160K context limit, ubatch 4096, VRAM 24g,
RAM 48g, 10 CPU threads, greedy sampling, MTP/prefetch/CPU miss compute disabled.
Each request generated 128 tokens; the first round is a cache/startup warm round.

| Path | First round tok/s | Warm round tok/s |
| --- | ---: | ---: |
| Existing CPU routing | 37.80 | 39.97 |
| GPU continuation, chunk 4 | 32.92 | 34.39 |

All four outputs had SHA256
`6af78aa12261fd4d0dd5b0d9f4ffe2a8dfef77b55ae53dc6018c059604bf7781`.
The first prototype is numerically consistent on this pilot, but regresses overall.
These short pilot rates are not the final 30K/150K matrix or a 50 tok/s claim.

Stage-1 binary SHA256:
`b43043b35d21d07756eaa9709aa862ed4e0b5eb1fb14674ad7d49f02073b749c`.
The source patch and two shader copies captured as `stage1-pilot-*` precede
stage-2 edits. Later pilot manifests captured the then-current working source,
not the source used to build this immutable stage-1 binary; use the original
patch/shaders and binary identity, not those later working hashes, for provenance.

A separate profiled 256-token/two-round run captured 192 forward samples:
mean wall 33.17 ms, GPU main 15.95 ms, main pending 5.68 ms, CPU command encode
3.46 ms (only 0.47 ms overlapping main), lowering/management 6.04 ms, and
queue-idle waiting 15.84 ms. Nested waits overlap device work; these are not
additive wall-time components. Profiling itself changes timing.

Raw results and trace are under
`benchmark-data/artifacts/gpu-resident-decode-20261011/`; test KV remains in
`D:/AIinfr/infr/test-kv-cache`, outside artifacts and production KV.
SSE text frames omit some generated tokens, so exact front/middle/tail token
rates cannot be inferred from frame counts in these runs; server final rates
remain the authoritative whole-generation metric.

## Stage 2 and 3 candidates

- Reuse predicate/journal buffers within the Decode runtime phase; release them
  with the existing phase workspace, not through a shared-device ownership cycle.
- `decode_gpu_resident_submit_layers`: submit bounded completed layer groups
  while recording subsequent groups. Zero retains stage-1 submission behavior.
- `decode_gpu_resident_inflight`: bound queued groups to one/two. A shared GPU
  predicate remains live across submissions; new groups never reset it on the CPU.
- `decode_gpu_resident_mapped_metadata`: independent mapped-VRAM predicate/LUT
  A/B, with host-memory fallback and the predicate-specific Vulkan usage flag.
- `decode_gpu_resident_fence_wait`: drain a resident group to its own fence rather
  than queue idle. Error/panic paths still drain before reclaiming readers.

All candidates remain opt-in and are pending matched real-model measurements.

### Stage 2/3 pilot outcomes

Binary `infr-stage23.exe`, SHA256
`ba7cf000b6d5a4d638d86089652aa1560d11abbe64e9ed983892f3c6a0d0f93c`.
Same 30K/one-active/two-slot/128-token/two-round protocol as the first pilot.
The fixed source snapshot was captured before any subsequent stage-4 edits.

| Variant | First round tok/s | Warm round tok/s |
| --- | ---: | ---: |
| Existing CPU routing | 39.43 | 41.07 |
| Chunk 2, submit each layer | 37.99 | 40.02 |
| Above + mapped metadata | 38.46 | 39.98 |
| Chunk 4, submit each layer, mapped metadata | 36.27 | 37.77 |
| Chunk 2, submit each layer, mapped metadata + fence drain | 35.70 | 38.27 |

Every output matched the stage-1 output hash. Rolling a two-layer chunk removes
most of the first-version regression, but no candidate beats the same-binary
baseline. Mapped metadata and fence draining are not promoted to defaults.
Short pilots are parameter screening, not final performance claims.

The profiled stage-1 run recorded about 101 expert-layer encoding spans per
48-layer forward, including resumed and skipped speculative layers. Reducing
unneeded recording and LUT snapshot/protection work is therefore still relevant.

## Stage 4 candidate (unit validated, model measurements pending)

- GPU computes per-row complete-triplet hit/miss masks and an immutable journal
  record; a per-layer Vulkan event publishes that record to the CPU.
- A second predicate allows actual hit/shared expert computation after the first
  miss stops later layers. Only the ordinary run predicate guards accumulation
  and later frontend/state writes.
- CPU pins actual hit triplets before admission, starts ordinary expert upload,
  then drains GPU and transfer readers before resuming only the missing FFN slots.
- Event reset/reuse is bounded by completed chunks. Predicate, journal and event
  lifetimes follow the existing Decode phase workspace; MTP remains excluded.
- An independent fast-LUT candidate caches immutable physical coordinates and
  replaces per-present-slot hash insertion with dense protection flags. The
  ordinary allocator, CPU LRU and unified-claim protection remain authoritative.

These are source candidates only. GPU correctness, validation layers, actual
upload overlap and matched model rates remain required before adoption.

### Correctness checks

- 239 infr-core and 211 default infr-vulkan unit tests passed; 106 explicitly
  ignored GPU/specialized cases are not included in that count.
- Two explicit GPU fixtures passed with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
  and `VK_LAYER_VALIDATE_SYNC=1`. The clean rerun is recorded in
  `stage4-sync-validation.log`; it contains no validation diagnostics.
- The two-layer adapter fixture now uses nonzero IQ2_S/IQ4_NL expert weights,
  keeps the preloaded top-ten routing bank, and compares final state against the
  ordinary path with a 1e-5 absolute/relative tolerance. It covers one/two rows,
  full hits, first/second-layer misses, rolling submission, mapped metadata,
  fence drains and event/hit-first combinations.
- Separate predicate/copy/fill coverage still verifies skipped state writes and
  exactly-once frontend recovery. CPU tests reject invalid/stale notice status,
  expert IDs, mask bits, all-hit miss notices and invalid row shapes.
- CPU polls notices periodically during rolling encoding. A notified miss cancels
  only unsubmitted commands that would have been predicated off; submitted hit
  work and upload dependencies are retained and drained before recovery.
- Self-review fixed mapped-predicate buffer usage, an event/fence recheck race,
  and the recorder's output-binding count for read/write ID masks. The check uses
  conservative write tracking; masked SiLU marks only its true output as written.

These fixture results are not proof of full-model speed, long-context stability,
Linux runtime parity, or achievement of the 50 tok/s target.

### Stage 4 screening

Immutable executable `infr-stage4.exe`, SHA256
`fb618ac5f8550955026337cbdbaa9a2c245387c7013980d62bc757b92d9b68a0`.
The `stage4-pilot-manifest.json` and source patch capture the exact working source.
Same 30K, two configured slots/one active, 128 output tokens and two rounds as
Stage 2/3; these short runs screen candidates, not final long-context results.

| Candidate | Cold tok/s | Warm tok/s |
| --- | ---: | ---: |
| Ordinary path | 38.40 | 40.20 |
| Two-layer / rolling one / fast LUT | 38.20 | 39.99 |
| Two-layer / rolling one / fast LUT / notices | 38.03 | 39.40 |
| Four-layer / rolling one / fast LUT / notices | 35.89 | 37.26 |
| Eight-layer / rolling one / fast LUT / notices | 35.21 | 37.35 |

All output hashes equal `6af78aa12261fd4d0dd5b0d9f4ffe2a8dfef77b55ae53dc6018c059604bf7781`.
No candidate is adopted as a default. Aggregate cache size stayed below 30 GiB;
the optional temporary 100 GiB allowance was not needed. Matched diagnostic
traces are needed to separate speculative encoding, publication and recovery.

### Matched Stage 4 diagnosis

Same Stage 4 binary, 30K/two slots/one active, 192 output tokens, two rounds.
`stage4-diagnostic-*` captures 24 sampled forwards per mode; tracing perturbs
performance, so these are decomposition numbers rather than adoption rates.

| Mean per forward | Ordinary | Two-layer notice |
| --- | ---: | ---: |
| Wall ms | 27.96 | 28.31 |
| Main queue envelope ms | 15.26 | 16.52 |
| Main pending union ms | 7.67 | 10.63 |
| Command encoding ms | 1.98 | 2.49 |
| Router readback ms | 0.141 | 0.003 |
| Main submits | 120.04 | 88.54 |
| DMA submits | 15.58 | 15.58 |
| DMA bytes | 44,595,200 | 44,595,200 |
| DMA/main overlap ms | 0.489 | 0.862 |

The notice path records 84.67 expert-lowering spans per forward, including skipped
speculation and recovery, versus 48 ordinary spans. There are 21.5 miss notices
per forward. LUT freezing is only 0.308 ms/forward. Raw notice wait spans total
8.893 ms/forward; the existing analyzer groups them into `lowering_other`, not
CPU computation. Do not add this nested/raw duration to the CPU partition.
Queue pending intervals overlap CPU and GPU activity and are not an additive
wall-time budget either. The driver-submission-delay hypothesis remains unproven.

A default-off dedicated host fence-waiter candidate tests whether explicit waits
on another thread change driver scheduling while the main thread keeps encoding
and processing notices. Pending segments retain every fence until the worker
returns; no reset/recycle is allowed during the worker's wait. This is a portable
Vulkan experiment, not a Windows-only API or a claimed scheduling fix.

### Recovery and wait-producer follow-up

- `decode_gpu_resident_waiter` remains default off. With the existing
  `decode_gpu_resident_fence_wait=true`, its worker waits only the submitted fence;
  false selects a queue-idle comparator that takes the ordinary queue gate and
  may delay later submissions. Neither mode is presumed faster.
- A notified miss now retains the upload ticket and binds its transfer timeline
  before encoding the miss consumers. CPU upload completion is checked before the
  next residency mutation/final drain, rather than before encoding. Actual hit
  slots are not recomputed, nor are router, frontend, main KV or recurrent state.
- Pending upload destinations retain exact unified-slot protection after old
  speculative LUT protection is cleared. Ordinary LRU pinning alone is not used
  as a substitute for protection from runtime scratch claims.
- Prefill-marked graphs are explicitly excluded, including singleton tail chunks.
- Expanded nonzero-weight GPU fixtures cover one/two-layer chunks, one/two rows,
  full hits and either-layer misses, direct push and enabled imported-host DMA,
  mapped metadata, rolling submission and both host-waiter modes. They passed
  with synchronization validation; `stage4-recovery-sync-validation.log` records
  two explicit GPU tests passing in 26.05 seconds without diagnostics. The default
  239 core and 211 Vulkan tests also passed.
- The case runner is now checked in with the wrapper. The wrapper checks every
  expected case/round and compares candidate hashes to the matching baseline;
  SSE frame counts are still not mislabeled as token counts.

Immutable recovery executable `infr-stage4-recovery.exe`, SHA256
`49622a6dd180be6d0b37f5c3494f84679f961ccf963bbeddc0944bc4086897a9`.
The source patches and manifests named `stage4-recovery-pilot` and
`stage4-l2-tuning` identify its frozen runtime sources. Both screens use 30K,
two configured slots/one active and two rounds, without GPU profiling.

| Recovery pilot (192 output tokens) | Cold tok/s | Warm tok/s |
| --- | ---: | ---: |
| Ordinary | 40.53 | 42.69 |
| Two-layer / rolling one / fast LUT / notices | 40.92 | 42.46 |
| One-layer / fast LUT / notices | 38.44 | 39.65 |
| One-layer / fence worker | 38.10 | 38.75 |
| One-layer / queue-idle worker | 37.99 | 38.67 |
| One-layer / fence worker / mapped metadata | 38.15 | 39.06 |

All six modes have the same 192-token output hash. One-layer boundaries and the
host wait producer do not independently establish an end-to-end gain. Shorter
chunks reduce speculative encoding, but can also interrupt useful GPU work.

| Two-layer tuning (256 output tokens) | Cold tok/s | Warm tok/s |
| --- | ---: | ---: |
| Ordinary | 41.73 | 42.39 |
| Two-layer / mapped metadata | 41.88 | 41.63 |
| Two-layer / fence worker / mapped metadata | 41.44 | 42.77 |
| Three-layer / mapped metadata | 42.22 | 43.93 |

All completed tuning modes match the baseline output hash
`f419e20a5bf2966e8ddbdc0121d61c80d1082fb7ae02e13985b228c0c69b6eb6`.
The small warm difference in the fence-worker combination is a screening result,
not a statistically established improvement. No candidate is enabled by default.
The longer one/two-active, 30K/150K matrix is still required. No 50 tok/s claim
is made from unit tests or from the earlier single-layer microbenchmark.

### Immutable predicate follow-up

An additional default-off `decode_gpu_resident_immutable_predicates` candidate
gives each checked layer separate continue/hit predicate words. The initial
predicate and each layer predicate remain unchanged until their readers drain.
This removes the conditional-read-to-shader-write dependency at block endings,
but retains shader-write-to-conditional-read publication and ordinary data
hazards. Mutable predicates retain the original read/write barrier behavior.
Control storage grows from 20 to 152 bytes; LUT, expert and KV storage is unchanged.

The explicit conditions and nonzero-weight recovery fixtures passed under
synchronization validation in 0.58 and 26.91 seconds, with no diagnostics. The
expanded graph fixture includes immutable predicates with direct push/DMA,
one/two rows, first/second-layer misses, full hits, rolling submits and mapped
metadata. Default core/Vulkan tests passed again (239 + 211). This is correctness
coverage, not evidence of a throughput gain or of the 50 tok/s target.

### Complete immutable-predicate matrix

Frozen executable `infr-stage4-immutable.exe`, SHA256
`6c481eba1ce88c3122cb522aba2674c3ad60303a08f09bfd2424312179e8ae0b`.
The `stage4-complete-matrix` source patch and shader copies describe this binary,
not subsequent working-tree changes. Both modes configure two slots, disable MTP
and generate 512 tokens for three rounds. The table averages warm rounds 1/2;
rates are whole-request Decode rates, not Prefill or SSE frame rates.

| Context / active slots | Ordinary tok/s | Three-layer immutable tok/s | Change |
| --- | ---: | ---: | ---: |
| 30K / one | 42.77 | 43.57 | +1.9% |
| 30K / two, per lane | 31.95 | 30.79 | -3.6% |
| 30K / two, aggregate | 63.90 | 61.59 | -3.6% |
| 150K / one | 39.25 | 40.20 | +2.4% |
| 150K / two, per lane | 24.38 | 23.68 | -2.9% |
| 150K / two, aggregate | 48.75 | 47.36 | -2.9% |

All eight services completed and exited gracefully. The original wrapper
reported a hash mismatch because it compared corresponding round numbers:
ordinary Decode itself produced different cold/cached output hashes in some
cases. The posthoc `stage4-complete-matrix-matched-cache-review.json` compares
complete batches with the same cache-count, prompt-count and output-count vectors.
All 12 candidate batch rounds match an ordinary batch in that same state. This
does not prove cold/cached equivalence or explain the baseline's variation.
The checked-in verifier now enforces joint batch comparisons and rejects lane
mixing, missing comparable states and genuinely changed output.

The experiment remains default off: it does not meet 50 tok/s and regresses
two-active throughput. Cache usage remained under the aggregate 30 GiB budget;
the authorized temporary 100 GiB limit was not needed.

### CPU notice producer

A separate default-off `decode_gpu_resident_notice_thread` candidate polls
already-submitted layer events and prepares/uploads notified misses while the
main thread encodes. It retains segment fences until worker acknowledgement,
protects both actual hit slots and pending miss destinations, and drains uploads
and abandoned admissions before releasing frozen LUT protection. Unsubmitted
speculation is never queued to the worker. Device dispatch caps remain honored
at rolling submission boundaries. These changes are not represented by the
immutable matrix above. The 239 core / 211 default Vulkan tests passed. Explicit
nonzero graph recovery (34.33 seconds) and predicate-state (0.40 seconds) tests
passed with synchronization validation and no diagnostics. The graph fixture
uses disjoint routes for two rows, so a missing expert affects only one lane;
it covers both layers, no misses, the notice producer and a one-dispatch submit
cap. Its completed end-to-end measurements are recorded below.

Producer executable `infr-stage4-producer.exe`, SHA256
`73ad1e57ee0d8aeb7504d0355ab7d88a90bf128d80062b3a3b36be993bbb19f3`.
The `stage4-producer-pilot` and `stage4-push-pilot` manifests freeze its sources.
All 256-token output hashes match ordinary Decode in the same cache state.

| 30K, two configured / one active, 256 tokens | Warm whole-request tok/s |
| --- | ---: |
| Ordinary / Host DMA | 42.73 |
| Three-layer producer / Host DMA | 42.78 |
| Two-layer producer / Host DMA | 42.90 |
| Ordinary / CPU push | 41.25 |
| Three-layer producer / CPU push | 42.73 |

The worker improves CPU-push throughput relative to its own baseline, but does
not beat Host DMA at an established margin. This screen does not establish a
50 tok/s result.

The completed `stage4-producer-matrix` repeats the full 512-token / three-round
matrix. All eight services completed and all 12 candidate batch rounds match
ordinary outputs at the same complete-batch cache state. Warm-round means:

| Context / active slots | Ordinary tok/s | Producer tok/s | Change |
| --- | ---: | ---: | ---: |
| 30K / one | 42.59 | 43.51 | +2.2% |
| 30K / two, per lane | 31.85 | 30.37 | -4.7% |
| 30K / two, aggregate | 63.70 | 60.73 | -4.7% |
| 150K / one | 39.30 | 39.61 | +0.8% |
| 150K / two, per lane | 24.65 | 23.44 | -4.9% |
| 150K / two, aggregate | 49.29 | 46.89 | -4.9% |

Aggregate test-cache usage after the matrix is 20,916,434,288 bytes. A read-only
Windows sample during the final 150K case reported about 5.8 GiB free physical
memory and 1.1 GiB free commit headroom. This is a point sample, not evidence
that memory pressure caused the throughput changes; forced 48 GiB RAM budgets
and commit-headroom warnings were shared benchmark conditions.

### Indirect predicates

Another default-off `decode_gpu_resident_indirect_predicates` candidate converts
each logical conditional region into GPU-produced indirect dispatch dimensions.
Zero x workgroups skip computation; nonzero preserves the original x/y/z grid.
It does not move LRU ownership or change expert/KV lifetimes. Region descriptors
are finalized before each submission and reused only after all GPU readers drain.
Shader writes are ordered before indirect command reads, and prior indirect
readers before subsequent producer writes. The same existing compute versions
of conditional copies/zeroes are retained. Pre-existing indirect dispatches are
explicitly rejected inside this experiment, rather than silently running them.

The bounded tape adds 160 KiB host metadata and 96 KiB device dispatch storage
per active phase. Its purpose is to isolate conditional-rendering command costs;
there is no claimed gain yet, and the producer matrix uses the earlier executable.

### Early-miss empty-tail removal

An early GPU notice discarded the unsubmitted speculative tail, then constructed
and submitted an empty replacement solely to wait for completion. Recovery now
drains the already-submitted segment fences instead. The notice's existing host
publication additionally covers the four final control/status words, so removing
the empty tail does not remove its host-read visibility guarantee. Exact miss
destination protection and the resumed consumer's DMA timeline dependency remain.
This change is not part of the producer measurements above.

Both refinements passed the nonzero, independently routed one/two-row recovery
fixture with Vulkan synchronization validation (82.88 seconds, no diagnostics).
The fixture covers both predicate mechanisms, full hits, misses in either layer,
rolling submissions, Host DMA/CPU push, notice producers and dispatch cap one.
The predicate-state test passed separately (0.42 seconds). Default tests passed:
239 core, 211 Vulkan, and four benchmark verification tests. End-to-end throughput
of these refinements is pending; the producer matrix is not their performance
evidence.

The completed `stage4-empty-indirect-pilot` uses executable
`infr-stage4-empty-indirect.exe`, SHA256
`5d15fded085d56dca18d8b788c8843ee2315c3fe20bb750cef4fafeb270649c6`.
Each 30K, two-configured/one-active case generates 512 tokens in three rounds;
all 12 candidate outputs match the ordinary complete-batch cache state. Means of
rounds one and two (round zero excluded):

| Candidate | Warm whole-request tok/s |
| --- | ---: |
| Ordinary | 42.58 |
| Empty-tail removal, three layers | 43.61 |
| Indirect predicates | 42.87 |
| Empty-tail removal plus notice producer | 43.74 |
| Indirect predicates plus notice producer | 42.48 |

Indirect predicates are not adopted: they do not beat the existing conditional
rendering path. The best screen is about +2.7%, not a 50 tok/s result. Final cache
usage is 18,599,738,192 bytes, still below 30 GiB.

`stage4-current-diagnostic` then captured 24 sampled forwards per case using
192-token/two-round requests, ordinary versus three-layer empty-tail removal
(producer disabled so its worker cannot disappear from trace accounting). Single
stream samples use identical position vectors. Profiler-enabled mean wall is
28.40 -> 27.50 ms; main-queue envelope 15.27 -> 16.68 ms; main submissions
120.04 -> 105.17; DMA/compute overlap 0.44 -> 1.11 ms. DMA payload remains
44,595,200 bytes per sampled forward. These envelopes include dispatch/synchronization
cost, not isolated kernel compute; pending intervals overlap and are not additive.
The analyzer is the unchanged repository `benchmark-data/analyze-decode-timeline.cjs`,
SHA256 `1fcde829783746562d15d8a0a9a336dc551068e54844ac4dae61934c28fe12e6`.

Do not derive a two-active gain from that diagnostic: its ordinary first round
had a cold second prefix, and sampled position vectors consequently differ from
the candidate. Both batches match ordinary output at the appropriate cache state,
but the profiled expert workloads are not matched. Whole-request unprofiled
matrices above remain the throughput evidence.

### Mapped journal and window screen

`decode_gpu_resident_mapped_journal` independently selects coherent mapped VRAM
for routing journals; ordinary host Readback is retained on allocation failure.
The allocator's GpuToCpu policy otherwise prefers HOST_CACHED memory, unlike the
DEVICE_LOCAL metadata path. This is a placement candidate, not evidence that PCIe
is the bottleneck. After observing the Vulkan event and its HOST_READ publication,
CPU readers copy the stable 128-byte journal range together instead of 32 scalar
volatile loads. Journal reuse still waits every GPU reader and producer acknowledgement.
No ordinary Decode, Prefill, MTP or expert-residency policy changes. The expanded
GPU fixture passed synchronization validation in 121.63 seconds; default tests
passed (239 core / 211 Vulkan). Earlier tables use earlier frozen binaries.

The completed `stage4-journal-window-pilot` uses `infr-stage4-journal.exe`, SHA256
`4141da2f27ad83db52d3fa3587b2d29eee70bad363687f3fe5a3b23372e0cc39`.
All 15 candidate batches match ordinary output. These are 30K, two-configured/one-
active, 384-token/three-round runs; means of the two warm rounds:

| Journal / checked-layer window | Warm whole-request tok/s |
| --- | ---: |
| Ordinary | 42.54 |
| Host / 3 | 43.47 |
| Mapped VRAM / 3 | 43.82 |
| Host / 8 | 44.48 |
| Mapped VRAM / 8 | 44.68 |
| Mapped VRAM / 16 | 44.93 |

Every resident candidate uses the notice producer, immutable predicates, mapped
metadata, fast LUT and one-layer rolling submissions. A larger window helps this
screen, but neither journal placement nor window size establishes 50 tok/s. The
largest window is not yet a selected 150K/two-active default. Cache usage after
the screen is 20,049,299,084 bytes.

### Earlier CPU speculation cancellation

The producer previously returned its first-miss result only after preparing the
RAM copy/DMA. It now publishes a host atomic cancellation ordinal immediately
after observing the GPU event, before upload preparation. The main encoder can
discard the unsubmitted, predicated-off tail sooner. Upload ownership still arrives
through the original result channel; recovery collects that result and drains
existing GPU fences/producer acknowledgements before resetting event/journal state.
An announcement without a prepared recovery result fails rather than silently
dropping computation. The ordinal must refer to submitted work. Unit coverage
checks empty/no-announcement, every bounded ordinal, future-work rejection and
overflowing journal ranges; the GPU fixture ends with a full-hit forward after
both miss cases to catch stale cancellation state.
This is a CPU protocol refinement, not a GPU memory-model change or a performance
claim. The preceding screen uses the older producer protocol.

Core tests (239) and Vulkan default tests (213 passed, 106 GPU tests ignored)
passed. The updated GPU fixture passed synchronization validation in 134.62
seconds, including a full-hit forward after each miss sequence. Frozen executable
`infr-stage4-early-stop.exe` has SHA256
`0cc6eb714b94f35194e8cef7c14288983fc986dd2a3187a3f6543931e8b45599`.
All six candidate batches in `stage4-early-stop-pilot` match ordinary output.
Do not use that pilot to choose speed parameters: compilation overlapped its
candidate requests. The executable includes every behavioral change; a later
source edit only wrapped an error string for formatting.

### Pager lock scope during resident encoding

Resident encoding previously held the pager mutex through Gate/Up/SiLU/Down
command recording. The producer needs the same mutex to replay actual hits and
prepare misses, creating avoidable CPU serialization. It now snapshots already-
frozen bank addresses and retains the same LUT buffer with an Arc, then releases
the mutex before encoding. LUT protection still lasts until all GPU readers drain;
no LRU ownership, unified allocation or arithmetic changes. This adds one buffer
reference, not another allocation or backend ownership cycle. Default tests pass.
The GPU fixture now covers producer rolling groups of one and two layers without
an explicit dispatch cap; a separate one-layer case keeps cap-one coverage.
The expanded fixture passed in 114.59 seconds with synchronization validation and
no validation errors. Layer-setting deprecation warnings are not synchronization
findings. Frozen executable `infr-stage4-lock-snapshot.exe` has SHA256
`b75a656701387186a266e0cb2be93642b3cf91abe86b329954c219f2e66ff42e`.

The harness also permits `RESIDENT_SLOTS=1` for a genuinely single-configured-slot
comparison. The default remains two; its one-active/two-active measurements must
not be relabeled as a single configured slot.

The completed `stage4-lock-snapshot-submit-screen` uses the frozen snapshot
executable above. Each case has 384 generated tokens and three rounds at 30K;
all 12 candidate batches match ordinary output. No compilation or GPU fixture
overlapped these requests. The checked-layer window is 16 throughout, with the
producer, hit-first, immutable predicates, mapped metadata/journal and fast LUT:

| Layers per rolling submission | Warm whole-request tok/s |
| --- | ---: |
| Ordinary | 42.27 |
| 1 | 45.01 |
| 2 | 43.02 |
| 4 | 38.52 |
| 8 | 34.09 |

Means exclude round zero. Larger rolling groups save submissions but delay
starting their GPU work/notices and increase useless encoding after a miss;
their measured whole-request result is worse. This experiment does not separately
measure the causal contribution of either cost. Select one-layer rolling for the
next matched 512-token/three-round 30K/150K, one-active/two-active matrix.
The local implementation commit is `098df3535`; the EXE predates only test-case
edits and the commit itself, not any runtime behavior in that commit. Cache usage
after the screen is 21,465,780,972 bytes; no temporary budget increase was needed.

## Latest complete matched matrix

`stage4-final-matched-matrix` completed all eight services, 512 generated tokens
per request and three rounds. Binary SHA256 is the snapshot EXE above. Every
one of the 12 candidate batches matches the ordinary entire batch's cache state
and output hashes; compare joint batches, never individual lanes from different
reference rounds. All measurements below use the two warm rounds, excluding
round zero. Two slots are configured throughout; the candidate is window 16,
rolling one layer, with the full producer/hit-first/mapped/immutable/fast-LUT set.

| Context / active slots | Ordinary whole-request per-lane tok/s | Resident per-lane tok/s | Change | Ordinary steady aggregate | Resident steady aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30K / 1 | 42.98 | 44.61 | +3.8% | 43.00 | 44.57 |
| 30K / 2 | 31.83 | 30.92 | -2.9% | 63.54 | 61.88 |
| 150K / 1 | 39.84 | 40.67 | +2.1% | 40.02 | 41.20 |
| 150K / 2 | 24.65 | 23.44 | -4.9% | 49.28 | 46.81 |

Steady aggregate uses the existing server's fully-contained active Decode windows,
weighted by measured duration, not the sum of rounded per-lane rates. Single-active
numbers in that column likewise exclude partial boundaries. Front/middle/late are
token-midpoint thirds from the server's existing one-second counters; each cell
below is the mean per-lane rate over warm requests, not independently profiled GPU
kernel time. Existing counters round to 0.1 tok/s.

| Context / active slots | Ordinary front / middle / late | Resident front / middle / late |
| --- | --- | --- |
| 30K / 1 | 42.05 / 43.63 / 43.02 | 43.53 / 45.87 / 44.33 |
| 30K / 2 | 31.95 / 32.39 / 31.08 | 31.24 / 31.38 / 29.90 |
| 150K / 1 | 38.77 / 40.59 / 39.90 | 38.94 / 41.71 / 41.07 |
| 150K / 2 | 25.58 / 23.93 / 24.35 | 24.13 / 22.84 / 23.23 |

The four runtime stages and matched correctness validation are complete, but the
50 tok/s whole-request single-stream target is NOT achieved by this matrix.
Keep the master switch disabled by default; the two-active result does not justify
a universal enablement. The microbenchmark's large local win is not an end-to-end
speed claim. This matrix ends with 20,916,434,288 cache bytes, below 30 GiB.
The guard added after this EXE only checks test eligibility, to prevent unsupported
hardware from passing the GPU fixture through ordinary fallback; it changes no
runtime behavior and will be validated separately.

The guard's expanded GPU fixture subsequently passed synchronization validation
in 113.40 seconds, with no validation errors and no deprecated layer settings.
The final default suite passes 239 core / 213 Vulkan tests (106 GPU tests ignored).

### Truly one configured slot

`stage4-true-single-slot` uses the same snapshot binary and 30K messages, 512
tokens, three rounds, but configures exactly one slot. All three candidate batches
match ordinary cache state and output. Ordinary rounds are 42.72 / 43.34 / 43.67
tok/s; resident rounds are 45.16 / 43.98 / 46.38. Warm means are 43.50 -> 45.18
(+3.9%). The 2.4 tok/s spread between the candidate warm rounds is material;
do not report the best round as its mean or claim 50 tok/s. Spare-slot allocation
does not explain the remaining target gap in this sample. This is distinct from
the two-configured-slot matrix, not a replacement for it. Cache bytes remain
20,916,434,288. The authorized temporary 100 GiB limit was never used.

## CPU-only timing diagnostic

`stage4-cpu-only-diagnostic` uses `infr-cpu-only-timeline.exe`, SHA256
`b6384b9f966a087d50e1a6454d15162dd85e539e247de72524a01ca10dfd92a0`,
two configured slots, one active, 30K, 384 generated tokens and three rounds.
The new default-off `prof.timeline_cpu_only` disables timeline device queries
and calibrated-clock probing, not the engine's ordinary submit tuner. Each trace
contains 12 forwards at exactly positions 30103 through 30114, with zero device
events/clock samples and `device_timestamps=false`. This avoids extra timeline
GPU commands but CPU span instrumentation still costs time. Do not use these
profiled rates as an unprofiled throughput result.

| Main-thread span per sampled forward | Ordinary ms | Resident ms |
| --- | ---: | ---: |
| backend_execute (both main and output head) | 22.1590 | 20.0260 |
| backend_setup | 0.4760 | 0.3904 |
| command_encode | 1.6992 | 2.8253 |
| queue_submit | 2.3075 | 1.4801 |
| recorder_acquire | 0.2652 | 0.2555 |
| gpu_resident_lut_freeze | 0 | 0.3505 |
| gpu_resident_notice_producer_wait | 0 | 1.0473 |
| fence_wait | 0.1012 | 4.1625 |
| queue_idle_wait | 13.2873 | 0.7131 |

Spans are nested: this table is not a disjoint partition and must not be summed.
Blocking waits are not CPU execution. The notice producer runs without the main
thread's timeline context; its own prepare/upload costs are absent, so reduced
main-thread expert_prepare does not mean all preparation work disappeared.
Without device events these spans do not establish the GPU critical path.

Resident averages 216.5 LUT freeze calls (three per encoded MoE layer), implying
72.17 encoded layers versus 48 executed model layers. Speculative recording after
a miss is a concrete candidate for reduction. Main queue_submit calls average
89.25 versus ordinary 111.33; recorder_acquire averages 102.42 versus 111.33.
The 0.35 ms aggregate LUT freeze and 0.39 ms setup do not justify assuming either
is the dominant remaining bottleneck. First screen the existing inflight bound
one against two, with profiling disabled, before changing ownership or replay.

`stage4-inflight-screen` subsequently completed using the same B638 binary with
profiling disabled. Ordinary warm mean is 42.7422 tok/s; inflight two is 43.7357,
inflight one 44.8735 (+2.6% versus two). All six candidate batches match ordinary
cache state and output; all services exited normally. This is a small screening
result, not proof of 50 tok/s or of improved dual-active/150K behavior. Final cache
usage is 20,466,326,552 bytes; the 100 GiB exception remains unused.

## Rejected asynchronous main-queue submit experiment

A separate FIFO submit worker moved driver calls out of command encoding. Each
pending segment retained its command/pools/buffers until both a successful submit
receipt and GPU fence completion. Notice/fence workers waited the same receipt;
failed enqueue/submit never waited an unsignaled fence. A host panic drained the
queue and made subsequent submits fail before releasing its command storage.
The optional worker was disabled during timeline capture to avoid falsely marking
an enqueued command as submitted. It did not change shader arithmetic or LRU.

Core 240 / Vulkan 215 default tests passed, including multi-observer and abandoned
receipt tests. The expanded actual GPU fixture, with asynchronous submit and an
inflight bound of one, passed synchronization validation in 113.61 seconds. The
frozen executable `infr-submit-worker.exe` has SHA256
`ff53e2e2eb25168f45e71bcbdee413d169f4dc413529d51a85677a70c79a9ec2`.

`stage4-submit-worker-screen` completed all four cases on that same executable,
30K, two configured/one active, 384 tokens, three rounds, no profiling or build
overlap. All nine candidate batches match ordinary cache state and output.

| Variant | Warm whole-request tok/s |
| --- | ---: |
| Ordinary | 42.6036 |
| Synchronous submit, inflight one | 45.4462 |
| Asynchronous submit, inflight one | 44.6494 |
| Asynchronous submit, inflight two | 44.1012 |

Moving the driver call to another thread alone is not a measured win here: both
asynchronous variants regress versus synchronous inflight one. The causal shares
of wakeup/handoff, encoding and GPU execution are not separately measured by this
unprofiled screen. Do not claim all queue_submit time was critical-path time.
Reject the worker addition; retain its frozen EXE, exact source patch and hashes
in the artifact directory for future research, without another production knob.
Final cache usage is 20,949,515,028 bytes, below 30 GiB; no 100 GiB increase.

## Final inflight-one matrix

`stage4-final-inflight1-matrix` completed eight service instances, each with
512 tokens and three rounds, profiling off, on the same B638 executable. All
12 resident batches match an ordinary batch jointly on prompt length, cached
length, output length and output hash. Two slots are configured throughout.
The table excludes round zero and averages the two warm rounds; dual-active
per-lane rates average both lanes. Aggregate rates use fully contained Decode
windows, not summed request rates. Front/middle/late use one-second progress
bins grouped by token midpoint thirds, rounded by the existing log format.

| Context / active | Ordinary per-lane tok/s | Resident per-lane tok/s | Ordinary aggregate | Resident aggregate |
| --- | ---: | ---: | ---: | ---: |
| 30K / 1 | 43.16 | 44.39 | 43.14 | 44.46 |
| 30K / 2 | 32.07 | 31.24 | 64.14 | 62.40 |
| 150K / 1 | 38.99 | 39.49 | 39.14 | 39.51 |
| 150K / 2 | 24.23 | 23.73 | 48.43 | 47.29 |

| Context / active | Ordinary front / middle / late | Resident front / middle / late |
| --- | ---: | ---: |
| 30K / 1 | 42.07 / 44.13 / 43.02 | 42.81 / 46.57 / 43.80 |
| 30K / 2 | 32.27 / 32.35 / 31.49 | 31.56 / 31.67 / 30.24 |
| 150K / 1 | 37.81 / 39.28 / 39.77 | 38.71 / 39.39 / 40.20 |
| 150K / 2 | 24.89 / 23.74 / 23.92 | 24.37 / 22.93 / 23.74 |

The full matrix does not confirm the 50 tok/s target. Single-active gains are
small (+2.8% at 30K, +1.3% at 150K), while dual-active still regresses. Keep the
master switch default off; do not promote the screened best round to a claimed
production rate. Cache usage ends at 20,916,434,288 bytes. No temporary 100 GiB
limit was needed, and production caches/executable were not changed.

## Short calibrated timeline comparison

The requested small follow-up is `stage4-new-timeline`: ordinary/resident,
30K/150K, two configured slots and one active, MTP off, 384 tokens and two rounds.
Each case captures only 12 consecutive forwards in the second round, at
positions 30103-30114 or 150107-150118. All four candidate batches match ordinary
cache state and output; all services finish normally. No query warnings or
dropped events occur; maximum clock calibration deviation is 18 ns. Test caches
end at 19,049,841,080 bytes, below the aggregate 30 GiB limit.

The standalone HTML, full traces, results, logs and source/throughput manifests
are preserved under the original checkout's
`benchmark-data/artifacts/decode-timeline-20261011-gpu-resident/`.
`scripts/run-gpu-resident-timeline.cjs` reproduces the capture;
`scripts/analyze-gpu-resident-timeline.cjs` accepts explicit input/output roots
and the independent unprofiled throughput manifest. The historical October 7
HTML and renderer are unchanged. Browser checks pass at 1440 and 390 px,
including canvas pixels, all four case selections, real wheel zoom, drag,
reset, layer hover and page-overflow checks. Five focused analyzer tests pass.

| Short-window mean per forward | 30K ordinary | 30K resident | 150K ordinary | 150K resident |
| --- | ---: | ---: | ---: | ---: |
| Step wall ms | 24.818 | 21.661 | 26.097 | 22.922 |
| Main-queue envelope union ms | 14.990 | 15.632 | 15.810 | 16.521 |
| Main submits | 111.33 | 77.17 | 111.33 | 77.17 |
| Exclusive command encoding ms | 1.612 | 2.052 | 1.646 | 2.415 |
| Exclusive main-thread submit API ms | 2.160 | 1.829 | 2.230 | 1.998 |
| Exclusive measurement ms | 2.576 | 2.663 | 2.584 | 2.688 |
| Pending interval union ms | 6.416 | 10.606 | 6.662 | 10.817 |
| Pending overlapping main work ms | 2.613 | 8.624 | 2.653 | 8.781 |

These are localized profiled windows, not throughput evidence or independent
phase medians. Main queue intervals include dependencies and barriers, not
just arithmetic. The improved submit count is real, but added speculative
encoding and a slightly larger main envelope oppose it. Longer pending mostly
overlaps queued main work; it cannot all be called wasted time. The much larger
main-thread `lowering_other` on the resident path is not established CPU busy
time: polling/control and uninstrumented delays remain mixed there. Neither
these short samples nor subtracting explicit measurement overhead proves 50.

The producer thread does not inherit the TLS trace context. Its preparation,
mapped copies and transfer-queue GPU intervals are absent, so a resident empty
DMA track does not mean zero transfer, and reduced main-thread preparation is
not eliminated work. A future precise miss critical-path audit needs producer
context instrumentation, not another throughput matrix inferred from this gap.

Ordinary CPU layer ranges remain graph-order-derived. Early submit now separates
MoE from the final HC, so the old same-submit MoE/HC GPU-boundary algorithm no
longer works. The new estimator associates each final HC's actual recorder and
host submit with its first GPU HC interval, rejects missing/ambiguous/shared
anchors, and uses isolated HC duration to split merged injection/norm intervals.
Ordinary samples have 48 inferred CPU and 48 estimated GPU layers. Resident
speculative replay emits repeated CPU layer spans and unconditional timestamp
markers even for skipped dispatches; its layer estimates are intentionally
omitted rather than pretending these are 48 executed-layer timestamps.

## Notice ACK and producer CPU phase diagnostic

`stage4-producer-phase-diagnostic` retains scheduling/arithmetic unchanged and
adds the missing producer-ACK wait before pending-segment fence recycling,
nonblocking poll classification, post-wait recycling, and explicit producer
phase envelopes. Worker events keep the submitted forward's context but do not
enter the main-thread exclusive partition. They are not transfer-queue GPU
intervals. The default-off timeline still records nothing outside selected
forwards. Core 241 / Vulkan 213 default tests pass (106 GPU tests ignored), along
with six analyzer tests, including ACK classification and worker exclusion.

Frozen `infr-notice-phase-profile.exe` SHA256 is
`8c878f2d51d3cb9e510d8d2ee6a47e1383c5b10e5485d54b0a12c9d1b90d6c16`.
The small diagnostic uses two configured slots, one active, 30K, 384 tokens,
two rounds and CPU-only tracing of 12 forwards at positions 30103-30114. Both
candidate batches match the preceding ordinary 384-token baseline jointly on
cache state and output. No events are dropped; there are no GPU events or clocks.
Cache usage ends at 19,983,136,028 bytes, below 30 GiB.

| Phase | Mean ms / captured forward |
| --- | ---: |
| Step wall | 20.406 |
| Main-thread notice ACK wait | 6.443 |
| Main-thread nonblocking notice polls | 0.217 |
| Main-thread post-wait segment recycling | 0.009 |
| Main-thread remaining lowering/other | 2.913 |
| Producer total wall envelope | 14.781 |
| Producer GPU-event wait / driver polling | 12.770 |
| Producer journal read | 0.367 |
| Producer hit LRU update | 0.332 |
| Producer miss admission / RAM copy preparation | 1.033 |
| Producer CPU DMA submission | 0.244 |

Producer total contains its subphases; neither these rows nor worker/main rows
may be summed as sequential work. ACK wait was previously unclassified wall
time, not a measured 6.4 ms CPU setup cost or a freely removable stall. Removing
it would race fence reuse with the worker's event/fence queries. Worker event
wait is mostly a polling envelope, not proof that the driver or GPU computation
individually costs 12.8 ms. The 59.67 producer notices average 34 hit updates and
13 miss admissions; the rest are skipped speculative notices. This differs
from the GPU-query-instrumented window's speculative workload, another reason
not to subtract phase values between these different profilers.

The next narrow candidate is amortizing the redundant fence-status fallback
inside event polling. Keep event queries on every iteration and the first fence
probe immediate, retain completed-without-notice/error detection, and require
unprofiled A/B before keeping a changed interval. The measured journal/LRU/prep
costs alone do not establish the 50 tok/s target. This diagnostic's profiled
request rates and best sampled step are not new unprofiled throughput claims.

## GPU-only miss critical-path capture

The user's next direction supersedes the proposed event/fence polling trial:
keep the already-good full-hit continuation and first measure miss recovery.
No polling interval or inference arithmetic/scheduling was changed here.

`c97fb0740` adds opt-in `prof.timeline_gpu_only=true` (mutually exclusive with
CPU-only tracing). Host phase clocks are disabled, not just hidden by the chart.
Forward position/origin, CPU submit instant and calibrated clock pairs remain
as the minimum metadata needed to compare the two GPU queues. A scoped trace
context propagates into the notice uploader so real dedicated-DMA queries are
no longer missing. Background host spans remain suppressed rather than being
misclassified as inference-thread work.

Per-marker layer context and FFN labels distinguish router notice, hit/shared
FFN and resumed miss FFN. GPU journal status and hit masks identify conditionally
skipped FFNs; unconditional timestamp markers do not prove that their kernels
executed. The PLE overlap runner executes L0 in its ordinary graph and L1-L47
in the resident tail. The analyzer requires all 47 resident notices/FFNs plus
ordinary L0 GPU MoE evidence per forward (or 48 notices for an unsplit graph),
checks each captured DMA signal against the resumed submission's wait value,
and refuses incomplete or ambiguous captures. Other GPU stage intervals remain
completion milestones, not isolated kernel or ALU busy time. Mapped CPU push
has no GPU copy timestamp and must not be fabricated as a DMA interval.

Validation before capture: core/Vulkan default tests passed (242/213; 106 GPU
tests ignored); the final six timeline tests include context restoration and
CPU/GPU-mode rejection, and three synthetic analyzer tests cover skipped FFNs,
DMA dependencies and mapped-push/non-GPU rejection. The bounded capture uses
30K/150K, two configured slots/one active, the selected inflight-one preset,
384 tokens twice and 12 forwards in the second round. Every candidate batch
must match the previous ordinary 384-token baseline on whole-batch KV state
and output. This is a diagnostic window, not a new sustained-throughput claim.

Capture completed with frozen `infr-gpu-only-miss-trace.exe`, SHA256
`5c71f0eb427b3927d4407282097140da7c97f42552375758190de21b04adccd3`.
The four whole-batch cache/output checks match the previous ordinary baseline.
Each depth has 12 steps, zero host phase events, zero dropped events, all 47
resident layers and ordinary L0 GPU evidence. Every captured producer DMA signal
is carried into its corresponding miss-resume wait. Max reported calibration
deviation is 19/21 ns. Services shut down normally; central cache aggregate
finished at 20,916,430,380 bytes, below the 30 GiB rolling budget. Production
service/executable/cache were not used or replaced.

| GPU diagnostic mean | 30K | 150K |
| --- | ---: | ---: |
| Step wall ms | 22.180 | 22.726 |
| Main queue envelope union ms / step | 15.582 | 16.544 |
| Miss layers / step | 13.000 | 13.083 |
| Routed experts missing / miss layer | 1.282 | 1.268 |
| Partial-hit/shared FFN interval us / miss layer | 74.97 | 74.13 |
| Miss-resume FFN interval us / miss layer | 33.42 | 32.86 |
| DMA interval us / DMA-bearing miss layer | 98.55 | 101.49 |
| Router-notice completion to DMA start us / DMA-bearing miss | 89.37 | 88.21 |
| DMA/hit overlap ms / step | 0.014 | 0.012 |
| Hit completion to resumed FFN start us / all miss layers | 278.69 | 280.21 |
| Same gap us / DMA-bearing miss layers | 246.62 | 241.37 |
| Same gap us / mapped-push-only miss layers | 344.72 | 438.06 |
| Both hit and DMA complete to resumed FFN start us / DMA-bearing miss | 132.05 | 124.83 |
| Total hit-to-resume gaps ms / step | 3.623 | 3.666 |
| Uncovered DMA tail after hit completion ms / step | 1.002 | 1.224 |
| Both-ready-to-resume gaps ms / step, DMA-bearing misses only | 1.155 | 1.311 |

The 30K window has 156 miss layers (105 DMA-bearing, 51 mapped-push-only); 150K
has 157 (126/31). No DMA duration is inferred for CPU push. A DMA-bearing miss
may additionally contain mapped copies; its timestamps cover only the DMA
portion. Table rows overlap: especially total miss gaps already contain exposed
transfer tails and post-ready gaps. They must not be added or indiscriminately
subtracted from unprofiled throughput. The query intervals also include barriers
and scheduling stalls, not just arithmetic. The 12-step workload is intentionally
small and not a long-term miss distribution.

The critical-path priority is now concrete:

1. Reduce ready-to-resume handoff. `finish_with_notice()` currently drains old
   segments/producer ACKs, verifies final control, clears/rebuilds frozen LUT
   windows and then re-encodes/submits recovery. Separate retirement from the
   recover command, keep old command/fence/event/tape resources alive, and use
   immutable IDs/masks/LUT windows rather than overwriting hit-reader buffers.
   Do not remove the existing ACK/fence ownership guarantees in place.
2. Prebuild/cache the miss-only GPU FFN path and consume the notice/prepared
   promotion directly. Avoid repeated generic control readback, whole-tape
   reconstruction and lowering on the recovery critical path where correctness
   permits. Main-thread CPU, driver and semaphore contributions cannot be
   individually assigned from this GPU-only capture.
3. Advance DMA start. At about 89 us after the notice it usually starts after
   the 75 us hit FFN, leaving almost no useful overlap. Compact producer
   admission/transfer preparation is worth testing after recovery is shortened.
   This trace does not prove that simultaneous transfer slows computation: the
   two scarcely overlap here. Faster bulk copy alone misses the larger handoff.

Standalone artifact:
`D:\AIinfr\infr\benchmark-data\artifacts\decode-timeline-20261011-gpu-miss\timeline.html`.
The directory includes raw traces/results/logs, frozen-build manifest, matching
correctness evidence and analysis JSON, not KV cache payloads. Six tracks show
main GPU envelopes, other completion milestones, actual hit FFN, miss FFN, DMA
and skipped FFN. Pointer drag/wheel zoom, desktop 1440/mobile 390 canvas pixels
and no-page-overflow checks pass. Four analyzer tests now additionally cover
the 47-layer PLE split and missing L0 evidence. This capture does not establish
the >=50 tok/s goal; resident execution remains opt-in.

### GPU forward-boundary audit (same capture, no additional GPU run)

The user additionally identified the three boundary regions: before the first
GPU work, the L0/L1 split, and after the last GPU work. Analyze all 12 forwards
per depth from the same GPU-only capture, not the selected image alone:

| Boundary, mean ms / forward | 30K | 150K |
|---|---:|---:|
| Forward start to first main queue GPU start | 0.584 | 0.597 |
| Independent L0 graph end to first L1 tail submission GPU start | 1.052 | 0.978 |
| Last main queue GPU end to forward end | 0.346 | 0.400 |
| Sum of the three disjoint boundary regions | 1.982 | 1.975 |

These sums are about 8.9% / 8.7% of the profiled forward wall time. Miss
hit-to-resume gaps total 3.623 / 3.666 ms per forward, so recovery handoff
remains first priority; the L0/L1 split is the next largest boundary. None of
these numbers establishes how much unprofiled throughput can be recovered.
The front/tail gaps include driver scheduling and profiler work as well as
ordinary host preparation/retirement. The last main GPU boundary includes the
output head and GPU sampler, not merely the last transformer layer.

The L0/L1 number uses the actual queue envelope containing L1's router/hit
marker and the last completed independent L0 submission before it. It is not
the distance to L1's router, which also includes L1 preparation/attention work.
An unsplit graph has no such split-gap measurement. Missing envelopes or GPU
intervals outside the forward clock are rejected rather than estimated.

Existing CPU+GPU trace `stage4-new-timeline` corroborates the mechanism, without
rerunning CPU tracing: across its 12 resident forwards, mean PLE ticket waits
are 0.000917 ms / 0.001250 ms (30K/150K); L1 `backend_setup` is 0.422 / 0.430
ms and `step_tail` is 0.134 / 0.150 ms. This is a different profiling capture,
so these are not subtractions from the GPU-only table. Source confirms:

- `seam/runner.rs` builds/compiles/executes the separate L0 graph, consumes the
  PLE ticket and uploads rows, then builds/compiles/binds the L1..47 graph.
- `ResidentChunk::new()` excludes graphs with only one `MoeFfn`, so ordinary
  L0 still uses CPU router/readback even when all selected experts hit.
- The tail separately recomputes scratch layout, fusion plan and paging scan,
  freezes residency, encodes its first chunk and submits it.

Next candidates are early miss submission with old readers retained; then
removing repeated tail setup or combining L0/tail when PLE readiness permits.
Moving setup to the front without overlap is not itself a throughput gain.
For the outer boundaries, separate invariant graph/planning work from genuine
position-dependent work and unnecessary final drains from required sampling,
output delivery and resource lifetime. Do not remove these guarantees solely
because a region is blank in the GPU chart.

The existing GPU-only HTML now includes a three-row per-sample/mean/P50 boundary
table with these limitations. The six measured tracks, drag and wheel behavior
are preserved; six analyzer tests and desktop/mobile browser checks pass. No
GPU service was launched for this boundary audit.

### Early miss-only submission: correct, but no throughput benefit

Default-off, uncommitted runtime experiment `decode_gpu_resident_early_recovery`
submits the recovery FFN before retiring old chunks/producer ACKs. Eligibility
is producer-enabled rolling `submit_layers=1`, `inflight=1`, no indirect
predicates. It appends recovery LUT windows and atomically preserves missing
expert slot protection; old fences/events/journals are still retired normally.
Neither inference arithmetic nor the LRU owner changes, and MTP remains off.

The scalar eligibility test, 243 infr-core and 214 infr-vulkan non-ignored tests
pass (106 Vulkan tests ignored). The explicit ignored GPU recovery fixture also
passes, including rows 1/2, full hits, first/second-layer misses and fallback
configurations, in 471.39 seconds. Release build took 7m08s. Frozen binary:
`target/gpu-resident-experiment/release/infr-early-recovery.exe`, SHA256
`4f455e1be9328a12f928294780160b506afbbf1530e689a5332b047d82fff25e`.

Unprofiled screen `stage4-early-recovery-screen`: same binary, configured two
slots with one active, ordinary technical questions, 384 tokens, two complete
regenerated rounds per depth. Both rounds have matching cache state and output
hashes against the same-binary control and the earlier ordinary goldens.

| API predicted rate, mean of two rounds tok/s | Control | Early recovery |
|---|---:|---:|
| 30K | 41.076 (42.105 / 40.046) | 40.927 (40.274 / 41.580) |
| 150K | 41.760 (41.458 / 42.063) | 38.704 (38.406 / 39.002) |

30K is effectively inconclusive, not a positive result; 150K regresses 7.3%.
No dual-active performance screen or larger matrix was run for this version.
The target >=50 tok/s is still unmet. Keep this experiment default-off and do
not commit it as a successful optimization.

GPU-only follow-up `stage4-early-recovery-gpu-trace` contains the same 12 positions
per depth as the earlier capture; all four batches again match ordinary output
and cache state. Compare diagnostic intervals, not sustained throughput:

| Mean measured GPU gaps | 30K old / early | 150K old / early |
|---|---:|---:|
| Both hit and DMA done -> recovery start, us / DMA-bearing miss | 132.05 / 68.25 | 124.83 / 73.39 |
| All hit end -> recovery start, ms / forward | 3.623 / 2.874 | 3.666 / 3.459 |
| Recovery FFN end -> subsequent main GPU work, ms / forward | 0.959 / 2.220 | 0.739 / 2.603 |
| Sum of those disjoint before/after-recovery gaps, ms / forward | 4.582 / 5.094 | 4.405 / 6.062 |

Miss counts agree exactly (156 / 157 across twelve forwards). The new post-FFN
metric is the gap to the next positive-duration main GPU completion interval;
it does not invent a precise next-layer start. No following recorded work means
unknown, not zero. The GPU-only old/new captures use different frozen binaries,
so they diagnose the path, not replace the same-binary unprofiled comparison.

Recovery starts earlier, but the additional standalone short submission leaves
more waiting before subsequent GPU work. Thus shortening the pre-miss interval
alone is insufficient. The next candidate is to record recovery plus the next
layer's unconditional frontend into one submission, then retire old metadata
while that larger useful GPU interval executes. Retain old LUT destinations,
ACK/fence ownership and avoid resetting control/journal until old readers are
finished; this combined candidate is not yet implemented or tested.

New diagnostic chart:
`D:\AIinfr\infr\benchmark-data\artifacts\decode-timeline-20261011-gpu-miss-early-recovery\timeline.html`.
The original GPU-only chart is retained and includes the additional post-miss
gap column as well as the three forward boundaries. Frozen manifests, source
patches, correctness results and raw traces are preserved outside test caches.
Aggregate test cache after both runs is 20,916,428,760 bytes, below 30 GiB.
Seven analyzer tests pass, including unknown post-miss boundaries; both charts
pass six-track/three-boundary desktop/mobile pixel, overflow, drag and zoom
checks. All benchmark services have stopped normally; the production binary
and production session cache were not touched.

### Queue the next frontend before retiring old chunks

The next default-off experiment delays `submit_early_recovery()` until the
paused FFN and the following unconditional frontend have been encoded, stopping
before the next `MoeFfn`, graph end, or explicit dispatch cap. Old LUT readers,
fences, producer ACKs and immutable notice/control storage remain alive until
retirement. Missing expert destinations remain protected under one pager lock.
This does not change LRU ownership, model arithmetic, MTP or Prefill scheduling.

Important submission detail, checked against both source and actual submit IDs:
the existing `decode_early_submit` still flushes the miss FFN itself. With this
test's automatic AMD setting, the new helper queues the following frontend
before retiring old chunks, but does **not** combine the FFN and frontend into
one command buffer. All 156 / 157 captured recoveries use a distinct subsequent
frontend submission. The gain is overlapping retirement with that useful
frontend, not an already-proven saving from eliminating a submit. The code also
handles the early-submit-disabled path, but that variant has no performance
result yet. No claim is made that turning early-submit off globally is better.

243 infr-core and 215 infr-vulkan non-ignored tests pass; 107 Vulkan tests are
ignored. A dedicated ignored GPU fixture passes for rows 1/2, first/last-layer
misses, full hits, subsequent full hits, mapped/nonmapped journals and the
dispatch-cap=1 fallback (40.99 seconds). This focused fixture is not the entire
ignored GPU matrix. Release build took 6m50s. Frozen binary:
`target/gpu-resident-experiment/release/infr-recovery-frontend.exe`, SHA256
`da96c58535169d00c2d402d8bdcab1b8e12175f3b66d54c0a63340cd07025666`.

Same-binary, unprofiled screen `stage4-recovery-frontend-screen`: two configured
slots, one active, 384 generated tokens, two regenerated rounds per depth.
All four batches match both the resident control and earlier ordinary golden
output hashes and cache state.

| API predicted rate, mean of two rounds tok/s | Resident control | Queued frontend |
|---|---:|---:|
| 30K | 45.007 (44.619 / 45.394) | 46.407 (45.792 / 47.021), +3.11% |
| 150K | 40.225 (38.407 / 42.043) | 42.400 (41.832 / 42.969), +5.41% |

These are screening results, not proof of sustained >=50 tok/s or dual-active
benefit. The cold first 150K round is slower than its second round; do not
silently present its average as a steady-state number.

GPU-only `stage4-recovery-frontend-gpu-trace` again contains twelve matching
positions per depth and matching output/cache state for all four batches:

| Mean diagnostic | 30K | 150K |
|---|---:|---:|
| Forward wall ms | 20.414 | 22.231 |
| Main queue envelope union ms | 15.496 | 16.326 |
| Actual miss layers across twelve forwards | 156 | 157 |
| DMA-bearing miss layers | 105 | 126 |
| Both hit and DMA done -> miss FFN start, us / DMA-bearing miss | 61.24 | 66.05 |
| All hit end -> miss FFN start, ms / forward | 2.562 | 2.731 |
| Miss FFN end -> next recorded main GPU interval, ms / forward | 0.191 | 0.354 |
| Sum of disjoint before/after miss gaps, ms / forward | 2.752 | 3.085 |
| Before first GPU work, ms / forward | 0.586 | 0.689 |
| L0 graph end -> L1 tail first GPU submit, ms / forward | 0.888 | 1.109 |
| After last GPU work, ms / forward | 0.298 | 0.412 |

The earlier original GPU-only gaps sum to 4.582 / 4.405 ms, and the rejected
miss-only-early experiment sums to 5.094 / 6.062 ms. This version avoids moving
the saved wait behind the miss FFN. GPU diagnostics use separate frozen builds
and small selected windows; they do not replace the unprofiled comparisons.
Completion-interval gaps can include GPU barriers and scheduling, not just host
delay. Outer boundary differences are not attributed to this patch, which does
not change the L0/tail graph split or sampling/output delivery.

By actual transfer path, hit-end to miss-start mean is 163.36 / 176.41 us for
DMA-bearing misses and 266.44 / 339.98 us for mapped-push-only misses
(30K / 150K). Those are path-group diagnostics with potentially different
expert counts and bytes, not a bandwidth-only A/B. They justify investigating
the unimported-RAM fallback as well as the notice/admission-to-DMA-start delay;
they do not establish a specific CPU copy or driver cost without further
instrumentation. The separate frontend submissions begin soon after the miss
FFN, so saving a submission alone is no longer the largest measured gap.

New chart, preserving earlier charts and all six measured tracks:
`D:\AIinfr\infr\benchmark-data\artifacts\decode-timeline-20261011-gpu-miss-recovery-frontend\timeline.html`.
Raw logs, traces, correctness and frozen source manifests are copied alongside
it; no KV payloads are copied. Seven analyzer tests pass; desktop 1440/mobile
390 checks confirm six nonblank measured tracks, three boundary rows, no page
overflow, wheel zoom and pointer dragging. Screenshots are preserved.

### Completed ordinary-control matrix for queued frontend recovery

`stage4-recovery-frontend-ordinary-matrix` compares the same frozen binary with
ordinary Decode against the resident preset plus queued frontend recovery.
Configured two slots; 30K/150K x one/two active; 512 generated tokens per
request, three regenerated rounds, no profiler. All twelve candidate batches
match a whole-batch ordinary output hash and identical prompt/generated/cache
state. Candidate cold-start rounds may match an ordinary warm round when the
cache state agrees; neither hash matching nor throughput is claimed against
an incompatible cold cache state.

The table uses only regenerated rounds 1 and 2 (zero-based), with the first
round excluded consistently for both cases. Per-lane means are API predicted
rates, averaged across those two rounds and all active lanes. Aggregate is the
mean of fully contained all-active Decode windows from existing server stats,
not merely per-lane rate multiplied by the lane count. Raw one-second bins
remain available; those window rates are rounded to 0.1 tok/s by the server.

| Context / active lanes | Ordinary per lane | Resident + queued frontend per lane | Change | Ordinary aggregate | Resident aggregate |
|---|---:|---:|---:|---:|---:|
| 30K / 1 | 43.247 | 47.766 | +10.45% | 43.199 | 47.925 |
| 30K / 2 | 32.279 | 33.459 | +3.66% | 64.646 | 66.818 |
| 150K / 1 | 40.179 | 43.171 | +7.45% | 40.235 | 43.545 |
| 150K / 2 | 24.433 | 24.979 | +2.23% | 48.855 | 50.037 |

Those deltas include the existing resident path as well as this new recovery
change. Only the earlier same-binary resident-control screen (+3.11% / +5.41%)
isolates the additional queued recovery against resident control; no isolation
claim is made for the dual-active deltas. Both warm rounds improve in each
case, but these are two warm repeats, not a confidence interval or a 24-hour
soak result. Default-off gates are retained; other GPUs/drivers are unvalidated.

| Warm request bins, per lane: front / middle / late tok/s | Ordinary | Resident + queued frontend |
|---|---|---|
| 30K / 1 | 42.37 / 44.20 / 42.91 | 46.86 / 48.53 / 47.86 |
| 30K / 2 | 32.41 / 32.72 / 31.59 | 33.54 / 34.27 / 32.39 |
| 150K / 1 | 39.39 / 40.62 / 40.28 | 42.15 / 43.94 / 43.18 |
| 150K / 2 | 25.31 / 23.80 / 24.03 | 26.28 / 24.55 / 24.17 |

Bins are grouped by token-midpoint thirds using the existing one-second
progress sampling, then averaged over the two warm rounds/all active requests.
They are not exact individual-token timestamp distributions. First-round
ordinary dual-active cold Prefill is excluded from the Decode comparison.
Neither 50.037 aggregate in the last row nor a faster short GPU-trace sample
establishes >=50 single-lane throughput: that goal remains unmet.

Final aggregate test-cache size is 20,916,434,288 bytes, below 30 GiB. Services
stopped normally. No production executable/service/cache was replaced. Next
priorities remain the independent L0/tail graph setup boundary and pre-recovery
miss transfer/admission, especially unimported-RAM mapped pushes. The front and
tail gaps remain separately measured; they are not automatically removable.

Runtime implementation is local commit `86898f496`. The frozen executable was
built at `0a43785ee` plus the source patch archived in each manifest; before
commit only the recovery comment was corrected to describe the actual separate
early submissions. The ordinary matrix and screening evidence are also copied
under the new chart directory's `full-matrix/` and `resident-screen/` folders.

### Full-graph experiment and PLE storage/cache distinction

Frozen experimental binary `infr-full-graph.exe`, SHA256
`afa999835b28d3fa75e2333d88d887b7be500377d551ee67a9ebf4c7ac776dc7`,
is built at `1b948ca71` plus the archived full-graph source patch. It keeps the
same model, fixed allocations, scheduler and existing resident preset. The
new default-off `kernels.vulkan.decode_gpu_resident_full_graph` prepares PLE
before executing one L0..L47 graph. Only ordinary text Vulkan FlashNext
Decode with one/two rows is eligible; partial Prefill, MTP, multimodal,
expert prefetch and CPU-miss variants retain the original split path.
Core/llama/Vulkan library tests: 243/210/215 passed; 107 Vulkan GPU tests
remain ignored. Real model screening and the GPU-only paired trace each
match all four whole batches against both same-binary split control and
the prior ordinary golden at identical prompt/generated/cache state.

Same-binary GPU-only capture uses the same twelve forward positions per
depth as the earlier chart. Split and full paths have respectively 47 and
48 executed resident FFN notices; the full path's L0 split gap is **not
applicable**, not an artificially measured zero.
The paired L1..L47 miss-layer/hit-count sequences are identical at every
captured position. Full-graph totals add four L0 misses at 30K and three at
150K; those additional classified layers are not a cache-hit-rate regression.

| Twelve-forward mean, ms | 30K split | 30K full | 150K split | 150K full |
|---|---:|---:|---:|---:|
| Wall | 21.349 | 20.372 | 21.656 | 21.246 |
| Main queue union | 15.457 | 15.528 | 16.324 | 16.305 |
| Before first GPU work | 0.598 | 1.226 | 0.585 | 1.191 |
| Independent L0 -> L1 graph gap | 0.912 | N/A | 0.954 | N/A |
| After last GPU work | 0.398 | 0.287 | 0.392 | 0.354 |

Removing the split does not remove all its time: some preparation moves
before the first GPU work. The short captured windows improve, but the
initial two-round unprofiled 384-token screen does not show stable sustained
gains, especially its second 150K full-graph round. Do not promote that short
trace improvement to a general throughput or restart/cache guarantee.

PLE is a GGUF mmap tier, not a fully resident RAM/VRAM table. Windows may
serve those pages from its file cache or fault them from SSD. Restarting the
service does not guarantee either state; model host-cache allocation, other
applications and RAM pressure can reclaim cached pages. No system file-cache
purge was performed. These are **first and repeated request rounds**, not
controlled cold-SSD/hot-cache benchmarks.

`stage4-full-graph-ple-profile-v3` separately enables only pager aggregate
timing, with `prof.stages=false`, `prof.ops=false`, no timeline queries.
All four candidate batches also match same-binary control and ordinary golden.
Every runner cohort contains 414 processed rows: 30 remaining teacher-forced
suffix rows plus 384 Decode rows. Times below are totals for that mixed
cohort, not pure Decode per-token or physical SSD read latencies.

| PLE total ms: gather / residual receiver wait | Split first | Split repeated | Full first | Full repeated |
|---|---|---|---|---|
| 30K | 414.0 / 45.3 | 16.1 / 0.1 | 340.5 / 317.2 | 21.1 / 29.1 |
| 150K | 398.3 / 22.2 | 18.1 / 0.1 | 409.3 / 382.0 | 16.7 / 23.4 |

The gather timer includes mmap access/page faults, copying and dequantization,
but excludes job-channel scheduling and some output-pool management. The
receiver wait includes their exposed time and can exceed gather time; it is
not simply SSD read time. Logical 4KiB pages are not physical SSD I/O bytes.
The upload timer is separate. Small split-path wait is consistent with L0
overlap, not proof that the underlying PLE gather was cheap. Full-graph warm
cache measurements alone cannot justify eliminating this overlap for fresh
requests or after restarts.

The earlier `ple-profile` launch failed before GPU work because profiling
arguments were duplicated. `ple-profile-v2` completed with verbose stage
logging, which produced many per-submit stderr lines and perturbed throughput;
it is retained as diagnostic evidence, not a release-speed baseline. V3
removes that logging. Its aggregate profiler throughput is also kept separate
from the unprofiled matrix. The parser tests distinguish work from wait and
reject malformed/unpaired cohorts.

Paired charts, preserving all prior charts:
`D:\AIinfr\infr\benchmark-data\artifacts\decode-timeline-20261011-gpu-full-graph-control\timeline.html`
and
`D:\AIinfr\infr\benchmark-data\artifacts\decode-timeline-20261011-gpu-full-graph\timeline.html`.
The new chart passes desktop/mobile nonblank-pixel, exact layer-set, overflow,
drag and wheel-zoom checks. A robust follow-up should preserve async PLE/L0
overlap while reducing duplicate graph/setup work; it must not assume Windows
file-cache residency. Do not yet enable unconditional full-graph Decode.

### Completed full-graph matrix: no-regression criterion not met

`stage4-full-graph-resident-matrix` uses the same frozen full-graph binary
for split and full controls, configured two slots, 30K/150K x one/two active,
512 generated tokens and three regenerated rounds. Profiler and timeline
queries are disabled. All twelve candidate batches match both the same-binary
split control and the prior ordinary golden as complete batches, including
prompt/generated/cache state (24 comparisons). This establishes correctness
for these cases, not a performance guarantee.

Per-lane rates below average all active lanes and regenerated rounds 1 and 2
(zero-based), consistently excluding the first round in both cases.

| Context / active lanes | Split per lane tok/s | Full per lane tok/s | Change | Individual warm-round changes |
|---|---:|---:|---:|---|
| 30K / 1 | 47.517 | 48.748 | +2.59% | +2.04%, +3.14% |
| 30K / 2 | 33.220 | 33.081 | -0.42% | +1.02%, -1.84% |
| 150K / 1 | 43.493 | 41.927 | -3.60% | -4.23%, -2.97% |
| 150K / 2 | 24.797 | 25.123 | +1.32% | +5.01%, -2.32% |

All four first-round mean per-lane changes are negative (-1.52%, -2.74%,
-2.58%, -1.69% in table order). Neither the positive 30K single-lane mean nor
the positive 150K dual-lane mean establishes absence of negative returns.
The 150K single-lane slowdown appears in both warm repeats, whereas small
dual-lane differences also contain substantial run variation. The cause of
the warm 150K slowdown is not isolated: do not attribute all of it to SSD,
nor conclude that storage hides every scheduling difference. Single-lane
throughput still falls short of the >=50 tok/s goal.

The full-graph flag remains default-off and is not recommended as a universal
optimization. Runtime source remains an uncommitted experiment; source patches
and hashes in the manifests preserve exactly what was measured. Test-cache
size after the matrix is 20,916,434,288 bytes, below the aggregate 30 GiB limit;
services stopped normally and production executables/caches were not changed.

### Storage-aware acceptance and next optimization boundary

The acceptance priority is **no reproducible negative return across storage
conditions**, before claiming a universal speedup. `device.ram_budget`
controls routed-expert host caching; the current PLE implementation still
loads its table through `tensor_bytes_arc` (GGUF mmap), not an explicit
RAM-budget-controlled resident ngram tier. Large free RAM may keep those
mapped pages resident through the OS cache. RAM capacity alone is therefore
not a sufficient label for a benchmark cohort.

Future comparisons must distinguish SSD/page-fault-dominated PLE, mixed
residency under memory pressure, and effectively RAM-resident PLE pages;
each must use matched one/two-active, 30K/150K prompts and generated outputs.
This session measured first/repeated rounds but did not control OS-cache
residency, so those storage cohorts remain unvalidated. Do not purge the
system cache or assume cache retention over restart to manufacture a result.
Small differences need matched repeats before being called gains/regressions.

The universal optimization boundary is to reduce duplicate graph preparation,
bindings and host allocation **while retaining async PLE/L0 overlap**. A single
prepared plan with separately schedulable L0 and tail segments is a candidate,
not an implemented or measured gain. It must preserve buffer lifetimes and
PLE readiness ordering, retain the existing split fallback, avoid added
per-token checks on the critical path, and pass both correctness and the
storage-aware no-regression gate before promotion. Storage-specific paths can
remain opt-in until equivalent guarantees or a measured cheap selector exist.

### Storage-independent shared-expert setup scan

The retained split path's prior pager-only profile reports 102.2 ms of
`setup_moe_scan` across 414 mixed suffix/Decode steps (about 247 us/step).
The old recognizer queried `INFR_NO_MOE_SHARED_SLOT` before checking the op
kind, including every non-MoE op. This is CPU/environment overhead, not SSD
read or GPU computation. The change samples the escape hatch once per
execute and runs the unchanged shape recognizer only for `MoeFfn` candidates.
Paged-buffer eligibility checks, skip indices, GPU math/submissions, PLE/L0
overlap, scratch allocation and fixed/dynamic memory residency are unchanged.
The escape hatch is still honored between executes, not cached permanently.

CPU-only release benchmark on Windows / Ryzen 5 5600X / Rust 1.97.1,
`target-cpu=native`: a synthetic 47-layer, 4089-op graph, 1024 scans per
measurement, six alternating-order legacy/filtered pairs. The old path uses
the original per-op environment-check ordering and the same pure recognizer;
both must find all 47 identical candidates. This is not a true-model graph
or a throughput extrapolation.

| Pair | Legacy us/scan | Filtered us/scan |
|---|---:|---:|
| 0 | 654.741 | 3.912 |
| 1 | 642.544 | 4.074 |
| 2 | 637.773 | 3.813 |
| 3 | 641.502 | 3.884 |
| 4 | 635.104 | 3.702 |
| 5 | 637.717 | 4.415 |

The real graph's old scan time above, rather than the larger synthetic delta,
sets the approximate maximum removable setup cost for that captured cohort.
Other CPUs/OSes may have different environment-query costs; no Linux speed
claim is made. Unit tests compare exact candidate structs/indices across
1/2/4/8/9/32 rows and eight routed dtypes, multiple tails, missing final tail
ops, an empty graph and an enabled/disabled escape hatch. Vulkan library:
216 passed, 107 GPU tests ignored plus one opt-in CPU timing test; the latter
also passes when run explicitly. Raw logs are `stage4-shared-scan-unit.log`
and `stage4-shared-scan-cpu.log`.

Frozen candidate `infr-shared-scan.exe`, SHA256
`dc3a12c0cbc4bb32ddfd1bfea87107605528cdcc5b6adb070df31c8893096577`,
uses the same retained resident preset, with full-graph explicitly disabled.
The control is the earlier frozen `infr-full-graph.exe` with full-graph also
disabled, hash `afa999835b28d3fa75e2333d88d887b7be500377d551ee67a9ebf4c7ac776dc7`.
The benchmark records each case's actual exe/hash and copies the control's
original build/source manifest, rather than attributing today's source snapshot
to that older binary. The runtime difference is confined to this CPU scan.
The first build command used a nonexistent `infr` package and failed before
compilation; `stage4-shared-scan-build-v2.log` records the successful
`cargo +1.97.1 build --release -p infr-cli --bin infr` (4m15s).

### Shared-scan end-to-end gate remains pending

`stage4-shared-scan-matrix` runs legacy first, then filtered, with two slots,
one/two active, 30K/150K, 512 tokens x three regenerated rounds, no profiler.
The warm means exclude round 0 consistently. Complete-batch output, prompt,
cache and generated-token checks match the legacy and ordinary golden in all
twelve candidate batches (24 comparisons).

| Context / active | Legacy tok/s per lane | Filtered tok/s per lane | Change |
|---|---:|---:|---:|
| 30K / 1 | 44.322 | 47.665 | +7.54% |
| 30K / 2 | 33.270 | 33.719 | +1.35% |
| 150K / 1 | 43.266 | 42.908 | -0.83% |
| 150K / 2 | 24.983 | 24.665 | -1.27% |

`stage4-shared-scan-reverse-150k` then reverses the case order at 150K,
retaining all other settings and verifying all six candidate batches against
both controls (12 comparisons). Warm single-lane legacy/filtered means are
42.348/43.465 tok/s (+2.64%); dual-lane means are 24.925/24.817 tok/s per
lane (-0.43%). The changing single-lane sign and stalled segments in some
rounds prevent assigning every rate difference to the scan. The slight
negative dual-lane means in both orders also prevent claiming a universal
no-regression result. No controlled SSD/page-cache residency claim is made.
The CPU scan improvement is established only at module level; real-model
phase timing and wider no-regression evidence are still pending. Do not
promote this experiment or label it stable on these measurements alone.

Aggregate test-cache size fell from 20,916,434,288 to 19,499,948,148 bytes
after normal service cache management, below the 30 GiB aggregate limit.
No production cache or executable was touched. Both matrices completed and
their test services stopped normally.

### Fresh stable GPU-only timeline, 2026-10-11 17:20-17:23 +08:00

The user's fresh-chart request uses the validated queued-frontend recovery
preset from `86898f496`, not the pending full-graph or shared-scan experiments.
Frozen `infr-recovery-frontend.exe` SHA256 is
`da96c58535169d00c2d402d8bdcab1b8e12175f3b66d54c0a63340cd07025666`.
The new manifest copies the original frozen build manifest; today's source
snapshot is not falsely attributed to that older binary. Capture tag:
`stage4-stable-current-gpu-trace`, two configured slots / one active,
30K/150K, no MTP, 384 tokens x two regenerated rounds. GPU-only queries
start at step 512 for twelve consecutive forwards, positions 30103-30114
and 150107-150118. CPU span/stage/pager profiling is disabled.

| Context | Forward mean ms | Forward P50 ms | Main envelope mean ms | Front mean ms | L0/L1 gap mean ms | Tail mean ms | Sum of hit-to-miss gaps mean ms/forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30K | 20.580 | 19.736 | 15.478 | 0.527 | 0.880 | 0.372 | 2.666 |
| 150K | 21.767 | 21.458 | 16.314 | 0.617 | 0.977 | 0.389 | 2.673 |

Main envelope includes GPU waits/barriers, not isolated compute time. DMA
overlaps other tracks, and the columns must not be summed as disjoint work.
These short GPU-query windows are not sustained Decode throughput claims.
The analyzer validates all 47 resident-layer journals and actual ordinary-L0
GPU evidence, DMA resume dependencies, calibration and zero dropped events.
There are 156/157 resident miss layers over twelve forwards; 105/126 have
DMA evidence. No DMA bar does not provide CPU-push transfer duration.

30K's first request had no cache entry after normal rolling eviction and
performed fresh Prefill at 1331.49 tok/s; its output hash has no same-cache
golden and is not claimed output-validated. The initial post-capture checker
correctly rejected this unmatched state after both services had completed.
Verification-only processing explicitly records this excluded first round;
30K round 1 and both 150K rounds match the complete ordinary golden at the
same prompt/generated/cache state. Both plotted windows are in the validated
second round; no additional GPU run or weakened hash comparison was used.
The failure and verification records remain archived. Cache before/after:
19,499,948,148 / 20,433,242,500 bytes, below 30 GiB. Services stopped normally.

New independent chart, preserving all earlier HTMLs:
`D:\AIinfr\infr\benchmark-data\artifacts\decode-timeline-20261011-gpu-stable-current\timeline.html`.
Raw traces, logs, launch arguments, correctness record, build provenance,
analysis and browser screenshots accompany it. Desktop 1440px and mobile
390px checks confirm two contexts, twelve samples each, six nonblank GPU
tracks, three real forward boundaries, all resident layers, no page overflow,
wheel zoom and pointer dragging. The default sample is nearest the P50;
all twelve samples remain selectable. GPU submissions keep their true
cross-layer scope; no estimated GPU layer boundaries replace measurements.

### Actual-model shared-scan phase timing

`stage4-shared-scan-phase-profile` completed the same frozen legacy/filtered
binary comparison at 30K/150K, two slots / one active, 512 tokens x two rounds.
Only aggregate pager profiling is enabled; verbose stages, op timestamps and
timeline queries are off. All four candidate batches match the same-cache
legacy and ordinary golden as complete batches (eight comparisons). Each
cohort includes 542 forwards: 512 Decode rows and 30 teacher-forced suffix
rows. The per-forward normalization below is not a pure Decode latency.

| Context / round | Legacy MoE scan us/forward | Filtered MoE scan us/forward | Removed us/forward |
|---|---:|---:|---:|
| 30K / first | 253.506 | 40.590 | 212.915 |
| 30K / repeat | 245.572 | 33.948 | 211.624 |
| 150K / first | 292.435 | 37.454 | 254.982 |
| 150K / repeat | 264.945 | 33.395 | 231.550 |

This confirms the scan's real-model module benefit (84-87% less measured
scan time), not an equal end-to-end improvement. Other host phases vary
between runs: repeat main graph construction/compile/bind is 167.5/196.5
us at 30K and 179.2/194.1 us at 150K for legacy/filtered. Layout/reset scan
is 122.7/145.4 and 128.6/136.7 us respectively. Those phases' code is
unchanged in these binaries, so their differences cannot all be attributed
to the shared-scan patch. The profiler also perturbs throughput; its rates
are not substituted for the earlier unprofiled gate.

PLE first-round gather totals are 464.9/621.4 ms at 30K and 595.6/536.7 ms
at 150K, but exposed pager waits are only 7.4/63.6 and 15.8/44.1 ms. Repeat
gathers are 21.7/24.6 and 24.5/23.0 ms, with exposed wait 0.1-0.2 ms total.
The unchanged split scheduling still overlaps PLE effectively, including
first rounds. No physical SSD timing or controlled page-cache residency is
claimed. Cache after this run is 19,049,842,020 bytes, under 30 GiB; all
services stopped. The updated parser validates paired host cohorts and uses
forward count rather than independent row count for CPU setup normalization.

### Owned-graph CPU experiment, awaiting real-model gate

Default-off `kernels.vulkan.decode_owned_graph` selects ownership transfer
for temporary graphs in the ordinary FlashNext token-row runner. It excludes
MTP, hidden capture, multimodal lanes and non-Vulkan backends. `Backend`
adds a default `compile_owned(Graph)` implementation that delegates to the
existing borrowed compiler; Vulkan overrides it by moving the unchanged
graph into `VkDecodePlan`. Borrowed compilation remains available and replay
eligibility uses the same function before ownership transfer. No op, tensor
ID, allocation, KV shape, PLE order, submission, sampling or scratch policy
is intentionally changed. The borrowed control retains its original L0
source graph lifetime; the owned path keeps that allocation in the plan.

Unit coverage verifies object-safe default-backend fallback, tensor/op/span
allocation identity after moving, graph metadata, memoized in-place input
contents, empty replay state and both eligibility values across 1/2/4/8/32
rows. Core focused test passes; Vulkan library has 217 passed and 108
ignored (107 opt-in GPU tests and the opt-in CPU timing test). The first
Vulkan test build failed due to the test's misspelled dtype `Q4_K`; it was
corrected to the existing `Q4K` before tests passed. Both logs are preserved.
The frozen CLI build and same-binary ordinary 30K/150K x one/two-active
matrix completed. Frozen `infr-owned-graph.exe` SHA256 is
`e2c36b99d82a5b8f9f3d6839b4fc4d522e3b457777e58b9e94c01a1e1cefa35a`.
The full-graph flag is false on both sides; only the owned-graph flag differs.
The original checker exited with an output mismatch at 30K/two-active,
round 0. All test services have stopped. No runtime promotion is justified.

### Owned-graph matrix: provisional performance exclusions

After the run the user reported an unstable computer while running compilation,
and authorized discarding clearly unreasonable rounds from performance analysis.
No extra GPU measurement is started. Raw manifests, outputs and logs are kept
unchanged. This is a provisional exclusion based on reported interference and
large latency excursions, not proof that compilation caused every difference.

Round numbers below are zero-based. All first rounds remain outside the existing
warmed summary on both sides, regardless of their rates. Exclude the entire
two-lane batch, not individual tokens or the slower lane. For an excluded warmed
round, exclude the matching control round too, rather than averaging an unequal
number of rounds or retaining only the fastest candidate samples.

| Context / active | Owned round | Rates tok/s per lane | Treatment |
|---|---:|---|---|
| 30K / two | 0 | 20.327 / 20.457 | Quarantine from performance; already outside warmed summary; preserve output mismatch separately |
| 150K / two | 1 | 17.431 / 17.430 | Quarantine entire batch and paired control round from provisional performance comparison |

The second exclusion has 29.374 s Decode time per lane, compared with 20.321 s
in round 2; round 0 is also about 20.5 s. Do not remove the less dramatic
30K/one-active or 30K/two-active warmed slowdowns solely because they are negative.

| Context / active | Paired retained rounds | Control tok/s per lane | Owned tok/s per lane | Difference |
|---|---|---:|---:|---:|
| 30K / one | 1, 2 | 48.446 | 47.546 | -1.859% |
| 30K / two | 1, 2 | 33.750 | 32.409 | -3.975% |
| 150K / one | 1, 2 | 43.505 | 43.370 | -0.310% |
| 150K / two | 2 only | 25.130 | 25.196 | +0.264% |

Rates are arithmetic means of retained request `predicted_per_second` values,
not aggregate throughput or a pooled tokens/time rate. The 150K/two-active row
has only one paired round left and is especially weak evidence. These values
are descriptive only: the machine interference prevents a reliable benefit or
no-regression decision. Keep the feature default-off, pending controlled tests.

Performance filtering does not change the correctness gate. Owned 30K/two-active
round 0 has cached counts 29945/29955, matching existing warmed references, but
lane 0 hash is `e7b00d02e2acce4685a8a764c73ca398be07c38035d08df827eb713a5db7b060`
instead of `60336787d9b1776d7a7f395049cd66e1db06121e5cd6ae7c7b76720784be7205`.
Lane 1 and both subsequent rounds match. Matching cache counts alone do not
prove identical restored checkpoint contents; causality remains unconfirmed.
The mismatch is preserved as an unresolved correctness finding, not converted
to a pass by removing the round from a performance average. All other owned
batches match the existing same-cache reference outputs. The two control first
rounds with a fresh second lane are not compared against warmed hashes.

Filtering record: `stage4-owned-graph-matrix-performance-filter.json` alongside
the original matrix in `benchmark-data/artifacts/gpu-resident-decode-20261011/`.
The fresh stable HTML uses a different, previously validated frozen executable
and is not replaced by these provisional experimental results.

### Owned-graph cold restore and request-order cross-check

After the user reported the computer idle, two targeted matrices completed:
`stage4-owned-graph-cache-order-forward` and
`stage4-owned-graph-cache-order-reverse`. Both use the same frozen owned-graph
binary SHA256 above, two configured slots / two active, 30K, 512 tokens x two
regenerated rounds. The first pass starts lanes 0/1 and runs control then owned;
the second starts lanes 1/0 and runs owned then control. These are launch orders,
not a guarantee of scheduler/slot order. Returned results remain sorted by lane
before comparison. Prompts, temperatures, seeds and native execution flags are
unchanged; no source edit or compilation changes the running frozen binary.

| Pass | Control first / repeat tok/s per lane | Owned first / repeat tok/s per lane | Repeat change |
|---|---|---|---:|
| Forward | 32.356 / 32.323 | 32.608 / 33.650 | +4.104% |
| Reverse | 32.645 / 33.638 | 32.803 / 33.739 | +0.302% |

All eight whole batches (sixteen requests) match the original same-cache ordinary
golden, including cold-restored first rounds. No batch is excluded. Both prompts'
JSON hashes match between all four services. The old mismatch is not reproduced;
it remains an unresolved finding rather than being erased or declared fixed.
One warmed round per order and the order-sensitive rate difference do not prove
a stable performance benefit or complete the wider no-regression gate.

The new opt-in read-only cache snapshot records v2 header geometry, live token
digest, checkpoint token digests, stored footer checksum and separate hashes of
the actual keyed checkpoint records. It rejects production paths, reparse paths,
truncated/invalid geometry and files changing during the snapshot. It neither
modifies entries nor performs cleanup. Reading the stored footer is explicitly
not full payload checksum validation. Optional checkpoint hashing occurs before
service startup and after shutdown, not inside measured Decode; it reads sizable
state payloads (the first snapshot hashed 2,360,770,560 checkpoint bytes across
ten entries), so no controlled SSD/page-cache residency claim is made.

Across every before/after catalog, candidate files at checkpoint lengths
29945 and 29955 have identical checkpoint record sets. Canonical keyed-record
set digests are respectively
`bac90905b7531d077a2d88a2b9ee7226d63b4417e9c41b7204f1c98b5d1c4077` and
`dce71f4021efb2a85c0a7648842af8052ad796361bddd773a4c6a2cd521460b5`.
This establishes the compared recurrent/PLE checkpoint contents, not identical
full main KV prefix bytes or a dump of the actual GPU restore. It cannot by
itself locate the original failure. Do not promote owned-graph as stable yet.

Aggregate cache remained under 30 GiB at every before/after check; final size
was 19,082,929,316 bytes after normal service GC. No production service, cache
or executable was touched, and all four test services exited. The snapshot's
four CPU-only tests and strict matched-round tests pass (eight tests total).
`analyze-gpu-resident-cache-order.cjs` reproduces provenance, hash, checkpoint
and metric checks from the preserved artifacts without another GPU run.

### Inline op-I/O scratch analysis: complete first model matrix, not promoted

Default-off `kernels.vulkan.scratch_io_inline` replaces per-op heap read/write
lists in retained-scratch analysis with `SmallVec` lists (eight reads / three
writes inline). Both `Op::io_inline()` and the original `Op::io()` signature
instantiate one generic exhaustive variant match. The Vec path directly uses
boxed-array construction, as the original `vec!` macro did; it does not build
an inline list first or convert one back into a Vec.
There is one semantic I/O definition, no duplicate table, no fixed arity limit
and no change to index order, duplicate handles, optional operands or zero-reset
policy. Future larger arities spill safely. The existing locked smallvec 1.15.2
is made a direct core dependency; no dependency version changes are introduced.

Coverage includes one/two/four/eight/thirty-two rows and 2048/4096-row metadata,
maximal DeltaNet inputs and trace outputs, QSA optional multimodal positions,
HC optional gates, duplicate handles, routed optional operands, packed/partial
writes and empty graphs. Core library: 245 passed. Vulkan library: 218 passed,
109 ignored (107 opt-in GPU tests and two opt-in CPU timing tests). V3 release
CPU scan timings for 4,089 synthetic ops, six alternating passes of 1,024 scans:

| Metadata rows | Heap scan us | Inline scan us |
| --- | ---: | ---: |
| 1 | 281.766-299.641 | 54.642-59.579 |
| 2 | 277.480-290.420 | 55.288-59.005 |

Reset indices match exactly. This is CPU module timing, not real model time
or proof of an end-to-end speedup. The fixture has many small ops and is not
an exact sampled runtime graph. Preliminary v1 timings are rejected as a
baseline because their compatibility path first constructed an inline list;
v2 removed that detour but changed Vec construction to iterator collection.
V3 restores the direct boxed-array construction. All three raw logs are kept.

The separate exact-source probe extracts the original I/O match from HEAD
`1b948ca714314f8faf2f5ab50d742d09b14cff0b`. V3 checks 3,854 ops per row shape
against both new APIs, including list and checksum equality. Its standalone
original function and linked core API have different call/inlining boundaries,
so their speed ratio is not a production legacy-path regression measurement.
The completed `stage4-inline-io-matrix` uses frozen `infr-inline-io.exe` SHA256
`c15149e9601ca9a174752718ba94476a5b6ff6615602d0668ff72b6b1a3989ad` on both
sides. Full-graph and owned-graph are false; only the scratch-I/O flag differs.
Configured two slots, one/two active, 30K/150K, 512 generated tokens and three
regenerated rounds. Both cases retain async PLE/L0 overlap. Profilers and GPU
queries are disabled, and no compilation runs during throughput measurement.

| Context / active lanes | Heap per lane tok/s | Inline per lane tok/s | Change |
| --- | ---: | ---: | ---: |
| 30K / 1 | 47.422 | 46.615 | -1.70% |
| 30K / 2 | 33.397 | 33.482 | +0.25% |
| 150K / 1 | 42.867 | 43.126 | +0.61% |
| 150K / 2 | 24.947 | 25.719 | +3.10% |

Definition: arithmetic mean of per-lane request rates in zero-based rounds
1 and 2, excluding round 0 identically on both sides. No other rounds are
discarded. All raw first rounds are preserved, including one 150K control lane
that had to prefill from an absent test cache. That complete cold batch matches
the same-state cold ordinary reference; subsequent regenerated rounds match
the warmed reference. Candidate/control and both ordinary-reference gates all
pass (12 whole batches per gate, 36 comparisons); no correctness exclusions.
Cache size after measurement: 20,916,434,288 bytes, below the aggregate 30 GiB
limit. All eight services stopped normally, without modifying production.

CPU module gain is established by the synthetic scan, but a universal
end-to-end gain is not: 30K single-lane is slower in both warmed repeats. The
small/mixed changes and one fixed case order need actual layout-phase profiling
and reversed matched repeats before attribution or promotion. The 150K dual
gain alone is not grounds to default-enable the flag. Request-progress thirds
and steady aggregate rates are retained in raw manifests; merged SSE frames
cannot be silently treated as one token per frame. Single-lane >=50 tok/s
remains unmet. This experiment is not enabled by default.

The core API and four dependency/source paths alone are locally committed as
`b21907a20` (`perf(core): add inline tensor I/O lists with one operand definition`).
That commit records the host/device/ubatch scope and synthetic measurements;
it does not include the Vulkan integration/config switch or claim a universal
Decode gain. Those runtime experiments remain separately uncommitted.

### Notice completion-fence poll experiment: no confirmed gain, removed

`kernels.vulkan.decode_gpu_resident_notice_fence_interval` defaults to 1,
preserving an event query and completion-fence query on each unsuccessful
iteration. Opt-in intervals are clamped to 1..64. Event status is still queried
on every iteration; only the redundant completion-fence safety check is spaced
out. The first unsuccessful iteration always checks the fence. A completed
fence still requires an immediate second event query before accepting the
notice or reporting a missing-notice error. Event-query errors propagate
immediately; a fence-only error is delayed by at most the remaining bounded
poll iterations, not masked indefinitely.

Both the producer and controller fallback share a monomorphized wait helper.
Their spin/yield policy, journal reads, hit/miss handling, retirement fences,
admission guards, transfer dependencies and producer acknowledgements are
unchanged. This does not skip the real recovery dependency or allow a slot to
be overwritten while its GPU work is live. Controller polling without a
pending segment retains the nonblocking return.

New CPU tests cover ready/delayed events, clamped intervals, exact query counts,
completion visibility races, completion without a notice, event device errors
and bounded delayed fence errors. Core tests pass (245); Vulkan tests pass
(220, with 109 opt-in tests ignored). Compilation began only after the complete
inline-I/O GPU matrix exited. Its frozen binary is separate from this new
experiment. Real-model screening holds inline-I/O, full-graph and owned-graph
off, and changes only fence interval from control 1 to 8/16.

The two completed 30K one-active screening orders use the same frozen
`infr-notice-poll.exe` SHA256
`4063ad5cf3c3f91e1921432493d6cc41f2f650625acb934eeaef05bda651f8ce`.
Forward order has three regenerated rounds per case; reverse has four.

| Order | Control 1 tok/s | Candidate 8 tok/s | Candidate 16 tok/s |
| --- | ---: | ---: | ---: |
| 1, 8, 16 | 46.296 | 46.594 (+0.64%) | 46.821 (+1.14%) |
| 16, 1 | 46.890 | not run | 46.771 (-0.25%) |

Rates average only zero-based rounds after the first, consistently in each
case/order. No low control round is discarded: the forward control's warmed
rates are 47.590 and 45.002, larger variation than the apparent gain. All 17
requests match the ordinary golden and whole-batch controls; no correctness
exclusions or failed requests. Aggregate test cache after both orders is
18,599,738,192 bytes. All five services stopped normally.

This does not confirm a throughput benefit, nor isolate the remaining miss
gap as fence-query contention. The implementation, config field and its two
tests are removed from active source, preserving the source patches, safety
logs and frozen binary for reproducibility. The historical runner requires
that preserved build record; it must not attribute this binary to current
source. The 20 recorded Rust/Cargo/shader files match the pre-poll, fully
tested inline-I/O matrix source hashes exactly after restoration (see
`stage4-notice-poll-restored-source.json`). Its 245/218 unit results and full
model matrix remain applicable; the temporary poll tests' 245/220 results
describe the archived experiment, not the restored source. No 150K or dual
poll throughput claim is made, and the >=50 single-lane target remains unmet.

### Small mapped miss-copy threshold experiment

The next candidate targets CPU-push-only miss preparation, not imported Host
DMA bandwidth. `prepare_upload_batch()` previously sent every multi-range
mapped batch through Rayon, including three sub-MiB role banks for one expert.
An experimental notice-producer executor now supports a total mapped-byte
threshold: at or below the threshold copy synchronously on that producer;
larger batches preserve Rayon. The control value is zero; the configuration
is in KiB and capped at 8192. Only the resident notice producer installs it;
ordinary/Prefill uploads and other background executors keep zero. Imported
DMA selection, slot admission, LRU, ownership, ACK and resume waits are intact.

Pure tests cover empty/single/multiple batches, threshold equality, integer
overflow, exact payloads and untouched guards. An opt-in Vulkan module benchmark
uses real DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT destinations, three different
role sizes, 1/2/3/4/8 experts, alternating order over six passes, and 96 copies
per timing. Each measurement is followed by HOST->TRANSFER synchronization and
actual GPU readback to cached staging to verify the payload. Timing covers host
copy preparation only, not that verification, inference or concurrent DMA.
The module uses an explicit 10-thread Rayon pool; it is not a full-model notice
thread scheduling benchmark. No throughput benefit is accepted from this test
alone. The same-executable model runner preserves strict same-cache whole-batch
output checks and separately records cold unmatched first rounds.

220 Vulkan non-ignored tests pass (110 ignored), including the new pure cases;
the explicit GPU module fixture then passes all 60 payload checks in 3.82 s.
Native Rust 1.97.1 build of the CLI completes in 7m32s. Module executable SHA256:
`ac068daa2d92b1b6a551320f6624d788320844f8c7c2272f1799b653062e6f1c`.
Frozen experimental CLI `infr-mapped-copy.exe` SHA256:
`9fdb19eea4c6589dde98f5b5648150c795764aff0e856a0932417b70f2391fc7`.

| Expert triples | Batch MiB | Rayon mean us | Serial mean us | Preparation reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.094 | 155.726 | 139.632 | 10.33% |
| 2 | 4.188 | 301.165 | 278.366 | 7.57% |
| 3 | 6.281 | 443.618 | 424.663 | 4.27% |
| 4 | 8.375 | 583.905 | 566.618 | 2.96% |
| 8 | 16.750 | 1153.840 | 1141.088 | 1.11% |

Every pass is retained, including the higher three-expert serial measurement;
no outlier is removed. These are module results on RX 7900 XTX/5600X, not
confirmed inference improvements. A 6144 KiB model screen would target the
one/two-expert cohorts and preserve parallelism above that limit; it has not
been run and is not a selected production parameter. The user requested a
new stable timeline at this checkpoint, so full-model A/B remains pending and
the candidate stays default-off/uncommitted. The new chart uses the already
validated recovery-frontend binary, not this candidate.

Test caches remain under the shared repository `test-kv-cache/`; the pre-test
aggregate is 18,599,738,192 bytes. No production service, executable or cache
has been touched. Raw module data: `stage4-mapped-copy-module.log` and generated
`stage4-mapped-copy-summary.json` in the campaign artifact directory.

### Fresh stable evening GPU timeline, 2026-10-11 21:20-21:23 +08:00

Following the user's checkpoint request, capture a new ordinary two-slot /
one-active 30K/150K timeline using the validated frozen recovery-frontend
binary from `86898f496` (SHA256 `da96c58535169d00c2d402d8bdcab1b8e12175f3b66d54c0a63340cd07025666`).
The current mapped-copy/inline-I/O/owned/full-graph trials are not included.
An unrelated three-slot VRAM pressure service was detected; the user explicitly
authorized stopping it, its executable/port identity was checked, and process
exit was confirmed before the capture. This was not the production service.

Tag: `stage4-stable-current-gpu-trace-night`. Four 384-token requests (two
regenerated rounds per depth) match the same-cache ordinary whole-output golden;
no cold exclusions or failures. Twelve consecutive forwards from each second
round are plotted, with host stage/pager instrumentation off. Both cases have
all 47 resident-layer journals plus actual ordinary L0 GPU evidence; every
recorded DMA signal is carried into its corresponding resume dependency.

| Context | Mean forward ms | P50 ms | Main envelope mean ms | Front mean ms | L0/L1 mean ms | Tail mean ms | Hit-to-miss gaps sum mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30K | 21.313 | 20.660 | 15.461 | 0.611 | 1.081 | 0.364 | 2.721 |
| 150K | 21.892 | 21.223 | 16.304 | 0.654 | 1.054 | 0.358 | 2.702 |

Miss counts are 156/157 across twelve forwards; 105/126 have DMA evidence.
Max reported clock-calibration deviation is 19 ns in both cases. Means retain
all samples, including slower outer-boundary observations; no performance
outlier is removed. Main envelopes include barriers/waits; DMA overlaps other
tracks, so these columns are not disjoint components to add. This short
profiled capture is not a new sustained-throughput gain or >=50 tok/s claim.

Standalone artifact:
`D:\AIinfr\infr\benchmark-data\artifacts\decode-timeline-20261011-gpu-stable-night\timeline.html`.
Raw logs/traces, analysis, frozen build record, correctness results and source
patches accompany it; no KV payloads are copied. Seven analyzer tests pass;
desktop 1440/mobile 390 browser checks verify two contexts, twelve samples,
six nonblank measured tracks, boundary rows, drag/wheel zoom and no page
overflow. Both test services have stopped; production executable/cache remain
untouched. Shared test-cache aggregate ends at 19,533,032,628 bytes, below 30 GiB.

### User-directed focus after the evening capture

The user narrows subsequent work to miss notification/upload/resume and the
outer forward boundaries. The reported DMA duration is 70 **us**, not ms.
The user explicitly includes mapped CPU-push misses as an equal-priority
branch: split admission, source preparation, mapped writes and recovery
handoff; absence of a DMA bar does not make that interval pure control cost.
Do not broaden this round into L0/L1, full-hit inter-layer gaps, kernel tuning,
LRU migration or MTP. Finish the already-running mapped-copy A/B without
launching another unrelated performance sweep.

Across the twelve evening forwards, the per-DMA-miss median notice-to-DMA
start is 86.68 us at 30K and 83.72 us at 150K. Median readiness-to-resumed-FFN
is 61.51/58.485 us, where readiness is the later of hit completion and DMA
completion. DMA/hit overlap is almost absent: median zero, mean 1.515/1.762 us
over all misses (including mapped-only misses). These are distinct intervals;
the upload itself is not the only unhidden cost. Front/tail means are
611.369/364.047 us at 30K and 653.703/358.468 us at 150K, about 1 ms per forward.

The GPU-only capture cannot assign these gaps to individual CPU operations.
The notice producer source still performs event observation, journal read,
pager admission/source preparation, then background DMA submission. Existing
`resident_producer_*` host spans can support a bounded diagnostic capture;
they must not be interpreted as measurements from the host-disabled chart.
First separate notification observation, lock wait, admission/copy preparation,
DMA enqueue and queue-start latency; then separate result handoff, recovery
encoding/submit and GPU dependency wait. Mixed DMA/mapped misses require
mapped-write readiness as well; DMA completion alone does not prove every
expert is ready. Preserve pins, reader retirement and transfer dependencies.

For the outer boundaries, separate runner preparation/plan binding and first
submission from final fence completion/readback/result handling before choosing
an optimization. The plotted forward boundary is the runner trace scope, not
the whole API request or all subsequent sampling/serialization. No fence can
be removed merely because its time falls outside the last GPU envelope.

### Completed mapped-copy model screen, 2026-10-11 evening +08:00

Tag `stage4-mapped-copy-matrix` uses the frozen `infr-mapped-copy.exe` above
for both threshold 0 and 6144 KiB. Four cohorts per threshold, three regenerated
512-token rounds each, configured slots=2, no MTP, no profiler. The control runs
first in every cohort, then the candidate; this is not an alternating-order
confirmation. Throughput averages zero-based rounds 1 and 2 in every case.

| Context | Active lanes | Control tok/s/lane | 6144 KiB tok/s/lane | Change |
| --- | ---: | ---: | ---: | ---: |
| 30K | 1 | 46.974 | 47.740 | +1.63% |
| 30K | 2 | 33.449 | 33.757 | +0.92% |
| 150K | 1 | 43.134 | 42.304 | -1.92% |
| 150K | 2 | 25.403 | 25.945 | +2.13% |

The candidate's warmed 150K single rounds are 40.613 and 43.996 tok/s;
both are retained. The corresponding control is 43.070/43.199. No slower
round is filtered or reset. All twelve candidate batches match same-cache
control whole outputs, and twelve batches per side match an available ordinary
golden; no unmatched correctness exclusions or failed requests. All eight test
services exit normally. Sustained single-lane >=50 remains unmet.
Shared test-cache aggregate is 20,916,434,288 bytes (19.480 GiB), below the
30 GiB limit. Production service, executable and cache are unchanged.

Do not promote this as a stable overall improvement or enable the threshold
by default. The module's faster copy does not isolate complete notice-producer
cost or guarantee inference gains. No reverse-order sweep is appended here,
following the user's narrowed focus. Preserve the default-off trial and frozen
source/binary for subsequent CPU-push critical-path diagnosis alongside DMA.
Raw manifest, source patch, logs and correctness file share the tag above;
`stage4-mapped-copy-summary.json` contains all warmed values and module passes.

### Host-enabled miss-boundary diagnosis, 2026-10-11 +08:00

Tag `stage4-miss-boundary-diagnostic` retains the same frozen mapped-copy
control binary and settings (threshold=0, staging/full-graph/owned/inline off).
Two regenerated 512-token requests per depth; twelve forwards from the middle
of the warmed second request. All four outputs match the same-cache control,
without exclusions. Maximum reported clock deviation is 18 ns.

| Per-miss median us | 30K DMA/mixed | 150K DMA/mixed | 30K mapped-only | 150K mapped-only |
| --- | ---: | ---: | ---: | ---: |
| Notice to admission | 5.66 | 5.56 | 5.75 | 6.08 |
| Admission envelope | 18.10 | 17.70 | 171.80 | 183.30 |
| Producer submit envelope | 35.80 | 33.20 | 0.20 | 0.20 |
| Notice to DMA start | 83.80 | 83.82 | n/a | n/a |
| Readiness to resumed GPU FFN | 59.34 | 58.50 | 133.99 | 140.03 |
| Ready to submit API entry | 0 | 0 | 82.50 | 93.10 |
| Submit API overlap after ready | 0 | 0 | 33.60 | 33.50 |
| API return/readiness to GPU start | 47.37 | 49.76 | 11.64 | 11.18 |

Counts across twelve forwards are 105/126 DMA-or-mixed and 51/31 mapped-only.
Readiness is the latest of hit completion, producer submit return and relevant
DMA completion. `host_submit_ns` is stamped **before** the queue-submit API;
the older `queuedUs` includes API time and must not be called pure GPU delay.
`postReadyApi` partitions the interval using the measured API entry/return.
Individual column medians do not add. Post-return time can still include GPU
dependencies and scheduling, not only driver latency.

Mapped-only admission includes CPU writes; producer submit then has no DMA.
The old worker deliberately suppresses ordinary host spans, so the absence of
an `expert_mapped_copy` bar was not evidence that the copy did not occur.
New explicit background spans cover pager-lock acquisition, import lookup,
mapped copy, DMA preparation and its submit API, without enabling ordinary
uploader spans or changing synchronization. GPU-only tracing remains off for
these host spans. The analyzer's four tests cover interval clipping, nested
worker spans, prequeued dependencies and GPU start before submit API return.

Outer front/tail means are 598.91/340.35 us at 30K and 603.42/369.81 us at
150K. First captured query-pool creation contributes a ~1.1 ms measurement
spike; the front measurement median is only ~29-31 us. Keep that sample and
label it as profiling overhead. Mapped recovery preparation also includes
query readback (~37-41 us in representative intervals); do not treat the
whole 83-93 us as production control work. Front graph preparation/binding
is ~174-185 us mean; the final graph/plan destruction is outside `step_tail`
but inside the forward boundary. These are candidates, not measured gains.

### CPU-push transport screens: do not promote

Explicit AVX2 streaming stores: six alternating passes, 1/2/3/4/8 complete
expert payloads, parallel/serial x temporal/non-temporal. All 120 GPU payload
checks pass. Mean differences are below 1%, including 155.40 -> 154.51 us
for one expert in parallel mode. The explicit non-temporal runtime is removed;
`stage4-mapped-nt-module-{source.patch,archive.json}` and frozen unit binary
preserve the rejected implementation and exact provenance.

Bounded host staging/DMA (`stage4-mapped-staging-module`): six alternating
passes, four warmups plus 64 timed iterations per pass; ten Rayon threads.
The prototype routes only notice-worker unimported mapped misses, keeps
imported DMA unchanged, retains at most 32 MiB in its pool and falls back to
direct push when live leases exhaust it. A completed transfer timeline value
permits releasing command-slot buffer references; never release live leases
or wait for the GPU under the pool lock. Allocation alignment can transiently
allocate a rejected buffer before physical-byte accounting rejects retention.
This is not a full host-tier mirror and allocates no additional device VRAM.

| Complete experts | Mapped default total us | Mapped serial total us | Staging prepare us | Staging total us |
| --- | ---: | ---: | ---: | ---: |
| 1 | 153.19 | 140.74 | 112.98 | 287.54 |
| 2 | 309.05 | 382.27 | 220.22 | 487.29 |
| 3 | 458.10 | 435.02 | 353.10 | 858.46 |
| 4 | 579.28 | 563.32 | 456.61 | 1020.01 |
| 8 | 1147.74 | 1127.07 | 906.66 | 1789.94 |

Total includes source preparation, submission and host completion wait;
there is no hit-FFN/CPU-encode overlap in this fixture. Five simultaneously
prepared distinct 1 MiB payloads and ninety guarded GPU readbacks pass;
every staged timed batch actually uses DMA. All passes are retained, including
the slower two-expert serial case. The faster preparation alone does not make
the end-to-end path faster. Leave the experimental runtime **default off and
uncommitted**; do not launch a full matrix or claim inference gains from this
screen. A future staged trial needs GPU completion timing and a demonstrated
overlap budget sufficient to cover the extra copy/submit cost first.

The source patch and frozen unit executable hash are in the module archive;
test binary SHA256 is
`3eed5c9bbe1a8afcee021e0c2184ca5b8dff666039752fa2587689e4f7973ba9`.
Current core/Vulkan non-ignored suites pass 244/223 tests (111 Vulkan tests
ignored); the mapped fixture now uses the existing typed device selection,
not an unregistered ambient runtime environment knob. These are module
checks, not sustained single-lane >=50 tok/s or a long-term stability result.

### Detailed-worker confirmation, 2026-10-11 22:32-22:35 +08:00

Tag `stage4-miss-detail-diagnostic`, frozen build
`infr-stage4-miss-detail-build.exe`, SHA256
`b20846176ea0ef8499588429e2f4690a89d401f65fb378d2412e1428f32ac6eb`.
Build provenance is `stage4-miss-detail-build.json`: 428 runtime source hashes,
source patch, Rust 1.97.1/native flags and pre-build HEAD `cfe335d22`.
The measurement-only code is separately committed as `ea4d86e0c`;
the binary also contains existing uncommitted A/B code, all disabled in this
capture. In particular the staging candidate is explicitly false.

Both depths retain twelve forwards; all four 512-token outputs match the
same-cache previous control, with no exclusions. The new worker events
confirm no mixed mapped writes in the 105/126 DMA cases. Every mapped-only
case has a measured copy event (51/31). A phase measured as zero at clock
resolution is retained as zero, not mislabeled missing.

| Worker median us | 30K | 150K |
| --- | ---: | ---: |
| Pager lock wait, DMA cases | 0.10 | 0.10 |
| Import lookup, DMA cases | 0.70 | 0.70 |
| DMA preparation including submit API | 36.90 | 34.45 |
| Dedicated submit API, nested in preparation | 18.10 | 17.20 |
| Actual mapped-copy batch, mapped-only cases | 172.20 | 164.80 |
| Total mapped admission | 194.60 | 189.30 |
| Mapped readiness to main submit API entry | 89.50 | 93.00 |
| Mapped main submit API overlap | 35.60 | 35.10 |
| Mapped API return to GPU start | 10.74 | 10.84 |

These are nested/overlapping per-case medians, not additive components.
CPU writes dominate mapped admission; lock/import lookup do not. Microtests
move complete three-bank expert payloads, whereas a real miss can need only
some banks: do not substitute complete-expert fixture times for every miss.
One 30K mapped notice-to-admission interval is 5.427 ms versus a 6.42 us
median; it is retained, its cause is not established, and it must not define
the typical notice response. Dedicated preparation has a 394.7 us 150K
outlier; retain it rather than silently dropping it.

The front/tail medians are 563.54/400.77 us at 30K and 523.83/333.14 us at
150K. First-capture query creation/measurement spikes remain (1.088/2.653 ms),
so front means are 670.05/783.25 us and cannot represent unprofiled overhead.
Outer CPU graph preparation/binding means are 210.40/194.95 us; tail graph
cleanup remains an uninstrumented component within `lowering_other`.
Do not remove synchronization or call profiler cleanup an inference win.

Warmed whole-request rates are 47.325/43.342 tok/s. This is a short
host-enabled diagnostic, not a matched unprofiled performance comparison;
single-lane >=50 remains unmet. Both services exit normally; aggregate shared
test cache ends at 19,049,843,128 bytes, below 30 GiB. Production executable,
service and KV cache are unchanged.

Next bounded implementation candidates, respecting the narrowed scope:

1. Separate mapped admission/address planning from the host copy, letting the
   main thread record recovery while the worker copies. Publish destination
   addresses early, but submit only after copy readiness; retain source pins,
   destination readers and failure semantics. This can hide preparation,
   not eliminate the physical PCIe write.
2. For DMA, reduce actual command preparation/recovery work; distinguish its
   ~17-18 us API cost from dependency/scheduling latency. A wait-stage change
   needs scoped consumer/barrier proof and resumed-FFN timing, not an earlier
   TOP timestamp that merely moves the plotted wait.
3. For outer boundaries, isolate repeated graph build/clone/drop and final
   result handling, then screen one change at a time without profiling.
   Previous owned-graph correctness exclusions still forbid broad promotion.

Do not retune L0/L1, full-hit gaps, kernels, LRU or MTP in this focused round.

### Rejected early-publication experiment, 2026-10-11

The first candidate above was implemented in four bounded variants and is
rejected. Publishing a miss result before the dedicated transfer submission
completed changed generated output even when the transfer timeline semaphore
was attached to the recovery submission. The affected tags were
`phase1-overlap-screen-a`, `phase1-overlap-screen-c` and
`phase1-overlap-screen-d`; the final timeline-semaphore variant produced
SHA256 `78df...` versus control `6af78aa...`. An apparent host-DMA-disabled
pass did not exercise the producer path and is not evidence of correctness.

The experiment's deferred pager upload and `Preparing` result state were
removed. Keep the correct ordering: prepare and submit the transfer, publish
the notice result, then let recovery consume the returned timeline value.
Real-model exact-output validation caught a dependency that the synthetic GPU
fixture did not model. Future overlap work must preserve this ordering unless
it supplies a stronger, explicitly proven ownership and queue-dependency
protocol.

### Phase 1/2 stable closeout, 2026-10-12 +08:00

Tag `phase12-final-20261012-final-stable` uses release binary SHA256
`de3dd904333d8e36f86cb9bc65606052af104827947f876cc3fb5bfd83b20ca7`,
two configured slots, no MTP or profiler, three regenerated 512-token rounds,
and the same 16-layer resident/hit-first settings used by the preceding A/B
campaign. The first round is retained as cold runtime state; the table reports
the two subsequent stable rounds. All requests complete normally, every lane's
whole-output SHA256 is identical across all three rounds, and the final 30K
single output also matches the prior exact control/A/B campaigns.

| Context | Active lanes | Stable tok/s/lane | Stable aggregate tok/s |
| --- | ---: | ---: | ---: |
| 30K | 1 | 52.201 / 52.657 | 52.201 / 52.657 |
| 30K | 2 | 34.713 / 34.911 | 69.426 / 69.823 |
| 150K | 1 | 46.275 / 46.178 | 46.275 / 46.178 |
| 150K | 2 | 26.023 / 25.998 | 52.046 / 51.995 |

The 30K single-lane stable latency is 19.157/18.991 ms per token. This meets
the Phase 1 requirement of stable >=50 tok/s and <20 ms per token. The Phase 2
55 tok/s target remains unmet: the measured stable ceiling is 52.657 tok/s,
4.3% below 55 and 12.2% below 60. Do not describe 55/60 as achieved.

Promoted changes are limited to the measured positive paths:

1. Context-gated ordinary-Decode full graph plus bounded mapped-copy policy.
2. Bit-identical Q8_0 fixed-projection Wave32 GEMV on measured Windows/RDNA3
   Decode shapes.
3. Windows/RDNA3 immutable-codebook expert NR2 policy and exact softmax/sigmoid
   Wave32 router selection.

The cross-layer HC candidate, resident Gate/Up/SwiGLU variants, one-token GDN
split, QSA indexed direct-read, IQ2/IQ3 dp4a, PLE successor prefetch, owned
graph, scratch inline and mapped staging were all screened and removed. Some
module kernels were faster, but no retained candidate may change output hashes
or reduce matched end-to-end throughput. Further progress toward 55/60 needs a
new exact critical-path change rather than combining these rejected switches.
