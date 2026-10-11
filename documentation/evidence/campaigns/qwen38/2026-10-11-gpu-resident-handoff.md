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
