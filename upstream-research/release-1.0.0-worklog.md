# MoE4All 1.0.0 worklog

Updated: 2026-10-13
Branch: `perf/gpu-resident-decode`

This file tracks implementation state, matched measurements, accepted changes, rejected experiments,
and the commit that closes each task. Performance claims must name the model, context depth,
concurrency, device, and measurement interval.

## Status

| ID | Task | State | Result / acceptance |
| --- | --- | --- | --- |
| 0 | Move the active branch to `D:\AIinfr\infr` | Done | Same branch and dirty state restored in the primary checkout; no merge or branch migration. |
| 1 | Q8 long-context attention optimization | Done | Commit `8d5a4875b`; XTX-only INT8 Q.K policy and depth-adaptive combine accepted. |
| 2 | Qwen3.6-35B vision | Done | Qwen3.6 enabled on the existing Qwen-VL MRoPE path. Small/large image recognition and image -> text -> image passed; later long-text Prefill matched the text-only baseline. |
| 3 | Qwen3.6-35B reasoning effort | Done | Model-scoped prompt mapping for Qwen3.5/3.6; CLI, wizard and API request controls validated while default and no-think prompts remain unchanged. |
| 4 | Review `perf/expert-mtp-optimization` after `release-0.10.0` | Done | No mature runtime optimization can be cherry-picked safely. The only novel experiments are reassigned to Tasks 5/6 for a current-architecture reimplementation and matched A/B. |
| 5 | Decode front/back bubble overlap | Done | No production change. Runner-side setup is already overlapped/small; splitting the first frontend into an earlier submit regressed matched 30K and 150K throughput and was removed. Further gains require reusable command buffers/GPU-driven replay, which is architecture work. |
| 6 | Expert-miss critical path | Done | No production change. Pool-streaming and mapped-copy worker variants were neutral/slower; the aggressive copy/encode overlap also produced nondeterministic output despite a host join before submit, so it was rejected. |
| 7 | Version 1.0.0 and release benchmark | Done | Native benchmark binary and portable static-CRT package built. Both models completed the full matrix; FlashNext 150K x3 exposed one output-stability warning, while all other rows were stable. |
| 8 | Recover MTP experiment and automatic splitter | Plan recorded; implementation pending | The WIP startup-reference splitter is not portable and was not copied. The current global tuner also mixes unrelated graph shapes; an implementation-ready class-isolated replacement and validation matrix are recorded below. |

## Accepted baseline

### Qwen3.6-35B reasoning effort

- The shipped Qwen3.6 template exposes `enable_thinking` but no native `reasoning_effort` variable.
- Explicit effort therefore maps to a Qwen3.5/3.6-only system instruction. Default requests remain byte-identical; non-Qwen templates still reject unsupported effort values.
- The wizard exposes `low`, `medium`, `high`, and `max`; the shared API parser also accepts `xhigh`.
- Actual 35B serving validated default, `low`, `max`, and `enable_thinking=false + max`. Debug-rendered prompts contained only the requested active instruction; no-think omitted it and returned `ready` directly.
- Unit coverage: 79 `infr-chat` tests and 32 launch-wizard tests passed.
- Evidence: `benchmark-data/artifacts/qwen36-35b-reasoning-20261013/run-2026-10-12T17-28-27.412Z/results.json` and `stderr.log`.

### Qwen3.6-35B vision, RX 7900 XTX

- Model: `Qwen3.6-35B-A3B-APEX-I-Balanced.gguf`; projector: `mmproj-Qwen3.6-35B-A3B-BF16.gguf`.
- Device and shape: `Vulkan1`, one 32K slot, Q8 KV, aggressive profile, ubatch 4096.
- Small 128x128 image: correct blue-square answer. End-to-end request Prefill was `53.5 tok/s`; after request-scoped projector work, the 43-token model Prefill was about `128 tok/s`.
- Large 1024x1024 image after a text turn: correct purple/larger-canvas answer. End-to-end request Prefill was `627.6 tok/s`; after projector work, the 4,515-token model Prefill was about `2,157 tok/s`.
- Matched long text: text-only `2,715 tok/s`; after the first image `2,721 tok/s`, with 36 prefix tokens reused. Vision did not leave a persistent text-Prefill slowdown.
- Final no-image recall reused 4,508 of 4,572 prompt tokens and correctly returned `Blue, purple`.
- Evidence: `benchmark-data/artifacts/qwen36-35b-vision-20261013/run-2026-10-12T17-18-33.030Z/summary.json` and `stderr.log`.

### Q8 attention, RX 7900 XTX

- INT8 Q.K continuous Decode: 30K `61.33 -> 62.79 tok/s`; 150K `36.49 -> 36.54 tok/s` in the matched five-run comparison.
- Final default confirmation: 30K `63.66 tok/s`; 150K `36.91 tok/s`.
- The INT8 path is automatic only on Windows RDNA3-class devices with packed integer-dot support.
- Depth-adaptive combine is neutral at 30K and lowers isolated 150K Attention from `1699.34` to `1657.12 us/layer`.
- 1024-key replay remains default-off: its isolated kernel is faster but paged-MoE end-to-end Decode regressed.
- Direct Q8 K write remains default-off: byte-exact but end-to-end neutral.
- Scoped buffer barriers were reverted after the correct hazard-preserving implementation showed no gain.

### `perf/expert-mtp-optimization` audit

- Range reviewed: `release-0.10.0..perf/expert-mtp-optimization`. Only `56a821e17` and
  `c722b854b` change engine runtime or diagnostics; the remaining commits are documentation,
  submodule, or packaging changes.
- The splitter in `56a821e17` is a startup/static rewrite. The current branch instead uses
  runtime pager feedback and adaptive calibration, so copying the old `splitter.rs` would replace
  newer policy rather than add an optimization.
- The `DecodeDmaWorker` experiment moved a frozen dedicated-transfer submit to one helper thread,
  but it predates the current resident notice producer, hit-first execution, early recovery, and
  promotion lifecycle. It had no accepted end-to-end A/B and cannot be transplanted without
  duplicating ownership and cancellation paths. Its useful intent, immediate asynchronous submit,
  is retained as a Task 6 design input.
- `micro_prefill_row_counts` packs several remaining prompt rows beside Decode lanes. It targets
  mixed short-Prefill/Decode batches, not the single-stream per-token front/back bubble, and was
  environment-only with no accepted throughput result. If useful, it must be rebuilt against the
  current scheduler and checkpoint/QSA boundaries; it is not a release-ready optimization.
- `c722b854b` is feature-gated expert omission diagnostics. It adds synchronization, downloads,
  and analysis scripts only when enabled; it has no production performance change to port.
- Result: no code cherry-pick. This avoids reintroducing obsolete splitter and pager assumptions;
  Tasks 5 and 6 will test the two still-relevant ideas on the current architecture.

### Decode front/back bubble experiment, RX 7900 XTX

- Model/config: FlashNext 4.27 bpw, `Vulkan1`, two configured slots with one active, Q8 KV,
  30K/150K contexts, 256 generated tokens, resident hit-first/notice/early-recovery path.
- Existing code already builds the next scheduler plan beside the current execute where dependencies
  permit. Pager profiles put runner front/setup work in the tens of microseconds per step; the larger
  visible pre-GPU interval is primarily Vulkan command recording, queue submission, and device start.
- Experiment: finish and submit the token frontend immediately before the first resident MoE layer,
  then continue recording the layer stack in a fresh recorder. The control and candidate used the
  same binary hash (`c67643d8...`) and differed only by the runtime switch.
- 30K warmed complete-request Decode: control `49.59`, `47.99 tok/s`; early-submit `45.04`,
  `44.60 tok/s` (about 8% slower).
- 150K warmed complete-request Decode: control was noisy at `38.79`, `43.83 tok/s`; early-submit was
  consistently slower at `39.47`, `39.37 tok/s`. Output hashes matched in every pair.
- Conclusion: an extra small submission moves a visual boundary but adds queue/synchronization cost
  and reduces complete-cycle throughput. The code was removed. A meaningful next step would be
  reusable/indirect resident command buffers or a GPU-driven replay loop, not another host submit.
- Evidence: `benchmark-data/artifacts/gpu-resident-decode-20261011/`
  (`release100-frontgap-samebin-ab-control-*` and `release100-frontgap-samebin-ab-frontsubmit-*`).

### Expert-miss critical-path experiments, RX 7900 XTX

- Model/config: FlashNext 4.27 bpw, `Vulkan1`, two configured slots with one active, Q8 KV,
  30K context, 256 generated tokens, resident hit-first/notice/early-recovery path.
- Pool-streaming experiment submitted each independent role-size transfer as soon as that pool was
  prepared. Matched complete-request Decode was neutral (`49.01/48.99` versus control
  `49.33/48.69 tok/s`). Combining it with parallel mapped copies regressed to `48.78/48.40`.
- Disabling the mapped-copy serial threshold alone measured `49.29/49.23 tok/s`, about 0.5% over
  the noisy control and below the acceptance threshold. All pool-streaming code was removed.
- A second experiment moved full-RAM mapped-VRAM copies to a private host worker while the main
  thread encoded miss recovery. The first version submitted before the host copy was joined and
  produced a different output hash on every round. The second joined before queue submit but
  consumed promotion metadata too early; warmup correctly rejected it. The final lifecycle fix
  retained metadata and completed, but output hashes still differed on every candidate round.
- Final same-binary steady comparison: control `49.47/49.52 tok/s`; one copy worker
  `41.39/41.76 tok/s`, about 16% slower. The candidate was therefore rejected on both correctness
  and performance grounds and fully removed.
- Conclusion: mapped host writes cannot safely overlap this recovery window with only a host join.
  A future attempt needs an explicit host-write publication and slot-version protocol, or must copy
  through a transfer submission whose timeline dependency is consumed by the recovery queue. More
  host threads alone do not remove the exposed miss tail.
- Evidence: `benchmark-data/artifacts/gpu-resident-decode-20261011/`
  (`release100-miss-pool-mapped-ab-*`, `release100-miss-copyworker-ab-v1-*`,
  `release100-miss-copyworker-ab-v2-*`, and `release100-miss-copyworker-ab-v3-*`).

### Release 1.0.0 benchmark, RX 7900 XTX

- Native benchmark binary: `infr 1.0.0`, SHA-256
  `8a3c80cd8f8872778ae2425568f108562353d051a3aa002edff0d3819feeafff`.
- Runtime source commit: `fabf3e614`; the later `5b949cd63` changes only benchmark reporting so an
  unstable row is retained instead of terminating the remaining matrix.
- Common Decode setup: `Vulkan1`, three configured slots, 1/2/3 active streams, Q8 K/V, aggressive
  automatic profile, ubatch starting at 4096 with physical fallback, MTP and expert prefetch disabled,
  512 generated tokens, one warm-up plus three measured rounds.
- Front/middle/late, average and peak are aggregate token rates across all active streams. Average is
  total completion tokens divided by the slowest lane's Decode duration; peak is the maximum fully
  contained one-second server Decode window.

FlashNext 4.27 bpw Decode (`tok/s`):

| Context | Active | Front | Middle | Late | Average | Peak | Three-run output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 30K | 1 | 45.50 | 51.29 | 50.34 | 49.15 | 53.83 | Stable |
| 30K | 2 | 65.93 | 66.58 | 62.85 | 65.17 | 73.17 | Stable |
| 30K | 3 | 71.69 | 68.67 | 68.15 | 69.60 | 77.03 | Stable |
| 150K | 1 | 42.95 | 45.04 | 44.27 | 44.16 | 47.70 | Stable |
| 150K | 2 | 51.96 | 48.86 | 47.83 | 49.50 | 56.70 | Stable |
| 150K | 3 | 47.35 | 45.87 | 43.29 | 45.47 | 51.07 | Unstable lanes 0 and 2 |

Qwen3.6-35B Decode (`tok/s`):

| Context | Active | Front | Middle | Late | Average | Peak | Three-run output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 30K | 1 | 81.79 | 80.24 | 79.12 | 80.49 | 81.93 | Stable |
| 30K | 2 | 119.41 | 117.45 | 117.37 | 118.34 | 120.07 | Stable |
| 30K | 3 | 112.09 | 110.24 | 109.50 | 110.78 | 113.33 | Stable |
| 150K | 1 | 42.07 | 41.69 | 41.71 | 41.91 | 42.67 | Stable |
| 150K | 2 | 51.19 | 50.59 | 50.50 | 50.87 | 52.60 | Stable |
| 150K | 3 | 49.38 | 49.11 | 49.04 | 49.27 | 51.90 | Stable |

Cold Prefill (`cached_n=0`):

| Model | 30K | 150K |
| --- | ---: | ---: |
| FlashNext 4.27 bpw | 1414.75 tok/s | 1445.50 tok/s |
| Qwen3.6-35B | 2749.06 tok/s | 1169.40 tok/s |

- FlashNext 150K x3 is a real release finding, not sampling noise: requests used temperature zero and
  fixed per-lane seeds, and lane 0/2 diverged from the beginning of the generated text. The raw rounds
  remain available for state/scheduler diagnosis; no correctness claim is made for that row.
- Evidence roots:
  `benchmark-data/artifacts/release-1.0.0-benchmark/2026-10-12T19-51-47-832Z-flashnext-4.27bpw`,
  `2026-10-12T20-15-46-153Z-flashnext-4.27bpw-prefill`, and
  `2026-10-12T20-20-14-502Z-qwen36-35b-apex-i-balanced`.
- The local CMD binary at `target/release/infr.exe` is the benchmarked native build. The Windows
  release archive uses `target-cpu=x86-64` and static CRT; its executable SHA-256 is
  `71f57f34c9043a98146f0c1dd896cbc6e03bacbb6069c556c60e51aa28998cc8`.
- Release archive: `dist/MoE4All-Windows-x86_64-v1.0.0.zip`, SHA-256
  `ecf8852fbaf8025890afd5d5436392ba835375913a64c8d0fa934116314c8428`. The extracted package passed
  install-manifest verification, update-transaction checks, command/wizard dry runs and dependency
  inspection. An earlier native package correctly failed the dependency check and was replaced.

### MTP experiment and automatic splitter recovery

- `research/flashnext-mtp-acceptance` does not contain a newer independent MTP implementation; it
  currently points at `8d5a4875b`, which this branch already contains. The relevant unfinished work is
  commit `56a821e17` on `perf/expert-mtp-optimization`.
- Its replacement `splitter.rs` calibrates one synthetic 16-row, width-2560, dense-F16 two-block
  graph at startup. It then scales that result with row-count multipliers measured locally at about
  22 us/dispatch. The graph excludes quantized model kernels, paged experts, GDN/QSA/HC, MTP
  target/head work and real ubatch shapes. The row multipliers are device-specific. Copying this
  implementation would therefore replace newer runtime feedback with a less representative policy.
- The current branch has the opposite failure mode: one process-global `SubmitAutoTuner` samples the
  first eligible Prefill/batched forwards, then freezes after a bounded calibration window. Text,
  MTP, vision and embedding graphs can therefore contaminate each other's cap according to request
  order. Pager-mandated submits are also included in whole-forward timing even though the splitter
  did not create them. Short services may never finish calibration and can later mix a different
  workload into the partially armed tuner.
- Release benchmark evidence matches this diagnosis. Aggressive automatic mode started at
  `split/16` and usually reached `split/256` after ten forwards, but the measured submissions were
  not homogeneous: FlashNext 150K x3 recorded 395 timed submits at about 711 us/submit with a
  67.6 ms longest submission, while Qwen3.6-35B 150K x3 recorded 490 at about 857 us/submit with a
  161 ms longest submission. These aggregates cannot be used as a clean dispatch-cap signal.

Repair plan:

1. Add a caller-selected `SubmitClass`; never infer it from tensor labels or maximum graph rows.
   Classes are text Prefill (model family, row band, paged/dense), MTP prime/catch-up, MTP verify,
   vision, and embedding. Single-token paged Decode keeps its pager/platform cap and does not train
   the Prefill tuner.
2. Replace the one global cap/tuner with independent state per class and row band. Each state owns
   its cap, warmed sample count, splitter-created segment count, maximum segment time and frozen
   status. Explicit `device.submit_dispatches` remains authoritative and bypasses every tuner.
3. Tag submit cause at recorder close. Feed a tuner only command buffers closed by its splitter;
   exclude pager rotations, router readback, DMA/acquire, resident hit/miss recovery, compilation
   and first-touch work.
4. Start from a platform-safe cap, collect two warmed samples, shrink immediately when the maximum
   splitter segment exceeds the watchdog budget, and grow geometrically only when the splitter
   actually created boundaries and timing has margin. Freeze a class after convergence or a no-split
   sample; opening a new class/row band creates a fresh bounded state instead of thawing old ones.
5. Report class, row band, starting/final cap, splitter-created segments, mean/maximum GPU segment
   time and stop reason. This makes automatic decisions reproducible without sampling user content.
6. Unit-test class isolation, explicit override, shrink/grow/freeze transitions and exclusion of
   pager submissions. GPU validation covers text Prefill at ubatch 256/1024/2048/4096, MTP
   prime/verify, small/large vision, embedding and an integrated-GPU watchdog run.

Acceptance requires byte-identical outputs, request-order-independent convergence within each
class, every measured splitter segment below its budget, and no more than 1% throughput regression
against the best fixed safe cap. Only the explicit workload taxonomy and the intent to avoid mixed
user-request sampling are retained from `56a821e17`; its synthetic graph and hardcoded multipliers
are rejected.

### Known release follow-up

- FlashNext 150K x3 produced unstable measured output on lanes 0 and 2 while temperature, seeds and
  prompts were fixed. The pattern converged over later rounds but is not enough to attribute the
  cause to stale SSD KV alone. Do not invalidate the cache format on that evidence.
- A focused follow-up must compare three isolated state origins with the same binary and payload:
  fresh full Prefill, live-slot seed, and process-restarted SSD restore. Capture recurrent/PLE
  checkpoint hashes before Decode and compare the first divergent token. This is a correctness
  investigation, not part of splitter calibration.

## Measurement rules

- Enumerate devices and verify RX 7900 XTX by identity before testing; Vulkan indices can change between starts while the 2 GiB adapters are present.
- Follow `D:\AIinfr\disk_usage_rule.md`. Generated data belongs under the documented categories of `D:\AIinfr\infr\local-work`; the old B-drive build rule is retired. Do not move source, Git data, models, user state or evidence as part of cache management.
- Separate isolated-kernel timing from complete-cycle throughput.
- For Decode, discard warm-up, report three warmed runs, and define front/middle/back windows before collection.
- Preserve unrelated `harness/plugins/dsh-llm-moe4all` and `test-kv-cache` state.

## Maintenance follow-up, 2026-10-23

- This remains the historical 1.0.0 measurement log. The source snapshot for this entry was
  `release/1.0.0-maintenance` at `0bbc19b79`; official `release-1.1.0` resolves to
  `0ee28889ad0f4d1e2ff19704284a93c0932f7a52`. No build or GPU run was performed for this documentation update.
- `689c9cd22` enables resident paging by default; `4d224762c` generalizes full-graph eligibility
  while retaining host-interlock and mode gates; `091278d97` unifies the eight-row ABI and reserves
  LM-head scratch before expert mappings freeze. CPU still owned cache allocation/LRU in that
  maintenance snapshot; this does not describe the later GPU-owned Decode branch.
- MTP resident VERIFY (`2180c9a77`), four-row Q8 fixed projections (`6477de582`), very-wide
  projections (`f006c165b`) and Q8 long-context attention (`8d5a4875b`) are in both maintenance
  HEAD and the release tag. Device/shape gates remain; INT8 query is not a bit-exact rewrite.
- `bc427a7f5` rejects non-finite output; `c08f5bd1d/ccfaea3d8` fix Q8 tail/store contracts;
  `c7eedfba8` preserves quantized KV planes during shared-prefix copies. These do not independently
  prove that every historical FlashNext 150K x3 divergence has been eliminated.
- Workload-isolated automatic splitter tuning is still a proposal: the current backend retains
  a global `SubmitAutoTuner`. Do not label the repair plan above as completed implementation.
- Recent official-1.1.0 fixed-token smoke results and raw evidence are under
  `local-work/results/infr/20261022-164526-standard-benchmark-1-1-0-b71d/README.md`.
  FlashNext partly overlapped a model download, and 35B aggressive failed before the separately
  labelled conservative run. Do not use these results as a matched replacement for the old matrix.
- Ornith 122B Q2_K versus Q3_K_S is under
  `local-work/results/infr/20261022-192955-ornith-q2-compare-8a31/README.md`:
  6K Prefill 1269.9 versus 1176.8 tok/s; Decode 33.3 versus 26.3 tok/s. Both use explicit
  extra 768 MiB VRAM reserve, Q8 KV and no MTP. One run per quant; not an engine-kernel A/B.
- Current upstream cache and overview remain in `D:\AIinfr\infr\upstream-research`, with
  `status-20261023.md` and per-source notes. This maintenance checkout tracks this worklog only;
  no research source cache or generated benchmark evidence is added to release Git history.
- Same-day discovery adds `projects-20261023.md`: 32 repository endpoints checked, 31 accessible,
  26 newly tracked sources and 1,464 observed branch entries (some lists are capped). Selected
  Colibri branches are already ancestors of dev; other fork lineages are not counted as independent
  speedups. Nathanw's Vulkan IQ views, SelfRef, FreeToken, ik_llama and mixed-backend/MTP sources
  are research inputs, not new local performance claims. No runtime code, build or GPU work.

## Five-model upstream coverage, 2026-10-23

- This documentation-only follow-up was recorded on `release/1.0.0-maintenance@0bbc19b79`.
  Existing worklog edits are preserved. No build, service change or GPU execution.
- Read GGUF metadata and tensor descriptors across the actual 35B APEX, Ornith
  Q2_K/Q3_K_S and FlashNext AD-4.27/UD-Q2 inputs, without reading weight payloads.
  Trunk experts and embedded NextN tensors are accounted separately across shards.
- APEX experts are mixed Q5_K/Q6_K; FlashNext AD is IQ2_S/IQ3_S/IQ4_NL,
  UD is IQ2_XS/IQ3_XXS/IQ4_NL. Ornith uses mixed k-quants, not IQ formats.
- Broadened official/research/community/edge engine and kernel-library coverage,
  including DeepInfer, DeepGEMM, DeepSelect, TileKernels, FlashMLA, DeepEP, FLA,
  Q36, reinstinct and the separate vLLM GGUF plugin. Fixed source snapshots,
  README-only observations and selected implementation reads remain distinct.
- The user-approved retention rule is now correctness plus reproducible unit/
  module benefit and no matched end-to-end regression. A universal 1% end-to-end
  uplift is not required. Masked positive modules, actual regressions and numerical
  failures are kept as separate records; existing negative measurements are not erased.
- Local entry points under `D:\AIinfr\infr\upstream-research` are
  `engines-catalog-20261023.md`, `model-coverage-20261023.md`,
  `model-quantization-details-20261023.md`, `optimization-atlas-20261023.md`
  and `optimization-acceptance.md`. Raw research evidence is under
  `local-work/evidence/infr/20261023-030346-engine-model-coverage-b8c4`.
- No research clones, model inputs, generated tables or raw snapshots are added
  to release Git history by this entry; no implementation is marked validated
  merely because an upstream kernel exists.
