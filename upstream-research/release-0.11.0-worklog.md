# MoE4All 0.11.0 worklog

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
| 7 | Version 0.11.0 and release benchmark | Pending | FlashNext 4.27 bpw and Qwen3.6-35B; 30K/150K; 1/2/3 active streams; warmed three-run Decode front/middle/back/mean/peak plus Prefill. |
| 8 | Recover MTP experiment and automatic splitter | Pending | Audit the MTP branch, reproduce the automatic splitter failure, and produce an implementation-ready repair plan or a validated fix. |

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
  (`release011-frontgap-samebin-ab-control-*` and `release011-frontgap-samebin-ab-frontsubmit-*`).

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
  (`release011-miss-pool-mapped-ab-*`, `release011-miss-copyworker-ab-v1-*`,
  `release011-miss-copyworker-ab-v2-*`, and `release011-miss-copyworker-ab-v3-*`).

## Measurement rules

- Explicitly use `Vulkan1` (RX 7900 XTX); never rely on automatic selection while the 2 GiB adapters are present.
- Put new Cargo build output under `B:\` and keep source, records, and release artifacts under `D:\AIinfr\infr`.
- Separate isolated-kernel timing from complete-cycle throughput.
- For Decode, discard warm-up, report three warmed runs, and define front/middle/back windows before collection.
- Preserve unrelated `harness/plugins/dsh-llm-moe4all` and `test-kv-cache` state.
