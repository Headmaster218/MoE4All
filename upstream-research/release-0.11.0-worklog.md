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
| 5 | Decode front/back bubble overlap | Pending | Pipeline safe pre/post work around the GPU submission; measure 30K and 150K, reject changes without end-to-end gain. |
| 6 | Expert-miss critical path | Pending | Start DMA/CPU copy immediately after miss IDs, move deferrable work off the transfer path, and overlap independent S5/S6 copies where correctness permits. |
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

## Measurement rules

- Explicitly use `Vulkan1` (RX 7900 XTX); never rely on automatic selection while the 2 GiB adapters are present.
- Put new Cargo build output under `B:\` and keep source, records, and release artifacts under `D:\AIinfr\infr`.
- Separate isolated-kernel timing from complete-cycle throughput.
- For Decode, discard warm-up, report three warmed runs, and define front/middle/back windows before collection.
- Preserve unrelated `harness/plugins/dsh-llm-moe4all` and `test-kv-cache` state.
