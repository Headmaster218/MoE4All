# infr docs

Design docs, backend architecture, performance playbooks, and campaign logs for
the `infr` inference engine. The top-level project overview lives in the root
[`README.md`](../README.md); everything here is deeper reference.

## Using infr

- [config.md](config.md) — the configuration reference: the four layers
  (defaults < config file < `INFR_*` env < CLI flags) and their precedence, the
  TOML file format and lookup order, `--set`, and a per-section walkthrough of
  what is tunable. Start here before reaching for an `INFR_*` variable.

## Performance & kernels

- [perf.md](perf.md) — the performance optimization playbook: the measure →
  profile → fix-one-lever loop, the bottleneck taxonomy, benchmarking/profiling
  tooling, and the coopmat-operand-tier dead-end writeup. GPU / general.
- [cpu-perf.md](cpu-perf.md) — CPU (`infr-cpu`) backend performance roadmap: the
  two-regime (decode DRAM-bound / prefill cache-bound) model, the native
  int8-quant landing history, and the remaining worklist.
- [kernels.md](kernels.md) — cross-backend fast-kernel coverage: which weight
  quant formats have a native kernel on CPU / Vulkan / Metal (24/24 on all
  three) and each backend's decode strategy.

## Backends

- [metal.md](metal.md) — Apple GPU backend (`infr-metal`) architecture: the
  `DEC16` decode kernels, decode-parity campaign, multi-slot serve, native-read
  KV, MTP, and the replay-tape correctness fix.
- [igpu.md](igpu.md) — integrated-GPU correctness campaign (AMD APU / Intel iGPU
  / Strix Halo class): the UMA heap-table insight, the per-submit watchdog
  root-cause + submit-splitter fix, and the model survey. Phase 1 complete.

## Models & architectures

- [qwen35.md](qwen35.md) — Qwen3.5 / Qwen3.6 (`qwen35`): the gated-DeltaNet
  linear-attention + full-attention hybrid, and the interleaved q+gate trap.
- [diffusion-gemma.md](diffusion-gemma.md) — DiffusionGemma design for the
  unified seam: block text-diffusion, the canvas denoise graph, and
  self-conditioning.
- [mtp.md](mtp.md) — multi-token prediction (MTP) speculative decoding for
  qwen35's single NextN head (issue #33).

## Roadmaps & history

- [plan.md](plan.md) — the original master project plan (historical). Most of it
  shipped against autoregressive decoders; kept for context.
- [train.md](train.md) — LLM training support plan (not yet built).

## Audit

- [audit.md](audit.md) — module-by-module codebase audit for bugs, correctness,
  perf, DRY, and YAGNI.
- [code-review.md](code-review.md) — risk-prioritised whole-tree review
  (2026-08-01): correctness bugs, security/robustness gaps, `unsafe` soundness,
  DRY, and YAGNI, each with a file:line and a suggested fix. All findings are
  resolved; residual work is in the backlog.
- [backlog.md](backlog.md) — triaged work that is deliberately not done, with
  why (blocked on hardware, scoped out, or declined), plus withdrawn findings
  recorded so they are not rediscovered.
