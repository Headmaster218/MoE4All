# infr docs

Design docs, backend architecture, performance playbooks, and campaign logs for
the `infr` inference engine. The top-level project overview lives in the root
[`README.md`](../README.md); everything here is deeper reference.

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

- [rocm-plan.md](rocm-plan.md) — the native ROCm/HIP AMD GPU backend
  (`infr-rocm`), **built**: correctness + real perf + all three paging modes.
  Now a ROCm↔Vulkan feature-parity tracker (all-quant fast kernels, attention,
  fusion, device sampling, multi-GPU, ≥1.0× endgame).
- [backend-unification-plan.md](backend-unification-plan.md) — audit + staged
  plan to unify shared/device-agnostic host logic across the CPU/Vulkan/Metal/
  ROCm backends (peephole fusion, graph-executor skeleton, chat/session
  wrappers, tiering policy, decode-spec + parity harness) into one seam.
- [config-plan.md](config-plan.md) — PLAN ONLY: replace the 138 `INFR_*` env
  gates with one layered `Config` (CLI flags > env > config file > defaults),
  resolved once and passed explicitly, so tests configure behaviour with a
  struct instead of mutating the process environment. Full knob inventory +
  slice-by-slice migration recipe.
- [cuda-plan.md](cuda-plan.md) — phased plan for a native CUDA NVIDIA GPU
  backend (`infr-cuda`), sibling of the ROCm plan; Tensor Cores / cuBLASLt /
  CUDA Graphs, validated on remote NVIDIA hardware. Not yet built.
- [sycl-plan.md](sycl-plan.md) — phased plan for a native Intel oneAPI/SYCL GPU
  backend (`infr-sycl`) on the Level Zero + SPIR-V substrate; XMX / oneDNN,
  validated on remote Intel Arc hardware. Not yet built.
- [plan.md](plan.md) — the original master project plan (historical). Most of it
  shipped against autoregressive decoders; kept for context.
- [train.md](train.md) — LLM training support plan (not yet built).

## Audit

- [audit.md](audit.md) — module-by-module codebase audit for bugs, correctness,
  perf, DRY, and YAGNI.
