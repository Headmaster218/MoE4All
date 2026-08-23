# Performance

Everything about how fast infr is, how it got that way, and how to measure it
yourself. Start here rather than in the root README — that only carries the
headline.

## What is where

- **[results.md](results.md)** — the numbers. Every validated model × quant
  against llama.cpp on an RX 7900 XTX, the per-row footnotes explaining each
  kernel slice that moved a column, and an honest accounting of where infr still
  loses. This is the file to regenerate after a perf campaign.
- **[benchmarking.md](benchmarking.md)** — how to produce those numbers:
  `infr bench` / `infr compare --sweep` flag-for-flag against `llama-bench`,
  per-op GPU profiling with `INFR_PROF_OPS`, shape-itemised buckets, and the
  CPU-side `samply` loop.
- **[playbook.md](playbook.md)** — the optimization method: measure → profile →
  one lever at a time, the bottleneck taxonomy, and a long tail of dead ends
  recorded so they are not re-tried. Read before starting a perf slice.
- **[kernels.md](kernels.md)** — cross-backend fast-kernel coverage: which
  weight-quant formats have a native kernel on CPU / Vulkan / Metal, and each
  backend's decode strategy.
- **[cpu.md](cpu.md)** — the CPU backend's own roadmap: the two-regime model
  (decode DRAM-bound, prefill cache-bound), the native int8-quant landing
  history, and what is left.
- **[vulkan-review.md](vulkan-review.md)** — multi-vendor review of the Vulkan
  backend: what is RDNA3-tuned versus genuinely portable, and the per-vendor
  gaps (Intel Arc, NVIDIA) that follow from it.
- **[deepseek-v4-flash-rx7900xtx-closeout-20260824.md](deepseek-v4-flash-rx7900xtx-closeout-20260824.md)**
  — DeepSeek V4 Flash bring-up and performance closeout: commit impact audit, retained and rejected
  optimizations, cache trace provenance, full-shadow decision and capacity simulations.

## Reading the numbers

Two things bite people:

**Ratios are against a moving target.** Every ratio is `infr ÷ llama.cpp` on
matched flags, so it shifts when _either_ side changes. A row that drops between
two snapshots may mean upstream got faster, not that infr got slower — the
provenance block in [results.md](results.md) names the exact `llama-bench` build
each snapshot used, and comparing across different oracles is invalid.

**To ask "did WE regress?", compare infr against infr.** Build the older commit
and A/B it on the same GPU, alternating order with cooldowns between runs. The
ratio table cannot answer that question and should not be used to try; see
[playbook.md](playbook.md) for the protocol and the measurement traps
(`pp4@d4096` in particular is ordering-sensitive, and a shape-itemised profile
is the only way to confirm the timed window holds the work you think it does).
