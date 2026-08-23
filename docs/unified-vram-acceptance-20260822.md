# Unified VRAM acceptance — 2026-08-22

> Historical first-stage result. The persistent-Embedding and separate-runtime-reserve behavior
> described below was superseded by the fully elastic design accepted on 2026-08-24; see
> [`unified-vram-elastic-acceptance-20260824.md`](unified-vram-elastic-acceptance-20260824.md).

## Result

The native LLM and native Embedding engine now share one logical VRAM arena. The arena may span
multiple physical Vulkan allocations, but allocation, eviction, reuse, and accounting are global.
Embedding weights are persistent allocations in that arena; Embedding runtime buffers borrow cold
expert slots and restore the affected expert generations in place when released.

This completes the intended first version of unified VRAM management. Vision and draft-model
allocation classes are reserved for later integration. Dynamic KV allocation is intentionally out of
scope.

## Implementation

Local commits, in dependency order:

1. `5aeb58b` — unified logical range allocator.
2. `02250a3` — shared mapped-ReBAR backing shards.
3. `c944e04` — pager loans cold expert slots to auxiliary engines.
4. `bc4fc93` — native Embedding uses the LLM Vulkan device, queue, and unified arena.
5. Final acceptance commit — fixes auxiliary execution locking and exposes startup accounting.

The final locking rule is:

- Primary LLM execution holds a read lease while submitted work may reference expert slots.
- Auxiliary execution does not hold that read lease while recording its static graph because graph
  recording may lazily allocate transient buffers.
- An auxiliary allocation that needs expert slots takes the write lease and therefore waits for any
  in-flight LLM execution instead of invalidating its slots.

This fixes the first-request deadlock caused by attempting to upgrade the same thread from a read
lease to a write lease while the unified arena was full.

## Memory acceptance

Configuration: APEX Balanced, Q8 KV, 200K synthetic context, 8 GiB expert-cache setting.

| Category | Bytes | Display size |
|---|---:|---:|
| Unified arena | 8,589,426,688 | 8.00 GiB |
| Expert cache | 8,314,765,312 | 7.74 GiB |
| Embedding weights | 273,530,880 | 260.86 MiB |
| Free after startup | 1,130,496 | 1.08 MiB |
| Embedding runtime loan observed | 1,572,864 | 1.50 MiB |

The runtime loan evicted two cold expert slots. No separate complete Embedding VRAM pool was
allocated. The full Host MoE payload remained one 23,571,988,480-byte store, with zero bytes of a
second GPU-visible Host payload mirror.

Raw service log: `target/perf/unified-vram-acceptance.stdout.log`.

## API and numerical acceptance

- `/v1/embeddings`: 768 dimensions, finite values, L2 norm 1.
- `/v1/chat/completions`: two requests completed after Embedding execution.
- The chat model emitted the sampled response in `reasoning_content`; empty ordinary `content` at a
  32-token limit was model behavior, not an API failure.

Native Embedding compared with the existing llama.cpp oracle:

| Case | Cosine similarity | Max absolute error |
|---|---:|---:|
| Chinese short | 0.999974766 | 0.000803143 |
| English short | 0.999965175 | 0.000919986 |
| Semantic pair | 0.999963835 | 0.001191165 |
| Batch 8 | 0.999966875 | 0.001312540 |
| Long input | 0.999955788 | 0.001098613 |

Raw results:

- `target/perf/unified-vram-api-results.json`
- `target/perf/unified-vram-embedding-parity.json`

## Performance acceptance

All measurements below use APEX Balanced and an 8 GiB expert-cache setting. Prefill uses 4096 new
tokens and two repetitions. Decode uses 1000 tokens; the original 500-token smoke values are not
used for comparison because they overweight cold-cache startup.

| KV | Depth | Operation | Historical matrix | Unified VRAM | Delta |
|---|---:|---|---:|---:|---:|
| Q8 | 0 | Prefill | 2881.8 t/s | 2855.6 t/s | -0.9% |
| Q8 | 250K | Prefill | 477.3 t/s | 477.9 t/s | +0.1% |
| F16 | 0 | Prefill | 2904.7 t/s | 2844.2 t/s | -2.1% |
| F16 | 250K | Prefill | 431.5 t/s | 437.8 t/s | +1.5% |
| Q8 | 0 | Decode | 93.0 t/s | 84.4 t/s | -9.2%* |
| Q8 | 250K | Decode | 41.1 t/s | 41.2 t/s | +0.2% |
| F16 | 0 | Decode | 91.8 t/s | 82.6 t/s | -10.0%* |
| F16 | 250K | Decode | 32.5 t/s | 31.7 t/s | -2.5% |

`*` The historical 0-depth Decode entries were single samples and are inconsistent with the later
three-repeat control range. For example, the pre-unification Q8 d0/c10g control was 84.7 t/s
(83.5–86.3), which contains the present 84.4 t/s result. The present Q8 d0 run also has exactly the
same 95.024% hit rate and hit/miss counts as the historical d0/c8g row. Therefore the 93.0/91.8
samples are retained transparently above but are not sufficient evidence of a unified-arena
regression. Long-context Q8, the primary target, is unchanged.

Raw results:

- Historical matrix: `target/perf/current-full-matrix-20260820/results.csv`
- Prefill and 500-token smoke: `target/perf/unified-vram-final-20260822/results.csv`
- 1000-token Decode retest: `target/perf/unified-vram-final-decode-retest-20260822/results.csv`

## Tests

- Unified allocator and live-GPU tests: 7/7 passed.
- `infr-embedding`: 7/7 passed.
- `ParallelSeam`: 3/3 passed.
- `cargo check` passed for `infr-cli`, `infr-embedding`, and `infr-llama`.
- Release build passed; accepted binary: `target/release/infr.exe`.

No profiler or diagnostic mode is required by the default execution path.
