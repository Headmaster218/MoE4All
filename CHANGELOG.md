# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `infr run`, `infr bench` and `infr serve` now notice the model file being
  overwritten underneath the live weight mapping and fail with a named error
  instead of serving output from weights that no longer match the file. `run`
  checks at both ends of every turn, `bench` before reporting any numbers, and
  `serve` at the start of each request. New `infr_gguf::watch::WeightWatch`,
  re-exported as `infr_llama::WeightWatch`.

- `INFR_DRAM_CACHE` / `paging.dram`: a host weight cache, read from the model
  file into a bounded arena under the engine's own cyclic-sweep eviction policy
  instead of being mapped and left to the OS page cache. Off by default (the
  zero-copy mmap path is unchanged and is right whenever the weights fit); a
  budget too small to seat a weight class leaves that class mapped and says so.
  - **CPU backend**: every weight above 1 MiB. Measured on a memory-capped
    Llama-3.2-1B F16: decode 2.06x faster at a 1.5 GB cap with 210x fewer major
    faults, prefill 3-7.5% slower (`docs/perf/results.md`).
  - **Vulkan backend**: a third tier under both dense weight streaming and the
    paged MoE expert cache, so a VRAM miss resolves against the arena and
    reaches the file only when that misses too. MoE pages ONE EXPERT at a time
    rather than a whole bank. A block the arena has no room for is read straight
    into the staging ring instead of evicting one, so the streaming majority
    costs one copy rather than two. **Still measured slower than the mmap path
    it replaces** (0.79x decode), so enable it because you are out of RAM, not
    for speed — `docs/perf/results.md` has the table and what is left of the
    gap.
  - `INFR_PAGER_STATS=1` reports hit rate, reads and bytes for each tier.

### Changed

- The Vulkan context window is now re-decided against the memory the device
  reports free once the weights are resident, instead of only against a pre-load
  estimate of them. That estimate is systematically light — the weight footprint
  prices tensor bytes while the resident-BDA arena commits them into ≥64 MiB
  blocks (measured +2.20% on gemma-4-31B, +2.43% on gemma-3-12b, +1.16% on
  Qwen3-14B), and no footprint has a term for the driver's own pipeline and
  descriptor memory. Sessions whose window used to be advertised and then fail
  mid-prefill on a `VRAM budget exceeded` now get a window they can fill. The
  clamp logs what it measured, only ever shrinks, and leaves a context set
  explicitly via `--ctx`/`INFR_CTX` alone.
- The activation reserve is re-fit to measured peaks and its interim 1.5x safety
  margin is gone, so gemma-3-12b now serves its full 131072-token f16 window at
  the default 1024-row prefill chunk (780 t/s, was 760 at the 256-row rung it
  used to be pushed onto). The reserve gained explicit terms for MoE expert
  scratch and for qwen35's DeltaNet mixer, both of which it previously
  under-counted.
- New `Backend::device_alloc_room` and `Backend::activation_peak`, both
  defaulting to `None` for backends that cannot report them (CPU, Metal — those
  keep their existing behaviour unchanged). The second is a high-water mark of
  live activation bytes that the runner compares against what it reserved,
  warning when a generation's real peak exceeds the prediction.

### Security

- Update `crossbeam-epoch` 0.9.18 → 0.9.20 for RUSTSEC-2026-0204 (invalid
  pointer dereference in the `fmt::Pointer` impl for `Atomic`/`Shared`). Reached
  through `rayon`, so it applies to every CPU-backend build.

### Fixed

- Reject GGUF tensors whose encoded byte count overflows `usize` and model
  metadata with zero attention heads.
- Stop malformed pipe-format tool arrays from entering a non-progress allocation
  loop.
- Treat model JSON as a tool call only when the request offers a non-empty tool
  list.
- Publish graceful-shutdown state and its signal number atomically so
  interrupted CLI commands retain the correct exit status.
- Drop completed CPU spin-pool results when a sibling task panics.
- The CPU backend's dequantized-weight and Q4_K/Q6_K repack caches now key on a
  never-reused buffer id instead of a memory address. A `CpuBackend` that
  outlives a model — `infr serve` reloading one — could otherwise return a
  cached weight built from the PREVIOUS model, because both the allocator and a
  fresh mmap hand out addresses that were just freed.
