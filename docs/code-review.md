# Code review

**Date:** 2026-08-03

**Scope:** clean `main` working tree, so review covered the entire repository.

**Depth:** low; only high-confidence correctness findings are reported.

## Findings

### Medium — Remote shard metadata can exhaust process memory

`crates/infr-hub/src/store.rs:268-289`

`parse_shard` accepts any `u32` shard count, and `shard_set` eagerly formats and
collects every filename through that count before any download starts.
`pull_repo_latest` passes a filename selected from remote repository metadata
into this expansion at `crates/infr-hub/src/pull.rs:54-72`.

Repro: remote metadata selects `m-Q4_K_M-0000000001-of-4294967295.gguf` Expect:
reject the unreasonable shard count before allocation Actual: collect
`1..=4294967295` formatted filenames, exhausting memory

### Medium — Diffusion generation can exceed `max_new`

`crates/infr-llama/src/diffusion.rs:560-615`

`diffusion_generate` forces at least one canvas block and appends the complete
trimmed canvas without capping it to the remaining token budget.
`DiffusionGemmaChat::generate_impl` passes `max_new` directly as `n_predict` at
`crates/infr-llama/src/chat/diffusion.rs:303-315`.

Repro: full `canvas_len`-token denoise result with `max_new = 1` and no EOS or
repetition Expect: one generated token Actual: the complete canvas is emitted
and counted

### Medium — Diffusion requests ignore cancellation and per-request sampling

`crates/infr-llama/src/chat/diffusion.rs:198-225`

Both diffusion `ChatModel` entry points discard `RequestCtx`; generation
therefore never polls its abort latch and resolves its seed only from process
configuration at `crates/infr-llama/src/chat/diffusion.rs:262-266`. This
violates the `ChatModel` request-state contract documented at
`crates/infr-llama/src/chat/mod.rs:58-70`.

Repro: start multi-block diffusion generation, then call `RequestCtx::abort()`
after the first block Expect: no later prefill or denoise block Actual: every
budgeted block continues running

Repro: run the same prompt with request seeds `7` and `8` under one process seed
Expect: each request uses its own seed Actual: both requests use the process
seed

### Medium — Forced tool choices become unconstrained without `tools`

`crates/infr-llama/src/grammar.rs:213-250`

`tool_constraint_for` returns `Ok(None)` before inspecting `tool_choice` when
`tools` is absent. The server also accepts every string as a tool choice at
`crates/infr-server/src/lib.rs:395-412`; `run_chat` then takes the ordinary
unconstrained path when no constraint is returned at
`crates/infr-cli/src/main.rs:1889-1951`.

Repro: `POST /v1/chat/completions` with `"tool_choice":"required"` and no
`tools` Expect: HTTP 400 `invalid_request_error` for `tool_choice` Actual:
unconstrained assistant-text generation proceeds

Repro: `POST /v1/chat/completions` with `"tool_choice":"bogus"` and no `tools`
Expect: HTTP 400 `invalid_request_error` for `tool_choice` Actual: the arbitrary
string is accepted and unconstrained generation proceeds

### Medium — Streaming generator panics look like successful termination

`crates/infr-server/src/lib.rs:1521-1641`

The streaming path discards the `spawn_blocking` join handle. A panic skips the
`match res` failure branch; unwinding drops `DoneGuard`, which emits `[DONE]`,
but no error frame, failure statistic, or failure log is produced.

Repro: a `ChatGenerator::chat` implementation panics after the opening role
chunk Expect: terminal `server_error` SSE frame, failed-request accounting, then
`[DONE]` Actual: role chunk followed only by `[DONE]`; failure counters remain
unchanged

### Low — Generated-token reconciliation loses corrections across drains

`crates/infr-server/src/lib.rs:586-627`

Live delta estimates and completion corrections share one globally drained
signed counter. A negative correction after an earlier drain cannot retract that
earlier overcount; if another request has added tokens, the correction can
instead subtract from that request's interval.

Repro: emit two deltas, drain stats, then complete with `gen_tokens = 1` and
drain again Expect: reported windows total one generated token Actual: first
window reports two and the later `-1` window is clamped to zero, so the total
remains two

### Low — Spin-pool panic attribution races with the next job

`crates/infr-cpu/src/pool.rs:248-265`

`SpinPool::run` releases `in_run` before consuming the pool-global `panicked`
flag. A new owned job can start and set that flag in the gap, allowing the prior
job to consume the new job's panic.

Repro: job A pauses after `in_run.store(false)`; job B starts and one task
panics; job A resumes and swaps `panicked` Expect: A returns successfully and B
re-panics Actual: A re-panics and B can return successfully with incomplete task
state

### Low — `matmul_f32` leaks Vulkan handles on error paths

`crates/infr-vulkan/src/matmul.rs:97-184`

Transient Vulkan handles are manually destroyed only on the success tail at
`crates/infr-vulkan/src/matmul.rs:322-331`. Every `?` after a handle is created
leaks preceding handles until device destruction; the explicit null-pipeline
return also leaks the shader module and layouts.

Repro: device creates the shader module, then `create_descriptor_set_layout`
returns an error Expect: shader module is destroyed before `matmul_f32` returns
`Err` Actual: shader module remains live until device destruction

## Cleared

- CLI backend strings such as `vulkanfoo` do not silently select Vulkan device
  zero; downstream numeric parsing rejects the suffix.
- GGUF loading validates metadata depth, tensor dimension arithmetic, alignment,
  duplicate names, quantization block divisibility, and mapped-file bounds
  before exposing tensor bytes.
- `StopMatcher` preserves split stop prefixes and UTF-8 boundaries.
- Vulkan upload, download, and copy paths validate buffer extents.
- Vulkan external-memory file descriptors transfer ownership on successful
  import and close the duplicate on failure.
- Metal derived-buffer cache keys include monotonic allocation identity,
  preventing recycled-address cache hits.
- Autoregressive generation explicitly handles `max_new == 0`; the budget defect
  is confined to block diffusion.
- `SpinPool` waits for every worker check-in before releasing an ordinary
  borrowed job; the surviving defect is panic attribution between jobs.
- Pager production callers reject zero slots before construction.
- Dense-runner token IDs are validated before embedding lookup on reviewed
  generation paths.
- Malformed Hermes tool markup is gated from production streaming and removed
  rather than exposed as assistant content.

## Hardening

These are not established current defects.

- `with_profiling_suppressed` restores a process-global boolean only after
  normal return; an RAII nesting counter would survive panics and overlapping
  scopes.
- `SpinPool::collect` leaks already initialized values when another task panics;
  unwind cleanup could track and drop initialized slots.
- Existing HuggingFace `blobs/<expected_sha>` files are trusted by pathname and
  existence without rehashing; optional verification would detect local cache
  corruption.
- Public `dequant_block` assumes block-sized input, while reviewed GGUF callers
  supply validated slices; explicit length checks would protect direct callers.
- The streaming SSE channel is unbounded. `max_tokens` bounds total retention,
  but a non-reading client can retain a large completion.

## Coverage

Reviewed the full clean-tree scope at low depth, with detailed tracing of:

- Core graph, pager, sampling, pool, and backend boundary paths.
- Llama autoregressive, diffusion, grammar, chat, and request-context control
  flow.
- Server validation, admission, deadlines, streaming/non-streaming responses,
  statistics, and cancellation plumbing.
- Hub model selection, shard expansion, cache resolution, downloads, integrity,
  symlinks, and path validation.
- GGUF metadata/tensor validation and host dequantization entry paths.
- CLI backend/model parsing and server generation adapters.
- Chat template, reasoning, stop, and tool-call parsing paths.
- Profiling aggregation and JSON output.
- Selected Vulkan and Metal allocation, transfer, cache, synchronization,
  dispatch, and lifecycle paths.
- Testkit synthetic weights and parity orchestration.

GAP — not deeply reviewed line-by-line:

- SIMD/scalar numerical bodies and architecture-specific branches in
  `crates/infr-cpu/src/kernels.rs`.
- Every graph-rewrite combination in `crates/infr-core/src/fusion.rs`.
- Static quantization tables in `crates/infr-core/src/iquant_grids.rs`.
- Every MTP and per-architecture seam graph formula.
- Most Vulkan recorder, adapter, GEMM, pager, tensor-parallel, expert-routing,
  and shader host/ABI combinations.
- Most of `crates/infr-metal/src/exec.rs` and full Metal shader/host ABI
  validation.
- Every quantization arm in `crates/infr-gguf/src/dequant.rs`.
- Large CLI benchmark and diffusion-specific command paths.
- Every test assertion and example.
- Live Vulkan or Metal execution.
- Platform-gated macOS code was read but not compiled.

No build or tests were run; this was a read-only code review apart from
replacing this report.
