# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **DeepSeek V4 attention primitives** (stage 4, op level — nothing emits them
  yet): `Op::QkNorm`'s `weight` became optional, so a weightless per-head
  RMSNorm is expressible without a fake ones-vector operand; `Op::Attention`
  gained an optional per-head `sinks` input (one extra logit per head joining
  the softmax max and denominator, contributing no value); and `Op::Rope` gained
  a `backward` flag for de-roping (llama.cpp's `ggml_rope_ext_back` — the
  forward rotation with `sin` negated). All three run on CPU, Vulkan and Metal;
  `sinks: None` / `backward: false` / `weight: Some(..)` reproduce the previous
  code path exactly, so no existing model's numerics move. V4's grouped low-rank
  output projection needs no new op — it composes from `Op::CopyStrided` and
  `Op::Linear`'s existing `w_off`.
- **DeepSeek V2 architecture support** (stage 2): registered `deepseek2` arch
  string, parsed MLA hyperparameters (`q_lora_rank`, `kv_lora_rank`,
  `qk_rope_dim`, `head_k_mla`, `v_head_dim`, lite detection via tensor
  presence), configurable MoE gating (`expert_gating_func` → softmax / sigmoid /
  sqrt-softplus), `expert_weights_norm`, group-limited routing fields, and
  `rope_yarn_log_mul` (with the convert-script ÷0.1 fix). Added
  `MoeGating::SqrtSoftplus` variant and wired it in CPU + Vulkan backends. See
  `docs/deepseek.md` § Stage 2.

- **MLA attention kernels** (DeepSeek V2/V3 absorbed form, `Op::Mla`): Vulkan
  `mla.comp` and Metal `mla_f16kv` compute kernels implement the full per-head
  pipeline — `wk_b` absorption of q_nope, internal q_pe RoPE (NORM interleaved),
  two-pass SDPA over the unified f16 KV cache (one row per token, V aliased from
  the first `kv_lora_rank` columns of K), and the `wv_b` output projection.
  Ring-buffer, causal / sliding-window / canvas masks supported. CPU math
  covered by `mla_parity` in `seam_op_parity.rs`; Metal dispatch covered by
  `mla_parity`/`mla_ff_parity` in the Metal parity suite, executed on the macOS
  CI job (a real Metal device).

- **DeepSeek V3 MoE routing** (Vulkan): `moe_topk` now selects on
  `probs + exp_probs_b` while weighting from the unbiased probs (the noaux_tc
  router bias), supports sqrt-softplus gating (`gating=2`), and enforces
  group-limited routing (per-group top-2, top `n_expert_groups_used` groups,
  mask the rest). The `blk.%d.exp_probs_b.bias` tensor loads from V3 GGUFs and
  threads into `Op::MoeFfn`.

- **DeepSeek V2-Lite tests**: `cpu_deepseek2_config`,
  `cpu_deepseek2_prefill_finite`, `cpu_deepseek2_prefill_paris` (CPU oracle +
  finiteness over the vocab) and `gpu_seam_matches_cpu_deepseek2` (Vulkan vs
  CPU, `#[ignore]`d behind a GPU) — gated behind a V2-Lite GGUF in the HF cache.

- **DeepSeek V1 support** (`deepseek` architecture): plain MHA attention +
  softmax-gated MoE with ungated shared expert, following llama.cpp's
  `src/models/deepseek.cpp`. First `n_layer_dense_lead` layers are dense FFN,
  the rest are MoE. Tokenizer pre-processing added for `deepseek-llm`,
  `deepseek-coder`, and `deepseek-v3` pre-types (see `docs/deepseek.md` § Stage
  1). Works on CPU + Vulkan backend via the existing `FfnW::Moe` with `shexp`
  path (same as llama4's plain-summed shared expert).

- `infr run`, `infr bench` and `infr serve` now notice the model file being
  overwritten underneath the live weight mapping and fail with a named error
  instead of serving output from weights that no longer match the file. `run`
  checks at both ends of every turn, `bench` before reporting any numbers, and
  `serve` at the start of each request. New `infr_gguf::watch::WeightWatch`,
  re-exported as `infr_llama::WeightWatch`.

- **A model that does not fit now streams from disk on its own.** Weights that
  fit stay exactly as they were — resident on the GPU, or zero-copy mmap on the
  CPU backend, with no arena and no copies. Only when they do not fit does the
  engine page them `DISK → DRAM → VRAM` (`DISK → DRAM` on the CPU backend),
  sizing the DRAM arena from the host memory actually available rather than
  requiring a budget nobody could guess. Measured on Qwen3-14B Q8_0 under an 8
  GB cap, the automatic budget lands on the same 7.4 GB that measured **2.17x
  faster decode than the mmap path** it replaces.
  - The probe honours **cgroup memory limits** (v2 and v1, tightest ancestor),
    not just `/proc/meminfo` — inside an 8 GiB container the host file still
    reports the whole machine, and sizing an anonymous arena from that is an OOM
    kill. Linux only; other platforms report "unknown" and keep the mmap path
    unless a budget is set by hand.
  - **Unified-memory devices (iGPU, APU) stream `DISK → GPU-accessible RAM` with
    no host cache between.** Their streaming arena is already host RAM, so a
    cache beneath it would hold a second copy the GPU cannot read in place;
    instead its misses are served by block-granular positioned reads rather than
    through the GGUF mapping, whose page cache thrashes on a forward pass's
    cyclic sweep. That is what lets a model far larger than the machine run on
    those parts at all. **Untested on unified hardware** — none was available —
    but the mechanism is covered on a discrete GPU by `INFR_DRAM_BYPASS`
    (below). Metal has no pager at all yet and is unaffected.
- `INFR_DRAM_CACHE` / `paging.dram`: the host weight cache's budget. **Unset now
  means "size it automatically"**; a value pins the arena and wins over every
  automatic decision (including on a machine where the model would have fit,
  which is how the streaming path gets exercised at all); and **`0` turns it off
  entirely**, which is what an A/B against the mmap path needs. A budget too
  small to seat a weight class leaves that class mapped and says so.
  - **CPU backend**: every weight above 1 MiB. Measured on a memory-capped
    Llama-3.2-1B F16: decode 2.06x faster at a 1.5 GB cap with 210x fewer major
    faults, prefill 3-7.5% slower (`docs/perf/results.md`).
  - **Vulkan backend**: a third tier under both dense weight streaming and the
    paged MoE expert cache, so a VRAM miss resolves against the arena and
    reaches the file only when that misses too. MoE pages ONE EXPERT at a time
    rather than a whole bank. A block the arena has no room for is read straight
    into the staging ring instead of evicting one, so the streaming majority
    costs one copy rather than two. Measured on a memory-capped Qwen3-14B Q8_0
    under a forced 2 GB VRAM budget: **decode 2.17x faster than the mmap path it
    replaces** at an 8 GB cap with a 7 GB arena (1.41x with a 3 GB one), 38x
    fewer major faults, 232 → 110 GB read, and parity when memory is plentiful
    (`docs/perf/results.md`). The arena budget is the dominant factor — 3 GB → 7
    GB is worth 1.6x on its own — which is why it is now sized automatically
    rather than left to a guess. The measurement covers one GPU, one drive and
    Linux only.
  - `INFR_PAGER_STATS=1` reports hit rate, reads and bytes for each tier.
  - The host arena admits a block on its SECOND miss, not its first. A tier
    above only calls down on its own misses, so first-miss admission filled the
    arena with the prefix the VRAM pager was about to keep resident forever —
    blocks that then never call down again. Measured on Qwen3-14B: 4 of 9 slots
    per pool were dead, and the rule turns useful hits per pass from 5 into 9
    while cutting bytes read ~9% at the same budget.
  - One paged block is read with several concurrent positioned reads rather than
    one, which is what puts the tier ahead of the mapping it replaces: a drive
    delivers its bandwidth on queue depth (measured 1.2-1.5 GB/s for a single
    read against a 2.2 GB/s device ceiling), while the page cache gets its
    readahead issued in parallel by the kernel for free. Reads stay correct on
    every platform, but the speedup is measured on Linux/NVMe only — a Windows
    handle not opened for overlapped I/O serializes them.
- `INFR_DRAM_BYPASS` / `paging.dram_bypass`: read paged blocks straight from
  disk into GPU memory with no host cache — the shape a unified-memory device
  takes automatically. It exists as a flag so that behaviour can be exercised on
  a discrete GPU, which is the only hardware it can be tested on here, and is
  also the honest choice on a machine whose RAM is better spent elsewhere. No
  effect on the CPU backend, where that arena is the only tier there is.
- `INFR_LAYER_MAJOR` / `paging.layer_major`: force the prefill loop order. `1` =
  layer-major, `0` = chunk-major, unset = layer-major exactly when the weights
  stream (see the Changed entry below). Both overrides are for A/B; forcing it
  on is the only way to put a resident model on the layer-major path.

### Changed

- **A streamed model now prefills LAYER-MAJOR: the prompt sweeps the weight set
  once instead of once per prefill chunk.** Prefill runs in `device.ubatch`
  chunks and every chunk used to run the whole model, so a P-token prompt paid
  `ceil(P/ubatch)` complete weight sweeps — invisible when the weights are
  resident and the entire bill when they stream. The chunk loop now runs INSIDE
  the layer loop, which reads each weight once per prompt at the same
  chunk-sized dispatches. (Raising `device.ubatch` to the prompt length reaches
  the same I/O and is not a substitute: it bakes a single multi-second submit
  that trips the GPU hang watchdog.) Measured on a memory-capped Qwen3-14B Q8_0
  (`MemoryMax=8G`, `paging.cache=2g`, `paging.dram=6g`, P=4096, RX 7900 XTX) at
  the 1024-row default chunk: **prefill reads 25.27 → 6.31 GB from disk (4.00x)
  and runs 341.9 → 779.9 tok/s (2.28x)**, with the read volume now identical to
  a single-chunk prefill's. The cost is holding every chunk's residual stream at
  once (`ctx * n_embd` f32), which the streaming budget reserves for. Resident
  models keep the chunk-major order, where the reorder would buy nothing and
  only add activation residency, and so does gemma4-E2B on any backend — its
  per-layer token embeddings are built by the graph prologue, which a span
  starting past layer 0 cannot see.

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

- **DeepSeek `deepseek-llm` pre-tokenizer split on the wrong character
  classes**: `DEEPSEEK_LLM_PRE_RES[2]` opened its quote range with `'` (U+0027)
  instead of `‘` (U+2018), so the class covered U+0027–U+201F — most of the BMP,
  including the ASCII digits, Hebrew, Arabic and Devanagari — rather than the
  eight quote characters; and `DEEPSEEK_LLM_PRE_RES[1]` had `ℹ-ℿ` where
  llama.cpp has `ℹℼ-ℿ`, adding U+213A and U+213B to the letter class. Both moved
  chunk boundaries silently, with no error: on DeepSeek-V2-Lite-Chat the first
  one changed real token ids wherever a non-ASCII space (U+00A0, U+0085)
  preceded punctuation, e.g. `"a \u{00A0}. b"` tokenized as
  `[64, 30683, 13, 270]` where llama.cpp gives `[64, 207, 1202, 13, 270]`. All
  fourteen DeepSeek regexes now match `llama-vocab.cpp` codepoint for codepoint,
  `infr` agrees with `llama-tokenize` on every text tried, and
  `deepseek_pre_split_boundaries_match_the_reference_lists` pins the chunk
  boundaries of all three lists without needing a model file.
- **DeepSeek V3 router bias applied to the wrong scores (Vulkan)**: `moe_topk`
  added `exp_probs_b` to the raw router LOGITS, but llama.cpp (and the CPU
  interpreter) add it to the gated PROBS
  (`selection_probs = ggml_add(probs, exp_probs_b)`). `probs + bias` is not
  monotone in `logits + bias`, so a real V3/V3.1 model (sigmoid gated, noaux_tc
  bias) routed to different experts than the reference. The shader now selects
  on `gating(logit) + bias`; the no-bias paths are unchanged. Caught by the new
  `moe_groups_bias_parity` op test, whose data makes the two semantics disagree.
- **Metal `mla_f16kv_ff` (YaRN twin) bound its buffers at the wrong indices**:
  the kernel declared `MlaParams` at buffer(5) and `freq_factors` at buffer(6),
  but `exec.rs` binds `[q, k, wk_b, wv_b, dst, ff]` with params at
  `bufs.len() = 6` — the shader would have read the 4-byte divisor buffer as its
  params struct. Never executed until the new `mla_parity`/`mla_ff_parity` tests
  ran it on the macOS CI job; buffer declarations swapped to match the binding
  convention.

- **DeepSeek V2/V3 decode writes every token's K to row 0**: the seam's
  `dyn_replay` gate (which must mirror the Vulkan adapter's record-once-replay
  eligibility) had no `Op::Mla` exclusion, so a DeepSeek MLA model built the
  replay tape with `pos = 0` baked into every `WriteKv`. The adapter rejects the
  MLA graph as ineligible and falls back to the static path — executing that
  baked `pos = 0` tape for every decode token, so each token's K row landed in
  row 0 and rows 1.. never got populated. Attention then ran against a one-row
  cache: logits diverged wholesale (GPU top-1 was unrelated to the CPU's, cosine
  ~0.5). Added the `c.deepseek2` exclusion to the gate so MLA models take the
  per-token static decode path.
- **GPU MoE routing nondeterminism**: `Recorder::moe_topk` bound
  `[logits, ids, wts, bias]` with `n_out = 2`, so the hazard tracker treated
  `ids` as a read and `bias` as a write — but the shader writes `ids` and only
  reads `bias`. The `ids` write never reached the RAW check, so the per-expert
  GEMVs that index the weight bank by `ids[slot]` got no barrier before reading
  it, and the pooled ids buffer (reused across layers) fed stale/torn expert ids
  into the routing — DeepSeek V2/V3 GPU logits changed run-to-run. Reordered the
  dispatch to `[logits, bias, ids, wts]` (and the shader bindings to match) so
  the two written buffers sit in the trailing write slots.
- **DeepSeek MLA absorption transposition**: the CPU/Vulkan/Metal MLA kernels
  read the per-head `attn_k_b` weight transposed, computing `W @ q_nope` where
  the absorbed-form math needs `Wᵀ @ q_nope` — the file stores it as
  `[qk_nope_dim, kv_lora_rank]` per head with the qk_nope dim fastest. The
  regression entered with the `6adb2de` "transpose fix" (which applied its own
  diagnosis backwards: the diff flipped `i` from fast to slow) and was then
  pinned by parity-test references written to the same wrong layout. The `wv_b`
  output projection had the identical transpose (`[kv_lora, v_head_dim]` read
  with the v_head_dim dim fastest) — invisible to every test because the parity
  synthetic `wv_b` is an identity. Both are now read file-order (the fast dim
  contracted) in all three backends and both parity references; the A/B check on
  `wv_b` shows the transposed orientation collapses the pearson correlation with
  llama.cpp from 0.79 to 0.10 and turns generation into `"ARSARSARS…"` garbage.
  The `mla_matches_cpu_reference` parity test dispatches two attended keys with
  random K rows so the absorbed-query scores actually shape the output (at
  `kv_len=1` softmax is trivial and the old test could not detect a
  transposition).
- **DeepSeek V2/V3 YaRN frequency ramp missing**: the GGUF declares
  `rope.scaling.type = yarn`, `factor = 40`, `original_context_length = 4096`,
  `yarn_log_multiplier = 0.0707` — which makes llama.cpp set
  `yarn_ext_factor = 1.0` and run the full per-dimension frequency ramp at EVERY
  context length. infr roped with plain `θ^(−2p/d)`, so its q_pe/k_pe
  frequencies were up to 40× off: the model output was coherent-looking but
  wrong (`"Reply Collabor…"` garbage, top-1 unrelated to llama.cpp's). The seam
  now computes the corr_dims spectral ramp divisors and feeds them through
  `Op::Rope.freq_factors` (k_pe) and the MLA kernels' internal q_pe rope (Vulkan
  `mla_ff`/`rope_ff` builds, Metal twin); the YaRN mscale² is a constant folded
  into `mla_scale = mscale²/√(qk_nope + qk_rope)` (√192 for V2-Lite, not √576 —
  `qk_nope = head_k_mla = 128`), replacing the previous context-length-gated
  approximation. The rope vector mscale cancels to `rope_attn_factor` for
  deepseek2, so the kernels need no vector scaling. Greedy output now matches
  llama.cpp's continuation " Paris." in content.
- **Vulkan hazard tracker mis-scoped the YaRN freq_factors dispatch**: the
  `mla_ff`/`rope_ff` dispatches bound the read-only ff divisors AFTER the output
  buffer with `n_out = 1`, so the tracker marked ff as the write and left the
  real output (dst / y) untracked — no store→read barrier before the
  wo-projection consumed the attention output, making DeepSeek V2 GPU logits
  nondeterministic run-to-run. Reordered both dispatches to `[.., ff, dst]` /
  `[x, ff, y]` and the shader bindings to match, so the trailing write slot is
  the actual output.
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
