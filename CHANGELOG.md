# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.4.0] - 2026-08-29

### Highlights

- **Automatic memory planning now follows the selected device and the live allocator ceiling.**
  Fixed weights, persistent KV/QSA/recurrent state, peak runtime workspace, weight-packing
  alignment, Windows driver headroom, and the elastic expert arena are priced in one plan. If a
  Windows driver exposes slightly less allocatable VRAM than its initial budget snapshot, automatic
  mode retries with a smaller expert arena while preserving fixed state and the minimum runnable
  expert working set.
- **Manual RAM budgets now mean total infr process memory.** The host expert arena subtracts the
  process's existing resident footprint instead of treating `device.ram_budget` as extra cache.
  Automatic mode continues to reserve system headroom, reducing the risk of Windows paging.
- **Qwen3.8-Flash-Next decode and prefill received a focused Vulkan performance pass.** QSA scoring
  and exact top-k selection use wider, parallel passes; Q8 attention reuses K/V data across GQA
  heads; narrow Q8 prefill projections and Q2 expert prefill take faster paths.

### Changed

- MoE prefill lane counts are bounded by the runtime memory plan instead of assuming every device
  can host the same concurrent working set.
- Windows release artifacts are produced from version tags only.
- Benchmark documentation now uses stable historical runs that did not enter system paging; the
  resource-pressure diagnostic sweep is not presented as product performance.

### Fixed

- Automatic Qwen3.5/Qwen3.6/Qwen3.8 placement no longer plans against a generic 24 GiB GPU shape or
  hands the Vulkan allocator bytes it has already reserved for model state and runtime work.
- DeepSeek V4's de-rope regression test now accounts for the production FP8 E4M3 KV rounding error
  while retaining wide separation from a missing de-rope implementation.

### Performance

- Removed full index radix scans from QSA top-k and parallelized deep-context QSA selection.
- Added wave32 MoE top-k reduction on supported AMD Vulkan devices.
- Parallelized deep Q8 attention combine and reused Q8 K/V blocks across GQA head pairs and groups.
- Tiled narrow Q8 prefill projections and accelerated Q2 paged-expert prefill.

## [0.3.0] - 2026-08-27

### Added

- The Windows launch wizard now shows the MoE4All version, project identity,
  maintainer, and repository, and checks GitHub Releases for updates without
  blocking offline startup.
- A GGUF can be dropped directly onto `Start-INFR-Wizard.cmd` or passed as its
  first argument. Existing history selection and in-terminal path pasting remain
  available.

### Changed

- Windows archives and workflow artifacts are named
  `MoE4All-Windows-x86_64-vVERSION.zip`; the engine remains `infr.exe` for CLI
  and script compatibility.
- GitHub Release titles use `MoE4All vVERSION`, and package smoke tests now cover
  the bilingual documentation, version banner, all three wizard modes, and the
  direct model-path argument.
- Links from packaged documentation to repository-only technical material use
  GitHub URLs, so they remain valid outside a source checkout.

## [0.2.0] - 2026-08-27

### Highlights

- **Qwen3.8-Flash-Next (`qwen4exp`) text inference is fully wired on Vulkan.**
  The implementation covers the released model's four-stream hyper-connections,
  gated DeltaNet/full-attention layer mix, SSD-backed PLE lookup, paged MoE
  experts, tokenizer/config plumbing, and model-specific thinking/non-thinking
  sampling defaults. The released UD-Q2_K_XL GGUF has generated coherent chat
  and passed short-context decode/prefill smoke tests on an RX 7900 XTX.
- **QSA sparse attention and its index-key cache are implemented end to end.**
  Full-attention layers cache raw index keys, pool and score complete blocks,
  select the exact top blocks, preserve the incomplete causal tail, gather the
  corresponding F16 K/V rows, and run the existing attention kernel over the
  chronological sparse view. CPU reference code and Vulkan parity coverage
  verify index selection and byte-exact K/V gathering. Real-cache depth 2052
  and synthetic depth 4096 runs both crossed the old approximately 2K boundary
  without a device loss.
- **Large-model host caching keeps adaptive RAM headroom.** Automatic host-cache
  sizing now leaves enough memory for runtime allocations and avoids driving
  Windows into system paging while a bounded SSD/RAM expert tier is active.
- **The Windows release workflow is more complete.** The bilingual wizard now
  accepts a pasted model path directly from its history prompt, and portable
  archives include both Chinese and English README files.

### Changed

- **`infr serve --parallel` (`-n`, `--np`) defaults to 1 slot, was 4.** Each
  slot owns a full KV cache and, with `--ctx` unset, the per-slot window is the
  VRAM-fit context divided by N — so the old default handed every user a quarter
  of the context window whether or not they ever issued a concurrent request.
  Concurrency is now opt-in and a single request gets the whole window; pass
  `--parallel N` for the previous behaviour. `infr multi` already defaulted to
  1, so the two now agree. The engine is unchanged: the concurrent seam runs at
  one slot exactly as before.

### Added

- **Embedding serving through a managed llama.cpp worker.** `infr serve-embedding` exposes an
  authenticated OpenAI-compatible `/v1/embeddings`, and `infr serve --embedding-model` can host it
  beside chat on the same public endpoint. INFR owns routing, admission, process cleanup, and
  resource accounting while reusing llama.cpp's mature GGUF/CPU/Vulkan implementation. The browser
  GUI can create, estimate, start, stop, and switch Embedding profiles.

- **DeepSeek V4's hash-routed MoE layers run, on CPU and Vulkan.** Such a layer
  takes its experts from an i32 `blk.N.ffn_gate_tid2eid`
  `[n_expert_used, n_vocab]` table indexed by TOKEN ID rather than from the
  router's top-k, and nothing in the tree could gather that selection, so `infr`
  refused the layer by name rather than silently running a different set of
  experts. A new `Op::GatherI32` (`ggml_get_rows` on an integer table — CPU
  interpreter arm and the `gather_i32.comp` Vulkan kernel) produces the
  `[batch, n_expert_used]` selection that `Op::MoeFfn::expert_ids` already
  consumed, and the seam emits it per hash-routed layer. `Op::EmbedGather` could
  not serve this: its kernel walks the table in whole 32-element sub-blocks, so
  a 6- or 8-wide row gathers nothing. The graph's token-id Input is now declared
  whenever a hash-routed layer needs it, not only when the token EMBEDDING is
  gathered on-device — a model whose `token_embd` dtype has no gather kernel is
  still hash-routed. A `ffn_gate_tid2eid` that is not
  `expert_used_count * vocab` entries is refused, because the gather indexes it
  by token id with no clamp. Metal refuses `Op::GatherI32` by name (its device
  MoE path already refuses V4's mandatory sqrt-softplus gating). This does NOT
  make the shipped `DeepSeek-V4-Flash-0731-GGUF` generate: every layer of that
  file past 1 is compressed (ratio 4/128), which is still unimplemented, and it
  refuses there.

- **GGUFs carrying `GGML_TYPE_I32` tensors open.** Integer tensors are lookup
  tables a kernel indexes with rather than weights, and `Gguf::open` refused
  them outright (`unsupported: ggml type 26`). The first shipped model to need
  one is DeepSeek-V4, whose hash-routed layers each carry an i32
  `blk.N.ffn_gate_tid2eid.weight` token-id → expert-id table: the reference
  conversion `ggml-org/DeepSeek-V4-Flash-0731-GGUF` would not open at all,
  failing before a single tensor shape was read. It now loads end to end — the
  config parses, every one of its 1328 tensors resolves, and the model reaches
  the existing designed refusal naming its compressed (ratio 4/128) layers,
  which are still unimplemented.

- **`infr pull` splits ONE file across connections too.** Shard-level
  concurrency did nothing for a model that ships as a single file, which most
  do: `unsloth/DeepSeek-V3.2-GGUF`'s `UD-TQ1_0` is one 161 GB GGUF and still
  went over one connection. It is now fetched as a grid of 64 MiB byte ranges,
  several at a time, and reassembled — measured on that object, in the same
  60-second window: 11.4 MB/s before, 80.5 MB/s after, and the whole 161 GB pull
  ran at 80.0 MB/s end to end (33.6 minutes against a projected 3.9 hours).
  `hub.pull_jobs` (default `8`) is now the TOTAL connection bound rather than a
  file count, so shards and ranges share one allowance and a 236-shard repo
  cannot turn into 64 sockets; `0` and `1` still mean strictly one connection,
  never split. An interrupted ranged download resumes from a per-range sidecar
  next to the partial, and a partial whose object was re-published upstream is
  discarded rather than continued — a resumed splice of two uploads is a
  plausible-sized corrupt file, and it is bound to HF's LFS sha256 rather than
  to an `ETag` so a different CDN edge cannot look like a different file.
  Servers that do not serve ranges (no `Accept-Ranges`, or a `200` where a `206`
  was asked for) fall back to the single stream, files of 64 MiB or less are
  never split, and the end-of-download sha256 gate is unchanged: one bar per
  file, one digest, one decision.

- **`infr pull` fetches a split model's shards concurrently.** A single
  connection to the HF CDN is what caps a download, not the link — measured
  against the same objects, one connection sustained 8.8 MB/s and five sustained
  78.7 MB/s — so the shards of a `-NNNNN-of-MMMMM` set are now fetched several
  at a time. A 245 GB five-shard `unsloth/DeepSeek-V3.2-GGUF:Q2_K` pull ran at
  80 MB/s end to end. Bounded by the new `hub.pull_jobs` (`INFR_PULL_JOBS`,
  `--set hub.pull_jobs=N`, default `8`), because shard counts come from the repo
  rather than from the user (`DeepSeek-V3.2-REAP` ships 236); `0` and `1` both
  keep the old strictly sequential behaviour. Resume, the LFS-sha256 gate and
  the refusal to link a mismatched blob are unchanged and per-file. Concurrent
  downloads each get their own progress line (`infr_core::progress::group`)
  instead of overwriting one another, and a failed file stops the others from
  claiming more work instead of pulling the rest of a model that cannot load.

- **Multi-shard (`gguf-split`) models load.** `Gguf::open` on any member of a
  `-NNNNN-of-MMMMM.gguf` set follows `split.count` to the whole set, maps every
  shard separately (each keeping the shared file mapping that makes a
  larger-than-RAM model runnable), takes the metadata from shard 1 and unions
  the tensor index across all of them — so a tensor resolves against the file
  that actually holds it instead of failing with `tensor not found`. Opening a
  LATER shard works too: it carries no model metadata, so the set is re-rooted
  at shard 1. The set is validated rather than trusted — the filename's
  `-of-MMMMM` must equal `split.count`, every shard's `split.no` must match the
  position its filename claims, all shards must agree on `split.count`, the
  assembled index must be exactly `split.tensors.count` entries with no repeated
  name, and a missing shard fails naming the file. Tested against a 5-shard 229
  GB DeepSeek-V3.2-Q2_K.
- `infr_core::gguf_split` — the `-NNNNN-of-MMMMM.gguf` naming convention, parsed
  in one place instead of two. `infr-hub` (is the cached set complete?) and
  `infr-gguf` (which files is this model?) now share it.
- `infr_core::blockio::FileBlockIo::open_shards` — the weight pager's disk tier
  reads a shard set as one address space, refusing at open any shard whose
  length is no longer the one the weights were loaded against (which would shift
  every offset in every later shard).

- **A DeepSeek V4 model whose `compress_ratios` are all zero now GENERATES**
  (stage 4, slice A). The `ratio == 0` tier — hyper-connections wrapping both
  sublayers, single-head MQA attention with a weightless per-head Q norm, a
  `[nope | rope]` tail rope, attention sinks, a de-roped grouped low-rank output
  projection, and sqrt-softplus MoE with the per-layer SwiGLU clamps — is
  emitted end to end and runs on CPU and Vulkan. Compressed layers (ratio 4
  or 128) are refused by name, the message saying which layer and what is
  missing, rather than approximated by the ratio-0 graph. Two supporting changes
  ship with it: Vulkan's `Op::Linear` now accepts `w_off` on an **F32** weight
  (by shifting the weight's `bufferDeviceAddress` base — the float GEMVs already
  address their weight by pointer, and `w_off` is row-aligned), which is what
  lets the grouped output projection run there at all; and V4 is excluded from
  the record-once decode replay tape, like `deepseek2`, because its ops have no
  dynamic twins. See `docs/deepseek.md` § Stage 4.
- **DeepSeek V4 hash-routed MoE and per-layer SwiGLU clamping** (stage 4, op
  level). `Op::MoeFfn` gains `expert_ids`: an optional pre-gathered
  `[rows, n_expert_used]` I32 handle (llama.cpp's `selected_experts_in`,
  gathered from `ffn_gate_tid2eid` by token id) that replaces the top-k
  selection on V4's first `hash_layer_count` layers. The router matmul and
  gating still run — the routing WEIGHTS remain the router's own probabilities
  at the hash-chosen experts, renormalised and scaled — while `argsort_top_k`,
  `exp_probs_b` and group-limited routing are skipped; the last two are refused
  alongside `expert_ids` rather than silently ignored. `Op::GatedAct`,
  `Op::GatedActFused` and `Op::MoeFfn` also gain `swiglu_clamp: Option<f32>`,
  built through the new `infr_core::graph::swiglu_clamp(limit)` which carries
  llama.cpp's `limit > 1e-6` disabled gate so no caller can pass a layer's `0.0`
  through as a real clamp. V4 clamps `up` symmetrically and the gate one-sided
  and **pre**-activation, where every other arch clamps post-SiLU. All of it
  runs on CPU, Vulkan and Metal; `None` leaves every existing model's numerics
  bit-identical. See `docs/deepseek.md` § Stage 4.
- **DeepSeek V4 Sinkhorn hyper-connections** (stage 4, op level): three new ops
  that replace `x = x + f(x)` with `hc_mult` parallel residual streams.
  `Op::HyperConnectMix` turns the mixing matmul's output into the `pre` collapse
  weights, the `post` output gates and the `comb` mixing matrix
  (Sinkhorn-normalised to approximately doubly stochastic);
  `Op::HyperConnectPre` collapses the streams to one vector for a sublayer; and
  `Op::HyperConnectPost` re-expands that sublayer's output back across the
  streams. `Op::HyperConnectMix`'s `gates` is `None` for the model head
  (llama.cpp's `build_hc_head`), which is the same arithmetic over a narrower
  `mixes`. All three run on CPU, Vulkan and Metal; `hc_mult` is accepted in
  `1..=infr_core::graph::HYPER_CONNECT_MAX_MULT` (8) and refused loudly on the
  host beyond that. See `docs/deepseek.md` § "Sinkhorn hyper-connections".
- **DeepSeek V4 attention primitives** (stage 4, op level): `Op::QkNorm`'s
  `weight` became optional, so a weightless per-head RMSNorm is expressible
  without a fake ones-vector operand; `Op::Attention` gained an optional
  per-head `sinks` input (one extra logit per head joining the softmax max and
  denominator, contributing no value); and `Op::Rope` gained a `backward` flag
  for de-roping (llama.cpp's `ggml_rope_ext_back` — the forward rotation with
  `sin` negated). All three run on CPU, Vulkan and Metal; `sinks: None` /
  `backward: false` / `weight: Some(..)` reproduce the previous code path
  exactly, so no existing model's numerics move. V4's grouped low-rank output
  projection needs no new op — it composes from `Op::CopyStrided` and
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

- **DeepSeek MLA attention gained a subgroup kernel tier, and it is the whole
  DeepSeek frame.** `mla.comp` was the only MLA kernel at any depth, and
  `INFR_PROF_OPS` measured it at 97.3% of decode GPU time at `-d 2048` and 89.4%
  of prefill. Its `-DMLA_SG` builds now walk the key range ONCE with an online
  softmax instead of twice, and score eight keys per `barrier()` with
  `subgroupAdd` instead of running a 128-lane shared-memory tree costing 18
  barriers per key. On an RX 7900 XTX with
  `JenniSD/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF`, three reps with the arms
  alternated: pp512 965.8 → 1035.5, pp2048 671.3 → 851.1, tg64 111.6 → 128.7,
  and tg64 at depth 2048 **6.3 → 11.9** t/s. Output is not bit-identical to the
  kernel it replaces (the online softmax sums the same terms in a different
  order); the CPU/Vulkan whole-vocab cosine on the 55-token seam test is
  0.99872, against 0.99861 for the kernel it replaces. `INFR_NO_MLA_SG` selects
  the scalar builds, which are unchanged and are also what a device that cannot
  pin a 32-wide subgroup gets.

- `Gguf::path()` → `Gguf::shards()`, which returns `(path, length)` per shard in
  order. `TensorBytes::file_range` / `Gguf::tensor_file_range` now report
  offsets in the model's concatenated file address space; for a single-file
  model that is the file offset it always was.

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

- **Every DeepSeek V2/V3 model added its attention output to the residual stream
  twice.** The MLA mixer arm in `seam/runner.rs` pushed its own `hidden += sub`
  and then fell through to the shared post-mixer residual add, so every layer of
  every `deepseek2`-family model (V2, V2-Lite, V3, V3.1, V3.2) ran
  `x + 2·Wo·attn` instead of `x + Wo·attn` — on the CPU, Vulkan and Metal
  backends alike, because the defect was in the shared graph rather than in a
  kernel. V4 is NOT affected: it builds `MixerW::Dsv4`, whose hyper-connection
  POST closes the residual wrap in place of the shared add, so it never emitted
  the second one. It never produced obviously broken text, which is why it
  survived: the only thing that catches it is an external oracle. Scored against
  llama.cpp 030ebb5's own logits over its own token ids, on
  `JenniSD/DeepSeek-V2-Lite-Chat-Q4_K_M-GGUF`, next-token probability cosine
  (CPU / Vulkan) goes 0.956/0.955 → 0.997/0.995 at 2 tokens, 0.334/0.316 →
  0.9998/0.99998 at 10, 0.862/0.862 → 0.997/0.999 at 27, and 0.980/0.920 →
  0.9999/0.9997 at 56. `infr`'s greedy continuation of "The capital of France
  is" is now llama.cpp's token for token, and the Vulkan repetition loop that
  DeepSeek-V2-Lite fell into after a dozen tokens is gone.

- **Every DeepSeek model prefilled one token per submit.** The batched-prefill
  eligibility scan (`moe_batched_ok` in `seam/runner.rs`) demanded a dp4a-mmq
  expert bank on EVERY layer, but DeepSeek's leading dense blocks ship a plain
  `ffn_gate/up/down` and no `ffn_*_exps` tensor at all — the missing lookup read
  as "unsupported dtype" and disqualified the whole model, so the entire prompt
  went through the per-token decode path. The scan now covers the layers that
  actually hold expert banks (`Config::is_moe_layer`), which also picks up
  llama4 checkpoints with a non-`1` `interleave_moe_layer_step`. Measured on an
  RX 7900 XTX: DeepSeek-V2-Lite-Chat Q4_K_M pp512 39.1 → 975.5 t/s and pp2048
  12.3 → 673.4 t/s; DeepSeek-MoE-16B-Chat Q4_K_M pp512 229.0 → 4185.0 t/s, level
  with llama.cpp's Vulkan backend on the same file. Decode is unchanged.
- **The int8 cooperative-matrix prefill GEMM assumed a fragment layout no device
  was ever asked about.** `native_gemm_i8cm_q8_0.comp` reads its accumulator
  elements at `(row, col) = (2*i + (lane>>4), lane&15)`, a mapping
  `KHR_cooperative_matrix` fixes per IMPLEMENTATION and which was derived
  empirically on RADV/RDNA3 — on a driver that lays the fragment out differently
  the kernel would have returned plausible wrong numbers with no error. The
  Vulkan backend now multiplies two known matrices on the device and reads the
  product back through that same mapping before arming the tier
  (`INFR_I8_COOPMAT=1` runs the check at init; a mismatch logs the offending
  element and leaves the tier off). Verified passing on an RX 7900 XTX, and
  verified to REFUSE when the mapping is perturbed.
- **Two drivers that misreport cooperative-matrix support are no longer
  believed.** AMD's proprietary and AMDVLK drivers advertise the unit on all
  GPUs, and Intel pre-Xe2 regresses on it (both documented upstream in
  llama.cpp). Cooperative matrix is now refused on `AMD_PROPRIETARY` /
  `AMD_OPEN_SOURCE` below RDNA3, and Intel pre-Xe2 keeps only the 8x8x16 tile
  that already required `INFR_CM_8X8=1`. The device is bucketed by probes (wave
  mode, wavefronts/SIMD, packed-dot bits, subgroup width, warps/SM), never by
  PCI device id, and the bucket plus driver id are printed in the GPU banner.
  Mesa RADV and NVIDIA are unaffected.
- **bf16 and fp8 were gated on an extension STRING with no feature bit, and the
  features were never enabled on the device.** `VK_KHR_shader_bfloat16` /
  `VK_EXT_shader_float8` now go through their feature structs like every other
  capability: `Capabilities::bf16` / `f8` mean the device will run it, the bf16
  and fp8 coopmat tiers additionally require `shaderBFloat16CooperativeMatrix` /
  `shaderFloat8CooperativeMatrix`, and both extension and feature are enabled on
  the logical device when a tier is live — without which those kernels' SPIR-V
  violated its VUID.
- **`maxPushConstantsSize` and `maxStorageBufferRange` are queried instead of
  assumed.** A kernel whose push block exceeds the device limit is refused by
  name at build time rather than failing a driver-side VUID, and the
  descriptor-range bind check now runs in RELEASE builds against the limit the
  device reported (it was a `debug_assert!` against an assumed 4 GiB).
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
