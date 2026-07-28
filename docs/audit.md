# infr Codebase Audit

Full module-by-module audit for bugs, correctness, perf, DRY, and YAGNI.
Delegated per-module to Opus agents; every finding below has been independently
verified against the source by the coordinator.

- **Repo state at audit start:** `main` @ `5457dbe`
- **Audit date:** 2026-07-21
- **Severity:** 🔴 critical (correctness/crash/UB) · 🟠 major (perf/latent bug)
  · 🟡 minor (DRY/YAGNI/style/robustness)

Each finding: `file:line` · severity · summary · why · suggested fix. Findings
that a delegated agent raised but the coordinator could **not**
reproduce/confirm are dropped (not listed) to keep this a verified-only ledger.

---

## Summary

**Living ledger — findings are pruned from this file as their fix lands on
`main` (TDD, one module slice at a time).**

- **Original audit:** 157 findings across 24 module slices (1 🔴 critical, 33 🟠
  major, 123 🟡 minor).
- **Remaining: 5 — and all actionable fixes are DONE.** 0 🔴, 2 🟠, 3 🟡, every
  one either a documented **deferral** or requiring hardware not on this box:
  - 🟠 a Mac-only latent metal wrong-kernel dispatch (iq4xs/iq4nl) — needs an
    Apple GPU to verify + re-point (tracked in
    [#80](https://github.com/kryptic-sh/infr/issues/80); the slice-23 metal
    fixes themselves are GPU-verified in
    [#81](https://github.com/kryptic-sh/infr/issues/81));
  - 🟠 `make_compute_kernel` OOM→`Result` — deferred (a 155-call-site
    `()`→`Result` migration disproportionate to a rare OOM path; panic messages
    improved);
  - 🟡×3 shader/dp4a DRY include-refactors — deferred (each would perturb the
    compiled SPV / recorded push-constant+descriptor stream the byte-identity
    goldens gate).

  **152 of 157 findings fixed** across 25 module slices (every crate cleared;
  the gated multi-GPU and parked MTP verified on real hardware), plus the
  `INFR_DEV`-unification feature. Every GPU-touching slice was byte-identity
  verified against the goldens on the RX 7900 XTX; every fix was TDD.

No finding was accepted on an agent's word — each was re-read against the source
by the coordinator; two agent-flagged "MAJOR"s (the Q5_1 clamp in the shader and
CPU quantizers) were **downgraded** to defensive-only after verifying the
overflow is unreachable. The MTP `catch_up` off-by-one first logged
**PLAUSIBLE** was later **CONFIRMED** by a 3-source position-convention trace
and fixed (`n_past+1`→`n_past`), with the acceptance-rate test passing.

### Resolved (landed on `main`)

- **`infr-hub` (all 6 findings)** — TDD, +10 tests. `1263bcc` verifies
  downloaded blobs against HF's expected LFS sha256 (the 🔴 critical); the slice
  adds `If-Range` resume (no stale-partial splice), an advisory `flock`
  serializing concurrent pulls, full split-shard (`-NNNNN-of-MMMMM`)
  download/relink, one shared `pick_gguf` selection for download+cache (kills
  the re-download loop) that excludes `mmproj`/float-master fallbacks,
  `refs/main` snapshot preference, trailing-colon ref parsing, and verify-once
  hashing.
- **`infr-llama` sampling + grammar (all 6 findings)** — TDD, +8 tests incl.
  byte-identity characterization guards on the default greedy/top-k paths.
  Repeat penalty is now per-distinct-token (was `repeat^K`); grammar
  `apply_mask` masks the padded-vocab tail to `-inf`; constrained decoding
  honors the `Sampler` (greedy at temp==0, stays inside the grammar mask);
  `seed|1` no longer collapses adjacent seeds; the `top_k==0` path uses a heap
  instead of a ~150K full sort; `truncated_dist`/`sample_logits` share one
  `truncated_softmax` helper. _Follow-up noted:_ `seed_rng()` (INFR_SEED path)
  has the same latent `|1` — deferred to avoid perturbing INFR_SEED determinism.
- **`infr-gguf` (all 5 findings)** — TDD, +9 tests. Corrupt/truncated GGUFs now
  return `Error::Loader` instead of panicking (`checked_add`/`checked_mul` in
  `ensure`/shape/offsets; `with_capacity` clamped to remaining bytes;
  non-pow2/zero `alignment` rejected). Host affine dequant is a single fused
  pass (no `sc`/`mn` materialization) — **bit-identical** (characterization
  tests on Q4_K/Q6_K assert `to_bits()` equality). `dequant_factored` returns
  `Result` (was `unreachable!`); `apply_signs`/`Gguf::resolve` de-duplicate the
  IQ sign loops and tensor-bounds lookup.
- **`infr-server` (all 6 findings)** — TDD, +13 tests. Streaming now emits a
  terminal SSE `error` frame on failure (never a fake `stop`) and a `DoneGuard`
  flushes `[DONE]` even on panic; a per-request `AtomicBool` cancel latch (set
  when the client's SSE `send` fails) is polled in `run_chat` → `req.abort()`,
  so a disconnected stream frees its GPU slot instead of running to
  `max_tokens`; `usage` carries real prompt/completion counts
  (`total=prompt+completion`); `make_id` appends a monotonic counter (no
  `--parallel` id collisions); optional `INFR_API_KEY` bearer auth (default
  open) + `max_tokens` clamp; `tools` passed as a borrowed `&Value` (no
  round-trip) and a malformed forced `tool_choice` now 400s instead of silently
  downgrading to auto. Trait `ChatGenerator::chat` signature updated with its
  two infr-cli impls. _Deferred:_ streaming `usage` chunk (needs
  `stream_options.include_usage` parsing) and the e2e disconnect→slot-release
  path (integration-only; the latch logic is unit-tested).
- **`infr-cli` (all 6 findings)** — TDD, +7 tests; dead-code `#![allow]`
  removed. `--dev` beats an inherited device setting via a pure
  `resolve_backend` + one unified `selected_backend()` reader. (As landed this
  `remove_var`d the sibling `INFR_METAL`/`INFR_CPU` envs; both keys were since
  removed outright, and the config campaign replaced the env writes with the CLI
  layer of the resolved `Config`, which already beats the env layer.) Model
  `resolve` only treats an existing `.gguf` FILE as local (mistyped paths give a
  clear error, not a network pull); the sweep sort is NaN-safe (`total_cmp`);
  the spurious "`--parallel` ignored" note fires only when explicitly set
  (`Option<usize>`); the DG→Metal→CPU→Vulkan funnel is one `build_chat_model`
  (was duplicated in run+serve); dead
  `ResolvedDevice`/`print_run_stats`/`bench -b`/`tg64@d` branch deleted. `#2`
  forced-tool retry now `reset_kv`s the session first (new no-op default
  `ChatModel::reset_kv` + Vulkan/Metal overrides). _Deferred:_ parse-GGUF- once
  (a large cross-crate API change, out of proportion to a 🟡).
- **`infr-chat` (all 5 findings)** — TDD, +14 tests. Streaming holdback now
  stops the content region at the earliest tool-call opener across every dialect
  `finish()` parses (Hermes `<tool_call>`, pipe `<|tool_call>`, and a
  whitespace-leading bare-`{` confirmed via `parse_bare_json_call`), so
  non-Hermes calls no longer leak as content **and** duplicate as a ToolCall —
  while ordinary content with a stray brace is preserved. The pipe arg-parser
  now translates JSON escapes (incl. `\uXXXX` surrogate pairs), keeps
  `inf`/`NaN` as strings (not `0`), uses `from_utf8_lossy` for keys, and strips
  a dangling `<|tool_call>` opener. The minijinja `Environment` is cached per
  template source (was rebuilt every render); `bos_token_id` falls back to `""`
  not `2` (EOS on Llama); `emit` scans via resumable cursors (was O(n²));
  `remove_spans` de-duplicates the two parsers.
- **`infr-llama` diffusion/parallel/util/chat (all 7 findings)** — TDD, +7
  tests. A failed `generate` now pops the just-pushed user turn (no orphaned
  double-user-turn); `parallel.rs checkout` drops the pool lock across the
  `seed_from` GPU submit (reserving the slots) and picks the longest-prefix free
  slot (was first-match); diffusion entropy sort is NaN-safe (`total_cmp`), the
  per-step `escratch` (~1MB) and `prev_argmax` clone are hoisted/reused, and the
  acceptance loop `break`s at the (proven-monotonic) entropy cutoff; per-block
  detok decodes the committed span once (was O(total²)); and
  `with_prof2_suppressed` / a `ChatModel::render` default / one
  `should_use_mtp(cfg)` de-duplicate the four backends (fixing the `wants_mtp`
  drift).
- **`infr-llama` seam/runner (all 7 findings)** — TDD, +9 tests. A new
  `last_written` tracker (the highest KV-written sequence position) + a pure
  `resident_after_gen` helper make the prefix cache record **exactly** the
  tokens whose KV rows exist: the `max_new==0` frontier and any unfed
  grammar-forced tokens are now excluded (were cached as resident → stale-KV
  corruption next turn), while a normal run reproduces the old
  `prompt ++ out[..len-1]` result (byte-identical, no logits/golden path
  touched). Empty non-denoise prompt errors instead of underflow-panicking;
  prompt/canvas token ids are validated `< vocab` (`validate_token_ids`);
  session-stable derivations are cached on `SeamKv` (`SessionStable`) instead of
  recomputed per warm call; host-embed skips the per-token `Vec` when
  `embed_scale==1`; `bind_layer_io` de-duplicates all 5 KV+ weight bind sites.
  _Deferred:_ the `build` 11-positional-bool param-soup (a call-site-wide
  refactor with a transposed-arg risk, zero correctness gain).
- **`infr-llama` seam/mod (all 7 findings)** — TDD, +5 tests. TP/pipeline
  binders exempt `chunk_covered_dense_tensor` (`lm_head`/`token_embd`) from the
  BDA element cap (a large-vocab model no longer hard-rejects under
  `INFR_TENSOR_PARALLEL`/`INFR_PIPELINE`); the process-global `PINNED_UBATCH`/
  `PINNED_KV_Q8` `OnceLock`s become per-session `PlacementPins` via a
  thread-scoped RAII guard (multi-model `infr multi` no longer leaks model A's
  chunk-height/q8 decision to model B; single-model falls back to the old
  behavior byte-identically); MoE KV placement estimate honors auto-q8;
  `TokenEmbd::get` is fallible (corrupt GGUF → error, not `.expect()`); warm
  sessions skip the unused `vulkan_moe_binder`; a fn-instrument macro is removed
  from an `impl Deref`; and `parse_device_list` / `run_dense_oneshot` /
  `dense_multi_gpu_guard` / `cpu_upload_bind`/`metal_upload_bind` de-duplicate
  the multi-GPU + backend-bind boilerplate.
- **`infr-llama` seam/model+weights+sc (all 6 findings)** — TDD, +5 tests.
  `SlotPool::pick` now picks the longest-prefix slot (was first-match, via a
  pure `pick_continuation`); `bench_vulkan`'s unsafe `INFR_PROF_OPS` env race is
  gone — suppression is a non-env `AtomicBool` in infr-prof-rt that the recorder
  reads (also fixes the warmup callers); `generate_metal_spec` reuses a
  persistent `feed` buffer (was O(n²) `committed.clone()` per round);
  `weight_footprint` is memoized on `SeamModel` (one scan/open);
  `spec_accept_stochastic` `.expect()`s the empty-distribution contract instead
  of emitting a bogus token 0; the ~8 `tokenizer.encode` sites and the
  DiffusionGemma upload closures route through shared helpers. No logits/golden
  path touched. **infr-llama's seam core is now fully cleared** (only the parked
  MTP slice remains in this crate).
- **`infr-vulkan` recorder (all 6 findings)** — TDD, +2 tests, **gpu_seam
  byte-identity verified on the RX 7900 XTX** (MoE-mmq resident+paged, GEMV
  row1/mrow/mw/id, and weight-addr goldens all match host). Per-dispatch
  `INFR_GEMV_*` env reads are resolved once into a `GemvKnobs` `OnceLock` (same
  routing — the `OnceLock` was later deleted by the config campaign, which
  hoists the same values off the backend's `Config` instead);
  `dispatch3`/`bind_descriptors` use stack arrays (no per-dispatch heap `Vec`);
  `finish`/`finish_nowait` free the cmd buffer + pools on their error paths (was
  a leak); the two ~180-line `matmul_mmq_experts`/`_paged` tables collapse to
  one `moe_mmq_desc` table (a drift test asserts the same `(kernel,nbind)` for
  every `MOE_MMQ_DTYPES`); the descriptor pool is proportioned so
  sets+descriptors exhaust together. #5: the 2 genuinely parity-only `_at` fns
  are gated behind a new `parity` feature — the other 9 carried a **stale** "not
  wired" doc but are in fact production-wired (correctly left `pub`).
- **`infr-vulkan` adapter (all 6 findings)** — TDD, +3 pure tests, **gpu_seam
  byte-identity verified** (GEMM prefill `nc_gemm`/`warp_gemm`, attention/KV
  `kv_addr` 15, MoE-mmq all match host). Batched-MoE `counts` drops the
  redundant host-blocking calloc (device-zeroed content unchanged, ~27µs/layer
  sync gone); the static split-K attention now bounds `n_chunks ≤ 1024` (was an
  OOB `attn_combine wexp[1024]` write above ~524k keys — the fix is proven
  identical to the old formula for all realistic spans); `GatedAct::Silu`
  rejects a strided gate instead of computing silently-wrong; cross-dtype
  `Op::Copy` with `src_off!=0` returns `Err` instead of panicking;
  `INFR_FLASH_MIN_ROWS` is read once; and
  `split_k_plan`/`native_warp_gemm`/`with_padded_dst` de-duplicate the three
  GEMM tiers (identical tile/split decisions).
- **`infr-vulkan` lib + pcache (all 7 findings)** — TDD, +5 tests, **gpu_seam
  byte-identity verified** (BDA weight-addr 26, attention/KV 15, MoE, GEMM). An
  `InstanceCleanup` RAII guard (disarmed before `Ok`) frees the device/instance/
  cmd-pool on every recoverable `new_selected` error path (was leaked for
  process life on the CPU-fallback path); device-local zero-init rounds the fill
  up (`fill_span`) so a non-4-multiple buffer's tail can't leak recycled VRAM;
  the write-only `uma_spilled` counter is removed;
  `alloc_arena_bda`/`bda_weight_alloc` reuse the already-stored `own_addr` (no
  re-query); the UMA-spill path skips its duplicate VRAM budget check; and
  `pcache` uses a unique temp suffix + a per-instance nonce on the
  poison-tripwire marker (no cross-thread/backend collisions). Success/alloc
  path byte-identical.
- **`infr-vulkan` ops + pager (6 of 7; #2 deferred)** — TDD, +7 tests,
  **gpu_seam byte-identity verified** (pager gemv/mmq/multi, MoE, attention).
  `kernel`/ `kernel_sg` build under one lock (`entry().or_insert_with`) — no
  double-compile/leaked-pipeline race; the cache hit `debug_assert`s the
  request's `n_buf`/`push_size`/spv-hash match (silent-mismatch guard);
  `GpuPager::new` + the `pub` accessors return `Err` on bad dims / unregistered
  buffers (were `assert!`/`.expect()`); the triplicated LUT evict+insert becomes
  one `record_placement` (silent-zero-safe, byte-identical);
  `touch_role`/`stage` reuse scratch buffers;
  `expert_bytes`/`pager_stats_enabled`/`stats_suffix` de-duplicate the two
  sessions. #2 (make_compute_kernel OOM→Result) deferred — 155 call sites across
  127 `()`-returning fns; panic messages improved instead.
- **`infr-vulkan` gemm + matmul + linear (all 7 findings)** — TDD, +4 tests,
  **gpu_seam byte-identity verified** (GEMV row1/mrow/mw, MoE id-gemv
  real-dims + new-formats all match host). Every `*_kernel_name` gate now
  DELEGATES to its `*_build_spv`/`*_spv` source (id/idm/mmv/mw/mrow/embed) so a
  name and its shader can't drift into a mid-inference panic — each table was
  diffed identical before delegating and a drift-guard test asserts equality
  across all dtypes; `Iq4Xs` gets an explicit `native_mmv_mrow_res_supported`
  predicate (no phantom residual panic) + the false adapter comment fixed;
  `matmul_f32` is `#[doc(hidden)]` with the per-call `println!` gone and shape
  math overflow-checked; push constants use `to_le_bytes` (matching the shader
  contract); the no-op `v!` macro and the re-inlined `spv_words` are removed;
  `moe_expert_floor_covers_dense_set` derives from `native_dense_dtypes()` (+ an
  exhaustive-`DType` guard).
- **`infr-vulkan` misc shaders (6 of 7; #6 DRY deferred)** — TDD, **gpu_seam
  verified** (`dg_eb_sample`, `chunked_delta_math`, `sample_topk`, MoE mmq/id +
  ragged-bucket `pager_mmq`/`weight_addr`, `add_bias` all pass). `dg_eb_sample`
  argmax now carries the lower-index tie-break (matches host/`argmax.comp` — the
  golden passes unchanged, no re-bless needed); `rope.comp` writes only the
  un-rotated tail (halves KV-store traffic, byte-identical); the `-1.0/0.0`
  `-inf` sentinel is `-1e30`; `part[8]` cross-subgroup arrays sized to
  `part[16]`; `embed_gather` gets a host `ne%32==0` guard; `deltanet_gates` uses
  `subgroupInclusiveAdd` (sg32-pinned, within tolerance); `moe_bucket_scan`
  drops the fused `fill[]=0` (zeroed by a separate overlapping dispatch — MoE
  goldens byte-identical).
- **`infr-vulkan` attention/flash shaders (7 of 8; #7 deferred)** — **gpu_seam
  verified** (flash stage/warp/coopmat/matches-cpu + `kv_addr` + `kv_q8` goldens
  pass, run serially). The direct `coopMatLoad` K/V over-read past `kv_len` is
  made SAFE by tightening the host `flash_geom` gate to
  `kv_len.div_ceil(64)*64 <= att_cap_rows` (a non-ring KV cache allocates
  non-64-aligned rows; flash reads 64-aligned tiles) — the masked over-read
  columns contribute nothing; `attn_flash.comp` documents the mpad precondition
  (keeps writing all BM rows to stay bit-identical with the staged/split-K paths
  — a store guard there was reverted after it broke the STAGE==direct golden);
  host debug-asserts guard `attn_pv` `tilesN`, `attn_combine` `ntile`, and
  `attn_partial_mrows` `chunk`; `quant_kv` `Q5_1` gets the `clamp(0,31)` (never
  triggers); dead `MAXS` removed. **Pre-existing note, since RESOLVED:** the
  flash parity LIB tests used to race on the process-global `INFR_FLASH_*`
  variables under parallel cargo threads and needed `--test-threads=1`. The
  config campaign moved them onto an explicit `Config`; `infr-vulkan`'s suite
  runs in parallel again.
- **`infr-vulkan` GEMM/GEMV shaders (7 of 8; #6 dp4a DRY deferred)** —
  **gpu_seam byte-identity verified** (`weight_addr`/`gemm_proj` 28,
  `sample_topk`, MoE-mmq, `nc_gemm` all match host). `gemm_proj.comp` now stages
  weights k-inner (coalesced global loads) and reads the arena through
  `native_weight_addr.glsl` so the compiler selects the scalar-base **saddr**
  `global_load` instead of the divergent-index 64-bit deref (the documented
  ~2.2× streamed regressor) — both bit-verified identical; the padded-C store
  contract is documented (all callers pad); `native_gemv_rm_v2` `reg_part` is
  sized `THREADS/16` (was OOB under WG128) + dead `tot` gone;
  `native_mmv_mrow`'s unused `part[]` is `#ifndef OUTS4`; `moe_sample` top-k
  gather is two-pass so a threshold tie can't evict a strictly-greater logit;
  and the superseded `gemm_dp4a`/`gemm_coopmat` probe kernels are removed
  (grep-proven never dispatched).
- **`infr-core` (all 6 findings)** — TDD, +8 tests, **gpu_seam smoke verified**
  (kv*addr/MoE/GEMM/pager goldens byte-identical). `copy_buffer` downloads only
  the requested prefix (was the whole `src`) + returns `Err` on oversize (was a
  panic); `MetaValue::as_u64` uses `try_from` (negative → `None`, not
  `u64::MAX`); `in_place_inputs` is memoized on `Graph` (was an O(ops) rescan
  per token on the execute path); the paged-dtype drift test asserts
  set-EQUALITY; `pager` `take_slot` prunes the `epoch` entry (now bounded by
  `n_slots`); `Bindings` is a `Vec<Option<&dyn Buffer>>` (hash-free, fully
  contained, semantics identical) and the seam caches `capabilities()`
  once/build. \_Deferred:* `capabilities()→&Capabilities` (the multi-GPU
  wrappers build a mutated caps copy per call — a borrow would change when caps
  are computed).
- **`infr-cpu` lib/pool/turbo/kvquant/moe (all 6 findings)** — TDD, +6 tests;
  byte-identity confirmed by the CPU tests + the CPU↔GPU `seam_op_parity`
  (11/11, incl. copystrided/deltanet/rope) + full-model `gpu_seam_matches_cpu`.
  `Op::Sample` treats `top_k==0` as no-truncation (was a `k-1` underflow panic);
  `weight_cache` is keyed on `(addr,len,dtype)` (no stale-address collision on
  model reload); `q5_1_block` clamps to 0..31; the q4k/q6k pack cache re-checks
  under the insert lock (no double-repack under parallel serve); the `Rope`/
  `Copy`/`CopyStrided`/`DeltaNet` op arms borrow instead of clone; the `WriteKv`
  Q8 rounding comment is corrected (no golden pins those bytes). _Deferred:_
  skipping the execute-prologue zero-fill (needs a per-tensor read-before-write
  dataflow pass — `CopyStrided`/`RmsNormAdd` dsts require it).
- **`infr-cpu` kernels (all 5 findings)** — TDD, +4 tests;
  `gpu_seam_bf16_matches_cpu` stays token-identical + `cpu_golden_qwen3`
  unaffected. `dot_bf16` now uses the same 8-independent-accumulator structure
  as `dot`/ `dot_f16` (was a latency-bound serial chain — the one intended
  bit-for-bit float reorder, within the bf16 golden's tolerance);
  `debug_assert`s catch a non-256-multiple `in_f` on the ten K-quant dispatchers
  and unequal lengths in `dot` (were silent wrong-answers);
  `vec_dot_q6k_batch_avx512bw` is renamed `_vnni` (it requires VNNI); the
  144-byte Q4*K decode block is extracted to a shared `q4k_decode_row`
  (bit-identical, verified through the real AVX512BW path); the Q6_K maddubs
  comment is corrected. **infr-cpu is fully cleared.** \_Deferred:* the
  batch-kernel per-call scratch-alloc reduction (a dedicated perf change).
- **`infr-prof` + `infr-prof-rt` (all 4 findings)** — TDD, +tests; profiling
  side-channel, no inference/golden impact (the `#[instrument]` macro only
  expands under the non-default `infr_profile` cfg). `should_skip` matches the
  attribute path structurally via `syn` (`infr_prof::skip`, incl. the `cfg_attr`
  form, excluding `doc` attrs); module `cfg`-skip fires only for exactly `test`/
  `all(...,test,...)` (not `not(test)`/`feature="test-*"`); the GPU-report sort
  uses `total_cmp` (NaN-safe); a real `[dropped]` aggregate surfaces over-cap
  sites; the intentional per-thread-table retention is documented.
  (`iquant_grids.rs` + `infr-engine` were audited **clean** — pure tables / a
  re-export shim.)
- **`infr-metal` exec.rs (all 6 findings)** — verified via
  `cargo check`/`clippy` against `x86_64-apple-darwin` (the crate is
  macOS-gated; no Apple GPU here to run it). `Op::Rope` is excluded from the
  decode-replay tape so a llama-family graph no longer replays token-0's frozen
  RoPE angle (there's no i32 `rope_f32` variant to live-bind like `qknormrope`,
  so exclude-from-replay is the safe pure-Rust fix); `sample_split` packs the
  same `top_k.min(64)` it sizes (was an OOB write); `Op::Softmax` gates on the
  threadgroup cap with a host scalar fallback; the 5 per-op transient buffers
  use `scratch_buf`; redundant per-dispatch PSO double-lookups fetch once; and
  the 4 parallel quik8 kernel-name tables collapse to one registry (a missing
  base is a loud `Err`; a test enumerates all 16×4 == the old arms). _Surfaced a
  new latent bug_ (iq4xs/iq4nl wrong-kernel default) — preserved + flagged for
  Mac verification (see the metal section).
- **`infr-vulkan` gated multi-GPU (all 8 findings)** — TDD, +7 tests, **the
  synthetic 2-device parity tests pass token-identical** on the RX 7900 XTX +
  iGPU (`tensor_parallel`/`ep`/`pipeline_matches_single_device`). P2P
  cross-device dma-buf buffers emit `VK_QUEUE_FAMILY_EXTERNAL` release/acquire
  ownership transfers (each backend has one queue family, so
  `EXCLUSIVE`+transfer is the spec-correct fix, not `CONCURRENT`); `AllReduce`
  carries the boundary dtype and cleanly rejects a non-f32 boundary (the add
  shader is f32-only); the per-rank KV decl is shrunk to `numel/W`; the pipeline
  residual handoff copies from the device that last _wrote_ each tensor;
  `tp_sem` frees command buffers on every error path; the host-bounce reduce
  downloads each producer once (was W−1×); `dtype_bytes` is one helper, boundary
  sizes are asserted equal, and OpaqueFd threads the exporter's memory-type
  index. Reduce arithmetic unchanged → parity holds. _Deferred:_ a true f16
  all-reduce (needs an f16 add shader).
- **`infr-llama` MTP (all 5 findings)** — TDD, +1 test; **`INFR_MTP=1`
  `mtp_spec_matches_target_only_greedy` stays token-identical** and
  `mtp_head_trunk_acceptance_rate` passes on the RX 7900 XTX. Cycle `catch_up`
  writes committed tokens at `start_pos = n_past` (was `n_past+1` — CONFIRMED a
  bug via a 3-source position-convention trace; the head-KV row at absolute pos
  `p` stores `(t_p, h_{p-1})`, so the `+1` stored every pair one row too high
  with a stale `h`); `catch_up` uses a `want_logits=false` KV-only forward
  (drops the dead lm*head GEMM+readback it discarded every cycle); `pending_h`
  takes `next_tok`'s own hidden in the reprime branch; the three
  `base=m-(cand+1)` sites use a guarded `nonleading_base()`; and the three
  lock-step leading `Option`s collapse to one `Option<Leading{…}>`. \_Deferred:*
  the `emit_mtp_layer`/forward- glue graph extractions (touch the recorded op
  sequence; the two builders differ structurally).

### Highest-priority (production default paths)

| #      | Sev | Location                           | Issue                                                                                                        |
| ------ | --- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| ~~1~~  | ✅  | `infr-hub`                         | ~~Downloaded blob never sha256-verified~~ — **FIXED** (`1263bcc`, + full hub slice).                         |
| ~~2~~  | ✅  | `infr-llama chat/mod.rs`           | ~~Generate error orphans the user turn~~ — **FIXED** (`Err` arm pops the user turn).                         |
| ~~3~~  | ✅  | `infr-server lib.rs`               | ~~Streaming swallows errors as `stop`~~ — **FIXED** (error frame + panic-safe `DoneGuard`).                  |
| ~~4~~  | ✅  | `infr-server lib.rs`               | ~~No per-request cancellation~~ — **FIXED** (cancel latch → `req.abort()` frees the slot).                   |
| ~~5~~  | ✅  | `infr-llama runner.rs`             | ~~Prefix-cache records unmaterialized KV rows~~ — **FIXED** (`last_written` tracker + `resident_after_gen`). |
| ~~6~~  | ✅  | `infr-vulkan adapter.rs`           | ~~Static split-K `n_chunks>1024` overruns `wexp[1024]`~~ — **FIXED** (bounds `n_chunks ≤ 1024`).             |
| ~~7~~  | ✅  | `infr-vulkan ops.rs`               | ~~Kernel-cache double-checked lock double-compiles + leaks~~ — **FIXED** (single-lock `or_insert_with`).     |
| ~~8~~  | ✅  | `infr-llama sampling.rs`           | ~~Repeat penalty per-occurrence~~ — **FIXED** (`70bbe4e`; now per-distinct-token).                           |
| ~~9~~  | ✅  | `infr-vulkan shaders dg_eb_sample` | ~~argmax reduce drops the lower-index tie-break~~ — **FIXED** (matches host/`argmax.comp` on ties).          |
| ~~10~~ | ✅  | `infr-gguf lib.rs`                 | ~~Corrupt GGUF `pos+n` overflow panic~~ — **FIXED** (`checked_add`/`checked_mul` → `Error::Loader`).         |
| ~~11~~ | ✅  | `infr-cli main.rs`                 | ~~`--dev` can't override inherited backend env~~ — **FIXED** (clears siblings; unified precedence).          |
| ~~12~~ | ✅  | `infr-metal exec.rs`               | ~~`Op::Rope` frozen RoPE on the replay tape~~ — **FIXED** (excluded from replay).                            |

The 2 remaining 🟠 are both non-actionable here: a Mac-only latent metal
wrong-kernel dispatch (needs an Apple GPU) and the deferred
`make_compute_kernel` OOM→Result. Full detail per module.

### Cross-cutting themes

- **`partial_cmp(...).unwrap()` NaN panics** recur in ≥5 files (diffusion, cli
  sweep, prof-rt, sampling) — a shared `by_desc_f64`/`total_cmp` comparator
  kills the class.
- **`assert!`/`.expect()`/`panic!` on recoverable input** (unregistered pager
  buffers, `GpuPager::new`, `Op::Copy` src_off, `make_compute_kernel` OOM, GGUF
  parse, `Op::Sample` `top_k==0`) — should return `Err`.
- **Name-table vs SPV-table drift guarded only at runtime by `.expect()`** —
  FIXED in vulkan `gemm.rs`/`linear.rs` (name gates now delegate to the SPV
  source); still open in Metal `exec.rs`.
- **Per-render / per-warm-call recomputation** (jinja env rebuild,
  session-stable seam derivations, capabilities clone, GGUF re-parse ×5 in the
  CLI).

<!-- SLICES APPENDED BELOW AS THEY ARE VERIFIED -->

## infr-vulkan/src/ops.rs + pager.rs

_6 of 7 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟠 `make_compute_kernel` panics on recoverable driver/OOM errors**
   (`.expect()`/ `panic!`); `kernel`/`kernel_sg` return a bare `ComputeKernel`,
   so a late `OUT_OF_DEVICE_MEMORY` on kernel compile aborts the process.
   _Deferred:_ threading `Result` ripples to **155 call sites across 127
   `()`-returning fns** (ops/gemm/recorder) — a byte-critical mass migration
   disproportionate to a rare OOM path. Interim: the panic messages now name the
   kernel + flag it as a recoverable alloc failure. A full `Result`-ification of
   the recorder dispatch surface is its own focused effort.

## infr-vulkan/shaders — attention / flash / KV / softmax

_7 of 8 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟡 perf/DRY** — `attn_combine.comp:36,40` recompute `mm`/`l` over all `nch`
   in every one of 32 lanes; the causal-mask + online-softmax rescale block is
   copy-pasted across 5 flash kernels; ring `RROW`/`rcap` + Q8/f16 vec4 readers
   are duplicated. _Deferred:_ `attn_combine` is dispatched WITHOUT a pinned
   subgroup size, so a `subgroupMax`/`subgroupAdd` reduce would be wrong on a
   16-lane device without adding `requiredSubgroupSize=32` (a host pipeline
   change with portability risk), and the `l` (weighted-sum) reduce is only
   within-tolerance, not byte-identical — both risk the byte-identity gate. The
   flash mask/rescale include-refactor would perturb the 5 kernels' numerics.

## infr-vulkan/shaders — GEMM / GEMV / MoE-expert matmul

_7 of 8 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟡 DRY — the int8 dp4a decode helpers (`rb`/`ru16`/`f16tof32`/`k4` +
   per-format `dpsub`/`wdec`) are copy-pasted across the `native_mmv*` +
   `native_gemm_mmq_*` shaders; the `KV_IQ4NL_W` table duplicated ≥4×.**
   _Deferred:_ the helper SETS differ per file (each activates exactly one
   `FMT_*`; `k4` only in Q4K/Q5K, odd-stride readers only in some), spread
   across ~15 mmq + 4 mmv shaders. A shared `native_dp4a.glsl` include would
   have to reconcile those conditional blocks + add build.rs include wiring, and
   any textual drift risks the compiled SPV on the byte-identity-critical
   GEMM/MoE goldens.

## infr-vulkan/shaders — norm / rope / activation / sampling / MoE-routing / misc

_6 of 7 findings fixed (see Resolved log); the one below is **DEFERRED**._

1. **🟡 DRY — two shader pairs (`qk_norm_rope` vs `_interleaved`, `sample_topk`
   vs `moe_sample`) look duplicated but aren't cleanly mergeable.** On
   inspection they differ by more than the audit assumed: `qk_norm_rope` carries
   an extra `kcap` push-constant + the SWA ring-modulo its interleaved twin
   lacks (different PC layout); `sample_topk` is two-stage (`PASS1`/`PASS2` + a
   `CHAIN` variant, buffer-sourced `u`) while `moe_sample` is single-stage with
   `u` in the push constant (different binding layouts). _Deferred:_ folding via
   `-DINTERLEAVED` would have to reconcile the PC/binding structs and the
   build.rs compile list, risking the recorded push-constant/descriptor stream
   for a 🟡 DRY gain.

## infr-metal/src/exec.rs (Metal backend — not runnable on this Linux box)

_All 6 original findings fixed (see Resolved log). The one below is a **new,
pre-existing** latent bug surfaced while unifying the kernel tables — preserved
byte-for-byte in the fix and flagged for a Mac to verify+fix._

1. **🟠 (Mac-verify) iq4xs/iq4nl reach the `cmm_ks` default that dispatches
   `linear_quik8_cmm_ks`, though their own `linear_iq4xs_cmm_ks`/
   `linear_iq4nl_cmm_ks` kernels exist** — so at m≥2/m≥5 with deep-k and
   `out_f%64==0` the wrong (quik8) kernel decodes their codes. Latent +
   Metal-only; preserved (not a regression). **Tracked in
   [#80](https://github.com/kryptic-sh/infr/issues/80)** (cc @digitaloten) — an
   Apple GPU re-points the two registry cells. The slice-23 metal fixes are
   GPU-verified in [#81](https://github.com/kryptic-sh/infr/issues/81).
