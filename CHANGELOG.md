# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Streamed tool-call arguments instead of one final blob: `/v1/chat/completions` now emits `delta.tool_calls[].function.arguments` fragments as the model writes them and sends only the remainder in the finished call, and `/v1/responses` forwards the same fragments as `response.function_call_arguments.delta` events.
- Made `include:["reasoning.encrypted_content"]` return a store-backed `encrypted_content` handle that a later request's `reasoning` input item resolves back into the prompt; the request is rejected with a named `include` parameter when `store=true` is not in force.
- Added `serve.response_store_path` (`INFR_RESPONSE_STORE_PATH`) to mirror the `/v1/responses` store to a JSON file, so `previous_response_id` and `conversation` chains survive a restart. Unset stays memory-only and nothing a client said touches a disk; a configured file is loaded at startup and rewritten atomically after every change, and a file that cannot be parsed refuses to start instead of silently presenting an empty store.
- Uncensor: a `controlvector` GGUF named by `INFR_UNCENSOR_VECTOR` projects one refusal direction out of each projected layer of the Qwen3.8 wide residual (`INFR_UNCENSOR_FIRST_LAYER` / `INFR_UNCENSOR_LAST_LAYER` set the range, default 4-44). Implemented in the new `crates/infr-uncensor` crate with one CPU arm and one Vulkan kernel; Metal reports it as unsupported. Without the variable nothing is allocated and generation is bit-for-bit what it was.
- Per-request `uncensor: true|false` on `/v1/chat/completions` and `/v1/responses`, with `experimental_speed_projection` accepted as an alias and a 400 when the two disagree. The switch is a device-side scalar, so the compiled plan survives; flipping a slot drops its materialized KV and re-reads the prompt.
- The launch wizard (`scripts/infr-wizard.ps1`) asks for the projection on models that support it, prints what it does in Chinese and English first, defaults to No, and on a yes fetches the pinned direction file (`cjxzdzh/…/cvec-qwen4exp.gguf`, sha256-verified, `huggingface.co` then `hf-mirror.com`) into `models/control-vectors/` and remembers the choice in the wizard state, emitting `--set uncensor.vector=<path>`. The `Start-INFR-Wizard.cmd` launcher takes no uncensor arguments: one place asks, one place runs.
- Added `debug.qsa_index_dump` (`INFR_QSA_INDEX_DUMP`, default off, Qwen3.8 single-stream decode only) to answer whether QSA selection makes KV pages migratable: it appends one record per frontier decode step to `qsa-index-dump.bin`, i32 `[pos, n_fed]` followed by `[n_layer, top_k/ratio]` selected block indices. The per-layer rows are the whole point — `qsa_indices` is ONE shared `g.internal` scratch reused by every attention layer, and internals live in a backend-owned scratch arena with no public readback handle, so a post-step readback of that tensor only ever shows the LAST layer and the cross-layer union (the number a residency policy actually needs) is unmeasurable. The build instead declares a `g.output` bound to a caller-allocated `Readback` buffer and pushes one `Op::Copy` per attention layer into row `l`; nothing is allocated when the knob is off. **Verdict: the QSA-hotness page-migration direction is falsified at frame granularity.** Measured on Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64, release binary sha `372b6676…`, ctx 65536, `-u 5120`, page 64, Q8_0 KV, greedy, `--parallel 1`, elastic corridor + 16 GiB RAM tier, hd256 flash forced off, over 101 concatenated documentation markdown files: 236 records, two sparse groups (12948 and 48100 prompt tokens). At 48100 tokens (754 pages, 128 steps, 12 attention layers) each layer selects **23.89%** of pages and the per-step cross-layer UNION is **64.89%** — so ~35% of frames really are cold for a given step, and my earlier uniform-independence estimate (per-layer 39.3%, 12-layer union 99.8%, "no cold frames exist") was **wrong**: selection IS concentrated (23.89% against the 49.4% a uniform draw would give at this depth, cross-layer Jaccard 0.4068 against ~0.25). But concentration is not temporal stability. The hot set rotates ~73 pages per step (step-to-step union Jaccard **0.7409**), a 16-step window already covers **96.70%** of all frames and a 64-step window **99.92%**; only **3** pages are hot in ≥90% of steps across all layers (180 in any one layer), every one of the 754 pages is touched somewhere, and the recent 10% of the prompt accounts for just **9.15%** of selections — the head of the document carries 43% of them. Ranking pages by their frequency over the WHOLE group (an oracle that uses future steps, hence an upper bound on any causal policy) and pinning that fraction still has to DMA the rest back every step: pin 25% → 36.24% union hit, **336.3 MiB/step** missing; pin 50% → 66.70%, **175.6 MiB/step**; pin 65% → 81.08%, 99.8 MiB/step; pin 80% → 92.04%, 42.0 MiB/step to save 0.2 GiB. Against a measured 23.7 ms decode step (42.3 t/s) the 50% pin costs +17.6 ms (+74%) at 10 GB/s or +8.8 ms (+37%) even at 20 GB/s, so any policy that avoids per-step DMA must hold ~97% of frames and saves ~3%. Splitting a frame per layer does not rescue it either: per-layer step Jaccard falls to **0.5600** and a layer's own selection needs **84.47%** residency over a 16-step window (89.25% over 64), i.e. a 15.5% saving on a KV ceiling of 1024 frames = **1.08 GiB**, about 0.17 GiB, next to an `expert_cache_target` of **4.15 GiB** and an `elastic_pool` of **13.19 GiB**. The shallow group is not even sparse enough to have cold frames (per-layer 60.46%, union 96.33%, step Jaccard 0.9625). The conversation-granularity cold-tail sink is therefore the right granularity and stays; the PCIe traffic worth attacking is expert residency, not KV. `bad_indices=0` in both groups, tree `entries=2 evictions=0 aux_refusals=0`, zero VVL/ERROR/panic, and the analyzer was self-validated on synthetic dumps first (perfectly correlated → Jaccard 1.0000; independent uniform → per-layer 39.77% / union 99.68% against a 39.3% / 99.78% theoretical prediction).

### Changed

- Made the paged-KV frame corridor commit lazily and hand idle space back to the experts, behind `kv.paged_elastic` (`INFR_KV_PAGED_ELASTIC=1`, default off). The corridor used to lease `sessions × ceil(ctx / page_tokens)` frames when the tree was built and never return them: a 4345-token request held 69 of 2048 frames (0.04 GiB) while 1.33 GiB stayed pledged to the unified arena where the expert pager could not reach it. With the knob on, the tree commits an eighth of the budget (at least 128 frames) and claims the next chunk from the arena only when free frames run out — the ceiling is the same worst case the arena's KV reserve was already sized for, so no other VRAM budget moves. Growth happens inside prefill and first returns Prefill's temporary borrow of the corridor's uncommitted units (`handoff_gpu_decode` + `enter_decode`, the same serial boundary segmented-KV growth uses), otherwise `corridor_blockers()` treats that live non-Expert allocation as a blocker. Return happens at the request boundary (`paged_publish_detach`) and retires the frame-pool slots BEFORE releasing the backend lease, so a failed return leaks recoverable arena bytes but can never leave a live frame pointing at memory already handed to an expert; slot indices are not recycled (`present` tombstones) and only a fully empty segment is returned, so a chain pinned by a tree entry cannot lose its own corridor. `free_kv_frames` runs under `with_unified_exclusive_result`, mutually exclusive with expert claims. Measured same-window, same release binary, Qwen3.6-35B-A3B-APEX-I-Balanced, ctx 131072, `-u 2048`, Q8_0 KV, greedy, `--parallel 1`: at 4345 tokens the lease went 1.33 → 0.17 GiB and decode 140.4 → 145.9 t/s (+3.9%, tree-free 165.1); at 30685 tokens one growth to 512 frames (0.33 GiB) left decode flat at 128.6 t/s; at 125445 tokens seven consecutive growths to the 2048-frame ceiling landed on exactly the same 1962 frames as the fixed corridor, decode 82.5 vs 82.7 (noise) and cold TTFT +0.3%. All three `cached_tokens` (4288 / 30656 / 125440) and generated-text shas are byte-identical to the fixed corridor, with zero VVL/ERROR/panic. A ctx-32768 pressure sequence exercised the full round trip: a 30K chain sank to RAM (`sunk=481`), the corridor shrank 512 → 128 frames (`corridor_shrinks=3`, 0.31 GiB back to the experts), and the revisit promoted it back (`promoted=479`) and re-grew — TTFT 287 ms against 13466 ms cold, `evictions=0`. `paged KV prefix cache status` gains `corridor_grows` / `corridor_shrinks` / `corridor_ceiling`. It reclaims only leases nobody is using: the remaining per-token paged decode cost is independent of corridor size (at 4K the experts held the same 218.6 MiB with a smaller KV footprint and `decode_ms` still went 775 → 879), which is what frame-level hot/cold tiering addresses rather than the elastic corridor.
- Sank the paged-KV cold TAIL instead of the whole chain, so a chain longer than the RAM frame pool survives instead of being deleted. The old sinker's whole-chain capacity check skipped any entry whose chain could not fit the free RAM frames, the cascade then yielded nothing, and the evictor dropped the entry — a revisit re-prefilled from zero. A sink now moves a bounded suffix (as much as the target tier can host), keeps the reused prefix resident, and takes it from the TAIL so the chain stays a monotone VRAM→RAM→disk gradient; promotion truncates reuse at the first host frame, so a host frame in front of device frames silently discards every page behind it. The warm-prefix floor follows the depth the last attach actually reused, not the depth `best_match` matched: a recurrent entry resumes only at a snapshot boundary, so a deep token match can attach nothing, and reserving that prefix would pin VRAM no request can read. Measured same-window, same release profile, Qwen3.6-35B-A3B, ctx 131072, `-u 2048`, page 64, Q8_0 KV, greedy, `kv.paged_cache_max=1GiB` (1542 frames): a 125450-token request holds 1960 frames, and after an unrelated 13150-token request forces VRAM pressure the revisit went from `sunk=0 evictions=1 cached_tokens=0` TTFT **94369 ms** to `evictions=0 cached_tokens=125440` TTFT **453 ms** (208×), with byte-identical generated text and zero validation errors. The pressure leg pays for it (6420→6875 ms, one extra sink DMA) and cold prefill is unchanged (95395/95910 ms). `paged KV prefix cache status` gains `tail_sinks`, counting only the sinks that actually kept a prefix behind. Re-measured on Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64 with one release binary (sha `a4122b2e…`), ctx 65536, `-u 5120`, page 64, Q8_0 KV, greedy, `--parallel 1`, elastic corridor, hd256 flash forced off, yield off in both arms, only `kv.paged_cache_max` differing (16GiB vs unset): 3 cold 21649-token conversations plus a fresh victim need 1354 frames against the 1024-frame corridor ceiling, so the exhaustion ladder must fire. With the RAM tier the tree sank 338 frames (0.36 GiB) and kept all four entries (`evictions=0`), and the verbatim revisit of the first conversation attached at `cached_tokens=21632` (99.9%) in **872 ms** (`prefill_ms=415`); without it the tree evicted twice (`entries=3`, `evictions=2`) and the same revisit re-prefilled from zero in **14252 ms** (`prefill_ms=13679`) — **16.3×**. The second revisit is identical in both arms (583 / 591 ms) because the OFF arm had by then re-cached what it destroyed. Every generated-text sha is byte-identical across both arms, `aux_refusals=0`, zero VVL/ERROR/panic. Two confounds were ruled out by measurement rather than assumed: the corridor-only tree caps aux at 512 MiB instead of pool/8, but the OFF arm used only 337.71 MiB, so no entry disappeared for aux reasons; and dropping the 16 GiB pool declaration left the host store at the same 47.17 GiB with the same `expert_cache_target=4.15 GiB`, because that model's routed experts already fit, so the OFF arm bought no expert residency with the freed envelope. Cold prefill is unchanged (1557 vs 1571 t/s mean over the four fills, −0.9%). The revisit needed no promotion DMA (`promoted=0`): the tier absorbed the pressure by sinking a DIFFERENT conversation, which is the point — the entry a client comes back to is not the one that pays.
- Let the paged-KV tree yield cold tails to a STARVING expert cache, behind `kv.paged_yield_under_pressure` (`INFR_KV_PAGED_YIELD_UNDER_PRESSURE=1`, default off). The expert cache and the paged tree sub-allocate ONE unified VRAM arena but each owner only sees its own ceiling, so a tree below its frame limit squats cold pages in device memory while the expert pager evicts on every admission and the RAM frame pool sits empty. With the knob on, `paged_publish_detach` samples the expert miss rate once per request boundary (before `release_idle_corridors`) and, above 3%, sinks cold conversation tails to the RAM frame pool so the emptied whole corridor segments go back to the experts in that same boundary. The window is "since the last sample", not a lifetime rate, or a model warmed an hour ago would make the tree yield forever. The last attach reuse depth stays a hard floor, and the sink is bounded to the frames this conversation owns INSIDE the target segment — `sink_entry_tail` moves a whole VRAM run, so without that floor a generous host tier also carries out the base chunk that is never returned, paying a DMA out and back for nothing. That floor has to be converted into the entry's OWN chain index: a segment's first frame is a GLOBAL pool slot while `keep` indexes the entry's chain, and the two coincide only for a single conversation owning the corridor from slot 0; comparing them raw makes every shorter chain satisfy `keep >= end` and skips it, so the whole feature silently no-ops in exactly the multi-session case it exists for. Requires `kv.paged_elastic` plus a RAM frame pool. The pressure signal itself needed a fix to be readable at all: `kernels.vulkan.decode_gpu_resident` is on by default and with MTP off the GPU owns decode residency, so the host LRU counters only record what the owner hands back — and the owner continues on-device through a fully-hit layer, pausing at the first miss, which biases the host-only miss rate systematically HIGH and would sink cold tails while the experts are comfortable. It now reads the existing `owned_statistics` drained-boundary transport under the same `with_unified_gate` as `free_kv_frames`, reports "no pressure" on a readback failure rather than guessing, and costs one device-LRU readback per boundary only when the knob is on. Measured same-window, same release binary (sha `e6249000…`), Qwen3.6-35B-A3B, ctx 65536, `-u 2048`, page 64, Q8_0 KV, greedy, `--parallel 1`, elastic corridor + 16 GiB RAM tier, only this knob differing: at the shipping `expert_cache_target=13.60 GiB` the windowed miss rate runs 7.35% → 2.64% → 0%, so the 3% gate correctly never fires and the knob is a no-op there; `paging.cache=4GiB` manufactures sustained pressure (~16% expert miss). Over 5 cold 14545-token conversations + 2 fresh 14578-token requests generating 1024 tokens each + 1 verbatim revisit, the ON arm yielded 7 times (`corridor_shrinks=13`) and ended holding 227/256 frames (0.15 GiB) against OFF's 940/1024 (0.61 GiB) — **0.46 GiB of corridor really went back**: expert resident slots 5359/3825 vs 4945/3550 (+8.4%/+7.7%) and evictions 166722/36533 vs 189197/41071 (−11.9%/−11.0%). The two long decodes ran 110.2/109.0 t/s vs 106.0/104.9 (**+4.0%/+3.9%**), `decode_ms` 9291/9398 vs 9661/9763, those requests' `total_ms` 17416/17628 vs 17781/17915 (−2.1%/−1.6%) and the whole sequence 76602 vs 77147 ms (−0.7%). Both arms report `cached_tokens=14528` (99.9%), byte-identical generated-text shas, `evictions=0 aux_refusals=0` and zero VVL/ERROR/panic. The gain lives in the DECODE leg: the same pressure with 24-token generations (97% prefill) sums +0.55% the other way, because the revisit's DMA back becomes the only visible difference.
- Sized the automatic host-memory envelope against the Windows **commit** ceiling and the declared paged-KV frame pool, instead of available physical RAM alone. `device.auto_profile` froze `device.ram_budget` from `available RAM - 3 GiB` (conservative) / `total RAM - 13 GiB` (aggressive), but WDDM charges a process's device-local allocations to the same commit limit the store must live under, and `kv.paged_cache_max` promises host RAM the expert store had also been counting. Measured on a 96 GiB box with a 9 GiB system-managed page file (commit ceiling 104.6 GB, ~20 GB held by the desktop): `Qwen3.8-Flash-Next-UD-Q4_K_XL` asked for a `full layer-contiguous RAM store` of **71.73 GiB** while the engine's own post-Vulkan ceiling said **61.1 GiB** — it printed that discrepancy as a warning, kept the number, and the memory circuit breaker ended the process 16 s later at **5 MiB** of commit headroom (exit 87). Both automatic shapes are now clipped by that ceiling, the envelope is the smaller of the physical axis and `commit headroom - 20 GiB` (conservative) / `- 10 GiB` (aggressive), the store takes that envelope whole, and the tree clamps its own frame count at session init against free PHYSICAL RAM (1 GiB kept unspendable), evicting past that instead of tripping — the first version pre-reserved the declared pool inside the store's budget and cost Qwen3.8-Flash-Next 4.27bpw its Full 47.17 GiB host store over 4 GiB, leaving the runtime SSD tier enabled for nothing. Only an explicit `device.ram_budget` (or legacy `paging.dram*`) is still honored verbatim, and it says so. Platforms without a commit probe (Linux/macOS) keep the established sizing byte-for-byte. The launch wizard states the same relationship: the paged frame-pool cap comes out of the automatic host budget on Windows, and a typed-in RAM budget is honored verbatim with the circuit breaker as the backstop.
- Made the Qwen3.8 MTP gate judge what drafting costs. It now accepts a GREEDY REQUEST even when the server default is not greedy — the old gate demanded both and silently ran ordinary decode for every `temperature: 0` request — and the new `spec.mtp_min_alpha` (`INFR_SPEC_MTP_MIN_ALPHA`, default 0.9) makes the NEXT request use ordinary decode when the driver's rolling draft acceptance sits below it. A VERIFY cycle costs several plain decode steps, so on a 7900 XTX with the shared Q4_K_M head the measured acceptance (alpha ≈ 0.43, verify 94% of the cycle) made MTP run at 29 t/s against 46 plain. When the gate refuses after MTP has already run it now releases the detached backend and sidecar KV, which otherwise left the ordinary path with 0 GiB of VRAM room and failed every following request, and it warns once that the Qwen3.8 sidecar driver is not gated by the parked `mtp_enabled()` switch.
- Removed the launch wizard's "force the fused hd256 flash kernel on amdvlk" switch. The engine ships the non-unrolled build and enables the kernel by default, so the prompt only carried the pre-fix crash warning; a driver that still faults is served by `INFR_AMDVLK_HD256_FLASH=0` or `--set kernels.vulkan.amdvlk_hd256_flash=false`.
- Routed narrow-output (n not a multiple of 64, so no tiled GEMM tile exists) Prefill Linears through a one-shot f32 expansion plus the batched f32r GEMV instead of the native per-row GEMV, which re-read the weight once per row. Qwen3.8-Flash-Next base at m=4096: `2560x48` 1.5 ms → 195 µs and `10240x4` 969 µs → 275 µs per dispatch, i.e. 202 ms → 42 ms of the pp4096 device time including the 11 µs casts; the Q8_0/BF16 expansion is exact (fits f32), and only weights up to 128 Ki elements are covered.
- Extended the BM128 (128x128 single-buffer) Prefill warptile from five hard-coded Q8_0 shapes to every native-dense format, still only on measured Windows RX 7900 XTX devices. The gate is now `m >= 1024` with whole 128-row tiles and wide k/n (`dense_bm128_applies`); short-K / narrow-N shapes and the m=512 Flash tuples that lost in isolated BAAB keep the BM64 tile. Qwen3.8-Flash-Next base pp30000 1720.3 → 1740.5 t/s and pp4096 +2.0% on/off with the same binary; m=1 decode cannot reach the tile.
- Raised the automatic prefill chunk (`ubatch`) on discrete GPUs from 4096 to 5120 rows. Ubatch is the FIRST call on the unified VRAM arena — its runtime reserve comes off the top and the expert cache is the remainder — so the default now follows the measured knee: `kernel-bench --sweep-m` puts the per-shape TF knee at m≈4096, and Qwen3.8-Flash-Next pp30000 with the paged KV tree measured 1110.9 / 1591.3 / 1712.2 / 1728.4 t/s at ub 2048 / 4096 / 5120 / 6144. 5120 is the tallest rung that still holds all four prefill lanes; a GPU too small for its reserve walks the existing fallback ladder down instead of failing.
- Accepted `instructions` or a system/developer message on a chained Responses request when it is byte-identical to the system prefix already stored for that chain, instead of rejecting every resend.
- Honoured `parallel_tool_calls=false` on `/v1/responses` by returning at most one call per turn and naming the withheld calls in `warnings`.
- Raised the non-streaming `/v1/responses` translation ceiling from 16 MiB to 64 MiB and made an oversized upstream reply a named 502 instead of a bare failure.

- Routed the MoE router (`ffn_gate_inp`) at prefill rows through the same tiled cooperative-matrix GEMM a dense `Op::Linear` of that shape takes, instead of the per-row GEMV that re-reads the whole router weight once per token, and added the missing bf16 A_GLOBAL warptiles so the tier is reachable for bf16 weights (the BN=64 twin covers the Qwen3.8 hyper-connection widths, `10240x320` / `320x10240`, which used to fall back to the legacy 64x64 bf16 GEMM). Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S (bf16 router + bf16 HC), paired `pp30000 --ctx 32768`: 1222.4 / 1217.5 → **1276.0 / 1272.4 t/s (+4.4%)**. Models whose router ships F32 and whose HC tensors are not bf16 keep the unchanged path — Qwen3.8-Flash-Next base measured 1565.8 → 1568.4 t/s (+0.2%, within noise). The logits buffer is padded to whole 64-row tiles only when the GEMM tier is chosen, so the GEMV route stays byte-identical.

### Fixed

- Kept the frontier snapshot a continuation replaces, so a sub-agent fork of the previous turn can still resume. `SeamKv::paged_frontier` was a single host snapshot that every prefill overwrote, and a continuation stays in its slot (`continuation_prefix != 0`, so nothing is published or detached) — which means the boundary the parent's PREVIOUS turn ended at, exactly the depth a fork branches from, was destroyed by the next turn's own capture. Ancestor inheritance could not cover this: it relocates snapshots between tree ENTRIES, and a continuation never creates one, so the branch point was gone before any publish could inherit it. The slot now keeps the replaced frontier beside the current one, bounded at one extra snapshot like the two device turn checkpoints; `collect_paged_ckpts` hands both to the publish, and a re-capture at the SAME boundary keeps the older snapshot instead of paying aux bytes for a state the tree would dedup anyway. Both leave together on attach, on detach and on every token-spine reset. Measured same-window, two release binaries differing only by this fix (sha `fe8125ec…` → `a4122b2e…`), Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64, ctx 65536, `-u 5120`, page 64, Q8_0 KV, greedy, `--parallel 1`, elastic corridor + 16 GiB RAM tier, hd256 flash forced off, yield off in both arms, a 200-segment shared document and a parent that advances one turn before two sub-agent forks branch off it: the fork at the parent's FIRST frontier went `cached_tokens=0` / prefill **2808 ms** / total **9349 ms** to `cached_tokens=3840` (99.3% of its 3868-token prompt) / prefill **491 ms** (5.7×) / total **6976 ms** (−25%), with `uncached_tokens` 3868 → 28. The other four requests are unchanged (0 / 4337 / 3840 / 4352 cached) and all five generated-text shas are byte-identical across both arms. The log names the mechanism: `publish … snapshots=[4352]` followed by `hit matched=3840 resumable=0` becomes `collected=[3840, 4352]` → `snapshots=[3840, 4352]` → `resumable=3840`. The cost is one extra snapshot per branched conversation — on this 48-layer hybrid each is 112.57 MiB, so aux went 225.14 → 562.85 MiB (roughly 9 concurrent conversations instead of 18 at the 2 GiB cap), and under aux pressure the shallowest snapshot is still yielded first. The fork attaching instead of replaying also left the corridor smaller: one growth to 143/256 frames before, none at all and 89/128 after. `evictions=0 aux_refusals=0` and zero VVL/ERROR/panic in both arms. Re-measured the four-arm 35B regression with the same two binaries (Qwen3.6-35B-A3B-APEX-I-Balanced, ctx 65536, `-u 2048`, dead 600 segments / live 604): `cached_tokens` 14336 (99.2%) and 14528 (99.9%) unchanged, revisits 1212/1213 → 1212/1217 ms and 395/290 → 292/302 ms, aux 376.88 MiB in ALL four arms (that workload has no continuation-then-fork, so the extra snapshot never fires), `entries=6 hits=7 misses=1 evictions=0 aux_refusals=0`, every sha identical, zero validation errors; the live pair's steady-state prefill is 1860.2 → 1858.7 t/s (−0.1%, noise), while the dead-old arm caught a faster clock window (2084 t/s) that all three later arms left (1894 / 1860 / 1859), so that one pair is not a quotable perf comparison. Only the last TWO frontiers are kept: a fork branching deeper than the previous turn still depends on the device turn checkpoints and on published-ancestor inheritance. `RUST_LOG=infr_llama::seam=debug` now logs every frontier capture with the boundary it replaced and every detach's collected snapshot boundaries, beside the existing `infr_llama::paged_cache=debug` publish/hit/miss lines.
- Captured the paged-KV frontier snapshot at the deepest chunk end at or below the frontier, instead of only when a chunk landed on it exactly. `prefill_chunk_end` skips a boundary that would starve the chunk below the layer-ring threshold (the 2026-10-19 memory-bomb fix), and it applies that filter to the paged frontier too — so on a prompt whose frontier leaves a short tail, NO chunk ends on the frontier. Both snapshot gates demanded exact equality, and a single-sequence request has no turn boundaries to fall back on, so `collect_paged_ckpts` returned empty and `publish_with_ckpts` refused the whole recurrent entry: the conversation never entered the tree and every later turn re-prefilled from zero. The frontier is still subject to the ring threshold (exempting it would cut a tiny chunk and storm-miss experts through the emptied corridor); the capture simply moves back to the previous chunk end, which is page-aligned like the frontier and describes the same tokens, just fewer of them. Measured four arms in one window, two release binaries differing only by this fix, Qwen3.6-35B-A3B-APEX-I-Balanced, ctx 65536, `-u 2048`, page 64, Q8_0 KV, greedy, `--parallel 1`, elastic corridor + 16 GiB RAM tier, 5 cold conversations + 1 victim + 2 verbatim revisits. The DEAD geometry (600 segments, a 14449-token prompt whose frontier 14400 leaves 64 rows) went from `aux_refusals=7`, `entries=0`, `cached_tokens=0` and revisits of **7769 / 7787 ms** to `aux_refusals=0`, `cached_tokens=14336` (99.2%) and **1309 / 1216 ms** — 5.9× / 6.4×. The LIVE geometry (604 segments, 14545 tokens, frontier 14528 leaves 192 rows) is unchanged: both arms `cached_tokens=14528` (99.9%), revisits 404 / 306 ms before against 427 / 308 ms after. Every request's generated-text sha is byte-identical across all four arms, with zero VVL/ERROR/panic.
- Kept a sub-agent fork's branch point resumable instead of re-prefilling it from zero. Two defects compounded, and neither fix alone changes what a client sees. (1) A deeper continuation publish superseded its ancestor entry and deleted the only snapshot that could resume the depth a fork branches at; a continuation captures no new snapshot AT its own start (the runner's `boundary > start` filter), so the branch point was simply gone. The publish now inherits each superseded ancestor's DEEPEST snapshot — relocated, not duplicated, and a boundary this publish captured itself wins — so aux moves between entries instead of growing with ancestry. (2) `best_match` scanned a node's subtree only when the prompt was FULLY consumed, so a fork diverging inside an edge (or stopping at a node with no matching child) was offered only the entry-less node the split created and missed, while the frames it needed sat one node deeper. The scan now runs from the walk's frontier in every case, and candidates are ranked by the depth actually resumable: a recurrent entry whose snapshots all sit above the branch point matches and then hands back zero tokens, so the longest chain is not the best candidate. Measured three arms, same window, same release profile, Qwen3.6-35B-A3B-APEX-I-Balanced, ctx 131072, `-u 2048`, page 64, Q8_0 KV, greedy, `--parallel 1`, elastic corridor + 16 GiB RAM tier, a 200-segment shared document and a parent conversation that advances one turn before two sub-agent forks branch off it: the fork at the parent's FIRST frontier went `cached_tokens=0` / **2094 ms** with both fixes off AND with inheritance alone, to `cached_tokens=3840` (99.3% of its 3868-token prompt) / **210 ms** with both — 2045 → 158 ms on the warm second pass, where the snapshot it attaches is the inherited one. Tree misses 3 → 1 (the remaining one is the genuinely cold first request), hits 7 → 9, `ckpts_inherited=2`, entries and held frames unchanged (3 / 83), `evictions=0 aux_refusals=0`, zero validation errors, and every request's generated-text sha byte-identical across all three arms. The cost is one extra snapshot per branched conversation: aux 188.44 → 251.25 MiB (+33%, roughly 8 concurrent conversations instead of 10 at the 2 GiB cap), and under aux pressure the shallowest snapshot is still yielded first, so the branch point is the first thing sacrificed when the budget is tight. `RUST_LOG=infr_llama::paged_cache=debug` now logs every publish, hit and miss with the node, chain length and snapshot boundaries, so a low reuse rate can be attributed from the log instead of guessed at.
- Restored decode command-ahead on the paged-KV path. Command-ahead uses `committed()` to prove the next token's KV rows already exist, but paged frames are shared with the radix tree and never appear in `segments`, so `committed()` was permanently 0 and the optimization was silently switched off — paged decode was not slow because of page-table lookups, it was slow because every step waited on an idle GPU. The page table now keeps a `table_pages` high-water mark and `committed()` returns the larger of it and `segments`; `truncate` / `reset` lower it in step so the mark never claims pages already returned. Paired same-window A/B at page 64: decode 965 → 912 ms, 133.0 → 140.8 t/s. Page sizes of 256 and above never paid the tax.
- Replayed a `reasoning` input item from either wire shape: the `summary[]` this server emits, or the raw `content[].reasoning_text` that pi-ai and DeepSeek Harness keep and send back INSTEAD (they drop the summary). That shape used to fail the second turn with `400 reasoning item requires readable summary`; an item carrying neither text nor a store reference is now dropped instead of failing the turn.
- Merged each assistant turn's reasoning, text and parallel `function_call` items back into ONE chat message on the way into the prompt. The Qwen3.8 template renders one `<|im_start|>assistant … <|im_end|>` block per message, so the old per-item split turned each real turn into three assistant turns, several of them "thought, then ended" — a pattern the model imitated by answering with an empty turn instead of calling the next tool.
- Reported a request that outgrows the session KV window as `400 invalid_request_error` ("prompt 33676 + gen 0 exceeds the session KV capacity 32768") instead of a `500 server_error` that reads as a server fault; a client that drops the stream mid-generation already cancelled the decode and still does.
- Accepted OpenAI's `reasoning.effort: "none"` on `/v1/responses` and `reasoning_effort: "none"` on `/v1/chat/completions`: it now turns thinking off for that request instead of failing with a 400 `unsupported reasoning effort` (DeepSeek Harness sends exactly this as soon as its thinking toggle is off). For Qwen3.8 models — whose template validates the level itself and accepts only low/medium/xhigh — `high` and `max` now snap to `xhigh` instead of failing the render with "Unexpected reasoning effort high", `minimal` keeps mapping to `low`, and a truly unknown level still 400s, now listing the accepted values.
- Sent the bf16 MoE router (`ffn_gate_inp`) and bf16 hyper-connection projections at prefill rows through the tiled cooperative-matrix GEMM instead of the per-row streamed GEMV / the 64x64 fallback: Bf16 was missing from the A_GLOBAL warptile tables, so 10240x320 projections ran at 2.6 ms per dispatch where the narrow-n tile takes 0.85 ms, and a 2560x512 router took 15.5 ms per layer. Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S pp30000 1215.2 → 1663.0 t/s (+36.8%); models whose router ships as F32 and whose HC tensors are not bf16 take the unchanged path.
- Made a FORCED `tool_choice` (`"required"` or a named function) stream its call progressively like an automatic one: the grammar-constrained body now goes through the same splitter, so the function name and each completed argument member arrive while the call is still generating instead of as one frame at the end. A constrained call that cannot be completed after its name has already been streamed (for example `max_tokens` was reached) now fails loudly instead of silently retrying unconstrained and appending a second call's arguments to the first.
- Made the hd256 fused flash-attention prefill kernel compile on the AMD proprietary driver instead of crashing it. `attn_flash_warp_hd256_bm16` faulted LLPC inside `vkCreateComputePipelines` (`0xc0000005` at `amdvlk64.dll+0x2b65705`) whenever it was handed the shader's FULLY-UNROLLED IR — reproducible on amdvlk 26.10.1 as the first full-attention layer of a ragged 4K prefill (4001 query rows). `[[unroll]]` is now a build variant: the kernel ships as the unrolled default plus a `-DNO_UNROLL` twin, and the recorder selects the twin on `VK_DRIVER_ID_AMD_PROPRIETARY` only, so RADV and every other driver keep the unrolled IR and the math and dispatch shape are identical in both. The tier is therefore enabled on amdvlk; `INFR_AMDVLK_HD256_FLASH=0` (`--set kernels.vulkan.amdvlk_hd256_flash=false`) still drops hd256 prefill back to the coopmat `attn_qk`/`softmax`/`attn_pv` tier for a driver that faults anyway. Measured on Qwen3.6-35B at 26.10.1: paired 4001-row runs 3824.4 vs 3767.4 t/s for non-FA, and in a fast-clock window 4001 / 8000 / 12001 rows ran 4643 / 4532 / 4062 against 3834 / 3646 / 3703 (+21% / +24% / +10%).

## [1.0.2] - 2026-10-15

### Added

- Added an experimental Apple Silicon release package with a system-zsh Metal launcher for text run, single-slot serve, benchmark, and explicit MTP workflows; it has no Homebrew, Python, jq, or Rust runtime dependency.

### Fixed

- Made Windows managed-package updates stream progress, relaunch the replaced wizard, verify the reported engine version, and restore the previous installation if the new launcher fails.
- Made Linux managed-package updates validate the replacement launcher and restart directly into the updated version after releasing updater resources.

## [1.0.1] - 2026-10-15

### Changed

- Moved additive repeat, frequency, and presence penalties onto Vulkan where supported, avoiding the CPU logit round trip while retaining the existing fallback path.
- Removed the Agent distribution and bundled plugin sources from the Engine repository; they are now maintained independently from the backend release.

### Fixed

- Unified resident Decode handling through eight simultaneous rows and reserved head-projection scratch before the Vulkan arena is frozen, preventing multi-slot Qwen3.6 35B allocation and execution regressions.
- Routed raw Vulkan timeline-label controls through the resolved profiler configuration so per-backend settings no longer depend on process-global cached environment state.
- Applied a single startup RAM budget to unified-memory GPUs, preventing the device arena and a duplicate host expert cache from consuming the same physical memory twice.

## [1.0.0] - 2026-10-15

### Added

- The Windows and Linux launch wizards provide a package-local download menu for selecting one or more Qwen3.6 35B, Qwen3.8 Flash-Next, Vision, and Embedding GGUF bundles. Downloads show progress, retry transient failures, resume from `.part` files, and return to local model selection without choosing a model automatically.
- GPU-resident paged-MoE Decode scheduling, later generalized to resident MoE layouts and Qwen3.5/Qwen3.6 layer-synchronous batched Decode.
- Native Qwen3.5/Qwen3.6 multimodal serving and model-aware reasoning-effort controls.
- Release benchmark and profiling coverage for resident hit/miss handoff, queue overlap, MTP cycles, and 30K/150K concurrency.

### Changed

- Removed MTP selection from both release wizards. MTP remains available through direct engine CLI/configuration, and the Linux launcher still accepts an explicit `--mtp`; ordinary interactive wizard launches use the ordinary Decode path.
- Synchronized the Linux release wizard with the post-0.10.0 Windows flow without adding Python or jq dependencies. Model downloads use curl or wget and can install curl through common Linux package managers after explicit confirmation when neither tool exists.
- Separated model downloads from Quick Start and added a same-host, methodology-qualified Strata v0.1.44.1 comparison to both READMEs.
- Resident paging is enabled by default on supported Decode paths; full-graph scheduling and queued miss retirement reduce host-side layer bubbles.
- Q8 long-context attention, very-wide Q8 projections, exact Flash-Next Decode kernels, and four-token MTP verification received targeted Vulkan optimization.
- The 1.0.0 release reference reports single-, dual-, and triple-stream 30K/150K throughput for Flash-Next and Qwen3.6 35B.

### Fixed

- Preserved frozen automatic RAM budgets, removed duplicate UMA host-cache allocation, and kept paged-MoE Prefill resident instead of rereading experts from disk.
- Applied Qwen3.6 anti-repeat sampling defaults and aligned suffix checkpoints in mixed parallel execution.
- Rejected stale numeric session-cache state, removed launch-time wizard probe stalls, and preserved pager cleanup after the main synchronization point.

## [0.10.0] - 2026-10-09

### Added

- Experimental CPU expert-miss computation with bounded worker pools and wizard controls for miss count and compute cores.
- Simplified Windows wizard setup with reusable saved settings, local SSD K/V defaults, and per-slot plus aggregate live throughput reporting.
- Experimental Linux release packages and an aligned single-shell launch wizard; native Vision decoding gained bounded HEIC and AVIF support.

### Changed

- Qwen3.8 Prefill gained fused QSA, KV reuse across GQA heads, scratch reuse, dynamic live-lane planning, and overlapped expert uploads.
- Decode and MTP paths gained faster sigmoid routing, fused DeltaNet/HC operations, cached quantization tables, and tighter router-to-hit handoff.
- Automatic GPU profiles and current-transaction KV/Prefill budgeting were rebalanced for larger useful ubatch and lane counts when memory permits.

### Fixed

- Smoothed chunked-Prefill progress, raised the API request-body limit to 64 MiB, and allowed idle SSD K/V writes to yield the inference gate.
- Preserved auxiliary Vulkan workspaces and serialized state transfers, corrected mixed-batch sampling suffix projection, and improved host-allocation failure diagnostics.

## [0.9.0] - 2026-09-28

### Added

- Stateless `POST /v1/responses` with JSON and typed SSE output. It rejects server-side conversation state, remote image URLs, and unsupported hosted-tool fields.
- Qwen3.8 MTP verification and batched decode across two slots, including multimodal serving and coexistence with the Vision and Embedding services.
- Ling 3.0 Tiny Q4_K_M support and a transactional Windows Engine self-update path in the wizard.

### Changed

- Session continuation can reuse Agent and edited-turn recurrent checkpoints. Concurrent MTP shares staging resources and grows head KV lazily.
- Streaming requests send keep-alives while queued or waiting for long-running generation.
- The wizard exposes two-slot MTP serving, local model recommendations, and local cache controls.

### Fixed

- Protected frozen expert LUT slots until their GPU command streams finish; automatic MoE arena placement can retry at a lower ubatch.
- Preserved reasoning and MTP state across supported session continuations, and bounded MTP generation by remaining long-context capacity.
- Corrected Vulkan residency reporting and several ragged GEMV and split-K paths; removed the former fixed Embedding request deadline.

## [0.8.0] - 2026-09-25

### Added

- Fixed four-token Qwen3.8 MTP drafting with batched target verification and accepted-prefix state restoration.
- Native Embedding support and MTP selection in the Windows wizard; automatic modes expose context sizing.

### Changed

- Automatic RAM budgets are calculated once at startup. Conservative and aggressive Prefill ubatch candidates begin at 2048 and 4096 rows on discrete GPUs, subject to placement fallback.
- Qwen3.8 Decode, parallel QSA/PLE, MTP verification, and prompt priming received targeted performance work.

### Fixed

- Parallel Prefill honors an explicitly selected ubatch and avoids impossible scratch reservations.

## [0.7.0] - 2026-09-20

### Highlights

- Qwen3.8-Flash-Next now supports native Vulkan vision inference with GGUF multimodal
  projectors and OpenAI-compatible image content parts. Repeated turns can reuse decoded image
  embeddings and resident multimodal K/V state.
- `infr serve --parallel` now schedules mixed prefill and decode work across independent K/V
  slots, batches compatible Qwen3.8 decode rows, and reports live per-slot progress and speed.
- Idle dynamic Q8 K/V sessions can be stored in a checksummed SSD cache and restored after a
  restart. The release defaults retain up to 5 GiB for 24 hours and flush resident sessions during
  orderly shutdown.

### Changed

- Qwen3.8 segmented QSA prefill uses online softmax and hoisted segment addressing, substantially
  improving long-context prefill without changing the model or K/V format.
- Batched PLE gathers and GPU sampling reduce CPU synchronization in parallel generation.
- Chat and OpenAI-compatible serving expose configurable reasoning controls while preserving each
  model's default behavior.

### Fixed

- Vulkan shared command-pool access is synchronized across concurrent inference and embedding
  work.
- Cached prefill steps are no longer double-counted, and live status reports actual prefill
  progress instead of remaining at `Starting`.
- Text-only continuation after a Qwen3.8 image turn uses the normal text RoPE path, restoring
  expected decode performance.
- Session-cache restore validates conversation identity and capacity before reusing persisted K/V
  state.

### Validation

- Qwen3.8 long-context prefill reached 750.7 tok/s for `pp3072` at 131K synthetic depth with Q8
  K/V on the RX 7900 XTX validation host.
- Staggered 32K + 8K two-request Qwen3.8 decode sustained 56.9 aggregate tok/s with 24 GiB VRAM,
  a 48 GiB process-RAM target, Q8 K/V, and `ubatch=3072`.

## [0.6.0-beta.1] - 2026-09-02

### Highlights

- Vulkan device arenas are no longer tied to mapped Resizable BAR memory. Startup selects either
  direct mapped writes or ordinary device-local VRAM with staged uploads, then freezes that
  transfer route for the session. This adds a compatibility fallback for GPUs and drivers with a
  small host-visible device-local heap, including the RDNA2 class reported by users.
- MoE experts now act as reclaimable filler inside one elastic VRAM layout. Segmented Q8 K/V,
  Prefill lanes, graph runtime memory, and optional embedding workloads claim explicit corridors
  transactionally; Decode safety floors remain protected while phase-exclusive Prefill can borrow
  and restore them.
- `infr serve` can host an optional embedding GGUF through the same VRAM budget. Embedding weights
  load lazily from SSD for `/v1/embeddings` requests and may be released after an idle timeout.

### Changed

- Paged MoE Decode promotions use one unified U/G/D transfer path, with hardware capability and
  source location resolved by the session transfer plan rather than scattered backend branches.
- Dynamic segmented K/V reserves frozen physical coordinates up front, while uncommitted ranges
  remain available to the expert cache until each 32K context boundary is crossed.

### Fixed

- Qwen3.8 multi-turn continuation now checkpoints and restores both per-layer DeltaNet state and
  model-level PLE history, avoiding full-context recomputation and recurrent-state drift.
- Manual `device.ram_budget` accounts for fixed model pages that have not yet entered the process
  working set when the host expert cache is planned, preventing startup paging under an explicit
  total-process limit.
- Elastic ownership changes preserve every Decode dispatch floor and bounded-cache exchange spare,
  avoid fragmented late K/V growth, and restore borrowed slots correctly across repeated
  Prefill/Decode transitions.

### Validation

- The Windows long-context resource matrix covers Qwen3.6 35B and Qwen3.8 IQ4_XS under simulated
  16 GiB VRAM + 32 GiB RAM and 24 GiB VRAM + 64 GiB RAM profiles, with automatic and manual
  budgets, repeated 32K/64K/96K K/V growth, API continuation, and a CLI smoke test.

## [0.5.2] - 2026-09-01

### Fixed

- Long-context Q8 prefill now retires superseded Vulkan dequantization scratch capacities after
  each execute, preventing historical context depths from exhausting unified VRAM.
- Vulkan buffer binding, one-shot commands, staging-ring construction, and recorder error paths
  now release their transient allocations and raw objects on failure or unwind.
- Manual `device.ram_budget` planning keeps room for process objects created after host-cache
  sizing, while release validation measures private resident RAM separately from reclaimable GGUF
  file pages.

### Validation

- The Windows release resource matrix passed all nine automatic/manual cases for Qwen3.6 35B and
  Qwen3.8 IQ4_XS, including repeated dynamic K/V growth through 32K, 64K, and 96K boundaries.

## [0.5.1] - 2026-08-31

### Fixed

- Vulkan MoE prefill now reserves its complete-layer lanes before phase scratch allocations.
  This prevents unified-VRAM fragmentation from causing long-context prefill allocation failures,
  especially while elastic K/V caches grow across segment boundaries.
- Elastic K/V and runtime allocations now preserve each MoE pool's widest-dispatch residency
  floor and budget a separate bounded-RAM exchange slot, preventing multi-turn inference from
  exhausting the pager's within-batch working set when the first dynamic K/V segment is committed.

## [0.5.0] - 2026-08-30

### Highlights

- **Qwen KV state now grows in 32K-token segments on Vulkan.** Compatible Qwen3.5/Qwen3.6 and
  Qwen3.8-Flash-Next sessions allocate Q8 K/V capacity as context crosses each segment boundary
  instead of reserving the complete configured window at startup. Qwen3.8's QSA index-key state
  follows the same elastic allocation model.
- **Compatible Vulkan sessions now default to Q8_0 K/V.** Explicit `kv.type_k` and `kv.type_v`
  choices remain authoritative; layouts that cannot use Q8_0 continue to fall back to F16.

### Changed

- Chunk-major prefill is again the production default for resident and streamed models.
  Layer-major execution remains available only through the explicit `paging.layer_major=true`
  diagnostic override.
- Memory planning, runtime allocation, cache reporting, synthetic-depth setup, and session cloning
  now resolve the same effective Vulkan K/V format and segmented layout.

### Fixed

- Qwen3.8 QSA planes are allocated from their segmented arena instead of retaining an unintended
  flat context-sized allocation.
- Automatic K/V budgeting no longer prices F16 while the Vulkan runner actually allocates Q8_0,
  preventing inconsistent context limits and VRAM accounting.
- Segmented K/V address lowering now handles reads, writes, Q8 dequantization, attention, and QSA
  gather/index operations across 32K boundaries.

### Known limitations

- Elastic segmented growth currently applies to coupled Q8_0 K/V on supported Qwen Vulkan paths.
  Explicit F16 or mixed K/V formats retain the established flat-allocation behavior.

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
  dynamic twins. See `documentation/evidence/changes/models/deepseek-family.md` § Stage 4.
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
  bit-identical. See `documentation/evidence/changes/models/deepseek-family.md` § Stage 4.
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
  host beyond that. See `documentation/evidence/changes/models/deepseek-family.md` § "Sinkhorn hyper-connections".
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
  `documentation/evidence/changes/models/deepseek-family.md` § Stage 2.

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
  `deepseek-coder`, and `deepseek-v3` pre-types (see `documentation/evidence/changes/models/deepseek-family.md` § Stage
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
    faults, prefill 3-7.5% slower (`documentation/evidence/benchmarks/2026-08-03-validated-models.md`).
  - **Vulkan backend**: a third tier under both dense weight streaming and the
    paged MoE expert cache, so a VRAM miss resolves against the arena and
    reaches the file only when that misses too. MoE pages ONE EXPERT at a time
    rather than a whole bank. A block the arena has no room for is read straight
    into the staging ring instead of evicting one, so the streaming majority
    costs one copy rather than two. Measured on a memory-capped Qwen3-14B Q8_0
    under a forced 2 GB VRAM budget: **decode 2.17x faster than the mmap path it
    replaces** at an 8 GB cap with a 7 GB arena (1.41x with a 3 GB one), 38x
    fewer major faults, 232 → 110 GB read, and parity when memory is plentiful
    (`documentation/evidence/benchmarks/2026-08-03-validated-models.md`). The arena budget is the dominant factor — 3 GB → 7
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
