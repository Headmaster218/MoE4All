# infr-metal — Apple GPU backend architecture

State as of 2026-07-05 (the decode-parity campaign + spec decode, multi-slot
serve, native-read KV, MTP, and the replay-tape correctness fix all landed).
`crates/infr-metal` implements the same `Backend` seam as `infr-cpu` and
`infr-vulkan` on Apple's Metal API. It started as a correctness-first reference
(per-op command buffers, dequant-to-f32 everything) and was rebuilt
profiling-first; this doc records the architecture that emerged and the measured
reasoning behind it.

## Numbers (M3 Pro 18-core, `infr compare --sweep --dev MTL0`, same GGUFs)

| model                 | pp512 | tg128 | tg64@d4096 |
| --------------------- | ----- | ----- | ---------- |
| Qwen3-0.6B Q4_K_M     | 0.88× | 1.00× | **1.02×**  |
| Qwen3-4B Q4_K_M       | 0.92× | 1.00× | 0.97×      |
| Qwen3-8B Q4_K_M       | 0.88× | 0.99× | 0.98×      |
| Qwen3-0.6B Q8_0       | 0.85× | 0.97× | 0.98×      |
| Qwen3-MoE-2.4B Q4_K_M | 0.87× | 0.97× | 0.97×      |
| gemma3-1b Q4_K_M      | 0.82× | 0.95× | 0.95×      |
| TinyLlama-1.1B Q4_0   | 0.86× | 1.00× | 0.97×      |
| Qwen3.5-4B Q4_K_M     | 0.80× | 0.96× | 0.97×      |
| Qwen2.5-0.5B Q4_K_M   | 0.82× | 0.95× | 0.99×      |
| Qwen3-4B IQ4_XS       | 0.89× | 0.89× | 0.87×      |

The SHORT TURN shape (a tiny suffix prefill on a warm session — multi-turn serve
TTFT) is healthy too: pp4@d4096 = 0.89× (0.6B) and 1.45× (4B, ahead) — the
m=2..8 multi-row GEMV and small-m attention routing cover it.

(ratios = infr / llama.cpp; >1 = infr ahead. Sweep-internal rows run a few
percent hot — a back-to-back sweep heats the chip; only trust a SOLO re-probe
for a regression call.) **Decode is at parity.** The prefill band decomposes by
ablation (strip the dequant scale math: −2.4% of GEMM time; strip ALL weight
loads + decode: −9%) to a tile floor of ~6.9 TFLOPS — ~53% of the f16
`simdgroup_matrix` peak and about llama.cpp's own effective rate — so what
remains there is MMA/tile scheduling margin, not an addressable component.
Decode was weight-stream + launch-latency bound; the levers that closed it are
recorded below in the order the profiler surfaced them.

From the naive reference implementation: decode ~44×, prefill ~250×+. Model load
is ~10× faster than the repack era and resident weight memory is the raw mmap
only (−3 GB on 4B).

## Execution model: device residency, one command buffer per forward

`exec::Resident` tracks, per graph tensor, whether the current value lives in
the host mirror (`vals`) or a device buffer (`dev`/`loc`). GPU ops encode into
one shared command buffer (lazily opened, Metal automatic hazard tracking orders
intra-batch dependencies) and the batch is committed+waited only at a host-side
op or the final write-back. A decode forward is exactly one commit+wait; the
naive per-op barrier version spent ~90% of wall in ~75k command buffers per run.

Measured floors that shape everything else (probe tests in
`crates/infr-metal/tests/dispatch_overhead.rs`):

- dispatch + hazard barrier inside a batch: ~1.2 µs/kernel → op-count fusion is
  NOT a lever (a fused Add+RmsNorm measured nothing and was removed);
- device compute-read roofline: ~133 GB/s (spec 150);
- quantized GEMV achieved: ~90-105 GB/s (probe `tests/gemv_bw.rs`).

## Weight forms (quantized `Op::Linear`)

Chosen per dtype by `weight_qui`, cached per bound buffer for the backend's
single-generation lifetime:

- **Native GGUF blocks — Q4_K, Q6_K** (the formats real checkpoints ship): the
  bound weight buffer is used AS-IS. No host repack, no extra residency; kernels
  decode raw block bytes in-place (`DEC16_Q4K`/`DEC16_Q6K`, one 16-element
  sub-block per call; `get_scale_min_k4` ported), and the 32-element legacy
  formats Q8_0 (34 B), Q5_0 (22 B) and Q4_0 (18 B) — every format real
  checkpoints ship. ~4.5-8.5 bpw on the wire. Two hard-won details: the Q6_K
  sub-block selector must be branch-free (a 4-way `if` serializes the simdgroup
  — measured 2.3× slower), and Q6_K blocks are 210 B (2-aligned) so only
  byte/ushort loads are legal, while Q4_K's 144 B blocks allow uint loads.
- **Factored form — every other affine quant** (`infr_gguf::dequant_factored`,
  the single decoder that `dequant_unified` now expands): bit-packed 4/6/8-bit
  codes + one i16 `(sc, m)` pair per 16 elements + one f16 `(d, dmin)` pair per
  quant block, so `weight = (d·sc)·code + (dmin·m)` reproduces the reference
  dequant bit-for-bit (same f32 products; the sign/power-of-two factors folded
  into `m` are exact scalings — Q6_K's `m = −32·sc` is why i16).
- Float weights (f16/bf16/f32) dequant to a cached f32 device buffer.

## Quantized Linear kernels: one decoder, three shapes

Per-format `DEC16_*` macros decode the 16-element block with global index `bi`
into `wk[16]`; three kernel shapes instantiate each format:

| shape                             | route                                                                  | precision                                |
| --------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------- |
| `linear_*` GEMV                   | m == 1 (decode)                                                        | exact f32 (bit-for-bit dequant, f32 dot) |
| `linear_*_ks` k-split GEMV        | m == 1, row groups ≤ 4096 AND k deep enough to feed 4 simdgroups twice | exact f32, reassociated                  |
| `linear_*_rt` row-tiled           | 1 < m < 16, or out_f % 16 ≠ 0                                          | exact f32                                |
| `linear_*_cmm` cooperative GEMM   | m ≥ 16 && out_f % 64 == 0                                              | f16 operands, f32 accumulate             |
| `linear_*_hmm` per-simdgroup GEMM | m ≥ 16 && out_f % 16 == 0 (fallback)                                   | f16 operands, f32 accumulate             |

The cooperative GEMM is the mul_mm shape (ported from llama.cpp and adapted to
the DEC16 decoders): a 64-output × 32-token tile per 128-thread threadgroup,
NK=32. The load-bearing piece is the **threadgroup-memory layout**: staged
operands are contiguous 8×8 half tiles (weights written pre-transposed), so
every `simdgroup_load` is stride-8 and conflict-free — tile size, barrier count,
and decode width were each probed separately and none of them was the gap. Two
more mul_mm details each measured real wins later: the device reads + dequant
issue BEFORE the barrier that drains the previous iteration's MMA (latency-hides
behind compute, +5% 4B pp2048), and the DEC16 decoders use the reference
dequantize shapes — in-place high nibble with the /16 folded into the scale
(Q4_K), uint32-lane ql/qh combines (Q6_K) — bit-identical values, ~+4% more.
Each thread dequantizes exactly one 16-block per K-step (128 threads = the whole
weight tile) and x is cast f32→f16 inline during staging. Each simdgroup owns a
32×16 quadrant as 8 f32 accumulators; partial token tiles stage through
threadgroup memory with a row-guarded copy-out so every tile takes the MMA path.

Precision policy: **decode is algebraically identical to `dequant_block`, FP
reassociation only** (the mul_mv-shape GEMV reconstructs the exact dequant
values but reorders the f32 dot's summation; parity holds at 1e-4). Prefill GEMM
rounds operands to f16 (~5e-4 relative, well under quantization error — the
llama.cpp trade); its parity tests compare against a reference that rounds
operands identically, keeping the tolerance at 1e-3 instead of testing nothing
with a loosened bound.

## Attention routing

All scalar variants share `AttnParams` and an online-softmax core; routing is by
launch shape because the scalar kernels are latency-bound on their serial
O(kv_len) chain:

- **`attnflash2_f16kv` (hd 64/128)** — the llama.cpp `kernel_flash_attn_ext`
  structure for wide launches: NSG=4 simdgroups cooperate on ONE (8-query tile,
  head) threadgroup over C=64 KV positions per iteration, each phase split along
  a different axis (QK^T by KV columns with K^T direct from the f16 cache;
  softmax by query rows, all 32 lanes on float2 scores, M/S stats pinned in the
  owning simdgroup's registers; P·V by output columns with O fragments held in
  registers). Analytic causal/window masking, zero-padded partial query tiles.
  Templated over compile-time head_dim (hd 64/128/256 instantiations — the
  fully-unrolled loops were worth +38% alone; hd=256 is gemma, whose absence was
  a 3× prefill crater); NSG=8 benched slower.
- **`attnflash_f16kv`** — the single-simdgroup flash kernel, kept for hd % 8 ==
  0 shapes outside the hd 64/128 instantiations: one simdgroup per (8-query
  tile, head), K^T/V 8×8 fragments direct from the f16 cache, scalar masked
  online softmax per 16-position KV block, diagonal-MMA row rescale. An earlier
  all-f32 flash kernel lost to the scalar split (it staged K/V through 8 KB
  threadgroup tiles); zero staging is what makes these win. Tail blocks may read
  up to 7 rows past the causal limit — always inside the bound cache buffer —
  and are masked.
- **`attnvec_f16kv` (hd 64/128)** — the llama.cpp `kernel_flash_attn_ext_vec`
  structure for long-context decode (kv_len ≥ 128; hd 64/128 at NSG=32, hd=256
  at NSG=16 — its per-simdgroup O partials are NSG·hd·4 B and 32 simdgroups
  would blow the threadgroup budget): NSG simdgroups per (query, head), each
  owning interleaved 32-position KV blocks with a private online softmax — 4 KV
  rows × 8-lane dots per pass, shuffle-tree reduce, ONE simd_max/simd_sum per
  block (the split kernels below run that chain once per position and are
  latency-bound on it) — merged once at the end by a log2 tree. Q stays f32
  (parity 1e-4); tail rows clamp into the cache and mask in the softmax.
- **`attnsplit32_*` / `attnsplit_*`** — narrow launches and long contexts on
  head sizes without a vec instantiation: NSG (32 or 8) simdgroups per (query,
  head), each owning a strided KV slice with a private online softmax, merged in
  threadgroup memory.
- **`attention_*`** — the lean one-simdgroup-per-(query, head) kernel for
  short-context leftovers.
- **`attention_canvas_*` / `attention_canvas32_*`** (Phase D, DiffusionGemma
  denoise, `AttnMask::Canvas` — see `docs/diffusion-gemma.md`) — every row
  attends the SAME fixed bidirectional `[lo, kv_len)` regardless of its own
  position, which none of the kernels above can express (they all derive their
  bound from `pos + row_index`). A dedicated `ATTNSPLIT_CANVAS_KERNEL` family
  (not a sentinel on the ordinary split kernels' otherwise-dead fields — that
  risked a real dispatch colliding with a canvas one on the same value) that
  never serves a non-canvas caller: `p.pos` carries the fixed `hi = kv_len - 1`
  and `p.kv_len` carries `lo` directly. `exec.rs` forces every `Canvas` mask
  straight here, bypassing flash/split/vec routing. UNVERIFIED on real hardware
  — written blind, like the KV-quant Metal work.

**Per-pipeline thread-cap gating**: `maxTotalThreadsPerThreadgroup` is
per-PIPELINE (register pressure or a paravirtual device — GitHub's macOS CI —
can push it below a kernel's requirement) and `encode_tg` clamps silently, which
for these kernels means skipped KV slices and garbage merges. Every
wide-threadgroup route checks its own pipeline's cap and degrades to the
next-narrower kernel.

## Quantized KV caches (`INFR_KV_TYPE_K` / `INFR_KV_TYPE_V`)

Per-side selection like llama's `--cache-type-k/-v`. Three read classes on
Metal, in descending preference:

- **Native read — f16, Q8_0, and coupled Q4_0 / IQ4_NL**: the attention kernels
  decode the compact blocks in-kernel with f32 accumulation, and the decode
  replay tape records them (dyn kernel variants read the live position). Q8_0
  (34 B / 32 elems, `INFR_KV_Q8=1` legacy alias): prefill rides
  `attnflash2_q8kv` (dequant-staged 24 KB KV tiles, K pre-transposed [hd][64]),
  decode `attnvec_q8kv`; **beats f16 at depth** (0.6B tg128@d16384 +26% over the
  f16 cache) on half the bandwidth. Q4_0/IQ4_NL (18 B / 32 elems, quarter
  footprint): the same vector-flash body over a KV accessor struct — decode at
  d4096 runs in the q8 class (~3× the prepass path it replaced), and q4_0 beats
  f16 at depth. The native route requires K and V COUPLED (same dtype); a mixed
  pair has no native dyn kernel and the replay gate rejects it (a K-only check
  here once taped the f16 kernel over a quant V cache — corrupt decode; the gate
  now requires kdt == vdt).
- **Dequant→f16 prepass — Q4_1, Q5_0, Q5_1, bf16, turbo2/3/4, and any mixed
  pair**: WriteKv quantizes into the compact cache; Attention expands each
  quantized side into a transient f16 scratch and runs the f16 kernels over it
  (the Vulkan model). Correct everywhere, but decode re-expands the WHOLE prefix
  every token (O(n²) over a generation) and forces static per-token decode —
  measured ~3× slower than f16 at d4096. Fine for prefill; footprint-only
  formats.
- **f32**: its own native f32 attention (exact; the CPU oracle's math).

Quality guidance (measured, planted-secret recall probes): coupled ≤4-bit KV on
BOTH sides costs long-context recall that the f32-attending CPU keeps — keep K
at f16/q8 and quantize V (llama guidance) unless footprint forces both.
turbo2/3/4 destroy small models outright (garbage on a 0.6B on every backend —
they need a model with margin to spend). Write quantization is byte-identical
between the CPU reference and every `writekv_*` kernel.

## Decode latency: wide small-op kernels + the replay tape

The GPU-counter profiler's first reading found gemma decode spending 20% of its
GPU time in RmsNorm: 105 launches/token, each a single 32-lane simdgroup
serializing dim/32 dependent loads (~20 µs) with nothing else resident at
rows==1. The same disease showed in three more places, each fixed the same way:

- `rmsnorm_wide_f32` / `qknormrope_wide_f32`: 8 simdgroups per row (per head for
  rope), partials folded through threadgroup memory, routed at rows ≤ 4. Fleet
  decode +8-19%.
- `rope_f32` ran ONE THREAD per (row, head) — serial pairs, trig included, plus
  a whole-head copy (19% of TinyLlama decode). Now one thread per rotation pair.
- The k-split GEMVs above (4 simdgroups per row group) — small/mid GEMVs
  underfill the GPU and serialize the whole k-dim.

The decode replay tape (`Capabilities::decode_replay`) records one token's
dispatches and re-encodes them per token with **no graph walk** — and replays on
a CONCURRENT encoder: each recorded dispatch carries a write mask
(`TapeEntry.wmask`), and barriers are placed exactly at RAW/WAW/WAR hazards, so
independent runs (q/k/v GEMVs, the two KV writes) overlap. Un-annotated
dispatches record writes-everything and replay serialized — never under-fenced.
MoE decode tapes (the device expert path resolves all data dependence on-GPU),
and so does the llama arch (`Op::Rope` reads the bound positions buffer,
replay-safe by construction).

**The tape is a per-backend cache matched on a
`(op-sequence, bound-buffer-address)` fingerprint, and it is invalidated on
`compile()`.** That invalidation is load-bearing correctness, not hygiene: the
fingerprint can COLLIDE across independent compile+execute calls once the
allocator reuses a freed buffer's address, which would replay a
structurally-stale tape (garbage / zeroed output). Only the seam's decode loop —
which compiles its plan ONCE and executes it per token — keeps a live tape; any
code that recompiles a decode-shaped graph with fresh IO buffers (the MTP head,
which rebuilt a rows==1 rope+attention graph every draft step) records afresh
instead. This was a real bug: the MTP head intermittently replayed a stale
zeroed tape, dropping its draft acceptance from 0.82 to 0.26 with no error
anywhere. When repeated- compile GPU output is intermittently zero/garbage,
suspect the tape fingerprint FIRST.

## Linear attention (DeltaNet / Qwen3-Next)

`deltanet_f32_k{1,2,4,8}`: one SIMDGROUP per (value-dim, head) — 32 lanes split
the k-dim with KPL = kd/32 state entries per lane held in REGISTERS across the
whole chunk, so the token recurrence needs only `simd_sum`s (no threadgroup
memory or barriers; the llama.cpp `kernel_gated_delta_net` parallelization). KPL
is a compile-time template parameter because MSL cannot register-promote
runtime-indexed arrays — the runtime-bound version silently spilled the
"register-resident" state to thread-private memory and ran 1.7× SLOWER than its
predecessor; fixed-KPL unrolled is 1.6× faster (qwen35 prefill 0.73× → 0.80×;
the residual is the Linear band). State updates the bound buffer in place;
`conv1d_silu_f32` runs one thread per channel with the ring state in registers.

## Speculative decoding (`INFR_SPEC_DRAFT=<gguf>`)

Engine-level draft-verify on the seam, `run` and `serve` through one selection
funnel (`metal_chat_model`): the draft session proposes up to `INFR_SPEC_K`
(default 6, adapted per round by an acceptance EMA) greedy tokens; ONE batched
target forward verifies all of them (`logits_rows` LM head over the candidate
rows; the m=2..8 rows ride the multi-row GEMV); rollback is the session
prefix-diff (free). Greedy-only — the committed stream is the target's greedy
stream over the verify forward, pinned end-to-end by
`metal_spec_decode_matches_target_only_greedy`. Pays for ≥8B-class targets: 8B +
0.6B draft = +12-16% on code-shaped output, prose break-even. The verify-cost
residual (~5-7 ms/row) is NOT simdgroup drift — a lockstep variant
(barrier-pinned token simdgroups) measured a wash, and register-hoisted
(ALU-bound) and split-K cooperative (latency-bound) forms lost too; all three
classes are measured and parked.

Speculative serve is slotted: the target/draft session pair each pick their
best-prefix slot per request (`SlotPool`, below), so a conversation switch on a
multi-user spec serve costs a suffix prefill, not a full re-prefill of both
models — measured 4.2× on conversation return (2.0 s vs 8.4 s at ~1600-token
contexts).

## MTP (multi-token prediction, `INFR_MTP=1`)

A qwen35 GGUF that ships a trained MTP head (`nextn.*` at `blk.{n_layer}`,
`Config::n_layer_nextn`) can draft with the head instead of a separate draft
model: `MtpHeadSession` runs the head's one-layer forward per draft step, the
target trunk verifies (the same `run_verify` all-rows-logits+hidden path spec
decode uses), catch-up re-syncs the head KV. The driver is backend-generic
(`generate_mtp_spec_core` over `&dyn Backend`) with `_vulkan`/`_metal`/`_cpu`
wrappers; the CPU one is the exact-f32 acceptance ORACLE (a backend whose head
numerics drift shows up as a lower alpha against it).

Correctness holds on Metal — MTP output is token-identical to target-only
greedy, and acceptance is 0.82 (the CPU/f32 oracle is 0.96; the residual is
ordinary f16). But MTP is **net-negative on Metal today**, and the profile says
why: it is NOT the head or the lm_head — it is the qwen35 no-rewind reprefill.
DeltaNet's recurrent state can't rewind, so every rejected draft re-prefills the
whole committed context (`start==0` verify, `m` = full length, ~100 ms+ at
depth), and at alpha 0.82 rejects are frequent enough that reprefills dominate
the 74% "verify" share. The fix is DeltaNet state checkpointing (save per
committed position, restore on reject) — a cross-backend engine feature, not a
Metal kernel lever; parked because at the Vulkan alpha (~0.95) reprefills are
~11% of verify, but Metal's lower f16 alpha amplifies them. See "Remaining
levers".

## Multi-turn serve: conversation slots

Both GPU sessions share the backend-agnostic `SlotPool` (`INFR_KV_SLOTS`,
default 4): best-prefix slot pick, fork off the Arc-shared weight upload,
cross-conversation prefix seeding via device-side KV copy (`copy_buffer`), LRU
recycling. Slot switches are replay-safe by construction — the tape fingerprint
covers the bound KV/IO buffer addresses, so a switch re-records (one graph-walk
token) instead of replaying into a stale slot's buffers.

## Shader compilation and the on-disk pipeline cache

Getting from MSL to something the GPU runs is two compilers, and only one of
them is cacheable:

- **front end**, MSL → AIR: `newLibraryWithSource:` over the ~340 KB
  `msl_source()` assembles (9 `.metal` files plus the IQ2/IQ3 grids emitted from
  `infr_core::iquant_grids`). Runs on **every launch** and there is no way
  around it at runtime — `MTLLibrary` cannot serialize the AIR back out, and the
  only supported way to persist a `.metallib` is `xcrun metal` ahead of time,
  which would make Xcode CLT a runtime dependency. If this turns out to dominate
  startup, the fix is a build-time `.metallib` in the release artifact.
- **back end**, AIR → GPU ISA: one run per `MTLComputePipelineState`, created
  lazily by `Pipelines::get` on a kernel's first use. **This is cached**
  (`src/pcache.rs`): an `MTLBinaryArchive` is seeded at backend init from
  `~/.cache/infr/metal-pipelines-<device>.bin`, passed as
  `MTLComputePipelineDescriptor.binaryArchives` at every creation, extended as
  new kernels appear, and re-serialized (debounced, and on drop).

The archive grows lazily during a run, so the lifecycle is Vulkan's
(`infr-vulkan/src/pcache.rs`), not ROCm's whole-artifact-at-init one; the bytes
go through the shared `infr_core::kernel_cache` seam, which owns the envelope,
the payload checksum, the atomic+durable write and the poisoned-blob tripwire.
The payload additionally carries a manifest of the function names inside the
archive — without it every launch would re-`add` (and so re-compile) every
kernel, and the cache would cost exactly what it saves.

The key is
`FNV(msl_source()) ++ source length ++ device name ++ MTLDevice.architecture.name ++ NSProcessInfo.operatingSystemVersionString ++ fast-math`,
so a kernel edit, a GPU swap or an OS bump discards the blob wholesale.
`registryID` is deliberately NOT in it (not stable across boots — it would
invalidate every reboot).

Every failure degrades: no binary-archive support, a rejected archive, a failed
serialize, a bad blob — all fall back to plain
`newComputePipelineStateWithFunction:` at cold speed. `FailOnBinaryArchiveMiss`
is never used; a miss must recompile, not error.
`--set kernels.metal.pipeline_cache=false` turns it off.

`INFR_PROF` prints the breakdown: library-compile ms, PSO count and ms, archive
hits vs misses, blob size.

## Profiling

`INFR_METAL_PROFILE=1`: per-op CPU-encode + GPU commit+wait wall, printed on
drop. `INFR_METAL_PROFILE=2`: additionally flushes after each op to attribute
GPU wall per op — costs the batching AND, because the replay tape re-executes
decode tokens without walking ops, only ever attributes the RECORDED token.
`INFR_METAL_PROFILE=3` is the honest mode: `MTLCounterSampleBuffer`
stage-boundary timestamps, one encoder per op inside ONE command buffer — true
in-context per-op GPU time, with the tape disabled so every token attributes.
(Apple silicon samples at stage boundaries only, not dispatch boundaries;
metal-rs 0.33's `resolve_counter_range` returns uninitialized memory, so the
resolve reads the NSData directly.)

## Tests

`tests/parity.rs` — 94 GPU parity tests vs the CPU reference: every op, and for
quantized Linear every (format × kernel shape) pair including partial tiles, the
small-m split-K regime at real verify shapes, and deep-coupled quant-KV
attention at kv_len=2048; attention covers
unsplit/split8/split32/flash/flash2/vec, sliding windows at both tile and block
granularity, partial query tiles, the retained hd=72/96 routes, and the hd=256
rows==1 short-kv `attnsplit` + hd=256 partial-rope (`freq_base` 1e7) cases the
MTP head decode exercises (which nothing taped ever reaches).
`tests/kernel_names.rs` — the missing-kernel tripwire: every kernel-shaped
literal in exec.rs must resolve in the compiled library (the cap-check fallback
otherwise turns a vanished kernel into silent perf loss — it happened once, 3×
decode loss with zero errors). `tests/dispatch_overhead.rs`, `tests/gemv_bw.rs`
— evidence probes (ignored), kept so the floors above stay reproducible.
`tests/pcache.rs` — the pipeline cache end to end (cold → save → reload →
pipelines from the archive, identical results), plus the damaged-blob,
rejected-archive, stale-key and cache-off degradation paths. It holds exactly
ONE test because it repoints `XDG_CACHE_HOME`; a second test in that binary
would race it. The device-free halves are unit tests: `src/pcache_blob.rs`
(payload framing, key composition, device token — runnable off a Mac with
`rustc --test`) and `src/pcache.rs` (save debounce, temp-path shapes).

## Negative results (measured — don't re-try without new evidence)

- Op fusion / dispatch-count reduction: per-kernel overhead is ~1.2 µs; a fused
  Add+RmsNorm measured nothing at prefill. (The decode Linear→Add residual
  peephole — mirroring the Vulkan adapter — is the exception: +2.5% on 4B
  decode, from removing a dependency stage per sublayer, not dispatch count.)
- Row-tiled GEMV for prefill weight reuse: zero change — the SLC already
  absorbed the re-reads; the scalar pipeline's ~1:1 load:FMA ratio was the limit
  (hence MMA).
- f32 flash attention (three tuning rounds) and a chunked-SIMD scoring variant:
  both lost to the scalar 32-way split.
- Same-shape half-fragment GEMM swap: zero — the f16 win only cashes as bigger
  tiles.
- 64-row private GEMM tiles: register pressure regression. BK=32 in both the
  private and the first cooperative tile: nothing / worse.
- Linear threadgroup-width sweep (32→1024), uint4-wide GEMV loads, select-based
  `get_scale_min_k4`: all flat.
- NSG=2 mid-k GEMV split: wash-to-negative (the reduce tax eats the halved chain
  at gemma's ne=1152 shapes).
- Vectorized DEC16 staging loads in the GEMM path: no change (staging is hidden
  behind the MMAs).
- B-direct cooperative GEMM (activation fragments straight from a pre-cast f16
  buffer, no B staging — the Vulkan A_GLOBAL idea): wash on M3; B staging was
  already free.
- Ungated k-split: −8% on 0.6B (in_f=1024 is four Q4_K superblocks; three of
  four simdgroups idle and the reduce is pure tax) — hence the per-format
  k-depth gates.
- Lockstep multi-row GEMV (threadgroup per row pair, m token simdgroups
  barrier-pinned per block): 80.1 vs mrv's 79.0 ms median on an 8B verify — the
  barrier costs nothing AND buys nothing, so the free-running simdgroups were
  already cache-coincident; the verify residual is not weight-stream drift.
- Runtime-bound local arrays in hot kernels: MSL cannot register-promote a
  runtime-indexed array — the first simdgroup-per-column DeltaNet spilled its
  "register" state to thread-private memory and ran 1.7× slower than the kernel
  it replaced. Compile-time template bounds + unrolled loops, always.

## Remaining levers

Decode is at parity; prefill sits at the measured tile floor. What's left needs
a new evidence class or new hardware:

- Prefill GEMM: ~53% of f16 simdgroup_matrix peak, same class as llama.cpp —
  Xcode's per-line shader profiler (not installed on the dev box) is the next
  instrument if the last ~10-15% ever matters.
- Metal-4 tensor ops: the hardware step change; `has tensor = false` on M3,
  likely M5-only.
- gemma decode at depth (0.92×): +0.58 ms/token of hd=256 vec attention on the
  global layers — 4% of one cell, measured and parked.
- The routing gates (k-split thresholds, wide-norm rows ≤ 4) were tuned on an M3
  Pro; a boot-time micro-sweep would adapt them across M1-M4.
- Spec-verify flat cost (the ~2× acceptance ceiling): all three kernel classes
  measured and excluded (register-hoisted ALU-bound, split-K cooperative
  latency-bound, lockstep a wash) — needs a genuinely new shape.
- Remaining codebook weight formats (IQ2*/IQ3*/TQ\*): guarded (hard error past
  the working set) but not native; larger codebook grids than IQ4's 16-entry
  table.
- qwen35 prefill residual (0.80×): the Linear band at the GEMM floor — qwen35
  runs more projections per token than dense; same instrument gap as the prefill
  GEMM lever above.
- MTP profitability on Metal (net-negative today, see the MTP section): the
  ceiling is the qwen35 no-rewind reprefill on every rejected draft, NOT any
  Metal kernel — the fix is engine-level DeltaNet state checkpointing
  (cross-backend), amplified on Metal only because f16 alpha (0.82) is below the
  Vulkan reference (~0.95). No Metal-side lever; measured and handed back.
- DiffusionGemma batched MoE (Phase D left this unbuilt, see
  `docs/diffusion-gemma.md`): the fused `ffn_gate_up_exps`/`ffn_down_exps`
  layout doesn't fit the device MoE kernels' assumed shape, so DG's MoE FFN
  still runs a per-token host loop on Metal — the biggest remaining DG perf gap,
  and too large to build blind (needs hardware to iterate against).
