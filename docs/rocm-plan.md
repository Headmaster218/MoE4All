# rocm-plan.md — a native ROCm/HIP backend for infr

Plan for adding `infr-rocm`, a fourth compute backend (alongside CPU, Vulkan,
Metal) targeting AMD GPUs through the ROCm/HIP stack. The plan is ordered
**correctness first, performance second**: it reaches a fully correct backend —
every model and every quant format infr supports, generating token-for-token
agreement with the CPU reference — before a single perf lever is pulled, then
climbs from naive kernels to a fast kernel for every supported model × quant
combination.

> Status: **BUILT & LANDED (37+ slices on `main`).** Part A correctness is
> complete for every arch that fits VRAM; the perf climb (Part B) is well under
> way (native/int8 decode, WMMA prefill, split-KV attention, decode fusion) and
> all three Vulkan-style memory-paging modes — **MoE experts, KV-cache overflow,
> and dense-weight paging** — are implemented and prefetch-optimized. Current
> resident perf is ~0.2–0.3× llama.cpp HIP on Q4_K_M (decode ~130 t/s, up ~60×
> from the naive baseline); ≥1.0× vs llama.cpp remains open (deep mmq/mmvq
> kernel work). Big-model capability: 30B MoE, 27 GiB KV contexts, and >VRAM
> dense models all run via paging on a 24 GB card. This doc is the roadmap +
> design record; phase checkboxes below are updated as slices land.

## Why a ROCm backend at all (infr already runs on AMD via Vulkan)

The Vulkan backend already runs on every AMD GPU and is heavily tuned (coopmat
GEMM, dp4a `mmq`, flash attention). ROCm/HIP is still worth a native backend:

- **A higher perf ceiling on matrix ops.** HIP exposes RDNA3+ **WMMA** and CDNA
  **MFMA** matrix cores directly, plus mature GEMM libraries (**rocBLAS**,
  **hipBLASLt**) and integer/`fp8` dot builtins. The Vulkan coopmat path was
  measured to top out around ~36 TF on wide shapes (`docs/perf.md`); HIP gives a
  second, independent route to close the remaining gap to llama.cpp.
- **Apples-to-apples with llama.cpp's HIP backend.** The reference target
  becomes `llama.cpp` built with `-DGGML_HIP=ON` on the _same_ device, so ratios
  are clean.
- **Tooling.** `rocprof` / `rocprofv3`, `roctracer`, and `omniperf` give
  hardware-counter attribution the Vulkan path can only approximate.
- **It is directly validatable on the dev box.** The RX 7900 XTX (RDNA3,
  `gfx1100`) is an officially ROCm-supported card, so the new backend can be
  A/B'd against the existing Vulkan backend on the _same_ silicon — the
  strongest possible correctness and perf oracle. (The RDNA2 iGPU, `gfx1036`, is
  not on the official ROCm support list; treat it as best-effort, not a gate.)

Non-goal: replacing Vulkan. ROCm is additive; Vulkan stays the portable default.

## What a backend actually has to do (the seam is already generic)

Everything above the device is backend-agnostic already: the `Graph`/`Op` IR,
the seam runner (`generate_dense_backend`), the chat REPL, KV-cache policy, MoE
routing policy. A backend is exactly:

1. Implement **`infr_core::backend::Backend`**
   (`crates/infr-core/src/backend.rs:464`) plus its helper traits `Buffer`
   (`:374`), `Plan` (`:397`), `ProgressScope` (`:372`).
2. Fill in **`Capabilities`** (`backend.rs:42`) honestly — the runner degrades
   to split-op fallbacks for anything reported unsupported, which is the entire
   correctness-first strategy.
3. Execute every **`Op`** variant (`crates/infr-core/src/graph.rs`, flat list in
   `Op::kind` `:554`).

`infr-metal` is the from-scratch template to mirror throughout — a non-Vulkan
backend that assembles kernel source, compiles once at init, caches pipelines by
name, and walks the graph in a `run_op` match. This plan follows its shape.

### The `Backend` trait surface (required unless noted)

| method                                        | ref              | correctness-phase behavior                                       |
| --------------------------------------------- | ---------------- | ---------------------------------------------------------------- |
| `name(&self) -> &str`                         | `backend.rs:465` | `"rocm"`                                                         |
| `capabilities(&self) -> Capabilities`         | `:466`           | conservative set (below)                                         |
| `alloc(bytes, usage) -> Box<dyn Buffer>`      | `:474`           | `hipMalloc` + **zero-init** (`hipMemset`) — calloc contract      |
| `upload(dst, &[u8])`                          | `:484`           | `hipMemcpy` H2D                                                  |
| `download(src, &mut [u8])`                    | `:485`           | `hipMemcpy` D2H                                                  |
| `compile(&Graph) -> Box<dyn Plan>`            | `:517`           | `GraphPlan::boxed(graph)` (clone; real work in execute)          |
| `execute(plan, bindings)`                     | `:518`           | walk graph → dispatch kernels → one stream sync                  |
| `sync()`                                      | `:552`           | `hipStreamSynchronize` / `hipDeviceSynchronize`                  |
| `alloc_uninit` (default → `alloc`)            | `:481`           | skip zero-init once proven; debug-poison                         |
| `weight_progress` (default no-op)             | `:498`           | drive the `indicatif` bar from inside `alloc` (Metal does)       |
| `copy_buffer` (default host-bounce)           | `:508`           | override with device `hipMemcpyDtoD` (KV prefix-share primitive) |
| `execute_chain` (default `Ok(None)`)          | `:526`           | perf phase: n back-to-back decode steps, one submit              |
| `max_decode_chain` (default `MAX`)            | `:548`           | perf phase clamp                                                 |
| `moe_paged` / `dense_paged` (default `false`) | `:564/:577`      | perf phase (VRAM streaming)                                      |
| `eb_sample_reduce` (default `Ok(false)`)      | `:595`           | perf phase (fused diffusion sampler)                             |

There is **no `warmup` on the trait** — warmup is `ChatModel::warmup`
(`chat/mod.rs`), a throwaway generate that forces lazy pipeline compiles; the
ROCm chat wrapper inherits it for free.

### `Capabilities` — the correctness dial

`Capabilities` (`backend.rs:42`) is how the backend tells the compiler/runner
what to emit. The correctness-first rule: **advertise the minimum, let the
runner split everything.** Start with:

- `name`, `f16 = true`, `i8 = false`, `i8_dot = false`, all `coopmat_* = None`.
- `integrated = false` (RX 7900 XTX is discrete; skip the submit-splitter math —
  `integrated_ubatch_rows`/`initial_submit_dispatch_cap`, `backend.rs:223/271`).
- `compute_units` = device CU count (cosmetic until perf).
- Every fused-op opt-in **false**: `decode_replay`, `combined_gu`,
  `embed_gather`, `gpu_sample`, `argmax_rows`, `argmax_prob`, `gated_rmsnorm`,
  `kv_swa_ring`.
- `buffer_device_address` — HIP pointers are raw device addresses, so this can
  be `true` early if the exec path wants pointer-fed kernels; leave `false`
  until a kernel needs it.

Each field flips to `true` in a later phase only once the corresponding native
kernel exists and passes parity. Flipping a cap before the kernel exists is the
one way to get silent garbage.

## Kernel authoring strategy

Two viable routes; the plan uses **(A) for correctness, migrates hot kernels to
(B/C) for perf**:

- **(A) `hiprtc` runtime compilation** — mirror Metal exactly: keep HIP kernels
  as `.hip`/`.cpp` source `include_str!`'d into the crate, assemble one
  translation unit at `RocmBackend::new()`, compile once with
  `hiprtcCompileProgram`, load the code object, and lazily fetch+cache
  `hipFunction_t` by name (`hipModuleGetFunction`). Quant codebook LUTs are
  emitted from `infr_core::iquant_grids` (the same statics the CPU/Metal/Vulkan
  paths use — bit-exact by construction, never hand-transcribed). This is the
  fastest path to "all kernels present" and matches `infr-metal/src/shaders.rs`.
- **(B) Offline `hipcc` in `build.rs`** — compile `.hip` → code objects
  (`--genco`) per `gfx` arch, `include_bytes!` the blobs, load with
  `hipModuleLoadData`. Better startup latency + real compiler optimization; add
  once the kernel set stabilizes (analogous to Vulkan's `build.rs` glslc
  pipeline, `crates/infr-vulkan/build.rs`). Multi-arch fat binaries cover
  `gfx1100`/`gfx1036`/CDNA.
- **(C) Vendor libraries** — **rocBLAS / hipBLASLt** for the f16/bf16 GEMM tiles
  once weights are dequantized, and int8 GEMM where available. Used in the perf
  phases for the wide prefill GEMM; custom kernels stay for quant-decode GEMV,
  attention, MoE routing, DeltaNet.

Reuse `infr_core::dequant` (block→f32/f16) and `infr_core::iquant_grids`
verbatim — these are the correctness anchors shared with every other backend.

## Crate + workspace wiring

Mirror `infr-metal`'s platform gating so the workspace still builds everywhere:

- New crate `crates/infr-rocm`, added to `Cargo.toml` `members`.
- Gate the real implementation behind **both** a target cfg and a cargo feature:
  the crate is a no-op empty lib unless `cfg(target_os = "linux")` **and**
  `feature = "rocm"` (HIP is not installed on most machines / CI). Off-gate it
  compiles to an empty lib with zero HIP deps, exactly as `infr-metal` does off
  macOS (`crates/infr-metal/src/lib.rs:4`).
- Deps: `infr-core`, `infr-gguf`, `half`, `bytemuck`, `indicatif`; HIP FFI
  behind the feature (either hand-written `extern "C"` bindings to
  `libamdhip64` + `hiprtc`, or a vetted `hip-sys`/`hiprt`-style crate — evaluate
  in Phase 0). Dev-dep `infr-cpu` as the parity oracle (same as Metal's
  `tests/`).

### Selection / registration (the `INFR_DEV` grammar)

Device selection is one grammar shared by `--dev` and `INFR_DEV`
(`crates/infr-cli/src/main.rs`). Add a `rocm` token:

1. `enum Backend { Vulkan(Option<String>), Metal, Cpu, Rocm }` (`main.rs:62`).
2. `parse_dev_spec` (`:99`) — accept `rocm` (optionally `rocm:N` to pin a device
   index, matching `VulkanN`).
3. `DeviceOpts::resolve` / `selected_backend` (`:187/:134`) — publish/read it.
4. `build_chat_model` (`:841`) and the diffusion-gemma 3-arm match (`:861`) —
   add a `Backend::Rocm` arm building a `RocmSeamChat` (twin of
   `metal_chat_model`, `:1392`). On a non-ROCm build, fall back with a clear
   error (mirror Metal's non-macOS `CpuDenseChat::new_metal` fallback).

### Chat + session plumbing

- `crates/infr-llama/src/chat/rocm.rs` — `RocmSeamChat: ChatModel`
  (`chat/mod.rs:38`), twin of `chat/metal.rs`. Re-export in `chat/mod.rs`.
- `DenseRocmSession` in `crates/infr-llama/src/seam/model.rs` (twin of
  `DenseMetalSession`, `:272`) owning a `RocmBackend`; a
  `SeamModel::rocm_session` that calls `RocmBackend::new()` (mirror
  `metal_session`, `:1194`).
- `rocm_upload_bind` in `crates/infr-llama/src/seam/mod.rs` (twin of
  `metal_upload_bind`, `:97`): alloc device buffer + upload the **raw native
  dtype** bytes, lazy-dequant on first use. Then drive the existing
  `generate_dense_backend` runner unchanged.

### The `be.name()` capability gates to thread `"rocm"` into

Several seam decisions branch on the backend name string. Each must learn about
`"rocm"` as the corresponding feature lands (do NOT enable a branch before its
kernel exists):

- KV-cache format gates in `crates/infr-llama/src/seam/runner.rs`:
  `kv_q8_backend` (`:432`), `kv_turbo_ok` (`:435`), `blk_ok` (`:439`),
  `dense_ok` (`:442`), the Metal-only `k_fmt != v_fmt` guard (`:477`).
- Self-conditioning `gpu_sc` gate (`:3030`) — diffusion-gemma, perf phase.

Track these as a checklist; a missing arm silently routes ROCm down the CPU-ish
slow path or, worse, an incompatible one.

---

# PART A — CORRECTNESS (all models, all quants, token-for-token)

Goal of Part A:
`INFR_DEV=rocm infr run <any supported model>:<any supported quant>` loads,
generates coherent text, and matches the CPU reference token-for-token on the
quant sweep — with **naive kernels only**. No perf work.

## Phase 0 — Scaffolding: an empty backend that builds and is selectable

- Create `crates/infr-rocm` (feature-gated empty lib), wire it into the
  workspace and the `INFR_DEV=rocm` selection chain (all sites above).
- Evaluate the HIP FFI surface: `hipMalloc/hipFree/hipMemcpy/hipMemset`,
  `hipStreamCreate/Synchronize`, `hiprtc*`, `hipModuleLoadData/GetFunction`,
  `hipModuleLaunchKernel`, device enumeration (`hipGetDeviceCount`,
  `hipGetDeviceProperties`). Decide hand-rolled `extern "C"` vs a bindings
  crate.
- `RocmBackend::new()` that enumerates the device, creates a stream, and reports
  `capabilities()` — no kernels yet.
- **Gate:** `cargo build -p infr-rocm --features rocm` on the dev box; workspace
  `cargo build` still green everywhere without the feature. `infr` recognizes
  `--dev rocm` and errors cleanly if the feature is absent.
- **Exit:** the backend constructs, allocs/uploads/downloads a buffer, `sync`s.
  (A round-trip `upload`→`download` byte-identity test is the first parity
  test.)

## Phase 1 — Correctness baseline: dequant→f16 everything

The trick that yields **full model + full quant coverage on day one**: never
decode a quant format natively yet. In `rocm_upload_bind`, keep the raw bytes;
in the `Linear`/`EmbedGather`/expert paths, dequantize the weight block to `f16`
on first touch via `infr_core::dequant` (host-side or a trivial device dequant),
cache it, and run a **single naive f16 GEMV/GEMM kernel**. Every one of the 24
weight quant formats + floats is then "supported" because they all reduce to the
same f16 matmul — exactly how Vulkan's fallback and Metal's `weight_buf`
dequant-cache behave (`docs/kernels.md`).

Kernels needed this phase (all naive, one variant each, no tiling):

- `linear_f16` GEMV/GEMM (`m=1` and `m>1`, dequant-cached weights).
- The norm/elementwise/rope set: `RmsNorm`, `RmsNormAdd`, `Softmax`, `QkNorm`,
  `Rope`, `QkNormRope`, `GatedRmsNorm`, `Add`, `AddBias`, `Scale`, `MulVec`,
  `Softcap`, `GatedAct`, `Copy`, `CopyStrided`, `EmbedGather`, `Argmax`,
  `Sample`.
- `WriteKv` + a scalar `Attention` (Causal + SlidingWindow masks), f16/f32 KV
  only (no KV quant yet).
- `MoeFfn` via the **split/unfused** path (router GEMV on host-read top-k is
  acceptable here — correctness, not speed), `MoeSharedExpertAdd`.
- `Conv1dSilu` + a sequential `DeltaNet` (persistent `S` state, mutated in place
  and **surviving across `execute` calls** — see the parity test below).

Advertise nothing fused. `combined_gu=false` → the runner emits `GatedAct` over
separate gate/up. `gated_rmsnorm=false` → separate `QkNorm`+gate. Etc. This
keeps the kernel count minimal.

- **Gate:** per-op parity via `crates/infr-llama/tests/seam_op_parity.rs` — the
  one-op agnostic `Graph` run on CPU vs ROCm, including
  `state_persists_across_executes` for `DeltaNet`/`Conv1d`. Run
  `--include-ignored` on the dev box.
- **Exit:** a small dense model (e.g. Qwen3-0.6B Q4_K_M, gemma-3-1b) generates
  coherent text end-to-end on `--dev rocm`.

## Phase 2 — Op completeness, all architectures, blessed goldens

Extend Phase 1 until **every** model family runs and the goldens are locked:

- Cover every arch the CPU/Vulkan backends do: llama, qwen2/2.5, qwen3
  dense+MoE, gemma3, gemma4 dense/E2B/26B-A4B MoE, qwen3.5/3.6 (`qwen35`
  DeltaNet) + `qwen35moe`, diffusion-gemma, llama4 Scout, plus the validated
  fine-tunes (Ornith on `qwen35`/`qwen35moe`, Ternary-Bonsai on `qwen3`). Each
  exercises a different Op subset (heterogeneous dims, per-layer embeddings,
  dual FFN, SSM, block-diffusion canvas attention).
- KV cache: at least f16 + f32 correct (`WriteKv` cast + native-read attention).
- Add the **ROCm `gpu_seam` golden**: a backend-specific FNV-1a hash blessed
  with `INFR_BLESS=1` (each backend locks its own hash — CPU is f32, GPU is f16,
  so cross-backend token-for-token is brittle; `cpu_backend.rs:48`), PLUS a
  token-for-token `seam_rocm_matches_cpu` sweep over the quant families (the
  twin of `seam_vulkan_matches_cpu`, `cpu_backend.rs:380+`). Wire the drift
  guards (`linear.rs` tests) so a new dtype can't silently lose coverage.
- Thread `"rocm"` into the `blk_ok`/`dense_ok` KV gates as those formats gain
  correct read/write.

- **Gate:** the full `gpu_seam` suite passes on the dev box; token-for-token
  with the CPU oracle on every arch × a representative quant.
- **Exit of PART A:** every supported model × every supported quant produces
  correct, coherent output on ROCm, matching the reference. **Zero perf
  claims.**

## CI / validation strategy (applies across the plan)

Like the Vulkan and Metal GPU tests, ROCm parity/goldens require real hardware +
ROCm and are `#[ignore]`d, run on the dev box (the 7900 XTX is the gate). For
CI:

- A **`rocm-check`** job that `cargo check`s the workspace **without** the
  `rocm` feature (the empty-lib path) to catch `Op`-signature / match-arm drift
  — the analogue of the existing `metal-check` cross-compile guard (`ci.yml`).
- If a self-hosted AMD+ROCm runner becomes available, add a `test-rocm` job
  running the ignored parity suite (twin of `test-macos`). Until then, the dev
  box is the documented gate (as with `docs/igpu.md`).
- Every kernel lands with a parity test vs a CPU reference **before** its
  `Capabilities` flag flips — the non-negotiable rule that kept the blind Metal
  quant kernels correct.

---

# PART B — PERFORMANCE (fast kernel per model × quant)

Goal of Part B: climb from the naive baseline to a fast kernel for every
supported model × quant, benchmarked against `llama.cpp` (HIP build) and the
existing Vulkan backend on the same 7900 XTX, targeting **≥1.0×** on prefill and
decode for every model×quant, without moving a golden (correctness stays locked
from Part A). Follow the `docs/perf.md` loop: sweep → profile → one lever →
validate → bench serially. Work biggest-gap-first.

The AMD hardware levers, mapped to the Vulkan tiering (`adapter.rs:1340-1560`):

| Vulkan tier                      | ROCm/HIP analogue                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| f16 coopmat GEMM (prefill)       | **WMMA** (RDNA3 `gfx11`, wave32 `16x16x16`) / **MFMA** (CDNA); or rocBLAS/hipBLASLt |
| dp4a `mmq` int8 (decode/prefill) | `__builtin_amdgcn_sdot4` / `V_DOT4_I32_I8`; WMMA int8 on RDNA3                      |
| scalar dequant-in-shader GEMV    | plain HIP GEMV (the Phase-1 baseline)                                               |
| non-coopmat `nc_fma` warptile    | shared-memory (LDS) HIP warptile for archs without WMMA                             |

## Phase 3 — Native quant decode (drop the host dequant)

Replace the dequant→f16 cache with **in-kernel block decode** GEMV per format,
so decode streams the compact quant bytes (the dominant decode lever — fewer
bytes, `docs/cpu-perf.md`). One kernel family, per-`DType` `#define`/template
variants (mirror Vulkan's `-DFMT_<QUANT>` build variants, `build.rs`), driven
off `ALL_DTYPES` (`linear.rs:373`) and the IQ grid LUTs. Each format lands with
a GEMV parity test vs the CPU `vec_dot` + `dequant_block` (the Metal DEC16
pattern, `docs/kernels.md`). Covers all 24 weight formats.

- **Exit:** decode reads native quant bytes; `linear_f16`-with-dequant retired
  from the hot path; per-format GEMV parity green.

## Phase 4 — Integer-dot decode (int8 / dp4a analogue)

Add the int8-activation path: quantize the activation row to int8 once, integer
dot against native weight codes using `V_DOT4`/`sdot4` (gfx9+, RDNA) — the
principled integer route (each lane owns its accumulator → scale-after free),
the same reasoning as Vulkan's `mmq` and the CPU VNNI dots. Flip `i8`/`i8_dot`
in `Capabilities`; derive coverage from the shared `MOE_MMQ_DTYPES` SSOT
(`tensor.rs`). Add the multi-row (`m=2..16`) GEMV variant (Vulkan `mrow`).

- **Exit:** int8 decode path parity-clean for the mmq dtype set; measured decode
  win over Phase-3 on the quant sweep.

## Phase 5 — Matrix-core GEMM (prefill) + the tiering decision tree

The big prefill lever. Implement the tiled GEMM in priority order, porting
`adapter.rs`'s decision tree:

- **WMMA/MFMA coopmat GEMM** with in-shader dequant (`matmul_native` analogue)
  and an f16 variant (`matmul_proj` analogue). Populate `coopmat_f16` in
  `Capabilities` with the real tile (RDNA3 WMMA is `16x16x16`).
- **dp4a/WMMA-int8 `mmq`** batched prefill for the mmq dtypes.
- Evaluate **hipBLASLt/rocBLAS** for the wide f16/bf16 tiles vs the custom
  warptile — pick per-shape on the micro-bench (a `gemm_bench` twin), the way
  `docs/perf.md` measured coopmat operand tiers before trusting them.
- Optimize the LDS/occupancy budget (RDNA3 has 64 KB LDS/CU); the Vulkan
  campaign found occupancy, not math, was the ceiling (A_GLOBAL slice) — expect
  the same class of win.

- **Exit:** prefill runs on matrix cores; per-shape TFLOPS within reach of
  rocBLAS ceiling; goldens unmoved (bit-identical construction where possible —
  fixed-order reductions, no value-affecting atomics, `docs/perf.md`).

## Phase 6 — Fast attention + KV quantization

- **Flash attention** (fused, `hd==128`, Causal, matrix-core) — the
  `attention_prefill_flash` analogue; the geometry guards (rows≥64 or rows≥24 &
  kv≥8192) port directly. **Split-KV / flash-decoding** across KV chunks for
  decode (`attention_kv_split_dynac` analogue).
- **KV quant:** Q8_0 planar + block formats (q4_0/…/iq4_nl) via a dequant→f16
  prepass first (correctness), then dequant-in-attention for the cheap formats
  (Vulkan's `INFR_FLASH_DEQUANT`, only Q4_0/Q4_1/Q5_0/Q5_1). TurboQuant (WHT
  turbo2/3/4) via the shared dequant→f16 prepass. Thread `"rocm"` into
  `kv_q8_backend`/`kv_turbo_ok`/`blk_ok` (`runner.rs:432-439`) as each lands.
  Enable `kv_swa_ring` once the sliding-window ring is correct.

- **Exit:** attention off the scalar fallback; KV-quant coverage matches the
  other GPU backends; decode-at-depth competitive.

## Phase 7 — MoE (batched, id-GEMV, paged) + DeltaNet fast paths

- **MoE:** GPU-side top-k routing (router GEMV → bucket count/scan/scatter, no
  host readback — the class-5 host-in-the-loop fix, `docs/perf.md`); resident
  small-`m` **id-indexed GEMVs** (`native_id`/`native_idm` analogue) and batched
  **`mmq` experts** for larger `m` (gated on the `moe_mmq_ok` SSOT). Enable
  `combined_gu`/fused gate-up.
- **Paged experts** for models that exceed VRAM (llama4 Scout, big MoE): a HIP
  `GpuPager` analogue (LUT hop to an arena base address; HIP raw pointers make
  the device-address path natural). Override `moe_paged()`/`dense_paged()`.
- **DeltaNet:** the chunked matrix-core prep-pass (`deltanet_prep` analogue,
  needs WMMA + `kd==128`) over the sequential fallback; `Conv1dSilu` batched.

- **Exit:** MoE + qwen35 engines run on the fast paths; paged models load and
  run.

## Phase 8 — Fused ops, chained decode, device-side sampling

Turn on the remaining `Capabilities` opt-ins, each behind its now-existing
kernel:

- **`decode_replay`** — a record-once decode tape (Metal `Tape` / Vulkan replay
  analogue): device-side pos/kv params so the host doesn't re-record per token.
  Then **`execute_chain`** + `max_decode_chain` for n back-to-back decode steps
  per submit.
- **Device sampling:** `gpu_sample`, `argmax_rows` (MTP verify), `argmax_prob`
  (MTP draft), and `eb_sample_reduce` (diffusion-gemma fused entropy-bound
  sampler) — enables MTP speculative decode and the DG in-graph sampler on ROCm.
- Fused `gated_rmsnorm`, `RmsNormAdd`, `QkNormRope`, `GatedActFused`,
  `embed_gather`.

- **Exit:** ROCm reaches feature parity with the Vulkan backend's fused-op set;
  MTP + diffusion-gemma fast paths live.

## Phase 9 — Perf endgame: per model × quant tuning to ≥1.0×

The closing campaign, run exactly like `docs/perf.md`:

- Sweep the full **model × quant** matrix (`infr compare --sweep`) against
  `llama.cpp` HIP and the Vulkan backend on the 7900 XTX; rank worst-gap-first.
- Profile with `rocprof`/`omniperf` (hardware counters — the thing samply/Vulkan
  timestamps can't give); classify each bottleneck by the `docs/perf.md`
  taxonomy (wrong-kernel gate, grid underfill, occupancy, dispatch overhead,
  host-in-the-loop, kernel micro-arch) and fix one lever at a time.
- Per-shape GEMM tuning on a `gemm_bench` twin; verify prefill and decode stay
  bit-identical (goldens never move).
- Record the campaign in this doc (landed slices + before/after ratios), the way
  `docs/perf.md`'s campaign log and `docs/cpu-perf.md` do.

- **Exit (end state):** a fast kernel for **every supported model × quant
  combination**, ≥1.0× vs llama.cpp HIP where the hardware allows, with the
  correctness goldens from Part A still green. Update `docs/kernels.md` to add a
  ROCm column (native coverage) and `docs/perf.md`/`README.md` to list ROCm as a
  first-class backend.

---

## Milestone checklist

- [x] **P0** crate scaffold, `--dev rocm` selectable, `RocmBackend` skeleton —
      done
- [x] **P1** dequant→f16 baseline: naive kernels for the full Op set (23
      kernels), HIP FFI wired, dequant cache, `DenseRocmSession` +
      `RocmSeamChat` wiring — **validated on the RX 7900 XTX (gfx1100)**: dense
      Qwen3-0.6B Q4_K_M generates coherent text on `INFR_DEV=rocm`. Correctness
      gate landed (`crates/infr-rocm/tests/parity.rs`, `#[ignore]`/GPU):
      calloc-contract alloc, upload→download byte-identity, and
      single-`Op::Linear` GEMV parity vs the CPU reference for F16 + Q4_K.
      Backend-layer fixes: `hipMemset` rc checked, `alloc` OOM returns `Err` (no
      panic), `copy_buffer` bounds-checks the destination. Feature-build link
      wiring: `crates/infr-rocm/build.rs` (`$ROCM_PATH/lib`) + `infr-cli` `rocm`
      feature passthrough.
- [x] **P2** blessed ROCm gpu_seam gate + token-for-token vs CPU across the
      validated archs (Qwen3, Gemma-3, qwen35/DeltaNet, llama, qwen2.5,
      gemma4-E2B, BitNet i2_s) and all 24 weight quant formats (op-level parity)
      → **PART A correctness complete for every arch that fits VRAM.** The
      big-model archs that OOM under the naive f16 weight cache
      (gemma4-dense-12B, gemma4-MoE-26B, llama4 Scout; diffusion-gemma is also
      not yet wired) are deferred to after **P3**, which drops the f16 residency
      so they fit.
- [ ] **P3** native per-DType quant-decode GEMV (all 24 formats) — also unblocks
      the OOMing big-model archs above
- [ ] **P4** int8/dp4a decode + multi-row GEMV
- [ ] **P5** WMMA/MFMA (or rocBLAS) prefill GEMM + tiering tree
- [ ] **P6** flash + split-KV attention + KV quant
- [ ] **P7** MoE (batched/id/paged) + DeltaNet fast paths
- [ ] **P8** fused ops + chained decode replay + device sampling (MTP, DG)
- [ ] **P9** per model×quant perf to ≥1.0× → **PART B complete**

## Risks & open questions

- **ROCm install / CI.** No AMD+ROCm GitHub runner today; the dev box is the
  gate (documented, mirrors `docs/igpu.md`). `rocm-check` catches signature
  drift; a self-hosted `test-rocm` job is the eventual goal.
- **HIP FFI choice.** Hand-rolled `extern "C"` (zero-dep, full control) vs a
  bindings crate (faster start, external churn). Decide in P0; hand-rolled is
  the safer default given how thin the surface is.
- **`hiprtc` vs offline `hipcc`.** Runtime compile is the fast path to coverage
  but adds startup latency and needs the ROCm toolchain present at runtime;
  offline code objects (P5+) fix both but need multi-`gfx` fat binaries.
- **RDNA3 WMMA specifics.** wave32, `16x16x16`, and the RDNA3 accumulator/format
  quirks differ from CDNA MFMA; the tile const in `Capabilities` and the kernel
  must match the actual arch. The iGPU (`gfx1036`, RDNA2) has **no** matrix core
  — it takes the non-coopmat warptile tier (best-effort, not a gate).
- **`maxMemoryAllocationSize` / large SSBOs.** HIP allocations and the 4 GiB
  binding limits that bit the Vulkan path (`docs/igpu.md`) — verify HIP's limits
  early; big-MoE expert banks may need the paged path sooner.
- **Numerical parity.** Match rounding/accumulation order to keep goldens
  bit-identical where possible; where the f16 path legitimately differs, bless a
  ROCm-specific golden after verifying coherence (never blind-accept a diff).
