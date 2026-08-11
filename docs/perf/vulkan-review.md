# Performance review — Vulkan backend, multi-vendor (Intel / NVIDIA / AMD)

2026-07-31

> **Partly superseded — read this first (noted 2026-08-11).** Finding #2
> ("Vendor detection should be capability detection") and the architectural note
> and tables that build on it describe a `vendor_intel` flag on `Capabilities`
> and four vendor-keyed routing decisions. **That flag no longer exists** —
> `vendor_intel` has no occurrences anywhere in `crates/`, `Capabilities` has no
> vendor field, and `adapter.rs`'s `unified_mmv_row1` comment records the
> removal as deliberate ("new hardware needs no vendor quirk here"). The
> recommendation was carried out; treat #2, the "Architectural note" section and
> the "What the unified defaults would look like" table as a record of what was
> decided, not as a description of the current tree.
>
> Every line number in this file predates that change and several other slices,
> so re-locate by symbol rather than trusting them.
>
> The 2026-08-11 hardware-capability audit re-derived the detection inventory
> against current `HEAD` and found one thing this review did not consider: the
> capability-first design trusts what the device enumerates, and llama.cpp
> documents two drivers that misreport cooperative-matrix support. See
> `backlog.md` § B-HWDET-DRIVERID, and § B-HWDET-LIMITS / §
> B-HWDET-I8CM-FRAGLAYOUT for the rest.

## Scope

Vulkan implementation (`crates/infr-vulkan/`), focused on per-vendor kernel
routing, GEMM/flash-attention dispatch, and feature-gating decisions that differ
across Intel Arc (ANV), NVIDIA (proprietary / NVK), and AMD (RADV).

Covered: adapter routing (`adapter.rs`), capabilities probe (`lib.rs`), recorder
dispatch hot paths (`recorder.rs`), GEMM kernel resolution (`gemm.rs`), shader
build matrix (`build.rs`), and the pipeline-cache (`pcache.rs`).

Not covered (out of scope for this pass): CPU backend, Metal backend, GGUF
dequant internals, model-graph compilation, the host-side seam/runner, and the
non-Vulkan config/profiling infrastructure. Any finding that would need
profiling to confirm is marked **Needs measurement**.

---

## Findings (ranked by estimated impact)

### 1. Intel Arc XMX coopmat gated behind opt-in — defaults to nc_mmq/nc_fma/nc_fa non-coopmat tier

- `crates/infr-vulkan/src/adapter.rs:1470` (`cm8_ok` gate)
- `crates/infr-vulkan/src/lib.rs:1596` (opt-in design comment)
- `crates/infr-vulkan/src/lib.rs:3499` (`select_coopmat_shape` — 8×8×16 only
  under `allow_8x8x16`)

**What happens:** Intel Arc A770 (Mesa ANV) enumerates f16 cooperative matrix
ONLY at the 8×8×16 shape — not the production 16×16×16. The
`select_coopmat_shape` preference ladder returns `None` for 8×8×16 unless
`INFR_CM_8X8=1` is set. Without it, `caps.f16_coopmat()` is `false`, and the
adapter routes ALL prefill GEMMs through the non-coopmat tier:

- `nc_mmq`: dp4a `matmul_mmq` for quantized weights (k-quants, Q8_0)
- `nc_fma`: shared-memory fma `matmul_fma` for f16/bf16/f32 weights — no
  subgroup ops, no f16 ALU (`native_gemm_fma.comp:2899`)
- `nc_fa`: shared-memory fma flash attention (`attn_nc_fa.comp`) — fixed bm=32
  tile, no subgroup ops, ≤54 KB shared

The comment at `adapter.rs:1483` says prefill was "field-measured 10-30x" gap
before this tier existed (the fallback was scalar per-row GEMV). The nc tier
closed that gap vs scalar, but the XMX coopmat path (`native_gemm_warp`'s `_cm8`
builds) is the actual Intel tensor-core path and remains **opt-in only**.

**Why the path is hot:** Every prefill forward on Intel Arc. A user running a
Qwen3-8B prefill on an A770 without `INFR_CM_8X8=1` pays the dp4a/fma GEMM cost
instead of XMX tensor cores. The gap between nc_mmq and cm8 warp GEMM on Intel
hardware has never been measured in infr — the opt-in gating prevents it.

**Why it's gated:** The comment references "Alchemist coopmat is a
llama.cpp-documented regression" (adapter.rs:3501). However, Mesa ANV merged
`VK_KHR_cooperative_matrix` support in Mesa 24.0 (Q1 2024; see
[Phoronix](https://www.phoronix.com/news/Intel-ANV-Cooperative-Matrix)). The
regression may be stale — re-measuring on current Mesa ANV (≥24.2, which this
project's minimum Mesa recommends) could flip this to default-on.

**Fix:** Measure `INFR_CM_8X8=1` on Intel Arc A770 with Mesa ≥24.2:

```bash
INFR_CM_8X8=1 infr bench model.gguf -p 512 -n 0 -r 3
```

If pp512 throughput beats the nc_mmq default by a measurable margin AND the
parity suites pass (`cargo test -p infr-vulkan --release -- --ignored` for
GPU-gated tests), change the default: remove the opt-in gate for Intel when Mesa
version ≥ the known-good release, or promote it to default-on with an
`INFR_NO_CM_8X8` escape. The `native_gemm_warp_cm8_build_spv` function already
gates on `n%128 && k%64` — shapes it can't cover fall to nc_mmq/nc_fma
unaffected, so turning it on is additive, not a wholesale switch.

**Risk:** If the original llama.cpp regression still reproduces on current Mesa,
the opt-in stays. The investigation cost is one A770 benchmark session.

---

### 2. Vendor detection should be capability detection — `vendor_intel` gates policy, not hardware

- `crates/infr-vulkan/src/lib.rs:1928` (`vendor_intel` probe)
- `crates/infr-vulkan/src/adapter.rs:531` (dtype set per vendor)
- `crates/infr-vulkan/src/adapter.rs:726` (kernel route per vendor)
- `crates/infr-vulkan/src/adapter.rs:749` (WARPS default per vendor)

**What happens:** `vendor_intel` (probed from `vendor_id == 0x8086`) drives four
decisions, each with different capability-substitutability:

| Use site                                 | What it gates                                                                                            | Replaceable by capability?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lib.rs:1935` — `sg_pref` default        | 16 for Intel, 32 for others                                                                              | **Yes.** `subgroup_min <= 16` is equivalent — a device that can pin 16 and has min ≤ 16 gets 16; a wave32 device (min=32) can't pin 16 anyway and falls to 32. Drop `vendor_intel &&` from the condition.                                                                                                                                                                                                                                                                                                                         |
| `adapter.rs:531` — decode int8 dtype set | Intel: {Q4K,Q6K,Q2K,Q3K}. AMD: {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL}                                       | **Partially.** The Intel set is EXCLUSIVE of the legacy 32-block dtypes (Q4_0, Q5_0, Q5_1, IQ4_NL) because they were never measured on Intel — the kernels exist, they just haven't been benched. The AMD set excludes Q3_K because of a coherence cliff on mixed GGUFs (`gpu_seam_matches_cpu_qwen3_q2k` broke). Capabilities can't express "Q3_K breaks on this GPU's memory/cache hierarchy." A unified default of the AMD set (= the safe intersection) works everywhere; `INFR_MMV_MW=1` already enables all dtypes for A/B. |
| `adapter.rs:726` — `unified_mmv_row1`    | `!vendor_intel` — Intel uses legacy `native_mmv_mw.comp`, AMD uses unified `native_mmv_mrow.comp` rows=1 | **Yes.** The kernels exist for all dtypes on all vendors. The split exists only because Intel was never re-measured after the unification. A single unified path works — the kernels are bit-identical by construction.                                                                                                                                                                                                                                                                                                           |
| `adapter.rs:749` — mmv WARPS default     | 8 for Intel/all-non-Q4K, 1 for AMD Q4_K (sweep winner)                                                   | **Partially.** The Q4_K warps=1 shape was sweep-tuned on AMD only. Intel's warps=8 was the "already shipped, already tuned" shape. Could default to warps=1 for Q4_K everywhere (it's the `rm_kq_int=1` shape from llama.cpp — one output row per workgroup), but without an Intel measurement this is a blind change.                                                                                                                                                                                                            |

**The architectural principle:** Every decision the hardware _declares_ —
cooperative matrix shapes, shared memory budget, subgroup range, buffer device
address, shaderFloat16 — is already capability-gated. The four `vendor_intel`
uses above are the ONLY places that key off vendor identity, and three of four
can be replaced by capabilities or a unified default.

**Fix (priority order):**

1. **Replace `sg_pref` vendor gate with `subgroup_min`** (trivial, no
   measurement needed): drop `vendor_intel &&` from `lib.rs:1935`. If
   `subgroup_min <= 16` and 16 is pinnable, default to 16. This is pure
   capability — a future non-Intel GPU with SIMD8/SIMD16 EUs gets the right
   default automatically.

2. **Make `unified_mmv_row1` unconditional** (low risk): the unified kernel
   exists for every dtype, is bit-identical by construction, and was proven in
   `mmv_row1_bit_identical`. The only reason Intel was excluded was "no Intel
   GPU in this validation environment." Turn it on; if it regresses Intel decode
   throughput (unlikely — same math, different kernel), the `INFR_MMV_MW=0`
   escape is already there.

3. **Unify the decode int8 dtype set to the AMD default** (the safe
   intersection): `&[Q2K, Q4K, Q6K, Q4_0, Q5_0, Q5_1, Iq4Nl]`. Intel loses Q3_K
   (unmeasured win that was default-on there) and gains Q4_0/Q5_0/ Q5_1/IQ4_NL
   (unmeasured on Intel but measured wins on AMD). All of this is overrideable
   via `INFR_MMV_MW=1` / `INFR_MMV_MW=0`. The Q3_K coherence cliff on AMD is the
   reason it can't be in the unified set.

4. **Drop the WARPS vendor split**: default to `1` for Q4_K everywhere
   (llama.cpp's `rm_kq_int=1` shape) and `8` for other dtypes. The vendor no
   longer decides — the dtype does. Overrideable via `INFR_MMV_MW_WARPS`.

   End state: `vendor_intel` is removed from `Capabilities` entirely. `sg_pref`
   is capability-driven; decode policy is one unified per-dtype table; kernel
   route is unconditional. New hardware needs no new vendor flag — capabilities
   and `INFR_*` knobs cover measurement.

---

### 3. NVIDIA flash attention uses half-size tiles vs AMD — occupancy vs tile-efficiency trade-off

- `crates/infr-vulkan/src/recorder.rs:4426-4442` (bm=64 → bm=32 for sub-64 KB
  shared)
- `crates/infr-vulkan/src/recorder.rs:4651-4658` (BR=128 → BR=64 for sub-64 KB
  shared)

**What happens:** NVIDIA GPUs expose `maxComputeSharedMemorySize = 48 KB` (vs
RADV's 64 KB). The flash-warp kernel's bm=64 tile needs 58112 B (~57 KB), so
NVIDIA falls to bm=32 (29056 B). The register-O kernel's BR=128 tile needs 58880
B, so NVIDIA falls to BR=64 (29440 B).

Smaller tiles → **2× the workgroups** for the same row count. The workgroups are
independent (no cross-WG sync in the split-K reduce), so the extra WGs fill the
GPU better — but each WG does half the work, so per-WG overhead (barrier,
shared-mem init) is paid twice.

**Why this matters:** On a prefill of 512 rows × 16 heads × deep KV, the warp
path launches `(512/64)*16 = 128` WGs on AMD and `(512/32)*16 = 256` WGs on
NVIDIA. The 256 WGs likely fill an RTX 4090's 128 SMs well, but on a smaller
NVIDIA GPU (RTX 4060, 24 SMs), 256 WGs means ~10 waves — still fine. The real
question is whether the per-WG overhead dominates the math.

**Is this fixable?** The shared memory budget is a hardware constant — the
kernel tile must fit. Options:

1. **Different kernel design**: A register-heavy variant that stages less in
   shared memory (e.g., stream K/V in smaller chunks, recompute Q tiles). This
   is class-6 (kernel micro-architecture) — expensive to build and measure.
2. **Accept the tile**: bm=32 is already measured and shipped for NVIDIA. The
   `limits_probe.rs` example exists specifically to debug this.

**Recommendation:** **Needs measurement.** Profile pp512 attention time on an
NVIDIA GPU with `INFR_PROF_OPS=1`. If attention is <20% of prefill wall time,
the tile gap is not the lever. If it's >40% and the bm=32 WG count visibly
underfills the SM array, a register-O redesign becomes worth scoping.

---

### 4. Intel `nc_fa` flash attention uses fixed bm=32 tile (no larger build)

- `crates/infr-vulkan/src/gemm.rs:1008-1024` (`attn_nc_fa_spv` — fixed bm)
- `crates/infr-vulkan/src/recorder.rs:4579-4623` (`attention_prefill_nc_fa`)

**What happens:** The `attn_nc_fa` kernel (the non-coopmat flash attention
companion of nc_mmq/nc_fma) uses a hardcoded `bm=32` tile for hd≤256 and `bm=16`
for hd≤512. There is no bm=64 build — unlike the coopmat flash-warp kernel,
which has both bm=64 and bm=32 variants.

This means even if Intel Arc had ≥64 KB shared memory (it has 64 KB on
Alchemist, like RADV), the nc*fa kernel cannot use a larger tile. The coopmat
flash-warp kernel \_could* use bm=64 on Intel — but only under `INFR_CM_8X8=1`,
which also gates the GEMM tier.

**Fix:** If Intel cm8 coopmat stays opt-in (finding #1), adding a bm=64 build of
`attn_nc_fa` (gated on `max_shared_memory >= 64*FLASH_SHARED_PER_ROW`) would let
Intel Arc prefill attention use the larger tile. The existing tile-selection
logic in `recorder.rs:4439` already works — it just needs a bm=64 SPIR-V build
of the nc_fa kernel. **Cost:** one new shader variant, one new `build.rs` entry,
~30 lines in `gemm.rs`.

---

### 5. `with_padded_dst` allocates a temp buffer per non-Internal GEMM output

- `crates/infr-vulkan/src/adapter.rs:1044`

**What happens:** Every tiled GEMM (coopmat, nc*mmq, nc_fma) that writes to a
non-Internal tensor (e.g., the lm_head `logits` Output) allocates a temporary
`ceil(m/64)*64`-row buffer via `be\*.alloc_uninit`, fills it, then copies the
`m` real rows back. This allocation happens ONCE per forward on the lm_head path
— intermediate layers produce Internal tensors and skip the copy.

**Impact:** Low. The lm_head is one op near the end of the forward, allocating a
buffer sized `vocab_size * n_embd * dtype_bytes`. For Qwen3-8B (vocab 152064,
n_embd 4096, f32 output), that's ~2.5 GB — but `with_padded_dst` pads to
`ceil(152064/64)*64 = 152064` rows (exact multiple, no padding), so the temp is
exactly the output size. This allocation is comparable to the output buffer
itself and happens once per forward.

**Fix:** Could pool this buffer per-shape in the `ScratchPool` (add a tag like
`"lin_pad_dst"`). But the lm_head output changes shape between prefill (m>1) and
decode (m=1), and the pool is per-execute anyway. Not worth the complexity for
one alloc per forward.

---

### 6. `rope_pos` HashMap rebuilt per execute_static call

- `crates/infr-vulkan/src/adapter.rs:4666-4677`

**What happens:** Before the op loop, `execute_static` scans ALL graph ops to
build a `HashMap<u32, usize>` of position-tensor → rope position. This is O(ops)
with one `read_pos0` call per unique position tensor (typically 1 per forward).

**Impact:** Negligible. Ops count is <1000, the scan happens once per forward,
and the decode replay path (`execute` → `record_decode_replay`) skips it
entirely. The `read_pos0` call reads a single u32 from a host-visible buffer (no
submit/wait).

---

### 7. Chained decode `replay_n` allocates `vec![seg.cmd; n]` per chain

- `crates/infr-vulkan/src/recorder.rs:9103`

**What happens:** `execute_chain` replays `n` copies of the recorded decode
command buffer in one submit. It builds `vec![seg.cmd; n]` — a `Vec` of `n`
`vk::CommandBuffer` handles.

**Impact:** Negligible. `n ≤ 64` (bounded by `max_decode_chain`), so the
allocation is ≤512 bytes. Chained decode is the hot path (every token batch),
but this allocation is dwarfed by the GPU work it submits.

---

## Architectural note: vendor detection vs capability detection

`vendor_intel` (`caps.vendor_intel`, probed from `vendor_id == 0x8086`) is used
in four places. None of them gate whether a kernel _can_ run — they gate which
kernel _should_ run by default, based on empirical throughput measurements on
specific hardware.

The principle: capability detection answers "can this GPU run this kernel?"
Vendor detection answers "should this GPU run this kernel by default?" The
second question is better answered by a single unified default table with
`INFR_*` knobs for per-dtype/per-shape A/B measurement on any hardware.

**Why capability-first is more robust:**

- A future non-Intel SIMD8 GPU (e.g., a new ARM Mali with subgroup_min=8) would
  get `sg_pref=32` under vendor detection (not Intel → 32), but should get 16.
  Under `subgroup_min <= 16` it gets 16 automatically.
- A future NVIDIA GPU with 64 KB shared memory would get bm=32 flash tiles under
  vendor detection (if we keyed bm on vendor), but under
  `max_shared_memory_bytes >= 58112` it gets bm=64 automatically — which is what
  the existing flash tile selection already does.
- A future Intel dGPU with subgroup_min=32 would get `sg_pref=16` under vendor
  detection (Intel → 16), which is wrong. Under `subgroup_min <= 16` it gets 32
  automatically.

**What stays vendor-agnostic (already capability-driven):**

- Cooperative matrix shape selection (`select_coopmat_shape`): enumerates device
  configs, picks 16×16×16 if present, else 8×8×16 if opt-in.
- Flash attention tile size: `max_shared_memory_bytes` → bm=64 vs bm=32.
- ShaderFloat16, shaderInt8, subgroup-size-control: all probed from device
  features.
- Pipeline cache: keyed per `(vendor_id, device_id)` — but this is a disk cache
  namespace, not a kernel routing decision, and the Vulkan driver's own
  `pipelineCacheUUID` already encodes device identity.

**What the unified defaults would look like:**

| Decision              | Current (vendor-split)                                             | Proposed (capability)                                               |
| --------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| `sg_pref`             | `vendor_intel && subgroup_min <= 16` → 16, else 32                 | `subgroup_min <= 16` → 16, else 32                                  |
| `unified_mmv_row1`    | `!vendor_intel`                                                    | `true` (unconditional)                                              |
| Decode int8 dtype set | Intel: {Q4K,Q6K,Q2K,Q3K}. AMD: {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} | {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} (AMD's set = safe intersection) |
| mmv WARPS default     | Intel/all-non-Q4K → 8, AMD Q4_K → 1                                | Q4_K → 1, everything else → 8                                       |

All four are overrideable via `INFR_SG`, `INFR_MMV_MW`, `INFR_MMV_MW_WARPS`.

---

## Coverage

**Traced paths (verified call-frequency):**

| Path                                                         | Frequency                    | Traced? |
| ------------------------------------------------------------ | ---------------------------- | ------- |
| Decode record-once replay (`execute` → `replay`)             | Per token                    | ✓       |
| Chained decode (`execute_chain` → `replay_n`)                | Per n-token batch            | ✓       |
| Static prefill (`execute_static` → `lower_op` loop)          | Per forward                  | ✓       |
| Coopmat GEMM routing (`is_gemm`, warp_ok, split-K)           | Per Linear op in prefill     | ✓       |
| Non-coopmat GEMM routing (`nc_mmq`, `nc_fma`)                | Per Linear on Intel/!coopmat | ✓       |
| Flash attention routing (`flash_ok`, `nonfa_ok`, `nc_fa_ok`) | Per Attention op             | ✓       |
| Decode GEMV routing (`mmv_mw_choice`, `unified_mmv_row1`)    | Per token × Linear           | ✓       |
| Capabilities probe (`VulkanBackend::new`)                    | Once at init                 | ✓       |
| Pipeline cache (`pcache.rs`)                                 | Once at init + periodic      | ✓       |

**Not traced (known gaps):**

- **MoE dispatch overhead** (paged expert GEMMs, router/top-k readback): The
  adapter's `MoeFfn` handling splits into paged vs batched paths. Not traced in
  this pass — the perf.md campaign log already covers the batched-MoE
  dispatch-collapse win (class 4, pp512 0.59→0.91×).

- **Dense layer streaming** (`streamed_prefill_gemm`): The adapter's
  `INFR_DENSE_PAGE` path for models too large to fit in VRAM. Not traced.

- **Canvas/DiffusionGemma**: The `AttnMask::Canvas` path in attention routing.
  Not traced — perf.md notes it's benchmarked differently (dg-step vs pp/tg).

- **Metal parity**: Linux builds compile Metal sources blind. Any type changes
  in shared types (`Capabilities` fields, `Op` variants) only get verified on
  macOS CI.

**Needs measurement (cannot settle without profiling):**

1. Intel Arc `INFR_CM_8X8=1` vs nc_mmq prefill throughput (finding #1).
2. NVIDIA flash attention bm=32 vs theoretical bm=64 wall-time share (finding
   #3).
3. Impact of unifying decode int8 dtype defaults on Intel (Q3_K loss,
   Q4_0/Q5_0/Q5_1/IQ4_NL gain) and on NVIDIA (currently inherits AMD's set —
   same as the proposed unified default) (finding #2).
4. `nc_fa` bm=64 build benefit on Intel Arc (finding #4).
5. The `nc_mmq` vs `nc_fma` throughput ratio on Intel Arc — are both arms tuned?

---

## Multi-vendor sanity checks

### Intel Arc (ANV)

| Check                                                  | Status                                                                               |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `sg_pref = 16` (SIMD8 EUs)                             | ✓ (`vendor_intel && subgroup_min <= 16` — replaceable by `subgroup_min <= 16` alone) |
| `sg_pref` falls back to 32 when 16 unpinnable          | ✓                                                                                    |
| Decode GEMV: `native_mmv_mw.comp` (SG=16, WARPS-tuned) | ✓ (`unified_mmv_row1 = false` for Intel — replaceable by unconditional `true`)       |
| Decode int8 dtypes: {Q4K,Q6K,Q2K,Q3K}                  | ✓ (`mmv_int8_decode_dtypes` Intel arm — would change under unified default)          |
| Coopmat: `f16_coopmat()` = false (8×8×16 only)         | ✓ (correct; cm8 is opt-in)                                                           |
| Non-coopmat GEMM tier: `nc_mmq` + `nc_fma`             | ✓ (default — see finding #1)                                                         |
| Non-coopmat flash: `nc_fa` (`attn_nc_fa.comp`)         | ✓                                                                                    |
| Pipeline cache: keyed per `(vendor_id, device_id)`     | ✓                                                                                    |

### NVIDIA (proprietary / NVK)

| Check                                                                  | Status                                          |
| ---------------------------------------------------------------------- | ----------------------------------------------- |
| `sg_pref = 32` (warp = 32)                                             | ✓ (capability: `subgroup_min > 16` → 32)        |
| Decode GEMV: `unified_mmv_row1` (mrow, bit-identical)                  | ✓                                               |
| Decode int8 dtypes: inherits AMD's {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} | ⚠ Unmeasured — same as proposed unified default |
| Coopmat: `f16_coopmat()` = true (16×16×16)                             | ✓ (Turing+)                                     |
| Flash bm=32 (48 KB shared → 29056 B fits, 58112 B doesn't)             | ✓ (capability: `max_shared_memory_bytes`)       |
| Flash BR=64 (48 KB shared → 29440 B fits, 58880 B doesn't)             | ✓ (capability: `max_shared_memory_bytes`)       |
| Flash warp path available                                              | ✓ (bm=32 build exists; `recorder.rs:4448`)      |
| Shared memory tile selection: `max_shared_memory_bytes()` check        | ✓                                               |
| Pipeline cache: keyed per `(vendor_id, device_id)`                     | ✓                                               |

### AMD (RADV)

| Check                                                   | Status                                    |
| ------------------------------------------------------- | ----------------------------------------- |
| `sg_pref = 32` (wave32)                                 | ✓ (capability: `subgroup_min == 32` → 32) |
| Decode GEMV: `unified_mmv_row1` (mrow, bit-identical)   | ✓                                         |
| Decode int8 dtypes: {Q4K,Q6K,Q2K,Q4_0,Q5_0,Q5_1,IQ4_NL} | ✓ (measured on 7900 XTX)                  |
| Coopmat: `f16_coopmat()` = true (16×16×16)              | ✓ (RDNA3+)                                |
| Flash bm=64 (64 KB shared → 58112 B fits)               | ✓ (capability: `max_shared_memory_bytes`) |
| Flash BR=128 (64 KB shared → 58880 B fits)              | ✓ (capability: `max_shared_memory_bytes`) |
| Flash warp path available (bm=64 + bm=32 builds)        | ✓                                         |
| Compute unit count probe (shader engine count)          | ✓ (`VK_AMD_shader_core_properties`)       |
| Integrated GPU detection + chunk sizing                 | ✓                                         |
| Pipeline cache: keyed per `(vendor_id, device_id)`      | ✓                                         |

---

## Summary

**Highest ROI, ready to act:**

1. **Measure Intel Arc `INFR_CM_8X8=1` on current Mesa ANV.** If the
   llama.cpp-documented regression is fixed in Mesa ≥24.2, flipping this to
   default-on gives Intel Arc users the XMX tensor-core GEMM path — the single
   largest lever for Intel prefill throughput.

2. **Remove `vendor_intel` in favor of capability detection.** Four-step plan:
   (a) replace `sg_pref` vendor gate with `subgroup_min <= 16`; (b) make
   `unified_mmv_row1` unconditional; (c) unify the decode int8 dtype set to
   AMD's safe default; (d) drop the WARPS vendor split. End state: zero vendor
   flags in `Capabilities`, every routing decision keyed off capabilities the
   device declares, `INFR_*` knobs for A/B measurement on any hardware. New GPUs
   need no new vendor flags.

3. **Profile NVIDIA flash attention tile overhead.** If bm=32 attention is a
   significant share of NVIDIA prefill wall time, a register-O redesign that
   fits in 48 KB shared could be the next class-6 lever. Until profiled, accept
   the current bm=32 as correct.

**Lower priority, but worth tracking:**

4. Add bm=64 build of `attn_nc_fa` for Intel Arc (if cm8 stays opt-in).
5. Measure the unified decode int8 dtype defaults on Intel and NVIDIA. Requires
   access to those GPUs.
