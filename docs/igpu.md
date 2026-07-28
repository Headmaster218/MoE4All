# iGPU support — integrated-GPU correctness campaign

Campaign log for running infr on integrated GPUs (this box's AMD Ryzen 9 9950X3D
iGPU / RADV RAPHAEL_MENDOCINO, and the Intel iGPU / Strix Halo-class APU target
class generally). Consolidated from the working notes; kept for the durable
findings and the root-cause trail. Point-in-time operational state (box-wedged
alarms, per-reboot recipes) has been dropped.

## Status — Phase 1 (correctness) COMPLETE

Every README-table model that fits loads and generates coherent text on the
iGPU; the seam suite is **25/25 on the non-coopmat tier**. UMA support landed
(`4771cda`). Remaining iGPU work is **Phase 2 (perf) only** — see the end.

- **Phase 1 — correctness first.** Every model in the README perf table runs on
  the iGPU: loads, generates coherent output, passes the tests. Done.
- **Phase 2 — perf.** Beat llama.cpp Vulkan on the iGPU. Not started; blocked on
  nothing but priority.

## The hardware (this box)

- `GPU1` = **AMD Ryzen 9 9950X3D integrated (RADV RAPHAEL_MENDOCINO)**, RDNA2,
  `--dev Vulkan1`.
- `GPU0` = RX 7900 XTX (RDNA3) — the discrete card everything else is tuned for.

Known iGPU facts (`vulkaninfo`), each a suspected breakage source the survey had
to confirm or kill:

| property                     | iGPU (Vulkan1) | dGPU (Vulkan0) | verdict                                                                                    |
| ---------------------------- | -------------- | -------------- | ------------------------------------------------------------------------------------------ |
| `subgroupSize`               | base 64        | base 64        | `minSubgroupSize=32` on BOTH; sg32 pins fine — **not a problem**                           |
| dedicated VRAM               | ~2 GB carveout | 24 GB          | carveout is NOT the ceiling — models live in GTT (below)                                   |
| system RAM                   | shared DDR     | —              | **the key lever** (UMA, below)                                                             |
| `maxMemoryAllocationSize`    | ~4 GiB         | ~4 GiB         | identical — not iGPU-specific                                                              |
| `maxStorageBufferRange`      | ~4 GiB         | ~4 GiB         | >4 GiB SSBO silent-zeros trap; not observed here                                           |
| `maxComputeSharedMemorySize` | 64 KB          | 64 KB          | same                                                                                       |
| cooperative matrix           | **absent**     | present        | RDNA2 has NO coopmat (`f16cm:n i8cm:n`) → fallback tier; a perf hit, not a correctness one |

## The UMA insight (the solid deliverable)

On an APU the "VRAM" carveout is not a real boundary — it is the same physical
DDR. Weights in host-visible/GTT read at the same bandwidth as the carveout. The
goal is NOT to squeeze models into 2 GB; it is to make placement
carveout-agnostic and let the model live in system RAM.

The heap table on this iGPU:

    heap[0]  10.73 GiB   (no flags)
    heap[1]  21.47 GiB   DEVICE_LOCAL      <- every GpuOnly buffer lands here
    type[0]  heap=1  DEVICE_LOCAL
    type[2]  heap=0  HOST_VISIBLE|HOST_COHERENT
    type[3]  heap=1  DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT

- `10.73 + 21.47 = 32.20 GiB` = EXACTLY `vram_total` (2 GiB carveout) +
  `gtt_total` (30.20 GiB). **RADV synthesizes the "DEVICE_LOCAL" heap as ~2/3 of
  (carveout + GTT).** It is neither VRAM nor the carveout, and it is **not
  enforced** — 41 GiB allocated from the "21.47 GiB" heap all succeeded, landing
  in GTT. infr's weights were already in system RAM on this device.
- The dGPU has a superficially identical two-heap shape, but over-committing
  there spills across PCIe (a real bandwidth cliff). **So the device-local-only
  guard is LOAD-BEARING on discrete and must stay.** That asymmetry is the whole
  design.
- Widening the budget is not enough alone: gpu-allocator resolves `GpuOnly` to
  the first DEVICE_LOCAL type and never falls back, so a full heap silently
  oversubscribes until a submit dies with "Not enough memory for command
  submission". Overflow must be PLACED (`probe_uma_overflow_type` + spill), not
  merely counted.

Verified on the real iGPU (`--dev Vulkan1`): Gemma-4-31B UD-Q5_K_XL loads
(UNIFIED MEMORY banner, 32.20 GiB budget, weights 20.37 GiB fit) and decodes
coherently — slow (prefill ~3 t/s, decode ~0.4 t/s on 2 CU) but correct.

## The blocker that dominated the campaign: the per-submit watchdog

**The GPU hang watchdog is armed per SUBMIT, and infr recorded an entire forward
pass into ONE command buffer.** On the 2-CU iGPU a Qwen3-8B prefill chunk is
~2.05 s of GPU work in a single job, and the device kills a job at ~2.06 s — a
~1% margin, hence an intermittent coin-flip hang (~3 fails in 15 long runs).

Diagnosis trail:

- Nailed by timeline correlation: the kernel resets the ring 2.06 s after the
  submit; the reset FORCE-SIGNALS the fence, so `vkQueueWaitIdle` returns
  _success_ — a killed chunk reports a plausible ~2046 ms and the process only
  dies on the NEXT submit. That is why the failure never pointed at itself.
  `INFR_PROF_OPS` timestamps confirm the GPU is busy the whole window: a job too
  long, NOT a hung shader.
- **Rows are the wrong knob.** Submit time is nearly flat in rows (a forward is
  757 dispatches + 684 barriers + one weight sweep, none of which shrink with
  rows): 8→1163 ms, 16→1199, 32→1861, 64→2047, 128→2048. A 16× row cut buys only
  1.76×. This is why an early 128-row prefill cap could not fix it.

**The fix (`0ea6600`, nit `0c14f26`): bound DISPATCHES PER SUBMIT** and cut the
forward into several back-to-back command buffers (same work, `finish_nowait`;
the watchdog only ever sees short jobs). Seeded from device class (**unlimited
on discrete — dGPU path untouched**) then re-tuned from each forward's measured
per-dispatch cost so it holds on unsurveyed hardware. `INFR_SUBMIT_DISPATCHES`
overrides. Chained decode (`replay_n`) is clamped by the same budget in the
runner. **llama.cpp does the same thing** (`max_nodes_per_submit`, plus a
CU-scaled FLOP budget — the more general bound, noted as the upgrade path).

Proof: 10/10 clean long runs (was ~3 fails in 15), gemma-3-12b 3/3, **dGPU
unchanged** (pp512 3562.7→3547.1, tg128 143.5→143.1 = noise), gpu_seam 25/25.

**Null results — disproved with evidence, do NOT re-open:** wave64 codegen
(`RADV_PERFTEST=cswave32` still hangs 3/3); VRAM overcommit/eviction (only 0.48
MiB `amdgpu_bo_move` across a run); clock throttling (SCLK pinned 2200 MHz
through the hang); a compute-only queue family (makes it DETERMINISTIC — the
`comp` ring budget is TIGHTER). `RADV_DEBUG=hang` is a **red herring**:
syncshaders serializes the IB so the job itself blows the budget, "naming"
whatever shader ran at the 2 s mark.

### The dense-decode routing bug (found while chasing the above)

A branch once routed dense single-token decode to `execute_static` (the
prefill/paged path) to dodge the unsplittable replay tape. That skips the
decode-specific `record_decode_replay` (`_dyn` params-driven pos kernels +
self-advancing ring) ⇒ wrong logits ⇒ repeated-token garbage (`1.1.1…`,
`))))))`). **Reproducible on the dGPU** with `INFR_SUBMIT_DISPATCHES=64`, across
gemma-4 AND qwen3-8b ⇒ not gemma-specific, breaks all dense decode. The correct
fix was to make the **replay path itself splittable**
(`RecordedCmd → Vec<RecordedSegment>`, separate submits) keeping the decode
kernels — landed with the UMA slice (`4771cda`).

## Survey — the full README table on `--dev Vulkan1`

Prompt "What is the capital of France? Answer in one short sentence.",
`INFR_MAX_NEW=40`. Banner:
`INTEGRATED (cu:2) — prefill chunk 128 rows, forward split every 128 dispatches`.

| #   | Model                 | Quant      | Loads | Coherent | Notes                                      |
| --- | --------------------- | ---------- | ----- | -------- | ------------------------------------------ |
| 1   | Gemma-4-26B-A4B (MoE) | UD-Q4_K_M  | yes   | yes      | pager 30/30 PAGED; tg 2.3 t/s              |
| 2   | Qwen3.6-27B           | Q4_K_M     | yes   | yes      | dense resident; tg 1.0 t/s (slowest)       |
| 3   | Qwen3-30B-A3B (MoE)   | Q4_K_M     | yes   | yes      | pager 48/48; ctx clamp 40960→24893         |
| 4   | Gemma-4-31B           | UD-Q5_K_XL | yes¹  | yes¹     | ¹with UMA slice; VRAM guard blocks pre-UMA |
| 5   | Ornith-1.0-35B        | Q4_K_M     | yes   | yes      | DeltaNet; tg 2.2 t/s                       |
| 6   | Qwen3.6-35B-A3B (MoE) | UD-IQ3_S   | yes   | yes      | resident 12.68 GiB; tg 3.7 t/s             |
| 7   | Qwen3.6-35B-A3B (MoE) | UD-Q4_K_M  | yes   | yes      | tg 1.8 t/s                                 |
| 8   | DiffusionGemma-26B    | Q4_K_M     | yes   | yes²     | ²after the DG seam test fix (`feb61b5`)    |
| 9   | Ternary-Bonsai-1.7B   | Q2_0_g64   | yes   | yes      |                                            |
| 10  | Ternary-Bonsai-4B     | Q2_0_g64   | yes   | yes      |                                            |
| 11  | Ternary-Bonsai-8B     | Q2_0_g64   | yes   | yes      |                                            |
| 12  | Ternary-Bonsai-1.7B   | plain Q2_0 | no    | —        | loader; device-independent known limit     |
| 13  | Ternary-Bonsai-1.7B   | PQ2_0      | no    | —        | `unsupported: ggml type 142` (not impl)    |

### Resolved failure classes

- **DiffusionGemma non-coopmat (`feb61b5`) — was an over-strict TEST, not a
  kernel bug.** The DG prefill last-token top-5 overlap assert tripped on a
  near-tie (whole-vocab cosine 0.80 nc / 0.811 coopmat — textbook int8<f16<f32
  laddering). Fix: gate on the distribution (`overlap || cos > 0.78`, keep hard
  `cos > 0.7`), mirroring the sibling `_denoise` check; tighten `_denoise` floor
  0.7→0.75 (measured healthy min 0.789). gpu_seam now 25/25 on BOTH tiers.
- **`--dev` was decorative (`c05b526`).** `let _ = dev;` — the backend hardcoded
  "first discrete GPU", so `infr bench --dev Vulkan1` silently benched the dGPU.
  Now real, with a hard error on an unknown device.
- **Vacuous-green test harness (`0a353c6`).** Seam tests reached the GPU via
  `gpu_available()` → `VulkanBackend::new().is_ok()`, so a failing device made
  them SKIP SILENTLY and report "passed" (`INFR_DEV=Vulkan9` → "1 passed" in
  0.02 s). Now a hard failure when `INFR_DEV` is set explicitly.

### Device-independent, not iGPU bugs

- Bonsai plain `Q2_0` (34B/128 elems) vs `Q2_0_g64` (18B/64) declare the
  **identical `ggml type 42` with no metadata discriminator**, so upstream made
  them indistinguishable — infr hardcodes g64 (`infr-gguf/src/lib.rs`). README
  pins `:Q2_0_g64`. `PQ2_0` is ggml type 142, not implemented.
- Qwen3-0.6B-Q2_K degenerates into repetition on the dGPU too = model quality.

## Operational rule (learned the hard way)

**NEVER let a `timeout`/SIGTERM kill `infr` mid-GPU-submit on the iGPU.** It
leaves an unkillable D-state task in `dma_fence_wait` holding the device, and
RADV then drops the iGPU from enumeration until reboot
(`no such Vulkan device`). Give iGPU runs a generous `timeout` (they are 30-60×
slower than the dGPU — a long prompt + decode is minutes) and never wrap them in
a tool call that can time out first. GPU runs are **serial**; on UMA, two big
models concurrently = GTT exhaustion = wedge.

## Phase 2 (perf) — remaining, order

1. **No-coopmat prefill tier is the whole gap** (pp 11-200 tok/s vs thousands on
   the dGPU). RDNA2 has no cooperative matrix; prefill falls to the
   scalar/f16-warp ladder. This is the dominant Phase-2 lever.
2. **UMA placement follow-ups:** `unified_memory` was hardcoded `false` in the
   Vulkan caps; the host→staging-ring→device copy is a pointless DDR→DDR pass on
   an APU; the VRAM guard should treat GTT as evictable system RAM. (Landed in
   the UMA slice; re-audit for headroom.)
3. **Revisit `INFR_UBATCH` on the iGPU.** Rows barely affect submit time now
   that submits are duration-bounded, so the 128-row prefill cap is costing
   throughput — do the FLOP-aware budget (llama.cpp's shape) before raising it.
4. Class-3 wart: default-ctx clamp collapsing to 1024 on many models.
