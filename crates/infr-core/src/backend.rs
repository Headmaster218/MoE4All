//! The compute backend seam — the ONLY device-aware trait. Everything above is generic over it.
//!
//! Object-safe on purpose so the engine can hold `Arc<dyn Backend>` and stay blind to whether
//! Vulkan / CPU / Metal is underneath. See docs/plan.md.
//!
//! ## Execution model
//!
//! 1. The model builds a [`Graph`] (op-list) once per forward *shape* and `compile`s it to a
//!    [`Plan`] (pipelines, buffer-size planning, recorded command buffer for Vulkan).
//! 2. The model owns long-lived buffers (weights, KV cache) and per-step input/output buffers,
//!    and binds them to the graph's `Input`/`Weight`/`Output` handles via [`Bindings`].
//! 3. `execute(plan, bindings)` allocates the graph's `Internal` scratch, runs the ops, and
//!    writes results into the bound `Output` buffers. `Internal`/`Output` scratch is transient.

use crate::error::{Error, Result};
use crate::graph::Graph;
use crate::tensor::TensorId;

/// Bounds-check a [`Backend::copy_buffer`] prefix request: the copied prefix (`bytes`) must fit
/// inside the source buffer (`src_len`). Returns a `Backend` error rather than letting a downstream
/// slice/download panic on an oversize `bytes`. Extracted as a pure helper so the check is unit-
/// testable without standing up a full mock backend.
pub fn check_copy_bytes(bytes: usize, src_len: usize) -> Result<()> {
    if bytes > src_len {
        return Err(Error::backend(format!(
            "copy_buffer: requested prefix of {bytes} bytes exceeds source buffer of {src_len} bytes"
        )));
    }
    Ok(())
}

/// The 16x16x16 (M,N,K) cooperative-matrix tile every production coopmat shader is built for
/// (RADV/RDNA3+, NVIDIA — every `coopmat<...,16,16,...>` declaration across
/// gemm_warp/native_gemm*/attn_*/deltanet_prep).
pub const COOPMAT_TILE_16: (u32, u32, u32) = (16, 16, 16);
/// The 8x8x16 (M,N,K) tile Intel Arc (Mesa ANV, XMX) enumerates for f16 — the ONLY non-16x16x16
/// shape any kernel here is built for, and only `native_gemm_warp`'s `_cm8` variants at that
/// (opt-in via `INFR_CM_8X8=1`; see the Vulkan backend's shape selection).
pub const COOPMAT_TILE_8: (u32, u32, u32) = (8, 8, 16);

/// Device capabilities the graph compiler queries to pick fast vs fallback kernels.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub name: String,
    // ── per-type compute capabilities ───────────────────────────────────────────────────────────
    // Naming: `f16`/`f8`/`i8` are the infr spelling for the numeric types (== fp16/fp8/int8). Two
    // ORTHOGONAL axes, expressed as flat `<type>` + `<type>_<primitive>` bools (a device can have
    // one without the other — e.g. coopmat with only f16 components, or f16 ALU without coopmat):
    //   • `<type>`          — the device supports this type for scalar/vector storage & math
    //                         (f16 = shaderFloat16, f8 = the float8 storage/convert ext,
    //                         i8 = shaderInt8). Drives the scalar / non-coopmat fallback kernels.
    //   • `<type>_<prim>`   — the type is usable by a matrix/dot PRIMITIVE: `_coopmat` = accepted as
    //                         a cooperative-matrix component (enumerated from the device's coopmat
    //                         config list); `i8_dot` = packed dp4a integer dot.
    // infr's coopmat GEMM dequants any on-disk weight dtype to f16 IN-SHADER before the multiply,
    // so today's GEMM keys off `f16_coopmat` specifically; `f8_coopmat` is the (pending) fp8 tier.
    // (No `f32` capability flag: f32 ALU is universal — every device has it — so a flag would gate
    // nothing. f32 weights that DO ship in some GGUFs, e.g. qwen3moe's router, run on the
    // unconditional `linear_f32` GEMV. And no target GPU enumerates an f32-operand coopmat config.)
    /// f16 (== fp16) scalar/vector ALU (`shaderFloat16`) — the non-coopmat f16 warp/GEMV fallback.
    pub f16: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for f16 components (f16×f16→f16/f32),
    /// picked from the device's enumerated config list by preference order: [`COOPMAT_TILE_16`]
    /// first (the production shape every coopmat shader is built for), then [`COOPMAT_TILE_8`]
    /// (Intel Arc XMX — only honored when the `INFR_CM_8X8=1` opt-in is set, since only the
    /// `native_gemm_warp` `_cm8` builds exist at that shape). `None` = no usable f16 coopmat
    /// (route to the non-coopmat tiers). Call sites that mean "the full 16x16x16 kernel set is
    /// live" use the [`f16_coopmat`](Self::f16_coopmat) accessor, which is EXACTLY the pre-shape-
    /// table boolean.
    pub coopmat_f16: Option<(u32, u32, u32)>,
    /// f8 (== fp8, E4M3/E5M2) storage/convert support (`VK_EXT_shader_float8`-class). infr has no
    /// scalar f8 math path today; this exists for symmetry / future use. False on RDNA3.
    pub f8: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for f8 (E4M3/E5M2) A/B components —
    /// the fp8 coopmat GEMM tier. Detected by enumerating the device's coopmat configs; NOT a
    /// subset of `coopmat_f16` (a unit can do f16 components but not f8). Only
    /// [`COOPMAT_TILE_16`] is ever selected (no f8 8x8x16 kernels exist). `None` on all
    /// pre-RDNA4/pre-Ada HW.
    pub coopmat_f8: Option<(u32, u32, u32)>,
    /// i8 (== int8) storage & math in shaders (`shaderInt8`) — the scalar integer path.
    pub i8: bool,
    /// The device advertises packed i8 dot-product (`VK_KHR_shader_integer_dot_product` /
    /// `dotPacked4x8AccSat`, core in Vulkan 1.3) — the decode i8 `mmv` path's dp4a accumulate. A
    /// SEPARATE primitive from coopmat (hence not `coopmat_i8`). False = route to the scalar
    /// dequant GEMV (needs no extension). Independent of `f16`/`coopmat_f16`.
    pub i8_dot: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for SINT8×SINT8→SINT32 components
    /// (enumerated from the device's coopmat config list, same discipline as `coopmat_f16`; only
    /// [`COOPMAT_TILE_16`] is ever selected — the shape every int8 coopmat shader is hardcoded
    /// to). DETECTION ONLY — Some on this RX 7900 XTX (Mesa 26.1.4), but int8 coopmat previously
    /// HUNG the GPU on an older Mesa (commit ad82a77; the standalone `coopmat_int8_test` harness
    /// confirmed the fix). The kernel is therefore an ADDITIONAL opt-in gate on top of this:
    /// the adapter only dispatches it when `INFR_I8_COOPMAT=1` is also set (default off) — see
    /// adapter.rs's `Op::Linear` GEMM branch.
    pub coopmat_i8: Option<(u32, u32, u32)>,
    /// bf16 (bfloat16) scalar storage/convert support (`VK_KHR_shader_bfloat16`-class). Distinct
    /// from `f16` (IEEE half): same 16 bits but 8 exponent / 7 mantissa. False on RDNA3.
    pub bf16: bool,
    /// The (M,N,K) tile the cooperative-matrix unit accepts for bf16 components
    /// (bf16×bf16→bf16/f32). Enumerated from the device's coopmat config list (raw
    /// `VK_COMPONENT_TYPE_BFLOAT16_KHR`, confirmed on RDNA4/Navi44). NOT a subset of
    /// `coopmat_f16`. Only [`COOPMAT_TILE_16`] is ever selected. `None` on all pre-RDNA4 HW.
    pub coopmat_bf16: Option<(u32, u32, u32)>,
    /// Supported subgroup-size range (`VkPhysicalDeviceSubgroupSizeControlProperties`). The coopmat
    /// GEMM pins `requiredSubgroupSize = 32` (RDNA3 wave32); a device whose range excludes 32 can't
    /// run the pinned kernel and must fall back to a non-pinned variant. `(0, 0)` =
    /// subgroup-size-control unsupported (can't pin at all — use the driver's default subgroup).
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    /// PREFERRED pinned subgroup size for the bandwidth-critical decode GEMV/reduction kernel
    /// family (the `_sg16` SPIR-V twins: `native_gemv_sg` / `native_gemv_id_multi_sg` /
    /// `native_mmv_mw` / `mul_mat_vec_q` / `quant_q8_row`). Capability-driven: devices where
    /// `subgroup_min <= 16` (SIMD8/SIMD16 EUs — Intel Arc, future small-subgroup GPUs) default
    /// to 16, everything else (RADV wave32, NVIDIA warp32) defaults to 32. Every OTHER pinned-32
    /// kernel (rmsnorm/softmax/coopmat GEMM/attention…) is unaffected by this field.
    /// `INFR_SG=16|32` overrides for A/B; a request the device can't pin falls back to 32.
    /// 0 on backends without subgroup pinning (CPU/Metal — the field gates Vulkan shader picks).
    pub sg_pref: u32,
    /// `VkPhysicalDeviceProperties.deviceType == INTEGRATED_GPU` — an iGPU/APU sharing the CPU's
    /// memory controller and carrying one to two ORDERS OF MAGNITUDE less compute than the
    /// discrete cards every dispatch shape here is tuned for (a Ryzen 9950X3D's RDNA2 iGPU is
    /// ~2 CU against a 7900 XTX's 96).
    ///
    /// The load-bearing consequence is the GPU WATCHDOG, not throughput: amdgpu resets a `gfx`-ring
    /// job that runs past ~10 s (`ring gfx_0.0.0 timeout` -> `VK_ERROR_DEVICE_LOST`), and infr
    /// submits ONE command buffer per forward pass — so a whole prefill chunk is a SINGLE watchdog
    /// job. A default 1024-row chunk measures 0.78-1.12 s of GPU on a 7900 XTX in the NON-COOPMAT
    /// tier an iGPU is forced onto (RDNA2 has no cooperative matrix), which at the ~32x slowdown
    /// calibrated from a real iGPU run lands at 25-36 s — a guaranteed TDR. Devices that set this
    /// get a smaller default prefill chunk (see the seam's `ubatch_rows`), the one knob that
    /// bounds per-submit GPU time. Discrete devices leave it false and every shape stays as tuned.
    pub integrated: bool,
    /// Shader/compute-unit count when the device advertises one (`VK_AMD_shader_core_properties`),
    /// else 0 = UNKNOWN. ADVISORY only — it scales the `integrated` chunk budget where present.
    /// Never gate correctness on it: most drivers report nothing.
    pub compute_units: u32,
    /// `VkPhysicalDeviceVulkan12Features.bufferDeviceAddress` — the device can hand a shader a
    /// 64-bit `VkDeviceAddress` for a buffer (`GL_EXT_buffer_reference`), so a kernel may read a
    /// buffer LARGER than `maxStorageBufferRange` (one SSBO binding's ~4 GiB reach on RADV). infr's
    /// paged-MoE expert arena REQUIRES this: a per-role pool now spans as much VRAM as the budget
    /// allows, addressed by a raw pointer rather than a bound SSBO. Core in Vulkan 1.2, so every
    /// real 1.3 target sets it — the Vulkan backend hard-errors at init when it is absent rather
    /// than keeping a u32 fallback. Backends without a device-address concept (CPU/Metal) leave
    /// this false; nothing there reads it.
    pub buffer_device_address: bool,
    /// `maxComputeSharedMemorySize` — the per-workgroup shared-memory budget. Vulkan only guarantees
    /// 16 KB; RADV gives 64 KB, NVIDIA 48 KB, MoltenVK/mobile often 32 KB. The flash-attention tile
    /// height is picked to fit this (and flash is skipped entirely if even the smallest tile won't).
    pub max_shared_memory_bytes: u32,
    pub unified_memory: bool,
    /// The backend records a single-token decode graph ONCE and replays it per token, reading the
    /// position from a device-side params buffer (Vulkan seam) instead of the graph's baked `pos`.
    /// When set, the runner may compile the decode graph once (pos=0) and reuse it across the whole
    /// decode loop. Backends that read the baked `pos`/`kv_len` (CPU interpreter) leave this false.
    pub decode_replay: bool,
    /// The backend prefers the dense FFN's gate+up weights CONCATENATED into one `[2*nff, ne]`
    /// tensor (one GEMV/GEMM + `Op::GatedActFused` instead of two Linears + `Op::GatedAct`).
    /// Costs an owned copy of both weights at load, so zero-copy mmap backends (CPU) leave this
    /// false and keep the separate-tensor form.
    pub combined_gu: bool,
    /// The backend executes [`crate::Op::EmbedGather`] (dequantize embedding-table rows selected
    /// by a device-side id buffer). When set, the runner uploads TOKEN IDS (4 bytes each) instead
    /// of host-dequantized f32 embedding rows — decode feeds 4B/token, prefill 4B/token instead
    /// of `4*n_embd`. Backends without a table-row dequant kernel leave this false and keep the
    /// host embed path.
    pub embed_gather: bool,
    /// The backend executes [`crate::Op::Sample`] (device-side temperature + top-k + top-p
    /// sampling; only the 4-byte token id reads back). False = the runner downloads the logits
    /// and samples on the host.
    pub gpu_sample: bool,
    /// The backend executes [`crate::Op::Argmax`] with `rows > 1` (per-row greedy argmax over
    /// `[rows, n]` logits — the MTP speculative-verify accept reads back m 4-byte ids instead of
    /// the m×vocab logits, issue #31). Every backend handles `rows == 1` (the decode loop);
    /// backends whose kernel is single-row-only (Metal) leave this false and the runner keeps
    /// the host-logits accept path there.
    pub argmax_rows: bool,
    /// The backend executes [`crate::Op::ArgmaxProb`] (fused single-row argmax + softmax top-1
    /// probability — the MTP draft-loop accept, issue #33 follow-up: only 8 bytes read back
    /// instead of the `[vocab]` logits + host argmax/softmax scan). Backends without the kernel
    /// leave this false and the MTP driver keeps the host-logits `top1_softmax` path.
    pub argmax_prob: bool,
    /// The backend executes [`crate::Op::GatedRmsNorm`] (fused per-head RMSNorm + SiLU gate
    /// multiply — qwen35's DeltaNet z-gate, one dispatch instead of `QkNorm`→`GatedAct`'s two).
    /// False = the runner keeps emitting the split pair (identical math, one extra
    /// read-after-write barrier on backends that have one).
    pub gated_rmsnorm: bool,
    /// The backend treats every KV cache buffer as a RING over its allocated row count: `WriteKv`
    /// lands row `pos % cap_rows` and `Attention` reads key/value position `j` at row
    /// `j % cap_rows`, where `cap_rows = declared cache elements / row width`. A full-context
    /// cache (`cap_rows >= every kv_len`) makes the modulo an identity, so setting this only
    /// matters when the runner allocates a sliding-window layer's cache at window size instead of
    /// full context (see the seam's SWA ring sizing). Backends whose kernels index rows directly
    /// by position (Metal) leave this false and get full-context allocations for every layer.
    pub kv_swa_ring: bool,
}

/// The DEFAULT prefill chunk (rows) for an INTEGRATED GPU with `cu` compute units (0 = unknown).
///
/// This bounds the cost of ONE DISPATCH, which is the one thing a backend cannot split: the GPU
/// hang watchdog kills a whole submit, and a submit can be cut into more command buffers (that is
/// [`Backend::submit_dispatch_cap`]'s job), but a single dispatch is indivisible, so its own
/// runtime has to stay well inside the budget on its own.
///
/// It is NOT the watchdog bound. It used to be sold as one, and that was measured WRONG on the
/// surveyed 2-CU RDNA2 part (Ryzen 9 9950X3D / RAPHAEL_MENDOCINO), where a Qwen3-8B Q4_K_M
/// prefill chunk is one command buffer of 757 dispatches and 684 barriers, and its GPU time is
/// nearly FLAT in the row count:
///
///   rows |   8    |  16    |  32    |  64    | 128
///   ─────┼────────┼────────┼────────┼────────┼───────
///   time | 1163ms | 1199ms | 1861ms | 2047ms | 2048ms
///
/// A 16x cut in rows buys 1.76x of submit time and bottoms out around 1.15 s, because what the
/// forward actually costs on a 2-CU part is per-dispatch launch + barrier overhead and one sweep
/// of the model's weights — neither of which shrinks with rows. Since that part kills a submit at
/// ~2.06 s, NO row cap can put a whole-forward command buffer safely under the watchdog. Shrinking
/// rows here to dodge a TDR is cargo cult; splitting the submit is the fix.
pub fn integrated_ubatch_rows(cu: u32) -> usize {
    /// The surveyed iGPU (RDNA2 Raphael/Mendocino) — the part the 128-row figure is calibrated on.
    const BASE_CU: u32 = 2;
    /// Rows at BASE_CU: the largest chunk whose heaviest single dispatch (`native_gemm_mmq_q6k`,
    /// measured at 15.0 ms/dispatch here) stays two orders of magnitude inside the watchdog.
    const BASE_ROWS: usize = 128;
    /// Never exceed the discrete default — this is a hardening cap, not a tuning knob.
    const MAX_ROWS: usize = 1024;
    if cu == 0 {
        return BASE_ROWS; // unknown: assume the weakest part we have measured
    }
    (BASE_ROWS * (cu as usize).div_ceil(BASE_CU as usize).max(1)).clamp(BASE_ROWS, MAX_ROWS)
}

/// Target GPU time for ONE submit, in nanoseconds — the budget the submit splitter aims at.
///
/// The surveyed iGPU kills a submit at ~2.06 s (measured: a job submitted at T is ring-reset at
/// T+2.06 s, and its fence is force-signaled so the wait even returns "successfully"). 250 ms
/// leaves an ~8x margin, which absorbs the run-to-run jitter that made the un-split 2.05 s forward
/// a coin flip against that budget.
pub const SUBMIT_BUDGET_NS: u64 = 250_000_000;

/// Initial cap on DISPATCHES PER SUBMIT for a device that has not been measured yet — the value
/// that has to protect the very first forward, before [`submit_cap_from_measurement`] has anything
/// to go on. `0` = unlimited (never split).
///
/// Discrete GPUs get 0: a whole forward there is tens of milliseconds, two orders of magnitude
/// inside any watchdog, and splitting it would only add barriers and submit overhead to the tuned
/// path. Integrated GPUs get a real cap, because they are the parts whose forward is measured in
/// SECONDS.
///
/// 128 comes from the surveyed 2-CU part: 757 dispatches ≈ 2.05 s ⇒ ~2.7 ms/dispatch ⇒ 128
/// dispatches ≈ 0.35 s, inside [`SUBMIT_BUDGET_NS`] with room to spare. It is deliberately NOT
/// scaled by CU count: a bigger iGPU takes proportionally more rows (`integrated_ubatch_rows`), so
/// its per-dispatch cost lands in the same place, and the same dispatch count lands in the same
/// segment time. Whatever this gets wrong on an unsurveyed part, the measured feedback loop in
/// [`submit_cap_from_measurement`] corrects after one forward.
///
/// PRIOR ART (independent confirmation of both the diagnosis and the shape of the fix):
/// llama.cpp's Vulkan backend splits its graph the same way, and its comment says the same thing —
/// "On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid
/// this" (ggml-vulkan.cpp). It submits on `submitted_nodes >= max_nodes_per_submit` (100, every
/// device) OR on a per-submit FLOP budget that scales with the CU count
/// (`2e9 * shader_core_count` for a non-GCN AMD part under 24 CUs — i.e. ~4 GFLOP/submit on this
/// 2-CU iGPU). Bounding WORK per submit rather than dispatches is the more general form of this
/// and would also stay correct if a caller forced a much larger `INFR_UBATCH` on an iGPU (each
/// dispatch then gets heavier while the count stays put); today that case is caught one forward
/// late by the measured loop instead. Worth adopting if this ever needs to be tighter.
pub fn initial_submit_dispatch_cap(integrated: bool) -> usize {
    if integrated {
        128
    } else {
        0
    }
}

/// The measured feedback loop behind the submit splitter: given how long the last forward's
/// dispatches ACTUALLY took on this device (`elapsed_ns` across `dispatches` of them), return the
/// dispatch cap that would have put each submit at [`SUBMIT_BUDGET_NS`]. `0` = unlimited.
///
/// This is what makes the splitter hardware-agnostic instead of a table of magic numbers: it does
/// not care whether the GPU is a 2-CU iGPU, a 96-CU dGPU, or something that does not exist yet —
/// it divides the budget by the per-dispatch cost this device just demonstrated. A fast GPU
/// produces a cap far larger than any forward's dispatch count, which never splits anything (and
/// is reported as unlimited); a slow one converges to segments of ~[`SUBMIT_BUDGET_NS`].
pub fn submit_cap_from_measurement(elapsed_ns: u64, dispatches: usize) -> usize {
    /// Never split below this — a submit is not free, and a cap this small means something other
    /// than dispatch cost (a first-forward pipeline compile, a stalled host) dominated the sample.
    const MIN_CAP: usize = 16;
    /// A cap at or above this covers any forward we record, so it means "never split".
    const UNLIMITED_ABOVE: usize = 1 << 20;
    if dispatches == 0 || elapsed_ns == 0 {
        return 0;
    }
    let ns_per_dispatch = (elapsed_ns / dispatches as u64).max(1);
    let cap = (SUBMIT_BUDGET_NS / ns_per_dispatch) as usize;
    if cap >= UNLIMITED_ABOVE {
        0
    } else {
        cap.max(MIN_CAP)
    }
}

/// A forward this slow (whole pass, nanoseconds) is close enough to a GPU hang watchdog to be
/// worth splitting even on a device we did NOT pre-classify as needing it.
///
/// This is the ONLY thing that can put a never-split device (every discrete GPU — see
/// [`initial_submit_dispatch_cap`]) onto the splitting path, and it is deliberately far above any
/// healthy discrete forward (tens of ms) so that the tuned discrete path is never touched by the
/// feedback loop, no matter how many dispatches a model's graph happens to have. What it DOES
/// cover is the case the CU-count heuristics cannot know about: a discrete GPU slow enough — or a
/// model big enough — that a forward pass lands in watchdog territory anyway.
pub const SUBMIT_DANGER_NS: u64 = 1_000_000_000;

impl Capabilities {
    /// The full 16x16x16 f16 coopmat kernel set is live (`coopmat_f16 == COOPMAT_TILE_16`) —
    /// byte-for-byte the boolean every pre-shape-table gate keyed on. All existing coopmat
    /// shaders (GEMM warptiles, flash attention, deltanet_prep) are built ONLY at this shape,
    /// so they must stay dark when the device's tile is 8x8x16.
    pub fn f16_coopmat(&self) -> bool {
        self.coopmat_f16 == Some(COOPMAT_TILE_16)
    }
    /// The 8x8x16 f16 tile was selected (Intel Arc XMX under `INFR_CM_8X8=1`) — gates ONLY the
    /// `native_gemm_warp` `_cm8` builds; every other coopmat family has no kernel at this shape.
    pub fn f16_coopmat_8x8(&self) -> bool {
        self.coopmat_f16 == Some(COOPMAT_TILE_8)
    }
    /// 16x16x16 f8 coopmat GEMM tier available (see `coopmat_f8`).
    pub fn f8_coopmat(&self) -> bool {
        self.coopmat_f8 == Some(COOPMAT_TILE_16)
    }
    /// 16x16x16 i8 coopmat available — DETECTION ONLY, the adapter additionally requires
    /// `INFR_I8_COOPMAT=1` (see `coopmat_i8`).
    pub fn i8_coopmat(&self) -> bool {
        self.coopmat_i8 == Some(COOPMAT_TILE_16)
    }
    /// 16x16x16 bf16 coopmat GEMM tier available (see `coopmat_bf16`).
    pub fn bf16_coopmat(&self) -> bool {
        self.coopmat_bf16 == Some(COOPMAT_TILE_16)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    Weights,
    Activations,
    /// Persistent per-layer, per-side KV cache buffers (`kbufs`/`vbufs` and their forks/
    /// checkpoints/MTP-draft twins) — and, on qwen35, the DeltaNet conv/S recurrent-state buffers
    /// that repurpose the same slots. Distinct from `Activations` ONLY so a backend can opt these
    /// specific long-lived buffers into a 64-bit device address (see `Buffer::device_addr`)
    /// without handing one to every scratch/partial/logits `Activations` allocation too. Slice 0
    /// of the KV/u64 migration (issue #74): allocation-only, no kernel reads this address yet — a
    /// backend that doesn't distinguish it (CPU, Metal today) may simply treat it like
    /// `Activations`.
    KvCache,
    /// Host→device staging (host-visible, mapped): `upload` is a direct memcpy.
    Staging,
    /// Device→host readback (host-visible, mapped): `download` is a direct memcpy.
    Readback,
    /// Weights pinned in HOST memory but device-readable (GTT) — the MoE auto-fit's offloaded
    /// expert banks. On ReBAR systems `Staging` (CpuToGpu) lands in device-local host-visible
    /// VRAM, which defeats offloading; this class guarantees system RAM. The GPU reads it over
    /// the bus. Backends without the distinction (CPU, unified-memory Metal) treat it as Weights.
    HostWeights,
}

/// Opaque device-memory handle owned by a backend.
/// RAII scope for a weight-load progress display (see [`Backend::weight_progress`]). Purely a
/// lifetime marker: constructing it starts the display, dropping it finishes/clears it.
pub trait ProgressScope: Send {}

pub trait Buffer: Send + Sync {
    /// The LOGICAL extent of the tensor this buffer holds — the byte count the model asked for —
    /// NOT a per-device shard of it. A multi-device wrapper that splits one logical buffer across
    /// `W` devices (`infr-vulkan`'s `TpBuffer` of kind `Kv`/`Weight`) must report the SUM over its
    /// per-rank buffers and shard byte counts at its own `alloc`/`copy_buffer` boundary, exactly
    /// once. The rule exists because backend-agnostic callers mix the two sources: `SeamKv`'s MTP
    /// checkpoint sizes an alloc from `len_bytes()` while `seed_from` derives its copy length from
    /// the model config. When `len_bytes()` returned the per-rank size, the first double-sharded
    /// (`full/W²` buffers) and the second did not shard at all (a `W`-fold overrun) — both only
    /// caught because Vulkan's `check_extent` bounds-checks every transfer.
    fn len_bytes(&self) -> usize;
    /// Downcast hook so a backend can recover its concrete buffer type from a `&dyn Buffer`
    /// bound by the model (every buffer a backend sees was allocated by itself).
    fn as_any(&self) -> &dyn std::any::Any;
    /// `Some(addr)` when this buffer's contents live inside a `bufferDeviceAddress` arena block at
    /// device address `addr` (a resident-weight sub-tensor — see `infr-vulkan`'s
    /// `INFR_RESIDENT_BDA`), rather than being its own independently-bound buffer object. A big
    /// matmul weight reads such a buffer through its 64-bit device address via a `-DSTREAMED` shader
    /// twin instead of a descriptor bind — required once the tensor's range would exceed
    /// `maxStorageBufferRange`/4 GiB, and the preferred path for the big matmul families regardless.
    /// A small weight tensor with no streamed twin (a norm gamma, bias, or rope table) may instead
    /// be BOUND as a descriptor: the bind must cover only this tensor's own `(sub_offset, range)`,
    /// never `(0, WHOLE_SIZE)` — a `WHOLE_SIZE` bind describes the WHOLE arena block, not this
    /// tensor's byte range within it (see `infr-vulkan`'s `Recorder::vkb`, the sole choke point that
    /// builds such binds correctly). `None` (the default) for every ordinary buffer, which is every
    /// buffer today outside `INFR_RESIDENT_BDA=1`.
    fn device_addr(&self) -> Option<u64> {
        None
    }
}

/// A compiled, ready-to-run graph (pipelines + command buffers for Vulkan, an op schedule for CPU).
pub trait Plan: Send + Sync {
    /// Downcast hook so a backend can recover its concrete plan type from a `&dyn Plan`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The trivial [`Plan`] both current backends share: it just carries a clone of the [`Graph`] — the
/// CPU interpreter and the Vulkan adapter each re-walk the ops every `execute`, so "compiling" is
/// only storing the graph. A backend's `compile` returns [`GraphPlan::boxed`]; its `execute` recovers
/// the graph via `plan.as_any().downcast_ref::<GraphPlan>()`. (Was duplicated as `CpuPlan`/`VkPlan`.)
pub struct GraphPlan {
    pub graph: crate::graph::Graph,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl GraphPlan {
    pub fn boxed(graph: &crate::graph::Graph) -> Box<dyn Plan> {
        Box::new(GraphPlan {
            graph: graph.clone(),
        })
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Plan for GraphPlan {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Binds a [`Graph`]'s `Input`/`Weight`/`Output` handles to concrete backend buffers for one
/// `execute`. The model holds the buffers; this only borrows them, so re-binding per step is cheap
/// and the graph/plan is reused across steps without recompilation.
///
/// Backed by a `Vec` indexed by `TensorId.0` (a DENSE `u32` handle into one graph's tensor list),
/// not a `HashMap` — `bind`/`get` are then hash-free O(1), which matters because decode rebinds
/// every step. `bind`/`get` semantics are identical to the old map: `get` on an unbound (or
/// never-grown-to) id returns `None`, and re-binding an id overwrites.
#[derive(Default)]
pub struct Bindings<'a> {
    /// `slots[id.0] = Some(buf)` for a bound handle, `None` for unbound. Grows on demand to the
    /// highest id bound (bounded by the graph's tensor count).
    slots: Vec<Option<&'a dyn Buffer>>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<'a> Bindings<'a> {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Bind a graph handle to a buffer the model owns.
    pub fn bind(&mut self, id: TensorId, buf: &'a dyn Buffer) -> &mut Self {
        let i = id.0 as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, None);
        }
        self.slots[i] = Some(buf);
        self
    }

    /// Look up a bound buffer (backend uses this while executing).
    pub fn get(&self, id: TensorId) -> Option<&'a dyn Buffer> {
        self.slots.get(id.0 as usize).copied().flatten()
    }
}

/// A compute device. Implementations: `infr-vulkan`, `infr-cpu`, `infr-metal`.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;

    // ---- memory ----
    /// Allocate a buffer of `bytes`, **guaranteed zero-initialized** (calloc semantics). This is the
    /// safe default: code that reads a buffer before fully writing it (accumulators, recurrent state,
    /// KV caches, padding rows) behaves identically on every backend. GPU backends return recycled,
    /// uninitialized VRAM otherwise, so relying on implicit zeroing is a silent CPU-works/GPU-garbage
    /// trap — always `alloc` unless you can prove every element is written before it's read.
    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>>;
    /// Allocate WITHOUT zero-initialization — an explicit opt-out for hot buffers whose full extent is
    /// provably written before any read (e.g. weights, which are immediately uploaded). Faster (skips
    /// the zero-fill) but UNSAFE if misused. In debug builds the returned memory is POISONED (filled
    /// with a non-zero pattern) so a read-before-write surfaces loudly in tests instead of silently
    /// working on CPU. The default forwards to [`alloc`](Self::alloc) (safe); backends override for
    /// the perf win.
    fn alloc_uninit(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        self.alloc(bytes, usage)
    }
    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()>;
    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()>;
    /// Emit the one-shot KV-cache placement summary for a VRAM-first spill backend (the Vulkan
    /// `INFR_KV_OVERFLOW` path): how many `BufferUsage::KvCache` buffers landed in device-local VRAM
    /// vs spilled to system RAM. The runner calls this ONCE, right after the per-layer KV allocation
    /// loop, so the user sees the resident/spilled split. Default no-op (backends without host spill,
    /// or with the flag off, print nothing).
    fn kv_overflow_report(&self) {}
    /// Open a weight-load progress scope: while the returned guard lives, this backend's weight
    /// allocations (`BufferUsage::Weights`/`HostWeights`) advance a visible progress display;
    /// dropping the guard finishes and clears it. The ticking lives in each backend's `alloc`, so
    /// no loader can forget a tensor — a loader only opens the scope around its upload loop.
    /// Backends without a display (the CPU interpreter's zero-copy mmap load is instant) keep
    /// this default no-op scope.
    fn weight_progress(&self, _total_bytes: Option<u64>) -> Box<dyn ProgressScope> {
        struct NoProgress;
        impl ProgressScope for NoProgress {}
        Box::new(NoProgress)
    }
    /// Copy the first `bytes` of `src` into the start of `dst` (both buffers this backend's).
    /// The KV prefix-sharing primitive: a new chat slot seeds its cache from another slot's
    /// common prefix instead of re-prefilling it. Default is a host bounce (download ONLY the
    /// prefix, then upload it) — correct everywhere; backends override with a device-side copy.
    /// Errors (rather than panics) if `bytes` exceeds `src`'s length.
    fn copy_buffer(&self, src: &dyn Buffer, dst: &dyn Buffer, bytes: usize) -> Result<()> {
        check_copy_bytes(bytes, src.len_bytes())?;
        // Only the prefix crosses the bus — NOT the whole (potentially `max_ctx`) `src` buffer.
        let mut tmp = vec![0u8; bytes];
        self.download(src, &mut tmp)?;
        self.upload(dst, &tmp)
    }

    // ---- execution (compile once per shape, execute per token/step) ----
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>>;
    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()>;
    /// Chained decode: run the record-once decode plan `n` times in ONE submission, the sampled
    /// token id flowing device-side from each iteration's `Op::Argmax`/`Op::Sample` into the next
    /// iteration's `Op::EmbedGather` — the caller MUST have bound the sampler output and the
    /// embed-gather ids input to the SAME buffer, seeded with the first token to feed. Returns
    /// the `n` sampled ids, or `None` when this backend/plan can't chain (the caller falls back
    /// to per-token `execute`). Positions/params self-advance (a chaining backend implies the
    /// device-side pos increment).
    fn execute_chain(
        &self,
        _plan: &dyn Plan,
        _bindings: &Bindings,
        _n: usize,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// The longest chain [`Backend::execute_chain`] may be asked for on this device.
    ///
    /// A chain is ONE submit carrying `n` back-to-back decode steps, and a GPU's hang watchdog is
    /// armed per SUBMIT — so on a slow device an unbounded chain is a device-lost waiting to
    /// happen (on the surveyed 2-CU integrated part a decode step is ~213 ms, so the default chain
    /// of 8 would be ~1.7 s in a single command buffer, against a device that kills a job at
    /// ~2.06 s).
    ///
    /// The CALLER must clamp with this before it commits to `n` — not `execute_chain` itself,
    /// because the runner draws one sampling uniform per chained step BEFORE the call, so a
    /// backend that quietly shortened the chain would advance the RNG stream further than the
    /// tokens it returned consumed. Default `usize::MAX`: no bound (every backend whose forward is
    /// nowhere near a watchdog, which is all of them except a slow iGPU).
    fn max_decode_chain(&self) -> usize {
        usize::MAX
    }

    fn sync(&self) -> Result<()>;

    /// Whether the currently loaded model is running any MoE expert layer through a paged VRAM
    /// cache (an expert bank too big to keep fully resident — see `infr_vulkan::pager::GpuPager`).
    /// Paged execution needs a host readback of the router's chosen expert ids BETWEEN a layer's
    /// GEMV stages (to resolve/upload cache misses before the id-indexed GEMV reads them), which a
    /// cached record-once decode replay can't express (the whole point of replay is recording
    /// every position-dependent op ONCE with no host round-trip in between). The seam's decode-loop
    /// gate and the Vulkan adapter's own `execute`/`execute_chain` both check this and force the
    /// per-execute static path instead — see `infr_vulkan::adapter::execute`'s doc.
    /// `false` for every backend but Vulkan, and for Vulkan whenever no paged model is loaded (the
    /// overwhelming common case: MoE experts that fit VRAM are fully resident with zero change).
    fn moe_paged(&self) -> bool {
        false
    }

    /// Whether the currently loaded DENSE model streams per-layer weight blocks through a paged
    /// VRAM cache (dense layer streaming — the `crate::pager::Pager::schedule` policy; see
    /// `infr_vulkan::pager`'s dense session). Unlike [`Backend::moe_paged`] no host readback is
    /// ever needed (dense layer order is deterministic, every "miss" is known in advance), but
    /// the record-once decode replay still can't express the per-token ring staging +
    /// arena-offset rebinding, so the same gates that force a paged-MoE model onto the
    /// per-execute static path check this too. `false` for every backend but Vulkan, and for
    /// Vulkan whenever the loaded model's dense weights fit VRAM (the overwhelming common case —
    /// zero change there).
    fn dense_paged(&self) -> bool {
        false
    }

    /// Perf (DiffusionGemma denoise, perf slice 3 — docs/diffusion-gemma.md): fused per-canvas-row
    /// entropy-bound sampler reduction over raw `[rows, dim]` logits, avoiding a full `[rows,
    /// dim]` host download. `u` is `rows` host-drawn uniform `[0,1)` floats (the seeded
    /// CDF-inversion target draw — reproducibility rides the SAME host RNG as the CPU sampler,
    /// only the reduction itself moves to the GPU). On success, writes:
    ///   - `argmax_out[r]` (u32): argmax token id over `raw[r,:] * temp_inv`.
    ///   - `entropy_out[r]` (f32): entropy of `softmax(raw[r,:] * temp_inv)`.
    ///   - `sampled_out[r]` (u32): one multinomial draw from that softmax via forward (vocab-
    ///     order) CDF inversion against `u[r]` — matches the host sampler's algorithm exactly
    ///     (same order, same target), NOT bit-identical (GPU float reduction order differs).
    ///
    /// Returns `Ok(false)` (default: every backend but Vulkan) when unsupported — the caller falls
    /// back to a full `download` + host reduction. Never partially writes the outputs on `Ok(false)`.
    #[allow(clippy::too_many_arguments)]
    fn eb_sample_reduce(
        &self,
        logits: &dyn Buffer,
        u: &dyn Buffer,
        rows: usize,
        dim: usize,
        temp_inv: f32,
        argmax_out: &dyn Buffer,
        entropy_out: &dyn Buffer,
        sampled_out: &dyn Buffer,
    ) -> Result<bool> {
        let _ = (
            logits,
            u,
            rows,
            dim,
            temp_inv,
            argmax_out,
            entropy_out,
            sampled_out,
        );
        Ok(false)
    }
}

/// Fan [`Backend::kv_overflow_report`] out over the per-device backends a multi-device wrapper
/// owns. Each rank made its OWN placement decision against its OWN card's budget, so each prints
/// its own banner — the wrapper has nothing to add and nothing to merge.
///
/// A three-line loop, but it was the three-line loop written three times (`TensorParallelBackend`,
/// `ExpertParallelBackend`, `PipelineBackend`), which is how the report gained a fourth spelling
/// while the body it forwards to had two.
pub fn kv_overflow_report_each<B: Backend>(backends: &[B]) {
    for b in backends {
        b.kv_overflow_report();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The integrated chunk policy. The load-bearing property is the CEILING: this may only ever
    /// SHRINK the discrete default (1024), never grow it — it is a watchdog guard, not a tuning
    /// knob — and it must never return 0 (a zero-row prefill chunk would not terminate).
    #[test]
    fn integrated_ubatch_rows_is_a_bounded_shrink() {
        // Unknown CU count (the common case: only AMD reports one) takes the conservative floor —
        // the ~2-CU RDNA2 iGPU the 128-row figure is calibrated on.
        assert_eq!(integrated_ubatch_rows(0), 128);
        assert_eq!(integrated_ubatch_rows(2), 128); // the calibrated part
        assert_eq!(integrated_ubatch_rows(1), 128); // never below the floor
                                                    // Linear in CU count, so a beefier iGPU is not needlessly throttled.
        assert_eq!(integrated_ubatch_rows(4), 256);
        assert_eq!(integrated_ubatch_rows(12), 768); // Radeon 780M class
                                                     // ...but clamped at the DISCRETE default: this can only shrink the chunk, never grow it.
        assert_eq!(integrated_ubatch_rows(16), 1024);
        assert_eq!(integrated_ubatch_rows(64), 1024);
        assert_eq!(integrated_ubatch_rows(u32::MAX), 1024);
        for cu in [0, 1, 2, 3, 7, 8, 33, 96, 1024, u32::MAX] {
            let r = integrated_ubatch_rows(cu);
            assert!((128..=1024).contains(&r), "cu={cu} -> {r} out of bounds");
        }
    }

    /// The submit splitter's pre-measurement default. The load-bearing property is that a DISCRETE
    /// GPU never splits: a dGPU forward is tens of milliseconds, nowhere near a hang watchdog, and
    /// splitting it would only add barriers and submits to a tuned path.
    #[test]
    fn initial_submit_cap_splits_only_integrated() {
        assert_eq!(initial_submit_dispatch_cap(false), 0); // discrete: never split
        assert_eq!(initial_submit_dispatch_cap(true), 128); // integrated: the measured default
    }

    /// The measured feedback loop. The property that matters is that the cap it returns puts a
    /// segment inside the budget on whatever device produced the sample.
    #[test]
    fn submit_cap_tracks_measured_dispatch_cost() {
        // The surveyed 2-CU iGPU: 757 dispatches, ~2.05 s ⇒ ~2.7 ms/dispatch. The cap it yields
        // must hold a segment inside the 250 ms budget...
        let cap = submit_cap_from_measurement(2_050_000_000, 757);
        assert!(cap > 0, "a 2 s forward must be split");
        let ns_per_dispatch = 2_050_000_000u64 / 757;
        assert!(
            cap as u64 * ns_per_dispatch <= SUBMIT_BUDGET_NS,
            "cap {cap} x {ns_per_dispatch}ns overruns the budget"
        );
        // ...and must cut that forward into a handful of segments, not one and not hundreds.
        assert!((2..=32).contains(&757usize.div_ceil(cap)), "cap={cap}");

        // A discrete GPU (same graph, ~30 ms) yields a cap so far above the forward's own
        // dispatch count that nothing is ever split — and the danger gate in the caller keeps it
        // formally unlimited anyway (see `submit_danger_threshold_is_far_above_a_healthy_forward`).
        let dcap = submit_cap_from_measurement(30_000_000, 757);
        assert!(
            dcap > 757,
            "a 30 ms forward must never be split (cap={dcap})"
        );
        // A truly trivial per-dispatch cost reports "unlimited" outright.
        assert_eq!(submit_cap_from_measurement(1_000, 757), 0);

        // Degenerate samples never produce a nonsense cap.
        assert_eq!(submit_cap_from_measurement(0, 757), 0);
        assert_eq!(submit_cap_from_measurement(2_050_000_000, 0), 0);
        // A pathologically slow sample still never splits below the floor (submits are not free).
        assert!(submit_cap_from_measurement(u64::MAX / 2, 4) >= 16);
    }

    /// The danger threshold is what keeps the feedback loop OFF a healthy discrete GPU: a forward
    /// has to be genuinely watchdog-adjacent before a never-split device starts splitting. Phrased
    /// against real forward timings rather than the bare constants, which is what actually matters.
    #[test]
    fn submit_danger_threshold_is_far_above_a_healthy_forward() {
        // Healthy discrete forwards (measured order of magnitude: tens of ms) stay well clear...
        for healthy_ns in [5_000_000u64, 30_000_000, 100_000_000, 250_000_000] {
            assert!(
                healthy_ns < SUBMIT_DANGER_NS,
                "{healthy_ns}ns must not look dangerous"
            );
        }
        // ...while the forward that actually caused the device-lost is over the line. (`black_box`
        // keeps these comparisons off clippy's `assertions_on_constants`: the property is about
        // the constants' RELATIONSHIP, which is exactly what a reader needs pinned down.)
        let danger = std::hint::black_box(SUBMIT_DANGER_NS);
        let budget = std::hint::black_box(SUBMIT_BUDGET_NS);
        assert!(std::hint::black_box(2_050_000_000u64) >= danger);
        // And the threshold is above the per-submit budget it guards (a submit that merely wants
        // splitting is not the same thing as a device that must start splitting).
        assert!(danger > budget);
    }

    /// A DISCRETE GPU must be untouched by any of this: `integrated` defaults false, so the seam's
    /// `default_ubatch_rows` takes its 1024 branch and no tuned dGPU shape moves.
    #[test]
    fn discrete_is_the_default() {
        let caps = Capabilities::default();
        assert!(!caps.integrated);
        assert_eq!(caps.compute_units, 0);
    }

    /// The `copy_buffer` bounds-check helper: an oversize prefix errors instead of panicking, and
    /// any prefix up to the source length is accepted.
    #[test]
    fn check_copy_bytes_rejects_oversize_prefix() {
        assert!(check_copy_bytes(0, 16).is_ok());
        assert!(check_copy_bytes(16, 16).is_ok()); // whole buffer is a valid prefix
        assert!(check_copy_bytes(8, 16).is_ok());
        assert!(check_copy_bytes(17, 16).is_err()); // one past the end
        assert!(check_copy_bytes(usize::MAX, 16).is_err());
    }

    /// A host-backed mock backend just large enough to exercise the DEFAULT `copy_buffer`: it must
    /// transfer ONLY the requested prefix (not the whole `src`) and it must NOT panic on an oversize
    /// `bytes` — it returns an `Err`.
    struct MockBuffer(std::sync::Mutex<Vec<u8>>);
    impl Buffer for MockBuffer {
        fn len_bytes(&self) -> usize {
            self.0.lock().unwrap().len()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Records how many bytes the last `download` was asked for, to prove the whole `src` never
    /// crosses the bus for a small prefix copy.
    struct MockBackend {
        last_download_len: std::sync::Mutex<usize>,
    }
    impl Backend for MockBackend {
        fn name(&self) -> &str {
            "mock"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }
        fn alloc(&self, bytes: usize, _usage: BufferUsage) -> Result<Box<dyn Buffer>> {
            Ok(Box::new(MockBuffer(std::sync::Mutex::new(vec![
                0u8;
                bytes
            ]))))
        }
        fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
            let dst = dst.as_any().downcast_ref::<MockBuffer>().unwrap();
            dst.0.lock().unwrap()[..src.len()].copy_from_slice(src);
            Ok(())
        }
        fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
            *self.last_download_len.lock().unwrap() = dst.len();
            let src = src.as_any().downcast_ref::<MockBuffer>().unwrap();
            dst.copy_from_slice(&src.0.lock().unwrap()[..dst.len()]);
            Ok(())
        }
        fn compile(&self, _graph: &Graph) -> Result<Box<dyn Plan>> {
            unimplemented!()
        }
        fn execute(&self, _plan: &dyn Plan, _bindings: &Bindings) -> Result<()> {
            unimplemented!()
        }
        fn sync(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn copy_buffer_default_copies_only_the_prefix() {
        let be = MockBackend {
            last_download_len: std::sync::Mutex::new(0),
        };
        let src = MockBuffer(std::sync::Mutex::new((0u8..64).collect()));
        let dst = MockBuffer(std::sync::Mutex::new(vec![0xAAu8; 64]));

        be.copy_buffer(&src, &dst, 8).unwrap();
        // Only the 8-byte prefix crossed the bus — NOT the whole 64-byte source.
        assert_eq!(*be.last_download_len.lock().unwrap(), 8);
        let out = dst.0.lock().unwrap();
        assert_eq!(&out[..8], &[0, 1, 2, 3, 4, 5, 6, 7]); // prefix copied
        assert!(out[8..].iter().all(|&b| b == 0xAA)); // tail untouched
    }

    #[test]
    fn bindings_vec_backed_get_matches_old_map_semantics() {
        // Distinguishable lengths so `get` can be checked to return the exact buffer bound.
        let a = MockBuffer(std::sync::Mutex::new(vec![0u8; 4]));
        let b = MockBuffer(std::sync::Mutex::new(vec![0u8; 8]));
        let c = MockBuffer(std::sync::Mutex::new(vec![0u8; 16]));
        let mut binds = Bindings::new();
        // Sparse, out-of-order binds (as a real graph does — handles aren't bound in id order).
        binds.bind(TensorId(5), &a).bind(TensorId(2), &b);
        // Bound ids resolve to the exact buffer; the gap (ids 0,1,3,4) and any id past the
        // high-water mark are None — identical to the old HashMap miss.
        assert_eq!(binds.get(TensorId(5)).unwrap().len_bytes(), 4);
        assert_eq!(binds.get(TensorId(2)).unwrap().len_bytes(), 8);
        assert!(binds.get(TensorId(0)).is_none());
        assert!(binds.get(TensorId(3)).is_none());
        assert!(binds.get(TensorId(99)).is_none());
        // Re-binding an id overwrites (identical to the old HashMap::insert semantics).
        binds.bind(TensorId(5), &c);
        assert_eq!(binds.get(TensorId(5)).unwrap().len_bytes(), 16);
    }

    #[test]
    fn copy_buffer_default_errors_on_oversize_bytes() {
        let be = MockBackend {
            last_download_len: std::sync::Mutex::new(0),
        };
        let src = MockBuffer(std::sync::Mutex::new(vec![0u8; 16]));
        let dst = MockBuffer(std::sync::Mutex::new(vec![0u8; 64]));
        // 32 > src's 16 bytes: an Err, not a panic, and nothing was downloaded.
        assert!(be.copy_buffer(&src, &dst, 32).is_err());
        assert_eq!(*be.last_download_len.lock().unwrap(), 0);
    }
}
