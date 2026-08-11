//! STANDALONE, ISOLATED probe: does `VK_NV_cooperative_matrix2`'s in-fragment per-element access
//! (`coopMatPerElementNV`) remove the int8 "rescale tax" that `VK_KHR_cooperative_matrix` (v1) pays?
//!
//! BACKGROUND: an int8 coopmat GEMM's `coopMatMulAdd` produces a raw SINT32 accumulator. Applying
//! a per-block rank-1 descale (`row_scale[r] * col_scale[c]`) to it needs per-*element* access to
//! the fragment. KHR_cooperative_matrix v1 has no spec-portable way to do that in-register — the
//! ORIGINAL form of `native_gemm_i8cm_q8_0.comp` (see that file's header, "PERF NOTES" item 0)
//! routed every block's accumulator through `coopMatStore(shared) -> barrier() -> reload` just to
//! read individual elements back out and multiply. (That kernel has SINCE been optimized to instead
//! read `csub[i]` directly off the KHR fragment — which works, but only via an EMPIRICALLY DERIVED,
//! implementation-defined `i -> (row,col)` mapping that must be re-derived per driver/config via
//! `coopmat_int8_test --fragment_layout`.) `VK_NV_cooperative_matrix2` adds `coopMatPerElementNV`,
//! which calls a user callback `elemOp(row, col, value, ...)` PER ELEMENT, in-fragment, with a
//! portable (row,col) addressing scheme — no shared-memory round trip, no driver-specific fragment
//! layout to reverse-engineer.
//!
//! This binary times both epilogue strategies back-to-back and reports the ratio, so the "does
//! per-element access actually win perf-wise" question has a number instead of a guess. It runs TWO
//! benches:
//!   1. SINGLE-TILE — one 16x16x16 int8 tile, one workgroup. Isolates raw epilogue op latency, but
//!      is DISPATCH-OVERHEAD-DOMINATED (~1.8us/dispatch), so it under-represents the v1 barrier tax:
//!      the barrier happens once, not per-block, and there's no occupancy pressure.
//!   2. FULL-GEMM — a realistic M×N×K int8 GEMM (default 512×2048×2048 → K/32 = 64 Q8_0-style
//!      blocks) tiled 16x16-per-workgroup over the full tilesM×tilesN grid, mimicking
//!      `native_gemm_i8cm_q8_0.comp`'s structure: a K-loop over the 64 blocks, each block doing an
//!      int8×int8→int32 `coopMatMulAdd` (2× 16-K sub-tiles) then applying that block's per-row-act ×
//!      per-col-weight descale and accumulating into a persistent FLOAT accumulator. Here v1's
//!      store→barrier→reload epilogue fires 64× per workgroup across 4096 workgroups, so its barrier
//!      cost AND its occupancy cost (a `shared int[256]` staging array per workgroup) both
//!      accumulate — the thing the single-tile bench can't see. If coopmat2 STILL loses here, the
//!      lever is dead; if it flips to winning, the real kernel is worth pursuing.
//!
//! It does NOT touch the production adapter/gemm path (own Vulkan instance/device, own tiny shaders
//! compiled at runtime via `glslc`, exactly like `coopmat_int8_test.rs`).
//!
//! NOTE on the full-GEMM absolute numbers: all buffers are HOST_VISIBLE|HOST_COHERENT (system RAM,
//! same as `coopmat_int8_test.rs`), so the weight/activation reads cross PCIe and the absolute
//! us/dispatch is bandwidth-bound, NOT representative of a device-local kernel. Both paths read the
//! exact same bytes, so the RATIO (which isolates the epilogue + occupancy difference) is the
//! meaningful output; ignore the absolute times.
//!
//! COOPMAT2 SIGNATURE NOTE (this took a probe to nail down): `coopMatPerElementNV` is a VOID
//! function with an `out` result parameter, NOT a value-returning expression —
//!   `void coopMatPerElementNV(out coopmat result, coopmat m, T elemOp, ...);`
//! i.e. call it as `coopMatPerElementNV(result, m, elemFn);`, never `result = coopMatPerElementNV(m,
//! elemFn)`. Source: `GL_NV_cooperative_matrix2`'s spec text (glslang/Khronos GLSL registry),
//! `extensions/nv/GLSL_NV_cooperative_matrix2.txt`, the `coopMatPerElementNV` prose block. The
//! `elemOp` callback's first two params are `const in uint32_t` (row, col), the third is `const in
//! <component type>` (the element value); its return type must match `result`'s component type.
//! Confirmed compiling with `glslc` and inspected via `spirv-dis`: lowers straight to
//! `OpCooperativeMatrixPerElementOpNV`, no shared memory, no `OpControlBarrier`.
//!
//! Also note: converting the SINT32 accumulator to a FLOAT32 accumulator (`coopmat<float,
//! gl_ScopeSubgroup,16,16,gl_MatrixUseAccumulator>(intMat)`, a GL_NV_cooperative_matrix2 "type
//! conversion constructor") lowers to a plain `OpConvertSToF` applied to the opaque coopmat type —
//! no separate `OpCooperativeMatrixConvertNV`/capability needed for this simple scalar-convert case,
//! confirmed via `spirv-dis`.
//!
//! HARDWARE GATING: `VK_NV_cooperative_matrix2` is checked for (both the device extension string
//! AND the `cooperativeMatrixPerElementOperations` feature bit) BEFORE any device is created with it
//! enabled. If absent — which is the case on THIS box (RDNA3, RADV/Mesa; the extension is RDNA4-only
//! and even there gated behind the driconf flag `radv_cooperative_matrix2_nv=true`) — the program
//! prints a clear message and exits 0. It never attempts to create a device with an unsupported
//! extension/feature enabled.
//!
//! NOT the production gate. `caps::check_coopmat2_support` (a transcription of llama.cpp's) is what
//! the backend uses to decide whether a device may run a coopmat2 PATH, and it is far stricter:
//! all seven feature bits, `bufferDeviceAddress`, and specific fp16/fp32 flexible-dimension tile
//! shapes. This probe deliberately does not go through it, because it exercises only
//! `coopMatPerElementNV` — flexible dimensions, tensor addressing and block loads are irrelevant
//! here, and requiring them would make this skip on hardware where its experiment would run.
//!
//! ash 0.38.0+1.3.281 has NO bindings for `VK_NV_cooperative_matrix2` (checked: no
//! `CooperativeMatrix2FeaturesNV`/`PropertiesNV` in `ash::vk::definitions`). The feature struct is
//! hand-rolled here (`#[repr(C)]`, layout + `sType` value taken from the locally installed
//! `/usr/include/vulkan/vulkan_core.h` — Vulkan SDK 1.4.350 — which DOES define it) and spliced into
//! the `pNext` chain by raw pointer assignment (`PhysicalDeviceFeatures2::p_next` /
//! `DeviceCreateInfo::p_next` are `pub` fields in ash 0.38, so this needs no `unsafe` beyond what the
//! surrounding Vulkan calls already require).
//!
//! Run (RDNA4, coopmat2-capable box only):
//!   `radv_cooperative_matrix2_nv=true cargo run --release -p infr-vulkan --example coopmat2_test`
//! On any other box (no `VK_NV_cooperative_matrix2`) this just prints a skip message and exits 0.

use ash::vk;
use std::ffi::{c_void, CStr};
use std::time::Instant;

// ── hand-rolled VK_NV_cooperative_matrix2 structs (absent from ash 0.38) ───────────────────────
// Field layout + sType values taken verbatim from /usr/include/vulkan/vulkan_core.h (Vulkan SDK
// 1.4.350, which defines VK_NV_cooperative_matrix2). Do not reorder fields — this is `#[repr(C)]`
// and must match the driver ABI exactly.

const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV: i32 = 1_000_593_000;

#[repr(C)]
#[derive(Clone, Copy)]
struct PhysicalDeviceCooperativeMatrix2FeaturesNv {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    cooperative_matrix_workgroup_scope: vk::Bool32,
    cooperative_matrix_flexible_dimensions: vk::Bool32,
    cooperative_matrix_reductions: vk::Bool32,
    cooperative_matrix_conversions: vk::Bool32,
    cooperative_matrix_per_element_operations: vk::Bool32,
    cooperative_matrix_tensor_addressing: vk::Bool32,
    cooperative_matrix_block_loads: vk::Bool32,
}

impl Default for PhysicalDeviceCooperativeMatrix2FeaturesNv {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
            ),
            p_next: std::ptr::null_mut(),
            cooperative_matrix_workgroup_scope: vk::FALSE,
            cooperative_matrix_flexible_dimensions: vk::FALSE,
            cooperative_matrix_reductions: vk::FALSE,
            cooperative_matrix_conversions: vk::FALSE,
            cooperative_matrix_per_element_operations: vk::FALSE,
            cooperative_matrix_tensor_addressing: vk::FALSE,
            cooperative_matrix_block_loads: vk::FALSE,
        }
    }
}

// ── the two epilogue shaders ─────────────────────────────────────────────────────────────────
// Both compute the SAME thing: C[16,16] = (A[16,16] @ B[16,16])_int32 * rowScale[r] * colScale[c],
// A row-major int8, B column-major int8 (mirrors `coopmat_int8_test`'s "baseline" variant — the
// exact config proven hang-free/bit-exact on RDNA3's KHR coopmat). Bindings: 0=A, 1=B, 2=Scale
// (rowScale[16] then colScale[16], std430: rowScale at byte 0, colScale at byte 64), 3=C (f32[256]).

/// Path A: coopmat2 in-fragment epilogue via `coopMatPerElementNV`. No shared memory, no barrier —
/// the descale happens directly on the (converted-to-float) accumulator fragment.
const SHADER_COOPMAT2: &str = r#"#version 460
#extension GL_NV_cooperative_matrix2 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuf { int8_t a[]; };
layout(binding = 1) readonly buffer BBuf { int8_t b[]; };
layout(binding = 2) readonly buffer ScaleBuf { float rowScale[16]; float colScale[16]; };
layout(binding = 3) writeonly buffer CBuf { float c[]; };

float descale(const in uint32_t r, const in uint32_t col, const in float v) {
    return v * rowScale[r] * colScale[col];
}

void main() {
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
    coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
        coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);

    coopMatLoad(af, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(bf, b, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
    acc = coopMatMulAdd(af, bf, acc);

    // Type-conversion constructor (int32 accumulator -> float32 accumulator), then the in-fragment
    // per-element descale. No coopMatStore/shared/barrier round trip anywhere in this shader.
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> accf =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> result;
    coopMatPerElementNV(result, accf, descale);

    coopMatStore(result, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
"#;

/// Path B: KHR_cooperative_matrix v1 baseline — the "rescale tax" pattern. No per-element access
/// exists in v1, so the raw int32 accumulator is routed through shared memory
/// (`coopMatStore` -> `barrier()` -> scalar reload) purely to apply the rank-1 descale. Mirrors the
/// ORIGINAL (pre-optimization) form of `native_gemm_i8cm_q8_0.comp`'s epilogue described in that
/// file's header ("PERF NOTES" item 0).
const SHADER_V1_BASELINE: &str = r#"#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuf { int8_t a[]; };
layout(binding = 1) readonly buffer BBuf { int8_t b[]; };
layout(binding = 2) readonly buffer ScaleBuf { float rowScale[16]; float colScale[16]; };
layout(binding = 3) writeonly buffer CBuf { float c[]; };

shared int Tmp[256]; // 16x16 int32 accumulator staged to shared, row-major — the "tax"

void main() {
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
    coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
        coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);

    coopMatLoad(af, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(bf, b, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
    acc = coopMatMulAdd(af, bf, acc);

    // No in-fragment element access in KHR v1 -> store to shared, barrier, reload scalar-wise.
    coopMatStore(acc, Tmp, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();

    for (uint idx = gl_LocalInvocationIndex; idx < 256u; idx += 32u) {
        uint r = idx / 16u;
        uint col = idx % 16u;
        c[idx] = float(Tmp[idx]) * rowScale[r] * colScale[col];
    }
}
"#;

// ── the two FULL-GEMM shaders ──────────────────────────────────────────────────────────────────
// Realistic int8 GEMM mimicking native_gemm_i8cm_q8_0.comp's structure: C[M,N] = for each K/32
// block, (A[M,·]·Wᵀ[·,N])_int32 * rowScale[m,blk] * colScale[n,blk], summed over blocks into a
// FLOAT accumulator (per-block rescale = the exact math that file's epilogue does). One workgroup ==
// one wave32 subgroup computes one 16x16 output tile; dispatched over the full (M/16)×(N/16) grid so
// it fills the GPU (occupancy matters — v1's `shared int[256]` limits it, coopmat2's has none). The
// K-loop runs K/32 blocks (default 64), each a 2×(16-K) coopMatMulAdd; the ONLY difference between
// the two shaders is the per-block descale epilogue. A/W are read straight from global memory (no
// shared staging) so the sole barrier difference is the v1 epilogue barrier, not staging.
// Bindings: 0=A int8[M,K] row-major, 1=W int8[N,K] row-major, 2=rowScale f16[M,K/32],
// 3=colScale f16[N,K/32], 4=C f32[M,N] row-major. push_constant {m,n,k}.

/// Path A (full GEMM, coopmat2): per-block descale via `coopMatPerElementNV`, in-fragment. The
/// callback needs the block index and this tile's global row/col base — supplied via module-scope
/// PRIVATE globals (`gRowBase`/`gColBase`/`gBlk`/`gNblk`), each invocation's own copy, all lanes set
/// them identically before each `coopMatPerElementNV` call. No shared memory, no barrier anywhere.
const SHADER_FULL_COOPMAT2: &str = r#"#version 460
#extension GL_NV_cooperative_matrix2 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC { uint m; uint n; uint k; } pc;
layout(binding = 0) readonly buffer ABuf { int8_t a[]; };
layout(binding = 1) readonly buffer WBuf { int8_t w[]; };
layout(binding = 2) readonly buffer RSBuf { float16_t rowScale[]; };
layout(binding = 3) readonly buffer CSBuf { float16_t colScale[]; };
layout(binding = 4) writeonly buffer CBuf { float c[]; };

uint gRowBase;
uint gColBase;
uint gBlk;
uint gNblk;

float descale(const in uint32_t r, const in uint32_t col, const in float v) {
    float rs = float(rowScale[(gRowBase + r) * gNblk + gBlk]);
    float cs = float(colScale[(gColBase + col) * gNblk + gBlk]);
    return v * rs * cs;
}

void main() {
    uint tilesN = pc.n / 16u;
    gRowBase = (gl_WorkGroupID.x / tilesN) * 16u;
    gColBase = (gl_WorkGroupID.x % tilesN) * 16u;
    gNblk = pc.k / 32u;

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> facc =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint blk = 0u; blk < gNblk; blk++) {
        gBlk = blk;
        coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
            coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
        [[unroll]] for (uint s = 0u; s < 2u; s++) {
            uint k0 = blk * 32u + s * 16u;
            coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
            coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
            coopMatLoad(af, a, gRowBase * pc.k + k0, pc.k, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(bf, w, gColBase * pc.k + k0, pc.k, gl_CooperativeMatrixLayoutColumnMajor);
            acc = coopMatMulAdd(af, bf, acc);
        }
        // In-fragment per-block descale: convert int32 accumulator -> float, apply rowScale·colScale
        // per element via coopMatPerElementNV, accumulate into the persistent float matrix. No shared
        // round trip, no barrier — this is the whole point of the extension.
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> accf =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(acc);
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> scaled;
        coopMatPerElementNV(scaled, accf, descale);
        facc = facc + scaled;
    }
    coopMatStore(facc, c, gRowBase * pc.n + gColBase, pc.n, gl_CooperativeMatrixLayoutRowMajor);
}
"#;

/// Path B (full GEMM, v1 "rescale tax"): identical GEMM, but each block's per-block descale goes
/// through the KHR-v1 store→barrier→reload epilogue — `coopMatStore` the int32 accumulator to a
/// `shared int[256]`, `barrier()`, then each lane reloads its 8 owned elements and scale-accumulates
/// into a persistent `float facc[8]` register array. Fires K/32 times per workgroup: two barriers
/// per block, plus the shared array itself pressuring occupancy across the full grid. This is
/// native_gemm_i8cm_q8_0.comp's ORIGINAL (pre-`csub[i]`-optimization) epilogue, per its header.
/// The lane→(row,col) ownership here is MY choice on a plain row-major shared array (lane L owns
/// linear outputs [L*8, L*8+8)), so it's portable — no driver-specific fragment layout involved.
const SHADER_FULL_V1: &str = r#"#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC { uint m; uint n; uint k; } pc;
layout(binding = 0) readonly buffer ABuf { int8_t a[]; };
layout(binding = 1) readonly buffer WBuf { int8_t w[]; };
layout(binding = 2) readonly buffer RSBuf { float16_t rowScale[]; };
layout(binding = 3) readonly buffer CSBuf { float16_t colScale[]; };
layout(binding = 4) writeonly buffer CBuf { float c[]; };

shared int Tmp[256]; // one 16x16 int32 accumulator staged to shared each block — the v1 "tax"

void main() {
    uint tilesN = pc.n / 16u;
    uint rowBase = (gl_WorkGroupID.x / tilesN) * 16u;
    uint colBase = (gl_WorkGroupID.x % tilesN) * 16u;
    uint nblk = pc.k / 32u;
    uint lane = gl_SubgroupInvocationID; // 0..31

    float facc[8]; // lane owns row-major linear outputs [lane*8 .. lane*8+8)
    [[unroll]] for (uint i = 0u; i < 8u; i++) { facc[i] = 0.0; }

    for (uint blk = 0u; blk < nblk; blk++) {
        coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
            coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
        [[unroll]] for (uint s = 0u; s < 2u; s++) {
            uint k0 = blk * 32u + s * 16u;
            coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
            coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
            coopMatLoad(af, a, rowBase * pc.k + k0, pc.k, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(bf, w, colBase * pc.k + k0, pc.k, gl_CooperativeMatrixLayoutColumnMajor);
            acc = coopMatMulAdd(af, bf, acc);
        }
        // v1 epilogue: no in-fragment element access -> store to shared, barrier, reload+scale.
        coopMatStore(acc, Tmp, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        [[unroll]] for (uint i = 0u; i < 8u; i++) {
            uint idx = lane * 8u + i;
            uint r = idx / 16u;
            uint col = idx % 16u;
            float rs = float(rowScale[(rowBase + r) * nblk + blk]);
            float cs = float(colScale[(colBase + col) * nblk + blk]);
            facc[i] += float(Tmp[idx]) * rs * cs;
        }
        barrier(); // Tmp fully consumed before the next block's store overwrites it
    }
    [[unroll]] for (uint i = 0u; i < 8u; i++) {
        uint idx = lane * 8u + i;
        uint r = idx / 16u;
        uint col = idx % 16u;
        c[(rowBase + r) * pc.n + (colBase + col)] = facc[i];
    }
}
"#;

// ── glslc invocation (mirrors coopmat_int8_test.rs) ─────────────────────────────────────────────

fn compile_glsl(src: &str, tag: &str) -> Result<Vec<u32>, String> {
    let dir = std::env::temp_dir();
    let stem = format!("infr_coopmat2_test_{tag}_{}", std::process::id());
    let src_path = dir.join(format!("{stem}.comp"));
    let spv_path = dir.join(format!("{stem}.spv"));
    std::fs::write(&src_path, src).map_err(|e| format!("write shader src: {e}"))?;
    let status = std::process::Command::new("glslc")
        .args([
            "-fshader-stage=comp",
            "--target-env=vulkan1.3",
            "-O",
            src_path.to_str().unwrap(),
            "-o",
            spv_path.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("spawn glslc: {e}"))?;
    if !status.success() {
        return Err(format!("glslc exited with {status}"));
    }
    let bytes = std::fs::read(&spv_path).map_err(|e| format!("read spv: {e}"))?;
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(words)
}

// ── Vulkan context ───────────────────────────────────────────────────────────────────────────

struct VkCtx {
    #[allow(dead_code)]
    entry: ash::Entry,
    #[allow(dead_code)]
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
}

/// Enumerates + probes; returns `Err(msg)` (a clean skip reason, NOT a panic) if
/// `VK_NV_cooperative_matrix2` or the `cooperativeMatrixPerElementOperations` feature it needs are
/// absent. Only creates a device (with the extension/feature enabled) once both are confirmed
/// present — so an unsupported box (this RDNA3 dev box included) never gets a device-creation call
/// with an extension the driver doesn't advertise.
fn try_init_ctx() -> Result<VkCtx, String> {
    unsafe {
        let entry = ash::Entry::load().map_err(|e| format!("ash::Entry::load: {e}"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .map_err(|e| format!("create_instance: {e}"))?;

        let pdevices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices: {e}"))?;
        if pdevices.is_empty() {
            return Err("no Vulkan physical devices".into());
        }
        // Prefer discrete GPU, matching production `VulkanBackend::new()` selection.
        let pdevice = pdevices
            .iter()
            .copied()
            .find(|&pd| {
                instance.get_physical_device_properties(pd).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(pdevices[0]);
        let dev_name = CStr::from_ptr(
            instance
                .get_physical_device_properties(pdevice)
                .device_name
                .as_ptr(),
        )
        .to_string_lossy()
        .into_owned();

        // 1. Device-extension string check — cheap, no device created yet.
        let ext_props = instance
            .enumerate_device_extension_properties(pdevice)
            .map_err(|e| format!("enumerate_device_extension_properties: {e}"))?;
        let has_ext = |name: &CStr| {
            ext_props
                .iter()
                .any(|p| p.extension_name_as_c_str() == Ok(name))
        };
        if !has_ext(c"VK_NV_cooperative_matrix2") {
            return Err(format!(
                "{dev_name}: VK_NV_cooperative_matrix2 not in the device's supported extension \
                 list (expected on this box — RDNA4-only, and driconf-gated via \
                 radv_cooperative_matrix2_nv=true even there)"
            ));
        }
        if !has_ext(c"VK_KHR_cooperative_matrix") {
            return Err(format!(
                "{dev_name}: VK_KHR_cooperative_matrix not supported"
            ));
        }

        // 2. Feature-bit check via vkGetPhysicalDeviceFeatures2 — still no device created.
        let mut f16i8_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut mm_feat = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        let mut sg_feat = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut cm_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut s8_feat = vk::PhysicalDevice8BitStorageFeatures::default();
        let mut cm2_feat = PhysicalDeviceCooperativeMatrix2FeaturesNv::default();
        let mut feat2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f16i8_feat)
            .push_next(&mut mm_feat)
            .push_next(&mut sg_feat)
            .push_next(&mut cm_feat)
            .push_next(&mut s8_feat);
        // Hand-rolled struct: ash doesn't know the `Extends*` trait for it, so splice it into the
        // chain by raw pointer instead of `.push_next()`. `feat2.p_next` currently points at the
        // head of the chain `push_next` built above; make our struct the new head, pointing at the
        // old head in turn.
        cm2_feat.p_next = feat2.p_next;
        feat2.p_next = &mut cm2_feat as *mut _ as *mut c_void;
        instance.get_physical_device_features2(pdevice, &mut feat2);

        if cm_feat.cooperative_matrix == 0 {
            return Err(format!("{dev_name}: no cooperativeMatrix (KHR) feature"));
        }
        if f16i8_feat.shader_int8 == 0 {
            return Err(format!("{dev_name}: no shaderInt8 feature"));
        }
        if mm_feat.vulkan_memory_model == 0 {
            return Err(format!("{dev_name}: no vulkanMemoryModel feature"));
        }
        if s8_feat.storage_buffer8_bit_access == 0 {
            return Err(format!("{dev_name}: no storageBuffer8BitAccess feature"));
        }
        if cm2_feat.cooperative_matrix_per_element_operations == 0 {
            return Err(format!(
                "{dev_name}: VK_NV_cooperative_matrix2 is in the extension list but \
                 cooperativeMatrixPerElementOperations feature bit is not set — skipping \
                 (this is the expected outcome on RDNA3; on RDNA4 make sure \
                 radv_cooperative_matrix2_nv=true is set BEFORE process start)"
            ));
        }
        eprintln!(
            "[ctx] {dev_name}: VK_NV_cooperative_matrix2 present, \
             cooperativeMatrixPerElementOperations=1, cooperativeMatrixConversions={}",
            cm2_feat.cooperative_matrix_conversions
        );

        // 3. Both confirmed present — now (and only now) create the device with them enabled.
        let qf_props = instance.get_physical_device_queue_family_properties(pdevice);
        let qfi = qf_props
            .iter()
            .enumerate()
            .find(|(_, p)| {
                p.queue_flags.contains(vk::QueueFlags::COMPUTE) && p.timestamp_valid_bits > 0
            })
            .map(|(i, _)| i as u32)
            .ok_or("no compute queue family with timestamp support")?;

        let ext_ptrs = [
            c"VK_KHR_cooperative_matrix".as_ptr(),
            c"VK_KHR_8bit_storage".as_ptr(),
            c"VK_NV_cooperative_matrix2".as_ptr(),
        ];
        let mut shader_f16i8_ci =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_int8(true);
        let mut s8_ci =
            vk::PhysicalDevice8BitStorageFeatures::default().storage_buffer8_bit_access(true);
        let mut mm_ci = vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);
        let mut cm_ci =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut sg_ci = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        let mut cm2_ci = PhysicalDeviceCooperativeMatrix2FeaturesNv {
            cooperative_matrix_per_element_operations: vk::TRUE,
            cooperative_matrix_conversions: vk::TRUE,
            ..Default::default()
        };

        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&priorities);
        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut shader_f16i8_ci)
            .push_next(&mut s8_ci)
            .push_next(&mut mm_ci)
            .push_next(&mut cm_ci)
            .push_next(&mut sg_ci);
        // Splice the hand-rolled coopmat2 feature struct in, same technique as the probe above.
        cm2_ci.p_next = device_ci.p_next as *mut c_void;
        device_ci.p_next = &cm2_ci as *const _ as *const c_void;

        let device = instance
            .create_device(pdevice, &device_ci, None)
            .map_err(|e| format!("create_device: {e}"))?;
        let queue = device.get_device_queue(qfi, 0);
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfi)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| format!("create_command_pool: {e}"))?;
        let mem_props = instance.get_physical_device_memory_properties(pdevice);

        eprintln!("[ctx] device = {dev_name}");
        Ok(VkCtx {
            entry,
            instance,
            physical_device: pdevice,
            device,
            queue,
            mem_props,
            cmd_pool,
        })
    }
}

unsafe fn alloc_host_buffer(ctx: &VkCtx, size: u64) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
    let buf = ctx
        .device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size.max(4))
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .expect("create_buffer");
    let req = ctx.device.get_buffer_memory_requirements(buf);
    let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mt = (0..ctx.mem_props.memory_type_count)
        .find(|&i| {
            (req.memory_type_bits & (1 << i)) != 0
                && ctx.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(want)
        })
        .expect("no host-visible+coherent memory type");
    let mem = ctx
        .device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            None,
        )
        .expect("allocate_memory");
    ctx.device
        .bind_buffer_memory(buf, mem, 0)
        .expect("bind_buffer_memory");
    let ptr = ctx
        .device
        .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        .expect("map_memory") as *mut u8;
    (buf, mem, ptr)
}

/// Builds a compute pipeline pinned to subgroup size 32 (one 16x16x16 tile == one wave32 subgroup),
/// same discipline as `coopmat_int8_test.rs`'s `dispatch_one`. `push_size` (bytes) adds a COMPUTE
/// push-constant range when > 0 (the full-GEMM shaders take a {m,n,k} push constant; the single-tile
/// shaders take none).
unsafe fn make_pipeline(
    ctx: &VkCtx,
    spv: &[u32],
    ds_layout: vk::DescriptorSetLayout,
    push_size: u32,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let shader = ctx
        .device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
        .expect("create_shader_module");
    let push_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_size)];
    let mut layout_ci =
        vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout));
    if push_size > 0 {
        layout_ci = layout_ci.push_constant_ranges(&push_ranges);
    }
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&layout_ci, None)
        .expect("create_pipeline_layout");
    let entry_name = c"main";
    let mut req_sz =
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry_name)
        .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
        .push_next(&mut req_sz);
    let pipeline = ctx
        .device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)],
            None,
        )
        .map_err(|(_, e)| e)
        .expect("create_compute_pipelines")[0];
    (pipeline, pipeline_layout)
}

// ── data / CPU reference ─────────────────────────────────────────────────────────────────────

/// Packs a logical 16x16 int8 matrix honoring the requested coopMatLoad layout (same helper as
/// `coopmat_int8_test.rs`).
fn pack16(row_major: bool, get: impl Fn(usize, usize) -> i32) -> Vec<u8> {
    let n = 16usize;
    let mut buf = vec![0u8; n * n];
    for i in 0..n {
        for k in 0..n {
            let idx = if row_major { i * n + k } else { k * n + i };
            buf[idx] = get(i, k) as u8;
        }
    }
    buf
}

fn cpu_reference(row_scale: &[f32; 16], col_scale: &[f32; 16]) -> Vec<f32> {
    let a_get = |i: usize, k: usize| ((i + k) % 5) as i32;
    let b_get = |k: usize, j: usize| ((k + j) % 3) as i32;
    let mut c = vec![0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0i32;
            for k in 0..16 {
                s += a_get(i, k) * b_get(k, j);
            }
            c[i * 16 + j] = (s as f32) * row_scale[i] * col_scale[j];
        }
    }
    c
}

// ── full-GEMM shape + data ─────────────────────────────────────────────────────────────────────

const REPS: u32 = 200; // single-tile bench (dispatch-overhead dominated, so many reps)

// Default full-GEMM shape: 512×2048×2048 -> K/32 = 64 Q8_0-style blocks, (M/16)×(N/16) = 32×128 =
// 4096 workgroups (fills the GPU). All three must be multiples of 16 (M,N) / 32 (K).
const FULL_M: usize = 512;
const FULL_N: usize = 2048;
const FULL_K: usize = 2048;
const FULL_REPS: u32 = 20; // each dispatch is heavy (full 4096-workgroup grid), so fewer reps

// int8 operands: small values so per-block int32 dots stay tiny and the f32/f16 references are exact.
fn a_val(m: usize, k: usize) -> i32 {
    ((m + k) % 3) as i32 // 0,1,2
}
fn w_val(n: usize, k: usize) -> i32 {
    ((2 * n + k) % 3) as i32 // 0,1,2 (asymmetric vs A -> catches a transposed-load bug)
}
// Per-block scales, values chosen exactly representable in f16 AND f32 (1.0 / 0.5), so the CPU
// reference needs no rounding model and a tight tolerance is valid.
fn row_scale_val(m: usize, blk: usize) -> f32 {
    if (m + blk).is_multiple_of(2) {
        1.0
    } else {
        0.5
    }
}
fn col_scale_val(n: usize, blk: usize) -> f32 {
    if (n + blk).is_multiple_of(2) {
        1.0
    } else {
        0.5
    }
}

/// Exact CPU value for ONE output element C[m,n], matching both shaders' math: for each K/32 block,
/// (Σ_{kk} A[m,k]·W[n,k]) · rowScale[m,blk] · colScale[n,blk], summed over blocks. Used to spot-check
/// a sample of outputs (computing all M·N would be ~2 G multiply-adds — the full-grid cross-path
/// agreement check covers the rest cheaply).
fn cpu_full_elem(m: usize, n: usize, k_dim: usize) -> f32 {
    let nblk = k_dim / 32;
    let mut acc = 0f32;
    for blk in 0..nblk {
        let mut dot = 0i32;
        for kk in 0..32 {
            let k = blk * 32 + kk;
            dot += a_val(m, k) * w_val(n, k);
        }
        acc += (dot as f32) * row_scale_val(m, blk) * col_scale_val(n, blk);
    }
    acc
}

// ── main ─────────────────────────────────────────────────────────────────────────────────────

fn main() {
    let ctx = match try_init_ctx() {
        Ok(ctx) => ctx,
        Err(msg) => {
            println!("SKIP: {msg}");
            std::process::exit(0);
        }
    };

    let mut failed = false;
    run_single_tile(&ctx, &mut failed);
    run_full_gemm(&ctx, &mut failed);
    if failed {
        eprintln!("CORRECTNESS FAILURE — see the per-bench diagnostics above");
        std::process::exit(1);
    }
}

/// Bench 1: single 16x16x16 tile, one workgroup. Isolates raw epilogue op latency.
fn run_single_tile(ctx: &VkCtx, failed: &mut bool) {
    // Data: same "baseline" packing as coopmat_int8_test (A row-major, B column-major, small
    // non-negative values so int8/uint8 signedness never matters and the CPU ref is exact).
    let a_get = |i: usize, k: usize| ((i + k) % 5) as i32;
    let b_get = |k: usize, j: usize| ((k + j) % 3) as i32;
    let a_bytes = pack16(true, a_get);
    let b_bytes = pack16(false, b_get);

    // Power-of-two fractions -> exactly representable in f32, so the CPU/GPU comparison can use a
    // tight tolerance without float-rounding-order ambiguity muddying a correctness bug.
    let row_scale: [f32; 16] = std::array::from_fn(|i| 1.0 + 0.125 * (i as f32));
    let col_scale: [f32; 16] = std::array::from_fn(|j| 1.0 + 0.0625 * (j as f32));
    let mut scale_bytes = vec![0u8; 128];
    for i in 0..16 {
        scale_bytes[i * 4..i * 4 + 4].copy_from_slice(&row_scale[i].to_ne_bytes());
        scale_bytes[64 + i * 4..64 + i * 4 + 4].copy_from_slice(&col_scale[i].to_ne_bytes());
    }
    let cpu_c = cpu_reference(&row_scale, &col_scale);

    let spv_coopmat2 = compile_or_die(SHADER_COOPMAT2, "single_coopmat2");
    let spv_v1 = compile_or_die(SHADER_V1_BASELINE, "single_v1");

    let inputs: [&[u8]; 3] = [&a_bytes, &b_bytes, &scale_bytes];
    let (us_coopmat2, out_coopmat2) = unsafe {
        run_bench(
            ctx,
            &spv_coopmat2,
            "single/coopmat2",
            &inputs,
            256 * 4,
            &[],
            (1, 1, 1),
            REPS,
            10_000_000_000,
        )
    };
    let (us_v1, out_v1) = unsafe {
        run_bench(
            ctx,
            &spv_v1,
            "single/v1",
            &inputs,
            256 * 4,
            &[],
            (1, 1, 1),
            REPS,
            10_000_000_000,
        )
    };
    let c_coopmat2 = as_f32(&out_coopmat2);
    let c_v1 = as_f32(&out_v1);

    let close = |v: &[f32]| {
        v.iter()
            .zip(cpu_c.iter())
            .all(|(gv, rv)| (*gv - *rv).abs() < 1e-2)
    };
    let correct_coopmat2 = close(&c_coopmat2);
    let correct_v1 = close(&c_v1);
    let agree = c_coopmat2
        .iter()
        .zip(c_v1.iter())
        .all(|(x, y)| (*x - *y).abs() < 1e-2);

    println!(
        "SINGLE-TILE 16x16x16: coopmat2_correct={correct_coopmat2} v1_correct={correct_v1} \
         paths_agree={agree}"
    );
    println!(
        "SINGLE-TILE 16x16x16: coopmat2 {us_coopmat2:.2}us  v1 {us_v1:.2}us  \
         ratio(v1/coopmat2): {:.3}",
        us_v1 / us_coopmat2
    );
    if !correct_coopmat2 || !correct_v1 || !agree {
        *failed = true;
    }
}

/// Bench 2: realistic FULL int8 GEMM (default 512×2048×2048), tiled 16x16-per-workgroup over the
/// full grid, K-looping the K/32 blocks with a per-block descale each — v1's barrier + shared-array
/// occupancy tax fires 64× per workgroup across 4096 workgroups, which the single-tile bench misses.
fn run_full_gemm(ctx: &VkCtx, failed: &mut bool) {
    let (m, n, k) = (FULL_M, FULL_N, FULL_K);
    assert!(
        m.is_multiple_of(16) && n.is_multiple_of(16) && k.is_multiple_of(32),
        "shape must tile"
    );
    let nblk = k / 32;
    println!(
        "FULL-GEMM {m}x{n}x{k}: K/32={nblk} blocks, grid={}x{}={} workgroups, {} reps",
        m / 16,
        n / 16,
        (m / 16) * (n / 16),
        FULL_REPS
    );

    // Activations A[M,K] int8, weights W[N,K] int8 (row-major; column-major B tile loaded from it).
    let mut a_bytes = vec![0u8; m * k];
    for mi in 0..m {
        for ki in 0..k {
            a_bytes[mi * k + ki] = a_val(mi, ki) as i8 as u8;
        }
    }
    let mut w_bytes = vec![0u8; n * k];
    for ni in 0..n {
        for ki in 0..k {
            w_bytes[ni * k + ki] = w_val(ni, ki) as i8 as u8;
        }
    }
    // f16 per-block scales: rowScale[M, K/32], colScale[N, K/32].
    let mut rs_bytes = vec![0u8; m * nblk * 2];
    for mi in 0..m {
        for blk in 0..nblk {
            let h = half::f16::from_f32(row_scale_val(mi, blk));
            rs_bytes[(mi * nblk + blk) * 2..(mi * nblk + blk) * 2 + 2]
                .copy_from_slice(&h.to_ne_bytes());
        }
    }
    let mut cs_bytes = vec![0u8; n * nblk * 2];
    for ni in 0..n {
        for blk in 0..nblk {
            let h = half::f16::from_f32(col_scale_val(ni, blk));
            cs_bytes[(ni * nblk + blk) * 2..(ni * nblk + blk) * 2 + 2]
                .copy_from_slice(&h.to_ne_bytes());
        }
    }
    let mut push = [0u8; 12];
    push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
    push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
    push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());

    let spv_coopmat2 = compile_or_die(SHADER_FULL_COOPMAT2, "full_coopmat2");
    let spv_v1 = compile_or_die(SHADER_FULL_V1, "full_v1");

    let out_bytes = m * n * 4;
    let grid = ((m / 16) as u32 * (n / 16) as u32, 1, 1);
    // Heavy dispatch over host-visible memory -> generous 60s bounded fence (still never u64::MAX).
    let timeout = 60_000_000_000u64;
    let inputs: [&[u8]; 4] = [&a_bytes, &w_bytes, &rs_bytes, &cs_bytes];
    let (us_coopmat2, out_coopmat2) = unsafe {
        run_bench(
            ctx,
            &spv_coopmat2,
            "full/coopmat2",
            &inputs,
            out_bytes,
            &push,
            grid,
            FULL_REPS,
            timeout,
        )
    };
    let (us_v1, out_v1) = unsafe {
        run_bench(
            ctx, &spv_v1, "full/v1", &inputs, out_bytes, &push, grid, FULL_REPS, timeout,
        )
    };
    let c_coopmat2 = as_f32(&out_coopmat2);
    let c_v1 = as_f32(&out_v1);

    // Cross-path agreement over ALL M·N outputs (cheap float compare).
    let agree = c_coopmat2
        .iter()
        .zip(c_v1.iter())
        .all(|(x, y)| (*x - *y).abs() < 1e-2);
    // CPU spot-check on a deterministic sample of coords (computing all M·N on CPU is ~2 G MACs).
    let mut correct_coopmat2 = true;
    let mut correct_v1 = true;
    let mut checked = 0usize;
    let mstep = (m / 16).max(1);
    let nstep = (n / 23).max(1); // coprime-ish strides -> the sample hits varied (m+blk),(n+blk) parities
    let mut mi = 0;
    while mi < m {
        let mut ni = 0;
        while ni < n {
            let want = cpu_full_elem(mi, ni, k);
            let idx = mi * n + ni;
            if (c_coopmat2[idx] - want).abs() >= 1e-2 {
                correct_coopmat2 = false;
            }
            if (c_v1[idx] - want).abs() >= 1e-2 {
                correct_v1 = false;
            }
            checked += 1;
            ni += nstep;
        }
        mi += mstep;
    }

    println!(
        "FULL-GEMM {m}x{n}x{k}: coopmat2_correct={correct_coopmat2} v1_correct={correct_v1} \
         paths_agree={agree} (cpu-checked {checked} sampled coords)"
    );
    println!(
        "FULL-GEMM {m}x{n}x{k}: coopmat2 {us_coopmat2:.2}us  v1 {us_v1:.2}us  \
         ratio(v1/coopmat2): {:.3}",
        us_v1 / us_coopmat2
    );
    if !correct_coopmat2 || !correct_v1 || !agree {
        *failed = true;
    }
}

fn compile_or_die(src: &str, tag: &str) -> Vec<u32> {
    match compile_glsl(src, tag) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("--- {tag} source ---\n{src}");
            panic!("glslc failed on {tag}: {e}");
        }
    }
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Runs `reps` back-to-back dispatches of one shader in a single command buffer, with a full
/// pipeline barrier + `BOTTOM_OF_PIPE` timestamp between each (so GPU-timestamp deltas isolate one
/// dispatch each, not overlapping work), and returns (median microseconds per dispatch, raw C
/// readback bytes). Generic over binding count / push constants / grid / reps / fence timeout so
/// both benches share it. `inputs` become bindings 0..N, the (poisoned) output buffer binding N.
#[allow(clippy::too_many_arguments)]
unsafe fn run_bench(
    ctx: &VkCtx,
    spv: &[u32],
    tag: &str,
    inputs: &[&[u8]],
    out_bytes: usize,
    push: &[u8],
    grid: (u32, u32, u32),
    reps: u32,
    timeout_ns: u64,
) -> (f64, Vec<u8>) {
    let n_bind = (inputs.len() + 1) as u32;
    let bindings: Vec<_> = (0..n_bind)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let ds_layout = ctx
        .device
        .create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
        .expect("create_descriptor_set_layout");
    let (pipeline, pipeline_layout) = make_pipeline(ctx, spv, ds_layout, push.len() as u32);

    // Allocate + fill an input buffer per `inputs`, then the output buffer (poisoned).
    let mut in_bufs: Vec<vk::Buffer> = Vec::with_capacity(inputs.len());
    for data in inputs {
        let (buf, _mem, ptr) = alloc_host_buffer(ctx, data.len() as u64);
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        in_bufs.push(buf);
    }
    let (c_buf, _c_mem, c_ptr) = alloc_host_buffer(ctx, out_bytes as u64);
    std::ptr::write_bytes(c_ptr, 0xAA, out_bytes); // poison — a no-op shader can't look "correct"

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: n_bind,
    }];
    let desc_pool = ctx
        .device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
        .expect("create_descriptor_pool");
    let ds = ctx
        .device
        .allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(std::slice::from_ref(&ds_layout)),
        )
        .expect("allocate_descriptor_sets")[0];
    let all_bufs: Vec<vk::Buffer> = in_bufs
        .iter()
        .copied()
        .chain(std::iter::once(c_buf))
        .collect();
    let infos: Vec<_> = all_bufs
        .iter()
        .map(|&b| {
            vk::DescriptorBufferInfo::default()
                .buffer(b)
                .offset(0)
                .range(vk::WHOLE_SIZE)
        })
        .collect();
    let writes: Vec<_> = (0..n_bind as usize)
        .map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i]))
        })
        .collect();
    ctx.device.update_descriptor_sets(&writes, &[]);

    let query_pool = ctx
        .device
        .create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(reps + 1),
            None,
        )
        .expect("create_query_pool");

    let cmd = ctx
        .device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(ctx.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .expect("allocate_command_buffers")[0];
    ctx.device
        .begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .expect("begin_command_buffer");
    ctx.device
        .cmd_reset_query_pool(cmd, query_pool, 0, reps + 1);
    ctx.device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[ds],
        &[],
    );
    if !push.is_empty() {
        ctx.device
            .cmd_push_constants(cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push);
    }
    ctx.device
        .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query_pool, 0);
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    for rep in 0..reps {
        ctx.device.cmd_dispatch(cmd, grid.0, grid.1, grid.2);
        ctx.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
        ctx.device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            query_pool,
            rep + 1,
        );
    }
    ctx.device
        .end_command_buffer(cmd)
        .expect("end_command_buffer");

    let fence = ctx
        .device
        .create_fence(&vk::FenceCreateInfo::default(), None)
        .expect("create_fence");
    let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    let t0 = Instant::now();
    ctx.device
        .queue_submit(ctx.queue, &[submit], fence)
        .expect("queue_submit");
    // Bounded wait — never u64::MAX, even in a "should be safe" microbenchmark.
    ctx.device
        .wait_for_fences(&[fence], true, timeout_ns)
        .unwrap_or_else(|e| {
            panic!(
                "[{tag}] wait_for_fences: {e:?} (elapsed {:?})",
                t0.elapsed()
            )
        });

    let mut ticks = vec![0u64; (reps + 1) as usize];
    ctx.device
        .get_query_pool_results(
            query_pool,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
        .expect("get_query_pool_results");
    let period = ctx
        .instance
        .get_physical_device_properties(ctx.physical_device)
        .limits
        .timestamp_period as f64; // ns per tick
    let mut deltas_us: Vec<f64> = (0..reps as usize)
        .map(|i| (ticks[i + 1].wrapping_sub(ticks[i]) as f64) * period / 1000.0)
        .collect();
    deltas_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = deltas_us[deltas_us.len() / 2];

    let mut out = vec![0u8; out_bytes];
    std::ptr::copy_nonoverlapping(c_ptr, out.as_mut_ptr(), out.len());

    eprintln!(
        "[{tag}] {reps} reps, median={median_us:.2}us min={:.2}us max={:.2}us",
        deltas_us[0],
        deltas_us[deltas_us.len() - 1]
    );

    (median_us, out)
}
