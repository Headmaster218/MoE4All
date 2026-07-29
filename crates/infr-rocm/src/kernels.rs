//! HIP kernel-source assembly and hiprtc compilation.
//!
//! Each kernel is a `__global__` function taking device pointers. Most operate on f16 or f32
//! buffers — uncovered quantized weights are dequantized to f16 on the host BEFORE they reach a
//! kernel (see `exec.rs`'s dequant cache), so those kernels stay format-agnostic and simple. The
//! `NATIVE_DECODE` kernels (Phase 3 — ALL 24 weight quant formats after R7: Q2_K/Q3_K/Q4_K/Q5_K/
//! Q6_K/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/IQ4_XS/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/
//! TQ1_0/TQ2_0/Q2_0/MXFP4/NVFP4) are the exception: they read the RAW quant bytes and decode each
//! block in-kernel, so no f16 cache is materialized (VRAM ≈ quant_size). Only the DENSE float
//! weight dtypes (F32/BF16 — F16 is already native via `linear_f16`) still take the host
//! convert→f16 path, which for them is a format cast, not a decode.
//!
//! On first use each kernel name is fetched via `hipModuleGetFunction` and cached in a
//! `HashMap`. The module is compiled at most ONCE per (source, arch, HIP stack): `Pipelines::build`
//! first tries the persisted code object (`infr_core::kernel_cache`,
//! `~/.cache/infr/rocm-module-<arch>.bin`) and only falls back to `hiprtcCompileProgram` — ~9.2 s
//! on a cold comgr cache, ~0.25 s even when comgr is hot — on a miss.

use crate::ffi;
use infr_core::config::Config;
use infr_core::error::Result;
use infr_core::kernel_cache::KernelCache;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CString};
use std::sync::Mutex;

/// Terse local shorthand for the shared backend-error constructor.
use infr_core::error::backend as be;

// ── Kernel source ────────────────────────────────────────────────────────────

/// Assemble the complete HIP source string from its parts.
pub fn hip_source() -> String {
    let mut s = String::with_capacity(256 * 1024);
    // The GENERATED parts (everything else is a literal in `HIP_PARTS`): the IQ4 codebook and the
    // IQ2/IQ3 grids are emitted from the host tables so the device tables cannot drift from the
    // decode oracle. Both are self-contained (they depend on nothing), so they lead the source;
    // NATIVE_DECODE and every later part that decodes an IQ format is assembled after them.
    s.push_str(&iq4nl_codebook_src());
    s.push_str(&mxfp4_codebook_src());
    s.push_str(&iquant_grid_src());
    for part in HIP_PARTS {
        s.push_str(part);
    }
    s
}

/// Emit the IQ4_NL / IQ4_XS 16-entry signed codebook (llama.cpp `kvalues_iq4nl`) as HIP source,
/// GENERATED from [`infr_gguf::dequant::KVALUES_IQ4NL`] — the single host-side table the GGUF decode
/// oracle (`dequant_codebook`) and the CPU kernels read. Emitting rather than re-typing it is what
/// makes a drift between the device decode and its oracle impossible.
///
/// **Why baked into the source rather than uploaded as a constant buffer.** The module is a string
/// this backend already assembles at run time and hands to hiprtc, so materializing 16 bytes of it
/// costs nothing and is visible to the optimizer as a compile-time constant. A device buffer would
/// instead need its own allocation plus an extra pointer parameter on EVERY IQ4 kernel — including
/// the WMMA and MoE kernels, whose signatures come from macros shared with the affine formats, so
/// the parameter would have to be threaded through those too — and it would put a memory load in
/// the innermost decode loop, where this is pure ALU.
///
/// The 16 values are packed 4 signed bytes per `u32` and read back with a 4-way word select + a
/// shift and `signed char` cast (the same shape `native_decode.glsl` uses, for the same reason): a
/// dynamically indexed 16-element array lowers to a long select cascade or a scratch spill, while
/// the 4-way select is a couple of `v_cndmask` and the intra-word extract is a single byte extract.
fn iq4nl_codebook_src() -> String {
    kv16_codebook_src(
        "kv_iq4nl",
        "IQ4_NL / IQ4_XS",
        "infr_gguf::dequant::KVALUES_IQ4NL",
        infr_gguf::dequant::KVALUES_IQ4NL,
    )
}

/// Emit the MXFP4 / NVFP4 16-entry signed E2M1 codebook (llama.cpp `kvalues_mxfp4`) as HIP source,
/// GENERATED from [`infr_gguf::dequant::KVALUES_MXFP4`] — R7's twin of [`iq4nl_codebook_src`], on
/// the same emitter and for the same reason (one host-side table, no re-typed second copy).
///
/// The two fp4 formats are structurally IQ4_NL/IQ4_XS with a DIFFERENT codebook and a different
/// scale ENCODING: the values `{0,±1,±2,±3,±4,±6,±8,±12}` are the E2M1 float4 levels written out as
/// exact small integers, so like R4's table they are already the signed int8 dp4a operand and carry
/// no offset. `kv_mxfp4` and `kv_iq4nl` therefore have identical shape and neither may be used for
/// the other's format — the entry sets are disjoint apart from the two zeros.
fn mxfp4_codebook_src() -> String {
    kv16_codebook_src(
        "kv_mxfp4",
        "MXFP4 / NVFP4 (E2M1)",
        "infr_gguf::dequant::KVALUES_MXFP4",
        infr_gguf::dequant::KVALUES_MXFP4,
    )
}

/// The shared body of the two 16-entry signed-codebook emitters above: pack `kv` 4 signed bytes per
/// `u32` (index `i` in byte `i & 3` of word `i >> 2`, little-endian) and emit a `__device__` accessor
/// that reads it back with a 4-way word select + a byte extract.
fn kv16_codebook_src(fname: &str, family: &str, host_path: &str, kv: [i8; 16]) -> String {
    let w = |n: usize| -> u32 {
        (0..4).fold(0u32, |acc, b| {
            acc | ((kv[n * 4 + b] as u8 as u32) << (8 * b))
        })
    };
    let (w0, w1, w2, w3) = (w(0), w(1), w(2), w(3));
    format!(
        "\n// ── {family} codebook — GENERATED from {host_path} ──\n\
         // {kv:?}\n\
         __device__ __forceinline__ int {fname}(int idx) {{\n\
         \x20   unsigned int w = (idx < 8) ? ((idx < 4) ? {w0:#010x}u : {w1:#010x}u)\n\
         \x20                             : ((idx < 12) ? {w2:#010x}u : {w3:#010x}u);\n\
         \x20   return (int)(signed char)((w >> ((idx & 3) * 8)) & 0xFFu);\n\
         }}\n"
    )
}

/// Emit the IQ1 / IQ2 / IQ3 **grids**, the shared `ksigns` sign-pattern table and the IQ1 `delta`
/// constant as HIP source, GENERATED from [`infr_core::iquant_grids`] and
/// [`infr_gguf::dequant::IQ1S_DELTA`] — the single host-side sources of truth that
/// `infr_gguf::dequant::dequant_codebook` (the decode oracle), the CPU kernels and Vulkan's
/// `native_grids.glsl` all read. As with R4's codebook, emitting rather than re-typing is what
/// makes a drift between the device decode and its oracle impossible; a unit test parses the
/// emitted text back and requires it to BE the host statics.
///
/// **Why baked into the module source rather than uploaded as a device buffer.** These are much
/// bigger than R4's 16-byte codebook — 33.1 KiB across seven tables, of which R6's 2048-entry
/// `g_iq1s` is 16 KiB on its own — so the trade was re-measured rather than inherited:
///
/// * A device buffer would need an extra pointer parameter on EVERY IQ1/IQ2/IQ3 kernel. Those
///   signatures come from macros SHARED with the affine formats (`GEN_LINEAR`, `GEN_EMBED`,
///   `GEN_DEQF16`, `GEN_MOE_FFN`, `GEN_MOE_GATE_UP`, `GEN_MOE_DOWN` and their routed twins), and
///   `exec.rs` binds those kernels' arguments positionally from ONE per-op arm each, so the
///   parameter would have to be threaded through every covered format's dispatch as well.
/// * It buys nothing at run time. On AMDGCN a module-scope `__device__ const` array IS device
///   global memory: a dynamic index compiles to the same `global_load` a buffer pointer would,
///   through the L2/scalar caches, with the base address folded into the instruction. (This is
///   exactly where HIP differs from Vulkan, where the same tables must be mirrored into LDS by
///   hand — glslang/ACO materialize a dynamically-indexed `const` array into per-invocation
///   scratch, which cost the IQ2_S GEMV ~1 MB of scratch per wave; see
///   `infr-vulkan/build.rs::gen_grids`. There is no such lowering here, so no LDS mirror and no
///   `grid_init()` barrier.)
/// * The measured cost is hiprtc parse time on a COLD comgr cache only, and it is small next to
///   the kernels themselves (see the R5 numbers on `moe_expert_kernel`).
///
/// The IQ1/IQ2 grids are `unsigned long long` (8 packed signed bytes per entry) and the IQ3 grids
/// `unsigned int` (4), matching the host types exactly so the index arithmetic is the oracle's.
fn iquant_grid_src() -> String {
    use infr_core::iquant_grids as g;
    let mut s = String::with_capacity(160 * 1024);
    s.push_str(
        "\n// ── IQ1/IQ2/IQ3 grids — GENERATED from infr_core::iquant_grids ──────────────────\n\
         // The stored code is an INDEX into a table of packed signed-byte vectors (8 bytes per\n\
         // entry for IQ1/IQ2, 4 for IQ3); a separate sign bit per element negates it (IQ2/IQ3) or\n\
         // a separate ±`IQ1S_DELTA` addend shifts it (IQ1). `ksigns_iq2xs` expands a 7-bit\n\
         // sign-pattern index into the 8 sign bits (IQ2_S / IQ3_S carry raw sign BYTES instead and\n\
         // do not use it; the IQ1 formats have no sign field at all).\n",
    );
    // One emitter for all six tables: `per_line` keeps the generated text greppable, and the
    // fixed-width hex makes the round-trip parse in `codebook_tests` unambiguous.
    let mut table =
        |ty: &str, name: &str, vals: &mut dyn Iterator<Item = String>, per_line: usize| {
            let items: Vec<String> = vals.collect();
            s.push_str(&format!(
                "__device__ static const {ty} {name}[{}] = {{",
                items.len()
            ));
            for (i, v) in items.iter().enumerate() {
                if i % per_line == 0 {
                    s.push_str("\n    ");
                }
                s.push_str(v);
                s.push(',');
            }
            s.push_str("\n};\n");
        };
    table(
        "unsigned char",
        "ksigns_iq2xs",
        &mut g::KSIGNS_IQ2XS.iter().map(|v| format!("{v}")),
        16,
    );
    for (name, grid) in [
        ("g_iq2xxs", &g::IQ2XXS_GRID[..]),
        ("g_iq2xs", &g::IQ2XS_GRID[..]),
        ("g_iq2s", &g::IQ2S_GRID[..]),
        // R6: IQ1_S and IQ1_M SHARE this one — 2048 entries, an 11-bit index (the widest in the
        // family), and every packed byte is −1/0/+1, which is what lets the int8 tier fold the
        // fractional delta into the code (`8·gv ± 1`). Same `unsigned long long` type as the IQ2
        // grids, so `gsb8` reads it unchanged.
        ("g_iq1s", &g::IQ1S_GRID[..]),
    ] {
        table(
            "unsigned long long",
            name,
            &mut grid.iter().map(|v| format!("{v:#018x}ull")),
            4,
        );
    }
    for (name, grid) in [
        ("g_iq3xxs", &g::IQ3XXS_GRID[..]),
        ("g_iq3s", &g::IQ3S_GRID[..]),
    ] {
        table(
            "unsigned int",
            name,
            &mut grid.iter().map(|v| format!("{v:#010x}u")),
            8,
        );
    }
    // R6: the IQ1 ADDEND, emitted from the host constant for the same reason the tables are — a
    // re-typed 0.125f would be a second source of truth for the one number that distinguishes this
    // family from R5's. `{:e}` keeps it a valid C float literal for any value the host might hold.
    s.push_str(&format!(
        "// IQ1_S / IQ1_M grid addend — GENERATED from infr_gguf::dequant::IQ1S_DELTA\n\
         #define IQ1S_DELTA {:e}f\n",
        infr_gguf::dequant::IQ1S_DELTA
    ));
    s
}

/// The individual kernel source parts (one per kernel, so the hot-patching assembly is greppable).
const HIP_PARTS: &[&str] = &[
    RMSNORM,
    RMSNORM_ADD,
    SOFTMAX,
    LINEAR_F16,
    QK_NORM,
    ROPE,
    QK_NORM_ROPE,
    GATED_RMSNORM,
    GATED_ACT,
    ADD,
    ADD_BIAS,
    SCALE,
    MUL_VEC,
    SOFTCAP,
    COPY,
    COPY_STRIDED,
    EMBED_GATHER,
    ARGMAX,
    WRITE_KV,
    ATTENTION,
    ATTENTION_PF,
    ATTENTION_FLASH,
    ATTENTION_SPLIT,
    ATTENTION_SPLIT_FLASH,
    MOE_FFN,
    CONV1D_SILU,
    DELTANET,
    DELTANET_DECODE,
    DELTANET_CHUNKED,
    MOE_SHARED_EXPERT_ADD,
    NATIVE_DECODE,
    DEQUANT_F16,
    MOE_FFN_NATIVE,
    INT8_DECODE,
    RMSNORM_QUANT_I8,
    MOE_FFN_INT8,
    WMMA_PREFILL,
    MOE_ROUTING,
    MOE_ID_MULTI,
    MOE_ID_BUCKET,
    SAMPLE_TOP_K,
    ARGMAX_PROB,
];

// One BLOCK per row (grid.x = rows, blockDim.x = RMS_BLOCK). The sum-of-squares is a strided
// partial per thread then a shared-mem tree reduce, so at m=1 (decode) the `dim` reduction spreads
// across a full wave instead of one serial thread. The tree reduce reorders the float adds vs the
// reference serial sum (sub-ULP); greedy decode absorbs it and the golden hash is verified unmoved.
// blockDim.x MUST be a power of two ≤ 256 (shared-mem `sdata[256]` bound + the tree-reduce step).
const RMSNORM: &str = r#"
extern "C" __global__ void rmsnorm(
    const float* __restrict__ x,     // [rows, dim] — F32 activation
    const __half* __restrict__ weight,// [dim] — dequantized F16
    float* __restrict__ dst,         // [rows, dim]
    int rows,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* xr = x + row * dim;
    float local = 0.0f;
    for (int i = tid; i < dim; i += nt) {
        float v = xr[i];
        local += v * v;
    }
    __shared__ float sdata[256];
    sdata[tid] = local;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float ss = sdata[0] / (float)dim;
    float rms = 1.0f / sqrtf(ss + eps);
    float* d = dst + row * dim;
    for (int i = tid; i < dim; i += nt) {
        d[i] = xr[i] * rms * __half2float(weight[i]);
    }
}
"#;

const RMSNORM_ADD: &str = r#"
extern "C" __global__ void rmsnorm_add(
    const float* __restrict__ x,      // [rows, dim] — F32 activation
    const __half* __restrict__ weight, // [dim] — dequantized F16 weight
    float* __restrict__ dst,           // [rows, dim] read + write in-place (F32)
    int rows,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* xr = x + row * dim;
    float local = 0.0f;
    for (int i = tid; i < dim; i += nt) {
        float v = xr[i];
        local += v * v;
    }
    __shared__ float sdata[256];
    sdata[tid] = local;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float ss = sdata[0] / (float)dim;
    float rms = 1.0f / sqrtf(ss + eps);
    float* d = dst + row * dim;
    for (int i = tid; i < dim; i += nt) {
        d[i] += xr[i] * rms * __half2float(weight[i]);
    }
}
"#;

const SOFTMAX: &str = r#"
extern "C" __global__ void softmax(
    const float* __restrict__ x, // [rows, dim]
    float* __restrict__ dst,     // [rows, dim]
    int rows,
    int dim,
    float scale
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * dim;
    float* dr = dst + row * dim;
    // find max
    float m = xr[0] * scale;
    for (int i = 1; i < dim; i++) {
        float v = xr[i] * scale;
        if (v > m) m = v;
    }
    // exp sum
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float v = expf(xr[i] * scale - m);
        dr[i] = v;
        sum += v;
    }
    // normalize
    float inv = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
        dr[i] *= inv;
    }
}
"#;

const LINEAR_F16: &str = r#"
extern "C" __global__ void linear_f16(
    const float* __restrict__ x,     // [m, in_f]
    const __half* __restrict__ w,    // [out_f, in_f] row-major
    float* __restrict__ dst,         // [m, out_f]
    int m,
    int in_f,
    int out_f
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    // Each thread handles 4 outputs via loop over out_f
    for (int o = tid; o < out_f; o += blockDim.x) {
        float acc = 0.0f;
        const float* xr = x + row * in_f;
        const __half* wr = w + o * in_f;
        for (int i = 0; i < in_f; i++) {
            acc += xr[i] * __half2float(wr[i]);
        }
        dst[row * out_f + o] = acc;
    }
}
"#;

const QK_NORM: &str = r#"
extern "C" __global__ void qk_norm(
    const float* __restrict__ x,       // [rows, n_head, head_dim] or strided
    const __half* __restrict__ weight, // [head_dim]
    float* __restrict__ dst,           // [rows, n_head, head_dim]
    int rows,
    int n_head,
    int head_dim,
    float eps,
    int x_stride       // per-row stride in elements; 0 = packed
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int r = head / n_head;
    int h = head % n_head;
    int stride = (x_stride > 0) ? x_stride : (n_head * head_dim);
    int off = r * stride + h * head_dim;
    float ss = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float v = x[off + i];
        ss += v * v;
    }
    ss /= (float)head_dim;
    float rms = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < head_dim; i++) {
        dst[off + i] = x[off + i] * rms * __half2float(weight[i]);
    }
}
"#;

const ROPE: &str = r#"
extern "C" __global__ void rope(
    float* __restrict__ x,              // [rows, n_head, head_dim] or strided — mutated in-place
    const int* __restrict__ positions,  // [rows]
    const float* __restrict__ freq_factors, // optional (null = unused)
    int rows,
    int n_head,
    int head_dim,
    int rope_dim,                       // first rope_dim elements get RoPE
    float theta,
    int x_stride       // per-row stride in elements; 0 = packed (n_head * head_dim)
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int pos = positions[row];
    // Per-row stride: 0 = packed. Heads stay packed within a strided row (off = h*head_dim),
    // mirroring the fused qk_norm_rope kernel's stride convention. Non-zero x_stride selects a
    // rotated slice out of a wider row buffer without a preceding gather (qwen35's q+g case).
    int stride = (x_stride > 0) ? x_stride : (n_head * head_dim);
    float* xr = x + row * stride;
    int half = rope_dim / 2;
    for (int h = 0; h < n_head; h++) {
        float* xh = xr + h * head_dim;
        for (int p = 0; p < half; p++) {
            // ggml NORM RoPE: INTERLEAVED pairs (2p, 2p+1) — matches infr-cpu Op::Rope, the Metal
            // `rope_f32` kernel, and the Vulkan `rope` shader. (The NEOX split-half rotation lives
            // in the fused qk_norm_rope kernel; the two styles are NOT interchangeable.)
            float freq = 1.0f / powf(theta, (float)(2 * p) / (float)rope_dim);
            if (freq_factors != nullptr) {
                freq /= freq_factors[p]; // Gemma proportional RoPE divides the per-pair angle
            }
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = xh[2 * p];
            float x1 = xh[2 * p + 1];
            xh[2 * p]     = x0 * c - x1 * s;
            xh[2 * p + 1] = x0 * s + x1 * c;
        }
    }
}
"#;

const QK_NORM_ROPE: &str = r#"
// Store ONE output element of the fused rope. Unfused (`kv == nullptr`) that is the packed f32
// scratch slot the standalone `write_kv` would then have read; fused (F1d, the `kv_write` peephole)
// it is the f16 KV-cache slot `write_kv` would have written, cast with the SAME `__float2half` from
// the SAME f32 value — an f32 register and an f32 round-trip through DRAM hold identical bits, so
// the cache ends up byte-for-byte what the elided kernel wrote. `kv` is a UNIFORM kernel argument,
// so this branch is scalar (SGPR): every lane of every wave takes the same side, and the unfused
// path pays a single s_cbranch, not per-lane divergence.
static __device__ __forceinline__ void qnr_store(
    float* __restrict__ dst, __half* __restrict__ kv, long doff, long kvoff, int i, float v
) {
    if (kv != nullptr) kv[kvoff + i] = __float2half(v);
    else               dst[doff + i] = v;
}

// One WAVE (32 lanes) per (row, head), vs the old one THREAD per head. At decode (rows==1) the old
// grid launched rows*n_head=n_head threads → ~16 threads on ONE CU each serially running a head_dim
// RMSNorm sum-of-squares reduction + the per-pair RoPE transcendentals (measured ~17% of decode).
// The dominant cost is the RoPE loop's per-pair powf/cosf/sinf, so THAT is what we fan across the
// wave: the pass-through norm-scale and the NEOX split-half RoPE pairs are partitioned across the 32
// lanes (each output element written by exactly ONE lane, no cross-lane reduction). The cheap
// sum-of-squares stays a full sequential 0..head_dim loop run redundantly on every lane (128 broadcast
// loads + FMA) — deliberately NOT a butterfly reduce, so `rms` is BIT-IDENTICAL to the old kernel and
// each output element reproduces the old arithmetic exactly (Qwen3 golden hash held unmoved). For
// rows>1 (prefill) the grid is a strict superset — one wave per (row,head) instead of one thread —
// so the strided q+g layout, freq_factors, and (i,i+half) rotation are byte-for-byte the same math.
extern "C" __global__ void qk_norm_rope(
    const float* __restrict__ x,        // input: [rows, x_stride] strided OR [rows, n_head*head_dim] packed
    const __half* __restrict__ weight,  // [head_dim]
    const int* __restrict__ positions,  // [rows]
    const float* __restrict__ freq_factors, // optional
    float* __restrict__ dst,            // OUTPUT: always packed [rows, n_head, head_dim]
    int rows,
    int n_head,
    int head_dim,
    int rope_dim,                       // first rope_dim elements get RoPE
    float eps,
    float theta,
    int x_stride,      // per-row stride in elements; 0 = packed (n_head * head_dim)
    __half* __restrict__ kv, // F1d: fused KV-cache write target (null = write the f32 `dst` scratch)
    int kv_row,        // first cache ROW to write (the absorbed WriteKv's `pos`)
    int kv_stride      // per-row elements in the cache (= n_head * head_dim when fused)
) {
    int head = blockIdx.x;             // one block == one wave == one (row, head)
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int tid = threadIdx.x;             // lane 0..31
    int r = head / n_head;
    int h = head % n_head;
    int pos = positions[r];
    // Read base: strided input packs each head into an `x_stride/n_head`-wide block (query is the
    // first head_dim elements), matching the qwen35 interleaved q+g buffer. Packed input (x_stride
    // == 0) reads the natural head slice. Write base is ALWAYS the packed [rows, n_head, head_dim]
    // slot — mirrors infr-cpu QkNormRope and the Metal `qknormrope_f32` kernel.
    int head_stride = (x_stride > 0) ? (x_stride / n_head) : head_dim;
    int xoff = (x_stride > 0) ? (r * x_stride + h * head_stride) : (head * head_dim);
    long doff = (long)head * head_dim;
    // Fused (F1d) write base: cache row `kv_row + r`, head `h` packed within it. The peephole only
    // fuses when `kv_stride == n_head * head_dim`, so this tiles the cache row exactly the way the
    // elided `write_kv` (one thread per (row, element) of a packed src) did.
    long kvoff = (long)(kv_row + r) * kv_stride + (long)h * head_dim;
    // rmsnorm over the head_dim query slice. Run the FULL sequential 0..head_dim sum on every lane
    // (redundant, but bit-identical to the old single-thread sum → `rms` unchanged; the sum is cheap
    // vs the RoPE transcendentals, and the reads broadcast across the wave).
    float ss = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float v = x[xoff + i];
        ss += v * v;
    }
    ss /= (float)head_dim;
    float rms = 1.0f / sqrtf(ss + eps);
    // Pass-through dims [rope_dim, head_dim): normed (× weight), no rotation. Strided across lanes.
    for (int i = rope_dim + tid; i < head_dim; i += 32) {
        qnr_store(dst, kv, doff, kvoff, i, x[xoff + i] * rms * __half2float(weight[i]));
    }
    // rope (NEOX split-half pairs (i, i+half)) on the first rope_dim elements, from normed values.
    // Each lane owns pairs i = tid, tid+32, … < half and writes BOTH the (i) and (i+half) slot.
    int half = rope_dim / 2;
    for (int i = tid; i < half; i += 32) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
        if (freq_factors != nullptr) {
            freq /= freq_factors[i]; // proportional RoPE divides the per-pair angle (matches CPU/Metal)
        }
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float a = x[xoff + i]        * rms * __half2float(weight[i]);
        float b = x[xoff + i + half] * rms * __half2float(weight[i + half]);
        qnr_store(dst, kv, doff, kvoff, i,        a * c - b * s);
        qnr_store(dst, kv, doff, kvoff, i + half, a * s + b * c);
    }
}
"#;

const GATED_RMSNORM: &str = r#"
extern "C" __global__ void gated_rmsnorm(
    const float* __restrict__ x,        // [rows, n_head, head_dim]
    const __half* __restrict__ weight,  // [head_dim]
    const float* __restrict__ gate,     // [rows, n_head, head_dim]
    float* __restrict__ dst,            // [rows, n_head, head_dim]
    int rows,
    int n_head,
    int head_dim,
    float eps
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int off = head * head_dim;
    // rmsnorm
    float ss = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float v = x[off + i];
        ss += v * v;
    }
    ss /= (float)head_dim;
    float rms = 1.0f / sqrtf(ss + eps);
    // gate (SiLU) multiply
    for (int i = 0; i < head_dim; i++) {
        float g = gate[off + i];
        float silu_g = g / (1.0f + expf(-g));
        dst[off + i] = x[off + i] * rms * __half2float(weight[i]) * silu_g;
    }
}
"#;

const GATED_ACT: &str = r#"
extern "C" __global__ void gated_act(
    const float* __restrict__ gate, // [rows, nff] or strided
    const float* __restrict__ up,   // [rows, nff] or strided
    float* __restrict__ dst,        // [rows, nff]
    int rows,
    int nff,
    int act_type,       // 0=SiLU, 1=GeLU(tanh), 2=Sigmoid
    int up_off,         // element offset into up
    int up_stride,      // 0 = packed
    int gate_stride,    // 0 = packed
    int gate_block_width // 0 = no interleave
) {
    // One thread per OUTPUT element (row, i). Each dst[i] = act(gate[i]) * up[i] is fully
    // independent (no reduction) → bit-exact vs the old per-row serial loop, but at m=1 (decode)
    // this fans the `nff` outputs across `ceil(rows*nff/block)` blocks instead of stranding the
    // whole loop on one thread of one CU.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * nff;
    if (tid >= total) return;
    int row = tid / nff;
    int i = tid % nff;
    int effective_gate_stride = (gate_stride > 0) ? gate_stride : nff;
    int effective_up_stride = (up_stride > 0) ? up_stride : nff;
    int gate_off = row * effective_gate_stride;
    int up_off_base = up_off + row * effective_up_stride;
    float g;
    if (gate_block_width > 0) {
        // Interleaved qg row: per head a [query(headw) | gate(headw)] block, so the full
        // per-head block is `gate_block_width` wide and the gate half starts at `headw`.
        // Output index `i` addresses the PACKED gate (headw per head); map it to the strided
        // gate half. Matches infr-cpu's GatedAct (headw = gate_block_width / 2).
        int headw = gate_block_width / 2;
        int head = i / headw;
        int off = i % headw;
        g = gate[gate_off + head * gate_block_width + headw + off];
    } else {
        g = gate[gate_off + i];
    }
    float u = up[up_off_base + i];
    float a;
    if (act_type == 0) {
        // SiLU: x * sigmoid(x)
        a = g / (1.0f + expf(-g));
    } else if (act_type == 1) {
        // GeLU (tanh approx)
        float x3 = g * g * g;
        float c = 0.044715f;
        a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + c * x3)));
    } else {
        // Sigmoid
        a = 1.0f / (1.0f + expf(-g));
    }
    dst[row * nff + i] = a * u;
}
"#;

const ADD: &str = r#"
extern "C" __global__ void add(
    const float* __restrict__ a, // [n]
    const float* __restrict__ b, // [n]
    float* __restrict__ dst,     // [n]
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = a[i] + b[i];
}
"#;

const ADD_BIAS: &str = r#"
extern "C" __global__ void add_bias(
    const float* __restrict__ x,    // [rows, n]
    const float* __restrict__ bias, // [n]
    float* __restrict__ dst,        // [rows, n]
    int rows,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * n;
    float* dr = dst + row * n;
    for (int i = 0; i < n; i++) {
        dr[i] = xr[i] + bias[i];
    }
}
"#;

const SCALE: &str = r#"
extern "C" __global__ void scale(
    const float* __restrict__ x, // [n]
    float* __restrict__ dst,     // [n]
    float s,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = x[i] * s;
}
"#;

const MUL_VEC: &str = r#"
extern "C" __global__ void mul_vec(
    const float* __restrict__ x,   // [rows, n]
    const float* __restrict__ vec, // [n]
    float* __restrict__ dst,       // [rows, n]
    int rows,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * n;
    float* dr = dst + row * n;
    for (int i = 0; i < n; i++) {
        dr[i] = xr[i] * vec[i];
    }
}
"#;

const SOFTCAP: &str = r#"
extern "C" __global__ void softcap(
    const float* __restrict__ x, // [n]
    float* __restrict__ dst,     // [n]
    float cap,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = cap * tanhf(x[i] / cap);
}
"#;

const COPY: &str = r#"
extern "C" __global__ void copy(
    const float* __restrict__ src,
    int src_off,
    float* __restrict__ dst,
    int dst_off,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[dst_off + i] = src[src_off + i];
}
"#;

const COPY_STRIDED: &str = r#"
extern "C" __global__ void copy_strided(
    const float* __restrict__ src,
    int src_off,
    int src_stride,
    float* __restrict__ dst,
    int dst_off,
    int dst_stride,
    int rows,
    int n
) {
    int r = blockIdx.x;                    // one block per row
    if (r >= rows) return;
    int tid = threadIdx.x;
    int bs  = blockDim.x;                  // threads per row, min(n, 256)
    const float* sr = src + src_off + r * src_stride;
    float* dr = dst + dst_off + r * dst_stride;
    // Vectorised float4 portion — each thread copies 16 B per iteration, so n=2048 takes
    // 2 iterations per thread at bs=256 (down from 2048 serial iterations on the old kernel).
    int nf4 = n >> 2;
    for (int i = tid; i < nf4; i += bs) {
        float4 v = *(const float4*)(sr + (i << 2));
        *(float4*)(dr + (i << 2)) = v;
    }
    // Scalar tail (0-3 elements).
    int nn = nf4 << 2;
    for (int i = nn + tid; i < n; i += bs) {
        dr[i] = sr[i];
    }
}
"#;

const EMBED_GATHER: &str = r#"
extern "C" __global__ void embed_gather(
    const int* __restrict__ ids,     // [rows]
    const __half* __restrict__ table, // [vocab, dim]
    float* __restrict__ dst,          // [rows, dim]
    int rows,
    int dim,
    float scale                       // per-op embedding scale (sqrt(n_embd) for Gemma; 1.0 otherwise)
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int id = ids[row];
    const __half* tr = table + id * dim;
    float* dr = dst + row * dim;
    for (int i = 0; i < dim; i++) {
        dr[i] = __half2float(tr[i]) * scale;
    }
}
"#;

// One BLOCK per row (grid.x = rows). Each thread scans a strided slice of the vocab tracking its
// best (strict `>`, ascending scan ⇒ lowest index on ties), then a shared-mem tree reduce keeps the
// lower index on equal values. That reproduces the reference serial first-max tie rule EXACTLY, so
// the argmax index is bit-identical — only the CU occupancy changes (at m=1 the vocab reduction
// spreads across a wave instead of one serial thread over 151936 elements).
const ARGMAX: &str = r#"
#define ARGMAX_CHUNK 2048  // floats per block — 8 per thread at bs=256

// Pass 1 — one block per (row, chunk) reduces its chunk to a (value, index) pair.
extern "C" __global__ void argmax_partial(
    const float* __restrict__ x,  // [rows, n]
    float* __restrict__ pval,     // [rows * n_chunks]  chunk max values
    int* __restrict__ pidx,       // [rows * n_chunks]  chunk max indices
    int rows,
    int n,
    int n_chunks
) {
    int gidx = blockIdx.x;                          // linearised: row * n_chunks + chunk
    int row = gidx / n_chunks;
    if (row >= rows) return;
    int chunk = gidx - row * n_chunks;
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    int j0 = chunk * ARGMAX_CHUNK;
    int j1 = j0 + ARGMAX_CHUNK;
    if (j1 > n) j1 = n;
    int len = j1 - j0;
    if (len <= 0) {                           // empty trailing chunk
        if (tid == 0) { pval[gidx] = -3.402823466e+38f; pidx[gidx] = 0; }
        return;
    }

    const float* xr = x + (long)row * n + j0;

    // Cooperative reduce: every thread reads one or more contiguous elements, then tree-reduce
    // through LDS. Coalesced per warp (32 consecutive floats per step at bs=256).
    float best_val = -3.402823466e+38f;
    int best_idx = 0;
    for (int i = tid; i < len; i += bs) {
        float v = xr[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    best_idx += j0;   // restore global index

    // Warp reduce (no barrier, 5 shfl steps)
    float wv = best_val;
    int wi = best_idx;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(wv, off);
        int oi = __shfl_xor(wi, off);
        if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
    }
    // LDS: only warp leaders
    __shared__ float sval[256];
    __shared__ int sidx[256];
    if (tid % 32 == 0) { sval[tid >> 5] = wv; sidx[tid >> 5] = wi; }
    __syncthreads();
    int nw = bs >> 5;
    for (int s = nw >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) { pval[gidx] = sval[0]; pidx[gidx] = sidx[0]; }
}

// Pass 2 — one block per row merges the per-chunk partials.
extern "C" __global__ void argmax_combine(
    const float* __restrict__ pval,  // [rows * n_chunks]
    const int* __restrict__ pidx,    // [rows * n_chunks]
    float* __restrict__ dst,         // [rows] — u32 bit-pattern in f32 slot
    int rows,
    int n_chunks
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    int base = row * n_chunks;

    float best_val = -3.402823466e+38f;
    int best_idx = 0;
    for (int i = tid; i < n_chunks; i += bs) {
        float v = pval[base + i];
        int idx = pidx[base + i];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }
    // Warp reduce
    float wv = best_val;
    int wi = best_idx;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(wv, off);
        int oi = __shfl_xor(wi, off);
        if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
    }
    __shared__ float sval[256];
    __shared__ int sidx[256];
    if (tid % 32 == 0) { sval[tid >> 5] = wv; sidx[tid >> 5] = wi; }
    __syncthreads();
    int nw = bs >> 5;
    for (int s = nw >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) dst[row] = __int_as_float(sidx[0]);
}
"#;

// ── Sample (top-k + top-p stochastic sampling) ────────────────────────────────
//
// Two-stage GPU-resident sampler: Stage 1 radix-selects the top-k per-slice
// (256 workgroups of 256 threads, one slice each), Stage 2 merges all candidates
// in one workgroup and does the nucleus/CDF walk with a host-supplied uniform draw.
//
// The order of operations mirrors the host `sample_logits` exactly (top-k select →
// softmax(temp) → nucleus cutoff → CDF walk), so the same u draws the same token
// regardless of backend — verified bit-identical against the CPU reference for all
// shapes.
//
// `f2ui` is the float→uint order-preserving map (sign-bit flip of the IEEE 754
// encoding): positive floats map to `[0x80000000, 0xFFFFFFFF]`, negative to
// `[0x00000000, 0x7FFFFFFF]`, so an unsigned radix sort directly compares them in
// descending float value order.
const SAMPLE_TOP_K: &str = r#"
static __device__ __forceinline__ unsigned int f2ui(float x) {
    unsigned int y = __float_as_uint(x);
    if (y & 0x80000000u) y = ~y; else y |= 0x80000000u;
    return y;
}

// Stage 1: 256-workgroup VOCAB scan — each block radix-selects the top-k of its
// contiguous slice and writes k (val, idx) candidate pairs to a global buffer.
extern "C" __global__ void sample_topk_partial(
    const float* __restrict__ logits,  // [n] vocab logits
    float* __restrict__ cand,          // [256*k] values, [256*k] u32 idx bit-patterns
    int n,                             // vocab size
    int top_k                          // 2..=64
) {
    int tid = threadIdx.x;
    int bs  = blockDim.x;   // 256
    int k   = top_k;

    int chunk = (n + bs - 1) / bs;
    int lo = (int)blockIdx.x * chunk;
    int hi = lo + chunk;
    if (hi > n) hi = n;

    // Phase A: radix N-ary select — MSB-first, 4 bits per pass over 8 levels.
    unsigned int prefix = 0u;
    unsigned int krem   = (unsigned int)k;
    __shared__ unsigned int bucket[16];
    __shared__ unsigned int sh_sel;
    __shared__ unsigned int sh_krem;

    for (int level = 0; level < 8; level++) {
        unsigned int shift = 28u - 4u * (unsigned int)level;
        unsigned int himask = (shift + 4u >= 32u) ? 0u : (0xFFFFFFFFu << (shift + 4u));
        unsigned int pfix = prefix & himask;
        if (tid < 16) { bucket[tid] = 0u; }
        __syncthreads();
        for (int i = lo + tid; i < hi; i += bs) {
            unsigned int key = f2ui(logits[i]);
            if ((key & himask) == pfix) {
                atomicAdd(&bucket[(key >> shift) & 0xFu], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u;
            unsigned int sel = 0u;
            unsigned int kr  = krem;
            for (int b = 15; b >= 0; b--) {
                unsigned int c = bucket[(unsigned int)b];
                if (cum + c >= krem) { sel = (unsigned int)b; kr = krem - cum; break; }
                cum += c;
            }
            sh_sel = sel; sh_krem = kr;
        }
        __syncthreads();
        prefix |= (sh_sel << shift);
        krem   = sh_krem;
    }
    unsigned int thresh = prefix;

    // Phase B: gather values with key ≥ threshold into shared memory.
    __shared__ unsigned int gcnt;
    __shared__ float gval[64];
    __shared__ unsigned int gidx[64];
    if (tid == 0) { gcnt = 0u; }
    __syncthreads();
    for (int i = lo + tid; i < hi; i += bs) {
        float v = logits[i];
        if (f2ui(v) >= thresh) {
            unsigned int slot = atomicAdd(&gcnt, 1u);
            if (slot < (unsigned int)k) { gval[slot] = v; gidx[slot] = (unsigned int)i; }
        }
    }
    __syncthreads();

    // Stage-1 epilogue: write candidates (tid 0 only — one warp leader).
    if (tid == 0) {
        unsigned int m = gcnt;
        if (m > (unsigned int)k) m = (unsigned int)k;
        unsigned int base  = (unsigned int)blockIdx.x * (unsigned int)k;
        unsigned int ncand = 256u * (unsigned int)k;
        for (unsigned int j = 0u; j < (unsigned int)k; j++) {
            cand[base + j]           = (j < m) ? gval[j] : -1e30f;
            cand[ncand + base + j]   = __uint_as_float((j < m) ? gidx[j] : 0u);
        }
    }
}

// Stage 2: 1-workgroup merge of the 256*k stage-1 candidates → radix-select
// global top-k → gather → sort-desc → softmax(temp) → nucleus(top_p) → CDF sample.
extern "C" __global__ void sample_topk_combine(
    const float* __restrict__ cand,    // [n_cand] values + [n_cand] u32 idx bit-patterns
    const float* __restrict__ u_buf,   // 1-float uniform draw
    float* __restrict__ out_id,        // [1] u32 token id as float
    int n_cand,                        // = 256 * top_k
    int top_k,
    float temp,
    float top_p
) {
    int tid = threadIdx.x;
    int bs  = blockDim.x;   // 256
    int k   = top_k;
    int lo  = 0;
    int hi  = n_cand;

    // Phase A: radix N-ary select (identical to stage 1, but reads from cand).
    unsigned int prefix = 0u;
    unsigned int krem   = (unsigned int)k;
    __shared__ unsigned int bucket[16];
    __shared__ unsigned int sh_sel;
    __shared__ unsigned int sh_krem;

    for (int level = 0; level < 8; level++) {
        unsigned int shift = 28u - 4u * (unsigned int)level;
        unsigned int himask = (shift + 4u >= 32u) ? 0u : (0xFFFFFFFFu << (shift + 4u));
        unsigned int pfix = prefix & himask;
        if (tid < 16) { bucket[tid] = 0u; }
        __syncthreads();
        for (int i = lo + tid; i < hi; i += bs) {
            unsigned int key = f2ui(cand[i]);
            if ((key & himask) == pfix) {
                atomicAdd(&bucket[(key >> shift) & 0xFu], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0u;
            unsigned int sel = 0u;
            unsigned int kr  = krem;
            for (int b = 15; b >= 0; b--) {
                unsigned int c = bucket[(unsigned int)b];
                if (cum + c >= krem) { sel = (unsigned int)b; kr = krem - cum; break; }
                cum += c;
            }
            sh_sel = sel; sh_krem = kr;
        }
        __syncthreads();
        prefix |= (sh_sel << shift);
        krem   = sh_krem;
    }
    unsigned int thresh = prefix;

    // Phase B: gather values with key ≥ threshold into shared memory.
    __shared__ unsigned int gcnt;
    __shared__ float gval[64];
    __shared__ unsigned int gidx[64];
    if (tid == 0) { gcnt = 0u; }
    __syncthreads();
    for (int i = lo + tid; i < hi; i += bs) {
        float v = cand[i];
        if (f2ui(v) >= thresh) {
            unsigned int slot = atomicAdd(&gcnt, 1u);
            if (slot < (unsigned int)k) {
                gval[slot] = v;
                gidx[slot] = __float_as_uint(cand[n_cand + i]);
            }
        }
    }
    __syncthreads();

    // Phase C: sort, softmax, nucleus cutoff, CDF sample (single lane).
    if (tid == 0) {
        unsigned int m = gcnt;
        if (m > (unsigned int)k) m = (unsigned int)k;

        // Insertion sort descending.
        for (unsigned int a = 1u; a < m; a++) {
            float vv = gval[a];
            unsigned int ii = gidx[a];
            unsigned int b  = a;
            while (b > 0u && gval[b - 1u] < vv) {
                gval[b] = gval[b - 1u];
                gidx[b] = gidx[b - 1u];
                b--;
            }
            gval[b] = vv;
            gidx[b] = ii;
        }

        // Softmax: exp((val - max) / temp), normalize.
        float maxl = gval[0];
        float sum  = 0.0f;
        for (unsigned int j = 0u; j < m; j++) {
            float p = expf((gval[j] - maxl) / temp);
            gval[j] = p;
            sum    += p;
        }
        for (unsigned int j = 0u; j < m; j++) { gval[j] /= sum; }

        // Nucleus (top_p) cutoff.
        float cum           = 0.0f;
        unsigned int cutoff = m;
        for (unsigned int j = 0u; j < m; j++) {
            cum += gval[j];
            if (cum >= top_p) { cutoff = j + 1u; break; }
        }

        // Renormalize and inverse-CDF with u.
        float total = 0.0f;
        for (unsigned int j = 0u; j < cutoff; j++) { total += gval[j]; }
        float u       = u_buf[0];
        float r       = u * total;
        unsigned int tok = gidx[cutoff - 1u];
        float acc     = 0.0f;
        for (unsigned int j = 0u; j < cutoff; j++) {
            acc += gval[j];
            if (r <= acc) { tok = gidx[j]; break; }
        }
        out_id[0] = __uint_as_float(tok);
    }
}
"#;

// ── ArgmaxProb (single-row argmax + softmax top-1 probability) ────────────────
//
// Fused argmax + softmax in two stages (multi-block partial → one-block combine),
// identical in structure to the existing `Op::Argmax` multi-block reduction but
// additionally carries the online-softmax sum_exp through the reduction tree so
// the final softmax top-1 probability is available without a second pass.
//
// Online softmax (Rabe '18 / Milakov-Gimelshein '18): `sum_exp` is rescaled
// when the running max is updated.  The merge of two (max, idx, sum) triples is
// associative, so a tree reduce is sound.
const ARGMAX_PROB: &str = r#"
#define ARGMAX_CHUNK 2048  // floats per block — 8 per thread at bs=256

// Pass 1 — one block per chunk reduces its chunk to a (max, idx, sum_exp) triple
// via online softmax (per-thread scan → warp reduce → LDS tree reduce).
extern "C" __global__ void argmax_prob_partial(
    const float* __restrict__ logits,  // [n]
    float* __restrict__ part,          // [3 * n_chunks]: (max, idx_bits, sum_exp) triples
    int n,
    int n_chunks
) {
    int chunk = blockIdx.x;
    if (chunk >= n_chunks) return;
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    int j0 = chunk * ARGMAX_CHUNK;
    int j1 = j0 + ARGMAX_CHUNK;
    if (j1 > n) j1 = n;

    // Per-thread online softmax over this chunk's elements.
    float best_val = -1e30f;
    int   best_idx = 0;
    float sum_exp  = 0.0f;

    for (int i = j0 + tid; i < j1; i += bs) {
        float v = logits[i];
        if (v > best_val) {
            sum_exp  *= expf(best_val - v);
            best_val  = v;
            best_idx  = i;
        } else if (v == best_val && i < best_idx) {
            best_idx  = i;   // tie-break: lower index
        }
        sum_exp += expf(v - best_val);
    }

    // Warp reduce (no barrier, 5 shfl steps).
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(best_val, off);
        int   oi = __shfl_xor(best_idx, off);
        float os = __shfl_xor(sum_exp,  off);
        if (ov > best_val) {
            sum_exp  = os + sum_exp * expf(best_val - ov);
            best_val = ov;
            best_idx = oi;
        } else if (ov == best_val) {
            sum_exp += os;
            if (oi < best_idx) best_idx = oi;
        } else {
            sum_exp  = sum_exp + os * expf(ov - best_val);
        }
    }

    // Cross-warp tree reduce through LDS.
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    __shared__ float ssum[256];
    if (tid % 32 == 0) {
        sval[tid >> 5] = best_val;
        sidx[tid >> 5] = best_idx;
        ssum[tid >> 5] = sum_exp;
    }
    __syncthreads();
    int nw = bs >> 5;
    for (int s = nw >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int   oi = sidx[tid + s];
            float os = ssum[tid + s];
            if (ov > sval[tid]) {
                ssum[tid]  = os + ssum[tid] * expf(sval[tid] - ov);
                sval[tid]  = ov;
                sidx[tid]  = oi;
            } else if (ov == sval[tid]) {
                ssum[tid] += os;
                if (oi < sidx[tid]) sidx[tid] = oi;
            } else {
                ssum[tid]  = ssum[tid] + os * expf(ov - sval[tid]);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        part[chunk * 3]     = sval[0];
        part[chunk * 3 + 1] = __int_as_float(sidx[0]);
        part[chunk * 3 + 2] = ssum[0];
    }
}

// Pass 2 — one block merges all per-chunk partials and writes the final
// (token_id, probability) pair.
extern "C" __global__ void argmax_prob_combine(
    const float* __restrict__ part,     // [3 * n_chunks]
    float* __restrict__ out_id,         // [1] u32 token id as float
    float* __restrict__ out_prob,       // [1] probability
    int n_chunks
) {
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    // Per-thread merge of the partial triples.
    float best_val = -1e30f;
    int   best_idx = 0;
    float sum_exp  = 0.0f;

    for (int i = tid; i < n_chunks; i += bs) {
        float v   = part[i * 3];
        int   idx = __float_as_int(part[i * 3 + 1]);
        float s   = part[i * 3 + 2];

        if (v > best_val) {
            sum_exp  = s + sum_exp * expf(best_val - v);
            best_val = v;
            best_idx = idx;
        } else if (v == best_val) {
            sum_exp += s;
            if (idx < best_idx) best_idx = idx;
        } else {
            sum_exp  = sum_exp + s * expf(v - best_val);
        }
    }

    // Warp reduce.
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor(best_val, off);
        int   oi = __shfl_xor(best_idx, off);
        float os = __shfl_xor(sum_exp,  off);
        if (ov > best_val) {
            sum_exp  = os + sum_exp * expf(best_val - ov);
            best_val = ov;
            best_idx = oi;
        } else if (ov == best_val) {
            sum_exp += os;
            if (oi < best_idx) best_idx = oi;
        } else {
            sum_exp  = sum_exp + os * expf(ov - best_val);
        }
    }

    // Cross-warp tree reduce.
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    __shared__ float ssum[256];
    if (tid % 32 == 0) {
        sval[tid >> 5] = best_val;
        sidx[tid >> 5] = best_idx;
        ssum[tid >> 5] = sum_exp;
    }
    __syncthreads();
    int nw = bs >> 5;
    for (int s = nw >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int   oi = sidx[tid + s];
            float os = ssum[tid + s];
            if (ov > sval[tid]) {
                ssum[tid]  = os + ssum[tid] * expf(sval[tid] - ov);
                sval[tid]  = ov;
                sidx[tid]  = oi;
            } else if (ov == sval[tid]) {
                ssum[tid] += os;
                if (oi < sidx[tid]) sidx[tid] = oi;
            } else {
                ssum[tid]  = ssum[tid] + os * expf(ov - sval[tid]);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_id[0]   = __int_as_float(sidx[0]);
        out_prob[0] = 1.0f / ssum[0];  // softmax prob of the argmax token
    }
}
"#;

const WRITE_KV: &str = r#"
extern "C" __global__ void write_kv(
    const float* __restrict__ src,  // [rows, n_kv, head_dim] at row-stride
    __half* __restrict__ cache,     // [kv_len_max, n_kv, head_dim]
    int row_offset,                 // pos in cache to write to
    int rows,
    int cache_stride,               // per-row elements in cache (= n_kv * head_dim)
    int src_stride                  // per-row stride in src (0 = packed = cache_stride)
) {
    // One thread per OUTPUT element (row, i). Each cache slot is an independent float→half cast (no
    // reduction) → bit-identical to the old per-row serial loop, but at decode (rows==1) this fans
    // the `cache_stride` (n_kv*head_dim) casts across `ceil(cache_stride/block)` blocks instead of
    // stranding all of them on ONE thread of one CU — the measured #1 decode cost (Slice 29).
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * cache_stride;
    if (tid >= total) return;
    int row = (int)(tid / cache_stride);
    int i = (int)(tid % cache_stride);
    int effective_src_stride = (src_stride > 0) ? src_stride : cache_stride;
    int cache_row = row_offset + row;
    cache[(long)cache_row * cache_stride + i] =
        __float2half(src[(long)row * effective_src_stride + i]);
}
"#;

const ATTENTION: &str = r#"
// Max head_dim/32 dims a single lane owns (runner gates decode to head_dim <= 512 → 16).
#define ATTN_MAX_PER_LANE 16

// Butterfly all-reduce of an f32 across a 32-lane wave: every lane ends with the full sum.
static __device__ __forceinline__ float attn_wave_allreduce32(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

// One WAVE (32 lanes) per (row, head), vs the old one THREAD per head. For decode (rows==1) the
// old grid launched rows*n_head=n_head threads → a handful of threads on ONE CU ran the whole
// two-pass softmax serially (the measured #1 decode cost). Here each lane owns the strided head-dim
// slice d = tid, tid+32, … : the q·k dot is a coalesced partial + a butterfly wave all-reduce, and
// the weighted-V output vector is partitioned across lanes (each lane owns its output dims, no
// cross-lane reduction). The softmax `max` and denominator `sum` are still accumulated sequentially
// over kv on every lane (identical order → bit-exact); ONLY the per-key q·k dot reduction order
// changes (butterfly vs sequential), a sub-ulp perturbation that greedy decode is robust to.
extern "C" __global__ void attention(
    const float* __restrict__ q,       // [rows, n_head, head_dim]
    const __half* __restrict__ k_cache,// [kv_len, n_kv, head_dim]
    const __half* __restrict__ v_cache,// [kv_len, n_kv, head_dim]
    float* __restrict__ dst,           // [rows, n_head, head_dim]
    int rows,
    int kv_len,
    int n_head,
    int n_kv,
    int head_dim,
    float scale,
    int pos,            // absolute position of first query row
    int mask_type,      // 0=Causal, 1=SlidingWindow, 2=Canvas
    int swa_window      // window size for SlidingWindow
) {
    int head = blockIdx.x;             // one block == one wave == one (row, head)
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int tid = threadIdx.x;             // lane 0..31
    int r = head / n_head;
    int h = head % n_head;
    int kv_h = h * n_kv / n_head;      // GQA head mapping
    int q_off = head * head_dim;
    int npl = (head_dim + 31) >> 5;    // dims this lane owns (strided by 32)

    // Preload this lane's owned q dims.
    float qreg[ATTN_MAX_PER_LANE];
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        qreg[c] = (d < head_dim) ? q[q_off + d] : 0.0f;
    }

    // Pass 1: max over unmasked keys (sequential over j on every lane → identical result).
    float max_score = -1e30f;
    for (int j = 0; j < kv_len; j++) {
        const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
        float part = 0.0f;
        for (int c = 0; c < npl; c++) {
            int d = (c << 5) + tid;
            if (d < head_dim) part += qreg[c] * __half2float(kr[d]);
        }
        float s = attn_wave_allreduce32(part) * scale;
        bool masked = false;
        if (mask_type == 0) {
            masked = (j > pos + r);
        } else if (mask_type == 1) {
            int q_pos = pos + r;
            masked = (j > q_pos || j < q_pos - swa_window + 1);
        } else if (mask_type == 2) {
            masked = (j < swa_window);
        }
        if (!masked && s > max_score) max_score = s;
    }

    // Pass 2: exp sum + weighted value sum. Each lane owns disjoint output dims → no reduction.
    float sum = 0.0f;
    float acc[ATTN_MAX_PER_LANE];
    for (int c = 0; c < npl; c++) acc[c] = 0.0f;
    for (int j = 0; j < kv_len; j++) {
        const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
        float part = 0.0f;
        for (int c = 0; c < npl; c++) {
            int d = (c << 5) + tid;
            if (d < head_dim) part += qreg[c] * __half2float(kr[d]);
        }
        float s = attn_wave_allreduce32(part) * scale;
        bool masked = false;
        if (mask_type == 0) {
            masked = (j > pos + r);
        } else if (mask_type == 1) {
            int q_pos = pos + r;
            masked = (j > q_pos || j < q_pos - swa_window + 1);
        } else if (mask_type == 2) {
            masked = (j < swa_window);
        }
        if (masked) continue;
        float w = expf(s - max_score);
        sum += w;
        const __half* vr = v_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
        for (int c = 0; c < npl; c++) {
            int d = (c << 5) + tid;
            if (d < head_dim) acc[c] += w * __half2float(vr[d]);
        }
    }
    float inv = 1.0f / sum;
    float* dr = dst + q_off;
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        if (d < head_dim) dr[d] = acc[c] * inv;
    }
}
"#;

// ── P6: batched-prefetch decode attention ────────────────────────────────────
//
// WHY. At DECODE the plain `attention` above runs ONE wave per (row, head) — 16 waves for a 16-head
// model, on a 96-SIMD GPU. Occupancy is not the problem and cannot be: there is no second wave to
// schedule. The problem is that each wave has exactly ONE memory request in flight at a time: the
// `j` loop loads key `j`'s K row (npl × 64 B), then immediately blocks on a 5-step `__shfl_xor`
// butterfly before it may compute the address of key `j+1`. Every iteration is therefore a full
// memory round trip that nothing hides. Measured on Qwen3-0.6B tg128: attention is 2.57 ms of a
// 5.09 ms token (50%, profiler-free — priced by skipping the op), reading its K/V at ~3 GB/s
// against ~960 GB/s of peak.
//
// THE FIX is memory-level parallelism, not occupancy: ISSUE PF keys' rows before consuming any of
// them, so a wave has PF requests outstanding instead of one. Everything else is held fixed.
//
// BIT-IDENTICAL BY CONSTRUCTION, and that constraint is what shapes the rest of the design. Only
// the LOADS move: each key's `part` is still accumulated over the same `c` order into the same lane
// partition and reduced by the same butterfly, so every score has the same bits; `max_score` is a
// max (exact, and still visited in `j` order); `sum` and `acc[c]` are still accumulated strictly in
// ascending `j`. `attn_pf_decode_is_bit_identical_to_the_plain_kernels` pins that as EQUALITY
// against the `kernels.rocm.attn_pf = false` control.
//
// THE STAGING BUFFER MUST BE REGISTERS, and that is why `MAXPL` is a template parameter: a register
// array has to be indexed by a COMPILE-TIME subscript or LLVM sinks it to scratch (measured: 592
// bytes/lane, i.e. the prefetch buffer lands in the very memory it exists to hide). The obvious way
// to dodge that — stage through LDS, where a runtime subscript is free — was built and measured, and
// the LDS round trip gives back most of the win: tg128 195.7 base / 212.8 registers / 193.1 LDS, and
// at d4096 84.8 / 106.3 / 92.9. Registers it is. `MAXPL` always EQUALS the live `npl`, because the
// executor selects the instantiation by `ceil(head_dim/32)`, so the unrolled compute loop covers
// exactly the dims the generic kernel's rolled one did.
//
// THE ONE THING THE OBVIOUS VERSION GETS WRONG, found by the equality test and worth stating because
// it is invisible from the source:
//
//     **Pass 2 must RE-DERIVE each score, not reuse pass 1's.** Caching pass 1's scores would drop
//     the whole second K read and the second butterfly — worth ~4% of tg128 — and it is the first
//     thing this kernel tried. It is wrong: the reference kernel computes each score twice and the
//     two copies are NOT bit-equal (LLVM contracts the two passes' dots differently), while
//     `max_score` comes from the pass-1 copy and `expf(s - max_score)` from the pass-2 copy.
//     Reusing one score for both is arithmetically MORE self-consistent and therefore differs from
//     the reference. Kept as the double computation; the 4% is the price of exactness.
const ATTENTION_PF: &str = r#"
// Keys staged per batch = the memory requests a wave keeps in flight. 8 at MAXPL<=4, 4 at MAXPL=8;
// the product PF*MAXPL is the VGPR cost of a staging buffer. Registers are the cheap resource here
// — one wave per SIMD leaves the file unspent, and the resource report still allows 16 waves/SIMD.
// Swept on Qwen3-0.6B tg128: PF=16 measured 213.3 against PF=8's 215.5 (and d4096 107.3 vs 107.6),
// so the curve is flat past 8 and this sits at the small end of the plateau.
template <int MAXPL, int PF>
static __device__ __forceinline__ void attn_pf_body(
    const float* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    float* __restrict__ dst,
    int rows, int kv_len, int n_head, int n_kv, int head_dim,
    float scale, int pos, int mask_type, int swa_window
) {
    int head = blockIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int tid = threadIdx.x;
    int r = head / n_head;
    int h = head % n_head;
    int kv_h = h * n_kv / n_head;
    int q_off = head * head_dim;
    int npl = (head_dim + 31) >> 5;

    float qreg[MAXPL];
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        qreg[c] = (d < head_dim) ? q[q_off + d] : 0.0f;
    }

    // Pass 1: max over unmasked keys. PF keys' K rows are ISSUED before any is consumed — that is
    // the whole change, and it is why the wave stops being one-request-in-flight.
    float max_score = -1e30f;
    for (int jb = 0; jb < kv_len; jb += PF) {
        float t[PF][MAXPL];
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                t[u][c] = (j < kv_len && c < npl && d < head_dim) ? __half2float(kr[d]) : 0.0f;
            }
        }
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            float part = 0.0f;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                if (d < head_dim) part += qreg[c] * t[u][c];
            }
            float s = attn_wave_allreduce32(part) * scale;
            bool masked = false;
            if (mask_type == 0) {
                masked = (j > pos + r);
            } else if (mask_type == 1) {
                int q_pos = pos + r;
                masked = (j > q_pos || j < q_pos - swa_window + 1);
            } else if (mask_type == 2) {
                masked = (j < swa_window);
            }
            // `j >= kv_len` is the tail guard: those slots staged zeros and must not vote.
            if (j < kv_len && !masked && s > max_score) max_score = s;
        }
    }

    // Pass 2: exp sum + weighted value sum. K IS re-read and its score RE-DERIVED rather than
    // carried over from pass 1 — see this kernel's header on why that is load-bearing.
    float sum = 0.0f;
    float acc[MAXPL];
    for (int c = 0; c < npl; c++) acc[c] = 0.0f;
    for (int jb = 0; jb < kv_len; jb += PF) {
        float t[PF][MAXPL];
        float tk[PF][MAXPL];
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            const __half* vr = v_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
            const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                t[u][c] = (j < kv_len && c < npl && d < head_dim) ? __half2float(vr[d]) : 0.0f;
                tk[u][c] = (j < kv_len && c < npl && d < head_dim) ? __half2float(kr[d]) : 0.0f;
            }
        }
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            if (j >= kv_len) break;
            float part2 = 0.0f;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                if (d < head_dim) part2 += qreg[c] * tk[u][c];
            }
            float s = attn_wave_allreduce32(part2) * scale;
            bool masked = false;
            if (mask_type == 0) {
                masked = (j > pos + r);
            } else if (mask_type == 1) {
                int q_pos = pos + r;
                masked = (j > q_pos || j < q_pos - swa_window + 1);
            } else if (mask_type == 2) {
                masked = (j < swa_window);
            }
            if (masked) continue;
            float w = expf(s - max_score);
            sum += w;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                if (d < head_dim) acc[c] += w * t[u][c];
            }
        }
    }
    float inv = 1.0f / sum;
    float* dr = dst + q_off;
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        if (d < head_dim) dr[d] = acc[c] * inv;
    }
}

#define ATTN_PF_INST(NAME, MAXPL, PF)                                                   \
extern "C" __global__ void NAME(                                                      \
    const float* __restrict__ q, const __half* __restrict__ k_cache,                  \
    const __half* __restrict__ v_cache, float* __restrict__ dst,                      \
    int rows, int kv_len, int n_head, int n_kv, int head_dim,                         \
    float scale, int pos, int mask_type, int swa_window)                              \
{                                                                                     \
    attn_pf_body<MAXPL, PF>(q, k_cache, v_cache, dst, rows, kv_len, n_head, n_kv,       \
                          head_dim, scale, pos, mask_type, swa_window);               \
}

ATTN_PF_INST(attention_pf_npl2, 2, 8)
ATTN_PF_INST(attention_pf_npl4, 4, 8)
ATTN_PF_INST(attention_pf_npl8, 8, 4)
"#;

// ── Tiled flash PREFILL attention (P1) ───────────────────────────────────────
//
// WHY. The P1 per-op profile (`kernels.rocm.prof_ops`) says prefill attention is **64% of a
// Qwen3-0.6B pp512 forward** — 2.77 ms of a 116 ms token batch, per layer, 28 layers. The plain
// `attention` kernel above gives each (query row, head) ONE wave that walks the WHOLE kv range
// TWICE (a max pass, then an exp/accumulate pass) straight out of global memory, and it evaluates
// the q·k dot for masked keys before throwing the score away. At pp512 that is
// `rows × n_head = 8192` waves each streaming ~320 KB of K and V — ~2.6 GB per layer for 2 MB of
// distinct cache, i.e. the SAME K/V re-read ~1000×. It was never a math problem: the arithmetic is
// ~1 GFLOP per layer, ~36 µs of this GPU.
//
// WHAT. The standard flash tiling, sized for RDNA3's 32-lane wave:
//
//   * A workgroup owns a TILE of `br = nw · ATTN_FLASH_QPW` consecutive query rows of ONE head, so
//     each K/V element it reads is reused by all `br` of them. That is the whole win — global K/V
//     traffic drops by `br`.
//   * ONE pass, online softmax: a running (max, denom, accumulator) per query row, rescaled by
//     `exp(m_old − m_new)` when a tile raises the max. Halves the K reads the two-pass kernel did
//     and drops the second dot product entirely. This is the same rescale Vulkan's flash kernels
//     use, and the ROCm qwen3 seam golden is Vulkan's — see the parity notes on the exec routing.
//   * WHOLE-TILE mask elision. The key range `[j_lo, j_hi)` is clamped once per workgroup from the
//     tile's own position span, so a causal prefill never launches a single wave at the ~half of
//     the score matrix that is masked, and a SWA model skips everything below its window. Inside a
//     tile, a masked lane also skips its dot instead of computing-then-discarding.
//   * LANE PER KEY for the scores (lane `t` owns key `j0+t` and walks the whole `head_dim`), so a
//     score needs NO cross-lane reduction at all — the plain kernel paid a 5-`shfl` butterfly per
//     key. Lane per DIM for the P·V accumulate, with the weight broadcast by one `__shfl`, so that
//     half needs no reduction either.
//   * K and V tiles live in LDS at a padded row stride of `head_dim + 2` halves. `head_dim` is a
//     multiple of 32 on every routed model (`attn_flash_tiling` requires it), so the stride is an
//     ODD number of 4-byte LDS words and the 32 lanes' strided reads land on 32 distinct banks.
//   * The score dot is rebuilt in the plain kernel's own reduction tree. That is a CORRECTNESS
//     requirement, not a detail — see the comment on the loop, and §2 of docs/rocm-plan.md.
//
// `ATTN_FLASH_QPW` (query rows per WAVE) is compile-time and the `u` loops are unrolled, because
// `acc[u][c]` must stay in registers — a runtime bound would spill the accumulator to scratch and
// undo the point. `nw` (waves per workgroup) and `bc` (keys per tile) are chosen HOST-side from
// `head_dim` to keep the LDS footprint inside 32 KiB so two workgroups stay co-resident per CU;
// `exec.rs`'s `attn_flash_tiling` is that policy and falls back to the plain kernel when no
// configuration fits.
const ATTENTION_FLASH: &str = r#"
// Query rows one WAVE owns. Compile-time: `acc[QPW][MAXP2]` is register state, so this is a VGPR
// dial as much as a reuse dial. Measured at pp512 (head_dim 128, br held at 16): 2 -> 278 us,
// 4 -> 313, 8 -> 441. More rows per wave reuses the lane's K element more, but the accumulator
// grows with it and occupancy falls faster than the reuse pays.
#define ATTN_FLASH_QPW 2
// head_dim/64 ceiling. A lane owns TWO CONSECUTIVE output dims per step, so it reads V as one
// `__half2` and stores `dst` as one `float2` — half the LDS reads of a dim-per-lane mapping, and
// `acc` costs the same registers. 4 covers head_dim <= 256, which is every head dim routed here;
// `acc[QPW][MAXP2]` is register state allocated in full, so a larger bound is not free.
#define ATTN_FLASH_MAXP2 4

// Butterfly all-reduce MAX across a 32-lane wave (the `attn_wave_allreduce32` sum's twin).
static __device__ __forceinline__ float attn_wave_allmax32(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

extern "C" __global__ void attention_prefill_flash(
    const float* __restrict__ q,        // [rows, n_head, head_dim]
    const __half* __restrict__ k_cache, // [kv_len, n_kv, head_dim]
    const __half* __restrict__ v_cache, // [kv_len, n_kv, head_dim]
    float* __restrict__ dst,            // [rows, n_head, head_dim]
    int rows,
    int kv_len,
    int n_head,
    int n_kv,
    int head_dim,
    float scale,
    int pos,            // absolute position of query row 0
    int mask_type,      // 0=Causal, 1=SlidingWindow, 2=Canvas
    int swa_window,
    int bc,             // keys per KV tile, <= 32 (one lane per key)
    int n_qtiles        // ceil(rows / br); blockIdx.x = head * n_qtiles + qtile
) {
    // 16-byte aligned: the hot loop reads the query tile as `float4`.
    extern __shared__ __align__(16) float smem[];
    int nt   = blockDim.x;
    int nw   = nt >> 5;
    int br   = nw * ATTN_FLASH_QPW;
    int kvs  = head_dim + 2;                  // padded LDS row stride, in halves
    float*  qs = smem;                        // [br][head_dim] f32
    __half* ks = (__half*)(smem + br * head_dim);
    __half* vs = ks + bc * kvs;

    int qt   = blockIdx.x % n_qtiles;
    int h    = blockIdx.x / n_qtiles;
    if (h >= n_head) return;
    int kv_h = h * n_kv / n_head;             // GQA head mapping (same as the plain kernel)
    int r0   = qt * br;
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wave = tid >> 5;

    // Stage this workgroup's query rows. Rows past `rows` are zeroed and never stored back.
    for (int i = tid; i < br * head_dim; i += nt) {
        int rr = i / head_dim;
        int d  = i - rr * head_dim;
        int gr = r0 + rr;
        qs[i] = (gr < rows) ? q[((long)gr * n_head + h) * head_dim + d] : 0.0f;
    }

    float m[ATTN_FLASH_QPW], l[ATTN_FLASH_QPW];
    float2 acc[ATTN_FLASH_QPW][ATTN_FLASH_MAXP2];
    #pragma unroll
    for (int u = 0; u < ATTN_FLASH_QPW; u++) {
        m[u] = -1e30f;
        l[u] = 0.0f;
        #pragma unroll
        for (int c = 0; c < ATTN_FLASH_MAXP2; c++) acc[u][c] = make_float2(0.0f, 0.0f);
    }

    // WORKGROUP-uniform key range: every thread must reach the tile-staging barriers the same
    // number of times, so the bound comes from the whole query tile, not from one row.
    int qp_lo = pos + r0;
    int qp_hi = pos + r0 + br - 1;
    int j_lo = 0, j_hi = kv_len;
    if (mask_type == 0) {
        j_hi = min(kv_len, qp_hi + 1);
    } else if (mask_type == 1) {
        j_hi = min(kv_len, qp_hi + 1);
        j_lo = max(0, qp_lo - swa_window + 1);
    } else if (mask_type == 2) {
        j_lo = min(max(0, swa_window), kv_len);
    }
    j_lo = (j_lo / bc) * bc;

    for (int j0 = j_lo; j0 < j_hi; j0 += bc) {
        // Both barriers are needed: the first retires the PREVIOUS iteration's readers before the
        // tile buffers are overwritten, the second publishes the new tile. (The first also covers
        // the `qs` staging above on the opening iteration.)
        __syncthreads();
        // Stage K and V EIGHT halves (16 B) per thread per step. Element-at-a-time staging was
        // measured to be the kernel's single biggest cost (~200 of 492 us/layer at pp512): 64
        // `global_load_ushort` per thread per tile, each wave pulling 64 B — half a cache line —
        // and far too few in flight to cover the ~500-cycle latency. A `uint4` makes it 8 loads of
        // a wave-contiguous 512 B. Both alignments hold: `head_dim % 32 == 0` (checked by
        // `attn_flash_tiling`) makes every cache row 16 B-aligned off a `hipMalloc` base, and the
        // padded LDS stride stays a whole number of 4-byte words for the 32-bit stores.
        int nv8 = bc * (head_dim >> 3);
        for (int i = tid; i < nv8; i += nt) {
            int jj = i / (head_dim >> 3);
            int d  = (i - jj * (head_dim >> 3)) << 3;
            int j  = j0 + jj;
            uint4 kk = make_uint4(0u, 0u, 0u, 0u), vv = kk;
            if (j < kv_len) {
                long off = ((long)j * n_kv + kv_h) * head_dim + d;
                kk = *(const uint4*)(k_cache + off);
                vv = *(const uint4*)(v_cache + off);
            }
            uint* kd = (uint*)(ks + jj * kvs + d);
            uint* vd = (uint*)(vs + jj * kvs + d);
            kd[0] = kk.x; kd[1] = kk.y; kd[2] = kk.z; kd[3] = kk.w;
            vd[0] = vv.x; vd[1] = vv.y; vd[2] = vv.z; vd[3] = vv.w;
        }
        __syncthreads();

        int j = j0 + lane;
        bool live = (lane < bc) && (j < j_hi) && (j < kv_len);

        // Q·Kᵀ, one query row at a time, in the EXACT reduction order of the `attention` kernel
        // this replaces — that identity is a correctness requirement, not an aesthetic one. The
        // plain kernel splits a head dim across the 32 lanes (lane t owning d = t, t+32, …),
        // chains an FMA per lane, then butterfly all-reduces. Here one lane owns the whole dot, so
        // the same tree is rebuilt IN REGISTERS: `g[t]` is lane t's partial, and the reduction
        // below pairs t with t+16, then t+8, … which is exactly what `__shfl_xor` at off=16,8,…
        // computes. Get this wrong and the difference is invisible in a tolerance test and shows
        // up as a flipped near-tie argmax fourteen tokens into a greedy run — which is how it
        // was found (the Q8_0 seam gate split "I know" / "I remember" on a plain serial dot).
        //
        // The access pattern survives the reordering: `t` is the INNER loop, so `q` is still read
        // as a `float4` and `k` as two `__half2` over consecutive `d`, and the 32 `g[t]` chains
        // are independent — MORE instruction-level parallelism than the per-row `s[u]` chains it
        // replaced, which is why matching the reference order costs ~4% and not more.
        float s[ATTN_FLASH_QPW];
        #pragma unroll
        for (int u = 0; u < ATTN_FLASH_QPW; u++) s[u] = 0.0f;
        if (live) {
            const __half* kr = ks + lane * kvs;
            const float*  q0 = qs + wave * ATTN_FLASH_QPW * head_dim;
            #pragma unroll
            for (int u = 0; u < ATTN_FLASH_QPW; u++) {
                const float* qr = q0 + u * head_dim;
                float g[32];
                #pragma unroll
                for (int t = 0; t < 32; t++) g[t] = 0.0f;
                // `head_dim` is a multiple of 32 (`attn_flash_tiling`), so every lane-group block
                // is whole and no `g[t]` is left a term short.
                for (int b = 0; b < head_dim; b += 32) {
                    #pragma unroll
                    for (int t = 0; t < 32; t += 4) {
                        int d = b + t;
                        float4 qv = *(const float4*)(qr + d);
                        float2 k01 = __half22float2(*(const __half2*)(kr + d));
                        float2 k23 = __half22float2(*(const __half2*)(kr + d + 2));
                        // `+=` and not `fmaf`, deliberately: the plain kernel writes
                        // `part += qreg[c] * __half2float(kr[d])`, and whatever the compiler
                        // decides to contract that into it must decide the same way here, or the
                        // two differ by a ULP on exactly the products it fuses in one and not the
                        // other.
                        g[t]     += qv.x * k01.x;
                        g[t + 1] += qv.y * k01.y;
                        g[t + 2] += qv.z * k23.x;
                        g[t + 3] += qv.w * k23.y;
                    }
                }
                // `attn_wave_allreduce32`'s tree, unrolled in registers (for t < off, t^off == t+off).
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    #pragma unroll
                    for (int t = 0; t < off; t++) g[t] += g[t + off];
                }
                s[u] = g[0];
            }
        }

        float w[ATTN_FLASH_QPW];
        #pragma unroll
        for (int u = 0; u < ATTN_FLASH_QPW; u++) {
            int lr = wave * ATTN_FLASH_QPW + u;
            int qp = pos + r0 + lr;
            bool masked = !live;
            if (!masked) {
                if (mask_type == 0)      masked = (j > qp);
                else if (mask_type == 1) masked = (j > qp || j < qp - swa_window + 1);
                else if (mask_type == 2) masked = (j < swa_window);
            }
            float sm = masked ? -1e30f : s[u] * scale;
            // Online rescale: fold this tile's max into the running one, correcting what is
            // already accumulated. `m` starts at -1e30, so the first live tile scales the (zero)
            // accumulator by exp(-1e30 - m_new) = 0 and an all-masked tile by exp(0) = 1.
            float nm = fmaxf(m[u], attn_wave_allmax32(sm));
            float corr = expf(m[u] - nm);
            m[u] = nm;
            float wu = masked ? 0.0f : expf(sm - nm);
            // Only the RESCALE happens here. The plain kernel sums the softmax denominator over
            // `j` ascending, so this one does too — the add lives in the P·V walk below, which
            // already visits the tile's keys in that order. A butterfly sum here would have been
            // cheaper and a different number.
            l[u] *= corr;
            w[u] = wu;
            if (corr != 1.0f) {
                #pragma unroll
                for (int c = 0; c < ATTN_FLASH_MAXP2; c++) {
                    acc[u][c].x *= corr;
                    acc[u][c].y *= corr;
                }
            }
        }

        // P·V. One V row is read once per lane and fanned across all QPW query rows, so the LDS
        // traffic here is amortized the same way the K reads are.
        int nj = min(bc, j_hi - j0);
        for (int jj = 0; jj < nj; jj++) {
            float wj[ATTN_FLASH_QPW];
            bool any = false;
            #pragma unroll
            for (int u = 0; u < ATTN_FLASH_QPW; u++) {
                wj[u] = __shfl(w[u], jj);
                any = any || (wj[u] != 0.0f);
            }
            #pragma unroll
            for (int u = 0; u < ATTN_FLASH_QPW; u++) l[u] += wj[u];
            if (!any) continue;   // wave-uniform: a fully-masked key contributes exactly nothing
            const __half* vr = vs + jj * kvs;
            // NO `break` on the head-dim bound in any of these unrolled `c` loops: an early exit
            // stops LLVM fully unrolling, `c` stays dynamic, and `acc[u][c]` lands in SCRATCH —
            // measured at 272 bytes/lane, and it cost ~2x on this kernel. A plain `d < head_dim`
            // value guard keeps every index a compile-time constant.
            #pragma unroll
            for (int c = 0; c < ATTN_FLASH_MAXP2; c++) {
                int d = (c << 6) + (lane << 1);
                if (d < head_dim) {
                    float2 vv = __half22float2(*(const __half2*)(vr + d));
                    // Same reason as the score dot: `acc[c] += w * __half2float(vr[d])` is how
                    // the plain kernel spells it.
                    #pragma unroll
                    for (int u = 0; u < ATTN_FLASH_QPW; u++) {
                        acc[u][c].x += wj[u] * vv.x;
                        acc[u][c].y += wj[u] * vv.y;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int u = 0; u < ATTN_FLASH_QPW; u++) {
        int gr = r0 + wave * ATTN_FLASH_QPW + u;
        if (gr >= rows) continue;
        float inv = 1.0f / l[u];
        float* dr = dst + ((long)gr * n_head + h) * head_dim;
        #pragma unroll
        for (int c = 0; c < ATTN_FLASH_MAXP2; c++) {
            int d = (c << 6) + (lane << 1);
            if (d < head_dim) {
                *(float2*)(dr + d) = make_float2(acc[u][c].x * inv, acc[u][c].y * inv);
            }
        }
    }
}
"#;

// ── Split-KV (flash-decoding) decode attention ───────────────────────────────
// The single-wave `attention` kernel scans ALL kv serially on ONE wave per (row, head): great at
// low depth (~n_head waves fill enough CUs when kv is short), but at DEPTH one wave crawls a long
// serial kv loop while 95 CUs sit idle — the measured decode-at-depth bottleneck. Split-KV
// PARALLELIZES the kv dimension: partition kv into `n_chunks` contiguous chunks and launch one wave
// per (row, head, chunk). Each wave computes its chunk's online-softmax partials (chunk-local max m,
// denom l, weighted-V accumulator over that chunk only), then a tiny combine kernel merges the
// per-chunk partials with the standard flash rescale. This fills the grid at depth
// (n_head × n_chunks waves instead of n_head). Cross-checked against the Vulkan
// attn_partial.comp / attn_combine.comp split-K pair. Decode-only (rows==1); exec routes rows>1
// (prefill) to the plain `attention` kernel, and short-context decode where n_chunks==1 too.
const ATTENTION_SPLIT: &str = r#"
#define ATTN_SPLIT_MAX_PER_LANE 16

static __device__ __forceinline__ float attn_split_allreduce32(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

// PASS 1 of split-KV: one WAVE (32 lanes) per (row, head, chunk). Computes this chunk's partial
// online-softmax over its kv sub-range [j0, j1): chunk-local max `m`, denom `l = Σ exp(s-m)`, and
// un-normalized weighted-V accumulator `acc = Σ exp(s-m)·v`. The q·k dot uses the SAME lane-owns-
// strided-hd-slice butterfly all-reduce as the single-wave kernel, so the per-key reduction order is
// unchanged; only the softmax is now chunk-local (the combine re-references to the global max). The
// per-chunk max/denom are identical on all 32 lanes (each runs the same sequential scan over the
// chunk); acc is partitioned across lanes (each lane owns its output dims — no cross-lane reduce).
extern "C" __global__ void attention_split_partial(
    const float* __restrict__ q,        // [rows, n_head, head_dim]
    const __half* __restrict__ k_cache, // [kv_len, n_kv, head_dim]
    const __half* __restrict__ v_cache, // [kv_len, n_kv, head_dim]
    float* __restrict__ pm,             // [rows*n_head, n_chunks] chunk-local max
    float* __restrict__ pl,             // [rows*n_head, n_chunks] chunk denom
    float* __restrict__ pacc,           // [rows*n_head, n_chunks, head_dim] weighted-V
    int rows,
    int kv_len,
    int n_head,
    int n_kv,
    int head_dim,
    float scale,
    int pos,            // absolute position of first query row
    int mask_type,      // 0=Causal, 1=SlidingWindow, 2=Canvas
    int swa_window,     // window size for SlidingWindow / lo for Canvas
    int chunk_size,     // kv keys per chunk
    int n_chunks
) {
    int gidx = blockIdx.x;              // one block == one wave == one (row, head, chunk)
    int total_heads = rows * n_head;
    int chunk = gidx % n_chunks;
    int head = gidx / n_chunks;
    if (head >= total_heads) return;
    int tid = threadIdx.x;              // lane 0..31
    int r = head / n_head;
    int h = head % n_head;
    int kv_h = h * n_kv / n_head;       // GQA head mapping
    int q_off = head * head_dim;
    int npl = (head_dim + 31) >> 5;

    float qreg[ATTN_SPLIT_MAX_PER_LANE];
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        qreg[c] = (d < head_dim) ? q[q_off + d] : 0.0f;
    }

    int j0 = chunk * chunk_size;
    int j1 = j0 + chunk_size;
    if (j1 > kv_len) j1 = kv_len;

    int pbase = gidx * head_dim;

    // Pass 1: chunk-local max over unmasked keys.
    float max_score = -1e30f;
    for (int j = j0; j < j1; j++) {
        const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
        float part = 0.0f;
        for (int c = 0; c < npl; c++) {
            int d = (c << 5) + tid;
            if (d < head_dim) part += qreg[c] * __half2float(kr[d]);
        }
        float s = attn_split_allreduce32(part) * scale;
        bool masked = false;
        if (mask_type == 0) {
            masked = (j > pos + r);
        } else if (mask_type == 1) {
            int q_pos = pos + r;
            masked = (j > q_pos || j < q_pos - swa_window + 1);
        } else if (mask_type == 2) {
            masked = (j < swa_window);
        }
        if (!masked && s > max_score) max_score = s;
    }

    // Pass 2: chunk denom + weighted-V accumulator, referenced to the chunk-local max.
    float sum = 0.0f;
    float acc[ATTN_SPLIT_MAX_PER_LANE];
    for (int c = 0; c < npl; c++) acc[c] = 0.0f;
    for (int j = j0; j < j1; j++) {
        const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
        float part = 0.0f;
        for (int c = 0; c < npl; c++) {
            int d = (c << 5) + tid;
            if (d < head_dim) part += qreg[c] * __half2float(kr[d]);
        }
        float s = attn_split_allreduce32(part) * scale;
        bool masked = false;
        if (mask_type == 0) {
            masked = (j > pos + r);
        } else if (mask_type == 1) {
            int q_pos = pos + r;
            masked = (j > q_pos || j < q_pos - swa_window + 1);
        } else if (mask_type == 2) {
            masked = (j < swa_window);
        }
        if (masked) continue;
        float w = expf(s - max_score);
        sum += w;
        const __half* vr = v_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
        for (int c = 0; c < npl; c++) {
            int d = (c << 5) + tid;
            if (d < head_dim) acc[c] += w * __half2float(vr[d]);
        }
    }

    if (tid == 0) {
        // A fully-masked/empty chunk leaves max_score = -1e30, sum = 0; the combine's exp(pm-mm)
        // weight test then skips it (pacc for such a chunk is all-zero, written below).
        pm[gidx] = max_score;
        pl[gidx] = sum;
    }
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        if (d < head_dim) pacc[pbase + d] = acc[c];
    }
}

// P6: the same batched-prefetch restructure as `attn_pf_body`, applied to the split-KV partial —
// this is the arm that runs at every real context depth, and it has the identical one-request-in-
// flight defect (a wave still walks its chunk one key at a time). Chunk-local `max`/`sum`/`acc`
// keep their exact `j` order and each score is still derived twice, so a chunk's partial is
// bit-identical to the generic partial's, and the combine that reads it is untouched.
template <int MAXPL, int PF>
static __device__ __forceinline__ void attn_split_pf_body(
    const float* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ pacc,
    int rows, int kv_len, int n_head, int n_kv, int head_dim,
    float scale, int pos, int mask_type, int swa_window,
    int chunk_size, int n_chunks
) {
    int gidx = blockIdx.x;
    int total_heads = rows * n_head;
    int chunk = gidx % n_chunks;
    int head = gidx / n_chunks;
    if (head >= total_heads) return;
    int tid = threadIdx.x;
    int r = head / n_head;
    int h = head % n_head;
    int kv_h = h * n_kv / n_head;
    int q_off = head * head_dim;
    int npl = (head_dim + 31) >> 5;

    float qreg[MAXPL];
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        qreg[c] = (d < head_dim) ? q[q_off + d] : 0.0f;
    }

    int j0 = chunk * chunk_size;
    int j1 = j0 + chunk_size;
    if (j1 > kv_len) j1 = kv_len;
    int pbase = gidx * head_dim;

    float max_score = -1e30f;
    for (int jb = j0; jb < j1; jb += PF) {
        float t[PF][MAXPL];
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                t[u][c] = (j < j1 && c < npl && d < head_dim) ? __half2float(kr[d]) : 0.0f;
            }
        }
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            float part = 0.0f;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                if (d < head_dim) part += qreg[c] * t[u][c];
            }
            float s = attn_split_allreduce32(part) * scale;
            bool masked = false;
            if (mask_type == 0) {
                masked = (j > pos + r);
            } else if (mask_type == 1) {
                int q_pos = pos + r;
                masked = (j > q_pos || j < q_pos - swa_window + 1);
            } else if (mask_type == 2) {
                masked = (j < swa_window);
            }
            if (j < j1 && !masked && s > max_score) max_score = s;
        }
    }

    float sum = 0.0f;
    float acc[MAXPL];
    for (int c = 0; c < npl; c++) acc[c] = 0.0f;
    for (int jb = j0; jb < j1; jb += PF) {
        float t[PF][MAXPL];
        float tk[PF][MAXPL];
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            const __half* vr = v_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
            const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                t[u][c] = (j < j1 && c < npl && d < head_dim) ? __half2float(vr[d]) : 0.0f;
                tk[u][c] = (j < j1 && c < npl && d < head_dim) ? __half2float(kr[d]) : 0.0f;
            }
        }
        #pragma unroll
        for (int u = 0; u < PF; u++) {
            int j = jb + u;
            if (j >= j1) break;
            float part2 = 0.0f;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                if (d < head_dim) part2 += qreg[c] * tk[u][c];
            }
            float s = attn_split_allreduce32(part2) * scale;
            bool masked = false;
            if (mask_type == 0) {
                masked = (j > pos + r);
            } else if (mask_type == 1) {
                int q_pos = pos + r;
                masked = (j > q_pos || j < q_pos - swa_window + 1);
            } else if (mask_type == 2) {
                masked = (j < swa_window);
            }
            if (masked) continue;
            float w = expf(s - max_score);
            sum += w;
            #pragma unroll
            for (int c = 0; c < MAXPL; c++) {
                int d = (c << 5) + tid;
                if (d < head_dim) acc[c] += w * t[u][c];
            }
        }
    }

    if (tid == 0) {
        pm[gidx] = max_score;
        pl[gidx] = sum;
    }
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        if (d < head_dim) pacc[pbase + d] = acc[c];
    }
}

#define ATTN_SPLIT_PF_INST(NAME, MAXPL, PF)                                              \
extern "C" __global__ void NAME(                                                       \
    const float* __restrict__ q, const __half* __restrict__ k_cache,                   \
    const __half* __restrict__ v_cache, float* __restrict__ pm,                        \
    float* __restrict__ pl, float* __restrict__ pacc,                                  \
    int rows, int kv_len, int n_head, int n_kv, int head_dim,                          \
    float scale, int pos, int mask_type, int swa_window,                               \
    int chunk_size, int n_chunks)                                                      \
{                                                                                      \
    attn_split_pf_body<MAXPL, PF>(q, k_cache, v_cache, pm, pl, pacc, rows, kv_len,       \
                                n_head, n_kv, head_dim, scale, pos, mask_type,         \
                                swa_window, chunk_size, n_chunks);                     \
}

ATTN_SPLIT_PF_INST(attention_split_partial_pf_npl2, 2, 8)
ATTN_SPLIT_PF_INST(attention_split_partial_pf_npl4, 4, 8)
ATTN_SPLIT_PF_INST(attention_split_partial_pf_npl8, 8, 4)

// COMBINE of split-KV: one WAVE (32 lanes) per (row, head). Merges the `n_chunks` partials via the
// standard online-softmax rescale — mm = max_c pm[c]; l = Σ_c pl[c]·exp(pm[c]-mm);
// out[d] = (Σ_c pacc[c,d]·exp(pm[c]-mm)) / l. The chunk sum runs in FIXED order c = 0..n_chunks so
// the float reduction is deterministic → goldens stable. Each lane owns the same strided hd slice
// the partial wrote, so no cross-lane reduction is needed.
extern "C" __global__ void attention_split_combine(
    const float* __restrict__ pm,   // [rows*n_head, n_chunks]
    const float* __restrict__ pl,   // [rows*n_head, n_chunks]
    const float* __restrict__ pacc, // [rows*n_head, n_chunks, head_dim]
    float* __restrict__ dst,        // [rows, n_head, head_dim]
    int rows,
    int n_head,
    int head_dim,
    int n_chunks
) {
    int head = blockIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int tid = threadIdx.x;
    int base = head * n_chunks;
    int npl = (head_dim + 31) >> 5;

    float mm = -1e30f;
    for (int c = 0; c < n_chunks; c++) mm = fmaxf(mm, pm[base + c]);

    float l = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        l += pl[base + c] * expf(pm[base + c] - mm);
    }
    float inv = 1.0f / l;

    int q_off = head * head_dim;
    for (int cc = 0; cc < npl; cc++) {
        int d = (cc << 5) + tid;
        if (d >= head_dim) continue;
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            float w = expf(pm[base + c] - mm);
            if (w != 0.0f) acc += pacc[(base + c) * head_dim + d] * w;
        }
        dst[q_off + d] = acc * inv;
    }
}
"#;

// ── P7: one-pass online-softmax + one-key-per-lane split-KV decode attention ──
//
// Replaces the two-pass, lane-per-dim `attention_split_partial` with:
//   1. ONE-PASS online softmax — the chunk-local max, denom, and weighted-V accumulator
//      are updated in a single KV pass (the old kernel reads K twice and V once; this reads
//      each once and accumulates on the fly).
//   2. ONE KEY PER LANE — lane t owns key j0+t and computes the full q·k dot in that lane
//      alone, removing the cross-lane allreduce entirely (5 __shfl_xor steps per key in the
//      old kernel).
//
// The output (pm, pl, pacc) has the identical shape and lane partition, so the existing
// `attention_split_combine` is untouched. Despite the changed floating-point reduction order,
// the qwen3 seam golden hash is UNMOVED (0xfd63781ea3bfa785) — greedy decode is identical.
//
// DESIGN:
//   * Q is staged into LDS cooperatively so every lane can read the full query row for its
//     one-key dot at LDS latency (banked, ~L1). The only __syncthreads() in the kernel.
//   * Each tile of bc=32 keys: lane t loads key j0+t's full K row from global (head_dim
//     contiguous halfs → ~4 cache lines gg and computes the complete dot against the
//     LDS-staged Q — no allreduce, no inter-lane communication for the score.
//   * Scores stay in registers; tile-max uses attn_wave_allmax32 (shfl-based, 5 steps,
//     but once per TILE of 32 keys instead of once per KEY). V accumulates via __shfl:
//     each lane shuffles the weight from the owning lane and reads V at its own output
//     dims (identical V read pattern to the old kernel).
//   * Online softmax: within each tile all scores are computed first, then the tile-max is
//     found, the running (m, l, acc) is rescaled by exp(m_old - m_new), and finally each
//     key's V is accumulated with the updated global max as the reference.
//
// This is structurally the flash-prefill's online-softmax loop simplified to one query row
// (no per-row `u` dimension, no `br>1` query-tile reuse of K/V in LDS).
const ATTENTION_SPLIT_FLASH: &str = r#"
extern "C" __global__ void attention_split_partial_flash(
    const float* __restrict__ q,        // [rows, n_head, head_dim]
    const __half* __restrict__ k_cache, // [kv_len, n_kv, head_dim]
    const __half* __restrict__ v_cache, // [kv_len, n_kv, head_dim]
    float* __restrict__ pm,             // [rows*n_head, n_chunks] chunk-local max
    float* __restrict__ pl,             // [rows*n_head, n_chunks] chunk denom
    float* __restrict__ pacc,           // [rows*n_head, n_chunks, head_dim] weighted-V
    int rows,
    int kv_len,
    int n_head,
    int n_kv,
    int head_dim,
    float scale,
    int pos,            // absolute position of first query row
    int mask_type,      // 0=Causal, 1=SlidingWindow, 2=Canvas
    int swa_window,     // window size for SlidingWindow / lo for Canvas
    int chunk_size,     // kv keys per chunk
    int n_chunks
) {
    int gidx = blockIdx.x;              // one block == one wave == one (row, head, chunk)
    int total_heads = rows * n_head;
    int chunk = gidx % n_chunks;
    int head = gidx / n_chunks;
    if (head >= total_heads) return;
    int tid = threadIdx.x;              // lane 0..31
    int lane = tid & 31;
    int r = head / n_head;
    int h = head % n_head;
    int kv_h = h * n_kv / n_head;       // GQA head mapping
    int q_off = head * head_dim;
    int npl = (head_dim + 31) >> 5;    // output dims this lane owns (strided by 32)

    // LDS: the full query row, so every lane can read all head_dim elements for its one-key dot.
    // Sized at dispatch time to head_dim floats; the 32 scores live in registers, not LDS.
    extern __shared__ __align__(16) float q_lds[];

    // ── Stage Q into LDS cooperatively ──
    // head_dim is a multiple of 32 for every model routed here (host gate), so every
    // lane-group block is whole and no element is left unstaged.
    for (int i = tid; i < head_dim; i += 32) {
        q_lds[i] = q[q_off + i];
    }
    __syncthreads();

    int j0 = chunk * chunk_size;
    int j1 = j0 + chunk_size;
    if (j1 > kv_len) j1 = kv_len;

    int pbase = gidx * head_dim;

    // ── Online-softmax state (chunk-local) ──
    // Starts at the identity: m = -∞ (any real score wins), l = 0, acc = 0.
    float m = -1e30f;
    float l = 0.0f;
    float acc[ATTN_SPLIT_MAX_PER_LANE];
    for (int c = 0; c < npl; c++) acc[c] = 0.0f;

    int bc = 32;  // one key per lane
    for (int j0t = j0; j0t < j1; j0t += bc) {
        int nj = bc;
        if (j0t + nj > j1) nj = j1 - j0t;

        // ── Each lane computes q·k for its assigned key ──
        // Lane `lane` owns key `j0t + lane` (when lane < nj). ONE lane computes the
        // ENTIRE dot — no cross-lane allreduce. Masked and out-of-range keys leave
        // my_s = -1e30f, a sentinel that max and expf skip.
        int j = j0t + lane;
        float my_s = -1e30f;
        if (lane < nj && j < kv_len) {
            bool masked = false;
            if (mask_type == 0) {
                masked = (j > pos + r);
            } else if (mask_type == 1) {
                int qp = pos + r;
                masked = (j > qp || j < qp - swa_window + 1);
            } else if (mask_type == 2) {
                masked = (j < swa_window);
            }
            if (!masked) {
                const __half* kr = k_cache + (long)j * n_kv * head_dim + kv_h * head_dim;
                float dot = 0.0f;
                // Full dot — one lane, all dims. K is read as contiguous halfs; Q is read
                // from LDS at banked latency. head_dim is a runtime argument here (the kernel
                // is compiled once via hiprtc, not per-head_dim), so this is a guarded loop,
                // not unrolled — tight enough that it doesn't matter.
                for (int d = 0; d < head_dim; d++) {
                    dot += q_lds[d] * __half2float(kr[d]);
                }
                my_s = dot * scale;
            }
        }

        // ── Tile-max and online rescale ──
        // attn_wave_allmax32 reduces across the wave: one max operation over 32 scores
        // (5 shfl steps), once per TILE of 32 keys instead of once per KEY. The sentinel
        // -1e30f ensures masked/out-of-range lanes do not distort the max.
        float tile_max = attn_wave_allmax32((lane < nj) ? my_s : -1e30f);
        float nm = fmaxf(m, tile_max);
        float corr = expf(m - nm);
        m = nm;
        l *= corr;
        for (int c = 0; c < npl; c++) acc[c] *= corr;

        // ── Accumulate V ──
        // Each lane owns output dims d = tid, tid+32, … (same partition as the old kernel).
        // Shuffle the weight from the key-owning lane, then read V at this lane's dims.
        for (int jj = 0; jj < nj; jj++) {
            float sj = __shfl(my_s, jj);
            if (sj <= -1e29f) continue;  // masked or out-of-range
            float w = expf(sj - m);
            l += w;

            int kj = j0t + jj;
            const __half* vr = v_cache + (long)kj * n_kv * head_dim + kv_h * head_dim;
            for (int c = 0; c < npl; c++) {
                int d = (c << 5) + tid;
                if (d < head_dim) acc[c] += w * __half2float(vr[d]);
            }
        }
    }

    // ── Write chunk partials (same format as the old kernel) ──
    if (tid == 0) {
        pm[gidx] = m;
        pl[gidx] = l;
    }
    for (int c = 0; c < npl; c++) {
        int d = (c << 5) + tid;
        if (d < head_dim) pacc[pbase + d] = acc[c];
    }
}
"#;

const MOE_FFN: &str = r#"
// Host-side router for MoE — this kernel runs ONE expert's gated FFN on x.
// The router (softmax + top-k selection) is done on the HOST in the execute() walk.
extern "C" __global__ void moe_ffn_expert(
    const float* __restrict__ x,        // [ne] — input row
    const __half* __restrict__ gate_w,  // [n_ff_exp, ne] — expert's gate weight
    const __half* __restrict__ up_w,    // [n_ff_exp, ne] — expert's up weight
    const __half* __restrict__ down_w,  // [ne, n_ff_exp] — expert's down weight
    float* __restrict__ dst,            // [ne] — accumulated * weight
    int ne,
    int n_ff_exp,
    int act_type,   // 0=SiLU, 1=GeLU, 2=Sigmoid
    float weight,   // routing weight for this expert
    float down_scale, // per-expert down-projection output scale (1 = no scale)
    int weight_before // 1 = apply `weight` to the gate/up inputs (llama4); 0 = to the output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // gate: [n_ff_exp]
    if (i < (int)n_ff_exp) {
        // `weight_before` (llama4): fold the routing weight into the gate/up projections
        // (silu(w·gate)·(w·up)) instead of scaling the down-projection output — the two
        // differ through the nonlinearity, so it cannot be a single output scalar.
        float wg = weight_before ? weight : 1.0f;
        float wo = weight_before ? 1.0f : weight;
        // compute gate[i] and up[i]
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < (int)ne; j++) {
            g += x[j] * __half2float(gate_w[i * ne + j]);
            u += x[j] * __half2float(up_w[i * ne + j]);
        }
        g *= wg;
        u *= wg;
        // activation
        float a;
        if (act_type == 0) {
            a = g / (1.0f + expf(-g)); // SiLU
        } else if (act_type == 1) {
            float x3 = g * g * g;
            a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3)));
        } else {
            a = 1.0f / (1.0f + expf(-g));
        }
        float h = a * u * wo * down_scale;
        // down projection: accumulate into dst
        const __half* dr = down_w + i; // column i of down_w (row-major: down_w[*, i])
        // Actually down_w is [ne, n_ff_exp], so column i has stride n_ff_exp
        for (int d = 0; d < (int)ne; d++) {
            atomicAdd(&dst[d], h * __half2float(down_w[d * n_ff_exp + i]));
        }
    }
}
"#;

const CONV1D_SILU: &str = r#"
extern "C" __global__ void conv1d_silu(
    const float* __restrict__ x,       // [rows, channels]
    const __half* __restrict__ weight, // [channels, kernel]
    const float* __restrict__ state,   // [(kernel-1), channels] — read-only history
    float* __restrict__ dst,           // [rows, channels]
    int rows,
    int channels,
    int kernel
) {
    // Depthwise causal conv over the VIRTUAL sequence seq = [state ‖ x]: the (kernel-1)
    // warmup columns of `state` (oldest first) followed by the `rows` input columns. For
    // output row `t` the causal window is seq[t .. t+kernel-1]; the current token uses the
    // last tap weight[c, kernel-1], the history taps use the earlier weights. Every
    // (row, channel) is independent — no cross-row carry inside the kernel — so multi-row
    // prefill is correct. The updated state (trailing kernel-1 columns of seq) is written
    // HOST-SIDE in execute() after the kernel. `state` is read-only here (no in-place race).
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int km1 = kernel - 1;
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        const __half* wc = weight + c * kernel;
        for (int k = 0; k < kernel; k++) {
            int i = row + k; // index into the virtual [state ‖ x] sequence
            float xv;
            if (i < km1) {
                xv = state[i * channels + c]; // warmup history column
            } else {
                xv = x[(i - km1) * channels + c]; // input column
            }
            acc += xv * __half2float(wc[k]);
        }
        // SiLU
        float v = acc / (1.0f + expf(-acc));
        dst[row * channels + c] = v;
    }
}
// State update is done HOST-SIDE in execute() after the kernel: the returned state is the
// trailing kernel-1 columns of the virtual [state ‖ x] sequence.
"#;

const DELTANET: &str = r#"
// Gated-DeltaNet linear-attention recurrence (qwen35). One thread per VALUE head; the token
// scan is inherently SEQUENTIAL (state S carries across the `rows` tokens, mutated in place),
// but value heads are fully independent — thread `vh` owns state slice `state[vh*head_k*head_v..]`
// and its own output columns. Matches infr-cpu `deltanet_scan` EXACTLY (within f32 tolerance):
//   - state layout is [n_vhead, head_k, head_v], row-major `S[k*head_v + d]` (NOT transposed),
//   - GQA is the INTERLEAVED `kh = vh % n_khead` tiling (qwen35, not the qwen3next grouping),
//   - the decay uses the NUMERICALLY-STABLE softplus `max(z,0)+log1p(exp(-|z|))` (the naive
//     `log(1+exp(z))` overflows to +inf for large z; with a_coef<0 that collapses decay to 0 and
//     silently wipes the state every token → incoherent output),
//   - `eps` is the caller's value (not a hardcoded constant),
//   - `src_stride>0` fuses q|k|v into one source buffer (q at row offset 0, k at n_khead*head_k,
//     v at 2*n_khead*head_k, per-row stride `src_stride`) — the decode strided path.
// The per-value-dim COLUMN reformulation needs NO per-head scratch arrays (the old `sk[256]`/
// `delta[256]` capped head_v at 256 with a silent OOB): each value dim `d` owns state column
// S[:,d], and kv[d]/delta[d]/out[d] all touch only that column, so decay→kv→delta→update→out
// fuse per-column with head_k/head_v unbounded.
extern "C" __global__ void deltanet(
    const float* __restrict__ q,         // [rows, n_khead*head_k] (or fused src when src_stride>0)
    const float* __restrict__ k,         // [rows, n_khead*head_k]
    const float* __restrict__ v,         // [rows, n_vhead*head_v]
    const float* __restrict__ b,         // [rows, n_vhead]
    const float* __restrict__ a,         // [rows, n_vhead]
    const __half* __restrict__ a_coef,   // [n_vhead]
    const __half* __restrict__ dt_bias,  // [n_vhead]
    float* __restrict__ state,           // [n_vhead, head_k, head_v] — mutated in-place
    float* __restrict__ dst,             // [rows, n_vhead*head_v]
    int rows,
    int n_khead,
    int n_vhead,
    int head_k,
    int head_v,
    float eps,
    int src_stride                       // >0: q/k/v are slices of one buffer with this row stride
) {
    int vh = blockIdx.x * blockDim.x + threadIdx.x;
    if (vh >= n_vhead) return;
    // GQA: value head vh uses q/k head vh % n_khead (interleaved tiling — matches CPU/Metal).
    int kh = vh % n_khead;
    float ac = __half2float(a_coef[vh]);
    float dtb = __half2float(dt_bias[vh]);
    float qscale = rsqrtf((float)head_k);

    // Row strides + within-row offsets for the fused (src_stride>0) vs packed (==0) layouts.
    int qrow = (src_stride > 0) ? src_stride : n_khead * head_k;
    int krow = (src_stride > 0) ? src_stride : n_khead * head_k;
    int vrow = (src_stride > 0) ? src_stride : n_vhead * head_v;
    int koff = (src_stride > 0) ? n_khead * head_k : 0;
    int voff = (src_stride > 0) ? 2 * n_khead * head_k : 0;
    const float* qbase = q;
    const float* kbase = (src_stride > 0) ? q : k;   // fused: k shares q's buffer
    const float* vbase = (src_stride > 0) ? q : v;

    float* S = state + (long)vh * head_k * head_v;
    for (int r = 0; r < rows; r++) {
        const float* qr = qbase + (long)r * qrow + kh * head_k;
        const float* kr = kbase + (long)r * krow + koff + kh * head_k;
        const float* vr = vbase + (long)r * vrow + voff + vh * head_v;
        // L2 norms over head_k (q also scaled by 1/sqrt(head_k)); reciprocal so we multiply below.
        float qsum = 0.0f, ksum = 0.0f;
        for (int i = 0; i < head_k; i++) { qsum += qr[i] * qr[i]; ksum += kr[i] * kr[i]; }
        float qn = 1.0f / sqrtf(qsum + eps);
        float kn = 1.0f / sqrtf(ksum + eps);
        float beta = 1.0f / (1.0f + expf(-b[r * n_vhead + vh]));
        // decay = exp(a_coef * softplus(a + dt_bias)); STABLE softplus (no overflow).
        float z = a[r * n_vhead + vh] + dtb;
        float sp = fmaxf(z, 0.0f) + log1pf(expf(-fabsf(z)));
        float decay = expf(ac * sp);
        float* dr = dst + (long)r * n_vhead * head_v + (long)vh * head_v;
        // Per value dim d (independent state column S[:,d]): decay → kv → delta → update → out.
        for (int d = 0; d < head_v; d++) {
            float kv = 0.0f;
            for (int kk = 0; kk < head_k; kk++) {
                float s = S[kk * head_v + d] * decay;   // S *= decay
                S[kk * head_v + d] = s;
                kv += s * (kr[kk] * kn);                // kv[d] = k_normᵀ S[:,d]
            }
            float delta = (vr[d] - kv) * beta;
            float o = 0.0f;
            for (int kk = 0; kk < head_k; kk++) {
                float s = S[kk * head_v + d] + (kr[kk] * kn) * delta;  // S += k_norm ⊗ delta
                S[kk * head_v + d] = s;
                o += s * (qr[kk] * qn * qscale);        // out[d] = q_normᵀ S[:,d]
            }
            dr[d] = o;
        }
    }
}
"#;

// ── Column-parallel single-token gated-DeltaNet DECODE (qwen35, rows==1) ───────
//
// The sequential `deltanet` kernel runs ONE thread per value head, and its inner `for d` loop walks
// every value column of that head serially — fine for correctness but at decode (rows==1) it leaves
// only n_vhead (~16) threads live on a 96-CU GPU, each grinding a head_v×head_k state update + readout.
// This kernel keeps the recurrence step BYTE-FOR-BYTE identical to that inner loop but spreads the
// value columns across the machine: one BLOCK per value head (grid.x = n_vhead), one THREAD per value
// dim `d` (grid-stride if head_v > blockDim). Column S[:,d] is owned wholly by thread d — the delta-rule
// coupling is the per-head SCALARS (beta, decay, the L2 norms qn/kn), which every thread recomputes
// from the same q/k rows in the same order — so the per-column arithmetic AND its float-reduction order
// match the sequential kernel exactly, and decode output/state are bit-identical to the pre-slice path
// (the qwen35 token-for-token seam gate holds without a re-bless). The token scan is a single step, so
// there is no sequential dependence to carry: `S = S·decay + k̂ ⊗ (β(v − k̂ᵀ(S·decay)))`, `o = q̂ᵀS_new`.
// Preserves GQA (`vh % n_khead`), the fused `src_stride>0` q|k|v layout, the stable softplus decay, and
// the caller's `eps`. exec.rs routes decode (rows==1) here; prefill (rows>1) stays on `deltanet_chunked`.
const DELTANET_DECODE: &str = r#"
extern "C" __global__ void deltanet_decode(
    const float* __restrict__ q,         // [1, n_khead*head_k] (or fused src when src_stride>0)
    const float* __restrict__ k,         // [1, n_khead*head_k]
    const float* __restrict__ v,         // [1, n_vhead*head_v]
    const float* __restrict__ b,         // [1, n_vhead]
    const float* __restrict__ a,         // [1, n_vhead]
    const __half* __restrict__ a_coef,   // [n_vhead]
    const __half* __restrict__ dt_bias,  // [n_vhead]
    float* __restrict__ state,           // [n_vhead, head_k, head_v] — mutated in-place
    float* __restrict__ dst,             // [1, n_vhead*head_v]
    int rows,                            // always 1 on this path (single token)
    int n_khead,
    int n_vhead,
    int head_k,
    int head_v,
    float eps,
    int src_stride                       // >0: q/k/v are slices of one buffer with this row stride
) {
    int vh = blockIdx.x;                 // one block == one value head
    if (vh >= n_vhead) return;
    int kh = vh % n_khead;               // GQA: interleaved value→key head map (matches CPU/Metal)
    float ac = __half2float(a_coef[vh]);
    float dtb = __half2float(dt_bias[vh]);
    float qscale = rsqrtf((float)head_k);

    // Fused (src_stride>0) vs packed (==0) row offsets — same layout as `deltanet` (row 0 only).
    int koff = (src_stride > 0) ? n_khead * head_k : 0;
    int voff = (src_stride > 0) ? 2 * n_khead * head_k : 0;
    const float* qbase = q;
    const float* kbase = (src_stride > 0) ? q : k;   // fused: k shares q's buffer
    const float* vbase = (src_stride > 0) ? q : v;
    const float* qr = qbase + kh * head_k;
    const float* kr = kbase + koff + kh * head_k;
    const float* vr = vbase + voff + vh * head_v;

    // Per-head scalars — recomputed per thread from the same rows in the same order as `deltanet`
    // (a single token's head_k reductions are cheaper than a shared-mem sync), so column math is
    // bit-identical to the sequential inner loop.
    float qsum = 0.0f, ksum = 0.0f;
    for (int i = 0; i < head_k; i++) { qsum += qr[i] * qr[i]; ksum += kr[i] * kr[i]; }
    float qn = 1.0f / sqrtf(qsum + eps);
    float kn = 1.0f / sqrtf(ksum + eps);
    float beta = 1.0f / (1.0f + expf(-b[vh]));
    // decay = exp(a_coef * softplus(a + dt_bias)); STABLE softplus (no overflow).
    float z = a[vh] + dtb;
    float sp = fmaxf(z, 0.0f) + log1pf(expf(-fabsf(z)));
    float decay = expf(ac * sp);

    float* S = state + (long)vh * head_k * head_v;
    float* dr = dst + (long)vh * head_v;
    // Each thread owns value column d (grid-stride covers head_v > blockDim): decay → kv → delta →
    // update → out — identical to the sequential kernel's per-column body.
    for (int d = threadIdx.x; d < head_v; d += blockDim.x) {
        float kv = 0.0f;
        for (int kk = 0; kk < head_k; kk++) {
            float s = S[kk * head_v + d] * decay;   // S *= decay
            S[kk * head_v + d] = s;
            kv += s * (kr[kk] * kn);                // kv[d] = k_normᵀ (S·decay)[:,d]
        }
        float delta = (vr[d] - kv) * beta;
        float o = 0.0f;
        for (int kk = 0; kk < head_k; kk++) {
            float s = S[kk * head_v + d] + (kr[kk] * kn) * delta;  // S += k_norm ⊗ delta
            S[kk * head_v + d] = s;
            o += s * (qr[kk] * qn * qscale);        // out[d] = q_normᵀ S[:,d]
        }
        dr[d] = o;
    }
}
"#;

// ── Chunked / parallel gated-DeltaNet PREFILL (qwen35) ────────────────────────
//
// The sequential `deltanet` kernel runs ONE thread per value head over all `rows` tokens: fine at
// decode (rows==1) but a catastrophic serial scan for prefill (n_vhead≈16 threads on 96 CUs). This
// kernel reformulates the delta-rule recurrence into the standard CHUNKED linear-attention form so
// the per-chunk work is a set of small matrix products done in PARALLEL, and only the chunk-level
// state S carries sequentially (rows/CHUNK steps, not `rows`).
//
// Math (per value head, chunk size C, state S₀ = state at chunk start, log-decay g_t = a_coef·
// softplus(a+dt_bias), inclusive prefix G_j = Σ_{l≤j} g_l; k̂/q̂ L2-normalized, q̂ also ×1/√kd) —
// a byte-for-byte port of the CPU oracle `chunk_delta` (infr-vulkan/tests/chunked_delta_math.rs),
// itself validated equal to the sequential recurrence `deltanet_scan`:
//   R_j     = β_j v_j − β_j e^{G_j}(k̂_jᵀ S₀)                 (initial Δ)
//   A[j][l] = β_j e^{G_j−G_l}(k̂_j·k̂_l)   (l<j, strict lower)
//   Δ_j     = R_j − Σ_{l<j} A[j][l] Δ_l    (unit-lower-triangular forward-substitution)
//   o_i     = e^{G_i}(q̂_iᵀ S₀) + Σ_{j≤i} e^{G_i−G_j}(q̂_i·k̂_j) Δ_j
//   S_C     = e^{G_{C−1}} S₀ + Σ_j e^{G_{C−1}−G_j} k̂_j ⊗ Δ_j
//
// PARALLELIZATION: one BLOCK per value head (grid.x = n_vhead), one THREAD per value dim `d`
// (blockDim.x = max(head_v, DN_CHUNK)). The value columns S[:,d] are INDEPENDENT throughout — the
// A/Δ coupling is via scalars A[j][l] shared by every d — so each thread carries its own column's
// Δ[0..C] in REGISTERS and never needs a cross-thread Δ exchange. Shared memory holds the
// d-independent per-chunk tensors: normalized k̂/q̂ (C×kd), the K̂K̂ᵀ / Q̂K̂ᵀ dot matrices (C×C), and
// the gates (β, G). GQA is the qwen35 INTERLEAVED `kh = vh % n_khead` tiling; `src_stride>0` fuses
// q|k|v into one buffer (same layout as the sequential kernel). Because the chunked form re-orders
// the float reductions vs the sequential scan, outputs match to ~1e-4 (not bit-exact); greedy decode
// and the 2e-2-rel parity gate absorb it. State S is mutated IN PLACE (persistent across calls).
//
// DN_CHUNK is fixed at 16: Δ lives in a `float[DN_CHUNK]` register array, and the shared footprint
// (2·C·kd + 2·C·C + 2·C floats ≈ 18 KB at kd=128) stays under the 32 KB dynamic-LDS ceiling this GPU
// permits a launch to request without the MaxDynamicSharedMemorySize opt-in (an over-budget launch
// silently corrupts LDS instead of erroring — validated: a 41 KB DN_CHUNK=32 build diverged at
// kd=128 while an 18 KB DN_CHUNK=16 build matched the CPU oracle to ~1e-7). exec.rs routes here only
// for rows>1 with head_v≥1 and the footprint within budget; otherwise (and always for decode) it
// uses the sequential `deltanet` kernel.
const DELTANET_CHUNKED: &str = r#"
#define DN_CHUNK 16

extern "C" __global__ void deltanet_chunked(
    const float* __restrict__ q,         // [rows, n_khead*head_k] (or fused src when src_stride>0)
    const float* __restrict__ k,         // [rows, n_khead*head_k]
    const float* __restrict__ v,         // [rows, n_vhead*head_v]
    const float* __restrict__ b,         // [rows, n_vhead]
    const float* __restrict__ a,         // [rows, n_vhead]
    const __half* __restrict__ a_coef,   // [n_vhead]
    const __half* __restrict__ dt_bias,  // [n_vhead]
    float* __restrict__ state,           // [n_vhead, head_k, head_v] — mutated in-place
    float* __restrict__ dst,             // [rows, n_vhead*head_v]
    int rows,
    int n_khead,
    int n_vhead,
    int head_k,
    int head_v,
    float eps,
    int src_stride                       // >0: q/k/v are slices of one buffer with this row stride
) {
    int vh = blockIdx.x;                 // one block == one value head
    if (vh >= n_vhead) return;
    int d = threadIdx.x;                 // one thread == one value dim (column of S)
    int nt = blockDim.x;
    int kh = vh % n_khead;               // GQA: interleaved value→key head map (matches CPU/Metal)
    int kd = head_k, vd = head_v;
    float ac = __half2float(a_coef[vh]);
    float dtb = __half2float(dt_bias[vh]);
    float qscale = rsqrtf((float)kd);

    // Fused (src_stride>0) vs packed (==0) row strides + within-row offsets — same as `deltanet`.
    int qrow = (src_stride > 0) ? src_stride : n_khead * kd;
    int krow = (src_stride > 0) ? src_stride : n_khead * kd;
    int vrow = (src_stride > 0) ? src_stride : n_vhead * vd;
    int koff = (src_stride > 0) ? n_khead * kd : 0;
    int voff = (src_stride > 0) ? 2 * n_khead * kd : 0;
    const float* qbase = q;
    const float* kbase = (src_stride > 0) ? q : k;
    const float* vbase = (src_stride > 0) ? q : v;

    float* S = state + (long)vh * kd * vd;   // [kd, vd] this head's state column-owned by thread d

    // Dynamic shared: d-independent per-chunk tensors.
    extern __shared__ float smem[];
    float* s_kn = smem;                              // [DN_CHUNK, kd]  normalized k̂
    float* s_qn = s_kn + DN_CHUNK * kd;              // [DN_CHUNK, kd]  normalized q̂ (×qscale)
    float* s_KK = s_qn + DN_CHUNK * kd;              // [DN_CHUNK, DN_CHUNK]  k̂_i·k̂_j
    float* s_QK = s_KK + DN_CHUNK * DN_CHUNK;        // [DN_CHUNK, DN_CHUNK]  q̂_i·k̂_j
    float* s_gg = s_QK + DN_CHUNK * DN_CHUNK;        // [DN_CHUNK]  inclusive prefix log-decay G
    float* s_beta = s_gg + DN_CHUNK;                 // [DN_CHUNK]  β

    float delta[DN_CHUNK];                           // this thread's column Δ[0..c) (registers)

    for (int base = 0; base < rows; base += DN_CHUNK) {
        int c = rows - base;
        if (c > DN_CHUNK) c = DN_CHUNK;

        // ── Phase 1: per-token norms, gates, k̂/q̂. Thread j (j<c) owns token j. ──
        if (d < c) {
            int j = d;
            int t = base + j;
            const float* qr = qbase + (long)t * qrow + kh * kd;
            const float* kr = kbase + (long)t * krow + koff + kh * kd;
            float qsum = 0.0f, ksum = 0.0f;
            for (int i = 0; i < kd; i++) { qsum += qr[i] * qr[i]; ksum += kr[i] * kr[i]; }
            float qn = 1.0f / sqrtf(qsum + eps);
            float kn = 1.0f / sqrtf(ksum + eps);
            for (int i = 0; i < kd; i++) {
                s_qn[j * kd + i] = qr[i] * qn * qscale;
                s_kn[j * kd + i] = kr[i] * kn;
            }
            s_beta[j] = 1.0f / (1.0f + expf(-b[t * n_vhead + vh]));
            // Per-token log-decay g_t = a_coef·softplus(a+dt_bias); STABLE softplus (no overflow).
            float z = a[t * n_vhead + vh] + dtb;
            float sp = fmaxf(z, 0.0f) + log1pf(expf(-fabsf(z)));
            s_gg[j] = ac * sp;
        }
        __syncthreads();

        // Inclusive prefix sum → G_j (serial, ≤32 adds; one thread keeps the order deterministic).
        if (d == 0) {
            float run = 0.0f;
            for (int j = 0; j < c; j++) { run += s_gg[j]; s_gg[j] = run; }
        }
        __syncthreads();

        // ── Phase 2: dot matrices K̂K̂ᵀ and Q̂K̂ᵀ (C×C), computed cooperatively over all threads. ──
        for (int idx = d; idx < c * c; idx += nt) {
            int i = idx / c, j = idx % c;
            float dkk = 0.0f, dqk = 0.0f;
            for (int kk = 0; kk < kd; kk++) {
                dkk += s_kn[i * kd + kk] * s_kn[j * kd + kk];
                dqk += s_qn[i * kd + kk] * s_kn[j * kd + kk];
            }
            s_KK[i * DN_CHUNK + j] = dkk;
            s_QK[i * DN_CHUNK + j] = dqk;
        }
        __syncthreads();

        // ── Phase 3: per-column pipeline. Thread d owns state column S[:,d] (columns independent). ──
        if (d < vd) {
            // R: Δ_j ← β_j (v_j[d] − e^{G_j}(k̂_jᵀ S₀[:,d])).
            for (int j = 0; j < c; j++) {
                int t = base + j;
                const float* vr = vbase + (long)t * vrow + voff + vh * vd;
                float ks0 = 0.0f;
                for (int kk = 0; kk < kd; kk++) ks0 += s_kn[j * kd + kk] * S[kk * vd + d];
                float eg = expf(s_gg[j]);
                delta[j] = s_beta[j] * (vr[d] - eg * ks0);
            }
            // Forward substitution: Δ_j −= Σ_{l<j} A[j][l] Δ_l  (Δ_l already finalized, l<j).
            for (int j = 1; j < c; j++) {
                float acc = delta[j];
                for (int l = 0; l < j; l++) {
                    float A = s_beta[j] * expf(s_gg[j] - s_gg[l]) * s_KK[j * DN_CHUNK + l];
                    acc -= A * delta[l];
                }
                delta[j] = acc;
            }
            // O: o_i[d] = e^{G_i}(q̂_iᵀ S₀[:,d]) + Σ_{j≤i} e^{G_i−G_j}(q̂_i·k̂_j) Δ_j[d].
            for (int i = 0; i < c; i++) {
                int t = base + i;
                float qs0 = 0.0f;
                for (int kk = 0; kk < kd; kk++) qs0 += s_qn[i * kd + kk] * S[kk * vd + d];
                float o = expf(s_gg[i]) * qs0;
                for (int j = 0; j <= i; j++) {
                    float w = expf(s_gg[i] - s_gg[j]) * s_QK[i * DN_CHUNK + j];
                    o += w * delta[j];
                }
                dst[(long)t * n_vhead * vd + (long)vh * vd + d] = o;
            }
            // State update: S[kk,d] ← e^{G_{c−1}} S₀[kk,d] + Σ_j e^{G_{c−1}−G_j} k̂_j[kk] Δ_j[d].
            // Reads S₀ (original), writes new S — the O/R reads above already consumed S₀, and no
            // other thread touches column d, so the in-place overwrite is safe.
            float gl = s_gg[c - 1];
            float egl = expf(gl);
            for (int kk = 0; kk < kd; kk++) {
                float acc = egl * S[kk * vd + d];
                for (int j = 0; j < c; j++) {
                    acc += expf(gl - s_gg[j]) * s_kn[j * kd + kk] * delta[j];
                }
                S[kk * vd + d] = acc;
            }
        }
        // Barrier before the next chunk overwrites the shared per-chunk tensors.
        __syncthreads();
    }
}
"#;

const MOE_SHARED_EXPERT_ADD: &str = r#"
extern "C" __global__ void moe_shared_expert_add(
    const float* __restrict__ moe,    // [rows, n]
    const float* __restrict__ shexp,   // [rows, n]
    const float* __restrict__ gate,     // [rows] — pre-sigmoid per-row gate
    float* __restrict__ dst,           // [rows, n]
    int rows,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float g = 1.0f / (1.0f + expf(-gate[row]));
    const float* mr = moe + row * n;
    const float* sr = shexp + row * n;
    float* dr = dst + row * n;
    for (int i = 0; i < n; i++) {
        dr[i] = mr[i] + g * sr[i];
    }
}
"#;

// ── Native in-kernel quant-decode GEMV / EmbedGather (Phase 3) ────────────────
//
// These kernels read the RAW quantized weight bytes and decode each block ON THE FLY,
// so a quantized weight never materializes as an f16 cache in VRAM (VRAM ≈ quant_size
// only) AND decode streams the compact quant bytes (the dominant decode bandwidth
// lever, docs/cpu-perf.md). Covered formats: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, the legacy
// 32-element round quants Q4_0, Q4_1, Q5_0, Q5_1, the codebook 4-bit quants IQ4_NL, IQ4_XS, the
// grid quants IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, the IQ1 quants IQ1_S, IQ1_M, the ternary
// quants TQ1_0, TQ2_0, Q2_0, and the fp4 microscaling quants MXFP4, NVFP4 — ALL 24 weight formats
// as of R7 (`infr_core::decode_spec::WEIGHT_QUANTS` in full; `native_decode_is_total` in exec.rs
// pins it). Nothing quantized takes the dequant→f16 fallback any more: what is left on it is the
// DENSE float dtypes F32/BF16, which have nothing to decode (F16 is already native via
// `linear_f16`).
//
// BIT-FAITHFULNESS to the dequant→f16 cache path (so the blessed goldens do NOT move):
// each element is decoded to the EXACT f32 the host `infr_gguf::dequant::dequant_block`
// produces — same operation order, `sc * code + mn`, with `sc`/`mn` derived identically
// — then rounded to f16 (`__float2half`) exactly as the old CPU dequant cache did
// (`half::f16::from_f32`), and read back as f32 (`__half2float`) exactly as the old
// `linear_f16`/`embed_gather` kernels read the cached f16. The f32 dequant expression is
// compiled with `fp contract(off)` so it is NEVER fused into an FMA — the host reference
// (Rust) does not fuse, and an FMA's single-rounding intermediate could flip the f16
// round and move a golden. The accumulation loop keeps the default contraction so it
// matches `linear_f16`'s accumulation exactly.
const NATIVE_DECODE: &str = r#"
// Read a little-endian f16 (2 bytes) → f32. Byte-wise assembly avoids any alignment
// assumption on the block pointer; the union type-pun is the portable bits→__half path.
__device__ __forceinline__ float rf16b(const unsigned char* p) {
    union { unsigned short u; __half h; } cvt;
    cvt.u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(cvt.h);
}

// Reproduce the host dequant's f32 value `sc*code + mn` WITHOUT FMA contraction, then
// round to f16 and back to f32 — the exact value the old dequant→f16 cache fed the GEMV.
__device__ __forceinline__ float fin(float sc, int code, float mn) {
#pragma clang fp contract(off)
    float val = sc * (float)code + mn;
    return __half2float(__float2half(val));
}

// `fin` for a CODEBOOK quant (IQ4_NL / IQ4_XS): the decoded value is `sc * KV[code]` with NO offset
// term at all — the 4-bit field is an INDEX into a fixed signed table, not a linear quant level, so
// there is nothing to centre. The host oracle (`dequant_codebook`) spells it as exactly one f32
// multiply, so this must NOT go through `fin(sc, kv, 0.0f)`: adding a literal zero is a no-op for
// every finite product but flips `-0.0` to `+0.0`, and the point of this helper is to reproduce the
// oracle's bits, not merely its value. Same f16 round-trip as `fin` (the dequant→f16 cache path).
__device__ __forceinline__ float finc(float sc, int kv) {
#pragma clang fp contract(off)
    float val = sc * (float)kv;
    return __half2float(__float2half(val));
}

// ── Q8_0: 32 elems / 34 bytes = [half d][int8 qs[32]]; y = d*q8 (code = q8+128). ──
__device__ __forceinline__ float deq_q80(const unsigned char* w, long i) {
    long blk = i >> 5;              // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 34;
    float d = rf16b(b);
    int code = (int)((signed char)b[2 + within]) + 128; // biased +128 (dequant_block)
    return fin(d, code, d * (float)(-128));             // sc = d*1, mn = d*(-128)
}

// ── Q5_0: 32 elems / 22 bytes = [half d][u8 qh[4]][u8 qs[16]]; y = d*(q5 − 16), q5 ∈ 0..31. ──
// The 5th bit of element `within` comes from bit `within` of the 32-bit `qh` (low nibbles are the
// first 16, high nibbles the last 16). scale = d, min = d·(−16) — mirrors dequant_row_q5_0.
__device__ __forceinline__ float deq_q50(const unsigned char* w, long i) {
    long blk = i >> 5;             // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 22;
    float d = rf16b(b);
    unsigned int qh = (unsigned int)b[2] | ((unsigned int)b[3] << 8)
                    | ((unsigned int)b[4] << 16) | ((unsigned int)b[5] << 24);
    const unsigned char* qs = b + 6;
    int code;
    if (within < 16) {
        int xh = (int)(((qh >> within) << 4) & 0x10);
        code = (qs[within] & 0x0F) | xh;
    } else {
        int j = within - 16;
        int xh = (int)((qh >> (j + 12)) & 0x10);
        code = (qs[j] >> 4) | xh;
    }
    return fin(d, code, d * (float)(-16));   // sc = d, mn = d·(−16)
}

// ── Legacy 32-block round quants (Q4_0 / Q4_1 / Q5_1) ────────────────────────
// All three share Q5_0's block shape: ONE f16 super-scale per 32 elements, nibble codes packed so the
// LOW nibbles of `qs[0..16]` are elements 0..15 and the HIGH nibbles are elements 16..31. They differ
// only in the min term and whether a 5th code bit exists:
//   Q4_0  sc = d, mn = d·(−8)   code 0..15   (symmetric, like Q5_0's d·(−16))
//   Q4_1  sc = d, mn = m        code 0..15   (AFFINE: a per-block f16 minimum, not a constant)
//   Q5_1  sc = d, mn = m        code 0..31   (affine + Q5_0's `qh` 5th bit)
// The oracle (`dequant_factored`) spells the affine pair as `dd = (d, m)` with multipliers
// `(sc, mn) = (1, 1)`, so the expanded value is `(d·1)·code + (m·1)` — i.e. exactly `fin(d, code, m)`.

// ── Q4_0: 32 elems / 18 bytes = [half d][u8 qs[16]]; y = d*(q4 − 8), q4 ∈ 0..15. ──
__device__ __forceinline__ float deq_q40(const unsigned char* w, long i) {
    long blk = i >> 5;             // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 18;
    float d = rf16b(b);
    const unsigned char* qs = b + 2;
    int code = (within < 16) ? (qs[within] & 0x0F) : (qs[within - 16] >> 4);
    return fin(d, code, d * (float)(-8));    // sc = d, mn = d·(−8)
}

// ── Q4_1: 32 elems / 20 bytes = [half d][half m][u8 qs[16]]; y = d*q4 + m. ──
__device__ __forceinline__ float deq_q41(const unsigned char* w, long i) {
    long blk = i >> 5;             // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 20;
    float d = rf16b(b);
    float mn = rf16b(b + 2);
    const unsigned char* qs = b + 4;
    int code = (within < 16) ? (qs[within] & 0x0F) : (qs[within - 16] >> 4);
    return fin(d, code, mn);                 // sc = d, mn = m (per-block, NOT a constant offset)
}

// ── Q5_1: 32 elems / 24 bytes = [half d][half m][u8 qh[4]][u8 qs[16]]; y = d*q5 + m. ──
// The 5th bit of element `within` comes from bit `within` of the 32-bit `qh`, exactly as in Q5_0 —
// only the header is 2 bytes longer, so `qh` sits at +4 and `qs` at +8.
__device__ __forceinline__ float deq_q51(const unsigned char* w, long i) {
    long blk = i >> 5;             // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 24;
    float d = rf16b(b);
    float mn = rf16b(b + 2);
    unsigned int qh = (unsigned int)b[4] | ((unsigned int)b[5] << 8)
                    | ((unsigned int)b[6] << 16) | ((unsigned int)b[7] << 24);
    const unsigned char* qs = b + 8;
    int code;
    if (within < 16) {
        int xh = (int)(((qh >> within) << 4) & 0x10);
        code = (qs[within] & 0x0F) | xh;
    } else {
        int j = within - 16;
        int xh = (int)((qh >> (j + 12)) & 0x10);
        code = (qs[j] >> 4) | xh;
    }
    return fin(d, code, mn);                 // sc = d, mn = m
}

// ── Codebook 4-bit quants (IQ4_NL / IQ4_XS) ──────────────────────────────────
// R4's new thing: the 4-bit field is an INDEX into the fixed 16-entry signed table `kv_iq4nl`
// (GENERATED from the host `KVALUES_IQ4NL`, assembled at the head of this module), not a linear
// quant level. So there is no `code − offset` and no min term anywhere in this family — the decoded
// value is one multiply, `scale · KV[idx]` (hence `finc`, not `fin`).
// Both share Q4_0's nibble packing: the LOW nibbles of a 16-byte `qs` group are elements 0..15 and
// the HIGH nibbles are elements 16..31. They differ only in where the scale comes from:
//   IQ4_NL  (18 B / 32 elems)   one f16 `d` per 32 elements                    → scale = d
//   IQ4_XS  (136 B / 256 elems) f16 `d` × a 6-bit per-sub-block `ls`, biased   → scale = d·(ls−32)

// ── IQ4_NL: 32 elems / 18 bytes = [half d][u8 qs[16]]; y = d * KV[q4]. ──
__device__ __forceinline__ float deq_iq4nl(const unsigned char* w, long i) {
    long blk = i >> 5;             // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 18;
    float d = rf16b(b);
    const unsigned char* qs = b + 2;
    int idx = (within < 16) ? (qs[within] & 0x0F) : (qs[within - 16] >> 4);
    return finc(d, kv_iq4nl(idx));
}

// ── IQ4_XS: 256 elems / 136 bytes = [half d][u16 scales_h][u8 scales_l[4]][u8 qs[128]]. ──
// y = d·(ls − 32) * KV[q4], where `ls` is the 6-bit scale of the element's 32-element sub-block
// `ib = within/32`: low 4 bits from nibble `ib&1` of `scales_l[ib/2]`, high 2 bits from bits
// `2·ib` of `scales_h`. The oracle computes `dl = d * (ls − 32)` as its own f32 multiply and then
// `dl * KV[idx]`, so the two multiplies stay separate here (two multiplies cannot contract, so no
// `fp contract` pragma is needed on `dl` itself).
__device__ __forceinline__ float deq_iq4xs(const unsigned char* w, long i) {
    long blk = i >> 8;             // / 256
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 136;
    float d = rf16b(b);
    unsigned int scales_h = (unsigned int)b[2] | ((unsigned int)b[3] << 8);
    int ib = p >> 5;               // 32-elem sub-block 0..7
    int within = p & 31;
    int lo = (b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F;
    int hi = (int)((scales_h >> (2 * ib)) & 3u);
    float dl = d * (float)((lo | (hi << 4)) - 32);
    const unsigned char* qs = b + 8 + 16 * ib;
    int idx = (within < 16) ? (qs[within] & 0x0F) : (qs[within - 16] >> 4);
    return finc(dl, kv_iq4nl(idx));
}

// ── GRID quants (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S) ─────────────────
// R5's new thing, one axis past R4's codebook: the stored code is an index into a table of packed
// signed-byte VECTORS (`iquant_grids`, generated into the head of this module), and a separate sign
// bit per element negates the entry's byte. So an element is `db · gv · sign` — still no offset
// (hence `fing`, not `fin`), but the value now comes from a 1..8 KiB table addressed by an 8/9/10-bit
// code instead of a 16-entry one, and the SIGN is a second, independently packed field.
//
// All five are 256-element super-blocks walked as 8 sub-blocks of 32, each sub-block as 4 groups of
// 8 — one group is exactly one grid entry (IQ2: one 8-byte entry; IQ3: TWO 4-byte entries) plus one
// 8-bit sign pattern. So for element `p` in 0..255:  ib32 = p>>5,  l = (p>>3)&3,  j = p&7.
// The families differ in only three places:
//                    grid index      sign source            f32 scale per 32-elem block
//   IQ2_XXS  66 B    8b  (aux0)      ksigns[7b of aux1]     ONE, d·(0.5+aux1>>28)·0.25
//   IQ2_XS   74 B    9b  (qs16)      ksigns[qs16>>9]        TWO, one per 16 elems (scales nibbles)
//   IQ2_S    82 B    8b+2b(qh)       raw byte qs[32+…]      TWO, one per 16 elems (scales nibbles)
//   IQ3_XXS  98 B    8b ×2 (qs)      ksigns[7b of aux32]    ONE, d·(0.5+aux32>>28)·0.5
//   IQ3_S   110 B    8b+1b(qh) ×2    raw byte signs[…]      ONE, d·(1+2·scale nibble)
// The "TWO scales per 32-element block" cases are why the int8 and WMMA tiers below carry a scale
// per 16-element K-tile rather than Q8_0's single one.

// Little-endian u32 from an unaligned byte pointer — nothing inside a 66/74/82/98/110-byte
// super-block is guaranteed 4-aligned, so this must not become an `int` load.
__device__ __forceinline__ unsigned int rd32b(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8)
         | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

// Signed byte `j` of a packed grid entry (8 bytes for the IQ2 grids, 4 for the IQ3 ones).
__device__ __forceinline__ int gsb8(unsigned long long g, int j) {
    return (int)(signed char)((g >> (8 * j)) & 0xFFull);
}
__device__ __forceinline__ int gsb4(unsigned int g, int j) {
    return (int)(signed char)((g >> (8 * j)) & 0xFFu);
}

// The three per-sub-block scale spellings, each the host oracle's expression verbatim (`dequant_
// codebook`'s IQ2_*/IQ3_* arms). Contraction is pinned off for the same reason `fin` pins it: the
// host reference does not fuse, and an FMA's single-rounding intermediate could move the f16 round.
__device__ __forceinline__ float iq2_db(float d, int ls) {
#pragma clang fp contract(off)
    return d * (0.5f + (float)ls) * 0.25f;
}
__device__ __forceinline__ float iq3xxs_db(float d, int ls) {
#pragma clang fp contract(off)
    return d * (0.5f + (float)ls) * 0.5f;
}
__device__ __forceinline__ float iq3s_db(float d, int ls) {
#pragma clang fp contract(off)
    return d * (1.0f + 2.0f * (float)ls);
}

// `fin` for a GRID quant. The host `apply_signs` spells the element as `db * gv * sign` with `sign`
// a literal ±1.0f, so reproduce THAT product — not `fin(db, gv, 0.0f)` (a literal +0.0 addend would
// flip −0.0) and not `-(db*gv)` (equal for every finite value, but this helper exists to reproduce
// the oracle's expression, not merely its value). Same f16 round-trip as every other native decode.
__device__ __forceinline__ float fing(float db, int gv, int neg) {
#pragma clang fp contract(off)
    float val = db * (float)gv * (neg ? -1.0f : 1.0f);
    return __half2float(__float2half(val));
}

// ── IQ2_XXS: 256 elems / 66 bytes = [half d][u16 qs[32]]. ──
// Sub-block `ib32` owns 8 bytes of `qs`: `aux0` = four 8-bit grid indices, `aux1` = four 7-bit
// sign-pattern indices (bits 0..27) + the 4-bit scale magnitude (bits 28..31).
__device__ __forceinline__ float deq_iq2xxs(const unsigned char* w, long i) {
    long blk = i >> 8;                 // / 256
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 66;
    float d = rf16b(b);
    const unsigned char* q8 = b + 2 + (p >> 5) * 8;
    unsigned int aux0 = rd32b(q8), aux1 = rd32b(q8 + 4);
    int l = (p >> 3) & 3, j = p & 7;
    unsigned long long g = g_iq2xxs[(aux0 >> (8 * l)) & 0xFFu];
    unsigned int sg = ksigns_iq2xs[(aux1 >> (7 * l)) & 127u];
    return fing(iq2_db(d, (int)(aux1 >> 28)), gsb8(g, j), (int)((sg >> j) & 1u));
}

// ── IQ2_XS: 256 elems / 74 bytes = [half d][u16 qs[32]][u8 scales[8]]. ──
// Each `qs` u16 is a 9-bit grid index + a 7-bit ksigns index. The sub-block's scale byte holds TWO
// 4-bit magnitudes: the low nibble scales groups l=0,1 (elements 0..15) and the high nibble l=2,3.
__device__ __forceinline__ float deq_iq2xs(const unsigned char* w, long i) {
    long blk = i >> 8;
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 74;
    float d = rf16b(b);
    int ib32 = p >> 5, l = (p >> 3) & 3, j = p & 7;
    const unsigned char* qs = b + 2 + ib32 * 8 + l * 2;
    unsigned int q16 = (unsigned int)qs[0] | ((unsigned int)qs[1] << 8);
    unsigned int sc = b[66 + ib32];
    float db = iq2_db(d, (int)((l < 2) ? (sc & 0xFu) : (sc >> 4)));
    unsigned long long g = g_iq2xs[q16 & 511u];
    unsigned int sg = ksigns_iq2xs[q16 >> 9];
    return fing(db, gsb8(g, j), (int)((sg >> j) & 1u));
}

// ── IQ2_S: 256 elems / 82 bytes = [half d][u8 qs[64]][u8 qh[8]][u8 scales[8]]. ──
// `qs[0..32]` are the low 8 bits of the grid index and `qs[32..64]` the sign patterns THEMSELVES
// (no ksigns indirection); `qh[ib32]` supplies each group's 2 high index bits at shift `8 − 2l`.
// The scale byte splits per 16 elements exactly as IQ2_XS's does.
__device__ __forceinline__ float deq_iq2s(const unsigned char* w, long i) {
    long blk = i >> 8;
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 82;
    float d = rf16b(b);
    int ib32 = p >> 5, l = (p >> 3) & 3, j = p & 7;
    const unsigned char* qs = b + 2;
    int qi = ib32 * 4 + l;
    unsigned int sc = b[74 + ib32];
    float db = iq2_db(d, (int)((l < 2) ? (sc & 0xFu) : (sc >> 4)));
    unsigned int gidx = (unsigned int)qs[qi] | ((((unsigned int)b[66 + ib32]) << (8 - 2 * l)) & 0x300u);
    unsigned int sg = qs[32 + qi];
    return fing(db, gsb8(g_iq2s[gidx], j), (int)((sg >> j) & 1u));
}

// ── IQ3_XXS: 256 elems / 98 bytes = [half d][u8 qs[64]][u8 scales_and_signs[32]]. ──
// The IQ3 grids hold FOUR bytes per entry, so a group of 8 needs TWO entries: elements 0..3 come
// from `qs[2l]` and 4..7 from `qs[2l+1]`, sharing one 8-bit sign pattern (element `j` always takes
// sign bit `j`, which is why the oracle's second `apply_signs` call passes `sign_off = 4`).
__device__ __forceinline__ float deq_iq3xxs(const unsigned char* w, long i) {
    long blk = i >> 8;
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 98;
    float d = rf16b(b);
    int ib32 = p >> 5, l = (p >> 3) & 3, j = p & 7;
    unsigned int aux32 = rd32b(b + 66 + 4 * ib32);
    unsigned int sg = ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
    unsigned int g = g_iq3xxs[b[2 + ib32 * 8 + 2 * l + (j >> 2)]];
    return fing(iq3xxs_db(d, (int)(aux32 >> 28)), gsb4(g, j & 3), (int)((sg >> j) & 1u));
}

// ── IQ3_S: 256 elems / 110 bytes = [half d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]]. ──
// Same two-entries-per-group shape as IQ3_XXS, with raw sign bytes instead of ksigns and a 9th
// index bit per entry from `qh[ib32]` (shift `8 − 2l` for the first entry, `7 − 2l` for the second).
// The scale is one nibble per 32-element sub-block: `scales[ib32/2]`, low nibble for even `ib32`.
__device__ __forceinline__ float deq_iq3s(const unsigned char* w, long i) {
    long blk = i >> 8;
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 110;
    float d = rf16b(b);
    int ib32 = p >> 5, l = (p >> 3) & 3, j = p & 7;
    unsigned int sc = b[106 + (ib32 >> 1)];
    float db = iq3s_db(d, (int)((ib32 & 1) ? (sc >> 4) : (sc & 0xFu)));
    int hlf = j >> 2;                  // 0 = first grid entry of the group, 1 = second
    unsigned int qh = b[66 + ib32];
    unsigned int gidx = (unsigned int)b[2 + ib32 * 8 + 2 * l + hlf]
                      | ((qh << ((8 - hlf) - 2 * l)) & 256u);
    unsigned int sg = b[74 + ib32 * 4 + l];
    return fing(db, gsb4(g_iq3s[gidx], j & 3), (int)((sg >> j) & 1u));
}

// ── IQ1 quants (IQ1_S / IQ1_M) — R5's grid shape plus a fractional ADDEND ────
// R6's genuinely new decode shape. Like R5 the stored code indexes a table of packed signed-byte
// vectors (here `g_iq1s`, 2048 entries / an 11-bit index — shared by BOTH formats), but there is no
// sign field: instead the decoded element is
//     y = dl · (gv + delta),   delta = ±IQ1S_DELTA (±0.125)
// i.e. a per-group ADDEND applied INSIDE the code's own scale. Nothing in R1..R5 has that shape —
// it is not the affine `d·code + m` either, whose `m` sits OUTSIDE the code's scale, which is why
// this needs its own `fina` helper rather than `fin` or `fing`.
//
// Both walk 256 elements as 8 sub-blocks of 32, each 4 groups of 8 (ib = p>>5, l = (p>>3)&3,
// j = p&7 — R5's traversal exactly). They differ in where the scale, the 3 high index bits and the
// delta sign come from:
//                grid index          delta sign        scale per group
//   IQ1_S  50 B  qs[4ib+l] | qh<<8   qh bit 15         ONE per 32: d·(2·((qh>>12)&7)+1)
//   IQ1_M  56 B  qs[4ib+l] | qh<<8|4 qh byte 0x08/0x80 TWO per 32: dl1 (l<2) / dl2 (l≥2)
// IQ1_M has no standalone `d` field at all — its f16 bits are the TOP NIBBLES of the four u16
// scale words, whose low 12 bits carry the four 3-bit `dl` sub-scales (`iq1m_d` reassembles it;
// `infr_core::decode_spec::ScaleEnc::Iq1mSplitF16` is the host description of the same layout).

// The `2·ls + 1` sub-scale form both IQ1 formats use, contraction pinned off for the same reason
// `fin` pins it: the host reference does not fuse and an FMA could move the f16 round.
__device__ __forceinline__ float iq1_dl(float d, int ls) {
#pragma clang fp contract(off)
    return d * (2.0f * (float)ls + 1.0f);
}

// IQ1_M's `d`: no standalone f16 field — nibble `i` of the f16 lives in bits 12..16 of the little-
// endian u16 scale word `i` (block bytes 48..56). Reassemble the bits and read them as a __half,
// exactly as the host oracle does with `half::f16::from_bits`.
__device__ __forceinline__ float iq1m_d(const unsigned char* b) {
    unsigned int s0 = (unsigned int)b[48] | ((unsigned int)b[49] << 8);
    unsigned int s1 = (unsigned int)b[50] | ((unsigned int)b[51] << 8);
    unsigned int s2 = (unsigned int)b[52] | ((unsigned int)b[53] << 8);
    unsigned int s3 = (unsigned int)b[54] | ((unsigned int)b[55] << 8);
    union { unsigned short u; __half h; } cvt;
    cvt.u = (unsigned short)((s0 >> 12) | ((s1 >> 8) & 0x00f0u)
                           | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u));
    return __half2float(cvt.h);
}

// `fin` for an IQ1 quant. The host spells the element as `dl * (gv + delta)` — ONE add inside ONE
// multiply — so reproduce THAT, not `fin(dl, gv, dl*delta)` (algebraically equal, two roundings
// instead of one) and not `finc`. Same f16 round-trip as every other native decode.
__device__ __forceinline__ float fina(float dl, int gv, float delta) {
#pragma clang fp contract(off)
    float val = dl * ((float)gv + delta);
    return __half2float(__float2half(val));
}

// ── IQ1_S: 256 elems / 50 bytes = [half d][u8 qs[32]][u16 qh[8]]. ──
// Sub-block `ib` owns one `qh` u16: bits 0..11 are the four groups' 3 high index bits (group `l` at
// shift `3l`), bits 12..14 the sub-scale, bit 15 the delta sign.
__device__ __forceinline__ float deq_iq1s(const unsigned char* w, long i) {
    long blk = i >> 8;                 // / 256
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 50;
    int ib = p >> 5, l = (p >> 3) & 3, j = p & 7;
    const unsigned char* qhp = b + 34 + 2 * ib;
    unsigned int qh = (unsigned int)qhp[0] | ((unsigned int)qhp[1] << 8);
    float dl = iq1_dl(rf16b(b), (int)((qh >> 12) & 7u));
    float delta = (qh & 0x8000u) ? -IQ1S_DELTA : IQ1S_DELTA;
    unsigned int gidx = (unsigned int)b[2 + ib * 4 + l] | (((qh >> (3 * l)) & 7u) << 8);
    return fina(dl, gsb8(g_iq1s[gidx], j), delta);
}

// ── IQ1_M: 256 elems / 56 bytes = [u8 qs[32]][u8 qh[16]][u8 scales[8]]. ──
// Two `qh` BYTES per sub-block: byte `ib*2` serves groups l=0,1 and byte `ib*2+1` groups l=2,3.
// Within a byte the even group takes its 3 high index bits at shift 8 and its delta from bit 0x08,
// the odd group at shift 4 and from bit 0x80. The sub-scale is the 3-bit field at `6·(ib&1)` (+3
// for l≥2) of scale word `ib>>1`.
__device__ __forceinline__ float deq_iq1m(const unsigned char* w, long i) {
    long blk = i >> 8;                 // / 256
    int p = (int)(i & 255);
    const unsigned char* b = w + blk * 56;
    int ib = p >> 5, l = (p >> 3) & 3, j = p & 7;
    const unsigned char* sp = b + 48 + 2 * (ib >> 1);
    unsigned int scw = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8);
    float dl = iq1_dl(iq1m_d(b), (int)((scw >> (6 * (ib & 1) + ((l < 2) ? 0 : 3))) & 7u));
    unsigned int qhb = b[32 + ib * 2 + (l >> 1)];
    unsigned int gidx = (unsigned int)b[ib * 4 + l] | ((qhb << ((l & 1) ? 4 : 8)) & 0x700u);
    float delta = (qhb & ((l & 1) ? 0x80u : 0x08u)) ? -IQ1S_DELTA : IQ1S_DELTA;
    return fina(dl, gsb8(g_iq1s[gidx], j), delta);
}

// ── TERNARY quants (TQ1_0 / TQ2_0 / Q2_0) ────────────────────────────────────
// The other half of R6, and the SIMPLEST family in the set: no grid, no codebook, no sign field, no
// sub-block scales — ONE f16 `d` per block and a small unsigned code that dequants to
// `y = (code − 1) · d` with the level set {−1, 0, +1} (TQ1_0) or {−1, 0, +1, +2} (TQ2_0 / Q2_0,
// whose top code is unused in practice). The `−1` is a CONSTANT offset, so unlike the affine quants
// it is folded straight into the signed weight code and the int8 tier carries no ones-dot either.
// Since the value is one f32 multiply with no addend, the oracle's expression is `finc`'s.
// They differ only in the packing:
//   TQ1_0  54 B / 256   FIVE base-3 digits per byte, `digit = ((u8)(byte·3ⁿ) · 3) >> 8`
//   TQ2_0  66 B / 256   4 elements per byte at 2 bits, two 32-byte chunks × 4 shifts × 32
//   Q2_0   18 B /  64   4 elements per byte at 2 bits, SEQUENTIAL (infr's own 64-element format)

// 3ⁿ for n ∈ 0..4 as a select cascade, NOT a local array: a dynamically indexed local `const int[5]`
// lowers to per-invocation scratch on AMDGCN, and this is on the innermost decode path.
__device__ __forceinline__ unsigned int tq1_pow3(int n) {
    return (n < 2) ? ((n < 1) ? 1u : 3u) : ((n < 3) ? 9u : ((n < 4) ? 27u : 81u));
}

// Signed ternary code (`digit − 1` ∈ {−1,0,+1}) for element `p` of the TQ1_0 super-block at `b`.
// The dequant emits its 256 elements in THREE segments, and this is that walk inverted:
//   p <  160   qs[0..32]  × 5 digit passes   byte = qs[p&31],        n = p>>5
//   p <  240   qs[32..48] × 5 digit passes   byte = qs[32+((p−160)&15)], n = (p−160)>>4
//   else       qh[0..4]   × 4 digit passes   byte = qh[(p−240)&3],   n = (p−240)>>2
// The byte×3ⁿ product WRAPS at 8 bits (the host does a `u8::wrapping_mul`), which is what makes the
// base-3 digit extraction work — masking to 0xFF here is that wrap, not a defensive clamp.
__device__ __forceinline__ int tq10_code(const unsigned char* b, int p) {
    int n, by;
    if (p < 160)      { n = p >> 5;              by = b[p & 31]; }
    else if (p < 240) { int q = p - 160; n = q >> 4; by = b[32 + (q & 15)]; }
    else              { int q = p - 240; n = q >> 2; by = b[48 + (q & 3)]; }
    unsigned int v = ((unsigned int)by * tq1_pow3(n)) & 0xFFu;
    return (int)((v * 3u) >> 8) - 1;
}

// ── TQ1_0: 256 elems / 54 bytes = [u8 qs[48]][u8 qh[4]][half d]. ──
__device__ __forceinline__ float deq_tq10(const unsigned char* w, long i) {
    const unsigned char* b = w + (i >> 8) * 54;
    return finc(rf16b(b + 52), tq10_code(b, (int)(i & 255)));
}

// ── TQ2_0: 256 elems / 66 bytes = [u8 qs[64]][half d]. ──
// Element `p` = chunk `p>>7` (which 32-byte half of `qs`), shift `2·((p>>5)&3)`, byte `p&31`.
__device__ __forceinline__ float deq_tq20(const unsigned char* w, long i) {
    int p = (int)(i & 255);
    const unsigned char* b = w + (i >> 8) * 66;
    int code = (int)((b[(p >> 7) * 32 + (p & 31)] >> (2 * ((p >> 5) & 3))) & 3u) - 1;
    return finc(rf16b(b + 64), code);
}

// ── Q2_0: 64 elems / 18 bytes = [half d][u8 qs[16]]. ──
// infr's OWN ternary format (the only 64-element block in the natively decoded set): element `j`
// is simply the 2-bit field at shift `2·(j&3)` of byte `j>>2` — no chunking, no digit packing.
__device__ __forceinline__ float deq_q20(const unsigned char* w, long i) {
    int p = (int)(i & 63);
    const unsigned char* b = w + (i >> 6) * 18;   // 64 elements per block
    int code = (int)((b[2 + (p >> 2)] >> (2 * (p & 3))) & 3u) - 1;
    return finc(rf16b(b), code);
}

// ── FP4 microscaling quants (MXFP4 / NVFP4) ──────────────────────────────────
// R7's new thing is NOT the value path — the 4-bit field is an index into the fixed 16-entry signed
// E2M1 codebook `kv_mxfp4` (GENERATED at the head of this module from the host `KVALUES_MXFP4`), so
// the element is one multiply `d · KV[idx]` with no offset, exactly R4's IQ4_NL shape and `finc`.
// What is new is the SCALE ENCODING: neither format stores an f16 `d`. Both are decoded here from
// the host oracle's own definitions (`infr_gguf::dequant::{e8m0_to_fp32_half, ue4m3_to_fp32}`),
// which is where the two-line functions below come from — not from the OCP/NVIDIA specs, whose
// "halved" convention llama.cpp folds into the decode differently than a naive reading would.
//
//   MXFP4  17 B / 32 elems   [u8 e (E8M0)][u8 qs[16]]        d = 2^(e − 128), ONE scale per 32
//   NVFP4  36 B / 64 elems   [u8 d[4] (UE4M3)][u8 qs[32]]    FOUR scales, one per 16 elems
//
// Nibble packing differs from IQ4_NL in NVFP4's case: MXFP4 keeps the familiar 16-wide split (low
// nibbles of `qs[0..16]` are elements 0..15, high nibbles 16..31), but NVFP4 splits per 16-element
// SUB-BLOCK — low nibbles of the sub-block's 8 code bytes are its elements 0..7, high nibbles 8..15.
//
// THE E8M0 SCALE IS A PURE POWER OF TWO, which is the property the int8 tier leans on: `d · code`
// is then exact for every codebook entry (no mantissa bits are consumed), so re-associating the
// oracle's per-element `d · KV[idx]` into the tier's per-block `d · Σ(KV[idx]·a)` is a re-association
// and not an approximation — R6's argument for the IQ1 ×8 fold, reached here for free. NVFP4's
// UE4M3 scale is a genuine FP8 value with a 3-bit mantissa, but its codebook entries are ≤ 4 bits
// wide, so `d · KV` still needs at most 7 mantissa bits and stays exact in f32 as well.

// E8M0: a BARE 8-bit exponent — no sign, no mantissa — so the value is exactly `2^(x − 128)` over
// the whole range. Transcribed from `infr_gguf::dequant::e8m0_to_fp32_half` (which mirrors
// llama.cpp's `ggml_e8m0_to_fp32_half`), INCLUDING its two-case form: for `x ≥ 2` the byte is
// dropped straight into the f32 exponent field as `x − 1` (biased 127, hence 2^(x−128)), and for
// `x ∈ {0,1}` — where `x − 1` would be a zero/negative exponent field — the result is the SUBNORMAL
// `0x00200000 << x`, i.e. 2^-128 and 2^-127. Reproducing both cases is what makes the smallest two
// scales decode rather than flushing to zero.
__device__ __forceinline__ float e8m0_half(unsigned int x) {
    union { unsigned int u; float f; } cvt;
    cvt.u = (x < 2u) ? (0x00200000u << x) : ((x - 1u) << 23);
    return cvt.f;
}

// UE4M3: an UNSIGNED FP8 (4 exponent bits biased 7, 3 mantissa bits) halved, i.e.
// `0.5 · 2^(e−7) · (1 + m/8)` for `e > 0` and the subnormal `0.5 · m · 2^-9` for `e == 0`, with the
// two reserved codes 0x00 and 0x7F decoding to 0.0. Transcribed from
// `infr_gguf::dequant::ue4m3_to_fp32` (llama.cpp's `ggml_ue4m3_to_fp32`) case for case — the `·0.5`
// tail and the 0x7F hole are both part of the oracle, and dropping either doubles or NaNs a scale.
__device__ __forceinline__ float ue4m3(unsigned int x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    int e = (int)((x >> 3) & 0xFu);
    float man = (float)(x & 7u);
    float raw = (e == 0) ? (man * exp2f(-9.0f)) : ((1.0f + man / 8.0f) * exp2f((float)(e - 7)));
    return raw * 0.5f;
}

// ── MXFP4: 32 elems / 17 bytes = [u8 e][u8 qs[16]]; y = KV[q4] · 2^(e−128). ──
__device__ __forceinline__ float deq_mxfp4(const unsigned char* w, long i) {
    long blk = i >> 5;             // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 17;
    float d = e8m0_half(b[0]);
    const unsigned char* qs = b + 1;
    int idx = (within < 16) ? (qs[within] & 0x0F) : (qs[within - 16] >> 4);
    return finc(d, kv_mxfp4(idx));
}

// ── NVFP4: 64 elems / 36 bytes = [u8 d[4]][u8 qs[32]]; y = KV[q4] · ue4m3(d[s]). ──
// Element `p` lives in 16-element sub-block `s = p>>4` at position `within = p&15`; the sub-block's
// 8 code bytes are `qs[s*8 .. s*8+8]` with the low nibbles as `within` 0..7 and the high nibbles as
// 8..15 — an 8-wide split, NOT the 16-wide one every other nibble format in this file uses.
__device__ __forceinline__ float deq_nvfp4(const unsigned char* w, long i) {
    int p = (int)(i & 63);
    const unsigned char* b = w + (i >> 6) * 36;   // 64 elements per block
    int s = p >> 4, within = p & 15;
    float d = ue4m3(b[s]);
    const unsigned char* qs = b + 4 + s * 8;
    int idx = (within < 8) ? (qs[within] & 0x0F) : (qs[within - 8] >> 4);
    return finc(d, kv_mxfp4(idx));
}

// ── Per-32-block int8 decode: the `wdec_*` family ────────────────────────────
// The SHARED body of the dp4a GEMV (`linear_i8_*` / `i8acc_*`) and the WMMA prefill tier: decode
// 32-block `blk` of output row `col` into 32 SIGNED codes plus the two f32 scales its 16-element
// halves carry (`*s0` for elements 0..15, `*s1` for 16..31 — equal for the formats whose scale is
// per-32). Having one decoder per format instead of one per format×tier is what keeps the three
// tiers provably in agreement, and it is what lets ONE `GEN_WMMA_WDEC` body serve all of them.
//
// EVERY format on this seam has a SIGNED code and therefore NO ones-dot / min-correction term
// anywhere — that is the property that admits the shared body, and it is why the affine formats
// (which need an `isum` against an all-ones B operand) keep their own hand-written kernels:
//   * R5's grid quants — the grid byte is already signed, the sign bit merely negates it.
//     |code| ≤ 62 (IQ3_XXS's widest grid byte).
//   * R6's IQ1 quants — the element is `dl·(gv + delta)` with `gv ∈ {−1,0,+1}` and
//     `delta = ±0.125`, so ×8 makes it EXACTLY integer: `code = 8·gv ± 1 ∈ {−9,−7,−1,+1,+7,+9}`
//     with the scale `dl·0.125`. Both halves of that identity are exact in binary (0.125 is a power
//     of two; `dl` is never near subnormal), so this is a re-association of the oracle, not an
//     approximation — and it dissolves the addend into the code instead of paying a per-group
//     ones-dot, which IQ1_M would need PER GROUP OF 8 (its delta sign varies per 8) rather than
//     per 32. |code| ≤ 9.
//   * R6's ternary quants — the constant `−1` offset is folded into the stored code directly,
//     giving `code ∈ {−1,0,+1}` (TQ1_0) or `{−1,0,+1,+2}` (TQ2_0/Q2_0). |code| ≤ 2.
//   * R7's fp4 quants — the E2M1 codebook entry IS the signed operand, R4's treatment with a
//     different table. |code| ≤ 12. What makes the re-association EXACT rather than merely
//     tolerable is the scale: MXFP4's E8M0 is a pure power of two, and NVFP4's 3-bit-mantissa
//     UE4M3 against a ≤4-bit code still fits f32's 24 bits — so `sc · Σ(code·a)` reproduces the
//     oracle's `Σ((sc·code)·a)` term for term.
// The widest of those is 62, so a 32-wide dot against int8 activations stays far inside i32, and
// every code fits the int8 WMMA operand.
//
// For the grid formats the grid entry and the sign pattern are fetched ONCE PER GROUP OF 8 and the
// 8 elements peeled off them in registers. That hoisting is the whole performance story for those:
// a per-element gather re-reads the same table entry 8 times and leaves an already ALU-heavy decode
// gather-bound (the finding Vulkan's grid GEMVs reached independently). The ternary formats have no
// table at all, so their bodies are a flat 32-element unpack.
__device__ __forceinline__ void wdec_iq2xxs(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 66;
    const unsigned char* q8 = b + 2 + (blk & 7) * 8;
    unsigned int aux0 = rd32b(q8), aux1 = rd32b(q8 + 4);
    float db = iq2_db(rf16b(b), (int)(aux1 >> 28));
    *s0 = db; *s1 = db;
    for (int l = 0; l < 4; l++) {
        unsigned long long g = g_iq2xxs[(aux0 >> (8 * l)) & 0xFFu];
        unsigned int sg = ksigns_iq2xs[(aux1 >> (7 * l)) & 127u];
        for (int j = 0; j < 8; j++) {
            int gv = gsb8(g, j);
            code[l * 8 + j] = (signed char)(((sg >> j) & 1u) ? -gv : gv);
        }
    }
}

__device__ __forceinline__ void wdec_iq2xs(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 74;
    int ib32 = blk & 7;
    const unsigned char* qs = b + 2 + ib32 * 8;
    float d = rf16b(b);
    unsigned int sc = b[66 + ib32];
    *s0 = iq2_db(d, (int)(sc & 0xFu));
    *s1 = iq2_db(d, (int)(sc >> 4));
    for (int l = 0; l < 4; l++) {
        unsigned int q16 = (unsigned int)qs[2 * l] | ((unsigned int)qs[2 * l + 1] << 8);
        unsigned long long g = g_iq2xs[q16 & 511u];
        unsigned int sg = ksigns_iq2xs[q16 >> 9];
        for (int j = 0; j < 8; j++) {
            int gv = gsb8(g, j);
            code[l * 8 + j] = (signed char)(((sg >> j) & 1u) ? -gv : gv);
        }
    }
}

__device__ __forceinline__ void wdec_iq2s(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 82;
    int ib32 = blk & 7;
    const unsigned char* qs = b + 2;
    float d = rf16b(b);
    unsigned int sc = b[74 + ib32], qh = b[66 + ib32];
    *s0 = iq2_db(d, (int)(sc & 0xFu));
    *s1 = iq2_db(d, (int)(sc >> 4));
    for (int l = 0; l < 4; l++) {
        int qi = ib32 * 4 + l;
        unsigned long long g = g_iq2s[(unsigned int)qs[qi] | ((qh << (8 - 2 * l)) & 0x300u)];
        unsigned int sg = qs[32 + qi];
        for (int j = 0; j < 8; j++) {
            int gv = gsb8(g, j);
            code[l * 8 + j] = (signed char)(((sg >> j) & 1u) ? -gv : gv);
        }
    }
}

__device__ __forceinline__ void wdec_iq3xxs(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 98;
    int ib32 = blk & 7;
    const unsigned char* qs = b + 2 + ib32 * 8;
    unsigned int aux32 = rd32b(b + 66 + 4 * ib32);
    float db = iq3xxs_db(rf16b(b), (int)(aux32 >> 28));
    *s0 = db; *s1 = db;
    for (int l = 0; l < 4; l++) {
        unsigned int sg = ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
        unsigned int g1 = g_iq3xxs[qs[2 * l]], g2 = g_iq3xxs[qs[2 * l + 1]];
        for (int j = 0; j < 8; j++) {
            int gv = gsb4((j < 4) ? g1 : g2, j & 3);
            code[l * 8 + j] = (signed char)(((sg >> j) & 1u) ? -gv : gv);
        }
    }
}

__device__ __forceinline__ void wdec_iq3s(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 110;
    int ib32 = blk & 7;
    const unsigned char* qs = b + 2 + ib32 * 8;
    const unsigned char* sgs = b + 74 + ib32 * 4;
    unsigned int qh = b[66 + ib32], sc = b[106 + (ib32 >> 1)];
    float db = iq3s_db(rf16b(b), (int)((ib32 & 1) ? (sc >> 4) : (sc & 0xFu)));
    *s0 = db; *s1 = db;
    for (int l = 0; l < 4; l++) {
        unsigned int g1 = g_iq3s[(unsigned int)qs[2 * l]     | ((qh << (8 - 2 * l)) & 256u)];
        unsigned int g2 = g_iq3s[(unsigned int)qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256u)];
        unsigned int sg = sgs[l];
        for (int j = 0; j < 8; j++) {
            int gv = gsb4((j < 4) ? g1 : g2, j & 3);
            code[l * 8 + j] = (signed char)(((sg >> j) & 1u) ? -gv : gv);
        }
    }
}

// R6 IQ1: the ×8 fold described above turns `dl·(gv + delta)` into `(dl·0.125)·(8·gv ± 1)`, so the
// delta never reaches the f32 epilogue and the family needs no ones-dot. IQ1_S's scale and delta
// sign are per-32 (`*s0 == *s1`); IQ1_M's scale is per-16 and its delta sign per-8, which is why
// the sign is recomputed inside the group loop there and hoisted out of it here.
__device__ __forceinline__ void wdec_iq1s(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 50;
    int ib = blk & 7;
    const unsigned char* qhp = b + 34 + 2 * ib;
    unsigned int qh = (unsigned int)qhp[0] | ((unsigned int)qhp[1] << 8);
    float ds = iq1_dl(rf16b(b), (int)((qh >> 12) & 7u)) * IQ1S_DELTA;
    *s0 = ds; *s1 = ds;
    int sgn = (qh & 0x8000u) ? -1 : 1;
    const unsigned char* qs = b + 2 + ib * 4;
    for (int l = 0; l < 4; l++) {
        unsigned long long g = g_iq1s[(unsigned int)qs[l] | (((qh >> (3 * l)) & 7u) << 8)];
        for (int j = 0; j < 8; j++) code[l * 8 + j] = (signed char)(8 * gsb8(g, j) + sgn);
    }
}

__device__ __forceinline__ void wdec_iq1m(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 56;
    int ib = blk & 7;
    float d = iq1m_d(b);
    const unsigned char* sp = b + 48 + 2 * (ib >> 1);
    unsigned int scw = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8);
    int sh = 6 * (ib & 1);
    *s0 = iq1_dl(d, (int)((scw >> sh) & 7u)) * IQ1S_DELTA;
    *s1 = iq1_dl(d, (int)((scw >> (sh + 3)) & 7u)) * IQ1S_DELTA;
    const unsigned char* qs = b + ib * 4;
    const unsigned char* qh = b + 32 + ib * 2;
    for (int l = 0; l < 4; l++) {
        unsigned int qhb = qh[l >> 1];
        unsigned long long g = g_iq1s[(unsigned int)qs[l] | ((qhb << ((l & 1) ? 4 : 8)) & 0x700u)];
        int sgn = (qhb & ((l & 1) ? 0x80u : 0x08u)) ? -1 : 1;
        for (int j = 0; j < 8; j++) code[l * 8 + j] = (signed char)(8 * gsb8(g, j) + sgn);
    }
}

// R6 ternary: ONE `d` per block for all three, so `*s0 == *s1` always and the whole decode is the
// element unpack with the `−1` folded in.
__device__ __forceinline__ void wdec_tq10(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 54;
    float d = rf16b(b + 52);
    *s0 = d; *s1 = d;
    int base = (blk & 7) * 32;
    for (int p = 0; p < 32; p++) code[p] = (signed char)tq10_code(b, base + p);
}

__device__ __forceinline__ void wdec_tq20(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 3) + (blk >> 3)) * 66;
    float d = rf16b(b + 64);
    *s0 = d; *s1 = d;
    // A 32-element block is exactly one (chunk, shift) pair, so the 32 codes are 32 CONSECUTIVE
    // bytes read at one shift — the tidiest inner loop of any covered format.
    int w32 = blk & 7;
    const unsigned char* qs = b + (w32 >> 2) * 32;
    int sh = 2 * (w32 & 3);
    for (int p = 0; p < 32; p++) code[p] = (signed char)((int)((qs[p] >> sh) & 3u) - 1);
}

// Q2_0 is the ONE covered format whose block is not 32 or 256 elements: 64, i.e. TWO activation
// 32-blocks per header. So the block index is `blk>>1` (not `blk>>3`) and the half selects which
// 8 of the 16 `qs` bytes this 32-block owns.
__device__ __forceinline__ void wdec_q20(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 1) + (blk >> 1)) * 18;
    float d = rf16b(b);
    *s0 = d; *s1 = d;
    const unsigned char* qs = b + 2 + (blk & 1) * 8;
    for (int p = 0; p < 32; p++) {
        code[p] = (signed char)((int)((qs[p >> 2] >> (2 * (p & 3))) & 3u) - 1);
    }
}

// R7 fp4: the codebook value IS the signed code (R4's shape), so the only per-format work is the
// scale decode and the nibble walk. MXFP4's block IS the 32-element tile (`*s0 == *s1`, one E8M0);
// NVFP4's is 64 elements like Q2_0 — so `blk>>1` indexes the block and `blk&1` the half — but
// UNLIKE Q2_0 each half carries TWO scales, one per 16 elements, which is exactly what `s0`/`s1`
// are for. NVFP4 is the only covered format that uses BOTH the 64-element stride AND the split
// scale, so it is the one case where passing the same scale twice would be wrong.
__device__ __forceinline__ void wdec_mxfp4(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * nblk + blk) * 17;
    float d = e8m0_half(b[0]);
    *s0 = d; *s1 = d;
    const unsigned char* qs = b + 1;
    for (int p = 0; p < 16; p++) {
        code[p]      = (signed char)kv_mxfp4(qs[p] & 0x0F);
        code[p + 16] = (signed char)kv_mxfp4(qs[p] >> 4);
    }
}

__device__ __forceinline__ void wdec_nvfp4(
    const unsigned char* __restrict__ w, long col, int nblk, int blk,
    signed char* code, float* s0, float* s1) {
    const unsigned char* b = w + (col * (nblk >> 1) + (blk >> 1)) * 36;   // 36 B / 64 elements
    int half = blk & 1;                    // which 32-element half of the 64-element block
    *s0 = ue4m3(b[2 * half]);              // elements 0..15  → sub-block 2·half
    *s1 = ue4m3(b[2 * half + 1]);          // elements 16..31 → sub-block 2·half + 1
    const unsigned char* qs = b + 4 + half * 16;
    for (int u = 0; u < 2; u++) {
        const unsigned char* q = qs + u * 8;
        for (int j = 0; j < 8; j++) {
            code[u * 16 + j]     = (signed char)kv_mxfp4(q[j] & 0x0F);
            code[u * 16 + j + 8] = (signed char)kv_mxfp4(q[j] >> 4);
        }
    }
}

// ── K-quant 16-elem sub-block traversal (Q2_K / Q3_K) ────────────────────────
// Both formats walk their 256 elements as n(2) × j(4) × half(2) × l(16), emitting 16 consecutive
// output elements per (n, j, half) group. So sub-block `g = within/16` (0..15) decomposes as
//   n = g>>3, j = (g>>1)&3, half = g&1,
// its 16 codes live in the CONTIGUOUS bytes `qs[n*32 + half*16 .. +16]` as the 2-bit field at shift
// `2j`, and the host oracle's per-group scale index `is` advances in lockstep with `out/16`, i.e.
// `is == g`. Q3_K's rolling high-mask bit `m` (1<<0 .. 1<<7 across the whole traversal) is
// `1 << (n*4 + j)` — and `n*4 + j == g>>1`, since `half` is exactly the bit `g` drops.

// ── Q2_K: 256 elems / 84 bytes = [u8 scales[16]][u8 qs[64]][half d][half dmin]. ──
// scale = d·(scales[g] & 0xF), min = dmin·(−(scales[g] >> 4)), code = 2 bits (0..3).
__device__ __forceinline__ float deq_q2k(const unsigned char* w, long i) {
    long blk = i >> 8;             // / 256
    int within = (int)(i & 255);
    const unsigned char* b = w + blk * 84;
    const unsigned char* scales = b;
    const unsigned char* qs = b + 16;
    float d = rf16b(b + 80);
    float dmin = rf16b(b + 82);
    int g = within >> 4;           // 16-elem sub-block 0..15
    int l = within & 15;
    int sc = scales[g] & 0x0F;
    int mm = scales[g] >> 4;
    int code = (qs[(g >> 3) * 32 + (g & 1) * 16 + l] >> (2 * ((g >> 1) & 3))) & 3;
    return fin(d * (float)sc, code, dmin * (float)(-mm));
}

// Decode the 6-bit sub-block scale `g` (0..15) of a Q3_K block from its packed 12 `scales` bytes,
// already biased by −32. Closed form of llama.cpp's kmask1/kmask2 aux shuffle (the host oracle's
// `dequant_factored` Q3K arm): with `wd = g/4`, `k = g%4`, aux word `wd` byte `k` is
//   wd=0: (raw[k]   & 0x0F) | ((raw[8+k]      & 3) << 4)
//   wd=1: (raw[4+k] & 0x0F) | (((raw[8+k]>>2) & 3) << 4)
//   wd=2: (raw[k]   >>   4) | (((raw[8+k]>>4) & 3) << 4)
//   wd=3: (raw[4+k] >>   4) | (((raw[8+k]>>6) & 3) << 4)
// i.e. low byte source `raw[(wd&1)*4 + k]`, nibble half `wd>=2`, 2 top bits at shift `2*wd`.
__device__ __forceinline__ int q3k_sc6(const unsigned char* sr, int g) {
    int wd = g >> 2, k = g & 3;
    int lo = sr[(wd & 1) * 4 + k];
    int nib = (wd >= 2) ? (lo >> 4) : (lo & 0x0F);
    int top = (sr[8 + k] >> (2 * wd)) & 3;
    return (nib | (top << 4)) - 32;   // 6-bit value 0..63 → signed scale −32..31
}

// ── Q3_K: 256 elems / 110 bytes = [u8 hmask[32]][u8 qs[64]][u8 scales[12]][half d]. ──
// scale = d·sc6, min = d·(−4·sc6), code = low 2 bits | (high-mask bit << 2), 0..7. NOTE the high
// bit's polarity: the host oracle stores `hmask` SET as code +4 and folds the constant −4 into the
// min, which is llama.cpp's `q3 − (hm ? 0 : 4)` — a flipped bit moves the value by 4·d·sc6.
__device__ __forceinline__ float deq_q3k(const unsigned char* w, long i) {
    long blk = i >> 8;             // / 256
    int within = (int)(i & 255);
    const unsigned char* b = w + blk * 110;
    const unsigned char* hmask = b;
    const unsigned char* qs = b + 32;
    float d = rf16b(b + 108);
    int g = within >> 4;           // 16-elem sub-block 0..15
    int l = within & 15;
    int sc = q3k_sc6(b + 96, g);
    int p = (g & 1) * 16 + l;      // byte within the 32-byte half-plane (shared by qs and hmask)
    int code = ((qs[(g >> 3) * 32 + p] >> (2 * ((g >> 1) & 3))) & 3)
             | (((hmask[p] >> (g >> 1)) & 1) << 2);
    return fin(d * (float)sc, code, d * (float)(-4 * sc));
}

// get_scale_min_k4: 6-bit scale `sc` + min `mm` for sub-block s (0..8) of a Q4_K block.
__device__ __forceinline__ void k4(const unsigned char* q, int s, int* sc, int* mm) {
    if (s < 4) {
        *sc = q[s] & 63;
        *mm = q[s + 4] & 63;
    } else {
        *sc = (q[s + 4] & 0x0F) | ((q[s - 4] >> 6) << 4);
        *mm = (q[s + 4] >> 4)   | ((q[s]     >> 6) << 4);
    }
}

// ── Q4_K: 256 elems / 144 bytes = [half d][half dmin][u8 scales[12]][u8 qs[128]]. ──
// Element `within`'s 16/32-block scale index is `within/32` (0..7); the nibble comes
// from qs[(s/2)*32 + within%32], low nibble for even s, high nibble for odd s.
__device__ __forceinline__ float deq_q4k(const unsigned char* w, long i) {
    long blk = i >> 8;             // / 256
    int within = (int)(i & 255);
    const unsigned char* b = w + blk * 144;
    float d = rf16b(b);
    float dmin = rf16b(b + 2);
    const unsigned char* scales = b + 4;
    const unsigned char* qs = b + 16;
    int s = within >> 5;           // sub-block 0..7
    int sc, mm;
    k4(scales, s, &sc, &mm);
    int p = within & 31;
    int nib_base = (s >> 1) * 32 + p;
    int code = (s & 1) ? (qs[nib_base] >> 4) : (qs[nib_base] & 0x0F);
    return fin(d * (float)sc, code, dmin * (float)(-mm));
}

// ── Q5_K: 256 elems / 176 bytes = [half d][half dmin][u8 scales[12]][u8 qh[32]][u8 qs[128]]. ──
// Identical structure to Q4_K (same 6-bit `k4` scale/min per 32-elem sub-block, same nibble layout)
// plus a 5th code bit: element `within` takes bit `s = within/32` of `qh[within%32]` — the host
// oracle's `u1 = 1<<2j` / `u2 = 2<<2j` for sub-block pair `j`, which is exactly `1 << s` because the
// low-nibble half is s = 2j and the high-nibble half is s = 2j+1.
__device__ __forceinline__ float deq_q5k(const unsigned char* w, long i) {
    long blk = i >> 8;             // / 256
    int within = (int)(i & 255);
    const unsigned char* b = w + blk * 176;
    float d = rf16b(b);
    float dmin = rf16b(b + 2);
    const unsigned char* scales = b + 4;
    const unsigned char* qh = b + 16;
    const unsigned char* qs = b + 48;
    int s = within >> 5;           // sub-block 0..7
    int sc, mm;
    k4(scales, s, &sc, &mm);
    int p = within & 31;
    int nib_base = (s >> 1) * 32 + p;
    int code = (s & 1) ? (qs[nib_base] >> 4) : (qs[nib_base] & 0x0F);
    code |= ((qh[p] >> s) & 1) << 4;
    return fin(d * (float)sc, code, dmin * (float)(-mm));
}

// ── Q6_K: 256 elems / 210 bytes = [u8 ql[128]][u8 qh[64]][int8 scales[16]][half d]. ──
// Scale index is `within/16` (0..15); the 6-bit code = 4 low bits (ql) + 2 high bits (qh),
// with the ql/qh byte + shift chosen by the region `(within%128)/32`.
__device__ __forceinline__ float deq_q6k(const unsigned char* w, long i) {
    long blk = i >> 8;             // / 256
    int within = (int)(i & 255);
    const unsigned char* b = w + blk * 210;
    const unsigned char* ql = b;
    const unsigned char* qh = b + 128;
    const signed char* scales = (const signed char*)(b + 192);
    float d = rf16b(b + 208);
    int s = (int)scales[within >> 4];   // scale index = within / 16
    int half = within >> 7;             // / 128
    int o = within & 127;
    int region = o >> 5;                // 0..3
    int l = o & 31;
    int qlo = half * 64;
    int qho = half * 32;
    int code;
    if (region == 0)      code = (ql[qlo + l] & 0x0F)      | ((qh[qho + l] & 3) << 4);
    else if (region == 1) code = (ql[qlo + 32 + l] & 0x0F) | (((qh[qho + l] >> 2) & 3) << 4);
    else if (region == 2) code = (ql[qlo + l] >> 4)        | (((qh[qho + l] >> 4) & 3) << 4);
    else                  code = (ql[qlo + 32 + l] >> 4)   | (((qh[qho + l] >> 6) & 3) << 4);
    return fin(d * (float)s, code, d * (float)(-32 * s));
}

// GEMV `dst[m, out_f] = x[m, in_f] · decode(w)[out_f, in_f]ᵀ`. One block per m-row, threads
// stride over out_f (mirrors `linear_f16`); the weight buffer is pre-advanced past `w_off`
// on the host so element (o, i) is global index `o*in_f + i`. Accumulation is in i-order —
// identical to `linear_f16` — so the f32 sum is bit-stable against the cache path.
#define GEN_LINEAR(SUFFIX) \
extern "C" __global__ void linear_##SUFFIX( \
    const float* __restrict__ x, \
    const unsigned char* __restrict__ w, \
    float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int row = blockIdx.x; \
    int tid = threadIdx.x; \
    const float* xr = x + row * in_f; \
    for (int o = tid; o < out_f; o += blockDim.x) { \
        float acc = 0.0f; \
        long base = (long)o * in_f; \
        for (int i = 0; i < in_f; i++) { \
            acc += xr[i] * deq_##SUFFIX(w, base + i); \
        } \
        dst[row * out_f + o] = acc; \
    } \
}

// EmbedGather: `dst[r, :] = decode(table[ids[r], :]) * scale`. One thread per row.
#define GEN_EMBED(SUFFIX) \
extern "C" __global__ void embed_##SUFFIX( \
    const int* __restrict__ ids, \
    const unsigned char* __restrict__ table, \
    float* __restrict__ dst, \
    int rows, int dim, float scale) { \
    int row = blockIdx.x * blockDim.x + threadIdx.x; \
    if (row >= rows) return; \
    int id = ids[row]; \
    long base = (long)id * dim; \
    float* dr = dst + row * dim; \
    for (int i = 0; i < dim; i++) { \
        dr[i] = deq_##SUFFIX(table, base + i) * scale; \
    } \
}

GEN_LINEAR(q80)
GEN_LINEAR(q2k)
GEN_LINEAR(q3k)
GEN_LINEAR(q4k)
GEN_LINEAR(q5k)
GEN_LINEAR(q6k)
GEN_LINEAR(q50)
GEN_LINEAR(q40)
GEN_LINEAR(q41)
GEN_LINEAR(q51)
GEN_LINEAR(iq4nl)
GEN_LINEAR(iq4xs)
GEN_LINEAR(iq2xxs)
GEN_LINEAR(iq2xs)
GEN_LINEAR(iq2s)
GEN_LINEAR(iq3xxs)
GEN_LINEAR(iq3s)
GEN_LINEAR(iq1s)
GEN_LINEAR(iq1m)
GEN_LINEAR(tq10)
GEN_LINEAR(tq20)
GEN_LINEAR(q20)
GEN_LINEAR(mxfp4)
GEN_LINEAR(nvfp4)
GEN_EMBED(q80)
GEN_EMBED(q2k)
GEN_EMBED(q3k)
GEN_EMBED(q4k)
GEN_EMBED(q5k)
GEN_EMBED(q6k)
GEN_EMBED(q50)
GEN_EMBED(q40)
GEN_EMBED(q41)
GEN_EMBED(q51)
GEN_EMBED(iq4nl)
GEN_EMBED(iq4xs)
GEN_EMBED(iq2xxs)
GEN_EMBED(iq2xs)
GEN_EMBED(iq2s)
GEN_EMBED(iq3xxs)
GEN_EMBED(iq3s)
GEN_EMBED(iq1s)
GEN_EMBED(iq1m)
GEN_EMBED(tq10)
GEN_EMBED(tq20)
GEN_EMBED(q20)
GEN_EMBED(mxfp4)
GEN_EMBED(nvfp4)
"#;

// ── Dequant-to-f16 + activation cast (Slice 26, rocBLAS f16 prefill GEMM) ─────
//
// The library f16 GEMM needs both operands materialized as f16: the quant weight is dequantized
// to a transient f16 buffer (`deqf16_*`) and the f32 activation is cast to f16 (`cast_f32_f16`),
// both drawn from the exec scratch pool (freed after the GEMM — NOT a permanent per-model cache).
// `deq_*` (NATIVE_DECODE, assembled before this part) already rounds through f16, so the stored
// __half is the exact value the retired dequant→f16 weight cache produced.
const DEQUANT_F16: &str = r#"
#define GEN_DEQF16(SUFFIX) \
extern "C" __global__ void deqf16_##SUFFIX( \
    const unsigned char* __restrict__ w, \
    __half* __restrict__ out, \
    int n) { \
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < (long)n) out[i] = __float2half(deq_##SUFFIX(w, i)); \
}
GEN_DEQF16(q80)
GEN_DEQF16(q2k)
GEN_DEQF16(q3k)
GEN_DEQF16(q4k)
GEN_DEQF16(q5k)
GEN_DEQF16(q6k)
GEN_DEQF16(q50)
GEN_DEQF16(q40)
GEN_DEQF16(q41)
GEN_DEQF16(q51)
GEN_DEQF16(iq4nl)
GEN_DEQF16(iq4xs)
GEN_DEQF16(iq2xxs)
GEN_DEQF16(iq2xs)
GEN_DEQF16(iq2s)
GEN_DEQF16(iq3xxs)
GEN_DEQF16(iq3s)
GEN_DEQF16(iq1s)
GEN_DEQF16(iq1m)
GEN_DEQF16(tq10)
GEN_DEQF16(tq20)
GEN_DEQF16(q20)
GEN_DEQF16(mxfp4)
GEN_DEQF16(nvfp4)

extern "C" __global__ void cast_f32_f16(
    const float* __restrict__ x, __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}
"#;

// ── Native-decode MoE expert FFN (Phase-3 for MoE) ───────────────────────────
//
// Twin of `moe_ffn_expert`, but the gate/up/down expert banks are the RAW quant bytes decoded
// in-kernel via `deq_*` (defined in NATIVE_DECODE, assembled before this part) — NO f16 cache
// is materialized, so a big quantized MoE fits in VRAM (footprint ≈ quant size). Bit-faithful to
// the old f16-cache path (each `deq_*` rounds through f16, same as the cache did), so it tracks
// `moe_ffn_expert` within f16 rounding.
//
// Gate & up share ONE format (`GU`) — they are the same tensor when fused, and every GGUF stores
// ffn_gate_exps / ffn_up_exps at the same quant type. The down bank has its OWN format (`DN`):
// Q4_K_M packs gate/up as Q4_K but ffn_down_exps as Q6_K, so the two suffixes must be independent.
// This cross product is NOT complete over `moe_native_fmt` (11 formats would be 121 pairs per
// macro): R3 measured the full expansion at +1.1 s of COLD hiprtc and took R2's escape hatch, so
// only the pairs a real GGUF can actually produce are instantiated — see `moe_expert_kernel` in
// exec.rs for the reachability argument, and `MOE_EXPERT_PAIRS` (exec.rs test module) for the exact
// set, which `moe_expert_pair_tables_agree` pins against both mappers. An absent pair is not a bug:
// the `MoeFfn` arm drops `native` and the expert takes the dequant→f16 `moe_ffn_expert` fallback,
// which is exactly the comparand of the `INFR_ROCM_NO_I8` switch these kernels serve. Uncovered
// FORMATS take that same fallback.
//
// Pointers are pre-advanced HOST-SIDE to this expert's block-aligned byte offset (see the MoeFfn
// arm), so the in-kernel element index is relative to the expert's own bank — identical geometry
// to the f16 kernel, just a quant-byte decode instead of an `__half2float` load.
const MOE_FFN_NATIVE: &str = r#"
#define GEN_MOE_FFN(GU, DN) \
extern "C" __global__ void moe_ffn_expert_##GU##_##DN( \
    const float* __restrict__ x,             /* [ne] — input row */ \
    const unsigned char* __restrict__ gate_w,/* raw GU bytes, [n_ff_exp, ne] (pre-advanced) */ \
    const unsigned char* __restrict__ up_w,  /* raw GU bytes, [n_ff_exp, ne] (pre-advanced) */ \
    const unsigned char* __restrict__ down_w,/* raw DN bytes, [ne, n_ff_exp] (pre-advanced) */ \
    float* __restrict__ dst,                 /* [ne] — accumulated * weight */ \
    int ne, \
    int n_ff_exp, \
    int act_type,   /* 0=SiLU, 1=GeLU, 2=Sigmoid */ \
    float weight,   /* routing weight for this expert */ \
    float down_scale, /* per-expert down-projection output scale (1 = no scale) */ \
    int weight_before /* 1 = apply `weight` to the gate/up inputs (llama4); 0 = to the output */ \
) { \
    int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < (int)n_ff_exp) { \
        float wg = weight_before ? weight : 1.0f; \
        float wo = weight_before ? 1.0f : weight; \
        float g = 0.0f, u = 0.0f; \
        for (int j = 0; j < (int)ne; j++) { \
            long idx = (long)i * ne + j; \
            g += x[j] * deq_##GU(gate_w, idx); \
            u += x[j] * deq_##GU(up_w, idx); \
        } \
        g *= wg; \
        u *= wg; \
        float a; \
        if (act_type == 0) { \
            a = g / (1.0f + expf(-g)); \
        } else if (act_type == 1) { \
            float x3 = g * g * g; \
            a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3))); \
        } else { \
            a = 1.0f / (1.0f + expf(-g)); \
        } \
        float h = a * u * wo * down_scale; \
        for (int d = 0; d < (int)ne; d++) { \
            atomicAdd(&dst[d], h * deq_##DN(down_w, (long)d * n_ff_exp + i)); \
        } \
    } \
}

GEN_MOE_FFN(q80, q80)
GEN_MOE_FFN(q80, q2k)
GEN_MOE_FFN(q80, q3k)
GEN_MOE_FFN(q80, q4k)
GEN_MOE_FFN(q80, q5k)
GEN_MOE_FFN(q80, q6k)
GEN_MOE_FFN(q2k, q80)
GEN_MOE_FFN(q2k, q2k)
GEN_MOE_FFN(q2k, q3k)
GEN_MOE_FFN(q2k, q4k)
GEN_MOE_FFN(q2k, q5k)
GEN_MOE_FFN(q2k, q6k)
GEN_MOE_FFN(q3k, q80)
GEN_MOE_FFN(q3k, q2k)
GEN_MOE_FFN(q3k, q3k)
GEN_MOE_FFN(q3k, q4k)
GEN_MOE_FFN(q3k, q5k)
GEN_MOE_FFN(q3k, q6k)
GEN_MOE_FFN(q4k, q80)
GEN_MOE_FFN(q4k, q2k)
GEN_MOE_FFN(q4k, q3k)
GEN_MOE_FFN(q4k, q4k)
GEN_MOE_FFN(q4k, q5k)
GEN_MOE_FFN(q4k, q6k)
GEN_MOE_FFN(q5k, q80)
GEN_MOE_FFN(q5k, q2k)
GEN_MOE_FFN(q5k, q3k)
GEN_MOE_FFN(q5k, q4k)
GEN_MOE_FFN(q5k, q5k)
GEN_MOE_FFN(q5k, q6k)
GEN_MOE_FFN(q6k, q80)
GEN_MOE_FFN(q6k, q2k)
GEN_MOE_FFN(q6k, q3k)
GEN_MOE_FFN(q6k, q4k)
GEN_MOE_FFN(q6k, q5k)
GEN_MOE_FFN(q6k, q6k)
GEN_MOE_FFN(q40, q40)
GEN_MOE_FFN(q40, q41)
GEN_MOE_FFN(q40, q51)
GEN_MOE_FFN(q40, q80)
GEN_MOE_FFN(q41, q40)
GEN_MOE_FFN(q41, q41)
GEN_MOE_FFN(q41, q51)
GEN_MOE_FFN(q41, q80)
GEN_MOE_FFN(q51, q40)
GEN_MOE_FFN(q51, q41)
GEN_MOE_FFN(q51, q51)
GEN_MOE_FFN(q51, q80)
GEN_MOE_FFN(iq4nl, iq4nl)
GEN_MOE_FFN(iq4nl, iq4xs)
GEN_MOE_FFN(iq4nl, q4k)
GEN_MOE_FFN(iq4nl, q5k)
GEN_MOE_FFN(iq4nl, q6k)
GEN_MOE_FFN(iq4nl, q80)
GEN_MOE_FFN(iq4xs, iq4nl)
GEN_MOE_FFN(iq4xs, iq4xs)
GEN_MOE_FFN(iq4xs, q4k)
GEN_MOE_FFN(iq4xs, q5k)
GEN_MOE_FFN(iq4xs, q6k)
GEN_MOE_FFN(iq4xs, q80)
GEN_MOE_FFN(q2k, iq4nl)
GEN_MOE_FFN(q3k, iq4nl)
GEN_MOE_FFN(iq2xxs, iq2s)
GEN_MOE_FFN(iq2xxs, iq3xxs)
GEN_MOE_FFN(iq2xxs, iq3s)
GEN_MOE_FFN(iq2xxs, iq4nl)
GEN_MOE_FFN(iq2xxs, iq4xs)
GEN_MOE_FFN(iq2xxs, q4k)
GEN_MOE_FFN(iq2xxs, q6k)
GEN_MOE_FFN(iq2xs, iq2s)
GEN_MOE_FFN(iq2xs, iq3xxs)
GEN_MOE_FFN(iq2xs, iq3s)
GEN_MOE_FFN(iq2xs, iq4nl)
GEN_MOE_FFN(iq2xs, iq4xs)
GEN_MOE_FFN(iq2xs, q4k)
GEN_MOE_FFN(iq2xs, q6k)
GEN_MOE_FFN(iq2s, iq2s)
GEN_MOE_FFN(iq2s, iq3xxs)
GEN_MOE_FFN(iq2s, iq3s)
GEN_MOE_FFN(iq2s, iq4nl)
GEN_MOE_FFN(iq2s, iq4xs)
GEN_MOE_FFN(iq2s, q4k)
GEN_MOE_FFN(iq2s, q6k)
GEN_MOE_FFN(iq3xxs, iq2s)
GEN_MOE_FFN(iq3xxs, iq3xxs)
GEN_MOE_FFN(iq3xxs, iq3s)
GEN_MOE_FFN(iq3xxs, iq4nl)
GEN_MOE_FFN(iq3xxs, iq4xs)
GEN_MOE_FFN(iq3xxs, q4k)
GEN_MOE_FFN(iq3xxs, q6k)
GEN_MOE_FFN(iq3s, iq2s)
GEN_MOE_FFN(iq3s, iq3xxs)
GEN_MOE_FFN(iq3s, iq3s)
GEN_MOE_FFN(iq3s, iq4nl)
GEN_MOE_FFN(iq3s, iq4xs)
GEN_MOE_FFN(iq3s, q4k)
GEN_MOE_FFN(iq3s, q6k)
GEN_MOE_FFN(iq1s, iq1s)
GEN_MOE_FFN(iq1s, iq1m)
GEN_MOE_FFN(iq1s, iq2xxs)
GEN_MOE_FFN(iq1s, iq2s)
GEN_MOE_FFN(iq1s, iq3s)
GEN_MOE_FFN(iq1s, iq4xs)
GEN_MOE_FFN(iq1s, q4k)
GEN_MOE_FFN(iq1s, q6k)
GEN_MOE_FFN(iq1m, iq1s)
GEN_MOE_FFN(iq1m, iq1m)
GEN_MOE_FFN(iq1m, iq2xxs)
GEN_MOE_FFN(iq1m, iq2s)
GEN_MOE_FFN(iq1m, iq3s)
GEN_MOE_FFN(iq1m, iq4xs)
GEN_MOE_FFN(iq1m, q4k)
GEN_MOE_FFN(iq1m, q6k)
GEN_MOE_FFN(tq10, tq10)
GEN_MOE_FFN(tq20, tq20)
GEN_MOE_FFN(q20, q20)
GEN_MOE_FFN(mxfp4, mxfp4)
GEN_MOE_FFN(nvfp4, nvfp4)
"#;

// ── Int8-activation dp4a decode GEMV (Phase 4) ───────────────────────────────
//
// The Phase-3 NATIVE_DECODE GEMV above is bit-faithful to the old f16 cache, but it pays a
// per-element f16 round-trip (`__half2float(__float2half(...))`) inside the hot dot loop — pure
// ALU that made small-model decode ALU-bound (regressed to ~1.9 t/s vs the old f16-cache ~4.5).
//
// This path drops the f16 round-trip entirely: the activation row is quantized to int8 ONCE (per
// 32-elem block, scale = amax/127), then integer-dotted against the decoded weight codes via
// `__builtin_amdgcn_sdot4` (V_DOT4_I32_I8 on gfx1100 — 4 signed int8 MACs / instruction). The
// per-block weight scale (and the Q4_K/Q6_K min) is applied to the int32 accumulator AFTER the
// integer dot — the "scale-after is free" mmq principle (each lane owns its own accumulator), the
// same reasoning as Vulkan's dp4a `mmq` and the CPU VNNI dots. This is a SANCTIONED PRECISION FLIP:
// int8 activation quantization is lossy, so the output differs (within tolerance) from the
// bit-faithful f16 path — the ROCm goldens are re-blessed after a coherence check (docs/perf.md).
//
// Grid: one block per (output-row `o`, m-row `row`); block = 32 threads (one RDNA3 wave32). The 32
// threads stride over the input's 32-elem blocks, each accumulates an f32 partial, then a wave
// shuffle reduces to lane 0. The int8 activation is quantized once per row (`quant_i8_32`) and
// REUSED across all `out_f` output rows AND — for m>1 (the `mrow` analogue) — the single quant pass
// covers every row, so the activation quant cost amortizes over the whole GEMV.
//
// Covered formats: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS, the
// R5 grid quants IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, and the R6 IQ1 quants IQ1_S, IQ1_M plus
// ternary quants TQ1_0, TQ2_0, Q2_0. `rf16b`/`k4`/`q3k_sc6`/`wdec_*` live in NATIVE_DECODE,
// `kv_iq4nl` and the grids in the generated parts (all assembled before this one). Uncovered
// formats (MXFP4, NVFP4) keep the dequant→f16 fallback.
//
// The affine formats fold their offset into a SECOND integer dot against an all-ones B operand
// (`isum`), weighted by the block's min. Everything on the `wdec_*` seam — the R4 codebook formats
// (IQ4_NL/IQ4_XS), the R5 grid formats, and R6's IQ1/ternary formats — carries NO `isum` term: the
// decoded code is already the signed weight (see the `wdec_*` header for the per-family reason,
// including IQ1's ×8 delta fold and ternary's folded `−1`).
const INT8_DECODE: &str = r#"
// Quantize x[m, in_f] to int8 qx[m, in_f] with a per-32-block scale xs[m, in_f/32].
// scale = amax/127 (llama.cpp/GPU convention: `roundf`, half-away-from-zero). One thread / 32-block.
extern "C" __global__ void quant_i8_32(
    const float* __restrict__ x,
    signed char* __restrict__ qx,
    float* __restrict__ xs,
    int m,
    int in_f
) {
    int nblk = m * (in_f >> 5);
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= nblk) return;
    const float* xr = x + (long)blk * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) { float a = fabsf(xr[j]); if (a > amax) amax = a; }
    float s = amax / 127.0f;
    float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    signed char* qr = qx + (long)blk * 32;
    for (int j = 0; j < 32; j++) {
        float v = roundf(xr[j] * inv);
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        qr[j] = (signed char)v;
    }
    xs[blk] = s;
}

// Reduce an f32 partial across a 32-lane wave to lane 0 (reads only higher, always-active lanes).
static __device__ __forceinline__ float wave_sum32(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
    return v;
}

// 4×int8 signed dot-accumulate: `c + Σ a.i8[k]·b.i8[k]` — the V_DOT4_I32_I8 (dp4a) primitive.
// The natural spelling is `__builtin_amdgcn_sdot4`, but that builtin requires the `dot1-insts` target
// feature, which hiprtc does NOT reliably enable for gfx1100: comgr's per-process DEFAULT feature set
// is nondeterministic — the SAME source + `--gpu-architecture=gfx1100` compiles WITH the dot feature
// in one process and WITHOUT it in another (observed: parity test process has it, the model/seam
// process does not), and the builtin then fails to codegen ("needs target feature dot1-insts").
// Forcing the feature on per-function via a `target` attribute either mangles the extern-"C" kernel
// symbol (hipModuleGetFunction not-found) or miscompiles the cross-feature call (runtime garbage).
// So this uses the portable scalar idiom below: it compiles in EVERY process, is bit-stable, and clang
// still lowers it to V_DOT4 when the module happens to have the dot feature. The decode win comes from
// dropping the Phase-3 per-element f16 round-trip (the ALU-bound cost), not the single instruction.
static __device__ __forceinline__ int idot4(int a, int b, int c) {
    // Extract each 8-bit lane with a right-shift + `signed char` cast (well-defined sign extension;
    // a signed LEFT-shift into the sign bit would be UB and the optimizer miscompiles it), then MAC.
    // clang lowers this idiom to V_DOT4_I32_I8 when the module's target features include the dot
    // instructions, and to plain integer MADs otherwise — either way the Phase-3 per-element f16
    // round-trip (the ALU-bound cost) is gone.
    for (int k = 0; k < 4; k++) {
        int av = (int)(signed char)(a >> (k * 8));
        int bv = (int)(signed char)(b >> (k * 8));
        c += av * bv;
    }
    return c;
}

// ── F4: 128-bit weight/activation loads for the Q4_K/Q5_K decode GEMV ────────────────────────────
//
// The decode GEMV is NOT bandwidth bound (it was measuring ~50 GB/s of a ~960 GB/s bus) — it is
// VMEM-ISSUE bound. Reading a 32-byte nibble plane one byte at a time costs 32 `global_load_ubyte`
// instructions where two `global_load_dwordx4` would do, and the memory pipe runs out of issue slots
// long before it runs out of bandwidth. The helpers below fetch a whole 16-byte quad at a time.
//
// ALIGNMENT — the reason this covers Q4_K and Q5_K and nothing else. A `uint4` load needs 16-byte
// alignment or it is at best split back into byte loads and at worst wrong. Every weight pointer a
// decode GEMV sees is `hipMalloc` base (256-byte aligned; `BufferPool`/`RocmBuffer` never
// sub-allocate) + a whole number of BLOCKS: the dense path adds `(w_off/qpb)*bpb` with `w_off` a
// multiple of `qpb`, the MoE path adds `expert*(rows*in_f/qpb)*bpb`, and the paged/staged tiers add
// `slot*slot_bytes` with `slot_bytes` rounded up to 256 (see `pager.rs` / `weight_pager.rs`). So a
// block base is 16-byte aligned exactly when `bpb % 16 == 0`, and of the 24 GGUF block strides
// (18/20/22/24/34/84/110/136/144/176/210/...) only Q4_K's 144 and Q5_K's 176 qualify. Within a
// block, both formats put their 16-byte header at offset 0 and their nibble planes at a multiple of
// 16 (Q4_K: qs at +16; Q5_K: qh at +16, qs at +48), so the sub-block quads are aligned too. The
// int8 activation row is `hipMalloc` base + `row*in_f` + `blk*32` bytes, both multiples of 32.
//
// Bit-faithfulness: these are pure re-spellings of the same bytes in the same order — `f16q_lo/hi`
// assemble the identical `unsigned short` `rf16b` did, `k4q` is `k4` indexed into the header quad,
// and `(wd >> sh) & 0x0F0F0F0F` is the identical per-byte nibble `q[r] >> 4` / `q[r] & 0xF`. The
// dot order and the f32 reassociation are untouched, so the goldens do not move.

// ── F4 mrow: output rows computed per wave by the converted decode GEMVs ─────────────────────────
//
// One wave used to own ONE output row, so a lane had exactly ONE weight stream in flight and
// nothing to overlap its latency with. `I8_MROW` consecutive rows per wave give it that many
// independent streams (the activation quad is fetched once and shared), which is what actually
// moves the bus: on the 1024x151936 lm_head shape Q4_K goes 772 → 930 GB/s of a 960 GB/s peak, and
// on 1024x3072 it goes 95 → 180 GB/s. Measured 2 > 4 > 8 on gfx1100 (VGPR pressure) — the sweep and
// the formats this LOST on (Q6_K) are in docs/rocm-plan.md §9.
//
// HOST MIRROR: `exec.rs::i8_gemv_mrow` must return this for every kernel that uses it — the launch
// grid is `out_f / I8_MROW` blocks, and a mismatch skips or double-writes output rows silently.
#define I8_MROW 2

__device__ __forceinline__ float f16q_lo(unsigned int wd) {
    union { unsigned short u; __half h; } cvt;
    cvt.u = (unsigned short)(wd & 0xFFFFu);
    return __half2float(cvt.h);
}
__device__ __forceinline__ float f16q_hi(unsigned int wd) {
    union { unsigned short u; __half h; } cvt;
    cvt.u = (unsigned short)(wd >> 16);
    return __half2float(cvt.h);
}
// Byte `i` (0..3) of a little-endian dword — the `q[...]` the byte-wise decoders indexed.
__device__ __forceinline__ int qb4(unsigned int wd, int i) {
    return (int)((wd >> (8 * i)) & 0xFFu);
}
// `get_scale_min_k4` read straight out of the Q4_K/Q5_K header quad: `h.y|h.z|h.w` ARE
// `scales[0..12]` (bytes 4..16 of the block), so `k4(b+4, s)` becomes three register extracts.
__device__ __forceinline__ void k4q(uint4 h, int s, int* sc, int* mm) {
    if (s < 4) {
        *sc = qb4(h.y, s) & 63;          // scales[s]
        *mm = qb4(h.z, s) & 63;          // scales[s+4]
    } else {
        int a = qb4(h.w, s - 4);         // scales[s+4]
        *sc = (a & 0x0F) | ((qb4(h.y, s - 4) >> 6) << 4);   // | scales[s-4] high 2
        *mm = (a >> 4)   | ((qb4(h.z, s - 4) >> 6) << 4);   // | scales[s]   high 2
    }
}

// ── Q8_0: 32 elems / 34 bytes = [half d][int8 qs[32]]; value = d * qs (signed int8). ──
extern "C" __global__ void linear_i8_q80(
    const signed char* __restrict__ qx,   // [m, in_f]
    const float* __restrict__ xs,          // [m, in_f/32]
    const unsigned char* __restrict__ w,   // raw Q8_0 weight bytes (pre-advanced past w_off)
    float* __restrict__ dst,               // [m, out_f]
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 34;
        float d = rf16b(b);
        const unsigned char* wq = b + 2;   // 32 signed int8 codes
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0;
        for (int k = 0; k < 8; k++) {
            const unsigned char* q = wq + k * 4;
            int wpack = (int)q[0] | ((int)q[1] << 8) | ((int)q[2] << 16) | ((int)q[3] << 24);
            idot = idot4(xp[k], wpack, idot);
        }
        acc += d * xsr[blk] * (float)idot;
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── Q2_K: 256 elems / 84 bytes; sub-block 16; code 0..3; value = d·sc·code + dmin·(−mm). ──
// A 32-elem activation block spans TWO 16-elem scale sub-blocks (same shape as Q6_K), so the inner
// loop runs the dp4a over 4 int-packed groups per half. Sub-block indexing per `deq_q2k`.
extern "C" __global__ void linear_i8_q2k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    int spr = nb >> 3;             // Q2_K super-blocks (256 elems) per output row
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int w32 = blk & 7;
        const unsigned char* b = w + (long)super * 84;
        const unsigned char* scales = b;
        const unsigned char* qs = b + 16;
        float d = rf16b(b + 80);
        float dmin = rf16b(b + 82);
        float sx = xsr[blk];
        for (int hh = 0; hh < 2; hh++) {
            int g = w32 * 2 + hh;          // 16-elem sub-block 0..15
            int sc = scales[g] & 0x0F;
            int mm = scales[g] >> 4;
            const unsigned char* qb = qs + (g >> 3) * 32 + (g & 1) * 16;
            int sh = 2 * ((g >> 1) & 3);
            const int* xp = (const int*)(qxr + blk * 32 + hh * 16);
            int idot = 0, isum = 0;
            for (int k = 0; k < 4; k++) {  // 4 groups of 4 = 16
                int wpack = 0;
                for (int r = 0; r < 4; r++) wpack |= (int)((qb[k * 4 + r] >> sh) & 3) << (r * 8);
                idot = idot4(xp[k], wpack, idot);
                isum = idot4(xp[k], 0x01010101, isum);
            }
            acc += (d * (float)sc) * sx * (float)idot + (dmin * (float)(-mm)) * sx * (float)isum;
        }
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── Q3_K: 256 elems / 110 bytes; sub-block 16 (6-bit packed scale); code 0..7; ──
// value = d·sc6·code + d·(−4·sc6). Same two-halves-per-32-block shape as Q2_K/Q6_K.
extern "C" __global__ void linear_i8_q3k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    int spr = nb >> 3;             // Q3_K super-blocks (256 elems) per output row
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int w32 = blk & 7;
        const unsigned char* b = w + (long)super * 110;
        const unsigned char* hmask = b;
        const unsigned char* qs = b + 32;
        float d = rf16b(b + 108);
        float sx = xsr[blk];
        for (int hh = 0; hh < 2; hh++) {
            int g = w32 * 2 + hh;
            int sc = q3k_sc6(b + 96, g);
            const unsigned char* qb = qs + (g >> 3) * 32 + (g & 1) * 16;
            const unsigned char* hb = hmask + (g & 1) * 16;
            int sh = 2 * ((g >> 1) & 3);
            int hsh = g >> 1;
            const int* xp = (const int*)(qxr + blk * 32 + hh * 16);
            int idot = 0, isum = 0;
            for (int k = 0; k < 4; k++) {
                int wpack = 0;
                for (int r = 0; r < 4; r++) {
                    int p = k * 4 + r;
                    int c = ((qb[p] >> sh) & 3) | (((hb[p] >> hsh) & 1) << 2);
                    wpack |= c << (r * 8);
                }
                idot = idot4(xp[k], wpack, idot);
                isum = idot4(xp[k], 0x01010101, isum);
            }
            acc += (d * (float)sc) * sx * (float)idot + (d * (float)(-4 * sc)) * sx * (float)isum;
        }
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── Q4_K: 256 elems / 144 bytes; sub-block 32; code 0..15; value = d·sc·code + dmin·(−mm). ──
extern "C" __global__ void linear_i8_q4k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o0 = blockIdx.x * I8_MROW, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    int spr = nb >> 3;             // Q4_K super-blocks (256 elems) per output row
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    int ov[I8_MROW];
    float accv[I8_MROW];
    #pragma unroll
    for (int r = 0; r < I8_MROW; r++) {
        ov[r] = (o0 + r < out_f) ? (o0 + r) : o0;   // clamp, not branch: keep the streams uniform
        accv[r] = 0.0f;
    }
    for (int blk = tid; blk < nb; blk += 32) {
        int s = blk & 7;           // sub-block 0..7 (== the 32-block)
        unsigned int sh = (unsigned int)(s & 1) * 4u;      // high nibble for odd sub-blocks
        // The activation quad is fetched ONCE and dotted against all I8_MROW weight rows.
        const int4* xq = (const int4*)(qxr + blk * 32);
        int4 xlo = xq[0], xhi = xq[1];
        int xv[8] = { xlo.x, xlo.y, xlo.z, xlo.w, xhi.x, xhi.y, xhi.z, xhi.w };
        // F4: three 128-bit loads per row replace 36 byte loads. `144 % 16 == 0` ⇒ the block base
        // is 16-byte aligned, and `qs` sits at +16 ⇒ so is each 32-byte nibble plane. Issuing all
        // I8_MROW rows' loads before any of the math is what gives the pipe something to overlap.
        uint4 hdr[I8_MROW], wlo[I8_MROW], whi[I8_MROW];
        #pragma unroll
        for (int r = 0; r < I8_MROW; r++) {
            const uint4* bq = (const uint4*)(w + ((long)ov[r] * spr + (blk >> 3)) * 144);
            hdr[r] = bq[0];                                // [d][dmin][scales[12]]
            const uint4* qq = bq + 1 + (s >> 1) * 2;       // qs + (s/2)*32, the nibble plane
            wlo[r] = qq[0];
            whi[r] = qq[1];
        }
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) isum = idot4(xv[k], 0x01010101, isum);
        float sx = xsr[blk];
        #pragma unroll
        for (int r = 0; r < I8_MROW; r++) {
            unsigned int wv[8] = { wlo[r].x, wlo[r].y, wlo[r].z, wlo[r].w,
                                   whi[r].x, whi[r].y, whi[r].z, whi[r].w };
            int idot = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                idot = idot4(xv[k], (int)((wv[k] >> sh) & 0x0F0F0F0Fu), idot);
            }
            int sc, mm; k4q(hdr[r], s, &sc, &mm);
            float d = f16q_lo(hdr[r].x), dmin = f16q_hi(hdr[r].x);
            accv[r] += (d * (float)sc) * sx * (float)idot + (dmin * (float)(-mm)) * sx * (float)isum;
        }
    }
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    #pragma unroll
    for (int r = 0; r < I8_MROW; r++) {
        float a = wave_sum32(accv[r]);
        if (tid == 0 && (r == 0 || ov[r] != o0)) {
            long oi = (long)row * out_f + ov[r];
            dst[oi] = resid ? (a + resid[oi]) : a;
        }
    }
}

// ── Q5_K: 256 elems / 176 bytes; sub-block 32; code 0..31 (nibble + qh bit `s`); Q4_K scale/min. ──
extern "C" __global__ void linear_i8_q5k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o0 = blockIdx.x * I8_MROW, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    int spr = nb >> 3;             // Q5_K super-blocks (256 elems) per output row
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    int ov[I8_MROW];
    float accv[I8_MROW];
    #pragma unroll
    for (int r = 0; r < I8_MROW; r++) {
        ov[r] = (o0 + r < out_f) ? (o0 + r) : o0;   // clamp, not branch: keep the streams uniform
        accv[r] = 0.0f;
    }
    for (int blk = tid; blk < nb; blk += 32) {
        int s = blk & 7;           // sub-block 0..7 (== the 32-block) == the qh bit index
        unsigned int sh = (unsigned int)(s & 1) * 4u;      // high nibble for odd sub-blocks
        const int4* xq = (const int4*)(qxr + blk * 32);
        int4 xlo = xq[0], xhi = xq[1];
        int xv[8] = { xlo.x, xlo.y, xlo.z, xlo.w, xhi.x, xhi.y, xhi.z, xhi.w };
        // F4: five 128-bit loads per row replace ~71 byte loads. `176 % 16 == 0` ⇒ the block base
        // is 16-byte aligned, and qh (+16) / qs (+48) keep every plane on a 16-byte boundary.
        uint4 hdr[I8_MROW], qh0[I8_MROW], qh1[I8_MROW], wlo[I8_MROW], whi[I8_MROW];
        #pragma unroll
        for (int r = 0; r < I8_MROW; r++) {
            const uint4* bq = (const uint4*)(w + ((long)ov[r] * spr + (blk >> 3)) * 176);
            hdr[r] = bq[0];                                // [d][dmin][scales[12]]
            qh0[r] = bq[1];                                // qh[0..32) at b+16
            qh1[r] = bq[2];
            const uint4* qq = bq + 3 + (s >> 1) * 2;       // qs (b+48) + (s/2)*32
            wlo[r] = qq[0];
            whi[r] = qq[1];
        }
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) isum = idot4(xv[k], 0x01010101, isum);
        float sx = xsr[blk];
        #pragma unroll
        for (int r = 0; r < I8_MROW; r++) {
            unsigned int wv[8] = { wlo[r].x, wlo[r].y, wlo[r].z, wlo[r].w,
                                   whi[r].x, whi[r].y, whi[r].z, whi[r].w };
            unsigned int hv[8] = { qh0[r].x, qh0[r].y, qh0[r].z, qh0[r].w,
                                   qh1[r].x, qh1[r].y, qh1[r].z, qh1[r].w };
            int idot = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                // Per byte: nibble | (bit `s` of qh[p] << 4). `s <= 7`, so the dword shift can only
                // pull bits from the NEXT byte into positions >= 1 — bit 0 of each byte is exactly
                // `(qh[p] >> s) & 1`, which is what the 0x01010101 mask keeps.
                unsigned int wpack = ((wv[k] >> sh) & 0x0F0F0F0Fu)
                                   | (((hv[k] >> s) & 0x01010101u) << 4);
                idot = idot4(xv[k], (int)wpack, idot);
            }
            int sc, mm; k4q(hdr[r], s, &sc, &mm);
            float d = f16q_lo(hdr[r].x), dmin = f16q_hi(hdr[r].x);
            accv[r] += (d * (float)sc) * sx * (float)idot + (dmin * (float)(-mm)) * sx * (float)isum;
        }
    }
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    #pragma unroll
    for (int r = 0; r < I8_MROW; r++) {
        float a = wave_sum32(accv[r]);
        if (tid == 0 && (r == 0 || ov[r] != o0)) {
            long oi = (long)row * out_f + ov[r];
            dst[oi] = resid ? (a + resid[oi]) : a;
        }
    }
}

// ── Q6_K: 256 elems / 210 bytes; sub-block 16 (int8 scale); code 0..63; value = d·s·code + d·(−32s). ──
extern "C" __global__ void linear_i8_q6k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    // F4 measured Q6_K on the mrow path too and it LOST — 149 → 145 GB/s on the lm_head shape and
    // −25% on the projection shapes. Q6_K's decode is the register-hungriest of the family (two
    // 16-element sub-blocks, a 4-way region select, a 6-bit reassembly per code), so a second row's
    // live state costs more occupancy than the extra memory stream buys. It stays one row per wave;
    // `exec.rs::i8_gemv_mrow` therefore returns 1 for it and `grid.x` stays `out_f`.
    //
    // P4: branchless + dword-wide, the same rewrite P3 applied to the MoE twin `i8acc_q6k` (see the
    // derivation there). The value stream is UNCHANGED — this is the same decode without the
    // per-lane branch and without the byte-at-a-time fetch:
    //
    //  1. The 4-way `if (region)` chain was per-LANE divergent (`region` derives from `blk`, which
    //     is `tid`-strided), so every lane retired all four arms. The selection is pure arithmetic:
    //     (ql byte offset, ql nibble shift, qh bit shift) = (32*(region&1), 4*(region>>1), 2*region).
    //  2. 16 scalar `global_load_u8` per 16 codes. Unlike `linear_i8_q4k`/`_q5k` (144 % 16 == 0,
    //     176 % 16 == 0) the 210-byte Q6_K super-block is only 2-byte aligned, so a `uint4` cast is
    //     not legal here. `__builtin_memcpy` states the align-1 contract honestly and still lowers
    //     to `global_load_b128` on gfx11, which has unaligned global access.
    //
    // That matters far more here than the instruction count suggests: consecutive lanes decode
    // consecutive `blk`, whose super-blocks are `spr*210` bytes apart, so every byte load is a fully
    // address-divergent 32-line L1 request. Cutting 32 byte loads to 2 b128 cuts the line requests
    // with them.
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    int spr = nb >> 3;             // Q6_K super-blocks (256 elems) per output row
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);   // global super-block for (output row o, this 32-block)
        int w32 = blk & 7;         // which 32-block within the super
        const unsigned char* b = w + (long)super * 210;
        const signed char* scales = (const signed char*)(b + 192);
        float d = rf16b(b + 208);
        float sx = xsr[blk];
        // sub16 = w32*2 + hh -> within0 = 32*w32 + 16*hh, so half = within0>>7 = w32>>2 and
        // region = (within0 & 127)>>5 = w32 & 3, both independent of hh; l0 = 16*hh. The 32-block's
        // two 16-code halves therefore share region/half, so the 32 ql bytes and 32 qh bytes it
        // needs are two CONTIGUOUS runs — hoisted out of the `hh` loop and fetched as 2x16 B each.
        int half = w32 >> 2;
        int region = w32 & 3;
        unsigned int qlsh = (unsigned int)(region >> 1) * 4u;
        unsigned int qhsh = (unsigned int)region * 2u;
        unsigned int qlv[8], qhv[8];
        __builtin_memcpy(qlv, b + half * 64 + (region & 1) * 32, 32);
        __builtin_memcpy(qhv, b + 128 + half * 32, 32);
        // The 32-block spans two 16-element sub-blocks, each with its own int8 scale.
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            int sc = (int)scales[w32 * 2 + hh];
            int4 xv = *(const int4*)(qxr + blk * 32 + hh * 16);
            int xa[4] = { xv.x, xv.y, xv.z, xv.w };
            int idot = 0, isum = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {  // 4 groups of 4 = 16
                // Byte j of the dword form is byte j of the scalar form: `>> qlsh & 0x0F0F0F0F`
                // keeps bits [8j+qlsh, 8j+qlsh+3] (the selected nibble of byte j) and
                // `>> qhsh & 0x03030303` keeps bits [8j+qhsh, 8j+qhsh+1] — neither mask lets a
                // neighbouring byte bleed in, and the `<< 4` of a 0..3 value cannot carry out.
                unsigned int lw = qlv[hh * 4 + k], hw = qhv[hh * 4 + k];
                unsigned int wpack = ((lw >> qlsh) & 0x0F0F0F0Fu)
                                   | (((hw >> qhsh) & 0x03030303u) << 4);
                idot = idot4(xa[k], (int)wpack, idot);
                isum = idot4(xa[k], 0x01010101, isum);
            }
            acc += (d * (float)sc) * sx * (float)idot + (d * (float)(-32 * sc)) * sx * (float)isum;
        }
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── Q5_0: 32 elems / 22 bytes; single scale d, offset −16; code 0..31. value = d·(code − 16). ──
// Per 32-block: acc += d·xs·(idot − 16·isum), where idot = Σ qx·code, isum = Σ qx. Same structure as
// the Q4_K min term with sc=1, mn=−16.
extern "C" __global__ void linear_i8_q50(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 22;
        float d = rf16b(b);
        unsigned int qh = (unsigned int)b[2] | ((unsigned int)b[3] << 8)
                        | ((unsigned int)b[4] << 16) | ((unsigned int)b[5] << 24);
        const unsigned char* qs = b + 6;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            int xh0 = (int)(((qh >> p) << 4) & 0x10);
            int xh1 = (int)((qh >> (p + 12)) & 0x10);
            code[p]      = (signed char)((qs[p] & 0x0F) | xh0);
            code[p + 16] = (signed char)((qs[p] >> 4) | xh1);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += d * sx * (float)idot + (d * (float)(-16)) * sx * (float)isum;
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── Q4_0: 32 elems / 18 bytes; single scale d, offset −8; code 0..15. value = d·(code − 8). ──
// Q5_0's inner loop without the `qh` 5th bit: acc += d·xs·(idot − 8·isum).
extern "C" __global__ void linear_i8_q40(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 18;
        float d = rf16b(b);
        const unsigned char* qs = b + 2;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)(qs[p] & 0x0F);
            code[p + 16] = (signed char)(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += d * sx * (float)idot + (d * (float)(-8)) * sx * (float)isum;
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── Q4_1: 32 elems / 20 bytes; AFFINE — scale d, per-block min m; code 0..15. value = d·code + m. ──
// The min term is `m·Σx` (not a constant multiple of `d`), so the `isum` ones-dot is weighted by the
// block's OWN `m` — the only structural difference from Q4_0/Q5_0 in this tier.
extern "C" __global__ void linear_i8_q41(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 20;
        float d = rf16b(b);
        float mn = rf16b(b + 2);
        const unsigned char* qs = b + 4;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)(qs[p] & 0x0F);
            code[p + 16] = (signed char)(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += d * sx * (float)idot + mn * sx * (float)isum;
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── Q5_1: 32 elems / 24 bytes; affine (d, m) + Q5_0's `qh` 5th bit; code 0..31. value = d·code + m. ──
extern "C" __global__ void linear_i8_q51(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 24;
        float d = rf16b(b);
        float mn = rf16b(b + 2);
        unsigned int qh = (unsigned int)b[4] | ((unsigned int)b[5] << 8)
                        | ((unsigned int)b[6] << 16) | ((unsigned int)b[7] << 24);
        const unsigned char* qs = b + 8;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            int xh0 = (int)(((qh >> p) << 4) & 0x10);
            int xh1 = (int)((qh >> (p + 12)) & 0x10);
            code[p]      = (signed char)((qs[p] & 0x0F) | xh0);
            code[p + 16] = (signed char)((qs[p] >> 4) | xh1);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += d * sx * (float)idot + mn * sx * (float)isum;
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── IQ4_NL (R4): 32 elems / 18 bytes; CODEBOOK — value = d · KV[idx], idx ∈ 0..15. ──
// The one structural difference from every affine format in this tier: `kv_iq4nl(idx)` is ALREADY a
// signed value (−127..113), so it IS the dp4a operand — there is nothing to centre and therefore NO
// `isum` ones-dot / min-correction term at all. Per block: acc += d·xs·idot. (This is exactly what
// the CPU reference `vec_dot_iq4nl_32_batch_scalar` does: one i32 `iprod`, one f32 multiply.)
// The table values fit int8, so the 32-element dot cannot overflow i32 (|Σ| ≤ 32·127·127).
extern "C" __global__ void linear_i8_iq4nl(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 18;
        float d = rf16b(b);
        const unsigned char* qs = b + 2;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)kv_iq4nl(qs[p] & 0x0F);
            code[p + 16] = (signed char)kv_iq4nl(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
        }
        acc += d * xsr[blk] * (float)idot;
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── IQ4_XS (R4): 256 elems / 136 bytes; codebook + a 6-bit per-sub-block scale. ──
// value = d·(ls − 32) · KV[idx]. One 32-elem activation block is EXACTLY one of the 8 sub-blocks
// (unlike Q2_K/Q3_K/Q6_K, whose 16-element sub-blocks make a 32-block span two), so this keeps
// Q4_K's one-scale-per-32-block shape: `super = o·spr + blk/8`, sub-block `ib = blk & 7`. Same
// codebook rule as IQ4_NL — signed table value straight into dp4a, no ones-dot.
extern "C" __global__ void linear_i8_iq4xs(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    int spr = nb >> 3;             // IQ4_XS super-blocks (256 elems) per output row
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int ib = blk & 7;          // 32-elem sub-block 0..7 (== this 32-block)
        const unsigned char* b = w + (long)super * 136;
        float d = rf16b(b);
        unsigned int scales_h = (unsigned int)b[2] | ((unsigned int)b[3] << 8);
        int lo = (b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F;
        int hi = (int)((scales_h >> (2 * ib)) & 3u);
        float dl = d * (float)((lo | (hi << 4)) - 32);
        const unsigned char* qs = b + 8 + 16 * ib;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)kv_iq4nl(qs[p] & 0x0F);
            code[p + 16] = (signed char)kv_iq4nl(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
        }
        acc += dl * xsr[blk] * (float)idot;
    }
    acc = wave_sum32(acc);
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
    }
}

// ── The `wdec_*` seam: R5's grid quants + R6's IQ1 and ternary quants. ──
// ONE macro for all ten, because `wdec_##FMT` (NATIVE_DECODE) already absorbed every difference
// between them: it hands back 32 signed codes and the two scales the block's 16-element halves
// carry. Every format on this seam has an already-signed code (see the `wdec_*` header for why —
// the grid byte, IQ1's ×8 delta fold, ternary's folded `−1`), so there is no ones-dot.
//
// Why TWO 16-wide dots rather than Q8_0's single 32-wide one: IQ2_XS, IQ2_S and IQ1_M put a
// separate scale on each half of a 32-element block, so the halves cannot share an int accumulator.
// The formats whose scale IS per-32 pass the same value twice — one extra f32 multiply per block,
// which keeps a single body for the family instead of splitting it two ways over a difference that
// costs nothing in a decode-bound kernel.
#define GEN_LINEAR_I8_WDEC(FMT) \
extern "C" __global__ void linear_i8_##FMT( \
    const signed char* __restrict__ qx, \
    const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, \
    float* __restrict__ dst, \
    const float* __restrict__ resid,  /* [m, out_f] residual folded into the epilogue (null = none) */ \
    int m, int in_f, int out_f \
) { \
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x; \
    int nb = in_f >> 5; \
    const signed char* qxr = qx + (long)row * in_f; \
    const float* xsr = xs + (long)row * nb; \
    float acc = 0.0f; \
    for (int blk = tid; blk < nb; blk += 32) { \
        signed char code[32]; \
        float sc0, sc1; \
        wdec_##FMT(w, (long)o, nb, blk, code, &sc0, &sc1); \
        const int* xp = (const int*)(qxr + blk * 32); \
        int d0 = 0, d1 = 0; \
        for (int k = 0; k < 4; k++) { \
            d0 = idot4(xp[k],     *(const int*)(code + k * 4),      d0); \
            d1 = idot4(xp[4 + k], *(const int*)(code + 16 + k * 4), d1); \
        } \
        float sx = xsr[blk]; \
        acc += sc0 * sx * (float)d0 + sc1 * sx * (float)d1; \
    } \
    acc = wave_sum32(acc); \
    /* Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is */ \
    /* bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes. */ \
    if (tid == 0) { \
        long oi = (long)row * out_f + o; \
        dst[oi] = resid ? (acc + resid[oi]) : acc; \
    } \
}
GEN_LINEAR_I8_WDEC(iq2xxs)
GEN_LINEAR_I8_WDEC(iq2xs)
GEN_LINEAR_I8_WDEC(iq2s)
GEN_LINEAR_I8_WDEC(iq3xxs)
GEN_LINEAR_I8_WDEC(iq3s)
GEN_LINEAR_I8_WDEC(iq1s)
GEN_LINEAR_I8_WDEC(iq1m)
GEN_LINEAR_I8_WDEC(tq10)
GEN_LINEAR_I8_WDEC(tq20)
GEN_LINEAR_I8_WDEC(q20)
GEN_LINEAR_I8_WDEC(mxfp4)
GEN_LINEAR_I8_WDEC(nvfp4)
"#;

// ── Fused RMSNorm → int8 activation quant (Slice 32) ──────────────────────────
//
// Decode fuses the `RmsNorm → Linear` boundary (input_norm→qkv, post_attn_norm→gate/up: 2 per
// decoder layer). Instead of the standalone `rmsnorm` kernel writing the normalized row to global
// memory and `quant_i8_32` reading it straight back, this ONE kernel reads the RAW pre-norm row,
// computes the rmsnorm scale, and int8-quantizes the normalized row in registers — the normalized
// activation never round-trips through DRAM and the `rmsnorm` launch is gone. The dp4a GEMV that
// consumes `qx`/`xs` is unchanged.
//
// Bit-faithful to `rmsnorm` THEN `quant_i8_32`: same block layout (grid.x = rows, blockDim = 256),
// the SAME shared-mem tree reduce for the sum-of-squares (identical float reassociation → identical
// `rms`), and the normalized value `xr*rms*half2float(weight)` is recomputed with the identical
// expression — an f32 store/load round-trip is exact, so the register value equals what
// `quant_i8_32` would have read. The int8 codes + per-32-block scales are therefore bit-identical
// and the golden hash does not move.
//
// F1c: `xn` (optional) additionally writes the NORMALIZED f32 row — the exact bytes the standalone
// `rmsnorm` would have written, from the same expression at the same point. It exists for the
// `RmsNorm → MoeFfn` fold, whose MoE arm has a second consumer of the normalized row that is not an
// int8 GEMV: the f16 router GEMV (`linear_f16`). Writing it here still nets a launch, because the
// pair it replaces is `rmsnorm` + `quant_i8_32`, and costs nothing extra in traffic — it is the same
// store the elided `rmsnorm` did. Null on the dense `RmsNorm → Linear` path, which needs codes only.
const RMSNORM_QUANT_I8: &str = r#"
extern "C" __global__ void rmsnorm_quant_i8_32(
    const float* __restrict__ x,       // [rows, dim] — RAW pre-norm F32 activation
    const __half* __restrict__ weight, // [dim] — dequantized F16 norm weight
    signed char* __restrict__ qx,      // [rows, dim] — int8 codes
    float* __restrict__ xs,            // [rows, dim/32] — per-32-block scales
    float* __restrict__ xn,            // [rows, dim] — normalized F32 row (null = don't write)
    int rows,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* xr = x + (long)row * dim;
    float local = 0.0f;
    for (int i = tid; i < dim; i += nt) {
        float v = xr[i];
        local += v * v;
    }
    __shared__ float sdata[256];
    sdata[tid] = local;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float ss = sdata[0] / (float)dim;
    float rms = 1.0f / sqrtf(ss + eps);
    int nblk = dim >> 5;
    for (int blk = tid; blk < nblk; blk += nt) {
        int base = blk * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float nv = xr[base + j] * rms * __half2float(weight[base + j]);
            float a = fabsf(nv);
            if (a > amax) amax = a;
        }
        float s = amax / 127.0f;
        float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
        signed char* qr = qx + (long)(row * nblk + blk) * 32;
        for (int j = 0; j < 32; j++) {
            float nv = xr[base + j] * rms * __half2float(weight[base + j]);
            float v = roundf(nv * inv);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            qr[j] = (signed char)v;
        }
        // The optional normalized-row write is its OWN loop, not a predicated store inside the
        // quantize loop above: folding it in cost the dense `RmsNorm → Linear` path (which passes
        // `xn == 0` and vastly outnumbers the MoE one) ~1% of decode, measured. This way the null
        // path's inner loop is byte-for-byte the pre-F1c one and only the MoE caller pays — a
        // recompute of `nv`, against a whole kernel launch saved.
        if (xn) {
            float* nr = xn + (long)row * dim + base;
            for (int j = 0; j < 32; j++) {
                nr[j] = xr[base + j] * rms * __half2float(weight[base + j]);
            }
        }
        xs[(long)row * nblk + blk] = s;
    }
}
"#;

// ── Int8-activation dp4a MoE expert FFN (Slice 20) ───────────────────────────
//
// The Phase-3 `moe_ffn_expert_<gu>_<dn>` kernel (MOE_FFN_NATIVE) decodes every expert weight
// element to f16 and re-parses its block header PER ELEMENT, one thread per `nff` row — the same
// per-element f16 round-trip that made small-model dense decode ALU-bound, but paid THREE times
// (gate + up + down) per element AND at a grid of only `nff` threads (one per row → wave underfill).
//
// This path applies the SAME int8 dp4a scheme the dense `linear_i8_*` GEMV uses (INT8_DECODE) to the
// three expert projections. An expert FFN is exactly three GEMVs:
//   gate:  g = x[ne]           · gate_w[nff, ne]ᵀ        (out = nff)
//   up:    u = x[ne]           · up_w[nff, ne]ᵀ          (out = nff)
//   down:  y = h[nff]          · down_w[ne, nff]ᵀ        (out = ne, accumulated over experts)
// with the elementwise activation h = act(g·wg)·(u·wg)·wo·down_scale between them. The gate/up GEMVs
// integer-dot the int8-quantized INPUT row `x` (quantized ONCE per token via `quant_i8_32`, reused
// across every expert AND both gate & up) against the raw expert quant codes; the down GEMV
// integer-dots the int8-quantized activation `h` (quantized per expert). The per-block weight scale
// (+ Q4_K/Q6_K min term) is applied to the int32 accumulator AFTER the dot — the mmq "scale-after is
// free" principle — reusing the exact `i8acc_*` decode+dot the dense path is parity-tested on.
//
// Grid: one wave32 block per OUTPUT ROW (gate/up → nff blocks, down → ne blocks), the 32 lanes
// striding over the input's 32-elem blocks then a `wave_sum32` reduce to lane 0 — the SAME grid the
// dense `linear_i8_*` GEMV uses, so decode parallelizes across nff/ne instead of the one-thread-per-
// row underfill of the Phase-3 fused kernel.
//
// SANCTIONED PRECISION FLIP: int8 activation quant is lossy in BOTH stages (x and h), so the output
// differs (within tolerance) from the bit-faithful f16 expert path — parity is checked vs the CPU
// reference with a widened int8 tolerance (docs/perf.md). Covered GU/DN formats: Q8_0, Q2_K, Q3_K,
// Q4_K, Q5_K, Q6_K (every K-quant expert bank a real GGUF ships, incl. llama4-Scout's Q2_K/Q3_K),
// the R3 legacy round quants Q4_0, Q4_1, Q5_1, and the R4 codebook quants IQ4_NL, IQ4_XS;
// uncovered keep the Phase-3 `moe_ffn_expert_*` fallback. Unlike the `moe_ffn_expert_<gu>_<dn>`
// cross product, this per-FORMAT pair of kernels IS total over `moe_native_fmt` — which is what
// keeps the SHIPPING (int8) expert path complete even though the cross product is not.
//
// `rf16b`/`k4` (NATIVE_DECODE) and `idot4`/`wave_sum32` (INT8_DECODE) are defined in the parts
// assembled before this one.
const MOE_FFN_INT8: &str = r#"
// Per-lane int8 dp4a accumulation for one output row `o` of a Q8_0 weight bank: mirrors the
// `linear_i8_q80` inner loop (bit-identical decode + dot), returning this lane's partial (pre-wave-
// reduce). `w` is pre-advanced to the expert's bank; row `o` spans `nb = in_f/32` 32-blocks.
__device__ __forceinline__ float i8acc_q80(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 34;
        float d = rf16b(b);
        const unsigned char* wq = b + 2;
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0;
        for (int k = 0; k < 8; k++) {
            const unsigned char* q = wq + k * 4;
            int wpack = (int)q[0] | ((int)q[1] << 8) | ((int)q[2] << 16) | ((int)q[3] << 24);
            idot = idot4(xp[k], wpack, idot);
        }
        acc += d * xsr[blk] * (float)idot;
    }
    return acc;
}

// Q2_K per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q2k`.
__device__ __forceinline__ float i8acc_q2k(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int w32 = blk & 7;
        const unsigned char* b = w + (long)super * 84;
        const unsigned char* scales = b;
        const unsigned char* qs = b + 16;
        float d = rf16b(b + 80);
        float dmin = rf16b(b + 82);
        float sx = xsr[blk];
        for (int hh = 0; hh < 2; hh++) {
            int g = w32 * 2 + hh;
            int sc = scales[g] & 0x0F;
            int mm = scales[g] >> 4;
            const unsigned char* qb = qs + (g >> 3) * 32 + (g & 1) * 16;
            int sh = 2 * ((g >> 1) & 3);
            const int* xp = (const int*)(qxr + blk * 32 + hh * 16);
            int idot = 0, isum = 0;
            for (int k = 0; k < 4; k++) {
                int wpack = 0;
                for (int r = 0; r < 4; r++) wpack |= (int)((qb[k * 4 + r] >> sh) & 3) << (r * 8);
                idot = idot4(xp[k], wpack, idot);
                isum = idot4(xp[k], 0x01010101, isum);
            }
            acc += (d * (float)sc) * sx * (float)idot + (dmin * (float)(-mm)) * sx * (float)isum;
        }
    }
    return acc;
}

// Q3_K per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q3k`.
__device__ __forceinline__ float i8acc_q3k(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int w32 = blk & 7;
        const unsigned char* b = w + (long)super * 110;
        const unsigned char* hmask = b;
        const unsigned char* qs = b + 32;
        float d = rf16b(b + 108);
        float sx = xsr[blk];
        for (int hh = 0; hh < 2; hh++) {
            int g = w32 * 2 + hh;
            int sc = q3k_sc6(b + 96, g);
            const unsigned char* qb = qs + (g >> 3) * 32 + (g & 1) * 16;
            const unsigned char* hb = hmask + (g & 1) * 16;
            int sh = 2 * ((g >> 1) & 3);
            int hsh = g >> 1;
            const int* xp = (const int*)(qxr + blk * 32 + hh * 16);
            int idot = 0, isum = 0;
            for (int k = 0; k < 4; k++) {
                int wpack = 0;
                for (int r = 0; r < 4; r++) {
                    int p = k * 4 + r;
                    int c = ((qb[p] >> sh) & 3) | (((hb[p] >> hsh) & 1) << 2);
                    wpack |= c << (r * 8);
                }
                idot = idot4(xp[k], wpack, idot);
                isum = idot4(xp[k], 0x01010101, isum);
            }
            acc += (d * (float)sc) * sx * (float)idot + (d * (float)(-4 * sc)) * sx * (float)isum;
        }
    }
    return acc;
}

// Q4_K per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q4k`.
__device__ __forceinline__ float i8acc_q4k(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int s = blk & 7;
        // F4: same 128-bit weight fetch as `linear_i8_q4k`. An expert bank starts at a
        // `hipMalloc` base plus `expert * (rows*in_f/256) * 144` (paged: `slot * stride_bytes`,
        // and the stride IS that per-expert byte count), so the block base is 16-byte aligned.
        const uint4* bq = (const uint4*)(w + (long)super * 144);
        uint4 hdr = bq[0];
        float d = f16q_lo(hdr.x);
        float dmin = f16q_hi(hdr.x);
        int sc, mm; k4q(hdr, s, &sc, &mm);
        const uint4* qq = bq + 1 + (s >> 1) * 2;
        uint4 wlo = qq[0], whi = qq[1];
        unsigned int sh = (unsigned int)(s & 1) * 4u;
        const int4* xq = (const int4*)(qxr + blk * 32);
        int4 xlo = xq[0], xhi = xq[1];
        unsigned int wv[8] = { wlo.x, wlo.y, wlo.z, wlo.w, whi.x, whi.y, whi.z, whi.w };
        int xv[8] = { xlo.x, xlo.y, xlo.z, xlo.w, xhi.x, xhi.y, xhi.z, xhi.w };
        int idot = 0, isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            idot = idot4(xv[k], (int)((wv[k] >> sh) & 0x0F0F0F0Fu), idot);
            isum = idot4(xv[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += (d * (float)sc) * sx * (float)idot + (dmin * (float)(-mm)) * sx * (float)isum;
    }
    return acc;
}

// Q5_K per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q5k`.
__device__ __forceinline__ float i8acc_q5k(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int s = blk & 7;
        // F4: same 128-bit weight fetch as `linear_i8_q5k` (176 % 16 == 0; qh at +16, qs at +48).
        const uint4* bq = (const uint4*)(w + (long)super * 176);
        uint4 hdr = bq[0];
        float d = f16q_lo(hdr.x);
        float dmin = f16q_hi(hdr.x);
        int sc, mm; k4q(hdr, s, &sc, &mm);
        uint4 qh0 = bq[1], qh1 = bq[2];
        const uint4* qq = bq + 3 + (s >> 1) * 2;
        uint4 wlo = qq[0], whi = qq[1];
        unsigned int sh = (unsigned int)(s & 1) * 4u;
        const int4* xq = (const int4*)(qxr + blk * 32);
        int4 xlo = xq[0], xhi = xq[1];
        unsigned int wv[8] = { wlo.x, wlo.y, wlo.z, wlo.w, whi.x, whi.y, whi.z, whi.w };
        unsigned int hv[8] = { qh0.x, qh0.y, qh0.z, qh0.w, qh1.x, qh1.y, qh1.z, qh1.w };
        int xv[8] = { xlo.x, xlo.y, xlo.z, xlo.w, xhi.x, xhi.y, xhi.z, xhi.w };
        int idot = 0, isum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned int wpack = ((wv[k] >> sh) & 0x0F0F0F0Fu)
                               | (((hv[k] >> s) & 0x01010101u) << 4);
            idot = idot4(xv[k], (int)wpack, idot);
            isum = idot4(xv[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += (d * (float)sc) * sx * (float)idot + (dmin * (float)(-mm)) * sx * (float)isum;
    }
    return acc;
}

// Q6_K per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q6k`.
//
// P3: branchless + dword-wide. The value stream is UNCHANGED (see the derivation below); this is
// the same decode expressed without the per-lane branch and without the byte-at-a-time fetch.
//
// The old form cost ~9x per MAC what the Q4_K twin does, which made Q6_K `ffn_down_exps` ~74% of a
// Q4_K_M MoE pp512 forward (the 24 Q6_K-down layers ran 40.1 ms vs 10.8 ms for the 24 Q4_K-down
// ones, on an otherwise identical op). Two causes, both fixed here:
//
//  1. A 4-way `if (region)` chain per code. `region` derives from `blk`, which is `tid`-strided, so
//     it is per-LANE divergent inside a wave32 and every lane retired all four arms. `region`
//     selects (ql byte offset, ql nibble shift, qh bit shift) = (32*(region&1), 4*(region>>1),
//     2*region) — pure arithmetic, so the chain is pure waste.
//  2. 16 scalar `global_load_u8` per 16 codes. Unlike `i8acc_q4k`/`i8acc_q5k` (144 % 16 == 0,
//     176 % 16 == 0) the 210-byte Q6_K super-block is only 2-byte aligned, so a `uint4` cast is
//     not legal here. `__builtin_memcpy` states the align-1 contract honestly and still lowers to
//     `global_load_b128` on gfx11, which has unaligned global access — verified in the ISA.
//
// The two 16-code halves of a 32-code block share `region` and `half` (their `sub16` differ only in
// bit 0, and that bit lands entirely in `l0` = 16*hh), so the 32 ql bytes and 32 qh bytes a block
// needs are two CONTIGUOUS runs — hoisted out of the `hh` loop and fetched as 2x16 B each.
__device__ __forceinline__ float i8acc_q6k(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int w32 = blk & 7;
        const unsigned char* b = w + (long)super * 210;
        const signed char* scales = (const signed char*)(b + 192);
        float d = rf16b(b + 208);
        float sx = xsr[blk];
        // sub16 = w32*2 + hh -> within0 = 32*w32 + 16*hh, so half = within0>>7 = w32>>2 and
        // region = (within0 & 127)>>5 = w32 & 3, both independent of hh; l0 = 16*hh.
        int half = w32 >> 2;
        int region = w32 & 3;
        unsigned int qlsh = (unsigned int)(region >> 1) * 4u;
        unsigned int qhsh = (unsigned int)region * 2u;
        unsigned int qlv[8], qhv[8];
        __builtin_memcpy(qlv, b + half * 64 + (region & 1) * 32, 32);
        __builtin_memcpy(qhv, b + 128 + half * 32, 32);
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            int sc = (int)scales[w32 * 2 + hh];
            int4 xv = *(const int4*)(qxr + blk * 32 + hh * 16);
            int xa[4] = { xv.x, xv.y, xv.z, xv.w };
            int idot = 0, isum = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                // Byte j of the dword form is byte j of the scalar form: `>> qlsh & 0x0F0F0F0F`
                // keeps bits [8j+qlsh, 8j+qlsh+3] (the selected nibble of byte j) and
                // `>> qhsh & 0x03030303` keeps bits [8j+qhsh, 8j+qhsh+1] — neither mask lets a
                // neighbouring byte bleed in, and the `<< 4` of a 0..3 value cannot carry out.
                unsigned int lw = qlv[hh * 4 + k], hw = qhv[hh * 4 + k];
                unsigned int wpack = ((lw >> qlsh) & 0x0F0F0F0Fu)
                                   | (((hw >> qhsh) & 0x03030303u) << 4);
                idot = idot4(xa[k], (int)wpack, idot);
                isum = idot4(xa[k], 0x01010101, isum);
            }
            acc += (d * (float)sc) * sx * (float)idot + (d * (float)(-32 * sc)) * sx * (float)isum;
        }
    }
    return acc;
}

// Q4_0 per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q40`.
__device__ __forceinline__ float i8acc_q40(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 18;
        float d = rf16b(b);
        const unsigned char* qs = b + 2;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)(qs[p] & 0x0F);
            code[p + 16] = (signed char)(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += d * sx * (float)idot + (d * (float)(-8)) * sx * (float)isum;
    }
    return acc;
}

// Q4_1 per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q41`.
__device__ __forceinline__ float i8acc_q41(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 20;
        float d = rf16b(b);
        float mn = rf16b(b + 2);
        const unsigned char* qs = b + 4;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)(qs[p] & 0x0F);
            code[p + 16] = (signed char)(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += d * sx * (float)idot + mn * sx * (float)isum;
    }
    return acc;
}

// Q5_1 per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q51`.
__device__ __forceinline__ float i8acc_q51(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 24;
        float d = rf16b(b);
        float mn = rf16b(b + 2);
        unsigned int qh = (unsigned int)b[4] | ((unsigned int)b[5] << 8)
                        | ((unsigned int)b[6] << 16) | ((unsigned int)b[7] << 24);
        const unsigned char* qs = b + 8;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            int xh0 = (int)(((qh >> p) << 4) & 0x10);
            int xh1 = (int)((qh >> (p + 12)) & 0x10);
            code[p]      = (signed char)((qs[p] & 0x0F) | xh0);
            code[p + 16] = (signed char)((qs[p] >> 4) | xh1);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += d * sx * (float)idot + mn * sx * (float)isum;
    }
    return acc;
}

// IQ4_NL per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_iq4nl`. Codebook:
// the table value IS the signed dp4a operand, so there is no ones-dot / min term.
__device__ __forceinline__ float i8acc_iq4nl(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        const unsigned char* b = w + ((long)o * nb + blk) * 18;
        float d = rf16b(b);
        const unsigned char* qs = b + 2;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)kv_iq4nl(qs[p] & 0x0F);
            code[p + 16] = (signed char)kv_iq4nl(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
        }
        acc += d * xsr[blk] * (float)idot;
    }
    return acc;
}

// IQ4_XS per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_iq4xs`.
__device__ __forceinline__ float i8acc_iq4xs(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int ib = blk & 7;
        const unsigned char* b = w + (long)super * 136;
        float d = rf16b(b);
        unsigned int scales_h = (unsigned int)b[2] | ((unsigned int)b[3] << 8);
        int lo = (b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F;
        int hi = (int)((scales_h >> (2 * ib)) & 3u);
        float dl = d * (float)((lo | (hi << 4)) - 32);
        const unsigned char* qs = b + 8 + 16 * ib;
        signed char code[32];
        for (int p = 0; p < 16; p++) {
            code[p]      = (signed char)kv_iq4nl(qs[p] & 0x0F);
            code[p + 16] = (signed char)kv_iq4nl(qs[p] >> 4);
        }
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0;
        for (int k = 0; k < 8; k++) {
            const int* cp = (const int*)(code + k * 4);
            idot = idot4(xp[k], cp[0], idot);
        }
        acc += dl * xsr[blk] * (float)idot;
    }
    return acc;
}

// Per-lane int8 dp4a accumulation for every format on the `wdec_*` seam (R5's grid quants, R6's
// IQ1 and ternary quants) — the `linear_i8_##FMT` inner loop verbatim, over the same shared
// `wdec_##FMT` decoder, so the MoE expert path and the dense GEMV cannot diverge. Same
// two-16-wide-dots shape and the same no-ones-dot rule as there.
#define GEN_I8ACC_WDEC(FMT) \
__device__ __forceinline__ float i8acc_##FMT( \
    const signed char* __restrict__ qxr, const float* __restrict__ xsr, \
    const unsigned char* __restrict__ w, int o, int nb, int tid) { \
    float acc = 0.0f; \
    for (int blk = tid; blk < nb; blk += 32) { \
        signed char code[32]; \
        float sc0, sc1; \
        wdec_##FMT(w, (long)o, nb, blk, code, &sc0, &sc1); \
        const int* xp = (const int*)(qxr + blk * 32); \
        int d0 = 0, d1 = 0; \
        for (int k = 0; k < 4; k++) { \
            d0 = idot4(xp[k],     *(const int*)(code + k * 4),      d0); \
            d1 = idot4(xp[4 + k], *(const int*)(code + 16 + k * 4), d1); \
        } \
        float sx = xsr[blk]; \
        acc += sc0 * sx * (float)d0 + sc1 * sx * (float)d1; \
    } \
    return acc; \
}
GEN_I8ACC_WDEC(iq2xxs)
GEN_I8ACC_WDEC(iq2xs)
GEN_I8ACC_WDEC(iq2s)
GEN_I8ACC_WDEC(iq3xxs)
GEN_I8ACC_WDEC(iq3s)
GEN_I8ACC_WDEC(iq1s)
GEN_I8ACC_WDEC(iq1m)
GEN_I8ACC_WDEC(tq10)
GEN_I8ACC_WDEC(tq20)
GEN_I8ACC_WDEC(q20)
GEN_I8ACC_WDEC(mxfp4)
GEN_I8ACC_WDEC(nvfp4)

// Gate+up+activation for one expert: block `o` (0..nff) computes h_out[o] = act(g·wg)·(u·wg)·wo·dsc.
// `qx`/`xs` are the int8 quantization of the token's input row x[ne] (produced ONCE per token, reused
// across experts + both gate & up). gate_w/up_w are pre-advanced to this expert's banks (fused gate/up
// simply passes up_w = gate_w + nff*ne offset). One wave32 block per nff output row.
#define GEN_MOE_GATE_UP(GU) \
extern "C" __global__ void moe_gate_up_act_i8_##GU( \
    const signed char* __restrict__ qx,       /* int8(x)  [ne] */ \
    const float* __restrict__ xs,             /* x scales [ne/32] */ \
    const unsigned char* __restrict__ gate_w, /* raw GU bytes [nff, ne] (pre-advanced) */ \
    const unsigned char* __restrict__ up_w,   /* raw GU bytes [nff, ne] (pre-advanced) */ \
    float* __restrict__ h_out,                /* [nff] */ \
    int ne, int nff, int act_type, float wg, float wo, float down_scale) { \
    int o = blockIdx.x; \
    int tid = threadIdx.x; \
    if (o >= nff) return; \
    int nb = ne >> 5; \
    float g = i8acc_##GU(qx, xs, gate_w, o, nb, tid); \
    float u = i8acc_##GU(qx, xs, up_w, o, nb, tid); \
    g = wave_sum32(g); \
    u = wave_sum32(u); \
    if (tid == 0) { \
        g *= wg; \
        u *= wg; \
        float a; \
        if (act_type == 0) { \
            a = g / (1.0f + expf(-g)); \
        } else if (act_type == 1) { \
            float x3 = g * g * g; \
            a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3))); \
        } else { \
            a = 1.0f / (1.0f + expf(-g)); \
        } \
        h_out[o] = a * u * wo * down_scale; \
    } \
}

// Down projection for one expert: block `d` (0..ne) computes y_d = h[nff] · down_w[d, :] and
// atomicAdds it into dst[d] (accumulating across the selected experts; the routing weight is already
// folded into h by the gate/up kernel). `hq`/`hs` are the int8 quantization of h[nff]. One wave32
// block per ne output row.
#define GEN_MOE_DOWN(DN) \
extern "C" __global__ void moe_down_i8_##DN( \
    const signed char* __restrict__ hq,       /* int8(h)  [nff] */ \
    const float* __restrict__ hs,             /* h scales [nff/32] */ \
    const unsigned char* __restrict__ down_w, /* raw DN bytes [ne, nff] (pre-advanced) */ \
    float* __restrict__ dst,                  /* [ne] — accumulated */ \
    int ne, int nff) { \
    int d = blockIdx.x; \
    int tid = threadIdx.x; \
    if (d >= ne) return; \
    int nb = nff >> 5; \
    float acc = i8acc_##DN(hq, hs, down_w, d, nb, tid); \
    acc = wave_sum32(acc); \
    if (tid == 0) atomicAdd(&dst[d], acc); \
}

GEN_MOE_GATE_UP(q80)
GEN_MOE_GATE_UP(q2k)
GEN_MOE_GATE_UP(q3k)
GEN_MOE_GATE_UP(q4k)
GEN_MOE_GATE_UP(q5k)
GEN_MOE_GATE_UP(q6k)
GEN_MOE_GATE_UP(q40)
GEN_MOE_GATE_UP(q41)
GEN_MOE_GATE_UP(q51)
GEN_MOE_GATE_UP(iq4nl)
GEN_MOE_GATE_UP(iq4xs)
GEN_MOE_GATE_UP(iq2xxs)
GEN_MOE_GATE_UP(iq2xs)
GEN_MOE_GATE_UP(iq2s)
GEN_MOE_GATE_UP(iq3xxs)
GEN_MOE_GATE_UP(iq3s)
GEN_MOE_GATE_UP(iq1s)
GEN_MOE_GATE_UP(iq1m)
GEN_MOE_GATE_UP(tq10)
GEN_MOE_GATE_UP(tq20)
GEN_MOE_GATE_UP(q20)
GEN_MOE_GATE_UP(mxfp4)
GEN_MOE_GATE_UP(nvfp4)
GEN_MOE_DOWN(q80)
GEN_MOE_DOWN(q2k)
GEN_MOE_DOWN(q3k)
GEN_MOE_DOWN(q4k)
GEN_MOE_DOWN(q5k)
GEN_MOE_DOWN(q6k)
GEN_MOE_DOWN(q40)
GEN_MOE_DOWN(q41)
GEN_MOE_DOWN(q51)
GEN_MOE_DOWN(iq4nl)
GEN_MOE_DOWN(iq4xs)
GEN_MOE_DOWN(iq2xxs)
GEN_MOE_DOWN(iq2xs)
GEN_MOE_DOWN(iq2s)
GEN_MOE_DOWN(iq3xxs)
GEN_MOE_DOWN(iq3s)
GEN_MOE_DOWN(iq1s)
GEN_MOE_DOWN(iq1m)
GEN_MOE_DOWN(tq10)
GEN_MOE_DOWN(tq20)
GEN_MOE_DOWN(q20)
GEN_MOE_DOWN(mxfp4)
GEN_MOE_DOWN(nvfp4)
"#;

// ── Matrix-core (WMMA) int8 prefill GEMM (Phase 5, RM×CN register-tiled — Slice 25) ──
//
// The Phase-4 `linear_i8_*` GEMV grids one wave32 block per (output row `o`, activation row) — so a
// weight column is DECODED ONCE PER ACTIVATION ROW; that redundant decode ceilings prefill (m>1).
// Slice-15 moved prefill onto the RDNA3 wave32 matrix cores
// (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`, signed int8 → int32, `16x16x16`), one 16×16 output
// tile per wave. Slice-25 register-tiles that: each wave now computes an `RM`×`CN` grid of 16×16
// output tiles (16*RM rows × 16*CN cols), reusing every loaded operand across the tile.
//
// Fragment layout (RDNA3 wave32, empirically confirmed by `examples/wmma_probe`):
//   * A fragment: lane l feeds row (l%16) of the M×K tile — 16 K-values packed 4×int8/int32 (i4v).
//   * B fragment: lane l feeds col (l%16) of the K×N tile — 16 K-values packed the same way.
//   * D/C accumulator: 8 int32/lane; element e of lane l is output (row = 2*e + l/16, col = l%16).
// int8 is SIGNED (neg_a = neg_b = true); unsigned would 256× the result (probe-verified).
//
// Why RM×CN (measure-driven, docs/perf.md occupancy taxonomy). The Slice-15 wave read the SAME
// activation rows once per output-column tile (out_f/16 redundant A reads) AND the SAME weight column
// once per output-row tile (m/16 redundant decodes). We measured which redundancy actually bounds the
// kernel on gfx1100 (24 GB RX 7900 XTX, 16 waves/SIMD max) with an ISOLATED-GEMM GFLOP/s micro-bench
// (`examples/wmma_bench`) — pp512 dilutes the GEMM with attention/norms/dispatch and hides the signal:
//   * The Slice-15 tile (RM=CN=1) already hits the 16 waves/SIMD occupancy cap (85 VGPR, 0 spill) —
//     so the kernel is not register-occupancy-starved; it is memory/latency bound at full occupancy.
//   * Blocking M (reuse the decoded weight tile across RM row tiles, `2x1`) STRICTLY beats `1x1` on
//     every shape (+2..16% GFLOP/s): fewer weight decodes/global reads, and the min-term ones-dot
//     (`sumacc`, which depends only on A) is computed once per row tile instead of once per column.
//   * Blocking M AND N (`2x2`) additionally wins the wide-N GEMMs (out_f ≥ 2048: up/gate, wide
//     projections, +4% over `2x1`) but loses ~11-14% on the square/narrow ones (qkv, down) where the
//     extra CN accumulators cost occupancy for no reuse win. Pure-N blocking (`1x2`/`1x4`, reuse A
//     only) measured strictly worse everywhere and was dropped.
// So the auto tier is `2x2` for wide-N GEMMs, `2x1` otherwise (see `wmma_tile` in exec.rs). A is read
// straight from global (no LDS staging): at the 16-wave occupancy cap, LDS-staging A only spends the
// shared-memory budget without buying latency hiding the scheduler doesn't already get from the wave
// pool — the Vulkan "A_GLOBAL" slice reached the same conclusion. The decoded weight tile and the
// RM×CN accumulators live in registers (2x2 = 192 VGPR / 8 waves/SIMD, 0 spill; 2x1 lighter).
// `INFR_ROCM_WMMA_TILE=RxC` (1x1/2x1/2x2) overrides the auto tier for A/B benchmarking.
//
// Bit-faithfulness (goldens MUST NOT move): the per-format code/scale/min extraction is byte-identical
// to Slice-15 and the parity-tested `linear_i8_*` GEMV (same `k4`/`q3k_sc6`/`rf16b`, same nibble/
// region math). The affine min term (Q4_K/Q5_K `dmin·(−mm)·Σqx`, Q6_K `d·(−32s)·Σqx`, Q5_0
// `d·(−16)·Σqx`, Q2_K `dmin·(−mm)·Σqx`, Q3_K `d·(−4·sc6)·Σqx`) is a second WMMA
// against an all-ones B fragment. Every output element `dst[re,col]` is still the SAME
// `Σ_blk axs·(wsc·dot + wmn·sum)` summed in the SAME block order — RM/CN only re-group which (row,col)
// tiles one wave owns, they never reorder an element's f32 accumulation. Pure scheduling change: no
// re-bless. `in_f` is 32-aligned for every covered format, so K needs no padding; m/out_f edges are
// masked (out-of-range rows/cols load zero and skip the store); RM/CN need not divide m/16 or out_f/16.
const WMMA_PREFILL: &str = r#"
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i8v __attribute__((ext_vector_type(8)));

// Signed int8 16x16x16 matrix multiply-accumulate (wave32). neg_a/neg_b=true → signed operands.
static __device__ __forceinline__ i8v wmma_dot(i4v a, i4v b, i8v c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

// Pack 16 signed-int8 codes (byte 0 = lowest K) from a contiguous buffer into an i4v K-fragment.
static __device__ __forceinline__ i4v pack16(const signed char* p) {
    i4v v;
    v[0] = *(const int*)(p + 0);
    v[1] = *(const int*)(p + 4);
    v[2] = *(const int*)(p + 8);
    v[3] = *(const int*)(p + 12);
    return v;
}

// Decode one Q6_K K-tile (16 codes, 0..63) straight into a packed K-fragment. `ql`/`qh` point at the
// 16-byte contiguous low-nibble and high-bit runs this tile needs; `qlsh`/`qhsh` are the region's
// nibble/bit shifts (see GEN_WMMA_Q6K). Byte j of each dword is byte j of the scalar form:
// `>> qlsh & 0x0F0F0F0F` keeps bits [8j+qlsh, 8j+qlsh+3] and `>> qhsh & 0x03030303` keeps bits
// [8j+qhsh, 8j+qhsh+1] — neither mask lets a neighbouring byte bleed in, and the `<< 4` of a 0..3
// value cannot carry out, so this is byte-identical to the per-code reassembly it replaces.
// The 210-byte super-block is only 2-byte aligned so a `uint4` cast is illegal; `__builtin_memcpy`
// states the align-1 contract and still lowers to `global_load_b128` on gfx11.
static __device__ __forceinline__ i4v q6k_tile16(const unsigned char* ql, const unsigned char* qh,
                                                 unsigned int qlsh, unsigned int qhsh) {
    unsigned int lv[4], hv[4];
    __builtin_memcpy(lv, ql, 16);
    __builtin_memcpy(hv, qh, 16);
    i4v v;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        v[k] = (int)(((lv[k] >> qlsh) & 0x0F0F0F0Fu) | (((hv[k] >> qhsh) & 0x03030303u) << 4));
    }
    return v;
}

// Load the A K-fragment (16 int8 of activation row `row_in` at absolute element offset `koff`),
// or zero if this lane's input row is past the m-edge. `koff` already includes `row_in*in_f`.
static __device__ __forceinline__ i4v load_a(const signed char* qx, int row_in, int m, long koff) {
    if (row_in >= m) return (i4v){0, 0, 0, 0};
    return pack16(qx + koff);
}

// ── Q8_0: 32 elems/block = 2 K-tiles, scale d, no min. ──
#define GEN_WMMA_Q80(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][32]; \
    float wsc[CN]; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                const unsigned char* b = w + ((long)col * nblk + blk) * 34; \
                wsc[c] = rf16b(b); \
                for (int j = 0; j < 32; j++) wc[c][j] = (signed char)b[2 + j]; \
            } else { wsc[c] = 0.0f; for (int j = 0; j < 32; j++) wc[c][j] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)blk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, pack16(wc[c]),      dotacc); \
                dotacc = wmma_dot(a1, pack16(wc[c] + 16), dotacc); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * wsc[c] * (float)dotacc[e]; \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q4_K: 256/super-block, 8 sub-blocks of 32 (= 2 K-tiles each), scale d·sc + min dmin·(−mm). ──
#define GEN_WMMA_Q4K(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    int spr = nblk >> 3; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][32]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                long super = (long)col * spr + (blk >> 3); \
                int s = blk & 7; \
                const unsigned char* b = w + super * 144; \
                float d = rf16b(b), dmin = rf16b(b + 2); \
                int sc, mm; k4(b + 4, s, &sc, &mm); \
                wsc[c] = d * (float)sc; \
                wmn[c] = dmin * (float)(-mm); \
                const unsigned char* qbase = (b + 16) + (s >> 1) * 32; \
                int hi = s & 1; \
                for (int p = 0; p < 32; p++) wc[c][p] = (signed char)(hi ? (qbase[p] >> 4) : (qbase[p] & 0x0F)); \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; for (int p = 0; p < 32; p++) wc[c][p] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)blk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            i8v sumacc = {0,0,0,0,0,0,0,0}; \
            sumacc = wmma_dot(a0, ones, sumacc); sumacc = wmma_dot(a1, ones, sumacc); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, pack16(wc[c]),      dotacc); \
                dotacc = wmma_dot(a1, pack16(wc[c] + 16), dotacc); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q5_K: 256/super-block, 8 sub-blocks of 32 (= 2 K-tiles each), Q4_K scale/min + a 5th code bit. ──
// Byte-identical decode to `linear_i8_q5k` / `deq_q5k`; only the register tiling differs, so the
// per-output f32 accumulation order matches the GEMV tier exactly (same Σ_blk axs·(wsc·dot + wmn·sum)).
#define GEN_WMMA_Q5K(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    int spr = nblk >> 3; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][32]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                long super = (long)col * spr + (blk >> 3); \
                int s = blk & 7; \
                const unsigned char* b = w + super * 176; \
                float d = rf16b(b), dmin = rf16b(b + 2); \
                int sc, mm; k4(b + 4, s, &sc, &mm); \
                wsc[c] = d * (float)sc; \
                wmn[c] = dmin * (float)(-mm); \
                const unsigned char* qh = b + 16; \
                const unsigned char* qbase = (b + 48) + (s >> 1) * 32; \
                int hi = s & 1; \
                for (int p = 0; p < 32; p++) \
                    wc[c][p] = (signed char)((hi ? (qbase[p] >> 4) : (qbase[p] & 0x0F)) | (((qh[p] >> s) & 1) << 4)); \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; for (int p = 0; p < 32; p++) wc[c][p] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)blk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            i8v sumacc = {0,0,0,0,0,0,0,0}; \
            sumacc = wmma_dot(a0, ones, sumacc); sumacc = wmma_dot(a1, ones, sumacc); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, pack16(wc[c]),      dotacc); \
                dotacc = wmma_dot(a1, pack16(wc[c] + 16), dotacc); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q6_K: 256/super-block, 16 sub-blocks of 16 (= 1 K-tile each), int8 scale, code 0..63. ──
//
// P4: branchless + dword-wide, the dense-prefill twin of the P3 rewrite of `i8acc_q6k`. The value
// stream is UNCHANGED. Two differences from the GEMV form, both from `sb` being the loop counter:
//
//  - `region`/`h128`/`l0` derive from `sb` alone, so unlike the GEMV they are WAVE-UNIFORM here and
//    the `if (region)` chain was a scalar branch, not a divergent one. It is still recomputed per
//    output-column tile for no reason, so it is hoisted out of the `c` loop and made arithmetic:
//    (ql byte offset, ql nibble shift, qh bit shift) = (32*(region&1), 4*(region>>1), 2*region).
//  - The real cost was 32 scalar `global_load_u8` per 16 codes (16 ql + 16 qh). Consecutive lanes
//    hold consecutive `col`, whose super-blocks are `spr*210` bytes apart, so each of those was a
//    fully address-divergent 16-line L1 request (lanes 0-15 and 16-31 share `col`). A K-tile needs
//    one contiguous 16-byte ql run and one contiguous 16-byte qh run, so this is 2 `global_load_b128`
//    — see `q6k_tile16`. Per column tile the VMEM goes 34 -> 4 (the `d`/scale loads remain).
#define GEN_WMMA_Q6K(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    int spr = nblk >> 3; \
    int n16 = in_f >> 4; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    i4v wc[CN]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int sb = 0; sb < n16; sb++) { \
        int blk32 = sb >> 1; \
        int w32 = blk32 & 7; \
        int sub16 = w32 * 2 + (sb & 1); \
        /* within0 = sub16*16 = 32*w32 + 16*(sb&1), so h128 = within0>>7 = w32>>2, */ \
        /* region = (within0 & 127)>>5 = w32 & 3, and l0 = within0 & 31 = 16*(sb&1). */ \
        int h128 = w32 >> 2; \
        int region = w32 & 3; \
        int l0 = (sb & 1) * 16; \
        unsigned int qlsh = (unsigned int)(region >> 1) * 4u; \
        unsigned int qhsh = (unsigned int)region * 2u; \
        int qlo = h128 * 64 + (region & 1) * 32 + l0; \
        int qho = 128 + h128 * 32 + l0; \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                const unsigned char* b = w + ((long)col * spr + (blk32 >> 3)) * 210; \
                float d = rf16b(b + 208); \
                int sc = (int)((const signed char*)(b + 192))[sub16]; \
                wsc[c] = d * (float)sc; \
                wmn[c] = d * (float)(-32 * sc); \
                wc[c] = q6k_tile16(b + qlo, b + qho, qlsh, qhsh); \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; wc[c] = (i4v){0, 0, 0, 0}; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long koff = (long)row_in * in_f + (long)sb * 16; \
            i4v a = load_a(qx, row_in, m, koff); \
            i8v sumacc = wmma_dot(a, ones, (i8v){0,0,0,0,0,0,0,0}); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = wmma_dot(a, wc[c], (i8v){0,0,0,0,0,0,0,0}); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk32] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q2_K: 256/super-block, 16 sub-blocks of 16 (= 1 K-tile each), 4-bit scale + 4-bit min. ──
// Same per-16 K-tile walk as GEN_WMMA_Q6K; the decode is byte-identical to `linear_i8_q2k`/`deq_q2k`.
#define GEN_WMMA_Q2K(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    int spr = nblk >> 3; \
    int n16 = in_f >> 4; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][16]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int sb = 0; sb < n16; sb++) { \
        int blk32 = sb >> 1; \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                long super = (long)col * spr + (blk32 >> 3); \
                int g = (blk32 & 7) * 2 + (sb & 1); \
                const unsigned char* b = w + super * 84; \
                float d = rf16b(b + 80), dmin = rf16b(b + 82); \
                int sc = b[g] & 0x0F, mm = b[g] >> 4; \
                wsc[c] = d * (float)sc; \
                wmn[c] = dmin * (float)(-mm); \
                const unsigned char* qb = (b + 16) + (g >> 3) * 32 + (g & 1) * 16; \
                int sh = 2 * ((g >> 1) & 3); \
                for (int rr = 0; rr < 16; rr++) wc[c][rr] = (signed char)((qb[rr] >> sh) & 3); \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; for (int rr = 0; rr < 16; rr++) wc[c][rr] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long koff = (long)row_in * in_f + (long)sb * 16; \
            i4v a = load_a(qx, row_in, m, koff); \
            i8v sumacc = wmma_dot(a, ones, (i8v){0,0,0,0,0,0,0,0}); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = wmma_dot(a, pack16(wc[c]), (i8v){0,0,0,0,0,0,0,0}); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk32] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q3_K: 256/super-block, 16 sub-blocks of 16 (= 1 K-tile each), packed 6-bit scale, code 0..7. ──
#define GEN_WMMA_Q3K(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    int spr = nblk >> 3; \
    int n16 = in_f >> 4; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][16]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int sb = 0; sb < n16; sb++) { \
        int blk32 = sb >> 1; \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                long super = (long)col * spr + (blk32 >> 3); \
                int g = (blk32 & 7) * 2 + (sb & 1); \
                const unsigned char* b = w + super * 110; \
                float d = rf16b(b + 108); \
                int sc = q3k_sc6(b + 96, g); \
                wsc[c] = d * (float)sc; \
                wmn[c] = d * (float)(-4 * sc); \
                const unsigned char* qb = (b + 32) + (g >> 3) * 32 + (g & 1) * 16; \
                const unsigned char* hb = b + (g & 1) * 16; \
                int sh = 2 * ((g >> 1) & 3), hsh = g >> 1; \
                for (int rr = 0; rr < 16; rr++) \
                    wc[c][rr] = (signed char)(((qb[rr] >> sh) & 3) | (((hb[rr] >> hsh) & 1) << 2)); \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; for (int rr = 0; rr < 16; rr++) wc[c][rr] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long koff = (long)row_in * in_f + (long)sb * 16; \
            i4v a = load_a(qx, row_in, m, koff); \
            i8v sumacc = wmma_dot(a, ones, (i8v){0,0,0,0,0,0,0,0}); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = wmma_dot(a, pack16(wc[c]), (i8v){0,0,0,0,0,0,0,0}); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk32] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q5_0: 32 elems/block = 2 K-tiles, scale d, min d·(−16), code 0..31. Q8_0 shape + Q4_K-style min. ──
#define GEN_WMMA_Q50(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][32]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                const unsigned char* b = w + ((long)col * nblk + blk) * 22; \
                float d = rf16b(b); \
                wsc[c] = d; \
                wmn[c] = d * (float)(-16); \
                unsigned int qh = (unsigned int)b[2] | ((unsigned int)b[3] << 8) \
                                | ((unsigned int)b[4] << 16) | ((unsigned int)b[5] << 24); \
                const unsigned char* qs = b + 6; \
                for (int p = 0; p < 16; p++) { \
                    int xh0 = (int)(((qh >> p) << 4) & 0x10); \
                    int xh1 = (int)((qh >> (p + 12)) & 0x10); \
                    wc[c][p]      = (signed char)((qs[p] & 0x0F) | xh0); \
                    wc[c][p + 16] = (signed char)((qs[p] >> 4) | xh1); \
                } \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; for (int p = 0; p < 32; p++) wc[c][p] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)blk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            i8v sumacc = {0,0,0,0,0,0,0,0}; \
            sumacc = wmma_dot(a0, ones, sumacc); sumacc = wmma_dot(a1, ones, sumacc); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, pack16(wc[c]),      dotacc); \
                dotacc = wmma_dot(a1, pack16(wc[c] + 16), dotacc); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q4_0 / Q4_1 / Q5_1: 32 elems/block = 2 K-tiles, one f16 scale per block (the Q5_0 shape). ──
// ONE macro for all three — they share Q5_0's geometry and differ only in three compile-time
// constants, so triplicating the 45-line body would only invite the three copies to drift:
//   `BPB`    bytes/block (18 / 20 / 24),
//   `HASMIN` 1 ⇒ the header is [d][m] and the ones-dot is weighted by that per-block `m`;
//            0 ⇒ the header is [d] alone and the weight is the constant `d·(−8)` (Q4_0),
//   `FIVEBIT` 1 ⇒ a 4-byte `qh` bitfield follows the header and supplies each code's 5th bit (Q5_1).
// `qs` therefore starts at `2 + 2·HASMIN + 4·FIVEBIT`. The decode is byte-identical to the matching
// `linear_i8_*` / `deq_*` (same nibble halves, same `qh` bit), so the per-output f32 accumulation
// order matches the GEMV tier exactly.
#define GEN_WMMA_R32(NAME, RM, CN, BPB, HASMIN, FIVEBIT) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][32]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                const unsigned char* b = w + ((long)col * nblk + blk) * (BPB); \
                float d = rf16b(b); \
                wsc[c] = d; \
                wmn[c] = (HASMIN) ? rf16b(b + 2) : d * (float)(-8); \
                const unsigned char* qhp = b + 2 + 2 * (HASMIN); \
                unsigned int qh = (FIVEBIT) \
                    ? ((unsigned int)qhp[0] | ((unsigned int)qhp[1] << 8) \
                       | ((unsigned int)qhp[2] << 16) | ((unsigned int)qhp[3] << 24)) \
                    : 0u; \
                const unsigned char* qs = qhp + 4 * (FIVEBIT); \
                for (int p = 0; p < 16; p++) { \
                    int xh0 = (FIVEBIT) ? (int)(((qh >> p) << 4) & 0x10) : 0; \
                    int xh1 = (FIVEBIT) ? (int)((qh >> (p + 12)) & 0x10) : 0; \
                    wc[c][p]      = (signed char)((qs[p] & 0x0F) | xh0); \
                    wc[c][p + 16] = (signed char)((qs[p] >> 4) | xh1); \
                } \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; for (int p = 0; p < 32; p++) wc[c][p] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)blk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            i8v sumacc = {0,0,0,0,0,0,0,0}; \
            sumacc = wmma_dot(a0, ones, sumacc); sumacc = wmma_dot(a1, ones, sumacc); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, pack16(wc[c]),      dotacc); \
                dotacc = wmma_dot(a1, pack16(wc[c] + 16), dotacc); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}
#define GEN_WMMA_Q40(NAME, RM, CN) GEN_WMMA_R32(NAME, RM, CN, 18, 0, 0)
#define GEN_WMMA_Q41(NAME, RM, CN) GEN_WMMA_R32(NAME, RM, CN, 20, 1, 0)
#define GEN_WMMA_Q51(NAME, RM, CN) GEN_WMMA_R32(NAME, RM, CN, 24, 1, 1)

// ── IQ4_NL / IQ4_XS (R4): codebook 4-bit, 32 elems per scale = 2 K-tiles, NO min term. ──
// ONE macro for both, as `GEN_WMMA_R32` is one for the three legacy round quants: they share the
// whole loop (identical nibble→`kv_iq4nl` decode, identical 2×wmma_dot, identical epilogue) and
// differ ONLY in where the per-32-element scale comes from, which `XS` (a literal, so the branch
// folds at compile time) selects:
//   XS=0  IQ4_NL, 18 B per 32 elements     → scale = `d`,           qs at +2
//   XS=1  IQ4_XS, 136 B per 256 elements   → scale = `d·(ls − 32)`, qs at +8 + 16·ib
// Structurally this is `GEN_WMMA_Q80`'s shape, NOT `GEN_WMMA_R32`'s: like Q8_0 the weight operand is
// already signed (the codebook value), so there is no ones-dot `sumacc` / `wmn` at all — which is
// also why the R32 body could not simply be reused with another flag. The decode is byte-identical
// to `linear_i8_iq4nl`/`linear_i8_iq4xs` and to `deq_iq4nl`/`deq_iq4xs`.
#define GEN_WMMA_IQ4(NAME, RM, CN, XS) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    int spr = nblk >> 3; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][32]; \
    float wsc[CN]; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                const unsigned char* qs; \
                if (XS) { \
                    const unsigned char* b = w + ((long)col * spr + (blk >> 3)) * 136; \
                    int ib = blk & 7; \
                    unsigned int sh = (unsigned int)b[2] | ((unsigned int)b[3] << 8); \
                    int lo = (b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F; \
                    int hi = (int)((sh >> (2 * ib)) & 3u); \
                    wsc[c] = rf16b(b) * (float)((lo | (hi << 4)) - 32); \
                    qs = b + 8 + 16 * ib; \
                } else { \
                    const unsigned char* b = w + ((long)col * nblk + blk) * 18; \
                    wsc[c] = rf16b(b); \
                    qs = b + 2; \
                } \
                for (int p = 0; p < 16; p++) { \
                    wc[c][p]      = (signed char)kv_iq4nl(qs[p] & 0x0F); \
                    wc[c][p + 16] = (signed char)kv_iq4nl(qs[p] >> 4); \
                } \
            } else { wsc[c] = 0.0f; for (int p = 0; p < 32; p++) wc[c][p] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)blk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, pack16(wc[c]),      dotacc); \
                dotacc = wmma_dot(a1, pack16(wc[c] + 16), dotacc); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * wsc[c] * (float)dotacc[e]; \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}
#define GEN_WMMA_IQ4NL(NAME, RM, CN) GEN_WMMA_IQ4(NAME, RM, CN, 0)
#define GEN_WMMA_IQ4XS(NAME, RM, CN) GEN_WMMA_IQ4(NAME, RM, CN, 1)

// ── The `wdec_*` seam — ONE body for all ten formats on it. ──
// R5's grid quants (IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S) and R6's IQ1 (IQ1_S/IQ1_M) and ternary
// (TQ1_0/TQ2_0/Q2_0) quants differ only inside `wdec_##FMT` (NATIVE_DECODE), which hands back this
// 32-block's 32 signed codes and the two scales its 16-element halves carry — so unlike
// `GEN_WMMA_IQ4`'s `XS` flag, there is no per-format branch left in the body at all, and the same
// decoder serves the `linear_i8_*` GEMV tier (they cannot drift).
//
// Structurally Q8_0's shape, not `GEN_WMMA_R32`'s: every code on this seam is already signed, so
// there is no ones-dot `sumacc`/`wmn` anywhere. The ONE difference from `GEN_WMMA_Q80` is that the
// two K-tiles of a 32-block are scaled INDEPENDENTLY (`ws0`/`ws1`) instead of sharing one `wsc` —
// IQ2_XS, IQ2_S and IQ1_M put a separate scale on each half. The per-32-scale formats pass the same
// value twice, which costs one extra f32 multiply per (block, column) and keeps one body.
#define GEN_WMMA_WDEC(NAME, RM, CN, FMT) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    signed char wc[CN][32]; \
    float ws0[CN], ws1[CN]; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                wdec_##FMT(w, (long)col, nblk, blk, wc[c], &ws0[c], &ws1[c]); \
            } else { ws0[c] = 0.0f; ws1[c] = 0.0f; for (int p = 0; p < 32; p++) wc[c][p] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)blk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            for (int c = 0; c < (CN); c++) { \
                i8v d0 = wmma_dot(a0, pack16(wc[c]),      (i8v){0,0,0,0,0,0,0,0}); \
                i8v d1 = wmma_dot(a1, pack16(wc[c] + 16), (i8v){0,0,0,0,0,0,0,0}); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * (ws0[c] * (float)d0[e] + ws1[c] * (float)d1[e]); \
                } \
            } \
        } \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q4_K PIPELINED (Slice 27): software-prefetched double-buffered weight nibbles. ──
// Same math / accumulation order as GEN_WMMA_Q4K (bit-faithful, goldens unmoved) — the ONLY change is
// scheduling: the 32 packed nibble bytes for weight-block blk+1 are issued as global `buffer_load`s
// into a second register buffer BEFORE the decode+WMMA of block blk consumes the first, so the load
// latency of blk+1 overlaps the matrix math of blk instead of serializing in front of it (the
// decode→WMMA dependency chain that ceilings Slice-25). Header/scale reads (d,dmin,k4) stay inline —
// 16 B/super, L2-hot across the 8 sub-blocks — and are cheap relative to the 128 B/super of nibbles.
//
// The ping-pong buffers `qA`/`qB` are COMPILE-TIME NAMED (the K-loop is unrolled by 2 so each stage
// names its buffer statically). A runtime-indexed `qbuf[cur]` measured 3.5× SLOWER: HIP lowers a
// dynamically-indexed local array to scratch (private memory), turning every nibble access into a
// global round-trip. With named buffers the prefetch schedule wins +7..13% GFLOP/s (2x1) on gfx1100.
// Only the 2x1 tile is shipped: the 2x2 variant (already 192 VGPR in Slice-25) plus the qA/qB
// prefetch buffers exceeds the register file and page-faults on launch, and 2x1-pipe already beats
// un-pipelined 2x2 on the wide-N shapes where 2x2 used to win — so 2x1-pipe supersedes both.
#define LD_Q4K_NIB(DST, COL, BLK) \
    do { \
        if ((COL) < out_f) { \
            long super = (long)(COL) * spr + ((BLK) >> 3); \
            int s_ = (BLK) & 7; \
            const unsigned int* qp = (const unsigned int*)(w + super * 144 + 16 + (s_ >> 1) * 32); \
            for (int i = 0; i < 8; i++) (DST)[i] = qp[i]; \
        } else { for (int i = 0; i < 8; i++) (DST)[i] = 0; } \
    } while (0)
// One K-block step: decode the prefetched nibble buffer QB for block BLK, then the RM×CN WMMA +
// bit-faithful f32 accumulation. QB is a compile-time-named buffer (qA/qB), so it stays in VGPRs —
// a runtime-indexed `qbuf[cur]` would spill the array to scratch and tank throughput.
#define Q4K_STEP(QB, BLK, RM, CN) \
    do { \
        int kblk = (BLK); \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                long super = (long)col * spr + (kblk >> 3); \
                int s = kblk & 7; \
                const unsigned char* b = w + super * 144; \
                float d = rf16b(b), dmin = rf16b(b + 2); \
                int sc, mm; k4(b + 4, s, &sc, &mm); \
                wsc[c] = d * (float)sc; \
                wmn[c] = dmin * (float)(-mm); \
                const unsigned char* qb = (const unsigned char*)(QB)[c]; \
                int hi = s & 1; \
                for (int p = 0; p < 32; p++) wc[c][p] = (signed char)(hi ? (qb[p] >> 4) : (qb[p] & 0x0F)); \
            } else { wsc[c] = 0.0f; wmn[c] = 0.0f; for (int p = 0; p < 32; p++) wc[c][p] = 0; } \
        } \
        for (int r = 0; r < (RM); r++) { \
            int row_in = row_base + r * 16 + (lane & 15); \
            long arow = (long)row_in * in_f + (long)kblk * 32; \
            i4v a0 = load_a(qx, row_in, m, arow), a1 = load_a(qx, row_in, m, arow + 16); \
            i8v sumacc = {0,0,0,0,0,0,0,0}; \
            sumacc = wmma_dot(a0, ones, sumacc); sumacc = wmma_dot(a1, ones, sumacc); \
            for (int c = 0; c < (CN); c++) { \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, pack16(wc[c]),      dotacc); \
                dotacc = wmma_dot(a1, pack16(wc[c] + 16), dotacc); \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + kblk] : 0.0f; \
                    acc[r][c][e] += axs * (wsc[c] * (float)dotacc[e] + wmn[c] * (float)sumacc[e]); \
                } \
            } \
        } \
    } while (0)
#define GEN_WMMA_Q4K_PIPE(NAME, RM, CN) \
extern "C" __global__ void NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    int lane = threadIdx.x; \
    int half = lane >> 4; \
    int col_base = blockIdx.x * (16 * (CN)); \
    int row_base = blockIdx.y * (16 * (RM)); \
    int nblk = in_f >> 5; \
    int spr = nblk >> 3; \
    float acc[RM][CN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    unsigned int qA[CN][8], qB[CN][8]; \
    signed char wc[CN][32]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int c = 0; c < (CN); c++) { int col = col_base + c * 16 + (lane & 15); LD_Q4K_NIB(qA[c], col, 0); } \
    for (int blk = 0; blk < nblk; blk += 2) { \
        if (blk + 1 < nblk) { for (int c = 0; c < (CN); c++) { int col = col_base + c * 16 + (lane & 15); LD_Q4K_NIB(qB[c], col, blk + 1); } } \
        Q4K_STEP(qA, blk, RM, CN); \
        if (blk + 2 < nblk) { for (int c = 0; c < (CN); c++) { int col = col_base + c * 16 + (lane & 15); LD_Q4K_NIB(qA[c], col, blk + 2); } } \
        if (blk + 1 < nblk) Q4K_STEP(qB, blk + 1, RM, CN); \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (CN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + r * 16 + 2 * e + half; \
        int col = col_base + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q4_K COOPERATIVE decode-once GEMM (Slice 28): multi-warp threadblock, LDS-shared weight tile. ──
// The single-wave GEN_WMMA_Q4K* / _PIPE kernels re-decode each Q4_K weight column once per output-row
// BLOCK: for m=512 that is ~m/32 ≈ 16 redundant decodes of the SAME nibbles (Slice-27 diagnosis). Here
// a threadblock of WM*WN wave32s cooperatively owns a BM×BN output tile (BM = 16*WM*RM rows, BN =
// 16*WN*RN cols) and decodes the BN-column weight tile into LDS int8 exactly ONCE per BK=32 K-step —
// then every warp reuses it (decode-once-reuse, the llama.cpp MMQ threadblock pattern adapted to RDNA3
// wave32). Activation int8 is staged into LDS the same way (shared across the BN columns).
//
// Per K-step (blk = one Q4_K 32-elem sub-block, so wsc/wmn are constant across BK for a column):
//   1. threads 0..BN cooperatively decode column `col_base+cl`'s sub-block into wLDS[cl][32] + wscL/wmnL
//   2. all threads cooperatively stage the BM×32 int8 activation tile into aLDS
//   3. __syncthreads
//   4. each warp runs the RM×RN WMMA sub-tile over the shared LDS tiles, min-term ones-dot per row-tile
//   5. __syncthreads before the next K-step overwrites LDS (single-buffered — decode-once is the win,
//      not the pipeline; Slice-27 already covered intra-wave load/decode overlap)
//
// Bit-faithful (goldens MUST NOT move): identical int8 weight codes (same k4/nibble math as
// GEN_WMMA_Q4K), identical int8 activation codes (same `quant_i8_32` qx), and identical per-output
// f32 accumulation — every dst[re,col] is Σ_blk axs·(wsc·dot + wmn·sum) summed in the SAME block order
// 0..nblk, dot = int32 over the SAME 32 K-codes (2 WMMA of 16). The cooperative staging changes only
// WHERE the operands are read from (LDS vs global), never the arithmetic or its order — pure scheduling.
// LDS reads are 4-byte-aligned (tile rows are 32-wide → offsets are ×32/×16). Tile buffers live in LDS
// (compile-time-sized __shared__) and the per-warp accumulators in registers (compile-time-indexed) —
// no dynamically-indexed local array (Slice-27 lesson: those lower to scratch and tank throughput).
#define GEN_WMMA_Q4K_COOP(NAME, WM, WN, RM, RN) \
extern "C" __global__ void __launch_bounds__((WM)*(WN)*32) NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    const int BM = 16*(WM)*(RM), BN = 16*(WN)*(RN); \
    __shared__ signed char aLDS[16*(WM)*(RM) * 32]; \
    __shared__ signed char wLDS[16*(WN)*(RN) * 32]; \
    __shared__ float wscL[16*(WN)*(RN)]; \
    __shared__ float wmnL[16*(WN)*(RN)]; \
    int tid = threadIdx.x, nthreads = (WM)*(WN)*32; \
    int warp = tid >> 5, lane = tid & 31, half = lane >> 4; \
    int warp_m = warp / (WN), warp_n = warp % (WN); \
    int col_base = blockIdx.x * BN, row_base = blockIdx.y * BM; \
    int nblk = in_f >> 5, spr = nblk >> 3; \
    float acc[RM][RN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (RN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int cl = tid; cl < BN; cl += nthreads) { \
            int col = col_base + cl; \
            if (col < out_f) { \
                long super = (long)col * spr + (blk >> 3); \
                int s = blk & 7; \
                const unsigned char* b = w + super * 144; \
                float d = rf16b(b), dmin = rf16b(b + 2); \
                int sc, mm; k4(b + 4, s, &sc, &mm); \
                wscL[cl] = d * (float)sc; \
                wmnL[cl] = dmin * (float)(-mm); \
                const unsigned char* qbase = (b + 16) + (s >> 1) * 32; \
                int hi = s & 1; \
                for (int p = 0; p < 32; p++) wLDS[cl * 32 + p] = (signed char)(hi ? (qbase[p] >> 4) : (qbase[p] & 0x0F)); \
            } else { wscL[cl] = 0.0f; wmnL[cl] = 0.0f; for (int p = 0; p < 32; p++) wLDS[cl * 32 + p] = 0; } \
        } \
        for (int idx = tid; idx < BM * 32; idx += nthreads) { \
            int rl = idx >> 5, kk = idx & 31; \
            int row = row_base + rl; \
            aLDS[idx] = (row < m) ? qx[(long)row * in_f + (long)blk * 32 + kk] : 0; \
        } \
        __syncthreads(); \
        for (int r = 0; r < (RM); r++) { \
            int rl = warp_m * 16 * (RM) + r * 16 + (lane & 15); \
            i4v a0 = pack16(aLDS + rl * 32), a1 = pack16(aLDS + rl * 32 + 16); \
            i8v sumacc = {0,0,0,0,0,0,0,0}; \
            sumacc = wmma_dot(a0, ones, sumacc); sumacc = wmma_dot(a1, ones, sumacc); \
            for (int c = 0; c < (RN); c++) { \
                int cl = warp_n * 16 * (RN) + c * 16 + (lane & 15); \
                i4v b0 = pack16(wLDS + cl * 32), b1 = pack16(wLDS + cl * 32 + 16); \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, b0, dotacc); dotacc = wmma_dot(a1, b1, dotacc); \
                float wsc = wscL[cl], wmn = wmnL[cl]; \
                for (int e = 0; e < 8; e++) { \
                    int re = row_base + warp_m * 16 * (RM) + r * 16 + 2 * e + half; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * (wsc * (float)dotacc[e] + wmn * (float)sumacc[e]); \
                } \
            } \
        } \
        __syncthreads(); \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (RN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + warp_m * 16 * (RM) + r * 16 + 2 * e + half; \
        int col = col_base + warp_n * 16 * (RN) + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// ── Q4_K COOPERATIVE, min-term via LDS row-sum (Slice 28, `_rs`). ──────────────────────────────────
// The single-wave WMMA kernels spend HALF their matrix-core ops on the Q4_K min term: `sumacc =
// wmma_dot(a, ones)` is a full 16×16×16 matmul against an all-ones B just to get Σ_k a[row,k]. On
// gfx1100 the isolated GEMM is matrix-core-bound, so that doubles the WMMA cost. Here the per-row,
// per-block activation sum is instead reduced ONCE into LDS (`rsLDS[row]`, a plain int add over the
// 32 int8 codes — order-independent, so bit-identical to the ones-dot int32 result), and the min
// contribution becomes `wmn * rsLDS[row]` — no WMMA. That halves the matrix-core ops (only the real
// a·w dot stays on WMMA). Combined with the decode-once weight tile, this is the llama.cpp MMQ shape.
// Bit-faithful: same int8 codes, same Σ_blk axs·(wsc·dot + wmn·rowsum) per-output accumulation order.
#define GEN_WMMA_Q4K_COOP_RS(NAME, WM, WN, RM, RN) \
extern "C" __global__ void __launch_bounds__((WM)*(WN)*32) NAME( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ w, float* __restrict__ dst, \
    int m, int in_f, int out_f) { \
    const int BM = 16*(WM)*(RM), BN = 16*(WN)*(RN); \
    __shared__ signed char aLDS[16*(WM)*(RM) * 32]; \
    __shared__ signed char wLDS[16*(WN)*(RN) * 32]; \
    __shared__ int rsLDS[16*(WM)*(RM)]; \
    __shared__ float wscL[16*(WN)*(RN)]; \
    __shared__ float wmnL[16*(WN)*(RN)]; \
    int tid = threadIdx.x, nthreads = (WM)*(WN)*32; \
    int warp = tid >> 5, lane = tid & 31, half = lane >> 4; \
    int warp_m = warp / (WN), warp_n = warp % (WN); \
    int col_base = blockIdx.x * BN, row_base = blockIdx.y * BM; \
    int nblk = in_f >> 5, spr = nblk >> 3; \
    float acc[RM][RN][8]; \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (RN); c++) for (int e = 0; e < 8; e++) acc[r][c][e] = 0.0f; \
    for (int blk = 0; blk < nblk; blk++) { \
        for (int cl = tid; cl < BN; cl += nthreads) { \
            int col = col_base + cl; \
            if (col < out_f) { \
                long super = (long)col * spr + (blk >> 3); \
                int s = blk & 7; \
                const unsigned char* b = w + super * 144; \
                float d = rf16b(b), dmin = rf16b(b + 2); \
                int sc, mm; k4(b + 4, s, &sc, &mm); \
                wscL[cl] = d * (float)sc; \
                wmnL[cl] = dmin * (float)(-mm); \
                const unsigned char* qbase = (b + 16) + (s >> 1) * 32; \
                int hi = s & 1; \
                for (int p = 0; p < 32; p++) wLDS[cl * 32 + p] = (signed char)(hi ? (qbase[p] >> 4) : (qbase[p] & 0x0F)); \
            } else { wscL[cl] = 0.0f; wmnL[cl] = 0.0f; for (int p = 0; p < 32; p++) wLDS[cl * 32 + p] = 0; } \
        } \
        for (int idx = tid; idx < BM * 32; idx += nthreads) { \
            int rl = idx >> 5, kk = idx & 31; \
            int row = row_base + rl; \
            aLDS[idx] = (row < m) ? qx[(long)row * in_f + (long)blk * 32 + kk] : 0; \
        } \
        for (int rl = tid; rl < BM; rl += nthreads) { \
            int row = row_base + rl; \
            int s = 0; \
            if (row < m) { const signed char* ar = qx + (long)row * in_f + (long)blk * 32; for (int k = 0; k < 32; k++) s += (int)ar[k]; } \
            rsLDS[rl] = s; \
        } \
        __syncthreads(); \
        for (int r = 0; r < (RM); r++) { \
            int rl = warp_m * 16 * (RM) + r * 16 + (lane & 15); \
            i4v a0 = pack16(aLDS + rl * 32), a1 = pack16(aLDS + rl * 32 + 16); \
            for (int c = 0; c < (RN); c++) { \
                int cl = warp_n * 16 * (RN) + c * 16 + (lane & 15); \
                i4v b0 = pack16(wLDS + cl * 32), b1 = pack16(wLDS + cl * 32 + 16); \
                i8v dotacc = {0,0,0,0,0,0,0,0}; \
                dotacc = wmma_dot(a0, b0, dotacc); dotacc = wmma_dot(a1, b1, dotacc); \
                float wsc = wscL[cl], wmn = wmnL[cl]; \
                for (int e = 0; e < 8; e++) { \
                    int rr = warp_m * 16 * (RM) + r * 16 + 2 * e + half; \
                    int re = row_base + rr; \
                    float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f; \
                    acc[r][c][e] += axs * (wsc * (float)dotacc[e] + wmn * (float)rsLDS[rr]); \
                } \
            } \
        } \
        __syncthreads(); \
    } \
    for (int r = 0; r < (RM); r++) for (int c = 0; c < (RN); c++) for (int e = 0; e < 8; e++) { \
        int re = row_base + warp_m * 16 * (RM) + r * 16 + 2 * e + half; \
        int col = col_base + warp_n * 16 * (RN) + c * 16 + (lane & 15); \
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[r][c][e]; \
    } \
}

// Cooperative tile instances swept on the isolated-GEMM micro-bench (`examples/wmma_bench`, Q4_K).
// OPT-IN via `INFR_ROCM_COOP=1`; `INFR_ROCM_COOP_TILE=<name>` picks the tile (default `128x64`). This
// family MEASURED A REGRESSION on gfx1100 (occupancy/barrier-bound, not decode-bound — see exec.rs
// `q4k_coop_kernel` and docs/perf.md), so the shipped Q4_K prefill default remains the Slice-27 pipe.
// Naming: `<BM>x<BN>_w<NWARPS>`. LDS/single-buffer = BM*32 + BN*32 + (BN*8) bytes; the wide RM=RN=2
// tiles are VGPR-bound to 7 waves/SIMD (`-Rpass-analysis`), which is a chunk of the regression.
GEN_WMMA_Q4K_COOP_RS(wmma_i8_q4k_coop_rs_128x64_w8, 4, 2, 2, 2)
GEN_WMMA_Q4K_COOP_RS(wmma_i8_q4k_coop_rs_128x32_w8, 8, 1, 1, 2)
GEN_WMMA_Q4K_COOP_RS(wmma_i8_q4k_coop_rs_64x64_w4,  2, 2, 2, 2)
GEN_WMMA_Q4K_COOP_RS(wmma_i8_q4k_coop_rs_64x32_w8,  4, 2, 1, 1)
GEN_WMMA_Q4K_COOP(wmma_i8_q4k_coop_128x64_w8,  4, 2, 2, 2)
GEN_WMMA_Q4K_COOP(wmma_i8_q4k_coop_128x32_w8,  8, 1, 1, 2)
GEN_WMMA_Q4K_COOP(wmma_i8_q4k_coop_64x64_w4,   2, 2, 2, 2)
GEN_WMMA_Q4K_COOP(wmma_i8_q4k_coop_64x32_w8,   4, 2, 1, 1)
GEN_WMMA_Q4K_COOP(wmma_i8_q4k_coop_128x128_w8, 4, 2, 2, 4)
GEN_WMMA_Q4K_COOP(wmma_i8_q4k_coop_256x64_w16, 8, 2, 2, 2)

// Tile instances kept after the Slice-25 sweep: `_2x1` (block M) and `_2x2` (block M+N) are the two
// shipped by the auto tier; `_1x1` is the un-blocked Slice-15 tiling, retained as the A/B reference
// for `INFR_ROCM_WMMA_TILE=1x1`. The pure-N tiles (`1x2`/`1x4`) measured strictly worse on every
// shape and were dropped. `INFR_ROCM_WMMA_TILE=RxC` selects at dispatch.
GEN_WMMA_Q4K_PIPE(wmma_i8_q4k_pipe_2x1, 2, 1)
GEN_WMMA_Q80(wmma_i8_q80_1x1, 1, 1)
GEN_WMMA_Q80(wmma_i8_q80_2x1, 2, 1)
GEN_WMMA_Q80(wmma_i8_q80_2x2, 2, 2)
GEN_WMMA_Q2K(wmma_i8_q2k_1x1, 1, 1)
GEN_WMMA_Q2K(wmma_i8_q2k_2x1, 2, 1)
GEN_WMMA_Q2K(wmma_i8_q2k_2x2, 2, 2)
GEN_WMMA_Q3K(wmma_i8_q3k_1x1, 1, 1)
GEN_WMMA_Q3K(wmma_i8_q3k_2x1, 2, 1)
GEN_WMMA_Q3K(wmma_i8_q3k_2x2, 2, 2)
GEN_WMMA_Q4K(wmma_i8_q4k_1x1, 1, 1)
GEN_WMMA_Q4K(wmma_i8_q4k_2x1, 2, 1)
GEN_WMMA_Q4K(wmma_i8_q4k_2x2, 2, 2)
GEN_WMMA_Q5K(wmma_i8_q5k_1x1, 1, 1)
GEN_WMMA_Q5K(wmma_i8_q5k_2x1, 2, 1)
GEN_WMMA_Q5K(wmma_i8_q5k_2x2, 2, 2)
GEN_WMMA_Q6K(wmma_i8_q6k_1x1, 1, 1)
GEN_WMMA_Q6K(wmma_i8_q6k_2x1, 2, 1)
GEN_WMMA_Q6K(wmma_i8_q6k_2x2, 2, 2)
GEN_WMMA_Q50(wmma_i8_q50_1x1, 1, 1)
GEN_WMMA_Q50(wmma_i8_q50_2x1, 2, 1)
GEN_WMMA_Q50(wmma_i8_q50_2x2, 2, 2)
// R3 legacy 32-block round quants. Plain tier only, for the same reason Q5_K/Q2_K/Q3_K are: the
// Slice-27 `_pipe` prefetch and the Slice-28 `_coop` family are Q4_K-only. The pipe buffer prefetches
// exactly the 32 packed NIBBLE bytes of a Q4_K sub-block and re-reads the header inline — Q4_1/Q5_1
// would have to carry a second f16 (and Q5_1 its `qh` word) through the ping-pong buffers, and these
// formats' headers are per-32-block (not amortized over 8 sub-blocks/super like Q4_K's), so the
// prefetch has far less to hide; the coop family is a measured gfx1100 regression regardless.
GEN_WMMA_Q40(wmma_i8_q40_1x1, 1, 1)
GEN_WMMA_Q40(wmma_i8_q40_2x1, 2, 1)
GEN_WMMA_Q40(wmma_i8_q40_2x2, 2, 2)
GEN_WMMA_Q41(wmma_i8_q41_1x1, 1, 1)
GEN_WMMA_Q41(wmma_i8_q41_2x1, 2, 1)
GEN_WMMA_Q41(wmma_i8_q41_2x2, 2, 2)
GEN_WMMA_Q51(wmma_i8_q51_1x1, 1, 1)
GEN_WMMA_Q51(wmma_i8_q51_2x1, 2, 1)
GEN_WMMA_Q51(wmma_i8_q51_2x2, 2, 2)
// R4 codebook quants. Plain tier only, for the same reason every format after Q4_K is: the Slice-27
// `_pipe` prefetch and the Slice-28 `_coop` family are Q4_K-only. The pipe buffer prefetches exactly
// Q4_K's 32 packed nibble bytes and re-reads the header inline; for IQ4 the nibbles are not the
// latency being hidden — the codebook gather is ALU (the same finding Vulkan's bytes-vs-speed sweep
// reached: IQ4_XS is 4.25 bpw, SMALLER than Q4_K's 4.5, yet 1.55-2.1x slower per dispatch at matched
// shapes, i.e. gather-ALU bound, not DRAM bound), so prefetching them buys nothing. The coop family
// is a measured gfx1100 regression regardless.
GEN_WMMA_IQ4NL(wmma_i8_iq4nl_1x1, 1, 1)
GEN_WMMA_IQ4NL(wmma_i8_iq4nl_2x1, 2, 1)
GEN_WMMA_IQ4NL(wmma_i8_iq4nl_2x2, 2, 2)
GEN_WMMA_IQ4XS(wmma_i8_iq4xs_1x1, 1, 1)
GEN_WMMA_IQ4XS(wmma_i8_iq4xs_2x1, 2, 1)
GEN_WMMA_IQ4XS(wmma_i8_iq4xs_2x2, 2, 2)
// R5 grid quants. Plain tier only, for the same reason every format after Q4_K is: the Slice-27
// `_pipe` prefetch and the Slice-28 `_coop` family are Q4_K-only, and the coop family is a measured
// gfx1100 regression regardless. R4's extra argument against the pipe applies here with MORE force,
// not less: the pipe prefetches Q4_K's 32 packed nibble bytes at a statically known offset, but for
// a grid quant the reads on the critical path are the GRID entries — table addresses that are not
// known until the block's own indices have been fetched and unpacked, so a fixed-shape prefetch
// cannot reach them at all. The measured end-to-end decode agrees that these are not purely
// DRAM-bound: at 2.06 bpw IQ2_XXS streams under HALF Q4_K's 4.5 bpw of weight bytes yet runs only
// 1.16x its decode rate (147.5 vs 127.3 t/s on Qwen3-0.6B), so the gather/ALU is a real share.
GEN_WMMA_WDEC(wmma_i8_iq2xxs_1x1, 1, 1, iq2xxs)
GEN_WMMA_WDEC(wmma_i8_iq2xxs_2x1, 2, 1, iq2xxs)
GEN_WMMA_WDEC(wmma_i8_iq2xxs_2x2, 2, 2, iq2xxs)
GEN_WMMA_WDEC(wmma_i8_iq2xs_1x1, 1, 1, iq2xs)
GEN_WMMA_WDEC(wmma_i8_iq2xs_2x1, 2, 1, iq2xs)
GEN_WMMA_WDEC(wmma_i8_iq2xs_2x2, 2, 2, iq2xs)
GEN_WMMA_WDEC(wmma_i8_iq2s_1x1, 1, 1, iq2s)
GEN_WMMA_WDEC(wmma_i8_iq2s_2x1, 2, 1, iq2s)
GEN_WMMA_WDEC(wmma_i8_iq2s_2x2, 2, 2, iq2s)
GEN_WMMA_WDEC(wmma_i8_iq3xxs_1x1, 1, 1, iq3xxs)
GEN_WMMA_WDEC(wmma_i8_iq3xxs_2x1, 2, 1, iq3xxs)
GEN_WMMA_WDEC(wmma_i8_iq3xxs_2x2, 2, 2, iq3xxs)
GEN_WMMA_WDEC(wmma_i8_iq3s_1x1, 1, 1, iq3s)
GEN_WMMA_WDEC(wmma_i8_iq3s_2x1, 2, 1, iq3s)
GEN_WMMA_WDEC(wmma_i8_iq3s_2x2, 2, 2, iq3s)
// R6 IQ1 + ternary quants. Plain tier only, same reason as every format after Q4_K: the Slice-27
// `_pipe` prefetch and the Slice-28 `_coop` family are Q4_K-only, and coop is a measured gfx1100
// regression regardless. R5's extra argument against the pipe holds UNCHANGED for IQ1_S/IQ1_M —
// their critical-path reads are `g_iq1s` gathers whose addresses are not known until the block's
// own `qs`/`qh` have been fetched and unpacked, so a fixed-shape prefetch cannot reach them at all.
// For the ternary formats the argument is the opposite one and lands in the same place: there is no
// table and the whole "weight tile" is 32 (TQ2_0/Q2_0) or ≤48 (TQ1_0) bytes per 32 elements read at
// a statically known offset, so a prefetch has essentially nothing left to hide — TQ1_0 at 1.69 bpw
// and Q2_0 at 2.25 bpw are the lightest weight streams in the covered set, and what remains on the
// critical path is the base-3 digit ALU (TQ1_0) or a plain shift-and-mask (TQ2_0/Q2_0).
GEN_WMMA_WDEC(wmma_i8_iq1s_1x1, 1, 1, iq1s)
GEN_WMMA_WDEC(wmma_i8_iq1s_2x1, 2, 1, iq1s)
GEN_WMMA_WDEC(wmma_i8_iq1s_2x2, 2, 2, iq1s)
GEN_WMMA_WDEC(wmma_i8_iq1m_1x1, 1, 1, iq1m)
GEN_WMMA_WDEC(wmma_i8_iq1m_2x1, 2, 1, iq1m)
GEN_WMMA_WDEC(wmma_i8_iq1m_2x2, 2, 2, iq1m)
GEN_WMMA_WDEC(wmma_i8_tq10_1x1, 1, 1, tq10)
GEN_WMMA_WDEC(wmma_i8_tq10_2x1, 2, 1, tq10)
GEN_WMMA_WDEC(wmma_i8_tq10_2x2, 2, 2, tq10)
GEN_WMMA_WDEC(wmma_i8_tq20_1x1, 1, 1, tq20)
GEN_WMMA_WDEC(wmma_i8_tq20_2x1, 2, 1, tq20)
GEN_WMMA_WDEC(wmma_i8_tq20_2x2, 2, 2, tq20)
GEN_WMMA_WDEC(wmma_i8_q20_1x1, 1, 1, q20)
GEN_WMMA_WDEC(wmma_i8_q20_2x1, 2, 1, q20)
GEN_WMMA_WDEC(wmma_i8_q20_2x2, 2, 2, q20)
GEN_WMMA_WDEC(wmma_i8_mxfp4_1x1, 1, 1, mxfp4)
GEN_WMMA_WDEC(wmma_i8_mxfp4_2x1, 2, 1, mxfp4)
GEN_WMMA_WDEC(wmma_i8_mxfp4_2x2, 2, 2, mxfp4)
GEN_WMMA_WDEC(wmma_i8_nvfp4_1x1, 1, 1, nvfp4)
GEN_WMMA_WDEC(wmma_i8_nvfp4_2x1, 2, 1, nvfp4)
GEN_WMMA_WDEC(wmma_i8_nvfp4_2x2, 2, 2, nvfp4)
"#;

// ── GPU-side MoE top-k routing + device-driven expert dispatch (Slice 38) ────
//
// Moves the resident-MoE router off the host. The Phase-1..4 `MoeFfn` arm computed the router GEMV
// on the GPU, then read the `[rows, n_expert]` logits BACK to the host (`hipStreamSynchronize` +
// D2H), did top-k + gating in Rust, and host-dispatched the selected expert GEMVs — that per-layer
// readback stalled the decode pipeline every MoE layer. `moe_topk` performs the identical top-k +
// gating on the GPU into a `[rows, n_used]` (expert_id, gate_weight) device buffer, and the
// `*_routed_*` expert kernels read (expert_id, gate) from that buffer to index the RESIDENT expert
// bank + scale — a fixed host launch grid (`rows * n_used` slots) that needs no host knowledge of
// WHICH experts were picked, so no readback. The paged path still reads back (the pager must know
// which experts to page in), so this only fires for `!paged`.
//
// `moe_topk` is a faithful port of the Vulkan `moe_topk.comp`: one 128-lane block per token row,
// each lane owning `ceil(n_expert/128)` experts (MAX_CHUNKS=8 ⇒ up to 1024 experts), top-k by raw
// logit (both gating funcs are monotone in the logit → same selection), ties broken toward the
// lower expert id, then weights by softmax(max-shifted)/sigmoid with optional renorm × scale. This
// matches the host reference math (exec.rs) within f32 rounding.
//
// The `*_routed_*` kernels are twins of the host-routed expert kernels (`moe_ffn_expert`,
// `moe_ffn_expert_<gu>_<dn>`, `moe_gate_up_act_i8_<gu>`, `moe_down_i8_<dn>`) with the per-expert bank
// pointer computed IN-kernel from a bank base + a host-supplied per-expert byte/element stride and
// the device-read `expert_id`, and the routing weight read from the device buffer. `deq_*` /
// `i8acc_*` / `wave_sum32` / `idot4` are all defined in parts assembled before this one.
const MOE_ROUTING: &str = r#"
#define MOE_TOPK_MAX_CHUNKS 8
extern "C" __global__ void moe_topk(
    const float* __restrict__ logits,  /* [rows, n_expert] */
    int* __restrict__ ids,             /* [rows, n_used] */
    float* __restrict__ wts,           /* [rows, n_used] */
    int n_expert, int n_used, float scale, int gating, int norm_w
) {
    __shared__ float sval[128];
    __shared__ int sidx[128];
    __shared__ float glmax;
    int tok = blockIdx.x;
    int t = threadIdx.x;
    long lbase = (long)tok * n_expert;
    long obase = (long)tok * n_used;
    float llog[MOE_TOPK_MAX_CHUNKS];
    bool taken[MOE_TOPK_MAX_CHUNKS];
    for (int c = 0; c < MOE_TOPK_MAX_CHUNKS; c++) {
        int e = t + c * 128;
        llog[c] = (e < n_expert) ? logits[lbase + e] : -1e30f;
        taken[c] = false;
    }
    for (int k = 0; k < n_used; k++) {
        float bv = -1e30f; int be = 0x7fffffff;
        for (int c = 0; c < MOE_TOPK_MAX_CHUNKS; c++) {
            if (!taken[c] && llog[c] > bv) { bv = llog[c]; be = t + c * 128; }
        }
        sval[t] = bv; sidx[t] = be;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (t < s) {
                bool better = sval[t + s] > sval[t]
                    || (sval[t + s] == sval[t] && sidx[t + s] < sidx[t]);
                if (better) { sval[t] = sval[t + s]; sidx[t] = sidx[t + s]; }
            }
            __syncthreads();
        }
        int winner = sidx[0];
        if (t == 0) { ids[obase + k] = winner; if (k == 0) glmax = sval[0]; }
        for (int c = 0; c < MOE_TOPK_MAX_CHUNKS; c++) {
            if (t + c * 128 == winner) taken[c] = true;
        }
        __syncthreads();
    }
    if (t == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < n_used; k++) {
            float lg = logits[lbase + ids[obase + k]];
            wsum += (gating == 0) ? expf(lg - glmax) : (1.0f / (1.0f + expf(-lg)));
        }
        if (norm_w != 0) {
            wsum = fmaxf(wsum, 1e-20f);
            for (int k = 0; k < n_used; k++) {
                float lg = logits[lbase + ids[obase + k]];
                float sc = (gating == 0) ? expf(lg - glmax) : (1.0f / (1.0f + expf(-lg)));
                wts[obase + k] = sc / wsum * scale;
            }
        } else if (gating == 0) {
            float full = 0.0f;
            for (int e = 0; e < n_expert; e++) full += expf(logits[lbase + e] - glmax);
            full = fmaxf(full, 1e-20f);
            for (int k = 0; k < n_used; k++) {
                wts[obase + k] = expf(logits[lbase + ids[obase + k]] - glmax) / full * scale;
            }
        } else {
            for (int k = 0; k < n_used; k++) {
                float lg = logits[lbase + ids[obase + k]];
                wts[obase + k] = (1.0f / (1.0f + expf(-lg))) * scale;
            }
        }
    }
}

// f16 dequant-cache fallback, device-routed. Element strides into the __half banks.
extern "C" __global__ void moe_ffn_expert_routed(
    const float* __restrict__ x,             /* [ne] input row (host-advanced) */
    const __half* __restrict__ gate_base,    /* [n_expert, n_ff_exp, ne] */
    const __half* __restrict__ up_base,      /* [n_expert, n_ff_exp, ne] */
    const __half* __restrict__ down_base,    /* [n_expert, ne, n_ff_exp] */
    float* __restrict__ dst,                 /* [ne] out row (host-advanced) */
    int ne, int n_ff_exp, int act_type, int weight_before,
    const float* __restrict__ dsc_dev,       /* [n_expert] or null (⇒ 1.0) */
    const int* __restrict__ route_ids, const float* __restrict__ route_wts, int slot,
    long gate_estride, long up_estride, long down_estride,
    int fused, long fused_up_half_eoff
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ff_exp) return;
    int e = route_ids[slot];
    float weight = route_wts[slot];
    float dsc = dsc_dev ? dsc_dev[e] : 1.0f;
    const __half* gate_w = gate_base + (long)e * gate_estride;
    const __half* up_w = fused ? gate_base + (long)e * gate_estride + fused_up_half_eoff
                               : up_base + (long)e * up_estride;
    const __half* down_w = down_base + (long)e * down_estride;
    float wg = weight_before ? weight : 1.0f;
    float wo = weight_before ? 1.0f : weight;
    float g = 0.0f, u = 0.0f;
    for (int j = 0; j < ne; j++) {
        g += x[j] * __half2float(gate_w[(long)i * ne + j]);
        u += x[j] * __half2float(up_w[(long)i * ne + j]);
    }
    g *= wg; u *= wg;
    float a;
    if (act_type == 0) { a = g / (1.0f + expf(-g)); }
    else if (act_type == 1) { float x3 = g * g * g; a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3))); }
    else { a = 1.0f / (1.0f + expf(-g)); }
    float h = a * u * wo * dsc;
    for (int d = 0; d < ne; d++) {
        atomicAdd(&dst[d], h * __half2float(down_w[(long)d * n_ff_exp + i]));
    }
}

// Native in-kernel decode, device-routed. Per-expert byte strides into the raw quant banks.
#define GEN_MOE_FFN_ROUTED(GU, DN) \
extern "C" __global__ void moe_ffn_expert_routed_##GU##_##DN( \
    const float* __restrict__ x, \
    const unsigned char* __restrict__ gate_base, \
    const unsigned char* __restrict__ up_base, \
    const unsigned char* __restrict__ down_base, \
    float* __restrict__ dst, \
    int ne, int n_ff_exp, int act_type, int weight_before, \
    const float* __restrict__ dsc_dev, \
    const int* __restrict__ route_ids, const float* __restrict__ route_wts, int slot, \
    long gate_bstride, long up_bstride, long down_bstride, int fused, long fused_up_half_boff) { \
    int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= n_ff_exp) return; \
    int e = route_ids[slot]; \
    float weight = route_wts[slot]; \
    float dsc = dsc_dev ? dsc_dev[e] : 1.0f; \
    const unsigned char* gate_w = gate_base + (long)e * gate_bstride; \
    const unsigned char* up_w = fused ? gate_base + (long)e * gate_bstride + fused_up_half_boff \
                                      : up_base + (long)e * up_bstride; \
    const unsigned char* down_w = down_base + (long)e * down_bstride; \
    float wg = weight_before ? weight : 1.0f; \
    float wo = weight_before ? 1.0f : weight; \
    float g = 0.0f, u = 0.0f; \
    for (int j = 0; j < ne; j++) { \
        long idx = (long)i * ne + j; \
        g += x[j] * deq_##GU(gate_w, idx); \
        u += x[j] * deq_##GU(up_w, idx); \
    } \
    g *= wg; u *= wg; \
    float a; \
    if (act_type == 0) { a = g / (1.0f + expf(-g)); } \
    else if (act_type == 1) { float x3 = g * g * g; a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3))); } \
    else { a = 1.0f / (1.0f + expf(-g)); } \
    float h = a * u * wo * dsc; \
    for (int d = 0; d < ne; d++) { \
        atomicAdd(&dst[d], h * deq_##DN(down_w, (long)d * n_ff_exp + i)); \
    } \
}
GEN_MOE_FFN_ROUTED(q80, q80)
GEN_MOE_FFN_ROUTED(q80, q2k)
GEN_MOE_FFN_ROUTED(q80, q3k)
GEN_MOE_FFN_ROUTED(q80, q4k)
GEN_MOE_FFN_ROUTED(q80, q5k)
GEN_MOE_FFN_ROUTED(q80, q6k)
GEN_MOE_FFN_ROUTED(q2k, q80)
GEN_MOE_FFN_ROUTED(q2k, q2k)
GEN_MOE_FFN_ROUTED(q2k, q3k)
GEN_MOE_FFN_ROUTED(q2k, q4k)
GEN_MOE_FFN_ROUTED(q2k, q5k)
GEN_MOE_FFN_ROUTED(q2k, q6k)
GEN_MOE_FFN_ROUTED(q3k, q80)
GEN_MOE_FFN_ROUTED(q3k, q2k)
GEN_MOE_FFN_ROUTED(q3k, q3k)
GEN_MOE_FFN_ROUTED(q3k, q4k)
GEN_MOE_FFN_ROUTED(q3k, q5k)
GEN_MOE_FFN_ROUTED(q3k, q6k)
GEN_MOE_FFN_ROUTED(q4k, q80)
GEN_MOE_FFN_ROUTED(q4k, q2k)
GEN_MOE_FFN_ROUTED(q4k, q3k)
GEN_MOE_FFN_ROUTED(q4k, q4k)
GEN_MOE_FFN_ROUTED(q4k, q5k)
GEN_MOE_FFN_ROUTED(q4k, q6k)
GEN_MOE_FFN_ROUTED(q5k, q80)
GEN_MOE_FFN_ROUTED(q5k, q2k)
GEN_MOE_FFN_ROUTED(q5k, q3k)
GEN_MOE_FFN_ROUTED(q5k, q4k)
GEN_MOE_FFN_ROUTED(q5k, q5k)
GEN_MOE_FFN_ROUTED(q5k, q6k)
GEN_MOE_FFN_ROUTED(q6k, q80)
GEN_MOE_FFN_ROUTED(q6k, q2k)
GEN_MOE_FFN_ROUTED(q6k, q3k)
GEN_MOE_FFN_ROUTED(q6k, q4k)
GEN_MOE_FFN_ROUTED(q6k, q5k)
GEN_MOE_FFN_ROUTED(q6k, q6k)
GEN_MOE_FFN_ROUTED(q40, q40)
GEN_MOE_FFN_ROUTED(q40, q41)
GEN_MOE_FFN_ROUTED(q40, q51)
GEN_MOE_FFN_ROUTED(q40, q80)
GEN_MOE_FFN_ROUTED(q41, q40)
GEN_MOE_FFN_ROUTED(q41, q41)
GEN_MOE_FFN_ROUTED(q41, q51)
GEN_MOE_FFN_ROUTED(q41, q80)
GEN_MOE_FFN_ROUTED(q51, q40)
GEN_MOE_FFN_ROUTED(q51, q41)
GEN_MOE_FFN_ROUTED(q51, q51)
GEN_MOE_FFN_ROUTED(q51, q80)
GEN_MOE_FFN_ROUTED(iq4nl, iq4nl)
GEN_MOE_FFN_ROUTED(iq4nl, iq4xs)
GEN_MOE_FFN_ROUTED(iq4nl, q4k)
GEN_MOE_FFN_ROUTED(iq4nl, q5k)
GEN_MOE_FFN_ROUTED(iq4nl, q6k)
GEN_MOE_FFN_ROUTED(iq4nl, q80)
GEN_MOE_FFN_ROUTED(iq4xs, iq4nl)
GEN_MOE_FFN_ROUTED(iq4xs, iq4xs)
GEN_MOE_FFN_ROUTED(iq4xs, q4k)
GEN_MOE_FFN_ROUTED(iq4xs, q5k)
GEN_MOE_FFN_ROUTED(iq4xs, q6k)
GEN_MOE_FFN_ROUTED(iq4xs, q80)
GEN_MOE_FFN_ROUTED(q2k, iq4nl)
GEN_MOE_FFN_ROUTED(q3k, iq4nl)
GEN_MOE_FFN_ROUTED(iq2xxs, iq2s)
GEN_MOE_FFN_ROUTED(iq2xxs, iq3xxs)
GEN_MOE_FFN_ROUTED(iq2xxs, iq3s)
GEN_MOE_FFN_ROUTED(iq2xxs, iq4nl)
GEN_MOE_FFN_ROUTED(iq2xxs, iq4xs)
GEN_MOE_FFN_ROUTED(iq2xxs, q4k)
GEN_MOE_FFN_ROUTED(iq2xxs, q6k)
GEN_MOE_FFN_ROUTED(iq2xs, iq2s)
GEN_MOE_FFN_ROUTED(iq2xs, iq3xxs)
GEN_MOE_FFN_ROUTED(iq2xs, iq3s)
GEN_MOE_FFN_ROUTED(iq2xs, iq4nl)
GEN_MOE_FFN_ROUTED(iq2xs, iq4xs)
GEN_MOE_FFN_ROUTED(iq2xs, q4k)
GEN_MOE_FFN_ROUTED(iq2xs, q6k)
GEN_MOE_FFN_ROUTED(iq2s, iq2s)
GEN_MOE_FFN_ROUTED(iq2s, iq3xxs)
GEN_MOE_FFN_ROUTED(iq2s, iq3s)
GEN_MOE_FFN_ROUTED(iq2s, iq4nl)
GEN_MOE_FFN_ROUTED(iq2s, iq4xs)
GEN_MOE_FFN_ROUTED(iq2s, q4k)
GEN_MOE_FFN_ROUTED(iq2s, q6k)
GEN_MOE_FFN_ROUTED(iq3xxs, iq2s)
GEN_MOE_FFN_ROUTED(iq3xxs, iq3xxs)
GEN_MOE_FFN_ROUTED(iq3xxs, iq3s)
GEN_MOE_FFN_ROUTED(iq3xxs, iq4nl)
GEN_MOE_FFN_ROUTED(iq3xxs, iq4xs)
GEN_MOE_FFN_ROUTED(iq3xxs, q4k)
GEN_MOE_FFN_ROUTED(iq3xxs, q6k)
GEN_MOE_FFN_ROUTED(iq3s, iq2s)
GEN_MOE_FFN_ROUTED(iq3s, iq3xxs)
GEN_MOE_FFN_ROUTED(iq3s, iq3s)
GEN_MOE_FFN_ROUTED(iq3s, iq4nl)
GEN_MOE_FFN_ROUTED(iq3s, iq4xs)
GEN_MOE_FFN_ROUTED(iq3s, q4k)
GEN_MOE_FFN_ROUTED(iq3s, q6k)
GEN_MOE_FFN_ROUTED(iq1s, iq1s)
GEN_MOE_FFN_ROUTED(iq1s, iq1m)
GEN_MOE_FFN_ROUTED(iq1s, iq2xxs)
GEN_MOE_FFN_ROUTED(iq1s, iq2s)
GEN_MOE_FFN_ROUTED(iq1s, iq3s)
GEN_MOE_FFN_ROUTED(iq1s, iq4xs)
GEN_MOE_FFN_ROUTED(iq1s, q4k)
GEN_MOE_FFN_ROUTED(iq1s, q6k)
GEN_MOE_FFN_ROUTED(iq1m, iq1s)
GEN_MOE_FFN_ROUTED(iq1m, iq1m)
GEN_MOE_FFN_ROUTED(iq1m, iq2xxs)
GEN_MOE_FFN_ROUTED(iq1m, iq2s)
GEN_MOE_FFN_ROUTED(iq1m, iq3s)
GEN_MOE_FFN_ROUTED(iq1m, iq4xs)
GEN_MOE_FFN_ROUTED(iq1m, q4k)
GEN_MOE_FFN_ROUTED(iq1m, q6k)
GEN_MOE_FFN_ROUTED(tq10, tq10)
GEN_MOE_FFN_ROUTED(tq20, tq20)
GEN_MOE_FFN_ROUTED(q20, q20)
GEN_MOE_FFN_ROUTED(mxfp4, mxfp4)
GEN_MOE_FFN_ROUTED(nvfp4, nvfp4)

// Int8-activation dp4a gate+up+activation, device-routed. One wave32 block per nff output row.
#define GEN_MOE_GATE_UP_ROUTED(GU) \
extern "C" __global__ void moe_gate_up_act_i8_routed_##GU( \
    const signed char* __restrict__ qx, const float* __restrict__ xs, \
    const unsigned char* __restrict__ gate_base, const unsigned char* __restrict__ up_base, \
    float* __restrict__ h_out, \
    int ne, int nff, int act_type, int weight_before, const float* __restrict__ dsc_dev, \
    const int* __restrict__ route_ids, const float* __restrict__ route_wts, int slot, \
    long gate_bstride, long up_bstride, int fused, long fused_up_half_boff) { \
    int o = blockIdx.x; int tid = threadIdx.x; \
    if (o >= nff) return; \
    int e = route_ids[slot]; \
    float weight = route_wts[slot]; \
    float wg = weight_before ? weight : 1.0f; \
    float wo = weight_before ? 1.0f : weight; \
    float dsc = dsc_dev ? dsc_dev[e] : 1.0f; \
    const unsigned char* gate_w = gate_base + (long)e * gate_bstride; \
    const unsigned char* up_w = fused ? gate_base + (long)e * gate_bstride + fused_up_half_boff \
                                      : up_base + (long)e * up_bstride; \
    int nb = ne >> 5; \
    float g = i8acc_##GU(qx, xs, gate_w, o, nb, tid); \
    float u = i8acc_##GU(qx, xs, up_w, o, nb, tid); \
    g = wave_sum32(g); u = wave_sum32(u); \
    if (tid == 0) { \
        g *= wg; u *= wg; \
        float a; \
        if (act_type == 0) { a = g / (1.0f + expf(-g)); } \
        else if (act_type == 1) { float x3 = g * g * g; a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3))); } \
        else { a = 1.0f / (1.0f + expf(-g)); } \
        h_out[o] = a * u * wo * dsc; \
    } \
}
GEN_MOE_GATE_UP_ROUTED(q80)
GEN_MOE_GATE_UP_ROUTED(q2k)
GEN_MOE_GATE_UP_ROUTED(q3k)
GEN_MOE_GATE_UP_ROUTED(q4k)
GEN_MOE_GATE_UP_ROUTED(q5k)
GEN_MOE_GATE_UP_ROUTED(q6k)
GEN_MOE_GATE_UP_ROUTED(q40)
GEN_MOE_GATE_UP_ROUTED(q41)
GEN_MOE_GATE_UP_ROUTED(q51)
GEN_MOE_GATE_UP_ROUTED(iq4nl)
GEN_MOE_GATE_UP_ROUTED(iq4xs)
GEN_MOE_GATE_UP_ROUTED(iq2xxs)
GEN_MOE_GATE_UP_ROUTED(iq2xs)
GEN_MOE_GATE_UP_ROUTED(iq2s)
GEN_MOE_GATE_UP_ROUTED(iq3xxs)
GEN_MOE_GATE_UP_ROUTED(iq3s)
GEN_MOE_GATE_UP_ROUTED(iq1s)
GEN_MOE_GATE_UP_ROUTED(iq1m)
GEN_MOE_GATE_UP_ROUTED(tq10)
GEN_MOE_GATE_UP_ROUTED(tq20)
GEN_MOE_GATE_UP_ROUTED(q20)
GEN_MOE_GATE_UP_ROUTED(mxfp4)
GEN_MOE_GATE_UP_ROUTED(nvfp4)

// Int8-activation dp4a down projection, device-routed. One wave32 block per ne output row.
#define GEN_MOE_DOWN_ROUTED(DN) \
extern "C" __global__ void moe_down_i8_routed_##DN( \
    const signed char* __restrict__ hq, const float* __restrict__ hs, \
    const unsigned char* __restrict__ down_base, float* __restrict__ dst, \
    int ne, int nff, const int* __restrict__ route_ids, int slot, long down_bstride) { \
    int d = blockIdx.x; int tid = threadIdx.x; \
    if (d >= ne) return; \
    int e = route_ids[slot]; \
    const unsigned char* down_w = down_base + (long)e * down_bstride; \
    int nb = nff >> 5; \
    float acc = i8acc_##DN(hq, hs, down_w, d, nb, tid); \
    acc = wave_sum32(acc); \
    if (tid == 0) atomicAdd(&dst[d], acc); \
}
GEN_MOE_DOWN_ROUTED(q80)
GEN_MOE_DOWN_ROUTED(q2k)
GEN_MOE_DOWN_ROUTED(q3k)
GEN_MOE_DOWN_ROUTED(q4k)
GEN_MOE_DOWN_ROUTED(q5k)
GEN_MOE_DOWN_ROUTED(q6k)
GEN_MOE_DOWN_ROUTED(q40)
GEN_MOE_DOWN_ROUTED(q41)
GEN_MOE_DOWN_ROUTED(q51)
GEN_MOE_DOWN_ROUTED(iq4nl)
GEN_MOE_DOWN_ROUTED(iq4xs)
GEN_MOE_DOWN_ROUTED(iq2xxs)
GEN_MOE_DOWN_ROUTED(iq2xs)
GEN_MOE_DOWN_ROUTED(iq2s)
GEN_MOE_DOWN_ROUTED(iq3xxs)
GEN_MOE_DOWN_ROUTED(iq3s)
GEN_MOE_DOWN_ROUTED(iq1s)
GEN_MOE_DOWN_ROUTED(iq1m)
GEN_MOE_DOWN_ROUTED(tq10)
GEN_MOE_DOWN_ROUTED(tq20)
GEN_MOE_DOWN_ROUTED(q20)
GEN_MOE_DOWN_ROUTED(mxfp4)
GEN_MOE_DOWN_ROUTED(nvfp4)
"#;

// ── Id-indexed MULTI-SLOT MoE expert GEMV (R8, `*_idm_*`) ────────────────────
//
// Vulkan's `native_gemv_id_multi` family, ported to the shape ROCm's `Op::MoeFfn` actually needs.
// The Slice-38 `*_routed_*` kernels above already resolve the expert bank IN-KERNEL from
// `route_ids[slot]` — ROCm never lacked *id indexing*. What it lacked is the MULTI: the executor
// ran them from a host `for row { for k in 0..n_used { … } }` loop, so a decode step issued
// `1 + 3*rows*n_used` dispatches PER MoE LAYER and, worse, SERIALIZED the selected experts —
// each slot's gate_up → quant_h → down chain shares one `h` scratch, so expert k+1 cannot start
// until expert k's down GEMV retires. At qwen3moe's 48 layers × 8 experts that is ~1150 launches
// per token, each filling a fraction of the device (one wave32 block per output row).
//
// These kernels take the WHOLE `[rows, n_used]` slot grid in ONE dispatch: `blockIdx.y` is the
// flat slot (Vulkan's `slot_global`), split back into `row = slot / n_used` for the activation and
// used directly to index `route_ids` / the per-slot scratch. Identical arithmetic per block to the
// `*_routed_*` twin — same `i8acc_##FMT` decode+dot, same wave32 reduction, same weight fold — so
// the two tiers cannot drift numerically; only the launch geometry changes.
//
// THE EXPERT ADDRESS IS A 64-BIT BYTE OFFSET ON A 64-BIT POINTER, `base + (long)e * bstride`,
// where `bstride` is the host-computed PER-EXPERT BYTE stride (`i64`) — never an element count
// scaled in the kernel. This is the u64/BDA lesson from the Vulkan campaign (its `native_gemv_id`
// STREAMED build had to move to `uint64_t(ids[slot]) * uint64_t(stride)` after an element-space
// u32 multiply wrapped past ~102 Scout-sized slots). HIP pointers are already 64-bit and `long` is
// 64-bit on AMDGCN, so the multiply is 64-bit by construction — but only because `bstride` is
// declared `long`: an `int` stride parameter would make `e * bstride` a 32-bit multiply that wraps
// at 2 GiB of bank, which Scout's 16 × 8192 × 5120 Q4_K down bank (2.7 GiB) reaches. The
// `moe_id_multi_strides_are_64_bit` test pins the host side of that contract.
//
// ACCUMULATION IS NOT ATOMIC. The `*_routed_*` down kernel `atomicAdd`s each slot's contribution
// into `dst` and is deterministic only because the host loop serializes the slots. Running the
// slots concurrently would make the f32 summation ORDER run-to-run nondeterministic — a moving
// golden hash. So `moe_down_i8_idm_*` writes its slot's row into a `[n_slots, ne]` scratch and
// `moe_accum_idm` sums the `n_used` slots of each row IN ASCENDING SLOT ORDER, which is exactly
// the order the serial loop's atomics ran in (starting from a zeroed `dst`) — bit-identical, not
// merely equivalent. Vulkan's small-m arm reaches the same shape (`ybuf` + `moe_accumulate`).
const MOE_ID_MULTI: &str = r#"
// Int8-activation dp4a gate+up+activation over ALL (row, slot) pairs. Grid: (nff, n_slots), one
// wave32 block per (output row, slot). `qx`/`xs` are the int8 quantization of the FULL [rows, ne]
// activation block (one `quant_i8_32` pass for every token); `h_out` is [n_slots, nff].
#define GEN_MOE_GATE_UP_IDM(GU) \
extern "C" __global__ void moe_gate_up_act_i8_idm_##GU( \
    const signed char* __restrict__ qx,       /* int8(x) [rows, ne] */ \
    const float* __restrict__ xs,             /* x scales [rows, ne/32] */ \
    const unsigned char* __restrict__ gate_base, \
    const unsigned char* __restrict__ up_base, \
    float* __restrict__ h_out,                /* [n_slots, nff] */ \
    int ne, int nff, int act_type, int weight_before, const float* __restrict__ dsc_dev, \
    const int* __restrict__ route_ids, const float* __restrict__ route_wts, \
    int n_slots, int n_used, \
    long gate_bstride, long up_bstride, int fused, long fused_up_half_boff) { \
    int o = blockIdx.x; int slot = blockIdx.y; int tid = threadIdx.x; \
    if (o >= nff || slot >= n_slots) return; \
    int row = slot / n_used; \
    int e = route_ids[slot]; \
    float weight = route_wts[slot]; \
    float wg = weight_before ? weight : 1.0f; \
    float wo = weight_before ? 1.0f : weight; \
    float dsc = dsc_dev ? dsc_dev[e] : 1.0f; \
    const unsigned char* gate_w = gate_base + (long)e * gate_bstride; \
    const unsigned char* up_w = fused ? gate_base + (long)e * gate_bstride + fused_up_half_boff \
                                      : up_base + (long)e * up_bstride; \
    int nb = ne >> 5; \
    const signed char* qxr = qx + (long)row * ne; \
    const float* xsr = xs + (long)row * nb; \
    float g = i8acc_##GU(qxr, xsr, gate_w, o, nb, tid); \
    float u = i8acc_##GU(qxr, xsr, up_w, o, nb, tid); \
    g = wave_sum32(g); u = wave_sum32(u); \
    if (tid == 0) { \
        g *= wg; u *= wg; \
        float a; \
        if (act_type == 0) { a = g / (1.0f + expf(-g)); } \
        else if (act_type == 1) { float x3 = g * g * g; a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3))); } \
        else { a = 1.0f / (1.0f + expf(-g)); } \
        h_out[(long)slot * nff + o] = a * u * wo * dsc; \
    } \
}
GEN_MOE_GATE_UP_IDM(q80)
GEN_MOE_GATE_UP_IDM(q2k)
GEN_MOE_GATE_UP_IDM(q3k)
GEN_MOE_GATE_UP_IDM(q4k)
GEN_MOE_GATE_UP_IDM(q5k)
GEN_MOE_GATE_UP_IDM(q6k)
GEN_MOE_GATE_UP_IDM(q40)
GEN_MOE_GATE_UP_IDM(q41)
GEN_MOE_GATE_UP_IDM(q51)
GEN_MOE_GATE_UP_IDM(iq4nl)
GEN_MOE_GATE_UP_IDM(iq4xs)
GEN_MOE_GATE_UP_IDM(iq2xxs)
GEN_MOE_GATE_UP_IDM(iq2xs)
GEN_MOE_GATE_UP_IDM(iq2s)
GEN_MOE_GATE_UP_IDM(iq3xxs)
GEN_MOE_GATE_UP_IDM(iq3s)
GEN_MOE_GATE_UP_IDM(iq1s)
GEN_MOE_GATE_UP_IDM(iq1m)
GEN_MOE_GATE_UP_IDM(tq10)
GEN_MOE_GATE_UP_IDM(tq20)
GEN_MOE_GATE_UP_IDM(q20)
GEN_MOE_GATE_UP_IDM(mxfp4)
GEN_MOE_GATE_UP_IDM(nvfp4)

// P7f: Q4_K idm gate/up with CN=2 column tiling — each 32-thread wave processes TWO output
// columns instead of one, halving the wave count (3.1M → 1.57M for nff=768, n_slots=4096).
// The activation row is shared between both columns; only the weight reads double per block.
// Same arithmetic as moe_gate_up_act_i8_idm_q4k — bit-identical to the CN=1 kernel because
// each (column, slot) pair is computed exactly once by exactly one wave.
extern "C" __global__ void moe_gate_up_act_i8_idm_q4k_cn2(
    const signed char* __restrict__ qx, const float* __restrict__ xs,
    const unsigned char* __restrict__ gate_base,
    const unsigned char* __restrict__ up_base,
    float* __restrict__ h_out,
    int ne, int nff, int act_type, int weight_before, const float* __restrict__ dsc_dev,
    const int* __restrict__ route_ids, const float* __restrict__ route_wts,
    int n_slots, int n_used,
    long gate_bstride, long up_bstride, int fused, long fused_up_half_boff
) {
    int o_pair = blockIdx.x; int slot = blockIdx.y; int tid = threadIdx.x;
    int o0 = o_pair * 2;
    if (o0 >= nff || slot >= n_slots) return;
    int o1 = o0 + 1;
    bool col1_live = (o1 < nff);
    int row = slot / n_used;
    int e = route_ids[slot];
    float weight = route_wts[slot];
    float wg = weight_before ? weight : 1.0f;
    float wo = weight_before ? 1.0f : weight;
    float dsc = dsc_dev ? dsc_dev[e] : 1.0f;
    const unsigned char* gate_w = gate_base + (long)e * gate_bstride;
    const unsigned char* up_w = fused ? gate_base + (long)e * gate_bstride + fused_up_half_boff
                                      : up_base + (long)e * up_bstride;
    int nb = ne >> 5;
    int spr = nb >> 3;
    const signed char* qxr = qx + (long)row * ne;
    const float* xsr = xs + (long)row * nb;

    // CN=2: two independent dot products per weight (gate, up) → 4 accumulators.
    float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        int s = blk & 7;
        unsigned int sh = (unsigned int)(s & 1) * 4u;
        // Activation — shared across both columns.
        const int4* xq = (const int4*)(qxr + blk * 32);
        int4 xlo = xq[0], xhi = xq[1];
        int xv[8] = { xlo.x, xlo.y, xlo.z, xlo.w, xhi.x, xhi.y, xhi.z, xhi.w };
        float sx = xsr[blk];

        // ── Column 0: gate ──
        {
            long super0 = (long)o0 * spr + (blk >> 3);
            const uint4* b0 = (const uint4*)(gate_w + super0 * 144);
            uint4 hdr0 = b0[0];
            float d0 = f16q_lo(hdr0.x), dmin0 = f16q_hi(hdr0.x);
            int sc0, mm0; k4q(hdr0, s, &sc0, &mm0);
            const uint4* qq0 = b0 + 1 + (s >> 1) * 2;
            uint4 wlo0 = qq0[0], whi0 = qq0[1];
            unsigned int wv0[8] = { wlo0.x,wlo0.y,wlo0.z,wlo0.w, whi0.x,whi0.y,whi0.z,whi0.w };
            int idot0 = 0, isum0 = 0;
            for (int k = 0; k < 8; k++) {
                idot0 = idot4(xv[k], (int)((wv0[k] >> sh) & 0x0F0F0F0Fu), idot0);
                isum0 = idot4(xv[k], 0x01010101, isum0);
            }
            g0 += (d0 * (float)sc0) * sx * (float)idot0 + (dmin0 * (float)(-mm0)) * sx * (float)isum0;
        }
        // ── Column 1: gate ──
        if (col1_live) {
            long super1 = (long)o1 * spr + (blk >> 3);
            const uint4* b1 = (const uint4*)(gate_w + super1 * 144);
            uint4 hdr1 = b1[0];
            float d1 = f16q_lo(hdr1.x), dmin1 = f16q_hi(hdr1.x);
            int sc1, mm1; k4q(hdr1, s, &sc1, &mm1);
            const uint4* qq1 = b1 + 1 + (s >> 1) * 2;
            uint4 wlo1 = qq1[0], whi1 = qq1[1];
            unsigned int wv1[8] = { wlo1.x,wlo1.y,wlo1.z,wlo1.w, whi1.x,whi1.y,whi1.z,whi1.w };
            int idot1 = 0, isum1 = 0;
            for (int k = 0; k < 8; k++) {
                idot1 = idot4(xv[k], (int)((wv1[k] >> sh) & 0x0F0F0F0Fu), idot1);
                isum1 = idot4(xv[k], 0x01010101, isum1);
            }
            g1 += (d1 * (float)sc1) * sx * (float)idot1 + (dmin1 * (float)(-mm1)) * sx * (float)isum1;
        }
        // ── Column 0: up (same activation, different weight bank) ──
        {
            long super0 = (long)o0 * spr + (blk >> 3);
            const uint4* b0 = (const uint4*)(up_w + super0 * 144);
            uint4 hdr0 = b0[0];
            float d0 = f16q_lo(hdr0.x), dmin0 = f16q_hi(hdr0.x);
            int sc0, mm0; k4q(hdr0, s, &sc0, &mm0);
            const uint4* qq0 = b0 + 1 + (s >> 1) * 2;
            uint4 wlo0 = qq0[0], whi0 = qq0[1];
            unsigned int wv0[8] = { wlo0.x,wlo0.y,wlo0.z,wlo0.w, whi0.x,whi0.y,whi0.z,whi0.w };
            int idot0 = 0, isum0 = 0;
            for (int k = 0; k < 8; k++) {
                idot0 = idot4(xv[k], (int)((wv0[k] >> sh) & 0x0F0F0F0Fu), idot0);
                isum0 = idot4(xv[k], 0x01010101, isum0);
            }
            u0 += (d0 * (float)sc0) * sx * (float)idot0 + (dmin0 * (float)(-mm0)) * sx * (float)isum0;
        }
        // ── Column 1: up ──
        if (col1_live) {
            long super1 = (long)o1 * spr + (blk >> 3);
            const uint4* b1 = (const uint4*)(up_w + super1 * 144);
            uint4 hdr1 = b1[0];
            float d1 = f16q_lo(hdr1.x), dmin1 = f16q_hi(hdr1.x);
            int sc1, mm1; k4q(hdr1, s, &sc1, &mm1);
            const uint4* qq1 = b1 + 1 + (s >> 1) * 2;
            uint4 wlo1 = qq1[0], whi1 = qq1[1];
            unsigned int wv1[8] = { wlo1.x,wlo1.y,wlo1.z,wlo1.w, whi1.x,whi1.y,whi1.z,whi1.w };
            int idot1 = 0, isum1 = 0;
            for (int k = 0; k < 8; k++) {
                idot1 = idot4(xv[k], (int)((wv1[k] >> sh) & 0x0F0F0F0Fu), idot1);
                isum1 = idot4(xv[k], 0x01010101, isum1);
            }
            u1 += (d1 * (float)sc1) * sx * (float)idot1 + (dmin1 * (float)(-mm1)) * sx * (float)isum1;
        }
    }
    g0 = wave_sum32(g0); u0 = wave_sum32(u0);
    if (col1_live) { g1 = wave_sum32(g1); u1 = wave_sum32(u1); }
    if (tid == 0) {
        // Column 0
        float g = g0 * wg, up = u0 * wg;
        float a;
        if (act_type == 0) a = g / (1.0f + expf(-g));
        else if (act_type == 1) { float x3=g*g*g; a=0.5f*g*(1.0f+tanhf(0.7978845608f*(g+0.044715f*x3))); }
        else a = 1.0f / (1.0f + expf(-g));
        h_out[(long)slot * nff + o0] = a * up * wo * dsc;
        // Column 1
        if (col1_live) {
            g = g1 * wg; up = u1 * wg;
            if (act_type == 0) a = g / (1.0f + expf(-g));
            else if (act_type == 1) { float x3=g*g*g; a=0.5f*g*(1.0f+tanhf(0.7978845608f*(g+0.044715f*x3))); }
            else a = 1.0f / (1.0f + expf(-g));
            h_out[(long)slot * nff + o1] = a * up * wo * dsc;
        }
    }
}

// Int8-activation dp4a down projection over ALL (row, slot) pairs. Grid: (ne, n_slots). Writes
// y[n_slots, ne] — NOT an atomicAdd into dst; `moe_accum_idm` does the (ordered) reduction.
#define GEN_MOE_DOWN_IDM(DN) \
extern "C" __global__ void moe_down_i8_idm_##DN( \
    const signed char* __restrict__ hq,       /* int8(h) [n_slots, nff] */ \
    const float* __restrict__ hs,             /* h scales [n_slots, nff/32] */ \
    const unsigned char* __restrict__ down_base, float* __restrict__ y, /* [n_slots, ne] */ \
    int ne, int nff, const int* __restrict__ route_ids, int n_slots, long down_bstride) { \
    int d = blockIdx.x; int slot = blockIdx.y; int tid = threadIdx.x; \
    if (d >= ne || slot >= n_slots) return; \
    int e = route_ids[slot]; \
    const unsigned char* down_w = down_base + (long)e * down_bstride; \
    int nb = nff >> 5; \
    const signed char* hqr = hq + (long)slot * nff; \
    const float* hsr = hs + (long)slot * nb; \
    float acc = i8acc_##DN(hqr, hsr, down_w, d, nb, tid); \
    acc = wave_sum32(acc); \
    if (tid == 0) y[(long)slot * ne + d] = acc; \
}
GEN_MOE_DOWN_IDM(q80)
GEN_MOE_DOWN_IDM(q2k)
GEN_MOE_DOWN_IDM(q3k)
GEN_MOE_DOWN_IDM(q4k)
GEN_MOE_DOWN_IDM(q5k)
GEN_MOE_DOWN_IDM(q6k)
GEN_MOE_DOWN_IDM(q40)
GEN_MOE_DOWN_IDM(q41)
GEN_MOE_DOWN_IDM(q51)
GEN_MOE_DOWN_IDM(iq4nl)
GEN_MOE_DOWN_IDM(iq4xs)
GEN_MOE_DOWN_IDM(iq2xxs)
GEN_MOE_DOWN_IDM(iq2xs)
GEN_MOE_DOWN_IDM(iq2s)
GEN_MOE_DOWN_IDM(iq3xxs)
GEN_MOE_DOWN_IDM(iq3s)
GEN_MOE_DOWN_IDM(iq1s)
GEN_MOE_DOWN_IDM(iq1m)
GEN_MOE_DOWN_IDM(tq10)
GEN_MOE_DOWN_IDM(tq20)
GEN_MOE_DOWN_IDM(q20)
GEN_MOE_DOWN_IDM(mxfp4)
GEN_MOE_DOWN_IDM(nvfp4)

// Ordered reduction of the id-GEMV's per-slot outputs into the token rows. One thread per
// (row, channel); sums slots 0..n_used-1 IN ORDER onto a 0.0f seed, reproducing the serial loop's
// `atomicAdd` sequence into a zeroed dst exactly (f32 addition is not associative — see the header).
// `dst` is ADDED to, not overwritten: the executor zeroes it, and a row whose slots were all
// dispatched here still reads its own accumulation only.
//
// F1c FUSED RESIDUAL. `res` non-null folds the MoE sublayer's following `Op::Add` into this
// epilogue: `dst[i] = res[i] + acc` instead of a standalone `add` over a zeroed `dst`. The
// SUMMATION ORDER IS UNTOUCHED — `acc` is still the ascending-slot reduction onto a 0.0f seed, and
// the residual joins it exactly where the elided `add` kernel joined it (`dst = a + b` with
// `a` = residual), so this is bit-identical, not merely equivalent:
//   split : add(res, 0.0f + acc)  fused : res + acc
// and `0.0f + acc == acc` for every value `acc` can take, because `acc` is seeded at +0.0 and
// round-to-nearest never produces -0.0 from `+0.0 + v` — the one case where adding zero is not the
// identity. `res == dst` (the in-place `hidden += moe` the seam emits) is safe: one thread owns
// element `i`, so its read-then-write has no other writer.
extern "C" __global__ void moe_accum_idm(
    const float* __restrict__ y,   /* [rows, n_used, ne] */
    float* __restrict__ dst,       /* [rows, ne] */
    const float* __restrict__ res, /* [rows, ne] fused residual (null = none) */
    int ne, int rows, int n_used
) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * ne;
    if (i >= total) return;
    int row = (int)(i / ne);
    int d = (int)(i - (long)row * ne);
    float acc = 0.0f;
    for (int k = 0; k < n_used; k++) {
        acc += y[((long)row * n_used + k) * ne + d];
    }
    if (res) dst[i] = res[i] + acc;
    else dst[i] += acc;
}
"#;

// ── Bucket-sorted BATCHED per-expert MoE GEMV (P2, `*_idb_*`) ────────────────
//
// The R8 `*_idm_*` tier above fixed the LAUNCH count; it did not fix the WEIGHT TRAFFIC. Its grid
// is `(out_row, slot)`, so every one of the `rows * n_used` slots re-reads its expert's whole bank
// independently. Measured on Qwen3-30B-A3B Q4_K_M `pp512` (`INFR_PROF_OPS=1`): 4096 slots over
// 128 experts is a **32× re-read** — 12.5 GB of weight fetch per layer against 391 MB of distinct
// bytes, at 59.7 ms a layer (97.1% of the whole forward, 2.86 s of a 2.95 s pp512).
//
// This tier BUCKETS THE SLOTS BY EXPERT and gives each expert ONE block per output row, which
// loops over its whole bucket. The grid is `(out_row, expert)` instead of `(out_row, slot)`, so an
// expert's bank row is fetched from memory ONCE per row-chunk and reused across every token routed
// to it — the re-read collapses from `rows * n_used / n_expert` to 1 within a chunk (Vulkan's
// `moe_small_m` batched arm, which is what its crossover picks above m=8).
//
// THE ARITHMETIC IS THE `*_idm_*` ARITHMETIC, UNCHANGED. Same `i8acc_##FMT` decode+dot, same
// `wave_sum32`, same epilogue, same per-slot destination (`h_out[slot]` / `y[slot]`) — only WHICH
// block computes a given `(out_row, slot)` pair moves, and each pair is still computed exactly
// once by exactly one wave. Nothing is summed across slots here (`moe_accum_idm` still owns the
// ordered reduction), so the bucket ORDER is invisible to the result: it may come out of an atomic
// scatter in any order and the output is still bit-identical to both the id tier and the pre-R8
// serial tier. `moe_ffn_bucket_tier_matches_the_id_tier_bitwise` pins that.
//
// The sort is ONE workgroup, `moe_bucket_sort` — an LDS histogram over `route_ids`, a serial
// exclusive scan by lane 0, then an LDS-atomic scatter. `n_expert` is bounded by
// `MOE_BUCKET_MAX_EXPERT` because the histogram is STATIC LDS (a dynamic-shared launch would need
// its own dispatch helper for one tiny kernel); the executor keeps the id tier for anything wider.
const MOE_ID_BUCKET: &str = r#"
#define MOE_BUCKET_MAX_EXPERT 1024
#define MOE_IDB_WAVES 4

// Counting sort of the `[n_slots]` expert ids into per-expert buckets. One workgroup.
//   ecnt[e]  = number of slots routed to expert e
//   eoff[e]  = exclusive prefix sum of ecnt (the bucket's start in `bslot`)
//   bslot[p] = the slot index at sorted position p
// Order WITHIN a bucket is unspecified (LDS-atomic scatter) and deliberately so — every consumer
// writes its result at the slot's own address, so no reduction order depends on it.
extern "C" __global__ void moe_bucket_sort(
    const int* __restrict__ route_ids, /* [n_slots] */
    int* __restrict__ ecnt,            /* [n_expert] */
    int* __restrict__ eoff,            /* [n_expert] */
    int* __restrict__ bslot,           /* [n_slots] */
    int n_slots, int n_expert
) {
    __shared__ int cnt[MOE_BUCKET_MAX_EXPERT];
    __shared__ int cur[MOE_BUCKET_MAX_EXPERT];
    int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int s = tid; s < n_slots; s += blockDim.x) atomicAdd(&cnt[route_ids[s]], 1);
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int e = 0; e < n_expert; e++) { cur[e] = run; run += cnt[e]; }
    }
    __syncthreads();
    for (int e = tid; e < n_expert; e += blockDim.x) { ecnt[e] = cnt[e]; eoff[e] = cur[e]; }
    __syncthreads();
    for (int s = tid; s < n_slots; s += blockDim.x) {
        int e = route_ids[s];
        bslot[atomicAdd(&cur[e], 1)] = s;
    }
}

// Gate+up+activation, batched over one expert's bucket. Grid: (nff, n_expert); one wave32 block
// per (output row, expert), looping over the bucket. The expert's gate/up row `o` is loaded once
// for the whole bucket instead of once per slot.
#define GEN_MOE_GATE_UP_IDB(GU) \
extern "C" __global__ void moe_gate_up_act_i8_idb_##GU( \
    const signed char* __restrict__ qx,       /* int8(x) [rows, ne] */ \
    const float* __restrict__ xs,             /* x scales [rows, ne/32] */ \
    const unsigned char* __restrict__ gate_base, \
    const unsigned char* __restrict__ up_base, \
    float* __restrict__ h_out,                /* [n_slots, nff] */ \
    int ne, int nff, int act_type, int weight_before, const float* __restrict__ dsc_dev, \
    const float* __restrict__ route_wts, int n_used, \
    long gate_bstride, long up_bstride, int fused, long fused_up_half_boff, \
    const int* __restrict__ bslot, const int* __restrict__ eoff, const int* __restrict__ ecnt) { \
    int tid = threadIdx.x & 31; \
    int o = blockIdx.x * MOE_IDB_WAVES + (int)(threadIdx.x >> 5); int e = blockIdx.y; \
    if (o >= nff) return; \
    int cnt = ecnt[e]; \
    if (cnt <= 0) return; \
    int off = eoff[e]; \
    float dsc = dsc_dev ? dsc_dev[e] : 1.0f; \
    const unsigned char* gate_w = gate_base + (long)e * gate_bstride; \
    const unsigned char* up_w = fused ? gate_base + (long)e * gate_bstride + fused_up_half_boff \
                                      : up_base + (long)e * up_bstride; \
    int nb = ne >> 5; \
    for (int i = 0; i < cnt; i++) { \
        int slot = bslot[off + i]; \
        int row = slot / n_used; \
        float weight = route_wts[slot]; \
        float wg = weight_before ? weight : 1.0f; \
        float wo = weight_before ? 1.0f : weight; \
        const signed char* qxr = qx + (long)row * ne; \
        const float* xsr = xs + (long)row * nb; \
        float g = i8acc_##GU(qxr, xsr, gate_w, o, nb, tid); \
        float u = i8acc_##GU(qxr, xsr, up_w, o, nb, tid); \
        g = wave_sum32(g); u = wave_sum32(u); \
        if (tid == 0) { \
            g *= wg; u *= wg; \
            float a; \
            if (act_type == 0) { a = g / (1.0f + expf(-g)); } \
            else if (act_type == 1) { float x3 = g * g * g; a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * x3))); } \
            else { a = 1.0f / (1.0f + expf(-g)); } \
            h_out[(long)slot * nff + o] = a * u * wo * dsc; \
        } \
    } \
}
GEN_MOE_GATE_UP_IDB(q80)
GEN_MOE_GATE_UP_IDB(q2k)
GEN_MOE_GATE_UP_IDB(q3k)
GEN_MOE_GATE_UP_IDB(q4k)
GEN_MOE_GATE_UP_IDB(q5k)
GEN_MOE_GATE_UP_IDB(q6k)
GEN_MOE_GATE_UP_IDB(q40)
GEN_MOE_GATE_UP_IDB(q41)
GEN_MOE_GATE_UP_IDB(q51)
GEN_MOE_GATE_UP_IDB(iq4nl)
GEN_MOE_GATE_UP_IDB(iq4xs)
GEN_MOE_GATE_UP_IDB(iq2xxs)
GEN_MOE_GATE_UP_IDB(iq2xs)
GEN_MOE_GATE_UP_IDB(iq2s)
GEN_MOE_GATE_UP_IDB(iq3xxs)
GEN_MOE_GATE_UP_IDB(iq3s)
GEN_MOE_GATE_UP_IDB(iq1s)
GEN_MOE_GATE_UP_IDB(iq1m)
GEN_MOE_GATE_UP_IDB(tq10)
GEN_MOE_GATE_UP_IDB(tq20)
GEN_MOE_GATE_UP_IDB(q20)
GEN_MOE_GATE_UP_IDB(mxfp4)
GEN_MOE_GATE_UP_IDB(nvfp4)

// Down projection, batched over one expert's bucket. Grid: (ne, n_expert). Writes y[slot] exactly
// as `moe_down_i8_idm_*` does — `moe_accum_idm` still owns the ordered reduction.
#define GEN_MOE_DOWN_IDB(DN) \
extern "C" __global__ void moe_down_i8_idb_##DN( \
    const signed char* __restrict__ hq,       /* int8(h) [n_slots, nff] */ \
    const float* __restrict__ hs,             /* h scales [n_slots, nff/32] */ \
    const unsigned char* __restrict__ down_base, float* __restrict__ y, /* [n_slots, ne] */ \
    int ne, int nff, long down_bstride, \
    const int* __restrict__ bslot, const int* __restrict__ eoff, const int* __restrict__ ecnt) { \
    int tid = threadIdx.x & 31; \
    int d = blockIdx.x * MOE_IDB_WAVES + (int)(threadIdx.x >> 5); int e = blockIdx.y; \
    if (d >= ne) return; \
    int cnt = ecnt[e]; \
    if (cnt <= 0) return; \
    int off = eoff[e]; \
    const unsigned char* down_w = down_base + (long)e * down_bstride; \
    int nb = nff >> 5; \
    for (int i = 0; i < cnt; i++) { \
        int slot = bslot[off + i]; \
        const signed char* hqr = hq + (long)slot * nff; \
        const float* hsr = hs + (long)slot * nb; \
        float acc = i8acc_##DN(hqr, hsr, down_w, d, nb, tid); \
        acc = wave_sum32(acc); \
        if (tid == 0) y[(long)slot * ne + d] = acc; \
    } \
}
GEN_MOE_DOWN_IDB(q80)
GEN_MOE_DOWN_IDB(q2k)
GEN_MOE_DOWN_IDB(q3k)
GEN_MOE_DOWN_IDB(q4k)
GEN_MOE_DOWN_IDB(q5k)
GEN_MOE_DOWN_IDB(q6k)
GEN_MOE_DOWN_IDB(q40)
GEN_MOE_DOWN_IDB(q41)
GEN_MOE_DOWN_IDB(q51)
GEN_MOE_DOWN_IDB(iq4nl)
GEN_MOE_DOWN_IDB(iq4xs)
GEN_MOE_DOWN_IDB(iq2xxs)
GEN_MOE_DOWN_IDB(iq2xs)
GEN_MOE_DOWN_IDB(iq2s)
GEN_MOE_DOWN_IDB(iq3xxs)
GEN_MOE_DOWN_IDB(iq3s)
GEN_MOE_DOWN_IDB(iq1s)
GEN_MOE_DOWN_IDB(iq1m)
GEN_MOE_DOWN_IDB(tq10)
GEN_MOE_DOWN_IDB(tq20)
GEN_MOE_DOWN_IDB(q20)
GEN_MOE_DOWN_IDB(mxfp4)
GEN_MOE_DOWN_IDB(nvfp4)
"#;

// ── Module cache ─────────────────────────────────────────────────────────────

/// hiprtc options, ONE list feeding both the compile call and the disk cache's key — a flag that
/// changed the generated code but not the key would let a stale code object be reloaded.
const COMPILE_OPTS: [&str; 1] = ["-std=c++17"];

/// Producer magic + on-disk layout version for the persisted HIP code object. Bump on any change
/// to what the payload MEANS; the envelope then rejects every old file instead of misreading it.
const MODULE_CACHE_MAGIC: [u8; 8] = *b"INFRRMC1";

/// The gfx arch this device reports (`gfx1100`, `gfx90a:sramecc+:xnack-`, …) as a filename-safe
/// token. Empty when the arch is unknown — the caller then refuses to cache rather than guessing,
/// since two different archs must never share a blob.
fn gfx_arch(device: c_int) -> String {
    sanitize_arch(&crate::backend::device_arch_name(device))
}

/// `gcnArchName` → a filename-safe token. Split out from [`gfx_arch`] so it is testable without a
/// device.
///
/// The escape is INJECTIVE, not merely "safe": a real arch carries feature flags
/// (`gfx90a:sramecc+:xnack-`) that CHANGE the generated code, so `xnack+` and `xnack-` must not
/// land on one file name — folding both to `_` did exactly that. Alphanumerics pass through
/// lowercased; everything else (including `_` itself, so the escape cannot be forged) becomes
/// `_<hex>`.
fn sanitize_arch(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for b in raw.bytes() {
        if b.is_ascii_alphanumeric() {
            out.push(b.to_ascii_lowercase() as char);
        } else {
            out.push_str(&format!("_{b:02x}"));
        }
    }
    out
}

/// The on-disk cache for this device's compiled HIP module (see [`infr_core::kernel_cache`] for the
/// envelope/durability/tripwire contract this rides on).
///
/// The KEY is what makes reloading safe, and it is composed so that NOTHING that changes the
/// generated code can leave it unmoved:
///
/// * `FNV-1a(hip_source())` **and the source length** — `hip_source()` is assembled at RUN time
///   (it embeds the IQ4 codebook and the IQ2/IQ3 grids emitted from the host tables), so hashing
///   the actual string covers every kernel edit by construction; no build.rs fingerprint is needed
///   and none can drift from it. The length is one extra cheap field against an FNV collision.
/// * the **gfx arch** — also in the FILE NAME, so two GPUs of different archs in one box never
///   share a blob; in the key too because the name is a convention and the key is a check.
/// * the **HIP runtime, HIP driver and hiprtc versions** — a code object is only loadable by the
///   stack that produced it, and any of the three can move independently on an upgrade.
/// * the **compile options** ([`COMPILE_OPTS`]).
///
/// Disabled (`kernels.rocm.module_cache = false`, or an unknown arch) ⇒ a total no-op.
fn module_cache(device: c_int, src: &str, cfg: &Config) -> KernelCache {
    let arch = gfx_arch(device);
    let enabled = cfg.kernels.rocm.module_cache && !arch.is_empty();
    let mut key = Vec::with_capacity(64 + arch.len());
    key.extend_from_slice(&infr_core::kernel_cache::fnv1a(src.as_bytes()).to_le_bytes());
    key.extend_from_slice(&(src.len() as u64).to_le_bytes());
    key.extend_from_slice(arch.as_bytes());
    key.push(0);
    for o in COMPILE_OPTS {
        key.extend_from_slice(o.as_bytes());
        key.push(0);
    }
    let (mut rt, mut drv, mut rtc_major, mut rtc_minor) = (0, 0, 0, 0);
    unsafe {
        ffi::hipRuntimeGetVersion(&mut rt);
        ffi::hipDriverGetVersion(&mut drv);
        ffi::hiprtcVersion(&mut rtc_major, &mut rtc_minor);
    }
    for v in [rt, drv, rtc_major, rtc_minor] {
        key.extend_from_slice(&v.to_le_bytes());
    }
    KernelCache::open(
        &format!("rocm-module-{arch}.bin"),
        MODULE_CACHE_MAGIC,
        key,
        enabled,
    )
}

/// Compiled HIP module + kernel-function cache.
pub struct Pipelines {
    module: ffi::hipModule_t,
    /// Kernel name → function handle (lazily fetched).
    cache: Mutex<HashMap<&'static str, ffi::hipFunction_t>>,
    /// The on-disk code-object cache this module came from (or a disabled no-op). Held for its
    /// tripwire: a clean drop disarms the marker that `load` armed.
    disk: KernelCache,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

impl Pipelines {
    /// Load this device's HIP module: from the on-disk code-object cache when it is valid,
    /// otherwise by compiling `hip_source()` with hiprtc and storing the result.
    ///
    /// `hiprtcCompileProgram` is ~9.2 s on a cold comgr cache and still ~0.25 s of a 0.48 s launch
    /// when comgr's own lower-level cache is hot — one-time work that re-ran on EVERY process
    /// launch before this. The cache is self-invalidating on any source / arch / runtime change
    /// (see [`module_cache`]).
    ///
    /// A cached blob that `hipModuleLoadData` REJECTS is not an error: the file is invalidated and
    /// this falls through to a normal compile.
    // The active device is already selected via `hipSetDevice` before `build`, and hiprtc targets
    // that arch via the auto-detect below; `device` is used to KEY the cache.
    pub fn build(device: c_int, cfg: &Config) -> Result<Self> {
        let src = hip_source();
        let disk = module_cache(device, &src, cfg);

        if let Some(code) = disk.load() {
            match Self::load_module(&code) {
                Ok(module) => return Ok(Self::with_module(module, disk)),
                Err(e) => {
                    // The stack moved under a key that did not see it (or the file is subtly
                    // wrong). Recompiling is the correct answer, not a failure — but say so, since
                    // a cache that silently misses every launch is a perf bug nobody would notice.
                    eprintln!(
                        "[infr] the cached ROCm module was rejected by the HIP runtime ({e}) — \
                         discarding it and recompiling."
                    );
                    disk.invalidate();
                }
            }
        }

        let code = Self::compile(&src)?;
        // Best-effort: a full disk or a read-only cache dir must not fail a backend that has a
        // perfectly good freshly-compiled module in hand.
        let _ = disk.store(&code);
        let module = Self::load_module(&code)?;
        Ok(Self::with_module(module, disk))
    }

    fn with_module(module: ffi::hipModule_t, disk: KernelCache) -> Self {
        Self {
            module,
            cache: Mutex::new(HashMap::new()),
            disk,
        }
    }

    /// hiprtc: assembled HIP source → a device code object.
    fn compile(src: &str) -> Result<Vec<u8>> {
        let csrc = CString::new(src).map_err(|e| be(format!("kernel source NUL-byte: {e}")))?;
        let mut prog: ffi::hiprtcProgram = std::ptr::null_mut();
        let name_cstr = CString::new("infr_kernels").unwrap();
        let rc = unsafe {
            ffi::hiprtcCreateProgram(
                &mut prog,
                csrc.as_ptr(),
                name_cstr.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if rc != ffi::HIPRTC_SUCCESS {
            return Err(be(format!("hiprtcCreateProgram: rc={rc}")));
        }

        // Compile without --gpu-architecture: hiprtc auto-detects the device from the active
        // hipSetDevice context. The int8 dp4a dot is written as a portable scalar idiom (not the
        // `sdot4` builtin), so no optional target feature (`dot1-insts`) needs to be pinned — the
        // plain auto-detect target compiles it in every launch context. (The auto-detected arch is
        // therefore NOT visible in the options, which is why the module cache keys the arch
        // explicitly — see [`module_cache`].)
        let opt_cstrs: Vec<CString> = COMPILE_OPTS
            .iter()
            .map(|o| CString::new(*o).unwrap())
            .collect();
        let opts: Vec<*const c_char> = opt_cstrs.iter().map(|o| o.as_ptr()).collect();
        let rc = unsafe { ffi::hiprtcCompileProgram(prog, opts.len() as i32, opts.as_ptr()) };
        if rc != ffi::HIPRTC_SUCCESS {
            // Fetch the compile log for diagnostics
            let mut log_size: usize = 0;
            unsafe { ffi::hiprtcGetProgramLogSize(prog, &mut log_size) };
            let mut log_buf: Vec<u8> = vec![0u8; log_size];
            unsafe { ffi::hiprtcGetProgramLog(prog, log_buf.as_mut_ptr() as *mut c_char) };
            let log = String::from_utf8_lossy(&log_buf);
            unsafe { ffi::hiprtcDestroyProgram(&mut prog) };
            return Err(be(format!("hiprtcCompileProgram failed (rc={rc}):\n{log}")));
        }

        // Get compiled code
        let mut code_size: usize = 0;
        let rc = unsafe { ffi::hiprtcGetCodeSize(prog, &mut code_size) };
        if rc != ffi::HIPRTC_SUCCESS {
            unsafe { ffi::hiprtcDestroyProgram(&mut prog) };
            return Err(be(format!("hiprtcGetCodeSize: rc={rc}")));
        }
        let mut code: Vec<u8> = vec![0u8; code_size];
        let rc = unsafe { ffi::hiprtcGetCode(prog, code.as_mut_ptr() as *mut c_char) };
        if rc != ffi::HIPRTC_SUCCESS {
            unsafe { ffi::hiprtcDestroyProgram(&mut prog) };
            return Err(be(format!("hiprtcGetCode: rc={rc}")));
        }
        unsafe { ffi::hiprtcDestroyProgram(&mut prog) };
        Ok(code)
    }

    /// Load a code object (freshly compiled or off disk) into a HIP module. An `Err` from a CACHED
    /// blob is recoverable — see [`build`](Self::build).
    fn load_module(code: &[u8]) -> Result<ffi::hipModule_t> {
        if code.is_empty() {
            return Err(be("hipModuleLoadData: empty code object"));
        }
        let mut module: ffi::hipModule_t = std::ptr::null_mut();
        let rc = unsafe {
            ffi::hipModuleLoadData(&mut module, code.as_ptr() as *const std::ffi::c_void)
        };
        if rc != ffi::HIP_SUCCESS {
            return Err(be(format!("hipModuleLoadData: rc={rc}")));
        }
        Ok(module)
    }

    /// Get (creating + caching on first use) the kernel function for a given name.
    pub fn get(&self, name: &'static str) -> Result<ffi::hipFunction_t> {
        if let Some(f) = self.cache.lock().unwrap().get(name) {
            return Ok(*f);
        }
        let cname = CString::new(name).map_err(|e| be(format!("kernel name NUL-byte: {e}")))?;
        let mut func: ffi::hipFunction_t = std::ptr::null_mut();
        let rc = unsafe { ffi::hipModuleGetFunction(&mut func, self.module, cname.as_ptr()) };
        if rc != ffi::HIP_SUCCESS {
            return Err(be(format!("hipModuleGetFunction({name}): rc={rc}")));
        }
        self.cache.lock().unwrap().insert(name, func);
        Ok(func)
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        // hipModuleDestroy doesn't exist in public API; the module leaks on drop.
        // This is fine for a single-backend-instance lifetime.
        //
        // TRIPWIRE step 2 (see `infr_core::kernel_cache`): we got here, so this run did NOT hang
        // the GPU on whatever it seeded from disk. Clear THIS instance's marker; a sibling
        // backend's stays armed. A run that dies without reaching here leaves its marker behind,
        // and the next launch discards the blob it accuses.
        self.disk.disarm();
    }
}

#[cfg(test)]
mod module_cache_tests {
    use super::*;

    /// The blob file name is what keeps two archs in one box off each other's code objects, so the
    /// arch token must be INJECTIVE — including over the feature suffixes (`:sramecc+:xnack-`),
    /// which change the generated code. Folding every non-alphanumeric to `_` failed exactly that:
    /// `xnack+` and `xnack-` became one name.
    #[test]
    fn the_arch_token_is_filename_safe_and_injective() {
        assert_eq!(sanitize_arch("gfx1100"), "gfx1100");
        assert_eq!(sanitize_arch("GFX1100"), "gfx1100", "case-folded");
        assert_eq!(
            sanitize_arch("gfx90a:sramecc+:xnack-"),
            "gfx90a_3asramecc_2b_3axnack_2d"
        );
        // Distinct arch strings ⇒ distinct file names. `_` is itself escaped, so no input can
        // forge another input's escape.
        let names: Vec<String> = [
            "gfx90a",
            "gfx90a:xnack-",
            "gfx90a:xnack+",
            "gfx90a:sramecc-:xnack+",
            "gfx90a:sramecc+:xnack-",
            "gfx90a_3axnack_2d",
            "gfx1100",
            "gfx1101",
        ]
        .iter()
        .map(|a| sanitize_arch(a))
        .collect();
        let mut uniq = names.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), names.len(), "arch tokens collided: {names:?}");
        for n in &names {
            assert!(
                n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "the token names a file: {n}"
            );
        }
        // An unknown arch (`device_arch_name` rejected what it read) is an EMPTY token, which
        // `module_cache` reads as "do not cache" rather than sharing a `rocm-module-.bin`.
        assert_eq!(sanitize_arch(""), "");
    }
}

#[cfg(test)]
mod codebook_tests {
    /// The IQ4 codebook is packed into four `u32`s before it is emitted, and NOTHING on a CPU-only
    /// box would catch a byte-order slip in that packing — the GPU parity tests need the device.
    /// So unpack the emitted words the way `kv_iq4nl` does (word `i>>2`, byte `i&3`, sign-extended)
    /// and require the result to BE `KVALUES_IQ4NL`, and require the emitted text to name the const
    /// it came from. A hardcoded second copy of the table would fail the first check the moment the
    /// host const changed; a wrong shift would fail it immediately.
    /// R7 extends it to the second table, [`infr_gguf::dequant::KVALUES_MXFP4`] (the E2M1 codebook
    /// MXFP4 and NVFP4 share), which now rides the same emitter — so this also pins that the two
    /// emitters really are producing DIFFERENT tables rather than one of them having been wired to
    /// the wrong host const, a slip nothing else on a CPU-only box would see.
    #[test]
    fn the_emitted_codebooks_unpack_back_to_the_host_consts() {
        // The words are emitted as `{:#010x}u` — "0x", exactly 8 hex digits, then the `u` suffix.
        // (Matching that shape skips the `0xFFu` byte mask in the same line.)
        let unpack = |src: &str| -> Vec<i8> {
            let words: Vec<u32> = src
                .split("0x")
                .skip(1)
                .filter(|t| {
                    t.len() > 8
                        && t.as_bytes()[8] == b'u'
                        && t[..8].bytes().all(|c| c.is_ascii_hexdigit())
                })
                .map(|t| u32::from_str_radix(&t[..8], 16).expect("8 hex digits per emitted word"))
                .collect();
            assert_eq!(words.len(), 4, "four packed codebook words:\n{src}");
            (0..16)
                .map(|i: usize| (words[i >> 2] >> ((i & 3) * 8)) as u8 as i8)
                .collect()
        };
        let iq4 = super::iq4nl_codebook_src();
        assert!(
            iq4.contains("GENERATED from infr_gguf::dequant::KVALUES_IQ4NL"),
            "the emitted table must name its source of truth:\n{iq4}"
        );
        assert!(
            iq4.contains("int kv_iq4nl(int idx)"),
            "accessor name:\n{iq4}"
        );
        assert_eq!(
            unpack(&iq4),
            infr_gguf::dequant::KVALUES_IQ4NL.to_vec(),
            "emitted codebook must unpack to the host KVALUES_IQ4NL"
        );
        let fp4 = super::mxfp4_codebook_src();
        assert!(
            fp4.contains("GENERATED from infr_gguf::dequant::KVALUES_MXFP4"),
            "the emitted table must name its source of truth:\n{fp4}"
        );
        assert!(
            fp4.contains("int kv_mxfp4(int idx)"),
            "accessor name:\n{fp4}"
        );
        assert_eq!(
            unpack(&fp4),
            infr_gguf::dequant::KVALUES_MXFP4.to_vec(),
            "emitted codebook must unpack to the host KVALUES_MXFP4"
        );
        assert_ne!(
            infr_gguf::dequant::KVALUES_IQ4NL,
            infr_gguf::dequant::KVALUES_MXFP4,
            "the two codebooks are different tables — neither accessor may serve the other format"
        );
    }

    /// The grids are ~33 KiB of generated table text, and a slip in the emitter — a wrong element
    /// count in the declared bound, a truncated table, `{:#x}` losing a leading zero — is exactly
    /// the kind of thing no CPU-only check would otherwise see and that the GPU parity tests would
    /// only report as "IQ2_S is wrong somewhere". So parse the emitted declarations back out (name,
    /// declared length, and every literal) and require each to BE the host static from
    /// `infr_core::iquant_grids`, element for element. R6 adds the 2048-entry `g_iq1s` (half the
    /// emitted bytes on its own) and the `IQ1S_DELTA` addend.
    #[test]
    fn the_emitted_grids_parse_back_to_the_host_statics() {
        use infr_core::iquant_grids as g;
        let src = super::iquant_grid_src();
        // `__device__ static const <ty> <name>[<n>] = { <lit>, ... };` — pull the body of one.
        let body = |name: &str| -> (usize, Vec<u128>) {
            let head = src
                .split(&format!(" {name}["))
                .nth(1)
                .unwrap_or_else(|| panic!("no emitted table named {name}"));
            let (n, rest) = head.split_once(']').expect("declared length");
            let decl: usize = n.parse().expect("numeric declared length");
            let body = rest.split_once('{').expect("table body").1;
            let body = body.split_once('}').expect("table body end").0;
            let vals = body
                .split(',')
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(|t| {
                    let t = t.trim_end_matches(['u', 'l']);
                    match t.strip_prefix("0x") {
                        Some(h) => u128::from_str_radix(h, 16).expect("hex literal"),
                        None => t.parse::<u128>().expect("decimal literal"),
                    }
                })
                .collect();
            (decl, vals)
        };
        let check = |name: &str, want: Vec<u128>| {
            let (decl, got) = body(name);
            assert_eq!(decl, want.len(), "{name}: declared array length");
            assert_eq!(got.len(), want.len(), "{name}: emitted element count");
            assert_eq!(got, want, "{name}: emitted values must BE the host static");
        };
        check(
            "ksigns_iq2xs",
            g::KSIGNS_IQ2XS.iter().map(|&v| v as u128).collect(),
        );
        for (name, grid) in [
            ("g_iq2xxs", &g::IQ2XXS_GRID[..]),
            ("g_iq2xs", &g::IQ2XS_GRID[..]),
            ("g_iq2s", &g::IQ2S_GRID[..]),
            ("g_iq1s", &g::IQ1S_GRID[..]),
        ] {
            check(name, grid.iter().map(|&v| v as u128).collect());
        }
        for (name, grid) in [
            ("g_iq3xxs", &g::IQ3XXS_GRID[..]),
            ("g_iq3s", &g::IQ3S_GRID[..]),
        ] {
            check(name, grid.iter().map(|&v| v as u128).collect());
        }
        assert!(
            src.contains("GENERATED from infr_core::iquant_grids"),
            "the emitted tables must name their source of truth"
        );
        // R6: the IQ1 addend rides along with the grids and is the one number that separates this
        // family from R5's, so pin that the emitted literal parses back to the host constant.
        let lit = src
            .split("#define IQ1S_DELTA ")
            .nth(1)
            .and_then(|t| t.split_whitespace().next())
            .expect("emitted IQ1S_DELTA define");
        assert_eq!(
            lit.trim_end_matches('f')
                .parse::<f32>()
                .expect("f32 literal"),
            infr_gguf::dequant::IQ1S_DELTA,
            "emitted IQ1S_DELTA must BE the host constant"
        );
    }

    /// Every kernel the R5/R6/R7 routing tables can name must actually exist in the assembled
    /// module. `exec.rs` looks kernels up by STRING at dispatch time (`hipModuleGetFunction`), so a
    /// format registered in `native_i8_fmt`/`native_wmma_fmt`/the MoE mappers but never
    /// instantiated in the source here fails only on the box, on the one model that uses it.
    #[test]
    fn the_wdec_seam_kernels_are_all_instantiated() {
        let src = super::hip_source();
        for f in [
            // R5 grid quants.
            "iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", // R6 IQ1 + ternary quants.
            "iq1s", "iq1m", "tq10", "tq20", "q20", // R7 fp4 microscaling quants.
            "mxfp4", "nvfp4",
        ] {
            for k in [
                format!("GEN_LINEAR({f})"),
                format!("GEN_EMBED({f})"),
                format!("GEN_DEQF16({f})"),
                format!("GEN_LINEAR_I8_WDEC({f})"),
                format!("GEN_I8ACC_WDEC({f})"),
                format!("GEN_MOE_GATE_UP({f})"),
                format!("GEN_MOE_DOWN({f})"),
                format!("GEN_MOE_GATE_UP_ROUTED({f})"),
                format!("GEN_MOE_DOWN_ROUTED({f})"),
                format!("wmma_i8_{f}_1x1"),
                format!("wmma_i8_{f}_2x1"),
                format!("wmma_i8_{f}_2x2"),
                format!("wdec_{f}("),
            ] {
                assert!(src.contains(&k), "missing HIP instantiation: {k}");
            }
        }
    }
}
