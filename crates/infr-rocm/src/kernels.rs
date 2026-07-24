//! HIP kernel-source assembly and hiprtc compilation.
//!
//! Each kernel is a `__global__` function taking device pointers. Most operate on f16 or f32
//! buffers — uncovered quantized weights are dequantized to f16 on the host BEFORE they reach a
//! kernel (see `exec.rs`'s dequant cache), so those kernels stay format-agnostic and simple. The
//! `NATIVE_DECODE` kernels (Phase 3, Q4_K/Q6_K/Q8_0) are the exception: they read the RAW quant
//! bytes and decode each block in-kernel, so no f16 cache is materialized (VRAM ≈ quant_size).
//!
//! On first use each kernel name is fetched via `hipModuleGetFunction` and cached in a
//! `HashMap`. The module is compiled once at backend init via `hiprtcCompileProgram`.

use crate::ffi;
use infr_core::error::{Error, Result};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CString};
use std::sync::Mutex;

fn be(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

// ── Kernel source ────────────────────────────────────────────────────────────

/// Assemble the complete HIP source string from its parts.
pub fn hip_source() -> String {
    let mut s = String::with_capacity(128 * 1024);
    for part in HIP_PARTS {
        s.push_str(part);
    }
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
    MOE_FFN,
    CONV1D_SILU,
    DELTANET,
    MOE_SHARED_EXPERT_ADD,
    NATIVE_DECODE,
    INT8_DECODE,
    WMMA_PREFILL,
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
    int x_stride       // per-row stride in elements; 0 = packed (n_head * head_dim)
) {
    int head = blockIdx.x * blockDim.x + threadIdx.x;
    int total_heads = rows * n_head;
    if (head >= total_heads) return;
    int r = head / n_head;
    int h = head % n_head;
    int pos = positions[r];
    // Read base: strided input packs each head into an `x_stride/n_head`-wide block (query is the
    // first head_dim elements), matching the qwen35 interleaved q+g buffer. Packed input (x_stride
    // == 0) reads the natural head slice. Write base is ALWAYS the packed [rows, n_head, head_dim]
    // slot — mirrors infr-cpu QkNormRope and the Metal `qknormrope_f32` kernel.
    int head_stride = (x_stride > 0) ? (x_stride / n_head) : head_dim;
    int xoff = (x_stride > 0) ? (r * x_stride + h * head_stride) : (head * head_dim);
    int doff = head * head_dim;
    // rmsnorm over the head_dim query slice
    float ss = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float v = x[xoff + i];
        ss += v * v;
    }
    ss /= (float)head_dim;
    float rms = 1.0f / sqrtf(ss + eps);
    // Pass-through dims [rope_dim, head_dim): normed (× weight), no rotation.
    for (int i = rope_dim; i < head_dim; i++) {
        dst[doff + i] = x[xoff + i] * rms * __half2float(weight[i]);
    }
    // rope (NEOX split-half pairs (i, i+half)) on the first rope_dim elements, from normed values.
    int half = rope_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
        if (freq_factors != nullptr) {
            freq /= freq_factors[i]; // proportional RoPE divides the per-pair angle (matches CPU/Metal)
        }
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float a = x[xoff + i]        * rms * __half2float(weight[i]);
        float b = x[xoff + i + half] * rms * __half2float(weight[i + half]);
        dst[doff + i]        = a * c - b * s;
        dst[doff + i + half] = a * s + b * c;
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
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* sr = src + src_off + r * src_stride;
    float* dr = dst + dst_off + r * dst_stride;
    for (int i = 0; i < n; i++) {
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
extern "C" __global__ void argmax(
    const float* __restrict__ x, // [rows, n]
    float* __restrict__ dst,     // [rows] — u32 bit-pattern in f32 slot
    int rows,
    int n
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    const float* xr = x + row * n;
    // Sentinel below any real logit; threads whose strided slice is empty (n < blockDim.x) keep it,
    // and it loses every reduction step against a real candidate. (hiprtc has no INFINITY macro.)
    float best_val = -3.402823466e+38f;
    int best_idx = 0;
    for (int i = tid; i < n; i += nt) {
        float v = xr[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    __shared__ float sval[256];
    __shared__ int sidx[256];
    sval[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            // Keep the strictly-greater value; on a tie keep the LOWER index (first-max rule).
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Store u32 bit-pattern in an f32 slot (the runner reads as u32)
        dst[row] = __int_as_float(sidx[0]);
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int effective_src_stride = (src_stride > 0) ? src_stride : cache_stride;
    int cache_row = row_offset + row;
    const float* sr = src + row * effective_src_stride;
    __half* cr = cache + cache_row * cache_stride;
    for (int i = 0; i < cache_stride; i++) {
        cr[i] = __float2half(sr[i]);
    }
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
// lever, docs/cpu-perf.md). Covered formats: Q4_K, Q6_K, Q8_0 (the set a Q4_K_M GGUF
// uses; F16 is already native via `linear_f16`).
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

// ── Q8_0: 32 elems / 34 bytes = [half d][int8 qs[32]]; y = d*q8 (code = q8+128). ──
__device__ __forceinline__ float deq_q80(const unsigned char* w, long i) {
    long blk = i >> 5;              // / 32
    int within = (int)(i & 31);
    const unsigned char* b = w + blk * 34;
    float d = rf16b(b);
    int code = (int)((signed char)b[2 + within]) + 128; // biased +128 (dequant_block)
    return fin(d, code, d * (float)(-128));             // sc = d*1, mn = d*(-128)
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
GEN_LINEAR(q4k)
GEN_LINEAR(q6k)
GEN_EMBED(q80)
GEN_EMBED(q4k)
GEN_EMBED(q6k)
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
// Covered formats: Q8_0, Q4_K, Q6_K (the Q4_K_M set). `rf16b`/`k4` are defined in NATIVE_DECODE
// (this part is assembled after it). Uncovered formats keep the Phase-3 / dequant→f16 fallback.
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

// ── Q8_0: 32 elems / 34 bytes = [half d][int8 qs[32]]; value = d * qs (signed int8). ──
extern "C" __global__ void linear_i8_q80(
    const signed char* __restrict__ qx,   // [m, in_f]
    const float* __restrict__ xs,          // [m, in_f/32]
    const unsigned char* __restrict__ w,   // raw Q8_0 weight bytes (pre-advanced past w_off)
    float* __restrict__ dst,               // [m, out_f]
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
    if (tid == 0) dst[(long)row * out_f + o] = acc;
}

// ── Q4_K: 256 elems / 144 bytes; sub-block 32; code 0..15; value = d·sc·code + dmin·(−mm). ──
extern "C" __global__ void linear_i8_q4k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    int m, int in_f, int out_f
) {
    int o = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    int nb = in_f >> 5;
    int spr = nb >> 3;             // Q4_K super-blocks (256 elems) per output row
    const signed char* qxr = qx + (long)row * in_f;
    const float* xsr = xs + (long)row * nb;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);   // global super-block for (output row o, this 32-block)
        int s = blk & 7;           // sub-block 0..7 (== the 32-block)
        const unsigned char* b = w + (long)super * 144;
        float d = rf16b(b);
        float dmin = rf16b(b + 2);
        const unsigned char* scales = b + 4;
        const unsigned char* qs = b + 16;
        int sc, mm; k4(scales, s, &sc, &mm);
        const unsigned char* qbase = qs + (s >> 1) * 32;   // nibble byte base
        int hi = s & 1;                                    // high nibble for odd sub-blocks
        const int* xp = (const int*)(qxr + blk * 32);
        int idot = 0, isum = 0;
        for (int k = 0; k < 8; k++) {
            const unsigned char* q = qbase + k * 4;
            int wpack;
            if (hi) {
                wpack = (int)(q[0] >> 4) | ((int)(q[1] >> 4) << 8)
                      | ((int)(q[2] >> 4) << 16) | ((int)(q[3] >> 4) << 24);
            } else {
                wpack = (int)(q[0] & 0xF) | ((int)(q[1] & 0xF) << 8)
                      | ((int)(q[2] & 0xF) << 16) | ((int)(q[3] & 0xF) << 24);
            }
            idot = idot4(xp[k], wpack, idot);
            isum = idot4(xp[k], 0x01010101, isum);
        }
        float sx = xsr[blk];
        acc += (d * (float)sc) * sx * (float)idot + (dmin * (float)(-mm)) * sx * (float)isum;
    }
    acc = wave_sum32(acc);
    if (tid == 0) dst[(long)row * out_f + o] = acc;
}

// ── Q6_K: 256 elems / 210 bytes; sub-block 16 (int8 scale); code 0..63; value = d·s·code + d·(−32s). ──
extern "C" __global__ void linear_i8_q6k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    int m, int in_f, int out_f
) {
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
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char* scales = (const signed char*)(b + 192);
        float d = rf16b(b + 208);
        float sx = xsr[blk];
        // The 32-block spans two 16-element sub-blocks, each with its own int8 scale.
        for (int hh = 0; hh < 2; hh++) {
            int sub16 = w32 * 2 + hh;      // 0..15
            int sc = (int)scales[sub16];
            int within0 = sub16 * 16;      // first element (0..255)
            int half = within0 >> 7;       // 0..1 (which 128-half)
            int o127 = within0 & 127;
            int region = o127 >> 5;        // 0..3
            int l0 = o127 & 31;            // 0 or 16 within the region
            int qlo = half * 64;
            int qho = half * 32;
            const int* xp = (const int*)(qxr + blk * 32 + hh * 16);
            int idot = 0, isum = 0;
            for (int k = 0; k < 4; k++) {  // 4 groups of 4 = 16
                int code[4];
                for (int r = 0; r < 4; r++) {
                    int l = l0 + k * 4 + r;
                    int c;
                    if (region == 0)      c = (ql[qlo + l] & 0x0F)       | ((qh[qho + l] & 3) << 4);
                    else if (region == 1) c = (ql[qlo + 32 + l] & 0x0F)  | (((qh[qho + l] >> 2) & 3) << 4);
                    else if (region == 2) c = (ql[qlo + l] >> 4)         | (((qh[qho + l] >> 4) & 3) << 4);
                    else                  c = (ql[qlo + 32 + l] >> 4)    | (((qh[qho + l] >> 6) & 3) << 4);
                    code[r] = c;
                }
                int wpack = code[0] | (code[1] << 8) | (code[2] << 16) | (code[3] << 24);
                idot = idot4(xp[k], wpack, idot);
                isum = idot4(xp[k], 0x01010101, isum);
            }
            acc += (d * (float)sc) * sx * (float)idot + (d * (float)(-32 * sc)) * sx * (float)isum;
        }
    }
    acc = wave_sum32(acc);
    if (tid == 0) dst[(long)row * out_f + o] = acc;
}
"#;

// ── Matrix-core (WMMA) int8 prefill GEMM (Phase 5) ───────────────────────────
//
// The Phase-4 `linear_i8_*` GEMV grids one wave32 block per (output row `o`, activation row) — so a
// weight column is DECODED ONCE PER ACTIVATION ROW. For prefill (m>1) that redundant decode is the
// ceiling. These kernels move prefill onto the RDNA3 wave32 matrix cores
// (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`, signed int8 → int32, `16x16x16`): one wave32 block
// computes a 16(m-rows)×16(out_f) output tile, so a weight column is decoded ONCE PER 16 ROWS (16×
// fewer decodes at m≥16). The activation is the SAME int8 quantization (`quant_i8_32`) the GEMV uses,
// and the integer dot is EXACT — this is the identical int8 precision as the Phase-4 prefill, only
// the f32 scale-after accumulation ORDER over blocks differs, so the goldens are the closest possible
// to the Phase-4 blessed state (verified on-device, not assumed).
//
// Fragment layout (RDNA3 wave32, empirically confirmed by `examples/wmma_probe`):
//   * A fragment: lane l feeds row (l%16) of the M×K tile — 16 K-values packed 4×int8/int32 (i4v).
//   * B fragment: lane l feeds col (l%16) of the K×N tile — 16 K-values packed the same way.
//   * D/C accumulator: 8 int32/lane; element e of lane l is output (row = 2*e + l/16, col = l%16).
// int8 is SIGNED (neg_a = neg_b = true); unsigned would 256× the result (probe-verified).
//
// The per-format weight code + block-scale/min extraction mirrors the parity-tested `linear_i8_*`
// exactly (same `k4`/`rf16b`, same nibble/region math), so the codes fed to the matrix core are
// bit-identical to the GEMV. The Q4_K/Q6_K min term (`dmin·(−mm)·Σqx` / `d·(−32s)·Σqx`) is produced
// by a second WMMA against an all-ones B fragment — its int32 result is `Σ_k qx[row][k]` for the
// output row, exactly the `isum` the GEMV computes — then scaled after. Q8_0 has no min term.
//
// Tiles are one wave32 each; grid = (ceil(out_f/16), ceil(m/16)). `in_f` is always 32-aligned for
// the covered formats, so K needs no padding; m and out_f edges are masked (out-of-range rows/cols
// load zero and skip the store). No LDS: each lane loads its own row/column from global (the naive,
// correctness-first tiling — occupancy/LDS tuning is a later Phase-5 lever).
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

// Load the A K-fragment (16 int8 of activation row `row_in` at absolute element offset `koff`),
// or zero if this lane's input row is past the m-edge. `koff` already includes `row_in*in_f`.
static __device__ __forceinline__ i4v load_a(const signed char* qx, int row_in, int m, long koff) {
    if (row_in >= m) return (i4v){0, 0, 0, 0};
    return pack16(qx + koff);
}

// ── Q8_0 prefill: 32 elems/block = 2 K-tiles, scale d, no min. ──
extern "C" __global__ void wmma_i8_q80(
    const signed char* __restrict__ qx,   // [m, in_f]
    const float* __restrict__ xs,          // [m, nblk]
    const unsigned char* __restrict__ w,   // raw Q8_0 bytes (pre-advanced)
    float* __restrict__ dst,               // [m, out_f]
    int m, int in_f, int out_f
) {
    int lane = threadIdx.x;
    int half = lane >> 4;
    int col = blockIdx.x * 16 + (lane & 15);   // output feature o
    int row_in = blockIdx.y * 16 + (lane & 15); // A-fragment input row
    int nblk = in_f >> 5;
    float acc[8]; for (int e = 0; e < 8; e++) acc[e] = 0.0f;
    signed char wc[32];
    for (int blk = 0; blk < nblk; blk++) {
        float wsc = 0.0f;
        if (col < out_f) {
            const unsigned char* b = w + ((long)col * nblk + blk) * 34;
            wsc = rf16b(b);
            for (int j = 0; j < 32; j++) wc[j] = (signed char)b[2 + j];
        } else {
            for (int j = 0; j < 32; j++) wc[j] = 0;
        }
        long arow = (long)row_in * in_f + (long)blk * 32;
        i8v dotacc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < 2; kt++) {
            i4v a = load_a(qx, row_in, m, arow + kt * 16);
            i4v b = pack16(wc + kt * 16);
            dotacc = wmma_dot(a, b, dotacc);
        }
        for (int e = 0; e < 8; e++) {
            int re = blockIdx.y * 16 + 2 * e + half;
            float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f;
            acc[e] += axs * wsc * (float)dotacc[e];
        }
    }
    for (int e = 0; e < 8; e++) {
        int re = blockIdx.y * 16 + 2 * e + half;
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[e];
    }
}

// ── Q4_K prefill: 256/super-block, 8 sub-blocks of 32 (= 2 K-tiles each), scale d·sc + min dmin·(−mm). ──
extern "C" __global__ void wmma_i8_q4k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    int m, int in_f, int out_f
) {
    int lane = threadIdx.x;
    int half = lane >> 4;
    int col = blockIdx.x * 16 + (lane & 15);
    int row_in = blockIdx.y * 16 + (lane & 15);
    int nblk = in_f >> 5;
    int spr = nblk >> 3;              // super-blocks per output row
    float acc[8]; for (int e = 0; e < 8; e++) acc[e] = 0.0f;
    signed char wc[32];
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    for (int blk = 0; blk < nblk; blk++) {
        float wsc = 0.0f, wmn = 0.0f;
        if (col < out_f) {
            long super = (long)col * spr + (blk >> 3);
            int s = blk & 7;
            const unsigned char* b = w + super * 144;
            float d = rf16b(b), dmin = rf16b(b + 2);
            int sc, mm; k4(b + 4, s, &sc, &mm);
            wsc = d * (float)sc;
            wmn = dmin * (float)(-mm);
            const unsigned char* qbase = (b + 16) + (s >> 1) * 32;
            int hi = s & 1;
            for (int p = 0; p < 32; p++)
                wc[p] = (signed char)(hi ? (qbase[p] >> 4) : (qbase[p] & 0x0F));
        } else {
            for (int p = 0; p < 32; p++) wc[p] = 0;
        }
        long arow = (long)row_in * in_f + (long)blk * 32;
        i8v dotacc = {0,0,0,0,0,0,0,0};
        i8v sumacc = {0,0,0,0,0,0,0,0};
        for (int kt = 0; kt < 2; kt++) {
            i4v a = load_a(qx, row_in, m, arow + kt * 16);
            i4v b = pack16(wc + kt * 16);
            dotacc = wmma_dot(a, b, dotacc);
            sumacc = wmma_dot(a, ones, sumacc);
        }
        for (int e = 0; e < 8; e++) {
            int re = blockIdx.y * 16 + 2 * e + half;
            float axs = (re < m) ? xs[(long)re * nblk + blk] : 0.0f;
            acc[e] += axs * (wsc * (float)dotacc[e] + wmn * (float)sumacc[e]);
        }
    }
    for (int e = 0; e < 8; e++) {
        int re = blockIdx.y * 16 + 2 * e + half;
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[e];
    }
}

// ── Q6_K prefill: 256/super-block, 16 sub-blocks of 16 (= 1 K-tile each), int8 scale, code 0..63. ──
extern "C" __global__ void wmma_i8_q6k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    int m, int in_f, int out_f
) {
    int lane = threadIdx.x;
    int half = lane >> 4;
    int col = blockIdx.x * 16 + (lane & 15);
    int row_in = blockIdx.y * 16 + (lane & 15);
    int nblk = in_f >> 5;
    int spr = nblk >> 3;
    int n16 = in_f >> 4;             // 16-element sub-blocks (one WMMA K-tile each)
    float acc[8]; for (int e = 0; e < 8; e++) acc[e] = 0.0f;
    signed char wc[16];
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    for (int sb = 0; sb < n16; sb++) {
        int blk32 = sb >> 1;         // 32-block index → the activation scale block
        float wsc = 0.0f, wmn = 0.0f;
        if (col < out_f) {
            long super = (long)col * spr + (blk32 >> 3);
            int w32 = blk32 & 7;
            int hh = sb & 1;
            int sub16 = w32 * 2 + hh;    // 0..15 within the super-block
            const unsigned char* b = w + super * 210;
            const unsigned char* ql = b;
            const unsigned char* qh = b + 128;
            const signed char* scales = (const signed char*)(b + 192);
            float d = rf16b(b + 208);
            int sc = (int)scales[sub16];
            wsc = d * (float)sc;
            wmn = d * (float)(-32 * sc);
            int within0 = sub16 * 16;
            int h128 = within0 >> 7;
            int o127 = within0 & 127;
            int region = o127 >> 5;
            int l0 = o127 & 31;
            int qlo = h128 * 64, qho = h128 * 32;
            for (int r = 0; r < 16; r++) {
                int l = l0 + r;
                int c;
                if (region == 0)      c = (ql[qlo + l] & 0x0F)      | ((qh[qho + l] & 3) << 4);
                else if (region == 1) c = (ql[qlo + 32 + l] & 0x0F) | (((qh[qho + l] >> 2) & 3) << 4);
                else if (region == 2) c = (ql[qlo + l] >> 4)        | (((qh[qho + l] >> 4) & 3) << 4);
                else                  c = (ql[qlo + 32 + l] >> 4)   | (((qh[qho + l] >> 6) & 3) << 4);
                wc[r] = (signed char)c;
            }
        } else {
            for (int r = 0; r < 16; r++) wc[r] = 0;
        }
        long koff = (long)row_in * in_f + (long)sb * 16;
        i4v a = load_a(qx, row_in, m, koff);
        i4v b = pack16(wc);
        i8v dotacc = wmma_dot(a, b, (i8v){0,0,0,0,0,0,0,0});
        i8v sumacc = wmma_dot(a, ones, (i8v){0,0,0,0,0,0,0,0});
        for (int e = 0; e < 8; e++) {
            int re = blockIdx.y * 16 + 2 * e + half;
            float axs = (re < m) ? xs[(long)re * nblk + blk32] : 0.0f;
            acc[e] += axs * (wsc * (float)dotacc[e] + wmn * (float)sumacc[e]);
        }
    }
    for (int e = 0; e < 8; e++) {
        int re = blockIdx.y * 16 + 2 * e + half;
        if (re < m && col < out_f) dst[(long)re * out_f + col] = acc[e];
    }
}
"#;

// ── Module cache ─────────────────────────────────────────────────────────────

/// Compiled HIP module + kernel-function cache.
pub struct Pipelines {
    module: ffi::hipModule_t,
    /// Kernel name → function handle (lazily fetched).
    cache: Mutex<HashMap<&'static str, ffi::hipFunction_t>>,
}

unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

impl Pipelines {
    /// Compile the assembled HIP source via hiprtc and load the resulting module.
    // `_device` is accepted for call-site symmetry with the other backends; the active device is
    // already selected via `hipSetDevice` before `build`, and hiprtc targets the arch via options.
    pub fn build(_device: c_int) -> Result<Self> {
        let src = hip_source();
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
        // plain auto-detect target compiles it in every launch context. `_device` is accepted for
        // call-site symmetry; the active device is already selected before `build`.
        let std_flag = CString::new("-std=c++17").unwrap();
        let opts: [*const c_char; 1] = [std_flag.as_ptr()];
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

        // Load the code object into a module
        let mut module: ffi::hipModule_t = std::ptr::null_mut();
        let rc = unsafe {
            ffi::hipModuleLoadData(&mut module, code.as_ptr() as *const std::ffi::c_void)
        };
        if rc != ffi::HIP_SUCCESS {
            return Err(be(format!("hipModuleLoadData: rc={rc}")));
        }

        Ok(Self {
            module,
            cache: Mutex::new(HashMap::new()),
        })
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
    }
}
