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
    ATTENTION_SPLIT,
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
    int x_stride       // per-row stride in elements; 0 = packed (n_head * head_dim)
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
    int doff = head * head_dim;
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
        dst[doff + i] = x[xoff + i] * rms * __half2float(weight[i]);
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
// lever, docs/cpu-perf.md). Covered formats: Q4_K, Q6_K, Q8_0, Q5_0 (the set a Q4_K_M GGUF
// uses — unsloth's gemma-3 Q4_K_M packs q/k/v + ffn_gate/up as Q5_0; F16 is already native
// via `linear_f16`).
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
GEN_LINEAR(q50)
GEN_EMBED(q80)
GEN_EMBED(q4k)
GEN_EMBED(q6k)
GEN_EMBED(q50)
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
GEN_DEQF16(q4k)
GEN_DEQF16(q6k)
GEN_DEQF16(q50)

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
// The 9 (GU × DN) combos over {q80, q4k, q6k} cover every mixed-precision MoE the covered formats
// produce; uncovered formats keep the dequant→f16 `moe_ffn_expert` fallback in exec.rs.
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
GEN_MOE_FFN(q80, q4k)
GEN_MOE_FFN(q80, q6k)
GEN_MOE_FFN(q4k, q80)
GEN_MOE_FFN(q4k, q4k)
GEN_MOE_FFN(q4k, q6k)
GEN_MOE_FFN(q6k, q80)
GEN_MOE_FFN(q6k, q4k)
GEN_MOE_FFN(q6k, q6k)
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
// Covered formats: Q8_0, Q4_K, Q6_K, Q5_0 (the Q4_K_M set). `rf16b`/`k4` are defined in NATIVE_DECODE
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

// ── Q4_K: 256 elems / 144 bytes; sub-block 32; code 0..15; value = d·sc·code + dmin·(−mm). ──
extern "C" __global__ void linear_i8_q4k(
    const signed char* __restrict__ qx,
    const float* __restrict__ xs,
    const unsigned char* __restrict__ w,
    float* __restrict__ dst,
    const float* __restrict__ resid,       // [m, out_f] residual to fold into the epilogue (null = none)
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
    // Slice-32 fused residual: fold the following Op::Add into the GEMV epilogue when `resid` is
    // bound (null = standalone GEMV, bit-identical to the pre-fusion write). Only lane 0 writes, so
    // the in-place `dst == resid` case is a safe read-then-write of one element (no cross-lane race).
    if (tid == 0) {
        long oi = (long)row * out_f + o;
        dst[oi] = resid ? (acc + resid[oi]) : acc;
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
const RMSNORM_QUANT_I8: &str = r#"
extern "C" __global__ void rmsnorm_quant_i8_32(
    const float* __restrict__ x,       // [rows, dim] — RAW pre-norm F32 activation
    const __half* __restrict__ weight, // [dim] — dequantized F16 norm weight
    signed char* __restrict__ qx,      // [rows, dim] — int8 codes
    float* __restrict__ xs,            // [rows, dim/32] — per-32-block scales
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
// reference with a widened int8 tolerance (docs/perf.md). Covered GU/DN formats: Q8_0, Q4_K, Q6_K
// (the Q4_K_M expert-bank set); uncovered formats keep the Phase-3 `moe_ffn_expert_*` fallback.
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

// Q4_K per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q4k`.
__device__ __forceinline__ float i8acc_q4k(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int s = blk & 7;
        const unsigned char* b = w + (long)super * 144;
        float d = rf16b(b);
        float dmin = rf16b(b + 2);
        const unsigned char* scales = b + 4;
        const unsigned char* qs = b + 16;
        int sc, mm; k4(scales, s, &sc, &mm);
        const unsigned char* qbase = qs + (s >> 1) * 32;
        int hi = s & 1;
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
    return acc;
}

// Q6_K per-lane int8 dp4a accumulation for output row `o` — mirrors `linear_i8_q6k`.
__device__ __forceinline__ float i8acc_q6k(
    const signed char* __restrict__ qxr, const float* __restrict__ xsr,
    const unsigned char* __restrict__ w, int o, int nb, int tid) {
    int spr = nb >> 3;
    float acc = 0.0f;
    for (int blk = tid; blk < nb; blk += 32) {
        long super = (long)o * spr + (blk >> 3);
        int w32 = blk & 7;
        const unsigned char* b = w + (long)super * 210;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char* scales = (const signed char*)(b + 192);
        float d = rf16b(b + 208);
        float sx = xsr[blk];
        for (int hh = 0; hh < 2; hh++) {
            int sub16 = w32 * 2 + hh;
            int sc = (int)scales[sub16];
            int within0 = sub16 * 16;
            int half = within0 >> 7;
            int o127 = within0 & 127;
            int region = o127 >> 5;
            int l0 = o127 & 31;
            int qlo = half * 64;
            int qho = half * 32;
            const int* xp = (const int*)(qxr + blk * 32 + hh * 16);
            int idot = 0, isum = 0;
            for (int k = 0; k < 4; k++) {
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
    return acc;
}

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
GEN_MOE_GATE_UP(q4k)
GEN_MOE_GATE_UP(q6k)
GEN_MOE_DOWN(q80)
GEN_MOE_DOWN(q4k)
GEN_MOE_DOWN(q6k)
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
// to Slice-15 and the parity-tested `linear_i8_*` GEMV (same `k4`/`rf16b`, same nibble/region math).
// The Q4_K/Q6_K/Q5_0 min term (`dmin·(−mm)·Σqx` / `d·(−32s)·Σqx` / `d·(−16)·Σqx`) is a second WMMA
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

// ── Q6_K: 256/super-block, 16 sub-blocks of 16 (= 1 K-tile each), int8 scale, code 0..63. ──
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
    signed char wc[CN][16]; \
    float wsc[CN], wmn[CN]; \
    const i4v ones = {0x01010101, 0x01010101, 0x01010101, 0x01010101}; \
    for (int sb = 0; sb < n16; sb++) { \
        int blk32 = sb >> 1; \
        for (int c = 0; c < (CN); c++) { \
            int col = col_base + c * 16 + (lane & 15); \
            if (col < out_f) { \
                long super = (long)col * spr + (blk32 >> 3); \
                int w32 = blk32 & 7; \
                int hh = sb & 1; \
                int sub16 = w32 * 2 + hh; \
                const unsigned char* b = w + super * 210; \
                const unsigned char* ql = b; \
                const unsigned char* qh = b + 128; \
                const signed char* scales = (const signed char*)(b + 192); \
                float d = rf16b(b + 208); \
                int sc = (int)scales[sub16]; \
                wsc[c] = d * (float)sc; \
                wmn[c] = d * (float)(-32 * sc); \
                int within0 = sub16 * 16; \
                int h128 = within0 >> 7; \
                int o127 = within0 & 127; \
                int region = o127 >> 5; \
                int l0 = o127 & 31; \
                int qlo = h128 * 64, qho = h128 * 32; \
                for (int rr = 0; rr < 16; rr++) { \
                    int l = l0 + rr; \
                    int cc; \
                    if (region == 0)      cc = (ql[qlo + l] & 0x0F)      | ((qh[qho + l] & 3) << 4); \
                    else if (region == 1) cc = (ql[qlo + 32 + l] & 0x0F) | (((qh[qho + l] >> 2) & 3) << 4); \
                    else if (region == 2) cc = (ql[qlo + l] >> 4)        | (((qh[qho + l] >> 4) & 3) << 4); \
                    else                  cc = (ql[qlo + 32 + l] >> 4)   | (((qh[qho + l] >> 6) & 3) << 4); \
                    wc[c][rr] = (signed char)cc; \
                } \
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
GEN_WMMA_Q4K(wmma_i8_q4k_1x1, 1, 1)
GEN_WMMA_Q4K(wmma_i8_q4k_2x1, 2, 1)
GEN_WMMA_Q4K(wmma_i8_q4k_2x2, 2, 2)
GEN_WMMA_Q6K(wmma_i8_q6k_1x1, 1, 1)
GEN_WMMA_Q6K(wmma_i8_q6k_2x1, 2, 1)
GEN_WMMA_Q6K(wmma_i8_q6k_2x2, 2, 2)
GEN_WMMA_Q50(wmma_i8_q50_1x1, 1, 1)
GEN_WMMA_Q50(wmma_i8_q50_2x1, 2, 1)
GEN_WMMA_Q50(wmma_i8_q50_2x2, 2, 2)
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
GEN_MOE_FFN_ROUTED(q80, q4k)
GEN_MOE_FFN_ROUTED(q80, q6k)
GEN_MOE_FFN_ROUTED(q4k, q80)
GEN_MOE_FFN_ROUTED(q4k, q4k)
GEN_MOE_FFN_ROUTED(q4k, q6k)
GEN_MOE_FFN_ROUTED(q6k, q80)
GEN_MOE_FFN_ROUTED(q6k, q4k)
GEN_MOE_FFN_ROUTED(q6k, q6k)

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
GEN_MOE_GATE_UP_ROUTED(q4k)
GEN_MOE_GATE_UP_ROUTED(q6k)

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
GEN_MOE_DOWN_ROUTED(q4k)
GEN_MOE_DOWN_ROUTED(q6k)
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
