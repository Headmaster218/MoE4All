// ---- elementwise ----
kernel void advance_position_i32(device int* position [[buffer(0)]]) {
    position[0] += 1;
}

kernel void add_f32(device const float* a   [[buffer(0)]],
                    device const float* b   [[buffer(1)]],
                    device float*       dst [[buffer(2)]],
                    constant uint&      n   [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = a[gid] + b[gid];
}

// Broadcast bias add (Qwen2/2.5 q/k/v `Wx + b`): dst[i] = x[i] + bias[i % n] over `total = rows*n`
// elements. Params: n = bias length / row width, total = rows*n.
struct AddBiasParams { uint n; uint total; };
kernel void add_bias_f32(device const float* x    [[buffer(0)]],
                         device const float* bias [[buffer(1)]],
                         device float*       dst  [[buffer(2)]],
                         constant AddBiasParams& p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < p.total) dst[gid] = x[gid] + bias[gid % p.n];
}

// Broadcast multiply (diffusion-gemma router input scale): dst[i] = x[i] * vec[i % n] over
// `total = rows*n` elements. The multiplicative twin of `add_bias_f32`.
struct MulVecParams { uint n; uint total; };
kernel void mul_vec_f32(device const float* x    [[buffer(0)]],
                        device const float* vec_ [[buffer(1)]],
                        device float*       dst  [[buffer(2)]],
                        constant MulVecParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < p.total) dst[gid] = x[gid] * vec_[gid % p.n];
}

struct ScaleParams { float s; uint n; };
kernel void scale_f32(device const float* x   [[buffer(0)]],
                      device float*       dst [[buffer(1)]],
                      constant ScaleParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < p.n) dst[gid] = x[gid] * p.s;
}

struct SoftcapParams { float cap; uint n; };
kernel void softcap_f32(device const float* x   [[buffer(0)]],
                        device float*       dst [[buffer(1)]],
                        constant SoftcapParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < p.n) dst[gid] = p.cap * tanh(x[gid] / p.cap);
}

// ---- norms: one SIMD group (32 lanes) per normalized group. Lanes stride the group, `simd_sum`
// reduces the sum-of-squares, then all 32 write the scaled output in parallel. (Decode has rows=1,
// so the old one-thread-per-row kernel ran the whole reduction on a single thread — pathological.)
struct RmsParams { uint rows; uint dim; float eps; };
kernel void rmsnorm_f32(device const float* x   [[buffer(0)]],
                        device const float* w   [[buffer(1)]],
                        device float*       dst [[buffer(2)]],
                        constant RmsParams& p   [[buffer(3)]],
                        uint gid  [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32u;
    if (row >= p.rows) return;
    uint base = row * p.dim;
    float ss = 0.0f;
    for (uint i = lane; i < p.dim; i += 32u) { float v = x[base + i]; ss += v * v; }
    ss = simd_sum(ss) / (float)p.dim;
    float s = 1.0f / sqrt(ss + p.eps);
    for (uint i = lane; i < p.dim; i += 32u) dst[base + i] = x[base + i] * s * w[i];
}

// Wide RMSNorm for DECODE (rows == 1): 8 simdgroups (256 threads) cooperate on the one row.
// The 32-lane kernel is latency-bound on its dim/32 serial loads — ~20 us per launch at
// dim=1152 (the counter profiler's first catch: gemma decode fires 105 of these per token,
// 20% of its GPU time). Partials fold through threadgroup memory; every thread re-sums the 8
// partials to skip a second barrier.
kernel void rmsnorm_wide_f32(device const float* x   [[buffer(0)]],
                             device const float* w   [[buffer(1)]],
                             device float*       dst [[buffer(2)]],
                             constant RmsParams& p   [[buffer(3)]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (row >= p.rows) return;
    uint base = row * p.dim;
    float ss = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        float v = x[base + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    float s = 1.0f / sqrt(tot / (float)p.dim + p.eps);
    for (uint i = tid; i < p.dim; i += 256u) dst[base + i] = x[base + i] * s * w[i];
}

// Wide decode RMSNorm with 16-byte loads/stores. Apple GPUs prefer the existing 256-thread
// launch width here: the float4 stream measured 23-43% faster at dim 2048..5376, while raising
// the threadgroup to Vulkan's 1024-thread form was slower. Host gating guarantees dim % 4 == 0,
// so every row base is naturally float4-aligned.
kernel void rmsnorm_vec4_f32(device const float4* x   [[buffer(0)]],
                             device const float4* w   [[buffer(1)]],
                             device float4*       dst [[buffer(2)]],
                             constant RmsParams& p    [[buffer(3)]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (row >= p.rows) return;
    uint n4 = p.dim / 4u;
    uint base = row * n4;
    float ss = 0.0f;
    for (uint i = tid; i < n4; i += 256u) {
        float4 v = x[base + i];
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    if (lane == 0u) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    float s = 1.0f / sqrt(tot / (float)p.dim + p.eps);
    for (uint i = tid; i < n4; i += 256u) dst[base + i] = x[base + i] * s * w[i];
}

// Mean-centred LayerNorm — llama.cpp's LLM_NORM (`ggml_norm`, then the weight multiply and bias
// add of `llm_graph_context::build_norm`):
//   mean = sum(x)/dim;  var = sum((x-mean)^2)/dim;  dst[i] = (x[i]-mean)*rsqrt(var+eps)*w[i]+b[i]
// `var` is the BIASED estimator (divided by dim, not dim-1) and `eps` sits INSIDE the sqrt, added
// to the variance — both read off `ggml_compute_forward_norm_f32`. One simdgroup per row like
// `rmsnorm_f32`, but TWO `simd_sum`s: the mean must land before any variance term can be formed.
// deepseek32's `indexer_k_norm` (`Op::LayerNorm`) is the only user — the family's one non-RMS norm.
kernel void layernorm_f32(device const float* x   [[buffer(0)]],
                          device const float* w   [[buffer(1)]],
                          device const float* b   [[buffer(2)]],
                          device float*       dst [[buffer(3)]],
                          constant RmsParams& p   [[buffer(4)]],
                          uint gid  [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32u;
    if (row >= p.rows) return;
    uint base = row * p.dim;
    float sum = 0.0f;
    for (uint i = lane; i < p.dim; i += 32u) sum += x[base + i];
    float mean = simd_sum(sum) / (float)p.dim;
    float vsum = 0.0f;
    for (uint i = lane; i < p.dim; i += 32u) { float d = x[base + i] - mean; vsum += d * d; }
    float s = 1.0f / sqrt(simd_sum(vsum) / (float)p.dim + p.eps);
    for (uint i = lane; i < p.dim; i += 32u) dst[base + i] = (x[base + i] - mean) * s * w[i] + b[i];
}

// Row-wise softmax: dst[r,:] = softmax(x[r,:] * scale), one threadgroup (8 simdgroups) per row —
// diffusion-gemma's in-graph self-conditioning (see docs/diffusion-gemma.md's Phase-B and the
// reference's `dg_canvas_embed`). Same wide-launch shape as `rmsnorm_wide_f32` since the row width
// (vocab) is large. NOTE: unlike the rest of this backend, this kernel is UNVERIFIED on real
// Metal hardware (added blind, following the CPU/Vulkan implementations — see infr-vulkan's
// `softmax.comp` for the sibling shader this mirrors).
struct SoftmaxParams { uint rows; uint dim; float scale; };
kernel void softmax_wide_f32(device const float* x   [[buffer(0)]],
                             device float*       dst [[buffer(1)]],
                             constant SoftmaxParams& p [[buffer(2)]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[8];
    if (row >= p.rows) return;
    uint base = row * p.dim;

    // row max (numerically stable exp)
    float m = -INFINITY;
    for (uint i = tid; i < p.dim; i += 256u) {
        m = max(m, x[base + i] * p.scale);
    }
    m = simd_max(m);
    if (lane == 0u) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m0 = max(max(max(red[0], red[1]), max(red[2], red[3])),
                   max(max(red[4], red[5]), max(red[6], red[7])));
    threadgroup_barrier(mem_flags::mem_threadgroup); // every thread read `red[]` before it's reused

    // row sum of exp(x*scale - m0)
    float s = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        s += exp(x[base + i] * p.scale - m0);
    }
    s = simd_sum(s);
    if (lane == 0u) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
    float inv = 1.0f / tot;

    for (uint i = tid; i < p.dim; i += 256u) {
        dst[base + i] = exp(x[base + i] * p.scale - m0) * inv;
    }
}

// per-head RMSNorm: one SIMD group (32 lanes) per (row, head), weight indexed within head_dim.
struct QkNormParams { uint rows; uint n_head; uint head_dim; float eps; };
kernel void qknorm_f32(device const float* x   [[buffer(0)]],
                       device const float* w   [[buffer(1)]],
                       device float*       dst [[buffer(2)]],
                       constant QkNormParams& p [[buffer(3)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint grp = gid / 32u;
    if (grp >= p.rows * p.n_head) return;
    uint base = grp * p.head_dim;
    float ss = 0.0f;
    for (uint i = lane; i < p.head_dim; i += 32u) { float v = x[base + i]; ss += v * v; }
    ss = simd_sum(ss) / (float)p.head_dim;
    float s = 1.0f / sqrt(ss + p.eps);
    for (uint i = lane; i < p.head_dim; i += 32u) dst[base + i] = x[base + i] * s * w[i];
}

// Qwen3.5 DeltaNet output normalization: per-head RMSNorm followed by an elementwise SiLU gate.
// This keeps qknorm_f32's exact 32-lane reduction and folds the dependent GatedAct dispatch into
// the store pass. x and dst may alias; simd_sum completes every lane's reads before stores begin.
kernel void gated_rmsnorm_f32(device const float* x    [[buffer(0)]],
                              device const float* w    [[buffer(1)]],
                              device const float* gate [[buffer(2)]],
                              device float*       dst  [[buffer(3)]],
                              constant QkNormParams& p [[buffer(4)]],
                              uint gid  [[thread_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint grp = gid / 32u;
    if (grp >= p.rows * p.n_head) return;
    uint base = grp * p.head_dim;
    float ss = 0.0f;
    for (uint i = lane; i < p.head_dim; i += 32u) {
        float v = x[base + i];
        ss += v * v;
    }
    float s = 1.0f / sqrt(simd_sum(ss) / (float)p.head_dim + p.eps);
    for (uint i = lane; i < p.head_dim; i += 32u) {
        uint at = base + i;
        float z = gate[at];
        float silu = z / (1.0f + exp(-z));
        dst[at] = x[at] * s * w[i] * silu;
    }
}

// Greedy argmax over `n` logits → token id (one 256-thread threadgroup, strided scan +
// threadgroup tree-reduce). Strict > keeps the lowest index on ties, matching the host argmax
// (same contract as the Vulkan argmax.comp). The id is written as a u32 bit-pattern into the
// f32 output slot — greedy decode reads back 4 bytes instead of the [vocab] logits.
struct ArgmaxParams { uint n; };
kernel void argmax_f32(device const float* logits [[buffer(0)]],
                       device uint*        out_id [[buffer(1)]],
                       constant ArgmaxParams& p   [[buffer(2)]],
                       uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    float best = -INFINITY;
    uint bi = 0u;
    for (uint i = t; i < p.n; i += 256u) {
        if (logits[i] > best) { best = logits[i]; bi = i; }
    }
    sval[t] = best;
    sidx[t] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s /= 2u) {
        if (t < s && sval[t + s] > sval[t]) {
            sval[t] = sval[t + s];
            sidx[t] = sidx[t + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { out_id[0] = sidx[0]; }
}

struct ArgmaxSplitParams { uint n; uint chunk; };
kernel void argmax_f32_stage1(device const float* logits [[buffer(0)]],
                              device float*       out_val [[buffer(1)]],
                              device uint*        out_idx [[buffer(2)]],
                              constant ArgmaxSplitParams& p [[buffer(3)]],
                              uint t [[thread_position_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    uint base = group * p.chunk;
    uint end = min(base + p.chunk, p.n);
    float best = -INFINITY;
    uint bi = base;
    for (uint i = base + t; i < end; i += 256u) {
        float v = logits[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    sval[t] = best;
    sidx[t] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s /= 2u) {
        if (t < s) {
            float v = sval[t + s];
            uint i = sidx[t + s];
            if (v > sval[t] || (v == sval[t] && i < sidx[t])) {
                sval[t] = v;
                sidx[t] = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { out_val[group] = sval[0]; out_idx[group] = sidx[0]; }
}

kernel void argmax_f32_stage2(device const float* values [[buffer(0)]],
                              device const uint*  indices [[buffer(1)]],
                              device uint*        out_id  [[buffer(2)]],
                              constant ArgmaxParams& p    [[buffer(3)]],
                              uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    float best = -INFINITY;
    uint bi = 0u;
    for (uint i = t; i < p.n; i += 256u) {
        float v = values[i];
        uint idx = indices[i];
        if (v > best || (v == best && idx < bi)) { best = v; bi = idx; }
    }
    sval[t] = best;
    sidx[t] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s /= 2u) {
        if (t < s) {
            float v = sval[t + s];
            uint i = sidx[t + s];
            if (v > sval[t] || (v == sval[t] && i < sidx[t])) {
                sval[t] = v;
                sidx[t] = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) { out_id[0] = sidx[0]; }
}

// GPU stochastic sampling over VOCAB-scale logits (Op::Sample): temperature + top-k + top-p,
// IDENTICAL order of operations to the host `sample_logits` (infr-cpu's Op::Sample arm) given the
// same uniform draw `u`, so the same `u` picks the same token (modulo exact-tie order — ties are
// legitimately unspecified, same caveat as the Vulkan sample_topk.comp reference).
//
// Correctness-first single-threadgroup version (this is the reference backend): `top_k` (bounded
// 2..=64 by the caller) selection is done via `top_k` sequential parallel-max reductions (each an
// argmax_f32-shaped strided-scan + threadgroup tree-reduce), skipping indices already selected in
// earlier rounds — descending order falls out for free since round `j` finds the (j+1)-th largest.
// This re-scans the `n` logits `top_k` times instead of Vulkan's one-pass radix select; fine for a
// per-token decode op where correctness, not throughput, is the bar. Phase 2 (single lane) mirrors
// the host: softmax(temp) over the selected set, nucleus (top-p) cutoff, inverse-CDF walk with `u`.
#define SAMPLE_KMAX 64u
struct SampleParams { uint n; uint top_k; float temp; float top_p; };
inline void sample_f32_finish(float uniform,
                              device uint* out_id,
                              constant SampleParams& p,
                              uint k,
                              threadgroup float* gval,
                              threadgroup uint* gidx) {
    float maxl = gval[0];
    float sum = 0.0f;
    for (uint j = 0u; j < k; j++) {
        float pr = exp((gval[j] - maxl) / p.temp);
        gval[j] = pr;
        sum += pr;
    }
    for (uint j = 0u; j < k; j++) { gval[j] /= sum; }
    float cum = 0.0f;
    uint cutoff = k;
    for (uint j = 0u; j < k; j++) {
        cum += gval[j];
        if (cum >= p.top_p) { cutoff = j + 1u; break; }
    }
    float total = 0.0f;
    for (uint j = 0u; j < cutoff; j++) { total += gval[j]; }
    float r = uniform * total;
    uint tok = gidx[cutoff - 1u];
    float acc = 0.0f;
    for (uint j = 0u; j < cutoff; j++) {
        acc += gval[j];
        if (r <= acc) { tok = gidx[j]; break; }
    }
    out_id[0] = tok;
}

inline void sample_f32_impl(device const float* logits,
                            float uniform,
                            device uint* out_id,
                            constant SampleParams& p,
                            uint t,
                            threadgroup float* sval,
                            threadgroup uint* sidx,
                            threadgroup float* gval,
                            threadgroup uint* gidx) {
    // Clamp like the host (`k = top_k.min(logits.len())`) — defensive against a vocab smaller
    // than top_k; never triggers in practice (vocab >> 64).
    uint k = min(p.top_k, p.n);
    k = min(k, SAMPLE_KMAX);
    for (uint iter = 0u; iter < k; iter++) {
        float best = -1e30f;
        uint bi = 0u;
        for (uint i = t; i < p.n; i += 256u) {
            bool used = false;
            for (uint j = 0u; j < iter; j++) {
                if (gidx[j] == i) { used = true; break; }
            }
            if (!used && logits[i] > best) { best = logits[i]; bi = i; }
        }
        sval[t] = best;
        sidx[t] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s /= 2u) {
            if (t < s && sval[t + s] > sval[t]) {
                sval[t] = sval[t + s];
                sidx[t] = sidx[t + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) {
            gval[iter] = sval[0];
            gidx[iter] = sidx[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Phase 2 (single lane): softmax(temp), nucleus cutoff, inverse-CDF sample.
    if (t == 0u) {
        sample_f32_finish(uniform, out_id, p, k, gval, gidx);
    }
}

kernel void sample_f32(device const float* logits [[buffer(0)]],
                       device const float* u_buf  [[buffer(1)]],
                       device uint*        out_id [[buffer(2)]],
                       constant SampleParams& p    [[buffer(3)]],
                       uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    threadgroup float gval[SAMPLE_KMAX];
    threadgroup uint  gidx[SAMPLE_KMAX];
    sample_f32_impl(logits, u_buf[0], out_id, p, t, sval, sidx, gval, gidx);
}

// Record-once decode variant: params are fixed in the tape, while the bound position and the
// runner's 64-slot uniform ring change per token.
kernel void sample_f32_dyn(device const float* logits    [[buffer(0)]],
                           device const float* u_buf     [[buffer(1)]],
                           device const int*   positions [[buffer(2)]],
                           device uint*        out_id    [[buffer(3)]],
                           constant SampleParams& p       [[buffer(4)]],
                           uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    threadgroup float gval[SAMPLE_KMAX];
    threadgroup uint  gidx[SAMPLE_KMAX];
    sample_f32_impl(
        logits, u_buf[(uint)positions[0] & 63u], out_id, p, t, sval, sidx, gval, gidx
    );
}

struct SampleSplitParams { uint n; uint top_k; uint chunk; };
kernel void sample_f32_stage1(device const float* logits [[buffer(0)]],
                              device float*       out_val [[buffer(1)]],
                              device uint*        out_idx [[buffer(2)]],
                              constant SampleSplitParams& p [[buffer(3)]],
                              uint t [[thread_position_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    uint base = group * p.chunk;
    uint end = min(base + p.chunk, p.n);
    uint k = min(p.top_k, SAMPLE_KMAX);
    // A 4K chunk gives each lane at most 16 strided logits. Once a lane wins a reduction,
    // remember that local slot directly instead of comparing every logit with all prior winners.
    uint used_mask = 0u;
    for (uint iter = 0u; iter < k; iter++) {
        float best = -INFINITY;
        uint bi = base;
        uint slot = 0u;
        for (uint i = base + t; i < end; i += 256u, slot++) {
            float v = logits[i];
            if ((used_mask & (1u << slot)) == 0u &&
                (v > best || (v == best && i < bi))) {
                best = v;
                bi = i;
            }
        }
        sval[t] = best;
        sidx[t] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s /= 2u) {
            if (t < s) {
                float v = sval[t + s];
                uint i = sidx[t + s];
                if (v > sval[t] || (v == sval[t] && i < sidx[t])) {
                    sval[t] = v;
                    sidx[t] = i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint winner = sidx[0];
        uint lane_base = base + t;
        if (winner >= lane_base && winner < end) {
            uint delta = winner - lane_base;
            if (delta % 256u == 0u) {
                uint slot = delta / 256u;
                used_mask |= 1u << slot;
            }
        }
        if (t == 0u) {
            uint out = group * k + iter;
            out_val[out] = sval[0];
            out_idx[out] = sidx[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void sample_f32_stage2_impl(device const float* values,
                                   device const uint* indices,
                                   float uniform,
                                   device uint* out_id,
                                   constant SampleParams& p,
                                   uint t,
                                   threadgroup float* sval,
                                   threadgroup uint* sidx,
                                   threadgroup float* gval,
                                   threadgroup uint* gidx,
                                   threadgroup uint* gslot) {
    uint k = min(min(p.top_k, p.n), SAMPLE_KMAX);
    for (uint iter = 0u; iter < k; iter++) {
        float best = -INFINITY;
        uint bi = 0u;
        for (uint i = t; i < p.n; i += 256u) {
            bool used = false;
            for (uint j = 0u; j < iter; j++) {
                if (gslot[j] == i) { used = true; break; }
            }
            float v = values[i];
            uint idx = indices[i];
            if (!used && (v > best || (v == best && idx < indices[bi]))) {
                best = v;
                bi = i;
            }
        }
        sval[t] = best;
        sidx[t] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 128u; s > 0u; s /= 2u) {
            if (t < s) {
                float v = sval[t + s];
                uint slot = sidx[t + s];
                if (v > sval[t] || (v == sval[t] && indices[slot] < indices[sidx[t]])) {
                    sval[t] = v;
                    sidx[t] = slot;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0u) {
            uint slot = sidx[0];
            gval[iter] = sval[0];
            gidx[iter] = indices[slot];
            gslot[iter] = slot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
        sample_f32_finish(uniform, out_id, p, k, gval, gidx);
    }
}

kernel void sample_f32_stage2(device const float* values [[buffer(0)]],
                              device const uint*  indices [[buffer(1)]],
                              device const float* u_buf   [[buffer(2)]],
                              device uint*        out_id  [[buffer(3)]],
                              constant SampleParams& p    [[buffer(4)]],
                              uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    threadgroup float gval[SAMPLE_KMAX];
    threadgroup uint  gidx[SAMPLE_KMAX];
    threadgroup uint  gslot[SAMPLE_KMAX];
    sample_f32_stage2_impl(
        values, indices, u_buf[0], out_id, p, t, sval, sidx, gval, gidx, gslot
    );
}

kernel void sample_f32_stage2_dyn(device const float* values    [[buffer(0)]],
                                  device const uint*  indices   [[buffer(1)]],
                                  device const float* u_buf     [[buffer(2)]],
                                  device const int*   positions [[buffer(3)]],
                                  device uint*        out_id    [[buffer(4)]],
                                  constant SampleParams& p      [[buffer(5)]],
                                  uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sval[256];
    threadgroup uint  sidx[256];
    threadgroup float gval[SAMPLE_KMAX];
    threadgroup uint  gidx[SAMPLE_KMAX];
    threadgroup uint  gslot[SAMPLE_KMAX];
    sample_f32_stage2_impl(
        values, indices, u_buf[(uint)positions[0] & 63u], out_id, p,
        t, sval, sidx, gval, gidx, gslot
    );
}
