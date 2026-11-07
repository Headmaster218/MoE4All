// ---- Gated DeltaNet (qwen35/Qwen3.5 linear attention) + its depthwise conv, ON DEVICE. Both are
// sequential-over-rows recurrences; the parallelism is across CHANNELS (conv: each thread owns a
// channel and its state column, no cross-thread deps) and across (value-dim, head) (deltanet:
// one SIMDGROUP per state column S[:, d], the column register-resident across the chunk — see
// the kernel comment).
// State lives in the BOUND buffer and is updated in place (no host round-trip per layer).
struct ConvSiluParams { uint rows; uint channels; uint kwidth; };
kernel void conv1d_silu_f32(device const float* x     [[buffer(0)]],
                            device const float* w     [[buffer(1)]],
                            device float*       state [[buffer(2)]],
                            device float*       dst   [[buffer(3)]],
                            constant ConvSiluParams& p [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    uint ch = gid;
    if (ch >= p.channels) return;
    uint kk = p.kwidth;             // host gates kwidth <= 8
    float st[7];
    float wv[8];
    for (uint j = 0; j + 1 < kk; j++) st[j] = state[j * p.channels + ch];
    for (uint j = 0; j < kk; j++) wv[j] = w[ch * kk + j];
    for (uint t = 0; t < p.rows; t++) {
        float xt = x[t * p.channels + ch];
        float acc = xt * wv[kk - 1u];
        for (uint j = 0; j + 1 < kk; j++) acc += st[j] * wv[j];
        dst[t * p.channels + ch] = acc / (1.0f + exp(-acc));
        for (uint j = 0; j + 2 < kk; j++) st[j] = st[j + 1];
        if (kk >= 2u) st[kk - 2u] = xt;
    }
    for (uint j = 0; j + 1 < kk; j++) state[j * p.channels + ch] = st[j];
}

// Multi-row pass 1: each output reads its causal window from the virtual sequence
// [old state | x], so all rows*channels elements are independent.
kernel void conv1d_silu_par_f32(device const float* x     [[buffer(0)]],
                                device const float* w     [[buffer(1)]],
                                device const float* state [[buffer(2)]],
                                device float*       dst   [[buffer(3)]],
                                constant ConvSiluParams& p [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.channels) return;
    uint t = gid / p.channels;
    uint ch = gid % p.channels;
    uint km1 = p.kwidth - 1u;
    float acc = 0.0f;
    if (t >= km1) {
        ulong base = (ulong)(t - km1) * p.channels + ch;
        for (uint k = 0; k < p.kwidth; k++) {
            acc += x[base + (ulong)k * p.channels] * w[ch * p.kwidth + k];
        }
    } else {
        for (uint k = 0; k < p.kwidth; k++) {
            uint i = t + k;
            float xv;
            if (i < km1) {
                xv = state[i * p.channels + ch];
            } else {
                xv = x[(i - km1) * p.channels + ch];
            }
            acc += xv * w[ch * p.kwidth + k];
        }
    }
    dst[gid] = acc / (1.0f + exp(-acc));
}

// Multi-row pass 2: rows >= K-1, so the final K-1 inputs directly become state.
kernel void conv1d_shift_f32(device const float* x [[buffer(0)]],
                             device float* state   [[buffer(1)]],
                             constant ConvSiluParams& p [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    uint km1 = p.kwidth - 1u;
    if (gid >= km1 * p.channels) return;
    uint k = gid / p.channels;
    uint ch = gid % p.channels;
    state[gid] = x[(p.rows - km1 + k) * p.channels + ch];
}

// One SIMDGROUP per (value-dim d, value-head h) — the llama.cpp kernel_gated_delta_net
// parallelization. The simdgroup's 32 lanes split the k-dim (KPL = kd/32 state entries per
// lane, register-resident ls[] for the WHOLE chunk — the old shape re-read the state column
// from device twice per token), so the token recurrence needs only simd_sums: no threadgroup
// memory, no barriers. DN_VPT simdgroups share a threadgroup purely for occupancy (grid =
// nv * vd/DN_VPT threadgroups of DN_VPT*32 threads — vd/DN_VPT times the old
// one-threadgroup-per-head grid, which left the GPU idle at qwen35's 32 heads). State
// reads/writes touch device memory once per chunk instead of 2*rows times.
struct DeltaNetParams { uint rows; uint nv; uint nk; uint kd; uint vd; float eps; };
struct DeltaNetGateParams { uint rows; uint nv; };

// Multi-row gate prep: compute the token/head scalars once instead of repeating four
// transcendentals in every state-column simdgroup and every lane of that simdgroup.
kernel void deltanet_gates_f32(device const float* b       [[buffer(0)]],
                               device const float* a       [[buffer(1)]],
                               device const float* a_coef  [[buffer(2)]],
                               device const float* dt_bias [[buffer(3)]],
                               device float2*      gates   [[buffer(4)]],
                               constant DeltaNetGateParams& p [[buffer(5)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.nv) return;
    uint h = gid % p.nv;
    float beta = 1.0f / (1.0f + exp(-b[gid]));
    float z = a[gid] + dt_bias[h];
    float sp = max(z, 0.0f) + log(1.0f + exp(-fabs(z)));
    gates[gid] = float2(beta, exp(a_coef[h] * sp));
}

#define DN_VPT 4u
struct DeltaNetNormParams { uint rows; uint nk; uint kd; float eps; };

// Multi-row norm prep: one simdgroup per token/key-head computes q/k L2 norms once,
// instead of repeating both reductions for every value-head state column.
template<uint KPL>
kernel void deltanet_norm_f32_t(device const float* q      [[buffer(0)]],
                                device const float* k      [[buffer(1)]],
                                device float*       q_norm [[buffer(2)]],
                                device float*       k_norm [[buffer(3)]],
                                constant DeltaNetNormParams& p [[buffer(4)]],
                                uint tgpig [[threadgroup_position_in_grid]],
                                uint lane  [[thread_index_in_simdgroup]],
                                uint sgid  [[simdgroup_index_in_threadgroup]]) {
    uint i = tgpig * DN_VPT + sgid;
    if (i >= p.rows * p.nk) return;
    ulong base = (ulong)i * p.kd + lane * KPL;
    float qv[KPL], kv[KPL];
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (uint j = 0; j < KPL; j++) {
        qv[j] = q[base + j];
        kv[j] = k[base + j];
        sq += qv[j] * qv[j];
        sk += kv[j] * kv[j];
    }
    float qn = sqrt(simd_sum(sq) + p.eps);
    float kn = sqrt(simd_sum(sk) + p.eps);
    float qscale = rsqrt((float)p.kd);
#pragma unroll
    for (uint j = 0; j < KPL; j++) {
        q_norm[base + j] = qv[j] / qn * qscale;
        k_norm[base + j] = kv[j] / kn;
    }
}

typedef decltype(deltanet_norm_f32_t<4>) deltanet_norm_f32_k;
template [[host_name("deltanet_norm_k1")]] kernel deltanet_norm_f32_k deltanet_norm_f32_t<1>;
template [[host_name("deltanet_norm_k2")]] kernel deltanet_norm_f32_k deltanet_norm_f32_t<2>;
template [[host_name("deltanet_norm_k4")]] kernel deltanet_norm_f32_k deltanet_norm_f32_t<4>;
template [[host_name("deltanet_norm_k8")]] kernel deltanet_norm_f32_k deltanet_norm_f32_t<8>;

// KPL (k entries per lane, = kd/32) is a COMPILE-TIME template parameter: with a runtime
// bound the ls[]/qv[]/kv[] arrays are runtime-indexed, the compiler cannot promote them to
// registers, and the "register-resident" state silently spills to thread-private memory —
// measured 1.7x SLOWER than the old threadgroup-staged shape (507 ms vs 302 per qwen35
// prefill). Unrolled at fixed KPL the arrays genuinely live in registers: 185 ms, 1.6x
// faster than the old shape.
template<uint KPL, bool PREPARED_GATES, bool PREPARED_NORM>
kernel void deltanet_f32_t(device const float* q       [[buffer(0)]],
                         device const float* k       [[buffer(1)]],
                         device const float* v       [[buffer(2)]],
                         device const float* b       [[buffer(3)]],
                         device const float* a       [[buffer(4)]],
                         device const float* a_coef  [[buffer(5)]],
                         device const float* dt_bias [[buffer(6)]],
                         device const float2* gates  [[buffer(7)]],
                         device float*       state   [[buffer(8)]],
                         device float*       dst     [[buffer(9)]],
                         constant DeltaNetParams& p  [[buffer(10)]],
                         uint   tgpig [[threadgroup_position_in_grid]],
                         uint   lane  [[thread_index_in_simdgroup]],
                         uint   sgid  [[simdgroup_index_in_threadgroup]]) {
    uint dpg = p.vd / DN_VPT;                 // d-groups per head
    uint h = tgpig / dpg;
    uint d = (tgpig % dpg) * DN_VPT + sgid;   // this simdgroup's value dim
    uint kh_idx = h % p.nk;
    float qscale = rsqrt((float)p.kd);

    // Lane `lane` owns k indices lane*kpl .. lane*kpl+kpl-1 of state column S[:, d],
    // register-resident across the whole chunk.
    device float* S = state + (ulong)h * p.kd * p.vd;
    float ls[KPL];
#pragma unroll
    for (uint j = 0; j < KPL; j++) ls[j] = S[(ulong)(lane * KPL + j) * p.vd + d];

    for (uint t = 0; t < p.rows; t++) {
        // This row's q/k head slice: each lane loads its kpl entries; L2 norms via simd_sum
        // (q also x 1/sqrt(kd)) — same formulas as the CPU reference.
        ulong qb = (ulong)t * p.nk * p.kd + (ulong)kh_idx * p.kd + lane * KPL;
        float qv[KPL], kv[KPL];
        float sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (uint j = 0; j < KPL; j++) {
            qv[j] = q[qb + j];
            kv[j] = k[qb + j];
            if (!PREPARED_NORM) {
                sq += qv[j] * qv[j];
                sk += kv[j] * kv[j];
            }
        }
        if (!PREPARED_NORM) {
            float qn = sqrt(simd_sum(sq) + p.eps);
            float kn = sqrt(simd_sum(sk) + p.eps);
#pragma unroll
            for (uint j = 0; j < KPL; j++) {
                qv[j] = qv[j] / qn * qscale;
                kv[j] = kv[j] / kn;
            }
        }
        float beta, decay;
        if (PREPARED_GATES) {
            float2 gd = gates[t * p.nv + h];
            beta = gd.x;
            decay = gd.y;
        } else {
            // Decode keeps one dispatch: these inputs are uniform across the simdgroup.
            float bv = b[t * p.nv + h];
            beta = 1.0f / (1.0f + exp(-bv));
            float z = a[t * p.nv + h] + dt_bias[h];
            float sp = max(z, 0.0f) + log(1.0f + exp(-fabs(z)));
            decay = exp(a_coef[h] * sp);
        }

        // Delta rule on the decayed state, fused with the output accumulation.
        float s_k = 0.0f;
#pragma unroll
        for (uint j = 0; j < KPL; j++) {
            ls[j] *= decay;
            s_k += ls[j] * kv[j];
        }
        s_k = simd_sum(s_k);
        float delta = (v[(ulong)t * p.nv * p.vd + (ulong)h * p.vd + d] - s_k) * beta;
        float y = 0.0f;
#pragma unroll
        for (uint j = 0; j < KPL; j++) {
            ls[j] += kv[j] * delta;
            y += ls[j] * qv[j];
        }
        y = simd_sum(y);
        if (lane == 0u) {
            dst[(ulong)t * p.nv * p.vd + (ulong)h * p.vd + d] = y;
        }
    }

#pragma unroll
    for (uint j = 0; j < KPL; j++) S[(ulong)(lane * KPL + j) * p.vd + d] = ls[j];
}

typedef decltype(deltanet_f32_t<4, false, false>) deltanet_f32_k;
template [[host_name("deltanet_f32_k1")]] kernel deltanet_f32_k deltanet_f32_t<1, false, false>;
template [[host_name("deltanet_f32_k2")]] kernel deltanet_f32_k deltanet_f32_t<2, false, false>;
template [[host_name("deltanet_f32_k4")]] kernel deltanet_f32_k deltanet_f32_t<4, false, false>;
template [[host_name("deltanet_f32_k8")]] kernel deltanet_f32_k deltanet_f32_t<8, false, false>;
template [[host_name("deltanet_gates_k1")]] kernel deltanet_f32_k deltanet_f32_t<1, true, false>;
template [[host_name("deltanet_gates_k2")]] kernel deltanet_f32_k deltanet_f32_t<2, true, false>;
template [[host_name("deltanet_gates_k4")]] kernel deltanet_f32_k deltanet_f32_t<4, true, false>;
template [[host_name("deltanet_gates_k8")]] kernel deltanet_f32_k deltanet_f32_t<8, true, false>;
template [[host_name("deltanet_prepared_k1")]] kernel deltanet_f32_k deltanet_f32_t<1, true, true>;
template [[host_name("deltanet_prepared_k2")]] kernel deltanet_f32_k deltanet_f32_t<2, true, true>;
template [[host_name("deltanet_prepared_k4")]] kernel deltanet_f32_k deltanet_f32_t<4, true, true>;
template [[host_name("deltanet_prepared_k8")]] kernel deltanet_f32_k deltanet_f32_t<8, true, true>;
