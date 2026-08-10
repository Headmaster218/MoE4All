// ---- MoE expert GEMVs: the shared GEMV bodies, batched over the SELECTED experts — one
// dispatch covers all n_used experts (slot = high grid bits), each picking its weight slice
// from the device expert table (`moe_topk` below), so a whole MoE FFN is 7 dispatches with no
// host round-trip. Gate/up read the shared x and write per-slot rows of the [n_used, out_f]
// scratch; the down variant (EPI 2) reads its slot's activation row and writes w[slot]*y to its
// slot's output row — `moe_reduce` then folds the weighted expert sum (slot-ascending, the CPU
// reference's accumulation order).
struct MoeLinParams { uint m; uint in_f; uint out_f; uint dshift; uint n_used; uint row0; };
#define MOE_WRAP(NAME, BODY, EPI, ROWB)                                                           \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 device const uint*   tbl   [[buffer(5)]],                                       \
                 constant MoeLinParams& p   [[buffer(6)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    uint sg = gid / 32u;                                                                          \
    uint per_out = p.out_f >> 1;         /* simdgroups per expert (2 weight rows each) */        \
    uint row = sg / (p.n_used * per_out);          /* token row within this chunk */             \
    uint rem = sg % (p.n_used * per_out);                                                         \
    uint slot = rem / per_out;                                                                    \
    uint e = tbl[(p.row0 + row) * 32u + slot];                                                    \
    float w = as_type<float>(tbl[(p.row0 + row) * 32u + 16u + slot]);                             \
    ulong row_b = (ulong)(p.in_f >> 8) * ROWB;                                                    \
    device const uchar* ec = codes + (ulong)e * p.out_f * row_b;                                  \
    /* gate/up read the token's row of x [rows, ne]; down reads its (row, slot) activation */     \
    device const float* xs = (EPI == 2) ? x + (ulong)(row * p.n_used + slot) * p.in_f             \
                                        : x + (ulong)(p.row0 + row) * p.in_f;                     \
    device float* ds = dst + (ulong)(row * p.n_used + slot) * p.out_f;                            \
    uint g2 = (rem % per_out) * 32u + lane;  /* body sees a per-(row, expert) grid */             \
    threadgroup float red[2];                                                                     \
    BODY<EPI, 1>(xs, ec, ds, xs, p, w, true, g2, lane, 0, red);                                   \
}
MOE_WRAP(linear_q4k_moe,     linear_q4k_body, 0, 144ul)
MOE_WRAP(linear_q4k_moe_acc, linear_q4k_body, 2, 144ul)
MOE_WRAP(linear_q6k_moe,     linear_q6k_body, 0, 210ul)
MOE_WRAP(linear_q6k_moe_acc, linear_q6k_body, 2, 210ul)
#undef MOE_WRAP

// ---- Expert-grouped GEMM for batched MoE prefill (the llama.cpp mul_mm_id shape). Rather than
// one GEMV per (token, slot) — no MMA, compute-bound at prefill widths — `moe_map` groups the
// chunk's (token, slot) pairs by EXPERT (one thread per expert scans the routing table, so each
// expert's list is token-ascending and deterministic), and the `*_cmm_id` kernels run the shared
// cooperative-GEMM tile over each expert's token group: grid = expert x token-tile x out-tile,
// tiles past an expert's count return before any barrier, activations stage through the same
// pre-transposed 8x8 layout with the token row resolved through the id list, and the output
// SCATTERS through threadgroup staging to each token's (row, slot) scratch row. The down variant
// folds the routing weight during the scatter. Ids are chunk-relative (`row0` rebases).
struct MoeMapParams { uint n_expert; uint n_used; uint rows; uint row0; uint cap; };
kernel void moe_map(device const uint* tbl [[buffer(0)]],
                    device uint*       ids [[buffer(1)]],
                    device uint*       tpe [[buffer(2)]],
                    constant MoeMapParams& p [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.n_expert) return;
    uint n = 0;
    for (uint r = 0; r < p.rows; r++) {
        device const uint* rt = tbl + (ulong)(p.row0 + r) * 32u;
        for (uint u = 0; u < p.n_used; u++) {
            if (rt[u] == tid) {
                ids[tid * p.cap + n] = r * p.n_used + u;
                n++;
            }
        }
    }
    tpe[tid] = n;
}

struct MoeCmmParams { uint in_f; uint out_f; uint n_used; uint cap; uint row0; uint ntt; };
// DIVROW: gate/up read x[row0 + t/n_used]; down (DIVROW=0) reads its (row, slot) activation row.
// WSCALE: fold the routing weight tbl[(row0 + t/n_used)*32 + 16 + t%n_used] during the scatter.
#define MOE_CMM_KERNEL(NAME, DEC, DIVROW, WSCALE)                                                 \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 device const uint*   ids   [[buffer(5)]],                                       \
                 device const uint*   tpe   [[buffer(6)]],                                       \
                 device const uint*   tbl   [[buffer(7)]],                                       \
                 constant MoeCmmParams& p   [[buffer(8)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint tgix = gid / 128u;                                                                       \
    uint nto = p.out_f / 64u;                                                                     \
    uint e = tgix / (p.ntt * nto);                                                                \
    uint rem = tgix % (p.ntt * nto);                                                              \
    uint tt0 = (rem / nto) * 32u;                                                                 \
    uint ro = (rem % nto) * 64u;                                                                  \
    uint cnt = tpe[e];                                                                            \
    if (tt0 >= cnt) return;   /* uniform per threadgroup, before any barrier */                   \
    uint nr1 = min(32u, cnt - tt0);                                                               \
    uint tid = sgid * 32u + lane;                                                                 \
                                                                                                  \
    threadgroup float shraw[2048];                                                                \
    threadgroup half* sa = (threadgroup half*)shraw;                                              \
    threadgroup half* sb = ((threadgroup half*)shraw) + 2048u;                                    \
                                                                                                  \
    uint lr0 = tid >> 1;                                                                          \
    uint il0 = tid & 1u;                                                                          \
    uint lr1 = tid >> 2;                                                                          \
    uint iyk = (tid & 3u) * 8u;                                                                   \
    uint lr1c = min(lr1, nr1 - 1u);                                                               \
    uint tident = ids[e * p.cap + tt0 + lr1c];                                                    \
    uint xrow = DIVROW ? (p.row0 + tident / p.n_used) : tident;                                   \
    /* Preserve the full-band sequence; partial final tiles skip dead 8-row matrix fragments. */ \
    uint row_base = 16u * (sgid >> 1);                                                            \
                                                                                                  \
    simdgroup_half8x8 ma[4];                                                                      \
    simdgroup_half8x8 mb[2];                                                                      \
    simdgroup_float8x8 mc[8];                                                                     \
    for (uint i = 0; i < 8u; i++) mc[i] = simdgroup_float8x8(0.0f);                               \
                                                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    ulong ebase = (ulong)e * p.out_f * nb;   /* expert's first block index */                     \
    for (uint k0 = 0; k0 < p.in_f; k0 += 32u) {                                                   \
        ulong bi = ebase + (ulong)(ro + lr0) * nb + (ulong)(k0 >> 4) + il0;                       \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        device const float4* yy =                                                                 \
            (device const float4*)(x + (ulong)xrow * p.in_f + k0 + iyk);                          \
        float4 yv0 = yy[0];                                                                       \
        float4 yv1 = yy[1];                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        {                                                                                         \
            uint sy = lr0 >> 3;                                                                   \
            uint lx = lr0 & 7u;                                                                   \
            for (uint i = 0; i < 16u; i++) {                                                      \
                uint sx = 2u * il0 + (i >> 3);                                                    \
                sa[64u * (8u * sx + sy) + 8u * (i & 7u) + lx] = (half)wk[i];                      \
            }                                                                                     \
        }                                                                                         \
        {                                                                                         \
            uint ib = 4u * (tid & 3u) + (lr1 >> 3);                                               \
            uint ly = lr1 & 7u;                                                                   \
            threadgroup half4* sb4 = (threadgroup half4*)(sb + 64u * ib + 8u * ly);               \
            sb4[0] = half4(yv0);                                                                  \
            sb4[1] = half4(yv1);                                                                  \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        threadgroup const half* lsma = sa + 4u * 64u * (sgid & 1u);                               \
        threadgroup const half* lsmb = sb + 2u * 64u * (sgid >> 1);                               \
        for (uint ik = 0; ik < 4u; ik++) {                                                        \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            if (row_base + 8u < nr1) {                                                            \
                for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], lsma + 64u * i, 8);           \
                simdgroup_barrier(mem_flags::mem_none);                                           \
                for (uint i = 0; i < 2u; i++) simdgroup_load(mb[i], lsmb + 64u * i, 8);           \
                simdgroup_barrier(mem_flags::mem_none);                                           \
                for (uint i = 0; i < 8u; i++)                                                     \
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);           \
            } else if (row_base < nr1) {                                                          \
                for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], lsma + 64u * i, 8);           \
                simdgroup_barrier(mem_flags::mem_none);                                           \
                simdgroup_load(mb[0], lsmb, 8);                                                   \
                simdgroup_barrier(mem_flags::mem_none);                                           \
                for (uint i = 0; i < 4u; i++)                                                     \
                    simdgroup_multiply_accumulate(mc[i], mb[0], ma[i], mc[i]);                    \
            }                                                                                     \
            lsma += 8u * 64u;                                                                     \
            lsmb += 4u * 64u;                                                                     \
        }                                                                                         \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
                                                                                                  \
    /* scatter through threadgroup staging: token rows are non-contiguous scratch rows */         \
    threadgroup float* tc = shraw + 32u * (sgid & 1u) + (16u * (sgid >> 1)) * 64u;                \
    for (uint i = 0; i < 8u; i++)                                                                 \
        simdgroup_store(mc[i], tc + 8u * (i & 3u) + 8u * 64u * (i >> 2), 64u);                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
    if (sgid == 0u) {                                                                             \
        for (uint j = lane; j < nr1; j += 32u) {                                                  \
            uint t = ids[e * p.cap + tt0 + j];                                                    \
            float w = WSCALE                                                                      \
                ? as_type<float>(tbl[(ulong)(p.row0 + t / p.n_used) * 32u + 16u + t % p.n_used])  \
                : 1.0f;                                                                           \
            device float* d2 = dst + (ulong)t * p.out_f + ro;                                     \
            threadgroup const float* c2 = shraw + j * 64u;                                        \
            for (uint i = 0; i < 64u; i++) d2[i] = c2[i] * w;                                     \
        }                                                                                         \
    }                                                                                             \
}
MOE_CMM_KERNEL(linear_q4k_cmm_id,   DEC16_Q4K, 1, 0)
MOE_CMM_KERNEL(linear_q4k_cmm_id_w, DEC16_Q4K, 0, 1)
MOE_CMM_KERNEL(linear_q6k_cmm_id,   DEC16_Q6K, 1, 0)
MOE_CMM_KERNEL(linear_q6k_cmm_id_w, DEC16_Q6K, 0, 1)
#undef MOE_CMM_KERNEL

// Weighted expert sum: dst[i] = sum_u y[u*ne + i] (weights already folded into y by the down
// GEMV's EPI-2 epilogue; slot-ascending order matches the CPU reference).
struct MoeReduceParams { uint ne; uint n_used; uint rows; uint row0; };
kernel void moe_reduce(device const float* y   [[buffer(0)]],
                       device float*       dst [[buffer(1)]],
                       constant MoeReduceParams& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.ne) return;
    uint row = gid / p.ne;
    uint i = gid % p.ne;
    float s = 0.0f;
    for (uint u = 0; u < p.n_used; u++) s += y[(row * p.n_used + u) * p.ne + i];
    dst[(p.row0 + row) * p.ne + i] = s;
}

// `hash != 0` = DeepSeek V4 hash-routed layer: the selection arrives pre-gathered in `hash_ids`
// (`ffn_gate_tid2eid` indexed by token id, llama.cpp's `selected_experts_in`), so the whole top-k
// reduction is SKIPPED rather than run and overwritten — it exists only to rank experts. The
// softmax shift/denominator and the weight formula are unchanged: `build_moe_ffn` reads the
// weights out of `ggml_get_rows(probs, selected_experts)` on both routing paths.
struct MoeTopkParams { uint n_expert; uint n_used; float scale; uint hash; };
kernel void moe_topk(device const float* logits_all [[buffer(0)]],
                     device const uint*  hash_ids   [[buffer(1)]],
                     device uint*        tbl_all    [[buffer(2)]],
                     constant MoeTopkParams& p  [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    // one simdgroup (32 threads) per token row; each lane owns experts e = lane + 32j
    uint row = gid / 32u;
    uint lane = gid % 32u;
    device const float* logits = logits_all + (ulong)row * p.n_expert;
    device uint* tbl = tbl_all + row * 32u;
    float lmax = -MAXFLOAT;
    for (uint e = lane; e < p.n_expert; e += 32u) lmax = max(lmax, logits[e]);
    float maxl = simd_max(lmax);
    // psum in the reference's ascending order (exact bit-match), broadcast from lane 0
    float psum = 0.0f;
    if (lane == 0u) {
        for (uint e = 0; e < p.n_expert; e++) psum += exp(logits[e] - maxl);
    }
    psum = simd_broadcast_first(psum);
    // top-k selection, lane-parallel: each round every lane offers its best untaken expert
    // (ascending scan + strict > == lowest index per lane), simd_max picks the winning logit
    // and simd_min the lowest tied index — exactly the reference's stable-sort order
    uint taken = 0u;   // bitmask over this lane's stride slots j
    uint sel[16];
    if (p.hash != 0u) {
        // Hash-routed: the table row IS the selection. Clamped for the same reason the top-k
        // winner is — an out-of-range id would send the expert GEMV past its weight bank.
        for (uint u = 0; u < p.n_used; u++) {
            sel[u] = min(hash_ids[row * p.n_used + u], p.n_expert - 1u);
        }
    } else {
        for (uint u = 0; u < p.n_used; u++) {
            float bv = -MAXFLOAT;
            uint be = 0xFFFFFFFFu;
            uint j = 0u;
            for (uint e = lane; e < p.n_expert; e += 32u, j++) {
                if ((taken & (1u << j)) == 0u && logits[e] > bv) { bv = logits[e]; be = e; }
            }
            float m = simd_max(bv);
            uint pick = simd_min(bv == m ? be : 0xFFFFFFFFu);
            sel[u] = pick;
            if ((pick & 31u) == lane) taken |= 1u << (pick >> 5);
        }
    }
    if (lane == 0u) {
        float wsum = 0.0f;
        float ws[16];
        for (uint u = 0; u < p.n_used; u++) {
            ws[u] = exp(logits[sel[u]] - maxl) / psum;
            wsum += ws[u];
        }
        wsum = max(wsum, 1e-20f);
        for (uint u = 0; u < p.n_used; u++) {
            tbl[u] = sel[u];
            tbl[16u + u] = as_type<uint>(ws[u] / wsum * p.scale);
        }
    }
}

GEMV_KERNEL(linear_quik4, DEC16_K4)
GEMV_KERNEL(linear_quik6, DEC16_K6)
GEMV_KERNEL(linear_quik8, DEC16_K8)
GEMV_KERNEL(linear_mxfp4, DEC16_MXFP4)
GEMV_KERNEL(linear_nvfp4, DEC16_NVFP4)
GEMV_KERNEL(linear_q2_0, DEC16_Q2_0)
GEMV_KERNEL(linear_tq2_0, DEC16_TQ2_0)
GEMV_KERNEL(linear_tq1_0, DEC16_TQ1_0)
GEMV_KERNEL(linear_iq4xs, DEC16_IQ4XS)
GEMV_KERNEL(linear_iq2xxs, DEC16_IQ2XXS)
GEMV_KERNEL(linear_iq3xxs, DEC16_IQ3XXS)
GEMV_KERNEL(linear_iq3s, DEC16_IQ3S)
GEMV_KERNEL(linear_iq2s, DEC16_IQ2S)
GEMV_KERNEL(linear_iq2xs, DEC16_IQ2XS)
GEMV_KERNEL(linear_iq1s, DEC16_IQ1S)
GEMV_KERNEL(linear_iq1m, DEC16_IQ1M)
GEMV_KERNEL(linear_q5k, DEC16_Q5K)
GEMV_KERNEL(linear_q2k, DEC16_Q2K)
GEMV_KERNEL(linear_q3k, DEC16_Q3K)
RT_KERNEL(linear_quik4_rt, DEC16_K4)
RT_KERNEL(linear_quik6_rt, DEC16_K6)
RT_KERNEL(linear_quik8_rt, DEC16_K8)
RT_KERNEL(linear_q4k_rt, DEC16_Q4K)
RT_KERNEL(linear_q5k_rt, DEC16_Q5K)
RT_KERNEL(linear_q6k_rt, DEC16_Q6K)
RT_KERNEL(linear_q2k_rt, DEC16_Q2K)
RT_KERNEL(linear_q3k_rt, DEC16_Q3K)
RT_KERNEL(linear_q8_0_rt, DEC16_Q8_0)
RT_KERNEL(linear_q5_0_rt, DEC16_Q5_0)
RT_KERNEL(linear_q4_0_rt, DEC16_Q4_0)
RT_KERNEL(linear_q5_1_rt, DEC16_Q5_1)
RT_KERNEL(linear_q4_1_rt, DEC16_Q4_1)
RT_KERNEL(linear_mxfp4_rt, DEC16_MXFP4)
RT_KERNEL(linear_nvfp4_rt, DEC16_NVFP4)
RT_KERNEL(linear_q2_0_rt, DEC16_Q2_0)
RT_KERNEL(linear_tq2_0_rt, DEC16_TQ2_0)
RT_KERNEL(linear_tq1_0_rt, DEC16_TQ1_0)
RT_KERNEL(linear_iq4xs_rt, DEC16_IQ4XS)
RT_KERNEL(linear_iq4nl_rt, DEC16_IQ4NL)
RT_KERNEL(linear_iq2xxs_rt, DEC16_IQ2XXS)
RT_KERNEL(linear_iq3xxs_rt, DEC16_IQ3XXS)
RT_KERNEL(linear_iq3s_rt, DEC16_IQ3S)
RT_KERNEL(linear_iq2s_rt, DEC16_IQ2S)
RT_KERNEL(linear_iq2xs_rt, DEC16_IQ2XS)
RT_KERNEL(linear_iq1s_rt, DEC16_IQ1S)
RT_KERNEL(linear_iq1m_rt, DEC16_IQ1M)
RT_KERNEL(linear_f16_rt, DEC16_F16)
RT_KERNEL(linear_bf16_rt, DEC16_BF16)
RT_KERNEL(linear_f32_rt, DEC16_F32)
// Cooperative-tile half-fragment GEMM, mul_mm-shape: one 64-output x 32-token tile per 128-thread
// threadgroup, NK=32 K-steps. What the simpler cooperative tile above lacked (each of its shapes
// measured and replaced): weights AND activations are staged into threadgroup memory as
// CONTIGUOUS 8x8 half tiles (stride-8, conflict-free simdgroup_load; the weight tile is written
// pre-transposed so no transposed loads), each thread dequantizes exactly one 16-element block
// per K-step (128 threads = the whole 64x32 weight tile), and x is cast f32->f16 inline during
// staging (no separate cast pass). Each simdgroup owns a 32-output x 16-token quadrant as 8 f32
// accumulators; a partial token tile stages through threadgroup memory and row-guards the copy,
// so every tile takes the MMA path. Requires out_f % 64 == 0 and in_f % 32 == 0; other shapes
// fall back to the per-simdgroup HGEMM.
#define CMM_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define CMM_KERNEL_TYPED(NAME, DEC, WTYPE, WMAT, XTYPE, XVEC, XMAT, SHWORDS, SBOFF)             \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint tgix = gid / 128u;                                                                       \
    uint nto = p.out_f / 64u;                                                                     \
    uint ntm = (p.m + 31u) / 32u;                                                                 \
    if (tgix >= ntm * nto) return;                                                                \
    uint to = tgix % nto;                                                                         \
    uint tm = tgix / nto;                                                                         \
    uint ro = to * 64u;                                                                           \
    uint rt = tm * 32u;                                                                           \
    uint tid = sgid * 32u + lane;                                                                 \
    uint nr1 = min(32u, p.m - rt);                                                                \
                                                                                                  \
    threadgroup float shraw[SHWORDS];                                                             \
    threadgroup WTYPE* sa = (threadgroup WTYPE*)shraw;                                            \
    threadgroup XTYPE* sb = (threadgroup XTYPE*)(shraw + SBOFF);                                  \
                                                                                                  \
    uint lr0 = tid >> 1;                 /* weight (output) row 0..63 */                          \
    uint il0 = tid & 1u;                 /* which 16-element half of the 32-K step */             \
    uint lr1 = tid >> 2;                 /* token row 0..31 */                                    \
    uint iyk = (tid & 3u) * 8u;          /* k offset within the 32-K step */                      \
    uint lr1c = min(lr1, nr1 - 1u);      /* clamp token loads to the matrix edge */               \
                                                                                                  \
    WMAT ma[4];                                                                                   \
    XMAT mb[2];                                                                                   \
    simdgroup_float8x8 mc[8];                                                                     \
    CMM_UNROLL(uint i = 0; i < 8u; i++) mc[i] = simdgroup_float8x8(0.0f);                         \
                                                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    for (uint k0 = 0; k0 < p.in_f; k0 += 32u) {                                                   \
        /* device reads + dequant FIRST, into registers — issued while the previous       */     \
        /* iteration's MMA phase (other simdgroups) is still draining; the barrier below  */     \
        /* orders only the threadgroup-memory stores (mul_mm does exactly this)           */     \
        ulong bi = (ulong)(ro + lr0) * nb + (ulong)(k0 >> 4) + il0;                               \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        device const float4* yy =                                                                 \
            (device const float4*)(x + (ulong)(rt + lr1c) * p.in_f + k0 + iyk);                   \
        float4 yv0 = yy[0];                                                                       \
        float4 yv1 = yy[1];                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        {   /* stage A: one 16-block per thread, into pre-transposed 8x8 tiles */                 \
            uint sy = lr0 >> 3;                                                                   \
            uint lx = lr0 & 7u;                                                                   \
            CMM_UNROLL(uint i = 0; i < 16u; i++) {                                                \
                uint sx = 2u * il0 + (i >> 3);                                                    \
                sa[64u * (8u * sx + sy) + 8u * (i & 7u) + lx] = (WTYPE)wk[i];                     \
            }                                                                                     \
        }                                                                                         \
        {   /* stage B: 8 activations per thread, two vectorized stores */                       \
            uint ib = 4u * (tid & 3u) + (lr1 >> 3);                                               \
            uint ly = lr1 & 7u;                                                                   \
            threadgroup XVEC* sb4 = (threadgroup XVEC*)(sb + 64u * ib + 8u * ly);                 \
            sb4[0] = XVEC(yv0);                                                                   \
            sb4[1] = XVEC(yv1);                                                                   \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        threadgroup const WTYPE* lsma = sa + 4u * 64u * (sgid & 1u);                              \
        threadgroup const XTYPE* lsmb = sb + 2u * 64u * (sgid >> 1);                              \
        CMM_UNROLL(uint ik = 0; ik < 4u; ik++) {                                                  \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            CMM_UNROLL(uint i = 0; i < 4u; i++) simdgroup_load(ma[i], lsma + 64u * i, 8);         \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            CMM_UNROLL(uint i = 0; i < 2u; i++) simdgroup_load(mb[i], lsmb + 64u * i, 8);         \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            CMM_UNROLL(uint i = 0; i < 8u; i++)                                                   \
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);              \
            lsma += 8u * 64u;                                                                     \
            lsmb += 4u * 64u;                                                                     \
        }                                                                                         \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
                                                                                                  \
    if (rt + 32u <= p.m) {                                                                        \
        device float* C = dst + (ro + 32u * (sgid & 1u)) +                                        \
                          (ulong)(rt + 16u * (sgid >> 1)) * p.out_f;                              \
        CMM_UNROLL(uint i = 0; i < 8u; i++)                                                       \
            simdgroup_store(mc[i], C + 8u * (i & 3u) + 8u * p.out_f * (i >> 2), p.out_f);         \
    } else {                                                                                      \
        /* partial token tile: stage through threadgroup memory, row-guard the copy-out */        \
        threadgroup float* tc = shraw + 32u * (sgid & 1u) + (16u * (sgid >> 1)) * 64u;            \
        CMM_UNROLL(uint i = 0; i < 8u; i++)                                                       \
            simdgroup_store(mc[i], tc + 8u * (i & 3u) + 8u * 64u * (i >> 2), 64u);                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        if (sgid == 0u) {                                                                         \
            for (uint j = lane; j < nr1; j += 32u) {                                              \
                device float* d2 = dst + ro + (ulong)(rt + j) * p.out_f;                          \
                threadgroup const float* c2 = shraw + j * 64u;                                    \
                for (uint i = 0; i < 64u; i++) d2[i] = c2[i];                                     \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
}

#define CMM_KERNEL(NAME, DEC)                                                                     \
    CMM_KERNEL_TYPED(NAME, DEC, half, simdgroup_half8x8, half, half4, simdgroup_half8x8, 2048, 1024)
#define CMM_F16_KERNEL(NAME, DEC)                                                                 \
    CMM_KERNEL_TYPED(NAME, DEC, half, simdgroup_half8x8, float, float4, simdgroup_float8x8, 2048, 1024)
#define CMM_BF16_KERNEL(NAME, DEC)                                                                \
    CMM_KERNEL_TYPED(NAME, DEC, bfloat, simdgroup_bfloat8x8, float, float4, simdgroup_float8x8, 2048, 1024)
#define CMM_F32_KERNEL(NAME, DEC)                                                                 \
    CMM_KERNEL_TYPED(NAME, DEC, float, simdgroup_float8x8, float, float4, simdgroup_float8x8, 3072, 2048)

// Split-K cooperative GEMM for SMALL-m deep-k shapes (a chat turn's short suffix prefill,
// speculative verify's k+1 rows): at m <= 15 the plain tile launches only out_f/64
// threadgroups, each serializing the whole k loop — grid underfill on exactly the shapes
// where the weight stream dominates. ksplit threadgroups share each (token, output) tile,
// each covering a kchunk-slice of k and writing an f32 partial plane [ksplit, m, out_f];
// `cmm_ks_reduce` folds the planes in fixed (ascending) order — deterministic.
struct QLinKsParams { uint m; uint in_f; uint out_f; uint dshift; uint ksplit; uint kchunk; };
#define CMMKS_KERNEL(NAME, DEC)                                                                     \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinKsParams& p   [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint tgix = gid / 128u;                                                                       \
    uint nto = p.out_f / 64u;                                                                     \
    uint ntm = (p.m + 31u) / 32u;                                                                 \
    uint sidx = tgix / (ntm * nto);                                                               \
    if (sidx >= p.ksplit) return;                                                                 \
    tgix %= ntm * nto;                                                                            \
    device float* dstp = dst + (ulong)sidx * p.m * p.out_f;                                       \
    uint to = tgix % nto;                                                                         \
    uint tm = tgix / nto;                                                                         \
    uint ro = to * 64u;                                                                           \
    uint rt = tm * 32u;                                                                           \
    uint tid = sgid * 32u + lane;                                                                 \
    uint nr1 = min(32u, p.m - rt);                                                                \
                                                                                                  \
    threadgroup float shraw[2048];  /* 8 KB: sa(4K half) + sb(2K half); reused f32 for stores */  \
    threadgroup half* sa = (threadgroup half*)shraw;                                              \
    threadgroup half* sb = ((threadgroup half*)shraw) + 2048u;                                    \
                                                                                                  \
    uint lr0 = tid >> 1;                 /* weight (output) row 0..63 */                          \
    uint il0 = tid & 1u;                 /* which 16-element half of the 32-K step */             \
    uint lr1 = tid >> 2;                 /* token row 0..31 */                                    \
    uint iyk = (tid & 3u) * 8u;          /* k offset within the 32-K step */                      \
    uint lr1c = min(lr1, nr1 - 1u);      /* clamp token loads to the matrix edge */               \
                                                                                                  \
    simdgroup_half8x8 ma[4];                                                                      \
    simdgroup_half8x8 mb[2];                                                                      \
    simdgroup_float8x8 mc[8];                                                                     \
    for (uint i = 0; i < 8u; i++) mc[i] = simdgroup_float8x8(0.0f);                               \
                                                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    uint k_lo = sidx * p.kchunk;                                                                  \
    uint k_hi = min(k_lo + p.kchunk, p.in_f);                                                     \
    for (uint k0 = k_lo; k0 < k_hi; k0 += 32u) {                                                  \
        /* device reads + dequant FIRST, into registers — issued while the previous       */     \
        /* iteration's MMA phase (other simdgroups) is still draining; the barrier below  */     \
        /* orders only the threadgroup-memory stores (mul_mm does exactly this)           */     \
        ulong bi = (ulong)(ro + lr0) * nb + (ulong)(k0 >> 4) + il0;                               \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        device const float4* yy =                                                                 \
            (device const float4*)(x + (ulong)(rt + lr1c) * p.in_f + k0 + iyk);                   \
        float4 yv0 = yy[0];                                                                       \
        float4 yv1 = yy[1];                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        {   /* stage A: one 16-block per thread, into pre-transposed 8x8 tiles */                 \
            uint sy = lr0 >> 3;                                                                   \
            uint lx = lr0 & 7u;                                                                   \
            for (uint i = 0; i < 16u; i++) {                                                      \
                uint sx = 2u * il0 + (i >> 3);                                                    \
                sa[64u * (8u * sx + sy) + 8u * (i & 7u) + lx] = (half)wk[i];                      \
            }                                                                                     \
        }                                                                                         \
        {   /* stage B: 8 activations per thread, two vectorized f32->f16 half4 stores */         \
            uint ib = 4u * (tid & 3u) + (lr1 >> 3);                                               \
            uint ly = lr1 & 7u;                                                                   \
            threadgroup half4* sb4 = (threadgroup half4*)(sb + 64u * ib + 8u * ly);               \
            sb4[0] = half4(yv0);                                                                  \
            sb4[1] = half4(yv1);                                                                  \
        }                                                                                         \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        threadgroup const half* lsma = sa + 4u * 64u * (sgid & 1u);                               \
        threadgroup const half* lsmb = sb + 2u * 64u * (sgid >> 1);                               \
        for (uint ik = 0; ik < 4u; ik++) {                                                        \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 4u; i++) simdgroup_load(ma[i], lsma + 64u * i, 8);               \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 2u; i++) simdgroup_load(mb[i], lsmb + 64u * i, 8);               \
            simdgroup_barrier(mem_flags::mem_none);                                               \
            for (uint i = 0; i < 8u; i++)                                                         \
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);              \
            lsma += 8u * 64u;                                                                     \
            lsmb += 4u * 64u;                                                                     \
        }                                                                                         \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
                                                                                                  \
    if (rt + 32u <= p.m) {                                                                        \
        device float* C = dstp + (ro + 32u * (sgid & 1u)) +                                        \
                          (ulong)(rt + 16u * (sgid >> 1)) * p.out_f;                              \
        for (uint i = 0; i < 8u; i++)                                                             \
            simdgroup_store(mc[i], C + 8u * (i & 3u) + 8u * p.out_f * (i >> 2), p.out_f);         \
    } else {                                                                                      \
        /* partial token tile: stage through threadgroup memory, row-guard the copy-out */        \
        threadgroup float* tc = shraw + 32u * (sgid & 1u) + (16u * (sgid >> 1)) * 64u;            \
        for (uint i = 0; i < 8u; i++)                                                             \
            simdgroup_store(mc[i], tc + 8u * (i & 3u) + 8u * 64u * (i >> 2), 64u);                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        if (sgid == 0u) {                                                                         \
            for (uint j = lane; j < nr1; j += 32u) {                                              \
                device float* d2 = dstp + ro + (ulong)(rt + j) * p.out_f;                          \
                threadgroup const float* c2 = shraw + j * 64u;                                    \
                for (uint i = 0; i < 64u; i++) d2[i] = c2[i];                                     \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
}


struct KsReduceParams { uint n; uint ksplit; };
kernel void cmm_ks_reduce(device const float* parts [[buffer(0)]],
                          device float*       dst   [[buffer(1)]],
                          constant KsReduceParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float s = 0.0f;
    for (uint i = 0; i < p.ksplit; i++) s += parts[(ulong)i * p.n + gid];
    dst[gid] = s;
}

CMMKS_KERNEL(linear_quik4_cmm_ks, DEC16_K4)
CMMKS_KERNEL(linear_quik6_cmm_ks, DEC16_K6)
CMMKS_KERNEL(linear_quik8_cmm_ks, DEC16_K8)
CMMKS_KERNEL(linear_q4k_cmm_ks, DEC16_Q4K)
CMMKS_KERNEL(linear_q5k_cmm_ks, DEC16_Q5K)
CMMKS_KERNEL(linear_q6k_cmm_ks, DEC16_Q6K)
CMMKS_KERNEL(linear_q2k_cmm_ks, DEC16_Q2K)
CMMKS_KERNEL(linear_q3k_cmm_ks, DEC16_Q3K)
CMMKS_KERNEL(linear_q8_0_cmm_ks, DEC16_Q8_0)
CMMKS_KERNEL(linear_q5_0_cmm_ks, DEC16_Q5_0)
CMMKS_KERNEL(linear_q4_0_cmm_ks, DEC16_Q4_0)
CMMKS_KERNEL(linear_q5_1_cmm_ks, DEC16_Q5_1)
CMMKS_KERNEL(linear_q4_1_cmm_ks, DEC16_Q4_1)
CMMKS_KERNEL(linear_mxfp4_cmm_ks, DEC16_MXFP4)
CMMKS_KERNEL(linear_nvfp4_cmm_ks, DEC16_NVFP4)
CMMKS_KERNEL(linear_q2_0_cmm_ks, DEC16_Q2_0)
CMMKS_KERNEL(linear_tq2_0_cmm_ks, DEC16_TQ2_0)
CMMKS_KERNEL(linear_tq1_0_cmm_ks, DEC16_TQ1_0)
CMMKS_KERNEL(linear_iq4xs_cmm_ks, DEC16_IQ4XS)
CMMKS_KERNEL(linear_iq4nl_cmm_ks, DEC16_IQ4NL)
CMMKS_KERNEL(linear_iq2xxs_cmm_ks, DEC16_IQ2XXS)
CMMKS_KERNEL(linear_iq3xxs_cmm_ks, DEC16_IQ3XXS)
CMMKS_KERNEL(linear_iq3s_cmm_ks, DEC16_IQ3S)
CMMKS_KERNEL(linear_iq2s_cmm_ks, DEC16_IQ2S)
CMMKS_KERNEL(linear_iq2xs_cmm_ks, DEC16_IQ2XS)
CMMKS_KERNEL(linear_iq1s_cmm_ks, DEC16_IQ1S)
CMMKS_KERNEL(linear_iq1m_cmm_ks, DEC16_IQ1M)

CMM_KERNEL(linear_quik4_cmm, DEC16_K4)
CMM_KERNEL(linear_quik6_cmm, DEC16_K6)
CMM_KERNEL(linear_quik8_cmm, DEC16_K8)
CMM_KERNEL(linear_q4k_cmm, DEC16_Q4K_WIDE)
CMM_KERNEL(linear_q5k_cmm, DEC16_Q5K_WIDE)
CMM_KERNEL(linear_q8_0_cmm, DEC16_Q8_0)
CMM_KERNEL(linear_q5_0_cmm, DEC16_Q5_0)
CMM_KERNEL(linear_q4_0_cmm, DEC16_Q4_0)
CMM_KERNEL(linear_q5_1_cmm, DEC16_Q5_1)
CMM_KERNEL(linear_q4_1_cmm, DEC16_Q4_1)
CMM_KERNEL(linear_mxfp4_cmm, DEC16_MXFP4)
CMM_KERNEL(linear_nvfp4_cmm, DEC16_NVFP4)
CMM_KERNEL(linear_q2_0_cmm, DEC16_Q2_0)
CMM_KERNEL(linear_tq2_0_cmm, DEC16_TQ2_0)
CMM_KERNEL(linear_tq1_0_cmm, DEC16_TQ1_0)
CMM_KERNEL(linear_iq4xs_cmm, DEC16_IQ4XS)
CMM_KERNEL(linear_iq4nl_cmm, DEC16_IQ4NL)
CMM_KERNEL(linear_iq2xxs_cmm, DEC16_IQ2XXS)
CMM_KERNEL(linear_iq3xxs_cmm, DEC16_IQ3XXS)
CMM_KERNEL(linear_iq3s_cmm, DEC16_IQ3S)
CMM_KERNEL(linear_iq2s_cmm, DEC16_IQ2S)
CMM_KERNEL(linear_iq2xs_cmm, DEC16_IQ2XS)
CMM_KERNEL(linear_iq1s_cmm, DEC16_IQ1S)
CMM_KERNEL(linear_iq1m_cmm, DEC16_IQ1M)
CMM_KERNEL(linear_q6k_cmm, DEC16_Q6K)
CMM_KERNEL(linear_q2k_cmm, DEC16_Q2K)
CMM_KERNEL(linear_q3k_cmm, DEC16_Q3K)
CMM_F16_KERNEL(linear_f16_cmm, DEC16_F16)
CMM_BF16_KERNEL(linear_bf16_cmm, DEC16_BF16)
CMM_F32_KERNEL(linear_f32_cmm, DEC16_F32)

HGEMM_KERNEL(linear_quik4_hmm, DEC16_K4)
HGEMM_KERNEL(linear_quik6_hmm, DEC16_K6)
HGEMM_KERNEL(linear_quik8_hmm, DEC16_K8)
HGEMM_KERNEL(linear_q4k_hmm, DEC16_Q4K)
HGEMM_KERNEL(linear_q5k_hmm, DEC16_Q5K)
HGEMM_KERNEL(linear_q6k_hmm, DEC16_Q6K)
HGEMM_KERNEL(linear_q2k_hmm, DEC16_Q2K)
HGEMM_KERNEL(linear_q3k_hmm, DEC16_Q3K)
HGEMM_KERNEL(linear_q8_0_hmm, DEC16_Q8_0)
HGEMM_KERNEL(linear_q5_0_hmm, DEC16_Q5_0)
HGEMM_KERNEL(linear_q4_0_hmm, DEC16_Q4_0)
HGEMM_KERNEL(linear_q5_1_hmm, DEC16_Q5_1)
HGEMM_KERNEL(linear_q4_1_hmm, DEC16_Q4_1)
HGEMM_KERNEL(linear_mxfp4_hmm, DEC16_MXFP4)
HGEMM_KERNEL(linear_nvfp4_hmm, DEC16_NVFP4)
HGEMM_KERNEL(linear_q2_0_hmm, DEC16_Q2_0)
HGEMM_KERNEL(linear_tq2_0_hmm, DEC16_TQ2_0)
HGEMM_KERNEL(linear_tq1_0_hmm, DEC16_TQ1_0)
HGEMM_KERNEL(linear_iq4xs_hmm, DEC16_IQ4XS)
HGEMM_KERNEL(linear_iq4nl_hmm, DEC16_IQ4NL)
HGEMM_KERNEL(linear_iq2xxs_hmm, DEC16_IQ2XXS)
HGEMM_KERNEL(linear_iq3xxs_hmm, DEC16_IQ3XXS)
HGEMM_KERNEL(linear_iq3s_hmm, DEC16_IQ3S)
HGEMM_KERNEL(linear_iq2s_hmm, DEC16_IQ2S)
HGEMM_KERNEL(linear_iq2xs_hmm, DEC16_IQ2XS)
HGEMM_KERNEL(linear_iq1s_hmm, DEC16_IQ1S)
HGEMM_KERNEL(linear_iq1m_hmm, DEC16_IQ1M)
