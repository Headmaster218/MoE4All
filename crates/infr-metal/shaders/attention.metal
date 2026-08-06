// ---- Scaled-dot-product attention (GQA + causal/sliding-window). One SIMD group (32 lanes) per
// (query, head): lanes split head_dim — the q·k score is a lane-strided dot reduced by `simd_sum`,
// and each lane owns a head_dim/32 slice of the online-softmax `acc`. All lanes see the same score,
// so `m`/`l` stay in sync with no cross-lane state. Fixes the old one-thread-per-(query,head) kernel,
// where decode (1 query) ran each head's whole O(kv_len·head_dim) pass on a single thread.
constant constexpr uint MAX_HD = 256;
constant constexpr uint MAX_DPL = MAX_HD / 32u;   // head_dim slots per lane (head_dim ≤ MAX_HD)
struct AttnParams { uint rows; uint kv_len; uint n_head; uint n_kv; uint head_dim; float scale; uint window; uint pos; };
kernel void attention_f32(device const float* q   [[buffer(0)]],
                          device const float* k   [[buffer(1)]],
                          device const float* v   [[buffer(2)]],
                          device float*       dst [[buffer(3)]],
                          constant AttnParams& p  [[buffer(4)]],
                          uint gid  [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.rows * p.n_head) return;
    uint ti = sg / p.n_head;
    uint h = sg % p.n_head;
    uint group = p.n_head / p.n_kv;
    uint kvh = h / group;
    uint qb = sg * p.head_dim;                    // (ti*n_head + h) * head_dim
    uint abs = p.pos + ti;                         // absolute position of this query
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;

    float acc[MAX_DPL];
    for (uint t = 0; t < MAX_DPL; t++) acc[t] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint j = lo; j <= abs; j++) {
        uint kb = (j * p.n_kv + kvh) * p.head_dim;
        float part = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * k[kb + d];
        float sc = simd_sum(part) * p.scale;
        float mnew = max(m, sc);
        float corr = exp(m - mnew);
        float pw = exp(sc - mnew);
        l = l * corr + pw;
        uint vb = (j * p.n_kv + kvh) * p.head_dim;
        uint t = 0;
        for (uint d = lane; d < p.head_dim; d += 32u) { acc[t] = acc[t] * corr + pw * v[vb + d]; t++; }
        m = mnew;
    }
    uint t = 0;
    for (uint d = lane; d < p.head_dim; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
}

// Same as attention_f32, but reads the KV cache in its native f16 straight from the bound buffer
// (no host materialize-to-f32 round-trip). Values match the CPU's f16→f32 read exactly.
kernel void attention_f16kv(device const float* q   [[buffer(0)]],
                            device const half*  k   [[buffer(1)]],
                            device const half*  v   [[buffer(2)]],
                            device float*       dst [[buffer(3)]],
                            constant AttnParams& p  [[buffer(4)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.rows * p.n_head) return;
    uint ti = sg / p.n_head;
    uint h = sg % p.n_head;
    uint group = p.n_head / p.n_kv;
    uint kvh = h / group;
    uint qb = sg * p.head_dim;
    uint abs = p.pos + ti;
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;

    float acc[MAX_DPL];
    for (uint t = 0; t < MAX_DPL; t++) acc[t] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint j = lo; j <= abs; j++) {
        uint kb = (j * p.n_kv + kvh) * p.head_dim;
        float part = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * (float)k[kb + d];
        float sc = simd_sum(part) * p.scale;
        float mnew = max(m, sc);
        float corr = exp(m - mnew);
        float pw = exp(sc - mnew);
        l = l * corr + pw;
        uint vb = (j * p.n_kv + kvh) * p.head_dim;
        uint t = 0;
        for (uint d = lane; d < p.head_dim; d += 32u) { acc[t] = acc[t] * corr + pw * (float)v[vb + d]; t++; }
        m = mnew;
    }
    uint t = 0;
    for (uint d = lane; d < p.head_dim; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
}

// ---- Split-KV ("flash-decode") attention: same math as attention_*, but NSG simdgroups per
// (query, head) threadgroup, each running a private online softmax over a strided slice of the KV
// positions, merged at the end through threadgroup memory (rescale each partial to the global max,
// sum, divide). Exists because decode has rows=1: the one-simdgroup kernel then launches only
// `n_head` simdgroups — far too few to occupy the GPU — and its runtime grows O(kv_len) on that
// fixed tiny width. Split kernels multiply decode parallelism by NSG; the host routes here only
// when rows*n_head is small, so prefill keeps the leaner kernel (this one's static ~8 KB of
// threadgroup memory would cap prefill occupancy).
// One macro instantiates each (KV type, split width) variant. NSG=8 covers short contexts and any
// head_dim; NSG=32 quarters the serial online-softmax chain per simdgroup (the kernel is
// latency-bound on that chain, ~kv_len/NSG dependent steps), but its threadgroup accumulator only
// fits head_dim <= 128 in the 32 KB threadgroup-memory budget, so the host routes to it only for
// long-context decode at hd <= 128.
#define ATTNSPLIT_KERNEL(NAME, KVT, NSG, MAXHD)                                                    \
kernel void NAME(device const float* q   [[buffer(0)]],                                           \
                 device const KVT*   k   [[buffer(1)]],                                           \
                 device const KVT*   v   [[buffer(2)]],                                           \
                 device float*       dst [[buffer(3)]],                                           \
                 constant AttnParams& p  [[buffer(4)]],                                           \
                 uint3 tgpig [[threadgroup_position_in_grid]],                                    \
                 uint sgid [[simdgroup_index_in_threadgroup]],                                    \
                 uint lane [[thread_index_in_simdgroup]]) {                                       \
    uint tg = tgpig.x;                                                                            \
    if (tg >= p.rows * p.n_head) return;   /* uniform per threadgroup — safe with the barrier */  \
    uint ti = tg / p.n_head;                                                                      \
    uint h = tg % p.n_head;                                                                       \
    uint group = p.n_head / p.n_kv;                                                               \
    uint kvh = h / group;                                                                         \
    uint qb = tg * p.head_dim;                                                                    \
    uint abs = p.pos + ti;                                                                        \
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;                \
                                                                                                  \
    float acc[MAXHD / 32u];                                                                       \
    for (uint t = 0; t < MAXHD / 32u; t++) acc[t] = 0.0f;                                         \
    float m = -INFINITY, l = 0.0f;                                                                \
    for (uint j = lo + sgid; j <= abs; j += NSG) {                                                \
        uint kb = (j * p.n_kv + kvh) * p.head_dim;                                                \
        float part = 0.0f;                                                                        \
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * (float)k[kb + d];       \
        float sc = simd_sum(part) * p.scale;                                                      \
        float mnew = max(m, sc);                                                                  \
        float corr = exp(m - mnew);                                                               \
        float pw = exp(sc - mnew);                                                                \
        l = l * corr + pw;                                                                        \
        uint t = 0;                                                                               \
        for (uint d = lane; d < p.head_dim; d += 32u) {                                           \
            acc[t] = acc[t] * corr + pw * (float)v[kb + d];                                       \
            t++;                                                                                  \
        }                                                                                         \
        m = mnew;                                                                                 \
    }                                                                                             \
    /* Merge the NSG partials. A simdgroup whose slice was empty has l==0 (skip; its m is -inf) */ \
    threadgroup float tg_m[NSG], tg_l[NSG], tg_acc[NSG * MAXHD];                                  \
    if (lane == 0u) { tg_m[sgid] = m; tg_l[sgid] = l; }                                           \
    uint t = 0;                                                                                   \
    for (uint d = lane; d < p.head_dim; d += 32u) {                                               \
        tg_acc[sgid * p.head_dim + d] = acc[t];                                                   \
        t++;                                                                                      \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
    if (sgid == 0u) {                                                                             \
        float gm = -INFINITY;                                                                     \
        for (uint i = 0; i < NSG; i++) if (tg_l[i] > 0.0f) gm = max(gm, tg_m[i]);                 \
        float gl = 0.0f;                                                                          \
        float w[NSG];                                                                             \
        for (uint i = 0; i < NSG; i++) {                                                          \
            w[i] = (tg_l[i] > 0.0f) ? exp(tg_m[i] - gm) : 0.0f;                                   \
            gl += tg_l[i] * w[i];                                                                 \
        }                                                                                         \
        for (uint d = lane; d < p.head_dim; d += 32u) {                                           \
            float s = 0.0f;                                                                       \
            for (uint i = 0; i < NSG; i++) s += tg_acc[i * p.head_dim + d] * w[i];                \
            dst[qb + d] = s / gl;                                                                 \
        }                                                                                         \
    }                                                                                             \
}

ATTNSPLIT_KERNEL(attnsplit_f32, float, 8u, 256u)
ATTNSPLIT_KERNEL(attnsplit_f16kv, half, 8u, 256u)
ATTNSPLIT_KERNEL(attnsplit32_f32, float, 32u, 128u)
ATTNSPLIT_KERNEL(attnsplit32_f16kv, half, 32u, 128u)

// ---- Canvas split-KV attention (DiffusionGemma denoise, `AttnMask::Canvas` —
// docs/diffusion-gemma.md): EVERY row attends the SAME fixed bidirectional `[lo, kv_len)` reach,
// unlike ATTNSPLIT_KERNEL above (and every other kernel in this file), which derives a per-row
// causal end from `pos + ti` and an optional sliding-window `lo`. Repurposing ATTNSPLIT_KERNEL's
// fields with a sentinel would risk colliding with a genuine non-canvas dispatch (both would see
// a nonzero `p.kv_len`, since that field — while dead in ATTNSPLIT_KERNEL — is always sent as the
// real cache length by the ordinary routing path), so this is a DEDICATED kernel that never serves
// a non-canvas caller: `p.pos` carries the fixed `hi = kv_len - 1` (identical for every row,
// instead of `pos + ti`) and `p.kv_len` carries `lo` directly (not the cache length). Otherwise
// identical split-KV structure (NSG simdgroups per (query, head), merged through threadgroup
// memory) — see ATTNSPLIT_KERNEL's doc for the shape rationale.
// UNVERIFIED: ported from the Vulkan `attn_partial.comp` canvas branch (`attention_kv_split`'s
// `canvas_lo` push field, `adapter.rs`) — Metal is Phase D's blind backend for this mask, never
// run on hardware; CPU + Vulkan remain the validated references.
#define ATTNSPLIT_CANVAS_KERNEL(NAME, KVT, NSG, MAXHD)                                             \
kernel void NAME(device const float* q   [[buffer(0)]],                                           \
                 device const KVT*   k   [[buffer(1)]],                                           \
                 device const KVT*   v   [[buffer(2)]],                                           \
                 device float*       dst [[buffer(3)]],                                           \
                 constant AttnParams& p  [[buffer(4)]],                                           \
                 uint3 tgpig [[threadgroup_position_in_grid]],                                    \
                 uint sgid [[simdgroup_index_in_threadgroup]],                                    \
                 uint lane [[thread_index_in_simdgroup]]) {                                       \
    uint tg = tgpig.x;                                                                            \
    if (tg >= p.rows * p.n_head) return;   /* uniform per threadgroup — safe with the barrier */  \
    uint h = tg % p.n_head;                                                                       \
    uint group = p.n_head / p.n_kv;                                                               \
    uint kvh = h / group;                                                                         \
    uint qb = tg * p.head_dim;                                                                    \
    uint hi = p.pos;      /* fixed for every row: kv_len - 1, NOT pos + ti */                      \
    uint lo = p.kv_len;   /* fixed for every row: the mask's lo, NOT the cache length */           \
                                                                                                  \
    float acc[MAXHD / 32u];                                                                       \
    for (uint t = 0; t < MAXHD / 32u; t++) acc[t] = 0.0f;                                         \
    float m = -INFINITY, l = 0.0f;                                                                \
    for (uint j = lo + sgid; j <= hi; j += NSG) {                                                 \
        uint kb = (j * p.n_kv + kvh) * p.head_dim;                                                \
        float part = 0.0f;                                                                        \
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * (float)k[kb + d];       \
        float sc = simd_sum(part) * p.scale;                                                      \
        float mnew = max(m, sc);                                                                  \
        float corr = exp(m - mnew);                                                               \
        float pw = exp(sc - mnew);                                                                \
        l = l * corr + pw;                                                                        \
        uint t = 0;                                                                               \
        for (uint d = lane; d < p.head_dim; d += 32u) {                                           \
            acc[t] = acc[t] * corr + pw * (float)v[kb + d];                                       \
            t++;                                                                                  \
        }                                                                                         \
        m = mnew;                                                                                 \
    }                                                                                             \
    /* Merge the NSG partials. A simdgroup whose slice was empty has l==0 (skip; its m is -inf) */ \
    threadgroup float tg_m[NSG], tg_l[NSG], tg_acc[NSG * MAXHD];                                  \
    if (lane == 0u) { tg_m[sgid] = m; tg_l[sgid] = l; }                                           \
    uint t = 0;                                                                                   \
    for (uint d = lane; d < p.head_dim; d += 32u) {                                               \
        tg_acc[sgid * p.head_dim + d] = acc[t];                                                   \
        t++;                                                                                      \
    }                                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                              \
    if (sgid == 0u) {                                                                             \
        float gm = -INFINITY;                                                                     \
        for (uint i = 0; i < NSG; i++) if (tg_l[i] > 0.0f) gm = max(gm, tg_m[i]);                 \
        float gl = 0.0f;                                                                          \
        float w[NSG];                                                                             \
        for (uint i = 0; i < NSG; i++) {                                                          \
            w[i] = (tg_l[i] > 0.0f) ? exp(tg_m[i] - gm) : 0.0f;                                   \
            gl += tg_l[i] * w[i];                                                                 \
        }                                                                                         \
        for (uint d = lane; d < p.head_dim; d += 32u) {                                           \
            float s = 0.0f;                                                                       \
            for (uint i = 0; i < NSG; i++) s += tg_acc[i * p.head_dim + d] * w[i];                \
            dst[qb + d] = s / gl;                                                                 \
        }                                                                                         \
    }                                                                                             \
}

ATTNSPLIT_CANVAS_KERNEL(attention_canvas_f32, float, 8u, 256u)
ATTNSPLIT_CANVAS_KERNEL(attention_canvas_f16kv, half, 8u, 256u)
ATTNSPLIT_CANVAS_KERNEL(attention_canvas32_f32, float, 32u, 128u)
ATTNSPLIT_CANVAS_KERNEL(attention_canvas32_f16kv, half, 32u, 128u)

// ---- Half-fragment flash attention for prefill (f16 KV cache, hd <= 128, hd % 8 == 0): one
// simdgroup per (8-query tile, head). Unlike the earlier f32 simdgroup_matrix attempt (built,
// benched, lost to the scalar split-KV kernel, removed), there is NO staging: K^T and V 8x8
// fragments load DIRECTLY from the f16 cache (strided, transposed for K), and Q is pre-cast once
// per op to f16 — the f32 version spent its time converting K/V through threadgroup memory and
// choked occupancy on the 8 KB tiles. Scores and the output tile accumulate in f32; the online
// softmax runs scalar in f32 on an 8x8 score tile per 8-position KV block, with the row-rescale
// applied as a diagonal f32 MMA. P rounds to f16 (same trade as the half-fragment GEMM).
// Tail KV blocks may read up to 7 rows past the causal limit — always inside the bound cache
// buffer (sized for the full context) — and those columns are masked in the softmax.
// A partial final query tile falls back to the serial per-query path.
kernel void attnflash_f16kv(device const half*  q   [[buffer(0)]],
                            device const half*  k   [[buffer(1)]],
                            device const half*  v   [[buffer(2)]],
                            device float*       dst [[buffer(3)]],
                            constant AttnParams& p  [[buffer(4)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    uint ntq = (p.rows + 7u) / 8u;
    if (sg >= ntq * p.n_head) return;
    /* same-head query tiles are ADJACENT simdgroups: concurrent tiles then stream the SAME
       head's KV region and hit the SLC, instead of 16 heads' regions at once (measured: the
       head-fastest order collapsed pp8k to ~1/3 of llama.cpp) */
    uint qt = sg % ntq;
    uint h = sg / ntq;
    uint group = p.n_head / p.n_kv;
    uint kvh = h / group;
    uint hd = p.head_dim;
    uint r0 = qt * 8u;

    if (r0 + 8u > p.rows) {
        // partial query tile: serial per-query fallback (lane-split dot, online softmax)
        for (uint ti = r0; ti < p.rows; ti++) {
            uint qb = (ti * p.n_head + h) * hd;
            uint abs = p.pos + ti;
            uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;
            float acc[MAX_DPL];
            for (uint t = 0; t < MAX_DPL; t++) acc[t] = 0.0f;
            float m = -INFINITY, l = 0.0f;
            for (uint j = lo; j <= abs; j++) {
                ulong kb = ((ulong)j * p.n_kv + kvh) * hd;
                float part = 0.0f;
                for (uint d = lane; d < hd; d += 32u) part += (float)q[qb + d] * (float)k[kb + d];
                float sc = simd_sum(part) * p.scale;
                float mnew = max(m, sc);
                float corr = exp(m - mnew);
                float pw = exp(sc - mnew);
                l = l * corr + pw;
                uint t = 0;
                for (uint d = lane; d < hd; d += 32u) {
                    acc[t] = acc[t] * corr + pw * (float)v[kb + d];
                    t++;
                }
                m = mnew;
            }
            uint t = 0;
            for (uint d = lane; d < hd; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
        }
        return;
    }

    threadgroup half tgP[128];
    threadgroup float tgS16[128];
    threadgroup float tgD[64];
    threadgroup float tgM[8], tgL[8];
    uint abs0 = p.pos + r0;                 // row i sees positions <= abs0 + i
    uint abs_max = abs0 + 7u;
    uint lo_min = (p.window > 0u && abs0 + 1u > p.window) ? (abs0 + 1u - p.window) : 0u;
    if (lane < 8u) { tgM[lane] = -INFINITY; tgL[lane] = 0.0f; }

    device const half* qbase = q + ((ulong)r0 * p.n_head + h) * hd;
    ulong qstride = (ulong)p.n_head * hd;
    ulong kvstride = (ulong)p.n_kv * hd;

    simdgroup_float8x8 oa[16];
    uint nfrag = hd / 8u;
    for (uint i = 0; i < nfrag; i++) oa[i] = simdgroup_float8x8(0.0f);

    for (uint j0 = lo_min & ~15u; j0 <= abs_max; j0 += 16u) {
        /* two 8-position score fragments per iteration: one scalar softmax phase and one
           rescale per 16 KV positions instead of per 8 — the scalar phase and its barriers,
           not KV bandwidth, are what this kernel waits on */
        device const half* kb = k + ((ulong)j0 * p.n_kv + kvh) * hd;
        simdgroup_float8x8 sf0 = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 sf1 = simdgroup_float8x8(0.0f);
        for (uint e0 = 0; e0 < hd; e0 += 8u) {
            simdgroup_half8x8 qa, kt;
            simdgroup_load(qa, qbase + e0, qstride);
            simdgroup_load(kt, kb + e0, kvstride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sf0, qa, kt, sf0);
            simdgroup_load(kt, kb + 8u * kvstride + e0, kvstride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sf1, qa, kt, sf1);
        }
        simdgroup_store(sf0, tgS16, 16);      // f32 score scratch, 8 rows x 16 cols
        simdgroup_store(sf1, tgS16 + 8u, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8u) {
            uint r = lane;
            uint absr = abs0 + r;
            uint lor = (p.window > 0u && absr + 1u > p.window) ? (absr + 1u - p.window) : 0u;
            float mr = tgM[r];
            float mnew = mr;
            float s[16];
            for (uint c = 0; c < 16u; c++) {
                uint j = j0 + c;
                bool valid = (j >= lor) && (j <= absr);
                s[c] = valid ? tgS16[r * 16u + c] * p.scale : -INFINITY;
                mnew = max(mnew, s[c]);
            }
            float corr = (mr == mnew) ? 1.0f : exp(mr - mnew);
            float lsum = 0.0f;
            for (uint c = 0; c < 16u; c++) {
                float pw = (s[c] == -INFINITY) ? 0.0f : exp(s[c] - mnew);
                tgP[r * 16u + c] = (half)pw;
                lsum += pw;
            }
            tgL[r] = tgL[r] * corr + lsum;
            tgM[r] = mnew;
            for (uint c = 0; c < 8u; c++) tgD[r * 8u + c] = (c == r) ? corr : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 df;
        simdgroup_half8x8 pf0, pf1;
        simdgroup_load(df, tgD, 8);
        simdgroup_load(pf0, tgP, 16);
        simdgroup_load(pf1, tgP + 8u, 16);
        device const half* vb = v + ((ulong)j0 * p.n_kv + kvh) * hd;
        for (uint i = 0; i < nfrag; i++) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, df, oa[i]);
            simdgroup_half8x8 vf;
            simdgroup_load(vf, vb + i * 8u, kvstride);
            simdgroup_multiply_accumulate(tmp, pf0, vf, tmp);
            simdgroup_load(vf, vb + 8u * kvstride + i * 8u, kvstride);
            simdgroup_multiply_accumulate(oa[i], pf1, vf, tmp);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < 8u) {
        for (uint c = 0; c < 8u; c++) tgD[lane * 8u + c] = (c == lane) ? 1.0f / tgL[lane] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 d2;
    simdgroup_load(d2, tgD, 8);
    ulong obase = ((ulong)r0 * p.n_head + h) * hd;
    ulong ostride = (ulong)p.n_head * hd;
    for (uint i = 0; i < nfrag; i++) {
        simdgroup_float8x8 tmp;
        simdgroup_multiply(tmp, d2, oa[i]);
        simdgroup_store(tmp, dst + obase + i * 8u, ostride);
    }
}

// ---- Cooperative flash attention for prefill (f16 KV cache, hd 64 or 128 instantiations): the
// llama.cpp `kernel_flash_attn_ext` structure — NSG=4 simdgroups cooperate on ONE (8-query tile,
// head) threadgroup, processing C=64 KV positions per iteration. The phases split the work along
// different axes so every lane stays busy (the single-simdgroup attnflash_f16kv above stalls in
// its scalar softmax, 8 of 32 lanes active, one phase per 16 KV):
//   QK^T    — the 8 score fragments (64 KV cols x 8 queries) split across simdgroups, 2 each;
//             K^T fragments load DIRECTLY from the f16 cache (transposed, no staging).
//   softmax — split by query ROWS (2 rows per simdgroup); each row's 64 scores are one float2
//             per lane, so all 32 lanes work; the online max/sum (M/S) stats live in that
//             simdgroup's registers for the whole KV loop — no cross-simdgroup stat merges.
//   P*V     — split by output COLUMNS (hd/32 8x8 O fragments per simdgroup) held in registers
//             across the MMA, staged through threadgroup `so` only for the softmax rescale.
// Masking is analytic (causal + window per row) — no mask buffer, no -inf staging; masked lanes
// force pw=0, and M is floored at -MAXFLOAT/2 so an all-masked block leaves S/O untouched.
// A partial final query tile zero-pads Q rows in shared memory and skips their output store
// (the fallback serial path in attnflash_f16kv is not needed here). Score/O accumulation is f32;
// P rounds through f32 shared and enters the V MMA as an f32 fragment against half V fragments.
// Tail KV blocks read up to 7 rows past the causal limit (same in-buffer contract as above);
// 8-row blocks entirely past it are skipped, so reads never go further.
template<uint hd, uint NSG, uint C> // compile-time shape: fully unrolled, exact shared sizing
kernel void attnflash2_f16kv_t(device const half*  q   [[buffer(0)]],
                               device const half*  k   [[buffer(1)]],
                               device const half*  v   [[buffer(2)]],
                               device float*       dst [[buffer(3)]],
                               constant AttnParams& p  [[buffer(4)]],
                               uint3  tgpig [[threadgroup_position_in_grid]],
                               ushort sgitg [[simdgroup_index_in_threadgroup]],
                               ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint QT = 8, NQ = QT / NSG, SH = C;
    constexpr uint NP = C / 64u;                  // score pairs owned per lane
    threadgroup half  sq[QT * hd];    // Q tile (rows x hd, half)
    threadgroup float so[QT * hd];    // O accumulator (rows x hd, f32)
    threadgroup float ss[QT * SH];    // scores, then P, per KV block (rows x C, f32)

    uint ntq = (p.rows + QT - 1u) / QT;
    uint qt = tgpig.x % ntq;          // same-head tiles adjacent (SLC — see attnflash_f16kv)
    uint h  = tgpig.x / ntq;
    constexpr uint hd4 = hd / 4u;
    constexpr uint no = hd / (8u * NSG);   // O column fragments owned per simdgroup
    constexpr uint NC = (C / 8u) / NSG;    // score fragments owned per simdgroup
    uint kvh = h / (p.n_head / p.n_kv);
    uint r0 = qt * QT;
    uint abs0 = p.pos + r0;
    uint abs_max = p.pos + min(p.rows - 1u, r0 + QT - 1u);
    uint lo_min = (p.window > 0u && abs0 + 1u > p.window) ? (abs0 + 1u - p.window) : 0u;
    ulong qstride = (ulong)p.n_head * hd;
    ulong kvstride = (ulong)p.n_kv * hd;

    // stage Q rows to shared (zero rows past p.rows), zero the O accumulator
    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        bool live = r0 + j < p.rows;
        // clamp the dead-row pointer (a select may still speculate the load)
        device const half4* q4 =
            (device const half4*)(q + (ulong)min(r0 + j, p.rows - 1u) * qstride + (ulong)h * hd);
        threadgroup half4*  sq4 = (threadgroup half4*)sq + j * hd4;
        threadgroup float4* so4 = (threadgroup float4*)so + j * hd4;
        for (uint i = tiisg; i < hd4; i += 32u) {
            sq4[i] = live ? q4[i] : half4(0.0h);
            so4[i] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (uint jj = 0; jj < NQ; jj++) { S[jj] = 0.0f; M[jj] = -MAXFLOAT / 2; }

    for (uint ic = lo_min & ~(C - 1u); ic <= abs_max; ic += C) {
        // Q*K^T — C / 8 score fragments split across simdgroups (fragment f covers KV rows
        // ic+8f, columns interleaved so each simdgroup's fragments are NSG apart)
        {
            device const half* pk = k + ((ulong)(ic + 8u * sgitg) * p.n_kv + kvh) * hd;
            threadgroup float* ps = ss + 8u * sgitg;
            for (uint cc = 0; cc < NC; cc++) {
                simdgroup_float8x8 mqk = simdgroup_float8x8(0.0f);
                if (ic + 8u * (sgitg + cc * NSG) <= abs_max) {
                    for (uint i = 0; i < hd; i += 16u) {
                        simdgroup_half8x8 mq, mk;
                        simdgroup_load(mq, sq + i, hd);
                        simdgroup_load(mk, pk + i, kvstride, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        simdgroup_load(mq, sq + i + 8u, hd);
                        simdgroup_load(mk, pk + i + 8u, kvstride, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                    }
                }
                simdgroup_store(mqk, ps, SH);
                pk += (ulong)(8u * NSG) * kvstride;
                ps += 8u * NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax — rows split across simdgroups, C / 32 scores (float2s) per lane
        for (uint jj = 0; jj < NQ; jj++) {
            uint j = jj * NSG + sgitg;
            uint absr = abs0 + j;   // rows past p.rows compute junk, never stored
            uint lor = (p.window > 0u && absr + 1u > p.window) ? (absr + 1u - p.window) : 0u;
            float m = M[jj];
            threadgroup float2* ss2 = (threadgroup float2*)(ss + j * SH);
            float2 scores[NP];
            bool2 valid[NP];
            float local_max = m;
            for (uint kk = 0; kk < NP; kk++) {
                scores[kk] = ss2[tiisg + 32u * kk] * p.scale;
                uint c0 = ic + 64u * kk + 2u * tiisg;
                valid[kk] = bool2((c0 >= lor) && (c0 <= absr),
                                  (c0 + 1u >= lor) && (c0 + 1u <= absr));
                local_max = max(local_max,
                                max(valid[kk].x ? scores[kk].x : -MAXFLOAT / 2,
                                    valid[kk].y ? scores[kk].y : -MAXFLOAT / 2));
            }
            float mnew = simd_max(local_max);
            float ms = exp(m - mnew);
            float local_sum = 0.0f;
            for (uint kk = 0; kk < NP; kk++) {
                float2 pw = float2(valid[kk].x ? exp(scores[kk].x - mnew) : 0.0f,
                                   valid[kk].y ? exp(scores[kk].y - mnew) : 0.0f);
                ss2[tiisg + 32u * kk] = pw;
                local_sum += pw.x + pw.y;
            }
            S[jj] = S[jj] * ms + simd_sum(local_sum);
            M[jj] = mnew;
            threadgroup float4* so4 = (threadgroup float4*)so + j * hd4;
            for (uint i = tiisg; i < hd4; i += 32u) so4[i] *= ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O += P*V — O column fragments split across simdgroups, held in registers; V fragments
        // load directly from the f16 cache; fully-causal-masked 8-row KV blocks skipped (their
        // P is all zero, and skipping keeps reads within 7 rows of the limit)
        {
            simdgroup_float8x8 lo[no];
            threadgroup float* sot = so + 8u * sgitg;
            for (uint ii = 0; ii < no; ii++) simdgroup_load(lo[ii], sot + 8u * NSG * ii, hd);
            device const half* pv = v + ((ulong)ic * p.n_kv + kvh) * hd + 8u * sgitg;
            // only KV blocks up to the causal limit (P is zero past it, and skipping keeps
            // reads within 7 rows); paired blocks keep 2 P and 2*no V loads in flight
            uint nblk = min(C / 8u, (abs_max - ic) / 8u + 1u);
            for (uint cc = 0; cc + 1u < nblk; cc += 2u) {
                simdgroup_float8x8 vs[2];
                simdgroup_load(vs[0], ss + 8u * cc, SH);
                simdgroup_load(vs[1], ss + 8u * cc + 8u, SH);
                for (uint ii = 0; ii < no; ii++) {
                    simdgroup_half8x8 mv[2];
                    simdgroup_load(mv[0], pv + 8u * NSG * ii, kvstride);
                    simdgroup_load(mv[1], pv + 8u * NSG * ii + 8u * kvstride, kvstride);
                    simdgroup_multiply_accumulate(lo[ii], vs[0], mv[0], lo[ii]);
                    simdgroup_multiply_accumulate(lo[ii], vs[1], mv[1], lo[ii]);
                }
                pv += 16u * kvstride;
            }
            if (nblk & 1u) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8u * (nblk - 1u), SH);
                for (uint ii = 0; ii < no; ii++) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + 8u * NSG * ii, kvstride);
                    simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                }
            }
            for (uint ii = 0; ii < no; ii++) simdgroup_store(lo[ii], sot + 8u * NSG * ii, hd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / S — same row split as the softmax (that simdgroup holds the row's S)
    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        if (r0 + j >= p.rows) continue;
        float sc = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        device float4* out = (device float4*)(dst + ((ulong)(r0 + j)) * qstride + (ulong)h * hd);
        threadgroup const float4* so4 = (threadgroup const float4*)so + j * hd4;
        for (uint i = tiisg; i < hd4; i += 32u) out[i] = so4[i] * sc;
    }
}

typedef decltype(attnflash2_f16kv_t<64, 4, 64>) attnflash2_t;
template [[host_name("attnflash2_f16kv_hd64")]]  kernel attnflash2_t attnflash2_f16kv_t<64, 4, 64>;
template [[host_name("attnflash2_f16kv_hd128")]] kernel attnflash2_t attnflash2_f16kv_t<128, 4, 64>;
// hd=256 (gemma): sq 4KB + so 8KB + ss 2KB = 14KB shared, 8 O fragments per simdgroup.
template [[host_name("attnflash2_f16kv_hd256")]] kernel attnflash2_t attnflash2_f16kv_t<256, 4, 64>;
typedef decltype(attnflash2_f16kv_t<128, 4, 128>) attnflash2_c128_t;
template [[host_name("attnflash2_c128_f16kv_hd128")]]
kernel attnflash2_c128_t attnflash2_f16kv_t<128, 4, 128>;

// ---- Vector flash attention for decode (f16 KV cache, hd 64 or 128, one query row per
// threadgroup): the llama.cpp `kernel_flash_attn_ext_vec` structure. NSG simdgroups each own
// interleaved C=32-position KV blocks with a PRIVATE online softmax, merged once at the end by a
// log2 tree — same split-KV idea as attnsplit32 above, but each simdgroup step handles 32
// positions instead of 1: lanes fold as (ty, tx) = 4 KV rows x 8-lane dots, a shuffle tree
// reduces the 8 partials per row, and ONE simd_max/simd_sum softmax pass covers the whole block.
// That cuts the serial chain per simdgroup from kv_len/NSG simd reductions to kv_len/(NSG*32)
// block passes — the attnsplit kernels are latency-bound on exactly that chain at long context.
// Q stays f32 in shared (no rounding: f32 dots over exactly-widened f16 K/V, same numeric class
// as attnsplit32, only reassociated). Tail positions clamp their row pointer to kv_len-1 and are
// masked in the softmax, so reads never leave the cache. O accumulates in shared per simdgroup
// (ty==0 lanes own hd/32 float4 columns each after the fold).
// The body is a plain inline function so the static kernel (baked pos/kv_len from AttnParams)
// and the DYNAMIC-POS kernel (pos read from the bound positions buffer — the decode-replay
// contract, where one recorded dispatch is replayed every token) share it exactly.
template<uint hd, uint NSG>
inline void attnvec_body(device const float* q,
                         device const half*  k,
                         device const half*  v,
                         device float*       dst,
                         constant AttnParams& p,
                         uint abs, uint kvl,
                         threadgroup float* sq,
                         threadgroup float* ssc,
                         threadgroup float* so,
                         uint3  tgpig,
                         ushort sgitg,
                         ushort tiisg) {
    constexpr uint C = 32, NE = 4, NL = 32u / NE;  // 4 KV rows x 8-lane dots per simdgroup pass
    constexpr uint hd4 = hd / 4u;
    constexpr uint NI = hd4 / NL;                  // float4s per lane per row (2 or 4)

    uint tg = tgpig.x;
    uint ti = tg / p.n_head;
    uint h  = tg % p.n_head;
    uint kvh = h / (p.n_head / p.n_kv);
    uint lo = (p.window > 0u && abs + 1u > p.window) ? (abs + 1u - p.window) : 0u;
    uint tx = tiisg % NL, ty = tiisg / NL;

    {
        device const float4* q4 = (device const float4*)(q + ((ulong)ti * p.n_head + h) * hd);
        threadgroup float4* sq4 = (threadgroup float4*)sq;
        for (uint i = sgitg * 32u + tiisg; i < hd4; i += NSG * 32u) sq4[i] = q4[i];
    }
    threadgroup float* ss = ssc + sgitg * C;
    threadgroup float4* so4 = (threadgroup float4*)so + sgitg * hd4;
    if (ty == 0) {
        for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f, M = -MAXFLOAT / 2;
    device const half4* k4 = (device const half4*)k;
    device const half4* v4 = (device const half4*)v;
    threadgroup const float4* sq4 = (threadgroup const float4*)sq;

    for (uint ic = sgitg * C; ic <= abs; ic += NSG * C) {
        if (ic + C <= lo) continue;   // whole block below the window (uniform per simdgroup)

        // Q*K^T — each (ty, tx) fold: row ic + NE*cc + ty, 8-lane split of the hd dot
        {
            float mqk[C / NE];
            for (uint cc = 0; cc < C / NE; cc++) {
                // clamp tail rows into the cache; their scores are masked below
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                device const half4* pk = k4 + ((ulong)rc * p.n_kv + kvh) * hd4;
                float acc = 0.0f;
                for (uint ii = 0; ii < NI; ii++)
                    acc += dot(float4(pk[ii * NL + tx]), sq4[ii * NL + tx]);
                // fold the 8 tx-lane partials of each row
                acc += simd_shuffle_down(acc, 4);
                acc += simd_shuffle_down(acc, 2);
                acc += simd_shuffle_down(acc, 1);
                mqk[cc] = simd_shuffle(acc, NL * ty);  // broadcast row sum within the ty group
            }
            ss[NE * tx + ty] = mqk[tx];  // lane (tx, ty) stores score of row ic + NE*tx + ty
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax — one pass over the whole 32-score block
        {
            float s = ss[tiisg] * p.scale;
            uint jkv = ic + tiisg;
            bool valid = (jkv >= lo) && (jkv <= abs);
            float m = M;
            M = simd_max(max(M, valid ? s : -MAXFLOAT / 2));
            float ms = exp(m - M);
            float vs = valid ? exp(s - M) : 0.0f;
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) {
                for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] *= ms;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O += P*V — same (ty, tx) fold as Q*K^T, accumulated into the ty==0 lanes' columns
        {
            float4 lov[NI];
            for (uint ii = 0; ii < NI; ii++) lov[ii] = float4(0.0f);
            for (uint cc = 0; cc < C / NE; cc++) {
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                device const half4* pv = v4 + ((ulong)rc * p.n_kv + kvh) * hd4;
                float pw = ss[NE * cc + ty];
                for (uint ii = 0; ii < NI; ii++)
                    lov[ii] += float4(pv[ii * NL + tx]) * pw;
            }
            for (uint ii = 0; ii < NI; ii++) {
                lov[ii] += simd_shuffle_down(lov[ii], 16);
                lov[ii] += simd_shuffle_down(lov[ii], 8);
            }
            if (ty == 0) {
                for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] += lov[ii];
            }
        }
    }

    // publish (S, M) for the merge (scores are dead)
    if (tiisg == 0) { ss[0] = S; ss[1] = M; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // log2 merge of the per-simdgroup partials
    for (uint rr = NSG / 2u; rr > 0u; rr >>= 1u) {
        if (sgitg < rr) {
            float s0 = ss[0], s1 = ssc[(sgitg + rr) * C + 0];
            float m0 = ss[1], m1 = ssc[(sgitg + rr) * C + 1];
            float mm = max(m0, m1);
            float ms0 = exp(m0 - mm), ms1 = exp(m1 - mm);
            if (tiisg == 0) { ss[0] = s0 * ms0 + s1 * ms1; ss[1] = mm; }
            threadgroup float4* sob = (threadgroup float4*)so + (sgitg + rr) * hd4;
            for (uint i = tiisg; i < hd4; i += 32u) so4[i] = so4[i] * ms0 + sob[i] * ms1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        float sinv = ssc[0] == 0.0f ? 0.0f : 1.0f / ssc[0];
        device float4* out = (device float4*)(dst + ((ulong)ti * p.n_head + h) * hd);
        for (uint i = tiisg; i < hd4; i += 32u) out[i] = so4[i] * sinv;
    }
}

template<uint hd, uint NSG>
kernel void attnvec_f16kv_t(device const float* q   [[buffer(0)]],
                            device const half*  k   [[buffer(1)]],
                            device const half*  v   [[buffer(2)]],
                            device float*       dst [[buffer(3)]],
                            constant AttnParams& p  [[buffer(4)]],
                            uint3  tgpig [[threadgroup_position_in_grid]],
                            ushort sgitg [[simdgroup_index_in_threadgroup]],
                            ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float sq[hd];
    threadgroup float ssc[NSG * 32];               // per-simdgroup scores, then P; (S, M) at merge
    threadgroup float so[NSG * hd];                // per-simdgroup O partials
    uint abs = p.pos + tgpig.x / p.n_head;
    attnvec_body<hd, NSG>(q, k, v, dst, p, abs, p.kv_len, sq, ssc, so, tgpig, sgitg, tiisg);
}

// Dynamic-pos variant for decode replay: `pos` comes from the bound positions buffer (updated by
// the host every token) instead of the recorded AttnParams, whose baked pos/kv_len are stale by
// the second replay. rows is 1 on this path, so kv_len is exactly pos + 1.
template<uint hd, uint NSG>
kernel void attnvec_dyn_f16kv_t(device const float* q    [[buffer(0)]],
                                device const half*  k    [[buffer(1)]],
                                device const half*  v    [[buffer(2)]],
                                device float*       dst  [[buffer(3)]],
                                device const int*   posb [[buffer(4)]],
                                constant AttnParams& p   [[buffer(5)]],
                                uint3  tgpig [[threadgroup_position_in_grid]],
                                ushort sgitg [[simdgroup_index_in_threadgroup]],
                                ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float sq[hd];
    threadgroup float ssc[NSG * 32];
    threadgroup float so[NSG * hd];
    uint abs = (uint)posb[0];
    attnvec_body<hd, NSG>(q, k, v, dst, p, abs, abs + 1u, sq, ssc, so, tgpig, sgitg, tiisg);
}

typedef decltype(attnvec_f16kv_t<64, 32>) attnvec_t;
template [[host_name("attnvec_f16kv_hd64")]]  kernel attnvec_t attnvec_f16kv_t<64, 32>;
template [[host_name("attnvec_f16kv_hd128")]] kernel attnvec_t attnvec_f16kv_t<128, 32>;
typedef decltype(attnvec_dyn_f16kv_t<64, 32>) attnvec_dyn_t;
template [[host_name("attnvec_dyn_f16kv_hd64")]]  kernel attnvec_dyn_t attnvec_dyn_f16kv_t<64, 32>;
template [[host_name("attnvec_dyn_f16kv_hd128")]] kernel attnvec_dyn_t attnvec_dyn_f16kv_t<128, 32>;
// hd=256 (gemma) drops to NSG=16: the per-simdgroup O partials are NSG*hd*4 bytes — 32 KB at
// NSG=32/hd=256, over the whole threadgroup budget before sq/ssc. 16 simdgroups still cut the
// serial chain 16x vs the plain split kernel; the merge tree just starts one level lower.
template [[host_name("attnvec_f16kv_hd256")]]     kernel attnvec_t     attnvec_f16kv_t<256, 16>;
template [[host_name("attnvec_dyn_f16kv_hd256")]] kernel attnvec_dyn_t attnvec_dyn_f16kv_t<256, 16>;

// ---- MLA attention (DeepSeek V2/V3 absorbed form). One thread per (token, head): load the head's
// q, absorb q_nope via wk_b[h]^T, rope q_pe internally, two-pass SDPA over the f16 KV cache, then
// project the accumulated V through wv_b[h]. The KV cache holds ONE row per token (key_len =
// kv_lora_rank + qk_rope_dim wide, 576 for V3); V is the first kv_lora_rank columns — aliased from
// K, no separate v_cache. The q_pe half of each query head is APPLIED ROPE internally here
// (matching the CPU kernel); the graph passes raw q_pe. Faithful port of the Vulkan mla.comp — the
// Metal f16 cache is `device half*` (one half per element), so unlike mla.comp there is no
// u32-packed-pair bit unpacking: index it directly.
struct MlaParams {
    uint rows;
    uint kv_len;
    uint n_head;
    uint q_head_dim;       // qk_nope_dim + qk_rope_dim
    uint kv_lora_rank;
    uint qk_nope_dim;
    uint qk_rope_dim;
    uint v_head_dim;
    float scale;
    uint pos;              // first token position
    uint mask_type;        // 0 = causal, 1 = sliding window, 2 = canvas
    uint window;
    float theta;
    uint cache_cap_rows;   // ring-buffer row capacity (0 or >= kv_len = full context)
};

kernel void mla_f16kv(device const float* q       [[buffer(0)]],
                      device const half*  k_cache [[buffer(1)]],
                      device const float* wk_b    [[buffer(2)]],
                      device const float* wv_b    [[buffer(3)]],
                      device float*       dst     [[buffer(4)]],
                      constant MlaParams& p       [[buffer(5)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.n_head) return;
    uint tok = gid / p.n_head;
    uint h = gid % p.n_head;
    uint abs_pos = p.pos + tok;
    uint key_len = p.kv_lora_rank + p.qk_rope_dim;

    // ── Load q for this head ──
    uint q_off = gid * p.q_head_dim;
    // Scratch: q_full after absorption + rope
    float q_full[576]; // max key_len (512+64)
    // q_nope → absorb via wk_b[h]^T
    uint wk_off = h * p.kv_lora_rank * p.qk_nope_dim;
    for (uint j = 0u; j < p.kv_lora_rank; j++) {
        float s = 0.0f;
        uint wk_row = wk_off + j * p.qk_nope_dim;
        for (uint i = 0u; i < p.qk_nope_dim; i++) {
            s += wk_b[wk_row + i] * q[q_off + i];
        }
        q_full[j] = s;
    }
    // Rope q_pe into q_full[kv_lora_rank ..]
    uint q_pe_off = q_off + p.qk_nope_dim;
    uint hf = p.qk_rope_dim >> 1u;
    for (uint pp = 0u; pp < hf; pp++) {
        uint i0 = 2u * pp;
        uint i1 = i0 + 1u;
        float ang = float(abs_pos) * pow(p.theta, -2.0 * float(pp) / float(p.qk_rope_dim));
        float s = sin(ang);
        float c = cos(ang);
        float a = q[q_pe_off + i0];
        float b = q[q_pe_off + i1];
        q_full[p.kv_lora_rank + i0] = a * c - b * s;
        q_full[p.kv_lora_rank + i1] = a * s + b * c;
    }

    // ── Mask range ──
    uint lo, hi;
    if (p.mask_type == 0u) {
        // Causal
        lo = 0u;
        hi = min(abs_pos + 1u, p.kv_len);
        if (p.window > 0u && abs_pos + 1u > p.window) {
            lo = abs_pos + 1u - p.window;
        }
    } else if (p.mask_type == 1u) {
        // Sliding window
        lo = (abs_pos + 1u > p.window) ? (abs_pos + 1u - p.window) : 0u;
        hi = abs_pos + 1u;
    } else {
        // Canvas / full
        lo = 0u;
        hi = p.kv_len;
    }
    hi = min(hi, p.kv_len);
    uint n_keys = hi - lo;
    if (n_keys == 0u) {
        // No keys to attend to: zero output.
        uint d_off = gid * p.v_head_dim;
        for (uint d = 0u; d < p.v_head_dim; d++) {
            dst[d_off + d] = 0.0f;
        }
        return;
    }

    // ── Pass 1: compute scores, find max ──
    uint cap = (p.cache_cap_rows > 0u) ? p.cache_cap_rows : p.kv_len;
    float lmax = -3.0e38;
    for (uint jj = 0u; jj < n_keys; jj++) {
        uint jr = (lo + jj) % cap;
        // Dot product: q_full · K[jr] (unscaled — scale applied below, like mla.comp)
        float d = 0.0f;
        for (uint di = 0u; di < key_len; di++) {
            d += q_full[di] * (float)k_cache[jr * key_len + di];
        }
        d *= p.scale;
        lmax = max(lmax, d);
    }

    // ── Pass 2: softmax + V accumulation ──
    float sumw = 0.0f;
    float vacc[512]; // max kv_lora_rank
    for (uint di = 0u; di < p.kv_lora_rank; di++) { vacc[di] = 0.0f; }
    for (uint jj = 0u; jj < n_keys; jj++) {
        uint jr = (lo + jj) % cap;
        float d = 0.0f;
        for (uint di = 0u; di < key_len; di++) {
            d += q_full[di] * (float)k_cache[jr * key_len + di];
        }
        float pr = exp(d * p.scale - lmax);
        sumw += pr;
        // V = first kv_lora_rank columns of K[jr]
        for (uint di = 0u; di < p.kv_lora_rank; di++) {
            vacc[di] += pr * (float)k_cache[jr * key_len + di];
        }
    }
    float inv_sum = 1.0f / max(sumw, 1e-20);

    // ── wv_b[h] applied to V_accum → output ──
    uint d_off = gid * p.v_head_dim;
    uint wv_off = h * p.kv_lora_rank * p.v_head_dim;
    for (uint d = 0u; d < p.v_head_dim; d++) {
        float s = 0.0f;
        uint wv_col = wv_off + d; // wv_b[h][i][d] — column-major over v_head_dim
        for (uint i = 0u; i < p.kv_lora_rank; i++) {
            s += vacc[i] * inv_sum * wv_b[wv_col + i * p.v_head_dim];
        }
        dst[d_off + d] = s;
    }
}
