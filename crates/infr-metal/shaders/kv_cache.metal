// ---- WriteKv:// ---- WriteKv: cast-copy `n` f32 source elems into the bound KV cache at row offset `base`, on the
// GPU so it stays in the batch (no host round-trip that would force a per-layer flush). The (half)
// cast is IEEE round-to-nearest-even — byte-identical to the host `f16::from_f32` reference.
struct WriteKvParams { uint n; uint base; };
kernel void writekv_f16(device const float* src   [[buffer(0)]],
                        device half*        cache [[buffer(1)]],
                        constant WriteKvParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    cache[p.base + gid] = (half)src[gid];
}
kernel void writekv_f32(device const float* src   [[buffer(0)]],
                        device float*       cache [[buffer(1)]],
                        constant WriteKvParams& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    cache[p.base + gid] = src[gid];
}
// Dynamic-pos WriteKv for decode replay: the row offset is pos*row_stride with pos read from the
// bound positions buffer per token (`base` in a recorded params blob would be stale). f16 cache
// only (the replay gate requires it).
struct WriteKvDynParams { uint n; uint row_stride; };
kernel void writekv_dyn_f16(device const float* src   [[buffer(0)]],
                            device half*        cache [[buffer(1)]],
                            device const int*   posb  [[buffer(2)]],
                            constant WriteKvDynParams& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    cache[(uint)posb[0] * p.row_stride + gid] = (half)src[gid];
}

// ---- Q8_0 KV cache (INFR_KV_Q8): 34-byte blocks of 32 codes — half the f16 footprint and
// bandwidth. Quantization matches the CPU reference bit-for-bit: d = amax/127 stored as f16,
// q = rint(x/d) (ties to even, matching Rust's round_ties_even). Rows are 32-aligned (the
// runner gates on it), so a written row never straddles a block.
inline float q8_at(device const uchar* c, ulong e) {
    device const uchar* blk = c + (e >> 5) * 34ul;
    float d = (float)*(device const half*)blk; /* 34*b is even — the f16 d is always aligned */
    return d * (float)(char)blk[2u + (e & 31u)];
}
inline float4 q8_float4(device const uchar* c, ulong e) { /* e % 4 == 0: never straddles */
    device const uchar* blk = c + (e >> 5) * 34ul;
    float d = (float)*(device const half*)blk;
    /* codes start at blk+2 (2-byte aligned): two ushort loads cover the 4 codes */
    device const ushort* q2 = (device const ushort*)(blk + 2u + (e & 31u));
    uint lo = q2[0];
    uint hi = q2[1];
    return d * float4((float)(char)(lo & 0xFFu), (float)(char)(lo >> 8),
                      (float)(char)(hi & 0xFFu), (float)(char)(hi >> 8));
}
kernel void writekv_q8(device const float* src   [[buffer(0)]],
                       device uchar*       cache [[buffer(1)]],
                       constant WriteKvParams& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;   // one thread per 32-elem block; p.base is in elements
    device const float* s = src + gid * 32u;
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) amax = max(amax, fabs(s[i]));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    device uchar* blk = cache + ((ulong)(p.base >> 5) + gid) * 34ul;
    *(device half*)blk = (half)d;
    for (uint i = 0; i < 32u; i++) blk[2u + i] = (uchar)(char)(int)rint(s[i] * id);
}
kernel void writekv_dyn_q8(device const float* src   [[buffer(0)]],
                           device uchar*       cache [[buffer(1)]],
                           device const int*   posb  [[buffer(2)]],
                           constant WriteKvDynParams& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) amax = max(amax, fabs(s[i]));
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    ulong base_blk = (ulong)(uint)posb[0] * (p.row_stride >> 5);
    device uchar* blk = cache + (base_blk + gid) * 34ul;
    *(device half*)blk = (half)d;
    for (uint i = 0; i < 32u; i++) blk[2u + i] = (uchar)(char)(int)rint(s[i] * id);
}

// Scalar attention over a Q8_0 cache — the attention_f16kv shape with dequant-on-read. The
// catch-all q8 route (prefill + odd shapes); the vector kernel below covers rows==1 decode.
kernel void attention_q8kv(device const float* q   [[buffer(0)]],
                           device const uchar* k   [[buffer(1)]],
                           device const uchar* v   [[buffer(2)]],
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
        ulong kb = ((ulong)j * p.n_kv + kvh) * p.head_dim;
        float part = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) part += q[qb + d] * q8_at(k, kb + d);
        float sc = simd_sum(part) * p.scale;
        float mnew = max(m, sc);
        float corr = exp(m - mnew);
        float pw = exp(sc - mnew);
        l = l * corr + pw;
        uint t = 0;
        for (uint d = lane; d < p.head_dim; d += 32u) {
            acc[t] = acc[t] * corr + pw * q8_at(v, kb + d);
            t++;
        }
        m = mnew;
    }
    uint t = 0;
    for (uint d = lane; d < p.head_dim; d += 32u) { dst[qb + d] = acc[t] / l; t++; }
}

// Quantized-KV accessors for the vector flash body below: each struct decodes 4 consecutive
// elements (e % 4 == 0, never straddling a nibble half) of one cache block to f32. All are
// exact ports of the CPU dequant formulas, so the attention math is f32 over exactly-decoded
// values — reassociation-only vs the CPU oracle, independent of the storage format.
struct KVQ8 {
    static float4 at4(device const uchar* c, ulong e) { return q8_float4(c, e); }
};
struct KVQ40 { /* 18 B block: [f16 d][16 B nibbles]; elem j<16 low nibble of byte j, else high */
    static float4 at4(device const uchar* c, ulong e) {
        device const uchar* blk = c + (e >> 5) * 18ul;
        float d = (float)*(device const half*)blk; /* block offset is even: d stays aligned */
        uint j = (uint)(e & 31u);
        /* 4 consecutive elems share the nibble half: 4 bytes via two aligned ushort loads */
        device const ushort* q2 = (device const ushort*)(blk + 2u + (j & 15u));
        uint lo = q2[0], hi = q2[1];
        uint s = (j < 16u) ? 0u : 4u;
        return d * (float4((float)((lo >> s) & 0xFu), (float)((lo >> (8u + s)) & 0xFu),
                           (float)((hi >> s) & 0xFu), (float)((hi >> (8u + s)) & 0xFu)) -
                    8.0f);
    }
};
struct KVIQ4NL { /* q4_0 layout, nibbles index the shared IQ4_NL codebook instead of -8 offset */
    static float4 at4(device const uchar* c, ulong e) {
        device const uchar* blk = c + (e >> 5) * 18ul;
        float d = (float)*(device const half*)blk;
        uint j = (uint)(e & 31u);
        device const ushort* q2 = (device const ushort*)(blk + 2u + (j & 15u));
        uint lo = q2[0], hi = q2[1];
        uint s = (j < 16u) ? 0u : 4u;
        return d * float4(kvalues_iq4nl_f[(lo >> s) & 0xFu], kvalues_iq4nl_f[(lo >> (8u + s)) & 0xFu],
                          kvalues_iq4nl_f[(hi >> s) & 0xFu], kvalues_iq4nl_f[(hi >> (8u + s)) & 0xFu]);
    }
};

// Vector flash attention over a quantized cache — the attnvec structure with dequant-on-read
// (see attnvec_body; kept as a sibling body because the KV accessor type differs). Same
// numeric class: f32 dots over exactly-dequantized values, reassociation only. The accessor
// struct is the only format-specific part (q8/q4_0/iq4_nl share the body verbatim).
template<uint hd, uint NSG, typename KV>
inline void attnvec_q_body(device const float* q,
                            device const uchar* k,
                            device const uchar* v,
                            device float*       dst,
                            constant AttnParams& p,
                            uint abs, uint kvl,
                            threadgroup float* sq,
                            threadgroup float* ssc,
                            threadgroup float* so,
                            uint3  tgpig,
                            ushort sgitg,
                            ushort tiisg) {
    constexpr uint C = 32, NE = 4, NL = 32u / NE;
    constexpr uint hd4 = hd / 4u;
    constexpr uint NI = hd4 / NL;

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
    threadgroup const float4* sq4 = (threadgroup const float4*)sq;

    for (uint ic = sgitg * C; ic <= abs; ic += NSG * C) {
        if (ic + C <= lo) continue;
        {
            float mqk[C / NE];
            for (uint cc = 0; cc < C / NE; cc++) {
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                ulong eb = ((ulong)rc * p.n_kv + kvh) * hd;
                float acc = 0.0f;
                for (uint ii = 0; ii < NI; ii++)
                    acc += dot(KV::at4(k, eb + (ii * NL + tx) * 4u), sq4[ii * NL + tx]);
                acc += simd_shuffle_down(acc, 4);
                acc += simd_shuffle_down(acc, 2);
                acc += simd_shuffle_down(acc, 1);
                mqk[cc] = simd_shuffle(acc, NL * ty);
            }
            ss[NE * tx + ty] = mqk[tx];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            float sv = ss[tiisg] * p.scale;
            uint jkv = ic + tiisg;
            bool valid = (jkv >= lo) && (jkv <= abs);
            float m = M;
            M = simd_max(max(M, valid ? sv : -MAXFLOAT / 2));
            float ms = exp(m - M);
            float vs = valid ? exp(sv - M) : 0.0f;
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) {
                for (uint ii = 0; ii < NI; ii++) so4[ii * NL + tx] *= ms;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
            float4 lov[NI];
            for (uint ii = 0; ii < NI; ii++) lov[ii] = float4(0.0f);
            for (uint cc = 0; cc < C / NE; cc++) {
                uint rc = min(ic + NE * cc + ty, kvl - 1u);
                ulong eb = ((ulong)rc * p.n_kv + kvh) * hd;
                float pw = ss[NE * cc + ty];
                for (uint ii = 0; ii < NI; ii++)
                    lov[ii] += KV::at4(v, eb + (ii * NL + tx) * 4u) * pw;
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

    if (tiisg == 0) { ss[0] = S; ss[1] = M; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

template<uint hd, uint NSG, typename KV>
kernel void attnvec_qkv_t(device const float* q   [[buffer(0)]],
                          device const uchar* k   [[buffer(1)]],
                          device const uchar* v   [[buffer(2)]],
                          device float*       dst [[buffer(3)]],
                          constant AttnParams& p  [[buffer(4)]],
                          uint3  tgpig [[threadgroup_position_in_grid]],
                          ushort sgitg [[simdgroup_index_in_threadgroup]],
                          ushort tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float sq[hd];
    threadgroup float ssc[NSG * 32];
    threadgroup float so[NSG * hd];
    uint abs = p.pos + tgpig.x / p.n_head;
    attnvec_q_body<hd, NSG, KV>(q, k, v, dst, p, abs, p.kv_len, sq, ssc, so, tgpig, sgitg, tiisg);
}

template<uint hd, uint NSG, typename KV>
kernel void attnvec_dyn_qkv_t(device const float* q    [[buffer(0)]],
                              device const uchar* k    [[buffer(1)]],
                              device const uchar* v    [[buffer(2)]],
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
    attnvec_q_body<hd, NSG, KV>(q, k, v, dst, p, abs, abs + 1u, sq, ssc, so, tgpig, sgitg, tiisg);
}

// Cooperative flash attention over a Q8_0 cache — the attnflash2 structure with a cooperative
// dequant-staging stage (the llama.cpp flash_attn_ext quantized-KV branch shape): K/V can't be
// simdgroup_load'ed from q8 blocks, so per 64-position KV block all 128 threads dequantize the
// tile into threadgroup memory — K PRE-TRANSPOSED [hd][64] so the QK fragments load
// non-transposed and conflict-free, V row-major [64][hd] staged into the SAME tile during the
// softmax phase (the K reads are done by then; one extra barrier per block). ~24 KiB threadgroup
// at hd=128 (vs 8 KiB for the f16 kernel — the occupancy cost of in-kernel dequant).
template<uint hd, uint NSG>
kernel void attnflash2_q8kv_t(device const half*  q   [[buffer(0)]],
                              device const uchar* k   [[buffer(1)]],
                              device const uchar* v   [[buffer(2)]],
                              device float*       dst [[buffer(3)]],
                              constant AttnParams& p  [[buffer(4)]],
                              uint3  tgpig [[threadgroup_position_in_grid]],
                              ushort sgitg [[simdgroup_index_in_threadgroup]],
                              ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint QT = 8, C = 64, NQ = QT / NSG, SH = C;
    constexpr uint NBR = hd / 32u;          // q8 blocks per KV row
    threadgroup half  sq[QT * hd];
    threadgroup float so[QT * hd];
    threadgroup float ss[QT * SH];
    threadgroup half  kvt[C * hd];          // K as [hd][C], then V as [C][hd]

    uint ntq = (p.rows + QT - 1u) / QT;
    uint qt = tgpig.x % ntq;
    uint h  = tgpig.x / ntq;
    constexpr uint hd4 = hd / 4u;
    constexpr uint no = hd / (8u * NSG);
    constexpr uint NC = (C / 8u) / NSG;
    uint kvh = h / (p.n_head / p.n_kv);
    uint r0 = qt * QT;
    uint abs0 = p.pos + r0;
    uint abs_max = p.pos + min(p.rows - 1u, r0 + QT - 1u);
    uint lo_min = (p.window > 0u && abs0 + 1u > p.window) ? (abs0 + 1u - p.window) : 0u;
    ulong qstride = (ulong)p.n_head * hd;
    uint tid = (uint)sgitg * 32u + tiisg;

    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        bool live = r0 + j < p.rows;
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
        // stage K [hd][C]: each thread dequantizes whole q8 blocks (clamped rows are masked
        // in the softmax, so their junk never contributes)
        for (uint b = tid; b < C * NBR; b += NSG * 32u) {
            uint rr = b / NBR;
            uint dsub = (b % NBR) * 32u;
            uint rc = min(ic + rr, p.kv_len - 1u);
            ulong eb = ((ulong)rc * p.n_kv + kvh) * hd + dsub;
            device const uchar* blk = k + (eb >> 5) * 34ul;
            float d = (float)*(device const half*)blk;
            for (uint i = 0; i < 32u; i++)
                kvt[(dsub + i) * C + rr] = (half)(d * (float)(char)blk[2u + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup const half* pk = kvt + 8u * sgitg;
            threadgroup float* ps = ss + 8u * sgitg;
            for (uint cc = 0; cc < NC; cc++) {
                simdgroup_float8x8 mqk = simdgroup_float8x8(0.0f);
                if (ic + 8u * (sgitg + cc * NSG) <= abs_max) {
                    for (uint i = 0; i < hd; i += 16u) {
                        simdgroup_half8x8 mq, mk;
                        simdgroup_load(mq, sq + i, hd);
                        simdgroup_load(mk, pk + i * C, C);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        simdgroup_load(mq, sq + i + 8u, hd);
                        simdgroup_load(mk, pk + (i + 8u) * C, C);
                        simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                    }
                }
                simdgroup_store(mqk, ps, SH);
                pk += 8u * NSG;
                ps += 8u * NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // softmax (rows split across simdgroups) + V staging [C][hd] in the same phase —
        // the K reads are complete, the V reads haven't started
        for (uint jj = 0; jj < NQ; jj++) {
            uint j = jj * NSG + sgitg;
            uint absr = abs0 + j;
            uint lor = (p.window > 0u && absr + 1u > p.window) ? (absr + 1u - p.window) : 0u;
            threadgroup float2* ss2 = (threadgroup float2*)(ss + j * SH);
            float2 s2 = ss2[tiisg] * p.scale;
            uint c0 = ic + 2u * tiisg;
            bool v0 = (c0 >= lor) && (c0 <= absr);
            bool v1 = (c0 + 1u >= lor) && (c0 + 1u <= absr);
            float m = M[jj];
            float mnew =
                simd_max(max(m, max(v0 ? s2.x : -MAXFLOAT / 2, v1 ? s2.y : -MAXFLOAT / 2)));
            float ms = exp(m - mnew);
            float pw0 = v0 ? exp(s2.x - mnew) : 0.0f;
            float pw1 = v1 ? exp(s2.y - mnew) : 0.0f;
            S[jj] = S[jj] * ms + simd_sum(pw0 + pw1);
            M[jj] = mnew;
            ss2[tiisg] = float2(pw0, pw1);
            threadgroup float4* so4 = (threadgroup float4*)so + j * hd4;
            for (uint i = tiisg; i < hd4; i += 32u) so4[i] *= ms;
        }
        for (uint b = tid; b < C * NBR; b += NSG * 32u) {
            uint rr = b / NBR;
            uint dsub = (b % NBR) * 32u;
            uint rc = min(ic + rr, p.kv_len - 1u);
            ulong eb = ((ulong)rc * p.n_kv + kvh) * hd + dsub;
            device const uchar* blk = v + (eb >> 5) * 34ul;
            float d = (float)*(device const half*)blk;
            for (uint i = 0; i < 32u; i++)
                kvt[rr * hd + dsub + i] = (half)(d * (float)(char)blk[2u + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 lo[no];
            threadgroup float* sot = so + 8u * sgitg;
            for (uint ii = 0; ii < no; ii++) simdgroup_load(lo[ii], sot + 8u * NSG * ii, hd);
            threadgroup const half* pv = kvt + 8u * sgitg;
            uint nblk = min(C / 8u, (abs_max - ic) / 8u + 1u);
            for (uint cc = 0; cc < nblk; cc++) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8u * cc, SH);
                for (uint ii = 0; ii < no; ii++) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, pv + 8u * NSG * ii, hd);
                    simdgroup_multiply_accumulate(lo[ii], vs, mv, lo[ii]);
                }
                pv += 8u * hd;
            }
            for (uint ii = 0; ii < no; ii++) simdgroup_store(lo[ii], sot + 8u * NSG * ii, hd);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint jj = 0; jj < NQ; jj++) {
        uint j = jj * NSG + sgitg;
        if (r0 + j >= p.rows) continue;
        float sc = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        device float4* out = (device float4*)(dst + ((ulong)(r0 + j)) * qstride + (ulong)h * hd);
        threadgroup const float4* so4 = (threadgroup const float4*)so + j * hd4;
        for (uint i = tiisg; i < hd4; i += 32u) out[i] = so4[i] * sc;
    }
}

typedef decltype(attnflash2_q8kv_t<64, 4>) attnflash2_q8_t;
template [[host_name("attnflash2_q8kv_hd64")]]  kernel attnflash2_q8_t attnflash2_q8kv_t<64, 4>;
template [[host_name("attnflash2_q8kv_hd128")]] kernel attnflash2_q8_t attnflash2_q8kv_t<128, 4>;

typedef decltype(attnvec_qkv_t<64, 32, KVQ8>) attnvec_q_t;
template [[host_name("attnvec_q8kv_hd64")]]  kernel attnvec_q_t attnvec_qkv_t<64, 32, KVQ8>;
template [[host_name("attnvec_q8kv_hd128")]] kernel attnvec_q_t attnvec_qkv_t<128, 32, KVQ8>;
typedef decltype(attnvec_dyn_qkv_t<64, 32, KVQ8>) attnvec_dyn_q_t;
template [[host_name("attnvec_dyn_q8kv_hd64")]]  kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<64, 32, KVQ8>;
template [[host_name("attnvec_dyn_q8kv_hd128")]] kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<128, 32, KVQ8>;
// hd=256 at NSG=16 — same threadgroup-budget math as the f16 vec kernel. (No q8 flash hd256:
// its dequant-staging tile alone is C*hd = 32 KiB half — needs a C=32 variant first.)
template [[host_name("attnvec_q8kv_hd256")]]     kernel attnvec_q_t     attnvec_qkv_t<256, 16, KVQ8>;
template [[host_name("attnvec_dyn_q8kv_hd256")]] kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<256, 16, KVQ8>;

// Native-read q4_0 / iq4_nl decode: the same vector flash over the compact 18 B blocks — no
// prepass scratch, no whole-prefix re-dequant per token, and f32 accumulation over exactly
// decoded values (the prepass path rounds the scratch to f16; at 4 bits on BOTH sides that
// compounding measurably costs long-context recall). Prefill keeps the dequant→f16 prepass.
template [[host_name("attnvec_q4_0kv_hd64")]]  kernel attnvec_q_t attnvec_qkv_t<64, 32, KVQ40>;
template [[host_name("attnvec_q4_0kv_hd128")]] kernel attnvec_q_t attnvec_qkv_t<128, 32, KVQ40>;
template [[host_name("attnvec_q4_0kv_hd256")]] kernel attnvec_q_t attnvec_qkv_t<256, 16, KVQ40>;
template [[host_name("attnvec_dyn_q4_0kv_hd64")]]  kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<64, 32, KVQ40>;
template [[host_name("attnvec_dyn_q4_0kv_hd128")]] kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<128, 32, KVQ40>;
template [[host_name("attnvec_dyn_q4_0kv_hd256")]] kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<256, 16, KVQ40>;
template [[host_name("attnvec_iq4nlkv_hd64")]]  kernel attnvec_q_t attnvec_qkv_t<64, 32, KVIQ4NL>;
template [[host_name("attnvec_iq4nlkv_hd128")]] kernel attnvec_q_t attnvec_qkv_t<128, 32, KVIQ4NL>;
template [[host_name("attnvec_iq4nlkv_hd256")]] kernel attnvec_q_t attnvec_qkv_t<256, 16, KVIQ4NL>;
template [[host_name("attnvec_dyn_iq4nlkv_hd64")]]  kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<64, 32, KVIQ4NL>;
template [[host_name("attnvec_dyn_iq4nlkv_hd128")]] kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<128, 32, KVIQ4NL>;
template [[host_name("attnvec_dyn_iq4nlkv_hd256")]] kernel attnvec_dyn_q_t attnvec_dyn_qkv_t<256, 16, KVIQ4NL>;

// ============================================================================================
// Decoupled KV-cache quant/dequant (mainline low-bit blocks q4_0/q4_1/q5_0/q5_1/iq4_nl, dense
// bf16, and TurboQuant turbo2/3/4). These follow the Vulkan "dequant -> f16 prepass" model:
// WriteKv quantizes the f32 activation into the compact cache; Attention expands the quantized
// prefix into a transient f16 scratch, then the existing f16 attention kernels read the scratch.
// Every quantize/dequant here is a bit-for-bit port of the CPU reference (crates/infr-cpu:
// kvquant.rs block quants, turbo.rs TurboQuant) so the parity tests hold against the CPU oracle.
// One thread per 32-elem block for the block quants (n/32 threads), per 128-elem block for turbo
// (n/128), per element for bf16 and the dequant expansions (dequant turbo is per 128-block).
// `p.base` is in ELEMENTS (row-aligned to the block size by the runner).
// ============================================================================================

// f16 read/write at a byte offset into a uchar cache (little-endian, IEEE half).
inline float kv_rf16(device const uchar* c, uint bo) {
    ushort h = (ushort)c[bo] | ((ushort)c[bo + 1u] << 8);
    return (float)as_type<half>(h);
}
inline void kv_wf16(device uchar* c, uint bo, float x) {
    ushort h = as_type<ushort>((half)x);
    c[bo] = (uchar)(h & 0xFFu);
    c[bo + 1u] = (uchar)(h >> 8);
}

// ---- q4_0 (18 B): d = max/-8, q = clamp(x/d + 8.5, 0, 15), 4-bit low/high halves.
kernel void writekv_q4_0(device const float* src [[buffer(0)]],
                         device uchar* cache [[buffer(1)]],
                         constant WriteKvParams& p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float vmax = -1e30f, vmin = 1e30f;
    for (uint j = 0; j < 32u; j++) { vmax = max(vmax, s[j]); vmin = min(vmin, s[j]); }
    float mx = (fabs(vmax) >= fabs(vmin)) ? vmax : vmin;
    float d = mx / -8.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint bo = (p.base / 32u + gid) * 18u;
    kv_wf16(cache, bo, d);
    for (uint j = 0; j < 16u; j++) {
        int xi0 = clamp(int(s[j] * id + 8.5f), 0, 15);
        int xi1 = clamp(int(s[j + 16u] * id + 8.5f), 0, 15);
        cache[bo + 2u + j] = (uchar)(xi0 | (xi1 << 4));
    }
}
kernel void dequant_q4_0_f16(device const uchar* cache [[buffer(0)]],
                             device half* dst [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint j = i % 32u; uint bd = (i / 32u) * 18u;
    float d = kv_rf16(cache, bd);
    uint nib = (j < 16u) ? (cache[bd + 2u + j] & 0xFu) : (cache[bd + 2u + j - 16u] >> 4u);
    dst[i] = (half)(d * (float(nib) - 8.0f));
}

// ---- q4_1 (20 B): asymmetric d = (max-min)/15, q = clamp((x-min)/d + 0.5, 0, 15), stores min.
kernel void writekv_q4_1(device const float* src [[buffer(0)]],
                         device uchar* cache [[buffer(1)]],
                         constant WriteKvParams& p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float vmin = 1e30f, vmax = -1e30f;
    for (uint j = 0; j < 32u; j++) { vmin = min(vmin, s[j]); vmax = max(vmax, s[j]); }
    float d = (vmax - vmin) / 15.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint bo = (p.base / 32u + gid) * 20u;
    kv_wf16(cache, bo, d);
    kv_wf16(cache, bo + 2u, vmin);
    for (uint j = 0; j < 16u; j++) {
        int xi0 = clamp(int((s[j] - vmin) * id + 0.5f), 0, 15);
        int xi1 = clamp(int((s[j + 16u] - vmin) * id + 0.5f), 0, 15);
        cache[bo + 4u + j] = (uchar)(xi0 | (xi1 << 4));
    }
}
kernel void dequant_q4_1_f16(device const uchar* cache [[buffer(0)]],
                             device half* dst [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint j = i % 32u; uint bd = (i / 32u) * 20u;
    float d = kv_rf16(cache, bd); float m = kv_rf16(cache, bd + 2u);
    uint nib = (j < 16u) ? (cache[bd + 4u + j] & 0xFu) : (cache[bd + 4u + j - 16u] >> 4u);
    dst[i] = (half)(d * float(nib) + m);
}

// ---- q5_0 (22 B): d = max/-16, 5-bit — low nibble in qs, 5th bit packed into the qh u32.
kernel void writekv_q5_0(device const float* src [[buffer(0)]],
                         device uchar* cache [[buffer(1)]],
                         constant WriteKvParams& p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float vmax = -1e30f, vmin = 1e30f;
    for (uint j = 0; j < 32u; j++) { vmax = max(vmax, s[j]); vmin = min(vmin, s[j]); }
    float mx = (fabs(vmax) >= fabs(vmin)) ? vmax : vmin;
    float d = mx / -16.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint bo = (p.base / 32u + gid) * 22u;
    kv_wf16(cache, bo, d);
    uint qh = 0u;
    for (uint j = 0; j < 16u; j++) {
        int xi0 = clamp(int(s[j] * id + 16.5f), 0, 31);
        int xi1 = clamp(int(s[j + 16u] * id + 16.5f), 0, 31);
        cache[bo + 6u + j] = (uchar)((xi0 & 0xF) | ((xi1 & 0xF) << 4));
        qh |= (uint(xi0 & 0x10) >> 4) << j;
        qh |= (uint(xi1 & 0x10) >> 4) << (j + 16u);
    }
    cache[bo + 2u] = (uchar)(qh & 0xFFu);
    cache[bo + 3u] = (uchar)((qh >> 8) & 0xFFu);
    cache[bo + 4u] = (uchar)((qh >> 16) & 0xFFu);
    cache[bo + 5u] = (uchar)((qh >> 24) & 0xFFu);
}
kernel void dequant_q5_0_f16(device const uchar* cache [[buffer(0)]],
                             device half* dst [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint j = i % 32u; uint bd = (i / 32u) * 22u;
    float d = kv_rf16(cache, bd);
    uint qh = (uint)cache[bd + 2u] | ((uint)cache[bd + 3u] << 8) | ((uint)cache[bd + 4u] << 16) | ((uint)cache[bd + 5u] << 24);
    uint val;
    if (j < 16u) { uint xh0 = ((qh >> j) << 4u) & 0x10u; val = (cache[bd + 6u + j] & 0xFu) | xh0; }
    else { uint jj = j - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (cache[bd + 6u + jj] >> 4u) | xh1; }
    dst[i] = (half)(d * (float(val) - 16.0f));
}

// ---- q5_1 (24 B): asymmetric 5-bit — d = (max-min)/31, low nibble in qs, 5th bit in qh, min.
kernel void writekv_q5_1(device const float* src [[buffer(0)]],
                         device uchar* cache [[buffer(1)]],
                         constant WriteKvParams& p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float vmin = 1e30f, vmax = -1e30f;
    for (uint j = 0; j < 32u; j++) { vmin = min(vmin, s[j]); vmax = max(vmax, s[j]); }
    float d = (vmax - vmin) / 31.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint bo = (p.base / 32u + gid) * 24u;
    kv_wf16(cache, bo, d);
    kv_wf16(cache, bo + 2u, vmin);
    uint qh = 0u;
    for (uint j = 0; j < 16u; j++) {
        int xi0 = int((s[j] - vmin) * id + 0.5f);
        int xi1 = int((s[j + 16u] - vmin) * id + 0.5f);
        cache[bo + 8u + j] = (uchar)((xi0 & 0xF) | ((xi1 & 0xF) << 4));
        qh |= (uint(xi0 & 0x10) >> 4) << j;
        qh |= (uint(xi1 & 0x10) >> 4) << (j + 16u);
    }
    cache[bo + 4u] = (uchar)(qh & 0xFFu);
    cache[bo + 5u] = (uchar)((qh >> 8) & 0xFFu);
    cache[bo + 6u] = (uchar)((qh >> 16) & 0xFFu);
    cache[bo + 7u] = (uchar)((qh >> 24) & 0xFFu);
}
kernel void dequant_q5_1_f16(device const uchar* cache [[buffer(0)]],
                             device half* dst [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint j = i % 32u; uint bd = (i / 32u) * 24u;
    float d = kv_rf16(cache, bd); float m = kv_rf16(cache, bd + 2u);
    uint qh = (uint)cache[bd + 4u] | ((uint)cache[bd + 5u] << 8) | ((uint)cache[bd + 6u] << 16) | ((uint)cache[bd + 7u] << 24);
    uint val;
    if (j < 16u) { uint xh0 = ((qh >> j) << 4u) & 0x10u; val = (cache[bd + 8u + j] & 0xFu) | xh0; }
    else { uint jj = j - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (cache[bd + 8u + jj] >> 4u) | xh1; }
    dst[i] = (half)(d * float(val) + m);
}

// ---- iq4_nl (18 B): non-linear 16-entry codebook. d = max/values[0], least-squares refine
//      d = Sum(w q x)/Sum(w q^2) (w = x^2, no imatrix -> ntry=-1 path). Indices low/high halves.
constant int KV_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
inline int kv_best_index(float x) {
    if (x <= (float)KV_IQ4NL[0]) return 0;
    if (x >= (float)KV_IQ4NL[15]) return 15;
    int ml = 0, mu = 15;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < (float)KV_IQ4NL[mav]) mu = mav; else ml = mav;
    }
    return ((x - (float)KV_IQ4NL[mu - 1]) < ((float)KV_IQ4NL[mu] - x)) ? (mu - 1) : mu;
}
kernel void writekv_iq4_nl(device const float* src [[buffer(0)]],
                           device uchar* cache [[buffer(1)]],
                           constant WriteKvParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float amax = 0.0f, mxv = 0.0f;
    for (uint j = 0; j < 32u; j++) { float a = fabs(s[j]); if (a > amax) { amax = a; mxv = s[j]; } }
    int L[32];
    float d = 0.0f;
    if (amax >= 1e-15f) {
        d = mxv / (float)KV_IQ4NL[0];
        float id = 1.0f / d;
        float sumqx = 0.0f, sumq2 = 0.0f;
        for (uint j = 0; j < 32u; j++) {
            int li = kv_best_index(id * s[j]);
            L[j] = li;
            float q = (float)KV_IQ4NL[li];
            float w = s[j] * s[j];
            sumqx += w * q * s[j];
            sumq2 += w * q * q;
        }
        d = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
    } else {
        for (uint j = 0; j < 32u; j++) L[j] = 0;
    }
    uint bo = (p.base / 32u + gid) * 18u;
    kv_wf16(cache, bo, d);
    for (uint j = 0; j < 16u; j++) cache[bo + 2u + j] = (uchar)(L[j] | (L[j + 16u] << 4));
}
kernel void dequant_iq4_nl_f16(device const uchar* cache [[buffer(0)]],
                               device half* dst [[buffer(1)]],
                               constant uint& n [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint j = i % 32u; uint bd = (i / 32u) * 18u;
    float d = kv_rf16(cache, bd);
    uint nib = (j < 16u) ? (cache[bd + 2u + j] & 0xFu) : (cache[bd + 2u + j - 16u] >> 4u);
    dst[i] = (half)(d * (float)KV_IQ4NL[nib]);
}

// Replay-tape (dynamic-pos) variants of the q4_0 / iq4_nl quantizing writes: the row offset
// comes from the bound positions buffer (the baked base is the recorded token's only) — the
// same contract as writekv_dyn_q8. Quantization formulas are byte-identical to the static
// kernels above.
kernel void writekv_dyn_q4_0(device const float* src   [[buffer(0)]],
                             device uchar*       cache [[buffer(1)]],
                             device const int*   posb  [[buffer(2)]],
                             constant WriteKvDynParams& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float vmax = -1e30f, vmin = 1e30f;
    for (uint j = 0; j < 32u; j++) { vmax = max(vmax, s[j]); vmin = min(vmin, s[j]); }
    float mx = (fabs(vmax) >= fabs(vmin)) ? vmax : vmin;
    float d = mx / -8.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    ulong base_blk = (ulong)(uint)posb[0] * (p.row_stride >> 5);
    uint bo = (uint)((base_blk + gid) * 18ul);
    kv_wf16(cache, bo, d);
    for (uint j = 0; j < 16u; j++) {
        int xi0 = clamp(int(s[j] * id + 8.5f), 0, 15);
        int xi1 = clamp(int(s[j + 16u] * id + 8.5f), 0, 15);
        cache[bo + 2u + j] = (uchar)(xi0 | (xi1 << 4));
    }
}

kernel void writekv_dyn_iq4_nl(device const float* src   [[buffer(0)]],
                               device uchar*       cache [[buffer(1)]],
                               device const int*   posb  [[buffer(2)]],
                               constant WriteKvDynParams& p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 32u) return;
    device const float* s = src + gid * 32u;
    float amax = 0.0f, mxv = 0.0f;
    for (uint j = 0; j < 32u; j++) { float a = fabs(s[j]); if (a > amax) { amax = a; mxv = s[j]; } }
    int L[32];
    float d = 0.0f;
    if (amax >= 1e-15f) {
        d = mxv / (float)KV_IQ4NL[0];
        float id = 1.0f / d;
        float sumqx = 0.0f, sumq2 = 0.0f;
        for (uint j = 0; j < 32u; j++) {
            int li = kv_best_index(id * s[j]);
            L[j] = li;
            float q = (float)KV_IQ4NL[li];
            float w = s[j] * s[j];
            sumqx += w * q * s[j];
            sumq2 += w * q * q;
        }
        d = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
    } else {
        for (uint j = 0; j < 32u; j++) L[j] = 0;
    }
    ulong base_blk = (ulong)(uint)posb[0] * (p.row_stride >> 5);
    uint bo = (uint)((base_blk + gid) * 18ul);
    kv_wf16(cache, bo, d);
    for (uint j = 0; j < 16u; j++) cache[bo + 2u + j] = (uchar)(L[j] | (L[j + 16u] << 4));
}

// ---- bf16 (2 B): top 16 bits of the f32 (round-to-nearest-even); dequant is a lossless <<16.
kernel void writekv_bf16(device const float* src [[buffer(0)]],
                         device uchar* cache [[buffer(1)]],
                         constant WriteKvParams& p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    uint b = as_type<uint>(src[gid]);
    ushort bf = (ushort)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    uint bo = (p.base + gid) * 2u;
    cache[bo] = (uchar)(bf & 0xFFu);
    cache[bo + 1u] = (uchar)(bf >> 8);
}
kernel void dequant_bf16_f16(device const uchar* cache [[buffer(0)]],
                             device half* dst [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    ushort b = (ushort)cache[i * 2u] | ((ushort)cache[i * 2u + 1u] << 8);
    dst[i] = (half)as_type<float>((uint)b << 16);
}

// ---- TurboQuant (turbo2/3/4): WHT-rotate 128-elem block -> nearest optimal centroid -> pack;
//      store norm = grp_norm/||recon||. Dequant unpacks centroid*norm (rotated domain) then the
//      inverse WHT to recover the original domain. Ports crates/infr-cpu/src/turbo.rs verbatim.
constant float KV_INV_SQRT = 0.088388350f;
constant char KV_S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
constant char KV_S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};
constant float KV_C2[4] = {-0.133462f, -0.039994f, 0.039994f, 0.133462f};
constant float KV_M2[3] = {-0.086728f, 0.0f, 0.086728f};
constant float KV_C3[8] = {-0.190207f, -0.118786f, -0.066822f, -0.021663f, 0.021663f, 0.066822f, 0.118786f, 0.190207f};
constant float KV_M3[7] = {-0.154496f, -0.092804f, -0.044243f, 0.0f, 0.044243f, 0.092804f, 0.154496f};
constant float KV_C4[16] = {-0.241529f,-0.182877f,-0.143016f,-0.111036f,-0.083292f,-0.058050f,-0.034299f,-0.011349f,0.011349f,0.034299f,0.058050f,0.083292f,0.111036f,0.143016f,0.182877f,0.241529f};
constant float KV_M4[15] = {-0.212203f,-0.162947f,-0.127026f,-0.097164f,-0.070671f,-0.046174f,-0.022824f,0.0f,0.022824f,0.046174f,0.070671f,0.097164f,0.127026f,0.162947f,0.212203f};

inline int kv_nearest(constant float* mid, int nmid, float v) {
    for (int i = 0; i < nmid; i++) if (v < mid[i]) return i;
    return nmid;
}
inline void kv_fwht(thread float* x) {
    for (int i = 0; i < 128; i++) x[i] *= (float)KV_S1[i];
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) { float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
    for (int i = 0; i < 128; i++) x[i] *= KV_INV_SQRT * (float)KV_S2[i];
}
inline void kv_fwht_inv(thread float* x) {
    for (int i = 0; i < 128; i++) x[i] *= (float)KV_S2[i];
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) { float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
    for (int i = 0; i < 128; i++) x[i] *= KV_INV_SQRT * (float)KV_S1[i];
}

kernel void writekv_turbo2(device const float* src [[buffer(0)]],
                           device uchar* cache [[buffer(1)]],
                           constant WriteKvParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 128u) return;
    float x[128]; float nsq = 0.0f;
    for (uint j = 0; j < 128u; j++) { x[j] = src[gid * 128u + j]; nsq += x[j] * x[j]; }
    float gn = sqrt(nsq); float inv = gn > 1e-10f ? 1.0f / gn : 0.0f;
    for (uint j = 0; j < 128u; j++) x[j] *= inv;
    kv_fwht(x);
    uint bo = (p.base / 128u + gid) * 34u;
    for (uint b = 2u; b < 34u; b++) cache[bo + b] = 0;
    float rsq = 0.0f;
    for (uint j = 0; j < 128u; j++) {
        int idx = kv_nearest(KV_M2, 3, x[j]);
        rsq += KV_C2[idx] * KV_C2[idx];
        cache[bo + 2u + j / 4u] |= (uchar)(idx << ((j % 4u) * 2u));
    }
    float rn = sqrt(rsq); float corr = rn > 1e-10f ? gn / rn : gn;
    kv_wf16(cache, bo, corr);
}
kernel void dequant_turbo2_f16(device const uchar* cache [[buffer(0)]],
                               device half* dst [[buffer(1)]],
                               constant uint& n [[buffer(2)]],
                               uint blk [[thread_position_in_grid]]) {
    if (blk * 128u >= n) return;
    uint bo = blk * 34u;
    float norm = kv_rf16(cache, bo);
    float v[128];
    for (uint j = 0; j < 128u; j++) {
        int idx = (cache[bo + 2u + j / 4u] >> ((j % 4u) * 2u)) & 0x3;
        v[j] = KV_C2[idx] * norm;
    }
    kv_fwht_inv(v);
    for (uint j = 0; j < 128u; j++) dst[blk * 128u + j] = (half)v[j];
}

kernel void writekv_turbo3(device const float* src [[buffer(0)]],
                           device uchar* cache [[buffer(1)]],
                           constant WriteKvParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 128u) return;
    float x[128]; float nsq = 0.0f;
    for (uint j = 0; j < 128u; j++) { x[j] = src[gid * 128u + j]; nsq += x[j] * x[j]; }
    float gn = sqrt(nsq); float inv = gn > 1e-10f ? 1.0f / gn : 0.0f;
    for (uint j = 0; j < 128u; j++) x[j] *= inv;
    kv_fwht(x);
    uint bo = (p.base / 128u + gid) * 50u;
    for (uint b = 2u; b < 50u; b++) cache[bo + b] = 0;
    float rsq = 0.0f;
    for (uint j = 0; j < 128u; j++) {
        int idx = kv_nearest(KV_M3, 7, x[j]);
        rsq += KV_C3[idx] * KV_C3[idx];
        cache[bo + 2u + j / 4u] |= (uchar)((idx & 0x3) << ((j % 4u) * 2u));
        if ((idx & 0x4) != 0) cache[bo + 34u + j / 8u] |= (uchar)(1u << (j % 8u));
    }
    float rn = sqrt(rsq); float corr = rn > 1e-10f ? gn / rn : gn;
    kv_wf16(cache, bo, corr);
}
kernel void dequant_turbo3_f16(device const uchar* cache [[buffer(0)]],
                               device half* dst [[buffer(1)]],
                               constant uint& n [[buffer(2)]],
                               uint blk [[thread_position_in_grid]]) {
    if (blk * 128u >= n) return;
    uint bo = blk * 50u;
    float norm = kv_rf16(cache, bo);
    float v[128];
    for (uint j = 0; j < 128u; j++) {
        uint low2 = (cache[bo + 2u + j / 4u] >> ((j % 4u) * 2u)) & 0x3u;
        uint hi1 = (cache[bo + 34u + j / 8u] >> (j % 8u)) & 0x1u;
        int idx = int(low2 | (hi1 << 2));
        v[j] = KV_C3[idx] * norm;
    }
    kv_fwht_inv(v);
    for (uint j = 0; j < 128u; j++) dst[blk * 128u + j] = (half)v[j];
}

kernel void writekv_turbo4(device const float* src [[buffer(0)]],
                           device uchar* cache [[buffer(1)]],
                           constant WriteKvParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n / 128u) return;
    float x[128]; float nsq = 0.0f;
    for (uint j = 0; j < 128u; j++) { x[j] = src[gid * 128u + j]; nsq += x[j] * x[j]; }
    float gn = sqrt(nsq); float inv = gn > 1e-10f ? 1.0f / gn : 0.0f;
    for (uint j = 0; j < 128u; j++) x[j] *= inv;
    kv_fwht(x);
    uint bo = (p.base / 128u + gid) * 66u;
    for (uint b = 2u; b < 66u; b++) cache[bo + b] = 0;
    float rsq = 0.0f;
    for (uint j = 0; j < 128u; j++) {
        int idx = kv_nearest(KV_M4, 15, x[j]);
        rsq += KV_C4[idx] * KV_C4[idx];
        cache[bo + 2u + j / 2u] |= (uchar)(idx << ((j % 2u) * 4u));
    }
    float rn = sqrt(rsq); float corr = rn > 1e-10f ? gn / rn : gn;
    kv_wf16(cache, bo, corr);
}
kernel void dequant_turbo4_f16(device const uchar* cache [[buffer(0)]],
                               device half* dst [[buffer(1)]],
                               constant uint& n [[buffer(2)]],
                               uint blk [[thread_position_in_grid]]) {
    if (blk * 128u >= n) return;
    uint bo = blk * 66u;
    float norm = kv_rf16(cache, bo);
    float v[128];
    for (uint j = 0; j < 128u; j++) {
        int idx = (cache[bo + 2u + j / 2u] >> ((j % 2u) * 4u)) & 0xF;
        v[j] = KV_C4[idx] * norm;
    }
    kv_fwht_inv(v);
    for (uint j = 0; j < 128u; j++) dst[blk * 128u + j] = (half)v[j];
}
