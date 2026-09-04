// ---- Linear: dst[m, out_f] = x[m, in_f] · Wᵀ, W row-major [out_f, in_f], pre-dequantized to f32.
// One SIMD group (32 lanes) per output element: lanes stride the weight row contiguously — so
// consecutive lanes read consecutive weights (coalesced, full memory bandwidth, vs the strided
// one-thread-per-output naive matvec) — then `simd_sum` reduces. The lane-interleaved partial sums
// change the f32 summation order (still within parity tolerance; Linear was never bit-identical).
struct LinearParams { uint m; uint in_f; uint out_f; };
kernel void linear_f32(device const float* x   [[buffer(0)]],
                       device const float* w   [[buffer(1)]],
                       device float*       dst [[buffer(2)]],
                       constant LinearParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;                       // one simdgroup per (row, output) pair
    if (sg >= p.m * p.out_f) return;
    uint r = sg / p.out_f;
    uint o = sg % p.out_f;
    device const float* xr = x + (ulong)r * p.in_f;
    device const float* wo = w + (ulong)o * p.in_f;
    float acc = 0.0f;
    for (uint i = lane; i < p.in_f; i += 32u) acc += xr[i] * wo[i];
    acc = simd_sum(acc);
    if (lane == 0u) dst[sg] = acc;
}

// Native f16-weight twin of linear_f32. Activations and accumulation stay f32; only the bound
// weight stream is half-width, matching the GGUF value exactly without a host f16->f32 cache.
kernel void linear_f16(device const float* x   [[buffer(0)]],
                       device const half*  w   [[buffer(1)]],
                       device float*       dst [[buffer(2)]],
                       constant LinearParams& p [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.m * p.out_f) return;
    uint r = sg / p.out_f;
    uint o = sg % p.out_f;
    device const float* xr = x + (ulong)r * p.in_f;
    device const half* wo = w + (ulong)o * p.in_f;
    float acc = 0.0f;
    for (uint i = lane; i < p.in_f; i += 32u) acc += xr[i] * (float)wo[i];
    acc = simd_sum(acc);
    if (lane == 0u) dst[sg] = acc;
}

// Native bf16-weight GEMV without requiring Metal bfloat support: GGUF BF16 stores the top 16
// bits of each f32, so widening is the exact integer shift used by the CPU and gather kernels.
kernel void linear_bf16(device const float*  x   [[buffer(0)]],
                        device const ushort* w   [[buffer(1)]],
                        device float*        dst [[buffer(2)]],
                        constant LinearParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    if (sg >= p.m * p.out_f) return;
    uint r = sg / p.out_f;
    uint o = sg % p.out_f;
    device const float* xr = x + (ulong)r * p.in_f;
    device const ushort* wo = w + (ulong)o * p.in_f;
    float acc = 0.0f;
    for (uint i = lane; i < p.in_f; i += 32u) {
        acc += xr[i] * as_type<float>((uint)wo[i] << 16u);
    }
    acc = simd_sum(acc);
    if (lane == 0u) dst[sg] = acc;
}

// ---- Linear over a NATIVE quantized weight. Two on-device forms:
//
// * FACTORED (`dequant_factored`): bit-packed 4/6/8-bit codes + one (sc, m) i16 pair per
//   16-element block + one (d, dmin) f16 pair per 2^dshift elements — for the affine formats
//   without a fast native decoder (legacy quants, Q2_K/Q3_K/Q5_K).
// * NATIVE GGUF blocks for Q4_K (144 B / 256 elems, ~4.5 bpw) and Q6_K (210 B, ~6.6 bpw): the
//   kernel decodes the raw block bytes — the bound weight buffer is used as-is, no host repack,
//   no extra residency, and the weight stream shrinks to the format's true size (the factored
//   form paid ~6.1/8.1 bpw). Decode GEMV is bound on this stream, so bits are throughput.
//
// Every decode reproduces the host dequant reference bit-for-bit: the same f32 products
// (f32(d)·f32(int)) and the same `scale*code + min` operation order.
//
// One DEC16_* macro per format decodes the 16-element block with global index `bi` into wk[16]
// (ambient: `codes`, `scm`, `dd`, `p`, `bi`). Three kernel shapes instantiate each format:
// GEMV (one simdgroup per output, decode), RT (row-tiled, small m), MM (simdgroup_matrix GEMM,
// prefill).
struct QLinParams { uint m; uint in_f; uint out_f; uint dshift; };

// Native f16 weights for the cooperative multi-row tile. `codes` is the raw half buffer;
// scm/dd/dshift are unused so this decoder can share the quantized CMM template unchanged.
#define DEC16_F16(wk)                                                                             \
    device const half* hp = (device const half*)codes + (ulong)bi * 16ul;                         \
    _Pragma("clang loop unroll(full)")                                                            \
    for (uint i = 0; i < 16u; i++) wk[i] = (float)hp[i];

#define DEC16_BF16(wk)                                                                            \
    device const ushort* bp = (device const ushort*)codes + (ulong)bi * 16ul;                     \
    _Pragma("clang loop unroll(full)")                                                            \
    for (uint i = 0; i < 16u; i++) wk[i] = as_type<float>((uint)bp[i] << 16u);

#define DEC16_F32(wk)                                                                             \
    device const float* fp = (device const float*)codes + (ulong)bi * 16ul;                       \
    _Pragma("clang loop unroll(full)")                                                            \
    for (uint i = 0; i < 16u; i++) wk[i] = fp[i];

// factored, 4-bit codes: one uint2 = 8 bytes = one 16-element block, code k at bits 4k.
#define DEC16_K4(wk)                                                                              \
    short2 s = ((device const short2*)scm)[bi];                                                   \
    half2 dv = ((device const half2*)dd)[((ulong)bi << 4) >> p.dshift];                           \
    float scale = (float)dv.x * (float)s.x;                                                       \
    float mn = (float)dv.y * (float)s.y;                                                          \
    uint2 cw = ((device const uint2*)codes)[bi];                                                  \
    for (uint k = 0; k < 8u; k++) {                                                               \
        wk[k]      = scale * (float)((cw.x >> (4u * k)) & 15u) + mn;                              \
        wk[k + 8u] = scale * (float)((cw.y >> (4u * k)) & 15u) + mn;                              \
    }

// factored, 6-bit codes: three uints = 12 bytes = one 16-element block, code k at bits 6k.
#define DEC16_K6(wk)                                                                              \
    short2 s = ((device const short2*)scm)[bi];                                                   \
    half2 dv = ((device const half2*)dd)[((ulong)bi << 4) >> p.dshift];                           \
    float scale = (float)dv.x * (float)s.x;                                                       \
    float mn = (float)dv.y * (float)s.y;                                                         \
    device const uint* cp = (device const uint*)codes + bi * 3ul;                                 \
    uint u0 = cp[0], u1 = cp[1], u2 = cp[2];                                                      \
    wk[0]  = scale * (float)(u0 & 63u) + mn;                                                      \
    wk[1]  = scale * (float)((u0 >> 6) & 63u) + mn;                                               \
    wk[2]  = scale * (float)((u0 >> 12) & 63u) + mn;                                              \
    wk[3]  = scale * (float)((u0 >> 18) & 63u) + mn;                                              \
    wk[4]  = scale * (float)((u0 >> 24) & 63u) + mn;                                              \
    wk[5]  = scale * (float)(((u0 >> 30) | (u1 << 2)) & 63u) + mn;                                \
    wk[6]  = scale * (float)((u1 >> 4) & 63u) + mn;                                               \
    wk[7]  = scale * (float)((u1 >> 10) & 63u) + mn;                                              \
    wk[8]  = scale * (float)((u1 >> 16) & 63u) + mn;                                              \
    wk[9]  = scale * (float)((u1 >> 22) & 63u) + mn;                                              \
    wk[10] = scale * (float)(((u1 >> 28) | (u2 << 4)) & 63u) + mn;                                \
    wk[11] = scale * (float)((u2 >> 2) & 63u) + mn;                                               \
    wk[12] = scale * (float)((u2 >> 8) & 63u) + mn;                                               \
    wk[13] = scale * (float)((u2 >> 14) & 63u) + mn;                                              \
    wk[14] = scale * (float)((u2 >> 20) & 63u) + mn;                                              \
    wk[15] = scale * (float)((u2 >> 26) & 63u) + mn;

// factored, 8-bit codes: one uchar16 = one 16-element block.
#define DEC16_K8(wk)                                                                              \
    short2 s = ((device const short2*)scm)[bi];                                                   \
    half2 dv = ((device const half2*)dd)[((ulong)bi << 4) >> p.dshift];                           \
    float scale = (float)dv.x * (float)s.x;                                                       \
    float mn = (float)dv.y * (float)s.y;                                                         \
    device const uchar4* cp = (device const uchar4*)codes + bi * 4ul;                             \
    for (uint q = 0; q < 4u; q++) {                                                               \
        uchar4 cq = cp[q];                                                                        \
        wk[q * 4u]      = scale * (float)cq.x + mn;                                               \
        wk[q * 4u + 1u] = scale * (float)cq.y + mn;                                               \
        wk[q * 4u + 2u] = scale * (float)cq.z + mn;                                               \
        wk[q * 4u + 3u] = scale * (float)cq.w + mn;                                               \
    }

// Q4_K/Q5_K blocks and resident weight bases are 16-byte aligned. Cooperative prefill benefits
// from one header quad; scalar decode keeps the narrow loads to avoid the measured tg regression.
inline uint4 load_k4_header(device const uchar* blk) {
    return *(device const uint4*)blk;
}

inline uint k4_header_byte(uint word, uint byte_index) {
    return (word >> (byte_index * 8u)) & 0xFFu;
}

inline uint2 decode_k4_scale_min(uint i, uint4 hdr) {
    if (i < 4u) {
        return uint2(k4_header_byte(hdr.y, i) & 63u,
                     k4_header_byte(hdr.z, i) & 63u);
    }
    uint j = i - 4u;
    uint sc = (k4_header_byte(hdr.w, j) & 0x0Fu)
        | ((k4_header_byte(hdr.y, j) >> 6u) << 4u);
    uint mn = (k4_header_byte(hdr.w, j) >> 4u)
        | ((k4_header_byte(hdr.z, j) >> 6u) << 4u);
    return uint2(sc, mn);
}

// NATIVE Q4_K block (144 B / 256 elems): [f16 d][f16 dmin][12 B 6-bit scales][128 B nibbles].
// 16-block `sub` within the 256-block: quarter j = sub/4 (qs bytes j*32..), low/high nibble half
// `hi`, and which 16 of the 32 l-values `l0`. Scale/min pair via get_scale_min_k4(2j + hi).
// scm/dd are unused (dummy bindings). 144 % 4 == 0, so uint loads stay aligned.
template<bool WIDE>
inline void decode16_q4k(device const uchar* codes, uint bi, thread float* wk) {
    device const uchar* blk = codes + (ulong)(bi >> 4) * 144ul;
    uint sub = bi & 15u;
    uint j = sub >> 2;
    uint hi = (sub >> 1) & 1u;
    uint l0 = (sub & 1u) * 16u;
    uint jj = 2u * j + hi;
    uint dm;
    uint2 sm;
    if (WIDE) {
        uint4 hdr = load_k4_header(blk);
        dm = hdr.x;
        sm = decode_k4_scale_min(jj, hdr);
    } else {
        dm = *(device const uint*)blk;
        device const uchar* scb = blk + 4u;
        if (jj < 4u) {
            sm = uint2(scb[jj] & 63u, scb[jj + 4u] & 63u);
        } else {
            sm.x = (scb[jj + 4u] & 0x0Fu) | ((scb[jj - 4u] >> 6u) << 4u);
            sm.y = (scb[jj + 4u] >> 4u) | ((scb[jj] >> 6u) << 4u);
        }
    }
    float d = (float)as_type<half>((ushort)(dm & 0xFFFFu));
    float dmin = (float)as_type<half>((ushort)(dm >> 16));
    float scale = (hi != 0u ? d * (1.0f / 16.0f) : d) * (float)sm.x;
    float mn = -(dmin * (float)sm.y);
    uint nibmask = hi != 0u ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    device const uint* qw4 = (device const uint*)(blk + 16u + j * 32u + l0);
    for (uint w = 0; w < 4u; w++) {
        uint u = qw4[w] & nibmask;
        for (uint k2 = 0; k2 < 4u; k2++) {
            wk[w * 4u + k2] = scale * (float)((u >> (8u * k2)) & 0xFFu) + mn;
        }
    }
}

#define DEC16_Q4K(wk) decode16_q4k<false>(codes, bi, wk);
#define DEC16_Q4K_WIDE(wk) decode16_q4k<true>(codes, bi, wk);

// NATIVE Q6_K block (210 B / 256 elems): [128 B ql][64 B qh][16 x i8 scales][f16 d].
// 16-block `sub`: half h6 = sub/8 (128 elems each), then group off = (sub%8 / 2)*32 with
// is = sub%2 selecting which 16 l-values; q built from ql/qh nibble+2-bit pieces, scale from the
// i8 scale at [h6*8 + off/16 + is], min = -32*scale (exact power-of-two scaling of the same f32
// product the reference computes). 210 % 4 != 0, so all loads are byte loads.
#define DEC16_Q6K(wk)                                                                             \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 210ul;                                   \
    uint sub = bi & 15u;                                                                          \
    uint h6 = sub >> 3;                                                                           \
    uint sub8 = sub & 7u;                                                                         \
    uint off = (sub8 >> 1) * 32u;                                                                 \
    uint is = sub8 & 1u;                                                                          \
    device const uchar* ql = blk + h6 * 64u;                                                      \
    device const uchar* qh = blk + 128u + h6 * 32u;                                               \
    device const uchar* scs = blk + 192u;                                                         \
    float d = (float)as_type<half>((ushort)(blk[208] | ((ushort)blk[209] << 8)));                 \
    float scale = d * (float)(char)scs[h6 * 8u + (off >> 4) + is];                                \
    float mn = -32.0f * scale;                                                                    \
    /* uint32-lane unpack (the reference dequantize_q6_K shape): four 32-bit combines cover the */ \
    /* 16 codes, nibble/crumb selection folded into masks and power-of-two scale variants — all */ \
    /* exact, so the value is bit-identical to the byte-at-a-time form this replaces            */ \
    uint qlo = (off & 32u);                                                                       \
    uint qhs = off >> 4;                                                                          \
    device const ushort* ql16 = (device const ushort*)ql + (qlo != 0u ? 16u : 0u) + 8u * is;      \
    device const ushort* qh16 = (device const ushort*)qh + 8u * is;                               \
    uint kmask1 = (off >= 64u) ? ((qhs > 4u) ? 0xC0C0C0C0u : 0x30303030u)                         \
                               : ((qhs > 0u) ? 0x0C0C0C0Cu : 0x03030303u);                        \
    uint kmask2 = (off >= 64u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;                                       \
    float dl0 = scale;                                                                            \
    float dl1 = dl0 * (1.0f / 256.0f);                                                            \
    float dl2 = dl1 * (1.0f / 256.0f);                                                            \
    float dl3 = dl2 * (1.0f / 256.0f);                                                            \
    uint shr_h = (qhs > 4u) ? 2u : 0u;                                                            \
    uint shl_h = (off >= 64u) ? 0u : ((qhs > 0u) ? 2u : 4u);                                      \
    uint shr_l = (off >= 64u) ? 4u : 0u;                                                          \
    for (uint i = 0; i < 4u; i++) {                                                               \
        uint low  = ((uint)ql16[2u * i] | ((uint)ql16[2u * i + 1u] << 16)) & kmask2;              \
        uint high = ((uint)qh16[2u * i] | ((uint)qh16[2u * i + 1u] << 16)) & kmask1;              \
        uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);                                     \
        wk[4u * i]      = dl0 * (float)(q & 0xFFu)       + mn;                                    \
        wk[4u * i + 1u] = dl1 * (float)(q & 0xFF00u)     + mn;                                    \
        wk[4u * i + 2u] = dl2 * (float)(q & 0xFF0000u)   + mn;                                    \
        wk[4u * i + 3u] = dl3 * (float)(q & 0xFF000000u) + mn;                                    \
    }

// NATIVE Q5_K block (176 B / 256 elems): [f16 d][f16 dmin][12 B 6-bit scales/mins][32 B qh]
// [128 B qs]. 5.5 bpw streamed vs the factored form's ~8. Same two-level (d,dmin)+(sc,m) scale as
// Q4_K (get_scale_min_k4), plus a 5th bit per element from the qh plane: sub-block s uses qh bit
// (1 << s). code = nibble + 16*qh_bit; value = d*sc*code - dmin*m — bit-exact vs dequant_block.
template<bool WIDE>
inline void decode16_q5k(device const uchar* codes, uint bi, thread float* wk) {
    device const uchar* blk = codes + (ulong)(bi >> 4) * 176ul;
    uint sub = bi & 15u;
    uint s = sub >> 1u;
    uint j = s >> 1u;
    bool is_low = (s & 1u) == 0u;
    uint l0 = (sub & 1u) * 16u;
    uint dm;
    uint2 sm;
    if (WIDE) {
        uint4 hdr = load_k4_header(blk);
        dm = hdr.x;
        sm = decode_k4_scale_min(s, hdr);
    } else {
        dm = *(device const uint*)blk;
        device const uchar* scb = blk + 4u;
        if (s < 4u) {
            sm = uint2(scb[s] & 63u, scb[s + 4u] & 63u);
        } else {
            sm.x = (scb[s + 4u] & 0x0Fu) | ((scb[s - 4u] >> 6u) << 4u);
            sm.y = (scb[s + 4u] >> 4u) | ((scb[s] >> 6u) << 4u);
        }
    }
    float d = (float)as_type<half>((ushort)(dm & 0xFFFFu));
    float dmin = (float)as_type<half>((ushort)(dm >> 16));
    float scale = d * (float)sm.x;
    float mn = -(dmin * (float)sm.y);
    device const uchar* qh = blk + 16u + l0;
    device const uchar* qs = blk + 48u + j * 32u + l0;
    device const uint* qh4 = (device const uint*)qh;
    device const uint* qs4 = (device const uint*)qs;
    uint qshift = is_low ? 0u : 4u;
    for (uint k = 0u; k < 4u; k++) {
        uint q = qs4[k] >> qshift;
        uint h = qh4[k] >> s;
        uint packed = (q & 0x0F0F0F0Fu) | ((h & 0x01010101u) << 4u);
        wk[4u * k] = scale * (float)(packed & 0x1Fu) + mn;
        wk[4u * k + 1u] = scale * (float)((packed >> 8u) & 0x1Fu) + mn;
        wk[4u * k + 2u] = scale * (float)((packed >> 16u) & 0x1Fu) + mn;
        wk[4u * k + 3u] = scale * (float)(packed >> 24u) + mn;
    }
}

#define DEC16_Q5K(wk) decode16_q5k<false>(codes, bi, wk);
#define DEC16_Q5K_WIDE(wk) decode16_q5k<true>(codes, bi, wk);

// NATIVE Q2_K block (84 B / 256 elems): [u8 scales[16]][u8 qs[64]][f16 d][f16 dmin]. 2.5 bpw.
// AFFINE like Q4_K, but 16 sub-blocks of 16 with a 4-bit UNSIGNED scale AND 4-bit UNSIGNED min
// packed in one byte per sub-block: scale=(scales[is]&0xF), min=(scales[is]>>4). The 2-bit code
// q2 is read from qs in the dequant reference's `n x shift x half` traversal, so for the 16-block
// counter `sub` (== `is`): n=sub/8, j=(sub/2)%4 -> shift=2j, half=sub&1 selecting which 16 of the
// 32-byte qs quarter (qoff=n*32). value = d*scale*q2 - dmin*min — bit-exact vs dequant_block /
// vec_dot_q2k_scalar (crates/infr-cpu/src/kernels.rs).
inline void decode16_q2k(device const uchar* codes, uint bi, thread float* wk) {
    device const uchar* blk = codes + (ulong)(bi >> 4) * 84ul;
    uint sub = bi & 15u;
    float d    = (float)as_type<half>((ushort)(blk[80] | ((ushort)blk[81] << 8)));
    float dmin = (float)as_type<half>((ushort)(blk[82] | ((ushort)blk[83] << 8)));
    uint scb = blk[sub];
    float scale = d * (float)(scb & 0xFu);
    float mn = -(dmin * (float)(scb >> 4));
    uint n = sub >> 3;
    uint j = (sub >> 1) & 3u;
    uint hf = sub & 1u;
    uint shift = 2u * j;
    device const uchar* qb = blk + 16u + n * 32u + hf * 16u;
    for (uint l = 0; l < 16u; l++) wk[l] = scale * (float)((qb[l] >> shift) & 3u) + mn;
}

#define DEC16_Q2K(wk) decode16_q2k(codes, bi, wk);

// NATIVE Q3_K block (110 B / 256 elems): [u8 hmask[32]][u8 qs[64]][u8 scales[12]][f16 d]. 3.44 bpw.
// SIGNED like Q6_K: 16 sub-blocks of 16, each a signed 6-bit scale (sc6-32) times a signed offset
// code (q3-4). The 16 x 6-bit scales are packed in 12 bytes via llama.cpp's aux bit-shuffle
// (ported EXACTLY from q3k_scales / dequant_block, ggml dequantize_row_q3_K). The 3-bit code q3 is
// 2 low bits from qs (same `n x shift x half` traversal as Q2_K) plus 1 high bit from an hmask
// bit-plane: for the 16-block counter `sub` (== `is`) the plane bit is m = 1 << (n*4 + j) (8 planes
// per superblock, 4 per n). value = d*(sc6-32)*(q3-4) — bit-exact vs vec_dot_q3k_scalar.
inline void decode16_q3k(device const uchar* codes, uint bi, thread float* wk) {
    device const uchar* blk = codes + (ulong)(bi >> 4) * 110ul;
    uint sub = bi & 15u;
    device const uchar* sr = blk + 96u;
    uint a0 = (uint)sr[0] | ((uint)sr[1] << 8) | ((uint)sr[2] << 16) | ((uint)sr[3] << 24);
    uint a1 = (uint)sr[4] | ((uint)sr[5] << 8) | ((uint)sr[6] << 16) | ((uint)sr[7] << 24);
    uint a2 = (uint)sr[8] | ((uint)sr[9] << 8) | ((uint)sr[10] << 16) | ((uint)sr[11] << 24);
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    uint tmp = a2;
    uint aux[4];
    aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (a0 & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int sc6 = (int)(char)((aux[sub >> 2] >> ((sub & 3u) * 8u)) & 0xFFu) - 32;
    float d = (float)as_type<half>((ushort)(blk[108] | ((ushort)blk[109] << 8)));
    float scale = d * (float)sc6;
    uint n = sub >> 3;
    uint j = (sub >> 1) & 3u;
    uint hf = sub & 1u;
    uint shift = 2u * j;
    uint m = 1u << (n * 4u + j);
    device const uchar* qb = blk + 32u + n * 32u + hf * 16u;
    device const uchar* hb = blk + hf * 16u;
    for (uint l = 0; l < 16u; l++) {
        uint low2 = (qb[l] >> shift) & 3u;
        uint high = (hb[l] & m) != 0u ? 4u : 0u;
        wk[l] = scale * ((float)(int)(low2 | high) - 4.0f);
    }
}

#define DEC16_Q3K(wk) decode16_q3k(codes, bi, wk);

// NATIVE Q4_0 block (18 B / 32 elems): [f16 d][16 B nibbles] — 4.5 bpw streamed vs the
// factored form's ~6.1. Element e < 16 is the low nibble of qs[e], e >= 16 the high nibble of
// qs[e-16]; value = d * (q - 8), exact per element.
#define DEC16_Q4_0(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 1) * 18ul;                                    \
    device const ushort* b16 = (device const ushort*)blk;                                         \
    float d = (float)as_type<half>(b16[0]);                                                      \
    device const ushort* q16 = (device const ushort*)(blk + 2u);                                  \
    uint hi4 = bi & 1u;                                                                           \
    for (uint k = 0; k < 8u; k++) {                                                               \
        ushort u = q16[k];                                                                        \
        uint b0 = hi4 ? ((u >> 4) & 0x0Fu) : (u & 0x0Fu);                                         \
        uint b1 = hi4 ? ((u >> 12) & 0x0Fu) : ((u >> 8) & 0x0Fu);                                 \
        wk[2u * k]      = d * ((float)b0 - 8.0f);                                                 \
        wk[2u * k + 1u] = d * ((float)b1 - 8.0f);                                                 \
    }

// NATIVE Q4_1 block (20 B / 32 elems): [f16 d][f16 m][16 B nibbles] — the AFFINE sibling of Q4_0.
// value = d * q + m (q in 0..15, NO -8 offset; a per-block f16 `m` min replaces it). Element e < 16
// is the low nibble of qs[e], e >= 16 the high nibble of qs[e-16]; block stride 20, qs at byte 4.
// Bit-for-bit dequantize_row_q4_1 / vec_dot_q4_1_32_batch_scalar (crates/infr-cpu/src/kernels.rs).
#define DEC16_Q4_1(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 1) * 20ul;                                    \
    device const ushort* b16 = (device const ushort*)blk;                                         \
    float d = (float)as_type<half>(b16[0]);                                                       \
    float mn = (float)as_type<half>(b16[1]);                                                      \
    device const ushort* q16 = (device const ushort*)(blk + 4u);                                  \
    uint hi4 = bi & 1u;                                                                           \
    for (uint k = 0; k < 8u; k++) {                                                               \
        ushort u = q16[k];                                                                        \
        uint b0 = hi4 ? ((u >> 4) & 0x0Fu) : (u & 0x0Fu);                                         \
        uint b1 = hi4 ? ((u >> 12) & 0x0Fu) : ((u >> 8) & 0x0Fu);                                 \
        wk[2u * k]      = d * (float)b0 + mn;                                                     \
        wk[2u * k + 1u] = d * (float)b1 + mn;                                                     \
    }

// IQ4_NL / IQ4_XS shared 16-entry signed codebook (llama.cpp kvalues_iq4nl).
constant float kvalues_iq4nl_f[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f,
};

// NATIVE IQ4_XS block (136 B / 256 elems): [f16 d][u16 scales_h][4 B scales_l][128 B nibbles],
// 8 sub-blocks of 32; value = (d * (ls - 32)) * kvalues[q4] — every factor exact in f32, so it
// matches dequant_codebook bit-for-bit. The dominant mac format (bartowski IQ4_XS mixes);
// without a native kernel it dequanted to a cached f32 weight, which OOM-corrupted any >2B
// model on a 18 GiB machine.
#define DEC16_IQ4XS(wk)                                                                           \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 136ul;                                   \
    device const ushort* b16 = (device const ushort*)blk;                                         \
    float d = (float)as_type<half>(b16[0]);                                                      \
    uint sub = bi & 15u;                                                                          \
    uint ib4 = sub >> 1;                                                                          \
    uint lo = ((uint)blk[4u + (ib4 >> 1)] >> (4u * (ib4 & 1u))) & 0xFu;                           \
    uint hi2 = ((uint)b16[1] >> (2u * ib4)) & 3u;                                                 \
    float dl = d * ((float)(lo | (hi2 << 4)) - 32.0f);                                            \
    device const uchar* qs = blk + 8u + ib4 * 16u;                                                \
    uint h4 = (sub & 1u) * 4u;                                                                    \
    for (uint k = 0; k < 16u; k++) wk[k] = dl * kvalues_iq4nl_f[(qs[k] >> h4) & 0xFu];

// NATIVE IQ4_NL block (18 B / 32 elems): [f16 d][16 B nibbles]; value = d * kvalues[q4].
#define DEC16_IQ4NL(wk)                                                                           \
    device const uchar* blk = codes + (ulong)(bi >> 1) * 18ul;                                    \
    device const ushort* b16 = (device const ushort*)blk;                                         \
    float d = (float)as_type<half>(b16[0]);                                                      \
    device const uchar* qs = blk + 2u;                                                            \
    uint h4 = (bi & 1u) * 4u;                                                                     \
    for (uint k = 0; k < 16u; k++) wk[k] = d * kvalues_iq4nl_f[(qs[k] >> h4) & 0xFu];

// MXFP4 / NVFP4 shared 16-entry signed integer codebook (E2M1 values ×2). Bit-for-bit
// infr_gguf::dequant::KVALUES_MXFP4 = [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12].
constant float kvalues_mxfp4_f[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

// E8M0 shared-exponent byte → f32, HALVED (2^(e-128)). Bit-for-bit
// infr_gguf::dequant::e8m0_to_fp32_half: e<2 → from_bits(0x00200000 << e); else from_bits((e-1)<<23).
inline float e8m0_to_fp32_half_msl(uint e) {
    return (e < 2u) ? as_type<float>(0x00200000u << e) : as_type<float>((e - 1u) << 23);
}

// UE4M3 byte (unsigned, 4 exp bits bias=7, 3 mantissa bits) → f32, HALVED. Bit-for-bit
// infr_gguf::dequant::ue4m3_to_fp32. Powers of two are built by bit construction (not exp2) so the
// scale is EXACT — every product below is a power-of-two exponent shift of an exact mantissa.
inline float ue4m3_to_fp32_msl(uint x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    int e = (int)((x >> 3) & 0xFu);
    float man = (float)(x & 0x7u);
    float raw;
    if (e == 0) {
        raw = man * as_type<float>((uint)(127 - 9) << 23);                     // man * 2^-9
    } else {
        raw = (1.0f + man / 8.0f) * as_type<float>((uint)(e - 7 + 127) << 23); // (1+man/8)*2^(e-7)
    }
    return raw * 0.5f;
}

// NATIVE MXFP4 block (17 B / 32 elems): [u8 e][16 B nibbles]. E8M0 shared exponent + E2M1 codebook;
// value = e8m0_to_fp32_half(e) * kvalues_mxfp4[q4]. Low nibble of qs[k] → element k (0..15), high
// nibble → element k+16 — the SAME nibble layout as IQ4_NL, so the 32-elem block spans two DEC16
// chunks (bi & 1 picks the half). Bit-for-bit dequantize_row_mxfp4 / vec_dot_mxfp4_32_batch_scalar.
#define DEC16_MXFP4(wk)                                                                           \
    device const uchar* blk = codes + (ulong)(bi >> 1) * 17ul;                                    \
    float d = e8m0_to_fp32_half_msl((uint)blk[0]);                                                \
    device const uchar* qs = blk + 1u;                                                            \
    uint h4 = (bi & 1u) * 4u;                                                                      \
    for (uint k = 0; k < 16u; k++) wk[k] = d * kvalues_mxfp4_f[(qs[k] >> h4) & 0xFu];

// NATIVE NVFP4 block (36 B / 64 elems): [u8 scales[4]][32 B nibbles]. Four 16-element sub-blocks,
// each with its own UE4M3 scale; value = ue4m3_to_fp32(scales[s]) * kvalues_mxfp4[q4]. Within
// sub-block s the 8 code bytes are qs[s*8..s*8+8]: low nibble of qs[s*8+j] → element s*16+j (0..7),
// high nibble → element s*16+j+8 (8..15). Each DEC16 chunk of 16 IS exactly one sub-block, so
// bi & 3 selects the sub-block and bi >> 2 the 64-elem block. Bit-for-bit dequantize_row_nvfp4 /
// vec_dot_nvfp4_batch_scalar.
#define DEC16_NVFP4(wk)                                                                           \
    device const uchar* blk = codes + (ulong)(bi >> 2) * 36ul;                                    \
    uint s = (uint)(bi & 3u);                                                                      \
    float d = ue4m3_to_fp32_msl((uint)blk[s]);                                                    \
    device const uchar* qs = blk + 4u + s * 8u;                                                   \
    for (uint k = 0; k < 8u; k++) {                                                               \
        wk[k]      = d * kvalues_mxfp4_f[qs[k] & 0xFu];                                           \
        wk[k + 8u] = d * kvalues_mxfp4_f[(qs[k] >> 4) & 0xFu];                                    \
    }

// NATIVE Q2_0 block (18 B / 64 elems): [f16 d][16 B qs] — Bonsai ternary, 2 bits/elem packed
// sequentially. value = d * (q - 1), q in 0..3. Element j maps to byte qs[j/4], shift (j%4)*2
// (fully sequential, NO nibble interleave). The 64-elem block spans FOUR DEC16 chunks, so
// bi >> 2 is the block and (bi & 3) picks the 16-element chunk covering global-in-block indices
// (bi&3)*16 .. +16. Bit-for-bit dequantize_row_q2_0 / vec_dot_q2_0_batch_scalar.
#define DEC16_Q2_0(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 2) * 18ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    device const uchar* qs = blk + 2u;                                                            \
    uint g0 = (bi & 3u) * 16u;                                                                     \
    for (uint e = 0; e < 16u; e++) {                                                              \
        uint g = g0 + e;                                                                          \
        uint q = (qs[g >> 2] >> ((g & 3u) * 2u)) & 3u;                                            \
        wk[e] = d * ((float)q - 1.0f);                                                            \
    }

// NATIVE TQ2_0 block (66 B / 256 elems): [64 B qs][f16 d] — ternary, 2 bits/elem. value =
// d * (q - 1), q in 0..3. Two 32-byte chunks: for chunk in 0..2, l in 0..4, m in 0..32 → element
// chunk*128 + l*32 + m, sourced from qs[chunk*32 + m] >> (l*2). The 256-elem block spans SIXTEEN
// DEC16 chunks (bi & 15 selects the 16-element chunk, bi >> 4 the block); each element's
// (chunk,l,m) is recovered from its global-in-block index. Bit-for-bit dequantize_row_tq2_0 /
// vec_dot_tq2_0_scalar.
#define DEC16_TQ2_0(wk)                                                                           \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 66ul;                                    \
    device const uchar* qs = blk;                                                                 \
    float d = (float)as_type<half>((ushort)(blk[64] | ((ushort)blk[65] << 8)));                   \
    uint g0 = (bi & 15u) * 16u;                                                                    \
    for (uint e = 0; e < 16u; e++) {                                                              \
        uint g = g0 + e;                                                                          \
        uint chunk = g >> 7;                                                                      \
        uint rem = g & 127u;                                                                      \
        uint l = rem >> 5;                                                                        \
        uint m = rem & 31u;                                                                       \
        uint q = (qs[chunk * 32u + m] >> (l * 2u)) & 3u;                                          \
        wk[e] = d * ((float)q - 1.0f);                                                            \
    }

// Base-3 place values for the TQ1_0 5-digits-per-byte packing (llama.cpp pow3).
constant uint kpow3_tq1[6] = {1u, 3u, 9u, 27u, 81u, 243u};

// NATIVE TQ1_0 block (54 B / 256 elems): [48 B qs][4 B qh][f16 d] — ternary base-3, 5 digits per
// byte. digit = (((byte * pow3[n]) & 0xFF) * 3) >> 8, in {0,1,2}; value = d * (digit - 1). Three
// element segments (mirroring the CPU scalar exactly):
//   * elems 0..160  : qs[0..32], 5 passes → element n*32 + m, source byte qs[m], pass n = g/32
//   * elems 160..240: qs[32..48], 5 passes → element 160 + n*16 + m, source qs[32+m], n = g'/16
//   * elems 240..256: qh[0..4], 4 passes → element 240 + n*4 + j, source qh[j], n = g''/4
// The 256-elem block spans SIXTEEN DEC16 chunks (bi & 15 picks the chunk, bi >> 4 the block); each
// element's (segment, pass, source-byte) is recovered from its global-in-block index. The u8
// wrapping-multiply (`& 0xFF`) mirrors the CPU `wrapping_mul`. Bit-for-bit dequantize_row_tq1_0 /
// vec_dot_tq1_0_scalar.
#define DEC16_TQ1_0(wk)                                                                           \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 54ul;                                    \
    device const uchar* qs = blk;                                                                 \
    device const uchar* qh = blk + 48u;                                                           \
    float d = (float)as_type<half>((ushort)(blk[52] | ((ushort)blk[53] << 8)));                   \
    uint g0 = (bi & 15u) * 16u;                                                                    \
    for (uint e = 0; e < 16u; e++) {                                                              \
        uint g = g0 + e;                                                                          \
        uint n;                                                                                   \
        uint bt;                                                                                  \
        if (g < 160u) {                                                                           \
            n = g >> 5;                                                                           \
            bt = (uint)qs[g & 31u];                                                               \
        } else if (g < 240u) {                                                                    \
            uint g2 = g - 160u;                                                                   \
            n = g2 >> 4;                                                                          \
            bt = (uint)qs[32u + (g2 & 15u)];                                                      \
        } else {                                                                                  \
            uint g3 = g - 240u;                                                                   \
            n = g3 >> 2;                                                                          \
            bt = (uint)qh[g3 & 3u];                                                               \
        }                                                                                         \
        uint prod = (bt * kpow3_tq1[n]) & 0xFFu;                                                  \
        uint digit = (prod * 3u) >> 8u;                                                           \
        wk[e] = d * ((float)digit - 1.0f);                                                        \
    }

// NATIVE IQ2_XXS block (66 B / 256 elems): [f16 d][64 B qs]. 2.06 bpw. 8 sub-blocks of 32 elems;
// each sub-block's 8 qs bytes are aux0 (four 8-bit grid indices into IQ2XXS_GRID[256]) + aux1
// (four 7-bit sign indices into KSIGNS_IQ2XS, and a 4-bit scale magnitude in the top nibble).
// Each grid entry expands to 8 signed values, so a 16-element group covers two grid entries.
// IQ2XXS_GRID/KSIGNS_IQ2XS are generated into the source ahead of this file from the same tables
// the CPU reads, so this matches dequant_codebook (ggml dequantize_row_iq2_xxs) bit-for-bit.
// Without it, IQ2_XXS dequanted to a cached f32 weight — OOM on any large model.
#define DEC16_IQ2XXS(wk)                                                                          \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 66ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    uint g = (uint)(bi & 15ul);                                                                   \
    device const uchar* qs = blk + 2u + (g >> 1u) * 8u;                                           \
    uint aux0 = (uint)qs[0] | ((uint)qs[1] << 8) | ((uint)qs[2] << 16) | ((uint)qs[3] << 24);     \
    uint aux1 = (uint)qs[4] | ((uint)qs[5] << 8) | ((uint)qs[6] << 16) | ((uint)qs[7] << 24);     \
    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;                                           \
    uint lbase = (g & 1u) * 2u;                                                                    \
    for (uint li = 0u; li < 2u; li++) {                                                            \
        uint l = lbase + li;                                                                       \
        ulong grid = IQ2XXS_GRID[(aux0 >> (8u * l)) & 0xFFu];                                      \
        uint signs = (uint)KSIGNS_IQ2XS[(aux1 >> (7u * l)) & 127u];                                \
        for (uint j = 0u; j < 8u; j++) {                                                           \
            char gv = (char)((grid >> (8u * j)) & 0xFFu);                                          \
            wk[li * 8u + j] = db * (float)gv * ((signs & (1u << j)) ? -1.0f : 1.0f);               \
        }                                                                                          \
    }

// NATIVE IQ3_XXS block (98 B / 256 elems): [f16 d][64 B grid indices][32 B scales_and_signs].
// 3.06 bpw. 8 sub-blocks of 32; each sub-block's 4-byte sas word holds four 7-bit sign indices
// into KSIGNS_IQ2XS plus a 4-bit scale in the top nibble. Each of the sub-block's 8 grid-index
// bytes selects an IQ3XXS_GRID[256] entry (a u32 packing FOUR signed values), and the entries
// come in pairs (g1 uses signs bits 0..3, g2 uses bits 4..7) — a 16-element group spans two
// pairs. Generated grid ⇒ bit-exact vs dequant_block (ggml dequantize_row_iq3_xxs).
#define DEC16_IQ3XXS(wk)                                                                          \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 98ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    uint g = (uint)(bi & 15ul);                                                                   \
    uint ib32 = g >> 1u;                                                                          \
    device const uchar* qs = blk + 2u + ib32 * 8u;                                                \
    device const uchar* sas = blk + 66u + ib32 * 4u;                                              \
    uint aux32 = (uint)sas[0] | ((uint)sas[1] << 8) | ((uint)sas[2] << 16) | ((uint)sas[3] << 24);\
    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;                                           \
    uint lbase = (g & 1u) * 2u;                                                                    \
    for (uint li = 0u; li < 2u; li++) {                                                            \
        uint l = lbase + li;                                                                       \
        uint signs = (uint)KSIGNS_IQ2XS[(aux32 >> (7u * l)) & 127u];                               \
        uint g1 = IQ3XXS_GRID[qs[2u * l]];                                                         \
        uint g2 = IQ3XXS_GRID[qs[2u * l + 1u]];                                                    \
        for (uint j = 0u; j < 4u; j++) {                                                           \
            char v1 = (char)((g1 >> (8u * j)) & 0xFFu);                                            \
            char v2 = (char)((g2 >> (8u * j)) & 0xFFu);                                            \
            wk[li * 8u + j] = db * (float)v1 * ((signs & (1u << j)) ? -1.0f : 1.0f);               \
            wk[li * 8u + j + 4u] = db * (float)v2 * ((signs & (1u << (j + 4u))) ? -1.0f : 1.0f);   \
        }                                                                                          \
    }

// NATIVE IQ2_XS block (74 B / 256 elems): [f16 d][64 B qs = 32 u16][8 B scales]. 2.31 bpw. Each
// u16 packs a 9-bit IQ2XS_GRID[512] index + a 7-bit KSIGNS_IQ2XS index; the sub-block's scale
// byte gives db for entries 0..1 (low nibble) and 2..3 (high nibble).
#define DEC16_IQ2XS(wk)                                                                           \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 74ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    uint g = (uint)(bi & 15ul);                                                                   \
    uint ib32 = g >> 1u;                                                                          \
    device const uchar* qs = blk + 2u + ib32 * 8u;                                                \
    uint sc = blk[66u + ib32];                                                                    \
    float db0 = d * (0.5f + (float)(sc & 0xFu)) * 0.25f;                                           \
    float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;                                             \
    uint lbase = (g & 1u) * 2u;                                                                    \
    for (uint li = 0u; li < 2u; li++) {                                                           \
        uint l = lbase + li;                                                                       \
        uint qs16 = (uint)qs[2u * l] | ((uint)qs[2u * l + 1u] << 8);                               \
        ulong grid = IQ2XS_GRID[qs16 & 511u];                                                      \
        uint signs = (uint)KSIGNS_IQ2XS[qs16 >> 9];                                                \
        float dl = (l < 2u) ? db0 : db1;                                                           \
        for (uint j = 0u; j < 8u; j++) {                                                           \
            char gv = (char)((grid >> (8u * j)) & 0xFFu);                                          \
            wk[li * 8u + j] = dl * (float)gv * ((signs & (1u << j)) ? -1.0f : 1.0f);               \
        }                                                                                          \
    }

// NATIVE IQ2_S block (82 B / 256 elems): [f16 d][32 B grid-idx-low][32 B sign bytes][8 B qh]
// [8 B scales]. 2.5 bpw. The grid index is qs_low | (2 high bits from qh) → IQ2S_GRID[1024], and
// the sign is a PER-ENTRY byte (not the KSIGNS table).
#define DEC16_IQ2S(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 82ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    uint g = (uint)(bi & 15ul);                                                                   \
    uint ib32 = g >> 1u;                                                                          \
    device const uchar* qsb = blk + 2u + ib32 * 4u;                                               \
    device const uchar* sgn = blk + 2u + 32u + ib32 * 4u;                                         \
    uint qh = blk[66u + ib32];                                                                    \
    uint sc = blk[74u + ib32];                                                                    \
    float db0 = d * (0.5f + (float)(sc & 0xFu)) * 0.25f;                                           \
    float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;                                             \
    uint lbase = (g & 1u) * 2u;                                                                    \
    for (uint li = 0u; li < 2u; li++) {                                                           \
        uint l = lbase + li;                                                                       \
        uint hi = (qh << (8u - 2u * l)) & 0x300u;                                                  \
        ulong grid = IQ2S_GRID[(uint)qsb[l] | hi];                                                 \
        uint sb = (uint)sgn[l];                                                                    \
        float dl = (l < 2u) ? db0 : db1;                                                           \
        for (uint j = 0u; j < 8u; j++) {                                                           \
            char gv = (char)((grid >> (8u * j)) & 0xFFu);                                          \
            wk[li * 8u + j] = dl * (float)gv * ((sb & (1u << j)) ? -1.0f : 1.0f);                  \
        }                                                                                          \
    }

// NATIVE IQ3_S block (110 B / 256 elems): [f16 d][64 B qs][8 B qh][32 B signs][4 B scales].
// 3.44 bpw. Grid entries pair up (g1 uses sign bits 0..3, g2 bits 4..7) into IQ3S_GRID[512]; the
// 9th index bit comes from qh, and the scale is d*(1 + 2*nibble) per 32-element group.
#define DEC16_IQ3S(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 110ul;                                   \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    uint g = (uint)(bi & 15ul);                                                                   \
    uint gr = g >> 1u;                                                                            \
    uint pair = gr >> 1u;                                                                         \
    uint gsel = gr & 1u;                                                                          \
    device const uchar* qs = blk + 2u + gr * 8u;                                                  \
    uint qh = blk[66u + gr];                                                                      \
    device const uchar* sgn = blk + 74u + gr * 4u;                                                \
    uint sc = blk[106u + pair];                                                                   \
    float db = (gsel == 0u) ? d * (1.0f + 2.0f * (float)(sc & 0xFu))                               \
                            : d * (1.0f + 2.0f * (float)(sc >> 4));                                \
    uint lbase = (g & 1u) * 2u;                                                                    \
    for (uint li = 0u; li < 2u; li++) {                                                           \
        uint l = lbase + li;                                                                       \
        uint g1 = IQ3S_GRID[(uint)qs[2u * l] | ((qh << (8u - 2u * l)) & 256u)];                    \
        uint g2 = IQ3S_GRID[(uint)qs[2u * l + 1u] | ((qh << (7u - 2u * l)) & 256u)];               \
        uint sb = (uint)sgn[l];                                                                    \
        for (uint j = 0u; j < 4u; j++) {                                                           \
            char v1 = (char)((g1 >> (8u * j)) & 0xFFu);                                            \
            char v2 = (char)((g2 >> (8u * j)) & 0xFFu);                                            \
            wk[li * 8u + j] = db * (float)v1 * ((sb & (1u << j)) ? -1.0f : 1.0f);                  \
            wk[li * 8u + j + 4u] = db * (float)v2 * ((sb & (1u << (j + 4u))) ? -1.0f : 1.0f);      \
        }                                                                                          \
    }

// NATIVE IQ1_S block (50 B / 256 elems): [f16 d][32 B qs][16 B qh = 8 u16]. 1.56 bpw. 8 sub-blocks
// of 32; each sub-block's u16 qh carries a 3-bit scale (bits 12..14 → dl = d*(2*sc+1)), a sign bit
// (0x8000 → delta = ±IQ1S_DELTA=0.125), and four 3-bit high parts that extend each of the 4 qs
// bytes into an 11-bit IQ1S_GRID[2048] index. UNLIKE the other grids the weight is dl*(grid+delta):
// delta is an ADDITIVE fractional term baked in here (there is no sign codebook). A 16-element group
// spans two 8-value grid entries. Generated IQ1S_GRID ⇒ bit-exact vs dequant_block (dequant_row_iq1_s).
#define DEC16_IQ1S(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 50ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    uint g = (uint)(bi & 15ul);                                                                   \
    uint sib = g >> 1u;                                                                           \
    device const uchar* qs = blk + 2u + sib * 4u;                                                 \
    uint qh = (uint)blk[34u + 2u * sib] | ((uint)blk[35u + 2u * sib] << 8);                       \
    float dl = d * (2.0f * (float)((qh >> 12) & 7u) + 1.0f);                                       \
    float delta = (qh & 0x8000u) ? -0.125f : 0.125f;                                              \
    uint lbase = (g & 1u) * 2u;                                                                    \
    for (uint li = 0u; li < 2u; li++) {                                                           \
        uint l = lbase + li;                                                                      \
        ulong grid = IQ1S_GRID[(uint)qs[l] | (((qh >> (3u * l)) & 7u) << 8u)];                    \
        for (uint j = 0u; j < 8u; j++) {                                                          \
            char gv = (char)((grid >> (8u * j)) & 0xFFu);                                         \
            wk[li * 8u + j] = dl * ((float)gv + delta);                                           \
        }                                                                                         \
    }

// NATIVE IQ1_M block (56 B / 256 elems): [32 B qs][16 B qh][8 B scales]. 1.75 bpw. NO separate d —
// it is a f16 assembled from the top nibbles of the four u16 scales. 8 sub-blocks of 32; each
// sub-block's scale u16 sc[ib/2] holds two 3-bit scales (dl1 for l=0,1 at bit 6*(ib%2); dl2 for
// l=2,3 at +3). The 11-bit IQ1S_GRID index is qs[l] | 3 high bits from the qh byte (l>>1); the
// delta sign is that qh byte's 0x08 (l even) / 0x80 (l odd) bit → ±IQ1S_DELTA. Weight = dl*(grid+
// delta), delta ADDITIVE and baked here. Generated IQ1S_GRID ⇒ bit-exact vs dequantize_row_iq1_m.
#define DEC16_IQ1M(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 4) * 56ul;                                    \
    device const uchar* sc = blk + 48u;                                                           \
    uint sc0 = (uint)sc[0] | ((uint)sc[1] << 8);                                                  \
    uint sc1 = (uint)sc[2] | ((uint)sc[3] << 8);                                                  \
    uint sc2 = (uint)sc[4] | ((uint)sc[5] << 8);                                                  \
    uint sc3 = (uint)sc[6] | ((uint)sc[7] << 8);                                                  \
    ushort d_bits =                                                                               \
        (ushort)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u));\
    float d = (float)as_type<half>(d_bits);                                                       \
    uint g = (uint)(bi & 15ul);                                                                   \
    uint sib = g >> 1u;                                                                           \
    device const uchar* qs = blk + sib * 4u;                                                      \
    device const uchar* qh = blk + 32u + sib * 2u;                                                \
    uint scw = (sib < 2u) ? sc0 : ((sib < 4u) ? sc1 : ((sib < 6u) ? sc2 : sc3));                  \
    uint shbase = 6u * (sib & 1u);                                                                \
    float dl1 = d * (2.0f * (float)((scw >> shbase) & 7u) + 1.0f);                                \
    float dl2 = d * (2.0f * (float)((scw >> (shbase + 3u)) & 7u) + 1.0f);                         \
    uint lbase = (g & 1u) * 2u;                                                                   \
    for (uint li = 0u; li < 2u; li++) {                                                           \
        uint l = lbase + li;                                                                      \
        uint qhb = (uint)qh[l >> 1u];                                                             \
        uint idx = (uint)qs[l] | ((qhb << (8u - 4u * (l & 1u))) & 0x700u);                        \
        float delta = (qhb & ((l & 1u) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;                      \
        float dl = (l < 2u) ? dl1 : dl2;                                                          \
        ulong grid = IQ1S_GRID[idx];                                                              \
        for (uint j = 0u; j < 8u; j++) {                                                          \
            char gv = (char)((grid >> (8u * j)) & 0xFFu);                                         \
            wk[li * 8u + j] = dl * ((float)gv + delta);                                           \
        }                                                                                         \
    }

// NATIVE Q8_0 block (34 B / 32 elems): [f16 d][32 x i8]. 8.5 bpw streamed vs the factored
// form's ~10.2 (codes + scm + dd) — the decode GEMV is bound on exactly this stream. 34 % 4 != 0,
// so d assembles from bytes and the quants are char loads (same convention as Q6_K's byte loads).
// wk = d * q, the exact dequantize_q8_0 product.
#define DEC16_Q8_0(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 1) * 34ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    device const char* qs = (device const char*)(blk + 2u + (bi & 1u) * 16u);                     \
    for (uint k = 0; k < 16u; k++) wk[k] = d * (float)qs[k];

// NATIVE Q5_0 block (22 B / 32 elems): [f16 d][4 B qh][16 B nibbles]. 5.5 bpw streamed vs the
// factored form's ~8.6 (5-bit codes land in the 6-bit packing + scm + dd) — gemma Q4_K_M ships
// its q/k/v/gate/up projections as Q5_0, so this is that model family's dominant weight stream.
// Element e < 16: low nibble of qs[e] plus bit e of qh as bit 4; element e >= 16: high nibble of
// qs[e-16] plus bit e of qh. Value = d * (q - 16) — every intermediate is exact in f32, so it
// matches the dequant reference bit-for-bit.
#define DEC16_Q5_0(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 1) * 22ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    uint qh = (uint)blk[2] | ((uint)blk[3] << 8) | ((uint)blk[4] << 16) | ((uint)blk[5] << 24);   \
    device const uchar* qs = blk + 6u;                                                            \
    if ((bi & 1u) == 0u) {                                                                        \
        for (uint k = 0; k < 16u; k++) {                                                          \
            uint q = (uint)(qs[k] & 0x0Fu) | (((qh >> k) << 4) & 0x10u);                          \
            wk[k] = d * ((float)q - 16.0f);                                                       \
        }                                                                                         \
    } else {                                                                                      \
        for (uint k = 0; k < 16u; k++) {                                                          \
            uint q = (uint)(qs[k] >> 4) | ((qh >> (k + 12u)) & 0x10u);                            \
            wk[k] = d * ((float)q - 16.0f);                                                       \
        }                                                                                         \
    }

// NATIVE Q5_1 block (24 B / 32 elems): [f16 d][f16 m][4 B qh][16 B nibbles] — the AFFINE sibling
// of Q5_0. value = d * q + m (q in 0..31, NO -16 offset; a per-block f16 `m` min replaces it). The
// 5th bit is assembled from qh EXACTLY as Q5_0 (block stride 24, qh at byte 4, qs at byte 8).
// Bit-for-bit dequantize_row_q5_1 / vec_dot_q5_1_32_batch_scalar (crates/infr-cpu/src/kernels.rs).
#define DEC16_Q5_1(wk)                                                                            \
    device const uchar* blk = codes + (ulong)(bi >> 1) * 24ul;                                    \
    float d = (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));                     \
    float mn = (float)as_type<half>((ushort)(blk[2] | ((ushort)blk[3] << 8)));                    \
    uint qh = (uint)blk[4] | ((uint)blk[5] << 8) | ((uint)blk[6] << 16) | ((uint)blk[7] << 24);   \
    device const uchar* qs = blk + 8u;                                                            \
    if ((bi & 1u) == 0u) {                                                                        \
        for (uint k = 0; k < 16u; k++) {                                                          \
            uint q = (uint)(qs[k] & 0x0Fu) | (((qh >> k) << 4) & 0x10u);                          \
            wk[k] = d * (float)q + mn;                                                            \
        }                                                                                         \
    } else {                                                                                      \
        for (uint k = 0; k < 16u; k++) {                                                          \
            uint q = (uint)(qs[k] >> 4) | ((qh >> (k + 12u)) & 0x10u);                            \
            wk[k] = d * (float)q + mn;                                                            \
        }                                                                                         \
    }

// GEMV: one simdgroup (32 lanes) per output element; each lane decodes one 16-element block per
// step, coalesced across lanes, `simd_sum` reduction. m=1 decode is bound on the weight stream.
#define GEMV_KERNEL(NAME, DEC)                                                                    \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    uint sg = gid / 32u;                                                                          \
    if (sg >= p.m * p.out_f) return;                                                              \
    uint r = sg / p.out_f;                                                                        \
    uint o = sg % p.out_f;                                                                        \
    uint nb = p.in_f >> 4;                                                                        \
    ulong row16 = (ulong)o * nb;                                                                  \
    device const float* xr = x + (ulong)r * p.in_f;                                               \
    float acc = 0.0f;                                                                             \
    for (uint b = lane; b < nb; b += 32u) {                                                       \
        ulong bi = row16 + b;                                                                     \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        device const float* xb = xr + (b << 4);                                                   \
        for (uint k = 0; k < 16u; k++) acc += xb[k] * wk[k];                                      \
    }                                                                                             \
    acc = simd_sum(acc);                                                                          \
    if (lane == 0u) dst[sg] = acc;                                                                \
}

// Row-tiled (1 < m < 16, or shapes the GEMM kernel can't take): one simdgroup per (8-row tile,
// output); each lane decodes a block once and applies it to all rows.
#define RT_KERNEL(NAME, DEC)                                                                      \
kernel void NAME(device const float*  x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    uint sg = gid / 32u;                                                                          \
    uint ntile = (p.m + 7u) / 8u;                                                                 \
    if (sg >= ntile * p.out_f) return;                                                            \
    uint t = sg / p.out_f;                                                                        \
    uint o = sg % p.out_f;                                                                        \
    uint r0 = t * 8u;                                                                             \
    uint rm = min(8u, p.m - r0);                                                                  \
    uint nb = p.in_f >> 4;                                                                        \
    ulong row16 = (ulong)o * nb;                                                                  \
    float acc[8];                                                                                 \
    for (uint rr = 0; rr < 8u; rr++) acc[rr] = 0.0f;                                              \
    for (uint b = lane; b < nb; b += 32u) {                                                       \
        ulong bi = row16 + b;                                                                     \
        float wk[16];                                                                             \
        DEC(wk)                                                                                   \
        for (uint rr = 0; rr < rm; rr++) {                                                        \
            device const float* xb = x + (ulong)(r0 + rr) * p.in_f + (b << 4);                    \
            for (uint k = 0; k < 16u; k++) acc[rr] += xb[k] * wk[k];                              \
        }                                                                                         \
    }                                                                                             \
    for (uint rr = 0; rr < rm; rr++) {                                                            \
        float v = simd_sum(acc[rr]);                                                              \
        if (lane == 0u) dst[(ulong)(r0 + rr) * p.out_f + o] = v;                                  \
    }                                                                                             \
}


// ---- Strided row copy (the fused-QKV prefill split: one wide GEMM output sliced into q/k/v).
// Pure data movement, but on-device — a host copy here would round-trip the [m, qkv] buffer and
// break the command buffer mid-forward.
struct CopyStridedParams {
    uint src_off;
    uint src_stride;
    uint dst_off;
    uint dst_stride;
    uint rows;
    uint n;
};
kernel void copy_strided_f32(device const float*         src [[buffer(0)]],
                             device float*               dst [[buffer(1)]],
                             constant CopyStridedParams& p   [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.rows * p.n) return;
    uint r = gid / p.n, c = gid % p.n;
    dst[p.dst_off + r * p.dst_stride + c] = src[p.src_off + r * p.src_stride + c];
}

// ---- Cast f32 -> f16 (prefill GEMM feeds on half activations; round-to-nearest-even).
kernel void cast_f32_f16(device const float* src [[buffer(0)]],
                         device half*        dst [[buffer(1)]],
                         constant uint&      n   [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < n) dst[gid] = (half)src[gid];
}

// Half-fragment GEMM: same 8x16-tile shape as GEMM_KERNEL, but x and the staged weight tile are
// f16 and the MMAs run half x half -> float (Apple's mixed-precision simdgroup_multiply_accumulate,
// double the f32 matrix rate and half the staging bytes). Accumulation stays f32. This is the one
// path that is NOT bit-exact against the dequant reference: weights and activations round to f16
// (~5e-4 relative) — the llama.cpp trade, well under quantization error. Decode (GEMV) stays f32.
#define HGEMM_KERNEL(NAME, DEC)                                                                   \
kernel void NAME(device const half*   x     [[buffer(0)]],                                       \
                 device const uchar*  codes [[buffer(1)]],                                       \
                 device const uchar*  scm   [[buffer(2)]],                                       \
                 device const uchar*  dd    [[buffer(3)]],                                       \
                 device float*        dst   [[buffer(4)]],                                       \
                 constant QLinParams& p     [[buffer(5)]],                                       \
                 uint gid  [[thread_position_in_grid]],                                          \
                 uint lane [[thread_index_in_simdgroup]],                                        \
                 uint sgid [[simdgroup_index_in_threadgroup]]) {                                 \
    uint sg = gid / 32u;                                                                          \
    uint ntm = (p.m + 31u) / 32u;                                                                 \
    uint nto = p.out_f / 16u;                                                                     \
    if (sg >= ntm * nto) return;                                                                  \
    uint tm = sg / nto;                                                                           \
    uint to = sg % nto;                                                                           \
    uint r0 = tm * 32u;                                                                           \
    uint o0 = to * 16u;                                                                           \
    uint nb = p.in_f >> 4;                                                                        \
    threadgroup half wt[4][16 * 16];                                                              \
    if (r0 + 32u <= p.m) {                                                                        \
        simdgroup_float8x8 acc[4][2];                                                             \
        for (uint i = 0; i < 4u; i++)                                                             \
            for (uint jx = 0; jx < 2u; jx++) acc[i][jx] = simdgroup_float8x8(0.0f);               \
        for (uint kb = 0; kb < nb; kb++) {                                                        \
            if (lane < 16u) {                                                                     \
                ulong bi = (ulong)(o0 + lane) * nb + kb;                                          \
                float wk[16];                                                                     \
                DEC(wk)                                                                           \
                for (uint k2 = 0; k2 < 16u; k2++) wt[sgid][lane * 16u + k2] = (half)wk[k2];       \
            }                                                                                     \
            simdgroup_barrier(mem_flags::mem_threadgroup);                                        \
            for (uint kh = 0; kh < 2u; kh++) {                                                    \
                simdgroup_half8x8 wb0, wb1;                                                       \
                simdgroup_load(wb0, &wt[sgid][kh * 8u], 16, ulong2(0, 0), true);                  \
                simdgroup_load(wb1, &wt[sgid][128u + kh * 8u], 16, ulong2(0, 0), true);           \
                for (uint rh = 0; rh < 4u; rh++) {                                                \
                    simdgroup_half8x8 xa;                                                         \
                    device const half* xp =                                                       \
                        x + (ulong)(r0 + rh * 8u) * p.in_f + ((ulong)kb << 4) + kh * 8u;          \
                    simdgroup_load(xa, xp, p.in_f);                                               \
                    simdgroup_multiply_accumulate(acc[rh][0], xa, wb0, acc[rh][0]);               \
                    simdgroup_multiply_accumulate(acc[rh][1], xa, wb1, acc[rh][1]);               \
                }                                                                                 \
            }                                                                                     \
            simdgroup_barrier(mem_flags::mem_threadgroup);                                        \
        }                                                                                         \
        for (uint rh = 0; rh < 4u; rh++)                                                          \
            for (uint oh = 0; oh < 2u; oh++)                                                      \
                simdgroup_store(acc[rh][oh],                                                      \
                                dst + (ulong)(r0 + rh * 8u) * p.out_f + o0 + oh * 8u, p.out_f);   \
    } else {                                                                                      \
        /* partial row tile: scalar dot per (row, output) element */                              \
        for (uint e = lane; e < 512u; e += 32u) {                                                 \
            uint rr = r0 + e / 16u;                                                               \
            uint o = o0 + (e % 16u);                                                              \
            if (rr >= p.m) continue;                                                              \
            float a2 = 0.0f;                                                                      \
            for (uint kb = 0; kb < nb; kb++) {                                                    \
                ulong bi = (ulong)o * nb + kb;                                                    \
                float wk[16];                                                                     \
                DEC(wk)                                                                           \
                device const half* xb = x + (ulong)rr * p.in_f + ((ulong)kb << 4);                \
                for (uint k2 = 0; k2 < 16u; k2++) a2 += (float)xb[k2] * (float)((half)wk[k2]);    \
            }                                                                                     \
            dst[(ulong)rr * p.out_f + o] = a2;                                                    \
        }                                                                                         \
    }                                                                                             \
}

// Shared GEMV epilogue: reduce each row's partial across the simdgroup, then — when the k-dim
// is SPLIT across NSG simdgroups (the ks variants; small/mid GEMVs underfill the GPU with one
// simdgroup per row group and serialize the whole k-loop) — across the threadgroup via `red`
// (NSG * R floats), simdgroup 0 folding and writing with the EPI contract. NSG == 1 keeps the
// original single-simdgroup store; the branch is compile-time.
// R = rows per group; LIM = live-row bound (R or a clamped nrows).
#define GEMV_EPILOGUE_N(R, LIM)                                                                   \
    for (uint row = 0; row < (LIM) && first_row + row < p.out_f; row++)                           \
        sumf[row] = simd_sum(sumf[row]);                                                          \
    if (NSG == 1u) {                                                                              \
        if (lane == 0u) {                                                                         \
            for (uint row = 0; row < (LIM) && first_row + row < p.out_f; row++) {                 \
                uint o = first_row + row;                                                         \
                float s = sumf[row];                                                              \
                if (EPI == 2)      dst[o] = (zeroacc ? 0.0f : dst[o]) + wgt * s;                  \
                else if (EPI == 1) dst[o] = s + res[o];                                           \
                else               dst[o] = s;                                                    \
            }                                                                                     \
        }                                                                                         \
    } else {                                                                                      \
        if (lane == 0u)                                                                           \
            for (uint row = 0; row < (R); row++) red[sgitg * (R) + row] = sumf[row];              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
        if (sgitg == 0 && lane == 0u) {                                                           \
            for (uint row = 0; row < (LIM) && first_row + row < p.out_f; row++) {                 \
                float s = 0.0f;                                                                   \
                for (uint g = 0; g < NSG; g++) s += red[g * (R) + row];                           \
                uint o = first_row + row;                                                         \
                if (EPI == 2)      dst[o] = (zeroacc ? 0.0f : dst[o]) + wgt * s;                  \
                else if (EPI == 1) dst[o] = s + res[o];                                           \
                else               dst[o] = s;                                                    \
            }                                                                                     \
        }                                                                                         \
    }
#define GEMV_EPILOGUE(R) GEMV_EPILOGUE_N(R, R)

// MULTI-ROW mul_mv GEMV (m = 2..8: speculative verify's candidate rows, short suffix
// prefills): the single-row bodies below re-stream the whole weight once PER TOKEN if simply
// looped, and the cooperative GEMM tile is latency-bound on its serial k-loop at these sizes
// (measured ~41 GiB/s effective vs the GEMV's ~127). This keeps the mul_mv access pattern,
// hoists each (block, out-row)'s weight bytes into registers, and loops up to 4 token rows
// inside — the weight streams from DRAM once per 4 tokens, the activations re-read from L1.
// Grid = out-row-pairs x token-blocks; exact f32 like the other GEMVs (reassociated dot only).
template<uint MR, typename PT>
inline void linear_q4k_mr_body(device const float*  x,
                               device const uchar*  codes,
                               device float*        dst,
                               constant PT& p,
                               uint gid, uint lane) {
    uint sg = gid / 32u;
    uint rowgroups = (p.out_f + 1u) / 2u;
    uint tb = sg / rowgroups;
    sg %= rowgroups;
    uint first_row = sg * 2u;
    uint t0 = tb * MR;
    if (first_row >= p.out_f || t0 >= p.m) return;
    uint mt = min(MR, p.m - t0);
    uint nb = p.in_f >> 8;
    ulong row_b = (ulong)nb * 144ul;
    device const uchar* xr = codes + first_row * row_b;

    const ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    uint ix = lane >> 3;
    uint it = lane & 7u;
    uint iq = it >> 2;
    uint ir = it & 3u;

    float sumf[2][MR];
    for (uint r = 0; r < 2u; r++)
        for (uint t = 0; t < MR; t++) sumf[r][t] = 0.0f;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        device const uchar* blk = xr + (ulong)ib * 144ul;
        device const ushort* sc = (device const ushort*)(blk + 4u) + iq;
        device const ushort* q1 = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
        device const half* dh = (device const half*)blk;
        for (uint row = 0; row < 2u; row++) {
            // weight bytes -> registers, reused across the token loop
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            ushort q1v[4], q2v[4];
            for (uint i = 0; i < 4u; i++) {
                q1v[i] = q1[i];
                q2v[i] = q1[i + 32u];
            }
            float d0 = (float)dh[0];
            float d1 = (float)dh[1];
            for (uint t = 0; t < mt; t++) {
                device const float* y4 = x + (t0 + t) * p.in_f + ib * 256u + 64u * iq + 8u * ir;
                float4 sumy = float4(0.0f);
                float yl[16], yh[16];
                for (uint i = 0; i < 8u; i++) {
                    yl[i]      = y4[i];        sumy[0] += yl[i];
                    yl[i + 8u] = y4[i + 32u];  sumy[1] += yl[i + 8u];
                    yh[i]      = y4[i + 128u]; sumy[2] += yh[i];
                    yh[i + 8u] = y4[i + 160u]; sumy[3] += yh[i + 8u];
                }
                float4 acc1 = float4(0.0f);
                float4 acc2 = float4(0.0f);
                for (uint i = 0; i < 4u; i++) {
                    acc1[0] += yl[2u * i]      * (float)(q1v[i] & 0x000F);
                    acc1[1] += yl[2u * i + 1u] * (float)(q1v[i] & 0x0F00);
                    acc1[2] += yl[2u * i + 8u] * (float)(q1v[i] & 0x00F0);
                    acc1[3] += yl[2u * i + 9u] * (float)(q1v[i] & 0xF000);
                    acc2[0] += yh[2u * i]      * (float)(q2v[i] & 0x000F);
                    acc2[1] += yh[2u * i + 1u] * (float)(q2v[i] & 0x0F00);
                    acc2[2] += yh[2u * i + 8u] * (float)(q2v[i] & 0x00F0);
                    acc2[3] += yh[2u * i + 9u] * (float)(q2v[i] & 0xF000);
                }
                sumf[row][t] += d0 * ((acc1[0] + 1.0f/256.0f * acc1[1]) * sc8[0] +
                                      (acc1[2] + 1.0f/256.0f * acc1[3]) * sc8[1] * 1.0f/16.0f +
                                      (acc2[0] + 1.0f/256.0f * acc2[1]) * sc8[4] +
                                      (acc2[2] + 1.0f/256.0f * acc2[3]) * sc8[5] * 1.0f/16.0f) -
                                d1 * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
            q1 += row_b / 2u;
            sc += row_b / 2u;
            dh += row_b / 2u;
        }
    }
    for (uint row = 0; row < 2u && first_row + row < p.out_f; row++) {
        for (uint t = 0; t < mt; t++) {
            float v = simd_sum(sumf[row][t]);
            if (lane == 0u) dst[(ulong)(t0 + t) * p.out_f + first_row + row] = v;
        }
    }
}

kernel void linear_q4k_mr(device const float*  x     [[buffer(0)]],
                          device const uchar*  codes [[buffer(1)]],
                          device const uchar*  scm   [[buffer(2)]],
                          device const uchar*  dd    [[buffer(3)]],
                          device float*        dst   [[buffer(4)]],
                          constant QLinParams& p     [[buffer(5)]],
                          uint gid  [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    linear_q4k_mr_body<8>(x, codes, dst, p, gid, lane);
}

// Q6_K multi-row: same register-hoisting structure over `linear_q6k_body`'s access pattern.
template<uint MR, typename PT>
inline void linear_q6k_mr_body(device const float*  x,
                               device const uchar*  codes,
                               device float*        dst,
                               constant PT& p,
                               uint gid, uint lane) {
    uint sg = gid / 32u;
    uint rowgroups = (p.out_f + 1u) / 2u;
    uint tb = sg / rowgroups;
    sg %= rowgroups;
    uint first_row = sg * 2u;
    uint t0 = tb * MR;
    if (first_row >= p.out_f || t0 >= p.m) return;
    uint mt = min(MR, p.m - t0);
    uint nb = p.in_f >> 8;
    ulong row_b = (ulong)nb * 210ul;
    device const uchar* xr = codes + first_row * row_b;

    const uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    uint tid2 = lane >> 1;
    uint ix = lane & 1u;
    uint ip = tid2 >> 3;
    uint il = tid2 & 7u;
    uint l0 = 4u * il;
    uint is = 8u * ip + (l0 >> 4);
    uint y_off = 128u * ip + l0;
    uint ql_off = 64u * ip + l0;
    uint qh_off = 32u * ip + l0;

    float sumf[2][MR];
    for (uint r = 0; r < 2u; r++)
        for (uint t = 0; t < MR; t++) sumf[r][t] = 0.0f;

    for (uint i = ix; i < nb; i += 2u) {
        device const uchar* blk = xr + (ulong)i * 210ul;
        device const uchar* q1 = blk + ql_off;
        device const uchar* q2 = q1 + 32u;
        device const uchar* qh = blk + 128u + qh_off;
        device const char* sc = (device const char*)(blk + 192u) + is;
        device const half* dh = (device const half*)(blk + 208u);

        for (uint row = 0; row < 2u; row++) {
            uchar q1v[4], q2v[4], qhv[4];
            char scv[4];
            for (uint l = 0; l < 4u; l++) {
                q1v[l] = q1[l];
                q2v[l] = q2[l];
                qhv[l] = qh[l];
            }
            scv[0] = sc[0];
            scv[1] = sc[2];
            scv[2] = sc[4];
            scv[3] = sc[6];
            float d = (float)dh[0];
            for (uint t = 0; t < mt; t++) {
                device const float* y = x + (t0 + t) * p.in_f + i * 256u + y_off;
                float4 sums = float4(0.0f);
                for (uint l = 0; l < 4u; l++) {
                    sums[0] += y[l]       * (float)((char)((q1v[l] & 0xF) | ((qhv[l] & kmask1) << 4)) - 32);
                    sums[1] += y[l + 32u] * (float)((char)((q2v[l] & 0xF) | ((qhv[l] & kmask2) << 2)) - 32);
                    sums[2] += y[l + 64u] * (float)((char)((q1v[l] >> 4)  | ((qhv[l] & kmask3) << 0)) - 32);
                    sums[3] += y[l + 96u] * (float)((char)((q2v[l] >> 4)  | ((qhv[l] & kmask4) >> 2)) - 32);
                }
                sumf[row][t] += d * (sums[0] * scv[0] + sums[1] * scv[1] +
                                     sums[2] * scv[2] + sums[3] * scv[3]);
            }
            q1 += row_b;
            q2 += row_b;
            qh += row_b;
            sc += row_b;
            dh += row_b / 2u;
        }
    }
    for (uint row = 0; row < 2u && first_row + row < p.out_f; row++) {
        for (uint t = 0; t < mt; t++) {
            float v = simd_sum(sumf[row][t]);
            if (lane == 0u) dst[(ulong)(t0 + t) * p.out_f + first_row + row] = v;
        }
    }
}

kernel void linear_q6k_mr(device const float*  x     [[buffer(0)]],
                          device const uchar*  codes [[buffer(1)]],
                          device const uchar*  scm   [[buffer(2)]],
                          device const uchar*  dd    [[buffer(3)]],
                          device float*        dst   [[buffer(4)]],
                          constant QLinParams& p     [[buffer(5)]],
                          uint gid  [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    linear_q6k_mr_body<8>(x, codes, dst, p, gid, lane);
}

// Decode GEMV for the native K-quant formats, mul_mv shape (ported from llama.cpp's
// kernel_mul_mv_q4_K_f32 / q6_K and adapted to our buffers): each simdgroup computes TWO output
// rows; activations load once into registers and are reused across both rows, and the inner loop
// is decode-free — masked integer nibble ops with the block scales applied once per group (Q4_K
// splits the affine min out via pre-summed activations; the 1/256 and 1/16 factors are exact
// power-of-two corrections for the high-nibble/high-byte lanes). Algebraically identical to the
// reference dequant dot, floating-point reassociated. This access pattern (4 blocks in flight per
// simdgroup for Q4_K, contiguous 8-element runs per lane) is what the sub-block-scatter DEC16
// GEMV left on the table.
// Body shared by the plain GEMV and the fused-residual variant (`FADD`: dst = W·x + res, the
// decode o_proj/down_proj + Add peephole — one dispatch and no sublayer-output round-trip).
// EPI epilogue modes: 0 = dst = s; 1 = dst = s + res (fused residual Add); 2 = MoE accumulate,
// dst = (zeroacc ? 0 : dst) + wgt*s (the weighted expert sum, first expert zeroes).
template<int EPI, uint NSG, typename PT>
inline void linear_q4k_body(device const float*  x,
                            device const uchar*  codes,
                            device float*        dst,
                            device const float*  res,
                            constant PT& p,
                            float wgt, bool zeroacc,
                            uint gid, uint lane,
                            ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 2u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 8;                 // 256-element blocks per row
    ulong row_b = (ulong)nb * 144ul;       // row stride in bytes
    device const uchar* xr = codes + first_row * row_b;

    const ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    uint ix = lane >> 3;                   // 0..3: which of 4 blocks in flight
    uint it = lane & 7u;
    uint iq = it >> 2;                     // 0/1: which 128-half
    uint ir = it & 3u;                     // 0..3: which 8-element run

    float yl[16], yh[16];
    float sumf[2] = {0.0f, 0.0f};
    device const float* y4 = x + (sgitg * 4u + ix) * 256u + 64u * iq + 8u * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = sgitg * 4u + ix; ib < nb; ib += NSG * 4u) {
        float4 sumy = float4(0.0f);
        for (uint i = 0; i < 8u; i++) {
            yl[i]      = y4[i];        sumy[0] += yl[i];
            yl[i + 8u] = y4[i + 32u];  sumy[1] += yl[i + 8u];
            yh[i]      = y4[i + 128u]; sumy[2] += yh[i];
            yh[i + 8u] = y4[i + 160u]; sumy[3] += yh[i + 8u];
        }
        device const uchar* blk = xr + (ulong)ib * 144ul;
        device const ushort* sc = (device const ushort*)(blk + 4u) + iq;
        device const ushort* q1 = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
        device const half* dh = (device const half*)blk;

        for (uint row = 0; row < 2u; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const ushort* q2 = q1 + 32u;
            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            for (uint i = 0; i < 4u; i++) {
                acc1[0] += yl[2u * i]      * (float)(q1[i] & 0x000F);
                acc1[1] += yl[2u * i + 1u] * (float)(q1[i] & 0x0F00);
                acc1[2] += yl[2u * i + 8u] * (float)(q1[i] & 0x00F0);
                acc1[3] += yl[2u * i + 9u] * (float)(q1[i] & 0xF000);
                acc2[0] += yh[2u * i]      * (float)(q2[i] & 0x000F);
                acc2[1] += yh[2u * i + 1u] * (float)(q2[i] & 0x0F00);
                acc2[2] += yh[2u * i + 8u] * (float)(q2[i] & 0x00F0);
                acc2[3] += yh[2u * i + 9u] * (float)(q2[i] & 0xF000);
            }
            sumf[row] += (float)dh[0] * ((acc1[0] + 1.0f/256.0f * acc1[1]) * sc8[0] +
                                         (acc1[2] + 1.0f/256.0f * acc1[3]) * sc8[1] * 1.0f/16.0f +
                                         (acc2[0] + 1.0f/256.0f * acc2[1]) * sc8[4] +
                                         (acc2[2] + 1.0f/256.0f * acc2[3]) * sc8[5] * 1.0f/16.0f) -
                         (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                         sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += row_b / 2u;
            sc += row_b / 2u;
            dh += row_b / 2u;
        }
        y4 += NSG * 4u * 256u;
    }
    GEMV_EPILOGUE(2u)
}

kernel void linear_q4k(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2];
    linear_q4k_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q4k_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2];
    linear_q4k_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}
// k-split variant: 4 simdgroups share the row group, each covering a strided quarter of the
// k-dim blocks — 4x the threadgroups on small/mid GEMVs and a quarter of the serial chain.
kernel void linear_q4k_ks(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 2];
    linear_q4k_body<0, 4>(x, codes, dst, x, p, 0.0f, false, gid, lane, sgitg, red);
}
kernel void linear_q4k_ks_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 2];
    linear_q4k_body<1, 4>(x, codes, dst, res, p, 0.0f, false, gid, lane, sgitg, red);
}

template<int EPI, uint NSG, typename PT>
inline void linear_q6k_body(device const float*  x,
                            device const uchar*  codes,
                            device float*        dst,
                            device const float*  res,
                            constant PT& p,
                            float wgt, bool zeroacc,
                            uint gid, uint lane,
                            ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 2u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 8;
    ulong row_b = (ulong)nb * 210ul;
    device const uchar* xr = codes + first_row * row_b;

    const uchar kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    uint tid2 = lane >> 1;
    uint ix = lane & 1u;                   // 0/1: which of 2 blocks in flight
    uint ip = tid2 >> 3;                   // 0/1: which 128-half
    uint il = tid2 & 7u;
    uint l0 = 4u * il;
    uint is = 8u * ip + (l0 >> 4);
    uint y_off = 128u * ip + l0;
    uint ql_off = 64u * ip + l0;
    uint qh_off = 32u * ip + l0;

    float sumf[2] = {0.0f, 0.0f};
    float yl[16];

    for (uint i = sgitg * 2u + ix; i < nb; i += NSG * 2u) {
        device const uchar* blk = xr + (ulong)i * 210ul;
        device const uchar* q1 = blk + ql_off;
        device const uchar* q2 = q1 + 32u;
        device const uchar* qh = blk + 128u + qh_off;
        device const char* sc = (device const char*)(blk + 192u) + is;
        device const half* dh = (device const half*)(blk + 208u);
        device const float* y = x + i * 256u + y_off;

        for (uint l = 0; l < 4u; l++) {
            yl[4u * l]      = y[l];
            yl[4u * l + 1u] = y[l + 32u];
            yl[4u * l + 2u] = y[l + 64u];
            yl[4u * l + 3u] = y[l + 96u];
        }
        for (uint row = 0; row < 2u; row++) {
            float4 sums = float4(0.0f);
            for (uint l = 0; l < 4u; l++) {
                sums[0] += yl[4u * l]      * (float)((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4u * l + 1u] * (float)((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4u * l + 2u] * (float)((char)((q1[l] >> 4)  | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4u * l + 3u] * (float)((char)((q2[l] >> 4)  | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += (float)dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                         sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += row_b;
            q2 += row_b;
            qh += row_b;
            sc += row_b;
            dh += row_b / 2u;
        }
    }
    GEMV_EPILOGUE(2u)
}

kernel void linear_q6k(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2];
    linear_q6k_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q6k_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2];
    linear_q6k_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}
// k-split variant: 4 simdgroups share the row group, each covering a strided quarter of the
// k-dim blocks — 4x the threadgroups on small/mid GEMVs and a quarter of the serial chain.
kernel void linear_q6k_ks(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 2];
    linear_q6k_body<0, 4>(x, codes, dst, x, p, 0.0f, false, gid, lane, sgitg, red);
}
kernel void linear_q6k_ks_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 2];
    linear_q6k_body<1, 4>(x, codes, dst, res, p, 0.0f, false, gid, lane, sgitg, red);
}

// Token-parallel variant: one simdgroup per (row pair, token) running the SINGLE-row mul_mv
// body — token is the FASTEST-varying grid index, so the m simdgroups sharing a row pair are
// scheduled near-simultaneously and their weight-block reads coalesce in the cache hierarchy
// instead of multiplying DRAM traffic. (The register-reuse variant above serializes the token
// loop inside each simdgroup — ALU-bound, ~24 ms per extra row on an 8B verify.)
kernel void linear_q4k_mrv(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           constant QLinParams& p     [[buffer(5)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    uint t = sg % p.m;
    uint rp = sg / p.m;
    threadgroup float red[2];
    linear_q4k_body<0, 1>(
        x + (ulong)t * p.in_f, codes, dst + (ulong)t * p.out_f, dst, p, 0.0f, false,
        rp * 32u + lane, lane, 0, red);
}
kernel void linear_q6k_mrv(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           constant QLinParams& p     [[buffer(5)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint sg = gid / 32u;
    uint t = sg % p.m;
    uint rp = sg / p.m;
    threadgroup float red[2];
    linear_q6k_body<0, 1>(
        x + (ulong)t * p.in_f, codes, dst + (ulong)t * p.out_f, dst, p, 0.0f, false,
        rp * 32u + lane, lane, 0, red);
}

// Decode GEMV for NATIVE IQ4_NL (18 B / 32-element blocks). Each simdgroup computes four
// output rows and reuses its eight activation values across them. The lane's eight nibbles are
// loaded as four aligned ushorts and unpacked in registers; only the codebook lookups remain
// scalar because Metal has no integer packed-dot intrinsic for the non-linear values.
template<int EPI, uint NSG, typename PT>
inline void linear_iq4nl_body(device const float*  x,
                              device const uchar*  codes,
                              device float*        dst,
                              device const float*  res,
                              constant PT& p,
                              float wgt, bool zeroacc,
                              uint gid, uint lane,
                              ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 4u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 5;
    ulong row_b = (ulong)nb * 18ul;
    uint nrows = min(4u, p.out_f - first_row);

    uint ix = lane >> 2;
    uint il = (lane & 3u) * 8u;
    bool hi = il >= 16u;
    uint jq = hi ? il - 16u : il;

    float yl[8];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* yb = x + (sgitg * 8u + ix) * 32u + il;

    for (uint ib = sgitg * 8u + ix; ib < nb; ib += NSG * 8u) {
        for (uint i = 0; i < 8u; i++) yl[i] = yb[i];
        for (uint row = 0; row < 4u; row++) {
            device const uchar* blk =
                codes + (ulong)(first_row + min(row, nrows - 1u)) * row_b + (ulong)ib * 18ul;
            device const ushort* b16 = (device const ushort*)blk;
            device const ushort* qsp = (device const ushort*)(blk + 2u + jq);
            uint w0 = (uint)qsp[0] | ((uint)qsp[1] << 16);
            uint w1 = (uint)qsp[2] | ((uint)qsp[3] << 16);
            uint q40 = hi ? (w0 >> 4) & 0x0F0F0F0Fu : w0 & 0x0F0F0F0Fu;
            uint q41 = hi ? (w1 >> 4) & 0x0F0F0F0Fu : w1 & 0x0F0F0F0Fu;
            float sumq = kvalues_iq4nl_f[(q40 >> 0u) & 0xFu] * yl[0]
                + kvalues_iq4nl_f[(q40 >> 8u) & 0xFu] * yl[1]
                + kvalues_iq4nl_f[(q40 >> 16u) & 0xFu] * yl[2]
                + kvalues_iq4nl_f[(q40 >> 24u) & 0xFu] * yl[3]
                + kvalues_iq4nl_f[(q41 >> 0u) & 0xFu] * yl[4]
                + kvalues_iq4nl_f[(q41 >> 8u) & 0xFu] * yl[5]
                + kvalues_iq4nl_f[(q41 >> 16u) & 0xFu] * yl[6]
                + kvalues_iq4nl_f[(q41 >> 24u) & 0xFu] * yl[7];
            sumf[row] += (float)as_type<half>(b16[0]) * sumq;
        }
        yb += NSG * 8u * 32u;
    }
    GEMV_EPILOGUE_N(4u, nrows)
}

kernel void linear_iq4nl(device const float*  x     [[buffer(0)]],
                         device const uchar*  codes [[buffer(1)]],
                         device const uchar*  scm   [[buffer(2)]],
                         device const uchar*  dd    [[buffer(3)]],
                         device float*        dst   [[buffer(4)]],
                         constant QLinParams& p     [[buffer(5)]],
                         uint gid  [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_iq4nl_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}

kernel void linear_iq4nl_add(device const float*  x     [[buffer(0)]],
                             device const uchar*  codes [[buffer(1)]],
                             device const uchar*  scm   [[buffer(2)]],
                             device const uchar*  dd    [[buffer(3)]],
                             device float*        dst   [[buffer(4)]],
                             device const float*  res   [[buffer(5)]],
                             constant QLinParams& p     [[buffer(6)]],
                             uint gid  [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_iq4nl_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}

// Decode GEMV for NATIVE Q8_0 (34 B / 32-elem blocks), mul_mv shape (ported from llama.cpp's
// kernel_mul_mv_q8_0_f32): each simdgroup computes FOUR output rows (N_R0_Q8_0); lanes fold as
// 8 blocks in flight x 4 threads per block (8 quants each), activations load once into registers
// and are reused across all four rows, and the inner product is d * sum(q*y) — the exact
// dequantize_q8_0 product, reassociated. The stream is the raw 8.5 bpw weight — the factored
// quik8 form paid ~10.2 bpw for the same values, and decode GEMV is bound on exactly this stream.
// Same EPI epilogue contract as `linear_q4k_body`. Tail rows clamp their pointer into the weight
// (their sums are discarded at the write).
template<int EPI, uint NSG, typename PT>
inline void linear_q8_0_body(device const float*  x,
                             device const uchar*  codes,
                             device float*        dst,
                             device const float*  res,
                             constant PT& p,
                             float wgt, bool zeroacc,
                             uint gid, uint lane,
                             ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 4u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 5;                 // 32-element blocks per row
    ulong row_b = (ulong)nb * 34ul;        // row stride in bytes
    uint nrows = min(4u, p.out_f - first_row);

    uint ix = lane >> 2;                   // 0..7: which of 8 blocks in flight
    uint il = (lane & 3u) * 8u;            // this thread's 8 quants within the block

    float yl[8];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* yb = x + (sgitg * 8u + ix) * 32u + il;

    for (uint ib = sgitg * 8u + ix; ib < nb; ib += NSG * 8u) {
        for (uint i = 0; i < 8u; i++) yl[i] = yb[i];
        for (uint row = 0; row < 4u; row++) {
            device const uchar* blk =
                codes + (ulong)(first_row + min(row, nrows - 1u)) * row_b + (ulong)ib * 34ul;
            device const char* qs = (device const char*)(blk + 2u) + il;
            float sumq = 0.0f;
            for (uint i = 0; i < 8u; i++) sumq += (float)qs[i] * yl[i];
            sumf[row] += sumq * (float)as_type<half>((ushort)(blk[0] | ((ushort)blk[1] << 8)));
        }
        yb += NSG * 8u * 32u;
    }
    GEMV_EPILOGUE_N(4u, nrows)
}

kernel void linear_q8_0(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q8_0_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q8_0_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q8_0_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}
// k-split variant: 4 simdgroups share the row group, each covering a strided quarter of the
// k-dim blocks — 4x the threadgroups on small/mid GEMVs and a quarter of the serial chain.
kernel void linear_q8_0_ks(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q8_0_body<0, 4>(x, codes, dst, x, p, 0.0f, false, gid, lane, sgitg, red);
}
kernel void linear_q8_0_ks_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q8_0_body<1, 4>(x, codes, dst, res, p, 0.0f, false, gid, lane, sgitg, red);
}

// Decode GEMV for NATIVE Q4_0 (18 B / 32-elem blocks) — `linear_q5_0_body` minus the high-bit
// stream: 4 rows per simdgroup, 8 blocks in flight, nibbles masked into byte lanes for the
// masked-FMA dot with exact 1/256-power folding, the -8 offset factored through sumy.
template<int EPI, uint NSG, typename PT>
inline void linear_q4_0_body(device const float*  x,
                             device const uchar*  codes,
                             device float*        dst,
                             device const float*  res,
                             constant PT& p,
                             float wgt, bool zeroacc,
                             uint gid, uint lane,
                             ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 4u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 5;
    ulong row_b = (ulong)nb * 18ul;
    uint nrows = min(4u, p.out_f - first_row);

    uint ix = lane >> 2;
    uint il = (lane & 3u) * 8u;
    bool hi = il >= 16u;
    uint jq = hi ? il - 16u : il;

    float yl[8];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* yb = x + (sgitg * 8u + ix) * 32u + il;

    for (uint ib = sgitg * 8u + ix; ib < nb; ib += NSG * 8u) {
        float sumy = 0.0f;
        for (uint i = 0; i < 8u; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }
        for (uint row = 0; row < 4u; row++) {
            device const uchar* blk =
                codes + (ulong)(first_row + min(row, nrows - 1u)) * row_b + (ulong)ib * 18ul;
            device const ushort* b16 = (device const ushort*)blk;
            device const ushort* qsp = (device const ushort*)(blk + 2u + jq);
            uint w0 = (uint)qsp[0] | ((uint)qsp[1] << 16);
            uint w1 = (uint)qsp[2] | ((uint)qsp[3] << 16);
            uint q40 = hi ? (w0 >> 4) & 0x0F0F0F0Fu : w0 & 0x0F0F0F0Fu;
            uint q41 = hi ? (w1 >> 4) & 0x0F0F0F0Fu : w1 & 0x0F0F0F0Fu;
            float4 acc;
            acc.x = (float)(q40 & 0x000000FFu) * yl[0] + (float)(q41 & 0x000000FFu) * yl[4];
            acc.y = (float)(q40 & 0x0000FF00u) * yl[1] + (float)(q41 & 0x0000FF00u) * yl[5];
            acc.z = (float)(q40 & 0x00FF0000u) * yl[2] + (float)(q41 & 0x00FF0000u) * yl[6];
            acc.w = (float)(q40 & 0xFF000000u) * yl[3] + (float)(q41 & 0xFF000000u) * yl[7];
            float d = (float)as_type<half>(b16[0]);
            sumf[row] += d
                * (acc.x + acc.y * (1.0f / 256.0f) + acc.z * (1.0f / 65536.0f)
                    + acc.w * (1.0f / 16777216.0f) - 8.0f * sumy);
        }
        yb += NSG * 8u * 32u;
    }
    GEMV_EPILOGUE_N(4u, nrows)
}

kernel void linear_q4_0(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q4_0_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q4_0_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q4_0_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q4_0_ks(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q4_0_body<0, 4>(x, codes, dst, x, p, 0.0f, false, gid, lane, sgitg, red);
}
kernel void linear_q4_0_ks_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q4_0_body<1, 4>(x, codes, dst, res, p, 0.0f, false, gid, lane, sgitg, red);
}

// Decode GEMV for NATIVE Q4_1 (20 B / 32-elem blocks) — the AFFINE sibling of Q4_0: identical
// masked-FMA structure, block stride 20 (extra f16 `m`), qs at byte 4, and value d*q + m — so the
// -8*sumy offset term becomes +m*sumy (Σ(d*q_i + m)*x_i = d*Σq_i*x_i + m*Σx_i). Bit-for-bit
// dequantize_row_q4_1 / vec_dot_q4_1_32_batch_scalar.
template<int EPI, uint NSG, typename PT>
inline void linear_q4_1_body(device const float*  x,
                             device const uchar*  codes,
                             device float*        dst,
                             device const float*  res,
                             constant PT& p,
                             float wgt, bool zeroacc,
                             uint gid, uint lane,
                             ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 4u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 5;
    ulong row_b = (ulong)nb * 20ul;
    uint nrows = min(4u, p.out_f - first_row);

    uint ix = lane >> 2;
    uint il = (lane & 3u) * 8u;
    bool hi = il >= 16u;
    uint jq = hi ? il - 16u : il;

    float yl[8];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* yb = x + (sgitg * 8u + ix) * 32u + il;

    for (uint ib = sgitg * 8u + ix; ib < nb; ib += NSG * 8u) {
        float sumy = 0.0f;
        for (uint i = 0; i < 8u; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }
        for (uint row = 0; row < 4u; row++) {
            device const uchar* blk =
                codes + (ulong)(first_row + min(row, nrows - 1u)) * row_b + (ulong)ib * 20ul;
            device const ushort* b16 = (device const ushort*)blk;
            device const ushort* qsp = (device const ushort*)(blk + 4u + jq);
            uint w0 = (uint)qsp[0] | ((uint)qsp[1] << 16);
            uint w1 = (uint)qsp[2] | ((uint)qsp[3] << 16);
            uint q40 = hi ? (w0 >> 4) & 0x0F0F0F0Fu : w0 & 0x0F0F0F0Fu;
            uint q41 = hi ? (w1 >> 4) & 0x0F0F0F0Fu : w1 & 0x0F0F0F0Fu;
            float4 acc;
            acc.x = (float)(q40 & 0x000000FFu) * yl[0] + (float)(q41 & 0x000000FFu) * yl[4];
            acc.y = (float)(q40 & 0x0000FF00u) * yl[1] + (float)(q41 & 0x0000FF00u) * yl[5];
            acc.z = (float)(q40 & 0x00FF0000u) * yl[2] + (float)(q41 & 0x00FF0000u) * yl[6];
            acc.w = (float)(q40 & 0xFF000000u) * yl[3] + (float)(q41 & 0xFF000000u) * yl[7];
            float d = (float)as_type<half>(b16[0]);
            float mn = (float)as_type<half>(b16[1]);
            sumf[row] += d
                * (acc.x + acc.y * (1.0f / 256.0f) + acc.z * (1.0f / 65536.0f)
                    + acc.w * (1.0f / 16777216.0f)) + mn * sumy;
        }
        yb += NSG * 8u * 32u;
    }
    GEMV_EPILOGUE_N(4u, nrows)
}

kernel void linear_q4_1(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q4_1_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q4_1_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q4_1_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q4_1_ks(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q4_1_body<0, 4>(x, codes, dst, x, p, 0.0f, false, gid, lane, sgitg, red);
}
kernel void linear_q4_1_ks_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q4_1_body<1, 4>(x, codes, dst, res, p, 0.0f, false, gid, lane, sgitg, red);
}

// Decode GEMV for NATIVE Q5_0 (22 B / 32-elem blocks), same mul_mv shape as `linear_q8_0_body`:
// four output rows per simdgroup, 8 blocks in flight x 4 threads per block (8 quants each),
// activations loaded once and reused across rows. Each thread's 8 elements sit in one nibble
// half (il in {0,8} low, {16,24} high), so the qh bit extraction is uniform per thread.
// wk = d * (q - 16), the exact dequantize_q5_0 value, reassociated over the dot only.
template<int EPI, uint NSG, typename PT>
inline void linear_q5_0_body(device const float*  x,
                             device const uchar*  codes,
                             device float*        dst,
                             device const float*  res,
                             constant PT& p,
                             float wgt, bool zeroacc,
                             uint gid, uint lane,
                             ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 4u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 5;                 // 32-element blocks per row
    ulong row_b = (ulong)nb * 22ul;        // row stride in bytes
    uint nrows = min(4u, p.out_f - first_row);

    uint ix = lane >> 2;                   // 0..7: which of 8 blocks in flight
    uint il = (lane & 3u) * 8u;            // this thread's 8 quants within the block
    bool hi = il >= 16u;
    uint jq = hi ? il - 16u : il;          // nibble byte offset within qs

    float yl[8];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* yb = x + (sgitg * 8u + ix) * 32u + il;

    for (uint ib = sgitg * 8u + ix; ib < nb; ib += NSG * 8u) {
        float sumy = 0.0f;
        for (uint i = 0; i < 8u; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }
        for (uint row = 0; row < 4u; row++) {
            device const uchar* blk =
                codes + (ulong)(first_row + min(row, nrows - 1u)) * row_b + (ulong)ib * 22ul;
            // 22-byte blocks keep every field on an even address: d and qh load as ushorts,
            // the thread's 8 nibble-source bytes as two 4-byte words — 5 loads per (row, block)
            // instead of 14 byte loads. Extraction stays in registers, q4k-body style: 4
            // elements live as the bytes of one uint (nibbles OR'd with the spread qh bits),
            // the dot runs on byte-masked floats with exact 1/256-power scale folding, and the
            // -16 offset factors out through sumy — no per-element subtract or shift chain.
            // (The qh bit for element e is bit e for BOTH nibble halves — the reference's two
            // expressions land on the same bit.)
            device const ushort* b16 = (device const ushort*)blk;
            uint qh = (uint)b16[1] | ((uint)b16[2] << 16);
            device const ushort* qsp = (device const ushort*)(blk + 6u + jq);
            uint w0 = (uint)qsp[0] | ((uint)qsp[1] << 16);
            uint w1 = (uint)qsp[2] | ((uint)qsp[3] << 16);
            uint n0 = hi ? (w0 >> 4) & 0x0F0F0F0Fu : w0 & 0x0F0F0F0Fu;
            uint n1 = hi ? (w1 >> 4) & 0x0F0F0F0Fu : w1 & 0x0F0F0F0Fu;
            uint hb = (qh >> il) & 0xFFu;   // this thread's 8 high bits (element i at bit i)
            uint q40 = n0 | ((hb & 1u) << 4) | ((hb & 2u) << 11) | ((hb & 4u) << 18)
                | ((hb & 8u) << 25);
            uint hb1 = hb >> 4;
            uint q41 = n1 | ((hb1 & 1u) << 4) | ((hb1 & 2u) << 11) | ((hb1 & 4u) << 18)
                | ((hb1 & 8u) << 25);
            float4 acc;
            acc.x = (float)(q40 & 0x000000FFu) * yl[0] + (float)(q41 & 0x000000FFu) * yl[4];
            acc.y = (float)(q40 & 0x0000FF00u) * yl[1] + (float)(q41 & 0x0000FF00u) * yl[5];
            acc.z = (float)(q40 & 0x00FF0000u) * yl[2] + (float)(q41 & 0x00FF0000u) * yl[6];
            acc.w = (float)(q40 & 0xFF000000u) * yl[3] + (float)(q41 & 0xFF000000u) * yl[7];
            float d = (float)as_type<half>(b16[0]);
            sumf[row] += d
                * (acc.x + acc.y * (1.0f / 256.0f) + acc.z * (1.0f / 65536.0f)
                    + acc.w * (1.0f / 16777216.0f) - 16.0f * sumy);
        }
        yb += NSG * 8u * 32u;
    }
    GEMV_EPILOGUE_N(4u, nrows)
}

kernel void linear_q5_0(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q5_0_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q5_0_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q5_0_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}
// k-split variant: 4 simdgroups share the row group, each covering a strided quarter of the
// k-dim blocks — 4x the threadgroups on small/mid GEMVs and a quarter of the serial chain.
kernel void linear_q5_0_ks(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q5_0_body<0, 4>(x, codes, dst, x, p, 0.0f, false, gid, lane, sgitg, red);
}
kernel void linear_q5_0_ks_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q5_0_body<1, 4>(x, codes, dst, res, p, 0.0f, false, gid, lane, sgitg, red);
}

// Decode GEMV for NATIVE Q5_1 (24 B / 32-elem blocks) — the AFFINE sibling of Q5_0: identical
// mul_mv shape and qh 5th-bit assembly, block stride 24 (extra f16 `m`), qh at byte 4, qs at byte
// 8, and value d*q + m — so the -16*sumy offset term becomes +m*sumy. Bit-for-bit
// dequantize_row_q5_1 / vec_dot_q5_1_32_batch_scalar.
template<int EPI, uint NSG, typename PT>
inline void linear_q5_1_body(device const float*  x,
                             device const uchar*  codes,
                             device float*        dst,
                             device const float*  res,
                             constant PT& p,
                             float wgt, bool zeroacc,
                             uint gid, uint lane,
                             ushort sgitg, threadgroup float* red) {
    uint first_row = (gid / (32u * NSG)) * 4u;
    if (first_row >= p.out_f) return;
    uint nb = p.in_f >> 5;                 // 32-element blocks per row
    ulong row_b = (ulong)nb * 24ul;        // row stride in bytes
    uint nrows = min(4u, p.out_f - first_row);

    uint ix = lane >> 2;                   // 0..7: which of 8 blocks in flight
    uint il = (lane & 3u) * 8u;            // this thread's 8 quants within the block
    bool hi = il >= 16u;
    uint jq = hi ? il - 16u : il;          // nibble byte offset within qs

    float yl[8];
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const float* yb = x + (sgitg * 8u + ix) * 32u + il;

    for (uint ib = sgitg * 8u + ix; ib < nb; ib += NSG * 8u) {
        float sumy = 0.0f;
        for (uint i = 0; i < 8u; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }
        for (uint row = 0; row < 4u; row++) {
            device const uchar* blk =
                codes + (ulong)(first_row + min(row, nrows - 1u)) * row_b + (ulong)ib * 24ul;
            // 24-byte blocks keep every field on an even address: d, m and qh load as ushorts, the
            // thread's nibble-source bytes as two 4-byte words. The qh 5th-bit spread and byte-
            // masked dot are IDENTICAL to Q5_0 (see there); only d/m, the byte offsets and the
            // affine `+ m*sumy` (in place of `- 16*sumy`) differ.
            device const ushort* b16 = (device const ushort*)blk;
            uint qh = (uint)b16[2] | ((uint)b16[3] << 16);
            device const ushort* qsp = (device const ushort*)(blk + 8u + jq);
            uint w0 = (uint)qsp[0] | ((uint)qsp[1] << 16);
            uint w1 = (uint)qsp[2] | ((uint)qsp[3] << 16);
            uint n0 = hi ? (w0 >> 4) & 0x0F0F0F0Fu : w0 & 0x0F0F0F0Fu;
            uint n1 = hi ? (w1 >> 4) & 0x0F0F0F0Fu : w1 & 0x0F0F0F0Fu;
            uint hb = (qh >> il) & 0xFFu;   // this thread's 8 high bits (element i at bit i)
            uint q40 = n0 | ((hb & 1u) << 4) | ((hb & 2u) << 11) | ((hb & 4u) << 18)
                | ((hb & 8u) << 25);
            uint hb1 = hb >> 4;
            uint q41 = n1 | ((hb1 & 1u) << 4) | ((hb1 & 2u) << 11) | ((hb1 & 4u) << 18)
                | ((hb1 & 8u) << 25);
            float4 acc;
            acc.x = (float)(q40 & 0x000000FFu) * yl[0] + (float)(q41 & 0x000000FFu) * yl[4];
            acc.y = (float)(q40 & 0x0000FF00u) * yl[1] + (float)(q41 & 0x0000FF00u) * yl[5];
            acc.z = (float)(q40 & 0x00FF0000u) * yl[2] + (float)(q41 & 0x00FF0000u) * yl[6];
            acc.w = (float)(q40 & 0xFF000000u) * yl[3] + (float)(q41 & 0xFF000000u) * yl[7];
            float d = (float)as_type<half>(b16[0]);
            float mn = (float)as_type<half>(b16[1]);
            sumf[row] += d
                * (acc.x + acc.y * (1.0f / 256.0f) + acc.z * (1.0f / 65536.0f)
                    + acc.w * (1.0f / 16777216.0f)) + mn * sumy;
        }
        yb += NSG * 8u * 32u;
    }
    GEMV_EPILOGUE_N(4u, nrows)
}

kernel void linear_q5_1(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q5_1_body<0, 1>(x, codes, dst, x, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q5_1_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4];
    linear_q5_1_body<1, 1>(x, codes, dst, res, p, 0.0f, false, gid, lane, 0, red);
}
kernel void linear_q5_1_ks(device const float*  x     [[buffer(0)]],
                       device const uchar*  codes [[buffer(1)]],
                       device const uchar*  scm   [[buffer(2)]],
                       device const uchar*  dd    [[buffer(3)]],
                       device float*        dst   [[buffer(4)]],
                       constant QLinParams& p     [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q5_1_body<0, 4>(x, codes, dst, x, p, 0.0f, false, gid, lane, sgitg, red);
}
kernel void linear_q5_1_ks_add(device const float*  x     [[buffer(0)]],
                           device const uchar*  codes [[buffer(1)]],
                           device const uchar*  scm   [[buffer(2)]],
                           device const uchar*  dd    [[buffer(3)]],
                           device float*        dst   [[buffer(4)]],
                           device const float*  res   [[buffer(5)]],
                           constant QLinParams& p     [[buffer(6)]],
                           uint gid  [[thread_position_in_grid]],
                           uint sgitg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 4];
    linear_q5_1_body<1, 4>(x, codes, dst, res, p, 0.0f, false, gid, lane, sgitg, red);
}
