// Shared native-block dequant library (raw GGUF blocks, padded to u32). Included by native_gemv
// (decode GEMV) and native_gemm (prefill tiled coopmat GEMM). The includer must declare the weight
// SSBO as `uint nw[]` (any binding). A -DFMT_* define selects one format. Two interfaces per format:
//   * dq(uint g)                      — single element g (global index into [N,K] = out*K + in)
//   * dqblk(uint gstart, out v[32])   — decode a contiguous 32-elem sub-block, scale decoded ONCE.
// dqblk is the amortized path; formats without an optimized one fall back to looping dq() (below).
// Grid-based i-quants pull their tables from native_grids.glsl.
//
// NW(i): the nw[] read chokepoint. An includer may pre-define NW to add a WORD-offset base to
// every weight read — the paged expert kernels (`native_gemv_id(_multi).comp`'s -DPAGED build) use
// this to relocate all dequant math into one arena slot: the within-expert ELEMENT offsets this
// library computes stay small (they fit u32 comfortably for any real expert), while the slot's
// arena position — whose ELEMENT/BYTE offset would overflow u32 past a few dozen ~14 MiB slots
// (u32 element math wrapped at slot ≥ ~102 on Scout, the task's coherent-but-wrong bug) — is
// carried as a u32 WORD index added only here, at the final indexing step (a 16 GiB-per-arena
// reach, enforced host-side; see `GpuPager`). Undefined = plain nw[i]: every non-paged build
// expands to exactly the code this library always had.
#ifndef NW
#define NW(i) nw[i]
#endif
// Wide 4-word read (see native_weight_addr.glsl). Non-streamed builds read the SSBO array directly;
// ACO's vectorizer already fuses the four adjacent nw[] loads into one buffer_load_b128, so this is
// the same code the scalar path generated — the macro only gives the streamed arena the same shape.
#ifndef NW4
#define NW4(wbase) uvec4(nw[(wbase)], nw[(wbase) + 1u], nw[(wbase) + 2u], nw[(wbase) + 3u])
#endif
// Wide 2-word read (see native_weight_addr.glsl's NW2) — non-streamed builds read the SSBO array
// directly; ACO's vectorizer fuses the adjacent pair into one buffer_load_b64.
#ifndef NW2
#define NW2(wbase) uvec2(nw[(wbase)], nw[(wbase) + 1u])
#endif

uint rb(uint bo) { return (NW(bo >> 2u) >> ((bo & 3u) << 3u)) & 0xFFu; }
uint ru16(uint bo) { return rb(bo) | (rb(bo + 1u) << 8u); }
uint ru32b(uint bo) { return rb(bo) | (rb(bo + 1u) << 8u) | (rb(bo + 2u) << 16u) | (rb(bo + 3u) << 24u); }
// Unaligned u32 via two word loads + funnel shift (vs ru32b's four byte-extract chains).
uint ru32u(uint bo) {
    uint w0 = NW(bo >> 2u);
    uint sh = (bo & 3u) << 3u;
    if (sh == 0u) { return w0; }
    return (w0 >> sh) | (NW((bo >> 2u) + 1u) << (32u - sh));
}
float f16tof32(uint bits) { return unpackHalf2x16(bits & 0xffffu).x; }
int sgn8(uint byte) { return int(byte) - int(byte >= 128u ? 256u : 0u); }

// Load four consecutive u32s starting at BYTE offset `bo` (any alignment) as a uvec4 —
// == uvec4(ru32u(bo), ru32u(bo+4), ru32u(bo+8), ru32u(bo+12)), bit-identical. The point is the
// read shape under -DSTREAMED: the four aligned words come from ONE NW4 (fused global_load_b128,
// saddr base — see native_weight_addr.glsl) instead of per-word scalar NW() loads the vectorizer
// can't fuse (distinct arena_word pointers), which lowered a run of ru32u/rb() reads to many
// unfused global_load_b32. A +N-misaligned block funnels in one neighbor word (the tail). Aligned
// blocks (sh==0) skip the funnel entirely. Used by the odd-stride native dqblk formats (Q3_K,
// Q4_0, Q5_0/1, IQ4_NL) to get the descriptor path's b128 vectorization on the arena path too;
// word-aligned formats (Q2_K, Q4_K/Q5_K/Q6_K, IQ4_XS) call NW4 directly.
uvec4 rw4(uint bo) {
    uint wb = bo >> 2u; uint sh = (bo & 3u) << 3u;
    uvec4 a = NW4(wb);
    if (sh == 0u) { return a; }
    uint t = NW(wb + 4u);
    return uvec4((a.x >> sh) | (a.y << (32u - sh)), (a.y >> sh) | (a.z << (32u - sh)),
                 (a.z >> sh) | (a.w << (32u - sh)), (a.w >> sh) | (t << (32u - sh)));
}

// Pair twin of rw4: two consecutive u32s at BYTE offset `bo` (any alignment) —
// == uvec2(ru32u(bo), ru32u(bo+4)), bit-identical. Aligned pairs are one fused b64 (NW2);
// a +2-misaligned pair funnels the 3 covering words out of one b128 (NW4) — one load either
// way, where two ru32u calls issued up to 4 unfused scalar loads under -DSTREAMED. The grid
// i-quants (IQ3_S/IQ3_XXS/IQ2_XS) read exactly such an 8-byte qs pair per 32-elem sub-block;
// their 2-mod-4 block strides make the pair aligned on odd blocks, +2-misaligned on even.
uvec2 rw2(uint bo) {
    uint wb = bo >> 2u; uint sh = (bo & 3u) << 3u;
    if (sh == 0u) { return NW2(wb); }
    uvec4 a = NW4(wb);
    return uvec2((a.x >> sh) | (a.y << (32u - sh)), (a.y >> sh) | (a.z << (32u - sh)));
}

// Single-u32 twin of rw2: one unaligned u32 at BYTE offset `bo` — == ru32u(bo), bit-identical —
// from ONE fused b64 (NW2) plus a branchless funnel-select. ru32u's aligned arm is already a
// single load, but its misaligned arm issues TWO scalar loads and the alignment branch diverges
// within a wave whenever neighboring lanes sit in blocks of opposite parity (every odd-stride
// format); this form is one load and no divergent load path either way. The select (not a shift
// by 32-sh, which is undefined at sh==0) keeps the aligned result exact.
uint rw1(uint bo) {
    uint wb = bo >> 2u; uint sh = (bo & 3u) << 3u;
    uvec2 a = NW2(wb);
    return (sh == 0u) ? a.x : ((a.x >> sh) | (a.y << (32u - sh)));
}

// USE_GRID builds stage their codebook tables into `shared` memory once per workgroup
// (grid_init() in the generated native_grids.glsl — see build.rs's gen_grids for why const-array
// indexing is a per-invocation-scratch catastrophe on RADV/ACO). Every includer runs `GRID_INIT;`
// at the TOP of main(): it contains a barrier(), so it must precede any early return / divergent
// flow. Non-grid builds expand it to an empty statement — byte-identical SPIR-V.
#ifdef USE_GRID
#include "native_grids.glsl"
#define GRID_INIT grid_init()
#else
#define GRID_INIT
#endif

#if defined(FMT_Q8_0)
// Q8_0: [f16 d][i8 qs[32]] = 34 bytes, 32 elements. y = d * qs[j].
float dq(uint g) {
    uint bd = (g / 32u) * 34u;
    return f16tof32(ru16(bd)) * float(sgn8(rb(bd + 2u + g % 32u)));
}
#define HAVE_DQBLK
// Word-parallel qs: the 32-byte qs body is pulled as two b128 quads (NW4) instead of eight per-word
// ru32u loads, and a SIGNED bitfieldExtract per element replaces the rb() byte-extract + sgn8 select
// chain — the byte-serial form left the Q8_0 decode GEMV at ~559-643 GiB/s (llama.cpp's runs ~792-838).
// bitfieldExtract(int, 8b, 8) IS the signed byte value — same integers, bit-identical.
//
// The b128 quads matter under -DSTREAMED: there each NW() is a scalar 64-bit BDA load that the
// vectorizer can't fuse (distinct arena_word pointers), so the old eight ru32u calls lowered to up to
// 16 unfused global_load_b32 per block — the streamed Q8_0 GEMV ran ~67% slower per dispatch than the
// descriptor-bound twin. NW4 reads four CONSTANT-index words off ONE pointer (fused global_load_b128,
// saddr base — see native_weight_addr.glsl), matching the descriptor path. The qs body starts at byte
// bd+2, so its word alignment flips with block parity: an ODD block (bd%4==2) has bd+2 word-aligned
// (qsh==0, read the quad words directly); an EVEN block (bd%4==0) is +2 misaligned (qsh==16, funnel a
// sliding neighbor word, word 8 the tail). Same funnel integers as the old ru32u form — bit-identical.
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 34u;
    float d = f16tof32(ru16(bd));
    uint qoff = bd + 2u;              // qs base byte
    uint qwb = qoff >> 2u;            // qs base word
    uint qsh = (qoff & 3u) << 3u;     // funnel shift: 0 (odd block, aligned) or 16 (even block, +2)
    uvec4 qA = NW4(qwb); uvec4 qB = NW4(qwb + 4u);
    if (qsh == 0u) {                  // aligned: word w8 IS ru32u(qoff + w8*4)
        for (uint w8 = 0u; w8 < 8u; w8++) {
            int q4 = int((w8 < 4u) ? qA[w8] : qB[w8 - 4u]);
            for (uint b = 0u; b < 4u; b++) {
                v[w8 * 4u + b] = d * float(bitfieldExtract(q4, int(8u * b), 8));
            }
        }
    } else {                         // +2 misaligned: funnel each word with its neighbor
        uint qtail = NW(qwb + 8u);
        for (uint w8 = 0u; w8 < 8u; w8++) {
            uint qcur = (w8 < 4u) ? qA[w8] : qB[w8 - 4u];
            uint w1 = w8 + 1u;
            uint qnxt = (w1 < 4u) ? qA[w1] : ((w1 < 8u) ? qB[w1 - 4u] : qtail);
            int q4 = int((qcur >> qsh) | (qnxt << (32u - qsh)));
            for (uint b = 0u; b < 4u; b++) {
                v[w8 * 4u + b] = d * float(bitfieldExtract(q4, int(8u * b), 8));
            }
        }
    }
}
#endif

#if defined(FMT_F16)
// F16: contiguous IEEE half (no blocks/scale). y = unpackHalf2x16 — EXACT. Element e lives at
// byte e*2 (packed 2-per-u32). Serves float weights that are UPLOADED as f16 (e.g. the
// DiffusionGemma SC soft-embedding, host-built f16 [ne, vocab]) when a warptile shape needs the
// native GEMM machinery (split-K) that the plain f16 `gemm_proj` kernel lacks.
float dq(uint g) {
    return f16tof32(ru16(g * 2u));
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bo = gstart * 2u;
    for (uint w = 0u; w < 32u; w++) { v[w] = f16tof32(ru16(bo + w * 2u)); }
}
#endif

#if defined(FMT_F32)
// F32: contiguous IEEE single (no blocks/scale) — one weight element per u32 word, EXACT. Serves
// the PAGED MoE expert path only: a paged float expert bank stages its RAW GGUF bytes into the
// arena (no host f16 conversion — resident float banks go through `bind_weight`, which converts
// to f16 and reports the effective dtype, so they hit FMT_F16 instead).
float dq(uint g) {
    return uintBitsToFloat(NW(g));
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    for (uint w = 0u; w < 32u; w++) { v[w] = uintBitsToFloat(NW(gstart + w)); }
}
#endif

#if defined(FMT_BF16)
// BF16: contiguous 16-bit truncated-f32 (no blocks/scale). y = bitcast(bits << 16) — EXACT (bf16 is
// the top 16 bits of an f32). Element e lives at byte e*2 (packed 2-per-u32).
float dq(uint g) {
    return uintBitsToFloat(ru16(g * 2u) << 16u);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bo = gstart * 2u;
    for (uint w = 0u; w < 32u; w++) { v[w] = uintBitsToFloat(ru16(bo + w * 2u) << 16u); }
}
#endif

#if defined(FMT_Q4_0)
// Q4_0: [f16 d][u8 qs[16]] = 18 bytes, 32 elements. y = d * (nibble - 8).
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    uint nib = (j < 16u) ? (rb(bd + 2u + j) & 0xFu) : (rb(bd + 2u + j - 16u) >> 4u);
    return d * (float(nib) - 8.0);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    // b128 the 16 qs bytes (4 words at bd+2) as one rw4 quad instead of 16 scalar rb() loads.
    // Same bytes — bit-identical. Low nibbles -> v[0..16), high nibbles -> v[16..32).
    uvec4 q = rw4(bd + 2u);
    for (uint w = 0u; w < 32u; w++) {
        uint jj = (w < 16u) ? w : (w - 16u);
        uint byteq = (q[jj >> 2u] >> ((jj & 3u) * 8u)) & 0xFFu;
        uint nib = (w < 16u) ? (byteq & 0xFu) : (byteq >> 4u);
        v[w] = d * (float(nib) - 8.0);
    }
}
#endif

#if defined(FMT_Q4_1)
// Q4_1: [f16 d][f16 m][u8 qs[16]] = 20 bytes, 32 elements. y = d*nibble + m.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 20u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u));
    uint nib = (j < 16u) ? (rb(bd + 4u + j) & 0xFu) : (rb(bd + 4u + j - 16u) >> 4u);
    return d * float(nib) + m;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 20u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u));
    for (uint w = 0u; w < 32u; w++) {
        uint nib = (w < 16u) ? (rb(bd + 4u + w) & 0xFu) : (rb(bd + 4u + w - 16u) >> 4u);
        v[w] = d * float(nib) + m;
    }
}
#endif

#if defined(FMT_Q5_0)
// Q5_0: [f16 d][u8 qh[4]][u8 qs[16]] = 22 bytes, 32 elements. y = d*(q5 - 16).
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 22u;
    float d = f16tof32(ru16(bd)); uint qh = ru32b(bd + 2u); uint val;
    if (j < 16u) { uint xh0 = ((qh >> j) << 4u) & 0x10u; val = (rb(bd + 6u + j) & 0xFu) | xh0; }
    else { uint jj = j - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 6u + jj) >> 4u) | xh1; }
    return d * (float(val) - 16.0);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 22u;
    float d = f16tof32(ru16(bd)); uint qh = ru32b(bd + 2u);
    for (uint w = 0u; w < 32u; w++) {
        uint val;
        if (w < 16u) { uint xh0 = ((qh >> w) << 4u) & 0x10u; val = (rb(bd + 6u + w) & 0xFu) | xh0; }
        else { uint jj = w - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 6u + jj) >> 4u) | xh1; }
        v[w] = d * (float(val) - 16.0);
    }
}
#endif

#if defined(FMT_Q5_1)
// Q5_1: [f16 d][f16 m][u8 qh[4]][u8 qs[16]] = 24 bytes, 32 elements. y = d*q5 + m.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 24u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u)); uint qh = ru32b(bd + 4u); uint val;
    if (j < 16u) { uint xh0 = ((qh >> j) << 4u) & 0x10u; val = (rb(bd + 8u + j) & 0xFu) | xh0; }
    else { uint jj = j - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 8u + jj) >> 4u) | xh1; }
    return d * float(val) + m;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 24u;
    float d = f16tof32(ru16(bd)); float m = f16tof32(ru16(bd + 2u)); uint qh = ru32b(bd + 4u);
    for (uint w = 0u; w < 32u; w++) {
        uint val;
        if (w < 16u) { uint xh0 = ((qh >> w) << 4u) & 0x10u; val = (rb(bd + 8u + w) & 0xFu) | xh0; }
        else { uint jj = w - 16u; uint xh1 = (qh >> (jj + 12u)) & 0x10u; val = (rb(bd + 8u + jj) >> 4u) | xh1; }
        v[w] = d * float(val) + m;
    }
}
#endif

#if defined(FMT_Q2K)
// Q2_K: [u8 scales[16]][u8 qs[64]][f16 d][f16 dmin] = 84 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 84u;
    float d = f16tof32(ru16(bd + 80u)); float dmin = f16tof32(ru16(bd + 82u));
    uint sc_byte = rb(bd + p / 16u);
    float dl = d * float(sc_byte & 0xFu); float ml = dmin * float(sc_byte >> 4u);
    uint n = p / 128u; uint p_half = p % 128u; uint jj = p_half / 32u; uint p_j = p_half % 32u;
    uint q2 = (rb(bd + 16u + 32u * n + p_j) >> (2u * jj)) & 3u;
    return dl * float(q2) - ml;
}
#define HAVE_DQBLK
// A 32-aligned run has CONSTANT n and jj, so its 2-bit quants are 32 CONSECUTIVE bytes = 8 aligned
// u32 words, under just 2 scale bytes — load those 10 values once instead of 2 byte-extract chains
// per element (the naive form was ~64 word loads per sub-block; the measured 14B-Q2_K decode
// catastrophe, 0.26x llama).
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 84u; uint p0 = gstart % 256u;
    float d = f16tof32(ru16(bd + 80u)); float dmin = f16tof32(ru16(bd + 82u));
    uint sb0 = rb(bd + p0 / 16u);        // scale/min nibbles for elements 0..15
    uint sb1 = rb(bd + p0 / 16u + 1u);   // … and 16..31
    float dl0 = d * float(sb0 & 0xFu); float ml0 = dmin * float(sb0 >> 4u);
    float dl1 = d * float(sb1 & 0xFu); float ml1 = dmin * float(sb1 >> 4u);
    uint shift = 2u * ((p0 % 128u) / 32u);
    uint qw = (bd + 16u + 32u * (p0 / 128u)) >> 2u; // 84-byte blocks are word-aligned; +16+32n too
    // b128 the 8 quant words as two NW4 quads — the streamed arena path gets the descriptor path's
    // buffer_load_b128 instead of 8 unfused scalar NW() loads. Same words — bit-identical.
    uvec4 qA = NW4(qw); uvec4 qB = NW4(qw + 4u);
    for (uint w8 = 0u; w8 < 8u; w8++) {
        // Hoist the plane shift to a single packed op: one shift+mask isolates all four 2-bit
        // codes, then a per-element bitfieldExtract (1 op) replaces the >>8b >>shift &3 chain.
        // Same integers, same v[w] expression — bit-identical to the per-element form.
        uint q4 = (((w8 < 4u) ? qA[w8] : qB[w8 - 4u]) >> shift) & 0x03030303u;
        for (uint b = 0u; b < 4u; b++) {
            uint w = w8 * 4u + b;
            uint q2 = bitfieldExtract(q4, int(8u * b), 2);
            v[w] = (w < 16u) ? (dl0 * float(q2) - ml0) : (dl1 * float(q2) - ml1);
        }
    }
}
#endif

#if defined(FMT_Q3K)
// Q3_K: [u8 hmask[32]][u8 qs[64]][u8 scales_raw[12]][f16 d] = 110 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 110u;
    float d_all = f16tof32(ru16(bd + 108u));
    uint a0 = ru32b(bd + 96u); uint a1 = ru32b(bd + 100u); uint a2 = ru32b(bd + 104u);
    uint k1 = 0x03030303u; uint k2 = 0x0f0f0f0fu; uint tmp = a2;
    uint aux[4];
    aux[2] = ((a0 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u);
    aux[3] = ((a1 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u);
    aux[0] = (a0 & k2) | ((tmp & k1) << 4u);
    aux[1] = (a1 & k2) | (((tmp >> 2u) & k1) << 4u);
    uint is = p / 16u;
    uint sc_byte = (aux[is >> 2u] >> ((is & 3u) * 8u)) & 0xFFu;
    int sc = int(sc_byte) - int(sc_byte >= 128u ? 256u : 0u) - 32;
    float dl = d_all * float(sc);
    uint n = p / 128u; uint p_half = p % 128u; uint jj = p_half / 32u; uint p_j = p_half % 32u;
    uint jg = 4u * n + jj; uint m = 1u << jg;
    uint low2 = (rb(bd + 32u + 32u * n + p_j) >> (2u * jj)) & 3u;
    uint high = ((rb(bd + p_j) & m) != 0u) ? 1u : 0u;
    return dl * (float(low2 | (high << 2u)) - 4.0);
}
#define HAVE_DQBLK
// A 32-aligned run has constant n/jj, so its low-2-bit quants are 32 CONSECUTIVE bytes (8 unaligned
// u32s via ru32u — 110-byte blocks aren't word-aligned) and its high bits are one fixed mask over
// the 32 hmask bytes; only 2 of the 16 6-bit scales apply. The naive form did 2 byte-extract
// chains per ELEMENT (the o/down Q3_K GEMVs in Q2_K-mix models were the decode bottleneck).
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 110u; uint p0 = gstart % 256u;
    float d_all = f16tof32(ru16(bd + 108u));
    // rw4 (one NW4 b128 + funnel) instead of three scalar ru32u — same words, bit-identical.
    uvec4 sc = rw4(bd + 96u); uint a0 = sc.x; uint a1 = sc.y; uint a2 = sc.z;
    uint k1 = 0x03030303u; uint k2 = 0x0f0f0f0fu; uint tmp = a2;
    uint aux[4];
    aux[2] = ((a0 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u);
    aux[3] = ((a1 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u);
    aux[0] = (a0 & k2) | ((tmp & k1) << 4u);
    aux[1] = (a1 & k2) | (((tmp >> 2u) & k1) << 4u);
    uint is0 = p0 / 16u;
    uint sb0 = (aux[is0 >> 2u] >> ((is0 & 3u) * 8u)) & 0xFFu;
    uint sb1 = (aux[(is0 + 1u) >> 2u] >> (((is0 + 1u) & 3u) * 8u)) & 0xFFu;
    float dl0 = d_all * float(sgn8(sb0) - 32);
    float dl1 = d_all * float(sgn8(sb1) - 32);
    uint n = p0 / 128u;
    uint jj = (p0 % 128u) / 32u;
    uint shift = 2u * jj;
    uint jg = 4u * n + jj;
    uint qb = bd + 32u + 32u * n;
    // b128 the 8 low-2-bit words and the 8 hmask words as two rw4 quads each (fused NW4 + funnel)
    // instead of two per-word ru32u loads per element — the streamed arena path's big regressor.
    // Same words as ru32u(qb+w8*4) / ru32u(bd+w8*4) — bit-identical.
    uvec4 qL0 = rw4(qb); uvec4 qL1 = rw4(qb + 16u);
    uvec4 hM0 = rw4(bd); uvec4 hM1 = rw4(bd + 16u);
    for (uint w8 = 0u; w8 < 8u; w8++) {
        // Packed plane extraction: one shift+mask isolates the four 2-bit codes, one
        // shift+mask+shl turns the four hmask bits into the packed +4 plane, one OR merges —
        // the per-element >>8b>>shift&3 chain and &m!=0 compare/select drop to a single
        // bitfieldExtract. XOR 0x04 flips the +4 plane so a SIGNED 3-bit extract yields q-4
        // directly ((q^4) as s3 == q-4 for q in 0..7), dropping the per-element -4.0 subtract.
        // float(int(q-4)) == float(q)-4.0 exactly (small integers) — bit-identical.
        uint qlo = (w8 < 4u) ? qL0[w8] : qL1[w8 - 4u];
        uint hbits = (w8 < 4u) ? hM0[w8] : hM1[w8 - 4u];
        uint q4 = ((qlo >> shift) & 0x03030303u)
            | (((hbits >> jg) & 0x01010101u) << 2u);
        int q4s = int(q4 ^ 0x04040404u);
        for (uint b = 0u; b < 4u; b++) {
            uint w = w8 * 4u + b;
            float dl = (w < 16u) ? dl0 : dl1;
            v[w] = dl * float(bitfieldExtract(q4s, int(8u * b), 3));
        }
    }
}
#endif

#if defined(FMT_Q4K) || defined(FMT_Q5K)
// k4: decode 6-bit scale/min for sub-block i from the 12-byte scales field at sb.
uvec2 k4(uint i, uint sb) {
    if (i < 4u) { return uvec2(rb(sb + i) & 63u, rb(sb + i + 4u) & 63u); }
    uint sc = (rb(sb + i + 4u) & 0xFu) | ((rb(sb + i - 4u) >> 6u) << 4u);
    uint mn = (rb(sb + i + 4u) >> 4u) | ((rb(sb + i) >> 6u) << 4u);
    return uvec2(sc, mn);
}
// k4 on the in-register 16-byte block header (hdr = NW4 of the block base: hdr.x = d|dmin,
// hdr.y/z/w = scales[12]). Same bytes k4 reads through rb() — bit-identical — but from registers:
// under -DSTREAMED each rb() is its own un-fusable BDA deref (measured: 6 scalar global_load_b32
// per dqblk on the Q4K warp GEMM, where the descriptor twin's header reads fuse to b32/b64/b96),
// so the caller loads the header ONCE as a b128 quad and extracts here. byte b of word w is
// bitfieldExtract(w, 8b, 8).
uint k4b(uint w, uint b) { return bitfieldExtract(w, int(b << 3u), 8); }
uvec2 k4w(uint i, uvec4 hdr) {
    if (i < 4u) { return uvec2(k4b(hdr.y, i) & 63u, k4b(hdr.z, i) & 63u); }
    uint j = i - 4u;
    uint sc = (k4b(hdr.w, j) & 0xFu) | ((k4b(hdr.y, j) >> 6u) << 4u);
    uint mn = (k4b(hdr.w, j) >> 4u) | ((k4b(hdr.z, j) >> 6u) << 4u);
    return uvec2(sc, mn);
}
#endif

#if defined(FMT_Q4K)
// Q4_K: [f16 d][f16 dmin][u8 scales[12]][u8 qs[128]] = 144 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 144u;
    float d = f16tof32(ru16(bd)); float dmin = f16tof32(ru16(bd + 2u));
    uint sb = bd + 4u; uint j = p / 64u; uint p_j = p % 64u; uint l = p_j % 32u;
    uvec2 k = (p_j < 32u) ? k4(2u * j, sb) : k4(2u * j + 1u, sb);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    uint qs_byte = rb(bd + 16u + j * 32u + l);
    uint val = (p_j < 32u) ? (qs_byte & 0xFu) : (qs_byte >> 4u);
    return dl * float(val) - mm;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {  // decode d/dmin/6-bit scale once for the sub-block.
    uint sblk = gstart / 32u; uint super = sblk / 8u; uint sub = sblk % 8u;
    uint bd = super * 144u;
    // One b128 header quad (bd is 16B-aligned: 144%16==0) instead of byte-chokepoint rb()/ru16()
    // reads — same bytes, bit-identical (see k4w).
    uvec4 hdr = NW4(bd >> 2u);
    float d = f16tof32(hdr.x); float dmin = f16tof32(hdr.x >> 16u);
    uvec2 k = k4w(sub, hdr);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    // Word-parallel qs: block byte offsets are 4-aligned (144%4==0, qs at +16, sub-row stride 32)
    // — load 8 whole u32s instead of 32 rb() byte-extract chains; nibble select hoisted (constant
    // per sub-block). The staging loop of the warptile GEMM is decode-ALU-bound, so this is
    // directly visible in prefill GEMM throughput.
    uint qw = (bd + 16u + (sub / 2u) * 32u) >> 2u;
    uint nsh = ((sub & 1u) == 0u) ? 0u : 4u;
    // b128 the two qs quads (qw is 16B-aligned: 144%16==0, +16, sub-stride 32). Matches the
    // descriptor path's buffer_load_b128; the streamed NW4 keeps a saddr base. Same words as
    // NW(qw+w8) — bit-identical.
    uvec4 qA = NW4(qw); uvec4 qB = NW4(qw + 4u);
    for (uint w8 = 0u; w8 < 8u; w8++) {
        uint q = ((w8 < 4u) ? qA[w8] : qB[w8 - 4u]) >> nsh;
        for (uint b = 0u; b < 4u; b++) {
            v[w8 * 4u + b] = dl * float((q >> (8u * b)) & 0xFu) - mm;
        }
    }
}
#endif

#if defined(FMT_Q5K)
// Q5_K: [f16 d][f16 dmin][u8 scales[12]][u8 qh[32]][u8 qs[128]] = 176 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 176u;
    float d = f16tof32(ru16(bd)); float dmin = f16tof32(ru16(bd + 2u));
    uint sb = bd + 4u; uint j = p / 64u; uint p_j = p % 64u; uint l = p_j % 32u;
    uvec2 k = (p_j < 32u) ? k4(2u * j, sb) : k4(2u * j + 1u, sb);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    uint qs_byte = rb(bd + 48u + j * 32u + l);
    uint qh_byte = rb(bd + 16u + l);
    uint val;
    if (p_j < 32u) { val = (qs_byte & 0xFu) + (((qh_byte & (1u << (2u * j))) != 0u) ? 16u : 0u); }
    else { val = (qs_byte >> 4u) + (((qh_byte & (2u << (2u * j))) != 0u) ? 16u : 0u); }
    return dl * float(val) - mm;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {  // decode d/dmin/6-bit scale once for the sub-block.
    uint sblk = gstart / 32u; uint super = sblk / 8u; uint sub = sblk % 8u;
    uint bd = super * 176u;
    // One b128 header quad (bd is 16B-aligned: 176%16==0) — same bytes, bit-identical (see k4w).
    uvec4 hdr = NW4(bd >> 2u);
    float d = f16tof32(hdr.x); float dmin = f16tof32(hdr.x >> 16u);
    uvec2 k = k4w(sub, hdr);
    float dl = d * float(k.x); float mm = dmin * float(k.y);
    // Word-parallel qs+qh: block byte offsets are 4-aligned (176%4==0, qh at +16, qs at +48) —
    // 16 whole-u32 loads instead of 64 rb() byte-extract chains, nibble/high-bit selects hoisted
    // (constant per sub-block). Same `nib | (h<<4)` value as the scalar loop — bit-identical.
    uint j = sub / 2u;
    uint qw = (bd + 48u + j * 32u) >> 2u;
    uint hw = (bd + 16u) >> 2u;
    uint nsh = ((sub & 1u) == 0u) ? 0u : 4u;      // low/high nibble of qs
    uint hsh = 2u * j + (((sub & 1u) == 0u) ? 0u : 1u); // qh bit index for this sub-block
    // SWAR the 5-bit rebuild across all four byte-lanes at once. The per-element form
    //   val = ((q >> 8b) & 0xF) | (((h >> 8b) & 1) << 4)
    // costs 6 integer ops PER ELEMENT (shift/and/shift/and/shl/or). Masking the whole word
    // instead — `q & 0x0F0F0F0F` leaves each byte's nibble in place, `(h & 0x01010101) << 4`
    // leaves each byte's 5th bit at bit 4 of its lane — rebuilds all four 5-bit codes in 4 ops
    // TOTAL, leaving one bitfieldExtract per element. 6 ops/elem -> ~2. Same integers, so this
    // is bit-identical to the byte-serial form.
    //
    // Why it matters: Q5_K's dequant ALU is CO-CRITICAL with DRAM at decode, not free. On a
    // 7900 XTX the gemma-4-31B Q5_K weights (14.90 GiB/token) have a 15.5 us/GiB DRAM floor of
    // 15.5 ms, while the old ~9 ops/elem cost ~12.7 ms of ALU — so the two could not overlap and
    // the GEMV landed at 686 GiB/s (77% of peak). Q8_0, whose decode is ~4 ops/elem (ALU
    // effectively free), hits 804 GiB/s (90%) through this SAME kernel — that gap was the ALU,
    // and this cuts it to ~5 ops/elem.
    // b128 the qs/qh quads (176%16==0, qh at +16, qs at +48 => qw/hw 16B-aligned). Matches the
    // descriptor path's buffer_load_b128; streamed NW4 keeps a saddr base. Bit-identical words.
    uvec4 qA = NW4(qw); uvec4 qB = NW4(qw + 4u);
    uvec4 hA = NW4(hw); uvec4 hB = NW4(hw + 4u);
    for (uint w8 = 0u; w8 < 8u; w8++) {
        uint q = ((w8 < 4u) ? qA[w8] : qB[w8 - 4u]) >> nsh;
        uint h = ((w8 < 4u) ? hA[w8] : hB[w8 - 4u]) >> hsh;
        uint packed = (q & 0x0F0F0F0Fu) | ((h & 0x01010101u) << 4u);
        for (uint b = 0u; b < 4u; b++) {
            uint val = bitfieldExtract(packed, int(8u * b), 5);
            v[w8 * 4u + b] = dl * float(val) - mm;
        }
    }
}
#endif

#if defined(FMT_Q6K)
// Q6_K: [u8 ql[128]][u8 qh[64]][i8 scales[16]][f16 d] = 210 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 210u;
    float d = f16tof32(ru16(bd + 208u));
    uint hf = p / 128u; uint p_half = p % 128u; uint og = p_half / 32u; uint l = p_half % 32u;
    uint qlo = hf * 64u; uint qho = hf * 32u;
    uint qa = rb(bd + qlo + l); uint qb = rb(bd + qlo + l + 32u); uint qh = rb(bd + 128u + qho + l);
    uint q;
    if (og == 0u) { q = (qa & 0xFu) | ((qh & 3u) << 4u); }
    else if (og == 1u) { q = (qb & 0xFu) | (((qh >> 2u) & 3u) << 4u); }
    else if (og == 2u) { q = (qa >> 4u) | (((qh >> 4u) & 3u) << 4u); }
    else { q = (qb >> 4u) | (((qh >> 6u) & 3u) << 4u); }
    uint sc_idx = hf * 8u + l / 16u + 2u * og;
    int sc = sgn8(rb(bd + 192u + sc_idx));
    return d * float(sc) * (float(q) - 32.0);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {  // hf/og/d constant; hoist branch, one ql byte/elem.
    uint bd = (gstart / 256u) * 210u; uint p0 = gstart % 256u;
    float d = f16tof32(ru16(bd + 208u));
    uint hf = p0 / 128u; uint og = (p0 % 128u) / 32u;
    uint scbase = bd + 192u + hf * 8u + 2u * og;
    bool useB = (og & 1u) == 1u; bool high = og >= 2u; uint qshift = og * 2u;
    uint qoff = bd + hf * 64u + (useB ? 32u : 0u);  // pick qa/qb half once
    uint qhoff = bd + 128u + hf * 32u;
    // Word-parallel ql+qh (blocks are 210 bytes — only 2-aligned, so ru32u funnel-shift loads):
    // 16 u32 loads instead of 96 rb() byte-extract chains; nibble select and the two per-16-elem
    // scales hoisted. Same (d*sc)*(q-32) product order as the scalar loop — bit-identical.
    float dsc0 = d * float(sgn8(rb(scbase)));
    float dsc1 = d * float(sgn8(rb(scbase + 1u)));
    uint nsh = high ? 4u : 0u;
    // Hoist the 210-byte block's funnel shift out of the per-word loop. qoff/qhoff have a FIXED
    // alignment for the whole sub-block (bd%4 = 2*(block parity), and hf*64/hf*32/±32 are all
    // 4-multiples), so ql/qh are BOTH word-aligned (even block) or BOTH +2 misaligned (odd block).
    // Even block: read NW(base+w8) directly — base+offset addressing, strength-reducible, and no
    // funnel at all. Odd block: slide one neighbor word (2 initial + 16 in-loop = 18 loads) instead
    // of ru32u recomputing (off+w8*4)>>2 and doing two loads per element (32). Same funnel integers
    // as the old ru32u form — bit-identical. Matters under -DSTREAMED: each NW() is a 64-bit BDA
    // load, so the base+w8 word index and the halved odd-block load count both cut address ALU.
    uint qwb = qoff >> 2u; uint qsh = (qoff & 3u) << 3u;
    uint qhb = qhoff >> 2u; uint hsh = (qhoff & 3u) << 3u;
    // Pull the 8 ql + 8 qh words as four b128 quads (see NW4). Same words as NW(base+w8), so the
    // funnel integers below are identical — bit-identical to the scalar form.
    uvec4 qA = NW4(qwb); uvec4 qB = NW4(qwb + 4u);
    uvec4 hA = NW4(qhb); uvec4 hB = NW4(qhb + 4u);
    if (qsh == 0u) {   // even block: fully word-aligned (qsh==0 <=> hsh==0), no funnel
        for (uint w8 = 0u; w8 < 8u; w8++) {
            uint q = ((w8 < 4u) ? qA[w8] : qB[w8 - 4u]) >> nsh;
            uint h = ((w8 < 4u) ? hA[w8] : hB[w8 - 4u]) >> qshift;
            float dsc = (w8 < 4u) ? dsc0 : dsc1;
            for (uint b = 0u; b < 4u; b++) {
                uint qq = ((q >> (8u * b)) & 0xFu) | (((h >> (8u * b)) & 3u) << 4u);
                v[w8 * 4u + b] = dsc * (float(qq) - 32.0);
            }
        }
    } else {           // odd block: +2 misaligned, funnel a sliding neighbor (word 8 is the tail)
        uint qtail = NW(qwb + 8u); uint htail = NW(qhb + 8u);
        for (uint w8 = 0u; w8 < 8u; w8++) {
            uint qcur = (w8 < 4u) ? qA[w8] : qB[w8 - 4u];
            uint hcur = (w8 < 4u) ? hA[w8] : hB[w8 - 4u];
            uint w1 = w8 + 1u;
            uint qnxt = (w1 < 4u) ? qA[w1] : ((w1 < 8u) ? qB[w1 - 4u] : qtail);
            uint hnxt = (w1 < 4u) ? hA[w1] : ((w1 < 8u) ? hB[w1 - 4u] : htail);
            uint q = ((qcur >> qsh) | (qnxt << (32u - qsh))) >> nsh;
            uint h = ((hcur >> hsh) | (hnxt << (32u - hsh))) >> qshift;
            float dsc = (w8 < 4u) ? dsc0 : dsc1;
            for (uint b = 0u; b < 4u; b++) {
                uint qq = ((q >> (8u * b)) & 0xFu) | (((h >> (8u * b)) & 3u) << 4u);
                v[w8 * 4u + b] = dsc * (float(qq) - 32.0);
            }
        }
    }
}
#endif

// ── codebook formats (small const tables / decode fns, no grid) ──
#if defined(FMT_IQ4NL) || defined(FMT_IQ4XS)
// kvalues_iq4nl, byte-packed 4-per-u32 (all 16 signed values fit int8) instead of a 16-entry
// `int[]`: a dynamically-indexed 16-element array lowers (RADV/ACO) to a far longer select
// cascade than a 4-element one, and the intra-word extract is a single native bitfieldExtract
// (V_BFE, sign-extending) instead of a second array read — idx>>2 picks the word (4-way), idx&3
// the byte lane. This IS the ALU floor of IQ4_XS decode: bytes-vs-speed showed IQ4_XS (4.25 bpw,
// SMALLER than Q4_K's 4.5) running 1.55-2.1x SLOWER per dispatch at matched shapes — codebook-
// gather ALU, not DRAM bandwidth.
const uint KV_IQ4NL_W[4] = uint[](0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);
int kv_iq4nl(uint idx) {
    return bitfieldExtract(int(KV_IQ4NL_W[idx >> 2u]), int((idx & 3u) << 3u), 8);
}
#endif
#if defined(FMT_MXFP4) || defined(FMT_NVFP4)
const int KV_MXFP4[16] = int[](0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
#endif
#if defined(FMT_TQ1_0)
const uint POW3[6] = uint[](1u, 3u, 9u, 27u, 81u, 243u);
#endif
#if defined(FMT_MXFP4)
float e8m0_half(uint x) {
    if (x < 2u) { return uintBitsToFloat(0x00200000u << x); }
    return uintBitsToFloat((x - 1u) << 23u);
}
#endif
#if defined(FMT_NVFP4)
float ue4m3(uint x) {
    if (x == 0u || x == 0x7Fu) { return 0.0; }
    uint e = (x >> 3u) & 0xFu; float man = float(x & 7u);
    float raw = (e == 0u) ? (man * exp2(-9.0)) : ((1.0 + man / 8.0) * exp2(float(int(e) - 7)));
    return raw * 0.5;
}
#endif

#if defined(FMT_IQ4NL)
// IQ4_NL: [f16 d][u8 qs[16]] = 18 bytes, 32 elements.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    uint idx = (j < 16u) ? (rb(bd + 2u + j) & 0xFu) : (rb(bd + 2u + j - 16u) >> 4u);
    return d * float(kv_iq4nl(idx));
}
#define HAVE_DQBLK
// One 32-elem sub-block = exactly one whole IQ4_NL block (gstart is 32-aligned). Decode d ONCE
// and load the 16 qs bytes as 4 (unaligned — 18-byte block stride) u32s instead of 32 rb()
// byte-extract chains; low nibble → v[0..16), high nibble → v[16..32). Same d*KV[nibble] value
// as the scalar loop — bit-identical. The warptile GEMM's staging loop is decode-ALU-bound,
// so this is directly visible in prefill GEMM throughput.
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 18u;
    float d = f16tof32(ru16(bd));
    for (uint w8 = 0u; w8 < 4u; w8++) {
        uint q = ru32u(bd + 2u + w8 * 4u);
        for (uint b = 0u; b < 4u; b++) {
            uint byteq = (q >> (8u * b)) & 0xFFu;
            v[w8 * 4u + b] = d * float(kv_iq4nl(byteq & 0xFu));
            v[w8 * 4u + b + 16u] = d * float(kv_iq4nl(byteq >> 4u));
        }
    }
}
#endif

#if defined(FMT_IQ4XS)
// IQ4_XS: [f16 d][u16 scales_h][u8 scales_l[4]][u8 qs[128]] = 136 bytes, 256 elements.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 136u;
    float d = f16tof32(ru16(bd));
    uint scales_h = ru16(bd + 2u);
    uint ib = p / 32u; uint within = p % 32u;
    uint lo = (rb(bd + 4u + (ib / 2u)) >> (4u * (ib & 1u))) & 0xFu;
    uint hi = (scales_h >> (2u * ib)) & 3u;
    uint ls = lo | (hi << 4u);
    float dl = d * float(int(ls) - 32);
    uint qoff = bd + 8u + 16u * ib;
    uint idx = (within < 16u) ? (rb(qoff + within) & 0xFu) : (rb(qoff + within - 16u) >> 4u);
    return dl * float(kv_iq4nl(idx));
}
#define HAVE_DQBLK
// One 32-elem sub-block = exactly one of the 8 IQ4_XS sub-blocks (gstart is 32-aligned).
// Decode d and the 6-bit sub-block scale ONCE. Word-parallel qs: block byte offsets are 4-aligned
// (136%4==0, qs at +8, sub-block stride 16) — load the 16 qs bytes as 4 whole u32s instead of 16
// rb() byte-extract chains, with the codebook gather hoisted per byte. Same dl*KV[nibble] value as
// the scalar loop — bit-identical. The warptile GEMM's staging loop is decode-ALU-bound, so this is
// directly visible in prefill GEMM (and the decode native_gemv) throughput.
void dqblk(uint gstart, out float v[32]) {
    uint p = gstart % 256u;
    uint bd = (gstart / 256u) * 136u;
    float d = f16tof32(ru16(bd));
    uint scales_h = ru16(bd + 2u);
    uint ib = p / 32u;
    uint lo = (rb(bd + 4u + (ib / 2u)) >> (4u * (ib & 1u))) & 0xFu;
    uint hi = (scales_h >> (2u * ib)) & 3u;
    float dl = d * float(int(lo | (hi << 4u)) - 32);
    uint qw = (bd + 8u + 16u * ib) >> 2u;
    // b128 the 4 qs words as one NW4 quad (word-aligned: 136%4==0, +8, stride 16) instead of 4
    // scalar NW() loads — the streamed arena path's fusion. Same words — bit-identical.
    uvec4 qv = NW4(qw);
    for (uint w4 = 0u; w4 < 4u; w4++) {
        uint q = qv[w4];
        for (uint b = 0u; b < 4u; b++) {
            uint byte = (q >> (8u * b)) & 0xFFu;
            uint j = w4 * 4u + b;
            v[j] = dl * float(kv_iq4nl(byte & 0xFu));
            v[j + 16u] = dl * float(kv_iq4nl(byte >> 4u));
        }
    }
}
#endif

#if defined(FMT_MXFP4)
// MXFP4: [u8 e8m0][u8 qs[16]] = 17 bytes, 32 elements.
float dq(uint g) {
    uint j = g % 32u; uint bd = (g / 32u) * 17u;
    float d = e8m0_half(rb(bd));
    uint idx = (j < 16u) ? (rb(bd + 1u + j) & 0xFu) : (rb(bd + 1u + j - 16u) >> 4u);
    return float(KV_MXFP4[idx]) * d;
}
#define HAVE_DQBLK
// One 32-element group is one complete MXFP4 block. Decode the shared E8M0 scale once and fetch
// the 16 packed code bytes as four unaligned words instead of repeating the address/scale path for
// every element. The 17-byte block stride makes aligned vector loads impossible, so `ru32u` keeps
// the same safe two-word funnel convention used by the other odd-stride formats.
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 32u) * 17u;
    float d = e8m0_half(rb(bd));
    for (uint w4 = 0u; w4 < 4u; w4++) {
        uint q = ru32u(bd + 1u + w4 * 4u);
        for (uint b = 0u; b < 4u; b++) {
            uint packed = (q >> (8u * b)) & 0xFFu;
            uint j = w4 * 4u + b;
            v[j] = float(KV_MXFP4[packed & 0xFu]) * d;
            v[j + 16u] = float(KV_MXFP4[packed >> 4u]) * d;
        }
    }
}
#endif

#if defined(FMT_NVFP4)
// NVFP4: [u8 scales[4]][u8 qs[32]] = 36 bytes, 64 elements.
float dq(uint g) {
    uint p = g % 64u; uint bd = (g / 64u) * 36u;
    uint s = p / 16u; uint within = p % 16u;
    float d = ue4m3(rb(bd + s));
    uint idx = (within < 8u) ? (rb(bd + 4u + s * 8u + within) & 0xFu) : (rb(bd + 4u + s * 8u + within - 8u) >> 4u);
    return float(KV_MXFP4[idx]) * d;
}
#endif

#if defined(FMT_TQ1_0)
// TQ1_0: [u8 qs[48]][u8 qh[4]][f16 d] = 54 bytes, 256 elements. Ternary via pow-3.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 54u;
    float d = f16tof32(ru16(bd + 52u));
    uint src; uint n;
    if (p < 160u) { n = p / 32u; src = rb(bd + (p % 32u)); }
    else if (p < 240u) { uint pp = p - 160u; n = pp / 16u; src = rb(bd + 32u + (pp % 16u)); }
    else { uint pp = p - 240u; n = pp / 4u; src = rb(bd + 48u + (pp % 4u)); }
    uint q = (src * POW3[n]) & 0xFFu;
    uint xi = (q * 3u) >> 8u;
    return float(int(xi) - 1) * d;
}
#endif

#if defined(FMT_TQ2_0)
// TQ2_0: [u8 qs[64]][f16 d] = 66 bytes, 256 elements. 2-bit ternary.
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 66u;
    float d = f16tof32(ru16(bd + 64u));
    uint chunk = p / 128u; uint rem = p % 128u; uint l = rem / 32u; uint m = rem % 32u;
    uint q = (rb(bd + chunk * 32u + m) >> (l * 2u)) & 3u;
    return (float(q) - 1.0) * d;
}
#endif

#if defined(FMT_Q2_0)
// Q2_0 (Bonsai ternary): [f16 d][u8 qs[16]] = 18 bytes, 64 elements. y = (q - 1) * d,
// q ∈ {0,1,2,3} → {-d, 0, +d, +2d}. Codes pack SEQUENTIALLY: element j → byte j/4,
// shift (j%4)*2 — so a 32-elem group is 8 contiguous bytes (2 unaligned u32s via ru32u;
// 18-byte blocks aren't word-aligned, same funnel-load convention as Q3_K's 110 B).
float dq(uint g) {
    uint p = g % 64u; uint bd = (g / 64u) * 18u;
    float d = f16tof32(ru16(bd));
    uint q = (rb(bd + 2u + p / 4u) >> ((p % 4u) * 2u)) & 3u;
    return (float(q) - 1.0) * d;
}
#define HAVE_DQBLK
// 32-aligned run = half a 64-block: scale decoded once, codes pulled as 2 unaligned words with a
// 2-bit bitfieldExtract per element — identical (q-1)*d product per element, bit-identical to dq().
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 64u) * 18u;
    float d = f16tof32(ru16(bd));
    uint qoff = bd + 2u + ((gstart % 64u) >> 2u); // 0 or 8 bytes into qs
    for (uint w = 0u; w < 2u; w++) {
        uint bits = ru32u(qoff + w * 4u);
        for (uint b = 0u; b < 16u; b++) {
            v[w * 16u + b] = (float((bits >> (b * 2u)) & 3u) - 1.0) * d;
        }
    }
}
#endif

// ── grid-based i-quants (tables from native_grids.glsl) ──
// All 7 grid formats get an amortized dqblk: a 32-aligned run is exactly one 32-elem group
// (ib32 / pair·grp / ib), so the per-group header work — scale decode, sign/aux words, qh bits,
// the 4 (or 8) codebook-grid SHARED-LUT gathers — is hoisted out of the per-element path. The
// naive dq() fallback re-derived ALL of it per element (5-8 dependent loads each): the measured
// id/idm expert-GEMV decode floor on Qwen3.6-35B-A3B IQ3_S (tg128 0.50x llama.cpp). Each dqblk
// computes the IDENTICAL `scale * float(sgn8(bv)) * sign` expression per element (same product
// order) — bit-identical to the per-element form.
#if defined(FMT_IQ2XXS)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 66u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint off = bd + 2u + ib32 * 8u;
    uint aux0 = ru32b(off); uint aux1 = ru32b(off + 4u);
    uint grid_idx = (aux0 >> (8u * l)) & 0xFFu;
    uint sign_idx = (aux1 >> (7u * l)) & 127u;
    float db = d * (0.5 + float(aux1 >> 28u)) * 0.25;
    uint bv = (j < 4u) ? ((G_IQ2XXS[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ2XXS[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    float sign = (((KSIGNS[sign_idx] >> j) & 1u) != 0u) ? -1.0 : 1.0;
    return db * float(sgn8(bv)) * sign;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 66u; uint ib32 = (gstart % 256u) / 32u;
    float d = f16tof32(ru16(bd));
    uint off = bd + 2u + ib32 * 8u;
    uint aux0 = ru32u(off); uint aux1 = ru32u(off + 4u);
    float db = d * (0.5 + float(aux1 >> 28u)) * 0.25;
    for (uint l = 0u; l < 4u; l++) {
        uint grid_idx = (aux0 >> (8u * l)) & 0xFFu;
        uint signs = KSIGNS[(aux1 >> (7u * l)) & 127u];
        uint gw0 = G_IQ2XXS[2u * grid_idx];
        uint gw1 = G_IQ2XXS[2u * grid_idx + 1u];
        for (uint j = 0u; j < 8u; j++) {
            uint bv = (j < 4u) ? ((gw0 >> (8u * j)) & 0xFFu) : ((gw1 >> (8u * (j - 4u))) & 0xFFu);
            float sign = (((signs >> j) & 1u) != 0u) ? -1.0 : 1.0;
            v[l * 8u + j] = db * float(sgn8(bv)) * sign;
        }
    }
}
#endif

#if defined(FMT_IQ2XS)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 74u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint qs16 = ru16(bd + 2u + (ib32 * 4u + l) * 2u);
    uint grid_idx = qs16 & 511u; uint sign_idx = qs16 >> 9u;
    uint sc = rb(bd + 66u + ib32);
    float dl = (l < 2u) ? (d * (0.5 + float(sc & 0xFu)) * 0.25) : (d * (0.5 + float(sc >> 4u)) * 0.25);
    uint bv = (j < 4u) ? ((G_IQ2XS[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ2XS[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    float sign = (((KSIGNS[sign_idx] >> j) & 1u) != 0u) ? -1.0 : 1.0;
    return dl * float(sgn8(bv)) * sign;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 74u; uint ib32 = (gstart % 256u) / 32u;
    float d = f16tof32(ru16(bd));
    uint sc = rb(bd + 66u + ib32);
    float dl0 = d * (0.5 + float(sc & 0xFu)) * 0.25;
    float dl1 = d * (0.5 + float(sc >> 4u)) * 0.25;
    uvec2 qp2 = rw2(bd + 2u + ib32 * 8u);           // fused qs pair (see rw2)
    uint qsw = qp2.x;                               // qs16[l] for l=0,1
    uint qsw2 = qp2.y;                              // qs16[l] for l=2,3
    for (uint l = 0u; l < 4u; l++) {
        uint qs16 = ((l < 2u ? qsw : qsw2) >> (16u * (l & 1u))) & 0xFFFFu;
        uint grid_idx = qs16 & 511u;
        uint signs = KSIGNS[qs16 >> 9u];
        float dl = (l < 2u) ? dl0 : dl1;
        uint gw0 = G_IQ2XS[2u * grid_idx];
        uint gw1 = G_IQ2XS[2u * grid_idx + 1u];
        for (uint j = 0u; j < 8u; j++) {
            uint bv = (j < 4u) ? ((gw0 >> (8u * j)) & 0xFFu) : ((gw1 >> (8u * (j - 4u))) & 0xFFu);
            float sign = (((signs >> j) & 1u) != 0u) ? -1.0 : 1.0;
            v[l * 8u + j] = dl * float(sgn8(bv)) * sign;
        }
    }
}
#endif

#if defined(FMT_IQ2S)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 82u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint qs_byte = rb(bd + 2u + ib32 * 4u + l);
    uint sign_byte = rb(bd + 2u + 32u + ib32 * 4u + l);
    uint qh_byte = rb(bd + 66u + ib32);
    uint grid_idx = qs_byte | ((qh_byte << (8u - 2u * l)) & 0x300u);
    uint sc = rb(bd + 74u + ib32);
    float dl = (l < 2u) ? (d * (0.5 + float(sc & 0xFu)) * 0.25) : (d * (0.5 + float(sc >> 4u)) * 0.25);
    uint bv = (j < 4u) ? ((G_IQ2S[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ2S[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    float sign = (((sign_byte >> j) & 1u) != 0u) ? -1.0 : 1.0;
    return dl * float(sgn8(bv)) * sign;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 82u; uint ib32 = (gstart % 256u) / 32u;
    float d = f16tof32(ru16(bd));
#ifdef STREAMED
    // Wide-fetch restructure (STREAMED only — the descriptor twin's rb()/ru32u already vectorize
    // and get SGPR-descriptor addressing for free; don't perturb its codegen). IQ2_S has no
    // adjacent qs word-PAIR to fuse (d@0, qs@2+4i, signs@34+4i, qh@66+i, sc@74+i — all mutually
    // non-adjacent), so instead each field is fetched as ONE fused load: the qs and sign words via
    // branchless rw1 (b64), and qh+sc — exactly 2 words apart — out of ONE b128 window anchored to
    // END at the sc word (`NW4(scw-3)`, so tail[1] is the qh word and tail[3] the sc word). The
    // END anchor is load-bearing for bounds: expert banks are bump-allocated at 256-byte alignment
    // and a bank whose size is an exact 256-multiple sits FLUSH with its dedicated block's end
    // (e.g. 128x 2048x768 IQ2_S banks = 64487424 bytes % 256 == 0), so a window anchored at the
    // qh word could read up to 7 bytes past the VkBuffer on the last block; word-granular reads
    // of needed bytes are always inside align4(size) <= block size. Same bytes as the scalar
    // rb()/ru32u chain — bit-identical. 8 scalar global_load_b32 -> 4 fused loads (b32+b64+b64+b128).
    uint scb = bd + 74u + ib32;                      // sc byte address
    uint scw = scb >> 2u;                            // sc word; qh word is exactly scw - 2
    uvec4 tail = NW4(scw - 3u);
    uint bsh = (scb & 3u) << 3u;                     // qh/sc share the in-word byte lane (8 apart)
    uint qh = (tail[1] >> bsh) & 0xFFu;
    uint sc = (tail[3] >> bsh) & 0xFFu;
    uint qs4 = rw1(bd + 2u + ib32 * 4u);             // qs[l] for l=0..3
    uint sg4 = rw1(bd + 2u + 32u + ib32 * 4u);       // sign bytes for l=0..3
#else
    uint qh = rb(bd + 66u + ib32);
    uint sc = rb(bd + 74u + ib32);
    uint qs4 = ru32u(bd + 2u + ib32 * 4u);          // qs[l] for l=0..3
    uint sg4 = ru32u(bd + 2u + 32u + ib32 * 4u);    // sign bytes for l=0..3
#endif
    float dl0 = d * (0.5 + float(sc & 0xFu)) * 0.25;
    float dl1 = d * (0.5 + float(sc >> 4u)) * 0.25;
    for (uint l = 0u; l < 4u; l++) {
        uint grid_idx = ((qs4 >> (8u * l)) & 0xFFu) | ((qh << (8u - 2u * l)) & 0x300u);
        uint signs = (sg4 >> (8u * l)) & 0xFFu;
        float dl = (l < 2u) ? dl0 : dl1;
        uint gw0 = G_IQ2S[2u * grid_idx];
        uint gw1 = G_IQ2S[2u * grid_idx + 1u];
        for (uint j = 0u; j < 8u; j++) {
            uint bv = (j < 4u) ? ((gw0 >> (8u * j)) & 0xFFu) : ((gw1 >> (8u * (j - 4u))) & 0xFFu);
            float sign = (((signs >> j) & 1u) != 0u) ? -1.0 : 1.0;
            v[l * 8u + j] = dl * float(sgn8(bv)) * sign;
        }
    }
}
#endif

#if defined(FMT_IQ3XXS)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 98u;
    float d = f16tof32(ru16(bd));
    uint ib32 = p / 32u; uint l = (p % 32u) / 8u; uint j8 = p % 8u;
    uint g1_idx = rb(bd + 2u + ib32 * 8u + 2u * l);
    uint g2_idx = rb(bd + 2u + ib32 * 8u + 2u * l + 1u);
    uint aux32 = ru32b(bd + 66u + 4u * ib32);
    float db = d * (0.5 + float(aux32 >> 28u)) * 0.5;
    uint signs = KSIGNS[(aux32 >> (7u * l)) & 127u];
    uint gidx = (j8 < 4u) ? g1_idx : g2_idx;
    uint bytej = (j8 < 4u) ? j8 : (j8 - 4u);
    uint bv = (G_IQ3XXS[gidx] >> (8u * bytej)) & 0xFFu;
    float sign = (((signs >> j8) & 1u) != 0u) ? -1.0 : 1.0;
    return db * float(sgn8(bv)) * sign;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 98u; uint ib32 = (gstart % 256u) / 32u;
    float d = f16tof32(ru16(bd));
    uint aux32 = ru32u(bd + 66u + 4u * ib32);
    float db = d * (0.5 + float(aux32 >> 28u)) * 0.5;
    uvec2 qp2 = rw2(bd + 2u + ib32 * 8u);          // fused qs pair (see rw2)
    uint qs8a = qp2.x;                             // grid-index bytes 0..3 (l=0,1)
    uint qs8b = qp2.y;                             // grid-index bytes 4..7 (l=2,3)
    for (uint l = 0u; l < 4u; l++) {
        uint pairw = (l < 2u) ? qs8a : qs8b;
        uint g1_idx = (pairw >> (16u * (l & 1u))) & 0xFFu;
        uint g2_idx = (pairw >> (16u * (l & 1u) + 8u)) & 0xFFu;
        uint signs = KSIGNS[(aux32 >> (7u * l)) & 127u];
        uint gw1 = G_IQ3XXS[g1_idx];
        uint gw2 = G_IQ3XXS[g2_idx];
        for (uint j8 = 0u; j8 < 8u; j8++) {
            uint bv = (j8 < 4u) ? ((gw1 >> (8u * j8)) & 0xFFu) : ((gw2 >> (8u * (j8 - 4u))) & 0xFFu);
            float sign = (((signs >> j8) & 1u) != 0u) ? -1.0 : 1.0;
            v[l * 8u + j8] = db * float(sgn8(bv)) * sign;
        }
    }
}
#endif

#if defined(FMT_IQ3S)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 110u;
    float d = f16tof32(ru16(bd));
    uint pair = p / 64u; uint grp = (p % 64u) / 32u; uint within = p % 32u; uint l = within / 8u; uint j8 = within % 8u;
    uint sc = rb(bd + 106u + pair);
    float db = (grp == 1u) ? (d * (1.0 + 2.0 * float(sc >> 4u))) : (d * (1.0 + 2.0 * float(sc & 0xFu)));
    uint qh = rb(bd + 66u + pair * 2u + grp);
    uint qs_base = bd + 2u + pair * 16u + grp * 8u;
    uint signs_byte = rb(bd + 74u + pair * 8u + grp * 4u + l);
    uint grididx; uint bytej;
    if (j8 < 4u) { grididx = rb(qs_base + 2u * l) | ((qh << (8u - 2u * l)) & 256u); bytej = j8; }
    else { grididx = rb(qs_base + 2u * l + 1u) | ((qh << (7u - 2u * l)) & 256u); bytej = j8 - 4u; }
    uint bv = (G_IQ3S[grididx] >> (8u * bytej)) & 0xFFu;
    float sign = (((signs_byte >> j8) & 1u) != 0u) ? -1.0 : 1.0;
    return db * float(sgn8(bv)) * sign;
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint p0 = gstart % 256u; uint bd = (gstart / 256u) * 110u;
    uint pair = p0 / 64u; uint grp = (p0 % 64u) / 32u;
    float d = f16tof32(ru16(bd));
    uint sc = rb(bd + 106u + pair);
    float db = (grp == 1u) ? (d * (1.0 + 2.0 * float(sc >> 4u))) : (d * (1.0 + 2.0 * float(sc & 0xFu)));
    uint qh = rb(bd + 66u + pair * 2u + grp);
    uvec2 qp2 = rw2(bd + 2u + pair * 16u + grp * 8u);          // fused qs pair (see rw2)
    uint qs8a = qp2.x;                                         // qs bytes 0..3 (l=0,1)
    uint qs8b = qp2.y;                                         // qs bytes 4..7 (l=2,3)
    uint sg4 = ru32u(bd + 74u + pair * 8u + grp * 4u);         // signs bytes for l=0..3
    for (uint l = 0u; l < 4u; l++) {
        uint pairw = (l < 2u) ? qs8a : qs8b;
        uint q0 = (pairw >> (16u * (l & 1u))) & 0xFFu;
        uint q1 = (pairw >> (16u * (l & 1u) + 8u)) & 0xFFu;
        uint gi0 = q0 | ((qh << (8u - 2u * l)) & 256u);
        uint gi1 = q1 | ((qh << (7u - 2u * l)) & 256u);
        uint signs = (sg4 >> (8u * l)) & 0xFFu;
        uint gw0 = G_IQ3S[gi0];
        uint gw1 = G_IQ3S[gi1];
        for (uint j8 = 0u; j8 < 8u; j8++) {
            uint bv = (j8 < 4u) ? ((gw0 >> (8u * j8)) & 0xFFu) : ((gw1 >> (8u * (j8 - 4u))) & 0xFFu);
            float sign = (((signs >> j8) & 1u) != 0u) ? -1.0 : 1.0;
            v[l * 8u + j8] = db * float(sgn8(bv)) * sign;
        }
    }
}
#endif

#if defined(FMT_IQ1S)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 50u;
    float d = f16tof32(ru16(bd));
    uint ib = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint qh = ru16(bd + 34u + 2u * ib);
    float dl = d * (2.0 * float((qh >> 12u) & 7u) + 1.0);
    float delta = ((qh & 0x8000u) != 0u) ? -0.125 : 0.125;
    uint grid_idx = rb(bd + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u);
    uint bv = (j < 4u) ? ((G_IQ1S[2u * grid_idx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ1S[2u * grid_idx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    return dl * (float(sgn8(bv)) + delta);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 50u; uint ib = (gstart % 256u) / 32u;
    float d = f16tof32(ru16(bd));
    uint qh = ru16(bd + 34u + 2u * ib);
    float dl = d * (2.0 * float((qh >> 12u) & 7u) + 1.0);
    float delta = ((qh & 0x8000u) != 0u) ? -0.125 : 0.125;
    uint qs4 = ru32u(bd + 2u + ib * 4u);  // qs bytes for l=0..3
    for (uint l = 0u; l < 4u; l++) {
        uint grid_idx = ((qs4 >> (8u * l)) & 0xFFu) | (((qh >> (3u * l)) & 7u) << 8u);
        uint gw0 = G_IQ1S[2u * grid_idx];
        uint gw1 = G_IQ1S[2u * grid_idx + 1u];
        for (uint j = 0u; j < 8u; j++) {
            uint bv = (j < 4u) ? ((gw0 >> (8u * j)) & 0xFFu) : ((gw1 >> (8u * (j - 4u))) & 0xFFu);
            v[l * 8u + j] = dl * (float(sgn8(bv)) + delta);
        }
    }
}
#endif

#if defined(FMT_IQ1M)
float dq(uint g) {
    uint p = g % 256u; uint bd = (g / 256u) * 56u;
    uint sc0 = ru16(bd + 48u); uint sc1 = ru16(bd + 50u); uint sc2 = ru16(bd + 52u); uint sc3 = ru16(bd + 54u);
    uint d_bits = (sc0 >> 12u) | ((sc1 >> 8u) & 0xF0u) | ((sc2 >> 4u) & 0xF00u) | (sc3 & 0xF000u);
    float d = f16tof32(d_bits);
    uint ib = p / 32u; uint l = (p % 32u) / 8u; uint j = p % 8u;
    uint scidx = ib / 2u; uint scw;
    if (scidx == 0u) { scw = sc0; } else if (scidx == 1u) { scw = sc1; }
    else if (scidx == 2u) { scw = sc2; } else { scw = sc3; }
    float dl1 = d * (2.0 * float((scw >> (6u * (ib & 1u))) & 7u) + 1.0);
    float dl2 = d * (2.0 * float((scw >> (6u * (ib & 1u) + 3u)) & 7u) + 1.0);
    float dl = (l >= 2u) ? dl2 : dl1;
    uint qh0 = rb(bd + 32u + ib * 2u); uint qh1 = rb(bd + 32u + ib * 2u + 1u);
    uint grididx; bool deltaneg;
    if (l == 0u) { grididx = rb(bd + ib * 4u) | ((qh0 << 8u) & 0x700u); deltaneg = (qh0 & 0x08u) != 0u; }
    else if (l == 1u) { grididx = rb(bd + ib * 4u + 1u) | ((qh0 << 4u) & 0x700u); deltaneg = (qh0 & 0x80u) != 0u; }
    else if (l == 2u) { grididx = rb(bd + ib * 4u + 2u) | ((qh1 << 8u) & 0x700u); deltaneg = (qh1 & 0x08u) != 0u; }
    else { grididx = rb(bd + ib * 4u + 3u) | ((qh1 << 4u) & 0x700u); deltaneg = (qh1 & 0x80u) != 0u; }
    float delta = deltaneg ? -0.125 : 0.125;
    uint bv = (j < 4u) ? ((G_IQ1S[2u * grididx] >> (8u * j)) & 0xFFu)
                       : ((G_IQ1S[2u * grididx + 1u] >> (8u * (j - 4u))) & 0xFFu);
    return dl * (float(sgn8(bv)) + delta);
}
#define HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    uint bd = (gstart / 256u) * 56u; uint ib = (gstart % 256u) / 32u;
    uint sc0 = ru16(bd + 48u); uint sc1 = ru16(bd + 50u); uint sc2 = ru16(bd + 52u); uint sc3 = ru16(bd + 54u);
    uint d_bits = (sc0 >> 12u) | ((sc1 >> 8u) & 0xF0u) | ((sc2 >> 4u) & 0xF00u) | (sc3 & 0xF000u);
    float d = f16tof32(d_bits);
    uint scidx = ib / 2u; uint scw;
    if (scidx == 0u) { scw = sc0; } else if (scidx == 1u) { scw = sc1; }
    else if (scidx == 2u) { scw = sc2; } else { scw = sc3; }
    float dl1 = d * (2.0 * float((scw >> (6u * (ib & 1u))) & 7u) + 1.0);
    float dl2 = d * (2.0 * float((scw >> (6u * (ib & 1u) + 3u)) & 7u) + 1.0);
    uint qh0 = rb(bd + 32u + ib * 2u); uint qh1 = rb(bd + 32u + ib * 2u + 1u);
    uint qs4 = ru32u(bd + ib * 4u);  // qs bytes for l=0..3
    for (uint l = 0u; l < 4u; l++) {
        uint qsb = (qs4 >> (8u * l)) & 0xFFu;
        uint grididx; bool deltaneg;
        if (l == 0u) { grididx = qsb | ((qh0 << 8u) & 0x700u); deltaneg = (qh0 & 0x08u) != 0u; }
        else if (l == 1u) { grididx = qsb | ((qh0 << 4u) & 0x700u); deltaneg = (qh0 & 0x80u) != 0u; }
        else if (l == 2u) { grididx = qsb | ((qh1 << 8u) & 0x700u); deltaneg = (qh1 & 0x08u) != 0u; }
        else { grididx = qsb | ((qh1 << 4u) & 0x700u); deltaneg = (qh1 & 0x80u) != 0u; }
        float dl = (l >= 2u) ? dl2 : dl1;
        float delta = deltaneg ? -0.125 : 0.125;
        uint gw0 = G_IQ1S[2u * grididx];
        uint gw1 = G_IQ1S[2u * grididx + 1u];
        for (uint j = 0u; j < 8u; j++) {
            uint bv = (j < 4u) ? ((gw0 >> (8u * j)) & 0xFFu) : ((gw1 >> (8u * (j - 4u))) & 0xFFu);
            v[l * 8u + j] = dl * (float(sgn8(bv)) + delta);
        }
    }
}
#endif

// Fallback: formats without an optimized dqblk loop dq() per element (identical math).
#ifndef HAVE_DQBLK
void dqblk(uint gstart, out float v[32]) {
    for (uint w = 0u; w < 32u; w++) { v[w] = dq(gstart + w); }
}
#endif
