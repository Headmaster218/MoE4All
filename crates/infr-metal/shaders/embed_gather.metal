// ---- Embedding-row gather + dequant (Op::EmbedGather): dst[r, :] = dequant(table[ids[r], :]) * scale.
//
// The host feeds TOKEN IDS (4 bytes each) instead of dequantized f32 embedding rows; the table is
// the resident quantized token_embd weight, bound as its RAW GGUF bytes (same buffer the native
// linear kernels read — no host repack, no extra residency). One simdgroup (32 lanes) per row:
// each lane decodes whole 16-element blocks of the token's row via the SAME DEC16_* decode macros
// the GEMV/RT/CMM family uses (defined in linear.metal, which MUST precede this file in the MSL
// concatenation), so the dequant math is bit-identical to the linear path by construction.
//
// `ids` is the graph's raw I32 input buffer. Keeping the native representation lets a recorded
// decode tape read the host's current token upload directly, and matches the uint bit pattern
// written by Op::Argmax / Op::Sample when their output aliases the next gather's input.
//
// `scale` bakes Gemma's sqrt(n_embd) embed scaling (1.0 elsewhere).
struct EmbedGatherParams {
    uint rows;
    uint ne;
    float scale;
};

// Quantized table: DEC decodes 16-element block `bi` (global 16-block index, ambient `codes`)
// into wk[16] — exactly the GEMV_KERNEL decode contract. Requires ne % 16 == 0 (the runner gates
// n_embd % 32 == 0, and GGUF rows are whole blocks of the stored format).
#define EMBED_GATHER_KERNEL(NAME, DEC)                                                            \
kernel void NAME(device const uchar*  codes [[buffer(0)]],                                       \
                 device const int*    ids   [[buffer(1)]],                                       \
                 device float*        dst   [[buffer(2)]],                                       \
                 constant EmbedGatherParams& p [[buffer(3)]],                                    \
                 uint gid [[thread_position_in_grid]]) {                                         \
    uint row = gid >> 5u;                                                                        \
    uint lane = gid & 31u;                                                                       \
    if (row >= p.rows) return;                                                                   \
    uint tok = (uint)ids[row];                                                                   \
    uint nb = p.ne >> 4u;                                                                        \
    ulong row16 = (ulong)tok * nb;                                                               \
    ulong obase = (ulong)row * p.ne;                                                             \
    for (uint b = lane; b < nb; b += 32u) {                                                      \
        ulong bi = row16 + b;                                                                    \
        float wk[16];                                                                            \
        DEC(wk)                                                                                  \
        device float* o = dst + obase + ((ulong)b << 4u);                                        \
        for (uint k = 0; k < 16u; k++) o[k] = wk[k] * p.scale;                                   \
    }                                                                                            \
}

EMBED_GATHER_KERNEL(embed_gather_q4k, DEC16_Q4K)
EMBED_GATHER_KERNEL(embed_gather_q6k, DEC16_Q6K)
EMBED_GATHER_KERNEL(embed_gather_q8_0, DEC16_Q8_0)
EMBED_GATHER_KERNEL(embed_gather_q4_0, DEC16_Q4_0)
EMBED_GATHER_KERNEL(embed_gather_q5_0, DEC16_Q5_0)
EMBED_GATHER_KERNEL(embed_gather_q4_1, DEC16_Q4_1)
EMBED_GATHER_KERNEL(embed_gather_q5_1, DEC16_Q5_1)
EMBED_GATHER_KERNEL(embed_gather_mxfp4, DEC16_MXFP4)
EMBED_GATHER_KERNEL(embed_gather_nvfp4, DEC16_NVFP4)
EMBED_GATHER_KERNEL(embed_gather_q2_0, DEC16_Q2_0)
EMBED_GATHER_KERNEL(embed_gather_tq2_0, DEC16_TQ2_0)
EMBED_GATHER_KERNEL(embed_gather_tq1_0, DEC16_TQ1_0)
EMBED_GATHER_KERNEL(embed_gather_iq4nl, DEC16_IQ4NL)
EMBED_GATHER_KERNEL(embed_gather_iq4xs, DEC16_IQ4XS)

// f16 table: contiguous half rows, dequant is a plain widen.
kernel void embed_gather_f16(device const half*   table [[buffer(0)]],
                             device const int*    ids   [[buffer(1)]],
                             device float*        dst   [[buffer(2)]],
                             constant EmbedGatherParams& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint row = gid >> 5u;
    uint lane = gid & 31u;
    if (row >= p.rows) return;
    ulong src = (ulong)((uint)ids[row]) * p.ne;
    ulong o = (ulong)row * p.ne;
    for (uint k = lane; k < p.ne; k += 32u) dst[o + k] = (float)table[src + k] * p.scale;
}

// bf16 table: top 16 bits of the f32; dequant is a lossless << 16 (same as dequant_bf16_f16).
kernel void embed_gather_bf16(device const ushort* table [[buffer(0)]],
                              device const int*    ids   [[buffer(1)]],
                              device float*        dst   [[buffer(2)]],
                              constant EmbedGatherParams& p [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    uint row = gid >> 5u;
    uint lane = gid & 31u;
    if (row >= p.rows) return;
    ulong src = (ulong)((uint)ids[row]) * p.ne;
    ulong o = (ulong)row * p.ne;
    for (uint k = lane; k < p.ne; k += 32u) {
        dst[o + k] = as_type<float>((uint)table[src + k] << 16u) * p.scale;
    }
}
