// Buffer-device-address (BDA) access to a weight by its 64-bit device address. Included by every
// kernel build that reads weights by device address rather than a bound SSBO. The address source is
// deliberately abstracted away: `w_addr` may be a resident weight's OWN VkDeviceAddress, or a
// pager-arena slot base (`arena_base + lut_slot * slot_bytes`) for the paged expert pool — both are
// just "the weight's 64-bit byte address", and the reader helpers below don't care which.
//
// Addressing by VkDeviceAddress (a buffer created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS) rather
// than a bound SSBO lifts the maxStorageBufferRange cap (~4 GiB on RADV) that a single SSBO binding
// imposes: a per-role paged expert pool can hold as many experts as VRAM allows (higher hit rate,
// fewer PCIe page-ins), and a resident dense weight is reached without an ever-rebound descriptor.
// The host passes the base address split into two u32 push-constant fields (`arena_lo`/`arena_hi` —
// a uvec2 avoids the 8-byte push-constant alignment a uint64_t member would force).
//
// `w_addr` is set ONCE in main() to this weight's base byte address, e.g. a resident tensor:
//     w_addr = arena_base(lo, hi)
// or, for a paged expert slot, with the 64-bit slot multiply (which no arena size can overflow — the
// u32 element-space multiply it replaces was the original coherent-but-wrong bug at slot ≥ ~102 on
// Scout):
//     w_addr = arena_base(lo, hi) + uint64_t(lut_slot) * uint64_t(slot_bytes)
// `arena_word(wi)` then reads word `wi` off that base — the drop-in replacement for the old
// `nw[nw_base + wi]` SSBO read.
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ArenaWords { uint v[]; };

uint64_t w_addr = 0ul; // this weight's base byte address (set once in main: resident BDA or arena slot)

uint64_t arena_base(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }
// Word read, shaped for the GLOBAL_LOAD saddr form: the byte offset is computed fully in 32-bit
// and added to the (wave-uniform) base pointer BEFORE the cast, and the deref index is the
// constant 0 — NIR then sees `iadd(uniform64, u2u64(divergent32))`, which ACO selects as a
// scalar-base global_load (64-bit base in SGPRs, one 32-bit VGPR offset). The former
// `ArenaWords(w_addr).v[wi]` promoted the index to 64-bit inside the deref, defeating saddr:
// every load materialized its own 64-bit VGPR address via a v_add_co/v_add_co_ci pair (measured
// ~1.5 carry-adds per load on the Q6K streamed warp GEMM — the 2.2x flag-on regression vs the
// descriptor-bound twin, which gets base+offset addressing for free from the SGPR descriptor).
// `w_addr` is wave-uniform by construction (push constants, or a LUT slot picked by workgroup id);
// no explicit uniformity hint is needed — this iadd shape alone gets saddr selected (a
// subgroupBroadcastFirst hint on w_addr was tried and measured a wash, so it was dropped).
uint arena_word(uint wi) {
    return ArenaWords(w_addr + uint64_t(wi << 2u)).v[0];
}

// Wide 4-word read. The descriptor-bound twin's `nw[i]` array lets ACO's load/store vectorizer
// fuse four adjacent reads into one buffer_load_b128; the scalar `arena_word` above can't be fused
// (each call is a distinct ArenaW4(base+off).v[0] pointer, so the vectorizer can't prove adjacency)
// and lowers to four global_load_b32 — 4x the load instructions, which on the bandwidth/latency-
// bound decode GEMV was the whole streamed-vs-resident gap. Here the four v[0..3] are CONSTANT
// indices off ONE pointer, so the vectorizer fuses them into a global_load_b128, while the base is
// still built as `w_addr + (wbase<<2)` in 32-bit offset form (saddr scalar base, no v_add_co).
// Requires the word base to be dword-aligned (all Q6K ql/qh word offsets are); b128 needs only
// dword alignment on RDNA, matching buffer_reference_align.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ArenaW4 { uint v[4]; };
uvec4 arena_word4(uint wbase) {
    ArenaW4 p = ArenaW4(w_addr + uint64_t(wbase << 2u));
    return uvec4(p.v[0], p.v[1], p.v[2], p.v[3]);
}
#define NW4(wbase) arena_word4(wbase)

// Wide 2-word read — the b64 twin of NW4, for formats whose natural fused unit is an 8-byte qs
// PAIR (the grid i-quants: IQ3_S/IQ3_XXS/IQ2_XS read two adjacent qs words per 32-elem sub-block).
// Same shape: two CONSTANT indices off ONE pointer fuse into a global_load_b64 with a saddr base,
// where two separate arena_word() calls stay two unfused global_load_b32.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ArenaW2 { uint v[2]; };
uvec2 arena_word2(uint wbase) {
    ArenaW2 p = ArenaW2(w_addr + uint64_t(wbase << 2u));
    return uvec2(p.v[0], p.v[1]);
}
#define NW2(wbase) arena_word2(wbase)
