// Buffer-device-address (BDA) access to the KV cache by its 64-bit device address — the KV-side
// twin of native_weight_addr.glsl (whose `w_addr`/`arena_word`/NW4 shapes this mirrors exactly).
// Included ONLY by the `-DKV_BDA` build of attention_kv.comp; the default (bound-SSBO) build never
// touches this header. See that shader's KV_BDA arm and tests/kv_addr_parity.rs.
//
// WHY TWO POINTERS: attention reads K and V independently (the QKᵀ dot loop, then the P·V weighted
// sum), off two distinct cache tensors at two distinct device addresses — so unlike the weight side
// (one `w_addr`) there are TWO wave-uniform base pointers, `k_addr` and `v_addr`. Each is passed
// from the host as a uvec2 push-constant pair (k_lo/k_hi, v_lo/v_hi) — a uvec2 split avoids the
// 8-byte push-constant alignment a uint64_t member would force, exactly like the weight side's
// arena_lo/arena_hi. main() sets both ONCE via `kv_base(lo, hi)`.
//
// ADDRESSING INVARIANT (same as the weight side): the u64 protects the cache tensor's BASE address;
// the intra-tensor element→byte offset is built entirely in u32 (<2^32 elements, <4 GiB per KV
// tensor at every supported context — the no-overflow claim on issue #74) and added to the base
// BEFORE the uint64_t cast, with a CONSTANT deref index. NIR then sees
// `iadd(uniform64, u2u64(divergent32))`, which ACO selects as a scalar-base global_load (64-bit
// base in SGPRs, one 32-bit VGPR offset) — NOT a per-load 64-bit VGPR address (the v_add_co /
// v_add_co_ci carry-add pair that measured the ~2.2-2.4x regression on the weight-side Q6K warp
// GEMM). The `base` args below are always fed a wave-uniform pointer (k_addr/v_addr, from push
// constants), so no explicit uniformity hint is needed — the iadd shape alone gets saddr selected.
//
// READS use the `readonly buffer` blocks below; WRITES (the KV STORE kernels, #74 slice 3) use the
// separate `writeonly buffer` blocks at the bottom. A single buffer_reference TYPE cannot be both
// `readonly` and `writeonly`, so the two paths get distinct type names over (possibly) the same
// bytes — the aliasing is fine, each access goes through its own typed reference.
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

uint64_t k_addr = 0ul; // K cache base byte address (set once in main from k_lo/k_hi)
uint64_t v_addr = 0ul; // V cache base byte address (set once in main from v_lo/v_hi)

uint64_t kv_base(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }

#ifdef KV_SEGMENTED
// A segmented cache binds a tiny uvec2 address table in the descriptor slot formerly occupied by
// the flat cache. `shift` is log2(logical elements per segment), so lookup is two integer ops and
// one uniform table load; the physical segment itself is still read through the established BDA
// helpers below. Qwen's 32K-row geometry guarantees a power-of-two element count per segment.
uint kv_segment(uint elem, uint shift) { return elem >> shift; }
uint kv_segment_local(uint elem, uint shift) { return elem & ((1u << shift) - 1u); }
uint64_t kv_segment_base(uvec2 address) { return kv_base(address.x, address.y); }
#endif

// ── Scalar reads (used by attention_kv this slice) ─────────────────────────────────────────────
// One f16 element `i` off `base` (the f16 KV cache read: `float(k[i])` / `float(v[i])`). Byte
// offset `i<<1` is built in u32 and added before the cast; deref index is the constant 0 → saddr.
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer KvHalf { float16_t v[]; };
float kv_half(uint64_t base, uint i) {
    return float(KvHalf(base + uint64_t(i << 1u)).v[0]);
}

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer KvF32 { float v[]; };
float kv_f32(uint64_t base, uint i) {
    return KvF32(base + uint64_t(i << 2u)).v[0];
}

// ── Cooperative-matrix tensor base (coopmat flash prefill, #74 slice 4 resurrection) ────────────
// The coopmat flash-prefill QK/PV `coopMatLoad` reads the KV tensor straight from a buffer_reference
// base: `coopMatLoad(M, KvMat(k_addr).v, elem_off, stride, layout)`. The pointer carries ONLY the
// wave-uniform arena base (k_addr / v_addr); the FULL intra-tensor element offset + row stride stay in
// coopMatLoad's own 32-bit args — the slice-4-PROVEN spelling (base = wave-uniform push base, all
// offset in the element/stride args). `v[]` is an unsized f16 runtime array, byte offset built by the
// driver's per-lane coopmat addressing. WHY OPT-IN (RADV/RDNA3): RADV's coopMatLoad lowering emits
// opaque per-lane addressing and will NOT select saddr from a buffer_reference base → per-lane 64-bit
// carry-adds (~260-520 v_add_co/v_addc), ~+33-40% code, prefill ~0.80-0.83x. So on RDNA3 the bound
// descriptor stays the DEFAULT; `-DKV_COOPMAT_BDA` is the alternate for silicon that addresses coopmat
// loads better (NVIDIA cm2, future drivers). See kv-u64-campaign slice 4 + kv-decode-perf-levers.
// Guarded on KV_COOPMAT_BDA so the decode `-DKV_BDA` includers (no coopmat) don't emit this block.
#ifdef KV_COOPMAT_BDA
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer KvMat { float16_t v[]; };
#endif

// One u32 word `wi` off `base` — the planar-Q8 code/scale read (`ku[wi]` / `vu[wi]`). Identical
// shape to native_weight_addr.glsl's `arena_word`: byte offset `wi<<2` in u32, deref index 0.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer KvWords { uint v[]; };
uint kv_word(uint64_t base, uint wi) {
    return KvWords(base + uint64_t(wi << 2u)).v[0];
}

// ── Wide constant-index reads (declared now; consumed by slices 2+ which read f16vec4 KV) ───────
// The b128/b64 analogs of native_weight_addr.glsl's NW4/NW2: four (KV4) or two (KV2) CONSTANT
// indices off ONE pointer, so ACO's load/store vectorizer fuses them into a single
// global_load_b128 / global_load_b64 with a saddr scalar base — where N separate kv_word() calls
// would stay N unfused global_load_b32. `wbase` is a u32 WORD base (dword-aligned); the byte offset
// `wbase<<2` is built in u32 exactly as the scalar reads above. Requires only dword alignment on
// RDNA (b128/b64 both), matching buffer_reference_align.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer KvW4 { uint v[4]; };
uvec4 kv_word4(uint64_t base, uint wbase) {
    KvW4 p = KvW4(base + uint64_t(wbase << 2u));
    return uvec4(p.v[0], p.v[1], p.v[2], p.v[3]);
}
#define KV4(base, wbase) kv_word4(base, wbase)

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer KvW2 { uint v[2]; };
uvec2 kv_word2(uint64_t base, uint wbase) {
    KvW2 p = KvW2(base + uint64_t(wbase << 2u));
    return uvec2(p.v[0], p.v[1]);
}
#define KV2(base, wbase) kv_word2(base, wbase)

// ── WRITABLE stores (KV STORE kernels, #74 slice 3) ─────────────────────────────────────────────
// The store kernels write their DESTINATION KV cache by device address. A store touches exactly ONE
// KV buffer, so it reuses the `k_addr` global above as that buffer's base (set once in main via
// kv_base) — v_addr stays 0/unused. Same saddr shape as the reads: the element→byte offset is built
// entirely in u32 and added to the base BEFORE the uint64_t cast, with a CONSTANT deref index, so
// NIR sees `iadd(uniform64, u2u64(divergent32))` → ACO selects a scalar-base `global_store_*` (64-bit
// base in SGPRs, one 32-bit VGPR offset) with NO per-store v_add_co/v_addc carry-add pair. These are
// scalar single-element stores (no kernel here writes a vec4), so there is no wide-store analog.
//
// NOTE the store kernels keep their destination BOUND at its write slot as an INERT descriptor the
// `-DKV_BDA` shader never touches — a BDA store is invisible to `Recorder::sync`'s descriptor hazard
// tracker, so the bound-at-write-slot dst is what preserves the store→(later)read barrier (the
// write-side analog of slice 1's inert read bind). See each Recorder `*_at` store method.

// One f16 element `i` (byte offset `i<<1`): store_f16 / qk_norm_rope* / rope(-DOUT_F16). float16 is
// enabled by every kv_addr includer (all read + write f16), so this block is unconditional.
layout(buffer_reference, std430, buffer_reference_align = 2) writeonly buffer KvHalfW { float16_t v[]; };
void kv_store_half(uint64_t base, uint i, float val) {
    KvHalfW(base + uint64_t(i << 1u)).v[0] = float16_t(val);
}

// One f32 element `i` (byte offset `i<<2`): store_kv_dense -DDST_F32. Core `float` type, unconditional.
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer KvF32W { float v[]; };
void kv_store_f32(uint64_t base, uint i, float val) {
    KvF32W(base + uint64_t(i << 2u)).v[0] = val;
}

// One u16 element `i` (byte offset `i<<1`): store_kv_dense -DDST_BF16 writes raw bf16 bits. Needs the
// int16 type — GUARDED so the f16-only readers (attention_kv / attn_partial, no int16 ext) still
// compile the header. Includers that write bf16 KV define KV_STORE_U16 before the include.
#ifdef KV_STORE_U16
layout(buffer_reference, std430, buffer_reference_align = 2) writeonly buffer KvU16W { uint16_t v[]; };
void kv_store_u16(uint64_t base, uint i, uint bits) {
    KvU16W(base + uint64_t(i << 1u)).v[0] = uint16_t(bits);
}
#endif

// One byte at BYTE offset `bo` (already in bytes → no shift): the low-bit quant packers
// (store_q8 / quant_kv) write their blocks byte-by-byte. Needs int8 + 8bit_storage — GUARDED
// (includers define KV_STORE_BYTE before the include). Byte stores don't vectorize; the quant
// packers are cold, so this is fine — the hot f16 stores go through kv_store_half.
#ifdef KV_STORE_BYTE
layout(buffer_reference, std430, buffer_reference_align = 1) writeonly buffer KvByteW { uint8_t v[]; };
void kv_store_byte(uint64_t base, uint bo, uint val) {
    KvByteW(base + uint64_t(bo)).v[0] = uint8_t(val);
}
#endif
