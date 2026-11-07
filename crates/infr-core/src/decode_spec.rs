//! **The** block-decode spec: one named description of every GGUF block format's on-disk geometry
//! and scale layout, that the two shader families (GLSL / MSL) and the host decoder are
//! *checked against* instead of each carrying its own copy of the numbers.
//!
//! Precedent: [`crate::iquant_grids`] already single-sources the IQ codebooks (cpu/metal read them
//! directly, vulkan emits them into shaders). This module does the same for the part of a block
//! format that is *not* a codebook — how many elements a block holds, how many bytes it occupies,
//! and where its scale field(s) live inside it.
//!
//! ## What lives here, and what does not
//!
//! - **Here:** `(elements, bytes)` per block ([`block_layout`]), the byte offset + encoding of each
//!   block's scale field(s) ([`BlockSpec::scales`]), and the roster of real GGUF weight quants
//!   ([`WEIGHT_QUANTS`]).
//! - **NOT here:** the decode arithmetic. `infr_gguf::dequant::dequant_block` stays the single host
//!   ORACLE, and each backend's kernel stays the device implementation. A fourth copy of the decode
//!   in Rust would be a liability, not a spec.
//!
//! The spec is only worth having if it is *load-bearing*, so it is wired both ways:
//! `infr_gguf::block_layout` and `dequant_factored`'s geometry now read from [`block_layout`],
//! and `infr-testkit`'s parity harness synthesizes valid
//! blocks for EVERY weight quant purely from [`BlockSpec`] — so a wrong offset here fails a test
//! rather than sitting unread.
//!
//! Sizes/offsets are the llama.cpp `ggml/src/ggml-quants.c` block structs; each is spelled out in
//! the `block_spec` arm that declares it.

use crate::DType;

/// How a block's scale field is encoded in the raw bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleEnc {
    /// IEEE binary16, little-endian, 2 bytes. Every affine/k-quant/i-quant/ternary format's `d`
    /// (and `dmin`/`m`) is one of these.
    F16,
    /// MXFP4's single-byte E8M0 shared exponent: `d = 2^(e - 127)`.
    E8M0,
    /// NVFP4's `n` consecutive single-byte UE4M3 per-sub-block scales.
    Ue4m3 { n: usize },
    /// IQ1_M's split `d`: there is no standalone `d` field — the f16 bits are the TOP nibbles of
    /// the four little-endian `u16` scale words starting at the slot offset (nibble `i` of `d`
    /// lives in bits 12..16 of word `i`). The low 12 bits of each word are the four 3-bit `dl`
    /// sub-scales, which are payload, not scale.
    Iq1mSplitF16,
}

impl ScaleEnc {
    /// Bytes this encoding occupies starting at its slot offset.
    pub fn width(self) -> usize {
        match self {
            ScaleEnc::F16 => 2,
            ScaleEnc::E8M0 => 1,
            ScaleEnc::Ue4m3 { n } => n,
            ScaleEnc::Iq1mSplitF16 => 8,
        }
    }
}

/// What a scale field means in the block's reconstruction formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleRole {
    /// The multiplicative super-scale `d`.
    D,
    /// The additive offset — `m` for the affine legacy quants (`y = d·q + m`), `dmin` for the
    /// k-quants (`y = (d·sc)·q − (dmin·mn)`). Only affine formats have one.
    Min,
}

/// One scale field of a block: where it starts, how it is encoded, and what it means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleSlot {
    /// Byte offset from the start of the block.
    pub offset: usize,
    pub enc: ScaleEnc,
    pub role: ScaleRole,
}

const fn d_f16(offset: usize) -> ScaleSlot {
    ScaleSlot {
        offset,
        enc: ScaleEnc::F16,
        role: ScaleRole::D,
    }
}

const fn min_f16(offset: usize) -> ScaleSlot {
    ScaleSlot {
        offset,
        enc: ScaleEnc::F16,
        role: ScaleRole::Min,
    }
}

// The distinct slot lists, named once so `block_spec` can stay a `const fn` (a `&[..]` literal
// containing calls cannot be promoted to `'static` inside a function body).
const NO_SCALE: &[ScaleSlot] = &[];
const D_AT_0: &[ScaleSlot] = &[d_f16(0)];
const D_AT_0_MIN_AT_2: &[ScaleSlot] = &[d_f16(0), min_f16(2)];
const D_AT_80_MIN_AT_82: &[ScaleSlot] = &[d_f16(80), min_f16(82)];
const D_AT_52: &[ScaleSlot] = &[d_f16(52)];
const D_AT_64: &[ScaleSlot] = &[d_f16(64)];
const D_AT_108: &[ScaleSlot] = &[d_f16(108)];
const D_AT_208: &[ScaleSlot] = &[d_f16(208)];
const IQ1M_SPLIT_D: &[ScaleSlot] = &[ScaleSlot {
    offset: 48,
    enc: ScaleEnc::Iq1mSplitF16,
    role: ScaleRole::D,
}];
const MXFP4_E8M0_D: &[ScaleSlot] = &[ScaleSlot {
    offset: 0,
    enc: ScaleEnc::E8M0,
    role: ScaleRole::D,
}];
const NVFP4_UE4M3_D: &[ScaleSlot] = &[ScaleSlot {
    offset: 0,
    enc: ScaleEnc::Ue4m3 { n: 4 },
    role: ScaleRole::D,
}];

/// The full decode spec for one [`DType`]'s block format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSpec {
    pub dtype: DType,
    /// Elements per block (`ggml` `blck_size`). `1` for the scalar float dtypes.
    pub block_elems: usize,
    /// Bytes per block (`ggml` `type_size`).
    pub block_bytes: usize,
    /// Scale fields in the block, in ascending offset order. Empty for the scalar float dtypes
    /// (the value IS the datum) and for `I2S`, whose only scale is per-TENSOR, not per-block.
    pub scales: &'static [ScaleSlot],
    /// Lowercase kernel-suffix name, as the backends spell it (`q4k`, `iq4_xs`, …). The name the
    /// GLSL/MSL decoders are keyed by.
    pub name: &'static str,
}

impl BlockSpec {
    /// Bytes for `numel` elements. Panics in debug if `numel` is not a whole number of blocks —
    /// a partial block has no defined byte size.
    pub fn nbytes(&self, numel: usize) -> usize {
        debug_assert_eq!(
            numel % self.block_elems,
            0,
            "{:?}: {numel} elements is not a whole number of {}-element blocks",
            self.dtype,
            self.block_elems
        );
        numel / self.block_elems * self.block_bytes
    }

    /// The `D`-role slot, if the format has one.
    pub fn d_slot(&self) -> Option<ScaleSlot> {
        self.scales.iter().copied().find(|s| s.role == ScaleRole::D)
    }

    /// The `Min`-role slot, if the format has one (affine formats only).
    pub fn min_slot(&self) -> Option<ScaleSlot> {
        self.scales
            .iter()
            .copied()
            .find(|s| s.role == ScaleRole::Min)
    }

    /// Write `d` (and `min`, where the format has one) into `blk` in this format's own encoding.
    ///
    /// This is the primitive that lets a test synthesize a VALID block of any quant format from
    /// nothing but the spec: fill `blk` with arbitrary bytes (every format's code/nibble/grid
    /// fields decode to finite values for any bit pattern), then call this so the scale slots hold
    /// sane finite magnitudes. `E8M0`/`UE4M3` slots take the nearest representable encoding of
    /// `d`; `Iq1mSplitF16` distributes the f16 bits of `d` across the four scale words' top
    /// nibbles, leaving their low 12 bits (the `dl` payload) untouched.
    ///
    /// Panics if `blk` is shorter than [`BlockSpec::block_bytes`].
    pub fn write_scales(&self, blk: &mut [u8], d: f32, min: f32) {
        assert!(
            blk.len() >= self.block_bytes,
            "{:?}: block buffer is {} bytes, need {}",
            self.dtype,
            blk.len(),
            self.block_bytes
        );
        for s in self.scales {
            let v = match s.role {
                ScaleRole::D => d,
                ScaleRole::Min => min,
            };
            match s.enc {
                ScaleEnc::F16 => {
                    blk[s.offset..s.offset + 2]
                        .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
                }
                ScaleEnc::E8M0 => blk[s.offset] = e8m0_from_f32(v),
                ScaleEnc::Ue4m3 { n } => {
                    // Spread the sub-block scales over a moderate band around `v` so a test
                    // exercises DISTINCT sub-scales rather than n copies of one value.
                    for i in 0..n {
                        blk[s.offset + i] = ue4m3_from_f32(v * (0.625 + 0.125 * i as f32));
                    }
                }
                ScaleEnc::Iq1mSplitF16 => {
                    let bits = half::f16::from_f32(v).to_bits();
                    for i in 0..4 {
                        let o = s.offset + 2 * i;
                        let word = u16::from_le_bytes([blk[o], blk[o + 1]]);
                        let nib = (bits >> (4 * i)) & 0xf;
                        let out = (nib << 12) | (word & 0x0fff);
                        blk[o..o + 2].copy_from_slice(&out.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// Nearest E8M0 byte for a positive `v`: the exponent `e` such that `2^(e-127)` is closest to `v`
/// in log space, clamped to the finite band. (MXFP4's shared exponent.)
fn e8m0_from_f32(v: f32) -> u8 {
    let e = (v.abs().max(f32::MIN_POSITIVE).log2().round() as i32) + 127;
    e.clamp(1, 254) as u8
}

/// Nearest UE4M3 byte for a positive `v` (unsigned 4-bit exponent, 3-bit mantissa, bias 7 — the
/// NVFP4 sub-block scale). Value = `2^(e-7) · (1 + mant/8)` for `e > 0`.
fn ue4m3_from_f32(v: f32) -> u8 {
    let v = v.abs();
    if v <= 0.0 {
        return 0;
    }
    let mut best = 1u8;
    let mut best_err = f32::INFINITY;
    for code in 1u8..=0x7e {
        let e = (code >> 3) as i32;
        let m = (code & 7) as f32 / 8.0;
        let val = if e == 0 {
            m * 2f32.powi(-6)
        } else {
            (1.0 + m) * 2f32.powi(e - 7)
        };
        let err = (val - v).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}

/// Every real GGUF **weight** quant format, in `DType` declaration order.
///
/// Excluded on purpose: the dense float dtypes (F32/F16/BF16 — no block structure to spec),
/// `I32`/`U32` (never weights), `Turbo2`/`Turbo3`/`Turbo4` (KV-cache-only formats, never GGUF
/// weights), and `I2S` (BitNet i2_s carries ONE per-TENSOR f32 scale after the codes, which the
/// per-block model cannot express — the seam host-converts it to f16 at weight load, so no backend
/// ever sees it). This is exactly the set [`DType::is_quant`] accepts, pinned as a list so a
/// harness can sweep it; `weight_quants_match_is_quant` holds the two in lockstep.
pub const WEIGHT_QUANTS: &[DType] = &[
    DType::Q4_0,
    DType::Q4_1,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q8_0,
    DType::Q2K,
    DType::Q3K,
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Iq1S,
    DType::Iq1M,
    DType::Iq2Xxs,
    DType::Iq2Xs,
    DType::Iq2S,
    DType::Iq3Xxs,
    DType::Iq3S,
    DType::Iq4Nl,
    DType::Iq4Xs,
    DType::Tq1_0,
    DType::Tq2_0,
    DType::Q2_0,
    DType::Mxfp4,
    DType::Nvfp4,
];

/// The KV-cache-only TurboQuant formats (never GGUF weights). Broken out so a KV-side harness has
/// the same named roster as the weight side.
pub const KV_ONLY_QUANTS: &[DType] = &[DType::Turbo2, DType::Turbo3, DType::Turbo4];

/// The decode spec for `dtype`. Total — every [`DType`] has one, so a new variant must decide here.
pub const fn block_spec(dtype: DType) -> BlockSpec {
    // Every arm: (block_elems, block_bytes, scale slots, kernel-suffix name). Layouts from
    // llama.cpp `ggml/src/ggml-quants.c`; the byte offsets are the scale fields' positions in
    // those structs and are cross-checked against `infr_gguf::dequant`'s decoders by
    // `infr-testkit`'s `spec_scale_slots_are_the_real_scales`.
    macro_rules! spec {
        ($e:expr, $b:expr, $s:expr, $n:expr) => {
            BlockSpec {
                dtype,
                block_elems: $e,
                block_bytes: $b,
                scales: $s,
                name: $n,
            }
        };
    }
    match dtype {
        // ── dense floats: one "block" per element, no scale ──
        DType::F32 => spec!(1, 4, NO_SCALE, "f32"),
        DType::F16 => spec!(1, 2, NO_SCALE, "f16"),
        DType::Bf16 => spec!(1, 2, NO_SCALE, "bf16"),
        DType::I32 => spec!(1, 4, NO_SCALE, "i32"),
        DType::U32 => spec!(1, 4, NO_SCALE, "u32"),
        // ── legacy round quants (QK=32) ──
        // block_q4_0: [half d][u8 qs[16]]
        DType::Q4_0 => spec!(32, 18, D_AT_0, "q4_0"),
        // block_q4_1: [half d][half m][u8 qs[16]] — affine, y = d·q + m
        DType::Q4_1 => spec!(32, 20, D_AT_0_MIN_AT_2, "q4_1"),
        // block_q5_0: [half d][u8 qh[4]][u8 qs[16]]
        DType::Q5_0 => spec!(32, 22, D_AT_0, "q5_0"),
        // block_q5_1: [half d][half m][u8 qh[4]][u8 qs[16]] — affine
        DType::Q5_1 => spec!(32, 24, D_AT_0_MIN_AT_2, "q5_1"),
        // block_q8_0: [half d][i8 qs[32]]
        DType::Q8_0 => spec!(32, 34, D_AT_0, "q8_0"),
        // ── k-quants (QK_K=256); note Q2_K/Q3_K/Q6_K carry `d` at the block TAIL ──
        // block_q2_K: [u8 scales[16]][u8 qs[64]][half d][half dmin]
        DType::Q2K => spec!(256, 84, D_AT_80_MIN_AT_82, "q2_k"),
        // block_q3_K: [u8 hmask[32]][u8 qs[64]][u8 scales[12]][half d]
        DType::Q3K => spec!(256, 110, D_AT_108, "q3_k"),
        // block_q4_K: [half d][half dmin][u8 scales[12]][u8 qs[128]]
        DType::Q4K => spec!(256, 144, D_AT_0_MIN_AT_2, "q4_k"),
        // block_q5_K: [half d][half dmin][u8 scales[12]][u8 qh[32]][u8 qs[128]]
        DType::Q5K => spec!(256, 176, D_AT_0_MIN_AT_2, "q5_k"),
        // block_q6_K: [u8 ql[128]][u8 qh[64]][i8 scales[16]][half d]
        DType::Q6K => spec!(256, 210, D_AT_208, "q6_k"),
        // ── i-quants (codebook/grid; the grids themselves live in `crate::iquant_grids`) ──
        // block_iq1_s: [half d][u8 qs[32]][u16 qh[8]]
        DType::Iq1S => spec!(256, 50, D_AT_0, "iq1_s"),
        // block_iq1_m: [u8 qs[32]][u8 qh[16]][u8 scales[8]] — NO standalone `d`; see
        // `ScaleEnc::Iq1mSplitF16`.
        DType::Iq1M => spec!(256, 56, IQ1M_SPLIT_D, "iq1_m"),
        // block_iq2_xxs: [half d][u16 qs[32]]
        DType::Iq2Xxs => spec!(256, 66, D_AT_0, "iq2_xxs"),
        // block_iq2_xs: [half d][u16 qs[32]][u8 scales[8]]
        DType::Iq2Xs => spec!(256, 74, D_AT_0, "iq2_xs"),
        // block_iq2_s: [half d][u8 qs[64]][u8 qh[8]][u8 scales[8]]
        DType::Iq2S => spec!(256, 82, D_AT_0, "iq2_s"),
        // block_iq3_xxs: [half d][u8 qs[96]]
        DType::Iq3Xxs => spec!(256, 98, D_AT_0, "iq3_xxs"),
        // block_iq3_s: [half d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]]
        DType::Iq3S => spec!(256, 110, D_AT_0, "iq3_s"),
        // block_iq4_nl: [half d][u8 qs[16]], QK4_NL=32
        DType::Iq4Nl => spec!(32, 18, D_AT_0, "iq4_nl"),
        // block_iq4_xs: [half d][u16 scales_h][u8 scales_l[4]][u8 qs[128]]
        DType::Iq4Xs => spec!(256, 136, D_AT_0, "iq4_xs"),
        // ── ternary quants (QK_K=256 for TQ*, 64 for Q2_0); TQ* carry `d` at the block TAIL ──
        // block_tq1_0: [u8 qs[48]][u8 qh[4]][half d]
        DType::Tq1_0 => spec!(256, 54, D_AT_52, "tq1_0"),
        // block_tq2_0: [u8 qs[64]][half d]
        DType::Tq2_0 => spec!(256, 66, D_AT_64, "tq2_0"),
        // block_q2_0 (Bonsai ternary, 2.25 bpw): [half d][u8 qs[16]], QK2_0=64
        DType::Q2_0 => spec!(64, 18, D_AT_0, "q2_0"),
        // ── fp4 quants (non-f16 scale encodings) ──
        // block_mxfp4: [u8 e (E8M0)][u8 qs[16]], QK_MXFP4=32
        DType::Mxfp4 => spec!(32, 17, MXFP4_E8M0_D, "mxfp4"),
        // block_nvfp4: [u8 d[4] (UE4M3, one per 16 elems)][u8 qs[32]], QK_NVFP4=64
        DType::Nvfp4 => spec!(64, 36, NVFP4_UE4M3_D, "nvfp4"),
        // ── BitNet i2_s: 4 ternary codes per byte; the SINGLE f32 scale is per-TENSOR (stored
        // after all codes), which the per-block model cannot express — hence no scale slot. See
        // `DType::I2S` and `infr_gguf`'s `tensor_nbytes` special case.
        DType::I2S => spec!(4, 1, NO_SCALE, "i2_s"),
        // ── TurboQuant KV-cache formats (128-elem blocks, `norm` f16 at the head) ──
        DType::Turbo2 => spec!(128, 34, D_AT_0, "turbo2"),
        DType::Turbo3 => spec!(128, 50, D_AT_0, "turbo3"),
        DType::Turbo4 => spec!(128, 66, D_AT_0, "turbo4"),
    }
}

/// `(elements_per_block, bytes_per_block)` for `dtype` — the single source of truth every other
/// block-geometry table in the workspace now reads (`infr_gguf::block_layout` re-exports it,
/// `dequant_factored` sizes its blocks with it).
pub const fn block_layout(dtype: DType) -> (usize, usize) {
    let s = block_spec(dtype);
    (s.block_elems, s.block_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The roster and the predicate must not drift: a new quant added to `DType::is_quant` without
    /// a `WEIGHT_QUANTS` entry would silently miss every harness sweep.
    #[test]
    fn weight_quants_match_is_quant() {
        for &dt in WEIGHT_QUANTS {
            assert!(
                dt.is_quant(),
                "{dt:?} is in WEIGHT_QUANTS but not is_quant()"
            );
        }
        assert_eq!(
            WEIGHT_QUANTS.len(),
            24,
            "WEIGHT_QUANTS changed size — update the harness sweeps and this count deliberately"
        );
        // No duplicates.
        for (i, a) in WEIGHT_QUANTS.iter().enumerate() {
            assert!(
                !WEIGHT_QUANTS[..i].contains(a),
                "{a:?} listed twice in WEIGHT_QUANTS"
            );
        }
    }

    /// Every declared scale slot must fit inside its own block — an off-by-one offset here would
    /// otherwise only surface as an out-of-bounds panic deep in a synth helper.
    #[test]
    fn scale_slots_fit_inside_their_block() {
        for &dt in WEIGHT_QUANTS.iter().chain(KV_ONLY_QUANTS) {
            let s = block_spec(dt);
            assert!(s.block_elems > 0 && s.block_bytes > 0, "{dt:?} empty block");
            for slot in s.scales {
                assert!(
                    slot.offset + slot.enc.width() <= s.block_bytes,
                    "{dt:?}: scale slot at {} (+{}) overruns the {}-byte block",
                    slot.offset,
                    slot.enc.width(),
                    s.block_bytes
                );
            }
            // Ascending, non-overlapping.
            for w in s.scales.windows(2) {
                assert!(
                    w[0].offset + w[0].enc.width() <= w[1].offset,
                    "{dt:?}: scale slots out of order / overlapping"
                );
            }
        }
    }

    /// Only AFFINE formats carry an additive offset; a `Min` slot on a non-affine format would mean
    /// `write_scales` corrupts a payload field. (Q2_K/Q4_K/Q5_K's `dmin` and Q4_1/Q5_1's `m`.)
    #[test]
    fn only_affine_formats_have_a_min_slot() {
        let with_min: Vec<DType> = WEIGHT_QUANTS
            .iter()
            .copied()
            .filter(|&dt| block_spec(dt).min_slot().is_some())
            .collect();
        assert_eq!(
            with_min,
            vec![DType::Q4_1, DType::Q5_1, DType::Q2K, DType::Q4K, DType::Q5K]
        );
    }

    /// `block_layout` must reproduce the llama.cpp `type_size` arithmetic for every format — the
    /// numbers restated as their defining formulas so a typo in the table cannot pass.
    #[test]
    fn block_bytes_match_the_ggml_formulas() {
        const QK_K: usize = 256;
        let cases: &[(DType, usize, usize)] = &[
            (DType::Q4_0, 32, 2 + 32 / 2),
            (DType::Q4_1, 32, 2 + 2 + 32 / 2),
            (DType::Q5_0, 32, 2 + 4 + 32 / 2),
            (DType::Q5_1, 32, 2 + 2 + 4 + 32 / 2),
            (DType::Q8_0, 32, 2 + 32),
            (DType::Q2K, QK_K, 2 * 2 + QK_K / 16 + QK_K / 4),
            (DType::Q3K, QK_K, 2 + QK_K / 4 + QK_K / 8 + 12),
            (DType::Q4K, QK_K, 2 * 2 + 12 + QK_K / 2),
            (DType::Q5K, QK_K, 2 * 2 + 12 + QK_K / 8 + QK_K / 2),
            (DType::Q6K, QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2),
            (DType::Iq1S, QK_K, 2 + QK_K / 8 + QK_K / 32 * 2),
            (DType::Iq1M, QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32),
            (DType::Iq2Xxs, QK_K, 2 + QK_K / 8 * 2),
            (DType::Iq2Xs, QK_K, 2 + QK_K / 8 * 2 + QK_K / 32),
            (DType::Iq2S, QK_K, 2 + QK_K / 4 + QK_K / 16),
            (DType::Iq3Xxs, QK_K, 2 + 3 * (QK_K / 8)),
            (DType::Iq3S, QK_K, 2 + 13 * (QK_K / 32) + QK_K / 64),
            (DType::Iq4Nl, 32, 2 + 32 / 2),
            (DType::Iq4Xs, QK_K, 2 + 2 + QK_K / 64 + QK_K / 2),
            (
                DType::Tq1_0,
                QK_K,
                2 + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5,
            ),
            (DType::Tq2_0, QK_K, 2 + QK_K / 4),
            (DType::Q2_0, 64, 2 + 64 / 4),
            (DType::Mxfp4, 32, 1 + 32 / 2),
            (DType::Nvfp4, 64, 4 + 64 / 2),
            (DType::Turbo2, 128, 2 + 32),
            (DType::Turbo3, 128, 2 + 32 + 16),
            (DType::Turbo4, 128, 2 + 64),
        ];
        for &(dt, elems, bytes) in cases {
            assert_eq!(
                block_layout(dt),
                (elems, bytes),
                "{dt:?} block layout disagrees with the ggml formula"
            );
        }
        // Every weight quant is covered by the formula table above.
        for &dt in WEIGHT_QUANTS {
            assert!(
                cases.iter().any(|c| c.0 == dt),
                "{dt:?} has no formula cross-check"
            );
        }
    }

    /// `nbytes` is the block arithmetic, not a per-element size.
    #[test]
    fn nbytes_scales_by_block() {
        let q4k = block_spec(DType::Q4K);
        assert_eq!(q4k.nbytes(0), 0);
        assert_eq!(q4k.nbytes(256), 144);
        assert_eq!(q4k.nbytes(2048), 8 * 144);
        assert_eq!(block_spec(DType::F32).nbytes(7), 28);
    }

    /// `write_scales` must round-trip through each encoding closely enough that a synthesized block
    /// carries the magnitude the caller asked for (the property the parity harness relies on when
    /// it keeps synthetic weights inside a sane band).
    #[test]
    fn write_scales_round_trips_each_encoding() {
        // F16: exact to f16 precision.
        let mut blk = vec![0u8; 144];
        block_spec(DType::Q4K).write_scales(&mut blk, 0.05, 0.10);
        assert_eq!(
            half::f16::from_le_bytes([blk[0], blk[1]]).to_f32(),
            0.05f32.to_f16_round()
        );
        assert_eq!(
            half::f16::from_le_bytes([blk[2], blk[3]]).to_f32(),
            0.10f32.to_f16_round()
        );
        // E8M0: a power of two is exact.
        let mut blk = vec![0u8; 17];
        block_spec(DType::Mxfp4).write_scales(&mut blk, 0.25, 0.0);
        assert_eq!(2f32.powi(blk[0] as i32 - 127), 0.25);
        // UE4M3: 4 distinct sub-scales, all finite and within 2x of the request.
        let mut blk = vec![0u8; 36];
        block_spec(DType::Nvfp4).write_scales(&mut blk, 1.0, 0.0);
        let codes = &blk[0..4];
        assert!(codes.iter().all(|&c| c != 0), "zero sub-scale flushes to 0");
        assert_eq!(
            codes.iter().collect::<std::collections::HashSet<_>>().len(),
            4,
            "sub-scales must be distinct"
        );
        // IQ1_M: the split `d` reassembles, and the low 12 payload bits survive.
        let mut blk = vec![0u8; 56];
        for (i, b) in blk[48..56].iter_mut().enumerate() {
            *b = 0xA5u8 ^ i as u8;
        }
        let payload: Vec<u16> = (0..4)
            .map(|i| u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]) & 0x0fff)
            .collect();
        block_spec(DType::Iq1M).write_scales(&mut blk, 0.03, 0.0);
        let mut bits = 0u16;
        for i in 0..4 {
            let w = u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]);
            assert_eq!(w & 0x0fff, payload[i], "IQ1_M payload nibbles clobbered");
            bits |= (w >> 12) << (4 * i);
        }
        assert_eq!(half::f16::from_bits(bits), half::f16::from_f32(0.03));
    }

    /// Small helper so the f16 round-trip assertions above read as intent, not bit-fiddling.
    trait F16Round {
        fn to_f16_round(self) -> f32;
    }
    impl F16Round for f32 {
        fn to_f16_round(self) -> f32 {
            half::f16::from_f32(self).to_f32()
        }
    }
}
