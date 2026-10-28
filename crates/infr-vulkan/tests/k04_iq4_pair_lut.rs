//! K04 host byte oracle; no device, model, runtime configuration, or new dependencies.
//! Parent-only execution: `cargo test -p infr-vulkan --test k04_iq4_pair_lut`.
//! Also supports standalone `rustc --edition=2021 --test` with a campaign output path.
//! These tests verify the packing math, not GPU execution or performance.
//!
//! The four EXPERT_GRID pair variants are registered in build.rs. STREAMED is added
//! automatically to resident variants, never PAGED. Raw bound-SSBO probes compile separately;
//! IQ4_NL is absent from mechanism_b_resident's whitelist. GPU probes use distinct A/B keys.
//!
//! Parent GPU checks, using the actual compiled entry point and raw buffer readback:
//! - Compare baseline/pair descriptors, push constants and local sizes. For both BM tiles,
//!   compile SSBO, STREAMED and EXPERT_GRID SSBO/STREAMED/PAGED. Run supported subgroup sizes
//!   (at least 32 and 64 where available), checking device limits rather than assuming support.
//! - Recover the signed Bs bytes without a diagnostic binding: K=32, N=64, unit half scales,
//!   32 one-hot QA rows padded to BM (two row tiles for BM=32 are also useful). Output [j,c]
//!   must equal the signed codebook byte for weight element j in column c exactly. Convert
//!   finite integral f32 readback to i8; compare every byte to an independent scalar decoder
//!   and baseline. Sweep all 256 pair indices in both pair positions, all eight p words,
//!   and both 18-byte block alignments. Repeat with K=64 and activate only the odd block.
//! - Then compare entire f32 output bit patterns on identical QA/DA/W bytes, with random
//!   full-range signed QA and finite positive/negative/zero half weight/activation scales.
//!   Separate any NaN/Inf policy cases. Require exact int8 and integer-dot equality; investigate
//!   any finite f32 bit mismatch before considering a separately justified numeric tolerance.
//! - FlashNext AD/UD IQ4_NL Down: K=640, N=2560, 20 K blocks and 40 N tiles; 512 experts.
//!   Test rows/counts 0,1,3,4,5,15,16,17,31,32,33,63,64,65,127,128,129,512; use packed
//!   non-BM-aligned offsets, empty buckets between live experts, and an empty final expert.
//!   Dispatch extra row tiles and leave empty PAGED LUT entries invalid to check early exits.
//!   Use nonzero w_base, STREAMED byte strides, and valid PAGED addresses/slot layouts.
//! - Preserve the existing input contract: QA/DA need a final BM-row padding region because
//!   staging is unconditional and live four-row stripes can read past rowEnd. Plain mode
//!   also stores full BM tiles: allocate padded output and compare valid rows separately.
//!   Place sentinels around actual output allocations and verify packed expert boundaries.
//! - Measure complete dispatches including one-time LUT setup after correctness. Shared bytes:
//!   BM64 4864 -> 5888, BM32 3712 -> 4736 (before implementation-specific allocation granularity).
//!   The pair path adds 256 table writes plus one barrier per surviving workgroup; empty
//!   EXPERT_GRID groups return first. Include short counts and assignment tails in timing.
//! - Campaign-wide model results require the parent's five-model harness. This kernel targets
//!   the two FlashNext IQ4_NL Down paths; no per-model selection is introduced here.

const CODEBOOK: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn shader_codebook_words() -> [u32; 4] {
    let source = include_str!("../shaders/native_gemm_mmq_iq4_nl.comp");
    let values = source
        .split_once("const uint KV_IQ4NL_W[4] = uint[](")
        .expect("production packed codebook")
        .1
        .split_once(");")
        .unwrap()
        .0;
    values
        .split(',')
        .map(|value| {
            u32::from_str_radix(value.trim().trim_end_matches('u').trim_start_matches("0x"), 16)
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn pair_table() -> [u32; 256] {
    let words = shader_codebook_words();
    let byte = |idx: usize| (words[idx >> 2] >> ((idx & 3) * 8)) & 255;
    std::array::from_fn(|pair| byte(pair & 15) | (byte(pair >> 4) << 8))
}

fn scalar_word(qs: &[u8; 16], p: usize) -> u32 {
    (0..4).fold(0, |packed, b| {
        let j = p * 4 + b;
        let idx = (qs[j % 16] >> (if j < 16 { 0 } else { 4 })) & 15;
        packed | ((CODEBOOK[idx as usize] as u8 as u32) << (b * 8))
    })
}

fn pair_word(qs: &[u8; 16], p: usize, table: &[u32; 256]) -> u32 {
    let start = (p & 3) * 4;
    let q = u32::from_le_bytes(qs[start..start + 4].try_into().unwrap());
    let mut pairs = (q >> ((p >> 2) * 4)) & 0x0f0f0f0f;
    pairs |= pairs >> 4;
    table[(pairs & 255) as usize] | (table[((pairs >> 16) & 255) as usize] << 16)
}

#[test]
fn production_codebook_and_all_pairs_match_signed_oracle() {
    let table = pair_table();
    for pair in 0..256 {
        assert_eq!(
            table[pair],
            u16::from_le_bytes([CODEBOOK[pair & 15] as u8, CODEBOOK[pair >> 4] as u8]) as u32,
            "pair {pair:02x}"
        );
    }
}

#[test]
fn all_four_nibble_patterns_match_every_packed_word() {
    let table = pair_table();
    for pattern in 0u32..65536 {
        let qs = std::array::from_fn(|j| {
            let lo = (pattern >> ((j % 4) * 4)) & 15;
            let hi = (pattern >> ((3 - j % 4) * 4)) & 15;
            (lo | (hi << 4)) as u8
        });
        for p in 0..8 {
            assert_eq!(
                pair_word(&qs, p, &table),
                scalar_word(&qs, p),
                "{pattern:04x}/{p}"
            );
        }
    }
}

#[test]
fn eighteen_byte_block_reads_and_signed_dots_match() {
    let table = pair_table();
    // Four blocks exercise halfword offsets 0/2, a nonzero base, and distinct adjacent scales.
    let mut bytes = [0xa5u8; 80];
    for block in 0..4 {
        let bd = 4 + block * 18;
        let qs: [u8; 16] = std::array::from_fn(|j| (j * 37 + block * 71) as u8);
        bytes[bd..bd + 2].copy_from_slice(&(0x3c00u16 + block as u16).to_le_bytes());
        bytes[bd + 2..bd + 18].copy_from_slice(&qs);
    }
    let nw: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let rb = |offset: usize| (nw[offset >> 2] >> ((offset & 3) * 8)) & 255;
    for block in 0..4 {
        let bd = 4 + block * 18;
        let qs: [u8; 16] = bytes[bd + 2..bd + 18].try_into().unwrap();
        for activation in [-128i32, -127, -1, 0, 1, 127] {
            let mut dot = 0;
            let mut reference = 0;
            for p in 0..8 {
                let qb = bd + 2 + (p & 3) * 4;
                let q = rb(qb)
                    | (rb(qb + 1) << 8)
                    | (rb(qb + 2) << 16)
                    | (rb(qb + 3) << 24);
                let mut pairs = (q >> ((p >> 2) * 4)) & 0x0f0f0f0f;
                pairs |= pairs >> 4;
                let packed =
                    table[(pairs & 255) as usize] | (table[((pairs >> 16) & 255) as usize] << 16);
                assert_eq!(packed, scalar_word(&qs, p));
                for b in 0..4 {
                    dot += ((packed >> (8 * b)) as u8 as i8 as i32) * activation;
                    let j = p * 4 + b;
                    let index = (qs[j % 16] >> (if j < 16 { 0 } else { 4 })) & 15;
                    reference += CODEBOOK[index as usize] as i32 * activation;
                }
            }
            assert_eq!(dot, reference);
        }
    }
}
