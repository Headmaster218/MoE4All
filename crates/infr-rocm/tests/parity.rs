//! GPU-gated parity tests for the ROCm backend — the correctness gate for Part A of
//! `docs/rocm-plan.md`. Every test is `#[ignore]`d: they require a real ROCm device
//! (the RX 7900 XTX dev box). Run on the dev box with:
//!
//!   cargo test -p infr-rocm --features rocm -- --include-ignored
//!
//! What is validated:
//!   * `alloc` honours the calloc contract (returns ZEROED VRAM),
//!   * `upload`→`download` is byte-identical,
//!   * a naive `Op::Linear` (dequant→f16 GEMV) matches the CPU reference
//!     (`infr_gguf::dequant::dequant_block` + f32 matmul, i.e. the `infr-cpu`
//!     backend running the same one-op graph) for F16 and a k-quant (Q4_K).
//!
//! The single-op agnostic-`Graph` pattern mirrors `infr-llama/tests/seam_op_parity.rs`.

#![cfg(all(target_os = "linux", feature = "rocm"))]

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, MoeGating, Op};
use infr_core::tensor::TensorDesc;
use infr_core::DType;
use infr_rocm::RocmBackend;

/// Construct the ROCm backend on device 0, or `None` if no ROCm device is present
/// (keeps the ignored tests a no-op on a machine without the hardware).
fn rocm() -> Option<RocmBackend> {
    RocmBackend::new(0).ok()
}

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

/// Deterministic small-magnitude pseudo-random f32 stream (same shape as the seam
/// op-parity generator — keeps values well inside f16 range).
fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
        .collect()
}

fn maxerr(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn maxabs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0, f32::max)
}

// ── alloc / upload / download ────────────────────────────────────────────────

/// `alloc` must return zero-initialized VRAM (the calloc contract every backend obeys).
#[test]
#[ignore = "requires a ROCm GPU"]
fn alloc_returns_zeroed() {
    let Some(be) = rocm() else {
        return;
    };
    let bytes = 4096usize;
    let buf = be.alloc(bytes, BufferUsage::Activations).expect("alloc");
    // Poison the host buffer so an all-zero readback can only come from the device.
    let mut host = vec![0xABu8; bytes];
    be.download(buf.as_ref(), &mut host).expect("download");
    assert!(
        host.iter().all(|&b| b == 0),
        "alloc did not zero-initialize VRAM (calloc contract violated)"
    );
}

/// `upload` then `download` round-trips byte-for-byte.
#[test]
#[ignore = "requires a ROCm GPU"]
fn upload_download_roundtrip() {
    let Some(be) = rocm() else {
        return;
    };
    let data: Vec<u8> = (0..8192u32).map(|i| ((i * 31 + 7) & 0xFF) as u8).collect();
    let buf = be
        .alloc(data.len(), BufferUsage::Activations)
        .expect("alloc");
    be.upload(buf.as_ref(), &data).expect("upload");
    let mut back = vec![0u8; data.len()];
    be.download(buf.as_ref(), &mut back).expect("download");
    assert_eq!(data, back, "upload→download is not byte-identical");
}

// ── Linear (dequant→f16 GEMV) vs the CPU reference ───────────────────────────

/// Run a single-`Op::Linear` graph on `be`: `dst[m, out_f] = x[m, in_f] · w[out_f, in_f]ᵀ`,
/// with `w` uploaded as its raw native `w_dtype` bytes (dequantized on first touch by the
/// backend). Returns the downloaded f32 output.
fn run_linear(
    be: &dyn Backend,
    x: &[f32],
    w_bytes: &[u8],
    w_dtype: DType,
    m: usize,
    in_f: usize,
    out_f: usize,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(m * in_f));
    let wid = g.weight(TensorDesc::new(vec![out_f * in_f], w_dtype));
    let dst = g.output(f32d(m * out_f));
    g.push(Op::Linear {
        x: xid,
        weight: wid,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let wb = be.alloc(w_bytes.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), w_bytes).unwrap();
    let ob = be.alloc(m * out_f * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; m * out_f];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// Run a 2-op `RmsNorm → Linear` graph (m=1 decode) on `be`. Exercises the Slice-32 decode
/// norm-fusion peephole (RmsNorm elided, fused into the int8 GEMV's quant) on the ROCm path;
/// the CPU runs the split ops. Returns the downloaded f32 output.
#[allow(clippy::too_many_arguments)]
fn run_rmsnorm_linear(
    be: &dyn Backend,
    x: &[f32],
    norm_w: &[f32],
    w_bytes: &[u8],
    w_dtype: DType,
    in_f: usize,
    out_f: usize,
) -> Vec<f32> {
    let m = 1usize;
    let mut g = Graph::new();
    let xid = g.input(f32d(m * in_f));
    let nwid = g.weight(TensorDesc::new(vec![in_f], DType::F32));
    let normed = g.internal(f32d(m * in_f));
    let wid = g.weight(TensorDesc::new(vec![out_f * in_f], w_dtype));
    let dst = g.output(f32d(m * out_f));
    g.push(Op::RmsNorm {
        x: xid,
        weight: nwid,
        dst: normed,
        rows: m as u32,
        dim: in_f as u32,
        eps: 1e-6,
    });
    g.push(Op::Linear {
        x: normed,
        weight: wid,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let nwb = be
        .alloc(norm_w.len() * 4, BufferUsage::Weights)
        .expect("nw");
    be.upload(nwb.as_ref(), bytemuck::cast_slice(norm_w))
        .unwrap();
    let wb = be.alloc(w_bytes.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), w_bytes).unwrap();
    let ob = be.alloc(m * out_f * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(nwid, nwb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; m * out_f];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// Slice-32 decode norm-fusion: `RmsNorm → Linear` (Q8_0, m=1) on ROCm (fused) vs the CPU split
/// ops. Within the int8 activation tolerance (only the activation is int8-quantized; Q8_0 weight
/// is near-lossless).
#[test]
#[ignore = "requires a ROCm GPU"]
fn rmsnorm_linear_i8_q80_fused_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (in_f, out_f) = (256usize, 8usize);
    let x = gen(in_f, 5);
    let norm_w = gen(in_f, 9);
    let w_bytes = q80_blocks((out_f * in_f) / 32);
    let c = run_rmsnorm_linear(&cpu, &x, &norm_w, &w_bytes, DType::Q8_0, in_f, out_f);
    let r = run_rmsnorm_linear(&be, &x, &norm_w, &w_bytes, DType::Q8_0, in_f, out_f);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "RmsNorm→Linear-i8 Q8_0 fused max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "fused RmsNorm→Linear reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 1.5e-2,
        "fused RmsNorm→Linear diverges from CPU: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// F16 weight: the CPU reference dequants f16→f32 exactly, ROCm reads f16 as-is, so parity
/// is near bit-exact.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_f16_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (m, in_f, out_f) = (3usize, 256usize, 8usize);
    let x = gen(m * in_f, 4);
    // f16 weight bytes (little-endian half per element).
    let wf32 = gen(out_f * in_f, 7);
    let w_bytes: Vec<u8> = wf32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let c = run_linear(&cpu, &x, &w_bytes, DType::F16, m, in_f, out_f);
    let r = run_linear(&be, &x, &w_bytes, DType::F16, m, in_f, out_f);
    let e = maxerr(&c, &r);
    println!("Linear F16 max_err={e:e} max|ref|={:e}", maxabs(&c));
    assert!(e < 1e-3, "Linear F16 diverges from CPU reference: {e:e}");
}

/// Q4_K weight: exercises the host block-dequant path. The CPU reference decodes the same
/// bytes with `dequant_block` + f32 matmul; ROCm decodes to f16 then GEMVs, so the tolerance
/// absorbs the f16 weight rounding.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_q4k_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // Q4_K super-block = 256 elems / 144 bytes. in_f must be a multiple of 256.
    let (m, in_f, out_f) = (2usize, 256usize, 4usize);
    let blocks = (out_f * in_f) / 256; // one block per output row here
                                       // Build valid Q4_K blocks: patterned bytes, but the two f16 scale slots (d, dmin) at the
                                       // block head overwritten with finite small values so codes span a sane range and never
                                       // decode to Inf/NaN (mirrors infr-gguf's `affine_single_pass_bit_identical_q4k`).
    let mut w_bytes = vec![0u8; blocks * 144];
    for (i, byte) in w_bytes.iter_mut().enumerate() {
        *byte = ((i * 37 + 11) & 0xFF) as u8;
    }
    for blk in 0..blocks {
        let base = blk * 144;
        w_bytes[base..base + 2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
        w_bytes[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
    }
    let x = gen(m * in_f, 5);
    let c = run_linear(&cpu, &x, &w_bytes, DType::Q4K, m, in_f, out_f);
    let r = run_linear(&be, &x, &w_bytes, DType::Q4K, m, in_f, out_f);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "Linear Q4_K max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        e / ref_mag < 2e-2,
        "Linear Q4_K diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
    // Guard against a silently-zero output masquerading as agreement.
    assert!(
        ref_mag > 1e-3,
        "Q4_K reference is all-zero — test is vacuous"
    );
}

// ── Int8-activation dp4a GEMV (Phase 4) vs the CPU f32 reference ──────────────
//
// The default `Op::Linear` path for Q4_K/Q6_K/Q8_0 now quantizes the activation row to int8 and
// integer-dots (`__builtin_amdgcn_sdot4`) against the native weight codes, applying the weight
// block scale AFTER the accumulation. This is a SANCTIONED PRECISION FLIP (int8 activation is
// lossy), so parity is checked against the CPU f32 reference within an int8 tolerance (the dot
// averages the per-element ~1/127 quant error down to well under the bound). Every case uses m=2 to
// exercise the multi-row (`mrow`) grid and carries a vacuity guard (a silently-zero output must not
// masquerade as agreement). Setting `INFR_ROCM_NO_I8` would route the Phase-3 f16 path instead.

/// Build `blocks` valid Q8_0 blocks (34 B = [f16 d][int8 qs[32]]) with a finite small scale and
/// patterned signed codes.
fn q80_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 34];
    for blk in 0..blocks {
        let base = blk * 34;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.02).to_le_bytes());
        for j in 0..32 {
            // signed int8 codes spanning a representative range.
            w[base + 2 + j] = (((blk * 7 + j * 5) % 251) as i32 - 125) as i8 as u8;
        }
    }
    w
}

/// Build `blocks` valid Q5_0 blocks (22 B = [f16 d][u8 qh[4]][u8 qs[16]]) with a finite small scale
/// and patterned 5-bit codes (4 low bits in `qs` + the 5th bit in the 32-bit `qh` bitfield).
fn q50_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 22];
    for blk in 0..blocks {
        let base = blk * 22;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        // qh: 4 bytes → one high bit per element; a patterned bitfield exercises both nibble halves.
        let qh = (blk as u32).wrapping_mul(2654435761);
        w[base + 2..base + 6].copy_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            w[base + 6 + j] = ((blk * 7 + j * 11) & 0xFF) as u8;
        }
    }
    w
}

/// Build `blocks` valid Q4_0 blocks (18 B = [f16 d][u8 qs[16]]) with a finite small scale and
/// patterned 4-bit codes (low nibbles are elements 0..15, high nibbles 16..31).
fn q40_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 18];
    for blk in 0..blocks {
        let base = blk * 18;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        for j in 0..16 {
            w[base + 2 + j] = ((blk * 7 + j * 11) & 0xFF) as u8;
        }
    }
    w
}

/// Build `blocks` valid Q4_1 blocks (20 B = [f16 d][f16 m][u8 qs[16]]). The AFFINE minimum `m`
/// ALTERNATES sign block-to-block: a constant `m` would still pass a kernel that read `m` from the
/// wrong block, and a zero `m` would pass one that dropped the min term entirely.
fn q41_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 20];
    for blk in 0..blocks {
        let base = blk * 20;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        let m = if blk % 2 == 0 { -0.11 } else { 0.07 };
        w[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(m).to_le_bytes());
        for j in 0..16 {
            w[base + 4 + j] = ((blk * 7 + j * 11) & 0xFF) as u8;
        }
    }
    w
}

/// Build `blocks` valid Q5_1 blocks (24 B = [f16 d][f16 m][u8 qh[4]][u8 qs[16]]) — `q41_blocks`'
/// alternating affine minimum plus `q50_blocks`' patterned `qh` bitfield carrying each code's 5th
/// bit, so both nibble halves AND both states of every high bit are exercised.
fn q51_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 24];
    for blk in 0..blocks {
        let base = blk * 24;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        let m = if blk % 2 == 0 { -0.11 } else { 0.07 };
        w[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(m).to_le_bytes());
        let qh = (blk as u32).wrapping_mul(2654435761);
        w[base + 4..base + 8].copy_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            w[base + 8 + j] = ((blk * 7 + j * 11) & 0xFF) as u8;
        }
    }
    w
}

/// Build `blocks` valid IQ4_NL blocks (18 B = [f16 d][u8 qs[16]] — Q4_0's block shape) with a
/// finite small scale and patterned 4-bit CODEBOOK INDICES (low nibbles are elements 0..15, high
/// nibbles 16..31). Because 11 is coprime with 16, the pattern walks every table entry, so a kernel
/// that indexed the codebook the wrong way (or read it byte-swapped) cannot pass by luck.
fn iq4nl_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 18];
    for blk in 0..blocks {
        let base = blk * 18;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        for j in 0..16 {
            w[base + 2 + j] = ((blk * 7 + j * 11) & 0xFF) as u8;
        }
    }
    w
}

/// Build `blocks` valid IQ4_XS blocks (136 B = [f16 d][u16 scales_h][u8 scales_l[4]][u8 qs[128]]).
/// The eight 6-bit sub-block scales are built EXPLICITLY (low 4 bits into `scales_l`, high 2 into
/// `scales_h`) and walk 0..63, so `ls − 32` takes BOTH signs across the sub-blocks — a kernel that
/// dropped the `−32` bias, or read `hi` from the wrong 2-bit field, lands at O(1) relative. The
/// codebook indices use the same coprime pattern as `iq4nl_blocks`.
fn iq4xs_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 136];
    for blk in 0..blocks {
        let base = blk * 136;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        let (mut scales_h, mut scales_l) = (0u16, [0u8; 4]);
        for ib in 0..8usize {
            let ls = ((blk * 5 + ib * 9) % 64) as u32;
            scales_l[ib / 2] |= ((ls & 0xF) as u8) << (4 * (ib % 2));
            scales_h |= (((ls >> 4) & 3) as u16) << (2 * ib);
        }
        w[base + 2..base + 4].copy_from_slice(&scales_h.to_le_bytes());
        w[base + 4..base + 8].copy_from_slice(&scales_l);
        for j in 0..128 {
            w[base + 8 + j] = ((blk * 7 + j * 11) & 0xFF) as u8;
        }
    }
    w
}

// ── R5 grid quants: IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S ───────────────
//
// These five take their blocks from the SHARED spec-driven builder
// (`infr_testkit::synth_weight`, which reads `infr_core::decode_spec`) rather than a bespoke
// per-format writer like the ones above, for two reasons specific to a grid quant:
//
//   * There is no "invalid" payload to avoid. Every field except `d` is either a grid index (the
//     kernels mask it to the table's exact size — 8/9/10 bits over a 256/512/1024-entry grid), a
//     sign pattern, or a 4-bit scale magnitude, so pseudo-random bytes are already a legal block
//     AND they walk the whole codebook — which is exactly the property the bespoke builders above
//     had to construct by hand with coprime strides.
//   * The `d` magnitudes are the ones the shared harness converged on per format
//     (`infr_testkit::synth_scales`), so these cases and the `shared_decode_parity.rs` sweep are
//     measuring the same weights rather than two independently tuned sets.
//
// One super-block is 256 elements for all five, so `blocks` here means super-blocks.
fn iq2xxs_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Iq2Xxs, blocks * 256, 0x5101)
}
fn iq2xs_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Iq2Xs, blocks * 256, 0x5102)
}
fn iq2s_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Iq2S, blocks * 256, 0x5103)
}
fn iq3xxs_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Iq3Xxs, blocks * 256, 0x5104)
}
fn iq3s_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Iq3S, blocks * 256, 0x5105)
}

// ── R6 IQ1 + ternary quants: IQ1_S / IQ1_M / TQ1_0 / TQ2_0 / Q2_0 ────────────
//
// Same shared spec-driven builder, for the same two reasons, plus one more that is specific to
// these five: three of them (IQ1_M's split `d`, and the ternary formats' single `d` at a non-zero
// offset) have scale slots a bespoke writer would have to re-spell, and `BlockSpec::write_scales`
// already knows all of them from `infr_core::decode_spec` — IQ1_M's `Iq1mSplitF16` in particular
// writes the f16 nibbles into the four scale words WITHOUT disturbing their low 12 bits, so the
// 3-bit `dl` sub-scales stay pseudo-random payload and the block exercises distinct sub-scales.
//
// `blocks` means super-blocks (256 elements) for the first four; Q2_0's block is 64 elements.
fn iq1s_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Iq1S, blocks * 256, 0x6101)
}
fn iq1m_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Iq1M, blocks * 256, 0x6102)
}
fn tq10_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Tq1_0, blocks * 256, 0x6103)
}
fn tq20_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Tq2_0, blocks * 256, 0x6104)
}
fn q20_blocks(blocks: usize) -> Vec<u8> {
    infr_testkit::synth_weight(DType::Q2_0, blocks * 64, 0x6105)
}

// ── R7 fp4 microscaling quants: MXFP4 / NVFP4 ───────────────────────────────
//
// These two start from the shared spec-driven builder (LCG payload, `write_scales` for a valid
// block) and then OVERWRITE the scale bytes with a varying, still-valid encoding — the one place
// in the suite where a bespoke step earns its keep, because `synth_weight` writes ONE magnitude
// into every scale slot of every block. That is the right default for a decode sweep, but it makes
// the two faults R7 can actually have invisible:
//
//   * a constant E8M0/UE4M3 byte never exercises the encodings' CASE STRUCTURE — `e8m0_half`'s
//     subnormal branch for `e ∈ {0,1}` (where the naive `(e−1) << 23` yields ±inf, not 2^(e−128)),
//     and `ue4m3`'s `e == 0` subnormal branch and its 0x00/0x7F zero holes;
//   * with all four of a NVFP4 block's sub-scales EQUAL, a kernel that broadcast one scale across
//     the whole 32-element tile — i.e. dropped the `s0`/`s1` split that is the format's whole
//     structural difference from MXFP4 — would still match to the last bit.
//
// So the scales here are written directly in their wire encoding rather than through a magnitude:
// the point is to vary the ENCODED byte, and `write_scales` takes an f32 and rounds it.
//
// `blocks` means whole native blocks: 32 elements for MXFP4, 64 for NVFP4.

/// MXFP4 blocks with a per-block E8M0 exponent cycling `{126,127,128,129}` → `d ∈ {¼,½,1,2}`, plus
/// `e = 1` on every 11th block to reach the SUBNORMAL branch. (`e = 1` gives `d = 2^-127`, whose
/// products round to zero — so what that block pins is not a value but that the branch exists at
/// all: the common-case formula would make it ±inf and poison the whole dot with a NaN.)
fn mxfp4_blocks(blocks: usize) -> Vec<u8> {
    let mut w = infr_testkit::synth_weight(DType::Mxfp4, blocks * 32, 0x7101);
    for b in 0..blocks {
        w[b * 17] = if b % 11 == 10 { 1 } else { 126 + (b % 4) as u8 };
    }
    w
}

/// NVFP4 blocks whose FOUR UE4M3 sub-block scales are all different from each other and vary block
/// to block: code `(e << 3) | m` with `e ∈ {6,7,8}` and a rotating mantissa, i.e.
/// `d = 0.5 · 2^(e−7) · (1 + m/8) ∈ [0.125, 0.9375]`. Sub-block 3 of every 7th block takes the
/// reserved code `0x7F`, which the oracle decodes to 0.0 and a kernel that skipped the hole decodes
/// to ~0.94 — a whole sub-block wrong.
fn nvfp4_blocks(blocks: usize) -> Vec<u8> {
    let mut w = infr_testkit::synth_weight(DType::Nvfp4, blocks * 64, 0x7102);
    for b in 0..blocks {
        for s in 0..4usize {
            let e = 6 + ((b + s) % 3) as u8;
            let m = ((b * 3 + s * 5) % 8) as u8;
            w[b * 36 + s] = if s == 3 && b % 7 == 6 {
                0x7F
            } else {
                (e << 3) | m
            };
        }
    }
    w
}

/// Build `blocks` valid Q6_K blocks (210 B = [ql 128][qh 64][int8 scales 16][f16 d]) with a finite
/// small `d`, a benign in-range int8 sub-block scale, and patterned ql/qh.
fn q6k_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 210];
    for (i, byte) in w.iter_mut().enumerate() {
        *byte = ((i * 37 + 11) & 0xFF) as u8;
    }
    for blk in 0..blocks {
        let base = blk * 210;
        // 16 int8 sub-block scales — a small positive constant keeps decode in a sane range.
        for s in 0..16 {
            w[base + 192 + s] = 8i8 as u8;
        }
        // f16 d in the last 2 bytes.
        w[base + 208..base + 210].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes());
    }
    w
}

/// Build `blocks` valid Q2_K blocks (84 B = [u8 scales[16]][u8 qs[64]][f16 d][f16 dmin]). Every
/// `scales` byte is legal (4-bit scale | 4-bit min), so the patterned fill sweeps sc/min across the
/// 16 sub-blocks AND — because 37 is odd — every 2-bit code position through all four values.
fn q2k_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 84];
    for (i, byte) in w.iter_mut().enumerate() {
        *byte = ((i * 37 + 11) & 0xFF) as u8;
    }
    for blk in 0..blocks {
        let base = blk * 84;
        w[base + 80..base + 82].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
        w[base + 82..base + 84].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
    }
    w
}

/// Build `blocks` valid Q3_K blocks (110 B = [u8 hmask[32]][u8 qs[64]][u8 scales[12]][f16 d]). All
/// 12 scale bytes are legal (they pack 16 × 6-bit values through the kmask1/kmask2 shuffle), and the
/// patterned `hmask` puts both states of every high bit in play — the bit whose polarity a wrong
/// port flips.
fn q3k_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 110];
    for (i, byte) in w.iter_mut().enumerate() {
        *byte = ((i * 37 + 11) & 0xFF) as u8;
    }
    for blk in 0..blocks {
        let base = blk * 110;
        w[base + 108..base + 110].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes());
    }
    w
}

/// Build `blocks` valid Q4_K blocks (144 B) — same construction as `linear_q4k_matches_cpu`.
fn q4k_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 144];
    for (i, byte) in w.iter_mut().enumerate() {
        *byte = ((i * 37 + 11) & 0xFF) as u8;
    }
    for blk in 0..blocks {
        let base = blk * 144;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
        w[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
    }
    w
}

/// Build `blocks` valid Q5_K blocks (176 B = [f16 d][f16 dmin][u8 scales[12]][u8 qh[32]][u8
/// qs[128]]) — the Q4_K construction plus the 32-byte `qh` plane that carries each code's 5th bit,
/// so the patterned fill exercises both nibble halves AND both states of every high bit.
fn q5k_blocks(blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 176];
    for (i, byte) in w.iter_mut().enumerate() {
        *byte = ((i * 37 + 11) & 0xFF) as u8;
    }
    for blk in 0..blocks {
        let base = blk * 176;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
        w[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
    }
    w
}

/// Shared int8-GEMV parity check: ROCm int8 `Linear` vs the CPU f32 reference, m=2, within `tol`.
fn check_i8_linear(w_bytes: &[u8], dt: DType, in_f: usize, out_f: usize, tol: f32, label: &str) {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let m = 2usize;
    let x = gen(m * in_f, 5);
    let c = run_linear(&cpu, &x, w_bytes, dt, m, in_f, out_f);
    let r = run_linear(&be, &x, w_bytes, dt, m, in_f, out_f);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "Linear-i8 {label} max_err={e:e} max|ref|={ref_mag:e} rel={:e} (tol={tol:e})",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "{label} int8 reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < tol,
        "{label} int8 GEMV diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

// Shapes: in_f=512 (2 super-blocks per output row → exercises the per-row weight offset AND the
// multi-super accumulation, which a single-super in_f=256 case would NOT catch), out_f=8 (distinct
// per-row weights → catches a kernel that drops the output-row offset and reads row 0 for every o).
const I8_IN_F: usize = 512;
const I8_OUT_F: usize = 8;

/// Q8_0 int8 GEMV: weight is near-lossless (only the activation is int8), so the tolerance is tight.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q80_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 32;
    check_i8_linear(
        &q80_blocks(blocks),
        DType::Q8_0,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "Q8_0",
    );
}

/// Q5_0 int8 GEMV: 5-bit weight (single per-32-block scale + `−16` offset) + int8 activation.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q50_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 32;
    check_i8_linear(
        &q50_blocks(blocks),
        DType::Q5_0,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "Q5_0",
    );
}

/// Q4_0 int8 GEMV (R3): 4-bit weight, single per-32-block scale + the constant `−8` offset via the
/// ones-dot + int8 activation.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q40_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 32;
    check_i8_linear(
        &q40_blocks(blocks),
        DType::Q4_0,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "Q4_0",
    );
}

/// Q4_1 int8 GEMV (R3): 4-bit weight with an AFFINE per-block minimum — the ones-dot is weighted by
/// each block's own `m` instead of a constant multiple of `d`, which is the one structural way this
/// tier differs from Q4_0/Q5_0.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q41_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 32;
    check_i8_linear(
        &q41_blocks(blocks),
        DType::Q4_1,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "Q4_1",
    );
}

/// Q5_1 int8 GEMV (R3): Q4_1's affine `(d, m)` header plus Q5_0's `qh` 5th code bit.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q51_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 32;
    check_i8_linear(
        &q51_blocks(blocks),
        DType::Q5_1,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "Q5_1",
    );
}

/// IQ4_NL int8 GEMV (R4): the first CODEBOOK format on this tier — the 4-bit field indexes the
/// signed 16-entry `kvalues_iq4nl` table, so the table value itself is the dp4a operand and there is
/// NO ones-dot/min term. A kernel that kept an affine `code − 8` (or fed the raw index to dp4a)
/// lands at O(1) relative here.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq4nl_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 32;
    check_i8_linear(
        &iq4nl_blocks(blocks),
        DType::Iq4Nl,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ4_NL",
    );
}

/// IQ4_XS int8 GEMV (R4): the same codebook, but scaled by a 6-bit per-32-element sub-block `ls`
/// biased by −32 on top of the super-block `d`. One 32-elem activation block is exactly one
/// sub-block, so this keeps Q4_K's one-scale-per-block loop shape (not Q6_K's two-halves one).
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq4xs_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq4xs_blocks(blocks),
        DType::Iq4Xs,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ4_XS",
    );
}

// ── R5 grid-quant int8 GEMV ──────────────────────────────────────────────────
// All five go through ONE shared per-32-block decoder (`wdec_*`) and ONE macro body, so what these
// five cases separate is precisely the five decoders: a swapped grid table, a sign field read from
// the wrong place, or a scale taken per-32 where it is per-16 shows up in exactly one of them.
// Like R4's codebook formats there is NO ones-dot term — the grid byte is already signed — so a
// kernel that grew one (by copying an affine format's body) lands at O(1) relative.

/// IQ2_XXS int8 GEMV (R5): the first GRID format — an 8-bit index into a 256-entry table of packed
/// 8-byte signed vectors, a 7-bit `ksigns` index per group of 8, and ONE scale per 32-element
/// block taken from the top nibble of the group's `aux1` word.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq2xxs_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq2xxs_blocks(blocks),
        DType::Iq2Xxs,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ2_XXS",
    );
}

/// IQ2_XS int8 GEMV (R5): a 9-bit grid index packed with its 7-bit sign index in one u16, and TWO
/// scales per 32-element block (the two nibbles of `scales[ib32]`, one per 16 elements) — the case
/// that pins the split accumulation. A kernel that applied one scale to all 32 lands at O(1).
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq2xs_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq2xs_blocks(blocks),
        DType::Iq2Xs,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ2_XS",
    );
}

/// IQ2_S int8 GEMV (R5): a 10-bit grid index (8 bits from `qs`, 2 more from `qh` at shift `8−2l`)
/// and RAW sign bytes — no `ksigns` indirection at all, unlike IQ2_XXS/IQ2_XS. Same two-scale
/// split as IQ2_XS.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq2s_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq2s_blocks(blocks),
        DType::Iq2S,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ2_S",
    );
}

/// IQ3_XXS int8 GEMV (R5): the IQ3 grids hold FOUR bytes per entry, so a group of 8 elements needs
/// TWO grid entries sharing one 8-bit sign pattern (elements 4..7 take sign bits 4..7). One scale
/// per 32, from `aux32`'s top nibble, with the `·0.5` IQ3 factor rather than IQ2's `·0.25`.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq3xxs_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq3xxs_blocks(blocks),
        DType::Iq3Xxs,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ3_XXS",
    );
}

/// IQ3_S int8 GEMV (R5): two 9-bit grid indices per group (the 9th bit from `qh` at shift `8−2l`
/// for the first entry and `7−2l` for the second — an off-by-one there is invisible to a coarse
/// tolerance but not to `embed_gather`), raw sign bytes, and the `d·(1+2·ls)` scale form that no
/// other format in this family uses.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq3s_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq3s_blocks(blocks),
        DType::Iq3S,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ3_S",
    );
}

// ── R6 IQ1 + ternary int8 GEMV ───────────────────────────────────────────────
// All five ride the same `GEN_LINEAR_I8_WDEC` body R5's grid quants do, so what these cases
// separate is precisely the five `wdec_*` decoders. Two R6-specific failure modes land at O(1)
// here and nowhere else in the GEMV tier: an IQ1 kernel that dropped the delta (or folded it with
// the wrong ×8 scale) and a ternary kernel that forgot the `−1` offset — the latter turns a
// zero-centred weight into an all-positive one, which no tolerance absorbs.

/// IQ1_S int8 GEMV (R6): 11-bit index into the 2048-entry IQ1 grid, ONE scale + ONE delta sign per
/// 32-element block. The delta is folded into the code as `8·gv ± 1` with the scale scaled by
/// 0.125, so a kernel that kept the addend outside (and therefore needed a ones-dot it does not
/// have) is off by `dl·delta·Σx` per block.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq1s_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq1s_blocks(blocks),
        DType::Iq1S,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ1_S",
    );
}

/// IQ1_M int8 GEMV (R6): the same grid, but the scale splits per 16 elements (`dl1`/`dl2`, the
/// `ws0`/`ws1` case) AND the delta sign varies per GROUP OF 8 — the finest-grained side-channel of
/// any covered format, and the reason the ×8 fold matters rather than being a convenience. Also
/// the only format whose `d` has no field of its own (split across the four scale words' nibbles).
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_iq1m_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &iq1m_blocks(blocks),
        DType::Iq1M,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "IQ1_M",
    );
}

/// TQ1_0 int8 GEMV (R6): 5 base-3 digits per byte over a THREE-SEGMENT element order (qs[0..32]×5,
/// qs[32..48]×5, qh[0..4]×4). The wrapping `byte·3ⁿ` product is what makes the digit extraction
/// work, so a kernel that widened before multiplying decodes a different ternary level entirely.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_tq10_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &tq10_blocks(blocks),
        DType::Tq1_0,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "TQ1_0",
    );
}

/// TQ2_0 int8 GEMV (R6): 2 bits per element over two 32-byte chunks × 4 shifts × 32 bytes, so one
/// 32-element activation block is exactly one (chunk, shift) pair — 32 consecutive bytes.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_tq20_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &tq20_blocks(blocks),
        DType::Tq2_0,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "TQ2_0",
    );
}

/// Q2_0 int8 GEMV (R6): infr's OWN format and the only 64-ELEMENT block in the covered set, so one
/// activation 32-block is HALF a weight block — the case that pins `wdec_q20`'s `blk>>1` super-block
/// index and its `(blk & 1) * 8` byte half against every other format's `blk>>3`.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q20_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 64;
    check_i8_linear(
        &q20_blocks(blocks),
        DType::Q2_0,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "Q2_0",
    );
}

// ── R7 fp4 int8 GEMV ─────────────────────────────────────────────────────────
// Both ride the same `GEN_LINEAR_I8_WDEC` body, so these separate the two `wdec_*` decoders — and
// specifically the SCALE decode, which is the only genuinely new thing in R7. A mis-decoded E8M0
// or UE4M3 is a POWER-OF-TWO error: the codes are still right, the result is 2× or ½× (or 2^k×)
// the reference for every element of a block. That is O(1) relative here, far outside any
// tolerance — which is exactly why the tolerance is left at the family's usual 1.5e-2.

/// MXFP4 int8 GEMV (R7): the E8M0 shared exponent. `e8m0_half` has a two-case form — the exponent
/// field is `e − 1` for `e ≥ 2` but a SUBNORMAL bit pattern for `e ∈ {0,1}` — and a kernel that
/// implemented only the common case flushes the two smallest scales to zero, which this shape
/// (512-deep, so many blocks) reaches.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_mxfp4_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 32;
    check_i8_linear(
        &mxfp4_blocks(blocks),
        DType::Mxfp4,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "MXFP4",
    );
}

/// NVFP4 int8 GEMV (R7): a 64-element block whose FOUR UE4M3 sub-block scales mean each 32-element
/// activation block carries two DIFFERENT weight scales — the `ws0`/`ws1` split, exercised here on
/// top of Q2_0's 64-element stride. A kernel that broadcast one scale over the whole 32-block (the
/// shape every per-32-scale format on this seam uses) is wrong on half of every block.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_nvfp4_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 64;
    check_i8_linear(
        &nvfp4_blocks(blocks),
        DType::Nvfp4,
        I8_IN_F,
        I8_OUT_F,
        1.5e-2,
        "NVFP4",
    );
}

/// Q2_K int8 GEMV (R2): 2-bit weight, 4-bit sub-block scale + 4-bit min per 16 elements (so a
/// 32-elem activation block spans TWO scale sub-blocks) + int8 activation.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q2k_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &q2k_blocks(blocks),
        DType::Q2K,
        I8_IN_F,
        I8_OUT_F,
        3e-2,
        "Q2_K",
    );
}

/// Q3_K int8 GEMV (R2): 3-bit weight (2 low bits + the `hmask` high bit), packed 6-bit sub-block
/// scale per 16 elements + int8 activation.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q3k_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &q3k_blocks(blocks),
        DType::Q3K,
        I8_IN_F,
        I8_OUT_F,
        3e-2,
        "Q3_K",
    );
}

/// Q4_K int8 GEMV: 4-bit weight + int8 activation; tolerance absorbs both.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q4k_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &q4k_blocks(blocks),
        DType::Q4K,
        I8_IN_F,
        I8_OUT_F,
        3e-2,
        "Q4_K",
    );
}

/// Q5_K int8 GEMV (R1): 5-bit weight (Q4_K's 6-bit sub-block scale/min + a `qh`-plane 5th bit)
/// + int8 activation; same tolerance as the other k-quants.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q5k_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &q5k_blocks(blocks),
        DType::Q5K,
        I8_IN_F,
        I8_OUT_F,
        3e-2,
        "Q5_K",
    );
}

/// Q6_K int8 GEMV: 6-bit weight + int8 activation.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_i8_q6k_matches_cpu() {
    let blocks = (I8_OUT_F * I8_IN_F) / 256;
    check_i8_linear(
        &q6k_blocks(blocks),
        DType::Q6K,
        I8_IN_F,
        I8_OUT_F,
        3e-2,
        "Q6_K",
    );
}

// ── Prefill GEMM (m > 1) vs the CPU f32 reference ────────────────────────────
//
// For `m > 1` (prefill) the DEFAULT `Op::Linear` path for Q4_K/Q6_K/Q8_0/Q5_0 routes to the RDNA3
// wave32 int8 matrix core (`wmma_i8_*`, one 16×16 output tile per wave); `INFR_ROCM_NO_WMMA` drops it
// to the Phase-4 dp4a GEMV, and `INFR_ROCM_BLAS=1` opts into the Slice-26 rocBLAS f16 GEMM — all land
// within the same int8 tolerance vs the CPU f32 reference (f16 is strictly MORE accurate than the
// int8 codes, so BLAS clears the int8 bound comfortably). Shapes deliberately break the 16-tile
// alignment on BOTH edges: `m` is NOT a multiple of 16 (row-edge masking) and `out_f` is NOT a
// multiple of 16 (column-edge masking + guarded weight decode). `in_f = 512` gives 2 super-blocks /
// output row (Q4_K/Q6_K) — exercises the per-super offset AND the multi-block scale-after
// accumulation. Every case carries a vacuity guard.

/// Shared prefill parity check: ROCm `Linear` (m>1 → int8 WMMA by default, or the rocBLAS f16 GEMM
/// under `INFR_ROCM_BLAS=1`) vs the CPU f32 reference. `m = 18` (16 + 2 → two row tiles, last
/// partially masked); `out_f = 40` (32 + 8 → three column tiles, last partially masked).
fn check_wmma_linear(
    w_bytes_for: impl Fn(usize) -> Vec<u8>,
    dt: DType,
    qpb: usize,
    tol: f32,
    label: &str,
) {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (m, in_f, out_f) = (18usize, 512usize, 40usize);
    let blocks = (out_f * in_f) / qpb;
    let w_bytes = w_bytes_for(blocks);
    let x = gen(m * in_f, 5);
    let c = run_linear(&cpu, &x, &w_bytes, dt, m, in_f, out_f);
    let r = run_linear(&be, &x, &w_bytes, dt, m, in_f, out_f);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "WMMA-i8 {label} m={m} out_f={out_f} max_err={e:e} max|ref|={ref_mag:e} rel={:e} (tol={tol:e})",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "{label} WMMA reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < tol,
        "{label} WMMA prefill GEMM diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q8_0 WMMA prefill GEMM (near-lossless weight → tight int8-activation tolerance).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q80_matches_cpu() {
    check_wmma_linear(q80_blocks, DType::Q8_0, 32, 1.5e-2, "Q8_0");
}

/// Q5_0 WMMA prefill GEMM (5-bit weight, single per-32-block scale + `−16` offset via ones-dot).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q50_matches_cpu() {
    check_wmma_linear(q50_blocks, DType::Q5_0, 32, 1.5e-2, "Q5_0");
}

/// Q4_0 WMMA prefill GEMM (R3): 4-bit weight, single per-32-block scale + the `−8` offset via the
/// ones-dot (the `GEN_WMMA_R32` shared body at `HASMIN=0, FIVEBIT=0`).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q40_matches_cpu() {
    check_wmma_linear(q40_blocks, DType::Q4_0, 32, 1.5e-2, "Q4_0");
}

/// Q4_1 WMMA prefill GEMM (R3): the affine variant — `HASMIN=1`, so the ones-dot carries the
/// block's own `m` (a kernel that reused Q4_0's `d·(−8)` lands at O(1) relative here).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q41_matches_cpu() {
    check_wmma_linear(q41_blocks, DType::Q4_1, 32, 1.5e-2, "Q4_1");
}

/// Q5_1 WMMA prefill GEMM (R3): affine min + the `qh` 5th code bit (`HASMIN=1, FIVEBIT=1`), which
/// also shifts `qs` to +8 — a stale +4 offset would read the `qh` word as codes.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q51_matches_cpu() {
    check_wmma_linear(q51_blocks, DType::Q5_1, 32, 1.5e-2, "Q5_1");
}

/// IQ4_NL WMMA prefill GEMM (R4): the codebook body `GEN_WMMA_IQ4` at `XS=0` — Q8_0's shape (no
/// ones-dot at all) with a nibble→`kv_iq4nl` gather in front of the B fragment.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq4nl_matches_cpu() {
    check_wmma_linear(iq4nl_blocks, DType::Iq4Nl, 32, 1.5e-2, "IQ4_NL");
}

/// IQ4_XS WMMA prefill GEMM (R4): `GEN_WMMA_IQ4` at `XS=1` — the same body with the super-block
/// address and the 6-bit `ls − 32` sub-block scale.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq4xs_matches_cpu() {
    check_wmma_linear(iq4xs_blocks, DType::Iq4Xs, 256, 1.5e-2, "IQ4_XS");
}

// R5 grid quants on the WMMA prefill tier. All five share ONE `GEN_WMMA_IQG` body over the same
// `wdec_*` decoders the int8 GEMV uses, so what these add over the GEMV cases is the TILING: the
// per-16 K-tile split (each half of a 32-block scaled independently, which is why the body carries
// `ws0`/`ws1` instead of Q8_0's single `wsc`) and the m/out_f edge masking.

/// IQ2_XXS WMMA prefill GEMM (R5).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq2xxs_matches_cpu() {
    check_wmma_linear(iq2xxs_blocks, DType::Iq2Xxs, 256, 1.5e-2, "IQ2_XXS");
}

/// IQ2_XS WMMA prefill GEMM (R5) — the two-scales-per-32-block case, so this is the one that pins
/// `ws0`/`ws1` reaching the right K-tile.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq2xs_matches_cpu() {
    check_wmma_linear(iq2xs_blocks, DType::Iq2Xs, 256, 1.5e-2, "IQ2_XS");
}

/// IQ2_S WMMA prefill GEMM (R5) — 10-bit grid index + raw sign bytes, two scales per block.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq2s_matches_cpu() {
    check_wmma_linear(iq2s_blocks, DType::Iq2S, 256, 1.5e-2, "IQ2_S");
}

/// IQ3_XXS WMMA prefill GEMM (R5) — two 4-byte grid entries per group of 8.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq3xxs_matches_cpu() {
    check_wmma_linear(iq3xxs_blocks, DType::Iq3Xxs, 256, 1.5e-2, "IQ3_XXS");
}

/// IQ3_S WMMA prefill GEMM (R5) — the `qh` 9th index bit and the `d·(1+2·ls)` scale form.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq3s_matches_cpu() {
    check_wmma_linear(iq3s_blocks, DType::Iq3S, 256, 1.5e-2, "IQ3_S");
}

// R6 IQ1 + ternary quants on the WMMA prefill tier. Same `GEN_WMMA_WDEC` body over the same
// `wdec_*` decoders the int8 GEMV uses, so what these add over the GEMV cases is the TILING —
// the per-16 K-tile split (`ws0`/`ws1`, which IQ1_M actually exercises) and the m/out_f edge
// masking. They also pin that every R6 code fits the SIGNED int8 WMMA operand: |code| ≤ 9 for IQ1
// (`8·gv ± 1`) and ≤ 2 for ternary, against R5's widest of 62.

/// IQ1_S WMMA prefill GEMM (R6).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq1s_matches_cpu() {
    check_wmma_linear(iq1s_blocks, DType::Iq1S, 256, 1.5e-2, "IQ1_S");
}

/// IQ1_M WMMA prefill GEMM (R6) — the two-scales-per-32-block case of this family, so this is the
/// one that pins `ws0`/`ws1` reaching the right K-tile with a per-8 delta sign underneath.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_iq1m_matches_cpu() {
    check_wmma_linear(iq1m_blocks, DType::Iq1M, 256, 1.5e-2, "IQ1_M");
}

/// TQ1_0 WMMA prefill GEMM (R6) — the base-3 digit walk across all three element segments.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_tq10_matches_cpu() {
    check_wmma_linear(tq10_blocks, DType::Tq1_0, 256, 1.5e-2, "TQ1_0");
}

/// TQ2_0 WMMA prefill GEMM (R6).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_tq20_matches_cpu() {
    check_wmma_linear(tq20_blocks, DType::Tq2_0, 256, 1.5e-2, "TQ2_0");
}

/// Q2_0 WMMA prefill GEMM (R6) — the 64-element block against a 32-element K-tile pair, i.e. the
/// one covered format where a weight block spans exactly the two K-tiles of ONE WMMA step.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q20_matches_cpu() {
    check_wmma_linear(q20_blocks, DType::Q2_0, 64, 1.5e-2, "Q2_0");
}

// R7 fp4 quants on the WMMA prefill tier — same `GEN_WMMA_WDEC` body over the same `wdec_*`
// decoders, so what these add over the GEMV cases is the tiling, and for NVFP4 in particular that
// the per-16 `ws0`/`ws1` really do reach the two K-tiles of a step in the right order. |code| ≤ 12
// for both (the widest E2M1 level), comfortably inside the signed int8 WMMA operand.

/// MXFP4 WMMA prefill GEMM (R7).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_mxfp4_matches_cpu() {
    check_wmma_linear(mxfp4_blocks, DType::Mxfp4, 32, 1.5e-2, "MXFP4");
}

/// NVFP4 WMMA prefill GEMM (R7) — the only covered format that is BOTH a 64-element block and
/// split-scale, i.e. the one case where the two K-tiles of one WMMA step come from the same weight
/// block but must not share a scale.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_nvfp4_matches_cpu() {
    check_wmma_linear(nvfp4_blocks, DType::Nvfp4, 64, 1.5e-2, "NVFP4");
}

/// Q2_K WMMA prefill GEMM (R2): 2-bit weight, per-16 sub-block 4-bit scale + 4-bit min (1 K-tile
/// per scale-block, the Q6_K walk).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q2k_matches_cpu() {
    check_wmma_linear(q2k_blocks, DType::Q2K, 256, 3e-2, "Q2_K");
}

/// Q3_K WMMA prefill GEMM (R2): 3-bit weight (2 low bits + `hmask`), per-16 sub-block packed 6-bit
/// scale with the folded −4 code offset in the min term.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q3k_matches_cpu() {
    check_wmma_linear(q3k_blocks, DType::Q3K, 256, 3e-2, "Q3_K");
}

/// Q4_K WMMA prefill GEMM (4-bit weight + int8 activation, per-32 sub-block scale + min).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q4k_matches_cpu() {
    check_wmma_linear(q4k_blocks, DType::Q4K, 256, 3e-2, "Q4_K");
}

/// Q5_K WMMA prefill GEMM (R1): Q4_K's per-32 sub-block scale + min, plus the `qh` 5th code bit.
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q5k_matches_cpu() {
    check_wmma_linear(q5k_blocks, DType::Q5K, 256, 3e-2, "Q5_K");
}

/// Q6_K WMMA prefill GEMM (6-bit weight, per-16 sub-block int8 scale — 1 K-tile per scale-block).
#[test]
#[ignore = "requires a ROCm GPU"]
fn wmma_q6k_matches_cpu() {
    check_wmma_linear(q6k_blocks, DType::Q6K, 256, 3e-2, "Q6_K");
}

// ── EmbedGather (gather + dequant embedding rows, ×scale) vs CPU ─────────────

/// Run a single-`Op::EmbedGather` graph on `be`: `dst[r, :] = dequant(table[ids[r], :]) * scale`.
/// `ids` is an I32 input (token ids); `table` uploads as its raw native `table_dtype` bytes.
/// Returns the downloaded f32 output `[rows, ne]`.
fn run_embed_gather(
    be: &dyn Backend,
    ids: &[i32],
    table_bytes: &[u8],
    table_dtype: DType,
    vocab: usize,
    ne: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = ids.len();
    let mut g = Graph::new();
    let ids_id = g.input(TensorDesc::new(vec![rows], DType::I32));
    let tbl = g.weight(TensorDesc::new(vec![vocab * ne], table_dtype));
    let dst = g.output(f32d(rows * ne));
    g.push(Op::EmbedGather {
        ids: ids_id,
        table: tbl,
        dst,
        rows: rows as u32,
        ne: ne as u32,
        scale,
    });
    let plan = be.compile(&g).expect("compile");
    let ids_bytes: &[u8] = bytemuck::cast_slice(ids);
    let ib = be
        .alloc(ids_bytes.len(), BufferUsage::Activations)
        .expect("ids");
    be.upload(ib.as_ref(), ids_bytes).unwrap();
    let tb = be
        .alloc(table_bytes.len(), BufferUsage::Weights)
        .expect("table");
    be.upload(tb.as_ref(), table_bytes).unwrap();
    let ob = be.alloc(rows * ne * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(ids_id, ib.as_ref());
    b.bind(tbl, tb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; rows * ne];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// EmbedGather with a NON-1.0 scale (Gemma's sqrt(n_embd)): the scale must be applied on-device.
/// The pre-fix bug dropped the scale entirely (the HIP kernel had no `scale` param), so a Gemma
/// model's token embeddings came out unscaled — this test would fail loudly against the CPU
/// reference (`v * scale`). Covers an F16 table and a Q4_K table.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];

    // ── F16 table ──
    // Fresh backend per case: `dequant_weight_or_cache` keys the dequantized-weight cache by the
    // table's raw device pointer, and a table buffer freed at the end of one case can have its VRAM
    // address recycled by the next case's table — a stale cache hit would then feed the wrong
    // dequantized rows. Real models never hit this (weights are long-lived), but back-to-back
    // single-op test cases do; a per-case backend gives each an empty cache.
    {
        let be = rocm().unwrap();
        let (vocab, ne) = (6usize, 8usize);
        let scale = (ne as f32).sqrt(); // non-1.0, mirrors Gemma's embed scaling
        let tf32 = gen(vocab * ne, 41);
        let t_bytes: Vec<u8> = tf32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::F16, vocab, ne, scale);
        let r = run_embed_gather(&be, &ids, &t_bytes, DType::F16, vocab, ne, scale);
        let e = maxerr(&c, &r);
        let ref_mag = maxabs(&c).max(1e-6);
        println!(
            "EmbedGather F16 scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
            e / ref_mag
        );
        assert!(
            ref_mag > 1e-3,
            "EmbedGather F16 reference is all-zero — test is vacuous"
        );
        assert!(
            e / ref_mag < 1e-3,
            "EmbedGather F16 diverges from CPU reference: abs={e:e} rel={:e}",
            e / ref_mag
        );
    }

    // ── Q4_K table (ne must be a multiple of 256 = one super-block per row) ──
    {
        let be = rocm().unwrap();
        let (vocab, ne) = (6usize, 256usize); // vocab > max(ids)=5
        let scale = (ne as f32).sqrt();
        let blocks = (vocab * ne) / 256; // one block per vocab row
        let mut t_bytes = vec![0u8; blocks * 144];
        for (i, byte) in t_bytes.iter_mut().enumerate() {
            *byte = ((i * 37 + 11) & 0xFF) as u8;
        }
        // Q4_K super-block = d(2) + dmin(2) + scales(12) + qs(128). Set the f16 d/dmin slots to
        // finite small values, and the 12 packed 6-bit sub-block scale/min bytes to a benign
        // constant. Random bytes in those scale nibbles hit adversarial corners where the two
        // independent Q4_K decoders (infr-cpu ref vs infr-gguf device dequant) diverge on a
        // handful of raw elements — a dot product (the linear test) averages that away, but a raw
        // per-element gather exposes it. A benign, in-range sub-scale keeps both decoders in lock-
        // step so the comparison isolates the embed gather + on-device SCALE, not quant corners.
        for blk in 0..blocks {
            let base = blk * 144;
            t_bytes[base..base + 2].copy_from_slice(&half::f16::from_f32(0.375).to_le_bytes());
            t_bytes[base + 2..base + 4].copy_from_slice(&half::f16::from_f32(-0.125).to_le_bytes());
            for b in t_bytes[base + 4..base + 16].iter_mut() {
                *b = 0x11;
            }
        }
        let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::Q4K, vocab, ne, scale);
        let r = run_embed_gather(&be, &ids, &t_bytes, DType::Q4K, vocab, ne, scale);
        let e = maxerr(&c, &r);
        let ref_mag = maxabs(&c).max(1e-3);
        println!(
            "EmbedGather Q4_K scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
            e / ref_mag
        );
        assert!(
            ref_mag > 1e-3,
            "EmbedGather Q4_K reference is all-zero — test is vacuous"
        );
        assert!(
            e / ref_mag < 2e-2,
            "EmbedGather Q4_K diverges from CPU reference: abs={e:e} rel={:e}",
            e / ref_mag
        );
    }
}

// ── MoeFfn (router GEMV → gating → top-k → expert FFN → weighted sum) vs CPU ──

/// f16 little-endian bytes for an f32 slice (expert weight banks upload as raw f16).
fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

/// Run a single-`Op::MoeFfn` graph on `be` and return the downloaded f32 output `[rows, ne]`.
/// `router` is F32 `[n_expert, ne]`; the gate/up/down expert banks upload as their raw `gate_dt`/
/// `up_dt`/`down_dt` bytes (gate/up are `[n_expert, n_ff_exp, ne]`, down is
/// `[n_expert, ne, n_ff_exp]`, row-major). `router_x` is bound to the SAME handle as `x` (the
/// qwen3moe convention). Passing F16 banks exercises the dequant→f16 fallback; passing a covered
/// quant (Q4_K/Q6_K/Q8_0, optionally mixed like Q4_K_M's Q6_K down) exercises the native path.
#[allow(clippy::too_many_arguments)]
fn run_moe(
    be: &dyn Backend,
    x: &[f32],
    router_f32: &[f32],
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    gate_dt: DType,
    up_dt: DType,
    down_dt: DType,
    rows: usize,
    ne: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    gating: MoeGating,
    norm_w: bool,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(rows * ne));
    let rid = g.weight(TensorDesc::new(vec![n_expert * ne], DType::F32));
    let gid = g.weight(TensorDesc::new(vec![n_expert * n_ff_exp * ne], gate_dt));
    let uid = g.weight(TensorDesc::new(vec![n_expert * n_ff_exp * ne], up_dt));
    let did = g.weight(TensorDesc::new(vec![n_expert * ne * n_ff_exp], down_dt));
    let dst = g.output(f32d(rows * ne));
    g.push(Op::MoeFfn {
        x: xid,
        router_x: xid,
        router: rid,
        gate_exps: gid,
        up_exps: uid,
        down_exps: did,
        down_scale: None,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: n_ff_exp as u32,
        scale: 1.0,
        act: Activation::Silu,
        gating,
        norm_w,
        weight_before: false,
        fused_gate_up: false,
        ep_band: None,
    });
    let plan = be.compile(&g).expect("compile");

    let up = |desc_bytes: &[u8], usage| {
        let b = be.alloc(desc_bytes.len(), usage).expect("alloc");
        be.upload(b.as_ref(), desc_bytes).unwrap();
        b
    };
    let xb = up(bytemuck::cast_slice(x), BufferUsage::Activations);
    let rb = up(bytemuck::cast_slice(router_f32), BufferUsage::Weights);
    let gb = up(gate_bytes, BufferUsage::Weights);
    let ub = up(up_bytes, BufferUsage::Weights);
    let db = up(down_bytes, BufferUsage::Weights);
    let ob = be.alloc(rows * ne * 4, BufferUsage::Readback).expect("out");

    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(rid, rb.as_ref());
    b.bind(gid, gb.as_ref());
    b.bind(uid, ub.as_ref());
    b.bind(did, db.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");

    let mut o = vec![0f32; rows * ne];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// Small synthetic MoE (F32 router + F16 experts): the ROCm router GEMV → gating → top-k →
/// renorm → per-expert gated FFN → weighted-sum path must match the CPU reference. Exercises
/// the softmax+renorm (qwen3moe) path and the sigmoid+no-renorm gating path.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, ne, n_expert, n_used, n_ff_exp) = (2usize, 128usize, 4usize, 2usize, 64usize);

    let x = gen(rows * ne, 3);
    // Distinct salts per bank so router logits are well-separated (deterministic top-k).
    let router = gen(n_expert * ne, 9);
    let gate = f16_bytes(&gen(n_expert * n_ff_exp * ne, 11));
    let up = f16_bytes(&gen(n_expert * n_ff_exp * ne, 17));
    let down = f16_bytes(&gen(n_expert * ne * n_ff_exp, 23));

    for (gating, norm_w, label) in [
        (MoeGating::Softmax, true, "softmax+renorm"),
        (MoeGating::Sigmoid, false, "sigmoid+no-renorm"),
    ] {
        let c = run_moe(
            &cpu,
            &x,
            &router,
            &gate,
            &up,
            &down,
            DType::F16,
            DType::F16,
            DType::F16,
            rows,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            gating,
            norm_w,
        );
        let r = run_moe(
            &be,
            &x,
            &router,
            &gate,
            &up,
            &down,
            DType::F16,
            DType::F16,
            DType::F16,
            rows,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            gating,
            norm_w,
        );
        let e = maxerr(&c, &r);
        let ref_mag = maxabs(&c).max(1e-6);
        println!(
            "MoeFfn [{label}] max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
            e / ref_mag
        );
        // Guard against a silently-zero output masquerading as agreement (the pre-fix bug
        // produced garbage/zeros because the router weight was never applied).
        assert!(
            ref_mag > 1e-3,
            "MoeFfn [{label}] reference is all-zero — test is vacuous"
        );
        assert!(
            e / ref_mag < 2e-2,
            "MoeFfn [{label}] diverges from CPU reference: abs={e:e} rel={:e}",
            e / ref_mag
        );
    }
}

/// Regression (R6): the dequant→f16 weight cache must key on the bound buffer's IDENTITY, not on
/// its device ADDRESS. HIP hands a just-freed address straight back to the next same-sized
/// `hipMalloc`, so a second graph run on the same backend sees the first run's addresses again —
/// and before the fix it was served the FIRST run's dequantized weights for a bank whose bytes had
/// changed, i.e. silently wrong weights with no error anywhere.
///
/// The MoE case makes it deterministic: `gate`/`up`/`down` are all the SAME byte length here
/// (`n_expert*n_ff_exp*ne` f16 either way), so the three banks are exactly the recycling pool for
/// each other. Run once with `(a, b, c)`, then again with the contents ROTATED to `(c, a, b)` — the
/// same three addresses now back different bytes — and require the second run to still match the
/// CPU reference. Pre-fix this fails by ~200% (the same signature as the intermittent
/// `moe_ffn_matches_cpu` flake, which hit the identical collision whenever concurrent test threads
/// perturbed the allocator's free-list order between its two gating arms).
#[test]
#[ignore = "requires a ROCm GPU"]
fn weight_dequant_cache_survives_recycled_device_addresses() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, ne, n_expert, n_used, n_ff_exp) = (2usize, 128usize, 4usize, 2usize, 64usize);

    let x = gen(rows * ne, 3);
    let router = gen(n_expert * ne, 9);
    // Three same-sized expert banks with WELL-SEPARATED contents, so serving one bank's dequant
    // for another cannot pass by luck.
    let a = f16_bytes(&gen(n_expert * n_ff_exp * ne, 11));
    let b = f16_bytes(&gen(n_expert * n_ff_exp * ne, 17));
    let c = f16_bytes(&gen(n_expert * ne * n_ff_exp, 23));
    assert_eq!(a.len(), b.len(), "banks must be the same size to collide");
    assert_eq!(a.len(), c.len(), "banks must be the same size to collide");

    let go = |backend: &dyn Backend, g: &[u8], u: &[u8], d: &[u8]| {
        run_moe(
            backend,
            &x,
            &router,
            g,
            u,
            d,
            DType::F16,
            DType::F16,
            DType::F16,
            rows,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            MoeGating::Softmax,
            true,
        )
    };

    // Pass 1 populates the cache; its buffers are freed on return.
    let _ = go(&be, &a, &b, &c);
    // Pass 2 reuses those addresses for ROTATED contents.
    let want = go(&cpu, &c, &a, &b);
    let got = go(&be, &c, &a, &b);

    let e = maxerr(&want, &got);
    let ref_mag = maxabs(&want).max(1e-6);
    println!(
        "MoeFfn [recycled addresses] max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "MoeFfn [recycled addresses] reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 2e-2,
        "stale dequant served for a recycled device address: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Quantized MoE experts (Slice 18 / Slice 20): gate/up as Q4_K + down as Q6_K — the Q4_K_M
/// expert-bank layout. By default this now exercises the Slice-20 int8-activation dp4a expert path
/// (`moe_gate_up_act_i8_q4k` + `moe_down_i8_q6k`): the token input is int8-quantized once, the
/// activation `h` int8-quantized per expert, and each projection is an integer dot with scale-after
/// — TWO lossy int8 stages, so the tolerance is widened past the dense single-stage int8 GEMV
/// (which is 3e-2 for Q4_K/Q6_K). It must still match the CPU reference (`dequant_block` + f32 FFN).
/// Uses the qwen3moe softmax+renorm gating path. Blocks are built by the same `q4k_blocks`/
/// `q6k_blocks` helpers the dense native-decode GEMV tests use, so both f16-scale slots stay finite.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_quant_experts_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // ne and n_ff_exp are multiples of 256 → one whole number of super-blocks per expert row,
    // so every per-expert byte offset lands on a block boundary (the native path's requirement).
    let (rows, ne, n_expert, n_used, n_ff_exp) = (2usize, 256usize, 4usize, 2usize, 256usize);

    let x = gen(rows * ne, 3);
    let router = gen(n_expert * ne, 9);
    // gate/up: Q4_K [n_expert, n_ff_exp, ne]; down: Q6_K [n_expert, ne, n_ff_exp].
    let gate = q4k_blocks(n_expert * n_ff_exp * ne / 256);
    let up = q4k_blocks(n_expert * n_ff_exp * ne / 256);
    let down = q6k_blocks(n_expert * ne * n_ff_exp / 256);

    let c = run_moe(
        &cpu,
        &x,
        &router,
        &gate,
        &up,
        &down,
        DType::Q4K,
        DType::Q4K,
        DType::Q6K,
        rows,
        ne,
        n_expert,
        n_used,
        n_ff_exp,
        MoeGating::Softmax,
        true,
    );
    let r = run_moe(
        &be,
        &x,
        &router,
        &gate,
        &up,
        &down,
        DType::Q4K,
        DType::Q4K,
        DType::Q6K,
        rows,
        ne,
        n_expert,
        n_used,
        n_ff_exp,
        MoeGating::Softmax,
        true,
    );
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-6);
    println!(
        "MoeFfn [Q4_K/Q4_K/Q6_K experts] max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "MoeFfn [quant experts] reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 6e-2,
        "MoeFfn [quant experts] diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q5_K MoE experts (R1): gate/up as Q5_K + down as Q6_K — the Q5_K_M expert-bank layout, and the
/// `("q5k", "q6k")` cell of the now-4×4 `moe_expert_kernel` / `moe_gate_up_i8_kernel` tables. Same
/// two-lossy-int8-stage path (and therefore the same widened tolerance) as the Q4_K_M case above.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_q5k_experts_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, ne, n_expert, n_used, n_ff_exp) = (2usize, 256usize, 4usize, 2usize, 256usize);

    let x = gen(rows * ne, 3);
    let router = gen(n_expert * ne, 9);
    // gate/up: Q5_K [n_expert, n_ff_exp, ne]; down: Q6_K [n_expert, ne, n_ff_exp].
    let gate = q5k_blocks(n_expert * n_ff_exp * ne / 256);
    let up = q5k_blocks(n_expert * n_ff_exp * ne / 256);
    let down = q6k_blocks(n_expert * ne * n_ff_exp / 256);

    let c = run_moe(
        &cpu,
        &x,
        &router,
        &gate,
        &up,
        &down,
        DType::Q5K,
        DType::Q5K,
        DType::Q6K,
        rows,
        ne,
        n_expert,
        n_used,
        n_ff_exp,
        MoeGating::Softmax,
        true,
    );
    let r = run_moe(
        &be,
        &x,
        &router,
        &gate,
        &up,
        &down,
        DType::Q5K,
        DType::Q5K,
        DType::Q6K,
        rows,
        ne,
        n_expert,
        n_used,
        n_ff_exp,
        MoeGating::Softmax,
        true,
    );
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-6);
    println!(
        "MoeFfn [Q5_K/Q5_K/Q6_K experts] max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "MoeFfn [Q5_K experts] reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 6e-2,
        "MoeFfn [Q5_K experts] diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Shared MoE-expert parity check: `run_moe` on ROCm vs the CPU f32 reference for one (gate/up,
/// down) format pair, at the widened two-lossy-int8-stage tolerance the Q4_K_M/Q5_K_M cases use.
fn check_moe_experts(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    gu: DType,
    dn: DType,
    label: &str,
) {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, ne, n_expert, n_used, n_ff_exp) = (2usize, 256usize, 4usize, 2usize, 256usize);
    let x = gen(rows * ne, 3);
    let router = gen(n_expert * ne, 9);
    let run = |b: &dyn Backend| {
        run_moe(
            b,
            &x,
            &router,
            gate_bytes,
            up_bytes,
            down_bytes,
            gu,
            gu,
            dn,
            rows,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            MoeGating::Softmax,
            true,
        )
    };
    let c = run(&cpu);
    let r = run(&be);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-6);
    println!(
        "MoeFfn [{label} experts] max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "MoeFfn [{label} experts] reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 6e-2,
        "MoeFfn [{label} experts] diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q2_K gate/up + Q3_K down MoE experts (R2): the `("q2k", "q3k")` cell of the now-6×6
/// `moe_expert_kernel` table, plus `moe_gate_up_act_i8_q2k` and `moe_down_i8_q3k`.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_q2k_gate_q3k_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 256;
    check_moe_experts(
        &q2k_blocks(gu_blocks),
        &q2k_blocks(gu_blocks),
        &q3k_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Q2K,
        DType::Q3K,
        "Q2_K/Q2_K/Q3_K",
    );
}

/// Q3_K gate/up + Q2_K down MoE experts (R2): the mirrored `("q3k", "q2k")` cell, so both new
/// formats are exercised in BOTH the gate/up and the down role across the two cases.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_q3k_gate_q2k_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 256;
    check_moe_experts(
        &q3k_blocks(gu_blocks),
        &q3k_blocks(gu_blocks),
        &q2k_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Q3K,
        DType::Q2K,
        "Q3_K/Q3_K/Q2_K",
    );
}

/// Q4_0 gate/up + Q4_1 down MoE experts (R3): `moe_gate_up_act_i8_q40` + `moe_down_i8_q41`, and the
/// `("q40", "q41")` cell of the `moe_expert_kernel` table (llama.cpp bumps a Q4_0 model's `ffn_down`
/// to Q4_1, so this is the shape a real legacy-quant MoE actually has).
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_q40_gate_q41_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    check_moe_experts(
        &q40_blocks(gu_blocks),
        &q40_blocks(gu_blocks),
        &q41_blocks(n_expert * ne * n_ff_exp / 32),
        DType::Q4_0,
        DType::Q4_1,
        "Q4_0/Q4_0/Q4_1",
    );
}

/// Q5_1 gate/up + Q8_0 down MoE experts (R3): the mirrored role for the affine 5-bit format, plus
/// the `("q51", "q80")` cell — legacy ftypes bump `ffn_down` to Q8_0, the other reachable legacy
/// pairing, and Q8_0 down is the one that carries no min term at all.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_q51_gate_q80_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    check_moe_experts(
        &q51_blocks(gu_blocks),
        &q51_blocks(gu_blocks),
        &q80_blocks(n_expert * ne * n_ff_exp / 32),
        DType::Q5_1,
        DType::Q8_0,
        "Q5_1/Q5_1/Q8_0",
    );
}

/// IQ4_XS gate/up + IQ4_NL down MoE experts (R4): `moe_gate_up_act_i8_iq4xs` + `moe_down_i8_iq4nl`
/// and the `("iq4xs", "iq4nl")` cell — llama.cpp's IQ4 mixes pair an IQ4_XS gate/up with an IQ4_NL
/// `ffn_down` whenever the down row is not 256-divisible, so this is a shape a real GGUF has.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq4xs_gate_iq4nl_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 256;
    check_moe_experts(
        &iq4xs_blocks(gu_blocks),
        &iq4xs_blocks(gu_blocks),
        &iq4nl_blocks(n_expert * ne * n_ff_exp / 32),
        DType::Iq4Xs,
        DType::Iq4Nl,
        "IQ4_XS/IQ4_XS/IQ4_NL",
    );
}

/// IQ4_NL gate/up + IQ4_XS down MoE experts (R4): the mirrored `("iq4nl", "iq4xs")` cell, so both
/// new formats are exercised in BOTH the gate/up and the down role across the two cases.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq4nl_gate_iq4xs_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    check_moe_experts(
        &iq4nl_blocks(gu_blocks),
        &iq4nl_blocks(gu_blocks),
        &iq4xs_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq4Nl,
        DType::Iq4Xs,
        "IQ4_NL/IQ4_NL/IQ4_XS",
    );
}

/// IQ2_S gate/up + IQ3_S down MoE experts (R5): the cell a real GGUF on this box actually needs —
/// `Qwen3.6-35B-A3B-UD-IQ3_S` packs `ffn_gate_exps`/`ffn_up_exps` as IQ2_S with `ffn_down_exps` as
/// IQ3_S. Exercises `moe_gate_up_act_i8_iq2s` + `moe_down_i8_iq2s`'s partner `moe_down_i8_iq3s`,
/// and the `("iq2s","iq3s")` cross-product cell, with a two-scales-per-block gate/up decode against
/// a one-scale-per-block down decode.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq2s_gate_iq3s_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &iq2s_blocks(n_expert * n_ff_exp * ne / 256),
        &iq2s_blocks(n_expert * n_ff_exp * ne / 256),
        &iq3s_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq2S,
        DType::Iq3S,
        "IQ2_S/IQ2_S/IQ3_S",
    );
}

/// IQ2_XXS gate/up + IQ4_XS down MoE experts (R5): the other real IQ shape — the same 35B GGUF
/// bumps 3 of its `ffn_down_exps` to IQ4_XS — and it pairs an R5 grid gate/up with an R4 CODEBOOK
/// down, i.e. two different table mechanisms in one expert.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq2xxs_gate_iq4xs_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &iq2xxs_blocks(n_expert * n_ff_exp * ne / 256),
        &iq2xxs_blocks(n_expert * n_ff_exp * ne / 256),
        &iq4xs_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq2Xxs,
        DType::Iq4Xs,
        "IQ2_XXS/IQ2_XXS/IQ4_XS",
    );
}

/// IQ3_XXS gate/up + Q6_K down MoE experts (R5): the `use_more_bits` K-quant down bump, pairing a
/// no-min grid gate/up decode with a down decode that HAS a min term — the mixed-mechanism cell
/// that a shared-scale or shared-ones-dot slip between the two families would break.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq3xxs_gate_q6k_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &iq3xxs_blocks(n_expert * n_ff_exp * ne / 256),
        &iq3xxs_blocks(n_expert * n_ff_exp * ne / 256),
        &q6k_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq3Xxs,
        DType::Q6K,
        "IQ3_XXS/IQ3_XXS/Q6_K",
    );
}

/// IQ2_XS gate/up + IQ3_XXS down MoE experts (R5): the two formats the first three cases leave
/// only in a gate/up or only in a down role, so across the four every R5 format is exercised in
/// BOTH roles at least once.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq2xs_gate_iq3xxs_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &iq2xs_blocks(n_expert * n_ff_exp * ne / 256),
        &iq2xs_blocks(n_expert * n_ff_exp * ne / 256),
        &iq3xxs_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq2Xs,
        DType::Iq3Xxs,
        "IQ2_XS/IQ2_XS/IQ3_XXS",
    );
}

/// IQ1_S gate/up + IQ1_S down MoE experts (R6): the cell the cached UD-IQ1_S mix actually needs —
/// `Qwen3-0.6B-UD-IQ1_S` leaves 18 of its 28 `ffn_down` tensors at IQ1_S under an IQ1_S gate/up,
/// and `llama_tensor_get_type` applies the same rule to `ffn_down_exps`. Unlike every R5 grid cell
/// this puts the SAME delta-carrying format on both sides of the activation.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq1s_gate_iq1s_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &iq1s_blocks(n_expert * n_ff_exp * ne / 256),
        &iq1s_blocks(n_expert * n_ff_exp * ne / 256),
        &iq1s_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq1S,
        DType::Iq1S,
        "IQ1_S/IQ1_S/IQ1_S",
    );
}

/// IQ1_M gate/up + IQ3_S down MoE experts (R6): the other observed UD-IQ1 shape (the IQ1_M mix
/// boosts 5 of its `ffn_down` tensors to IQ3_S), and it pairs the per-16-scale/per-8-delta gate/up
/// decode against an R5 grid down decode — two different table mechanisms in one expert.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq1m_gate_iq3s_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &iq1m_blocks(n_expert * n_ff_exp * ne / 256),
        &iq1m_blocks(n_expert * n_ff_exp * ne / 256),
        &iq3s_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq1M,
        DType::Iq3S,
        "IQ1_M/IQ1_M/IQ3_S",
    );
}

/// IQ1_S gate/up + Q4_K down MoE experts (R6): the `use_more_bits` K-quant down bump, pairing a
/// no-min IQ1 gate/up decode with a down decode that HAS a min term — the mixed-mechanism cell a
/// shared ones-dot slip between the two families would break.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq1s_gate_q4k_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &iq1s_blocks(n_expert * n_ff_exp * ne / 256),
        &iq1s_blocks(n_expert * n_ff_exp * ne / 256),
        &q4k_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq1S,
        DType::Q4K,
        "IQ1_S/IQ1_S/Q4_K",
    );
}

/// TQ2_0 experts throughout (R6): the ternary self pair. A ternary checkpoint carries ONE type on
/// every FFN tensor, so this is the whole reachable shape for the family — and it is the only MoE
/// case in the suite whose weights are zero-centred by a folded constant rather than by a table.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_tq20_gate_tq20_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &tq20_blocks(n_expert * n_ff_exp * ne / 256),
        &tq20_blocks(n_expert * n_ff_exp * ne / 256),
        &tq20_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Tq2_0,
        DType::Tq2_0,
        "TQ2_0/TQ2_0/TQ2_0",
    );
}

/// Q2_0 experts throughout (R6): the same ternary self pair for infr's own 64-element format, which
/// additionally pins the per-expert BYTE OFFSET arithmetic at a block size no other MoE case uses
/// (`(elem_off / 64) * 18` rather than `/ 256 * bpb`).
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_q20_gate_q20_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &q20_blocks(n_expert * n_ff_exp * ne / 64),
        &q20_blocks(n_expert * n_ff_exp * ne / 64),
        &q20_blocks(n_expert * ne * n_ff_exp / 64),
        DType::Q2_0,
        DType::Q2_0,
        "Q2_0/Q2_0/Q2_0",
    );
}

/// MXFP4 experts throughout (R7): the ONE cell `gpt-oss` needs, and the whole reachable set for the
/// format. `llama_tensor_get_type` handles `MXFP4_MOE` as a single unconditional arm — MoE tensors
/// MXFP4, everything else Q8_0 — so gate/up and down are the same type by construction, and the
/// cached `gpt-oss-20b-MXFP4` is exactly that shape (all 72 `ffn_*_exps` MXFP4, every dense tensor
/// Q8_0). This is also the only MoE case in the suite whose per-expert byte offset divides by a
/// 17-byte block: `(elem_off / 32) * 17`, the only odd block stride in the covered set.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_mxfp4_gate_mxfp4_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &mxfp4_blocks(n_expert * n_ff_exp * ne / 32),
        &mxfp4_blocks(n_expert * n_ff_exp * ne / 32),
        &mxfp4_blocks(n_expert * ne * n_ff_exp / 32),
        DType::Mxfp4,
        DType::Mxfp4,
        "MXFP4/MXFP4/MXFP4",
    );
}

/// NVFP4 experts throughout (R7): the same self pair for the 64-element, four-sub-block-scale
/// sibling — no NVFP4 GGUF is cached, so this synthetic cell is the only thing that runs the
/// format's MoE kernels at all.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_nvfp4_gate_nvfp4_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    check_moe_experts(
        &nvfp4_blocks(n_expert * n_ff_exp * ne / 64),
        &nvfp4_blocks(n_expert * n_ff_exp * ne / 64),
        &nvfp4_blocks(n_expert * ne * n_ff_exp / 64),
        DType::Nvfp4,
        DType::Nvfp4,
        "NVFP4/NVFP4/NVFP4",
    );
}

/// IQ4_XS gate/up + Q6_K down MoE experts (R4): the mixed codebook×K-quant cell `("iq4xs", "q6k")`
/// — an IQ4_XS ftype bumps `ffn_down` to Q6_K under `use_more_bits`, which is the most common real
/// IQ4 MoE shape, and it pairs a no-min gate/up decode with a down decode that HAS a min term.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_iq4xs_gate_q6k_down_experts_matches_cpu() {
    let (ne, n_expert, n_ff_exp) = (256usize, 4usize, 256usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 256;
    check_moe_experts(
        &iq4xs_blocks(gu_blocks),
        &iq4xs_blocks(gu_blocks),
        &q6k_blocks(n_expert * ne * n_ff_exp / 256),
        DType::Iq4Xs,
        DType::Q6K,
        "IQ4_XS/IQ4_XS/Q6_K",
    );
}

// ── R8: the id-indexed multi-slot MoE expert GEMV (`moe_*_idm_*`) ────────────
//
// Every `moe_ffn_*_experts_matches_cpu` case above now RUNS on this tier (it took over the
// resident int8 expert path wholesale), so they are its aggregate gate. These four cases attack
// what an aggregate tolerance cannot see.
//
// The bug this tier invites is a wrong `id → bank slice` mapping. It does not produce garbage: it
// produces a perfectly plausible FFN output computed by the WRONG expert, and with the usual
// pseudo-random per-expert banks every expert has about the same magnitude, so an off-by-one lands
// inside a 6e-2 relative bound about as often as not. So the banks below are built so that expert
// `e`'s contribution scales as `1.35^(3e)` — an off-by-one expert is a ~2.5× error, and the test
// PROVES that by re-running the reference with the banks rotated and requiring the GPU to disagree
// with it.

/// `blocks` Q8_0 blocks whose CODES are identical everywhere and whose per-block f16 scale is
/// `scale(block / blocks_per_expert)` — i.e. every expert holds the same quantized pattern at its
/// own magnitude. That factorization is what makes a mis-routed expert a pure scale error, visible
/// in a single number instead of hidden in a dot product's noise.
fn q80_per_expert_scaled(
    blocks: usize,
    blocks_per_expert: usize,
    scale: impl Fn(usize) -> f32,
) -> Vec<u8> {
    let mut w = vec![0u8; blocks * 34];
    for blk in 0..blocks {
        let base = blk * 34;
        let e = blk / blocks_per_expert;
        w[base..base + 2].copy_from_slice(&half::f16::from_f32(scale(e)).to_le_bytes());
        for j in 0..32 {
            // Code pattern depends on the WITHIN-expert block index only, never on `e`.
            let wb = blk % blocks_per_expert;
            w[base + 2 + j] = (((wb * 7 + j * 5) % 251) as i32 - 125) as i8 as u8;
        }
    }
    w
}

/// The `Op::MoeFfn` runner with the two flags `run_moe` pins off — `fused_gate_up` (gate|up packed
/// as one `[n_expert, 2*n_ff_exp, ne]` bank, the `fused_up_half_boff` address path) and
/// `weight_before` (llama4's pre-activation routing-weight fold) — exposed, plus a caller-chosen
/// backend so the same problem can be run at two different `moe_id_rows`.
#[allow(clippy::too_many_arguments)]
fn run_moe_flags(
    be: &dyn Backend,
    x: &[f32],
    router_f32: &[f32],
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    dt: DType,
    rows: usize,
    ne: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    fused_gate_up: bool,
    weight_before: bool,
) -> Vec<f32> {
    let gu_elems = if fused_gate_up {
        n_expert * 2 * n_ff_exp * ne
    } else {
        n_expert * n_ff_exp * ne
    };
    let mut g = Graph::new();
    let xid = g.input(f32d(rows * ne));
    let rid = g.weight(TensorDesc::new(vec![n_expert * ne], DType::F32));
    let gid = g.weight(TensorDesc::new(vec![gu_elems], dt));
    // Fused: `up_exps` is never read (the executor takes the second half of the gate bank), but it
    // still has to be a bound handle of the right dtype — bind the gate bank itself, as the seam does.
    let uid = if fused_gate_up {
        gid
    } else {
        g.weight(TensorDesc::new(vec![n_expert * n_ff_exp * ne], dt))
    };
    let did = g.weight(TensorDesc::new(vec![n_expert * ne * n_ff_exp], dt));
    let dst = g.output(f32d(rows * ne));
    g.push(Op::MoeFfn {
        x: xid,
        router_x: xid,
        router: rid,
        gate_exps: gid,
        up_exps: uid,
        down_exps: did,
        down_scale: None,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: n_ff_exp as u32,
        scale: 1.0,
        act: Activation::Silu,
        gating: MoeGating::Softmax,
        norm_w: true,
        weight_before,
        fused_gate_up,
        ep_band: None,
    });
    let plan = be.compile(&g).expect("compile");

    let up = |desc_bytes: &[u8], usage| {
        let b = be.alloc(desc_bytes.len(), usage).expect("alloc");
        be.upload(b.as_ref(), desc_bytes).unwrap();
        b
    };
    let xb = up(bytemuck::cast_slice(x), BufferUsage::Activations);
    let rb = up(bytemuck::cast_slice(router_f32), BufferUsage::Weights);
    let gb = up(gate_bytes, BufferUsage::Weights);
    let ub = (!fused_gate_up).then(|| up(up_bytes, BufferUsage::Weights));
    let db = up(down_bytes, BufferUsage::Weights);
    let ob = be.alloc(rows * ne * 4, BufferUsage::Readback).expect("out");

    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(rid, rb.as_ref());
    b.bind(gid, gb.as_ref());
    if let Some(ub) = &ub {
        b.bind(uid, ub.as_ref());
    }
    b.bind(did, db.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");

    let mut o = vec![0f32; rows * ne];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// A ROCm backend with `kernels.rocm.moe_id_rows` overridden — the id tier's row-chunk bound.
fn rocm_id_rows(chunk: usize) -> Option<RocmBackend> {
    let mut cfg = infr_core::config::Config::default();
    cfg.kernels.rocm.moe_id_rows = chunk;
    RocmBackend::new_with(0, std::sync::Arc::new(cfg)).ok()
}

/// Per-expert magnitudes spanning ~6× over 8 experts, so one expert step is a ~2.5× output change.
fn expert_scale(e: usize) -> f32 {
    0.02 * 1.35f32.powi(e as i32)
}

/// Rotate a per-expert bank by one expert — the "off by one expert" the id tier could produce.
fn rotate_experts(bytes: &[u8], n_expert: usize) -> Vec<u8> {
    let per = bytes.len() / n_expert;
    let mut out = vec![0u8; bytes.len()];
    for e in 0..n_expert {
        let src = ((e + 1) % n_expert) * per;
        out[e * per..(e + 1) * per].copy_from_slice(&bytes[src..src + per]);
    }
    out
}

/// **The id→slice mapping test.** Each of `rows × n_used` slots must read the expert `moe_topk`
/// picked FOR THAT SLOT — not slot 0's, not the slot index, not a neighbour. Run at both the decode
/// shape (`rows == 1`, one chunk of one) and a multi-row shape, and each result is required BOTH to
/// match the CPU reference AND to be far from the reference computed with the expert banks rotated
/// by one. Without the second half the first is not evidence: it is exactly the tolerance's blind
/// spot (see the section header).
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_id_tier_maps_each_slot_to_its_own_expert() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (ne, n_expert, n_used, n_ff_exp) = (256usize, 8usize, 3usize, 128usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    let dn_blocks = n_expert * ne * n_ff_exp / 32;
    let gate = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let up = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let down = q80_per_expert_scaled(dn_blocks, dn_blocks / n_expert, expert_scale);
    let (rgate, rup, rdown) = (
        rotate_experts(&gate, n_expert),
        rotate_experts(&up, n_expert),
        rotate_experts(&down, n_expert),
    );

    for rows in [1usize, 4] {
        let x = gen(rows * ne, 3);
        let router = gen(n_expert * ne, 9);
        let go = |b: &dyn Backend, g: &[u8], u: &[u8], d: &[u8]| {
            run_moe_flags(
                b,
                &x,
                &router,
                g,
                u,
                d,
                DType::Q8_0,
                rows,
                ne,
                n_expert,
                n_used,
                n_ff_exp,
                false,
                false,
            )
        };
        let want = go(&cpu, &gate, &up, &down);
        let wrong = go(&cpu, &rgate, &rup, &rdown);
        let got = go(&be, &gate, &up, &down);

        let ref_mag = maxabs(&want).max(1e-6);
        let e = maxerr(&want, &got);
        let e_wrong = maxerr(&wrong, &got);
        println!(
            "MoeFfn [id tier, rows={rows}] rel={:e} rel_vs_rotated={:e} max|ref|={ref_mag:e}",
            e / ref_mag,
            e_wrong / ref_mag
        );
        assert!(
            ref_mag > 1e-3,
            "MoeFfn [id tier] reference is all-zero — test is vacuous"
        );
        assert!(
            e / ref_mag < 6e-2,
            "MoeFfn [id tier, rows={rows}] diverges from CPU: abs={e:e} rel={:e}",
            e / ref_mag
        );
        // The tripwire: the rotated-expert reference must be FAR outside the tolerance the check
        // above passed at, or that check proved nothing about which expert ran.
        assert!(
            e_wrong / ref_mag > 5e-1,
            "an off-by-one expert would have passed the tolerance — the mapping is untested \
             (rel_vs_rotated={:e})",
            e_wrong / ref_mag
        );
    }
}

/// The FUSED gate|up bank (`fused_up_half_boff`) and llama4's `weight_before` fold, on the id
/// tier. Fused is the one shape where the up-projection's expert address is not
/// `up_base + e*up_bstride` but `gate_base + e*gate_bstride + half`, i.e. TWO terms that must both
/// be right — swapping them still reads inside the bank and still produces a finite FFN.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_id_tier_fused_gate_up_and_weight_before_match_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, ne, n_expert, n_used, n_ff_exp) = (3usize, 256usize, 6usize, 2usize, 128usize);
    // Fused: [n_expert, 2*n_ff_exp, ne] — gate rows first, up rows second, per expert. The
    // per-expert scale ladder rides the WHOLE double-width slice, so a half-offset that lands in
    // the wrong expert is still a magnitude error.
    let gu_blocks = n_expert * 2 * n_ff_exp * ne / 32;
    let dn_blocks = n_expert * ne * n_ff_exp / 32;
    let gate = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let down = q80_per_expert_scaled(dn_blocks, dn_blocks / n_expert, expert_scale);
    let x = gen(rows * ne, 5);
    let router = gen(n_expert * ne, 13);

    for weight_before in [false, true] {
        let go = |b: &dyn Backend| {
            run_moe_flags(
                b,
                &x,
                &router,
                &gate,
                &[],
                &down,
                DType::Q8_0,
                rows,
                ne,
                n_expert,
                n_used,
                n_ff_exp,
                true,
                weight_before,
            )
        };
        let want = go(&cpu);
        let got = go(&be);
        let ref_mag = maxabs(&want).max(1e-6);
        let e = maxerr(&want, &got);
        println!(
            "MoeFfn [id tier, fused gate|up, weight_before={weight_before}] rel={:e} max|ref|={ref_mag:e}",
            e / ref_mag
        );
        assert!(ref_mag > 1e-3, "fused id-tier reference is all-zero");
        assert!(
            e / ref_mag < 6e-2,
            "MoeFfn [id tier, fused, weight_before={weight_before}] diverges: abs={e:e} rel={:e}",
            e / ref_mag
        );
    }
}

/// The ROW-CHUNK arithmetic (`MOE_ID_ROWS`). A chunk is a contiguous window of `rows × n_used`
/// slots, and the executor advances FIVE pointers per chunk (`x`, `dst`, `route_ids`,
/// `route_wts`, and implicitly the chunk-local `row = slot / n_used`); getting any of them wrong
/// mixes tokens, which a per-token comparison catches and an aggregate one may not.
///
/// Chunking is also a pure REGROUPING — it changes no per-slot arithmetic and no summation order —
/// so a 1-row chunk and a whole-batch chunk must agree BIT FOR BIT, not merely within tolerance.
/// That is a much sharper statement than the CPU comparison and it is the one that would catch a
/// chunk boundary that dropped or double-counted a row.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_id_tier_row_chunking_is_bit_neutral() {
    let (Some(be1), Some(be3), Some(be_all)) =
        (rocm_id_rows(1), rocm_id_rows(3), rocm_id_rows(1024))
    else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // 7 rows over a 3-row chunk: 3 + 3 + 1, so the last chunk is SHORT — the case an
    // `r0 + chunk <= rows` bound would silently drop.
    let (rows, ne, n_expert, n_used, n_ff_exp) = (7usize, 256usize, 8usize, 3usize, 128usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    let dn_blocks = n_expert * ne * n_ff_exp / 32;
    let gate = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let up = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let down = q80_per_expert_scaled(dn_blocks, dn_blocks / n_expert, expert_scale);
    let x = gen(rows * ne, 7);
    let router = gen(n_expert * ne, 19);
    let go = |b: &dyn Backend| {
        run_moe_flags(
            b,
            &x,
            &router,
            &gate,
            &up,
            &down,
            DType::Q8_0,
            rows,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            false,
            false,
        )
    };

    let want = go(&cpu);
    let all = go(&be_all);
    let ref_mag = maxabs(&want).max(1e-6);
    let e = maxerr(&want, &all);
    println!(
        "MoeFfn [id tier, 7 rows, unchunked] rel={:e} max|ref|={ref_mag:e}",
        e / ref_mag
    );
    assert!(ref_mag > 1e-3, "row-chunk reference is all-zero");
    assert!(
        e / ref_mag < 6e-2,
        "MoeFfn [id tier, unchunked] diverges from CPU: rel={:e}",
        e / ref_mag
    );

    for (label, got) in [("chunk=1", go(&be1)), ("chunk=3", go(&be3))] {
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            all.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "id-tier row chunking is not bit-neutral at {label} — the chunk changed the math, \
             not just the batching"
        );
    }
    // Per-token, not just in aggregate: a chunk-boundary pointer slip mixes whole token rows, and
    // a max-over-everything comparison against a same-magnitude neighbour can survive it.
    for r in 0..rows {
        let (w, g) = (&want[r * ne..(r + 1) * ne], &all[r * ne..(r + 1) * ne]);
        let rm = maxabs(w).max(1e-6);
        assert!(
            maxerr(w, g) / rm < 6e-2,
            "id-tier row {r} of {rows} does not match the CPU reference (rel={:e}) — a chunk \
             boundary crossed token rows",
            maxerr(w, g) / rm
        );
    }
}

/// **The id tier and the pre-R8 serial tier must agree BIT FOR BIT** (`moe_id_rows = 0` selects
/// the latter).
///
/// This is the strongest statement available about R8 and it is deliberately not a tolerance. The
/// two tiers run the SAME per-slot arithmetic — the same `i8acc_*` decode+dot, the same `wave_sum32`
/// reduction, the same per-32-block activation quant (block-independent, so batching the quant pass
/// across slots cannot change a single scale) — and the id tier's `moe_accum_idm` was written to
/// reproduce the serial loop's `atomicAdd` sequence onto a zeroed `dst` in ascending slot order
/// rather than to be "equivalent". A float that moves at all here means one of those claims is
/// wrong, most likely that the concurrent slots are racing on the output after all — the failure
/// mode that would show up on the box as a golden hash that drifts between runs rather than as a
/// wrong answer.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_id_tier_matches_the_serial_tier_bitwise() {
    let (Some(idm), Some(serial)) = (rocm_id_rows(128), rocm_id_rows(0)) else {
        return;
    };
    let (ne, n_expert, n_used, n_ff_exp) = (256usize, 8usize, 4usize, 128usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    let dn_blocks = n_expert * ne * n_ff_exp / 32;
    let gate = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let up = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let down = q80_per_expert_scaled(dn_blocks, dn_blocks / n_expert, expert_scale);
    let router = gen(n_expert * ne, 23);
    // Decode (1) and two prefill shapes, one of them a repeat run to catch a race that only
    // sometimes reorders.
    for rows in [1usize, 5, 5] {
        let x = gen(rows * ne, 11);
        let go = |b: &dyn Backend| {
            run_moe_flags(
                b,
                &x,
                &router,
                &gate,
                &up,
                &down,
                DType::Q8_0,
                rows,
                ne,
                n_expert,
                n_used,
                n_ff_exp,
                false,
                false,
            )
        };
        let a = go(&idm);
        let b = go(&serial);
        assert!(maxabs(&a) > 1e-3, "bitwise tier comparison is vacuous");
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the id tier does not reproduce the serial tier bit-for-bit at rows={rows} \
             (max_err={:e})",
            maxerr(&a, &b)
        );
    }
}

/// The id tier must not change what the A/B comparand does. With `kernels.rocm.i8 = false` the
/// resident path drops back to the pre-R8 per-`(row, slot)` `moe_ffn_expert_routed_*` loop; that
/// path still has to be there and still has to match the CPU reference, or the switch that exists
/// to isolate an int8 numerics question has quietly become a second untested tier.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_ffn_serial_tier_still_matches_cpu_with_i8_off() {
    let mut cfg = infr_core::config::Config::default();
    cfg.kernels.rocm.i8 = false;
    let Some(be) = RocmBackend::new_with(0, std::sync::Arc::new(cfg)).ok() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, ne, n_expert, n_used, n_ff_exp) = (2usize, 256usize, 4usize, 2usize, 256usize);
    let gate = q4k_blocks(n_expert * n_ff_exp * ne / 256);
    let up = q4k_blocks(n_expert * n_ff_exp * ne / 256);
    let down = q6k_blocks(n_expert * ne * n_ff_exp / 256);
    let x = gen(rows * ne, 3);
    let router = gen(n_expert * ne, 9);
    let go = |b: &dyn Backend| {
        run_moe(
            b,
            &x,
            &router,
            &gate,
            &up,
            &down,
            DType::Q4K,
            DType::Q4K,
            DType::Q6K,
            rows,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            MoeGating::Softmax,
            true,
        )
    };
    let want = go(&cpu);
    let got = go(&be);
    let ref_mag = maxabs(&want).max(1e-6);
    let e = maxerr(&want, &got);
    println!(
        "MoeFfn [serial tier, i8 off] rel={:e} max|ref|={ref_mag:e}",
        e / ref_mag
    );
    assert!(ref_mag > 1e-3, "serial-tier reference is all-zero");
    assert!(
        e / ref_mag < 6e-2,
        "MoeFfn [serial tier, i8 off] diverges from CPU: rel={:e}",
        e / ref_mag
    );
}

// ── Rope (ggml NORM interleaved RoPE, packed + strided) vs CPU ────────────────

/// Run a single-`Op::Rope` graph on `be` and return the FULL output buffer (length = `x.len()`).
/// `x` is the raw input: packed `[rows, n_head, head_dim]` when `x_stride == 0`, else a wider
/// `[rows, x_stride]` row buffer whose per-row `n_head*head_dim` query slice packs at the row
/// start. `positions` is an I32 tensor. `dst != x`, so the backend copies the (possibly strided)
/// source and rotates in place — the rotated query lands at `row*x_stride + h*head_dim`.
#[allow(clippy::too_many_arguments)]
fn run_rope(
    be: &dyn Backend,
    x: &[f32],
    positions: &[i32],
    rows: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    x_stride: usize,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(x.len()));
    let pid = g.input(TensorDesc::new(vec![positions.len()], DType::I32));
    let dst = g.output(f32d(x.len()));
    g.push(Op::Rope {
        x: xid,
        positions: pid,
        dst,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        rope_dim: rope_dim as u32,
        theta,
        freq_factors: None,
        x_stride: x_stride as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let pbytes: &[u8] = bytemuck::cast_slice(positions);
    let pb = be
        .alloc(pbytes.len(), BufferUsage::Activations)
        .expect("pos");
    be.upload(pb.as_ref(), pbytes).unwrap();
    let ob = be.alloc(x.len() * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(pid, pb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; x.len()];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// `Op::Rope` (the no-qk-norm llama-family INTERLEAVED rotation) must match the CPU reference for
/// BOTH a packed input and a NON-trivial `x_stride`. The pre-fix kernel had three defects any of
/// which this catches: (1) split-half (NEOX) pairing instead of interleaved (2p, 2p+1), (2) the
/// dropped `x_stride` — a strided view read the wrong elements — plus a `dst != x` copy that
/// grabbed a packed prefix regardless of stride, and (3) `freq *= freq_factors` (the wrong
/// direction). The strided case is the qwen35 q+g shape: the rotated query is a slice inside a
/// wider row buffer; a stride-blind kernel rotates the poison tail as extra heads and diverges.
#[test]
#[ignore = "requires a ROCm GPU"]
fn rope_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n_head, head_dim, rope_dim) = (3usize, 2usize, 8usize, 8usize);
    let theta = 10000.0f32;
    let positions: Vec<i32> = vec![1, 7, 4]; // non-zero + distinct per row so RoPE actually rotates
    let hw = n_head * head_dim; // packed per-row width
    let np = rows * hw;
    let packed = gen(np, 6); // logical query, packed [rows, n_head, head_dim]

    // ── (a) packed input (x_stride = 0 / natural) ──
    let c = run_rope(
        &cpu, &packed, &positions, rows, n_head, head_dim, rope_dim, theta, 0,
    );
    let r = run_rope(
        &be, &packed, &positions, rows, n_head, head_dim, rope_dim, theta, 0,
    );
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-6);
    println!(
        "Rope packed max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "Rope packed reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 1e-3,
        "Rope packed diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );

    // ── (b) NON-trivial x_stride: query slice inside a wider row buffer (qwen35 q+g shape) ──
    // Each row is `stride` wide; the query packs at the row start, the tail is POISON the kernel
    // must never touch. The CPU reference is the SAME query packed (CPU Op::Rope is packed-only),
    // so parity holds on the logical query values regardless of the wider ROCm layout.
    let stride = hw * 2; // double-width row, like the interleaved q+g buffer
    let mut wide = vec![0f32; rows * stride];
    for row in 0..rows {
        for i in 0..hw {
            wide[row * stride + i] = packed[row * hw + i];
        }
        for j in hw..stride {
            wide[row * stride + j] = 1000.0 + (row * stride + j) as f32; // large, distinctive poison
        }
    }
    let rs = run_rope(
        &be, &wide, &positions, rows, n_head, head_dim, rope_dim, theta, stride,
    );
    // Extract the packed roped query out of each strided row.
    let mut rs_packed = vec![0f32; np];
    for row in 0..rows {
        for i in 0..hw {
            rs_packed[row * hw + i] = rs[row * stride + i];
        }
    }
    let e2 = maxerr(&c, &rs_packed);
    let ref_mag2 = maxabs(&c).max(1e-6);
    println!(
        "Rope strided(stride={stride}) max_err={e2:e} max|ref|={ref_mag2:e} rel={:e}",
        e2 / ref_mag2
    );
    assert!(
        ref_mag2 > 1e-3,
        "Rope strided reference is all-zero — test is vacuous"
    );
    assert!(
        e2 / ref_mag2 < 1e-3,
        "Rope strided diverges from CPU reference (x_stride dropped?): abs={e2:e} rel={:e}",
        e2 / ref_mag2
    );
    // The poison tail must survive untouched: a stride-correct kernel only rotates the query slice.
    for row in 0..rows {
        for j in hw..stride {
            let idx = row * stride + j;
            assert!(
                (rs[idx] - wide[idx]).abs() < 1e-6,
                "rope touched the strided-row tail at {idx} — kernel read/wrote outside the query slice"
            );
        }
    }
}

// ── QkNormRope (fused per-head RMSNorm + NEOX split-half RoPE, strided q+g) vs CPU ──

/// Run a single-`Op::QkNormRope` graph on `be` and return the downloaded PACKED f32 output
/// `[rows, n_head, head_dim]`. `x` is the raw input: packed `[rows, n_head, head_dim]` when
/// `x_stride == 0`, else a wider `[rows, x_stride]` row buffer whose per-head query slice packs at
/// `row*x_stride + h*(x_stride/n_head)` (the qwen35 interleaved q+g layout). `weight` is the F16
/// per-head RMSNorm weight `[head_dim]`; `positions` is an I32 tensor.
#[allow(clippy::too_many_arguments)]
fn run_qk_norm_rope(
    be: &dyn Backend,
    x: &[f32],
    weight_f16: &[u8],
    positions: &[i32],
    rows: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    eps: f32,
    x_stride: usize,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(x.len()));
    let wid = g.weight(TensorDesc::new(vec![head_dim], DType::F16));
    let pid = g.input(TensorDesc::new(vec![positions.len()], DType::I32));
    let dst = g.output(f32d(rows * n_head * head_dim));
    g.push(Op::QkNormRope {
        x: xid,
        weight: wid,
        positions: pid,
        dst,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        rope_dim: rope_dim as u32,
        theta,
        eps,
        freq_factors: None,
        x_stride: x_stride as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let wb = be.alloc(weight_f16.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), weight_f16).unwrap();
    let pbytes: &[u8] = bytemuck::cast_slice(positions);
    let pb = be
        .alloc(pbytes.len(), BufferUsage::Activations)
        .expect("pos");
    be.upload(pb.as_ref(), pbytes).unwrap();
    let ob = be
        .alloc(rows * n_head * head_dim * 4, BufferUsage::Readback)
        .expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(pid, pb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; rows * n_head * head_dim];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// `Op::QkNormRope` (fused per-head RMSNorm + NEOX split-half RoPE) must match the CPU reference for
/// a MULTI-ROW (prefill) input with a NON-trivial `x_stride` — the qwen35 interleaved q+g layout
/// where each attention head is a strided slice of a wider `[q | gate]` row buffer. The pre-fix
/// kernel indexed the per-head base as `r*x_stride + h*head_dim` (packed head stride) instead of
/// `h*(x_stride/n_head)`, AND wrote the rotation in place into a packed-size buffer while indexing
/// it with the strided stride — an out-of-bounds read/write past the buffer on rows > 1 that MAFFs
/// on-device (qwen35 prefill op 67). It also divided the RoPE angle by the wrong `freq_factors`
/// direction. This test runs the SAME graph on `RocmBackend` and `infr_cpu::CpuBackend` and compares
/// the packed outputs; it fails loudly without the fix.
#[test]
#[ignore = "requires a ROCm GPU"]
fn qk_norm_rope_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n_head, head_dim, rope_dim) = (4usize, 3usize, 8usize, 8usize);
    let theta = 10000.0f32;
    let eps = 1e-6f32;
    let positions: Vec<i32> = vec![0, 1, 5, 9]; // distinct per row so RoPE actually rotates
    let hw = n_head * head_dim; // packed per-row query width

    // Per-head RMSNorm weight [head_dim], F16 (non-trivial so the norm scale is observable).
    let wf32 = gen(head_dim, 31);
    let weight_f16: Vec<u8> = wf32
        .iter()
        .flat_map(|&v| half::f16::from_f32(1.0 + v).to_bits().to_le_bytes())
        .collect();

    // Interleaved q+g row: stride = n_head * 2 * head_dim, head h at `h*2*head_dim`, query = first
    // head_dim of the head block, the trailing head_dim is POISON (the gate half) the kernel must
    // never read. Mirrors qwen35's attn q+g buffer (x_stride = nh*2*hd).
    let head_stride = 2 * head_dim;
    let stride = n_head * head_stride;
    let qpacked = gen(rows * hw, 6); // logical per-head queries, packed [rows, n_head, head_dim]
    let mut wide = vec![0f32; rows * stride];
    for row in 0..rows {
        for h in 0..n_head {
            for i in 0..head_dim {
                wide[row * stride + h * head_stride + i] =
                    qpacked[(row * n_head + h) * head_dim + i];
            }
            // poison the gate half of each head block
            for i in head_dim..head_stride {
                wide[row * stride + h * head_stride + i] = 1000.0 + (row * stride + h) as f32;
            }
        }
    }

    let c = run_qk_norm_rope(
        &cpu,
        &wide,
        &weight_f16,
        &positions,
        rows,
        n_head,
        head_dim,
        rope_dim,
        theta,
        eps,
        stride,
    );
    let r = run_qk_norm_rope(
        &be,
        &wide,
        &weight_f16,
        &positions,
        rows,
        n_head,
        head_dim,
        rope_dim,
        theta,
        eps,
        stride,
    );
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-6);
    println!(
        "QkNormRope strided(stride={stride}) max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    // Guard against a silently-zero output masquerading as agreement.
    assert!(
        ref_mag > 1e-3,
        "QkNormRope reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 2e-3,
        "QkNormRope strided diverges from CPU reference (OOB head stride / packed-vs-strided?): abs={e:e} rel={:e}",
        e / ref_mag
    );
}

// ── Conv1dSilu (depthwise causal conv + SiLU, rolling state) vs CPU ───────────

/// Run a single-`Op::Conv1dSilu` graph on `be` and return BOTH the downloaded output
/// `[rows, channels]` AND the updated `state` `[(kernel-1), channels]`. `state` is bound as an
/// F32 Input so the op mutates it in place; the backend must write the trailing `kernel-1`
/// columns of the virtual `[state ‖ x]` sequence back to that buffer (verified by downloading it
/// after execute — the same in-place-state-persistence contract as `seam_op_parity`'s state test).
/// `weight` uploads as raw F16 bytes (dequantized on first touch), so CPU (f16→f32) and ROCm
/// (f16 as-is) see identical kernel taps.
fn run_conv1d_silu(
    be: &dyn Backend,
    x: &[f32],
    weight_f16: &[u8],
    state_init: &[f32],
    rows: usize,
    channels: usize,
    kernel: usize,
) -> (Vec<f32>, Vec<f32>) {
    let km1 = kernel - 1;
    let mut g = Graph::new();
    let xid = g.input(f32d(rows * channels));
    let wid = g.weight(TensorDesc::new(vec![channels * kernel], DType::F16));
    let sid = g.input(f32d(km1 * channels)); // F32 Input → mutated in place, read back after
    let dst = g.output(f32d(rows * channels));
    g.push(Op::Conv1dSilu {
        x: xid,
        weight: wid,
        state: sid,
        dst,
        rows: rows as u32,
        channels: channels as u32,
        kernel: kernel as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let wb = be.alloc(weight_f16.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), weight_f16).unwrap();
    let sb = be
        .alloc(state_init.len() * 4, BufferUsage::Activations)
        .expect("state");
    be.upload(sb.as_ref(), bytemuck::cast_slice(state_init))
        .unwrap();
    let ob = be
        .alloc(rows * channels * 4, BufferUsage::Readback)
        .expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(sid, sb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut out = vec![0f32; rows * channels];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut out))
        .unwrap();
    let mut ns = vec![0f32; km1 * channels];
    be.download(sb.as_ref(), bytemuck::cast_slice_mut(&mut ns))
        .unwrap();
    (out, ns)
}

/// `Op::Conv1dSilu` (qwen35's depthwise causal 1-D conv + SiLU, rolling `state`) must match the CPU
/// reference for a MULTI-ROW (prefill) input with a NON-trivial initial `state` — BOTH the output
/// AND the updated state. The pre-fix ROCm kernel applied the SAME unchanged `state` to every one of
/// the `rows` output rows (no per-row window advance) and the host shift chained from the ORIGINAL
/// `x` for each row, so for `rows > 1` both the conv outputs and the returned state were wrong (only
/// `rows == 1` decode was correct) — one of the two bugs making qwen35 prefill incoherent. This runs
/// the SAME single-op graph on `RocmBackend` and `infr_cpu::CpuBackend`; it fails loudly without the
/// fix. Correct semantics: convolve the virtual sequence `[state ‖ x]` per (row, channel), and the
/// returned state is that sequence's trailing `kernel-1` columns.
#[test]
#[ignore = "requires a ROCm GPU"]
fn conv1d_silu_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, channels, kernel) = (6usize, 32usize, 4usize); // rows > 1 (prefill), rows > kernel-1
    let km1 = kernel - 1;

    let x = gen(rows * channels, 6);
    // Per-channel kernel [channels, kernel], F16 bytes (CPU dequants f16→f32, ROCm reads f16 as-is).
    let wf32 = gen(channels * kernel, 7);
    let w_bytes: Vec<u8> = wf32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    // NON-trivial initial state (exercises the cross-row warmup carry — a zeroed state would hide
    // the "state applied to every row unchanged" bug on the first km1 rows).
    let state_init = gen(km1 * channels, 13);

    let (c_out, c_state) = run_conv1d_silu(&cpu, &x, &w_bytes, &state_init, rows, channels, kernel);
    let (r_out, r_state) = run_conv1d_silu(&be, &x, &w_bytes, &state_init, rows, channels, kernel);

    let eo = maxerr(&c_out, &r_out);
    let out_mag = maxabs(&c_out).max(1e-6);
    let es = maxerr(&c_state, &r_state);
    let st_mag = maxabs(&c_state).max(1e-6);
    println!(
        "Conv1dSilu multirow(rows={rows}) out max_err={eo:e} max|ref|={out_mag:e} rel={:e} | state max_err={es:e} max|ref|={st_mag:e} rel={:e}",
        eo / out_mag,
        es / st_mag
    );
    // Guard against a silently-zero output/state masquerading as agreement.
    assert!(
        out_mag > 1e-3,
        "Conv1dSilu output reference is all-zero — test is vacuous"
    );
    assert!(
        st_mag > 1e-3,
        "Conv1dSilu state reference is all-zero — test is vacuous"
    );
    assert!(
        eo / out_mag < 2e-3,
        "Conv1dSilu multirow output diverges from CPU reference (per-row window not advanced?): abs={eo:e} rel={:e}",
        eo / out_mag
    );
    // The updated state is a pure gather from `[state ‖ x]` (no arithmetic), so f16 weight rounding
    // does not touch it — the returned state must match the CPU reference near-exactly.
    assert!(
        es / st_mag < 1e-5,
        "Conv1dSilu multirow updated state diverges from CPU reference (host chain from original x?): abs={es:e} rel={:e}",
        es / st_mag
    );
}

// ── DeltaNet (gated linear-attention recurrence, persistent S state) vs CPU ──

/// Run a single-`Op::DeltaNet` graph on `be` and return BOTH the downloaded output
/// `[rows, n_vhead*head_v]` AND the mutated recurrent state `[n_vhead, head_k, head_v]`. `state` is
/// bound as an F32 Input the op mutates IN PLACE (read back after execute — the persistent-state
/// contract: qwen35's DeltaNet-S survives across `execute` calls). `a_coef`/`dt_bias` upload as raw
/// F16 bytes (CPU dequants f16→f32, ROCm reads f16 as-is, so both see identical per-head scalars).
#[allow(clippy::too_many_arguments)]
fn run_deltanet(
    be: &dyn Backend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bcoef: &[f32],
    acoef_in: &[f32],
    a_coef_f16: &[u8],
    dt_bias_f16: &[u8],
    state_init: &[f32],
    rows: usize,
    n_vhead: usize,
    n_khead: usize,
    head_k: usize,
    head_v: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = Graph::new();
    let qid = g.input(f32d(rows * n_khead * head_k));
    let kid = g.input(f32d(rows * n_khead * head_k));
    let vid = g.input(f32d(rows * n_vhead * head_v));
    let bid = g.input(f32d(rows * n_vhead));
    let aid = g.input(f32d(rows * n_vhead));
    let acid = g.weight(TensorDesc::new(vec![n_vhead], DType::F16));
    let dtid = g.weight(TensorDesc::new(vec![n_vhead], DType::F16));
    let sid = g.input(f32d(n_vhead * head_k * head_v)); // F32 Input → mutated in place, read back
    let dst = g.output(f32d(rows * n_vhead * head_v));
    g.push(Op::DeltaNet {
        q: qid,
        k: kid,
        v: vid,
        b: bid,
        a: aid,
        a_coef: acid,
        dt_bias: dtid,
        state: sid,
        dst,
        rows: rows as u32,
        n_vhead: n_vhead as u32,
        n_khead: n_khead as u32,
        head_k: head_k as u32,
        head_v: head_v as u32,
        eps,
        src_stride: 0,
    });
    let plan = be.compile(&g).expect("compile");
    let up_f32 = |data: &[f32], usage| {
        let b = be.alloc(data.len() * 4, usage).expect("alloc f32");
        be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
        b
    };
    let up_bytes = |data: &[u8], usage| {
        let b = be.alloc(data.len(), usage).expect("alloc bytes");
        be.upload(b.as_ref(), data).unwrap();
        b
    };
    let qb = up_f32(q, BufferUsage::Activations);
    let kb = up_f32(k, BufferUsage::Activations);
    let vb = up_f32(v, BufferUsage::Activations);
    let bb = up_f32(bcoef, BufferUsage::Activations);
    let ab = up_f32(acoef_in, BufferUsage::Activations);
    let acb = up_bytes(a_coef_f16, BufferUsage::Weights);
    let dtb = up_bytes(dt_bias_f16, BufferUsage::Weights);
    let sb = up_f32(state_init, BufferUsage::Activations);
    let ob = be
        .alloc(rows * n_vhead * head_v * 4, BufferUsage::Readback)
        .expect("out");
    let mut bnd = Bindings::new();
    bnd.bind(qid, qb.as_ref());
    bnd.bind(kid, kb.as_ref());
    bnd.bind(vid, vb.as_ref());
    bnd.bind(bid, bb.as_ref());
    bnd.bind(aid, ab.as_ref());
    bnd.bind(acid, acb.as_ref());
    bnd.bind(dtid, dtb.as_ref());
    bnd.bind(sid, sb.as_ref());
    bnd.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &bnd).expect("execute");
    let mut out = vec![0f32; rows * n_vhead * head_v];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut out))
        .unwrap();
    let mut ns = vec![0f32; n_vhead * head_k * head_v];
    be.download(sb.as_ref(), bytemuck::cast_slice_mut(&mut ns))
        .unwrap();
    (out, ns)
}

/// `Op::DeltaNet` (qwen35's gated-DeltaNet linear-attention recurrence) must match the CPU reference
/// (`infr_cpu` `deltanet_scan`) for BOTH the output AND the mutated persistent `S` state, on a
/// MULTI-ROW (prefill) input with a NON-trivial initial `S` — the token recurrence carries `S`
/// sequentially across rows, so a per-row-independent or mis-sequenced kernel is wrong for rows>1.
/// Uses GQA (`n_khead < n_vhead`) so the value→key head mapping is exercised, and injects large
/// `a` values so the decay's softplus is pushed into its overflow regime.
///
/// The pre-fix ROCm kernel had four divergences any of which this catches: (1) the state was stored
/// TRANSPOSED (`S[d*head_k+k]`) so the mutated-state readback disagreed with the CPU `[head_k,
/// head_v]` layout; (2) GQA used `vh/(n_vhead/n_khead)` (grouped) instead of the CPU/qwen35
/// INTERLEAVED `vh % n_khead`, so every value head past the first group read the wrong q/k; (3) the
/// decay used the naive `log(1+exp(z))` softplus, which overflows to +inf for large z and (with
/// a_coef<0) collapses decay to 0, silently wiping the state every token; (4) `eps` was hardcoded.
/// It also runs a decode (rows==1) case. Fails loudly without the fix; guarded against vacuous
/// all-zero agreement.
#[test]
#[ignore = "requires a ROCm GPU"]
fn deltanet_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let eps = 1e-6f32;
    // Two shapes: a small GQA case (nk=2 < nv=4 — exercises the interleaved value→key head map),
    // and the REAL qwen35-0.8B DeltaNet shape (nv=nk=16, head_k=head_v=128) to rule out any
    // large-dim / long-reduction divergence at the size the model actually runs.
    for &(n_vhead, n_khead, head_k, head_v) in
        &[(4usize, 2usize, 8usize, 8usize), (16, 16, 128, 128)]
    {
        // Per-head scalars: a_coef modestly NEGATIVE (the sign that makes an overflowing softplus
        // collapse decay to zero), dt_bias small. F16 like the seam's dequant path.
        let acoef_f32: Vec<f32> = (0..n_vhead).map(|h| -0.02 * (1.0 + h as f32)).collect();
        let dtbias_f32: Vec<f32> = gen(n_vhead, 71);
        let a_coef_f16: Vec<u8> = acoef_f32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let dt_bias_f16: Vec<u8> = dtbias_f32
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();

        for &rows in &[5usize, 1usize] {
            let q = gen(rows * n_khead * head_k, 6);
            let k = gen(rows * n_khead * head_k, 9);
            let v = gen(rows * n_vhead * head_v, 12);
            let bcoef = gen(rows * n_vhead, 15);
            // `a`: mostly small, but a couple large-z entries so softplus(z) overflows the naive form
            // (z ~ 100 → exp(z) is +inf) while stable softplus stays finite (sp ≈ z, decay ≈ exp(ac·z)).
            let mut acoef_in = gen(rows * n_vhead, 18);
            acoef_in[0] = 100.0;
            if rows * n_vhead > n_vhead {
                acoef_in[n_vhead] = 100.0;
            }
            // NON-trivial initial S state (a zeroed S would hide the transposed-layout bug entirely).
            let state_init = gen(n_vhead * head_k * head_v, 21);

            let (c_out, c_state) = {
                // The CPU backend mutates `state` in place too — run it through the SAME single-op graph.
                run_deltanet(
                    &cpu,
                    &q,
                    &k,
                    &v,
                    &bcoef,
                    &acoef_in,
                    &a_coef_f16,
                    &dt_bias_f16,
                    &state_init,
                    rows,
                    n_vhead,
                    n_khead,
                    head_k,
                    head_v,
                    eps,
                )
            };
            // Fresh backend per case: `run_deltanet` allocates then frees its a_coef/dt_bias weight
            // buffers each call, and the dequant cache is keyed by (device-address, byte length).
            // Reusing one backend across the shapes/rows lets a freed address recycle into a
            // SAME-SIZE sibling weight (a_coef ↔ dt_bias are both `[n_vhead]` f16), aliasing the
            // stale dequant — the source of the historical rows=1 decode nondeterminism. A fresh
            // backend keeps each case to a single execute with no cross-execute buffer churn.
            let be = rocm().expect("rocm backend");
            let (r_out, r_state) = run_deltanet(
                &be,
                &q,
                &k,
                &v,
                &bcoef,
                &acoef_in,
                &a_coef_f16,
                &dt_bias_f16,
                &state_init,
                rows,
                n_vhead,
                n_khead,
                head_k,
                head_v,
                eps,
            );

            let eo = maxerr(&c_out, &r_out);
            let out_mag = maxabs(&c_out).max(1e-6);
            let es = maxerr(&c_state, &r_state);
            let st_mag = maxabs(&c_state).max(1e-6);
            println!(
            "DeltaNet nv={n_vhead} nk={n_khead} kd={head_k} vd={head_v} rows={rows} out max_err={eo:e} max|ref|={out_mag:e} rel={:e} | state max_err={es:e} max|ref|={st_mag:e} rel={:e}",
            eo / out_mag,
            es / st_mag
        );
            // Guard against a silently-zero output/state masquerading as agreement.
            assert!(
                out_mag > 1e-3,
                "DeltaNet rows={rows} output reference is all-zero — test is vacuous"
            );
            assert!(
                st_mag > 1e-3,
                "DeltaNet rows={rows} state reference is all-zero — test is vacuous"
            );
            assert!(
            eo / out_mag < 2e-2,
            "DeltaNet rows={rows} output diverges from CPU reference (GQA map / softplus / sequencing?): abs={eo:e} rel={:e}",
            eo / out_mag
        );
            assert!(
            es / st_mag < 2e-2,
            "DeltaNet rows={rows} mutated state diverges from CPU reference (transposed layout / decay?): abs={es:e} rel={:e}",
            es / st_mag
        );
        }
    }
}

/// The CHUNKED DeltaNet PREFILL kernel (rows>1) must match the CPU reference across CHUNK BOUNDARIES:
/// the chunked reformulation carries the recurrent state `S` sequentially at chunk (not token)
/// granularity, so a mis-carried `S₀`, a wrong partial-tail length, or an off-by-one in the inclusive
/// prefix log-decay only shows up once `rows` spans several chunks (DN_CHUNK = 16 on ROCm) plus a
/// partial. Runs the real qwen35-0.8B shape (nv=nk=16, kd=vd=128) at `rows` ∈ {130, 96, 33} — 130 =
/// 8 full chunks + a 2-token tail, 96 = 6 exact full chunks, 33 = 2 full chunks + a 1-token tail —
/// and asserts BOTH the output and the mutated `S` match `infr_cpu` `deltanet_scan` (which itself is
/// bit-identical to the naive serial recurrence). A FRESH backend per `rows` keeps each case to a
/// single execute, so the run never depends on cross-execute weight-buffer recycling.
#[test]
#[ignore = "requires a ROCm GPU"]
fn deltanet_prefill_chunk_boundary_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let eps = 1e-6f32;
    let (n_vhead, n_khead, head_k, head_v) = (16usize, 16usize, 128usize, 128usize);
    let acoef_f32: Vec<f32> = (0..n_vhead).map(|h| -0.02 * (1.0 + h as f32)).collect();
    let dtbias_f32: Vec<f32> = gen(n_vhead, 71);
    let a_coef_f16: Vec<u8> = acoef_f32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let dt_bias_f16: Vec<u8> = dtbias_f32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();

    for &rows in &[130usize, 96usize, 33usize] {
        // Fresh backend → exactly one execute on this device, no weight-buffer churn to alias.
        let be = rocm().expect("rocm backend");
        let q = gen(rows * n_khead * head_k, 6);
        let k = gen(rows * n_khead * head_k, 9);
        let v = gen(rows * n_vhead * head_v, 12);
        let bcoef = gen(rows * n_vhead, 15);
        // A couple of large-z entries so the decay's softplus is pushed into its overflow regime.
        let mut acoef_in = gen(rows * n_vhead, 18);
        acoef_in[0] = 100.0;
        acoef_in[n_vhead] = 100.0;
        let state_init = gen(n_vhead * head_k * head_v, 21);

        let (c_out, c_state) = run_deltanet(
            &cpu,
            &q,
            &k,
            &v,
            &bcoef,
            &acoef_in,
            &a_coef_f16,
            &dt_bias_f16,
            &state_init,
            rows,
            n_vhead,
            n_khead,
            head_k,
            head_v,
            eps,
        );
        let (r_out, r_state) = run_deltanet(
            &be,
            &q,
            &k,
            &v,
            &bcoef,
            &acoef_in,
            &a_coef_f16,
            &dt_bias_f16,
            &state_init,
            rows,
            n_vhead,
            n_khead,
            head_k,
            head_v,
            eps,
        );
        let eo = maxerr(&c_out, &r_out);
        let out_mag = maxabs(&c_out).max(1e-6);
        let es = maxerr(&c_state, &r_state);
        let st_mag = maxabs(&c_state).max(1e-6);
        println!(
            "DeltaNet chunked-prefill rows={rows} out max_err={eo:e} rel={:e} | state max_err={es:e} rel={:e}",
            eo / out_mag,
            es / st_mag
        );
        assert!(
            out_mag > 1e-3,
            "rows={rows} output reference all-zero — vacuous"
        );
        assert!(
            st_mag > 1e-3,
            "rows={rows} state reference all-zero — vacuous"
        );
        assert!(
            eo / out_mag < 2e-2,
            "chunked-prefill rows={rows} output diverges from CPU (chunk carry / prefix-decay?): rel={:e}",
            eo / out_mag
        );
        assert!(
            es / st_mag < 2e-2,
            "chunked-prefill rows={rows} state diverges from CPU (S₀ carry across chunks?): rel={:e}",
            es / st_mag
        );
    }
}

// ── GatedAct interleaved output gate (qwen35 attn_out_gate) vs CPU ────────────

/// The qwen35 attention output gate reads its per-head SIGMOID gate from the INTERLEAVED q+gate
/// projection `qg` (`[rows, nh*(2*hd)]`, each head a `[query(hd) | gate(hd)]` block) via
/// `gate_stride = nh*2*hd` / `gate_block_width = 2*hd`, and multiplies `sigmoid(gate)` into the
/// packed attention output `up` (`[rows, nh*hd]`). The bug (kernel used `gate_block_width` directly
/// as the head width instead of `gate_block_width/2`) read the WRONG half of each block, corrupting
/// the gate — the divergence that made qwen35 emit only `<think>` on ROCm. Single-op parity vs CPU.
#[test]
#[ignore = "requires a ROCm GPU"]
fn gated_act_interleaved_gate_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, hd) = (3usize, 4usize, 8usize);
    let nff = nh * hd; // packed output width
    let gate_w = nh * 2 * hd; // interleaved q+gate row width
    let qg = gen(rows * gate_w, 3);
    let up = gen(rows * nff, 8);
    let run = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let gid = g.input(f32d(rows * gate_w));
        let uid = g.input(f32d(rows * nff));
        let dst = g.output(f32d(rows * nff));
        g.push(Op::GatedAct {
            gate: gid,
            up: uid,
            dst,
            rows: rows as u32,
            nff: nff as u32,
            act: Activation::Sigmoid,
            up_off: 0,
            up_stride: 0,
            gate_stride: gate_w as u32,
            gate_block_width: (2 * hd) as u32,
        });
        let plan = b.compile(&g).expect("compile");
        let gb = b
            .alloc(qg.len() * 4, BufferUsage::Activations)
            .expect("gate");
        b.upload(gb.as_ref(), bytemuck::cast_slice(&qg)).unwrap();
        let ub = b.alloc(up.len() * 4, BufferUsage::Activations).expect("up");
        b.upload(ub.as_ref(), bytemuck::cast_slice(&up)).unwrap();
        let ob = b.alloc(rows * nff * 4, BufferUsage::Readback).expect("out");
        let mut bd = Bindings::new();
        bd.bind(gid, gb.as_ref());
        bd.bind(uid, ub.as_ref());
        bd.bind(dst, ob.as_ref());
        b.execute(plan.as_ref(), &bd).expect("execute");
        let mut o = vec![0f32; rows * nff];
        b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = run(&cpu);
    let r = run(&be);
    let e = maxerr(&c, &r);
    let mag = maxabs(&c).max(1e-3);
    println!("GatedAct interleaved gate max_err={e:e} max|ref|={mag:e}");
    assert!(mag > 1e-3, "GatedAct reference all-zero — test is vacuous");
    assert!(
        e / mag < 1e-3,
        "GatedAct interleaved gate diverges from CPU reference (wrong strided-gate index): abs={e:e}"
    );
}

// ── Copy / CopyStrided partial update into a pre-existing dst vs CPU ──────────

/// `Copy`/`CopyStrided` write only a slice/strided rows of `dst` and MUST preserve the rest — `dst`
/// is a real, full-extent tensor (the CPU reference copies into a pre-sized `vals[dst]`). The bug
/// re-allocated a fresh ZEROED, wrong-sized `dst` per call, dropping prior content and any strided
/// gap. Here op 1 fills `dst` with a pattern, then a strided op (`dst_stride > n`, leaving gaps)
/// overwrites some rows — the gaps must retain the pattern. Parity vs CPU.
#[test]
#[ignore = "requires a ROCm GPU"]
fn copy_strided_partial_update_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n, dst_stride) = (2usize, 2usize, 4usize);
    let numel = rows * dst_stride; // 8; strided rows [0,1],[4,5], gaps [2,3],[6,7]
    let pat = gen(numel, 2); // prior content that the gaps MUST preserve
    let src2 = gen(rows * n, 5); // strided source
    let run = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let pid = g.input(f32d(numel));
        let sid = g.input(f32d(rows * n));
        let dst = g.output(f32d(numel));
        // 1) fill dst with the prior pattern (full-extent Copy)
        g.push(Op::Copy {
            src: pid,
            src_off: 0,
            dst,
            dst_off: 0,
            n: numel as u32,
        });
        // 2) partial strided update — the gaps (dst_stride > n) must retain the pattern
        g.push(Op::CopyStrided {
            src: sid,
            src_off: 0,
            src_stride: n as u32,
            dst,
            dst_off: 0,
            dst_stride: dst_stride as u32,
            rows: rows as u32,
            n: n as u32,
        });
        let plan = b.compile(&g).expect("compile");
        let pb = b
            .alloc(pat.len() * 4, BufferUsage::Activations)
            .expect("pat");
        b.upload(pb.as_ref(), bytemuck::cast_slice(&pat)).unwrap();
        let sb = b
            .alloc(src2.len() * 4, BufferUsage::Activations)
            .expect("src");
        b.upload(sb.as_ref(), bytemuck::cast_slice(&src2)).unwrap();
        let ob = b.alloc(numel * 4, BufferUsage::Readback).expect("out");
        let mut bd = Bindings::new();
        bd.bind(pid, pb.as_ref());
        bd.bind(sid, sb.as_ref());
        bd.bind(dst, ob.as_ref());
        b.execute(plan.as_ref(), &bd).expect("execute");
        let mut o = vec![0f32; numel];
        b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = run(&cpu);
    let r = run(&be);
    // Vacuity: the reference must actually exercise BOTH a preserved gap and an overwritten row.
    assert_eq!(
        c[2], pat[2],
        "reference gap not preserved — test setup wrong"
    );
    assert_eq!(
        c[0], src2[0],
        "reference strided row not written — test setup wrong"
    );
    let e = maxerr(&c, &r);
    println!("Copy/CopyStrided partial-update max_err={e:e}");
    assert!(
        e < 1e-6,
        "Copy/CopyStrided partial update diverges from CPU reference (lost prior dst content): {e:e}"
    );
}

// ── Conv1dSilu rolling-state update across a REUSED x tensor vs CPU ───────────

/// Two `Conv1dSilu` ops in one graph share the SAME `x` handle, which is rewritten between them
/// (mirrors the seam's per-DeltaNet-layer `dn_qkvbuf` reuse). The host-side rolling-state update
/// must read `x`'s CURRENT device content — the bug read it through a per-tensor-id host cache, so
/// the second conv's state update saw the FIRST conv's stale `x`, corrupting the carried conv
/// history for every DeltaNet layer past the first (qwen35 decoded one token then stalled). Compare
/// the second op's mutated state to CPU.
#[test]
#[ignore = "requires a ROCm GPU"]
fn conv1d_reused_x_state_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, channels, kernel) = (3usize, 8usize, 4usize);
    let km1 = kernel - 1;
    let wf32 = gen(channels * kernel, 7);
    let weight_f16: Vec<u8> = wf32
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let pat1 = gen(rows * channels, 2);
    let pat2 = gen(rows * channels, 9); // the DIFFERENT content the 2nd conv must actually see
    let s1_init = gen(km1 * channels, 11);
    let s2_init = gen(km1 * channels, 13);
    // Returns the SECOND conv's mutated state (the one the stale-cache bug corrupts).
    let run = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let p1 = g.input(f32d(rows * channels));
        let p2 = g.input(f32d(rows * channels));
        let x = g.internal(f32d(rows * channels)); // shared, rewritten between the two convs
        let wid = g.weight(TensorDesc::new(vec![channels * kernel], DType::F16));
        let s1 = g.input(f32d(km1 * channels));
        let s2 = g.input(f32d(km1 * channels));
        let d1 = g.output(f32d(rows * channels));
        let d2 = g.output(f32d(rows * channels));
        let conv = |xh, sh, dh| Op::Conv1dSilu {
            x: xh,
            weight: wid,
            state: sh,
            dst: dh,
            rows: rows as u32,
            channels: channels as u32,
            kernel: kernel as u32,
        };
        g.push(Op::Copy {
            src: p1,
            src_off: 0,
            dst: x,
            dst_off: 0,
            n: (rows * channels) as u32,
        });
        g.push(conv(x, s1, d1));
        g.push(Op::Copy {
            src: p2,
            src_off: 0,
            dst: x,
            dst_off: 0,
            n: (rows * channels) as u32,
        });
        g.push(conv(x, s2, d2));
        let plan = b.compile(&g).expect("compile");
        let up = |data: &[f32], usage| {
            let buf = b.alloc(data.len() * 4, usage).expect("alloc");
            b.upload(buf.as_ref(), bytemuck::cast_slice(data)).unwrap();
            buf
        };
        let p1b = up(&pat1, BufferUsage::Activations);
        let p2b = up(&pat2, BufferUsage::Activations);
        let wb = b.alloc(weight_f16.len(), BufferUsage::Weights).expect("w");
        b.upload(wb.as_ref(), &weight_f16).unwrap();
        let s1b = up(&s1_init, BufferUsage::Activations);
        let s2b = up(&s2_init, BufferUsage::Activations);
        let d1b = b
            .alloc(rows * channels * 4, BufferUsage::Readback)
            .expect("d1");
        let d2b = b
            .alloc(rows * channels * 4, BufferUsage::Readback)
            .expect("d2");
        let mut bd = Bindings::new();
        bd.bind(p1, p1b.as_ref());
        bd.bind(p2, p2b.as_ref());
        bd.bind(wid, wb.as_ref());
        bd.bind(s1, s1b.as_ref());
        bd.bind(s2, s2b.as_ref());
        bd.bind(d1, d1b.as_ref());
        bd.bind(d2, d2b.as_ref());
        b.execute(plan.as_ref(), &bd).expect("execute");
        let mut ns2 = vec![0f32; km1 * channels];
        b.download(s2b.as_ref(), bytemuck::cast_slice_mut(&mut ns2))
            .unwrap();
        ns2
    };
    let c = run(&cpu);
    let r = run(&be);
    // Vacuity: the 2nd conv's state MUST differ from its init (it rolled in pat2's tail).
    assert!(
        maxerr(&c, &s2_init) > 1e-3,
        "reference 2nd-conv state unchanged — test is vacuous"
    );
    let e = maxerr(&c, &r);
    println!(
        "Conv1dSilu reused-x state max_err={e:e} max|ref|={:e}",
        maxabs(&c)
    );
    assert!(
        e < 1e-3,
        "Conv1dSilu reused-x state diverges from CPU reference (stale host cache of x): {e:e}"
    );
}

// ── AddBias (qwen2/2.5 QKV projection bias) vs CPU ───────────────────────────

/// `Op::AddBias` (`dst[r, j] = x[r, j] + bias[j]`) is the qwen2/qwen2.5 QKV-projection bias add —
/// the op that distinguishes the qwen2 attention block from the bias-free qwen3/llama path. The
/// `bias` is a bound Weight (qwen2 ships it F32); the ROCm `add_bias` kernel and the CPU reference
/// both read it as f32, so parity is bit-exact. Multi-row (prefill) input so the per-row broadcast
/// of the shared bias vector is exercised. Single-op parity vs `infr_cpu::CpuBackend`, vacuity
/// guarded against a silently-zero output.
#[test]
#[ignore = "requires a ROCm GPU"]
fn add_bias_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n) = (4usize, 96usize); // rows > 1 → per-row broadcast of the shared bias
    let x = gen(rows * n, 3);
    let bias_f32 = gen(n, 19);
    let bias_bytes: &[u8] = bytemuck::cast_slice(&bias_f32);
    let run = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let xid = g.input(f32d(rows * n));
        let bid = g.weight(TensorDesc::new(vec![n], DType::F32)); // qwen2 bias is a bound F32 weight
        let dst = g.output(f32d(rows * n));
        g.push(Op::AddBias {
            x: xid,
            bias: bid,
            dst,
            rows: rows as u32,
            n: n as u32,
        });
        let plan = b.compile(&g).expect("compile");
        let xb = b.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
        b.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let bb = b
            .alloc(bias_bytes.len(), BufferUsage::Weights)
            .expect("bias");
        b.upload(bb.as_ref(), bias_bytes).unwrap();
        let ob = b.alloc(rows * n * 4, BufferUsage::Readback).expect("out");
        let mut bd = Bindings::new();
        bd.bind(xid, xb.as_ref());
        bd.bind(bid, bb.as_ref());
        bd.bind(dst, ob.as_ref());
        b.execute(plan.as_ref(), &bd).expect("execute");
        let mut o = vec![0f32; rows * n];
        b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = run(&cpu);
    let r = run(&be);
    let e = maxerr(&c, &r);
    let mag = maxabs(&c).max(1e-6);
    println!("AddBias max_err={e:e} max|ref|={mag:e}");
    assert!(mag > 1e-3, "AddBias reference all-zero — test is vacuous");
    assert!(
        e < 1e-5,
        "AddBias diverges from CPU reference (bias broadcast / dtype): {e:e}"
    );
}

// ── Softcap (gemma4 attn-logit / final-logit soft cap) vs CPU ────────────────

/// `Op::Softcap` (`dst[i] = cap * tanh(x[i] / cap)`) is the gemma-family logit soft-cap applied to
/// attention scores and final logits — a gemma4-distinctive op absent from the qwen3/llama path. The
/// input spans both the linear regime (|x| ≪ cap) and the saturating tail (|x| ≫ cap) so the tanh
/// curvature is exercised, not just the identity middle. The ROCm `softcap` kernel uses `tanhf` and
/// the CPU reference uses `f32::tanh`, both in f32, so parity is tight. Single-op parity vs
/// `infr_cpu::CpuBackend`, vacuity guarded.
#[test]
#[ignore = "requires a ROCm GPU"]
fn softcap_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let n = 512usize;
    let cap = 30.0f32; // gemma's attn-logit softcap magnitude
                       // Values spanning [-4*cap, 4*cap]: linear near 0, saturating in the tails.
    let x: Vec<f32> = (0..n)
        .map(|i| ((i as f32 / n as f32) - 0.5) * 8.0 * cap)
        .collect();
    let run = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let xid = g.input(f32d(n));
        let dst = g.output(f32d(n));
        g.push(Op::Softcap {
            x: xid,
            dst,
            cap,
            n: n as u32,
        });
        let plan = b.compile(&g).expect("compile");
        let xb = b.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
        b.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ob = b.alloc(n * 4, BufferUsage::Readback).expect("out");
        let mut bd = Bindings::new();
        bd.bind(xid, xb.as_ref());
        bd.bind(dst, ob.as_ref());
        b.execute(plan.as_ref(), &bd).expect("execute");
        let mut o = vec![0f32; n];
        b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = run(&cpu);
    let r = run(&be);
    let e = maxerr(&c, &r);
    let mag = maxabs(&c).max(1e-6);
    println!("Softcap cap={cap} max_err={e:e} max|ref|={mag:e}");
    assert!(mag > 1e-3, "Softcap reference all-zero — test is vacuous");
    // The saturating tail must actually be reached (|out| approaches cap), else the test is
    // exercising only the near-identity middle.
    assert!(
        mag > 0.9 * cap,
        "Softcap did not reach the saturating tail — test is under-exercised"
    );
    assert!(
        e < 1e-3,
        "Softcap diverges from CPU reference (wrong cap formula / tanh): {e:e}"
    );
}

// ── All-weight-quant Linear parity sweep (Slice 10, docs/rocm-plan.md Part A) ─────────────────────
//
// The ROCm Linear path handles each weight quant one of two ways: Q4_K/Q6_K/Q8_0 are decoded
// IN-KERNEL from their raw bytes (Phase 3 native decode, `native_decode_fmt`), every other quant is
// dequantized to f32 via the shared `infr_gguf::dequant::dequant_block`, rounded to f16, then run
// through the f32-accumulating `linear_f16` GEMV (kernels.rs). Both paths round the weight to f16 and
// accumulate in f32, so every weight quant format is supported by construction — the only per-format
// risk is a bad block-byte assumption or an odd block stride. This sweep proves each of the 24 real
// WEIGHT quant formats decodes and GEMVs in agreement with `infr_cpu::CpuBackend` running the SAME
// one-op graph (CPU dequants the same bytes with `dequant_block` + an f32 matmul). Because both
// backends share the SAME decoder, the ONLY error source is the ROCm side's f16 weight rounding
// (the GEMV accumulates in f32), so tolerances are the ~2e-2 rel bound the Q4_K test uses, tightened
// per format where the f16 rounding lands well inside it.
//
// EXCLUDED (not weight quants): F32/F16/Bf16/I32/U32 (dense, covered by `linear_f16_matches_cpu`);
// I2S (BitNet i2_s — host-converted to f16 at weight load, never reaches a backend as I2S, so ROCm
// only ever sees f16; validated end-to-end in the plan's BitNet run, not here); Turbo2/3/4 (KV-cache
// -only formats, never GGUF weights).

/// Deterministic LCG byte stream — an arbitrary but reproducible payload for quant code/nibble
/// fields, which decode to FINITE values for ANY byte pattern (only the f16 scale slots must be
/// sane). Ported from the Metal parity suite's `lcg_bytes`.
fn lcg_bytes(mut seed: u32, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect()
}

/// Synthesize a valid block-quantized weight of `n_elem` elements for a format whose only
/// "must be finite" fields are one or two f16 scale slots at fixed block offsets: LCG-random code
/// bytes (finite-decoding for any pattern) with each `(offset, value)` in `scales` written as a
/// little-endian f16. Covers 21 of the 24 formats; MXFP4/NVFP4/IQ1_M have non-f16 scale encodings
/// and get bespoke builders below. Block byte layouts cross-checked against
/// `infr_gguf::block_layout` and the `dequant_block` decoders.
fn synth_q(
    n_elem: usize,
    block_elems: usize,
    bpb: usize,
    seed: u32,
    scales: &[(usize, f32)],
) -> Vec<u8> {
    assert_eq!(
        n_elem % block_elems,
        0,
        "n_elem not a multiple of block size"
    );
    let mut out = Vec::with_capacity((n_elem / block_elems) * bpb);
    for blk_i in 0..(n_elem / block_elems) {
        let mut blk = lcg_bytes(seed ^ blk_i as u32, bpb);
        for &(off, v) in scales {
            blk[off..off + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        out.extend_from_slice(&blk);
    }
    out
}

/// MXFP4 (32e / 17B): `[u8 E8M0 exponent][16B nibbles]`. The E8M0 byte is a shared exponent
/// `d = 2^(e-127)`; keep `e ∈ {124..=132}` so `d` stays in `2^-3..2^5` — decoded products stay well
/// inside f32 while still exercising the E8M0 decode across a band. Nibbles LCG (KVALUES_MXFP4).
fn synth_mxfp4(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 32, 0, "MXFP4 blocks are 32 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = lcg_bytes(seed ^ blk_i as u32, 17);
        blk[0] = 124 + (blk_i % 9) as u8; // E8M0 exponent, moderate band
        out.extend_from_slice(&blk);
    }
    out
}

/// NVFP4 (64e / 36B): `[u8 UE4M3 sub-scale[4]][32B nibbles]`. The four bytes are per-16-element
/// UE4M3 scales; 0x3A/0x3C/0x3E/0x40 decode to 0.625/0.75/0.875/1.0 (moderate, none the zero-flush
/// corners), exercising four distinct sub-block scales. Nibbles LCG (shared KVALUES_MXFP4).
fn synth_nvfp4(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 64, 0, "NVFP4 blocks are 64 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 64) {
        let mut blk = lcg_bytes(seed ^ blk_i as u32, 36);
        blk[0..4].copy_from_slice(&[0x3A, 0x3C, 0x3E, 0x40]);
        out.extend_from_slice(&blk);
    }
    out
}

/// IQ1_M (256e / 56B): `[32B qs][16B qh][8B scales]` with NO separate `d` — `d` is a f16 assembled
/// from the TOP nibbles of the four u16 scale words, so random scale bytes would yield a garbage/NaN
/// `d`. Set `d` deliberately (its four nibbles → the four scale-word top nibbles, bits 12..15); the
/// low 12 bits (four 3-bit `dl` fields) plus all qs/qh (11-bit grid index + delta sign) are LCG.
/// Ported from the Metal parity suite's `synth_iq1m`.
fn synth_iq1m(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0, "IQ1_M blocks are 256 elems");
    let d_bits = half::f16::from_f32(0.03).to_bits();
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 56];
        blk[0..48].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 48)); // qs + qh
        let low = lcg_bytes(seed.wrapping_add(0x9e37).wrapping_add(blk_i as u32), 8);
        for i in 0..4usize {
            let nib = (d_bits >> (4 * i)) & 0xf;
            let lo12 = ((low[2 * i] as u16) | ((low[2 * i + 1] as u16) << 8)) & 0x0fff;
            let scw = (nib << 12) | lo12;
            blk[48 + 2 * i..48 + 2 * i + 2].copy_from_slice(&scw.to_le_bytes());
        }
        out.extend_from_slice(&blk);
    }
    out
}

/// Sweep EVERY real weight quant format (24) through a one-`Op::Linear` graph on `RocmBackend` vs
/// `infr_cpu::CpuBackend` and assert per-format parity. See the module comment above for what this
/// covers and why the tolerance is the f16-weight-rounding bound.
#[test]
#[ignore = "requires a ROCm GPU"]
fn all_quant_linear_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    // n = out_f*in_f = 2048 is divisible by every block size (32 / 64 / 256), so one dimension set
    // covers all formats. m=2 exercises the multi-row GEMV.
    let (m, in_f, out_f) = (2usize, 256usize, 8usize);
    let n = out_f * in_f;
    let x = gen(m * in_f, 5);

    // (dtype, weight bytes, rel tol, label). Block layouts / byte offsets from `block_layout` and the
    // `dequant_block` decoders; the f16 `d` (and dmin/m) magnitudes mirror the Metal parity synths so
    // synthetic weight magnitudes stay realistic (esp. the signed-codebook i-quants that cancel).
    #[rustfmt::skip]
    let cases: Vec<(DType, Vec<u8>, f32, &str)> = vec![
        // ── legacy round quants ──
        (DType::Q4_0, synth_q(n, 32, 18, 201, &[(0, 0.04)]),              6e-3, "Q4_0"),
        (DType::Q4_1, synth_q(n, 32, 20, 202, &[(0, 0.04), (2, -0.30)]), 6e-3, "Q4_1"),
        (DType::Q5_0, synth_q(n, 32, 22, 203, &[(0, 0.04)]),              6e-3, "Q5_0"),
        (DType::Q5_1, synth_q(n, 32, 24, 204, &[(0, 0.04), (2, -0.30)]), 6e-3, "Q5_1"),
        (DType::Q8_0, synth_q(n, 32, 34, 205, &[(0, 0.01)]),              1e-2, "Q8_0"),
        // ── k-quants (d/dmin/scale offsets differ per format; Q2_K's scales sit at the block TAIL) ──
        (DType::Q2K, synth_q(n, 256, 84, 206, &[(80, 0.05), (82, 0.10)]), 2e-2, "Q2_K"),
        (DType::Q3K, synth_q(n, 256, 110, 207, &[(108, 0.03)]),           2e-2, "Q3_K"),
        (DType::Q4K, synth_q(n, 256, 144, 208, &[(0, 0.05), (2, 0.10)]),  2e-2, "Q4_K"),
        (DType::Q5K, synth_q(n, 256, 176, 209, &[(0, 0.05), (2, 0.10)]),  2e-2, "Q5_K"),
        (DType::Q6K, synth_q(n, 256, 210, 210, &[(208, 0.03)]),           2e-2, "Q6_K"),
        // ── i-quants (codebook / grid): signed values cancel in the dot, so keep the ~2e-2 bound ──
        (DType::Iq4Nl,  synth_q(n, 32, 18, 211, &[(0, 0.004)]),  6e-3, "IQ4_NL"),
        (DType::Iq4Xs,  synth_q(n, 256, 136, 212, &[(0, 0.06)]), 2e-2, "IQ4_XS"),
        (DType::Iq2Xxs, synth_q(n, 256, 66, 213, &[(0, 0.015)]), 2e-2, "IQ2_XXS"),
        (DType::Iq2Xs,  synth_q(n, 256, 74, 214, &[(0, 0.015)]), 2e-2, "IQ2_XS"),
        (DType::Iq2S,   synth_q(n, 256, 82, 215, &[(0, 0.015)]), 2e-2, "IQ2_S"),
        (DType::Iq3Xxs, synth_q(n, 256, 98, 216, &[(0, 0.008)]), 2e-2, "IQ3_XXS"),
        (DType::Iq3S,   synth_q(n, 256, 110, 217, &[(0, 0.002)]), 2e-2, "IQ3_S"),
        (DType::Iq1S,   synth_q(n, 256, 50, 218, &[(0, 0.03)]),  2e-2, "IQ1_S"),
        (DType::Iq1M,   synth_iq1m(n, 219),                      2e-2, "IQ1_M"),
        // ── ternary quants (d at block TAIL for TQ*, head for Q2_0) ──
        (DType::Tq1_0, synth_q(n, 256, 54, 220, &[(52, 0.05)]), 2e-2, "TQ1_0"),
        (DType::Tq2_0, synth_q(n, 256, 66, 221, &[(64, 0.05)]), 2e-2, "TQ2_0"),
        (DType::Q2_0,  synth_q(n, 64, 18, 222, &[(0, 0.05)]),   6e-3, "Q2_0"),
        // ── fp4 quants (non-f16 scale encodings) ──
        (DType::Mxfp4, synth_mxfp4(n, 223), 2e-2, "MXFP4"),
        (DType::Nvfp4, synth_nvfp4(n, 224), 2e-2, "NVFP4"),
    ];

    let mut failures = Vec::new();
    for (dt, wbytes, tol, label) in cases {
        // Fresh ROCm backend per format: `dequant_weight_or_cache` keys the dequantized-weight cache
        // by the weight's raw device pointer, and a weight buffer freed at the end of one case can
        // have its VRAM address recycled by the next — a stale cache hit would feed the previous
        // format's dequantized rows. (The same hazard the embed_gather test documents.)
        let be = rocm().unwrap();
        let c = run_linear(&cpu, &x, &wbytes, dt, m, in_f, out_f);
        let r = run_linear(&be, &x, &wbytes, dt, m, in_f, out_f);
        let e = maxerr(&c, &r);
        let ref_mag = maxabs(&c).max(1e-3);
        let rel = e / ref_mag;
        println!("Linear[{label:7}] max_err={e:e} max|ref|={ref_mag:e} rel={rel:e} tol={tol:e}");
        // Vacuity: a silently-zero output must not masquerade as agreement.
        assert!(
            ref_mag > 1e-3,
            "Linear[{label}] reference is all-zero — test is vacuous"
        );
        if rel >= tol {
            failures.push(format!("{label}: rel={rel:e} >= tol={tol:e} (abs={e:e})"));
        }
    }
    assert!(
        failures.is_empty(),
        "weight-quant Linear parity failures:\n  {}",
        failures.join("\n  ")
    );
}

// ── Native in-kernel quant-decode GEMV / EmbedGather (Slice 12, Phase 3) ──────────────────────────
//
// Q4_K / Q6_K / Q8_0 route through the NATIVE in-kernel decode path (`native_decode_fmt` →
// `linear_q4k`/`linear_q6k`/`linear_q80` and `embed_q4k`/`embed_q6k`/`embed_q80`): the GEMV reads the
// RAW quant bytes and decodes each block on the fly, so no f16 cache is materialized in VRAM. The
// decode is bit-faithful to the old dequant→f16 cache (same `sc*code + mn`, contract-off, then round
// to f16), so parity vs `infr_cpu::CpuBackend` (which dequants the same bytes with `dequant_block`)
// holds within the f16-weight-rounding tolerance — exactly as the cached path did. `linear_q4k`/the
// Q4_K `embed_gather` and the `all_quant_linear` sweep already exercise these three under the native
// router; the tests below add the explicit per-format Q6_K/Q8_0 coverage the plan calls for.

/// Q6_K weight through the native `linear_q6k` in-kernel decode GEMV vs the CPU reference.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_q6k_native_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // Q6_K super-block = 256 elems / 210 bytes; in_f a multiple of 256.
    let (m, in_f, out_f) = (2usize, 256usize, 4usize);
    let n = out_f * in_f;
    // Valid Q6_K blocks: LCG code/scale bytes (finite for any pattern), f16 `d` at byte 208 set sane.
    let w_bytes = synth_q(n, 256, 210, 310, &[(208, 0.03)]);
    let x = gen(m * in_f, 5);
    let c = run_linear(&cpu, &x, &w_bytes, DType::Q6K, m, in_f, out_f);
    let r = run_linear(&be, &x, &w_bytes, DType::Q6K, m, in_f, out_f);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "Linear Q6_K (native) max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "Q6_K reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 2e-2,
        "Linear Q6_K native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q8_0 weight through the native `linear_q80` in-kernel decode GEMV vs the CPU reference. Q8_0 is
/// near-lossless (int8 blocks), so the f16-weight-rounding tolerance is tight.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_q80_native_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // Q8_0 block = 32 elems / 34 bytes; in_f a multiple of 32.
    let (m, in_f, out_f) = (2usize, 256usize, 4usize);
    let n = out_f * in_f;
    let w_bytes = synth_q(n, 32, 34, 311, &[(0, 0.02)]);
    let x = gen(m * in_f, 5);
    let c = run_linear(&cpu, &x, &w_bytes, DType::Q8_0, m, in_f, out_f);
    let r = run_linear(&be, &x, &w_bytes, DType::Q8_0, m, in_f, out_f);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "Linear Q8_0 (native) max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "Q8_0 reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 1e-2,
        "Linear Q8_0 native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Shared native-EmbedGather parity check (R3): `embed_*` is the ONE path that reaches `deq_*`
/// ELEMENT BY ELEMENT — every GEMV tier goes through int8 codes, where a dot product averages a
/// decode slip away. So this is where a format's layout is pinned bit for bit, at the f16-rounding
/// tolerance (`sc*code + mn` rounded once) rather than a dot's error-averaging one.
fn check_embed_native(w_bytes_for: impl Fn(usize) -> Vec<u8>, dt: DType, qpb: usize, label: &str) {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];
    let (vocab, ne) = (6usize, 256usize); // ne a multiple of every block size; vocab > max(ids)
    let scale = (ne as f32).sqrt(); // non-1.0 (Gemma-style) — must be applied on-device
    let t_bytes = w_bytes_for(vocab * ne / qpb);
    let c = run_embed_gather(&cpu, &ids, &t_bytes, dt, vocab, ne, scale);
    let r = run_embed_gather(&be, &ids, &t_bytes, dt, vocab, ne, scale);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "EmbedGather {label} (native) scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "EmbedGather {label} reference is all-zero — test is vacuous"
    );
    // Bit-faithful decode: the only loss is the f16 round. A swapped nibble half, a min read from
    // the wrong offset, or a dropped `qh` bit lands at O(1) relative — far outside this.
    assert!(
        e / ref_mag < 2e-3,
        "EmbedGather {label} native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q4_0 embedding table through the native `embed_q40` decode gather (R3): pins the nibble split
/// (low nibbles are elements 0..15, high nibbles 16..31) and the constant `d·(−8)` min.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q40_native_matches_cpu() {
    check_embed_native(q40_blocks, DType::Q4_0, 32, "Q4_0");
}

/// Q4_1 embedding table through `embed_q41` (R3): the AFFINE min — the oracle's `dd = (d, m)` with
/// multipliers `(1, 1)`, so the decoded value is `d·code + m`. The alternating-sign `m` in
/// `q41_blocks` means a min read from the neighbouring block also lands at O(1) relative.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q41_native_matches_cpu() {
    check_embed_native(q41_blocks, DType::Q4_1, 32, "Q4_1");
}

/// Q5_1 embedding table through `embed_q51` (R3): affine min + the `qh` 5th bit at header offset 4
/// (so `qs` starts at 8) — the two places a Q5_0-derived port goes wrong.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q51_native_matches_cpu() {
    check_embed_native(q51_blocks, DType::Q5_1, 32, "Q5_1");
}

/// IQ4_NL embedding table through `embed_iq4nl` (R4). For a CODEBOOK format this element-wise path
/// is doubly load-bearing: a wrong table index still produces a plausible-magnitude weight (the
/// table spans −127..113), so it can hide inside a dot's error averaging — but not here, where every
/// element is compared on its own.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq4nl_native_matches_cpu() {
    check_embed_native(iq4nl_blocks, DType::Iq4Nl, 32, "IQ4_NL");
}

/// IQ4_XS embedding table through `embed_iq4xs` (R4): the same codebook plus the 6-bit `ls − 32`
/// sub-block scale split across `scales_h`/`scales_l`, pinned element by element.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq4xs_native_matches_cpu() {
    check_embed_native(iq4xs_blocks, DType::Iq4Xs, 256, "IQ4_XS");
}

// ── R5 grid quants through the element-wise `embed_*` decode gather ──────────
//
// This is THE load-bearing case for a grid format, more so than for any format before it, and the
// only one in the suite that compares decoded elements individually rather than through a dot.
// Two whole classes of bug are invisible to the GEMV/WMMA tolerances and caught only here:
//
//   * a wrong GRID INDEX — every entry of an IQ2/IQ3 grid is a plausible-magnitude weight vector,
//     so reading entry 137 instead of 138 produces a perfectly reasonable-looking dot, and a
//     256-term sum with cancellation buries the difference well inside 1.5e-2;
//   * a flipped or misaligned SIGN BIT — the sign field is packed separately from the index (in
//     `ksigns` for IQ2_XXS/IQ2_XS/IQ3_XXS, in raw bytes for IQ2_S/IQ3_S), and one wrong bit in
//     eight changes a dot by ~2/8 of one term's magnitude, again inside the bound.
//
// Element by element, both land at O(1) relative. `check_embed_native` also runs the decode
// through `deq_*` rather than `wdec_*`, so these are the cases that pin the two decoders (the
// per-element one and the per-32-block one) against each other and against the host oracle.

/// IQ2_XXS embedding table through `embed_iq2xxs` (R5).
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq2xxs_native_matches_cpu() {
    check_embed_native(iq2xxs_blocks, DType::Iq2Xxs, 256, "IQ2_XXS");
}

/// IQ2_XS embedding table through `embed_iq2xs` (R5) — 9-bit index + 7-bit sign index in one u16,
/// and the per-16 scale nibble split, all pinned per element.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq2xs_native_matches_cpu() {
    check_embed_native(iq2xs_blocks, DType::Iq2Xs, 256, "IQ2_XS");
}

/// IQ2_S embedding table through `embed_iq2s` (R5) — the 2 high index bits from `qh` at shift
/// `8−2l` and the raw sign byte at `qs[32 + …]`, the two fields most easily read from the wrong
/// offset.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq2s_native_matches_cpu() {
    check_embed_native(iq2s_blocks, DType::Iq2S, 256, "IQ2_S");
}

/// IQ3_XXS embedding table through `embed_iq3xxs` (R5) — the two-grid-entries-per-group split, in
/// which elements 4..7 take grid entry 2 but sign bits 4..7 (not 0..3).
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq3xxs_native_matches_cpu() {
    check_embed_native(iq3xxs_blocks, DType::Iq3Xxs, 256, "IQ3_XXS");
}

/// IQ3_S embedding table through `embed_iq3s` (R5) — the asymmetric `qh` shifts (`8−2l` for the
/// first grid entry of a group, `7−2l` for the second). Swapping them moves at most one bit of one
/// index per group, which only an element-wise comparison sees.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq3s_native_matches_cpu() {
    check_embed_native(iq3s_blocks, DType::Iq3S, 256, "IQ3_S");
}

// ── R6 IQ1 + ternary quants through the element-wise `embed_*` decode gather ─
//
// The load-bearing case for R6 exactly as it was for R5, and for the same structural reason: this
// is the ONLY comparison in the suite that looks at decoded elements individually rather than
// through a dot, so it is the only one that can see a fault whose effect is smaller than the int8
// tolerance once 256 terms have partially cancelled. What it catches here:
//
//   * a wrong GRID INDEX (IQ1_S/IQ1_M) — every entry of the 2048-entry IQ1 grid is a plausible
//     ±1/0 vector, so reading entry 137 instead of 138 gives a perfectly reasonable-looking dot;
//   * a mis-scaled or wrong-signed DELTA — ±0.125 against a grid value of at most 1 is up to an
//     eighth of one term, invisible in a dot and glaring per element. This is the field R6 adds
//     that nothing before it has, so it is precisely what these two cases exist for;
//   * a wrong TERNARY LEVEL — TQ1_0's base-3 digit is the one decode in the covered set that is not
//     a shift-and-mask, and an off-by-one `n` or a non-wrapping multiply picks a different digit
//     for a minority of elements only;
//   * Q2_0's 64-element block boundary — a `>>3`-vs-`>>1` slip reads a neighbouring block's `d`.
//
// `check_embed_native` also drives the decode through `deq_*` rather than `wdec_*`, so these are
// the cases that pin the per-element and per-32-block decoders against each other AND against the
// host oracle — for R6 that includes checking the ×8 delta fold in `wdec_*` really does reproduce
// `deq_*`'s `dl·(gv + delta)`.

/// IQ1_S embedding table through `embed_iq1s` (R6) — the 11-bit index (8 bits from `qs`, 3 more
/// from `qh` at shift `3l`), the `(qh>>12)&7` sub-scale and the `0x8000` delta sign, per element.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq1s_native_matches_cpu() {
    check_embed_native(iq1s_blocks, DType::Iq1S, 256, "IQ1_S");
}

/// IQ1_M embedding table through `embed_iq1m` (R6) — the split `d` reassembled from four scale-word
/// nibbles, the per-16 `dl1`/`dl2` at shift `6·(ib&1)` (+3), and the per-8 index/delta pair whose
/// shifts alternate 8/4 and whose delta bits alternate `0x08`/`0x80` within one `qh` byte. Every
/// one of those alternations is a one-bit choice that only an element-wise comparison sees.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_iq1m_native_matches_cpu() {
    check_embed_native(iq1m_blocks, DType::Iq1M, 256, "IQ1_M");
}

/// TQ1_0 embedding table through `embed_tq10` (R6) — the three-segment base-3 walk, element by
/// element, which is the only way to see that segment 2 (`qs[32..48]`, 16 wide) and segment 3
/// (`qh`, 4 wide) use different strides from segment 1.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_tq10_native_matches_cpu() {
    check_embed_native(tq10_blocks, DType::Tq1_0, 256, "TQ1_0");
}

/// TQ2_0 embedding table through `embed_tq20` (R6) — the chunk/shift/byte decomposition of the
/// element index (`p>>7`, `(p>>5)&3`, `p&31`), which is NOT the obvious `p/4`, `p%4` one.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_tq20_native_matches_cpu() {
    check_embed_native(tq20_blocks, DType::Tq2_0, 256, "TQ2_0");
}

/// Q2_0 embedding table through `embed_q20` (R6) — 64-element blocks, so this also pins that the
/// gather's element→block division is `i>>6` and not the `i>>8` every other 2-bit format uses.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q20_native_matches_cpu() {
    check_embed_native(q20_blocks, DType::Q2_0, 64, "Q2_0");
}

// ── R7 fp4 quants through the element-wise `embed_*` decode gather ───────────
//
// THE load-bearing case for this slice, and more sharply so than for any before it. R7's new thing
// is a scale ENCODING, and a mis-decoded exponent is a FACTOR-OF-TWO error — a wrong `e8m0_half`
// case boundary, a dropped `·0.5` in `ue4m3`, an off-by-one exponent bias. Those are not small, but
// at m=1 they are also not necessarily visible: the int8 GEMV's tolerance is set against a 256-deep
// dot with partial cancellation, and a coarse relative bound can swallow a factor of two on a
// subset of blocks. Here every element is compared on its own, against the host oracle's own
// `e8m0_to_fp32_half` / `ue4m3_to_fp32`, so a wrong power of two has nowhere to hide.
//
// These are also the cases that pin `deq_*` against `wdec_*`: the two decoders derive the same
// scale by different routes (per element vs per 32-block half), and only this path runs `deq_*`.

/// MXFP4 embedding table through `embed_mxfp4` (R7) — the E8M0 exponent byte at offset 0 (where
/// every other 32-block format has an f16 `d`), the 17-byte stride, and the 16-wide nibble split.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_mxfp4_native_matches_cpu() {
    check_embed_native(mxfp4_blocks, DType::Mxfp4, 32, "MXFP4");
}

/// NVFP4 embedding table through `embed_nvfp4` (R7) — the four UE4M3 sub-block scales (including
/// the 0x00/0x7F holes and the subnormal `e == 0` branch), the 64-element block, and the 8-WIDE
/// nibble split, which is the one place NVFP4 departs from every other nibble format in the file:
/// within a 16-element sub-block the low nibbles are elements 0..7, not 0..15. A 16-wide split
/// decodes plausible weights in the wrong ORDER, which a dot cannot see and this does.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_nvfp4_native_matches_cpu() {
    check_embed_native(nvfp4_blocks, DType::Nvfp4, 64, "NVFP4");
}

/// Q2_K embedding table through the native `embed_q2k` in-kernel decode gather (×scale) vs CPU.
/// `embed_*` is the ONE path that reaches `deq_q2k` element-by-element (the GEMV tiers go through
/// the int8 codes), so this pins EVERY bit of the layout — the n×shift×half sub-block traversal that
/// makes the scale index `is` advance per 16-elem group, and the 4-bit scale / 4-bit min split — at
/// the f16-rounding tolerance rather than a dot product's error-averaging one.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q2k_native_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];
    let be = rocm().unwrap();
    let (vocab, ne) = (6usize, 256usize); // ne = one whole Q2_K super-block; vocab > max(ids)
    let scale = (ne as f32).sqrt(); // non-1.0 (Gemma-style) — must be applied on-device
    let t_bytes = q2k_blocks(vocab * ne / 256);
    let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::Q2K, vocab, ne, scale);
    let r = run_embed_gather(&be, &ids, &t_bytes, DType::Q2K, vocab, ne, scale);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "EmbedGather Q2_K (native) scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "EmbedGather Q2_K reference is all-zero — test is vacuous"
    );
    // Bit-faithful decode: the only loss is the f16 round of `sc*code + mn`. A mis-ordered `is`
    // (linear instead of n×shift×half) or a swapped scale/min nibble lands at O(1) relative.
    assert!(
        e / ref_mag < 2e-3,
        "EmbedGather Q2_K native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q3_K embedding table through the native `embed_q3k` in-kernel decode gather (×scale) vs CPU.
/// Same load-bearing role as the Q2_K case: the only element-wise path, so it pins the two places a
/// Q3_K port goes wrong — the kmask1/kmask2 6-bit scale shuffle and the polarity of the `hmask` high
/// bit (a flipped bit shifts the value by 4·d·sc6, i.e. O(1) relative).
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q3k_native_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];
    let be = rocm().unwrap();
    let (vocab, ne) = (6usize, 256usize); // ne = one whole Q3_K super-block; vocab > max(ids)
    let scale = (ne as f32).sqrt(); // non-1.0 (Gemma-style) — must be applied on-device
    let t_bytes = q3k_blocks(vocab * ne / 256);
    let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::Q3K, vocab, ne, scale);
    let r = run_embed_gather(&be, &ids, &t_bytes, DType::Q3K, vocab, ne, scale);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "EmbedGather Q3_K (native) scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "EmbedGather Q3_K reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 2e-3,
        "EmbedGather Q3_K native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q6_K embedding table through the native `embed_q6k` in-kernel decode gather (×scale) vs CPU.
/// THE table format that matters in practice: llama.cpp's Q4_K_M recipe stores `token_embd.weight`
/// as Q6_K (Qwen3-0.6B-Q4_K_M does), so this is the decode the `embed_gather` capability actually
/// runs on the golden model — and it was the one k-quant `embed_*` case with no coverage.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q6k_native_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];
    let be = rocm().unwrap();
    let (vocab, ne) = (6usize, 256usize); // ne = one whole Q6_K super-block; vocab > max(ids)
    let scale = (ne as f32).sqrt(); // non-1.0 (Gemma-style) — must be applied on-device
    let t_bytes = q6k_blocks(vocab * ne / 256);
    let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::Q6K, vocab, ne, scale);
    let r = run_embed_gather(&be, &ids, &t_bytes, DType::Q6K, vocab, ne, scale);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "EmbedGather Q6_K (native) scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "EmbedGather Q6_K reference is all-zero — test is vacuous"
    );
    // Bit-faithful decode: the only loss is the f16 round of `d * sc * code`, so the same
    // f16-rounding bound the Q5_K case uses.
    assert!(
        e / ref_mag < 2e-3,
        "EmbedGather Q6_K native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q5_K embedding table through the native `embed_q5k` in-kernel decode gather (×scale) vs CPU.
/// `embed_*` is the ONE path that reaches `deq_q5k` element-by-element (the GEMV tiers go through
/// the int8 codes), so this pins the bit-faithful `sc*code + mn` decode — including the `qh` 5th
/// bit — against `dequant_block` for every element of a row, not just a dot product of them.
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q5k_native_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];
    let be = rocm().unwrap();
    let (vocab, ne) = (6usize, 256usize); // ne = one whole Q5_K super-block; vocab > max(ids)
    let scale = (ne as f32).sqrt(); // non-1.0 (Gemma-style) — must be applied on-device
    let t_bytes = q5k_blocks(vocab * ne / 256);
    let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::Q5K, vocab, ne, scale);
    let r = run_embed_gather(&be, &ids, &t_bytes, DType::Q5K, vocab, ne, scale);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "EmbedGather Q5_K (native) scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "EmbedGather Q5_K reference is all-zero — test is vacuous"
    );
    // Bit-faithful decode: the only loss is the f16 round of `sc*code + mn`, so the bound is the
    // f16-rounding one, an order tighter than the int8-GEMV tolerances above. A wrong `qh` bit
    // moves a code by 16/31 of its range and lands at O(1) relative — far outside this.
    assert!(
        e / ref_mag < 2e-3,
        "EmbedGather Q5_K native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

/// Q8_0 embedding table through the native `embed_q80` in-kernel decode gather (×scale) vs CPU. Q8_0
/// has no sub-block scales — just `d` + int8 codes — so the per-element gather is a clean check of
/// the native decode + on-device embed scale (the token_embd bank that must NOT be f16-cached).
#[test]
#[ignore = "requires a ROCm GPU"]
fn embed_gather_q80_native_matches_cpu() {
    if rocm().is_none() {
        return;
    }
    let cpu = infr_cpu::CpuBackend::new();
    let ids = [0i32, 3, 5, 1, 5, 2];
    let be = rocm().unwrap();
    let (vocab, ne) = (6usize, 256usize); // ne a multiple of 32; vocab > max(ids)
    let scale = (ne as f32).sqrt(); // non-1.0 (Gemma-style) — must be applied on-device
    let t_bytes = synth_q(vocab * ne, 32, 34, 312, &[(0, 0.02)]);
    let c = run_embed_gather(&cpu, &ids, &t_bytes, DType::Q8_0, vocab, ne, scale);
    let r = run_embed_gather(&be, &ids, &t_bytes, DType::Q8_0, vocab, ne, scale);
    let e = maxerr(&c, &r);
    let ref_mag = maxabs(&c).max(1e-3);
    println!(
        "EmbedGather Q8_0 (native) scale={scale:e} max_err={e:e} max|ref|={ref_mag:e} rel={:e}",
        e / ref_mag
    );
    assert!(
        ref_mag > 1e-3,
        "EmbedGather Q8_0 reference is all-zero — test is vacuous"
    );
    assert!(
        e / ref_mag < 1e-2,
        "EmbedGather Q8_0 native decode diverges from CPU reference: abs={e:e} rel={:e}",
        e / ref_mag
    );
}

// ── the on-disk HIP module cache (slice RC) ──────────────────────────────────

/// The persisted code object for THIS box's arch, or `None` if no backend has written one yet.
/// (Globbed rather than recomputed: the arch token is `kernels.rs`-private, and a test that
/// duplicated the naming rule would pass while the real one drifted.)
fn module_cache_blob() -> Option<std::path::PathBuf> {
    let dir = infr_core::kernel_cache::cache_dir()?;
    let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("rocm-module-") && n.ends_with(".bin"))
        })
        .collect();
    found.sort();
    found.pop()
}

/// **A cache must never be able to produce a wrong result silently.** Two ways a persisted code
/// object can be wrong, and the recovery each takes:
///
/// 1. **Bit-rot in the payload** — caught by OUR envelope checksum, so the bytes never reach
///    `hipModuleLoadData` (where invalid cache data is undefined behavior, i.e. a hung ring).
/// 2. **A payload that is internally consistent but not a code object this runtime accepts** — a
///    checksum cannot see that, so `hipModuleLoadData` is the last check: its rejection must be a
///    RECOVERY (invalidate → compile → store), not a backend-init failure.
///
/// After each, a real `Op::Linear` must still match the CPU reference, and the file must be a
/// working blob again for the NEXT launch.
#[test]
#[ignore = "requires a ROCm GPU"]
fn a_corrupt_module_cache_blob_recovers_cleanly() {
    let Some(be) = rocm() else {
        return;
    };
    // One backend has built, so the blob is on disk (written by this run or a previous one).
    drop(be);
    let Some(path) = module_cache_blob() else {
        panic!("no rocm-module-*.bin after a backend build — the module cache never stored");
    };
    let good = std::fs::read(&path).expect("read the cached code object");
    assert!(good.len() > 28, "an envelope plus a payload");

    // The reference answer, and a closure that rebuilds a backend and re-runs it.
    let cpu = infr_cpu::CpuBackend::new();
    let (m, in_f, out_f) = (3usize, 256usize, 8usize);
    let x = gen(m * in_f, 4);
    let w_bytes: Vec<u8> = gen(out_f * in_f, 7)
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let want = run_linear(&cpu, &x, &w_bytes, DType::F16, m, in_f, out_f);
    let rebuild_and_check = |case: &str| {
        let be = rocm().expect("a corrupt cache must never fail backend init");
        let got = run_linear(&be, &x, &w_bytes, DType::F16, m, in_f, out_f);
        let e = maxerr(&want, &got);
        assert!(
            e < 1e-3,
            "{case}: recovered module diverges from CPU: {e:e}"
        );
    };

    // Envelope offsets (`infr_core::kernel_cache`): magic(8) ++ version(2) ++ key_len(2) ++
    // payload_len(8) ++ payload_hash(8), then the key, then the payload.
    let key_len = u16::from_le_bytes(good[10..12].try_into().unwrap()) as usize;
    let start = 28 + key_len;
    assert!(good.len() > start, "the payload is non-empty");

    // ── 1. bit-rot: the checksum must catch it, and the file is discarded ──
    let mut rot = good.clone();
    rot[start + 64] ^= 0x01;
    std::fs::write(&path, &rot).unwrap();
    rebuild_and_check("bit-rotted payload");
    assert_eq!(
        std::fs::read(&path).unwrap()[..start],
        good[..start],
        "after a checksum miss the blob must be re-stored, not left damaged"
    );

    // ── 2. a well-enveloped payload the runtime cannot load ──
    // Scramble the payload AND fix up the checksum, so every check we own passes and only
    // `hipModuleLoadData` can say no.
    let mut poison = good.clone();
    for b in poison[start..].iter_mut() {
        *b ^= 0xA5;
    }
    let sum = infr_core::kernel_cache::fnv1a(&poison[start..]);
    poison[20..28].copy_from_slice(&sum.to_le_bytes());
    std::fs::write(&path, &poison).unwrap();
    rebuild_and_check("runtime-rejected code object");
    let after = std::fs::read(&path).unwrap();
    assert_ne!(
        after, poison,
        "a rejected blob must be replaced by a freshly compiled one, not kept"
    );
    assert_eq!(
        after.len(),
        good.len(),
        "and it is a real code object again"
    );

    // The recovered file must itself load cleanly — otherwise every later launch pays a compile.
    rebuild_and_check("re-stored blob");
}

// ── F1 fusion gate: the ops the `combined_gu` / `gated_rmsnorm` / `argmax_rows` capabilities ────
//    make the seam emit. Each was implemented during ROCm bring-up but never proved against the
//    CPU reference, because the "start with NOTHING fused" capability dial kept the seam from ever
//    emitting it. These are the proofs that had to exist before the flags could be flipped.

/// Declare a plain f32 `[n]` graph Input (the shape every activation in these fusion cases uses).
fn f32_in(g: &mut Graph, n: usize) -> infr_core::tensor::TensorId {
    g.input(f32d(n))
}

/// `Op::GatedActFused` over the COMBINED `gu` buffer `[rows, 2*nff]` (gate half first, up half
/// second per row): `dst[r,i] = act(gu[r,i]) * gu[r, nff+i]`. This is what a `combined_gu` backend
/// emits after the one wide `[2*nff, ne]` FFN projection, replacing `Linear`+`Linear`+`GatedAct`.
/// The ROCm handler drives the SAME `gated_act` kernel as the split form, with `up_off = nff` and
/// `up_stride = gate_stride = 2*nff`, so the half-selection arithmetic is the thing under test: an
/// off-by-`nff` reads the up half as the gate and silently produces plausible-but-wrong
/// activations. Decode (rows=1) and prefill (rows=4) shapes, SiLU and GeLU.
#[test]
#[ignore = "requires a ROCm GPU"]
fn gated_act_fused_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let nff = 128usize;
    for (rows, act, label) in [
        (1usize, Activation::Silu, "silu rows=1"),
        (4usize, Activation::Silu, "silu rows=4"),
        (1usize, Activation::Gelu, "gelu rows=1"),
        (4usize, Activation::Gelu, "gelu rows=4"),
    ] {
        let gu = gen(rows * 2 * nff, 17);
        let run = |b: &dyn Backend| -> Vec<f32> {
            let mut g = Graph::new();
            let gid = f32_in(&mut g, rows * 2 * nff);
            let dst = g.output(f32d(rows * nff));
            g.push(Op::GatedActFused {
                gu: gid,
                dst,
                rows: rows as u32,
                nff: nff as u32,
                act,
            });
            let plan = b.compile(&g).expect("compile");
            let gb = b.alloc(gu.len() * 4, BufferUsage::Activations).expect("gu");
            b.upload(gb.as_ref(), bytemuck::cast_slice(&gu)).unwrap();
            let ob = b.alloc(rows * nff * 4, BufferUsage::Readback).expect("out");
            let mut bd = Bindings::new();
            bd.bind(gid, gb.as_ref());
            bd.bind(dst, ob.as_ref());
            b.execute(plan.as_ref(), &bd).expect("execute");
            let mut o = vec![0f32; rows * nff];
            b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
                .unwrap();
            o
        };
        let c = run(&cpu);
        let r = run(&be);
        let e = maxerr(&c, &r);
        let mag = maxabs(&c).max(1e-6);
        println!("GatedActFused {label} max_err={e:e} max|ref|={mag:e}");
        assert!(
            mag > 1e-3,
            "GatedActFused {label} reference all-zero — vacuous"
        );
        assert!(
            e / mag < 1e-3,
            "GatedActFused {label} diverges from CPU reference (wrong gate/up half?): abs={e:e}"
        );
    }
}

/// `Op::Linear` with a non-zero `w_off` — the OFFSET decode GEMV into a combined weight bank. A
/// `combined_gu` backend also opts the seam into the fused QKV upload (`fuse_qkv_decision` reads
/// the same capability), whose decode path issues three GEMVs at `w_off = 0 / qrow*ne /
/// (qrow+kvrow)*ne` into ONE `[qrow+2*kvrow, ne]` buffer. Nothing on ROCm exercised `w_off` before
/// this: the block-offset arithmetic (`(w_off/qpb)*bpb`) is quant-layout-specific, and getting it
/// wrong reads a shifted weight bank — coherent-looking garbage. Q4_K weight, decode (m=1) and
/// prefill (m=4), every slice of a 3-way split checked against the CPU reference.
#[test]
#[ignore = "requires a ROCm GPU"]
fn linear_w_off_matches_cpu() {
    let cpu = infr_cpu::CpuBackend::new();
    let in_f = 256usize; // one Q4_K super-block per weight row
    let slice_rows = 32usize; // rows per projection slice
    let out_f_total = 3 * slice_rows;
    // Per-BLOCK distinct payload (`lcg_bytes(seed ^ block)`), NOT the local `q4k_blocks` byte ramp:
    // that ramp has a 256-byte period and a slice is 32×144 = 4608 bytes ≡ 0 (mod 256), so all
    // three slices would carry byte-identical weights and a wrong `w_off` could not be observed.
    let w_bytes = infr_testkit::synth_weight(DType::Q4K, out_f_total * in_f, 0x5F01);
    for m in [1usize, 4] {
        let x = gen(m * in_f, 23);
        // Every slice must produce a DIFFERENT reference output, else a wrong offset is invisible.
        let mut refs: Vec<Vec<f32>> = Vec::new();
        for slice in 0..3 {
            // Fresh backend per case — `dequant_weight_or_cache` keys on the device pointer
            // (see `embed_gather_matches_cpu`'s note on recycled addresses).
            let Some(be) = rocm() else {
                return;
            };
            let w_off = slice * slice_rows * in_f;
            let run = |b: &dyn Backend| -> Vec<f32> {
                let mut g = Graph::new();
                let xid = f32_in(&mut g, m * in_f);
                let wid = g.weight(TensorDesc::new(vec![out_f_total * in_f], DType::Q4K));
                let dst = g.output(f32d(m * slice_rows));
                g.push(Op::Linear {
                    x: xid,
                    weight: wid,
                    dst,
                    m: m as u32,
                    in_f: in_f as u32,
                    out_f: slice_rows as u32,
                    w_off: w_off as u32,
                });
                let plan = b.compile(&g).expect("compile");
                let xb = b.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
                b.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
                let wb = b.alloc(w_bytes.len(), BufferUsage::Weights).expect("w");
                b.upload(wb.as_ref(), &w_bytes).unwrap();
                let ob = b
                    .alloc(m * slice_rows * 4, BufferUsage::Readback)
                    .expect("out");
                let mut bd = Bindings::new();
                bd.bind(xid, xb.as_ref());
                bd.bind(wid, wb.as_ref());
                bd.bind(dst, ob.as_ref());
                b.execute(plan.as_ref(), &bd).expect("execute");
                let mut o = vec![0f32; m * slice_rows];
                b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
                    .unwrap();
                o
            };
            let c = run(&cpu);
            let r = run(&be);
            let e = maxerr(&c, &r);
            let mag = maxabs(&c).max(1e-6);
            println!(
                "Linear w_off m={m} slice={slice} max_err={e:e} max|ref|={mag:e} rel={:e}",
                e / mag
            );
            assert!(
                mag > 1e-3,
                "Linear w_off m={m} slice={slice} reference all-zero — vacuous"
            );
            // int8-activation dp4a decode / WMMA prefill vs the f32 CPU dot: the same tolerance
            // the other native-quant GEMV cases here use.
            assert!(
                e / mag < 3e-2,
                "Linear w_off m={m} slice={slice} diverges from CPU (wrong block offset?): rel={:e}",
                e / mag
            );
            refs.push(c);
        }
        for i in 0..refs.len() {
            for j in i + 1..refs.len() {
                assert!(
                    maxerr(&refs[i], &refs[j]) > 1e-3,
                    "m={m}: slices {i} and {j} produce the SAME reference output — the weight \
                     payload does not vary per slice, so this case could not catch a wrong w_off"
                );
            }
        }
    }
}

/// Compile + run a 2-input (`x`, `gate`) + 1-weight graph on `b` and download `dst` as `out_len`
/// f32s. Shared by the fused and split arms of [`gated_rmsnorm_matches_cpu`] so the two differ
/// only in the ops pushed.
#[allow(clippy::too_many_arguments)]
fn run_xzw(
    b: &dyn Backend,
    g: Graph,
    xid: infr_core::tensor::TensorId,
    zid: infr_core::tensor::TensorId,
    wid: infr_core::tensor::TensorId,
    dst: infr_core::tensor::TensorId,
    x: &[f32],
    z: &[f32],
    w_bytes: &[u8],
    out_len: usize,
) -> Vec<f32> {
    let plan = b.compile(&g).expect("compile");
    let xb = b.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    b.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let zb = b.alloc(z.len() * 4, BufferUsage::Activations).expect("z");
    b.upload(zb.as_ref(), bytemuck::cast_slice(z)).unwrap();
    let wb = b.alloc(w_bytes.len(), BufferUsage::Weights).expect("w");
    b.upload(wb.as_ref(), w_bytes).unwrap();
    let ob = b.alloc(out_len * 4, BufferUsage::Readback).expect("out");
    let mut bd = Bindings::new();
    bd.bind(xid, xb.as_ref());
    bd.bind(zid, zb.as_ref());
    bd.bind(wid, wb.as_ref());
    bd.bind(dst, ob.as_ref());
    b.execute(plan.as_ref(), &bd).expect("execute");
    let mut o = vec![0f32; out_len];
    b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// `Op::GatedRmsNorm` (qwen35 DeltaNet z-gate): per-head RMSNorm × learned weight × `silu(gate)`
/// in ONE dispatch, replacing the `QkNorm`→`GatedAct` pair the runner emits when the capability is
/// off. Checked against the CPU interpreter's own `GatedRmsNorm` (which it implements precisely so
/// this parity can exist) AND against the split pair on BOTH backends — so the fusion is proved to
/// be the same math as what it replaces, not merely self-consistent.
#[test]
#[ignore = "requires a ROCm GPU"]
fn gated_rmsnorm_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, hd) = (3usize, 4usize, 64usize);
    let eps = 1e-6f32;
    let x = gen(rows * nh * hd, 5);
    let gate = gen(rows * nh * hd, 29);
    let wf32 = gen(hd, 7);
    let w_bytes = f16_bytes(&wf32);
    let fused = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let xid = f32_in(&mut g, rows * nh * hd);
        let zid = f32_in(&mut g, rows * nh * hd);
        let wid = g.weight(TensorDesc::new(vec![hd], DType::F16));
        let dst = g.output(f32d(rows * nh * hd));
        g.push(Op::GatedRmsNorm {
            x: xid,
            weight: wid,
            gate: zid,
            dst,
            rows: rows as u32,
            n_head: nh as u32,
            head_dim: hd as u32,
            eps,
        });
        run_xzw(
            b,
            g,
            xid,
            zid,
            wid,
            dst,
            &x,
            &gate,
            &w_bytes,
            rows * nh * hd,
        )
    };
    // The SPLIT pair the runner emits without the capability: QkNorm (per-head rmsnorm × weight)
    // then GatedAct (SiLU gate × the normed value, packed strides).
    let split = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let xid = f32_in(&mut g, rows * nh * hd);
        let zid = f32_in(&mut g, rows * nh * hd);
        let wid = g.weight(TensorDesc::new(vec![hd], DType::F16));
        let normed = g.internal(f32d(rows * nh * hd));
        let dst = g.output(f32d(rows * nh * hd));
        g.push(Op::QkNorm {
            x: xid,
            weight: wid,
            dst: normed,
            rows: rows as u32,
            n_head: nh as u32,
            head_dim: hd as u32,
            eps,
            x_stride: 0,
        });
        g.push(Op::GatedAct {
            gate: zid,
            up: normed,
            dst,
            rows: rows as u32,
            nff: (nh * hd) as u32,
            act: Activation::Silu,
            up_off: 0,
            up_stride: 0,
            gate_stride: 0,
            gate_block_width: 0,
        });
        run_xzw(
            b,
            g,
            xid,
            zid,
            wid,
            dst,
            &x,
            &gate,
            &w_bytes,
            rows * nh * hd,
        )
    };
    let c_fused = fused(&cpu);
    let c_split = split(&cpu);
    let mag = maxabs(&c_fused).max(1e-6);
    assert!(mag > 1e-3, "GatedRmsNorm reference all-zero — vacuous");
    let e_ref = maxerr(&c_fused, &c_split);
    println!("GatedRmsNorm cpu fused-vs-split max_err={e_ref:e} max|ref|={mag:e}");
    assert!(
        e_ref / mag < 1e-6,
        "the fused op is not the same math as the QkNorm→GatedAct pair it replaces: rel={:e}",
        e_ref / mag
    );
    let r_fused = fused(&be);
    let e = maxerr(&c_fused, &r_fused);
    println!("GatedRmsNorm rocm-vs-cpu max_err={e:e} rel={:e}", e / mag);
    assert!(
        e / mag < 1e-3,
        "GatedRmsNorm diverges from the CPU reference: abs={e:e} rel={:e}",
        e / mag
    );
    // And ROCm's fused op must agree with the split pair ROCm itself runs today — the exact swap
    // flipping the capability performs.
    let r_split = split(&be);
    let e2 = maxerr(&r_split, &r_fused);
    println!(
        "GatedRmsNorm rocm fused-vs-split max_err={e2:e} rel={:e}",
        e2 / mag
    );
    assert!(
        e2 / mag < 1e-3,
        "ROCm's fused op disagrees with its own split pair: abs={e2:e}"
    );
}

/// `Op::Argmax` with `rows > 1` — the `argmax_rows` capability (MTP speculative-verify accept: m
/// u32 ids read back instead of the m×vocab logits). Bit-EXACT: the id is a u32 bit-pattern, so
/// any difference is a wrong token, not a rounding error. The tie case is the point — the
/// reference rule is first-max (strict `>`, lowest index wins), and the HIP kernel reaches it
/// through a shared-memory tree reduce that has to break ties the same way. Includes a real
/// vocab-sized row (151936) and a deliberate duplicate maximum inside row 0.
#[test]
#[ignore = "requires a ROCm GPU"]
fn argmax_multirow_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    for (rows, n) in [(1usize, 151936usize), (4, 151936), (6, 1024), (3, 257)] {
        let mut x = gen(rows * n, 11);
        // A unique maximum per row at a scattered index …
        for r in 0..rows {
            x[r * n + (r * 7919 + 13) % n] = 9.0 + r as f32;
        }
        // … plus, in row 0, the SAME value planted a second time so the first-max tie rule is
        // exercised (the LOWER index must win, on both backends).
        if rows > 1 {
            x[(n / 3).min(n - 1)] = 9.0;
        }
        let run = |b: &dyn Backend| -> Vec<u32> {
            let mut g = Graph::new();
            let xid = f32_in(&mut g, rows * n);
            let dst = g.output(f32d(rows));
            g.push(Op::Argmax {
                x: xid,
                dst,
                n: n as u32,
                rows: rows as u32,
            });
            let plan = b.compile(&g).expect("compile");
            let xb = b.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
            b.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
            let ob = b.alloc(rows * 4, BufferUsage::Readback).expect("out");
            let mut bd = Bindings::new();
            bd.bind(xid, xb.as_ref());
            bd.bind(dst, ob.as_ref());
            b.execute(plan.as_ref(), &bd).expect("execute");
            let mut o = vec![0u32; rows];
            b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
                .unwrap();
            o
        };
        let c = run(&cpu);
        let r = run(&be);
        println!("Argmax rows={rows} n={n} cpu={c:?} rocm={r:?}");
        assert_eq!(
            c, r,
            "Argmax rows={rows} n={n} picked a different token than the CPU reference \
             (tie rule or cross-thread reduce?)"
        );
    }
}

// ── F1b: the sibling-GEMV activation-quant memo ────────────────────────────────────────────────
//    Consecutive decode `Linear`s that read the SAME activation row (q/k/v off one input norm, the
//    `w_off` slices of a fused-QKV bank, gate+up off one post-attention norm) used to re-run the
//    int8 quantization pass per projection. The executor now runs it ONCE and rebinds the same
//    codes/scales for the siblings. The claim is not "close enough" but BIT-IDENTICAL — the GEMV
//    reads the exact bytes the elided pass would have written — so these tests assert `==`, not a
//    tolerance, against the same projections issued in a shape where no memo can form.

/// Run `n_proj` decode `Linear`s off ONE `RmsNorm`, either as a single graph (siblings adjacent →
/// the memo fires for projections 2..n) or as one graph per projection (nothing to memoize).
/// Returns the concatenated outputs.
#[allow(clippy::too_many_arguments)]
fn run_sibling_projections(
    be: &dyn Backend,
    x: &[f32],
    norm_w: &[f32],
    w: &[Vec<u8>],
    dt: DType,
    in_f: usize,
    out_f: usize,
    together: bool,
) -> Vec<f32> {
    let groups: Vec<std::ops::Range<usize>> = if together {
        // ONE graph holding every projection — siblings adjacent, so the memo fires.
        std::iter::once(0..w.len()).collect()
    } else {
        // One graph per projection — nothing for the memo to reuse.
        (0..w.len()).map(|i| i..i + 1).collect()
    };
    let mut out = Vec::new();
    for grp in groups {
        let mut g = Graph::new();
        let xid = g.input(f32d(in_f));
        let nwid = g.weight(TensorDesc::new(vec![in_f], DType::F32));
        let normed = g.internal(f32d(in_f));
        g.push(Op::RmsNorm {
            x: xid,
            weight: nwid,
            dst: normed,
            rows: 1,
            dim: in_f as u32,
            eps: 1e-6,
        });
        let ids: Vec<_> = grp
            .map(|i| {
                let wid = g.weight(TensorDesc::new(vec![out_f * in_f], dt));
                let dst = g.output(f32d(out_f));
                g.push(Op::Linear {
                    x: normed,
                    weight: wid,
                    dst,
                    m: 1,
                    in_f: in_f as u32,
                    out_f: out_f as u32,
                    w_off: 0,
                });
                (i, wid, dst)
            })
            .collect();
        let plan = be.compile(&g).expect("compile");
        let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
        be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
        let nwb = be
            .alloc(norm_w.len() * 4, BufferUsage::Weights)
            .expect("nw");
        be.upload(nwb.as_ref(), bytemuck::cast_slice(norm_w))
            .unwrap();
        let mut bd = Bindings::new();
        bd.bind(xid, xb.as_ref());
        bd.bind(nwid, nwb.as_ref());
        let wbs: Vec<_> = ids
            .iter()
            .map(|(i, wid, _)| {
                let wb = be.alloc(w[*i].len(), BufferUsage::Weights).expect("w");
                be.upload(wb.as_ref(), &w[*i]).unwrap();
                (*wid, wb)
            })
            .collect();
        for (wid, wb) in &wbs {
            bd.bind(*wid, wb.as_ref());
        }
        let obs: Vec<_> = ids
            .iter()
            .map(|(_, _, dst)| {
                let ob = be.alloc(out_f * 4, BufferUsage::Readback).expect("out");
                (*dst, ob)
            })
            .collect();
        for (dst, ob) in &obs {
            bd.bind(*dst, ob.as_ref());
        }
        be.execute(plan.as_ref(), &bd).expect("execute");
        for (_, ob) in &obs {
            let mut o = vec![0f32; out_f];
            be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
                .unwrap();
            out.extend_from_slice(&o);
        }
    }
    out
}

/// The memo must change NOTHING it computes. Three decode projections off one `RmsNorm` (the
/// q/k/v shape, where only the first re-quantizes) are compared BIT-FOR-BIT against the same three
/// projections issued one graph at a time, where no sibling exists to memoize — and, at a
/// tolerance, against the CPU reference. Run for a k-quant, a legacy round quant and Q8_0 so the
/// rebound `(codes, scales)` pair is proved against more than one shape of int8 GEMV.
///
/// The weights must DIFFER per projection: identical banks would make a mis-bound memo (or a
/// mis-bound weight) invisible.
#[test]
#[ignore = "requires a ROCm GPU"]
fn quant_memo_sibling_linears_are_bit_identical() {
    let cpu = infr_cpu::CpuBackend::new();
    let (in_f, out_f) = (256usize, 24usize);
    let x = gen(in_f, 31);
    let norm_w = gen(in_f, 37);
    for dt in [DType::Q4K, DType::Q4_0, DType::Q8_0] {
        let w: Vec<Vec<u8>> = (0..3)
            .map(|i| infr_testkit::synth_weight(dt, out_f * in_f, 0xB100 + i as u32))
            .collect();
        // Fresh backend per case — `dequant_weight_or_cache` keys on the device pointer, and a
        // recycled address across cases would serve a stale norm weight (see `linear_w_off`).
        let Some(be) = rocm() else {
            return;
        };
        let split = run_sibling_projections(&be, &x, &norm_w, &w, dt, in_f, out_f, false);
        let Some(be) = rocm() else {
            return;
        };
        let memoed = run_sibling_projections(&be, &x, &norm_w, &w, dt, in_f, out_f, true);
        assert_eq!(
            split, memoed,
            "{dt:?}: the memoized sibling GEMVs are NOT bit-identical to the same projections \
             quantized individually — the rebound codes/scales do not describe the same row"
        );
        // Sanity: the three projections must actually differ, else bit-identity is vacuous.
        assert_ne!(
            &memoed[..out_f],
            &memoed[out_f..2 * out_f],
            "{dt:?}: sibling projections produced identical output — test is vacuous"
        );
        let c = run_sibling_projections(&cpu, &x, &norm_w, &w, dt, in_f, out_f, true);
        let e = maxerr(&c, &memoed);
        let mag = maxabs(&c).max(1e-3);
        println!(
            "quant-memo {dt:?} vs CPU max_err={e:e} max|ref|={mag:e} rel={:e}",
            e / mag
        );
        assert!(mag > 1e-3, "{dt:?}: reference all-zero — vacuous");
        assert!(
            e / mag < 2e-2,
            "{dt:?}: memoized siblings diverge from the CPU reference: abs={e:e} rel={:e}",
            e / mag
        );
    }
}

/// The memo's invalidation guard. A decode `Linear` whose residual `Add` is fused writes into the
/// residual stream — which, on a real graph, is also the next norm's input row. If such a GEMV
/// published its activation memo, a following projection off that same row would dot a
/// quantization of the row's PREVIOUS contents: a silent, plausible-looking wrong answer.
///
/// So: `Linear(x) → Add(into x)`, then a second `Linear(x)`. The second must see the UPDATED row.
/// Checked against the CPU reference, which has neither memo nor fusion.
#[test]
#[ignore = "requires a ROCm GPU"]
fn quant_memo_is_dropped_when_the_gemv_rewrites_its_own_source() {
    let cpu = infr_cpu::CpuBackend::new();
    let n = 256usize; // square projections so the fused Add can land back on `x`
    let x0 = gen(n, 41);
    let w1 = infr_testkit::synth_weight(DType::Q4K, n * n, 0xB201);
    let w2 = infr_testkit::synth_weight(DType::Q4K, n * n, 0xB202);
    let run = |b: &dyn Backend| -> Vec<f32> {
        let mut g = Graph::new();
        let xid = g.input(f32d(n));
        let w1id = g.weight(TensorDesc::new(vec![n * n], DType::Q4K));
        let w2id = g.weight(TensorDesc::new(vec![n * n], DType::Q4K));
        let t = g.internal(f32d(n));
        let dst = g.output(f32d(n));
        g.push(Op::Linear {
            x: xid,
            weight: w1id,
            dst: t,
            m: 1,
            in_f: n as u32,
            out_f: n as u32,
            w_off: 0,
        });
        // Residual add straight back onto the row the GEMV read (`plan_linear_add` folds this into
        // the GEMV epilogue on ROCm, so the GEMV's own write target IS its quant source).
        g.push(Op::Add {
            a: t,
            b: xid,
            dst: xid,
            n: n as u32,
        });
        g.push(Op::Linear {
            x: xid,
            weight: w2id,
            dst,
            m: 1,
            in_f: n as u32,
            out_f: n as u32,
            w_off: 0,
        });
        let plan = b.compile(&g).expect("compile");
        let xb = b.alloc(n * 4, BufferUsage::Activations).expect("x");
        b.upload(xb.as_ref(), bytemuck::cast_slice(&x0)).unwrap();
        let w1b = b.alloc(w1.len(), BufferUsage::Weights).expect("w1");
        b.upload(w1b.as_ref(), &w1).unwrap();
        let w2b = b.alloc(w2.len(), BufferUsage::Weights).expect("w2");
        b.upload(w2b.as_ref(), &w2).unwrap();
        let ob = b.alloc(n * 4, BufferUsage::Readback).expect("out");
        let mut bd = Bindings::new();
        bd.bind(xid, xb.as_ref());
        bd.bind(w1id, w1b.as_ref());
        bd.bind(w2id, w2b.as_ref());
        bd.bind(dst, ob.as_ref());
        b.execute(plan.as_ref(), &bd).expect("execute");
        let mut o = vec![0f32; n];
        b.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let Some(be) = rocm() else {
        return;
    };
    let c = run(&cpu);
    let r = run(&be);
    let e = maxerr(&c, &r);
    let mag = maxabs(&c).max(1e-3);
    println!(
        "quant-memo invalidation max_err={e:e} max|ref|={mag:e} rel={:e}",
        e / mag
    );
    assert!(mag > 1e-3, "reference all-zero — vacuous");
    assert!(
        e / mag < 2e-2,
        "the second projection did not see the updated residual row — a stale activation memo \
         survived the GEMV that rewrote its source: abs={e:e} rel={:e}",
        e / mag
    );
}

// ── F1c: the two MoE decode folds (`MoeFfn → Add`, `RmsNorm → MoeFfn`) ────────
//
// Both are BIT-IDENTITY folds, not tolerance folds, and both are asserted that way: the comparand
// is the SAME graph run with the fold's own escape hatch cleared, so the two runs differ in exactly
// one thing — whether the pattern was planned. `assert_eq!` on the raw f32 bits, because anything
// less would not see the failure these folds could actually produce.
//
// For `MoeFfn → Add` that failure is a REASSOCIATED expert sum. R8 (`d787890`) deliberately kept
// `moe_accum_idm` reducing the slots in ascending order onto a 0.0f seed rather than using
// atomicAdd, so the result is deterministic and the seam golden depends on it; folding the residual
// in as a seed (`((h + s0) + s1) …`) instead of at the end (`h + ((s0 + s1) …)`) would still be
// "correct" to any tolerance and would still move the golden. An `==` comparand catches it.

/// Build a `[RmsNorm →] MoeFfn [→ Add]` graph and run it on `be`, returning the final output.
/// `pre_norm` prepends the norm the F1c fold elides (the MoE input norm's shape: the SAME tensor
/// feeds `x` and `router_x`); `residual` appends the in-place residual `Add` the seam emits
/// (`hidden = hidden + moe_out`), and the returned buffer is then the `Add`'s dst.
#[allow(clippy::too_many_arguments)]
fn run_moe_folds(
    be: &dyn Backend,
    x: &[f32],
    router_f32: &[f32],
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    dt: DType,
    rows: usize,
    ne: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    pre_norm: Option<&[u8]>,
    residual: Option<&[f32]>,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(f32d(rows * ne));
    let rid = g.weight(TensorDesc::new(vec![n_expert * ne], DType::F32));
    let gid = g.weight(TensorDesc::new(vec![n_expert * n_ff_exp * ne], dt));
    let uid = g.weight(TensorDesc::new(vec![n_expert * n_ff_exp * ne], dt));
    let did = g.weight(TensorDesc::new(vec![n_expert * ne * n_ff_exp], dt));
    // The norm's output is an INTERNAL scratch — the live-range bound the shared pass applies only
    // ever fires on Internal handles, so an Output here would silently un-plan the fold.
    let nw = pre_norm.map(|_| g.weight(TensorDesc::new(vec![ne], DType::F16)));
    let moe_in = match nw {
        Some(w) => {
            let hn = g.internal(f32d(rows * ne));
            g.push(Op::RmsNorm {
                x: xid,
                weight: w,
                dst: hn,
                rows: rows as u32,
                dim: ne as u32,
                eps: 1e-6,
            });
            hn
        }
        None => xid,
    };
    // Likewise the MoE's own dst when a residual `Add` follows it.
    let (moe_dst, resid_id, dst) = match residual {
        Some(_) => {
            let sub = g.internal(f32d(rows * ne));
            let h = g.input(f32d(rows * ne));
            let out = g.output(f32d(rows * ne));
            (sub, Some(h), out)
        }
        None => {
            let out = g.output(f32d(rows * ne));
            (out, None, out)
        }
    };
    g.push(Op::MoeFfn {
        x: moe_in,
        router_x: moe_in,
        router: rid,
        gate_exps: gid,
        up_exps: uid,
        down_exps: did,
        down_scale: None,
        dst: moe_dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: n_ff_exp as u32,
        scale: 1.0,
        act: Activation::Silu,
        gating: MoeGating::Softmax,
        norm_w: true,
        weight_before: false,
        fused_gate_up: false,
        ep_band: None,
    });
    if let Some(h) = resid_id {
        g.push(Op::Add {
            a: h,
            b: moe_dst,
            dst,
            n: (rows * ne) as u32,
        });
    }
    let plan = be.compile(&g).expect("compile");

    let up = |bytes: &[u8], usage| {
        let b = be.alloc(bytes.len(), usage).expect("alloc");
        be.upload(b.as_ref(), bytes).unwrap();
        b
    };
    let xb = up(bytemuck::cast_slice(x), BufferUsage::Activations);
    let rb = up(bytemuck::cast_slice(router_f32), BufferUsage::Weights);
    let gb = up(gate_bytes, BufferUsage::Weights);
    let ub = up(up_bytes, BufferUsage::Weights);
    let db = up(down_bytes, BufferUsage::Weights);
    let nb = pre_norm.map(|w| up(w, BufferUsage::Weights));
    let hb = residual.map(|h| up(bytemuck::cast_slice(h), BufferUsage::Activations));
    let ob = be.alloc(rows * ne * 4, BufferUsage::Readback).expect("out");

    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(rid, rb.as_ref());
    b.bind(gid, gb.as_ref());
    b.bind(uid, ub.as_ref());
    b.bind(did, db.as_ref());
    if let (Some(w), Some(buf)) = (nw, &nb) {
        b.bind(w, buf.as_ref());
    }
    if let (Some(h), Some(buf)) = (resid_id, &hb) {
        b.bind(h, buf.as_ref());
    }
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");

    let mut o = vec![0f32; rows * ne];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// A ROCm backend with one `kernels.rocm` field overridden — the A/B comparand for each fold.
fn rocm_cfg(f: impl FnOnce(&mut infr_core::config::RocmCfg)) -> Option<RocmBackend> {
    let mut cfg = infr_core::config::Config::default();
    f(&mut cfg.kernels.rocm);
    RocmBackend::new_with(0, std::sync::Arc::new(cfg)).ok()
}

/// F16 norm weights, ~1.0 with per-channel variation (a flat 1.0 weight would hide a mis-indexed
/// `weight[i]` in the fused normalize).
fn f16_norm_weight(n: usize) -> Vec<u8> {
    let mut w = vec![0u8; n * 2];
    for i in 0..n {
        let v = 0.7 + 0.6 * ((i % 7) as f32 / 7.0);
        w[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
    }
    w
}

/// **`MoeFfn → Add` must be bit-identical.** Same graph, fold on vs fold off (`fuse_add`), `==` on
/// every element — so a residual folded into the accumulate as a SEED (which would reassociate the
/// ascending-slot sum R8 pinned) fails here rather than in a moved golden hash three slices later.
/// Run at `n_used = 1..4`: with one slot the sum has no order to lose, so only `n_used > 1` puts
/// the association under test at all.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_residual_fold_is_bit_identical() {
    let (Some(on), Some(off)) = (rocm(), rocm_cfg(|c| c.fuse_add = false)) else {
        return;
    };
    let (ne, n_expert, n_ff_exp) = (256usize, 8usize, 128usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    let dn_blocks = n_expert * ne * n_ff_exp / 32;
    let gate = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let upw = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, |e| expert_scale(e) * 0.8);
    let down = q80_per_expert_scaled(dn_blocks, dn_blocks / n_expert, expert_scale);
    for n_used in [1usize, 2, 4] {
        let x = gen(ne, 3);
        let router = gen(n_expert * ne, 9);
        let plain = run_moe_folds(
            &on,
            &x,
            &router,
            &gate,
            &upw,
            &down,
            DType::Q8_0,
            1,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            None,
            None,
        );
        assert!(
            maxabs(&plain) > 1e-3,
            "MoeFfn residual fold [n_used={n_used}] reference is all-zero — test is vacuous"
        );
        // Scale the residual to the MoE output's OWN magnitude. A tiny residual would be absorbed
        // whole into the sum's rounding, and then a reassociated accumulate — the failure this test
        // exists for — would produce the same bits and pass.
        let rscale = maxabs(&plain) / maxabs(&gen(ne, 17)).max(1e-6);
        let resid: Vec<f32> = gen(ne, 17).iter().map(|v| v * rscale).collect();
        let go = |b: &dyn Backend| {
            run_moe_folds(
                b,
                &x,
                &router,
                &gate,
                &upw,
                &down,
                DType::Q8_0,
                1,
                ne,
                n_expert,
                n_used,
                n_ff_exp,
                None,
                Some(&resid),
            )
        };
        let fused = go(&on);
        let split = go(&off);
        // Not a tolerance: the standalone `add` and the folded epilogue must produce the SAME bits.
        assert_eq!(
            fused,
            split,
            "MoeFfn→Add fold is not bit-identical at n_used={n_used} (max_err={:e}) — the expert \
             accumulate was reassociated",
            maxerr(&fused, &split)
        );
        // ... and it must actually be the residual that was added, not a dropped or doubled one.
        let want: Vec<f32> = plain.iter().zip(&resid).map(|(m, r)| r + m).collect();
        assert_eq!(
            fused, want,
            "the folded epilogue did not add the residual exactly once at n_used={n_used}"
        );
    }
}

/// **The declined tier still gets its `Add`.** The shared pass plans `MoeFfn → Add` and ELIDES the
/// standalone op from the walk; a tier that cannot carry the fold (here `moe_id_rows = 0`, the
/// pre-R8 per-slot loop whose `atomicAdd` accumulate has no epilogue) must REPLAY it. Without the
/// replay this test reads back the bare MoE output — the residual silently dropped, in a
/// configuration no golden covers.
#[test]
#[ignore = "requires a ROCm GPU"]
fn moe_residual_fold_declined_tier_replays_the_add() {
    let (Some(loop_tier), Some(off)) = (
        rocm_cfg(|c| c.moe_id_rows = 0),
        rocm_cfg(|c| {
            c.moe_id_rows = 0;
            c.fuse_add = false;
        }),
    ) else {
        return;
    };
    let (ne, n_expert, n_used, n_ff_exp) = (256usize, 8usize, 3usize, 128usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    let dn_blocks = n_expert * ne * n_ff_exp / 32;
    let gate = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let upw = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, |e| expert_scale(e) * 0.8);
    let down = q80_per_expert_scaled(dn_blocks, dn_blocks / n_expert, expert_scale);
    let x = gen(ne, 3);
    let router = gen(n_expert * ne, 9);
    let resid = gen(ne, 17);
    let go = |b: &dyn Backend| {
        run_moe_folds(
            b,
            &x,
            &router,
            &gate,
            &upw,
            &down,
            DType::Q8_0,
            1,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            None,
            Some(&resid),
        )
    };
    let planned = go(&loop_tier);
    let never_planned = go(&off);
    assert!(
        maxabs(&never_planned) > 1e-3,
        "declined-tier replay reference is all-zero — test is vacuous"
    );
    assert_eq!(
        planned, never_planned,
        "the pre-R8 tier declined the residual fold but did not replay the elided `Add` — the \
         residual was dropped"
    );
}

/// **`RmsNorm → MoeFfn` must be bit-identical.** The fold makes one `rmsnorm_quant_i8_32` produce
/// BOTH the experts' int8 codes and the f32 row the router GEMV reads; the codes have been
/// bit-faithful since Slice 32, but the `xn` output is new and it is what the router's expert
/// SELECTION depends on — a normalized row off by an ulp can flip a top-k tie and change which
/// experts run, which is a large output change, not a small one. Same graph, fold on vs off
/// (`fuse_norm`), `==`.
#[test]
#[ignore = "requires a ROCm GPU"]
fn rmsnorm_moe_fold_is_bit_identical() {
    let (Some(on), Some(off)) = (rocm(), rocm_cfg(|c| c.fuse_norm = false)) else {
        return;
    };
    let (ne, n_expert, n_used, n_ff_exp) = (256usize, 8usize, 3usize, 128usize);
    let gu_blocks = n_expert * n_ff_exp * ne / 32;
    let dn_blocks = n_expert * ne * n_ff_exp / 32;
    let gate = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, expert_scale);
    let upw = q80_per_expert_scaled(gu_blocks, gu_blocks / n_expert, |e| expert_scale(e) * 0.8);
    let down = q80_per_expert_scaled(dn_blocks, dn_blocks / n_expert, expert_scale);
    let nw = f16_norm_weight(ne);
    let x = gen(ne, 5);
    // The router weights are scaled DOWN so the softmax over `n_expert` stays SOFT. With the raw
    // generator the `ne`-wide dot saturates (top-1 weight rounds to exactly 1.0, the rest to 0.0)
    // and the gate weights become a step function of the logits — at which point the routed sum no
    // longer depends on the normalized row's low bits at all and this test would pass with an `xn`
    // that is merely CLOSE. Mutation-checked: at this scale a 0.1% perturbation of `xn` fails the
    // assert below; at the raw scale it does not.
    let router: Vec<f32> = gen(n_expert * ne, 9).iter().map(|v| v * 0.01).collect();
    let resid = gen(ne, 17);
    // Both folds together (the shipping shape: the MoE layer's norm elided AND its residual
    // folded) and the norm fold alone, so neither is only ever exercised in the other's company.
    for with_resid in [false, true] {
        let go = |b: &dyn Backend| {
            run_moe_folds(
                b,
                &x,
                &router,
                &gate,
                &upw,
                &down,
                DType::Q8_0,
                1,
                ne,
                n_expert,
                n_used,
                n_ff_exp,
                Some(&nw),
                with_resid.then_some(&resid[..]),
            )
        };
        let fused = go(&on);
        let split = go(&off);
        assert!(
            maxabs(&split) > 1e-3,
            "RmsNorm→MoeFfn fold [resid={with_resid}] reference is all-zero — test is vacuous"
        );
        assert_eq!(
            fused,
            split,
            "RmsNorm→MoeFfn fold is not bit-identical [resid={with_resid}] (max_err={:e}) — the \
             normalized row the router reads is not the one the standalone `rmsnorm` wrote",
            maxerr(&fused, &split)
        );
    }
}

// ── F1d: the `QkNormRope → WriteKv` peephole ─────────────────────────────────
//
// The fold makes `qk_norm_rope` cast each rotated element to f16 and store it in the KV cache row
// the elided `write_kv` would have filled. Two things can go wrong and neither shows up in a short
// greedy check: the VALUE can differ (a different rounding, a dropped norm weight), or the ROW can
// be wrong — and a wrong row corrupts a DIFFERENT position's attention, which the next token's
// argmax will happily not notice. Both get their own `==` case below, at a NON-ZERO write position
// with the rest of the cache poisoned.

/// Run `QkNormRope → WriteKv` on `be` and return the FULL f16 KV cache, downloaded as f32.
/// The cache is pre-filled with `poison` so a fused write that lands on the wrong row, or that
/// writes fewer elements than the row holds, is visible as a surviving/destroyed poison value.
/// `dst` of the rope is an `Internal` f16 handle and the `WriteKv` immediately follows it — the
/// exact shape `plan_kv_write` matches, so this graph fuses under the default config and splits
/// under `fuse_kv_write = false`.
#[allow(clippy::too_many_arguments)]
fn run_qk_norm_rope_write_kv(
    be: &dyn Backend,
    x: &[f32],
    weight_f16: &[u8],
    positions: &[i32],
    rows: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    eps: f32,
    cache_rows: usize,
    decl_rows: usize,
    row_stride: usize,
    pos: usize,
    poison: f32,
) -> Vec<f32> {
    let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
    let mut g = Graph::new();
    let xid = g.input(f32d(x.len()));
    let wid = g.weight(TensorDesc::new(vec![head_dim], DType::F16));
    let pid = g.input(TensorDesc::new(vec![positions.len()], DType::I32));
    // Internal + F16: `plan_kv_write` requires both (it redirects the write, so an Output `dst`
    // someone else reads back would silently lose its content).
    let k16 = g.internal(f16d(rows * n_head * head_dim));
    // `decl_rows` is the cache's DECLARED row capacity (what the shared pass derives its ring
    // modulo from); `cache_rows` is what is actually allocated and read back. They differ in exactly
    // one test — the one that pins that ROCm does NOT wrap.
    let cache = g.input(f16d(decl_rows * row_stride));
    g.push(Op::QkNormRope {
        x: xid,
        weight: wid,
        positions: pid,
        dst: k16,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        rope_dim: rope_dim as u32,
        theta,
        eps,
        freq_factors: None,
        x_stride: 0,
    });
    g.push(Op::WriteKv {
        src: k16,
        cache,
        rows: rows as u32,
        row_stride: row_stride as u32,
        pos: pos as u32,
    });
    let plan = be.compile(&g).expect("compile");

    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(x)).unwrap();
    let wb = be.alloc(weight_f16.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), weight_f16).unwrap();
    let pbytes: &[u8] = bytemuck::cast_slice(positions);
    let pb = be
        .alloc(pbytes.len(), BufferUsage::Activations)
        .expect("pos");
    be.upload(pb.as_ref(), pbytes).unwrap();
    let n_cache = cache_rows * row_stride;
    let mut poison_bytes = vec![0u8; n_cache * 2];
    let ph = half::f16::from_f32(poison).to_bits().to_le_bytes();
    for c in poison_bytes.chunks_exact_mut(2) {
        c.copy_from_slice(&ph);
    }
    let cb = be.alloc(n_cache * 2, BufferUsage::KvCache).expect("cache");
    be.upload(cb.as_ref(), &poison_bytes).unwrap();

    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(pid, pb.as_ref());
    b.bind(cache, cb.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");

    let mut raw = vec![0u8; n_cache * 2];
    be.download(cb.as_ref(), &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// **The fused K write is bit-identical to the pair it replaces, at a non-zero row.** Same graph,
/// fold on vs off (`fuse_kv_write`), `==` over the WHOLE cache — so it also pins that the fused
/// kernel wrote *only* its own row (every other row must still hold the poison) and that it filled
/// that row *completely* (no poison left inside it).
///
/// `pos` is deliberately not 0: the elided `write_kv` indexes `pos + row` and the fused kernel
/// indexes `kv_row + r`, and at `pos == 0` those two agree even if one of them ignores the row
/// offset entirely. Multi-row (`rows > 1`, the prefill shape) is included because the fused kernel
/// derives its row from `blockIdx.x / n_head` rather than from a thread-linear id.
#[test]
#[ignore = "requires a ROCm GPU"]
fn kv_write_fold_is_bit_identical_at_a_nonzero_row() {
    let (Some(on), Some(off)) = (rocm(), rocm_cfg(|c| c.fuse_kv_write = false)) else {
        return;
    };
    let (n_head, head_dim, rope_dim) = (4usize, 64usize, 64usize);
    let (cache_rows, theta, eps, poison) = (16usize, 10000.0f32, 1e-6f32, -7.5f32);
    let w = f16_norm_weight(head_dim);
    let row_stride = n_head * head_dim;
    for (rows, pos) in [(1usize, 0usize), (1, 1), (1, 11), (4, 5), (4, 12)] {
        let x = gen(rows * n_head * head_dim, 3 + pos);
        let positions: Vec<i32> = (0..rows).map(|r| (pos + r) as i32).collect();
        let go = |b: &dyn Backend| {
            run_qk_norm_rope_write_kv(
                b, &x, &w, &positions, rows, n_head, head_dim, rope_dim, theta, eps, cache_rows,
                cache_rows, row_stride, pos, poison,
            )
        };
        let fused = go(&on);
        let split = go(&off);
        let written = &split[pos * row_stride..(pos + rows) * row_stride];
        assert!(
            maxabs(written) > 1e-3,
            "kv_write fold [rows={rows} pos={pos}] reference row is all-zero — test is vacuous"
        );
        assert!(
            written.iter().all(|v| (v - poison).abs() > 1e-6),
            "kv_write fold [rows={rows} pos={pos}] control left poison INSIDE the written row — \
             the un-fused write did not fill it, so the comparison proves nothing"
        );
        assert_eq!(
            fused,
            split,
            "the fused K write is not bit-identical to `qk_norm_rope`+`write_kv` at rows={rows} \
             pos={pos} (max_err={:e}) — wrong row, wrong rounding, or a partially written row",
            maxerr(&fused, &split)
        );
        // ...and every row outside [pos, pos+rows) must still be poison in the FUSED run. The `==`
        // above already implies it, but stated separately this is the assertion that fails loudly
        // when the row arithmetic is mutated (`kv_row + r` → `r`, `* kv_stride` dropped, …).
        for r in 0..cache_rows {
            if (pos..pos + rows).contains(&r) {
                continue;
            }
            let row = &fused[r * row_stride..(r + 1) * row_stride];
            assert!(
                row.iter().all(|v| (v - poison).abs() < 1e-6),
                "the fused K write corrupted cache row {r} (wrote at rows {pos}..{} for rows={rows})",
                pos + rows
            );
        }
    }
}

/// **A K row written at a non-zero position is the row attention reads back there.** The case above
/// compares two ROCm runs against each other; this one pins the ABSOLUTE row against the CPU
/// reference, so a row offset that is wrong in the same way on both sides of that A/B cannot
/// survive. The CPU backend runs the identical two-op graph (it never fuses), and the comparison is
/// the whole cache — poison rows included.
#[test]
#[ignore = "requires a ROCm GPU"]
fn kv_write_fold_lands_on_the_row_the_cpu_writes() {
    let Some(on) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (n_head, head_dim, rope_dim) = (4usize, 64usize, 64usize);
    let (cache_rows, theta, eps, poison) = (16usize, 10000.0f32, 1e-6f32, -7.5f32);
    let row_stride = n_head * head_dim;
    let w = f16_norm_weight(head_dim);
    for (rows, pos) in [(1usize, 9usize), (3, 6)] {
        let x = gen(rows * n_head * head_dim, 7 + pos);
        let positions: Vec<i32> = (0..rows).map(|r| (pos + r) as i32).collect();
        let go = |b: &dyn Backend| {
            run_qk_norm_rope_write_kv(
                b, &x, &w, &positions, rows, n_head, head_dim, rope_dim, theta, eps, cache_rows,
                cache_rows, row_stride, pos, poison,
            )
        };
        let gpu = go(&on);
        let cpu_out = go(&cpu);
        let e = maxerr(&gpu, &cpu_out);
        let mag = maxabs(&cpu_out).max(1e-6);
        assert!(
            maxabs(&cpu_out) > 1e-3,
            "kv_write row reference [rows={rows} pos={pos}] is all-zero — test is vacuous"
        );
        assert!(
            e / mag < 1e-3,
            "the fused K write does not land where the CPU writes [rows={rows} pos={pos}]: \
             abs={e:e} rel={:e}",
            e / mag
        );
    }
}

/// **The hatch really disables the fold, and the K row still gets written.** `fuse_kv_write = false`
/// stops the shared pass planning this pattern at all, so nothing is elided and the standalone
/// `write_kv` runs — the A/B control the two `==` cases above depend on. (The *decline* path, where
/// the pass DID plan the fold and `kv_fuse_ok` rejected it, must additionally un-skip the elided
/// op; that is pinned by `kv_write_fold_declines_a_row_the_write_path_would_not_wrap`, which fails
/// if the un-skip is dropped.)
#[test]
#[ignore = "requires a ROCm GPU"]
fn kv_write_fold_hatch_leaves_the_standalone_write() {
    let Some(off) = rocm_cfg(|c| c.fuse_kv_write = false) else {
        return;
    };
    let (rows, n_head, head_dim, rope_dim, pos) = (1usize, 4usize, 64usize, 64usize, 11usize);
    let poison = -7.5f32;
    let w = f16_norm_weight(head_dim);
    let x = gen(rows * n_head * head_dim, 3);
    let out = run_qk_norm_rope_write_kv(
        &off,
        &x,
        &w,
        &[pos as i32],
        rows,
        n_head,
        head_dim,
        rope_dim,
        10000.0,
        1e-6,
        16,
        16,
        n_head * head_dim,
        pos,
        poison,
    );
    let row_stride = n_head * head_dim;
    let written = &out[pos * row_stride..(pos + rows) * row_stride];
    assert!(
        written.iter().any(|v| (v - poison).abs() > 1e-6),
        "with the fold hatched off nothing wrote the K row — cache row {pos} is still poison"
    );
}

/// **ROCm does NOT wrap, and the fold must not start.** `kv_swa_ring: false` means the seam gives
/// this backend full-context caches and `write_kv` indexes `pos + row` with no modulo at all — but
/// the SHARED plan hands back `pos % cap_rows` regardless, because Vulkan's ring caches need it.
/// `kv_fuse_ok` therefore declines any entry whose planned row is not the raw `pos`.
///
/// The shape is built by hand: the cache is DECLARED with 8 rows (so the shared pass plans row
/// `11 % 8 == 3`) and ALLOCATED with 16 (so the un-wrapped write at row 11 lands inside real
/// memory). Row 11 must hold the K row and row 3 must still be poison — i.e. the fused path is
/// bit-identical to the write path it replaces *including where that write path is unusual*.
/// Deleting the `row != pos` gate inverts both assertions.
#[test]
#[ignore = "requires a ROCm GPU"]
fn kv_write_fold_declines_a_row_the_write_path_would_not_wrap() {
    let Some(on) = rocm() else {
        return;
    };
    let (rows, n_head, head_dim, rope_dim, pos) = (1usize, 4usize, 64usize, 64usize, 11usize);
    let (decl_rows, alloc_rows, poison) = (8usize, 16usize, -7.5f32);
    let row_stride = n_head * head_dim;
    let w = f16_norm_weight(head_dim);
    let x = gen(rows * n_head * head_dim, 3);
    let out = run_qk_norm_rope_write_kv(
        &on,
        &x,
        &w,
        &[pos as i32],
        rows,
        n_head,
        head_dim,
        rope_dim,
        10000.0,
        1e-6,
        alloc_rows,
        decl_rows,
        row_stride,
        pos,
        poison,
    );
    let wrapped = pos % decl_rows;
    assert!(
        out[pos * row_stride..(pos + 1) * row_stride]
            .iter()
            .any(|v| (v - poison).abs() > 1e-6),
        "the K row did not land at the un-wrapped row {pos} — the fold took a ring row ROCm's \
         `write_kv` has never taken"
    );
    assert!(
        out[wrapped * row_stride..(wrapped + 1) * row_stride]
            .iter()
            .all(|v| (v - poison).abs() < 1e-6),
        "the fold wrapped the write to row {wrapped}; ROCm's `write_kv` does not wrap, so this \
         silently corrupts whatever position actually lives there"
    );
}

/// **A rope whose `dst` is still read after the write does not fuse.** The shared plan carries no
/// live-range bound for this pattern (Vulkan's record-once decode REQUIRES its K write fused, so it
/// cannot afford one), but eliding the write leaves the rope's `dst` scratch NEVER WRITTEN on ROCm
/// — the executor would hand a later reader a freshly zeroed buffer. `kv_fuse_ok` gate 4 declines
/// the fold instead. With the gate removed, `out` reads back all zeros.
#[test]
#[ignore = "requires a ROCm GPU"]
fn kv_write_fold_declines_when_the_rope_dst_is_read_again() {
    let Some(be) = rocm() else {
        return;
    };
    let (rows, n_head, head_dim, rope_dim, pos) = (1usize, 4usize, 64usize, 64usize, 5usize);
    let (cache_rows, n) = (16usize, n_head * head_dim);
    let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
    let w = f16_norm_weight(head_dim);
    let x = gen(rows * n, 3);

    let mut g = Graph::new();
    let xid = g.input(f32d(x.len()));
    let wid = g.weight(TensorDesc::new(vec![head_dim], DType::F16));
    let pid = g.input(TensorDesc::new(vec![1], DType::I32));
    let k16 = g.internal(f16d(rows * n));
    let cache = g.input(f16d(cache_rows * n));
    let out = g.output(f32d(rows * n));
    g.push(Op::QkNormRope {
        x: xid,
        weight: wid,
        positions: pid,
        dst: k16,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        rope_dim: rope_dim as u32,
        theta: 10000.0,
        eps: 1e-6,
        freq_factors: None,
        x_stride: 0,
    });
    g.push(Op::WriteKv {
        src: k16,
        cache,
        rows: rows as u32,
        row_stride: n as u32,
        pos: pos as u32,
    });
    // The extra reader: without gate 4 this copies a scratch the fused rope never wrote.
    g.push(Op::Copy {
        src: k16,
        src_off: 0,
        dst: out,
        dst_off: 0,
        n: (rows * n) as u32,
    });
    let plan = be.compile(&g).expect("compile");

    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).expect("x");
    be.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let wb = be.alloc(w.len(), BufferUsage::Weights).expect("w");
    be.upload(wb.as_ref(), &w).unwrap();
    let pv = [pos as i32];
    let pbytes: &[u8] = bytemuck::cast_slice(&pv);
    let pb = be
        .alloc(pbytes.len(), BufferUsage::Activations)
        .expect("pos");
    be.upload(pb.as_ref(), pbytes).unwrap();
    let cb = be
        .alloc(cache_rows * n * 2, BufferUsage::KvCache)
        .expect("cache");
    let ob = be.alloc(rows * n * 4, BufferUsage::Readback).expect("out");
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(pid, pb.as_ref());
    b.bind(cache, cb.as_ref());
    b.bind(out, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; rows * n];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    assert!(
        maxabs(&o) > 1e-3,
        "the rope's `dst` read back as all zeros — the fold elided the write of a scratch that is \
         still live (live-range gate dropped)"
    );
}

/// **A rope that does not tile the cache row exactly does not fuse.** The elided `write_kv` copies
/// `row_stride` elements per row; the fused kernel instead writes `n_head * head_dim` of them,
/// because that is the grid it runs. When the two differ, fusing either leaves the tail of the row
/// unwritten or — as here, with a cache row NARROWER than the rope output — runs off the end of the
/// row and into the NEXT position's slot, which is the exact "corrupts an unrelated position"
/// failure a greedy check cannot see. `kv_fuse_ok` gate 2 declines the shape instead.
///
/// Nothing the seam emits has `row_stride != n_kv * head_dim`, so this is built by hand: a 4×64
/// rope feeding a cache whose declared row holds only 128 elements. With the gate, row `pos + 1`
/// stays poison; without it, the fused write spills into it.
#[test]
#[ignore = "requires a ROCm GPU"]
fn kv_write_fold_declines_a_row_stride_it_would_overrun() {
    let Some(on) = rocm() else {
        return;
    };
    let (rows, n_head, head_dim, rope_dim, pos) = (1usize, 4usize, 64usize, 64usize, 5usize);
    let (cache_rows, row_stride, poison) = (16usize, 128usize, -7.5f32);
    let w = f16_norm_weight(head_dim);
    let x = gen(rows * n_head * head_dim, 3);
    let out = run_qk_norm_rope_write_kv(
        &on,
        &x,
        &w,
        &[pos as i32],
        rows,
        n_head,
        head_dim,
        rope_dim,
        10000.0,
        1e-6,
        cache_rows,
        cache_rows,
        row_stride,
        pos,
        poison,
    );
    assert!(
        out[pos * row_stride..(pos + 1) * row_stride]
            .iter()
            .any(|v| (v - poison).abs() > 1e-6),
        "the declined fold did not replay the elided `WriteKv` — cache row {pos} is still poison"
    );
    assert!(
        out[(pos + 1) * row_stride..(pos + 2) * row_stride]
            .iter()
            .all(|v| (v - poison).abs() < 1e-6),
        "the fused write overran its cache row into row {} — a row stride the fused kernel does \
         not tile must not fuse",
        pos + 1
    );
}

// ── P1: the tiled flash PREFILL attention kernel ─────────────────────────────
//
// `attention_prefill_flash` replaces the single-wave scalar `attention` on every `rows > 1`
// dispatch, and it is the ONE op in this backend where a wrong answer is invisible until long
// context: the plain kernel's bug surface is a head index, while a tiled one adds a query-tile
// origin, a KV-tile origin, a workgroup-uniform key range derived from the mask, a padded LDS
// stride, and an online-softmax rescale. All of those are exercised only when there is more than
// one tile in each direction, so every case below runs multiple tiles, and two of them run at
// 2048 keys.
//
// Two independent gates, because they fail differently:
//   * vs the CPU reference — catches a wrong ABSOLUTE row/key/head, which an A/B against another
//     ROCm kernel cannot see if both are wrong the same way;
//   * vs the plain kernel through `kernels.rocm.attn_flash = false` — the SAME graph, both ways,
//     so "the tiling reproduces what it replaced" is a checked property and not a comment.

/// Deterministic f16 KV-cache bytes, small magnitude and NOT symmetric across the head dim (a
/// palindromic cache would hide a mirrored `d` index).
fn f16_kv(n: usize, salt: usize) -> Vec<u8> {
    let mut v = vec![0u8; n * 2];
    for i in 0..n {
        let f = (((i * 7 + salt * 13) % 23) as f32 - 11.0) * 0.045 + (i % 5) as f32 * 0.01;
        v[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(f).to_bits().to_le_bytes());
    }
    v
}

/// Run a single-`Op::Attention` graph on `be` and return the downloaded f32 output.
#[allow(clippy::too_many_arguments)]
fn run_attention(
    be: &dyn Backend,
    q: &[f32],
    k_bytes: &[u8],
    v_bytes: &[u8],
    rows: usize,
    kv_len: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    mask: AttnMask,
    pos: usize,
) -> Vec<f32> {
    let f16d = |n: usize| TensorDesc::new(vec![n], DType::F16);
    let mut g = Graph::new();
    let qid = g.input(f32d(q.len()));
    let kid = g.input(f16d(kv_len * n_kv * head_dim));
    let vid = g.input(f16d(kv_len * n_kv * head_dim));
    let dst = g.output(f32d(rows * n_head * head_dim));
    g.push(Op::Attention {
        q: qid,
        k_cache: kid,
        v_cache: vid,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: n_head as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        mask,
        pos: pos as u32,
    });
    let plan = be.compile(&g).expect("compile");
    let qb = be.alloc(q.len() * 4, BufferUsage::Activations).expect("q");
    be.upload(qb.as_ref(), bytemuck::cast_slice(q)).unwrap();
    let kb = be.alloc(k_bytes.len(), BufferUsage::KvCache).expect("k");
    be.upload(kb.as_ref(), k_bytes).unwrap();
    let vb = be.alloc(v_bytes.len(), BufferUsage::KvCache).expect("v");
    be.upload(vb.as_ref(), v_bytes).unwrap();
    let ob = be
        .alloc(rows * n_head * head_dim * 4, BufferUsage::Readback)
        .expect("out");
    let mut b = Bindings::new();
    b.bind(qid, qb.as_ref());
    b.bind(kid, kb.as_ref());
    b.bind(vid, vb.as_ref());
    b.bind(dst, ob.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; rows * n_head * head_dim];
    be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

/// Every prefill shape this slice cares about, as
/// `(rows, kv_len, n_head, n_kv, head_dim, mask, pos)`. Each entry is here for one reason:
///
/// | case | what only it can break |
/// | ---- | ---------------------- |
/// | 512×512 GQA d=128 | the shipping qwen3 prefill shape; `n_kv != n_head` pins the GQA head map |
/// | 2048×2048 d=128 | LONG CONTEXT — 128 query tiles × 64 key tiles, where a tile-origin slip first shows |
/// | 2048×2048 SWA(384) d=256 | the SWA lower bound `j_lo` AND the wide (gemma-3) head dim, which takes the `bc = 16` tiling |
/// | 300×300 | rows NOT a multiple of `br` and kv NOT a multiple of `bc` — the two ragged tails |
/// | 40×160 pos=120 | a non-zero `pos`: the causal diagonal is offset from the tile origin |
/// | 96×96 Canvas | the third mask arm, whose bound is a lower one like SWA's but unconditional |
#[allow(clippy::type_complexity)]
const ATTN_FLASH_CASES: &[(usize, usize, usize, usize, usize, AttnMask, usize)] = &[
    (512, 512, 8, 2, 128, AttnMask::Causal, 0),
    (2048, 2048, 4, 4, 128, AttnMask::Causal, 0),
    (2048, 2048, 2, 1, 256, AttnMask::SlidingWindow(384), 0),
    (300, 300, 4, 2, 128, AttnMask::Causal, 0),
    (40, 160, 4, 2, 64, AttnMask::Causal, 120),
    (96, 96, 2, 2, 128, AttnMask::Canvas { lo: 17 }, 0),
];

/// **The flash prefill kernel matches the CPU reference on every routed prefill shape**, including
/// two at 2048 keys and one with a sliding window. Relative tolerance rather than equality: the
/// tiled kernel accumulates the softmax ONLINE (a running max, rescaled per KV tile) where the
/// reference takes a global max first, so the two agree to fp rounding, not to the bit.
#[test]
#[ignore = "requires a ROCm GPU"]
fn attention_prefill_flash_matches_cpu() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    for &(rows, kv_len, n_head, n_kv, head_dim, mask, pos) in ATTN_FLASH_CASES {
        let q = gen(rows * n_head * head_dim, rows + head_dim);
        let k = f16_kv(kv_len * n_kv * head_dim, 1);
        let v = f16_kv(kv_len * n_kv * head_dim, 2);
        let go = |b: &dyn Backend| {
            run_attention(
                b, &q, &k, &v, rows, kv_len, n_head, n_kv, head_dim, mask, pos,
            )
        };
        let gpu = go(&be);
        let want = go(&cpu);
        let mag = maxabs(&want);
        assert!(
            mag > 1e-3,
            "attention reference is all-zero for {rows}x{kv_len} h={n_head}/{n_kv} \
             d={head_dim} {mask:?} pos={pos} — the case is vacuous"
        );
        assert!(
            gpu.iter().all(|v| v.is_finite()),
            "flash prefill attention produced a non-finite value for {rows}x{kv_len} \
             h={n_head}/{n_kv} d={head_dim} {mask:?} pos={pos}"
        );
        let e = maxerr(&gpu, &want);
        println!(
            "attn-flash {rows}x{kv_len} h={n_head}/{n_kv} d={head_dim} {mask:?} pos={pos}: \
             max_err={e:e} max|ref|={mag:e} rel={:e}",
            e / mag
        );
        assert!(
            e / mag < 1e-4,
            "flash prefill attention diverged from the CPU reference for {rows}x{kv_len} \
             h={n_head}/{n_kv} d={head_dim} {mask:?} pos={pos}: abs={e:e} rel={:e}",
            e / mag
        );
    }
}

/// **The tiling reproduces the single-wave kernel it replaces.** `kernels.rocm.attn_flash = false`
/// routes the identical graph back to the plain `attention`, so this is the A/B control for the
/// case above — and it is the one that fails if a future edit to the tiling breaks a shape the CPU
/// oracle is too slow to be run against at every size.
#[test]
#[ignore = "requires a ROCm GPU"]
fn attention_prefill_flash_matches_the_single_wave_kernel() {
    let (Some(on), Some(off)) = (rocm(), rocm_cfg(|c| c.attn_flash = false)) else {
        return;
    };
    for &(rows, kv_len, n_head, n_kv, head_dim, mask, pos) in ATTN_FLASH_CASES {
        let q = gen(rows * n_head * head_dim, rows + head_dim);
        let k = f16_kv(kv_len * n_kv * head_dim, 1);
        let v = f16_kv(kv_len * n_kv * head_dim, 2);
        let go = |b: &dyn Backend| {
            run_attention(
                b, &q, &k, &v, rows, kv_len, n_head, n_kv, head_dim, mask, pos,
            )
        };
        let flash = go(&on);
        let plain = go(&off);
        let e = maxerr(&flash, &plain);
        let mag = maxabs(&plain).max(1e-6);
        assert!(
            e / mag < 1e-4,
            "flash prefill attention diverged from the plain kernel for {rows}x{kv_len} \
             h={n_head}/{n_kv} d={head_dim} {mask:?} pos={pos}: abs={e:e} rel={:e}",
            e / mag
        );
    }
}

/// **A shape the tiling cannot serve must fall back, not run wrong.** `head_dim = 20` is not a
/// multiple of 8, so `attn_flash_tiling` declines it and the plain kernel runs; the answer must
/// still be the CPU's. (This is the gate that turns "the routing predicate is right" into
/// something a test fails on, rather than a comment above a `%`.)
#[test]
#[ignore = "requires a ROCm GPU"]
fn attention_prefill_falls_back_for_an_untileable_head_dim() {
    let Some(be) = rocm() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, kv_len, n_head, n_kv, head_dim) = (48usize, 48usize, 3usize, 3usize, 20usize);
    let q = gen(rows * n_head * head_dim, 5);
    let k = f16_kv(kv_len * n_kv * head_dim, 1);
    let v = f16_kv(kv_len * n_kv * head_dim, 2);
    let go = |b: &dyn Backend| {
        run_attention(
            b,
            &q,
            &k,
            &v,
            rows,
            kv_len,
            n_head,
            n_kv,
            head_dim,
            AttnMask::Causal,
            0,
        )
    };
    let gpu = go(&be);
    let want = go(&cpu);
    let e = maxerr(&gpu, &want);
    let mag = maxabs(&want).max(1e-6);
    assert!(
        e / mag < 1e-4,
        "the head_dim={head_dim} fallback diverged from the CPU reference: abs={e:e} rel={:e}",
        e / mag
    );
}

/// **The score reduction TREE is the plain kernel's, and the seam goldens depend on it.** This is
/// the tightest statement that is actually true, and the gap between it and the 1e-4 oracle gates
/// above is not cosmetic.
///
/// The plain kernel splits a head dim across the 32 lanes, chains an FMA per lane, then butterfly
/// all-reduces. The flash kernel gives ONE lane the whole dot, so it rebuilds that exact tree in
/// registers (`g[t]` is lane `t`'s partial; the reduction pairs `t` with `t+16`, then `t+8`, …).
/// Writing the dot the obvious way instead — a serial sum — is only ~1e-7 off, passes every
/// tolerance gate in this file, and STILL flipped a near-tie argmax fourteen tokens into the Q8_0
/// seam run ("I know" → "I remember"). A tolerance alone cannot separate those two: measured on
/// these shapes the tree is 5.6e-8 relative and the serial sum 1.1e-7, a factor of two.
///
/// What DOES separate them is how many outputs land EXACTLY. Matching the tree leaves ≤6% of the
/// output differing from the plain kernel at all; a serial sum leaves 25–41%. So that is the
/// assertion, on shapes where the tiled softmax contributes nothing — the whole kv range fits ONE
/// key tile, so the online rescale is a multiply by exactly zero against a zero accumulator, and
/// the denominator and P·V sums both run ascending in `j` exactly as the plain kernel's do.
///
/// It is not bit-EQUALITY, and that is a property of the REFERENCE rather than of the tiling: the
/// plain kernel evaluates the same `q·k` dot TWICE — once for the softmax max, once for the
/// weighted sum — and nothing makes the compiler round two separately-scheduled copies of that
/// loop identically, so even its own `w` for a single-key row is not exactly 1. The single-pass
/// kernel is the self-consistent one here; `attention_prefill_flash_matches_cpu` is what pins it
/// to the oracle.
#[test]
#[ignore = "requires a ROCm GPU"]
fn attention_prefill_flash_keeps_the_plain_kernels_score_tree() {
    let (Some(on), Some(off)) = (rocm(), rocm_cfg(|c| c.attn_flash = false)) else {
        return;
    };
    // `bc` is 32 for head_dim <= 128 and 16 above it, so 16 keys is one tile for either tiling.
    for &(rows, kv_len, n_head, n_kv, head_dim, mask, pos) in &[
        (
            24usize,
            16usize,
            4usize,
            2usize,
            128usize,
            AttnMask::Causal,
            0usize,
        ),
        (24, 16, 2, 1, 256, AttnMask::Causal, 0),
        (16, 16, 4, 4, 64, AttnMask::SlidingWindow(8), 0),
        (9, 16, 3, 3, 128, AttnMask::Causal, 0),
    ] {
        let q = gen(rows * n_head * head_dim, rows + head_dim);
        let k = f16_kv(kv_len * n_kv * head_dim, 1);
        let v = f16_kv(kv_len * n_kv * head_dim, 2);
        let go = |b: &dyn Backend| {
            run_attention(
                b, &q, &k, &v, rows, kv_len, n_head, n_kv, head_dim, mask, pos,
            )
        };
        let flash = go(&on);
        let plain = go(&off);
        let differing = flash
            .iter()
            .zip(&plain)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let frac = differing as f32 / flash.len() as f32;
        let e = maxerr(&flash, &plain);
        let mag = maxabs(&plain).max(1e-6);
        println!(
            "attn-flash tree {rows}x{kv_len} h={n_head}/{n_kv} d={head_dim} {mask:?}: \
             inexact {differing}/{} ({:.1}%) max_err={e:e} rel={:e}",
            flash.len(),
            frac * 100.0,
            e / mag
        );
        assert!(
            frac < 0.15 && e / mag < 1e-7,
            "flash prefill attention no longer reproduces the plain kernel's score reduction \
             tree for {rows}x{kv_len} h={n_head}/{n_kv} d={head_dim} {mask:?} pos={pos}: \
             {differing}/{} outputs inexact ({:.1}%), rel={:e} — a serial dot lands at 25-41% \
             and 1.1e-7, and it moves the seam goldens",
            flash.len(),
            frac * 100.0,
            e / mag
        );
    }
}
