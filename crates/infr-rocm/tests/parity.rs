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
use infr_core::graph::{Activation, Graph, MoeGating, Op};
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
