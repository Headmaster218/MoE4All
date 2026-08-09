//! Numeric-parity tests: run the SAME graph op on `infr-cpu` (the trusted reference interpreter)
//! and on `infr-metal`, assert the outputs match. This is the contract a backend must satisfy.
//!
//! macOS-only (the backend is), and each test is `#[ignore]`d — it needs a real Metal device, like
//! the Vulkan GPU tests. Run them with `cargo test -p infr-metal -- --include-ignored`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_cpu::CpuBackend;
use infr_metal::MetalBackend;

// ---- deterministic test data (LCG, no rng dependency) ----
fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}
fn rand_f32(n: usize, mut seed: u64) -> Vec<f32> {
    (0..n)
        .map(|_| ((lcg(&mut seed) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// Bind raw byte buffers to graph handles, run one graph on `be`, and read back `out` as f32.
fn run(
    be: &dyn Backend,
    g: &Graph,
    bound: &[(TensorId, Vec<u8>)],
    out: TensorId,
    out_n: usize,
) -> Vec<f32> {
    let mut bufs: Vec<(TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in bound {
        let b = be
            .alloc(bytes.len().max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    let ob = be.alloc(out_n * 4, BufferUsage::Activations).unwrap();
    bufs.push((out, ob));

    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    be.sync().unwrap();

    let ob = &bufs.iter().find(|(i, _)| *i == out).unwrap().1;
    let mut bytes = vec![0u8; out_n * 4];
    be.download(ob.as_ref(), &mut bytes).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

/// Run one graph on `be`, then download raw bytes of the bound tensor `readback` (used for
/// stateful ops like WriteKv that mutate a bound buffer instead of producing an Output).
fn run_readback(
    be: &dyn Backend,
    g: &Graph,
    bound: &[(TensorId, Vec<u8>)],
    readback: TensorId,
    byte_len: usize,
) -> Vec<u8> {
    let mut bufs: Vec<(TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in bound {
        let b = be
            .alloc(bytes.len().max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    be.sync().unwrap();
    let rb = &bufs.iter().find(|(i, _)| *i == readback).unwrap().1;
    let mut bytes = vec![0u8; byte_len];
    be.download(rb.as_ref(), &mut bytes).unwrap();
    bytes
}

/// Run one graph on `be` and read back several tensors as f32 (Outputs, or mutated f32 Inputs like
/// recurrent state). Any `read` id not present in `bound` is allocated zeroed and bound.
fn run_multi(
    be: &dyn Backend,
    g: &Graph,
    bound: &[(TensorId, Vec<u8>)],
    reads: &[(TensorId, usize)],
) -> Vec<Vec<f32>> {
    let mut bufs: Vec<(TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in bound {
        let b = be
            .alloc(bytes.len().max(4), BufferUsage::Activations)
            .unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    for (id, n) in reads {
        if !bound.iter().any(|(bid, _)| bid == id) {
            let b = be.alloc(n * 4, BufferUsage::Activations).unwrap();
            bufs.push((*id, b));
        }
    }
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(g).unwrap();
    be.execute(plan.as_ref(), &binds).unwrap();
    be.sync().unwrap();
    reads
        .iter()
        .map(|(id, n)| {
            let b = &bufs.iter().find(|(i, _)| i == id).unwrap().1;
            let mut bytes = vec![0u8; n * 4];
            be.download(b.as_ref(), &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        })
        .collect()
}

fn assert_close(cpu: &[f32], mtl: &[f32], tol: f32, what: &str) {
    assert_eq!(cpu.len(), mtl.len(), "{what}: length");
    for (i, (c, m)) in cpu.iter().zip(mtl.iter()).enumerate() {
        let err = (c - m).abs() / c.abs().max(1.0);
        assert!(
            err <= tol,
            "{what} elem {i}: cpu={c} metal={m} err={err} > {tol}"
        );
    }
}

/// Run on both backends and assert close.
fn assert_parity(g: &Graph, bound: &[(TensorId, Vec<u8>)], out: TensorId, out_n: usize, tol: f32) {
    let cpu = run(&CpuBackend::new(), g, bound, out, out_n);
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        g,
        bound,
        out,
        out_n,
    );
    assert_eq!(cpu.len(), mtl.len());
    for (i, (c, m)) in cpu.iter().zip(mtl.iter()).enumerate() {
        let err = (c - m).abs() / c.abs().max(1.0);
        assert!(
            err <= tol,
            "elem {i}: cpu={c} metal={m} rel_err={err} > tol={tol}"
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn add_parity() {
    let n = 4096usize;
    let mut g = Graph::new();
    let a = g.input(TensorDesc::new(vec![n], DType::F32));
    let b = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Add {
        a,
        b,
        dst,
        n: n as u32,
    });
    let bound = vec![
        (a, f32_bytes(&rand_f32(n, 1))),
        (b, f32_bytes(&rand_f32(n, 2))),
    ];
    assert_parity(&g, &bound, dst, n, 0.0);
}

// Broadcast bias add (Qwen2/2.5 q/k/v `Wx + b`): `dst[r*n+c] = x[r*n+c] + bias[c]`. `bias` is a
// bound weight; `n=7` (not a 64-wide-workgroup multiple) exercises the `% n` broadcast + the tail.
// Exact (both backends do f32 x + f32 bias), so tol 0.
#[test]
#[ignore = "requires a Metal GPU"]
fn add_bias_parity() {
    let (rows, n) = (5usize, 7usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, n], DType::F32));
    let bias = g.weight(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, n], DType::F32));
    g.push(Op::AddBias {
        x,
        bias,
        dst,
        rows: rows as u32,
        n: n as u32,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * n, 71))),
        (bias, f32_bytes(&rand_f32(n, 72))),
    ];
    assert_parity(&g, &bound, dst, rows * n, 0.0);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn scale_parity() {
    let n = 4096usize;
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Scale {
        x,
        dst,
        s: 0.125,
        n: n as u32,
    });
    let bound = vec![(x, f32_bytes(&rand_f32(n, 3)))];
    assert_parity(&g, &bound, dst, n, 0.0);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn softcap_parity() {
    let n = 4096usize;
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Softcap {
        x,
        dst,
        cap: 30.0,
        n: n as u32,
    });
    // scale inputs up so tanh saturation is exercised
    let xs: Vec<f32> = rand_f32(n, 4).iter().map(|v| v * 60.0).collect();
    let bound = vec![(x, f32_bytes(&xs))];
    assert_parity(&g, &bound, dst, n, 1e-5);
}

// naive reference matmul: dst[r,o] = sum_i x[r,i] * w[o,i]   (w row-major [out_f, in_f])
fn ref_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * out_f];
    for r in 0..m {
        for o in 0..out_f {
            let mut acc = 0f32;
            for i in 0..in_f {
                acc += x[r * in_f + i] * w[o * in_f + i];
            }
            out[r * out_f + o] = acc;
        }
    }
    out
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}

// Quantize a whole row-major weight to GGUF Q8_0 (32-elem blocks: f16 scale + 32×i8).
fn quantize_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in w.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in blk {
            let q = if d > 0.0 {
                (v / d).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            out.push(q as u8);
        }
    }
    out
}

// Well-formed Q5_0 blocks (22 B / 32 elems: [f16 d][4 B qh][16 B nibbles]) — like the k-quant
// synths, any nibble/bit payload decodes to finite values; only d must be a sane f16.
fn synth_q5_0(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 32, 0, "Q5_0 blocks are 32 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 22];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes());
        blk[2..22].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 20));
        out.extend_from_slice(&blk);
    }
    out
}

// Well-formed Q4_1 blocks (20 B / 32 elems: [f16 d][f16 m][16 B nibbles]) — the AFFINE sibling of
// Q4_0 (value = d*q + m). Any nibble payload decodes to finite values; only d and m must be sane
// f16. The parity test compares Metal's Linear against a reference dequant of these SAME bytes.
fn synth_q4_1(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 32, 0, "Q4_1 blocks are 32 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 20];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes()); // d
        blk[2..4].copy_from_slice(&half::f16::from_f32(-0.3).to_le_bytes()); // m
        blk[4..20].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 16)); // nibbles
        out.extend_from_slice(&blk);
    }
    out
}

// Well-formed Q5_1 blocks (24 B / 32 elems: [f16 d][f16 m][4 B qh][16 B nibbles]) — the AFFINE
// sibling of Q5_0 (value = d*q + m, 5-bit code). Any qh/nibble payload decodes to finite values.
fn synth_q5_1(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 32, 0, "Q5_1 blocks are 32 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 24];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.04).to_le_bytes()); // d
        blk[2..4].copy_from_slice(&half::f16::from_f32(-0.3).to_le_bytes()); // m
        blk[4..24].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 20)); // qh + nibbles
        out.extend_from_slice(&blk);
    }
    out
}

// Well-formed MXFP4 blocks (17 B / 32 elems: [u8 E8M0 exponent][16 B nibbles]). The E8M0 byte is
// a shared exponent `d = 2^(e-128)`; keeping e ∈ {124..132} bounds d to 2^-4..2^4 so the decoded
// codebook×scale products stay well inside f32 (no overflow) while still exercising the E8M0 decode
// across the band. Any nibble payload decodes to finite values (KVALUES_MXFP4 codebook). Parity
// compares Metal's Linear against dequant of these SAME bytes.
fn synth_mxfp4(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 32, 0, "MXFP4 blocks are 32 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 17];
        blk[0] = 124 + (blk_i % 9) as u8; // E8M0 exponent, moderate band 124..=132
        blk[1..17].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 16)); // nibbles
        out.extend_from_slice(&blk);
    }
    out
}

// Well-formed NVFP4 blocks (36 B / 64 elems: [u8 scales[4]][32 B nibbles]). The four bytes are
// UE4M3 per-16-element sub-block scales; 0x3A/0x3C/0x3E/0x40 decode to 0.625/0.75/0.875/1.0 (all
// moderate, none the 0x00/0x7F zero-flush cases), exercising four DISTINCT sub-block scales per
// block. Any nibble payload decodes to finite values (shared KVALUES_MXFP4 codebook).
fn synth_nvfp4(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 64, 0, "NVFP4 blocks are 64 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 64) {
        let mut blk = vec![0u8; 36];
        blk[0..4].copy_from_slice(&[0x3A, 0x3C, 0x3E, 0x40]); // 4 × UE4M3 sub-block scales
        blk[4..36].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 32)); // nibbles
        out.extend_from_slice(&blk);
    }
    out
}

// Well-formed Q2_0 blocks (18 B / 64 elems: [f16 d][16 B qs]) — Bonsai ternary, 2 bits/elem packed
// sequentially, value = d*(q - 1). Any qs payload decodes to finite values; only d must be a sane
// f16. Parity compares Metal's Linear against a reference dequant of these SAME bytes.
fn synth_q2_0(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 64, 0, "Q2_0 blocks are 64 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 64) {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes()); // d
        blk[2..18].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 16)); // 2-bit codes
        out.extend_from_slice(&blk);
    }
    out
}

// Well-formed TQ2_0 blocks (66 B / 256 elems: [64 B qs][f16 d]) — ternary, 2 bits/elem, value =
// d*(q - 1). Any qs payload decodes to finite values; only d must be a sane f16.
fn synth_tq2_0(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0, "TQ2_0 blocks are 256 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 66];
        blk[0..64].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 64)); // 2-bit codes
        blk[64..66].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes()); // d
        out.extend_from_slice(&blk);
    }
    out
}

// Well-formed TQ1_0 blocks (54 B / 256 elems: [48 B qs][4 B qh][f16 d]) — ternary base-3, 5 digits
// per byte, value = d*(digit - 1). Any qs/qh payload decodes to finite digits in {0,1,2}; only d
// must be a sane f16. Parity exercises the fiddly base-3 element→byte mapping against dequant.
fn synth_tq1_0(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0, "TQ1_0 blocks are 256 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 54];
        blk[0..52].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 52)); // qs[48] + qh[4]
        blk[52..54].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes()); // d
        out.extend_from_slice(&blk);
    }
    out
}

// A deterministic LCG byte stream — arbitrary but reproducible payload for the k-quant nibble
// fields (which decode to finite values for *any* byte pattern).
fn lcg_bytes(mut seed: u32, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect()
}

// Synthesize a well-formed GGUF Q4_K weight: 144-byte / 256-elem blocks laid out as
// [f16 d][f16 dmin][12B scales][128B qs]. We only need *valid* bytes with finite f16 scales — not a
// faithful quantization of any target — because the parity test compares Metal's dequant against a
// reference dequant of these SAME bytes. Scale/nibble fields take an arbitrary reproducible pattern.
fn synth_q4k(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0, "Q4_K blocks are 256 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes()); // d
        blk[2..4].copy_from_slice(&half::f16::from_f32(0.10).to_le_bytes()); // dmin
        blk[4..144].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 140)); // scales + qs
        out.extend_from_slice(&blk);
    }
    out
}

// Synthesize a well-formed GGUF Q6_K weight: 210-byte / 256-elem blocks laid out as
// [128B ql][64B qh][16×i8 scales][f16 d]. Same rationale as `synth_q4k`.
fn synth_q6k(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0, "Q6_K blocks are 256 elems");
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 210];
        blk[0..208].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 208)); // ql + qh + scales
        blk[208..210].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes()); // d
        out.extend_from_slice(&blk);
    }
    out
}

// Linear (m=1, K-quant) immediately followed by a residual Add: the backend's peephole fuses the
// pair into `linear_*_add` (one dispatch, Add's dst written directly). Compare against the CPU
// reference running the UNFUSED pair — the fusion must be invisible.
fn check_linear_add_fusion(dtype: DType, wbytes: Vec<u8>, in_f: usize, out_f: usize) {
    let xs = rand_f32(in_f, 91);
    let res = rand_f32(out_f, 92);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![1, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let rt = g.input(TensorDesc::new(vec![out_f], DType::F32));
    let mid = g.internal(TensorDesc::new(vec![1, out_f], DType::F32));
    let dst = g.output(TensorDesc::new(vec![out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst: mid,
        m: 1,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    g.push(Op::Add {
        a: mid,
        b: rt,
        dst,
        n: out_f as u32,
    });
    let bound = vec![
        (x, f32_bytes(&xs)),
        (w, wbytes.clone()),
        (rt, f32_bytes(&res)),
    ];
    // Reference: dequant the SAME bytes + f32 matmul + add (the CPU backend Q8-quantizes the
    // activation for quant Linear, so it is not the oracle here — same as the other quant tests).
    let wref = infr_gguf::dequant::dequant_block(dtype, &wbytes).unwrap();
    let mut reference = ref_linear(&xs, &wref, 1, in_f, out_f);
    for (o, r) in reference.iter_mut().zip(res.iter()) {
        *o += r;
    }
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        out_f,
    );
    assert_close(&reference, &mtl, 1e-3, "linear+add fusion");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q4k_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q4K, synth_q4k(out_f * in_f, 93), in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q8_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    let wf = rand_f32(out_f * in_f, 95);
    check_linear_add_fusion(DType::Q8_0, quantize_q8_0(&wf), in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q6k_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q6K, synth_q6k(out_f * in_f, 94), in_f, out_f);
}

// Shared quant-Linear parity check: Metal dequants `wbytes` (via infr_gguf) and matmuls; compare to
// a reference that dequants the SAME bytes and matmuls — isolates Metal's quant-weight path.
fn check_quant_linear_parity(dtype: DType, wbytes: Vec<u8>, m: usize, in_f: usize, out_f: usize) {
    check_quant_linear_parity_tol(dtype, wbytes, m, in_f, out_f, 1e-3);
}

fn check_quant_linear_parity_tol(
    dtype: DType,
    wbytes: Vec<u8>,
    m: usize,
    in_f: usize,
    out_f: usize,
    tol: f32,
) {
    check_quant_linear_parity_impl(dtype, wbytes, m, in_f, out_f, tol, false);
}

fn check_quant_linear_parity_impl(
    dtype: DType,
    wbytes: Vec<u8>,
    m: usize,
    in_f: usize,
    out_f: usize,
    tol: f32,
    half_ops: bool,
) {
    use infr_gguf::dequant::dequant_block;
    let mut xs = rand_f32(m * in_f, 24);
    let mut wref = dequant_block(dtype, &wbytes).unwrap();
    // Half-fragment GEMM path (m >= 16): the kernel rounds weights and activations to f16, so
    // the reference mirrors that rounding — the comparison then checks the kernel, not f16.
    if half_ops {
        for v in xs.iter_mut().chain(wref.iter_mut()) {
            *v = half::f16::from_f32(*v).to_f32();
        }
    }
    let reference = ref_linear(&xs, &wref, m, in_f, out_f);

    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], dtype));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let bound = vec![(x, f32_bytes(&xs)), (w, wbytes)];
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        m * out_f,
    );
    for (i, (r, mm)) in reference.iter().zip(mtl.iter()).enumerate() {
        let err = (r - mm).abs() / r.abs().max(1.0);
        assert!(
            err <= tol,
            "{dtype:?} elem {i}: ref={r} metal={mm} err={err} > {tol}"
        );
    }
}

// Fused-QKV slices: several Linear ops share ONE concatenated [Σslices, in_f] weight, each
// reading its rows at `w_off` (the runner's combined-QKV shape — `Op::Linear.w_off`). Every
// slice must match a reference matmul over the dequant of just that slice's rows; the
// non-zero offsets exercise the byte-offset binds into the codes/scm/dd streams.
fn check_linear_woff(
    dtype: DType,
    wbytes: Vec<u8>,
    m: usize,
    in_f: usize,
    slices: &[usize],
    half_ops: bool,
    tol: f32,
) {
    use infr_gguf::dequant::dequant_block;
    let rows_total: usize = slices.iter().sum();
    let mut xs = rand_f32(m * in_f, 34);
    let mut wref = dequant_block(dtype, &wbytes).unwrap();
    if half_ops {
        for v in xs.iter_mut().chain(wref.iter_mut()) {
            *v = half::f16::from_f32(*v).to_f32();
        }
    }
    let be = MetalBackend::new().expect("metal backend");
    let mut row0 = 0usize;
    for &out_f in slices {
        let wslice = &wref[row0 * in_f..(row0 + out_f) * in_f];
        let reference = ref_linear(&xs, wslice, m, in_f, out_f);
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
        let w = g.weight(TensorDesc::new(vec![rows_total, in_f], dtype));
        let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
        g.push(Op::Linear {
            x,
            weight: w,
            dst,
            m: m as u32,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: (row0 * in_f) as u32,
        });
        let bound = vec![(x, f32_bytes(&xs)), (w, wbytes.clone())];
        let mtl = run(&be, &g, &bound, dst, m * out_f);
        for (i, (r, mm)) in reference.iter().zip(mtl.iter()).enumerate() {
            let err = (r - mm).abs() / r.abs().max(1.0);
            assert!(
                err <= tol,
                "{dtype:?} slice@{row0} elem {i}: ref={r} metal={mm} err={err} > {tol}"
            );
        }
        row0 += out_f;
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q8_0_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 35);
    check_linear_woff(
        DType::Q8_0,
        quantize_q8_0(&wf),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f16_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 350);
    check_linear_woff(DType::F16, f16_bytes(&wf), 1, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f32_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 353);
    check_linear_woff(DType::F32, f32_bytes(&wf), 1, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f32_cmm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 358);
    check_linear_woff(DType::F32, f32_bytes(&wf), 40, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f32_rt() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 359);
    check_linear_woff(DType::F32, f32_bytes(&wf), 4, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f32_cmm_small() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 360);
    check_linear_woff(DType::F32, f32_bytes(&wf), 8, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_bf16_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 354);
    check_linear_woff(DType::Bf16, bf16_bytes(&wf), 1, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_bf16_rt() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 355);
    check_linear_woff(DType::Bf16, bf16_bytes(&wf), 4, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_bf16_cmm_small() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 361);
    check_linear_woff(DType::Bf16, bf16_bytes(&wf), 6, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_bf16_rt_multirow() {
    let (in_f, slices) = (256usize, [96usize, 80, 80]);
    let wf = rand_f32(256 * in_f, 356);
    check_linear_woff(DType::Bf16, bf16_bytes(&wf), 32, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_bf16_cmm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 357);
    check_linear_woff(DType::Bf16, bf16_bytes(&wf), 40, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_bf16_cmm_preserves_wide_finite_weights() {
    let (m, in_f, out_f) = (16usize, 32usize, 64usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::Bf16));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let bound = vec![
        (x, f32_bytes(&vec![2.0f32.powi(-14); m * in_f])),
        (w, bf16_bytes(&vec![65536.0; out_f * in_f])),
    ];
    assert_parity(&g, &bound, dst, m * out_f, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f16_rt() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 352);
    check_linear_woff(DType::F16, f16_bytes(&wf), 4, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f16_cmm_small() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 362);
    check_linear_woff(DType::F16, f16_bytes(&wf), 8, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_f16_cmm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 351);
    check_linear_woff(DType::F16, f16_bytes(&wf), 40, in_f, &slices, false, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q8_0_coop_gemm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    let wf = rand_f32(256 * in_f, 36);
    check_linear_woff(
        DType::Q8_0,
        quantize_q8_0(&wf),
        40,
        in_f,
        &slices,
        true,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q4k_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q4K,
        synth_q4k(256 * in_f, 37),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q4k_coop_gemm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q4K,
        synth_q4k(256 * in_f, 38),
        40,
        in_f,
        &slices,
        true,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q5k_coop_gemm() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q5K,
        synth_q5k(256 * in_f, 124),
        40,
        in_f,
        &slices,
        true,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q6k_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q6K,
        synth_q6k(256 * in_f, 39),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_f32_parity() {
    let (m, in_f, out_f) = (3usize, 512usize, 200usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F32));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(m * in_f, 20))),
        (w, f32_bytes(&rand_f32(out_f * in_f, 21))),
    ];
    assert_parity(&g, &bound, dst, m * out_f, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_f16_parity() {
    let (m, in_f, out_f) = (2usize, 256usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::F16));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(m * in_f, 22))),
        (w, f16_bytes(&rand_f32(out_f * in_f, 23))),
    ];
    assert_parity(&g, &bound, dst, m * out_f, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_bf16_parity() {
    let (m, in_f, out_f) = (2usize, 256usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::Bf16));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(m * in_f, 231))),
        (w, bf16_bytes(&rand_f32(out_f * in_f, 232))),
    ];
    assert_parity(&g, &bound, dst, m * out_f, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_f16_cmm_parity() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    let wf = rand_f32(out_f * in_f, 230);
    check_quant_linear_parity_impl(DType::F16, f16_bytes(&wf), m, in_f, out_f, 1e-3, false);
}

// Quantized Linear: Metal dequants the weight to f32 (via infr_gguf) and matmuls. Compare to a
// reference that dequants the SAME bytes and matmuls — isolates Metal's quant-weight path.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q8_0_matches_dequant_reference() {
    use infr_gguf::dequant::dequant_block;
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    let xs = rand_f32(m * in_f, 24);
    let wf = rand_f32(out_f * in_f, 25);
    let wbytes = quantize_q8_0(&wf);
    let wref = dequant_block(DType::Q8_0, &wbytes).unwrap();
    let reference = ref_linear(&xs, &wref, m, in_f, out_f);

    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![m, in_f], DType::F32));
    let w = g.weight(TensorDesc::new(vec![out_f, in_f], DType::Q8_0));
    let dst = g.output(TensorDesc::new(vec![m, out_f], DType::F32));
    g.push(Op::Linear {
        x,
        weight: w,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let bound = vec![(x, f32_bytes(&xs)), (w, wbytes)];
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        m * out_f,
    );
    for (i, (r, mm)) in reference.iter().zip(mtl.iter()).enumerate() {
        let err = (r - mm).abs() / r.abs().max(1.0);
        assert!(err <= 1e-3, "elem {i}: ref={r} metal={mm} err={err}");
    }
}

// Native Q5_0: GEMV (four rows per simdgroup, out_f=94 exercises clamped tail rows), HGEMM,
// coop-GEMM, and the Linear+Add fusion — the gemma-family dominant weight format.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Q5_0, synth_q5_0(out_f * in_f, 98), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Q5_0,
        synth_q5_0(out_f * in_f, 99),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_0_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q5_0,
        synth_q5_0(out_f * in_f, 100),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q5_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q5_0, synth_q5_0(out_f * in_f, 101), in_f, out_f);
}

// Native Q4_0: quantizer + GEMV/HGEMM/coop-GEMM/add-fusion (TinyLlama-class checkpoints ship
// this format; it rode the factored path at ~6.1 bpw vs the native 4.5).
fn quantize_q4_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in w.chunks(32) {
        let amax = blk
            .iter()
            .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        let d = amax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for j in 0..16 {
            let q = |v: f32| ((v * id + 8.5) as u8).min(15);
            out.push(q(blk[j]) | (q(blk[j + 16]) << 4));
        }
    }
    out
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    let wf = rand_f32(out_f * in_f, 102);
    check_quant_linear_parity(DType::Q4_0, quantize_q4_0(&wf), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    let wf = rand_f32(out_f * in_f, 103);
    check_quant_linear_parity_impl(DType::Q4_0, quantize_q4_0(&wf), m, in_f, out_f, 1e-3, true);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_0_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    let wf = rand_f32(out_f * in_f, 104);
    check_quant_linear_parity_impl(DType::Q4_0, quantize_q4_0(&wf), m, in_f, out_f, 1e-3, true);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q4_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    let wf = rand_f32(out_f * in_f, 105);
    check_linear_add_fusion(DType::Q4_0, quantize_q4_0(&wf), in_f, out_f);
}

// Native Q4_1: AFFINE sibling of Q4_0 (value = d*q + m). GEMV (four rows per simdgroup, out_f=94
// exercises clamped tail rows), HGEMM, coop-GEMM, and the Linear+Add fusion. Reference is the CPU
// dequant of the SAME 20-byte blocks, so real Metal CI validates the DEC16_Q4_1 / linear_q4_1_body
// decode against dequantize_row_q4_1 numerically.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_1_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Q4_1, synth_q4_1(out_f * in_f, 110), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_1_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Q4_1,
        synth_q4_1(out_f * in_f, 111),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4_1_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q4_1,
        synth_q4_1(out_f * in_f, 112),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q4_1_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q4_1, synth_q4_1(out_f * in_f, 113), in_f, out_f);
}

// Native Q5_1: AFFINE sibling of Q5_0 (value = d*q + m, 5-bit code). Same coverage set — GEMV,
// HGEMM, coop-GEMM, Linear+Add fusion — validating DEC16_Q5_1 / linear_q5_1_body against
// dequantize_row_q5_1 on real Metal CI.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_1_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Q5_1, synth_q5_1(out_f * in_f, 114), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_1_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Q5_1,
        synth_q5_1(out_f * in_f, 115),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_1_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q5_1,
        synth_q5_1(out_f * in_f, 116),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q5_1_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Q5_1, synth_q5_1(out_f * in_f, 117), in_f, out_f);
}

// Native MXFP4: E8M0 shared-exponent 4-bit codebook (17 B / 32-elem blocks), the codebook sibling
// of NVFP4. Same coverage set as the other native codebook quants — GEMV (out_f=94 clamps tail
// rows), half-fragment GEMM, coop-GEMM, Linear+Add fusion — validating DEC16_MXFP4 against
// dequantize_row_mxfp4 on real Metal CI (reference dequants the SAME bytes).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_mxfp4_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Mxfp4, synth_mxfp4(out_f * in_f, 130), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_mxfp4_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Mxfp4,
        synth_mxfp4(out_f * in_f, 131),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_mxfp4_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Mxfp4,
        synth_mxfp4(out_f * in_f, 132),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_mxfp4_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Mxfp4, synth_mxfp4(out_f * in_f, 133), in_f, out_f);
}

// Native NVFP4: four UE4M3 per-16-element sub-block scales sharing MXFP4's codebook (36 B / 64-elem
// blocks). in_f must be a multiple of 64 (one 64-elem block). Same coverage set — GEMV, GEMM,
// coop-GEMM, Linear+Add fusion — validating DEC16_NVFP4 against dequantize_row_nvfp4 on real Metal.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_nvfp4_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Nvfp4, synth_nvfp4(out_f * in_f, 134), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_nvfp4_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Nvfp4,
        synth_nvfp4(out_f * in_f, 135),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_nvfp4_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Nvfp4,
        synth_nvfp4(out_f * in_f, 136),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_nvfp4_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Nvfp4, synth_nvfp4(out_f * in_f, 137), in_f, out_f);
}

// Native Q2_0: Bonsai ternary (18 B / 64-elem blocks), value = d*(q - 1). 64-elem block so in_f is
// a multiple of 64. Same coverage set — GEMV (out_f=94 clamps tail rows), half-fragment GEMM,
// coop-GEMM, Linear+Add fusion — validating DEC16_Q2_0 against dequantize_row_q2_0 on real Metal CI
// (reference dequants the SAME 18-byte blocks).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q2_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 128usize, 94usize);
    check_quant_linear_parity(DType::Q2_0, synth_q2_0(out_f * in_f, 140), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q2_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 128usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Q2_0,
        synth_q2_0(out_f * in_f, 141),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q2_0_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 320usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q2_0,
        synth_q2_0(out_f * in_f, 142),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_q2_0_parity() {
    let (in_f, out_f) = (320usize, 384usize);
    check_linear_add_fusion(DType::Q2_0, synth_q2_0(out_f * in_f, 143), in_f, out_f);
}

// Native TQ2_0: ternary 2-bit (66 B / 256-elem blocks), value = d*(q - 1). 256-elem block so in_f
// is a multiple of 256. Same coverage set validating DEC16_TQ2_0 against dequantize_row_tq2_0.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_tq2_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Tq2_0, synth_tq2_0(out_f * in_f, 144), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_tq2_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Tq2_0,
        synth_tq2_0(out_f * in_f, 145),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_tq2_0_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 512usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Tq2_0,
        synth_tq2_0(out_f * in_f, 146),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_tq2_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Tq2_0, synth_tq2_0(out_f * in_f, 147), in_f, out_f);
}

// Native TQ1_0: ternary base-3, 5 digits per byte (54 B / 256-elem blocks), value = d*(digit - 1).
// 256-elem block so in_f is a multiple of 256. The fiddliest decode — the base-3 element→byte
// mapping (three segments qs[0..32]/qs[32..48]/qh[0..4]) is validated against dequantize_row_tq1_0
// bit-for-bit on real Metal CI. Same coverage set: GEMV, GEMM, coop-GEMM, Linear+Add fusion.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_tq1_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Tq1_0, synth_tq1_0(out_f * in_f, 148), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_tq1_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    check_quant_linear_parity_impl(
        DType::Tq1_0,
        synth_tq1_0(out_f * in_f, 149),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_tq1_0_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 512usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Tq1_0,
        synth_tq1_0(out_f * in_f, 150),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_tq1_0_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Tq1_0, synth_tq1_0(out_f * in_f, 151), in_f, out_f);
}

// Native Q8_0 half-fragment GEMM (m=18 → the hmm route; out_f % 64 != 0 keeps cmm out).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q8_0_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    let wf = rand_f32(out_f * in_f, 96);
    check_quant_linear_parity_impl(DType::Q8_0, quantize_q8_0(&wf), m, in_f, out_f, 1e-3, true);
}

// Native Q8_0 GEMV (m=1, the mul_mv_q8_0 shape: FOUR rows per simdgroup; out_f=94 exercises the
// clamped tail rows of a partial 4-row group).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q8_0_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    let wf = rand_f32(out_f * in_f, 97);
    check_quant_linear_parity(DType::Q8_0, quantize_q8_0(&wf), m, in_f, out_f);
}

// Q5_K (176-byte / 256-elem blocks) rides the FACTORED path — first exercised by bartowski
// IQ4_XS mixes, which ship attn_v as Q5_K.
fn synth_q5k(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 176];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        blk[2..4].copy_from_slice(&half::f16::from_f32(0.10).to_le_bytes());
        blk[4..176].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 172));
        out.extend_from_slice(&blk);
    }
    out
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5k_matches_dequant_reference() {
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q5K, synth_q5k(out_f * in_f, 120), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5k_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q5K, synth_q5k(out_f * in_f, 121), m, in_f, out_f);
}

// Native Q5_K: m=4 uses the exact-f32 row tile; m=18 uses the f16 cooperative GEMM route. The
// fused-QKV test below covers sliced weight offsets separately.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5k_gemm_matches_dequant_reference() {
    let (in_f, out_f) = (512usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q5K,
        synth_q5k(out_f * in_f, 122),
        4,
        in_f,
        out_f,
        1e-3,
        false,
    );
    check_quant_linear_parity_impl(
        DType::Q5K,
        synth_q5k(out_f * in_f, 122),
        18,
        in_f,
        out_f,
        2.5e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_q5k_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Q5K,
        synth_q5k(256 * in_f, 123),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

// IQ4_XS is codebook (host-dequant to a cached f32 device weight on Metal); the fused-QKV
// runner slices it with w_off, so both the plain and offset routes need coverage. Valid blocks:
// 136 B / 256 elems = [f16 d][u16 scales_h][u32 scales_l... layout per gguf]; LCG payload works
// because the parity compares against dequant of the SAME bytes.
fn synth_iq4xs(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 136];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.06).to_le_bytes());
        blk[2..136].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 134));
        out.extend_from_slice(&blk);
    }
    out
}

fn synth_iq4nl(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 32, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 32) {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.004).to_le_bytes());
        blk[2..18].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 16));
        out.extend_from_slice(&blk);
    }
    out
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq4nl_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Iq4Nl, synth_iq4nl(out_f * in_f, 121), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq4nl_multirow_matches_dequant_reference() {
    let (m, in_f, out_f) = (8usize, 256usize, 94usize);
    check_quant_linear_parity(DType::Iq4Nl, synth_iq4nl(out_f * in_f, 118), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq4nl_small_multirow_matches_dequant_reference() {
    let (m, in_f, out_f) = (4usize, 256usize, 128usize);
    check_quant_linear_parity(DType::Iq4Nl, synth_iq4nl(out_f * in_f, 117), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq4nl_split_k_matches_dequant_reference() {
    let (m, in_f, out_f) = (5usize, 256usize, 64usize);
    check_quant_linear_parity_impl(
        DType::Iq4Nl,
        synth_iq4nl(out_f * in_f, 124),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_add_fusion_iq4nl_parity() {
    let (in_f, out_f) = (512usize, 384usize);
    check_linear_add_fusion(DType::Iq4Nl, synth_iq4nl(out_f * in_f, 120), in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq4xs_matches_dequant_reference() {
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Iq4Xs, synth_iq4xs(out_f * in_f, 122), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq4xs_split_k_matches_dequant_reference() {
    let (m, in_f, out_f) = (2usize, 256usize, 64usize);
    check_quant_linear_parity_impl(
        DType::Iq4Xs,
        synth_iq4xs(out_f * in_f, 125),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_iq4xs_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Iq4Xs,
        synth_iq4xs(256 * in_f, 123),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

// IQ2_XXS (2.06 bpw codebook): block = [f16 d][64 B qs], 66 B / 256 elems. Random qs bytes are
// valid — grid indices are any byte (grid has 256 entries), sign indices are 7-bit (<128). The
// native GEMV/RT/cmm/hmm decode must match `dequant_block`'s IQ2XXS_GRID lookup bit-for-bit.
fn synth_iq2xxs(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 66];
        // IQ2_XXS carries an extra per-sub-block scale up to (0.5 + 15) * 0.25 = 3.875, so a small
        // d keeps synthetic weight magnitudes realistic (≈ IQ4_XS's), testing the decode rather
        // than f32 accumulation/cancellation limits at pathologically large values.
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.015).to_le_bytes());
        blk[2..66].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 64));
        out.extend_from_slice(&blk);
    }
    out
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq2xxs_matches_dequant_reference() {
    // f32 routes: GEMV (m=1) and RT (m=2, out_f%64!=0). 5e-3: the decode is bit-exact vs
    // dequant_block (verified by inspection), but IQ2_XXS's signed grid values cancel heavily in
    // the dot product, so the f32 kernel (16-lane tree reduction) reassociates away from
    // ref_linear's sequential f32 sum more than the denser K-quants do — the benign class the
    // deep-k 2.5e-3 tolerances already document.
    for (m, in_f, out_f) in [(1usize, 256usize, 96usize), (2, 256, 96)] {
        check_quant_linear_parity_tol(
            DType::Iq2Xxs,
            synth_iq2xxs(out_f * in_f, 210),
            m,
            in_f,
            out_f,
            5e-3,
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq2xxs_gemm_matches_dequant_reference() {
    // f16 routes: cmm (m=4, out_f%64==0) and hmm (m=18). half_ops mirrors the kernel's f16 tile
    // rounding into the reference, so this checks the decode + tiling, not f16 precision.
    for (m, in_f, out_f) in [(4usize, 512usize, 128usize), (18, 512, 128)] {
        check_quant_linear_parity_impl(
            DType::Iq2Xxs,
            synth_iq2xxs(out_f * in_f, 211),
            m,
            in_f,
            out_f,
            2.5e-3,
            true,
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_iq2xxs_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Iq2Xxs,
        synth_iq2xxs(256 * in_f, 211),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

// IQ3_XXS (3.06 bpw codebook): block = [f16 d][64 B grid indices][32 B scales_and_signs], 98 B /
// 256 elems. Random bytes are valid (indices are any byte, sign indices 7-bit). Small d keeps
// magnitudes realistic — IQ3_XXS's scale reaches (0.5 + 15) * 0.5 = 7.75.
fn synth_iq3xxs(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 98];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.008).to_le_bytes());
        blk[2..98].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 96));
        out.extend_from_slice(&blk);
    }
    out
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq3xxs_matches_dequant_reference() {
    for (m, in_f, out_f) in [(1usize, 256usize, 96usize), (2, 256, 96)] {
        check_quant_linear_parity_tol(
            DType::Iq3Xxs,
            synth_iq3xxs(out_f * in_f, 310),
            m,
            in_f,
            out_f,
            5e-3,
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq3xxs_gemm_matches_dequant_reference() {
    for (m, in_f, out_f) in [(4usize, 512usize, 128usize), (18, 512, 128)] {
        check_quant_linear_parity_impl(
            DType::Iq3Xxs,
            synth_iq3xxs(out_f * in_f, 311),
            m,
            in_f,
            out_f,
            2.5e-3,
            true,
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_woff_iq3xxs_gemv() {
    let (in_f, slices) = (256usize, [128usize, 64, 64]);
    check_linear_woff(
        DType::Iq3Xxs,
        synth_iq3xxs(256 * in_f, 312),
        1,
        in_f,
        &slices,
        false,
        1e-3,
    );
}

// IQ2_XS (74 B), IQ2_S (82 B), IQ3_S (110 B) — random bytes are valid quant blocks (grid indices
// stay in range, sign indices are 7-bit / per-entry bytes). Small d keeps synthetic magnitudes
// realistic (IQ3_S's scale reaches d*(1 + 2*15) = 31*d, so it needs the smallest d).
fn synth_iq_block(n_elem: usize, seed: u32, bpb: usize, d: f32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; bpb];
        blk[0..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
        blk[2..bpb].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, bpb - 2));
        out.extend_from_slice(&blk);
    }
    out
}

fn iq_parity_suite(dtype: DType, bpb: usize, d: f32, seed: u32) {
    // f32 routes: GEMV (m=1), RT (m=2, out_f%64!=0). 5e-3 for the signed-codebook reassociation.
    for (m, in_f, out_f) in [(1usize, 256usize, 96usize), (2, 256, 96)] {
        check_quant_linear_parity_tol(
            dtype,
            synth_iq_block(out_f * in_f, seed, bpb, d),
            m,
            in_f,
            out_f,
            5e-3,
        );
    }
    // f16 routes: cmm (m=4), hmm (m=18), half_ops mirrors the kernel's f16 tile rounding.
    for (m, in_f, out_f) in [(4usize, 512usize, 128usize), (18, 512, 128)] {
        check_quant_linear_parity_impl(
            dtype,
            synth_iq_block(out_f * in_f, seed + 1, bpb, d),
            m,
            in_f,
            out_f,
            2.5e-3,
            true,
        );
    }
    // w_off (fused-QKV slices) through the GEMV.
    check_linear_woff(
        dtype,
        synth_iq_block(256 * 256, seed + 2, bpb, d),
        1,
        256,
        &[128usize, 64, 64],
        false,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq2xs_matches_dequant_reference() {
    iq_parity_suite(DType::Iq2Xs, 74, 0.015, 410);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq2s_matches_dequant_reference() {
    iq_parity_suite(DType::Iq2S, 82, 0.015, 420);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq3s_matches_dequant_reference() {
    iq_parity_suite(DType::Iq3S, 110, 0.002, 430);
}

// IQ1_S (1.56 bpw): block = [f16 d][32 B qs][16 B qh = 8 u16], 50 B / 256 elems. Every qs byte plus
// the 3 qh high-bits form an 11-bit index into IQ1S_GRID[2048], so ALL random bytes are in-range
// grid indices; per-sub-block dl (qh bits 12..14) and delta sign (qh bit 15) vary across the random
// qh. Weight is dl*(grid + delta) with an ADDITIVE delta=±0.125 — the native DEC16_IQ1S must match
// that, not a sign codebook. dl reaches d*(2*7+1)=15*d and grid values are ±1, so a small d keeps
// synthetic magnitudes realistic.
fn synth_iq1s(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 50];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.03).to_le_bytes());
        blk[2..50].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 48));
        out.extend_from_slice(&blk);
    }
    out
}

// IQ1_M (1.75 bpw): block = [32 B qs][16 B qh][8 B scales], 56 B / 256 elems. There is NO separate d
// — it is a f16 assembled from the TOP nibbles of the four u16 scales, so random scale bytes would
// yield a garbage/NaN d. Set d deliberately: its four nibbles go into the four scale-u16 top nibbles
// (bits 12..15), and the low 12 bits (the four 3-bit dl fields) plus all qs/qh (grid index + delta
// sign) are randomized. Every grid index is 11-bit ⇒ in range. dequant_block reads exactly those
// top nibbles back into d.
fn synth_iq1m(n_elem: usize, seed: u32) -> Vec<u8> {
    assert_eq!(n_elem % 256, 0);
    let d_bits = half::f16::from_f32(0.03).to_bits();
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 56];
        // qs[0..32] + qh[32..48] — random, all valid.
        blk[0..48].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 48));
        // scales[48..56]: top nibble of each u16 carries a nibble of d, low 12 bits are the varied
        // 3-bit dl fields.
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

// Same coverage as `iq_parity_suite` (GEMV m=1, RT m=2, cmm m=4, hmm m=18, fused-QKV w_off), but the
// IQ1 synth builders differ per format (IQ1_M packs d into trailing scales), so take the builder as
// a closure. Tolerances match the signed-codebook reassociation the other grid i-quants document.
fn iq1_parity_suite(dtype: DType, seed: u32, synth: impl Fn(usize, u32) -> Vec<u8>) {
    for (m, in_f, out_f) in [(1usize, 256usize, 96usize), (2, 256, 96)] {
        check_quant_linear_parity_tol(dtype, synth(out_f * in_f, seed), m, in_f, out_f, 5e-3);
    }
    for (m, in_f, out_f) in [(4usize, 512usize, 128usize), (18, 512, 128)] {
        check_quant_linear_parity_impl(
            dtype,
            synth(out_f * in_f, seed + 1),
            m,
            in_f,
            out_f,
            2.5e-3,
            true,
        );
    }
    check_linear_woff(
        dtype,
        synth(256 * 256, seed + 2),
        1,
        256,
        &[128usize, 64, 64],
        false,
        1e-3,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq1s_matches_dequant_reference() {
    iq1_parity_suite(DType::Iq1S, 440, synth_iq1s);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_iq1m_matches_dequant_reference() {
    iq1_parity_suite(DType::Iq1M, 450, synth_iq1m);
}

// K-quants are the formats real checkpoints actually ship. Exercise the Metal dequant path
// (`weight_buf` → `dequant_block`) for Q4_K and Q6_K, same dequant-reference comparison as Q8_0.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_matches_dequant_reference() {
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q4K, synth_q4k(out_f * in_f, 26), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_matches_dequant_reference() {
    let (m, in_f, out_f) = (2usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q6K, synth_q6k(out_f * in_f, 27), m, in_f, out_f);
}

// m = 2..8 K-quants route to the MULTI-ROW mul_mv GEMV (weight registers reused across 4
// token rows); m=5 exercises the partial token block, out_f=94 the partial row pair.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_multirow_matches_dequant_reference() {
    let (m, in_f, out_f) = (5usize, 512usize, 94usize);
    check_quant_linear_parity(DType::Q4K, synth_q4k(out_f * in_f, 110), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_multirow_matches_dequant_reference() {
    let (m, in_f, out_f) = (3usize, 512usize, 96usize);
    check_quant_linear_parity(DType::Q6K, synth_q6k(out_f * in_f, 111), m, in_f, out_f);
}

// m=1 routes to the GEMV kernels — decode's path, distinct from the m=2 row-tiled and m>=16 GEMM
// routes the tests above/below take.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q4K, synth_q4k(out_f * in_f, 30), m, in_f, out_f);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_gemv_matches_dequant_reference() {
    let (m, in_f, out_f) = (1usize, 256usize, 96usize);
    check_quant_linear_parity(DType::Q6K, synth_q6k(out_f * in_f, 31), m, in_f, out_f);
}

// m >= 16 routes to the simdgroup_matrix GEMM kernels (`linear_quik*_mm`); m=18 also covers the
// partial row tile's scalar fallback (18 = 2 full 8-row tiles + 2 remainder rows).
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    // m >= 16 runs the half-fragment GEMM (f16 operands, f32 accumulate — the llama.cpp trade,
    // well under quantization error); the reference below rounds its operands the same way.
    check_quant_linear_parity_impl(
        DType::Q4K,
        synth_q4k(out_f * in_f, 28),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

// out_f % 64 == 0 routes to the cooperative-tile GEMM (`linear_*_cmm`); m=40 covers one full
// 32-row tile plus a partial one. Same f16-operand reference as the other GEMM tests.
#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q4K,
        synth_q4k(out_f * in_f, 32),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_coop_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (40usize, 256usize, 128usize);
    check_quant_linear_parity_impl(
        DType::Q6K,
        synth_q6k(out_f * in_f, 33),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_gemm_matches_dequant_reference() {
    let (m, in_f, out_f) = (18usize, 256usize, 96usize);
    // Half-fragment GEMM path — see the Q4K GEMM test for the f16-operand rationale.
    check_quant_linear_parity_impl(
        DType::Q6K,
        synth_q6k(out_f * in_f, 29),
        m,
        in_f,
        out_f,
        1e-3,
        true,
    );
}

// ─── Split-K coop-GEMM at REAL verify shapes ──────────────────────────────────────
//
// The m >= 2 cmm gate routes small multi-row batches (spec verify's k+1 candidate rows, a chat
// turn's short suffix prefill) through the cooperative tile, and m < 16 with deep k engages the
// split-K variants (`linear_*_cmm_ks` + `cmm_ks_reduce`): ks_split = min(160/(nto*ntm), 8,
// in_f/128) partial planes reduced in fixed order. The shapes below make ks_split collapse to
// its cap of 8 (out_f/64 threadgroups few, k deep), so the k-partition arithmetic, the f32
// partial plane, and the fixed-order reduce are all on the tested path — the m=40 coop tests
// keep ks_split == 1 and never touch them. K-quants at m in 2..=8 route to the multi-row GEMV
// instead (covered by the multirow tests), so the K-quant cases here use m in 9..15.
//
// Tolerance 2.5e-3 (not the shallow tests' 1e-3): the reference mirrors the f16 OPERAND
// rounding but computes f32 products, while the MMA rounds per-product at ~2^-11 relative —
// accumulated over k=2048..4096 dots that's ~1.5e-3 worst case (observed 1.4e-3 on Q6K),
// deep-k accumulation, not a kernel defect. The shallow k=256 GEMM tests keep 1e-3.

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q8_0_splitk_verify_shape_matches_dequant_reference() {
    // m=3: a k=2 verify round's [t_next, cand..] rows. nto=4, ntm=1 → ks_split = 8.
    let (m, in_f, out_f) = (3usize, 2048usize, 256usize);
    let wf = rand_f32(out_f * in_f, 41);
    check_quant_linear_parity_impl(
        DType::Q8_0,
        quantize_q8_0(&wf),
        m,
        in_f,
        out_f,
        2.5e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q5_0_splitk_verify_shape_matches_dequant_reference() {
    // m=5: a k=4 verify round. Deep-k Q5_0 (gemma's gate/up class) through cmm_ks.
    let (m, in_f, out_f) = (5usize, 2048usize, 256usize);
    check_quant_linear_parity_impl(
        DType::Q5_0,
        synth_q5_0(out_f * in_f, 42),
        m,
        in_f,
        out_f,
        2.5e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q4k_splitk_verify_shape_matches_dequant_reference() {
    // m=12 skips the 2..=8 multi-row GEMV route and lands on cmm_ks (m < 16, deep k).
    let (m, in_f, out_f) = (12usize, 4096usize, 512usize);
    check_quant_linear_parity_impl(
        DType::Q4K,
        synth_q4k(out_f * in_f, 43),
        m,
        in_f,
        out_f,
        2.5e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn linear_q6k_splitk_verify_shape_matches_dequant_reference() {
    let (m, in_f, out_f) = (9usize, 2048usize, 256usize);
    check_quant_linear_parity_impl(
        DType::Q6K,
        synth_q6k(out_f * in_f, 44),
        m,
        in_f,
        out_f,
        2.5e-3,
        true,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rmsnorm_parity() {
    let (rows, dim) = (7usize, 512usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
    let w = g.weight(TensorDesc::new(vec![dim], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
    g.push(Op::RmsNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-6,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * dim, 10))),
        (w, f32_bytes(&rand_f32(dim, 11))),
    ];
    assert_parity(&g, &bound, dst, rows * dim, 1e-5);
}

fn check_rmsnorm_parity(rows: usize, dim: usize, seed: u64) {
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, dim], DType::F32));
    let w = g.weight(TensorDesc::new(vec![dim], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, dim], DType::F32));
    g.push(Op::RmsNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps: 1e-6,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * dim, seed))),
        (w, f32_bytes(&rand_f32(dim, seed + 1))),
    ];
    assert_parity(&g, &bound, dst, rows * dim, 1e-5);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rmsnorm_vec4_decode_shape_parity() {
    check_rmsnorm_parity(1, 5376, 101);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rmsnorm_vec4_multirow_gate_parity() {
    check_rmsnorm_parity(4, 2048, 103);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rmsnorm_scalar_fallback_shape_parity() {
    check_rmsnorm_parity(1, 2049, 105);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn qknorm_parity() {
    let (rows, nh, hd) = (5usize, 8usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let w = g.weight(TensorDesc::new(vec![hd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::QkNorm {
        x,
        weight: w,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        eps: 1e-6,
        x_stride: 0,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 12))),
        (w, f32_bytes(&rand_f32(hd, 13))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-5);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gated_rmsnorm_in_place_parity() {
    let (rows, nh, hd) = (3usize, 16usize, 128usize);
    let n = rows * nh * hd;
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let w = g.weight(TensorDesc::new(vec![hd], DType::F32));
    let gate = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::GatedRmsNorm {
        x,
        weight: w,
        gate,
        dst: x,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        eps: 1e-6,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(n, 107))),
        (w, f32_bytes(&rand_f32(hd, 108))),
        (gate, f32_bytes(&rand_f32(n, 109))),
    ];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &[(x, n)]).remove(0);
    let metal_be = MetalBackend::new().expect("metal backend");
    assert!(metal_be.capabilities().gated_rmsnorm);
    let metal = run_multi(&metal_be, &g, &bound, &[(x, n)]).remove(0);
    assert_close(&cpu, &metal, 1e-5, "in-place gated rmsnorm");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rope_parity() {
    let (rows, nh, hd, rd) = (4usize, 6usize, 128usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Rope {
        x,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rd as u32,
        theta: 10000.0,
        freq_factors: None,
        x_stride: 0,
    });
    let positions: Vec<i32> = (0..rows as i32).map(|i| i + 3).collect();
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 30))),
        (pos, i32_bytes(&positions)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn rope_partial_with_freq_factors_parity() {
    // rope_dim < head_dim (dims beyond rope_dim pass through) + per-pair freq_factors divisor
    let (rows, nh, hd, rd) = (3usize, 4usize, 128usize, 64usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let ff = g.input(TensorDesc::new(vec![rd / 2], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Rope {
        x,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rd as u32,
        theta: 1000000.0,
        freq_factors: Some(ff),
        x_stride: 0,
    });
    let positions: Vec<i32> = (0..rows as i32).map(|i| i * 2 + 1).collect();
    let ffv: Vec<f32> = (0..rd / 2).map(|i| 1.0 + i as f32 * 0.1).collect();
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 31))),
        (pos, i32_bytes(&positions)),
        (ff, f32_bytes(&ffv)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn qknormrope_parity() {
    let (rows, nh, hd, rd) = (4usize, 8usize, 128usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let w = g.weight(TensorDesc::new(vec![hd], DType::F32));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::QkNormRope {
        x,
        weight: w,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rd as u32,
        theta: 10000.0,
        eps: 1e-6,
        x_stride: 0,
        freq_factors: None,
    });
    let positions: Vec<i32> = (0..rows as i32).map(|i| i + 1).collect();
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 32))),
        (w, f32_bytes(&rand_f32(hd, 33))),
        (pos, i32_bytes(&positions)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// The qwen35 MTP head's exact QkNormRope shape: head_dim=256 with a PARTIAL rope_dim=64 (dims
// 64..256 pass through unrotated) and a high freq_base (1e7). The `qknormrope_parity` above only
// exercises head_dim==rope_dim==128 at theta 1e4, so partial rope at hd=256 was uncovered — the
// one caller is the MTP head, whose per-draft decode diverged from CPU starting at position 2
// (position 0's rotation is identity, so a rotation bug only shows once the angle is non-trivial).
// rows here span positions 0..6 explicitly so the ≥2 positions are on the tested path.
#[test]
#[ignore = "requires a Metal GPU"]
fn qknormrope_hd256_partial_parity() {
    let (rows, nh, hd, rd) = (6usize, 16usize, 256usize, 64usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let w = g.weight(TensorDesc::new(vec![hd], DType::F32));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::QkNormRope {
        x,
        weight: w,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rd as u32,
        theta: 1.0e7,
        eps: 1e-6,
        x_stride: 0,
        freq_factors: None,
    });
    let positions: Vec<i32> = (0..rows as i32).collect(); // 0,1,2,3,4,5 — includes pos >= 2
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * nh * hd, 34))),
        (w, f32_bytes(&rand_f32(hd, 35))),
        (pos, i32_bytes(&positions)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn writekv_f16_parity() {
    // WriteKv casts f32 rows into an f16 cache at row `pos`. Both backends must produce identical
    // f16 bytes.
    let (rows, row_stride, max_ctx, pos) = (2usize, 256usize, 8usize, 3usize);
    let cache_elems = max_ctx * row_stride;
    let mut g = Graph::new();
    let src = g.input(TensorDesc::new(vec![rows, row_stride], DType::F32));
    let cache = g.input(TensorDesc::new(vec![cache_elems], DType::F16));
    g.push(Op::WriteKv {
        src,
        cache,
        rows: rows as u32,
        row_stride: row_stride as u32,
        pos: pos as u32,
    });
    let bound = vec![
        (src, f32_bytes(&rand_f32(rows * row_stride, 40))),
        (cache, vec![0u8; cache_elems * 2]),
    ];
    let cpu = run_readback(&CpuBackend::new(), &g, &bound, cache, cache_elems * 2);
    let mtl = run_readback(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        cache,
        cache_elems * 2,
    );
    assert_eq!(cpu, mtl, "WriteKv f16 cache bytes must be identical");
}

// Q8_0 KV cache (INFR_KV_Q8): the quantization on write must be BYTE-identical between the CPU
// reference and the Metal kernel (d = amax/127 as f16, q = rint(x/d)).
#[test]
#[ignore = "requires a Metal GPU"]
fn writekv_q8_parity() {
    let (rows, row_stride, max_ctx, pos) = (2usize, 256usize, 8usize, 3usize);
    let cache_bytes = max_ctx * row_stride / 32 * 34;
    let mut g = Graph::new();
    let src = g.input(TensorDesc::new(vec![rows, row_stride], DType::F32));
    let cache = g.input(TensorDesc::new(vec![max_ctx * row_stride], DType::Q8_0));
    g.push(Op::WriteKv {
        src,
        cache,
        rows: rows as u32,
        row_stride: row_stride as u32,
        pos: pos as u32,
    });
    let bound = vec![
        (src, f32_bytes(&rand_f32(rows * row_stride, 44))),
        (cache, vec![0u8; cache_bytes]),
    ];
    let cpu = run_readback(&CpuBackend::new(), &g, &bound, cache, cache_bytes);
    let mtl = run_readback(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        cache,
        cache_bytes,
    );
    assert_eq!(cpu, mtl, "WriteKv q8 cache bytes must be identical");
}

// Attention over a Q8_0 cache: WriteKv quantizes, Attention dequantizes on read. Both routes:
// the scalar fallback (prefill shape) and the rows==1 vector kernel (decode at depth).
fn q8_attention_test(rows: usize, kv_len: usize, hd: usize, pos: usize, tol: f32, seed: u64) {
    let (nh, nkv) = (8usize, 2usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::Q8_0));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::Q8_0));
    let ksrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let vsrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let row = (nkv * hd) as u32;
    g.push(Op::WriteKv {
        src: ksrc,
        cache: kc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: vsrc,
        cache: vc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let nkv_elems = kv_len * nkv * hd;
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, seed))),
        (kc, vec![0u8; nkv_elems / 32 * 34]),
        (vc, vec![0u8; nkv_elems / 32 * 34]),
        (ksrc, f32_bytes(&rand_f32(nkv_elems, seed + 1))),
        (vsrc, f32_bytes(&rand_f32(nkv_elems, seed + 2))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, tol);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_scalar_parity() {
    q8_attention_test(3, 6, 64, 3, 1e-4, 240);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_vec_parity() {
    q8_attention_test(1, 200, 128, 199, 1e-4, 250);
}

// Wide q8 launch: routes to the cooperative q8 flash (dequant-staged KV tiles). Q rounds to f16
// on this path (the flash trade), hence the flash-class tolerance.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_flash_parity() {
    q8_attention_test(17, 136, 128, 119, 5e-3, 260);
}

// hd=256 q8 decode (gemma + INFR_KV_Q8): the NSG=16 q8 vector instantiation.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_q8_vec_hd256_parity() {
    q8_attention_test(1, 200, 256, 199, 1e-4, 270);
}

// ── Decoupled quant KV (mainline block quants q4_0/q4_1/q5_0/q5_1/iq4_nl, dense bf16, TurboQuant
// turbo2/3/4, dense f32). WriteKv quantizes into the compact cache; Attention expands each
// quantized/bf16 side into a transient f16 scratch (f32 reads natively via attention_f32) and runs
// the standard f16 attention over it — the ported Vulkan dequant→f16 prepass. Parity is against the
// CPU oracle, which dequants to f32 and runs f32 SDPA; the tolerances cover the extra f16-scratch
// attention rounding (looser than the q8 native-read path, which accumulates in float). Each
// quantize/dequant kernel is a bit-for-bit port of the CPU reference so only the attention precision
// differs, not the stored quant values. K stays f16 in the common decoupled shape (high-precision K,
// quantized V — llama's guidance); coupled quant/quant is also covered.

/// KV cache sizing — the SEAM's sizer, not a local restatement of it, so a cache this suite
/// allocates is exactly the cache the runner would (this used to be a hand-copied table that had
/// already lost the Q8_0 arm).
use infr_core::budget::kv_fmt_bytes as kv_bytes;

#[allow(clippy::too_many_arguments)]
fn kvquant_attention_test(
    kdt: DType,
    vdt: DType,
    rows: usize,
    kv_len: usize,
    hd: usize,
    pos: usize,
    tol: f32,
    seed: u64,
) {
    let (nh, nkv) = (8usize, 2usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], kdt));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], vdt));
    let ksrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let vsrc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let row = (nkv * hd) as u32;
    g.push(Op::WriteKv {
        src: ksrc,
        cache: kc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: vsrc,
        cache: vc,
        rows: kv_len as u32,
        row_stride: row,
        pos: 0,
    });
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let nkv_elems = kv_len * nkv * hd;
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, seed))),
        (kc, vec![0u8; kv_bytes(kdt, nkv_elems)]),
        (vc, vec![0u8; kv_bytes(vdt, nkv_elems)]),
        (ksrc, f32_bytes(&rand_f32(nkv_elems, seed + 1))),
        (vsrc, f32_bytes(&rand_f32(nkv_elems, seed + 2))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, tol);
}

// Block quants, decoupled (K=f16 native, V=quant prepassed) at the rows==1 vector-flash decode
// shape, and coupled (quant/quant) at the scalar prefill shape. Both routes read the f16 scratch.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q4_0_parity() {
    kvquant_attention_test(DType::F16, DType::Q4_0, 1, 200, 128, 199, 6e-3, 300);
    kvquant_attention_test(DType::Q4_0, DType::Q4_0, 3, 6, 64, 3, 6e-3, 305);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q4_1_parity() {
    kvquant_attention_test(DType::F16, DType::Q4_1, 1, 200, 128, 199, 6e-3, 310);
    kvquant_attention_test(DType::Q4_1, DType::Q4_1, 3, 6, 64, 3, 6e-3, 315);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q5_0_parity() {
    kvquant_attention_test(DType::F16, DType::Q5_0, 1, 200, 128, 199, 6e-3, 320);
    kvquant_attention_test(DType::Q5_0, DType::Q5_0, 3, 6, 64, 3, 6e-3, 325);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_q5_1_parity() {
    kvquant_attention_test(DType::F16, DType::Q5_1, 1, 200, 128, 199, 6e-3, 330);
    kvquant_attention_test(DType::Q5_1, DType::Q5_1, 3, 6, 64, 3, 6e-3, 335);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_iq4_nl_parity() {
    kvquant_attention_test(DType::F16, DType::Iq4Nl, 1, 200, 128, 199, 6e-3, 340);
    kvquant_attention_test(DType::Iq4Nl, DType::Iq4Nl, 3, 6, 64, 3, 6e-3, 345);
}

// Coupled quant/quant at DEPTH: the existing coupled cases run at kv_len=6, which never
// exercises the prepass scratch indexing past the first blocks or the decode-shape read at a
// deep position. kv_len=2048 at both the rows==1 decode shape and an 8-row prefill shape pins
// the block arithmetic at real conversation depth (found relevant while investigating an e2e
// recall gap on coupled iq4_nl — the kernels are clean at depth; the gap is 4-bit-loss-on-
// both-sides × f16-attention precision compounding, which these tolerances bound).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_deep_coupled_parity() {
    for (kdt, vdt, seed) in [
        (DType::Q4_0, DType::Q4_0, 360),
        (DType::Q4_1, DType::Q4_1, 362),
        (DType::Q5_0, DType::Q5_0, 364),
        (DType::Q5_1, DType::Q5_1, 366),
        (DType::Iq4Nl, DType::Iq4Nl, 368),
    ] {
        kvquant_attention_test(kdt, vdt, 1, 2048, 128, 2047, 6e-3, seed);
        kvquant_attention_test(kdt, vdt, 8, 2048, 128, 2040, 6e-3, seed + 1);
    }
}

// Dense bf16 (near-lossless top-16-bits store, dequant <<16 → f16). Decoupled + coupled.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_bf16_parity() {
    kvquant_attention_test(DType::F16, DType::Bf16, 1, 200, 128, 199, 5e-3, 350);
    kvquant_attention_test(DType::Bf16, DType::Bf16, 3, 6, 64, 3, 5e-3, 355);
}

// Dense f32: the native f32 attention path (no prepass) — coupled f32/f32 (the Metal clamp forbids
// a mixed f32/other request). Both backends run f32 SDPA, so a tight tolerance.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_f32_parity() {
    kvquant_attention_test(DType::F32, DType::F32, 1, 200, 128, 199, 1e-3, 360);
    kvquant_attention_test(DType::F32, DType::F32, 3, 6, 64, 3, 1e-3, 365);
}

// TurboQuant (WHT-rotated, 128-elem blocks = head_dim slices, so hd must be a multiple of 128).
// Coupled turbo/turbo at the vector shape + K=f16/V=turbo at the scalar shape. The inverse-WHT
// dequant plus f16-scratch storage widen the tolerance vs the block quants.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_turbo2_parity() {
    kvquant_attention_test(DType::Turbo2, DType::Turbo2, 1, 200, 128, 199, 1.2e-2, 370);
    kvquant_attention_test(DType::F16, DType::Turbo2, 3, 6, 128, 3, 1.2e-2, 375);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_turbo3_parity() {
    kvquant_attention_test(DType::Turbo3, DType::Turbo3, 1, 200, 128, 199, 1.2e-2, 380);
    kvquant_attention_test(DType::F16, DType::Turbo3, 3, 6, 128, 3, 1.2e-2, 385);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_kv_turbo4_parity() {
    kvquant_attention_test(DType::Turbo4, DType::Turbo4, 1, 200, 128, 199, 1.2e-2, 390);
    kvquant_attention_test(DType::F16, DType::Turbo4, 3, 6, 128, 3, 1.2e-2, 395);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_gqa_causal_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (3usize, 6usize, 8usize, 2usize, 64usize, 0usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 41))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 42))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 43))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// rows==1 decode at hd=256 with a SHORT kv_len (< 128): rows*n_head < 128 so it's not a wide
// launch, and kv_len < 128 so neither the vec nor split32 tier applies (split32 is hd<=128
// only) — it routes to the 8-way `attnsplit_f16kv`. Every other hd=256 decode in the engine is
// TAPED (attnvec_dyn_hd256), so this attnsplit path is otherwise unexercised; the MTP head's
// per-draft-step decode is the one real caller (kv_len grows 1,2,3,… as the head accumulates
// its own KV), and it diverged there from kv_len 3 on. GQA (nkv=4) like the qwen35 head.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_decode_hd256_short_kv_parity() {
    for kv_len in [1usize, 2, 3, 5, 8] {
        let (rows, nh, nkv, hd) = (1usize, 16usize, 4usize, 256usize);
        let pos = kv_len - 1;
        let mut g = Graph::new();
        let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
        let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
        let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
        g.push(Op::Attention {
            q,
            k_cache: kc,
            v_cache: vc,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            mask: infr_core::graph::AttnMask::Causal,
            pos: pos as u32,
        });
        let bound = vec![
            (q, f32_bytes(&rand_f32(rows * nh * hd, 71 + kv_len as u64))),
            (
                kc,
                f16_bytes(&rand_f32(kv_len * nkv * hd, 72 + kv_len as u64)),
            ),
            (
                vc,
                f16_bytes(&rand_f32(kv_len * nkv * hd, 73 + kv_len as u64)),
            ),
        ];
        assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
    }
}

// Wide launch, short context (rows*n_head >= 128, kv_len < 128): routes to the lean unsplit
// kernel (`attention_*`), which the small-shape tests above never reach at this width.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_prefill_wide_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 24usize, 8usize, 2usize, 64usize, 7usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 61))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 62))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 63))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// Wide launch, long f16 context: routes to the half-fragment flash kernel (`attnflash_f16kv`).
// Q and P round to f16 in that path (accumulation stays f32), hence the wider tolerance than the
// exact-f32 attention kernels. kv_len is a multiple of 8 so the kernel's tail-block reads stay
// inside these exact-sized test buffers (the runtime cache is sized for the full context).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 64usize, 119usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 71))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 72))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 73))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// Same flash shape at hd = 72 (% 8 but not % 32): routes to the single-simdgroup flash kernel
// (`attnflash_f16kv`), which the hd % 32 == 0 tests above no longer reach (those take the
// cooperative `attnflash2_f16kv`).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash_hd72_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 72usize, 119usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 171))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 172))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 173))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// The cooperative flash kernel (`attnflash2_f16kv`) at the real model head size (hd = 128), with
// a partial final query tile (rows = 17) and a KV length that lands mid-block (the kernel's
// causal-skip keeps tail reads within 7 rows of the limit).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash2_hd128_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 128usize, 119usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 181))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 182))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 183))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

fn attention_flash2_four_row_prefill_match(
    kv_len: usize,
    mask: infr_core::graph::AttnMask,
    pos: usize,
    seed: u64,
) {
    let (rows, nh, nkv, hd) = (4usize, 16usize, 8usize, 128usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, seed))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, seed + 1))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, seed + 2))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// Four-row deep prefill reuses each K/V tile across the query rows through cooperative flash.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash2_four_row_prefill_matches_reference() {
    // Causal prefill: rows=4, nh=16, hd=128. kv_len=136 crosses the first C128 chunk
    // and has a non-aligned 8-position tail.
    attention_flash2_four_row_prefill_match(136, infr_core::graph::AttnMask::Causal, 131, 601);

    // Sliding-window masking should clip the same prefill window with its lower bound still in
    // the first C128 chunk.
    let kv_len = 136usize;
    let win = 64usize;
    let pos = 131usize;
    let lo = pos.saturating_sub(win - 1);
    assert!(lo < 128, "lower bound must sit in the first C128 chunk");
    attention_flash2_four_row_prefill_match(
        kv_len,
        infr_core::graph::AttnMask::SlidingWindow(win),
        pos,
        602,
    );
}

// hd=256 (gemma): the cooperative flash instantiation with 8 O fragments per simdgroup.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash2_hd256_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 256usize, 119usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::SlidingWindow(64),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 401))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 402))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 403))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// hd=256 decode (gemma): the NSG=16 vector flash instantiation, sliding window active (gemma's
// local layers decode with window clipping — the shape the sweep found on the split fallback).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_vec_hd256_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (1usize, 200usize, 4usize, 1usize, 256usize, 199usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::SlidingWindow(96),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 404))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 405))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 406))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// Sliding-window masking through the cooperative flash kernel: the analytic per-row window
// lower bound must match the CPU reference (whole leading KV blocks fall below some rows'
// windows but not others').
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_flash2_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (17usize, 136usize, 8usize, 2usize, 128usize, 119usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::SlidingWindow(64),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 191))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 192))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 193))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// Long-context decode shape (rows=1, kv_len >= 128, hd=128): routes to the VECTOR flash kernel
// (`attnvec_f16kv_hd128`) — 32 simdgroups, 32 KV positions per simdgroup step, log2 merge. The
// kv_len=200 tail lands mid-block, exercising the clamped+masked tail rows.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_long_context_split32_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (1usize, 200usize, 8usize, 2usize, 128usize, 199usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 51))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 52))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 53))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// The 32-way split-KV kernel (`attnsplit32_*`) retained for head sizes without a vec-kernel
// instantiation (hd=96 here): same long-context decode shape as above.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_long_context_split32_hd96_parity() {
    let (rows, kv_len, nh, nkv, hd, pos) = (1usize, 200usize, 8usize, 2usize, 96usize, 199usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Causal,
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 201))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 202))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 203))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

// Sliding-window decode at depth through the vector flash kernel: whole leading KV blocks fall
// below the window (the kernel's block-skip), and the window edge lands mid-block.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_vec_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos, win) = (
        1usize, 300usize, 8usize, 2usize, 64usize, 299usize, 100usize,
    );
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::SlidingWindow(win),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 211))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 212))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 213))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_sliding_window_parity() {
    let (rows, kv_len, nh, nkv, hd, pos, win) =
        (4usize, 10usize, 4usize, 4usize, 64usize, 2usize, 3usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::SlidingWindow(win),
        pos: pos as u32,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 44))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 45))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 46))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 1e-4);
}

fn gated_test(act: infr_core::graph::Activation, up_off: usize, seed: u64) {
    let (rows, nff) = (3usize, 512usize);
    let up_len = rows * nff + up_off;
    let mut g = Graph::new();
    let gate = g.input(TensorDesc::new(vec![rows, nff], DType::F32));
    let up = g.input(TensorDesc::new(vec![up_len], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nff], DType::F32));
    g.push(Op::GatedAct {
        gate,
        up,
        dst,
        rows: rows as u32,
        nff: nff as u32,
        act,
        up_off: up_off as u32,
        up_stride: 0,
        gate_stride: 0,
        gate_block_width: 0,
    });
    let bound = vec![
        (gate, f32_bytes(&rand_f32(rows * nff, seed))),
        (up, f32_bytes(&rand_f32(up_len, seed + 1))),
    ];
    assert_parity(&g, &bound, dst, rows * nff, 1e-5);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gatedact_silu_parity() {
    gated_test(infr_core::graph::Activation::Silu, 0, 50);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gatedact_gelu_parity() {
    gated_test(infr_core::graph::Activation::Gelu, 0, 52);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gatedact_upoff_parity() {
    gated_test(infr_core::graph::Activation::Silu, 128, 54);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gatedactfused_parity() {
    let (rows, nff) = (3usize, 256usize);
    let mut g = Graph::new();
    let gu = g.input(TensorDesc::new(vec![rows, 2 * nff], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nff], DType::F32));
    g.push(Op::GatedActFused {
        gu,
        dst,
        rows: rows as u32,
        nff: nff as u32,
        act: infr_core::graph::Activation::Silu,
    });
    let bound = vec![(gu, f32_bytes(&rand_f32(rows * 2 * nff, 270)))];
    assert_parity(&g, &bound, dst, rows * nff, 1e-5);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_parity() {
    let (ne, n_expert, n_used, nff) = (64usize, 8usize, 2usize, 128usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::F32));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::F32));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::F32));
    let dst = g.output(TensorDesc::new(vec![ne], DType::F32));
    g.push(Op::MoeFfn {
        x,
        router_x: x,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: nff as u32,
        scale: 1.0,
        act: infr_core::graph::Activation::Silu,
        gating: infr_core::graph::MoeGating::Softmax,
        norm_w: true,
        weight_before: false,
        ep_band: None,
        exp_probs_b: None,
        n_expert_groups: 0,
        n_expert_groups_used: 0,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(ne, 60))),
        (router, f32_bytes(&rand_f32(n_expert * ne, 61))),
        (gate, f32_bytes(&rand_f32(n_expert * nff * ne, 62))),
        (up, f32_bytes(&rand_f32(n_expert * nff * ne, 63))),
        (down, f32_bytes(&rand_f32(n_expert * ne * nff, 64))),
    ];
    assert_parity(&g, &bound, dst, ne, 1e-3);
}

// Quantized experts route to the DEVICE MoE path (router GEMV + on-device top-k + expert-table
// GEMVs); the f32 test above keeps the host fallback covered. CPU is the oracle here (its MoE
// matvec dequants the same bytes and dots in f32 — no Q8 activation quantization).
fn moe_quant_test(dtype: DType, synth: fn(usize, u32) -> Vec<u8>, seed: u32) {
    let (ne, n_expert, n_used, nff) = (256usize, 8usize, 3usize, 256usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], dtype));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], dtype));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], dtype));
    let dst = g.output(TensorDesc::new(vec![ne], DType::F32));
    g.push(Op::MoeFfn {
        x,
        router_x: x,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: nff as u32,
        scale: 1.0,
        act: infr_core::graph::Activation::Silu,
        gating: infr_core::graph::MoeGating::Softmax,
        norm_w: true,
        weight_before: false,
        ep_band: None,
        exp_probs_b: None,
        n_expert_groups: 0,
        n_expert_groups_used: 0,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(ne, seed as u64))),
        (router, f32_bytes(&rand_f32(n_expert * ne, seed as u64 + 1))),
        (gate, synth(n_expert * nff * ne, seed + 2)),
        (up, synth(n_expert * nff * ne, seed + 3)),
        (down, synth(n_expert * ne * nff, seed + 4)),
    ];
    assert_parity(&g, &bound, dst, ne, 1e-3);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_q4k_device_parity() {
    moe_quant_test(DType::Q4K, synth_q4k, 80);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_q6k_device_parity() {
    moe_quant_test(DType::Q6K, synth_q6k, 90);
}

// Batched rows through the device MoE path (rows spanning two 256-row chunks): every row routes
// independently; parity vs the CPU reference's per-row loop.
#[test]
#[ignore = "requires a Metal GPU"]
fn moe_ffn_batched_rows_parity() {
    let (rows, ne, n_expert, n_used, nff) = (300usize, 256usize, 8usize, 3usize, 256usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, ne], DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q4K));
    let dst = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
    g.push(Op::MoeFfn {
        x,
        router_x: x,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: nff as u32,
        scale: 1.0,
        act: infr_core::graph::Activation::Silu,
        gating: infr_core::graph::MoeGating::Softmax,
        norm_w: true,
        weight_before: false,
        ep_band: None,
        exp_probs_b: None,
        n_expert_groups: 0,
        n_expert_groups_used: 0,
    });
    // x scaled down: the ~50x-real synthetic weights would push gate/up activations past f16
    // range (the kernels' operand precision) with unit-scale inputs — real hidden states don't.
    let xs_small: Vec<f32> = rand_f32(rows * ne, 95).iter().map(|v| v * 0.02).collect();
    let bound = vec![
        (x, f32_bytes(&xs_small)),
        (router, f32_bytes(&rand_f32(n_expert * ne, 96))),
        (gate, synth_q4k(n_expert * nff * ne, 97)),
        (up, synth_q4k(n_expert * nff * ne, 98)),
        (down, synth_q4k(n_expert * ne * nff, 99)),
    ];
    // Reference mirrors the grouped-GEMM path's numerics (same policy as the dense GEMM parity
    // tests): expert weights and stage inputs round to f16 (the kernels' operand precision, f32
    // accumulate), router/top-k stay f32. Residual tolerance covers reassociation over the
    // ~50x-real-magnitude synthetic weights' cancellation tail.
    let r16 =
        |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect() };
    let xs: Vec<f32> = {
        let (_, b) = &bound[0];
        bytemuck::cast_slice::<u8, f32>(b).to_vec()
    };
    let rw: Vec<f32> = {
        let (_, b) = &bound[1];
        bytemuck::cast_slice::<u8, f32>(b).to_vec()
    };
    use infr_gguf::dequant::dequant_block;
    let gw = r16(&dequant_block(DType::Q4K, &bound[2].1).unwrap());
    let uw = r16(&dequant_block(DType::Q4K, &bound[3].1).unwrap());
    let dw = r16(&dequant_block(DType::Q4K, &bound[4].1).unwrap());
    let mut reference = vec![0f32; rows * ne];
    for row in 0..rows {
        let x = &xs[row * ne..(row + 1) * ne];
        let logits: Vec<f32> = (0..n_expert)
            .map(|e| (0..ne).map(|i| rw[e * ne + i] * x[i]).sum::<f32>())
            .collect();
        let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
        let psum: f32 = probs.iter().sum();
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(n_used);
        let wsum: f32 = idx.iter().map(|&e| probs[e] / psum).sum::<f32>().max(1e-20);
        let x16 = r16(x);
        for &e in &idx {
            let gs = &gw[e * nff * ne..(e + 1) * nff * ne];
            let us = &uw[e * nff * ne..(e + 1) * nff * ne];
            let ds = &dw[e * ne * nff..(e + 1) * ne * nff];
            let gate: Vec<f32> = (0..nff)
                .map(|o| (0..ne).map(|i| gs[o * ne + i] * x16[i]).sum::<f32>())
                .collect();
            let up: Vec<f32> = (0..nff)
                .map(|o| (0..ne).map(|i| us[o * ne + i] * x16[i]).sum::<f32>())
                .collect();
            let act: Vec<f32> = (0..nff)
                .map(|i| {
                    let g = gate[i];
                    (g / (1.0 + (-g).exp())) * up[i]
                })
                .collect();
            let a16 = r16(&act);
            let w_e = (probs[e] / psum) / wsum;
            for o in 0..ne {
                let y: f32 = (0..nff).map(|i| ds[o * nff + i] * a16[i]).sum();
                reference[row * ne + o] += w_e * y;
            }
        }
    }
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &g,
        &bound,
        dst,
        rows * ne,
    );
    assert_close(&reference, &mtl, 5e-3, "moe batched grouped");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn conv1d_silu_parity() {
    let (cc, kk) = (256usize, 4usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![cc], DType::F32));
    let w = g.weight(TensorDesc::new(vec![cc, kk], DType::F32));
    let state = g.input(TensorDesc::new(vec![kk - 1, cc], DType::F32));
    let dst = g.output(TensorDesc::new(vec![cc], DType::F32));
    g.push(Op::Conv1dSilu {
        x,
        weight: w,
        state,
        dst,
        rows: 1,
        channels: cc as u32,
        kernel: kk as u32,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(cc, 70))),
        (w, f32_bytes(&rand_f32(cc * kk, 71))),
        (state, f32_bytes(&rand_f32((kk - 1) * cc, 72))),
    ];
    let reads = [(dst, cc), (state, (kk - 1) * cc)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(&cpu[0], &mtl[0], 1e-5, "conv1d dst");
    assert_close(&cpu[1], &mtl[1], 0.0, "conv1d state"); // shift is exact
}

#[test]
#[ignore = "requires a Metal GPU"]
fn deltanet_parity() {
    let (nv, nk, kd, vd) = (4usize, 2usize, 64usize, 64usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![nk * kd], DType::F32));
    let k = g.input(TensorDesc::new(vec![nk * kd], DType::F32));
    let v = g.input(TensorDesc::new(vec![nv * vd], DType::F32));
    let b = g.input(TensorDesc::new(vec![nv], DType::F32));
    let a = g.input(TensorDesc::new(vec![nv], DType::F32));
    let a_coef = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let dt_bias = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let state = g.input(TensorDesc::new(vec![nv * kd * vd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![nv * vd], DType::F32));
    g.push(Op::DeltaNet {
        q,
        k,
        v,
        b,
        a,
        a_coef,
        dt_bias,
        state,
        dst,
        rows: 1,
        n_vhead: nv as u32,
        n_khead: nk as u32,
        head_k: kd as u32,
        head_v: vd as u32,
        eps: 1e-6,
        src_stride: 0,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(nk * kd, 80))),
        (k, f32_bytes(&rand_f32(nk * kd, 81))),
        (v, f32_bytes(&rand_f32(nv * vd, 82))),
        (b, f32_bytes(&rand_f32(nv, 83))),
        (a, f32_bytes(&rand_f32(nv, 84))),
        (a_coef, f32_bytes(&rand_f32(nv, 85))),
        (dt_bias, f32_bytes(&rand_f32(nv, 86))),
        (state, f32_bytes(&rand_f32(nv * kd * vd, 87))),
    ];
    let reads = [(dst, nv * vd), (state, nv * kd * vd)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(&cpu[0], &mtl[0], 1e-4, "deltanet dst");
    assert_close(&cpu[1], &mtl[1], 1e-4, "deltanet state");
}

// Multi-row scan at the qwen3-next head shape: the state must carry across rows exactly (the
// device kernel loops rows with each lane owning its state column).
//
// Run at rows on BOTH sides of the q/k-norm-prep gate (`prefer_deltanet_norm_prep`, rows >= 8),
// because that gate picks a DIFFERENT KERNEL: at 8 rows a separate pass normalizes q/k into
// scratch and `deltanet_prepared_*` consumes it, while below 8 the scan normalizes inline
// (`deltanet_gates_*`). One row count exercises one of those and says nothing about the other.
#[test]
#[ignore = "requires a Metal GPU"]
fn deltanet_multirow_parity_inline_norm() {
    deltanet_multirow_case(5); // below the gate: inline normalization, no scratch
}

#[test]
#[ignore = "requires a Metal GPU"]
fn deltanet_multirow_parity_prepared_norm() {
    deltanet_multirow_case(8); // at the gate: the hoisted q/k norm pass feeds the scan
}

fn deltanet_multirow_case(rows: usize) {
    let (nv, nk, kd, vd) = (4usize, 2usize, 128usize, 128usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nk * kd], DType::F32));
    let k = g.input(TensorDesc::new(vec![rows, nk * kd], DType::F32));
    let v = g.input(TensorDesc::new(vec![rows, nv * vd], DType::F32));
    let b = g.input(TensorDesc::new(vec![rows, nv], DType::F32));
    let a = g.input(TensorDesc::new(vec![rows, nv], DType::F32));
    let a_coef = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let dt_bias = g.weight(TensorDesc::new(vec![nv], DType::F32));
    let state = g.input(TensorDesc::new(vec![nv * kd * vd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, nv * vd], DType::F32));
    g.push(Op::DeltaNet {
        q,
        k,
        v,
        b,
        a,
        a_coef,
        dt_bias,
        state,
        dst,
        rows: rows as u32,
        n_vhead: nv as u32,
        n_khead: nk as u32,
        head_k: kd as u32,
        head_v: vd as u32,
        eps: 1e-6,
        src_stride: 0,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nk * kd, 300))),
        (k, f32_bytes(&rand_f32(rows * nk * kd, 301))),
        (v, f32_bytes(&rand_f32(rows * nv * vd, 302))),
        (b, f32_bytes(&rand_f32(rows * nv, 303))),
        (a, f32_bytes(&rand_f32(rows * nv, 304))),
        (a_coef, f32_bytes(&rand_f32(nv, 305))),
        (dt_bias, f32_bytes(&rand_f32(nv, 306))),
        (state, f32_bytes(&rand_f32(nv * kd * vd, 307))),
    ];
    let reads = [(dst, rows * nv * vd), (state, nv * kd * vd)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(
        &cpu[0],
        &mtl[0],
        1e-4,
        &format!("deltanet multirow dst (rows={rows})"),
    );
    assert_close(
        &cpu[1],
        &mtl[1],
        1e-4,
        &format!("deltanet multirow state (rows={rows})"),
    );
}

// Multi-row conv: the rolling state shifts once per row and survives to the next.
#[test]
#[ignore = "requires a Metal GPU"]
fn conv1d_silu_multirow_parity() {
    let (rows, cc, kk) = (5usize, 256usize, 4usize);
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, cc], DType::F32));
    let w = g.weight(TensorDesc::new(vec![cc, kk], DType::F32));
    let state = g.input(TensorDesc::new(vec![kk - 1, cc], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows, cc], DType::F32));
    g.push(Op::Conv1dSilu {
        x,
        weight: w,
        state,
        dst,
        rows: rows as u32,
        channels: cc as u32,
        kernel: kk as u32,
    });
    let bound = vec![
        (x, f32_bytes(&rand_f32(rows * cc, 310))),
        (w, f32_bytes(&rand_f32(cc * kk, 311))),
        (state, f32_bytes(&rand_f32((kk - 1) * cc, 312))),
    ];
    let reads = [(dst, rows * cc), (state, (kk - 1) * cc)];
    let cpu = run_multi(&CpuBackend::new(), &g, &bound, &reads);
    let mtl = run_multi(&MetalBackend::new().expect("metal"), &g, &bound, &reads);
    assert_close(&cpu[0], &mtl[0], 1e-4, "conv multirow dst");
    assert_close(&cpu[1], &mtl[1], 0.0, "conv multirow state");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn copy_parity() {
    let n = 4096usize;
    let (src_off, dst_off, cnt) = (1000usize, 64usize, 2048usize);
    let mut g = Graph::new();
    let src = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![n], DType::F32));
    g.push(Op::Copy {
        src,
        src_off: src_off as u32,
        dst,
        dst_off: dst_off as u32,
        n: cnt as u32,
    });
    let bound = vec![(src, f32_bytes(&rand_f32(n, 5)))];
    assert_parity(&g, &bound, dst, n, 0.0);
}

// ---- DiffusionGemma canvas denoise (Phase D — docs/diffusion-gemma.md, `AttnMask::Canvas`) ----
// Metal's blind implementation (attention_canvas*/attention_canvas32* in attention.metal, see
// exec.rs's `canvas_lo` routing) checked against the CPU reference — the SAME numeric-parity
// contract every other attention tier in this file gets. Unlike a bare "doesn't return
// Unsupported" smoke test, this actually exercises the fixed-`[lo, kv_len)`-for-every-row math on
// real hardware whenever one is present (still `#[ignore]`d off CI, like every other GPU test
// here). `lo` here is an arbitrary fixed split (not derived from a real prompt/SWA-window pair) —
// the mask doesn't care, it's just a row-independent bound.

// hd=128: routes to the NSG=32 kernel (attention_canvas32_f16kv — hd <= 128 and the device's
// threadgroup cap fits 1024 threads).
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_canvas_split32_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, lo) = (32usize, 136usize, 8usize, 2usize, 128usize, 40usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Canvas { lo },
        // `pos` is unused by Canvas (every row's bound is `[lo, kv_len)` regardless of position)
        // — 0 here matches how the denoise call site sizes it (see `Op::Attention`'s doc).
        pos: 0,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 501))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 502))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 503))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// hd=256 (gemma-shaped): hd > 128 excludes the NSG=32 kernel — routes to attention_canvas_f16kv
// (NSG=8, MAXHD=256) instead.
#[test]
#[ignore = "requires a Metal GPU"]
fn attention_canvas_split8_hd256_matches_reference() {
    let (rows, kv_len, nh, nkv, hd, lo) = (17usize, 200usize, 4usize, 1usize, 256usize, 60usize);
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    let kc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len, nkv, hd], DType::F16));
    let dst = g.output(TensorDesc::new(vec![rows, nh, hd], DType::F32));
    g.push(Op::Attention {
        q,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: infr_core::graph::AttnMask::Canvas { lo },
        pos: 0,
    });
    let bound = vec![
        (q, f32_bytes(&rand_f32(rows * nh * hd, 511))),
        (kc, f16_bytes(&rand_f32(kv_len * nkv * hd, 512))),
        (vc, f16_bytes(&rand_f32(kv_len * nkv * hd, 513))),
    ];
    assert_parity(&g, &bound, dst, rows * nh * hd, 5e-3);
}

// ============================================================================
// GPU-resident decode path: Op::Argmax / Op::Sample / Op::EmbedGather.
//
// These ops move the last decode step onto the GPU so decode only reads back
// the 4-byte token id (Argmax/Sample) or the gathered embedding rows
// (EmbedGather), not the [vocab] logits or a host-dequantized embed table. The
// kernels (argmax_f32 / sample_f32 in elementwise_norms.metal, embed_gather_*
// in embed_gather.metal) were added without a Metal device in the loop, so
// until now only the kernel-name tripwire covered them — nothing checked their
// NUMBERS. Each test runs the SAME graph op on the CPU interpreter (the trusted
// reference for these arms) and on Metal and asserts the result matches.
// ============================================================================

// A token id is a u32 bit-pattern stored in the f32 dst slot; compare raw bits
// (a wrong token is a different id, an exact match is bit-equal — no tolerance,
// and NaN-safe unlike a float subtract).
fn assert_id_parity(g: &Graph, bound: &[(TensorId, Vec<u8>)], dst: TensorId, rows: usize) {
    let cpu = run(&CpuBackend::new(), g, bound, dst, rows);
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        g,
        bound,
        dst,
        rows,
    );
    for r in 0..rows {
        assert_eq!(
            cpu[r].to_bits(),
            mtl[r].to_bits(),
            "token id row {r}: cpu={} metal={}",
            cpu[r].to_bits(),
            mtl[r].to_bits(),
        );
    }
}

// bf16 store: the top 16 bits of the f32 (truncation) — the CPU and Metal
// dequant both widen this same u16 back with a lossless << 16, so a bf16 embed
// row round-trips bit-exactly.
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

// argmax_f32: greedy token = highest logit (lowest index on ties). With
// distinct logits — the real decode case, since vocab-scale logit sums are
// ~never bit-equal — Metal must land on the same id as the host argmax. An
// injected unique peak anchors a known answer; the surrounding random spread
// exercises the strided-scan + tree reduce over a vocab-scale buffer.
#[test]
#[ignore = "requires a Metal GPU"]
fn argmax_f32_matches_cpu() {
    for (n, seed, peak) in [
        (151936usize, 7u64, 90210usize),
        (8192, 8, 4001),
        (4099, 9, 0),
    ] {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![n], DType::F32));
        let dst = g.output(TensorDesc::new(vec![1], DType::F32));
        g.push(Op::Argmax {
            x,
            dst,
            n: n as u32,
            rows: 1,
        });
        let mut xs: Vec<f32> = rand_f32(n, seed).iter().map(|v| v * 10.0).collect();
        xs[peak] = 1000.0; // unique max
        assert_id_parity(&g, &[(x, f32_bytes(&xs))], dst, 1);
        // Known answer too, not just CPU==Metal agreement.
        let got = run(
            &MetalBackend::new().unwrap(),
            &g,
            &[(x, f32_bytes(&xs))],
            dst,
            1,
        );
        assert_eq!(got[0].to_bits() as usize, peak, "argmax n={n}");
    }

    let n = 151_936usize;
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![n], DType::F32));
    let dst = g.output(TensorDesc::new(vec![1], DType::F32));
    g.push(Op::Argmax {
        x,
        dst,
        n: n as u32,
        rows: 1,
    });
    let mut xs = vec![-1.0f32; n];
    xs[17] = 10.0;
    xs[90_210] = 10.0;
    let got = run(
        &MetalBackend::new().unwrap(),
        &g,
        &[(x, f32_bytes(&xs))],
        dst,
        1,
    );
    assert_eq!(got[0].to_bits(), 17, "argmax must keep the lowest tie");

    xs.fill(f32::NEG_INFINITY);
    xs[90_210] = -1e35;
    let got = run(
        &MetalBackend::new().unwrap(),
        &g,
        &[(x, f32_bytes(&xs))],
        dst,
        1,
    );
    assert_eq!(
        got[0].to_bits() as usize,
        90_210,
        "argmax must cover the full finite f32 range",
    );
}

// sample_f32: temperature + top-k + top-p stochastic pick, with the uniform
// draw factored out into the `u` input so the op is a pure function. It must
// mirror the host `sample_logits` order of operations exactly, so the same `u`
// picks the same token. Distinct logits; sweep `u` and the (top_k, temp, top_p)
// knobs. top_k stays <= 64 (the kernel's SAMPLE_KMAX cap, which the caller
// respects — a larger top_k would diverge from the uncapped host reference).
#[test]
#[ignore = "requires a Metal GPU"]
fn sample_f32_matches_cpu() {
    let n = 8192usize;
    let xs: Vec<f32> = rand_f32(n, 21).iter().map(|v| v * 8.0).collect();
    for (top_k, temp, top_p) in [
        (40u32, 0.8f32, 0.95f32),
        (64, 1.0, 1.0),
        (20, 0.7, 0.90),
        (8, 1.2, 0.98),
    ] {
        for &uu in &[0.03f32, 0.29, 0.51, 0.74, 0.97] {
            let mut g = Graph::new();
            let x = g.input(TensorDesc::new(vec![n], DType::F32));
            let u = g.input(TensorDesc::new(vec![1], DType::F32));
            let dst = g.output(TensorDesc::new(vec![1], DType::F32));
            g.push(Op::Sample {
                x,
                u,
                dst,
                n: n as u32,
                top_k,
                temp,
                top_p,
            });
            assert_id_parity(&g, &[(x, f32_bytes(&xs)), (u, f32_bytes(&[uu]))], dst, 1);
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn sample_f32_vocab_split_matches_cpu() {
    let n = 151_936usize;
    let xs: Vec<f32> = rand_f32(n, 121).iter().map(|v| v * 8.0).collect();
    for (top_k, temp, top_p, uu) in [
        (20u32, 0.7f32, 0.95f32, 0.03f32),
        (20, 0.7, 0.95, 0.51),
        (20, 0.7, 0.95, 0.97),
        (64, 1.0, 1.0, 0.74),
    ] {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![n], DType::F32));
        let u = g.input(TensorDesc::new(vec![1], DType::F32));
        let dst = g.output(TensorDesc::new(vec![1], DType::F32));
        g.push(Op::Sample {
            x,
            u,
            dst,
            n: n as u32,
            top_k,
            temp,
            top_p,
        });
        assert_id_parity(&g, &[(x, f32_bytes(&xs)), (u, f32_bytes(&[uu]))], dst, 1);
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn sample_f32_vocab_split_clamps_large_top_k() {
    let n = 151_936usize;
    let xs: Vec<f32> = rand_f32(n, 126).iter().map(|v| v * 8.0).collect();
    let sample = |top_k| {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![n], DType::F32));
        let u = g.input(TensorDesc::new(vec![1], DType::F32));
        let dst = g.output(TensorDesc::new(vec![1], DType::F32));
        g.push(Op::Sample {
            x,
            u,
            dst,
            n: n as u32,
            top_k,
            temp: 1.0,
            top_p: 1.0,
        });
        (g, x, u, dst)
    };
    let (mtl_g, mtl_x, mtl_u, mtl_dst) = sample(100);
    let (cpu_g, cpu_x, cpu_u, cpu_dst) = sample(64);
    let mtl = run(
        &MetalBackend::new().expect("metal backend"),
        &mtl_g,
        &[(mtl_x, f32_bytes(&xs)), (mtl_u, f32_bytes(&[0.999]))],
        mtl_dst,
        1,
    );
    let cpu = run(
        &CpuBackend::new(),
        &cpu_g,
        &[(cpu_x, f32_bytes(&xs)), (cpu_u, f32_bytes(&[0.999]))],
        cpu_dst,
        1,
    );
    assert_eq!(
        mtl[0].to_bits(),
        cpu[0].to_bits(),
        "top_k=100 must match the effective top_k=64 CPU reference"
    );
}

// embed_gather_*: dst[r, :] = dequant(table[ids[r], :]) * scale, gathering the
// resident quantized token_embd row on-device (the SAME DEC16_* decode the
// linear kernels use) instead of a host dequant + upload. Covers both kernel
// families — the DEC16 quant macro (Q4_K/Q6_K/Q5_0/Q8_0/IQ4_XS) and the
// plain-widen f16/bf16 kernels — plus multi-row gather and a non-unit embed
// scale (Gemma's sqrt(n_embd)).
#[test]
#[ignore = "requires a Metal GPU"]
fn embed_gather_matches_cpu() {
    let (vocab, ne) = (8usize, 256usize); // ne % 32 == 0, whole K-quant blocks
    let ids: Vec<i32> = vec![5, 0, 7, 3]; // gather these rows (out of order, repeats none)
    let rows = ids.len();
    let check = |dt: DType, bytes: Vec<u8>, scale: f32, tol: f32, tag: &str| {
        let mut g = Graph::new();
        let id = g.input(TensorDesc::new(vec![rows], DType::I32));
        let table = g.weight(TensorDesc::new(vec![vocab, ne], dt));
        let dst = g.output(TensorDesc::new(vec![rows, ne], DType::F32));
        g.push(Op::EmbedGather {
            ids: id,
            table,
            dst,
            rows: rows as u32,
            ne: ne as u32,
            scale,
        });
        let bound = vec![(id, i32_bytes(&ids)), (table, bytes)];
        let cpu = run(&CpuBackend::new(), &g, &bound, dst, rows * ne);
        let mtl = run(
            &MetalBackend::new().expect("metal backend"),
            &g,
            &bound,
            dst,
            rows * ne,
        );
        assert_close(&cpu, &mtl, tol, &format!("embed_gather {tag}"));
    };
    let rf = rand_f32(vocab * ne, 31);
    // f16/bf16 are a lossless widen → bit-exact; the quant decodes match the
    // linear path's dequant to ULP (tol like the linear quant tests).
    check(DType::F16, f16_bytes(&rf), 1.0, 0.0, "f16");
    check(DType::Bf16, bf16_bytes(&rf), 22.627417, 0.0, "bf16 scaled");
    check(DType::Q8_0, quantize_q8_0(&rf), 1.0, 1e-3, "q8_0");
    check(DType::Q5_0, synth_q5_0(vocab * ne, 32), 1.0, 1e-3, "q5_0");
    check(
        DType::Q4K,
        synth_q4k(vocab * ne, 33),
        1.5,
        1e-3,
        "q4k scaled",
    );
    check(DType::Q6K, synth_q6k(vocab * ne, 34), 1.0, 1e-3, "q6k");
    check(
        DType::Iq4Xs,
        synth_iq4xs(vocab * ne, 35),
        1.0,
        1e-3,
        "iq4xs",
    );
}

// MLA parity — exercises the `mla_f16kv` Metal kernel (no freq_factors). Tiny dims so the
// reference is trivially verifiable by hand (same synthetic data as seam_op_parity::mla_parity):
// rows=2, n_head=2, kv_lora=3, qk_nope=2, qk_rope=2, v_head=2.
#[test]
#[ignore = "requires a Metal GPU"]
fn mla_parity() {
    let (rows, nh, kv_lora, qk_nope, qk_rope, vhd) =
        (2usize, 2usize, 3usize, 2usize, 2usize, 2usize);
    let key_len = kv_lora + qk_rope; // 5
    let q_head_dim = qk_nope + qk_rope; // 4
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows * nh * q_head_dim], DType::F32));
    // F16 KV cache — the Metal kernel reads `device const half*`, one half per element.
    let k_cache = g.input(TensorDesc::new(vec![rows * key_len], DType::F16));
    let wk_b = g.weight(TensorDesc::new(vec![nh * kv_lora * qk_nope], DType::F32));
    let wv_b = g.weight(TensorDesc::new(vec![nh * kv_lora * vhd], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows * nh * vhd], DType::F32));
    let scale = 1.0 / ((qk_nope + qk_rope) as f32).sqrt(); // 0.5
    g.push(Op::Mla {
        q,
        k_cache,
        wk_b,
        wv_b,
        dst,
        rows: rows as u32,
        kv_len: rows as u32, // attend to all rows
        n_head: nh as u32,
        q_head_dim: q_head_dim as u32,
        kv_lora_rank: kv_lora as u32,
        qk_nope_dim: qk_nope as u32,
        qk_rope_dim: qk_rope as u32,
        v_head_dim: vhd as u32,
        scale,
        mask: infr_core::graph::AttnMask::Causal,
        pos: 0,
        theta: 10000.0,
        freq_factors: None,
    });
    // Q: row-major [row, head, q_head_dim], values 1..=16.
    let qi: Vec<f32> = (1..=((rows * nh * q_head_dim) as i32))
        .map(|x| x as f32)
        .collect();
    // K cache rows: latent=[10,11,12], k_pe_raw=[1,2] per row.
    let ki: Vec<f32> = (0..rows * key_len)
        .map(|i| {
            let col = i % key_len;
            if col < kv_lora {
                (10 + col) as f32
            } else {
                (1 + (col - kv_lora)) as f32
            }
        })
        .collect();
    // wk_b[h][latent_idx][nope_idx] i-fast: h=0 maps nope0->latent0, nope1->latent1; h=1 maps
    // nope0->latent1, nope1->latent2.
    let mut wk: Vec<f32> = vec![0f32; nh * kv_lora * qk_nope];
    let s = kv_lora * qk_nope; // per-head stride
    wk[0] = 1.0;
    wk[qk_nope + 1] = 1.0;
    wk[s + qk_nope] = 1.0;
    wk[s + 2 * qk_nope + 1] = 1.0;
    // wv_b[h][a][o]: identity per head.
    let mut wv: Vec<f32> = vec![0f32; nh * kv_lora * vhd];
    for h in 0..nh {
        let off = h * kv_lora * vhd;
        for a in 0..kv_lora.min(vhd) {
            wv[off + a * vhd + a] = 1.0;
        }
    }
    let bound = vec![
        (q, f32_bytes(&qi)),
        (k_cache, f16_bytes(&ki)),
        (wk_b, f32_bytes(&wk)),
        (wv_b, f32_bytes(&wv)),
    ];
    assert_parity(&g, &bound, dst, rows * nh * vhd, 1e-3);
}

// MLA ff parity — exercises the `mla_f16kv_ff` Metal kernel (freq_factors bound): the q_pe rope
// angle is divided by the per-pair divisor ff (a real YaRN-style divisor). Identical synthetic
// data to `mla_parity`; ff = [0.5] halves the single qk_rope/2=1 pair's angle.
#[test]
#[ignore = "requires a Metal GPU"]
fn mla_ff_parity() {
    let (rows, nh, kv_lora, qk_nope, qk_rope, vhd) =
        (2usize, 2usize, 3usize, 2usize, 2usize, 2usize);
    let key_len = kv_lora + qk_rope; // 5
    let q_head_dim = qk_nope + qk_rope; // 4
    let mut g = Graph::new();
    let q = g.input(TensorDesc::new(vec![rows * nh * q_head_dim], DType::F32));
    // F16 KV cache — the Metal kernel reads `device const half*`, one half per element.
    let k_cache = g.input(TensorDesc::new(vec![rows * key_len], DType::F16));
    let wk_b = g.weight(TensorDesc::new(vec![nh * kv_lora * qk_nope], DType::F32));
    let wv_b = g.weight(TensorDesc::new(vec![nh * kv_lora * vhd], DType::F32));
    // freq_factors: one divisor per rope pair (qk_rope/2 = 1 pair here).
    let ff = g.input(TensorDesc::new(vec![qk_rope / 2], DType::F32));
    let dst = g.output(TensorDesc::new(vec![rows * nh * vhd], DType::F32));
    let scale = 1.0 / ((qk_nope + qk_rope) as f32).sqrt(); // 0.5
    g.push(Op::Mla {
        q,
        k_cache,
        wk_b,
        wv_b,
        dst,
        rows: rows as u32,
        kv_len: rows as u32, // attend to all rows
        n_head: nh as u32,
        q_head_dim: q_head_dim as u32,
        kv_lora_rank: kv_lora as u32,
        qk_nope_dim: qk_nope as u32,
        qk_rope_dim: qk_rope as u32,
        v_head_dim: vhd as u32,
        scale,
        mask: infr_core::graph::AttnMask::Causal,
        pos: 0,
        theta: 10000.0,
        freq_factors: Some(ff),
    });
    // Q: row-major [row, head, q_head_dim], values 1..=16.
    let qi: Vec<f32> = (1..=((rows * nh * q_head_dim) as i32))
        .map(|x| x as f32)
        .collect();
    // K cache rows: latent=[10,11,12], k_pe_raw=[1,2] per row.
    let ki: Vec<f32> = (0..rows * key_len)
        .map(|i| {
            let col = i % key_len;
            if col < kv_lora {
                (10 + col) as f32
            } else {
                (1 + (col - kv_lora)) as f32
            }
        })
        .collect();
    // wk_b[h][latent_idx][nope_idx] i-fast: h=0 maps nope0->latent0, nope1->latent1; h=1 maps
    // nope0->latent1, nope1->latent2.
    let mut wk: Vec<f32> = vec![0f32; nh * kv_lora * qk_nope];
    let s = kv_lora * qk_nope; // per-head stride
    wk[0] = 1.0;
    wk[qk_nope + 1] = 1.0;
    wk[s + qk_nope] = 1.0;
    wk[s + 2 * qk_nope + 1] = 1.0;
    // wv_b[h][a][o]: identity per head.
    let mut wv: Vec<f32> = vec![0f32; nh * kv_lora * vhd];
    for h in 0..nh {
        let off = h * kv_lora * vhd;
        for a in 0..kv_lora.min(vhd) {
            wv[off + a * vhd + a] = 1.0;
        }
    }
    // Divisor 0.5 halves the rope angle — a real YaRN-style divisor.
    let bound = vec![
        (q, f32_bytes(&qi)),
        (k_cache, f16_bytes(&ki)),
        (wk_b, f32_bytes(&wk)),
        (wv_b, f32_bytes(&wv)),
        (ff, f32_bytes(&[0.5f32])),
    ];
    assert_parity(&g, &bound, dst, rows * nh * vhd, 1e-3);
}

/// One `mla_mask_ring_parity` case.
struct MlaCase {
    name: &'static str,
    rows: usize,
    pos: u32,
    kv_len: usize,
    /// Ring row capacity — the K cache tensor is declared `cap * key_len` wide, which is where
    /// both backends read `cache_cap_rows` from. `cap < kv_len` is a genuinely WRAPPED cache, so
    /// `mla_f16kv_one`'s `(lo + jj) % cap` has to fold.
    cap: usize,
    mask: infr_core::graph::AttnMask,
}

/// `mla_f16kv` over the axes `mla_parity` never moves: a WRAPPED ring cache (`cap < kv_len`),
/// `AttnMask::SlidingWindow`, `AttnMask::Canvas` and a non-zero `pos` — the gap recorded as
/// `docs/backlog.md` B46's second bullet. Same case table, same dims and same synthetic data as
/// `infr-llama`'s `mla_mask_ring_parity` and `infr-vulkan`'s
/// `mla_ring_and_mask_matches_cpu_reference`, so a Metal-only divergence is identifiable by case.
///
/// This one compares against the CPU arm (the file's `assert_parity` convention) rather than
/// against a from-semantics reference; the CPU arm's own mask and ring arithmetic is what
/// `mla_mask_ring_parity` pins semantically, so the chain closes there.
///
/// **Canvas is covered at `lo = 0` only.** `MlaParams` carries no canvas `lo` — `mla_f16kv_one`
/// hardcodes `lo = 0u` on the `mask_type == 2` arm — while the CPU arm honours `AttnMask::Canvas
/// { lo }`. A `lo > 0` case would fail here, and it is a kernel defect, not a test gap; see B46.
#[test]
#[ignore = "requires a Metal GPU"]
fn mla_mask_ring_parity() {
    // key_len = kv_lora + qk_rope is EVEN so this table also runs on Vulkan, whose f16 cache is
    // read as u32-packed f16 PAIRS.
    let (nh, kv_lora, qk_nope, qk_rope, vhd) = (2usize, 4usize, 2usize, 2usize, 2usize);
    let key_len = kv_lora + qk_rope; // 6
    let q_head_dim = qk_nope + qk_rope; // 4
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let theta = 10000.0f32;
    // One-hot wk_b/wv_b in the READ convention both kernels use (`i` / `a` the FAST dim).
    let mut wk: Vec<f32> = vec![0f32; nh * kv_lora * qk_nope];
    let mut wv: Vec<f32> = vec![0f32; nh * kv_lora * vhd];
    for h in 0..nh {
        for i in 0..qk_nope {
            wk[h * kv_lora * qk_nope + i + ((h + i) % kv_lora) * qk_nope] = 1.0;
        }
        for o in 0..vhd {
            wv[h * kv_lora * vhd + (h + o) % kv_lora + o * kv_lora] = 1.0;
        }
    }
    // One distinct key per absolute position. Values are 1/16ths, so the f16 cache round-trip is
    // EXACT; and they are O(1), which keeps the softmax SOFT — under a near-one-hot softmax the
    // output is just the winning key's V, and adding or dropping a LOSING key (what an off-by-one
    // in `lo`/`hi` does) would leave the output unchanged.
    let key_at = |j: usize| -> Vec<f32> {
        (0..key_len)
            .map(|d| ((j * 7 + d * 3) % 13) as f32 / 16.0 + 0.125)
            .collect()
    };
    let q_at = |i: usize| ((i * 5 + 3) % 11) as f32 / 8.0 - 0.5;
    // Which ABSOLUTE positions a query at `abs` may attend to — from what each mask MEANS, used
    // here only to check each case stays inside what the ring can still hold.
    let attends = |mask: infr_core::graph::AttnMask, abs: usize, kv_len: usize| match mask {
        infr_core::graph::AttnMask::Causal => 0..(abs + 1).min(kv_len),
        infr_core::graph::AttnMask::SlidingWindow(w) => {
            (abs + 1).saturating_sub(w)..(abs + 1).min(kv_len)
        }
        infr_core::graph::AttnMask::Canvas { lo } => lo..kv_len,
    };

    let cases = [
        MlaCase {
            name: "causal pos=0, no wrap",
            rows: 2,
            pos: 0,
            kv_len: 2,
            cap: 2,
            mask: infr_core::graph::AttnMask::Causal,
        },
        MlaCase {
            name: "causal pos=3, no wrap",
            rows: 2,
            pos: 3,
            kv_len: 5,
            cap: 8,
            mask: infr_core::graph::AttnMask::Causal,
        },
        MlaCase {
            name: "sliding window w=3, pos=3, no wrap",
            rows: 2,
            pos: 3,
            kv_len: 5,
            cap: 8,
            mask: infr_core::graph::AttnMask::SlidingWindow(3),
        },
        // cap=5 over 14 positions is two full laps plus four rows, and the abs=12 row attends
        // 9..13 → rows 4,0,1,2: `lo` and `hi-1` land on OPPOSITE sides of the wrap boundary,
        // which a single lap starting at row 0 would not have caught.
        MlaCase {
            name: "sliding window w=4, wrapped ring (lo/hi straddle)",
            rows: 2,
            pos: 12,
            kv_len: 14,
            cap: 5,
            mask: infr_core::graph::AttnMask::SlidingWindow(4),
        },
        MlaCase {
            name: "sliding window w=cap=5, wrapped ring",
            rows: 1,
            pos: 13,
            kv_len: 14,
            cap: 5,
            mask: infr_core::graph::AttnMask::SlidingWindow(5),
        },
        MlaCase {
            name: "canvas lo=0, pos=3",
            rows: 2,
            pos: 3,
            kv_len: 5,
            cap: 8,
            mask: infr_core::graph::AttnMask::Canvas { lo: 0 },
        },
    ];

    for c in cases {
        // Ring writer: absolute position j lands in row j % cap, ascending, so a row reached more
        // than once keeps the LATER position.
        let mut cache = vec![0f32; c.cap * key_len];
        for j in 0..c.kv_len {
            cache[(j % c.cap) * key_len..][..key_len].copy_from_slice(&key_at(j));
        }
        // A ring holds only the last `cap` positions. If an attended position's row was reused by
        // a LATER one, the cache no longer holds that position's key at all and the case is
        // asking a question with no answer — catch that here, not as a mystery rel_err.
        for ti in 0..c.rows {
            let abs = c.pos as usize + ti;
            for j in attends(c.mask, abs, c.kv_len) {
                let last = (0..c.kv_len)
                    .rfind(|p| p % c.cap == j % c.cap)
                    .expect("attended position is inside 0..kv_len");
                assert_eq!(
                    last, j,
                    "{}: attended position {j} was overwritten by {last} in the ring — the case \
                     attends a wider span than cap={} holds",
                    c.name, c.cap
                );
            }
        }
        let qi: Vec<f32> = (0..c.rows * nh * q_head_dim).map(q_at).collect();

        let mut g = Graph::new();
        let q = g.input(TensorDesc::new(vec![c.rows * nh * q_head_dim], DType::F32));
        let k_cache = g.input(TensorDesc::new(vec![c.cap * key_len], DType::F16));
        let wk_b = g.weight(TensorDesc::new(vec![nh * kv_lora * qk_nope], DType::F32));
        let wv_b = g.weight(TensorDesc::new(vec![nh * kv_lora * vhd], DType::F32));
        let dst = g.output(TensorDesc::new(vec![c.rows * nh * vhd], DType::F32));
        g.push(Op::Mla {
            q,
            k_cache,
            wk_b,
            wv_b,
            dst,
            rows: c.rows as u32,
            kv_len: c.kv_len as u32,
            n_head: nh as u32,
            q_head_dim: q_head_dim as u32,
            kv_lora_rank: kv_lora as u32,
            qk_nope_dim: qk_nope as u32,
            qk_rope_dim: qk_rope as u32,
            v_head_dim: vhd as u32,
            scale,
            mask: c.mask,
            pos: c.pos,
            theta,
            freq_factors: None,
        });
        let bound = vec![
            (q, f32_bytes(&qi)),
            (k_cache, f16_bytes(&cache)),
            (wk_b, f32_bytes(&wk)),
            (wv_b, f32_bytes(&wv)),
        ];
        println!("MLA metal case: {}", c.name);
        assert_parity(&g, &bound, dst, c.rows * nh * vhd, 1e-3);
    }
}
