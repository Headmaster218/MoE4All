//! Parity + determinism for the NON-COOPMAT prefill GEMM tier's kernels (adapter.rs
//! `nc_mmq`/`nc_fma` — the Intel Arc route, exercisable anywhere since these kernels use no
//! coopmat):
//!
//!  • the 12 NEWLY WIRED dense (non-expert-grid) dp4a mmq GEMM builds (`matmul_mmq` over every
//!    `MOE_MMQ_DTYPES` member — Q4_K/Q6_K's dense builds pre-existed but ride the same dispatch
//!    here, so all 16 run — incl. the IQ2_S/IQ3_S grid pair);
//!  • the 3 fma-warp float GEMMs (`matmul_fma`: f16/bf16/f32 weights, native_gemm_fma.comp).
//!
//! Each config dispatches the SAME GEMM three times in one submission — bitwise-identical
//! outputs required (the mmq barrier-race lesson: goldens can't catch intra-dispatch races; see
//! `mmq_wide_bn_determinism.rs`, the template) — and checks the first output against a host
//! reference: `infr_gguf::dequant::dequant_block` weights × the GPU's own downloaded int8
//! activation codes for mmq (the exact-reference trick from `moe_mmq_fp4_parity.rs`), plain f32
//! dot for fma (whose weights are bit-exact on both sides). Min-carrying mmq dtypes
//! (Q4_K/Q5_K/Q4_1/Q5_1 via the f16 `sact` Σx term, Q2_K via its self-computed one) keep the
//! coarser tolerance the other mmq parity tests carry; symmetric dtypes decompose exactly.
//!
//! Run: `cargo test -p infr-vulkan --release --test nc_gemm_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::VulkanBackend;

/// Deterministic byte stream so failures reproduce.
struct Rng(u64);
impl Rng {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// (elements per block, bytes per block) for every mmq dtype under test.
/// `(elements, bytes)` per block, from the shared decode spec (`infr_core::decode_spec` via
/// `infr_gguf::block_layout`) — not a hand-kept local table.
fn block_geom(dt: DType) -> (usize, usize) {
    infr_gguf::block_layout(dt)
}

/// Synthetic VALID bank: random payload bytes with the scale fields patched small (a valid
/// encoding for every format here — quant/code bits cover their full ranges), decodable by the
/// production host dequant (`dequant_block`).
fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = block_geom(dt);
    assert_eq!(n_elems % epb, 0);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
        let d16 = |r: &mut Rng| half::f16::from_f32(0.004 + (r.byte() as f32) * 1e-5).to_le_bytes();
        let m16 = |r: &mut Rng| half::f16::from_f32(0.002 + (r.byte() as f32) * 1e-5).to_le_bytes();
        match dt {
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Iq4Nl | DType::Q2_0 => {
                b[0..2].copy_from_slice(&d16(&mut rng))
            }
            DType::Q4_1 | DType::Q5_1 => {
                b[0..2].copy_from_slice(&d16(&mut rng));
                b[2..4].copy_from_slice(&m16(&mut rng));
            }
            DType::Q2K => {
                b[80..82].copy_from_slice(&d16(&mut rng));
                b[82..84].copy_from_slice(&m16(&mut rng));
            }
            DType::Q3K => b[108..110].copy_from_slice(&d16(&mut rng)),
            DType::Q4K | DType::Q5K => {
                b[0..2].copy_from_slice(&d16(&mut rng));
                b[2..4].copy_from_slice(&m16(&mut rng));
            }
            DType::Q6K => b[208..210].copy_from_slice(&d16(&mut rng)),
            DType::Iq4Xs => b[0..2].copy_from_slice(&d16(&mut rng)),
            // grid i-quants: leading f16 d; grid-index/sign bits cover their full table ranges
            DType::Iq2S | DType::Iq3S => b[0..2].copy_from_slice(&d16(&mut rng)),
            // e8m0 scale byte: 122..=125 → 2^-6..2^-3
            DType::Mxfp4 => b[0] = 122 + (rng.byte() & 3),
            // ue4m3 scale bytes: [0x18, 0x37] → small positive, never 0/0x7F
            DType::Nvfp4 => {
                for s in b.iter_mut().take(4) {
                    *s = 0x18 + (rng.byte() & 0x1F);
                }
            }
            other => panic!("no synth for {other:?}"),
        }
        out.extend_from_slice(&b);
    }
    out
}

/// The GEMM shape shared by every config: m NOT a tile multiple (exercises the padded row
/// range), k covers 2 superblocks, n two BN=64 column tiles.
const M: usize = 100;
const K: usize = 512;
const N: usize = 128;

fn download_rows(be: &VulkanBackend, b: &dyn Buffer, mpad: usize) -> Vec<f32> {
    let mut v = vec![0f32; mpad * N];
    be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
    v.truncate(M * N);
    v
}

fn assert_deterministic(be: &VulkanBackend, outs: &[Box<dyn Buffer>], mpad: usize, label: &str) {
    let a = download_rows(be, outs[0].as_ref(), mpad);
    assert!(a.iter().all(|v| v.is_finite()), "{label}: non-finite");
    for (run, out) in outs.iter().enumerate().skip(1) {
        let b = download_rows(be, out.as_ref(), mpad);
        let ndiff = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        assert_eq!(
            ndiff,
            0,
            "{label}: nondeterministic — dispatch 0 vs {run}: {ndiff}/{} differ",
            a.len()
        );
    }
}

fn assert_parity(got: &[f32], want: &[f32], tol: f32, label: &str) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < tol + tol * w.abs(),
            "{label}: mismatch at row {} col {}: got {g} want {w}",
            i / N,
            i % N
        );
    }
}

fn run_mmq(be: &VulkanBackend, dt: DType) {
    let label = format!("{dt:?} dense mmq");
    let bank = synth_bank(dt, N * K, 0x5eed ^ dt as u64);
    let w_host = infr_gguf::dequant::dequant_block(dt, &bank).unwrap();
    let padded = pad_to_u32_align(&bank);
    let w = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &padded).unwrap();

    let x: Vec<f32> = (0..M * K)
        .map(|i| (i as f32 * 0.11).sin() * 0.15 + 0.02)
        .collect();
    let xbuf = be.alloc(M * K * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let mpad = M.div_ceil(64) * 64;
    // zero-init allocs: the GEMM's As stage reads rows M..mpad (zeros, results discarded).
    let qa = be.alloc(mpad * K, BufferUsage::Activations).unwrap();
    let dact = be
        .alloc(mpad * (K / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let sact = be
        .alloc(mpad * (K / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let outs: Vec<_> = (0..3)
        .map(|_| be.alloc(mpad * N * 4, BufferUsage::Activations).unwrap())
        .collect();

    let rec = be.recorder().unwrap();
    rec.quant_q8(
        xbuf.as_ref(),
        qa.as_ref(),
        dact.as_ref(),
        sact.as_ref(),
        M,
        K,
    );
    for out in &outs {
        rec.matmul_mmq(
            dt,
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            w.as_ref(),
            0,
            out.as_ref(),
            M,
            K,
            N,
        );
    }
    rec.finish().unwrap();

    assert_deterministic(be, &outs, mpad, &label);

    // Host reference over the GPU's ACTUAL quantized activations (download + dequantize the int8
    // codes) — the dp4a decomposition then matches to float-association noise for symmetric
    // dtypes; min-carrying ones add the f16 `sact` rounding, hence their coarser tolerance.
    let mut cb = vec![0u8; mpad * K];
    be.download(qa.as_ref(), &mut cb).unwrap();
    let mut sb = vec![0u8; mpad * (K / 32) * 2];
    be.download(dact.as_ref(), &mut sb).unwrap();
    let xq: Vec<f32> = (0..M * K)
        .map(|i| {
            let (r, c) = (i / K, i % K);
            let si = (r * (K / 32) + c / 32) * 2;
            let d = half::f16::from_le_bytes([sb[si], sb[si + 1]]).to_f32();
            (cb[r * K + c] as i8 as f32) * d
        })
        .collect();
    let mut want = vec![0f32; M * N];
    for r in 0..M {
        let xr = &xq[r * K..(r + 1) * K];
        for o in 0..N {
            want[r * N + o] = w_host[o * K..(o + 1) * K]
                .iter()
                .zip(xr)
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    let min_carrying = infr_core::tensor::moe_mmq_needs_sact(dt) || matches!(dt, DType::Q2K);
    let tol = if min_carrying { 5e-2 } else { 1e-3 };
    let got = download_rows(be, outs[0].as_ref(), mpad);
    assert_parity(&got, &want, tol, &label);
    println!("{label}: OK");
}

fn run_fma(be: &VulkanBackend, dt: DType, w_base: usize) {
    let label = format!("{dt:?} fma-warp (w_base={w_base})");
    let mut rng = Rng(0xfab ^ dt as u64);
    // Weight values built directly in the storage format — bit-exact on both sides.
    let n_w = w_base + N * K;
    let (bytes, w_host): (Vec<u8>, Vec<f32>) = match dt {
        DType::F16 => {
            let mut by = Vec::with_capacity(n_w * 2);
            let mut ho = Vec::with_capacity(n_w);
            for _ in 0..n_w {
                // sign | exp 11..14 of 31 (2^-4..2^-1) | mantissa
                let bits: u16 = (((rng.byte() & 1) as u16) << 15)
                    | ((11 + (rng.byte() & 3) as u16) << 10)
                    | ((rng.byte() as u16) << 2 | (rng.byte() & 3) as u16);
                by.extend_from_slice(&bits.to_le_bytes());
                ho.push(half::f16::from_bits(bits).to_f32());
            }
            (by, ho)
        }
        DType::Bf16 => {
            let mut by = Vec::with_capacity(n_w * 2);
            let mut ho = Vec::with_capacity(n_w);
            for _ in 0..n_w {
                // sign | exp 0x7B..0x7E (2^-4..2^-1) | mantissa(7)
                let bits: u16 = (((rng.byte() & 1) as u16) << 15)
                    | (((0x7B + (rng.byte() & 3) as u16) & 0xFF) << 7)
                    | (rng.byte() & 0x7F) as u16;
                by.extend_from_slice(&bits.to_le_bytes());
                ho.push(f32::from_bits((bits as u32) << 16));
            }
            (by, ho)
        }
        DType::F32 => {
            let mut by = Vec::with_capacity(n_w * 4);
            let mut ho = Vec::with_capacity(n_w);
            for _ in 0..n_w {
                let v = ((rng.byte() as f32) - 127.5) * 0.004;
                by.extend_from_slice(&v.to_le_bytes());
                ho.push(v);
            }
            (by, ho)
        }
        other => panic!("no fma build for {other:?}"),
    };
    let w = be.alloc(bytes.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &bytes).unwrap();

    let x: Vec<f32> = (0..M * K)
        .map(|i| (i as f32 * 0.13).sin() * 0.2 - 0.01)
        .collect();
    let xbuf = be.alloc(M * K * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let mpad = M.div_ceil(64) * 64;
    let outs: Vec<_> = (0..3)
        .map(|_| be.alloc(mpad * N * 4, BufferUsage::Activations).unwrap())
        .collect();

    let rec = be.recorder().unwrap();
    for out in &outs {
        rec.matmul_fma(dt, xbuf.as_ref(), w.as_ref(), w_base, out.as_ref(), M, K, N);
    }
    rec.finish().unwrap();

    assert_deterministic(be, &outs, mpad, &label);

    let mut want = vec![0f32; M * N];
    for r in 0..M {
        for o in 0..N {
            want[r * N + o] = w_host[w_base + o * K..w_base + (o + 1) * K]
                .iter()
                .zip(&x[r * K..(r + 1) * K])
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    let got = download_rows(be, outs[0].as_ref(), mpad);
    assert_parity(&got, &want, 1e-3, &label);
    println!("{label}: OK");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_mmq_dense_gemm_parity_and_determinism() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // NB: no i8_dot gate — production callers check `caps().i8_dot`; this targets dp4a dev boxes.
    for &dt in infr_core::tensor::MOE_MMQ_DTYPES {
        run_mmq(&be, dt);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn nc_fma_gemm_parity_and_determinism() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    run_fma(&be, DType::F16, 0);
    run_fma(&be, DType::Bf16, 0);
    run_fma(&be, DType::F32, 0);
    // Non-zero element base (a bf16 fused-QKV slice / streamed-slot offset rides `w_base`).
    run_fma(&be, DType::Bf16, K);
}
