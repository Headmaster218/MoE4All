//! Determinism regression for the batched-MoE dp4a expert GEMMs (`matmul_mmq_experts`), at the
//! shapes that select the BN=128 `_xp32w` wide tile — the configuration that exposed a
//! shared-memory data race in the whole `native_gemm_mmq_*` family: the per-column scale arrays
//! (`Bdl`/`Bmm`) staged in shared were read AFTER the loop's second `barrier()`, racing the next
//! k-iteration's staging writes from faster threads. Under the wide tile (256 threads, 8 waves,
//! twice the Bdl range) the window hit reliably: DiffusionGemma's 128-expert denoise (Q4_K
//! gate_up rows=256 k=2816 n=1408 — the exact first config below) amplified one corrupted
//! 4-row×BN slab into ALL 67M logits differing between two identical sessions
//! (`gpu_diffusion_gemma_denoise_replay_matches_static`).
//!
//! The check is pure determinism: the SAME dispatch recorded three times into three outputs must
//! produce bitwise-identical results. Pre-fix this failed on the first attempt (512 elements — one
//! tr-group's 4×128 output slab — differing between two dispatches in one submission).
//!
//! Run: `cargo test -p infr-vulkan --release --test mmq_wide_bn_determinism -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::VulkanBackend;

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

/// Synthetic VALID Q4_K bank (144 B / 256 elems): random scales/quants, d/dmin patched small.
fn synth_q4k(n_elems: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / 256 * 144);
    for _ in 0..n_elems / 256 {
        let d = half::f16::from_f32(0.004 + (rng.byte() as f32) * 1e-5);
        let m = half::f16::from_f32(0.002 + (rng.byte() as f32) * 1e-5);
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&m.to_le_bytes());
        out.extend((0..140).map(|_| rng.byte()));
    }
    out
}

/// Synthetic VALID Q6_K bank (210 B / 256 elems): random quants/scales, d patched small.
fn synth_q6k(n_elems: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / 256 * 210);
    for _ in 0..n_elems / 256 {
        out.extend((0..208).map(|_| rng.byte()));
        let d = half::f16::from_f32(0.004 + (rng.byte() as f32) * 1e-5);
        out.extend_from_slice(&d.to_le_bytes());
    }
    out
}

fn run_config(be: &VulkanBackend, dtype: DType, label: &str) {
    // The DG denoise gate_up shape: 256 canvas rows × top-8 over 128 experts, k=2816, n=1408
    // (n%128==0 + avg_rows=16 → the small wide tile `_xp32w`).
    let (rows, k, n, n_expert, n_used) = (256usize, 2816usize, 1408usize, 128usize, 8usize);
    let n_pairs = rows * n_used;
    // Ragged counts summing to n_pairs, each <= rows (the grid-sizing bound).
    let mut rng = Rng(0x5eed);
    let mut counts_host = vec![0u32; n_expert];
    let mut left = n_pairs as u32;
    for c in counts_host.iter_mut() {
        let want = ((rng.byte() as u32) % 33).min(left);
        *c = want;
        left -= want;
    }
    counts_host[n_expert - 1] += left;
    assert!(counts_host.iter().all(|&c| c <= rows as u32));
    let mut offsets_host = vec![0u32; n_expert];
    for e in 1..n_expert {
        offsets_host[e] = offsets_host[e - 1] + counts_host[e - 1];
    }

    let stride = k * n; // elements per expert
    let synth = |e: usize| match dtype {
        DType::Q4K => synth_q4k(stride, 0x9000 + e as u64),
        DType::Q6K => synth_q6k(stride, 0x9000 + e as u64),
        _ => unreachable!(),
    };
    let mut bank: Vec<u8> = Vec::new();
    for e in 0..n_expert {
        bank.extend_from_slice(&synth(e));
    }
    let bank = pad_to_u32_align(&bank);
    let w = be.alloc(bank.len(), BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &bank).unwrap();

    let f = |i: usize| ((i as f32) * 0.11).sin() * 0.15 + 0.02;
    let x: Vec<f32> = (0..n_pairs * k).map(f).collect();
    let xbuf = be.alloc(n_pairs * k * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let npad = n_pairs.div_ceil(64) * 64 + 64; // GEMM As-stage overread padding
    let qa = be.alloc(npad * k, BufferUsage::Activations).unwrap();
    let qda = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let qsa = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let outs: Vec<_> = (0..3)
        .map(|_| be.alloc(npad * n * 4, BufferUsage::Activations).unwrap())
        .collect();

    let mk_u32 = |v: &[u32]| {
        let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    };
    let counts = mk_u32(&counts_host);
    let offsets = mk_u32(&offsets_host);

    let rec = be.recorder().unwrap();
    rec.quant_q8(
        xbuf.as_ref(),
        qa.as_ref(),
        qda.as_ref(),
        qsa.as_ref(),
        n_pairs,
        k,
    );
    for out in &outs {
        rec.matmul_mmq_experts(
            dtype,
            "expert_gateup",
            qa.as_ref(),
            qda.as_ref(),
            // Q4_K is min-carrying (binds sact); Q6_K is symmetric.
            matches!(dtype, DType::Q4K).then(|| qsa.as_ref() as _),
            w.as_ref(),
            0,
            stride,
            counts.as_ref(),
            offsets.as_ref(),
            out.as_ref(),
            rows,
            k,
            n,
            n_expert,
            n_used,
        );
    }
    rec.finish().unwrap();

    let dl = |b: &dyn infr_core::backend::Buffer| -> Vec<f32> {
        let mut v = vec![0f32; npad * n];
        be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
        v.truncate(n_pairs * n);
        v
    };
    let a = dl(outs[0].as_ref());
    assert!(
        a.iter().all(|v| v.is_finite()),
        "{label}: non-finite output"
    );
    for (run, out) in outs.iter().enumerate().skip(1) {
        let b = dl(out.as_ref());
        let mut ndiff = 0usize;
        let mut maxabs = 0f32;
        let mut first = usize::MAX;
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            if p != q {
                if first == usize::MAX {
                    first = i;
                }
                ndiff += 1;
                maxabs = maxabs.max((p - q).abs());
            }
        }
        assert_eq!(
            ndiff,
            0,
            "{label}: wide-tile expert GEMM nondeterministic — dispatch 0 vs {run}: {ndiff}/{} \
             differ, maxabs={maxabs}, first at row {} col {}",
            a.len(),
            first / n,
            first % n
        );
    }
    println!("{label}: OK (3 dispatches bitwise identical)");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmq_wide_bn_expert_gemm_deterministic() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    run_config(&be, DType::Q4K, "q4k xp32w");
    run_config(&be, DType::Q6K, "q6k xp32w");
}
