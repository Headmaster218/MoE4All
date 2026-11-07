// DiffusionGemma perf slice 3 (docs/diffusion-gemma.md): validates the fused GPU entropy-bound
// sampler reduction (`dg_eb_sample.comp` / `Backend::eb_sample_reduce`) against a host reference
// that's a straight port of `diffusion.rs::denoise_block`'s `per_pos` closure — same math, same
// iteration order — so a divergence here means the GPU kernel, not the host sampler, is wrong.
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

/// Host reference: EXACTLY `diffusion.rs::denoise_block`'s `per_pos` closure (argmax over
/// `raw*temp_inv`, softmax entropy, forward-CDF multinomial draw against `u`).
fn host_reduce(row: &[f32], temp_inv: f32, u: f32) -> (u32, f32, u32) {
    let vocab = row.len();
    let mut m = f32::NEG_INFINITY;
    let mut amax = 0u32;
    for (v, &raw) in row.iter().enumerate() {
        let z = raw * temp_inv;
        if z > m {
            m = z;
            amax = v as u32;
        }
    }
    let mut escratch = vec![0f32; vocab];
    let mut z_sum = 0f32;
    for (v, &raw) in row.iter().enumerate() {
        let e = (raw * temp_inv - m).exp();
        escratch[v] = e;
        z_sum += e;
    }
    let target = u * z_sum;
    let mut cum = 0f32;
    let mut h = 0f32;
    let mut sampled = (vocab - 1) as u32;
    let mut picked = false;
    for (v, &e) in escratch.iter().enumerate() {
        let p = e / z_sum;
        if p > 0.0 {
            h -= p * p.ln();
        }
        cum += e;
        if !picked && cum >= target {
            sampled = v as u32;
            picked = true;
        }
    }
    (amax, h, sampled)
}

/// Deterministic xorshift-ish LCG for reproducible test logits (no external RNG dep in this crate).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // roughly [-1, 1)
    }
    fn next_u01(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 // [0, 1)
    }
}

/// Realistic-shape logits: mostly small noise plus a handful of "hot" spikes — like a real
/// model's vocab logits (a few plausible tokens dominate, the rest is noise floor), UNLIKE flat
/// random noise across the whole vocab. This matters at `vocab` ~ 262144: a near-uniform
/// distribution's CDF crossing point is maximally sensitive to floating-point summation-order
/// drift (the GPU's chunked reduction and the host's flat serial sum are algebraically the same
/// sum but not bit-identical — non-associative fp), which can shift the "exact index" the
/// multinomial draw lands on by a few slots even though both sides agree on the underlying
/// distribution to high precision. A peaked distribution (this sampler's actual operating regime:
/// `denoiser[pos]` is only ever CONSUMED for LOW-entropy — i.e. peaked — accepted positions, see
/// `denoise_block`) keeps the crossing point in a well-conditioned region, so the exact-match bar
/// holds without weakening it.
fn peaked_logits(rng: &mut Lcg, vocab: usize, n_hot: usize) -> Vec<f32> {
    let mut row: Vec<f32> = (0..vocab).map(|_| rng.next_f32() * 0.5).collect();
    for _ in 0..n_hot {
        let idx = (rng.next_u01() * vocab as f32) as usize % vocab;
        row[idx] += 20.0 + rng.next_f32() * 5.0;
    }
    row
}

fn run_case(be: &VulkanBackend, rows: usize, vocab: usize, temp_inv: f32, seed: u64) {
    let mut rng = Lcg(seed | 1);
    let logits: Vec<f32> = (0..rows * vocab).map(|_| rng.next_f32() * 8.0).collect();
    run_case_with(be, rows, vocab, temp_inv, &mut rng, logits);
}

fn run_case_peaked(be: &VulkanBackend, rows: usize, vocab: usize, temp_inv: f32, seed: u64) {
    let mut rng = Lcg(seed | 1);
    let mut logits = Vec::with_capacity(rows * vocab);
    for _ in 0..rows {
        logits.extend(peaked_logits(&mut rng, vocab, 5));
    }
    run_case_with(be, rows, vocab, temp_inv, &mut rng, logits);
}

fn run_case_with(
    be: &VulkanBackend,
    rows: usize,
    vocab: usize,
    temp_inv: f32,
    rng: &mut Lcg,
    logits: Vec<f32>,
) {
    let u: Vec<f32> = (0..rows).map(|_| rng.next_u01()).collect();

    let want: Vec<(u32, f32, u32)> = (0..rows)
        .map(|r| host_reduce(&logits[r * vocab..(r + 1) * vocab], temp_inv, u[r]))
        .collect();

    let logits_buf = be
        .alloc(logits.len() * 4, BufferUsage::Activations)
        .unwrap();
    be.upload(logits_buf.as_ref(), bytemuck::cast_slice(&logits))
        .unwrap();
    let u_buf = be.alloc(u.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(u_buf.as_ref(), bytemuck::cast_slice(&u)).unwrap();
    let argmax_buf = be.alloc(rows * 4, BufferUsage::Activations).unwrap();
    let entropy_buf = be.alloc(rows * 4, BufferUsage::Activations).unwrap();
    let sampled_buf = be.alloc(rows * 4, BufferUsage::Activations).unwrap();

    let ok = be
        .eb_sample_reduce(
            logits_buf.as_ref(),
            u_buf.as_ref(),
            rows,
            vocab,
            temp_inv,
            argmax_buf.as_ref(),
            entropy_buf.as_ref(),
            sampled_buf.as_ref(),
        )
        .unwrap();
    assert!(ok, "Vulkan backend should implement eb_sample_reduce");

    let mut argmax_out = vec![0u8; rows * 4];
    be.download(argmax_buf.as_ref(), &mut argmax_out).unwrap();
    let argmax_got: &[u32] = bytemuck::cast_slice(&argmax_out);
    let mut entropy_out = vec![0u8; rows * 4];
    be.download(entropy_buf.as_ref(), &mut entropy_out).unwrap();
    let entropy_got: &[f32] = bytemuck::cast_slice(&entropy_out);
    let mut sampled_out = vec![0u8; rows * 4];
    be.download(sampled_buf.as_ref(), &mut sampled_out).unwrap();
    let sampled_got: &[u32] = bytemuck::cast_slice(&sampled_out);

    for r in 0..rows {
        let (want_amax, want_h, want_sampled) = want[r];
        assert_eq!(
            argmax_got[r], want_amax,
            "row {r} (rows={rows} vocab={vocab}): argmax got {} want {want_amax}",
            argmax_got[r]
        );
        let rel = (entropy_got[r] - want_h).abs() / want_h.abs().max(1e-6);
        assert!(
            rel < 1e-3 || (entropy_got[r] - want_h).abs() < 1e-4,
            "row {r} (rows={rows} vocab={vocab}): entropy got {} want {want_h} (rel {rel})",
            entropy_got[r]
        );
        assert_eq!(
            sampled_got[r], want_sampled,
            "row {r} (rows={rows} vocab={vocab}): sampled got {} want {want_sampled}",
            sampled_got[r]
        );
    }
}

#[test]
fn dg_eb_sample_matches_host_reference() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // Small, non-power-of-two vocab (exercises the tail of both the strided and the 256-chunked
    // passes) at a few temperatures.
    run_case(&be, 5, 1000, 1.0, 1);
    run_case(&be, 3, 777, 0.5, 2);
    run_case(&be, 8, 2049, 2.0, 3);
    // Production-scale shape: diffusiongemma's real canvas/vocab (262144) — the case this slice
    // exists for. Peaked logits (see `peaked_logits`'s doc) — this sampler's real operating
    // regime — keep the CDF crossing well-conditioned at this width. Fewer rows to keep it fast.
    run_case_peaked(&be, 4, 262144, 1.3, 4);
    eprintln!("Vulkan dg_eb_sample OK (matches host reference across all shapes)");
}
