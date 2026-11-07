//! GPU↔host parity for the NEW batched dp4a mmq expert-GEMM dtypes: MXFP4 and NVFP4 (the
//! MXFP4_MOE quant family — signed E2M1 codebook → dp4a, IQ4_NL's treatment; NVFP4 additionally
//! splits each 32-block into two dp4a halves for its per-16 UE4M3 sub-block scales). Two
//! configurations per (resident, paged) pair so BOTH row-tile variants of every new kernel run:
//! small-avg (9 packed rows over 5 experts, ragged 4/3/2 buckets) → the `_xp32`/`_xpg32` BM=32
//! builds, with pager eviction churn on the paged run; large-avg (64 rows over 2 experts) → the
//! `_xp`/`_xpg` BM=64 builds.
//!
//! Each config chains quant → MXFP4 gate-GEMM → re-quant → NVFP4 down-GEMM in ONE recorder (the
//! dependency chain catches missing barriers, not just wrong math — the pager_gemv_multi_parity
//! lesson). Host reference: `infr_gguf::dequant::dequant_block` weights × the GPU's own
//! downloaded activation codes (see the exact-reference note at the download site).
//!
//! Run: `cargo test -p infr-vulkan --test moe_mmq_fp4_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::pager::GpuPager;
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

/// Synthetic VALID MXFP4 bank: random nibbles, e8m0 scale patched small (2^-6..2^-3).
fn synth_mxfp4(n_elems: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / 32 * 17);
    for _ in 0..n_elems / 32 {
        out.push(122 + (rng.byte() & 3));
        out.extend((0..16).map(|_| rng.byte()));
    }
    out
}

/// Synthetic VALID NVFP4 bank: random nibbles, ue4m3 scales patched small (never 0/0x7F).
fn synth_nvfp4(n_elems: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / 64 * 36);
    for _ in 0..n_elems / 64 {
        out.extend((0..4).map(|_| 0x18 + (rng.byte() & 0x1F)));
        out.extend((0..32).map(|_| rng.byte()));
    }
    out
}

/// Host mirror of `quant_q8` (per-32-block symmetric int8): the GEMM's activation operand.
fn quant_act(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = half::f16::from_f32(amax / 127.0).to_f32();
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for &v in blk {
            out.push((v * id).round().clamp(-127.0, 127.0) * d);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_config(
    be: &VulkanBackend,
    // `Some((slots, expect_evict))` = paged through a GpuPager with that many arena slots
    // (ALL routed experts must fit simultaneously — the within-batch safety invariant — so churn
    // is only possible when some experts go unrouted); `None` = resident stacked banks.
    paged: Option<(usize, bool)>,
    n_expert: usize,
    counts_host: &[u32],
    offsets_host: &[u32],
    rows: usize,
    label: &str,
) {
    // gate: MXFP4 [n, k] per expert; down: NVFP4 [k, n] per expert (down's GEMM k = gate's n).
    let (k, n) = (256usize, 64usize);
    let gate_stride = k * n;
    let down_stride = n * k;
    let gate_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_mxfp4(gate_stride, 0xabc0 + e as u64))
        .collect();
    let down_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_nvfp4(down_stride, 0xdef0 + e as u64))
        .collect();
    let gate_host: Vec<Vec<f32>> = gate_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Mxfp4, b).unwrap())
        .collect();
    let down_host: Vec<Vec<f32>> = down_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Nvfp4, b).unwrap())
        .collect();

    let n_pairs: usize = counts_host.iter().sum::<u32>() as usize;
    let npad = n_pairs.div_ceil(64) * 64 + 64; // GEMM As-stage overread padding
    let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.15;
    let x: Vec<f32> = (0..n_pairs * k).map(|i| f(i, 0.11) + 0.02).collect();
    let xq = quant_act(&x);

    let mk_u32 = |v: &[u32]| {
        let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    };
    let counts = mk_u32(counts_host);
    let offsets = mk_u32(offsets_host);
    let xbuf = be.alloc(n_pairs * k * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    // zero-init allocs: padded overread rows read zeros, results discarded at the store.
    let qa = be.alloc(npad * k, BufferUsage::Activations).unwrap();
    let qda = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let qsa = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let gbuf = be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();
    let dqa = be.alloc(npad * n, BufferUsage::Activations).unwrap();
    let dda = be
        .alloc(npad * (n / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let dsa = be
        .alloc(npad * (n / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let ybuf = be.alloc(npad * k * 4, BufferUsage::Activations).unwrap();

    // Resident stacked banks OR pagers with eviction churn (fill slots with the first experts,
    // then route to the tail ones so LRU eviction + slot reuse happens before the dispatch).
    let mut gate_pager = None;
    let mut down_pager = None;
    let mut gate_res = None;
    let mut down_res = None;
    if let Some((slots, _)) = paged {
        let gate_sb = gate_stride / 32 * 17;
        let down_sb = down_stride / 64 * 36;
        let mut gp = GpuPager::new(be, n_expert, slots, gate_sb).unwrap();
        let mut dp = GpuPager::new(be, n_expert, slots, down_sb).unwrap();
        let staging = be
            .alloc_uninit(gate_sb.max(down_sb), BufferUsage::Staging)
            .unwrap();
        // Warm with experts 0..slots, then make ALL routed experts resident (evicts the LRU).
        for pre in 0..slots as u32 {
            gp.ensure_resident(be, staging.as_ref(), pre, &gate_banks[pre as usize])
                .unwrap();
            dp.ensure_resident(be, staging.as_ref(), pre, &down_banks[pre as usize])
                .unwrap();
        }
        for e in 0..n_expert as u32 {
            if counts_host[e as usize] > 0 {
                gp.ensure_resident(be, staging.as_ref(), e, &gate_banks[e as usize])
                    .unwrap();
                dp.ensure_resident(be, staging.as_ref(), e, &down_banks[e as usize])
                    .unwrap();
            }
        }
        gp.flush_lut(be).unwrap();
        dp.flush_lut(be).unwrap();
        gate_pager = Some(gp);
        down_pager = Some(dp);
    } else {
        let cat = |banks: &[Vec<u8>]| {
            let flat: Vec<u8> = banks.concat();
            let padded = pad_to_u32_align(&flat);
            let b = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
            be.upload(b.as_ref(), &padded).unwrap();
            b
        };
        gate_res = Some(cat(&gate_banks));
        down_res = Some(cat(&down_banks));
    }

    let rec = be.recorder().unwrap();
    rec.quant_q8(
        xbuf.as_ref(),
        qa.as_ref(),
        qda.as_ref(),
        qsa.as_ref(),
        n_pairs,
        k,
    );
    match (&gate_pager, &gate_res) {
        (Some(gp), _) => rec.matmul_mmq_experts_paged(
            DType::Mxfp4,
            "expert_gateup",
            qa.as_ref(),
            qda.as_ref(),
            None, // MXFP4 is symmetric (signed codebook) — no `sact`
            gp.arena_addr(),
            gp.slot_bytes() as u32,
            gp.lut_buffer(),
            0,
            counts.as_ref(),
            offsets.as_ref(),
            gbuf.as_ref(),
            rows,
            k,
            n,
            n_expert,
            1,
        ),
        (_, Some(gr)) => rec.matmul_mmq_experts(
            DType::Mxfp4,
            "expert_gateup",
            qa.as_ref(),
            qda.as_ref(),
            None,
            gr.as_ref(),
            0,
            gate_stride,
            counts.as_ref(),
            offsets.as_ref(),
            gbuf.as_ref(),
            rows,
            k,
            n,
            n_expert,
            1,
        ),
        _ => unreachable!(),
    }
    rec.quant_q8(
        gbuf.as_ref(),
        dqa.as_ref(),
        dda.as_ref(),
        dsa.as_ref(),
        n_pairs,
        n,
    );
    match (&down_pager, &down_res) {
        (Some(dp), _) => rec.matmul_mmq_experts_paged(
            DType::Nvfp4,
            "expert_down",
            dqa.as_ref(),
            dda.as_ref(),
            None, // NVFP4 is symmetric too — no `sact`
            dp.arena_addr(),
            dp.slot_bytes() as u32,
            dp.lut_buffer(),
            0,
            counts.as_ref(),
            offsets.as_ref(),
            ybuf.as_ref(),
            rows,
            n,
            k,
            n_expert,
            1,
        ),
        (_, Some(dr)) => rec.matmul_mmq_experts(
            DType::Nvfp4,
            "expert_down",
            dqa.as_ref(),
            dda.as_ref(),
            None,
            dr.as_ref(),
            0,
            down_stride,
            counts.as_ref(),
            offsets.as_ref(),
            ybuf.as_ref(),
            rows,
            n,
            k,
            n_expert,
            1,
        ),
        _ => unreachable!(),
    }
    rec.finish().unwrap();

    let mut g_out = vec![0f32; npad * n];
    be.download(gbuf.as_ref(), bytemuck::cast_slice_mut(&mut g_out))
        .unwrap();
    let mut y_out = vec![0f32; npad * k];
    be.download(ybuf.as_ref(), bytemuck::cast_slice_mut(&mut y_out))
        .unwrap();
    // Download the GPU's ACTUAL quantized activations for both stages and dequantize them on the
    // host — the reference then shares the kernels' exact inputs, so the dp4a decomposition (an
    // exact-in-f32 factoring for these codebook formats: e8m0/ue4m3 scale × small-int kv) must
    // match to float-association noise. Referencing a host re-quantization instead (`quant_act`)
    // leaks ±1-code slack (GLSL `round()` half-cases are implementation-defined vs Rust's
    // half-away-from-zero) and forced the mushy 5e-2 tolerance the other mmq tests carry.
    let deq_gpu = |codes: &dyn infr_core::backend::Buffer,
                   scales: &dyn infr_core::backend::Buffer,
                   width: usize|
     -> Vec<f32> {
        let mut cb = vec![0u8; npad * width];
        be.download(codes, &mut cb).unwrap();
        let mut sb = vec![0u8; npad * (width / 32) * 2];
        be.download(scales, &mut sb).unwrap();
        (0..n_pairs * width)
            .map(|i| {
                let (r, c) = (i / width, i % width);
                let si = (r * (width / 32) + c / 32) * 2;
                let d = half::f16::from_le_bytes([sb[si], sb[si + 1]]).to_f32();
                (cb[r * width + c] as i8 as f32) * d
            })
            .collect()
    };
    let xq_gpu = deq_gpu(qa.as_ref(), qda.as_ref(), k);
    let gq_gpu = deq_gpu(dqa.as_ref(), dda.as_ref(), n);

    let mut want_g = vec![0f32; n_pairs * n];
    for e in 0..n_expert {
        let (off, cnt) = (offsets_host[e] as usize, counts_host[e] as usize);
        for r in off..off + cnt {
            let xr = &xq_gpu[r * k..(r + 1) * k];
            for o in 0..n {
                want_g[r * n + o] = gate_host[e][o * k..(o + 1) * k]
                    .iter()
                    .zip(xr)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
    }
    for i in 0..n_pairs * n {
        assert!(
            (g_out[i] - want_g[i]).abs() < 1e-3 + 1e-3 * want_g[i].abs(),
            "{label}: MXFP4 GEMM mismatch at {i}: got {} want {}",
            g_out[i],
            want_g[i]
        );
    }
    for e in 0..n_expert {
        let (off, cnt) = (offsets_host[e] as usize, counts_host[e] as usize);
        for r in off..off + cnt {
            let gr = &gq_gpu[r * n..(r + 1) * n];
            for o in 0..k {
                let want: f32 = down_host[e][o * n..(o + 1) * n]
                    .iter()
                    .zip(gr)
                    .map(|(a, b)| a * b)
                    .sum();
                let got = y_out[r * k + o];
                assert!(
                    (got - want).abs() < 1e-3 + 1e-3 * want.abs(),
                    "{label}: NVFP4 GEMM mismatch row {r} out {o}: got {got} want {want}"
                );
            }
        }
    }
    // Sanity: the host-mirrored activation quant stays a faithful stand-in at coarse tolerance
    // (guards quant_q8 itself against drifting from the documented q = round(a/d) contract).
    for (i, (a, b)) in xq_gpu.iter().zip(xq.iter()).enumerate() {
        assert!(
            (a - b).abs() < 5e-3,
            "{label}: quant_q8 vs host mirror drifted at {i}: {a} vs {b}"
        );
    }
    if let (Some(gp), Some((_, expect_evict))) = (&gate_pager, paged) {
        if expect_evict {
            assert!(
                gp.stats().evictions > 0,
                "{label}: paged config must actually exercise eviction"
            );
        }
    }
    println!("{label}: OK");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmq_fp4_expert_gemm_matches_host() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // NB: no i8_dot capability gate here — production callers check `caps().i8_dot` before
    // choosing these kernels; this test targets dp4a-capable dev boxes.
    // Small-avg ragged config (rows·n_used/n_expert = 1 ≤ 24) → the BM=32 `_xp32`/`_xpg32`
    // builds. The paged run gives the pagers 3 slots for 5 experts (3 routed): warming 0/1/2
    // then routing 2/3/4 evicts LRU experts into reused slots before the dispatch.
    run_config(
        &be,
        None,
        5,
        &[0, 0, 4, 3, 2],
        &[0, 0, 0, 4, 7],
        9,
        "fp4 resident xp32",
    );
    run_config(
        &be,
        Some((3, true)),
        5,
        &[0, 0, 4, 3, 2],
        &[0, 0, 0, 4, 7],
        9,
        "fp4 paged xpg32",
    );
    // Large-avg config (64·1/2 = 32 > 24) → the BM=64 `_xp`/`_xpg` builds. Both experts are
    // routed, so the paged run needs both simultaneously resident (2 slots, no churn — the
    // within-batch safety invariant).
    run_config(&be, None, 2, &[40, 24], &[0, 40], 64, "fp4 resident xp");
    run_config(
        &be,
        Some((2, false)),
        2,
        &[40, 24],
        &[0, 40],
        64,
        "fp4 paged xp",
    );
}
