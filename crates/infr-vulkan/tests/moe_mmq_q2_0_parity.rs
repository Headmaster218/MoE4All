//! GPU↔host parity + determinism for the Q2_0 (Bonsai ternary) batched dp4a mmq expert-GEMM
//! family: `matmul_mmq_experts` (`_xp`/`_xp32`) and `matmul_mmq_experts_paged`
//! (`_xpg`/`_xpg32`), both row-tile variants each — the same two configurations per
//! (resident, paged) pair as `moe_mmq_fp4_parity.rs` (small-avg ragged buckets → BM=32, with
//! pager eviction churn on the paged run; large-avg → BM=64). Q2_0 is symmetric small-int
//! (codes-1 = {-1,0,+1,+2} feed dp4a directly, one f16 d per 64-elem block spanning two 32-elem
//! activation blocks) — no `sact` anywhere.
//!
//! Both chained stages (gate then down on the gate's re-quantized output) run Q2_0, so a missing
//! barrier surfaces as wrong numbers (the pager_gemv_multi_parity lesson). Host reference:
//! `infr_gguf::dequant::dequant_block` weights × the GPU's own downloaded activation codes (the
//! exact-reference trick — the dp4a decomposition d_w·d_act·Σ(q_w·q_act) is an exact-in-f32
//! factoring for this format). The gate stage is additionally dispatched 3× in the same
//! submission with bitwise-identical outputs required (the mmq barrier-race lesson: goldens
//! can't catch intra-dispatch races).
//!
//! Synthetic banks: APERIODIC pseudo-random block bytes (full 2-bit code range incl. the +2
//! code) with the leading f16 d patched small — a valid encoding decodable by the production
//! host dequant.
//!
//! Run: `cargo test -p infr-vulkan --test moe_mmq_q2_0_parity -- --ignored --nocapture`
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

/// Synthetic VALID Q2_0 bank: random 2-bit codes (all of {0,1,2,3}), leading f16 d patched small.
fn synth_q2_0(n_elems: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n_elems / 64 * 18);
    for _ in 0..n_elems / 64 {
        let d = half::f16::from_f32(0.004 + (rng.byte() as f32) * 1e-5).to_le_bytes();
        out.extend_from_slice(&d);
        out.extend((0..16).map(|_| rng.byte()));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_config(
    be: &VulkanBackend,
    // `Some((slots, expect_evict))` = paged through a GpuPager; `None` = resident stacked banks.
    paged: Option<(usize, bool)>,
    n_expert: usize,
    counts_host: &[u32],
    offsets_host: &[u32],
    rows: usize,
    label: &str,
) {
    // gate: Q2_0 [n, k] per expert; down: Q2_0 [k, n] per expert (down's GEMM k = gate's n).
    let (k, n) = (256usize, 64usize);
    let gate_stride = k * n;
    let down_stride = n * k;
    let gate_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_q2_0(gate_stride, 0xb0b0 + e as u64))
        .collect();
    let down_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_q2_0(down_stride, 0x50da + e as u64))
        .collect();
    let gate_host: Vec<Vec<f32>> = gate_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Q2_0, b).unwrap())
        .collect();
    let down_host: Vec<Vec<f32>> = down_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Q2_0, b).unwrap())
        .collect();

    let n_pairs: usize = counts_host.iter().sum::<u32>() as usize;
    let npad = n_pairs.div_ceil(64) * 64 + 64; // GEMM As-stage overread padding
    let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.15;
    let x: Vec<f32> = (0..n_pairs * k).map(|i| f(i, 0.11) + 0.02).collect();

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
    // 3 gate outputs: same dispatch three times, bitwise-identical required.
    let gbufs: Vec<_> = (0..3)
        .map(|_| be.alloc(npad * n * 4, BufferUsage::Activations).unwrap())
        .collect();
    let dqa = be.alloc(npad * n, BufferUsage::Activations).unwrap();
    let dda = be
        .alloc(npad * (n / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let dsa = be
        .alloc(npad * (n / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let ybuf = be.alloc(npad * k * 4, BufferUsage::Activations).unwrap();

    let mut gate_pager = None;
    let mut down_pager = None;
    let mut gate_res = None;
    let mut down_res = None;
    if let Some((slots, _)) = paged {
        let gate_sb = gate_stride / 64 * 18;
        let down_sb = down_stride / 64 * 18;
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
    for gbuf in &gbufs {
        match (&gate_pager, &gate_res) {
            (Some(gp), _) => rec.matmul_mmq_experts_paged(
                DType::Q2_0,
                "expert_gateup",
                qa.as_ref(),
                qda.as_ref(),
                None, // Q2_0 is symmetric — no `sact`
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
                DType::Q2_0,
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
    }
    rec.quant_q8(
        gbufs[0].as_ref(),
        dqa.as_ref(),
        dda.as_ref(),
        dsa.as_ref(),
        n_pairs,
        n,
    );
    match (&down_pager, &down_res) {
        (Some(dp), _) => rec.matmul_mmq_experts_paged(
            DType::Q2_0,
            "expert_down",
            dqa.as_ref(),
            dda.as_ref(),
            None,
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
            DType::Q2_0,
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

    // Same-dispatch determinism: the three gate outputs must be BITWISE identical.
    let dl = |b: &dyn infr_core::backend::Buffer, len: usize| -> Vec<f32> {
        let mut v = vec![0f32; len];
        be.download(b, bytemuck::cast_slice_mut(&mut v)).unwrap();
        v
    };
    let g_out = dl(gbufs[0].as_ref(), npad * n);
    for (rep, gb) in gbufs.iter().enumerate().skip(1) {
        let other = dl(gb.as_ref(), npad * n);
        for i in 0..n_pairs * n {
            assert!(
                g_out[i].to_bits() == other[i].to_bits(),
                "{label}: gate GEMM nondeterministic at {i} (rep {rep}): {} vs {}",
                g_out[i],
                other[i]
            );
        }
    }
    let y_out = dl(ybuf.as_ref(), npad * k);

    // Download the GPU's ACTUAL quantized activations for both stages and dequantize on the host
    // — the reference shares the kernels' exact inputs (see moe_mmq_fp4_parity.rs's doc).
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
            "{label}: Q2_0 gate GEMM mismatch at {i}: got {} want {}",
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
                    "{label}: Q2_0 down GEMM mismatch row {r} out {o}: got {got} want {want}"
                );
            }
        }
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
fn mmq_q2_0_expert_gemm_matches_host() {
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
        "q2_0 resident xp32",
    );
    run_config(
        &be,
        Some((3, true)),
        5,
        &[0, 0, 4, 3, 2],
        &[0, 0, 0, 4, 7],
        9,
        "q2_0 paged xpg32",
    );
    // Large-avg config (64·1/2 = 32 > 24) → the BM=64 `_xp`/`_xpg` builds. Both experts are
    // routed, so the paged run needs both simultaneously resident (2 slots, no churn — the
    // within-batch safety invariant).
    run_config(&be, None, 2, &[40, 24], &[0, 40], 64, "q2_0 resident xp");
    run_config(
        &be,
        Some((2, false)),
        2,
        &[40, 24],
        &[0, 40],
        64,
        "q2_0 paged xp",
    );
}
