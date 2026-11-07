//! GPU↔host parity + determinism for the GRID-codebook batched expert mmq GEMMs (IQ2_S gate/up +
//! IQ3_S down — the Qwen3.6-35B-A3B-UD-IQ3_S expert pair), all four kernel builds each:
//! `_xp` (BM=64), `_xp32` (BM=32) resident, and `_xpg`/`_xpg32` paged under eviction churn.
//!
//! Banks are APERIODIC pseudo-random block bytes with the f16 scale field patched small (the
//! Q6_K lesson: a periodic synthetic bank once hid a sub-block addressing bug because every
//! sub-block decoded identically). That is a VALID encoding for these formats — grid-index and
//! sign bits cover their full table ranges — and the host reference is
//! `infr_gguf::dequant::dequant_block` (the production host dequant), same discipline as
//! `moe_id_gemv_new_formats_parity.rs`.
//!
//! Determinism: the same GEMM dispatched 3x over identical inputs in one submit must produce
//! BITWISE-identical outputs (the barrier-race lesson: the k-loop's restage barrier sits after
//! ALL shared reads of an iteration — a misplaced one shows up exactly here).
//!
//! Run: `cargo test -p infr-vulkan --test grid_mmq_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
use infr_vulkan::pager::GpuPager;
use infr_vulkan::VulkanBackend;

/// Deterministic byte stream (SplitMix64-ish) so failures reproduce.
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

fn block_geom(dt: DType) -> (usize, usize) {
    match dt {
        DType::Iq2S => (256, 82),
        DType::Iq2Xs => (256, 74),
        DType::Iq3S => (256, 110),
        DType::Iq3Xxs => (256, 98),
        other => panic!("no geometry for {other:?}"),
    }
}

/// One synthetic VALID block: random payload bytes, leading f16 `d` patched small.
fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = block_geom(dt);
    assert_eq!(n_elems % epb, 0, "bank must be block-aligned");
    let mut rng = Rng(seed);
    let d16 = half::f16::from_f32(0.02).to_le_bytes();
    let mut out = Vec::with_capacity(n_elems / epb * bpb);
    for _ in 0..n_elems / epb {
        let mut b: Vec<u8> = (0..bpb).map(|_| rng.byte()).collect();
        b[0..2].copy_from_slice(&d16);
        out.extend_from_slice(&b);
    }
    out
}

/// Host mirror of `quant_q8` (per-32-block symmetric int8): the GEMM's activation operand.
/// EXACT shader order: quantize against the f32 scale (`d = amax/127`), but dequantize against
/// the f16-ROUNDED stored scale — the mmq kernel multiplies by the f16 `dact` it reads back.
fn quant_act(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if amax > 0.0 { 1.0 / d } else { 0.0 };
        let d16 = half::f16::from_f32(d).to_f32();
        for &v in blk {
            out.push((v * id).round().clamp(-127.0, 127.0) * d16);
        }
    }
    out
}

fn assert_close(kind: &str, got: &[f32], want: &[f32]) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 5e-2 + 2e-3 * w.abs(),
            "{kind} mismatch at {i}: got {g} want {w}"
        );
    }
}

/// Host reference for one bucketed expert GEMM over quantized activations.
fn host_expert_gemm(
    host_w: &[Vec<f32>],
    xq: &[f32],
    counts: &[u32],
    offsets: &[u32],
    k: usize,
    n: usize,
) -> Vec<f32> {
    let rows = counts.iter().sum::<u32>() as usize;
    let mut want = vec![0f32; rows * n];
    for e in 0..counts.len() {
        let (off, cnt) = (offsets[e] as usize, counts[e] as usize);
        for r in off..off + cnt {
            let xr = &xq[r * k..(r + 1) * k];
            for o in 0..n {
                want[r * n + o] = host_w[e][o * k..(o + 1) * k]
                    .iter()
                    .zip(xr)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
    }
    want
}

/// Resident `_xp`/`_xp32` parity + 3x-dispatch bitwise determinism: IQ2_S "gate" GEMM chained
/// into an IQ3_S "down" GEMM (the dependency chain catches missing barriers, not just wrong
/// math), at BOTH row-tile selections (avg rows/expert 1.8 → BM=32, 40 → BM=64).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn grid_mmq_expert_gemm_matches_host_and_is_deterministic() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (k, n, n_expert) = (256usize, 64usize, 5usize);
    let stride = k * n;

    let gate_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_bank(DType::Iq2S, stride, 0x5eed ^ ((e as u64) << 32)))
        .collect();
    let down_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_bank(DType::Iq3S, stride, 0xfeed ^ ((e as u64) << 32)))
        .collect();
    let gate_host: Vec<Vec<f32>> = gate_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Iq2S, b).unwrap())
        .collect();
    let down_host: Vec<Vec<f32>> = down_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Iq3S, b).unwrap())
        .collect();

    // One stacked resident bank per role.
    let mk_bank = |banks: &[Vec<u8>]| {
        let flat: Vec<u8> = banks.concat();
        let padded = pad_to_u32_align(&flat);
        let b = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
        be.upload(b.as_ref(), &padded).unwrap();
        b
    };
    let gate_w = mk_bank(&gate_banks);
    let down_w = mk_bank(&down_banks);

    // (counts, label): 9 rows → avg 1.8/expert → BM=32 `_xp32`; 200 rows → avg 40 → BM=64 `_xp`.
    for (counts_host, label) in [
        (vec![0u32, 0, 4, 3, 2], "xp32"),
        (vec![50u32, 40, 45, 35, 30], "xp"),
    ] {
        let offsets_host: Vec<u32> = {
            let mut acc = 0;
            counts_host
                .iter()
                .map(|&c| {
                    let o = acc;
                    acc += c;
                    o
                })
                .collect()
        };
        let rows = counts_host.iter().sum::<u32>() as usize;
        let npad = rows.div_ceil(64) * 64 + 64; // GEMM As-stage overread padding

        let x: Vec<f32> = (0..rows * k)
            .map(|i| (((i * 37 + 11) % 61) as f32 - 30.0) * 0.01)
            .collect();
        let xq = quant_act(&x);

        let mk_u32 = |v: &[u32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let counts = mk_u32(&counts_host);
        let offsets = mk_u32(&offsets_host);
        let xbuf = be.alloc(rows * k * 4, BufferUsage::Activations).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let qa = be.alloc(npad * k, BufferUsage::Activations).unwrap();
        let qda = be
            .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
            .unwrap();
        let qsa = be
            .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
            .unwrap();
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
        let ybufs: Vec<_> = (0..3)
            .map(|_| be.alloc(npad * k * 4, BufferUsage::Activations).unwrap())
            .collect();

        let rec = be.recorder().unwrap();
        rec.quant_q8(
            xbuf.as_ref(),
            qa.as_ref(),
            qda.as_ref(),
            qsa.as_ref(),
            rows,
            k,
        );
        // 3 identical IQ2_S dispatches (determinism), then the chained IQ3_S down off dispatch 0.
        for g in &gbufs {
            rec.matmul_mmq_experts(
                DType::Iq2S,
                "expert_gateup",
                qa.as_ref(),
                qda.as_ref(),
                None, // symmetric — no sact
                gate_w.as_ref(),
                0,
                stride,
                counts.as_ref(),
                offsets.as_ref(),
                g.as_ref(),
                rows,
                k,
                n,
                n_expert,
                1,
            );
        }
        rec.quant_q8(
            gbufs[0].as_ref(),
            dqa.as_ref(),
            dda.as_ref(),
            dsa.as_ref(),
            rows,
            n,
        );
        for y in &ybufs {
            rec.matmul_mmq_experts(
                DType::Iq3S,
                "expert_down",
                dqa.as_ref(),
                dda.as_ref(),
                None,
                down_w.as_ref(),
                0,
                stride,
                counts.as_ref(),
                offsets.as_ref(),
                y.as_ref(),
                rows,
                n,
                k,
                n_expert,
                1,
            );
        }
        rec.finish().unwrap();

        let dl = |b: &dyn infr_core::backend::Buffer, len: usize| {
            let mut out = vec![0u8; len * 4];
            be.download(b, &mut out).unwrap();
            bytemuck::cast_slice::<u8, f32>(&out).to_vec()
        };
        let g0 = dl(gbufs[0].as_ref(), npad * n);
        let g1 = dl(gbufs[1].as_ref(), npad * n);
        let g2 = dl(gbufs[2].as_ref(), npad * n);
        assert!(
            g0[..rows * n]
                .iter()
                .zip(&g1[..rows * n])
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && g0[..rows * n]
                    .iter()
                    .zip(&g2[..rows * n])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "IQ2_S {label} GEMM nondeterministic across identical dispatches"
        );
        let y0 = dl(ybufs[0].as_ref(), npad * k);
        let y1 = dl(ybufs[1].as_ref(), npad * k);
        let y2 = dl(ybufs[2].as_ref(), npad * k);
        assert!(
            y0[..rows * k]
                .iter()
                .zip(&y1[..rows * k])
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && y0[..rows * k]
                    .iter()
                    .zip(&y2[..rows * k])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "IQ3_S {label} GEMM nondeterministic across identical dispatches"
        );

        let want_g = host_expert_gemm(&gate_host, &xq, &counts_host, &offsets_host, k, n);
        assert_close(&format!("IQ2_S {label}"), &g0[..rows * n], &want_g);
        // Down reference from the GPU gate output (not `want_g`): the GPU quantizes ITS OWN gate
        // result, and an int8 rounding-boundary flip between two near-equal gate values is a
        // legitimate ±d_act·w difference that would swamp the down GEMM's own error budget.
        let gq = quant_act(&g0[..rows * n]);
        let want_y = host_expert_gemm(&down_host, &gq, &counts_host, &offsets_host, n, k);
        assert_close(&format!("IQ3_S {label}"), &y0[..rows * k], &want_y);
        println!("resident IQ2_S+IQ3_S {label}: parity + 3x determinism OK ({rows} rows)");
    }
}

/// Paged `_xpg`/`_xpg32` parity under eviction churn (LRU slot reuse — the coherent-but-wrong
/// word-base bug class): 5 experts through 3-slot pagers, fill 0/1/2 then route {2,3,4}.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn grid_mmq_paged_expert_gemm_matches_host_under_eviction() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (k, n, n_expert) = (256usize, 64usize, 5usize);
    let stride = k * n;
    let gate_slot_bytes = stride / 256 * 82; // IQ2_S
    let down_slot_bytes = stride / 256 * 110; // IQ3_S

    let gate_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_bank(DType::Iq2S, stride, 0xabcd ^ ((e as u64) << 32)))
        .collect();
    let down_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| synth_bank(DType::Iq3S, stride, 0xdcba ^ ((e as u64) << 32)))
        .collect();
    let gate_host: Vec<Vec<f32>> = gate_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Iq2S, b).unwrap())
        .collect();
    let down_host: Vec<Vec<f32>> = down_banks
        .iter()
        .map(|b| infr_gguf::dequant::dequant_block(DType::Iq3S, b).unwrap())
        .collect();

    let mut gate_pager = GpuPager::new(&be, n_expert, 3, gate_slot_bytes).unwrap();
    let mut down_pager = GpuPager::new(&be, n_expert, 3, down_slot_bytes).unwrap();
    let staging = be
        .alloc_uninit(gate_slot_bytes.max(down_slot_bytes), BufferUsage::Staging)
        .unwrap();
    for pre in [0u32, 1, 2] {
        gate_pager
            .ensure_resident(&be, staging.as_ref(), pre, &gate_banks[pre as usize])
            .unwrap();
        down_pager
            .ensure_resident(&be, staging.as_ref(), pre, &down_banks[pre as usize])
            .unwrap();
    }
    for &eid in &[2u32, 3, 4] {
        gate_pager
            .ensure_resident(&be, staging.as_ref(), eid, &gate_banks[eid as usize])
            .unwrap();
        down_pager
            .ensure_resident(&be, staging.as_ref(), eid, &down_banks[eid as usize])
            .unwrap();
    }
    gate_pager.flush_lut(&be).unwrap();
    down_pager.flush_lut(&be).unwrap();
    assert!(
        gate_pager.stats().evictions > 0,
        "churn must actually evict"
    );

    // 9 rows → avg 1.8/expert → `_xpg32`; 200 rows → avg 40 → `_xpg`.
    for (counts_host, label) in [
        (vec![0u32, 0, 4, 3, 2], "xpg32"),
        (vec![0u32, 0, 70, 65, 65], "xpg"),
    ] {
        let offsets_host: Vec<u32> = {
            let mut acc = 0;
            counts_host
                .iter()
                .map(|&c| {
                    let o = acc;
                    acc += c;
                    o
                })
                .collect()
        };
        let rows = counts_host.iter().sum::<u32>() as usize;
        let npad = rows.div_ceil(64) * 64 + 64;

        let x: Vec<f32> = (0..rows * k)
            .map(|i| (((i * 29 + 7) % 53) as f32 - 26.0) * 0.012)
            .collect();
        let xq = quant_act(&x);

        let mk_u32 = |v: &[u32]| {
            let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
            b
        };
        let counts = mk_u32(&counts_host);
        let offsets = mk_u32(&offsets_host);
        let xbuf = be.alloc(rows * k * 4, BufferUsage::Activations).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
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

        let rec = be.recorder().unwrap();
        rec.quant_q8(
            xbuf.as_ref(),
            qa.as_ref(),
            qda.as_ref(),
            qsa.as_ref(),
            rows,
            k,
        );
        rec.matmul_mmq_experts_paged(
            DType::Iq2S,
            "expert_gateup",
            qa.as_ref(),
            qda.as_ref(),
            None,
            gate_pager.arena_addr(),
            gate_pager.slot_bytes() as u32,
            gate_pager.lut_buffer(),
            0,
            counts.as_ref(),
            offsets.as_ref(),
            gbuf.as_ref(),
            rows,
            k,
            n,
            n_expert,
            1,
        );
        rec.quant_q8(
            gbuf.as_ref(),
            dqa.as_ref(),
            dda.as_ref(),
            dsa.as_ref(),
            rows,
            n,
        );
        rec.matmul_mmq_experts_paged(
            DType::Iq3S,
            "expert_down",
            dqa.as_ref(),
            dda.as_ref(),
            None,
            down_pager.arena_addr(),
            down_pager.slot_bytes() as u32,
            down_pager.lut_buffer(),
            0,
            counts.as_ref(),
            offsets.as_ref(),
            ybuf.as_ref(),
            rows,
            n,
            k,
            n_expert,
            1,
        );
        rec.finish().unwrap();

        let mut g_out = vec![0f32; npad * n];
        be.download(gbuf.as_ref(), bytemuck::cast_slice_mut(&mut g_out))
            .unwrap();
        let mut y_out = vec![0f32; npad * k];
        be.download(ybuf.as_ref(), bytemuck::cast_slice_mut(&mut y_out))
            .unwrap();

        let want_g = host_expert_gemm(&gate_host, &xq, &counts_host, &offsets_host, k, n);
        assert_close(&format!("paged IQ2_S {label}"), &g_out[..rows * n], &want_g);
        // Down reference from the GPU gate output — see the resident test's comment.
        let gq = quant_act(&g_out[..rows * n]);
        let want_y = host_expert_gemm(&down_host, &gq, &counts_host, &offsets_host, n, k);
        assert_close(&format!("paged IQ3_S {label}"), &y_out[..rows * k], &want_y);
        println!("paged IQ2_S+IQ3_S {label}: parity under eviction churn OK ({rows} rows)");
    }
}

/// Qwen3.8 Q2_K_XL uses IQ2_XS for almost every gate/up bank and IQ3_XXS for one pair. Exercise
/// both resident and paged kernels at the BM=32 and BM=64 routing thresholds against the GGUF
/// decoder. The paged copy is deliberately churned through fewer slots than experts.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn qwen38_xs_grid_mmq_all_variants_match_host() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (k, n, n_expert) = (256usize, 64usize, 5usize);
    let stride = k * n;

    for dt in [DType::Iq2Xs, DType::Iq3Xxs] {
        let banks: Vec<Vec<u8>> = (0..n_expert)
            .map(|e| synth_bank(dt, stride, 0x38f1a5 ^ (dt as u64) ^ ((e as u64) << 32)))
            .collect();
        let host: Vec<Vec<f32>> = banks
            .iter()
            .map(|b| infr_gguf::dequant::dequant_block(dt, b).unwrap())
            .collect();
        let joined = banks.concat();
        let flat = pad_to_u32_align(&joined);
        let resident = be.alloc(flat.len(), BufferUsage::Weights).unwrap();
        be.upload(resident.as_ref(), &flat).unwrap();

        let slot_bytes = banks[0].len();
        let mut pager = GpuPager::new(&be, n_expert, 3, slot_bytes).unwrap();
        let staging = be.alloc_uninit(slot_bytes, BufferUsage::Staging).unwrap();
        for eid in [0u32, 1, 2, 3, 4] {
            pager
                .ensure_resident(&be, staging.as_ref(), eid, &banks[eid as usize])
                .unwrap();
        }
        pager.flush_lut(&be).unwrap();
        assert!(pager.stats().evictions > 0, "test must reuse pager slots");

        for (counts_host, label) in [
            (vec![0u32, 0, 4, 3, 2], "BM32"),
            (vec![0u32, 0, 70, 65, 65], "BM64"),
        ] {
            let offsets_host: Vec<u32> = {
                let mut acc = 0;
                counts_host
                    .iter()
                    .map(|&count| {
                        let offset = acc;
                        acc += count;
                        offset
                    })
                    .collect()
            };
            let rows = counts_host.iter().sum::<u32>() as usize;
            let npad = rows.div_ceil(64) * 64 + 64;
            let x: Vec<f32> = (0..rows * k)
                .map(|i| (((i * 43 + 19) % 79) as f32 - 39.0) * 0.009)
                .collect();
            let xq = quant_act(&x);

            let mk_u32 = |v: &[u32]| {
                let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
                be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
                b
            };
            let counts = mk_u32(&counts_host);
            let offsets = mk_u32(&offsets_host);
            let xbuf = be.alloc(rows * k * 4, BufferUsage::Activations).unwrap();
            be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
            let qa = be.alloc(npad * k, BufferUsage::Activations).unwrap();
            let qda = be
                .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
                .unwrap();
            let qsa = be
                .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
                .unwrap();
            let y_res = be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();
            let y_paged = be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();

            let rec = be.recorder().unwrap();
            rec.quant_q8(
                xbuf.as_ref(),
                qa.as_ref(),
                qda.as_ref(),
                qsa.as_ref(),
                rows,
                k,
            );
            rec.matmul_mmq_experts(
                dt,
                "expert_gateup",
                qa.as_ref(),
                qda.as_ref(),
                None,
                resident.as_ref(),
                0,
                stride,
                counts.as_ref(),
                offsets.as_ref(),
                y_res.as_ref(),
                rows,
                k,
                n,
                n_expert,
                1,
            );
            rec.matmul_mmq_experts_paged(
                dt,
                "expert_gateup",
                qa.as_ref(),
                qda.as_ref(),
                None,
                pager.arena_addr(),
                pager.slot_bytes() as u32,
                pager.lut_buffer(),
                0,
                counts.as_ref(),
                offsets.as_ref(),
                y_paged.as_ref(),
                rows,
                k,
                n,
                n_expert,
                1,
            );
            rec.finish().unwrap();

            let download = |buf: &dyn infr_core::backend::Buffer| {
                let mut bytes = vec![0u8; npad * n * 4];
                be.download(buf, &mut bytes).unwrap();
                bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
            };
            let got_res = download(y_res.as_ref());
            let got_paged = download(y_paged.as_ref());
            let want = host_expert_gemm(&host, &xq, &counts_host, &offsets_host, k, n);
            assert_close(
                &format!("resident {dt:?} {label}"),
                &got_res[..rows * n],
                &want,
            );
            assert_close(
                &format!("paged {dt:?} {label}"),
                &got_paged[..rows * n],
                &want,
            );
            println!("{dt:?} {label}: resident+paged parity OK ({rows} rows)");
        }
    }
}
