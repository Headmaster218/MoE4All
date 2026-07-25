//! The multi-warp dp4a decode GEMV (`native_mmv_mw.comp`, wave32-native route) must match the
//! validated 64-thread `native_mmv` dp4a kernel at the real projection shapes — both quantize the
//! activation identically (quant_q8) and use the identical `dpsub` math, differing ONLY in the
//! reduction (warp-per-row subgroupAdd vs shared-tree) and lane→sub-block mapping, so the gap is
//! pure float-reassociation. Proves mmv_mw inherits native_mmv's correctness. Run:
//!   cargo test -p infr-vulkan --test mmv_mw_parity -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

/// Bytes per block, from the shared decode spec (`infr_core::decode_spec` via
/// `infr_gguf::block_layout`) — not a hand-kept local table.
fn blk_bytes(dt: DType) -> usize {
    infr_gguf::block_layout(dt).1
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mw_matches_native_mmv() {
    // No wave32-native gate: the kernel pins requiredSubgroupSize = sg_pref (32 outside Intel)
    // and every device the backend accepts can pin 32 — so this validates on AMD/RADV too (the
    // route is INFR_MMV_MW=1 opt-in there, default-on on Intel; see adapter::mmv_mw_choice).
    let be = VulkanBackend::new().unwrap();
    // (in_f, out_f, dtype) — the decode projection shapes mmv_mw serves.
    let shapes = [
        (2048usize, 2048usize, DType::Q4K), // q / o
        (2048, 6144, DType::Q4K),           // gate+up (fused)
        (6144, 2048, DType::Q4K),           // down
        (2048, 2048, DType::Q6K),
        (6144, 2048, DType::Q6K),   // down-q6
        (2048, 151936, DType::Q6K), // lm_head (odd out_f: 151936 % 8 != 0)
    ];
    let xmax = 6144;
    let xs: Vec<f32> = (0..xmax)
        .map(|i| ((i % 61) as f32 - 30.0) * 0.013)
        .collect();
    let x = be.alloc(xmax * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let read = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<f32> {
        let mut out = vec![0u8; n * 4];
        be.download(b, &mut out).unwrap();
        bytemuck::cast_slice::<u8, f32>(&out).to_vec()
    };

    let mut worst = 0f32;
    for (in_f, out_f, dt) in shapes {
        let wbytes = in_f * out_f / 256 * blk_bytes(dt);
        let mut src: Vec<u8> = (0..wbytes)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        for blk in src.chunks_exact_mut(blk_bytes(dt)) {
            match dt {
                DType::Q4K => {
                    blk[0..2].copy_from_slice(&[0x00, 0x1C]); // d = 1.0 (f16)
                    blk[2..4].copy_from_slice(&[0x00, 0x18]); // dmin
                }
                DType::Q6K => blk[208..210].copy_from_slice(&[0x00, 0x1C]),
                _ => unreachable!(),
            }
        }
        let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
        be.upload(w.as_ref(), &src).unwrap();
        let nblk = in_f / 32;
        let qa = be.alloc(in_f, BufferUsage::Activations).unwrap();
        let dact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let y_ref = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let y_mw = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.quant_q8(
            x.as_ref(),
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            1,
            in_f,
        );
        rec.linear_mmv(
            dt,
            w.as_ref(),
            0,
            qa.as_ref(),
            dact.as_ref(),
            sact.as_ref(),
            y_ref.as_ref(),
            in_f,
            out_f,
        );
        rec.finish().unwrap();
        let r = read(y_ref.as_ref(), out_f);
        for warps in [4u32, 8u32] {
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mw(
                dt,
                warps,
                w.as_ref(),
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y_mw.as_ref(),
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let g = read(y_mw.as_ref(), out_f);
            // Mixed abs/rel metric: the pure-relative form blew up on near-zero lm_head outputs
            // (ref 2.92e-4 vs mw 2.96e-4 — a 4e-6 reassociation residue reading as 1.3e-2 "rel")
            // once this test started actually running on RADV (it used to skip on subgroup_max
            // != 32). An absolute floor of 1e-4 covers the cancellation band; genuinely wrong
            // math (bad scale/nibble map) is orders of magnitude above both bounds.
            let mut mr = 0f32;
            let (mut wa, mut wb) = (0f32, 0f32);
            for (a, b) in r.iter().zip(&g) {
                let e = ((a - b).abs() - 1e-4).max(0.0) / (a.abs().max(b.abs()) + 1e-6);
                if e > mr {
                    mr = e;
                    wa = *a;
                    wb = *b;
                }
            }
            worst = worst.max(mr);
            println!(
                "  {dt:?} in{in_f} out{out_f} W{warps}: max rel err {mr:.2e} (worst ref {wa:.6} vs mw {wb:.6})"
            );
            assert!(
                mr < 5e-3,
                "{dt:?} {in_f}x{out_f} W{warps}: mmv_mw diverges from native_mmv (rel {mr:.2e})"
            );
        }
    }
    println!("mmv_mw == native_mmv within reassociation tolerance (worst {worst:.2e})");
}

// ── Q2_K / Q3_K host-reference parity + determinism (Intel R3 dtypes) ────────────────────────────
// Q2_K/Q3_K have NO native_mmv (64-thread tree) twin to diff against, so the reference is a host
// emulation of the kernel's exact integer math: the GPU's own quant_q8 outputs (qa/dact) are read
// back and the per-sub-block dp4a + scale formula is replayed in Rust — the only difference left is
// float summation order across sub-blocks (lane striding + subgroupAdd), so the tolerance is pure
// reassociation. Weight banks are APERIODIC random bytes (the Q6_K mmq lesson: a 64-periodic input
// hid a nibble-map transposition) with sane f16 block scales patched in.

fn f16b(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}

fn f16v(lo: u8, hi: u8) -> f32 {
    half::f16::from_le_bytes([lo, hi]).to_f32()
}

/// Host replay of native_mmv_mw.comp's FMT_Q2K dpsub (exact integer dot per 16-elem scale group).
fn dpsub_q2k(w: &[u8], gelem: usize, qa: &[i8], xbase: usize, da: f32) -> f32 {
    let bd = (gelem / 256) * 84;
    let p0 = gelem % 256;
    let d = f16v(w[bd + 80], w[bd + 81]);
    let dmin = f16v(w[bd + 82], w[bd + 83]);
    let sb0 = w[bd + p0 / 16] as u32;
    let sb1 = w[bd + p0 / 16 + 1] as u32;
    let (dl0, ml0) = (d * (sb0 & 0xF) as f32, dmin * (sb0 >> 4) as f32);
    let (dl1, ml1) = (d * (sb1 & 0xF) as f32, dmin * (sb1 >> 4) as f32);
    let shift = 2 * ((p0 % 128) / 32);
    let qbase = bd + 16 + 32 * (p0 / 128);
    let (mut dp0, mut dp1, mut s0, mut s1) = (0i32, 0i32, 0i32, 0i32);
    for e in 0..32 {
        let q = ((w[qbase + e] as u32 >> shift) & 3) as i32;
        let a = qa[xbase + e] as i32;
        if e < 16 {
            dp0 += q * a;
            s0 += a;
        } else {
            dp1 += q * a;
            s1 += a;
        }
    }
    da * (dl0 * dp0 as f32 + dl1 * dp1 as f32) - da * (ml0 * s0 as f32 + ml1 * s1 as f32)
}

/// Host replay of native_mmv_mw.comp's FMT_Q3K dpsub.
fn dpsub_q3k(w: &[u8], gelem: usize, qa: &[i8], xbase: usize, da: f32) -> f32 {
    let bd = (gelem / 256) * 110;
    let p0 = gelem % 256;
    let d_all = f16v(w[bd + 108], w[bd + 109]);
    let rd32 = |o: usize| -> u32 { u32::from_le_bytes(w[bd + o..bd + o + 4].try_into().unwrap()) };
    let (a0, a1, a2) = (rd32(96), rd32(100), rd32(104));
    let (k1, k2) = (0x03030303u32, 0x0f0f0f0fu32);
    let aux = [
        (a0 & k2) | ((a2 & k1) << 4),
        (a1 & k2) | (((a2 >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4),
    ];
    let scb = |is: usize| -> i32 {
        let b = ((aux[is >> 2] >> ((is & 3) * 8)) & 0xFF) as i32;
        (if b >= 128 { b - 256 } else { b }) - 32
    };
    let is0 = p0 / 16;
    let (sc0, sc1) = (scb(is0) as f32, scb(is0 + 1) as f32);
    let n = p0 / 128;
    let jj = (p0 % 128) / 32;
    let shift = 2 * jj;
    let jg = 4 * n + jj;
    let qb = bd + 32 + 32 * n;
    let (mut dp0, mut dp1) = (0i32, 0i32);
    for e in 0..32 {
        let low2 = (w[qb + e] as u32 >> shift) & 3;
        let hbit = (w[bd + e] as u32 >> jg) & 1;
        let q = (low2 | (hbit << 2)) as i32 - 4;
        let a = qa[xbase + e] as i32;
        if e < 16 {
            dp0 += q * a;
        } else {
            dp1 += q * a;
        }
    }
    d_all * da * (sc0 * dp0 as f32 + sc1 * dp1 as f32)
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mw_q2k_q3k_match_host_reference() {
    let be = VulkanBackend::new().unwrap();
    // (in_f, out_f, dtype) — projection-band + lm_head-ish shapes (odd out_f exercises the tail).
    let shapes = [
        (2048usize, 2048usize, DType::Q2K),
        (6144, 2048, DType::Q2K),
        (2048, 6145, DType::Q2K), // odd out_f: padding-workgroup guard
        (2048, 2048, DType::Q3K),
        (6144, 2048, DType::Q3K),
        (2048, 6145, DType::Q3K),
    ];
    let blk = |dt: DType| match dt {
        DType::Q2K => 84usize,
        DType::Q3K => 110,
        _ => unreachable!(),
    };
    let xmax = 6144;
    let xs: Vec<f32> = (0..xmax)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.011 + ((i % 7) as f32) * 0.003)
        .collect();
    let x = be.alloc(xmax * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let read_bytes = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<u8> {
        let mut out = vec![0u8; n];
        be.download(b, &mut out).unwrap();
        out
    };

    let mut worst = 0f32;
    for (in_f, out_f, dt) in shapes {
        let nblocks = in_f * out_f / 256;
        let wbytes = nblocks * blk(dt);
        // Aperiodic random bank (multiplicative hash of the byte index), then patch each block's
        // f16 scales to sane, block-varying values.
        let mut src: Vec<u8> = (0..wbytes)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        for (bi, b) in src.chunks_exact_mut(blk(dt)).enumerate() {
            let d = 0.25 + (bi % 11) as f32 * 0.05;
            match dt {
                DType::Q2K => {
                    b[80..82].copy_from_slice(&f16b(d));
                    b[82..84].copy_from_slice(&f16b(0.1 + (bi % 5) as f32 * 0.02));
                }
                DType::Q3K => b[108..110].copy_from_slice(&f16b(d)),
                _ => unreachable!(),
            }
        }
        let w = be
            .alloc(wbytes.next_multiple_of(4), BufferUsage::Weights)
            .unwrap();
        be.upload(w.as_ref(), &src).unwrap();
        let nblk = in_f / 32;
        let qa = be.alloc(in_f, BufferUsage::Activations).unwrap();
        let dact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

        for warps in [4u32, 8u32] {
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mw(
                dt,
                warps,
                w.as_ref(),
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y.as_ref(),
                in_f,
                out_f,
            );
            rec.finish().unwrap();

            // Host reference from the GPU's own quantized activation (exact same integers).
            let qa_h: Vec<i8> = read_bytes(qa.as_ref(), in_f)
                .iter()
                .map(|&b| b as i8)
                .collect();
            let dact_h: Vec<f32> = read_bytes(dact.as_ref(), nblk * 2)
                .chunks_exact(2)
                .map(|c| f16v(c[0], c[1]))
                .collect();
            let got: Vec<f32> =
                bytemuck::cast_slice::<u8, f32>(&read_bytes(y.as_ref(), out_f * 4)).to_vec();
            let mut mr = 0f32;
            for (o, &g) in got.iter().enumerate().take(out_f) {
                let mut acc = 0f32;
                for (s, &da) in dact_h.iter().enumerate().take(nblk) {
                    let gelem = o * in_f + s * 32;
                    acc += match dt {
                        DType::Q2K => dpsub_q2k(&src, gelem, &qa_h, s * 32, da),
                        DType::Q3K => dpsub_q3k(&src, gelem, &qa_h, s * 32, da),
                        _ => unreachable!(),
                    };
                }
                mr = mr.max((g - acc).abs() / (g.abs().max(acc.abs()) + 1e-6));
            }
            worst = worst.max(mr);
            println!("  {dt:?} in{in_f} out{out_f} W{warps}: max rel err vs host {mr:.2e}");
            assert!(
                mr < 2e-3,
                "{dt:?} {in_f}x{out_f} W{warps}: mmv_mw diverges from host reference ({mr:.2e})"
            );
        }
    }
    println!("mmv_mw Q2_K/Q3_K == host int-dot reference (worst {worst:.2e})");
}

// ── Q5_K host-reference parity (NEW int8 arm — previously no int8 kernel in either stream) ──────
// Q5_K has no `native_mmv` (64-thread tree) twin either, so the reference is the same host-emulation
// technique as Q2_K/Q3_K above: replay `native_mmv_mw.comp`'s FMT_Q5K `dpsub` byte-at-a-time in Rust
// against the GPU's own quant_q8 outputs (qa/dact/sact read back), independent of the shader's
// word-parallel u32 unpack (a from-scratch re-derivation of the block_q5_K bit layout — 4-bit `qs`
// nibble + a 5th bit from a `qh` plane shared across all 8 sub-blocks, same 6-bit packed scale/min
// as Q4_K — not a copy of the GLSL). Unlike Q2_K/Q3_K, Q5_K's `dpsub` DOES need the min term, so
// (unlike those two) this reads `sact` back from the GPU rather than recomputing per-block sums —
// same shape as Q4_K's `f0*da*dp - f1*sa` accumulation.

/// Host replay of `native_mmv_mw.comp`'s Q4_K/Q5_K-shared 6-bit scale/min unpack (`k4` in GLSL).
fn k4_host(i: usize, sb: usize, w: &[u8]) -> (u32, u32) {
    if i < 4 {
        (w[sb + i] as u32 & 63, w[sb + i + 4] as u32 & 63)
    } else {
        let sc = (w[sb + i + 4] as u32 & 0xF) | ((w[sb + i - 4] as u32 >> 6) << 4);
        let mn = (w[sb + i + 4] as u32 >> 4) | ((w[sb + i] as u32 >> 6) << 4);
        (sc, mn)
    }
}

/// Host replay of `native_mmv_mw.comp`'s FMT_Q5K dpsub. Block layout: [f16 d][f16 dmin][u8
/// scales[12]][u8 qh[32]][u8 qs[128]] = 176 bytes, 256 elements — `qh` is one shared 32-byte plane
/// for the whole super-block, bit `sub` (0..7) selects the 5th bit for sub-block `sub`.
fn dpsub_q5k(w: &[u8], gelem: usize, qa: &[i8], xbase: usize, da: f32, sa: f32) -> f32 {
    let bd = (gelem / 256) * 176;
    let sub = (gelem / 32) & 7;
    let d = f16v(w[bd], w[bd + 1]);
    let dmin = f16v(w[bd + 2], w[bd + 3]);
    let (sc, mn) = k4_host(sub, bd + 4, w);
    let dl = d * sc as f32;
    let mm = dmin * mn as f32;
    let qlbase = bd + 48 + (sub / 2) * 32;
    let qhbase = bd + 16;
    let mut dp = 0i32;
    for e in 0..32 {
        let ql_byte = w[qlbase + e] as u32;
        let qh_byte = w[qhbase + e] as u32;
        let nib = if sub.is_multiple_of(2) {
            ql_byte & 0xF
        } else {
            ql_byte >> 4
        };
        let bit = (qh_byte >> sub) & 1;
        let q = (nib | (bit << 4)) as i32;
        let a = qa[xbase + e] as i32;
        dp += q * a;
    }
    dl * da * dp as f32 - mm * sa
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mw_q5k_match_host_reference() {
    let be = VulkanBackend::new().unwrap();
    // Same shapes as the Q2_K/Q3_K reference test (projection-band + lm_head-ish, odd out_f
    // exercises the tail-workgroup guard).
    let shapes = [(2048usize, 2048usize), (6144, 2048), (2048, 6145)];
    let blk = 176usize;
    let xmax = 6144;
    let xs: Vec<f32> = (0..xmax)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.011 + ((i % 7) as f32) * 0.003)
        .collect();
    let x = be.alloc(xmax * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    let read_bytes = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<u8> {
        let mut out = vec![0u8; n];
        be.download(b, &mut out).unwrap();
        out
    };

    let mut worst = 0f32;
    for (in_f, out_f) in shapes {
        let nblocks = in_f * out_f / 256;
        let wbytes = nblocks * blk;
        // Aperiodic random bank (multiplicative hash), then patch each block's f16 d/dmin to sane,
        // block-varying values — same aperiodicity discipline as the Q2_K/Q3_K test (the Q6_K mmq
        // lesson: a periodic input can hide a nibble-map transposition bug).
        let mut src: Vec<u8> = (0..wbytes)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        for (bi, b) in src.chunks_exact_mut(blk).enumerate() {
            let d = 0.2 + (bi % 11) as f32 * 0.04;
            let dmin = 0.05 + (bi % 5) as f32 * 0.015;
            b[0..2].copy_from_slice(&f16b(d));
            b[2..4].copy_from_slice(&f16b(dmin));
        }
        let w = be
            .alloc(wbytes.next_multiple_of(4), BufferUsage::Weights)
            .unwrap();
        be.upload(w.as_ref(), &src).unwrap();
        let nblk = in_f / 32;
        let qa = be.alloc(in_f, BufferUsage::Activations).unwrap();
        let dact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

        for warps in [4u32, 8u32] {
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mw(
                DType::Q5K,
                warps,
                w.as_ref(),
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y.as_ref(),
                in_f,
                out_f,
            );
            rec.finish().unwrap();

            let qa_h: Vec<i8> = read_bytes(qa.as_ref(), in_f)
                .iter()
                .map(|&b| b as i8)
                .collect();
            let dact_h: Vec<f32> = read_bytes(dact.as_ref(), nblk * 2)
                .chunks_exact(2)
                .map(|c| f16v(c[0], c[1]))
                .collect();
            let sact_h: Vec<f32> = read_bytes(sact.as_ref(), nblk * 2)
                .chunks_exact(2)
                .map(|c| f16v(c[0], c[1]))
                .collect();
            let got: Vec<f32> =
                bytemuck::cast_slice::<u8, f32>(&read_bytes(y.as_ref(), out_f * 4)).to_vec();
            // Accumulate the per-superblock host reference in f64: Q5_K's 6-bit scale (up to 63)
            // and 5-bit quant (up to 31) make individual dp4a terms an order of magnitude larger
            // than Q2_K/Q3_K's (4-bit scale / 2-3-bit quant), so this adversarial random-byte
            // weight bank produces heavy cross-term cancellation at large in_f (192 superblocks) —
            // an f32 flat-order host sum picked up ~4e-3 rel error purely from summation-order
            // reassociation (confirmed against the GPU's own f32 result, which sits CLOSER to the
            // f64 value than the naive-order f32 host sum does — i.e. the kernel is fine, the f32
            // host replay just isn't a tight enough oracle at this magnitude). f64 accumulation
            // removes that ambiguity.
            //
            // Mixed abs/rel metric, same shape as `mmv_mw_matches_native_mmv`'s (which hit the
            // identical class on lm_head's near-zero outputs): the in2048/out6145 shape's o=3972
            // is a genuine cancellation zero (g=0.006592 vs host 0.006805, |diff|=2.1e-4 — the
            // *neighboring* outputs o=3970..3974, magnitude 3.8e3..1.4e4, agree to 6 significant
            // figures), which a pure-relative metric reads as a false 3.1e-2 "error". A 5e-4
            // absolute floor covers that residue; genuinely wrong math would be orders of
            // magnitude above it.
            let mut mr = 0f32;
            for (o, &g) in got.iter().enumerate().take(out_f) {
                let mut acc = 0f64;
                for (s, (&da, &sa)) in dact_h.iter().zip(&sact_h).enumerate().take(nblk) {
                    let gelem = o * in_f + s * 32;
                    acc += dpsub_q5k(&src, gelem, &qa_h, s * 32, da, sa) as f64;
                }
                let acc = acc as f32;
                let e = ((g - acc).abs() - 5e-4).max(0.0) / (g.abs().max(acc.abs()) + 1e-6);
                if e > mr {
                    mr = e;
                }
            }
            worst = worst.max(mr);
            println!("  Q5K in{in_f} out{out_f} W{warps}: max rel err vs host {mr:.2e}");
            assert!(
                mr < 5e-3,
                "Q5K {in_f}x{out_f} W{warps}: mmv_mw diverges from host reference ({mr:.2e})"
            );
        }
    }
    println!("mmv_mw Q5_K == host int-dot reference (worst {worst:.2e})");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mw_q2k_q3k_same_dispatch_deterministic() {
    let be = VulkanBackend::new().unwrap();
    let (in_f, out_f) = (2048usize, 2048usize);
    let xs: Vec<f32> = (0..in_f)
        .map(|i| ((i % 61) as f32 - 30.0) * 0.013)
        .collect();
    let x = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
    for dt in [DType::Q2K, DType::Q3K, DType::Q5K] {
        let blk = match dt {
            DType::Q2K => 84,
            DType::Q3K => 110,
            DType::Q5K => 176,
            _ => unreachable!(),
        };
        let wbytes = in_f * out_f / 256 * blk;
        let mut src: Vec<u8> = (0..wbytes)
            .map(|i| ((i as u32).wrapping_mul(0x9E3779B1) >> 23) as u8)
            .collect();
        for b in src.chunks_exact_mut(blk) {
            match dt {
                DType::Q2K => {
                    b[80..82].copy_from_slice(&f16b(0.5));
                    b[82..84].copy_from_slice(&f16b(0.125));
                }
                DType::Q3K => b[108..110].copy_from_slice(&f16b(0.5)),
                DType::Q5K => {
                    b[0..2].copy_from_slice(&f16b(0.5));
                    b[2..4].copy_from_slice(&f16b(0.125));
                }
                _ => unreachable!(),
            }
        }
        let w = be
            .alloc(wbytes.next_multiple_of(4), BufferUsage::Weights)
            .unwrap();
        be.upload(w.as_ref(), &src).unwrap();
        let nblk = in_f / 32;
        let qa = be.alloc(in_f, BufferUsage::Activations).unwrap();
        let dact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let sact = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
        let y = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
        let mut runs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..3 {
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mw(
                dt,
                8,
                w.as_ref(),
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y.as_ref(),
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut out = vec![0u8; out_f * 4];
            be.download(y.as_ref(), &mut out).unwrap();
            runs.push(out);
        }
        assert!(
            runs[0] == runs[1] && runs[1] == runs[2],
            "{dt:?}: same-dispatch mmv_mw output differs across replays"
        );
        println!("  {dt:?}: 3x same-dispatch bit-identical");
    }
}
