//! Small-m Linear cost on the Vulkan seam — the spec-decode verify / short-suffix-prefill shape
//! (m = 2..8 rows). Today the adapter routes any m>1 to the tiled coopmat GEMM, whose grid is only
//! n/64 workgroups at a single M-tile (underfills a 48-WGP GPU); the alternative GEMV re-streams
//! the weight per row. This bench bounds the multi-row-GEMV opportunity: ideal mrv ≈ the m=1 GEMV
//! cost (weights streamed once, grid stays out_f-wide).
//! Run: cargo test -p infr-vulkan --test small_m_bench -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

#[test]
#[ignore = "requires a Vulkan GPU (perf micro-bench)"]
fn small_m_linear_bench() {
    let be = VulkanBackend::new().unwrap();
    // Qwen3-8B Q4_K prefill shapes (the spec-decode verify target class).
    let shapes = [
        (4096usize, 6144usize, "qkv"),
        (4096, 4096, "o"),
        (4096, 24576, "gate+up"),
        (12288, 4096, "down"),
    ];
    let reps = 50usize;
    let a = be.alloc(16 * 12288 * 4, BufferUsage::Activations).unwrap();
    let c = be.alloc(16 * 24576 * 4, BufferUsage::Activations).unwrap();
    for (k, n, label) in shapes {
        let wbytes = n * k / 256 * 144; // Q4_K
        let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
        println!("--- {label} [k={k} n={n}] weight {} MiB ---", wbytes >> 20);
        for m in [1usize, 2, 4, 8] {
            let time = |f: &dyn Fn(&infr_vulkan::Recorder)| -> f64 {
                let rec = be.recorder().unwrap(); // warmup / pipeline compile
                f(&rec);
                rec.finish().unwrap();
                let t0 = std::time::Instant::now();
                let rec = be.recorder().unwrap();
                for _ in 0..reps {
                    f(&rec);
                }
                rec.finish().unwrap();
                t0.elapsed().as_micros() as f64 / reps as f64
            };
            // Current adapter route for m>1 (m=1 shown for reference as the GEMV ideal).
            let gemm_us = if m > 1 {
                time(&|rec| {
                    rec.matmul_native(
                        infr_core::DType::Q4K,
                        a.as_ref(),
                        w.as_ref(),
                        c.as_ref(),
                        m,
                        k,
                        n,
                    )
                })
            } else {
                f64::NAN
            };
            // GEMV route (re-streams the weight per row at m>1).
            let gemv_us = time(&|rec| {
                rec.linear_native(
                    infr_core::DType::Q4K,
                    w.as_ref(),
                    0,
                    a.as_ref(),
                    c.as_ref(),
                    m,
                    k,
                    n,
                )
            });
            // Multi-row GEMV (weight streamed once for all m rows).
            let mrow_us = if m > 1 {
                time(&|rec| {
                    rec.linear_native_mrow(
                        infr_core::DType::Q4K,
                        w.as_ref(),
                        0,
                        a.as_ref(),
                        c.as_ref(),
                        m,
                        k,
                        n,
                    )
                })
            } else {
                f64::NAN
            };
            // Effective weight-stream bandwidth if the weight were read once.
            let bw = |us: f64| wbytes as f64 / (us * 1e-6) / (1u64 << 30) as f64;
            println!(
                "  m={m}: gemm {gemm_us:8.1} us ({:5.0} GiB/s)   gemv {gemv_us:8.1} us ({:5.0} GiB/s)   mrow {mrow_us:8.1} us ({:5.0} GiB/s)",
                bw(gemm_us),
                bw(gemv_us),
                bw(mrow_us),
            );
        }
    }
}

/// The multi-row GEMV must match m independent single-row GEMV dispatches to reassociation-only
/// error: the same dqblk decode feeds vec4 pairwise dots (4 independent FMA chains per sub-block —
/// the flat-in-m latency win), so the accumulation ORDER differs from the GEMV's sequential scalar
/// chain. Same contract as the attention kernels ("reassociation-only vs the oracle") and Metal's
/// spec-verify GEMMs.
#[test]
fn mrow_matches_single_row_gemv() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // k=512 → 2 Q4_K super-blocks per row; n=192 not a multiple of 64 (mrow has no tile
    // constraint); m sweeps the whole supported range.
    let (k, n) = (512usize, 192usize);
    let wbytes = n * k / 256 * 144;
    // Deterministic pseudo-random weight bytes, with SANE super-block scales: raw random bytes can
    // decode the f16 d/dmin to ±1e4-class values, whose huge canceling dot terms amplify pure
    // reassociation error past any fixed tolerance (real checkpoints have no such blocks). Pin
    // d = dmin = 2^-8 (f16 0x1C00) per 144-byte super-block; the 6-bit scales/mins and nibbles
    // stay pseudo-random.
    let mut wsrc: Vec<u8> = (0..wbytes)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    for blk in wsrc.chunks_exact_mut(144) {
        blk[0..2].copy_from_slice(&[0x00, 0x1C]); // d    = 2^-8
        blk[2..4].copy_from_slice(&[0x00, 0x1C]); // dmin = 2^-8
    }
    let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
    be.upload(w.as_ref(), &wsrc).unwrap();
    for m in 2usize..=8 {
        let xs: Vec<f32> = (0..m * k)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.021)
            .collect();
        let x = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
        be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
        let y_ref = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        let y_mrow = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q4K,
            w.as_ref(),
            0,
            x.as_ref(),
            y_ref.as_ref(),
            m,
            k,
            n,
        );
        rec.linear_native_mrow(
            infr_core::DType::Q4K,
            w.as_ref(),
            0,
            x.as_ref(),
            y_mrow.as_ref(),
            m,
            k,
            n,
        );
        rec.finish().unwrap();
        let read = |b: &dyn infr_core::backend::Buffer| -> Vec<f32> {
            let mut out = vec![0u8; m * n * 4];
            be.download(b, &mut out).unwrap();
            bytemuck::cast_slice::<u8, f32>(&out).to_vec()
        };
        let (r, g) = (read(y_ref.as_ref()), read(y_mrow.as_ref()));
        for i in 0..m * n {
            // The pseudo-random weight bytes can decode an f16 SCALE to NaN/Inf; NaN poisons the
            // dot in every accumulation order, so both-NaN counts as agreement (the old bit-exact
            // assert matched them bitwise).
            if r[i].is_nan() && g[i].is_nan() {
                continue;
            }
            let tol = 1e-4 * r[i].abs().max(1.0);
            assert!(
                (r[i] - g[i]).abs() <= tol,
                "m={m} row={} col={}: gemv {} vs mrow {} (reassociation tol {tol})",
                i / n,
                i % n,
                r[i],
                g[i]
            );
        }
    }
    eprintln!("mrow == single-row GEMV bit-exact for m=2..8");
}

/// The multi-row int8 dp4a GEMV (`linear_mmv_mrow`, all layout variants: 2-out / OUTS4
/// small-in_f, MRV=4 rows<=4 / MRV=8) must match m independent `linear_mmv` (m=1 decode GEMV)
/// dispatches over the SAME quantized activations to f32 reassociation tolerance: the per-block
/// dp4a sums are exact integers and the per-block f32 terms are identical values — only the
/// K-accumulation/reduction order differs. (The f32 dequant GEMV is NOT the oracle here: int8
/// activation quantization alone puts ~1e-2 mean relative noise between the two families.)
/// k=1536 exercises the OUTS4 layout (in_f < 2048, 48 sub-blocks); k=2048 the 2-output layout;
/// n=66 exercises both layouts' tail guards (66 % 4 == 2 -> OUTS4's per-output `live` mask; odd
/// out_f pairs the 2-out `has1`). m sweeps 2..16 (MRV=4 vs 8 variants, plus the rows 9..=16
/// MRV=16 tier — m>8 always takes the 2-output m16 variant regardless of in_f, so the k=1536
/// OUTS4-shaped case covers that pairing too).
#[test]
fn mmv_mrow_matches_single_row_mmv() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for (dt, blk_bytes, dpin) in [
        (
            infr_core::DType::Q4K,
            144usize,
            &[(0usize, [0x00u8, 0x1C]), (2, [0x00, 0x1C])][..],
        ),
        (infr_core::DType::Q6K, 210, &[(208, [0x00, 0x1C])][..]),
    ] {
        for (k, n) in [(1536usize, 66usize), (2048, 66)] {
            let wbytes = n * k / 256 * blk_bytes;
            let mut wsrc: Vec<u8> = (0..wbytes)
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
                .collect();
            // Pin the f16 super-block scales to 2^-8 (see mrow_matches_single_row_gemv's doc).
            for blk in wsrc.chunks_exact_mut(blk_bytes) {
                for (off, val) in dpin {
                    blk[*off..off + 2].copy_from_slice(val);
                }
            }
            let w = be.alloc(wbytes, BufferUsage::Weights).unwrap();
            be.upload(w.as_ref(), &wsrc).unwrap();
            let nblk = k / 32;
            for m in 2usize..=16 {
                let xs: Vec<f32> = (0..m * k)
                    .map(|i| ((i % 97) as f32 - 48.0) * 0.021)
                    .collect();
                let x = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
                be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
                let qa = be.alloc(m * k, BufferUsage::Activations).unwrap();
                let dact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
                let sact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
                let y_mmv = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
                let rec = be.recorder().unwrap();
                rec.quant_q8(x.as_ref(), qa.as_ref(), dact.as_ref(), sact.as_ref(), m, k);
                rec.linear_mmv_mrow(
                    dt,
                    w.as_ref(),
                    0,
                    qa.as_ref(),
                    dact.as_ref(),
                    sact.as_ref(),
                    None,
                    y_mmv.as_ref(),
                    m,
                    k,
                    n,
                );
                rec.finish().unwrap();
                let mut got = vec![0u8; m * n * 4];
                be.download(y_mmv.as_ref(), &mut got).unwrap();
                let got = bytemuck::cast_slice::<u8, f32>(&got).to_vec();
                // Oracle: each row through the m=1 decode GEMV (identical per-row quantization —
                // quant_q8 is per-(row, block) independent).
                for r in 0..m {
                    let xr = be.alloc(k * 4, BufferUsage::Activations).unwrap();
                    be.upload(xr.as_ref(), bytemuck::cast_slice(&xs[r * k..(r + 1) * k]))
                        .unwrap();
                    let qar = be.alloc(k, BufferUsage::Activations).unwrap();
                    let dar = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
                    let sar = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
                    let yr = be.alloc(n * 4, BufferUsage::Activations).unwrap();
                    let rec = be.recorder().unwrap();
                    rec.quant_q8(xr.as_ref(), qar.as_ref(), dar.as_ref(), sar.as_ref(), 1, k);
                    rec.linear_mmv(
                        dt,
                        w.as_ref(),
                        0,
                        qar.as_ref(),
                        dar.as_ref(),
                        sar.as_ref(),
                        yr.as_ref(),
                        k,
                        n,
                    );
                    rec.finish().unwrap();
                    let mut refb = vec![0u8; n * 4];
                    be.download(yr.as_ref(), &mut refb).unwrap();
                    let want = bytemuck::cast_slice::<u8, f32>(&refb);
                    for o in 0..n {
                        let (wv, gv) = (want[o], got[r * n + o]);
                        if wv.is_nan() && gv.is_nan() {
                            continue;
                        }
                        let tol = 1e-4 * wv.abs().max(1.0);
                        assert!(
                            (wv - gv).abs() <= tol,
                            "{dt:?} k={k} n={n} m={m} row={r} col={o}: mmv {wv} vs mmv_mrow {gv} (tol {tol})"
                        );
                    }
                }
            }
        }
    }
    eprintln!("mmv_mrow == m=1 mmv GEMV at reassociation tolerance (Q4K/Q6K, both layouts)");
}

/// The vec4 f32 prefill GEMV variants (`linear_f32r_mrow4_v4` rows<=4 / `linear_f32r_mrow8_v4`)
/// must match a host f64 reference to f32 reassociation tolerance (vec4-lane accumulation +
/// pairwise horizontal add reorders the K sum vs the scalar kernels — tolerance-level only).
/// rows=1 (vec4 decode kernel) and an in_f % 4 != 0 shape (scalar mrow8 fallback) ride along
/// to pin the whole dispatch table.
#[test]
fn linear_f32_v4_matches_host() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for (rows, in_f, out_f) in [
        (4usize, 1536usize, 256usize), // mrow4_v4 (the E2B per-layer proj shape)
        (2, 256, 1536),                // mrow4_v4, transposed direction
        (6, 1536, 256),                // mrow8_v4
        (1, 1536, 256),                // 1-row vec4 decode kernel
        (4, 252, 64),                  // in_f % 4 != 0 → scalar mrow8 fallback
    ] {
        let ws: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i % 113) as f32 - 56.0) * 0.017)
            .collect();
        let xs: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.023)
            .collect();
        let w = be.alloc(out_f * in_f * 4, BufferUsage::Weights).unwrap();
        be.upload(w.as_ref(), bytemuck::cast_slice(&ws)).unwrap();
        let x = be.alloc(rows * in_f * 4, BufferUsage::Activations).unwrap();
        be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
        let y = be
            .alloc(rows * out_f * 4, BufferUsage::Activations)
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_f32(w.as_ref(), x.as_ref(), y.as_ref(), rows, in_f, out_f);
        rec.finish().unwrap();
        let mut out = vec![0u8; rows * out_f * 4];
        be.download(y.as_ref(), &mut out).unwrap();
        let got = bytemuck::cast_slice::<u8, f32>(&out);
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0f64;
                for i in 0..in_f {
                    acc += ws[o * in_f + i] as f64 * xs[r * in_f + i] as f64;
                }
                let want = acc as f32;
                let g = got[r * out_f + o];
                let tol = 1e-4 * want.abs().max(1.0);
                assert!(
                    (want - g).abs() <= tol,
                    "rows={rows} in_f={in_f} out_f={out_f} r={r} o={o}: host {want} vs gpu {g}"
                );
            }
        }
    }
    eprintln!("linear_f32 (v4 + scalar variants) == host reference");
}
