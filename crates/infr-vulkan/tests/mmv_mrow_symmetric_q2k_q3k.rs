//! Symmetric-int8 proof for Q2_K/Q3_K/Q5_K: the m=1 DECODE tier (`linear_mmv_mw`, gated by
//! `adapter::mmv_mw_choice`) and the m>=2 VERIFY/prefill tier (`linear_mmv_mrow`, gated by
//! `adapter::mmv_gate`) must agree row-for-row, since both now take the int8 dp4a path for these
//! dtypes by default on AMD (see adapter.rs's `mmv_mw_choice` doc). A dtype that is int8 in one
//! stream and f32-exact (or differently-reassociated int8) in the other is exactly the bug class
//! that broke Q5_K MTP token-identity — this test is the guard against reintroducing it for
//! Q2_K/Q3_K (and now Q5_K itself, whose original break IS this bug class) now that
//! `native_mmv_mrow.comp` has FMT_Q2K/FMT_Q3K/FMT_Q5K support (ported from `native_mmv_mw.comp`'s
//! already host-reference-validated dequant math — see `mmv_mw_parity::mmv_mw_q2k_q3k_match_host_
//! reference` / `mmv_mw_q5k_match_host_reference`). Q5_K's own m=1/m>=2 kernels are exercised here
//! at the RAW `Recorder` level (bypassing `mrow_int8_dtype_ok`'s decode-tied policy gate, which
//! only controls whether either tier is *reachable by default* — this test proves the two kernels
//! themselves agree, independent of that policy switch, exactly as it already does for Q2_K/Q3_K).
//!
//! Run: cargo test -p infr-vulkan --test mmv_mrow_symmetric_q2k_q3k -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

#[test]
#[ignore = "requires a Vulkan GPU"]
fn mrow_matches_mmv_mw_q2k_q3k() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let blk = |dt: DType| match dt {
        DType::Q2K => 84usize,
        DType::Q3K => 110,
        DType::Q5K => 176,
        _ => unreachable!(),
    };
    let f16b = |x: f32| half::f16::from_f32(x).to_le_bytes();
    for dt in [DType::Q2K, DType::Q3K, DType::Q5K] {
        // Same shapes small_m_bench's mmv_mrow_matches_single_row_mmv uses: k=1536 hits the OUTS4
        // layout, k=2048 the 2-output layout; n=66 exercises both layouts' tail guards.
        for (k, n) in [(1536usize, 66usize), (2048, 66)] {
            let wbytes = n * k / 256 * blk(dt);
            let mut wsrc: Vec<u8> = (0..wbytes)
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
                .collect();
            for (bi, b) in wsrc.chunks_exact_mut(blk(dt)).enumerate() {
                let d = 0.25 + (bi % 11) as f32 * 0.05;
                match dt {
                    DType::Q2K => {
                        b[80..82].copy_from_slice(&f16b(d));
                        b[82..84].copy_from_slice(&f16b(0.1 + (bi % 5) as f32 * 0.02));
                    }
                    DType::Q3K => b[108..110].copy_from_slice(&f16b(d)),
                    DType::Q5K => {
                        b[0..2].copy_from_slice(&f16b(d));
                        b[2..4].copy_from_slice(&f16b(0.05 + (bi % 5) as f32 * 0.015));
                    }
                    _ => unreachable!(),
                }
            }
            let w = be
                .alloc(wbytes.next_multiple_of(4), BufferUsage::Weights)
                .unwrap();
            be.upload(w.as_ref(), &wsrc).unwrap();
            let nblk = k / 32;
            for m in [2usize, 3, 6, 7, 8, 11, 16] {
                let xs: Vec<f32> = (0..m * k)
                    .map(|i| ((i % 97) as f32 - 48.0) * 0.021 + ((i % 13) as f32) * 0.004)
                    .collect();
                let x = be.alloc(m * k * 4, BufferUsage::Activations).unwrap();
                be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
                let qa = be.alloc(m * k, BufferUsage::Activations).unwrap();
                let dact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
                let sact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
                let y_mrow = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
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
                    y_mrow.as_ref(),
                    m,
                    k,
                    n,
                );
                rec.finish().unwrap();
                let mut got = vec![0u8; m * n * 4];
                be.download(y_mrow.as_ref(), &mut got).unwrap();
                let got = bytemuck::cast_slice::<u8, f32>(&got).to_vec();

                // Oracle: the m=1 DECODE tier (linear_mmv_mw, the same kernel adapter.rs's
                // `mmv_mw_choice` picks for plain decode) on each row independently, same
                // per-row quant_q8 activations.
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
                    rec.linear_mmv_mw(
                        dt,
                        8, // WARPS=8: adapter.rs's default INFR_MMV_MW_WARPS
                        w.as_ref(),
                        0,
                        qar.as_ref(),
                        dar.as_ref(),
                        sar.as_ref(),
                        None,
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
                        let tol = 5e-3 * wv.abs().max(1e-2);
                        assert!(
                            (wv - gv).abs() <= tol,
                            "{dt:?} k={k} n={n} m={m} row={r} col={o}: mmv_mw(decode) {wv} vs \
                             mmv_mrow(verify) {gv} (tol {tol}) — decode/verify DISAGREE"
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "Q2_K/Q3_K/Q5_K: verify tier (mmv_mrow, m=2..16) == decode tier (mmv_mw, m=1) at \
         reassociation tolerance — symmetric int8"
    );
}
