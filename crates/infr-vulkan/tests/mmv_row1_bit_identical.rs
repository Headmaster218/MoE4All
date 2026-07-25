//! THE bit-identity guard the mmv_mw/mrow reassociation gap needed (README footnote 3): the m=1
//! DECODE tier and the m>=3 VERIFY/prefill tier must produce EXACT (not 5e-3-tolerant) results for
//! the same weights/activations/position, or `mtp_spec_matches_target_only_greedy` can fail even
//! when both streams are int8 (both-int8 is necessary but not sufficient for token-identity).
//!
//! The fix: [`infr_vulkan`]'s `Recorder::linear_mmv_mrow` is now the SAME shader dispatched at
//! `rows=1` (decode) or `rows=2..16` (verify) — see `native_mmv_mrow.comp`'s header. A row's
//! result depends only on its own `qa`/`dact`/`sact` slice, never on `rows`, so this test proves
//! decode-shape (rows=1) and verify-shape (rows=3) agree bit-for-bit, for every dtype on the
//! int8-decode/mrow tier — the set `native_mmv_mrow_variant_name` builds a `-DUSE_RES` rows=1 twin
//! for: the k-quants (Q4_K, Q6_K, Q2_K, Q3_K, Q5_K) AND the legacy 32-block family (Q8_0, Q4_0,
//! Q5_0, Q4_1, Q5_1, IQ4_NL). The legacy formats belong here for exactly the same reason Q5_K does
//! — several of them ship PREFILL-only (`mrow_int8_prefill_dtypes`) with their decode tier off on a
//! throughput tradeoff, but the kernels are dispatched at rows=1 the moment anyone flips one on, so
//! the bit-identity guard has to be standing BEFORE that edit, not after it.
//!
//! Q5_K is the dtype this whole invariant is NAMED after: the historical Q5_K int8 attempt wired
//! only the verify batch, left plain decode f32-exact, and flipped greedy tokens. It now has an
//! int8 arm in the unified kernel, so it belongs here — even though its decode tier ships OFF (a
//! measured throughput wash; see `adapter::mmv_int8_decode_dtypes`). This test exercises the
//! kernels directly, so it proves bit-identity regardless of that policy switch — i.e. it stays
//! meaningful as the guard for whoever flips Q5_K on later.
//!
//! Before this fix, decode (m=1) dispatched a DIFFERENT kernel (`native_mmv_mw.comp`,
//! warp-per-row `subgroupAdd`) whose cross-sub-block summation order differs from mrow's row-tile
//! accumulation — a reassociation-tolerance gap `mmv_mw_parity` already accepted at 5e-3 for
//! throughput purposes, but wide enough to flip an occasional greedy argmax. A standalone repro of
//! that pre-fix gap (`linear_mmv_mw` WARPS=1 vs `linear_mmv_mrow` m=3 row 0, Q4_K, same exact-
//! equality check this file uses) measured 1481/2048 output elements differing bit-for-bit (worst
//! abs diff 1.53e-5) — plenty to flip a greedy token. `linear_mmv_mw` itself is untouched by this
//! fix (Intel still uses it, see adapter.rs's `unified_mmv_row1`); AMD's decode tier simply no
//! longer calls it for these dtypes.
//!
//! Run: cargo test -p infr-vulkan --test mmv_row1_bit_identical -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

/// (bytes per block, elements per block) — the k-quants are 256-element super-blocks, the legacy
/// "_0"/"_1" family and IQ4_NL are flat 32-element blocks. Read from the shared decode spec
/// (`infr_core::decode_spec` via `infr_gguf::block_layout`, which reports `(elems, bytes)`) rather
/// than a hand-kept local table.
fn blk_shape(dt: DType) -> (usize, usize) {
    let (elems, bytes) = infr_gguf::block_layout(dt);
    (bytes, elems)
}

fn f16b(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}

fn patch_scales(dt: DType, bi: usize, blk: &mut [u8]) {
    let blen = blk.len();
    let d = 0.25 + (bi % 11) as f32 * 0.05;
    match dt {
        DType::Q4K => {
            blk[0..2].copy_from_slice(&f16b(d));
            blk[2..4].copy_from_slice(&f16b(0.1 + (bi % 5) as f32 * 0.02));
        }
        DType::Q6K => blk[blen - 2..blen].copy_from_slice(&f16b(d)),
        DType::Q2K => {
            blk[80..82].copy_from_slice(&f16b(d));
            blk[82..84].copy_from_slice(&f16b(0.1 + (bi % 5) as f32 * 0.02));
        }
        DType::Q3K => blk[108..110].copy_from_slice(&f16b(d)),
        // Q5_K: same leading [f16 d][f16 dmin] as Q4_K (the two share a scale/min layout family).
        DType::Q5K => {
            blk[0..2].copy_from_slice(&f16b(d));
            blk[2..4].copy_from_slice(&f16b(0.05 + (bi % 5) as f32 * 0.015));
        }
        // Legacy 32-block family: `d` always leads; the `_1` twins carry an additive (signed) min
        // right after it. Q8_0/Q4_0/Q5_0/IQ4_NL are single-scale.
        DType::Q8_0 | DType::Q4_0 | DType::Q5_0 | DType::Iq4Nl => {
            blk[0..2].copy_from_slice(&f16b(d));
        }
        DType::Q4_1 | DType::Q5_1 => {
            blk[0..2].copy_from_slice(&f16b(d));
            blk[2..4].copy_from_slice(&f16b(((bi % 7) as f32 - 3.0) * 0.04));
        }
        _ => unreachable!(),
    }
}

/// Exact bit-for-bit comparison: y computed by decode-shape dispatch (`linear_mmv_mrow`, rows=1)
/// must equal row 0 of a verify-shape dispatch (`linear_mmv_mrow`, rows=m) at the SAME weights and
/// the SAME row-0 activation. Also checks the fused-residual decode variant against the
/// non-residual result + a host-side add (same float, so still exact).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_row1_matches_mrow_row0_exact() {
    let be = VulkanBackend::new().unwrap();
    // (in_f, out_f) pairs exercise both layout gates: in_f < 2048 takes -DOUTS4, in_f >= 2048
    // takes the 2-output layout; odd out_f exercises the has1/tail guard.
    let shapes = [(1536usize, 66usize), (2048, 66), (2048, 2049), (6144, 2048)];
    for dt in [
        DType::Q4K,
        DType::Q6K,
        DType::Q2K,
        DType::Q3K,
        DType::Q5K,
        DType::Q8_0,
        DType::Q4_0,
        DType::Q5_0,
        DType::Q4_1,
        DType::Q5_1,
        DType::Iq4Nl,
    ] {
        for &(in_f, out_f) in &shapes {
            let (blk, belems) = blk_shape(dt);
            let wbytes = in_f * out_f / belems * blk;
            let mut wsrc: Vec<u8> = (0..wbytes)
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
                .collect();
            for (bi, b) in wsrc.chunks_exact_mut(blk).enumerate() {
                patch_scales(dt, bi, b);
            }
            let w = be
                .alloc(wbytes.next_multiple_of(4), BufferUsage::Weights)
                .unwrap();
            be.upload(w.as_ref(), &wsrc).unwrap();
            let nblk = in_f / 32;

            let m = 3usize;
            let xs: Vec<f32> = (0..m * in_f)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.021 + ((i % 13) as f32) * 0.004)
                .collect();
            let x = be.alloc(m * in_f * 4, BufferUsage::Activations).unwrap();
            be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
            let qa_m = be.alloc(m * in_f, BufferUsage::Activations).unwrap();
            let dact_m = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
            let sact_m = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
            let y_verify = be.alloc(m * out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa_m.as_ref(),
                dact_m.as_ref(),
                sact_m.as_ref(),
                m,
                in_f,
            );
            rec.linear_mmv_mrow(
                dt,
                w.as_ref(),
                0,
                qa_m.as_ref(),
                dact_m.as_ref(),
                sact_m.as_ref(),
                None,
                y_verify.as_ref(),
                m,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut verify_bytes = vec![0u8; m * out_f * 4];
            be.download(y_verify.as_ref(), &mut verify_bytes).unwrap();
            let verify_row0 = bytemuck::cast_slice::<u8, f32>(&verify_bytes)[0..out_f].to_vec();

            // Decode-shape: rows=1, same row-0 activation, quantized independently (as the real
            // decode call site does — a fresh quant_q8 over just that row).
            let x0 = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
            be.upload(x0.as_ref(), bytemuck::cast_slice(&xs[0..in_f]))
                .unwrap();
            let qa1 = be.alloc(in_f, BufferUsage::Activations).unwrap();
            let dact1 = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
            let sact1 = be.alloc(nblk * 2, BufferUsage::Activations).unwrap();
            let y_decode = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x0.as_ref(),
                qa1.as_ref(),
                dact1.as_ref(),
                sact1.as_ref(),
                1,
                in_f,
            );
            rec.linear_mmv_mrow(
                dt,
                w.as_ref(),
                0,
                qa1.as_ref(),
                dact1.as_ref(),
                sact1.as_ref(),
                None,
                y_decode.as_ref(),
                1,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut decode_bytes = vec![0u8; out_f * 4];
            be.download(y_decode.as_ref(), &mut decode_bytes).unwrap();
            let decode = bytemuck::cast_slice::<u8, f32>(&decode_bytes);

            let mut n_diff = 0usize;
            for (o, (&d, &v)) in decode.iter().zip(verify_row0.iter()).enumerate() {
                if d.to_bits() != v.to_bits() {
                    n_diff += 1;
                    if n_diff <= 3 {
                        eprintln!("  {dt:?} in{in_f} out{out_f}: col {o} decode={d} verify={v}");
                    }
                }
            }
            assert_eq!(
                n_diff, 0,
                "{dt:?} in_f={in_f} out_f={out_f}: decode (rows=1) and verify (rows={m}, row 0) \
                 disagree bit-for-bit on {n_diff}/{out_f} elements"
            );

            // Fused-residual decode variant: y = residual + x·Wᵀ must equal the plain decode
            // result plus a host-side add of the SAME residual values (IEEE-754 add is exact for
            // equal operands regardless of where it happens, so this stays an exact check too).
            let res_vals: Vec<f32> = (0..out_f)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
                .collect();
            let res = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            be.upload(res.as_ref(), bytemuck::cast_slice(&res_vals))
                .unwrap();
            let y_res = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.linear_mmv_mrow(
                dt,
                w.as_ref(),
                0,
                qa1.as_ref(),
                dact1.as_ref(),
                sact1.as_ref(),
                Some(res.as_ref()),
                y_res.as_ref(),
                1,
                in_f,
                out_f,
            );
            rec.finish().unwrap();
            let mut res_bytes = vec![0u8; out_f * 4];
            be.download(y_res.as_ref(), &mut res_bytes).unwrap();
            let got_res = bytemuck::cast_slice::<u8, f32>(&res_bytes);
            let mut n_diff_res = 0usize;
            for i in 0..out_f {
                let want = res_vals[i] + decode[i];
                if got_res[i].to_bits() != want.to_bits() {
                    n_diff_res += 1;
                }
            }
            assert_eq!(
                n_diff_res, 0,
                "{dt:?} in_f={in_f} out_f={out_f}: fused-residual decode disagrees with \
                 residual + plain-decode on {n_diff_res}/{out_f} elements"
            );
        }
    }
    eprintln!("mmv_row1 (decode) == mrow row 0 (verify) bit-for-bit, every int8-tier dtype");
}
