//! Host-reference parity for the LEGACY 32-block int8 dp4a GEMV arms — Q8_0, Q4_0, Q5_0, Q4_1,
//! Q5_1, IQ4_NL (`native_mmv_mrow.comp`'s `FMT_Q8_0`/`FMT_Q4_0`/… `wdec`s).
//!
//! Same technique as `mmv_mw_parity`'s Q2_K/Q3_K/Q5_K host references, and for the same reason: the
//! shader unpacks these blocks WORD-parallel (aligned/funnel-shifted u32 loads, nibble masks, SWAR
//! rebias, a 4-bit→4-byte-lane `qh` spread), so a reference that merely re-expressed the same
//! word tricks would prove nothing. This one re-derives each block layout from the GGUF spec
//! BYTE-at-a-time and element-at-a-time (`wcode` below), accumulates in f64 to take reassociation
//! out of the comparison, and dots it against the GPU's OWN `quant_q8` output (qa/dact/sact read
//! back) so only the weight-side math is under test.
//!
//! These are the traps this file exists to catch:
//!   * **Alignment.** Q8_0's 34-byte, Q5_0's 22-byte and Q4_0/IQ4_NL's 18-byte block strides are
//!     all `% 4 == 2`, so every odd block sits at a 2-mod-4 byte offset and a plain `nw[byte >> 2]`
//!     word load would silently misread HALF the blocks (Q4_1's 20 B and Q5_1's 24 B are aligned —
//!     a per-format property, not a family one). Odd `out_f` shapes below put both parities on
//!     every dtype.
//!   * **SWAR rebias.** `q − 8` (Q4_0) / `q − 16` (Q5_0) done as one 32-bit subtract would borrow
//!     across byte lanes; the shader uses the `(q + K) ^ 0x80808080` identity instead.
//!   * **Min folding.** Q4_1/Q5_1's `w = q·d + m` has NO zero-point — the min rides the ones-dot Σa
//!     (`sact`), with `f1 = −m` through the shared `f0·dp − f1·sact` arm.
//!
//! Run: cargo test -p infr-vulkan --test mmv_mrow_legacy_formats -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

/// Bytes per 32-element block (llama.cpp `block_q*` sizes), from the shared decode spec
/// (`infr_core::decode_spec` via `infr_gguf::block_layout`) — not a hand-kept local table. Each
/// format's field layout is documented in the spec's `block_spec` arm.
fn blk_bytes(dt: DType) -> usize {
    infr_gguf::block_layout(dt).1
}

/// The IQ4_NL non-linear codebook (llama.cpp `kvalues_iq4nl`), spelled out rather than decoded from
/// the shader's packed-word form — the whole point of a from-scratch reference.
const KV_IQ4NL: [i32; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn f16b(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_le_bytes()
}
fn f16v(lo: u8, hi: u8) -> f32 {
    half::f16::from_le_bytes([lo, hi]).to_f32()
}
fn u32le(w: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([w[at], w[at + 1], w[at + 2], w[at + 3]])
}

/// From-scratch decode of the 32-element block containing `gelem`: the 32 SIGNED integer codes the
/// dp4a operand must hold, the block scale `d`, and the additive min `m` (0 for the symmetric
/// formats). Element order per llama.cpp's `dequantize_row_*`: for the 4-bit families element
/// `j < 16` is the LOW nibble of `qs[j]` and element `16 + j` the HIGH nibble of the SAME byte.
fn wcode(dt: DType, w: &[u8], gelem: usize) -> ([i32; 32], f32, f32) {
    let bd = (gelem / 32) * blk_bytes(dt);
    let mut q = [0i32; 32];
    let d = f16v(w[bd], w[bd + 1]);
    let mut m = 0.0f32;
    match dt {
        DType::Q8_0 => {
            for (j, qj) in q.iter_mut().enumerate() {
                *qj = w[bd + 2 + j] as i8 as i32;
            }
        }
        DType::Q4_0 => {
            for j in 0..16 {
                let b = w[bd + 2 + j];
                q[j] = (b & 0xF) as i32 - 8;
                q[16 + j] = (b >> 4) as i32 - 8;
            }
        }
        DType::Iq4Nl => {
            for j in 0..16 {
                let b = w[bd + 2 + j];
                q[j] = KV_IQ4NL[(b & 0xF) as usize];
                q[16 + j] = KV_IQ4NL[(b >> 4) as usize];
            }
        }
        DType::Q5_0 => {
            let qh = u32le(w, bd + 2);
            for j in 0..16 {
                let b = w[bd + 6 + j];
                q[j] = (((b & 0xF) as u32 | (((qh >> j) & 1) << 4)) as i32) - 16;
                q[16 + j] = (((b >> 4) as u32 | (((qh >> (16 + j)) & 1) << 4)) as i32) - 16;
            }
        }
        DType::Q4_1 => {
            m = f16v(w[bd + 2], w[bd + 3]);
            for j in 0..16 {
                let b = w[bd + 4 + j];
                q[j] = (b & 0xF) as i32; // NO zero-point — the min is additive
                q[16 + j] = (b >> 4) as i32;
            }
        }
        DType::Q5_1 => {
            m = f16v(w[bd + 2], w[bd + 3]);
            let qh = u32le(w, bd + 4);
            for j in 0..16 {
                let b = w[bd + 8 + j];
                q[j] = ((b & 0xF) as u32 | (((qh >> j) & 1) << 4)) as i32;
                q[16 + j] = ((b >> 4) as u32 | (((qh >> (16 + j)) & 1) << 4)) as i32;
            }
        }
        _ => unreachable!(),
    }
    (q, d, m)
}

/// Pseudo-random weight bank + sane per-block f16 scales (and mins for the `_1` twins).
fn make_weights(dt: DType, nblocks: usize) -> Vec<u8> {
    let blk = blk_bytes(dt);
    let mut src: Vec<u8> = (0..nblocks * blk)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    for (bi, b) in src.chunks_exact_mut(blk).enumerate() {
        b[0..2].copy_from_slice(&f16b(0.25 + (bi % 11) as f32 * 0.05));
        if matches!(dt, DType::Q4_1 | DType::Q5_1) {
            // Signed min, block-varying — a positive-only min would hide a sign error in the
            // `f1 = −m` fold.
            b[2..4].copy_from_slice(&f16b(((bi % 7) as f32 - 3.0) * 0.04));
        }
    }
    src
}

/// The GPU int8 mrow GEMV must match a from-scratch host replay of the block layout at the m>=3
/// prefill/verify shape. (The m=1 decode shape's agreement with row 0 of this dispatch is the
/// separate, EXACT bit-identity contract — `mmv_row1_bit_identical` covers every dtype here.)
#[test]
#[ignore = "requires a Vulkan GPU"]
fn mmv_mrow_legacy_formats_match_host_reference() {
    let be = VulkanBackend::new().unwrap();
    // in_f < 2048 takes the -DOUTS4 layout, >= 2048 the 2-output one; odd out_f exercises the tail
    // guard AND puts the last row's blocks at the opposite 4-byte parity for the odd-stride
    // formats (Q8_0/Q5_0/Q4_0/IQ4_NL).
    let shapes = [(1536usize, 66usize), (2048, 2048), (6144, 2049)];
    let dtypes = [
        DType::Q8_0,
        DType::Q4_0,
        DType::Q5_0,
        DType::Q4_1,
        DType::Q5_1,
        DType::Iq4Nl,
    ];
    let read = |b: &dyn infr_core::backend::Buffer, n: usize| -> Vec<u8> {
        let mut out = vec![0u8; n];
        be.download(b, &mut out).unwrap();
        out
    };
    let mut worst = 0f64;
    for dt in dtypes {
        for &(in_f, out_f) in &shapes {
            let src = make_weights(dt, in_f * out_f / 32);
            let w = be
                .alloc(src.len().next_multiple_of(4), BufferUsage::Weights)
                .unwrap();
            be.upload(w.as_ref(), &src).unwrap();

            let m = 3usize;
            let nblk = in_f / 32;
            let xs: Vec<f32> = (0..m * in_f)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.021 + ((i % 13) as f32) * 0.004)
                .collect();
            let x = be.alloc(m * in_f * 4, BufferUsage::Activations).unwrap();
            be.upload(x.as_ref(), bytemuck::cast_slice(&xs)).unwrap();
            let qa = be.alloc(m * in_f, BufferUsage::Activations).unwrap();
            let dact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
            let sact = be.alloc(m * nblk * 2, BufferUsage::Activations).unwrap();
            let y = be.alloc(m * out_f * 4, BufferUsage::Activations).unwrap();
            let rec = be.recorder().unwrap();
            rec.quant_q8(
                x.as_ref(),
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                m,
                in_f,
            );
            rec.linear_mmv_mrow(
                dt,
                w.as_ref(),
                0,
                qa.as_ref(),
                dact.as_ref(),
                sact.as_ref(),
                None,
                y.as_ref(),
                m,
                in_f,
                out_f,
            );
            rec.finish().unwrap();

            // The GPU's own quantized activations — only the weight-side unpack is under test.
            let qa_h: Vec<i32> = read(qa.as_ref(), m * in_f)
                .iter()
                .map(|&b| b as i8 as i32)
                .collect();
            let da_h: Vec<f32> = read(dact.as_ref(), m * nblk * 2)
                .chunks_exact(2)
                .map(|c| f16v(c[0], c[1]))
                .collect();
            let sa_h: Vec<f32> = read(sact.as_ref(), m * nblk * 2)
                .chunks_exact(2)
                .map(|c| f16v(c[0], c[1]))
                .collect();
            let got: Vec<f32> =
                bytemuck::cast_slice::<u8, f32>(&read(y.as_ref(), m * out_f * 4)).to_vec();

            // Metric: worst absolute deviation, scaled by the RMS of the reference outputs — NOT a
            // per-element relative error. These weight banks are pseudo-random, so an output dot is
            // a random walk and a handful of the 2049 outputs land near zero by cancellation; their
            // per-element relative error explodes on f32-vs-f64 rounding alone and says nothing
            // about the unpack (Q5_0's codes span −16..15 with a nonzero mean, so it cancels
            // hardest and peaked at 4.6e-3 per-element while being bit-exact). Against the output
            // RMS, an actual mis-decoded lane — one wrong nibble, one dropped `qh` bit, one SWAR
            // borrow across a byte — moves this by orders of magnitude, which is what it's for.
            let mut worst_abs = 0f64;
            let mut sumsq = 0f64;
            for r in 0..m {
                for o in 0..out_f {
                    let mut acc = 0f64;
                    for s in 0..nblk {
                        let (q, d, mn) = wcode(dt, &src, o * in_f + s * 32);
                        let mut dp = 0i64;
                        for (j, &qj) in q.iter().enumerate() {
                            dp += qj as i64 * qa_h[r * in_f + s * 32 + j] as i64;
                        }
                        acc += d as f64 * da_h[r * nblk + s] as f64 * dp as f64
                            + mn as f64 * sa_h[r * nblk + s] as f64;
                    }
                    let g = got[r * out_f + o] as f64;
                    worst_abs = worst_abs.max((g - acc).abs());
                    sumsq += acc * acc;
                }
            }
            let rms = (sumsq / (m * out_f) as f64).sqrt();
            let mr = worst_abs / rms;
            worst = worst.max(mr);
            println!(
                "  {dt:?} in{in_f} out{out_f} m{m}: worst |Δ| {worst_abs:.3e} / out-RMS {rms:.3e} \
                 = {mr:.2e}"
            );
            assert!(
                mr < 2e-3,
                "{dt:?} {in_f}x{out_f}: int8 mrow diverges from the host reference ({mr:.2e} of \
                 the output RMS)"
            );
        }
    }
    println!("legacy 32-block int8 mrow == from-scratch host reference (worst {worst:.2e})");
}
