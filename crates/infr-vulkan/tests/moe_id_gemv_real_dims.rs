//! REAL-dims regression for the grid i-quant MoE id-GEMV floor: Qwen3.6-35B-A3B-UD-IQ3_S ships
//! IQ2_S gate/up banks (in=2048, out=512) and IQ3_S down banks (in=512, out=2048) over 256
//! experts / 8 used / 40 layers. The synthetic small-bank parity suite
//! (`moe_id_gemv_new_formats_parity.rs`) passes while the real model device-losts: one decode
//! step submits 40x(gate,up,down) = 120 `linear_native_id_multi` dispatches in one queue submit,
//! and when the grid decode is scratch-bound (dynamically-indexed const grid LUTs → one
//! per-invocation scratch table copy PER ACCESS SITE; the IQ2_S pipeline carried 1 MiB of scratch)
//! the real model's decode step exceeds amdgpu's ~10s gfx-ring timeout (TDR, `ring gfx_0.0.0
//! timeout` in dmesg — NOT a page fault). This test replays that decode-step shape with synthetic
//! banks and asserts the submit stays FAR from the timeout: the scratch-bound kernels measured
//! 1.4 s here (and >10 s on the real model, whose 256 cold experts/layer defeat every cache);
//! the shared-memory-staged kernels (grid_init() — see build.rs::gen_grids) measure ~14 ms.
//! Parity vs the host dequant is checked on every path so a fast-but-wrong kernel still fails.
//!
//! Run: `cargo test -p infr-vulkan --test moe_id_gemv_real_dims -- --ignored --nocapture`
use std::time::Instant;

use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::linear::pad_to_u32_align;
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

/// (elements per block, bytes per block) — grid i-quant subset under test.
fn block_geom(dt: DType) -> (usize, usize) {
    match dt {
        DType::Iq2Xxs => (256, 66),
        DType::Iq2Xs => (256, 74),
        DType::Iq2S => (256, 82),
        DType::Iq3Xxs => (256, 98),
        DType::Iq3S => (256, 110),
        DType::Iq1S => (256, 50),
        DType::Iq1M => (256, 56),
        other => panic!("no geometry for {other:?}"),
    }
}

/// One synthetic VALID block: random payload, scale patched small (same recipe as the small-bank
/// parity suite — see moe_id_gemv_new_formats_parity.rs's module doc for why this is valid).
fn synth_bank(dt: DType, n_elems: usize, seed: u64) -> Vec<u8> {
    let (epb, bpb) = block_geom(dt);
    assert_eq!(n_elems % epb, 0);
    let mut rng = Rng(seed);
    let d16 = half::f16::from_f32(0.02).to_le_bytes();
    let mut out = vec![0u8; n_elems / epb * bpb];
    for blk in out.chunks_exact_mut(bpb) {
        for b in blk.iter_mut() {
            *b = rng.byte();
        }
        match dt {
            DType::Iq1M => {
                let d_bits = half::f16::from_f32(0.02).to_bits();
                for i in 0..4usize {
                    let lo = u16::from_le_bytes([blk[48 + 2 * i], blk[49 + 2 * i]]) & 0x0FFF;
                    let w = lo | (((d_bits >> (4 * i)) & 0xF) << 12);
                    blk[48 + 2 * i..50 + 2 * i].copy_from_slice(&w.to_le_bytes());
                }
            }
            _ => blk[0..2].copy_from_slice(&d16),
        }
    }
    out
}

fn host_gemv(w: &[f32], x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    (0..out_f)
        .map(|o| (0..in_f).map(|i| w[o * in_f + i] * x[i]).sum())
        .collect()
}

/// One real-shape role bank on the GPU plus everything needed to dispatch/check it.
struct Role {
    dt: DType,
    in_f: usize,
    out_f: usize,
    stride: usize,
    bank: Vec<u8>,
    wbuf: Box<dyn infr_core::backend::Buffer>,
    x_buf: Box<dyn infr_core::backend::Buffer>,
    y_buf: Box<dyn infr_core::backend::Buffer>,
    x: Vec<f32>,
}

const N_EXPERT: usize = 256;
const N_USED: usize = 8;
const N_LAYER: usize = 40;
/// Distinct weight banks cycled across the 40 layers: one shared bank would sit in the 7900 XTX's
/// 96 MiB Infinity Cache and hide the real model's per-layer cold-bank traffic (each real layer
/// streams ~290 MiB of distinct expert weights).
const N_BANKS: usize = 8;

fn make_role(be: &VulkanBackend, dt: DType, in_f: usize, out_f: usize, seed: u64) -> Role {
    let stride = in_f * out_f;
    let bank = synth_bank(dt, N_EXPERT * stride, 0x5eed ^ dt as u64 ^ (seed << 32));
    let padded = pad_to_u32_align(&bank);
    let wbuf = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
    be.upload(wbuf.as_ref(), &padded).unwrap();
    let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let y_buf = be
        .alloc(N_USED * out_f * 4, BufferUsage::Activations)
        .unwrap();
    Role {
        dt,
        in_f,
        out_f,
        stride,
        bank,
        wbuf,
        x_buf,
        y_buf,
        x,
    }
}

/// One idm dispatch of `role` in its own submit, returning the wall time of record+submit+wait.
fn dispatch_role(
    be: &VulkanBackend,
    role: &Role,
    ids_buf: &dyn infr_core::backend::Buffer,
) -> std::time::Duration {
    let t0 = Instant::now();
    let rec = be.recorder().unwrap();
    rec.linear_native_id_multi(
        role.dt,
        role.wbuf.as_ref(),
        ids_buf,
        N_USED,
        role.stride,
        role.x_buf.as_ref(),
        false,
        role.y_buf.as_ref(),
        role.in_f,
        role.out_f,
        1,
    );
    rec.finish().unwrap();
    t0.elapsed()
}

fn check_role(role: &Role, be: &VulkanBackend, ids: &[u32]) {
    let (epb, bpb) = block_geom(role.dt);
    let stride_bytes = role.stride / epb * bpb;
    let mut out = vec![0u8; N_USED * role.out_f * 4];
    be.download(role.y_buf.as_ref(), &mut out).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
    for (slot, &eid) in ids.iter().enumerate() {
        let eb = &role.bank[eid as usize * stride_bytes..(eid as usize + 1) * stride_bytes];
        let w = infr_gguf::dequant::dequant_block(role.dt, eb).unwrap();
        let want = host_gemv(&w, &role.x, role.in_f, role.out_f);
        let g = &got[slot * role.out_f..(slot + 1) * role.out_f];
        for (i, (gv, wv)) in g.iter().zip(want.iter()).enumerate() {
            assert!(
                (gv - wv).abs() < 1e-3 + 1e-3 * wv.abs(),
                "{:?} slot {slot} (expert {eid}) mismatch at {i}: got {gv} want {wv}",
                role.dt
            );
        }
    }
}

/// Real decode-step shape for the exact formats in Qwen3.6-35B-A3B-UD-IQ3_S: IQ2_S gate/up +
/// IQ3_S down, 40 layers x 3 id-multi dispatches in ONE submit. Device-losts (TDR) when the
/// grid decode is scratch-bound; parity-checked against the host dequant when it survives.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn real_dims_decode_step_iq2s_iq3s() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let gates: Vec<Role> = (0..N_BANKS)
        .map(|i| make_role(&be, DType::Iq2S, 2048, 512, i as u64))
        .collect();
    let downs: Vec<Role> = (0..N_BANKS)
        .map(|i| make_role(&be, DType::Iq3S, 512, 2048, i as u64))
        .collect();
    let (gate, down) = (&gates[0], &downs[0]);

    // ids scrambled across the full 256-expert range so wrong-expert/oob strides surface.
    let ids: Vec<u32> = vec![255, 0, 128, 37, 200, 91, 250, 3];
    let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();

    // Per-dispatch timing (diagnosis aid): warm-up dispatch first (first use compiles the
    // pipeline — 100+ ms that would otherwise drown the kernel time), then one timed dispatch.
    for role in [gate, down] {
        dispatch_role(&be, role, ids_buf.as_ref());
        let el = dispatch_role(&be, role, ids_buf.as_ref());
        println!(
            "{:?} single idm dispatch ({}x{} x{} slots): {el:.1?}",
            role.dt, role.in_f, role.out_f, N_USED,
        );
        check_role(role, &be, &ids);
    }

    // The real decode step: 40 layers x (gate, up, down) in ONE submit — this is the shape that
    // must stay under the ~10s ring timeout.
    let t0 = Instant::now();
    let rec = be.recorder().unwrap();
    for l in 0..N_LAYER {
        for role in [
            &gates[l % N_BANKS],
            &gates[l % N_BANKS],
            &downs[l % N_BANKS],
        ] {
            rec.linear_native_id_multi(
                role.dt,
                role.wbuf.as_ref(),
                ids_buf.as_ref(),
                N_USED,
                role.stride,
                role.x_buf.as_ref(),
                false,
                role.y_buf.as_ref(),
                role.in_f,
                role.out_f,
                1,
            );
        }
    }
    rec.finish().unwrap();
    let dt = t0.elapsed();
    println!("decode-step submit (40x[gate,up,down]): {dt:.1?}");
    check_role(gate, &be, &ids);
    check_role(down, &be, &ids);
    // 200 ms = 14x above the shared-staged kernels' measured 14 ms (headroom for slow hosts),
    // 7x below the scratch-bound kernels' measured 1.4 s (which TDR'd the real model, whose
    // decode step is strictly heavier than this replay: 256 cold experts/layer, no bank reuse).
    assert!(
        dt.as_secs_f64() < 0.2,
        "decode-step submit took {dt:.1?} — scratch-bound grid decode (TDR territory on a real \
         model; amdgpu ring timeout is ~10s)"
    );
}

/// Every OTHER grid format at the same real bank shape (one idm dispatch each, parity-checked):
/// catches common-mode slowness/indexing bugs across the whole USE_GRID family, not just the two
/// formats the first field model happened to ship.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn real_dims_all_grid_formats() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let ids: Vec<u32> = vec![255, 0, 128, 37, 200, 91, 250, 3];
    let ids_buf = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();
    for dt in [
        DType::Iq2Xxs,
        DType::Iq2Xs,
        DType::Iq3Xxs,
        DType::Iq1S,
        DType::Iq1M,
    ] {
        let role = make_role(&be, dt, 2048, 512, 0);
        // Warm-up (first use compiles the pipeline), then timed.
        dispatch_role(&be, &role, ids_buf.as_ref());
        let el = dispatch_role(&be, &role, ids_buf.as_ref());
        println!("{dt:?} single idm dispatch (2048x512 x8 slots): {el:.1?}");
        check_role(&role, &be, &ids);
        // 120 dispatches/step must fit the ~10s ring timeout with wide margin: shared-staged
        // kernels measure well under 1 ms here; the scratch-bound ones measured 3-30 ms warm
        // (scaling with grid table size), which stacked past the timeout on the real model.
        assert!(
            el.as_secs_f64() < 0.003,
            "{dt:?} single dispatch took {el:.1?} — scratch-bound grid decode (TDR territory at \
             real decode-step scale)"
        );
    }
}
