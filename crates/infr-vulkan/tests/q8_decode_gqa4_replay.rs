//! Parity coverage for the hd256 planar-Q8 GQA4 record-once Decode partial.
//!
//! Both legs use BDA explicitly. The candidate reuses each K/V load across four adjacent query
//! heads; the reference is the established per-query-head dynamic partial.

use std::sync::Arc;

use infr_core::{
    backend::{Backend, BufferUsage},
    config::Config,
};
use infr_vulkan::VulkanBackend;

fn sample(i: usize, seed: u32) -> f32 {
    let mut x = (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(seed);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    ((x & 0xffff) as f32 / 32767.5) - 1.0
}

fn to_f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn run_case(
    gqa4: bool,
    qk_i8: bool,
    chunk1024: bool,
    combine_sg: bool,
    kv_len: usize,
) -> Option<Vec<f32>> {
    let mut cfg = Config::default();
    cfg.device.dev = Some("Vulkan1".to_string());
    cfg.kernels.vulkan.q8_decode_d8 = true;
    cfg.kernels.vulkan.q8_decode_ls128 = true;
    cfg.kernels.vulkan.q8_decode_ls256 = true;
    cfg.kernels.vulkan.q8_qk_f16 = true;
    cfg.kernels.vulkan.q8_qk_i8 = qk_i8;
    cfg.kernels.vulkan.q8_pv_f16 = true;
    cfg.kernels.vulkan.q8_decode_chunk1024 = chunk1024;
    cfg.kernels.vulkan.q8_decode_replay_chunk1024 = chunk1024;
    cfg.kernels.vulkan.q8_decode_combine_sg = combine_sg;
    cfg.kernels.vulkan.q8_decode_gqa2 = true;
    cfg.kernels.vulkan.q8_decode_gqa4 = gqa4;
    let be = VulkanBackend::new_with(Arc::new(cfg)).ok()?;

    let (nh, nkv, hd) = (16usize, 2usize, 256usize);
    let elems = kv_len * nkv * hd;
    let q: Vec<f32> = (0..nh * hd)
        .map(|i| sample(i, 0x1234_5678) * 0.37)
        .collect();
    let kv: Vec<f32> = (0..elems).map(|i| sample(i, 0x89ab_cdef) * 0.53).collect();

    let qb = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &to_f16_bytes(&q)).unwrap();
    let kf = be.alloc(elems * 2, BufferUsage::Activations).unwrap();
    let vf = be.alloc(elems * 2, BufferUsage::Activations).unwrap();
    be.upload(kf.as_ref(), &to_f16_bytes(&kv)).unwrap();
    be.upload(vf.as_ref(), &to_f16_bytes(&kv)).unwrap();

    let cap_rows = kv_len + 37;
    let cap = cap_rows * nkv * hd;
    let cache_bytes = (cap / 32 * 34).next_multiple_of(4);
    let kq = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let vq = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let out = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let chunk = cap_rows.div_ceil(1024).max(64);
    let n_chunks = cap_rows.div_ceil(chunk);
    let pm = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();

    let rec = be.recorder().unwrap();
    rec.store_q8(kf.as_ref(), kq.as_ref(), elems, 0, cap, true, 0);
    rec.store_q8(vf.as_ref(), vq.as_ref(), elems, 0, cap, true, 0);
    let ka = kq.device_addr().expect("KvCache K must expose BDA");
    let va = vq.device_addr().expect("KvCache V must expose BDA");
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(
        params.as_ref(),
        bytemuck::cast_slice(&[(kv_len - 1) as u32, kv_len as u32]),
    )
    .unwrap();
    let args = be.alloc(16, BufferUsage::Activations).unwrap();
    rec.attn_live_prologue(
        params.as_ref(),
        args.as_ref(),
        if gqa4 { nh / 4 } else { nh },
        chunk,
        if chunk1024 { 1024 } else { 512 },
        0,
    );
    rec.attention_kv_split_dynac_at(
        qb.as_ref(),
        kq.as_ref(),
        vq.as_ref(),
        ka,
        va,
        out.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        params.as_ref(),
        args.as_ref(),
        nh,
        nkv,
        hd,
        chunk,
        n_chunks,
        0.0,
        0,
        true,
        gqa4,
        cap,
    );
    rec.finish().unwrap();

    let mut bytes = vec![0u8; nh * hd * 4];
    be.download(out.as_ref(), &mut bytes).unwrap();
    Some(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

fn bench_case(
    label: &str,
    kv_len: usize,
    chunk1024: bool,
    combine_sg: bool,
    qk_i8: bool,
) -> Option<f64> {
    let mut cfg = Config::default();
    cfg.device.dev = Some("Vulkan1".to_string());
    cfg.kernels.vulkan.q8_decode_d8 = true;
    cfg.kernels.vulkan.q8_decode_ls128 = true;
    cfg.kernels.vulkan.q8_decode_ls256 = true;
    cfg.kernels.vulkan.q8_qk_f16 = true;
    cfg.kernels.vulkan.q8_qk_i8 = qk_i8;
    cfg.kernels.vulkan.q8_pv_f16 = true;
    cfg.kernels.vulkan.q8_decode_chunk1024 = chunk1024;
    cfg.kernels.vulkan.q8_decode_replay_chunk1024 = chunk1024;
    cfg.kernels.vulkan.q8_decode_combine_sg = combine_sg;
    cfg.kernels.vulkan.q8_decode_gqa2 = false;
    cfg.kernels.vulkan.q8_decode_gqa4 = false;
    let be = VulkanBackend::new_with(Arc::new(cfg)).ok()?;

    let (nh, nkv, hd) = (16usize, 2usize, 256usize);
    let cap_rows = kv_len + 145;
    let cap = cap_rows * nkv * hd;
    let cache_bytes = (cap / 32 * 34).next_multiple_of(4);
    let q = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    let k = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let v = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let out = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(
        params.as_ref(),
        bytemuck::cast_slice(&[(kv_len - 1) as u32, kv_len as u32]),
    )
    .unwrap();
    let args = be.alloc(16, BufferUsage::Activations).unwrap();
    let chunk = cap_rows.div_ceil(512).max(64);
    let n_chunks = cap_rows.div_ceil(chunk);
    let pm = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pl = be
        .alloc(nh * n_chunks * 4, BufferUsage::Activations)
        .unwrap();
    let pacc = be
        .alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
        .unwrap();
    let ka = k.device_addr().unwrap();
    let va = v.device_addr().unwrap();

    let record = |rec: &infr_vulkan::Recorder| {
        rec.attention_kv_split_dynac_at(
            q.as_ref(),
            k.as_ref(),
            v.as_ref(),
            ka,
            va,
            out.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            params.as_ref(),
            args.as_ref(),
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
            true,
            false,
            cap,
        );
    };

    let rec = be.recorder().unwrap();
    rec.zero(q.as_ref(), q.len_bytes() / 4);
    rec.zero(k.as_ref(), cache_bytes / 4);
    rec.zero(v.as_ref(), cache_bytes / 4);
    rec.attn_live_prologue(
        params.as_ref(),
        args.as_ref(),
        nh,
        chunk,
        if chunk1024 { 1024 } else { 512 },
        0,
    );
    record(&rec);
    rec.finish().unwrap();

    let reps = if kv_len >= 100_000 { 128 } else { 256 };
    let rec = be.recorder().unwrap();
    rec.attn_live_prologue(
        params.as_ref(),
        args.as_ref(),
        nh,
        chunk,
        if chunk1024 { 1024 } else { 512 },
        0,
    );
    for _ in 0..reps {
        record(&rec);
    }
    let t0 = std::time::Instant::now();
    rec.finish().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1.0e6 / reps as f64;
    eprintln!("{label:>20} kv={kv_len}: {us:.2} us/layer");
    Some(us)
}

fn assert_close(reference: &[f32], candidate: &[f32], label: &str) {
    let mut max_err = 0.0f32;
    for (i, (&a, &b)) in reference.iter().zip(candidate).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label} out {i}: {a} vs {b}"
        );
        let err = (a - b).abs();
        max_err = max_err.max(err);
        assert!(err <= 1.0e-3, "{label} out {i}: {a} vs {b}, err={err}");
    }
    eprintln!("{label}: max_err={max_err}");
}

#[test]
fn q8_hd256_gqa4_replay_matches_general_partial() {
    for kv_len in [1usize, 97, 8193] {
        let Some(reference) = run_case(false, false, true, true, kv_len) else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let Some(candidate) = run_case(true, false, true, true, kv_len) else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        assert_close(
            &reference,
            &candidate,
            &format!("replay GQA4 kv_len={kv_len}"),
        );
    }
}

#[test]
fn q8_hd256_i8_qk_replay_tracks_f16_qk() {
    for kv_len in [1usize, 97, 8193, 150_000] {
        let Some(reference) = run_case(false, false, false, true, kv_len) else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let Some(candidate) = run_case(false, true, false, true, kv_len) else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let mut max_err = 0.0f32;
        for (i, (&a, &b)) in reference.iter().zip(&candidate).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "i8 QK out {i}: {a} vs {b}");
            max_err = max_err.max((a - b).abs());
            assert!(
                (a - b).abs() <= 2.0e-2,
                "i8 QK out {i}: {a} vs {b}, err={}",
                (a - b).abs()
            );
        }
        eprintln!("replay i8 QK kv_len={kv_len}: max_err={max_err}");
    }
}

#[test]
fn q8_hd256_chunk1024_and_combine_track_baseline() {
    for kv_len in [97usize, 8193, 30_000, 150_000] {
        let Some(reference) = run_case(false, false, false, false, kv_len) else {
            eprintln!("skip: no Vulkan1 device");
            return;
        };
        let Some(candidate) = run_case(false, false, true, true, kv_len) else {
            eprintln!("skip: no Vulkan1 device");
            return;
        };
        assert_close(
            &reference,
            &candidate,
            &format!("replay chunk1024+combine kv_len={kv_len}"),
        );
    }
}

#[test]
#[ignore = "requires Vulkan1/RX 7900 XTX (perf micro-bench)"]
fn q8_hd256_replay_attention_bench() {
    for kv_len in [30_000usize, 150_000] {
        for (label, chunk1024, combine_sg, qk_i8) in [
            ("baseline", false, false, false),
            ("combine_sg", false, true, false),
            ("chunk1024", true, false, false),
            ("i8_qk+combine", false, true, true),
        ] {
            assert!(
                bench_case(label, kv_len, chunk1024, combine_sg, qk_i8).is_some(),
                "Vulkan1 unavailable"
            );
        }
    }
}
