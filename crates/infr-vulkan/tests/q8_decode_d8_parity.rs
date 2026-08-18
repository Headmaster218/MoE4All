//! Direct parity coverage for the hd256 planar-Q8 decode specializations.
//!
//! The shipping D8 kernel changes only the Q.K reduction geometry.  Keep the historical D32
//! kernel compiled and compare both paths on a single-key tail, a ragged chunk, and a deeper
//! context so the runtime fallback remains a trustworthy A/B and compatibility path.

use std::sync::Arc;

use infr_core::{
    backend::{Backend, BufferUsage},
    config::Config,
};
use infr_vulkan::VulkanBackend;

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn backend(d8: bool, ls128: bool) -> Option<VulkanBackend> {
    let mut cfg = Config::default();
    cfg.kernels.vulkan.q8_decode_d8 = d8;
    cfg.kernels.vulkan.q8_decode_ls128 = ls128;
    VulkanBackend::new_with(Arc::new(cfg)).ok()
}

fn run_case(d8: bool, ls128: bool, kv_len: usize) -> Option<Vec<f32>> {
    let be = backend(d8, ls128)?;
    let (nh, nkv, hd) = (4usize, 2usize, 256usize);
    let n = kv_len * nkv * hd;
    let q: Vec<f32> = (0..nh * hd)
        .map(|i| (((i * 13 + 7) % 37) as f32 - 18.0) * 0.013)
        .collect();
    let kv: Vec<f32> = (0..n)
        .map(|i| (((i * 17 + 11) % 53) as f32 - 26.0) * 0.009)
        .collect();

    let qb = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &to_f16_bytes(&q)).unwrap();
    let kf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    let vf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    be.upload(kf.as_ref(), &to_f16_bytes(&kv)).unwrap();
    be.upload(vf.as_ref(), &to_f16_bytes(&kv)).unwrap();

    // Deliberately pad the cache so the planar scale base differs from the written length.
    let cap_rows = kv_len + 37;
    let cap = cap_rows * nkv * hd;
    let cbytes = (cap / 32 * 34).next_multiple_of(4);
    let kq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let vq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let out = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();

    let chunk = (kv_len / 32).clamp(64, 512);
    let n_chunks = kv_len.div_ceil(chunk);
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
    rec.store_q8(kf.as_ref(), kq.as_ref(), n, 0, cap, true, 0);
    rec.store_q8(vf.as_ref(), vq.as_ref(), n, 0, cap, true, 0);
    rec.attention_kv_split(
        qb.as_ref(),
        kq.as_ref(),
        vq.as_ref(),
        out.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        1,
        kv_len - 1,
        kv_len,
        nh,
        nkv,
        hd,
        chunk,
        n_chunks,
        0.0,
        0,
        None,
        true,
        true,
        cap,
        false,
    );
    rec.finish().unwrap();

    let mut bytes = vec![0u8; nh * hd * 4];
    be.download(out.as_ref(), &mut bytes).unwrap();
    Some(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

#[test]
fn q8_hd256_d8_matches_d32_for_tail_ragged_and_deep_contexts() {
    for kv_len in [1usize, 97, 8193] {
        let Some(d32) = run_case(false, false, kv_len) else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let Some(d8_ls64) = run_case(true, false, kv_len) else {
            eprintln!("skip: no Vulkan device");
            return;
        };
        let Some(d8_ls128) = run_case(true, true, kv_len) else {
            eprintln!("skip: no Vulkan device");
            return;
        };

        for (name, candidate) in [("D8-LS64", d8_ls64), ("D8-LS128", d8_ls128)] {
            let mut max_err = 0.0f32;
            for (i, (&reference, &clustered)) in d32.iter().zip(&candidate).enumerate() {
                assert!(
                    reference.is_finite() && clustered.is_finite(),
                    "kv_len={kv_len} out {i}: D32={reference}, {name}={clustered}"
                );
                let err = (reference - clustered).abs();
                max_err = max_err.max(err);
                assert!(
                    err <= 5.0e-4,
                    "kv_len={kv_len} out {i}: D32={reference}, {name}={clustered}, err={err}"
                );
            }
            eprintln!("Q8 hd256 D32/{name} parity: kv_len={kv_len}, max_err={max_err}");
        }
    }
}
