//! Exact cache-byte parity for decode QkNorm+RoPE -> planar-Q8 K-write fusion.

use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

fn sample(i: usize, seed: u32) -> f32 {
    let mut x = (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(seed);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    ((x & 0xffff) as f32 / 32767.5) - 1.0
}

#[test]
fn fused_q8_k_write_matches_f16_then_store_q8() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (nh, hd, rope_dim, cap_rows, pos) = (2usize, 256usize, 64usize, 128usize, 37u32);
    let n = nh * hd;
    let cap = cap_rows * n;
    let cache_bytes = (cap / 32 * 34).next_multiple_of(4);

    let x = (0..n).map(|i| sample(i, 0x1234_5678)).collect::<Vec<_>>();
    let w = (0..hd)
        .map(|i| 0.8 + 0.4 * sample(i, 0x89ab_cdef))
        .collect::<Vec<_>>();
    let xb = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    let wb = be.alloc(hd * 4, BufferUsage::Weights).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    be.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(params.as_ref(), bytemuck::cast_slice(&[pos, pos + 1]))
        .unwrap();
    let tmp = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    let reference = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let candidate = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();

    let rec = be.recorder().unwrap();
    rec.zero(reference.as_ref(), cache_bytes / 4);
    rec.zero(candidate.as_ref(), cache_bytes / 4);
    rec.qk_norm_rope_dyn(
        xb.as_ref(),
        wb.as_ref(),
        params.as_ref(),
        None,
        tmp.as_ref(),
        1,
        nh,
        hd,
        rope_dim,
        1_000_000.0,
        0,
        1.0e-6,
        0,
    );
    rec.store_q8_dyn(
        tmp.as_ref(),
        params.as_ref(),
        reference.as_ref(),
        n,
        cap,
        true,
    );
    rec.qk_norm_rope_q8_dyn_at(
        xb.as_ref(),
        wb.as_ref(),
        params.as_ref(),
        candidate.as_ref(),
        candidate.device_addr().unwrap(),
        1,
        nh,
        hd,
        rope_dim,
        1_000_000.0,
        1.0e-6,
        cap_rows,
    );
    rec.finish().unwrap();

    let mut a = vec![0u8; cache_bytes];
    let mut b = vec![0u8; cache_bytes];
    be.download(reference.as_ref(), &mut a).unwrap();
    be.download(candidate.as_ref(), &mut b).unwrap();
    let diffs = a
        .iter()
        .zip(&b)
        .enumerate()
        .filter_map(|(i, (&lhs, &rhs))| (lhs != rhs).then_some((i, lhs, rhs)))
        .collect::<Vec<_>>();
    let code_diffs = diffs.iter().filter(|(i, _, _)| *i < cap).count();
    let scale_diffs = diffs.len() - code_diffs;
    assert!(
        diffs.is_empty(),
        "{} cache-byte differences (codes={}, scales={}); first={:?}",
        diffs.len(),
        code_diffs,
        scale_diffs,
        &diffs[..diffs.len().min(32)]
    );
}
