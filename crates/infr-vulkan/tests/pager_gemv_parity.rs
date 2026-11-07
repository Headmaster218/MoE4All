//! Proves the MoE expert pager's LUT indirection (`infr_vulkan::pager::GpuPager` +
//! `native_gemv_id_*_paged` shaders, `w_addr = arena_base + uint64_t(lut[expert_id]) *
//! slot_bytes` slot-index addressing) computes the SAME GEMV a host
//! reference does, across a token sequence that churns the cache (more distinct experts than
//! slots — every step but the first two is a miss that evicts the LRU resident expert). This is
//! the "pages a synthetic bank through a small arena and checks GEMV results match a host
//! reference" test the pager task's validation step asks for.
//!
//! Run: `cargo test -p infr-vulkan --test pager_gemv_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::pager::GpuPager;
use infr_vulkan::VulkanBackend;

// Q8_0: 32-elem blocks, f16 scale + 32 int8 = 34 bytes/block.
fn q8_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in blk {
            out.push(((v * id).round().clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}
fn deq_q8(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 34 * 32);
    for blk in bytes.chunks(34) {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for i in 0..32 {
            out.push((blk[2 + i] as i8 as f32) * d);
        }
    }
    out
}

/// Host GEMV reference: `y[o] = sum_i x[i] * W[o*in_f + i]` against a dequanted expert's weights.
fn host_gemv(w_dequant: &[f32], x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    (0..out_f)
        .map(|o| {
            (0..in_f)
                .map(|i| w_dequant[o * in_f + i] * x[i])
                .sum::<f32>()
        })
        .collect()
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn paged_gemv_matches_host_reference_under_eviction_churn() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };

    let (in_f, out_f, n_expert) = (64usize, 4usize, 5usize);
    let stride_elems = in_f * out_f;
    let stride_bytes = stride_elems / 32 * 34;

    // One synthetic expert bank per id: distinct pseudo-random weights so a wrong-expert read
    // (stale LUT, mis-addressed slot, ...) is detectable, not accidentally equal.
    let banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| {
            let w: Vec<f32> = (0..stride_elems)
                .map(|i| (((i * 7 + e * 13) % 23) as f32 - 11.0) * 0.1)
                .collect();
            q8_0(&w)
        })
        .collect();
    let host_w: Vec<Vec<f32>> = banks.iter().map(|b| deq_q8(b)).collect();

    let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    // 2 slots for 5 experts: every step past the first two is an eviction.
    let mut pager = GpuPager::new(&be, n_expert, 2, stride_bytes).unwrap();
    let staging = be.alloc_uninit(stride_bytes, BufferUsage::Staging).unwrap();

    let ids_buf = be.alloc(4, BufferUsage::Activations).unwrap();
    let y_buf = be.alloc(out_f * 4, BufferUsage::Activations).unwrap();

    let seq = [0u32, 1, 2, 3, 4, 0, 2, 4, 1, 3, 0, 0, 4];
    for &eid in &seq {
        pager
            .ensure_resident(&be, staging.as_ref(), eid, &banks[eid as usize])
            .unwrap();
        pager.flush_lut(&be).unwrap();

        // `ids` holds the single active expert id at slot 0 — mirrors the real decode dispatch's
        // shape (device-computed top-k ids), just with one entry for this isolation test.
        be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&[eid]))
            .unwrap();

        let rec = be.recorder().unwrap();
        // `lut_base = 0`: this synthetic single-layer bank's local ids ARE its LUT indices.
        rec.linear_native_id_paged(
            DType::Q8_0,
            pager.arena_addr(),
            pager.slot_bytes() as u32,
            pager.lut_buffer(),
            ids_buf.as_ref(),
            0,
            0,
            x_buf.as_ref(),
            y_buf.as_ref(),
            1,
            in_f,
            out_f,
        );
        rec.finish().unwrap();

        let mut out = vec![0u8; out_f * 4];
        be.download(y_buf.as_ref(), &mut out).unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
        let want = host_gemv(&host_w[eid as usize], &x, in_f, out_f);
        for o in 0..out_f {
            assert!(
                (got[o] - want[o]).abs() < 1e-3,
                "expert {eid} out {o}: paged GEMV {} != host reference {} (seq step for id {eid})",
                got[o],
                want[o]
            );
        }
    }
    let stats = pager.stats();
    println!(
        "pager stats: hits={} misses={} evictions={} hit_rate={:.2}",
        stats.hits,
        stats.misses,
        stats.evictions,
        stats.hit_rate()
    );
    assert!(stats.evictions > 0, "test must actually exercise eviction");
}
