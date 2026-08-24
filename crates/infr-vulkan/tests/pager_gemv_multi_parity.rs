//! Proves `linear_native_id_multi_paged` (the multi-slot paged GEMV `execute_paged_moe`'s gate/up/
//! down dispatches use) is correct WHEN CHAINED with a dependent dispatch in the SAME recording —
//! not just in isolation. A single-dispatch-then-`finish()` test (`pager_gemv_parity.rs`) can't
//! catch a hazard-tracking bug: `Recorder::finish`'s `queue_wait_idle` makes any dispatch's output
//! visible by the time you download it, whether or not a barrier was recorded between it and a
//! LATER dependent dispatch in the same command buffer. This test instead mirrors
//! `execute_paged_moe`'s real shape — gate GEMV, up GEMV, then `silu_mul` reading both outputs, ALL
//! in one `Recorder` — which is exactly the sequence that caught a real bug during this task: the
//! paged multi-slot kernels bound their LUT buffer AFTER `y` (`[w, x, ids, y, lut]`), so
//! `Recorder::dispatch3`'s hazard tracking (last `n_out` buffers = writes, everything before =
//! reads — one contiguous split) mis-classified `y` as a READ and `lut` as the WRITE, silently
//! dropping the barrier `silu_mul` needed before reading `y` — `abuf` came back with a handful of
//! ~1e37-magnitude garbage values despite `gbuf`/`ubuf` (its own inputs) being completely clean.
//! Fixed by binding LUT ahead of Y in both the shaders (`native_gemv_id(_multi).comp`) and the
//! Rust wrapper (`Recorder::linear_native_id_(multi_)paged`) — Y is now always the last binding.
//!
//! Run: `cargo test -p infr-vulkan --test pager_gemv_multi_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::pager::GpuPager;
use infr_vulkan::VulkanBackend;

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
fn host_gemv(w_dequant: &[f32], x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    (0..out_f)
        .map(|o| {
            (0..in_f)
                .map(|i| w_dequant[o * in_f + i] * x[i])
                .sum::<f32>()
        })
        .collect()
}
fn host_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn multi_paged_gemv_chained_in_one_recorder_matches_host() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (in_f, out_f, n_expert) = (64usize, 4usize, 5usize);
    let stride_elems = in_f * out_f;
    let stride_bytes = stride_elems / 32 * 34;

    // Two independent synthetic banks (gate, up) + two independent pagers, exactly like the
    // seam's per-role GpuPager split.
    let make_banks = |seed: usize| -> Vec<Vec<u8>> {
        (0..n_expert)
            .map(|e| {
                let w: Vec<f32> = (0..stride_elems)
                    .map(|i| (((i * 7 + e * 13 + seed) % 23) as f32 - 11.0) * 0.1)
                    .collect();
                q8_0(&w)
            })
            .collect()
    };
    let gate_banks = make_banks(0);
    let up_banks = make_banks(101);
    let gate_host: Vec<Vec<f32>> = gate_banks.iter().map(|b| deq_q8(b)).collect();
    let up_host: Vec<Vec<f32>> = up_banks.iter().map(|b| deq_q8(b)).collect();

    let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let x_buf = be.alloc(in_f * 4, BufferUsage::Activations).unwrap();
    be.upload(x_buf.as_ref(), bytemuck::cast_slice(&x)).unwrap();

    let mut gate_pager = GpuPager::new(&be, n_expert, n_expert, stride_bytes).unwrap();
    let mut up_pager = GpuPager::new(&be, n_expert, n_expert, stride_bytes).unwrap();
    let staging = be.alloc_uninit(stride_bytes, BufferUsage::Staging).unwrap();

    let n_used = 3usize;
    let ids: Vec<u32> = vec![0, 2, 4];
    for &eid in &ids {
        gate_pager
            .ensure_resident(&be, staging.as_ref(), eid, &gate_banks[eid as usize])
            .unwrap();
        up_pager
            .ensure_resident(&be, staging.as_ref(), eid, &up_banks[eid as usize])
            .unwrap();
    }
    gate_pager.flush_lut(&be).unwrap();
    up_pager.flush_lut(&be).unwrap();

    let ids_buf = be.alloc(n_used * 4, BufferUsage::Activations).unwrap();
    be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&ids))
        .unwrap();
    let gbuf = be
        .alloc(n_used * out_f * 4, BufferUsage::Activations)
        .unwrap();
    let ubuf = be
        .alloc(n_used * out_f * 4, BufferUsage::Activations)
        .unwrap();
    let abuf = be
        .alloc(n_used * out_f * 4, BufferUsage::Activations)
        .unwrap();

    // ONE recorder: gate GEMV, up GEMV, silu_mul(gate,up)->abuf — the exact chained shape
    // `execute_paged_moe` uses. A missing barrier between either GEMV's write and silu_mul's read
    // would leave `abuf` with stale/garbage values despite `gbuf`/`ubuf` looking fine on their own.
    let rec = be.recorder().unwrap();
    // `lut_base = 0`: this synthetic single-layer bank's local ids ARE its LUT indices (the
    // adapter passes a frozen tape-window base here — `lut[base + local_id]`).
    rec.linear_native_id_multi_paged(
        DType::Q8_0,
        gate_pager.arena_addr(),
        gate_pager.slot_bytes() as u32,
        gate_pager.lut_buffer(),
        ids_buf.as_ref(),
        n_used,
        0,
        x_buf.as_ref(),
        false,
        gbuf.as_ref(),
        in_f,
        out_f,
        1,
        u32::MAX,
    );
    rec.linear_native_id_multi_paged(
        DType::Q8_0,
        up_pager.arena_addr(),
        up_pager.slot_bytes() as u32,
        up_pager.lut_buffer(),
        ids_buf.as_ref(),
        n_used,
        0,
        x_buf.as_ref(),
        false,
        ubuf.as_ref(),
        in_f,
        out_f,
        1,
        u32::MAX,
    );
    rec.silu_mul(
        gbuf.as_ref(),
        ubuf.as_ref(),
        abuf.as_ref(),
        n_used * out_f,
        None,
    );
    rec.finish().unwrap();

    let mut out = vec![0u8; n_used * out_f * 4];
    be.download(abuf.as_ref(), &mut out).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
    for (slot, &eid) in ids.iter().enumerate() {
        let want_g = host_gemv(&gate_host[eid as usize], &x, in_f, out_f);
        let want_u = host_gemv(&up_host[eid as usize], &x, in_f, out_f);
        for o in 0..out_f {
            let want = host_silu(want_g[o]) * want_u[o];
            let g = got[slot * out_f + o];
            assert!(
                (g - want).abs() < 1e-2,
                "slot {slot} expert {eid} out {o}: chained paged silu {g} != host reference {want} \
                 (got={got:?})",
            );
        }
    }
    println!("chained multi paged gemv OK: {got:?}");
}
