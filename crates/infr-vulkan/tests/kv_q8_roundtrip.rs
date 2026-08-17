// Isolate the Vulkan planar Q8_0 KV write/read: store an f32 ramp → planar Q8 cache, dequant it back
// to f16, and confirm the round-trip is within Q8 tolerance. Splits store_q8 / dequant_q8_f16 from
// the full attention path so a layout bug is caught directly.
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

#[test]
fn planar_q8_store_dequant_roundtrip() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let n: usize = 256; // 8 blocks of 32 — the region we write + read back
    let cap: usize = 1024; // PADDED cache (cap > n) — scales region base = full cap, like max_ctx
    let cache_bytes = (cap / 32 * 34).next_multiple_of(4);
    // A ramp with varied magnitudes so per-block scales differ.
    let src: Vec<f32> = (0..n).map(|i| ((i as f32) - 128.0) * 0.05).collect();
    let sbuf = be.alloc(n * 4, BufferUsage::Activations).unwrap();
    be.upload(sbuf.as_ref(), bytemuck::cast_slice(&src))
        .unwrap();

    let cache = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let dst = be.alloc(n * 2, BufferUsage::Activations).unwrap(); // f16 out

    let rec = be.recorder().unwrap();
    rec.store_q8(sbuf.as_ref(), cache.as_ref(), n, 0, cap, false, 0);
    rec.dequant_q8_f16(cache.as_ref(), dst.as_ref(), n, cap);
    rec.finish().unwrap();

    let mut out = vec![0u8; n * 2];
    be.download(dst.as_ref(), &mut out).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&out)
        .iter()
        .map(|&h| half::f16::from_bits(h).to_f32())
        .collect();

    let mut max_err = 0.0f32;
    for (i, (&g, &s)) in got.iter().zip(src.iter()).enumerate() {
        let err = (g - s).abs();
        if err > max_err {
            max_err = err;
        }
        // Per-block scale d = amax/127; worst quant error ~ d/2. Block amax ≤ ~6.4 → d ≤ 0.05, so
        // tolerance 0.05 is comfortable.
        assert!(
            err < 0.05,
            "elem {i}: got {g}, want {s}, err {err} (max so far {max_err})"
        );
    }
    eprintln!("planar Q8 roundtrip OK, max_err = {max_err}");
}

/// The record-once dynamic write (store_q8_dyn: off = p_pos*n) must land the same planar bytes as the
/// static store. Writes a row at pos=3 via store_q8_dyn, dequants the prefix, checks that row.
#[test]
fn planar_q8_store_dyn_roundtrip() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let rs: usize = 64; // one KV row = 2 blocks
    let pos: usize = 3;
    let cap: usize = 1024; // padded capacity
    let cache_bytes = (cap / 32 * 34).next_multiple_of(4);
    let row: Vec<f32> = (0..rs).map(|i| ((i as f32) - 32.0) * 0.04).collect();
    let sbuf = be.alloc(rs * 4, BufferUsage::Activations).unwrap();
    be.upload(sbuf.as_ref(), bytemuck::cast_slice(&row))
        .unwrap();
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(
        params.as_ref(),
        bytemuck::cast_slice(&[pos as u32, (pos + 1) as u32]),
    )
    .unwrap();
    let cache = be.alloc(cache_bytes, BufferUsage::KvCache).unwrap();
    let ne = (pos + 1) * rs;
    let dst = be.alloc(ne * 2, BufferUsage::Activations).unwrap();

    let rec = be.recorder().unwrap();
    rec.store_q8_dyn(
        sbuf.as_ref(),
        params.as_ref(),
        cache.as_ref(),
        rs,
        cap,
        false,
    );
    rec.dequant_q8_f16(cache.as_ref(), dst.as_ref(), ne, cap);
    rec.finish().unwrap();

    let mut out = vec![0u8; ne * 2];
    be.download(dst.as_ref(), &mut out).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&out)
        .iter()
        .map(|&h| half::f16::from_bits(h).to_f32())
        .collect();
    // Row `pos` occupies elements [pos*rs, pos*rs+rs).
    for i in 0..rs {
        let g = got[pos * rs + i];
        let err = (g - row[i]).abs();
        assert!(
            err < 0.03,
            "dyn row elem {i}: got {g}, want {}, err {err}",
            row[i]
        );
    }
    eprintln!("store_q8_dyn roundtrip OK");
}

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

/// The planar-Q8 split-K decode kernel (attn_partial_q8) must match the f16 attn_partial on the same
/// logical KV within Q8 tolerance. Isolates the decode read from the full model path.
#[test]
fn planar_q8_attn_partial_matches_f16() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // hd256 exercises the decode-only coupled-Q8 specialization; the mixed and dynac tests below
    // retain hd128 coverage of the general attn_partial variants.
    let (nh, nkv, hd) = (4usize, 2usize, 256usize);
    let kv_len = 96usize;
    let n = kv_len * nkv * hd;
    // Deterministic small values so QK/softmax is well-conditioned.
    let q: Vec<f32> = (0..nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
        .collect();
    let kv: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) * 0.03).collect();

    let qb = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &to_f16_bytes(&q)).unwrap();
    let kf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    let vf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    be.upload(kf.as_ref(), &to_f16_bytes(&kv)).unwrap();
    be.upload(vf.as_ref(), &to_f16_bytes(&kv)).unwrap();

    let chunk = (kv_len / 32).clamp(64, 512);
    let n_chunks = kv_len.div_ceil(chunk);
    let mk = || {
        (
            be.alloc(nh * n_chunks * 4, BufferUsage::Activations)
                .unwrap(),
            be.alloc(nh * n_chunks * 4, BufferUsage::Activations)
                .unwrap(),
            be.alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
                .unwrap(),
        )
    };

    // f16 reference
    let of = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let (pm, pl, pacc) = mk();
    let rec = be.recorder().unwrap();
    rec.attention_kv_split(
        qb.as_ref(),
        kf.as_ref(),
        vf.as_ref(),
        of.as_ref(),
        pm.as_ref(),
        pl.as_ref(),
        pacc.as_ref(),
        1,          // rows (decode shape)
        kv_len - 1, // pos of the single query row
        kv_len,
        nh,
        nkv,
        hd,
        chunk,
        n_chunks,
        0.0,
        0,
        None, // canvas_lo
        false,
        false,
        0,
        false, // batched: decode shape stays on the per-row grid
    );
    rec.finish().unwrap();
    let mut ofb = vec![0u8; nh * hd * 4];
    be.download(of.as_ref(), &mut ofb).unwrap();
    let o_f16: &[f32] = bytemuck::cast_slice(&ofb);

    // planar Q8: quantize the f16 KV into planar Q8 caches, then run the Q8 split kernel. Use a
    // PADDED capacity (cache holds 200 rows, only kv_len written) to mirror the real max_ctx cache —
    // the scales region base = full cap, not the written length.
    let cap = 200 * nkv * hd;
    let cbytes = (cap / 32 * 34).next_multiple_of(4);
    let kq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let vq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let oq = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let (pm2, pl2, pacc2) = mk();
    let rec = be.recorder().unwrap();
    rec.store_q8(kf.as_ref(), kq.as_ref(), n, 0, cap, true, 0);
    rec.store_q8(vf.as_ref(), vq.as_ref(), n, 0, cap, true, 0);
    rec.attention_kv_split(
        qb.as_ref(),
        kq.as_ref(),
        vq.as_ref(),
        oq.as_ref(),
        pm2.as_ref(),
        pl2.as_ref(),
        pacc2.as_ref(),
        1,          // rows (decode shape)
        kv_len - 1, // pos of the single query row
        kv_len,
        nh,
        nkv,
        hd,
        chunk,
        n_chunks,
        0.0,
        0,
        None, // canvas_lo
        true,
        true,
        cap,
        false, // batched: decode shape stays on the per-row grid
    );
    rec.finish().unwrap();
    let mut oqb = vec![0u8; nh * hd * 4];
    be.download(oq.as_ref(), &mut oqb).unwrap();
    let o_q8: &[f32] = bytemuck::cast_slice(&oqb);

    let mut max_err = 0.0f32;
    for (i, (&a, &b)) in o_f16.iter().zip(o_q8.iter()).enumerate() {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(err < 0.02, "out {i}: f16 {a}, q8 {b}, err {err}");
    }
    eprintln!("attn_partial_q8 matches f16, max_err = {max_err}");
}

/// DECOUPLED K/V: mixed caches (K=q8 V=f16, and K=f16 V=q8) via the per-side attn_partial variants
/// must match the all-f16 split within Q8 tolerance. Proves KQ8 and VQ8 are independent.
#[test]
fn planar_q8_attn_partial_mixed_kv() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (nh, nkv, hd) = (4usize, 2usize, 128usize);
    let kv_len = 96usize;
    let n = kv_len * nkv * hd;
    let q: Vec<f32> = (0..nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
        .collect();
    let kv: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) * 0.03).collect();
    let qb = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &to_f16_bytes(&q)).unwrap();
    let kf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    let vf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    be.upload(kf.as_ref(), &to_f16_bytes(&kv)).unwrap();
    be.upload(vf.as_ref(), &to_f16_bytes(&kv)).unwrap();
    let cap = 200 * nkv * hd;
    let cbytes = (cap / 32 * 34).next_multiple_of(4);
    let kq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let vq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let chunk = (kv_len / 32).clamp(64, 512);
    let n_chunks = kv_len.div_ceil(chunk);
    let split = |k: &dyn infr_core::backend::Buffer,
                 v: &dyn infr_core::backend::Buffer,
                 k_q8: bool,
                 v_q8: bool,
                 c: usize|
     -> Vec<f32> {
        let o = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
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
        if k_q8 {
            rec.store_q8(kf.as_ref(), kq.as_ref(), n, 0, cap, true, 0);
        }
        if v_q8 {
            rec.store_q8(vf.as_ref(), vq.as_ref(), n, 0, cap, true, 0);
        }
        rec.attention_kv_split(
            qb.as_ref(),
            k,
            v,
            o.as_ref(),
            pm.as_ref(),
            pl.as_ref(),
            pacc.as_ref(),
            1,          // rows (decode shape)
            kv_len - 1, // pos of the single query row
            kv_len,
            nh,
            nkv,
            hd,
            chunk,
            n_chunks,
            0.0,
            0,
            None, // canvas_lo
            k_q8,
            v_q8,
            c,
            false, // batched: decode shape stays on the per-row grid
        );
        rec.finish().unwrap();
        let mut ob = vec![0u8; nh * hd * 4];
        be.download(o.as_ref(), &mut ob).unwrap();
        bytemuck::cast_slice::<u8, f32>(&ob).to_vec()
    };

    let ref_f16 = split(kf.as_ref(), vf.as_ref(), false, false, 0);
    let mixed_kq8 = split(kq.as_ref(), vf.as_ref(), true, false, cap); // K=q8, V=f16
    let mixed_vq8 = split(kf.as_ref(), vq.as_ref(), false, true, cap); // K=f16, V=q8
    for (name, got) in [("K=q8,V=f16", &mixed_kq8), ("K=f16,V=q8", &mixed_vq8)] {
        let mut max_err = 0.0f32;
        for (i, (&a, &b)) in ref_f16.iter().zip(got.iter()).enumerate() {
            let err = (a - b).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(err < 0.02, "{name} out {i}: f16 {a}, mixed {b}, err {err}");
        }
        eprintln!("{name} matches f16, max_err = {max_err}");
    }
}

/// The record-once self-chunking planar-Q8 decode (attn_partial_dynac_q8) must match the f16 dynac.
#[test]
fn planar_q8_dynac_matches_f16() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (nh, nkv, hd) = (4usize, 2usize, 128usize);
    let kv_len = 96usize;
    let n = kv_len * nkv * hd;
    let q: Vec<f32> = (0..nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
        .collect();
    let kv: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) * 0.03).collect();
    let qb = be.alloc(nh * hd * 2, BufferUsage::Activations).unwrap();
    be.upload(qb.as_ref(), &to_f16_bytes(&q)).unwrap();
    let kf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    let vf = be.alloc(n * 2, BufferUsage::Activations).unwrap();
    be.upload(kf.as_ref(), &to_f16_bytes(&kv)).unwrap();
    be.upload(vf.as_ref(), &to_f16_bytes(&kv)).unwrap();
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(
        params.as_ref(),
        bytemuck::cast_slice(&[kv_len as u32 - 1, kv_len as u32]),
    )
    .unwrap();

    let chunk = 64usize;
    let n_chunks = kv_len.div_ceil(chunk).max(2);
    let mk = || {
        (
            be.alloc(nh * n_chunks * 4, BufferUsage::Activations)
                .unwrap(),
            be.alloc(nh * n_chunks * 4, BufferUsage::Activations)
                .unwrap(),
            be.alloc(nh * n_chunks * hd * 4, BufferUsage::Activations)
                .unwrap(),
            be.alloc(16, BufferUsage::Activations).unwrap(),
        )
    };

    let of = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let (pm, pl, pacc, args) = mk();
    let rec = be.recorder().unwrap();
    rec.attn_live_prologue(params.as_ref(), args.as_ref(), nh, chunk, 0);
    rec.attention_kv_split_dynac(
        qb.as_ref(),
        kf.as_ref(),
        vf.as_ref(),
        of.as_ref(),
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
        false,
        0,
    );
    rec.finish().unwrap();
    let mut ofb = vec![0u8; nh * hd * 4];
    be.download(of.as_ref(), &mut ofb).unwrap();
    let o_f16: &[f32] = bytemuck::cast_slice(&ofb);

    let cap = 200 * nkv * hd;
    let cbytes = (cap / 32 * 34).next_multiple_of(4);
    let kq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let vq = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let oq = be.alloc(nh * hd * 4, BufferUsage::Activations).unwrap();
    let (pm2, pl2, pacc2, args2) = mk();
    let rec = be.recorder().unwrap();
    rec.store_q8(kf.as_ref(), kq.as_ref(), n, 0, cap, true, 0);
    rec.store_q8(vf.as_ref(), vq.as_ref(), n, 0, cap, true, 0);
    rec.attn_live_prologue(params.as_ref(), args2.as_ref(), nh, chunk, 0);
    rec.attention_kv_split_dynac(
        qb.as_ref(),
        kq.as_ref(),
        vq.as_ref(),
        oq.as_ref(),
        pm2.as_ref(),
        pl2.as_ref(),
        pacc2.as_ref(),
        params.as_ref(),
        args2.as_ref(),
        nh,
        nkv,
        hd,
        chunk,
        n_chunks,
        0.0,
        0,
        true,
        cap,
    );
    rec.finish().unwrap();
    let mut oqb = vec![0u8; nh * hd * 4];
    be.download(oq.as_ref(), &mut oqb).unwrap();
    let o_q8: &[f32] = bytemuck::cast_slice(&oqb);

    let mut max_err = 0.0f32;
    for (i, (&a, &b)) in o_f16.iter().zip(o_q8.iter()).enumerate() {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(err < 0.02, "dynac out {i}: f16 {a}, q8 {b}, err {err}");
    }
    eprintln!("attn_partial_dynac_q8 matches f16, max_err = {max_err}");
}

/// The record-once REPLAY of the Q8 K/V writes must land bit-identical planar bytes to the static
/// per-token stores. This is the production decode-tape shape: ONE persistent recording
/// (`global_barrier` → `params_advance` → `store_q8_f16_dyn` (K) → `store_q8_dyn` (V)) whose params
/// self-advance on the DEVICE, replayed once per "token" (host re-uploads the row between replays,
/// exactly like the runner's per-token embed upload), then a chained `replay_n` chunk. The historic
/// "store_q8_dyn mis-decodes under replay" gate turned out to be the Attention lowering passing the
/// ROW capacity as the planar scales base (`cap`); this test pins the write path itself so a real
/// regression can't hide behind that story again.
#[test]
fn planar_q8_store_replay_tape_matches_static() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let rs: usize = 64; // one KV row = 2 blocks
    let cap_rows: usize = 64;
    let cap = cap_rows * rs; // planar scales region base (elements)
    let cbytes = (cap / 32 * 34).next_multiple_of(4);
    let t_single = 7usize; // per-token replays (host upload between)
    let t_chain = 4usize; // one chained replay_n chunk (constant source)
    let row_at = |t: usize| -> Vec<f32> {
        (0..rs)
            .map(|i| ((i as f32) - 30.0) * 0.03 + t as f32 * 0.11)
            .collect()
    };

    // ── replay tape: params seeded [pos0-1, pos0] = [-1, 0]; advance lands row 0 first ──
    let kf = be.alloc(rs * 2, BufferUsage::Activations).unwrap(); // f16 K staging
    let vf = be.alloc(rs * 4, BufferUsage::Activations).unwrap(); // f32 V
    let params = be.alloc(8, BufferUsage::Activations).unwrap();
    be.upload(params.as_ref(), bytemuck::cast_slice(&[u32::MAX, 0u32]))
        .unwrap();
    let kq_dyn = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let vq_dyn = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let rec = be.recorder_persistent().unwrap();
    rec.global_barrier();
    rec.params_advance(params.as_ref());
    rec.store_q8_dyn(kf.as_ref(), params.as_ref(), kq_dyn.as_ref(), rs, cap, true);
    rec.store_q8_dyn(
        vf.as_ref(),
        params.as_ref(),
        vq_dyn.as_ref(),
        rs,
        cap,
        false,
    );
    let tape = rec.finish_record().unwrap();
    for t in 0..t_single {
        let row = row_at(t);
        be.upload(kf.as_ref(), &to_f16_bytes(&row)).unwrap();
        be.upload(vf.as_ref(), bytemuck::cast_slice(&row)).unwrap();
        tape.replay().unwrap();
    }
    // chained chunk: rows t_single..t_single+t_chain, constant source (the device must still
    // advance p_pos per iteration — a stale pos would leave rows unwritten/zero).
    let row = row_at(t_single);
    be.upload(kf.as_ref(), &to_f16_bytes(&row)).unwrap();
    be.upload(vf.as_ref(), bytemuck::cast_slice(&row)).unwrap();
    tape.replay_n(t_chain).unwrap();

    // ── static reference: per-token store_q8 at baked offsets ──
    let kq_st = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    let vq_st = be.alloc(cbytes, BufferUsage::KvCache).unwrap();
    for t in 0..t_single + t_chain {
        let row = row_at(t.min(t_single));
        be.upload(kf.as_ref(), &to_f16_bytes(&row)).unwrap();
        be.upload(vf.as_ref(), bytemuck::cast_slice(&row)).unwrap();
        let rec = be.recorder().unwrap();
        rec.store_q8(kf.as_ref(), kq_st.as_ref(), rs, t * rs, cap, true, 0);
        rec.store_q8(vf.as_ref(), vq_st.as_ref(), rs, t * rs, cap, false, 0);
        rec.finish().unwrap();
    }

    // ── compare: code bytes of the written rows + their scale bytes, bit-exact ──
    let rows = t_single + t_chain;
    let mut bufs = vec![vec![0u8; cbytes]; 4];
    for (b, buf) in [&kq_dyn, &vq_dyn, &kq_st, &vq_st]
        .iter()
        .zip(bufs.iter_mut())
    {
        be.download(b.as_ref(), buf).unwrap();
    }
    let (kd, vd, ks, vs) = (&bufs[0], &bufs[1], &bufs[2], &bufs[3]);
    assert_eq!(
        &kd[..rows * rs],
        &ks[..rows * rs],
        "replayed K codes differ from static"
    );
    assert_eq!(
        &vd[..rows * rs],
        &vs[..rows * rs],
        "replayed V codes differ from static"
    );
    let (s0, s1) = (cap, cap + rows * rs / 32 * 2);
    assert_eq!(&kd[s0..s1], &ks[s0..s1], "replayed K scales differ");
    assert_eq!(&vd[s0..s1], &vs[s0..s1], "replayed V scales differ");
    // and the device-side position really advanced once per replay
    let mut p = [0u8; 8];
    be.download(params.as_ref(), &mut p).unwrap();
    let p: &[u32] = bytemuck::cast_slice(&p);
    assert_eq!(p[0] as usize, rows - 1, "params[0] after {rows} replays");
    eprintln!("replayed Q8 store tape bit-matches static ({rows} rows)");
}
