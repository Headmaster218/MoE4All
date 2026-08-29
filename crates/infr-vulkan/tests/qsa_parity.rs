use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

fn h(v: &[u8], i: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([v[2 * i], v[2 * i + 1]])).to_f32()
}

#[test]
fn qsa_index_and_gather_match_reference() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (blocks, ratio, hd, nh, rope_dim, top) =
        (9usize, 4usize, 128usize, 4usize, 64usize, 3usize);
    let kv_len = blocks * ratio + 2;
    let theta = 10_000.0f32;
    let eps = 1e-6f32;
    let scale = 1.0 / (hd as f32).sqrt();
    let qv: Vec<f32> = (0..nh * hd)
        .map(|i| ((i * 37 + 11) % 101) as f32 / 50.0 - 1.0)
        .collect();
    let rawv: Vec<f32> = (0..kv_len * hd)
        .map(|i| ((i * 53 + 7) % 127) as f32 / 63.0 - 1.0)
        .collect();
    let norm: Vec<f32> = (0..hd).map(|i| 0.75 + (i % 17) as f32 * 0.01).collect();
    let qb = f16_bytes(&qv);
    let rawb = f16_bytes(&rawv);

    let mut scores_ref = vec![0.0f32; blocks];
    let mut key = vec![0.0f32; hd];
    for block in 0..blocks {
        for d in 0..hd {
            key[d] = (0..ratio)
                .map(|r| h(&rawb, (block * ratio + r) * hd + d))
                .sum::<f32>()
                / ratio as f32;
            key[d] = half::f16::from_f32(key[d]).to_f32();
        }
        let inv = (key.iter().map(|v| v * v).sum::<f32>() / hd as f32 + eps)
            .sqrt()
            .recip();
        for d in 0..hd {
            key[d] *= inv * norm[d];
        }
        for pair in 0..rope_dim / 2 {
            let d = 2 * pair;
            let angle = (block * ratio) as f32 * theta.powf(-((2 * pair) as f32) / rope_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (key[d], key[d + 1]);
            key[d] = a * cos - b * sin;
            key[d + 1] = a * sin + b * cos;
        }
        for head in 0..nh {
            let dot = (0..hd).map(|d| h(&qb, head * hd + d) * key[d]).sum::<f32>();
            scores_ref[block] += dot.max(0.0) * scale;
        }
    }
    let mut rank: Vec<usize> = (0..blocks).collect();
    rank.sort_unstable_by(|&a, &b| {
        scores_ref[b]
            .total_cmp(&scores_ref[a])
            .then_with(|| a.cmp(&b))
    });
    rank.truncate(top);
    rank.sort_unstable();

    let q = be.alloc(qb.len(), BufferUsage::Activations).unwrap();
    let raw = be.alloc(rawb.len(), BufferUsage::KvCache).unwrap();
    let block_keys = be.alloc(blocks * hd * 4, BufferUsage::KvCache).unwrap();
    let nw = be.alloc(norm.len() * 4, BufferUsage::Weights).unwrap();
    let scores = be.alloc(blocks * 4, BufferUsage::Activations).unwrap();
    let topk_work = be
        .alloc((64 * 256 + 2) * 4, BufferUsage::Activations)
        .unwrap();
    let ids = be.alloc(top * 4, BufferUsage::Activations).unwrap();
    be.upload(q.as_ref(), &qb).unwrap();
    be.upload(raw.as_ref(), &rawb).unwrap();
    be.upload(nw.as_ref(), bytemuck::cast_slice(&norm)).unwrap();

    let row = 32usize;
    let kvals: Vec<f32> = (0..kv_len * row).map(|i| i as f32 * 0.001).collect();
    let vvals: Vec<f32> = (0..kv_len * row).map(|i| -1.0 - i as f32 * 0.001).collect();
    let kb = f16_bytes(&kvals);
    let vb = f16_bytes(&vvals);
    let k = be.alloc(kb.len(), BufferUsage::KvCache).unwrap();
    let v = be.alloc(vb.len(), BufferUsage::KvCache).unwrap();
    be.upload(k.as_ref(), &kb).unwrap();
    be.upload(v.as_ref(), &vb).unwrap();
    let out_rows = top * ratio + 2;
    let kd = be
        .alloc(out_rows * row * 2, BufferUsage::Activations)
        .unwrap();
    let vd = be
        .alloc(out_rows * row * 2, BufferUsage::Activations)
        .unwrap();

    let rec = be.recorder().unwrap();
    rec.qsa_indexer(
        q.as_ref(),
        raw.as_ref(),
        block_keys.as_ref(),
        nw.as_ref(),
        scores.as_ref(),
        Some(topk_work.as_ref()),
        ids.as_ref(),
        1,
        kv_len as u32,
        0,
        nh as u32,
        hd as u32,
        top as u32,
        ratio as u32,
        rope_dim as u32,
        theta,
        eps,
        scale,
        None,
    );
    rec.qsa_gather(
        k.as_ref(),
        v.as_ref(),
        ids.as_ref(),
        kd.as_ref(),
        vd.as_ref(),
        top as u32,
        blocks as u32,
        2,
        ratio as u32,
        row as u32,
        false,
        false,
        0,
        0,
        None,
    );
    rec.finish().unwrap();

    let mut sb = vec![0u8; blocks * 4];
    let mut ib = vec![0u8; top * 4];
    be.download(scores.as_ref(), &mut sb).unwrap();
    be.download(ids.as_ref(), &mut ib).unwrap();
    let got_scores = bytemuck::cast_slice::<u8, f32>(&sb);
    let got_ids = bytemuck::cast_slice::<u8, u32>(&ib);
    assert_eq!(got_ids, rank.iter().map(|&i| i as u32).collect::<Vec<_>>());
    for (i, (&got, &want)) in got_scores.iter().zip(&scores_ref).enumerate() {
        assert!(
            (got - want).abs() < 2e-3,
            "score {i}: got {got}, want {want}"
        );
    }

    let mut got_k = vec![0u8; out_rows * row * 2];
    let mut got_v = vec![0u8; out_rows * row * 2];
    be.download(kd.as_ref(), &mut got_k).unwrap();
    be.download(vd.as_ref(), &mut got_v).unwrap();
    let mut want_rows = Vec::new();
    for &block in &rank {
        want_rows.extend(block * ratio..block * ratio + ratio);
    }
    want_rows.extend(blocks * ratio..blocks * ratio + 2);
    let mut want_k = Vec::new();
    let mut want_v = Vec::new();
    for src in want_rows {
        want_k.extend_from_slice(&kb[src * row * 2..(src + 1) * row * 2]);
        want_v.extend_from_slice(&vb[src * row * 2..(src + 1) * row * 2]);
    }
    assert_eq!(got_k, want_k);
    assert_eq!(got_v, want_v);

    let cap = kv_len * row;
    let q8_bytes = (cap / 32 * 34).next_multiple_of(4);
    let kq = be.alloc(q8_bytes, BufferUsage::KvCache).unwrap();
    let vq = be.alloc(q8_bytes, BufferUsage::KvCache).unwrap();
    let rec = be.recorder().unwrap();
    rec.store_q8(k.as_ref(), kq.as_ref(), cap, 0, cap, true, 0);
    rec.store_q8(v.as_ref(), vq.as_ref(), cap, 0, cap, true, 0);
    rec.qsa_gather(
        kq.as_ref(),
        vq.as_ref(),
        ids.as_ref(),
        kd.as_ref(),
        vd.as_ref(),
        top as u32,
        blocks as u32,
        2,
        ratio as u32,
        row as u32,
        true,
        true,
        cap as u32,
        cap as u32,
        None,
    );
    rec.finish().unwrap();
    be.download(kd.as_ref(), &mut got_k).unwrap();
    be.download(vd.as_ref(), &mut got_v).unwrap();
    for (name, got, want) in [("K", &got_k, &want_k), ("V", &got_v, &want_v)] {
        for i in 0..out_rows * row {
            let err = (h(got, i) - h(want, i)).abs();
            assert!(err < 0.03, "Q8 {name} gather elem {i}: err {err}");
        }
    }

    // Equal scores exercise the secondary key: the earliest block indices must win exactly as
    // they did in the repeated-max implementation.
    be.upload(q.as_ref(), &vec![0; qb.len()]).unwrap();
    be.upload(raw.as_ref(), &vec![0; rawb.len()]).unwrap();
    let rec = be.recorder().unwrap();
    rec.qsa_indexer(
        q.as_ref(),
        raw.as_ref(),
        block_keys.as_ref(),
        nw.as_ref(),
        scores.as_ref(),
        Some(topk_work.as_ref()),
        ids.as_ref(),
        1,
        kv_len as u32,
        0,
        nh as u32,
        hd as u32,
        top as u32,
        ratio as u32,
        rope_dim as u32,
        theta,
        eps,
        scale,
        None,
    );
    rec.finish().unwrap();
    be.download(ids.as_ref(), &mut ib).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, u32>(&ib),
        &(0..top as u32).collect::<Vec<_>>()
    );
}

#[test]
fn qsa_batched_rows_match_causal_reference() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let (rows, ratio, index_hd, index_heads, rope_dim, top) =
        (3usize, 4usize, 128usize, 4usize, 64usize, 3usize);
    let (n_head, n_kv, attn_hd) = (4usize, 2usize, 256usize);
    let kv_len = 26usize;
    let max_blocks = kv_len / ratio;
    let theta = 10_000.0f32;
    let eps = 1e-6f32;
    let index_scale = 1.0 / (index_hd as f32).sqrt();
    let attn_scale = 1.0 / (attn_hd as f32).sqrt();

    let index_qv: Vec<f32> = (0..rows * index_heads * index_hd)
        .map(|i| ((i * 37 + 11) % 101) as f32 / 65.0 - 0.75)
        .collect();
    let rawv: Vec<f32> = (0..kv_len * index_hd)
        .map(|i| ((i * 53 + 7) % 127) as f32 / 75.0 - 0.8)
        .collect();
    let norm: Vec<f32> = (0..index_hd)
        .map(|i| 0.75 + (i % 17) as f32 * 0.01)
        .collect();
    let index_qb = f16_bytes(&index_qv);
    let rawb = f16_bytes(&rawv);

    let mut want_ids = Vec::with_capacity(rows * top);
    let mut key = vec![0.0f32; index_hd];
    for row in 0..rows {
        let visible = kv_len - rows + row + 1;
        let blocks = visible / ratio;
        let mut scores = vec![0.0f32; blocks];
        for (block, score) in scores.iter_mut().enumerate() {
            for d in 0..index_hd {
                key[d] = (0..ratio)
                    .map(|r| h(&rawb, (block * ratio + r) * index_hd + d))
                    .sum::<f32>()
                    / ratio as f32;
                key[d] = half::f16::from_f32(key[d]).to_f32();
            }
            let inv = (key.iter().map(|v| v * v).sum::<f32>() / index_hd as f32 + eps)
                .sqrt()
                .recip();
            for d in 0..index_hd {
                key[d] *= inv * norm[d];
            }
            for pair in 0..rope_dim / 2 {
                let d = 2 * pair;
                let angle =
                    (block * ratio) as f32 * theta.powf(-((2 * pair) as f32) / rope_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let (a, b) = (key[d], key[d + 1]);
                key[d] = a * cos - b * sin;
                key[d + 1] = a * sin + b * cos;
            }
            let qbase = row * index_heads * index_hd;
            for head in 0..index_heads {
                let dot = (0..index_hd)
                    .map(|d| h(&index_qb, qbase + head * index_hd + d) * key[d])
                    .sum::<f32>();
                *score += dot.max(0.0) * index_scale;
            }
        }
        let mut rank: Vec<usize> = (0..blocks).collect();
        rank.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
        rank.truncate(top);
        rank.sort_unstable();
        want_ids.extend(rank.into_iter().map(|i| i as u32));
    }

    let attn_qv: Vec<f32> = (0..rows * n_head * attn_hd)
        .map(|i| ((i * 29 + 19) % 113) as f32 / 90.0 - 0.6)
        .collect();
    let kvals: Vec<f32> = (0..kv_len * n_kv * attn_hd)
        .map(|i| ((i * 43 + 5) % 109) as f32 / 85.0 - 0.65)
        .collect();
    let vvals: Vec<f32> = (0..kv_len * n_kv * attn_hd)
        .map(|i| ((i * 31 + 23) % 103) as f32 / 80.0 - 0.6)
        .collect();
    let attn_qb = f16_bytes(&attn_qv);
    let kb = f16_bytes(&kvals);
    let vb = f16_bytes(&vvals);

    let mut want_out = vec![0.0f32; rows * n_head * attn_hd];
    for row in 0..rows {
        let visible = kv_len - rows + row + 1;
        let complete = visible / ratio;
        let tail = visible % ratio;
        let mut source_rows = Vec::with_capacity(top * ratio + tail);
        for &block in &want_ids[row * top..(row + 1) * top] {
            source_rows.extend(block as usize * ratio..(block as usize + 1) * ratio);
        }
        source_rows.extend(complete * ratio..complete * ratio + tail);
        for head in 0..n_head {
            let kv_head = head / (n_head / n_kv);
            let qbase = (row * n_head + head) * attn_hd;
            let mut logits = Vec::with_capacity(source_rows.len());
            for &src_row in &source_rows {
                let kbase = (src_row * n_kv + kv_head) * attn_hd;
                logits.push(
                    (0..attn_hd)
                        .map(|d| h(&attn_qb, qbase + d) * h(&kb, kbase + d))
                        .sum::<f32>()
                        * attn_scale,
                );
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom = logits.iter().map(|x| (x - max).exp()).sum::<f32>();
            for (&src_row, logit) in source_rows.iter().zip(logits) {
                let p = (logit - max).exp() / denom;
                let vbase = (src_row * n_kv + kv_head) * attn_hd;
                for d in 0..attn_hd {
                    want_out[qbase + d] += p * h(&vb, vbase + d);
                }
            }
        }
    }

    let index_q = be.alloc(index_qb.len(), BufferUsage::Activations).unwrap();
    let raw = be.alloc(rawb.len(), BufferUsage::KvCache).unwrap();
    let block_keys = be
        .alloc(max_blocks * index_hd * 4, BufferUsage::KvCache)
        .unwrap();
    let nw = be.alloc(norm.len() * 4, BufferUsage::Weights).unwrap();
    let scores = be
        .alloc(rows * max_blocks * 4, BufferUsage::Activations)
        .unwrap();
    let ids = be.alloc(rows * top * 4, BufferUsage::Activations).unwrap();
    let attn_q = be.alloc(attn_qb.len(), BufferUsage::Activations).unwrap();
    let k = be.alloc(kb.len(), BufferUsage::KvCache).unwrap();
    let v = be.alloc(vb.len(), BufferUsage::KvCache).unwrap();
    let out = be
        .alloc(want_out.len() * 4, BufferUsage::Activations)
        .unwrap();
    be.upload(index_q.as_ref(), &index_qb).unwrap();
    be.upload(raw.as_ref(), &rawb).unwrap();
    be.upload(nw.as_ref(), bytemuck::cast_slice(&norm)).unwrap();
    be.upload(attn_q.as_ref(), &attn_qb).unwrap();
    be.upload(k.as_ref(), &kb).unwrap();
    be.upload(v.as_ref(), &vb).unwrap();

    let rec = be.recorder().unwrap();
    rec.qsa_indexer(
        index_q.as_ref(),
        raw.as_ref(),
        block_keys.as_ref(),
        nw.as_ref(),
        scores.as_ref(),
        None,
        ids.as_ref(),
        rows as u32,
        kv_len as u32,
        0,
        index_heads as u32,
        index_hd as u32,
        top as u32,
        ratio as u32,
        rope_dim as u32,
        theta,
        eps,
        index_scale,
        None,
    );
    rec.qsa_attention_batch(
        attn_q.as_ref(),
        k.as_ref(),
        v.as_ref(),
        ids.as_ref(),
        out.as_ref(),
        rows as u32,
        kv_len as u32,
        n_head as u32,
        n_kv as u32,
        attn_hd as u32,
        top as u32,
        ratio as u32,
        attn_scale,
        false,
        false,
        0,
        0,
        None,
    );
    rec.finish().unwrap();

    let mut got_ids_bytes = vec![0u8; rows * top * 4];
    be.download(ids.as_ref(), &mut got_ids_bytes).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, u32>(&got_ids_bytes),
        want_ids.as_slice()
    );
    let mut got_out_bytes = vec![0u8; want_out.len() * 4];
    be.download(out.as_ref(), &mut got_out_bytes).unwrap();
    let got_out = bytemuck::cast_slice::<u8, f32>(&got_out_bytes);
    for (i, (&got, &want)) in got_out.iter().zip(&want_out).enumerate() {
        assert!(
            (got - want).abs() < 3e-3,
            "output {i}: got {got}, want {want}"
        );
    }

    let cap = kv_len * n_kv * attn_hd;
    let q8_bytes = (cap / 32 * 34).next_multiple_of(4);
    let kq = be.alloc(q8_bytes, BufferUsage::KvCache).unwrap();
    let vq = be.alloc(q8_bytes, BufferUsage::KvCache).unwrap();
    let q8_out = be
        .alloc(want_out.len() * 4, BufferUsage::Activations)
        .unwrap();
    let rec = be.recorder().unwrap();
    rec.store_q8(k.as_ref(), kq.as_ref(), cap, 0, cap, true, 0);
    rec.store_q8(v.as_ref(), vq.as_ref(), cap, 0, cap, true, 0);
    rec.qsa_attention_batch(
        attn_q.as_ref(),
        kq.as_ref(),
        vq.as_ref(),
        ids.as_ref(),
        q8_out.as_ref(),
        rows as u32,
        kv_len as u32,
        n_head as u32,
        n_kv as u32,
        attn_hd as u32,
        top as u32,
        ratio as u32,
        attn_scale,
        true,
        true,
        cap as u32,
        cap as u32,
        None,
    );
    rec.finish().unwrap();
    let mut q8_out_bytes = vec![0u8; want_out.len() * 4];
    be.download(q8_out.as_ref(), &mut q8_out_bytes).unwrap();
    let q8_out = bytemuck::cast_slice::<u8, f32>(&q8_out_bytes);
    for (i, (&got, &want)) in q8_out.iter().zip(&want_out).enumerate() {
        assert!(
            (got - want).abs() < 0.02,
            "Q8 output {i}: got {got}, want {want}"
        );
    }
}
