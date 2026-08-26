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
    let nw = be.alloc(norm.len() * 4, BufferUsage::Weights).unwrap();
    let scores = be.alloc(blocks * 4, BufferUsage::Activations).unwrap();
    let ids = be.alloc(top * 4, BufferUsage::Activations).unwrap();
    be.upload(q.as_ref(), &qb).unwrap();
    be.upload(raw.as_ref(), &rawb).unwrap();
    be.upload(nw.as_ref(), bytemuck::cast_slice(&norm)).unwrap();

    let row = 16usize;
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
        nw.as_ref(),
        scores.as_ref(),
        ids.as_ref(),
        kv_len as u32,
        nh as u32,
        hd as u32,
        top as u32,
        ratio as u32,
        rope_dim as u32,
        theta,
        eps,
        scale,
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
}
