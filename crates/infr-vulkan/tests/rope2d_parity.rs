// Parity for the vision 2D RoPE (`rope2d.comp`, `Op::Rope2D`) — the qwen3vl ViT's
// GGML_ROPE_TYPE_VISION rotation. Reference = the infr-cpu interpreter's Rope2D arm semantics
// (split-half pairs over the full head, per-section theta reset, [y, x, y, x] streams).
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

fn reference_rope2d(
    q: &[f32],
    k: &[f32],
    pos_hw: &[i32],
    nh: usize,
    hd: usize,
    theta: f32,
    sections: [u32; 4],
) -> (Vec<f32>, Vec<f32>) {
    let rows = pos_hw.len() / 2;
    let n_pairs = hd / 2;
    let mut qo = q.to_vec();
    let mut ko = k.to_vec();
    let mut sect_start = [0usize; 4];
    let mut acc = 0usize;
    for s in 0..4 {
        sect_start[s] = acc;
        acc += sections[s] as usize;
    }
    let theta_scale = theta.powf(-2.0 / n_pairs as f32);
    for r in 0..rows {
        let (py, px) = (pos_hw[r * 2] as f32, pos_hw[r * 2 + 1] as f32);
        for h in 0..nh {
            let b = (r * nh + h) * hd;
            for p in 0..n_pairs {
                let sect = (0..4)
                    .find(|&s| p < sect_start[s] + sections[s] as usize)
                    .unwrap_or(0);
                let l = p - sect_start[sect];
                let pos_val = if sect % 2 == 0 { py } else { px };
                let ang = pos_val * theta_scale.powi(l as i32);
                let (s_, c_) = (ang.sin(), ang.cos());
                let (i0, i1) = (p, p + n_pairs);
                let (qa, qb) = (q[b + i0], q[b + i1]);
                qo[b + i0] = qa * c_ - qb * s_;
                qo[b + i1] = qa * s_ + qb * c_;
                let (ka, kb) = (k[b + i0], k[b + i1]);
                ko[b + i0] = ka * c_ - kb * s_;
                ko[b + i1] = ka * s_ + kb * c_;
            }
        }
    }
    (qo, ko)
}

#[test]
fn rope2d_matches_host_reference() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // qwen3vl ViT geometry (nh 16, hd 72, sections {18}x4) plus a tiny odd shape.
    for (rows, nh, hd, sections) in [
        (3usize, 16usize, 72usize, [18u32; 4]),
        (5, 2, 8, [2, 2, 2, 2]),
        (1, 1, 4, [1, 1, 1, 1]),
    ] {
        let n = rows * nh * hd;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.173 - 3.1).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.071 + 1.7).cos()).collect();
        let pos_hw: Vec<i32> = (0..rows * 2).map(|i| (i * 7 % 23) as i32 - 5).collect();
        let theta = 10_000.0f32;

        let qbuf = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        be.upload(qbuf.as_ref(), bytemuck::cast_slice(&q)).unwrap();
        let kbuf = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        be.upload(kbuf.as_ref(), bytemuck::cast_slice(&k)).unwrap();
        let pbuf = be.alloc(rows * 2 * 4, BufferUsage::Activations).unwrap();
        be.upload(pbuf.as_ref(), bytemuck::cast_slice(&pos_hw))
            .unwrap();
        let dq = be.alloc(n * 4, BufferUsage::Activations).unwrap();
        let dk = be.alloc(n * 4, BufferUsage::Activations).unwrap();

        let rec = be.recorder().unwrap();
        rec.rope2d(
            pbuf.as_ref(),
            qbuf.as_ref(),
            kbuf.as_ref(),
            dq.as_ref(),
            dk.as_ref(),
            rows,
            nh,
            hd,
            theta,
            sections,
        );
        rec.finish().unwrap();

        let mut out = vec![0u8; n * 4];
        be.download(dq.as_ref(), &mut out).unwrap();
        let gq: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();
        be.download(dk.as_ref(), &mut out).unwrap();
        let gk: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out).to_vec();

        let (wq, wk) = reference_rope2d(&q, &k, &pos_hw, nh, hd, theta, sections);
        let (mut max_q, mut max_k) = (0.0f32, 0.0f32);
        for i in 0..n {
            max_q = max_q.max((gq[i] - wq[i]).abs());
            max_k = max_k.max((gk[i] - wk[i]).abs());
        }
        eprintln!("rows={rows} nh={nh} hd={hd}: max_q={max_q:.7} max_k={max_k:.7}");
        assert!(max_q < 1e-4, "q parity: {max_q}");
        assert!(max_k < 1e-4, "k parity: {max_k}");
    }
}
