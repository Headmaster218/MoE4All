//! Probe: device vs host DeltaNet + Conv1dSilu at qwen35 layer shapes (one linear-attention
//! sublayer). `INFR_METAL_NODELTA=1` forces the host path for the comparison:
//! `cargo test -p infr-metal --release --test deltanet_bw -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc};
use infr_metal::MetalBackend;

fn randv(n: usize, mut seed: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

#[test]
#[ignore = "requires a Metal GPU (evidence probe)"]
fn deltanet_layer_wall() {
    // qwen35 linear-attention shape: 32 value heads, 16 key heads, 128/128 dims.
    let (nv, nk, kd, vd) = (32usize, 16usize, 128usize, 128usize);
    for rows in [1usize, 64] {
        let be = MetalBackend::new().unwrap();
        let mut g = Graph::new();
        let q = g.input(TensorDesc::new(vec![rows, nk * kd], DType::F32));
        let k = g.input(TensorDesc::new(vec![rows, nk * kd], DType::F32));
        let v = g.input(TensorDesc::new(vec![rows, nv * vd], DType::F32));
        let b = g.input(TensorDesc::new(vec![rows, nv], DType::F32));
        let a = g.input(TensorDesc::new(vec![rows, nv], DType::F32));
        let a_coef = g.weight(TensorDesc::new(vec![nv], DType::F32));
        let dt_bias = g.weight(TensorDesc::new(vec![nv], DType::F32));
        let state = g.input(TensorDesc::new(vec![nv * kd * vd], DType::F32));
        let dst = g.output(TensorDesc::new(vec![rows, nv * vd], DType::F32));
        g.push(Op::DeltaNet {
            q,
            k,
            v,
            b,
            a,
            a_coef,
            dt_bias,
            state,
            dst,
            rows: rows as u32,
            n_vhead: nv as u32,
            n_khead: nk as u32,
            head_k: kd as u32,
            head_v: vd as u32,
            eps: 1e-6,
            src_stride: 0,
        });
        let bound = vec![
            (q, randv(rows * nk * kd, 1)),
            (k, randv(rows * nk * kd, 2)),
            (v, randv(rows * nv * vd, 3)),
            (b, randv(rows * nv, 4)),
            (a, randv(rows * nv, 5)),
            (a_coef, randv(nv, 6)),
            (dt_bias, randv(nv, 7)),
            (state, randv(nv * kd * vd, 8)),
        ];
        let mut bufs: Vec<(infr_core::tensor::TensorId, Box<dyn Buffer>)> = Vec::new();
        for (id, vals) in &bound {
            let bytes: &[u8] = bytemuck::cast_slice(vals);
            let bb = be
                .alloc(bytes.len().max(4), BufferUsage::Activations)
                .unwrap();
            be.upload(bb.as_ref(), bytes).unwrap();
            bufs.push((*id, bb));
        }
        bufs.push((
            dst,
            be.alloc(rows * nv * vd * 4, BufferUsage::Activations)
                .unwrap(),
        ));
        let mut binds = Bindings::new();
        for (id, bb) in &bufs {
            binds.bind(*id, bb.as_ref());
        }
        let plan = be.compile(&g).unwrap();
        for _ in 0..3 {
            be.execute(plan.as_ref(), &binds).unwrap();
        }
        be.sync().unwrap();
        let reps = 30;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            be.execute(plan.as_ref(), &binds).unwrap();
        }
        be.sync().unwrap();
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        // S6: read the label off the backend's OWN config (`kernels.metal.deltanet`) so it cannot
        // disagree with the arm that ran. `MetalBackend::new()` still folds the environment, so
        // `INFR_METAL_NODELTA=1 cargo test …` drives this exactly as before.
        let path = if be.cfg().kernels.metal.deltanet {
            "device"
        } else {
            "host"
        };
        println!("deltanet rows={rows} ({path}): {:.3} ms/op", per * 1e3);
    }
}
