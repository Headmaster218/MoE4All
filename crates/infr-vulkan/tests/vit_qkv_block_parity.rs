// Reproduce the ViT engine's per-block prologue exactly: LayerNorm → THREE f16-weight Linears
// sharing one INTERNAL input tensor → AddBias, all dsts INTERNAL, dumped via Copy to outputs.
use infr_core::{
    backend::{Backend, Bindings, BufferUsage},
    graph::{Graph, Op},
    tensor::{DType, TensorDesc},
};
use infr_vulkan::VulkanBackend;

#[test]
fn vit_qkv_internal_parity() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for (m, d) in [(4usize, 1152usize), (256usize, 1152usize)] {
        // Three DIFFERENT weights (like q/k/v column slices of one fused tensor).
        let mut ws_f32 = Vec::new();
        let mut ws_f16 = Vec::new();
        for wj in 0..3 {
            let w: Vec<f32> = (0..d * d)
                .map(|i| ((i + wj * 977) as f32 * 0.113 - 21.0).sin() * 0.31)
                .collect();
            ws_f16.push(
                w.iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
                    .collect::<Vec<u16>>(),
            );
            ws_f32.push(w);
        }
        let bs: Vec<Vec<f32>> = (0..3)
            .map(|wj| {
                (0..d)
                    .map(|i| ((i + wj * 31) as f32 * 0.7 - 40.0).cos() * 0.13)
                    .collect()
            })
            .collect();
        let x: Vec<f32> = (0..m * d)
            .map(|i| (i as f32 * 0.31 - 7.0).sin() * 1.1)
            .collect();

        let mut graph = Graph::new();
        let xin = graph.input(TensorDesc::new(vec![m, d], DType::F32));
        let wq = graph.weight(TensorDesc::new(vec![d, d], DType::F16));
        let wk = graph.weight(TensorDesc::new(vec![d, d], DType::F16));
        let wv = graph.weight(TensorDesc::new(vec![d, d], DType::F16));
        let bq = graph.weight(TensorDesc::new(vec![d], DType::F32));
        let bk = graph.weight(TensorDesc::new(vec![d], DType::F32));
        let bv = graph.weight(TensorDesc::new(vec![d], DType::F32));
        let normed = graph.internal(TensorDesc::new(vec![m, d], DType::F32));
        let q = graph.internal(TensorDesc::new(vec![m, d], DType::F32));
        let k = graph.internal(TensorDesc::new(vec![m, d], DType::F32));
        let v = graph.internal(TensorDesc::new(vec![m, d], DType::F32));
        let oq = graph.output(TensorDesc::new(vec![m, d], DType::F32));
        let ok = graph.output(TensorDesc::new(vec![m, d], DType::F32));
        let ov = graph.output(TensorDesc::new(vec![m, d], DType::F32));
        graph.push(Op::Copy {
            src: xin,
            src_off: 0,
            dst: normed,
            dst_off: 0,
            n: (m * d) as u32,
        });
        for (dst, w) in [(q, wq), (k, wk), (v, wv)] {
            graph.push(Op::Linear {
                x: normed,
                weight: w,
                dst,
                m: m as u32,
                in_f: d as u32,
                out_f: d as u32,
                w_off: 0,
            });
        }
        for (t, b) in [(q, bq), (k, bk), (v, bv)] {
            graph.push(Op::AddBias {
                x: t,
                bias: b,
                dst: t,
                rows: m as u32,
                n: d as u32,
            });
        }
        for (src, dst) in [(q, oq), (k, ok), (v, ov)] {
            graph.push(Op::Copy {
                src,
                src_off: 0,
                dst,
                dst_off: 0,
                n: (m * d) as u32,
            });
        }
        let plan = be.compile(&graph).unwrap();

        let xbuf = be.alloc(m * d * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let mut wbufs = Vec::new();
        for w in &ws_f16 {
            let b = be.alloc(d * d * 2, BufferUsage::Weights).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(w)).unwrap();
            wbufs.push(b);
        }
        let mut bbufs = Vec::new();
        for b in &bs {
            let buf = be.alloc(d * 4, BufferUsage::Weights).unwrap();
            be.upload(buf.as_ref(), bytemuck::cast_slice(b)).unwrap();
            bbufs.push(buf);
        }
        let mut obufs = Vec::new();
        for _ in 0..3 {
            obufs.push(be.alloc(m * d * 4, BufferUsage::Readback).unwrap());
        }
        let mut bindings = Bindings::new();
        bindings
            .bind(xin, xbuf.as_ref())
            .bind(wq, wbufs[0].as_ref())
            .bind(wk, wbufs[1].as_ref())
            .bind(wv, wbufs[2].as_ref())
            .bind(bq, bbufs[0].as_ref())
            .bind(bk, bbufs[1].as_ref())
            .bind(bv, bbufs[2].as_ref())
            .bind(oq, obufs[0].as_ref())
            .bind(ok, obufs[1].as_ref())
            .bind(ov, obufs[2].as_ref());
        be.execute(plan.as_ref(), &bindings).unwrap();

        let outs: Vec<Vec<f32>> = obufs
            .iter()
            .map(|b| {
                let mut bytes = vec![0u8; m * d * 4];
                be.download(b.as_ref(), &mut bytes).unwrap();
                bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
            })
            .collect();

        for (wj, got) in outs.iter().enumerate() {
            let mut max = 0.0f32;
            let mut bad_rows = 0usize;
            for r in 0..m {
                let mut rmax = 0.0f32;
                for o in 0..d {
                    let mut acc = bs[wj][o];
                    for i in 0..d {
                        acc += x[r * d + i] * ws_f32[wj][o * d + i];
                    }
                    rmax = rmax.max((got[r * d + o] - acc).abs());
                }
                if rmax > 0.02 {
                    bad_rows += 1;
                }
                max = max.max(rmax);
            }
            eprintln!("w{wj}: max_err={max:.6} bad_rows={bad_rows}/{m}");
            assert!(max < 0.02, "w{wj} parity blown: {max}");
        }
    }
}
