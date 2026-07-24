//! Isolated timing of the int8 WMMA prefill GEMM (Slice 25), decoupled from the full-model pipeline.
//! Runs a single `Op::Linear` (Q4_K) at representative prefill shapes in a tight loop and reports the
//! effective GFLOP/s, so the RM×CN tile sweep can be judged on the KERNEL, not on pp512 (which mixes
//! in attention / norms / dispatch). Select the tile with `INFR_ROCM_WMMA_TILE=RxC`.
//!
//! Build: cargo build --release --features rocm -p infr-rocm --example wmma_bench
//! Run:   INFR_ROCM_WMMA_TILE=1x2 LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/examples/wmma_bench

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::DType;
use infr_rocm::RocmBackend;
use std::time::Instant;

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

fn bench(be: &RocmBackend, m: usize, in_f: usize, out_f: usize, iters: usize) -> f64 {
    // Q4_K: 256 elems / 144 bytes per super-block.
    let blocks = (out_f * in_f) / 256;
    let w_bytes: Vec<u8> = (0..blocks * 144)
        .map(|i| (i as u32 * 2654435761) as u8)
        .collect();
    let x: Vec<f32> = (0..m * in_f)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();

    let mut g = Graph::new();
    let xid = g.input(f32d(m * in_f));
    let wid = g.weight(TensorDesc::new(vec![out_f * in_f], DType::Q4K));
    let dst = g.output(f32d(m * out_f));
    g.push(Op::Linear {
        x: xid,
        weight: wid,
        dst,
        m: m as u32,
        in_f: in_f as u32,
        out_f: out_f as u32,
        w_off: 0,
    });
    let plan = be.compile(&g).unwrap();
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let wb = be.alloc(w_bytes.len(), BufferUsage::Weights).unwrap();
    be.upload(wb.as_ref(), &w_bytes).unwrap();
    let ob = be.alloc(m * out_f * 4, BufferUsage::Readback).unwrap();
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(dst, ob.as_ref());

    // Warmup (compile + fill caches).
    for _ in 0..5 {
        be.execute(plan.as_ref(), &b).unwrap();
    }
    let t = Instant::now();
    for _ in 0..iters {
        be.execute(plan.as_ref(), &b).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    // 2*m*in_f*out_f flops per GEMM.
    let flops = 2.0 * m as f64 * in_f as f64 * out_f as f64 * iters as f64;
    flops / secs / 1e9
}

fn main() {
    let be = RocmBackend::new(0).expect("rocm backend");
    let tile = std::env::var("INFR_ROCM_WMMA_TILE").unwrap_or_else(|_| "1x2(default)".into());
    println!("tile={tile}");
    let shapes = [
        (512usize, 1024usize, 1024usize, "qkv/o  512x1024x1024"),
        (512, 1024, 3072, "up/gate 512x1024x3072"),
        (512, 3072, 1024, "down   512x3072x1024"),
        (512, 1024, 4096, "wide   512x1024x4096"),
    ];
    for (m, k, n, label) in shapes {
        let g = bench(&be, m, k, n, 200);
        println!("  {label}: {g:8.1} GFLOP/s");
    }
}
