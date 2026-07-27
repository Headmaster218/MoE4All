//! Isolated bandwidth timing of the int8 decode GEMV (`linear_i8_*`, m == 1), decoupled from the
//! full-model pipeline — the ROCm twin of the Vulkan `decode_gemv_bw` test.
//!
//! Decode is a weight-streaming problem: the ONLY number that matters is how much of the bus the
//! GEMV keeps busy, so this reports effective GB/s = (weight bytes read) / (time), not GFLOP/s.
//! Run it before and after a kernel change to see whether the change moved the bus, without
//! pp/tg mixing in attention, norms and dispatch overhead.
//!
//! Build: cargo build --release --features rocm -p infr-rocm --example decode_gemv_bw
//! Run:   LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/examples/decode_gemv_bw

use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::TensorDesc;
use infr_core::DType;
use infr_rocm::RocmBackend;
use std::time::Instant;

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

/// Effective GB/s of an `m=1` GEMV of `dt` weights at `in_f × out_f`, plus its per-GEMV µs.
///
/// `chain` copies of the SAME GEMV go into one graph (distinct dsts, shared x/weight) so a single
/// `execute` amortizes the host-side replay + end-of-graph sync — one op per `execute` measures the
/// harness, not the kernel (the floor is ~50 µs, which swamps every projection-sized GEMV).
fn bench(
    be: &RocmBackend,
    dt: DType,
    in_f: usize,
    out_f: usize,
    chain: usize,
    iters: usize,
) -> (f64, f64) {
    let (qpb, bpb) = infr_core::decode_spec::block_layout(dt);
    let w_bytes_len = (out_f * in_f / qpb) * bpb;
    let w_bytes: Vec<u8> = (0..w_bytes_len)
        .map(|i| (i as u32).wrapping_mul(2654435761) as u8)
        .collect();
    let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

    let mut g = Graph::new();
    let xid = g.input(f32d(in_f));
    let wid = g.weight(TensorDesc::new(vec![out_f * in_f], dt));
    let mut dsts = Vec::new();
    for i in 0..chain {
        // Only the LAST dst is an Output: every Output costs an end-of-graph `hipMemcpyDtoD`
        // writeback, which would otherwise be counted as GEMV time (~5 µs each).
        let dst = if i + 1 == chain {
            g.output(f32d(out_f))
        } else {
            g.internal(f32d(out_f))
        };
        g.push(Op::Linear {
            x: xid,
            weight: wid,
            dst,
            m: 1,
            in_f: in_f as u32,
            out_f: out_f as u32,
            w_off: 0,
        });
        dsts.push(dst);
    }
    let plan = be.compile(&g).unwrap();
    let xb = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
    be.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    let wb = be.alloc(w_bytes.len(), BufferUsage::Weights).unwrap();
    be.upload(wb.as_ref(), &w_bytes).unwrap();
    let ob = be.alloc(out_f * 4, BufferUsage::Readback).unwrap();
    let mut b = Bindings::new();
    b.bind(xid, xb.as_ref());
    b.bind(wid, wb.as_ref());
    b.bind(*dsts.last().unwrap(), ob.as_ref());

    for _ in 0..5 {
        be.execute(plan.as_ref(), &b).unwrap();
    }
    let t = Instant::now();
    for _ in 0..iters {
        be.execute(plan.as_ref(), &b).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    let n = (chain * iters) as f64;
    let gbps = w_bytes_len as f64 * n / secs / 1e9;
    (gbps, secs / n * 1e6)
}

fn main() {
    let be = RocmBackend::new(0).expect("rocm backend");
    // Qwen3-0.6B decode shapes (in_f × out_f) — the projections, then the lm_head, which alone is
    // ~22% of the model's per-token weight traffic and is where the bus either shows up or does not.
    let shapes = [
        (1024usize, 2048usize, "q      1024x2048"),
        (1024, 1024, "o      1024x1024"),
        (1024, 3072, "gate/up 1024x3072"),
        (3072, 1024, "down   3072x1024"),
        (1024, 151936, "lm_head 1024x151936"),
    ];
    // Per-op FLOOR: a GEMV whose whole weight is 18 KB. Whatever this costs is host/dispatch
    // overhead, not memory traffic — subtract it before believing any GB/s below.
    let (_, floor) = bench(&be, DType::Q4K, 1024, 32, 64, 50);
    println!("floor  1024x32: {floor:8.2} us/gemv (dispatch overhead)");
    // Q4_K/Q5_K carry the F4 128-bit weight fetch; Q6_K is the byte-wise control.
    for dt in [DType::Q4K, DType::Q5K, DType::Q6K] {
        println!("{dt:?}");
        for (k, n, label) in shapes {
            // 64 chained GEMVs per execute for the small projections; the lm_head is already big
            // enough that the harness floor is noise, and 64 copies of it would not fit.
            let chain = if n > 8192 { 4 } else { 64 };
            let (gbps, us) = bench(&be, dt, k, n, chain, 50);
            println!("  {label}: {gbps:7.1} GB/s  ({us:8.1} us/gemv)");
        }
    }
}
