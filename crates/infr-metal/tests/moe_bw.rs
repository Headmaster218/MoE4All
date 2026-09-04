//! Probe: device vs host MoeFfn path at Qwen3-30B-A3B decode shapes (one token, one layer).
//! Temporary evidence for the MoE-on-device work — run with
//! `INFR_METAL_NOMOE=1` to force the host path for the comparison:
//! `cargo test -p infr-metal --release --test moe_bw -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Graph, Op};
use infr_core::tensor::{DType, TensorDesc};
use infr_metal::MetalBackend;

fn lcg_bytes(mut seed: u32, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect()
}

fn synth_q4k(n_elem: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for blk_i in 0..(n_elem / 256) {
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.05).to_le_bytes());
        blk[2..4].copy_from_slice(&half::f16::from_f32(0.10).to_le_bytes());
        blk[4..144].copy_from_slice(&lcg_bytes(seed ^ blk_i as u32, 140));
        out.extend_from_slice(&blk);
    }
    out
}

#[test]
#[ignore = "requires a Metal GPU (evidence probe)"]
fn moe_layer_wall() {
    // Qwen3-30B-A3B MoE layer: ne=2048, n_ff_exp=768, 128 experts, 8 used.
    let (ne, n_expert, n_used, nff) = (2048usize, 128usize, 8usize, 768usize);
    let rows = std::env::var("INFR_METAL_MOE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize);
    let be = MetalBackend::new().unwrap();

    // CHAIN of layers in one graph (dst feeds the next op's x), so the timing reports the
    // MARGINAL per-layer cost inside a batched forward, not the fixed per-execute overhead.
    let nlayers = if rows == 1 { 8usize } else { 1usize };
    let shape = if rows == 1 { vec![ne] } else { vec![rows, ne] };
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(shape.clone(), DType::F32));
    let router = g.weight(TensorDesc::new(vec![n_expert, ne], DType::F32));
    let gate = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let up = g.weight(TensorDesc::new(vec![n_expert, nff, ne], DType::Q4K));
    let down = g.weight(TensorDesc::new(vec![n_expert, ne, nff], DType::Q4K));
    let mut cur = x;
    let mut dst = x;
    for l in 0..nlayers {
        dst = if l + 1 == nlayers {
            g.output(TensorDesc::new(shape.clone(), DType::F32))
        } else {
            g.internal(TensorDesc::new(shape.clone(), DType::F32))
        };
        g.push(Op::MoeFfn {
            x: cur,
            router_x: cur,
            router,
            gate_exps: gate,
            up_exps: up,
            down_exps: down,
            down_scale: None,
            fused_gate_up: false,
            dst,
            ne: ne as u32,
            n_expert: n_expert as u32,
            n_used: n_used as u32,
            n_ff_exp: nff as u32,
            scale: 1.0,
            act: infr_core::graph::Activation::Silu,
            gating: infr_core::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
            swiglu_clamp: None,
            expert_ids: None,
        });
        cur = dst;
    }

    // Identity router plus eight positive expert coordinates per row gives deterministic,
    // balanced top-8 routing. At rows=256 every expert receives 16 live rows, exposing the
    // grouped kernel's half-full 32-row tile without relying on random routing distribution.
    let mut xs = vec![0.0f32; rows * ne];
    for r in 0..rows {
        for u in 0..n_used {
            let e = (r * 17 + u * 37) % n_expert;
            xs[r * ne + e] = 2.0 - u as f32 * 0.05;
        }
    }
    let mut rw = vec![0.0f32; n_expert * ne];
    for e in 0..n_expert {
        rw[e * ne + e] = 1.0;
    }
    let bound: Vec<(infr_core::tensor::TensorId, Vec<u8>)> = vec![
        (x, bytemuck::cast_slice(&xs).to_vec()),
        (router, bytemuck::cast_slice(&rw).to_vec()),
        (gate, synth_q4k(n_expert * nff * ne, 1)),
        (up, synth_q4k(n_expert * nff * ne, 2)),
        (down, synth_q4k(n_expert * ne * nff, 3)),
    ];

    let mut bufs: Vec<(infr_core::tensor::TensorId, Box<dyn Buffer>)> = Vec::new();
    for (id, bytes) in &bound {
        let b = be.alloc(bytes.len().max(4), BufferUsage::Weights).unwrap();
        be.upload(b.as_ref(), bytes).unwrap();
        bufs.push((*id, b));
    }
    let ob = be.alloc(rows * ne * 4, BufferUsage::Activations).unwrap();
    bufs.push((dst, ob));
    let mut binds = Bindings::new();
    for (id, b) in &bufs {
        binds.bind(*id, b.as_ref());
    }
    let plan = be.compile(&g).unwrap();

    // Warmup (weight-cache build + pipeline compile), then timed reps.
    for _ in 0..3 {
        be.execute(plan.as_ref(), &binds).unwrap();
    }
    be.sync().unwrap();
    let reps = std::env::var("INFR_METAL_MOE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if rows == 1 { 50 } else { 5 });
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        be.execute(plan.as_ref(), &binds).unwrap();
    }
    be.sync().unwrap();
    let per = t0.elapsed().as_secs_f64() / (reps * nlayers) as f64;
    // S6: the escape hatch is `kernels.metal.moe` now — read the label off the backend's OWN
    // config so it can never disagree with the arm that actually ran. `MetalBackend::new()` still
    // folds the environment, so `INFR_METAL_NOMOE=1 cargo test …` drives this exactly as before.
    let path = if be.cfg().kernels.metal.moe {
        "device"
    } else {
        "host"
    };
    // Dense-equivalent bytes: grouped GEMM reuses each expert's weights across routed rows.
    let mib = (3 * n_used * nff * ne) as f64 * 0.5625 / (1u64 << 20) as f64;
    println!(
        "moe layer ({path}, rows={rows}, marginal of {nlayers}-chain): {:.3} ms/op, dense-equivalent expert bytes {:.1} MiB -> {:.1} effective GiB/s",
        per * 1e3,
        mib * rows as f64,
        mib * rows as f64 / 1024.0 / per
    );
}
