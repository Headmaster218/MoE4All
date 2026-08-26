//! Per-op parity probe: run a one-op agnostic Graph on the CPU reference backend and on Vulkan,
//! compare outputs. Isolates which qwen35-specific op diverges on the Vulkan seam (the whole-model
//! seam garbles; this pinpoints the culprit). Run with:
//!   cargo test -p infr-llama --release --test seam_op_parity -- --include-ignored --nocapture
use infr_core::backend::{Backend, Bindings, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, MoeGating, Op};
use infr_core::tensor::TensorDesc;
use infr_core::{DType, TensorId};

fn f32d(n: usize) -> TensorDesc {
    TensorDesc::new(vec![n], DType::F32)
}

/// Run `build` (returns the graph + the ordered (handle, data) inputs + the output handle+len) on
/// `be`, returning the downloaded output.
fn run(
    be: &dyn Backend,
    g: &Graph,
    inputs: &[(TensorId, &[f32])],
    weights: &[(TensorId, &[f32])],
    out: TensorId,
    out_len: usize,
) -> Vec<f32> {
    let plan = be.compile(g).expect("compile");
    // Alloc + upload all inputs/weights first (owned), then bind from the Vec so the bound refs
    // outlive `execute`.
    let mut keep: Vec<(TensorId, Box<dyn infr_core::backend::Buffer>)> = Vec::new();
    for (id, data) in inputs {
        let buf = be
            .alloc(data.len() * 4, BufferUsage::Activations)
            .expect("alloc in");
        be.upload(buf.as_ref(), bytemuck::cast_slice(data)).unwrap();
        keep.push((*id, buf));
    }
    for (id, data) in weights {
        let buf = be
            .alloc(data.len() * 4, BufferUsage::Weights)
            .expect("alloc w");
        be.upload(buf.as_ref(), bytemuck::cast_slice(data)).unwrap();
        keep.push((*id, buf));
    }
    let obuf = be
        .alloc(out_len * 4, BufferUsage::Readback)
        .expect("alloc out");
    let mut b = Bindings::new();
    for (id, buf) in &keep {
        b.bind(*id, buf.as_ref());
    }
    b.bind(out, obuf.as_ref());
    be.execute(plan.as_ref(), &b).expect("execute");
    let mut o = vec![0f32; out_len];
    be.download(obuf.as_ref(), bytemuck::cast_slice_mut(&mut o))
        .unwrap();
    o
}

fn gpu() -> Option<infr_vulkan::VulkanBackend> {
    infr_vulkan::VulkanBackend::new().ok()
}

/// Does an in-place-mutated recurrent state Input PERSIST across `execute` calls? (Decode runs one
/// token per execute, carrying conv/SSM state in the bound buffer.) Runs Conv1dSilu twice reusing the
/// same state buffer on each backend; the 2nd output must match — if Vulkan doesn't persist the
/// in-place state write, its 2nd token diverges (the whole-model seam garble).
#[test]
#[ignore = "requires a Vulkan GPU"]
fn state_persists_across_executes() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (cc, kernel) = (32usize, 4usize);
    let build = || {
        let mut g = Graph::new();
        let x = g.input(f32d(cc));
        let w = g.weight(f32d(cc * kernel));
        let state = g.input(f32d((kernel - 1) * cc));
        let dst = g.output(f32d(cc));
        g.push(Op::Conv1dSilu {
            x,
            weight: w,
            state,
            dst,
            rows: 1,
            channels: cc as u32,
            kernel: kernel as u32,
        });
        (g, x, w, state, dst)
    };
    let wi = gen(cc * kernel, 7);
    let x1 = gen(cc, 10);
    let x2 = gen(cc, 11);
    // Second-token output when the SAME state buffer is reused across two executes.
    let second = |be: &dyn Backend| -> Vec<f32> {
        let (g, x, w, state, dst) = build();
        let plan = be.compile(&g).unwrap();
        let sbuf = be
            .alloc((kernel - 1) * cc * 4, BufferUsage::Activations)
            .unwrap(); // zeroed
        let wbuf = be.alloc(cc * kernel * 4, BufferUsage::Weights).unwrap();
        be.upload(wbuf.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let xbuf = be.alloc(cc * 4, BufferUsage::Activations).unwrap();
        let obuf = be.alloc(cc * 4, BufferUsage::Readback).unwrap();
        let run1 = |xin: &[f32]| {
            be.upload(xbuf.as_ref(), bytemuck::cast_slice(xin)).unwrap();
            let mut b = Bindings::new();
            b.bind(x, xbuf.as_ref());
            b.bind(w, wbuf.as_ref());
            b.bind(state, sbuf.as_ref());
            b.bind(dst, obuf.as_ref());
            be.execute(plan.as_ref(), &b).unwrap();
        };
        run1(&x1);
        run1(&x2);
        let mut o = vec![0f32; cc];
        be.download(obuf.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = second(&cpu);
    let v = second(&vk);
    println!("state-persist 2nd-token max_err={:e}", maxerr(&c, &v));
    assert!(
        maxerr(&c, &v) < 1e-3,
        "recurrent state does NOT persist across executes on Vulkan"
    );
}

fn maxerr(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn gen(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 13 + salt) % 29) as f32 - 14.0) * 0.05)
        .collect()
}

#[test]
fn qwen4_primitives_match_reference() {
    let cpu = infr_cpu::CpuBackend::new();

    let n = 37usize;
    let mut silu_graph = Graph::new();
    let silu_x = silu_graph.input(f32d(n));
    let silu_dst = silu_graph.output(f32d(n));
    let silu_scale = 0.25f32;
    silu_graph.push(Op::Silu {
        x: silu_x,
        dst: silu_dst,
        n: n as u32,
        scale: silu_scale,
    });
    let silu_in = gen(n, 31);
    let silu_ref: Vec<f32> = silu_in
        .iter()
        .map(|&x| {
            let v = x * silu_scale;
            v / (1.0 + (-v).exp())
        })
        .collect();
    let silu_cpu = run(&cpu, &silu_graph, &[(silu_x, &silu_in)], &[], silu_dst, n);
    assert!(maxerr(&silu_cpu, &silu_ref) < 1e-6);

    let (rows, hc, ne) = (2usize, 4usize, 320usize);
    let wide_n = rows * hc * ne;
    let mut mix_graph = Graph::new();
    let mix_x = mix_graph.input(f32d(wide_n));
    let mix_gate = mix_graph.input(f32d(wide_n));
    let mix_dst = mix_graph.output(f32d(rows * ne));
    mix_graph.push(Op::QwenHcMix {
        x: mix_x,
        gate: mix_gate,
        dst: mix_dst,
        rows: rows as u32,
        hc: hc as u32,
        n_embd: ne as u32,
    });
    let mix_in = gen(wide_n, 32);
    let mix_gate_in = gen(wide_n, 33);
    let mut mix_ref = vec![0.0f32; rows * ne];
    for r in 0..rows {
        for d in 0..ne {
            for h in 0..hc {
                let i = (r * hc + h) * ne + d;
                mix_ref[r * ne + d] += mix_in[i] / (1.0 + (-mix_gate_in[i]).exp()) / hc as f32;
            }
        }
    }
    let mix_cpu = run(
        &cpu,
        &mix_graph,
        &[(mix_x, &mix_in), (mix_gate, &mix_gate_in)],
        &[],
        mix_dst,
        rows * ne,
    );
    assert!(maxerr(&mix_cpu, &mix_ref) < 1e-6);

    let mut inject_graph = Graph::new();
    let inject_residual = inject_graph.input(f32d(wide_n));
    let inject_block = inject_graph.input(f32d(rows * ne));
    let inject_gate = inject_graph.input(f32d(rows * hc));
    let inject_dst = inject_graph.output(f32d(wide_n));
    inject_graph.push(Op::QwenHcInject {
        residual: inject_residual,
        block: inject_block,
        gate: inject_gate,
        dst: inject_dst,
        rows: rows as u32,
        hc: hc as u32,
        n_embd: ne as u32,
    });
    let residual_in = gen(wide_n, 34);
    let block_in = gen(rows * ne, 35);
    let inject_gate_in = gen(rows * hc, 36);
    let mut inject_ref = vec![0.0f32; wide_n];
    for r in 0..rows {
        for h in 0..hc {
            let scale = 2.0 / (1.0 + (-(inject_gate_in[r * hc + h] / hc as f32)).exp());
            for d in 0..ne {
                let i = (r * hc + h) * ne + d;
                inject_ref[i] = residual_in[i] + block_in[r * ne + d] * scale;
            }
        }
    }
    let inject_cpu = run(
        &cpu,
        &inject_graph,
        &[
            (inject_residual, &residual_in),
            (inject_block, &block_in),
            (inject_gate, &inject_gate_in),
        ],
        &[],
        inject_dst,
        wide_n,
    );
    assert!(maxerr(&inject_cpu, &inject_ref) < 1e-6);

    let mut ple_graph = Graph::new();
    let ple_key = ple_graph.input(f32d(wide_n));
    let ple_query = ple_graph.input(f32d(wide_n));
    let ple_value = ple_graph.input(f32d(rows * ne));
    let ple_dst = ple_graph.output(f32d(wide_n));
    ple_graph.push(Op::QwenPleGate {
        key: ple_key,
        query: ple_query,
        value: ple_value,
        dst: ple_dst,
        rows: rows as u32,
        hc: hc as u32,
        n_embd: ne as u32,
    });
    let key_in = gen(wide_n, 37);
    let query_in = gen(wide_n, 38);
    let value_in = gen(rows * ne, 39);
    let mut ple_ref = vec![0.0f32; wide_n];
    for r in 0..rows {
        for h in 0..hc {
            let base = (r * hc + h) * ne;
            let dot: f32 = key_in[base..base + ne]
                .iter()
                .zip(&query_in[base..base + ne])
                .map(|(k, q)| k * q)
                .sum();
            let score = dot / (ne as f32).sqrt();
            let signed_root = if score > 0.0 {
                score.abs().max(1e-6).sqrt()
            } else if score < 0.0 {
                -score.abs().max(1e-6).sqrt()
            } else {
                0.0
            };
            let scale = 1.0 / (1.0 + (-signed_root).exp());
            for d in 0..ne {
                ple_ref[base + d] = value_in[r * ne + d] * scale;
            }
        }
    }
    let ple_cpu = run(
        &cpu,
        &ple_graph,
        &[
            (ple_key, &key_in),
            (ple_query, &query_in),
            (ple_value, &value_in),
        ],
        &[],
        ple_dst,
        wide_n,
    );
    assert!(maxerr(&ple_cpu, &ple_ref) < 1e-6);

    if let Some(vk) = gpu() {
        let silu_vk = run(&vk, &silu_graph, &[(silu_x, &silu_in)], &[], silu_dst, n);
        let mix_vk = run(
            &vk,
            &mix_graph,
            &[(mix_x, &mix_in), (mix_gate, &mix_gate_in)],
            &[],
            mix_dst,
            rows * ne,
        );
        let inject_vk = run(
            &vk,
            &inject_graph,
            &[
                (inject_residual, &residual_in),
                (inject_block, &block_in),
                (inject_gate, &inject_gate_in),
            ],
            &[],
            inject_dst,
            wide_n,
        );
        let ple_vk = run(
            &vk,
            &ple_graph,
            &[
                (ple_key, &key_in),
                (ple_query, &query_in),
                (ple_value, &value_in),
            ],
            &[],
            ple_dst,
            wide_n,
        );
        println!(
            "qwen4 primitive Vulkan errors: silu={:e} mix={:e} inject={:e} ple={:e}",
            maxerr(&silu_vk, &silu_ref),
            maxerr(&mix_vk, &mix_ref),
            maxerr(&inject_vk, &inject_ref),
            maxerr(&ple_vk, &ple_ref),
        );
        assert!(maxerr(&silu_vk, &silu_ref) < 1e-5);
        assert!(maxerr(&mix_vk, &mix_ref) < 1e-5);
        assert!(maxerr(&inject_vk, &inject_ref) < 1e-5);
        assert!(maxerr(&ple_vk, &ple_ref) < 1e-4);
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn copystrided_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // convout[rows, cc=q|k|v] → split q (first key_dim) with per-row stride cc.
    let (rows, key_dim, nv_vd) = (3usize, 8usize, 6usize);
    let cc = 2 * key_dim + nv_vd;
    let mut g = Graph::new();
    let src = g.input(f32d(rows * cc));
    let dq = g.output(f32d(rows * key_dim));
    g.push(Op::CopyStrided {
        src,
        src_off: key_dim as u32, // k slice
        src_stride: cc as u32,
        dst: dq,
        dst_off: 0,
        dst_stride: key_dim as u32,
        rows: rows as u32,
        n: key_dim as u32,
    });
    let input = gen(rows * cc, 1);
    let c = run(&cpu, &g, &[(src, &input)], &[], dq, rows * key_dim);
    let v = run(&vk, &g, &[(src, &input)], &[], dq, rows * key_dim);
    println!(
        "CopyStrided max_err={:e}\n cpu={:?}\n vk ={:?}",
        maxerr(&c, &v),
        c,
        v
    );
    assert!(maxerr(&c, &v) < 1e-5, "CopyStrided diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn gated_sigmoid_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nff) = (2usize, 16usize);
    let mut g = Graph::new();
    let gate = g.input(f32d(rows * nff));
    let up = g.input(f32d(rows * nff));
    let dst = g.output(f32d(rows * nff));
    g.push(Op::GatedAct {
        gate,
        up,
        dst,
        rows: rows as u32,
        nff: nff as u32,
        act: Activation::Sigmoid,
        up_off: 0,
        up_stride: 0,
        gate_stride: 0,
        gate_block_width: 0,
        swiglu_clamp: None,
    });
    let gi = gen(rows * nff, 2);
    let ui = gen(rows * nff, 3);
    let c = run(&cpu, &g, &[(gate, &gi), (up, &ui)], &[], dst, rows * nff);
    let v = run(&vk, &g, &[(gate, &gi), (up, &ui)], &[], dst, rows * nff);
    println!("GatedAct(sigmoid) max_err={:e}", maxerr(&c, &v));
    assert!(maxerr(&c, &v) < 1e-3, "GatedAct sigmoid diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn gated_gelu_offset_parity() {
    // gemma4 E2B's per-layer input mix: `gelu(gate) * up[up_off..]` — the only GatedAct with a
    // nonzero up_off (the layer's slice of the per-layer input vector).
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nff, up_off) = (1usize, 16usize, 32usize);
    let mut g = Graph::new();
    let gate = g.input(f32d(rows * nff));
    let up = g.input(f32d(up_off + rows * nff + 8));
    let dst = g.output(f32d(rows * nff));
    g.push(Op::GatedAct {
        gate,
        up,
        dst,
        rows: rows as u32,
        nff: nff as u32,
        act: Activation::Gelu,
        up_off: up_off as u32,
        up_stride: 0,
        gate_stride: 0,
        gate_block_width: 0,
        swiglu_clamp: None,
    });
    let gi = gen(rows * nff, 2);
    let ui = gen(up_off + rows * nff + 8, 3);
    let c = run(&cpu, &g, &[(gate, &gi), (up, &ui)], &[], dst, rows * nff);
    let v = run(&vk, &g, &[(gate, &gi), (up, &ui)], &[], dst, rows * nff);
    println!("GatedAct(gelu,up_off) max_err={:e}", maxerr(&c, &v));
    assert!(maxerr(&c, &v) < 1e-3, "GatedAct gelu+offset diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn qknorm_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // per-head rmsnorm over head_dim (qwen35 ssm_norm, applied to the DeltaNet output).
    let (rows, n_head, head_dim) = (2usize, 4usize, 16usize);
    let mut g = Graph::new();
    let x = g.input(f32d(rows * n_head * head_dim));
    let w = g.weight(f32d(head_dim));
    let dst = g.output(f32d(rows * n_head * head_dim));
    g.push(Op::QkNorm {
        x,
        weight: Some(w),
        dst,
        rows: rows as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        eps: 1e-6,
        x_stride: 0,
    });
    let xi = gen(rows * n_head * head_dim, 4);
    let wi = gen(head_dim, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let c = run(
        &cpu,
        &g,
        &[(x, &xi)],
        &[(w, &wi)],
        dst,
        rows * n_head * head_dim,
    );
    let v = run(
        &vk,
        &g,
        &[(x, &xi)],
        &[(w, &wi)],
        dst,
        rows * n_head * head_dim,
    );
    println!("QkNorm max_err={:e}", maxerr(&c, &v));
    assert!(maxerr(&c, &v) < 1e-3, "QkNorm diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn qknormrope_parity_qwen35_dims() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    // qwen35 attention: head_dim=256, PARTIAL rope (rope_dim=64), batched rows>1.
    let (rows, nh, hd, rope_dim) = (15usize, 4usize, 256usize, 64usize);
    let mut g = Graph::new();
    let x = g.input(f32d(rows * nh * hd));
    let w = g.weight(f32d(hd));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let dst = g.output(f32d(rows * nh * hd));
    g.push(Op::QkNormRope {
        x,
        weight: w,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta: 1e7,
        eps: 1e-6,
        freq_factors: None,
        x_stride: 0,
    });
    let xi = gen(rows * nh * hd, 4);
    let wi = gen(hd, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let posv: Vec<i32> = (0..rows as i32).collect();
    // positions are I32; upload the raw bytes as if f32 (same 4-byte width) via a tiny inline run.
    let run256 = |be: &dyn Backend| -> Vec<f32> {
        let plan = be.compile(&g).unwrap();
        let xb = be.alloc(xi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(xb.as_ref(), bytemuck::cast_slice(&xi)).unwrap();
        let wb = be.alloc(wi.len() * 4, BufferUsage::Weights).unwrap();
        be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let pb = be.alloc(posv.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(pb.as_ref(), bytemuck::cast_slice(&posv)).unwrap();
        let ob = be.alloc(xi.len() * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(x, xb.as_ref());
        b.bind(w, wb.as_ref());
        b.bind(pos, pb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut o = vec![0f32; xi.len()];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = run256(&cpu);
    let v = run256(&vk);
    let nan = v.iter().any(|x| x.is_nan());
    println!(
        "QkNormRope(qwen35 hd=256,rope=64) max_err={:e} vulkan_nan={nan}",
        maxerr(&c, &v)
    );
    // NOTE: qk_norm_rope writes f16 into `dst`; declaring `dst` f32 above reads f16-packed bytes as
    // f32 → nominal max_err is huge (expected). The DECISIVE test is `qknormrope_attn_chain` below,
    // which chains QkNormRope→Attention exactly as the seam does (f16 producer→consumer, f32 out).
    let _ = (nan, c, v);
}

/// The REAL qwen35 attention handshake: QkNormRope (writes f16 q) → Attention (reads f16 q, f16 KV
/// cache, writes f32 o). Reproduces the exact producer→consumer dtype flow at qwen35 dims (hd=256,
/// PARTIAL rope=64, GQA nh=4/nkv=2, BATCHED rows>1). The dense seam never exercises attention_kv at
/// rows>1 (hd=128 → flash) and the bespoke qwen35 only runs it at rows=1, so batched attention_kv is
/// untested. Output is f32 → clean CPU-vs-Vulkan comparison. Localizes the seam NaN to this pair.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn qknormrope_attn_chain() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, nkv, hd, rope_dim) = (15usize, 4usize, 2usize, 256usize, 64usize);
    let kv_len = rows; // pos=0, causal: query ti attends keys [0, ti]
    let mut g = Graph::new();
    let x = g.input(f32d(rows * nh * hd));
    let qw = g.weight(f32d(hd));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let kc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let qa = g.internal(f32d(rows * nh * hd));
    let dst = g.output(f32d(rows * nh * hd));
    g.push(Op::QkNormRope {
        x,
        weight: qw,
        positions: pos,
        dst: qa,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta: 1e7,
        eps: 1e-6,
        freq_factors: None,
        x_stride: 0,
    });
    g.push(Op::Attention {
        q: qa,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: AttnMask::Causal,
        pos: 0,
        sinks: None,
    });
    let xi = gen(rows * nh * hd, 4);
    let wi = gen(hd, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let posv: Vec<i32> = (0..rows as i32).collect();
    // f16 KV cache (as the seam's WriteKv produces).
    let f16b = |vals: &[f32]| -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect()
    };
    let kf = f16b(&gen(kv_len * nkv * hd, 8));
    let vf = f16b(&gen(kv_len * nkv * hd, 9));
    let runner = |be: &dyn Backend| -> Vec<f32> {
        let plan = be.compile(&g).unwrap();
        let xb = be.alloc(xi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(xb.as_ref(), bytemuck::cast_slice(&xi)).unwrap();
        let wb = be.alloc(wi.len() * 4, BufferUsage::Weights).unwrap();
        be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let pb = be.alloc(posv.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(pb.as_ref(), bytemuck::cast_slice(&posv)).unwrap();
        let kb = be.alloc(kf.len(), BufferUsage::Activations).unwrap();
        be.upload(kb.as_ref(), &kf).unwrap();
        let vb = be.alloc(vf.len(), BufferUsage::Activations).unwrap();
        be.upload(vb.as_ref(), &vf).unwrap();
        let ob = be.alloc(xi.len() * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(x, xb.as_ref());
        b.bind(qw, wb.as_ref());
        b.bind(pos, pb.as_ref());
        b.bind(kc, kb.as_ref());
        b.bind(vc, vb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut o = vec![0f32; xi.len()];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = runner(&cpu);
    let v = runner(&vk);
    let nan = v.iter().any(|x| x.is_nan());
    println!(
        "QkNormRope→Attention(qwen35) max_err={:e} vulkan_nan={nan}",
        maxerr(&c, &v)
    );
    assert!(!nan && maxerr(&c, &v) < 5e-2, "qwen35 attn chain diverges");
}

/// FULL qwen35 attention core in ONE graph/command buffer: QkNormRope(K)→WriteKv (fused peephole,
/// f16 cache write at rows>1) + WriteKv(V) + Attention — all reading/writing the SAME kc/vc cache
/// buffers within a single execute. This is what the seam does but the isolated chain above does
/// NOT: it tests (a) the fused K-QkNormRope→cache write at rows>1 and (b) the WriteKv→Attention
/// read-after-write ordering inside one command buffer. If THIS diverges, the bug is the in-buffer
/// KV write→read handshake (barrier) or the fused K path at batched rows.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn qwen35_attn_core_writekv() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, nkv, hd, rope_dim) = (15usize, 4usize, 2usize, 256usize, 64usize);
    let kv_len = rows;
    let mut g = Graph::new();
    let qx = g.input(f32d(rows * nh * hd));
    let kx = g.input(f32d(rows * nkv * hd));
    let vx = g.input(f32d(rows * nkv * hd));
    let qw = g.weight(f32d(hd));
    let kw = g.weight(f32d(hd));
    let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
    let qa = g.internal(f32d(rows * nh * hd));
    // K-norm output is an F16 scratch → the Vulkan peephole fuses QkNormRope+WriteKv into a direct
    // cache write. (An F32 `ka` here reproduces the seam bug: f16 written into f32, then store_f16
    // reads it as f32 → garbage cache.)
    let ka = g.internal(TensorDesc::new(vec![rows * nkv * hd], DType::F16));
    let kc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let vc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
    let dst = g.output(f32d(rows * nh * hd));
    let qknr = |x, weight, dst, n_head| Op::QkNormRope {
        x,
        weight,
        positions: pos,
        dst,
        rows: rows as u32,
        n_head,
        head_dim: hd as u32,
        rope_dim: rope_dim as u32,
        theta: 1e7,
        eps: 1e-6,
        freq_factors: None,
        x_stride: 0,
    };
    g.push(qknr(qx, qw, qa, nh as u32));
    g.push(qknr(kx, kw, ka, nkv as u32)); // fused with the next WriteKv by the peephole
    g.push(Op::WriteKv {
        src: ka,
        cache: kc,
        rows: rows as u32,
        row_stride: (nkv * hd) as u32,
        pos: 0,
    });
    g.push(Op::WriteKv {
        src: vx,
        cache: vc,
        rows: rows as u32,
        row_stride: (nkv * hd) as u32,
        pos: 0,
    });
    g.push(Op::Attention {
        q: qa,
        k_cache: kc,
        v_cache: vc,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: nh as u32,
        n_kv: nkv as u32,
        head_dim: hd as u32,
        scale: 1.0 / (hd as f32).sqrt(),
        mask: AttnMask::Causal,
        pos: 0,
        sinks: None,
    });
    let qi = gen(rows * nh * hd, 4);
    let ki = gen(rows * nkv * hd, 8);
    let vi = gen(rows * nkv * hd, 9);
    let qwi = gen(hd, 5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let kwi = gen(hd, 6).iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let posv: Vec<i32> = (0..rows as i32).collect();
    let out_len = rows * nh * hd;
    let cache_bytes = kv_len * nkv * hd * 2;
    let runner = |be: &dyn Backend| -> Vec<f32> {
        let plan = be.compile(&g).unwrap();
        let up = |data: &[f32], usage| {
            let b = be.alloc(data.len() * 4, usage).unwrap();
            be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
            b
        };
        let qb = up(&qi, BufferUsage::Activations);
        let kb = up(&ki, BufferUsage::Activations);
        let vb = up(&vi, BufferUsage::Activations);
        let qwb = up(&qwi, BufferUsage::Weights);
        let kwb = up(&kwi, BufferUsage::Weights);
        let pbuf = be.alloc(posv.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(pbuf.as_ref(), bytemuck::cast_slice(&posv))
            .unwrap();
        let kcb = be.alloc(cache_bytes, BufferUsage::Activations).unwrap(); // zeroed
        let vcb = be.alloc(cache_bytes, BufferUsage::Activations).unwrap();
        let ob = be.alloc(out_len * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(qx, qb.as_ref());
        b.bind(kx, kb.as_ref());
        b.bind(vx, vb.as_ref());
        b.bind(qw, qwb.as_ref());
        b.bind(kw, kwb.as_ref());
        b.bind(pos, pbuf.as_ref());
        b.bind(kc, kcb.as_ref());
        b.bind(vc, vcb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut o = vec![0f32; out_len];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };
    let c = runner(&cpu);
    let v = runner(&vk);
    let nan = v.iter().any(|x| x.is_nan());
    println!(
        "qwen35 attn-core(WriteKv) max_err={:e} vulkan_nan={nan}",
        maxerr(&c, &v)
    );
    assert!(
        !nan && maxerr(&c, &v) < 5e-2,
        "qwen35 attn core (WriteKv) diverges"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn conv1d_silu_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, cc, kernel) = (4usize, 32usize, 4usize);
    let mut g = Graph::new();
    let x = g.input(f32d(rows * cc));
    let w = g.weight(f32d(cc * kernel));
    let state = g.input(f32d((kernel - 1) * cc)); // zeroed history (calloc)
    let dst = g.output(f32d(rows * cc));
    g.push(Op::Conv1dSilu {
        x,
        weight: w,
        state,
        dst,
        rows: rows as u32,
        channels: cc as u32,
        kernel: kernel as u32,
    });
    let xi = gen(rows * cc, 6);
    let wi = gen(cc * kernel, 7);
    let st = vec![0f32; (kernel - 1) * cc];
    let c = run(
        &cpu,
        &g,
        &[(x, &xi), (state, &st)],
        &[(w, &wi)],
        dst,
        rows * cc,
    );
    let v = run(
        &vk,
        &g,
        &[(x, &xi), (state, &st)],
        &[(w, &wi)],
        dst,
        rows * cc,
    );
    println!("Conv1dSilu max_err={:e}", maxerr(&c, &v));
    assert!(maxerr(&c, &v) < 1e-3, "Conv1dSilu diverges");
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn deltanet_chunked_parity() {
    // rows ≥ 32 routes to the CHUNKED delta-rule kernel (deltanet_chunked.comp): qwen35-like dims,
    // GQA tiling, a NONZERO initial state (exercises the cross-chunk carry) and a partial last
    // chunk (130 = 4×32 + 2). The CPU oracle is the sequential recurrence.
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nv, nk, kd, vd) = (130usize, 8usize, 4usize, 128usize, 128usize);
    let mut g = Graph::new();
    let q = g.input(f32d(rows * nk * kd));
    let k = g.input(f32d(rows * nk * kd));
    let v = g.input(f32d(rows * nv * vd));
    let b = g.input(f32d(rows * nv));
    let a = g.input(f32d(rows * nv));
    let a_coef = g.weight(f32d(nv));
    let dt_bias = g.weight(f32d(nv));
    let state = g.input(f32d(nv * kd * vd));
    let dst = g.output(f32d(rows * nv * vd));
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
    let (qi, ki, vi) = (
        gen(rows * nk * kd, 1),
        gen(rows * nk * kd, 2),
        gen(rows * nv * vd, 3),
    );
    let (bi, ai) = (gen(rows * nv, 4), gen(rows * nv, 5));
    // a_coef must be negative (log-decay scale); gen() is symmetric, so force sign.
    let aci: Vec<f32> = gen(nv, 8).iter().map(|x| -x.abs() - 0.1).collect();
    let dti = gen(nv, 9);
    let st = gen(nv * kd * vd, 10);
    let ins = [
        (q, &qi[..]),
        (k, &ki[..]),
        (v, &vi[..]),
        (b, &bi[..]),
        (a, &ai[..]),
        (state, &st[..]),
    ];
    let ws = [(a_coef, &aci[..]), (dt_bias, &dti[..])];
    let c = run(&cpu, &g, &ins, &ws, dst, rows * nv * vd);
    let vv = run(&vk, &g, &ins, &ws, dst, rows * nv * vd);
    let e = maxerr(&c, &vv);
    println!("DeltaNet-chunked rows={rows} max_err={e:e}");
    assert!(
        e < 1e-3,
        "chunked DeltaNet diverges from the sequential oracle"
    );
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn deltanet_parity() {
    let Some(vk) = gpu() else {
        return;
    };
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nv, nk, kd, vd) = (4usize, 4usize, 2usize, 16usize, 16usize);
    let mut g = Graph::new();
    let q = g.input(f32d(rows * nk * kd));
    let k = g.input(f32d(rows * nk * kd));
    let v = g.input(f32d(rows * nv * vd));
    let b = g.input(f32d(rows * nv));
    let a = g.input(f32d(rows * nv));
    let a_coef = g.weight(f32d(nv));
    let dt_bias = g.weight(f32d(nv));
    let state = g.input(f32d(nv * kd * vd)); // zeroed
    let dst = g.output(f32d(rows * nv * vd));
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
    let (qi, ki, vi) = (
        gen(rows * nk * kd, 1),
        gen(rows * nk * kd, 2),
        gen(rows * nv * vd, 3),
    );
    let (bi, ai) = (gen(rows * nv, 4), gen(rows * nv, 5));
    let (aci, dti) = (gen(nv, 8), gen(nv, 9));
    let st = vec![0f32; nv * kd * vd];
    let ins = [
        (q, &qi[..]),
        (k, &ki[..]),
        (v, &vi[..]),
        (b, &bi[..]),
        (a, &ai[..]),
        (state, &st[..]),
    ];
    let ws = [(a_coef, &aci[..]), (dt_bias, &dti[..])];
    let c = run(&cpu, &g, &ins, &ws, dst, rows * nv * vd);
    let vv = run(&vk, &g, &ins, &ws, dst, rows * nv * vd);
    println!(
        "DeltaNet max_err={:e}\n cpu={:?}\n vk ={:?}",
        maxerr(&c, &vv),
        c,
        vv
    );
    assert!(maxerr(&c, &vv) < 1e-2, "DeltaNet diverges");
}

/// MLA (Multi-head Latent Attention) parity: CPU backend vs a hand-written f32 reference that
/// replicates the absorbed-form math (rope q_pe, absorb q_nope via wk_b, SDPA, wv_b output).
/// Small synthetic dims — no GGUF, no model load — so this runs in every CI and catches
/// regressions in the kernel independently of the graph builder.
#[test]
fn mla_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    // Tiny dims so the reference is trivially verifiable by hand.
    let (rows, nh, kv_lora, qk_nope, qk_rope, vhd) =
        (2usize, 2usize, 3usize, 2usize, 2usize, 2usize);
    let key_len = kv_lora + qk_rope; // 5
    let q_head_dim = qk_nope + qk_rope; // 4

    let mut g = Graph::new();
    // Q: [rows, nh, q_head_dim] — each row has nh heads of [nope(2)|rope(2)].
    let q = g.input(f32d(rows * nh * q_head_dim));
    // K cache: [kv_len, key_len] — kv_len rows of key_len=5 elements each (latent 3 + rope 2).
    // V = first kv_lora=3 columns of each K row (aliased).
    let k_cache = g.input(f32d(rows * key_len)); // kv_len = rows (simple case)
                                                 // wk_b: [nh, kv_lora, qk_nope] = [2, 3, 2]
    let wk_b = g.weight(f32d(nh * kv_lora * qk_nope));
    // wv_b: [nh, kv_lora, vhd] = [2, 3, 2]
    let wv_b = g.weight(f32d(nh * kv_lora * vhd));
    let dst = g.output(f32d(rows * nh * vhd));
    let scale = 1.0 / ((qk_nope + qk_rope) as f32).sqrt(); // 1/sqrt(4) = 0.5
    g.push(Op::Mla {
        q,
        k_cache,
        wk_b,
        wv_b,
        dst,
        rows: rows as u32,
        kv_len: rows as u32, // attend to all rows
        n_head: nh as u32,
        q_head_dim: q_head_dim as u32,
        kv_lora_rank: kv_lora as u32,
        qk_nope_dim: qk_nope as u32,
        qk_rope_dim: qk_rope as u32,
        v_head_dim: vhd as u32,
        scale,
        mask: AttnMask::Causal,
        pos: 0,
        theta: 10000.0,
        freq_factors: None,
        key_bias: None,
    });

    // Synthetic inputs — small integers for traceability.
    // Q: row 0 head 0 = [1,2,3,4] (nope=[1,2], pe_raw=[3,4]), head 1 = [5,6,7,8]
    //    row 1 head 0 = [9,10,11,12], head 1 = [13,14,15,16]
    let qi: Vec<f32> = (1..=((rows * nh * q_head_dim) as i32))
        .map(|x| x as f32)
        .collect();
    // K cache: each row = [10,11,12, 1,2] (latent=[10,11,12], k_pe_raw=[1,2])
    let ki: Vec<f32> = (0..rows * key_len)
        .map(|i| {
            let col = i % key_len;
            if col < kv_lora {
                (10 + col) as f32
            } else {
                (1 + (col - kv_lora)) as f32
            }
        })
        .collect();
    // wk_b[h][a_idx][nope_idx]: lay out as [nh][kv_lora][qk_nope] row-major within each head.
    // wk_b[h=0] = [[1,0], [0,1], [0,0]]  — maps nope[0]→latent[0], nope[1]→latent[1]
    // wk_b[h=1] = [[0,0], [1,0], [0,1]]  — maps nope[0]→latent[1], nope[1]→latent[2]
    let mut wk: Vec<f32> = vec![0f32; nh * kv_lora * qk_nope];
    let s = kv_lora * qk_nope; // stride per head
    wk[0] = 1.0; // h=0, latent 0 ← nope 0
    wk[qk_nope + 1] = 1.0; // h=0, latent 1 ← nope 1
    wk[s + qk_nope] = 1.0; // h=1, latent 1 ← nope 0
    wk[s + 2 * qk_nope + 1] = 1.0; // h=1, latent 2 ← nope 1
                                   // wv_b[h][a_idx][o_idx]: identity for h=0, shifted for h=1.
    let mut wv: Vec<f32> = vec![0f32; nh * kv_lora * vhd];
    for h in 0..nh {
        let off = h * kv_lora * vhd;
        for a in 0..kv_lora.min(vhd) {
            wv[off + a * vhd + a] = 1.0; // wv_b[h][a][a] = 1
        }
    }
    let ins = [(q, &qi[..]), (k_cache, &ki[..])];
    let ws = [(wk_b, &wk[..]), (wv_b, &wv[..])];
    let c = run(&cpu, &g, &ins, &ws, dst, rows * nh * vhd);

    // Hand-written reference: for each (row, head), absorb q_nope → dot K → softmax → wv_b.
    let theta: f32 = 10000.0;
    let hf = qk_rope / 2;
    let mut ref_out = vec![0f32; rows * nh * vhd];
    for ti in 0..rows {
        let abs = ti; // pos=0, causal
        for h in 0..nh {
            // Extract q for this (row, head).
            let q_off = (ti * nh + h) * q_head_dim;
            let q_nope = &qi[q_off..q_off + qk_nope];
            let q_pe_raw = &qi[q_off + qk_nope..q_off + q_head_dim];
            // Absorb: q_full[0..kv_lora] = wk_b[h]^T @ q_nope
            let wk_off = h * kv_lora * qk_nope;
            let mut q_full = vec![0f32; key_len];
            for j in 0..kv_lora {
                let mut s = 0f32;
                for i in 0..qk_nope {
                    s += wk[wk_off + i + j * qk_nope] * q_nope[i];
                }
                q_full[j] = s;
            }
            // Rope q_pe
            for p in 0..hf {
                let (i0, i1) = (2 * p, 2 * p + 1);
                let ang = abs as f32 * theta.powf(-2.0 * p as f32 / qk_rope as f32);
                let (s, c) = (ang.sin(), ang.cos());
                q_full[kv_lora + i0] = q_pe_raw[i0] * c - q_pe_raw[i1] * s;
                q_full[kv_lora + i1] = q_pe_raw[i0] * s + q_pe_raw[i1] * c;
            }
            // SDPA: attend to positions [0..abs] (causal).
            let n_keys = abs + 1;
            let mut sc = vec![0f32; n_keys];
            let mut mx = f32::NEG_INFINITY;
            for (jj, scj) in sc.iter_mut().enumerate().take(n_keys) {
                let kb = jj * key_len;
                *scj = dot_ref(&q_full, &ki[kb..kb + key_len]) * scale;
                mx = mx.max(*scj);
            }
            let mut l = 0f32;
            for &s in &sc {
                l += (s - mx).exp();
            }
            // Accumulate wv_b[h] @ V[j] into output for this head.
            for (jj, &s) in sc.iter().enumerate().take(n_keys) {
                let p = (s - mx).exp() / l;
                let kb = jj * key_len;
                let wv_off = h * kv_lora * vhd;
                for o_idx in 0..vhd {
                    let mut vs = 0f32;
                    for a in 0..kv_lora {
                        vs += wv[wv_off + a + o_idx * kv_lora] * ki[kb + a];
                    }
                    ref_out[(ti * nh + h) * vhd + o_idx] += p * vs;
                }
            }
        }
    }
    // Compare.
    let err = maxerr(&c, &ref_out);
    assert!(err < 1e-4, "MLA parity diverges: max_err={err:e}");
}

/// f32 dot product (avoids pulling in the full crate::kernels::dot).
fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Which ABSOLUTE positions a query at absolute position `abs` may attend to, given `kv_len`
/// cached positions. Stated from what each [`AttnMask`] MEANS (see its doc), NOT from any
/// backend's expression for it — `mla_mask_ring_parity` is only worth running because the
/// reference and the kernel reach the same range by different routes.
///
/// Note `hi` is clamped to `kv_len` on the causal/window arms: a query may not attend past the end
/// of what has been cached. The CPU `Op::Mla` arm omits that clamp (it takes `hi = abs + 1`
/// outright); the two agree for every `abs < kv_len`, which is all the graph builder ever emits
/// (`kv_len = start_pos + batch`, `pos = start_pos`) — see `docs/backlog.md` B46.
fn mla_attends(mask: AttnMask, abs: usize, kv_len: usize) -> std::ops::Range<usize> {
    match mask {
        // Every earlier position plus its own.
        AttnMask::Causal => 0..(abs + 1).min(kv_len),
        // The `w` most recent positions, its own included.
        AttnMask::SlidingWindow(w) => (abs + 1).saturating_sub(w)..(abs + 1).min(kv_len),
        // One fixed span for EVERY row — `abs` is not consulted at all.
        AttnMask::Canvas { lo } => lo..kv_len,
    }
}

/// Hand-written f32 reference for one `Op::Mla` dispatch.
///
/// `keys[j]` is the logical `key_len`-wide key for ABSOLUTE position `j`. The reference never
/// computes a cache ROW index — that is the whole point: the kernel reaches its key through
/// `j % cap_rows` into the ring buffer, this reaches it through the absolute position, and they
/// agree only if the kernel's modulus lands on the row the ring writer actually used.
///
/// The absorb/rope/softmax/output arithmetic still follows the same index formulas the CPU arm
/// uses for `wk_b`/`wv_b` (B46's first bullet: this reference is not an independent oracle for
/// weight ORIENTATION). Masking and ring addressing are the parts it derives independently.
#[allow(clippy::too_many_arguments)]
fn mla_ref(
    qi: &[f32],
    keys: &[Vec<f32>],
    wk: &[f32],
    wv: &[f32],
    rows: usize,
    nh: usize,
    kv_lora: usize,
    qk_nope: usize,
    qk_rope: usize,
    vhd: usize,
    kv_len: usize,
    scale: f32,
    theta: f32,
    mask: AttnMask,
    pos: usize,
) -> Vec<f32> {
    let key_len = kv_lora + qk_rope;
    let q_head_dim = qk_nope + qk_rope;
    let hf = qk_rope / 2;
    let mut out = vec![0f32; rows * nh * vhd];
    for ti in 0..rows {
        let abs = pos + ti;
        for h in 0..nh {
            let q_off = (ti * nh + h) * q_head_dim;
            let q_nope = &qi[q_off..q_off + qk_nope];
            let q_pe_raw = &qi[q_off + qk_nope..q_off + q_head_dim];
            // q_full[0..kv_lora] = wk_b[h]^T @ q_nope
            let wk_off = h * kv_lora * qk_nope;
            let mut q_full = vec![0f32; key_len];
            for (j, qf) in q_full.iter_mut().enumerate().take(kv_lora) {
                let mut s = 0f32;
                for i in 0..qk_nope {
                    s += wk[wk_off + i + j * qk_nope] * q_nope[i];
                }
                *qf = s;
            }
            // Rope q_pe at the query's ABSOLUTE position.
            for p in 0..hf {
                let (i0, i1) = (2 * p, 2 * p + 1);
                let ang = abs as f32 * theta.powf(-2.0 * p as f32 / qk_rope as f32);
                let (s, c) = (ang.sin(), ang.cos());
                q_full[kv_lora + i0] = q_pe_raw[i0] * c - q_pe_raw[i1] * s;
                q_full[kv_lora + i1] = q_pe_raw[i0] * s + q_pe_raw[i1] * c;
            }
            let span = mla_attends(mask, abs, kv_len);
            let sc: Vec<f32> = span
                .clone()
                .map(|j| dot_ref(&q_full, &keys[j]) * scale)
                .collect();
            let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = sc.iter().map(|&s| (s - mx).exp()).sum();
            for (j, &s) in span.zip(&sc) {
                let p = (s - mx).exp() / l;
                let wv_off = h * kv_lora * vhd;
                for o_idx in 0..vhd {
                    let mut vs = 0f32;
                    for a in 0..kv_lora {
                        vs += wv[wv_off + a + o_idx * kv_lora] * keys[j][a];
                    }
                    out[(ti * nh + h) * vhd + o_idx] += p * vs;
                }
            }
        }
    }
    out
}

/// One `mla_mask_ring_parity` case: a mask, a query batch and a ring capacity.
struct MlaCase {
    name: &'static str,
    rows: usize,
    pos: usize,
    kv_len: usize,
    /// Ring row capacity — the K cache tensor is declared `cap * key_len` wide, which is where the
    /// CPU arm reads `cap_rows` from. `cap < kv_len` is a genuinely wrapped cache.
    cap: usize,
    mask: AttnMask,
}

/// `Op::Mla` over the axes `mla_parity` never moves: a WRAPPED ring cache (`cap_rows < kv_len`),
/// `AttnMask::SlidingWindow`, `AttnMask::Canvas`, and a non-zero `pos` — the gap recorded as
/// `docs/backlog.md` B46's second bullet.
///
/// The cache is filled by an explicit ring WRITER (position `j` → row `j % cap`, ascending, so a
/// row reached twice keeps the later position), and [`mla_ref`] then reads keys by absolute
/// position. Kernel and reference therefore only agree if the kernel's `(lo + jj) % cap_rows`
/// resolves to the same row the writer used, which is what makes the wrap cases informative.
#[test]
fn mla_mask_ring_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    // Tiny dims, hand-checkable on failure. `kv_lora + qk_rope` is EVEN so the same case table
    // transfers to `infr-vulkan`'s `mla_ring_and_mask_matches_cpu_reference`, whose f16 cache is
    // read as u32-packed f16 PAIRS (`mla.comp`'s `kread`) — an odd key_len would put a row's last
    // element in a half-word past the end of the buffer.
    let (nh, kv_lora, qk_nope, qk_rope, vhd) = (2usize, 4usize, 2usize, 2usize, 2usize);
    let key_len = kv_lora + qk_rope; // 6
    let q_head_dim = qk_nope + qk_rope; // 4
    let scale = 1.0 / (q_head_dim as f32).sqrt();
    let theta = 10000.0f32;

    // One-hot wk_b/wv_b in the READ convention both kernels use (`i` / `a` the FAST dim): head h
    // absorbs q_nope dim `i` into latent slot `(h+i) % kv_lora`, and reads latent slot `(h+o) %
    // kv_lora` back out into output dim `o`. Distinct per output dim on purpose — `mla_parity`'s
    // `wv[off + a*vhd + a] = 1` is one-hot in the WRITE convention, which the read convention
    // collapses onto latent slot 0 for BOTH output dims, so its every output pair came out equal.
    let mut wk: Vec<f32> = vec![0f32; nh * kv_lora * qk_nope];
    let mut wv: Vec<f32> = vec![0f32; nh * kv_lora * vhd];
    for h in 0..nh {
        for i in 0..qk_nope {
            wk[h * kv_lora * qk_nope + i + ((h + i) % kv_lora) * qk_nope] = 1.0;
        }
        for o in 0..vhd {
            wv[h * kv_lora * vhd + (h + o) % kv_lora + o * kv_lora] = 1.0;
        }
    }
    // One distinct key per absolute position, all O(1): with scores this small the softmax stays
    // SOFT, so every attended key moves the output. Under a near-one-hot softmax (what large
    // values give) the result is just the winning key's V, and dropping or adding a LOSING key —
    // exactly what an off-by-one in `lo`/`hi` does — would leave the output unchanged.
    let key_at = |j: usize| -> Vec<f32> {
        (0..key_len)
            .map(|d| ((j * 7 + d * 3) % 13) as f32 / 16.0 + 0.125)
            .collect()
    };
    let q_at = |i: usize| ((i * 5 + 3) % 11) as f32 / 8.0 - 0.5;

    let cases = [
        // The `mla_parity` shape, restated here as the baseline the wrap/mask cases move away from.
        MlaCase {
            name: "causal pos=0, no wrap",
            rows: 2,
            pos: 0,
            kv_len: 2,
            cap: 2,
            mask: AttnMask::Causal,
        },
        MlaCase {
            name: "causal pos=3, no wrap",
            rows: 2,
            pos: 3,
            kv_len: 5,
            cap: 8,
            mask: AttnMask::Causal,
        },
        MlaCase {
            name: "sliding window w=3, pos=3, no wrap",
            rows: 2,
            pos: 3,
            kv_len: 5,
            cap: 8,
            mask: AttnMask::SlidingWindow(3),
        },
        // cap=5 over 14 positions is two full laps plus four rows, and the abs=12 row attends
        // 9..13 → rows 4,0,1,2: `lo` and `hi-1` sit on OPPOSITE sides of the wrap boundary, which
        // the single lap starting at row 0 would not have caught.
        MlaCase {
            name: "sliding window w=4, wrapped ring (lo/hi straddle)",
            rows: 2,
            pos: 12,
            kv_len: 14,
            cap: 5,
            mask: AttnMask::SlidingWindow(4),
        },
        // Window exactly the ring capacity: every row is read once, starting mid-ring.
        MlaCase {
            name: "sliding window w=cap=5, wrapped ring",
            rows: 1,
            pos: 13,
            kv_len: 14,
            cap: 5,
            mask: AttnMask::SlidingWindow(5),
        },
        MlaCase {
            name: "canvas lo=0, pos=3",
            rows: 2,
            pos: 3,
            kv_len: 5,
            cap: 8,
            mask: AttnMask::Canvas { lo: 0 },
        },
        // Canvas ignores `abs` entirely: both rows attend 2..5 even though their causal bounds
        // differ, and `pos` still moves the internal q_pe rope.
        MlaCase {
            name: "canvas lo=2, pos=3",
            rows: 2,
            pos: 3,
            kv_len: 5,
            cap: 8,
            mask: AttnMask::Canvas { lo: 2 },
        },
        MlaCase {
            name: "canvas lo=9, wrapped ring (straddles)",
            rows: 1,
            pos: 13,
            kv_len: 14,
            cap: 5,
            mask: AttnMask::Canvas { lo: 9 },
        },
    ];

    for case in cases {
        let MlaCase {
            name,
            rows,
            pos,
            kv_len,
            cap,
            mask,
        } = case;
        // Ring writer: absolute position j lands in row j % cap, written in ascending order.
        let keys: Vec<Vec<f32>> = (0..kv_len).map(key_at).collect();
        let mut cache = vec![0f32; cap * key_len];
        for (j, k) in keys.iter().enumerate() {
            cache[(j % cap) * key_len..][..key_len].copy_from_slice(k);
        }
        // A ring only holds the last `cap` positions. If an attended position's row was reused by
        // a LATER position, the cache no longer holds the key the reference expects and the case
        // is asking a question with no answer — catch that here rather than in a max_err.
        for ti in 0..rows {
            let abs = pos + ti;
            for j in mla_attends(mask, abs, kv_len) {
                let last = (0..kv_len)
                    .rfind(|p| p % cap == j % cap)
                    .expect("attended position is inside 0..kv_len");
                assert_eq!(
                    last, j,
                    "{name}: attended position {j} was overwritten by {last} in the ring — \
                     the case attends a wider span than cap={cap} holds"
                );
            }
        }
        let qi: Vec<f32> = (0..rows * nh * q_head_dim).map(q_at).collect();

        let mut g = Graph::new();
        let q = g.input(f32d(rows * nh * q_head_dim));
        let k_cache = g.input(f32d(cap * key_len));
        let wk_b = g.weight(f32d(nh * kv_lora * qk_nope));
        let wv_b = g.weight(f32d(nh * kv_lora * vhd));
        let dst = g.output(f32d(rows * nh * vhd));
        g.push(Op::Mla {
            q,
            k_cache,
            wk_b,
            wv_b,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            q_head_dim: q_head_dim as u32,
            kv_lora_rank: kv_lora as u32,
            qk_nope_dim: qk_nope as u32,
            qk_rope_dim: qk_rope as u32,
            v_head_dim: vhd as u32,
            scale,
            mask,
            pos: pos as u32,
            theta,
            freq_factors: None,
            key_bias: None,
        });
        let ins = [(q, &qi[..]), (k_cache, &cache[..])];
        let ws = [(wk_b, &wk[..]), (wv_b, &wv[..])];
        let got = run(&cpu, &g, &ins, &ws, dst, rows * nh * vhd);
        let want = mla_ref(
            &qi, &keys, &wk, &wv, rows, nh, kv_lora, qk_nope, qk_rope, vhd, kv_len, scale, theta,
            mask, pos,
        );
        let err = maxerr(&got, &want);
        println!("MLA {name}: max_err={err:e}\n  got ={got:?}\n  want={want:?}");
        assert!(err < 1e-5, "MLA {name} diverges: max_err={err:e}");
    }
}

/// Hand-written f32 reference for `Op::MoeFfn`'s DeepSeek V2/V3 selection path (rows=1, norm_w=true,
/// weight_before=false, SiLU, no down_scale, split gate/up), mirroring the CPU interpreter in
/// `crates/infr-cpu/src/lib.rs` (MoeFfn arm, ~2076-2702): router matvec → `gating` probs → optional
/// `bias` added to a selection-only copy → optional group-limited routing (per-group top-2 score,
/// mask non-chosen groups to -inf) → descending top-`n_used` → renormalized weights × `scale` →
/// per-expert `silu(gate·x)·(up·x)` → `down·` accumulate in top-k order.
#[allow(clippy::too_many_arguments)]
fn moe_ref(
    x: &[f32],
    router: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    ne: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    scale: f32,
    gating: MoeGating,
    bias: Option<&[f32]>,
    n_expert_groups: usize,
    n_expert_groups_used: usize,
) -> Vec<f32> {
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let logits: Vec<f32> = (0..n_expert)
        .map(|e| dot(&router[e * ne..(e + 1) * ne], x))
        .collect();
    let probs: Vec<f32> = match gating {
        MoeGating::Softmax => {
            let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
            let psum: f32 = p.iter().sum();
            p.iter_mut().for_each(|v| *v /= psum);
            p
        }
        MoeGating::Sigmoid => logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
        MoeGating::SqrtSoftplus => logits
            .iter()
            .map(|&v| {
                let sp = if v > 20.0 {
                    v
                } else {
                    (1.0_f32 + v.exp()).ln()
                };
                sp.sqrt()
            })
            .collect(),
    };
    // Selection-only copy: `bias` shifts top-k selection; the UNBIASED `probs` still drive weights.
    let mut sel: Vec<f32> = match bias {
        Some(b) => probs.iter().zip(b).map(|(&p, &bi)| p + bi).collect(),
        None => probs.clone(),
    };
    // Group-limited routing: per-group score = sum of the top-2 sel values; keep the top
    // `n_expert_groups_used` groups, mask the rest to -inf (llama.cpp `build_moe_ffn`).
    if n_expert_groups > 1 && n_expert_groups_used > 0 {
        let per = n_expert / n_expert_groups;
        let mut gscore = Vec::with_capacity(n_expert_groups);
        for g in 0..n_expert_groups {
            let mut best = [f32::NEG_INFINITY; 2];
            for &s in &sel[g * per..(g + 1) * per] {
                if s > best[0] {
                    best[1] = best[0];
                    best[0] = s;
                } else if s > best[1] {
                    best[1] = s;
                }
            }
            gscore.push(best[0] + best[1]);
        }
        let mut gidx: Vec<usize> = (0..n_expert_groups).collect();
        gidx.sort_by(|&a, &b| gscore[b].partial_cmp(&gscore[a]).unwrap());
        gidx.truncate(n_expert_groups_used);
        for g in 0..n_expert_groups {
            if !gidx.contains(&g) {
                for s in sel[g * per..(g + 1) * per].iter_mut() {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n_expert).collect();
    idx.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap());
    idx.truncate(n_used);
    // norm_w: renormalize the selected (UNBIASED) probs to sum to 1, then scale.
    let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
    let mut out = vec![0f32; ne];
    for &e in &idx {
        // gate/up are [n_expert, n_ff_exp, ne], down is [n_expert, ne, n_ff_exp] (row-major).
        let gs = e * n_ff_exp * ne;
        let ds = e * ne * n_ff_exp;
        let actv: Vec<f32> = (0..n_ff_exp)
            .map(|j| {
                let g = dot(&gate[gs + j * ne..gs + (j + 1) * ne], x);
                let u = dot(&up[gs + j * ne..gs + (j + 1) * ne], x);
                let silu = |z: f32| z / (1.0 + (-z).exp());
                silu(g) * u
            })
            .collect();
        let w_e = probs[e] / wsum * scale;
        for i in 0..ne {
            out[i] += w_e * dot(&down[ds + i * n_ff_exp..ds + (i + 1) * n_ff_exp], &actv);
        }
    }
    out
}

/// `Op::MoeFfn` with DeepSeek V4 gating — `MoeGating::SqrtSoftplus` (`sqrt(softplus(logit))`,
/// including the `v > 20` softplus shortcut branch): CPU backend vs a hand-written f32 reference,
/// plus a CPU-vs-Vulkan cross-check when a GPU is present. V2-Lite (the only real deepseek model
/// exercised here) uses plain softmax, so this gating path has never run in any model test.
/// ne/n_ff_exp = 32 (not tiny): the Vulkan expert id-GEMV decodes 32-element sub-blocks, which is
/// a hard floor the dispatch now asserts — smaller dims panic instead of cross-checking against a
/// silent all-zero GPU output.
#[test]
fn moe_sqrt_softplus_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    // ne/n_ff_exp ≥ 32: the Vulkan expert id-GEMV decodes 32-element sub-blocks
    // (`nsub = in_f/32`). Below that the dispatch is refused outright by the recorder's
    // `assert_native_k` guard; before that guard it was a silent all-zero no-op (backlog B39).
    let (ne, n_expert, n_used, n_ff_exp) = (32usize, 6usize, 2usize, 32usize);
    let mut g = Graph::new();
    let x = g.input(f32d(ne));
    let router_x = g.input(f32d(ne)); // the router's own row handle; bound data == `x`'s
    let router = g.weight(f32d(n_expert * ne));
    let gate_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
    let up_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
    let down_exps = g.weight(f32d(n_expert * ne * n_ff_exp));
    let dst = g.output(f32d(ne));
    g.push(Op::MoeFfn {
        x,
        router_x,
        router,
        gate_exps,
        up_exps,
        down_exps,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: n_ff_exp as u32,
        scale: 1.0,
        act: Activation::Silu,
        gating: MoeGating::SqrtSoftplus,
        norm_w: true,
        weight_before: false,
        ep_band: None,
        exp_probs_b: None,
        n_expert_groups: 0,
        n_expert_groups_used: 0,
        swiglu_clamp: None,
        expert_ids: None,
    });
    // Router rows = lead[e] * [1, 0, 0, …] → logits (x = 1) are [24, 1, 0.75, 0.5, -1.5, -1.0]:
    // expert 0's logit 24 > 20 exercises the softplus shortcut (`sp = v`), the rest take the exact
    // `ln(1 + exp(v))` branch. All logits distinct → top-2 (experts 0, 1) is unambiguous.
    let lead = [24.0f32, 1.0, 0.75, 0.5, -1.5, -1.0];
    let xi = [1.0f32; 32];
    let ri: Vec<f32> = (0..n_expert * ne)
        .map(|i| if i % ne == 0 { lead[i / ne] } else { 0.0 })
        .collect();
    let gi = gen(n_expert * n_ff_exp * ne, 12);
    let ui = gen(n_expert * n_ff_exp * ne, 13);
    let di = gen(n_expert * ne * n_ff_exp, 14);
    let ins = [(x, &xi[..]), (router_x, &xi[..])];
    let ws = [
        (router, &ri[..]),
        (gate_exps, &gi[..]),
        (up_exps, &ui[..]),
        (down_exps, &di[..]),
    ];
    let c = run(&cpu, &g, &ins, &ws, dst, ne);
    let reference = moe_ref(
        &xi,
        &ri,
        &gi,
        &ui,
        &di,
        ne,
        n_expert,
        n_used,
        n_ff_exp,
        1.0,
        MoeGating::SqrtSoftplus,
        None,
        0,
        0,
    );
    let e = maxerr(&c, &reference);
    println!("MoeFfn(sqrt-softplus) cpu-vs-ref max_err={e:e}");
    assert!(
        e < 1e-4,
        "MoeFfn sqrt-softplus diverges from reference: max_err={e:e}"
    );
    if let Some(vk) = gpu() {
        let v = run(&vk, &g, &ins, &ws, dst, ne);
        let e = maxerr(&c, &v);
        println!("MoeFfn(sqrt-softplus) cpu-vs-vulkan max_err={e:e}");
        assert!(
            e < 1e-3,
            "MoeFfn sqrt-softplus diverges on Vulkan: max_err={e:e}"
        );
    }
}

/// `Op::MoeFfn` with the DeepSeek V3 selection path — the `exp_probs_b` router bias (added to the
/// SELECTION scores only; the unbiased probs still drive the routing weights) plus group-limited
/// routing (`n_expert_groups`/`n_expert_groups_used`, per-group top-2 score, non-chosen groups
/// masked out): CPU backend vs a hand-written f32 reference, plus a CPU-vs-Vulkan cross-check when
/// a GPU is present. V2-Lite uses no bias and no groups, so neither feature has ever run in a model
/// test. ne/n_ff_exp = 32 (not tiny): the Vulkan expert id-GEMV decodes 32-element sub-blocks,
/// which is a hard floor the dispatch now asserts — smaller dims panic instead of cross-checking
/// against a silent all-zero GPU output.
#[test]
fn moe_groups_bias_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    // ne/n_ff_exp ≥ 32: the Vulkan expert id-GEMV decodes 32-element sub-blocks
    // (`nsub = in_f/32`). Below that the dispatch is refused outright by the recorder's
    // `assert_native_k` guard; before that guard it was a silent all-zero no-op (backlog B39).
    let (ne, n_expert, n_used, n_ff_exp) = (32usize, 8usize, 2usize, 32usize);
    let mut g = Graph::new();
    let x = g.input(f32d(ne));
    let router_x = g.input(f32d(ne));
    let router = g.weight(f32d(n_expert * ne));
    let gate_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
    let up_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
    let down_exps = g.weight(f32d(n_expert * ne * n_ff_exp));
    let exp_probs_b = g.weight(f32d(n_expert));
    let dst = g.output(f32d(ne));
    g.push(Op::MoeFfn {
        x,
        router_x,
        router,
        gate_exps,
        up_exps,
        down_exps,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: ne as u32,
        n_expert: n_expert as u32,
        n_used: n_used as u32,
        n_ff_exp: n_ff_exp as u32,
        scale: 1.0,
        act: Activation::Silu,
        gating: MoeGating::Sigmoid,
        norm_w: true,
        weight_before: false,
        ep_band: None,
        exp_probs_b: Some(exp_probs_b),
        n_expert_groups: 2,
        n_expert_groups_used: 1,
        swiglu_clamp: None,
        expert_ids: None,
    });
    // Group 0 (experts 0-3, sigmoid of logits 3.0/2.5/2.0/1.5) has the higher unbiased probs and
    // would win top-k (group score 1.877 vs 1.442); group 1 (experts 4-7, logits 1.0/0.9/0.8/0.7)
    // gets a +0.6 bias per expert so the biased selection picks group 1 and experts 4/5 instead.
    // The data is chosen so the two candidate semantics DISAGREE: under probs+bias (llama.cpp
    // `selection_probs = ggml_add(probs, exp_probs_b)` and the CPU) group 1 wins (2.642 > 1.877),
    // while under the old shader's logits+bias group 0 wins (5.5 > 3.1) — so this test FAILS on a
    // shader that biases raw logits, and passes only when the bias is added to the gated probs.
    // The reference output also pins that the UNBIASED probs still drive the weights. All 8 sel
    // values (and both per-group top-2 pairs) are distinct, so the group score and final top-2
    // are unambiguous.
    let xi = [1.0f32; 32];
    let lead = [3.0f32, 2.5, 2.0, 1.5, 1.0, 0.9, 0.8, 0.7];
    let ri: Vec<f32> = (0..n_expert * ne)
        .map(|i| if i % ne == 0 { lead[i / ne] } else { 0.0 })
        .collect();
    let bi = [0.0f32, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6];
    let gi = gen(n_expert * n_ff_exp * ne, 15);
    let ui = gen(n_expert * n_ff_exp * ne, 16);
    let di = gen(n_expert * ne * n_ff_exp, 17);
    let ins = [(x, &xi[..]), (router_x, &xi[..])];
    let ws = [
        (router, &ri[..]),
        (gate_exps, &gi[..]),
        (up_exps, &ui[..]),
        (down_exps, &di[..]),
        (exp_probs_b, &bi[..]),
    ];
    let c = run(&cpu, &g, &ins, &ws, dst, ne);
    let reference = moe_ref(
        &xi,
        &ri,
        &gi,
        &ui,
        &di,
        ne,
        n_expert,
        n_used,
        n_ff_exp,
        1.0,
        MoeGating::Sigmoid,
        Some(&bi),
        2,
        1,
    );
    let e = maxerr(&c, &reference);
    println!("MoeFfn(groups+bias) cpu-vs-ref max_err={e:e}");
    assert!(
        e < 1e-4,
        "MoeFfn groups+bias diverges from reference: max_err={e:e}"
    );
    if let Some(vk) = gpu() {
        let v = run(&vk, &g, &ins, &ws, dst, ne);
        let e = maxerr(&c, &v);
        println!("MoeFfn(groups+bias) cpu-vs-vulkan max_err={e:e}");
        assert!(
            e < 1e-3,
            "MoeFfn groups+bias diverges on Vulkan: max_err={e:e}"
        );
    }
}

/// Mean-centred LayerNorm reference, written from the DEFINITION rather than transcribed from any
/// backend: per row subtract the row mean, divide by `sqrt(var + eps)`, scale by `weight`, add
/// `bias`. `var` is the population (biased) variance — `Σ(x-mean)²` over `dim`, not `dim-1` — and
/// `eps` is added to the variance BEFORE the square root. Those two are what llama.cpp's
/// `ggml_compute_forward_norm_f32` pins down and where a plausible-looking variant is a silent
/// precision bug, so the reference states them explicitly; the accumulation runs in f64 so it is
/// an accuracy oracle for the f32 kernels too, not just a shape check.
fn layernorm_ref(x: &[f32], w: &[f32], b: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / dim as f64;
        let var = row
            .iter()
            .map(|v| (*v as f64 - mean) * (*v as f64 - mean))
            .sum::<f64>()
            / dim as f64;
        let sd = (var + eps as f64).sqrt();
        for c in 0..dim {
            out[r * dim + c] = (((row[c] as f64 - mean) / sd) as f32) * w[c] + b[c];
        }
    }
    out
}

/// Input rows chosen so the two things that make a mean-centred norm different are OBSERVABLE:
///
/// * row 0 — mean ≈ 20 with a spread of ≈ ±2, so an RMS norm (which never subtracts the mean)
///   divides by ≈ 20 where LayerNorm divides by ≈ 1.2 and the numerator differs entirely. A test
///   whose rows were already zero-mean would pass against `Op::RmsNorm`.
/// * row 1 — `0.5 ± 1/1024`, i.e. `var ≈ 9.54e-7` against `eps = 1e-6`: the two are the same
///   order, so eps INSIDE the sqrt (`1/sqrt(var+eps)` ≈ 715) and eps outside it
///   (`1/(sqrt(var)+eps)` ≈ 1023) disagree by 43% and the row decides between them.
///
/// The rest are ordinary mixed-sign rows. Callers pass a `dim` that is not a multiple of either
/// GPU reduction width (256 threads on Vulkan, 32 lanes on Metal) so the strided loops' tail runs.
fn layernorm_rows(rows: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; rows * dim];
    for r in 0..rows {
        for c in 0..dim {
            v[r * dim + c] = match r {
                0 => 20.0 + (((c * 7) % 13) as f32 - 6.0) * 0.3,
                1 => 0.5 + (if c % 2 == 0 { 1.0 } else { -1.0 }) / 1024.0,
                _ => (((c * 13 + r * 5) % 29) as f32 - 14.0) * 0.05,
            };
        }
    }
    v
}

/// `Op::LayerNorm` (deepseek32's `indexer_k_norm`, the DeepSeek family's only non-RMS norm):
/// CPU backend vs the hand-written reference above, plus a CPU-vs-Vulkan cross-check when a GPU
/// is present. `dim = 300` is a multiple of neither 256 (the Vulkan workgroup) nor 32 (the Metal
/// simdgroup), so both reductions run a partial tail iteration.
#[test]
fn layernorm_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, dim) = (7usize, 300usize);
    let eps = 1e-6f32; // deepseek32's hardcoded f_norm_eps

    let mut g = Graph::new();
    let x = g.input(f32d(rows * dim));
    let w = g.weight(f32d(dim));
    let b = g.weight(f32d(dim));
    let dst = g.output(f32d(rows * dim));
    g.push(Op::LayerNorm {
        x,
        weight: w,
        bias: b,
        dst,
        rows: rows as u32,
        dim: dim as u32,
        eps,
    });

    let xi = layernorm_rows(rows, dim);
    let wi = gen(dim, 3);
    let bi = gen(dim, 17);
    let ins = [(x, &xi[..])];
    let ws = [(w, &wi[..]), (b, &bi[..])];

    let c = run(&cpu, &g, &ins, &ws, dst, rows * dim);
    let reference = layernorm_ref(&xi, &wi, &bi, rows, dim, eps);
    println!("LayerNorm cpu-vs-ref max_err={:e}", maxerr(&c, &reference));

    // Assert PER ROW, not on the whole-tensor max: a single number hides which case broke, and
    // the mean-far-from-zero row (0) and the var≈eps row (1) are the two this test exists for.
    // The per-row maxima cover every element, so there is no separate whole-tensor assert.
    for r in 0..rows {
        let (lo, hi) = (r * dim, (r + 1) * dim);
        let e = maxerr(&c[lo..hi], &reference[lo..hi]);
        println!("  row {r} cpu-vs-ref max_err={e:e}");
        assert!(e < 1e-4, "LayerNorm row {r} diverges: max_err={e:e}");
    }

    if let Some(vk) = gpu() {
        let v = run(&vk, &g, &ins, &ws, dst, rows * dim);
        let e = maxerr(&c, &v);
        println!("LayerNorm cpu-vs-vulkan max_err={e:e}");
        assert!(e < 1e-4, "LayerNorm diverges on Vulkan: max_err={e:e}");
        let e = maxerr(&v, &reference);
        println!("LayerNorm vulkan-vs-ref max_err={e:e}");
        assert!(
            e < 1e-4,
            "LayerNorm Vulkan diverges from reference: max_err={e:e}"
        );
    }
}

// ── Op::LightningIndexer (deepseek32's top-k key selector) ───────────────────────────────────

/// Hand-written reference for one `Op::LightningIndexer` dispatch, derived from the FORMULA in
/// `docs/deepseek.md` § "The lightning indexer" (equivalently llama.cpp `deepseek32.cpp`'s
/// non-fused `// lightning indexer` block) — deliberately NOT transcribed from the CPU interpreter
/// arm, which is the thing under test:
///
/// ```text
/// score[t, j] = Σ_h (w[t, h] * scale) * ReLU( q[t, h] · k[j] )   for j <= pos + t
/// dst[t, :]   = the top_k key positions by (score DESC, index ASC)
/// ```
///
/// Two things it does differently from every backend on purpose. It accumulates in **f64**, so it
/// is an accuracy oracle and not a re-run of the same f32 rounding (which is why
/// `assert_scores_separated` exists: the two precisions may only be asked to agree about scores
/// that are either exactly equal or comfortably apart); and it takes `keys[j]` as the key for
/// position `j` from a list the caller builds, never touching the cache layout the backends read.
/// The ordering is expressed as a STABLE sort over the ascending index list, which is what makes
/// "ties break toward the lower index" fall out of the spec rather than out of a hand-written
/// comparison.
///
/// Returns the selected indices AND the f64 scores, so callers can check their case is well posed
/// (see `assert_scores_separated`).
#[allow(clippy::too_many_arguments)]
fn lightning_indexer_ref(
    q: &[f32],
    keys: &[Vec<f32>],
    w: &[f32],
    rows: usize,
    n_head: usize,
    head_dim: usize,
    kv_len: usize,
    top_k: usize,
    scale: f32,
    pos: usize,
) -> (Vec<u32>, Vec<Vec<f64>>) {
    let mut idx_out = Vec::with_capacity(rows * top_k);
    let mut score_out = Vec::with_capacity(rows);
    for t in 0..rows {
        // Causal: a key at an absolute position past the query's is not eligible; the cache only
        // holds `kv_len` positions.
        let hi = (pos + t + 1).min(kv_len);
        let mut sc = vec![0f64; kv_len];
        for (j, s) in sc.iter_mut().enumerate().take(hi) {
            let mut acc = 0f64;
            for h in 0..n_head {
                let qo = (t * n_head + h) * head_dim;
                let dot: f64 = (0..head_dim)
                    .map(|i| q[qo + i] as f64 * keys[j][i] as f64)
                    .sum();
                // ReLU INSIDE the head-weighted sum, and `scale` on the WEIGHT.
                acc += (w[t * n_head + h] as f64 * scale as f64) * dot.max(0.0);
            }
            *s = acc;
        }
        let mut order: Vec<usize> = (0..kv_len).collect();
        order.sort_by(|&a, &b| {
            let (ea, eb) = (a < hi, b < hi);
            // Eligible (true) before ineligible, then score descending among the eligible. The
            // sort is STABLE and `order` starts ascending, so every tie — and the whole
            // ineligible tail — keeps ascending index order, which IS the op's tie-break.
            eb.cmp(&ea).then_with(|| {
                if ea {
                    sc[b].partial_cmp(&sc[a]).expect("scores are never NaN")
                } else {
                    std::cmp::Ordering::Equal
                }
            })
        });
        idx_out.extend(order[..top_k].iter().map(|&j| j as u32));
        score_out.push(sc);
    }
    (idx_out, score_out)
}

/// A top-k over f32 scores only has ONE right answer when the eligible scores are either exactly
/// equal (a deliberate tie, which every precision reproduces) or far enough apart that f32 and the
/// f64 reference cannot disagree about their order. Assert that here rather than discover it as a
/// flake: `hi` is the case's causal bound for this row.
fn assert_scores_separated(sc: &[f64], hi: usize, what: &str) {
    for a in 0..hi {
        for b in (a + 1)..hi {
            let (x, y) = (sc[a], sc[b]);
            if x == y {
                continue; // an exact tie: decided by index at every precision
            }
            let rel = (x - y).abs() / x.abs().max(y.abs()).max(1e-12);
            assert!(
                rel > 1e-4,
                "{what}: keys {a}/{b} score {x} vs {y} (rel {rel:e}) — too close for f32 and the \
                 f64 reference to be guaranteed to agree on the order; the case is not well posed"
            );
        }
    }
}

/// One `lightning_indexer_parity` case.
struct LidxCase {
    name: &'static str,
    rows: usize,
    pos: usize,
    kv_len: usize,
    /// Ring row capacity — the K cache tensor is declared `cap * head_dim` wide, which is where
    /// the backends read `cap_rows` from. `cap < kv_len` is a genuinely wrapped cache.
    cap: usize,
    n_head: usize,
    head_dim: usize,
    top_k: usize,
}

/// Test data for `lightning_indexer_parity`. Values are 1/16ths so the f16 KV cache round-trip is
/// EXACT — the tolerance then measures the kernel, not the cast — and keys 2 and 5 are deliberately
/// IDENTICAL, so their scores tie exactly at every precision and the selection has to fall through
/// to the index tie-break.
fn lidx_key_at(j: usize, head_dim: usize) -> Vec<f32> {
    let src = if j == 5 { 2 } else { j }; // exact-tie pair
    (0..head_dim)
        .map(|d| (((src * 11 + d * 5) % 17) as f32 - 8.0) / 16.0)
        .collect()
}

/// `Op::LightningIndexer` — CPU backend vs the from-formula f64 reference above, plus a
/// CPU-vs-Vulkan cross-check when a GPU is present. Indices are discrete, so both comparisons are
/// EXACT equality: there is no tolerance to hide behind.
///
/// The case table moves the axes that decide the answer: several query rows so the causal cut
/// differs per row (and, at `pos = 0`, so the first rows have FEWER eligible keys than `top_k` —
/// the short case); a case where `top_k` exceeds the eligible count outright; the exact-tie pair
/// above; a wrapped ring (`cap < kv_len`), which only agrees with the reference if the kernels'
/// `j % cap_rows` lands on the row the writer used; and a wide case whose `kv_len` is not a
/// multiple of the 256-lane Vulkan/Metal workgroup (so the strided key loop runs a partial tail)
/// with an odd `n_head` (the head sum is serial inside a lane, so the head count never divides the
/// workgroup — the axis the workgroup splits is the KEY axis).
#[test]
fn lightning_indexer_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    let cases = [
        // pos=0 with top_k=3: row 0 has ONE eligible key, row 1 two, row 2 exactly three — the
        // short case on the first two rows and an exact fit on the third, in one dispatch.
        LidxCase {
            name: "prefill pos=0, causal cut per row (short on rows 0-1)",
            rows: 4,
            pos: 0,
            kv_len: 6,
            cap: 6,
            n_head: 3,
            head_dim: 8,
            top_k: 3,
        },
        // Decode at a position where the exact-tie pair (keys 2 and 5) is BOTH eligible and inside
        // the selected prefix.
        LidxCase {
            name: "decode pos=7, exact-tie pair eligible",
            rows: 1,
            pos: 7,
            kv_len: 8,
            cap: 8,
            n_head: 4,
            head_dim: 8,
            top_k: 6,
        },
        // top_k far past the eligible count: one eligible key, seven slots to fill from the
        // ineligible tail.
        LidxCase {
            name: "top_k 8 over 1 eligible key",
            rows: 1,
            pos: 0,
            kv_len: 8,
            cap: 8,
            n_head: 2,
            head_dim: 8,
            top_k: 8,
        },
        // The production cache shape: allocated for the whole context, only `kv_len` rows in use.
        // `cap != kv_len` is what tells the backends' `cap_rows` derivation apart from `kv_len`.
        LidxCase {
            name: "cache wider than kv_len (cap=32, kv_len=9)",
            rows: 2,
            pos: 7,
            kv_len: 9,
            cap: 32,
            n_head: 3,
            head_dim: 8,
            top_k: 4,
        },
        // kv_len 300 is not a multiple of 256, so the strided key loop runs a partial tail;
        // n_head 5 and head_dim 6 are both awkward widths for the serial inner loops.
        LidxCase {
            name: "wide kv_len=300 (not a workgroup multiple), n_head=5",
            rows: 3,
            pos: 296,
            kv_len: 300,
            cap: 300,
            n_head: 5,
            head_dim: 6,
            top_k: 17,
        },
    ];

    for case in cases {
        let LidxCase {
            name,
            rows,
            pos,
            kv_len,
            cap,
            n_head,
            head_dim,
            top_k,
        } = case;
        let scale = 1.0 / ((head_dim * n_head) as f32).sqrt();
        let keys: Vec<Vec<f32>> = (0..kv_len).map(|j| lidx_key_at(j, head_dim)).collect();
        // Cache writer: position j at row j, the layout the op requires (no ring fold — see
        // `Op::LightningIndexer`'s doc). Rows past kv_len stay zeroed and must never be read.
        assert!(cap >= kv_len, "{name}: the op refuses cap_rows < kv_len");
        let mut cache = vec![0f32; cap * head_dim];
        for (j, k) in keys.iter().enumerate() {
            cache[j * head_dim..][..head_dim].copy_from_slice(k);
        }
        // q and w: mixed-sign 1/8ths and 1/4ths. NEGATIVE weights matter — they are what makes the
        // ReLU's placement (inside the head sum, before the weight) observable at all.
        let qi: Vec<f32> = (0..rows * n_head * head_dim)
            .map(|i| (((i * 7 + 3) % 13) as f32 - 6.0) / 8.0)
            .collect();
        let wi: Vec<f32> = (0..rows * n_head)
            .map(|i| (((i * 5 + 1) % 9) as f32 - 4.0) / 4.0)
            .collect();

        let mut g = Graph::new();
        let q = g.input(f32d(rows * n_head * head_dim));
        let k_cache = g.input(TensorDesc::new(vec![cap * head_dim], DType::F16));
        let w = g.input(f32d(rows * n_head));
        let dst = g.output(TensorDesc::new(vec![rows * top_k], DType::I32));
        g.push(Op::LightningIndexer {
            q,
            k_cache,
            weights: w,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            top_k: top_k as u32,
            scale,
            pos: pos as u32,
        });

        // The cache is f16 (what `WriteKv` produces and what both GPU kernels read), so the shared
        // `run` helper — which uploads f32 — cannot carry it.
        let kf: Vec<u8> = cache
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let runner = |be: &dyn Backend| -> Vec<u32> {
            let plan = be.compile(&g).unwrap();
            let qb = be.alloc(qi.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(qb.as_ref(), bytemuck::cast_slice(&qi)).unwrap();
            let wb = be.alloc(wi.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
            let kb = be.alloc(kf.len(), BufferUsage::Activations).unwrap();
            be.upload(kb.as_ref(), &kf).unwrap();
            let ob = be.alloc(rows * top_k * 4, BufferUsage::Readback).unwrap();
            let mut b = Bindings::new();
            b.bind(q, qb.as_ref());
            b.bind(w, wb.as_ref());
            b.bind(k_cache, kb.as_ref());
            b.bind(dst, ob.as_ref());
            be.execute(plan.as_ref(), &b).unwrap();
            let mut bytes = vec![0u8; rows * top_k * 4];
            be.download(ob.as_ref(), &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
        };

        let (want, scores) = lightning_indexer_ref(
            &qi, &keys, &wi, rows, n_head, head_dim, kv_len, top_k, scale, pos,
        );
        for (t, sc) in scores.iter().enumerate() {
            assert_scores_separated(sc, (pos + t + 1).min(kv_len), &format!("{name} row {t}"));
        }
        let c = runner(&cpu);
        println!("LightningIndexer {name}: cpu={c:?}\n  ref ={want:?}");
        assert_eq!(
            c, want,
            "LightningIndexer {name}: CPU diverges from reference"
        );

        if let Some(vk) = gpu() {
            let v = runner(&vk);
            println!("LightningIndexer {name}: vulkan={v:?}");
            assert_eq!(v, c, "LightningIndexer {name}: Vulkan diverges from CPU");
        }
    }
}

/// The head-weighted score is a SUM over heads, not a max: a key with one big positive dot must
/// lose to a key that scores moderately in EVERY head. With `n_head = 4` unit queries (head `h`
/// selects component `h`) and unit weights:
///
/// * key 0 = `[9,0,0,0]` — dots `(9,0,0,0)`, so `max` is 9 and the SUM is 9;
/// * key 1 = `[3,3,3,3]` — dots `(3,3,3,3)`, so `max` is 3 and the SUM is 12.
///
/// The right answer ranks key 1 first. A max-over-heads (or any single-head) reduction ranks key 0
/// first, and — the reason this case is spelled out rather than left to the random table — so does
/// a reduction that only ever sees head 0.
#[test]
fn lightning_indexer_head_sum_is_a_sum_not_a_max() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n_head, head_dim, kv_len, top_k) = (1usize, 4usize, 4usize, 3usize, 3usize);
    // q[h] = e_h, so head h reads component h of the key.
    let mut qi = vec![0f32; rows * n_head * head_dim];
    for h in 0..n_head {
        qi[h * head_dim + h] = 1.0;
    }
    let wi = vec![1.0f32; rows * n_head];
    let keys: Vec<Vec<f32>> = vec![
        vec![9.0, 0.0, 0.0, 0.0], // one big head:  max 9, sum 9
        vec![3.0, 3.0, 3.0, 3.0], // every head:    max 3, sum 12
        vec![0.0, 0.0, 0.0, 0.0], // nothing:       max 0, sum 0
    ];
    let mut cache = vec![0f32; kv_len * head_dim];
    for (j, k) in keys.iter().enumerate() {
        cache[j * head_dim..][..head_dim].copy_from_slice(k);
    }

    let mut g = Graph::new();
    let q = g.input(f32d(rows * n_head * head_dim));
    let k_cache = g.input(TensorDesc::new(vec![kv_len * head_dim], DType::F16));
    let w = g.input(f32d(rows * n_head));
    let dst = g.output(TensorDesc::new(vec![rows * top_k], DType::I32));
    g.push(Op::LightningIndexer {
        q,
        k_cache,
        weights: w,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        top_k: top_k as u32,
        scale: 1.0,
        pos: (kv_len - 1) as u32, // every key eligible
    });

    let kf: Vec<u8> = cache
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let runner = |be: &dyn Backend| -> Vec<u32> {
        let plan = be.compile(&g).unwrap();
        let qb = be.alloc(qi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(qb.as_ref(), bytemuck::cast_slice(&qi)).unwrap();
        let wb = be.alloc(wi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let kb = be.alloc(kf.len(), BufferUsage::Activations).unwrap();
        be.upload(kb.as_ref(), &kf).unwrap();
        let ob = be.alloc(rows * top_k * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(q, qb.as_ref());
        b.bind(w, wb.as_ref());
        b.bind(k_cache, kb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut bytes = vec![0u8; rows * top_k * 4];
        be.download(ob.as_ref(), &mut bytes).unwrap();
        bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
    };

    let c = runner(&cpu);
    println!("LightningIndexer head-sum: cpu={c:?} (want [1, 0, 2])");
    assert_eq!(c, vec![1, 0, 2], "the head reduction is not a sum");
    if let Some(vk) = gpu() {
        let v = runner(&vk);
        println!("LightningIndexer head-sum: vulkan={v:?}");
        assert_eq!(v, c, "LightningIndexer head-sum: Vulkan diverges from CPU");
    }
}

/// `Op::LightningIndexer::scale` cannot be guarded through this op's output, and this test exists
/// to SAY so rather than let someone add a test that only looks like it does.
///
/// The op emits ranks, and multiplying every per-head weight by one positive constant multiplies
/// every score by that constant, which is order-preserving — so dropping the `1/sqrt(head_dim *
/// n_head)` normaliser, or moving it from the weight onto the score, leaves the selected indices
/// identical. (Verified by injection while writing this: removing the `* scale` from the CPU arm
/// left every case in `lightning_indexer_parity` green.) The field is still carried, and still
/// applied to the WEIGHT rather than the score, because that is where llama.cpp's `ggml_scale` on
/// `indexer_weights` puts it — which is what keeps the intermediate SCORES comparable with the
/// reference during bring-up, and the only thing that could ever make the placement observable is
/// a knife-edge tie that rounding collapses.
///
/// So: this asserts the invariance, not the arithmetic. It goes red only if the op stops being a
/// pure ranking (e.g. if it ever emitted scores), which is exactly when a real scale test would
/// become possible.
#[test]
fn lightning_indexer_scale_cannot_change_the_selection() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n_head, head_dim, kv_len, top_k, pos) = (2usize, 3usize, 8usize, 9usize, 5usize, 7);
    let keys: Vec<Vec<f32>> = (0..kv_len).map(|j| lidx_key_at(j, head_dim)).collect();
    let mut cache = vec![0f32; kv_len * head_dim];
    for (j, k) in keys.iter().enumerate() {
        cache[j * head_dim..][..head_dim].copy_from_slice(k);
    }
    let qi: Vec<f32> = (0..rows * n_head * head_dim)
        .map(|i| (((i * 7 + 3) % 13) as f32 - 6.0) / 8.0)
        .collect();
    let wi: Vec<f32> = (0..rows * n_head)
        .map(|i| (((i * 5 + 1) % 9) as f32 - 4.0) / 4.0)
        .collect();
    let kf: Vec<u8> = cache
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();

    let select = |scale: f32| -> Vec<u32> {
        let mut g = Graph::new();
        let q = g.input(f32d(rows * n_head * head_dim));
        let k_cache = g.input(TensorDesc::new(vec![kv_len * head_dim], DType::F16));
        let w = g.input(f32d(rows * n_head));
        let dst = g.output(TensorDesc::new(vec![rows * top_k], DType::I32));
        g.push(Op::LightningIndexer {
            q,
            k_cache,
            weights: w,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            top_k: top_k as u32,
            scale,
            pos: pos as u32,
        });
        let be: &dyn Backend = &cpu;
        let plan = be.compile(&g).unwrap();
        let qb = be.alloc(qi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(qb.as_ref(), bytemuck::cast_slice(&qi)).unwrap();
        let wb = be.alloc(wi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let kb = be.alloc(kf.len(), BufferUsage::Activations).unwrap();
        be.upload(kb.as_ref(), &kf).unwrap();
        let ob = be.alloc(rows * top_k * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(q, qb.as_ref());
        b.bind(w, wb.as_ref());
        b.bind(k_cache, kb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut bytes = vec![0u8; rows * top_k * 4];
        be.download(ob.as_ref(), &mut bytes).unwrap();
        bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
    };

    let normalised = select(1.0 / ((head_dim * n_head) as f32).sqrt());
    println!("LightningIndexer scale invariance: {normalised:?}");
    assert_eq!(normalised, select(1.0), "scale 1 changed the selection");
    assert_eq!(normalised, select(64.0), "scale 64 changed the selection");
}

// ── Op::Rope's NEOX pairing (deepseek32's lightning indexer) ─────────────────────────────────

/// Hand-written reference for one `Op::Rope` dispatch, from the DEFINITION of the two rope types
/// (llama.cpp `ggml_compute_forward_rope_f32`'s `is_neox` fork), not transcribed from the CPU arm.
///
/// Pair `p` (of `rope_dim/2`) rotates by `position * theta^(-2p/rope_dim)`, DIVIDED by `ff[p]` when
/// YaRN freq_factors are present; the two elements it rotates are `(2p, 2p+1)` for NORM and
/// `(p, p + rope_dim/2)` for NEOX. Dims at or past `rope_dim` pass through untouched in both.
///
/// `backward` is `ggml_rope_ext_back`: `ggml_compute_forward_rope_back` runs the SAME kernel with
/// `forward = false`, whose only effect is `sin_sign = -1` applied to the cached sine
/// (`ggml_rope_cache_init`) — `cos` untouched. See `Op::Rope::backward`.
#[allow(clippy::too_many_arguments)]
fn rope_ref(
    x: &[f32],
    positions: &[i32],
    rows: usize,
    n_head: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    neox: bool,
    freq_factors: Option<&[f32]>,
    backward: bool,
) -> Vec<f32> {
    let mut out = x.to_vec();
    let hf = rope_dim / 2;
    let sin_sign = if backward { -1.0f32 } else { 1.0 };
    for (r, &p0) in positions.iter().enumerate().take(rows) {
        for h in 0..n_head {
            let b = (r * n_head + h) * head_dim;
            for p in 0..hf {
                let (i0, i1) = if neox {
                    (p, p + hf)
                } else {
                    (2 * p, 2 * p + 1)
                };
                let mut ang = p0 as f32 * theta.powf(-2.0 * p as f32 / rope_dim as f32);
                if let Some(ff) = freq_factors {
                    ang /= ff[p];
                }
                let (s, c) = (ang.sin() * sin_sign, ang.cos());
                out[b + i0] = x[b + i0] * c - x[b + i1] * s;
                out[b + i1] = x[b + i0] * s + x[b + i1] * c;
            }
        }
    }
    out
}

/// `Op::Rope`'s two pairings, each against the from-definition reference above, on CPU and Vulkan.
///
/// `rope_dim < head_dim` so the pass-through tail is exercised, and `rope_dim/2` is odd so the NEOX
/// half-split does not coincide with any power-of-two lane boundary.
#[test]
fn rope_neox_and_norm_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, n_head, head_dim, rope_dim) = (3usize, 4usize, 24usize, 12usize);
    let theta = 10000.0f32;
    let xi = gen(rows * n_head * head_dim, 11);
    // CONSECUTIVE from a base: `rope.comp` derives each row's position as `pos_offset + row`,
    // which is what the seam always binds (one contiguous ubatch). A non-consecutive vector would
    // be testing something no caller can produce.
    let positions: Vec<i32> = (0..rows as i32).map(|i| i + 5).collect();

    let mut got = Vec::new();
    for neox in [false, true] {
        let mut g = Graph::new();
        let x = g.input(f32d(rows * n_head * head_dim));
        let pos = g.input(TensorDesc::new(vec![rows], DType::I32));
        let dst = g.output(f32d(rows * n_head * head_dim));
        g.push(Op::Rope {
            x,
            positions: pos,
            dst,
            rows: rows as u32,
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            theta,
            freq_factors: None,
            x_stride: 0,
            neox,
            backward: false,
        });
        // `run` uploads f32 for every bound input; the positions tensor is I32, so bind by hand.
        let runner = |be: &dyn Backend| -> Vec<f32> {
            let plan = be.compile(&g).unwrap();
            let xb = be.alloc(xi.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(xb.as_ref(), bytemuck::cast_slice(&xi)).unwrap();
            let pb = be.alloc(rows * 4, BufferUsage::Activations).unwrap();
            be.upload(pb.as_ref(), bytemuck::cast_slice(&positions))
                .unwrap();
            let ob = be.alloc(xi.len() * 4, BufferUsage::Readback).unwrap();
            let mut b = Bindings::new();
            b.bind(x, xb.as_ref());
            b.bind(pos, pb.as_ref());
            b.bind(dst, ob.as_ref());
            be.execute(plan.as_ref(), &b).unwrap();
            let mut bytes = vec![0u8; xi.len() * 4];
            be.download(ob.as_ref(), &mut bytes).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };
        let want = rope_ref(
            &xi, &positions, rows, n_head, head_dim, rope_dim, theta, neox, None, false,
        );
        let c = runner(&cpu);
        let e = maxerr(&c, &want);
        println!("Rope(neox={neox}) cpu-vs-ref max_err={e:e}");
        assert!(e < 1e-5, "Rope(neox={neox}) diverges from reference: {e:e}");
        if let Some(vk) = gpu() {
            let v = runner(&vk);
            let e = maxerr(&v, &want);
            println!("Rope(neox={neox}) vulkan-vs-ref max_err={e:e}");
            assert!(e < 1e-4, "Rope(neox={neox}) diverges on Vulkan: {e:e}");
        }
        got.push(c);
    }

    // The two pairings must not be interchangeable — the failure mode this whole field exists for
    // is a port that picks the wrong one, which raises nothing and merely rotates other elements.
    // The pass-through tail is identical in both, so compare only the rotated slice.
    let mut worst = 0f32;
    for r in 0..rows {
        for h in 0..n_head {
            let b = (r * n_head + h) * head_dim;
            for i in 0..rope_dim {
                worst = worst.max((got[0][b + i] - got[1][b + i]).abs());
            }
            for i in rope_dim..head_dim {
                assert_eq!(
                    got[0][b + i],
                    got[1][b + i],
                    "the un-rotated tail must not depend on the pairing"
                );
            }
        }
    }
    println!("Rope NORM vs NEOX: max|Δ| over the rotated slice = {worst:e}");
    assert!(
        worst > 1e-2,
        "NORM and NEOX produced the same rotation — one of the two pairings is not being applied"
    );
}

// ── Op::TopkMask (the indexer's top-k → the MLA score mask) ──────────────────────────────────

/// `Op::TopkMask` on CPU and Vulkan, driven by a REAL `Op::LightningIndexer` in the same graph.
///
/// Chained rather than fed a hand-made index tensor on purpose: the indices travel as i32 words in
/// an Internal handle (`Op::Argmax`'s carrier convention), and a standalone fixture would have to
/// fake that carrier — which is exactly the part a wrong reading would get wrong. Here the producer
/// and the consumer are the two real ops, and the expected mask is derived from
/// [`lightning_indexer_ref`], which knows nothing about either implementation.
#[test]
fn topk_mask_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    // kv_len 300 is not a multiple of the 256-lane Vulkan workgroup, so the mask's fill loop runs a
    // partial tail; `pos` puts the causal cut inside the row so the tail of the selection comes
    // from the ineligible keys (whose mask slots the MLA loop never reads, but which must still be
    // written somewhere legal).
    let (rows, n_head, head_dim, kv_len, top_k, pos) =
        (3usize, 3usize, 8usize, 300usize, 7usize, 294usize);
    let scale = 1.0 / ((head_dim * n_head) as f32).sqrt();
    let keys: Vec<Vec<f32>> = (0..kv_len).map(|j| lidx_key_at(j, head_dim)).collect();
    let mut cache = vec![0f32; kv_len * head_dim];
    for (j, k) in keys.iter().enumerate() {
        cache[j * head_dim..][..head_dim].copy_from_slice(k);
    }
    let qi: Vec<f32> = (0..rows * n_head * head_dim)
        .map(|i| (((i * 7 + 3) % 13) as f32 - 6.0) / 8.0)
        .collect();
    let wi: Vec<f32> = (0..rows * n_head)
        .map(|i| (((i * 5 + 1) % 9) as f32 - 4.0) / 4.0)
        .collect();

    let mut g = Graph::new();
    let q = g.input(f32d(rows * n_head * head_dim));
    let k_cache = g.input(TensorDesc::new(vec![kv_len * head_dim], DType::F16));
    let w = g.input(f32d(rows * n_head));
    let idx = g.internal(TensorDesc::new(vec![rows * top_k], DType::I32));
    let dst = g.output(f32d(rows * kv_len));
    g.push(Op::LightningIndexer {
        q,
        k_cache,
        weights: w,
        dst: idx,
        rows: rows as u32,
        kv_len: kv_len as u32,
        n_head: n_head as u32,
        head_dim: head_dim as u32,
        top_k: top_k as u32,
        scale,
        pos: pos as u32,
    });
    g.push(Op::TopkMask {
        idx,
        dst,
        rows: rows as u32,
        kv_len: kv_len as u32,
        top_k: top_k as u32,
    });

    let kf: Vec<u8> = cache
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let runner = |be: &dyn Backend| -> Vec<f32> {
        let plan = be.compile(&g).unwrap();
        let qb = be.alloc(qi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(qb.as_ref(), bytemuck::cast_slice(&qi)).unwrap();
        let wb = be.alloc(wi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(wb.as_ref(), bytemuck::cast_slice(&wi)).unwrap();
        let kb = be.alloc(kf.len(), BufferUsage::Activations).unwrap();
        be.upload(kb.as_ref(), &kf).unwrap();
        let ob = be.alloc(rows * kv_len * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(q, qb.as_ref());
        b.bind(w, wb.as_ref());
        b.bind(k_cache, kb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).unwrap();
        let mut bytes = vec![0u8; rows * kv_len * 4];
        be.download(ob.as_ref(), &mut bytes).unwrap();
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    };

    // Expected mask, from the indexer's own from-formula reference: 0.0 at each selected key,
    // -inf everywhere else.
    let (sel, _) = lightning_indexer_ref(
        &qi, &keys, &wi, rows, n_head, head_dim, kv_len, top_k, scale, pos,
    );
    let mut want = vec![f32::NEG_INFINITY; rows * kv_len];
    for r in 0..rows {
        for s in 0..top_k {
            want[r * kv_len + sel[r * top_k + s] as usize] = 0.0;
        }
    }
    // The check is only meaningful if the mask really is mostly -inf.
    let zeros = want.iter().filter(|v| **v == 0.0).count();
    println!("TopkMask: {zeros} selected slots of {}", rows * kv_len);
    assert_eq!(
        zeros,
        rows * top_k,
        "the reference selection has duplicate indices — the fixture, not the op, is wrong"
    );

    let c = runner(&cpu);
    assert_eq!(c, want, "TopkMask: CPU diverges from the contract");
    if let Some(vk) = gpu() {
        let v = runner(&vk);
        assert_eq!(v, c, "TopkMask: Vulkan diverges from CPU");
    }
}

// ── Op::Mla's optional top-k score mask ──────────────────────────────────────────────────────

/// `Op::Mla::key_bias` really removes the masked keys, on CPU and Vulkan.
///
/// Two runs of the SAME dispatch: one over `kv_len = 3` keys with a bias that `-inf`s key 1, and
/// one over a 2-key cache holding only keys 0 and 2 (at their own positions, via a `Canvas` span
/// that would otherwise attend everything). Masking a key must be exactly equivalent to the key
/// not being there — `exp(-inf - max) == 0` contributes nothing to either the softmax denominator
/// or the V accumulation.
///
/// It also asserts the masked run DIFFERS from the unmasked one over the same three keys: without
/// that, a `key_bias` the kernel silently ignored would pass the first half by construction.
#[test]
fn mla_key_bias_removes_the_masked_keys() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, kv_lora, qk_nope, qk_rope, vhd) =
        (2usize, 2usize, 4usize, 2usize, 2usize, 3usize);
    let key_len = kv_lora + qk_rope;
    let q_head_dim = qk_nope + qk_rope;
    let (theta, scale) = (10000.0f32, 1.0 / (q_head_dim as f32).sqrt());
    let qi = gen(rows * nh * q_head_dim, 5);
    let wk = gen(nh * kv_lora * qk_nope, 6);
    let wv = gen(nh * kv_lora * vhd, 7);
    // Three logical keys; the middle one is the one the mask removes.
    // Key 1 — the one the mask removes — is scaled up so it DOMINATES the softmax: removing a key
    // that barely contributed would make "the output changed" a tolerance argument instead of an
    // observation.
    let keys: Vec<Vec<f32>> = (0..3)
        .map(|j| {
            let s = if j == 1 { 4.0 } else { 1.0 };
            gen(key_len, 40 + j).iter().map(|v| v * s).collect()
        })
        .collect();

    // `keep` selects which of `keys` the cache holds; `bias` is the per-(row, key) mask (empty =
    // no mask). Both runs use a Canvas mask over the whole cache so every row sees every key —
    // the ONLY thing that removes a key is the bias.
    let run_mla = |be: &dyn Backend, cache_keys: &[Vec<f32>], bias: Option<&[f32]>| -> Vec<f32> {
        let kv_len = cache_keys.len();
        let mut g = Graph::new();
        let q = g.input(f32d(rows * nh * q_head_dim));
        let k_cache = g.input(TensorDesc::new(vec![kv_len * key_len], DType::F16));
        let wk_b = g.weight(f32d(nh * kv_lora * qk_nope));
        let wv_b = g.weight(f32d(nh * kv_lora * vhd));
        let kb = bias.map(|_| g.input(f32d(rows * kv_len)));
        let dst = g.output(f32d(rows * nh * vhd));
        g.push(Op::Mla {
            q,
            k_cache,
            wk_b,
            wv_b,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            q_head_dim: q_head_dim as u32,
            kv_lora_rank: kv_lora as u32,
            qk_nope_dim: qk_nope as u32,
            qk_rope_dim: qk_rope as u32,
            v_head_dim: vhd as u32,
            scale,
            mask: AttnMask::Canvas { lo: 0 },
            pos: 0,
            theta,
            freq_factors: None,
            key_bias: kb,
        });
        let flat: Vec<f32> = cache_keys.iter().flatten().copied().collect();
        let kf: Vec<u8> = flat
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let plan = be.compile(&g).unwrap();
        let qb = be.alloc(qi.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(qb.as_ref(), bytemuck::cast_slice(&qi)).unwrap();
        let kcb = be.alloc(kf.len(), BufferUsage::Activations).unwrap();
        be.upload(kcb.as_ref(), &kf).unwrap();
        let wkb = be.alloc(wk.len() * 4, BufferUsage::Weights).unwrap();
        be.upload(wkb.as_ref(), bytemuck::cast_slice(&wk)).unwrap();
        let wvb = be.alloc(wv.len() * 4, BufferUsage::Weights).unwrap();
        be.upload(wvb.as_ref(), bytemuck::cast_slice(&wv)).unwrap();
        let ob = be
            .alloc(rows * nh * vhd * 4, BufferUsage::Readback)
            .unwrap();
        let mut b = Bindings::new();
        b.bind(q, qb.as_ref());
        b.bind(k_cache, kcb.as_ref());
        b.bind(wk_b, wkb.as_ref());
        b.bind(wv_b, wvb.as_ref());
        b.bind(dst, ob.as_ref());
        let bb = bias.map(|bv| {
            let buf = be.alloc(bv.len() * 4, BufferUsage::Activations).unwrap();
            be.upload(buf.as_ref(), bytemuck::cast_slice(bv)).unwrap();
            buf
        });
        if let (Some(id), Some(buf)) = (kb, bb.as_ref()) {
            b.bind(id, buf.as_ref());
        }
        be.execute(plan.as_ref(), &b).unwrap();
        let mut bytes = vec![0u8; rows * nh * vhd * 4];
        be.download(ob.as_ref(), &mut bytes).unwrap();
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    };

    // Mask key 1 out of the 3-key cache; the 2-key cache is the same computation with key 1 absent.
    let ninf = f32::NEG_INFINITY;
    let bias3: Vec<f32> = (0..rows).flat_map(|_| [0.0, ninf, 0.0]).collect();
    let kept: Vec<Vec<f32>> = vec![keys[0].clone(), keys[2].clone()];
    for (name, be) in [("cpu", &cpu as &dyn Backend)]
        .into_iter()
        .chain(gpu().iter().map(|vk| ("vulkan", vk as &dyn Backend)))
    {
        let masked = run_mla(be, &keys, Some(&bias3));
        let subset = run_mla(be, &kept, None);
        let e = maxerr(&masked, &subset);
        println!("Mla key_bias {name}: masked-vs-subset max_err={e:e}");
        assert!(
            e < 1e-4,
            "Mla {name}: a -inf-masked key still influenced the output (max_err={e:e})\n  \
             masked={masked:?}\n  subset={subset:?}"
        );
        let unmasked = run_mla(be, &keys, None);
        let d = maxerr(&masked, &unmasked);
        println!("Mla key_bias {name}: masked-vs-unmasked max|Δ|={d:e}");
        assert!(
            d > 1e-3,
            "Mla {name}: masking a key changed nothing — key_bias is not reaching the kernel"
        );
    }
}

// ── DeepSeek V4 attention primitives (docs/deepseek.md § Stage 4) ─────────────────────────────
//
// Four op-level capabilities, each with a reference written from the DEFINITION (llama.cpp's
// `deepseek4.cpp` / `ggml`), in f64, deliberately NOT transcribed from the interpreter arms under
// test. Nothing emits any of them yet.

/// Hand-written reference for `Op::QkNorm { weight: None }` — DeepSeek V4's Q norm, which is a bare
/// `ggml_rms_norm` over a `[head_dim, n_head, n_tokens]` reshape (`deepseek4.cpp`'s `build_attention`,
/// the `q_norm` callback). `ggml_rms_norm` normalizes over `ne[0]`, so the reduction is PER HEAD:
/// `out = x / sqrt(mean_head(x²) + eps)`, no weight anywhere. f64, from the definition.
fn head_rmsnorm_ref(x: &[f32], rows: usize, n_head: usize, head_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * n_head * head_dim];
    for hh in 0..rows * n_head {
        let b = hh * head_dim;
        let ss: f64 = (0..head_dim)
            .map(|i| (x[b + i] as f64).powi(2))
            .sum::<f64>()
            / head_dim as f64;
        let s = 1.0 / (ss + eps as f64).sqrt();
        for i in 0..head_dim {
            out[b + i] = (x[b + i] as f64 * s) as f32;
        }
    }
    out
}

/// The MISTAKE this test exists to catch: one reduction over the whole `n_head*head_dim` row
/// instead of one per head. Same formula, wrong `dim`.
fn row_rmsnorm_ref(x: &[f32], rows: usize, n_head: usize, head_dim: usize, eps: f32) -> Vec<f32> {
    head_rmsnorm_ref(x, rows, 1, n_head * head_dim, eps)
}

/// Rows whose per-head vectors have WILDLY different magnitudes (1e2 / 1e-2 / 1e0 / 3e1). A norm
/// taken across the whole row is dominated by head 0 and crushes heads 1-3 toward zero, so the two
/// references are far apart — which is what makes this input able to fail.
fn head_scale_rows(rows: usize, n_head: usize, head_dim: usize) -> Vec<f32> {
    let mag = [100.0f32, 0.01, 1.0, 30.0];
    (0..rows * n_head * head_dim)
        .map(|i| {
            let h = (i / head_dim) % n_head;
            let c = i % head_dim;
            mag[h % mag.len()] * ((((c * 7 + h * 3) % 13) as f32 - 6.0) * 0.15 + 1.0)
        })
        .collect()
}

/// `Op::QkNorm` with NO weight (V4's Q norm) normalizes PER HEAD, on CPU and on Vulkan.
///
/// The input's four heads span four orders of magnitude, so the whole-row reduction — the one
/// plausible wrong answer — is not merely different, it is off by orders of magnitude on three of
/// the four heads. That gap is asserted explicitly (`well-posed`), so the test cannot pass by
/// comparing two things that happen to agree.
#[test]
fn qknorm_weightless_normalizes_per_head() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, hd) = (2usize, 4usize, 16usize);
    let eps = 1e-6f32;
    let n = rows * nh * hd;

    let mut g = Graph::new();
    let x = g.input(f32d(n));
    let dst = g.output(f32d(n));
    g.push(Op::QkNorm {
        x,
        weight: None,
        dst,
        rows: rows as u32,
        n_head: nh as u32,
        head_dim: hd as u32,
        eps,
        x_stride: 0,
    });

    let xi = head_scale_rows(rows, nh, hd);
    let ins = [(x, &xi[..])];
    let per_head = head_rmsnorm_ref(&xi, rows, nh, hd, eps);
    let per_row = row_rmsnorm_ref(&xi, rows, nh, hd, eps);
    let gap = maxerr(&per_head, &per_row);
    println!("QkNorm(weightless) per-head-vs-per-row reference gap={gap:e}");
    assert!(
        gap > 0.5,
        "input is not well posed: a whole-row norm would pass this test (gap={gap:e})"
    );

    let c = run(&cpu, &g, &ins, &[], dst, n);
    let e = maxerr(&c, &per_head);
    println!("QkNorm(weightless) cpu-vs-ref max_err={e:e}");
    assert!(e < 1e-5, "weightless QkNorm diverges on CPU: max_err={e:e}");

    if let Some(vk) = gpu() {
        let v = run(&vk, &g, &ins, &[], dst, n);
        let e = maxerr(&v, &per_head);
        println!("QkNorm(weightless) vulkan-vs-ref max_err={e:e}");
        assert!(
            e < 1e-5,
            "weightless QkNorm diverges on Vulkan: max_err={e:e}"
        );
    }
}

/// A weightless `Op::QkNorm` must equal a weighted one whose weight is all ones — the convention
/// `Op::RmsNorm`'s doc records, and the reason `weight: None` is a REPRESENTATION change and not a
/// numerics change. Pins that both arms compute the same thing on every backend.
#[test]
fn qknorm_weightless_matches_a_ones_weight() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, hd) = (2usize, 4usize, 16usize);
    let eps = 1e-6f32;
    let n = rows * nh * hd;
    let xi = head_scale_rows(rows, nh, hd);
    let ones = vec![1.0f32; hd];

    let build = |weighted: bool| {
        let mut g = Graph::new();
        let x = g.input(f32d(n));
        let w = g.weight(f32d(hd));
        let dst = g.output(f32d(n));
        g.push(Op::QkNorm {
            x,
            weight: weighted.then_some(w),
            dst,
            rows: rows as u32,
            n_head: nh as u32,
            head_dim: hd as u32,
            eps,
            x_stride: 0,
        });
        (g, x, w, dst)
    };
    let each = |be: &dyn Backend, name: &str| {
        let (gw, xw, ww, dw) = build(true);
        let with = run(be, &gw, &[(xw, &xi)], &[(ww, &ones)], dw, n);
        let (gn, xn, _, dn) = build(false);
        let without = run(be, &gn, &[(xn, &xi)], &[], dn, n);
        let e = maxerr(&with, &without);
        println!("QkNorm ones-weight vs weightless ({name}) max_err={e:e}");
        assert!(
            e == 0.0,
            "{name}: weightless QkNorm is not x*1.0 (err={e:e})"
        );
    };
    each(&cpu, "cpu");
    if let Some(vk) = gpu() {
        each(&vk, "vulkan");
    }
}

// ── Op::Attention sinks (deepseek4's attn_sinks) ─────────────────────────────────────────────

/// Hand-written softmax attention with per-head SINKS, in f64, from
/// `ggml_compute_forward_soft_max_f32`'s `src2` handling (llama.cpp `ggml/src/ggml-cpu/ops.cpp`):
///
/// ```text
/// m = max_j(score[j]);  if sinks: m = max(m, sink[h])
/// l = Σ_j exp(score[j] - m);  if sinks: l += exp(sink[h] - m)
/// out = Σ_j (exp(score[j] - m) / l) * V[j]
/// ```
///
/// `sink` is `None` for the plain softmax, `Some((s, extra_value))` otherwise: `extra_value` is the
/// deliberate WRONG variant — the sink also contributing a value row (`Σ` gains `exp(sink-m)/l *
/// V[extra]`), which is what "attention sinks" means in the register-token reading and is a
/// different function from the one llama.cpp implements. The correct call passes `None` for it.
#[allow(clippy::too_many_arguments)]
fn attention_sinks_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    kv_len: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    scale: f32,
    pos: usize,
    sink: Option<(&[f32], Option<usize>)>,
) -> Vec<f32> {
    let group = n_head / n_kv;
    let mut out = vec![0f32; rows * n_head * hd];
    for ti in 0..rows {
        for h in 0..n_head {
            let kvh = h / group;
            let qb = (ti * n_head + h) * hd;
            let hi = (pos + ti + 1).min(kv_len);
            let sc: Vec<f64> = (0..hi)
                .map(|j| {
                    let kb = (j * n_kv + kvh) * hd;
                    (0..hd)
                        .map(|d| q[qb + d] as f64 * k[kb + d] as f64)
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let mut m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if let Some((s, _)) = sink {
                m = m.max(s[h] as f64);
            }
            let mut l: f64 = sc.iter().map(|&s| (s - m).exp()).sum();
            if let Some((s, _)) = sink {
                l += (s[h] as f64 - m).exp();
            }
            for (j, &s) in sc.iter().enumerate() {
                let p = (s - m).exp() / l;
                let vb = (j * n_kv + kvh) * hd;
                for d in 0..hd {
                    out[qb + d] += (p * v[vb + d] as f64) as f32;
                }
            }
            // The wrong variant: the sink ALSO carries a value.
            if let Some((s, Some(extra))) = sink {
                let p = (s[h] as f64 - m).exp() / l;
                let vb = (extra * n_kv + kvh) * hd;
                for d in 0..hd {
                    out[qb + d] += (p * v[vb + d] as f64) as f32;
                }
            }
        }
    }
    out
}

/// `Op::Attention`'s sinks join the softmax MAX and DENOMINATOR only — never the numerator.
///
/// Two regimes, because they fail differently:
///
/// * **Dominant sink** (`+18`, far above every real score): every real key's weight collapses
///   toward `exp(-18)`, so the output is ~1e-8 of the sink-free one. This is the case that catches
///   "sink left out of the denominator" (which would return the sink-free output outright) and
///   "sink also contributes a value" (which would return ≈`V[0]`). Both wrong answers are asserted
///   to be far from the right one, so the test provably discriminates.
/// * **Negligible sink** (`-18`): `exp(sink - m) ≈ 0`, so the output must be the sink-free one to
///   within f32 noise. This is the case that catches a sink applied with the wrong sign or scaled
///   by `scale`.
///
/// q and the KV cache are f16 — the seam's real producer→consumer dtype flow, and what the Vulkan
/// `attention_kv` family reads (`qknormrope_attn_chain` above pins the same convention). The CPU
/// interpreter converts them to f32 on load, so both backends see identical values.
#[test]
fn attention_sinks_are_denominator_only() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, nkv, hd) = (3usize, 2usize, 1usize, 8usize);
    let kv_len = rows;
    let scale = 1.0 / (hd as f32).sqrt();
    let n_out = rows * nh * hd;

    let to_f16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect()
    };
    let deq = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    };
    let qf = to_f16(&gen(rows * nh * hd, 4));
    let kf = to_f16(&gen(kv_len * nkv * hd, 8));
    let vf = to_f16(&gen(kv_len * nkv * hd, 9));
    // The references must see the SAME f16-rounded values the kernels read.
    let (qd, kd, vd) = (deq(&qf), deq(&kf), deq(&vf));

    let build = |with_sinks: bool| {
        let mut g = Graph::new();
        let q = g.input(TensorDesc::new(vec![rows * nh * hd], DType::F16));
        let kc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
        let vc = g.input(TensorDesc::new(vec![kv_len * nkv * hd], DType::F16));
        let sk = g.weight(f32d(nh));
        let dst = g.output(f32d(n_out));
        g.push(Op::Attention {
            q,
            k_cache: kc,
            v_cache: vc,
            dst,
            rows: rows as u32,
            kv_len: kv_len as u32,
            n_head: nh as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            scale,
            mask: AttnMask::Causal,
            pos: 0,
            sinks: with_sinks.then_some(sk),
        });
        (g, q, kc, vc, sk, dst)
    };

    // Bespoke runner: q/K/V are f16 BYTES, which `run` (f32 slices) cannot upload.
    let go = |be: &dyn Backend, sinks: Option<&[f32]>| -> Vec<f32> {
        let (g, q, kc, vc, sk, dst) = build(sinks.is_some());
        let plan = be.compile(&g).expect("compile");
        let up = |bytes: &[u8], usage| {
            let b = be.alloc(bytes.len(), usage).expect("alloc");
            be.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let qb = up(&qf, BufferUsage::Activations);
        let kb = up(&kf, BufferUsage::KvCache);
        let vb = up(&vf, BufferUsage::KvCache);
        let sb = sinks.map(|s| up(bytemuck::cast_slice(s), BufferUsage::Weights));
        let ob = be.alloc(n_out * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(q, qb.as_ref());
        b.bind(kc, kb.as_ref());
        b.bind(vc, vb.as_ref());
        if let Some(sb) = &sb {
            b.bind(sk, sb.as_ref());
        }
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).expect("execute");
        let mut o = vec![0f32; n_out];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };

    let r = |sink: Option<(&[f32], Option<usize>)>| {
        attention_sinks_ref(&qd, &kd, &vd, rows, kv_len, nh, nkv, hd, scale, 0, sink)
    };
    let no_sink = r(None);
    let dominant = vec![18.0f32; nh];
    let negligible = vec![-18.0f32; nh];
    let want_dom = r(Some((&dominant, None)));
    let want_neg = r(Some((&negligible, None)));
    // The two wrong answers, computed from the same reference with one clause changed.
    let dom_no_denom = no_sink.clone(); // sink dropped from the denominator
    let dom_with_value = r(Some((&dominant, Some(0)))); // sink also contributes V[0]

    let gap_denom = maxerr(&want_dom, &dom_no_denom);
    let gap_numer = maxerr(&want_dom, &dom_with_value);
    println!("Attention sinks: dominant-vs-no-denominator gap={gap_denom:e}");
    println!("Attention sinks: dominant-vs-sink-has-value gap={gap_numer:e}");
    assert!(
        gap_denom > 0.1,
        "input not well posed: dropping the sink from the denominator would pass (gap={gap_denom:e})"
    );
    assert!(
        gap_numer > 0.1,
        "input not well posed: giving the sink a value row would pass (gap={gap_numer:e})"
    );

    let check = |be: &dyn Backend, name: &str| {
        let plain = go(be, None);
        let e = maxerr(&plain, &no_sink);
        println!("Attention sinks({name}) none-vs-ref max_err={e:e}");
        assert!(e < 1e-3, "{name}: sink-free attention moved: max_err={e:e}");

        let dom = go(be, Some(&dominant));
        let e = maxerr(&dom, &want_dom);
        println!("Attention sinks({name}) dominant-vs-ref max_err={e:e}");
        assert!(e < 1e-4, "{name}: dominant sink wrong: max_err={e:e}");
        // ...and it is genuinely a different answer from both wrong variants on this backend.
        assert!(
            maxerr(&dom, &dom_no_denom) > 0.1,
            "{name}: dominant-sink output equals the sink-free one — the sink never reached the \
             denominator"
        );
        assert!(
            maxerr(&dom, &dom_with_value) > 0.1,
            "{name}: dominant-sink output equals the sink-has-a-value one — the sink is in the \
             numerator"
        );

        let neg = go(be, Some(&negligible));
        let e = maxerr(&neg, &want_neg);
        println!("Attention sinks({name}) negligible-vs-ref max_err={e:e}");
        assert!(e < 1e-3, "{name}: negligible sink wrong: max_err={e:e}");
        let e = maxerr(&neg, &no_sink);
        println!("Attention sinks({name}) negligible-vs-sinkless max_err={e:e}");
        assert!(
            e < 1e-4,
            "{name}: a sink 18 below the max changed the output: max_err={e:e}"
        );
    };
    check(&cpu, "cpu");
    if let Some(vk) = gpu() {
        check(&vk, "vulkan");
    }
}

// ── Op::Rope { backward } (deepseek4's attention-output de-rope) ──────────────────────────────

/// De-roping is the exact inverse of roping: `Rope { backward: false }` then
/// `Rope { backward: true }` at the SAME position/theta/freq_factors returns the input.
///
/// That is the property, so it is what is asserted — and it is a property only because
/// `Op::Rope` carries no magnitude scale (see `Op::Rope::backward`: ggml's own `rope_back` is the
/// transpose, not the inverse, whenever YaRN's `mscale != 1`, and V4's `dsv4_rope_attn_factor`
/// is precisely the constant that makes `mscale == 1` at every one of its call sites).
///
/// The roundtrip alone would also pass if BOTH directions were no-ops, so the backward leg is
/// additionally compared against the f64 reference and asserted to differ from the forward leg.
/// Positions are non-trivial (37/38/39) and `rope_dim < head_dim`, so the pass-through tail is
/// exercised too.
#[test]
fn rope_back_inverts_rope_forward() {
    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nh, hd, rope_dim) = (3usize, 2usize, 8usize, 4usize);
    let theta = 1e4f32;
    let n = rows * nh * hd;
    let xi = gen(n, 11);
    // Non-trivial positions: 37/38/39, not 0/1/2 (a de-rope at position 0 is the identity, so a
    // dropped sign would pass unnoticed there).
    let posv: Vec<i32> = vec![37, 38, 39];
    // YaRN per-pair divisors: V4's compressed layers rope (and de-rope) with a ramp, its ratio-0
    // layers plain. Both must invert, so both are run.
    let ffi: Vec<f32> = (0..rope_dim / 2).map(|p| 1.0 + p as f32 * 0.37).collect();

    for (name, use_ff) in [("plain", false), ("freq_factors", true)] {
        let rope = |backward: bool| {
            let mut g = Graph::new();
            let x = g.input(f32d(n));
            let p = g.input(TensorDesc::new(vec![rows], DType::I32));
            let ff = g.input(f32d(rope_dim / 2));
            let mid = g.internal(f32d(n));
            let dst = g.output(f32d(n));
            g.push(Op::Rope {
                x,
                positions: p,
                dst: if backward { mid } else { dst },
                rows: rows as u32,
                n_head: nh as u32,
                head_dim: hd as u32,
                rope_dim: rope_dim as u32,
                theta,
                freq_factors: use_ff.then_some(ff),
                x_stride: 0,
                neox: false,
                backward: false,
            });
            if backward {
                g.push(Op::Rope {
                    x: mid,
                    positions: p,
                    dst,
                    rows: rows as u32,
                    n_head: nh as u32,
                    head_dim: hd as u32,
                    rope_dim: rope_dim as u32,
                    theta,
                    freq_factors: use_ff.then_some(ff),
                    x_stride: 0,
                    neox: false,
                    backward: true,
                });
            }
            (g, x, p, ff, dst)
        };
        // A standalone backward rope, for the direct comparison against the reference.
        let back_only = || {
            let mut g = Graph::new();
            let x = g.input(f32d(n));
            let p = g.input(TensorDesc::new(vec![rows], DType::I32));
            let ff = g.input(f32d(rope_dim / 2));
            let dst = g.output(f32d(n));
            g.push(Op::Rope {
                x,
                positions: p,
                dst,
                rows: rows as u32,
                n_head: nh as u32,
                head_dim: hd as u32,
                rope_dim: rope_dim as u32,
                theta,
                freq_factors: use_ff.then_some(ff),
                x_stride: 0,
                neox: false,
                backward: true,
            });
            (g, x, p, ff, dst)
        };
        // `positions` is an I32 input; `run` uploads f32 words, so bind the bit-patterns.
        let posi: Vec<f32> = posv.iter().map(|&p| f32::from_bits(p as u32)).collect();
        let ff_used = use_ff.then_some(&ffi[..]);
        let fwd_ref = rope_ref(
            &xi, &posv, rows, nh, hd, rope_dim, theta, false, ff_used, false,
        );
        let back_ref = rope_ref(
            &xi, &posv, rows, nh, hd, rope_dim, theta, false, ff_used, true,
        );
        let sep = maxerr(&fwd_ref, &back_ref);
        println!("Rope({name}) forward-vs-backward reference separation={sep:e}");
        assert!(
            sep > 0.01,
            "{name}: input not well posed — forward and backward agree (sep={sep:e})"
        );

        let check = |be: &dyn Backend, bname: &str| {
            let (g, x, p, ff, dst) = back_only();
            let b = run(be, &g, &[(x, &xi), (p, &posi), (ff, &ffi)], &[], dst, n);
            let e = maxerr(&b, &back_ref);
            println!("Rope back({name},{bname}) vs ref max_err={e:e}");
            assert!(e < 1e-5, "{bname} {name}: backward rope wrong: {e:e}");
            let e = maxerr(&b, &fwd_ref);
            assert!(
                e > 0.01,
                "{bname} {name}: backward rope equals the forward one — the sign flip never landed"
            );

            let (g, x, p, ff, dst) = rope(true);
            let rt = run(be, &g, &[(x, &xi), (p, &posi), (ff, &ffi)], &[], dst, n);
            let e = maxerr(&rt, &xi);
            println!("Rope roundtrip({name},{bname}) vs input max_err={e:e}");
            assert!(
                e < 1e-5,
                "{bname} {name}: forward∘backward is not the identity: max_err={e:e}"
            );
        };
        check(&cpu, "cpu");
        if let Some(vk) = gpu() {
            check(&vk, "vulkan");
        }
    }
}

// ── The grouped low-rank output projection (deepseek4's wo_a/wo_b) ───────────────────────────
//
// NO new op: `deepseek4.cpp` reshapes the (de-roped) attention output to `[o_group_dim, n_groups,
// nt]`, permutes, and runs ONE batched `ggml_mul_mat` against `wo_a` reshaped to `[o_group_dim,
// o_lora_rank, n_groups]`. Because that batch axis is the OUTERMOST axis of both operands, group
// `g` is exactly `Op::Linear` over rows `[g*o_lora_rank, (g+1)*o_lora_rank)` of `wo_a` — which is
// what `Op::Linear::w_off` already selects (`w_off = g*o_lora_rank*o_group_dim`, row-aligned) —
// applied to columns `[g*o_group_dim, (g+1)*o_group_dim)` of the output row, which is what
// `Op::CopyStrided` already slices. So the composition below IS the batched matmul, built out of
// two ops the seam already emits for exactly this shape of job (qwen35 splits its interleaved q|k|v
// rows the same way). A batched-GEMM op would have one caller and one shape.

/// Hand-written reference for the grouped projection, in f64, from `deepseek4.cpp`'s
/// `attn_wo_a` block: for each token row and each group `g`,
/// `oa[r, g*o_lora_rank + i] = Σ_d out[r, g*o_group_dim + d] * wo_a[g][i, d]`, then the plain
/// `wo_b` Linear over the concatenated `[nt, o_lora_rank*n_groups]`.
#[allow(clippy::too_many_arguments)]
fn grouped_out_proj_ref(
    out: &[f32],
    wo_a: &[f32],
    wo_b: &[f32],
    m: usize,
    n_groups: usize,
    o_group_dim: usize,
    o_lora_rank: usize,
    n_embd: usize,
    // Force every group to read group 0's weights AND group 0's input slice — the mistake a
    // hard-coded offset makes.
    pin_group0: bool,
) -> Vec<f32> {
    let oa_w = o_lora_rank * n_groups;
    let mut oa = vec![0f64; m * oa_w];
    for r in 0..m {
        for g in 0..n_groups {
            let sg = if pin_group0 { 0 } else { g };
            for i in 0..o_lora_rank {
                let wrow = (sg * o_lora_rank + i) * o_group_dim;
                let xoff = r * (n_groups * o_group_dim) + sg * o_group_dim;
                oa[r * oa_w + g * o_lora_rank + i] = (0..o_group_dim)
                    .map(|d| out[xoff + d] as f64 * wo_a[wrow + d] as f64)
                    .sum();
            }
        }
    }
    let mut dst = vec![0f32; m * n_embd];
    for r in 0..m {
        for o in 0..n_embd {
            dst[r * n_embd + o] = (0..oa_w)
                .map(|i| oa[r * oa_w + i] * wo_b[o * oa_w + i] as f64)
                .sum::<f64>() as f32;
        }
    }
    dst
}

/// V4's grouped low-rank output projection composes out of ops that already exist:
/// `CopyStrided` (slice group `g`'s columns) → `Linear { w_off }` (group `g`'s `wo_a` rows) →
/// `CopyStrided` (place into the concatenated `oa`) per group, then one `Linear` for `wo_b`.
///
/// The groups genuinely differ — group `g`'s weights are scaled by `(g+1)` — so a version that
/// reused group 0's weights (or group 0's input slice) for every group produces a different answer.
/// That variant is not merely described: the test BUILDS it (`pin_group0`, both offsets forced to
/// 0) and asserts the real graph does not match it, so the offsets are proven load-bearing.
///
/// `wo_a` is **bf16**, not f32, and that is load-bearing rather than incidental: Vulkan's
/// `Op::Linear` accepts `w_off` only on the offset-capable NATIVE-block kernels
/// (`native_dense_dtypes` — every quant format plus bf16), and refuses it outright on the f32/f16
/// fallbacks ("Linear w_off on a non-native weight"). A real V4 GGUF's `wo_a` is quantized, so the
/// composition holds in production; an f32 `wo_a` would need a per-group pack copy instead. The
/// reference rounds `wo_a` through bf16 first, so the comparison stays a test of the grouping and
/// not of the weight's precision.
#[test]
fn grouped_output_projection_composes_from_linear_and_copystrided() {
    let cpu = infr_cpu::CpuBackend::new();
    // `o_group_dim` (the per-group `in_f`) and `oa_w` are multiples of 32: the native-block GEMV /
    // GEMM kernels `w_off` rides on quantize the activation in 32-wide blocks, and an `in_f` below
    // that granularity yields all-zero output rather than an error. V4's real `o_group_dim` is
    // `n_head*head_dim/o_groups`, comfortably above it.
    let (m, nh, hd, n_groups, o_lora_rank, n_embd) =
        (3usize, 4usize, 32usize, 4usize, 8usize, 32usize);
    let o_group_dim = nh * hd / n_groups;
    let oa_w = o_lora_rank * n_groups;

    // Group g's weights scaled by (g+1): the groups are unmistakably different. Rounded through
    // bf16, which is the dtype actually uploaded (see this test's doc on `w_off`).
    let wo_a: Vec<f32> = gen(n_groups * o_lora_rank * o_group_dim, 21)
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            half::bf16::from_f32(v * ((i / (o_lora_rank * o_group_dim)) + 1) as f32).to_f32()
        })
        .collect();
    let wo_a_bytes: Vec<u8> = wo_a
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let wo_b = gen(n_embd * oa_w, 23);
    let outi = gen(m * nh * hd, 25);

    let build = |pin_group0: bool| {
        let mut g = Graph::new();
        let out = g.input(f32d(m * nh * hd));
        let wa = g.weight(TensorDesc::new(
            vec![n_groups * o_lora_rank * o_group_dim],
            DType::Bf16,
        ));
        let wb = g.weight(f32d(n_embd * oa_w));
        let oa = g.internal(f32d(m * oa_w));
        let dst = g.output(f32d(m * n_embd));
        for gi in 0..n_groups {
            let src_g = if pin_group0 { 0 } else { gi };
            let packed = g.internal(f32d(m * o_group_dim));
            let proj = g.internal(f32d(m * o_lora_rank));
            g.push(Op::CopyStrided {
                src: out,
                src_off: (src_g * o_group_dim) as u32,
                src_stride: (nh * hd) as u32,
                dst: packed,
                dst_off: 0,
                dst_stride: o_group_dim as u32,
                rows: m as u32,
                n: o_group_dim as u32,
            });
            g.push(Op::Linear {
                x: packed,
                weight: wa,
                dst: proj,
                m: m as u32,
                in_f: o_group_dim as u32,
                out_f: o_lora_rank as u32,
                w_off: (src_g * o_lora_rank * o_group_dim) as u32,
            });
            g.push(Op::CopyStrided {
                src: proj,
                src_off: 0,
                src_stride: o_lora_rank as u32,
                dst: oa,
                dst_off: (gi * o_lora_rank) as u32,
                dst_stride: oa_w as u32,
                rows: m as u32,
                n: o_lora_rank as u32,
            });
        }
        g.push(Op::Linear {
            x: oa,
            weight: wb,
            dst,
            m: m as u32,
            in_f: oa_w as u32,
            out_f: n_embd as u32,
            w_off: 0,
        });
        (g, out, wa, wb, dst)
    };

    let want = grouped_out_proj_ref(
        &outi,
        &wo_a,
        &wo_b,
        m,
        n_groups,
        o_group_dim,
        o_lora_rank,
        n_embd,
        false,
    );
    let pinned = grouped_out_proj_ref(
        &outi,
        &wo_a,
        &wo_b,
        m,
        n_groups,
        o_group_dim,
        o_lora_rank,
        n_embd,
        true,
    );
    let gap = maxerr(&want, &pinned);
    println!("GroupedOutProj grouped-vs-pinned reference gap={gap:e}");
    assert!(
        gap > 0.1,
        "input not well posed: pinning every group to group 0 would pass (gap={gap:e})"
    );

    // Bespoke runner: `wa` is bf16 BYTES, which `run` (f32 slices) cannot upload.
    let go = |be: &dyn Backend, pin_group0: bool| -> Vec<f32> {
        let (g, out, wa, wb, dst) = build(pin_group0);
        let plan = be.compile(&g).expect("compile");
        let up = |bytes: &[u8], usage| {
            let b = be.alloc(bytes.len(), usage).expect("alloc");
            be.upload(b.as_ref(), bytes).unwrap();
            b
        };
        let xb = up(bytemuck::cast_slice(&outi), BufferUsage::Activations);
        let ab = up(&wo_a_bytes, BufferUsage::Weights);
        let bb = up(bytemuck::cast_slice(&wo_b), BufferUsage::Weights);
        let ob = be.alloc(m * n_embd * 4, BufferUsage::Readback).unwrap();
        let mut b = Bindings::new();
        b.bind(out, xb.as_ref());
        b.bind(wa, ab.as_ref());
        b.bind(wb, bb.as_ref());
        b.bind(dst, ob.as_ref());
        be.execute(plan.as_ref(), &b).expect("execute");
        let mut o = vec![0f32; m * n_embd];
        be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut o))
            .unwrap();
        o
    };

    let check = |be: &dyn Backend, name: &str| {
        let got = go(be, false);
        let e = maxerr(&got, &want);
        println!("GroupedOutProj({name}) vs ref max_err={e:e}");
        assert!(e < 1e-4, "{name}: grouped projection wrong: max_err={e:e}");

        // The RED case, executed: same graph with both offsets pinned to group 0.
        let got_p = go(be, true);
        let e = maxerr(&got_p, &pinned);
        println!("GroupedOutProj({name}) pinned vs pinned-ref max_err={e:e}");
        assert!(
            e < 1e-4,
            "{name}: the pinned variant does not even match its own reference ({e:e}) — the test \
             is not measuring what it claims"
        );
        assert!(
            maxerr(&got, &got_p) > 0.1,
            "{name}: pinning every group to group 0 changed nothing — w_off/src_off are not \
             reaching the kernels"
        );
    };
    check(&cpu, "cpu");
    if let Some(vk) = gpu() {
        check(&vk, "vulkan");
    }
}

// ---- DeepSeek V4 Sinkhorn hyper-connections: `Op::HyperConnectMix` / `Pre` / `Post`.
//
// The references below are written from the DEFINITION — `ggml.h`'s `ggml_dsv4_hc_comb` /
// `ggml_dsv4_hc_pre` / `ggml_dsv4_hc_post` header comments for the index formulas, and
// `deepseek4.cpp`'s UNFUSED branches (`build_hc_pre`, `build_hc_sinkhorn`, `build_hc_post`) for the
// arithmetic. They are NOT transcribed from `infr-cpu`'s op arms: they are plain f64 triple loops
// written from the reference, so agreement between them and the CPU backend is evidence about the
// port rather than a restatement of it.

/// Shape + hyper-parameters of one hyper-connection case.
#[derive(Clone, Copy)]
struct HcDims {
    rows: usize,
    hc: usize,
    n_embd: usize,
    eps: f32,
    n_iter: u32,
    /// `true` = `build_hc_head`'s form — `mixes` is the `pre` chunk alone, no `post`/`comb`.
    head: bool,
}

impl HcDims {
    /// `(2 + hc)*hc` for the sublayer wrap, `hc` for the head (whose `output_hc_fn` is
    /// `{hc_dim, hc}`).
    fn mix_dim(&self) -> usize {
        if self.head {
            self.hc
        } else {
            (2 + self.hc) * self.hc
        }
    }
    /// `hc_scale` is `{3}` for the wrap and `{1}` for the head.
    fn n_scale(&self) -> usize {
        if self.head {
            1
        } else {
            3
        }
    }
}

/// One deliberate deviation from the definition, for the negative controls. [`FAITHFUL`] is the
/// definition itself; each field below names exactly one thing done wrong, and the test that flips
/// it asserts the answer MOVES — which is how each of these details is shown to be load-bearing
/// rather than decorative. Every one of them still runs and still produces plausible numbers.
#[derive(Clone, Copy)]
struct HcVariant {
    /// Read `comb`'s chunk as `src + hc*dst` instead of `dst + hc*src`.
    transpose_comb: bool,
    /// Run `n_iter` of BOTH normalisations instead of `n_iter` over-src and `n_iter - 1` over-dst.
    symmetric_iters: bool,
    /// Give the EXTRA normalisation to the `dst` axis instead of `src` — start and end on
    /// `norm_dst`. This is what trusting llama.cpp's `norm_rows`/`norm_cols` NAMES produces: the
    /// lambdas are named for the opposite axis to the one they reduce over.
    swap_norm_axes: bool,
    /// Drop the `+ eps` applied to every element right after the softmax.
    drop_eps_softmax: bool,
    /// Drop the `+ eps` on the over-src sum before it divides.
    drop_eps_src: bool,
    /// Drop the `+ eps` on the over-dst sum before it divides.
    drop_eps_dst: bool,
    /// Drop the `+ eps` on `pre` after its sigmoid (the fourth, non-Sinkhorn site).
    drop_eps_pre: bool,
    /// Swap the `pre` and `post` chunk offsets (and their scale index / base offset with them) —
    /// adjacent, equal-width chunks of the same tensor.
    swap_pre_post: bool,
}

const FAITHFUL: HcVariant = HcVariant {
    transpose_comb: false,
    symmetric_iters: false,
    swap_norm_axes: false,
    drop_eps_softmax: false,
    drop_eps_src: false,
    drop_eps_dst: false,
    drop_eps_pre: false,
    swap_pre_post: false,
};

/// `build_hc_sinkhorn` in f64, in place over `m` laid out `dst + hc*src`.
fn hc_ref_sinkhorn(m: &mut [f64], hc: usize, eps: f64, n_iter: u32, v: HcVariant) {
    // ggml_soft_max over ne[0] — for `comb` reshaped `[dst_hc, src_hc, n_tokens]` that is the
    // `dst` axis, one softmax per src column. Then `ggml_add(comb, eps)`.
    let e_soft = if v.drop_eps_softmax { 0.0 } else { eps };
    for src in 0..hc {
        let col = &mut m[src * hc..][..hc];
        let mx = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0.0;
        for x in col.iter_mut() {
            *x = (*x - mx).exp();
            s += *x;
        }
        for x in col.iter_mut() {
            *x = *x / s + e_soft;
        }
    }
    let e_src = if v.drop_eps_src { 0.0 } else { eps };
    let e_dst = if v.drop_eps_dst { 0.0 } else { eps };
    // llama.cpp's `norm_cols`: permute so `src` is ne[0], `ggml_sum_rows` (i.e. sum over SRC),
    // add eps, divide. Reduces over src, despite the name.
    let norm_src = |m: &mut [f64]| {
        for dst in 0..hc {
            let s: f64 = (0..hc).map(|src| m[dst + hc * src]).sum();
            let q = s + e_src;
            for src in 0..hc {
                m[dst + hc * src] /= q;
            }
        }
    };
    // llama.cpp's `norm_rows`: `ggml_sum_rows` of the matrix as laid out, whose ne[0] is DST.
    // Reduces over dst, despite the name.
    let norm_dst = |m: &mut [f64]| {
        for src in 0..hc {
            let col = &mut m[src * hc..][..hc];
            let q: f64 = col.iter().sum::<f64>() + e_dst;
            for x in col.iter_mut() {
                *x /= q;
            }
        }
    };
    if v.symmetric_iters {
        for _ in 0..n_iter {
            norm_dst(m);
            norm_src(m);
        }
    } else if v.swap_norm_axes {
        norm_dst(m);
        for _ in 1..n_iter {
            norm_src(m);
            norm_dst(m);
        }
    } else {
        norm_src(m);
        for _ in 1..n_iter {
            norm_dst(m);
            norm_src(m);
        }
    }
}

/// `build_hc_pre`'s coefficient half in f64 — returns `(pre, post, comb)`, the latter two empty in
/// the head form.
fn hc_ref_mix(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    d: HcDims,
    v: HcVariant,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (hc, md, eps) = (d.hc, d.mix_dim(), d.eps as f64);
    // The pre chunk is at offset 0 with scale[0]; post at hc with scale[1]. `swap_pre_post` reads
    // them the other way round — the mistake nothing but a value check catches.
    let (pre_off, post_off) = if v.swap_pre_post { (hc, 0) } else { (0, hc) };
    let (pre_si, post_si) = if v.swap_pre_post { (1, 0) } else { (0, 1) };
    let e_pre = if v.drop_eps_pre { 0.0 } else { eps };
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let mut pre = vec![0f64; d.rows * hc];
    for t in 0..d.rows {
        for h in 0..hc {
            let z = mixes[t * md + pre_off + h] as f64 * scale[pre_si] as f64
                + base[pre_off + h] as f64;
            pre[t * hc + h] = sig(z) + e_pre;
        }
    }
    if d.head {
        return (pre, Vec::new(), Vec::new());
    }
    let mut post = vec![0f64; d.rows * hc];
    let mut comb = vec![0f64; d.rows * hc * hc];
    for t in 0..d.rows {
        for h in 0..hc {
            let z = mixes[t * md + post_off + h] as f64 * scale[post_si] as f64
                + base[post_off + h] as f64;
            post[t * hc + h] = 2.0 * sig(z);
        }
        let m = &mut comb[t * hc * hc..][..hc * hc];
        for dst in 0..hc {
            for src in 0..hc {
                // logits[dst, src, t] = mixes[2*hc + dst + hc*src, t]*scale[2] + base[2*hc + ...]
                let k = if v.transpose_comb {
                    src + hc * dst
                } else {
                    dst + hc * src
                };
                m[dst + hc * src] =
                    mixes[t * md + 2 * hc + k] as f64 * scale[2] as f64 + base[2 * hc + k] as f64;
            }
        }
        hc_ref_sinkhorn(m, hc, eps, d.n_iter, v);
    }
    (pre, post, comb)
}

/// `ggml_dsv4_hc_pre`: `result[i, t] = Σ_h x[i, h, t]*weights[h, t]`.
fn hc_ref_pre(x: &[f32], w: &[f32], d: HcDims) -> Vec<f64> {
    let (hc, ne) = (d.hc, d.n_embd);
    let mut out = vec![0f64; d.rows * ne];
    for t in 0..d.rows {
        for h in 0..hc {
            for i in 0..ne {
                out[t * ne + i] += x[(t * hc + h) * ne + i] as f64 * w[t * hc + h] as f64;
            }
        }
    }
    out
}

/// `ggml_dsv4_hc_post`: `result[i, dst, t] = x[i,t]*post[dst,t] + Σ_src residual[i,src,t]*comb[dst,src,t]`.
fn hc_ref_post(x: &[f32], residual: &[f32], post: &[f32], comb: &[f32], d: HcDims) -> Vec<f64> {
    let (hc, ne) = (d.hc, d.n_embd);
    let mut out = vec![0f64; d.rows * hc * ne];
    for t in 0..d.rows {
        for dst in 0..hc {
            for i in 0..ne {
                let mut acc = x[t * ne + i] as f64 * post[t * hc + dst] as f64;
                for src in 0..hc {
                    acc += residual[(t * hc + src) * ne + i] as f64
                        * comb[t * hc * hc + dst + hc * src] as f64;
                }
                out[(t * hc + dst) * ne + i] = acc;
            }
        }
    }
    out
}

fn maxerr64(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "reference/backend length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - y).abs())
        .fold(0.0, f64::max)
}

fn maxdiff64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `mixes` values for a case: mixed-sign, no symmetry between the `dst`/`src` axes of the `comb`
/// chunk (a symmetric one would hide a transposed index) and spread wide enough that the softmax
/// inside Sinkhorn is not near-uniform.
fn hc_mixes(rows: usize, mix_dim: usize) -> Vec<f32> {
    (0..rows * mix_dim)
        .map(|i| (((i * 37 + 11) % 23) as f32 - 11.0) * 0.31)
        .collect()
}

/// Build + run one `Op::HyperConnectMix`, returning `(pre, post, comb)` (the latter two empty in
/// the head form). Bespoke because the op writes THREE outputs and `run` handles one.
fn hc_run_mix(
    be: &dyn Backend,
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    d: HcDims,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hc, md) = (d.hc, d.mix_dim());
    let mut g = Graph::new();
    let mx = g.input(f32d(d.rows * md));
    let sc = g.weight(f32d(d.n_scale()));
    let bs = g.weight(f32d(md));
    let pre = g.output(f32d(d.rows * hc));
    let gates = (!d.head).then(|| infr_core::graph::HyperGates {
        post: g.output(f32d(d.rows * hc)),
        comb: g.output(f32d(d.rows * hc * hc)),
    });
    g.push(Op::HyperConnectMix {
        mixes: mx,
        scale: sc,
        base: bs,
        pre,
        gates,
        rows: d.rows as u32,
        hc: hc as u32,
        eps: d.eps,
        n_iter: d.n_iter,
    });

    let plan = be.compile(&g).expect("compile HyperConnectMix");
    let up = |data: &[f32], usage| {
        let b = be.alloc(data.len().max(1) * 4, usage).expect("alloc");
        be.upload(b.as_ref(), bytemuck::cast_slice(data)).unwrap();
        b
    };
    let mb = up(mixes, BufferUsage::Activations);
    let sb = up(scale, BufferUsage::Weights);
    let bb = up(base, BufferUsage::Weights);
    let pb = be
        .alloc(d.rows * hc * 4, BufferUsage::Readback)
        .expect("alloc pre");
    let mut b = Bindings::new();
    b.bind(mx, mb.as_ref());
    b.bind(sc, sb.as_ref());
    b.bind(bs, bb.as_ref());
    b.bind(pre, pb.as_ref());
    let gb = gates.map(|gt| {
        (
            gt,
            be.alloc(d.rows * hc * 4, BufferUsage::Readback).unwrap(),
            be.alloc(d.rows * hc * hc * 4, BufferUsage::Readback)
                .unwrap(),
        )
    });
    if let Some((gt, ob, cb)) = &gb {
        b.bind(gt.post, ob.as_ref());
        b.bind(gt.comb, cb.as_ref());
    }
    be.execute(plan.as_ref(), &b).expect("execute");
    let dl = |buf: &dyn infr_core::backend::Buffer, n: usize| {
        let mut o = vec![0f32; n];
        be.download(buf, bytemuck::cast_slice_mut(&mut o)).unwrap();
        o
    };
    let pre_o = dl(pb.as_ref(), d.rows * hc);
    match &gb {
        Some((_, ob, cb)) => (
            pre_o,
            dl(ob.as_ref(), d.rows * hc),
            dl(cb.as_ref(), d.rows * hc * hc),
        ),
        None => (pre_o, Vec::new(), Vec::new()),
    }
}

/// The case table shared by every hyper-connection test below. `hc = 4` is production (llama.cpp
/// `GGML_ASSERT`s it); the rest move the axes a kernel can get wrong — `hc = 1` (a degenerate
/// 1×1 Sinkhorn), `hc = 3` (not a power of two), `hc = 8` (`HYPER_CONNECT_MAX_MULT`, the widest
/// any backend accepts) — across `n_iter` 1 (the loop body never runs, only the lone `norm_src`),
/// 2, 3 and 5.
fn hc_cases() -> Vec<(&'static str, HcDims)> {
    let e = 1e-6f32;
    vec![
        (
            "decode hc=4 n_iter=3, 1 token",
            HcDims {
                rows: 1,
                hc: 4,
                n_embd: 5,
                eps: e,
                n_iter: 3,
                head: false,
            },
        ),
        (
            "production hc=4 n_iter=3, 7 tokens",
            HcDims {
                rows: 7,
                hc: 4,
                n_embd: 5,
                eps: e,
                n_iter: 3,
                head: false,
            },
        ),
        (
            "hc=4 n_iter=1 (only the lone norm_src runs)",
            HcDims {
                rows: 5,
                hc: 4,
                n_embd: 6,
                eps: e,
                n_iter: 1,
                head: false,
            },
        ),
        (
            "hc=3 (not a power of two) n_iter=2",
            HcDims {
                rows: 6,
                hc: 3,
                n_embd: 7,
                eps: e,
                n_iter: 2,
                head: false,
            },
        ),
        (
            "hc=1 (degenerate 1x1 Sinkhorn) n_iter=4",
            HcDims {
                rows: 3,
                hc: 1,
                n_embd: 9,
                eps: e,
                n_iter: 4,
                head: false,
            },
        ),
        (
            "hc=8 (HYPER_CONNECT_MAX_MULT) n_iter=5",
            HcDims {
                rows: 4,
                hc: 8,
                n_embd: 3,
                eps: e,
                n_iter: 5,
                head: false,
            },
        ),
        (
            "model head form (pre only), hc=4",
            HcDims {
                rows: 7,
                hc: 4,
                n_embd: 5,
                eps: e,
                n_iter: 3,
                head: true,
            },
        ),
        (
            "large eps 1e-2 (every eps site visible), hc=4 n_iter=3",
            HcDims {
                rows: 5,
                hc: 4,
                n_embd: 4,
                eps: 1e-2,
                n_iter: 3,
                head: false,
            },
        ),
    ]
}

/// Scale/base for a case. `base` is per-element and mixed-sign, `scale` differs per chunk so
/// reading the wrong index is visible.
fn hc_scale_base(d: HcDims) -> (Vec<f32>, Vec<f32>) {
    let scale: Vec<f32> = vec![0.7, -1.3, 1.9][..d.n_scale()].to_vec();
    let base = (0..d.mix_dim())
        .map(|i| (((i * 13 + 5) % 17) as f32 - 8.0) * 0.17)
        .collect();
    (scale, base)
}

/// Tolerance for every hyper-connection backend comparison (CPU vs the f64 reference, and each GPU
/// vs CPU). All these outputs are O(1) — `pre` and `comb` are in `(0, 1]`, `post` in `(0, 2)` — so
/// an absolute bound is the meaningful one.
///
/// Chosen, not measured-and-rounded: f32 carries ~1e-7 relative, a Sinkhorn round is a handful of
/// dependent ops, and a GPU `exp` is only required to be within ~3 ULP of the host's, so ~1e-6 is
/// the floor a correct port can reach. `1e-5` sits an order of magnitude above that and two orders
/// BELOW the smallest structural defect `hyper_connect_details_are_load_bearing` measures (a
/// transposed `comb`, a swapped `pre`/`post`, or a symmetric iteration count each move the answer
/// by >1e-3). Observed with this table: CPU vs reference ≤1.8e-7, Vulkan vs CPU ≤2.4e-7.
const HC_TOL: f32 = 1e-5;

/// `Op::HyperConnectMix` — CPU vs the from-definition f64 reference, plus CPU-vs-Vulkan.
#[test]
fn hyper_connect_mix_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    for (name, d) in hc_cases() {
        let mixes = hc_mixes(d.rows, d.mix_dim());
        let (scale, base) = hc_scale_base(d);
        let (wpre, wpost, wcomb) = hc_ref_mix(&mixes, &scale, &base, d, FAITHFUL);
        let (pre, post, comb) = hc_run_mix(&cpu, &mixes, &scale, &base, d);
        let (ep, eo, ec) = (
            maxerr64(&pre, &wpre),
            maxerr64(&post, &wpost),
            maxerr64(&comb, &wcomb),
        );
        println!("HyperConnectMix {name}: cpu vs ref pre={ep:e} post={eo:e} comb={ec:e}");
        let tol = HC_TOL as f64;
        assert!(ep < tol, "{name}: pre diverges from the reference ({ep:e})");
        assert!(
            eo < tol,
            "{name}: post diverges from the reference ({eo:e})"
        );
        assert!(
            ec < tol,
            "{name}: comb diverges from the reference ({ec:e})"
        );

        if let Some(vk) = gpu() {
            let (vp, vo, vc) = hc_run_mix(&vk, &mixes, &scale, &base, d);
            let (ep, eo, ec) = (maxerr(&vp, &pre), maxerr(&vo, &post), maxerr(&vc, &comb));
            println!("HyperConnectMix {name}: vulkan vs cpu pre={ep:e} post={eo:e} comb={ec:e}");
            assert!(ep < HC_TOL, "{name}: Vulkan pre diverges from CPU ({ep:e})");
            assert!(
                eo < HC_TOL,
                "{name}: Vulkan post diverges from CPU ({eo:e})"
            );
            assert!(
                ec < HC_TOL,
                "{name}: Vulkan comb diverges from CPU ({ec:e})"
            );
        }
    }
}

/// **The doubly-stochastic property itself** — the whole point of the Sinkhorn iteration, asserted
/// on the CPU BACKEND's output (not on the reference).
///
/// Two data sets, because the property and the ORIENTATION of the normalisations cannot be shown
/// by the same one:
///
/// * `converged` — mild logits (`|z| < 1`) and 40 iterations. Sinkhorn converges and BOTH sums
///   land within 1e-4: that is the property.
/// * `production` — the `hc = 4, n_iter = 3` case from the shared table, with the table's peaked
///   logits. Sinkhorn has NOT converged there (its rate collapses as the matrix approaches a
///   permutation — a property of the algorithm, not a port bug: `|sum_dst - 1| ≈ 0.5`), and that
///   is exactly what makes the orientation visible. `norm_src` is the LAST normalisation the
///   asymmetric loop runs and it divides by `sum + eps`, so `sum_src` is exact to eps level while
///   `sum_dst` is only as close as the iteration got.
///
/// The RED case here is `swap_norm_axes` — giving the extra normalisation to `dst`, which is what
/// trusting llama.cpp's INVERTED `norm_rows`/`norm_cols` names produces. It swaps which axis comes
/// out exact and is asserted to fail the tight bound.
///
/// What these sums do NOT catch, asserted so the gap is on the record rather than assumed: a
/// transposed `comb` INDEX (`src + hc*dst`). Sinkhorn applied to a transposed matrix is just
/// Sinkhorn applied to a different matrix — it still ends on `norm_src` and its sums are just as
/// well behaved. That one is caught by value: `hyper_connect_details_are_load_bearing` and
/// `hyper_connect_post_parity`'s executed transpose, which move the answer by ~1.
#[test]
fn hyper_connect_comb_is_doubly_stochastic() {
    let cpu = infr_cpu::CpuBackend::new();
    let converged = HcDims {
        rows: 5,
        hc: 4,
        n_embd: 4,
        eps: 1e-6,
        n_iter: 40,
        head: false,
    };
    let production = hc_cases()[0].1;
    for (name, d, want_converged) in [
        ("converged (mild logits, n_iter=40)", converged, true),
        ("production hc=4 n_iter=3", production, false),
    ] {
        let (hc, eps) = (d.hc, d.eps as f64);
        // The converged case needs logits small enough that the softmax is not near-one-hot; the
        // production case uses the shared generator unchanged.
        let mixes: Vec<f32> = if want_converged {
            (0..d.rows * d.mix_dim())
                .map(|i| (((i * 37 + 11) % 23) as f32 - 11.0) * 0.04)
                .collect()
        } else {
            hc_mixes(d.rows, d.mix_dim())
        };
        let (scale, base) = if want_converged {
            (
                vec![0.7, -1.3, 0.5],
                (0..d.mix_dim())
                    .map(|i| (((i * 13 + 5) % 17) as f32 - 8.0) * 0.02)
                    .collect::<Vec<f32>>(),
            )
        } else {
            hc_scale_base(d)
        };
        let (_, _, comb) = hc_run_mix(&cpu, &mixes, &scale, &base, d);
        // The RED variant: the extra normalisation given to `dst` instead of `src`.
        let (_, _, swapped) = hc_ref_mix(
            &mixes,
            &scale,
            &base,
            d,
            HcVariant {
                swap_norm_axes: true,
                ..FAITHFUL
            },
        );
        // And the transposed INDEX, which these sums are asserted below NOT to be able to see.
        let (_, _, tcomb) = hc_ref_mix(
            &mixes,
            &scale,
            &base,
            d,
            HcVariant {
                transpose_comb: true,
                ..FAITHFUL
            },
        );

        let sums = |m: &[f64]| -> (f64, f64) {
            let (mut esrc, mut edst) = (0f64, 0f64);
            for t in 0..d.rows {
                let b = t * hc * hc;
                for dst in 0..hc {
                    let s: f64 = (0..hc).map(|src| m[b + dst + hc * src]).sum();
                    esrc = esrc.max((s - 1.0).abs());
                }
                for src in 0..hc {
                    let s: f64 = (0..hc).map(|dst| m[b + dst + hc * src]).sum();
                    edst = edst.max((s - 1.0).abs());
                }
            }
            (esrc, edst)
        };
        let got: Vec<f64> = comb.iter().map(|&v| v as f64).collect();
        let (esrc, edst) = sums(&got);
        println!("comb[{name}]: |sum_src - 1| = {esrc:e}, |sum_dst - 1| = {edst:e} (eps={eps:e})");
        let tight = 4.0 * eps + hc as f64 * 1e-6;
        let (wsrc, wdst) = sums(&swapped);
        let (tsrc, tdst) = sums(&tcomb);
        println!("comb[{name}] swapped norm axes: src={wsrc:e} dst={wdst:e}");
        println!("comb[{name}] transposed index:  src={tsrc:e} dst={tdst:e}");
        // The gap, asserted rather than assumed: a transposed INDEX leaves both sums exactly as
        // well behaved as the faithful one, so nothing here can see it.
        assert!(
            (tsrc < tight) == (esrc < tight) && (tdst < tight) == (edst < tight),
            "{name}: a transposed comb index changed which axis is exact (src {tsrc:e} vs \
             {esrc:e}, dst {tdst:e} vs {edst:e}) — then the doc above is wrong and this test \
             could have been asserting the index too"
        );
        if want_converged {
            assert!(
                esrc < 1e-4 && edst < 1e-4,
                "{name}: comb is not doubly stochastic (src {esrc:e}, dst {edst:e})"
            );
        } else {
            // Stated as an assertion so it goes red if it ever stops being true: at n_iter=3 on
            // peaked logits the over-dst sums have NOT converged, which is why the case above
            // exists at all.
            assert!(
                edst > 1e-2,
                "{name}: sum_dst is already converged ({edst:e}) — the `converged` case above is \
                 no longer testing anything the production one does not"
            );
            // The orientation. `norm_src` divides by `sum + eps`, so `sum_src`'s residual is
            // bounded by eps regardless of convergence; f32 storage of comb adds ~hc ULP.
            assert!(
                esrc < tight,
                "{name}: the LAST normalisation is over src, so |sum_src - 1| must be at eps \
                 level, not {esrc:e} (> {tight:e}) — comb is transposed"
            );
            // The RED half: the same bound applied to a Sinkhorn whose extra normalisation went
            // to `dst` must FAIL, and `sum_dst` must be the exact one there instead.
            assert!(
                wsrc > tight,
                "{name}: swapping the normalisation axes passes the tight sum_src bound \
                 ({wsrc:e} <= {tight:e}) — the assertion above proves nothing"
            );
            assert!(
                wdst < tight,
                "{name}: with the axes swapped, sum_dst should be the exact one ({wdst:e})"
            );
        }
    }
}

/// Each detail `docs/deepseek.md` warns about, shown to CHANGE the answer. If the CPU arm had
/// implemented any of these variants, `hyper_connect_mix_parity` would be comparing against the
/// wrong reference and would still pass — so this test is what makes that one mean something.
///
/// Bounds, and what each says:
///
/// * a STRUCTURAL variant (swapped `pre`/`post` offsets, transposed `comb` index, the extra
///   normalisation given to `dst`) must move the answer by >1e-3, two orders above `HC_TOL`. Those
///   are pinned by the backend parity test on every case.
/// * an EPS-SITE drop must move it by >1e-9 — unambiguously not inert. At `eps = 1e-6` that can
///   fall BELOW `HC_TOL` (the over-dst site is the smallest, since the final over-src
///   normalisation partly washes it out), which is why the table carries an `eps = 1e-2` case
///   where every site is required to clear `HC_TOL` and the backends are pinned too.
/// * `symmetric_iters` gets its own, much smaller bound — see the comment at its call site.
///
/// Three details are genuinely INERT on some cases, and each is asserted to move the answer by
/// EXACTLY zero there rather than skipped quietly: `comb`'s index at `hc = 1` (a 1×1 transpose is
/// the identity), the over-dst eps site at `n_iter = 1` (the reference's `for (i = 1; i < n_iter)`
/// body never runs, so `norm_dst` is never called — a direct check that the loop is the asymmetric
/// one), and every `comb`/`post` variant in the head form, which has neither.
#[test]
fn hyper_connect_details_are_load_bearing() {
    for (name, d) in hc_cases() {
        let mixes = hc_mixes(d.rows, d.mix_dim());
        let (scale, base) = hc_scale_base(d);
        let (pre0, post0, comb0) = hc_ref_mix(&mixes, &scale, &base, d, FAITHFUL);
        // How far this variant moves the faithful answer, over all three outputs.
        let moved = |what: &str, v: HcVariant| -> f64 {
            let (pre, post, comb) = hc_ref_mix(&mixes, &scale, &base, d, v);
            let e = maxdiff64(&pre0, &pre)
                .max(maxdiff64(&post0, &post))
                .max(maxdiff64(&comb0, &comb));
            println!("HyperConnect {name}: {what} moves the answer by {e:e}");
            e
        };
        let must_move = |what: &str, v: HcVariant, min: f64| {
            let e = moved(what, v);
            assert!(
                e > min,
                "{name}: {what} changed the answer by only {e:e} (<= {min:e}) — that detail is \
                 not pinned by this case"
            );
        };
        let must_be_inert = |what: &str, v: HcVariant, why: &str| {
            let e = moved(what, v);
            assert_eq!(
                e, 0.0,
                "{name}: {what} was expected to be inert here ({why}) but moved \
                 the answer by {e:e}"
            );
        };

        let big_eps = d.eps as f64 >= 1e-3;
        // An eps site must not be inert; on the large-eps case it must also clear the tolerance
        // the backend comparisons run at.
        let tiny = if big_eps { 10.0 * HC_TOL as f64 } else { 1e-9 };

        must_move(
            "dropping eps on pre",
            HcVariant {
                drop_eps_pre: true,
                ..FAITHFUL
            },
            tiny,
        );
        if d.head {
            continue; // no post, no comb, no Sinkhorn
        }
        must_move(
            "reading pre/post at each other's offsets",
            HcVariant {
                swap_pre_post: true,
                ..FAITHFUL
            },
            1e-3,
        );
        if d.hc == 1 {
            // Every Sinkhorn detail below is genuinely inert on a 1x1 matrix, for three separate
            // reasons — see `hyper_connect_hc1_sinkhorn_details_are_inert`, which asserts each one.
            continue;
        }
        must_move(
            "indexing comb as src + hc*dst",
            HcVariant {
                transpose_comb: true,
                ..FAITHFUL
            },
            1e-3,
        );
        must_move(
            "dropping eps after the softmax",
            HcVariant {
                drop_eps_softmax: true,
                ..FAITHFUL
            },
            tiny,
        );
        must_move(
            "dropping eps on the over-src sum",
            HcVariant {
                drop_eps_src: true,
                ..FAITHFUL
            },
            tiny,
        );
        let drop_dst = HcVariant {
            drop_eps_dst: true,
            ..FAITHFUL
        };
        // At n_iter = 1 the reference's `for (i = 1; i < n_iter)` body never runs, so `norm_dst`
        // is never called — a direct check that the loop really is the asymmetric one.
        if d.n_iter > 1 {
            must_move("dropping eps on the over-dst sum", drop_dst, tiny);
        } else {
            must_be_inert(
                "dropping eps on the over-dst sum",
                drop_dst,
                "at n_iter = 1 the loop body never runs, so norm_dst is never called",
            );
        }
        must_move(
            "giving the extra normalisation to dst (llama.cpp's inverted lambda names)",
            HcVariant {
                swap_norm_axes: true,
                ..FAITHFUL
            },
            1e-3,
        );
        // The count itself is pinned, but only just, and the mechanism is worth stating: the
        // softmax already left every src column summing to `1 + hc*eps`, so the symmetric
        // variant's EXTRA LEADING `norm_dst` divides the whole matrix by one and the same
        // constant — a uniform rescale that the following `norm_src` undoes. What survives is
        // second order in eps, ~1e-11 at eps = 1e-6 and ~1e-4 at eps = 1e-2. So the count IS
        // observable, but below `HC_TOL` on the small-eps cases: the large-eps case is what pins
        // it for the backends.
        must_move(
            "running n_iter of BOTH normalisations",
            HcVariant {
                symmetric_iters: true,
                ..FAITHFUL
            },
            if big_eps { HC_TOL as f64 } else { 1e-13 },
        );
    }
}

/// `hc = 1` is degenerate three times over, and each degeneracy makes a Sinkhorn detail inert.
/// `hyper_connect_details_are_load_bearing` therefore runs only its non-Sinkhorn checks there, and
/// this test states exactly what is lost — as assertions, so a change that made any of them
/// observable would show up here rather than silently widen that test's coverage claim:
///
/// * a 1×1 matrix is its own transpose, so `comb`'s index formula cannot be got wrong;
/// * `norm_src` and `norm_dst` are THE SAME operation on a 1×1 matrix, so the asymmetric loop and
///   its axis-swapped variant collapse onto each other;
/// * `v ← v/(v + eps)` has a fixed point independent of the initial value, and `n_iter = 4`
///   reaches it — so a perturbation applied BEFORE the iteration (the post-softmax eps) is washed
///   out entirely.
///
/// What is NOT lost, asserted too: the eps sites INSIDE the normalisations change the fixed point
/// itself, so they still move the answer — the over-src one at first order in eps (it runs last,
/// so nothing re-normalises after it) and the over-dst one at second order (the following
/// `norm_src` almost washes it out, the same mechanism that makes `symmetric_iters` tiny).
/// `hc = 1` is in the shared table for kernel SHAPE coverage (a degenerate loop bound on every
/// backend), not for semantics.
#[test]
fn hyper_connect_hc1_sinkhorn_details_are_inert() {
    let d = HcDims {
        rows: 3,
        hc: 1,
        n_embd: 9,
        eps: 1e-6,
        n_iter: 4,
        head: false,
    };
    let mixes = hc_mixes(d.rows, d.mix_dim());
    let (scale, base) = hc_scale_base(d);
    let (_, _, c0) = hc_ref_mix(&mixes, &scale, &base, d, FAITHFUL);
    let comb_of = |v: HcVariant| hc_ref_mix(&mixes, &scale, &base, d, v).2;
    for (what, v) in [
        (
            "transposing comb's index",
            HcVariant {
                transpose_comb: true,
                ..FAITHFUL
            },
        ),
        (
            "swapping the normalisation axes",
            HcVariant {
                swap_norm_axes: true,
                ..FAITHFUL
            },
        ),
        (
            "dropping eps after the softmax",
            HcVariant {
                drop_eps_softmax: true,
                ..FAITHFUL
            },
        ),
    ] {
        assert_eq!(
            c0,
            comb_of(v),
            "hc=1: {what} was expected to be exactly inert"
        );
    }
    for (what, v, min) in [
        (
            "dropping eps on the over-src sum",
            HcVariant {
                drop_eps_src: true,
                ..FAITHFUL
            },
            1e-9, // first order in eps: nothing re-normalises after the last norm_src
        ),
        (
            "dropping eps on the over-dst sum",
            HcVariant {
                drop_eps_dst: true,
                ..FAITHFUL
            },
            1e-13, // second order: the following norm_src almost washes it out
        ),
    ] {
        let e = maxdiff64(&c0, &comb_of(v));
        println!("HyperConnect hc=1: {what} moves comb by {e:e}");
        assert!(
            e > min,
            "hc=1: {what} changes the iteration's fixed point, so it must still move the answer \
             (moved {e:e}, wanted > {min:e})"
        );
    }
}

/// `Op::HyperConnectPre` — CPU vs the from-definition reference, plus CPU-vs-Vulkan.
///
/// The `hc` streams are given magnitudes that differ by three orders of magnitude (stream `h` is
/// scaled by `100^h`), so a collapse that dropped `weights`, used the wrong stream, or summed the
/// streams unweighted is off by orders of magnitude rather than by a rounding error. The
/// `weights` are a real `Op::HyperConnectMix` output (sigmoids in `(0, 1]`), not a synthetic
/// vector, so this exercises the pair as it will be wired.
#[test]
fn hyper_connect_pre_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    for (name, d) in hc_cases() {
        let mixes = hc_mixes(d.rows, d.mix_dim());
        let (scale, base) = hc_scale_base(d);
        let (w, _, _) = hc_run_mix(&cpu, &mixes, &scale, &base, d);
        let x: Vec<f32> = (0..d.rows * d.hc * d.n_embd)
            .map(|i| {
                let h = (i / d.n_embd) % d.hc;
                (((i * 7 + 3) % 11) as f32 - 5.0) * 0.25 * 100f32.powi(h as i32)
            })
            .collect();
        let want = hc_ref_pre(&x, &w, d);

        let mut g = Graph::new();
        let xi = g.input(f32d(d.rows * d.hc * d.n_embd));
        let wi = g.input(f32d(d.rows * d.hc));
        let dst = g.output(f32d(d.rows * d.n_embd));
        g.push(Op::HyperConnectPre {
            x: xi,
            weights: wi,
            dst,
            rows: d.rows as u32,
            hc: d.hc as u32,
            n_embd: d.n_embd as u32,
        });
        let go = |be: &dyn Backend| run(be, &g, &[(xi, &x), (wi, &w)], &[], dst, d.rows * d.n_embd);
        let c = go(&cpu);
        // Relative: stream 3's magnitude is ~1e6, so an absolute f32 bound would be meaningless.
        let scale_of = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1.0);
        let e = maxerr64(&c, &want) / scale_of;
        println!("HyperConnectPre {name}: cpu vs ref rel={e:e} (|out|max={scale_of:e})");
        assert!(
            e < HC_TOL as f64,
            "{name}: HyperConnectPre diverges from the reference ({e:e})"
        );
        if let Some(vk) = gpu() {
            let v = go(&vk);
            let e = maxerr(&v, &c) as f64 / scale_of;
            println!("HyperConnectPre {name}: vulkan vs cpu rel={e:e}");
            assert!(
                e < HC_TOL as f64,
                "{name}: Vulkan HyperConnectPre diverges from CPU ({e:e})"
            );
        }
    }
}

/// `Op::HyperConnectPost` — CPU vs the from-definition reference, plus CPU-vs-Vulkan. `post` and
/// `comb` are real `Op::HyperConnectMix` outputs. The residual streams again differ by orders of
/// magnitude, which is what makes a transposed `comb` (or an ignored `post`) a large error rather
/// than a small one.
#[test]
fn hyper_connect_post_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    for (name, d) in hc_cases() {
        if d.head {
            continue; // the head form has no post/comb, and no sublayer to wrap
        }
        let mixes = hc_mixes(d.rows, d.mix_dim());
        let (scale, base) = hc_scale_base(d);
        let (_, post, comb) = hc_run_mix(&cpu, &mixes, &scale, &base, d);
        let residual: Vec<f32> = (0..d.rows * d.hc * d.n_embd)
            .map(|i| {
                let h = (i / d.n_embd) % d.hc;
                (((i * 5 + 2) % 13) as f32 - 6.0) * 0.25 * 100f32.powi(h as i32)
            })
            .collect();
        let x: Vec<f32> = (0..d.rows * d.n_embd)
            .map(|i| (((i * 11 + 4) % 9) as f32 - 4.0) * 0.5)
            .collect();
        let want = hc_ref_post(&x, &residual, &post, &comb, d);

        let mut g = Graph::new();
        let xi = g.input(f32d(d.rows * d.n_embd));
        let ri = g.input(f32d(d.rows * d.hc * d.n_embd));
        let pi = g.input(f32d(d.rows * d.hc));
        let ci = g.input(f32d(d.rows * d.hc * d.hc));
        let dst = g.output(f32d(d.rows * d.hc * d.n_embd));
        g.push(Op::HyperConnectPost {
            x: xi,
            residual: ri,
            post: pi,
            comb: ci,
            dst,
            rows: d.rows as u32,
            hc: d.hc as u32,
            n_embd: d.n_embd as u32,
        });
        let ins = [
            (xi, &x[..]),
            (ri, &residual[..]),
            (pi, &post[..]),
            (ci, &comb[..]),
        ];
        let n = d.rows * d.hc * d.n_embd;
        let go = |be: &dyn Backend| run(be, &g, &ins, &[], dst, n);
        let c = go(&cpu);
        let scale_of = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1.0);
        let e = maxerr64(&c, &want) / scale_of;
        println!("HyperConnectPost {name}: cpu vs ref rel={e:e} (|out|max={scale_of:e})");
        assert!(
            e < HC_TOL as f64,
            "{name}: HyperConnectPost diverges from the reference ({e:e})"
        );

        // The RED case, executed on the backend: feed the op a TRANSPOSED comb and confirm the
        // output moves. `dst + hc*src` vs `src + hc*dst` is otherwise invisible — same buffer
        // length, same value distribution, and (per the test above) still doubly stochastic.
        // Skipped at `hc = 1`, where the transpose is the identity — see
        // `hyper_connect_hc1_transpose_is_genuinely_inert`.
        if d.hc > 1 {
            let mut tcomb = comb.clone();
            for t in 0..d.rows {
                for dst_h in 0..d.hc {
                    for src in 0..d.hc {
                        tcomb[t * d.hc * d.hc + dst_h + d.hc * src] =
                            comb[t * d.hc * d.hc + src + d.hc * dst_h];
                    }
                }
            }
            let ins_t = [
                (xi, &x[..]),
                (ri, &residual[..]),
                (pi, &post[..]),
                (ci, &tcomb[..]),
            ];
            let ct = run(&cpu, &g, &ins_t, &[], dst, n);
            let moved = maxerr(&c, &ct) as f64 / scale_of;
            println!("HyperConnectPost {name}: transposed comb moves the output by rel={moved:e}");
            assert!(
                moved > 1e-3,
                "{name}: transposing comb changed the output by only {moved:e} — this case cannot \
             detect the index formula being backwards"
            );
        }

        if let Some(vk) = gpu() {
            let v = go(&vk);
            let e = maxerr(&v, &c) as f64 / scale_of;
            println!("HyperConnectPost {name}: vulkan vs cpu rel={e:e}");
            assert!(
                e < HC_TOL as f64,
                "{name}: Vulkan HyperConnectPost diverges from CPU ({e:e})"
            );
        }
    }
}

/// The `hc_mult` bound is the ONLY thing keeping the GPU kernels' fixed-size per-token matrix in
/// range (an out-of-range private-array write is undefined, and neither kernel can refuse its own
/// dispatch), so it has to be shown to fire rather than assumed. `HYPER_CONNECT_MAX_MULT + 1`
/// must be refused — loudly, before anything is dispatched — on every backend.
///
/// CPU refuses by panicking (the interpreter has no error channel in the op loop, like every other
/// precondition there); Vulkan refuses with an `Err` from compile or execute.
#[test]
fn hyper_connect_refuses_hc_above_the_bound() {
    let hc = infr_core::graph::HYPER_CONNECT_MAX_MULT as usize + 1;
    let (rows, md) = (2usize, (2 + hc) * hc);
    let build = || {
        let mut g = Graph::new();
        let mx = g.input(f32d(rows * md));
        let sc = g.weight(f32d(3));
        let bs = g.weight(f32d(md));
        let pre = g.output(f32d(rows * hc));
        let gates = infr_core::graph::HyperGates {
            post: g.output(f32d(rows * hc)),
            comb: g.output(f32d(rows * hc * hc)),
        };
        g.push(Op::HyperConnectMix {
            mixes: mx,
            scale: sc,
            base: bs,
            pre,
            gates: Some(gates),
            rows: rows as u32,
            hc: hc as u32,
            eps: 1e-6,
            n_iter: 3,
        });
        (g, mx, sc, bs, pre, gates)
    };
    let go = |be: &dyn Backend| -> Result<(), infr_core::error::Error> {
        let (g, mx, sc, bs, pre, gates) = build();
        let plan = be.compile(&g)?;
        let alloc = |n: usize| be.alloc(n * 4, BufferUsage::Activations).unwrap();
        let (mb, sb, bb) = (alloc(rows * md), alloc(3), alloc(md));
        let (pb, ob, cb) = (alloc(rows * hc), alloc(rows * hc), alloc(rows * hc * hc));
        let mut b = Bindings::new();
        b.bind(mx, mb.as_ref());
        b.bind(sc, sb.as_ref());
        b.bind(bs, bb.as_ref());
        b.bind(pre, pb.as_ref());
        b.bind(gates.post, ob.as_ref());
        b.bind(gates.comb, cb.as_ref());
        be.execute(plan.as_ref(), &b)
    };

    let cpu = std::panic::catch_unwind(|| go(&infr_cpu::CpuBackend::new()));
    assert!(
        cpu.is_err(),
        "CPU accepted hc = {hc}, past HYPER_CONNECT_MAX_MULT"
    );
    if let Some(vk) = gpu() {
        let e = go(&vk).expect_err("Vulkan accepted hc past HYPER_CONNECT_MAX_MULT");
        let msg = e.to_string();
        println!("Vulkan refusal: {msg}");
        assert!(
            msg.contains("hc_mult"),
            "Vulkan refused hc = {hc} for the wrong reason: {msg}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// DeepSeek V4: per-layer SwiGLU clamping + hash-routed MoE (`docs/deepseek.md` § Stage 4, 8-10).
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Hand-written f64 reference for one clamped gated-FFN element, written from the DEFINITION in
/// `llm_graph_context::build_ffn` / `build_moe_ffn` (llama.cpp `src/llama-graph.cpp`, the
/// `LLM_FFN_SILU` arms' `arch == LLM_ARCH_DEEPSEEK4` branch), NOT transcribed from infr's CPU arm:
///
/// ```text
/// up   = ggml_clamp(up,   -limit, +limit);          // symmetric
/// gate = ggml_clamp(gate, -INFINITY, limit);        // one-sided, UPPER bound only
/// out  = silu(gate) * up;                            // clamp is BEFORE the activation
/// ```
///
/// Every other arch llama.cpp clamps runs `silu` first and clamps the RESULT; V4 does not.
fn swiglu_clamp_ref(gate: f64, up: f64, limit: Option<f64>) -> f64 {
    let silu = |z: f64| z / (1.0 + (-z).exp());
    match limit {
        None => silu(gate) * up,
        Some(l) => {
            let u = up.clamp(-l, l);
            let g = gate.min(l); // ggml_clamp(gate, -INFINITY, limit)
            silu(g) * u
        }
    }
}

/// The gate values the clamp cases run on. Chosen so the pre-SiLU and post-SiLU orders genuinely
/// disagree: values ABOVE the limit (where clamping the gate changes what SiLU sees) and values
/// BELOW ZERO, which is where SiLU's non-monotone lobe makes the two orders diverge most —
/// `silu(-4) = -0.072`, so clamping after SiLU at limit 0.5 leaves it untouched while clamping
/// before does nothing there either, but at limit `-0.05` (exercised by `limit` sweeps below) the
/// orders separate. The `up` values straddle ±limit so the symmetric bound bites on both signs.
const CLAMP_GATES: [f32; 12] = [
    -6.0, -4.0, -2.5, -1.5, -0.75, -0.25, 0.1, 0.6, 1.2, 2.0, 4.0, 9.0,
];
const CLAMP_UPS: [f32; 12] = [
    3.0, -3.0, 0.4, -0.4, 1.1, -1.1, 0.9, -0.9, 5.0, -5.0, 0.05, -0.05,
];

/// `Op::GatedAct` and `Op::GatedActFused` with DeepSeek V4's per-layer SwiGLU clamp, on the CPU
/// reference and (when a GPU is present) on Vulkan, against `swiglu_clamp_ref`.
///
/// Four things this pins, each shown red before green by injecting the deviation into the CPU arm
/// (`infr-cpu`'s `gated_act_fn`) — see the commit message for the pasted failures:
/// 1. the gate clamp is PRE-activation, not post;
/// 2. the gate clamp is ONE-SIDED (upper bound only);
/// 3. the `up` clamp is SYMMETRIC;
/// 4. `infr_core::graph::swiglu_clamp(limit)` disables at `limit <= 1e-6`, and the disabled path
///    is BIT-identical to a graph built with no clamp at all.
#[test]
fn swiglu_clamp_gated_act_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    let rows = 4usize;
    let nff = CLAMP_GATES.len();
    // One row per (gate, up) rotation so every gate meets several `up` signs/magnitudes.
    let gi: Vec<f32> = (0..rows * nff)
        .map(|i| CLAMP_GATES[i % nff])
        .collect::<Vec<_>>();
    let ui: Vec<f32> = (0..rows * nff)
        .map(|i| CLAMP_UPS[(i + i / nff) % nff])
        .collect();
    let gu: Vec<f32> = (0..rows)
        .flat_map(|r| {
            gi[r * nff..(r + 1) * nff]
                .iter()
                .chain(&ui[r * nff..(r + 1) * nff])
                .copied()
                .collect::<Vec<f32>>()
        })
        .collect();

    for limit in [0.5f32, 1.0, 3.0] {
        let clamp = infr_core::graph::swiglu_clamp(limit);
        assert_eq!(clamp, Some(limit), "limit {limit} must clamp");
        let want: Vec<f64> = (0..rows * nff)
            .map(|i| swiglu_clamp_ref(gi[i] as f64, ui[i] as f64, Some(limit as f64)))
            .collect();

        // Split gate/up form.
        let mut g = Graph::new();
        let gate = g.input(f32d(rows * nff));
        let up = g.input(f32d(rows * nff));
        let dst = g.output(f32d(rows * nff));
        g.push(Op::GatedAct {
            gate,
            up,
            dst,
            rows: rows as u32,
            nff: nff as u32,
            act: Activation::Silu,
            up_off: 0,
            up_stride: 0,
            gate_stride: 0,
            gate_block_width: 0,
            swiglu_clamp: clamp,
        });
        let ins = [(gate, &gi[..]), (up, &ui[..])];
        let c = run(&cpu, &g, &ins, &[], dst, rows * nff);
        let e = maxerr64(&c, &want);
        println!("GatedAct(clamp={limit}) cpu-vs-ref max_err={e:e}");
        assert!(e < 1e-6, "GatedAct clamp={limit} diverges: max_err={e:e}");
        if let Some(vk) = gpu() {
            let v = run(&vk, &g, &ins, &[], dst, rows * nff);
            let e = maxerr64(&v, &want);
            println!("GatedAct(clamp={limit}) vulkan-vs-ref max_err={e:e}");
            assert!(e < 1e-5, "GatedAct clamp={limit} diverges on Vulkan: {e:e}");
        }

        // Fused [rows, 2*nff] gate|up form — same reference, different buffer layout.
        let mut g2 = Graph::new();
        let gub = g2.input(f32d(rows * 2 * nff));
        let dst2 = g2.output(f32d(rows * nff));
        g2.push(Op::GatedActFused {
            gu: gub,
            dst: dst2,
            rows: rows as u32,
            nff: nff as u32,
            act: Activation::Silu,
            swiglu_clamp: clamp,
        });
        let ins2 = [(gub, &gu[..])];
        let c2 = run(&cpu, &g2, &ins2, &[], dst2, rows * nff);
        let e = maxerr64(&c2, &want);
        println!("GatedActFused(clamp={limit}) cpu-vs-ref max_err={e:e}");
        assert!(e < 1e-6, "GatedActFused clamp={limit} diverges: {e:e}");
        if let Some(vk) = gpu() {
            let v = run(&vk, &g2, &ins2, &[], dst2, rows * nff);
            let e = maxerr64(&v, &want);
            println!("GatedActFused(clamp={limit}) vulkan-vs-ref max_err={e:e}");
            assert!(e < 1e-5, "GatedActFused clamp={limit} on Vulkan: {e:e}");
        }
    }
}

/// The pre-SiLU vs post-SiLU orders are not interchangeable, and the two one-sidedness choices are
/// not either — stated as a property of the REFERENCE so the numbers this suite asserts on are
/// known to be capable of separating the variants. Without this, a backend that clamped in the
/// wrong order could still pass `swiglu_clamp_gated_act_parity` if the inputs happened not to
/// distinguish them.
///
/// One thing worth writing down because it is the opposite of what one expects: for a POSITIVE
/// limit the two orders agree everywhere the gate is negative. `silu(g) < 0 < limit` there, so
/// `min(silu(g), limit) == silu(g) == silu(min(g, limit))` — the whole negative lobe is inert, and
/// the orders separate only where `gate > limit`. The separation therefore grows with the limit
/// (`|silu(l) - l|` times the clamped `up`), not with how far the gate reaches below zero.
#[test]
fn swiglu_clamp_orders_are_distinguishable() {
    let silu = |z: f64| z / (1.0 + (-z).exp());
    for limit in [0.5f64, 1.0, 3.0] {
        // Post-SiLU (every arch but V4): silu first, then clamp the ACTIVATION.
        let post = |gate: f64, up: f64| silu(gate).min(limit) * up.clamp(-limit, limit);
        // Symmetric gate clamp instead of one-sided.
        let sym = |gate: f64, up: f64| silu(gate.clamp(-limit, limit)) * up.clamp(-limit, limit);
        // One-sided `up` clamp instead of symmetric.
        let one_up = |gate: f64, up: f64| silu(gate.min(limit)) * up.min(limit);

        let (mut d_post, mut d_sym, mut d_up) = (0.0f64, 0.0f64, 0.0f64);
        let mut neg_lobe = 0.0f64;
        // Full cross product, matching the parity test's row rotation: pairing each gate with ONE
        // `up` would let the largest gate land on the smallest `|up|` and understate every gap.
        for &gz in CLAMP_GATES.iter() {
            for &uz in CLAMP_UPS.iter() {
                let (gz, uz) = (gz as f64, uz as f64);
                let want = swiglu_clamp_ref(gz, uz, Some(limit));
                d_post = d_post.max((want - post(gz, uz)).abs());
                d_sym = d_sym.max((want - sym(gz, uz)).abs());
                d_up = d_up.max((want - one_up(gz, uz)).abs());
                if gz < 0.0 {
                    neg_lobe = neg_lobe.max((want - post(gz, uz)).abs());
                }
            }
        }
        println!(
            "clamp variant separation @limit={limit}: post-SiLU={d_post:e} sym-gate={d_sym:e} \
             one-sided-up={d_up:e} (negative-gate-only post-SiLU={neg_lobe:e})"
        );
        assert!(
            d_post > 1e-2,
            "post-SiLU clamp indistinguishable @{limit}: {d_post:e}"
        );
        assert!(
            d_sym > 1e-2,
            "symmetric gate clamp indistinguishable @{limit}: {d_sym:e}"
        );
        assert!(
            d_up > 1e-2,
            "one-sided up clamp indistinguishable @{limit}: {d_up:e}"
        );
        assert_eq!(
            neg_lobe, 0.0,
            "a positive limit cannot separate the orders on a negative gate — if this fires the \
             reasoning above is wrong, not the number"
        );
    }
}

/// `limit <= 1e-6` is llama.cpp's DISABLED state, not "clamp everything to zero".
/// [`infr_core::graph::swiglu_clamp`] is the single place that gate lives, and a graph built from a
/// disabled per-layer entry must be BIT-identical to one with no clamp field set at all.
#[test]
fn swiglu_clamp_disabled_is_bit_identical() {
    for off in [0.0f32, 1e-9, 1e-7, 1e-6] {
        assert_eq!(
            infr_core::graph::swiglu_clamp(off),
            None,
            "limit {off} must read as DISABLED (llama.cpp gates on `limit > 1e-6`)"
        );
    }
    assert_eq!(infr_core::graph::swiglu_clamp(1.1e-6), Some(1.1e-6));

    let cpu = infr_cpu::CpuBackend::new();
    let (rows, nff) = (4usize, CLAMP_GATES.len());
    let gi: Vec<f32> = (0..rows * nff).map(|i| CLAMP_GATES[i % nff]).collect();
    let ui: Vec<f32> = (0..rows * nff).map(|i| CLAMP_UPS[i % nff]).collect();
    let build = |clamp: Option<f32>| {
        let mut g = Graph::new();
        let gate = g.input(f32d(rows * nff));
        let up = g.input(f32d(rows * nff));
        let dst = g.output(f32d(rows * nff));
        g.push(Op::GatedAct {
            gate,
            up,
            dst,
            rows: rows as u32,
            nff: nff as u32,
            act: Activation::Silu,
            up_off: 0,
            up_stride: 0,
            gate_stride: 0,
            gate_block_width: 0,
            swiglu_clamp: clamp,
        });
        (g, gate, up, dst)
    };
    let go = |be: &dyn Backend, clamp: Option<f32>| -> Vec<f32> {
        let (g, gate, up, dst) = build(clamp);
        run(
            be,
            &g,
            &[(gate, &gi[..]), (up, &ui[..])],
            &[],
            dst,
            rows * nff,
        )
    };
    // `swiglu_clamp(0.0)` is the per-layer array entry a non-clamping V4 layer carries.
    let disabled = go(&cpu, infr_core::graph::swiglu_clamp(0.0));
    let none = go(&cpu, None);
    assert_eq!(
        disabled.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        none.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a disabled clamp must be bit-identical to no clamp on CPU"
    );
    if let Some(vk) = gpu() {
        let d = go(&vk, infr_core::graph::swiglu_clamp(0.0));
        let n = go(&vk, None);
        assert_eq!(
            d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            n.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a disabled clamp must be bit-identical to no clamp on Vulkan"
        );
    }
}

/// Hand-written f64 reference for a hash-routed / clamped `Op::MoeFfn`, written from
/// `deepseek4.cpp`'s MoE block plus `llm_graph_context::build_moe_ffn` (llama.cpp
/// `src/llama-graph.cpp`), NOT transcribed from infr's CPU arm.
///
/// `expert_ids = Some(ids)` is llama.cpp's `selected_experts_in`. Read off `build_moe_ffn`: the
/// router matmul STILL runs and the gating function still produces `probs`; only `ggml_argsort_top_k`
/// (and the `selection_probs` it consumes) is skipped. The routing weights stay
/// `ggml_get_rows(probs, selected_experts)` — the ROUTER's probability at each hash-chosen expert —
/// then `norm_w` renormalizes over the selected set and `w_scale` scales. They are NOT uniform.
#[allow(clippy::too_many_arguments)]
fn moe_v4_ref(
    x: &[f32],
    router: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    ne: usize,
    n_expert: usize,
    n_used: usize,
    n_ff_exp: usize,
    scale: f64,
    expert_ids: Option<&[usize]>,
    swiglu_clamp: Option<f64>,
) -> Vec<f64> {
    let rows = x.len() / ne;
    let dot = |w: &[f32], v: &[f64]| w.iter().zip(v).map(|(a, b)| *a as f64 * b).sum::<f64>();
    let mut out = vec![0f64; rows * ne];
    for row in 0..rows {
        let xr: Vec<f64> = x[row * ne..(row + 1) * ne]
            .iter()
            .map(|&v| v as f64)
            .collect();
        // Router logits → sqrt(softplus) probs (V4's mandatory gating).
        let probs: Vec<f64> = (0..n_expert)
            .map(|e| {
                let l = dot(&router[e * ne..(e + 1) * ne], &xr);
                (1.0 + l.exp()).ln().sqrt()
            })
            .collect();
        let idx: Vec<usize> = match expert_ids {
            Some(ids) => ids[row * n_used..(row + 1) * n_used].to_vec(),
            None => {
                let mut i: Vec<usize> = (0..n_expert).collect();
                i.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                i.truncate(n_used);
                i
            }
        };
        let wsum: f64 = idx.iter().map(|&e| probs[e]).sum::<f64>().max(1e-20);
        for &e in &idx {
            let gs = e * n_ff_exp * ne;
            let ds = e * ne * n_ff_exp;
            let actv: Vec<f64> = (0..n_ff_exp)
                .map(|j| {
                    let gv = dot(&gate[gs + j * ne..gs + (j + 1) * ne], &xr);
                    let uv = dot(&up[gs + j * ne..gs + (j + 1) * ne], &xr);
                    swiglu_clamp_ref(gv, uv, swiglu_clamp)
                })
                .collect();
            let w_e = probs[e] / wsum * scale;
            for i in 0..ne {
                out[row * ne + i] += w_e
                    * down[ds + i * n_ff_exp..ds + (i + 1) * n_ff_exp]
                        .iter()
                        .zip(&actv)
                        .map(|(a, b)| *a as f64 * b)
                        .sum::<f64>();
            }
        }
    }
    out
}

/// Two tokens whose `ffn_gate_tid2eid` rows name DIFFERENT experts, plus a per-layer SwiGLU clamp
/// on the routed experts. Pins, red-then-green (deviations injected into the CPU arm — see the
/// commit message):
/// * the ids drive the selection (falling back to top-k routes both rows the same and goes red);
/// * the routing WEIGHTS are the router's own `sqrt(softplus)` probs at the hash ids, renormalized
///   — the other plausible reading, uniform `1/n_used`, is asserted to differ and to fail;
/// * the routed-expert clamp is the same pre-SiLU / one-sided-gate arithmetic as the dense path.
///
/// `ne`/`n_ff_exp` = 32: the Vulkan expert id-GEMV decodes 32-element sub-blocks, so anything
/// smaller would cross-check against a silent all-zero GPU output.
#[test]
fn moe_hash_routing_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    let (ne, n_expert, n_used, n_ff_exp) = (32usize, 6usize, 2usize, 32usize);
    let rows = 2usize;
    // Row 0 → experts {4, 1}; row 1 → experts {0, 5}. Disjoint, and NEITHER is the top-2 the
    // router would pick (the router's lead terms below rank 0 > 1 > 2 > 3 > 5 > 4 for both rows).
    let ids: [usize; 4] = [4, 1, 0, 5];
    // The handle is I32; `run` uploads an f32 slice verbatim, so carry the i32 BYTES through the
    // f32 wire type. Backends widen them back to plain integers (`bytes_to_f32`'s I32 arm).
    let idwire: Vec<f32> = ids.iter().map(|&e| f32::from_bits(e as u32)).collect();
    let limit = 0.5f32;

    let build = |hash: bool, clamp: Option<f32>| {
        let mut g = Graph::new();
        let x = g.input(f32d(rows * ne));
        let router_x = g.input(f32d(rows * ne));
        let eids = g.input(TensorDesc::new(vec![rows, n_used], DType::I32));
        let router = g.weight(f32d(n_expert * ne));
        let gate_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
        let up_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
        let down_exps = g.weight(f32d(n_expert * ne * n_ff_exp));
        let dst = g.output(f32d(rows * ne));
        g.push(Op::MoeFfn {
            x,
            router_x,
            router,
            gate_exps,
            up_exps,
            down_exps,
            down_scale: None,
            fused_gate_up: false,
            dst,
            ne: ne as u32,
            n_expert: n_expert as u32,
            n_used: n_used as u32,
            n_ff_exp: n_ff_exp as u32,
            scale: 1.0,
            act: Activation::Silu,
            swiglu_clamp: clamp,
            gating: MoeGating::SqrtSoftplus,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
            expert_ids: hash.then_some(eids),
        });
        (
            g, x, router_x, eids, router, gate_exps, up_exps, down_exps, dst,
        )
    };

    // Router rows = lead[e] * [1, 0, …]; `x` rows differ in their first element, so the two tokens
    // get different logits (and different weights) while the top-2 ranking stays 0, 1 for both.
    let lead = [3.0f32, 2.0, 1.5, 1.0, -1.0, 0.5];
    let mut xi = gen(rows * ne, 21);
    xi[0] = 1.0;
    xi[ne] = 0.6;
    let ri: Vec<f32> = (0..n_expert * ne)
        .map(|i| if i % ne == 0 { lead[i / ne] } else { 0.0 })
        .collect();
    // Scaled up so the gate/up projections land well outside ±limit and the clamp actually bites.
    let gi: Vec<f32> = gen(n_expert * n_ff_exp * ne, 12)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let ui: Vec<f32> = gen(n_expert * n_ff_exp * ne, 13)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let di = gen(n_expert * ne * n_ff_exp, 14);

    let go = |be: &dyn Backend, hash: bool, clamp: Option<f32>| -> Vec<f32> {
        let (g, x, rx, eids, router, ge, ue, de, dst) = build(hash, clamp);
        run(
            be,
            &g,
            &[(x, &xi[..]), (rx, &xi[..]), (eids, &idwire[..])],
            &[
                (router, &ri[..]),
                (ge, &gi[..]),
                (ue, &ui[..]),
                (de, &di[..]),
            ],
            dst,
            rows * ne,
        )
    };
    let reference = |hash: bool, clamp: Option<f32>| -> Vec<f64> {
        moe_v4_ref(
            &xi,
            &ri,
            &gi,
            &ui,
            &di,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            1.0,
            hash.then_some(&ids[..]),
            clamp.map(|l| l as f64),
        )
    };

    for clamp in [None, infr_core::graph::swiglu_clamp(limit)] {
        let want = reference(true, clamp);
        let c = go(&cpu, true, clamp);
        let e = maxerr64(&c, &want);
        println!("MoeFfn(hash, clamp={clamp:?}) cpu-vs-ref max_err={e:e}");
        assert!(
            e < 1e-4,
            "hash-routed MoeFfn diverges (clamp={clamp:?}): {e:e}"
        );
        if let Some(vk) = gpu() {
            let v = go(&vk, true, clamp);
            let e = maxerr(&c, &v);
            println!("MoeFfn(hash, clamp={clamp:?}) cpu-vs-vulkan max_err={e:e}");
            assert!(e < 1e-3, "hash-routed MoeFfn diverges on Vulkan: {e:e}");
        }
    }

    // The ids must actually SELECT: top-k routing over the same inputs picks experts {0, 1} for
    // both rows, so ignoring them is a visibly different output rather than a no-op.
    let hashed = go(&cpu, true, None);
    let topk = go(&cpu, false, None);
    let sep = maxerr(&hashed, &topk);
    println!("hash-vs-topk separation: {sep:e}");
    assert!(
        sep > 1e-2,
        "hash ids and top-k route to the same experts here — the test cannot fail"
    );
    // And the two ROWS must route differently from each other: rows 0/1 share `x` up to their
    // first element, so with a single shared selection their outputs would be near-identical.
    let cross = maxerr(&hashed[..ne], &hashed[ne..]);
    println!("row0-vs-row1 separation: {cross:e}");
    assert!(
        cross > 1e-2,
        "the two tokens' hash rows did not route differently"
    );

    // The WEIGHTS are the router's probs at the hash ids, not uniform. Assert the other plausible
    // reading is measurably different and does NOT match the backend.
    let uniform: Vec<f64> = {
        let mut r = vec![0f64; rows * ne];
        for row in 0..rows {
            let xr: Vec<f64> = xi[row * ne..(row + 1) * ne]
                .iter()
                .map(|&v| v as f64)
                .collect();
            for k in 0..n_used {
                let e = ids[row * n_used + k];
                let (gs, ds) = (e * n_ff_exp * ne, e * ne * n_ff_exp);
                let actv: Vec<f64> = (0..n_ff_exp)
                    .map(|j| {
                        let gv: f64 = gi[gs + j * ne..gs + (j + 1) * ne]
                            .iter()
                            .zip(&xr)
                            .map(|(a, b)| *a as f64 * b)
                            .sum();
                        let uv: f64 = ui[gs + j * ne..gs + (j + 1) * ne]
                            .iter()
                            .zip(&xr)
                            .map(|(a, b)| *a as f64 * b)
                            .sum();
                        swiglu_clamp_ref(gv, uv, None)
                    })
                    .collect();
                for i in 0..ne {
                    r[row * ne + i] += (1.0 / n_used as f64)
                        * di[ds + i * n_ff_exp..ds + (i + 1) * n_ff_exp]
                            .iter()
                            .zip(&actv)
                            .map(|(a, b)| *a as f64 * b)
                            .sum::<f64>();
                }
            }
        }
        r
    };
    let d_uniform = maxerr64(&hashed, &uniform);
    println!("hash weights: router-probs-vs-uniform separation={d_uniform:e}");
    assert!(
        d_uniform > 1e-2,
        "uniform 1/n_used weights are indistinguishable from the router probs here — the \
         weight-source assertion cannot fail"
    );
}

/// `Op::GatherI32` feeding `Op::MoeFfn::expert_ids` — the two-op chain a DeepSeek V4 hash-routed
/// layer emits, against the same from-definition f64 reference [`moe_hash_routing_parity`] uses.
///
/// The reference is fed ids gathered ON THE HOST out of the same table bytes, so the assertion
/// pins the GATHERED IDS themselves, not merely that something ran: one wrong id runs a different
/// expert, and the separation that buys is measured and printed below (`right-row-vs-row-0`)
/// rather than assumed. Both backends are held to the same reference, so CPU and Vulkan are
/// cross-checked on the ids as well as on the MoE arithmetic.
#[test]
fn gather_i32_hash_selection_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    let (ne, n_expert, n_used, n_ff_exp) = (32usize, 6usize, 2usize, 32usize);
    let (rows, vocab) = (2usize, 8usize);
    // The two tokens this "prompt" feeds. Neither is row 0, so a gather that ignored `ids` and read
    // the first row would be caught, and they differ from each other, so one that read a single
    // shared row would be too.
    let toks: [i32; 2] = [5, 2];
    // `[n_used, vocab]` table: token t names experts {(3t) % 6, (5t + 1) % 6}. Every row is
    // distinct, so gathering the wrong ROW is a different selection.
    let table: Vec<i32> = (0..vocab)
        .flat_map(|t| [((3 * t) % n_expert) as i32, ((5 * t + 1) % n_expert) as i32])
        .collect();
    // The i32 handles ride the f32 wire type `run` uploads (see `moe_hash_routing_parity`).
    let wire = |v: &[i32]| -> Vec<f32> { v.iter().map(|&e| f32::from_bits(e as u32)).collect() };
    let (tokwire, tabwire) = (wire(&toks), wire(&table));
    // What the gather must produce: row r = the table row of token `toks[r]`.
    let want_ids: Vec<usize> = toks
        .iter()
        .flat_map(|&t| {
            let b = t as usize * n_used;
            table[b..b + n_used].iter().map(|&e| e as usize)
        })
        .collect();
    println!("gather_i32 expects ids {want_ids:?} for tokens {toks:?}");

    let build = || {
        let mut g = Graph::new();
        let x = g.input(f32d(rows * ne));
        let ids = g.input(TensorDesc::new(vec![rows], DType::I32));
        let tbl = g.weight(TensorDesc::new(vec![n_used, vocab], DType::I32));
        let sel = g.internal(TensorDesc::new(vec![rows, n_used], DType::I32));
        let router = g.weight(f32d(n_expert * ne));
        let gate_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
        let up_exps = g.weight(f32d(n_expert * n_ff_exp * ne));
        let down_exps = g.weight(f32d(n_expert * ne * n_ff_exp));
        let dst = g.output(f32d(rows * ne));
        g.push(Op::GatherI32 {
            ids,
            table: tbl,
            dst: sel,
            rows: rows as u32,
            ne: n_used as u32,
        });
        g.push(Op::MoeFfn {
            x,
            router_x: x,
            router,
            gate_exps,
            up_exps,
            down_exps,
            down_scale: None,
            fused_gate_up: false,
            dst,
            ne: ne as u32,
            n_expert: n_expert as u32,
            n_used: n_used as u32,
            n_ff_exp: n_ff_exp as u32,
            scale: 1.0,
            act: Activation::Silu,
            swiglu_clamp: None,
            gating: MoeGating::SqrtSoftplus,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
            expert_ids: Some(sel),
        });
        (g, x, ids, tbl, router, gate_exps, up_exps, down_exps, dst)
    };

    let mut xi = gen(rows * ne, 21);
    xi[0] = 1.0;
    xi[ne] = 0.6;
    let ri = gen(n_expert * ne, 5);
    let gi = gen(n_expert * n_ff_exp * ne, 12);
    let ui = gen(n_expert * n_ff_exp * ne, 13);
    let di = gen(n_expert * ne * n_ff_exp, 14);

    let go = |be: &dyn Backend| -> Vec<f32> {
        let (g, x, ids, tbl, router, ge, ue, de, dst) = build();
        run(
            be,
            &g,
            &[(x, &xi[..]), (ids, &tokwire[..])],
            &[
                (tbl, &tabwire[..]),
                (router, &ri[..]),
                (ge, &gi[..]),
                (ue, &ui[..]),
                (de, &di[..]),
            ],
            dst,
            rows * ne,
        )
    };
    let reference = |ids: &[usize]| -> Vec<f64> {
        moe_v4_ref(
            &xi,
            &ri,
            &gi,
            &ui,
            &di,
            ne,
            n_expert,
            n_used,
            n_ff_exp,
            1.0,
            Some(ids),
            None,
        )
    };

    let want = reference(&want_ids);
    let c = go(&cpu);
    let e = maxerr64(&c, &want);
    println!("GatherI32+MoeFfn cpu-vs-ref max_err={e:e}");
    assert!(e < 1e-4, "the CPU gather selected different experts: {e:e}");

    // The separation that makes the assertion above meaningful: reading the WRONG table row (token
    // 0's, for both output rows) is a different answer by a whole signal, not a rounding.
    let wrong: Vec<usize> = (0..rows)
        .flat_map(|_| table[..n_used].iter().map(|&e| e as usize))
        .collect();
    let sep = maxerr64(&c, &reference(&wrong));
    println!("GatherI32 right-row-vs-row-0 separation={sep:e} (wrong ids {wrong:?})");
    assert!(
        sep > 1e-2,
        "gathering row 0 instead of the token's row is indistinguishable here — the test cannot \
         fail"
    );

    if let Some(vk) = gpu() {
        let v = go(&vk);
        let e = maxerr64(&v, &want);
        println!("GatherI32+MoeFfn vulkan-vs-ref max_err={e:e}");
        assert!(
            e < 1e-3,
            "the Vulkan gather selected different experts: {e:e}"
        );
        let cv = maxerr(&c, &v);
        println!("GatherI32+MoeFfn cpu-vs-vulkan max_err={cv:e}");
        assert!(
            cv < 1e-3,
            "GatherI32 diverges between CPU and Vulkan: {cv:e}"
        );
    }
}
