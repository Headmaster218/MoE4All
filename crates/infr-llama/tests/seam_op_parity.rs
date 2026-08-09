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
        weight: w,
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
/// ne/n_ff_exp = 32 (not tiny): the Vulkan expert id-GEMV decodes 32-element sub-blocks, so
/// smaller dims would make the cross-check compare against a silent all-zero GPU output.
#[test]
fn moe_sqrt_softplus_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    // ne/n_ff_exp ≥ 32: the Vulkan expert id-GEMV decodes 32-element sub-blocks
    // (`nsub = in_f/32`), so below that its MoE output is a silent all-zero no-op.
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
/// test. ne/n_ff_exp = 32 (not tiny): the Vulkan expert id-GEMV decodes 32-element sub-blocks, so
/// smaller dims would make the cross-check compare against a silent all-zero GPU output.
#[test]
fn moe_groups_bias_parity() {
    let cpu = infr_cpu::CpuBackend::new();
    // ne/n_ff_exp ≥ 32: the Vulkan expert id-GEMV decodes 32-element sub-blocks
    // (`nsub = in_f/32`), so below that its MoE output is a silent all-zero no-op.
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
