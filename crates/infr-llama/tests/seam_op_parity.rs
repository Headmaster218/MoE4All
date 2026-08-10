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
