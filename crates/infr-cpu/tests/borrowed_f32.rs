#![cfg(feature = "f32-gemm")]
use infr_core::{config::Config, Backend, Bindings, BufferUsage, DType, Graph, Op, TensorDesc};
use infr_cpu::CpuBackend;
use std::sync::Arc;

#[test]
fn borrowed_f32_matches_graph_with_short_dimensions_and_tails() {
    for (rows, input, output) in [
        (1, 7, 9),
        (4, 32, 7),
        (8, 7, 8),
        (9, 33, 9),
        (32, 96, 192),
        (128, 384, 128),
        (16, 2816, 256),
    ] {
        let cpu = CpuBackend::new_with_thread_limit(Arc::new(Config::default()), 2)
            .unwrap()
            .with_f32_gemm()
            .unwrap();
        let x = (0..rows * input)
            .map(|i| (i as f32 * 0.071).sin() * 0.25)
            .collect::<Vec<_>>();
        let w = (0..output * input)
            .map(|i| (i as f32 * 0.037).cos() * 0.125)
            .collect::<Vec<_>>();
        let xb = cpu.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
        let wb = cpu.alloc(w.len() * 4, BufferUsage::Weights).unwrap();
        let yb = cpu
            .alloc(rows * output * 4, BufferUsage::Activations)
            .unwrap();
        cpu.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        cpu.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
        let mut graph = Graph::new();
        let xi = graph.input(TensorDesc::new(vec![rows, input], DType::F32));
        let wi = graph.weight(TensorDesc::new(vec![output, input], DType::F32));
        let yi = graph.output(TensorDesc::new(vec![rows, output], DType::F32));
        graph.push(Op::Linear {
            x: xi,
            weight: wi,
            dst: yi,
            m: rows as u32,
            in_f: input as u32,
            out_f: output as u32,
            w_off: 0,
        });
        let plan = cpu.compile(&graph).unwrap();
        let mut bindings = Bindings::new();
        bindings
            .bind(xi, xb.as_ref())
            .bind(wi, wb.as_ref())
            .bind(yi, yb.as_ref());
        cpu.execute(plan.as_ref(), &bindings).unwrap();
        let mut expected = vec![0f32; rows * output];
        cpu.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut expected))
            .unwrap();
        let mut actual = vec![0f32; rows * output];
        cpu.linear_f32_into(&x, rows, input, wb.as_ref(), output, &mut actual)
            .unwrap();
        assert_eq!(actual, expected, "shape [{rows},{input},{output}]");
    }
}

#[test]
fn borrowed_f32_is_opt_in_and_rejects_bad_shapes_before_writing() {
    let cfg = Arc::new(Config::default());
    let cpu = CpuBackend::new_with_thread_limit(cfg.clone(), 1).unwrap();
    let wb = cpu.alloc(4, BufferUsage::Weights).unwrap();
    cpu.upload(wb.as_ref(), bytemuck::cast_slice(&[2f32]))
        .unwrap();
    let mut out = [99f32];
    assert!(cpu
        .linear_f32_into(&[3.], 1, 1, wb.as_ref(), 1, &mut out)
        .is_err());
    assert_eq!(out, [99.]);
    let cpu = CpuBackend::new_with_thread_limit(cfg, 1)
        .unwrap()
        .with_f32_gemm()
        .unwrap();
    assert!(cpu
        .linear_f32_into(&[], 1, 1, wb.as_ref(), 1, &mut out)
        .is_err());
    assert!(cpu
        .linear_f32_into(&[], usize::MAX, 2, wb.as_ref(), 1, &mut out)
        .is_err());
    assert!(cpu
        .linear_f32_into(&[1., 2.], 1, 2, wb.as_ref(), 1, &mut out)
        .is_err());
    assert_eq!(out, [99.]);
}

#[test]
fn borrowed_f32_respects_reference_accumulation_order() {
    let mut cfg = Config::default();
    cfg.kernels.cpu.reference = true;
    let cpu = CpuBackend::new_with_thread_limit(Arc::new(cfg), 2)
        .unwrap()
        .with_f32_gemm()
        .unwrap();
    let (rows, input, output) = (9, 33, 9);
    let x = (0..rows * input)
        .map(|i| (i as f32 * 0.09).sin())
        .collect::<Vec<_>>();
    let w = (0..output * input)
        .map(|i| (i as f32 * 0.07).cos())
        .collect::<Vec<_>>();
    let xb = cpu.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
    let wb = cpu.alloc(w.len() * 4, BufferUsage::Weights).unwrap();
    let yb = cpu
        .alloc(rows * output * 4, BufferUsage::Activations)
        .unwrap();
    cpu.upload(xb.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    cpu.upload(wb.as_ref(), bytemuck::cast_slice(&w)).unwrap();
    let mut graph = Graph::new();
    let xi = graph.input(TensorDesc::new(vec![rows, input], DType::F32));
    let wi = graph.weight(TensorDesc::new(vec![output, input], DType::F32));
    let yi = graph.output(TensorDesc::new(vec![rows, output], DType::F32));
    graph.push(Op::Linear {
        x: xi,
        weight: wi,
        dst: yi,
        m: rows as u32,
        in_f: input as u32,
        out_f: output as u32,
        w_off: 0,
    });
    let plan = cpu.compile(&graph).unwrap();
    let mut bindings = Bindings::new();
    bindings
        .bind(xi, xb.as_ref())
        .bind(wi, wb.as_ref())
        .bind(yi, yb.as_ref());
    cpu.execute(plan.as_ref(), &bindings).unwrap();
    let mut expected = vec![0f32; rows * output];
    cpu.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut expected))
        .unwrap();
    let mut actual = vec![0.; rows * output];
    cpu.linear_f32_into(&x, rows, input, wb.as_ref(), output, &mut actual)
        .unwrap();
    assert_eq!(actual, expected);
}
