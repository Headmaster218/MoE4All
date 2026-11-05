//! CPU-only mathematical and IR contracts for the stateless dense primitives.
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::config::Config;
use infr_core::graph::{Graph, Op, TensorKind};
use infr_core::{DType, TensorDesc, TensorId};
use infr_cpu::CpuBackend;
use std::sync::Arc;

fn backend() -> CpuBackend {
    CpuBackend::new_with(Arc::new(Config::default()))
}

fn buffer(be: &CpuBackend, values: &[f32]) -> Box<dyn Buffer> {
    let b = be.alloc(values.len() * 4, BufferUsage::Weights).unwrap();
    be.upload(b.as_ref(), bytemuck::cast_slice(values)).unwrap();
    b
}

fn download(be: &CpuBackend, b: &dyn Buffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0; n * 4];
    be.download(b, &mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn floats(values: &[f32], dtype: DType) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            DType::F32 => v.to_ne_bytes().to_vec(),
            DType::F16 => half::f16::from_f32(v).to_bits().to_ne_bytes().to_vec(),
            DType::Bf16 => half::bf16::from_f32(v).to_bits().to_ne_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect()
}

fn typed_buffer(be: &CpuBackend, values: &[f32], dtype: DType) -> Box<dyn Buffer> {
    let bytes = floats(values, dtype);
    let b = be.alloc(bytes.len(), BufferUsage::Weights).unwrap();
    be.upload(b.as_ref(), &bytes).unwrap();
    b
}

#[derive(Clone, Copy)]
enum Alias {
    Separate,
    X,
    Weight,
    All,
}

#[allow(clippy::too_many_arguments)]
fn conv(
    be: &CpuBackend,
    x: &[f32],
    w: &[f32],
    rows: u32,
    channels: u32,
    kernel: u32,
    left: u32,
    alias: Alias,
    dynamic_weight: bool,
    dtype: DType,
) -> Vec<f32> {
    let mut g = Graph::new();
    let xid = g.input(TensorDesc::new(
        vec![rows as usize, channels as usize],
        dtype,
    ));
    let wd = TensorDesc::new(vec![channels as usize, kernel as usize], dtype);
    let wid = match alias {
        Alias::All => xid,
        Alias::Weight => g.input(wd),
        _ if dynamic_weight => g.input(wd),
        _ => g.weight(wd),
    };
    let dst = match alias {
        Alias::X | Alias::All => xid,
        Alias::Weight => wid,
        Alias::Separate => g.output(TensorDesc::new(
            vec![rows as usize, channels as usize],
            DType::F32,
        )),
    };
    g.push(Op::DepthwiseConv1d {
        x: xid,
        weight: wid,
        dst,
        rows,
        channels,
        kernel,
        pad_left: left,
        pad_right: kernel - 1 - left,
    });
    let plan = be.compile(&g).unwrap();
    let xb = typed_buffer(be, x, dtype);
    let wb = typed_buffer(be, w, dtype);
    let out = buffer(be, &vec![0.0; x.len()]);
    let mut bindings = Bindings::new();
    bindings.bind(xid, xb.as_ref());
    if wid != xid {
        bindings.bind(wid, wb.as_ref());
    }
    if dst != xid && dst != wid {
        bindings.bind(dst, out.as_ref());
    }
    be.execute(plan.as_ref(), &bindings).unwrap();
    let got = download(be, bindings.get(dst).unwrap(), x.len());
    // Independent executes must have no hidden history. Rebind fresh input data after aliases.
    be.upload(xb.as_ref(), &floats(x, dtype)).unwrap();
    if wid != xid {
        be.upload(wb.as_ref(), &floats(w, dtype)).unwrap();
    }
    be.execute(plan.as_ref(), &bindings).unwrap();
    assert_eq!(download(be, bindings.get(dst).unwrap(), x.len()), got);
    got
}

/// A separately materialized zero-padded sequence and f64 dot, independent of CPU indexing.
fn reference(
    x: &[f32],
    w: &[f32],
    rows: usize,
    channels: usize,
    kernel: usize,
    left: usize,
) -> Vec<f32> {
    let mut padded = vec![0.0f64; (rows + kernel - 1) * channels];
    for r in 0..rows {
        for c in 0..channels {
            padded[(left + r) * channels + c] = x[r * channels + c] as f64;
        }
    }
    (0..rows * channels)
        .map(|i| {
            let (r, c) = (i / channels, i % channels);
            (0..kernel)
                .map(|j| padded[(r + j) * channels + c] * w[c * kernel + j] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-6 * (1.0 + w.abs()),
            "element {i}: got {g}, expected {w}"
        );
    }
}

#[test]
fn depthwise_matches_reference_for_all_padding_splits_and_shapes() {
    let be = backend();
    for rows in [1, 2, 5, 11] {
        for channels in [1, 2, 7] {
            for kernel in [1, 2, 3, 6] {
                let x: Vec<f32> = (0..rows * channels)
                    .map(|i| (i as f32 * 0.7).sin() - 0.3)
                    .collect();
                let w: Vec<f32> = (0..channels * kernel)
                    .map(|i| (i as f32 * 0.4).cos() - 0.6)
                    .collect();
                for left in 0..kernel {
                    let want = reference(
                        &x,
                        &w,
                        rows as usize,
                        channels as usize,
                        kernel as usize,
                        left as usize,
                    );
                    close(
                        &conv(
                            &be,
                            &x,
                            &w,
                            rows,
                            channels,
                            kernel,
                            left,
                            Alias::Separate,
                            false,
                            DType::F32,
                        ),
                        &want,
                    );
                }
            }
        }
    }
}

#[test]
fn padding_edges_orientation_and_no_activation_are_explicit() {
    let be = backend();
    let x = [1.0, 2.0, -3.0];
    let w = [10.0, 100.0, 1000.0];
    for (left, want) in [
        (0, [-2790.0, -280.0, -30.0]),
        (1, [2100.0, -2790.0, -280.0]),
        (2, [1000.0, 2100.0, -2790.0]),
    ] {
        assert_eq!(
            conv(
                &be,
                &x,
                &w,
                3,
                1,
                3,
                left,
                Alias::Separate,
                false,
                DType::F32
            ),
            want
        );
    }
    // Kernel wider than the entire input: only one tap can see the single input row.
    assert_eq!(
        conv(
            &be,
            &[-2.0],
            &[1.0, 2.0, 3.0, 4.0],
            1,
            1,
            4,
            2,
            Alias::Separate,
            false,
            DType::F32
        ),
        [-6.0]
    );
}

#[test]
fn convolution_supports_dynamic_weights_and_same_handle_aliases() {
    let be = backend();
    let x: Vec<f32> = (0..9).map(|i| i as f32 - 4.0).collect();
    let w: Vec<f32> = (0..9).map(|i| (i as f32 - 2.0) / 8.0).collect();
    for left in 0..3 {
        let want = reference(&x, &w, 3, 3, 3, left);
        for alias in [Alias::Separate, Alias::X, Alias::Weight] {
            close(
                &conv(&be, &x, &w, 3, 3, 3, left as u32, alias, true, DType::F32),
                &want,
            );
        }
        close(
            &conv(
                &be,
                &x,
                &x,
                3,
                3,
                3,
                left as u32,
                Alias::All,
                true,
                DType::F32,
            ),
            &reference(&x, &x, 3, 3, 3, left),
        );
    }
}

#[test]
fn float_storage_types_and_weight_kinds_are_supported() {
    let be = backend();
    let x = [-2.0, 1.0, 0.5, 3.0, -1.0, 2.0];
    let w = [0.5, -1.0, 2.0, -0.5];
    let want = reference(&x, &w, 3, 2, 2, 1);
    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        for dynamic in [false, true] {
            close(
                &conv(&be, &x, &w, 3, 2, 2, 1, Alias::Separate, dynamic, dtype),
                &want,
            );
        }
    }
}

#[test]
fn relu_math_special_values_and_in_place_execution() {
    let be = backend();
    let values = [
        f32::NEG_INFINITY,
        -2.0,
        -0.0,
        0.0,
        0.125,
        f32::INFINITY,
        f32::NAN,
    ];
    for in_place in [false, true] {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![1, values.len()], DType::F32));
        let dst = if in_place {
            x
        } else {
            g.output(g.desc(x).clone())
        };
        g.push(Op::Relu {
            x,
            dst,
            n: values.len() as u32,
        });
        let plan = be.compile(&g).unwrap();
        let xb = buffer(&be, &values);
        let out = buffer(&be, &vec![0.0; values.len()]);
        let mut bindings = Bindings::new();
        bindings.bind(x, xb.as_ref());
        if dst != x {
            bindings.bind(dst, out.as_ref());
        }
        be.execute(plan.as_ref(), &bindings).unwrap();
        let got = download(&be, bindings.get(dst).unwrap(), values.len());
        for v in &got[..4] {
            assert_eq!(v.to_bits(), 0.0f32.to_bits());
        }
        assert_eq!(got[4], 0.125);
        assert_eq!(got[5], f32::INFINITY);
        assert!(got[6].is_nan());
    }
}

#[test]
fn relu_accepts_float_inputs_and_static_weights() {
    let be = backend();
    let values = [-2.0, -0.5, 0.0, 0.5, 2.0, 4.0];
    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        for is_weight in [false, true] {
            let mut g = Graph::new();
            let desc = TensorDesc::new(vec![2, 3], dtype);
            let x = if is_weight {
                g.weight(desc)
            } else {
                g.input(desc)
            };
            let dst = g.output(TensorDesc::new(vec![6], DType::F32));
            g.push(Op::Relu { x, dst, n: 6 });
            let plan = be.compile(&g).unwrap();
            let xb = typed_buffer(&be, &values, dtype);
            let out = buffer(&be, &[0.0; 6]);
            let mut bindings = Bindings::new();
            bindings.bind(x, xb.as_ref()).bind(dst, out.as_ref());
            be.execute(plan.as_ref(), &bindings).unwrap();
            assert_eq!(
                download(&be, out.as_ref(), 6),
                [0.0, 0.0, 0.0, 0.5, 2.0, 4.0]
            );
        }
    }
}

#[test]
fn primitives_compose_with_internal_activations_and_generated_weights() {
    let be = backend();
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![3, 2], DType::F32));
    let w = g.weight(TensorDesc::new(vec![2, 2], DType::F32));
    let positive_w = g.internal(g.desc(w).clone());
    let mid = g.internal(g.desc(x).clone());
    let dst = g.output(g.desc(x).clone());
    g.push(Op::Relu {
        x: w,
        dst: positive_w,
        n: 4,
    });
    g.push(Op::DepthwiseConv1d {
        x,
        weight: positive_w,
        dst: mid,
        rows: 3,
        channels: 2,
        kernel: 2,
        pad_left: 1,
        pad_right: 0,
    });
    g.push(Op::Relu {
        x: mid,
        dst: mid,
        n: 6,
    });
    g.push(Op::Relu { x: mid, dst, n: 6 });
    let plan = be.compile(&g).unwrap();
    let xb = buffer(&be, &[-1.0, 2.0, 3.0, -4.0, 5.0, 6.0]);
    let wb = buffer(&be, &[-2.0, 1.0, 2.0, -1.0]);
    let out = buffer(&be, &[0.0; 6]);
    let mut bindings = Bindings::new();
    bindings
        .bind(x, xb.as_ref())
        .bind(w, wb.as_ref())
        .bind(dst, out.as_ref());
    be.execute(plan.as_ref(), &bindings).unwrap();
    assert_eq!(
        download(&be, out.as_ref(), 6),
        [0.0, 0.0, 3.0, 4.0, 5.0, 0.0]
    );
}

fn valid_graph() -> (Graph, TensorId, TensorId, TensorId) {
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![2, 3], DType::F32));
    let w = g.weight(TensorDesc::new(vec![3, 4], DType::F32));
    let dst = g.output(g.desc(x).clone());
    g.push(Op::DepthwiseConv1d {
        x,
        weight: w,
        dst,
        rows: 2,
        channels: 3,
        kernel: 4,
        pad_left: 1,
        pad_right: 2,
    });
    (g, x, w, dst)
}

#[test]
fn ir_names_io_and_validation_include_both_primitives() {
    let (mut g, x, w, dst) = valid_graph();
    let relu = Op::Relu { x, dst, n: 6 };
    assert_eq!(relu.kind(), "Relu");
    assert_eq!(g.ops[0].kind(), "DepthwiseConv1d");
    assert_eq!(relu.io(), (vec![x], vec![dst]));
    assert_eq!(g.ops[0].io(), (vec![x, w], vec![dst]));
    g.push(relu);
    assert_eq!(g.inputs, [x]);
    assert_eq!(g.weights, [w]);
    assert_eq!(g.outputs, [dst]);
    assert!(g.in_place_inputs().is_empty());
    for op in &g.ops {
        op.validate(&g).unwrap();
        let (r, w) = op.io_inline();
        assert_eq!((r.to_vec(), w.to_vec()), op.io());
    }
    let alias = Op::DepthwiseConv1d {
        x,
        weight: w,
        dst: x,
        rows: 2,
        channels: 3,
        kernel: 4,
        pad_left: 3,
        pad_right: 0,
    };
    assert_eq!(alias.io(), (vec![x, w], vec![x]));
    alias.validate(&g).unwrap();
}

#[test]
fn invalid_dimensions_padding_and_overflow_return_errors_at_compile() {
    let be = backend();
    for (rows, channels, kernel, left, right) in [
        (0, 3, 4, 1, 2),
        (2, 0, 4, 1, 2),
        (2, 3, 0, 0, 0),
        (2, 3, 4, 1, 1),
        (2, 3, 4, 3, 1),
        (2, 3, 4, u32::MAX, 4),
        (u32::MAX, u32::MAX, 4, 1, 2),
    ] {
        let (mut g, x, w, dst) = valid_graph();
        g.tensors[x.0 as usize].desc.shape = vec![rows as usize, channels as usize];
        g.tensors[dst.0 as usize].desc.shape = vec![rows as usize, channels as usize];
        g.ops[0] = Op::DepthwiseConv1d {
            x,
            weight: w,
            dst,
            rows,
            channels,
            kernel,
            pad_left: left,
            pad_right: right,
        };
        assert!(
            be.compile(&g).is_err(),
            "{rows}/{channels}/{kernel}/{left}/{right}"
        );
    }
    for shape in [
        vec![usize::MAX, 2],
        vec![usize::MAX / 4 + 1],
        vec![0],
        vec![5],
        vec![7],
    ] {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(shape, DType::F32));
        let dst = g.output(TensorDesc::new(vec![6], DType::F32));
        g.push(Op::Relu { x, dst, n: 6 });
        assert!(be.compile(&g).is_err());
    }
}

#[test]
fn invalid_rank_shape_dtype_handle_and_readonly_destination_return_errors() {
    let be = backend();
    for which in 0..8 {
        let (mut g, x, w, dst) = valid_graph();
        match which {
            0 => g.tensors[x.0 as usize].desc.shape = vec![6],
            1 => g.tensors[dst.0 as usize].desc.shape = vec![3, 2],
            2 => g.tensors[w.0 as usize].desc.shape = vec![4, 3],
            3 => g.tensors[x.0 as usize].desc.dtype = DType::I32,
            4 => g.tensors[w.0 as usize].desc.dtype = DType::Q8_0,
            5 => g.tensors[dst.0 as usize].desc.dtype = DType::F16,
            6 => g.tensors[dst.0 as usize].kind = TensorKind::Weight,
            7 => {
                if let Op::DepthwiseConv1d { weight, .. } = &mut g.ops[0] {
                    *weight = TensorId(u32::MAX);
                }
            }
            _ => unreachable!(),
        }
        assert!(be.compile(&g).is_err(), "case {which}");
    }
    let (mut g, x, _, dst) = valid_graph();
    g.ops[0] = Op::Relu {
        x,
        dst: TensorId(u32::MAX),
        n: 6,
    };
    assert!(be.compile(&g).is_err());
    g.ops[0] = Op::Relu { x, dst, n: 0 };
    assert!(be.compile(&g).is_err());
}

#[test]
fn missing_wrong_sized_and_distinct_handle_alias_bindings_return_errors() {
    let be = backend();
    let (g, x, w, dst) = valid_graph();
    let plan = be.compile(&g).unwrap();
    let xb = buffer(&be, &[1.0; 6]);
    let wb = buffer(&be, &[1.0; 12]);
    let db = buffer(&be, &[0.0; 6]);
    for missing in [x, w, dst] {
        let mut bindings = Bindings::new();
        for (id, buf) in [(x, xb.as_ref()), (w, wb.as_ref()), (dst, db.as_ref())] {
            if id != missing {
                bindings.bind(id, buf);
            }
        }
        assert!(be.execute(plan.as_ref(), &bindings).is_err());
    }
    for id in [x, w, dst] {
        for bytes in [1, 4, 20, 52] {
            let bad = be.alloc(bytes, BufferUsage::Weights).unwrap();
            let mut bindings = Bindings::new();
            bindings
                .bind(x, xb.as_ref())
                .bind(w, wb.as_ref())
                .bind(dst, db.as_ref())
                .bind(id, bad.as_ref());
            assert!(be.execute(plan.as_ref(), &bindings).is_err());
        }
    }
    let mut bindings = Bindings::new();
    bindings
        .bind(x, xb.as_ref())
        .bind(w, wb.as_ref())
        .bind(dst, xb.as_ref());
    assert!(be.execute(plan.as_ref(), &bindings).is_err());
}
