//! Parent-mounted adapter child: production Q8 LinearPair graph and binding contracts.
use super::*;
use infr_core::{DType, TensorDesc, WeightSource};

const K: usize = 2560;
const N: usize = 48;
const GUARD: usize = 64;
const POISON: f32 = -777.25;

fn backend(enabled: bool) -> VulkanBackend {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.kernels.vulkan.q8_linear_pair = enabled;
    // Keep the in-process pipeline cache, but never write a default HOME/XDG cache in this test.
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let properties = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    };
    let name =
        unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
    assert!(
        name.contains("7900 XTX"),
        "Q8 graph campaign requires XTX, got {name}"
    );
    assert_eq!(
        be.caps().linear_pair_q8,
        enabled,
        "expected measured Windows RDNA3 policy"
    );
    be
}

fn input(rows: usize, seed: usize) -> Vec<f32> {
    (0..rows * K)
        .map(|i| match seed % 4 {
            0 => ((i.wrapping_mul(73).wrapping_add(seed * 31) % 521) as f32 - 260.0) / 521.0,
            1 => [0.125, -0.25, 1.0, -1.0, 0.00390625][(i * 7 + seed) % 5],
            2 => {
                if i % 2 == 0 {
                    0.0
                } else {
                    -0.0
                }
            }
            _ => [0.0, -0.0, 1e-20, -1e-20, 32.0, -32.0][(i + seed) % 6],
        })
        .collect()
}

fn synthetic(seed: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; K * N / 32 * 34];
    for (block, row) in bytes.chunks_exact_mut(34).enumerate() {
        let scale = [0x0000u16, 0x8000, 0x2800, 0xa800, 0x3400, 0xb400][(block + seed) % 6];
        row[..2].copy_from_slice(&scale.to_le_bytes());
        for j in 0..32 {
            row[j + 2] = (block.wrapping_mul(73) + j * 137 + seed * 29) as u8;
        }
    }
    bytes
}

fn real(layer: usize) -> [Vec<u8>; 2] {
    let path = std::env::var_os("D01_Q8PAIR_MODEL")
        .filter(|path| !path.is_empty())
        .expect("D01_Q8PAIR_MODEL must name actual AD GGUF; no synthetic real-test fallback");
    let path = std::path::PathBuf::from(path);
    assert!(
        path.is_file(),
        "D01_Q8PAIR_MODEL must name an existing GGUF file: {}",
        path.display()
    );
    let gguf = infr_gguf::Gguf::open(&path).unwrap();
    ["ssm_beta.weight", "ssm_alpha.weight"].map(|suffix| {
        let name = format!("blk.{layer}.{suffix}");
        let tensor = gguf.tensors().iter().find(|t| t.name == name).unwrap();
        assert_eq!(tensor.dtype, DType::Q8_0);
        assert_eq!(tensor.shape, [K, N]);
        let raw = gguf.tensor_bytes(&name).unwrap().to_vec();
        assert_eq!(raw.len(), K * N / 32 * 34);
        eprintln!(
            "D01 graph real layer={layer} tensor={name} bytes={}",
            raw.len()
        );
        raw
    })
}

struct Weights {
    buffers: [Box<dyn Buffer>; 2],
    original: [Vec<u8>; 2],
}

impl Weights {
    fn new(be: &VulkanBackend, raw: [Vec<u8>; 2]) -> Self {
        assert_ne!(raw[0], raw[1]);
        let buffers = raw.each_ref().map(|bytes| {
            let buffer = be.alloc(bytes.len(), BufferUsage::Weights).unwrap();
            be.upload(buffer.as_ref(), bytes).unwrap();
            buffer
        });
        Self {
            buffers,
            original: raw,
        }
    }

    fn unchanged(&self, be: &VulkanBackend) {
        for i in 0..2 {
            let mut actual = vec![0u8; self.original[i].len()];
            be.download(self.buffers[i].as_ref(), &mut actual).unwrap();
            assert_eq!(actual, self.original[i], "weight{i} was modified");
        }
    }
}

struct Shape {
    graph: Graph,
    x: TensorId,
    weights: [TensorId; 2],
    outputs: [TensorId; 2],
    internal: Option<[TensorId; 3]>,
    paired: bool,
}

// This is fixture construction, not a substitute for the runner's CPU policy regression test.
fn graph(rows: usize, enabled: bool, force_split: bool, scratch: bool) -> Shape {
    let mut g = Graph::new();
    let x = g.input(TensorDesc::new(vec![rows, K], DType::F32));
    let weights = std::array::from_fn(|_| g.weight(TensorDesc::new(vec![N, K], DType::Q8_0)));
    let outputs = std::array::from_fn(|_| g.output(TensorDesc::new(vec![rows, N], DType::F32)));
    let internal = scratch.then(|| {
        [
            g.internal(TensorDesc::new(vec![rows, K], DType::F32)),
            g.internal(TensorDesc::new(vec![rows, N], DType::F32)),
            g.internal(TensorDesc::new(vec![rows, N], DType::F32)),
        ]
    });
    let (activation, destinations) = if let Some([activation, a, b]) = internal {
        g.push(Op::Copy {
            src: x,
            src_off: 0,
            dst: activation,
            dst_off: 0,
            n: (rows * K) as u32,
        });
        (activation, [a, b])
    } else {
        (x, outputs)
    };
    let paired = enabled && rows == 1 && !force_split;
    if paired {
        g.push(Op::LinearPair {
            x: activation,
            weight_a: weights[0],
            weight_b: weights[1],
            dst_a: destinations[0],
            dst_b: destinations[1],
            m: rows as u32,
            in_f: K as u32,
            out_f: N as u32,
        });
    } else {
        for i in 0..2 {
            g.push(Op::Linear {
                x: activation,
                weight: weights[i],
                dst: destinations[i],
                m: rows as u32,
                in_f: K as u32,
                out_f: N as u32,
                w_off: 0,
            });
        }
    }
    if scratch {
        for i in 0..2 {
            g.push(Op::Copy {
                src: destinations[i],
                src_off: 0,
                dst: outputs[i],
                dst_off: 0,
                n: (rows * N) as u32,
            });
        }
    }
    Shape {
        graph: g,
        x,
        weights,
        outputs,
        internal,
        paired,
    }
}

struct Case {
    shape: Shape,
    plan: Box<dyn Plan>,
    input: Box<dyn Buffer>,
    outputs: [Box<dyn Buffer>; 2],
    rows: usize,
}

fn assert_distinct(buffers: [&dyn Buffer; 3]) {
    for a in 0..3 {
        for b in a + 1..3 {
            let a = buffers[a]
                .as_any()
                .downcast_ref::<crate::VkBuffer>()
                .unwrap();
            let b = buffers[b]
                .as_any()
                .downcast_ref::<crate::VkBuffer>()
                .unwrap();
            if a.buffer == b.buffer {
                assert!(
                    a.sub_offset + a.size <= b.sub_offset || b.sub_offset + b.size <= a.sub_offset,
                    "paired activation and outputs overlap their physical buffer ranges"
                );
            }
        }
    }
}

impl Case {
    fn new(be: &VulkanBackend, rows: usize, force_split: bool, scratch: bool) -> Self {
        let shape = graph(rows, be.caps().linear_pair_q8, force_split, scratch);
        if let Some(ids) = shape.internal {
            // Exercise the actual adapter layout/allocator, not a handcrafted pretend mapping.
            let allocations = alloc_scratch(be, &shape.graph).unwrap();
            assert_distinct(ids.map(|id| allocations[id.0 as usize].as_ref().unwrap().as_ref()));
        }
        let plan = be.compile(&shape.graph).unwrap();
        let input = be
            .alloc((rows * K + GUARD) * 4, BufferUsage::Activations)
            .unwrap();
        let outputs = std::array::from_fn(|_| {
            be.alloc((rows * N + GUARD) * 4, BufferUsage::Readback)
                .unwrap()
        });
        assert_distinct([input.as_ref(), outputs[0].as_ref(), outputs[1].as_ref()]);
        Self {
            shape,
            plan,
            input,
            outputs,
            rows,
        }
    }

    fn snapshot(&self, be: &VulkanBackend) -> [Vec<u32>; 2] {
        self.outputs.each_ref().map(|buffer| {
            let mut bits = vec![0u32; self.rows * N + GUARD];
            be.download(buffer.as_ref(), bytemuck::cast_slice_mut(&mut bits))
                .unwrap();
            bits
        })
    }

    fn execute(&self, be: &VulkanBackend, weights: &Weights, values: &[f32]) -> [Vec<u32>; 2] {
        assert_eq!(values.len(), self.rows * K);
        let mut padded = values.to_vec();
        padded.extend(std::iter::repeat_n(POISON, GUARD));
        be.upload(self.input.as_ref(), bytemuck::cast_slice(&padded))
            .unwrap();
        for out in &self.outputs {
            be.upload(
                out.as_ref(),
                bytemuck::cast_slice(&vec![POISON; self.rows * N + GUARD]),
            )
            .unwrap();
        }
        let mut bindings = Bindings::new();
        bindings.bind(self.shape.x, self.input.as_ref());
        for i in 0..2 {
            bindings.bind(self.shape.weights[i], weights.buffers[i].as_ref());
            bindings.bind(self.shape.outputs[i], self.outputs[i].as_ref());
        }
        be.execute(self.plan.as_ref(), &bindings).unwrap();
        let mut input_bits = vec![0u32; padded.len()];
        be.download(
            self.input.as_ref(),
            bytemuck::cast_slice_mut(&mut input_bits),
        )
        .unwrap();
        assert_eq!(
            input_bits,
            padded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "input changed"
        );
        let result = self.snapshot(be);
        for output in &result {
            assert!(output[..self.rows * N]
                .iter()
                .all(|&bits| f32::from_bits(bits).is_finite()));
            assert!(
                output[self.rows * N..]
                    .iter()
                    .all(|&bits| bits == POISON.to_bits()),
                "output tail overwritten"
            );
        }
        result
    }
}

fn actual_contract(be: &VulkanBackend, raw: [Vec<u8>; 2], label: &str) {
    let weights = Weights::new(be, raw);
    for scratch in [false, true] {
        let pair_a = Case::new(be, 1, false, scratch);
        let pair_b = Case::new(be, 1, false, scratch);
        let split = Case::new(be, 1, true, scratch);
        assert!(pair_a.shape.paired && pair_b.shape.paired && !split.shape.paired);
        assert!(!std::ptr::eq(pair_a.plan.as_ref(), pair_b.plan.as_ref()));
        for round in 0..4 {
            let x = input(1, round);
            let first = pair_a.execute(be, &weights, &x);
            assert_eq!(
                first,
                split.execute(be, &weights, &x),
                "{label} scratch={scratch} round={round}"
            );
            let other_x = input(1, round + 5);
            let other = pair_b.execute(be, &weights, &other_x);
            assert_eq!(other, split.execute(be, &weights, &other_x));
            assert_eq!(
                pair_a.snapshot(be),
                first,
                "independent plan B overwrote plan A output"
            );
            assert_eq!(
                pair_a.execute(be, &weights, &other_x),
                other,
                "repeat execute kept stale input/output"
            );
            assert_eq!(pair_b.snapshot(be), other);
        }
        weights.unchanged(be);
        eprintln!("D01 graph label={label} scratch_last_use={scratch} two_plans=true repeat_rounds=4 both_outputs_bit_exact=true input_and_canaries_unchanged=true");
    }
    assert!(
        be.built_kernel_names()
            .iter()
            .any(|name| name.contains("linear_native_q8_pair")),
        "actual Op::LinearPair did not reach production paired Q8 kernel"
    );
}

#[test]
fn q8_pair_graph_construct_contract() {
    for enabled in [false, true] {
        for rows in [1, 2, 4, 8, 16, 128, 2048] {
            for scratch in [false, true] {
                let shape = graph(rows, enabled, false, scratch);
                let pairs = shape
                    .graph
                    .ops
                    .iter()
                    .filter(|op| matches!(op, Op::LinearPair { .. }))
                    .count();
                let singles = shape
                    .graph
                    .ops
                    .iter()
                    .filter(|op| matches!(op, Op::Linear { .. }))
                    .count();
                assert_eq!(pairs, usize::from(enabled && rows == 1));
                assert_eq!(singles, if pairs == 1 { 0 } else { 2 });
                for op in &shape.graph.ops {
                    if let Op::LinearPair {
                        x,
                        weight_a,
                        weight_b,
                        dst_a,
                        dst_b,
                        ..
                    } = op
                    {
                        assert_eq!(
                            op.io(),
                            (vec![*x, *weight_a, *weight_b], vec![*dst_a, *dst_b])
                        );
                    }
                    if let Op::Linear {
                        m,
                        in_f,
                        out_f,
                        w_off,
                        ..
                    } = op
                    {
                        assert_eq!(
                            (*m, *in_f, *out_f, *w_off),
                            (rows as u32, K as u32, N as u32, 0)
                        );
                    }
                }
                assert!(shape.graph.in_place_inputs().is_empty());
            }
        }
    }
}

#[test]
#[ignore = "parent serial XTX only: real adapter/Graph Q8 pair, guards, scratch and repeated plans"]
fn q8_pair_graph_execute_synthetic_contract() {
    let be = backend(true);
    actual_contract(&be, [synthetic(5), synthetic(29)], "synthetic_signed_Q8");
}

#[test]
#[ignore = "parent serial XTX; D01_Q8PAIR_MODEL actual AD Q8 weights required"]
fn q8_pair_graph_execute_real_contract() {
    // Missing/invalid model input must fail before creating a Vulkan device.
    let weights = [real(0), real(1)];
    let be = backend(true);
    for (layer, raw) in weights.into_iter().enumerate() {
        actual_contract(&be, raw, &format!("AD_layer{layer}"));
    }
}

#[test]
#[ignore = "parent serial XTX: config-off and multi-row use original two Linear nodes"]
fn q8_pair_graph_config_off_and_batch_fallback() {
    for enabled in [false, true] {
        let be = backend(enabled);
        let weights = Weights::new(&be, [synthetic(7), synthetic(41)]);
        for rows in [1, 2, 4, 8, 16, 128] {
            if enabled && rows == 1 {
                continue;
            }
            let normal = Case::new(&be, rows, false, true);
            let split = Case::new(&be, rows, true, true);
            assert!(!normal.shape.paired);
            for seed in [0, 1] {
                let x = input(rows, seed);
                assert_eq!(
                    normal.execute(&be, &weights, &x),
                    split.execute(&be, &weights, &x)
                );
            }
            eprintln!("D01 graph fallback enabled={enabled} rows={rows} original_two_linears=true outputs_exact=true");
        }
        assert!(
            !be.built_kernel_names()
                .iter()
                .any(|name| name.contains("linear_native_q8_pair")),
            "fallback path secretly dispatched paired kernel"
        );
        weights.unchanged(&be);
    }
}

#[test]
#[ignore = "parent serial XTX: direct unsupported Q8 LinearPair must return an error"]
fn q8_pair_graph_direct_unsupported_contract() {
    for enabled in [false, true] {
        let be = backend(enabled);
        let weights = Weights::new(&be, [synthetic(11), synthetic(31)]);
        let rows = if enabled { 2 } else { 1 };
        let mut direct = Case::new(&be, rows, true, false);
        direct.shape.graph.ops.clear();
        direct.shape.graph.push(Op::LinearPair {
            x: direct.shape.x,
            weight_a: direct.shape.weights[0],
            weight_b: direct.shape.weights[1],
            dst_a: direct.shape.outputs[0],
            dst_b: direct.shape.outputs[1],
            m: rows as u32,
            in_f: K as u32,
            out_f: N as u32,
        });
        direct.plan = be.compile(&direct.shape.graph).unwrap();
        let mut bindings = Bindings::new();
        bindings.bind(direct.shape.x, direct.input.as_ref());
        for i in 0..2 {
            bindings.bind(direct.shape.weights[i], weights.buffers[i].as_ref());
            bindings.bind(direct.shape.outputs[i], direct.outputs[i].as_ref());
        }
        let error = be
            .execute(direct.plan.as_ref(), &bindings)
            .expect_err("unsupported direct Q8 pair accepted");
        assert!(
            error.to_string().contains("LinearPair"),
            "unexpected error: {error}"
        );
        assert!(!be
            .built_kernel_names()
            .iter()
            .any(|name| name.contains("linear_native_q8_pair")));
        eprintln!(
            "D01 graph unsupported enabled={enabled} rows={rows} rejected=true error={error}"
        );
    }
}
