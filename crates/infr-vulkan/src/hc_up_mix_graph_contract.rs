//! Adapter child only; parent supplies the four-line cfg(test)/path/module mount.
//! Ignored tests execute real graphs, not raw recorder substitutes. No timing claims.
use super::*;
use infr_core::tensor::{DType, TensorDesc};

const K: usize = 320;
const WIDE: usize = 10240;
const N: usize = 2560;
const GUARD: usize = 32;
const CANARY: u32 = 0x7fc6_cafe;
const FUSED: &str = "qwen_hc_up_mix_e14_wave32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Case {
    Eligible,
    LowDstAlias,
    XDstAlias,
    WeightDstAlias,
    Multirow,
    F32Weight,
    Wave32Off,
    IndependentRows,
    MtpVerify,
    InterveningReader,
    LaterReader,
    GatePrefixWrite,
    GateSuffixWrite,
    GateStridedPartial,
    GateFullCopy,
    GateFullStrided,
    GatePartialThenFull,
    ExternalGate,
    WeightInput,
    NoOutputAddress,
}

impl Case {
    fn fuses(self) -> bool {
        matches!(
            self,
            Self::Eligible | Self::GateFullCopy | Self::GateFullStrided | Self::GatePartialThenFull
        )
    }

    fn rewrites_gate(self) -> bool {
        matches!(
            self,
            Self::GatePrefixWrite
                | Self::GateSuffixWrite
                | Self::GateStridedPartial
                | Self::GateFullCopy
                | Self::GateFullStrided
                | Self::GatePartialThenFull
        )
    }
}

struct ContractGraph {
    graph: Graph,
    low: TensorId,
    x: TensorId,
    weight: TensorId,
    gate: TensorId,
    dst: TensorId,
    side: Option<TensorId>,
    rows: usize,
}

fn contract_graph(case: Case) -> ContractGraph {
    let rows = if case == Case::Multirow { 2 } else { 1 };
    let mut graph = Graph::new();
    let f32d = |n| TensorDesc::new(vec![n], DType::F32);
    let low = graph.input(f32d(rows * K));
    let x = graph.input(f32d(rows * WIDE));
    let dtype = if case == Case::F32Weight {
        DType::F32
    } else {
        DType::Q8_0
    };
    let desc = TensorDesc::new(vec![WIDE, K], dtype);
    let weight = if case == Case::WeightInput {
        graph.input(desc)
    } else {
        graph.weight(desc)
    };
    let gate = if case == Case::ExternalGate {
        graph.output(f32d(rows * WIDE))
    } else {
        graph.internal(f32d(rows * WIDE))
    };
    let dst = graph.output(f32d(rows * N));
    let side = (matches!(case, Case::InterveningReader | Case::LaterReader)
        || case.rewrites_gate())
    .then(|| graph.output(f32d(rows * WIDE)));
    graph.push(Op::Linear {
        x: low,
        weight,
        dst: gate,
        m: rows as u32,
        in_f: K as u32,
        out_f: WIDE as u32,
        w_off: 0,
    });
    let copy = side.map(|dst| Op::Copy {
        src: gate,
        src_off: 0,
        dst,
        dst_off: 0,
        n: (rows * WIDE) as u32,
    });
    if case == Case::InterveningReader {
        graph.push(copy.clone().unwrap());
    }
    graph.push(Op::QwenHcMix {
        x,
        gate,
        dst,
        rows: rows as u32,
        hc: 4,
        n_embd: N as u32,
    });
    if case.rewrites_gate() {
        let write = match case {
            Case::GateStridedPartial | Case::GateFullStrided => Op::CopyStrided {
                src: x,
                src_off: 0,
                src_stride: (WIDE / 2) as u32,
                dst: gate,
                dst_off: 0,
                dst_stride: (WIDE / 2) as u32,
                rows: 2,
                n: if case == Case::GateFullStrided {
                    (WIDE / 2) as u32
                } else {
                    (WIDE / 4) as u32
                },
            },
            _ => Op::Copy {
                src: x,
                src_off: 0,
                dst: gate,
                dst_off: if case == Case::GateSuffixWrite {
                    (WIDE - 1) as u32
                } else {
                    0
                },
                n: if case == Case::GateFullCopy {
                    WIDE as u32
                } else {
                    1
                },
            },
        };
        graph.push(write);
        if case == Case::GatePartialThenFull {
            graph.push(Op::Copy {
                src: x,
                src_off: 0,
                dst: gate,
                dst_off: 0,
                n: WIDE as u32,
            });
        }
        graph.push(copy.unwrap());
    } else if case == Case::LaterReader {
        graph.push(copy.unwrap());
    }
    graph.independent_rows = case == Case::IndependentRows;
    if graph.independent_rows {
        graph.sequence_spans.push(SequenceSpan {
            row_start: 0,
            rows: 1,
            start_pos: 0,
        });
    }
    graph.mtp_verify = case == Case::MtpVerify;
    ContractGraph {
        graph,
        low,
        x,
        weight,
        gate,
        dst,
        side,
        rows,
    }
}

fn signal(n: usize, seed: u32) -> Vec<u8> {
    let values: Vec<f32> = (0..n)
        .map(|i| {
            let bits = (i as u32)
                .wrapping_mul(1664525)
                .wrapping_add(seed.wrapping_mul(1013904223));
            if i % 23 == 0 {
                0.0
            } else {
                ((bits >> 8) as f32 / 16777216.0 - 0.5) * 0.5
            }
        })
        .collect();
    bytemuck::cast_slice(&values).to_vec()
}

fn q8(seed: u32) -> Vec<u8> {
    let scales = [0u16, 0x8000, 0x1400, 0x9400, 0x1800, 0x9800, 0x1c00, 0x9c00];
    let mut data = vec![0u8; WIDE * K / 32 * 34];
    for (b, block) in data.chunks_exact_mut(34).enumerate() {
        block[..2].copy_from_slice(&scales[(b + seed as usize) % scales.len()].to_le_bytes());
        for (j, v) in block[2..].iter_mut().enumerate() {
            *v = (b as u8)
                .wrapping_mul(37)
                .wrapping_add((j as u8).wrapping_mul(73))
                .wrapping_add(seed as u8);
        }
    }
    data
}

struct Data {
    buffer: Box<dyn Buffer>,
    initial: Vec<u8>,
    bytes: usize,
}

impl Data {
    fn new(be: &VulkanBackend, mut initial: Vec<u8>, address: bool) -> Self {
        let bytes = initial.len();
        initial.extend_from_slice(bytemuck::cast_slice(&[CANARY; GUARD]));
        // BDA-backed storage forces the actual physical-range gate even without a unified arena.
        let usage = if address {
            BufferUsage::Weights
        } else {
            BufferUsage::Activations
        };
        let buffer = be.alloc(initial.len(), usage).unwrap();
        assert_eq!(buffer.device_addr().is_some(), address);
        Self {
            buffer,
            initial,
            bytes,
        }
    }

    fn reset(&self, be: &VulkanBackend) {
        be.upload(self.buffer.as_ref(), &self.initial).unwrap();
    }

    fn check_guard(&self, be: &VulkanBackend) {
        let mut got = [0u32; GUARD];
        be.download_range(
            self.buffer.as_ref(),
            self.bytes,
            bytemuck::cast_slice_mut(&mut got),
        )
        .unwrap();
        assert_eq!(got, [CANARY; GUARD], "physical buffer tail changed");
    }
}

fn exact(got: &[u32], expected: &[u32], case: Case, output: usize) {
    assert_eq!(got.len(), expected.len());
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            a, b,
            "{case:?} output={output} element={i} got={a:08x} expected={b:08x}"
        );
        assert!(
            f32::from_bits(a).is_finite(),
            "{case:?} output={output} element={i} is poison/NaN/Inf"
        );
    }
}

fn run_case(be: &mut VulkanBackend, case: Case, seed: u32) -> Vec<u32> {
    let c = contract_graph(case);
    let weight = if case == Case::F32Weight {
        signal(WIDE * K, seed ^ 55)
    } else {
        q8(seed)
    };
    let low_len = if case == Case::LowDstAlias {
        c.rows * N
    } else {
        c.rows * K
    };
    let data = [
        Data::new(be, signal(low_len, seed), true),
        Data::new(be, signal(c.rows * WIDE, seed ^ 23), true),
        Data::new(be, weight, true),
        Data::new(
            be,
            bytemuck::cast_slice(&vec![0x7fc0_1234u32; c.rows * N]).to_vec(),
            case != Case::NoOutputAddress,
        ),
        Data::new(
            be,
            bytemuck::cast_slice(&vec![0x7fc0_5678u32; c.rows * WIDE]).to_vec(),
            true,
        ),
        Data::new(
            be,
            bytemuck::cast_slice(&vec![0x7fc0_9abcu32; c.rows * WIDE]).to_vec(),
            true,
        ),
    ];
    let out_index = match case {
        Case::LowDstAlias => 0,
        Case::XDstAlias => 1,
        Case::WeightDstAlias => 2,
        _ => 3,
    };
    let output = data[out_index].buffer.as_ref();
    let mut bindings = Bindings::new();
    bindings.bind(c.low, data[0].buffer.as_ref());
    bindings.bind(c.x, data[1].buffer.as_ref());
    bindings.bind(c.weight, data[2].buffer.as_ref());
    bindings.bind(c.dst, output);
    if case == Case::ExternalGate {
        bindings.bind(c.gate, data[4].buffer.as_ref());
    }
    if let Some(side) = c.side {
        bindings.bind(side, data[5].buffer.as_ref());
    }
    if out_index < 3 {
        assert_ne!(c.dst, [c.low, c.x, c.weight][out_index]);
        assert_eq!(output.device_addr(), data[out_index].buffer.device_addr());
        assert_eq!(
            crate::as_vk_buf(output).unwrap().buffer,
            crate::as_vk_buf(data[out_index].buffer.as_ref())
                .unwrap()
                .buffer
        );
    }
    let scratch: Vec<Option<Box<dyn Buffer>>> = (0..c.graph.tensors.len()).map(|_| None).collect();
    let mut expected: Option<Vec<Vec<u32>>> = None;
    for enabled in [false, true] {
        let mut cfg = be.cfg().clone();
        cfg.kernels.vulkan.hc_up_mix = enabled;
        cfg.kernels.vulkan.gemv.q8_wave32 = Some(case != Case::Wave32Off);
        be.cfg = Arc::new(cfg);
        assert!(!be.dense_paged(), "test must not enter a dense pager");
        for input in &data {
            input.reset(be);
        }
        let matched = hc_up_mix_pair(be, &c.graph, 0, &scratch, &bindings);
        assert_eq!(
            matched.is_some(),
            enabled && case.fuses(),
            "matcher decision {case:?} enabled={enabled}"
        );
        if let Some(pair) = matched {
            assert_eq!(
                (pair.low, pair.weight, pair.x, pair.dst),
                (c.low, c.weight, c.x, c.dst)
            );
        }
        let was_built = be.shared.kernels.lock().unwrap().contains_key(FUSED);
        let plan = be.compile(&c.graph).unwrap();
        be.execute(plan.as_ref(), &bindings).unwrap();
        if !enabled && !was_built {
            assert!(!be.shared.kernels.lock().unwrap().contains_key(FUSED));
        }
        if enabled && !case.fuses() && !was_built {
            assert!(
                !be.shared.kernels.lock().unwrap().contains_key(FUSED),
                "fallback unexpectedly built HC fusion for {case:?}"
            );
        }
        if enabled && case.fuses() {
            let kernels = be.shared.kernels.lock().unwrap();
            let fused = kernels
                .get(FUSED)
                .expect("actual execute(Graph) never built production HC fusion");
            assert_eq!(
                fused.subgroup_contract,
                crate::ops::SubgroupContract::Fixed(32)
            );
            assert!(
                kernels.contains_key("native_q8_0_wave32_nr4"),
                "split baseline did not use current NR4"
            );
        }
        for input in &data {
            input.check_guard(be);
        }
        let out_bytes = c.rows * N * 4;
        let mut untouched = [0u8; GUARD * 4];
        be.download_range(output, out_bytes, &mut untouched)
            .unwrap();
        assert_eq!(
            &untouched[..],
            &data[out_index].initial[out_bytes..out_bytes + GUARD * 4],
            "write past logical output in {case:?}"
        );
        let mut reads = vec![(output, c.rows * N)];
        if case == Case::ExternalGate {
            reads.push((data[4].buffer.as_ref(), c.rows * WIDE));
        }
        if c.side.is_some() {
            reads.push((data[5].buffer.as_ref(), c.rows * WIDE));
        }
        let got: Vec<Vec<u32>> = reads
            .into_iter()
            .map(|(buffer, n)| {
                let mut words = vec![0u32; n];
                be.download_range(buffer, 0, bytemuck::cast_slice_mut(&mut words))
                    .unwrap();
                words
            })
            .collect();
        if let Some(ref want) = expected {
            for (index, (a, b)) in got.iter().zip(want).enumerate() {
                exact(a, b, case, index);
            }
        } else {
            expected = Some(got);
        }
    }
    eprintln!("HC_GRAPH case={case:?} seed={seed} exact=true canary=true physical_alias={} eligible={} actual_execute_graph=true", out_index < 3, case.fuses());
    expected.unwrap().remove(0)
}

fn backend() -> VulkanBackend {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.kernels.vulkan.hc_up_mix = false;
    cfg.kernels.vulkan.gemv.q8_wave32 = Some(true);
    let be =
        VulkanBackend::new_with(Arc::new(cfg)).expect("HC graph contract requires the parent GPU");
    assert!(
        be.caps().name.contains("7900 XTX"),
        "wrong adapter: {}",
        be.caps().name
    );
    assert!(be.prefers_qwen_hc_down_prefetch4());
    be
}

#[test]
fn hc_up_mix_graph_fixture_shapes() {
    for case in [
        Case::Eligible,
        Case::Multirow,
        Case::F32Weight,
        Case::ExternalGate,
    ] {
        let c = contract_graph(case);
        assert_eq!(c.graph.desc(c.low).numel(), c.rows * K);
        assert_eq!(c.graph.desc(c.x).numel(), c.rows * WIDE);
        assert_eq!(c.graph.desc(c.weight).numel(), WIDE * K);
        assert_eq!(c.graph.desc(c.dst).numel(), c.rows * N);
        assert_ne!(c.low, c.dst);
    }
}

#[test]
fn hc_up_mix_gate_partial_overwrite_observer_contract() {
    for case in [
        Case::Eligible,
        Case::LaterReader,
        Case::ExternalGate,
        Case::GatePrefixWrite,
        Case::GateSuffixWrite,
        Case::GateStridedPartial,
        Case::GateFullCopy,
        Case::GateFullStrided,
        Case::GatePartialThenFull,
    ] {
        let c = contract_graph(case);
        assert_eq!(
            hc_up_mix_gate_unobserved(&c.graph, 2, c.gate),
            case.fuses(),
            "observer decision {case:?}"
        );
    }
    let mut c = contract_graph(Case::GatePrefixWrite);
    c.graph.ops.pop();
    assert!(hc_up_mix_gate_unobserved(&c.graph, 2, c.gate));
    c.graph.outputs.push(c.gate);
    assert!(!hc_up_mix_gate_unobserved(&c.graph, 2, c.gate));
}

#[test]
fn hc_up_mix_gate_full_redefinition_and_read_order_contract() {
    let copy_to_gate = |src, gate, dst_off, n| Op::Copy {
        src,
        src_off: 0,
        dst: gate,
        dst_off,
        n,
    };
    for (suffix, expected) in [
        (0, true),  // Full Linear, then observer of the new value.
        (1, false), // Partial Linear does not kill the old gate.
        (2, false), // Self-copy reads the old value before rewriting it.
        (3, false), // An observer before a full write still needs the old value.
        (4, false), // Unknown full-looking writers remain conservative.
        (5, false), // Nonzero destination offset is not full coverage.
        (6, false), // An oversized copy is not a proved exact-size definition.
    ] {
        let mut c = contract_graph(Case::LaterReader);
        let observer = c.graph.ops.pop().unwrap();
        match suffix {
            0 | 1 => c.graph.push(Op::Linear {
                x: c.low,
                weight: c.weight,
                dst: c.gate,
                m: 1,
                in_f: K as u32,
                out_f: if suffix == 0 { WIDE } else { WIDE / 2 } as u32,
                w_off: 0,
            }),
            2 => c.graph.push(copy_to_gate(c.gate, c.gate, 0, WIDE as u32)),
            3 => {
                c.graph.push(observer.clone());
                c.graph.push(copy_to_gate(c.x, c.gate, 0, WIDE as u32));
            }
            4 => c.graph.push(Op::Add {
                a: c.x,
                b: c.x,
                dst: c.gate,
                n: WIDE as u32,
            }),
            5 => c
                .graph
                .push(copy_to_gate(c.x, c.gate, 1, (WIDE - 1) as u32)),
            _ => c
                .graph
                .push(copy_to_gate(c.x, c.gate, 0, (WIDE + 1) as u32)),
        }
        c.graph.push(observer);
        assert_eq!(
            hc_up_mix_gate_unobserved(&c.graph, 2, c.gate),
            expected,
            "suffix={suffix}"
        );
    }
    let mut c = contract_graph(Case::GateFullCopy);
    c.graph.tensors[c.gate.0 as usize].desc.shape = vec![usize::MAX, 2];
    assert!(!hc_up_mix_gate_unobserved(&c.graph, 2, c.gate));
}

#[test]
#[ignore = "parent unit4 serialized GPU: actual HC split/fused Graph and physical aliases"]
fn hc_up_mix_graph_exact_and_physical_alias_contract() {
    let mut be = backend();
    for seed in [0, 19] {
        let reference = run_case(&mut be, Case::Eligible, seed);
        for case in [Case::LowDstAlias, Case::XDstAlias, Case::WeightDstAlias] {
            let got = run_case(&mut be, case, seed);
            exact(&got, &reference, case, 0);
        }
    }
}

#[test]
#[ignore = "parent unit4 serialized GPU: actual HC Graph fallback gates and side readers"]
fn hc_up_mix_graph_fallback_contract() {
    let mut be = backend();
    for case in [
        Case::Multirow,
        Case::F32Weight,
        Case::Wave32Off,
        Case::IndependentRows,
        Case::MtpVerify,
        Case::InterveningReader,
        Case::LaterReader,
        Case::ExternalGate,
        Case::WeightInput,
        Case::NoOutputAddress,
    ] {
        run_case(&mut be, case, 37);
    }
}

#[test]
#[ignore = "parent-only GPU: O06 partial/full gate rewrite with exact side observers"]
fn hc_up_mix_graph_gate_rewrite_contract() {
    for case in [
        Case::GatePrefixWrite,
        Case::GateSuffixWrite,
        Case::GateStridedPartial,
        Case::GateFullCopy,
        Case::GateFullStrided,
        Case::GatePartialThenFull,
    ] {
        let mut be = backend();
        for seed in [19, 37] {
            run_case(&mut be, case, seed);
        }
    }
}
