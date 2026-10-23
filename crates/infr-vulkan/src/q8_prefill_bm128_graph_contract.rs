//! Parent-mounted actual adapter Graph + Recorder parity, not a dispatch simulation.
use super::*;
use infr_core::{DType, TensorDesc};

const QUALIFIED: [(usize, usize, usize); 6] = [
    (3072, 8192, 512),
    (8192, 3072, 1024),
    (3072, 12288, 512),
    (3072, 12288, 1024),
    (2560, 10240, 1024),
    (2560, 12288, 1024),
];
const EXCLUDED_FLASH_M512: [(usize, usize, usize); 3] = [
    (2560, 6144, 512),
    (6144, 2560, 512),
    (2560, 12288, 512),
];
const GUARD: usize = 64;
const POISON: f32 = -7777.25;
const BM128: &str = "native_gemm_warp_q8_0_n128_ag_bm128";

fn exact_bytes(label: &str, expected: &[u8], actual: &[u8]) {
    assert_eq!(expected.len(), actual.len(), "{label}: byte length");
    if expected == actual {
        return;
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .unwrap();
    let mismatches = expected.iter().zip(actual).filter(|(a, b)| a != b).count();
    let hash = |bytes: &[u8]| {
        bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        })
    };
    let word = |bytes: &[u8]| {
        bytes
            .get(first / 4 * 4..first / 4 * 4 + 4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
    };
    panic!("{label}: exact byte mismatch length={} first_byte={first} first_word={} mismatched_bytes={mismatches} expected_byte={:02x} actual_byte={:02x} expected_word={:?} actual_word={:?} expected_hash={:016x} actual_hash={:016x}",
        expected.len(), first / 4, expected[first], actual[first], word(expected), word(actual), hash(expected), hash(actual));
}

fn exact_bits(label: &str, expected: &[u32], actual: &[u32]) {
    exact_bytes(
        label,
        bytemuck::cast_slice(expected),
        bytemuck::cast_slice(actual),
    );
}

#[test]
fn q8_prefill_bm128_exact_diagnostics_cpu() {
    let expected = vec![0u32; 16_384];
    let mut actual = expected.clone();
    exact_bits("equal", &expected, &actual);
    actual[16_383] = 0x80000000;
    let failure = std::panic::catch_unwind(|| exact_bits("signed zero", &expected, &actual))
        .expect_err("different bits must fail even when floats compare equal");
    let message = failure.downcast_ref::<String>().unwrap();
    assert!(
        message.len() < 1024,
        "diagnostic must not dump input vectors"
    );
    for field in [
        "first_word=16383",
        "mismatched_bytes=1",
        "expected_hash=",
        "actual_hash=",
    ] {
        assert!(message.contains(field), "missing diagnostic field {field}");
    }
}

fn backend(enabled: bool) -> VulkanBackend {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.kernels.vulkan.q8_prefill_bm128 = enabled;
    cfg.kernels.vulkan.gemm_warp = true;
    cfg.kernels.vulkan.gemm_direct_a = false;
    cfg.kernels.vulkan.gemm_wide_tile = false;
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(
        be.prefers_q8_prefill_bm128(),
        "O01 graph contract requires measured Windows RX7900XTX"
    );
    be
}

fn raw(k: usize, n: usize) -> Vec<u8> {
    let mut data = vec![0u8; (n + 1) * k / 32 * 34];
    for (i, b) in data.chunks_exact_mut(34).enumerate() {
        b[..2].copy_from_slice(
            &half::f16::from_f32([1.0 / 512.0, -1.0 / 1024.0, 1.0 / 2048.0][i % 3]).to_le_bytes(),
        );
        let mut rng = (i as u32).wrapping_mul(0x85ebca6b) ^ 0x101b128;
        for code in &mut b[2..] {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *code = (rng >> 24) as u8;
        }
    }
    data
}

fn input(m: usize, k: usize, seed: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| ((i.wrapping_mul(73).wrapping_add(seed * 31) % 521) as f32 - 260.0) / 521.0)
        .collect()
}

fn output(be: &VulkanBackend, buffer: &dyn Buffer, elems: usize) -> Vec<u32> {
    let mut bits = vec![0u32; elems + GUARD];
    be.download(buffer, bytemuck::cast_slice_mut(&mut bits))
        .unwrap();
    assert!(bits[..elems]
        .iter()
        .all(|&b| f32::from_bits(b).is_finite() && b != POISON.to_bits()));
    assert!(
        bits[elems..].iter().all(|&b| b == POISON.to_bits()),
        "output suffix overwritten"
    );
    bits
}

fn buffer(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let b = be.alloc(bytes.len(), usage).unwrap();
    be.upload(b.as_ref(), bytes).unwrap();
    b
}

struct Case {
    graph: Graph,
    plan: Box<dyn Plan>,
    x_id: TensorId,
    w_id: TensorId,
    y_id: TensorId,
    x: Box<dyn Buffer>,
    w: Box<dyn Buffer>,
    y: Box<dyn Buffer>,
    m: usize,
    k: usize,
    n: usize,
}

impl Case {
    fn new(be: &VulkanBackend, k: usize, n: usize, m: usize, internal: bool, bytes: &[u8]) -> Self {
        let mut graph = Graph::new();
        let x_id = graph.input(TensorDesc::new(vec![m, k], DType::F32));
        let w_id = graph.weight(TensorDesc::new(vec![n + 1, k], DType::Q8_0));
        let y_id = graph.output(TensorDesc::new(vec![m, n], DType::F32));
        let dst = if internal {
            graph.internal(TensorDesc::new(vec![m, n], DType::F32))
        } else {
            y_id
        };
        graph.push(Op::Linear {
            x: x_id,
            weight: w_id,
            dst,
            m: m as u32,
            in_f: k as u32,
            out_f: n as u32,
            w_off: k as u32,
        });
        if internal {
            graph.push(Op::Copy {
                src: dst,
                src_off: 0,
                dst: y_id,
                dst_off: 0,
                n: (m * n) as u32,
            });
        }
        let plan = be.compile(&graph).unwrap();
        let x = buffer(
            be,
            bytemuck::cast_slice(&vec![POISON; m * k + GUARD]),
            BufferUsage::Activations,
        );
        let w = buffer(be, bytes, BufferUsage::Weights);
        let y = buffer(
            be,
            bytemuck::cast_slice(&vec![POISON; m * n + GUARD]),
            BufferUsage::Readback,
        );
        Self {
            graph,
            plan,
            x_id,
            w_id,
            y_id,
            x,
            w,
            y,
            m,
            k,
            n,
        }
    }

    fn execute(&self, be: &VulkanBackend, values: &[f32]) -> Vec<u32> {
        let mut padded = values.to_vec();
        padded.extend(std::iter::repeat_n(POISON, GUARD));
        be.upload(self.x.as_ref(), bytemuck::cast_slice(&padded))
            .unwrap();
        be.upload(
            self.y.as_ref(),
            bytemuck::cast_slice(&vec![POISON; self.m * self.n + GUARD]),
        )
        .unwrap();
        let mut bindings = Bindings::new();
        bindings.bind(self.x_id, self.x.as_ref());
        bindings.bind(self.w_id, self.w.as_ref());
        bindings.bind(self.y_id, self.y.as_ref());
        be.execute(self.plan.as_ref(), &bindings).unwrap();
        let mut actual = vec![0u32; padded.len()];
        be.download(self.x.as_ref(), bytemuck::cast_slice_mut(&mut actual))
            .unwrap();
        exact_bits(
            "input/guard changed",
            &padded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            &actual,
        );
        output(be, self.y.as_ref(), self.m * self.n)
    }

    fn recorder(&self, be: &VulkanBackend) -> Vec<u32> {
        let mpad = self.m.next_multiple_of(64);
        let splits = split_k_plan(self.m, self.k, self.n);
        let a16 = buffer(be, &vec![0; mpad * self.k * 2], BufferUsage::Activations);
        let dst = buffer(
            be,
            bytemuck::cast_slice(&vec![POISON; self.m * self.n + GUARD]),
            BufferUsage::Activations,
        );
        // Match native_warp_gemm's original planner, partial-plane stride and reduce order.
        let partials = (splits > 1).then(|| {
            buffer(
                be,
                &vec![0; splits * mpad * self.n * 4],
                BufferUsage::Activations,
            )
        });
        let rec = be.recorder().unwrap();
        rec.store_f16(self.x.as_ref(), a16.as_ref(), self.m * self.k, 0);
        if let Some(partials) = &partials {
            assert!(crate::gemm::native_gemm_warp_sk_kernel_name(DType::Q8_0).is_some());
            rec.matmul_native_splitk(
                DType::Q8_0,
                a16.as_ref(),
                self.w.device_addr().unwrap(),
                self.k,
                partials.as_ref(),
                dst.as_ref(),
                self.m,
                self.k,
                self.n,
                splits,
                true,
            );
        } else {
            rec.matmul_native_f16a(
                DType::Q8_0,
                a16.as_ref(),
                self.w.device_addr().unwrap(),
                self.k,
                dst.as_ref(),
                self.m,
                self.k,
                self.n,
            );
        }
        rec.finish().unwrap();
        output(be, dst.as_ref(), self.m * self.n)
    }

    fn unchanged_weights(&self, be: &VulkanBackend, expected: &[u8]) {
        let mut actual = vec![0u8; expected.len()];
        be.download(self.w.as_ref(), &mut actual).unwrap();
        exact_bytes("weights changed", expected, &actual);
    }
}

#[test]
fn q8_prefill_bm128_six_gate_and_graph_contract_cpu() {
    assert_eq!(QUALIFIED.len(), 6);
    assert_eq!(
        QUALIFIED
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        6
    );
    let split_neighbors: Vec<_> = QUALIFIED
        .iter()
        .chain(EXCLUDED_FLASH_M512.iter())
        .filter_map(|&(k, n, m)| {
            let splits = split_k_plan(m - 128, k, n);
            (splits > 1).then_some((k, n, m - 128, splits))
        })
        .collect();
    assert_eq!(
        split_neighbors,
        [(6144, 2560, 384, 2)],
        "neighbor fixture must preserve the original split-K route"
    );
    for (k, n, m) in QUALIFIED {
        assert_eq!(
            n % 128,
            0,
            "only the existing BN128 registration is qualified"
        );
        assert_eq!(
            split_k_plan(m, k, n),
            1,
            "all six qualified tuples are unsplit; do not register a split BM128 arm"
        );
        assert_eq!(m.next_multiple_of(64), m.next_multiple_of(128));
        for enabled in [false, true] {
            for measured in [false, true] {
                for default_ag in [false, true] {
                    for dtype in [DType::Q8_0, DType::Q6K, DType::Q5K, DType::Q3K, DType::F16] {
                        for rows in [m - 1, m, m + 1] {
                            assert_eq!(
                                crate::recorder::q8_prefill_bm128_applies(
                                    enabled, measured, default_ag, dtype, rows, k, n
                                ),
                                enabled
                                    && measured
                                    && default_ag
                                    && dtype == DType::Q8_0
                                    && rows == m
                            );
                        }
                    }
                }
            }
        }
        for internal in [false, true] {
            let mut graph = Graph::new();
            let x = graph.input(TensorDesc::new(vec![m, k], DType::F32));
            let w = graph.weight(TensorDesc::new(vec![n + 1, k], DType::Q8_0));
            let y = graph.output(TensorDesc::new(vec![m, n], DType::F32));
            let dst = if internal {
                graph.internal(TensorDesc::new(vec![m, n], DType::F32))
            } else {
                y
            };
            graph.push(Op::Linear {
                x,
                weight: w,
                dst,
                m: m as u32,
                in_f: k as u32,
                out_f: n as u32,
                w_off: k as u32,
            });
            assert!(
                matches!(graph.ops[0],Op::Linear{m:rows,in_f,out_f,w_off,..} if rows==m as u32&&in_f==k as u32&&out_f==n as u32&&w_off==k as u32)
            );
        }
    }
    for (k, n) in [
        (2048, 512),
        (512, 2048),
        (3072, 1024),
        (1024, 3072),
        (2560, 640),
        (640, 2560),
        (3072, 512),
        (2560, 512),
        (3072, 8192),
        (8192, 3072),
        (3072, 12288),
        (2560, 10240),
        (2560, 12288),
        (2560, 2560),
        (2560, 6144),
        (320, 10240),
        (6144, 2560),
        (10240, 320),
        (2560, 248320),
    ] {
        for m in [
            1, 384, 511, 512, 513, 896, 1023, 1024, 1025, 1920, 2048, 2049, 3968, 4096, 4097,
        ] {
            let expected = QUALIFIED.contains(&(k, n, m));
            assert_eq!(
                crate::recorder::q8_prefill_bm128_applies(true, true, true, DType::Q8_0, m, k, n),
                expected
            );
            if split_k_plan(m, k, n) > 1 {
                assert!(
                    !expected,
                    "unproven split-K cases must retain the original route"
                );
            }
        }
    }
    for (k, n, m) in EXCLUDED_FLASH_M512 {
        assert!(!QUALIFIED.contains(&(k, n, m)));
        assert!(
            !crate::recorder::q8_prefill_bm128_applies(true, true, true, DType::Q8_0, m, k, n),
            "Flash M512 tuples excluded after isolated warm BAAB must stay baseline"
        );
    }
    for (k, n, m) in [
        (512, 2048, 2048),
        (512, 2048, 4096),
        (1024, 3072, 4096),
        (640, 2560, 1024),
        (640, 2560, 4096),
        (2560, 640, 4096),
    ] {
        assert!(
            !crate::recorder::q8_prefill_bm128_applies(true, true, true, DType::Q8_0, m, k, n),
            "old six failed independent BAAB and must stay baseline"
        );
    }
}

fn actual(k: usize, n: usize, m: usize, expect_candidate: bool) {
    let splits = split_k_plan(m, k, n);
    assert!(
        !expect_candidate || splits == 1,
        "BM128 never overrides split-K"
    );
    let off = backend(false);
    let on = backend(true);
    let bytes = raw(k, n);
    for internal in [false, true] {
        let a = Case::new(&off, k, n, m, internal, &bytes);
        let b = Case::new(&on, k, n, m, internal, &bytes);
        assert_eq!(a.graph.ops.len(), b.graph.ops.len());
        for seed in 0..3 {
            let values = input(m, k, seed);
            let baseline = a.execute(&off, &values);
            let candidate = b.execute(&on, &values);
            let context = format!("K{k} N{n} M{m} splits={splits} internal={internal} seed={seed}");
            exact_bits(
                &format!("Graph config off/on {context}"),
                &baseline,
                &candidate,
            );
            exact_bits(
                &format!("baseline Graph/Recorder {context}"),
                &baseline,
                &a.recorder(&off),
            );
            exact_bits(
                &format!("candidate Graph/Recorder {context}"),
                &candidate,
                &b.recorder(&on),
            );
        }
        a.unchanged_weights(&off, &bytes);
        b.unchanged_weights(&on, &bytes);
    }
    assert!(
        !off.built_kernel_names().contains(&BM128),
        "config-off secretly selected candidate"
    );
    assert_eq!(
        on.built_kernel_names().contains(&BM128),
        expect_candidate,
        "Graph/Recorder did not reach expected production variant"
    );
    let base = if splits > 1 {
        crate::gemm::native_gemm_warp_sk_ag_kernel_name(DType::Q8_0).unwrap()
    } else {
        "native_gemm_warp_q8_0_n128_ag"
    };
    assert!(
        off.built_kernel_names().contains(&base),
        "baseline did not reach its original planned route"
    );
    if splits > 1 {
        for be in [&off, &on] {
            let names = be.built_kernel_names();
            assert!(names.contains(&base) && names.contains(&"splitk_reduce"));
            assert!(!names.contains(&"native_gemm_warp_q8_0_n128_ag"));
        }
    } else if !expect_candidate {
        assert!(on.built_kernel_names().contains(&base));
    }
    eprintln!("O01_GRAPH K={k} N={n} M={m} splits={splits} actual_graph_and_recorder=true candidate_reached={expect_candidate} internal_and_output=true three_inputs=true w_base={k} exact=true guards=true weights_unchanged=true synthetic_weights=true");
}

#[test]
#[ignore = "parent-only six qualified BN128 tuples: actual Graph/Recorder exactness"]
fn q8_prefill_bm128_graph_and_recorder_exact() {
    assert_eq!(QUALIFIED.len(), 6);
    for (k, n, m) in QUALIFIED {
        actual(k, n, m, true);
    }
    eprintln!("O01_GRAPH_SUMMARY qualified_cases=6 BN=128 exact=true");
}

#[test]
#[ignore = "parent-only nine qualified/excluded neighboring M cases stay on original Recorder route"]
fn q8_prefill_bm128_graph_neighbor_fallback() {
    assert_eq!(QUALIFIED.len() + EXCLUDED_FLASH_M512.len(), 9);
    for &(k, n, m) in QUALIFIED.iter().chain(EXCLUDED_FLASH_M512.iter()) {
        assert!(!QUALIFIED.contains(&(k, n, m - 128)));
        assert!(!crate::recorder::q8_prefill_bm128_applies(
            true,
            true,
            true,
            DType::Q8_0,
            m - 128,
            k,
            n
        ));
        actual(k, n, m - 128, false);
    }
    eprintln!("O01_GRAPH_SUMMARY neighboring_fallback_cases=9 candidate_reached=false exact=true");
}

#[test]
#[ignore = "parent-only three Flash M512 exclusions: config-on must retain original Graph/Recorder route"]
fn q8_prefill_bm128_graph_excluded_flash_m512_fallback() {
    assert_eq!(EXCLUDED_FLASH_M512.len(), 3);
    for (k, n, m) in EXCLUDED_FLASH_M512 {
        assert!(!crate::recorder::q8_prefill_bm128_applies(
            true, true, true, DType::Q8_0, m, k, n
        ));
        actual(k, n, m, false);
    }
    eprintln!("O01_GRAPH_SUMMARY excluded_flash_m512_cases=3 candidate_reached=false exact=true");
}
