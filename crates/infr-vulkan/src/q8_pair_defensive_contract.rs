//! Parent mounts as an adapter child after applying the evidence-only validation patch.
use super::*;
use infr_core::{DType, TensorDesc};

const K: usize = 2560;
const N: usize = 48;
const ELEMS: [usize; 5] = [K * N, K * N, K, N, N];
const BYTES: [usize; 5] = [K * N / 32 * 34, K * N / 32 * 34, K * 4, N * 4, N * 4];
const GUARD: usize = 64;

#[derive(Clone, Copy, Debug)]
enum Bad {
    Dtype(usize, DType),
    Descriptor(usize),
    Overflow(usize),
    Buffer(usize),
    Alias { output: usize, source: usize },
}

fn cases() -> Vec<Bad> {
    let mut cases = Vec::new();
    for index in 2..5 {
        for dtype in [DType::F16, DType::Bf16, DType::I32, DType::U32] {
            cases.push(Bad::Dtype(index, dtype));
        }
    }
    for index in 0..5 {
        cases.extend([
            Bad::Descriptor(index),
            Bad::Overflow(index),
            Bad::Buffer(index),
        ]);
    }
    for output in 3..5 {
        for source in 0..output {
            cases.push(Bad::Alias { output, source });
        }
    }
    cases
}

fn graph(bad: Option<Bad>, split: bool) -> (Graph, [TensorId; 5]) {
    let mut g = Graph::new();
    let ids = [
        g.weight(TensorDesc::new(vec![N, K], DType::Q8_0)),
        g.weight(TensorDesc::new(vec![N, K], DType::Q8_0)),
        g.input(TensorDesc::new(vec![1, K], DType::F32)),
        g.output(TensorDesc::new(vec![1, N], DType::F32)),
        g.output(TensorDesc::new(vec![1, N], DType::F32)),
    ];
    match bad {
        Some(Bad::Dtype(index, dtype)) => g.tensors[ids[index].0 as usize].desc.dtype = dtype,
        Some(Bad::Descriptor(index)) => {
            g.tensors[ids[index].0 as usize].desc.shape = vec![ELEMS[index] - 1]
        }
        Some(Bad::Overflow(index)) => {
            g.tensors[ids[index].0 as usize].desc.shape = vec![usize::MAX, 2]
        }
        _ => (),
    }
    if split {
        for i in 0..2 {
            g.push(Op::Linear {
                x: ids[2],
                weight: ids[i],
                dst: ids[i + 3],
                m: 1,
                in_f: K as u32,
                out_f: N as u32,
                w_off: 0,
            });
        }
    } else {
        g.push(Op::LinearPair {
            x: ids[2],
            weight_a: ids[0],
            weight_b: ids[1],
            dst_a: ids[3],
            dst_b: ids[4],
            m: 1,
            in_f: K as u32,
            out_f: N as u32,
        });
    }
    g.no_decode_replay = true;
    (g, ids)
}

#[derive(Clone, Copy)]
struct Address {
    base: u64,
    bytes: usize,
}

struct WithoutAddress(usize);
impl Buffer for WithoutAddress {
    fn len_bytes(&self) -> usize {
        self.0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl Buffer for Address {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device_addr(&self) -> Option<u64> {
        Some(self.base)
    }
}

#[test]
fn q8_pair_defensive_cpu_descriptors_lengths_and_ranges() {
    let ordinary = std::array::from_fn::<_, 5, _>(|i| Address {
        base: (i as u64 + 1) * 0x10_0000,
        bytes: BYTES[i],
    });
    let (g, ids) = graph(None, false);
    assert!(
        validate_q8_linear_pair(&g, ids, ordinary.each_ref().map(|b| b as &dyn Buffer)).is_ok()
    );
    for bad in cases() {
        let (g, ids) = graph(Some(bad), false);
        let mut buffers = ordinary;
        match bad {
            Bad::Buffer(index) => buffers[index].bytes = BYTES[index] - 1,
            Bad::Alias { output, source } => buffers[output] = buffers[source],
            _ => (),
        }
        assert!(
            validate_q8_linear_pair(&g, ids, buffers.each_ref().map(|b| b as &dyn Buffer)).is_err(),
            "accepted {bad:?}"
        );
    }
    // A same-arena adjacent view is legal; even one byte of overlap is not.
    for a_bytes in [N * 4, K * 4, K * N / 32 * 34] {
        let a = Address {
            base: 0x100000,
            bytes: a_bytes,
        };
        let next = Address {
            base: a.base + a_bytes as u64,
            bytes: N * 4,
        };
        assert!(q8_linear_pair_views_disjoint(&a, a_bytes, &next, N * 4));
        for offset in [0, 1, a_bytes - 1] {
            let overlap = Address {
                base: a.base + offset as u64,
                bytes: N * 4,
            };
            assert!(!q8_linear_pair_views_disjoint(&a, a_bytes, &overlap, N * 4));
        }
    }
    let overflow = Address {
        base: u64::MAX - 3,
        bytes: 8,
    };
    assert!(!q8_linear_pair_views_disjoint(
        &overflow,
        8,
        &ordinary[2],
        4
    ));
    let missing_weight = WithoutAddress(BYTES[0]);
    let (g, ids) = graph(None, false);
    let mut buffers = ordinary.each_ref().map(|buffer| buffer as &dyn Buffer);
    buffers[0] = &missing_weight;
    let error = validate_q8_linear_pair(&g, ids, buffers)
        .unwrap_err()
        .to_string();
    assert!(error.contains("fixed BDA"));
}

fn backend() -> VulkanBackend {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.kernels.vulkan.q8_linear_pair = true;
    // Avoid default HOME/XDG filesystem output; production shader embeds remain unchanged.
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
        "defensive campaign requires XTX, got {name}"
    );
    assert!(be.caps().linear_pair_q8);
    be
}

struct Data {
    buffers: [Box<dyn Buffer>; 5],
    initial: [Vec<u8>; 5],
}
impl Data {
    fn new(be: &VulkanBackend, bad: Option<Bad>) -> Self {
        let mut initial = std::array::from_fn(|i| vec![0u8; BYTES[i] + GUARD]);
        initial[0][..BYTES[0]].copy_from_slice(&infr_testkit::synth_weight(DType::Q8_0, K * N, 7));
        initial[1][..BYTES[1]].copy_from_slice(&infr_testkit::synth_weight(DType::Q8_0, K * N, 41));
        let x: Vec<f32> = (0..K)
            .map(|i| ((i * 73 % 521) as f32 - 260.0) / 521.0)
            .collect();
        initial[2][..BYTES[2]].copy_from_slice(bytemuck::cast_slice(&x));
        for i in 0..5 {
            initial[i][BYTES[i]..].fill(0xa7);
        }
        for i in 3..5 {
            initial[i][..BYTES[i]].fill(0xc3);
        }
        if let Some(Bad::Buffer(index)) = bad {
            initial[index].truncate(BYTES[index] - 4);
        }
        let buffers = std::array::from_fn(|i| {
            let usage = if i < 2 {
                BufferUsage::Weights
            } else {
                BufferUsage::Activations
            };
            let buffer = be.alloc(initial[i].len(), usage).unwrap();
            be.upload(buffer.as_ref(), &initial[i]).unwrap();
            buffer
        });
        Self { buffers, initial }
    }

    fn bindings(&self, ids: [TensorId; 5], bad: Option<Bad>) -> Bindings<'_> {
        let mut map = [0, 1, 2, 3, 4];
        if let Some(Bad::Alias { output, source }) = bad {
            map[output] = source;
        }
        let mut bindings = Bindings::new();
        for i in 0..5 {
            bindings.bind(ids[i], self.buffers[map[i]].as_ref());
        }
        bindings
    }

    fn read(&self, be: &VulkanBackend, index: usize) -> Vec<u8> {
        let mut actual = vec![0u8; self.buffers[index].len_bytes()];
        be.download(self.buffers[index].as_ref(), &mut actual)
            .unwrap();
        actual
    }

    fn unchanged(&self, be: &VulkanBackend) {
        for i in 0..5 {
            assert_eq!(
                self.read(be, i),
                self.initial[i],
                "buffer{i} changed before rejection"
            );
        }
    }
}

#[test]
#[ignore = "parent serial XTX after defensive patch: malformed Q8 pair must fail before dispatch"]
fn q8_pair_defensive_graph_rejects_before_dispatch() {
    let be = backend();
    let mut count = 0;
    for bad in cases() {
        // Descriptor-product overflow is exercised directly by the CPU validator test; do not
        // assume unrelated generic Graph inspection accepts a malicious overflowing descriptor.
        if matches!(bad, Bad::Overflow(_)) {
            continue;
        }
        let (g, ids) = graph(Some(bad), false);
        let data = Data::new(&be, Some(bad));
        let plan = be.compile(&g).unwrap();
        let error = be
            .execute(plan.as_ref(), &data.bindings(ids, Some(bad)))
            .expect_err("malformed Q8 pair reached dispatch");
        let message = error.to_string();
        assert!(
            message.contains("Q8 LinearPair"),
            "unexpected error for {bad:?}: {message}"
        );
        let expected = match bad {
            Bad::Dtype(_, _) => "requires F32",
            Bad::Descriptor(_) | Bad::Buffer(_) => "length",
            Bad::Alias { .. } => "unsafe overlap",
            Bad::Overflow(_) => unreachable!(),
        };
        assert!(
            message.contains(expected),
            "wrong rejection for {bad:?}: {message}"
        );
        data.unchanged(&be);
        assert!(
            !be.built_kernel_names()
                .iter()
                .any(|name| name.contains("linear_native_q8_pair")),
            "paired shader was built before validation finished"
        );
        eprintln!("D01 defensive case={bad:?} rejected_before_pair_dispatch=true buffers_unchanged=true error={message}");
        count += 1;
    }
    assert_eq!(count, 29);
}

#[test]
#[ignore = "parent serial XTX: defensive checks preserve actual Q8 Pair versus original Linear route"]
fn q8_pair_defensive_supported_graph_stays_exact() {
    let be = backend();
    let (pair, pair_ids) = graph(None, false);
    let (split, split_ids) = graph(None, true);
    let pair_plan = be.compile(&pair).unwrap();
    let split_plan = be.compile(&split).unwrap();
    let candidate = Data::new(&be, None);
    let reference = Data::new(&be, None);
    for round in 0..3 {
        let input: Vec<f32> = (0..K)
            .map(|i| ((i * 73 + round * 41) % 521) as f32 / 521.0 - 0.5)
            .collect();
        for data in [&candidate, &reference] {
            be.upload(data.buffers[2].as_ref(), bytemuck::cast_slice(&input))
                .unwrap();
        }
        be.execute(pair_plan.as_ref(), &candidate.bindings(pair_ids, None))
            .unwrap();
        be.execute(split_plan.as_ref(), &reference.bindings(split_ids, None))
            .unwrap();
        for i in 3..5 {
            let actual = candidate.read(&be, i);
            assert_eq!(actual, reference.read(&be, i), "round{round} output{i}");
            for bytes in actual[..BYTES[i]].chunks_exact(4) {
                assert!(f32::from_le_bytes(bytes.try_into().unwrap()).is_finite());
            }
        }
        for data in [&candidate, &reference] {
            for i in 0..2 {
                assert_eq!(data.read(&be, i), data.initial[i]);
            }
            let got_input = data.read(&be, 2);
            assert_eq!(
                &got_input[..BYTES[2]],
                bytemuck::cast_slice::<f32, u8>(&input)
            );
            for i in 2..5 {
                assert_eq!(&data.read(&be, i)[BYTES[i]..], &data.initial[i][BYTES[i]..]);
            }
        }
    }
    assert!(be
        .built_kernel_names()
        .iter()
        .any(|name| name.contains("linear_native_q8_pair")));
    eprintln!("D01 defensive supported_graph=true original_linear_reference=true both_outputs_bit_exact=true rounds=3 input_without_BDA={} guards_unchanged=true",
        candidate.buffers[2].device_addr().is_none());
    // Also exercises disjoint shared BDA-arena views when the configured weight allocator packs them.
    let same_arena = crate::as_vk_buf(candidate.buffers[0].as_ref())
        .unwrap()
        .buffer
        == crate::as_vk_buf(candidate.buffers[1].as_ref())
            .unwrap()
            .buffer;
    eprintln!("D01 defensive supported_weight_views_share_arena={same_arena}");
}
