//! NEW, unmounted adapter child for parent unit5. Actual Graph execution, not raw dispatches.
use super::*;
use infr_core::backend::SegmentedKvSpec;
use infr_core::{DType, TensorDesc};
use std::path::PathBuf;

const HEADS: usize = 2;
const HD: usize = 256;
const WIDTH: usize = HEADS * HD;
const SEG_ROWS: usize = 32768;
const SEG_ELEMS: usize = SEG_ROWS * WIDTH;
const SEG_BYTES: usize = SEG_ELEMS / 32 * 34;
const GUARD: usize = 64;
const CANARY: u8 = 0xa5;
const NORMAL: &str = "qk_norm_rope_q8_segmented_q8";
const MROPE: &str = "qk_norm_rope_q8_segmented_mrope_q8";

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: usize,
    pos: usize,
    rope: usize,
    mrope: bool,
}

impl Case {
    fn key(self) -> String {
        format!(
            "r{}-p{}-rope{}-mrope{}",
            self.rows, self.pos, self.rope, self.mrope
        )
    }

    fn kernel(self) -> &'static str {
        if self.mrope {
            MROPE
        } else {
            NORMAL
        }
    }

    fn positions(self) -> Vec<i32> {
        if self.mrope {
            (0..self.rows)
                .flat_map(|r| {
                    [
                        (self.pos + 17 + r * 3) as i32,
                        (self.pos + 29 + r / 7) as i32,
                        (self.pos + 47 + (r * 11) % 251) as i32,
                        (self.pos + 61 + (r * 17) % 509) as i32,
                    ]
                })
                .collect()
        } else {
            // The production plain reference reads only pos0 and uses pos0+r. Deliberately
            // non-linear suffixes catch accidental candidate reads of positions[r].
            (0..self.rows)
                .map(|r| {
                    if r == 0 {
                        self.pos as i32
                    } else {
                        (self.pos + 5000 + (r * 29) % 123) as i32
                    }
                })
                .collect()
        }
    }
}

struct ContractGraph {
    graph: Graph,
    x: TensorId,
    norm: TensorId,
    positions: TensorId,
    value: TensorId,
    caches: [TensorId; 2],
    indices: TensorId,
    outputs: [TensorId; 2],
    ids: Vec<u32>,
    gathered_rows: usize,
    capacity_rows: usize,
}

fn contract_graph(case: Case) -> ContractGraph {
    let mut graph = Graph::new();
    graph.no_decode_replay = true;
    let x = graph.input(TensorDesc::new(vec![case.rows, WIDTH], DType::F32));
    let norm = graph.weight(TensorDesc::new(vec![HD], DType::F32));
    let positions = graph.input(TensorDesc::new(
        vec![case.rows, if case.mrope { 4 } else { 1 }],
        DType::I32,
    ));
    let value = graph.input(TensorDesc::new(vec![case.rows, WIDTH], DType::F32));
    let k16 = graph.internal(TensorDesc::new(vec![case.rows, WIDTH], DType::F16));
    let end = case.pos + case.rows;
    let capacity_rows = end.div_ceil(SEG_ROWS) * SEG_ROWS;
    let caches = std::array::from_fn(|_| {
        graph.input(TensorDesc::new(vec![capacity_rows, WIDTH], DType::Q8_0))
    });
    let complete = end / 4;
    // Gather every block touched by this write, plus a known-zero chronological prefix. The
    // real incomplete causal tail follows selected complete blocks just as in production QSA.
    let mut ids = vec![0u32];
    ids.extend((case.pos / 4..complete).map(|id| id as u32));
    ids.sort_unstable();
    ids.dedup();
    assert!(ids.iter().all(|&id| (id as usize) < complete));
    let gathered_rows = ids.len() * 4 + end % 4;
    let indices = graph.input(TensorDesc::new(vec![ids.len()], DType::I32));
    let outputs = std::array::from_fn(|_| {
        graph.output(TensorDesc::new(vec![gathered_rows, WIDTH], DType::F16))
    });
    if case.mrope {
        graph.push(Op::QkNormMrope {
            x,
            weight: norm,
            positions4: positions,
            dst: k16,
            rows: case.rows as u32,
            n_head: HEADS as u32,
            head_dim: HD as u32,
            rope_dim: case.rope as u32,
            theta: 10000.0,
            eps: 1e-6,
            sections: [16; 4],
            x_stride: 0,
        });
    } else {
        graph.push(Op::QkNormRope {
            x,
            weight: norm,
            positions,
            dst: k16,
            rows: case.rows as u32,
            n_head: HEADS as u32,
            head_dim: HD as u32,
            rope_dim: case.rope as u32,
            theta: 10000.0,
            eps: 1e-6,
            freq_factors: None,
            x_stride: 0,
        });
    }
    graph.push(Op::WriteKv {
        src: k16,
        cache: caches[0],
        rows: case.rows as u32,
        row_stride: WIDTH as u32,
        pos: case.pos as u32,
    });
    graph.push(Op::WriteKv {
        src: value,
        cache: caches[1],
        rows: case.rows as u32,
        row_stride: WIDTH as u32,
        pos: case.pos as u32,
    });
    graph.push(Op::QsaGather {
        k_cache: caches[0],
        v_cache: caches[1],
        indices,
        k_dst: outputs[0],
        v_dst: outputs[1],
        selected_blocks: ids.len() as u32,
        complete_blocks: complete as u32,
        tail: (end % 4) as u32,
        ratio: 4,
        row_elems: WIDTH as u32,
    });
    ContractGraph {
        graph,
        x,
        norm,
        positions,
        value,
        caches,
        indices,
        outputs,
        ids,
        gathered_rows,
        capacity_rows,
    }
}

fn signal(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let bits = (i as u32)
                .wrapping_mul(1664525)
                .wrapping_add(seed.wrapping_mul(1013904223));
            ((bits >> 8) as f32 / 16777216.0 - 0.5) * scale
        })
        .collect()
}

fn fnv(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

fn read(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u8> {
    let mut data = vec![0; buffer.len_bytes()];
    // Virtual segmented buffers must use the existing range API, never download(wrapper).
    be.download_range(buffer, 0, &mut data).unwrap();
    data
}

fn failure_evidence_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MOE4ALL_TEST_EVIDENCE_DIR")?);
    let work = PathBuf::from(std::env::var_os("MOE4ALL_WORK_ROOT")?);
    if !dir.is_absolute()
        || !work.is_absolute()
        || dir
            .components()
            .chain(work.components())
            .any(|c| c == std::path::Component::ParentDir)
    {
        return None;
    }
    let work = work.canonicalize().ok()?;
    let allowed = work.join("evidence");
    if allowed.canonicalize().ok()? != allowed {
        return None;
    }
    let dir = dir.canonicalize().ok()?;
    (dir.is_dir() && dir != allowed && dir.starts_with(&allowed)).then_some(dir)
}

fn exact(a: &[u8], b: &[u8], key: &str, half_values: bool) {
    assert_eq!(a.len(), b.len(), "{key}: length");
    if let Some(i) = a.iter().zip(b).position(|(a, b)| a != b) {
        let maxabs = if half_values {
            a.chunks_exact(2)
                .zip(b.chunks_exact(2))
                .map(|(a, b)| {
                    (half::f16::from_bits(u16::from_le_bytes(a.try_into().unwrap())).to_f32()
                        - half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                    .abs()
                })
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max)
        } else {
            0.0
        };
        let summary = format!(
            "firstdiff={i} off={} on={} off_fnv={:016x} on_fnv={:016x} maxabs_f16={maxabs:e}\n",
            a[i],
            b[i],
            fnv(a),
            fnv(b)
        );
        println!("{key}: {summary}");
        if std::env::var_os("MOE4ALL_TEST_EVIDENCE_DIR").is_some() {
            if let Some(dir) = failure_evidence_dir() {
                let dump = (|| -> std::io::Result<PathBuf> {
                    let stamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos();
                    let root = dir.join(format!("failure-{key}-{stamp}"));
                    // A fresh child only: never follow a pre-existing failure-directory link.
                    std::fs::create_dir(&root)?;
                    std::fs::write(root.join("off.bin"), a)?;
                    std::fs::write(root.join("on.bin"), b)?;
                    std::fs::write(root.join("firstdiff.txt"), &summary)?;
                    Ok(root)
                })();
                println!("optional raw evidence: {dump:?}");
            } else {
                println!("raw evidence disabled: absolute existing directory below MOE4ALL_WORK_ROOT/evidence required");
            }
        }
        panic!("{key}: {summary}");
    }
}

struct Data {
    buffer: Box<dyn Buffer>,
    initial: Vec<u8>,
}

impl Data {
    fn new(be: &VulkanBackend, mut data: Vec<u8>, usage: BufferUsage) -> Self {
        data.extend(std::iter::repeat_n(CANARY, GUARD));
        let buffer = be.alloc(data.len(), usage).unwrap();
        Self {
            buffer,
            initial: data,
        }
    }

    fn reset(&self, be: &VulkanBackend) {
        be.upload_range(self.buffer.as_ref(), 0, &self.initial)
            .unwrap();
    }
}

fn zero_cache(be: &VulkanBackend, segments: usize) -> Box<dyn Buffer> {
    let buffer = be
        .alloc_segmented_kv(SegmentedKvSpec {
            logical_bytes: segments * SEG_BYTES,
            segment_bytes: SEG_BYTES,
            segment_elements: SEG_ELEMS,
            max_segments: segments,
        })
        .unwrap()
        .expect("actual segmented KV required");
    be.ensure_segmented_kv_batch(&[buffer.as_ref()], segments)
        .unwrap();
    buffer
}

fn reset_cache(be: &VulkanBackend, buffer: &dyn Buffer) {
    // Both code and scale planes start at valid Q8 zero, including the unwritten physical suffix.
    let zeros = vec![0u8; SEG_BYTES];
    for off in (0..buffer.len_bytes()).step_by(SEG_BYTES) {
        be.upload_range(buffer, off, &zeros).unwrap();
    }
}

fn check_write_extent(raw: &[u8], case: Case, label: &str) {
    let mut expected = vec![0u8; raw.len()];
    let first = case.pos * WIDTH;
    let end = (case.pos + case.rows) * WIDTH;
    for segment in 0..raw.len() / SEG_BYTES {
        let lo = first.max(segment * SEG_ELEMS);
        let hi = end.min((segment + 1) * SEG_ELEMS);
        if lo >= hi {
            continue;
        }
        let code = segment * SEG_BYTES + lo % SEG_ELEMS;
        expected[code..code + hi - lo].copy_from_slice(&raw[code..code + hi - lo]);
        let scale = segment * SEG_BYTES + SEG_ELEMS + (lo % SEG_ELEMS / 32) * 2;
        let count = (hi - lo) / 32 * 2;
        expected[scale..scale + count].copy_from_slice(&raw[scale..scale + count]);
    }
    exact(
        raw,
        &expected,
        &format!("{}-{label}-unwritten", case.key()),
        false,
    );
    assert!(
        raw.iter().any(|&b| b != 0),
        "{label}: vacuous all-zero cache write"
    );
}

fn gather_pack_half_bits(value: f32) -> u16 {
    // The qualified Windows/Navi31 gather's PackHalf2x16 rounds toward zero, unlike
    // half::f16::from_f32's nearest-even conversion. Keep this oracle bit-exact, not tolerant.
    assert!(value.is_finite(), "nonfinite physical Q8 gather value");
    let nearest = half::f16::from_f32(value);
    if nearest.to_f32().abs() > value.abs() {
        nearest.to_bits() - 1
    } else {
        nearest.to_bits()
    }
}

fn expected_gather(raw: &[u8], graph: &ContractGraph, case: Case) -> Vec<u8> {
    let end = case.pos + case.rows;
    let source_rows = graph
        .ids
        .iter()
        .flat_map(|&id| (0..4).map(move |r| id as usize * 4 + r))
        .chain(end / 4 * 4..end);
    let mut expected = Vec::with_capacity(graph.gathered_rows * WIDTH * 2 + GUARD);
    for row in source_rows {
        for dim in 0..WIDTH {
            let element = row * WIDTH + dim;
            let segment = element / SEG_ELEMS;
            let local = element % SEG_ELEMS;
            let code = raw[segment * SEG_BYTES + local] as i8;
            let scale = segment * SEG_BYTES + SEG_ELEMS + local / 32 * 2;
            let d = half::f16::from_bits(u16::from_le_bytes(
                raw[scale..scale + 2].try_into().unwrap(),
            ))
            .to_f32();
            expected.extend_from_slice(&gather_pack_half_bits(d * f32::from(code)).to_le_bytes());
        }
    }
    assert_eq!(expected.len(), graph.gathered_rows * WIDTH * 2);
    expected.extend(std::iter::repeat_n(CANARY, GUARD));
    expected
}

fn backend() -> VulkanBackend {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.kernels.vulkan.q8_kv_write_segmented = false;
    cfg.kernels.vulkan.q8_kv_write_fusion = false;
    cfg.kernels.vulkan.q8_decode_segmented = false; // Isolate O04 from A01 in both legs.
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(
        be.caps().name.contains("7900 XTX"),
        "requires parent campaign XTX, got {}",
        be.caps().name
    );
    assert!(
        be.prefers_segmented_q8_decode(),
        "production O04 device gate is closed"
    );
    be
}

#[derive(Clone, Copy, Debug)]
enum ObserverRewrite {
    PartialCopy,
    SparseCopyStrided,
    FullCopy,
}

impl ObserverRewrite {
    fn full(self) -> bool {
        matches!(self, Self::FullCopy)
    }

    fn patched(self, index: usize) -> bool {
        match self {
            Self::PartialCopy => index < 32,
            Self::SparseCopyStrided => index % 64 < 32,
            Self::FullCopy => true,
        }
    }
}

struct SideObserver {
    kind: ObserverRewrite,
    patch_id: TensorId,
    tap_id: TensorId,
    patch: Data,
    tap: Data,
}

impl SideObserver {
    fn new(be: &VulkanBackend, c: &mut ContractGraph, kind: ObserverRewrite) -> Self {
        let k16 = match c.graph.ops[1] {
            Op::WriteKv { src, .. } => src,
            _ => unreachable!(),
        };
        assert_eq!(c.graph.desc(k16).numel(), WIDTH);
        let patch_id = c.graph.input(TensorDesc::new([WIDTH], DType::F16));
        let tap_id = c.graph.output(TensorDesc::new([WIDTH], DType::F16));
        match kind {
            ObserverRewrite::PartialCopy | ObserverRewrite::FullCopy => c.graph.push(Op::Copy {
                src: patch_id,
                src_off: 0,
                dst: k16,
                dst_off: 0,
                n: if kind.full() { WIDTH as u32 } else { 32 },
            }),
            ObserverRewrite::SparseCopyStrided => c.graph.push(Op::CopyStrided {
                src: patch_id,
                src_off: 0,
                src_stride: 64,
                dst: k16,
                dst_off: 0,
                dst_stride: 64,
                rows: 8,
                n: 32,
            }),
        }
        c.graph.push(Op::Copy {
            src: k16,
            src_off: 0,
            dst: tap_id,
            dst_off: 0,
            n: WIDTH as u32,
        });
        let bits: Vec<u16> = (0..WIDTH)
            .map(|i| half::f16::from_f32(-4.0 + (i % 17) as f32 / 16.0).to_bits())
            .collect();
        Self {
            kind,
            patch_id,
            tap_id,
            patch: Data::new(
                be,
                bytemuck::cast_slice(&bits).to_vec(),
                BufferUsage::Activations,
            ),
            tap: Data::new(be, vec![CANARY; WIDTH * 2], BufferUsage::Readback),
        }
    }

    fn capture(&self, be: &VulkanBackend, key: &str) -> Vec<u8> {
        exact(
            &read(be, self.patch.buffer.as_ref()),
            &self.patch.initial,
            &format!("{key}-patch-unchanged"),
            true,
        );
        let got = read(be, self.tap.buffer.as_ref());
        assert!(got[WIDTH * 2..].iter().all(|&b| b == CANARY));
        let mut nonzero_old = false;
        let mut distinct_old = false;
        for i in 0..WIDTH {
            let value = &got[i * 2..i * 2 + 2];
            let patch = &self.patch.initial[i * 2..i * 2 + 2];
            let f = half::f16::from_bits(u16::from_le_bytes(value.try_into().unwrap())).to_f32();
            assert!(f.is_finite(), "{key}: nonfinite observer element {i}");
            if self.kind.patched(i) {
                assert_eq!(value, patch, "{key}: patched observer element {i}");
            } else {
                nonzero_old |= f != 0.0;
                distinct_old |= value != patch;
            }
        }
        if !self.kind.full() {
            assert!(
                nonzero_old && distinct_old,
                "{key}: vacuous preserved F16 suffix/gaps"
            );
        }
        got
    }
}

fn run(case: Case) {
    run_with_observer(case, None);
}

fn run_with_observer(case: Case, rewrite: Option<ObserverRewrite>) {
    // A fresh backend per geometry gives an unambiguous actual kernel-construction witness.
    let mut be = backend();
    let mut c = contract_graph(case);
    let segments = c.capacity_rows / SEG_ROWS;
    let _pool = be.init_unified_vram(512 * 1024 * 1024).unwrap();
    let observer = rewrite.map(|kind| SideObserver::new(&be, &mut c, kind));
    let key = match rewrite {
        Some(kind) => format!("{}-observer-{kind:?}", case.key()),
        None => case.key(),
    };
    let caches = [zero_cache(&be, segments), zero_cache(&be, segments)];
    let mut x = signal(case.rows * WIDTH, 211, 3.0);
    let mut value = signal(case.rows * WIDTH, 227, 1.0);
    if case.rows > 1 {
        x[..WIDTH].fill(0.0);
        value[..WIDTH].fill(0.0);
    }
    let norm: Vec<f32> = signal(HD, 223, 0.3).into_iter().map(|x| x + 1.0).collect();
    let positions = case.positions();
    let inputs = [
        Data::new(
            &be,
            bytemuck::cast_slice(&x).to_vec(),
            BufferUsage::Activations,
        ),
        Data::new(
            &be,
            bytemuck::cast_slice(&norm).to_vec(),
            BufferUsage::Weights,
        ),
        Data::new(
            &be,
            bytemuck::cast_slice(&positions).to_vec(),
            BufferUsage::Staging,
        ),
        Data::new(
            &be,
            bytemuck::cast_slice(&value).to_vec(),
            BufferUsage::Activations,
        ),
        Data::new(
            &be,
            bytemuck::cast_slice(&c.ids).to_vec(),
            BufferUsage::Activations,
        ),
    ];
    let output_bytes = c.gathered_rows * WIDTH * 2;
    let outputs: [Data; 2] =
        std::array::from_fn(|_| Data::new(&be, vec![CANARY; output_bytes], BufferUsage::Readback));
    let mut bindings = Bindings::new();
    for (id, input) in [c.x, c.norm, c.positions, c.value, c.indices]
        .into_iter()
        .zip(&inputs)
    {
        bindings.bind(id, input.buffer.as_ref());
    }
    for side in 0..2 {
        bindings.bind(c.caches[side], caches[side].as_ref());
        bindings.bind(c.outputs[side], outputs[side].buffer.as_ref());
    }
    if let Some(observer) = &observer {
        bindings.bind(observer.patch_id, observer.patch.buffer.as_ref());
        bindings.bind(observer.tap_id, observer.tap.buffer.as_ref());
    }
    let tables_before: [Vec<u8>; 2] = caches.each_ref().map(|b| {
        read(
            &be,
            crate::as_segmented_kv(b.as_ref()).unwrap().table_buffer(),
        )
    });
    let mut reference: Option<([Vec<u8>; 2], [Vec<u8>; 2], Option<Vec<u8>>)> = None;
    for enabled in [false, true] {
        let mut cfg = be.cfg().clone();
        cfg.kernels.vulkan.q8_kv_write_segmented = enabled;
        be.cfg = Arc::new(cfg);
        assert!(!be.cfg().kernels.vulkan.q8_decode_segmented);
        assert!(!be.cfg().kernels.vulkan.q8_kv_write_fusion);
        for input in &inputs {
            input.reset(&be);
        }
        for output in &outputs {
            output.reset(&be);
        }
        if let Some(observer) = &observer {
            observer.patch.reset(&be);
            observer.tap.reset(&be);
        }
        for cache in &caches {
            reset_cache(&be, cache.as_ref());
        }
        let scratch: Vec<Option<Box<dyn Buffer>>> =
            (0..c.graph.tensors.len()).map(|_| None).collect();
        let mut rope_pos = HashMap::new();
        if !case.mrope {
            // Same existing pos0 read as execute_static; never substitute MRoPE's T coordinate.
            rope_pos.insert(
                c.positions.0,
                read_pos0(&be, inputs[2].buffer.as_ref()).unwrap() as usize,
            );
        }
        let selected = q8_kv_write_segmented_fusions(
            &be,
            &c.graph,
            &scratch,
            &bindings,
            &RopeMode::Static(&rope_pos),
            &HashSet::new(),
        )
        .unwrap();
        let expect_fusion = enabled && rewrite.is_none_or(ObserverRewrite::full);
        assert_eq!(
            selected.len(),
            usize::from(expect_fusion),
            "{key}: non-vacuous production matcher gate"
        );
        if expect_fusion {
            let pair = &selected[&0];
            assert_eq!(
                (pair.cache, pair.positions, pair.pos),
                (c.caches[0], c.positions, case.pos as u32)
            );
        }
        assert!(
            !be.built_kernel_names().contains(&case.kernel()),
            "{key}: candidate was built before its on execution"
        );
        let plan = be.compile(&c.graph).unwrap();
        let vk_plan = plan.as_any().downcast_ref::<VkDecodePlan>().unwrap();
        assert!(
            !vk_plan.eligible,
            "actual Graph must execute static lowering"
        );
        let kernels_before = be.built_kernel_names();
        be.execute(plan.as_ref(), &bindings).unwrap();
        {
            let kernels = be.shared.kernels.lock().unwrap();
            let mut built_by_graph: Vec<_> = kernels
                .keys()
                .copied()
                .filter(|name| !kernels_before.contains(name))
                .collect();
            built_by_graph.sort_unstable();
            eprintln!("O04_GRAPH built_by_execute={built_by_graph:?} enabled={enabled} expected_fusion={expect_fusion} key={key}");
            if matches!(rewrite, Some(ObserverRewrite::SparseCopyStrided)) {
                assert!(
                    kernels.contains_key("copy_strided"),
                    "{key}: actual sparse overwrite dispatch did not run"
                );
            }
            assert!(
                kernels.contains_key("qsa_gather_q8_seg"),
                "{key}: actual downstream gather did not run"
            );
            {
                use std::hash::{Hash, Hasher};
                let gather = &kernels["qsa_gather_q8_seg"];
                let mut expected = std::collections::hash_map::DefaultHasher::new();
                crate::gemm::qsa_gather_q8_seg_spv().hash(&mut expected);
                assert_eq!((gather.n_buf, gather.push_size), (5, 36));
                assert_eq!(
                    gather.spv_hash,
                    expected.finish(),
                    "{key}: gather SPV identity"
                );
            }
            if expect_fusion {
                let k = kernels
                    .get(case.kernel())
                    .expect("actual execute(Graph) did not build O04 kernel");
                assert_eq!(k.subgroup_contract, crate::ops::SubgroupContract::Fixed(32));
                assert_eq!((k.n_buf, k.push_size), (4, 48));
                eprintln!(
                    "O04_GRAPH actual_kernel={} spv_hash={:016x} bindings={} push={} key={key}",
                    k.name, k.spv_hash, k.n_buf, k.push_size
                );
            } else {
                assert!(!kernels.contains_key(NORMAL) && !kernels.contains_key(MROPE));
                // This fixture installs a unified pool. Its Internal F16 activation scratch is
                // UnifiedSub-backed and exposes a BDA, so the plain static adapter selects _at.
                // MRoPE has a separate bound recorder method irrespective of scratch address.
                assert!(
                    be.unified_vram().is_some(),
                    "{key}: baseline allocation premise changed"
                );
                let (name, words, n_buf) = if case.mrope {
                    (
                        "qk_norm_rope_mrope",
                        crate::gemm::qk_norm_rope_mrope_spv(),
                        4,
                    )
                } else {
                    ("qk_norm_rope_bda", crate::gemm::qk_norm_rope_bda_spv(), 3)
                };
                let baseline = kernels.get(name).unwrap_or_else(|| {
                    panic!("{key}: expected actual baseline {name}; newly built={built_by_graph:?}")
                });
                assert!(
                    enabled || !kernels_before.contains(&name),
                    "{key}: stale baseline pipeline witness"
                );
                assert_eq!(
                    (baseline.name, baseline.n_buf, baseline.push_size),
                    (name, n_buf, 44)
                );
                // Same word hashing as ops::hash_spv; prove shader identity, not a permissive name list.
                use std::hash::{Hash, Hasher};
                let mut expected = std::collections::hash_map::DefaultHasher::new();
                words.hash(&mut expected);
                assert_eq!(
                    baseline.spv_hash,
                    expected.finish(),
                    "{key}: baseline SPV identity"
                );
                eprintln!(
                    "O04_GRAPH actual_baseline={} spv_hash={:016x} bindings={} push={} key={key}",
                    baseline.name, baseline.spv_hash, baseline.n_buf, baseline.push_size
                );
                assert!(kernels.contains_key("store_q8_f16_seg"));
            }
            assert!(
                kernels.contains_key("store_q8_seg"),
                "{key}: actual F32 V write did not run"
            );
        }
        for (i, input) in inputs.iter().enumerate() {
            exact(
                &read(&be, input.buffer.as_ref()),
                &input.initial,
                &format!("{key}-input{i}-on{enabled}"),
                false,
            );
        }
        let got_cache = caches.each_ref().map(|b| read(&be, b.as_ref()));
        let got_output = outputs.each_ref().map(|b| read(&be, b.buffer.as_ref()));
        let got_observer = observer.as_ref().map(|o| o.capture(&be, &key));
        for side in 0..2 {
            check_write_extent(&got_cache[side], case, if side == 0 { "K" } else { "V" });
            exact(
                &expected_gather(&got_cache[side], &c, case),
                &got_output[side],
                &format!("{key}-gather{side}-physical-layout-on{enabled}"),
                true,
            );
            assert!(
                got_output[side][output_bytes..]
                    .iter()
                    .all(|&b| b == CANARY),
                "{key}: output{side} physical tail"
            );
            let values: Vec<f32> = got_output[side][..output_bytes]
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect();
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(
                values.iter().any(|v| *v != 0.0),
                "{key}: gathered output{side} is vacuous"
            );
            let segmented = crate::as_segmented_kv(caches[side].as_ref()).unwrap();
            assert_eq!(
                segmented.committed(),
                segments,
                "Graph changed committed segment count"
            );
            exact(
                &read(&be, segmented.table_buffer()),
                &tables_before[side],
                &format!("{key}-LUT{side}"),
                false,
            );
        }
        if let Some((ref cache_ref, ref output_ref, ref observer_ref)) = reference {
            for side in 0..2 {
                exact(
                    &cache_ref[side],
                    &got_cache[side],
                    &format!("{key}-physical-cache{side}"),
                    false,
                );
                exact(
                    &output_ref[side],
                    &got_output[side],
                    &format!("{key}-gather-output{side}"),
                    true,
                );
            }
            assert_eq!(observer_ref.is_some(), got_observer.is_some());
            if let (Some(expected), Some(actual)) = (observer_ref, &got_observer) {
                exact(expected, actual, &format!("{key}-side-observer"), true);
            }
        } else {
            reference = Some((got_cache, got_output, got_observer));
        }
    }
    eprintln!("O04_GRAPH PASS key={key} actual_execute_graph=true all_physical_KV_exact=true outputs_exact=true A01_segmented_decode=false flat_replay=false no_lifecycle_change=true");
}

#[test]
fn o04_gather_pack_half_rounding_contract() {
    // Unit6's first baseline failure: code72 * scale0x215f is not a midpoint.
    let product = half::f16::from_bits(0x215f).to_f32() * 72.0;
    assert_eq!(product, 0.75531005859375);
    assert_eq!(half::f16::from_f32(product).to_bits(), 0x3a0b);
    assert_eq!(gather_pack_half_bits(product), 0x3a0a);
    assert_eq!(gather_pack_half_bits(-product), 0xba0a);
    assert_eq!(gather_pack_half_bits(1.0), 0x3c00);
    assert_eq!(gather_pack_half_bits(-0.0), 0x8000);
    assert_eq!(gather_pack_half_bits(3.0 * 2.0f32.powi(-25)), 0x0001);
    assert_eq!(gather_pack_half_bits(-3.0 * 2.0f32.powi(-25)), 0x8001);
    assert_eq!(gather_pack_half_bits(65520.0), 0x7bff);
}

#[test]
fn o04_segmented_write_graph_fixture_contract() {
    for mrope in [false, true] {
        for (rows, pos, rope) in [
            (1, 32767, 64),
            (33, 32765, 64),
            (129, 32765, 256),
            (129, 150003, 256),
        ] {
            let case = Case {
                rows,
                pos,
                rope,
                mrope,
            };
            let c = contract_graph(case);
            assert_eq!(c.graph.ops.len(), 4);
            assert!(c.graph.no_decode_replay && !c.graph.independent_rows && !c.graph.mtp_verify);
            assert_eq!(
                Q8SegmentedGraphFacts::analyze(&c.graph).peers.get(&1),
                Some(&c.caches[1])
            );
            assert!(Q8SegmentedGraphFacts::analyze(&c.graph)
                .dead_scratch
                .contains(&1));
            assert_eq!(c.graph.desc(c.outputs[0]).numel(), c.gathered_rows * WIDTH);
            let p = case.positions();
            assert_eq!(p.len(), rows * if mrope { 4 } else { 1 });
            if mrope {
                assert_ne!(p[0] as usize, pos);
            } else {
                assert_eq!(p[0] as usize, pos);
                if rows > 1 {
                    assert_ne!(p[1] as usize, pos + 1);
                }
            }
            if rows == 33 {
                assert!(pos < SEG_ROWS && pos + rows > SEG_ROWS);
                assert_ne!((pos + rows) % 4, 0);
            }
        }
    }
}

#[test]
#[ignore = "parent unit5 serial GPU: actual static Norm->WriteKvK->WriteKvV->QsaGather Graph on/off contract"]
fn o04_segmented_write_actual_graph_byte_contract() {
    let mut executed = 0;
    for mrope in [false, true] {
        for (rows, pos, rope) in [
            (1, 32767, 64),
            (33, 32765, 64),
            (129, 32765, 256),
            (129, 150003, 256),
        ] {
            let case = Case {
                rows,
                pos,
                rope,
                mrope,
            };
            if std::env::var("O04_GRAPH_CASE").is_ok_and(|key| key != case.key()) {
                continue;
            }
            executed += 1;
            run(case);
        }
    }
    assert!(executed > 0, "O04_GRAPH_CASE selected no Graph case");
}

#[test]
fn o04_partial_overwrite_preserves_later_f16_observers() {
    for mrope in [false, true] {
        for case in 0..10 {
            let mut c = contract_graph(Case {
                rows: 1,
                pos: 32767,
                rope: 64,
                mrope,
            });
            let k16 = match c.graph.ops[1] {
                Op::WriteKv { src, .. } => src,
                _ => unreachable!(),
            };
            let patch = c.graph.input(TensorDesc::new([512], DType::F16));
            let tap = c.graph.output(TensorDesc::new([512], DType::F16));
            let copy = |n| Op::Copy {
                src: patch,
                src_off: 0,
                dst: k16,
                dst_off: 0,
                n,
            };
            let observe = Op::Copy {
                src: k16,
                src_off: 0,
                dst: tap,
                dst_off: 0,
                n: 512,
            };
            match case {
                0 => c.graph.push(copy(32)),
                1 => c.graph.push(Op::Copy {
                    src: patch,
                    src_off: 0,
                    dst: k16,
                    dst_off: 32,
                    n: 480,
                }),
                2 => c.graph.push(Op::CopyStrided {
                    src: patch,
                    src_off: 0,
                    src_stride: 32,
                    dst: k16,
                    dst_off: 0,
                    dst_stride: 64,
                    rows: 8,
                    n: 32,
                }),
                3 => c.graph.push(copy(512)),
                4 => c.graph.push(Op::CopyStrided {
                    src: patch,
                    src_off: 0,
                    src_stride: 64,
                    dst: k16,
                    dst_off: 0,
                    dst_stride: 64,
                    rows: 8,
                    n: 64,
                }),
                5 => {
                    c.graph.push(copy(32));
                    c.graph.push(copy(512));
                }
                6 => {
                    c.graph.push(copy(32));
                    c.graph.push(observe.clone());
                    c.graph.push(copy(512));
                }
                7 => c.graph.push(Op::Copy {
                    src: k16,
                    src_off: 0,
                    dst: k16,
                    dst_off: 0,
                    n: 512,
                }),
                8 => {} // A direct observer without any rewrite must also decline fusion.
                9 => {
                    // Together these cover every element, but neither proves a full rewrite.
                    c.graph.push(copy(256));
                    c.graph.push(Op::Copy {
                        src: patch,
                        src_off: 256,
                        dst: k16,
                        dst_off: 256,
                        n: 256,
                    });
                }
                _ => unreachable!(),
            }
            c.graph.push(observe);
            let facts = Q8SegmentedGraphFacts::analyze(&c.graph);
            assert_eq!(
                facts.dead_scratch.contains(&1),
                matches!(case, 3 | 4 | 5),
                "mrope={mrope} rewrite_case={case}"
            );
            // The fix must not alter the first-cache observer or coupled-K/V proof.
            assert_eq!(facts.peers.get(&1), Some(&c.caches[1]));
        }
    }
}

#[test]
#[ignore = "parent serial GPU: O04 partial/sparse observer fallback and full-overwrite positive control"]
fn o04_overwrite_observer_actual_graph_byte_contract() {
    for mrope in [false, true] {
        for rewrite in [
            ObserverRewrite::PartialCopy,
            ObserverRewrite::SparseCopyStrided,
            ObserverRewrite::FullCopy,
        ] {
            run_with_observer(
                Case {
                    rows: 1,
                    pos: 32767,
                    rope: 64,
                    mrope,
                },
                Some(rewrite),
            );
        }
    }
}

#[test]
fn o04_only_proven_full_norm_redefinitions_kill_scratch() {
    for mrope in [false, true] {
        for full in [false, true] {
            let mut c = contract_graph(Case {
                rows: 1,
                pos: 32767,
                rope: 64,
                mrope,
            });
            let k16 = match c.graph.ops[1] {
                Op::WriteKv { src, .. } => src,
                _ => unreachable!(),
            };
            let tap = c.graph.output(TensorDesc::new([512], DType::F16));
            let mut redefine = c.graph.ops[0].clone();
            match &mut redefine {
                Op::QkNormRope { n_head, .. } | Op::QkNormMrope { n_head, .. } => {
                    *n_head = if full { 2 } else { 1 };
                }
                _ => unreachable!(),
            }
            assert_eq!(
                q8_segmented_full_scratch_write(&c.graph, &redefine, k16),
                full
            );
            c.graph.push(redefine);
            c.graph.push(Op::Copy {
                src: k16,
                src_off: 0,
                dst: tap,
                dst_off: 0,
                n: 512,
            });
            let facts = Q8SegmentedGraphFacts::analyze(&c.graph);
            assert_eq!(facts.dead_scratch.contains(&1), full);
            assert_eq!(facts.peers.get(&1), Some(&c.caches[1]));
        }
        let c = contract_graph(Case {
            rows: 1,
            pos: 32767,
            rope: 64,
            mrope,
        });
        // Existing genuinely dead intermediate and committed-cache facts remain eligible.
        assert!(Q8SegmentedGraphFacts::analyze(&c.graph)
            .dead_scratch
            .contains(&1));
    }
}
