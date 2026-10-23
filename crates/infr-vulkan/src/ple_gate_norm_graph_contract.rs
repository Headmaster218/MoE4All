//! NEW adapter child, mounted only by the proposed O07 integration artifact.
//! Portable tests call production pure helpers. Ignored tests execute real Graphs.
use super::*;
use infr_core::{DType, TensorDesc};

const D: usize = 2560;
const W: usize = 4 * D;
const GUARD: usize = 32;
const CANARY: u32 = 0x7fc3_cafe;
const FUSED: &str = "qwen_ple_gate_norm_lds2560";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Base,
    DeadKey,
    FlagOff,
    SpecMtp,
    Verify,
    Independent,
    BetweenCopy,
    LaterObserver,
    ExternalGated,
    NormQueryAlias,
    NormValueAlias,
    NormGammaAlias,
    UnmeasuredRows,
}

struct Fixture {
    graph: Graph,
    key_in: TensorId,
    key: TensorId,
    query: TensorId,
    value: TensorId,
    norm: TensorId,
    gated: TensorId,
    normed: TensorId,
    zero: TensorId,
    out: [TensorId; 2],
    side: TensorId,
    rows: usize,
}

fn fixture(rows: usize, mode: Mode) -> Fixture {
    let mut g = Graph::new();
    let desc = |n| TensorDesc::new(vec![n], DType::F32);
    let key_in = g.input(desc(rows * W));
    let key = g.internal(desc(rows * W));
    let query = g.input(desc(rows * W));
    let value = g.input(desc(rows * D));
    let norm = g.weight(desc(W));
    let gated = if mode == Mode::ExternalGated {
        g.output(desc(rows * W))
    } else {
        g.internal(desc(rows * W))
    };
    let normed = if mode == Mode::DeadKey {
        key
    } else {
        g.output(desc(rows * W))
    };
    let zero = g.input(desc(rows * W));
    let out = [g.output(desc(rows * W)), g.output(desc(rows * W))];
    let side = g.output(desc(rows * W));
    g.push(Op::Copy {
        src: key_in,
        src_off: 0,
        dst: key,
        dst_off: 0,
        n: (rows * W) as u32,
    });
    g.push(Op::QwenPleGate {
        key,
        query,
        value,
        dst: gated,
        rows: rows as u32,
        hc: 4,
        n_embd: D as u32,
    });
    let observer = || Op::Copy {
        src: gated,
        src_off: 0,
        dst: side,
        dst_off: 0,
        n: (rows * W) as u32,
    };
    if mode == Mode::BetweenCopy {
        g.push(observer());
    }
    g.push(Op::QwenHcNorm {
        x: gated,
        norm,
        dst: normed,
        rows: rows as u32,
        hc: 4,
        n_embd: D as u32,
        eps: 1e-6,
    });
    if mode == Mode::LaterObserver {
        g.push(observer());
    }
    // -0 is an exact finite add identity including both signs of zero. This is the
    // ordinary later gated-residual Add, not a Copy/tap of the intermediate.
    g.push(Op::Add {
        a: gated,
        b: zero,
        dst: out[0],
        n: (rows * W) as u32,
    });
    g.push(Op::Copy {
        src: normed,
        src_off: 0,
        dst: out[1],
        dst_off: 0,
        n: (rows * W) as u32,
    });
    g.no_decode_replay = true;
    g.independent_rows = mode == Mode::Independent;
    if g.independent_rows {
        g.sequence_spans.push(SequenceSpan {
            row_start: 0,
            rows: rows as u32,
            start_pos: 17,
        });
    }
    g.mtp_verify = mode == Mode::Verify;
    Fixture {
        graph: g,
        key_in,
        key,
        query,
        value,
        norm,
        gated,
        normed,
        zero,
        out,
        side,
        rows,
    }
}

struct Address {
    start: Option<u64>,
    bytes: usize,
}
impl Buffer for Address {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device_addr(&self) -> Option<u64> {
        self.start
    }
}

fn fake_scratch(g: &Graph) -> Vec<Option<Box<dyn Buffer>>> {
    g.tensors
        .iter()
        .enumerate()
        .map(|(i, d)| {
            Some(Box::new(Address {
                start: Some(0x1000_0000 + i as u64 * 0x1000_0000),
                bytes: d.desc.numel() * 4,
            }) as Box<dyn Buffer>)
        })
        .collect()
}

fn set_view(
    scratch: &mut [Option<Box<dyn Buffer>>],
    id: TensorId,
    address: Option<u64>,
    bytes: usize,
) {
    scratch[id.0 as usize] = Some(Box::new(Address {
        start: address,
        bytes,
    }));
}

#[test]
fn o07_device_policy_is_portable_and_narrow() {
    let base = infr_core::backend::Capabilities {
        subgroup_min: 32,
        subgroup_max: 64,
        max_shared_memory_bytes: 11296,
        ..Default::default()
    };
    assert!(crate::ple_gate_norm_lds_device_ok(
        true,
        crate::caps::DeviceArch::AmdRdna3,
        &base
    ));
    assert!(!crate::ple_gate_norm_lds_device_ok(
        false,
        crate::caps::DeviceArch::AmdRdna3,
        &base
    ));
    assert!(!crate::ple_gate_norm_lds_device_ok(
        true,
        crate::caps::DeviceArch::Other,
        &base
    ));
    for i in 0..4 {
        let mut caps = base.clone();
        match i {
            0 => caps.integrated = true,
            1 => caps.subgroup_min = 64,
            2 => caps.subgroup_min = 0,
            _ => caps.max_shared_memory_bytes = 11295,
        }
        assert!(!crate::ple_gate_norm_lds_device_ok(
            true,
            crate::caps::DeviceArch::AmdRdna3,
            &caps
        ));
    }
    let mut caps = base.clone();
    caps.subgroup_max = 16;
    assert!(!crate::ple_gate_norm_lds_device_ok(
        true,
        crate::caps::DeviceArch::AmdRdna3,
        &caps
    ));
    assert_eq!(crate::PLE_GATE_NORM_LDS_SHARED, (256 + 8 + 2560) * 4);
}

#[test]
fn o07_shape_dtype_observer_and_row_fallbacks() {
    for rows in [1, 2, 4, 9, 512, 513] {
        for mode in [Mode::Base, Mode::DeadKey] {
            let f = fixture(rows, mode);
            assert!(ple_gate_norm_at(&f.graph, 1).is_some());
            assert!(ple_gate_norm_at(&f.graph, usize::MAX).is_none());
        }
    }
    for rows in [0, 3, 8, 511, 514, 1024] {
        assert!(ple_gate_norm_at(&fixture(rows, Mode::Base).graph, 1).is_none());
    }
    for mode in [
        Mode::Verify,
        Mode::Independent,
        Mode::BetweenCopy,
        Mode::LaterObserver,
        Mode::ExternalGated,
    ] {
        assert!(
            ple_gate_norm_at(&fixture(1, mode).graph, 1).is_none(),
            "{mode:?}"
        );
    }
    let f = fixture(1, Mode::Base);
    for id in [f.key, f.query, f.value, f.norm, f.gated, f.normed] {
        let mut g = f.graph.clone();
        g.tensors[id.0 as usize].desc.dtype = DType::F16;
        assert!(ple_gate_norm_at(&g, 1).is_none());
    }
    for i in 0..5 {
        let mut g = f.graph.clone();
        if let Op::QwenHcNorm {
            x,
            hc,
            n_embd,
            rows,
            eps,
            ..
        } = &mut g.ops[2]
        {
            match i {
                0 => *x = f.query,
                1 => *hc = 1,
                2 => *n_embd = 2561,
                3 => *rows = 2,
                _ => *eps = f32::NAN,
            }
        }
        assert!(ple_gate_norm_at(&g, 1).is_none());
    }
}

#[test]
fn o07_physical_aliases_and_dead_key_liveness() {
    let f = fixture(1, Mode::Base);
    let pair = ple_gate_norm_at(&f.graph, 1).unwrap();
    let bindings = Bindings::new();
    let original = fake_scratch(&f.graph);
    assert!(ple_gate_norm_buffers(
        &f.graph, 1, pair, &original, &bindings
    ));
    for dst in [f.gated, f.normed] {
        for src in [f.key, f.query, f.value, f.norm] {
            let mut scratch = fake_scratch(&f.graph);
            let address = scratch[src.0 as usize].as_ref().unwrap().device_addr();
            set_view(&mut scratch, dst, address, W * 4);
            assert_eq!(
                ple_gate_norm_buffers(&f.graph, 1, pair, &scratch, &bindings),
                dst == f.normed && src == f.key,
                "dst={dst:?} src={src:?}"
            );
        }
    }
    let key_addr = original[f.key.0 as usize]
        .as_ref()
        .unwrap()
        .device_addr()
        .unwrap();
    for address in [None, Some(key_addr + 4), Some(u64::MAX - 3)] {
        let mut scratch = fake_scratch(&f.graph);
        set_view(&mut scratch, f.normed, address, W * 4);
        assert!(!ple_gate_norm_buffers(
            &f.graph, 1, pair, &scratch, &bindings
        ));
    }
    let mut scratch = fake_scratch(&f.graph);
    let gated_addr = scratch[f.gated.0 as usize].as_ref().unwrap().device_addr();
    set_view(&mut scratch, f.normed, gated_addr, W * 4);
    assert!(!ple_gate_norm_buffers(
        &f.graph, 1, pair, &scratch, &bindings
    ));
    let mut scratch = fake_scratch(&f.graph);
    set_view(&mut scratch, f.gated, None, W * 4);
    assert!(!ple_gate_norm_buffers(
        &f.graph, 1, pair, &scratch, &bindings
    ));
    let mut scratch = fake_scratch(&f.graph);
    let addr = scratch[f.query.0 as usize].as_ref().unwrap().device_addr();
    set_view(&mut scratch, f.query, addr, 4);
    assert!(!ple_gate_norm_buffers(
        &f.graph, 1, pair, &scratch, &bindings
    ));

    let mut g = f.graph.clone();
    g.push(Op::Copy {
        src: f.key,
        src_off: 0,
        dst: f.side,
        dst_off: 0,
        n: W as u32,
    });
    let mut scratch = fake_scratch(&g);
    set_view(&mut scratch, f.normed, Some(key_addr), W * 4);
    assert!(
        !ple_gate_norm_buffers(&g, 1, pair, &scratch, &bindings),
        "old key is not dead"
    );
    assert!(
        ple_gate_norm_buffers(&g, 1, pair, &original, &bindings),
        "a live old key is harmless when the output does not overlap it"
    );
    let mut redefined = f.graph.clone();
    redefined.push(Op::Copy {
        src: f.key_in,
        src_off: 0,
        dst: f.key,
        dst_off: 0,
        n: W as u32,
    });
    redefined.push(Op::Copy {
        src: f.key,
        src_off: 0,
        dst: f.side,
        dst_off: 0,
        n: W as u32,
    });
    assert!(
        ple_gate_norm_buffers(&redefined, 1, pair, &scratch, &bindings),
        "a full redefinition ends the old key lifetime before subsequent reads"
    );
    let mut partial = f.graph.clone();
    partial.push(Op::Copy {
        src: f.key_in,
        src_off: 0,
        dst: f.key,
        dst_off: 0,
        n: 1,
    });
    assert!(
        !ple_gate_norm_buffers(&partial, 1, pair, &scratch, &bindings),
        "a partial write cannot kill the entire old key"
    );
    let dead = fixture(1, Mode::DeadKey);
    let pair = ple_gate_norm_at(&dead.graph, 1).unwrap();
    assert!(ple_gate_norm_buffers(
        &dead.graph,
        1,
        pair,
        &fake_scratch(&dead.graph),
        &bindings
    ));
}

#[test]
fn o07_read_only_aliases_and_dead_frame_reuse() {
    let f = fixture(1, Mode::Base);
    let pair = ple_gate_norm_at(&f.graph, 1).unwrap();
    let bindings = Bindings::new();
    let mut scratch = fake_scratch(&f.graph);
    let output = scratch[f.normed.0 as usize].as_ref().unwrap().device_addr();
    let key = scratch[f.key.0 as usize].as_ref().unwrap().device_addr();
    set_view(&mut scratch, f.query, key, W * 4);
    assert!(
        ple_gate_norm_buffers(&f.graph, 1, pair, &scratch, &bindings),
        "read/read alias is allowed"
    );
    let mut g = f.graph.clone();
    let dead = g.internal(TensorDesc::new(vec![W], DType::F32));
    let mut scratch = fake_scratch(&g);
    set_view(&mut scratch, dead, output, W * 4);
    assert!(
        ple_gate_norm_buffers(&g, 1, pair, &scratch, &bindings),
        "unrelated dead frame IDs may overlap outputs in a baseline-valid pooled graph"
    );
}

#[test]
fn o07_production_wiring_and_signed_zero_identity() {
    let compact = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    let adapter = compact(include_str!("adapter.rs"));
    assert_eq!(
        adapter.matches("ple_gate_norm_pair(be_,graph,").count(),
        3,
        "lowering and BOTH static/replay skip sets must use the same physical matcher"
    );
    let predicate = adapter.split("fnple_gate_norm_pair(").nth(1).unwrap();
    assert!(predicate
        .split("fn")
        .next()
        .unwrap()
        .contains("be_.cfg().spec.mtp"));
    let recorder = compact(include_str!("recorder.rs"));
    let method = recorder
        .split("fnqwen_ple_gate_norm_lds(")
        .nth(1)
        .unwrap()
        .split("///")
        .next()
        .unwrap();
    assert!(method.contains("Self::vkb(gated_dst),Self::vkb(normed_dst),],2,&push,rows*4"));
    assert!(method.contains("6,16,32"));
    for bits in [0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 1, 0x8000_0001] {
        assert_eq!((f32::from_bits(bits) + -0.0f32).to_bits(), bits);
    }
}

fn signal(n: usize, seed: u32) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(1664525)
                .wrapping_add(seed.wrapping_mul(1013904223));
            if i % 41 == 0 {
                0x8000_0000
            } else if i % 29 == 0 {
                0
            } else {
                (((x >> 8) as f32 / 16777216.0 - 0.5) * 0.5).to_bits()
            }
        })
        .collect()
}

struct Data {
    buffer: Box<dyn Buffer>,
    initial: Vec<u32>,
    n: usize,
}
impl Data {
    fn new(be: &VulkanBackend, n: usize, seed: u32, zero: bool) -> Self {
        let mut initial = if zero {
            vec![0x8000_0000; n]
        } else {
            signal(n, seed)
        };
        initial.extend_from_slice(&[CANARY; GUARD]);
        let buffer = be.alloc(initial.len() * 4, BufferUsage::KvCache).unwrap();
        Self { buffer, initial, n }
    }
    fn reset(&self, be: &VulkanBackend) {
        be.upload(self.buffer.as_ref(), bytemuck::cast_slice(&self.initial))
            .unwrap();
    }
    fn read(&self, be: &VulkanBackend) -> Vec<u32> {
        let mut words = vec![0; self.n + GUARD];
        be.download(self.buffer.as_ref(), bytemuck::cast_slice_mut(&mut words))
            .unwrap();
        assert_eq!(&words[self.n..], &[CANARY; GUARD]);
        words.truncate(self.n);
        words
    }
}

fn run_graph(be: &mut VulkanBackend, rows: usize, mode: Mode) {
    let f = fixture(rows, mode);
    // Allocate source aliases at full output capacity; only their logical prefixes are inputs.
    let data: Vec<_> = (0..10)
        .map(|i| Data::new(be, rows * W, 17 + i * 31, i == 4))
        .collect();
    let norm_index = match mode {
        Mode::NormQueryAlias => 1,
        Mode::NormValueAlias => 2,
        Mode::NormGammaAlias => 3,
        _ => 5,
    };
    let mut bindings = Bindings::new();
    for (id, index) in [
        (f.key_in, 0),
        (f.query, 1),
        (f.value, 2),
        (f.norm, 3),
        (f.zero, 4),
        (f.out[0], 6),
        (f.out[1], 7),
        (f.side, 8),
    ] {
        bindings.bind(id, data[index].buffer.as_ref());
    }
    if mode != Mode::DeadKey {
        bindings.bind(f.normed, data[norm_index].buffer.as_ref());
    }
    if mode == Mode::ExternalGated {
        bindings.bind(f.gated, data[9].buffer.as_ref());
    }
    let mut expected = None;
    let is_eligible = matches!(mode, Mode::Base | Mode::DeadKey);
    for enabled in [false, true] {
        let mut cfg = be.cfg().clone();
        cfg.kernels.vulkan.ple_gate_norm_lds = enabled && mode != Mode::FlagOff;
        cfg.spec.mtp = mode == Mode::SpecMtp;
        be.cfg = Arc::new(cfg);
        for input in &data {
            input.reset(be);
        }
        let scratch = alloc_scratch(be, &f.graph).unwrap();
        assert_eq!(
            ple_gate_norm_pair(be, &f.graph, 1, &scratch, &bindings).is_some(),
            enabled && is_eligible,
            "actual physical matcher rows={rows} mode={mode:?} enabled={enabled}"
        );
        drop(scratch);
        let plan = be.compile(&f.graph).unwrap();
        be.execute(plan.as_ref(), &bindings).unwrap();
        if enabled && is_eligible {
            assert_eq!(
                be.shared
                    .kernels
                    .lock()
                    .unwrap()
                    .get(FUSED)
                    .expect("execute(Graph) never built O07 LDS")
                    .subgroup_contract,
                crate::ops::SubgroupContract::Fixed(32)
            );
        }
        let out = [data[6].read(be), data[7].read(be)];
        for words in &out {
            assert!(words.iter().all(|&x| f32::from_bits(x).is_finite()));
        }
        for input in &data {
            input.read(be);
        }
        if let Some(ref reference) = expected {
            assert_eq!(
                &out, reference,
                "both outputs exact rows={rows} mode={mode:?}"
            );
        } else {
            expected = Some(out);
        }
    }
    eprintln!("O07_GRAPH rows={rows} mode={mode:?} execute_graph=true exact=true canary=true");
}

fn backend() -> VulkanBackend {
    let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
    cfg.kernels.vulkan.ple_gate_norm_lds = false;
    cfg.spec.mtp = false;
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    assert!(be.caps().name.contains("7900 XTX"));
    assert!(be.prefers_ple_gate_norm_lds());
    // Test-owned arena makes actual Internal scratch addressable, as in resident execution.
    // No test hook redirects execute(Graph) or changes production scratch planning.
    let _pool = be.init_unified_vram(512 * 1024 * 1024).unwrap();
    be
}

#[test]
#[ignore = "parent serialized O07 GPU: actual Graph, measured rows, dual outputs and dead-key reuse"]
fn o07_actual_graph_exact_measured_rows() {
    let mut be = backend();
    for rows in [1, 2, 4, 9, 512, 513] {
        for mode in [Mode::Base, Mode::DeadKey] {
            run_graph(&mut be, rows, mode);
        }
    }
}

#[test]
#[ignore = "parent serialized O07 GPU: actual Graph fallback/observer/physical alias controls"]
fn o07_actual_graph_fallbacks() {
    let mut be = backend();
    for mode in [
        Mode::FlagOff,
        Mode::SpecMtp,
        Mode::Verify,
        Mode::Independent,
        Mode::BetweenCopy,
        Mode::LaterObserver,
        Mode::ExternalGated,
        Mode::NormQueryAlias,
        Mode::NormValueAlias,
        Mode::NormGammaAlias,
    ] {
        run_graph(&mut be, 1, mode);
    }
    run_graph(&mut be, 3, Mode::UnmeasuredRows);
}
