//! Self-contained Recorder child; parent mounts with #[cfg(test)] only.
//! No campaign modules, external SPVs, model files, environment probes, filesystem
//! output, timing or additional dependencies. GPU tests are ignored and must run
//! serially on a device/driver accepted by the production exact-order selector.
//! Full QA/DA/SA bits and guards are compared with the real split Recorder path.
//! F16 DA/SA overflow in finite extreme inputs is compared bitwise, not sanitized.
//! The graph contract is resident MMQ only, not a paged or end-to-end model proof.

use super::{Recorder, VulkanBackend};
use infr_core::{Backend, Bindings, Buffer, BufferUsage, DType, Graph, Op, TensorDesc};

const GUARD: usize = 64;
const CANARY: u8 = 0xa5;
const GRAPH_ROWS: usize = 257;
const GRAPH_NE: usize = 256;

fn exact(label: &str, got: &[u8], expected: &[u8], unit: usize) {
    assert_eq!(got.len(), expected.len(), "{label}: byte count");
    assert_eq!(got.len() % unit, 0, "{label}: element alignment");
    for (i, (a, b)) in got
        .chunks_exact(unit)
        .zip(expected.chunks_exact(unit))
        .enumerate()
    {
        assert!(
            a == b,
            "{label}: first differing element={i} actual={a:02x?} expected={b:02x?}"
        );
    }
}

fn backend(enabled: bool) -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    cfg.kernels.vulkan.silu_quant_q8 = enabled;
    cfg.kernels.vulkan.moe_small_m = 8;
    let be = VulkanBackend::new_with(std::sync::Arc::new(cfg)).unwrap();
    assert!(
        be.prefers_silu_quant_q8(),
        "ignored contract requires a production-eligible exact-order device/driver"
    );
    assert!(
        be.caps().i8_dot,
        "resident graph fixture requires native MMQ"
    );
    assert!(!be.moe_paged(), "this contract does not initialize a pager");
    run(&be, |r| {
        assert_eq!(r.can_silu_quant_q8(1, 512), enabled);
        assert!(!r.can_silu_quant_q8(0, 512));
        assert!(!r.can_silu_quant_q8(40961, 512));
        assert!(!r.can_silu_quant_q8(513, 768));
    });
    be
}

fn run(be: &VulkanBackend, record: impl FnOnce(&Recorder<'_>)) {
    let rec = be.recorder().unwrap();
    record(&rec);
    rec.finish().unwrap();
}

fn guarded(be: &VulkanBackend, bytes: usize, usage: BufferUsage) -> Box<dyn Buffer> {
    let b = be.alloc(bytes + GUARD, usage).unwrap();
    be.upload_range(b.as_ref(), bytes, &[CANARY; GUARD])
        .unwrap();
    b
}

fn upload_bytes(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let b = guarded(be, bytes.len(), usage);
    be.upload_range(b.as_ref(), 0, bytes).unwrap();
    b
}

fn guard(be: &VulkanBackend, b: &dyn Buffer, label: &str) {
    let mut tail = [0; GUARD];
    be.download_range(b, b.len_bytes() - GUARD, &mut tail)
        .unwrap();
    exact(label, &tail, &[CANARY; GUARD], 1);
}

fn read(be: &VulkanBackend, b: &dyn Buffer, label: &str) -> Vec<u8> {
    let mut data = vec![0; b.len_bytes()];
    be.download(b, &mut data).unwrap();
    let live = data.len() - GUARD;
    exact(label, &data[live..], &[CANARY; GUARD], 1);
    data.truncate(live);
    data
}

struct Quantized([Box<dyn Buffer>; 3]);

impl Quantized {
    fn new(be: &VulkanBackend, elems: usize) -> Self {
        Self(
            [elems, elems / 16, elems / 16]
                .map(|bytes| guarded(be, bytes, BufferUsage::Activations)),
        )
    }

    fn reset(&self, be: &VulkanBackend) {
        for b in &self.0 {
            be.upload(b.as_ref(), &vec![CANARY; b.len_bytes()]).unwrap();
        }
    }

    fn read(&self, be: &VulkanBackend, label: &str) -> [Vec<u8>; 3] {
        std::array::from_fn(|i| read(be, self.0[i].as_ref(), &format!("{label}/output{i}_guard")))
    }
}

#[derive(Clone, Copy, Debug)]
enum Recipe {
    Signed,
    Zero,
    NearRound,
    NearSiluRound,
    Extremes,
}

struct HostInput {
    gate: Vec<f32>,
    up: Vec<f32>,
}

fn next(x: &mut u32) -> u32 {
    *x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    *x
}

fn input(rows: usize, k: usize, recipe: Recipe) -> HostInput {
    let mut gate = vec![0.0; rows * k];
    let mut up = vec![0.0; rows * k];
    let mut seed = 0xe0100000u32 ^ rows as u32 ^ (k as u32).rotate_left(16);
    let a_pos = 0.7f32 / (1.0 + (-0.7f32).exp());
    let a_neg = -0.7f32 / (1.0 + 0.7f32.exp());
    for i in 0..gate.len() {
        let lane = i % 32;
        let signed = (next(&mut seed) >> 8) as f32 / 16777216.0 * 2.0 - 1.0;
        let (g, u) = match recipe {
            Recipe::Signed => (
                signed * 12.0,
                ((next(&mut seed) >> 8) as f32 / 16777216.0 * 2.0 - 1.0) * 9.0,
            ),
            Recipe::Zero => (
                if i % 2 == 0 { 0.0 } else { -0.0 },
                if i % 4 < 2 { -0.0 } else { 7.0 },
            ),
            Recipe::NearRound | Recipe::NearSiluRound => {
                let step = [1.0 / 128.0, 1.0, 1.0 / 65536.0][(i / 32) % 3];
                let mut v = if lane == 0 {
                    127.0 * step
                } else {
                    ((lane as f32 * 7.0) % 126.0 + 0.5) * step
                };
                if lane > 0 {
                    v = f32::from_bits(match (i / 32 + lane) % 3 {
                        0 => v.to_bits() - 1,
                        1 => v.to_bits(),
                        _ => v.to_bits() + 1,
                    });
                }
                if lane % 2 == 1 {
                    v = -v;
                }
                if matches!(recipe, Recipe::NearRound) {
                    (32.0, v / 32.0)
                } else if lane % 2 == 0 {
                    (0.7, v / a_pos)
                } else {
                    (-0.7, v / a_neg)
                }
            }
            Recipe::Extremes => match lane % 16 {
                0 => (f32::MAX, 1.0),
                1 => (f32::MAX, -1.0),
                2 => (-f32::MAX, 1.0),
                3 => (-104.0, -1.0),
                4 => (80.0, 1000.0),
                5 => (80.0, -1000.0),
                6 => (f32::MIN_POSITIVE, 1.0),
                7 => (-f32::MIN_POSITIVE, 1.0),
                8 => (1.0, f32::from_bits(1)),
                9 => (-1.0, -f32::from_bits(1)),
                10 => (-80.0, 65504.0),
                11 => (0.0, f32::MAX),
                12 => (-0.0, -f32::MAX),
                13 => (7.0, -7.0),
                14 => (-7.0, 7.0),
                _ => (0.5, -0.5),
            },
        };
        gate[i] = g;
        up[i] = u;
    }
    HostInput { gate, up }
}

struct Projection {
    gate: Box<dyn Buffer>,
    up: Option<Box<dyn Buffer>>,
    rows: usize,
    k: usize,
}

impl Projection {
    fn workspace(be: &VulkanBackend, rows: usize, k: usize, combined: bool) -> Self {
        Self {
            gate: guarded(
                be,
                rows * k * 4 * if combined { 2 } else { 1 },
                BufferUsage::Activations,
            ),
            up: (!combined).then(|| guarded(be, rows * k * 4, BufferUsage::Activations)),
            rows,
            k,
        }
    }

    fn source(be: &VulkanBackend, host: &HostInput, rows: usize, k: usize, combined: bool) -> Self {
        let data = Self::workspace(be, rows, k, combined);
        if combined {
            let mut gu = vec![0.0; rows * k * 2];
            for row in 0..rows {
                gu[row * k * 2..row * k * 2 + k]
                    .copy_from_slice(&host.gate[row * k..(row + 1) * k]);
                gu[row * k * 2 + k..(row + 1) * k * 2]
                    .copy_from_slice(&host.up[row * k..(row + 1) * k]);
            }
            be.upload_range(data.gate.as_ref(), 0, bytemuck::cast_slice(&gu))
                .unwrap();
        } else {
            be.upload_range(data.gate.as_ref(), 0, bytemuck::cast_slice(&host.gate))
                .unwrap();
            be.upload_range(
                data.up.as_deref().unwrap(),
                0,
                bytemuck::cast_slice(&host.up),
            )
            .unwrap();
        }
        data
    }

    fn restore(&self, r: &Recorder<'_>, source: &Self, w: &dyn Buffer, weight_before: bool) {
        let width = self.k * if self.up.is_none() { 2 } else { 1 };
        r.copy(
            source.gate.as_ref(),
            0,
            self.gate.as_ref(),
            0,
            self.rows * width * 4,
        );
        if let Some(up) = &self.up {
            r.copy(
                source.up.as_deref().unwrap(),
                0,
                up.as_ref(),
                0,
                self.rows * self.k * 4,
            );
        }
        if weight_before {
            r.moe_weight_scale(self.gate.as_ref(), w, self.rows, width);
            if let Some(up) = &self.up {
                r.moe_weight_scale(up.as_ref(), w, self.rows, self.k);
            }
        }
    }

    fn split(&self, r: &Recorder<'_>, y: &dyn Buffer, out: &Quantized, clamp: Option<f32>) {
        if let Some(up) = &self.up {
            r.silu_mul(
                self.gate.as_ref(),
                up.as_ref(),
                y,
                self.rows * self.k,
                clamp,
            );
        } else {
            r.silu_mul_fused(self.gate.as_ref(), y, self.rows, self.k, clamp);
        }
        r.quant_q8(
            y,
            out.0[0].as_ref(),
            out.0[1].as_ref(),
            out.0[2].as_ref(),
            self.rows,
            self.k,
        );
    }

    fn fused(&self, r: &Recorder<'_>, out: &Quantized, clamp: Option<f32>) {
        assert!(r.can_silu_quant_q8(self.rows, self.k));
        r.silu_mul_quant_q8(
            self.gate.as_ref(),
            self.up.as_deref(),
            out.0[0].as_ref(),
            out.0[1].as_ref(),
            out.0[2].as_ref(),
            self.rows,
            self.k,
            clamp,
        );
    }

    fn guards(&self, be: &VulkanBackend, label: &str) {
        guard(be, self.gate.as_ref(), &format!("{label}/gate_guard"));
        if let Some(up) = &self.up {
            guard(be, up.as_ref(), &format!("{label}/up_guard"));
        }
    }
}

struct RecorderCase {
    split: Projection,
    fused: Projection,
    y: Box<dyn Buffer>,
    baseline: Quantized,
    candidate: Quantized,
}

impl RecorderCase {
    fn new(be: &VulkanBackend, rows: usize, k: usize, combined: bool) -> Self {
        Self {
            split: Projection::workspace(be, rows, k, combined),
            fused: Projection::workspace(be, rows, k, combined),
            y: guarded(be, rows * k * 4, BufferUsage::Activations),
            baseline: Quantized::new(be, rows * k),
            candidate: Quantized::new(be, rows * k),
        }
    }

    fn check(
        &self,
        be: &VulkanBackend,
        source: &Projection,
        w: &dyn Buffer,
        clamp: Option<f32>,
        weight_before: bool,
        label: &str,
    ) -> [Vec<u8>; 3] {
        let mut previous: Option<[Vec<u8>; 3]> = None;
        for repeat in 0..2 {
            self.baseline.reset(be);
            self.candidate.reset(be);
            run(be, |r| {
                self.split.restore(r, source, w, weight_before);
                self.split.split(r, self.y.as_ref(), &self.baseline, clamp);
                self.fused.restore(r, source, w, weight_before);
                self.fused.fused(r, &self.candidate, clamp);
            });
            let label = format!("{label}/repeat{repeat}");
            let expected = self.baseline.read(be, &format!("{label}/split"));
            let actual = self.candidate.read(be, &format!("{label}/fused"));
            for (i, name) in ["QA", "DA", "SA"].into_iter().enumerate() {
                let unit = if i == 0 { 1 } else { 2 };
                exact(&format!("{label}/{name}"), &actual[i], &expected[i], unit);
                if let Some(previous) = &previous {
                    exact(
                        &format!("{label}/stable_{name}"),
                        &expected[i],
                        &previous[i],
                        unit,
                    );
                }
            }
            previous = Some(expected);
            self.split.guards(be, &format!("{label}/split"));
            self.fused.guards(be, &format!("{label}/fused"));
            source.guards(be, &format!("{label}/source"));
            guard(be, self.y.as_ref(), &format!("{label}/F32_guard"));
            guard(be, w, &format!("{label}/weight_guard"));
        }
        previous.unwrap()
    }
}

fn weights(be: &VulkanBackend, rows: usize) -> Box<dyn Buffer> {
    let values: Vec<f32> = (0..rows)
        .map(|i| [0.0, -0.0, 0.125, -0.25, 0.99999994, -1.5, 2.0][(i + 2) % 7])
        .collect();
    upload_bytes(be, bytemuck::cast_slice(&values), BufferUsage::Activations)
}

#[test]
#[ignore = "requires production-eligible Vulkan driver and >=320-MiB SSBO range; run serially"]
fn swiglu_quant_recorder_exact() {
    let be = backend(true);
    let props = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    };
    assert!(props.limits.max_storage_buffer_range as usize >= 40960 * 1024 * 8 + GUARD);
    let mut cases = 0;
    for rows in [1usize, 513, 5120, 40960] {
        for k in [512usize, 640, 1024] {
            let host = input(rows, k, Recipe::Signed);
            let w = weights(&be, rows);
            for combined in [false, true] {
                let source = Projection::source(&be, &host, rows, k, combined);
                let case = RecorderCase::new(&be, rows, k, combined);
                for clamp in [None, Some(7.0f32)] {
                    for weight_before in [false, true] {
                        case.check(&be, &source, w.as_ref(), clamp, weight_before, &format!("recorder/rows{rows}/k{k}/gu{combined}/clamp{clamp:?}/weight_before{weight_before}"));
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 96);
    eprintln!("swiglu_quant recorder: {cases} exact cases, two repeats, embedded production dispatch only");
}

#[test]
#[ignore = "requires production-eligible Vulkan driver; finite exact rounding regressions, serial"]
fn swiglu_quant_rounding_exact() {
    let be = backend(true);
    let (rows, k) = (3usize, 512usize);
    let w = weights(&be, rows);
    let mut cases = 0;
    for recipe in [
        Recipe::Zero,
        Recipe::NearRound,
        Recipe::NearSiluRound,
        Recipe::Extremes,
    ] {
        let host = input(rows, k, recipe);
        let mut separate: [Option<[Vec<u8>; 3]>; 4] = std::array::from_fn(|_| None);
        for combined in [false, true] {
            let source = Projection::source(&be, &host, rows, k, combined);
            let case = RecorderCase::new(&be, rows, k, combined);
            for (ci, clamp) in [None, Some(0.0f32), Some(0.5), Some(7.0)]
                .into_iter()
                .enumerate()
            {
                let reference = case.check(
                    &be,
                    &source,
                    w.as_ref(),
                    clamp,
                    false,
                    &format!("rounding/{recipe:?}/gu{combined}/clamp{clamp:?}"),
                );
                if let Some(separate) = &separate[ci] {
                    for i in 0..3 {
                        exact(
                            &format!("rounding/{recipe:?}/clamp{clamp:?}/layout{i}"),
                            &reference[i],
                            &separate[i],
                            if i == 0 { 1 } else { 2 },
                        );
                    }
                } else {
                    separate[ci] = Some(reference);
                }
                cases += 1;
            }
        }
    }
    // Original SA regression shape: same Signed generator seed, including block80.
    let (rows, k) = (16usize, 640usize);
    let host = input(rows, k, Recipe::Signed);
    let w = weights(&be, rows);
    let mut separate: Option<[Vec<u8>; 3]> = None;
    for combined in [false, true] {
        let source = Projection::source(&be, &host, rows, k, combined);
        let reference = RecorderCase::new(&be, rows, k, combined).check(
            &be,
            &source,
            w.as_ref(),
            None,
            false,
            &format!("sa_float_scale/rows16/k640/gu{combined}"),
        );
        if let Some(separate) = &separate {
            for i in 0..3 {
                exact(
                    &format!("sa_float_scale/layout{i}"),
                    &reference[i],
                    &separate[i],
                    if i == 0 { 1 } else { 2 },
                );
            }
        } else {
            separate = Some(reference);
        }
        cases += 1;
    }
    assert_eq!(cases, 34);
    eprintln!("swiglu_quant rounding: {cases} exact cases, two repeats; no tolerance or CPU SiLU substitute");
}

fn q8_bank(elems: usize, seed: usize) -> Vec<u8> {
    assert_eq!(elems % 32, 0);
    let mut bytes = vec![0u8; elems / 32 * 34];
    for (i, block) in bytes.chunks_exact_mut(34).enumerate() {
        block[..2].copy_from_slice(&0x2800u16.to_le_bytes()); // Exactly 1/32.
        for j in 0..32 {
            block[j + 2] = (((i * 13 + j * 7 + seed * 19) % 31) as i8 - 15) as u8;
        }
    }
    bytes
}

#[test]
#[ignore = "requires production-eligible Vulkan driver; resident MoeFfn config-on/off exact contract"]
fn swiglu_quant_moe_graph_exact() {
    moe_graph_contract(false);
}

#[test]
#[ignore = "requires production-eligible Vulkan driver; invalid graph must reject before output writes"]
fn swiglu_quant_moe_graph_rejects_weight_before_clamp() {
    moe_graph_contract(true);
}

fn moe_graph_contract(expect_rejection: bool) {
    let backends = [backend(false), backend(true)];
    let (rows, ne, experts, used) = (GRAPH_ROWS, GRAPH_NE, 4usize, 2usize);
    assert_eq!(ne % 64, 0, "Down N must contain complete BN64 tiles");
    assert_eq!(ne % 32, 0, "Gate/Up K must contain complete Q8 blocks");
    let mut cases = 0;
    for k in [512usize, 640, 1024] {
        for combined in [false, true] {
            let mut unclamped = [Vec::<Vec<u8>>::new(), Vec::new()];
            for clamp in [None, Some(7.0f32)] {
                let mut weight_after = Vec::new();
                for weight_before in [false, true] {
                    if (weight_before && clamp.is_some()) != expect_rejection {
                        continue;
                    }
                    let label = format!(
                        "graph/k{k}/gu{combined}/clamp{clamp:?}/weight_before{weight_before}"
                    );
                    let mut graph = Graph::new();
                    let x = graph.input(TensorDesc::new([rows, ne], DType::F32));
                    let router = graph.weight(TensorDesc::new([experts, ne], DType::F32));
                    let gate = graph.weight(TensorDesc::new(
                        [experts, k * if combined { 2 } else { 1 }, ne],
                        DType::Q8_0,
                    ));
                    let up = if combined {
                        gate
                    } else {
                        graph.weight(TensorDesc::new([experts, k, ne], DType::Q8_0))
                    };
                    let down = graph.weight(TensorDesc::new([experts, ne, k], DType::Q8_0));
                    let out = graph.output(TensorDesc::new([rows, ne], DType::F32));
                    graph.push(Op::MoeFfn {
                        x,
                        router_x: x,
                        router,
                        gate_exps: gate,
                        up_exps: up,
                        down_exps: down,
                        down_scale: None,
                        fused_gate_up: combined,
                        dst: out,
                        ne: ne as u32,
                        n_expert: experts as u32,
                        n_used: used as u32,
                        n_ff_exp: k as u32,
                        scale: 0.875,
                        act: infr_core::graph::Activation::Silu,
                        swiglu_clamp: clamp,
                        gating: infr_core::graph::MoeGating::Softmax,
                        norm_w: true,
                        weight_before,
                        ep_band: None,
                        exp_probs_b: None,
                        n_expert_groups: 0,
                        n_expert_groups_used: 0,
                        expert_ids: None,
                    });
                    let router_values: Vec<f32> = (0..experts * ne)
                        .map(|i| ((i * 17 % 29) as f32 - 14.0) / 64.0)
                        .collect();
                    let mut banks = vec![
                        (router, bytemuck::cast_slice(&router_values).to_vec()),
                        (
                            gate,
                            q8_bank(experts * k * ne * if combined { 2 } else { 1 }, 1),
                        ),
                        (down, q8_bank(experts * ne * k, 3)),
                    ];
                    if !combined {
                        banks.push((up, q8_bank(experts * k * ne, 2)));
                    }
                    let mut results = Vec::new();
                    for (enabled, be) in backends.iter().enumerate() {
                        let plan = be.compile(&graph).unwrap();
                        let weights: Vec<_> = banks
                            .iter()
                            .map(|(id, bytes)| (*id, upload_bytes(be, bytes, BufferUsage::Weights)))
                            .collect();
                        let xb = guarded(be, rows * ne * 4, BufferUsage::Activations);
                        let yb = guarded(be, rows * ne * 4, BufferUsage::Activations);
                        let mut bind = Bindings::new();
                        bind.bind(x, xb.as_ref());
                        bind.bind(out, yb.as_ref());
                        for (id, b) in &weights {
                            bind.bind(*id, b.as_ref());
                        }
                        let mut repeats = Vec::new();
                        for repeat in 0..2 {
                            let label = format!("{label}/enabled{enabled}/repeat{repeat}");
                            let values: Vec<f32> = (0..rows * ne)
                                .map(|i| (((i * 37 + repeat * 11) % 97) as f32 / 16.0 - 3.0) * 4.0)
                                .collect();
                            be.upload_range(xb.as_ref(), 0, bytemuck::cast_slice(&values))
                                .unwrap();
                            be.upload(yb.as_ref(), &vec![CANARY; yb.len_bytes()])
                                .unwrap();
                            let result = be.execute(plan.as_ref(), &bind);
                            let actual_input =
                                read(be, xb.as_ref(), &format!("{label}/input_guard"));
                            exact(
                                &format!("{label}/input"),
                                &actual_input,
                                bytemuck::cast_slice(&values),
                                4,
                            );
                            let output = read(be, yb.as_ref(), &format!("{label}/output_guard"));
                            if expect_rejection {
                                let error =
                                    result.expect_err("weight_before + Clamp must be rejected");
                                assert!(error.to_string().contains("MoeFfn swiglu_clamp needs a gated SiLU/GELU with output-side weighting"), "{label}: unexpected rejection: {error}");
                                exact(
                                    &format!("{label}/rejected_output_unchanged"),
                                    &output,
                                    &vec![CANARY; rows * ne * 4],
                                    1,
                                );
                                continue;
                            }
                            result.unwrap();
                            assert!(
                                output
                                    .chunks_exact(4)
                                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                                "{label}: nonfinite graph output"
                            );
                            assert!(
                                output
                                    .chunks_exact(4)
                                    .any(|b| f32::from_le_bytes(b.try_into().unwrap()) != 0.0),
                                "{label}: nontrivial graph output required"
                            );
                            repeats.push(output);
                        }
                        if !expect_rejection {
                            assert!(
                                repeats[0] != repeats[1],
                                "{label}: second execute must consume changed input"
                            );
                        }
                        for ((_, b), (_, original)) in weights.iter().zip(&banks) {
                            let actual = read(be, b.as_ref(), &format!("{label}/weight_guard"));
                            exact(&format!("{label}/weight_unchanged"), &actual, original, 1);
                        }
                        if expect_rejection {
                            assert!(repeats.is_empty());
                        } else {
                            results.push(repeats);
                        }
                    }
                    if expect_rejection {
                        assert!(results.is_empty());
                        cases += 1;
                        continue;
                    }
                    for repeat in 0..2 {
                        exact(
                            &format!("{label}/repeat{repeat}/output"),
                            &results[1][repeat],
                            &results[0][repeat],
                            4,
                        );
                    }
                    if weight_before {
                        assert!(
                            results[0] != weight_after,
                            "{label}: weight_before control must change the real graph result"
                        );
                    } else {
                        weight_after = results[0].clone();
                    }
                    if clamp.is_some() {
                        assert!(
                            results[0] != unclamped[usize::from(weight_before)],
                            "{label}: Clamp control must change the real graph result"
                        );
                    } else {
                        unclamped[usize::from(weight_before)] = results[0].clone();
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, if expect_rejection { 6 } else { 18 });
    eprintln!("swiglu_quant resident graph: {cases} cases; rejection={expect_rejection}; two executions per config/plan");
}

#[test]
fn swiglu_quant_recipe_and_geometry_contract() {
    assert_eq!(GRAPH_NE % 64, 0);
    assert_eq!(GRAPH_NE % 32, 0);
    assert!(GRAPH_ROWS > 8);
    for rows in [1usize, 513, 5120, 40960] {
        for k in [512usize, 640, 1024] {
            let groups = rows * (k / 32);
            let gy = groups.div_ceil(65535);
            let gx = groups.div_ceil(gy);
            assert!(gx <= 65535 && gx * gy >= groups && gx * gy - groups < gy);
            assert!(rows * k * 2 <= u32::MAX as usize);
        }
    }
    let near = input(3, 512, Recipe::NearRound);
    for (i, &u) in near.up.iter().enumerate().filter(|(i, _)| i % 32 != 0) {
        let lane = i % 32;
        let step = [1.0 / 128.0, 1.0, 1.0 / 65536.0][(i / 32) % 3];
        let center = ((lane as f32 * 7.0) % 126.0 + 0.5) * step;
        let expected = match (i / 32 + lane) % 3 {
            0 => center.to_bits() - 1,
            1 => center.to_bits(),
            _ => center.to_bits() + 1,
        };
        assert_eq!(
            (32.0 * u).abs().to_bits(),
            expected,
            "half-step +/-1ULP control"
        );
    }
    let zero = input(1, 512, Recipe::Zero);
    assert_eq!(zero.gate[0].to_bits(), 0);
    assert_eq!(zero.gate[1].to_bits(), 0x80000000);
    assert_eq!(zero.up[0].to_bits(), 0x80000000);
    for recipe in [
        Recipe::Signed,
        Recipe::Zero,
        Recipe::NearRound,
        Recipe::NearSiluRound,
        Recipe::Extremes,
    ] {
        let values = input(3, 512, recipe);
        assert!(values.gate.iter().chain(&values.up).all(|x| x.is_finite()));
    }
}
