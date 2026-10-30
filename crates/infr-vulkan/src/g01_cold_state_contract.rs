#![cfg(test)]

//! Parent-serialized Unit8 rotating-state contract for the production Recorder.
//! Production embedded kernels/APIs only; no external SPVs or hand-packed G01 PC.
//! v1 MAY FAIL the strict output contract. No tolerant fallback or timing qualification.
use super::*;
use infr_core::backend::{Bindings, BufferUsage, Plan};
use infr_core::graph::{Activation, Graph, Op};
use infr_core::tensor::{DType, TensorDesc};
use infr_core::TensorId;
use std::sync::Arc;
use std::time::Instant;

const NK: usize = 16;
const HD: usize = 128;
const TAPE: usize = 4;
const ROUNDS: usize = 5;
const GUARD_WORDS: usize = 64;
const CANARY: u32 = 0xc5f3_5555;
const POISON: u32 = 0x7fc0_1234;
// AMD lists 96 MB of Infinity Cache on RX7900XTX. Compare against the stricter
// 96 MiB threshold; this is NOT an L2-size claim or a cache-miss guarantee.
const CACHE_EXPOSURE_THRESHOLD: usize = 96 * 1024 * 1024;

fn bank_count(nv: usize) -> usize {
    match nv {
        32 => 64,
        48 | 64 => 48,
        _ => panic!("unsupported G01 rotating-state head count: {nv}"),
    }
}

fn planned_owned_bytes(nv: usize) -> usize {
    let cc = (2 * NK + nv) * HD;
    2 * bank_count(nv) * (nv * HD * HD * 4 + 48 * cc + 2600 * nv + 1536)
}

fn device_uuid(be: &VulkanBackend) -> [u8; vk::UUID_SIZE] {
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    unsafe {
        be.shared
            .instance
            .get_physical_device_properties2(be.shared.physical_device, &mut properties);
    }
    assert_ne!(
        id.device_uuid,
        [0; vk::UUID_SIZE],
        "device UUID unavailable"
    );
    id.device_uuid
}

fn signal(n: usize, mut seed: u32, amplitude: f32) -> Vec<f32> {
    assert_ne!(seed, 0);
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * amplitude
        })
        .collect()
}

fn seed(nv: usize, bank: usize, tag: u32) -> u32 {
    0x6d21_a9b7u32
        .wrapping_add(nv as u32 * 0x1_0101)
        .wrapping_add(bank as u32 * 0x11_1735)
        .wrapping_add(tag * 0x10_2071)
}

struct Guard {
    buffer: Box<dyn Buffer>,
    words: usize,
}

impl Guard {
    fn new(be: &VulkanBackend, words: usize) -> Self {
        let buffer = be
            .alloc((words + GUARD_WORDS) * 4, BufferUsage::Activations)
            .expect("G01 guarded allocation; resource failure is not a skip");
        let this = Self { buffer, words };
        this.write_bits(be, &vec![POISON; words]);
        this
    }
    fn write_bits(&self, be: &VulkanBackend, values: &[u32]) {
        assert_eq!(values.len(), self.words);
        let mut all = values.to_vec();
        all.extend(std::iter::repeat_n(CANARY, GUARD_WORDS));
        be.upload(self.buffer.as_ref(), bytemuck::cast_slice(&all))
            .unwrap();
    }
    fn write(&self, be: &VulkanBackend, values: &[f32]) {
        self.write_bits(be, bytemuck::cast_slice(values));
    }
    fn capture(&self, be: &VulkanBackend, label: &str) -> Vec<u32> {
        let mut all = vec![0u32; self.words + GUARD_WORDS];
        be.download(self.buffer.as_ref(), bytemuck::cast_slice_mut(&mut all))
            .unwrap();
        assert!(
            all[self.words..].iter().all(|&v| v == CANARY),
            "{label}: suffix canary overwritten"
        );
        all.truncate(self.words);
        assert!(
            all.iter().all(|&v| f32::from_bits(v).is_finite()),
            "{label}: nonfinite or unwritten payload"
        );
        assert!(
            all.iter().any(|&v| f32::from_bits(v) != 0.0),
            "{label}: vacuous all-zero payload"
        );
        all
    }
}

fn upload(be: &VulkanBackend, values: &[f32], usage: BufferUsage) -> Box<dyn Buffer> {
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v != 0.0));
    let b = be.alloc(values.len() * 4, usage).unwrap();
    be.upload(b.as_ref(), bytemuck::cast_slice(values)).unwrap();
    b
}

struct Input {
    x: Box<dyn Buffer>,
    b: Box<dyn Buffer>,
    a: Box<dyn Buffer>,
    gate: Box<dyn Buffer>,
}

struct Bank {
    nv: usize,
    index: usize,
    tape: Vec<Input>,
    cw: Box<dyn Buffer>,
    ac: Box<dyn Buffer>,
    dt: Box<dyn Buffer>,
    norm: Box<dyn Buffer>,
    state: Guard,
    history: Guard,
    conv: Guard,
    out: Guard,
}

impl Bank {
    fn new(be: &VulkanBackend, nv: usize, index: usize) -> Self {
        let cc = (2 * NK + nv) * HD;
        let tape = (0..TAPE)
            .map(|step| {
                let tag = 10 + step as u32 * 4;
                let amplitude = [1e-4, 0.7, 1.2, 4.0][step];
                Input {
                    x: upload(
                        be,
                        &signal(cc, seed(nv, index, tag), amplitude),
                        BufferUsage::Activations,
                    ),
                    b: upload(
                        be,
                        &signal(nv, seed(nv, index, tag + 1), 1.0),
                        BufferUsage::Activations,
                    ),
                    a: upload(
                        be,
                        &signal(nv, seed(nv, index, tag + 2), 1.0),
                        BufferUsage::Activations,
                    ),
                    gate: upload(
                        be,
                        &signal(nv * HD, seed(nv, index, tag + 3), 4.0),
                        BufferUsage::Activations,
                    ),
                }
            })
            .collect();
        let ac: Vec<f32> = signal(nv, seed(nv, index, 2), 0.2)
            .into_iter()
            .map(|x| -0.2 + x)
            .collect();
        let norm: Vec<f32> = signal(HD, seed(nv, index, 4), 0.3)
            .into_iter()
            .map(|x| 1.0 + x)
            .collect();
        let bank = Self {
            nv,
            index,
            tape,
            cw: upload(
                be,
                &signal(cc * 4, seed(nv, index, 1), 0.5),
                BufferUsage::Weights,
            ),
            ac: upload(be, &ac, BufferUsage::Weights),
            dt: upload(
                be,
                &signal(nv, seed(nv, index, 3), 0.4),
                BufferUsage::Weights,
            ),
            norm: upload(be, &norm, BufferUsage::Weights),
            state: Guard::new(be, nv * HD * HD),
            history: Guard::new(be, 3 * cc),
            conv: Guard::new(be, cc),
            out: Guard::new(be, nv * HD),
        };
        bank.reset(be);
        bank
    }
    fn reset(&self, be: &VulkanBackend) {
        self.state.write(
            be,
            &signal(self.state.words, seed(self.nv, self.index, 5), 0.02),
        );
        self.history.write(
            be,
            &signal(self.history.words, seed(self.nv, self.index, 6), 0.8),
        );
        self.conv.write_bits(be, &vec![POISON; self.conv.words]);
        self.out.write_bits(be, &vec![POISON; self.out.words]);
    }
    fn bytes(&self) -> usize {
        self.tape
            .iter()
            .map(|t| t.x.len_bytes() + t.b.len_bytes() + t.a.len_bytes() + t.gate.len_bytes())
            .sum::<usize>()
            + self.cw.len_bytes()
            + self.ac.len_bytes()
            + self.dt.len_bytes()
            + self.norm.len_bytes()
            + self.state.buffer.len_bytes()
            + self.history.buffer.len_bytes()
            + self.conv.buffer.len_bytes()
            + self.out.buffer.len_bytes()
    }
}

struct Ids {
    x: TensorId,
    cw: TensorId,
    history: TensorId,
    b: TensorId,
    a: TensorId,
    ac: TensorId,
    dt: TensorId,
    state: TensorId,
    gate: TensorId,
    norm: TensorId,
    out: TensorId,
}

fn graph(nv: usize, sigmoid: bool) -> (Graph, Ids) {
    let mut g = Graph::new();
    let cc = (2 * NK + nv) * HD;
    let desc = |n| TensorDesc::new(vec![n], DType::F32);
    let x = g.input(desc(cc));
    let cw = g.weight(desc(cc * 4));
    let history = g.input(desc(cc * 3));
    let b = g.input(desc(nv));
    let a = g.input(desc(nv));
    let ac = g.weight(desc(nv));
    let dt = g.weight(desc(nv));
    let state = g.input(TensorDesc::new(vec![nv, HD, HD], DType::F32));
    let gate = g.input(desc(nv * HD));
    let norm = g.weight(desc(HD));
    let conv = g.internal(desc(cc));
    let normal = g.internal(desc(nv * HD));
    let out = g.output(desc(nv * HD));
    g.push(Op::Conv1dSilu {
        x,
        weight: cw,
        state: history,
        state_trace: None,
        dst: conv,
        rows: 1,
        channels: cc as u32,
        kernel: 4,
    });
    g.push(Op::DeltaNet {
        q: conv,
        k: conv,
        v: conv,
        b,
        a,
        a_coef: ac,
        dt_bias: dt,
        state,
        state_trace: None,
        dst: normal,
        rows: 1,
        n_vhead: nv as u32,
        n_khead: NK as u32,
        head_k: HD as u32,
        head_v: HD as u32,
        eps: 1e-6,
        src_stride: 0,
    });
    g.push(Op::GatedRmsNorm {
        x: normal,
        weight: norm,
        gate,
        dst: normal,
        rows: 1,
        n_head: nv as u32,
        head_dim: HD as u32,
        eps: 1e-6,
        act: if sigmoid {
            Activation::Sigmoid
        } else {
            Activation::Silu
        },
    });
    // Expose only the small norm output. Persistent state is inspected directly;
    // do not add a full-state Copy to the timed Graph or its working set.
    g.push(Op::Copy {
        src: normal,
        src_off: 0,
        dst: out,
        dst_off: 0,
        n: (nv * HD) as u32,
    });
    (
        g,
        Ids {
            x,
            cw,
            history,
            b,
            a,
            ac,
            dt,
            state,
            gate,
            norm,
            out,
        },
    )
}

struct Leg {
    be: VulkanBackend,
    plan: Box<dyn Plan>,
    banks: Vec<Bank>,
    step: usize,
}

impl Leg {
    fn new(nv: usize, fused: bool, g: &Graph) -> Self {
        let mut cfg = (*VulkanBackend::cfg_from_env().unwrap()).clone();
        cfg.kernels.vulkan.delta_fused_norm = fused;
        cfg.kernels.vulkan.delta_strided = true;
        cfg.kernels.vulkan.pipeline_cache_disk = false;
        cfg.kernels.gated_rmsnorm = true;
        cfg.spec.mtp = false;
        cfg.device.submit_dispatches = Some(0);
        let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        assert!(
            be.prefers_deltanet_fused_norm(),
            "requires production Windows discrete RDNA3 gate"
        );
        assert!(
            be.caps().name.contains("7900 XTX"),
            "bounded cache contract is qualified only for RX7900XTX; parent must select it"
        );
        let plan = be.compile(g).unwrap();
        let banks = (0..bank_count(nv)).map(|i| Bank::new(&be, nv, i)).collect();
        Self {
            be,
            plan,
            banks,
            step: 0,
        }
    }
    fn reset(&mut self) {
        for bank in &self.banks {
            bank.reset(&self.be);
        }
        self.step = 0;
    }
    fn run_graph(&self, ids: &Ids, bank: usize, step: usize) {
        let b = &self.banks[bank];
        let input = &b.tape[step % TAPE];
        let mut binds = Bindings::new();
        for (id, buf) in [
            (ids.x, input.x.as_ref()),
            (ids.cw, b.cw.as_ref()),
            (ids.history, b.history.buffer.as_ref()),
            (ids.b, input.b.as_ref()),
            (ids.a, input.a.as_ref()),
            (ids.ac, b.ac.as_ref()),
            (ids.dt, b.dt.as_ref()),
            (ids.state, b.state.buffer.as_ref()),
            (ids.gate, input.gate.as_ref()),
            (ids.norm, b.norm.as_ref()),
            (ids.out, b.out.buffer.as_ref()),
        ] {
            binds.bind(id, buf);
        }
        self.be.execute(self.plan.as_ref(), &binds).unwrap();
    }
    fn record(&self, r: &Recorder<'_>, bank: usize, step: usize, fused: bool, sigmoid: bool) {
        let b = &self.banks[bank];
        let input = &b.tape[step % TAPE];
        let cc = (2 * NK + b.nv) * HD;
        r.conv1d_silu(
            input.x.as_ref(),
            b.cw.as_ref(),
            b.history.buffer.as_ref(),
            b.conv.buffer.as_ref(),
            1,
            cc,
            4,
        );
        if fused {
            // Production API packs the v2b PC20 ABI; do not hand-pack push words
            // or load a standalone replacement SPV here.
            r.deltanet_gated_norm(
                b.conv.buffer.as_ref(),
                input.b.as_ref(),
                input.a.as_ref(),
                b.ac.as_ref(),
                b.dt.as_ref(),
                b.norm.as_ref(),
                input.gate.as_ref(),
                b.state.buffer.as_ref(),
                b.out.buffer.as_ref(),
                b.nv,
                NK,
                1e-6,
                1e-6,
                sigmoid,
            );
        } else {
            r.deltanet_strided(
                b.conv.buffer.as_ref(),
                b.conv.buffer.as_ref(),
                b.conv.buffer.as_ref(),
                input.b.as_ref(),
                input.a.as_ref(),
                b.ac.as_ref(),
                b.dt.as_ref(),
                b.state.buffer.as_ref(),
                b.out.buffer.as_ref(),
                1,
                b.nv,
                NK,
                HD,
                HD,
                1e-6,
                cc,
            );
            if sigmoid {
                r.rmsnorm_gate_sigmoid(
                    b.out.buffer.as_ref(),
                    b.norm.as_ref(),
                    input.gate.as_ref(),
                    b.out.buffer.as_ref(),
                    b.nv,
                    HD,
                    1e-6,
                );
            } else {
                r.rmsnorm_gate(
                    b.out.buffer.as_ref(),
                    b.norm.as_ref(),
                    input.gate.as_ref(),
                    b.out.buffer.as_ref(),
                    b.nv,
                    HD,
                    1e-6,
                );
            }
        }
    }
    fn record_step(&self, bank: usize, step: usize, fused: bool, sigmoid: bool) {
        let r = self.be.recorder().unwrap();
        self.record(&r, bank, step, fused, sigmoid);
        r.finish().unwrap();
    }
    fn record_sweep(&self, r: &Recorder<'_>, active: usize, fused: bool, sigmoid: bool) {
        let calls = self.sweep_calls(active);
        for i in 0..calls {
            self.record(r, i % active, self.step + i / active, fused, sigmoid);
            // Distinct banks have no data dependency. Force a real module
            // boundary so this is serial latency, not independent bank
            // modules accidentally overlapping through the command processor.
            r.global_barrier();
        }
    }
    fn graph_sweep(&self, ids: &Ids, active: usize) {
        for i in 0..self.sweep_calls(active) {
            self.run_graph(ids, i % active, self.step + i / active);
        }
    }
    fn sweep_calls(&self, active: usize) -> usize {
        let calls = self.banks.len();
        assert!(active > 0 && active <= calls, "invalid active-bank range");
        assert_eq!(calls % active, 0, "incomplete bank sweep");
        calls
    }
    fn snapshot(&self, active: usize) -> Snapshot {
        self.sweep_calls(active);
        Snapshot(
            (0..active)
                .map(|i| {
                    let b = &self.banks[i];
                    [
                        b.state.capture(&self.be, "state"),
                        b.history.capture(&self.be, "Conv history"),
                        b.out.capture(&self.be, "norm output"),
                    ]
                })
                .collect(),
        )
    }
}

struct Snapshot(Vec<[Vec<u32>; 3]>);

fn exact(a: &Snapshot, b: &Snapshot, where_: &str) {
    assert_eq!(a.0.len(), b.0.len());
    for (bank, (aa, bb)) in a.0.iter().zip(&b.0).enumerate() {
        for (slot, label) in ["persistent state", "Conv history", "final output"]
            .iter()
            .enumerate()
        {
            assert_eq!(aa[slot].len(), bb[slot].len());
            let count = aa[slot]
                .iter()
                .zip(&bb[slot])
                .filter(|(x, y)| x != y)
                .count();
            let first = aa[slot]
                .iter()
                .zip(&bb[slot])
                .enumerate()
                .find(|(_, (x, y))| x != y)
                .map(|(i, (&x, &y))| (i, x, y));
            assert_eq!(count,0,"{where_}: bank={bank} {label}, first differing word={first:?}; STRICT F32 bits including signed zero; do not loosen for v1/v2");
        }
    }
}

fn routes(off: &Leg, on: &Leg, sigmoid: bool) {
    let split = off.be.built_kernel_names();
    let fused = on.be.built_kernel_names();
    assert!(
        split.contains(&"conv1d_silu") && fused.contains(&"conv1d_silu"),
        "both actual Graph routes must execute the same Conv baseline"
    );
    let norm = if sigmoid {
        "rmsnorm_gate_sigmoid"
    } else {
        "rmsnorm_gate"
    };
    let selected = if sigmoid {
        "deltanet_fused_norm_sigmoid"
    } else {
        "deltanet_fused_norm"
    };
    assert!(
        split.contains(&"deltanet_strided") && split.contains(&norm),
        "actual Graph baseline missing: {split:?}"
    );
    assert!(
        !split.iter().any(|n| n.starts_with("deltanet_fused_norm")),
        "off Graph fused unexpectedly"
    );
    assert!(
        fused.contains(&selected),
        "actual Graph did not select {selected}: {fused:?}"
    );
    {
        let kernels = on.be.shared.kernels.lock().unwrap();
        let kernel = kernels.get(selected).expect("actual fused Graph pipeline");
        assert_eq!(kernel.push_size, 20, "G01v2b fixture requires PC20");
        assert_eq!(kernel.n_buf, 9, "G01v2b descriptor count changed");
    }
    assert!(
        !fused.contains(&"deltanet_strided") && !fused.contains(&norm),
        "on Graph kept split work: {fused:?}"
    );
    assert!(
        !split
            .iter()
            .chain(&fused)
            .any(|n| n.starts_with("deltanet_fused_chain_r")),
        "unselected full-Conv prototype must not run"
    );
}

fn strict_graph_and_recorder(off: &mut Leg, on: &mut Leg, ids: &Ids, sigmoid: bool) {
    let banks = off.banks.len();
    assert_eq!(banks, on.banks.len());
    off.reset();
    on.reset();
    // Route proof happens BEFORE any direct Recorder control can populate the
    // kernel cache and falsely satisfy a positive or negative Graph assertion.
    for step in 0..TAPE {
        for bank in 0..banks {
            off.run_graph(ids, bank, step);
            on.run_graph(ids, bank, step);
        }
        routes(off, on, sigmoid);
        exact(
            &off.snapshot(banks),
            &on.snapshot(banks),
            "actual Graph off/on each input step",
        );
    }
    // No extra state-bank copies are needed for cross-route proof: temporarily
    // use the other leg's buffers with an explicit direct route. Later assertions
    // never treat these deliberately loaded control kernels as Graph route proof.
    for fused in [false, true] {
        off.reset();
        on.reset();
        for step in 0..TAPE {
            for bank in 0..banks {
                if fused {
                    on.run_graph(ids, bank, step);
                    off.record_step(bank, step, true, sigmoid);
                } else {
                    off.run_graph(ids, bank, step);
                    on.record_step(bank, step, false, sigmoid);
                }
            }
            exact(
                &off.snapshot(banks),
                &on.snapshot(banks),
                "same-variant Graph/Recorder each input step",
            );
        }
    }
    off.reset();
    on.reset();
}

struct QueryTimer<'a> {
    be: &'a VulkanBackend,
    pool: vk::QueryPool,
}
impl Drop for QueryTimer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.be.shared.device.destroy_query_pool(self.pool, None);
        }
    }
}

fn device_sweep(leg: &Leg, active: usize, fused: bool, sigmoid: bool) -> f64 {
    let calls = leg.sweep_calls(active);
    let device = &leg.be.shared.device;
    let pool = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )
    }
    .unwrap();
    let timer = QueryTimer { be: &leg.be, pool };
    let r = leg.be.recorder().unwrap();
    unsafe {
        device.cmd_reset_query_pool(r.cmd, timer.pool, 0, 2);
        device.cmd_write_timestamp(r.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, timer.pool, 0);
    }
    leg.record_sweep(&r, active, fused, sigmoid);
    unsafe {
        device.cmd_write_timestamp(r.cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timer.pool, 1);
    }
    r.finish().unwrap();
    let mut ticks = [0u64; 2];
    unsafe {
        device.get_query_pool_results(
            timer.pool,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    }
    .unwrap();
    assert!(ticks[1] > ticks[0], "invalid device timestamp interval");
    let us = (ticks[1] - ticks[0]) as f64 * leg.be.shared.submit_timestamp_period_ns as f64
        / 1000.0
        / calls as f64;
    assert!(us.is_finite() && us > 0.0);
    us
}

fn sample(
    leg: &mut Leg,
    ids: &Ids,
    active: usize,
    fused: bool,
    sigmoid: bool,
    graph_wall: bool,
) -> (f64, Snapshot) {
    let calls = leg.sweep_calls(active);
    // No reset/upload/readback between the priming sweep and the timed sweep.
    // Rotating banks therefore have one full bank-set reuse distance. Warm mode
    // executes bank0 repeatedly, but cycles four nonblank token inputs.
    if graph_wall {
        leg.graph_sweep(ids, active);
    } else {
        let r = leg.be.recorder().unwrap();
        leg.record_sweep(&r, active, fused, sigmoid);
        r.finish().unwrap();
    }
    leg.step += calls / active;
    let us = if graph_wall {
        let start = Instant::now();
        leg.graph_sweep(ids, active);
        start.elapsed().as_secs_f64() * 1e6 / calls as f64
    } else {
        device_sweep(leg, active, fused, sigmoid)
    };
    leg.step += calls / active;
    // These downloads happen AFTER timing and before the next sample's complete
    // priming sweep. At most two snapshots are held; no GPU state capture copies.
    (us, leg.snapshot(active))
}

fn median(v: &[f64]) -> f64 {
    let mut a = v.to_vec();
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}

fn paired(
    off: &mut Leg,
    on: &mut Leg,
    ids: &Ids,
    nv: usize,
    sigmoid: bool,
    active: usize,
    graph_wall: bool,
) {
    assert_eq!(off.banks.len(), on.banks.len());
    assert_eq!(off.sweep_calls(active), on.sweep_calls(active));
    off.reset();
    on.reset();
    let level = if graph_wall {
        "Graph_wall_with_output_Copy"
    } else {
        "Recorder_GPU_Conv_recurrence_norm"
    };
    let working_set = if active == 1 {
        "warm1".to_owned()
    } else {
        format!("rotating{active}")
    };
    let mut ratios = Vec::new();
    for round in 0..ROUNDS {
        let order = if round % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        };
        let mut times = [Vec::new(), Vec::new()];
        let mut pending: [Option<Snapshot>; 2] = [None, None];
        for fused in order {
            let (us, snapshot) = sample(
                if fused { &mut *on } else { &mut *off },
                ids,
                active,
                fused,
                sigmoid,
                graph_wall,
            );
            let k = usize::from(fused);
            let ordinal = times[k].len();
            times[k].push(us);
            if let Some(other) = pending[ordinal].take() {
                exact(
                    &other,
                    &snapshot,
                    "paired same-ordinal sample after priming+timed sweeps",
                );
            } else {
                pending[ordinal] = Some(snapshot);
            }
        }
        assert!(pending.iter().all(Option::is_none));
        let baseline = (times[0][0] + times[0][1]) * 0.5;
        let candidate = (times[1][0] + times[1][1]) * 0.5;
        let ratio = candidate / baseline;
        ratios.push(ratio);
        eprintln!("G01COLD ROUND nv={nv} sigmoid={sigmoid} working_set={working_set} level={level} round={round} order={} off_us={:?} on_us={:?} paired_time_ratio={ratio:.6} strict_bytes=true",if round%2==0 {"ABBA"}else{"BAAB"},times[0],times[1]);
    }
    eprintln!("G01COLD SUMMARY nv={nv} sigmoid={sigmoid} working_set={working_set} level={level} rounds={ROUNDS} median_paired_time_ratio={:.6} ratios={ratios:?} strict_bytes=true",median(&ratios));
}

#[test]
fn g01_cold_state_fixture_cpu_contract() {
    for nv in [32usize, 48, 64] {
        let banks = bank_count(nv);
        assert_eq!(banks, if nv == 32 { 64 } else { 48 });
        assert_eq!(banks % TAPE, 0);
        let per_leg = banks * nv * HD * HD * 4;
        assert!(per_leg > CACHE_EXPOSURE_THRESHOLD);
        assert_eq!(
            per_leg / (1024 * 1024),
            match nv {
                32 => 128,
                48 => 144,
                64 => 192,
                _ => unreachable!(),
            }
        );
        for active in [1, banks] {
            let mut visits = vec![0usize; active];
            let mut steps = vec![Vec::new(); active];
            for sweep in 0..2 {
                let base = sweep * banks / active;
                for i in 0..banks {
                    visits[i % active] += 1;
                    steps[i % active].push(base + i / active);
                }
            }
            for bank in 0..active {
                assert_eq!(visits[bank], 2 * banks / active);
                assert_eq!(steps[bank], (0..2 * banks / active).collect::<Vec<_>>());
            }
        }
        assert!(planned_owned_bytes(nv) < 512 * 1024 * 1024);
        for sigmoid in [false, true] {
            let (g, _) = graph(nv, sigmoid);
            assert_eq!(g.ops.len(), 4);
            assert!(!g.independent_rows && !g.mtp_verify);
            assert!(matches!(g.ops[0], Op::Conv1dSilu { rows: 1, .. }));
            assert!(matches!(
                g.ops[1],
                Op::DeltaNet {
                    rows: 1,
                    n_khead: 16,
                    head_k: 128,
                    head_v: 128,
                    state_trace: None,
                    ..
                }
            ));
            assert!(matches!(
                g.ops[2],
                Op::GatedRmsNorm {
                    rows: 1,
                    head_dim: 128,
                    ..
                }
            ));
        }
        let a = signal(1024, seed(nv, 0, 10), 0.7);
        let b = signal(1024, seed(nv, 1, 10), 0.7);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_ne!(a, b);
    }
}

#[test]
#[ignore = "parent Unit8 serial RX7900XTX only: strict Graph/Recorder state+output bytes then warm/rotating ABBA; v1 may FAIL"]
fn g01_cold_state_graph_recorder_exact_and_paired() {
    let mut completed = 0;
    for nv in [32usize, 48, 64] {
        for sigmoid in [false, true] {
            let (g, ids) = graph(nv, sigmoid);
            assert!(planned_owned_bytes(nv) < 512 * 1024 * 1024);
            let mut off = Leg::new(nv, false, &g);
            let mut on = Leg::new(nv, true, &g);
            assert_eq!(
                device_uuid(&off.be),
                device_uuid(&on.be),
                "A/B legs selected different physical GPUs"
            );
            let owned = off
                .banks
                .iter()
                .chain(&on.banks)
                .map(Bank::bytes)
                .sum::<usize>();
            assert_eq!(owned, planned_owned_bytes(nv));
            assert!(
                owned < 512 * 1024 * 1024,
                "fixture exceeded its explicit owned-buffer bound"
            );
            let banks = bank_count(nv);
            assert_eq!(off.banks.len(), banks);
            assert_eq!(on.banks.len(), banks);
            let calls = off.sweep_calls(banks);
            let per_leg = banks * nv * HD * HD * 4;
            assert!(per_leg > CACHE_EXPOSURE_THRESHOLD);
            eprintln!("G01COLD SETUP nv={nv} sigmoid={sigmoid} banks_per_leg={banks} state_bytes_per_leg={per_leg} owned_gpu_bytes={owned} cache_exposure_threshold_bytes={CACHE_EXPOSURE_THRESHOLD} cache_residency_measured=false calls_per_bracket={calls} input_tape={TAPE} v2b_pc20=compiled_production_API no_external_SPV=true");
            strict_graph_and_recorder(&mut off, &mut on, &ids, sigmoid);
            for active in [1, banks] {
                for graph_wall in [false, true] {
                    paired(&mut off, &mut on, &ids, nv, sigmoid, active, graph_wall);
                }
            }
            completed += 1;
        }
    }
    assert_eq!(
        completed, 6,
        "all head counts and both gates are required; no silent skip"
    );
}
