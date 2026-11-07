#![cfg(test)]

//! PR146 9e7ddc1e/28e87078/165fd0dc integration regressions, mounted under adapter.
//! CPU tests are policy/table contracts; ignored tests execute real Graphs, not simulated dispatch.
//! Run GPU cases serially with explicit MOE4ALL_TEST_DEVICE=VulkanN. Pipeline disk cache is off.
//! Task: 20261102-095500-pr146-three-ports-6b91. These tests write no fixtures or evidence files.

use super::*;
use infr_core::{graph::MoeGating, DType, TensorDesc};

const GUARD: usize = 64;
const POISON: f32 = -7777.25;

#[test]
fn pr146_router_shape_gate_boundaries_and_fallbacks_cpu() {
    for (dt, threshold) in [(DType::Bf16, 256), (DType::Q8_0, 512)] {
        for k in [32, 64, 2560] {
            for n in [64, 128, 512] {
                for rows in [0, 1, 16, 65, threshold - 1, threshold, threshold + 3, 4096] {
                    assert_eq!(
                        moe_router_gemm_shape_applies(dt, rows, k, n),
                        rows >= threshold,
                        "{dt:?} rows={rows} K={k} experts={n}"
                    );
                }
            }
        }
        for (k, n) in [(0, 64), (31, 64), (33, 64), (64, 0), (64, 63), (64, 65)] {
            assert!(!moe_router_gemm_shape_applies(dt, 4096, k, n));
        }
    }
    // F32 routers must stay on their original F32 path, even at a large prefill M.
    for dt in [DType::F32, DType::F16, DType::Q4K, DType::Q6K, DType::Iq3S] {
        for rows in [1, 256, 512, 4096] {
            assert!(!moe_router_gemm_shape_applies(dt, rows, 2560, 512));
        }
    }
}

#[test]
fn pr146_router_logits_padding_tracks_qualified_rows_cpu() {
    for (dt, rows, expected) in [
        (DType::Bf16, 255, 255),
        (DType::Bf16, 256, 256),
        (DType::Bf16, 257, 320),
        (DType::Bf16, 319, 320),
        (DType::Q8_0, 511, 511),
        (DType::Q8_0, 512, 512),
        (DType::Q8_0, 513, 576),
        (DType::F32, 513, 513),
    ] {
        let admitted = moe_router_gemm_shape_applies(dt, rows, 64, 128);
        assert_eq!(moe_router_logits_rows(rows, admitted), expected);
        assert_eq!(moe_router_logits_rows(rows, false), rows);
        if admitted {
            assert_eq!(expected % 64, 0);
            assert!(expected >= rows && expected - rows < 64);
        }
    }
}

#[test]
fn pr146_bf16_ag_tables_have_distinct_spirv_and_cache_keys_cpu() {
    let names = [
        crate::gemm::native_gemm_warp_ag_kernel_name(DType::Bf16).unwrap(),
        crate::gemm::native_gemm_warp_n128_ag_kernel_name(DType::Bf16).unwrap(),
        crate::gemm::native_gemm_warp_n64_ag_kernel_name(DType::Bf16).unwrap(),
    ];
    assert_eq!(
        names,
        [
            "native_gemm_warp_bf16_ag",
            "native_gemm_warp_bf16_n128_ag",
            "native_gemm_warp_bf16_n64_ag",
        ]
    );
    let compiled: Vec<_> = names
        .iter()
        .map(|name| crate::gemm::native_gemm_spv(name))
        .collect();
    for (key, spv) in &compiled {
        assert_eq!(spv.first(), Some(&0x0723_0203), "{key}: invalid SPIR-V");
        assert!(spv.len() > 5, "{key}: missing shader body");
    }
    for i in 0..compiled.len() {
        for j in i + 1..compiled.len() {
            assert_ne!(compiled[i].0, compiled[j].0, "tile pipeline keys alias");
            assert_ne!(compiled[i].1, compiled[j].1, "tile shader bodies alias");
        }
    }
    assert!(crate::gemm::native_gemm_warp_ag_kernel_name(DType::F32).is_none());
    assert!(crate::gemm::native_gemm_warp_n64_ag_kernel_name(DType::F32).is_none());
}

fn backend(warp: bool, small_m: bool, wide_tile: bool) -> VulkanBackend {
    let selected = std::env::var("MOE4ALL_TEST_DEVICE")
        .expect("explicit MOE4ALL_TEST_DEVICE=VulkanN required");
    assert_eq!(selected, "Vulkan1", "explicit idle XTX selection required");
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(selected);
    cfg.kernels.vulkan.pipeline_cache_disk = false;
    cfg.kernels.vulkan.gemm_warp = warp;
    cfg.kernels.vulkan.gemm_direct_a = false;
    cfg.kernels.vulkan.gemm_wide_tile = wide_tile;
    cfg.kernels.vulkan.bf16_coopmat = false;
    cfg.kernels.vulkan.narrow_f32_gemv = false;
    // Production clamps this knob to 64; a larger value cannot force a prefill into small-m.
    cfg.kernels.vulkan.moe_small_m = if small_m { 64 } else { 0 };
    let be = VulkanBackend::new_on_with(1, Arc::new(cfg)).expect("PR146 explicit Vulkan1 backend");
    assert!(
        be.caps().name.contains("7900 XTX") && !be.caps().integrated,
        "requires RX 7900 XTX, got {:?}",
        be.caps()
    );
    assert!(
        be.caps().f16_coopmat(),
        "PR146 tests require f16 cooperative matrices"
    );
    assert!(
        be.caps().i8_dot,
        "batched MoE control requires integer dot products"
    );
    be
}

fn upload(be: &VulkanBackend, bytes: &[u8], usage: BufferUsage) -> Box<dyn Buffer> {
    let buffer = be.alloc(bytes.len(), usage).unwrap();
    be.upload(buffer.as_ref(), bytes).unwrap();
    buffer
}

fn read_f32(be: &VulkanBackend, buffer: &dyn Buffer, count: usize) -> Vec<f32> {
    let mut values = vec![0.0; count];
    be.download(buffer, bytemuck::cast_slice_mut(&mut values))
        .unwrap();
    values
}

fn guarded_output(be: &VulkanBackend, buffer: &dyn Buffer, count: usize) -> Vec<f32> {
    let values = read_f32(be, buffer, count + GUARD);
    assert!(values[..count]
        .iter()
        .all(|v| v.is_finite() && v.to_bits() != POISON.to_bits()));
    assert!(
        values[count..]
            .iter()
            .all(|v| v.to_bits() == POISON.to_bits()),
        "output guard overwritten"
    );
    values[..count].to_vec()
}

fn unchanged(be: &VulkanBackend, buffer: &dyn Buffer, expected: &[u8]) {
    let mut got = vec![0; expected.len()];
    be.download(buffer, &mut got).unwrap();
    assert_eq!(got, expected, "input/weight/guard modified");
}

fn close(label: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (&actual, &expected)) in got.iter().zip(want).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{label}[{i}]: {actual} vs {expected}"
        );
    }
}

// Scale/code construction is exact in f16, BF16 and Q8, isolating routing from rounding noise.
fn router_weights(dt: DType, k: usize, experts: usize) -> Vec<u8> {
    let values: Vec<f32> = (0..experts * k)
        .map(|i| {
            if i % k == 0 {
                (i / k) as f32 / 128.0 - 0.5
            } else {
                0.0
            }
        })
        .collect();
    match dt {
        DType::Bf16 => values
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect(),
        DType::F32 => infr_testkit::f32_bytes(&values),
        DType::Q8_0 => values
            .chunks_exact(32)
            .flat_map(|block| {
                let mut bytes = half::f16::from_f32(1.0 / 128.0).to_le_bytes().to_vec();
                bytes.extend(block.iter().map(|&v| (v * 128.0) as i8 as u8));
                bytes
            })
            .collect(),
        _ => unreachable!(),
    }
}

fn expert_weights(experts: usize, elements_per_expert: usize, role: usize) -> Vec<u8> {
    (0..experts * elements_per_expert / 32)
        .flat_map(|block| {
            let expert = block * 32 / elements_per_expert;
            let code = 1 + (expert * (role + 1) + role) % 7;
            let mut bytes = half::f16::from_f32(1.0 / 32.0).to_le_bytes().to_vec();
            bytes.extend(std::iter::repeat_n(code as u8, 32));
            bytes
        })
        .collect()
}

struct MoeCase {
    graph: Graph,
    ids: [TensorId; 7],
    bytes: [Vec<u8>; 6],
    rows: usize,
    router_dtype: DType,
}

impl MoeCase {
    const K: usize = 64;
    const NFF: usize = 64;
    const EXPERTS: usize = 128;

    fn new(dt: DType, rows: usize) -> Self {
        let (k, nff, experts) = (Self::K, Self::NFF, Self::EXPERTS);
        let mut graph = Graph::new();
        let x = graph.input(TensorDesc::new(vec![rows, k], DType::F32));
        let rx = graph.input(TensorDesc::new(vec![rows, k], DType::F32));
        let router = graph.weight(TensorDesc::new(vec![experts, k], dt));
        let gate = graph.weight(TensorDesc::new(vec![experts, nff, k], DType::Q8_0));
        let up = graph.weight(TensorDesc::new(vec![experts, nff, k], DType::Q8_0));
        let down = graph.weight(TensorDesc::new(vec![experts, k, nff], DType::Q8_0));
        let y = graph.output(TensorDesc::new(vec![rows, k], DType::F32));
        graph.push(Op::MoeFfn {
            x,
            router_x: rx,
            router,
            gate_exps: gate,
            up_exps: up,
            down_exps: down,
            down_scale: None,
            fused_gate_up: false,
            dst: y,
            ne: k as u32,
            n_expert: experts as u32,
            n_used: 2,
            n_ff_exp: nff as u32,
            scale: 1.0,
            act: Activation::Silu,
            gating: MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
            swiglu_clamp: None,
            expert_ids: None,
        });
        let mut input = vec![0.25; rows * k];
        let mut router_input = vec![0.0; rows * k];
        for row in 0..rows {
            // Deliberately different expert/router input; selected experts swap with row parity.
            router_input[row * k] = if row % 2 == 0 { 2.0 } else { -2.0 };
            // Keep each Q8 activation block uniform: every code is exactly +127 before the
            // scale's existing f16 rounding. A single perturbed element instead shifts all
            // other values off their Q8 grid and obscures the router regression with expert
            // activation-quantization error. Rows still differ and router_x remains separate.
            input[row * k..(row + 1) * k].fill(0.25 + (row % 5) as f32 / 128.0);
        }
        input.extend(std::iter::repeat_n(POISON, GUARD));
        router_input.extend(std::iter::repeat_n(POISON, GUARD));
        Self {
            graph,
            ids: [x, rx, router, gate, up, down, y],
            rows,
            router_dtype: dt,
            bytes: [
                infr_testkit::f32_bytes(&input),
                infr_testkit::f32_bytes(&router_input),
                router_weights(dt, k, experts),
                expert_weights(experts, nff * k, 0),
                expert_weights(experts, nff * k, 1),
                expert_weights(experts, k * nff, 2),
            ],
        }
    }

    fn run(&self, be: &VulkanBackend, _small_m: bool) -> Vec<f32> {
        self.run_mode(be, false)
    }

    fn run_mode(&self, be: &VulkanBackend, paged: bool) -> Vec<f32> {
        let buffers: Vec<_> = self
            .bytes
            .iter()
            .enumerate()
            .map(|(i, bytes)| {
                if paged && i >= 3 {
                    be.alloc(4, BufferUsage::Staging).unwrap()
                } else {
                    upload(
                        be,
                        bytes,
                        if i < 2 {
                            BufferUsage::Activations
                        } else {
                            BufferUsage::Weights
                        },
                    )
                }
            })
            .collect();
        if paged {
            install_pager(be, &buffers[3..], &self.bytes[3..]);
        }
        let output = upload(
            be,
            bytemuck::cast_slice(&vec![POISON; self.rows * Self::K + GUARD]),
            BufferUsage::Readback,
        );
        let plan = be.compile(&self.graph).unwrap();
        let mut bindings = Bindings::new();
        for (id, buffer) in self.ids.iter().zip(&buffers) {
            bindings.bind(*id, buffer.as_ref());
        }
        bindings.bind(self.ids[6], output.as_ref());
        let admitted =
            moe_router_gemm_applies(be, self.router_dtype, self.rows, Self::K, Self::EXPERTS);
        for _ in 0..2 {
            be.execute(plan.as_ref(), &bindings).unwrap();
            let values = guarded_output(be, output.as_ref(), self.rows * Self::K);
            assert!(
                values.iter().any(|v| v.abs() > 0.1),
                "degenerate MoE oracle"
            );
        }
        for (i, (buffer, bytes)) in buffers.iter().zip(&self.bytes).enumerate() {
            if !paged || i < 3 {
                unchanged(be, buffer.as_ref(), bytes);
            }
        }
        if paged {
            let arena = be.runtime_phase.lock().unwrap();
            let capacities: Vec<_> = arena
                .pool
                .buffers
                .keys()
                .filter(|(tag, _)| *tag == "moe_paged_logits")
                .map(|(_, bytes)| *bytes)
                .collect();
            let required = moe_router_logits_rows(self.rows, admitted) * Self::EXPERTS * 4;
            assert!(
                capacities.contains(&required),
                "router logits reservation: {capacities:?}, want {required}"
            );
        }
        let router_gemm = be.built_kernel_names().iter().any(|name| {
            name.starts_with("native_gemm_warp_bf16_n128")
                || name.starts_with("native_gemm_warp_q8_0_n128")
        });
        // Q8 experts use id-native/MMQ kernels, never the dense router GEMM names above.
        assert_eq!(router_gemm, admitted, "router dispatch/cache mismatch");
        guarded_output(be, output.as_ref(), self.rows * Self::K)
    }

    fn oracle(&self) -> Vec<f32> {
        // Expert matrices are constant within each expert/role. Top-2 logits are separated by
        // exactly 1/64 and all router values are exactly representable in the three dtypes.
        let first_weight = 1.0 / (1.0 + (-1.0f32 / 64.0).exp());
        let mut output = Vec::with_capacity(self.rows * Self::K);
        for row in 0..self.rows {
            let selected = if row % 2 == 0 { [127, 126] } else { [0, 1] };
            let sum = Self::K as f32 * (0.25 + (row % 5) as f32 / 128.0);
            let mut value = 0.0;
            for (rank, expert) in selected.into_iter().enumerate() {
                let gate = sum * (1 + expert % 7) as f32 / 32.0;
                let up = sum * (1 + (expert * 2 + 1) % 7) as f32 / 32.0;
                let down = (1 + (expert * 3 + 2) % 7) as f32 * Self::NFF as f32 / 32.0;
                let weight = if rank == 0 {
                    first_weight
                } else {
                    1.0 - first_weight
                };
                value += weight * gate / (1.0 + (-gate).exp()) * up * down;
            }
            output.extend(std::iter::repeat_n(value, Self::K));
        }
        output
    }
}

fn append_planner_anchor(graph: &mut Graph) {
    let x = graph.input(TensorDesc::new(vec![1, 32], DType::F32));
    let router = graph.weight(TensorDesc::new(vec![4, 32], DType::F32));
    let gate = graph.weight(TensorDesc::new(vec![4, 32, 32], DType::Q8_0));
    let up = graph.weight(TensorDesc::new(vec![4, 32, 32], DType::Q8_0));
    let down = graph.weight(TensorDesc::new(vec![4, 32, 32], DType::Q8_0));
    let dst = graph.output(TensorDesc::new(vec![1, 32], DType::F32));
    graph.push(Op::MoeFfn {
        x,
        router_x: x,
        router,
        gate_exps: gate,
        up_exps: up,
        down_exps: down,
        down_scale: None,
        fused_gate_up: false,
        dst,
        ne: 32,
        n_expert: 4,
        n_used: 2,
        n_ff_exp: 32,
        scale: 1.0,
        act: Activation::Silu,
        gating: MoeGating::Softmax,
        norm_w: true,
        weight_before: false,
        ep_band: None,
        exp_probs_b: None,
        n_expert_groups: 0,
        n_expert_groups_used: 0,
        swiglu_clamp: None,
        expert_ids: None,
    });
}

fn install_pager(be: &VulkanBackend, placeholders: &[Box<dyn Buffer>], banks: &[Vec<u8>]) {
    use crate::pager::{ExpertSource, MoeHostChunkSpec, MoePagerLayout, MoePoolSpec, Role};
    use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
    struct FixtureIo(Vec<Vec<u8>>);
    impl BlockIo for FixtureIo {
        fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
            let bank = &self.0[desc.id as usize];
            assert_eq!(desc.nbytes(), bank.len());
            dst[..bank.len()].copy_from_slice(bank);
            Ok(())
        }
    }
    let bytes = banks[0].len();
    assert!(banks.iter().all(|bank| bank.len() == bytes));
    let stride = bytes / MoeCase::EXPERTS;
    let runtime_bytes: usize = 64 << 20;
    let expert_floor = 3 * MoeCase::EXPERTS;
    // The unified runtime corridor borrows physical expert slots. Reserving runtime bytes
    // without enough surplus slots leaves only 7*stride loanable bytes in the old fixture.
    // Price the floor's exchange spare, the complete runtime corridor, one whole-layer Prefill
    // lane, and a few granularity spares independently; do not weaken the production floor.
    let physical_floor = expert_floor + 1;
    let prefill_bytes = 3 * bytes;
    let slots =
        physical_floor + runtime_bytes.div_ceil(stride) + prefill_bytes.div_ceil(stride) + 8;
    let loanable_bytes = (slots - physical_floor) * stride;
    assert!(loanable_bytes >= runtime_bytes + prefill_bytes);
    be.init_moe_pager(MoePagerLayout {
        load_reserve_bytes: 0,
        n_blocks: MoeCase::EXPERTS,
        pools: vec![MoePoolSpec {
            slot_bytes: stride,
            n_slots: slots,
            min_enabled_slots: expert_floor,
        }],
        host_tier: None,
        host_store_io: Some(Arc::new(FixtureIo(banks.to_vec()))),
        dynamic_state_reserve_bytes: 0,
        dynamic_state_max_allocation_bytes: 0,
        prefill_min_lane_bytes: prefill_bytes as u64,
        runtime_reserve_bytes: runtime_bytes as u64,
        host_chunks: vec![MoeHostChunkSpec {
            base_offset: 0,
            bytes: 3 * bytes,
        }],
        prefill_target_lanes: 1,
        prefill_cache_bytes: prefill_bytes as u64,
    })
    .unwrap();
    for (i, role) in [Role::Gate, Role::Up, Role::Down].into_iter().enumerate() {
        be.register_paged_expert(
            role,
            crate::pager::buffer_identity(placeholders[i].as_ref()),
            ExpertSource {
                bank: Arc::new(banks[i].clone()),
                stride_bytes: stride,
                layer_base: 0,
                host_offset: i * bytes,
                file: Some(BlockDesc {
                    id: i as u32,
                    extents: vec![BlockExtent {
                        offset: 0,
                        len: bytes,
                    }],
                }),
            },
            MoeCase::EXPERTS,
        )
        .unwrap();
    }
    be.finish_session_allocations().unwrap();
}

#[test]
#[ignore = "serial explicit XTX; real paged MoeFfn, padded logits and guarded payloads"]
fn pr146_moe_router_actual_paged_graph_and_staged_padding() {
    let be = backend(true, true, false);
    let fallback = backend(false, true, false);
    let case = MoeCase::new(DType::Bf16, 259);
    let expected = case.run_mode(&fallback, true);
    let actual = case.run_mode(&be, true);
    close("paged MoE old/candidate router", &actual, &expected, 1e-5);
    close("paged MoE old host oracle", &expected, &case.oracle(), 3e-3);
    close("paged MoE host oracle", &actual, &case.oracle(), 3e-3);

    // A larger expert topology keeps the same ragged M in Decode phase, where staged workspace
    // is frozen. This is planner-only metadata: no large expert banks are allocated or executed.
    for (dt, rows, padded) in [
        (DType::Bf16, 259, 320),
        (DType::Q8_0, 515, 576),
        (DType::F32, 259, 259),
    ] {
        let mut metadata = MoeCase::new(dt, rows);
        for id in &metadata.ids[2..6] {
            metadata.graph.tensors[id.0 as usize].desc.shape[0] = 1024;
        }
        if let Op::MoeFfn { n_expert, .. } = &mut metadata.graph.ops[0] {
            *n_expert = 1024;
        }
        assert_eq!(
            paged_static_phase(&metadata.graph),
            Some(RuntimePhase::Decode)
        );
        let requests = stages::compiled_workspace_requests(&be, &metadata.graph).unwrap();
        assert_eq!(
            requests
                .activation
                .iter()
                .find(|(tag, _)| *tag == "moe_paged_logits")
                .map(|(_, bytes)| *bytes),
            Some(padded * 1024 * 4)
        );
    }
}

fn moe_matrix(small_m: bool, cases: &[(DType, usize)]) {
    // Reuse pipeline caches across cases; buffers and guarded bindings belong to each graph.
    let candidate = backend(true, small_m, false);
    let fallback = backend(false, small_m, false);
    for &(dt, rows) in cases {
        let case = MoeCase::new(dt, rows);
        let expected = case.run(&fallback, small_m);
        let got = case.run(&candidate, small_m);
        // Establish router parity FIRST, before consulting an independent host reference.
        // Uniform expert input blocks isolate the router without dropping the 0.3% host bar.
        close(
            &format!("real MoeFfn old/candidate {dt:?} rows={rows} small_m_config={small_m}"),
            &got,
            &expected,
            1e-5,
        );
        close("MoE old host oracle", &expected, &case.oracle(), 3e-3);
        close("MoE independent host oracle", &got, &case.oracle(), 3e-3);
    }
}

#[test]
#[ignore = "serial explicit Vulkan GPU; real MoeFfn router, ragged small-m scratch and output guards"]
fn pr146_moe_router_actual_graph_small_m_ragged_guards() {
    // Both are truly <= the production 64-row small-m ceiling; the separate batched case
    // retains qualified BF16 M=259 and Q8 M=515 router-GEMM coverage.
    const {
        assert!(33 <= MOE_SMALL_M.max && 63 <= MOE_SMALL_M.max);
    }
    moe_matrix(true, &[(DType::Bf16, 33), (DType::Q8_0, 63)]);
}

#[test]
#[ignore = "serial explicit Vulkan GPU; real batched MoeFfn router and ragged output guards"]
fn pr146_moe_router_actual_graph_batched_ragged_guards() {
    moe_matrix(false, &[(DType::Bf16, 259), (DType::Q8_0, 515)]);
}

#[test]
#[ignore = "serial explicit Vulkan GPU; real MoeFfn F32 and sub-threshold router fallback"]
fn pr146_moe_router_actual_graph_f32_and_small_m_fallbacks() {
    moe_matrix(
        true,
        &[(DType::F32, 259), (DType::Bf16, 255), (DType::Q8_0, 511)],
    );
}

fn linear_graph_case(be: &VulkanBackend, rows: usize, k: usize, n: usize, kernel: &str) {
    linear_graph_case_mode(be, rows, k, n, kernel, false);
}

fn linear_graph_case_mode(
    be: &VulkanBackend,
    rows: usize,
    k: usize,
    n: usize,
    kernel: &str,
    streamed: bool,
) {
    let mut graph = Graph::new();
    let xs =
        std::array::from_fn::<_, 2, _>(|_| graph.input(TensorDesc::new(vec![rows, k], DType::F32)));
    let ws = std::array::from_fn::<_, 2, _>(|_| {
        graph.weight(TensorDesc::new(vec![n + 1, k], DType::Bf16))
    });
    let ys = std::array::from_fn::<_, 2, _>(|_| {
        graph.output(TensorDesc::new(vec![rows, n], DType::F32))
    });
    let internal = graph.internal(TensorDesc::new(vec![rows, n], DType::F32));
    for j in 0..2 {
        let dst = if j == 0 { internal } else { ys[j] };
        graph.push(Op::Linear {
            x: xs[j],
            weight: ws[j],
            dst,
            m: rows as u32,
            in_f: k as u32,
            out_f: n as u32,
            w_off: k as u32,
        });
        if j == 0 {
            graph.push(Op::Copy {
                src: internal,
                src_off: 0,
                dst: ys[j],
                dst_off: 0,
                n: (rows * n) as u32,
            });
        }
    }
    // Full workspace enumeration is Decode-phase only. Use a metadata-only one-row MoE
    // anchor to exercise that planner on these same Linear ops; execute the original graph.
    let mut planned = graph.clone();
    append_planner_anchor(&mut planned);
    let requests = stages::compiled_workspace_requests(be, &planned).unwrap();
    assert_eq!(
        requests
            .activation
            .iter()
            .find(|(tag, _)| *tag == "lin_a16")
            .map(|(_, bytes)| *bytes),
        Some(rows.div_ceil(64) * 64 * k * 2),
        "A_GLOBAL scratch must mirror actual tile selection"
    );
    assert!(!requests.activation.iter().any(|(tag, _)| *tag == "lin_w32"));
    let inputs: [Vec<f32>; 2] = std::array::from_fn(|j| {
        let mut values: Vec<_> = (0..rows * k)
            .map(|i| {
                let row = (i / k + j * 2) % 11;
                (row as f32 - 5.0) * ((i % k * 3 % 7) as f32 - 3.0) / 8.0
            })
            .collect();
        values.extend(std::iter::repeat_n(POISON, GUARD));
        values
    });
    let weights: [Vec<u8>; 2] = std::array::from_fn(|j| {
        (0..(n + 1) * k)
            .flat_map(|i| {
                let value = if i < k {
                    16.0
                } else {
                    let column = (i / k - 1) * 5 + j * 3;
                    (column % 15) as f32 - 7.0
                } * ((i % k * 5 % 11) as f32 - 5.0)
                    / 16.0;
                half::bf16::from_f32(value).to_le_bytes()
            })
            .collect()
    });
    let xb: [Box<dyn Buffer>; 2] = std::array::from_fn(|j| {
        upload(
            be,
            bytemuck::cast_slice(&inputs[j]),
            BufferUsage::Activations,
        )
    });
    let wb: [Box<dyn Buffer>; 2] = std::array::from_fn(|j| {
        if streamed {
            be.alloc(4, BufferUsage::Weights).unwrap()
        } else {
            upload(be, &weights[j], BufferUsage::Weights)
        }
    });
    if streamed {
        let slot_bytes = weights[0].len().next_multiple_of(4);
        be.init_dense_pager(crate::pager::DensePagerLayout {
            pools: vec![crate::pager::DensePoolSpec {
                slot_bytes,
                n_slots: 1,
                n_blocks: 2,
                host: None,
            }],
            ring_bytes: slot_bytes * 2,
        })
        .unwrap();
        for j in 0..2 {
            be.register_dense_stream(
                0,
                crate::pager::buffer_identity(wb[j].as_ref()),
                crate::pager::DenseSource {
                    bytes: crate::pager::DenseBytes::Mmap(vec![Arc::new(weights[j].clone())]),
                    block_id: j as u32,
                },
            )
            .unwrap();
        }
    }
    let yb: [Box<dyn Buffer>; 2] = std::array::from_fn(|_| {
        upload(
            be,
            bytemuck::cast_slice(&vec![POISON; rows * n + GUARD]),
            BufferUsage::Readback,
        )
    });
    let plan = be.compile(&graph).unwrap();
    let mut bindings = Bindings::new();
    for j in 0..2 {
        bindings.bind(xs[j], xb[j].as_ref());
        bindings.bind(ws[j], wb[j].as_ref());
        bindings.bind(ys[j], yb[j].as_ref());
    }
    let basis: f32 = (0..k)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) * ((i * 5 % 11) as f32 - 5.0))
        .sum();
    for _ in 0..2 {
        be.execute(plan.as_ref(), &bindings).unwrap();
        for (j, y) in yb.iter().enumerate() {
            let got = guarded_output(be, y.as_ref(), rows * n);
            let expected: Vec<_> = (0..rows * n)
                .map(|i| {
                    (((i / n + j * 2) % 11) as f32 - 5.0)
                        * (((i % n * 5 + j * 3) % 15) as f32 - 7.0)
                        * basis
                        / 128.0
                })
                .collect();
            assert!(expected.iter().any(|v| v.abs() > 0.1));
            close("BF16 actual Linear", &got, &expected, 2e-4);
        }
    }
    for j in 0..2 {
        unchanged(be, xb[j].as_ref(), bytemuck::cast_slice(&inputs[j]));
        unchanged(
            be,
            wb[j].as_ref(),
            if streamed { &[0; 4] } else { &weights[j] },
        );
    }
    let names = be.built_kernel_names();
    assert!(
        names.iter().any(|name| name.starts_with(kernel)),
        "missing {kernel}, built {names:?}"
    );
    assert!(
        !names.iter().any(|name| name.starts_with("weight_f32_bf16")),
        "must exercise native BF16 weights"
    );
}

#[test]
#[ignore = "serial explicit Vulkan GPU; native BF16 wide/n128/n64 A_GLOBAL, actual Linear, scratch/cache/guards"]
fn pr146_bf16_linear_actual_graph_ag_scratch_guards_and_cache() {
    let be = backend(true, true, true);
    // The first shape meets the production wide_grid>=128 gate; the others select n128/n64.
    for (rows, k, n, kernel) in [
        (259, 64, 8192, "native_gemm_warp_bf16_ag"),
        (259, 64, 128, "native_gemm_warp_bf16_n128_ag"),
        (259, 64, 320, "native_gemm_warp_bf16_n64_ag"),
    ] {
        linear_graph_case(&be, rows, k, n, kernel);
    }
}

#[test]
#[ignore = "serial explicit XTX; streamed BF16 A_GLOBAL with one weight slot, nonzero offsets and repeated graphs"]
fn pr146_bf16_linear_streamed_ag_slot_reuse_and_guards() {
    let be = backend(true, true, false);
    linear_graph_case_mode(&be, 259, 256, 320, "native_gemm_warp_bf16_n64_ag", true);
}
