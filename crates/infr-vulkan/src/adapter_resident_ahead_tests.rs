//! Small model-free resident command-ahead regressions; declared as a child of `adapter`.
use super::*;
use crate::pager::{ExpertSource, MoeHostChunkSpec, MoePagerLayout, MoePoolSpec, Role};
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::{DType, SizeSpec, TensorDesc};
use std::any::Any;
use std::sync::atomic::AtomicUsize;

const NE: usize = 32;
const FF: usize = 32;
const EXP: usize = 4;
const TOP: usize = 2;
const LAYERS: usize = 2;
const SLOT_BYTES: usize = 64 * 1024;
const BANK_BYTES: usize = EXP * SLOT_BYTES;
const ROUTED: [u32; TOP] = [0, 1];
const HIT_MASK: u32 = (1 << TOP) - 1;

struct CpuBuffer(usize);

impl Buffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        self.0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn resident_ahead_snapshot_drop_preserves_live_miss_announcement() {
    let (_output, results) = mpsc::channel();
    let announced = Arc::new(AtomicUsize::new(1));
    let producer = Arc::new(ResidentNoticeProducer {
        sender: None,
        results: Mutex::new(results),
        thread: None,
        announced_miss: Arc::clone(&announced),
    });
    let chunk = ResidentChunk {
        control: Arc::new(CpuBuffer(crate::recorder::RESIDENT_CONTROL_BYTES)),
        journal: Arc::new(CpuBuffer(3 * crate::recorder::RESIDENT_JOURNAL_BYTES)),
        entries: Vec::new(),
        early_recovery: true,
        recovery_op: None,
        limit: 3,
        recording: true,
        submit_layers: 1,
        submitted_entries: 0,
        notices: None,
        notice_cursor: 0,
        notice_resume: None,
        notice_stop: false,
        producer: Some(Arc::clone(&producer)),
        notices_sent: 0,
        mapped_serial_max_bytes: 0,
        waiter: None,
        indirect: None,
    };

    for miss in [0, 1, usize::MAX] {
        announced.store(miss, Ordering::Release);
        let snapshot = chunk.preparation_snapshot().expect("eligible snapshot");
        assert!(snapshot.producer.is_none());
        assert!(snapshot.waiter.is_none());
        assert!(Arc::ptr_eq(&snapshot.control, &chunk.control));
        drop(snapshot);
        assert_eq!(announced.load(Ordering::Acquire), miss);
        assert_eq!(Arc::strong_count(&producer), 2);
    }
    // The real controller still owns teardown; only speculative teardown must be inert.
    announced.store(1, Ordering::Release);
    drop(chunk);
    assert_eq!(announced.load(Ordering::Acquire), usize::MAX);
}

struct DiagonalExpertIo;

impl BlockIo for DiagonalExpertIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        if desc.nbytes() != BANK_BYTES || dst.len() < BANK_BYTES || desc.id >= 6 {
            return Err(be("invalid resident-ahead expert fixture range"));
        }
        dst[..BANK_BYTES].fill(0);
        for expert in 0..EXP {
            let scale = 0.125 + desc.id as f32 / 64.0 + expert as f32 / 32.0;
            for diagonal in 0..NE {
                let offset = expert * SLOT_BYTES + (diagonal * NE + diagonal) * 4;
                dst[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
            }
        }
        Ok(())
    }
}

struct Fixture {
    graph: Graph,
    frontend: TensorId,
    outputs: [TensorId; LAYERS],
    buffers: Vec<(TensorId, Box<dyn Buffer>)>,
    roles: [[usize; 3]; LAYERS],
    backend: VulkanBackend,
}

impl Fixture {
    fn new(submit_layers: usize, producer: bool) -> Self {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.submit_dispatches = Some(0);
        cfg.device.decode_command_ahead = true;
        cfg.device.decode_command_early_submit = true;
        cfg.device.decode_command_resident = true;
        cfg.paging.expert_prefetch = false;
        cfg.paging.host_dma = true;
        cfg.paging.ring = Some(SizeSpec::Bytes((4 * SLOT_BYTES) as u64));
        cfg.kernels.vulkan.cpu_miss_threads = 0;
        cfg.kernels.vulkan.static_scratch_memo = true;
        cfg.kernels.vulkan.decode_gpu_resident = true;
        cfg.kernels.vulkan.decode_gpu_resident_layers = 3;
        cfg.kernels.vulkan.decode_gpu_resident_submit_layers = submit_layers;
        cfg.kernels.vulkan.decode_gpu_resident_inflight = 1;
        cfg.kernels.vulkan.decode_gpu_resident_hit_first = true;
        cfg.kernels.vulkan.decode_gpu_resident_fast_lut = true;
        cfg.kernels.vulkan.decode_gpu_resident_immutable_predicates = true;
        cfg.kernels.vulkan.decode_gpu_resident_indirect_predicates = false;
        cfg.kernels.vulkan.decode_gpu_resident_notice_thread = producer;
        cfg.kernels.vulkan.decode_gpu_resident_early_recovery = producer;
        cfg.kernels.vulkan.decode_gpu_resident_waiter = producer;
        cfg.prof.ops = false;
        cfg.prof.stages = false;
        cfg.prof.timeline_path = None;

        let requested_device = std::env::var("MOE4ALL_TEST_DEVICE").ok();
        if let Some(device) = &requested_device {
            cfg.device.dev = Some(device.clone());
            let devices = VulkanBackend::enumerate_devices(&cfg).expect("enumerate test device");
            let selected = devices
                .iter()
                .find(|device| device.is_default_pick)
                .expect("test device must resolve to one device");
            assert!(
                selected.name.contains("7900 XTX"),
                "MOE4ALL_TEST_DEVICE selected {}, expected RX 7900 XTX",
                selected.name
            );
        }
        let backend = VulkanBackend::new_with(Arc::new(cfg)).expect("Vulkan fixture backend");
        if requested_device.is_some() {
            assert!(backend.caps().name.contains("7900 XTX"));
        }
        assert!(backend.shared.conditional_rendering.is_some());
        backend
            .init_moe_pager(MoePagerLayout {
                load_reserve_bytes: 0,
                n_blocks: LAYERS * EXP,
                pools: vec![MoePoolSpec {
                    slot_bytes: SLOT_BYTES,
                    n_slots: 64,
                    min_enabled_slots: LAYERS * EXP * 3,
                }],
                host_tier: None,
                host_store_io: Some(Arc::new(DiagonalExpertIo)),
                dynamic_state_reserve_bytes: 0,
                dynamic_state_max_allocation_bytes: 0,
                prefill_min_lane_bytes: 0,
                runtime_reserve_bytes: 1024 * 1024,
                host_chunks: vec![MoeHostChunkSpec {
                    base_offset: 0,
                    bytes: LAYERS * 3 * BANK_BYTES,
                }],
                prefill_target_lanes: 1,
                prefill_cache_bytes: (64 * SLOT_BYTES) as u64,
            })
            .expect("small padded expert pool");
        assert!(
            command_prefix::enabled(&backend),
            "preparation must be active"
        );

        let mut graph = Graph::new();
        let frontend = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
        let one = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
        let input = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
        let routing = graph.input(TensorDesc::new(vec![1, NE], DType::F32));
        let router = graph.weight(TensorDesc::new(vec![EXP, NE], DType::F32));
        let mut role_handles = Vec::new();
        let mut outputs = Vec::new();
        let mut x = input;
        for _ in 0..LAYERS {
            graph.push(Op::Add {
                a: frontend,
                b: one,
                dst: frontend,
                n: NE as u32,
            });
            let gate_exps = graph.weight(TensorDesc::new(vec![EXP, FF, NE], DType::F32));
            let up_exps = graph.weight(TensorDesc::new(vec![EXP, FF, NE], DType::F32));
            let down_exps = graph.weight(TensorDesc::new(vec![EXP, NE, FF], DType::F32));
            role_handles.extend([gate_exps, up_exps, down_exps]);
            let dst = graph.output(TensorDesc::new(vec![1, NE], DType::F32));
            graph.push(Op::MoeFfn {
                x,
                router_x: routing,
                router,
                gate_exps,
                up_exps,
                down_exps,
                down_scale: None,
                fused_gate_up: false,
                dst,
                ne: NE as u32,
                n_expert: EXP as u32,
                n_used: TOP as u32,
                n_ff_exp: FF as u32,
                scale: 1.0,
                act: Activation::Silu,
                gating: infr_core::graph::MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                ep_band: None,
                exp_probs_b: None,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
                swiglu_clamp: None,
                expert_ids: None,
            });
            outputs.push(dst);
            x = dst;
        }
        let mut buffers = Vec::new();
        for (id, values, usage) in [
            (frontend, vec![0.0f32; NE], BufferUsage::Staging),
            (one, vec![1.0f32; NE], BufferUsage::Activations),
            (input, vec![1.0f32; NE], BufferUsage::Activations),
            (routing, vec![1.0f32; NE], BufferUsage::Activations),
        ] {
            let buffer = backend.alloc(NE * 4, usage).unwrap();
            backend
                .upload(buffer.as_ref(), bytemuck::cast_slice(&values))
                .unwrap();
            buffers.push((id, buffer));
        }
        let router_buffer = backend.alloc(EXP * NE * 4, BufferUsage::Weights).unwrap();
        let mut router_values = vec![0.0f32; EXP * NE];
        for expert in 0..EXP {
            router_values[expert * NE] = (EXP - expert) as f32;
        }
        backend
            .upload(router_buffer.as_ref(), bytemuck::cast_slice(&router_values))
            .unwrap();
        buffers.push((router, router_buffer));
        let bank = Arc::new(vec![0u8; BANK_BYTES]);
        let mut role_ids = Vec::new();
        for (index, id) in role_handles.into_iter().enumerate() {
            let placeholder = backend.alloc(4, BufferUsage::Staging).unwrap();
            let role_id = crate::pager::buffer_identity(placeholder.as_ref());
            backend
                .register_paged_expert(
                    [Role::Gate, Role::Up, Role::Down][index % 3],
                    role_id,
                    ExpertSource {
                        bank: bank.clone(),
                        stride_bytes: SLOT_BYTES,
                        layer_base: (index / 3 * EXP) as u32,
                        host_offset: index * BANK_BYTES,
                        file: Some(BlockDesc {
                            id: index as u32,
                            extents: vec![BlockExtent {
                                offset: 0,
                                len: BANK_BYTES,
                            }],
                        }),
                    },
                    EXP,
                )
                .unwrap();
            buffers.push((id, placeholder));
            role_ids.push(role_id);
        }
        backend.finish_session_allocations().unwrap();
        for &id in &outputs {
            buffers.push((id, backend.alloc(NE * 4, BufferUsage::Readback).unwrap()));
        }
        Self {
            graph,
            frontend,
            outputs: outputs.try_into().unwrap(),
            buffers,
            roles: [
                role_ids[..3].try_into().unwrap(),
                role_ids[3..].try_into().unwrap(),
            ],
            backend,
        }
    }

    fn buffer(&self, id: TensorId) -> &dyn Buffer {
        self.buffers
            .iter()
            .find(|(key, _)| *key == id)
            .unwrap()
            .1
            .as_ref()
    }

    fn bindings(&self) -> Bindings<'_> {
        let mut bindings = Bindings::new();
        for (id, buffer) in &self.buffers {
            bindings.bind(*id, buffer.as_ref());
        }
        bindings
    }

    fn read(&self, buffer: &dyn Buffer) -> Vec<f32> {
        let mut values = vec![0.0f32; NE];
        self.backend
            .download(buffer, bytemuck::cast_slice_mut(&mut values))
            .unwrap();
        values
    }

    fn reset_frontend(&self) {
        self.backend
            .upload(
                self.buffer(self.frontend),
                bytemuck::cast_slice(&[0.0f32; NE]),
            )
            .unwrap();
    }

    fn assert_frontend(&self, buffer: &dyn Buffer, count: f32) {
        assert_eq!(
            self.read(buffer),
            vec![count; NE],
            "frontend must run exactly once per layer"
        );
    }

    fn outputs(&self) -> Vec<Vec<f32>> {
        self.outputs
            .iter()
            .map(|&id| self.read(self.buffer(id)))
            .collect()
    }

    fn tape(&self) -> Vec<u8> {
        let tape = self
            .backend
            .moe_pager()
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .retained_tape();
        // Read only initialized windows belonging to this complete two-layer execution.
        let mut bytes = vec![0u8; LAYERS * 3 * EXP * 8];
        self.backend.download(tape.as_ref(), &mut bytes).unwrap();
        bytes
    }

    fn preload(&self) {
        let mut guard = self.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        for roles in &self.roles {
            session.begin_role_batches(roles).unwrap();
            session
                .push_roles_cpu(
                    &self.backend,
                    &roles.map(|role| (role, ROUTED.as_slice())),
                    false,
                )
                .unwrap()
                .complete_without_recorder(&self.backend)
                .unwrap();
        }
    }

    fn prepare(&self, submit_layers: usize, producer: bool) -> (VkDecodePlan, Vec<Vec<f32>>) {
        let current = VkDecodePlan::new_cpu(&self.graph, false, true);
        let next = VkDecodePlan::new_cpu(&self.graph, false, true);
        let bindings = self.bindings();
        self.preload();
        self.reset_frontend();
        execute_static(&self.backend, &current, &bindings).unwrap();
        self.assert_frontend(self.buffer(self.frontend), 2.0);
        let expected = self.outputs();
        assert!(expected
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value > 0.0));
        let expected_tape = self.tape();
        {
            let phase = self.backend.runtime_phase.lock().unwrap();
            let resident = phase
                .resident_buffers
                .as_ref()
                .expect("resident controller warmup");
            assert_eq!(
                resident.producer.is_some(),
                producer,
                "requested producer must be active"
            );
        }
        self.reset_frontend();
        let epoch = self.backend.shared.resource_epoch.load(Ordering::Acquire);
        execute_static_next(
            &self.backend,
            &current,
            &bindings,
            Some((&next, &bindings, &[1])),
        )
        .unwrap();
        {
            let packet = next.prepared_prefix.lock().unwrap();
            let packet = packet
                .as_ref()
                .expect("actual command preparation, not calibration fallback");
            // The short prefix ends at 1; an end after MoeFfn proves resident content was recorded.
            assert_eq!(packet.end, submit_layers * 2);
            assert!(packet.end > command_prefix::prefix_end(&self.graph).unwrap());
        }
        self.assert_frontend(self.buffer(self.frontend), 2.0);
        assert_close(&self.outputs(), &expected);
        assert_eq!(
            self.tape(),
            expected_tape,
            "recording must not overwrite the current token's LUT tape"
        );
        assert_eq!(
            self.backend.shared.resource_epoch.load(Ordering::Acquire),
            epoch,
            "warm recording and fixture inspection must not invalidate the prepared packet"
        );
        (next, expected)
    }

    fn run_case(&self, submit_layers: usize, producer: bool, missing: Option<usize>) {
        let (next, expected) = self.prepare(submit_layers, producer);
        let epoch = self.backend.shared.resource_epoch.load(Ordering::Acquire);
        self.reset_frontend();
        {
            let mut guard = self.backend.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            for roles in &self.roles {
                assert_eq!(
                    session.routed_roles_resident_mask(roles, &ROUTED).unwrap(),
                    HIT_MASK
                );
            }
            if let Some(layer) = missing {
                session.discard_cpu_miss(self.roles[layer], &[ROUTED[0]]);
                assert_eq!(
                    session
                        .routed_roles_resident_mask(&self.roles[layer], &ROUTED)
                        .unwrap(),
                    HIT_MASK & !1
                );
            }
        }
        assert_eq!(
            self.backend.shared.resource_epoch.load(Ordering::Acquire),
            epoch,
            "forcing a miss must preserve the packet's resource epoch"
        );
        execute_static(&self.backend, &next, &self.bindings()).unwrap();
        assert!(
            next.prepared_prefix.lock().unwrap().is_none(),
            "packet must be consumed"
        );
        self.assert_frontend(self.buffer(self.frontend), 2.0);
        assert_close(&self.outputs(), &expected);
        if let Some(layer) = missing {
            let guard = self.backend.moe_pager().lock().unwrap();
            assert_eq!(
                guard
                    .as_ref()
                    .unwrap()
                    .routed_roles_resident_mask(&self.roles[layer], &ROUTED)
                    .unwrap(),
                HIT_MASK
            );
        }
    }

    fn binding_invalidation(&self, submit_layers: usize, producer: bool) {
        let (next, expected) = self.prepare(submit_layers, producer);
        self.reset_frontend();
        let replacement = self.backend.alloc(NE * 4, BufferUsage::Staging).unwrap();
        self.backend
            .upload(replacement.as_ref(), bytemuck::cast_slice(&[0.0f32; NE]))
            .unwrap();
        let mut bindings = self.bindings();
        bindings.bind(self.frontend, replacement.as_ref());
        execute_static(&self.backend, &next, &bindings).unwrap();
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        self.assert_frontend(replacement.as_ref(), 2.0);
        self.assert_frontend(self.buffer(self.frontend), 0.0);
        assert_close(&self.outputs(), &expected);
    }

    fn unsupported_next(&self) {
        let current = VkDecodePlan::new_cpu(&self.graph, false, true);
        let mut graph = self.graph.clone();
        graph.prefill_target_tokens = Some(1);
        let next = VkDecodePlan::new_cpu(&graph, false, true);
        self.reset_frontend();
        let bindings = self.bindings();
        execute_static_next(
            &self.backend,
            &current,
            &bindings,
            Some((&next, &bindings, &[1])),
        )
        .unwrap();
        assert!(
            next.prepared_prefix.lock().unwrap().is_none(),
            "unsupported next graphs must not produce a packet"
        );
        self.assert_frontend(self.buffer(self.frontend), 2.0);
    }
}

fn assert_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                actual.is_finite()
                    && (actual - expected).abs() <= 1e-6f32.max(expected.abs() * 1e-5),
                "{actual} != {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires an idle Vulkan GPU; model-free 4 MiB expert pool"]
fn gpu_resident_ahead_prepares_and_recovers_without_repeating_frontends() {
    for (submit_layers, producer) in [(1, false), (1, true), (2, false), (2, true)] {
        let fixture = Fixture::new(submit_layers, producer);
        for missing in [None, Some(0), Some(1)] {
            fixture.run_case(submit_layers, producer, missing);
        }
        fixture.binding_invalidation(submit_layers, producer);
        fixture.unsupported_next();
    }
}
