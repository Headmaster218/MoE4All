#![cfg(test)]

//! Actual ordinary adapter and ACK-activated prepared C, independent of NE32/F32 fixtures.
use super::*;
use crate::pager::{ExpertSource, MoeHostChunkSpec, MoePagerLayout, MoePoolSpec, Role};
use crate::recorder::owned_role_pipeline as pipeline;
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::{DType, SizeSpec, TensorDesc};

#[path = "adapter_owned_role_pipeline_mixed_tests.rs"]
mod mixed_tests;
#[path = "adapter_owned_splitk_tests.rs"]
mod splitk_tests;

const NE: usize = 256;
const FF: usize = 256;
const EXP: usize = 4;
const TOP: usize = 2;
const GUARD_WORDS: usize = 64;
const GUARD: f32 = -1234.5;

struct ExpertIo([Arc<Vec<u8>>; 3]);

impl BlockIo for ExpertIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        let bank = self
            .0
            .get(desc.id as usize)
            .ok_or_else(|| be("role-pipeline fixture bank"))?;
        if desc.nbytes() != bank.len() || dst.len() < bank.len() {
            return Err(be("role-pipeline fixture bank extent"));
        }
        dst[..bank.len()].copy_from_slice(bank.as_slice());
        Ok(())
    }
}

struct Fixture {
    backend: VulkanBackend,
    graph: Graph,
    frontend: TensorId,
    output: TensorId,
    roles: [usize; 3],
    buffers: Vec<(TensorId, Box<dyn Buffer>)>,
    ne: usize,
    nff: usize,
    rows: usize,
    used: usize,
    routed: Vec<u32>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_profile(NE, FF, TOP, 1, [DType::Q5K; 3], true)
    }

    fn with_profile(
        ne: usize,
        nff: usize,
        used: usize,
        rows: usize,
        dtypes: [DType; 3],
        enabled: bool,
    ) -> Self {
        Self::with_profile_and_host_ready(ne, nff, used, rows, dtypes, enabled, false)
    }

    fn with_profile_and_host_ready(
        ne: usize,
        nff: usize,
        used: usize,
        rows: usize,
        dtypes: [DType; 3],
        enabled: bool,
        host_ready: bool,
    ) -> Self {
        Self::with_profile_modes(
            ne, nff, used, rows, dtypes, enabled, host_ready, false, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_profile_modes(
        ne: usize,
        nff: usize,
        used: usize,
        rows: usize,
        dtypes: [DType; 3],
        enabled: bool,
        host_ready: bool,
        splitk: bool,
        expert_count: Option<usize>,
    ) -> Self {
        Self::with_profile_iq3_modes(
            ne,
            nff,
            used,
            rows,
            dtypes,
            enabled,
            host_ready,
            splitk,
            expert_count,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_profile_iq3_modes(
        ne: usize,
        nff: usize,
        used: usize,
        rows: usize,
        dtypes: [DType; 3],
        enabled: bool,
        host_ready: bool,
        splitk: bool,
        expert_count: Option<usize>,
        iq3_trial: bool,
    ) -> Self {
        let experts = expert_count.unwrap_or(EXP.max(used));
        let slots = 64.max(experts * 3 + 4 + usize::from(splitk) * 32);
        let max_weight = dtypes
            .into_iter()
            .map(|dtype| infr_testkit::synth_weight(dtype, ne * nff, 0).len())
            .max()
            .unwrap();
        let slot_bytes = ((max_weight + GUARD_WORDS * 4).div_ceil(4096) * 4096).max(64 * 1024);
        let bank_bytes = experts * slot_bytes;
        let value_count = rows * ne;
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("parent selects the idle test GPU"));
        cfg.device.submit_dispatches = Some(0);
        cfg.device.decode_command_ahead = true;
        cfg.device.decode_command_early_submit = true;
        cfg.device.decode_command_resident = true;
        cfg.spec.mtp = false;
        cfg.paging.expert_prefetch = false;
        cfg.paging.host_dma = true;
        cfg.paging.ring = Some(SizeSpec::Bytes((4 * slot_bytes) as u64));
        let vk = &mut cfg.kernels.vulkan;
        vk.cpu_miss_threads = 0;
        vk.static_scratch_memo = true;
        vk.decode_gpu_resident = true;
        vk.decode_gpu_resident_copy = Some(true);
        vk.decode_gpu_resident_copy_overlap = true;
        vk.decode_gpu_resident_role_pipeline = enabled;
        vk.decode_gpu_resident_splitk_hit = splitk;
        vk.decode_gpu_resident_splitk_iq3_hit = iq3_trial;
        vk.decode_gpu_resident_layers = 1;
        vk.decode_gpu_resident_submit_layers = 1;
        vk.decode_gpu_resident_inflight = 1;
        vk.decode_gpu_resident_hit_first = true;
        vk.decode_gpu_resident_fast_lut = true;
        vk.decode_gpu_resident_mapped_metadata = true;
        vk.decode_gpu_resident_mapped_journal = true;
        vk.decode_gpu_resident_immutable_predicates = true;
        vk.decode_gpu_resident_indirect_predicates = false;
        vk.decode_gpu_resident_notice_thread = true;
        vk.decode_gpu_resident_early_recovery = true;
        vk.decode_gpu_resident_waiter = false;
        vk.decode_gpu_resident_host_ready = host_ready;
        vk.decode_gpu_resident_pure_mapped_layers = 0;
        cfg.prof.ops = false;
        cfg.prof.stages = false;
        cfg.prof.timeline_path = None;
        let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        assert!(backend.shared.conditional_rendering.is_some());

        let banks: [Arc<Vec<u8>>; 3] = std::array::from_fn(|role| {
            let mut bank = vec![0xa5; bank_bytes];
            for expert in 0..experts {
                let weight = infr_testkit::synth_weight(
                    dtypes[role],
                    ne * nff,
                    101 + (role * experts + expert) as u32,
                );
                assert!(weight.len() + GUARD_WORDS * 4 <= slot_bytes);
                bank[expert * slot_bytes..expert * slot_bytes + weight.len()]
                    .copy_from_slice(&weight);
            }
            Arc::new(bank)
        });
        backend
            .init_moe_pager(MoePagerLayout {
                load_reserve_bytes: 0,
                n_blocks: experts,
                pools: vec![MoePoolSpec {
                    slot_bytes,
                    n_slots: slots,
                    min_enabled_slots: experts * 3,
                }],
                host_tier: None,
                host_store_io: Some(Arc::new(ExpertIo(banks.clone()))),
                dynamic_state_reserve_bytes: 0,
                dynamic_state_max_allocation_bytes: 0,
                prefill_min_lane_bytes: 0,
                runtime_reserve_bytes: if splitk {
                    16 * 1024 * 1024
                } else {
                    1024 * 1024
                },
                host_chunks: vec![MoeHostChunkSpec {
                    base_offset: 0,
                    bytes: 3 * bank_bytes,
                }],
                prefill_target_lanes: 1,
                prefill_cache_bytes: (slots * slot_bytes) as u64,
            })
            .unwrap();
        assert!(
            command_prefix::enabled(&backend),
            "real prepared-command gate requires the initialized MoE pager"
        );

        let mut graph = Graph::new();
        graph.independent_rows = rows > 1;
        if graph.independent_rows {
            graph.sequence_spans = (0..rows)
                .map(|row| infr_core::graph::SequenceSpan {
                    row_start: row as u32,
                    rows: 1,
                    start_pos: 0,
                })
                .collect();
        }
        let frontend = graph.input(TensorDesc::new([rows, ne], DType::F32));
        let one = graph.input(TensorDesc::new([rows, ne], DType::F32));
        let input = graph.input(TensorDesc::new([rows, ne], DType::F32));
        let routing = graph.input(TensorDesc::new([rows, ne], DType::F32));
        let router = graph.weight(TensorDesc::new([experts, ne], DType::F32));
        let weights = std::array::from_fn::<_, 3, _>(|role| {
            let dims = if role == 2 {
                [experts, ne, nff]
            } else {
                [experts, nff, ne]
            };
            graph.weight(TensorDesc::new(dims, dtypes[role]))
        });
        let output = graph.output(TensorDesc::new([rows, ne], DType::F32));
        graph.push(Op::Add {
            a: frontend,
            b: one,
            dst: frontend,
            n: value_count as u32,
        });
        graph.push(Op::MoeFfn {
            x: input,
            router_x: routing,
            router,
            gate_exps: weights[0],
            up_exps: weights[1],
            down_exps: weights[2],
            down_scale: None,
            fused_gate_up: false,
            dst: output,
            ne: ne as u32,
            n_expert: experts as u32,
            n_used: used as u32,
            n_ff_exp: nff as u32,
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
        let mut route = vec![0.0; value_count];
        for row in 0..rows {
            route[row * ne] = 1.0;
        }
        let mut buffers = Vec::new();
        for (id, values) in [
            (frontend, vec![0.0; value_count]),
            (one, vec![1.0; value_count]),
            (
                input,
                (0..value_count)
                    .map(|i| ((i % 17) as f32 - 8.0) / 128.0)
                    .collect(),
            ),
            (routing, route),
            (output, vec![GUARD; value_count]),
        ] {
            let buffer = backend
                .alloc((value_count + GUARD_WORDS) * 4, BufferUsage::Readback)
                .unwrap();
            let mut guarded = values;
            guarded.resize(value_count + GUARD_WORDS, GUARD);
            backend
                .upload(buffer.as_ref(), bytemuck::cast_slice(&guarded))
                .unwrap();
            buffers.push((id, buffer));
        }
        let (router_buffer, router_address) = backend
            .alloc_mapped_arena_bda((experts * ne + GUARD_WORDS) * 4)
            .unwrap();
        assert_ne!(router_address, 0);
        assert_eq!(router_buffer.device_addr(), Some(router_address));
        let mut router_values = vec![0.0; experts * ne];
        for expert in 0..experts {
            router_values[expert * ne] = (experts - expert) as f32;
        }
        router_values.resize(experts * ne + GUARD_WORDS, GUARD);
        backend
            .upload(router_buffer.as_ref(), bytemuck::cast_slice(&router_values))
            .unwrap();
        buffers.push((router, router_buffer));
        let mut roles = Vec::new();
        for (role, id) in weights.into_iter().enumerate() {
            let placeholder = backend.alloc(4, BufferUsage::Staging).unwrap();
            let identity = crate::pager::buffer_identity(placeholder.as_ref());
            backend
                .register_paged_expert(
                    [Role::Gate, Role::Up, Role::Down][role],
                    identity,
                    ExpertSource {
                        bank: banks[role].clone(),
                        stride_bytes: slot_bytes,
                        layer_base: 0,
                        host_offset: role * bank_bytes,
                        file: Some(BlockDesc {
                            id: role as u32,
                            extents: vec![BlockExtent {
                                offset: 0,
                                len: bank_bytes,
                            }],
                        }),
                    },
                    experts,
                )
                .unwrap();
            buffers.push((id, placeholder));
            roles.push(identity);
        }
        backend.finish_session_allocations().unwrap();
        Self {
            backend,
            graph,
            frontend,
            output,
            roles: roles.try_into().unwrap(),
            buffers,
            ne,
            nff,
            rows,
            used,
            routed: (0..used as u32).collect(),
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

    fn read(&self, buffer: &dyn Buffer, count: usize) -> Vec<f32> {
        let mut values = vec![0.0; count];
        self.backend
            .download(buffer, bytemuck::cast_slice_mut(&mut values))
            .unwrap();
        values
    }

    fn reset_frontend(&self) {
        self.backend
            .upload(
                self.buffer(self.frontend),
                bytemuck::cast_slice(&vec![0.0f32; self.rows * self.ne]),
            )
            .unwrap();
    }

    fn preload(&self) {
        let mut guard = self.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&self.backend).unwrap();
        session.begin_role_batches(&self.roles).unwrap();
        session
            .push_roles_cpu(
                &self.backend,
                &self.roles.map(|role| (role, self.routed.as_slice())),
                false,
            )
            .unwrap()
            .complete_without_recorder(&self.backend)
            .unwrap();
    }

    fn force_mapped_miss(&self) {
        let mut guard = self.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&self.backend).unwrap();
        assert_eq!(
            session
                .routed_roles_resident_mask(&self.roles, &self.routed)
                .unwrap(),
            (1 << self.used) - 1
        );
        session.discard_cpu_miss(self.roles, &[self.routed[0]]);
        assert_eq!(
            session
                .routed_roles_resident_mask(&self.roles, &self.routed)
                .unwrap(),
            (1 << self.used) - 2
        );
    }

    fn activation(&self) -> Vec<f32> {
        let phase = self.backend.runtime_phase.lock().unwrap();
        let buffers: Vec<_> = phase
            .pool
            .buffers
            .iter()
            .filter(|((tag, _), _)| *tag == "moe_paged_a")
            .collect();
        assert_eq!(buffers.len(), 1, "one retained ACT capacity");
        self.read(buffers[0].1.as_ref(), self.rows * self.used * self.nff)
    }

    fn poison_activation(&self) {
        let rec = self.backend.recorder().unwrap();
        rec.seed_barrier();
        {
            let phase = self.backend.runtime_phase.lock().unwrap();
            let buffers: Vec<_> = phase
                .pool
                .buffers
                .iter()
                .filter(|((tag, _), _)| *tag == "moe_paged_a")
                .collect();
            assert_eq!(buffers.len(), 1);
            rec.scale(
                buffers[0].1.as_ref(),
                f32::NAN,
                self.rows * self.used * self.nff,
            );
        }
        rec.resident_stage_handoff_barrier();
        rec.publish_host_reads();
        rec.finish().unwrap();
        assert!(
            self.activation().iter().all(|x| x.is_nan()),
            "ACT poison must take effect"
        );
    }

    fn poison_role_scratch(&self) {
        let rec = self.backend.recorder().unwrap();
        rec.seed_barrier();
        {
            let phase = self.backend.runtime_phase.lock().unwrap();
            for tag in ["moe_paged_g", "moe_paged_u", "moe_paged_a", "moe_paged_y"] {
                let buffers: Vec<_> = phase
                    .pool
                    .buffers
                    .iter()
                    .filter(|((name, _), _)| *name == tag)
                    .collect();
                assert_eq!(buffers.len(), 1, "one retained {tag} capacity");
                let width = if tag == "moe_paged_y" {
                    self.ne
                } else {
                    self.nff
                };
                rec.scale(
                    buffers[0].1.as_ref(),
                    f32::NAN,
                    self.rows * self.used * width,
                );
            }
        }
        rec.resident_stage_handoff_barrier();
        rec.publish_host_reads();
        rec.finish().unwrap();
        assert!(self.activation().iter().all(|x| x.is_nan()));
        self.backend
            .upload(
                self.buffer(self.output),
                bytemuck::cast_slice(&vec![f32::NAN; self.rows * self.ne]),
            )
            .unwrap();
    }

    fn epoch(&self) -> u64 {
        self.backend.shared.resource_epoch.load(Ordering::Acquire)
    }

    fn assert_mapped_completion(&self) {
        self.assert_completion(true);
    }

    fn assert_completion(&self, selected: bool) {
        let (journal, copy) = {
            let phase = self.backend.runtime_phase.lock().unwrap();
            let resident = phase.resident_buffers.as_ref().unwrap();
            (resident.journal.clone(), resident.gpu_copy.clone().unwrap())
        };
        assert!(copy.is_owned() && copy.owned_self_service() && !copy.owned_in_forward());
        let mut words = vec![0u32; crate::recorder::RESIDENT_JOURNAL_WORDS];
        self.backend
            .download(journal.as_ref(), bytemuck::cast_slice_mut(&mut words))
            .unwrap();
        for row in 0..self.rows {
            assert_eq!(&words[row * self.used..(row + 1) * self.used], &self.routed);
            assert_eq!(
                words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + row],
                (1 << self.used) - 2,
                "one real M per row"
            );
        }
        assert_eq!(words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD], 4);
        assert_eq!(
            words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD],
            3,
            "all three mapped roles missed"
        );
        let mut predicate = [0u32; 4];
        self.backend
            .download(
                copy.miss_predicates.as_ref().unwrap().as_ref(),
                bytemuck::cast_slice_mut(&mut predicate),
            )
            .unwrap();
        assert_eq!(predicate, [3, copy.copy_groups, 1, 1]);
        let names = self.backend.built_kernel_names();
        if selected {
            assert!(names.iter().any(|name| {
                name.starts_with("owned_role_gu_tree64_") && name.ends_with("_256_nr2")
            }));
            assert!(names.contains(&if copy.wide_copy {
                "owned_role_copy1024"
            } else {
                "owned_role_copy256"
            }));
        } else {
            assert!(!names.iter().any(|name| name.starts_with("owned_role_")));
        }
        let mut guard = self.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&self.backend).unwrap();
        assert_eq!(
            session
                .routed_roles_resident_mask(&self.roles, &self.routed)
                .unwrap(),
            (1 << self.used) - 1
        );
        assert!(!session.gpu_decode_in_forward());
    }

    fn assert_result(&self, expected: &[f32], activation: &[f32]) -> (f32, f32) {
        assert_eq!(
            self.read(self.buffer(self.frontend), self.rows * self.ne),
            vec![1.0; self.rows * self.ne],
            "frontend runs exactly once"
        );
        for (_, buffer) in self
            .buffers
            .iter()
            .filter(|(_, b)| b.len_bytes() > GUARD_WORDS * 4)
        {
            let values = self.read(buffer.as_ref(), buffer.len_bytes() / 4);
            assert!(
                values[values.len() - GUARD_WORDS..]
                    .iter()
                    .all(|&x| x == GUARD),
                "tensor guards exact"
            );
        }
        (
            compare(
                &self.read(self.buffer(self.output), self.rows * self.ne),
                expected,
            ),
            compare(&self.activation(), activation),
        )
    }
}

fn compare(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut maxerr = 0.0f32;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-6 + 1e-4 * b.abs(),
            "actual={a} expected={b}"
        );
        maxerr = maxerr.max((a - b).abs());
    }
    maxerr
}

struct ActivationTrace;

impl ActivationTrace {
    fn begin() -> Self {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            assert!(state
                .replace(Some(StagedActivationTestState {
                    events: Vec::new(),
                    setup_failure: None,
                    retirement_failure: None,
                }))
                .is_none());
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| assert!(state.replace(Some([0; 3])).is_none()));
        Self
    }

    fn assert_submitted(&self, expected: usize) {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            let state = state.borrow();
            let events = &state.as_ref().unwrap().events;
            assert_eq!(
                events
                    .iter()
                    .filter(|e| **e == StagedActivationTestEvent::Submitted)
                    .count(),
                expected,
                "{events:?}"
            );
        });
        OWNED_RECOVERY_TEST_COUNTS
            .with(|state| assert_eq!(state.get().unwrap()[0], 0, "no CPU miss recovery"));
    }
}

impl Drop for ActivationTrace {
    fn drop(&mut self) {
        STAGED_ACTIVATION_TEST_STATE.with(|state| {
            state.take();
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.set(None));
    }
}

#[test]
#[ignore = "parent serialized GPU; actual ordinary C and ACK-activated prepared C, real mapped miss, poisoned ACT"]
fn gpu_owned_role_pipeline_adapter_prepared_mapped_miss() {
    let fixture = Fixture::new();
    let _on = pipeline::test_scope(&fixture.backend, true);
    let counts = pipeline::test_selection_scope(&fixture.backend);
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.preload();
    let (expected, expected_activation) = {
        // Nested scoped false: an actual adapter old-path oracle, not fixture_dispatch.
        let _off = pipeline::test_scope(&fixture.backend, false);
        execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
        assert_eq!(counts.counts(), [0, 0]);
        assert!(!fixture
            .backend
            .built_kernel_names()
            .iter()
            .any(|name| name.starts_with("owned_role_")));
        let output = fixture.read(fixture.buffer(fixture.output), NE);
        let activation = fixture.activation();
        assert!(output.iter().all(|x| x.is_finite()) && output.iter().any(|x| x.abs() > 1e-7));
        assert!(
            activation.iter().all(|x| x.is_finite()) && activation.iter().any(|x| x.abs() > 1e-7)
        );
        assert!(
            activation[..FF].iter().any(|x| x.abs() > 1e-4),
            "miss-slot ACT oracle must distinguish a skipped GU from zero scratch"
        );
        (output, activation)
    };

    fixture.force_mapped_miss();
    fixture.reset_frontend();
    fixture.poison_activation();
    let ordinary_errors = {
        let trace = ActivationTrace::begin();
        execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
        assert_eq!(
            counts.counts(),
            [1, 0],
            "actual ordinary adapter selected C"
        );
        assert_eq!(
            counts.phase2_counts(),
            [0, 1],
            "ordinary pure-compute proof selected no-entry"
        );
        trace.assert_submitted(0);
        fixture.assert_mapped_completion();
        fixture.assert_result(&expected, &expected_activation)
    };

    fixture.reset_frontend();
    let epoch = fixture.epoch();
    execute_static_next(
        &fixture.backend,
        &current,
        &fixture.bindings(),
        Some((&next, &fixture.bindings(), &[1])),
    )
    .unwrap();
    assert_eq!(
        counts.counts(),
        [2, 1],
        "one ordinary and one speculative C recording"
    );
    assert_eq!(
        counts.phase2_counts(),
        [0, 3],
        "ordinary and prepared C preserve the no-entry proof"
    );
    {
        let packet = next.prepared_prefix.lock().unwrap();
        let packet = packet.as_ref().expect("actual complete prepared C packet");
        let (head_end, head_layers, last_end, layers, self_service, owned) =
            packet.recorded_shape();
        assert_eq!(
            (head_end, last_end),
            (fixture.graph.ops.len(), fixture.graph.ops.len())
        );
        assert_eq!((head_layers, layers), (1, 1));
        assert!(self_service && owned);
    }
    assert_eq!(
        fixture.epoch(),
        epoch,
        "preparation must not replace captured resources"
    );
    fixture.assert_result(&expected, &expected_activation);
    fixture.force_mapped_miss();
    fixture.reset_frontend();
    fixture.poison_activation();
    assert_eq!(
        fixture.epoch(),
        epoch,
        "miss and poison preserve prepared resource identity"
    );
    let before_replay = counts.counts();
    let phase2_before_replay = counts.phase2_counts();
    let trace = ActivationTrace::begin();
    execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
    trace.assert_submitted(1);
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(
        counts.counts(),
        before_replay,
        "accepted prepared C executes without ordinary re-recording"
    );
    assert_eq!(counts.phase2_counts(), phase2_before_replay);
    fixture.assert_mapped_completion();
    let prepared_errors = fixture.assert_result(&expected, &expected_activation);
    println!(
        "OWNED_ROLE_ADAPTER_PREPARED ne={NE} nff={FF} rows=1 q5k=true selections={before_replay:?} prepared_submits=1 real_mapped_jobs=3 poisoned_act_restored=true ordinary_maxerr={} ordinary_act_maxerr={} prepared_maxerr={} prepared_act_maxerr={} frontend_once=true guards_exact=true owner_drained=true",
        ordinary_errors.0, ordinary_errors.1, prepared_errors.0, prepared_errors.1
    );
}

fn assert_exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "nonfinite at {index}"
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "exact element {index}"
        );
    }
}

fn exercise_configured(fixture: Fixture, selected: bool, prepared: bool) -> (Vec<f32>, Vec<f32>) {
    let counts = pipeline::test_selection_scope(&fixture.backend);
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.preload();
    let (expected, expected_act) = {
        let _old = pipeline::test_scope(&fixture.backend, false);
        execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
        assert_eq!(counts.counts(), [0, 0]);
        assert!(!fixture
            .backend
            .built_kernel_names()
            .iter()
            .any(|name| name.starts_with("owned_role_")));
        (
            fixture.read(fixture.buffer(fixture.output), fixture.rows * fixture.ne),
            fixture.activation(),
        )
    };
    // No test override here or during preparation/replay: exercise the config toggle directly.
    fixture.force_mapped_miss();
    fixture.reset_frontend();
    fixture.poison_role_scratch();
    execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
    assert_eq!(counts.counts(), [usize::from(selected), 0]);
    assert_eq!(
        counts.phase2_counts(),
        [0, usize::from(selected)],
        "qualified H suffix preserves the no-entry proof"
    );
    fixture.assert_completion(selected);
    fixture.assert_result(&expected, &expected_act);
    assert_exact(
        &fixture.read(fixture.buffer(fixture.output), fixture.rows * fixture.ne),
        &expected,
    );
    assert_exact(&fixture.activation(), &expected_act);

    if prepared {
        fixture.reset_frontend();
        let epoch = fixture.epoch();
        execute_static_next(
            &fixture.backend,
            &current,
            &fixture.bindings(),
            Some((&next, &fixture.bindings(), &[1])),
        )
        .unwrap();
        assert_eq!(
            counts.counts(),
            [2 * usize::from(selected), usize::from(selected)]
        );
        assert_eq!(counts.phase2_counts(), [0, 3 * usize::from(selected)]);
        assert!(
            next.prepared_prefix.lock().unwrap().is_some(),
            "real captured packet required"
        );
        fixture.force_mapped_miss();
        fixture.reset_frontend();
        fixture.poison_role_scratch();
        assert_eq!(
            fixture.epoch(),
            epoch,
            "fresh scratch must preserve the captured leases"
        );
        let before = counts.counts();
        let phase2_before = counts.phase2_counts();
        let trace = ActivationTrace::begin();
        execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
        trace.assert_submitted(1);
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        assert_eq!(
            counts.counts(),
            before,
            "replay must not fall back to ordinary recording"
        );
        assert_eq!(counts.phase2_counts(), phase2_before);
        fixture.assert_completion(selected);
        fixture.assert_result(&expected, &expected_act);
        assert_exact(
            &fixture.read(fixture.buffer(fixture.output), fixture.rows * fixture.ne),
            &expected,
        );
        assert_exact(&fixture.activation(), &expected_act);
    }
    (expected, expected_act)
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; serialized adapter explicit on/off, real miss and prepared replay"]
fn gpu_owned_role_pipeline_adapter_explicit_on_off() {
    let off = exercise_configured(
        Fixture::with_profile(2048, 512, 8, 1, [DType::Q5K; 3], false),
        false,
        true,
    );
    let on = exercise_configured(
        Fixture::with_profile(2048, 512, 8, 1, [DType::Q5K; 3], true),
        true,
        true,
    );
    assert_exact(&on.0, &off.0);
    assert_exact(&on.1, &off.1);
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; serialized explicit role pipeline, Q5K rows 2/4/8 and Flash geometry"]
fn gpu_owned_role_pipeline_adapter_explicit_multirow_and_flash() {
    for rows in [2, 4, 8] {
        exercise_configured(
            Fixture::with_profile(2048, 512, 8, rows, [DType::Q5K; 3], true),
            true,
            false,
        );
    }
    exercise_configured(
        Fixture::with_profile(
            2560,
            640,
            10,
            1,
            [DType::Iq2S, DType::Iq2S, DType::Iq4Nl],
            true,
        ),
        true,
        false,
    );
}
