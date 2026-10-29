//! Small model-free resident command-ahead regressions; declared as a child of `adapter`.
#[path = "adapter_owned_stage_tail_tests.rs"]
mod owned_stage_tail_tests;
#[path = "adapter_owned_topk_tests.rs"]
mod owned_topk_tests;
#[path = "adapter_owned_upload_tests.rs"]
mod owned_upload_tests;
#[path = "adapter_prepared_empty_stage_tests.rs"]
mod prepared_empty_stage_tests;
#[path = "adapter_prepared_mapped_run_tests.rs"]
mod prepared_mapped_run_tests;
#[path = "adapter_promotion_isolation_tests.rs"]
pub(crate) mod promotion_isolation_tests;
#[path = "adapter_pure_mapped_coalesce_tests.rs"]
mod pure_mapped_coalesce_tests;
#[path = "adapter_rolling_mapped_run_tests.rs"]
mod rolling_mapped_run_tests;
#[path = "adapter_single_verify_tests.rs"]
mod single_verify_tests;
#[path = "adapter_staged_next_tests.rs"]
mod staged_next_tests;

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

fn notice_entries(required: &[bool]) -> Vec<ResidentJournalEntry> {
    required
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, host_notice)| ResidentJournalEntry {
            op_idx: ordinal * 2,
            roles: [ordinal * 3, ordinal * 3 + 1, ordinal * 3 + 2],
            layout: ResidentMoeLayout {
                rows: 1,
                n_expert: EXP,
                n_used: TOP,
            },
            host_notice,
        })
        .collect()
}

fn setup_cache_graph() -> (Graph, HcUpMixPair) {
    let mut g = Graph::new();
    let low = g.input(TensorDesc::new([320], DType::F32));
    let x = g.input(TensorDesc::new([10240], DType::F32));
    let weight = g.weight(TensorDesc::new([10240, 320], DType::Q8_0));
    let gate = g.internal(TensorDesc::new([10240], DType::F32));
    let dst = g.output(TensorDesc::new([2560], DType::F32));
    g.push(Op::Linear {
        x: low,
        weight,
        dst: gate,
        m: 1,
        in_f: 320,
        out_f: 10240,
        w_off: 0,
    });
    g.push(Op::QwenHcMix {
        x,
        gate,
        dst,
        rows: 1,
        hc: 4,
        n_embd: 2560,
    });
    let value = g.input(TensorDesc::new([2560], DType::F32));
    let norm = g.weight(TensorDesc::new([10240], DType::F32));
    let gated = g.internal(TensorDesc::new([10240], DType::F32));
    let normed = g.output(TensorDesc::new([10240], DType::F32));
    g.push(Op::QwenPleGate {
        key: x,
        query: x,
        value,
        dst: gated,
        rows: 1,
        hc: 4,
        n_embd: 2560,
    });
    g.push(Op::QwenHcNorm {
        x: gated,
        norm,
        dst: normed,
        rows: 1,
        hc: 4,
        n_embd: 2560,
        eps: 1e-6,
    });
    let positions = g.input(TensorDesc::new([1], DType::I32));
    let q = g.input(TensorDesc::new([512], DType::F32));
    let gamma = g.weight(TensorDesc::new([256], DType::F32));
    let k = g.internal(TensorDesc::new([512], DType::F16));
    let cache = g.input(TensorDesc::new([32768, 512], DType::Q8_0));
    g.push(Op::QkNormRope {
        x: q,
        weight: gamma,
        positions,
        dst: k,
        rows: 1,
        n_head: 2,
        head_dim: 256,
        rope_dim: 64,
        theta: 10000.0,
        eps: 1e-6,
        freq_factors: None,
        x_stride: 0,
    });
    g.push(Op::WriteKv {
        src: k,
        cache,
        rows: 1,
        row_stride: 512,
        pos: 0,
    });
    (
        g,
        HcUpMixPair {
            low,
            weight,
            x,
            dst,
        },
    )
}

#[test]
fn graph_setup_cached_candidates_match_original_scans_and_config() {
    for mode in 0..4 {
        let (mut graph, _) = setup_cache_graph();
        graph.independent_rows = mode == 1;
        graph.mtp_verify = mode == 2;
        if mode == 3 {
            graph.prefill_target_tokens = Some(512);
        }
        let compiled = VkDecodePlan::new_cpu(&graph, false, true);
        let cached = &compiled.setup_analysis;
        assert_eq!(
            cached.hc.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            (0..graph.ops.len())
                .filter(|&i| hc_up_mix_at(&graph, i).is_some())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            cached.ple.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            (0..graph.ops.len())
                .filter(|&i| ple_gate_norm_at(&graph, i).is_some())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            cached.delta,
            (0..graph.ops.len())
                .filter(|&i| deltanet_norm_pair(&graph, i).is_some())
                .collect::<Vec<_>>()
        );
        let facts = Q8SegmentedGraphFacts::analyze(&graph);
        assert_eq!(cached.q8_facts.written, facts.written);
        assert_eq!(cached.q8_facts.peers, facts.peers);
        assert_eq!(cached.q8_facts.dead_scratch, facts.dead_scratch);
        assert_eq!(cached.rope_ids.len(), 1);
        assert_eq!(cached.write_kv, [5]);
        for disabled in [false, true] {
            for verify in [false, true] {
                assert_eq!(
                    cached
                        .shared_candidates(disabled, verify)
                        .collect::<Vec<_>>(),
                    paged_moe_shared_candidates_with_policy(&graph, disabled, verify)
                        .collect::<Vec<_>>()
                );
            }
        }
        for enabled in [false, true] {
            let mut cached_plan = cached.fusions(enabled);
            let original = infr_core::fusion::plan_fusions(&graph, &graph_fusion_cfg(enabled));
            assert_eq!(cached_plan.kv_write, original.kv_write);
            assert_eq!(cached_plan.linear_add, original.linear_add);
            assert_eq!(cached_plan.skip, original.skip);
            cached_plan.skip.clear();
            assert_eq!(
                cached.fusions(enabled).skip,
                original.skip,
                "execute must own its mutable skip set"
            );
        }
    }
}

#[test]
fn graph_setup_hc_candidate_rechecks_changed_binding_aliases_and_addresses() {
    struct AddressBuffer(Option<u64>);
    impl Buffer for AddressBuffer {
        fn len_bytes(&self) -> usize {
            4 * 10240 * 320
        }
        fn device_addr(&self) -> Option<u64> {
            self.0
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    let (graph, pair) = setup_cache_graph();
    let compiled = VkDecodePlan::new_cpu(&graph, false, true);
    let cached = compiled.setup_analysis.hc[0].1;
    let scratch: ScratchSet = (0..graph.tensors.len()).map(|_| None).collect();
    let low = AddressBuffer(Some(0x100000));
    let x = AddressBuffer(Some(0x200000));
    let weight = AddressBuffer(Some(0x300000));
    for address in [Some(0x1000000), low.0, x.0, weight.0, None] {
        let output = AddressBuffer(address);
        let mut bindings = Bindings::new();
        bindings
            .bind(pair.low, &low)
            .bind(pair.x, &x)
            .bind(pair.weight, &weight)
            .bind(pair.dst, &output);
        let original = hc_up_mix_at(&graph, 0).unwrap();
        assert_eq!(
            hc_up_mix_buffers(cached, &scratch, &bindings).is_some(),
            hc_up_mix_buffers(original, &scratch, &bindings).is_some()
        );
        assert_eq!(
            hc_up_mix_buffers(cached, &scratch, &bindings).is_some(),
            address == Some(0x1000000)
        );
    }
}

#[test]
fn per_layer_host_notices_preserve_sparse_ordinals_and_submitted_bounds() {
    // Exhaust all short masks, including CPU-service at the first/middle/last layer.
    for mask in 0u32..128 {
        let required = (0..7).map(|i| mask & (1 << i) != 0).collect::<Vec<_>>();
        let entries = notice_entries(&required);
        let expected = required
            .iter()
            .enumerate()
            .filter_map(|(i, &yes)| yes.then_some(i))
            .collect::<Vec<_>>();
        let jobs = resident_host_notice_entries(&entries, 0, entries.len()).unwrap();
        assert_eq!(jobs.iter().map(|(i, _)| *i).collect::<Vec<_>>(), expected);
        let mut cursor = 0;
        let mut seen = Vec::new();
        for end in 0..=entries.len() {
            while let Some(ordinal) = next_resident_host_notice(&entries, &mut cursor, end).unwrap()
            {
                seen.push(ordinal);
                accept_resident_host_notice(&mut cursor, ordinal, ordinal).unwrap();
            }
        }
        assert_eq!(seen, expected);
        assert_eq!(cursor, entries.len());
        for split in 0..=entries.len() {
            let a = resident_host_notice_entries(&entries, 0, split).unwrap();
            let b = resident_host_notice_entries(&entries, split, entries.len()).unwrap();
            assert_eq!(
                a.iter().chain(&b).map(|(i, _)| *i).collect::<Vec<_>>(),
                expected
            );
        }
    }
    let entries = notice_entries(&[false; 128]);
    let mut cursor = 0;
    assert_eq!(
        next_resident_host_notice(&entries, &mut cursor, 128).unwrap(),
        None
    );
    assert_eq!(cursor, 128);
    assert!(next_resident_host_notice(&entries, &mut cursor, 127).is_err());
    assert!(resident_host_notice_entries(&entries, 2, 1).is_err());
    assert!(resident_host_notice_entries(&entries, 0, 129).is_err());
    assert!(accept_resident_host_notice(&mut cursor, 17, 16).is_err());
    assert_eq!(
        cursor, 18,
        "an invalid result was consumed and must not be awaited twice"
    );
}

#[test]
fn sparse_notice_drop_drains_errors_and_acks_without_waiting_for_gpu_only_layers() {
    for cpu_layer in 0..3 {
        for failed in [false, true] {
            let (output, results) = mpsc::channel();
            let announced = Arc::new(AtomicUsize::new(cpu_layer));
            let producer = Arc::new(ResidentNoticeProducer {
                host_ready: None,
                sender: None,
                results: Mutex::new(results),
                thread: None,
                announced_miss: Arc::clone(&announced),
                pager: Arc::new(Mutex::new(None)),
            });
            let mut required = [false; 3];
            required[cpu_layer] = true;
            let (done, ack) = mpsc::channel();
            let chunk = ResidentChunk {
                control: Arc::new(CpuBuffer(crate::recorder::RESIDENT_CONTROL_BYTES)),
                journal: Arc::new(CpuBuffer(3 * crate::recorder::RESIDENT_JOURNAL_BYTES)),
                entries: notice_entries(&required),
                early_recovery: false,
                recovery_op: None,
                limit: 3,
                recording: false,
                self_service: false,
                submit_layers: 1,
                submitted_entries: 3,
                pure_mapped_schedule: None,
                notices: None,
                notice_cursor: 0,
                notice_resume: None,
                notice_stop: false,
                producer: Some(Arc::clone(&producer)),
                notices_sent: 3,
                owned_notice_acks: vec![ack],
                mapped_serial_max_bytes: 0,
                waiter: None,
                indirect: None,
                gpu_copy: None,
            };
            let result = if failed {
                Err(be("synthetic service failure"))
            } else {
                Ok(ProducedNotice {
                    words: [0; crate::recorder::RESIDENT_JOURNAL_WORDS],
                    promotion: None,
                    owned_cpu: false,
                })
            };
            assert!(output.send((cpu_layer, result)).is_ok());
            done.send(()).unwrap();
            // Keep output connected: a mistaken receive for a GPU-only hole would block.
            drop(chunk);
            assert!(matches!(
                producer.results.lock().unwrap().try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
            assert_eq!(announced.load(Ordering::Acquire), usize::MAX);
            drop(output);
        }
    }
}

#[test]
fn sparse_notice_dirty_reset_preserves_failed_and_future_events() {
    let dirty = ResidentNoticeDirty::default();
    for ordinal in [0, 2, 64, 127] {
        dirty.mark(ordinal);
    }
    let mut reset = Vec::new();
    dirty
        .reset_with(128, |ordinal| {
            reset.push(ordinal);
            Ok(())
        })
        .unwrap();
    assert_eq!(reset, [0, 2, 64, 127]);
    dirty
        .reset_with(128, |_| panic!("clean events must not be reset again"))
        .unwrap();
    for ordinal in [1, 3, 65] {
        dirty.mark(ordinal);
    }
    let mut reset = Vec::new();
    assert!(dirty
        .reset_with(128, |ordinal| {
            if ordinal == 3 {
                return Err(be("synthetic reset failure"));
            }
            reset.push(ordinal);
            Ok(())
        })
        .is_err());
    assert_eq!(reset, [1]);
    reset.clear();
    dirty
        .reset_with(128, |ordinal| {
            reset.push(ordinal);
            Ok(())
        })
        .unwrap();
    assert_eq!(reset, [3, 65]);
    dirty.mark(127);
    assert!(dirty.reset_with(127, |_| Ok(())).is_err());
    reset.clear();
    dirty
        .reset_with(128, |ordinal| {
            reset.push(ordinal);
            Ok(())
        })
        .unwrap();
    assert_eq!(reset, [127]);
}

#[test]
fn owned_execution_window_and_preparation_budget_are_independent() {
    for layers in [1, 16, 17, 48, 128] {
        assert_eq!(
            resident_execution_limit(true, layers, 16, 128).unwrap(),
            layers
        );
        assert_eq!(
            resident_execution_limit(false, layers, 16, 128).unwrap(),
            16
        );
        assert!(resident_execution_limit(true, layers, 16, layers - 1).is_err());
        let budget = command_prefix::partial_preparation_limit(16, layers);
        assert_eq!(budget, layers.min(16));
        assert_eq!(command_prefix::partial_preparation_limit(1, layers), 1);
        assert_eq!(
            command_prefix::partial_preparation_limit(48, layers),
            layers.min(48)
        );
        for cadence in [1, 2] {
            for recorded in 1..=layers {
                assert_eq!(
                    resident_submit_due(recorded, 0, cadence, layers),
                    recorded < layers && recorded >= cadence.min(layers),
                );
            }
        }
    }
    assert!(resident_execution_limit(true, 0, 16, 128).is_err());
    assert!(resident_execution_limit(false, 48, 16, 15).is_err());
    assert_eq!(resident_execution_limit(true, 129, 16, 128).unwrap(), 128);
    assert_eq!(command_prefix::partial_preparation_limit(0, 48), 1);
    assert_eq!(
        command_prefix::partial_preparation_limit(usize::MAX, 128),
        128
    );
}

#[test]
fn owned_execution_window_checks_actual_protocol_buffer_capacities() {
    let job_layer_bytes =
        crate::recorder::RESIDENT_JOURNAL_ID_WORDS * 3 * std::mem::size_of::<[u32; 8]>();
    let predicate_layer_bytes = crate::recorder::GPU_COPY_PREDICATE_WORDS * 4;
    for layers in [1, 16, 17, 48, 128] {
        let control = (6 + layers * 2) * 4;
        let journal = layers * crate::recorder::RESIDENT_JOURNAL_BYTES;
        let jobs = layers * job_layer_bytes;
        let predicates = layers * predicate_layer_bytes;
        let capacity = resident_owned_protocol_capacity(
            control,
            true,
            journal,
            layers,
            jobs,
            Some(predicates),
        );
        assert_eq!(capacity, layers);
        assert_eq!(
            resident_owned_protocol_capacity(control, true, journal, layers, jobs, None),
            layers,
        );
        for reduced in 0..5 {
            let actual = resident_owned_protocol_capacity(
                if reduced == 0 { control - 1 } else { control },
                true,
                if reduced == 1 { journal - 1 } else { journal },
                if reduced == 2 { layers - 1 } else { layers },
                if reduced == 3 { jobs - 1 } else { jobs },
                Some(if reduced == 4 {
                    predicates - 1
                } else {
                    predicates
                }),
            );
            assert_eq!(actual, layers - 1, "truncated protocol component {reduced}");
            assert!(resident_execution_limit(true, layers, 16, actual).is_err());
        }
        assert_eq!(
            resident_owned_protocol_capacity(20, false, journal, layers, jobs, None),
            layers,
        );
        assert_eq!(
            resident_owned_protocol_capacity(19, false, journal, layers, jobs, None),
            0,
        );
    }
}

#[test]
fn resident_ahead_snapshot_drop_preserves_live_miss_announcement() {
    let (_output, results) = mpsc::channel();
    let announced = Arc::new(AtomicUsize::new(1));
    let producer = Arc::new(ResidentNoticeProducer {
        host_ready: None,
        sender: None,
        results: Mutex::new(results),
        thread: None,
        announced_miss: Arc::clone(&announced),
        pager: Arc::new(Mutex::new(None)),
    });
    let chunk = ResidentChunk {
        control: Arc::new(CpuBuffer(crate::recorder::RESIDENT_CONTROL_BYTES)),
        journal: Arc::new(CpuBuffer(3 * crate::recorder::RESIDENT_JOURNAL_BYTES)),
        entries: Vec::new(),
        early_recovery: true,
        recovery_op: None,
        limit: 3,
        recording: true,
        self_service: false,
        submit_layers: 1,
        submitted_entries: 0,
        pure_mapped_schedule: None,
        notices: None,
        notice_cursor: 0,
        notice_resume: None,
        notice_stop: false,
        producer: Some(Arc::clone(&producer)),
        notices_sent: 0,
        owned_notice_acks: Vec::new(),
        mapped_serial_max_bytes: 0,
        waiter: None,
        indirect: None,
        gpu_copy: None,
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

fn drained_owner_epoch(fixture: &Fixture) -> (Arc<crate::pager::GpuCopyBuffers>, u64) {
    let copy = fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .gpu_decode_owner
        .clone()
        .expect("active GPU owner");
    assert!(!copy.owned_in_forward());
    let mut header = [0u32; 8];
    fixture
        .backend
        .download(
            copy.candidates.as_ref(),
            bytemuck::cast_slice_mut(&mut header),
        )
        .unwrap();
    assert_eq!(header[0], 4);
    (copy, u64::from(header[3]) | (u64::from(header[4]) << 32))
}

impl Fixture {
    fn new(submit_layers: usize, producer: bool) -> Self {
        Self::new_with_copy_policy(submit_layers, producer, Some(false))
    }

    fn new_with_copy_policy(submit_layers: usize, producer: bool, copy: Option<bool>) -> Self {
        Self::new_with_copy_policy_and_kv(submit_layers, producer, copy, 0, 0)
    }

    fn new_with_copy_policy_and_kv(
        submit_layers: usize,
        producer: bool,
        copy: Option<bool>,
        kv_reserve_bytes: usize,
        kv_segment_bytes: usize,
    ) -> Self {
        Self::new_with_copy_policy_and_kv_ready(
            submit_layers,
            producer,
            copy,
            kv_reserve_bytes,
            kv_segment_bytes,
            false,
        )
    }

    fn new_with_copy_policy_and_kv_ready(
        submit_layers: usize,
        producer: bool,
        copy: Option<bool>,
        kv_reserve_bytes: usize,
        kv_segment_bytes: usize,
        host_ready: bool,
    ) -> Self {
        Self::new_with_preparation_budget(
            submit_layers,
            producer,
            copy,
            kv_reserve_bytes,
            kv_segment_bytes,
            host_ready,
            3,
        )
    }

    fn new_with_preparation_budget(
        submit_layers: usize,
        producer: bool,
        copy: Option<bool>,
        kv_reserve_bytes: usize,
        kv_segment_bytes: usize,
        host_ready: bool,
        preparation_layers: usize,
    ) -> Self {
        Self::new_with_combined_policy(
            submit_layers,
            producer,
            copy,
            kv_reserve_bytes,
            kv_segment_bytes,
            host_ready,
            preparation_layers,
            false,
        )
    }

    fn new_with_combined_policy(
        submit_layers: usize,
        producer: bool,
        copy: Option<bool>,
        kv_reserve_bytes: usize,
        kv_segment_bytes: usize,
        host_ready: bool,
        preparation_layers: usize,
        copy_overlap: bool,
    ) -> Self {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.submit_dispatches = Some(0);
        cfg.device.decode_command_ahead = true;
        cfg.device.decode_command_early_submit = true;
        cfg.device.decode_command_resident = true;
        cfg.paging.expert_prefetch = false;
        cfg.paging.host_dma = true;
        cfg.kernels.vulkan.decode_gpu_resident_host_ready = host_ready;
        cfg.kernels.vulkan.decode_gpu_resident_copy_overlap = copy_overlap;
        cfg.kernels.vulkan.decode_gpu_resident_upload_worker = true;
        cfg.paging.ring = Some(SizeSpec::Bytes((4 * SLOT_BYTES) as u64));
        cfg.kernels.vulkan.cpu_miss_threads = 0;
        cfg.kernels.vulkan.static_scratch_memo = true;
        cfg.kernels.vulkan.decode_gpu_resident = true;
        cfg.kernels.vulkan.decode_gpu_resident_copy = copy;
        cfg.kernels.vulkan.decode_gpu_resident_layers = preparation_layers;
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
                dynamic_state_reserve_bytes: kv_reserve_bytes as u64,
                dynamic_state_max_allocation_bytes: kv_segment_bytes as u64,
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
        session.handoff_gpu_decode(&self.backend).unwrap();
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
        // Keep the legacy whole-batch discard/recovery coverage; the source-horizon test
        // below compares both policies using the same real preparation worker.
        let _legacy = command_prefix::source_horizon_scope(&self.backend, false);
        let current = VkDecodePlan::new_cpu(&self.graph, false, true);
        let mut next_graph = self.graph.clone();
        let next_position = if next_graph.independent_rows {
            assert_eq!(next_graph.sequence_spans.len(), 1);
            let span = &mut next_graph.sequence_spans[0];
            span.start_pos = span.start_pos.checked_add(1).unwrap();
            span.start_pos as usize
        } else {
            1
        };
        let next = VkDecodePlan::new_cpu(&next_graph, false, true);
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
            if resident
                .gpu_copy
                .as_ref()
                .is_none_or(|copy| !copy.is_owned())
            {
                let notices = resident.notices.as_ref().expect("fixture event notices");
                for ordinal in 0..LAYERS {
                    assert_ne!(
                        notices.dirty.words[ordinal / 64].load(Ordering::Acquire)
                            & (1u64 << (ordinal % 64)),
                        0,
                        "live legacy notice {ordinal} must be dirty before prepared reuse"
                    );
                }
            }
        }
        self.reset_frontend();
        let epoch = self.backend.shared.resource_epoch.load(Ordering::Acquire);
        execute_static_next(
            &self.backend,
            &current,
            &bindings,
            Some((&next, &bindings, &[next_position])),
        )
        .unwrap();
        {
            let packet = next.prepared_prefix.lock().unwrap();
            let packet = packet
                .as_ref()
                .expect("actual command preparation, not calibration fallback");
            let (head_end, head_layers, last_end, layers, self_service, owned) =
                packet.recorded_shape();
            // Preparation spans the graph, but CPU-service submissions keep their old cadence.
            assert_eq!(
                packet.end,
                if owned {
                    self.graph.ops.len()
                } else {
                    submit_layers * 2
                }
            );
            assert_eq!(
                head_end,
                if self_service {
                    packet.end
                } else {
                    submit_layers * 2
                }
            );
            assert_eq!(
                head_layers,
                if self_service {
                    self.roles.len()
                } else {
                    submit_layers
                }
            );
            assert_eq!(last_end, packet.end);
            assert_eq!(
                layers,
                if owned {
                    self.roles.len()
                } else {
                    submit_layers
                }
            );
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
            session.handoff_gpu_decode(&self.backend).unwrap();
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
        // Validate ownership even for all-hit reuse: correct numeric output alone can hide
        // duplicate LRU owners created by a prepared packet clearing the seeded directory.
        let mut guard = self.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&self.backend).unwrap();
        for roles in &self.roles {
            assert_eq!(
                session.routed_roles_resident_mask(roles, &ROUTED).unwrap(),
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
#[ignore = "parent serial GPU; private prepared setup/op/finalize cancellation and ready-prefix retention"]
fn gpu_resident_prepared_cooperative_cancellation_preserves_forward_and_fallback() {
    use command_prefix::{RecordingCheckpoint, StageRecording, TestCancelAt};
    let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let extra_frontend = fixture.graph.ops[0].clone();
    fixture.graph.ops.insert(1, extra_frontend);
    let current = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    let bindings = fixture.bindings();
    fixture.preload();
    execute_static(&fixture.backend, &current, &bindings).unwrap();
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 3.0);
    let expected = fixture.outputs();
    let tape = fixture.tape();
    let epoch = fixture
        .backend
        .shared
        .resource_epoch
        .load(Ordering::Acquire);

    for (next_token, point) in [false, true].into_iter().flat_map(|next_token| {
        [
            RecordingCheckpoint::Setup,
            RecordingCheckpoint::Op(1),
            RecordingCheckpoint::Finalize,
        ]
        .map(|point| (next_token, point))
    }) {
        let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
        let cancelled = AtomicBool::new(false);
        {
            let mut arena = fixture.backend.runtime_phase.lock().unwrap();
            let RuntimePhaseArena { scratch, pool, .. } = &mut *arena;
            let job = if next_token {
                StageRecording::snapshot_next_token(
                    &fixture.backend,
                    &next,
                    &bindings,
                    scratch,
                    pool,
                    &[1],
                )
            } else {
                StageRecording::snapshot(
                    &fixture.backend,
                    &next,
                    &bindings,
                    scratch,
                    pool,
                    None,
                    &HashMap::new(),
                    &next.prepared_prefix,
                )
            }
            .unwrap()
            .expect("real optional command preparation");
            let _cancel = TestCancelAt::enter(point);
            job.record(&fixture.backend, &cancelled).unwrap();
        }
        assert!(
            cancelled.load(Ordering::Acquire),
            "must reach {point:?}, next_token={next_token}"
        );
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 3.0);
        assert_eq!(
            fixture.outputs(),
            expected,
            "unsubmitted work must not execute"
        );
        assert_eq!(fixture.tape(), tape);
        assert_eq!(
            fixture
                .backend
                .shared
                .resource_epoch
                .load(Ordering::Acquire),
            epoch
        );
        fixture.reset_frontend();
        execute_static(&fixture.backend, &next, &bindings).unwrap();
        fixture.assert_frontend(fixture.buffer(fixture.frontend), 3.0);
        assert_close(&fixture.outputs(), &expected);
    }

    // Complete a real prefix before the stop signal; successful scope completion keeps it.
    let next = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    let next_cancelled = AtomicBool::new(false);
    let continuation_cancelled = AtomicBool::new(false);
    {
        let mut arena = fixture.backend.runtime_phase.lock().unwrap();
        let RuntimePhaseArena { scratch, pool, .. } = &mut *arena;
        let job = StageRecording::snapshot_next_token(
            &fixture.backend,
            &next,
            &bindings,
            scratch,
            pool,
            &[1],
        )
        .unwrap()
        .expect("real completed prefix");
        job.record(&fixture.backend, &next_cancelled).unwrap();
    }
    assert!(next.prepared_prefix.lock().unwrap().is_some());
    command_prefix::test_finish_preparation_scope(
        &next_cancelled,
        &continuation_cancelled,
        false,
        true,
    );
    assert!(
        next.prepared_prefix.lock().unwrap().is_some(),
        "ready packet survives stop"
    );
    fixture.reset_frontend();
    execute_static(&fixture.backend, &next, &bindings).unwrap();
    assert!(
        next.prepared_prefix.lock().unwrap().is_none(),
        "ready packet is consumed"
    );
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 3.0);
    assert_close(&fixture.outputs(), &expected);
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

#[test]
#[ignore = "requires an idle Vulkan GPU; model-free auto-copy/prepared transition"]
fn gpu_resident_copy_prepared_prefix_preserves_owned_directory() {
    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    for missing in [None, Some(0), Some(1), None] {
        fixture.run_case(1, true, missing);
        let guard = fixture.backend.moe_pager().lock().unwrap();
        assert!(guard
            .as_ref()
            .unwrap()
            .gpu_decode_owner
            .as_ref()
            .is_some_and(|copy| copy.is_owned()));
    }
}

#[test]
#[ignore = "central GPU runner only; sparse notices with CPU service first/middle/last"]
fn gpu_owned_sparse_notice_cpu_service_at_first_middle_last() {
    for cpu_layer in 0..3 {
        let mut fixture = Fixture::new_with_copy_policy(1, true, Some(true));
        let frontend = fixture.graph.ops[0].clone();
        let mapped_moe = fixture.graph.ops[1].clone();
        let cpu_moe = fixture.graph.ops[3].clone();
        let Op::MoeFfn { x: mut input, .. } = mapped_moe.clone() else {
            unreachable!()
        };
        let extra = fixture.graph.output(TensorDesc::new([1, NE], DType::F32));
        fixture.buffers.push((
            extra,
            fixture
                .backend
                .alloc(NE * 4, BufferUsage::Readback)
                .unwrap(),
        ));
        let outputs = [fixture.outputs[0], fixture.outputs[1], extra];
        fixture.graph.ops.clear();
        for (layer, &output) in outputs.iter().enumerate() {
            fixture.graph.push(frontend.clone());
            let mut moe = if layer == cpu_layer {
                cpu_moe.clone()
            } else {
                mapped_moe.clone()
            };
            if let Op::MoeFfn { x, dst, .. } = &mut moe {
                *x = input;
                *dst = output;
            }
            fixture.graph.push(moe);
            input = output;
        }
        let plan = fixture.backend.compile(&fixture.graph).unwrap();
        fixture.preload();
        fixture
            .backend
            .execute(plan.as_ref(), &fixture.bindings())
            .unwrap();
        let expected = outputs
            .iter()
            .map(|&id| fixture.read(fixture.buffer(id)))
            .collect::<Vec<_>>();
        let (mut sources, seed, cpu_bases, mapped_bases) = {
            let mut guard = fixture.backend.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            let previous = session.gpu_decode_owner.clone().unwrap();
            session.handoff_gpu_decode(&fixture.backend).unwrap();
            let cpu = previous.role_params(session, &fixture.roles[1]).unwrap();
            let mapped = previous.role_params(session, &fixture.roles[0]).unwrap();
            session.discard_cpu_miss(fixture.roles[1], &[ROUTED[1]]);
            session.gpu_decode_owner = None;
            (
                session.gpu_copy_sources().unwrap().unwrap(),
                crate::pager::owned_seed(session, 0).unwrap(),
                [cpu[0], cpu[1], cpu[2]],
                [mapped[0], mapped[1], mapped[2]],
            )
        };
        for base in cpu_bases {
            sources.test_cpu_only_source(base as usize + ROUTED[1] as usize);
        }
        let owner = crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, 3, seed)
            .unwrap()
            .unwrap();
        assert!(!owner.owned_self_service());
        assert!(owner.owned_roles_self_service(mapped_bases, EXP as u32));
        assert!(!owner.owned_roles_self_service(cpu_bases, EXP as u32));
        fixture
            .backend
            .moe_pager()
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .gpu_decode_owner = Some(owner.clone());
        fixture
            .backend
            .runtime_phase
            .lock()
            .unwrap()
            .resident_buffers
            .as_mut()
            .unwrap()
            .gpu_copy = Some(owner.clone());
        for _ in 0..2 {
            fixture.reset_frontend();
            fixture
                .backend
                .execute(plan.as_ref(), &fixture.bindings())
                .unwrap();
            fixture.assert_frontend(fixture.buffer(fixture.frontend), 3.0);
            let actual = outputs
                .iter()
                .map(|&id| fixture.read(fixture.buffer(id)))
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected,
                "sparse recovery must accumulate hit/miss output once"
            );
        }
        fixture.backend.with_unified_exclusive(|| {
            let mut guard = fixture.backend.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            assert!(Arc::ptr_eq(
                session.gpu_decode_owner.as_ref().unwrap(),
                &owner
            ));
            session.handoff_gpu_decode(&fixture.backend).unwrap();
            assert_eq!(
                session
                    .routed_roles_resident_mask(&fixture.roles[1], &ROUTED)
                    .unwrap(),
                HIT_MASK
            );
        });
    }
}

#[test]
#[ignore = "parent serial GPU; source-certified pre-lowering horizon versus legacy 16-layer packet"]
fn gpu_prepared_source_certificate_horizon_preserves_execution_and_recovery() {
    const WINDOW: usize = 16;
    for cadence in [1, 2] {
        // WINDOW means an unmapped bank exists outside the recorded graph: this owner
        // is mixed, but its wholly mapped 16-layer prefix must NOT be shortened.
        for cpu_layer in [None, Some(0), Some(7), Some(15), Some(WINDOW)] {
            let mut fixture = Fixture::new_with_preparation_budget(
                cadence,
                true,
                Some(true),
                0,
                0,
                false,
                WINDOW,
            );
            // Exercise the same host-grid cutoff both with and without pure-run grouping.
            Arc::make_mut(&mut fixture.backend.cfg)
                .kernels
                .vulkan
                .decode_gpu_resident_pure_mapped_layers = if cadence == 2 { 4 } else { 0 };
            let frontend = fixture.graph.ops[0].clone();
            let mapped_moe = fixture.graph.ops[1].clone();
            let cpu_moe = fixture.graph.ops[3].clone();
            let Op::MoeFfn { x: input, .. } = mapped_moe.clone() else {
                unreachable!()
            };
            let mut outputs = Vec::new();
            fixture.graph.ops.clear();
            for layer in 0..WINDOW {
                let output = fixture.graph.output(TensorDesc::new([1, NE], DType::F32));
                fixture.buffers.push((
                    output,
                    fixture
                        .backend
                        .alloc(NE * 4, BufferUsage::Readback)
                        .unwrap(),
                ));
                let mut moe = if cpu_layer == Some(layer) {
                    cpu_moe.clone()
                } else {
                    mapped_moe.clone()
                };
                if let Op::MoeFfn { x, dst, .. } = &mut moe {
                    *x = input;
                    *dst = output;
                }
                fixture.graph.push(frontend.clone());
                fixture.graph.push(moe);
                outputs.push(output);
            }
            let make_plan = || {
                let mut plan = VkDecodePlan::new_cpu(&fixture.graph, false, true);
                plan.phase_requests = Some(
                    stages::compiled_workspace_requests(&fixture.backend, &fixture.graph).unwrap(),
                );
                plan
            };
            let current = make_plan();
            fixture.preload();
            fixture.reset_frontend();
            execute_static(&fixture.backend, &current, &fixture.bindings()).unwrap();
            let read_outputs = || {
                outputs
                    .iter()
                    .map(|&id| fixture.read(fixture.buffer(id)))
                    .collect::<Vec<_>>()
            };
            let expected = read_outputs();
            assert!(expected
                .iter()
                .flatten()
                .all(|value| value.is_finite() && *value > 0.0));

            // Keep the exact imported-source leases, changing only one bank's GPU-readable
            // certificate. Residency initially remains all-hit; a later trial forces status 5.
            let (sources, seed) = {
                let mut guard = fixture.backend.moe_pager().lock().unwrap();
                let session = guard.as_mut().unwrap();
                let previous = session.gpu_decode_owner.clone().unwrap();
                session.handoff_gpu_decode(&fixture.backend).unwrap();
                let params = previous.role_params(session, &fixture.roles[1]).unwrap();
                let mut sources = session.gpu_copy_sources().unwrap().unwrap();
                if cpu_layer.is_some() {
                    for base in &params[..3] {
                        sources.test_cpu_only_source(*base as usize + ROUTED[1] as usize);
                    }
                }
                session.gpu_decode_owner = None;
                (sources, crate::pager::owned_seed(session, 0).unwrap())
            };
            let owner =
                crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, WINDOW, seed)
                    .unwrap()
                    .unwrap();
            assert_eq!(owner.owned_self_service(), cpu_layer.is_none());
            fixture
                .backend
                .moe_pager()
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .gpu_decode_owner = Some(owner.clone());
            fixture
                .backend
                .runtime_phase
                .lock()
                .unwrap()
                .resident_buffers
                .as_mut()
                .unwrap()
                .gpu_copy = Some(owner.clone());
            let mut legacy_parts = Vec::new();
            for enabled in [false, true] {
                let _horizon = command_prefix::source_horizon_scope(&fixture.backend, enabled);
                for miss in [false, true] {
                    if miss && cpu_layer.is_none_or(|layer| layer >= WINDOW) {
                        continue;
                    }
                    fixture.preload();
                    fixture.reset_frontend();
                    let next = make_plan();
                    execute_static_next(
                        &fixture.backend,
                        &current,
                        &fixture.bindings(),
                        Some((&next, &fixture.bindings(), &[1])),
                    )
                    .unwrap();
                    let layers = if enabled {
                        cpu_layer.map_or(WINDOW, |layer| {
                            ((layer + 1).div_ceil(cadence) * cadence).min(WINDOW)
                        })
                    } else {
                        WINDOW
                    };
                    {
                        let packet = next.prepared_prefix.lock().unwrap();
                        let packet = packet.as_ref().expect("actual isolated-worker preparation");
                        assert_eq!(
                            packet.resident_layers(),
                            layers,
                            "must shorten RECORDING, not submission only"
                        );
                        assert_eq!(packet.end, layers * 2);
                        let parts = packet.coalesced_test_parts();
                        if !enabled {
                            legacy_parts = parts;
                        } else {
                            assert_eq!(
                                parts,
                                legacy_parts
                                    .iter()
                                    .copied()
                                    .take_while(|&end| end <= layers)
                                    .collect::<Vec<_>>(),
                                "retained groups keep their original boundaries"
                            );
                        }
                    }
                    let epoch = fixture
                        .backend
                        .shared
                        .resource_epoch
                        .load(Ordering::Acquire);
                    let tape = fixture.tape();
                    {
                        let arena = fixture.backend.runtime_phase.lock().unwrap();
                        assert_eq!(arena.resident_buffers.as_ref().unwrap().capacity, WINDOW);
                    }
                    assert_eq!(
                        fixture
                            .backend
                            .cfg()
                            .kernels
                            .vulkan
                            .decode_gpu_resident_submit_layers,
                        cadence
                    );
                    assert_eq!(
                        fixture
                            .backend
                            .cfg()
                            .kernels
                            .vulkan
                            .decode_gpu_resident_inflight,
                        1
                    );
                    if miss {
                        let mut guard = fixture.backend.moe_pager().lock().unwrap();
                        let session = guard.as_mut().unwrap();
                        session.handoff_gpu_decode(&fixture.backend).unwrap();
                        session.discard_cpu_miss(fixture.roles[1], &[ROUTED[1]]);
                        assert_ne!(
                            session
                                .routed_roles_resident_mask(&fixture.roles[1], &ROUTED)
                                .unwrap(),
                            HIT_MASK
                        );
                    }
                    fixture.reset_frontend();
                    let accepted = command_prefix::test_accepted_prefixes();
                    execute_static(&fixture.backend, &next, &fixture.bindings()).unwrap();
                    assert_eq!(
                        command_prefix::test_accepted_prefixes(),
                        accepted + 1,
                        "must execute the prepared path, not silently fall back"
                    );
                    assert!(next.prepared_prefix.lock().unwrap().is_none());
                    fixture.assert_frontend(fixture.buffer(fixture.frontend), WINDOW as f32);
                    assert_eq!(read_outputs(), expected);
                    assert_eq!(
                        fixture
                            .backend
                            .shared
                            .resource_epoch
                            .load(Ordering::Acquire),
                        epoch
                    );
                    // All-hit replay preserves the same initialized LUT windows. Recovery may
                    // legally reuse coordinates, but must retire every owner/notice lease.
                    if !miss {
                        assert_eq!(fixture.tape(), tape);
                    }
                    let mut guard = fixture.backend.moe_pager().lock().unwrap();
                    let session = guard.as_mut().unwrap();
                    assert!(Arc::ptr_eq(
                        session.gpu_decode_owner.as_ref().unwrap(),
                        &owner
                    ));
                    session.handoff_gpu_decode(&fixture.backend).unwrap();
                    for roles in &fixture.roles {
                        assert_eq!(
                            session.routed_roles_resident_mask(roles, &ROUTED).unwrap(),
                            HIT_MASK
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "central GPU runner only; staged owned workspace and failed callback lifetime"]
fn gpu_staged_callback_failure_drains_without_replaying_prefix() {
    use infr_core::backend::ExecutionStage;

    let fixture = Fixture::new_with_copy_policy(1, true, Some(true));
    let bindings = fixture.bindings();
    let full = VkDecodePlan::new_cpu(&fixture.graph, false, true);
    fixture.preload();
    execute_static(&fixture.backend, &full, &bindings).unwrap();
    let expected = fixture.outputs();

    let mut head_graph = fixture.graph.clone();
    head_graph.ops.truncate(2);
    let mut tail_graph = fixture.graph.clone();
    tail_graph.ops.drain(..2);
    let head = VkDecodePlan::new_cpu(&head_graph, false, true);
    let tail = VkDecodePlan::new_cpu(&tail_graph, false, true);
    let stages = [
        ExecutionStage::new(&head, &bindings),
        ExecutionStage::new(&tail, &bindings),
    ];
    let sentinel = vec![-17.0f32; NE];
    fixture
        .backend
        .upload(
            fixture.buffer(fixture.outputs[1]),
            bytemuck::cast_slice(&sentinel),
        )
        .unwrap();
    fixture.reset_frontend();
    let caller = std::thread::current().id();
    let mut callbacks = Vec::new();
    let error = fixture
        .backend
        .execute_staged(
            &stages,
            &mut |index| {
                assert_eq!(std::thread::current().id(), caller);
                assert!(fixture.backend.runtime_phase.try_lock().is_ok());
                assert!(fixture.backend.moe_pager().try_lock().is_ok());
                assert!(staged_continuity_supported(
                    &fixture.backend,
                    &[&head, &tail]
                ));
                callbacks.push(index);
                if index == 1 {
                    return Err(be("synthetic tail input failure"));
                }
                Ok(())
            },
            None,
        )
        .unwrap_err();
    assert!(error.to_string().contains("synthetic tail input failure"));
    assert_eq!(callbacks, [0, 1]);
    // All submitted readers must have drained before inspecting or reusing these allocations.
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 1.0);
    assert_close(
        &[fixture.read(fixture.buffer(fixture.outputs[0]))],
        &expected[..1],
    );
    assert_eq!(fixture.read(fixture.buffer(fixture.outputs[1])), sentinel);
    fixture.backend.with_unified_exclusive(|| {
        fixture
            .backend
            .moe_pager()
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .handoff_gpu_decode(&fixture.backend)
            .unwrap();
    });

    // Re-enter using the same plans/buffers after dropping the unsubmitted tail packet.
    fixture.reset_frontend();
    callbacks.clear();
    fixture
        .backend
        .execute_staged(
            &stages,
            &mut |index| {
                callbacks.push(index);
                Ok(())
            },
            None,
        )
        .unwrap();
    assert_eq!(callbacks, [0, 1]);
    fixture.assert_frontend(fixture.buffer(fixture.frontend), 2.0);
    assert_close(&fixture.outputs(), &expected);
}

#[test]
#[ignore = "central GPU runner only; single independent-lane prepared prefix and recovery"]
fn gpu_resident_ahead_single_independent_lane_prepares_and_recovers() {
    for copy in [Some(false), Some(true)] {
        let mut fixture = Fixture::new_with_copy_policy(1, true, copy);
        fixture.graph.independent_rows = true;
        fixture.graph.sequence_spans = vec![SequenceSpan {
            row_start: 0,
            rows: 1,
            start_pos: 37,
        }];
        for missing in [None, Some(0), Some(1), None] {
            fixture.run_case(1, true, missing);
        }
        fixture.binding_invalidation(1, true);
    }
}

#[test]
fn resident_notice_trace_preserves_shape_and_marks_compact_payload_absent() {
    for rows in [1, 2, 4, 8] {
        for used in [1, 10, 16] {
            let layout = ResidentMoeLayout {
                rows,
                n_expert: 128,
                n_used: used,
            };
            let mut words = std::array::from_fn(|i| i as u32 + 17);
            for status in [0, 1, 2, 3, 4, 5, u32::MAX] {
                words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = status;
                for compact in [false, true] {
                    let trace = resident_notice_trace(
                        infr_core::timeline::Context {
                            step: 7,
                            lanes: rows,
                            layer: 9,
                        },
                        13,
                        &words,
                        layout,
                        compact,
                    );
                    assert_eq!(trace.status, status);
                    assert_eq!(trace.ordinal, 13);
                    assert_eq!(trace.count, rows * used);
                    assert_eq!(trace.context.lanes, rows);
                    assert_eq!(trace.hit_masks.len(), infr_core::backend::MAX_DECODE_ROWS);
                    let payload = !compact || !matches!(status, 0 | 1 | 4);
                    assert_eq!(trace.route_payload_present, payload);
                    if payload {
                        assert_eq!(trace.routed_ids, words[..rows * used]);
                        assert_eq!(
                            trace.hit_masks,
                            std::array::from_fn(|i| {
                                if i < rows {
                                    words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + i]
                                } else {
                                    0
                                }
                            })
                        );
                    } else {
                        assert!(trace.routed_ids.is_empty());
                        assert_eq!(trace.hit_masks, [0; infr_core::backend::MAX_DECODE_ROWS]);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "central GPU runner only; mapped journal completion and CPU-service boundary"]
fn gpu_resident_owned_notice_compacts_completions_but_preserves_cpu_service_payload() {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select test GPU"));
    let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let (journal, _) = backend
        .alloc_mapped_arena_bda(crate::recorder::RESIDENT_JOURNAL_BYTES)
        .unwrap();
    let mut expected: Vec<_> = (0..crate::recorder::RESIDENT_JOURNAL_WORDS as u32).collect();
    for status in [0, 1, 4, 2, 3, 5] {
        expected[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = status;
        backend
            .upload(journal.as_ref(), bytemuck::cast_slice(&expected))
            .unwrap();
        assert_eq!(
            resident_notice_words(journal.as_ref(), 0)
                .unwrap()
                .as_slice(),
            expected
        );
        let compact = resident_notice_words_impl(journal.as_ref(), 0, true).unwrap();
        if matches!(status, 0 | 1 | 4) {
            let mut summary = [0; crate::recorder::RESIDENT_JOURNAL_WORDS];
            summary[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = status;
            assert_eq!(compact, summary);
        } else {
            assert_eq!(compact.as_slice(), expected);
        }
    }
    expected[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 1;
    backend
        .upload(journal.as_ref(), bytemuck::cast_slice(&expected))
        .unwrap();
    for compact in [false, true] {
        let started = std::time::Instant::now();
        for _ in 0..256 {
            std::hint::black_box(resident_notice_words_impl(journal.as_ref(), 0, compact).unwrap());
        }
        eprintln!(
            "OWNED_NOTICE_READ compact={compact} mean_us={:.3}",
            started.elapsed().as_secs_f64() * 1e6 / 256.0
        );
    }
}

#[test]
fn sparse_cpu_notice_prototype_rejects_invalid_shapes_and_ranges() {
    let valid = ResidentMoeLayout {
        rows: 1,
        n_expert: 64,
        n_used: 10,
    };
    let stride = crate::recorder::RESIDENT_JOURNAL_BYTES;
    for rows in [1, 2, 4, 8] {
        for used in [8, 10, 16] {
            let layout = ResidentMoeLayout {
                rows,
                n_used: used,
                ..valid
            };
            assert_eq!(
                sparse_cpu_notice_range(3 * stride, 2, layout).unwrap(),
                (2 * stride, rows * used)
            );
        }
    }
    for layout in [
        ResidentMoeLayout { rows: 0, ..valid },
        ResidentMoeLayout {
            rows: crate::recorder::RESIDENT_JOURNAL_MASK_WORDS + 1,
            ..valid
        },
        ResidentMoeLayout { n_used: 0, ..valid },
        ResidentMoeLayout {
            n_used: 17,
            ..valid
        },
        ResidentMoeLayout {
            n_expert: 9,
            ..valid
        },
        ResidentMoeLayout {
            rows: usize::MAX,
            n_used: 16,
            ..valid
        },
    ] {
        assert!(sparse_cpu_notice_range(usize::MAX, 0, layout).is_err());
    }
    for (bytes, ordinal) in [
        (0, 0),
        (stride - 1, 0),
        (3 * stride, 3),
        (3 * stride - 1, 2),
        (usize::MAX, usize::MAX),
    ] {
        assert!(sparse_cpu_notice_range(bytes, ordinal, valid).is_err());
    }
}

#[test]
#[ignore = "parent serial GPU only; production sparse CPU miss read correctness and bounded timing"]
fn gpu_resident_cpu_miss_sparse_read_prototype_exact_fields_and_timing() {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select test GPU"));
    let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    let stride = crate::recorder::RESIDENT_JOURNAL_WORDS;
    let (journal, _) = backend.alloc_mapped_arena_bda(3 * stride * 4).unwrap();
    for rows in [1, 2, 4, 8] {
        for used in [8, 10, 16] {
            let layout = ResidentMoeLayout {
                rows,
                n_expert: 64,
                n_used: used,
            };
            for status in [2, 5] {
                let mut fixture = (0..3 * stride)
                    .map(|i| 0xa5000000 | i as u32)
                    .collect::<Vec<_>>();
                for ordinal in 0..3 {
                    let words = &mut fixture[ordinal * stride..(ordinal + 1) * stride];
                    for (i, id) in words[..rows * used].iter_mut().enumerate() {
                        *id = (i % used) as u32;
                    }
                    for row in 0..rows {
                        words[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + row] =
                            ((1u32 << used) - 1) ^ (1u32 << (row % used));
                    }
                    words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD] = 3;
                    words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD + 1] = 0x12345678;
                    words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD + 2] = 0x9abcdef0;
                    words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = status;
                }
                backend
                    .upload(journal.as_ref(), bytemuck::cast_slice(&fixture))
                    .unwrap();
                for ordinal in 0..3 {
                    let full = resident_notice_words_impl(journal.as_ref(), ordinal, true).unwrap();
                    assert_eq!(
                        full.as_slice(),
                        &fixture[ordinal * stride..(ordinal + 1) * stride]
                    );
                    assert_eq!(
                        resident_notice_words(journal.as_ref(), ordinal).unwrap(),
                        full,
                        "legacy must retain the full journal including padding"
                    );
                    let sparse =
                        resident_owned_notice_words(journal.as_ref(), ordinal, layout).unwrap();
                    let tail = crate::recorder::RESIDENT_JOURNAL_MASK_WORD;
                    assert_eq!(&sparse[..rows * used], &full[..rows * used]);
                    assert_eq!(&sparse[tail..], &full[tail..]);
                    assert!(sparse[rows * used..tail].iter().all(|&word| word == 0));
                    let mut full_recovery = full;
                    let mut sparse_recovery = sparse;
                    full_recovery[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 2;
                    sparse_recovery[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 2;
                    assert_eq!(
                        resident_notice_slots(&sparse_recovery, layout).unwrap(),
                        resident_notice_slots(&full_recovery, layout).unwrap()
                    );
                    let context = infr_core::timeline::Context {
                        step: 1,
                        lanes: rows,
                        layer: 0,
                    };
                    let a = resident_notice_trace(context, ordinal, &full, layout, true);
                    let b = resident_notice_trace(context, ordinal, &sparse, layout, true);
                    assert!(a.route_payload_present && b.route_payload_present);
                    assert_eq!(a.routed_ids, b.routed_ids);
                    assert_eq!(a.hit_masks, b.hit_masks);
                }
                let mut samples = [Vec::with_capacity(256), Vec::with_capacity(256)];
                for iteration in 0..256 {
                    for variant in [iteration % 2, 1 - iteration % 2] {
                        let start = std::time::Instant::now();
                        let words = if variant == 0 {
                            resident_notice_words_impl(journal.as_ref(), 1, true).unwrap()
                        } else {
                            resident_owned_notice_words(journal.as_ref(), 1, layout).unwrap()
                        };
                        std::hint::black_box(words);
                        samples[variant].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
                for (variant, times) in samples.iter_mut().enumerate() {
                    let mean = times.iter().sum::<f64>() / times.len() as f64;
                    times.sort_by(f64::total_cmp);
                    eprintln!("CPU_NOTICE_SPARSE rows={rows} top={used} status={status} sparse={} bytes={} n={} mean_us={mean:.3} p50_us={:.3} p95_us={:.3}",
                        variant == 1, if variant == 0 { stride * 4 + 4 } else {
                            rows * used * 4 + (stride - crate::recorder::RESIDENT_JOURNAL_MASK_WORD) * 4 + 4
                        }, times.len(), times[times.len() / 2], times[(times.len() * 95).div_ceil(100) - 1]);
                }
            }
        }
    }
    for status in [0, 1, 3, 4, u32::MAX] {
        let mut words = (0..3 * stride)
            .map(|i| 0xb6000000 | i as u32)
            .collect::<Vec<_>>();
        words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = status;
        backend
            .upload(journal.as_ref(), bytemuck::cast_slice(&words))
            .unwrap();
        let layout = ResidentMoeLayout {
            rows: 1,
            n_expert: 64,
            n_used: 10,
        };
        let got = resident_owned_notice_words(journal.as_ref(), 0, layout).unwrap();
        if matches!(status, 0 | 1 | 4) {
            let mut expected = [0; crate::recorder::RESIDENT_JOURNAL_WORDS];
            expected[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = status;
            assert_eq!(got, expected);
        } else {
            assert_eq!(got.as_slice(), &words[..stride]);
        }
    }
    let mut words = vec![0u32; 3 * stride];
    words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD] = 5;
    backend
        .upload(journal.as_ref(), bytemuck::cast_slice(&words))
        .unwrap();
    for layout in [
        ResidentMoeLayout {
            rows: 0,
            n_expert: 64,
            n_used: 10,
        },
        ResidentMoeLayout {
            rows: 8,
            n_expert: 64,
            n_used: 17,
        },
        ResidentMoeLayout {
            rows: usize::MAX,
            n_expert: 64,
            n_used: 16,
        },
    ] {
        assert!(resident_owned_notice_words(journal.as_ref(), 0, layout).is_err());
    }
}

#[test]
#[ignore = "requires an idle Vulkan GPU; model-free auto-copy/prepared transition"]
fn resident_copy_switches_owned_legacy_and_reuses_session_owner() {
    let fixture = Fixture::new_with_copy_policy(1, false, Some(true));
    fixture.preload();
    fixture.backend.with_unified_exclusive(|| {
        let mut retained = None;
        let chunk = ResidentChunk::new(&fixture.backend, &fixture.graph, true, &mut retained)
            .unwrap()
            .unwrap();
        let owner = chunk.gpu_copy.clone().unwrap();
        assert!(owner.is_owned());
        drop(chunk);
        {
            let mut pager = fixture.backend.moe_pager().lock().unwrap();
            let session = pager.as_mut().unwrap();
            owner.begin(&fixture.backend, session).unwrap();
            session.finish_gpu_decode_forward();
            session.handoff_gpu_decode(&fixture.backend).unwrap();
        }
        let buffers = retained.as_mut().unwrap();
        buffers
            .ensure_gpu_copy(&fixture.backend, true, false)
            .unwrap();
        let legacy = buffers.gpu_copy.clone().unwrap();
        assert!(!legacy.is_owned());
        assert!(!Arc::ptr_eq(&legacy, &owner));
        assert!(!owner.owned_active());
        assert!(Arc::ptr_eq(
            fixture
                .backend
                .moe_pager()
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .gpu_decode_owner
                .as_ref()
                .unwrap(),
            &owner
        ));
        buffers
            .ensure_gpu_copy(&fixture.backend, true, true)
            .unwrap();
        assert!(Arc::ptr_eq(buffers.gpu_copy.as_ref().unwrap(), &owner));
        assert!(buffers.gpu_copy_init_attempted);
        buffers
            .ensure_gpu_copy(&fixture.backend, true, true)
            .unwrap();
        assert!(Arc::ptr_eq(buffers.gpu_copy.as_ref().unwrap(), &owner));
    });
}

#[test]
#[ignore = "requires an idle Vulkan GPU; model-free auto-copy/prepared transition"]
fn gpu_resident_copy_auto_includes_token_rows_but_excludes_verify() {
    for verify in [false, true] {
        let fixture = Fixture::new_with_copy_policy(1, true, None);
        let mut losing = fixture.graph.clone();
        losing.mtp_verify = verify;
        losing.independent_rows = !verify;
        if !verify {
            losing.sequence_spans = (0..2)
                .map(|row_start| SequenceSpan {
                    row_start,
                    rows: 1,
                    start_pos: row_start * 100,
                })
                .collect();
        }
        let inputs: Vec<_> = losing
            .ops
            .iter()
            .filter_map(|op| match op {
                Op::MoeFfn { x, .. } => Some(*x),
                _ => None,
            })
            .collect();
        for id in inputs {
            losing.tensors[id.0 as usize].desc = TensorDesc::new(vec![2, NE], DType::F32);
        }
        // Constructor-only probe: no mismatched fixture tensors are dispatched.
        let mut retained = None;
        let chunk = ResidentChunk::new(&fixture.backend, &losing, true, &mut retained)
            .unwrap()
            .expect("multirow graph retains ordinary recovery");
        assert_eq!(chunk.gpu_copy.is_some(), !verify);
        drop(chunk);
        assert_eq!(retained.as_ref().unwrap().gpu_copy_init_attempted, !verify);
        fixture
            .backend
            .runtime_phase
            .lock()
            .unwrap()
            .resident_buffers = retained;

        // The ordinary graph reuses common buffers, initializes copy if needed, then prepares
        // and consumes a real resident packet with a forced miss and numerical comparison.
        fixture.run_case(1, true, Some(0));
        let mut phase = fixture.backend.runtime_phase.lock().unwrap();
        let copy = phase
            .resident_buffers
            .as_ref()
            .unwrap()
            .gpu_copy
            .clone()
            .expect("ordinary graph must lazily allocate copy metadata");
        let chunk =
            ResidentChunk::new(&fixture.backend, &losing, true, &mut phase.resident_buffers)
                .unwrap()
                .unwrap();
        assert_eq!(
            chunk.gpu_copy.is_some(),
            !verify,
            "warm copy follows token-row policy"
        );
        drop(chunk);
        assert!(Arc::ptr_eq(
            phase
                .resident_buffers
                .as_ref()
                .unwrap()
                .gpu_copy
                .as_ref()
                .unwrap(),
            &copy
        ));
        drop(phase);
        fixture.run_case(1, true, Some(1));
    }
}
