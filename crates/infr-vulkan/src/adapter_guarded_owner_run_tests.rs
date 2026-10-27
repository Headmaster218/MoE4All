//! Parent-serialized real CPU-only source/recovery tests for guarded rolling admission.
use super::*;
use crate::pager::{ExpertSource, MoeHostChunkSpec, MoePagerLayout, MoePoolSpec, Role};
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};
use infr_core::{DType, SizeSpec, TensorDesc};
use std::sync::{Condvar, Weak};
use std::time::{Duration, Instant};

const NE: usize = 32;
const EXP: usize = 4;
const TOP: usize = 2;
const LAYERS: usize = 6;
const SLOT: usize = 64 * 1024;
const BANK: usize = EXP * SLOT;
const ROUTED: [u32; TOP] = [0, 1];

#[derive(Clone, Debug)]
struct Decision {
    first: usize,
    end: usize,
    confirmed: usize,
    pending: usize,
    limit: usize,
    eligible: bool,
}

#[derive(Default)]
struct GateState {
    released: bool,
    skipped: Vec<usize>,
}

#[derive(Default)]
struct Gate {
    state: Mutex<GateState>,
    wake: Condvar,
}

impl Gate {
    fn release(&self) {
        self.state.lock().unwrap().released = true;
        self.wake.notify_all();
    }
}

fn gates() -> &'static Mutex<std::collections::HashMap<usize, Weak<Gate>>> {
    static GATES: OnceLock<Mutex<std::collections::HashMap<usize, Weak<Gate>>>> = OnceLock::new();
    GATES.get_or_init(Mutex::default)
}

struct State {
    disabled: bool,
    gate: Arc<Gate>,
    decisions: Vec<Decision>,
    failure: Option<(usize, bool)>,
}

std::thread_local! {
    static STATE: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn disabled() -> bool {
    STATE.with(|state| state.borrow().as_ref().is_some_and(|state| state.disabled))
}

pub(crate) fn before_produce(device: usize) {
    let gate = gates().lock().unwrap().get(&device).and_then(Weak::upgrade);
    if let Some(gate) = gate {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut state = gate.state.lock().unwrap();
        while !state.released {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "guarded-run notice gate was not released"
            );
            state = gate.wake.wait_timeout(state, remaining).unwrap().0;
        }
    }
}

pub(crate) fn retired_skip(device: usize, ordinal: usize) {
    if let Some(gate) = gates().lock().unwrap().get(&device).and_then(Weak::upgrade) {
        gate.state.lock().unwrap().skipped.push(ordinal);
    }
}

pub(crate) fn checkpoint(
    _be: &VulkanBackend,
    chunk: &ResidentChunk,
    pending: usize,
    eligible: bool,
) -> Result<()> {
    let action = STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state.as_mut()?;
        let end = chunk.entries.len();
        state.decisions.push(Decision {
            first: chunk.submitted_entries,
            end,
            confirmed: chunk.notice_cursor,
            pending,
            limit: chunk.limit,
            eligible,
        });
        let failure = state.failure.filter(|&(at, _)| end == at);
        (end >= LAYERS - 1 || (!eligible && pending > 1) || failure.is_some())
            .then(|| (Arc::clone(&state.gate), failure))
    });
    if let Some((gate, failure)) = action {
        gate.release();
        if let Some((_, abort)) = failure {
            return Err(if abort {
                Error::Aborted
            } else {
                be("injected guarded-owner-run submission error")
            });
        }
    }
    Ok(())
}

struct Trace {
    device: usize,
    gate: Arc<Gate>,
}

impl Trace {
    fn begin(be_: &VulkanBackend, disabled: bool, failure: Option<(usize, bool)>) -> Self {
        let device = Arc::as_ptr(&be_.shared) as usize;
        let gate = Arc::new(Gate::default());
        if disabled {
            gate.release();
        }
        assert!(
            gates()
                .lock()
                .unwrap()
                .insert(device, Arc::downgrade(&gate))
                .is_none()
        );
        STATE.with(|state| {
            assert!(
                state
                    .replace(Some(State {
                        disabled,
                        gate: Arc::clone(&gate),
                        decisions: Vec::new(),
                        failure,
                    }))
                    .is_none()
            )
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| assert!(state.replace(Some([0; 3])).is_none()));
        Self { device, gate }
    }

    fn decisions(&self) -> Vec<Decision> {
        STATE.with(|state| state.borrow().as_ref().unwrap().decisions.clone())
    }

    fn recoveries(&self) -> usize {
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.get().unwrap()[0])
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        self.gate.release();
        gates().lock().unwrap().remove(&self.device);
        STATE.with(|state| {
            state.take();
        });
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.set(None));
    }
}

#[test]
fn guarded_owner_run_bounds_allow_host_without_changing_mapped_contract() {
    for mask in 0..16 {
        let entries: Vec<_> = (0..4)
            .map(|i| ResidentJournalEntry {
                op_idx: i * 2,
                roles: [i * 3, i * 3 + 1, i * 3 + 2],
                layout: ResidentMoeLayout {
                    rows: 1,
                    n_expert: EXP,
                    n_used: TOP,
                },
                host_notice: mask & (1 << i) != 0,
            })
            .collect();
        for confirmed in 0..=4 {
            for submitted in 0..=4 {
                for pending in 0..=5 {
                    let expected = confirmed <= submitted
                        && submitted < 4
                        && (1..=4).contains(&pending)
                        && entries[confirmed..].iter().any(|entry| entry.host_notice);
                    assert_eq!(
                        bounded_host_run(
                            true, &entries, confirmed, submitted, submitted, 4, pending
                        ),
                        expected
                    );
                    assert!(!bounded_host_run(
                        false, &entries, confirmed, submitted, submitted, 4, pending
                    ));
                    assert!(!bounded_host_run(
                        true,
                        &entries,
                        confirmed,
                        submitted,
                        submitted + 1,
                        4,
                        pending
                    ));
                    assert!(!bounded_host_run(
                        true, &entries, confirmed, submitted, submitted, 3, pending
                    ));
                }
            }
        }
        if mask != 0 {
            assert!(!prepared_mapped_run_eligible(true, &entries, 0, 0, 4));
        }
    }
}

#[test]
fn guarded_owner_host_queue_credits_follow_existing_window() {
    for (inflight, expected) in [(0, 2), (1, 2), (2, 3), (3, 3), (usize::MAX, 3)] {
        assert_eq!(host_queue_budget(inflight), expected);
    }
}

struct ExpertIo;

impl BlockIo for ExpertIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        if desc.nbytes() != BANK || dst.len() < BANK || desc.id >= (LAYERS * 3) as u32 {
            return Err(be("guarded-run expert bank bounds"));
        }
        dst[..BANK].fill(0);
        for expert in 0..EXP {
            let value = 0.125 + desc.id as f32 / 64.0 + expert as f32 / 32.0;
            for i in 0..NE {
                let offset = expert * SLOT + (i * NE + i) * 4;
                dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(())
    }
}

// Same diagonal-bank graph and real pager setup as resident_ahead_tests. Kept local
// so this regression adds no visibility changes or edits to the existing fixture contract.
struct Fixture {
    be: VulkanBackend,
    graph: Graph,
    frontend: TensorId,
    outputs: Vec<TensorId>,
    buffers: Vec<(TensorId, Box<dyn Buffer>)>,
    roles: Vec<[usize; 3]>,
    rows: usize,
    cpu_owner: std::cell::RefCell<Option<Arc<crate::pager::GpuCopyBuffers>>>,
}

impl Fixture {
    fn new(rows: usize) -> Self {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev = Some(
            std::env::var("MOE4ALL_TEST_DEVICE")
                .expect("parent must explicitly select an idle GPU"),
        );
        cfg.device.submit_dispatches = Some(0);
        cfg.device.decode_command_ahead = true;
        cfg.device.decode_command_early_submit = true;
        cfg.device.decode_command_resident = true;
        cfg.paging.expert_prefetch = false;
        cfg.paging.host_dma = true;
        cfg.paging.ring = Some(SizeSpec::Bytes((4 * SLOT) as u64));
        let vk = &mut cfg.kernels.vulkan;
        vk.cpu_miss_threads = 0;
        vk.static_scratch_memo = true;
        vk.decode_gpu_resident = true;
        vk.decode_gpu_resident_copy = Some(true);
        vk.decode_gpu_resident_copy_overlap = false;
        // One accepted prepared resident prefix, then ordinary rolling fallback.
        vk.decode_gpu_resident_layers = 1;
        vk.decode_gpu_resident_submit_layers = 1;
        vk.decode_gpu_resident_inflight = 1;
        vk.decode_gpu_resident_pure_mapped_layers = 0;
        vk.decode_gpu_resident_hit_first = true;
        vk.decode_gpu_resident_fast_lut = true;
        vk.decode_gpu_resident_immutable_predicates = true;
        vk.decode_gpu_resident_indirect_predicates = false;
        vk.decode_gpu_resident_notice_thread = true;
        vk.decode_gpu_resident_early_recovery = true;
        vk.decode_gpu_resident_waiter = false;
        vk.decode_gpu_resident_host_ready = true;
        let be_ = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        assert!(be_.shared.conditional_rendering.is_some());
        be_.init_moe_pager(MoePagerLayout {
            load_reserve_bytes: 0,
            n_blocks: LAYERS * EXP,
            pools: vec![MoePoolSpec {
                slot_bytes: SLOT,
                n_slots: 128,
                min_enabled_slots: LAYERS * EXP * 3,
            }],
            host_tier: None,
            host_store_io: Some(Arc::new(ExpertIo)),
            dynamic_state_reserve_bytes: 0,
            dynamic_state_max_allocation_bytes: 0,
            prefill_min_lane_bytes: 0,
            runtime_reserve_bytes: 1024 * 1024,
            host_chunks: vec![MoeHostChunkSpec {
                base_offset: 0,
                bytes: LAYERS * 3 * BANK,
            }],
            prefill_target_lanes: 1,
            prefill_cache_bytes: (128 * SLOT) as u64,
        })
        .unwrap();
        let mut graph = Graph::new();
        if rows > 1 {
            graph.independent_rows = true;
            graph.sequence_spans = (0..rows)
                .map(|row_start| SequenceSpan {
                    row_start: row_start as u32,
                    rows: 1,
                    start_pos: (row_start * 100) as u32,
                })
                .collect();
        }
        let frontend = graph.input(TensorDesc::new(vec![rows, NE], DType::F32));
        let one = graph.input(TensorDesc::new(vec![rows, NE], DType::F32));
        let input = graph.input(TensorDesc::new(vec![rows, NE], DType::F32));
        let routing = graph.input(TensorDesc::new(vec![rows, NE], DType::F32));
        let router = graph.weight(TensorDesc::new(vec![EXP, NE], DType::F32));
        let mut handles = Vec::new();
        let mut outputs = Vec::new();
        let mut x = input;
        for _ in 0..LAYERS {
            graph.push(Op::Add {
                a: frontend,
                b: one,
                dst: frontend,
                n: (rows * NE) as u32,
            });
            let gate_exps = graph.weight(TensorDesc::new(vec![EXP, NE, NE], DType::F32));
            let up_exps = graph.weight(TensorDesc::new(vec![EXP, NE, NE], DType::F32));
            let down_exps = graph.weight(TensorDesc::new(vec![EXP, NE, NE], DType::F32));
            handles.extend([gate_exps, up_exps, down_exps]);
            let dst = graph.output(TensorDesc::new(vec![rows, NE], DType::F32));
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
                n_ff_exp: NE as u32,
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
        for (id, value, usage) in [
            (frontend, 0.0f32, BufferUsage::Staging),
            (one, 1.0, BufferUsage::Activations),
            (input, 1.0, BufferUsage::Activations),
            (routing, 1.0, BufferUsage::Activations),
        ] {
            let buffer = be_.alloc(rows * NE * 4, usage).unwrap();
            be_.upload(
                buffer.as_ref(),
                bytemuck::cast_slice(&vec![value; rows * NE]),
            )
            .unwrap();
            buffers.push((id, buffer));
        }
        let weights = be_.alloc(EXP * NE * 4, BufferUsage::Weights).unwrap();
        let mut values = vec![0.0f32; EXP * NE];
        for expert in 0..EXP {
            values[expert * NE] = (EXP - expert) as f32;
        }
        be_.upload(weights.as_ref(), bytemuck::cast_slice(&values))
            .unwrap();
        buffers.push((router, weights));
        let bank: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(vec![0u8; BANK]);
        let mut roles = Vec::new();
        for (i, id) in handles.into_iter().enumerate() {
            let placeholder = be_.alloc(4, BufferUsage::Staging).unwrap();
            let role = crate::pager::buffer_identity(placeholder.as_ref());
            be_.register_paged_expert(
                [Role::Gate, Role::Up, Role::Down][i % 3],
                role,
                ExpertSource {
                    bank: Arc::clone(&bank),
                    stride_bytes: SLOT,
                    layer_base: (i / 3 * EXP) as u32,
                    host_offset: i * BANK,
                    file: Some(BlockDesc {
                        id: i as u32,
                        extents: vec![BlockExtent {
                            offset: 0,
                            len: BANK,
                        }],
                    }),
                },
                EXP,
            )
            .unwrap();
            roles.push(role);
            buffers.push((id, placeholder));
        }
        be_.finish_session_allocations().unwrap();
        for &id in &outputs {
            buffers.push((id, be_.alloc(rows * NE * 4, BufferUsage::Readback).unwrap()));
        }
        Self {
            be: be_,
            graph,
            frontend,
            outputs,
            buffers,
            roles: roles
                .chunks_exact(3)
                .map(|roles| roles.try_into().unwrap())
                .collect(),
            rows,
            cpu_owner: std::cell::RefCell::new(None),
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

    fn bits(&self, id: TensorId) -> Vec<u32> {
        let mut values = vec![0u32; self.rows * NE];
        self.be
            .download(self.buffer(id), bytemuck::cast_slice_mut(&mut values))
            .unwrap();
        values
    }

    fn reset(&self) {
        self.be
            .upload(
                self.buffer(self.frontend),
                bytemuck::cast_slice(&vec![0.0f32; self.rows * NE]),
            )
            .unwrap();
    }

    fn preload(&self) {
        let mut guard = self.be.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        session.handoff_gpu_decode(&self.be).unwrap();
        for roles in &self.roles {
            session.begin_role_batches(roles).unwrap();
            session
                .push_roles_cpu(
                    &self.be,
                    &roles.map(|role| (role, ROUTED.as_slice())),
                    false,
                )
                .unwrap()
                .complete_without_recorder(&self.be)
                .unwrap();
        }
    }

    fn plan(&self) -> VkDecodePlan {
        let mut plan = VkDecodePlan::new_cpu(&self.graph, false, true);
        plan.phase_requests =
            Some(stages::compiled_workspace_requests(&self.be, &self.graph).unwrap());
        plan
    }

    fn install_cpu_owner(&self, misses: &[usize]) {
        self.preload();
        // Rebuilding metadata changes unified Expert loans. A/B must reseed the
        // same inactive owner, not compare two different physical cache layouts.
        if let Some(owner) = self.cpu_owner.borrow().as_ref() {
            assert!(!owner.owned_active() && !owner.owned_in_forward());
            {
                let mut guard = self.be.moe_pager().lock().unwrap();
                let session = guard.as_mut().unwrap();
                assert!(Arc::ptr_eq(
                    session.gpu_decode_owner.as_ref().unwrap(),
                    owner
                ));
                for &layer in misses {
                    session.discard_cpu_miss(self.roles[layer], &[ROUTED[1]]);
                }
            }
            {
                let phase = self.be.runtime_phase.lock().unwrap();
                assert!(Arc::ptr_eq(
                    phase
                        .resident_buffers
                        .as_ref()
                        .unwrap()
                        .gpu_copy
                        .as_ref()
                        .unwrap(),
                    owner,
                ));
            }
            self.reset();
            return;
        }
        let capacity = self
            .be
            .runtime_phase
            .lock()
            .unwrap()
            .resident_buffers
            .as_ref()
            .unwrap()
            .capacity;
        let (sources, seed) = {
            let mut guard = self.be.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            session.handoff_gpu_decode(&self.be).unwrap();
            let previous = session.gpu_decode_owner.clone().expect("warm owner");
            let banks: Vec<_> = self
                .roles
                .iter()
                .map(|roles| previous.role_params(session, roles).unwrap())
                .collect();
            for &layer in misses {
                session.discard_cpu_miss(self.roles[layer], &[ROUTED[1]]);
            }
            session.gpu_decode_owner = None;
            session.test_unimport_host_uploads();
            let mut sources = session.gpu_copy_sources().unwrap().unwrap();
            for bank in banks {
                for base in &bank[..3] {
                    sources.test_cpu_only_source(*base as usize + ROUTED[1] as usize);
                }
            }
            (sources, crate::pager::owned_seed(session, 0).unwrap())
        };
        let owner = crate::pager::GpuCopyBuffers::new_owned(&self.be, sources, capacity, seed)
            .unwrap()
            .unwrap();
        assert!(!owner.owned_self_service());
        self.be
            .moe_pager()
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .gpu_decode_owner = Some(Arc::clone(&owner));
        self.be
            .runtime_phase
            .lock()
            .unwrap()
            .resident_buffers
            .as_mut()
            .unwrap()
            .gpu_copy = Some(Arc::clone(&owner));
        *self.cpu_owner.borrow_mut() = Some(owner);
        self.reset();
    }

    fn assert_result(&self, expected: &[Vec<u32>]) -> Vec<(Vec<u32>, usize)> {
        assert_eq!(
            self.bits(self.frontend),
            vec![(LAYERS as f32).to_bits(); self.rows * NE]
        );
        assert_eq!(
            self.outputs
                .iter()
                .map(|&id| self.bits(id))
                .collect::<Vec<_>>(),
            expected
        );
        let mut guard = self.be.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        // Handoff validates the complete GPU directory, FIFO, overlay, and exact LRU.
        session.handoff_gpu_decode(&self.be).unwrap();
        assert!(!session.gpu_decode_in_forward());
        for roles in &self.roles {
            assert_eq!(
                session.routed_roles_resident_mask(roles, &ROUTED).unwrap(),
                (1 << TOP) - 1
            );
        }
        session.test_resident_lru_snapshot().unwrap()
    }
}

fn warm(rows: usize) -> (Fixture, VkDecodePlan, Vec<Vec<u32>>) {
    let fixture = Fixture::new(rows);
    fixture.preload();
    let plan = fixture.plan();
    execute_static(&fixture.be, &plan, &fixture.bindings()).unwrap();
    let expected: Vec<Vec<u32>> = fixture.outputs.iter().map(|&id| fixture.bits(id)).collect();
    // Convert once, then warm that exact CPU-only owner before either A/B arm
    // captures its full LRU/free-count reference. No per-arm metadata allocations.
    fixture.install_cpu_owner(&[]);
    execute_static(&fixture.be, &plan, &fixture.bindings()).unwrap();
    fixture.assert_result(&expected);
    (fixture, plan, expected)
}

#[test]
#[ignore = "parent serial GPU only; HOST-potential all-hit queue exceeds inflight with unchanged outputs/LRU"]
fn gpu_guarded_owner_run_host_potential_all_hits_and_missing_receipt() {
    let (fixture, mut plan, expected) = warm(1);
    let mut reference = None;
    for mode in 0..3 {
        fixture.install_cpu_owner(&[]);
        if mode == 2 {
            plan.phase_requests = None;
        }
        let trace = Trace::begin(&fixture.be, mode == 0, None);
        if mode == 2 {
            trace.gate.release();
        }
        execute_static(&fixture.be, &plan, &fixture.bindings()).unwrap();
        let decisions = trace.decisions();
        assert_eq!(
            decisions.iter().any(|d| d.eligible && d.pending > 1),
            mode == 1,
            "{decisions:?}"
        );
        if mode == 1 {
            assert!(
                decisions
                    .iter()
                    .any(|d| d.pending == 2 && d.eligible && d.confirmed == 0)
            );
            assert!(decisions.iter().all(|d| d.pending <= 3));
            assert!(decisions.iter().filter(|d| d.eligible).all(|d| d.pending <= 2));
        }
        for d in &decisions {
            assert!(d.pending <= d.end && d.pending <= d.limit && d.first < d.end);
        }
        assert_eq!(trace.recoveries(), 0);
        let lru = fixture.assert_result(&expected);
        if let Some(reference) = &reference {
            assert_eq!(&lru, reference);
        } else {
            reference = Some(lru);
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; actual unimported CPU misses first/middle/late, SKIP tail and exact recovery/LRU"]
fn gpu_guarded_owner_run_cpu_first_middle_late_and_two_rows() {
    for rows in [1, 2] {
        let (fixture, plan, expected) = warm(rows);
        for misses in [vec![0], vec![2], vec![4], vec![0, 3]] {
            let mut reference = None;
            for disabled in [true, false] {
                fixture.install_cpu_owner(&misses);
                let trace = Trace::begin(&fixture.be, disabled, None);
                execute_static(&fixture.be, &plan, &fixture.bindings()).unwrap();
                assert_eq!(
                    trace.recoveries(),
                    misses.len(),
                    "rows={rows} misses={misses:?}"
                );
                let decisions = trace.decisions();
                if !disabled {
                    assert!(
                        decisions.iter().any(|d| d.eligible && d.pending > 1),
                        "{decisions:?}"
                    );
                    let submitted_tail = misses[0] < LAYERS - 2
                        || decisions
                            .iter()
                            .any(|d| d.eligible && d.confirmed == 0 && d.end > misses[0] + 1);
                    if submitted_tail {
                        assert!(
                            !trace.gate.state.lock().unwrap().skipped.is_empty(),
                            "submitted CPU tail must be SKIP"
                        );
                    }
                }
                let lru = fixture.assert_result(&expected);
                if let Some(reference) = &reference {
                    assert_eq!(&lru, reference);
                } else {
                    reference = Some(lru);
                }
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; accepted prepared resident prefix then ordinary HOST-sensitive rolling suffix"]
fn gpu_guarded_owner_run_after_accepted_prepared_prefix() {
    let (fixture, current, expected) = warm(1);
    fixture.install_cpu_owner(&[]);
    let next = fixture.plan();
    execute_static_next(
        &fixture.be,
        &current,
        &fixture.bindings(),
        Some((&next, &fixture.bindings(), &[1])),
    )
    .unwrap();
    {
        let packet = next.prepared_prefix.lock().unwrap();
        let (head_end, head_layers, end, layers, self_service, owned) =
            packet.as_ref().expect("prepared prefix").recorded_shape();
        assert_eq!((head_layers, layers), (1, 1));
        assert_eq!(head_end, end);
        assert!(end < fixture.graph.ops.len() && owned && !self_service);
    }
    fixture.reset();
    let trace = Trace::begin(&fixture.be, false, None);
    execute_static(&fixture.be, &next, &fixture.bindings()).unwrap();
    let decisions = trace.decisions();
    assert!(
        decisions
            .iter()
            .any(|d| d.first >= 1 && d.eligible && d.pending > 1),
        "{decisions:?}"
    );
    assert!(next.prepared_prefix.lock().unwrap().is_none());
    assert_eq!(trace.recoveries(), 0);
    fixture.assert_result(&expected);
}

#[test]
#[ignore = "parent serial GPU only; error/cancel after multiple HOST-sensitive queued segments drains before next request"]
fn gpu_guarded_owner_run_error_cancel_drain_and_next_request() {
    for abort in [false, true] {
        let (fixture, plan, expected) = warm(1);
        fixture.install_cpu_owner(&[1]);
        let trace = Trace::begin(&fixture.be, false, Some((3, abort)));
        let result = execute_static(&fixture.be, &plan, &fixture.bindings());
        if abort {
            assert!(matches!(result, Err(Error::Aborted)));
        } else {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("injected guarded-owner-run")
            );
        }
        assert!(
            trace
                .decisions()
                .iter()
                .any(|d| d.end == 3 && !d.eligible && d.pending == 3)
        );
        drop(trace);
        fixture.install_cpu_owner(&[1]);
        execute_static(&fixture.be, &plan, &fixture.bindings()).unwrap();
        fixture.assert_result(&expected);
    }
}
