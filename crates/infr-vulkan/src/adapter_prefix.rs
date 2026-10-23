//! Scoped, unsubmitted command preparation. Resource destruction invalidates the packet.
use super::*;
use std::cell::Cell;

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn preparing_commands() -> bool {
    RECORDING.with(Cell::get)
}

struct RecordingGuard;

/// Keep unready work out of Rayon: nested parallel expert copies may steal queued tasks.
pub(super) struct DeferredPreparation<F>(Option<F>);

impl<F> DeferredPreparation<F> {
    pub(super) fn new(job: Option<F>) -> Self {
        Self(job)
    }

    pub(super) fn start<'scope>(&mut self, scope: &rayon::Scope<'scope>)
    where
        F: FnOnce() + Send + 'scope,
    {
        if let Some(job) = self.0.take() {
            scope.spawn(move |_| job());
        }
    }
}
impl RecordingGuard {
    fn enter() -> Self {
        assert!(!RECORDING.with(|value| value.replace(true)));
        Self
    }
}
impl Drop for RecordingGuard {
    fn drop(&mut self) {
        RECORDING.with(|value| value.set(false));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BufferKey {
    pointer: usize,
    bytes: usize,
    address: Option<u64>,
}
fn buffer_key(buffer: &dyn Buffer) -> BufferKey {
    BufferKey {
        pointer: (buffer as *const dyn Buffer as *const ()) as usize,
        bytes: buffer.len_bytes(),
        address: buffer.device_addr(),
    }
}

pub(super) struct PreparedPrefix {
    command: crate::recorder::PreparedCommand,
    epoch: u64,
    tensors: Vec<(TensorId, Vec<BufferKey>)>,
    pool: Vec<(ScratchKey, u64, usize, usize)>,
    positions: Vec<(TensorId, usize)>,
    pub(super) end: usize,
    resident: Option<PreparedResident>,
}

struct PreparedResident {
    chunk: ResidentChunk,
    tape: Arc<dyn Buffer>,
    banks: Vec<(usize, u64, usize)>,
    tape_words: usize,
}

pub(super) struct ActivatedPrefix {
    pub(super) command: crate::recorder::PreparedCommand,
    pub(super) end: usize,
    pub(super) resident: Option<ResidentChunk>,
    pub(super) tape_words: usize,
}

fn resident_end(graph: &Graph, layers: usize) -> Option<usize> {
    prefix_end(graph)?;
    let mut count = 0;
    for (index, op) in graph.ops.iter().enumerate() {
        if matches!(op, Op::MoeFfn { .. }) {
            ResidentMoeLayout::from_op(graph, op)?;
            count += 1;
            if count == layers {
                return Some(index + 1);
            }
        } else if !supported_op(op) {
            return None;
        }
    }
    None
}

fn positions_match(
    expected: &[(TensorId, usize)],
    mut actual: impl FnMut(TensorId) -> Option<u32>,
) -> bool {
    expected
        .iter()
        .all(|&(id, position)| actual(id).map(|value| value as usize) == Some(position))
}

fn mapped_position(buffer: &dyn Buffer) -> Option<u32> {
    if buffer.len_bytes() < 4 {
        return None;
    }
    let pointer = crate::as_vk_buf(buffer).ok()?.mapped_ptr()?;
    let mut bytes = [0u8; 4];
    // The execution gate and caller retain this coherent position buffer through validation.
    unsafe { std::ptr::copy_nonoverlapping(pointer.cast_const(), bytes.as_mut_ptr(), 4) };
    Some(u32::from_le_bytes(bytes))
}

fn keys(scratch: &ScratchSet, bindings: &Bindings, id: TensorId) -> Result<Vec<BufferKey>> {
    if let Some(rows) = bindings.get_rows(id) {
        return Ok(rows.iter().map(|buffer| buffer_key(*buffer)).collect());
    }
    Ok(vec![buffer_key(resolve(scratch, bindings, id)?)])
}

pub(crate) fn supported(be_: &VulkanBackend) -> bool {
    be_.cfg().device.decode_command_ahead
        && be_.moe_paged()
        && !infr_core::pager_profile::active()
        && !be_.cfg().prof.ops
        && !be_.cfg().prof.stages
        && be_.unified_client.is_none()
        && !be_.dense_paged()
        && !be_.cfg().spec.mtp
        && !be_.cfg().paging.expert_prefetch
}

pub(crate) fn enabled(be_: &VulkanBackend) -> bool {
    supported(be_) && !be_.shared.submit_tune_active.load(Ordering::Acquire)
}

pub(super) fn prefix_end(graph: &Graph) -> Option<usize> {
    if graph.mtp_verify || graph.prefill_target_tokens.is_some() || graph.prefill_next_chunk {
        return None;
    }
    let end = graph
        .ops
        .iter()
        .position(|op| matches!(op, Op::MoeFfn { .. }))?;
    if graph.ops.iter().any(|op| match op {
        Op::MoeFfn { x, ne, .. } => *ne == 0 || graph.desc(*x).numel() != *ne as usize,
        _ => false,
    }) {
        return None;
    }
    (end > 0 && graph.ops[..end].iter().all(supported_op)).then_some(end)
}

fn supported_op(op: &Op) -> bool {
    matches!(
        op,
        Op::EmbedGather { .. }
            | Op::Copy { .. }
            | Op::CopyStrided { .. }
            | Op::Linear { m: 1, .. }
            | Op::LinearPair { m: 1, .. }
            | Op::RmsNorm { .. }
            | Op::RmsNormAdd { .. }
            | Op::LayerNorm { .. }
            | Op::QwenHcNorm { .. }
            | Op::QwenHcMix { .. }
            | Op::QwenHcDownInject { .. }
            | Op::QwenHcInject { .. }
            | Op::QwenHcInjectNorm { .. }
            | Op::QwenPleGate { .. }
            | Op::QkNorm { .. }
            | Op::GatedRmsNorm { .. }
            | Op::QkNormRope { .. }
            | Op::Rope { .. }
            | Op::WriteKv { .. }
            | Op::Attention { rows: 1, .. }
            | Op::Conv1dSilu { rows: 1, .. }
            | Op::DeltaNet { rows: 1, .. }
            | Op::GatedAct { .. }
            | Op::GatedActFused { .. }
            | Op::Silu { .. }
            | Op::Gelu { .. }
            | Op::Add { .. }
            | Op::AddBias { .. }
            | Op::MulVec { .. }
            | Op::Scale { .. }
            | Op::HeadwiseSigmoidMul { .. }
            | Op::MoeSharedExpertAdd { rows: 1, .. }
    )
}

/// Alias allocation objects only for the scoped recorder. Pin every source capacity against
/// same-execute replacement until the preparation thread has joined and finish_execute compacts.
pub(super) fn snapshot_pool(pool: &mut ScratchPool) -> Result<ScratchPool> {
    let mut snapshot = ScratchPool::default();
    for (&key, buffer) in &pool.buffers {
        // SAFETY: the caller joins preparation before finish_execute or phase release, and the
        // in_use set prevents a growing op from retiring this capacity during that recording.
        let alias = unsafe { crate::as_vk_buf(buffer.as_ref())?.recording_alias() };
        snapshot.buffers.insert(key, Box::new(alias));
        pool.in_use.insert(key);
    }
    Ok(snapshot)
}

pub(super) fn compatible(current: &Graph, next: &Graph, scratch: &ScratchSet) -> bool {
    if decode_token_rows(next) != Some(1) || next.independent_rows || current.independent_rows {
        return false;
    }
    let Ok(layout) = scratch_layout(next) else {
        return false;
    };
    if layout.len() != scratch.len()
        || !layout
            .iter()
            .zip(scratch)
            .all(|(need, buffer)| match (need, buffer) {
                (None, None) => true,
                (Some(bytes), Some(buffer)) => buffer.len_bytes() >= *bytes,
                _ => false,
            })
    {
        return false;
    }
    prefix_end(next).is_some()
}

impl PreparedPrefix {
    pub(super) fn resident_layers(&self) -> usize {
        self.resident
            .as_ref()
            .map_or(0, |resident| resident.chunk.entries.len())
    }
    pub(super) fn take_if_valid(
        self,
        be_: &VulkanBackend,
        scratch: &ScratchSet,
        pool: &mut ScratchPool,
        bindings: &Bindings,
        retained: Option<&ResidentBuffers>,
    ) -> Option<ActivatedPrefix> {
        if !enabled(be_) || !self.command.trace_compatible() {
            return None;
        }
        let epoch = be_.shared.resource_epoch.load(Ordering::Acquire);
        if self.epoch != epoch {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                tracing::info!(
                    prepared_epoch = self.epoch,
                    current_epoch = epoch,
                    "Decode command prefix invalidated by resource replacement"
                )
            });
            return None;
        }
        if self
            .tensors
            .iter()
            .any(|(id, expected)| keys(scratch, bindings, *id).as_ref().ok() != Some(expected))
            || !positions_match(&self.positions, |id| {
                mapped_position(resolve(scratch, bindings, id).ok()?)
            })
            || self.pool.iter().any(|(key, handle, offset, bytes)| {
                pool.buffers
                    .get(key)
                    .and_then(|buffer| crate::as_vk_buf(buffer.as_ref()).ok())
                    .is_none_or(|buffer| {
                        use ash::vk::Handle;
                        buffer.buffer.as_raw() != *handle
                            || buffer.sub_offset != *offset
                            || buffer.size != *bytes
                    })
            })
        {
            return None;
        }
        // begin_execute cleared the previous step's pins. Keep referenced capacities alive
        // through this submission even when no further command is being prepared.
        pool.in_use.extend(self.pool.iter().map(|(key, ..)| *key));
        let mut tape_words = 0;
        let resident = if let Some(mut prepared) = self.resident {
            let buffers = retained?;
            if !Arc::ptr_eq(&buffers.control, &prepared.chunk.control)
                || !Arc::ptr_eq(&buffers.journal, &prepared.chunk.journal)
                || !match (&buffers.notices, &prepared.chunk.notices) {
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    (None, None) => true,
                    _ => false,
                }
            {
                return None;
            }
            let mut guard = be_.moe_pager().lock().ok()?;
            let session = guard.as_mut()?;
            if !Arc::ptr_eq(&session.retained_tape(), &prepared.tape)
                || prepared.banks.iter().any(|&(role, address, bytes)| {
                    session.arena_addr(role).ok() != Some(address)
                        || session.slot_bytes(role).ok() != Some(bytes)
                })
            {
                return None;
            }
            // No submitted command references these new tape ranges yet. Only now publish
            // fresh addresses; recording on the previous token never froze or wrote them.
            for entry in &prepared.chunk.entries {
                for &role in &entry.roles {
                    session
                        .resident_lut_window(&mut tape_words, role, entry.layout.n_expert)
                        .ok()?;
                }
            }
            if tape_words != prepared.tape_words {
                return None;
            }
            if let Some(notices) = &prepared.chunk.notices {
                notices.reset().ok()?;
            }
            // Speculative packets must not own the live producer: ResidentChunk::drop drains
            // and resets it. Attach those runtime owners only after successful activation.
            prepared.chunk.producer = buffers.producer.clone();
            prepared.chunk.waiter = buffers.waiter.clone();
            Some(prepared.chunk)
        } else {
            None
        };
        Some(ActivatedPrefix {
            command: self.command,
            end: self.end,
            resident,
            tape_words,
        })
    }
}

pub(super) fn record(
    be_: &VulkanBackend,
    plan: &VkDecodePlan,
    scratch: &ScratchSet,
    bindings: &Bindings,
    positions: &[usize],
    mut pool: ScratchPool,
    mut resident: Option<ResidentChunk>,
) -> Result<PreparedPrefix> {
    use ash::vk::Handle;
    let _guard = RecordingGuard::enter();
    let graph = &plan.graph;
    if let Some(chunk) = resident.as_mut() {
        let cfg = &be_.cfg().kernels.vulkan;
        let submit_layers = match cfg.decode_gpu_resident_submit_layers {
            0 => graph.resident_submit_layers.unwrap_or(1),
            explicit => explicit,
        };
        chunk.submit_layers = submit_layers;
        chunk.mapped_serial_max_bytes =
            resident_mapped_serial_max_bytes(cfg.decode_gpu_resident_mapped_serial_max_kib, graph);
        if !(1..chunk.limit).contains(&submit_layers) {
            resident = None;
        }
    }
    let end = match resident
        .as_ref()
        .and_then(|chunk| resident_end(graph, chunk.submit_layers))
    {
        Some(end) => end,
        None => {
            resident = None;
            prefix_end(graph).ok_or_else(|| be("unsupported prepared prefix"))?
        }
    };
    if positions.len() != 1 {
        return Err(be("prepared prefix requires one row"));
    }
    let epoch = be_.shared.resource_epoch.load(Ordering::Acquire);
    let mut ids = HashSet::new();
    for op in &graph.ops[..end] {
        let (reads, writes) = op.io_inline();
        ids.extend(reads.into_iter().chain(writes));
    }
    let metadata = plan.static_scratch_metadata()?;
    ids.extend(metadata.reset_indices.iter().map(|&i| TensorId(i as u32)));
    // Fused shared-expert weights/outputs can belong to ops just past the submission boundary.
    // Validate those bindings too, even though their standalone commands are suppressed.
    for (index, _) in paged_moe_shared_candidates(graph, false) {
        if index < end {
            for op in graph.ops.iter().skip(index + 1).take(6) {
                let (reads, writes) = op.io_inline();
                ids.extend(reads.into_iter().chain(writes));
            }
        }
    }
    let tensors = ids
        .into_iter()
        .map(|id| Ok((id, keys(scratch, bindings, id)?)))
        .collect::<Result<Vec<_>>>()?;
    let pool_keys = pool
        .buffers
        .iter()
        .map(|(&key, buffer)| {
            let buffer = crate::as_vk_buf(buffer.as_ref())?;
            Ok((key, buffer.buffer.as_raw(), buffer.sub_offset, buffer.size))
        })
        .collect::<Result<Vec<_>>>()?;
    let rec = Recorder::new_prepared(be_)?;
    rec.seed_barrier();
    for &index in &metadata.reset_indices {
        let buffer = scratch[index]
            .as_deref()
            .ok_or_else(|| be("missing prepared scratch reset"))?;
        rec.zero(buffer, buffer.len_bytes().div_ceil(4));
    }
    let mut rope_positions = HashMap::new();
    for op in &graph.ops[..end] {
        if let Op::Rope { positions: id, .. } | Op::QkNormRope { positions: id, .. } = op {
            rope_positions.insert(id.0, positions[0]);
        }
    }
    let fusions = infr_core::fusion::plan_fusions(graph, &fusion_cfg(be_));
    let mut skip = fusions.skip;
    let mut shared_ops = HashMap::new();
    if resident.is_some() {
        let disabled = std::env::var_os("INFR_NO_MOE_SHARED_SLOT").is_some();
        for (index, shared) in paged_moe_shared_candidates(graph, disabled) {
            let Op::MoeFfn { gate_exps, .. } = &graph.ops[index] else {
                unreachable!()
            };
            let gate = resolve(scratch, bindings, *gate_exps)?;
            if be_
                .moe_pager()
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|session| {
                    session.is_paged(
                        crate::pager::Role::Gate,
                        crate::pager::buffer_identity(gate),
                    )
                })
            {
                shared_ops.insert(index, shared);
                skip.extend(index + 1..index + 7);
            }
        }
    }
    if be_.cfg().kernels.vulkan.hc_up_mix {
        skip.extend(
            (0..end)
                .filter_map(|i| hc_up_mix_pair(be_, graph, i, scratch, bindings).map(|_| i + 1)),
        );
    }
    if be_.cfg().kernels.vulkan.ple_gate_norm_lds {
        skip.extend(
            (0..end).filter_map(|i| {
                ple_gate_norm_pair(be_, graph, i, scratch, bindings).map(|_| i + 1)
            }),
        );
    }
    if deltanet_norm_enabled(be_) {
        skip.extend((0..end).filter_map(|i| deltanet_norm_pair(graph, i).map(|_| i + 1)));
    }
    let mode = RopeMode::Static(&rope_positions);
    let segmented = q8_kv_write_segmented_fusions(be_, graph, scratch, bindings, &mode, &skip)?;
    skip.extend(segmented.keys().map(|&i| i + 1));
    let empty = HashMap::new();
    let mut transient = Vec::new();
    let mut dynamic = Vec::new();
    let mut memo = None;
    let mut stream = PagedStream::default();
    if let Some(chunk) = resident.as_mut() {
        rec.arena_stream_barrier();
        rec.gpu_resident_check(
            chunk.control.as_ref(),
            chunk.control.as_ref(),
            chunk.control.as_ref(),
            chunk.journal.as_ref(),
            [0; 6],
        );
        chunk.begin_conditional(&rec);
        chunk.recording = true;
    }
    let mut recorder = Some(rec);
    let (cap, _) = crate::static_submit_mode(
        be_.shared.submit_dispatch_cap_explicit,
        be_.caps().integrated,
        true,
        be_.submit_dispatch_cap(),
    );
    for (index, op) in infr_core::exec::live_ops(&graph.ops[..end], &skip) {
        if resident.is_some() && cap != 0 && recorder.as_ref().unwrap().dispatches() >= cap {
            return Err(be("prepared resident segment reaches a splitter boundary"));
        }
        if let (
            Some(chunk),
            Op::MoeFfn {
                gate_exps,
                up_exps,
                down_exps,
                ..
            },
        ) = (resident.as_mut(), op)
        {
            let roles = [*gate_exps, *up_exps, *down_exps]
                .map(|id| resolve(scratch, bindings, id).map(crate::pager::buffer_identity));
            let [gate, up, down] = roles;
            let roles = [gate?, up?, down?];
            let layout = ResidentMoeLayout::from_op(graph, op)
                .ok_or_else(|| be("unsupported prepared resident layout"))?;
            chunk.entries.push(ResidentJournalEntry {
                op_idx: index,
                layout,
                roles,
            });
            let ordinal = chunk.entries.len() - 1;
            stream.gpu_route = Some(GpuRoute {
                control: Arc::clone(&chunk.control),
                journal: Arc::clone(&chunk.journal),
                ordinal: ordinal as u32,
                notice: chunk
                    .notices
                    .as_ref()
                    .map(|notices| notices.events[ordinal]),
            });
            execute_paged_moe(
                be_,
                graph,
                index,
                op,
                scratch,
                bindings,
                &mut pool,
                &mut recorder,
                &mut stream,
                shared_ops.get(&index).copied(),
            )?;
            continue;
        }
        lower_op(
            be_,
            graph,
            index,
            op,
            recorder.as_ref().unwrap(),
            scratch,
            bindings,
            &fusions.kv_write,
            &empty,
            &segmented,
            &fusions.linear_add,
            &mode,
            &mut transient,
            &mut dynamic,
            &mut pool,
            &mut memo,
            None,
        )?;
    }
    if !transient.is_empty() || !dynamic.is_empty() {
        return Err(be("prepared prefix produced transient GPU resources"));
    }
    let resident = resident
        .map(|chunk| -> Result<_> {
            let guard = be_.moe_pager().lock().unwrap();
            let session = guard.as_ref().ok_or_else(|| be("missing prepared pager"))?;
            let mut banks = Vec::new();
            for entry in &chunk.entries {
                for &role in &entry.roles {
                    banks.push((role, session.arena_addr(role)?, session.slot_bytes(role)?));
                }
            }
            Ok(PreparedResident {
                chunk,
                tape: session.retained_tape(),
                banks,
                tape_words: stream.tape_cursor,
            })
        })
        .transpose()?;
    let command = recorder.take().unwrap().finish_prepared()?;
    let positions = rope_positions
        .into_iter()
        .map(|(id, position)| (TensorId(id), position))
        .collect();
    Ok(PreparedPrefix {
        command,
        epoch,
        tensors,
        pool: pool_keys,
        positions,
        end,
        resident,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::{DType, TensorDesc};
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn deferred_recording_does_not_block_nested_parallel_uploads() {
        use rayon::prelude::*;
        for threads in [1, 2] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let recorded = AtomicUsize::new(0);
            let copied = AtomicUsize::new(0);
            pool.install(|| {
                rayon::in_place_scope(|scope| {
                    let mut preparation = DeferredPreparation::new(Some(|| {
                        assert_eq!(copied.load(Ordering::Acquire), 128);
                        recorded.fetch_add(1, Ordering::Release);
                    }));
                    let upload = || {
                        (0..64).into_par_iter().for_each(|_| {
                            copied.fetch_add(1, Ordering::AcqRel);
                        })
                    };
                    rayon::join(upload, upload);
                    assert_eq!(recorded.load(Ordering::Acquire), 0);
                    preparation.start(scope);
                    preparation.start(scope);
                })
            });
            assert_eq!(recorded.load(Ordering::Acquire), 1);
        }
    }

    #[test]
    fn position_value_change_invalidates_without_a_buffer_replacement() {
        let expected = [(TensorId(1), 32768), (TensorId(2), 32768)];
        assert!(positions_match(&expected, |_| Some(32768)));
        assert!(!positions_match(&expected, |_| Some(32767)));
        assert!(!positions_match(&expected, |id| (id.0 == 1).then_some(32768)));
        assert!(!positions_match(&expected, |id| Some(if id.0 == 1 {
            32768
        } else {
            32769
        })));
    }

    #[test]
    fn nonpositional_prefix_needs_no_position_readback() {
        assert!(positions_match(&[], |_| panic!(
            "no position access is needed"
        )));
    }

    struct CpuBuffer {
        bytes: AtomicUsize,
        address: AtomicU64,
    }

    impl CpuBuffer {
        fn new(bytes: usize, address: u64) -> Self {
            Self {
                bytes: AtomicUsize::new(bytes),
                address: AtomicU64::new(address),
            }
        }
    }

    impl Buffer for CpuBuffer {
        fn len_bytes(&self) -> usize {
            self.bytes.load(Ordering::Relaxed)
        }

        fn device_addr(&self) -> Option<u64> {
            let address = self.address.load(Ordering::Relaxed);
            (address != 0).then_some(address)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn decode_graph() -> Graph {
        let mut graph = Graph::new();
        let x = graph.input(TensorDesc::new([1, 8], DType::F32));
        let norm = graph.weight(TensorDesc::new([8], DType::F32));
        let weight = graph.weight(TensorDesc::new([8, 8], DType::F32));
        let normalized = graph.internal(TensorDesc::new([1, 8], DType::F32));
        let projected = graph.internal(TensorDesc::new([1, 8], DType::F32));
        let router = graph.weight(TensorDesc::new([4, 8], DType::F32));
        let gate = graph.weight(TensorDesc::new([4, 16, 8], DType::F32));
        let up = graph.weight(TensorDesc::new([4, 16, 8], DType::F32));
        let down = graph.weight(TensorDesc::new([4, 8, 16], DType::F32));
        let dst = graph.output(TensorDesc::new([1, 8], DType::F32));
        graph.push(Op::RmsNorm {
            x,
            weight: norm,
            dst: normalized,
            rows: 1,
            dim: 8,
            eps: 1e-6,
        });
        graph.push(Op::Linear {
            x: normalized,
            weight,
            dst: projected,
            m: 1,
            in_f: 8,
            out_f: 8,
            w_off: 0,
        });
        graph.push(Op::MoeFfn {
            x: projected,
            router_x: projected,
            router,
            gate_exps: gate,
            up_exps: up,
            down_exps: down,
            down_scale: None,
            fused_gate_up: false,
            dst,
            ne: 8,
            n_expert: 4,
            n_used: 2,
            n_ff_exp: 16,
            scale: 1.0,
            act: Activation::Silu,
            swiglu_clamp: None,
            gating: infr_core::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
            expert_ids: None,
        });
        graph
    }

    fn cpu_scratch(graph: &Graph) -> ScratchSet {
        scratch_layout(graph)
            .unwrap()
            .into_iter()
            .map(|bytes| bytes.map(|bytes| Box::new(CpuBuffer::new(bytes, 0)) as Box<dyn Buffer>))
            .collect()
    }

    #[test]
    fn recording_guard_restores_state_after_success_and_error() {
        assert!(!preparing_commands());
        {
            let _guard = RecordingGuard::enter();
            assert!(preparing_commands());
        }
        assert!(!preparing_commands());
        let result = (|| -> Result<()> {
            let _guard = RecordingGuard::enter();
            assert!(preparing_commands());
            Err(be("CPU preparation failure"))
        })();
        assert!(result.is_err());
        assert!(!preparing_commands());
    }

    #[test]
    fn recording_guard_restores_state_on_error_and_unwind() {
        assert!(!preparing_commands());
        let result = std::panic::catch_unwind(|| {
            let _guard = RecordingGuard::enter();
            assert!(preparing_commands());
            panic!("scoped preparation");
        });
        assert!(result.is_err());
        assert!(!preparing_commands());
    }

    #[test]
    fn recording_guard_is_thread_local() {
        let _guard = RecordingGuard::enter();
        std::thread::spawn(|| {
            assert!(!preparing_commands());
            {
                let _guard = RecordingGuard::enter();
                assert!(preparing_commands());
            }
            assert!(!preparing_commands());
        })
        .join()
        .unwrap();
        assert!(preparing_commands());
    }

    #[test]
    fn single_row_prefix_stops_before_first_router() {
        let mut graph = decode_graph();
        assert_eq!(prefix_end(&graph), Some(2));
        let x = graph.inputs[0];
        graph.push(Op::Argmax {
            x,
            dst: x,
            n: 8,
            rows: 1,
        });
        assert_eq!(prefix_end(&graph), Some(2));
        let second_moe = graph.ops[2].clone();
        graph.push(second_moe);
        assert_eq!(prefix_end(&graph), Some(2));
    }

    #[test]
    fn resident_prefix_stops_at_the_existing_layer_submit_boundary() {
        let mut graph = decode_graph();
        assert_eq!(resident_end(&graph, 1), Some(3));
        assert_eq!(resident_end(&graph, 0), None);
        assert_eq!(resident_end(&graph, 2), None);
        graph.ops.extend(graph.ops.clone());
        assert_eq!(resident_end(&graph, 2), Some(6));
        assert_eq!(resident_end(&graph, 1), Some(3));
        let x = graph.inputs[0];
        graph.ops.insert(
            3,
            Op::Argmax {
                x,
                dst: x,
                n: 8,
                rows: 1,
            },
        );
        assert_eq!(resident_end(&graph, 1), Some(3));
        assert_eq!(resident_end(&graph, 2), None);
    }

    #[test]
    fn resident_prefix_keeps_unsupported_moe_layouts_on_the_old_path() {
        let mut graph = decode_graph();
        if let Op::MoeFfn { fused_gate_up, .. } = &mut graph.ops[2] {
            *fused_gate_up = true;
        }
        assert_eq!(prefix_end(&graph), Some(2));
        assert_eq!(resident_end(&graph, 1), None);
    }

    #[test]
    fn resident_prefix_accepts_the_single_row_shared_expert_tail() {
        let mut graph = decode_graph();
        graph.ops.extend(graph.ops.clone());
        let x = graph.inputs[0];
        graph.ops.insert(
            3,
            Op::MoeSharedExpertAdd {
                moe: x,
                shexp: x,
                gate: x,
                dst: x,
                rows: 1,
                n: 8,
            },
        );
        assert_eq!(resident_end(&graph, 2), Some(7));
        if let Op::MoeSharedExpertAdd { rows, .. } = &mut graph.ops[3] {
            *rows = 2;
        }
        assert_eq!(resident_end(&graph, 2), None);
    }

    #[test]
    fn prefix_requires_both_a_prefix_and_a_router() {
        assert_eq!(prefix_end(&Graph::new()), None);
        let mut graph = decode_graph();
        graph.ops.truncate(2);
        assert_eq!(prefix_end(&graph), None);
        let mut graph = decode_graph();
        drop(graph.ops.drain(..2));
        assert_eq!(prefix_end(&graph), None);
    }

    #[test]
    fn verify_and_each_prefill_marker_reject_an_otherwise_supported_prefix() {
        for marker in 0..3 {
            let mut graph = decode_graph();
            match marker {
                0 => graph.mtp_verify = true,
                1 => graph.prefill_target_tokens = Some(1),
                _ => graph.prefill_next_chunk = true,
            }
            assert_eq!(prefix_end(&graph), None, "marker {marker}");
        }
    }

    #[test]
    fn every_moe_input_must_have_exactly_one_nonempty_row() {
        for elements in [0, 7, 9, 16, 64] {
            for later_layer in [false, true] {
                let mut graph = decode_graph();
                let x = graph.input(TensorDesc::new([elements], DType::F32));
                let mut moe = graph.ops[2].clone();
                if let Op::MoeFfn {
                    x: input, router_x, ..
                } = &mut moe
                {
                    *input = x;
                    *router_x = x;
                }
                if later_layer {
                    graph.push(moe);
                } else {
                    graph.ops[2] = moe;
                }
                assert_eq!(
                    prefix_end(&graph),
                    None,
                    "elements={elements}, later={later_layer}"
                );
            }
        }
        let mut graph = decode_graph();
        let x = graph.input(TensorDesc::new([0], DType::F32));
        if let Op::MoeFfn { x: input, ne, .. } = &mut graph.ops[2] {
            *input = x;
            *ne = 0;
        }
        assert_eq!(prefix_end(&graph), None);
    }

    #[test]
    fn multirow_linear_and_unsupported_operations_reject_prefix() {
        let mut graph = decode_graph();
        if let Op::Linear { m, .. } = &mut graph.ops[1] {
            *m = 2;
        }
        assert_eq!(prefix_end(&graph), None);
        let mut graph = decode_graph();
        let x = graph.inputs[0];
        graph.ops.insert(
            1,
            Op::Argmax {
                x,
                dst: x,
                n: 8,
                rows: 1,
            },
        );
        assert_eq!(prefix_end(&graph), None);
    }

    #[test]
    fn mrope_prefix_remains_on_ordinary_recording_path() {
        let mut graph = decode_graph();
        let x = graph.inputs[0];
        let weight = graph.weights[0];
        let positions4 = graph.input(TensorDesc::new([1, 4], DType::I32));
        graph.ops.insert(
            0,
            Op::QkNormMrope {
                x,
                weight,
                positions4,
                dst: x,
                rows: 1,
                n_head: 1,
                head_dim: 8,
                rope_dim: 8,
                theta: 10000.0,
                eps: 1e-6,
                sections: [2; 4],
                x_stride: 0,
            },
        );
        assert_eq!(prefix_end(&graph), None);
    }

    #[test]
    fn compatibility_requires_scratch_capacity_and_single_sequence_graphs() {
        let current = decode_graph();
        let next = current.clone();
        let mut scratch = cpu_scratch(&next);
        assert!(compatible(&current, &next, &scratch));
        let index = scratch.iter().position(Option::is_some).unwrap();
        let required = scratch[index].as_ref().unwrap().len_bytes();
        scratch[index] = Some(Box::new(CpuBuffer::new(required - 1, 0)));
        assert!(!compatible(&current, &next, &scratch));
        scratch[index] = Some(Box::new(CpuBuffer::new(required + 256, 0)));
        assert!(compatible(&current, &next, &scratch));
        scratch[index] = None;
        assert!(!compatible(&current, &next, &scratch));
        let mut short = cpu_scratch(&next);
        short.pop();
        assert!(!compatible(&current, &next, &short));
        let scratch = cpu_scratch(&next);
        let mut independent = next.clone();
        independent.independent_rows = true;
        assert!(!compatible(&current, &independent, &scratch));
        assert!(!compatible(&independent, &next, &scratch));
    }

    #[test]
    fn row_keys_invalidate_on_reorder_replacement_or_row_count_change() {
        let rows = [CpuBuffer::new(32, 0x1000), CpuBuffer::new(32, 0x2000)];
        let replacement = CpuBuffer::new(32, 0x2000);
        let scratch: ScratchSet = vec![None];
        let id = TensorId(0);
        let mut bindings = Bindings::new();
        bindings.bind_rows(id, vec![&rows[0], &rows[1]]);
        let expected = keys(&scratch, &bindings, id).unwrap();
        assert_eq!(keys(&scratch, &bindings, id).unwrap(), expected);
        bindings.bind_rows(id, vec![&rows[1], &rows[0]]);
        assert_ne!(keys(&scratch, &bindings, id).unwrap(), expected);
        bindings.bind_rows(id, vec![&rows[0], &replacement]);
        assert_ne!(keys(&scratch, &bindings, id).unwrap(), expected);
        bindings.bind_rows(id, vec![&rows[0]]);
        assert_ne!(keys(&scratch, &bindings, id).unwrap(), expected);
        bindings.bind_rows(id, vec![]);
        assert_ne!(keys(&scratch, &bindings, id).unwrap(), expected);
    }

    #[test]
    fn row_keys_detect_size_and_address_changes_at_the_same_object_pointer() {
        let row = CpuBuffer::new(32, 0x1000);
        let scratch: ScratchSet = vec![None];
        let id = TensorId(0);
        let mut bindings = Bindings::new();
        bindings.bind_rows(id, vec![&row]);
        let expected = keys(&scratch, &bindings, id).unwrap();
        row.bytes.store(64, Ordering::Relaxed);
        let resized = keys(&scratch, &bindings, id).unwrap();
        assert_eq!(resized[0].pointer, expected[0].pointer);
        assert_ne!(resized, expected);
        row.bytes.store(32, Ordering::Relaxed);
        for address in [0, 0x2000] {
            row.address.store(address, Ordering::Relaxed);
            let changed = keys(&scratch, &bindings, id).unwrap();
            assert_eq!(changed[0].pointer, expected[0].pointer);
            assert_ne!(changed, expected);
        }
        row.address.store(0x1000, Ordering::Relaxed);
        assert_eq!(keys(&scratch, &bindings, id).unwrap(), expected);
    }

    #[test]
    fn tensor_keys_resolve_scratch_single_bindings_and_missing_bindings() {
        let id = TensorId(0);
        let bound = CpuBuffer::new(32, 0x1000);
        let scratch: ScratchSet = vec![Some(Box::new(CpuBuffer::new(64, 0x2000)))];
        let mut bindings = Bindings::new();
        bindings.bind(id, &bound);
        assert_eq!(
            keys(&vec![None], &bindings, id).unwrap(),
            vec![buffer_key(&bound)]
        );
        assert_eq!(
            keys(&scratch, &bindings, id).unwrap(),
            vec![buffer_key(scratch[0].as_deref().unwrap())]
        );
        assert!(keys(&vec![None], &Bindings::new(), id).is_err());
    }
}
