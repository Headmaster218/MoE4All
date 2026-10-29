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

/// Copy workers may enter global Rayon while retaining the pager lock. A ready recording
/// also needs that lock, so it must never be stolen by a copy worker's nested join.
pub(super) fn recording_pool() -> Result<&'static rayon::ThreadPool> {
    static POOL: OnceLock<std::result::Result<rayon::ThreadPool, String>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "infr-command-prepare".into())
            .build()
            .map_err(|error| error.to_string())
    })
    .as_ref()
    .map_err(|error| be(format!("command preparation worker: {error}")))
}

pub(super) fn recording_scope<'scope, R>(
    isolated: bool,
    run: impl FnOnce(&rayon::Scope<'scope>) -> R,
) -> Result<R> {
    Ok(if isolated {
        recording_pool()?.in_place_scope(run)
    } else {
        rayon::in_place_scope(run)
    })
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RecordingCheckpoint {
    Setup,
    Op(usize),
    Finalize,
}

fn recording_checkpoint(cancelled: Option<&AtomicBool>, _point: RecordingCheckpoint) -> Result<()> {
    #[cfg(test)]
    if let Some(cancelled) = cancelled {
        if TEST_CANCEL_AT.with(|point| point.get() == Some(_point)) {
            cancelled.store(true, Ordering::Release);
        }
    }
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::Aborted);
    }
    Ok(())
}

fn cancellable_recording<T>(
    cancelled: &AtomicBool,
    record: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    if cancelled.load(Ordering::Acquire) {
        return Ok(None);
    }
    match record() {
        Ok(packet) => Ok((!cancelled.load(Ordering::Acquire)).then_some(packet)),
        Err(Error::Aborted) if cancelled.load(Ordering::Acquire) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Successful NEXT and following-stage work remains useful; errors cancel before joining.
pub(super) fn finish_preparation_scope(
    next_cancelled: &AtomicBool,
    continuation_cancelled: &AtomicBool,
    failed: bool,
) {
    if failed {
        next_cancelled.store(true, Ordering::Release);
        continuation_cancelled.store(true, Ordering::Release);
    }
}

#[cfg(test)]
pub(super) fn test_finish_preparation_scope(
    next_cancelled: &AtomicBool,
    continuation_cancelled: &AtomicBool,
    failed: bool,
    experimental_end_cancel: bool,
) {
    finish_preparation_scope(next_cancelled, continuation_cancelled, failed || experimental_end_cancel);
}

#[cfg(test)]
fn source_horizon_overrides() -> &'static Mutex<HashMap<usize, bool>> {
    static STATES: OnceLock<Mutex<HashMap<usize, bool>>> = OnceLock::new();
    STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
pub(super) struct SourceHorizonScope {
    shared: Arc<crate::VulkanShared>,
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for SourceHorizonScope {
    fn drop(&mut self) {
        let key = Arc::as_ptr(&self.shared) as usize;
        let mut states = source_horizon_overrides().lock().unwrap();
        states.remove(&key);
        if let Some(previous) = self.previous {
            states.insert(key, previous);
        }
    }
}

/// Device-scoped so the isolated recording worker observes the same A/B choice.
#[cfg(test)]
pub(super) fn source_horizon_scope(be_: &VulkanBackend, enabled: bool) -> SourceHorizonScope {
    let previous = source_horizon_overrides()
        .lock()
        .unwrap()
        .insert(Arc::as_ptr(&be_.shared) as usize, enabled);
    SourceHorizonScope {
        shared: Arc::clone(&be_.shared),
        previous,
    }
}

#[cfg(test)]
thread_local! {
    static TEST_CANCEL_AT: Cell<Option<RecordingCheckpoint>> = const { Cell::new(None) };
    static TEST_ACCEPTED_PREFIXES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn test_accepted_prefixes() -> usize {
    TEST_ACCEPTED_PREFIXES.with(Cell::get)
}

#[cfg(test)]
pub(super) struct TestCancelAt(Option<RecordingCheckpoint>);

#[cfg(test)]
impl TestCancelAt {
    pub(super) fn enter(point: RecordingCheckpoint) -> Self {
        Self(TEST_CANCEL_AT.with(|current| current.replace(Some(point))))
    }
}

#[cfg(test)]
impl Drop for TestCancelAt {
    fn drop(&mut self) {
        TEST_CANCEL_AT.with(|current| current.set(self.0));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BufferKey {
    pointer: usize,
    bytes: usize,
    address: Option<u64>,
    segmented: Option<(u64, usize, usize, usize)>,
}
fn buffer_key(buffer: &dyn Buffer) -> BufferKey {
    use ash::vk::Handle;
    BufferKey {
        pointer: (buffer as *const dyn Buffer as *const ()) as usize,
        bytes: buffer.len_bytes(),
        address: buffer.device_addr(),
        segmented: crate::as_segmented_kv(buffer).map(|kv| {
            let table = kv.table_buffer();
            (
                table.buffer.as_raw(),
                table.sub_offset,
                kv.spec().segment_elements,
                kv.committed(),
            )
        }),
    }
}

pub(super) struct PreparedPrefix {
    command: crate::recorder::PreparedCommand,
    parts: Vec<PreparedPart>,
    head_end: usize,
    head_layers: usize,
    epoch: u64,
    tensors: Vec<(TensorId, Vec<BufferKey>)>,
    pool: Vec<(ScratchKey, u64, usize, usize)>,
    positions: Vec<(TensorId, Vec<usize>)>,
    pub(super) end: usize,
    resident: Option<PreparedResident>,
}

struct PreparedResident {
    chunk: ResidentChunk,
    tape: Arc<dyn Buffer>,
    banks: Vec<(usize, u64, usize)>,
    tape_words: usize,
    copy_params: Vec<([usize; 3], [u32; 9])>,
}

pub(super) struct ActivatedPrefix {
    pub(super) command: crate::recorder::PreparedCommand,
    pub(super) parts: Vec<PreparedPart>,
    pub(super) head_end: usize,
    pub(super) head_layers: usize,
    pub(super) end: usize,
    pub(super) resident: Option<ResidentChunk>,
    pub(super) tape_words: usize,
}

pub(super) struct PreparedPart {
    pub(super) command: crate::recorder::PreparedCommand,
    pub(super) end: usize,
    pub(super) layers: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecordingHorizon {
    Bounded,
    Wide,
}

impl RecordingHorizon {
    fn for_next_stage(stage: usize) -> Self {
        if stage == 0 {
            Self::Wide
        } else {
            Self::Bounded
        }
    }
}

/// A stage packet. Aliases are valid only while the outer forward pins every
/// retained topology and pool capacity; no workspace begin/finish may run before this joins.
pub(super) struct StageRecording<'a> {
    plan: &'a VkDecodePlan,
    bindings: &'a Bindings<'a>,
    scratch: ScratchSet,
    canonical: Vec<Option<BufferKey>>,
    pool: ScratchPool,
    resident: Option<ResidentChunk>,
    positions: Vec<usize>,
    output: &'a Mutex<Option<PreparedPrefix>>,
    next_token: bool,
    horizon: RecordingHorizon,
}

/// An idle, producer-free template captured before any live prefix is activated.
/// Keeping construction private prevents rejected attachments from resetting a live producer.
pub(super) struct ResidentPreparationSnapshot {
    chunk: ResidentChunk,
}

impl ResidentPreparationSnapshot {
    pub(super) fn capture(template: &ResidentChunk) -> Option<Self> {
        if !resident_preparation_idle(template) {
            return None;
        }
        let chunk = template.preparation_snapshot()?;
        resident_preparation_detached(&chunk).then_some(Self { chunk })
    }
}

fn resident_preparation_idle(chunk: &ResidentChunk) -> bool {
    chunk.entries.is_empty()
        && !chunk.recording
        && chunk.recovery_op.is_none()
        && chunk.submitted_entries == 0
        && chunk.notice_cursor == 0
        && chunk.notice_resume.is_none()
        && !chunk.notice_stop
        && chunk.notices_sent == 0
        && chunk.owned_notice_acks.is_empty()
}

fn resident_preparation_detached(chunk: &ResidentChunk) -> bool {
    resident_preparation_idle(chunk)
        && chunk.producer.is_none()
        && chunk.waiter.is_none()
        && chunk.indirect.is_none()
}

fn resident_preparation_matches(
    chunk: &ResidentChunk,
    retained: &ResidentBuffers,
    requested: bool,
    limit: usize,
    submit_layers: usize,
    mapped_serial_max_bytes: usize,
) -> bool {
    let copy = retained.selected_gpu_copy(requested);
    let owned = copy.is_some_and(|copy| copy.is_owned());
    resident_preparation_detached(chunk)
        && Arc::ptr_eq(&chunk.control, &retained.control)
        && Arc::ptr_eq(&chunk.journal, &retained.journal)
        && match (&chunk.notices, &retained.notices) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
        && copy_binding_matches(copy, chunk.gpu_copy.as_ref())
        && chunk.self_service == copy.is_some_and(|copy| copy.owned_self_service())
        && (copy.is_none()
            || chunk.self_service
            || (retained.producer.is_some() && chunk.notices.is_some()))
        && retained.indirect.is_none()
        && limit != 0
        && limit <= retained.capacity
        && chunk.limit == limit
        && chunk.submit_layers == submit_layers
        && chunk.mapped_serial_max_bytes == mapped_serial_max_bytes
        && resident_cadence_supported(submit_layers, limit, owned)
}

/// Declared outside the recording scope so started workers join before cancellation clears output.
pub(super) struct PreparedOutputGuard<'a> {
    output: Option<&'a Mutex<Option<PreparedPrefix>>>,
    committed: bool,
}

impl<'a> PreparedOutputGuard<'a> {
    pub(super) fn new(output: Option<&'a Mutex<Option<PreparedPrefix>>>) -> Self {
        Self {
            output,
            committed: false,
        }
    }
    pub(super) fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for PreparedOutputGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            if let Some(output) = self.output {
                output.lock().unwrap().take();
            }
        }
    }
}

impl<'a> StageRecording<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn snapshot(
        be_: &VulkanBackend,
        plan: &'a VkDecodePlan,
        bindings: &'a Bindings<'a>,
        scratch: &ScratchSet,
        pool: &mut ScratchPool,
        resident: Option<&ResidentChunk>,
        positions: &HashMap<u32, usize>,
        output: &'a Mutex<Option<PreparedPrefix>>,
    ) -> Result<Option<Self>> {
        let _span = infr_core::timeline::span("staged_packet_snapshot");
        if !enabled(be_) || prefix_end(&plan.graph).is_none() {
            return Ok(None);
        }
        let lane_positions = if plan.graph.independent_rows {
            let Some(lanes) = independent_positions(&plan.graph) else {
                return Ok(None);
            };
            lanes
        } else {
            let position = positions.values().next().copied().unwrap_or(0);
            if positions.values().any(|&value| value != position) {
                return Ok(None);
            }
            vec![position]
        };
        if !kv_depth_ready(&plan.graph, scratch, bindings, &lane_positions) {
            return Ok(None);
        }
        let mut aliases = Vec::with_capacity(scratch.len());
        let mut canonical = Vec::with_capacity(scratch.len());
        for buffer in scratch {
            canonical.push(buffer.as_deref().map(buffer_key));
            aliases.push(match buffer {
                // SAFETY: stages retain the original scratch until all scoped recording and
                // submitted work drains. These objects never own or release the allocation.
                Some(buffer) => {
                    Some(
                        Box::new(unsafe { crate::as_vk_buf(buffer.as_ref())?.recording_alias() })
                            as Box<dyn Buffer>,
                    )
                }
                None => None,
            });
        }
        Ok(Some(Self {
            plan,
            bindings,
            scratch: aliases,
            canonical,
            pool: snapshot_pool(pool)?,
            resident: resident.and_then(ResidentChunk::preparation_snapshot),
            positions: lane_positions,
            output,
            next_token: false,
            horizon: RecordingHorizon::Bounded,
        }))
    }

    pub(super) fn snapshot_next_token(
        be_: &VulkanBackend,
        plan: &'a VkDecodePlan,
        bindings: &'a Bindings<'a>,
        scratch: &ScratchSet,
        pool: &mut ScratchPool,
        positions: &[usize],
    ) -> Result<Option<Self>> {
        if !next_positions_supported(&plan.graph, positions)
            || !kv_depth_ready(&plan.graph, scratch, bindings, positions)
        {
            return Ok(None);
        }
        let expected = plan
            .setup_analysis
            .rope_ids
            .iter()
            .map(|id| (id.0, positions[0]))
            .collect();
        let Some(mut packet) = Self::snapshot(
            be_,
            plan,
            bindings,
            scratch,
            pool,
            None,
            &expected,
            &plan.prepared_prefix,
        )?
        else {
            return Ok(None);
        };
        packet.positions = positions.to_vec();
        packet.next_token = true;
        packet.horizon = RecordingHorizon::for_next_stage(0);
        Ok(Some(packet))
    }

    pub(super) fn set_next_stage_horizon(&mut self, stage: usize) {
        debug_assert!(self.next_token);
        self.horizon = RecordingHorizon::for_next_stage(stage);
    }

    pub(super) fn is_next_token(&self) -> bool {
        self.next_token
    }

    pub(super) fn next_token_output(&self) -> Option<&'a Mutex<Option<PreparedPrefix>>> {
        self.next_token.then_some(self.output)
    }

    /// Copy only the active protocol's handles; never construct/reset a speculative live owner.
    pub(super) fn attach_next_resident(
        &mut self,
        be_: &VulkanBackend,
        current: Option<&ResidentChunk>,
        retained: Option<&ResidentBuffers>,
    ) {
        if self.resident.is_some()
            || !self.next_token
            || !be_.cfg().device.decode_command_early_submit
            || !be_.cfg().device.decode_command_resident
            || !resident_mtp_graph_supported(be_, &self.plan.graph, &self.scratch, self.bindings)
        {
            return;
        }
        let Some(layouts) = resident_graph_layouts(be_, &self.plan.graph, be_.moe_paged()) else {
            return;
        };
        let requested = resident_copy_requested(
            be_.cfg().kernels.vulkan.decode_gpu_resident_copy,
            &self.plan.graph,
            &layouts,
        );
        let Some(mut snapshot) = current.and_then(ResidentChunk::preparation_snapshot) else {
            return;
        };
        let Some(retained) = retained else { return };
        snapshot.gpu_copy = retained.selected_gpu_copy(requested).cloned();
        self.resident = Some(snapshot);
    }

    /// Consume a preadmitted template without snapshotting it again or changing its copy owner.
    /// A false result leaves this packet unchanged; the caller may discard the optional packet.
    pub(super) fn attach_next_resident_snapshot(
        &mut self,
        be_: &VulkanBackend,
        snapshot: ResidentPreparationSnapshot,
        retained: &ResidentBuffers,
    ) -> bool {
        if self.resident.is_some()
            || !self.next_token
            || !be_.cfg().device.decode_command_early_submit
            || !be_.cfg().device.decode_command_resident
            || !resident_mtp_graph_supported(be_, &self.plan.graph, &self.scratch, self.bindings)
        {
            return false;
        }
        let cfg = &be_.cfg().kernels.vulkan;
        let Some(layouts) = resident_graph_layouts(be_, &self.plan.graph, be_.moe_paged()) else {
            return false;
        };
        let requested =
            resident_copy_requested(cfg.decode_gpu_resident_copy, &self.plan.graph, &layouts);
        let copy = retained.selected_gpu_copy(requested);
        let owned = copy.is_some_and(|copy| copy.is_owned());
        let capacity = if let Some(copy) = copy.filter(|copy| copy.is_owned()) {
            retained.capacity.min(resident_owned_protocol_capacity(
                retained.control.len_bytes(),
                cfg.decode_gpu_resident_immutable_predicates,
                retained.journal.len_bytes(),
                retained
                    .notices
                    .as_ref()
                    .map_or(crate::recorder::RESIDENT_MAX_LAYERS, |notices| {
                        notices.events.len()
                    }),
                copy.jobs.len_bytes(),
                copy.miss_predicates
                    .as_ref()
                    .map(|buffer| buffer.len_bytes()),
            ))
        } else {
            retained.capacity
        };
        let Ok(limit) = resident_execution_limit(
            owned,
            layouts.len(),
            cfg.decode_gpu_resident_layers.clamp(1, 16),
            capacity,
        ) else {
            return false;
        };
        let submit_layers = match cfg.decode_gpu_resident_submit_layers {
            0 => self.plan.graph.resident_submit_layers.unwrap_or(1),
            explicit => explicit,
        };
        if !resident_preparation_matches(
            &snapshot.chunk,
            retained,
            requested,
            limit,
            submit_layers,
            resident_mapped_serial_max_bytes(
                cfg.decode_gpu_resident_mapped_serial_max_kib,
                &self.plan.graph,
            ),
        ) {
            return false;
        }
        self.resident = Some(snapshot.chunk);
        true
    }

    pub(super) fn record(self, be_: &VulkanBackend, cancelled: &AtomicBool) -> Result<()> {
        let _span = infr_core::timeline::background_span(if self.next_token {
            "staged_next_packet_record"
        } else {
            "staged_packet_record"
        });
        if cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        // Bound continuation work independently of next-token activation and position guards.
        // Standalone/first-stage next packets retain the existing wide recording policy.
        let prefix = if self.horizon == RecordingHorizon::Wide {
            record_cancellable(
                be_,
                self.plan,
                &self.scratch,
                self.bindings,
                &self.positions,
                self.pool,
                self.resident,
                cancelled,
            )?
        } else {
            cancellable_recording(cancelled, || {
                record_with_horizon(
                    be_,
                    self.plan,
                    &self.scratch,
                    self.bindings,
                    &self.positions,
                    self.pool,
                    self.resident,
                    false,
                    Some(cancelled),
                )
            })?
        };
        let Some(mut prefix) = prefix else {
            return Ok(());
        };
        // Recording aliases have different Rust object identities. Activation must validate
        // against the real retained graph scratch, not against those temporary wrappers.
        for (id, expected) in &mut prefix.tensors {
            if cancelled.load(Ordering::Acquire) {
                return Ok(());
            }
            if self.bindings.get_rows(*id).is_none() {
                if let Some(Some(canonical)) = self.canonical.get(id.0 as usize) {
                    *expected = vec![canonical.clone()];
                }
            }
        }
        if !cancelled.load(Ordering::Acquire) {
            *self.output.lock().unwrap() = Some(prefix);
        }
        Ok(())
    }
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
        } else if !supported_op_rows(op, packet_rows(graph)?) {
            return None;
        }
    }
    None
}

fn positions_match(
    expected: &[(TensorId, Vec<usize>)],
    mut actual: impl FnMut(TensorId, usize) -> Option<Vec<usize>>,
) -> bool {
    expected
        .iter()
        .all(|(id, positions)| actual(*id, positions.len()).as_ref() == Some(positions))
}

fn mapped_positions(buffer: &dyn Buffer, rows: usize) -> Option<Vec<usize>> {
    let bytes = rows.checked_mul(4)?;
    if buffer.len_bytes() < bytes {
        return None;
    }
    let pointer = crate::as_vk_buf(buffer).ok()?.mapped_ptr()?;
    let mut data = vec![0u8; bytes];
    // The execution gate and caller retain this coherent position buffer through validation.
    unsafe { std::ptr::copy_nonoverlapping(pointer.cast_const(), data.as_mut_ptr(), bytes) };
    Some(
        data.chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize)
            .collect(),
    )
}

fn keys(scratch: &ScratchSet, bindings: &Bindings, id: TensorId) -> Result<Vec<BufferKey>> {
    if let Some(rows) = bindings.get_rows(id) {
        return Ok(rows.iter().map(|buffer| buffer_key(*buffer)).collect());
    }
    Ok(vec![buffer_key(resolve(scratch, bindings, id)?)])
}

fn ack_buffer<'a>(
    scratch: &'a ScratchSet,
    bindings: &'a Bindings<'_>,
    id: TensorId,
) -> Option<&'a dyn Buffer> {
    match scratch.get(id.0 as usize)? {
        Some(buffer) => Some(buffer.as_ref()),
        None => bindings.get(id),
    }
}

fn key_sequence_matches(
    expected: &[BufferKey],
    actual: impl ExactSizeIterator<Item = BufferKey>,
) -> bool {
    expected.len() == actual.len() && expected.iter().zip(actual).all(|(a, b)| *a == b)
}

fn keys_match(
    scratch: &ScratchSet,
    bindings: &Bindings,
    id: TensorId,
    expected: &[BufferKey],
) -> bool {
    if let Some(rows) = bindings.get_rows(id) {
        return key_sequence_matches(expected, rows.iter().map(|buffer| buffer_key(*buffer)));
    }
    ack_buffer(scratch, bindings, id)
        .is_some_and(|buffer| key_sequence_matches(expected, std::iter::once(buffer_key(buffer))))
}

fn position_byte_len(rows: usize, available: usize) -> Option<usize> {
    rows.checked_mul(4)
        .filter(|&bytes| bytes <= available && bytes <= isize::MAX as usize)
}

fn position_bytes_match(data: &[u8], expected: &[usize]) -> bool {
    let Some(bytes) = position_byte_len(expected.len(), data.len()) else {
        return false;
    };
    data[..bytes]
        .chunks_exact(4)
        .zip(expected)
        .all(|(value, &expected)| {
            u32::from_le_bytes(value.try_into().unwrap()) as usize == expected
        })
}

fn mapped_positions_match(buffer: &dyn Buffer, expected: &[usize]) -> bool {
    let Some(bytes) = position_byte_len(expected.len(), buffer.len_bytes()) else {
        return false;
    };
    let Some(pointer) = crate::as_vk_buf(buffer)
        .ok()
        .and_then(|buffer| buffer.mapped_ptr())
    else {
        return false;
    };
    if pointer.is_null() {
        return false;
    }
    // Same execution-gate/coherent-mapping lifetime as capture; compare without two Vecs.
    let data = unsafe { std::slice::from_raw_parts(pointer.cast_const(), bytes) };
    position_bytes_match(data, expected)
}

fn ack_positions_match(
    scratch: &ScratchSet,
    bindings: &Bindings,
    expected: &[(TensorId, Vec<usize>)],
) -> bool {
    expected.iter().all(|(id, positions)| {
        ack_buffer(scratch, bindings, *id)
            .is_some_and(|buffer| mapped_positions_match(buffer, positions))
    })
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
        && !be_.cfg().spec.lookup
        && !be_.cfg().paging.expert_prefetch
}

pub(crate) fn enabled(be_: &VulkanBackend) -> bool {
    supported(be_) && !be_.shared.submit_tune_active.load(Ordering::Acquire)
}

pub(super) fn prefix_end(graph: &Graph) -> Option<usize> {
    let rows = packet_rows(graph)?;
    let end = graph
        .ops
        .iter()
        .position(|op| matches!(op, Op::MoeFfn { .. }))?;
    if graph.ops.iter().any(|op| match op {
        Op::MoeFfn { x, ne, .. } => *ne == 0 || graph.desc(*x).numel() != rows * *ne as usize,
        _ => false,
    }) {
        return None;
    }
    (end > 0
        && graph.ops[..end]
            .iter()
            .all(|op| supported_op_rows(op, rows)))
    .then_some(end)
}

#[cfg(test)]
fn supported_op(op: &Op) -> bool {
    supported_op_rows(op, 1)
}

pub(super) fn supported_op_rows(op: &Op, rows: usize) -> bool {
    match op {
        Op::Linear { m, .. } | Op::LinearPair { m, .. } => return *m as usize == rows,
        Op::QsaPrepare { rows: n, .. }
        | Op::QsaIndexer { rows: n, .. }
        | Op::QsaBatchAttention { rows: n, .. }
        | Op::Attention { rows: n, .. }
        | Op::WriteKv { rows: n, .. }
        | Op::Conv1dSilu { rows: n, .. }
        | Op::DeltaNet { rows: n, .. }
        | Op::MoeSharedExpertAdd { rows: n, .. } => return *n as usize == rows,
        Op::QkNorm { rows: n, .. }
        | Op::GatedRmsNorm { rows: n, .. }
        | Op::QkNormRope { rows: n, .. }
        | Op::Rope { rows: n, .. }
        | Op::QwenHcNorm { rows: n, .. }
        | Op::QwenHcMix { rows: n, .. }
        | Op::QwenHcInject { rows: n, .. }
        | Op::QwenHcInjectNorm { rows: n, .. }
        | Op::QwenPleGate { rows: n, .. } => return *n as usize == rows,
        Op::QwenHcDownInject { .. } => return rows == 1,
        _ => {}
    }
    matches!(
        op,
        Op::EmbedGather { .. }
            | Op::Copy { .. }
            | Op::CopyStrided { .. }
            | Op::RmsNorm { .. }
            | Op::RmsNormAdd { .. }
            | Op::LayerNorm { .. }
            | Op::QsaGather { .. }
            | Op::GatedAct { .. }
            | Op::GatedActFused { .. }
            | Op::Silu { .. }
            | Op::Gelu { .. }
            | Op::Add { .. }
            | Op::AddBias { .. }
            | Op::MulVec { .. }
            | Op::Scale { .. }
            | Op::HeadwiseSigmoidMul { .. }
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

fn independent_positions(graph: &Graph) -> Option<Vec<usize>> {
    let rows = decode_token_rows(graph)?;
    if !(1..=infr_core::backend::MAX_DECODE_ROWS).contains(&rows) {
        return None;
    }
    Some(
        sequence_spans(graph, rows)
            .ok()?
            .iter()
            .map(|span| span.start_pos as usize)
            .collect(),
    )
}

fn packet_rows(graph: &Graph) -> Option<usize> {
    if graph.mtp_verify || graph.prefill_target_tokens.is_some() || graph.prefill_next_chunk {
        return None;
    }
    if graph.independent_rows {
        Some(independent_positions(graph)?.len())
    } else {
        Some(1)
    }
}

pub(super) fn next_positions_supported(graph: &Graph, positions: &[usize]) -> bool {
    packet_rows(graph) == Some(positions.len())
        && positions.iter().all(|&p| p < u32::MAX as usize)
        && (!graph.independent_rows || independent_positions(graph).as_deref() == Some(positions))
}

fn committed_depth_fits(
    position: usize,
    ratio: usize,
    stride: usize,
    capacity: usize,
    committed: usize,
) -> bool {
    if ratio == 0 || stride == 0 {
        return false;
    }
    position
        .checked_add(1)
        .and_then(|depth| (depth / ratio).min(capacity / stride).checked_mul(stride))
        .is_some_and(|required| required <= committed)
}

// A predicted next token cannot grow the live KV table from the recording worker.
// Activation separately keys both its table identity and committed segment count.
fn kv_depth_ready(
    graph: &Graph,
    scratch: &ScratchSet,
    bindings: &Bindings,
    positions: &[usize],
) -> bool {
    let check = |id: TensorId, stride: usize, ratio: usize| -> bool {
        if scratch.get(id.0 as usize).is_some_and(Option::is_some) {
            return true;
        }
        let buffers = if graph.independent_rows {
            let Some(rows) = bindings.get_rows(id) else {
                return false;
            };
            if rows.len() != positions.len() {
                return false;
            }
            rows.to_vec()
        } else {
            let Ok(buffer) = resolve(scratch, bindings, id) else {
                return false;
            };
            vec![buffer]
        };
        for (lane, &buffer) in buffers.iter().enumerate() {
            if graph.independent_rows
                && buffers[..lane].iter().any(|other| {
                    let a = buffer_key(buffer);
                    let b = buffer_key(*other);
                    a.pointer == b.pointer
                        || match (a.address, b.address) {
                            (Some(a0), Some(b0)) => {
                                a0 < b0.saturating_add(b.bytes as u64)
                                    && b0 < a0.saturating_add(a.bytes as u64)
                            }
                            _ => false,
                        }
                })
            {
                return false;
            }
            if let Some(kv) = crate::as_segmented_kv(buffer) {
                let Some(committed) = kv.committed().checked_mul(kv.spec().segment_elements) else {
                    return false;
                };
                if !committed_depth_fits(
                    positions[lane],
                    ratio,
                    stride,
                    graph.desc(id).numel(),
                    committed,
                ) {
                    return false;
                }
            }
        }
        true
    };
    graph.ops.iter().all(|op| match op {
        Op::WriteKv {
            cache, row_stride, ..
        } => check(*cache, *row_stride as usize, 1),
        Op::Attention {
            k_cache,
            v_cache,
            n_kv,
            head_dim,
            ..
        }
        | Op::QsaBatchAttention {
            k_cache,
            v_cache,
            n_kv,
            head_dim,
            ..
        } => {
            let stride = *n_kv as usize * *head_dim as usize;
            check(*k_cache, stride, 1) && check(*v_cache, stride, 1)
        }
        Op::QsaPrepare {
            k_cache, head_dim, ..
        } => check(*k_cache, *head_dim as usize, 1),
        Op::QsaIndexer {
            k_cache,
            block_cache,
            head_dim,
            ratio,
            ..
        } => {
            check(*k_cache, *head_dim as usize, 1)
                && check(*block_cache, *head_dim as usize, *ratio as usize)
        }
        Op::Conv1dSilu {
            state, state_trace, ..
        }
        | Op::DeltaNet {
            state, state_trace, ..
        } if graph.independent_rows => {
            check(*state, 1, 1) && state_trace.is_none_or(|id| check(id, 1, 1))
        }
        _ => true,
    })
}

pub(super) fn next_token_ready(
    graph: &Graph,
    scratch: &ScratchSet,
    bindings: &Bindings,
    positions: &[usize],
) -> bool {
    next_positions_supported(graph, positions)
        && kv_depth_ready(graph, scratch, bindings, positions)
}

fn prepared_fusions(plan: &VkDecodePlan, fuse_add: bool) -> infr_core::fusion::FusionPlan {
    let mut fusions = plan.setup_analysis.fusions(fuse_add);
    if plan.graph.independent_rows {
        fusions.kv_write.clear();
        for index in &plan.setup_analysis.write_kv {
            fusions.skip.remove(index);
        }
    }
    fusions
}

#[cfg(test)]
fn predicted_position_matches(graph: &Graph, position: usize) -> bool {
    next_positions_supported(graph, &[position])
}

fn sequence_compatible(current: &Graph, next: &Graph) -> bool {
    if current.independent_rows != next.independent_rows
        || packet_rows(current).is_none()
        || packet_rows(current) != packet_rows(next)
    {
        return false;
    }
    if current.independent_rows {
        let Some(positions) = independent_positions(current) else {
            return false;
        };
        let Some(next_positions) = independent_positions(next) else {
            return false;
        };
        if !positions
            .iter()
            .zip(next_positions)
            .all(|(&position, next)| position.checked_add(1) == Some(next))
        {
            return false;
        }
        return prefix_end(current).is_some();
    }
    true
}

pub(super) fn compatible(current: &Graph, next: &Graph, scratch: &ScratchSet) -> bool {
    if !sequence_compatible(current, next) {
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

pub(super) fn cohort_bindings_match(
    current: &Graph,
    current_bindings: &Bindings,
    next: &Graph,
    next_bindings: &Bindings,
) -> bool {
    if !current.independent_rows && !next.independent_rows {
        return true;
    }
    let state_keys = |graph: &Graph, bindings: &Bindings| -> Option<Vec<Vec<BufferKey>>> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        prefix_end(graph)?;
        for op in &graph.ops {
            let ids = match op {
                Op::WriteKv { cache, .. } => vec![*cache],
                Op::Attention {
                    k_cache, v_cache, ..
                }
                | Op::QsaBatchAttention {
                    k_cache, v_cache, ..
                }
                | Op::QsaGather {
                    k_cache, v_cache, ..
                } => vec![*k_cache, *v_cache],
                Op::QsaPrepare { k_cache, .. } => vec![*k_cache],
                Op::QsaIndexer {
                    k_cache,
                    block_cache,
                    ..
                } => vec![*k_cache, *block_cache],
                Op::Conv1dSilu {
                    state, state_trace, ..
                }
                | Op::DeltaNet {
                    state, state_trace, ..
                } => std::iter::once(*state).chain(*state_trace).collect(),
                _ => vec![],
            };
            for id in ids {
                if !seen.insert(id) || !graph.inputs.contains(&id) {
                    continue;
                }
                let rows = bindings.get_rows(id)?;
                if rows.len() != packet_rows(graph)? {
                    return None;
                }
                result.push(rows.iter().map(|buffer| buffer_key(*buffer)).collect());
            }
        }
        Some(result)
    };
    match (
        state_keys(current, current_bindings),
        state_keys(next, next_bindings),
    ) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

pub(super) fn copy_binding_matches(
    selected: Option<&Arc<crate::pager::GpuCopyBuffers>>,
    recorded: Option<&Arc<crate::pager::GpuCopyBuffers>>,
) -> bool {
    match (selected, recorded) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

impl PreparedPrefix {
    #[cfg(test)]
    pub(super) fn coalesced_test_parts(&self) -> Vec<usize> {
        std::iter::once(self.head_layers).chain(self.parts.iter().map(|part| part.layers)).collect()
    }

    #[cfg(test)]
    pub(super) fn recorded_shape(&self) -> (usize, usize, usize, usize, bool, bool) {
        let resident = self.resident.as_ref();
        (
            self.head_end,
            self.head_layers,
            self.parts.last().map_or(self.head_end, |part| part.end),
            self.resident_layers(),
            resident.is_some_and(|value| value.chunk.self_service),
            resident.is_some_and(|value| {
                value
                    .chunk
                    .gpu_copy
                    .as_ref()
                    .is_some_and(|copy| copy.is_owned())
            }),
        )
    }

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
        copy_enabled: bool,
    ) -> Option<ActivatedPrefix> {
        let _span = infr_core::timeline::span("decode_packet_activate");
        let reject = |name: &'static str| {
            let _reason = infr_core::timeline::span(name);
            None
        };
        if !enabled(be_) {
            return reject("decode_packet_reject_disabled");
        }
        if !self.command.trace_compatible() {
            return reject("decode_packet_reject_trace_head");
        }
        if self
            .parts
            .iter()
            .any(|part| !part.command.trace_compatible())
        {
            return reject("decode_packet_reject_trace_part");
        }
        if !packet_geometry(
            (self.head_end, self.head_layers),
            self.end,
            self.resident
                .as_ref()
                .map_or(0, |resident| resident.chunk.entries.len()),
            self.parts.iter().map(|part| (part.end, part.layers)),
        ) {
            return reject("decode_packet_reject_geometry");
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
            return reject("decode_packet_reject_epoch");
        }
        if self
            .tensors
            .iter()
            .any(|(id, expected)| !keys_match(scratch, bindings, *id, expected))
        {
            return reject("decode_packet_reject_tensor_keys");
        }
        if !ack_positions_match(scratch, bindings, &self.positions) {
            return reject("decode_packet_reject_positions");
        }
        if self.pool.iter().any(|(key, handle, offset, bytes)| {
            pool.buffers
                .get(key)
                .and_then(|buffer| crate::as_vk_buf(buffer.as_ref()).ok())
                .is_none_or(|buffer| {
                    use ash::vk::Handle;
                    buffer.buffer.as_raw() != *handle
                        || buffer.sub_offset != *offset
                        || buffer.size != *bytes
                })
        }) {
            return reject("decode_packet_reject_pool");
        }
        // begin_execute cleared the previous step's pins. Keep referenced capacities alive
        // through this submission even when no further command is being prepared.
        pool.in_use.extend(self.pool.iter().map(|(key, ..)| *key));
        let mut tape_words = 0;
        let resident = if let Some(mut prepared) = self.resident {
            let Some(buffers) = retained else {
                return reject("decode_packet_reject_resident_missing");
            };
            if !Arc::ptr_eq(&buffers.control, &prepared.chunk.control)
                || !Arc::ptr_eq(&buffers.journal, &prepared.chunk.journal)
                || !match (&buffers.notices, &prepared.chunk.notices) {
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    (None, None) => true,
                    _ => false,
                }
                || !copy_binding_matches(
                    buffers.selected_gpu_copy(copy_enabled),
                    prepared.chunk.gpu_copy.as_ref(),
                )
                || (prepared.chunk.gpu_copy.is_some()
                    && !prepared.chunk.self_service
                    && buffers.producer.is_none())
            {
                return reject("decode_packet_reject_resident_bindings");
            }
            let Ok(mut guard) = be_.moe_pager().lock() else {
                return reject("decode_packet_reject_pager_lock");
            };
            let Some(session) = guard.as_mut() else {
                return reject("decode_packet_reject_pager_missing");
            };
            if !Arc::ptr_eq(&session.retained_tape(), &prepared.tape) {
                return reject("decode_packet_reject_tape");
            }
            if prepared.banks.iter().any(|&(role, address, bytes)| {
                session.arena_addr(role).ok() != Some(address)
                    || session.slot_bytes(role).ok() != Some(bytes)
            }) {
                return reject("decode_packet_reject_bank_addresses");
            }
            if prepared.copy_params.iter().any(|(roles, expected)| {
                prepared
                    .chunk
                    .gpu_copy
                    .as_ref()
                    .and_then(|copy| copy.role_params(session, roles).ok())
                    .as_ref()
                    != Some(expected)
            }) {
                return reject("decode_packet_reject_role_params");
            }
            if prepared.chunk.pure_mapped_schedule.as_ref()
                .is_some_and(|schedule| !schedule.resources_current(be_, &prepared.chunk, pool))
            {
                return reject("decode_packet_reject_coalescing_policy");
            }
            let mut publish = |session: &mut crate::pager::MoePagerSession| -> Result<()> {
                // Victims must be selected before these windows freeze the live LUT slots.
                for entry in &prepared.chunk.entries {
                    for &role in &entry.roles {
                        if prepared
                            .chunk
                            .gpu_copy
                            .as_ref()
                            .is_some_and(|copy| copy.is_owned())
                        {
                            session.resident_lut_coordinates(
                                &mut tape_words,
                                role,
                                entry.layout.n_expert,
                            )?;
                        } else {
                            session.resident_lut_window(
                                &mut tape_words,
                                role,
                                entry.layout.n_expert,
                            )?;
                        }
                    }
                }
                if tape_words != prepared.tape_words {
                    return Err(be("prepared resident tape layout changed"));
                }
                if !prepared.chunk.self_service {
                    if let Some(notices) = &prepared.chunk.notices {
                        notices.reset()?;
                        // Conservative dirty marking is at ACK, never speculative recording.
                        // Unsubmitted suffixes may cause an extra reset, never a stale event.
                        for (ordinal, entry) in prepared.chunk.entries.iter().enumerate() {
                            if entry.host_notice {
                                notices.dirty.mark(ordinal);
                            }
                        }
                    }
                }
                Ok(())
            };
            if let Some(copy) = &prepared.chunk.gpu_copy {
                if copy.activate_prepared(be_, session, publish).is_err() {
                    return reject("decode_packet_reject_owner_activation");
                }
            } else {
                if publish(session).is_err() {
                    return reject("decode_packet_reject_lut_publication");
                }
            }
            // Speculative packets must not own the live producer: ResidentChunk::drop drains
            // and resets it. Attach those runtime owners only after successful activation.
            prepared.chunk.producer = if prepared.chunk.self_service {
                None
            } else {
                buffers.producer.clone()
            };
            prepared.chunk.waiter = buffers.waiter.clone();
            Some(prepared.chunk)
        } else {
            None
        };
        let _accepted = infr_core::timeline::span("decode_packet_accepted");
        #[cfg(test)]
        TEST_ACCEPTED_PREFIXES.with(|count| count.set(count.get() + 1));
        Some(ActivatedPrefix {
            command: self.command,
            parts: self.parts,
            head_end: self.head_end,
            head_layers: self.head_layers,
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
    pool: ScratchPool,
    resident: Option<ResidentChunk>,
) -> Result<PreparedPrefix> {
    record_inner(be_, plan, scratch, bindings, positions, pool, resident, None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_cancellable(
    be_: &VulkanBackend,
    plan: &VkDecodePlan,
    scratch: &ScratchSet,
    bindings: &Bindings,
    positions: &[usize],
    pool: ScratchPool,
    resident: Option<ResidentChunk>,
    cancelled: &AtomicBool,
) -> Result<Option<PreparedPrefix>> {
    cancellable_recording(cancelled, || {
        record_inner(
            be_, plan, scratch, bindings, positions, pool, resident, Some(cancelled),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn record_inner(
    be_: &VulkanBackend,
    plan: &VkDecodePlan,
    scratch: &ScratchSet,
    bindings: &Bindings,
    positions: &[usize],
    mut pool: ScratchPool,
    resident: Option<ResidentChunk>,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedPrefix> {
    #[cfg(test)]
    super::promotion_isolation_note(be_, "record");
    let _span = infr_core::timeline::background_span("decode_packet_record");
    recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
    let owned = resident
        .as_ref()
        .is_some_and(|chunk| chunk.gpu_copy.as_ref().is_some_and(|copy| copy.is_owned()));
    let fallback = if owned {
        Some((
            snapshot_pool(&mut pool)?,
            resident
                .as_ref()
                .and_then(ResidentChunk::preparation_snapshot),
        ))
    } else {
        None
    };
    recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
    match record_with_horizon(
        be_, plan, scratch, bindings, positions, pool, resident, true, cancelled,
    ) {
        Ok(packet) => Ok(packet),
        Err(error) => {
            // Cancellation must not restart the entire preparation with a bounded horizon.
            recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
            match fallback {
                Some((pool, resident)) => {
                    tracing::debug!(%error, "long owned preparation declined; retrying bounded prefix");
                    record_with_horizon(
                        be_, plan, scratch, bindings, positions, pool, resident, false, cancelled,
                    )
                }
                None => Err(error),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_with_horizon(
    be_: &VulkanBackend,
    plan: &VkDecodePlan,
    scratch: &ScratchSet,
    bindings: &Bindings,
    positions: &[usize],
    mut pool: ScratchPool,
    mut resident: Option<ResidentChunk>,
    wide: bool,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedPrefix> {
    use ash::vk::Handle;
    recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
    let setup_span = infr_core::timeline::background_span("decode_packet_setup");
    let _span = infr_core::timeline::background_span(if wide {
        "decode_packet_wide_record"
    } else {
        "decode_packet_bounded_record"
    });
    let _guard = RecordingGuard::enter();
    let graph = &plan.graph;
    prefix_end(graph).ok_or_else(|| be("unsupported prepared graph"))?;
    let (cap, _) = crate::static_submit_mode(
        be_.shared.submit_dispatch_cap_explicit,
        be_.caps().integrated,
        true,
        be_.submit_dispatch_cap(),
    );
    if let Some(chunk) = resident.as_mut() {
        let cfg = &be_.cfg().kernels.vulkan;
        let copy_requested =
            resident_graph_layouts(be_, graph, be_.moe_paged()).is_some_and(|layouts| {
                resident_copy_requested(cfg.decode_gpu_resident_copy, graph, &layouts)
            });
        if !copy_requested {
            chunk.gpu_copy = None;
        }
        let submit_layers = match cfg.decode_gpu_resident_submit_layers {
            0 => graph.resident_submit_layers.unwrap_or(1),
            explicit => explicit,
        };
        chunk.submit_layers = submit_layers;
        chunk.mapped_serial_max_bytes =
            resident_mapped_serial_max_bytes(cfg.decode_gpu_resident_mapped_serial_max_kib, graph);
        if !resident_cadence_supported(
            submit_layers,
            chunk.limit,
            chunk.gpu_copy.as_ref().is_some_and(|copy| copy.is_owned()),
        ) {
            resident = None;
        }
    }
    // Submission cadence is NOT the preparation horizon. With fully mapped GPU ownership,
    // the entire forward can be encoded while the previous token executes. Activation/ACK
    // still gates submission. Slow/watchdog-split devices and CPU-serviced owners retain
    // their bounded packet, and all lowering still runs under the no-allocation/no-submit guard.
    let full_forward = wide
        && resident.as_ref().is_some_and(|chunk| {
            prepare_full_forward(
                chunk.self_service,
                chunk.gpu_copy.as_ref().is_some_and(|copy| copy.is_owned()),
                cap,
                graph
                    .ops
                    .iter()
                    .filter(|op| matches!(op, Op::MoeFfn { .. }))
                    .count(),
                chunk.limit,
            )
        });
    let owned_batch = wide
        && resident.as_ref().is_some_and(|chunk| {
            !chunk.self_service
                && cap == 0
                && chunk.gpu_copy.as_ref().is_some_and(|copy| copy.is_owned())
        });
    if let Some(chunk) = resident.as_mut() {
        chunk.install_pure_mapped_schedule(be_, plan, scratch, bindings, &pool, true, cap, false);
    }
    recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
    let end = if full_forward {
        graph.ops.len()
    } else if owned_batch {
        let chunk = resident.as_ref().unwrap();
        let preparation_limit = partial_preparation_limit(
            be_.cfg().kernels.vulkan.decode_gpu_resident_layers,
            chunk.limit,
        );
        let old_end = owned_prefix_end(graph, preparation_limit)
            .ok_or_else(|| be("unsupported owned packet window"))?;
        source_certificate_prefix_end(be_, graph, scratch, bindings, chunk, old_end)
            .unwrap_or(old_end)
    } else {
        match resident
            .as_ref()
            .and_then(|chunk| resident_end(graph, chunk.submit_layers))
        {
            Some(end) => end,
            None => {
                resident = None;
                prefix_end(graph).ok_or_else(|| be("unsupported prepared prefix"))?
            }
        }
    };
    if !next_positions_supported(graph, positions)
        || !kv_depth_ready(graph, scratch, bindings, positions)
    {
        return Err(be(
            "prepared prefix positions or committed lane KV depth changed",
        ));
    }
    let epoch = be_.shared.resource_epoch.load(Ordering::Acquire);
    let mut ids = HashSet::new();
    for op in &graph.ops[..end] {
        recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
        let (reads, writes) = op.io_inline();
        ids.extend(reads.into_iter().chain(writes));
    }
    if let Some(schedule) = resident.as_ref().and_then(|chunk| chunk.pure_mapped_schedule.as_ref()) {
        // Classification may look past the recording horizon solely to stop before a
        // future host entry. Its complete physical binding proof belongs to this ACK too.
        ids.extend(schedule.bank_tensors.iter().copied());
    }
    let metadata = plan.static_scratch_metadata()?;
    ids.extend(metadata.reset_indices.iter().map(|&i| TensorId(i as u32)));
    // Fused shared-expert weights/outputs can belong to ops just past the submission boundary.
    // Validate those bindings too, even though their standalone commands are suppressed.
    for (index, _) in plan.setup_analysis.shared_candidates(false, false) {
        recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
        if index < end {
            for op in graph.ops.iter().skip(index + 1).take(6) {
                let (reads, writes) = op.io_inline();
                ids.extend(reads.into_iter().chain(writes));
            }
        }
    }
    let tensors = ids
        .into_iter()
        .map(|id| {
            recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
            Ok((id, keys(scratch, bindings, id)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let pool_keys = pool
        .buffers
        .iter()
        .map(|(&key, buffer)| {
            recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
            let buffer = crate::as_vk_buf(buffer.as_ref())?;
            Ok((key, buffer.buffer.as_raw(), buffer.sub_offset, buffer.size))
        })
        .collect::<Result<Vec<_>>>()?;
    recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
    let rec = Recorder::new_prepared(be_)?;
    rec.seed_barrier();
    for &index in &metadata.reset_indices {
        recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
        let buffer = scratch[index]
            .as_deref()
            .ok_or_else(|| be("missing prepared scratch reset"))?;
        rec.zero(buffer, buffer.len_bytes().div_ceil(4));
    }
    let mut rope_positions = HashMap::new();
    for &(index, id) in &plan.setup_analysis.rope_ops {
        recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
        if index < end {
            rope_positions.insert(id.0, positions[0]);
        }
    }
    // QSA reads its position / optional IMROPE tables on the GPU. Their complete binding
    // identities are already in `tensors` via io_inline; do not replace those positions with
    // the predicted token index (multimodal RoPE positions need not equal cache offsets).
    let analysis = &plan.setup_analysis;
    let fusions = prepared_fusions(plan, be_.cfg().kernels.vulkan.fuse_add);
    let mut skip = fusions.skip;
    let mut shared_ops = HashMap::new();
    if resident.is_some() {
        let disabled = std::env::var_os("INFR_NO_MOE_SHARED_SLOT").is_some();
        for (index, shared) in analysis.shared_candidates(disabled, false) {
            recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
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
        recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
        skip.extend(
            analysis
                .hc
                .iter()
                .filter(|(i, _)| *i < end)
                .filter_map(|&(i, pair)| {
                    hc_up_mix_candidate(be_, pair, scratch, bindings).map(|_| i + 1)
                }),
        );
    }
    if be_.cfg().kernels.vulkan.ple_gate_norm_lds {
        recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
        skip.extend(
            analysis
                .ple
                .iter()
                .filter(|(i, _)| *i < end)
                .filter_map(|&(i, pair)| {
                    ple_gate_norm_candidate(be_, graph, i, pair, scratch, bindings).map(|_| i + 1)
                }),
        );
    }
    if deltanet_norm_enabled(be_) {
        skip.extend(analysis.delta.iter().filter(|&&i| i < end).map(|&i| i + 1));
    }
    recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
    let mode = RopeMode::Static(&rope_positions);
    let segmented = analysis.segmented_fusions(be_, graph, scratch, bindings, &mode, &skip)?;
    recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
    skip.extend(segmented.keys().map(|&i| i + 1));
    let empty = HashMap::new();
    let mut transient = Vec::new();
    let mut dynamic = Vec::new();
    let mut memo = None;
    let mut stream = PagedStream::default();
    if let Some(chunk) = resident.as_mut() {
        if let Some(copy) = &chunk.gpu_copy {
            // Owned overlays are the authoritative block directory seeded at activation.
            // Reset only the legacy per-window overlay; its companion LRU is CPU-owned.
            if !copy.is_owned() {
                rec.zero(copy.overlay.as_ref(), copy.overlay.len_bytes() / 4);
            }
        }
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
    let mut parts = Vec::new();
    drop(setup_span);
    for (index, op) in infr_core::exec::live_ops(&graph.ops[..end], &skip) {
        recording_checkpoint(cancelled, RecordingCheckpoint::Op(index))?;
        let _op_span = infr_core::timeline::background_span(op.kind());
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
            chunk.push_entry(
                be_,
                ResidentJournalEntry {
                    op_idx: index,
                    layout,
                    roles,
                    host_notice: true,
                },
            )?;
            let ordinal = chunk.entries.len() - 1;
            stream.gpu_route = Some(GpuRoute {
                control: Arc::clone(&chunk.control),
                journal: Arc::clone(&chunk.journal),
                ordinal: ordinal as u32,
                copy: chunk.gpu_copy.clone(),
                notice: chunk.notice_event(ordinal),
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
            recording_checkpoint(cancelled, RecordingCheckpoint::Setup)?;
            let chunk = resident.as_ref().unwrap();
            let submitted = parts.last().map_or(0, |part: &PreparedPart| part.layers);
            if owned_batch {
                chunk.preserve_coalesced_boundary(recorder.as_ref().unwrap(), submitted, true, true);
            }
            if owned_batch
                && chunk.pure_group_due(parts.last().map_or(0, |part: &PreparedPart| part.layers), true, true)
                && index + 1 < end
            {
                parts.push(PreparedPart {
                    command: recorder.take().unwrap().finish_prepared()?,
                    end: index + 1,
                    layers: chunk.entries.len(),
                });
                let next = Recorder::new_prepared(be_)?;
                next.seed_barrier();
                chunk.begin_conditional(&next);
                recorder = Some(next);
            }
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
    recording_checkpoint(cancelled, RecordingCheckpoint::Finalize)?;
    if !transient.is_empty() || !dynamic.is_empty() {
        return Err(be("prepared prefix produced transient GPU resources"));
    }
    let resident = resident
        .map(|chunk| -> Result<_> {
            recording_checkpoint(cancelled, RecordingCheckpoint::Finalize)?;
            let guard = be_.moe_pager().lock().unwrap();
            let session = guard.as_ref().ok_or_else(|| be("missing prepared pager"))?;
            let mut banks = Vec::new();
            let mut copy_params = Vec::new();
            for entry in &chunk.entries {
                recording_checkpoint(cancelled, RecordingCheckpoint::Finalize)?;
                for &role in &entry.roles {
                    banks.push((role, session.arena_addr(role)?, session.slot_bytes(role)?));
                }
                if let Some(copy) = &chunk.gpu_copy {
                    copy_params.push((entry.roles, copy.role_params(session, &entry.roles)?));
                }
            }
            Ok(PreparedResident {
                chunk,
                tape: session.retained_tape(),
                banks,
                tape_words: stream.tape_cursor,
                copy_params,
            })
        })
        .transpose()?;
    recording_checkpoint(cancelled, RecordingCheckpoint::Finalize)?;
    if full_forward {
        let last = recorder.as_ref().unwrap();
        last.end_conditional();
        last.publish_host_reads();
    }
    parts.push(PreparedPart {
        command: recorder.take().unwrap().finish_prepared()?,
        end,
        layers: resident
            .as_ref()
            .map_or(0, |resident| resident.chunk.entries.len()),
    });
    recording_checkpoint(cancelled, RecordingCheckpoint::Finalize)?;
    let head = parts.remove(0);
    let positions = rope_positions
        .into_iter()
        .map(|(id, position)| {
            (
                TensorId(id),
                if graph.independent_rows {
                    positions.to_vec()
                } else {
                    vec![position]
                },
            )
        })
        .collect();
    Ok(PreparedPrefix {
        command: head.command,
        head_end: head.end,
        head_layers: head.layers,
        parts,
        epoch,
        tensors,
        pool: pool_keys,
        positions,
        end,
        resident,
    })
}

fn prepare_full_forward(
    self_service: bool,
    owned: bool,
    cap: usize,
    layers: usize,
    capacity: usize,
) -> bool {
    self_service && owned && cap == 0 && layers > 0 && layers <= capacity
}

pub(super) fn partial_preparation_limit(configured_layers: usize, execution_limit: usize) -> usize {
    execution_limit.min(configured_layers.clamp(1, crate::recorder::RESIDENT_MAX_LAYERS))
}

fn source_certificate_group_limit(certificates: &[Option<bool>], cadence: usize) -> Option<usize> {
    if certificates.is_empty() || cadence == 0 {
        return None;
    }
    // Validate the whole captured range: unknown geometry/source proof keeps the old horizon.
    let mut first_host = None;
    for (index, certificate) in certificates.iter().enumerate() {
        if !(*certificate)? && first_host.is_none() {
            first_host = Some(index + 1);
        }
    }
    let Some(first_host) = first_host else {
        return Some(certificates.len());
    };
    // Pure groups stop BEFORE a host-sensitive entry. A group containing that entry uses
    // the original cadence grid, even when the preceding pure group ended off-grid.
    Some(
        first_host
            .div_ceil(cadence)
            .checked_mul(cadence)?
            .min(certificates.len()),
    )
}

fn source_certificate_prefix_end(
    be_: &VulkanBackend,
    graph: &Graph,
    scratch: &ScratchSet,
    bindings: &Bindings,
    chunk: &ResidentChunk,
    old_end: usize,
) -> Option<usize> {
    #[cfg(test)]
    if source_horizon_overrides()
        .lock()
        .unwrap()
        .get(&(Arc::as_ptr(&be_.shared) as usize))
        .copied()
        == Some(false)
    {
        return None;
    }
    let owner = chunk.gpu_copy.as_ref().filter(|copy| copy.is_owned())?;
    let pager = be_.moe_pager().lock().ok()?;
    let session = pager.as_ref()?;
    if !session
        .gpu_decode_owner
        .as_ref()
        .is_some_and(|live| Arc::ptr_eq(live, owner))
    {
        return None;
    }
    let mut banks = HashMap::new();
    let mut certificates = Vec::new();
    for op in &graph.ops[..old_end] {
        let Op::MoeFfn {
            gate_exps,
            up_exps,
            down_exps,
            ..
        } = op
        else {
            continue;
        };
        let layout = ResidentMoeLayout::from_op(graph, op)?;
        let roles = [*gate_exps, *up_exps, *down_exps].map(|id| {
            resolve(scratch, bindings, id)
                .ok()
                .map(crate::pager::buffer_identity)
        });
        let roles = [roles[0]?, roles[1]?, roles[2]?];
        let key = (roles, layout.n_expert);
        let mapped = if let Some(&mapped) = banks.get(&key) {
            mapped
        } else {
            let params = owner.role_params(session, &roles).ok()?;
            let bases = [params[0], params[1], params[2]];
            owner.owned_role_geometry(bases, layout.n_expert as u32)?;
            let mapped = owner.owned_roles_self_service(bases, layout.n_expert as u32);
            banks.insert(key, mapped);
            mapped
        };
        certificates.push(Some(mapped));
    }
    let layers =
        source_certificate_group_limit(&certificates, chunk.submit_layers.min(chunk.limit))?;
    owned_prefix_end(graph, layers)
}

pub(super) fn resident_cadence_supported(
    submit_layers: usize,
    capacity: usize,
    owned: bool,
) -> bool {
    submit_layers > 0 && submit_layers <= capacity && (owned || submit_layers < capacity)
}

fn owned_prefix_end(graph: &Graph, capacity: usize) -> Option<usize> {
    prefix_end(graph)?;
    let mut count = 0;
    let mut end = None;
    for (index, op) in graph.ops.iter().enumerate() {
        if matches!(op, Op::MoeFfn { .. }) {
            ResidentMoeLayout::from_op(graph, op)?;
            count += 1;
            end = Some(index + 1);
            if count == capacity {
                break;
            }
        }
    }
    end
}

fn packet_geometry(
    mut last: (usize, usize),
    end: usize,
    layers: usize,
    parts: impl IntoIterator<Item = (usize, usize)>,
) -> bool {
    if last.0 == 0 || last.0 > end || last.1 > layers {
        return false;
    }
    for part in parts {
        if part.0 <= last.0 || part.0 > end || part.1 <= last.1 || part.1 > layers {
            return false;
        }
        last = part;
    }
    last == (end, layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::{DType, TensorDesc};
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn cooperative_preparation_scope_end_is_test_only_and_preserves_ready_packets() {
        for failed in [false, true] {
            for experiment in [false, true] {
                let next = AtomicBool::new(false);
                let continuation = AtomicBool::new(false);
                let lease = Arc::new(());
                let ready = Mutex::new(Some(Arc::clone(&lease)));
                test_finish_preparation_scope(&next, &continuation, failed, experiment);
                assert_eq!(next.load(Ordering::Acquire), failed || experiment);
                assert_eq!(continuation.load(Ordering::Acquire), failed || experiment);
                assert!(Arc::ptr_eq(ready.lock().unwrap().as_ref().unwrap(), &lease));
                assert_eq!(Arc::strong_count(&lease), 2, "ready packet lease survives");
            }
        }
        let next = AtomicBool::new(false);
        let continuation = AtomicBool::new(true);
        finish_preparation_scope(&next, &continuation, false);
        assert_eq!(
            cancellable_recording(&next, || Ok(37usize)).unwrap(),
            Some(37)
        );
        assert!(cancellable_recording(&continuation, || -> Result<usize> {
            panic!("an obsolete continuation must not begin recording")
        })
        .unwrap()
        .is_none());
    }

    #[test]
    fn cooperative_preparation_checkpoints_decline_without_hiding_other_errors() {
        for point in [
            RecordingCheckpoint::Setup,
            RecordingCheckpoint::Op(1),
            RecordingCheckpoint::Finalize,
        ] {
            let cancelled = AtomicBool::new(false);
            let _at = TestCancelAt::enter(point);
            let ops = AtomicUsize::new(0);
            let result = cancellable_recording(&cancelled, || {
                recording_checkpoint(Some(&cancelled), RecordingCheckpoint::Setup)?;
                for op in 0..4 {
                    recording_checkpoint(Some(&cancelled), RecordingCheckpoint::Op(op))?;
                    ops.fetch_add(1, Ordering::Relaxed);
                }
                recording_checkpoint(Some(&cancelled), RecordingCheckpoint::Finalize)?;
                Ok(17usize)
            })
            .unwrap();
            assert!(result.is_none());
            assert!(cancelled.load(Ordering::Acquire));
            let expected = match point {
                RecordingCheckpoint::Setup => 0,
                RecordingCheckpoint::Op(_) => 1,
                RecordingCheckpoint::Finalize => 4,
            };
            assert_eq!(ops.load(Ordering::Relaxed), expected);
            // The noncancellable/direct-record path never observes this scoped hook.
            recording_checkpoint(None, point).unwrap();
        }
        let cancelled = AtomicBool::new(false);
        assert!(matches!(
            cancellable_recording(&cancelled, || -> Result<()> { Err(Error::Aborted) }),
            Err(Error::Aborted)
        ));
        assert!(cancellable_recording(&cancelled, || -> Result<()> {
            Err(be("ordinary unsupported preparation"))
        })
        .unwrap_err()
        .to_string()
        .contains("ordinary unsupported preparation"));
    }

    #[test]
    fn cooperative_preparation_scope_joins_cancelled_private_work_and_leases() {
        use std::sync::mpsc;
        use std::time::Duration;
        const WATCHDOG: Duration = Duration::from_secs(10);
        let next = AtomicBool::new(false);
        let continuation = AtomicBool::new(false);
        let ops = AtomicUsize::new(0);
        let dropped = Arc::new(AtomicBool::new(false));
        struct Lease(Arc<AtomicBool>);
        impl Drop for Lease {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let lease = Lease(Arc::clone(&dropped));
        let outcome = Mutex::new(None);
        let (entered, started) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let entered_ok = recording_scope(true, |scope| {
            let cancelled = &next;
            let ops = &ops;
            let outcome = &outcome;
            scope.spawn(move |_| {
                let _lease = lease;
                let result = cancellable_recording(cancelled, || {
                    recording_checkpoint(Some(cancelled), RecordingCheckpoint::Setup)?;
                    ops.fetch_add(1, Ordering::Relaxed);
                    let _ = entered.send(());
                    gate.recv_timeout(WATCHDOG)
                        .map_err(|_| be("checkpoint test release timed out"))?;
                    recording_checkpoint(Some(cancelled), RecordingCheckpoint::Op(1))?;
                    ops.fetch_add(1, Ordering::Relaxed);
                    Ok(41usize)
                });
                *outcome.lock().unwrap() = Some(result);
            });
            let entered_ok = started.recv_timeout(WATCHDOG).is_ok();
            test_finish_preparation_scope(&next, &continuation, false, true);
            let still_leased = !dropped.load(Ordering::Acquire);
            let _ = release.send(());
            entered_ok && still_leased
        })
        .unwrap();
        assert!(
            entered_ok,
            "worker must enter before the simulated body completion"
        );
        assert_eq!(ops.load(Ordering::Relaxed), 1);
        assert!(outcome.lock().unwrap().take().unwrap().unwrap().is_none());
        assert!(
            dropped.load(Ordering::Acquire),
            "private work retires before scope return"
        );
    }

    #[test]
    fn next_stage_recording_horizon_bounds_only_continuations() {
        assert_eq!(RecordingHorizon::for_next_stage(0), RecordingHorizon::Wide);
        for stage in [1, 2, 7, usize::MAX] {
            assert_eq!(
                RecordingHorizon::for_next_stage(stage),
                RecordingHorizon::Bounded
            );
        }
    }

    #[test]
    fn source_certificate_horizon_stops_at_existing_host_group() {
        for cadence in [1, 2, 4, 16] {
            assert_eq!(
                source_certificate_group_limit(&[Some(true); 16], cadence),
                Some(16)
            );
            for host in 0..16 {
                let mut certificates = [Some(true); 16];
                certificates[host] = Some(false);
                let layers = source_certificate_group_limit(&certificates, cadence).unwrap();
                assert_eq!(layers, (host + 1).div_ceil(cadence) * cadence);
                assert!(layers > host);
                assert!(layers == 16 || resident_submit_due(layers, 0, cadence, 128));
                // The execution window is NOT the speculative preparation limit.
                assert_eq!(resident_execution_limit(true, 48, 16, 128).unwrap(), 48);
            }
        }
        assert_eq!(
            source_certificate_group_limit(&[Some(false); 16], 1),
            Some(1)
        );
        assert_eq!(
            source_certificate_group_limit(&[Some(false); 16], 16),
            Some(16)
        );
        assert_eq!(
            source_certificate_group_limit(&[Some(true), Some(false), Some(true)], 2),
            Some(2)
        );
    }

    #[test]
    fn source_certificate_horizon_unknown_proof_keeps_old_range() {
        for unknown in 0..16 {
            let mut certificates = [Some(true); 16];
            certificates[0] = Some(false);
            certificates[unknown] = None;
            assert_eq!(source_certificate_group_limit(&certificates, 1), None);
        }
        assert_eq!(source_certificate_group_limit(&[], 1), None);
        assert_eq!(source_certificate_group_limit(&[Some(false)], 0), None);
    }

    #[test]
    fn recording_scope_cannot_be_stolen_by_nested_global_copy_work() {
        let pager = Mutex::new(());
        let locked = pager.lock().unwrap();
        let recorded = AtomicBool::new(false);
        let copied = AtomicUsize::new(0);
        let caller = std::thread::current().id();
        recording_pool().unwrap().in_place_scope(|scope| {
            assert_eq!(std::thread::current().id(), caller);
            scope.spawn(|_| {
                let _pager = pager.lock().unwrap();
                recorded.store(true, Ordering::Release);
            });
            // Real mapped copies can join global Rayon while retaining this lock. The
            // recording worker must remain elsewhere until the copy releases the lock.
            rayon::scope(|copy_scope| {
                for _ in 0..8 {
                    copy_scope.spawn(|_| {
                        copied.fetch_add(1, Ordering::Relaxed);
                    });
                }
            });
            assert_eq!(copied.load(Ordering::Relaxed), 8);
            drop(locked);
        });
        assert!(recorded.load(Ordering::Acquire));
    }

    #[test]
    fn one_layer_owned_stage_keeps_its_resident_packet() {
        assert!(resident_cadence_supported(1, 1, true));
        assert!(resident_cadence_supported(2, 2, true));
        assert!(resident_cadence_supported(1, 2, true));
        assert!(resident_cadence_supported(1, 2, false));
        for owned in [false, true] {
            assert!(!resident_cadence_supported(0, 1, owned));
            assert!(!resident_cadence_supported(1, 0, owned));
            assert!(!resident_cadence_supported(2, 1, owned));
        }
        assert!(!resident_cadence_supported(1, 1, false));
        assert!(!resident_cadence_supported(2, 2, false));
        assert!(prepare_full_forward(true, true, 0, 1, 1));
    }

    fn token_qsa_ops(graph: &mut Graph) -> (Vec<Op>, TensorId, TensorId) {
        let qk = graph.input(TensorDesc::new([1, 256], DType::F32));
        let norm = graph.weight(TensorDesc::new([128], DType::F32));
        let positions = graph.input(TensorDesc::new([1], DType::I32));
        let positions4 = graph.input(TensorDesc::new([64, 4], DType::I32));
        let raw = graph.input(TensorDesc::new([64, 128], DType::F16));
        let blocks = graph.input(TensorDesc::new([16, 128], DType::F32));
        let q = graph.output(TensorDesc::new([1, 128], DType::F16));
        let indices = graph.output(TensorDesc::new([1, 4], DType::I32));
        let k = graph.input(TensorDesc::new([64, 128], DType::F16));
        let v = graph.input(TensorDesc::new([64, 128], DType::F16));
        let gathered_k = graph.output(TensorDesc::new([16, 128], DType::F16));
        let gathered_v = graph.output(TensorDesc::new([16, 128], DType::F16));
        let output = graph.output(TensorDesc::new([1, 128], DType::F32));
        (
            vec![
                Op::QsaPrepare {
                    qk,
                    q_norm: norm,
                    positions,
                    positions4: Some(positions4),
                    k_cache: raw,
                    q_dst: q,
                    rows: 1,
                    n_head: 1,
                    head_dim: 128,
                    rope_dim: 64,
                    theta: 10000.0,
                    eps: 1e-6,
                    sections: [16; 4],
                    pos: 63,
                },
                Op::QsaIndexer {
                    q,
                    k_cache: raw,
                    block_cache: blocks,
                    k_norm: norm,
                    positions4: Some(positions4),
                    dst: indices,
                    rows: 1,
                    kv_len: 64,
                    compress_from: 15,
                    n_head: 1,
                    head_dim: 128,
                    top_blocks: 4,
                    ratio: 4,
                    rope_dim: 64,
                    theta: 10000.0,
                    eps: 1e-6,
                    scale: 1.0,
                    sections: [16; 4],
                },
                Op::QsaGather {
                    k_cache: k,
                    v_cache: v,
                    indices,
                    k_dst: gathered_k,
                    v_dst: gathered_v,
                    selected_blocks: 4,
                    complete_blocks: 16,
                    tail: 0,
                    ratio: 4,
                    row_elems: 128,
                },
                Op::QsaBatchAttention {
                    q,
                    k_cache: k,
                    v_cache: v,
                    indices,
                    dst: output,
                    rows: 1,
                    kv_len: 64,
                    n_head: 1,
                    n_kv: 1,
                    head_dim: 128,
                    top_blocks: 4,
                    ratio: 4,
                    scale: 1.0,
                },
            ],
            positions,
            positions4,
        )
    }

    #[test]
    fn token_qsa_prefix_and_resident_window_keep_every_position_binding() {
        let mut graph = decode_graph();
        let (ops, positions, positions4) = token_qsa_ops(&mut graph);
        for op in &ops {
            assert!(supported_op(op));
        }
        let (prepare_reads, _) = ops[0].io_inline();
        assert!(prepare_reads.contains(&positions));
        assert!(prepare_reads.contains(&positions4));
        let (indexer_reads, _) = ops[1].io_inline();
        assert!(indexer_reads.contains(&positions4));
        let moe = graph.ops[2].clone();
        graph.ops.extend(ops.clone());
        graph.ops.push(moe);
        assert_eq!(resident_end(&graph, 2), Some(8));
        assert_eq!(owned_prefix_end(&graph, 2), Some(8));
        graph.ops.splice(0..0, ops);
        assert_eq!(prefix_end(&graph), Some(6));
    }

    #[test]
    fn qsa_preparation_does_not_expand_to_prefill_or_verify_rows() {
        let mut graph = decode_graph();
        let (ops, _, _) = token_qsa_ops(&mut graph);
        for mut op in ops {
            match &mut op {
                Op::QsaPrepare { rows, .. }
                | Op::QsaIndexer { rows, .. }
                | Op::QsaBatchAttention { rows, .. } => *rows = 2,
                Op::QsaGather { .. } => continue,
                _ => unreachable!(),
            }
            assert!(!supported_op(&op));
            let mut batched = graph.clone();
            batched.ops.insert(0, op);
            assert!(prefix_end(&batched).is_none());
        }
        for verify in [false, true] {
            let mut marked = graph.clone();
            if verify {
                marked.mtp_verify = true;
            } else {
                marked.prefill_target_tokens = Some(1);
            }
            assert!(prefix_end(&marked).is_none());
        }
    }

    #[test]
    fn long_preparation_does_not_change_cpu_service_or_watchdog_submit_policy() {
        assert!(prepare_full_forward(true, true, 0, 48, 128));
        for (service, owned, cap, layers, capacity) in [
            (false, true, 0, 48, 128),
            (true, false, 0, 48, 128),
            (true, true, 256, 48, 128),
            (true, true, 0, 129, 128),
            (true, true, 0, 0, 128),
        ] {
            assert!(!prepare_full_forward(service, owned, cap, layers, capacity));
        }
        let mut graph = decode_graph();
        graph.ops.extend(graph.ops.clone());
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
        // Owned packet recording delegates to guarded lowering instead of expanding the
        // old prefix Op whitelist. Legacy retains its original conservative rejection.
        assert_eq!(resident_end(&graph, 2), None);
        assert_eq!(owned_prefix_end(&graph, 2), Some(7));
        assert_eq!(owned_prefix_end(&graph, 1), Some(3));
    }

    #[test]
    fn prepared_batch_geometry_covers_exact_submitted_prefixes() {
        assert!(packet_geometry((12, 2), 48, 8, [(24, 4), (36, 6), (48, 8)]));
        assert!(packet_geometry((48, 8), 48, 8, []));
        assert!(packet_geometry((2, 0), 2, 0, []));
        for parts in [
            vec![(12, 4), (48, 8)],
            vec![(24, 2), (48, 8)],
            vec![(24, 4), (49, 8)],
            vec![(24, 4), (48, 9)],
            vec![(24, 4)],
        ] {
            assert!(!packet_geometry((12, 2), 48, 8, parts));
        }
        assert!(!packet_geometry((0, 0), 0, 0, []));
    }

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
        let expected = [
            (TensorId(1), vec![32768, 37]),
            (TensorId(2), vec![32768, 37]),
        ];
        assert!(positions_match(&expected, |_, rows| (rows == 2).then_some(vec![32768, 37])));
        assert!(!positions_match(&expected, |_, _| Some(vec![32767, 37])));
        assert!(!positions_match(&expected, |_, _| Some(vec![32768, 38])));
        assert!(!positions_match(&expected, |_, _| Some(vec![37, 32768])));
        assert!(!positions_match(&expected, |id, _| (id.0 == 1).then_some(vec![32768, 37])));
        assert!(!positions_match(&expected, |id, _| Some(if id.0 == 1 {
            vec![32768, 37]
        } else {
            vec![32769, 37]
        })));
    }

    #[test]
    fn nonpositional_prefix_needs_no_position_readback() {
        assert!(positions_match(&[], |_, _| panic!(
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

    fn resident_preparation_fixture() -> (ResidentChunk, ResidentBuffers) {
        let control: Arc<dyn Buffer> = Arc::new(CpuBuffer::new(20, 0));
        let journal: Arc<dyn Buffer> = Arc::new(CpuBuffer::new(
            2 * crate::recorder::RESIDENT_JOURNAL_BYTES,
            0,
        ));
        let chunk = ResidentChunk {
            control: Arc::clone(&control),
            journal: Arc::clone(&journal),
            entries: Vec::new(),
            early_recovery: false,
            recovery_op: None,
            limit: 2,
            recording: false,
            self_service: false,
            submit_layers: 1,
            pure_mapped_schedule: None,
            submitted_entries: 0,
            notices: None,
            notice_cursor: 0,
            notice_resume: None,
            notice_stop: false,
            producer: None,
            notices_sent: 0,
            owned_notice_acks: Vec::new(),
            mapped_serial_max_bytes: 0,
            waiter: None,
            indirect: None,
            gpu_copy: None,
        };
        let retained = ResidentBuffers {
            control,
            journal,
            capacity: 2,
            notices: None,
            producer: None,
            waiter: None,
            indirect: None,
            gpu_copy: None,
            gpu_copy_init_attempted: false,
        };
        (chunk, retained)
    }

    #[test]
    fn resident_preparation_capture_retains_identity_and_drops_only_its_handles() {
        let (template, retained) = resident_preparation_fixture();
        let references = Arc::strong_count(&retained.control);
        let snapshot = ResidentPreparationSnapshot::capture(&template).unwrap();
        assert!(resident_preparation_matches(
            &snapshot.chunk,
            &retained,
            false,
            2,
            1,
            0,
        ));
        assert_eq!(Arc::strong_count(&retained.control), references + 1);
        assert!(snapshot.chunk.producer.is_none());
        assert!(snapshot.chunk.waiter.is_none());
        assert_eq!(snapshot.chunk.limit, template.limit);
        drop(snapshot);
        assert_eq!(Arc::strong_count(&retained.control), references);
    }

    #[test]
    fn resident_preparation_capture_rejects_live_protocol_coordinates() {
        for mutation in 0..9 {
            let (mut template, _) = resident_preparation_fixture();
            match mutation {
                0 => template.recording = true,
                1 => template.submitted_entries = 1,
                2 => template.notice_cursor = 1,
                3 => template.notice_resume = Some((0, vec![1])),
                4 => template.notice_stop = true,
                5 => template.notices_sent = 1,
                6 => {
                    template.recovery_op = Some(ResidentRecovery {
                        op_idx: 0,
                        miss_tape_start: 0,
                        minimum_tape_words: 1,
                    });
                }
                7 => {
                    let (_, receiver) = std::sync::mpsc::channel();
                    template.owned_notice_acks.push(receiver);
                }
                _ => {
                    let graph = decode_graph();
                    template.entries.push(ResidentJournalEntry {
                        op_idx: 2,
                        layout: ResidentMoeLayout::from_op(&graph, &graph.ops[2]).unwrap(),
                        host_notice: true,
                        roles: [0; 3],
                    });
                }
            }
            assert!(ResidentPreparationSnapshot::capture(&template).is_none());
        }
    }

    #[test]
    fn resident_preparation_validation_rejects_replaced_storage_and_window_policy() {
        let (template, mut retained) = resident_preparation_fixture();
        let snapshot = ResidentPreparationSnapshot::capture(&template).unwrap();
        let matches = |retained: &ResidentBuffers, limit, cadence| {
            resident_preparation_matches(&snapshot.chunk, retained, false, limit, cadence, 0)
        };
        assert!(matches(&retained, 2, 1));
        assert!(!matches(&retained, 0, 1));
        assert!(!matches(&retained, 1, 1));
        assert!(!matches(&retained, 2, 0));
        assert!(!matches(&retained, 2, 2));
        retained.capacity = 1;
        assert!(!matches(&retained, 2, 1));
        retained.capacity = 2;
        retained.control = Arc::new(CpuBuffer::new(template.control.len_bytes(), 0));
        assert!(!matches(&retained, 2, 1));
        retained.control = Arc::clone(&template.control);
        retained.journal = Arc::new(CpuBuffer::new(template.journal.len_bytes(), 0));
        assert!(!matches(&retained, 2, 1));
    }

    #[test]
    fn resident_preparation_capture_and_validation_require_supported_cadence() {
        let (mut template, retained) = resident_preparation_fixture();
        template.submit_layers = 0;
        assert!(ResidentPreparationSnapshot::capture(&template).is_none());
        template.submit_layers = template.limit;
        assert!(ResidentPreparationSnapshot::capture(&template).is_none());
        template.submit_layers = 1;
        let mut snapshot = ResidentPreparationSnapshot::capture(&template).unwrap();
        snapshot.chunk.self_service = true;
        assert!(!resident_preparation_matches(
            &snapshot.chunk,
            &retained,
            false,
            2,
            1,
            0,
        ));
        snapshot.chunk.self_service = false;
        snapshot.chunk.recording = true;
        assert!(!resident_preparation_matches(
            &snapshot.chunk,
            &retained,
            false,
            2,
            1,
            0,
        ));
    }

    #[test]
    fn resident_preparation_policy_boundary_rejects_stale_current_graph_template() {
        let (mut template, retained) = resident_preparation_fixture();
        let mut graph = decode_graph();
        graph.independent_rows = true;
        graph.sequence_spans = vec![SequenceSpan {
            row_start: 0,
            rows: 1,
            start_pos: (RESIDENT_AUTO_MAPPED_SERIAL_CONTEXT_MAX - 1) as u32,
        }];
        template.mapped_serial_max_bytes = resident_mapped_serial_max_bytes(None, &graph);
        let snapshot = ResidentPreparationSnapshot::capture(&template).unwrap();
        let matches =
            |limit| resident_preparation_matches(&snapshot.chunk, &retained, false, 2, 1, limit);
        assert_eq!(
            template.mapped_serial_max_bytes,
            RESIDENT_AUTO_MAPPED_SERIAL_MAX_BYTES
        );
        assert!(matches(resident_mapped_serial_max_bytes(None, &graph)));
        graph.sequence_spans[0].start_pos += 1;
        assert!(!matches(resident_mapped_serial_max_bytes(None, &graph)));
        // Explicit policy remains explicit; crossing a context boundary must not alter it.
        assert!(matches(resident_mapped_serial_max_bytes(
            Some(6144),
            &graph
        )));
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
        let result: Result<()> = {
            let _guard = RecordingGuard::enter();
            assert!(preparing_commands());
            Err(be("CPU preparation failure"))
        };
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

    fn independent_graph(position: u32) -> Graph {
        let mut graph = decode_graph();
        graph.independent_rows = true;
        graph.sequence_spans = vec![SequenceSpan {
            row_start: 0,
            rows: 1,
            start_pos: position,
        }];
        graph
    }

    fn independent_cohort(positions: &[u32]) -> Graph {
        let mut graph = decode_graph();
        let rows = positions.len();
        for tensor in &mut graph.tensors {
            if tensor.desc.shape.as_slice() == [1, 8] {
                tensor.desc = TensorDesc::new([rows, 8], tensor.desc.dtype);
            }
        }
        if let Op::RmsNorm { rows: n, .. } = &mut graph.ops[0] {
            *n = rows as u32;
        }
        if let Op::Linear { m, .. } = &mut graph.ops[1] {
            *m = rows as u32;
        }
        graph.independent_rows = true;
        graph.sequence_spans = positions
            .iter()
            .enumerate()
            .map(|(lane, &start_pos)| SequenceSpan {
                row_start: lane as u32,
                rows: 1,
                start_pos,
            })
            .collect();
        graph
    }

    #[test]
    fn prepared_cohort_advances_every_ordered_lane_without_causal_batch_aliasing() {
        for rows in 1..=infr_core::backend::MAX_DECODE_ROWS {
            let positions = (0..rows)
                .map(|lane| 7 + lane as u32 * 13)
                .collect::<Vec<_>>();
            let current = independent_cohort(&positions);
            let next_positions = positions.iter().map(|&p| p + 1).collect::<Vec<_>>();
            let next = independent_cohort(&next_positions);
            let scratch = cpu_scratch(&next);
            assert_eq!(prefix_end(&current), Some(2));
            assert!(compatible(&current, &next, &scratch));
            assert!(next_positions_supported(
                &next,
                &next_positions
                    .iter()
                    .map(|&p| p as usize)
                    .collect::<Vec<_>>()
            ));
            for lane in 0..rows {
                let mut bad = next.clone();
                bad.sequence_spans[lane].start_pos += 1;
                assert!(!compatible(&current, &bad, &scratch));
            }
            if rows > 1 {
                let mut reordered = next.clone();
                reordered.sequence_spans[0].start_pos = next_positions[1];
                reordered.sequence_spans[1].start_pos = next_positions[0];
                assert!(!compatible(&current, &reordered, &scratch));
                let mut causal = next.clone();
                causal.independent_rows = false;
                assert!(prefix_end(&causal).is_none());
                let mut span_batch = next.clone();
                span_batch.sequence_spans = vec![SequenceSpan {
                    row_start: 0,
                    rows: rows as u32,
                    start_pos: 7,
                }];
                assert!(prefix_end(&span_batch).is_none());
            }
            for mode in 0..3 {
                let mut bad = next.clone();
                match mode {
                    0 => bad.mtp_verify = true,
                    1 => bad.prefill_target_tokens = Some(2),
                    _ => bad.prefill_next_chunk = true,
                }
                assert!(!compatible(&current, &bad, &scratch));
            }
        }
        assert!(prefix_end(&independent_cohort(
            &[1; infr_core::backend::MAX_DECODE_ROWS + 1]
        ))
        .is_none());
    }

    #[test]
    fn predicted_kv_depth_declines_at_uncommitted_segment_boundary_and_allows_ring_wrap() {
        assert!(committed_depth_fits(3, 1, 8, 128, 32));
        assert!(!committed_depth_fits(4, 1, 8, 128, 32));
        assert!(committed_depth_fits(4, 1, 8, 128, 64));
        assert!(committed_depth_fits(1000, 1, 8, 128, 128));
        assert!(!committed_depth_fits(1000, 1, 8, 128, 120));
        assert!(committed_depth_fits(6, 4, 8, 128, 8));
        assert!(!committed_depth_fits(7, 4, 8, 128, 8));
        assert!(!committed_depth_fits(usize::MAX, 1, 8, 128, 128));
        assert!(!committed_depth_fits(0, 0, 8, 128, 128));
    }

    #[test]
    fn prepared_independent_kv_fusion_is_cleared_and_writekv_restored() {
        let mut graph = independent_cohort(&[3, 7]);
        let positions = graph.input(TensorDesc::new([2], DType::I32));
        let cache = graph.input(TensorDesc::new([16, 8], DType::F16));
        let rotated = graph.internal(TensorDesc::new([2, 8], DType::F16));
        let x = graph.inputs[0];
        let weight = graph.weights[0];
        graph.ops.splice(
            0..0,
            [
                Op::QkNormRope {
                    x,
                    weight,
                    positions,
                    dst: rotated,
                    rows: 2,
                    n_head: 1,
                    head_dim: 8,
                    rope_dim: 8,
                    theta: 10000.0,
                    eps: 1e-6,
                    freq_factors: None,
                    x_stride: 0,
                },
                Op::WriteKv {
                    src: rotated,
                    cache,
                    rows: 2,
                    row_stride: 8,
                    pos: 0,
                },
            ],
        );
        let plan = VkDecodePlan::new_cpu(&graph, false, true);
        let original = plan.setup_analysis.fusions(true);
        assert!(!original.kv_write.is_empty());
        assert!(original.skip.contains(&1));
        let prepared = prepared_fusions(&plan, true);
        assert!(prepared.kv_write.is_empty());
        assert!(!prepared.skip.contains(&1));
        let states = [CpuBuffer::new(256, 0x1000), CpuBuffer::new(256, 0x2000)];
        let scratch = cpu_scratch(&graph);
        let mut bindings = Bindings::new();
        bindings.bind_rows(cache, vec![&states[0], &states[1]]);
        assert!(kv_depth_ready(&graph, &scratch, &bindings, &[3, 7]));
        assert!(cohort_bindings_match(&graph, &bindings, &graph, &bindings));
        let mut swapped = Bindings::new();
        swapped.bind_rows(cache, vec![&states[1], &states[0]]);
        assert!(!cohort_bindings_match(&graph, &bindings, &graph, &swapped));
        bindings.bind_rows(cache, vec![&states[0], &states[0]]);
        assert!(!kv_depth_ready(&graph, &scratch, &bindings, &[3, 7]));
        bindings.bind_rows(cache, vec![&states[0]]);
        assert!(!kv_depth_ready(&graph, &scratch, &bindings, &[3, 7]));
    }

    #[test]
    fn compatibility_accepts_only_a_proven_single_independent_lane() {
        for position in [0, 37, 32767, u32::MAX - 2] {
            let current = independent_graph(position);
            let next = independent_graph(position + 1);
            let mut scratch = cpu_scratch(&next);
            assert!(compatible(&current, &next, &scratch));
            let index = scratch.iter().position(Option::is_some).unwrap();
            let required = scratch[index].as_ref().unwrap().len_bytes();
            scratch[index] = Some(Box::new(CpuBuffer::new(required - 1, 0)));
            assert!(!compatible(&current, &next, &scratch));
        }
    }

    #[test]
    fn independent_lane_rejects_missing_malformed_and_multiple_spans() {
        let current = independent_graph(37);
        let next = independent_graph(38);
        let scratch = cpu_scratch(&next);
        let valid = current.sequence_spans[0];
        for spans in [
            vec![],
            vec![SequenceSpan {
                row_start: 1,
                ..valid
            }],
            vec![SequenceSpan { rows: 0, ..valid }],
            vec![SequenceSpan { rows: 2, ..valid }],
            vec![
                valid,
                SequenceSpan {
                    row_start: 1,
                    ..valid
                },
            ],
            vec![SequenceSpan {
                start_pos: u32::MAX,
                ..valid
            }],
        ] {
            let mut bad_current = current.clone();
            bad_current.sequence_spans = spans.clone();
            assert!(!compatible(&bad_current, &next, &scratch), "{spans:?}");
            let mut bad_next = next.clone();
            bad_next.sequence_spans = spans;
            assert!(!compatible(&current, &bad_next, &scratch));
        }
    }

    #[test]
    fn independent_lane_requires_consecutive_positions_and_unchanged_mode() {
        let current = independent_graph(37);
        let next = independent_graph(38);
        let scratch = cpu_scratch(&next);
        for position in [0, 36, 37, 39, u32::MAX] {
            assert!(!compatible(
                &current,
                &independent_graph(position),
                &scratch
            ));
        }
        assert!(!compatible(
            &independent_graph(u32::MAX),
            &independent_graph(0),
            &scratch
        ));
        assert!(!compatible(
            &independent_graph(u32::MAX - 1),
            &independent_graph(u32::MAX),
            &scratch
        ));
        let mut ordinary = current.clone();
        ordinary.independent_rows = false;
        assert!(!compatible(&ordinary, &next, &scratch));
        ordinary = next.clone();
        ordinary.independent_rows = false;
        assert!(!compatible(&current, &ordinary, &scratch));
        assert!(predicted_position_matches(&next, 38));
        assert!(!predicted_position_matches(&next, 37));
        assert!(!predicted_position_matches(&next, 39));
    }

    #[test]
    fn independent_lane_span_does_not_hide_multirow_moe_operations() {
        let current = independent_graph(37);
        let next = independent_graph(38);
        let scratch = cpu_scratch(&next);
        let mut bad_current = current.clone();
        if let Op::MoeFfn { ne, .. } = &mut bad_current.ops[2] {
            *ne = 4;
        }
        assert!(!compatible(&bad_current, &next, &scratch));
        let mut bad_next = next;
        if let Op::MoeFfn { ne, .. } = &mut bad_next.ops[2] {
            *ne = 4;
        }
        assert!(!compatible(&current, &bad_next, &scratch));
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

    #[test]
    fn ack_keys_match_capture_without_allocating_or_reordering_rows() {
        let id = TensorId(0);
        let rows = [CpuBuffer::new(32, 0x1000), CpuBuffer::new(32, 0x2000)];
        let replacement = CpuBuffer::new(32, 0x2000);
        let scratch: ScratchSet = vec![Some(Box::new(CpuBuffer::new(64, 0x3000)))];
        let mut bindings = Bindings::new();
        bindings.bind_rows(id, vec![&rows[0], &rows[1]]);
        let expected = keys(&scratch, &bindings, id).unwrap();
        let check = |bindings: &Bindings<'_>| {
            let original = keys(&scratch, bindings, id).as_ref().ok() == Some(&expected);
            assert_eq!(keys_match(&scratch, bindings, id, &expected), original);
        };
        check(&bindings);
        // Row bindings take precedence even when scratch has a different allocation.
        assert!(keys_match(&vec![], &bindings, id, &expected));
        bindings.bind_rows(id, vec![&rows[1], &rows[0]]);
        check(&bindings);
        bindings.bind_rows(id, vec![&rows[0], &replacement]);
        check(&bindings);
        bindings.bind_rows(id, vec![&rows[0]]);
        check(&bindings);
        bindings.bind_rows(id, vec![]);
        check(&bindings);
        assert!(keys_match(&scratch, &bindings, id, &[]));
        bindings.bind_rows(id, vec![&rows[0], &rows[1]]);
        for (bytes, address) in [(64, 0x1000), (32, 0), (32, 0x4000)] {
            rows[0].bytes.store(bytes, Ordering::Relaxed);
            rows[0].address.store(address, Ordering::Relaxed);
            check(&bindings);
            assert!(!keys_match(&scratch, &bindings, id, &expected));
        }
        bindings.bind(id, &rows[1]);
        let single = keys(&scratch, &bindings, id).unwrap();
        assert!(keys_match(&scratch, &bindings, id, &single));
        assert!(!keys_match(&scratch, &bindings, id, &[]));
        let bound_only: ScratchSet = vec![None];
        let single = keys(&bound_only, &bindings, id).unwrap();
        assert!(keys_match(&bound_only, &bindings, id, &single));
        assert!(!keys_match(&bound_only, &Bindings::new(), id, &single));
        assert!(!keys_match(&vec![], &bindings, id, &single));
        assert!(!keys_match(
            &scratch,
            &bindings,
            TensorId(u32::MAX),
            &single
        ));
    }

    #[test]
    fn ack_key_sequence_keeps_every_segmented_identity_and_committed_field() {
        let key = BufferKey {
            pointer: 1,
            bytes: 256,
            address: Some(0x1000),
            segmented: Some((7, 16, 128, 2)),
        };
        assert!(key_sequence_matches(
            std::slice::from_ref(&key),
            std::iter::once(key.clone())
        ));
        for field in 0..8 {
            let mut changed = key.clone();
            match field {
                0 => changed.pointer += 1,
                1 => changed.bytes += 1,
                2 => changed.address = None,
                3 => changed.segmented = None,
                4 => changed.segmented.as_mut().unwrap().0 += 1,
                5 => changed.segmented.as_mut().unwrap().1 += 1,
                6 => changed.segmented.as_mut().unwrap().2 += 1,
                7 => changed.segmented.as_mut().unwrap().3 += 1,
                _ => unreachable!(),
            }
            assert!(
                !key_sequence_matches(std::slice::from_ref(&key), std::iter::once(changed)),
                "field {field}"
            );
        }
        assert!(!key_sequence_matches(
            std::slice::from_ref(&key),
            std::iter::empty()
        ));
        assert!(!key_sequence_matches(&[], std::iter::once(key)));
    }

    #[test]
    fn ack_position_bytes_keep_lane_order_unaligned_reads_and_exact_bounds() {
        let positions = [0usize, 0x1234_5678, u32::MAX as usize, 37];
        let mut bytes = vec![0xa5];
        bytes.extend(
            positions
                .iter()
                .flat_map(|&value| (value as u32).to_le_bytes()),
        );
        bytes.push(0x5a);
        let data = &bytes[1..];
        assert!(position_bytes_match(data, &positions));
        assert!(position_bytes_match(&data[..16], &positions));
        assert!(position_bytes_match(&[], &[]));
        for size in 0..16 {
            assert!(!position_bytes_match(&data[..size], &positions));
        }
        for index in 0..16 {
            let mut changed = data.to_vec();
            changed[index] ^= 1;
            assert!(!position_bytes_match(&changed, &positions));
        }
        let mut reordered = positions;
        reordered.swap(1, 3);
        assert!(!position_bytes_match(data, &reordered));
        if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
            assert!(!position_bytes_match(&[0; 4], &[too_large]));
        }
        assert_eq!(position_byte_len(usize::MAX, usize::MAX), None);
        assert_eq!(position_byte_len(usize::MAX / 4 + 1, usize::MAX), None);
        assert_eq!(
            position_byte_len(isize::MAX as usize / 4 + 1, usize::MAX),
            None
        );
        assert_eq!(position_byte_len(1, 3), None);
        assert_eq!(position_byte_len(1, 4), Some(4));
        assert_eq!(position_byte_len(0, 0), Some(0));
        let scratch: ScratchSet = vec![None];
        assert!(ack_positions_match(&scratch, &Bindings::new(), &[]));
        let expected = [(TensorId(0), vec![37])];
        assert!(!ack_positions_match(&scratch, &Bindings::new(), &expected));
        assert!(!ack_positions_match(&vec![], &Bindings::new(), &expected));
        let non_vulkan = CpuBuffer::new(4, 0x1000);
        let mut bindings = Bindings::new();
        bindings.bind(TensorId(0), &non_vulkan);
        assert!(!ack_positions_match(&scratch, &bindings, &expected));
    }

    #[test]
    #[ignore = "bounded CPU-only ACK validation allocation comparison; no Vulkan backend"]
    fn ack_validation_allocation_free_cpu_probe() {
        use std::hint::black_box;
        const TENSORS: usize = 128;
        const REPEATS: usize = 2048;
        for rows in [1, 2, 8] {
            let buffers: Vec<_> = (0..rows)
                .map(|i| CpuBuffer::new(4096, 0x1000 + i as u64 * 4096))
                .collect();
            let scratch: ScratchSet = (0..TENSORS).map(|_| None).collect();
            let mut bindings = Bindings::new();
            for tensor in 0..TENSORS {
                if rows == 1 {
                    bindings.bind(TensorId(tensor as u32), &buffers[0]);
                } else {
                    bindings.bind_rows(
                        TensorId(tensor as u32),
                        buffers.iter().map(|v| v as &dyn Buffer).collect(),
                    );
                }
            }
            let tensors: Vec<_> = (0..TENSORS)
                .map(|i| {
                    let id = TensorId(i as u32);
                    (id, keys(&scratch, &bindings, id).unwrap())
                })
                .collect();
            let positions: Vec<_> = (0..rows).map(|i| 30000 + i).collect();
            let bytes: Vec<_> = positions
                .iter()
                .flat_map(|&v| (v as u32).to_le_bytes())
                .collect();
            let expected: Vec<_> = (0..16).map(|i| (TensorId(i), positions.clone())).collect();
            let mut elapsed = [Vec::new(), Vec::new()];
            for round in 0..6 {
                for arm in 0..2 {
                    let variant = (round + arm) % 2;
                    let start = std::time::Instant::now();
                    for _ in 0..REPEATS {
                        let valid = if variant == 0 {
                            tensors.iter().all(|(id, value)| {
                                keys(black_box(&scratch), black_box(&bindings), *id)
                                    .as_ref()
                                    .ok()
                                    == Some(value)
                            }) && positions_match(&expected, |_, rows| {
                                // The original mapping path copies into bytes, then collects usize.
                                let data = black_box(&bytes)[..rows * 4].to_vec();
                                Some(
                                    data.chunks_exact(4)
                                        .map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize)
                                        .collect(),
                                )
                            })
                        } else {
                            tensors.iter().all(|(id, value)| {
                                keys_match(black_box(&scratch), black_box(&bindings), *id, value)
                            }) && expected
                                .iter()
                                .all(|(_, value)| position_bytes_match(black_box(&bytes), value))
                        };
                        assert!(black_box(valid));
                    }
                    if round >= 2 {
                        elapsed[variant].push(start.elapsed().as_secs_f64() * 1e6 / REPEATS as f64);
                    }
                }
            }
            for (arm, elapsed) in elapsed.iter().enumerate() {
                let mean = elapsed.iter().sum::<f64>() / elapsed.len() as f64;
                eprintln!("ACK_VALIDATION_CPU scope=keys+position-bytes tensors={TENSORS} position_tensors=16 rows={rows} rounds=6 warmup=2 repeats={REPEATS} arm={} mean_us={mean:.3}", if arm == 0 { "allocating" } else { "allocation-free" });
            }
        }
    }
}
