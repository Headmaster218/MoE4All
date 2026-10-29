//! Opt-in CB grouping. This never changes an ordinal, predicate, source, or notice owner.
use super::*;

const MAX_PURE_LAYERS: usize = 4;

#[cfg(test)]
#[derive(Clone, Debug)]
pub(super) struct Decision {
    pub entries: usize,
    pub submitted: usize,
    pub due: bool,
    pub preparing: bool,
}

#[cfg(test)]
std::thread_local! {
    static DECISIONS: std::cell::RefCell<Option<Vec<Decision>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(super) struct TestTrace;
#[cfg(test)]
impl TestTrace {
    pub(super) fn begin() -> Self {
        DECISIONS.with(|trace| assert!(trace.replace(Some(Vec::new())).is_none()));
        Self
    }
    pub(super) fn decisions(&self) -> Vec<Decision> {
        DECISIONS.with(|trace| trace.borrow().as_ref().unwrap().clone())
    }
}
#[cfg(test)]
impl Drop for TestTrace {
    fn drop(&mut self) {
        DECISIONS.with(|trace| {
            trace.take();
        });
    }
}

#[derive(Clone)]
struct Entry {
    op_idx: usize,
    roles: [usize; 3],
    layout: ResidentMoeLayout,
    pure: bool,
}

pub(super) struct Schedule {
    owner: Arc<crate::pager::GpuCopyBuffers>,
    epoch: u64,
    submit_setting: usize,
    resources: RollingMappedAdmissionKey,
    entries: Vec<Entry>,
    pub(super) bank_tensors: Vec<TensorId>,
}

fn enabled(be_: &VulkanBackend) -> bool {
    be_.cfg()
        .kernels
        .vulkan
        .decode_gpu_resident_pure_mapped_layers
        == MAX_PURE_LAYERS
}

/// First useful work and every potentially host-serviced entry keep the legacy grid.
/// Pure runs may end between grid points, but must never postpone the next host boundary.
fn target(first_pure: bool, run: usize, submitted: usize, base: usize) -> Option<usize> {
    if !(1..=2).contains(&base) || run == 0 {
        return None;
    }
    let legacy = base - submitted % base;
    if submitted == 0 || !first_pure {
        return Some(legacy);
    }
    Some(run.min(MAX_PURE_LAYERS))
}

impl Schedule {
    fn capture(
        be_: &VulkanBackend,
        plan: &VkDecodePlan,
        scratch: &ScratchSet,
        bindings: &Bindings,
        pool: &ScratchPool,
        chunk: &ResidentChunk,
        phase_arena: bool,
        cap: usize,
    ) -> Option<Arc<Self>> {
        if !enabled(be_)
            || chunk.self_service
            || !phase_arena
            || cap != 0
            || !(1..=2).contains(&chunk.submit_layers)
            || !be_
                .cfg()
                .kernels
                .vulkan
                .decode_gpu_resident_immutable_predicates
            || chunk.control.len_bytes() != crate::recorder::RESIDENT_CONTROL_BYTES
            || chunk.indirect.is_some()
            || plan.graph.mtp_verify
            || plan.graph.prefill_target_tokens.is_some()
            || be_.cfg().spec.mtp
            || !rolling_mapped_admission(be_, plan, pool, phase_arena, true, cap)
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
        // One current-graph binding/classification pass under the existing phase lease.
        // Aliased roles/banks reuse the exact same registered-bank certificate.
        let mut banks = HashMap::new();
        let mut entries = Vec::new();
        let mut bank_tensors = Vec::new();
        for (op_idx, op) in plan.graph.ops.iter().enumerate() {
            let Op::MoeFfn {
                gate_exps,
                up_exps,
                down_exps,
                ..
            } = op
            else {
                continue;
            };
            let roles = [*gate_exps, *up_exps, *down_exps].map(|id| {
                resolve(scratch, bindings, id)
                    .ok()
                    .map(crate::pager::buffer_identity)
            });
            bank_tensors.extend([*gate_exps, *up_exps, *down_exps]);
            let roles = [roles[0]?, roles[1]?, roles[2]?];
            let layout = ResidentMoeLayout::from_op(&plan.graph, op)?;
            let key = (roles, layout.n_expert);
            let pure = if let Some(&pure) = banks.get(&key) {
                pure
            } else {
                let params = owner.role_params(session, &roles).ok()?;
                let pure = owner.owned_roles_self_service(
                    [params[0], params[1], params[2]],
                    layout.n_expert as u32,
                );
                banks.insert(key, pure);
                pure
            };
            entries.push(Entry {
                op_idx,
                roles,
                layout,
                pure,
            });
        }
        (!entries.is_empty()).then(|| {
            Arc::new(Self {
                owner: Arc::clone(owner),
                epoch: be_.shared.resource_epoch.load(Ordering::Acquire),
                submit_setting: be_.cfg().kernels.vulkan.decode_gpu_resident_submit_layers,
                resources: RollingMappedAdmissionKey::current(be_, pool),
                entries,
                bank_tensors,
            })
        })
    }

    fn index(&self, entry: &ResidentJournalEntry) -> Option<usize> {
        let i = self
            .entries
            .binary_search_by_key(&entry.op_idx, |entry| entry.op_idx)
            .ok()?;
        let expected = &self.entries[i];
        (entry.roles == expected.roles
            && entry.layout == expected.layout
            && entry.host_notice != expected.pure)
            .then_some(i)
    }

    pub(super) fn notice(
        &self,
        be_: &VulkanBackend,
        copy: Option<&Arc<crate::pager::GpuCopyBuffers>>,
        entry: &ResidentJournalEntry,
    ) -> Option<bool> {
        if !enabled(be_)
            || self.epoch != be_.shared.resource_epoch.load(Ordering::Acquire)
            || !copy.is_some_and(|copy| Arc::ptr_eq(copy, &self.owner))
        {
            return None;
        }
        let i = self
            .entries
            .binary_search_by_key(&entry.op_idx, |entry| entry.op_idx)
            .ok()?;
        let expected = &self.entries[i];
        (expected.roles == entry.roles && expected.layout == entry.layout).then_some(!expected.pure)
    }

    fn target(
        &self,
        entries: &[ResidentJournalEntry],
        submitted: usize,
        base: usize,
    ) -> Option<usize> {
        let first = self.index(entries.get(submitted)?)?;
        // Every already-recorded request must still name this exact graph/binding certificate.
        for (offset, entry) in entries[submitted..].iter().enumerate() {
            if self.index(entry)? != first + offset {
                return None;
            }
        }
        let run = if self.entries[first].pure {
            self.entries[first..]
                .iter()
                .take(MAX_PURE_LAYERS)
                .take_while(|entry| entry.pure)
                .count()
        } else {
            1
        };
        target(self.entries[first].pure, run, submitted, base)
    }

    fn next_group(&self, entries: &[ResidentJournalEntry], submitted: usize, base: usize) -> bool {
        if submitted == 0 || entries.len() != submitted {
            return false;
        }
        let Some(last) = entries.last().and_then(|entry| self.index(entry)) else {
            return false;
        };
        self.entries[last + 1..]
            .iter()
            .take(MAX_PURE_LAYERS)
            .take_while(|entry| entry.pure)
            .count()
            > base
    }

    pub(super) fn current(&self, be_: &VulkanBackend, chunk: &ResidentChunk) -> bool {
        enabled(be_)
            && be_
                .cfg()
                .kernels
                .vulkan
                .decode_gpu_resident_immutable_predicates
            && self.submit_setting == be_.cfg().kernels.vulkan.decode_gpu_resident_submit_layers
            && self.epoch == be_.shared.resource_epoch.load(Ordering::Acquire)
            && chunk
                .gpu_copy
                .as_ref()
                .is_some_and(|copy| Arc::ptr_eq(copy, &self.owner))
    }

    pub(super) fn resources_current(
        &self,
        be_: &VulkanBackend,
        chunk: &ResidentChunk,
        pool: &ScratchPool,
    ) -> bool {
        self.current(be_, chunk) && self.resources == RollingMappedAdmissionKey::current(be_, pool)
    }
}

fn unconfirmed_host(entries: &[ResidentJournalEntry], confirmed: usize, submitted: usize) -> bool {
    confirmed > submitted
        || submitted > entries.len()
        || entries[confirmed..submitted]
            .iter()
            .any(|entry| entry.host_notice)
}

impl ResidentChunk {
    pub(super) fn coalescing_allowed(
        &self,
        be_: &VulkanBackend,
        pool: &ScratchPool,
        ps: &PagedStream,
    ) -> bool {
        self.pure_mapped_schedule.as_ref().is_some_and(|schedule| {
            schedule.current(be_, self)
                && schedule.resources == RollingMappedAdmissionKey::current(be_, pool)
                && schedule.owner.owned_active()
                && schedule.owner.owned_in_forward()
                && rolling_mapped_stream_clear(ps)
        })
    }
    pub(super) fn install_pure_mapped_schedule(
        &mut self,
        be_: &VulkanBackend,
        plan: &VkDecodePlan,
        scratch: &ScratchSet,
        bindings: &Bindings,
        pool: &ScratchPool,
        phase_arena: bool,
        cap: usize,
        acknowledged: bool,
    ) {
        // Global self-service already prepares a single whole-forward CB. There is no
        // mixed-owner run to coalesce, so do not scan/classify/extend ACK bindings for it.
        // Keep the disabled configuration equally cheap.
        if self.self_service || !enabled(be_) {
            self.pure_mapped_schedule = None;
            return;
        }
        // The ACK validated every bank_tensors BufferKey, including banks beyond the
        // prefix horizon. Reuse that physical-binding proof instead of re-scanning sources.
        if acknowledged
            && self.pure_mapped_schedule.as_ref().is_some_and(|schedule| {
                schedule.current(be_, self)
                    && schedule.resources == RollingMappedAdmissionKey::current(be_, pool)
            })
        {
            return;
        }
        // Re-capture on the actual graph/bindings, never substitute a different copy owner.
        // A declined admission drops the optional policy and leaves legacy cadence intact.
        self.pure_mapped_schedule =
            Schedule::capture(be_, plan, scratch, bindings, pool, self, phase_arena, cap);
    }

    pub(super) fn pure_group_due(&self, submitted: usize, allow: bool, preparing: bool) -> bool {
        let legacy = resident_submit_due(
            self.entries.len(),
            submitted,
            self.submit_layers,
            self.limit,
        );
        if !allow
            || self.recovery_op.is_some()
            || self.notice_stop
            || self.notice_resume.is_some()
            || !(1..=2).contains(&self.submit_layers)
        {
            return legacy;
        }
        let Some(schedule) = &self.pure_mapped_schedule else {
            return legacy;
        };
        let Some(target) = schedule.target(&self.entries, submitted, self.submit_layers) else {
            return legacy;
        };
        // Pre-recording chooses boundaries only; confirm_before_pure_group still gates
        // actual submission against the REAL cursor. No speculative ACK is published.
        if !preparing && unconfirmed_host(&self.entries, self.notice_cursor, submitted) {
            return legacy;
        }
        let due = self.entries.len() < self.limit && self.entries.len() - submitted >= target;
        #[cfg(test)]
        DECISIONS.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                trace.push(Decision {
                    entries: self.entries.len(),
                    submitted,
                    due,
                    preparing,
                });
            }
        });
        due
    }

    pub(super) fn confirm_before_pure_group(
        &mut self,
        be_: &VulkanBackend,
        ps: &mut PagedStream,
        pending: Option<&crate::recorder::PendingSegment>,
        prepared_end: Option<usize>,
    ) -> Result<()> {
        if !enabled(be_)
            || self.notice_resume.is_some()
            || self.notice_stop
            || self.recovery_op.is_some()
        {
            return Ok(());
        }
        let Some(schedule) = &self.pure_mapped_schedule else {
            return Ok(());
        };
        let first = self.submitted_entries;
        let grouped = match prepared_end {
            Some(end) => {
                first < end
                    && end <= self.entries.len()
                    && end - first > self.submit_layers
                    && self.entries[first..end]
                        .iter()
                        .all(|entry| !entry.host_notice)
            }
            None => schedule.next_group(&self.entries, first, self.submit_layers),
        };
        if grouped
            && (!schedule.current(be_, self)
                || !rolling_mapped_stream_clear(ps)
                || !schedule.owner.owned_active()
                || !schedule.owner.owned_in_forward())
        {
            return Err(be("coalesced group lost its admitted owned scope"));
        }
        if grouped && unconfirmed_host(&self.entries, self.notice_cursor, first) {
            // This is the existing real event/result protocol, NOT an invented ACK.
            // A miss is handled by the caller's original stop/truncate/recovery branch.
            let pending = pending.ok_or_else(|| be("coalesced group lost predecessor pending"))?;
            self.poll_notices(be_, ps, Some(pending))?;
        }
        if grouped
            && self.notice_resume.is_none()
            && !self.notice_stop
            && unconfirmed_host(&self.entries, self.notice_cursor, first)
        {
            return Err(be("coalesced group predecessor remains unconfirmed"));
        }
        Ok(())
    }

    pub(super) fn preserve_coalesced_boundary(
        &self,
        recorder: &Recorder<'_>,
        submitted: usize,
        allow: bool,
        preparing: bool,
    ) {
        if self.pure_mapped_schedule.is_some()
            && self.entries.len() < self.limit
            && self.entries.len().is_multiple_of(self.submit_layers)
            && resident_submit_due(
                self.entries.len(),
                submitted,
                self.submit_layers,
                self.limit,
            )
            && !self.pure_group_due(submitted, allow, preparing)
        {
            // Remove only the CB cut. Preserve the old inter-CB memory dependencies
            // in command order, including BDA accesses the descriptor tracker cannot see.
            recorder.end_conditional();
            recorder.seed_barrier();
            if !preparing {
                recorder.arena_stream_barrier();
            }
            self.begin_conditional(recorder);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_mapped_coalesce_first_submit_host_grid_and_run_bound() {
        for base in [1, 2] {
            for mask in 0u32..256 {
                let pure: Vec<_> = (0..8).map(|i| mask & (1 << i) == 0).collect();
                let mut start = 0;
                while start < pure.len() {
                    let run = if pure[start] {
                        pure[start..].iter().take_while(|&&p| p).count()
                    } else {
                        1
                    };
                    let count = target(pure[start], run, start, base)
                        .unwrap()
                        .min(pure.len() - start);
                    assert!((1..=MAX_PURE_LAYERS).contains(&count));
                    if start == 0 {
                        assert_eq!(count, base);
                    }
                    if start != 0 && pure[start] {
                        assert!(pure[start..start + count].iter().all(|&p| p));
                    }
                    for (host, &is_pure) in pure
                        .iter()
                        .enumerate()
                        .skip(start)
                        .take(count)
                    {
                        if !is_pure {
                            assert!(start + count <= (host + 1).div_ceil(base) * base);
                        }
                    }
                    start += count;
                }
            }
        }
        assert_eq!(target(true, 9, 1, 1), Some(4));
        assert_eq!(target(true, 2, 1, 1), Some(2));
        assert_eq!(target(true, 8, 2, 2), Some(4));
        assert_eq!(target(true, 1, 0, 0), None);
        assert_eq!(target(true, 1, 0, 3), None);
    }

    #[test]
    fn pure_mapped_coalesce_frontier_never_confuses_holes_with_ack() {
        let entries: Vec<_> = [false, true, false, false]
            .into_iter()
            .enumerate()
            .map(|(op_idx, host_notice)| ResidentJournalEntry {
                op_idx,
                roles: [0; 3],
                layout: ResidentMoeLayout {
                    rows: 1,
                    n_expert: 4,
                    n_used: 2,
                },
                host_notice,
            })
            .collect();
        assert!(unconfirmed_host(&entries, 0, 2));
        assert!(unconfirmed_host(&entries, 1, 2));
        assert!(!unconfirmed_host(&entries, 2, 4));
        assert!(unconfirmed_host(&entries, 3, 2));
        assert!(unconfirmed_host(&entries, 0, 5));
    }

    #[test]
    fn pure_mapped_coalesce_layout_and_binding_match_are_not_shape_only() {
        for rows in [1, 2, 4] {
            let layout = ResidentMoeLayout {
                rows,
                n_expert: 512,
                n_used: 10,
            };
            let mut expected = ResidentJournalEntry {
                op_idx: 7,
                roles: [10, 20, 10],
                layout,
                host_notice: false,
            };
            let raw: Arc<dyn Buffer> = Arc::new(Stub);
            let schedule = Schedule {
                owner: crate::pager::GpuCopyBuffers::test_stub(raw),
                epoch: 0,
                submit_setting: 0,
                resources: RollingMappedAdmissionKey {
                    resource_epoch: 0,
                    pool_buffers: 0,
                    needs_compaction: false,
                },
                entries: vec![Entry {
                    op_idx: 7,
                    roles: expected.roles,
                    layout,
                    pure: true,
                }],
                bank_tensors: Vec::new(),
            };
            assert_eq!(schedule.index(&expected), Some(0));
            expected.roles[1] = 30;
            assert_eq!(schedule.index(&expected), None);
            expected.roles[1] = 20;
            expected.layout.n_used = 8;
            assert_eq!(schedule.index(&expected), None);
            expected.layout = layout;
            expected.host_notice = true;
            assert_eq!(schedule.index(&expected), None);
        }
    }

    struct Stub;
    impl Buffer for Stub {
        fn len_bytes(&self) -> usize {
            0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}
