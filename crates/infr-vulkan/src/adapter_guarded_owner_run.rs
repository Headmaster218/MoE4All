//! Same-window queue admission for GPU-owned banks which may require host service.
use super::*;

#[derive(Default)]
pub(super) struct Admission {
    // Only workspace capacity is cached. Owner/notice/window provenance is checked each time.
    workspace: RollingMappedAdmissionCache,
}

fn host_queue_budget(inflight: usize) -> usize {
    // Keep one speculative successor beyond the existing rolling window. Mapped-only
    // spans do not use these credits and retain their whole-span queue admission.
    inflight.clamp(1, 2) + 1
}

fn bounded_host_run(
    same_window: bool,
    entries: &[ResidentJournalEntry],
    confirmed: usize,
    submitted: usize,
    sent: usize,
    capacity: usize,
    pending: usize,
) -> bool {
    same_window
        && confirmed <= submitted
        && sent == submitted
        && submitted < entries.len()
        && entries.len() <= capacity
        && pending > 0
        && pending <= entries.len()
        && entries[confirmed..].iter().any(|entry| entry.host_notice)
}

impl Admission {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check(
        &mut self,
        be_: &VulkanBackend,
        plan: &VkDecodePlan,
        pool: &ScratchPool,
        ps: &PagedStream,
        chunk: &ResidentChunk,
        same_window: bool,
        using_phase_arena: bool,
        resident_banks_supported: bool,
        cap: usize,
        pending_after_submit: usize,
    ) -> bool {
        #[cfg(test)]
        if tests::disabled() {
            return false;
        }
        if !bounded_host_run(
            same_window,
            &chunk.entries,
            chunk.notice_cursor,
            chunk.submitted_entries,
            chunk.notices_sent,
            chunk.limit,
            pending_after_submit,
        ) || pending_after_submit
            > host_queue_budget(be_.cfg().kernels.vulkan.decode_gpu_resident_inflight)
            || !using_phase_arena
            || !resident_banks_supported
            || cap != 0
            || !rolling_mapped_stream_clear(ps)
            || chunk.recovery_op.is_some()
            || chunk.notice_stop
            || chunk.notice_resume.is_some()
            || chunk.indirect.is_some()
            || chunk.self_service
            || !be_
                .cfg()
                .kernels
                .vulkan
                .decode_gpu_resident_immutable_predicates
            || be_
                .cfg()
                .kernels
                .vulkan
                .decode_gpu_resident_indirect_predicates
            || be_.shared.conditional_rendering.is_none()
            || chunk.control.len_bytes() != crate::recorder::RESIDENT_CONTROL_BYTES
        {
            return false;
        }
        let (Some(copy), Some(producer), Some(notices)) = (
            chunk.gpu_copy.as_ref(),
            chunk.producer.as_ref(),
            chunk.notices.as_ref(),
        ) else {
            return false;
        };
        if !copy.is_owned()
            || !copy.owned_active()
            || !copy.owned_in_forward()
            || producer.sender.is_none()
            || !Arc::ptr_eq(&producer.pager, &be_.moe_pager)
            || !Arc::ptr_eq(&notices.shared, &be_.shared)
            || producer.announced_miss.load(Ordering::Acquire) != usize::MAX
            || resident_owned_protocol_capacity(
                chunk.control.len_bytes(),
                true,
                chunk.journal.len_bytes(),
                notices.events.len(),
                copy.jobs.len_bytes(),
                copy.miss_predicates
                    .as_ref()
                    .map(|buffer| buffer.len_bytes()),
            ) < chunk.limit
            || ![
                chunk.control.as_ref(),
                chunk.journal.as_ref(),
                copy.jobs.as_ref(),
            ]
            .into_iter()
            .all(|buffer| {
                crate::as_vk_buf(buffer).is_ok_and(|buffer| {
                    Arc::ptr_eq(&buffer.shared, &be_.shared) && buffer.mapped_ptr().is_some()
                })
            })
        {
            return false;
        }
        // Unlike mapped admission, HOST-potential is allowed. A real status5 clears run;
        // every later ordinal still publishes its own SKIP mailbox unconditionally. No
        // CPU placement/LUT writes occur while recording this owned window. Recovery must
        // retain the existing ALL old segments + producer ACK/event reset/seed barriers.
        self.workspace.check_if(
            true,
            || RollingMappedAdmissionKey::current(be_, pool),
            || {
                rolling_mapped_admission(
                    be_,
                    plan,
                    pool,
                    using_phase_arena,
                    resident_banks_supported,
                    cap,
                )
            },
        )
    }
}

#[cfg(test)]
#[path = "adapter_guarded_owner_run_tests.rs"]
pub(super) mod tests;
