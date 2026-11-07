//! Experimental payload-only transfer stage; normal hit-chain barriers stay compute-only.
use super::*;

const CAPACITY: usize = infr_core::backend::MAX_DECODE_ROWS * 16 * 3;
const COMMAND_STRIDE: usize = 32;

impl Recorder<'_> {
    pub(crate) fn mark_owned_indirect_copy_ready(&self, event: vk::Event) {
        assert!(!self.conditional_active.get());
        unsafe {
            self.be
                .shared
                .device
                .cmd_set_event(self.cmd, event, vk::PipelineStageFlags::TRANSFER);
        }
    }

    /// The event's first scope captures only copies recorded before its SetEvent. Later
    /// Down transfers must not enter the Gate/Up data-ready dependency through a global join.
    pub(crate) fn wait_owned_indirect_copy_ready(&self, event: vk::Event, transfer: bool) {
        assert!(!self.conditional_active.get());
        let (stage, access) = if transfer {
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
            )
        } else {
            (
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ,
            )
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(access);
        unsafe {
            self.be.shared.device.cmd_wait_events(
                self.cmd,
                &[event],
                vk::PipelineStageFlags::TRANSFER,
                stage,
                &[barrier],
                &[],
                &[],
            );
        }
        *self.barriers.borrow_mut() += 1;
    }

    /// The caller retains every source, destination and sidecar until completion, and proves
    /// current hit weights cannot be selected as copy destinations. This must execute outside
    /// conditional rendering: indirect-copy commands themselves ignore that predicate.
    pub(crate) fn begin_owned_indirect_copy(
        &self,
        journal: &dyn Buffer,
        jobs: &dyn Buffer,
        commands: &dyn Buffer,
        dummy: &dyn Buffer,
        ordinal: u32,
        count: u32,
    ) -> Result<()> {
        self.begin_owned_indirect_copy_phase(journal, jobs, commands, dummy, ordinal, count, [0; 4])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_owned_indirect_copy_phase(
        &self,
        journal: &dyn Buffer,
        jobs: &dyn Buffer,
        commands: &dyn Buffer,
        dummy: &dyn Buffer,
        ordinal: u32,
        count: u32,
        phase: [u32; 4],
    ) -> Result<()> {
        self.begin_owned_indirect_copy_phase_flags(
            journal, jobs, commands, dummy, ordinal, count, phase, 0, 0,
        )
    }

    /// Test-only opt-in. The caller proves both hints for every real AND dummy payload;
    /// a mapped pointer or command-array allocation alone is not residency evidence.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_owned_indirect_copy_phase_flags(
        &self,
        journal: &dyn Buffer,
        jobs: &dyn Buffer,
        commands: &dyn Buffer,
        dummy: &dyn Buffer,
        ordinal: u32,
        count: u32,
        phase: [u32; 4],
        src_copy_flags: crate::indirect_copy::AddressCopyFlagsKHR,
        dst_copy_flags: crate::indirect_copy::AddressCopyFlagsKHR,
    ) -> Result<()> {
        if self.conditional_active.get()
            || self.independent_dispatch.borrow().is_some()
            || self.suppress.get()
            || self.no_barrier
            || self.full_barrier
            || count == 0
            || count as usize > CAPACITY
            || phase[0] > 2
            || (phase[0] != 0
                && (phase[3] == 0
                    || phase[1].checked_add(phase[3]).is_none()
                    || phase[2].checked_add(phase[3]).is_none()))
        {
            return Err(be(
                "indirect payload requires a bounded unconditional recording",
            ));
        }
        let layer_end = (ordinal as usize)
            .checked_add(1)
            .ok_or_else(|| be("indirect payload ordinal overflow"))?;
        let needed = layer_end
            .checked_mul(CAPACITY * COMMAND_STRIDE)
            .ok_or_else(|| be("indirect payload range overflow"))?;
        if journal.len_bytes() / RESIDENT_JOURNAL_BYTES < layer_end
            || jobs.len_bytes() < needed
            || commands.len_bytes() < needed
            || dummy.len_bytes() < 16 + layer_end * CAPACITY * 16
        {
            return Err(be("indirect payload sidecar is too small"));
        }
        let indirect = self
            .be
            .shared
            .indirect_copy
            .as_ref()
            .ok_or_else(|| be("indirect memory copy is not supported on this queue"))?;
        let command_address = commands
            .device_addr()
            .ok_or_else(|| be("indirect payload commands have no BDA"))?;
        let dummy_address = dummy
            .device_addr()
            .ok_or_else(|| be("indirect payload dummy has no BDA"))?;
        if (command_address | dummy_address) & 15 != 0 {
            return Err(be("indirect payload BDA must be 16-byte aligned"));
        }
        let kernel = self.be.kernel(
            "gpu_owned_indirect_jobs",
            crate::gemm::gpu_owned_indirect_jobs_spv(),
            3,
            32,
        );
        self.dispatch(
            kernel,
            &[Self::vkb(journal), Self::vkb(jobs), Self::vkb(commands)],
            1,
            bytemuck::cast_slice(&[
                ordinal,
                count,
                dummy_address as u32,
                (dummy_address >> 32) as u32,
                phase[0],
                phase[1],
                phase[2],
                phase[3],
            ]),
            1,
        );
        // ALL_COMMANDS includes the post-ash COPY_INDIRECT parameter stage. This barrier
        // precedes the payload, so it cannot join that later transfer into the hit chain.
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(
                vk::AccessFlags::INDIRECT_COMMAND_READ
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE,
            );
        unsafe {
            self.be.shared.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            indirect.record_with_flags(
                self.cmd,
                command_address + ordinal as u64 * (CAPACITY * COMMAND_STRIDE) as u64,
                count,
                COMMAND_STRIDE as u64,
                count as u64 * COMMAND_STRIDE as u64,
                src_copy_flags,
                dst_copy_flags,
            );
        }
        *self.barriers.borrow_mut() += 1;
        // Deliberately exclude this transfer from the ordinary compute hazard set. The
        // caller MUST join with global_barrier() before miss consumers or resource reuse.
        // This experimental entry point is test-only until its lifecycle is qualified.
        Ok(())
    }
}
