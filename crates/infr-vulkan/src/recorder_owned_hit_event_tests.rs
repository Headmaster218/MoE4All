#![cfg(test)]

//! Register as a child of recorder::owned_indirect_tests::fused:
//! #[path = "recorder_owned_hit_event_tests.rs"] pub(super) mod hit_event;
//!
//! Matrix child fused::splitk_hit_v2::hitfirst_v3 can call
//! super::super::hit_event::{HitEventResources, record_complete_hit} with &f.ff.
//! Keep resources alive until EVERY frame/replay referencing them has drained.
//! No submission, host wait, event polling, or persistent-file writes occur here.
//!
//! Proof: entry/reset barriers precede GU. GU writes only H ACT; SetEvent's
//! COMPUTE source scope ends before fullcopy. WaitEvent makes that SHADER_WRITE
//! visible to native Down's SHADER_READ, without including fullcopy. Immutable
//! jobs write disjoint M weights, never H weights or ACT/output. The final
//! unconditional global barrier joins Down AND fullcopy before the caller's M.
//! Descriptor hazards remain accumulated throughout; none are erased.
//!
//! Spec: https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWaitEvents2.html
//! https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWaitEvents.html
//! The scope is bounded by the signal, not by the wait. Same queue is required.
//! Events permit independent scheduling; they do not guarantee hardware overlap.
use super::*;

#[derive(Debug)]
pub(super) struct ComputeQueueDiagnostic {
    pub(super) main_family: u32,
    pub(super) main_family_queue_count: u32,
    /// Physical-device capacity only. lib.rs currently creates main queue 0 only.
    pub(super) same_family_aux_advertised: bool,
    pub(super) other_compute_families: Vec<(u32, u32)>,
}

/// Call once after backend creation, OUTSIDE frame timestamps. This only queries
/// physical-device properties and prints them; no device/queue creation, queue
/// retrieval, submission, synchronization, shader compilation, or GPU work.
pub(super) fn print_compute_queue_diagnostic(be: &VulkanBackend) -> ComputeQueueDiagnostic {
    let families = unsafe {
        be.shared
            .instance
            .get_physical_device_queue_family_properties(be.shared.physical_device)
    };
    let main_family = be.shared.queue_family_index;
    let main = &families[main_family as usize];
    let other_compute_families: Vec<_> = families
        .iter()
        .enumerate()
        .filter(|(index, p)| {
            *index != main_family as usize
                && p.queue_count > 0
                && p.queue_flags.contains(vk::QueueFlags::COMPUTE)
        })
        .map(|(index, p)| (index as u32, p.queue_count))
        .collect();
    let diagnostic = ComputeQueueDiagnostic {
        main_family,
        main_family_queue_count: main.queue_count,
        same_family_aux_advertised: main.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && main.queue_count >= 2,
        other_compute_families,
    };
    for (index, p) in families.iter().enumerate() {
        eprintln!(
            "HIT_EVENT_QUEUE_FAMILY family={index} main={} count={} flags=0x{:x} compute={} graphics={} transfer={} timestamp_bits={}",
            index == main_family as usize,
            p.queue_count,
            p.queue_flags.as_raw(),
            p.queue_flags.contains(vk::QueueFlags::COMPUTE),
            p.queue_flags.contains(vk::QueueFlags::GRAPHICS),
            p.queue_flags.contains(vk::QueueFlags::TRANSFER),
            p.timestamp_valid_bits,
        );
    }
    let boundary = if diagnostic.same_family_aux_advertised {
        "same_family_capacity_only_requires_two_queue_device_creation_and_semaphore_sync"
    } else if !diagnostic.other_compute_families.is_empty() {
        "other_family_only_requires_device_creation_semaphore_sync_and_buffer_sharing_or_ownership_audit"
    } else {
        "no_aux_compute_queue_advertised"
    };
    eprintln!(
        "HIT_EVENT_QUEUE_SUMMARY main_family={} main_advertised_count={} main_created_count=1 same_family_aux_advertised={} same_family_aux_created=false other_compute_families={:?} boundary={boundary} hardware_concurrency_verified=false",
        diagnostic.main_family,
        diagnostic.main_family_queue_count,
        diagnostic.same_family_aux_advertised,
        diagnostic.other_compute_families,
    );
    // Same-family queues need no ownership transfer merely for changing queues,
    // but still need two queues requested at device creation and semaphore edges.
    // Other-family compute requires auditing every accessed allocation (including
    // BDA weights and imported sources): concurrent sharing lists or matching
    // ownership transfers, command pools, lifetimes, and cross-queue semaphores.
    // Queue counts alone never establish independent hardware execution engines.
    diagnostic
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HitEventApi {
    Legacy,
    Sync2,
}

/// One event per retained frame resource. Serial replay is supported by the
/// explicit per-stage reset boundaries. Concurrent use from other queues is not.
pub(super) struct HitEventResources {
    shared: Arc<crate::VulkanShared>,
    event: vk::Event,
    producer: Option<ComputeKernel>,
    profile: Option<Profile>,
    shared_q8: bool,
    grid_buffer: bool,
    local: usize,
    nr: usize,
    api: HitEventApi,
    runtime_control: bool,
    batch_command: std::cell::Cell<vk::CommandBuffer>,
    batch_consumed: std::cell::Cell<bool>,
}

impl HitEventResources {
    /// No shader/pipeline creation. Use with the closure API for native GU/SiLU
    /// or existing split partial/reduce kernels, including shared Q8 fixtures.
    pub(super) fn event_only(be: &VulkanBackend) -> Self {
        Self::event_only_with_api(be, HitEventApi::Legacy)
    }

    pub(super) fn event_only_with_api(be: &VulkanBackend, api: HitEventApi) -> Self {
        if api == HitEventApi::Sync2 {
            // lib.rs chains this feature only when this test loader is enabled.
            assert!(
                be.shared
                    .indirect_copy
                    .as_ref()
                    .is_some_and(|d| d.synchronization2),
                "sync2 was not enabled by this backend's device creation"
            );
        }
        let flags = if api == HitEventApi::Sync2 {
            vk::EventCreateFlags::DEVICE_ONLY
        } else {
            vk::EventCreateFlags::empty()
        };
        let event = unsafe {
            be.shared
                .device
                .create_event(&vk::EventCreateInfo::default().flags(flags), None)
        }
        .unwrap();
        Self {
            shared: Arc::clone(&be.shared),
            event,
            producer: None,
            profile: None,
            shared_q8: false,
            grid_buffer: false,
            local: 0,
            nr: 0,
            api,
            runtime_control: std::env::var("MOE4ALL_HITFIRST_RUNTIME_CONTROL")
                .is_ok_and(|v| v == "1"),
            batch_command: std::cell::Cell::new(vk::CommandBuffer::null()),
            batch_consumed: std::cell::Cell::new(false),
        }
    }

    /// Default uses only Vulkan 1.0 event commands; no sync2 enable is assumed.
    /// Compile gpu_owned_hit_gu_activation.comp externally into
    /// MOE4ALL_HITFIRST_V3_DIR/hit-event-{q5k|iq2s|iq3s}-l{local}-nr{nr}[-shared][-buffer].spv.
    /// Suffix order is always -shared before -buffer. MOE4ALL_HITFIRST_GRID_BUFFER=1
    /// opts only IQ2S/IQ3S into buffer mode, sampled once at construction.
    /// Flags: -DLOCAL=N -DNR=N -DMAX_NFF=1024 -DFMT_Q5K (or FMT_IQ2S/IQ3S),
    /// plus -DUSE_GRID for LDS IQ, or the matching -DIQ2_GRID_BUFFER/-DIQ3_GRID_BUFFER,
    /// and -DSHARED_Q8 for the shared variant. Buffer and USE_GRID are exclusive.
    pub(super) fn new(
        be: &VulkanBackend,
        p: Profile,
        shared: bool,
        local: usize,
        nr: usize,
    ) -> Self {
        Self::with_api(be, p, shared, local, nr, HitEventApi::Legacy)
    }

    pub(super) fn with_api(
        be: &VulkanBackend,
        p: Profile,
        shared: bool,
        local: usize,
        nr: usize,
        api: HitEventApi,
    ) -> Self {
        assert!(matches!(local, 64 | 128 | 256) && matches!(nr, 1 | 2 | 4 | 8));
        assert!(p.nff <= 1024);
        assert!(matches!(p.gate_up, DType::Q5K | DType::Iq2S | DType::Iq3S));
        let grid_requested = std::env::var("MOE4ALL_HITFIRST_GRID_BUFFER").is_ok_and(|v| v == "1");
        let grid_buffer = grid_requested && matches!(p.gate_up, DType::Iq2S | DType::Iq3S);
        if grid_buffer {
            let grid = match p.gate_up {
                DType::Iq2S => &be.iq2s_grid,
                DType::Iq3S => &be.iq3s_grid,
                _ => unreachable!(),
            };
            assert!(
                grid.is_some(),
                "existing native grid buffer must be available"
            );
        }
        let suffix = if shared { "-shared" } else { "" };
        let grid_suffix = if grid_buffer { "-buffer" } else { "" };
        let file = format!(
            "hit-event-{}-l{local}-nr{nr}{suffix}{grid_suffix}.spv",
            format_name(p)
        );
        let root = PathBuf::from(
            std::env::var_os("MOE4ALL_HITFIRST_V3_DIR")
                .expect("external hit-event SPV directory required"),
        );
        let path = root.join(&file);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(bytes.len() % 4, 0);
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(words.first(), Some(&0x0723_0203));
        let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
        });
        eprintln!(
            "HIT_EVENT_SPV path={} bytes={} fnv1a64={hash:016x} api={api:?}",
            path.display(),
            bytes.len()
        );
        let name = Box::leak(format!("hit_event_{file}_{hash:016x}").into_boxed_str());
        let producer = be.kernel_sg(
            name,
            &words,
            4 + usize::from(grid_buffer),
            if shared { 64 } else { 40 },
            32,
        );
        let mut resources = Self::event_only_with_api(be, api);
        resources.producer = Some(producer);
        resources.profile = Some(p);
        resources.shared_q8 = shared;
        resources.grid_buffer = grid_buffer;
        resources.local = local;
        resources.nr = nr;
        resources
    }

    pub(super) fn api(&self) -> HitEventApi {
        self.api
    }

    fn producer_dependency(produced: &dyn Buffer) -> vk::BufferMemoryBarrier2<'static> {
        let b = Recorder::vkb_exact(produced);
        vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(b.buffer)
            .offset(b.offset)
            .size(b.range)
    }

    fn reset_before_producer(&self, rec: &Recorder<'_>) {
        if self.batch_command.get() != vk::CommandBuffer::null() {
            assert_eq!(self.batch_command.get(), rec.cmd);
            assert!(
                !self.batch_consumed.replace(true),
                "one signal per batch event"
            );
            return;
        }
        // ALL_COMMANDS explicitly orders the previous event wait/signal before
        // reset. A second boundary orders reset before the next signal. Neither
        // boundary can join this stage's copy: that dispatch is still unrecorded.
        event_boundary(rec);
        unsafe {
            match self.api {
                HitEventApi::Legacy => rec.be.shared.device.cmd_reset_event(
                    rec.cmd,
                    self.event,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                ),
                HitEventApi::Sync2 => rec.be.shared.device.cmd_reset_event2(
                    rec.cmd,
                    self.event,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                ),
            }
        }
        event_boundary(rec);
    }

    fn signal(&self, rec: &Recorder<'_>, dependency: &vk::BufferMemoryBarrier2<'_>) {
        unsafe {
            match self.api {
                HitEventApi::Legacy => rec.be.shared.device.cmd_set_event(
                    rec.cmd,
                    self.event,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                ),
                HitEventApi::Sync2 => rec.be.shared.device.cmd_set_event2(
                    rec.cmd,
                    self.event,
                    &vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(dependency)),
                ),
            }
        }
    }

    fn wait(&self, rec: &Recorder<'_>, dependency: &vk::BufferMemoryBarrier2<'_>) {
        unsafe {
            match self.api {
                HitEventApi::Legacy => {
                    let b = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(dependency.buffer)
                        .offset(dependency.offset)
                        .size(dependency.size);
                    rec.be.shared.device.cmd_wait_events(
                        rec.cmd,
                        &[self.event],
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        &[],
                        &[b],
                        &[],
                    );
                }
                HitEventApi::Sync2 => rec.be.shared.device.cmd_wait_events2(
                    rec.cmd,
                    &[self.event],
                    &[vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(dependency))],
                ),
            }
        }
        *rec.barriers.borrow_mut() += 1;
    }

    fn produce(&self, rec: &Recorder<'_>, f: &FusedFixture) {
        let i = &f.inner;
        let mut push = vec![
            i.p.ne as u32,
            i.p.nff as u32,
            i.rows as u32,
            i.p.used as u32,
            i.p.used as u32,
            f.physical_used() as u32,
            0,
            i.p.used as u32,
            (2 * i.p.used) as u32,
            1,
        ];
        if let Some(shared) = &f.shared {
            for matrix in &shared.matrices {
                let a = matrix.device_addr().unwrap();
                push.extend([a as u32, (a >> 32) as u32]);
            }
        }
        let mut act = Recorder::vkb_exact(i.activation.as_ref());
        act.range -= GUARD as u64;
        let groups = i.rows * f.physical_used() * i.p.nff.div_ceil(self.local / 64 * self.nr);
        let mut buffers = Vec::with_capacity(4 + usize::from(self.grid_buffer));
        if self.grid_buffer {
            // Borrow the backend-owned codebook under the existing frame lifetime contract.
            let grid = match i.p.gate_up {
                DType::Iq2S => &rec.be.iq2s_grid,
                DType::Iq3S => &rec.be.iq3s_grid,
                _ => panic!("buffer-grid hit producer requires IQ2S or IQ3S"),
            };
            buffers.push(Recorder::vkb_exact(
                grid.as_ref().expect("existing native grid buffer").as_ref(),
            ));
        }
        buffers.extend([
            Recorder::vkb_exact(i.input.as_ref()),
            Recorder::vkb_exact(i.ids.as_ref()),
            Recorder::vkb_exact(i.tape.as_ref()),
            act,
        ]);
        rec.dispatch_wide(
            self.producer
                .expect("compact producer requires external shader constructor"),
            &buffers,
            1,
            bytemuck::cast_slice(&push),
            groups as u32,
        );
    }
}

/// Test-only epoch: reset distinct events once at frame entry. Each resource is
/// consumed exactly once while recording this command, and retained until its
/// serial replays drain. The reset boundaries remain INSIDE frame timing.
/// Do not reuse these resources for another recording or a concurrent replay.
pub(super) fn record_batch_preamble(rec: &Recorder<'_>, resources: &[HitEventResources]) {
    check_recorder(rec);
    assert!(!resources.is_empty());
    let mut unique = std::collections::HashSet::new();
    for resource in resources {
        assert!(Arc::ptr_eq(&resource.shared, &rec.be.shared));
        assert_eq!(resource.batch_command.get(), vk::CommandBuffer::null());
        assert!(
            unique.insert(resource.event),
            "batch events must be distinct"
        );
    }
    event_boundary(rec);
    for resource in resources {
        unsafe {
            match resource.api {
                HitEventApi::Legacy => rec.be.shared.device.cmd_reset_event(
                    rec.cmd,
                    resource.event,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                ),
                HitEventApi::Sync2 => rec.be.shared.device.cmd_reset_event2(
                    rec.cmd,
                    resource.event,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                ),
            }
        }
        resource.batch_command.set(rec.cmd);
    }
    event_boundary(rec);
}

impl Drop for HitEventResources {
    fn drop(&mut self) {
        // Same ownership contract as fixture buffers: drain frame commands FIRST.
        unsafe { self.shared.device.destroy_event(self.event, None) };
    }
}

fn event_boundary(rec: &Recorder<'_>) {
    let b = vk::MemoryBarrier::default()
        .src_access_mask(
            vk::AccessFlags::HOST_WRITE
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::TRANSFER_WRITE,
        )
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        rec.be.shared.device.cmd_pipeline_barrier(
            rec.cmd,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[b],
            &[],
            &[],
        )
    };
    *rec.barriers.borrow_mut() += 1;
}

/// Bound suppression to exactly fullcopy + event wait + ONE native Down dispatch.
/// The event supplies the internal RAW; Drop ALWAYS joins the copy and Down.
/// Retaining dirty sets preserves conservative synchronization after this scope.
struct ActivationConsumerScope<'r, 'a>(&'r Recorder<'a>);
impl Drop for ActivationConsumerScope<'_, '_> {
    fn drop(&mut self) {
        self.0.suppress_sync(false);
        self.0.global_barrier();
    }
}

fn check_recorder(rec: &Recorder<'_>) {
    assert!(!rec.no_barrier && !rec.full_barrier && !rec.prof_ops);
    assert!(
        rec.timeline_query.borrow().is_none(),
        "per-op timestamps can distort overlap"
    );
    assert!(!rec.suppress.get() && !rec.conditional_active.get());
    assert!(!rec.indirect_pending.get() && rec.resident_indirect.borrow().is_none());
    assert!(rec.independent_dispatch.borrow().is_none());
}

fn interval(address: u64, bytes: usize) -> Range<u64> {
    assert_ne!(address, 0);
    assert!(bytes > 0);
    address
        ..address
            .checked_add(bytes as u64)
            .expect("BDA range overflow")
}
fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}
fn matrix_bytes(dtype: DType, elements: usize) -> usize {
    let (elements_per_block, bytes_per_block) = infr_core::decode_spec::block_layout(dtype);
    assert_eq!(elements % elements_per_block, 0);
    elements / elements_per_block * bytes_per_block
}

/// Validate BDA hazards which recorder descriptor tracking cannot see. Called at
/// recording time, with mapped metadata immutable and all earlier runs drained.
/// No owner admission or victim selection is performed here.
fn prove_disjoint(f: &FusedFixture, produced: &dyn Buffer) {
    let i = &f.inner;
    assert!((1..=8).contains(&i.rows) && (1..=16).contains(&i.p.used));
    assert!(i.job_count <= CAPACITY);
    let mut ids = i.routed.clone();
    let shared_bit = if f.shared.is_some() {
        1u32 << i.p.used
    } else {
        0
    };
    ids.extend(i.hits.iter().map(|m| m | shared_bit));
    let all = (1u32 << i.p.used) - 1;
    ids.extend(i.hits.iter().map(|m| all ^ m));
    assert_eq!(read(i.ids.as_ref()), bytemuck::cast_slice::<u32, u8>(&ids));
    assert_eq!(
        read(i.jobs.as_ref()),
        bytemuck::cast_slice::<u32, u8>(&i.job_words)
    );
    let journal = read(i.journal.as_ref());
    let word =
        |index: usize| u32::from_le_bytes(journal[index * 4..index * 4 + 4].try_into().unwrap());
    assert_eq!(
        word(ORDINAL * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_JOB_COUNT_WORD) as usize,
        i.job_count
    );
    let status = word(ORDINAL * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_STATUS_WORD);
    assert!(if i.job_count == 0 {
        status == 1
    } else {
        status == 4
    });
    let tape_bytes = read(i.tape.as_ref());
    let tape: Vec<u64> = tape_bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(tape.len(), 3 * i.p.used);
    let arena = interval(i.weights.device_addr().unwrap(), i.weights.len_bytes());
    let role_bytes = [
        matrix_bytes(i.p.gate_up, i.p.ne * i.p.nff),
        matrix_bytes(i.p.gate_up, i.p.ne * i.p.nff),
        matrix_bytes(i.p.down, i.p.ne * i.p.nff),
    ];
    let mut protected: Vec<Range<u64>> = [
        &i.input,
        &i.ids,
        &i.tape,
        &i.activation,
        &i.output,
        &i.gate,
        &i.up,
        &i.jobs,
        &i.journal,
    ]
    .into_iter()
    .map(|b| interval(b.device_addr().unwrap(), b.len_bytes()))
    .collect();
    protected.push(interval(
        produced.device_addr().unwrap(),
        produced.len_bytes(),
    ));
    for row in 0..i.rows {
        assert_eq!(i.hits[row] & !all, 0);
        for slot in 0..i.p.used {
            let expert = i.routed[row * i.p.used + slot] as usize;
            assert!(expert < i.p.used);
            if i.hits[row] & (1 << slot) != 0 {
                for role in 0..3 {
                    protected.push(interval(tape[role * i.p.used + expert], role_bytes[role]));
                }
            }
        }
    }
    if let Some(shared) = &f.shared {
        protected.extend(
            shared
                .matrices
                .iter()
                .map(|b| interval(b.device_addr().unwrap(), b.len_bytes())),
        );
    }
    let mut destinations = Vec::new();
    let mut sources = Vec::new();
    for job in 0..i.job_count {
        let w = &i.job_words[(ORDINAL * CAPACITY + job) * 8..][..8];
        let block = w[5] as usize;
        assert!(block < tape.len());
        let src = w[0] as u64 | ((w[1] as u64) << 32);
        let dst = w[2] as u64 | ((w[3] as u64) << 32);
        let bytes = w[4] as usize;
        assert_eq!((src | dst | bytes as u64) & 15, 0);
        assert_eq!(dst, tape[block]);
        assert_eq!(bytes, role_bytes[block / i.p.used]);
        let target = interval(dst, bytes);
        assert!(target.start >= arena.start && target.end <= arena.end);
        assert!(
            protected.iter().all(|p| !overlaps(&target, p)),
            "copy aliases H/metadata/scratch"
        );
        assert!(
            destinations.iter().all(|p| !overlaps(&target, p)),
            "copy WAW"
        );
        destinations.push(target);
        sources.push(interval(src, bytes));
    }
    assert!(
        sources
            .iter()
            .all(|s| destinations.iter().all(|d| !overlaps(s, d))),
        "copy RAW/WAR"
    );
}

/// Records H GU -> Set -> fullcopy -> Wait -> native H Down -> fulljoin.
/// Caller records M projections/activation/Down AFTER this returns. The entire
/// original job list is copied once, including all three roles; never role-split.
/// Retain resources and fixture buffers through all submissions/replays.
pub(super) fn record_complete_hit(
    rec: &Recorder<'_>,
    f: &FusedFixture,
    resources: &HitEventResources,
) {
    let profile = resources
        .profile
        .expect("compact producer requires external shader constructor");
    assert_eq!(
        (
            f.inner.p.ne,
            f.inner.p.nff,
            f.inner.p.used,
            f.inner.p.gate_up,
            f.inner.p.down
        ),
        (
            profile.ne,
            profile.nff,
            profile.used,
            profile.gate_up,
            profile.down
        )
    );
    assert_eq!(f.shared.is_some(), resources.shared_q8);
    record_producer_copy_consumer(
        rec,
        f,
        resources,
        f.inner.activation.as_ref(),
        |rec| resources.produce(rec, f),
        |rec| f.down(rec, 1, 0),
    );
}

/// Generic complete-H path; no new shaders are required. Example matrix arms:
///
/// record_producer_copy_consumer(rec, &f.ff, event, f.ff.inner.activation.as_ref(),
///     |r| { f.ff.projections(r, 1, 0); f.ff.activation(r); },
///     |r| f.ff.down(r, 1, 0));
/// record_producer_copy_consumer(rec, &f.ff, event, f.partial.as_ref(),
///     |r| f.partial(r, split_kernels), |r| f.reduce(r, split_kernels));
///
/// Producer is normal recorder code and may contain internal barriers. Consumer
/// MUST be one direct compute dispatch reading produced + immutable H weights,
/// IDs/tape; it must not access M weights. Direct dispatch rules out consumer
/// internal dependencies hidden by suppression. All jobs/metadata are immutable.
/// The caller retains imported Sources through completion, as for Fixture::cu_copy.
/// M FFN is recorded by the caller after return. Frame-level timestamps may wrap
/// this call; per-dispatch timestamps are rejected to keep the overlap observable.
pub(super) fn record_producer_copy_consumer<P, C>(
    rec: &Recorder<'_>,
    f: &FusedFixture,
    resources: &HitEventResources,
    produced: &dyn Buffer,
    producer: P,
    consumer: C,
) where
    P: FnOnce(&Recorder<'_>),
    C: FnOnce(&Recorder<'_>),
{
    check_recorder(rec);
    assert!(Arc::ptr_eq(&resources.shared, &rec.be.shared));
    prove_disjoint(f, produced);
    resources.reset_before_producer(rec);
    let before_producer = rec.dispatches.get();
    producer(rec);
    assert!(
        rec.dispatches.get() > before_producer,
        "producer must dispatch"
    );
    check_recorder(rec);
    let dependency = HitEventResources::producer_dependency(produced);
    resources.signal(rec, &dependency);
    rec.suppress_sync(true);
    let scope = ActivationConsumerScope(rec);
    let barriers = *rec.barriers.borrow();
    let dispatches = rec.dispatches.get();
    let copy_schedule = f.inner.job_count != 0 || resources.runtime_control;
    if copy_schedule {
        f.inner.cu_copy(rec);
    }
    resources.wait(rec, &dependency);
    consumer(rec);
    assert_eq!(
        *rec.barriers.borrow(),
        barriers + 1,
        "only event wait allowed between signal and H consumer; no generic copy join"
    );
    assert_eq!(
        rec.dispatches.get() - dispatches,
        1 + usize::from(copy_schedule),
        "suppression proof requires ONE copy dispatch and ONE consumer dispatch"
    );
    drop(scope); // Mandatory full copy/H join; no dirty-state retirement.
}

/// Matched conservative control / fallback. Same compact GU and selected native
/// Down; the entire copy is fully joined before GU. No events or reset overhead.
pub(super) fn record_complete_hit_serial(
    rec: &Recorder<'_>,
    f: &FusedFixture,
    resources: &HitEventResources,
) {
    record_producer_copy_consumer_serial(
        rec,
        f,
        f.inner.activation.as_ref(),
        |rec| resources.produce(rec, f),
        |rec| f.down(rec, 1, 0),
    );
}

/// Event-free fallback using exactly the same closures. All ordinary recorder
/// synchronization stays enabled. Unlike the event arm, copy completes first.
pub(super) fn record_producer_copy_consumer_serial<P, C>(
    rec: &Recorder<'_>,
    f: &FusedFixture,
    produced: &dyn Buffer,
    producer: P,
    consumer: C,
) where
    P: FnOnce(&Recorder<'_>),
    C: FnOnce(&Recorder<'_>),
{
    check_recorder(rec);
    prove_disjoint(f, produced);
    rec.seed_barrier();
    if f.inner.job_count != 0 {
        f.inner.cu_copy(rec);
    }
    rec.global_barrier();
    producer(rec);
    rec.global_barrier();
    consumer(rec);
    rec.global_barrier();
}
