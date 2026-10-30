//! Optional owned role-copy/GU schedule; owner and host-upload policy are unchanged.
use super::*;
use infr_core::DType;

const MAX_NFF: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Settings {
    pub local: u32,
    pub nr: u32,
}

pub(crate) const DEFAULT_SETTINGS: Settings = Settings { local: 256, nr: 2 };

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub ne: usize,
    pub nff: usize,
    pub rows: usize,
    pub used: usize,
    pub experts: usize,
    pub physical_used: usize,
    pub tape_bases: [u32; 3],
    pub block_bases: [u32; 3],
}

pub(crate) struct Request<'a> {
    pub copy: &'a Arc<crate::pager::GpuCopyBuffers>,
    pub journal: &'a Arc<dyn Buffer>,
    pub control: &'a dyn Buffer,
    pub ids: &'a dyn Buffer,
    pub tape: &'a dyn Buffer,
    pub input: &'a dyn Buffer,
    pub activation: &'a dyn Buffer,
    /// H Gate/Up outputs and H Down output; all are retained by the existing scratch pool.
    pub hit_scratch: [&'a dyn Buffer; 3],
    pub geometry: Geometry,
    pub dtypes: [DType; 2],
    pub ordinal: u32,
    pub ordinary: bool,
    pub preparing: bool,
}

pub(crate) struct Plan<'a> {
    copy: &'a crate::pager::GpuCopyBuffers,
    dispatch: Dispatch<'a>,
    hit_predicate: (vk::Buffer, usize, usize),
    no_entry_ranges: bool,
    phase1: std::cell::Cell<Option<Phase1Proof>>,
}

#[derive(Clone, Copy)]
struct Phase1Proof {
    recorder: usize,
    command: vk::CommandBuffer,
    epoch: u64,
    dispatches: usize,
    barriers: usize,
    remaining: usize,
    conditional_mutable: bool,
}

fn suffix_matches(
    dispatches: usize,
    barriers: usize,
    remaining: usize,
    actual: (usize, usize),
) -> bool {
    let joins = match remaining {
        4 => 3,
        2 => 2,
        _ => return false,
    };
    dispatches.checked_add(remaining) == Some(actual.0)
        && barriers.checked_add(joins) == Some(actual.1)
}

impl Phase1Proof {
    fn capture(rec: &Recorder<'_>, remaining: usize) -> Self {
        Self {
            recorder: rec as *const Recorder<'_> as usize,
            command: rec.cmd,
            epoch: rec
                .be
                .shared
                .resource_epoch
                .load(std::sync::atomic::Ordering::Acquire),
            dispatches: rec.dispatches(),
            barriers: *rec.barriers.borrow(),
            remaining,
            conditional_mutable: rec.conditional_mutable.get(),
        }
    }

    fn ready(self, rec: &Recorder<'_>) -> bool {
        recording_supported(rec)
            && !rec.dirty_transfer.get()
            && rec.indirect_region.get().is_none()
            && self.conditional_mutable == rec.conditional_mutable.get()
            && self.recorder == rec as *const Recorder<'_> as usize
            && self.command == rec.cmd
            && self.epoch == rec.be.shared.resource_epoch.load(std::sync::atomic::Ordering::Acquire)
            // After production phase1 copy: G, U, ACT, Down and three barriers
            // (phase1 exit, G/U -> ACT, ACT -> Down). Extra helpers fail closed.
            && suffix_matches(self.dispatches, self.barriers, self.remaining,
                (rec.dispatches(), *rec.barriers.borrow()))
    }
}

// Only minted after the complete fixed H suffix has passed the checks above. There
// is no recorder-wide no-entry API. Keep the old dirty sets: they still describe H
// Down reads/Y writes, and the ordinary unconditional scope exit joins ALL of them.
struct Phase2Proof(Phase1Proof);

fn no_entry_scope<'r, 'a>(
    rec: &'r Recorder<'a>,
    proof: Phase2Proof,
) -> IndependentDispatchScope<'r, 'a> {
    assert!(proof.0.ready(rec));
    *rec.independent_dispatch.borrow_mut() = Some(IndependentDispatchBatch::default());
    IndependentDispatchScope { recorder: rec }
}

#[cfg(test)]
pub(crate) struct TestPhase1Join(Phase1Proof);

/// Call immediately after the phase1 copy/G/U scope joins. The module fixture
/// supplies the H/M mask and BDA non-alias proof; then record only H ACT and Down.
#[cfg(test)]
pub(crate) fn test_phase1_joined(rec: &Recorder<'_>) -> TestPhase1Join {
    assert!(recording_supported(rec) && !rec.dirty_transfer.get());
    TestPhase1Join(Phase1Proof::capture(rec, 2))
}

/// Same callback in both arms; only the phase2 entry differs. Always joins on
/// return/unwind, never clears old hazards, and does not manipulate predicates.
#[cfg(test)]
pub(crate) fn test_phase2_scope<'a>(
    rec: &Recorder<'a>,
    proof: TestPhase1Join,
    request_no_entry: bool,
    record: impl FnOnce(&Recorder<'a>),
) -> bool {
    let optimized = request_no_entry && proof.0.ready(rec);
    let scope = if optimized {
        no_entry_scope(rec, Phase2Proof(proof.0))
    } else {
        rec.independent_dispatch_scope()
    };
    record(rec);
    drop(scope);
    optimized
}

pub(super) struct Dispatch<'a> {
    journal: &'a dyn Buffer,
    jobs: &'a dyn Buffer,
    predicate: &'a dyn Buffer,
    geometry: Geometry,
    settings: Settings,
    ordinal: u32,
    copy_kernel: ComputeKernel,
    gu_kernel: ComputeKernel,
    gu_buffers: [vk::DescriptorBufferInfo; 4],
}

fn settings(be: &VulkanBackend) -> Option<Settings> {
    #[cfg(test)]
    if let Some(override_settings) = test_states()
        .lock()
        .unwrap()
        .get(&(Arc::as_ptr(&be.shared) as usize))
        .copied()
    {
        return scoped_settings(Some(override_settings));
    }
    be.cfg()
        .kernels
        .vulkan
        .decode_gpu_resident_role_pipeline
        .then_some(DEFAULT_SETTINGS)
}

#[inline]
pub(crate) fn enabled(be: &VulkanBackend) -> bool {
    settings(be).is_some()
}

#[cfg(test)]
fn scoped_settings(override_settings: Option<Option<Settings>>) -> Option<Settings> {
    override_settings.flatten()
}

fn route_eligible(
    ordinary: bool,
    preparing: bool,
    owned: bool,
    active: bool,
    in_forward: bool,
    same_dtype: bool,
) -> bool {
    ordinary && owned && same_dtype && (preparing || (active && in_forward))
}

fn geometry_valid(g: Geometry) -> bool {
    let fits = |base: u32| base.checked_add(g.experts as u32).is_some();
    let separate = |a: u32, b: u32| {
        a.checked_add(g.experts as u32).is_some_and(|end| end <= b)
            || b.checked_add(g.experts as u32).is_some_and(|end| end <= a)
    };
    (1..=8).contains(&g.rows)
        && (1..=16).contains(&g.used)
        && (1..=65536).contains(&g.ne)
        && g.ne.is_multiple_of(256)
        && (1..=MAX_NFF).contains(&g.nff)
        && g.nff.is_multiple_of(32)
        && g.experts > 0
        && g.experts <= u32::MAX as usize
        && (g.used..=g.used + 1).contains(&g.physical_used)
        && g.tape_bases.into_iter().all(fits)
        && g.block_bases.into_iter().all(fits)
        // Gate and Up may intentionally alias. Down must never enter phase1.
        && separate(g.block_bases[0], g.block_bases[2])
        && separate(g.block_bases[1], g.block_bases[2])
}

fn sidecars_fit(ordinal: u32, journal: usize, jobs: usize, predicate: usize) -> bool {
    let Some(layers) = (ordinal as usize).checked_add(1) else {
        return false;
    };
    layers <= RESIDENT_MAX_LAYERS
        && layers <= journal / RESIDENT_JOURNAL_BYTES
        && layers <= jobs / (RESIDENT_JOURNAL_ID_WORDS * 3 * 32)
        && layers <= predicate / (GPU_COPY_PREDICATE_WORDS * 4)
}

fn disjoint(a: vk::DescriptorBufferInfo, b: vk::DescriptorBufferInfo) -> bool {
    a.buffer != b.buffer
        || a.offset
            .checked_add(a.range)
            .is_some_and(|end| end <= b.offset)
        || b.offset
            .checked_add(b.range)
            .is_some_and(|end| end <= a.offset)
}

fn recording_supported(rec: &Recorder<'_>) -> bool {
    rec.be.shared.conditional_rendering.is_some()
        && rec.resident_indirect.borrow().is_none()
        && !rec.conditional_active.get()
        && rec.independent_dispatch.borrow().is_none()
        && !rec.suppress.get()
        && !rec.indirect_pending.get()
        && !rec.no_barrier
        && !rec.full_barrier
        && !rec.stages
}

fn device_supported(be: &VulkanBackend, wide: bool) -> bool {
    type Entry = (std::sync::Weak<crate::VulkanShared>, [bool; 2]);
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Vec<Entry>>> = std::sync::OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache.retain(|(backend, _)| backend.strong_count() != 0);
    if let Some((_, supported)) = cache
        .iter()
        .find(|(backend, _)| backend.ptr_eq(&Arc::downgrade(&be.shared)))
    {
        return supported[usize::from(wide)];
    }
    let operations = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC;
    let mut features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features);
    let mut subgroup = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    unsafe {
        be.shared
            .instance
            .get_physical_device_features2(be.shared.physical_device, &mut features2);
        be.shared
            .instance
            .get_physical_device_properties2(be.shared.physical_device, &mut properties);
    }
    let limits = properties.properties.limits;
    let common = be.shared.compute_subgroup_operations.contains(operations)
        && features.subgroup_size_control != 0
        && features.compute_full_subgroups != 0
        && subgroup.min_subgroup_size <= 32
        && subgroup.max_subgroup_size >= 32
        && subgroup
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
        && subgroup.max_compute_workgroup_subgroups >= 256 / 32
        && limits.max_compute_work_group_count[0] >= 32768
        && limits.max_compute_work_group_count[1] >= 2
        && limits.max_compute_shared_memory_size >= 32 * 1024
        && limits.max_push_constants_size >= 40;
    let supported = [256, 1024].map(|copy_local| {
        common
            && limits.max_compute_work_group_invocations >= copy_local
            && limits.max_compute_work_group_size[0] >= copy_local
    });
    cache.push((Arc::downgrade(&be.shared), supported));
    supported[usize::from(wide)]
}

fn kernel_name(dtype: DType, settings: Settings) -> Option<&'static str> {
    macro_rules! names {
        ($fmt:literal) => {
            match (settings.local, settings.nr) {
                (128, 1) => concat!("owned_role_gu_tree64_", $fmt, "_128_nr1"),
                (128, 2) => concat!("owned_role_gu_tree64_", $fmt, "_128_nr2"),
                (256, 1) => concat!("owned_role_gu_tree64_", $fmt, "_256_nr1"),
                (256, 2) => concat!("owned_role_gu_tree64_", $fmt, "_256_nr2"),
                _ => return None,
            }
        };
    }
    Some(match dtype {
        DType::Q4K => names!("q4k"),
        DType::Q5K => names!("q5k"),
        DType::Q6K => names!("q6k"),
        DType::Iq2S => names!("iq2s"),
        DType::Iq3S => names!("iq3s"),
        _ => return None,
    })
}

fn native_tree_compatible(
    dtype: DType,
    ne: usize,
    nff: usize,
    gemv: &infr_core::config::GemvCfg,
) -> bool {
    // Tree64 preserves the native tree, not an enrolled SG32 reduction ordering.
    native_id_sg_choice(dtype, ne, nff, gemv).is_none()
}

#[allow(clippy::too_many_arguments)]
fn make_dispatch<'a>(
    rec: &Recorder<'_>,
    journal: &'a dyn Buffer,
    jobs: &'a dyn Buffer,
    predicate: &'a dyn Buffer,
    ids: &dyn Buffer,
    tape: &dyn Buffer,
    input: &dyn Buffer,
    activation: &dyn Buffer,
    geometry: Geometry,
    ordinal: u32,
    dtype: DType,
    wide: bool,
    groups: u32,
    settings: Settings,
) -> Option<Dispatch<'a>> {
    let name = kernel_name(dtype, settings)?;
    if !recording_supported(rec)
        || !geometry_valid(geometry)
        || !native_tree_compatible(dtype, geometry.ne, geometry.nff, rec.gemv())
        || !sidecars_fit(
            ordinal,
            journal.len_bytes(),
            jobs.len_bytes(),
            predicate.len_bytes(),
        )
        || !(1..=8).contains(&groups)
        || !device_supported(rec.be, wide)
        || ids.len_bytes() < (geometry.rows * geometry.used + 2 * geometry.rows) * 4
        || input.len_bytes() < geometry.rows * geometry.ne * 4
        || activation.len_bytes() < geometry.rows * geometry.physical_used * geometry.nff * 4
        || geometry
            .tape_bases
            .into_iter()
            .any(|base| (base as usize + geometry.experts) * 8 > tape.len_bytes())
    {
        return None;
    }
    let buffers = [journal, jobs, predicate, ids, tape, input, activation];
    if !buffers.into_iter().all(|buffer| {
        as_vk_buf(buffer).is_ok_and(|buffer| Arc::ptr_eq(&buffer.shared, &rec.be.shared))
    }) {
        return None;
    }
    let ranges = buffers.map(Recorder::vkb_exact);
    // All BDA destinations belong to frozen owner weight slots, not these scratch/sidecars.
    // Also reject descriptor aliasing before recording even the first candidate command.
    // Unchanged admission protects H and assigns distinct live slots to distinct blocks;
    // disjoint canonical Down/GU ranges therefore make phase2 copy and M GU BDA-disjoint.
    if !(0..ranges.len()).all(|i| (i + 1..ranges.len()).all(|j| disjoint(ranges[i], ranges[j]))) {
        return None;
    }
    let spv = crate::gemm::gpu_owned_miss_gu_tree64_spv(dtype, settings.local, settings.nr)?;
    Some(Dispatch {
        journal,
        jobs,
        predicate,
        geometry,
        settings,
        ordinal,
        copy_kernel: rec.be.kernel(
            if wide {
                "owned_role_copy1024"
            } else {
                "owned_role_copy256"
            },
            crate::gemm::gpu_owned_role_copy_spv(wide),
            2,
            20,
        ),
        gu_kernel: rec.be.kernel_sg(name, spv, 4, 40, 32),
        gu_buffers: [ranges[5], ranges[3], ranges[4], ranges[6]],
    })
}

impl Recorder<'_> {
    /// Prepared templates capture the same buffers as the old M FFN. Existing ACK-time
    /// activate_prepared/begin_owned validates the live generation before any submission;
    /// encoding this schedule neither reads owner state nor starts a new epoch.
    pub(crate) fn try_owned_role_pipeline<'b>(&self, request: Request<'b>) -> Option<Plan<'b>> {
        let r = request;
        let settings = settings(self.be)?;
        if !route_eligible(
            r.ordinary,
            r.preparing,
            r.copy.is_owned(),
            r.copy.owned_active(),
            r.copy.owned_in_forward(),
            r.dtypes[0] == r.dtypes[1],
        ) || r.control.len_bytes() != RESIDENT_CONTROL_BYTES
            || r.copy
                .owned_role_geometry(r.geometry.block_bases, r.geometry.experts as u32)
                .is_none()
        {
            return None;
        }
        if !as_vk_buf(r.control).is_ok_and(|buffer| Arc::ptr_eq(&buffer.shared, &self.be.shared)) {
            return None;
        }
        let predicate = r.copy.miss_predicates.as_ref()?;
        let dispatch = make_dispatch(
            self,
            r.journal.as_ref(),
            r.copy.jobs.as_ref(),
            predicate.as_ref(),
            r.ids,
            r.tape,
            r.input,
            r.activation,
            r.geometry,
            r.ordinal,
            r.dtypes[0],
            r.copy.wide_copy,
            r.copy.copy_groups,
            settings,
        )?;
        let predicate_range = Self::vkb_exact(predicate.as_ref());
        if [
            r.copy.sources.as_ref(),
            r.copy.candidates.as_ref(),
            r.copy.overlay.as_ref(),
            r.control,
        ]
        .into_iter()
        .any(|buffer| !disjoint(predicate_range, Self::vkb_exact(buffer)))
        {
            return None;
        }
        for buffer in [
            r.journal.clone(),
            predicate.clone(),
            r.copy.jobs.clone(),
            r.copy.sources.clone(),
            r.copy.candidates.clone(),
            r.copy.overlay.clone(),
        ] {
            self.retain_buffer(buffer);
        }
        let control = Self::vk_of(r.control);
        let scratch = r.hit_scratch.map(Self::vkb_exact);
        let ranges = [
            dispatch.gu_buffers[0],
            dispatch.gu_buffers[1],
            dispatch.gu_buffers[2],
            dispatch.gu_buffers[3],
            Self::vkb_exact(r.journal.as_ref()),
            Self::vkb_exact(r.copy.jobs.as_ref()),
            Self::vkb_exact(predicate.as_ref()),
            Self::vkb_exact(r.control),
            Self::vkb_exact(r.copy.sources.as_ref()),
            Self::vkb_exact(r.copy.candidates.as_ref()),
            Self::vkb_exact(r.copy.overlay.as_ref()),
        ];
        let no_entry_ranges = r.hit_scratch.into_iter().all(|buffer| {
            as_vk_buf(buffer).is_ok_and(|buffer| Arc::ptr_eq(&buffer.shared, &self.be.shared))
        }) && r.hit_scratch[0].len_bytes()
            >= r.geometry.rows * r.geometry.physical_used * r.geometry.nff * 4
            && r.hit_scratch[1].len_bytes()
                >= r.geometry.rows * r.geometry.physical_used * r.geometry.nff * 4
            && r.hit_scratch[2].len_bytes()
                >= r.geometry.rows * r.geometry.physical_used * r.geometry.ne * 4
            && scratch
                .iter()
                .all(|&range| ranges.iter().all(|&other| disjoint(range, other)))
            && (0..3).all(|i| (i + 1..3).all(|j| disjoint(scratch[i], scratch[j])));
        #[cfg(test)]
        test_note_selection(self.be, r.preparing);
        Some(Plan {
            copy: r.copy.as_ref(),
            dispatch,
            hit_predicate: (
                control.buffer,
                control.sub_offset,
                6 + r.ordinal as usize * 2,
            ),
            no_entry_ranges,
            phase1: std::cell::Cell::new(None),
        })
    }
}

// This token cannot escape the specialized phase2 method. Its import covers every later
// conditional read of exactly this W0, until the existing stage handoff/replay retirement.
// Payload writes frozen weight slots, GU writes ACT, publish writes owner/overlay: none can
// overwrite W0. A same-ordinal prepare, host update, submission or epoch change is forbidden
// between import and the last resume; this module never calls any of them.
struct ImportedMiss<'a> {
    predicate: &'a dyn Buffer,
    word: usize,
}

impl ImportedMiss<'_> {
    fn resume(&self, rec: &Recorder<'_>) {
        assert!(!rec.conditional_active.get() && rec.immutable_condition.get().is_none());
        assert!(rec.resident_indirect.borrow().is_none() && rec.indirect_region.get().is_none());
        let buffer = Recorder::vk_of(self.predicate);
        rec.conditional_mutable.set(false);
        rec.immutable_condition
            .set(Some((buffer.buffer, buffer.sub_offset, self.word)));
        unsafe {
            (rec.be
                .shared
                .conditional_rendering
                .as_ref()
                .unwrap()
                .fp()
                .cmd_begin_conditional_rendering_ext)(
                rec.cmd,
                &vk::ConditionalRenderingBeginInfoEXT::default()
                    .buffer(buffer.buffer)
                    .offset((buffer.sub_offset + self.word * 4) as u64),
            );
        }
        rec.conditional_active.set(true);
    }
}

impl Dispatch<'_> {
    fn copy_phase(&self, rec: &Recorder<'_>, phase: u32) {
        assert!(matches!(phase, 1 | 2));
        assert!(rec.independent_dispatch.borrow().is_some());
        let g = self.geometry;
        let args = Recorder::vk_of(self.predicate);
        rec.dispatch_indirect(
            self.copy_kernel,
            &[
                Recorder::vkb_exact(self.journal),
                Recorder::vkb_exact(self.jobs),
            ],
            0,
            bytemuck::cast_slice(&[
                self.ordinal,
                phase,
                g.block_bases[0],
                g.block_bases[1],
                g.experts as u32,
            ]),
            args.buffer,
            (args.sub_offset + (self.ordinal as usize * GPU_COPY_PREDICATE_WORDS + 1) * 4) as u64,
        );
    }

    #[cfg(test)]
    fn phase2(&self, rec: &Recorder<'_>) -> ImportedMiss<'_> {
        self.phase2_proven(rec, None)
    }

    fn phase2_proven(&self, rec: &Recorder<'_>, proof: Option<Phase2Proof>) -> ImportedMiss<'_> {
        assert!(recording_supported(rec));
        let imported = ImportedMiss {
            predicate: self.predicate,
            word: self.ordinal as usize * GPU_COPY_PREDICATE_WORDS,
        };
        // Import BEFORE launching Down copy; no compute->conditional edge may join that copy.
        let optimized = proof.is_some();
        let independent = if let Some(proof) = proof {
            // Initial H begin used a GLOBAL SHADER_WRITE -> CONDITIONAL_READ barrier
            // after this ordinal's prepare. Immutable W0 is already imported. The
            // phase1 entry/exit and existing args barrier retain BDA/HOST/indirect visibility.
            no_entry_scope(rec, proof)
        } else {
            rec.begin_immutable_conditional_at(imported.predicate, imported.word);
            rec.end_conditional();
            rec.independent_dispatch_scope()
        };
        #[cfg(test)]
        test_note_phase2(rec.be, optimized);
        #[cfg(not(test))]
        let _ = optimized;
        self.copy_phase(rec, 2);
        imported.resume(rec);
        let g = self.geometry;
        let units = (g.rows
            * g.used
            * g.nff
                .div_ceil((self.settings.local / 64 * self.settings.nr) as usize))
            as u32;
        rec.dispatch3(
            self.gu_kernel,
            &self.gu_buffers,
            1,
            bytemuck::cast_slice(&[
                g.ne as u32,
                g.nff as u32,
                g.rows as u32,
                g.used as u32,
                g.experts as u32,
                g.physical_used as u32,
                g.tape_bases[0],
                g.tape_bases[1],
                g.tape_bases[2],
                1,
            ]),
            units.min(32768),
            units.div_ceil(32768),
            1,
        );
        rec.end_conditional();
        drop(independent);
        imported
    }
}

impl Plan<'_> {
    pub(crate) fn phase1(&self, rec: &Recorder<'_>) {
        assert!(rec.conditional_active.get() && !rec.conditional_mutable.get());
        assert_eq!(rec.immutable_condition.get(), Some(self.hit_predicate));
        self.dispatch.copy_phase(rec, 1);
        // This is the first dispatch in the phase1 independent scope. H begin has
        // just globally imported prepare's H and W0 writes. Both remain immutable
        // through H GU/ACT/Down, phase2, publish and final M Down. Masked H Down
        // reads ACT_H only; fused GU writes ACT_M only (M excludes the shared slot).
        let batch = rec.independent_dispatch.borrow();
        if self.no_entry_ranges
            && !rec.dirty_transfer.get()
            && batch
                .as_ref()
                .is_some_and(|b| !b.closed && b.accesses.len() == 3)
        {
            self.phase1.set(Some(Phase1Proof::capture(rec, 4)));
        } else {
            self.phase1.set(None);
        }
    }

    /// Leaves only M Down conditional active. The single READY publication remains after
    /// both role copies and GU have unconditionally joined, including the status5 path.
    pub(crate) fn phase2_finish_and_begin_miss(self, rec: &Recorder<'_>) {
        let proof = self
            .phase1
            .take()
            .filter(|proof| proof.ready(rec))
            .map(Phase2Proof);
        let imported = self.dispatch.phase2_proven(rec, proof);
        rec.gpu_resident_copy_finish_overlap(
            self.copy,
            self.dispatch.journal,
            self.dispatch.ordinal,
        );
        imported.resume(rec);
    }
}

#[cfg(test)]
fn test_selection_states() -> &'static std::sync::Mutex<std::collections::HashMap<usize, [usize; 4]>>
{
    static STATES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, [usize; 4]>>,
    > = std::sync::OnceLock::new();
    STATES.get_or_init(Default::default)
}

#[cfg(test)]
fn test_note_phase2(be: &VulkanBackend, optimized: bool) {
    if let Some(counts) = test_selection_states()
        .lock()
        .unwrap()
        .get_mut(&(Arc::as_ptr(&be.shared) as usize))
    {
        counts[2 + usize::from(optimized)] += 1;
    }
}

#[cfg(test)]
fn test_note_selection(be: &VulkanBackend, preparing: bool) {
    if let Some(counts) = test_selection_states()
        .lock()
        .unwrap()
        .get_mut(&(Arc::as_ptr(&be.shared) as usize))
    {
        counts[usize::from(preparing)] += 1;
    }
}

#[cfg(test)]
pub(crate) struct TestSelectionGuard {
    key: usize,
    // Preparation may run on another thread; keep the backend identity unique until drop.
    _backend: Arc<crate::VulkanShared>,
}

#[cfg(test)]
impl TestSelectionGuard {
    /// Successful real-adapter Plan selections: [ordinary recording, prepared recording].
    pub(crate) fn counts(&self) -> [usize; 2] {
        let counts = test_selection_states().lock().unwrap()[&self.key];
        [counts[0], counts[1]]
    }

    /// Recorded phase2 schedules: [conservative import+entry, proven no-entry].
    pub(crate) fn phase2_counts(&self) -> [usize; 2] {
        let counts = test_selection_states().lock().unwrap()[&self.key];
        [counts[2], counts[3]]
    }
}

#[cfg(test)]
impl Drop for TestSelectionGuard {
    fn drop(&mut self) {
        test_selection_states().lock().unwrap().remove(&self.key);
    }
}

#[cfg(test)]
pub(crate) fn test_selection_scope(be: &VulkanBackend) -> TestSelectionGuard {
    let key = Arc::as_ptr(&be.shared) as usize;
    let mut states = test_selection_states().lock().unwrap();
    assert!(
        !states.contains_key(&key),
        "one selection observer per backend"
    );
    states.insert(key, [0; 4]);
    TestSelectionGuard {
        key,
        _backend: Arc::clone(&be.shared),
    }
}

#[cfg(test)]
fn test_states() -> &'static std::sync::Mutex<std::collections::HashMap<usize, Option<Settings>>> {
    static STATES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, Option<Settings>>>,
    > = std::sync::OnceLock::new();
    STATES.get_or_init(Default::default)
}

#[cfg(test)]
pub(crate) struct TestGuard {
    key: usize,
    previous: Option<Option<Settings>>,
}

#[cfg(test)]
impl Drop for TestGuard {
    fn drop(&mut self) {
        let mut states = test_states().lock().unwrap();
        states.remove(&self.key);
        if let Some(previous) = self.previous {
            states.insert(self.key, previous);
        }
    }
}

#[cfg(test)]
// The scope must enclose preparation AND replay. Cached recordings capture this choice;
// A/B callers use separate backends/plans rather than toggling around a cached prefix.
pub(crate) fn test_scope(be: &VulkanBackend, enable: bool) -> TestGuard {
    test_variant_scope(be, enable.then_some(DEFAULT_SETTINGS))
}

#[cfg(test)]
pub(crate) fn test_variant_scope(be: &VulkanBackend, settings: Option<Settings>) -> TestGuard {
    let key = Arc::as_ptr(&be.shared) as usize;
    let previous = test_states().lock().unwrap().insert(key, settings);
    TestGuard { key, previous }
}

#[cfg(test)]
pub(super) mod contract {
    pub(in crate::recorder) fn phase2_suffix_matches(
        dispatches: usize,
        barriers: usize,
        remaining: usize,
        actual: (usize, usize),
    ) -> bool {
        super::suffix_matches(dispatches, barriers, remaining, actual)
    }

    pub(in crate::recorder) fn kernel_name(
        dtype: infr_core::DType,
        settings: super::Settings,
    ) -> Option<&'static str> {
        super::kernel_name(dtype, settings)
    }

    pub(in crate::recorder) fn native_tree_compatible(
        dtype: infr_core::DType,
        ne: usize,
        nff: usize,
        gemv: &infr_core::config::GemvCfg,
    ) -> bool {
        super::native_tree_compatible(dtype, ne, nff, gemv)
    }

    pub(in crate::recorder) fn policy_enabled(value: Option<bool>) -> bool {
        super::scoped_settings(value.map(|enabled| enabled.then_some(super::DEFAULT_SETTINGS)))
            .is_some()
    }

    pub(in crate::recorder) fn resolved_settings(
        override_settings: Option<Option<super::Settings>>,
    ) -> Option<super::Settings> {
        super::scoped_settings(override_settings)
    }

    pub(in crate::recorder) fn route_eligible(
        ordinary: bool,
        preparing: bool,
        owned: bool,
        active: bool,
        in_forward: bool,
        same_dtype: bool,
    ) -> bool {
        super::route_eligible(ordinary, preparing, owned, active, in_forward, same_dtype)
    }

    pub(in crate::recorder) fn geometry_valid(g: super::Geometry) -> bool {
        super::geometry_valid(g)
    }

    pub(in crate::recorder) fn sidecars_fit(
        ordinal: u32,
        journal: usize,
        jobs: usize,
        predicate: usize,
    ) -> bool {
        super::sidecars_fit(ordinal, journal, jobs, predicate)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::recorder) fn fixture_dispatch<'a>(
        rec: &super::Recorder<'_>,
        journal: &'a dyn super::Buffer,
        jobs: &'a dyn super::Buffer,
        predicate: &'a dyn super::Buffer,
        ids: &dyn super::Buffer,
        tape: &dyn super::Buffer,
        input: &dyn super::Buffer,
        activation: &dyn super::Buffer,
        geometry: super::Geometry,
        ordinal: u32,
        dtype: infr_core::DType,
        wide: bool,
        groups: u32,
        settings: super::Settings,
    ) -> Option<super::Dispatch<'a>> {
        super::make_dispatch(
            rec, journal, jobs, predicate, ids, tape, input, activation, geometry, ordinal, dtype,
            wide, groups, settings,
        )
    }

    pub(in crate::recorder) fn phase1(plan: &super::Dispatch<'_>, rec: &super::Recorder<'_>) {
        plan.copy_phase(rec, 1);
    }

    pub(in crate::recorder) fn phase2_finish(
        plan: &super::Dispatch<'_>,
        rec: &super::Recorder<'_>,
        candidates: &dyn super::Buffer,
        overlay: &dyn super::Buffer,
        before_publish: &dyn super::Buffer,
    ) {
        let imported = plan.phase2(rec);
        rec.copy(candidates, 0, before_publish, 0, candidates.len_bytes());
        let kernel = rec.be.kernel(
            "gpu_owned_publish",
            crate::gemm::gpu_owned_publish_spv(),
            4,
            8,
        );
        rec.dispatch(
            kernel,
            &[
                super::Recorder::vkb_exact(plan.journal),
                super::Recorder::vkb_exact(plan.jobs),
                super::Recorder::vkb_exact(candidates),
                super::Recorder::vkb_exact(overlay),
            ],
            2,
            bytemuck::cast_slice(&[plan.ordinal, 0]),
            1,
        );
        rec.owned_copy_barrier(false, false);
        imported.resume(rec);
    }
}
