//! Default-off H-only trial. The adapter owns route admission, pooled storage and copy joins.
use super::*;
use infr_core::DType;

pub(crate) const PARTIAL_TAG: &str = "moe_owned_hit_splitk";
const CHUNK: usize = 32;
const LOCAL: u32 = 128;

fn shape_matches(dtypes: [DType; 3], ne: usize, nff: usize, rows: usize, used: usize) -> bool {
    dtypes == [DType::Iq2S, DType::Iq2S, DType::Iq4Nl]
        && [ne, nff, rows, used] == [2560, 640, 4, 10]
}

/// Shader bounds, independent of policy. No shared slot is included in the capacity.
pub(crate) fn partial_bytes(ne: usize, nff: usize, rows: usize, used: usize) -> Option<usize> {
    if !(1..=8192).contains(&ne)
        || !ne.is_multiple_of(256)
        || !(1..=1024).contains(&nff)
        || !nff.is_multiple_of(CHUNK)
        || !(1..=8).contains(&rows)
        || !(1..=16).contains(&used)
    {
        return None;
    }
    rows.checked_mul(used)?
        .checked_mul(nff / CHUNK)?
        .checked_mul(ne)?
        .checked_mul(4)
}

#[derive(Clone, Copy)]
struct DeviceLimits {
    alignment: u64,
    range: u64,
}

fn device_limits(be: &VulkanBackend) -> Option<DeviceLimits> {
    type Entry = (std::sync::Weak<crate::VulkanShared>, Option<DeviceLimits>);
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Vec<Entry>>> = std::sync::OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache.retain(|(backend, _)| backend.strong_count() != 0);
    let identity = Arc::downgrade(&be.shared);
    if let Some((_, limits)) = cache.iter().find(|(backend, _)| backend.ptr_eq(&identity)) {
        return *limits;
    }
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
    let operations = vk::SubgroupFeatureFlags::BASIC
        | vk::SubgroupFeatureFlags::ARITHMETIC
        | vk::SubgroupFeatureFlags::SHUFFLE;
    let supported = be.shared.conditional_rendering.is_some()
        && be.shared.compute_subgroup_operations.contains(operations)
        && features.subgroup_size_control != 0
        && features.compute_full_subgroups != 0
        && subgroup.min_subgroup_size <= 32
        && subgroup.max_subgroup_size >= 32
        && subgroup.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
        && subgroup.max_compute_workgroup_subgroups >= LOCAL / 32
        && limits.max_compute_work_group_invocations >= LOCAL
        && limits.max_compute_work_group_size[0] >= LOCAL
        && limits.max_compute_work_group_count[0] >= 800
        // IQ2S grid is 8 KiB; Gate/Up halves and activation add five floats per chunk row.
        && limits.max_compute_shared_memory_size as usize >= (8192 + CHUNK * 5 * 4).max(32 * 129 * 4)
        && limits.max_push_constants_size >= 40
        && limits.max_per_stage_descriptor_storage_buffers >= 4
        && limits.max_descriptor_set_storage_buffers >= 4
        && limits.max_storage_buffer_range as usize >= 8_192_000;
    let result = supported.then_some(DeviceLimits {
        alignment: limits.min_storage_buffer_offset_alignment.max(1),
        range: u64::from(
            limits
                .max_storage_buffer_range
                .min(crate::max_storage_buffer_range()),
        ),
    });
    cache.push((identity, result));
    result
}

pub(crate) fn enabled_for_shape(
    be: &VulkanBackend,
    dtypes: [DType; 3],
    ne: usize,
    nff: usize,
    rows: usize,
    used: usize,
) -> bool {
    be.cfg().kernels.vulkan.decode_gpu_resident_splitk_hit
        && shape_matches(dtypes, ne, nff, rows, used)
        && device_limits(be).is_some()
}

fn window_fits(base: u32, experts: usize, tape_bytes: usize) -> bool {
    (base as usize)
        .checked_add(experts)
        .and_then(|end| end.checked_mul(8))
        .is_some_and(|end| end <= tape_bytes)
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

fn buffer_range(
    be: &VulkanBackend,
    limits: DeviceLimits,
    buffer: &dyn Buffer,
    required: usize,
    element: usize,
) -> Option<vk::DescriptorBufferInfo> {
    let b = as_vk_buf(buffer).ok()?;
    let bytes = buffer.len_bytes();
    let offset = u64::try_from(b.sub_offset).ok()?;
    let range = u64::try_from(bytes).ok()?;
    if !Arc::ptr_eq(&b.shared, &be.shared)
        || b.buffer == vk::Buffer::null()
        || bytes < required
        || bytes == 0
        || !bytes.is_multiple_of(element)
        || range
            > limits
                .range
                .min(u64::from(crate::max_storage_buffer_range()))
        || !offset.is_multiple_of(limits.alignment)
        || offset.checked_add(range).is_none()
    {
        return None;
    }
    Some(
        vk::DescriptorBufferInfo::default()
            .buffer(b.buffer)
            .offset(offset)
            .range(range),
    )
}

pub(crate) struct Plan<'a> {
    buffers: [&'a dyn Buffer; 5],
    views: [vk::DescriptorBufferInfo; 5],
    partial_kernel: ComputeKernel,
    reduce_kernel: ComputeKernel,
    push: [u32; 10],
    partial_groups: u32,
    reduce_groups: u32,
    command: vk::CommandBuffer,
}

/// Call before entering the H conditional or independent scope. All validation precedes factories.
/// The caller checks enabled_for_shape with actual weight dtypes, proves ordinary independent
/// Decode, F32 tensors, no shared slot or unsupported activation/scales/recovery, and pre-reserves
/// PARTIAL_TAG before freezing owner storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_plan<'a>(
    rec: &Recorder<'_>,
    input: &'a dyn Buffer,
    ids: &'a dyn Buffer,
    tape: &'a dyn Buffer,
    partial: &'a dyn Buffer,
    output: &'a dyn Buffer,
    dimensions: [usize; 5],
    windows: [u32; 3],
) -> Option<Plan<'a>> {
    let [ne, nff, rows, used, experts] = dimensions;
    if !enabled_for_shape(
        rec.be,
        [DType::Iq2S, DType::Iq2S, DType::Iq4Nl],
        ne,
        nff,
        rows,
        used,
    ) || !(used..=512).contains(&experts)
        || rec.conditional_active.get()
        || rec.resident_indirect.borrow().is_some()
        || rec.indirect_region.get().is_some()
        || rec.independent_dispatch.borrow().is_some()
        || rec.suppress.get()
        || rec.indirect_pending.get()
        || rec.no_barrier
        || rec.full_barrier
        || rec.stages
    {
        return None;
    }
    let bytes = partial_bytes(ne, nff, rows, used)?;
    let limits = device_limits(rec.be)?;
    let buffers = [input, ids, tape, partial, output];
    let mut views = [
        buffer_range(rec.be, limits, input, rows * ne * 4, 4)?,
        buffer_range(rec.be, limits, ids, (rows * used + 2 * rows) * 4, 4)?,
        buffer_range(rec.be, limits, tape, 8, 8)?,
        buffer_range(rec.be, limits, partial, bytes, 4)?,
        buffer_range(rec.be, limits, output, rows * used * ne * 4, 4)?,
    ];
    if windows
        .into_iter()
        .any(|base| !window_fits(base, experts, tape.len_bytes()))
        || !(0..views.len()).all(|i| (i + 1..views.len()).all(|j| disjoint(views[i], views[j])))
    {
        return None;
    }
    // A pooled high-water capacity must not enlarge the shader's logical partial directory.
    views[3].range = u64::try_from(bytes).ok()?;
    let parts = nff / CHUNK;
    let partial_groups = u32::try_from(rows.checked_mul(used)?.checked_mul(parts)?).ok()?;
    let reduce_groups = u32::try_from(
        rows.checked_mul(used)?
            .checked_mul(ne)?
            .div_ceil(LOCAL as usize),
    )
    .ok()?;
    let push = [
        ne as u32,
        nff as u32,
        rows as u32,
        used as u32,
        experts as u32,
        used as u32,
        windows[0],
        windows[1],
        windows[2],
        parts as u32,
    ];
    let partial_kernel = rec.be.kernel_sg(
        "gpu_owned_hit_splitk_iq2s_32",
        crate::gemm::gpu_owned_hit_splitk_spv(false),
        4,
        40,
        32,
    );
    let reduce_kernel = rec.be.kernel_sg(
        "gpu_owned_hit_splitk_reduce_32",
        crate::gemm::gpu_owned_hit_splitk_spv(true),
        4,
        40,
        32,
    );
    Some(Plan {
        buffers,
        views,
        partial_kernel,
        reduce_kernel,
        push,
        partial_groups,
        reduce_groups,
        command: rec.cmd,
    })
}

impl Plan<'_> {
    fn check_recorder(&self, rec: &Recorder<'_>) {
        assert_eq!(self.command, rec.cmd);
        assert!(Arc::ptr_eq(
            &as_vk_buf(self.buffers[0]).unwrap().shared,
            &rec.be.shared
        ));
        assert!(rec.conditional_active.get() && !rec.conditional_mutable.get());
        assert!(rec.resident_indirect.borrow().is_none() && rec.indirect_region.get().is_none());
        assert!(!rec.suppress.get() && !rec.no_barrier && !rec.full_barrier && !rec.stages);
    }

    /// Record alongside the existing complete payload copy; the caller proves H/BDA disjointness.
    pub(crate) fn partial(&self, rec: &Recorder<'_>) {
        self.check_recorder(rec);
        assert!(rec.independent_dispatch.borrow().is_some());
        rec.dispatch(
            self.partial_kernel,
            &[self.views[0], self.views[1], self.views[2], self.views[3]],
            1,
            bytemuck::cast_slice(&self.push),
            self.partial_groups,
        );
    }

    /// Record after the unconditional independent-scope join, still under the existing H predicate.
    pub(crate) fn reduce(&self, rec: &Recorder<'_>) {
        self.check_recorder(rec);
        assert!(rec.independent_dispatch.borrow().is_none());
        rec.dispatch(
            self.reduce_kernel,
            &[self.views[3], self.views[1], self.views[2], self.views[4]],
            1,
            bytemuck::cast_slice(&self.push),
            self.reduce_groups,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_capacity_is_bounded_and_has_no_shared_slot() {
        assert_eq!(partial_bytes(2560, 640, 4, 10), Some(8_192_000));
        assert_eq!(partial_bytes(8192, 1024, 8, 16), Some(134_217_728));
        for shape in [
            [0, 640, 4, 10],
            [2561, 640, 4, 10],
            [2560, 641, 4, 10],
            [2560, 640, 0, 10],
            [2560, 640, 9, 10],
            [2560, 640, 4, 17],
            [usize::MAX, usize::MAX, usize::MAX, usize::MAX],
        ] {
            assert_eq!(partial_bytes(shape[0], shape[1], shape[2], shape[3]), None);
        }
    }

    #[test]
    fn policy_shape_is_exact() {
        let formats = [DType::Iq2S, DType::Iq2S, DType::Iq4Nl];
        assert!(shape_matches(formats, 2560, 640, 4, 10));
        assert!(!shape_matches(
            [DType::Iq3S, DType::Iq3S, DType::Iq4Nl],
            2560,
            640,
            4,
            10
        ));
        for rows in [1, 2, 3, 5, 8] {
            assert!(!shape_matches(formats, 2560, 640, rows, 10));
        }
        assert!(!shape_matches([DType::Q5K; 3], 2048, 512, 4, 8));
    }

    #[test]
    fn tape_windows_are_entry_offsets_and_ranges_do_not_overlap() {
        assert!(window_fits(512, 512, 1024 * 8));
        assert!(!window_fits(513, 512, 1024 * 8));
        assert!(!window_fits(u32::MAX, usize::MAX, usize::MAX));
        let a = vk::DescriptorBufferInfo::default().offset(256).range(512);
        assert!(disjoint(
            a,
            vk::DescriptorBufferInfo::default().offset(768).range(4)
        ));
        assert!(!disjoint(
            a,
            vk::DescriptorBufferInfo::default().offset(764).range(4)
        ));
    }
}
