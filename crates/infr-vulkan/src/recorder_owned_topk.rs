//! Certified, ordinary router/admission fusion. No payload/publication policy lives here.
use super::*;

#[cfg(test)]
#[path = "recorder_owned_topk_overlap_recording_tests.rs"]
mod overlap_recording_tests;

#[derive(Clone, Copy)]
pub(crate) struct TopkParams {
    pub rows: usize,
    pub experts: usize,
    pub used: usize,
    pub scale: f32,
    pub gating: u32,
    pub norm_w: bool,
    pub has_bias: bool,
    pub groups: u32,
    pub groups_used: u32,
    pub hash: bool,
}

impl TopkParams {
    pub(super) fn wave32(self, enabled: bool) -> bool {
        enabled
            && self.rows == 1
            && self.gating <= 1
            && self.norm_w
            && !self.has_bias
            && (self.groups <= 1 || self.groups_used == 0)
            && !self.hash
            && self.experts > 0
            && self.experts <= 512
            && self.used > 0
            && self.used <= 32
    }

    fn two_row_general(self, enabled: bool) -> bool {
        enabled
            && self.rows == 2
            && matches!(self.used, 8 | 10)
            && self.gating <= 1
            && self.norm_w
            && !self.has_bias
            && (self.groups <= 1 || self.groups_used == 0)
            && !self.hash
            && self.experts > 0
            && self.experts <= 512
    }

    fn owned_mode(self, enabled: bool) -> Option<OwnedTopkMode> {
        if self.wave32(enabled) && self.used <= 16 {
            Some(OwnedTopkMode::Wave32)
        } else if self.two_row_general(enabled) {
            Some(OwnedTopkMode::TwoRowGeneral)
        } else {
            None
        }
    }

    fn push(self) -> [u32; 9] {
        [
            self.experts as u32,
            self.used as u32,
            self.scale.to_bits(),
            self.gating,
            u32::from(self.norm_w),
            u32::from(self.has_bias),
            self.groups,
            self.groups_used,
            u32::from(self.hash),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnedTopkMode {
    Wave32,
    TwoRowGeneral,
}

#[derive(Clone, Copy)]
pub(crate) struct OwnedTopkRoute {
    pub(super) logits: vk::DescriptorBufferInfo,
    weights: vk::DescriptorBufferInfo,
    params: TopkParams,
    mode: OwnedTopkMode,
    predicate: usize,
    ordinal: u32,
    geometry: [u32; 12],
    overlap_descriptors: bool,
}

fn descriptor_limits(
    stage: u32,
    set: u32,
    resources: u32,
    push: Option<u32>,
    overlap: bool,
) -> bool {
    let required = if overlap { 11 } else { 10 };
    stage >= required
        && set >= required
        && resources >= required
        && push.is_none_or(|n| n >= required)
}

fn subgroup_capabilities(operations: vk::SubgroupFeatureFlags) -> bool {
    operations.contains(
        vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::ARITHMETIC
            | vk::SubgroupFeatureFlags::BALLOT,
    )
}

#[cfg(test)]
fn device_descriptors(be: &VulkanBackend, overlap: bool) -> bool {
    device_descriptor_modes(be)[usize::from(overlap)]
}

fn device_descriptor_modes(be: &VulkanBackend) -> [bool; 2] {
    type Cached = (std::sync::Weak<crate::VulkanShared>, [bool; 2]);
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Vec<Cached>>> = std::sync::OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache.retain(|(owner, _)| owner.strong_count() != 0);
    if let Some((_, supported)) = cache
        .iter()
        .find(|(owner, _)| std::sync::Weak::ptr_eq(owner, &Arc::downgrade(&be.shared)))
    {
        return *supported;
    }
    let mut push_properties = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
    let mut properties = vk::PhysicalDeviceProperties2::default();
    if be.shared.push_descriptor.is_some() {
        properties = properties.push_next(&mut push_properties);
    }
    unsafe {
        be.shared
            .instance
            .get_physical_device_properties2(be.shared.physical_device, &mut properties)
    };
    let limits = properties.properties.limits;
    let supported = [false, true].map(|overlap| {
        descriptor_limits(
            limits.max_per_stage_descriptor_storage_buffers,
            limits.max_descriptor_set_storage_buffers,
            limits.max_per_stage_resources,
            be.shared
                .push_descriptor
                .as_ref()
                .map(|_| push_properties.max_push_descriptors),
            overlap,
        )
    });
    cache.push((Arc::downgrade(&be.shared), supported));
    supported
}

fn predicate_word(ordinal: u32) -> Option<usize> {
    ((ordinal as usize) < RESIDENT_MAX_LAYERS).then(|| {
        ordinal
            .checked_sub(1)
            .map_or(RESIDENT_SEED_WORD, |n| 5 + n as usize * 2)
    })
}

impl Recorder<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn defer_owned_topk(
        &self,
        copy: &crate::pager::GpuCopyBuffers,
        control: &dyn Buffer,
        logits: &dyn Buffer,
        weights: &dyn Buffer,
        bases: [u32; 3],
        ordinal: u32,
        params: TopkParams,
        ordinary: bool,
        input_tap: bool,
    ) -> Option<OwnedTopkRoute> {
        let mode = params.owned_mode(self.vk().moe_topk_sg)?;
        if !ordinary
            || input_tap
            || !copy.is_owned()
            || self.no_barrier
            || self.full_barrier
            || self.suppress.get()
            || self.stages
            || self.independent_dispatch.borrow().is_some()
            || control.len_bytes() != RESIDENT_CONTROL_BYTES
            || logits.len_bytes() < params.rows * params.experts * 4
            || weights.len_bytes() < params.rows * params.used * 4
        {
            return None;
        }
        if !subgroup_capabilities(self.be.shared.compute_subgroup_operations) {
            return None;
        }
        // Ten-binding devices may still defer routing for overlap: admission falls
        // back to the original immutable router. Capture both immutable limits once.
        let descriptor_modes = device_descriptor_modes(self.be);
        if !descriptor_modes[0] {
            return None;
        }
        #[cfg(test)]
        if !test_enabled(self.be) {
            return None;
        }
        let geometry = copy.owned_role_geometry(bases, params.experts as u32)?;
        let word = predicate_word(ordinal)?;
        let buffer = Self::vk_of(control);
        if !self.conditional_active.get()
            || self.immutable_condition.get() != Some((buffer.buffer, buffer.sub_offset, word))
        {
            return None;
        }
        Some(OwnedTopkRoute {
            logits: Self::vkb(logits),
            weights: Self::vkb(weights),
            params,
            mode,
            predicate: word,
            ordinal,
            geometry,
            overlap_descriptors: descriptor_modes[1],
        })
    }

    pub(super) fn dispatch_deferred_topk(
        &self,
        ids: &dyn Buffer,
        control: &dyn Buffer,
        topk: OwnedTopkRoute,
    ) {
        // Eligibility may decline after snapshotting. Restore the exact original routing
        // predicate, not run and not an unconditionally executed substitute.
        self.begin_immutable_conditional_at(control, topk.predicate);
        let (name, spv) = match topk.mode {
            OwnedTopkMode::Wave32 => (
                "moe_topk_sigmoid_wave32",
                crate::gemm::moe_topk_sigmoid_wave32_spv(),
            ),
            OwnedTopkMode::TwoRowGeneral => ("moe_topk_sg", crate::gemm::moe_topk_sg_spv()),
        };
        let kernel = self.be.kernel_sg(name, spv, 5, 36, 32);
        self.dispatch(
            kernel,
            &[
                topk.logits,
                topk.logits,
                topk.logits,
                Self::vkb(ids),
                topk.weights,
            ],
            2,
            bytemuck::cast_slice(&topk.params.push()),
            topk.params.rows as u32,
        );
        self.end_conditional();
        #[cfg(test)]
        test_note(self.be, |counts| counts.routers += 1);
    }

    pub(super) fn try_dispatch_owned_topk(
        &self,
        copy: &crate::pager::GpuCopyBuffers,
        buffers: &[vk::DescriptorBufferInfo],
        base: &[u32; 19],
        topk: OwnedTopkRoute,
        overlap: bool,
    ) -> bool {
        let geometry = copy.owned_role_geometry([base[9], base[10], base[11]], base[1]);
        if base[12] != 0
            || base[5] != topk.ordinal
            || base[0] != (topk.params.rows * topk.params.used) as u32
            || base[6] != topk.params.used as u32
            || base[1] != topk.params.experts as u32
            || geometry != Some(topk.geometry)
            || buffers.len() != (if overlap { 9 } else { 8 })
            || (overlap && !topk.overlap_descriptors)
        {
            return false;
        }
        let mut push = gpu_owned_fast_push(base, &topk.geometry);
        push[13] = topk.params.scale.to_bits();
        push[14] = topk.params.gating;
        push[15] = topk.predicate as u32;
        let mut bindings = buffers.to_vec();
        bindings.extend([topk.logits, topk.weights]);
        let (name, spv) = match (topk.mode, overlap) {
            (OwnedTopkMode::Wave32, false) => (
                "gpu_owned_prepare_topk_fused",
                crate::gemm::gpu_owned_prepare_topk_fused_spv(),
            ),
            (OwnedTopkMode::Wave32, true) => (
                "gpu_owned_prepare_topk_fused_overlap",
                crate::gemm::gpu_owned_prepare_topk_fused_overlap_spv(),
            ),
            (OwnedTopkMode::TwoRowGeneral, false) => (
                "gpu_owned_prepare_topk_fused_two_row",
                crate::gemm::gpu_owned_prepare_topk_fused_two_row_spv(),
            ),
            (OwnedTopkMode::TwoRowGeneral, true) => (
                "gpu_owned_prepare_topk_fused_two_row_overlap",
                crate::gemm::gpu_owned_prepare_topk_fused_two_row_overlap_spv(),
            ),
        };
        let kernel = self
            .be
            .kernel_sg(name, spv, bindings.len(), GPU_OWNED_FAST_PUSH_BYTES, 32);
        let _phase = infr_core::timeline::device_enabled()
            .then(|| infr_core::timeline::device_phase_scope("resident_owned_topk_prepare"));
        // Match owner admission's conservative tracking, including logits: its WAW imports
        // the router projection and orders previous compute BDA readers before admission.
        self.dispatch(
            kernel,
            &bindings,
            bindings.len(),
            bytemuck::cast_slice(&push),
            1,
        );
        self.owned_timeline_boundary("resident_owned_mailbox_completion_interval");
        #[cfg(test)]
        test_note(self.be, |counts| counts.fused += 1);
        true
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Counts {
    pub fused: usize,
    pub routers: usize,
    pub admissions: usize,
    pub recovery: usize,
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct TestState {
    enabled: bool,
    counts: Counts,
}

#[cfg(test)]
fn test_states() -> &'static std::sync::Mutex<std::collections::HashMap<usize, TestState>> {
    static STATES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, TestState>>,
    > = std::sync::OnceLock::new();
    STATES.get_or_init(Default::default)
}

#[cfg(test)]
fn test_enabled(be: &VulkanBackend) -> bool {
    test_states()
        .lock()
        .unwrap()
        .get(&(Arc::as_ptr(&be.shared) as usize))
        .is_none_or(|state| state.enabled)
}

#[cfg(test)]
pub(super) fn test_note(be: &VulkanBackend, note: impl FnOnce(&mut Counts)) {
    if let Some(state) = test_states()
        .lock()
        .unwrap()
        .get_mut(&(Arc::as_ptr(&be.shared) as usize))
    {
        note(&mut state.counts);
    }
}

#[cfg(test)]
pub(crate) struct TestGuard {
    key: usize,
    previous: Option<TestState>,
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
pub(crate) fn test_scope(be: &VulkanBackend, enabled: bool) -> TestGuard {
    let key = Arc::as_ptr(&be.shared) as usize;
    let previous = test_states().lock().unwrap().insert(
        key,
        TestState {
            enabled,
            counts: Counts::default(),
        },
    );
    TestGuard { key, previous }
}

#[cfg(test)]
pub(crate) fn test_counts(be: &VulkanBackend) -> Counts {
    test_states()
        .lock()
        .unwrap()
        .get(&(Arc::as_ptr(&be.shared) as usize))
        .unwrap()
        .counts
}

#[cfg(test)]
pub(crate) fn test_supported(be: &VulkanBackend) -> bool {
    test_supported_mode(be, false)
}

#[cfg(test)]
pub(crate) fn test_supported_mode(be: &VulkanBackend, overlap: bool) -> bool {
    be.cfg().kernels.vulkan.moe_topk_sg
        && subgroup_capabilities(be.shared.compute_subgroup_operations)
        && device_descriptors(be, overlap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_topk_descriptor_limits_and_original_wave32_rules() {
        let all = vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::ARITHMETIC
            | vk::SubgroupFeatureFlags::BALLOT;
        assert!(subgroup_capabilities(all));
        for missing in [
            vk::SubgroupFeatureFlags::BASIC,
            vk::SubgroupFeatureFlags::ARITHMETIC,
            vk::SubgroupFeatureFlags::BALLOT,
        ] {
            assert!(!subgroup_capabilities(all & !missing));
        }
        assert!(descriptor_limits(10, 10, 10, Some(10), false));
        assert!(descriptor_limits(10, 10, 10, None, false));
        for args in [
            (9, 10, 10, Some(10)),
            (10, 9, 10, Some(10)),
            (10, 10, 9, Some(10)),
            (10, 10, 10, Some(9)),
        ] {
            assert!(!descriptor_limits(args.0, args.1, args.2, args.3, false));
        }
        let good = TopkParams {
            rows: 1,
            experts: 512,
            used: 10,
            scale: 1.0,
            gating: 1,
            norm_w: true,
            has_bias: false,
            groups: 0,
            groups_used: 0,
            hash: false,
        };
        assert!(good.wave32(true));
        assert!(!good.wave32(false));
        for bad in [
            TopkParams { rows: 2, ..good },
            TopkParams {
                experts: 513,
                ..good
            },
            TopkParams { gating: 2, ..good },
            TopkParams {
                norm_w: false,
                ..good
            },
            TopkParams {
                has_bias: true,
                ..good
            },
            TopkParams { hash: true, ..good },
            TopkParams {
                groups: 2,
                groups_used: 1,
                ..good
            },
        ] {
            assert!(!bad.wave32(true));
        }
        assert_eq!(predicate_word(0), Some(RESIDENT_SEED_WORD));
        assert_eq!(predicate_word(1), Some(5));
        assert_eq!(predicate_word(128), None);
    }

    #[test]
    fn owned_topk_overlap_descriptor_limits_preserve_serial_ten_binding_devices() {
        assert!(!descriptor_limits(10, 10, 10, Some(10), true));
        assert!(descriptor_limits(10, 10, 10, Some(10), false));
        assert!(descriptor_limits(11, 11, 11, Some(11), true));
        assert!(descriptor_limits(11, 11, 11, None, true));
        for args in [
            (10, 11, 11, Some(11)),
            (11, 10, 11, Some(11)),
            (11, 11, 10, Some(11)),
            (11, 11, 11, Some(10)),
        ] {
            assert!(!descriptor_limits(args.0, args.1, args.2, args.3, true));
            assert!(descriptor_limits(args.0, args.1, args.2, args.3, false));
        }
    }

    #[test]
    fn owned_topk_two_row_general_gate_does_not_change_ordinary_wave32() {
        let good = TopkParams {
            rows: 2,
            experts: 512,
            used: 10,
            scale: 1.375,
            gating: 1,
            norm_w: true,
            has_bias: false,
            groups: 0,
            groups_used: 0,
            hash: false,
        };
        for used in [8, 10] {
            let params = TopkParams { used, ..good };
            assert!(!params.wave32(true));
            assert_eq!(params.owned_mode(true), Some(OwnedTopkMode::TwoRowGeneral));
            assert_eq!(params.owned_mode(false), None);
        }
        for bad in [
            TopkParams { rows: 0, ..good },
            TopkParams { rows: 4, ..good },
            TopkParams { used: 9, ..good },
            TopkParams { used: 16, ..good },
            TopkParams { experts: 0, ..good },
            TopkParams {
                experts: 513,
                ..good
            },
            TopkParams { gating: 2, ..good },
            TopkParams {
                norm_w: false,
                ..good
            },
            TopkParams {
                has_bias: true,
                ..good
            },
            TopkParams { hash: true, ..good },
            TopkParams {
                groups: 2,
                groups_used: 1,
                ..good
            },
        ] {
            assert_eq!(bad.owned_mode(true), None);
        }
        assert!(TopkParams { gating: 0, ..good }.two_row_general(true));
        let single = TopkParams { rows: 1, ..good };
        assert!(single.wave32(true));
        assert_eq!(single.owned_mode(true), Some(OwnedTopkMode::Wave32));
        let ordinary_only = TopkParams { used: 32, ..single };
        assert!(ordinary_only.wave32(true));
        assert_eq!(ordinary_only.owned_mode(true), None);
    }
}
