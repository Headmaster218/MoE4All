//! VERIFY pair routing follows the selected baseline, never a requested tuning value alone.
use super::*;
use crate::adapter::verify_pair::{self, Cohort, Layer, Path, Phase, Role, RolePolicy};

/// Admission proof for replacing the current NONSHARED staged NR8 plus dense Q8 tail.
/// Do not enroll a device's SG/buffer-grid path or a different explicit NR setting.
pub(crate) fn shared_iq2_verify_fusion_allowed(backend: &VulkanBackend) -> bool {
    let cfg = &backend.cfg().kernels.vulkan;
    if !cfg.verify_shared_iq2_nr8 {
        return false;
    }
    if !verify_pair::shared_iq2_fusion_staged_policy(
        true, cfg.gemv.id_grid_nr,
        native_id_sg_choice(infr_core::DType::Iq2S, 2560, 640, &cfg.gemv).is_some(),
        backend.use_paged_grid_buffer(2560, 640, 4),
    ) {
        return false;
    }
    crate::gemm::native_idm_grid_nr_paged_build_spv(infr_core::DType::Iq2S, cfg.gemv.id_grid_nr)
        .is_some_and(|(name, _)| name == "native_idm_iq2s_nr8_paged")
}

pub(crate) fn baseline_policy(
    backend: &VulkanBackend,
    dtype: infr_core::DType,
    cohort: Cohort,
) -> Option<RolePolicy> {
    let cfg = &backend.cfg().kernels.vulkan;
    if !cfg.verify_pair || !cohort.eligible() || cfg.decode_gpu_resident_indirect_predicates {
        return None;
    }
    let (k, n, rows) = (cohort.in_f, cohort.out_f, cohort.rows);
    if cohort.shared_q8 {
        if let Some(nr) = native_id_sg_choice(dtype, k, n, &cfg.gemv) {
            if crate::gemm::native_idm_sg_paged_shared_build_spv(dtype, nr, false).is_some() {
                return None;
            }
        }
    }
    if backend.use_paged_grid_buffer(k, n, rows) {
        let grid = match dtype {
            infr_core::DType::Iq2S => &backend.iq2s_grid,
            infr_core::DType::Iq3S => &backend.iq3s_grid,
            _ => return None,
        };
        if grid.is_some() {
            let nr = backend.paged_grid_buffer_nr();
            let build = if cohort.shared_q8 {
                crate::gemm::native_idm_grid_buffer_shared_spv(dtype, nr)
            } else {
                crate::gemm::native_idm_grid_buffer_paged_spv(dtype, nr)
            };
            if build.is_some() {
                return None;
            }
        }
    }
    let name = if cohort.shared_q8 {
        let name = crate::gemm::native_idm_paged_shared_build_spv(dtype)?.0;
        if verify_pair::shared_iq2_nr8_eligible(
            cfg.verify_shared_iq2_nr8, cohort.verify && !cohort.independent_rows,
            dtype, [rows, k, n], cohort.used, false, name,
        ) {
            crate::gemm::native_idm_grid_nr_paged_shared_build_spv(dtype, 8)?.0
        } else {
            name
        }
    } else if let Some((name, _)) =
        crate::gemm::native_idm_grid_nr_paged_build_spv(dtype, cfg.gemv.id_grid_nr)
    {
        name
    } else {
        crate::linear::native_idm_paged_kernel_name(dtype)?
    };
    RolePolicy::measured(dtype, name, cohort)
}

impl Recorder<'_> {
    /// Return false to use the original dispatch, including every unsupported device policy.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_verify_pair(
        &self,
        layer: &Layer,
        role: Role,
        dtype: infr_core::DType,
        arena_addr: u64,
        slot_bytes: u32,
        lut: &dyn Buffer,
        ids: &dyn Buffer,
        lut_base: usize,
        shared: Option<&dyn Buffer>,
        x: &dyn Buffer,
        y: &dyn Buffer,
        active_mask: u32,
        row_mask_bank: u32,
    ) -> bool {
        let Some(policy) = layer.roles[role as usize] else {
            return false;
        };
        if row_mask_bank > 2
            || self.resident_indirect.borrow().is_some()
            || shared.is_some() != layer.cohort.shared_q8
            || baseline_policy(self.be, dtype, layer.cohort) != Some(policy)
        {
            return false;
        }
        let cohort = layer.cohort;
        let phase = match row_mask_bank {
            0 => Phase::Full,
            1 => Phase::Hits,
            _ => Phase::Misses,
        };
        let (baseline_name, baseline_spv) = if shared.is_some() {
            crate::gemm::native_idm_paged_shared_build_spv(dtype).unwrap()
        } else if policy.nr == 8 {
            crate::gemm::native_idm_grid_nr_paged_build_spv(dtype, policy.nr).unwrap()
        } else {
            (
                crate::linear::native_idm_paged_kernel_name(dtype).unwrap(),
                crate::gemm::native_idm_paged_build_spv(dtype).unwrap(),
            )
        };
        let pair_name = match (dtype, shared.is_some()) {
            (infr_core::DType::Iq2S, false) => "verify_pair_iq2s_nr8",
            (infr_core::DType::Iq3S, false) => "verify_pair_iq3s_nr1",
            (infr_core::DType::Iq3S, true) => "verify_pair_iq3s_nr1_shared",
            _ => return false,
        };
        let physical_used = cohort.used + usize::from(shared.is_some());
        let shared_address = shared
            .map(|buffer| {
                buffer
                    .device_addr()
                    .expect("shared VERIFY expert must expose a BDA address")
            })
            .unwrap_or(0);
        let words = [
            cohort.in_f as u32,
            cohort.out_f as u32,
            physical_used as u32,
            lut_base as u32,
            0,
            cohort.rows as u32,
            arena_addr as u32,
            (arena_addr >> 32) as u32,
            slot_bytes,
            active_mask,
            row_mask_bank,
            shared_address as u32,
            (shared_address >> 32) as u32,
            cohort.used as u32,
        ];
        let push = bytemuck::cast_slice(&words[..if shared.is_some() { 14 } else { 11 }]);
        let metadata = Self::vkb(layer.buffer.as_ref());
        // Both command buffers retain the metadata allocation; a notice is NOT its retirement.
        self.retain_buffer(Arc::clone(&layer.buffer));
        let baseline = self
            .be
            .kernel(baseline_name, baseline_spv, 5, push.len() as u32);
        self.dispatch_indirect(
            baseline,
            &[
                Self::vkb(lut),
                Self::vkb(x),
                Self::vkb(ids),
                Self::vkb(lut),
                Self::vkb(y),
            ],
            1,
            push,
            metadata.buffer,
            metadata.offset + verify_pair::command_byte_offset(role, phase, Path::Baseline),
        );
        let paired = self.be.kernel(
            pair_name,
            crate::gemm::verify_pair_spv(pair_name),
            6,
            push.len() as u32,
        );
        self.dispatch_indirect(
            paired,
            &[
                Self::vkb(lut),
                Self::vkb(x),
                Self::vkb(ids),
                Self::vkb(lut),
                metadata,
                Self::vkb(y),
            ],
            1,
            push,
            metadata.buffer,
            metadata.offset + verify_pair::command_byte_offset(role, phase, Path::Pair),
        );
        true
    }
}
