//! Measured narrow projection admission, shared by planning and recording.
use crate::caps::{DeviceArch, DeviceProbe, VENDOR_AMD};
use ash::vk;
use infr_core::{config::VulkanCfg, DType};

pub(crate) const MAX_SCRATCH_BYTES: usize = 512 * 1024;

pub(crate) fn device_eligible(p: &DeviceProbe, arch: DeviceArch, bda: bool, windows: bool) -> bool {
    windows
        && bda
        && !p.integrated
        && p.vendor_id == VENDOR_AMD
        && p.driver_id_reported
        && p.driver_id == vk::DriverId::AMD_PROPRIETARY
        && arch == DeviceArch::AmdRdna3
        // The architecture bucket also contains RDNA4. Keep unmeasured devices off;
        // 0x744c is the Navi31 family used by the 7900 XTX campaign.
        && p.device_id == 0x744c
}

pub(crate) fn scratch_bytes(
    cfg: &VulkanCfg,
    measured_device: bool,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    w_off: usize,
) -> Option<usize> {
    if !measured_device
        || !cfg.narrow_f32_gemv
        || !cfg.f32_v4
        || !cfg.f32_mrow
        || !matches!(dtype, DType::Q8_0 | DType::Bf16)
    {
        return None;
    }
    let min_m = match (k, n) {
        (2560, 48) => 256,
        (10240, 4) => 1024,
        _ => return None,
    };
    let elems = k.checked_mul(n)?;
    let bytes = elems.checked_mul(4)?;
    // All shader indices/push fields are u32. This conservative end bound also
    // keeps native_decode's packed-word byte offsets below 4 GiB for both dtypes.
    if m < min_m
        || m.checked_mul(k)? > u32::MAX as usize
        || w_off % k != 0
        || w_off.checked_add(elems)? > (u32::MAX as usize / 2)
        || bytes > MAX_SCRATCH_BYTES
    {
        return None;
    }
    Some(bytes)
}
