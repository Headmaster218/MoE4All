//! Read-only, post-run attribution for the physical device already bound by the backend.

use crate::{vk, CStr, VulkanBackend};

#[derive(Debug, Clone)]
pub struct BenchmarkDeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub unified_memory: bool,
    pub driver_id: Option<i32>,
    pub driver_id_name: Option<String>,
    pub driver_name: Option<String>,
    pub driver_info: Option<String>,
    /// Vulkan's vendor-specific driverVersion encoding. Do not decode as an API version.
    pub driver_version_raw: u32,
    pub driver_unknown_reason: Option<&'static str>,
    pub pci_bdf: Option<String>,
    pub pci_unknown_reason: Option<&'static str>,
    /// Unknown when enforcement takes the minimum of live headroom and policy ceilings.
    pub vram_budget_bytes: Option<u64>,
    pub vram_budget_source: Option<&'static str>,
    pub vram_budget_unknown_reason: Option<&'static str>,
}

impl VulkanBackend {
    /// Query only physical-device properties. No queue access, submission, or GPU work.
    pub fn benchmark_device_info(&self) -> BenchmarkDeviceInfo {
        let instance = &self.shared.instance;
        let pd = self.shared.physical_device;
        let base = unsafe { instance.get_physical_device_properties(pd) };
        let extensions = unsafe { instance.enumerate_device_extension_properties(pd) };
        let has_ext = |name: &CStr| {
            extensions.as_ref().is_ok_and(|extensions| {
                extensions
                    .iter()
                    .any(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) == name })
            })
        };
        let has_driver =
            base.api_version >= vk::API_VERSION_1_2 || has_ext(c"VK_KHR_driver_properties");
        let has_pci = has_ext(c"VK_EXT_pci_bus_info");
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default();
        if has_driver {
            props = props.push_next(&mut driver);
        }
        if has_pci {
            props = props.push_next(&mut pci);
        }
        unsafe { instance.get_physical_device_properties2(pd, &mut props) };

        BenchmarkDeviceInfo {
            name: unsafe { CStr::from_ptr(base.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            vendor_id: base.vendor_id,
            device_id: base.device_id,
            unified_memory: self.shared.caps.unified_memory,
            driver_id: has_driver.then_some(driver.driver_id.as_raw()),
            driver_id_name: has_driver.then(|| format!("{:?}", driver.driver_id)),
            driver_name: has_driver.then(|| {
                unsafe { CStr::from_ptr(driver.driver_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            }),
            driver_info: has_driver.then(|| {
                unsafe { CStr::from_ptr(driver.driver_info.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            }),
            driver_version_raw: base.driver_version,
            driver_unknown_reason: (!has_driver).then_some("driver_properties_not_supported"),
            pci_bdf: has_pci.then(|| {
                format_pci_bdf(
                    pci.pci_domain,
                    pci.pci_bus,
                    pci.pci_device,
                    pci.pci_function,
                )
            }),
            pci_unknown_reason: (!has_pci).then_some(if extensions.is_err() {
                "device_extension_query_failed"
            } else {
                "VK_EXT_pci_bus_info_not_supported"
            }),
            vram_budget_bytes: None,
            vram_budget_source: None,
            vram_budget_unknown_reason: Some(if self.cfg.kernels.vulkan.no_vram_guard {
                "vram_allocation_guard_disabled"
            } else {
                "allocation_guard_uses_dynamic_minimum_of_live_headroom_and_policy_ceilings"
            }),
        }
    }
}

fn format_pci_bdf(domain: u32, bus: u32, device: u32, function: u32) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}")
}

#[cfg(test)]
mod tests {
    #[test]
    fn pci_bdf_preserves_domain_and_zero_values() {
        assert_eq!(super::format_pci_bdf(0, 0, 0, 0), "0000:00:00.0");
        assert_eq!(super::format_pci_bdf(0x1234, 0xab, 0x1f, 7), "1234:ab:1f.7");
    }
}
