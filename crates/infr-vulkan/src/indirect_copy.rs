//! Optional memory-to-memory subset of `VK_KHR_copy_memory_indirect` for ash 0.38.
//!
//! ABI source: installed Vulkan SDK 1.4.357.0, `Include/vulkan/vulkan_core.h`,
//! lines 1308-1311, 7208, 12518-12522, and 14115-14175 (also `share/vulkan/registry/vk.xml`).
//! The properties sType is shared with NV; the KHR FEATURES sType and command ABI are not.
//! No NV command fallback, image feature enable, or recorder policy is implied by this module.

use std::ffi::{c_void, CStr};
use std::ptr;

use ash::vk;

pub(crate) const EXTENSION_NAME: &CStr = c"VK_KHR_copy_memory_indirect";
const ST_FEATURES: i32 = 1_000_549_000;
const ST_PROPERTIES: i32 = 1_000_426_001;
const ST_COPY_INFO: i32 = 1_000_549_002;

pub(crate) type AddressCopyFlagsKHR = vk::Flags;
pub(crate) const ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR: AddressCopyFlagsKHR = 0x0000_0001;
pub(crate) const ADDRESS_COPY_SPARSE_BIT_KHR: AddressCopyFlagsKHR = 0x0000_0002;
pub(crate) const ADDRESS_COPY_PROTECTED_BIT_KHR: AddressCopyFlagsKHR = 0x0000_0004;
pub(crate) const COPY_INDIRECT_STAGE_KHR: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::from_raw(0x4000_0000_0000);

/// `VkCopyMemoryIndirectCommandKHR`: GPU-readable copy payload, with 4-byte-aligned addresses
/// and byte count. These are Vulkan buffer device addresses, never ordinary host pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CopyMemoryIndirectCommandKHR {
    pub src_address: vk::DeviceAddress,
    pub dst_address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
}

/// `VkStridedDeviceAddressRangeKHR`. Do not substitute ash's ray-tracing
/// `StridedDeviceAddressRegionKHR`, which places stride BEFORE size.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StridedDeviceAddressRangeKHR {
    pub address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    pub stride: vk::DeviceSize,
}

/// `VkCopyMemoryIndirectInfoKHR`. Zero source flags permit non-device-local source memory;
/// DEVICE_LOCAL is an optional residency hint, not a requirement of the KHR copy feature.
/// For non-device-local to local copies, leave src flags zero and set only dst DEVICE_LOCAL.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CopyMemoryIndirectInfoKHR {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub src_copy_flags: AddressCopyFlagsKHR,
    pub dst_copy_flags: AddressCopyFlagsKHR,
    pub copy_count: u32,
    pub copy_address_range: StridedDeviceAddressRangeKHR,
}

impl Default for CopyMemoryIndirectInfoKHR {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(ST_COPY_INFO),
            p_next: ptr::null(),
            src_copy_flags: 0,
            dst_copy_flags: 0,
            copy_count: 0,
            copy_address_range: StridedDeviceAddressRangeKHR::default(),
        }
    }
}

/// `VkPhysicalDeviceCopyMemoryIndirectFeaturesKHR` (not the NV feature struct).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CopyMemoryIndirectFeaturesKHR {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub indirect_memory_copy: vk::Bool32,
    pub indirect_memory_to_image_copy: vk::Bool32,
}

impl Default for CopyMemoryIndirectFeaturesKHR {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(ST_FEATURES),
            p_next: ptr::null_mut(),
            indirect_memory_copy: vk::FALSE,
            indirect_memory_to_image_copy: vk::FALSE,
        }
    }
}

impl crate::vkext::RawFeatureStruct for CopyMemoryIndirectFeaturesKHR {
    fn p_next_slot(&mut self) -> &mut *mut c_void {
        &mut self.p_next
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CopyMemoryIndirectPropertiesKHR {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    supported_queues: vk::QueueFlags,
}

impl Default for CopyMemoryIndirectPropertiesKHR {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(ST_PROPERTIES),
            p_next: ptr::null_mut(),
            supported_queues: vk::QueueFlags::empty(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Support {
    pub indirect_memory_copy: bool,
    pub supported_queues: vk::QueueFlags,
    /// Queried core-1.3 feature, not inferred merely from the instance's requested version.
    pub synchronization2: bool,
}

impl Support {
    pub(crate) fn supports_queue(self, queue_flags: vk::QueueFlags) -> bool {
        self.indirect_memory_copy && self.supported_queues.intersects(queue_flags)
    }

    /// Only memory-to-memory is enabled; never request the optional image feature.
    pub(crate) fn enable_features(self) -> CopyMemoryIndirectFeaturesKHR {
        CopyMemoryIndirectFeaturesKHR {
            indirect_memory_copy: self.indirect_memory_copy as vk::Bool32,
            ..Default::default()
        }
    }
}

/// Query only an advertised KHR extension. No query at all is issued when it is absent.
/// Sync2 is optional and queried only on a physical device exposing core Vulkan 1.3.
///
/// # Safety
/// When `extension_available` is true, `physical_device` must belong to `instance` and advertise
/// the KHR extension. The instance must support the core properties2/features2 API.
pub(crate) unsafe fn query(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extension_available: bool,
) -> Support {
    if !extension_available {
        return Support::default();
    }
    let api_version = instance
        .get_physical_device_properties(physical_device)
        .api_version;
    let core13 = (
        vk::api_version_major(api_version),
        vk::api_version_minor(api_version),
    ) >= (1, 3);
    let mut features = CopyMemoryIndirectFeaturesKHR::default();
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2 {
        p_next: ptr::from_mut(&mut features).cast(),
        ..Default::default()
    };
    if core13 {
        features2 = features2.push_next(&mut sync2);
    }
    instance.get_physical_device_features2(physical_device, &mut features2);
    let mut properties = CopyMemoryIndirectPropertiesKHR::default();
    let mut properties2 = vk::PhysicalDeviceProperties2 {
        p_next: ptr::from_mut(&mut properties).cast(),
        ..Default::default()
    };
    instance.get_physical_device_properties2(physical_device, &mut properties2);
    Support {
        indirect_memory_copy: features.indirect_memory_copy != vk::FALSE,
        supported_queues: properties.supported_queues,
        synchronization2: core13 && sync2.synchronization2 != vk::FALSE,
    }
}

/// `PFN_vkCmdCopyMemoryIndirectKHR`; use the KHR info pointer ABI, not NV's address/count/stride.
pub(crate) type PfnCmdCopyMemoryIndirectKHR =
    unsafe extern "system" fn(vk::CommandBuffer, *const CopyMemoryIndirectInfoKHR);

/// Enabled memory-to-memory command plus its queue/synchronization constraints. Presence is
/// capability only: recorder policy must opt in separately. Fields contain no borrowed pointers.
#[derive(Clone, Copy)]
pub(crate) struct Device {
    pub cmd_copy_memory_indirect: PfnCmdCopyMemoryIndirectKHR,
    pub supported_queues: vk::QueueFlags,
    /// True only when the queried feature was also chained into this device's creation.
    /// A caller using COPY_INDIRECT_STAGE_KHR with barrier2 must require this flag.
    pub synchronization2: bool,
}

impl Device {
    /// Record exactly the caller's bounded payload range, with both address-copy flags zero.
    /// This command is NOT affected by Vulkan conditional rendering: even a skipped layer must
    /// publish safe payload records before reaching it (including any dummy-copy records).
    ///
    /// # Safety
    /// The command buffer must be recording on a supported queue family of this device, outside
    /// rendering/video scopes and unprotected. The caller must validate 4-byte alignment,
    /// `stride >= 24`, and `copy_count <= address_range_size / stride`. The range must belong to
    /// a live buffer with INDIRECT_BUFFER and device-address usage. Its records, source and
    /// destination memory must satisfy Vulkan bounds, lifetime, non-overlap and synchronization
    /// requirements. Device-generated payload writes need an explicit dependency before this call.
    pub(crate) unsafe fn record(
        &self,
        cmd: vk::CommandBuffer,
        address: u64,
        copy_count: u32,
        stride: u64,
        address_range_size: u64,
    ) {
        self.record_with_flags(cmd, address, copy_count, stride, address_range_size, 0, 0);
    }

    /// Explicit, command-wide address hints; does not change the default recording policy.
    ///
    /// # Safety
    /// All `record` requirements apply. The caller must additionally prove the flags describe
    /// EVERY payload range, including generated dummy records, not just the command array.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record_with_flags(
        &self,
        cmd: vk::CommandBuffer,
        address: u64,
        copy_count: u32,
        stride: u64,
        address_range_size: u64,
        src_copy_flags: AddressCopyFlagsKHR,
        dst_copy_flags: AddressCopyFlagsKHR,
    ) {
        let info = CopyMemoryIndirectInfoKHR {
            src_copy_flags,
            dst_copy_flags,
            copy_count,
            copy_address_range: StridedDeviceAddressRangeKHR {
                address,
                size: address_range_size,
                stride,
            },
            ..Default::default()
        };
        (self.cmd_copy_memory_indirect)(cmd, &info);
    }

    /// # Safety
    /// `device` must belong to `instance`. Its KHR extension and memory-copy feature must have
    /// been enabled using `support`, and its sync2 feature enabled iff `support.synchronization2`.
    /// `queue_flags` must describe the command pool's actual queue family.
    pub(crate) unsafe fn load(
        instance: &ash::Instance,
        device: &ash::Device,
        support: Support,
        queue_flags: vk::QueueFlags,
    ) -> Option<Self> {
        if !support.supports_queue(queue_flags) {
            return None;
        }
        Self::from_proc_addr(
            support,
            queue_flags,
            instance.get_device_proc_addr(device.handle(), c"vkCmdCopyMemoryIndirectKHR".as_ptr()),
        )
    }

    unsafe fn from_proc_addr(
        support: Support,
        queue_flags: vk::QueueFlags,
        address: vk::PFN_vkVoidFunction,
    ) -> Option<Self> {
        if !support.supports_queue(queue_flags) {
            return None;
        }
        let address = address?;
        Some(Self {
            cmd_copy_memory_indirect: std::mem::transmute::<
                unsafe extern "system" fn(),
                PfnCmdCopyMemoryIndirectKHR,
            >(address),
            supported_queues: support.supported_queues,
            synchronization2: support.synchronization2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn payload_and_address_range_match_header_abi() {
        assert_eq!(size_of::<CopyMemoryIndirectCommandKHR>(), 24);
        assert_eq!(offset_of!(CopyMemoryIndirectCommandKHR, src_address), 0);
        assert_eq!(offset_of!(CopyMemoryIndirectCommandKHR, dst_address), 8);
        assert_eq!(offset_of!(CopyMemoryIndirectCommandKHR, size), 16);
        assert_eq!(
            align_of::<CopyMemoryIndirectCommandKHR>(),
            align_of::<u64>()
        );
        assert_eq!(size_of::<StridedDeviceAddressRangeKHR>(), 24);
        assert_eq!(offset_of!(StridedDeviceAddressRangeKHR, address), 0);
        assert_eq!(offset_of!(StridedDeviceAddressRangeKHR, size), 8);
        assert_eq!(offset_of!(StridedDeviceAddressRangeKHR, stride), 16);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn chain_structs_match_64_bit_header_abi() {
        assert_eq!(size_of::<CopyMemoryIndirectFeaturesKHR>(), 24);
        assert_eq!(align_of::<CopyMemoryIndirectFeaturesKHR>(), 8);
        assert_eq!(offset_of!(CopyMemoryIndirectFeaturesKHR, s_type), 0);
        assert_eq!(offset_of!(CopyMemoryIndirectFeaturesKHR, p_next), 8);
        assert_eq!(
            offset_of!(CopyMemoryIndirectFeaturesKHR, indirect_memory_copy),
            16
        );
        assert_eq!(
            offset_of!(CopyMemoryIndirectFeaturesKHR, indirect_memory_to_image_copy),
            20
        );
        assert_eq!(size_of::<CopyMemoryIndirectPropertiesKHR>(), 24);
        assert_eq!(align_of::<CopyMemoryIndirectPropertiesKHR>(), 8);
        assert_eq!(offset_of!(CopyMemoryIndirectPropertiesKHR, s_type), 0);
        assert_eq!(offset_of!(CopyMemoryIndirectPropertiesKHR, p_next), 8);
        assert_eq!(
            offset_of!(CopyMemoryIndirectPropertiesKHR, supported_queues),
            16
        );
        assert_eq!(size_of::<CopyMemoryIndirectInfoKHR>(), 56);
        assert_eq!(align_of::<CopyMemoryIndirectInfoKHR>(), 8);
        assert_eq!(offset_of!(CopyMemoryIndirectInfoKHR, s_type), 0);
        assert_eq!(offset_of!(CopyMemoryIndirectInfoKHR, p_next), 8);
        assert_eq!(offset_of!(CopyMemoryIndirectInfoKHR, src_copy_flags), 16);
        assert_eq!(offset_of!(CopyMemoryIndirectInfoKHR, dst_copy_flags), 20);
        assert_eq!(offset_of!(CopyMemoryIndirectInfoKHR, copy_count), 24);
        assert_eq!(
            offset_of!(CopyMemoryIndirectInfoKHR, copy_address_range),
            32
        );
    }

    #[test]
    fn defaults_and_non_local_source_use_khr_constants() {
        let features = CopyMemoryIndirectFeaturesKHR::default();
        let properties = CopyMemoryIndirectPropertiesKHR::default();
        let info = CopyMemoryIndirectInfoKHR {
            dst_copy_flags: ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR,
            ..Default::default()
        };
        assert_eq!(features.s_type.as_raw(), 1_000_549_000);
        assert_eq!(properties.s_type.as_raw(), 1_000_426_001);
        assert_eq!(info.s_type.as_raw(), 1_000_549_002);
        assert!(features.p_next.is_null() && properties.p_next.is_null() && info.p_next.is_null());
        assert_eq!(info.src_copy_flags, 0);
        assert_eq!(info.dst_copy_flags, 1);
        assert_eq!(ADDRESS_COPY_SPARSE_BIT_KHR, 2);
        assert_eq!(ADDRESS_COPY_PROTECTED_BIT_KHR, 4);
        assert_eq!(COPY_INDIRECT_STAGE_KHR.as_raw(), 1u64 << 46);
        assert_eq!(features.indirect_memory_copy, vk::FALSE);
        assert_eq!(features.indirect_memory_to_image_copy, vk::FALSE);
    }

    #[test]
    fn enable_is_memory_only_and_preserves_existing_device_chain() {
        let support = Support {
            indirect_memory_copy: true,
            ..Default::default()
        };
        let mut features = support.enable_features();
        assert_eq!(features.indirect_memory_copy, vk::TRUE);
        assert_eq!(features.indirect_memory_to_image_copy, vk::FALSE);
        let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
        let mut ci = vk::DeviceCreateInfo::default()
            .push_next(&mut bda)
            .push_next(&mut sync2);
        let old_head = ci.p_next;
        let raw_head = ptr::from_mut(&mut features).cast::<c_void>().cast_const();
        unsafe { crate::vkext::chain_into_device_ci(&mut ci, &mut features) };
        assert_eq!(ci.p_next, raw_head);
        assert_eq!(features.p_next.cast_const(), old_head);
        assert_eq!(sync2.p_next, ptr::from_mut(&mut bda).cast());
    }

    #[test]
    fn queue_gate_requires_feature_and_any_supported_queue_bit() {
        let queue = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        assert!(!Support::default().supports_queue(queue));
        let support = Support {
            indirect_memory_copy: true,
            supported_queues: vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            synchronization2: false,
        };
        assert!(support.supports_queue(queue));
        assert!(!support.supports_queue(vk::QueueFlags::GRAPHICS));
        assert!(!support.supports_queue(vk::QueueFlags::empty()));
        assert!(!Support {
            indirect_memory_copy: false,
            ..support
        }
        .supports_queue(queue));
        assert!(!Support {
            supported_queues: vk::QueueFlags::empty(),
            ..support
        }
        .supports_queue(queue));
    }

    #[test]
    fn probe_skips_absent_extension_and_reads_raw_chains_without_driver() {
        use std::sync::atomic::{AtomicU32, Ordering};

        static CALLS: AtomicU32 = AtomicU32::new(0);
        static API: AtomicU32 = AtomicU32::new(vk::API_VERSION_1_3);
        unsafe extern "system" fn properties(
            _: vk::PhysicalDevice,
            out: *mut vk::PhysicalDeviceProperties,
        ) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            out.write(vk::PhysicalDeviceProperties {
                api_version: API.load(Ordering::Relaxed),
                ..Default::default()
            });
        }
        unsafe extern "system" fn features2(
            _: vk::PhysicalDevice,
            out: *mut vk::PhysicalDeviceFeatures2<'_>,
        ) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            let mut node = (*out).p_next.cast::<vk::BaseOutStructure<'_>>();
            while !node.is_null() {
                if (*node).s_type.as_raw() == ST_FEATURES {
                    let features = &mut *node.cast::<CopyMemoryIndirectFeaturesKHR>();
                    features.indirect_memory_copy = vk::TRUE;
                    features.indirect_memory_to_image_copy = vk::TRUE;
                } else if (*node).s_type
                    == vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
                {
                    (*node.cast::<vk::PhysicalDeviceSynchronization2Features<'_>>())
                        .synchronization2 = vk::TRUE;
                }
                node = (*node).p_next;
            }
        }
        unsafe extern "system" fn properties2(
            _: vk::PhysicalDevice,
            out: *mut vk::PhysicalDeviceProperties2<'_>,
        ) {
            CALLS.fetch_add(1, Ordering::Relaxed);
            let properties = &mut *(*out).p_next.cast::<CopyMemoryIndirectPropertiesKHR>();
            if properties.s_type.as_raw() == ST_PROPERTIES {
                properties.supported_queues = vk::QueueFlags::COMPUTE;
            }
        }
        let instance = unsafe {
            ash::Instance::load_with(
                |name| match name.to_bytes() {
                    b"vkGetPhysicalDeviceProperties" => properties as *const () as *const c_void,
                    b"vkGetPhysicalDeviceFeatures2" => features2 as *const () as *const c_void,
                    b"vkGetPhysicalDeviceProperties2" => properties2 as *const () as *const c_void,
                    _ => ptr::null(),
                },
                vk::Instance::null(),
            )
        };
        let absent = unsafe { query(&instance, vk::PhysicalDevice::null(), false) };
        assert_eq!(CALLS.load(Ordering::Relaxed), 0);
        assert!(!absent.indirect_memory_copy && !absent.synchronization2);
        assert!(absent.supported_queues.is_empty());

        let supported = unsafe { query(&instance, vk::PhysicalDevice::null(), true) };
        assert_eq!(CALLS.load(Ordering::Relaxed), 3);
        assert!(supported.supports_queue(vk::QueueFlags::COMPUTE));
        assert!(!supported.supports_queue(vk::QueueFlags::TRANSFER));
        assert!(supported.synchronization2);
        assert_eq!(
            supported.enable_features().indirect_memory_to_image_copy,
            vk::FALSE
        );

        API.store(vk::API_VERSION_1_2, Ordering::Relaxed);
        let pre13 = unsafe { query(&instance, vk::PhysicalDevice::null(), true) };
        assert_eq!(CALLS.load(Ordering::Relaxed), 6);
        assert!(pre13.indirect_memory_copy);
        assert!(!pre13.synchronization2);
    }

    #[test]
    fn optional_loader_rejects_null_and_calls_exact_khr_signature() {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

        static STYPE: AtomicU32 = AtomicU32::new(0);
        static SRC_FLAGS: AtomicU32 = AtomicU32::new(u32::MAX);
        static DST_FLAGS: AtomicU32 = AtomicU32::new(u32::MAX);
        static COUNT: AtomicU32 = AtomicU32::new(0);
        static ADDRESS: AtomicU64 = AtomicU64::new(0);
        static STRIDE: AtomicU64 = AtomicU64::new(0);
        static SIZE: AtomicU64 = AtomicU64::new(0);
        unsafe extern "system" fn stub(
            _: vk::CommandBuffer,
            info: *const CopyMemoryIndirectInfoKHR,
        ) {
            STYPE.store((*info).s_type.as_raw() as u32, Ordering::Relaxed);
            SRC_FLAGS.store((*info).src_copy_flags, Ordering::Relaxed);
            DST_FLAGS.store((*info).dst_copy_flags, Ordering::Relaxed);
            COUNT.store((*info).copy_count, Ordering::Relaxed);
            ADDRESS.store((*info).copy_address_range.address, Ordering::Relaxed);
            STRIDE.store((*info).copy_address_range.stride, Ordering::Relaxed);
            SIZE.store((*info).copy_address_range.size, Ordering::Relaxed);
        }
        let queue = vk::QueueFlags::COMPUTE;
        let support = Support {
            indirect_memory_copy: true,
            supported_queues: queue,
            synchronization2: true,
        };
        let erased = unsafe {
            std::mem::transmute::<PfnCmdCopyMemoryIndirectKHR, unsafe extern "system" fn()>(stub)
        };
        assert!(unsafe { Device::from_proc_addr(support, queue, None) }.is_none());
        assert!(
            unsafe { Device::from_proc_addr(Support::default(), queue, Some(erased)) }.is_none()
        );
        assert!(
            unsafe { Device::from_proc_addr(support, vk::QueueFlags::TRANSFER, Some(erased)) }
                .is_none()
        );
        let loader = unsafe { Device::from_proc_addr(support, queue, Some(erased)) }.unwrap();
        assert_eq!(loader.supported_queues, queue);
        assert!(loader.synchronization2);
        unsafe { loader.record(vk::CommandBuffer::null(), 0x1000, 3, 32, 96) };
        assert_eq!(STYPE.load(Ordering::Relaxed), ST_COPY_INFO as u32);
        assert_eq!(SRC_FLAGS.load(Ordering::Relaxed), 0);
        assert_eq!(DST_FLAGS.load(Ordering::Relaxed), 0);
        assert_eq!(COUNT.load(Ordering::Relaxed), 3);
        assert_eq!(ADDRESS.load(Ordering::Relaxed), 0x1000);
        assert_eq!(STRIDE.load(Ordering::Relaxed), 32);
        assert_eq!(SIZE.load(Ordering::Relaxed), 96);
    }
}
