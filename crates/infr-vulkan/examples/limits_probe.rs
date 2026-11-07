//! Report the device limits that the flash-attention shaders implicitly assume (shared memory,
//! subgroup size) plus the fp16→fp32 16x16x16 coopmat config, to explain the NVIDIA device-lost.
use ash::vk;

fn main() {
    unsafe {
        let entry = ash::Entry::load().unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let inst_exts = [ash::khr::get_physical_device_properties2::NAME.as_ptr()];
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&inst_exts),
                None,
            )
            .unwrap();
        let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        for pd in instance.enumerate_physical_devices().unwrap() {
            let props = instance.get_physical_device_properties(pd);
            let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();

            let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sub);
            instance.get_physical_device_properties2(pd, &mut p2);

            let shared = props.limits.max_compute_shared_memory_size;
            println!("== {name} ==");
            println!(
                "  maxComputeSharedMemorySize = {shared} bytes ({} KiB)",
                shared / 1024
            );
            println!("  subgroupSize               = {}", sub.subgroup_size);
            println!(
                "  flash-warp shared need     = 58112 bytes (56.75 KiB)  -> {}",
                if 58112 > shared {
                    "OVER LIMIT (device-lost)"
                } else {
                    "ok"
                }
            );

            let v = cm
                .get_physical_device_cooperative_matrix_properties(pd)
                .unwrap();
            let has_1616 = v.iter().any(|p| {
                p.m_size == 16
                    && p.n_size == 16
                    && p.k_size == 16
                    && p.a_type == vk::ComponentTypeKHR::FLOAT16
                    && p.b_type == vk::ComponentTypeKHR::FLOAT16
                    && p.result_type == vk::ComponentTypeKHR::FLOAT32
                    && p.scope == vk::ScopeKHR::SUBGROUP
            });
            println!(
                "  fp16x16x16->fp32 coopmat    = {}",
                if has_1616 { "supported" } else { "MISSING" }
            );
        }
        instance.destroy_instance(None);
    }
}
