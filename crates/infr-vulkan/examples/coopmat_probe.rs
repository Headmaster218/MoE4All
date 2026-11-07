//! Enumerate the device's cooperative-matrix configurations to see whether RADV exposes an
//! int8×int8→int32 config (needed to decide: integer WMMA vs scalar dp4a for the mmq prefill GEMM).
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
            let v = cm
                .get_physical_device_cooperative_matrix_properties(pd)
                .unwrap();
            println!("== {name} : {} coopmat configs ==", v.len());
            let ty = |t: vk::ComponentTypeKHR| format!("{t:?}");
            for p in &v {
                println!(
                    "  {}x{}x{}  A={} B={} C={} R={} sat={} scope={:?}",
                    p.m_size,
                    p.n_size,
                    p.k_size,
                    ty(p.a_type),
                    ty(p.b_type),
                    ty(p.c_type),
                    ty(p.result_type),
                    p.saturating_accumulation,
                    p.scope,
                );
            }
        }
        instance.destroy_instance(None);
    }
}
