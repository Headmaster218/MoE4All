//! Linux-only probe: is the per-`vkQueueSubmit` cost a property of the ALIASED host range
//! (`VK_EXT_external_memory_host` -> userptr), or of host-visible memory in general?
//!
//! Rationale. A paged-MoE host tier aliased through `VK_EXT_external_memory_host` makes every
//! command submission cost ~6.8 ms per GiB of aliased range on amdgpu (measured on RX 7900 XTX
//! with RADV Mesa 26.0.8 and AMDVLK v-2025.Q2.1 alike), which is what turns Qwen3.8-Flash-Next
//! into 0.07 tok/s. The candidate fix is to hand the GPU the host tier as an ORDINARY host-visible
//! allocation instead of importing process memory (a normal GEM/GTT BO rather than a userptr BO).
//! This probe decides whether that actually avoids the cost, BEFORE any engine surgery.
//!
//! It allocates the same size twice — once imported, once as a plain `HOST_VISIBLE` allocation —
//! binds a buffer over each, submits a trivial `vkCmdFillBuffer` over it 64 times, and reports the
//! per-submit latency. The driver charges the imported range and not the plain one if the
//! hypothesis holds.
//!
//! Run (GPU must be free):
//! `cargo test -p infr-vulkan --release --test linux_host_submit_probe -- --ignored --nocapture`

#![cfg(target_os = "linux")]

use ash::vk;
use std::time::Instant;

const GIB: u64 = 1 << 30;
/// Sized so the prediction is unambiguous: at the measured 6.8 ms/GiB, an imported range of this
/// size should show tens of milliseconds per submit, while a plain allocation should show ~0.
const PROBE_BYTES: u64 = 4 * GIB;
const SUBMITS: usize = 64;

fn main() {
    // `#[test]`-less harness: run as a normal test binary body so `cargo test -- --ignored` finds it.
}

#[test]
#[ignore = "requires a Vulkan GPU; allocates several GiB and leaves the GPU busy"]
fn imported_alias_is_what_costs_each_submit() {
    unsafe { run() }
}

unsafe fn run() {
    let entry = unsafe { ash::Entry::load() }.expect("ash::Entry::load (libvulkan)");
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
    let instance = unsafe {
        entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .expect("create_instance")
    };

    let devices = unsafe { instance.enumerate_physical_devices() }.expect("enumerate");
    let physical = devices
        .into_iter()
        .find(|d| {
            let p = unsafe { instance.get_physical_device_properties(*d) };
            p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        })
        .expect("a discrete Vulkan GPU");

    let props = unsafe { instance.get_physical_device_properties(physical) };
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
    println!("device: {}", name.to_string_lossy());

    let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
        .iter()
        .position(|q| {
            q.queue_flags
                .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
        })
        .expect("a compute+transfer queue") as u32;

    // Who is this really?
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
    println!(
        "driver: {:?} {}",
        driver.driver_id,
        unsafe { std::ffi::CStr::from_ptr(driver.driver_info.as_ptr()) }.to_string_lossy()
    );

    let has_ext_host = {
        let ext = unsafe { instance.enumerate_device_extension_properties(physical) }
            .expect("extensions");
        ext.iter().any(|e| {
            let n = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            n.to_bytes() == b"VK_EXT_external_memory_host"
        })
    };
    assert!(has_ext_host, "VK_EXT_external_memory_host is required");

    let priority = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priority)];
    let exts = [ash::ext::external_memory_host::NAME.as_ptr()];
    let device = unsafe {
        instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&exts),
                None,
            )
            .expect("create_device")
    };
    let external_host = ash::ext::external_memory_host::Device::new(&instance, &device);
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
            None,
        )
    }
    .expect("command pool");
    let command = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("command buffer")[0];

    let fence =
        unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.expect("fence");

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

    let host = HostMapping::new(PROBE_BYTES as usize);
    println!(
        "\nprobe size: {:.2} GiB   submits: {SUBMITS}",
        PROBE_BYTES as f64 / GIB as f64
    );

    // --- (a) imported (userptr) ----------------------------------------------------------
    let imported = unsafe { make_imported(&device, &external_host, &mem_props, &host) }
        .expect("import host memory");
    let a = unsafe { time_submits(&device, queue, command, fence, imported.buffer) };
    unsafe { imported.destroy(&device) };

    // --- (b) plain host-visible allocation ------------------------------------------------
    let plain = unsafe { make_plain_host_visible(&device, &mem_props, PROBE_BYTES) }
        .expect("plain host-visible allocation");
    let b = unsafe { time_submits(&device, queue, command, fence, plain.buffer) };
    unsafe { plain.destroy(&device) };

    println!("\n  per-submit latency");
    println!("  imported (VK_EXT_external_memory_host / userptr) : {a:8.3} ms");
    println!("  plain host-visible (ordinary GTT BO)             : {b:8.3} ms");
    let verdict = if b * 4.0 < a {
        "HYPOTHESIS HOLDS: the alias is the cost -> a plain host-visible host tier should fix it"
    } else {
        "HYPOTHESIS FAILS: a plain host-visible tier is just as slow -> do NOT rewrite the import"
    };
    println!("  verdict: {verdict}");

    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}

/// Submits a trivial fill over `buffer` `SUBMITS` times and returns the mean submit+wait latency.
unsafe fn time_submits(
    device: &ash::Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    buffer: vk::Buffer,
) -> f64 {
    unsafe {
        device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
            )
            .expect("begin");
        // Touch only a tiny slice so this measures SUBMIT overhead, not the PCIe bandwidth of
        // pushing the whole range. Filling a whole host-visible (GTT) allocation is PCIe-bound
        // (~22 GB/s) and would mask the thing under test.
        device.cmd_fill_buffer(command, buffer, 0, 4096, 0);
        device.end_command_buffer(command).expect("end");

        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command));
        // Warm up once (pipeline/BO validation caches) so the first sample is not a one-off.
        device
            .queue_submit(queue, &[submit], fence)
            .expect("warmup submit");
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .expect("warmup wait");
        device.reset_fences(&[fence]).expect("reset");

        let start = Instant::now();
        for _ in 0..SUBMITS {
            device
                .queue_submit(queue, &[submit], fence)
                .expect("submit");
            device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("wait_for_fences");
            device.reset_fences(&[fence]).expect("reset");
        }
        start.elapsed().as_secs_f64() * 1000.0 / SUBMITS as f64
    }
}

struct RawBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}
impl RawBuffer {
    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

unsafe fn make_imported(
    device: &ash::Device,
    external_host: &ash::ext::external_memory_host::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    host: &HostMapping,
) -> Result<RawBuffer, String> {
    unsafe {
        let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
        let info = vk::BufferCreateInfo::default()
            .push_next(&mut external)
            .size(host.bytes as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device
            .create_buffer(&info, None)
            .map_err(|e| format!("create_buffer: {e}"))?;
        let req = device.get_buffer_memory_requirements(buffer);
        let mem_type = (0..mem_props.memory_type_count)
            .find(|i| {
                req.memory_type_bits & (1 << i) != 0
                    && mem_props.memory_types[*i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
            .ok_or("no host-visible memory type for the import")?;
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle_type)
            .host_pointer(host.ptr.cast());
        let allocation = vk::MemoryAllocateInfo::default()
            .push_next(&mut import)
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = device
            .allocate_memory(&allocation, None)
            .map_err(|e| format!("allocate imported memory: {e}"))?;
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("bind: {e}"))?;
        let _ = external_host; // the extension device is what validates the import at allocate time
        Ok(RawBuffer { buffer, memory })
    }
}

unsafe fn make_plain_host_visible(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    bytes: u64,
) -> Result<RawBuffer, String> {
    unsafe {
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device
            .create_buffer(&info, None)
            .map_err(|e| format!("create_buffer: {e}"))?;
        let req = device.get_buffer_memory_requirements(buffer);
        // Prefer a HOST_VISIBLE type that is NOT DEVICE_LOCAL: that is the ordinary GTT BO a host
        // cache would live in (DEVICE_LOCAL|HOST_VISIBLE would be ReBAR VRAM, a different thing).
        let pick = |want_device_local: bool| {
            (0..mem_props.memory_type_count).find(|i| {
                let f = mem_props.memory_types[*i as usize].property_flags;
                req.memory_type_bits & (1 << i) != 0
                    && f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) == want_device_local
            })
        };
        let mem_type = pick(false)
            .or_else(|| pick(true))
            .ok_or("no host-visible type")?;
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
            .map_err(|e| format!("allocate: {e}"))?;
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("bind: {e}"))?;
        Ok(RawBuffer { buffer, memory })
    }
}

/// A page-aligned anonymous host mapping the import can alias.
struct HostMapping {
    ptr: *mut u8,
    bytes: usize,
}
impl HostMapping {
    fn new(bytes: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(bytes, 4096).expect("layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "host allocation of {bytes} bytes failed");
        Self { ptr, bytes }
    }
}
