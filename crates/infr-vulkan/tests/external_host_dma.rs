//! DMA bandwidth probe for ordinary host allocations imported with
//! `VK_EXT_external_memory_host`.
//!
//! Unlike `BufferUsage::HostWeights`, these bytes are allocated by the process first and then
//! imported in place.  This is the storage shape needed by the bounded RAM expert cache: SSD fills
//! and cache metadata keep using the original CPU pointer while Vulkan can DMA directly from (or
//! back to) the same allocation without a second host mirror.
//!
//! Run on the GPU server:
//! `cargo test -p infr-vulkan --release --test external_host_dma -- --ignored --nocapture`

use ash::vk;
use std::ffi::{c_void, CString};
use std::time::{Duration, Instant};

const MATRIX_BYTES: usize = 3_072 * 1_024 / 256 * 144;
const MATRIX_GAP: usize = 4 * 1024;
const MATRIX_STRIDE: usize = MATRIX_BYTES + MATRIX_GAP;
const MAX_EXPERTS: usize = 8;
const MAX_MATRICES: usize = 3 * MAX_EXPERTS;
const BUFFER_BYTES: usize = MAX_MATRICES * MATRIX_STRIDE;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(
        address: *mut c_void,
        size: usize,
        allocation_type: u32,
        protect: u32,
    ) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, size: usize, free_type: u32) -> i32;
}

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 0x04;

struct HostAllocation {
    ptr: *mut u8,
    bytes: usize,
}

impl HostAllocation {
    fn new(bytes: usize, alignment: usize) -> Self {
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                bytes,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!ptr.is_null(), "VirtualAlloc({bytes}) failed");
        assert_eq!(
            ptr as usize % alignment,
            0,
            "VirtualAlloc pointer does not meet Vulkan host-import alignment"
        );
        Self { ptr, bytes }
    }

    unsafe fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        let ok = unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        assert_ne!(ok, 0, "VirtualFree failed");
    }
}

struct RawBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

unsafe fn imported_host_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    external_host: &ash::ext::external_memory_host::Device,
    host: &HostAllocation,
) -> RawBuffer {
    unsafe {
        try_imported_host_range(
            instance,
            device,
            physical,
            external_host,
            host.ptr,
            host.bytes,
        )
    }
    .expect("import ordinary host allocation as Vulkan memory")
}

unsafe fn try_imported_host_range(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    external_host: &ash::ext::external_memory_host::Device,
    host_ptr: *mut u8,
    host_bytes: usize,
) -> Result<RawBuffer, String> {
    let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
    let info = vk::BufferCreateInfo::default()
        .push_next(&mut external)
        .size(host_bytes as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }
        .map_err(|e| format!("create imported-host buffer: {e}"))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    if requirements.size > host_bytes as u64 {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(format!(
            "host range {host_bytes} is smaller than Vulkan requirement {}",
            requirements.size,
        ));
    }

    let mut host_properties = vk::MemoryHostPointerPropertiesEXT::default();
    let result = unsafe {
        (external_host.fp().get_memory_host_pointer_properties_ext)(
            device.handle(),
            handle_type,
            host_ptr.cast(),
            &mut host_properties,
        )
    };
    if result != vk::Result::SUCCESS {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(format!(
            "vkGetMemoryHostPointerPropertiesEXT failed: {result}"
        ));
    }

    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let compatible = requirements.memory_type_bits & host_properties.memory_type_bits;
    let Some(memory_type_index) = (0..memory_properties.memory_type_count).find(|&index| {
        let bit = 1u32 << index;
        let flags = memory_properties.memory_types[index as usize].property_flags;
        compatible & bit != 0
            && flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
    }) else {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(format!(
            "no coherent memory type for imported host allocation: buffer={:#x}, host={:#x}",
            requirements.memory_type_bits, host_properties.memory_type_bits,
        ));
    };
    let flags = memory_properties.memory_types[memory_type_index as usize].property_flags;

    let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
        .handle_type(handle_type)
        .host_pointer(host_ptr.cast());
    let allocation = vk::MemoryAllocateInfo::default()
        .push_next(&mut import)
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(format!("allocate imported host memory: {e}"));
        }
    };
    if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(format!("bind imported host allocation: {e}"));
    }
    println!(
        "imported host memory: ptr={:#x}, alignment={}, type={}, flags={flags:?}",
        host_ptr as usize, requirements.alignment, memory_type_index
    );
    Ok(RawBuffer { buffer, memory })
}

unsafe fn device_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    bytes: usize,
) -> RawBuffer {
    let info = vk::BufferCreateInfo::default()
        .size(bytes as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }.expect("create device buffer");
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let memory_type_index = (0..properties.memory_type_count)
        .find(|&index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && !properties.memory_types[index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        })
        .expect("find device-local memory type");
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.allocate_memory(&allocation, None) }.expect("allocate VRAM");
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("bind VRAM buffer");
    RawBuffer { buffer, memory }
}

unsafe fn rebar_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    bytes: usize,
) -> RawBuffer {
    let info = vk::BufferCreateInfo::default()
        .size(bytes as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }.expect("create ReBAR buffer");
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let want = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    let memory_type_index = (0..properties.memory_type_count)
        .find(|&index| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(want)
        })
        .expect("find DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT ReBAR memory type");
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.allocate_memory(&allocation, None) }.expect("allocate ReBAR VRAM");
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("bind ReBAR VRAM");
    RawBuffer { buffer, memory }
}

fn copy_regions(matrices: usize) -> Vec<vk::BufferCopy> {
    (0..matrices)
        .map(|matrix| {
            let offset = (matrix * MATRIX_STRIDE) as u64;
            vk::BufferCopy::default()
                .src_offset(offset)
                .dst_offset(offset)
                .size(MATRIX_BYTES as u64)
        })
        .collect()
}

unsafe fn record_copy(
    device: &ash::Device,
    pool: vk::CommandPool,
    src: vk::Buffer,
    dst: vk::Buffer,
    regions: &[vk::BufferCopy],
) -> vk::CommandBuffer {
    let command = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("allocate copy command")[0];
    unsafe { device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default()) }
        .expect("begin copy command");
    unsafe { device.cmd_copy_buffer(command, src, dst, regions) };
    unsafe { device.end_command_buffer(command) }.expect("end copy command");
    command
}

unsafe fn run_copy(
    device: &ash::Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    fence: vk::Fence,
) {
    unsafe { device.reset_fences(&[fence]) }.expect("reset copy fence");
    let commands = [command];
    let submit = vk::SubmitInfo::default().command_buffers(&commands);
    unsafe { device.queue_submit(queue, &[submit], fence) }.expect("submit copy");
    unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.expect("wait copy");
}

unsafe fn bench_copy(
    device: &ash::Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    iters: usize,
) -> Duration {
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .expect("create copy fence");
    for _ in 0..3 {
        unsafe { run_copy(device, queue, command, fence) };
    }
    let start = Instant::now();
    for _ in 0..iters {
        unsafe { run_copy(device, queue, command, fence) };
    }
    let elapsed = start.elapsed();
    unsafe { device.destroy_fence(fence, None) };
    elapsed
}

fn iterations(payload: usize) -> usize {
    ((1024 * 1024 * 1024usize) / payload).clamp(24, 192)
}

fn gib_s(payload: usize, iterations: usize, elapsed: Duration) -> f64 {
    payload as f64 * iterations as f64 / elapsed.as_secs_f64() / GIB
}

#[test]
#[ignore = "requires a Vulkan GPU with VK_EXT_external_memory_host"]
fn imported_ordinary_ram_dma_matrix() {
    unsafe {
        let entry = ash::Entry::load().expect("load Vulkan");
        let name = CString::new("infr-external-host-dma").unwrap();
        let application = vk::ApplicationInfo::default()
            .application_name(&name)
            .engine_name(&name)
            .api_version(vk::API_VERSION_1_2);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&application),
                None,
            )
            .expect("create Vulkan instance");
        let physical = instance
            .enumerate_physical_devices()
            .expect("enumerate Vulkan devices")
            .into_iter()
            .find(|&candidate| {
                instance
                    .get_physical_device_properties(candidate)
                    .device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .expect("find discrete Vulkan GPU");
        let extensions = instance
            .enumerate_device_extension_properties(physical)
            .expect("enumerate device extensions");
        assert!(extensions.iter().any(|extension| {
            std::ffi::CStr::from_ptr(extension.extension_name.as_ptr())
                == ash::ext::external_memory_host::NAME
        }));

        let queue_properties = instance.get_physical_device_queue_family_properties(physical);
        let family = queue_properties
            .iter()
            .position(|properties| properties.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .expect("find transfer-capable queue") as u32;
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extension_names = [ash::ext::external_memory_host::NAME.as_ptr()];
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&extension_names),
                None,
            )
            .expect("create Vulkan device with external-host import");
        let external_host = ash::ext::external_memory_host::Device::new(&instance, &device);
        let queue = device.get_device_queue(family, 0);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )
            .expect("create command pool");

        let mut external_properties = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut properties2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut external_properties);
        instance.get_physical_device_properties2(physical, &mut properties2);
        let alignment = external_properties.min_imported_host_pointer_alignment as usize;
        assert!(alignment > 0);
        assert_eq!(BUFFER_BYTES % alignment, 0);

        let mut h2d_host = HostAllocation::new(BUFFER_BYTES, alignment);
        let mut d2h_host = HostAllocation::new(BUFFER_BYTES, alignment);
        h2d_host.bytes_mut().fill(0xcd);
        d2h_host.bytes_mut().fill(0);
        for matrix in 0..MAX_MATRICES {
            let start = matrix * MATRIX_STRIDE;
            h2d_host.bytes_mut()[start..start + MATRIX_BYTES]
                .fill((matrix as u8).wrapping_mul(17).wrapping_add(3));
        }

        let h2d = imported_host_buffer(&instance, &device, physical, &external_host, &h2d_host);
        let d2h = imported_host_buffer(&instance, &device, physical, &external_host, &d2h_host);
        let gpu = device_buffer(&instance, &device, physical, BUFFER_BYTES);

        println!(
            "ordinary RAM import: allocation={:.2} MiB, min alignment={} bytes",
            BUFFER_BYTES as f64 / 1048576.0,
            alignment
        );
        println!("N  payload MiB  regions   H2D GiB/s   D2H GiB/s");
        for experts in 1..=MAX_EXPERTS {
            let matrices = 3 * experts;
            let payload = matrices * MATRIX_BYTES;
            let regions = copy_regions(matrices);
            let h2d_command = record_copy(&device, pool, h2d.buffer, gpu.buffer, &regions);
            let d2h_command = record_copy(&device, pool, gpu.buffer, d2h.buffer, &regions);
            let iters = iterations(payload);
            let h2d_elapsed = bench_copy(&device, queue, h2d_command, iters);
            let d2h_elapsed = bench_copy(&device, queue, d2h_command, iters);
            println!(
                "{experts:<2} {:>11.2} {:>8} {:>11.2} {:>11.2}",
                payload as f64 / 1048576.0,
                regions.len(),
                gib_s(payload, iters, h2d_elapsed),
                gib_s(payload, iters, d2h_elapsed),
            );
            device.free_command_buffers(pool, &[h2d_command, d2h_command]);
        }

        let all_regions = copy_regions(MAX_MATRICES);
        let roundtrip_h2d = record_copy(&device, pool, h2d.buffer, gpu.buffer, &all_regions);
        let roundtrip_d2h = record_copy(&device, pool, gpu.buffer, d2h.buffer, &all_regions);
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("create validation fence");
        run_copy(&device, queue, roundtrip_h2d, fence);
        run_copy(&device, queue, roundtrip_d2h, fence);
        device.destroy_fence(fence, None);
        for matrix in 0..MAX_MATRICES {
            let start = matrix * MATRIX_STRIDE;
            assert_eq!(
                &h2d_host.bytes_mut()[start..start + MATRIX_BYTES],
                &d2h_host.bytes_mut()[start..start + MATRIX_BYTES],
                "round-trip mismatch in matrix {matrix}"
            );
        }

        let rebar_gib = std::env::var("INFR_EXTERNAL_HOST_REBAR_GIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut rebar_buffers = Vec::new();
        let mut rebar_left = rebar_gib * 1024 * 1024 * 1024;
        while rebar_left > 0 {
            let bytes = rebar_left.min(2 * 1024 * 1024 * 1024);
            rebar_buffers.push(rebar_buffer(&instance, &device, physical, bytes));
            rebar_left -= bytes;
        }
        if rebar_gib > 0 {
            println!(
                "ReBAR preallocation: {rebar_gib} GiB in {} shard(s)",
                rebar_buffers.len(),
            );
        }

        let dummy_gib = std::env::var("INFR_EXTERNAL_HOST_DUMMY_GIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let _dummy_host =
            (dummy_gib > 0).then(|| HostAllocation::new(dummy_gib * 1024 * 1024 * 1024, alignment));
        if dummy_gib > 0 {
            println!("ordinary RAM preallocation: {dummy_gib} GiB (not imported)");
        }

        // Match the production arena shape: one large VirtualAlloc split into independent Vulkan
        // imports. This distinguishes aggregate capacity from a repeated-import platform limit.
        let single_capacity_gib = std::env::var("INFR_EXTERNAL_HOST_SINGLE_GIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let single_capacity_host = (single_capacity_gib > 0)
            .then(|| HostAllocation::new(single_capacity_gib * 1024 * 1024 * 1024, alignment));
        let mut single_capacity_buffers = Vec::new();
        if let Some(host) = single_capacity_host.as_ref() {
            let mut offset = 0usize;
            while offset < host.bytes {
                let bytes = (host.bytes - offset).min(2 * 1024 * 1024 * 1024);
                match try_imported_host_range(
                    &instance,
                    &device,
                    physical,
                    &external_host,
                    host.ptr.add(offset),
                    bytes,
                ) {
                    Ok(raw) => {
                        single_capacity_buffers.push(raw);
                        offset += bytes;
                    }
                    Err(err) => {
                        println!(
                            "single-allocation capacity stopped at {:.2}/{single_capacity_gib} GiB: {err}",
                            offset as f64 / GIB,
                        );
                        break;
                    }
                }
            }
            if offset == host.bytes {
                println!(
                    "single-allocation capacity: {single_capacity_gib} GiB in {} shard(s) succeeded",
                    single_capacity_buffers.len(),
                );
            }
        }

        // Optional capacity probe: import many independent 2-GiB views without touching their
        // pages. This validates the production sharding limit and aggregate host-arena scale while
        // keeping the ordinary ignored bandwidth test lightweight by default.
        let capacity_gib = std::env::var("INFR_EXTERNAL_HOST_CAPACITY_GIB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut capacity_hosts = Vec::new();
        let mut capacity_buffers = Vec::new();
        let mut capacity_left = capacity_gib * 1024 * 1024 * 1024;
        while capacity_left > 0 {
            let bytes = capacity_left.min(2 * 1024 * 1024 * 1024);
            capacity_hosts.push(HostAllocation::new(bytes, alignment));
            let host = capacity_hosts
                .last()
                .expect("capacity host was just appended");
            capacity_buffers.push(imported_host_buffer(
                &instance,
                &device,
                physical,
                &external_host,
                host,
            ));
            capacity_left -= bytes;
        }
        if capacity_gib > 0 {
            println!(
                "capacity import: {} GiB in {} shard(s) succeeded",
                capacity_gib,
                capacity_buffers.len(),
            );
        }

        device.device_wait_idle().expect("final device idle");
        device.free_command_buffers(pool, &[roundtrip_h2d, roundtrip_d2h]);
        for raw in single_capacity_buffers {
            device.destroy_buffer(raw.buffer, None);
            device.free_memory(raw.memory, None);
        }
        for raw in capacity_buffers {
            device.destroy_buffer(raw.buffer, None);
            device.free_memory(raw.memory, None);
        }
        for raw in rebar_buffers {
            device.destroy_buffer(raw.buffer, None);
            device.free_memory(raw.memory, None);
        }
        for raw in [h2d, d2h, gpu] {
            device.destroy_buffer(raw.buffer, None);
            device.free_memory(raw.memory, None);
        }
        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
