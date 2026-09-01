//! Windows Vulkan copy/compute overlap probe.
//!
//! This is deliberately an ignored diagnostic, not a product benchmark. It separates:
//! - CPU RAM copies and direct mapped-ReBAR pushes;
//! - imported-host H2D/D2H DMA;
//! - device-local buffer copies;
//! - CU-driven copies through a compute shader;
//! - copy/compute overlap on universal, same-family secondary, compute-only, and transfer-only
//!   queues;
//! - the pager-shaped `resident compute || miss copy -> miss compute` dependency chain.
//!
//! Run:
//! `cargo test -p infr-vulkan --release --test vulkan_overlap_matrix -- --ignored --nocapture`

#![cfg(target_os = "windows")]

use ash::vk;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ffi::{c_void, CStr, CString};
use std::process::Command;
use std::sync::Barrier;
use std::time::{Duration, Instant};

const MIB: usize = 1024 * 1024;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const MATRIX_BYTES: usize = 3_072 * 1_024 / 256 * 144;
const MATRIX_GAP: usize = 4 * 1024;
const MATRIX_STRIDE: usize = MATRIX_BYTES + MATRIX_GAP;
const MAX_EXPERTS: usize = 8;
const MAX_MATRICES: usize = 3 * MAX_EXPERTS;
const COPY_BUFFER_BYTES: usize = MAX_MATRICES * MATRIX_STRIDE;
const COMPUTE_SEGMENT_BYTES: usize = 128 * MIB;
const COMPUTE_SEGMENTS: usize = 4;
const COMPUTE_BUFFER_BYTES: usize = COMPUTE_SEGMENT_BYTES * COMPUTE_SEGMENTS;
const COMPUTE_GROUPS: u32 = 1024;
const WARMUPS: usize = 3;
const SAMPLES: usize = 9;

const READ_SHADER: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Src { uint src_words[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Dst { uint dst_words[]; };
layout(push_constant) uniform Params {
    uint base_word;
    uint word_count;
    uint salt;
    uint out_base;
} p;
shared uint partial[256];
void main() {
    uint lane = gl_LocalInvocationID.x;
    uint global = gl_GlobalInvocationID.x;
    uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    uint acc = p.salt ^ global;
    for (uint i = global; i < p.word_count; i += stride) {
        uint v = src_words[p.base_word + i];
        acc = (acc ^ v) * 1664525u + 1013904223u;
    }
    partial[lane] = acc;
    barrier();
    for (uint step = 128u; step != 0u; step >>= 1u) {
        if (lane < step) {
            partial[lane] ^= partial[lane + step];
        }
        barrier();
    }
    if (lane == 0u) {
        dst_words[p.out_base + gl_WorkGroupID.x] = partial[0];
    }
}
"#;

const COPY_SHADER: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Src { uint src_words[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Dst { uint dst_words[]; };
layout(push_constant) uniform Params {
    uint base_word;
    uint word_count;
    uint salt;
    uint out_base;
} p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.word_count) {
        dst_words[p.out_base + i] = src_words[p.base_word + i] ^ p.salt;
    }
}
"#;

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
        assert_eq!(ptr as usize % alignment, 0);
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
    mapped: Option<*mut u8>,
    bytes: usize,
}

impl RawBuffer {
    unsafe fn destroy(self, device: &ash::Device) {
        if self.mapped.is_some() {
            unsafe { device.unmap_memory(self.memory) };
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone, Copy)]
struct QueueRef {
    name: &'static str,
    family: u32,
    index: u32,
    queue: vk::Queue,
    flags: vk::QueueFlags,
    timestamp_bits: u32,
}

struct TimedCommand {
    cmd: vk::CommandBuffer,
    query: vk::QueryPool,
    family: u32,
    timestamp_bits: u32,
}

struct H2dCase {
    queue: QueueRef,
    experts: usize,
    command: TimedCommand,
    solo: SoloSample,
}

#[derive(Clone, Copy, Default)]
struct Interval {
    start: u64,
    end: u64,
    us: f64,
}

#[derive(Clone, Copy, Default)]
struct SoloSample {
    gpu_us: f64,
    wall_us: f64,
}

#[derive(Clone, Copy, Default)]
struct PairSample {
    a_us: f64,
    b_us: f64,
    span_us: f64,
    overlap_us: f64,
    wall_us: f64,
}

#[derive(Clone, Copy, Default)]
struct PipelineSample {
    hit_us: f64,
    copy_us: f64,
    miss_us: f64,
    span_us: f64,
    hidden_us: f64,
    wall_us: f64,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn median_solo(samples: &[SoloSample]) -> SoloSample {
    SoloSample {
        gpu_us: median(samples.iter().map(|v| v.gpu_us).collect()),
        wall_us: median(samples.iter().map(|v| v.wall_us).collect()),
    }
}

fn median_pair(samples: &[PairSample]) -> PairSample {
    PairSample {
        a_us: median(samples.iter().map(|v| v.a_us).collect()),
        b_us: median(samples.iter().map(|v| v.b_us).collect()),
        span_us: median(samples.iter().map(|v| v.span_us).collect()),
        overlap_us: median(samples.iter().map(|v| v.overlap_us).collect()),
        wall_us: median(samples.iter().map(|v| v.wall_us).collect()),
    }
}

fn median_pipeline(samples: &[PipelineSample]) -> PipelineSample {
    PipelineSample {
        hit_us: median(samples.iter().map(|v| v.hit_us).collect()),
        copy_us: median(samples.iter().map(|v| v.copy_us).collect()),
        miss_us: median(samples.iter().map(|v| v.miss_us).collect()),
        span_us: median(samples.iter().map(|v| v.span_us).collect()),
        hidden_us: median(samples.iter().map(|v| v.hidden_us).collect()),
        wall_us: median(samples.iter().map(|v| v.wall_us).collect()),
    }
}

fn parallel_copy(src: &[u8], dst: *mut u8) {
    const CHUNK: usize = 4 * MIB;
    if src.len() <= CHUNK {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let dst_addr = dst as usize;
    src.par_chunks(CHUNK)
        .enumerate()
        .for_each(|(i, chunk)| unsafe {
            std::ptr::copy_nonoverlapping(
                chunk.as_ptr(),
                (dst_addr + i * CHUNK) as *mut u8,
                chunk.len(),
            );
        });
}

fn bench_cpu_copy(src: &[u8], dst: *mut u8, parallel: bool) -> Duration {
    for _ in 0..WARMUPS {
        if parallel {
            parallel_copy(src, dst);
        } else {
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        }
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        if parallel {
            parallel_copy(src, dst);
        } else {
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        }
        samples.push(t0.elapsed().as_secs_f64());
    }
    Duration::from_secs_f64(median(samples))
}

fn compile_shader(name: &str, source: &str) -> Vec<u32> {
    let dir = std::env::temp_dir().join(format!("infr-vk-overlap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create shader temp directory");
    let src = dir.join(format!("{name}.comp"));
    let dst = dir.join(format!("{name}.spv"));
    std::fs::write(&src, source).expect("write overlap shader");
    let native_args = [
        "-O".to_string(),
        "--target-env=vulkan1.2".to_string(),
        src.to_string_lossy().into_owned(),
        "-o".to_string(),
        dst.to_string_lossy().into_owned(),
    ];
    let status = Command::new("glslc")
        .args(&native_args)
        .status()
        .expect("run glslc");
    assert!(status.success(), "glslc failed for {name}");
    let bytes = std::fs::read(dst).expect("read overlap SPIR-V");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

unsafe fn create_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    bytes: usize,
    usage: vk::BufferUsageFlags,
    required: vk::MemoryPropertyFlags,
    forbidden: vk::MemoryPropertyFlags,
    families: &[u32],
    map: bool,
) -> RawBuffer {
    let unique = unique_families(families);
    let mut info = vk::BufferCreateInfo::default()
        .size(bytes as u64)
        .usage(usage);
    if unique.len() > 1 {
        info = info
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&unique);
    } else {
        info = info.sharing_mode(vk::SharingMode::EXCLUSIVE);
    }
    let buffer = unsafe { device.create_buffer(&info, None) }.expect("create buffer");
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let properties = unsafe { instance.get_physical_device_memory_properties(physical) };
    let memory_type = (0..properties.memory_type_count)
        .find(|&index| {
            let flags = properties.memory_types[index as usize].property_flags;
            requirements.memory_type_bits & (1 << index) != 0
                && flags.contains(required)
                && !flags.intersects(forbidden)
        })
        .unwrap_or_else(|| {
            panic!(
                "no memory type for required={required:?}, forbidden={forbidden:?}, bits={:#x}",
                requirements.memory_type_bits
            )
        });
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe { device.allocate_memory(&allocation, None) }.expect("allocate buffer");
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("bind buffer");
    let mapped = if map {
        Some(
            unsafe { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty()) }
                .expect("map buffer") as *mut u8,
        )
    } else {
        None
    };
    RawBuffer {
        buffer,
        memory,
        mapped,
        bytes,
    }
}

unsafe fn import_host_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    external_host: &ash::ext::external_memory_host::Device,
    host: &HostAllocation,
    usage: vk::BufferUsageFlags,
    families: &[u32],
) -> RawBuffer {
    let handle_type = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
    let unique = unique_families(families);
    let mut info = vk::BufferCreateInfo::default()
        .push_next(&mut external)
        .size(host.bytes as u64)
        .usage(usage);
    if unique.len() > 1 {
        info = info
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&unique);
    } else {
        info = info.sharing_mode(vk::SharingMode::EXCLUSIVE);
    }
    let buffer = unsafe { device.create_buffer(&info, None) }.expect("create imported buffer");
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    assert!(requirements.size <= host.bytes as u64);
    let mut host_props = vk::MemoryHostPointerPropertiesEXT::default();
    let result = unsafe {
        (external_host.fp().get_memory_host_pointer_properties_ext)(
            device.handle(),
            handle_type,
            host.ptr.cast(),
            &mut host_props,
        )
    };
    assert_eq!(result, vk::Result::SUCCESS);
    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let compatible = requirements.memory_type_bits & host_props.memory_type_bits;
    let memory_type = (0..memory_props.memory_type_count)
        .find(|&index| {
            let flags = memory_props.memory_types[index as usize].property_flags;
            compatible & (1 << index) != 0
                && flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
        })
        .expect("find imported host memory type");
    let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
        .handle_type(handle_type)
        .host_pointer(host.ptr.cast());
    let allocation = vk::MemoryAllocateInfo::default()
        .push_next(&mut import)
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = unsafe { device.allocate_memory(&allocation, None) }
        .expect("allocate imported host memory");
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("bind imported host memory");
    RawBuffer {
        buffer,
        memory,
        mapped: None,
        bytes: host.bytes,
    }
}

fn unique_families(families: &[u32]) -> Vec<u32> {
    let mut out = families.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn expert_regions(experts: usize) -> Vec<vk::BufferCopy> {
    (0..experts * 3)
        .map(|matrix| {
            let offset = (matrix * MATRIX_STRIDE) as u64;
            vk::BufferCopy::default()
                .src_offset(offset)
                .dst_offset(offset)
                .size(MATRIX_BYTES as u64)
        })
        .collect()
}

fn payload_bytes(experts: usize) -> usize {
    experts * 3 * MATRIX_BYTES
}

unsafe fn create_pool(device: &ash::Device, family: u32) -> vk::CommandPool {
    unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
    }
    .expect("create command pool")
}

unsafe fn allocate_command(device: &ash::Device, pool: vk::CommandPool) -> vk::CommandBuffer {
    unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .expect("allocate command buffer")[0]
}

unsafe fn record_timed<F>(
    device: &ash::Device,
    pool: vk::CommandPool,
    family: u32,
    timestamp_bits: u32,
    record: F,
) -> TimedCommand
where
    F: FnOnce(vk::CommandBuffer),
{
    let query = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )
    }
    .expect("create timestamp query pool");
    let cmd = unsafe { allocate_command(device, pool) };
    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
        )
    }
    .expect("begin timed command");
    unsafe {
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, query, 0);
    }
    record(cmd);
    unsafe {
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query, 1);
        device.end_command_buffer(cmd)
    }
    .expect("end timed command");
    TimedCommand {
        cmd,
        query,
        family,
        timestamp_bits,
    }
}

unsafe fn reset_timed(device: &ash::Device, command: &TimedCommand) {
    unsafe { device.reset_query_pool(command.query, 0, 2) };
}

unsafe fn read_interval(
    device: &ash::Device,
    command: &TimedCommand,
    timestamp_period_ns: f64,
) -> Interval {
    let mut ticks = [0u64; 2];
    unsafe {
        device.get_query_pool_results(
            command.query,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )
    }
    .expect("read timestamp results");
    let bits = command.timestamp_bits;
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let start = ticks[0] & mask;
    let end = ticks[1] & mask;
    let delta = end.wrapping_sub(start) & mask;
    Interval {
        start,
        end,
        us: delta as f64 * timestamp_period_ns / 1000.0,
    }
}

unsafe fn destroy_timed(
    device: &ash::Device,
    pools: &BTreeMap<u32, vk::CommandPool>,
    cmd: TimedCommand,
) {
    unsafe {
        device.destroy_query_pool(cmd.query, None);
        device.free_command_buffers(pools[&cmd.family], &[cmd.cmd]);
    }
}

unsafe fn record_copy(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: QueueRef,
    src: vk::Buffer,
    dst: vk::Buffer,
    regions: &[vk::BufferCopy],
) -> TimedCommand {
    unsafe {
        record_timed(device, pool, queue.family, queue.timestamp_bits, |cmd| {
            device.cmd_copy_buffer(cmd, src, dst, regions);
        })
    }
}

struct ComputePipelines {
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    read_pipeline: vk::Pipeline,
    copy_pipeline: vk::Pipeline,
}

impl ComputePipelines {
    unsafe fn new(device: &ash::Device) -> Self {
        let bindings = [0u32, 1u32].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .expect("create descriptor layout");
        let set_layouts = [descriptor_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )
        }
        .expect("create pipeline layout");
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(16)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 32,
                    }]),
                None,
            )
        }
        .expect("create descriptor pool");
        let read_pipeline = unsafe {
            create_compute_pipeline(device, pipeline_layout, "overlap_read", READ_SHADER)
        };
        let copy_pipeline = unsafe {
            create_compute_pipeline(device, pipeline_layout, "overlap_copy", COPY_SHADER)
        };
        Self {
            descriptor_layout,
            pipeline_layout,
            descriptor_pool,
            read_pipeline,
            copy_pipeline,
        }
    }

    unsafe fn descriptor_set(
        &self,
        device: &ash::Device,
        src: &RawBuffer,
        dst: &RawBuffer,
    ) -> vk::DescriptorSet {
        let layouts = [self.descriptor_layout];
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .expect("allocate descriptor set")[0];
        let src_info = [vk::DescriptorBufferInfo::default()
            .buffer(src.buffer)
            .offset(0)
            .range(src.bytes as u64)];
        let dst_info = [vk::DescriptorBufferInfo::default()
            .buffer(dst.buffer)
            .offset(0)
            .range(dst.bytes as u64)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&src_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&dst_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        set
    }

    unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.read_pipeline, None);
            device.destroy_pipeline(self.copy_pipeline, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

unsafe fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    name: &str,
    source: &str,
) -> vk::Pipeline {
    let words = compile_shader(name, source);
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
    }
    .expect("create overlap shader module");
    let main = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&main);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)],
            None,
        )
    }
    .map_err(|(_, err)| err)
    .expect("create overlap compute pipeline")[0];
    unsafe { device.destroy_shader_module(module, None) };
    pipeline
}

unsafe fn record_read_compute(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: QueueRef,
    pipelines: &ComputePipelines,
    set: vk::DescriptorSet,
    segments: usize,
    output_base: u32,
) -> TimedCommand {
    unsafe {
        record_timed(device, pool, queue.family, queue.timestamp_bits, |cmd| {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.read_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.pipeline_layout,
                0,
                &[set],
                &[],
            );
            for segment in 0..segments {
                let params = [
                    (segment * COMPUTE_SEGMENT_BYTES / 4) as u32,
                    (COMPUTE_SEGMENT_BYTES / 4) as u32,
                    0x9e37_79b9u32.wrapping_mul(segment as u32 + 1),
                    output_base + segment as u32 * COMPUTE_GROUPS,
                ];
                device.cmd_push_constants(
                    cmd,
                    pipelines.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::cast_slice(&params),
                );
                device.cmd_dispatch(cmd, COMPUTE_GROUPS, 1, 1);
            }
        })
    }
}

unsafe fn record_cu_copy(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: QueueRef,
    pipelines: &ComputePipelines,
    set: vk::DescriptorSet,
    bytes: usize,
) -> TimedCommand {
    assert_eq!(bytes % 4, 0);
    unsafe {
        record_timed(device, pool, queue.family, queue.timestamp_bits, |cmd| {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.copy_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.pipeline_layout,
                0,
                &[set],
                &[],
            );
            let params = [0u32, (bytes / 4) as u32, 0x51ed_270b, 0u32];
            device.cmd_push_constants(
                cmd,
                pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            device.cmd_dispatch(cmd, ((bytes / 4) as u32).div_ceil(256), 1, 1);
        })
    }
}

unsafe fn submit_waiting_on_gate(
    device: &ash::Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    gate: vk::Semaphore,
    value: u64,
    fence: vk::Fence,
) {
    let commands = [command];
    let waits = [gate];
    let values = [value];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .push_next(&mut timeline);
    unsafe { device.queue_submit(queue, &[submit], fence) }.expect("submit gated command");
}

unsafe fn create_timeline(device: &ash::Device, initial: u64) -> vk::Semaphore {
    let mut ty = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(initial);
    unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None) }
        .expect("create timeline semaphore")
}

unsafe fn signal_timeline(device: &ash::Device, semaphore: vk::Semaphore, value: u64) {
    unsafe {
        device.signal_semaphore(
            &vk::SemaphoreSignalInfo::default()
                .semaphore(semaphore)
                .value(value),
        )
    }
    .expect("host signal timeline semaphore");
}

unsafe fn run_solo(
    device: &ash::Device,
    queue: QueueRef,
    command: &TimedCommand,
    period_ns: f64,
) -> SoloSample {
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .expect("create solo fence");
    let mut samples = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUPS + SAMPLES {
        unsafe { device.reset_fences(&[fence]) }.expect("reset solo fence");
        unsafe { reset_timed(device, command) };
        let commands = [command.cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&commands);
        let t0 = Instant::now();
        unsafe { device.queue_submit(queue.queue, &[submit], fence) }.expect("submit solo command");
        unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.expect("wait solo fence");
        let wall_us = t0.elapsed().as_secs_f64() * 1e6;
        let interval = unsafe { read_interval(device, command, period_ns) };
        if sample >= WARMUPS {
            samples.push(SoloSample {
                gpu_us: interval.us,
                wall_us,
            });
        }
    }
    unsafe { device.destroy_fence(fence, None) };
    median_solo(&samples)
}

unsafe fn run_pair(
    device: &ash::Device,
    a_queue: QueueRef,
    a: &TimedCommand,
    b_queue: QueueRef,
    b: &TimedCommand,
    period_ns: f64,
) -> PairSample {
    assert_eq!(
        a.timestamp_bits, b.timestamp_bits,
        "timestamp domains differ"
    );
    assert!(
        a.timestamp_bits >= 48,
        "timestamp counter too narrow for cross-queue comparison"
    );
    let gate = unsafe { create_timeline(device, 0) };
    let fence_a = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .expect("create pair fence A");
    let fence_b = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .expect("create pair fence B");
    let mut samples = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUPS + SAMPLES {
        let value = sample as u64 + 1;
        unsafe { device.reset_fences(&[fence_a, fence_b]) }.expect("reset pair fences");
        unsafe {
            reset_timed(device, a);
            reset_timed(device, b);
        }
        unsafe { submit_waiting_on_gate(device, a_queue.queue, a.cmd, gate, value, fence_a) };
        unsafe { submit_waiting_on_gate(device, b_queue.queue, b.cmd, gate, value, fence_b) };
        let t0 = Instant::now();
        unsafe { signal_timeline(device, gate, value) };
        unsafe { device.wait_for_fences(&[fence_a, fence_b], true, u64::MAX) }
            .expect("wait pair fences");
        let wall_us = t0.elapsed().as_secs_f64() * 1e6;
        let ai = unsafe { read_interval(device, a, period_ns) };
        let bi = unsafe { read_interval(device, b, period_ns) };
        let start = ai.start.min(bi.start);
        let end = ai.end.max(bi.end);
        let overlap_start = ai.start.max(bi.start);
        let overlap_end = ai.end.min(bi.end);
        let span_us = end.saturating_sub(start) as f64 * period_ns / 1000.0;
        let overlap_us = overlap_end.saturating_sub(overlap_start) as f64 * period_ns / 1000.0;
        if sample >= WARMUPS {
            samples.push(PairSample {
                a_us: ai.us,
                b_us: bi.us,
                span_us,
                overlap_us,
                wall_us,
            });
        }
    }
    unsafe {
        device.destroy_fence(fence_a, None);
        device.destroy_fence(fence_b, None);
        device.destroy_semaphore(gate, None);
    }
    median_pair(&samples)
}

unsafe fn submit_gate_and_signal(
    device: &ash::Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    gate: vk::Semaphore,
    gate_value: u64,
    done: vk::Semaphore,
    done_value: u64,
) {
    let commands = [command];
    let waits = [gate];
    let wait_values = [gate_value];
    let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let signals = [done];
    let signal_values = [done_value];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values)
        .signal_semaphore_values(&signal_values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&wait_stages)
        .signal_semaphores(&signals)
        .push_next(&mut timeline);
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }
        .expect("submit gated signaling command");
}

unsafe fn submit_waiting_on_done(
    device: &ash::Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    done: vk::Semaphore,
    value: u64,
    fence: vk::Fence,
) {
    let commands = [command];
    let waits = [done];
    let values = [value];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .push_next(&mut timeline);
    unsafe { device.queue_submit(queue, &[submit], fence) }.expect("submit done-waiting command");
}

unsafe fn run_pipeline(
    device: &ash::Device,
    compute_queue: QueueRef,
    hit: &TimedCommand,
    copy_queue: QueueRef,
    copy: &TimedCommand,
    miss: &TimedCommand,
    period_ns: f64,
) -> PipelineSample {
    assert_eq!(hit.timestamp_bits, copy.timestamp_bits);
    assert_eq!(hit.timestamp_bits, miss.timestamp_bits);
    let gate = unsafe { create_timeline(device, 0) };
    let done = unsafe { create_timeline(device, 0) };
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .expect("create pipeline fence");
    let mut samples = Vec::with_capacity(SAMPLES);
    for sample in 0..WARMUPS + SAMPLES {
        let value = sample as u64 + 1;
        unsafe { device.reset_fences(&[fence]) }.expect("reset pipeline fence");
        unsafe {
            reset_timed(device, hit);
            reset_timed(device, copy);
            reset_timed(device, miss);
            submit_waiting_on_gate(
                device,
                compute_queue.queue,
                hit.cmd,
                gate,
                value,
                vk::Fence::null(),
            );
            submit_gate_and_signal(device, copy_queue.queue, copy.cmd, gate, value, done, value);
            submit_waiting_on_done(device, compute_queue.queue, miss.cmd, done, value, fence);
        }
        let t0 = Instant::now();
        unsafe { signal_timeline(device, gate, value) };
        unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.expect("wait pipeline fence");
        let wall_us = t0.elapsed().as_secs_f64() * 1e6;
        let hi = unsafe { read_interval(device, hit, period_ns) };
        let ci = unsafe { read_interval(device, copy, period_ns) };
        let mi = unsafe { read_interval(device, miss, period_ns) };
        let start = hi.start.min(ci.start).min(mi.start);
        let end = hi.end.max(ci.end).max(mi.end);
        let span_us = end.saturating_sub(start) as f64 * period_ns / 1000.0;
        let serial_us = hi.us + ci.us + mi.us;
        if sample >= WARMUPS {
            samples.push(PipelineSample {
                hit_us: hi.us,
                copy_us: ci.us,
                miss_us: mi.us,
                span_us,
                hidden_us: (serial_us - span_us).max(0.0),
                wall_us,
            });
        }
    }
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_semaphore(done, None);
        device.destroy_semaphore(gate, None);
    }
    median_pipeline(&samples)
}

unsafe fn fill_buffer(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    buffer: vk::Buffer,
    bytes: usize,
) {
    let cmd = unsafe { allocate_command(device, pool) };
    unsafe { device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()) }
        .expect("begin fill command");
    unsafe { device.cmd_fill_buffer(cmd, buffer, 0, bytes as u64, 0x7f4a_7c15) };
    unsafe { device.end_command_buffer(cmd) }.expect("end fill command");
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .expect("create fill fence");
    let commands = [cmd];
    let submit = vk::SubmitInfo::default().command_buffers(&commands);
    unsafe { device.queue_submit(queue, &[submit], fence) }.expect("submit fill");
    unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.expect("wait fill");
    unsafe {
        device.destroy_fence(fence, None);
        device.free_command_buffers(pool, &[cmd]);
    }
}

fn print_solo(label: &str, queue: QueueRef, bytes: usize, sample: SoloSample) {
    let gib_s = bytes as f64 / (sample.gpu_us * 1e-6) / GIB;
    println!(
        "SOLO  {label:<18} q={:<18} payload={:>6.2} MiB gpu={:>8.1} us wall={:>8.1} us bw={:>6.2} GiB/s",
        queue.name,
        bytes as f64 / MIB as f64,
        sample.gpu_us,
        sample.wall_us,
        gib_s,
    );
}

fn print_pair(
    label: &str,
    a_queue: QueueRef,
    b_queue: QueueRef,
    standalone_a: f64,
    standalone_b: f64,
    sample: PairSample,
) {
    let min_live = sample.a_us.min(sample.b_us);
    let overlap_pct = if min_live > 0.0 {
        100.0 * sample.overlap_us / min_live
    } else {
        0.0
    };
    let saved_pct =
        100.0 * (standalone_a + standalone_b - sample.span_us) / (standalone_a + standalone_b);
    println!(
        "PAIR  {label:<20} {:<16}+{:<16} A={:>7.1} B={:>7.1} span={:>7.1} overlap={:>6.1}% saved={:>6.1}% slowA={:>4.2}x slowB={:>4.2}x wall={:>7.1}",
        a_queue.name,
        b_queue.name,
        sample.a_us,
        sample.b_us,
        sample.span_us,
        overlap_pct,
        saved_pct,
        sample.a_us / standalone_a,
        sample.b_us / standalone_b,
        sample.wall_us,
    );
}

fn print_pipeline(label: &str, copy_queue: QueueRef, sample: PipelineSample) {
    println!(
        "PIPE  {label:<20} copy_q={:<18} hit={:>7.1} copy={:>7.1} miss={:>7.1} span={:>7.1} hidden={:>7.1} ({:>5.1}%) wall={:>7.1}",
        copy_queue.name,
        sample.hit_us,
        sample.copy_us,
        sample.miss_us,
        sample.span_us,
        sample.hidden_us,
        100.0 * sample.hidden_us / (sample.hit_us + sample.copy_us + sample.miss_us),
        sample.wall_us,
    );
}

#[test]
#[ignore = "requires a discrete Vulkan GPU and disturbs GPU performance state"]
fn windows_vulkan_overlap_matrix() {
    unsafe {
        let entry = ash::Entry::load().expect("load Vulkan");
        let app_name = CString::new("infr-vulkan-overlap-matrix").unwrap();
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&app_name)
            .api_version(vk::API_VERSION_1_2);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
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
        let props = instance.get_physical_device_properties(physical);
        let device_name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
        let queue_props = instance.get_physical_device_queue_family_properties(physical);
        println!(
            "DEVICE {device_name} timestamp_period={} ns",
            props.limits.timestamp_period
        );
        for (index, family) in queue_props.iter().enumerate() {
            println!(
                "QF {index}: count={} flags={:?} timestamp_bits={}",
                family.queue_count, family.queue_flags, family.timestamp_valid_bits
            );
        }

        let compute_family = queue_props
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .expect("find compute queue family") as u32;
        let compute_only_family = queue_props.iter().enumerate().find_map(|(index, q)| {
            (q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .then_some(index as u32)
        });
        let transfer_only_family = queue_props.iter().enumerate().find_map(|(index, q)| {
            (q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !q
                    .queue_flags
                    .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
            .then_some(index as u32)
        });
        let transfer_only_family = transfer_only_family.expect("find transfer-only queue family");

        let extensions = instance
            .enumerate_device_extension_properties(physical)
            .expect("enumerate device extensions");
        assert!(extensions.iter().any(|ext| {
            CStr::from_ptr(ext.extension_name.as_ptr()) == ash::ext::external_memory_host::NAME
        }));
        assert!(extensions.iter().any(|ext| {
            CStr::from_ptr(ext.extension_name.as_ptr()) == ash::ext::calibrated_timestamps::NAME
        }));
        let mut timeline_support = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut host_query_reset_support = vk::PhysicalDeviceHostQueryResetFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut timeline_support)
            .push_next(&mut host_query_reset_support);
        instance.get_physical_device_features2(physical, &mut features);
        assert_ne!(
            timeline_support.timeline_semaphore, 0,
            "timeline semaphores unsupported"
        );
        assert_ne!(
            host_query_reset_support.host_query_reset, 0,
            "host query reset unsupported"
        );

        let mut requested: BTreeMap<u32, u32> = BTreeMap::new();
        requested.insert(
            compute_family,
            queue_props[compute_family as usize].queue_count.min(2),
        );
        if let Some(family) = compute_only_family {
            requested.entry(family).or_insert(1);
        }
        requested.entry(transfer_only_family).or_insert(1);
        let priority_storage: Vec<(u32, Vec<f32>)> = requested
            .iter()
            .map(|(&family, &count)| (family, vec![1.0; count as usize]))
            .collect();
        let queue_infos: Vec<_> = priority_storage
            .iter()
            .map(|(family, priorities)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*family)
                    .queue_priorities(priorities)
            })
            .collect();
        let extension_names = [
            ash::ext::external_memory_host::NAME.as_ptr(),
            ash::ext::calibrated_timestamps::NAME.as_ptr(),
        ];
        let mut timeline_enable =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut host_query_reset_enable =
            vk::PhysicalDeviceHostQueryResetFeatures::default().host_query_reset(true);
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&extension_names)
                    .push_next(&mut timeline_enable)
                    .push_next(&mut host_query_reset_enable),
                None,
            )
            .expect("create Vulkan device");
        let external_host = ash::ext::external_memory_host::Device::new(&instance, &device);
        let qref = |name, family: u32, index: u32| QueueRef {
            name,
            family,
            index,
            queue: device.get_device_queue(family, index),
            flags: queue_props[family as usize].queue_flags,
            timestamp_bits: queue_props[family as usize].timestamp_valid_bits,
        };
        let main = qref("universal-main", compute_family, 0);
        let same_family = (queue_props[compute_family as usize].queue_count >= 2)
            .then(|| qref("universal-q1", compute_family, 1));
        let compute_only = compute_only_family
            .filter(|&family| family != compute_family)
            .map(|family| qref("compute-only", family, 0));
        let transfer_only = qref("transfer-only", transfer_only_family, 0);
        let mut queues = vec![main];
        if let Some(queue) = same_family {
            queues.push(queue);
        }
        if let Some(queue) = compute_only {
            queues.push(queue);
        }
        queues.push(transfer_only);
        println!("QUEUES");
        for queue in &queues {
            println!(
                "  {} family={} index={} flags={:?}",
                queue.name, queue.family, queue.index, queue.flags
            );
        }
        let all_families = unique_families(&queues.iter().map(|q| q.family).collect::<Vec<_>>());
        let mut pools = BTreeMap::new();
        for &family in &all_families {
            pools.insert(family, create_pool(&device, family));
        }

        let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host_props);
        instance.get_physical_device_properties2(physical, &mut props2);
        let host_alignment = host_props.min_imported_host_pointer_alignment as usize;
        println!("external host alignment={host_alignment} bytes");

        let common_usage = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut h2d_host = HostAllocation::new(COPY_BUFFER_BYTES, host_alignment);
        let mut d2h_host = HostAllocation::new(COPY_BUFFER_BYTES, host_alignment);
        let mut ram_copy_dst = HostAllocation::new(COPY_BUFFER_BYTES, host_alignment);
        for (i, byte) in h2d_host.bytes_mut().iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
        }
        d2h_host.bytes_mut().fill(0);
        ram_copy_dst.bytes_mut().fill(0);
        let h2d_import = import_host_buffer(
            &instance,
            &device,
            physical,
            &external_host,
            &h2d_host,
            common_usage,
            &all_families,
        );
        let d2h_import = import_host_buffer(
            &instance,
            &device,
            physical,
            &external_host,
            &d2h_host,
            common_usage,
            &all_families,
        );
        let staging = create_buffer(
            &instance,
            &device,
            physical,
            COPY_BUFFER_BYTES,
            common_usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &all_families,
            true,
        );
        std::ptr::copy_nonoverlapping(
            h2d_host.ptr,
            staging.mapped.expect("mapped staging"),
            COPY_BUFFER_BYTES,
        );
        let rebar = create_buffer(
            &instance,
            &device,
            physical,
            COPY_BUFFER_BYTES,
            common_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::empty(),
            &all_families,
            true,
        );
        let h2d_dst = create_buffer(
            &instance,
            &device,
            physical,
            COPY_BUFFER_BYTES,
            common_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            &all_families,
            false,
        );
        let d2h_src = create_buffer(
            &instance,
            &device,
            physical,
            COPY_BUFFER_BYTES,
            common_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            &all_families,
            false,
        );
        let vram_copy_dst = create_buffer(
            &instance,
            &device,
            physical,
            COPY_BUFFER_BYTES,
            common_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            &all_families,
            false,
        );
        let cu_copy_dst = create_buffer(
            &instance,
            &device,
            physical,
            COPY_BUFFER_BYTES,
            common_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            &all_families,
            false,
        );
        let compute_src = create_buffer(
            &instance,
            &device,
            physical,
            COMPUTE_BUFFER_BYTES,
            common_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            &all_families,
            false,
        );
        let compute_out = create_buffer(
            &instance,
            &device,
            physical,
            COMPUTE_GROUPS as usize * 16 * 4,
            common_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            &all_families,
            false,
        );
        fill_buffer(
            &device,
            pools[&main.family],
            main.queue,
            compute_src.buffer,
            compute_src.bytes,
        );
        fill_buffer(
            &device,
            pools[&main.family],
            main.queue,
            d2h_src.buffer,
            d2h_src.bytes,
        );

        println!(
            "\nCPU COPY / PUSH (payload {:.2} MiB)",
            COPY_BUFFER_BYTES as f64 / MIB as f64
        );
        let source = std::slice::from_raw_parts(h2d_host.ptr, COPY_BUFFER_BYTES);
        for (name, ptr, parallel) in [
            ("RAM->RAM single", ram_copy_dst.ptr, false),
            ("RAM->RAM parallel", ram_copy_dst.ptr, true),
            ("RAM->staging single", staging.mapped.unwrap(), false),
            ("RAM->staging parallel", staging.mapped.unwrap(), true),
            ("RAM->ReBAR single", rebar.mapped.unwrap(), false),
            ("RAM->ReBAR parallel", rebar.mapped.unwrap(), true),
        ] {
            let elapsed = bench_cpu_copy(source, ptr, parallel);
            println!(
                "CPU   {name:<22} {:>8.1} us {:>6.2} GiB/s",
                elapsed.as_secs_f64() * 1e6,
                COPY_BUFFER_BYTES as f64 / elapsed.as_secs_f64() / GIB,
            );
        }

        let pipelines = ComputePipelines::new(&device);
        let read_set = pipelines.descriptor_set(&device, &compute_src, &compute_out);
        let cu_h2d_set = pipelines.descriptor_set(&device, &h2d_import, &cu_copy_dst);
        let cu_vram_set = pipelines.descriptor_set(&device, &d2h_src, &cu_copy_dst);
        let period_ns = props.limits.timestamp_period as f64;

        let compute_short = record_read_compute(
            &device,
            pools[&main.family],
            main,
            &pipelines,
            read_set,
            1,
            0,
        );
        let compute_medium = record_read_compute(
            &device,
            pools[&main.family],
            main,
            &pipelines,
            read_set,
            2,
            COMPUTE_GROUPS * 4,
        );
        let compute_long = record_read_compute(
            &device,
            pools[&main.family],
            main,
            &pipelines,
            read_set,
            4,
            COMPUTE_GROUPS * 8,
        );
        let compute_miss = record_read_compute(
            &device,
            pools[&main.family],
            main,
            &pipelines,
            read_set,
            1,
            COMPUTE_GROUPS * 12,
        );
        // Bring the device out of its idle clock state before collecting short samples.
        let _ = run_solo(&device, main, &compute_long, period_ns);
        let short_solo = run_solo(&device, main, &compute_short, period_ns);
        let medium_solo = run_solo(&device, main, &compute_medium, period_ns);
        let long_solo = run_solo(&device, main, &compute_long, period_ns);
        println!("\nSYNTHETIC COMPUTE (read/reduce, disjoint 128 MiB segments)");
        print_solo("compute-128MiB", main, COMPUTE_SEGMENT_BYTES, short_solo);
        print_solo(
            "compute-256MiB",
            main,
            2 * COMPUTE_SEGMENT_BYTES,
            medium_solo,
        );
        print_solo("compute-512MiB", main, 4 * COMPUTE_SEGMENT_BYTES, long_solo);

        println!("\nCOPY PATHS");
        let copy_queues: Vec<QueueRef> = queues
            .iter()
            .copied()
            .filter(|q| q.flags.contains(vk::QueueFlags::TRANSFER))
            .collect();
        let mut retained = Vec::<TimedCommand>::new();
        let mut h2d_commands = Vec::<H2dCase>::new();
        let mut d2h_40_commands = Vec::<(QueueRef, TimedCommand, SoloSample)>::new();
        let mut vram_40_commands = Vec::<(QueueRef, TimedCommand, SoloSample)>::new();
        for &queue in &copy_queues {
            for experts in [1usize, 4, 8] {
                let regions = expert_regions(experts);
                let bytes = payload_bytes(experts);
                let h2d = record_copy(
                    &device,
                    pools[&queue.family],
                    queue,
                    h2d_import.buffer,
                    h2d_dst.buffer,
                    &regions,
                );
                let h2d_solo = run_solo(&device, queue, &h2d, period_ns);
                print_solo("H2D imported", queue, bytes, h2d_solo);
                h2d_commands.push(H2dCase {
                    queue,
                    experts,
                    command: h2d,
                    solo: h2d_solo,
                });
                let d2h = record_copy(
                    &device,
                    pools[&queue.family],
                    queue,
                    d2h_src.buffer,
                    d2h_import.buffer,
                    &regions,
                );
                let d2h_solo = run_solo(&device, queue, &d2h, period_ns);
                print_solo("D2H imported", queue, bytes, d2h_solo);
                if experts == 8 {
                    d2h_40_commands.push((queue, d2h, d2h_solo));
                } else {
                    retained.push(d2h);
                }
                let vram = record_copy(
                    &device,
                    pools[&queue.family],
                    queue,
                    d2h_src.buffer,
                    vram_copy_dst.buffer,
                    &regions,
                );
                let vram_solo = run_solo(&device, queue, &vram, period_ns);
                print_solo("VRAM->VRAM", queue, bytes, vram_solo);
                if experts == 8 {
                    vram_40_commands.push((queue, vram, vram_solo));
                } else {
                    retained.push(vram);
                }
            }
            let regions = expert_regions(8);
            let staging_h2d = record_copy(
                &device,
                pools[&queue.family],
                queue,
                staging.buffer,
                h2d_dst.buffer,
                &regions,
            );
            let staging_solo = run_solo(&device, queue, &staging_h2d, period_ns);
            print_solo("H2D staging", queue, payload_bytes(8), staging_solo);
            retained.push(staging_h2d);
            let rebar_copy = record_copy(
                &device,
                pools[&queue.family],
                queue,
                rebar.buffer,
                h2d_dst.buffer,
                &regions,
            );
            let rebar_solo = run_solo(&device, queue, &rebar_copy, period_ns);
            print_solo("ReBAR->VRAM", queue, payload_bytes(8), rebar_solo);
            retained.push(rebar_copy);
        }

        println!("\nCU COPY PATHS (40.50 MiB contiguous shader copy)");
        let cu_bytes = payload_bytes(8);
        let compute_queues: Vec<QueueRef> = queues
            .iter()
            .copied()
            .filter(|q| q.flags.contains(vk::QueueFlags::COMPUTE))
            .collect();
        let mut cu_h2d_commands = Vec::<(QueueRef, TimedCommand, SoloSample)>::new();
        let mut cu_vram_commands = Vec::<(QueueRef, TimedCommand, SoloSample)>::new();
        for queue in compute_queues {
            let cu_h2d = record_cu_copy(
                &device,
                pools[&queue.family],
                queue,
                &pipelines,
                cu_h2d_set,
                cu_bytes,
            );
            let h2d_solo = run_solo(&device, queue, &cu_h2d, period_ns);
            print_solo("CU H2D", queue, cu_bytes, h2d_solo);
            cu_h2d_commands.push((queue, cu_h2d, h2d_solo));
            let cu_vram = record_cu_copy(
                &device,
                pools[&queue.family],
                queue,
                &pipelines,
                cu_vram_set,
                cu_bytes,
            );
            let vram_solo = run_solo(&device, queue, &cu_vram, period_ns);
            print_solo("CU VRAM copy", queue, cu_bytes, vram_solo);
            cu_vram_commands.push((queue, cu_vram, vram_solo));
        }

        println!("\nCOPY + COMPUTE OVERLAP (40.50 MiB copy, 256 MiB read compute)");
        for case in h2d_commands.iter().filter(|case| case.experts == 8) {
            let pair = run_pair(
                &device,
                main,
                &compute_medium,
                case.queue,
                &case.command,
                period_ns,
            );
            print_pair(
                "compute + H2D",
                main,
                case.queue,
                medium_solo.gpu_us,
                case.solo.gpu_us,
                pair,
            );
        }
        for (queue, command, solo) in &d2h_40_commands {
            let pair = run_pair(&device, main, &compute_medium, *queue, command, period_ns);
            print_pair(
                "compute + D2H",
                main,
                *queue,
                medium_solo.gpu_us,
                solo.gpu_us,
                pair,
            );
        }
        for (queue, command, solo) in &vram_40_commands {
            let pair = run_pair(&device, main, &compute_medium, *queue, command, period_ns);
            print_pair(
                "compute + VRAM copy",
                main,
                *queue,
                medium_solo.gpu_us,
                solo.gpu_us,
                pair,
            );
        }
        for (queue, command, solo) in &cu_h2d_commands {
            if queue.queue == main.queue {
                continue;
            }
            let pair = run_pair(&device, main, &compute_medium, *queue, command, period_ns);
            print_pair(
                "compute + CU H2D",
                main,
                *queue,
                medium_solo.gpu_us,
                solo.gpu_us,
                pair,
            );
        }

        println!("\nOVERLAP WINDOW SWEEP (transfer-only H2D 40.50 MiB)");
        let transfer_h2d = h2d_commands
            .iter()
            .find(|case| case.experts == 8 && case.queue.family == transfer_only.family)
            .expect("transfer-only H2D command");
        for (label, command, solo) in [
            ("compute128 + H2D", &compute_short, short_solo),
            ("compute256 + H2D", &compute_medium, medium_solo),
            ("compute512 + H2D", &compute_long, long_solo),
        ] {
            let pair = run_pair(
                &device,
                main,
                command,
                transfer_only,
                &transfer_h2d.command,
                period_ns,
            );
            print_pair(
                label,
                main,
                transfer_only,
                solo.gpu_us,
                transfer_h2d.solo.gpu_us,
                pair,
            );
        }

        println!("\nH2D + D2H FULL-DUPLEX CHECK");
        let (_, universal_d2h, universal_d2h_solo) = d2h_40_commands
            .iter()
            .find(|(queue, _, _)| queue.queue == main.queue)
            .expect("universal D2H command");
        let pair = run_pair(
            &device,
            transfer_only,
            &transfer_h2d.command,
            main,
            universal_d2h,
            period_ns,
        );
        print_pair(
            "H2D + D2H",
            transfer_only,
            main,
            transfer_h2d.solo.gpu_us,
            universal_d2h_solo.gpu_us,
            pair,
        );

        println!("\nPAGER-SHAPED PIPELINE MATRIX: hit compute || H2D -> miss compute (128 MiB)");
        for experts in [1usize, 4, 8] {
            let same_h2d = h2d_commands
                .iter()
                .find(|case| case.experts == experts && case.queue.queue == main.queue)
                .expect("main-queue H2D command");
            let dedicated_h2d = h2d_commands
                .iter()
                .find(|case| case.experts == experts && case.queue.family == transfer_only.family)
                .expect("transfer-only H2D command");
            for (window, hit) in [
                ("hit128", &compute_short),
                ("hit256", &compute_medium),
                ("hit512", &compute_long),
            ] {
                let label = format!("{experts}exp {window} same");
                let pipe_same = run_pipeline(
                    &device,
                    main,
                    hit,
                    main,
                    &same_h2d.command,
                    &compute_miss,
                    period_ns,
                );
                print_pipeline(&label, main, pipe_same);

                let label = format!("{experts}exp {window} DMA");
                let pipe_transfer = run_pipeline(
                    &device,
                    main,
                    hit,
                    transfer_only,
                    &dedicated_h2d.command,
                    &compute_miss,
                    period_ns,
                );
                print_pipeline(&label, transfer_only, pipe_transfer);
            }
        }

        println!("\nCPU ReBAR PUSH + COMPUTE (coarse host/GPU overlap)");
        let gate = create_timeline(&device, 0);
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("create CPU/GPU overlap fence");
        let mut cpu_gpu_samples = Vec::new();
        for sample in 0..WARMUPS + SAMPLES {
            let value = sample as u64 + 1;
            device.reset_fences(&[fence]).expect("reset CPU/GPU fence");
            reset_timed(&device, &compute_medium);
            submit_waiting_on_gate(&device, main.queue, compute_medium.cmd, gate, value, fence);
            let ready = Barrier::new(2);
            let dst = rebar.mapped.unwrap() as usize;
            let src_addr = h2d_host.ptr as usize;
            let wall = Instant::now();
            let cpu_duration = std::thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    ready.wait();
                    let t0 = Instant::now();
                    let src = std::slice::from_raw_parts(src_addr as *const u8, COPY_BUFFER_BYTES);
                    parallel_copy(src, dst as *mut u8);
                    t0.elapsed()
                });
                signal_timeline(&device, gate, value);
                ready.wait();
                device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .expect("wait CPU/GPU overlap fence");
                worker.join().unwrap()
            });
            let wall_us = wall.elapsed().as_secs_f64() * 1e6;
            let gpu = read_interval(&device, &compute_medium, period_ns);
            if sample >= WARMUPS {
                cpu_gpu_samples.push((cpu_duration.as_secs_f64() * 1e6, gpu.us, wall_us));
            }
        }
        println!(
            "CPU+GPU RAM->ReBAR parallel + compute256: cpu={:.1} us gpu={:.1} us wall={:.1} us (max={:.1}, sum={:.1})",
            median(cpu_gpu_samples.iter().map(|x| x.0).collect()),
            median(cpu_gpu_samples.iter().map(|x| x.1).collect()),
            median(cpu_gpu_samples.iter().map(|x| x.2).collect()),
            median(cpu_gpu_samples.iter().map(|x| x.0.max(x.1)).collect()),
            median(cpu_gpu_samples.iter().map(|x| x.0 + x.1).collect()),
        );
        device.destroy_fence(fence, None);
        device.destroy_semaphore(gate, None);

        device.device_wait_idle().expect("final device idle");
        for case in h2d_commands {
            destroy_timed(&device, &pools, case.command);
        }
        for (_, command, _) in d2h_40_commands {
            destroy_timed(&device, &pools, command);
        }
        for (_, command, _) in vram_40_commands {
            destroy_timed(&device, &pools, command);
        }
        for (_, command, _) in cu_h2d_commands {
            destroy_timed(&device, &pools, command);
        }
        for (_, command, _) in cu_vram_commands {
            destroy_timed(&device, &pools, command);
        }
        for command in retained {
            destroy_timed(&device, &pools, command);
        }
        for command in [compute_short, compute_medium, compute_long, compute_miss] {
            destroy_timed(&device, &pools, command);
        }
        pipelines.destroy(&device);
        for buffer in [
            compute_out,
            compute_src,
            cu_copy_dst,
            vram_copy_dst,
            d2h_src,
            h2d_dst,
            rebar,
            staging,
            d2h_import,
            h2d_import,
        ] {
            buffer.destroy(&device);
        }
        for (_, pool) in pools {
            device.destroy_command_pool(pool, None);
        }
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
