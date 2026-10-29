//! Isolated imported-RAM copy probe. No pager policy or production dispatch changes.
use super::*;

const COPY: &str = r#"
#version 450
layout(local_size_x = 64) in;
layout(binding = 0, std430) readonly buffer Src { uvec4 src[]; };
layout(binding = 1, std430) writeonly buffer Dst { uvec4 dst[]; };
layout(push_constant) uniform Params {
    uint src_base; uint dst_base; uint vectors; uint unused;
} p;
void main() {
    uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    for (uint i = gl_GlobalInvocationID.x; i < p.vectors; i += stride) {
        dst[p.dst_base + i] = src[p.src_base + i];
    }
}
"#;

const SRC_OFFSET: usize = 16;
const DST_OFFSET: usize = 32;

pub(super) unsafe fn print_indirect_support(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) {
    // These KHR structs postdate ash 0.38; layouts and tags follow Vulkan SDK 1.4.357.
    #[repr(C)]
    struct Features {
        s_type: vk::StructureType,
        p_next: *mut c_void,
        copy: vk::Bool32,
        image: vk::Bool32,
    }
    #[repr(C)]
    struct Properties {
        s_type: vk::StructureType,
        p_next: *mut c_void,
        queues: vk::QueueFlags,
    }
    let mut copy = Features {
        s_type: vk::StructureType::from_raw(1_000_549_000),
        p_next: std::ptr::null_mut(),
        copy: 0,
        image: 0,
    };
    let mut features = vk::PhysicalDeviceFeatures2 {
        p_next: (&mut copy as *mut Features).cast(),
        ..Default::default()
    };
    let mut copy_properties = Properties {
        s_type: vk::StructureType::from_raw(1_000_426_001),
        p_next: std::ptr::null_mut(),
        queues: vk::QueueFlags::empty(),
    };
    let mut properties = vk::PhysicalDeviceProperties2 {
        p_next: (&mut copy_properties as *mut Properties).cast(),
        ..Default::default()
    };
    unsafe {
        instance.get_physical_device_features2(physical, &mut features);
        instance.get_physical_device_properties2(physical, &mut properties);
    }
    println!(
        "COPY_INDIRECT feature={} supported_queues={:?} enabled=false",
        copy.copy, copy_properties.queues
    );
}

unsafe fn bind_copy(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    set: vk::DescriptorSet,
    bytes: usize,
    groups: u32,
) {
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[set],
            &[],
        );
        let params = [
            (SRC_OFFSET / 16) as u32,
            (DST_OFFSET / 16) as u32,
            (bytes / 16) as u32,
            0,
        ];
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&params),
        );
        device.cmd_dispatch(cmd, groups, 1, 1);
    }
}

unsafe fn barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    source: vk::PipelineStageFlags,
    target: vk::PipelineStageFlags,
    writes: vk::AccessFlags,
    reads: vk::AccessFlags,
) {
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            source,
            target,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(writes)
                .dst_access_mask(reads)],
            &[],
            &[],
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn verify(
    device: &ash::Device,
    pools: &BTreeMap<u32, vk::CommandPool>,
    queue: QueueRef,
    command: &TimedCommand,
    destination: &RawBuffer,
    readback: &RawBuffer,
    readback_ptr: *const u8,
    source_ptr: *const u8,
    bytes: usize,
    period: f64,
) {
    unsafe {
        let initialize = record_timed(
            device,
            pools[&queue.family],
            queue.family,
            queue.timestamp_bits,
            |cmd| {
                barrier(
                    device,
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                device.cmd_fill_buffer(
                    cmd,
                    destination.buffer,
                    0,
                    (DST_OFFSET + bytes + 16) as u64,
                    0xa5a5_a5a5,
                );
                barrier(
                    device,
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_WRITE,
                );
            },
        );
        run_solo(device, queue, &initialize, period);
        run_solo(device, queue, command, period);
        let copy_back = record_timed(
            device,
            pools[&queue.family],
            queue.family,
            queue.timestamp_bits,
            |cmd| {
                barrier(
                    device,
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );
                device.cmd_copy_buffer(
                    cmd,
                    destination.buffer,
                    readback.buffer,
                    &[vk::BufferCopy::default().size((DST_OFFSET + bytes + 16) as u64)],
                );
                barrier(
                    device,
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::HOST_READ,
                );
            },
        );
        run_solo(device, queue, &copy_back, period);
        let actual = std::slice::from_raw_parts(readback_ptr, DST_OFFSET + bytes + 16);
        let expected = std::slice::from_raw_parts(source_ptr.add(SRC_OFFSET), bytes);
        assert!(actual[..DST_OFFSET].iter().all(|&byte| byte == 0xa5));
        assert_eq!(
            &actual[DST_OFFSET..DST_OFFSET + bytes],
            expected,
            "copy bytes differ"
        );
        assert!(actual[DST_OFFSET + bytes..]
            .iter()
            .all(|&byte| byte == 0xa5));
        destroy_timed(device, pools, initialize);
        destroy_timed(device, pools, copy_back);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run(
    device: &ash::Device,
    pools: &BTreeMap<u32, vk::CommandPool>,
    main: QueueRef,
    dma: QueueRef,
    source: &RawBuffer,
    source_ptr: *const u8,
    destination: &RawBuffer,
    readback: &RawBuffer,
    readback_ptr: *const u8,
    compute_source: &RawBuffer,
    compute_out: &RawBuffer,
    period: f64,
) {
    unsafe {
        let pipelines = ComputePipelines::new(device);
        let copy_set = pipelines.descriptor_set(device, source, destination);
        let consume_set = pipelines.descriptor_set(device, destination, compute_out);
        let warm_set = pipelines.descriptor_set(device, compute_source, compute_out);
        let warm = record_read_compute(
            device,
            pools[&main.family],
            main,
            &pipelines,
            warm_set,
            4,
            0,
        );
        run_solo(device, main, &warm, period);
        println!("CU_PROBE units=us statistic=median warmups={WARMUPS} samples={SAMPLES}");
        println!(
            "CU_PROBE workgroups are scheduling caps, NOT measured CU counts or hard CU partitions"
        );
        let mut best_cases = Vec::new();
        for local in [64u32, 256, 1024] {
            let shader = COPY.replace("local_size_x = 64", &format!("local_size_x = {local}"));
            let copy_pipeline = create_compute_pipeline(
                device,
                pipelines.pipeline_layout,
                &format!("cu_import_grid_stride_{local}"),
                &shader,
            );
            for bytes in [65_536usize, 1_971_200, 3_942_400, 7_884_800] {
                assert_eq!(bytes % 16, 0);
                assert!(DST_OFFSET + bytes + 16 < destination.bytes);
                let region = [vk::BufferCopy::default()
                    .src_offset(SRC_OFFSET as u64)
                    .dst_offset(DST_OFFSET as u64)
                    .size(bytes as u64)];
                let dma_command = record_copy(
                    device,
                    pools[&dma.family],
                    dma,
                    source.buffer,
                    destination.buffer,
                    &region,
                );
                let dma_time = run_solo(device, dma, &dma_command, period);
                println!(
                    "CU_DMA local={local} bytes={bytes} gpu_us={:.3} wall_us={:.3} gib_s={:.3}",
                    dma_time.gpu_us,
                    dma_time.wall_us,
                    bytes as f64 / dma_time.gpu_us / 1e-6 / GIB
                );
                let mut best = (0u32, f64::MAX);
                for cap in [1u32, 2, 4, 8, 16, 32, 64, 96, 192, 0] {
                    let full = (bytes as u32 / 16).div_ceil(local);
                    let groups = if cap == 0 { full } else { cap.min(full) };
                    let command = record_timed(
                        device,
                        pools[&main.family],
                        main.family,
                        main.timestamp_bits,
                        |cmd| {
                            bind_copy(
                                device,
                                cmd,
                                pipelines.pipeline_layout,
                                copy_pipeline,
                                copy_set,
                                bytes,
                                groups,
                            );
                        },
                    );
                    let sample = run_solo(device, main, &command, period);
                    println!(
                    "CU_COPY local={local} bytes={bytes} groups={groups} gpu_us={:.3} wall_us={:.3} gib_s={:.3}",
                    sample.gpu_us,
                    sample.wall_us,
                    bytes as f64 / sample.gpu_us / 1e-6 / GIB
                );
                    if sample.gpu_us < best.1 {
                        best = (groups, sample.gpu_us);
                    }
                    verify(
                        device,
                        pools,
                        main,
                        &command,
                        destination,
                        readback,
                        readback_ptr,
                        source_ptr,
                        bytes,
                        period,
                    );
                    destroy_timed(device, pools, command);
                }
                // Both phases are pre-recorded in one command buffer; the host never intervenes.
                let chain = record_timed(
                    device,
                    pools[&main.family],
                    main.family,
                    main.timestamp_bits,
                    |cmd| {
                        bind_copy(
                            device,
                            cmd,
                            pipelines.pipeline_layout,
                            copy_pipeline,
                            copy_set,
                            bytes,
                            best.0,
                        );
                        barrier(
                            device,
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::AccessFlags::SHADER_WRITE,
                            vk::AccessFlags::SHADER_READ,
                        );
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipelines.read_pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipelines.pipeline_layout,
                            0,
                            &[consume_set],
                            &[],
                        );
                        let params = [(DST_OFFSET / 4) as u32, (bytes / 4) as u32, 0x913bu32, 0];
                        device.cmd_push_constants(
                            cmd,
                            pipelines.pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::cast_slice(&params),
                        );
                        device.cmd_dispatch(cmd, 16, 1, 1);
                    },
                );
                let chain_time = run_solo(device, main, &chain, period);
                println!("CU_CHAIN local={local} bytes={bytes} groups={} copy_gpu_us={:.3} copy_consume_gpu_us={:.3} wall_us={:.3}",
                best.0, best.1, chain_time.gpu_us, chain_time.wall_us);
                let consume_back = record_timed(
                    device,
                    pools[&main.family],
                    main.family,
                    main.timestamp_bits,
                    |cmd| {
                        barrier(
                            device,
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::AccessFlags::SHADER_WRITE,
                            vk::AccessFlags::TRANSFER_READ,
                        );
                        device.cmd_copy_buffer(
                            cmd,
                            compute_out.buffer,
                            readback.buffer,
                            &[vk::BufferCopy::default().size(16 * 4)],
                        );
                        barrier(
                            device,
                            cmd,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::HOST,
                            vk::AccessFlags::TRANSFER_WRITE,
                            vk::AccessFlags::HOST_READ,
                        );
                    },
                );
                run_solo(device, main, &consume_back, period);
                let actual = std::slice::from_raw_parts(readback_ptr.cast::<u32>(), 16);
                for (group, &got) in actual.iter().enumerate() {
                    let mut expected = 0u32;
                    for lane in 0..256 {
                        let global = group * 256 + lane;
                        let mut accumulator = 0x913bu32 ^ global as u32;
                        for i in (global..bytes / 4).step_by(16 * 256) {
                            let word = source_ptr
                                .add(SRC_OFFSET + i * 4)
                                .cast::<u32>()
                                .read_unaligned();
                            accumulator = (accumulator ^ word)
                                .wrapping_mul(1_664_525)
                                .wrapping_add(1_013_904_223);
                        }
                        expected ^= accumulator;
                    }
                    assert_eq!(
                        got, expected,
                        "copy-to-consumer group={group} bytes={bytes}"
                    );
                }
                best_cases.push((local, bytes, best.0, best.1, dma_time.gpu_us));
                destroy_timed(device, pools, consume_back);
                destroy_timed(device, pools, chain);
                destroy_timed(device, pools, dma_command);
            }
            device.destroy_pipeline(copy_pipeline, None);
        }
        for (local, bytes, groups, cu_us, dma_us) in best_cases {
            println!("CU_BEST local={local} bytes={bytes} groups={groups} gpu_us={cu_us:.3} dma_us={dma_us:.3}");
        }
        println!("CU_VALIDATION copies=120 guards=120 dependent_consumers=12 no_model_run=true");
        destroy_timed(device, pools, warm);
        pipelines.destroy(device);
    }
}
