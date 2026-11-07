//! STANDALONE, ISOLATED int8 cooperative-matrix hang investigation (issue: commit ad82a77 found
//! SINT8×UINT8 coopmat "enumerates fine, hangs the GPU at execution, validation flags nothing").
//! This binary is a throwaway diagnostic, kept as a committed artifact so the investigation is
//! reproducible — it does NOT touch the production adapter/gemm path (own Vulkan instance/device,
//! own tiny shaders compiled at runtime via `glslc`).
//!
//! SAFETY MODEL (headless TTY/SSH — a GPU hang here is recoverable via amdgpu TDR, not a desktop
//! freeze, but still treated as hazardous):
//!   - ONE dispatch per process invocation. Pick a variant via argv[1].
//!   - The fence wait is bounded (3s, never u64::MAX) — `vkWaitForFences` is spec-guaranteed to
//!     return by the timeout even if the GPU is wedged (the kernel ioctl itself is bounded).
//!   - On ANY outcome (completed, timeout, device-lost) the process prints one structured result
//!     line and calls `std::process::exit` immediately — no Vulkan object destruction. Tearing down
//!     a device tied to a possibly-wedged queue is itself a hang risk; an abrupt process exit lets
//!     the OS reclaim everything without touching the driver again.
//!   - The caller (a shell driver, see docs/notes in the investigation) wraps every invocation in
//!     an outer `timeout` as a second, coarser bound, and checks `dmesg`/`journalctl -k` for
//!     amdgpu reset messages plus re-runs a known-good dispatch (`plain_add`, `f16_known_good`)
//!     after every hang before trying the next variant.
//!
//! Run: `cargo run -p infr-vulkan --example coopmat_int8_test -- <variant>`
//! List variants: `cargo run -p infr-vulkan --example coopmat_int8_test -- --list`

use ash::vk;
use std::ffi::CStr;
use std::time::{Duration, Instant};

// ── variant table (int8 coopmat configs) ────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    a_glsl: &'static str, // "int8_t" | "uint8_t"
    b_glsl: &'static str,
    a_row_major: bool,
    b_row_major: bool,
    saturating: bool,
    subgroup_pin: Option<u32>, // None = production-style pin omitted
    local_size: u32,           // workgroup size == 1 subgroup for a single 16x16x16 tile
}

const INT8_VARIANTS: &[Variant] = &[
    // 1. Cleanest mirror of the production f16 coopmat kernel's setup, varied only in component
    //    types: pure SINT8xSINT8->SINT32, subgroup pinned 32 (RDNA3 coopmat is wave32), A
    //    RowMajor / B ColMajor (matches native_gemm_warp.comp), non-saturating.
    Variant {
        name: "baseline",
        a_glsl: "int8_t",
        b_glsl: "int8_t",
        a_row_major: true,
        b_row_major: false,
        saturating: false,
        subgroup_pin: Some(32),
        local_size: 32,
    },
    // 2/3. The MIXED-sign config the commit specifically named (SINT8xUINT8), both operand orders.
    Variant {
        name: "mixed_a_sint_b_uint",
        a_glsl: "int8_t",
        b_glsl: "uint8_t",
        a_row_major: true,
        b_row_major: false,
        saturating: false,
        subgroup_pin: Some(32),
        local_size: 32,
    },
    Variant {
        name: "mixed_a_uint_b_sint",
        a_glsl: "uint8_t",
        b_glsl: "int8_t",
        a_row_major: true,
        b_row_major: false,
        saturating: false,
        subgroup_pin: Some(32),
        local_size: 32,
    },
    // 4. Subgroup size 64 instead of the wave32 pin.
    Variant {
        name: "sg64",
        a_glsl: "int8_t",
        b_glsl: "int8_t",
        a_row_major: true,
        b_row_major: false,
        saturating: false,
        subgroup_pin: Some(64),
        local_size: 64,
    },
    // 5. No requiredSubgroupSize pin at all (production always pins for its coopmat kernels).
    Variant {
        name: "sg_unpinned",
        a_glsl: "int8_t",
        b_glsl: "int8_t",
        a_row_major: true,
        b_row_major: false,
        saturating: false,
        subgroup_pin: None,
        local_size: 32,
    },
    // 6/7. Layout variations (both RowMajor, both ColMajor) vs the baseline's Row/Col mix.
    Variant {
        name: "layout_rowrow",
        a_glsl: "int8_t",
        b_glsl: "int8_t",
        a_row_major: true,
        b_row_major: true,
        saturating: false,
        subgroup_pin: Some(32),
        local_size: 32,
    },
    Variant {
        name: "layout_colcol",
        a_glsl: "int8_t",
        b_glsl: "int8_t",
        a_row_major: false,
        b_row_major: false,
        saturating: false,
        subgroup_pin: Some(32),
        local_size: 32,
    },
    // 8. Saturating accumulation variant (gl_MatrixOperandsSaturatingAccumulation).
    Variant {
        name: "saturating",
        a_glsl: "int8_t",
        b_glsl: "int8_t",
        a_row_major: true,
        b_row_major: false,
        saturating: true,
        subgroup_pin: Some(32),
        local_size: 32,
    },
    // 9. Pure UINT8xUINT8->SINT32 (also enumerated by coopmat_probe).
    Variant {
        name: "uint_uint",
        a_glsl: "uint8_t",
        b_glsl: "uint8_t",
        a_row_major: true,
        b_row_major: false,
        saturating: false,
        subgroup_pin: Some(32),
        local_size: 32,
    },
];

fn gen_int8_shader(v: &Variant) -> String {
    let a_layout = if v.a_row_major {
        "gl_CooperativeMatrixLayoutRowMajor"
    } else {
        "gl_CooperativeMatrixLayoutColumnMajor"
    };
    let b_layout = if v.b_row_major {
        "gl_CooperativeMatrixLayoutRowMajor"
    } else {
        "gl_CooperativeMatrixLayoutColumnMajor"
    };
    let sat_arg = if v.saturating {
        ", gl_MatrixOperandsSaturatingAccumulation"
    } else {
        ""
    };
    format!(
        r#"#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = {lsz}, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuf {{ {a_ty} a[]; }};
layout(binding = 1) readonly buffer BBuf {{ {b_ty} b[]; }};
layout(binding = 2) writeonly buffer CBuf {{ int c[]; }};

void main() {{
    coopmat<{a_ty}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
    coopmat<{b_ty}, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
    coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
        coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);

    coopMatLoad(af, a, 0, 16, {a_layout});
    coopMatLoad(bf, b, 0, 16, {b_layout});
    acc = coopMatMulAdd(af, bf, acc{sat_arg});
    coopMatStore(acc, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}}
"#,
        lsz = v.local_size,
        a_ty = v.a_glsl,
        b_ty = v.b_glsl,
        a_layout = a_layout,
        b_layout = b_layout,
        sat_arg = sat_arg
    )
}

const F16_KNOWN_GOOD_SHADER: &str = r#"#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuf { float16_t a[]; };
layout(binding = 1) readonly buffer BBuf { float16_t b[]; };
layout(binding = 2) writeonly buffer CBuf { float c[]; };

void main() {
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    coopMatLoad(af, a, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(bf, b, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
    acc = coopMatMulAdd(af, bf, acc);
    coopMatStore(acc, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
"#;

// Escalation beyond the single-fragment tests: a K-loop of 16 chained coopMatMulAdd calls into
// the SAME accumulator (K=256), single workgroup — the shape every real GEMM actually uses
// (native_gemm_warp.comp's inner loop), in case the hang needs repeated accumulation to surface.
const INT8_LOOPED_K_SHADER: &str = r#"#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuf { int8_t a[]; };  // [16,256] row-major
layout(binding = 1) readonly buffer BBuf { int8_t b[]; };  // [256,16] column-major
layout(binding = 2) writeonly buffer CBuf { int c[]; };    // [16,16] row-major

void main() {
    coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
        coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
    for (uint k0 = 0u; k0 < 256u; k0 += 16u) {
        coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
        coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
        coopMatLoad(af, a, k0, 256, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(bf, b, k0, 256, gl_CooperativeMatrixLayoutColumnMajor);
        acc = coopMatMulAdd(af, bf, acc);
    }
    coopMatStore(acc, c, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
}
"#;

// Escalation: 4 CONCURRENT workgroups (2x2 grid), each computing an independent 16x16x16 tile —
// tests whether multiple wavefronts hitting the matrix core simultaneously (real GEMM dispatch
// grids always have many workgroups) is what's needed to trigger the hang, vs a single wavefront.
const INT8_MULTI_WG_SHADER: &str = r#"#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuf { int8_t a[]; };  // [32,16] row-major
layout(binding = 1) readonly buffer BBuf { int8_t b[]; };  // [16,32] column-major
layout(binding = 2) writeonly buffer CBuf { int c[]; };    // [32,32] row-major

void main() {
    uint wgRow = gl_WorkGroupID.y * 16u;
    uint wgCol = gl_WorkGroupID.x * 16u;
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> af;
    coopmat<int8_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bf;
    coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
        coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
    coopMatLoad(af, a, wgRow * 16u, 16, gl_CooperativeMatrixLayoutRowMajor);
    coopMatLoad(bf, b, wgCol * 16u, 16, gl_CooperativeMatrixLayoutColumnMajor);
    acc = coopMatMulAdd(af, bf, acc);
    coopMatStore(acc, c, wgRow * 32u + wgCol, 32, gl_CooperativeMatrixLayoutRowMajor);
}
"#;

// Idea 1a probe: derive RADV's int32-accumulator 16x16x16 fragment `i -> (row,col)` mapping.
// Loads a KNOWN int32 matrix (M[r*16+c] = r*16+c) into an accumulator-use coopmat via coopMatLoad
// RowMajor (the same load semantics coopMatStore RowMajor writes use, just inverted), then has
// every lane dump `csub[i]` for i in 0..csub.length() to a buffer indexed by [lane*16+i]. Since
// M's value AT (r,c) directly encodes r*16+c, csub[i]'s value tells us r=csub[i]/16, c=csub[i]%16
// for whichever (r,c) lane/i owns — no separate cross-check against coopMatStore needed, the
// encoding IS the cross-check (RowMajor load and RowMajor store share the same linearization).
const INT8_FRAGMENT_LAYOUT_SHADER: &str = r#"#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer MBuf { int m[]; };       // [16,16] row-major, m[r*16+c]=r*16+c
layout(binding = 2) writeonly buffer OutBuf { int outv[]; }; // outv[0]=length, outv[1+lane*16+i]=csub[i]

void main() {
    coopmat<int, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> csub;
    coopMatLoad(csub, m, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    uint lane = gl_SubgroupInvocationID;
    uint len = csub.length();
    if (lane == 0u) { outv[0] = int(len); }
    for (uint i = 0u; i < len; i++) {
        outv[1u + lane * 16u + i] = csub[i];
    }
}
"#;

const PLAIN_ADD_SHADER: &str = r#"#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) readonly buffer ABuf { float a[]; };
layout(binding = 1) readonly buffer BBuf { float b[]; };
layout(binding = 2) writeonly buffer CBuf { float c[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    c[i] = a[i] + b[i];
}
"#;

// ── glslc invocation ─────────────────────────────────────────────────────────────────────────

fn compile_glsl(src: &str, tag: &str) -> Result<Vec<u32>, String> {
    let dir = std::env::temp_dir();
    let stem = format!("infr_coopmat_i8_test_{tag}_{}", std::process::id());
    let src_path = dir.join(format!("{stem}.comp"));
    let spv_path = dir.join(format!("{stem}.spv"));
    std::fs::write(&src_path, src).map_err(|e| format!("write shader src: {e}"))?;
    let status = std::process::Command::new("glslc")
        .args([
            "-fshader-stage=comp",
            "--target-env=vulkan1.3",
            "-O",
            src_path.to_str().unwrap(),
            "-o",
            spv_path.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("spawn glslc: {e}"))?;
    if !status.success() {
        return Err(format!("glslc exited with {status}"));
    }
    let bytes = std::fs::read(&spv_path).map_err(|e| format!("read spv: {e}"))?;
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(words)
}

// ── Vulkan context (own instance/device, mirrors production's feature-enable discipline) ──────

struct VkCtx {
    #[allow(dead_code)]
    entry: ash::Entry,
    #[allow(dead_code)]
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
}

fn init_ctx() -> VkCtx {
    unsafe {
        let entry = ash::Entry::load().expect("ash::Entry::load");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .expect("create_instance");

        let pdevices = instance
            .enumerate_physical_devices()
            .expect("enumerate_physical_devices");
        assert!(!pdevices.is_empty(), "no Vulkan physical devices");
        // Prefer discrete GPU — same selection rule as production `VulkanBackend::new()`, so this
        // targets the RX 7900 XTX even on a box with an APU also enumerated.
        let pdevice = pdevices
            .iter()
            .copied()
            .find(|&pd| {
                instance.get_physical_device_properties(pd).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(pdevices[0]);
        let dev_name = CStr::from_ptr(
            instance
                .get_physical_device_properties(pdevice)
                .device_name
                .as_ptr(),
        )
        .to_string_lossy()
        .into_owned();

        let qf_props = instance.get_physical_device_queue_family_properties(pdevice);
        let qfi = qf_props
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .expect("no compute queue family");

        // Probe features (defensive — fail loudly with a clear message rather than a cryptic
        // create_device error if this box's driver doesn't have what we need).
        let mut f16_feat = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut mm_feat = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        let mut sg_feat = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut cm_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut s8_feat = vk::PhysicalDevice8BitStorageFeatures::default();
        let mut s16_feat = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut feat2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f16_feat)
            .push_next(&mut mm_feat)
            .push_next(&mut sg_feat)
            .push_next(&mut cm_feat)
            .push_next(&mut s8_feat)
            .push_next(&mut s16_feat);
        instance.get_physical_device_features2(pdevice, &mut feat2);
        assert!(
            cm_feat.cooperative_matrix != 0,
            "{dev_name}: no cooperativeMatrix feature"
        );
        assert!(
            f16_feat.shader_int8 != 0,
            "{dev_name}: no shaderInt8 feature"
        );
        assert!(
            f16_feat.shader_float16 != 0,
            "{dev_name}: no shaderFloat16 feature"
        );
        assert!(
            mm_feat.vulkan_memory_model != 0,
            "{dev_name}: no vulkanMemoryModel feature"
        );
        assert!(
            s8_feat.storage_buffer8_bit_access != 0,
            "{dev_name}: no storageBuffer8BitAccess"
        );
        assert!(
            s16_feat.storage_buffer16_bit_access != 0,
            "{dev_name}: no storageBuffer16BitAccess"
        );

        let ext_ptrs = [
            c"VK_KHR_cooperative_matrix".as_ptr(),
            c"VK_KHR_8bit_storage".as_ptr(),
            c"VK_KHR_16bit_storage".as_ptr(),
        ];
        let mut shader_f16_ci = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(true)
            .shader_int8(true);
        let mut s8_ci =
            vk::PhysicalDevice8BitStorageFeatures::default().storage_buffer8_bit_access(true);
        let mut s16_ci =
            vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
        let mut mm_ci = vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);
        let mut cm_ci =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut sg_ci = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);

        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&priorities);
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut shader_f16_ci)
            .push_next(&mut s8_ci)
            .push_next(&mut s16_ci)
            .push_next(&mut mm_ci)
            .push_next(&mut cm_ci)
            .push_next(&mut sg_ci);
        let device = instance
            .create_device(pdevice, &device_ci, None)
            .expect("create_device");
        let queue = device.get_device_queue(qfi, 0);
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfi)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .expect("create_command_pool");
        let mem_props = instance.get_physical_device_memory_properties(pdevice);

        eprintln!("[ctx] device = {dev_name}");
        VkCtx {
            entry,
            instance,
            device,
            queue,
            mem_props,
            cmd_pool,
        }
    }
}

unsafe fn alloc_host_buffer(ctx: &VkCtx, size: u64) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
    let buf = ctx
        .device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size.max(4))
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .expect("create_buffer");
    let req = ctx.device.get_buffer_memory_requirements(buf);
    let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mt = (0..ctx.mem_props.memory_type_count)
        .find(|&i| {
            (req.memory_type_bits & (1 << i)) != 0
                && ctx.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(want)
        })
        .expect("no host-visible+coherent memory type");
    let mem = ctx
        .device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt),
            None,
        )
        .expect("allocate_memory");
    ctx.device
        .bind_buffer_memory(buf, mem, 0)
        .expect("bind_buffer_memory");
    let ptr = ctx
        .device
        .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        .expect("map_memory") as *mut u8;
    (buf, mem, ptr)
}

// ── dispatch outcome ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum Outcome {
    Completed(Duration),
    Timeout,
    DeviceLost(vk::Result),
    PipelineCreateFailed(vk::Result),
    PipelineNull,
    SubmitFailed(vk::Result),
}

/// Runs ONE dispatch of a 16x16x16 single-fragment coopMatMulAdd (or a trivial add kernel) and
/// waits on a BOUNDED fence (3s, never u64::MAX). Never destroys any Vulkan object — the caller
/// process exits right after this returns, so cleanup happens via OS process teardown, not driver
/// calls that could themselves block on a wedged queue.
unsafe fn dispatch_one(
    ctx: &VkCtx,
    spv: &[u32],
    subgroup_pin: Option<u32>,
    a_bytes: &[u8],
    b_bytes: &[u8],
    c_size: usize,
    wg: (u32, u32, u32),
) -> (Outcome, Option<Vec<u8>>) {
    let shader = ctx
        .device
        .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)
        .expect("create_shader_module");
    let bindings: Vec<_> = (0..3u32)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let ds_layout = ctx
        .device
        .create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
        .expect("create_descriptor_set_layout");
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout)),
            None,
        )
        .expect("create_pipeline_layout");

    let entry_name = c"main";
    let mut req_sz = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
    let mut stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry_name);
    if let Some(sz) = subgroup_pin {
        req_sz = req_sz.required_subgroup_size(sz);
        stage = stage
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
            .push_next(&mut req_sz);
    }
    let pipeline = match ctx.device.create_compute_pipelines(
        vk::PipelineCache::null(),
        &[vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)],
        None,
    ) {
        Ok(p) => {
            if p[0] == vk::Pipeline::null() {
                return (Outcome::PipelineNull, None);
            }
            p[0]
        }
        Err((_, e)) => return (Outcome::PipelineCreateFailed(e), None),
    };

    let (a_buf, _a_mem, a_ptr) = alloc_host_buffer(ctx, a_bytes.len() as u64);
    let (b_buf, _b_mem, b_ptr) = alloc_host_buffer(ctx, b_bytes.len() as u64);
    let (c_buf, _c_mem, c_ptr) = alloc_host_buffer(ctx, c_size as u64);
    std::ptr::copy_nonoverlapping(a_bytes.as_ptr(), a_ptr, a_bytes.len());
    std::ptr::copy_nonoverlapping(b_bytes.as_ptr(), b_ptr, b_bytes.len());
    // Poison C before dispatch so a no-op / early-exit shader can't look "correct" by accident.
    std::ptr::write_bytes(c_ptr, 0xAA, c_size);

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3,
    }];
    let desc_pool = ctx
        .device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )
        .expect("create_descriptor_pool");
    let ds = ctx
        .device
        .allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(std::slice::from_ref(&ds_layout)),
        )
        .expect("allocate_descriptor_sets")[0];
    let infos = [
        vk::DescriptorBufferInfo::default()
            .buffer(a_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(b_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(c_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE),
    ];
    let writes: Vec<_> = (0..3usize)
        .map(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[i]))
        })
        .collect();
    ctx.device.update_descriptor_sets(&writes, &[]);

    let cmd = ctx
        .device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(ctx.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .expect("allocate_command_buffers")[0];
    ctx.device
        .begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .expect("begin_command_buffer");
    ctx.device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[ds],
        &[],
    );
    ctx.device.cmd_dispatch(cmd, wg.0, wg.1, wg.2);
    ctx.device
        .end_command_buffer(cmd)
        .expect("end_command_buffer");

    let fence = ctx
        .device
        .create_fence(&vk::FenceCreateInfo::default(), None)
        .expect("create_fence");
    let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    let t0 = Instant::now();
    if let Err(e) = ctx.device.queue_submit(ctx.queue, &[submit], fence) {
        return (Outcome::SubmitFailed(e), None);
    }
    // HARD SAFETY RAIL: bounded fence wait, never u64::MAX. 3s.
    const TIMEOUT_NS: u64 = 3_000_000_000;
    let wait_result = ctx.device.wait_for_fences(&[fence], true, TIMEOUT_NS);
    let elapsed = t0.elapsed();
    match wait_result {
        Ok(()) => {
            let mut out = vec![0u8; c_size];
            std::ptr::copy_nonoverlapping(c_ptr, out.as_mut_ptr(), c_size);
            (Outcome::Completed(elapsed), Some(out))
        }
        Err(vk::Result::TIMEOUT) => (Outcome::Timeout, None),
        Err(e) => (Outcome::DeviceLost(e), None),
    }
}

// ── data packing / CPU reference ────────────────────────────────────────────────────────────

/// Packs a logical 16x16 matrix (small non-negative values, valid bit-identically as int8_t or
/// uint8_t) into a flat buffer honoring the requested coopMatLoad layout.
fn pack16(row_major: bool, get: impl Fn(usize, usize) -> i32) -> Vec<u8> {
    let n = 16usize;
    let mut buf = vec![0u8; n * n];
    for i in 0..n {
        for k in 0..n {
            let idx = if row_major { i * n + k } else { k * n + i };
            buf[idx] = get(i, k) as u8;
        }
    }
    buf
}

fn cpu_matmul16(a: &dyn Fn(usize, usize) -> i32, b: &dyn Fn(usize, usize) -> i32) -> Vec<i32> {
    let n = 16usize;
    let mut c = vec![0i32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0i32;
            for k in 0..n {
                s += a(i, k) * b(k, j);
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn print_result(variant: &str, outcome: &Outcome, correct: Option<bool>) {
    let (status, extra) = match outcome {
        Outcome::Completed(d) => (
            "COMPLETED",
            format!("elapsed_ms={:.2}", d.as_secs_f64() * 1000.0),
        ),
        Outcome::Timeout => ("TIMEOUT", String::new()),
        Outcome::DeviceLost(e) => ("DEVICE_LOST", format!("vk_result={e:?}")),
        Outcome::PipelineCreateFailed(e) => ("PIPELINE_CREATE_FAILED", format!("vk_result={e:?}")),
        Outcome::PipelineNull => ("PIPELINE_NULL", String::new()),
        Outcome::SubmitFailed(e) => ("SUBMIT_FAILED", format!("vk_result={e:?}")),
    };
    let correct_s = match correct {
        Some(true) => " correct=true",
        Some(false) => " correct=false",
        None => "",
    };
    println!("RESULT variant={variant} outcome={status}{correct_s} {extra}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "--list" {
        eprintln!("usage: coopmat_int8_test <variant>");
        eprintln!("int8 variants:");
        for v in INT8_VARIANTS {
            eprintln!("  {}", v.name);
        }
        eprintln!("  looped_k (K=256, 16 chained coopMatMulAdd)");
        eprintln!("  multi_wg (2x2=4 concurrent workgroups)");
        eprintln!("  fragment_layout (dump csub[i] -> (row,col) mapping, idea 1a)");
        eprintln!("recovery-check variants:");
        eprintln!("  f16_known_good");
        eprintln!("  plain_add");
        std::process::exit(1);
    }
    let name = args[1].as_str();

    if name == "plain_add" {
        let ctx = init_ctx();
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5).collect();
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let spv = match compile_glsl(PLAIN_ADD_SHADER, "plain_add") {
            Ok(w) => w,
            Err(e) => {
                println!("RESULT variant=plain_add outcome=COMPILE_ERROR msg={e:?}");
                std::process::exit(4);
            }
        };
        let (outcome, data) =
            unsafe { dispatch_one(&ctx, &spv, None, &a_bytes, &b_bytes, 64 * 4, (1, 1, 1)) };
        let correct = data.map(|bytes| {
            let c: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            c.iter()
                .zip(a.iter().zip(b.iter()))
                .all(|(cv, (av, bv))| (*cv - (av + bv)).abs() < 1e-6)
        });
        print_result("plain_add", &outcome, correct);
        std::process::exit(match outcome {
            Outcome::Completed(_) => 0,
            Outcome::Timeout => 2,
            _ => 3,
        });
    }

    if name == "f16_known_good" {
        let ctx = init_ctx();
        // A[i][k] = (i+k)%5, B[k][j]=(k+j)%3, small values exact in f16.
        let a_get = |i: usize, k: usize| ((i + k) % 5) as i32;
        let b_get = |k: usize, j: usize| ((k + j) % 3) as i32;
        let a16: Vec<half::f16> = (0..16)
            .flat_map(|i| (0..16).map(move |k| half::f16::from_f32(a_get(i, k) as f32)))
            .collect();
        // b16 packed ColumnMajor to match the shader's coopMatLoad layout: buf[j*16+k] = B[k][j].
        let mut b16_col = vec![half::f16::from_f32(0.0); 256];
        for k in 0..16 {
            for j in 0..16 {
                b16_col[j * 16 + k] = half::f16::from_f32(b_get(k, j) as f32);
            }
        }
        let a_bytes: Vec<u8> = a16.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let b_bytes: Vec<u8> = b16_col.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let spv = match compile_glsl(F16_KNOWN_GOOD_SHADER, "f16_known_good") {
            Ok(w) => w,
            Err(e) => {
                println!("RESULT variant=f16_known_good outcome=COMPILE_ERROR msg={e:?}");
                std::process::exit(4);
            }
        };
        let (outcome, data) = unsafe {
            dispatch_one(
                &ctx,
                &spv,
                Some(32),
                &a_bytes,
                &b_bytes,
                16 * 16 * 4,
                (1, 1, 1),
            )
        };
        let cpu_c = cpu_matmul16(&a_get, &b_get);
        let correct = data.map(|bytes| {
            let c: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            c.iter()
                .zip(cpu_c.iter())
                .all(|(gv, rv)| (*gv - (*rv as f32)).abs() < 1e-3)
        });
        print_result("f16_known_good", &outcome, correct);
        std::process::exit(match outcome {
            Outcome::Completed(_) => 0,
            Outcome::Timeout => 2,
            _ => 3,
        });
    }

    if name == "looped_k" {
        let ctx = init_ctx();
        // A [16,256] row-major, B [256,16] column-major, K=256 (16 chained coopMatMulAdd calls
        // into one accumulator) — the shape every real GEMM kernel actually uses.
        let a_get = |i: usize, k: usize| ((i + k) % 5) as i32;
        let b_get = |k: usize, j: usize| ((k + j) % 3) as i32;
        let mut a_bytes = vec![0u8; 16 * 256];
        for i in 0..16 {
            for k in 0..256 {
                a_bytes[i * 256 + k] = a_get(i, k) as u8;
            }
        }
        let mut b_bytes = vec![0u8; 256 * 16];
        for k in 0..256 {
            for j in 0..16 {
                b_bytes[j * 256 + k] = b_get(k, j) as u8; // column-major: b[col*256+row]
            }
        }
        let spv = match compile_glsl(INT8_LOOPED_K_SHADER, "looped_k") {
            Ok(w) => w,
            Err(e) => {
                println!("RESULT variant=looped_k outcome=COMPILE_ERROR msg={e:?}");
                std::process::exit(4);
            }
        };
        let (outcome, data) = unsafe {
            dispatch_one(
                &ctx,
                &spv,
                Some(32),
                &a_bytes,
                &b_bytes,
                16 * 16 * 4,
                (1, 1, 1),
            )
        };
        let cpu_c = {
            let mut c = vec![0i32; 16 * 16];
            for i in 0..16 {
                for j in 0..16 {
                    let mut s = 0i32;
                    for k in 0..256 {
                        s += a_get(i, k) * b_get(k, j);
                    }
                    c[i * 16 + j] = s;
                }
            }
            c
        };
        let correct = data.map(|bytes| {
            let c: Vec<i32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            c == cpu_c
        });
        print_result("looped_k", &outcome, correct);
        std::process::exit(match outcome {
            Outcome::Completed(_) => 0,
            Outcome::Timeout => 2,
            _ => 3,
        });
    }

    if name == "multi_wg" {
        let ctx = init_ctx();
        // A [32,16] row-major, B [16,32] column-major, dispatch(2,2,1) -> 4 concurrent workgroups
        // each computing an independent 16x16x16 tile into a shared [32,32] C.
        let a_get = |i: usize, k: usize| ((i + k) % 5) as i32;
        let b_get = |k: usize, j: usize| ((k + j) % 3) as i32;
        let mut a_bytes = vec![0u8; 32 * 16];
        for i in 0..32 {
            for k in 0..16 {
                a_bytes[i * 16 + k] = a_get(i, k) as u8;
            }
        }
        let mut b_bytes = vec![0u8; 16 * 32];
        for k in 0..16 {
            for j in 0..32 {
                b_bytes[j * 16 + k] = b_get(k, j) as u8; // column-major: b[col*16+row]
            }
        }
        let spv = match compile_glsl(INT8_MULTI_WG_SHADER, "multi_wg") {
            Ok(w) => w,
            Err(e) => {
                println!("RESULT variant=multi_wg outcome=COMPILE_ERROR msg={e:?}");
                std::process::exit(4);
            }
        };
        let (outcome, data) = unsafe {
            dispatch_one(
                &ctx,
                &spv,
                Some(32),
                &a_bytes,
                &b_bytes,
                32 * 32 * 4,
                (2, 2, 1),
            )
        };
        let cpu_c = {
            let mut c = vec![0i32; 32 * 32];
            for i in 0..32 {
                for j in 0..32 {
                    let mut s = 0i32;
                    for k in 0..16 {
                        s += a_get(i, k) * b_get(k, j);
                    }
                    c[i * 32 + j] = s;
                }
            }
            c
        };
        let correct = data.map(|bytes| {
            let c: Vec<i32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            c == cpu_c
        });
        print_result("multi_wg", &outcome, correct);
        std::process::exit(match outcome {
            Outcome::Completed(_) => 0,
            Outcome::Timeout => 2,
            _ => 3,
        });
    }

    if name == "fragment_layout" {
        let ctx = init_ctx();
        let mut m_bytes = vec![0u8; 16 * 16 * 4];
        for r in 0..16usize {
            for c in 0..16usize {
                let v = (r * 16 + c) as i32;
                let idx = r * 16 + c;
                m_bytes[idx * 4..idx * 4 + 4].copy_from_slice(&v.to_ne_bytes());
            }
        }
        let dummy_b = vec![0u8; 4];
        let out_size = (1 + 32 * 16) * 4;
        let spv = match compile_glsl(INT8_FRAGMENT_LAYOUT_SHADER, "fragment_layout") {
            Ok(w) => w,
            Err(e) => {
                println!("RESULT variant=fragment_layout outcome=COMPILE_ERROR msg={e:?}");
                eprintln!("--- shader source ---\n{INT8_FRAGMENT_LAYOUT_SHADER}");
                std::process::exit(4);
            }
        };
        let (outcome, data) = unsafe {
            dispatch_one(
                &ctx,
                &spv,
                Some(32),
                &m_bytes,
                &dummy_b,
                out_size,
                (1, 1, 1),
            )
        };
        match &outcome {
            Outcome::Completed(_) => {
                let bytes = data.unwrap();
                let vals: Vec<i32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let len = vals[0];
                println!("RESULT variant=fragment_layout outcome=COMPLETED length={len}");
                for lane in 0..32usize {
                    let mut row = format!("lane={lane}:");
                    for i in 0..(len.max(0) as usize) {
                        let v = vals[1 + lane * 16 + i];
                        let r = v / 16;
                        let c = v % 16;
                        row.push_str(&format!(" i={i}->(r={r},c={c},raw={v})"));
                    }
                    println!("{row}");
                }
            }
            _ => print_result("fragment_layout", &outcome, None),
        }
        std::process::exit(match outcome {
            Outcome::Completed(_) => 0,
            Outcome::Timeout => 2,
            _ => 3,
        });
    }

    let variant = match INT8_VARIANTS.iter().find(|v| v.name == name) {
        Some(v) => *v,
        None => {
            eprintln!("unknown variant {name:?}; run with --list");
            std::process::exit(1);
        }
    };

    let ctx = init_ctx();
    // Data: small non-negative values (0..4 for A, 0..2 for B), representable identically as
    // int8_t or uint8_t — lets ONE CPU reference serve every sign combination without overflow.
    let a_get = |i: usize, k: usize| ((i + k) % 5) as i32;
    let b_get = |k: usize, j: usize| ((k + j) % 3) as i32;
    let a_bytes = pack16(variant.a_row_major, a_get);
    let b_bytes = pack16(variant.b_row_major, b_get);

    let src = gen_int8_shader(&variant);
    let spv = match compile_glsl(&src, variant.name) {
        Ok(w) => w,
        Err(e) => {
            println!(
                "RESULT variant={} outcome=COMPILE_ERROR msg={:?}",
                variant.name, e
            );
            eprintln!("--- shader source ---\n{src}");
            std::process::exit(4);
        }
    };

    let (outcome, data) = unsafe {
        dispatch_one(
            &ctx,
            &spv,
            variant.subgroup_pin,
            &a_bytes,
            &b_bytes,
            16 * 16 * 4,
            (1, 1, 1),
        )
    };
    let cpu_c = cpu_matmul16(&a_get, &b_get);
    let correct = data.map(|bytes| {
        let c: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        c == cpu_c
    });
    print_result(variant.name, &outcome, correct);
    std::process::exit(match outcome {
        Outcome::Completed(_) => 0,
        Outcome::Timeout => 2,
        _ => 3,
    });
}
