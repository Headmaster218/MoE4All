//! Capability DECISIONS — the pure half of hardware detection.
//!
//! `lib.rs` PROBES the device (extension strings, feature bits, property structs, cooperative-matrix
//! enumeration); everything in here decides what to DO with those answers. The split is deliberate
//! and load-bearing: this box has one RX 7900 XTX (RDNA3, RADV) and no NVIDIA, Intel or Apple part,
//! so a decision that only exists inside the probe can never be exercised for those vendors. As
//! free functions over plain structs, each rule below is unit-tested against synthetic NVIDIA /
//! Intel Arc / AMD shapes — including the two drivers upstream has caught lying (see
//! [`coopmat_trust`]).
//!
//! Nothing here reads the environment, the config, or a `VkPhysicalDevice`.

use infr_core::{COOPMAT_TILE_16, COOPMAT_TILE_8};

use ash::vk;

// ── cooperative-matrix tile selection ────────────────────────────────────────────────────────────

/// Pick ONE cooperative-matrix (M,N,K) tile for a component type from the device's enumerated
/// shape list, by preference order:
///
/// 1. [`COOPMAT_TILE_16`] (16x16x16) — the shape EVERY production coopmat shader is built for; a
///    device that enumerates it (RADV/RDNA3+, NVIDIA, and reportedly some Battlemage drivers)
///    always gets it, regardless of `allow_8x8x16` — the env knob must never move a device off the
///    proven kernel set.
/// 2. [`COOPMAT_TILE_8`] (8x8x16, Intel Arc/ANV XMX) — only when `allow_8x8x16` (the
///    `INFR_CM_8X8=1` opt-in; only `native_gemm_warp`'s `_cm8` builds exist at this shape, and
///    Alchemist coopmat is a llama.cpp-documented regression, so it stays default-OFF).
/// 3. `None` — no shape any kernel here is built for; the non-coopmat tiers take over.
///
/// Pure function of the enumerated list + the opt-in flag (no env reads) so the selection is
/// unit-testable with synthetic property lists. The caller filters the list through
/// [`coopmat_shape_trusted`] FIRST, so a shape this driver may not be believed about never reaches
/// the preference order.
pub(crate) fn select_coopmat_shape(
    shapes: impl IntoIterator<Item = (u32, u32, u32)>,
    allow_8x8x16: bool,
) -> Option<(u32, u32, u32)> {
    let mut has_8x8x16 = false;
    for s in shapes {
        if s == COOPMAT_TILE_16 {
            return Some(COOPMAT_TILE_16);
        }
        has_8x8x16 |= s == COOPMAT_TILE_8;
    }
    (allow_8x8x16 && has_8x8x16).then_some(COOPMAT_TILE_8)
}

// ── architecture bucket ──────────────────────────────────────────────────────────────────────────

/// The raw device facts the architecture, trust and shader-core-count decisions key on. Every field
/// is a direct copy of a Vulkan query result — no policy, no env, no config — so a test can state a
/// device by writing one of these down. (`VK_NV_cooperative_matrix2` has its own probe struct,
/// [`Coopmat2Probe`], because its shape list is not `Copy`.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DeviceProbe {
    /// `VkPhysicalDeviceProperties::vendorID` — the PCI vendor id (0x1002 AMD, 0x8086 Intel,
    /// 0x10de NVIDIA; see [`VENDOR_AMD`] and friends).
    pub vendor_id: u32,
    /// `VkPhysicalDeviceDriverProperties::driverID` (core Vulkan 1.2). [`vk::DriverId::default()`]
    /// (== `AMD_PROPRIETARY`, value 1) is NOT a safe "unknown": ash's Default is the first
    /// enumerant, so the probe must only fill this from a device that actually reports Vulkan 1.2+
    /// and leave `driver_id_reported` false otherwise.
    pub driver_id: vk::DriverId,
    /// Whether `driver_id` above came from a real query (see its note).
    pub driver_id_reported: bool,
    /// `VkPhysicalDeviceProperties::deviceType == INTEGRATED_GPU`.
    pub integrated: bool,
    /// `VkPhysicalDeviceSubgroupSizeControlProperties` min/max — `(0, 0)` when the device has no
    /// subgroup-size control (then no architecture below AMD/Intel can be identified).
    pub subgroup_min: u32,
    pub subgroup_max: u32,
    /// `VkPhysicalDeviceProperties::deviceID` — the PCI device id. Read for exactly one decision:
    /// Intel exposes no shader-core count in any property struct, so [`shader_core_count`] has to
    /// look the part up in a table (see [`intel_shader_core_count`]).
    pub device_id: u32,
    /// `VkPhysicalDeviceShaderCorePropertiesAMD::wavefrontsPerSimd`, 0 when the AMD extension is
    /// absent (which is itself the signal that the AMD probe cannot classify).
    pub wavefronts_per_simd: u32,
    /// `VK_AMD_shader_core_properties` advertised.
    pub has_amd_shader_core: bool,
    /// The three `VkPhysicalDeviceShaderCorePropertiesAMD` counts whose PRODUCT is the device's
    /// total compute-unit count; all 0 when `VK_AMD_shader_core_properties` is absent.
    pub shader_engine_count: u32,
    pub shader_arrays_per_engine_count: u32,
    pub compute_units_per_shader_array: u32,
    /// `VK_AMD_shader_core_properties2` advertised.
    pub has_amd_shader_core2: bool,
    /// `VkPhysicalDeviceShaderCoreProperties2AMD::activeComputeUnitCount` — the count with harvested
    /// CUs already excluded, which the v1 product above does not do. 0 when the extension is absent.
    pub active_compute_unit_count: u32,
    /// `VK_KHR_shader_integer_dot_product` advertised (the properties struct is core in 1.3, but
    /// upstream keys off the extension string and so does this).
    pub has_integer_dot: bool,
    /// `VkPhysicalDeviceShaderIntegerDotProductProperties::integerDotProduct4x8BitPackedSignedAccelerated`.
    pub dot4x8_signed_accelerated: bool,
    /// …`integerDotProduct4x8BitPackedMixedSignednessAccelerated` — the bit that separates RDNA3
    /// from RDNA2 upstream.
    pub dot4x8_mixed_accelerated: bool,
    /// `VK_KHR_cooperative_matrix` advertised (the STRING; the feature bit is a separate gate).
    pub has_coopmat_ext: bool,
    /// `VK_NV_shader_sm_builtins` advertised.
    pub has_nv_sm_builtins: bool,
    /// `VkPhysicalDeviceShaderSMBuiltinsPropertiesNV::shaderWarpsPerSM`, 0 when absent.
    pub warps_per_sm: u32,
    /// `VkPhysicalDeviceShaderSMBuiltinsPropertiesNV::shaderSMCount`, 0 when absent. A DIFFERENT
    /// field from [`warps_per_sm`](Self::warps_per_sm) above: that one buckets the architecture,
    /// this one counts the cores.
    pub sm_count: u32,
}

/// PCI vendor ids (PCI-SIG registry values — fixed outside this codebase, same three llama.cpp
/// spells out in `ggml-vulkan.cpp`).
pub(crate) const VENDOR_AMD: u32 = 0x1002;
pub(crate) const VENDOR_INTEL: u32 = 0x8086;
pub(crate) const VENDOR_NVIDIA: u32 = 0x10de;

/// The GPU architecture generation, bucketed the way llama.cpp's `get_device_architecture` buckets
/// it (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:404`, read at pin 030ebb5) — from PROBES, never from
/// PCI device ids. infr uses it for one thing today: deciding whether a driver's cooperative-matrix
/// enumeration can be believed ([`coopmat_trust`]).
///
/// `AmdRdna3` is upstream's `AMD_RDNA3` and means "RDNA3 **or newer**": the discriminator is the
/// mixed-signedness packed-dot bit, which RDNA4 also sets, and there is no RDNA4 bucket upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceArch {
    /// Not identified — every vendor this has no probe for, and any device whose probe extensions
    /// are missing. The neutral bucket: no rule may treat it as evidence of anything.
    Other,
    /// Wave64-only AMD (GCN/CDNA). infr REFUSES such a device earlier than this — the whole kernel
    /// set pins subgroup 32 — so it cannot reach a dispatch; it is classified anyway because the
    /// AMD coopmat trust rule is stated over architectures and a wave64 part misfiled as `Other`
    /// would read as "unknown" instead of "known-unsupported".
    AmdGcn,
    AmdRdna1,
    AmdRdna2,
    /// RDNA3 or newer (see the type note).
    AmdRdna3,
    /// Xe1/Gen (SIMD8) — Alchemist/Arc A-series and the Xe iGPUs.
    IntelXe1,
    /// Xe2 or newer (SIMD16) — Battlemage/Lunar Lake.
    IntelXe2,
    /// No cooperative matrix at all (pre-Turing).
    NvidiaPreTuring,
    /// Turing exactly (32 warps/SM).
    NvidiaTuring,
}

/// Bucket the device into a [`DeviceArch`] from probed properties.
///
/// A transcription of llama.cpp's `get_device_architecture` (verified line-by-line against
/// `ggml-vulkan.cpp:404-522` at pin 030ebb5), with upstream's fall-throughs preserved exactly:
/// a missing probe extension yields [`DeviceArch::Other`] rather than a guess, an Ampere-or-newer
/// NVIDIA part is `Other` (only Turing is singled out), and an NVIDIA part without
/// `VK_NV_shader_sm_builtins` is `Other` even if it is a Turing.
pub(crate) fn device_architecture(p: &DeviceProbe) -> DeviceArch {
    match p.vendor_id {
        VENDOR_AMD => {
            // Upstream requires all three probe extensions before classifying at all.
            if !(p.has_amd_shader_core && p.has_integer_dot && p.subgroup_min > 0) {
                return DeviceArch::Other;
            }
            if p.subgroup_max == 64 && p.subgroup_min == 64 {
                return DeviceArch::AmdGcn;
            }
            if p.subgroup_max == 64 && p.subgroup_min == 32 {
                // RDNA (dual wave mode). RDNA1 is the only generation with 20 wavefronts/SIMD;
                // RDNA3+ is the only one accelerating the MIXED-SIGNEDNESS packed dot product.
                if p.wavefronts_per_simd == 20 {
                    return DeviceArch::AmdRdna1;
                }
                if p.dot4x8_mixed_accelerated {
                    return DeviceArch::AmdRdna3;
                }
                return DeviceArch::AmdRdna2;
            }
            DeviceArch::Other
        }
        VENDOR_INTEL => {
            if !(p.has_integer_dot && p.subgroup_min > 0) {
                return DeviceArch::Other;
            }
            // minSubgroupSize IS the SIMD width: 16 on Xe2+, 8 on Xe1/Gen.
            if p.subgroup_min == 16 {
                return DeviceArch::IntelXe2;
            }
            if p.subgroup_min == 8 && p.dot4x8_signed_accelerated {
                return DeviceArch::IntelXe1;
            }
            DeviceArch::Other
        }
        VENDOR_NVIDIA => {
            if !p.has_coopmat_ext {
                return DeviceArch::NvidiaPreTuring;
            }
            if p.has_nv_sm_builtins && p.warps_per_sm == 32 {
                return DeviceArch::NvidiaTuring;
            }
            DeviceArch::Other
        }
        _ => DeviceArch::Other,
    }
}

// ── shader core count ────────────────────────────────────────────────────────────────────────────

/// How many independent shader cores this device has — NVIDIA SMs, AMD compute units, Intel
/// Xe-cores — or **0 when it cannot be determined**.
///
/// 0 is the codebase's existing spelling of "unknown" for this quantity
/// (`Capabilities::compute_units`, and `infr_core::integrated_ubatch_rows` already branches on it),
/// and it must never be read as "zero cores": a caller that cannot answer without a count has to
/// keep whatever it did before rather than derive something from 0.
///
/// The source order is llama.cpp's, from the `shader_core_count` assignment in `ggml_vk_get_device`
/// (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:6363-6371`, read at pin 030ebb5):
///
/// 1. `VK_NV_shader_sm_builtins`' `shaderSMCount` — NVIDIA's only report of the number.
/// 2. `VK_AMD_shader_core_properties2`' `activeComputeUnitCount`.
/// 3. Intel: a device-id table, because Intel exposes the count nowhere ([`intel_shader_core_count`]).
///
/// infr adds a FOURTH step upstream does not have: the `VK_AMD_shader_core_properties` (v1) product,
/// which is what this codebase read before this function existed. Dropping it would hand 0 to
/// `integrated_ubatch_rows` on any AMD part whose driver advertises v1 but not v2 — a behaviour
/// change on the one consumer that exists, in the direction of a smaller prefill chunk, for no gain.
/// v2 is preferred over it because `activeComputeUnitCount` excludes harvested CUs and the v1 product
/// counts the physical array.
///
/// UNVALIDATED per vendor: only the AMD steps run on hardware here (an RX 7900 XTX and a Raphael
/// iGPU, both RADV, both advertising v1 and v2). The NVIDIA and Intel steps have never executed —
/// there is no such part on this machine — and their correctness rests on the transcription and the
/// tests below, nothing more.
pub(crate) fn shader_core_count(p: &DeviceProbe) -> u32 {
    if p.has_nv_sm_builtins && p.sm_count > 0 {
        return p.sm_count;
    }
    if p.has_amd_shader_core2 && p.active_compute_unit_count > 0 {
        return p.active_compute_unit_count;
    }
    if p.vendor_id == VENDOR_INTEL {
        let n = intel_shader_core_count(p.device_id);
        if n > 0 {
            return n;
        }
    }
    if p.has_amd_shader_core {
        return p.shader_engine_count
            * p.shader_arrays_per_engine_count
            * p.compute_units_per_shader_array;
    }
    0
}

/// Intel Xe-core counts keyed on PCI device id, because Intel reports the number in no property
/// struct at all.
///
/// **Transcribed verbatim** from llama.cpp's `ggml_vk_intel_shader_core_count`
/// (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:18805-18846`, read at pin 030ebb5). Every id and count
/// here is upstream's; none was inferred, and an id upstream does not list returns 0 (unknown)
/// rather than a nearby part's number. The comments are the SKU names upstream records.
///
/// UNVALIDATED: no Intel GPU exists on this machine, so not one of these ids has ever been matched
/// by this code against a real device. The test below checks the transcription, not the hardware.
fn intel_shader_core_count(device_id: u32) -> u32 {
    match device_id {
        0x56A6 => 6,  // A310
        0x5693 => 8,  // A370M
        0x56A5 => 8,  // A380
        0x56B1 => 8,  // Pro A40/A50
        0x5697 => 12, // A530M
        0x5692 => 16, // A550M
        0x56B3 => 16, // Pro A60
        0x56A2 => 24, // A580
        0x5691 => 28, // A730M
        0x56A1 => 28, // A750
        0x56A0 => 32, // A770
        0x5690 => 32, // A770M
        0xE212 => 16, // Pro B50
        0xE20C => 18, // B570
        0xE20B => 20, // B580
        0xE211 => 20, // Pro B60
        0xB080 => 12, // PTL Xe3 LPG 2x6 (12 subslices)
        _ => 0,
    }
}

// ── cooperative-matrix trust ─────────────────────────────────────────────────────────────────────

/// How far this driver's cooperative-matrix ENUMERATION may be believed.
///
/// infr is capability-first: it reads no vendor id for any other decision, and that is what lets
/// new hardware run with no code. This is the one narrow exception, because capability-first
/// assumes the device tells the truth and llama.cpp has documented two drivers that do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CoopmatTrust {
    /// Believe the enumeration — every shape the device listed is usable. Every device except the
    /// two cases below.
    Enumerated,
    /// The 16x16x16 tier is refused; only [`COOPMAT_TILE_8`] may be selected, and that shape
    /// already requires the default-off `INFR_CM_8X8=1` opt-in whose own documentation carries this
    /// hardware's regression warning. Intel pre-Xe2 discrete (Arc A770 and friends).
    Tile8Only(&'static str),
    /// Refuse cooperative matrix outright: this driver reports the unit on hardware that does not
    /// have it.
    Refused(&'static str),
}

/// Decide how far to trust `probe`'s cooperative-matrix enumeration.
///
/// The two rules are llama.cpp's, from `ggml_vk_khr_cooperative_matrix_support`
/// (`ggml-vulkan.cpp:18788-18803`, read at pin 030ebb5), which is the evidence that both cases are
/// real regressions upstream hit on hardware:
///
/// * **AMD's proprietary drivers report cooperative matrix on all GPUs** (upstream's own comment).
///   `AMD_PROPRIETARY` (amdgpu-pro/Windows) and `AMD_OPEN_SOURCE` (AMDVLK) are therefore believed
///   only on RDNA3+. `MESA_RADV` — this box — is believed on any architecture, exactly as upstream
///   does, so RADV behaviour is unchanged by this rule.
/// * **Intel pre-Xe2 regresses on cooperative matrix** (upstream allows Xe2, plus an INTEGRATED Xe1
///   on the Windows proprietary driver). infr refuses only the 16x16x16 tier there rather than all
///   coopmat: the 8x8x16 tile Alchemist actually enumerates is already behind a default-off opt-in
///   that says so, and keeping it reachable is what leaves the `_cm8` A/B path alive on the
///   hardware it was written for. The hole this closes is an Arc part that enumerates 16x16x16 and
///   would otherwise sail into the full kernel set.
///
/// Anything else — NVIDIA, RADV, MoltenVK, a vendor with no rule — is believed. That default is the
/// point: a deny-list must not become the thing new hardware has to be added to.
///
/// A device that never reported a `driverID` (pre-1.2, or a driver that ignored the query) is
/// believed too. The AMD rule needs a POSITIVE identification of a lying driver, and ash's
/// `DriverId::default()` is `AMD_PROPRIETARY` — so treating "unreported" as a driver id would
/// refuse coopmat on every device whose driver skipped the field.
pub(crate) fn coopmat_trust(p: &DeviceProbe, arch: DeviceArch) -> CoopmatTrust {
    match p.vendor_id {
        VENDOR_INTEL => {
            let allowed = arch == DeviceArch::IntelXe2
                || (arch == DeviceArch::IntelXe1
                    && p.integrated
                    && p.driver_id_reported
                    && p.driver_id == vk::DriverId::INTEL_PROPRIETARY_WINDOWS);
            if allowed {
                CoopmatTrust::Enumerated
            } else {
                CoopmatTrust::Tile8Only(
                    "Intel pre-Xe2: cooperative matrix is a documented performance regression on \
                     this generation (llama.cpp allows it only on Xe2+, or an integrated Xe1 on \
                     the Windows proprietary driver), so only the 8x8x16 tile stays reachable and \
                     only under INFR_CM_8X8=1",
                )
            }
        }
        VENDOR_AMD => {
            let proprietary = p.driver_id_reported
                && matches!(
                    p.driver_id,
                    vk::DriverId::AMD_PROPRIETARY | vk::DriverId::AMD_OPEN_SOURCE
                );
            if proprietary && arch != DeviceArch::AmdRdna3 {
                return CoopmatTrust::Refused(
                    "AMD proprietary/AMDVLK driver: it reports cooperative-matrix support on ALL \
                     GPUs, and only RDNA3+ actually has the unit (llama.cpp restricts the same two \
                     driverIDs the same way). Use the Mesa RADV driver on this device, or run the \
                     non-coopmat tiers",
                );
            }
            CoopmatTrust::Enumerated
        }
        _ => CoopmatTrust::Enumerated,
    }
}

/// Whether `shape` may be selected under `trust` — the filter the enumerated shape list goes
/// through before [`select_coopmat_shape`] applies its preference order.
pub(crate) fn coopmat_shape_trusted(trust: CoopmatTrust, shape: (u32, u32, u32)) -> bool {
    match trust {
        CoopmatTrust::Enumerated => true,
        CoopmatTrust::Tile8Only(_) => shape == COOPMAT_TILE_8,
        CoopmatTrust::Refused(_) => false,
    }
}

// ── VK_NV_cooperative_matrix2 gate ───────────────────────────────────────────────────────────────

/// One row of `vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV` — a
/// (component types, granularity, workgroup size) combination the device can build a
/// flexible-dimension cooperative matrix for. Field-for-field
/// `VkCooperativeMatrixFlexibleDimensionsPropertiesNV` minus its `sType`/`pNext`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlexibleDimension {
    pub m_granularity: u32,
    pub n_granularity: u32,
    pub k_granularity: u32,
    pub a_type: vk::ComponentTypeKHR,
    pub b_type: vk::ComponentTypeKHR,
    pub c_type: vk::ComponentTypeKHR,
    pub result_type: vk::ComponentTypeKHR,
    pub saturating_accumulation: bool,
    pub scope: vk::ScopeKHR,
    pub workgroup_invocations: u32,
}

/// Everything `VK_NV_cooperative_matrix2` reports about a device, as probed. Every field is a
/// direct copy of a query result; the policy is [`check_coopmat2_support`] alone.
///
/// The `*_reported` flag matters more here than anywhere else in this file, because a coopmat2
/// kernel that runs on a device which merely LOOKS capable produces wrong numbers rather than an
/// error. A query that could not be made leaves its flag false, and the gate refuses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Coopmat2Probe {
    /// `VK_NV_cooperative_matrix2` in the device's extension list.
    pub has_ext: bool,
    /// `VkPhysicalDeviceCooperativeMatrix2FeaturesNV`, in declaration order. All false when the
    /// features struct could not be chained (i.e. when `has_ext` is false).
    pub workgroup_scope: bool,
    pub flexible_dimensions: bool,
    pub reductions: bool,
    pub conversions: bool,
    pub per_element_operations: bool,
    pub tensor_addressing: bool,
    pub block_loads: bool,
    /// `VkPhysicalDeviceVulkan12Features::bufferDeviceAddress` — not part of the extension, but
    /// upstream's gate requires it because its coopmat2 shaders address tensors by pointer.
    pub buffer_device_address: bool,
    /// `VkPhysicalDeviceCooperativeMatrix2PropertiesNV::cooperativeMatrixFlexibleDimensionsMaxDimension`.
    pub flexible_dimensions_max_dimension: u32,
    /// Whether `vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV` was actually
    /// resolved and called. False means the shape list below is UNKNOWN, not empty — and the gate
    /// treats the two the same way (refuse) rather than reading silence as agreement.
    pub flexible_dimensions_reported: bool,
    pub flexible_dimensions_list: Vec<FlexibleDimension>,
}

/// Required workgroup sizes and the largest granularity acceptable at each, from upstream's gate.
/// A granularity is a *minimum step*, so a device qualifies when its value is at or below the
/// number here — the tile upstream's shaders want is `32x16x16` at 128 invocations and `32x32x16`
/// at 256.
const COOPMAT2_REQUIRED_TILES: [(u32, u32, u32, u32); 2] = [
    // (workgroupInvocations, max MGranularity, max NGranularity, max KGranularity)
    (128, 32, 16, 16),
    (256, 32, 32, 16),
];

/// The smallest `cooperativeMatrixFlexibleDimensionsMaxDimension` upstream will accept.
const COOPMAT2_MIN_MAX_DIMENSION: u32 = 512;

/// Whether this device may run a `VK_NV_cooperative_matrix2` path.
///
/// A transcription of llama.cpp's coopmat2 gate (`ggml-vulkan.cpp:6675-6776`, read at pin 030ebb5),
/// which is deliberately much more than the extension string: the extension can be present while the
/// features that make it usable are off (RADV exposes it only behind the `radv_cooperative_matrix2_nv`
/// driconf flag, and lavapipe exposes it with a different feature set again). Upstream requires all
/// seven feature bits, `bufferDeviceAddress`, then fp16 A/B with BOTH an fp16 and an fp32 accumulator
/// at BOTH the 128- and 256-invocation workgroup sizes, and `maxDimension >= 512`.
///
/// **Fails closed.** Every path that cannot establish a positive answer returns `Err`: no extension,
/// a clear feature bit, a flexible-dimensions list that could not be queried, a missing tile shape,
/// a `maxDimension` the device did not report (0 < 512). "Unknown" is never "supported".
///
/// infr does NOT copy upstream's optional bf16 probe: that only sets a separate
/// `coopmat2_bf16_support` flag feeding bf16 coopmat2 shaders, and infr has no coopmat2 shader of any
/// kind yet.
///
/// **NEVER EXERCISED ON QUALIFYING HARDWARE.** This machine has an RX 7900 XTX (RADV, which does not
/// advertise the extension) and a lavapipe software device (which does). What has actually been run
/// here is the refusal path; no device on this box has ever reached `Ok`, so the accept side is
/// checked by the unit tests below and by nothing else.
pub(crate) fn check_coopmat2_support(p: &Coopmat2Probe) -> Result<(), String> {
    if !p.has_ext {
        return Err("VK_NV_cooperative_matrix2 is not in this device's extension list".into());
    }
    for (present, name) in [
        (p.workgroup_scope, "cooperativeMatrixWorkgroupScope"),
        (p.flexible_dimensions, "cooperativeMatrixFlexibleDimensions"),
        (p.reductions, "cooperativeMatrixReductions"),
        (p.conversions, "cooperativeMatrixConversions"),
        (
            p.per_element_operations,
            "cooperativeMatrixPerElementOperations",
        ),
        (p.tensor_addressing, "cooperativeMatrixTensorAddressing"),
        (p.block_loads, "cooperativeMatrixBlockLoads"),
        (p.buffer_device_address, "bufferDeviceAddress"),
    ] {
        if !present {
            return Err(format!(
                "VK_NV_cooperative_matrix2 is advertised but the {name} feature is not enabled — \
                 every one of the seven coopmat2 features plus bufferDeviceAddress is required"
            ));
        }
    }
    if !p.flexible_dimensions_reported {
        return Err(
            "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV could not be \
             called, so this device's flexible-dimension shapes are unknown — refusing rather than \
             assuming"
                .into(),
        );
    }
    for (invocations, max_m, max_n, max_k) in COOPMAT2_REQUIRED_TILES {
        for (acc, name) in [
            (vk::ComponentTypeKHR::FLOAT16, "fp16"),
            (vk::ComponentTypeKHR::FLOAT32, "fp32"),
        ] {
            let found = p.flexible_dimensions_list.iter().any(|d| {
                !d.saturating_accumulation
                    && d.scope == vk::ScopeKHR::WORKGROUP
                    && d.a_type == vk::ComponentTypeKHR::FLOAT16
                    && d.b_type == vk::ComponentTypeKHR::FLOAT16
                    && d.c_type == acc
                    && d.result_type == acc
                    && d.workgroup_invocations == invocations
                    && d.m_granularity <= max_m
                    && d.n_granularity <= max_n
                    && d.k_granularity <= max_k
            });
            if !found {
                return Err(format!(
                    "this device enumerates no fp16xfp16 -> {name} workgroup-scope flexible \
                     dimension at {invocations} invocations with granularity \
                     <= {max_m}x{max_n}x{max_k}"
                ));
            }
        }
    }
    if p.flexible_dimensions_max_dimension < COOPMAT2_MIN_MAX_DIMENSION {
        return Err(format!(
            "cooperativeMatrixFlexibleDimensionsMaxDimension is {}, below the \
             {COOPMAT2_MIN_MAX_DIMENSION} a coopmat2 path needs",
            p.flexible_dimensions_max_dimension
        ));
    }
    Ok(())
}

// ── int8 coopmat accumulator fragment layout ─────────────────────────────────────────────────────

/// Elements of a 16x16 accumulator each lane owns at subgroup size 32 (256 elements / 32 lanes).
/// `native_gemm_i8cm_q8_0.comp` unrolls its in-fragment epilogue over exactly this many.
const FRAG_ELEMS_PER_LANE: i32 = 8;
/// The subgroup size the int8 coopmat kernel is pinned to (`kernel_sg(..., 32)`) and whose lane
/// numbering its `lane>>4` / `lane&15` mapping is written for.
const FRAG_SUBGROUP: i32 = 32;
/// Header words the layout probe writes before the matrix: `[0]` elements per lane, `[1]` subgroup
/// size.
pub(crate) const FRAG_PROBE_HEADER: usize = 2;
/// Total i32 words the layout probe's output buffer holds.
pub(crate) const FRAG_PROBE_WORDS: usize = FRAG_PROBE_HEADER + 16 * 16;

/// The known-answer inputs for the accumulator-layout probe: `(A row-major, B column-major)`, each
/// 16x16 int8, laid out exactly as `coopmat_i8_layout.comp` loads them.
///
/// `A[r][k] = [1, r, 0…]`, `B[k][c] = [c+1, 16, 0…]`, so the product is
/// `C[r][c] = 16*r + c + 1` — every element DISTINCT (a scrambled fragment layout cannot
/// accidentally reproduce it) and every element NON-ZERO (an element the layout never writes stays
/// at the output buffer's zero-init and is therefore caught too).
pub(crate) fn frag_probe_inputs() -> (Vec<i8>, Vec<i8>) {
    let mut a = vec![0i8; 16 * 16];
    let mut b = vec![0i8; 16 * 16];
    for r in 0..16i8 {
        a[r as usize * 16] = 1; // A[r][0] = 1
        a[r as usize * 16 + 1] = r; // A[r][1] = r
    }
    for c in 0..16i8 {
        b[c as usize * 16] = c + 1; // B[0][c] = c+1 (column-major: b[c*16 + k])
        b[c as usize * 16 + 1] = 16; // B[1][c] = 16
    }
    (a, b)
}

/// The known product the probe must reproduce: `C[row][col]`.
pub(crate) fn frag_probe_expected(row: usize, col: usize) -> i32 {
    16 * row as i32 + col as i32 + 1
}

/// Verify a layout-probe readback: does this device lay its int8 coopmat accumulator out the way
/// `native_gemm_i8cm_q8_0.comp` assumes?
///
/// The kernel reads accumulator element `csub[i]` as matrix element
/// `(row, col) = (2*i + (lane>>4), lane&15)` — a mapping `KHR_cooperative_matrix` fixes per
/// IMPLEMENTATION, not across implementations, and which was derived empirically on RADV/RDNA3.
/// The probe shader multiplies [`frag_probe_inputs`] and writes each element back through that same
/// assumed mapping, so a device that lays the fragment out differently produces a matrix that is
/// not the known product. Returns the reason on mismatch, for a log line and a refused tier — the
/// alternative is plausible wrong numbers on hardware nobody here can test.
pub(crate) fn check_i8_coopmat_layout(out: &[i32]) -> Result<(), String> {
    if out.len() < FRAG_PROBE_WORDS {
        return Err(format!(
            "layout probe returned {} words, expected {FRAG_PROBE_WORDS}",
            out.len()
        ));
    }
    if out[0] != FRAG_ELEMS_PER_LANE {
        return Err(format!(
            "this device's 16x16 int32 coopmat accumulator gives each lane {} elements, but the \
             kernel's in-fragment epilogue is written for {FRAG_ELEMS_PER_LANE}",
            out[0]
        ));
    }
    if out[1] != FRAG_SUBGROUP {
        return Err(format!(
            "the layout probe ran at subgroup size {}, not the {FRAG_SUBGROUP} the kernel pins — \
             its lane->(row,col) arithmetic does not hold at any other width",
            out[1]
        ));
    }
    for row in 0..16usize {
        for col in 0..16usize {
            let got = out[FRAG_PROBE_HEADER + row * 16 + col];
            let want = frag_probe_expected(row, col);
            if got != want {
                return Err(format!(
                    "accumulator element ({row},{col}) read back as {got}, expected {want} — this \
                     driver does not lay a 16x16 SINT32 accumulator fragment out as \
                     (2*i + (lane>>4), lane&15), which is what the int8 coopmat GEMM's in-fragment \
                     descale assumes"
                ));
            }
        }
    }
    Ok(())
}

// ── device limits ────────────────────────────────────────────────────────────────────────────────

/// Refuse a kernel whose push-constant block does not fit this device's `maxPushConstantsSize`.
///
/// Vulkan only guarantees 128 bytes and nothing here queried the real number before, so every
/// push block was safe by inspection rather than by check. `vkCreatePipelineLayout` would fail the
/// VUID anyway, but as a driver error at pipeline-build time naming nothing useful; this names the
/// kernel and the two numbers.
pub(crate) fn check_push_constant_size(
    name: &str,
    push_size: u32,
    limit: u32,
) -> Result<(), String> {
    if push_size > limit {
        return Err(format!(
            "kernel {name:?} pushes {push_size} bytes of push constants, but this device's \
             maxPushConstantsSize is {limit}"
        ));
    }
    Ok(())
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── tile selection ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn coopmat_shape_selection() {
        let t16 = COOPMAT_TILE_16;
        let t8 = COOPMAT_TILE_8;
        // RADV/RDNA3 + NVIDIA: 16x16x16 always wins, opt-in or not.
        let radv = [t16];
        assert_eq!(select_coopmat_shape(radv, false), Some(t16));
        assert_eq!(select_coopmat_shape(radv, true), Some(t16));
        // A device listing both keeps the production shape even with the knob set.
        let both = [t8, t16];
        assert_eq!(select_coopmat_shape(both, false), Some(t16));
        assert_eq!(select_coopmat_shape(both, true), Some(t16));
        // Intel Arc (ANV): 8x8x16 only — dark unless the opt-in asks for it.
        let anv = [t8];
        assert_eq!(select_coopmat_shape(anv, false), None);
        assert_eq!(select_coopmat_shape(anv, true), Some(t8));
        // Nothing enumerated, and a shape no kernel exists for.
        assert_eq!(select_coopmat_shape([], false), None);
        assert_eq!(select_coopmat_shape([], true), None);
        assert_eq!(select_coopmat_shape([(32, 32, 16)], true), None);
    }

    // ── architecture bucketing ──────────────────────────────────────────────────────────────────

    /// This box: RX 7900 XTX on RADV. Every other synthetic device below is written against
    /// llama.cpp's probe, since no such hardware exists here.
    fn rdna3_radv() -> DeviceProbe {
        DeviceProbe {
            vendor_id: VENDOR_AMD,
            driver_id: vk::DriverId::MESA_RADV,
            driver_id_reported: true,
            integrated: false,
            subgroup_min: 32,
            subgroup_max: 64,
            device_id: 0x744c,
            wavefronts_per_simd: 16,
            has_amd_shader_core: true,
            // Both AMD sources agree on this box: the v1 PRODUCT and the v2 active count each came
            // back as the banner's `cores:96` when the other branch was disabled. The three v1
            // factors below are a synthetic factorisation of that measured product — the individual
            // fields were never read out, only their product.
            shader_engine_count: 6,
            shader_arrays_per_engine_count: 2,
            compute_units_per_shader_array: 8,
            has_amd_shader_core2: true,
            active_compute_unit_count: 96,
            has_integer_dot: true,
            dot4x8_signed_accelerated: true,
            dot4x8_mixed_accelerated: true,
            has_coopmat_ext: true,
            has_nv_sm_builtins: false,
            warps_per_sm: 0,
            sm_count: 0,
        }
    }

    fn arc_a770_anv() -> DeviceProbe {
        DeviceProbe {
            vendor_id: VENDOR_INTEL,
            driver_id: vk::DriverId::INTEL_OPEN_SOURCE_MESA,
            driver_id_reported: true,
            integrated: false,
            subgroup_min: 8,
            subgroup_max: 32,
            device_id: 0x56A0,
            wavefronts_per_simd: 0,
            has_amd_shader_core: false,
            shader_engine_count: 0,
            shader_arrays_per_engine_count: 0,
            compute_units_per_shader_array: 0,
            has_amd_shader_core2: false,
            active_compute_unit_count: 0,
            has_integer_dot: true,
            dot4x8_signed_accelerated: true,
            dot4x8_mixed_accelerated: false,
            has_coopmat_ext: true,
            has_nv_sm_builtins: false,
            warps_per_sm: 0,
            sm_count: 0,
        }
    }

    fn nvidia(warps: u32, coopmat: bool) -> DeviceProbe {
        DeviceProbe {
            vendor_id: VENDOR_NVIDIA,
            driver_id: vk::DriverId::NVIDIA_PROPRIETARY,
            driver_id_reported: true,
            integrated: false,
            subgroup_min: 32,
            subgroup_max: 32,
            device_id: 0x2684,
            wavefronts_per_simd: 0,
            has_amd_shader_core: false,
            shader_engine_count: 0,
            shader_arrays_per_engine_count: 0,
            compute_units_per_shader_array: 0,
            has_amd_shader_core2: false,
            active_compute_unit_count: 0,
            has_integer_dot: true,
            dot4x8_signed_accelerated: true,
            dot4x8_mixed_accelerated: true,
            has_coopmat_ext: coopmat,
            has_nv_sm_builtins: warps > 0,
            warps_per_sm: warps,
            sm_count: if warps > 0 { 128 } else { 0 },
        }
    }

    #[test]
    fn amd_architectures_split_on_wave_mode_and_dot_bits() {
        // RDNA3 (this box): dual wave mode + the mixed-signedness packed dot.
        assert_eq!(device_architecture(&rdna3_radv()), DeviceArch::AmdRdna3);
        // RDNA2: same wave mode, no mixed-signedness bit.
        let mut rdna2 = rdna3_radv();
        rdna2.dot4x8_mixed_accelerated = false;
        assert_eq!(device_architecture(&rdna2), DeviceArch::AmdRdna2);
        // RDNA1: 20 wavefronts/SIMD, checked BEFORE the dot bit.
        let mut rdna1 = rdna3_radv();
        rdna1.wavefronts_per_simd = 20;
        assert_eq!(device_architecture(&rdna1), DeviceArch::AmdRdna1);
        // GCN/CDNA: wave64 only.
        let mut gcn = rdna3_radv();
        gcn.subgroup_min = 64;
        assert_eq!(device_architecture(&gcn), DeviceArch::AmdGcn);
        // A missing probe extension must NOT be guessed at.
        let mut no_core = rdna3_radv();
        no_core.has_amd_shader_core = false;
        assert_eq!(device_architecture(&no_core), DeviceArch::Other);
        let mut no_dot = rdna3_radv();
        no_dot.has_integer_dot = false;
        assert_eq!(device_architecture(&no_dot), DeviceArch::Other);
        let mut no_sgctl = rdna3_radv();
        no_sgctl.subgroup_min = 0;
        no_sgctl.subgroup_max = 0;
        assert_eq!(device_architecture(&no_sgctl), DeviceArch::Other);
    }

    #[test]
    fn intel_architectures_split_on_simd_width() {
        assert_eq!(device_architecture(&arc_a770_anv()), DeviceArch::IntelXe1);
        let mut xe2 = arc_a770_anv();
        xe2.subgroup_min = 16;
        assert_eq!(device_architecture(&xe2), DeviceArch::IntelXe2);
        // SIMD8 without the signed packed-dot acceleration is not identified as Xe1.
        let mut odd = arc_a770_anv();
        odd.dot4x8_signed_accelerated = false;
        assert_eq!(device_architecture(&odd), DeviceArch::Other);
    }

    #[test]
    fn nvidia_architectures_split_on_coopmat_and_warps() {
        // Pre-Turing is identified by the ABSENCE of the coopmat extension.
        assert_eq!(
            device_architecture(&nvidia(0, false)),
            DeviceArch::NvidiaPreTuring
        );
        assert_eq!(
            device_architecture(&nvidia(32, true)),
            DeviceArch::NvidiaTuring
        );
        // Ampere and newer report 48 warps/SM and are deliberately NOT singled out.
        assert_eq!(device_architecture(&nvidia(48, true)), DeviceArch::Other);
        // Turing without VK_NV_shader_sm_builtins cannot be identified (upstream's behaviour too).
        let mut turing_no_builtins = nvidia(32, true);
        turing_no_builtins.has_nv_sm_builtins = false;
        assert_eq!(device_architecture(&turing_no_builtins), DeviceArch::Other);
    }

    // ── shader core count ───────────────────────────────────────────────────────────────────────

    #[test]
    fn shader_core_count_per_vendor() {
        // (probe, expected count, what the case is)
        let mut amd_v1_only = rdna3_radv();
        amd_v1_only.has_amd_shader_core2 = false;
        amd_v1_only.active_compute_unit_count = 0;
        let mut amd_harvested = rdna3_radv();
        amd_harvested.active_compute_unit_count = 84; // v2 reports the harvested part, v1 the array
        let mut amd_none = rdna3_radv();
        amd_none.has_amd_shader_core = false;
        amd_none.has_amd_shader_core2 = false;
        amd_none.active_compute_unit_count = 0;
        let mut nv_no_builtins = nvidia(48, true);
        nv_no_builtins.has_nv_sm_builtins = false;
        let mut intel_unlisted = arc_a770_anv();
        intel_unlisted.device_id = 0xDEAD;
        let mut intel_b580 = arc_a770_anv();
        intel_b580.device_id = 0xE20B;
        let cases = [
            (rdna3_radv(), 96, "AMD with both extensions prefers v2"),
            (amd_v1_only, 96, "AMD v1 product is the fallback infr keeps"),
            (
                amd_harvested,
                84,
                "v2's active count wins over the v1 product",
            ),
            (amd_none, 0, "AMD with neither extension is unknown"),
            (nvidia(48, true), 128, "NVIDIA shaderSMCount"),
            (nv_no_builtins, 0, "NVIDIA without the extension is unknown"),
            (arc_a770_anv(), 32, "Intel A770 from the transcribed table"),
            (intel_b580, 20, "Intel B580 from the transcribed table"),
            (intel_unlisted, 0, "an Intel id upstream does not list"),
            (DeviceProbe::default(), 0, "a device that reported nothing"),
        ];
        for (probe, want, why) in cases {
            assert_eq!(shader_core_count(&probe), want, "{why}");
        }
    }

    #[test]
    fn intel_core_table_matches_upstream() {
        // The full transcription of `ggml_vk_intel_shader_core_count`, re-stated here so a typo in
        // either copy fails rather than silently mis-sizing an Intel part. Ids upstream groups on a
        // shared `return` are listed individually.
        let upstream = [
            (0x56A6u32, 6u32),
            (0x5693, 8),
            (0x56A5, 8),
            (0x56B1, 8),
            (0x5697, 12),
            (0x5692, 16),
            (0x56B3, 16),
            (0x56A2, 24),
            (0x5691, 28),
            (0x56A1, 28),
            (0x56A0, 32),
            (0x5690, 32),
            (0xE212, 16),
            (0xE20C, 18),
            (0xE20B, 20),
            (0xE211, 20),
            (0xB080, 12),
        ];
        for (id, want) in upstream {
            assert_eq!(intel_shader_core_count(id), want, "device id {id:#06x}");
        }
        // Anything not in the table is UNKNOWN, never a nearby part's count.
        for id in [0u32, 0x56A7, 0xE20D, 0xFFFF] {
            assert_eq!(intel_shader_core_count(id), 0, "device id {id:#06x}");
        }
        // …and the table is only consulted for Intel: an AMD part whose deviceID happens to collide
        // with an Arc id must not pick up its count.
        let mut collide = rdna3_radv();
        collide.device_id = 0x56A0;
        collide.has_amd_shader_core = false;
        collide.has_amd_shader_core2 = false;
        collide.active_compute_unit_count = 0;
        assert_eq!(shader_core_count(&collide), 0);
    }

    // ── coopmat trust (the two lying drivers) ───────────────────────────────────────────────────

    fn trust_of(p: &DeviceProbe) -> CoopmatTrust {
        coopmat_trust(p, device_architecture(p))
    }

    #[test]
    fn radv_and_nvidia_are_believed() {
        // This box, and the vendor with no rule at all: enumeration is authoritative.
        assert_eq!(trust_of(&rdna3_radv()), CoopmatTrust::Enumerated);
        let mut rdna2_radv = rdna3_radv();
        rdna2_radv.dot4x8_mixed_accelerated = false;
        assert_eq!(
            trust_of(&rdna2_radv),
            CoopmatTrust::Enumerated,
            "RADV is trusted on ANY architecture — the AMD rule is about the proprietary drivers"
        );
        assert_eq!(trust_of(&nvidia(48, true)), CoopmatTrust::Enumerated);
        assert_eq!(trust_of(&nvidia(32, true)), CoopmatTrust::Enumerated);
        // A device that never reported a driverID is believed (ash's DriverId default is
        // AMD_PROPRIETARY, so an "unreported" id must never be read as one).
        let mut unreported = rdna3_radv();
        unreported.driver_id = vk::DriverId::AMD_PROPRIETARY;
        unreported.driver_id_reported = false;
        unreported.dot4x8_mixed_accelerated = false; // i.e. RDNA2, which the rule would refuse
        assert_eq!(trust_of(&unreported), CoopmatTrust::Enumerated);
    }

    #[test]
    fn amd_proprietary_drivers_are_believed_only_on_rdna3() {
        for driver in [vk::DriverId::AMD_PROPRIETARY, vk::DriverId::AMD_OPEN_SOURCE] {
            let mut rdna3 = rdna3_radv();
            rdna3.driver_id = driver;
            assert_eq!(
                trust_of(&rdna3),
                CoopmatTrust::Enumerated,
                "{driver:?} on RDNA3 is the one case upstream allows"
            );
            // RDNA2 / RDNA1 / GCN / unidentified: the driver claims a unit the part lacks.
            let mut rdna2 = rdna3;
            rdna2.dot4x8_mixed_accelerated = false;
            let mut rdna1 = rdna3;
            rdna1.wavefronts_per_simd = 20;
            let mut gcn = rdna3;
            gcn.subgroup_min = 64;
            let mut unknown = rdna3;
            unknown.has_amd_shader_core = false;
            for p in [rdna2, rdna1, gcn, unknown] {
                assert!(
                    matches!(trust_of(&p), CoopmatTrust::Refused(_)),
                    "{driver:?} on {:?} must be refused",
                    device_architecture(&p)
                );
            }
        }
    }

    #[test]
    fn intel_pre_xe2_keeps_only_the_opt_in_tile() {
        // Arc A770 (discrete Xe1, ANV): the 16x16x16 tier is refused BY RULE. It already never
        // engaged here because A770 enumerates 8x8x16 only — this is what closes the hole for an
        // Arc part that DOES enumerate 16x16x16.
        let a770 = arc_a770_anv();
        let trust = trust_of(&a770);
        assert!(matches!(trust, CoopmatTrust::Tile8Only(_)), "{trust:?}");
        assert!(!coopmat_shape_trusted(trust, COOPMAT_TILE_16));
        assert!(coopmat_shape_trusted(trust, COOPMAT_TILE_8));
        // Xe2 (Battlemage) is trusted with everything it enumerates.
        let mut xe2 = arc_a770_anv();
        xe2.subgroup_min = 16;
        assert_eq!(trust_of(&xe2), CoopmatTrust::Enumerated);
        // An INTEGRATED Xe1 on the Windows proprietary driver is upstream's one exception…
        let mut xe1_win = arc_a770_anv();
        xe1_win.integrated = true;
        xe1_win.driver_id = vk::DriverId::INTEL_PROPRIETARY_WINDOWS;
        assert_eq!(trust_of(&xe1_win), CoopmatTrust::Enumerated);
        // …and it needs BOTH halves: integrated on Mesa, or discrete on Windows, stays refused.
        let mut xe1_mesa_igpu = xe1_win;
        xe1_mesa_igpu.driver_id = vk::DriverId::INTEL_OPEN_SOURCE_MESA;
        assert!(matches!(
            trust_of(&xe1_mesa_igpu),
            CoopmatTrust::Tile8Only(_)
        ));
        let mut xe1_win_dgpu = xe1_win;
        xe1_win_dgpu.integrated = false;
        assert!(matches!(
            trust_of(&xe1_win_dgpu),
            CoopmatTrust::Tile8Only(_)
        ));
    }

    #[test]
    fn trust_filters_the_shape_list_selection_sees() {
        // The filter is how a refusal reaches the tile picker: a refused device's enumeration is
        // emptied, so `select_coopmat_shape` returns None with no special case of its own.
        let refused = CoopmatTrust::Refused("test");
        let shapes = [COOPMAT_TILE_16, COOPMAT_TILE_8];
        let keep = |t| {
            shapes
                .into_iter()
                .filter(move |&s| coopmat_shape_trusted(t, s))
        };
        assert_eq!(select_coopmat_shape(keep(refused), true), None);
        assert_eq!(select_coopmat_shape(keep(refused), false), None);
        let tile8 = CoopmatTrust::Tile8Only("test");
        assert_eq!(
            select_coopmat_shape(keep(tile8), true),
            Some(COOPMAT_TILE_8)
        );
        assert_eq!(select_coopmat_shape(keep(tile8), false), None);
        assert_eq!(
            select_coopmat_shape(keep(CoopmatTrust::Enumerated), false),
            Some(COOPMAT_TILE_16)
        );
    }

    // ── coopmat2 gate ───────────────────────────────────────────────────────────────────────────

    fn flex(
        invocations: u32,
        (m, n, k): (u32, u32, u32),
        acc: vk::ComponentTypeKHR,
    ) -> FlexibleDimension {
        FlexibleDimension {
            m_granularity: m,
            n_granularity: n,
            k_granularity: k,
            a_type: vk::ComponentTypeKHR::FLOAT16,
            b_type: vk::ComponentTypeKHR::FLOAT16,
            c_type: acc,
            result_type: acc,
            saturating_accumulation: false,
            scope: vk::ScopeKHR::WORKGROUP,
            workgroup_invocations: invocations,
        }
    }

    /// A synthetic device that satisfies upstream's gate exactly. No hardware here produces this —
    /// it is the shape the gate is written against, not one that was observed.
    fn coopmat2_ok() -> Coopmat2Probe {
        Coopmat2Probe {
            has_ext: true,
            workgroup_scope: true,
            flexible_dimensions: true,
            reductions: true,
            conversions: true,
            per_element_operations: true,
            tensor_addressing: true,
            block_loads: true,
            buffer_device_address: true,
            flexible_dimensions_max_dimension: 512,
            flexible_dimensions_reported: true,
            flexible_dimensions_list: vec![
                flex(128, (32, 16, 16), vk::ComponentTypeKHR::FLOAT16),
                flex(128, (32, 16, 16), vk::ComponentTypeKHR::FLOAT32),
                flex(256, (32, 32, 16), vk::ComponentTypeKHR::FLOAT16),
                flex(256, (32, 32, 16), vk::ComponentTypeKHR::FLOAT32),
            ],
        }
    }

    #[test]
    fn coopmat2_gate_accepts_a_device_meeting_every_requirement() {
        assert_eq!(check_coopmat2_support(&coopmat2_ok()), Ok(()));
        // Finer granularity than required is still a pass (the value is a minimum STEP), and extra
        // rows the gate does not care about are ignored.
        let mut finer = coopmat2_ok();
        for d in &mut finer.flexible_dimensions_list {
            d.m_granularity = 16;
            d.n_granularity = 8;
            d.k_granularity = 8;
        }
        finer
            .flexible_dimensions_list
            .push(flex(64, (8, 8, 8), vk::ComponentTypeKHR::FLOAT64));
        finer.flexible_dimensions_max_dimension = 4096;
        assert_eq!(check_coopmat2_support(&finer), Ok(()));
    }

    #[test]
    fn coopmat2_gate_refuses_the_extension_string_alone() {
        // The case the gate exists for: the extension is advertised, so a string check would say
        // yes, and one required feature is off.
        for (name, clear) in [
            (
                "cooperativeMatrixWorkgroupScope",
                (|p: &mut Coopmat2Probe| p.workgroup_scope = false) as fn(&mut Coopmat2Probe),
            ),
            ("cooperativeMatrixFlexibleDimensions", |p| {
                p.flexible_dimensions = false
            }),
            ("cooperativeMatrixReductions", |p| p.reductions = false),
            ("cooperativeMatrixConversions", |p| p.conversions = false),
            ("cooperativeMatrixPerElementOperations", |p| {
                p.per_element_operations = false
            }),
            ("cooperativeMatrixTensorAddressing", |p| {
                p.tensor_addressing = false
            }),
            ("cooperativeMatrixBlockLoads", |p| p.block_loads = false),
            ("bufferDeviceAddress", |p| p.buffer_device_address = false),
        ] {
            let mut p = coopmat2_ok();
            clear(&mut p);
            let err =
                check_coopmat2_support(&p).expect_err(&format!("{name} cleared must be refused"));
            assert!(err.contains(name), "{err}");
        }
    }

    #[test]
    fn coopmat2_gate_fails_closed_on_anything_unknown() {
        // No extension at all.
        assert!(check_coopmat2_support(&Coopmat2Probe::default()).is_err());
        // The shape list could not be queried: UNKNOWN must not read as "the shapes are fine".
        let mut unqueried = coopmat2_ok();
        unqueried.flexible_dimensions_reported = false;
        let err = check_coopmat2_support(&unqueried).unwrap_err();
        assert!(err.contains("unknown"), "{err}");
        // Reported, but empty — same verdict by a different route.
        let mut empty = coopmat2_ok();
        empty.flexible_dimensions_list.clear();
        assert!(check_coopmat2_support(&empty).is_err());
        // A device that reported no maxDimension (the struct default) is below the floor.
        let mut no_dim = coopmat2_ok();
        no_dim.flexible_dimensions_max_dimension = 0;
        assert!(check_coopmat2_support(&no_dim).is_err());
        let mut small_dim = coopmat2_ok();
        small_dim.flexible_dimensions_max_dimension = COOPMAT2_MIN_MAX_DIMENSION - 1;
        assert!(check_coopmat2_support(&small_dim).is_err());
    }

    #[test]
    fn coopmat2_gate_requires_every_tile_shape() {
        // Drop each of the four required rows in turn.
        for drop in 0..4usize {
            let mut p = coopmat2_ok();
            p.flexible_dimensions_list.remove(drop);
            assert!(
                check_coopmat2_support(&p).is_err(),
                "row {drop} removed must refuse"
            );
        }
        // Present at the right workgroup size but too COARSE a granularity: 256-invocation rows
        // need N <= 32, and a device offering only 64 cannot build the tile.
        let mut coarse = coopmat2_ok();
        for d in &mut coarse.flexible_dimensions_list {
            if d.workgroup_invocations == 256 {
                d.n_granularity = 64;
            }
        }
        assert!(check_coopmat2_support(&coarse).is_err());
        // Right shapes, wrong scope — coopmat2's whole point is workgroup scope.
        let mut subgroup = coopmat2_ok();
        for d in &mut subgroup.flexible_dimensions_list {
            d.scope = vk::ScopeKHR::SUBGROUP;
        }
        assert!(check_coopmat2_support(&subgroup).is_err());
        // Saturating accumulation is a different operation and upstream excludes it explicitly.
        let mut saturating = coopmat2_ok();
        for d in &mut saturating.flexible_dimensions_list {
            d.saturating_accumulation = true;
        }
        assert!(check_coopmat2_support(&saturating).is_err());
        // bf16 inputs do not substitute for the required fp16 ones.
        let mut bf16 = coopmat2_ok();
        for d in &mut bf16.flexible_dimensions_list {
            d.a_type = vk::ComponentTypeKHR::from_raw(1_000_141_000);
            d.b_type = vk::ComponentTypeKHR::from_raw(1_000_141_000);
        }
        assert!(check_coopmat2_support(&bf16).is_err());
    }

    // ── int8 coopmat fragment layout ────────────────────────────────────────────────────────────

    /// Replay what the probe shader does on a device whose TRUE accumulator layout is `truth`:
    /// every lane writes the element it owns into the slot the kernel's ASSUMED mapping names.
    /// `truth(lane, i) -> (row, col)` is the driver's real fragment layout.
    fn simulate_probe(truth: impl Fn(usize, usize) -> (usize, usize)) -> Vec<i32> {
        let mut out = vec![0i32; FRAG_PROBE_WORDS];
        out[0] = FRAG_ELEMS_PER_LANE;
        out[1] = FRAG_SUBGROUP;
        for lane in 0..FRAG_SUBGROUP as usize {
            for i in 0..FRAG_ELEMS_PER_LANE as usize {
                let (true_row, true_col) = truth(lane, i);
                // The value this lane actually holds is the product element at its TRUE position…
                let value = frag_probe_expected(true_row, true_col);
                // …and the shader stores it at the ASSUMED position.
                let (row, col) = (2 * i + (lane >> 4), lane & 15);
                if row < 16 {
                    out[FRAG_PROBE_HEADER + row * 16 + col] = value;
                }
            }
        }
        out
    }

    /// The layout `native_gemm_i8cm_q8_0.comp` assumes, and which RADV/RDNA3 was measured to have.
    fn radv_layout(lane: usize, i: usize) -> (usize, usize) {
        (2 * i + (lane >> 4), lane & 15)
    }

    #[test]
    fn layout_check_accepts_the_layout_the_kernel_assumes() {
        assert_eq!(
            check_i8_coopmat_layout(&simulate_probe(radv_layout)),
            Ok(())
        );
    }

    #[test]
    fn layout_check_rejects_a_different_fragment_layout() {
        // A plausible alternative an implementation is free to have: lanes own a contiguous
        // 8-element run of one row-pair block, i.e. row = i + 8*(lane>>4), col = lane&15.
        let alt = |lane: usize, i: usize| (i + 8 * (lane >> 4), lane & 15);
        let err = check_i8_coopmat_layout(&simulate_probe(alt)).unwrap_err();
        assert!(err.contains("accumulator element"), "{err}");
        // Column-major-ish: the lane indexes the ROW and i the column block.
        let transposed = |lane: usize, i: usize| (lane & 15, 2 * i + (lane >> 4));
        assert!(check_i8_coopmat_layout(&simulate_probe(transposed)).is_err());
        // A device that hands each lane a different element COUNT, or ran at another width.
        let mut short = simulate_probe(radv_layout);
        short[0] = 4;
        assert!(check_i8_coopmat_layout(&short)
            .unwrap_err()
            .contains("elements"));
        let mut wide = simulate_probe(radv_layout);
        wide[1] = 64;
        assert!(check_i8_coopmat_layout(&wide)
            .unwrap_err()
            .contains("subgroup size"));
        // A truncated readback is a failure, not a panic.
        assert!(check_i8_coopmat_layout(&[]).is_err());
        // The all-zero buffer an unwritten (or never-dispatched) probe leaves behind.
        assert!(check_i8_coopmat_layout(&vec![0i32; FRAG_PROBE_WORDS]).is_err());
    }

    #[test]
    fn probe_inputs_multiply_to_the_expected_product() {
        // The host-side known answer, computed from the actual buffers the probe uploads: if this
        // and the shader ever disagree about A/B, every device would "fail" the layout check.
        let (a, b) = frag_probe_inputs();
        for r in 0..16usize {
            for c in 0..16usize {
                let mut sum = 0i32;
                for k in 0..16usize {
                    sum += a[r * 16 + k] as i32 * b[c * 16 + k] as i32;
                }
                assert_eq!(sum, frag_probe_expected(r, c), "C[{r}][{c}]");
            }
        }
        // Distinct and non-zero is what makes a scrambled layout detectable (see the doc).
        let mut seen = std::collections::HashSet::new();
        for r in 0..16usize {
            for c in 0..16usize {
                let v = frag_probe_expected(r, c);
                assert_ne!(v, 0);
                assert!(seen.insert(v), "duplicate product element {v}");
            }
        }
    }

    // ── limits ──────────────────────────────────────────────────────────────────────────────────

    #[test]
    fn push_constant_check_refuses_only_an_oversize_block() {
        // The Vulkan-guaranteed floor, which is what every kernel here was sized against.
        assert_eq!(check_push_constant_size("k", 128, 128), Ok(()));
        assert_eq!(check_push_constant_size("k", 0, 128), Ok(()));
        let err = check_push_constant_size("native_gemv", 132, 128).unwrap_err();
        assert!(
            err.contains("native_gemv") && err.contains("132") && err.contains("128"),
            "{err}"
        );
    }
}
