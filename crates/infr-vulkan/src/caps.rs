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

/// The raw device facts an architecture/trust decision may key on. Every field is a direct copy of
/// a Vulkan query result — no policy, no env, no config — so a test can state a device by writing
/// one of these down.
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
    /// `VkPhysicalDeviceShaderCorePropertiesAMD::wavefrontsPerSimd`, 0 when the AMD extension is
    /// absent (which is itself the signal that the AMD probe cannot classify).
    pub wavefronts_per_simd: u32,
    /// `VK_AMD_shader_core_properties` advertised.
    pub has_amd_shader_core: bool,
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
            wavefronts_per_simd: 16,
            has_amd_shader_core: true,
            has_integer_dot: true,
            dot4x8_signed_accelerated: true,
            dot4x8_mixed_accelerated: true,
            has_coopmat_ext: true,
            has_nv_sm_builtins: false,
            warps_per_sm: 0,
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
            wavefronts_per_simd: 0,
            has_amd_shader_core: false,
            has_integer_dot: true,
            dot4x8_signed_accelerated: true,
            dot4x8_mixed_accelerated: false,
            has_coopmat_ext: true,
            has_nv_sm_builtins: false,
            warps_per_sm: 0,
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
            wavefronts_per_simd: 0,
            has_amd_shader_core: false,
            has_integer_dot: true,
            dot4x8_signed_accelerated: true,
            dot4x8_mixed_accelerated: true,
            has_coopmat_ext: coopmat,
            has_nv_sm_builtins: warps > 0,
            warps_per_sm: warps,
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
