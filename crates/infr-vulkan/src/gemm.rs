//! Cooperative-matrix GEMM (the production matmul primitive). Uses the GLSL coopmat shader
//! compiled by build.rs. f16 inputs, f32 accumulate/output. v1 requires m,n,k multiples of 16.

use std::sync::OnceLock;

use ash::vk;
use half::f16;

use infr_core::{backend::BufferUsage, error::Result, Backend};

use super::{as_vk_buf, VulkanBackend};

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn spv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// Generated `native_gemm_spv(name) -> &'static [u32]`: resident dense-GEMM kernel name to
// its `-DSTREAMED` arena-addressed twin (see build.rs and the Recorder's `matmul_native_*` streamed
// remap). Uses `spv_words`/`OnceLock` from this module scope.
include!(concat!(env!("OUT_DIR"), "/native_gemm_map.rs"));

/// Kernel-cache NAME of the multi-row native GEMV (m = 2..8 — the spec-decode verify /
/// short-suffix-prefill shape) for `dtype`, or `None` for formats without an mrow build (they keep
/// the tiled GEMM route). Availability + name oracle; the SPIR-V (the sole, arena-addressed weight
/// build) is loaded via [`native_mrow_spv`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mrow_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q8_0 => "native_mrow_q8_0",
        Bf16 => "native_mrow_bf16",
        Q4_0 => "native_mrow_q4_0",
        Q4_1 => "native_mrow_q4_1",
        Q5_0 => "native_mrow_q5_0",
        Q5_1 => "native_mrow_q5_1",
        Q2K => "native_mrow_q2k",
        Q3K => "native_mrow_q3k",
        Q4K => "native_mrow_q4k",
        Q5K => "native_mrow_q5k",
        Q6K => "native_mrow_q6k",
        Iq4Nl => "native_mrow_iq4nl",
        Iq4Xs => "native_mrow_iq4xs",
        Q2_0 => "native_mrow_q2_0",
        _ => return None,
    })
}

/// The dense layer-streaming decode GEMV (kernel-cache name + SPIR-V): reads the weight from a
/// `bufferDeviceAddress` arena pool by 64-bit address (native_weight_addr.glsl) instead of a bound
/// SSBO — the ONLY weight build for this family (the bound-SSBO resident build died with the eager
/// `VulkanBackend::linear_native` / `Recorder::linear_native_{sg,rm,rm_v2}` callers that were its
/// sole consumers). All streamed dense Linears (any m) route through this one kernel's rows loop —
/// see [`crate::recorder::Recorder::linear_native_at`] and [`crate::pager::DensePagerSession`].
/// `res`: production dispatches fused-residual GEMVs too (the decode Linear+Add fusion,
/// [`crate::recorder::Recorder::linear_add_native`]), so both arms are load-bearing. `None` for a
/// dtype without a native GEMV build.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_streamed_build_spv(
    dtype: infr_core::DType,
    res: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res) {
        (Q8_0, false) => v!("native_q8_0"),
        (Q8_0, true) => v!("native_q8_0_res"),
        (Bf16, false) => v!("native_bf16"),
        (Bf16, true) => v!("native_bf16_res"),
        (Q4_0, false) => v!("native_q4_0"),
        (Q4_0, true) => v!("native_q4_0_res"),
        (Q4_1, false) => v!("native_q4_1"),
        (Q4_1, true) => v!("native_q4_1_res"),
        (Q5_0, false) => v!("native_q5_0"),
        (Q5_0, true) => v!("native_q5_0_res"),
        (Q5_1, false) => v!("native_q5_1"),
        (Q5_1, true) => v!("native_q5_1_res"),
        (Q2K, false) => v!("native_q2k"),
        (Q2K, true) => v!("native_q2k_res"),
        (Q3K, false) => v!("native_q3k"),
        (Q3K, true) => v!("native_q3k_res"),
        (Q4K, false) => v!("native_q4k"),
        (Q4K, true) => v!("native_q4k_res"),
        (Q5K, false) => v!("native_q5k"),
        (Q5K, true) => v!("native_q5k_res"),
        (Q6K, false) => v!("native_q6k"),
        (Q6K, true) => v!("native_q6k_res"),
        (Iq4Nl, false) => v!("native_iq4nl"),
        (Iq4Nl, true) => v!("native_iq4nl_res"),
        (Iq4Xs, false) => v!("native_iq4xs"),
        (Iq4Xs, true) => v!("native_iq4xs_res"),
        (Mxfp4, false) => v!("native_mxfp4"),
        (Mxfp4, true) => v!("native_mxfp4_res"),
        (Nvfp4, false) => v!("native_nvfp4"),
        (Nvfp4, true) => v!("native_nvfp4_res"),
        (Tq1_0, false) => v!("native_tq1_0"),
        (Tq1_0, true) => v!("native_tq1_0_res"),
        (Tq2_0, false) => v!("native_tq2_0"),
        (Tq2_0, true) => v!("native_tq2_0_res"),
        (Q2_0, false) => v!("native_q2_0"),
        (Q2_0, true) => v!("native_q2_0_res"),
        (Iq2Xxs, false) => v!("native_iq2xxs"),
        (Iq2Xxs, true) => v!("native_iq2xxs_res"),
        (Iq2Xs, false) => v!("native_iq2xs"),
        (Iq2Xs, true) => v!("native_iq2xs_res"),
        (Iq2S, false) => v!("native_iq2s"),
        (Iq2S, true) => v!("native_iq2s_res"),
        (Iq3Xxs, false) => v!("native_iq3xxs"),
        (Iq3Xxs, true) => v!("native_iq3xxs_res"),
        (Iq3S, false) => v!("native_iq3s"),
        (Iq3S, true) => v!("native_iq3s_res"),
        (Iq1S, false) => v!("native_iq1s"),
        (Iq1S, true) => v!("native_iq1s_res"),
        (Iq1M, false) => v!("native_iq1m"),
        (Iq1M, true) => v!("native_iq1m_res"),
        _ => None,
    }
}

/// SPIR-V (kernel-cache name + words) for [`native_mrow_kernel_name`]'s pick — the multi-row GEMV's
/// weight read from a `bufferDeviceAddress` arena; the sole weight build. Production dispatches it
/// via [`crate::recorder::Recorder::linear_native_mrow`]; parity tests call the streamed entry
/// directly with an explicit arena address (`tests/weight_addr_parity.rs`). `None` for a dtype
/// without an mrow build.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mrow_spv(dtype: infr_core::DType) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match dtype {
        Q8_0 => v!("native_mrow_q8_0"),
        Bf16 => v!("native_mrow_bf16"),
        Q4_0 => v!("native_mrow_q4_0"),
        Q4_1 => v!("native_mrow_q4_1"),
        Q5_0 => v!("native_mrow_q5_0"),
        Q5_1 => v!("native_mrow_q5_1"),
        Q2K => v!("native_mrow_q2k"),
        Q3K => v!("native_mrow_q3k"),
        Q4K => v!("native_mrow_q4k"),
        Q5K => v!("native_mrow_q5k"),
        Q6K => v!("native_mrow_q6k"),
        Iq4Nl => v!("native_mrow_iq4nl"),
        Iq4Xs => v!("native_mrow_iq4xs"),
        Q2_0 => v!("native_mrow_q2_0"),
        _ => None,
    }
}

/// The multi-output-row decode GEMV (`RM` rows/workgroup, bit-identical per row to the RM=1 native
/// GEMV) — kernel-cache name + SPIR-V, the ONLY weight build for this family (the bound-SSBO
/// resident build died with the eager `Recorder::linear_native_rm` caller, its sole consumer). `rm`
/// is 2 or 4; `res`: `linear_add_native`'s fallback route (RM=1..8 out_f band, reached when the SG
/// route and the default "reg" variant route both miss — e.g. `INFR_NO_GEMV_REG`) dispatches this
/// table with `res = true`, so the residual arm is load-bearing for
/// [`crate::recorder::Recorder::linear_add_native_at`], not just a parity convenience. Only
/// the K-quant formats that dominate decode (Q4_K/Q6_K) have RM builds; everything else stays on
/// the RM=1 path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_rm_streamed_build_spv(
    dtype: infr_core::DType,
    res: bool,
    rm: u32,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res, rm) {
        (Q4K, false, 2) => v!("native_q4k_rm2"),
        (Q4K, true, 2) => v!("native_q4k_rm2_res"),
        (Q4K, false, 4) => v!("native_q4k_rm4"),
        (Q4K, true, 4) => v!("native_q4k_rm4_res"),
        (Q6K, false, 2) => v!("native_q6k_rm2"),
        (Q6K, true, 2) => v!("native_q6k_rm2_res"),
        (Q6K, false, 4) => v!("native_q6k_rm4"),
        (Q6K, true, 4) => v!("native_q6k_rm4_res"),
        _ => None,
    }
}

/// SPIR-V + kernel name for an experimental RM kernel variant (env-gated, default OFF) — the ONLY
/// weight build for this family (the bound-SSBO resident build died with the eager
/// `Recorder::linear_native_rm_v2` caller, its sole consumer). Returns (kernel_name, spv) for the
/// given variant + dtype + res combination. `res`: `linear_add_native`'s default route (the "reg"
/// variant, `INFR_GEMV_VARIANT`/`INFR_NO_GEMV_REG` escapes aside) dispatches THIS table with
/// `res = true`, so the residual arm is load-bearing for
/// [`crate::recorder::Recorder::linear_add_native_at`], not just a parity convenience.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_rm_v2_streamed_build_spv(
    variant: &str,
    dtype: infr_core::DType,
    res: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (variant, dtype, res) {
        ("sg", Q4K, false) => v!("native_q4k_rm2_sg"),
        ("sg", Q4K, true) => v!("native_q4k_rm2_sg_res"),
        ("sg", Q6K, false) => v!("native_q6k_rm2_sg"),
        ("sg", Q6K, true) => v!("native_q6k_rm2_sg_res"),
        ("dbuf", Q4K, false) => v!("native_q4k_rm2_dbuf"),
        ("dbuf", Q4K, true) => v!("native_q4k_rm2_dbuf_res"),
        ("wg128", Q4K, false) => v!("native_q4k_rm2_wg128"),
        ("wg128", Q4K, true) => v!("native_q4k_rm2_wg128_res"),
        ("reg", Q4K, false) => v!("native_q4k_rm2_reg"),
        ("reg", Q4K, true) => v!("native_q4k_rm2_reg_res"),
        _ => None,
    }
}

/// The reassociation-tolerant subgroup+NUM_ROWS decode GEMV (`native_gemv_sg.comp`, one pinned
/// subgroup per workgroup + subgroupAdd) — kernel-cache name + SPIR-V, the ONLY weight build for
/// this family (the bound-SSBO resident build died with the eager `Recorder::linear_native_sg`
/// caller, its sole consumer). NOT bit-identical to the tree GEMV — reordered accumulation; the
/// caller must gate to the projection band and re-bless any changed golden. Only Q6_K has an SG
/// build (on Q4_K the tree/RM kernel already saturates — SG regressed). `nr` ∈ {2,4,8}; `res`:
/// `linear_add_native`'s SG route ([`crate::recorder::Recorder::linear_add_native`]) dispatches this
/// with `res = true` before ever falling to RM, so the residual arm is load-bearing for
/// [`crate::recorder::Recorder::linear_add_native_at`], not just a parity convenience. `sg16`
/// selects the `-DSG=16` twin (Intel, `caps.sg_pref == 16` — pass false everywhere else for the
/// byte-identical SG=32 build).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_sg_streamed_build_spv(
    dtype: infr_core::DType,
    res: bool,
    nr: u32,
    sg16: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res, nr, sg16) {
        (Q6K, false, 2, false) => v!("native_q6k_sg2"),
        (Q6K, true, 2, false) => v!("native_q6k_sg2_res"),
        (Q6K, false, 4, false) => v!("native_q6k_sg4"),
        (Q6K, true, 4, false) => v!("native_q6k_sg4_res"),
        (Q6K, false, 8, false) => v!("native_q6k_sg8"),
        (Q6K, true, 8, false) => v!("native_q6k_sg8_res"),
        (Q6K, false, 2, true) => v!("native_q6k_sg2_sg16"),
        (Q6K, true, 2, true) => v!("native_q6k_sg2_res_sg16"),
        (Q6K, false, 4, true) => v!("native_q6k_sg4_sg16"),
        (Q6K, true, 4, true) => v!("native_q6k_sg4_res_sg16"),
        (Q6K, false, 8, true) => v!("native_q6k_sg8_sg16"),
        (Q6K, true, 8, true) => v!("native_q6k_sg8_res_sg16"),
        _ => None,
    }
}

/// Paged variant of the id-indexed native GEMV (`infr_vulkan::pager`) — one extra LUT-buffer
/// binding. (The old bound-SSBO resident id GEMV is gone; STREAMED / this paged build are the
/// only weight builds.)
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_id_paged_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| {
                spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
            .as_slice()
        }};
    }
    Some(match dtype {
        Q8_0 => v!("native_id_q8_0_paged"),
        Q4_0 => v!("native_id_q4_0_paged"),
        Q4_1 => v!("native_id_q4_1_paged"),
        Q5_0 => v!("native_id_q5_0_paged"),
        Q5_1 => v!("native_id_q5_1_paged"),
        Q2K => v!("native_id_q2k_paged"),
        Q3K => v!("native_id_q3k_paged"),
        Q4K => v!("native_id_q4k_paged"),
        Q5K => v!("native_id_q5k_paged"),
        Q6K => v!("native_id_q6k_paged"),
        Iq4Nl => v!("native_id_iq4nl_paged"),
        Iq4Xs => v!("native_id_iq4xs_paged"),
        Mxfp4 => v!("native_id_mxfp4_paged"),
        Nvfp4 => v!("native_id_nvfp4_paged"),
        Tq1_0 => v!("native_id_tq1_0_paged"),
        Tq2_0 => v!("native_id_tq2_0_paged"),
        Q2_0 => v!("native_id_q2_0_paged"),
        Iq2Xxs => v!("native_id_iq2xxs_paged"),
        Iq2Xs => v!("native_id_iq2xs_paged"),
        Iq2S => v!("native_id_iq2s_paged"),
        Iq3Xxs => v!("native_id_iq3xxs_paged"),
        Iq3S => v!("native_id_iq3s_paged"),
        Iq1S => v!("native_id_iq1s_paged"),
        Iq1M => v!("native_id_iq1m_paged"),
        Bf16 => v!("native_id_bf16_paged"),
        F16 => v!("native_id_f16_paged"),
        F32 => v!("native_id_f32_paged"),
        _ => return None,
    })
}
/// Paged variant of the multi-slot id-indexed native GEMV (`infr_vulkan::pager`) — one extra
/// LUT-buffer binding. (The old bound-SSBO resident build is gone; STREAMED / this paged build are
/// the only weight builds.)
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_idm_paged_build_spv(dtype: infr_core::DType) -> Option<&'static [u32]> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| {
                spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
            })
            .as_slice()
        }};
    }
    Some(match dtype {
        Q8_0 => v!("native_idm_q8_0_paged"),
        Q4_0 => v!("native_idm_q4_0_paged"),
        Q4_1 => v!("native_idm_q4_1_paged"),
        Q5_0 => v!("native_idm_q5_0_paged"),
        Q5_1 => v!("native_idm_q5_1_paged"),
        Q2K => v!("native_idm_q2k_paged"),
        Q3K => v!("native_idm_q3k_paged"),
        Q4K => v!("native_idm_q4k_paged"),
        Q5K => v!("native_idm_q5k_paged"),
        Q6K => v!("native_idm_q6k_paged"),
        Iq4Nl => v!("native_idm_iq4nl_paged"),
        Iq4Xs => v!("native_idm_iq4xs_paged"),
        Mxfp4 => v!("native_idm_mxfp4_paged"),
        Nvfp4 => v!("native_idm_nvfp4_paged"),
        Tq1_0 => v!("native_idm_tq1_0_paged"),
        Tq2_0 => v!("native_idm_tq2_0_paged"),
        Q2_0 => v!("native_idm_q2_0_paged"),
        Iq2Xxs => v!("native_idm_iq2xxs_paged"),
        Iq2Xs => v!("native_idm_iq2xs_paged"),
        Iq2S => v!("native_idm_iq2s_paged"),
        Iq3Xxs => v!("native_idm_iq3xxs_paged"),
        Iq3S => v!("native_idm_iq3s_paged"),
        Iq1S => v!("native_idm_iq1s_paged"),
        Iq1M => v!("native_idm_iq1m_paged"),
        Bf16 => v!("native_idm_bf16_paged"),
        F16 => v!("native_idm_f16_paged"),
        F32 => v!("native_idm_f32_paged"),
        _ => return None,
    })
}

/// The single-slot id-indexed native GEMV (kernel-cache name + SPIR-V) — the stacked expert
/// tensor read from a `bufferDeviceAddress` arena; `stride` becomes the per-expert BYTE stride
/// applied on the 64-bit pointer (no LUT); the sole weight build. Dispatched via
/// [`crate::recorder::Recorder::linear_native_id`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_id_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match dtype {
        Q8_0 => v!("native_id_q8_0"),
        Q4_0 => v!("native_id_q4_0"),
        Q4_1 => v!("native_id_q4_1"),
        Q5_0 => v!("native_id_q5_0"),
        Q5_1 => v!("native_id_q5_1"),
        Q2K => v!("native_id_q2k"),
        Q3K => v!("native_id_q3k"),
        Q4K => v!("native_id_q4k"),
        Q5K => v!("native_id_q5k"),
        Q6K => v!("native_id_q6k"),
        Iq4Nl => v!("native_id_iq4nl"),
        Iq4Xs => v!("native_id_iq4xs"),
        Mxfp4 => v!("native_id_mxfp4"),
        Nvfp4 => v!("native_id_nvfp4"),
        Tq1_0 => v!("native_id_tq1_0"),
        Tq2_0 => v!("native_id_tq2_0"),
        Q2_0 => v!("native_id_q2_0"),
        Iq2Xxs => v!("native_id_iq2xxs"),
        Iq2Xs => v!("native_id_iq2xs"),
        Iq2S => v!("native_id_iq2s"),
        Iq3Xxs => v!("native_id_iq3xxs"),
        Iq3S => v!("native_id_iq3s"),
        Iq1S => v!("native_id_iq1s"),
        Iq1M => v!("native_id_iq1m"),
        Bf16 => v!("native_id_bf16"),
        F16 => v!("native_id_f16"),
        F32 => v!("native_id_f32"),
        _ => None,
    }
}
/// The multi-slot id-indexed native GEMV (kernel-cache name + SPIR-V) — the stacked expert
/// tensor read from a `bufferDeviceAddress` arena; `stride` becomes the per-expert BYTE stride
/// applied on the 64-bit pointer (no LUT); the sole weight build. Production dispatches it via
/// [`crate::recorder::Recorder::linear_native_id_multi`] (resident MoE decode).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_idm_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match dtype {
        Q8_0 => v!("native_idm_q8_0"),
        Q4_0 => v!("native_idm_q4_0"),
        Q4_1 => v!("native_idm_q4_1"),
        Q5_0 => v!("native_idm_q5_0"),
        Q5_1 => v!("native_idm_q5_1"),
        Q2K => v!("native_idm_q2k"),
        Q3K => v!("native_idm_q3k"),
        Q4K => v!("native_idm_q4k"),
        Q5K => v!("native_idm_q5k"),
        Q6K => v!("native_idm_q6k"),
        Iq4Nl => v!("native_idm_iq4nl"),
        Iq4Xs => v!("native_idm_iq4xs"),
        Mxfp4 => v!("native_idm_mxfp4"),
        Nvfp4 => v!("native_idm_nvfp4"),
        Tq1_0 => v!("native_idm_tq1_0"),
        Tq2_0 => v!("native_idm_tq2_0"),
        Q2_0 => v!("native_idm_q2_0"),
        Iq2Xxs => v!("native_idm_iq2xxs"),
        Iq2Xs => v!("native_idm_iq2xs"),
        Iq2S => v!("native_idm_iq2s"),
        Iq3Xxs => v!("native_idm_iq3xxs"),
        Iq3S => v!("native_idm_iq3s"),
        Iq1S => v!("native_idm_iq1s"),
        Iq1M => v!("native_idm_iq1m"),
        Bf16 => v!("native_idm_bf16"),
        F16 => v!("native_idm_f16"),
        F32 => v!("native_idm_f32"),
        _ => None,
    }
}
/// The subgroup multi-slot id GEMV (kernel-cache name + SPIR-V); the sole weight build for this
/// family. Production dispatches it via [`crate::recorder::Recorder::linear_native_id_multi`]'s
/// SG routing (m=1 Q5K/Q6K/IQ3S projection band).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_idm_sg_build_spv(
    dtype: infr_core::DType,
    nr: u32,
    sg16: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, nr, sg16) {
        (Q6K, 2, false) => v!("native_idm_q6k_sg2"),
        (Q6K, 4, false) => v!("native_idm_q6k_sg4"),
        (Q6K, 8, false) => v!("native_idm_q6k_sg8"),
        (Q6K, 2, true) => v!("native_idm_q6k_sg2_sg16"),
        (Q6K, 4, true) => v!("native_idm_q6k_sg4_sg16"),
        (Q6K, 8, true) => v!("native_idm_q6k_sg8_sg16"),
        (Q5K, 2, false) => v!("native_idm_q5k_sg2"),
        (Q5K, 4, false) => v!("native_idm_q5k_sg4"),
        (Q5K, 8, false) => v!("native_idm_q5k_sg8"),
        (Q5K, 2, true) => v!("native_idm_q5k_sg2_sg16"),
        (Q5K, 4, true) => v!("native_idm_q5k_sg4_sg16"),
        (Q5K, 8, true) => v!("native_idm_q5k_sg8_sg16"),
        (Iq3S, 2, false) => v!("native_idm_iq3s_sg2"),
        (Iq3S, 4, false) => v!("native_idm_iq3s_sg4"),
        (Iq3S, 8, false) => v!("native_idm_iq3s_sg8"),
        (Iq3S, 2, true) => v!("native_idm_iq3s_sg2_sg16"),
        (Iq3S, 4, true) => v!("native_idm_iq3s_sg4_sg16"),
        (Iq3S, 8, true) => v!("native_idm_iq3s_sg8_sg16"),
        _ => None,
    }
}
/// The id-indexed int8 Q4_K decode GEMV; the sole weight build for this variant. Env-gated
/// measurement entry ([`crate::recorder::Recorder::linear_mmv_id_multi_q4k`]) + parity tests.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_id_q4k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_mmv_id_q4k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// Kernel-cache NAME + availability for the int8 dp4a decode GEMV (m=1, NUM_ROWS=2,
/// `native_mmv.comp`); the SPIR-V lives in [`native_mmv_spv`]. `None` = format
/// has no int-dot build (falls back to the dequant `native_gemv`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_kernel_name(dtype: infr_core::DType, res: bool) -> Option<&'static str> {
    // Delegate to the SPIR-V source of truth so the name gate and the loaded shader can never
    // drift apart (a name-table-only dtype was a mid-inference `expect()` panic; see AUDIT #1).
    native_mmv_spv(dtype, res).map(|(name, _)| name)
}
/// SPIR-V + cache name for the multi-warp int8 dp4a decode GEMV (`native_mmv_mw.comp`, warp-per-row
/// subgroupAdd, `warps` rows/block). Gated by the adapter's `mmv_mw_choice` (default-on Intel,
/// opt-in elsewhere). `warps` ∈ {4, 8}. `sg16` selects the `-DSG=16` twin (`caps.sg_pref == 16`).
/// `None` for formats/warp counts without a build.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mw_kernel_name(
    dtype: infr_core::DType,
    res: bool,
    warps: u32,
    sg16: bool,
) -> Option<&'static str> {
    // Delegate to the SPIR-V source of truth so gate and shader can never drift (AUDIT #1).
    native_mmv_mw_spv(dtype, res, warps, sg16).map(|(name, _)| name)
}
/// Kernel-cache NAME + availability for the multi-row int8 dp4a GEMV (m = 2..8,
/// `native_mmv_mrow.comp`); the SPIR-V lives in [`native_mmv_mrow_variant_spv`]. `None` = format
/// has no int-dot build (falls back to the dequant `native_gemv_mrow`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    // The plain (o4=false, m4=false, res=false) build IS this table's entry; delegate to the
    // SPIR-V source of truth so the availability gate can never drift from the shader (AUDIT #1).
    native_mmv_mrow_variant_spv(dtype, false, false, false).map(|(name, _)| name)
}
/// Whether the multi-row int8 dp4a GEMV family has a fused-residual (`-DUSE_RES`) build for `dtype`.
/// The residual arm is only ever declared at `m4 = true` (decode is always `rows == 1`), so this
/// probes exactly that build. Iq4Xs is the sole mrow-eligible dtype WITHOUT a `_res` twin, which is
/// why `linear_mmv_mrow_at` and the adapter must consult this predicate before dispatching a
/// residual mrow decode (AUDIT #2) rather than assuming a plain build implies a res build.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_res_supported(dtype: infr_core::DType) -> bool {
    native_mmv_mrow_variant_spv(dtype, false, true, true).is_some()
}
/// SPIR-V for a multi-row int8 dp4a GEMV layout variant: `o4` = the small-in_f 4-outputs ×
/// 16-K-lanes workgroup split (-DOUTS4), `m4` = the rows<=4 MR specialization (-DMRV=4), `res` =
/// the fused-residual decode build (-DUSE_RES, only ever paired with rows=1, hence requires
/// `m4=true`; see `Recorder::linear_mmv_mrow`'s gates and its doc on the rows=1 decode/verify
/// unification).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_variant_kernel_name(
    dtype: infr_core::DType,
    o4: bool,
    m4: bool,
    res: bool,
) -> Option<&'static str> {
    // Delegate to the SPIR-V source of truth so gate and shader can never drift (AUDIT #1).
    native_mmv_mrow_variant_spv(dtype, o4, m4, res).map(|(name, _)| name)
}
/// Kernel-cache NAME for the rows 9..=16 multi-row int8 dp4a GEMV tier (`-DMRV=16`, 2-output layout
/// — no OUTS4 twin: the tier is gated to >= 8M-element weights, whose in_f is comfortably >= 2048);
/// the SPIR-V lives in [`native_mmv_mrow_m16_spv`]. `None` = format not covered (same coverage as
/// [`native_mmv_mrow_kernel_name`]; those shapes stay on the split-K GEMM tile).
pub(crate) fn native_mmv_mrow_m16_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    // Delegate to the SPIR-V source of truth so gate and shader can never drift (AUDIT #1).
    native_mmv_mrow_m16_spv(dtype).map(|(name, _)| name)
}
/// `-DSTREAMED` twin of [`native_mmv_kernel_name`] (kernel-cache name + SPIR-V) — the int8 dp4a
/// decode GEMV's weight read from a `bufferDeviceAddress` arena instead of a bound SSBO. `res`
/// mirrors [`native_mmv_kernel_name`]'s residual arm: `linear_add_mmv`
/// ([`crate::recorder::Recorder::linear_add_mmv`]) dispatches this table with `res = true` for its
/// fused-residual int8 decode GEMV, so the residual twin
/// ([`crate::recorder::Recorder::linear_add_mmv_at`]) is load-bearing, not just a parity-test
/// convenience like the plain arm ([`crate::recorder::Recorder::linear_mmv_at`]).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_spv(
    dtype: infr_core::DType,
    res: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res) {
        (Q4K, false) => v!("native_mmv_q4k"),
        (Q4K, true) => v!("native_mmv_q4k_res"),
        (Q6K, false) => v!("native_mmv_q6k"),
        (Q6K, true) => v!("native_mmv_q6k_res"),
        (Iq4Xs, false) => v!("native_mmv_iq4xs"),
        (Iq4Xs, true) => v!("native_mmv_iq4xs_res"),
        _ => None,
    }
}
/// `-DSTREAMED` twin of [`native_mmv_mrow_variant_kernel_name`] (kernel-cache name + SPIR-V), covering the
/// rows<=8 layout matrix (`o4`/`m4`/`res`). `res` mirrors [`native_mmv_mrow_variant_kernel_name`]'s residual
/// arm — build-declared ONLY at `m4 = true` (the decode Linear+Add fusion always dispatches this
/// shader at `rows = 1`, which forces `m4 = true`; see [`crate::recorder::Recorder::linear_mmv_mrow`]'s
/// doc) and NEVER for `Iq4Xs` (no `-DUSE_RES` build exists for that dtype in this family). `res =
/// true` is load-bearing for [`crate::recorder::Recorder::linear_mmv_mrow_at`]'s residual
/// path, not just a parity-test convenience.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_variant_spv(
    dtype: infr_core::DType,
    o4: bool,
    m4: bool,
    res: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, o4, m4, res) {
        (Q4K, false, false, false) => v!("native_mmv_mrow_q4k"),
        (Q4K, false, true, false) => v!("native_mmv_mrow_q4k_m4"),
        (Q4K, true, false, false) => v!("native_mmv_mrow_q4k_o4"),
        (Q4K, true, true, false) => v!("native_mmv_mrow_q4k_o4_m4"),
        (Q4K, false, true, true) => v!("native_mmv_mrow_q4k_m4_res"),
        (Q4K, true, true, true) => v!("native_mmv_mrow_q4k_o4_m4_res"),
        (Q6K, false, false, false) => v!("native_mmv_mrow_q6k"),
        (Q6K, false, true, false) => v!("native_mmv_mrow_q6k_m4"),
        (Q6K, true, false, false) => v!("native_mmv_mrow_q6k_o4"),
        (Q6K, true, true, false) => v!("native_mmv_mrow_q6k_o4_m4"),
        (Q6K, false, true, true) => v!("native_mmv_mrow_q6k_m4_res"),
        (Q6K, true, true, true) => v!("native_mmv_mrow_q6k_o4_m4_res"),
        (Iq4Xs, false, false, false) => v!("native_mmv_mrow_iq4xs"),
        (Iq4Xs, false, true, false) => v!("native_mmv_mrow_iq4xs_m4"),
        (Iq4Xs, true, false, false) => v!("native_mmv_mrow_iq4xs_o4"),
        (Iq4Xs, true, true, false) => v!("native_mmv_mrow_iq4xs_o4_m4"),
        (Q2K, false, false, false) => v!("native_mmv_mrow_q2k"),
        (Q2K, false, true, false) => v!("native_mmv_mrow_q2k_m4"),
        (Q2K, true, false, false) => v!("native_mmv_mrow_q2k_o4"),
        (Q2K, true, true, false) => v!("native_mmv_mrow_q2k_o4_m4"),
        (Q2K, false, true, true) => v!("native_mmv_mrow_q2k_m4_res"),
        (Q2K, true, true, true) => v!("native_mmv_mrow_q2k_o4_m4_res"),
        (Q3K, false, false, false) => v!("native_mmv_mrow_q3k"),
        (Q3K, false, true, false) => v!("native_mmv_mrow_q3k_m4"),
        (Q3K, true, false, false) => v!("native_mmv_mrow_q3k_o4"),
        (Q3K, true, true, false) => v!("native_mmv_mrow_q3k_o4_m4"),
        (Q3K, false, true, true) => v!("native_mmv_mrow_q3k_m4_res"),
        (Q3K, true, true, true) => v!("native_mmv_mrow_q3k_o4_m4_res"),
        (Q5K, false, false, false) => v!("native_mmv_mrow_q5k"),
        (Q5K, false, true, false) => v!("native_mmv_mrow_q5k_m4"),
        (Q5K, true, false, false) => v!("native_mmv_mrow_q5k_o4"),
        (Q5K, true, true, false) => v!("native_mmv_mrow_q5k_o4_m4"),
        (Q5K, false, true, true) => v!("native_mmv_mrow_q5k_m4_res"),
        (Q5K, true, true, true) => v!("native_mmv_mrow_q5k_o4_m4_res"),
        (Q8_0, false, false, false) => v!("native_mmv_mrow_q8_0"),
        (Q8_0, false, true, false) => v!("native_mmv_mrow_q8_0_m4"),
        (Q8_0, true, false, false) => v!("native_mmv_mrow_q8_0_o4"),
        (Q8_0, true, true, false) => v!("native_mmv_mrow_q8_0_o4_m4"),
        (Q8_0, false, true, true) => v!("native_mmv_mrow_q8_0_m4_res"),
        (Q8_0, true, true, true) => v!("native_mmv_mrow_q8_0_o4_m4_res"),
        (Q4_0, false, false, false) => v!("native_mmv_mrow_q4_0"),
        (Q4_0, false, true, false) => v!("native_mmv_mrow_q4_0_m4"),
        (Q4_0, true, false, false) => v!("native_mmv_mrow_q4_0_o4"),
        (Q4_0, true, true, false) => v!("native_mmv_mrow_q4_0_o4_m4"),
        (Q4_0, false, true, true) => v!("native_mmv_mrow_q4_0_m4_res"),
        (Q4_0, true, true, true) => v!("native_mmv_mrow_q4_0_o4_m4_res"),
        (Q5_0, false, false, false) => v!("native_mmv_mrow_q5_0"),
        (Q5_0, false, true, false) => v!("native_mmv_mrow_q5_0_m4"),
        (Q5_0, true, false, false) => v!("native_mmv_mrow_q5_0_o4"),
        (Q5_0, true, true, false) => v!("native_mmv_mrow_q5_0_o4_m4"),
        (Q5_0, false, true, true) => v!("native_mmv_mrow_q5_0_m4_res"),
        (Q5_0, true, true, true) => v!("native_mmv_mrow_q5_0_o4_m4_res"),
        (Q4_1, false, false, false) => v!("native_mmv_mrow_q4_1"),
        (Q4_1, false, true, false) => v!("native_mmv_mrow_q4_1_m4"),
        (Q4_1, true, false, false) => v!("native_mmv_mrow_q4_1_o4"),
        (Q4_1, true, true, false) => v!("native_mmv_mrow_q4_1_o4_m4"),
        (Q4_1, false, true, true) => v!("native_mmv_mrow_q4_1_m4_res"),
        (Q4_1, true, true, true) => v!("native_mmv_mrow_q4_1_o4_m4_res"),
        (Q5_1, false, false, false) => v!("native_mmv_mrow_q5_1"),
        (Q5_1, false, true, false) => v!("native_mmv_mrow_q5_1_m4"),
        (Q5_1, true, false, false) => v!("native_mmv_mrow_q5_1_o4"),
        (Q5_1, true, true, false) => v!("native_mmv_mrow_q5_1_o4_m4"),
        (Q5_1, false, true, true) => v!("native_mmv_mrow_q5_1_m4_res"),
        (Q5_1, true, true, true) => v!("native_mmv_mrow_q5_1_o4_m4_res"),
        (Iq4Nl, false, false, false) => v!("native_mmv_mrow_iq4nl"),
        (Iq4Nl, false, true, false) => v!("native_mmv_mrow_iq4nl_m4"),
        (Iq4Nl, true, false, false) => v!("native_mmv_mrow_iq4nl_o4"),
        (Iq4Nl, true, true, false) => v!("native_mmv_mrow_iq4nl_o4_m4"),
        (Iq4Nl, false, true, true) => v!("native_mmv_mrow_iq4nl_m4_res"),
        (Iq4Nl, true, true, true) => v!("native_mmv_mrow_iq4nl_o4_m4_res"),
        _ => None,
    }
}

/// FUSE_QUANT twin of [`native_mmv_mrow_variant_kernel_name`] — inline activation quantization,
/// opt-in behind INFR_MMV_FUSE_QUANT=1. Q4_K/Q6_K only, m4=true + res combos only (decode).
pub(crate) fn native_mmv_mrow_variant_fuseq_kernel_name(
    dtype: infr_core::DType,
    o4: bool,
    _m4: bool,
    res: bool,
) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match (dtype, o4, res) {
        (Q4K, false, false) => "native_mmv_mrow_q4k_m4_fuseq",
        (Q4K, true, false) => "native_mmv_mrow_q4k_o4_m4_fuseq",
        (Q4K, false, true) => "native_mmv_mrow_q4k_m4_res_fuseq",
        (Q4K, true, true) => "native_mmv_mrow_q4k_o4_m4_res_fuseq",
        (Q6K, false, false) => "native_mmv_mrow_q6k_m4_fuseq",
        (Q6K, true, false) => "native_mmv_mrow_q6k_o4_m4_fuseq",
        (Q6K, false, true) => "native_mmv_mrow_q6k_m4_res_fuseq",
        (Q6K, true, true) => "native_mmv_mrow_q6k_o4_m4_res_fuseq",
        _ => return None,
    })
}

/// FUSE_QUANT twin of [`native_mmv_mrow_variant_spv`] — kernel-cache name + SPIR-V for the
/// inline-activation-quantization variants. Q4_K/Q6_K only, m4+res combos only (decode).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_variant_fuseq_spv(
    dtype: infr_core::DType,
    o4: bool,
    _m4: bool,
    res: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, o4, res) {
        (Q4K, false, false) => v!("native_mmv_mrow_q4k_m4_fuseq"),
        (Q4K, true, false) => v!("native_mmv_mrow_q4k_o4_m4_fuseq"),
        (Q4K, false, true) => v!("native_mmv_mrow_q4k_m4_res_fuseq"),
        (Q4K, true, true) => v!("native_mmv_mrow_q4k_o4_m4_res_fuseq"),
        (Q6K, false, false) => v!("native_mmv_mrow_q6k_m4_fuseq"),
        (Q6K, true, false) => v!("native_mmv_mrow_q6k_o4_m4_fuseq"),
        (Q6K, false, true) => v!("native_mmv_mrow_q6k_m4_res_fuseq"),
        (Q6K, true, true) => v!("native_mmv_mrow_q6k_o4_m4_res_fuseq"),
        _ => None,
    }
}
/// SPIR-V (kernel-cache name + words) for [`native_mmv_mrow_m16_kernel_name`]'s pick — the rows 9..=16
/// tier; the sole weight build. Production dispatches it via
/// [`crate::recorder::Recorder::linear_mmv_mrow`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mrow_m16_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match dtype {
        Q4K => v!("native_mmv_mrow_q4k_m16"),
        Q6K => v!("native_mmv_mrow_q6k_m16"),
        Iq4Xs => v!("native_mmv_mrow_iq4xs_m16"),
        Q2K => v!("native_mmv_mrow_q2k_m16"),
        Q3K => v!("native_mmv_mrow_q3k_m16"),
        Q5K => v!("native_mmv_mrow_q5k_m16"),
        Q8_0 => v!("native_mmv_mrow_q8_0_m16"),
        Q4_0 => v!("native_mmv_mrow_q4_0_m16"),
        Q5_0 => v!("native_mmv_mrow_q5_0_m16"),
        Q4_1 => v!("native_mmv_mrow_q4_1_m16"),
        Q5_1 => v!("native_mmv_mrow_q5_1_m16"),
        Iq4Nl => v!("native_mmv_mrow_iq4nl_m16"),
        _ => None,
    }
}
/// `-DSTREAMED` twin of [`native_mmv_mw_kernel_name`] (kernel-cache name + SPIR-V) — the multi-warp
/// int8 dp4a decode GEMV. `res` mirrors [`native_mmv_mw_kernel_name`]'s residual arm, present at every
/// `(dtype, warps, sg16)` combination the resident table has (Intel's default-on `mmv_mw_choice`
/// route dispatches the residual arm for its decode Linear+Add fusion — see
/// [`crate::recorder::Recorder::linear_mmv_mw`] — so the twin isn't just a parity-test convenience).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_mmv_mw_spv(
    dtype: infr_core::DType,
    res: bool,
    warps: u32,
    sg16: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match (dtype, res, warps, sg16) {
        (Q4K, false, 1, false) => v!("native_mmv_mw_q4k_w1"),
        (Q4K, true, 1, false) => v!("native_mmv_mw_q4k_w1_res"),
        (Q4K, false, 2, false) => v!("native_mmv_mw_q4k_w2"),
        (Q4K, true, 2, false) => v!("native_mmv_mw_q4k_w2_res"),
        (Q4K, false, 16, false) => v!("native_mmv_mw_q4k_w16"),
        (Q4K, true, 16, false) => v!("native_mmv_mw_q4k_w16_res"),
        (Q4K, false, 4, false) => v!("native_mmv_mw_q4k_w4"),
        (Q4K, true, 4, false) => v!("native_mmv_mw_q4k_w4_res"),
        (Q4K, false, 8, false) => v!("native_mmv_mw_q4k_w8"),
        (Q4K, true, 8, false) => v!("native_mmv_mw_q4k_w8_res"),
        (Q6K, false, 4, false) => v!("native_mmv_mw_q6k_w4"),
        (Q6K, true, 4, false) => v!("native_mmv_mw_q6k_w4_res"),
        (Q6K, false, 8, false) => v!("native_mmv_mw_q6k_w8"),
        (Q6K, true, 8, false) => v!("native_mmv_mw_q6k_w8_res"),
        (Q2K, false, 4, false) => v!("native_mmv_mw_q2k_w4"),
        (Q2K, true, 4, false) => v!("native_mmv_mw_q2k_w4_res"),
        (Q2K, false, 8, false) => v!("native_mmv_mw_q2k_w8"),
        (Q2K, true, 8, false) => v!("native_mmv_mw_q2k_w8_res"),
        (Q3K, false, 4, false) => v!("native_mmv_mw_q3k_w4"),
        (Q3K, true, 4, false) => v!("native_mmv_mw_q3k_w4_res"),
        (Q3K, false, 8, false) => v!("native_mmv_mw_q3k_w8"),
        (Q3K, true, 8, false) => v!("native_mmv_mw_q3k_w8_res"),
        (Q4K, false, 4, true) => v!("native_mmv_mw_q4k_w4_sg16"),
        (Q4K, true, 4, true) => v!("native_mmv_mw_q4k_w4_res_sg16"),
        (Q4K, false, 8, true) => v!("native_mmv_mw_q4k_w8_sg16"),
        (Q4K, true, 8, true) => v!("native_mmv_mw_q4k_w8_res_sg16"),
        (Q6K, false, 4, true) => v!("native_mmv_mw_q6k_w4_sg16"),
        (Q6K, true, 4, true) => v!("native_mmv_mw_q6k_w4_res_sg16"),
        (Q6K, false, 8, true) => v!("native_mmv_mw_q6k_w8_sg16"),
        (Q6K, true, 8, true) => v!("native_mmv_mw_q6k_w8_res_sg16"),
        (Q2K, false, 4, true) => v!("native_mmv_mw_q2k_w4_sg16"),
        (Q2K, true, 4, true) => v!("native_mmv_mw_q2k_w4_res_sg16"),
        (Q2K, false, 8, true) => v!("native_mmv_mw_q2k_w8_sg16"),
        (Q2K, true, 8, true) => v!("native_mmv_mw_q2k_w8_res_sg16"),
        (Q3K, false, 4, true) => v!("native_mmv_mw_q3k_w4_sg16"),
        (Q3K, true, 4, true) => v!("native_mmv_mw_q3k_w4_res_sg16"),
        (Q3K, false, 8, true) => v!("native_mmv_mw_q3k_w8_sg16"),
        (Q3K, true, 8, true) => v!("native_mmv_mw_q3k_w8_res_sg16"),
        (Q5K, false, 4, false) => v!("native_mmv_mw_q5k_w4"),
        (Q5K, true, 4, false) => v!("native_mmv_mw_q5k_w4_res"),
        (Q5K, false, 8, false) => v!("native_mmv_mw_q5k_w8"),
        (Q5K, true, 8, false) => v!("native_mmv_mw_q5k_w8_res"),
        _ => None,
    }
}
/// SPIR-V for the tiled Q4_K dp4a (mmq) GEMM.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the tiled Q6_K dp4a (mmq) GEMM (the MoE down projection).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// The `-DSTREAMED` dense tiled dp4a (mmq) GEMM (kernel-cache name + SPIR-V) — the weight read from
/// a `bufferDeviceAddress` arena; the sole weight build for the dense mmq set. Production
/// dispatches it via [`crate::recorder::Recorder::matmul_mmq`]; parity tests call
/// [`crate::recorder::Recorder::matmul_mmq_at`] directly with an explicit arena address. The `_xp*` EXPERT_GRID twins are also emitted (same gate in build.rs) but get their lookup
/// when the expert router grows an `arena: Option<u64>` arm at integration time.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_dense_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    macro_rules! spv {
        ($name:literal) => {{
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            ($name, S.get_or_init(|| spv_words(BYTES)).as_slice())
        }};
    }
    use infr_core::DType::*;
    Some(match dtype {
        Q4K => spv!("native_gemm_mmq_q4k_streamed"),
        Q6K => spv!("native_gemm_mmq_q6k_streamed"),
        Q8_0 => spv!("native_gemm_mmq_q8_0"),
        Q5_0 => spv!("native_gemm_mmq_q5_0"),
        Q5K => spv!("native_gemm_mmq_q5k"),
        Q5_1 => spv!("native_gemm_mmq_q5_1"),
        Q2K => spv!("native_gemm_mmq_q2_k"),
        Q3K => spv!("native_gemm_mmq_q3_k"),
        Q4_0 => spv!("native_gemm_mmq_q4_0"),
        Q4_1 => spv!("native_gemm_mmq_q4_1"),
        Iq4Nl => spv!("native_gemm_mmq_iq4_nl"),
        Iq4Xs => spv!("native_gemm_mmq_iq4_xs"),
        Iq2S => spv!("native_gemm_mmq_iq2_s"),
        Iq3S => spv!("native_gemm_mmq_iq3_s"),
        Mxfp4 => spv!("native_gemm_mmq_mxfp4"),
        Nvfp4 => spv!("native_gemm_mmq_nvfp4"),
        Q2_0 => spv!("native_gemm_mmq_q2_0"),
        _ => return None,
    })
}
/// The f16-weight decode GEMV (`linear_f16.comp`; weight read through a typed 64-bit
/// buffer_reference — see the shader's STREAMED doc) — the ONLY weight build (the bound-SSBO
/// resident build died with the eager `VulkanBackend::linear_f16` caller, its sole consumer).
/// Production entry: [`crate::recorder::Recorder::linear_at`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f16_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f16.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `linear_f16_noext` (slice A4; see the shader's
/// STREAMED doc — weight read through a typed 64-bit buffer_reference). Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f16_noext_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f16_noext.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// The bf16-weight decode GEMV (`linear_bf16.comp`; weight read through a typed 64-bit
/// buffer_reference — see the shader's STREAMED doc) — the ONLY weight build (the bound-SSBO
/// resident build died with the eager `VulkanBackend::linear_bf16` caller, its sole consumer).
/// Production entry: [`crate::recorder::Recorder::linear_bf16_at`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_bf16_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_bf16.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `linear_f32r` (slice A4; see the shader's
/// STREAMED doc — weight read through a typed 64-bit buffer_reference). Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f32r.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `linear_f32r_mrow8` (slice A4; see the shader's
/// STREAMED doc — weight read through a typed 64-bit buffer_reference). Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_mrow8_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f32r_mrow8.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `linear_f32r_mrow4_v4` (slice A4; see the shader's
/// STREAMED doc — weight read through a typed 64-bit buffer_reference). Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_mrow4_v4_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f32r_mrow4_v4.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `linear_f32r_mrow8_v4` (slice A4; see the shader's
/// STREAMED doc — weight read through a typed 64-bit buffer_reference). Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_f32r_mrow8_v4_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_f32r_mrow8_v4.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `linear_res` (slice A4; see the shader's
/// STREAMED doc — weight read through a typed 64-bit buffer_reference). Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn linear_res_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_res.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V (kernel-cache name + words) for [`native_gemm_fma_kernel_name`]'s pick; the sole weight
/// build — production dispatches it via [`crate::recorder::Recorder::matmul_fma`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_fma_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    macro_rules! spv {
        ($name:literal) => {{
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            ($name, S.get_or_init(|| spv_words(BYTES)).as_slice())
        }};
    }
    use infr_core::DType::*;
    Some(match dtype {
        F16 => spv!("native_gemm_fma_f16"),
        Bf16 => spv!("native_gemm_fma_bf16"),
        F32 => spv!("native_gemm_fma_f32"),
        _ => return None,
    })
}
/// SPIR-V for the non-coopmat float-weight prefill GEMM (the "fma-warp" tier, see
/// `native_gemm_fma.comp`): shared-memory fma warptile for f16/bf16/f32 weights on devices
/// without a usable f16 coopmat (adapter.rs `nc_fma`). Returns the kernel-cache NAME (SPIR-V via
/// [`native_gemm_fma_spv`]);
/// `None` for any other dtype.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_fma_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        F16 => "native_gemm_fma_f16",
        Bf16 => "native_gemm_fma_bf16",
        F32 => "native_gemm_fma_f32",
        _ => return None,
    })
}
/// SPIR-V for the non-coopmat shared-memory fma flash-attention prefill (the "nc_fa" tier, see
/// `attn_nc_fa.comp`): fused QK^T → online softmax → PV for devices without a usable f16 coopmat
/// (adapter.rs `nc_fa_ok`). Three shared-Os builds; returns the smallest that fits `hd` as
/// `(kernel_cache_name, spv, BM row tile)`, or `None` for hd > 512.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_nc_fa_spv(hd: usize) -> Option<(&'static str, &'static [u32], usize)> {
    macro_rules! spv {
        ($name:literal, $bm:literal) => {{
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            ($name, S.get_or_init(|| spv_words(BYTES)).as_slice(), $bm)
        }};
    }
    Some(if hd <= 128 {
        spv!("attn_nc_fa_hd128", 32)
    } else if hd <= 256 {
        spv!("attn_nc_fa_hd256", 32)
    } else if hd <= 512 {
        spv!("attn_nc_fa_hd512", 16)
    } else {
        return None;
    })
}
/// SPIR-V for the int8 cooperative-matrix (WMMA) prefill GEMM, Q8_0 only — measurement kernel
/// gated behind `INFR_I8_COOPMAT=1` (see `native_gemm_i8cm_q8_0.comp` for the design doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_i8cm_q8_0_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_i8cm_q8_0.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin of [`native_gemm_i8cm_q8_0_spv`]: the raw Q8_0 weight bytes read through an
/// inline 64-bit buffer_reference into the resident arena instead of the bound binding-2 SSBO —
/// see the shader's STREAMED doc. Parity-test entry (the `-DROW_SCALE` twin
/// `native_gemm_i8cm_q8_0_rowscale_streamed.spv` also compiles — same `rb()`/`NW()` seam, only the
/// activation-scale binding differs — but isn't wired into Rust yet; covered by composition).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_i8cm_q8_0_streamed_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_i8cm_q8_0_streamed.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V + cache name for the row-wise (whole-K) activation quant pass — int8-coopmat GEMM
/// "Idea 2" measurement variant (see `quant_q8_row.comp`), gated behind `INFR_I8_ROW_SCALE=1`.
/// `sg16` selects the `-DSG=16` twin (`caps.sg_pref == 16`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_q8_row_build_spv(sg16: bool) -> (&'static str, &'static [u32]) {
    if sg16 {
        const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8_row_sg16.spv"));
        static S: OnceLock<Vec<u32>> = OnceLock::new();
        ("quant_q8_row_sg16", S.get_or_init(|| spv_words(BYTES)))
    } else {
        const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8_row.spv"));
        static S: OnceLock<Vec<u32>> = OnceLock::new();
        ("quant_q8_row", S.get_or_init(|| spv_words(BYTES)))
    }
}
/// SPIR-V for the fp8 (E4M3) cooperative-matrix (WMMA) prefill GEMM, Q8_0 only, WIDE tile
/// (BM=64xBN=256, same warptile shape as `native_gemm_warp`) — gated behind `INFR_F8_COOPMAT=1` +
/// `caps.f8_coopmat` (see `native_gemm_f8cm_q8_0.comp` for the design doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_f8cm_q8_0.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the fp8-coopmat GEMM's NARROW_N tile (BM=64xBN=128, BK=64) — the occupancy fix for
/// n%128 (not n%256) shapes, mirroring `native_gemm_warp_n128_kernel_name`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_n128_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_f8cm_q8_0_n128.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the row-wise (whole-K) fp8 activation quant pass — the activation-scaling step for
/// `native_gemm_f8cm_q8_0` (see `quant_f8_row.comp`), gated behind `INFR_F8_COOPMAT=1` +
/// `caps.f8_coopmat`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_f8_row_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_f8_row.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the `-DPREPACK` fp8-coopmat GEMM WIDE tile: reads a pre-packed E4M3 weight buffer
/// directly (no in-shader Q8_0 dequant) — the measurement variant for whether removing the dqblk
/// bottleneck lets fp8 beat f16 (see `native_gemm_f8cm_q8_0.comp` header + `repack_q8_to_f8.comp`).
/// Gated behind `INFR_F8_COOPMAT=1` + `INFR_F8_PREPACK=1` + `caps.f8_coopmat`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_prepack_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_f8cm_q8_0_prepack.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the `-DPREPACK` fp8-coopmat GEMM's NARROW_N tile (BM=64xBN=128, BK=64) — the n%128
/// occupancy fix, mirroring `native_gemm_f8cm_q8_0_n128_spv` but reading pre-packed E4M3 weights.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_f8cm_q8_0_prepack_n128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_f8cm_q8_0_prepack_n128.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the NATIVE bf16 cooperative-matrix (WMMA) variant of the production
/// `native_gemm_warp` kernel, WIDE tile (BM=64xBN=256) — `-DFMT_BF16 -DBF16CM`, gated behind
/// `INFR_BF16_COOPMAT=1` + `caps.bf16_coopmat` (see `native_gemm_warp.comp`'s BF16CM doc). Same
/// tuned warptile as `native_gemm_warp_bf16_spv`'s f16-clamped path, just bf16-typed coopmat
/// operands — replaces the old standalone `native_gemm_bf16cm.comp` measurement kernel.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_bf16cm_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_warp_bf16cm.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the bf16-coopmat warptile's NARROW_N tile (BM=64xBN=128, BK=64) — the occupancy fix
/// for n%128 (not n%256) shapes, mirroring `native_gemm_warp_bf16cm_spv`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_bf16cm_n128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_warp_bf16cm_n128.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for `repack_q8_to_f8.comp`: bakes each Q8_0 32-block's scale into an E4M3 output
/// (decode-once via `dqblk`), producing the pre-packed weight buffer the PREPACK GEMM variants
/// above read directly. Gated behind `INFR_F8_COOPMAT=1` + `INFR_F8_PREPACK=1`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn repack_q8_to_f8_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/repack_q8_to_f8.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin of [`repack_q8_to_f8_spv`]: the Q8_0 source is read through
/// `native_weight_addr.glsl`'s `NW()` chokepoint (this file already includes `native_decode.glsl`,
/// same as `native_gemm_warp`) into the resident arena instead of the bound binding-0 SSBO. The
/// E4M3 output buffer (binding 1) is a derived scratch buffer and stays bound in both builds — see
/// the shader's STREAMED doc. Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn repack_q8_to_f8_streamed_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/repack_q8_to_f8_streamed.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the int8-coopmat GEMM's "Idea 2" whole-row-activation-scale measurement variant
/// (see `native_gemm_i8cm_q8_0.comp` #ifdef ROW_SCALE), gated behind `INFR_I8_ROW_SCALE=1`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_i8cm_q8_0_rowscale_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_i8cm_q8_0_rowscale.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the MoE weighted-accumulate (sum of selected experts' down outputs into hidden).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_accumulate_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_accumulate.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the MoE weighted-accumulate with a per-expert down-output scale (diffusion-gemma
/// `ffn_down_exps.scale`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_accumulate_scaled_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_accumulate_scaled.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the GPU MoE router top-k.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_topk_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_topk.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the expert-parallel (multi-GPU EP) band remap — rewrite router-selected GLOBAL expert
/// ids into this rank's LOCAL expert-shard indices (out-of-band → id 0, weight 0). See
/// `Op::MoeFfn`'s `ep_band` and `moe_ep_band_remap.comp`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_ep_band_remap_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_ep_band_remap.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// Kernel-cache NAME + availability for the embedding-row gather+dequant (`Op::EmbedGather`); the
/// SPIR-V lives in [`embed_gather_spv`]. `None` = format has no
/// build (grid-table IQ formats) — the runner then keeps the host embed path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn embed_gather_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    // Delegate to the SPIR-V source of truth so the gate and the loaded shader can never drift
    // (a name-table-only dtype was a mid-inference `expect()` panic; see AUDIT #1).
    embed_gather_spv(dtype).map(|(name, _)| name)
}
/// SPIR-V (kernel-cache name + words) for [`embed_gather_kernel_name`]'s pick — the token-embedding
/// table read from a `bufferDeviceAddress` arena, same convention as [`native_streamed_build_spv`];
/// the sole weight build. Production dispatches it via
/// [`crate::recorder::Recorder::embed_gather`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn embed_gather_spv(dtype: infr_core::DType) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            let s = S
                .get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice();
            Some(($name, s))
        }};
    }
    match dtype {
        Q8_0 => v!("embed_gather_q8_0"),
        Bf16 => v!("embed_gather_bf16"),
        F16 => v!("embed_gather_f16"),
        Q4_0 => v!("embed_gather_q4_0"),
        Q4_1 => v!("embed_gather_q4_1"),
        Q5_0 => v!("embed_gather_q5_0"),
        Q5_1 => v!("embed_gather_q5_1"),
        Q2K => v!("embed_gather_q2k"),
        Q3K => v!("embed_gather_q3k"),
        Q4K => v!("embed_gather_q4k"),
        Q5K => v!("embed_gather_q5k"),
        Q6K => v!("embed_gather_q6k"),
        Iq4Nl => v!("embed_gather_iq4nl"),
        Iq4Xs => v!("embed_gather_iq4xs"),
        Q2_0 => v!("embed_gather_q2_0"),
        _ => None,
    }
}
/// SPIR-V for the chained-decode id ring log (ring[pos & 63] = sampled id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn id_log_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/id_log.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the decode-replay params advance (device-side [pos, kv_len] increment).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn params_advance_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/params_advance.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the vocab sampler's slice pass (256 workgroups → 256*k (val, idx) candidates).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_topk_part_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample_topk_part.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the vocab sampler's select+softmax+nucleus+CDF pass (candidates → token id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_topk_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample_topk.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the vocab sampler's chained-decode select+softmax+nucleus+CDF pass: `u` is read
/// from the 64-slot ring at `u_buf[params[0] & 63]` instead of `u_buf[0]` (see sample_topk.comp).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn sample_topk_chain_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sample_topk_chain.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the greedy-argmax slice pass (256 workgroups → 256 (val, idx) partials).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_part_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_part.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the greedy-argmax reduce pass (256 partials → token id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the fused argmax+top1-prob slice pass (256 workgroups → 256 (max, idx, sum_exp)
/// partials) — `Op::ArgmaxProb`, the MTP draft-loop accept.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_prob_part_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_prob_part.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the fused argmax+top1-prob reduce pass (256 partials → token id + top1 probability).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn argmax_prob_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax_prob.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for GPU stochastic sampling (radix top-k + temp + top-p → token id).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_sample_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_sample.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the DiffusionGemma entropy-bound sampler reduction (perf slice 3): per-canvas-row
/// argmax/entropy/CDF-sample over `[rows, vocab]` logits — see `shaders/dg_eb_sample.comp`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dg_eb_sample_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dg_eb_sample.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// Max `top_k` the GPU sampler supports (matches the shader's KMAX); larger falls back to host.
pub const SAMPLE_KMAX: usize = 64;
/// SPIR-V for the MoE expert-bucketing passes (count / exclusive-scan / scatter).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_count_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_count.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_scan_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scan.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_scatter_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scatter.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for `moe_bucket_scatter`'s per-expert-`dscale`-baking variant (diffusion-gemma).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_bucket_scatter_scaled_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_bucket_scatter_scaled.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the indexed axpy (`acc += wts[slot]*x`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_scaled_id_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled_id.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for llama4's weight-before-FFN in-place per-row scale (`Op::MoeFfn`'s `weight_before`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_weight_scale_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_weight_scale.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

/// Kernel-cache NAME of the LARGE-WARPTILE native-block prefill GEMM (8-warp BM=64×BN=256,
/// gemm_proj_warp structure with in-shader native dequant) for `dtype`, or `None` if only the hot
/// formats are compiled (the caller falls back to the 64×64 `native_gemm` kernel). Availability +
/// kernel-cache NAME only — the SPIR-V is loaded via [`native_gemm_spv`]; the weight is read by
/// 64-bit device address from the arena, so no resident SSBO tile is bound.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Bf16 => "native_gemm_warp_bf16",
        Iq4Nl => "native_gemm_warp_iq4nl",
        Iq4Xs => "native_gemm_warp_iq4xs",
        Q2K => "native_gemm_warp_q2k",
        Q3K => "native_gemm_warp_q3k",
        Q5_0 => "native_gemm_warp_q5_0",
        Q5_1 => "native_gemm_warp_q5_1",
        Q4_0 => "native_gemm_warp_q4_0",
        Q4K => "native_gemm_warp_q4k",
        Q5K => "native_gemm_warp_q5k",
        Q6K => "native_gemm_warp_q6k",
        Q8_0 => "native_gemm_warp_q8_0",
        _ => return None,
    })
}

/// Kernel-cache NAME for the NARROW-N warptile (BN=128/BK=64) — same math per thread, 2× the workgroups; the
/// occupancy fix for n=1024/2048 GEMMs. `None` for formats without a warp build.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Bf16 => "native_gemm_warp_bf16_n128",
        Iq4Nl => "native_gemm_warp_iq4nl_n128",
        Iq4Xs => "native_gemm_warp_iq4xs_n128",
        Q2K => "native_gemm_warp_q2k_n128",
        Q3K => "native_gemm_warp_q3k_n128",
        Q5_0 => "native_gemm_warp_q5_0_n128",
        Q5_1 => "native_gemm_warp_q5_1_n128",
        Q4_0 => "native_gemm_warp_q4_0_n128",
        Q4K => "native_gemm_warp_q4k_n128",
        Q5K => "native_gemm_warp_q5k_n128",
        Q6K => "native_gemm_warp_q6k_n128",
        Q8_0 => "native_gemm_warp_q8_0_n128",
        _ => return None,
    })
}

/// 8x8x16-fragment `_cm8` warptile variants (Intel Arc/ANV XMX; `-DCM_M=8 -DCM_N=8` on the
/// NARROW_N+BM32 tile, 128 threads / pinned subgroup 16) — dispatched ONLY when the device's
/// selected f16 coopmat shape is 8x8x16 (`caps.f16_coopmat_8x8()`, which itself requires the
/// `INFR_CM_8X8=1` opt-in; see lib.rs `select_coopmat_shape`). Name+SPIR-V so `kernel_sg` caches
/// distinct pipelines. Hot k-quants + Q8_0 only; `None` routes the shape to the nc_mmq/nc_fma
/// non-coopmat tier (the Arc default).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_cm8_build_spv(
    dtype: infr_core::DType,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::*;
    macro_rules! v {
        ($name:literal) => {{
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            (
                $name,
                S.get_or_init(|| {
                    spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))
                })
                .as_slice(),
            )
        }};
    }
    Some(match dtype {
        Q4K => v!("native_gemm_warp_q4k_cm8"),
        Q6K => v!("native_gemm_warp_q6k_cm8"),
        Q8_0 => v!("native_gemm_warp_q8_0_cm8"),
        _ => return None,
    })
}

/// A_GLOBAL warptile variants: A pre-converted to f16 by the caller and coopMatLoad'd straight
/// from global memory — no As staging, no As LDS. Shrinking LDS to Bs-only lifts occupancy from
/// 2 to 3 workgroups/WGP, which is worth ~1.5x on the 8B prefill shapes (29→44 TF on the o
/// projection). Name+SPIR-V per tile so `kernel_sg` caches distinct pipelines.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_ag_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Iq4Nl => "native_gemm_warp_iq4nl_ag",
        Iq4Xs => "native_gemm_warp_iq4xs_ag",
        Q2K => "native_gemm_warp_q2k_ag",
        Q3K => "native_gemm_warp_q3k_ag",
        Q4_0 => "native_gemm_warp_q4_0_ag",
        Q4K => "native_gemm_warp_q4k_ag",
        Q5K => "native_gemm_warp_q5k_ag",
        Q6K => "native_gemm_warp_q6k_ag",
        Q8_0 => "native_gemm_warp_q8_0_ag",
        _ => return None,
    })
}

/// NARROW_N (BN=128) + A_GLOBAL.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_ag_kernel_name(
    dtype: infr_core::DType,
) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Iq4Nl => "native_gemm_warp_iq4nl_n128_ag",
        Iq4Xs => "native_gemm_warp_iq4xs_n128_ag",
        Q2K => "native_gemm_warp_q2k_n128_ag",
        Q3K => "native_gemm_warp_q3k_n128_ag",
        Q4_0 => "native_gemm_warp_q4_0_n128_ag",
        Q4K => "native_gemm_warp_q4k_n128_ag",
        Q5K => "native_gemm_warp_q5k_n128_ag",
        Q6K => "native_gemm_warp_q6k_n128_ag",
        Q8_0 => "native_gemm_warp_q8_0_n128_ag",
        _ => return None,
    })
}

/// SPLIT_K (NARROW_N tile) + A_GLOBAL.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_sk_ag_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Iq4Nl => "native_gemm_warp_iq4nl_sk_ag",
        Iq4Xs => "native_gemm_warp_iq4xs_sk_ag",
        Q2K => "native_gemm_warp_q2k_sk_ag",
        Q3K => "native_gemm_warp_q3k_sk_ag",
        Q4_0 => "native_gemm_warp_q4_0_sk_ag",
        Q4K => "native_gemm_warp_q4k_sk_ag",
        Q5K => "native_gemm_warp_q5k_sk_ag",
        Q6K => "native_gemm_warp_q6k_sk_ag",
        Q8_0 => "native_gemm_warp_q8_0_sk_ag",
        _ => return None,
    })
}

/// BM=32 row-tile variant of [`native_gemm_warp_n128_ag_kernel_name`] — see the recorder's
/// `DENSE_SMALL_TILE_MAX_M` doc. Only the formats MTP verify's qwen35-4B-UD-Q4_K_XL run actually
/// hits (Q4_K/Q5_K/Q6_K/Q8_0) are built. NOT built for the sk_ag (split-K) family —
/// `dense_small_m_row_tile_bench` measured a net LOSS there (split-K's own `splits` dimension
/// already fills the device; see `matmul_native_splitk`'s doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_ag_bm32_kernel_name(
    dtype: infr_core::DType,
) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q4K => "native_gemm_warp_q4k_n128_ag_bm32",
        Q5K => "native_gemm_warp_q5k_n128_ag_bm32",
        Q6K => "native_gemm_warp_q6k_n128_ag_bm32",
        Q8_0 => "native_gemm_warp_q8_0_n128_ag_bm32",
        _ => return None,
    })
}

/// BM=16 row-tile variant of [`native_gemm_warp_n128_ag_kernel_name`] — one coopmat M-frag floor,
/// see the recorder's `DENSE_SMALL_TILE_MAX_M16` doc. Same format coverage as the BM=32 variant
/// (Q4_K/Q5_K/Q6_K/Q8_0), no sk_ag family.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_ag_bm16_kernel_name(
    dtype: infr_core::DType,
) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q4K => "native_gemm_warp_q4k_n128_ag_bm16",
        Q5K => "native_gemm_warp_q5k_n128_ag_bm16",
        Q6K => "native_gemm_warp_q6k_n128_ag_bm16",
        Q8_0 => "native_gemm_warp_q8_0_n128_ag_bm16",
        _ => return None,
    })
}

/// Kernel-cache NAME for the A_DIRECT n128 warptile (NARROW_N + A_DIRECT, BM=64/32/16) — f32
/// activations read directly, staged to As with f32→f16 cast. Same tile shapes as the n128_ag
/// family; replaces the store_f16 + n128_ag dispatch pair. Opt-in behind INFR_GEMM_DIRECT_A=1.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_direct_kernel_name(
    dtype: infr_core::DType,
) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q4K => "native_gemm_warp_q4k_n128_direct",
        Q5K => "native_gemm_warp_q5k_n128_direct",
        Q6K => "native_gemm_warp_q6k_n128_direct",
        Q8_0 => "native_gemm_warp_q8_0_n128_direct",
        _ => return None,
    })
}

/// A_DIRECT BM=32 variant — see [`native_gemm_warp_n128_direct_kernel_name`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_direct_bm32_kernel_name(
    dtype: infr_core::DType,
) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q4K => "native_gemm_warp_q4k_n128_direct_bm32",
        Q5K => "native_gemm_warp_q5k_n128_direct_bm32",
        Q6K => "native_gemm_warp_q6k_n128_direct_bm32",
        Q8_0 => "native_gemm_warp_q8_0_n128_direct_bm32",
        _ => return None,
    })
}

/// A_DIRECT BM=16 variant — see [`native_gemm_warp_n128_direct_kernel_name`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_n128_direct_bm16_kernel_name(
    dtype: infr_core::DType,
) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q4K => "native_gemm_warp_q4k_n128_direct_bm16",
        Q5K => "native_gemm_warp_q5k_n128_direct_bm16",
        Q6K => "native_gemm_warp_q6k_n128_direct_bm16",
        Q8_0 => "native_gemm_warp_q8_0_n128_direct_bm16",
        _ => return None,
    })
}

/// Kernel-cache NAME for the SPLIT-K narrow warptile (NARROW_N + a k-split grid dimension writing partials).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_warp_sk_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        // F16 weights are floats, not quants (native_dense_supported is false) — this sk build
        // exists ONLY for the adapter's targeted deep-k narrow-n F16 route (the DiffusionGemma
        // SC soft-embedding GEMM); see the matching arm in `matmul_native_splitk`.
        F16 => "native_gemm_warp_f16_sk",
        Iq4Nl => "native_gemm_warp_iq4nl_sk",
        Iq4Xs => "native_gemm_warp_iq4xs_sk",
        Q2K => "native_gemm_warp_q2k_sk",
        Q3K => "native_gemm_warp_q3k_sk",
        Q4_0 => "native_gemm_warp_q4_0_sk",
        Q4K => "native_gemm_warp_q4k_sk",
        Q5K => "native_gemm_warp_q5k_sk",
        Q6K => "native_gemm_warp_q6k_sk",
        Q8_0 => "native_gemm_warp_q8_0_sk",
        _ => return None,
    })
}

/// SPIR-V for the split-K reduce (sum the partials planes).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn splitk_reduce_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/splitk_reduce.spv"
        )))
    })
}

const GEMM_TILED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_tiled.spv"));
const GEMM_WARP_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_warp.spv"));
const QUANT_Q8_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8.spv"));
const ATTN_PARTIAL_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_partial.spv"));
const ATTN_PARTIAL_DYNAC_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_partial_dynac.spv"));
const ATTN_COMBINE_LIVE_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_combine_live.spv"));
const ATTN_LIVE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_live.spv"));
const ATTN_QK_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_qk.spv"));
const ATTN_QK_WARP_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_qk_warp.spv"));
const ATTN_FLASH_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash.spv"));
const ATTN_FLASH_BM32_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_bm32.spv"));
const ATTN_FLASH_PARTIAL_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial.spv"));
const ATTN_FLASH_PARTIAL_BM32_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_bm32.spv"));
// Levers 2 & 5 (kv-decode-perf-levers): shmem-staged coopMatLoad prefill variants (opt-in).
const ATTN_FLASH_PARTIAL_STAGE_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_stage.spv"));
const ATTN_FLASH_PARTIAL_STAGE_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_partial_stage_bm32.spv"
));
const ATTN_FLASH_PARTIAL_DEQ_Q4_0_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_deq_q4_0.spv"));
const ATTN_FLASH_PARTIAL_DEQ_Q4_0_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_partial_deq_q4_0_bm32.spv"
));
const ATTN_FLASH_PARTIAL_DEQ_Q4_1_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_deq_q4_1.spv"));
const ATTN_FLASH_PARTIAL_DEQ_Q4_1_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_partial_deq_q4_1_bm32.spv"
));
const ATTN_FLASH_PARTIAL_DEQ_Q5_0_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_deq_q5_0.spv"));
const ATTN_FLASH_PARTIAL_DEQ_Q5_0_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_partial_deq_q5_0_bm32.spv"
));
const ATTN_FLASH_PARTIAL_DEQ_Q5_1_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_deq_q5_1.spv"));
const ATTN_FLASH_PARTIAL_DEQ_Q5_1_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_partial_deq_q5_1_bm32.spv"
));
const ATTN_FLASH_WARP_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp.spv"));
const ATTN_FLASH_WARP_BM32_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_bm32.spv"));
// Lever 2 on the warp kernel: dequant-in-flash builds that keep the register-tiled warp path.
const ATTN_FLASH_WARP_DEQ_Q4_0_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_deq_q4_0.spv"));
const ATTN_FLASH_WARP_DEQ_Q4_0_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_warp_deq_q4_0_bm32.spv"
));
const ATTN_FLASH_WARP_DEQ_Q4_1_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_deq_q4_1.spv"));
const ATTN_FLASH_WARP_DEQ_Q4_1_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_warp_deq_q4_1_bm32.spv"
));
const ATTN_FLASH_WARP_DEQ_Q5_0_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_deq_q5_0.spv"));
const ATTN_FLASH_WARP_DEQ_Q5_0_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_warp_deq_q5_0_bm32.spv"
));
const ATTN_FLASH_WARP_DEQ_Q5_1_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_deq_q5_1.spv"));
const ATTN_FLASH_WARP_DEQ_Q5_1_BM32_SPV_BYTES: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/attn_flash_warp_deq_q5_1_bm32.spv"
));
const ATTN_FLASH_REG_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_reg.spv"));
const ATTN_FLASH_REG_BR64_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_reg_br64.spv"));
// KV-cache u64/BDA twins (#74 slice 4 resurrection): the coopmat flash-prefill kernels reading K/V by
// device address (INFR_KV_COOPMAT_BDA=1, opt-in, default OFF). See attn_flash_partial.comp KV_COOPMAT_BDA.
const ATTN_FLASH_PARTIAL_BDA_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_bda.spv"));
const ATTN_FLASH_PARTIAL_BM32_BDA_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_partial_bm32_bda.spv"));
const ATTN_FLASH_WARP_BDA_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_bda.spv"));
const ATTN_FLASH_WARP_BM32_BDA_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_warp_bm32_bda.spv"));
const ATTN_FLASH_REG_BDA_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_reg_bda.spv"));
const ATTN_FLASH_REG_BR64_BDA_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_reg_br64_bda.spv"));
const ATTN_FLASH_COMBINE_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_flash_combine.spv"));
const ATTN_SM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_softmax.spv"));
const ATTN_PV_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_pv.spv"));
const ATTN_PV_WARP_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_pv_warp.spv"));
const ATTN_PV_REDUCE_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_pv_reduce.spv"));
const MLA_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mla.spv"));
const MLA_FF_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mla_ff.spv"));
const RMSNORM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
const RMSNORM_GATE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_gate.spv"));
const DELTANET_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deltanet.spv"));
const DELTANET_CHUNKED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_chunked.spv"));
const DELTANET_PREP_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_prep.spv"));
const DELTANET_GATES_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_gates.spv"));
const DELTANET_SCAN_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_scan.spv"));
const DELTANET_NORM_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_norm.spv"));
const DELTANET_GATES_SEQ_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_gates_seq.spv"));
const DELTANET_SEQ_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deltanet_seq.spv"));
const CONV1D_SHIFT_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv1d_shift.spv"));
const COPY_STRIDED_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_strided.spv"));
const MUL_SIGMOID_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_sigmoid.spv"));
const ADD_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));
const SILU_MUL_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul.spv"));
const GELU_MUL_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gelu_mul.spv"));
const SILU_MUL_FUSED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul_fused.spv"));
const GELU_MUL_FUSED_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gelu_mul_fused.spv"));
const STORE_F16_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/store_f16.spv"));
const ROPE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));
const ROPE_FF_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_ff.spv"));
const ATTENTION_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv"));
const ATTN_COMBINE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_combine.spv"));
const ATTENTION_KV_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention_kv.spv"));
const QK_NORM_ROPE_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm_rope.spv"));
const QK_NORM_ROPE_FF_SPV_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm_rope_ff.spv"));
static GEMM_TILED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static QUANT_Q8_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_PROJ_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static GEMM_PROJ_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PARTIAL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_QK_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_QK_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_STAGE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_STAGE_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q4_0_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q4_0_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q4_1_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q4_1_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q5_0_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q5_0_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q5_1_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_DEQ_Q5_1_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q4_0_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q4_0_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q4_1_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q4_1_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q5_0_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q5_0_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q5_1_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_DEQ_Q5_1_BM32_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_REG_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_REG_BR64_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_BDA_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_PARTIAL_BM32_BDA_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_BDA_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_WARP_BM32_BDA_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_REG_BDA_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_REG_BR64_BDA_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_FLASH_COMBINE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_SM_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PV_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PV_WARP_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static ATTN_PV_REDUCE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
static RMSNORM_SPV: OnceLock<Vec<u32>> = OnceLock::new();

#[cfg_attr(infr_profile, infr_prof::instrument)]
fn gemm_tiled_spv() -> &'static [u32] {
    GEMM_TILED_SPV.get_or_init(|| spv_words(GEMM_TILED_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn gemm_warp_spv() -> &'static [u32] {
    GEMM_WARP_SPV.get_or_init(|| spv_words(GEMM_WARP_SPV_BYTES))
}
/// SPIR-V for the activation int8 quantize pass (Q8 per block) feeding the dp4a mmq matmul.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_q8_spv() -> &'static [u32] {
    QUANT_Q8_SPV.get_or_init(|| spv_words(QUANT_Q8_SPV_BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `gemm_proj` (the coopmat f16 projection GEMM, weight read through a
/// typed 64-bit buffer_reference — see the shader's STREAMED doc). Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gemm_proj_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_proj.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// `-DSTREAMED` twin SPIR-V of `gemm_proj_warp` (same STREAMED convention as `gemm_proj`).
/// Parity-test entry.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gemm_proj_warp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_proj_warp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the subgroup-reduction flash-decoding pass-1 (split-K) kernel. Used by the recorder.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_partial_spv() -> &'static [u32] {
    ATTN_PARTIAL_SPV.get_or_init(|| spv_words(ATTN_PARTIAL_SPV_BYTES))
}
/// Rows-batched split pass 1 (K/V streamed once per 4-row group; chunk <= 256).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_partial_mrows_c256_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_partial_mrows_c256.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the non-FA prefill attention kernels (QK scores / row softmax / PV). Recorder use.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_qk_spv() -> &'static [u32] {
    ATTN_QK_SPV.get_or_init(|| spv_words(ATTN_QK_SPV_BYTES))
}
/// 8-warp/256-thread QK GEMM (kv_pad % 256). Matches ollama's mul_mm warptile; the recorder uses it
/// over the 4-warp attn_qk unless INFR_NO_QK_WARP is set.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_qk_warp_spv() -> &'static [u32] {
    ATTN_QK_WARP_SPV.get_or_init(|| spv_words(ATTN_QK_WARP_SPV_BYTES))
}
/// Fused flash-attention prefill (QK→softmax→PV, no materialized S). Recorder `attention_prefill_flash`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_spv() -> &'static [u32] {
    ATTN_FLASH_SPV.get_or_init(|| spv_words(ATTN_FLASH_SPV_BYTES))
}
/// BM=32 build of the fused flash prefill (29056 B shared) for sub-64 KB shared devices.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_BM32_SPV_BYTES))
}
/// Flash-attention split-K partial pass (per kv-split online-softmax partials). Recorder use.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_SPV_BYTES))
}
/// BM=32 build of the split-K flash partial (29056 B shared) for sub-64 KB shared devices.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_BM32_SPV_BYTES))
}
/// Lever 5 (INFR_FLASH_STAGE): f16-staged coopMatLoad build of the flash partial. Stages the K/V
/// tile into a 32-wide `KVstg` shmem slab (reused K→V) and `coopMatLoad`s from shmem. `bm32` picks
/// the small-tile twin.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_stage_spv(bm32: bool) -> (&'static str, &'static [u32]) {
    if bm32 {
        (
            "attn_flash_partial_stage_bm32",
            ATTN_FLASH_PARTIAL_STAGE_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_STAGE_BM32_SPV_BYTES)),
        )
    } else {
        (
            "attn_flash_partial_stage",
            ATTN_FLASH_PARTIAL_STAGE_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_STAGE_SPV_BYTES)),
        )
    }
}
/// Lever 2 (INFR_FLASH_DEQUANT): dequant-in-flash build for a quantized KV cache — decodes the
/// GGUF block cache to f16 in the stage (native_decode `dqblk`), skipping the dequant_kv_f16
/// prepass. `None` for a format without a build. `bm32` picks the small-tile twin.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_deq_spv(
    dt: infr_core::DType,
    bm32: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::{Q4_0, Q4_1, Q5_0, Q5_1};
    let (name, spv): (&'static str, &'static [u32]) = match (dt, bm32) {
        (Q4_0, false) => (
            "attn_flash_partial_deq_q4_0",
            ATTN_FLASH_PARTIAL_DEQ_Q4_0_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q4_0_SPV_BYTES)),
        ),
        (Q4_0, true) => (
            "attn_flash_partial_deq_q4_0_bm32",
            ATTN_FLASH_PARTIAL_DEQ_Q4_0_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q4_0_BM32_SPV_BYTES)),
        ),
        (Q4_1, false) => (
            "attn_flash_partial_deq_q4_1",
            ATTN_FLASH_PARTIAL_DEQ_Q4_1_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q4_1_SPV_BYTES)),
        ),
        (Q4_1, true) => (
            "attn_flash_partial_deq_q4_1_bm32",
            ATTN_FLASH_PARTIAL_DEQ_Q4_1_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q4_1_BM32_SPV_BYTES)),
        ),
        (Q5_0, false) => (
            "attn_flash_partial_deq_q5_0",
            ATTN_FLASH_PARTIAL_DEQ_Q5_0_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q5_0_SPV_BYTES)),
        ),
        (Q5_0, true) => (
            "attn_flash_partial_deq_q5_0_bm32",
            ATTN_FLASH_PARTIAL_DEQ_Q5_0_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q5_0_BM32_SPV_BYTES)),
        ),
        (Q5_1, false) => (
            "attn_flash_partial_deq_q5_1",
            ATTN_FLASH_PARTIAL_DEQ_Q5_1_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q5_1_SPV_BYTES)),
        ),
        (Q5_1, true) => (
            "attn_flash_partial_deq_q5_1_bm32",
            ATTN_FLASH_PARTIAL_DEQ_Q5_1_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_DEQ_Q5_1_BM32_SPV_BYTES)),
        ),
        _ => return None,
    };
    Some((name, spv))
}
/// Register-blocked flash partial (hd=128). Used over attn_flash_partial when hd==128.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_warp_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_SPV_BYTES))
}
/// BM=32 build of the flash partial (29056 B shared vs 58112 B): for devices whose
/// maxComputeSharedMemorySize is under the 64 KB the default BM=64 tile needs (NVIDIA, MoltenVK).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_warp_bm32_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_BM32_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_BM32_SPV_BYTES))
}
/// Lever 2 on the WARP kernel (INFR_FLASH_DEQUANT): dequant-in-flash build that KEEPS the fast
/// register-tiled warp path — decodes the quantized GGUF block cache (Q4_0/Q4_1/Q5_0/Q5_1) to f16
/// in the QK/PV stage (native_decode `dqblk`), skipping the dequant_kv_f16 prepass without dropping
/// to the ~2x-slower `attn_flash_partial`. `None` for a format without a build. `bm32` picks the
/// small-tile twin (small-m deep-kv split tiles / INFR_FLASH_BM=32 coverage).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_warp_deq_spv(
    dt: infr_core::DType,
    bm32: bool,
) -> Option<(&'static str, &'static [u32])> {
    use infr_core::DType::{Q4_0, Q4_1, Q5_0, Q5_1};
    let (name, spv): (&'static str, &'static [u32]) = match (dt, bm32) {
        (Q4_0, false) => (
            "attn_flash_warp_deq_q4_0",
            ATTN_FLASH_WARP_DEQ_Q4_0_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q4_0_SPV_BYTES)),
        ),
        (Q4_0, true) => (
            "attn_flash_warp_deq_q4_0_bm32",
            ATTN_FLASH_WARP_DEQ_Q4_0_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q4_0_BM32_SPV_BYTES)),
        ),
        (Q4_1, false) => (
            "attn_flash_warp_deq_q4_1",
            ATTN_FLASH_WARP_DEQ_Q4_1_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q4_1_SPV_BYTES)),
        ),
        (Q4_1, true) => (
            "attn_flash_warp_deq_q4_1_bm32",
            ATTN_FLASH_WARP_DEQ_Q4_1_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q4_1_BM32_SPV_BYTES)),
        ),
        (Q5_0, false) => (
            "attn_flash_warp_deq_q5_0",
            ATTN_FLASH_WARP_DEQ_Q5_0_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q5_0_SPV_BYTES)),
        ),
        (Q5_0, true) => (
            "attn_flash_warp_deq_q5_0_bm32",
            ATTN_FLASH_WARP_DEQ_Q5_0_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q5_0_BM32_SPV_BYTES)),
        ),
        (Q5_1, false) => (
            "attn_flash_warp_deq_q5_1",
            ATTN_FLASH_WARP_DEQ_Q5_1_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q5_1_SPV_BYTES)),
        ),
        (Q5_1, true) => (
            "attn_flash_warp_deq_q5_1_bm32",
            ATTN_FLASH_WARP_DEQ_Q5_1_BM32_SPV
                .get_or_init(|| spv_words(ATTN_FLASH_WARP_DEQ_Q5_1_BM32_SPV_BYTES)),
        ),
        _ => return None,
    };
    Some((name, spv))
}
/// FlashAttention-2 register-O flash partial (Br=128, per-thread register accumulator). hd=128.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_reg_spv() -> &'static [u32] {
    ATTN_FLASH_REG_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_SPV_BYTES))
}
/// BR=64 build of the register-O flash partial (29440 B shared) for sub-64 KB shared devices.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_reg_br64_spv() -> &'static [u32] {
    ATTN_FLASH_REG_BR64_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_BR64_SPV_BYTES))
}
// KV-cache u64/BDA twins (#74 slice 4 resurrection): the coopmat flash-prefill kernels reading K/V by
// device address (kv_addr.glsl `KvMat(k_addr)/(v_addr)` coopMatLoad base). OPT-IN via
// `INFR_KV_COOPMAT_BDA=1` (default OFF): on RADV/RDNA3 the coopmat load gets no saddr from a
// buffer_reference base (~+33-40% code, ~0.8x prefill) so the bound build stays the default; kept for
// silicon that addresses coopmat loads better. Selected by `attention_prefill_flash`/`_reg` when the
// KV buffers report a device address AND the caller opts in (adapter's `kv_coopmat_bda` gate).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_bda_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_BDA_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_BDA_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_partial_bm32_bda_spv() -> &'static [u32] {
    ATTN_FLASH_PARTIAL_BM32_BDA_SPV.get_or_init(|| spv_words(ATTN_FLASH_PARTIAL_BM32_BDA_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_warp_bda_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_BDA_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_BDA_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_warp_bm32_bda_spv() -> &'static [u32] {
    ATTN_FLASH_WARP_BM32_BDA_SPV.get_or_init(|| spv_words(ATTN_FLASH_WARP_BM32_BDA_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_reg_bda_spv() -> &'static [u32] {
    ATTN_FLASH_REG_BDA_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_BDA_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_reg_br64_bda_spv() -> &'static [u32] {
    ATTN_FLASH_REG_BR64_BDA_SPV.get_or_init(|| spv_words(ATTN_FLASH_REG_BR64_BDA_SPV_BYTES))
}
/// Flash-attention split-K combine (merge partials → final O). Recorder use.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_flash_combine_spv() -> &'static [u32] {
    ATTN_FLASH_COMBINE_SPV.get_or_init(|| spv_words(ATTN_FLASH_COMBINE_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_softmax_spv() -> &'static [u32] {
    ATTN_SM_SPV.get_or_init(|| spv_words(ATTN_SM_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_pv_spv() -> &'static [u32] {
    ATTN_PV_SPV.get_or_init(|| spv_words(ATTN_PV_SPV_BYTES))
}
/// 8-warp/256-thread PV GEMM (BN=128=hd, hd % 128). The recorder uses it over the 4-warp attn_pv
/// when hd % 128 == 0 and INFR_NO_PV_WARP is unset.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_pv_warp_spv() -> &'static [u32] {
    ATTN_PV_WARP_SPV.get_or_init(|| spv_words(ATTN_PV_WARP_SPV_BYTES))
}
/// SPIR-V for the attn_pv split-K partial reducer (sums n_splits partial-O buffers).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_pv_reduce_spv() -> &'static [u32] {
    ATTN_PV_REDUCE_SPV.get_or_init(|| spv_words(ATTN_PV_REDUCE_SPV_BYTES))
}
/// SPIR-V for MLA attention (`mla.comp`) — DeepSeek V2/V3 absorbed-form latent attention.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mla_spv() -> &'static [u32] {
    static MLA_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    MLA_SPV.get_or_init(|| spv_words(MLA_SPV_BYTES))
}
/// SPIR-V for MLA attention with YaRN freq_factors (`mla.comp`'s -DFREQ_FACTORS build): the
/// internal q_pe rope DIVIDES its angle by the bound per-pair divisor `ff[p]`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mla_ff_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(MLA_FF_SPV_BYTES))
}
/// SPIR-V for MLA attention with deepseek32's additive top-k score mask (`mla.comp`'s -DKEY_BIAS
/// build, `Op::Mla::key_bias`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mla_bias_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/mla_bias.spv"))))
}
/// SPIR-V for MLA attention with BOTH the YaRN divisors and the top-k score mask (`mla.comp`'s
/// -DFREQ_FACTORS -DKEY_BIAS build) — deepseek32's production shape.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mla_ff_bias_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/mla_ff_bias.spv"))))
}
/// SPIR-V for the int8 coopmat accumulator-layout known-answer probe (`coopmat_i8_layout.comp`) —
/// dispatched once at init when the i8 coopmat tier is opted into, to check that this driver lays a
/// 16x16 SINT32 accumulator fragment out the way `native_gemm_i8cm_q8_0.comp` reads it.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn coopmat_i8_layout_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/coopmat_i8_layout.spv"
        )))
    })
}
/// SPIR-V for the 256-thread subgroup RMSNorm (`y=rmsnorm(x,w)`). Used by the recorder's `rmsnorm`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_spv() -> &'static [u32] {
    RMSNORM_SPV.get_or_init(|| spv_words(RMSNORM_SPV_BYTES))
}
/// SPIR-V for the 1024-thread vec4 RMSNorm (`rmsnorm.comp`'s -DWIDE build) — the decode-shaped
/// twin of [`rmsnorm_spv`], picked by the recorder when `rows==1 && dim>=2048 && dim%4==0`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_wide_spv() -> &'static [u32] {
    static RMSNORM_WIDE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    RMSNORM_WIDE_SPV.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/rmsnorm_wide.spv"
        )))
    })
}
/// SPIR-V for the fused per-head RMSNorm + SiLU gate multiply (`rmsnorm.comp`'s -DGATE build,
/// `Op::GatedRmsNorm`) — the qwen35 DeltaNet z-gate, one dispatch instead of `rmsnorm`+`silu_mul`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_gate_spv() -> &'static [u32] {
    static RMSNORM_GATE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    RMSNORM_GATE_SPV.get_or_init(|| spv_words(RMSNORM_GATE_SPV_BYTES))
}
/// SPIR-V for fused RMSNorm + in-place add (`rmsnorm.comp`'s -DADD build, `Op::RmsNormAdd`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_add_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_add.spv"))))
}
/// SPIR-V for f16-in/f16-out RMSNorm (`rmsnorm.comp`'s -DF16IO build) — llama4's post-rope
/// weightless Q/K L2-norm (`Op::QkNorm` on the f16 rope scratch).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_f16_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_f16.spv"))))
}
/// SPIR-V for the WEIGHTLESS RMSNorm (`rmsnorm.comp`'s -DNO_WEIGHT build) — `Op::QkNorm` with
/// `weight: None`, deepseek4's bare per-head `ggml_rms_norm` on Q. No `w` binding at all.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rmsnorm_nw_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_nw.spv"))))
}
/// SPIR-V for the 256-thread subgroup mean-centred LayerNorm (`layernorm.comp`, `Op::LayerNorm`)
/// — deepseek32's `indexer_k_norm`. Used by the recorder's `layernorm`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn layernorm_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/layernorm.spv"))))
}
/// SPIR-V for the DeepSeek V3.2 lightning indexer (`lightning_indexer.comp`,
/// `Op::LightningIndexer`) — one workgroup per query row, scores then top-k. Used by the recorder's
/// `lightning_indexer`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn lightning_indexer_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/lightning_indexer.spv"
        )))
    })
}
/// SPIR-V for the top-k index → additive score mask expansion (`topk_mask.comp`, `Op::TopkMask`)
/// — one workgroup per query row, fill then scatter. Used by the recorder's `topk_mask`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn topk_mask_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/topk_mask.spv"))))
}
/// SPIR-V for the DeepSeek V4 hyper-connection coefficients (`hyper_mix.comp`,
/// `Op::HyperConnectMix`) — the `build_hc_head` form: `pre` only, no `post`/`comb`. Used by the
/// recorder's `hyper_mix`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn hyper_mix_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/hyper_mix.spv"))))
}
/// SPIR-V for the sublayer-wrapping `hyper_mix.comp` build (`-DGATES`) — `pre`, `post` and the
/// Sinkhorn-normalised `comb`. Used by the recorder's `hyper_mix`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn hyper_mix_gates_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/hyper_mix_gates.spv"
        )))
    })
}
/// SPIR-V for the hyper-connection stream collapse (`hyper_pre.comp`, `Op::HyperConnectPre`).
/// Used by the recorder's `hyper_pre`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn hyper_pre_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/hyper_pre.spv"))))
}
/// SPIR-V for the hyper-connection stream re-expansion (`hyper_post.comp`,
/// `Op::HyperConnectPost`). Used by the recorder's `hyper_post`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn hyper_post_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/hyper_post.spv"))))
}
/// SPIR-V for the 256-thread subgroup row-softmax (`y=softmax(x*scale)`). Used by the recorder's
/// `softmax` (diffusion-gemma's in-graph self-conditioning).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn softmax_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the elementwise add (`y=a+b`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_spv() -> &'static [u32] {
    static ADD_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ADD_SPV.get_or_init(|| spv_words(ADD_SPV_BYTES))
}
/// SPIR-V for the scaled add / axpy (`acc += scale*x`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_scaled_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_scaled.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the broadcast bias add (`dst[i] = x[i] + bias[i % n]`; Qwen2 q/k/v projections).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn add_bias_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_bias.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the broadcast vector multiply (`dst[i] = x[i] * vec[i % n]`; diffusion-gemma's
/// router input scale — the multiplicative twin of `add_bias`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mul_vec_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mul_vec.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the qwen35moe shared-expert combine (`dst[r,c] = moe[r,c] + sigmoid(gate[r]) *
/// shexp[r,c]`; row-broadcast gate — the shared-expert twin of `mul_vec`'s column broadcast).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_shared_expert_add_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_shared_expert_add.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the in-place scalar multiply (`y *= scale`). gemma4 per-layer output scale.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn scale_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scale.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for elementwise softcap `y = cap·tanh(x/cap)` (gemma logit softcap).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn softcap_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softcap.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

// BM=32 row-tile expert-grid variants — see build.rs's `_xp32` entries and
// `matmul_mmq_experts`'s small-rows-per-expert doc.
// BM=32 + BN=128 (wide-N) expert-grid variants — see build.rs's `_xp32w` entries and
// `matmul_mmq_experts`'s wide-BN doc. Only the two dtypes real small-row MoE pools use.
// BM=64 (default row tile) + BN=128 (wide-N) expert-grid variants — see build.rs's `_xp128`
// entries and `matmul_mmq_experts`'s wide-BN doc. Same two dtypes as the `_xp32w` small-tile pair.

// PAGED expert-grid variants — the batched-MoE mmq GEMM reading a GpuPager arena through the
// word-base LUT (Scout's batched paged prefill; see the shaders' PAGED doc).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_nl_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_mmq_iq4_nl_xpg32.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_xs_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/native_gemm_mmq_iq4_xs_xpg32.spv"
    ));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

// PAGED expert-grid variants for the remaining mmq dtype set (Q8_0/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K)
// — `MOE_MMQ_PAGED_DTYPES` now mirrors `MOE_MMQ_DTYPES` in full (fused gemma-4 MoE /
// DiffusionGemma banks and UD mixed-dtype roles page through these).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xpg_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xpg.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xpg32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xpg32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

// STREAMED (`-DSTREAMED`) twins of the expert-grid `_xp*` getters above — the
// resident-BDA fork of `matmul_mmq_experts` (see recorder.rs) reads the stacked expert
// bank from a `bufferDeviceAddress` arena instead of a bound SSBO; `pc.m` is repurposed as
// the per-expert BYTE stride under this build (see the shaders' `#elif defined(STREAMED)`
// EXPERT_GRID arm). PAGED builds never get a twin (STREAMED is the LUT-less resident form
// of the same arena read) — same exclusion build.rs applies to the dense family.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp32w_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp32w.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4k_xp128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4k_xp128.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp32w_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp32w.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q6k_xp128_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q6k_xp128.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q8_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q8_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q5_1_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q5_1_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q3_k_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q3_k_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q2_0_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q2_0_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq2_s_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq2_s_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq3_s_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq3_s_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_q4_1_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_q4_1_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_nl_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_nl_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_nl_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_xs_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_iq4_xs_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_iq4_xs_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_mxfp4_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_mxfp4_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xp_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xp.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn native_gemm_mmq_nvfp4_xp32_spv() -> &'static [u32] {
    const BYTES: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/native_gemm_mmq_nvfp4_xp32.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_q8_gather_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quant_q8_gather.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn moe_scatter_reduce_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/moe_scatter_reduce.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the row gather (`dst[j]=src[idx[j]]`).
/// SPIR-V for the weighted row scatter-add (`dst[idx[j]] += w[j]*y[j]`).
/// SPIR-V for the SwiGLU activation (`y=silu(gate)*up`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn silu_mul_spv() -> &'static [u32] {
    static SILU_MUL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    SILU_MUL_SPV.get_or_init(|| spv_words(SILU_MUL_SPV_BYTES))
}
/// SPIR-V for the gated-DeltaNet recurrence step (qwen35/Qwen3.5 SSM).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SPV_BYTES))
}
/// SPIR-V for DeltaNet reading q/k/v from a single strided source buffer (env-gated).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_strided_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/deltanet_strided.spv"
        )))
    })
}
/// SPIR-V for the chunked-DeltaNet PREP pass (normalize + intra-chunk dot matrices).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_prep_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_PREP_SPV_BYTES))
}
/// SPIR-V for the chunked-DeltaNet GATES pass (β + prefix log-decay per chunk/head).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_gates_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_GATES_SPV_BYTES))
}
/// SPIR-V for the chunked-DeltaNet SCAN pass (the sequential state-coupled part).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_scan_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SCAN_SPV_BYTES))
}
/// SPIR-V for the sequential-DeltaNet NORM pass (q/k L2-normalize; prep without the D/Dq dots).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_norm_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_NORM_SPV_BYTES))
}
/// SPIR-V for the sequential-DeltaNet GATES pass (β + per-token decay, flat rows·nv grid).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_gates_seq_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_GATES_SEQ_SPV_BYTES))
}
/// SPIR-V for the token-serial DeltaNet SCAN with the state column register-resident.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_seq_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_SEQ_SPV_BYTES))
}
/// SPIR-V for the CHUNKED gated-DeltaNet prefill (chunkwise delta rule, C=32).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn deltanet_chunked_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(DELTANET_CHUNKED_SPV_BYTES))
}
/// SPIR-V for the BATCH depthwise conv1d history rebuild (pass 2; never reads the weight, so it
/// has no STREAMED twin — the same build serves both the sequential and batch conv paths).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn conv1d_shift_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(CONV1D_SHIFT_SPV_BYTES))
}
/// The qwen35 SSM input conv, one token (`conv1d_silu.comp`; weight read through a typed 64-bit
/// buffer_reference — see the shader's STREAMED doc) — the ONLY weight build (the bound-SSBO
/// resident build died with `Recorder::conv1d_silu`'s own resident dispatch, its sole consumer).
/// Production entry: [`crate::recorder::Recorder::conv1d_silu`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn conv1d_silu_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv1d_silu.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// BATCH pass 1 of the qwen35 SSM input conv (`conv1d_silu_par.comp`; same STREAMED convention as
/// `conv1d_silu`) — the ONLY weight build (the bound-SSBO resident build died with
/// `Recorder::conv1d_silu_batch`'s own resident dispatch, its sole consumer). Production entry:
/// [`crate::recorder::Recorder::conv1d_silu_batch`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn conv1d_silu_par_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/conv1d_silu_par.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for the batched strided row copy (Op::CopyStrided in one dispatch).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn copy_strided_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(COPY_STRIDED_SPV_BYTES))
}
/// SPIR-V for the elementwise sigmoid gate `y = a * sigmoid(b)`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn mul_sigmoid_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(MUL_SIGMOID_SPV_BYTES))
}
/// SPIR-V for the GeGLU activation with separate gate/up buffers (`y=gelu(gate)*up`). gemma4's
/// per-layer-embd gate.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gelu_mul_spv() -> &'static [u32] {
    static GELU_MUL_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    GELU_MUL_SPV.get_or_init(|| spv_words(GELU_MUL_SPV_BYTES))
}
/// SPIR-V for the fused SwiGLU over a combined gate||up buffer.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn silu_mul_fused_spv() -> &'static [u32] {
    static SILU_MUL_FUSED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    SILU_MUL_FUSED_SPV.get_or_init(|| spv_words(SILU_MUL_FUSED_SPV_BYTES))
}
/// SPIR-V for the fused GeGLU (GELU tanh-approx gate) over a combined gate||up buffer (gemma).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn gelu_mul_fused_spv() -> &'static [u32] {
    static GELU_MUL_FUSED_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    GELU_MUL_FUSED_SPV.get_or_init(|| spv_words(GELU_MUL_FUSED_SPV_BYTES))
}
/// SPIR-V for the f32→f16 cast-store into an f16 cache.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn store_f16_spv() -> &'static [u32] {
    static STORE_F16_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    STORE_F16_SPV.get_or_init(|| spv_words(STORE_F16_SPV_BYTES))
}
/// SPIR-V for RoPE (ggml NORM, interleaved pairs).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rope_spv() -> &'static [u32] {
    static ROPE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ROPE_SPV.get_or_init(|| spv_words(ROPE_SPV_BYTES))
}
/// SPIR-V for RoPE with YaRN freq_factors (`rope.comp`'s -DFREQ_FACTORS build): the rotation
/// angle is DIVIDED by the bound per-pair divisor `ff[p]`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rope_ff_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(ROPE_FF_SPV_BYTES))
}
/// SPIR-V for NEOX RoPE (`rope.comp`'s -DNEOX build): pairs `(i, i + rope_dim/2)` instead of the
/// interleaved `(2i, 2i+1)` — deepseek32's lightning indexer (`Op::Rope::neox`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rope_neox_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/rope_neox.spv"))))
}
/// SPIR-V for NEOX RoPE with YaRN freq_factors (`rope.comp`'s -DNEOX -DFREQ_FACTORS build) —
/// deepseek32's production indexer shape (its rope carries the same YaRN ramp as the MLA one).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rope_ff_neox_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/rope_ff_neox.spv"
        )))
    })
}
/// SPIR-V for BACKWARD RoPE (`rope.comp`'s -DBACKWARD build): `sin` negated, `cos` untouched —
/// llama.cpp's `ggml_rope_ext_back` (`Op::Rope::backward`), deepseek4's attention-output de-rope.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rope_back_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(include_bytes!(concat!(env!("OUT_DIR"), "/rope_back.spv"))))
}
/// SPIR-V for backward RoPE with YaRN freq_factors (`rope.comp`'s -DBACKWARD -DFREQ_FACTORS
/// build) — V4's compressed layers rope (and de-rope) with the YaRN ramp; its ratio-0 layers use
/// the plain [`rope_back_spv`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn rope_ff_back_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/rope_ff_back.spv"
        )))
    })
}
/// gemma4 E2B's per-layer fused inp_gate GEMV (`e2b_gate.comp`; weight read through a typed
/// 64-bit buffer_reference — see the shader's STREAMED doc) — the ONLY weight build (the
/// bound-SSBO resident build died with `Recorder::e2b_gate`'s own resident dispatch, its sole
/// production consumer). Production entry: [`crate::recorder::Recorder::e2b_gate`].
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn e2b_gate_spv() -> &'static [u32] {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/e2b_gate.spv"));
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(BYTES))
}
/// SPIR-V for fused QkNormRope reading from interleaved q+g buffer (qwen35 CopyStrided elim).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn qk_norm_rope_interleaved_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/qk_norm_rope_interleaved.spv"
        )))
    })
}
/// SPIR-V for interleaved QkNormRope with USE_PARAMS (record-once decode replay).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn qk_norm_rope_interleaved_dyn_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| {
        spv_words(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/qk_norm_rope_interleaved_dyn.spv"
        )))
    })
}
/// SPIR-V for the online-softmax GQA attention (hd<=128).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attention_spv() -> &'static [u32] {
    static ATTENTION_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTENTION_SPV.get_or_init(|| spv_words(ATTENTION_SPV_BYTES))
}
/// SPIR-V for flash-decode combine (merge split-K partials).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_combine_spv() -> &'static [u32] {
    static ATTN_COMBINE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTN_COMBINE_SPV.get_or_init(|| spv_words(ATTN_COMBINE_SPV_BYTES))
}
/// SPIR-V for tiled online-softmax attention over an f16 KV cache.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attention_kv_spv() -> &'static [u32] {
    static ATTENTION_KV_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    ATTENTION_KV_SPV.get_or_init(|| spv_words(ATTENTION_KV_SPV_BYTES))
}
/// SPIR-V for fused per-head QK-norm + NEOX RoPE (f16 out).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn qk_norm_rope_spv() -> &'static [u32] {
    static QK_NORM_ROPE_SPV: OnceLock<Vec<u32>> = OnceLock::new();
    QK_NORM_ROPE_SPV.get_or_init(|| spv_words(QK_NORM_ROPE_SPV_BYTES))
}
/// SPIR-V for QK-norm + RoPE with proportional-rope freq_factors (gemma4 full-attention layers).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn qk_norm_rope_ff_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(QK_NORM_ROPE_FF_SPV_BYTES))
}
// Record-once decode variants (`-DUSE_PARAMS`): read the per-token pos/kv_len from a host-updated
// params SSBO instead of push constants, so the decode command buffer can be replayed across tokens.
macro_rules! dyn_spv {
    ($f:ident, $name:literal) => {
        pub(crate) fn $f() -> &'static [u32] {
            const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"));
            static S: OnceLock<Vec<u32>> = OnceLock::new();
            S.get_or_init(|| spv_words(BYTES))
        }
    };
}
dyn_spv!(qk_norm_rope_dyn_spv, "qk_norm_rope_dyn");
dyn_spv!(qk_norm_rope_dyn_ff_spv, "qk_norm_rope_dyn_ff");
// Same "read a scalar from a device buffer instead of a push constant" idea, but for
// DiffusionGemma denoise self-conditioning's softmax scale (`-DUSE_SCALE_BUF`) rather than the
// record-once decode replay's pos/kv_len — see `Op::Softmax::scale_buf`'s doc and `Recorder::
// softmax_dyn`.
dyn_spv!(softmax_dyn_spv, "softmax_dyn");
dyn_spv!(rope_f16_spv, "rope_f16");
dyn_spv!(rope_f16_dyn_spv, "rope_f16_dyn");
dyn_spv!(store_f16_dyn_spv, "store_f16_dyn");
dyn_spv!(attention_kv_dyn_spv, "attention_kv_dyn");
dyn_spv!(attn_partial_dyn_spv, "attn_partial_dyn");
// A/B escape for the hd=256/512 attn_partial fast paths (INFR_NO_ATTN_HD=1): the same three f16
// form-factors compiled with -DNO_HD_SPEC, so a regression on those shapes is diagnosable
// against the general per-key loops.
dyn_spv!(attn_partial_nohd_spv, "attn_partial_nohd");
dyn_spv!(attn_partial_dyn_nohd_spv, "attn_partial_dyn_nohd");
dyn_spv!(attn_partial_dynac_nohd_spv, "attn_partial_dynac_nohd");
/// `INFR_NO_ATTN_HD=1` (`kernels.vulkan.no_attn_hd_spec`) — select the `-DNO_HD_SPEC` attn_partial
/// variants (general loops only).
///
/// Takes the knob as a BORROWED config (S5b): it used to be a `std::env::var` here, deliberately
/// un-memoized because a `OnceLock` latched whatever the first attention record saw and made the
/// knob unsettable from a test (§10.6). Reading it off the recorder's backing `Config` removes both
/// the memo question and the `getenv`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_hd_spec_disabled(vk: &infr_core::config::VulkanCfg) -> bool {
    vk.no_attn_hd_spec
}
// Coupled Q8_0 KV cache: scalar dequant-on-read attention (static + record-once) and the row
// quantize-store kernels (f32 V + f16 K sources, each with a params/decode variant).
dyn_spv!(attention_kv_q8_spv, "attention_kv_q8");
dyn_spv!(attention_kv_kq8_spv, "attention_kv_kq8");
dyn_spv!(attention_kv_vq8_spv, "attention_kv_vq8");
dyn_spv!(attention_kv_dyn_q8_spv, "attention_kv_dyn_q8");
// Per-head attention sinks (`attention_kv.comp`'s -DSINKS build, `Op::Attention::sinks`) — the one
// build that carries them; the adapter routes a sinks op onto this scalar f16 path.
dyn_spv!(attention_kv_sinks_spv, "attention_kv_sinks");
// KV-cache u64/BDA twins (#74, slice 1): the `-DKV_BDA` builds of the scalar attention_kv family —
// K/V read by 64-bit device address (kv_addr.glsl) instead of bound SSBOs. Bound twins above stay
// for kv_addr_parity.rs; production forks here via Recorder::attention_kv_at / attention_kv_dyn_at.
dyn_spv!(attention_kv_bda_spv, "attention_kv_bda");
dyn_spv!(attention_kv_dyn_bda_spv, "attention_kv_dyn_bda");
dyn_spv!(attention_kv_q8_bda_spv, "attention_kv_q8_bda");
dyn_spv!(attention_kv_kq8_bda_spv, "attention_kv_kq8_bda");
dyn_spv!(attention_kv_vq8_bda_spv, "attention_kv_vq8_bda");
dyn_spv!(attention_kv_dyn_q8_bda_spv, "attention_kv_dyn_q8_bda");
dyn_spv!(store_q8_spv, "store_q8");
dyn_spv!(store_q8_dyn_spv, "store_q8_dyn");
dyn_spv!(store_q8_f16_spv, "store_q8_f16");
dyn_spv!(store_q8_f16_dyn_spv, "store_q8_f16_dyn");
// Mainline low-bit KV quants: per-format quantize (f32 V / f16 K) + dequant→f16 prefix expander.
dyn_spv!(quant_kv_q4_0_spv, "quant_kv_q4_0");
dyn_spv!(quant_kv_q4_0_f16_spv, "quant_kv_q4_0_f16");
dyn_spv!(quant_kv_q4_1_spv, "quant_kv_q4_1");
dyn_spv!(quant_kv_q4_1_f16_spv, "quant_kv_q4_1_f16");
dyn_spv!(quant_kv_q5_0_spv, "quant_kv_q5_0");
dyn_spv!(quant_kv_q5_0_f16_spv, "quant_kv_q5_0_f16");
dyn_spv!(quant_kv_q5_1_spv, "quant_kv_q5_1");
dyn_spv!(quant_kv_q5_1_f16_spv, "quant_kv_q5_1_f16");
dyn_spv!(quant_kv_iq4_nl_spv, "quant_kv_iq4_nl");
dyn_spv!(quant_kv_iq4_nl_f16_spv, "quant_kv_iq4_nl_f16");
dyn_spv!(dequant_kv_q4_0_spv, "dequant_kv_q4_0");
dyn_spv!(dequant_kv_q4_1_spv, "dequant_kv_q4_1");
dyn_spv!(dequant_kv_q5_0_spv, "dequant_kv_q5_0");
dyn_spv!(dequant_kv_q5_1_spv, "dequant_kv_q5_1");
dyn_spv!(dequant_kv_iq4_nl_spv, "dequant_kv_iq4_nl");
dyn_spv!(dequant_kv_bf16_spv, "dequant_kv_bf16");
// Dense KV cast-store (f32 / bf16 cache, from f16 K or f32 V).
dyn_spv!(store_kv_f32_spv, "store_kv_f32");
dyn_spv!(store_kv_f32_from_f16_spv, "store_kv_f32_from_f16");
dyn_spv!(store_kv_bf16_spv, "store_kv_bf16");
dyn_spv!(store_kv_bf16_from_f16_spv, "store_kv_bf16_from_f16");

// ── KV-cache u64/BDA store + dequant-read twins (#74 slice 3) ────────────────────────────────────
// The `-DKV_BDA` builds: store kernels write the DEST cache / dequant kernels read the SOURCE cache by
// 64-bit device address (kv_addr.glsl) instead of the bound KV SSBO. The store_f16 twin keeps its
// bound build (it ALSO writes a non-KV f16 scratch, the f32-cast dequant read); the qk_norm_rope*/
// rope_f16* twins keep theirs (un-fused Q-scratch write). The store_q8/quant_kv/store_kv_dense/
// dequant_q8_f16/dequant_kv_f16 kernels went BDA-only (slice 5): their base getters below now yield
// the (unconditional) BDA build and their bound twins/`*_bda` getters were deleted.
dyn_spv!(store_f16_bda_spv, "store_f16_bda");
dyn_spv!(store_f16_dyn_bda_spv, "store_f16_dyn_bda");
dyn_spv!(rope_f16_bda_spv, "rope_f16_bda");
dyn_spv!(rope_f16_dyn_bda_spv, "rope_f16_dyn_bda");
dyn_spv!(qk_norm_rope_bda_spv, "qk_norm_rope_bda");
dyn_spv!(qk_norm_rope_dyn_bda_spv, "qk_norm_rope_dyn_bda");
dyn_spv!(qk_norm_rope_ff_bda_spv, "qk_norm_rope_ff_bda");
dyn_spv!(qk_norm_rope_dyn_ff_bda_spv, "qk_norm_rope_dyn_ff_bda");
dyn_spv!(
    qk_norm_rope_interleaved_bda_spv,
    "qk_norm_rope_interleaved_bda"
);
dyn_spv!(
    qk_norm_rope_interleaved_dyn_bda_spv,
    "qk_norm_rope_interleaved_dyn_bda"
);

dyn_spv!(quant_turbo_t2_spv, "quant_turbo_t2");
dyn_spv!(quant_turbo_t2_f16_spv, "quant_turbo_t2_f16");
dyn_spv!(quant_turbo_t3_spv, "quant_turbo_t3");
dyn_spv!(quant_turbo_t3_f16_spv, "quant_turbo_t3_f16");
dyn_spv!(quant_turbo_t4_spv, "quant_turbo_t4");
dyn_spv!(quant_turbo_t4_f16_spv, "quant_turbo_t4_f16");
dyn_spv!(dequant_turbo_t2_spv, "dequant_turbo_t2");
dyn_spv!(dequant_turbo_t3_spv, "dequant_turbo_t3");
dyn_spv!(dequant_turbo_t4_spv, "dequant_turbo_t4");

/// (kernel name, SPIR-V) for the TurboQuant KV quantize of `dt` (`src_f16` = f16 K, else f32 V).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_turbo_kernel(
    dt: infr_core::DType,
    src_f16: bool,
) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match (dt, src_f16) {
        (Turbo2, false) => ("quant_turbo_t2", quant_turbo_t2_spv()),
        (Turbo2, true) => ("quant_turbo_t2_f16", quant_turbo_t2_f16_spv()),
        (Turbo3, false) => ("quant_turbo_t3", quant_turbo_t3_spv()),
        (Turbo3, true) => ("quant_turbo_t3_f16", quant_turbo_t3_f16_spv()),
        (Turbo4, false) => ("quant_turbo_t4", quant_turbo_t4_spv()),
        (Turbo4, true) => ("quant_turbo_t4_f16", quant_turbo_t4_f16_spv()),
        _ => unreachable!("quant_turbo_kernel for non-turbo dtype {dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the TurboQuant KV dequant→f16 of `dt`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dequant_turbo_kernel(dt: infr_core::DType) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match dt {
        Turbo2 => ("dequant_turbo_t2", dequant_turbo_t2_spv()),
        Turbo3 => ("dequant_turbo_t3", dequant_turbo_t3_spv()),
        Turbo4 => ("dequant_turbo_t4", dequant_turbo_t4_spv()),
        _ => unreachable!("dequant_turbo_kernel for non-turbo dtype {dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the dense KV cast-store into `dst_dt` (F32/Bf16), `src_f16` = f16 K.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn store_kv_dense_kernel(
    dst_dt: infr_core::DType,
    src_f16: bool,
) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match (dst_dt, src_f16) {
        (F32, false) => ("store_kv_f32", store_kv_f32_spv()),
        (F32, true) => ("store_kv_f32_from_f16", store_kv_f32_from_f16_spv()),
        (Bf16, false) => ("store_kv_bf16", store_kv_bf16_spv()),
        (Bf16, true) => ("store_kv_bf16_from_f16", store_kv_bf16_from_f16_spv()),
        _ => unreachable!("store_kv_dense_kernel for non-dense KV dtype {dst_dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the KV quantize kernel of `dt` (`src_f16` = f16 K source, else f32 V).
/// Distinct names per variant so the recorder's name-keyed pipeline cache never collides.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn quant_kv_kernel(
    dt: infr_core::DType,
    src_f16: bool,
) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match (dt, src_f16) {
        (Q4_0, false) => ("quant_kv_q4_0", quant_kv_q4_0_spv()),
        (Q4_0, true) => ("quant_kv_q4_0_f16", quant_kv_q4_0_f16_spv()),
        (Q4_1, false) => ("quant_kv_q4_1", quant_kv_q4_1_spv()),
        (Q4_1, true) => ("quant_kv_q4_1_f16", quant_kv_q4_1_f16_spv()),
        (Q5_0, false) => ("quant_kv_q5_0", quant_kv_q5_0_spv()),
        (Q5_0, true) => ("quant_kv_q5_0_f16", quant_kv_q5_0_f16_spv()),
        (Q5_1, false) => ("quant_kv_q5_1", quant_kv_q5_1_spv()),
        (Q5_1, true) => ("quant_kv_q5_1_f16", quant_kv_q5_1_f16_spv()),
        (Iq4Nl, false) => ("quant_kv_iq4_nl", quant_kv_iq4_nl_spv()),
        (Iq4Nl, true) => ("quant_kv_iq4_nl_f16", quant_kv_iq4_nl_f16_spv()),
        _ => unreachable!("quant_kv_kernel for non-KV-quant dtype {dt:?}"),
    }
}

/// (kernel name, SPIR-V) for the KV dequant→f16 prefix expander of `dt`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn dequant_kv_kernel(dt: infr_core::DType) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match dt {
        Q4_0 => ("dequant_kv_q4_0", dequant_kv_q4_0_spv()),
        Q4_1 => ("dequant_kv_q4_1", dequant_kv_q4_1_spv()),
        Q5_0 => ("dequant_kv_q5_0", dequant_kv_q5_0_spv()),
        Q5_1 => ("dequant_kv_q5_1", dequant_kv_q5_1_spv()),
        Iq4Nl => ("dequant_kv_iq4_nl", dequant_kv_iq4_nl_spv()),
        Bf16 => ("dequant_kv_bf16", dequant_kv_bf16_spv()),
        _ => unreachable!("dequant_kv_kernel for non-prepass KV dtype {dt:?}"),
    }
}
// Coupled Q8_0 KV: coalesced split-K decode partials reading Q8 blocks (static / dyn / self-chunk).
dyn_spv!(attn_partial_q8_spv, "attn_partial_q8");
dyn_spv!(attn_partial_kq8_spv, "attn_partial_kq8");
dyn_spv!(attn_partial_vq8_spv, "attn_partial_vq8");
dyn_spv!(attn_partial_dyn_q8_spv, "attn_partial_dyn_q8");
dyn_spv!(attn_partial_dynac_q8_spv, "attn_partial_dynac_q8");
// KV-cache u64/BDA twins (#74, slice 2): the `-DKV_BDA` builds of the flash-decoding split-K
// partial family — K/V read by 64-bit device address (kv_addr.glsl; wide KV2 b64 for the f16 vec4
// loads, scalar kv_word for Q8) instead of bound SSBOs. Bound twins above stay for kv_addr_parity.rs;
// production forks here via Recorder::attention_kv_split_at / _dyn_at / _dynac_at.
dyn_spv!(attn_partial_bda_spv, "attn_partial_bda");
dyn_spv!(attn_partial_nohd_bda_spv, "attn_partial_nohd_bda");
dyn_spv!(attn_partial_kq8_bda_spv, "attn_partial_kq8_bda");
dyn_spv!(attn_partial_vq8_bda_spv, "attn_partial_vq8_bda");
dyn_spv!(attn_partial_q8_bda_spv, "attn_partial_q8_bda");
dyn_spv!(attn_partial_dyn_bda_spv, "attn_partial_dyn_bda");
dyn_spv!(attn_partial_dyn_nohd_bda_spv, "attn_partial_dyn_nohd_bda");
dyn_spv!(attn_partial_dynac_bda_spv, "attn_partial_dynac_bda");
dyn_spv!(
    attn_partial_dynac_nohd_bda_spv,
    "attn_partial_dynac_nohd_bda"
);
dyn_spv!(attn_partial_dynac_q8_bda_spv, "attn_partial_dynac_q8_bda");
dyn_spv!(
    attn_partial_mrows_c256_bda_spv,
    "attn_partial_mrows_c256_bda"
);
// B7 slice 3a: DECODE-ONLY specialization of the split-K pass 1 (`attn_decode.comp`) — the same
// hd=128 f16-BDA inner loop `attn_partial_bda` runs, with every non-decode arm deleted so it fits
// 96 VGPRs instead of 120 and 2.75 KB of LDS instead of 4.75 KB. Bit-identical output (the
// reduction stays a plain `subgroupAdd`). Selected in `Recorder::attention_kv_split_impl` /
// `attention_kv_split_dynac_impl` when the shape qualifies; `INFR_NO_ATTN_DECODE=1`
// (`kernels.vulkan.attn_decode`) forces `attn_partial` back for the A/B.
dyn_spv!(attn_decode_spv, "attn_decode");
dyn_spv!(attn_decode_dynac_spv, "attn_decode_dynac");
// B7 slice 3b: the same kernel, widened by build-time specialization to the two exclusions that
// actually cost throughput — sliding-window layers (gemma3/gemma4 run SWA on most layers, and their
// SWA caches are RINGS: position j at row j % (cap/(nkv*hd))) and hd 256/512 (gemma4's per-layer
// head dims, qwen3.5). `-DSWA` always ships with `-DRING`, since the modulo is the identity on a
// full-context cache and one build then covers both.
dyn_spv!(attn_decode_swa_spv, "attn_decode_swa");
dyn_spv!(attn_decode_swa_dynac_spv, "attn_decode_swa_dynac");
dyn_spv!(attn_decode_hd256_spv, "attn_decode_hd256");
dyn_spv!(attn_decode_hd256_dynac_spv, "attn_decode_hd256_dynac");
dyn_spv!(attn_decode_hd256_swa_spv, "attn_decode_hd256_swa");
dyn_spv!(
    attn_decode_hd256_swa_dynac_spv,
    "attn_decode_hd256_swa_dynac"
);
dyn_spv!(attn_decode_hd512_spv, "attn_decode_hd512");
dyn_spv!(attn_decode_hd512_dynac_spv, "attn_decode_hd512_dynac");
dyn_spv!(attn_decode_hd512_swa_spv, "attn_decode_hd512_swa");
dyn_spv!(
    attn_decode_hd512_swa_dynac_spv,
    "attn_decode_hd512_swa_dynac"
);
/// `INFR_NO_ATTN_DECODE=1` (`kernels.vulkan.attn_decode` = false) — keep the general
/// `attn_partial` builds on the decode fast path so the specialization can be A/B'd against the
/// kernel it replaces with one env var and one binary.
pub(crate) fn attn_decode_disabled(vk: &infr_core::config::VulkanCfg) -> bool {
    !vk.attn_decode
}

/// The decode-only split-K pass-1 build for `(hd, swa, dynac)`, or `None` when no member of the
/// family covers the shape (the caller then keeps its general `attn_partial` selection).
///
/// `swa` is `window > 0`: those builds carry BOTH the window mask floor and the ring-row modulo,
/// so they are correct on a ring cache and on a full-context one. `hd` must be exactly 128, 256 or
/// 512 — every other head dim keeps `attn_partial`'s runtime `hd4` loops.
pub(crate) fn attn_decode_kernel(
    hd: usize,
    swa: bool,
    dynac: bool,
) -> Option<(&'static str, &'static [u32])> {
    Some(match (hd, swa, dynac) {
        (128, false, false) => ("attn_decode", attn_decode_spv()),
        (128, false, true) => ("attn_decode_dynac", attn_decode_dynac_spv()),
        (128, true, false) => ("attn_decode_swa", attn_decode_swa_spv()),
        (128, true, true) => ("attn_decode_swa_dynac", attn_decode_swa_dynac_spv()),
        (256, false, false) => ("attn_decode_hd256", attn_decode_hd256_spv()),
        (256, false, true) => ("attn_decode_hd256_dynac", attn_decode_hd256_dynac_spv()),
        (256, true, false) => ("attn_decode_hd256_swa", attn_decode_hd256_swa_spv()),
        (256, true, true) => (
            "attn_decode_hd256_swa_dynac",
            attn_decode_hd256_swa_dynac_spv(),
        ),
        (512, false, false) => ("attn_decode_hd512", attn_decode_hd512_spv()),
        (512, false, true) => ("attn_decode_hd512_dynac", attn_decode_hd512_dynac_spv()),
        (512, true, false) => ("attn_decode_hd512_swa", attn_decode_hd512_swa_spv()),
        (512, true, true) => (
            "attn_decode_hd512_swa_dynac",
            attn_decode_hd512_swa_dynac_spv(),
        ),
        _ => return None,
    })
}
// PROBE (B7 slice 1): LDS-staged K-tile decode pass 1 — one whole 128-dim dot per THREAD out of
// shared memory instead of attn_partial's per-key cross-lane `subgroupAdd`. Reachable ONLY from
// `Recorder::attention_kv_split_ktile_at`, whose only caller is tests/attn_ktile_probe.rs.
dyn_spv!(attn_ktile_w64_spv, "attn_ktile_w64");
dyn_spv!(attn_ktile_w64_nopad_spv, "attn_ktile_w64_nopad");
dyn_spv!(attn_ktile_w128_spv, "attn_ktile_w128");
dyn_spv!(attn_ktile_w64_dw32_spv, "attn_ktile_w64_dw32");
// PROBE (B7 slice 2): attn_partial's hd==128 QK loop with a build-time reduction WIDTH — w lanes
// per key via `subgroupClusteredAdd(., w)`, 32/w keys in flight per wave. Reachable ONLY from
// `Recorder::attention_kv_split_dsplit_at`, whose only caller is tests/attn_dsplit_probe.rs.
dyn_spv!(attn_dsplit_w1_spv, "attn_dsplit_w1");
dyn_spv!(attn_dsplit_w2_spv, "attn_dsplit_w2");
dyn_spv!(attn_dsplit_w4_spv, "attn_dsplit_w4");
dyn_spv!(attn_dsplit_w8_spv, "attn_dsplit_w8");
dyn_spv!(attn_dsplit_w16_spv, "attn_dsplit_w16");
dyn_spv!(attn_dsplit_w32_spv, "attn_dsplit_w32");
dyn_spv!(attn_dsplit_w1_wg128_spv, "attn_dsplit_w1_wg128");
dyn_spv!(attn_dsplit_w2_wg128_spv, "attn_dsplit_w2_wg128");
dyn_spv!(attn_dsplit_w4_wg128_spv, "attn_dsplit_w4_wg128");
dyn_spv!(attn_dsplit_w8_wg128_spv, "attn_dsplit_w8_wg128");
dyn_spv!(attn_dsplit_w16_wg128_spv, "attn_dsplit_w16_wg128");
dyn_spv!(attn_dsplit_w32_wg128_spv, "attn_dsplit_w32_wg128");
// Lever 1 (kv-decode-perf-levers #1): mainline low-bit KV decode reads the compact GGUF block
// format INLINE (native_decode `dq()` via `dq_addr`=k_addr/v_addr) instead of the dequant→f16
// prepass — the decode-side twin of dequant-in-flash. Per-format `-DKMAINLINE -DVMAINLINE -DFMT_*`
// (K and V share the cache dtype) with the f16 `_nohd` A/B twin. Selected by `attn_partial_ml_kernel`.
dyn_spv!(attn_partial_ml_q4_0_bda_spv, "attn_partial_ml_q4_0_bda");
dyn_spv!(
    attn_partial_ml_q4_0_nohd_bda_spv,
    "attn_partial_ml_q4_0_nohd_bda"
);
dyn_spv!(attn_partial_ml_q4_1_bda_spv, "attn_partial_ml_q4_1_bda");
dyn_spv!(
    attn_partial_ml_q4_1_nohd_bda_spv,
    "attn_partial_ml_q4_1_nohd_bda"
);
dyn_spv!(attn_partial_ml_q5_0_bda_spv, "attn_partial_ml_q5_0_bda");
dyn_spv!(
    attn_partial_ml_q5_0_nohd_bda_spv,
    "attn_partial_ml_q5_0_nohd_bda"
);
dyn_spv!(attn_partial_ml_q5_1_bda_spv, "attn_partial_ml_q5_1_bda");
dyn_spv!(
    attn_partial_ml_q5_1_nohd_bda_spv,
    "attn_partial_ml_q5_1_nohd_bda"
);

/// (kernel name, SPIR-V) for the mainline-INLINE split-K decode partial pass (`attn_partial`,
/// `-DKMAINLINE -DVMAINLINE -DFMT_* -DKV_BDA`) of the low-bit block quant `dt`. `nohd` selects the
/// `-DNO_HD_SPEC` A/B twin (INFR_NO_ATTN_HD). Covers the formats with a coalesced inline vec4 decoder
/// (q4_0/q4_1/q5_0/q5_1); iq4_nl (codebook gather) + other prepass dtypes keep the dequant→f16 path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_partial_ml_kernel(
    dt: infr_core::DType,
    nohd: bool,
) -> (&'static str, &'static [u32]) {
    use infr_core::DType::*;
    match (dt, nohd) {
        (Q4_0, false) => ("attn_partial_ml_q4_0_bda", attn_partial_ml_q4_0_bda_spv()),
        (Q4_0, true) => (
            "attn_partial_ml_q4_0_nohd_bda",
            attn_partial_ml_q4_0_nohd_bda_spv(),
        ),
        (Q4_1, false) => ("attn_partial_ml_q4_1_bda", attn_partial_ml_q4_1_bda_spv()),
        (Q4_1, true) => (
            "attn_partial_ml_q4_1_nohd_bda",
            attn_partial_ml_q4_1_nohd_bda_spv(),
        ),
        (Q5_0, false) => ("attn_partial_ml_q5_0_bda", attn_partial_ml_q5_0_bda_spv()),
        (Q5_0, true) => (
            "attn_partial_ml_q5_0_nohd_bda",
            attn_partial_ml_q5_0_nohd_bda_spv(),
        ),
        (Q5_1, false) => ("attn_partial_ml_q5_1_bda", attn_partial_ml_q5_1_bda_spv()),
        (Q5_1, true) => (
            "attn_partial_ml_q5_1_nohd_bda",
            attn_partial_ml_q5_1_nohd_bda_spv(),
        ),
        _ => unreachable!("attn_partial_ml_kernel for non-inline-decode KV dtype {dt:?}"),
    }
}
dyn_spv!(dequant_q8_f16_spv, "dequant_q8_f16");
/// SPIR-V for the SELF-CHUNKING record-once split-K decode partial (adaptive chunk from the live
/// kv_len; workgroups past the live range early-exit with a zero-weight header).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_partial_dynac_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(ATTN_PARTIAL_DYNAC_SPV_BYTES))
}
/// SPIR-V for the live-count combine (record-once replay; loops the prologue's live chunks).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_combine_live_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(ATTN_COMBINE_LIVE_SPV_BYTES))
}
/// SPIR-V for the split-K replay prologue (indirect args + live count from kv_len).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn attn_live_spv() -> &'static [u32] {
    static S: OnceLock<Vec<u32>> = OnceLock::new();
    S.get_or_init(|| spv_words(ATTN_LIVE_SPV_BYTES))
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// mul_mm-style warp-tiled coopmat GEMM `C[m,n]=A[m,k]·B[k,n]`. m,n %128, k %16.
    pub fn matmul_warp(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        assert!(m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(16));
        let kern = self.kernel_sg("gemm_warp", gemm_warp_spv(), 3, 12, 32);
        self.run_gemm(kern, a, b, m, k, n, (n / 128) as u32, (m / 128) as u32)
    }

    /// Tiled cooperative-matrix GEMM (shared-memory, 64x64 tiles): `C[m,n]=A[m,k]*B[k,n]`.
    /// f16 inputs, f32 output. v1 requires m,n multiples of 64 and k multiple of 32.
    pub fn matmul_f16(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        assert!(
            m.is_multiple_of(64) && n.is_multiple_of(64) && k.is_multiple_of(32),
            "tiled coopmat GEMM needs m,n %64 and k %32 (got {m},{k},{n})"
        );
        let kern = self.kernel_sg("gemm_coopmat_tiled", gemm_tiled_spv(), 3, 12, 32);
        self.run_gemm(kern, a, b, m, k, n, (n / 64) as u32, (m / 64) as u32)
    }

    /// Benchmark ONLY the tiled GEMM dispatch (weights pre-uploaded as f16; no host
    /// conversion / transfer in the loop). Returns avg seconds per dispatch.
    #[doc(hidden)]
    pub fn bench_tiled_gemm(&self, m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let kern = self.kernel_sg("gemm_coopmat_tiled", gemm_tiled_spv(), 3, 12, 32);
        let a16 = vec![0u16; m * k];
        let b16 = vec![0u16; k * n];
        let buf_a = self.alloc(a16.len() * 2, BufferUsage::Staging).unwrap();
        let buf_b = self.alloc(b16.len() * 2, BufferUsage::Staging).unwrap();
        let buf_c = self.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        self.upload(buf_a.as_ref(), bytemuck::cast_slice(&a16))
            .unwrap();
        self.upload(buf_b.as_ref(), bytemuck::cast_slice(&b16))
            .unwrap();

        // `expect` matches the surrounding `unwrap`s: this is a `#[doc(hidden)]` bench helper
        // returning `f64`, and it downcasts buffers IT just allocated three lines up, so a failure
        // here is impossible short of a refactor bug — and the bench harness has no error channel.
        let bufs = [
            as_vk_buf(buf_a.as_ref())
                .expect("bench buffer is a VkBuffer")
                .buffer,
            as_vk_buf(buf_b.as_ref())
                .expect("bench buffer is a VkBuffer")
                .buffer,
            as_vk_buf(buf_c.as_ref())
                .expect("bench buffer is a VkBuffer")
                .buffer,
        ];
        let binding = self.eager_bind(&kern, &bufs).unwrap();

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        let (gx, gy) = ((n / 64) as u32, (m / 64) as u32);

        let dispatch = || {
            let shared = &self.shared;
            self.one_shot(|cmd| unsafe {
                shared
                    .device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
                binding.bind(shared, cmd, kern.pipeline_layout);
                shared.device.cmd_push_constants(
                    cmd,
                    kern.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                shared.device.cmd_dispatch(cmd, gx, gy, 1);
            })
            .unwrap();
        };
        dispatch(); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            dispatch();
        }
        t.elapsed().as_secs_f64() / iters as f64
    }

    /// Benchmark the mul_mm-style warp-tiled GEMM (m,n %128, k %16). Returns avg sec/dispatch.
    #[doc(hidden)]
    pub fn bench_warp_gemm(&self, m: usize, k: usize, n: usize, iters: usize) -> f64 {
        let kern = self.kernel_sg("gemm_warp", gemm_warp_spv(), 3, 12, 32);
        let a16 = vec![0u16; m * k];
        let b16 = vec![0u16; k * n];
        let buf_a = self.alloc(a16.len() * 2, BufferUsage::Staging).unwrap();
        let buf_b = self.alloc(b16.len() * 2, BufferUsage::Staging).unwrap();
        let buf_c = self.alloc(m * n * 4, BufferUsage::Activations).unwrap();
        self.upload(buf_a.as_ref(), bytemuck::cast_slice(&a16))
            .unwrap();
        self.upload(buf_b.as_ref(), bytemuck::cast_slice(&b16))
            .unwrap();
        // `expect` matches the surrounding `unwrap`s: this is a `#[doc(hidden)]` bench helper
        // returning `f64`, and it downcasts buffers IT just allocated three lines up, so a failure
        // here is impossible short of a refactor bug — and the bench harness has no error channel.
        let bufs = [
            as_vk_buf(buf_a.as_ref())
                .expect("bench buffer is a VkBuffer")
                .buffer,
            as_vk_buf(buf_b.as_ref())
                .expect("bench buffer is a VkBuffer")
                .buffer,
            as_vk_buf(buf_c.as_ref())
                .expect("bench buffer is a VkBuffer")
                .buffer,
        ];
        let binding = self.eager_bind(&kern, &bufs).unwrap();
        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        let (gx, gy) = ((n / 128) as u32, (m / 128) as u32);
        let dispatch = || {
            let shared = &self.shared;
            self.one_shot(|cmd| unsafe {
                shared
                    .device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
                binding.bind(shared, cmd, kern.pipeline_layout);
                shared.device.cmd_push_constants(
                    cmd,
                    kern.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                shared.device.cmd_dispatch(cmd, gx, gy, 1);
            })
            .unwrap();
        };
        dispatch(); // warm
        let t = std::time::Instant::now();
        for _ in 0..iters {
            dispatch();
        }
        t.elapsed().as_secs_f64() / iters as f64
    }

    fn run_gemm(
        &self,
        kern: super::ops::ComputeKernel,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        gx: u32,
        gy: u32,
    ) -> Result<Vec<f32>> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);

        let a16: Vec<u16> = a.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let b16: Vec<u16> = b.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let buf_a = self.alloc(a16.len() * 2, BufferUsage::Staging)?;
        let buf_b = self.alloc(b16.len() * 2, BufferUsage::Staging)?;
        let buf_c = self.alloc(m * n * 4, BufferUsage::Readback)?;
        self.upload(buf_a.as_ref(), bytemuck::cast_slice(&a16))?;
        self.upload(buf_b.as_ref(), bytemuck::cast_slice(&b16))?;

        let bufs = [
            as_vk_buf(buf_a.as_ref())?.buffer,
            as_vk_buf(buf_b.as_ref())?.buffer,
            as_vk_buf(buf_c.as_ref())?.buffer,
        ];
        let binding = self.eager_bind(&kern, &bufs)?;

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_le_bytes());

        let shared = &self.shared;
        self.one_shot(|cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
            binding.bind(shared, cmd, kern.pipeline_layout);
            shared.device.cmd_push_constants(
                cmd,
                kern.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            shared.device.cmd_dispatch(cmd, gx, gy, 1);
        })?;

        let mut c_bytes = vec![0u8; m * n * 4];
        self.download(buf_c.as_ref(), &mut c_bytes)?;
        Ok(bytemuck::cast_slice(&c_bytes).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B7 slice 3a — the decode specialization's reduction must be a PLAIN subgroup reduce, never
    /// a clustered one. `attn_partial_dsplit.comp` (the width probe) spells the same width-32
    /// reduction as `subgroupClusteredAdd(., 32)`, and glslang emits `ClusteredReduce` for it;
    /// ACO then builds a different reduction tree and the output drifts ~1e-6 from
    /// `attn_partial`'s — enough to need every `gpu_seam_matches_cpu*` golden re-blessed.
    /// `attn_decode.comp` must stay on `subgroupAdd` so the bits match exactly.
    ///
    /// Checked in the SPIR-V rather than trusted from the source, because the two spellings are a
    /// one-word edit apart and nothing else would notice: `OpGroupNonUniformFAdd` (opcode 350)
    /// carries its Operation right after the scope <id> — 0 = Reduce, 3 = ClusteredReduce.
    #[test]
    fn attn_decode_reduction_is_not_clustered() {
        const OP_GROUP_NON_UNIFORM_FADD: u16 = 350;
        const REDUCE: u32 = 0;
        // A SPIR-V module is a 5-word header then a stream of instructions; word 0 of each is
        // (wordcount << 16) | opcode.
        fn reduce_ops(spv: &[u32]) -> Vec<u32> {
            let mut ops = Vec::new();
            let mut i = 5usize;
            while i < spv.len() {
                let wc = (spv[i] >> 16) as usize;
                assert!(
                    wc > 0 && i + wc <= spv.len(),
                    "malformed SPIR-V at word {i}"
                );
                if (spv[i] & 0xffff) as u16 == OP_GROUP_NON_UNIFORM_FADD {
                    // [0] opcode, [1] result type, [2] result id, [3] scope, [4] Operation
                    ops.push(spv[i + 4]);
                }
                i += wc;
            }
            ops
        }
        let mut family: Vec<(&'static str, &'static [u32])> = Vec::new();
        for hd in [128usize, 256, 512] {
            for swa in [false, true] {
                for dynac in [false, true] {
                    family.push(attn_decode_kernel(hd, swa, dynac).expect("family member"));
                }
            }
        }
        assert_eq!(family.len(), 12, "the attn_decode family lost a member");
        // The kernels being matched, so the invariant is pinned to a real reference rather
        // than to a remembered fact about them.
        family.push(("attn_partial_bda", attn_partial_bda_spv()));
        family.push(("attn_partial_dynac_bda", attn_partial_dynac_bda_spv()));
        for (name, spv) in family {
            let ops = reduce_ops(spv);
            assert!(
                !ops.is_empty(),
                "{name}: found no OpGroupNonUniformFAdd at all — this check matched nothing"
            );
            assert!(
                ops.iter().all(|&o| o == REDUCE),
                "{name}: subgroup float add is not a plain Reduce (operations {ops:?}); \
                 3 = ClusteredReduce, which changes the reduction tree and breaks bit-identity"
            );
        }
    }

    fn cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn check(got: &[f32], want: &[f32], label: &str) {
        let mut max_rel = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
        }
        println!("{label} max_rel_err = {max_rel:.4e}");
        assert!(max_rel < 2e-2, "{label} rel err {max_rel} too high");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU with cooperative matrix"]
    fn coopmat_gemm_tiled_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (128usize, 96usize, 64usize); // m,n %64, k %32
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        let got = be.matmul_f16(&a, &b, m, k, n).unwrap();
        check(&got, &cpu(&a, &b, m, k, n), "tiled");
    }

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn warp_gemm_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        for &(m, k, n) in &[
            (128usize, 16usize, 128usize),
            (256, 128, 256),
            (128, 512, 128),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
            let got = be.matmul_warp(&a, &b, m, k, n).unwrap();
            check(&got, &cpu(&a, &b, m, k, n), "warp");
        }
    }

    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn coopmat_gemm_bench() {
        let be = VulkanBackend::new().unwrap();
        for s in [1024usize, 2048, 4096] {
            let dt = be.bench_tiled_gemm(s, s, s, 20);
            let flops = 2.0 * (s as f64).powi(3);
            println!(
                "tiled coopmat GEMM {s}^3: {:.3} ms, {:.0} GFLOP/s",
                dt * 1e3,
                flops / dt / 1e9
            );
        }
        // Attention shapes (per head, 32k ctx): QK=[512,128]·[128,32768], PV=[512,32768]·[32768,128]
        for &(m, k, n, label) in &[
            (512usize, 128usize, 32768usize, "QK m512 k128 n32k"),
            (512, 32768, 128, "PV m512 k32k n128"),
        ] {
            let dt = be.bench_tiled_gemm(m, k, n, 20);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "tiled coopmat GEMM {label}: {:.3} ms, {:.0} GFLOP/s",
                dt * 1e3,
                flops / dt / 1e9
            );
        }
        // mul_mm-style warp-tiled GEMM at the same shapes (m,n %128, k %16)
        for &(m, k, n, label) in &[
            (2048usize, 2048usize, 2048usize, "warp 2048^3"),
            (512, 128, 32768, "warp QK m512 k128 n32k"),
            (512, 32768, 128, "warp PV m512 k32k n128"),
        ] {
            let dt = be.bench_warp_gemm(m, k, n, 20);
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            println!(
                "{label}: {:.3} ms, {:.0} GFLOP/s",
                dt * 1e3,
                flops / dt / 1e9
            );
        }
    }
}
