//! Persistent-weight upload helpers (`upload_weight*`) and the native-block/id-indexed decode-GEMV
//! kernel-name tables. Weights are addressed exclusively by 64-bit device address (resident-BDA,
//! see [`crate::VulkanBackend::alloc_arena_bda`] / `Backing::BdaSub`) — every `BufferUsage::Weights`
//! allocation lands in that arena, and the actual dispatch (streamed, arena-addressed reads) lives
//! on [`crate::Recorder`], not here.
//!
//! Build-compiled GLSL → SPIR-V (see build.rs / shaders/).

use infr_core::{
    backend::{Buffer, BufferUsage},
    error::Result,
    Backend,
};

use super::VulkanBackend;

/// Unified quant dequant GEMV with fused residual add: `y = residual + x·Wᵀ`.
// ─── Native-block dequant GEMV shaders (Phase 0-2) ─────────────────────────
//
// Each shader reads raw GGUF block bytes (uploaded padded to a u32-multiple)
// from `w_buf: array<u32>` and dequantizes elements in-shader. The outer GEMV
// cooperative-over-K structure matches LINEAR_F16_WGSL: one workgroup per
// output element, 64 threads stride K, tree-reduce.
//
/// Return the static kernel name for a native-block GEMV (Phase 0-2).
/// Kernel cache name for the id-indexed native GEMV; `None` only for non-weight dtypes. Covers the
/// FULL dense native-GEMV format set (affine quants, codebook/grid i-quants, fp4, ternary, bf16)
/// plus F16/F32 for float expert banks — resident float banks arrive as effective f16 (the seam's
/// `bind_weight` converts and reports the effective dtype), while PAGED float banks stage raw GGUF
/// bytes into the arena, so both float variants exist.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_id_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    // Delegate to the SPIR-V source of truth so the recorder's `is_some()` gate and its
    // `native_id_build_spv().expect()` load can never disagree — a name-table-only dtype used to be
    // a mid-inference panic (see AUDIT #1). The name is `build_spv`'s first tuple field.
    crate::gemm::native_id_build_spv(dtype).map(|(name, _)| name)
}

/// Kernel cache name for the multi-slot id-indexed native GEMV; `None` for formats without it.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_idm_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    // Delegate to the SPIR-V source of truth so the gate and the loaded shader can never drift
    // (see AUDIT #1 and [`native_id_kernel_name`]).
    crate::gemm::native_idm_build_spv(dtype).map(|(name, _)| name)
}

/// Whether the Vulkan MoE expert paths can dispatch a bank of this dtype AT ALL — the id-indexed
/// GEMV kernels are the floor every MoE model needs (decode + the per-token fallback). A dtype
/// missing here would `expect`-panic mid-inference in `linear_native_id(_multi)`. Since the id
/// family reached dense parity (every dense-GEMV format + F16/F32 for float banks) this is true
/// for EVERY dtype a GGUF expert bank can hold — `moe_expert_floor_covers_dense_set` (this
/// module's tests) pins that invariant, which let the seam's old load-time reject go (field
/// report: an MXFP4_MOE quant panicked mid-inference before that gate existed; the gate then
/// clean-rejected MXFP4 until the id family covered everything and the gate went dead).
pub fn moe_expert_dtype_ok(dtype: infr_core::DType) -> bool {
    native_id_kernel_name(dtype).is_some() && native_idm_kernel_name(dtype).is_some()
}

/// [`native_id_kernel_name`]'s paged twin (`infr_vulkan::pager::GpuPager` build — one extra LUT
/// hop, `w_addr = arena_base + uint64_t(lut[expert_id]) * slot_bytes` (a resident slot index
/// scaled onto the arena's 64-bit device address), see `shaders/native_gemv_id.comp`'s `-DPAGED`
/// doc comment).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_id_paged_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q8_0 => "native_id_q8_0_paged",
        Q4_0 => "native_id_q4_0_paged",
        Q4_1 => "native_id_q4_1_paged",
        Q5_0 => "native_id_q5_0_paged",
        Q5_1 => "native_id_q5_1_paged",
        Q2K => "native_id_q2k_paged",
        Q3K => "native_id_q3k_paged",
        Q4K => "native_id_q4k_paged",
        Q5K => "native_id_q5k_paged",
        Q6K => "native_id_q6k_paged",
        Iq4Nl => "native_id_iq4nl_paged",
        Iq4Xs => "native_id_iq4xs_paged",
        Mxfp4 => "native_id_mxfp4_paged",
        Nvfp4 => "native_id_nvfp4_paged",
        Tq1_0 => "native_id_tq1_0_paged",
        Tq2_0 => "native_id_tq2_0_paged",
        Q2_0 => "native_id_q2_0_paged",
        Iq2Xxs => "native_id_iq2xxs_paged",
        Iq2Xs => "native_id_iq2xs_paged",
        Iq2S => "native_id_iq2s_paged",
        Iq3Xxs => "native_id_iq3xxs_paged",
        Iq3S => "native_id_iq3s_paged",
        Iq1S => "native_id_iq1s_paged",
        Iq1M => "native_id_iq1m_paged",
        Bf16 => "native_id_bf16_paged",
        F16 => "native_id_f16_paged",
        F32 => "native_id_f32_paged",
        _ => return None,
    })
}

/// [`native_idm_kernel_name`]'s paged twin — same LUT hop, for the decode/small-m multi-expert
/// dispatch.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_idm_paged_kernel_name(dtype: infr_core::DType) -> Option<&'static str> {
    use infr_core::DType::*;
    Some(match dtype {
        Q8_0 => "native_idm_q8_0_paged",
        Q4_0 => "native_idm_q4_0_paged",
        Q4_1 => "native_idm_q4_1_paged",
        Q5_0 => "native_idm_q5_0_paged",
        Q5_1 => "native_idm_q5_1_paged",
        Q2K => "native_idm_q2k_paged",
        Q3K => "native_idm_q3k_paged",
        Q4K => "native_idm_q4k_paged",
        Q5K => "native_idm_q5k_paged",
        Q6K => "native_idm_q6k_paged",
        Iq4Nl => "native_idm_iq4nl_paged",
        Iq4Xs => "native_idm_iq4xs_paged",
        Mxfp4 => "native_idm_mxfp4_paged",
        Nvfp4 => "native_idm_nvfp4_paged",
        Tq1_0 => "native_idm_tq1_0_paged",
        Tq2_0 => "native_idm_tq2_0_paged",
        Q2_0 => "native_idm_q2_0_paged",
        Iq2Xxs => "native_idm_iq2xxs_paged",
        Iq2Xs => "native_idm_iq2xs_paged",
        Iq2S => "native_idm_iq2s_paged",
        Iq3Xxs => "native_idm_iq3xxs_paged",
        Iq3S => "native_idm_iq3s_paged",
        Iq1S => "native_idm_iq1s_paged",
        Iq1M => "native_idm_iq1m_paged",
        Bf16 => "native_idm_bf16_paged",
        F16 => "native_idm_f16_paged",
        F32 => "native_idm_f32_paged",
        _ => return None,
    })
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_kernel_name(dtype: infr_core::DType, residual: bool) -> &'static str {
    use infr_core::DType::*;
    match (dtype, residual) {
        (Q8_0, false) => "native_q8_0",
        (Q8_0, true) => "native_q8_0_res",
        (Bf16, false) => "native_bf16",
        (Bf16, true) => "native_bf16_res",
        (Q4_0, false) => "native_q4_0",
        (Q4_0, true) => "native_q4_0_res",
        (Q4_1, false) => "native_q4_1",
        (Q4_1, true) => "native_q4_1_res",
        (Q5_0, false) => "native_q5_0",
        (Q5_0, true) => "native_q5_0_res",
        (Q5_1, false) => "native_q5_1",
        (Q5_1, true) => "native_q5_1_res",
        (Q2K, false) => "native_q2k",
        (Q2K, true) => "native_q2k_res",
        (Q3K, false) => "native_q3k",
        (Q3K, true) => "native_q3k_res",
        (Q4K, false) => "native_q4k",
        (Q4K, true) => "native_q4k_res",
        (Q5K, false) => "native_q5k",
        (Q5K, true) => "native_q5k_res",
        (Q6K, false) => "native_q6k",
        (Q6K, true) => "native_q6k_res",
        (Iq4Nl, false) => "native_iq4nl",
        (Iq4Nl, true) => "native_iq4nl_res",
        (Iq4Xs, false) => "native_iq4xs",
        (Iq4Xs, true) => "native_iq4xs_res",
        (Mxfp4, false) => "native_mxfp4",
        (Mxfp4, true) => "native_mxfp4_res",
        (Nvfp4, false) => "native_nvfp4",
        (Nvfp4, true) => "native_nvfp4_res",
        (Tq1_0, false) => "native_tq1_0",
        (Tq1_0, true) => "native_tq1_0_res",
        (Tq2_0, false) => "native_tq2_0",
        (Tq2_0, true) => "native_tq2_0_res",
        (Q2_0, false) => "native_q2_0",
        (Q2_0, true) => "native_q2_0_res",
        (Iq2Xxs, false) => "native_iq2xxs",
        (Iq2Xxs, true) => "native_iq2xxs_res",
        (Iq2Xs, false) => "native_iq2xs",
        (Iq2Xs, true) => "native_iq2xs_res",
        (Iq2S, false) => "native_iq2s",
        (Iq2S, true) => "native_iq2s_res",
        (Iq3Xxs, false) => "native_iq3xxs",
        (Iq3Xxs, true) => "native_iq3xxs_res",
        (Iq3S, false) => "native_iq3s",
        (Iq3S, true) => "native_iq3s_res",
        (Iq1S, false) => "native_iq1s",
        (Iq1S, true) => "native_iq1s_res",
        (Iq1M, false) => "native_iq1m",
        (Iq1M, true) => "native_iq1m_res",
        _ => panic!("no native GEMV for {:?}", dtype),
    }
}

/// Kernel-cache key for the native-block prefill GEMM (one coopmat pipeline per quant format).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_gemm_kernel_name(dtype: infr_core::DType) -> &'static str {
    use infr_core::DType::*;
    match dtype {
        Q8_0 => "native_gemm_q8_0",
        Bf16 => "native_gemm_bf16",
        Q4_0 => "native_gemm_q4_0",
        Q4_1 => "native_gemm_q4_1",
        Q5_0 => "native_gemm_q5_0",
        Q5_1 => "native_gemm_q5_1",
        Q2K => "native_gemm_q2k",
        Q3K => "native_gemm_q3k",
        Q4K => "native_gemm_q4k",
        Q5K => "native_gemm_q5k",
        Q6K => "native_gemm_q6k",
        Iq4Nl => "native_gemm_iq4nl",
        Iq4Xs => "native_gemm_iq4xs",
        Mxfp4 => "native_gemm_mxfp4",
        Nvfp4 => "native_gemm_nvfp4",
        Tq1_0 => "native_gemm_tq1_0",
        Tq2_0 => "native_gemm_tq2_0",
        Q2_0 => "native_gemm_q2_0",
        Iq2Xxs => "native_gemm_iq2xxs",
        Iq2Xs => "native_gemm_iq2xs",
        Iq2S => "native_gemm_iq2s",
        Iq3Xxs => "native_gemm_iq3xxs",
        Iq3S => "native_gemm_iq3s",
        Iq1S => "native_gemm_iq1s",
        Iq1M => "native_gemm_iq1m",
        _ => panic!("no native GEMM for {:?}", dtype),
    }
}

/// True if `dtype` has the full dense native-block pipeline — a decode GEMV (`native_*`, see
/// [`native_kernel_name`]) AND a prefill coopmat GEMM (`native_gemm_*`, see
/// [`native_gemm_kernel_name`]). When true, the weight can be uploaded as raw GGUF block bytes and
/// run on the GPU with in-shader dequant — no host dequant → f16. Covers every quant format
/// (affine k-quants, legacy round, codebook i-quants, fp4, ternary, and grid i-quants). Float types
/// (F16/F32/BF16) are not quants and stay on the plain f16 GEMV.
///
/// The MoE *stacked/id-indexed* path (`native_id_*`/`native_idm_*`) covers this whole set PLUS
/// F16/F32 (float expert banks); use [`native_id_kernel_name`] for that.
/// Formats the `embed_gather` kernel family covers (`Op::EmbedGather` — see
/// `gemm::embed_gather_kernel_name`). The runner gates the token-ids input path on this.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn embed_gather_supported(dtype: infr_core::DType) -> bool {
    crate::gemm::embed_gather_kernel_name(dtype).is_some()
}

/// The canonical set of dtypes with the full dense native-block pipeline — the SINGLE SOURCE for
/// [`native_dense_supported`] (a `.contains` over this) and, crucially, the iteration set the MoE-
/// floor drift guard (`moe_expert_floor_covers_dense_set`) walks. Adding a format here therefore
/// enrolls it in that guard automatically — a dtype can no longer become dense-supported yet slip
/// past the "does the MoE id-GEMV floor cover it?" check (AUDIT #7).
pub fn native_dense_dtypes() -> &'static [infr_core::DType] {
    use infr_core::DType::*;
    &[
        Bf16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2K, Q3K, Q4K, Q5K, Q6K, Iq4Nl, Iq4Xs, Mxfp4, Nvfp4,
        Tq1_0, Tq2_0, Q2_0, Iq2Xxs, Iq2Xs, Iq2S, Iq3Xxs, Iq3S, Iq1S, Iq1M,
    ]
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn native_dense_supported(dtype: infr_core::DType) -> bool {
    native_dense_dtypes().contains(&dtype)
}

/// Pad raw GGUF block bytes to the next multiple of 4 for upload as `array<u32>`.
/// Appends zero bytes; the final u32 word's padding bytes are never read (they
/// contain only out-of-block data which the shader never accesses for valid g).
///
/// Returns a `Cow` so the common case is ZERO-COPY: nearly every GGUF tensor's byte length is
/// already a multiple of 4 (block sizes 18/20/34/144/210… × a block count), so padding is a no-op
/// and we can hand the caller the mmap slice straight through. This used to unconditionally
/// `to_vec()` every tensor — a full host copy of the entire model (~9 GiB on Qwen3-14B, ~1.26s of
/// pure memcpy + allocation) purely to append zero bytes that, in the overwhelming majority of
/// cases, did not exist.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn pad_to_u32_align(bytes: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    let padded = (bytes.len() + 3) & !3;
    if padded == bytes.len() {
        return std::borrow::Cow::Borrowed(bytes);
    }
    let mut v = bytes.to_vec();
    v.resize(padded, 0u8);
    std::borrow::Cow::Owned(v)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl VulkanBackend {
    /// Upload an `[out, in]` f32 weight to a persistent device buffer.
    pub fn upload_weight(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buf = self.alloc(bytes.len(), BufferUsage::Weights)?;
        self.upload(buf.as_ref(), bytes)?;
        Ok(buf)
    }

    /// Upload an `[out, in]` weight as f16 (halves device bandwidth for the GEMV/matmul kernels
    /// that read weights). Source stays f32; converted on the host.
    pub fn upload_weight_f16(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let f16: Vec<u16> = data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        self.upload_weight_bytes(bytemuck::cast_slice(&f16))
    }

    /// Upload an `[out, in]` weight as bf16 (truncate-round of f32; bf16 is the top 16 bits of f32).
    /// Read back losslessly to f32 in-shader by `LINEAR_BF16_WGSL`. Preserves f32's exponent range
    /// (unlike f16), so it's the correct GPU storage for bf16-source tensors that would overflow f16.
    pub fn upload_weight_bf16(&self, data: &[f32]) -> Result<Box<dyn Buffer>> {
        let bf16: Vec<u16> = data
            .iter()
            .map(|&x| {
                // round-to-nearest-even on the f32→bf16 truncation
                let bits = x.to_bits();
                let round = 0x7fffu32 + ((bits >> 16) & 1);
                ((bits.wrapping_add(round)) >> 16) as u16
            })
            .collect();
        self.upload_weight_bytes(bytemuck::cast_slice(&bf16))
    }

    /// Upload raw weight bytes (already in the target dtype) to a persistent device buffer.
    /// Use for f16 GGUF tensors to skip the f16→f32→f16 round-trip.
    pub fn upload_weight_bytes(&self, bytes: &[u8]) -> Result<Box<dyn Buffer>> {
        let buf = self.alloc(bytes.len(), BufferUsage::Weights)?;
        self.upload(buf.as_ref(), bytes)?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The drift guard the task's step 4 asks for: every dtype `infr_core::tensor::MOE_MMQ_DTYPES`
    /// lists (the batched-MoE dp4a mmq expert-GEMM family's SINGLE SOURCE OF TRUTH — see its doc)
    /// must ALSO have small-m decode coverage — id-GEMV (`native_id_kernel_name`), its multi-slot
    /// twin (`native_idm_kernel_name`), and BOTH of their paged twins. Pure-function, no GPU: these
    /// are `&'static str` lookups, not device calls. Forgetting to wire a newly-added mmq format
    /// into the id-GEMV families (this test's whole reason to exist — IQ4_NL/IQ4_XS were missing
    /// here until this change) used to only surface as a silent decode-perf regression (GPU-
    /// resident MoE decode falling back to the host top-k path for that format), not a build/test
    /// failure — this test turns that into an immediate, CI-visible failure instead.
    /// Pins the invariant `moe_expert_dtype_ok`'s doc promises (and the reason the seam's MoE
    /// load-time dtype gate could be removed): EVERY dtype the dense native path supports — plus
    /// F16/F32, the float-bank forms — has the complete id-GEMV floor: id, idm, and both paged
    /// twins. Pure name-table lookups, no GPU.
    #[test]
    fn moe_expert_floor_covers_dense_set() {
        use infr_core::DType::{F16, F32};
        // Derive the iteration set from `native_dense_dtypes` (the SINGLE SOURCE behind
        // `native_dense_supported`) plus the two float-bank forms — a newly dense-supported dtype is
        // now covered here automatically, it can't escape the guard by being absent from a
        // hand-written list (AUDIT #7).
        let all = native_dense_dtypes()
            .iter()
            .copied()
            .chain([F16, F32])
            .collect::<Vec<_>>();
        for &d in &all {
            assert!(
                moe_expert_dtype_ok(d),
                "{d:?} is dense-supported but the MoE expert floor rejects it"
            );
            assert!(
                native_id_paged_kernel_name(d).is_some()
                    && native_idm_paged_kernel_name(d).is_some(),
                "{d:?} has resident id kernels but no paged twins"
            );
        }
    }

    /// Unsloth's Qwen3.8-Flash-Next UD-Q2_K_XL is a mixed recipe rather than a new GGUF dtype:
    /// most expert gate/up banks are IQ2_XS, one pair is IQ3_XXS, and down banks are IQ4_NL.
    /// qwen4exp v1 uses the small-m path for both decode and its intentionally per-token prefill,
    /// so all three must retain resident and paged single-/multi-id coverage. The remaining fixed
    /// quantized/BF16 fixed weights in that file use the dense-native formats listed below (its
    /// small F32 tensors keep the established loader conversion/specialized paths).
    #[test]
    fn qwen38_q2_k_xl_recipe_has_complete_decode_coverage() {
        use infr_core::DType::{Bf16, Iq2Xs, Iq3Xxs, Iq4Nl, Q4K, Q5K, Q6K, Q8_0};

        for dtype in [Iq2Xs, Iq3Xxs, Iq4Nl] {
            assert!(
                moe_expert_dtype_ok(dtype),
                "{dtype:?} lacks resident id/idm coverage"
            );
            assert!(
                native_id_paged_kernel_name(dtype).is_some(),
                "{dtype:?} lacks paged id coverage"
            );
            assert!(
                native_idm_paged_kernel_name(dtype).is_some(),
                "{dtype:?} lacks paged idm coverage"
            );
        }

        for dtype in [Bf16, Q4K, Q5K, Q6K, Q8_0, Iq4Nl] {
            assert!(
                native_dense_supported(dtype),
                "Q2_K_XL fixed-weight dtype {dtype:?} lacks a native dense path"
            );
        }
    }

    /// Canonical enumeration of every `DType` variant, used by the drift guards below to mean
    /// literally "for EVERY dtype". [`all_dtypes_is_exhaustive`] pins it exhaustive: adding a variant
    /// to the enum breaks that test's compile until it's listed here, so this can't silently omit a
    /// new dtype.
    const ALL_DTYPES: &[infr_core::DType] = {
        use infr_core::DType::*;
        &[
            F32, F16, Bf16, I32, U32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, Iq1S,
            Iq1M, Iq2Xxs, Iq2Xs, Iq2S, Iq3Xxs, Iq3S, Iq4Nl, Iq4Xs, Tq1_0, Tq2_0, Q2_0, Mxfp4,
            Nvfp4, I2S, Turbo2, Turbo3, Turbo4,
        ]
    };

    #[test]
    fn all_dtypes_is_exhaustive() {
        use infr_core::DType::*;
        // Exhaustive match — a new DType variant makes this fail to COMPILE until both this arm and
        // ALL_DTYPES are updated, keeping ALL_DTYPES a true canonical enumeration.
        fn _covered(d: infr_core::DType) {
            match d {
                F32 | F16 | Bf16 | I32 | U32 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2K | Q3K
                | Q4K | Q5K | Q6K | Iq1S | Iq1M | Iq2Xxs | Iq2Xs | Iq2S | Iq3Xxs | Iq3S | Iq4Nl
                // I2S (BitNet i2_s): host-converted to f16 in the runner's wload, so it never
                // reaches the Vulkan backend as I2S — no native kernel (all *_kernel_name/spv gates
                // return None), correctly excluded from native_dense_dtypes.
                | Iq4Xs | Tq1_0 | Tq2_0 | Q2_0 | Mxfp4 | Nvfp4 | I2S | Turbo2 | Turbo3 | Turbo4 => {}
            }
        }
        // Count sanity: the arm above and ALL_DTYPES must list the same number of variants.
        assert_eq!(ALL_DTYPES.len(), 33);
    }

    /// AUDIT #1 drift guard: for EVERY `DType`, each `*_kernel_name` gate the recorder tests with
    /// `is_some()` returns EXACTLY the name its `*_build_spv` twin loads with `expect()`. The
    /// delegation makes this true by construction; the test pins it so a future edit that re-forks
    /// the tables (or a paged twin that goes out of sync) fails in CI, not mid-inference. Pure
    /// name/`&[u32]`-pointer lookups — no GPU.
    #[test]
    fn kernel_name_gate_matches_spv_source() {
        use crate::gemm;
        for &d in ALL_DTYPES {
            // id / idm families (name table in this module, spv in gemm).
            assert_eq!(
                native_id_kernel_name(d),
                gemm::native_id_build_spv(d).map(|(n, _)| n),
                "native_id gate vs spv disagree for {d:?}"
            );
            assert_eq!(
                native_idm_kernel_name(d),
                gemm::native_idm_build_spv(d).map(|(n, _)| n),
                "native_idm gate vs spv disagree for {d:?}"
            );
            // Paged twins: the gate returns a name, the spv returns words — so pin availability
            // parity (the recorder gates on the name, loads the paged spv).
            assert_eq!(
                native_id_paged_kernel_name(d).is_some(),
                gemm::native_id_paged_build_spv(d).is_some(),
                "native_id_paged gate vs spv availability disagree for {d:?}"
            );
            assert_eq!(
                native_idm_paged_kernel_name(d).is_some(),
                gemm::native_idm_paged_build_spv(d).is_some(),
                "native_idm_paged gate vs spv availability disagree for {d:?}"
            );
            // embed_gather family (both in gemm).
            assert_eq!(
                gemm::embed_gather_kernel_name(d),
                gemm::embed_gather_spv(d).map(|(n, _)| n),
                "embed_gather gate vs spv disagree for {d:?}"
            );
            // mmv families.
            for res in [false, true] {
                assert_eq!(
                    gemm::native_mmv_kernel_name(d, res),
                    gemm::native_mmv_spv(d, res).map(|(n, _)| n),
                    "native_mmv gate vs spv disagree for {d:?} res={res}"
                );
            }
            assert_eq!(
                gemm::native_mmv_mrow_kernel_name(d),
                gemm::native_mmv_mrow_variant_spv(d, false, false, false).map(|(n, _)| n),
                "native_mmv_mrow gate vs spv disagree for {d:?}"
            );
            assert_eq!(
                gemm::native_mmv_mrow_m16_kernel_name(d),
                gemm::native_mmv_mrow_m16_spv(d).map(|(n, _)| n),
                "native_mmv_mrow_m16 gate vs spv disagree for {d:?}"
            );
            for o4 in [false, true] {
                for m4 in [false, true] {
                    for res in [false, true] {
                        assert_eq!(
                            gemm::native_mmv_mrow_variant_kernel_name(d, o4, m4, res),
                            gemm::native_mmv_mrow_variant_spv(d, o4, m4, res).map(|(n, _)| n),
                            "native_mmv_mrow_variant gate vs spv disagree {d:?} o4={o4} m4={m4} res={res}"
                        );
                    }
                }
            }
            for &warps in &[1u32, 2, 4, 8, 16] {
                for res in [false, true] {
                    for sg16 in [false, true] {
                        assert_eq!(
                            gemm::native_mmv_mw_kernel_name(d, res, warps, sg16),
                            gemm::native_mmv_mw_spv(d, res, warps, sg16).map(|(n, _)| n),
                            "native_mmv_mw gate vs spv disagree {d:?} res={res} warps={warps} sg16={sg16}"
                        );
                    }
                }
            }
        }
    }

    /// AUDIT #2: Iq4Xs is mrow-eligible (`native_mmv_mrow_kernel_name` = Some) but has NO fused-
    /// residual build — the res-legality predicate must report that, and the variant table must
    /// agree, so `linear_mmv_mrow_at`'s residual assert catches an illegal residual Iq4Xs decode at
    /// the boundary instead of panicking in the SPV `expect()`.
    #[test]
    fn iq4xs_reports_no_mrow_residual() {
        use crate::gemm;
        use infr_core::DType::{Iq4Xs, Q4K};
        assert!(
            gemm::native_mmv_mrow_kernel_name(Iq4Xs).is_some(),
            "Iq4Xs should still be plain-mrow eligible"
        );
        assert!(
            !gemm::native_mmv_mrow_res_supported(Iq4Xs),
            "Iq4Xs has no _res mrow build — predicate must say so"
        );
        assert!(
            gemm::native_mmv_mrow_variant_spv(Iq4Xs, false, true, true).is_none(),
            "Iq4Xs _res variant must not exist"
        );
        // A dtype that DOES have the residual build reports true, so the predicate isn't vacuously
        // false for everyone.
        assert!(gemm::native_mmv_mrow_res_supported(Q4K));
        assert!(gemm::native_mmv_mrow_variant_spv(Q4K, false, true, true).is_some());
    }

    #[test]
    fn moe_mmq_dtypes_have_id_gemv_coverage() {
        for &d in infr_core::tensor::MOE_MMQ_DTYPES {
            assert!(
                native_id_kernel_name(d).is_some(),
                "{d:?} is in MOE_MMQ_DTYPES but native_id_kernel_name has no variant"
            );
            assert!(
                native_idm_kernel_name(d).is_some(),
                "{d:?} is in MOE_MMQ_DTYPES but native_idm_kernel_name has no variant"
            );
            assert!(
                native_id_paged_kernel_name(d).is_some(),
                "{d:?} is in MOE_MMQ_DTYPES but native_id_paged_kernel_name has no variant"
            );
            assert!(
                native_idm_paged_kernel_name(d).is_some(),
                "{d:?} is in MOE_MMQ_DTYPES but native_idm_paged_kernel_name has no variant"
            );
        }
    }

    // CPU reference GEMV for the production-entry-point tests below (odd in_f exercises bf16
    // packing / non-u32-aligned addressing).
    fn cpu_gemv(w: &[f32], x: &[f32], rows: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * out_f];
        for r in 0..rows {
            for o in 0..out_f {
                let mut acc = 0.0;
                for i in 0..in_f {
                    acc += x[r * in_f + i] * w[o * in_f + i];
                }
                y[r * out_f + o] = acc;
            }
        }
        y
    }

    /// Uploads `x` and allocates `y` as ordinary activation buffers, records `dispatch` (one of the
    /// production `Recorder::linear*` entries) on a fresh recorder TWICE — exercising the cached
    /// kernel-pipeline path the same way the old eager-runner tests did — and returns the second
    /// run's downloaded `y`. `w` must already be a `BufferUsage::Weights` allocation (resident-BDA,
    /// `device_addr()` is `Some`) — the ONLY shape production ever hands these entry points.
    #[allow(clippy::too_many_arguments)]
    fn run_linear(
        be: &VulkanBackend,
        w: &dyn Buffer,
        x: &[f32],
        rows: usize,
        in_f: usize,
        out_f: usize,
        dispatch: impl Fn(&crate::Recorder, &dyn Buffer, &dyn Buffer, &dyn Buffer, usize, usize, usize),
    ) -> Vec<f32> {
        let x_buf = be.alloc(x.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(x_buf.as_ref(), bytemuck::cast_slice(x)).unwrap();
        let y_buf = be
            .alloc(rows * out_f * 4, BufferUsage::Activations)
            .unwrap();
        for _ in 0..2 {
            let rec = be.recorder().unwrap();
            dispatch(&rec, w, x_buf.as_ref(), y_buf.as_ref(), rows, in_f, out_f);
            rec.finish().unwrap();
        }
        let mut y_bytes = vec![0u8; rows * out_f * 4];
        be.download(y_buf.as_ref(), &mut y_bytes).unwrap();
        bytemuck::cast_slice(&y_bytes).to_vec()
    }

    /// f32 weight: production entry is [`crate::Recorder::linear_f32`] (routes to the
    /// `linear_f32r` mrow/vec4 family at rows>1 — see that fn's doc). Formerly covered by the
    /// now-deleted eager `VulkanBackend::linear` (linear.rs), which bound the weight at raw offset
    /// 0 instead of through `vkb`'s sub-range — a latent bug for arena sub-tensors this rewrite
    /// also retires by construction (the weight here is a real `BufferUsage::Weights` allocation).
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (3usize, 5usize, 4usize);
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.01).collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32) * 0.02).collect();
        let wbuf = be.upload_weight(&w).unwrap();
        let y = run_linear(
            &be,
            wbuf.as_ref(),
            &x,
            rows,
            in_f,
            out_f,
            |rec, w, x, y, r, i, o| {
                rec.linear_f32(w, x, y, r, i, o);
            },
        );
        let want = cpu_gemv(&w, &x, rows, in_f, out_f);
        for (g, w) in y.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-3, "{g} vs {w}");
        }
    }

    /// f16 weight: production entry is [`crate::Recorder::linear`]. Formerly covered by the
    /// now-deleted eager `VulkanBackend::linear_f16` (ops.rs), which had the same raw-offset-0
    /// binding bug `linear_matches_cpu`'s doc describes.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_f16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (rows, in_f, out_f) = (2usize, 70usize, 5usize);
        let w: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 % 9.0) * 0.05 - 0.2)
            .collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32 % 7.0) * 0.03).collect();
        let wbuf = be.upload_weight_f16(&w).unwrap();
        let y = run_linear(
            &be,
            wbuf.as_ref(),
            &x,
            rows,
            in_f,
            out_f,
            |rec, w, x, y, r, i, o| {
                rec.linear(w, x, y, r, i, o);
            },
        );
        for (g, c) in y.iter().zip(cpu_gemv(&w, &x, rows, in_f, out_f).iter()) {
            assert!((g - c).abs() < 1e-2, "{g} vs {c}");
        }
    }

    /// bf16 weight: production entry is [`crate::Recorder::linear_bf16`]. Formerly covered by the
    /// now-deleted eager `VulkanBackend::linear_bf16` (ops.rs), which had the same raw-offset-0
    /// binding bug `linear_matches_cpu`'s doc describes.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn linear_bf16_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        // in_f odd → rows are NOT u32-aligned in the packed bf16 stream (exercises global addressing)
        let (rows, in_f, out_f) = (3usize, 65usize, 4usize);
        let w: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 % 11.0) * 0.04 - 0.2)
            .collect();
        let x: Vec<f32> = (0..rows * in_f).map(|i| (i as f32 % 5.0) * 0.06).collect();
        let wbuf = be.upload_weight_bf16(&w).unwrap();
        let y = run_linear(
            &be,
            wbuf.as_ref(),
            &x,
            rows,
            in_f,
            out_f,
            |rec, w, x, y, r, i, o| {
                rec.linear_bf16(w, x, y, r, i, o);
            },
        );
        // bf16 has 8 mantissa bits → looser tolerance than f16
        for (g, c) in y.iter().zip(cpu_gemv(&w, &x, rows, in_f, out_f).iter()) {
            assert!((g - c).abs() < 5e-2, "{g} vs {c}");
        }
    }
}
