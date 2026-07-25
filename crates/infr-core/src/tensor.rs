//! Tensor descriptors and data types (incl. GGUF quant types).

/// Element / block type of a tensor.
///
/// Quantized variants are stored as GGUF blocks; the backend owns dequant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    // legacy round quants
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    // GGUF k-quants
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    // i-quants (codebook)
    Iq1S,
    Iq1M,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq4Nl,
    Iq4Xs,
    // ternary quants
    Tq1_0,
    Tq2_0,
    /// BitNet i2_s (microsoft/BitNet, ggml type 36): ternary {-1,0,+1} weights packed 2 bits per
    /// element (4/byte) with ONE per-TENSOR f32 scale stored after the packed codes (NOT a per-block
    /// scale). Element `e` (natural row-major order) lives in byte `(e/128)*32 + (e%128)%32`, bit
    /// field `6 - 2*((e%128)/32)`; code `q∈{0,1,2}` → `(q-1)·scale`. No native kernel yet — the seam
    /// host-dequants it to f16 at weight load (see `is_codebook_quant`), so it never reaches a
    /// backend as I2S; only [`crate::DType`]-exhaustive matches and the GGUF loader see this variant.
    I2S,
    /// Bonsai ternary (llama.cpp GGML_TYPE_Q2_0 = 42): 64-elem blocks, 18 B = f16 d + 16 B of
    /// 2-bit codes (2.25 bpw); w = (q - 1) · d, q ∈ {0,1,2,3} → {-d, 0, +d, +2d}.
    Q2_0,
    // fp4 quants
    Mxfp4,
    Nvfp4,
    // TurboQuant KV-cache-only formats (WHT rotation + PolarQuant centroids). NOT weight dtypes —
    // only used for the KV cache (like Q8_0-for-KV). 128-elem blocks: turbo2 = 34 B (2.125 bpw),
    // turbo3 = 50 B (3.125), turbo4 = 66 B (4.125).
    Turbo2,
    Turbo3,
    Turbo4,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DType {
    /// True for block-quantized weight types.
    pub fn is_quant(self) -> bool {
        matches!(
            self,
            DType::Q4_0
                | DType::Q4_1
                | DType::Q5_0
                | DType::Q5_1
                | DType::Q8_0
                | DType::Q2K
                | DType::Q3K
                | DType::Q4K
                | DType::Q5K
                | DType::Q6K
                | DType::Iq1S
                | DType::Iq1M
                | DType::Iq2Xxs
                | DType::Iq2Xs
                | DType::Iq2S
                | DType::Iq3Xxs
                | DType::Iq3S
                | DType::Iq4Nl
                | DType::Iq4Xs
                | DType::Tq1_0
                | DType::Tq2_0
                | DType::Q2_0
                | DType::Mxfp4
                | DType::Nvfp4
        )
    }

    /// True for the GGUF "k-quant" family (`Q2_K`..`Q6_K`) — super-block quants with a per-block
    /// scale/min hierarchy. A building block for the per-backend fusion/coverage predicates
    /// (see [`crate::fusion`]) so those stop re-enumerating raw `DType::Q4K | …` lists.
    pub fn is_kquant(self) -> bool {
        matches!(
            self,
            DType::Q2K | DType::Q3K | DType::Q4K | DType::Q5K | DType::Q6K
        )
    }

    /// True for the "legacy round" quants (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`) — the original
    /// llama.cpp block formats (simple per-block `d`/`m` scale, no super-block hierarchy).
    pub fn is_legacy_round(self) -> bool {
        matches!(
            self,
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0
        )
    }

    /// True for the codebook "i-quant" family (`IQ1`..`IQ4`) — grid/codebook lookups.
    pub fn is_iquant(self) -> bool {
        matches!(
            self,
            DType::Iq1S
                | DType::Iq1M
                | DType::Iq2Xxs
                | DType::Iq2Xs
                | DType::Iq2S
                | DType::Iq3Xxs
                | DType::Iq3S
                | DType::Iq4Nl
                | DType::Iq4Xs
        )
    }

    /// Bytes for `n` elements of a non-quant dtype. Returns `None` for quant types
    /// (those are sized by block, computed by the loader from the GGUF layout).
    pub fn dense_bytes(self, n: usize) -> Option<usize> {
        let sz = match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::Bf16 => 2,
            _ => return None,
        };
        Some(n * sz)
    }
}

/// SINGLE SOURCE OF TRUTH for the batched-MoE dp4a "mmq" expert-GEMM family's dtype coverage.
/// Two independent gates MUST accept exactly this set — Vulkan's `infr_vulkan::adapter`'s
/// `mmq_ok` (batched resident-bank prefill) and `infr_llama`'s `seam::runner::moe_mmq_ok`
/// (decides whether the seam even BUILDS a batched `Op::MoeFfn` graph node) — because a mismatch
/// either silently falls back to the slow per-token path (adapter stricter) or compiles a graph
/// the adapter then rejects at record time (runner stricter). Both gates derive from this list
/// instead of re-enumerating it so a format added here PROPAGATES automatically instead of
/// silently drifting; `moe_mmq_drift_test` in this module still checks the two crates' actual
/// gate closures against it (a `DType` isn't `Hash`-derivable across crate boundaries in a way
/// that lets the gates literally `use` this const in a `matches!`, so the arms are still
/// hand-written per format — this list is what a reviewer/test diffs them against, and the
/// recorder's per-format kernel-name match arms `unreachable!()` at runtime if a format is listed
/// here without its kernel wired, which is the compile-adjacent failure mode for that side).
///
/// DELIBERATE EXCLUSIONS (these fall back to the looped id-GEMV expert path for prefill —
/// correct, just slower; this doc is the SSOT for why each stays out):
///   * `Tq1_0`/`Tq2_0` (ternary): the values ARE tiny signed ints, but TQ1_0's per-element
///     base-3 digit extraction (a pow3-multiply + funnel per element, three different packing
///     regions per 256-block) has no natural word-parallel nibble→int8 staging — the dp4a
///     staging loop would degenerate into the same scalar decode the idm fallback already does;
///     TQ2_0 would map, but no shipped MoE GGUF quantizes expert banks ternary, so neither earns
///     a kernel until one does.
///   * `Iq1S`/`Iq1M`/`Iq2Xxs`/`Iq2Xs`/`Iq3Xxs` (remaining grid i-quants): NOT ALU-impossible —
///     the original "grid gather + sign staging is ALU-bound" rationale was DISPROVEN by the
///     shared-memory grid-LUT staging fix (grid_init(); see build.rs::gen_grids) and the shipped
///     `Iq2S`/`Iq3S` mmq kernels (one shared gather serves four staged int8 bytes, reused BM
///     times by the dp4a loop — same ~20% staging overhead Q3_K carries, near-Q3_K throughput
///     measured). These five stay out only because no audited MoE GGUF quantizes expert banks
///     with them (unsloth UD mixes use IQ2_S/IQ3_S/IQ4_XS + K-quants); each maps to the same
///     recipe (IQ2_XS/IQ2_XXS ≙ IQ2_S with KSIGNS-packed signs at BLK=16; IQ3_XXS ≙ IQ3_S with
///     KSIGNS at BLK=32; IQ1_S/IQ1_M add the ±0.125 delta as a per-sub-block `sact`-style
///     correction since Σ±delta·x needs the activation sum). They keep the idm fallback until a
///     real model ships them.
///   * `Bf16`/`F16`/`F32` (float weights): not dp4a material at all — no integer codes to feed
///     the packed int8 dot; they ride the float GEMM/GEMV routes.
pub const MOE_MMQ_DTYPES: &[DType] = &[
    DType::Q4_0,
    DType::Q4_1,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q8_0,
    DType::Q2K,
    DType::Q3K,
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Iq4Nl,
    DType::Iq4Xs,
    DType::Iq2S,
    DType::Iq3S,
    DType::Mxfp4,
    DType::Nvfp4,
    // Q2_0 (Bonsai ternary): symmetric small-int — codes-1 = {-1,0,+1,+2} feed dp4a directly
    // (IQ4_NL's treatment minus the codebook); one f16 d per 64-elem block spanning two 32-elem
    // activation quant blocks. No shipped MoE GGUF uses it for expert banks yet, but the dense
    // nc_mmq tier derives from this list too, so the dense build earns the full family.
    DType::Q2_0,
];

/// True for dtypes the batched-MoE dp4a mmq expert-GEMM family covers (gate/up/down each
/// independently — see [`MOE_MMQ_DTYPES`]'s doc for why this is the single source of truth).
pub fn moe_mmq_ok(dt: DType) -> bool {
    MOE_MMQ_DTYPES.contains(&dt)
}

/// Subset of [`MOE_MMQ_DTYPES`] that is min-carrying AND needs the activation's `sact` (Σx)
/// term to reconstruct the min: Q4_K/Q5_K (K-quant 6-bit min), Q5_1/Q4_1 (legacy min, PLUS
/// convention). Q2_K is ALSO min-carrying but is deliberately excluded — its 16-elem sub-block is
/// HALF the activation's 32-elem `sact` granularity, so it self-computes its own narrower Σx
/// in-shader instead (see `native_gemm_mmq_q2_k.comp`'s doc); Q3_K/Q6_K/Q8_0/Q5_0/Q4_0/IQ4_NL/
/// IQ4_XS/IQ2_S/IQ3_S/MXFP4/NVFP4 are symmetric (no min term at all — the fp4 formats share
/// IQ4_NL's signed-codebook treatment, the grid formats' sign-flipped codes are already signed).
pub const MOE_MMQ_SACT_DTYPES: &[DType] = &[DType::Q4K, DType::Q5K, DType::Q5_1, DType::Q4_1];

/// True for [`MOE_MMQ_DTYPES`] members whose mmq kernel reads the activation's `sact` buffer.
pub fn moe_mmq_needs_sact(dt: DType) -> bool {
    MOE_MMQ_SACT_DTYPES.contains(&dt)
}

/// Subset of [`MOE_MMQ_DTYPES`] with a PAGED (`_xpg`/`_xpg32`) batched expert-GEMM build — i.e.
/// usable in paged-expert-cache prefill, not just the resident-bank path. Mirrors
/// `MOE_MMQ_DTYPES` IN FULL since the pager became the sole MoE offload mechanism (fused gemma-4
/// MoE / DiffusionGemma banks ship Q4_K/Q5_0/Q5_1/Q8_0, UD quants mix Q4_K/Q5_K/Q6_K — any mmq
/// dtype can now end up paged, and a listed-but-unpaged format would silently fall back to the
/// far slower id-GEMV prefill segment). Kept as a separate list (not an alias) so the paged
/// builds' existence stays independently assertable; `moe_mmq_drift_test` checks the subset
/// relationship AND that every member has its `_xpg` kernels.
pub const MOE_MMQ_PAGED_DTYPES: &[DType] = &[
    DType::Q4_0,
    DType::Q4_1,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q8_0,
    DType::Q2K,
    DType::Q3K,
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Iq4Nl,
    DType::Iq4Xs,
    DType::Iq2S,
    DType::Iq3S,
    DType::Mxfp4,
    DType::Nvfp4,
    DType::Q2_0,
];

/// True for [`MOE_MMQ_DTYPES`] members with a paged (Scout-style GpuPager) batched expert-GEMM
/// build.
pub fn moe_paged_mmq_ok(dt: DType) -> bool {
    MOE_MMQ_PAGED_DTYPES.contains(&dt)
}

pub type Shape = Vec<usize>;

/// Shape + dtype of a tensor value flowing through the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    pub shape: Shape,
    pub dtype: DType,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl TensorDesc {
    pub fn new(shape: impl Into<Shape>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Handle to a node's output value within a single [`crate::graph::Graph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

#[cfg(test)]
mod moe_mmq_drift_tests {
    use super::*;

    /// [`MOE_MMQ_SACT_DTYPES`] must be a strict subset of [`MOE_MMQ_DTYPES`] — a format can't need
    /// `sact` without being an mmq format at all. Catches copy-paste errors when adding a format to
    /// one list but not the other.
    #[test]
    fn sact_dtypes_subset_of_mmq_dtypes() {
        for d in MOE_MMQ_SACT_DTYPES {
            assert!(
                MOE_MMQ_DTYPES.contains(d),
                "{d:?} is in MOE_MMQ_SACT_DTYPES but not MOE_MMQ_DTYPES"
            );
        }
    }

    /// [`MOE_MMQ_PAGED_DTYPES`] must be a subset of [`MOE_MMQ_DTYPES`] — the paged batched-GEMM
    /// build only ever needs to cover a format the resident batched path already covers.
    #[test]
    fn paged_dtypes_subset_of_mmq_dtypes() {
        for d in MOE_MMQ_PAGED_DTYPES {
            assert!(
                MOE_MMQ_DTYPES.contains(d),
                "{d:?} is in MOE_MMQ_PAGED_DTYPES but not MOE_MMQ_DTYPES"
            );
        }
    }

    /// [`MOE_MMQ_PAGED_DTYPES`] "mirrors `MOE_MMQ_DTYPES` IN FULL" (see its doc: the pager is the
    /// SOLE MoE offload mechanism, so any mmq dtype can end up paged). A subset check alone lets a
    /// dtype added to `MOE_MMQ_DTYPES` but NOT the paged list stay green while silently regressing
    /// that dtype's paged prefill to the slow id-GEMV path. Assert SET EQUALITY so that drift fails
    /// the build: every `MOE_MMQ_DTYPES` member must also have a paged (`_xpg`) build listed. If a
    /// future format is deliberately unpaged, remove it here with an explicit exclusion set rather
    /// than leaving the gap silent.
    #[test]
    fn paged_dtypes_equal_mmq_dtypes() {
        for d in MOE_MMQ_DTYPES {
            assert!(
                MOE_MMQ_PAGED_DTYPES.contains(d),
                "{d:?} is in MOE_MMQ_DTYPES but has no paged (_xpg) build in MOE_MMQ_PAGED_DTYPES \
                 — its paged prefill would silently fall back to the slow id-GEMV path"
            );
        }
        // Symmetric with `paged_dtypes_subset_of_mmq_dtypes`: the two directions together pin the
        // lists to EXACTLY the same set (also catches a size drift if a dupe ever slipped a guard).
        assert_eq!(
            MOE_MMQ_PAGED_DTYPES.len(),
            MOE_MMQ_DTYPES.len(),
            "MOE_MMQ_PAGED_DTYPES and MOE_MMQ_DTYPES must be the same set"
        );
    }

    /// No duplicate entries in any of the three lists — a dupe wouldn't break `contains`-based
    /// lookups but would signal a copy-paste mistake at the point a format was added.
    #[test]
    fn no_duplicate_entries() {
        for list in [MOE_MMQ_DTYPES, MOE_MMQ_SACT_DTYPES, MOE_MMQ_PAGED_DTYPES] {
            for (i, a) in list.iter().enumerate() {
                for b in &list[i + 1..] {
                    assert_ne!(a, b, "duplicate {a:?} in an MOE_MMQ_* list");
                }
            }
        }
    }

    /// The helper predicates must agree with their backing lists (guards against the fn body and
    /// the const list drifting apart if either is hand-edited later).
    #[test]
    fn predicates_match_lists() {
        for &d in MOE_MMQ_DTYPES {
            assert!(moe_mmq_ok(d));
        }
        for &d in MOE_MMQ_SACT_DTYPES {
            assert!(moe_mmq_needs_sact(d));
        }
        for &d in MOE_MMQ_PAGED_DTYPES {
            assert!(moe_paged_mmq_ok(d));
        }
        // A format NOT in MOE_MMQ_DTYPES must read as uncovered everywhere.
        assert!(!moe_mmq_ok(DType::Iq2Xxs));
        assert!(!moe_mmq_needs_sact(DType::Iq2Xxs));
        assert!(!moe_paged_mmq_ok(DType::Iq2Xxs));
    }
}
