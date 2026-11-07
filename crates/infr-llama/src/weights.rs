//! Tensor-payload accounting for resident weights (dense vs MoE-expert). Exact Vulkan layout
//! planning additionally accounts for the loader's groups, model-specific rewrites and constants.
use infr_core::WeightSource;
use infr_gguf::Gguf;

/// VRAM the model's weights will occupy once resident, split dense vs MoE-expert. Experts are
/// tracked separately so a future expert-streaming / partial-offload mode can budget them apart
/// from the always-resident dense weights — for a dense model `expert` is 0.
#[derive(Clone, Copy, Debug)]
pub struct WeightFootprint {
    /// Always-resident weights: projections, embeddings, norms.
    pub dense: u64,
    /// MoE expert weights (GGUF `*_exps` stacked tensors). 0 for dense models.
    pub expert: u64,
}
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl WeightFootprint {
    /// All-resident footprint: dense + every expert kept in VRAM.
    pub fn total(&self) -> u64 {
        self.dense + self.expert
    }

    /// Footprint if experts are STREAMED through an `n_slots`-slot pool of `stride`-byte slots
    /// (a VRAM slot pool) instead of all kept resident: `dense + n_slots·stride`, bounded
    /// regardless of the model's expert count. The MoE loader picks all-resident ([`total`]) when it
    /// fits VRAM, else reserves this and streams. (`stride` = one expert's max packed weight bytes.)
    pub fn streaming_total(&self, n_slots: usize, stride: usize) -> u64 {
        self.dense + n_slots as u64 * stride as u64
    }
}

/// Resident payload bytes for one tensor, padded to u32 alignment. Only BitNet I2S is converted
/// to F16 here; every other dtype retains its native bytes, including F32 norms and biases.
/// Model-specific rewrites are accounted for by the seam's exact upload plan.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn tensor_resident_bytes(dtype: infr_core::DType, numel: usize, nbytes: usize) -> u64 {
    let bytes = if dtype == infr_core::DType::I2S {
        numel as u64 * 2
    } else {
        nbytes as u64
    };
    (bytes + 3) & !3
}

/// Sum the resident weight footprint across all tensors (MoE-aware). Enumerating every tensor means
/// stacked expert tensors are counted in full, so this is correct for MoE the moment the arch is
/// supported. `token_embd` counts at its NATIVE upload size like every other tensor: the runner
/// uploads it raw-quant — as the lm head for tied models (no `output.weight`), and as the GPU
/// embed-gather table for untied ones (task #28). This used to count the tied case as a dequanted
/// f16 copy (`numel*2`), overstating a big-vocab model by GiBs — gemma-4-31B (262k vocab, Q6_K
/// embd) carried a phantom +1.6 GiB that alone pushed the dense placement into streaming a model
/// whose weights fit resident. (The rare host-embed fallback paths keep the table off-VRAM, so
/// counting it is the safe over-estimate direction there.)
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn weight_footprint(g: &Gguf) -> WeightFootprint {
    let mut dense = 0u64;
    let mut expert = 0u64;
    let qwen4exp = g.metadata().str("general.architecture") == Some(crate::arch::QWEN4EXP);
    for t in g.tensors() {
        // Qwen3.8's PLE table is deliberately an SSD/mmap tier: one inference token gathers only
        // 16 short rows. Pricing the multi-hundred-GiB table as an always-resident GPU weight would
        // reject the model before the loader can establish that tier.
        if qwen4exp && t.name == "per_layer_token_embd.weight" {
            continue;
        }
        let numel: usize = t.shape.iter().product();
        let bytes = tensor_resident_bytes(t.dtype, numel, t.nbytes);
        if t.name.contains("_exps") {
            expert += bytes;
        } else {
            dense += bytes;
        }
    }
    WeightFootprint { dense, expert }
}

#[cfg(test)]
mod tests {
    use super::tensor_resident_bytes;
    use infr_core::DType;

    #[test]
    fn resident_payload_preserves_native_float_integer_and_quant_bytes() {
        for (dtype, numel, nbytes, expected) in [
            (DType::F32, 3, 12, 12),
            (DType::F16, 3, 6, 8),
            (DType::Bf16, 3, 6, 8),
            (DType::I32, 3, 12, 12),
            (DType::U32, 3, 12, 12),
            (DType::Q4_0, 32, 18, 20),
            (DType::Q8_0, 32, 34, 36),
            (DType::Tq1_0, 256, 54, 56),
            (DType::F32, 0, 0, 0),
        ] {
            assert_eq!(
                tensor_resident_bytes(dtype, numel, nbytes),
                expected,
                "{dtype:?} must retain its native payload"
            );
        }
    }

    #[test]
    fn resident_payload_expands_only_i2s_to_f16() {
        // 128 two-bit values plus the per-tensor F32 scale become 128 F16 values.
        assert_eq!(tensor_resident_bytes(DType::I2S, 128, 36), 256);
        assert_eq!(tensor_resident_bytes(DType::I2S, 256, 68), 512);
    }
}
