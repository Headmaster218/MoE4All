//! Weight-footprint accounting: how many VRAM bytes a model's weights will occupy once resident
//! (dense vs MoE-expert), mirroring the native-quant/f16 upload policy used by the production
//! agnostic-seam loaders.
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

/// Resident VRAM bytes for one tensor, mirroring [`upload_wt`]'s path so the estimate matches what
/// actually gets allocated: native raw blocks (padded to u32) for every quant format, else f16
/// (float/norms dequanted to half).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn tensor_resident_bytes(dtype: infr_core::DType, numel: usize, nbytes: usize) -> u64 {
    if infr_vulkan::linear::native_dense_supported(dtype) {
        ((nbytes + 3) & !3) as u64 // raw blocks, padded to u32 alignment
    } else {
        (numel * 2) as u64 // f16
    }
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
