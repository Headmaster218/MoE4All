//! Shared model-config types. (The bespoke KV-cache/scratch structs that lived here died with
//! the bespoke engine — the seam holds its state in `seam::SeamKv`, including qwen35's
//! gated-DeltaNet conv/S state.)

/// Routed-expert (MoE) shape: expert count, top-k, per-expert FFN width, routed-weight scale, and
/// the gating semantics (`gating`/`norm_w`/`weight_before`) that differ between softmax MoE
/// (qwen3moe/qwen35moe/diffusion-gemma) and llama4's sigmoid top-1.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    pub scale: f32,
    /// Router gating function (softmax over experts vs per-expert sigmoid — `infr_core::graph::MoeGating`).
    pub gating: infr_core::graph::MoeGating,
    /// Renormalize the selected top-k weights to sum to 1 (`true` for softmax MoE; `false` for
    /// llama4's un-normalized sigmoid top-1). Threaded into `Op::MoeFfn.norm_w`.
    pub norm_w: bool,
    /// Apply the routing weight to the expert INPUT (llama4 `weight_before_ffn`) vs the output.
    /// Threaded into `Op::MoeFfn.weight_before`.
    pub weight_before: bool,
    /// DeepSeek V3+: number of expert groups for group-limited routing (0 = no grouping).
    pub n_expert_groups: u32,
    /// DeepSeek V3+: number of groups selected per routing decision.
    pub n_expert_groups_used: u32,
}
