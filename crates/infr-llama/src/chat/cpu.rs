//! [`CpuDenseChat`]: the CPU reference [`ChatModel`] for dense/MoE (`INFR_CPU=1`), also driving
//! the reference Metal backend when constructed via [`CpuDenseChat::new_metal`].

use super::ChatModel;
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// CPU reference backend (`INFR_CPU=1`) for dense/MoE: the agnostic compute-graph forward, no GPU.
/// Stateless full-prefill each turn (no cross-turn KV yet), but the shared `Chat` now feeds the FULL
/// rendered history in every turn, so multi-turn context works.
pub struct CpuDenseChat {
    model: SeamModel,
    /// Run the dense forward on the reference Metal backend instead of the CPU interpreter.
    metal: bool,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl CpuDenseChat {
    pub fn new(model: SeamModel) -> Self {
        Self {
            model,
            metal: false,
        }
    }

    /// Same dense model, but driven through the reference Metal backend (`INFR_METAL`).
    pub fn new_metal(model: SeamModel) -> Self {
        Self { model, metal: true }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for CpuDenseChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.metal {
            #[cfg(target_os = "macos")]
            return self
                .model
                .generate_metal(prompt, max_new, req, |p| on_piece(p));
            #[cfg(not(target_os = "macos"))]
            return Err(anyhow::anyhow!(
                "the Metal backend is only available on macOS"
            ));
        }
        // CPU MTP (INFR_MTP=1 on a head-bearing qwen35 GGUF): the exact-f32 reference for the
        // draft-verify loop — its acceptance rate is the oracle a GPU backend's alpha is judged
        // against (a backend whose head numerics drift from its trunk shows a lower alpha here).
        if crate::mtp::should_use_mtp(self.model.config(), self.model.engine_cfg()) {
            let head = crate::mtp::load_mtp_head(self.model.gguf(), self.model.config())?;
            return crate::mtp::generate_mtp_spec_cpu_timed(
                &self.model,
                &head,
                prompt,
                max_new,
                |p| on_piece(p),
            )
            .map(|(stats, _)| stats);
        }
        self.model
            .generate_cpu(prompt, max_new, req, |p| on_piece(p))
    }
}
