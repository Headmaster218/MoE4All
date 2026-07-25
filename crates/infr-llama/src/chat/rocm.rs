//! ROCm-backed [`ChatModel`] (Linux + ROCm/HIP only): the AMD-GPU twin of
//! [`crate::chat::MetalSeamChat`]: weights upload once, the KV cache persists across
//! turns, and each turn prefills only the suffix that differs from the previous rendered
//! history.
//!
//! The real implementation lives behind `cfg(all(target_os = "linux", feature = "rocm"))`;
//! without the feature the constructor returns a clean error message.

use super::ChatModel;
use crate::{GenStats, SeamModel};

#[cfg(all(target_os = "linux", feature = "rocm"))]
use crate::seam::model::DenseRocmSession;
#[cfg(all(target_os = "linux", feature = "rocm"))]
use anyhow::Result;

/// ROCm seam backend — the AMD-GPU twin of [`MetalSeamChat`]: persistent session,
/// KV cache across turns, suffix-only prefill.
#[cfg(all(target_os = "linux", feature = "rocm"))]
pub struct RocmSeamChat {
    model: SeamModel,
    session: Option<DenseRocmSession>,
    dev_idx: u32,
}

#[cfg(all(target_os = "linux", feature = "rocm"))]
impl RocmSeamChat {
    pub fn new(model: SeamModel, dev_idx: u32) -> Result<Self> {
        Ok(Self {
            model,
            session: None,
            dev_idx,
        })
    }

    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            // INFR_CTX shared size grammar; % resolves against the trained context (shared with
            // the Metal path — `chat::env_ctx`).
            let train = self.model.config().n_ctx_train;
            let max_ctx = super::env_ctx(train).unwrap_or(train);
            self.session = Some(self.model.rocm_session(max_ctx, self.dev_idx)?);
        }
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "rocm"))]
impl ChatModel for RocmSeamChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn reset_kv(&mut self) {
        super::reset_session(&mut self.session);
    }

    fn warmup(&mut self) -> Result<()> {
        // The shared session warmup, unwrapped (the ROCm backend has no INFR_PROF2 recorder to
        // suppress) — same body Metal uses.
        self.warmup_session()
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_session()?;
        let session = self.session.as_mut().unwrap();
        self.model
            .generate_rocm_session(session, prompt, max_new, req, on_piece)
    }
}

// ── Placeholder (feature not active) ─────────────────────────────────────────

/// ROCm seam backend placeholder — returns a clean error from `new()` so the CLI
/// can surface it as a build-time feature gate.
#[cfg(not(all(target_os = "linux", feature = "rocm")))]
pub struct RocmSeamChat {
    #[allow(dead_code)]
    _model: SeamModel,
}

#[cfg(not(all(target_os = "linux", feature = "rocm")))]
impl RocmSeamChat {
    pub fn new(_model: SeamModel, _dev_idx: u32) -> anyhow::Result<Self> {
        anyhow::bail!(
            "ROCm backend not compiled — build with `cargo build --features rocm` \
             on a Linux machine with ROCm/HIP installed (docs/rocm-plan.md Phase 0)"
        )
    }
}

#[cfg(not(all(target_os = "linux", feature = "rocm")))]
impl ChatModel for RocmSeamChat {
    fn render_model(&self) -> &SeamModel {
        unreachable!("RocmSeamChat placeholder")
    }

    fn reset_kv(&mut self) {}

    fn warmup(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn generate(
        &mut self,
        _prompt: &str,
        _max_new: usize,
        _req: Option<&crate::sampling::RequestCtx>,
        _on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<GenStats> {
        unreachable!("RocmSeamChat::generate: backend not compiled")
    }
}
