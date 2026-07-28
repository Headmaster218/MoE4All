//! Metal-backed [`ChatModel`]s (macOS only): [`MetalSeamChat`] (dense/MoE persistent-session
//! twin of [`crate::chat::DenseSeamChat`]) and [`SpecMetalChat`] (target+draft speculative decode).

#[cfg(target_os = "macos")]
use super::ChatModel;
#[cfg(target_os = "macos")]
use crate::{GenStats, SeamModel};
#[cfg(target_os = "macos")]
use anyhow::Result;

/// Metal seam backend for dense/MoE with a persistent session — the Apple-GPU twin of
/// [`DenseSeamChat`]: weights upload once, the KV cache persists across turns, and each turn
/// prefills only the suffix that differs from the previous rendered history.
#[cfg(target_os = "macos")]
pub struct MetalSeamChat {
    model: SeamModel,
    session: Option<crate::seam::model::DenseMetalSession>,
    mtp_head: Option<crate::mtp::MtpHeadWeights>,
    mtp_checked: bool,
}

#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl MetalSeamChat {
    pub fn new(model: SeamModel) -> Self {
        Self {
            model,
            session: None,
            mtp_head: None,
            mtp_checked: false,
        }
    }

    /// MTP mode is opt-in (`INFR_MTP=1`) and only for a qwen35 GGUF that ships an MTP head — the
    /// Metal twin of [`DenseSeamChat::wants_mtp`]. Memoized so a non-MTP GGUF doesn't re-parse
    /// its `Config` every turn.
    fn wants_mtp(&mut self) -> Result<bool> {
        if self.mtp_head.is_some() {
            return Ok(true);
        }
        if self.mtp_checked {
            return Ok(false);
        }
        self.mtp_checked = true;
        // Shared gate (`crate::mtp::should_use_mtp`) — the Metal twin of Vulkan's `wants_mtp`, one
        // decision so the two can't drift (Metal previously ignored the parked path SILENTLY while
        // Vulkan warned; the shared gate warns for both).
        if !crate::mtp::should_use_mtp(self.model.config(), self.model.engine_cfg()) {
            return Ok(false);
        }
        self.mtp_head = Some(crate::mtp::load_mtp_head(
            self.model.gguf(),
            self.model.config(),
        )?);
        Ok(true)
    }

    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            // `device.ctx` (INFR_CTX) shared size grammar; % resolves against the trained context
            // (this path has no VRAM-fit calc — the Metal working-set guard is the backstop).
            let train = self.model.config().n_ctx_train;
            let max_ctx = super::cfg_ctx(self.model.engine_cfg(), train).unwrap_or(train);
            self.session = Some(self.model.metal_session(max_ctx)?);
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for MetalSeamChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn reset_kv(&mut self) {
        super::reset_session(&mut self.session);
    }

    fn warmup(&mut self) -> Result<()> {
        // The shared session warmup, UNWRAPPED — no `with_profiling_suppressed` here, unlike the
        // vulkan/diffusion paths. The Metal backend resolves `prof.ops` at CONSTRUCTION, which
        // happens inside this first generate, so wrapping the call would resolve `profiling: false`
        // and disable the profiler for the whole session rather than just the warmup.
        //
        // The warmup forward is still excluded from the profile: `ProfileGate::enabled` consults
        // the process-wide suppression flag at CALL time (and this backend's own RAII counter),
        // so a bench that suppresses around its untimed turns gets a table of the timed reps.
        self.warmup_session()
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        // MTP (INFR_MTP=1 on a qwen35 head-bearing GGUF): the draft-verify-catchup loop over the
        // Metal trunk + head, instead of the plain session decode. The committed stream is the
        // target's greedy stream (the same invariant Vulkan MTP holds), so the two are
        // token-identical — pinned by the greedy-equivalence check in the CLI validation.
        if self.wants_mtp()? {
            let head = self.mtp_head.as_ref().expect("wants_mtp loaded it");
            return crate::mtp::generate_mtp_spec_metal(&self.model, head, prompt, max_new, |p| {
                on_piece(p)
            });
        }
        self.ensure_session()?;
        self.model.generate_metal_session(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            req,
            |p| on_piece(p),
        )
    }

    fn generate_constrained(
        &mut self,
        prompt: &str,
        max_new: usize,
        constraint: &mut crate::grammar::Constraint,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_session()?;
        self.model.generate_metal_session_constrained(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            Some(constraint),
            req,
            |p| on_piece(p),
        )
    }
}

/// Speculative Metal chat (`INFR_SPEC_DRAFT=<gguf>`): TARGET model verified decode with a small
/// same-tokenizer DRAFT proposing k tokens per round. Greedy-only — the committed stream is
/// exactly the target's greedy stream. Pays off for ≥8B-class targets (see issue #16's
/// measurements); the CLI warns when the size ratio looks too thin.
#[cfg(target_os = "macos")]
pub struct SpecMetalChat {
    target: SeamModel,
    draft: SeamModel,
    k: usize,
    target_session: Option<crate::seam::model::DenseMetalSession>,
    draft_session: Option<crate::seam::model::DenseMetalSession>,
}

#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SpecMetalChat {
    pub fn new(target: SeamModel, draft: SeamModel, k: usize) -> Self {
        Self {
            target,
            draft,
            k,
            target_session: None,
            draft_session: None,
        }
    }

    fn ensure_sessions(&mut self) -> Result<()> {
        if self.target_session.is_none() {
            // TWO models + TWO KV caches share the working set — a full-n_ctx_train pair
            // (40k tokens on qwen3) thrashes an 18 GB machine into second-long forwards.
            // Default to 8k unless `device.ctx` (INFR_CTX) says otherwise (shared size grammar;
            // % of the trained context, same note as ensure_session above).
            let train = self.target.config().n_ctx_train;
            let max_ctx =
                super::cfg_ctx(self.target.engine_cfg(), train).unwrap_or_else(|| train.min(8192));
            self.target_session = Some(self.target.metal_session(max_ctx)?);
            self.draft_session = Some(self.draft.metal_session(max_ctx)?);
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for SpecMetalChat {
    fn render_model(&self) -> &SeamModel {
        &self.target
    }

    fn warmup(&mut self) -> Result<()> {
        // Compile BOTH models' pipelines now (a spec round drives target prefill, draft decode,
        // and the batched verify) so serve's first request doesn't pay two cold starts.
        self.generate("Hi", 2, None, &mut |_| {})?;
        // Drop the warmup tokens so the first real prompt prefills clean slots from row 0. Both
        // sessions, so this can't use the shared `warmup_session` (whose discard is `reset_kv`,
        // which this backend deliberately leaves a no-op — see the trait's doc).
        super::reset_session(&mut self.target_session);
        super::reset_session(&mut self.draft_session);
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        _req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_sessions()?;
        // Split borrows: sessions come out of the options for the call.
        let mut ts = self.target_session.take().unwrap();
        let mut ds = self.draft_session.take().unwrap();
        let r = self.target.generate_metal_spec(
            &mut ts,
            &self.draft,
            &mut ds,
            prompt,
            max_new,
            self.k,
            |p| on_piece(p),
        );
        self.target_session = Some(ts);
        self.draft_session = Some(ds);
        r
    }
}
