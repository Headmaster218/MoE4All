//! Vulkan-backed [`ChatModel`]: [`DenseSeamChat`] (dense/MoE — and, since the phase-3 cutover,
//! qwen35 too — on the Vulkan agnostic seam with a persistent KV session).

use super::ChatModel;
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// Dense/MoE on the VULKAN agnostic seam with a persistent KV session (`INFR_SEAM=1` for
/// `infr run`): weights upload once, and every turn prefills only the token suffix that differs
/// from the previous turn — the seam twin of the bespoke `ChatSession`'s incremental prefill.
///
/// This is the default `infr run`/`infr serve` path for EVERY arch including qwen35 (Phase 3
/// cutover — see the matching comment at both CLI call sites), so it's also where MTP mode
/// (issue #33, `docs/mtp.md`) lives: `mtp_head` is `Some` once resolved+loaded, built lazily on
/// the first [`generate`](ChatModel::generate) call when [`wants_mtp`](Self::wants_mtp) is true
/// (opt-in `INFR_MTP=1`, and only for a qwen35 GGUF that actually ships an MTP head —
/// `Config::n_layer_nextn`'s doc). `INFR_MTP` unset/`0`, or a GGUF without an MTP head:
/// `wants_mtp` is always false, `mtp_head` stays `None` forever, and `generate` takes the EXACT
/// same `session` path it always has — zero risk to non-MTP models/GGUFs.
pub struct DenseSeamChat {
    model: SeamModel,
    session: Option<crate::seam::model::DenseVulkanSession>,
    mtp_head: Option<crate::mtp::MtpHeadWeights>,
    mtp_checked: bool,
    /// The caller-held MTP TRUNK session state (`crate::seam::SeamKv`) MTP mode keeps across
    /// `generate()` calls - on the SAME Vulkan backend as `session` (the MTP branch drives
    /// [`crate::mtp::generate_mtp_spec_vulkan_timed_on_state`] with `session`'s backend, pins, and
    /// `max_ctx`). The bound trunk weights + pager registration persist turn to turn (the cold
    /// call binds through the paged-MoE placement binder); only the KV rows are reset per call
    /// (`crate::mtp`'s "no cross-turn KV reuse" doc). `None` until the first MTP turn; stays
    /// `None` forever on a non-MTP model.
    mtp_trunk: Option<crate::seam::SeamKv>,
    /// Physical device this chat's session pins: `Some(idx)` = `VulkanN` (the multi-device path,
    /// `new_on`), `None` = the default device (`new`, byte-identical to before). Threaded into
    /// [`ensure_session`](Self::ensure_session) so the whole model - weights, KV, MTP trunk/head -
    /// lands on the one chosen GPU.
    dev: Option<usize>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DenseSeamChat {
    pub fn new(model: SeamModel) -> Self {
        Self {
            model,
            session: None,
            mtp_head: None,
            mtp_checked: false,
            mtp_trunk: None,
            dev: None,
        }
    }

    /// [`new`](Self::new) pinned to physical device `idx` (`VulkanN`) — the multi-device `infr run`
    /// / serialised-serve path. Everything this chat allocates lands on that GPU. `new` (the default
    /// device) is unchanged.
    pub fn new_on(model: SeamModel, idx: usize) -> Self {
        Self {
            model,
            session: None,
            mtp_head: None,
            mtp_checked: false,
            mtp_trunk: None,
            dev: Some(idx),
        }
    }

    /// MTP mode is opt-in (`INFR_MTP=1`) and Vulkan-only this phase (the invariant test + the
    /// oracle comparison in `docs/mtp.md` are both pinned on Vulkan — CPU/Metal MTP is
    /// unimplemented, not merely untested; `DenseSeamChat` IS always Vulkan, so no backend gate
    /// is needed here beyond the GGUF check). Memoized after the first call (`mtp_checked`) so a
    /// non-MTP GGUF doesn't re-parse its `Config` every turn.
    fn wants_mtp(&mut self) -> Result<bool> {
        if self.mtp_head.is_some() {
            return Ok(true);
        }
        if self.mtp_checked {
            return Ok(false);
        }
        self.mtp_checked = true;
        // Shared gate (`crate::mtp::should_use_mtp`) so Vulkan and Metal can't drift: `INFR_MTP=1`,
        // MTP not parked, and a head-bearing GGUF. It emits the "parked" warning itself.
        if !crate::mtp::should_use_mtp(self.model.config(), self.model.engine_cfg()) {
            return Ok(false);
        }
        self.mtp_head = Some(crate::mtp::load_mtp_head(
            self.model.gguf(),
            self.model.config(),
        )?);
        Ok(true)
    }

    /// Lazily open the persistent Vulkan session. Explicit `INFR_CTX` = user override (shared
    /// size grammar: `8192`, `256k`, or `50%` of the free-VRAM KV capacity — see
    /// `infr_core::parse_size`); token counts are used verbatim (NEVER clamped — the Vulkan VRAM
    /// budget guard still errors cleanly at alloc time if it truly doesn't fit); unset = the
    /// model's trained context, clamped to the VRAM budget (`vulkan_session_default`) so a
    /// long-context model's default KV cache can't blow VRAM.
    fn ensure_session(&mut self) -> Result<()> {
        if self.session.is_none() {
            let user_ctx = super::cfg_ctx_spec(self.model.engine_cfg());
            self.session = Some(match user_ctx {
                Some(infr_core::SizeSpec::Bytes(ctx)) => {
                    self.model.vulkan_session_on(self.dev, ctx as usize)?
                }
                Some(infr_core::SizeSpec::Percent(f)) => {
                    self.model.vulkan_session_frac_on(self.dev, f)?
                }
                None => self.model.vulkan_session_default_on(self.dev)?,
            });
        }
        Ok(())
    }

    fn generate_turn_impl(
        &mut self,
        prompt: &str,
        stable_prefix: Option<&str>,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.wants_mtp()? {
            // MTP runs on the MAIN session's backend (one VkDevice for the whole model): the
            // trunk verify routes through the same paged-MoE session infrastructure the non-MTP
            // path uses - the first (cold) call binds the trunk weights via the placement
            // planner/pager, later calls reuse them through `mtp_trunk`. In MTP mode `session`
            // itself never generates, so this backend carries exactly one trunk weight upload.
            self.ensure_session()?;
            let session = self.session.as_ref().expect("ensure_session set it");
            // Placement/pager decisions read the CURRENT placement pins - enter this session's
            // own scope around the MTP call, exactly like the non-MTP generation entry does.
            let _scope = crate::seam::PlacementScope::enter(session.pins().clone());
            // The trunk's KV was sized at the session's max_ctx on the cold call, but the cold
            // init may have re-clamped it against live free VRAM - once bound, the trunk's own
            // max_ctx is the authority (same rule as the non-MTP path's `session.max_ctx` refresh).
            let max_ctx = self
                .mtp_trunk
                .as_ref()
                .map_or(session.max_ctx, |st| st.max_ctx());
            let head = self.mtp_head.as_ref().expect("wants_mtp loaded it");
            return crate::mtp::generate_mtp_spec_vulkan_timed_on_state(
                &session.be,
                &mut self.mtp_trunk,
                &self.model,
                head,
                max_ctx,
                prompt,
                max_new,
                |p| on_piece(p),
            )
            .map(|(stats, _)| stats);
        }
        self.ensure_session()?;
        self.model.generate_vulkan_session_turn(
            self.session.as_mut().unwrap(),
            prompt,
            stable_prefix,
            max_new,
            req,
            |p| on_piece(p),
        )
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for DenseSeamChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn render_stable_prefix(&self, messages: &[(&str, &str)]) -> Result<Option<String>> {
        self.model.render_chat_messages_stable(messages).map(Some)
    }

    fn reset_kv(&mut self) {
        super::reset_session(&mut self.session);
        // MTP twin of the slot reset: forget the trunk's materialized tokens (weights + pager
        // registration stay bound - the next MTP turn re-prefills into the same session).
        if let Some(st) = self.mtp_trunk.as_mut() {
            st.reset();
        }
    }

    fn warmup(&mut self) -> Result<()> {
        // The shared session warmup (throwaway generate + reset so the first real prompt prefills
        // clean slots from row 0), wrapped in the INFR_PROF_OPS suppression the Vulkan recorders need.
        crate::with_profiling_suppressed(|| {
            self.generate_turn_impl("Hi", Some(""), 2, None, &mut |_| {})?;
            self.reset_kv();
            Ok(())
        })
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_turn_impl(prompt, None, max_new, req, on_piece)
    }

    fn generate_turn_with_step_hook(
        &mut self,
        prompt: &str,
        stable_prefix: Option<&str>,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
        _on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.generate_turn_impl(prompt, stable_prefix, max_new, req, on_piece)
    }

    fn generate_constrained_turn(
        &mut self,
        prompt: &str,
        stable_prefix: Option<&str>,
        max_new: usize,
        constraint: &mut crate::grammar::Constraint,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.ensure_session()?;
        self.model.generate_vulkan_session_turn_constrained(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            stable_prefix,
            Some(constraint),
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
        self.generate_constrained_turn(prompt, None, max_new, constraint, req, on_piece)
    }
}
