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
    /// The ONE Vulkan backend MTP mode's trunk+head share across every `generate()` call
    /// (`ensure_mtp_backend`). MTP's driver rebuilds a fresh trunk+head SESSION every call by
    /// design (`crate::mtp`'s "no cross-turn KV reuse" doc) — but that's an ordinary allocation,
    /// not a device re-init, so this field is what keeps `warmup()`'s call and every real chat
    /// turn on the SAME VkDevice/allocator/pipeline-cache instead of constructing a new one each
    /// time (previously: two full Vulkan backends for a single-turn `INFR_MTP=1` run).
    mtp_vk: Option<infr_vulkan::VulkanBackend>,
    /// Physical device this chat's session pins: `Some(idx)` = `VulkanN` (the multi-device path,
    /// `new_on`), `None` = the default device (`new`, byte-identical to before). Threaded into
    /// [`ensure_session`](Self::ensure_session) and [`ensure_mtp_backend`](Self::ensure_mtp_backend)
    /// so the whole model — weights, KV, MTP trunk/head — lands on the one chosen GPU.
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
            mtp_vk: None,
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
            mtp_vk: None,
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
            let user_ctx = super::env_ctx_spec();
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

    /// Lazily construct the shared MTP Vulkan backend (see [`mtp_vk`](Self::mtp_vk)'s doc) —
    /// `generate`'s MTP branch calls this instead of letting `crate::mtp::generate_mtp_spec_vulkan`
    /// construct its own per-call backend.
    fn ensure_mtp_backend(&mut self) -> Result<()> {
        if self.mtp_vk.is_none() {
            self.mtp_vk = Some(match self.dev {
                Some(idx) => infr_vulkan::VulkanBackend::new_on(idx)
                    .map_err(|e| anyhow::anyhow!("vulkan init (Vulkan{idx}): {e}"))?,
                None => infr_vulkan::VulkanBackend::new()
                    .map_err(|e| anyhow::anyhow!("vulkan init: {e}"))?,
            });
        }
        Ok(())
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for DenseSeamChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn reset_kv(&mut self) {
        super::reset_session(&mut self.session);
    }

    fn warmup(&mut self) -> Result<()> {
        // The shared session warmup (throwaway generate + reset so the first real prompt prefills
        // clean slots from row 0), wrapped in the INFR_PROF2 suppression the Vulkan recorders need.
        crate::with_prof2_suppressed(|| self.warmup_session())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.wants_mtp()? {
            self.ensure_mtp_backend()?;
            let head = self.mtp_head.as_ref().expect("wants_mtp loaded it");
            let vk = self.mtp_vk.as_ref().expect("ensure_mtp_backend set it");
            return crate::mtp::generate_mtp_spec_vulkan_timed_on(
                vk,
                &self.model,
                head,
                prompt,
                max_new,
                |p| on_piece(p),
            )
            .map(|(stats, _)| stats);
        }
        self.ensure_session()?;
        self.model.generate_vulkan_session(
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
        self.model.generate_vulkan_session_constrained(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            Some(constraint),
            req,
            |p| on_piece(p),
        )
    }
}
