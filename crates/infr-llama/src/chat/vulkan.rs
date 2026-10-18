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
    session: Option<crate::seam::model::SeamSession>,
    mtp_head: Option<crate::mtp::MtpHeadWeights>,
    mtp_checked: bool,
    /// The Vulkan backend qwen35 MTP mode uses for its one-shot head path (`ensure_mtp_backend`).
    /// Qwen3.8's detached MTP runtime shares the persistent seam session's backend instead, so a
    /// layer-split `INFR_PIPELINE` keeps the trunk and MTP head on the same two devices.
    mtp_vk: Option<infr_vulkan::VulkanBackend>,
    /// Qwen3.8's detached MTP head and target slot. Created before the target's first forward so
    /// the fixed head weights/runtime precede unified expert-pool finalization.
    qwen4_mtp: Option<crate::mtp::Qwen4MtpRuntime>,
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
            qwen4_mtp: None,
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
            qwen4_mtp: None,
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
            let tp_devices =
                crate::seam::tensor_parallel_devices(self.model.engine_cfg()).map(<[usize]>::to_vec);
            if let Some(devices) = tp_devices {
                let ctx = match user_ctx {
                    Some(infr_core::SizeSpec::Bytes(ctx)) => Some(ctx as usize),
                    _ => None,
                };
                self.session = Some(crate::seam::model::SeamSession::TensorParallel(
                    self.model.vulkan_tensor_parallel_session_on(&devices, ctx)?,
                ));
                return Ok(());
            }
            // Row-parallel dense + expert-parallel MoE session: every sharded projection runs on
            // both cards concurrently, so this is the session shape that removes the layer-split
            // serialization rather than merely moving weights across devices.
            let ep_devices =
                crate::seam::expert_parallel_devices(self.model.engine_cfg()).map(<[usize]>::to_vec);
            if let Some(devices) = ep_devices {
                let ctx = match user_ctx {
                    Some(infr_core::SizeSpec::Bytes(ctx)) => Some(ctx as usize),
                    _ => None,
                };
                self.session = Some(crate::seam::model::SeamSession::ExpertParallel(
                    self.model.vulkan_expert_parallel_session_on(&devices, ctx)?,
                ));
                return Ok(());
            }
            // LAYER-SPLIT session: `INFR_PIPELINE=Vulkan0,Vulkan2` opens ONE session spanning
            // cards, each keeping its own layers' weights + KV. The multi-device shape `serve`
            // never had — the SAME placement `infr run` uses, held open across turns.
            let devices =
                crate::seam::pipeline_devices(self.model.engine_cfg()).map(<[usize]>::to_vec);
            if let Some(devices) = devices {
                // ponytail: an explicit `INFR_CTX` is honoured verbatim; a percentage falls back
                // to the single-device default clamp (add a per-device fraction when asked for).
                let ctx = match user_ctx {
                    Some(infr_core::SizeSpec::Bytes(ctx)) => Some(ctx as usize),
                    _ => None,
                };
                self.session = Some(crate::seam::model::SeamSession::Pipeline(
                    self.model.vulkan_pipeline_session_on(&devices, ctx)?,
                ));
                return Ok(());
            }
            self.session = Some(crate::seam::model::SeamSession::Single(match user_ctx {
                Some(infr_core::SizeSpec::Bytes(ctx)) => {
                    self.model.vulkan_session_on(self.dev, ctx as usize)?
                }
                Some(infr_core::SizeSpec::Percent(f)) => {
                    self.model.vulkan_session_frac_on(self.dev, f)?
                }
                None => self.model.vulkan_session_default_on(self.dev)?,
            }));
        }
        Ok(())
    }

    /// Lazily construct the shared MTP Vulkan backend (see [`mtp_vk`](Self::mtp_vk)'s doc) —
    /// `generate`'s MTP branch calls this instead of letting `crate::mtp::generate_mtp_spec_vulkan`
    /// construct its own per-call backend.
    fn ensure_mtp_backend(&mut self) -> Result<()> {
        if self.mtp_vk.is_none() {
            let cfg = self.model.cfg().clone();
            self.mtp_vk = Some(match self.dev {
                Some(idx) => infr_vulkan::VulkanBackend::new_on_with(idx, cfg)
                    .map_err(|e| anyhow::anyhow!("vulkan init (Vulkan{idx}): {e}"))?,
                None => infr_vulkan::VulkanBackend::new_with(cfg)
                    .map_err(|e| anyhow::anyhow!("vulkan init: {e}"))?,
            });
        }
        Ok(())
    }

    fn wants_qwen4_mtp(&self, req: Option<&crate::sampling::RequestCtx>) -> Result<bool> {
        if !self.model.config().qwen4exp || !self.model.engine_cfg().spec.mtp {
            return Ok(false);
        }
        if self.model.engine_cfg().spec.draft.is_none() {
            anyhow::bail!("Qwen3.8 MTP requires a sidecar path (`INFR_SPEC_DRAFT` / `spec.draft`)");
        }
        let base = crate::sampling::Sampler::from_cfg(&self.model.engine_cfg().sampling);
        let effective = crate::sampling::Sampler::resolve(req, &self.model.engine_cfg().sampling);
        let greedy = |sampler: crate::sampling::Sampler| sampler.temp <= 0.0 || sampler.top_k == 1;
        let neutral_penalties = req.is_none_or(|ctx| !ctx.sampling().penalties_active());
        if !greedy(base) || !greedy(effective) || !neutral_penalties {
            tracing::warn!("Qwen3.8 MTP v1 is greedy-only; using ordinary decode for this request");
            return Ok(false);
        }
        Ok(true)
    }

    fn ensure_qwen4_mtp(&mut self) -> Result<()> {
        self.ensure_session()?;
        if self.qwen4_mtp.is_none() {
            let sidecar = self
                .model
                .engine_cfg()
                .spec
                .draft
                .as_deref()
                .expect("wants_qwen4_mtp checked the sidecar");
            let train_ctx = self.model.config().n_ctx_train;
            let max_ctx = super::cfg_ctx(self.model.engine_cfg(), train_ctx).unwrap_or(train_ctx);
            let be = self
                .session
                .as_ref()
                .expect("ensure_session set it")
                .backend();
            self.qwen4_mtp = Some(crate::mtp::Qwen4MtpRuntime::new_backend(
                be,
                &self.model,
                sidecar,
                max_ctx,
            )?);
        }
        Ok(())
    }

    fn generate_turn_impl(
        &mut self,
        prompt: &str,
        checkpoint_prefixes: crate::seam::TurnCheckpointPrefixes<'_>,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.wants_qwen4_mtp(req)? {
            self.ensure_qwen4_mtp()?;
            let be = self
                .session
                .as_ref()
                .expect("ensure_qwen4_mtp set it")
                .backend();
            let runtime = self.qwen4_mtp.as_mut().expect("ensure_qwen4_mtp set it");
            return runtime
                .generate_backend(be, &self.model, prompt, max_new, req, |p| on_piece(p))
                .map(|(stats, _)| stats);
        }
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
        self.model
            .generate_seam_session_turn_with_checkpoints_constrained(
                self.session.as_mut().unwrap(),
                prompt,
                max_new,
                checkpoint_prefixes,
                None,
                req,
                |p| on_piece(p),
            )
    }

    pub fn generate_serve_turn(
        &mut self,
        prompt: &str,
        checkpoint_prefixes: crate::seam::TurnCheckpointPrefixes<'_>,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if let Some(constraint) = constraint {
            self.ensure_session()?;
            return self
                .model
                .generate_seam_session_turn_with_checkpoints_constrained(
                    self.session.as_mut().unwrap(),
                    prompt,
                    max_new,
                    checkpoint_prefixes,
                    Some(constraint),
                    req,
                    |piece| on_piece(piece),
                );
        }
        self.generate_turn_impl(prompt, checkpoint_prefixes, max_new, req, on_piece)
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
        if let Some(s) = self.session.as_mut() {
            s.reset_cache();
        }
        if let Some(runtime) = self.qwen4_mtp.as_mut() {
            runtime.reset();
        }
    }

    fn warmup(&mut self) -> Result<()> {
        // The shared session warmup (throwaway generate + reset so the first real prompt prefills
        // clean slots from row 0), wrapped in the INFR_PROF_OPS suppression the Vulkan recorders need.
        crate::with_profiling_suppressed(|| {
            self.generate_turn_impl(
                "Hi",
                crate::seam::TurnCheckpointPrefixes::edit(Some("")),
                2,
                None,
                &mut |_| {},
            )?;
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
        self.generate_turn_impl(
            prompt,
            crate::seam::TurnCheckpointPrefixes::default(),
            max_new,
            req,
            on_piece,
        )
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
        self.generate_turn_impl(
            prompt,
            crate::seam::TurnCheckpointPrefixes::edit(stable_prefix),
            max_new,
            req,
            on_piece,
        )
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
        self.model
            .generate_seam_session_turn_with_checkpoints_constrained(
                self.session.as_mut().unwrap(),
                prompt,
                max_new,
                crate::seam::TurnCheckpointPrefixes::edit(stable_prefix),
                Some(constraint),
                req,
                |p| on_piece(p),
            )
    }

    fn generate_with_checkpoints(
        &mut self,
        prompt: &str,
        checkpoint_prefixes: crate::seam::TurnCheckpointPrefixes<'_>,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_serve_turn(
            prompt,
            checkpoint_prefixes,
            max_new,
            constraint,
            req,
            on_piece,
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
