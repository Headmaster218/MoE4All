//! Vulkan-backed [`ChatModel`]: [`DenseSeamChat`] (dense/MoE — and, since the phase-3 cutover,
//! qwen35 too — on the Vulkan agnostic seam with a persistent KV session).

use super::ChatModel;
use crate::{GenStats, ImageSpanEmbeds, MropePlan, SeamModel};
use anyhow::{Context, Result};

/// The `<|image_pad|>` special token (type-3 atomic): the rendered prompt carries ONE per image
/// (the template marker `infr_chat` appends, see `template::VISION_MARKER`); the vision
/// expansion pass below fans each marker out to `nx*ny` copies — one per merged ViT token —
/// paired with the span's [`MropePlan`].
const IMAGE_PAD: u32 = 248_056;

/// Locate the mmproj (vision projector) GGUF beside `model_path` — the same discovery rule as
/// `infr-gui`'s `catalog::find_projector`: a file named `mmproj*.gguf` (case-insensitive) in the
/// model file's parent directory, with an exact `mmproj.gguf` winning over any suffixed variant.
/// `None` = the model directory ships no projector.
fn find_mmproj(model_path: &std::path::Path) -> Option<std::path::PathBuf> {
    let dir = model_path.parent()?;
    let is_mmproj = |n: &str| {
        let n = n.to_ascii_lowercase();
        n.starts_with("mmproj") && n.ends_with(".gguf")
    };
    let exact = dir.join("mmproj.gguf");
    if exact.is_file() {
        return Some(exact);
    }
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| is_mmproj(n))
        .collect();
    names.sort();
    names.first().map(|n| dir.join(n))
}

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
    /// `generate()` calls — on the SAME Vulkan backend as `session` (the MTP branch drives
    /// [`crate::mtp::generate_mtp_spec_vulkan_timed_on_state`] with `session`'s backend, pins, and
    /// `max_ctx`). The bound trunk weights + pager registration persist turn to turn (the cold
    /// call binds through the paged-MoE placement binder); only the KV rows are reset per call
    /// (`crate::mtp`'s "no cross-turn KV reuse" doc). `None` until the first MTP turn; stays
    /// `None` forever on a non-MTP model.
    mtp_trunk: Option<crate::seam::SeamKv>,
    /// Physical device this chat's session pins: `Some(idx)` = `VulkanN` (the multi-device path,
    /// `new_on`), `None` = the default device (`new`, byte-identical to before). Threaded into
    /// [`ensure_session`](Self::ensure_session) so the whole model — weights, KV, MTP trunk/head —
    /// lands on the one chosen GPU.
    dev: Option<usize>,
    /// Vision (stage V5): the CPU ViT tower, loaded LAZILY on the first [`generate_mm`] —
    /// dequantizing an mmproj costs seconds + ~0.9 GB host RAM, so a process that never serves an
    /// image request never pays it. `Some` also on the `--mmproj` override path.
    vision: Option<std::sync::Arc<infr_vision::VitEngine>>,
    /// Whether a mmproj discovery/load was already ATTEMPTED (memoizing the failure: once true
    /// with `vision` still `None`, later image requests fail fast instead of rescanning the
    /// model directory every turn).
    vision_checked: bool,
    /// Explicit mmproj path (`infr serve/run --mmproj`) overriding the beside-the-model
    /// discovery. `None` = discover lazily (default, zero startup cost).
    mmproj_override: Option<std::path::PathBuf>,
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
            vision: None,
            vision_checked: false,
            mmproj_override: None,
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
            vision: None,
            vision_checked: false,
            mmproj_override: None,
        }
    }

    /// Pin the vision projector explicitly (`infr serve/run --mmproj PATH`), overriding the lazy
    /// beside-the-model `mmproj*.gguf` discovery. Builder-style; unset leaves discovery on.
    pub fn with_mmproj_override(mut self, p: Option<std::path::PathBuf>) -> Self {
        self.mmproj_override = p;
        self
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
            // path uses — the first (cold) call binds the trunk weights via the placement
            // planner/pager, later calls reuse them through `mtp_trunk`. In MTP mode `session`
            // itself never generates, so this backend carries exactly one trunk weight upload.
            self.ensure_session()?;
            let session = self.session.as_ref().expect("ensure_session set it");
            // Placement/pager decisions read the CURRENT placement pins — enter this session's
            // own scope around the MTP call, exactly like the non-MTP generation entry does.
            let _scope = crate::seam::PlacementScope::enter(session.pins().clone());
            // The trunk's KV was sized at the session's max_ctx on the cold call, but the cold
            // init may have re-clamped it against live free VRAM — once bound, the trunk's own
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

    /// Lazily resolve + load the vision tower (first [`generate_mm`] only — see the `vision` field
    /// doc). Returns a CLONE of the cached `Arc`. Memoizes the failure via `vision_checked` so a
    /// projector-less model doesn't rescan its directory on every image request.
    fn vision(&mut self) -> Result<std::sync::Arc<infr_vision::VitEngine>> {
        if let Some(v) = &self.vision {
            return Ok(v.clone());
        }
        if self.vision_checked {
            anyhow::bail!(
                "no usable mmproj for this model (discovery/load already failed once); pass \
                 --mmproj <PATH> to point at the vision projector explicitly"
            );
        }
        self.vision_checked = true;
        let path = match &self.mmproj_override {
            Some(p) => p.clone(),
            None => {
                // The model file's own path (shard 0 of the set) — discovery looks BESIDE it.
                let model_path = self
                    .model
                    .gguf()
                    .shards()
                    .first()
                    .map(|(p, _)| p.to_path_buf())
                    .context("model GGUF has no backing file")?;
                find_mmproj(&model_path).context("no mmproj.gguf beside the model")?
            }
        };
        // Stage V7: try the Vulkan ViT (native-f16 weights, GPU graph) first; ANY failure —
        // backend init, unsupported mmproj dtype, op-lowering gap — falls back to the CPU tower
        // with the reason logged (the CPU forward is the correctness oracle and always works).
        // The tower is pinned to THIS chat's device (`self.dev`, the `VulkanN` the model was
        // opened on) so it never lands on a different GPU in multi-device / iGPU+dGPU machines
        // (PR #21 review fix).
        let engine = match infr_vision::VitEngine::new_vulkan_on(&path, self.dev) {
            Ok(vk) => std::sync::Arc::new(vk),
            Err(e) => {
                tracing::warn!("Vulkan vision tower unavailable, using CPU ViT: {e:#}");
                std::sync::Arc::new(infr_vision::VitEngine::new_cpu(&path)?)
            }
        };
        self.vision = Some(engine.clone());
        Ok(engine)
    }

    /// The vision expansion pass: encode every image through the ViT, then rewrite the encoded
    /// prompt token array — each `<|image_pad|>` marker becomes `nx*ny` pad tokens, and every
    /// token's (T,H,W,E) mrope position is recorded into `prompt_pos4` (text: the running cursor
    /// on T/H/W; image token `i` of a span based at `base`: `H = base + i/nx`, `W = base + i%nx`).
    /// The cursor advances by `max(nx, ny)` across a span so the text AFTER an image never shares
    /// a position with it. Returns the expanded ids + the [`MropePlan`] the seam consumes.
    fn expand_prompt(
        &self,
        tokens: &[u32],
        imgs: &[(std::sync::Arc<Vec<f32>>, usize, usize)],
    ) -> Result<(Vec<u32>, MropePlan)> {
        let mut expanded: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut pos4: Vec<i32> = Vec::with_capacity(tokens.len() * 4);
        let mut spans: Vec<ImageSpanEmbeds> = Vec::new();
        let mut cursor: i32 = 0;
        let mut next_img = 0usize;
        for &t in tokens {
            if t == IMAGE_PAD && next_img < imgs.len() {
                let (embeds, nx, ny) = &imgs[next_img];
                let (nx, ny) = (*nx, *ny);
                next_img += 1;
                let start = expanded.len(); // span start, BEFORE pushing
                let base = cursor;
                let n = nx * ny;
                for i in 0..n {
                    expanded.push(IMAGE_PAD);
                    pos4.extend_from_slice(&[
                        base,
                        base + (i / nx) as i32,
                        base + (i % nx) as i32,
                        0,
                    ]);
                }
                spans.push(ImageSpanEmbeds {
                    start,
                    n_tokens: n,
                    embeds: embeds.clone(),
                });
                cursor += nx.max(ny) as i32;
            } else {
                expanded.push(t);
                pos4.extend_from_slice(&[cursor, cursor, cursor, 0]);
                cursor += 1;
            }
        }
        if next_img < imgs.len() {
            anyhow::bail!(
                "the request carries {} image(s) but the rendered prompt contains only {} \
                 <|image_pad|> marker(s) — the model's chat template did not emit one per image",
                imgs.len(),
                next_img
            );
        }
        Ok((
            expanded,
            MropePlan {
                prompt_pos4: pos4,
                spans,
                decode_base: cursor,
            },
        ))
    }

    fn generate_mm_impl(
        &mut self,
        prompt: &str,
        images: &[String],
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        // No images → the EXACT text path (no KV reset, no mm plan) — byte-identical to pre-V5.
        if images.is_empty() {
            return self.generate_turn_impl(prompt, None, max_new, req, on_piece);
        }
        // The vision mrope turn rides the qwen35/qwen35moe IMROPE emission (`Op::QkNormMrope`,
        // seam/runner.rs) — v1 supports nothing else.
        if !self.model.config().qwen35 {
            anyhow::bail!("vision only supported for qwen35moe (IMROPE) models in v1");
        }
        let vision = self.vision()?;
        let n_embd = self.model.config().n_embd;
        // Decode + encode every image IN ORDER, shape-checking against the LM's n_embd: the span
        // rows overwrite token-embedding rows during prefill, so a projector/LM mismatch would
        // corrupt the forward silently if it got this far.
        let mut imgs: Vec<(std::sync::Arc<Vec<f32>>, usize, usize)> =
            Vec::with_capacity(images.len());
        for (idx, img) in images.iter().enumerate() {
            let bytes = infr_vision::decode_image_input(img)
                .with_context(|| format!("image #{idx}: decode"))?;
            let (embeds, nx, ny) = vision
                .encode_image_bytes_with_grid(&bytes)
                .with_context(|| format!("image #{idx}: encode"))?;
            let n_tokens = nx * ny;
            if embeds.len() != n_tokens * n_embd {
                anyhow::bail!(
                    "image #{idx}: vision encoder produced {} floats for {} tokens (expected \
                     {} = n_tokens × n_embd) — mmproj/LM projection mismatch",
                    embeds.len(),
                    n_tokens,
                    n_tokens * n_embd
                );
            }
            imgs.push((std::sync::Arc::new(embeds), nx, ny));
        }
        let tokens = self.model.encode(prompt)?;
        let (expanded_ids, mm) = self.expand_prompt(&tokens, &imgs)?;
        tracing::info!(
            "[vision] mm plan: {} images, {} spans, tokens {} → {}, decode_base {}",
            imgs.len(),
            mm.spans.len(),
            tokens.len(),
            expanded_ids.len(),
            mm.decode_base
        );
        // v1 prefix-cache safety: a vision turn's expanded token array differs in LENGTH from any
        // rendered text turn, so a stale common-prefix rew could splice a pad run onto text KV
        // rows. Full re-prefill instead.
        self.reset_kv();
        self.ensure_session()?;
        self.model.generate_vulkan_session_turn_tokens_on(
            self.session.as_mut().unwrap(),
            &expanded_ids,
            None, // stable_prefix: a vision turn never reuses a text-turn prefix (reset above)
            max_new,
            None, // constraint: unconstrained decode (forced tool_choice + vision is a v1 non-goal)
            req,
            Some(&mm),
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
        // registration stay bound — the next MTP turn re-prefills into the same session).
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

    fn generate_mm(
        &mut self,
        prompt: &str,
        images: &[String],
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_mm_impl(prompt, images, max_new, req, on_piece)
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
