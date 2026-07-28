//! [`DiffusionGemmaChat`]: the diffusion-gemma [`ChatModel`], over a persistent per-backend
//! session ([`DiffusionSess`]) selected by [`DgBackend`] at construction.

use super::ChatModel;
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// Either DiffusionGemma session (Phase 2/D, `seam/model.rs`) behind
/// [`crate::diffusion::DiffusionSession`] — lets [`DiffusionGemmaChat`] hold ONE persistent
/// session across turns regardless of backend.
enum DiffusionSess {
    // Boxed so the enum's size is a pointer regardless of which session it holds — the three
    // session structs differ enough in size to trip clippy's `large_enum_variant` (the
    // `#[cfg(macos)]` Metal variant makes the spread, so the check only fires on a macOS build —
    // the Linux clippy CI job never compiles it).
    Cpu(Box<crate::seam::model::DiffusionGemmaCpuSession>),
    Vulkan(Box<crate::seam::model::DiffusionGemmaVulkanSession>),
    /// Phase D: the Metal twin, macOS only (see `DiffusionGemmaChat::new_metal`).
    #[cfg(target_os = "macos")]
    Metal(Box<crate::seam::model::DiffusionGemmaMetalSession>),
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl crate::diffusion::DiffusionSession for DiffusionSess {
    fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        match self {
            DiffusionSess::Cpu(s) => s.prefill(model, tokens),
            DiffusionSess::Vulkan(s) => s.prefill(model, tokens),
            #[cfg(target_os = "macos")]
            DiffusionSess::Metal(s) => s.prefill(model, tokens),
        }
    }
    fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
        sample_temp_inv: f32,
        u: &[f32],
    ) -> Result<crate::seam::DenoiseOutcome> {
        use crate::seam::DenoiseOutcome;
        match self {
            DiffusionSess::Cpu(s) => Ok(DenoiseOutcome::Logits(s.denoise(
                model,
                canvas_tokens,
                sc_logits,
                temp_inv,
            )?)),
            DiffusionSess::Vulkan(s) => s.denoise(
                model,
                canvas_tokens,
                sc_logits,
                temp_inv,
                sample_temp_inv,
                Some(u),
            ),
            #[cfg(target_os = "macos")]
            DiffusionSess::Metal(s) => Ok(DenoiseOutcome::Logits(s.denoise(
                model,
                canvas_tokens,
                sc_logits,
                temp_inv,
            )?)),
        }
    }
}

/// Which backend [`DiffusionGemmaChat`] opens its session on — decided at construction, before a
/// session is ever opened (mirrors `DiffusionSess`'s three variants).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DgBackend {
    Vulkan,
    Cpu,
    Metal,
}

/// diffusion-gemma (Phase 3/D — block text-diffusion, see `docs/diffusion-gemma.md` and
/// `crate::diffusion`): the entropy-bound decode loop over a persistent session, Vulkan by
/// default, or the CPU/Metal reference backends under `INFR_CPU`/`INFR_METAL` (Phase D added the
/// Metal DG session — macOS only; the non-macOS build still compiles `new_metal`, `generate`
/// errors clearly at runtime instead, matching every other INFR_METAL backend on this crate). The
/// session is opened lazily on the first turn (its KV cache is sized once the model's
/// `n_ctx_train`/`INFR_CTX` is known) and stays open across turns: multi-turn REPL re-sends
/// the WHOLE running token stream as the "prefix" each turn, and the session's own prefix-diff
/// prefill (see `DiffusionGemmaCpuSession::prefill`'s doc) re-sends only the un-cached suffix,
/// exactly like every other seam session on this crate.
pub struct DiffusionGemmaChat {
    model: SeamModel,
    backend: DgBackend,
    sess: Option<DiffusionSess>,
    max_ctx: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaChat {
    /// Production Vulkan session.
    pub fn new(model: SeamModel) -> Self {
        Self {
            model,
            backend: DgBackend::Vulkan,
            sess: None,
            max_ctx: 0,
        }
    }

    /// Reference CPU session (`INFR_CPU=1`).
    pub fn new_cpu(model: SeamModel) -> Self {
        Self {
            model,
            backend: DgBackend::Cpu,
            sess: None,
            max_ctx: 0,
        }
    }

    /// Reference Metal session (`INFR_METAL=1`, Phase D). Compiles on every target (like
    /// `CpuDenseChat::new_metal`) — the macOS check happens in `generate`, at session-open time.
    pub fn new_metal(model: SeamModel) -> Self {
        Self {
            model,
            backend: DgBackend::Metal,
            sess: None,
            max_ctx: 0,
        }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaChat {
    /// Open (or grow) the session for a turn needing `needed` KV rows — see the sizing comment at
    /// the call site in [`generate`](ChatModel::generate).
    fn ensure_sess(&mut self, needed: usize) -> Result<()> {
        if self.sess.is_some() && needed <= self.max_ctx {
            return Ok(());
        }
        let cfg = self.model.config();
        // `device.ctx` (INFR_CTX) shared size grammar; % resolves against the trained context (DG
        // sessions size against the canvas/prompt shape, not a VRAM-fit calc).
        let max_ctx = super::cfg_ctx(self.model.engine_cfg(), cfg.n_ctx_train)
            .unwrap_or_else(|| cfg.n_ctx_train.min(8192))
            .max(needed);
        self.max_ctx = max_ctx;
        self.sess = Some(match self.backend {
            DgBackend::Cpu => {
                DiffusionSess::Cpu(Box::new(self.model.diffusion_gemma_cpu_session(max_ctx)))
            }
            DgBackend::Vulkan => DiffusionSess::Vulkan(Box::new(
                self.model.diffusion_gemma_vulkan_session(max_ctx)?,
            )),
            DgBackend::Metal => {
                #[cfg(target_os = "macos")]
                {
                    DiffusionSess::Metal(Box::new(
                        self.model.diffusion_gemma_metal_session(max_ctx)?,
                    ))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    return Err(anyhow::anyhow!(
                        "the Metal backend is only available on macOS"
                    ));
                }
            }
        });
        Ok(())
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for DiffusionGemmaChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn warmup(&mut self) -> Result<()> {
        use crate::diffusion::DiffusionSession;
        // A tiny PREFILL-only forward compiles the lazily-built pipelines (GEMM/attention/batched
        // MoE — a cold DG prefill was measured at 26 t/s vs 1424 t/s warm, ~5s of one-time compile
        // otherwise billed to the first request). No denoise: the canvas plans build per (cc, p)
        // anyway, and their couple of extra pipelines (canvas attention, softmax) are cheap next
        // to the shared set warmed here. The throwaway tokens pollute the session's cached prefix
        // harmlessly — a real prompt prefix-diffs to 0 and re-prefills from scratch.
        crate::with_profiling_suppressed(|| -> Result<()> {
            self.ensure_sess(64)?;
            let enc = self
                .model
                .tokenizer()
                .encode("Hi", false)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            self.sess
                .as_mut()
                .unwrap()
                .prefill(&self.model, enc.get_ids())
        })
    }

    /// `_req` is ACCEPTED AND IGNORED, which is exactly what this backend did before the
    /// per-sequence conversion too — and is therefore not a regression, but it IS a gap.
    ///
    /// DiffusionGemma does not sample in the autoregressive decode loop at all: its tokens come out
    /// of the block-diffusion denoise (`crate::diffusion::diffusion_generate`, `dg_eb_sample`),
    /// which has its OWN sampler and never consults `Sampler::resolve`. So the old thread-local
    /// never reached it either — a `temperature`/`seed`/`stop` on a DG serve request was already a
    /// no-op. Wiring per-request sampling (and the stop-sequence abort latch) into the denoise loop
    /// is a separate piece of work; it is reported, not silently papered over.
    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        _req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_impl(prompt, max_new, on_piece, None)
    }

    fn generate_with_step_hook(
        &mut self,
        prompt: &str,
        max_new: usize,
        _req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
        on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.generate_impl(prompt, max_new, on_piece, on_step)
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaChat {
    /// Shared body of [`ChatModel::generate`]/[`ChatModel::generate_with_step_hook`] — `on_step:
    /// None` (the plain `generate` path) is byte-identical to the pre-hook implementation: the
    /// hook is only ever read inside [`crate::diffusion::diffusion_generate`]'s step loop, and a
    /// `None` there is a single `if let` check with no other side effect.
    ///
    /// Streams each block's COMMITTED text through `on_piece` as soon as that block finishes
    /// (`diffusion_generate`'s `on_block`), rather than detokenizing the whole reply once
    /// `diffusion_generate` returns — this is what lets `INFR_DIFFUSION_VISUAL` print a finished
    /// block permanently while the next one is still denoising. The emitted BYTES are unchanged
    /// either way: `on_block` fires with exactly the same tokens, in the same order, that used to
    /// be detokenized in one pass over `result.tokens` afterward — only the timing/chunking of
    /// `on_piece` calls changes, never the concatenated text.
    fn generate_impl(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
        on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        let enc = self
            .model
            .tokenizer()
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();

        let cfg = self.model.config();
        let canvas_len = cfg.canvas_length;
        let vocab = cfg.vocab;
        let eos_ids = cfg.eos_ids.clone();
        let eb = crate::diffusion::EbConfig::from_config(cfg);
        // Seed determinism (see `crate::diffusion::diffusion_generate`'s doc): the reference
        // reseeds its RNG from a fixed value every block, so a fixed `sampling.seed` (INFR_SEED;
        // default 42 HERE — §6.12's two-defaults knob, both kept at their own site — matching the
        // oracle's `-s 42`) makes every turn reproducible.
        let seed: u64 = self.model.engine_cfg().sampling.seed.unwrap_or(42);

        // Size the session to THIS turn's [prompt | every block's canvas] plus REPL headroom —
        // NOT n_ctx_train: DG's per-token KV is heavy (hd 256/512 across 30 layers ≈ 225 KB/tok)
        // and this model trains at 262144 ctx, so the n_ctx_train default every AR chat uses
        // would ask the backend for a ~59 GB KV cache (observed: radv device-lost at submit).
        // Default headroom is min(n_ctx_train, 8192) ≈ 1.8 GB — the same clamp the spec-decode
        // pair uses. A later REPL turn that outgrows the session reopens it bigger (the KV is
        // rebuilt by a from-scratch prefill; correct, just slower for that one turn).
        let blocks = max_new.div_ceil(canvas_len.max(1)).max(1);
        let needed = prompt_tokens.len() + blocks * canvas_len + 64;
        self.ensure_sess(needed)?;

        // Stream each block's committed tokens through the shared incremental UTF-8-safe detok —
        // the same helper every other backend's per-token decode loop uses (`crate::stream_token`)
        // — as soon as `diffusion_generate`'s `on_block` hands them over, so a block's text
        // appears as soon as it's decided rather than only after the whole turn finishes. `acc`/
        // `printed` persist across every block (not reset per block), same as they would across
        // every token of a single flat loop: a multi-byte char split across a block boundary is
        // held back exactly like one split across two tokens of the same block.
        let tokenizer = self.model.tokenizer();
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let mut on_block = |committed: &[u32]| {
            // The whole block's ids are in hand, so decode the committed span in ONE pass through
            // the shared incremental UTF-8-safe detok — NOT once per id over the growing `acc`
            // (that was O(total²): every token re-decoded the entire accumulator). Same holdback
            // semantics, same emitted bytes; only the per-token redundant decode is gone.
            crate::stream_tokens(
                tokenizer,
                &mut acc,
                &mut printed,
                committed,
                &mut |p: &str| on_piece(p),
            );
        };

        let result = crate::diffusion::diffusion_generate(
            self.sess.as_mut().unwrap(),
            &self.model,
            &prompt_tokens,
            canvas_len,
            vocab,
            &eos_ids,
            &eb,
            max_new,
            seed,
            self.max_ctx,
            on_step,
            Some(&mut on_block),
        )?;

        Ok(result.stats)
    }
}
