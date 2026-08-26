//! Agnostic chat orchestration: ONE shared [`Chat`] over a per-backend [`ChatModel`], so every
//! architecture runs the identical multi-turn REPL instead of 5 bespoke chat loops.
//!
//! Each backend keeps its own bespoke forward (dense `ChatSession` with a persistent KV cache, the
//! eager `generate_moe`, qwen35's per-token loop, the CPU compute-graph runner). A backend only
//! implements two primitives — [`render`](ChatModel::render) a message list through the model's OWN
//! chat template, and [`generate`](ChatModel::generate) a reply for an already-rendered prompt. The
//! shared [`Chat`] owns the conversation history, re-renders it every turn, strips the model's
//! `<think>` reasoning from what it stores, and reports [`GenStats`]. Turn orchestration lives in
//! ONE place; only the two primitives differ per architecture.
//!
//! The boxed trait object is lifetime-bounded (`Box<dyn ChatModel + '_>`) so the borrow-based dense
//! `ChatSession` (which borrows `&Llama`) needs no ownership change — the caller owns the `Llama`,
//! the box borrows it.
//!
//! Per-backend [`ChatModel`] impls live one file per backend (module split): [`vulkan`] (dense +
//! qwen35 on the Vulkan seam), [`metal`] (macOS-only dense + speculative), [`cpu`] (the CPU/Metal
//! reference dense path), [`diffusion`] (diffusion-gemma). This module keeps the agnostic core —
//! the trait, the shared REPL, and the OpenAI-shaped renderer.

use crate::{no_template_err, GenStats};
use anyhow::Result;

mod cpu;
mod diffusion;
mod metal;
mod vulkan;

pub use cpu::CpuDenseChat;
pub use diffusion::DiffusionGemmaChat;
#[cfg(target_os = "macos")]
pub use metal::{MetalSeamChat, SpecMetalChat};
pub use vulkan::DenseSeamChat;

/// The two arch-specific primitives the shared [`Chat`] drives. Object-safe: no generics, callbacks
/// are `&mut dyn FnMut`. A stateful backend (dense) may keep a KV cache warm across `generate` calls;
/// stateless ones prefill the whole prompt each turn.
pub trait ChatModel {
    /// The [`SeamModel`](crate::SeamModel) whose embedded chat template renders this backend's
    /// prompts. Backends that keep the default [`render`](Self::render) implement this one-liner
    /// instead of repeating the render body (every real backend funnels to the same
    /// `SeamModel::render_chat_messages`). Backends that override `render` — the test mocks, which
    /// own no `SeamModel` — never call it; the default panics so a missing override is a loud bug,
    /// not a silent wrong render.
    fn render_model(&self) -> &crate::SeamModel {
        unreachable!("ChatModel::render_model must be implemented when using the default render()")
    }

    /// Render a conversation `(role, content)` into a prompt string via the model's embedded chat
    /// template. All backends funnel to `infr_chat::render_chat_jinja` (the single prompt renderer).
    /// The provided default routes through [`render_model`](Self::render_model) —
    /// `SeamModel::render_chat_messages` — so a backend names its field once rather than repeating
    /// this body; override only where the render genuinely differs (the test mocks do).
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.render_model().render_chat_messages(messages)
    }

    /// Optional stable history prefix for stateful recurrent backends. Stateless backends and
    /// test doubles keep `None`; Vulkan seam chat renders the same messages without opening the
    /// assistant generation turn.
    fn render_stable_prefix(&self, _messages: &[(&str, &str)]) -> Result<Option<String>> {
        Ok(None)
    }

    /// Generate a completion for the already-rendered `prompt`, streaming decoded text to `on_piece`.
    ///
    /// `req` is the in-flight SEQUENCE's own state (`infr serve`): its sampling overrides, its
    /// stop-sequence abort latch, and its turn on the GPU baton. `None` — `infr run`, `bench`, every
    /// test — means "inherit the process default", i.e. exactly the pre-existing behavior. It is an
    /// explicit parameter rather than ambient state precisely because a server steps several
    /// sequences at once; see [`crate::sampling::RequestCtx`].
    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats>;

    /// Like [`generate`](Self::generate), plus an optional per-denoise-step observable view for
    /// architectures with an internal multi-step decode inside ONE `generate` call (currently only
    /// diffusion-gemma's block-diffusion loop — [`crate::diffusion::StepView`], driving
    /// `infr run`'s `INFR_DIFFUSION_VISUAL` live canvas view). Default: ignore the hook and
    /// delegate to `generate` — every autoregressive backend already streams token-by-token via
    /// `on_piece` and has no internal notion of a "step", so this default keeps every existing
    /// `ChatModel` impl byte-identical with zero code change.
    fn generate_with_step_hook(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
        _on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.generate(prompt, max_new, req, on_piece)
    }

    /// Turn-aware generation hook carrying the stable rendered-history prefix. The default keeps
    /// every stateless/non-recurrent backend byte-identical by ignoring it.
    fn generate_turn_with_step_hook(
        &mut self,
        prompt: &str,
        _stable_prefix: Option<&str>,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
        on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.generate_with_step_hook(prompt, max_new, req, on_piece, on_step)
    }

    /// Constrained turn-aware counterpart used by OpenAI forced tool calls.
    fn generate_constrained_turn(
        &mut self,
        prompt: &str,
        _stable_prefix: Option<&str>,
        max_new: usize,
        constraint: &mut crate::grammar::Constraint,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_constrained(prompt, max_new, constraint, req, on_piece)
    }

    /// Optional REPL status (e.g. dense returns `ctx N/MAX`); `None` for stateless backends.
    fn status(&self) -> Option<String> {
        None
    }

    /// Drop any persistent KV-session state so the NEXT `generate` re-prefills its prompt from
    /// scratch. The serve-side forced-tool fallback uses this: after a constrained attempt has
    /// advanced the session past the primed `<tool_call>` opener, resetting before the unconstrained
    /// retry guarantees the retry can never inherit that divergent state (rather than relying on the
    /// seam's common-prefix rewind to unwind it). Default: no-op — stateless backends (CPU/Metal
    /// reference) re-prefill the full prompt every call and have nothing to reset.
    fn reset_kv(&mut self) {}

    /// Run a tiny throwaway generation through the real forward so every lazily-built pipeline
    /// compiles NOW instead of on the first user request (a cold Vulkan seam pays seconds of
    /// pipeline builds otherwise). INFR_PROF_OPS is suppressed for the duration — recorders read it
    /// at construction, and warmup submits would pollute a later bench's per-op aggregate.
    /// Default: no-op (stateless/CPU backends have nothing to warm).
    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }

    /// The warmup body every session-backed backend (Vulkan/Metal) copy-pasted: a two-token
    /// throwaway generation to force the lazy session open + every pipeline compiled, then
    /// [`reset_kv`](Self::reset_kv) so the warmup tokens are dropped and the first REAL prompt
    /// prefills clean slots from row 0 instead of forking off a garbage prefix.
    ///
    /// Deliberately NOT the [`warmup`](Self::warmup) default — that stays a no-op so the stateless
    /// backends and the test mocks are unaffected. A session-backed backend opts in with
    /// `fn warmup(&mut self) -> Result<()> { self.warmup_session() }` (Vulkan wraps the call in
    /// `with_profiling_suppressed`; Metal deliberately does not — see that helper's doc).
    fn warmup_session(&mut self) -> Result<()> {
        self.generate("Hi", 2, None, &mut |_| {})?;
        self.reset_kv();
        Ok(())
    }

    /// Like [`generate`](Self::generate), with an llguidance grammar constraint applied to the
    /// decode (serve's FORCED tool_choice). Backends without constraint support return an error —
    /// the caller falls back to unconstrained generation.
    fn generate_constrained(
        &mut self,
        _prompt: &str,
        _max_new: usize,
        _constraint: &mut crate::grammar::Constraint,
        _req: Option<&crate::sampling::RequestCtx>,
        _on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        Err(anyhow::anyhow!(
            "grammar-constrained generation is not supported by this backend"
        ))
    }
}

/// Drop a lazily-opened seam session's materialized tokens, if it has been opened yet — the body
/// every session-backed [`ChatModel::reset_kv`] (and warmup discard) copy-pasted. Generic over the
/// backend/extension pairing so ONE implementation serves Vulkan and Metal; a chat whose
/// session is still `None` has nothing to reset (its first `generate` opens a clean one).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn reset_session<B, X>(session: &mut Option<crate::seam::model::DenseSession<B, X>>) {
    if let Some(s) = session {
        s.reset_cache();
    }
}

/// The user's `device.ctx` (`INFR_CTX`) override in the shared size grammar (`8192`, `256k`,
/// `50%`), unresolved. Vulkan wants the raw [`infr_core::SizeSpec`] (it routes `Bytes`/`Percent`
/// to different VRAM-fit constructors); Metal resolves it through [`cfg_ctx`]. `None` = no
/// override.
///
/// Takes the borrowed engine config each chat already owns through its [`crate::SeamModel`] (S7) —
/// this used to read the `INFR_CTX` variable from the process environment, where an unparseable
/// value fell through to `None`; the env LAYER now drops it just the same (R5).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn cfg_ctx_spec(cfg: &crate::EngineConfig) -> Option<infr_core::SizeSpec> {
    cfg.device.ctx
}

/// [`cfg_ctx_spec`] resolved to a token count: a percentage resolves against the model's TRAINED
/// context, since these paths have no VRAM-fit calc to take a fraction of. `None` = no override —
/// the caller supplies its own default. Shared by the Metal chat and the diffusion-gemma chat,
/// whose own copy of the resolve this replaced.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn cfg_ctx(cfg: &crate::EngineConfig, n_ctx_train: usize) -> Option<usize> {
    cfg_ctx_spec(cfg).map(|s| s.resolve(n_ctx_train as u64) as usize)
}

/// Store only the ANSWER, dropping the model's reasoning (Qwen3 excludes prior-turn thinking;
/// keeping it degrades the model). Delegates to `infr-chat`'s splitter — the SAME reasoning
/// grammar `infr run`'s display and `infr serve`'s deltas use — so what history keeps always
/// matches what the surfaces call "content". Unterminated reasoning (truncated turn) → empty.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn strip_think(reply: &str) -> String {
    infr_chat::split_reasoning(reply).1
}

/// The SINGLE arch-agnostic chat: owns the conversation history + `<think>`-stripping, drives any
/// [`ChatModel`] backend. Every architecture uses this one turn implementation and REPL, so a working
/// multi-turn REPL is uniform (no per-backend "one-shot only" special cases).
pub struct Chat<'a> {
    model: Box<dyn ChatModel + 'a>,
    history: Vec<(String, String)>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<'a> Chat<'a> {
    /// Wrap a per-backend [`ChatModel`] in the shared multi-turn chat.
    pub fn new(model: Box<dyn ChatModel + 'a>) -> Self {
        Self {
            model,
            history: Vec::new(),
        }
    }

    /// Run one user turn: append the message, re-render the FULL history through the model's chat
    /// template, generate the reply (streamed to `on_piece`), strip the `<think>` reasoning from what
    /// we store, and keep the answer for the next turn. Returns per-turn [`GenStats`].
    pub fn turn(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.turn_with_step_hook(message, max_new, on_piece, None)
    }

    /// Like [`turn`](Self::turn), plus an optional per-denoise-step hook
    /// ([`crate::diffusion::StepView`]) forwarded to the backend's
    /// [`generate_with_step_hook`](ChatModel::generate_with_step_hook) — `infr run`'s
    /// `INFR_DIFFUSION_VISUAL` live canvas view is the only caller today. `on_step: None` behaves
    /// EXACTLY like `turn` (same default no-op path in every `ChatModel` impl).
    pub fn turn_with_step_hook(
        &mut self,
        message: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
        on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
    ) -> Result<GenStats> {
        self.history.push(("user".into(), message.to_string()));
        let (prompt, stable_prefix) = {
            let refs: Vec<(&str, &str)> = self
                .history
                .iter()
                .map(|(r, c)| (r.as_str(), c.as_str()))
                .collect();
            match self.model.render(&refs).and_then(|prompt| {
                self.model
                    .render_stable_prefix(&refs)
                    .map(|stable| (prompt, stable))
            }) {
                Ok(rendered) => rendered,
                Err(e) => {
                    self.history.pop();
                    return Err(e);
                }
            }
        };
        let mut answer = String::new();
        let mut emit = |p: &str| {
            answer.push_str(p);
            on_piece(p);
        };
        // Template-prefilled thinking (Qwen3.5-style: the PROMPT ends with the `<think>` opener,
        // so the output starts mid-reasoning): inject a synthetic opener so the display styler
        // and the history stripper see a well-formed block.
        if infr_chat::prompt_prefills_think(&prompt) {
            emit("<think>");
        }
        // `req: None` — the interactive REPL is a sole sequence: sampling comes from the env/CLI
        // defaults, there is no stop-sequence latch, and there is no GPU to share.
        //
        // A generate failure must roll the just-pushed user turn back, exactly like the render
        // failure above: leaving it in `history` with no assistant reply would make the NEXT turn
        // push a second consecutive `user` message, permanently corrupting the alternating-role
        // conversation state (the model re-renders with two user turns and no assistant between).
        let stats = match self.model.generate_turn_with_step_hook(
            &prompt,
            stable_prefix.as_deref(),
            max_new,
            None,
            &mut emit,
            on_step,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.history.pop();
                return Err(e);
            }
        };
        self.history
            .push(("assistant".into(), strip_think(&answer)));
        Ok(stats)
    }

    /// Every backend now supports the interactive multi-turn REPL (kept for the CLI's call site).
    pub fn supports_repl(&self) -> bool {
        true
    }

    /// Optional status line (e.g. `ctx 12/8192`) for the REPL prompt.
    pub fn repl_status(&self) -> Option<String> {
        self.model.status()
    }
}

/// Standalone OpenAI-shaped prompt renderer over a GGUF's own chat template — tool specs and prior
/// tool calls/results included (`infr_chat::render_chat_oai`). Model-independent: the serve-side
/// render for seam-backed [`ChatModel`]s, so serve never needs the bespoke `Llama` just to render.
pub struct OaiRenderer {
    gguf: infr_gguf::Gguf,
    tokenizer: tokenizers::Tokenizer,
    eos: u32,
    /// The engine configuration this renderer was opened with — `infr-chat`'s renderer reads
    /// `sampling.no_think` / `debug.chat` off it (S7). Held as the `Arc` the caller resolved in
    /// `main()`, borrowed at the render site (R6).
    ecfg: std::sync::Arc<crate::EngineConfig>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl OaiRenderer {
    pub fn open(path: &std::path::Path, ecfg: std::sync::Arc<crate::EngineConfig>) -> Result<Self> {
        let gguf = infr_gguf::Gguf::open(path).map_err(|e| anyhow::anyhow!("open gguf: {e}"))?;
        let tokenizer = crate::build_tokenizer(&gguf)?;
        // Raw metadata (NOT Config::from_gguf — that parser is dense-only and rejects qwen35).
        use infr_core::WeightSource;
        let eos = gguf
            .metadata()
            .u64("tokenizer.ggml.eos_token_id")
            .unwrap_or(2) as u32;
        Ok(Self {
            gguf,
            tokenizer,
            eos,
            ecfg,
        })
    }

    /// The engine configuration this renderer holds (see the `ecfg` field) — the serve adapter
    /// reads `sampling.max_new` off it for a request that names no budget.
    pub fn engine_cfg(&self) -> &std::sync::Arc<crate::EngineConfig> {
        &self.ecfg
    }

    pub fn render(
        &self,
        messages: &[infr_chat::ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        self.render_mode(messages, tools, true)
    }

    /// Render both the generation prompt and its stable no-generation history prefix. The two are
    /// produced by the same embedded template with only `add_generation_prompt` changed.
    pub fn render_turn(
        &self,
        messages: &[infr_chat::ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<(String, String)> {
        Ok((
            self.render_mode(messages, tools, true)?,
            self.render_mode(messages, tools, false)?,
        ))
    }

    fn render_mode(
        &self,
        messages: &[infr_chat::ChatMessage],
        tools: Option<&serde_json::Value>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        infr_chat::render_chat_oai(
            &self.gguf,
            &self.tokenizer,
            self.eos,
            messages,
            tools,
            add_generation_prompt,
            &self.ecfg,
        )
        .map_err(|e| match e {
            // No embedded template at all → the standard "infr requires an instruct model" error.
            infr_chat::TemplateError::NoTemplate => no_template_err(),
            // The template EXISTS but failed to render → surface the actual jinja error so
            // serve's 500 body says what broke (not a generic "no usable template").
            e @ infr_chat::TemplateError::Render(_) => anyhow::anyhow!("{e}"),
        })
    }

    /// Build the FORCED tool-call grammar constraint for this model's tokenizer (see
    /// [`crate::grammar::tool_constraint_for`]); `None` for auto/none/absent tool_choice.
    pub fn tool_constraint(
        &self,
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&str>,
    ) -> Result<Option<crate::grammar::Constraint>> {
        let vocab = self.tokenizer.get_vocab_size(true);
        crate::grammar::tool_constraint_for(&self.tokenizer, vocab, &[self.eos], tools, tool_choice)
    }

    /// Detokenize ids (markers preserved) — the serve-side parse of a constrained tool-call body.
    pub fn decode_ids(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scripted [`ChatModel`] that records every rendered prompt and replies with a canned string
    /// (optionally wrapped in `<think>`). Proves the shared [`Chat`] orchestration WITHOUT a model:
    /// history accumulates, both prior turns render into turn 2's prompt, and `<think>` is stripped
    /// from stored history but streamed to the caller.
    struct ScriptModel {
        rendered: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
        reply: String,
    }

    impl ChatModel for ScriptModel {
        fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
            // A trivial deterministic "template": one line per message.
            let s = messages
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n");
            self.rendered.borrow_mut().push(s.clone());
            Ok(s)
        }

        fn generate(
            &mut self,
            _prompt: &str,
            _max_new: usize,
            _req: Option<&crate::sampling::RequestCtx>,
            on_piece: &mut dyn FnMut(&str),
        ) -> Result<GenStats> {
            on_piece(&self.reply);
            Ok(GenStats::default())
        }
    }

    #[test]
    fn shared_chat_accumulates_history_and_strips_think() {
        let rendered = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let model = ScriptModel {
            rendered: rendered.clone(),
            reply: "<think>reason</think>\n\nParis".into(),
        };
        let mut chat = Chat::new(Box::new(model));

        // Turn 1.
        let mut streamed = String::new();
        chat.turn("capital of France?", 8, &mut |p| streamed.push_str(p))
            .unwrap();
        // The caller sees the full reply (think included — the CLI's ThinkRender dims it live).
        assert_eq!(streamed, "<think>reason</think>\n\nParis");
        // Stored history keeps only the answer after </think>.
        assert_eq!(
            chat.history[0],
            ("user".into(), "capital of France?".into())
        );
        assert_eq!(chat.history[1], ("assistant".into(), "Paris".into()));

        // Turn 2: the rendered prompt must include BOTH prior turns (multi-turn context) with the
        // think-free assistant answer, then the new user message.
        chat.turn("and of Italy?", 8, &mut |_| {}).unwrap();
        let second = &rendered.borrow()[1];
        assert!(second.contains("user: capital of France?"), "{second}");
        assert!(second.contains("assistant: Paris"), "{second}");
        assert!(
            !second.contains("reason"),
            "prior think must be excluded: {second}"
        );
        assert!(second.contains("user: and of Italy?"), "{second}");
    }

    #[test]
    fn shared_chat_passes_the_stable_history_prefix_to_generation() {
        struct TurnAware {
            seen: std::rc::Rc<std::cell::RefCell<Option<(String, String)>>>,
        }

        impl ChatModel for TurnAware {
            fn render(&self, _messages: &[(&str, &str)]) -> Result<String> {
                Ok("stable-prefix<assistant>".into())
            }

            fn render_stable_prefix(&self, _messages: &[(&str, &str)]) -> Result<Option<String>> {
                Ok(Some("stable-prefix".into()))
            }

            fn generate(
                &mut self,
                _prompt: &str,
                _max_new: usize,
                _req: Option<&crate::sampling::RequestCtx>,
                _on_piece: &mut dyn FnMut(&str),
            ) -> Result<GenStats> {
                unreachable!("Chat must use the turn-aware hook")
            }

            fn generate_turn_with_step_hook(
                &mut self,
                prompt: &str,
                stable_prefix: Option<&str>,
                _max_new: usize,
                _req: Option<&crate::sampling::RequestCtx>,
                on_piece: &mut dyn FnMut(&str),
                _on_step: Option<&mut dyn FnMut(crate::diffusion::StepView)>,
            ) -> Result<GenStats> {
                *self.seen.borrow_mut() = Some((
                    prompt.to_string(),
                    stable_prefix.expect("stable prefix").to_string(),
                ));
                on_piece("ok");
                Ok(GenStats::default())
            }
        }

        let seen = std::rc::Rc::new(std::cell::RefCell::new(None));
        let mut chat = Chat::new(Box::new(TurnAware { seen: seen.clone() }));
        chat.turn("hello", 8, &mut |_| {}).unwrap();
        assert_eq!(
            seen.borrow().as_ref(),
            Some(&("stable-prefix<assistant>".into(), "stable-prefix".into()))
        );
    }

    /// A mock whose FIRST `generate` fails and whose later ones succeed — proves a failed generate
    /// does not orphan the just-pushed user turn (finding: a `generate` `Err` used to propagate
    /// with the user turn still in `history`, so the next `turn` pushed a 2nd consecutive user
    /// message and corrupted the conversation).
    struct FlakyModel {
        rendered: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
        fail_next: std::cell::Cell<bool>,
    }

    impl ChatModel for FlakyModel {
        fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
            let s = messages
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n");
            self.rendered.borrow_mut().push(s.clone());
            Ok(s)
        }

        fn generate(
            &mut self,
            _prompt: &str,
            _max_new: usize,
            _req: Option<&crate::sampling::RequestCtx>,
            on_piece: &mut dyn FnMut(&str),
        ) -> Result<GenStats> {
            if self.fail_next.get() {
                self.fail_next.set(false);
                return Err(anyhow::anyhow!("transient generate failure"));
            }
            on_piece("ok");
            Ok(GenStats::default())
        }
    }

    #[test]
    fn failed_generate_does_not_orphan_user_turn() {
        let rendered = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let model = FlakyModel {
            rendered: rendered.clone(),
            fail_next: std::cell::Cell::new(true),
        };
        let mut chat = Chat::new(Box::new(model));

        // Turn 1 fails inside `generate`.
        let before = chat.history.len();
        assert!(chat.turn("hi", 8, &mut |_| {}).is_err());
        // The failed turn must leave history exactly as it was — no dangling user turn.
        assert_eq!(
            chat.history.len(),
            before,
            "a failed generate must not leave an orphaned user turn"
        );

        // Turn 2 succeeds: history stays alternating (user, assistant) — NOT (user, user, …).
        chat.turn("hello again", 8, &mut |_| {}).unwrap();
        assert_eq!(chat.history.len(), 2);
        assert_eq!(chat.history[0].0, "user");
        assert_eq!(chat.history[1].0, "assistant");
        // The successful render saw exactly ONE user turn (no doubled user message).
        let last = rendered.borrow().last().unwrap().clone();
        assert_eq!(
            last.matches("user:").count(),
            1,
            "render must not contain two consecutive user turns: {last}"
        );
    }

    /// `device.ctx` (`INFR_CTX`) reaches the session-backed chats off the `Config` their
    /// `SeamModel` carries — never `std::env::var` (S7, R7). Both halves of the knob: the raw
    /// `SizeSpec` the Vulkan chat routes on, and the token count Metal/diffusion resolve.
    #[test]
    fn ctx_override_comes_from_the_config() {
        let mut cfg = crate::EngineConfig::default();
        assert_eq!(cfg_ctx_spec(&cfg), None, "unset = no override");
        assert_eq!(cfg_ctx(&cfg, 40_960), None);

        // An explicit token count is used verbatim.
        cfg.device.ctx = Some(infr_core::SizeSpec::Bytes(8192));
        assert_eq!(cfg_ctx_spec(&cfg), Some(infr_core::SizeSpec::Bytes(8192)));
        assert_eq!(cfg_ctx(&cfg, 40_960), Some(8192));

        // A percentage resolves against the model's TRAINED context on these paths (no VRAM fit).
        cfg.device.ctx = Some(infr_core::SizeSpec::Percent(0.5));
        assert_eq!(cfg_ctx(&cfg, 40_960), Some(20_480));
    }

    #[test]
    fn strip_think_variants() {
        // Reasoning spans stripped (ALL of them), content concatenated and trimmed.
        assert_eq!(strip_think("<think>a</think>b"), "b");
        assert_eq!(strip_think("<think>a</think>x<think>c</think>y"), "xy");
        // Unterminated think → nothing.
        assert_eq!(strip_think("<think>a"), "");
        // No think → the whole reply.
        assert_eq!(strip_think("plain"), "plain");
    }
}
