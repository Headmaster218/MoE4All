//! The GPU-free `SeamModel` for the CPU reference backend (no Vulkan/VRAM; weights streamed
//! from the GGUF mmap at forward time). Split out of `lib.rs` (no logic change).
use crate::*;
use anyhow::{anyhow, Result};
use infr_chat::{render_chat_jinja, render_chat_user};
use infr_core::backend::Backend;
use infr_core::DType;
use infr_cpu::CpuBackend;
use infr_gguf::Gguf;
use std::path::Path;
use tokenizers::Tokenizer;

/// The placement/dispatch choices a Vulkan bench run actually settled on — reported next to the
/// throughput figure by [`SeamModel::bench_vulkan`]'s caller.
///
/// **Why a throughput number is not interpretable without this.** Two of the inputs are decided at
/// RUN time rather than by the shape being measured: the prefill chunk comes off the placement
/// ladder ([`crate::seam::ubatch_rows`], which the VRAM residency sweep may pin below the default),
/// and the submit cap is latched from a wall-clock sample of the first slow forward
/// (`VulkanBackend::observe_forward`). Both change how the same graph is executed while leaving the
/// dispatched KERNELS byte-identical, so neither is visible in `INFR_PROF_OPS` and a run that
/// differs in either looks like unexplained variance. Printing them makes an old number
/// attributable and an A/B checkable — if the two legs disagree here, they did not measure the
/// same thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchPlacement {
    /// Prefill chunk height in rows, as the prefill loop and the activation reserve saw it.
    pub ubatch: usize,
    /// The height came from the user (`-u` / `INFR_UBATCH`), not from the placement ladder.
    pub ubatch_pinned: bool,
    /// The benchmark session actually allocated coupled q8_0 KV caches.
    pub kv_q8: bool,
    /// Dispatches per submit; `0` = unlimited, one command buffer per forward (every healthy
    /// discrete GPU). Anything else means the submit splitter armed during this run.
    pub submit_cap: usize,
}

/// A **GPU-free** model for the CPU reference backend. Holds only what the agnostic CPU compute
/// graph needs — the parsed [`Config`], the host f32 token embeddings (for the gather + tied lm
/// head), the tokenizer, and the gemma4 E2B per-layer-embd tensors. No `VulkanBackend`, no VRAM,
/// no weight upload: the projection weights are streamed straight from the kept-open GGUF mmap at
/// forward time. Dense Qwen3/Llama, Gemma 3, Gemma 4 (dense + E2B), qwen3moe, and qwen35
/// (`MixerW::DeltaNet`) all drive this same struct.
pub struct SeamModel {
    gguf: Gguf,
    cfg: Config,
    /// The ENGINE configuration every seam path this model drives reads its knobs from —
    /// `kv.*`, `spec.*`, `sampling.*`, `multi.*`, `device.ubatch*`, `paging.cache`, the seam's
    /// `prof.*` diagnostics — held for the model's whole life and BORROWED at each read site
    /// (`docs/config-plan.md` R4/R6). Shared by `Arc` into every backend and session this model
    /// opens, which is exactly the per-process scope §5.1 prescribes.
    ///
    /// Always HANDED in: [`SeamModel::load_with`] is the only constructor. S4's transitional
    /// `load()` — which built one from the environment for callers that had none — is gone as of
    /// S8; every caller in the tree (CLI commands, benches, tests) has a `Config` to give, and a
    /// library caller that wants today's defaults writes `Arc::new(EngineConfig::default())`
    /// rather than silently inheriting the developer's shell.
    ecfg: std::sync::Arc<crate::EngineConfig>,
    /// Host f32 token-embedding table — materialized LAZILY (see [`SeamModel::token_embd`]).
    /// Dequantizing it eagerly at load cost ~4s and ~3.1 GiB of RSS on Qwen3-14B (a 151936×5120
    /// Q4_K table blown up to f32) for every load, while the GPU/Metal dense path never reads it:
    /// those upload `token_embd.weight` to the device in its NATIVE dtype and gather on-device.
    /// Only the host-gather consumers (MTP heads, the DiffusionGemma SC soft-embed, the CPU
    /// runner) touch it, so they now pay for it — and only on first use.
    token_embd: std::sync::OnceLock<Vec<f32>>,
    per_layer_embd: Option<PerLayerEmbd>,
    tokenizer: Tokenizer,
    /// Memoized resident weight footprint (a full tensor-metadata scan). The session-open clamp
    /// path recomputes this 2-3× (`kv_fit_ctx` → `kv_fit_ctx_fmt`, the auto-q8 re-probe, then the
    /// clamp log line) — cache it so the scan runs once per model. See [`SeamModel::footprint`].
    footprint: std::sync::OnceLock<crate::weights::WeightFootprint>,
}

/// The conversation SLOTS a persistent GPU seam session owns: up to `INFR_KV_SLOTS` (default 4)
/// [`crate::seam::SeamKv`]s — each a KV cache + the token ids materialized in it, all
/// sharing one weight upload (`Arc<SeamWeights>`). Per request the best-prefix slot is picked: a
/// prompt that EXTENDS a slot's cache continues it (the classic next-turn suffix prefill); a
/// prompt that diverges early (a different conversation) forks a fresh slot and SEEDS it with the
/// longest shared prefix (e.g. a common system prompt) via a device-side KV copy instead of
/// re-prefilling it; when all slots are taken the LRU one is recycled. Single-conversation
/// callers (run/bench/spec drivers) stay on one slot. Backend-agnostic: fork and seed go through
/// `&dyn Backend` (`copy_buffer` is the seeding primitive), so the Vulkan and Metal sessions
/// share this policy verbatim.
pub(crate) struct SlotPool {
    slots: Vec<Option<crate::seam::SeamKv>>,
    last_used: Vec<u64>,
    tick: u64,
}

/// Pure continuation-slot selection (the "this conversation continuing" arm of [`SlotPool::pick`],
/// and the twin of [`crate::parallel`]'s `pick_continuation`). Given `(slot_idx, prefix_score,
/// cached_len)` per candidate slot and `prompt_len`, pick the qualifying slot with the LONGEST
/// reusable prefix — a slot qualifies when the prompt EXTENDS its cache (`score == cached_len`) or
/// EQUALS it (`score == prompt_len`), and its score is positive. Returns the winning `slot_idx`, or
/// `None` if no slot qualifies. Handing out the shorter of two prefix-matching slots (the old
/// `find`-first behavior) is merely correct — the runner just re-prefills more suffix than needed.
///
/// Split out as a pure fn so this decision is unit-testable without a live backend / KV slots.
fn pick_continuation(
    candidates: impl IntoIterator<Item = (usize, usize, usize)>,
    prompt_len: usize,
) -> Option<usize> {
    candidates
        .into_iter()
        .filter(|&(_, score, cached)| score > 0 && (score == cached || score == prompt_len))
        .max_by_key(|&(_, score, _)| score)
        .map(|(idx, _, _)| idx)
}

/// Open a Vulkan backend on physical device `dev`: `Some(idx)` pins `VulkanN`
/// ([`infr_vulkan::VulkanBackend::new_on_with`], bypassing `device.dev`/the discrete-default rule
/// for the multi-device path), `None` is the historical default
/// ([`infr_vulkan::VulkanBackend::new_with`]). The single funnel every seam-session constructor
/// uses so the pinned and default paths stay in lockstep.
///
/// `cfg` is the MODEL's `Arc<Config>` (S4/S5a): the backend reads its construction-time knobs —
/// device pick, capability masking, VRAM guard, pager diagnostics — off it instead of the
/// environment.
fn open_backend(
    dev: Option<usize>,
    cfg: std::sync::Arc<crate::EngineConfig>,
) -> Result<infr_vulkan::VulkanBackend> {
    match dev {
        Some(idx) => infr_vulkan::VulkanBackend::new_on_with(idx, cfg)
            .map_err(|e| anyhow!("vulkan init (Vulkan{idx}): {e}")),
        None => infr_vulkan::VulkanBackend::new_with(cfg).map_err(|e| anyhow!("vulkan init: {e}")),
    }
}

/// A persistent seam session: the backend it owns, the conversation [`SlotPool`], and the context
/// window its generations are sized to.
///
/// `B` is the backend (it is only ever passed to the seam as `&dyn Backend`, so no bound is needed
/// here); `X` is per-backend extension state — `()` where a backend needs none (Metal),
/// Vulkan's per-session [placement pins](crate::seam::PlacementPins) otherwise. Each backend names
/// its own concrete pairing through a type alias ([`DenseVulkanSession`], [`DenseMetalSession`])
/// so every call site is unchanged.
pub struct DenseSession<B, X = ()> {
    pub(crate) be: B,
    pub(crate) pool: SlotPool,
    pub(crate) max_ctx: usize,
    /// The ENGINE configuration this session was OPENED with — cloned from the [`SeamModel`] that
    /// opened it, so every later call on the session (which must agree with the buffers placement
    /// sized) reads exactly the values the placement decided on. Read as [`Self::cfg`]; never
    /// cloned per token (R6).
    pub(crate) cfg: std::sync::Arc<crate::EngineConfig>,
    /// Backend-specific extension state (see the struct doc); `()` for the backends with none.
    pub(crate) ext: X,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl<B, X> DenseSession<B, X> {
    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots. The ONE implementation
    /// every backend's session shares (it is pure [`SlotPool`] policy — no device involvement).
    pub fn reset_cache(&mut self) {
        self.pool.reset_cache();
    }

    /// The engine configuration this session runs on (see the `cfg` field) — what the seam runner
    /// and every session-driven path read their knobs from instead of the process environment.
    pub fn cfg(&self) -> &std::sync::Arc<crate::EngineConfig> {
        &self.cfg
    }
}

/// The Vulkan session's [`DenseSession::ext`] state: that session's OWN placement pins (see
/// `crate::seam::PlacementPins`) — per-session so a multi-model process never leaks one model's
/// pinned chunk / auto-q8 decision into another. Opaque: the pins stay crate-private, this wrapper
/// only exists so the public [`DenseVulkanSession`] alias can name a public type.
pub struct VulkanSessionPins(std::sync::Arc<crate::seam::PlacementPins>);

/// A persistent Vulkan seam session (see [`SeamModel::vulkan_session`]): owns the backend and the
/// conversation [`SlotPool`], plus this session's [`VulkanSessionPins`] — entered as the current
/// `crate::seam::PlacementScope` around the default-ctx clamp (at construction) and every
/// generation.
pub type DenseVulkanSession = DenseSession<infr_vulkan::VulkanBackend, VulkanSessionPins>;

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DenseVulkanSession {
    /// This session's placement pins — the `ext` state the Vulkan pairing carries (see
    /// [`VulkanSessionPins`]); entered as a placement scope around every generation.
    pub(crate) fn pins(&self) -> &std::sync::Arc<crate::seam::PlacementPins> {
        &self.ext.0
    }

    /// The name of the physical device this session's backend bound (e.g. the discrete GPU or the
    /// iGPU). Multi-device introspection: two sessions pinned to different indices report different
    /// names — the proof that a model-level device pool actually placed each session on its own GPU.
    pub fn device_name(&self) -> String {
        self.be.capabilities().name
    }

    /// Live VRAM snapshot for THIS session's device (see [`infr_vulkan::VramInfo`]). Used to confirm
    /// where a session's weights + KV actually landed: snapshot `available` before/after the first
    /// generation and the drop is this device's allocation (on the iGPU that is system RAM — its
    /// heaps are unified). The discrete/iGPU split is visible because each session owns its OWN
    /// backend/device, so the snapshot is per-device, never a global sum.
    pub fn vram(&self) -> infr_vulkan::VramInfo {
        self.be.vram()
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SlotPool {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            last_used: Vec::new(),
            tick: 0,
        }
    }

    /// Forget every slot's materialized tokens (buffers and the weight upload stay) — discards a
    /// warmup generation so the first real prompt starts from clean slots.
    fn reset_cache(&mut self) {
        for s in self.slots.iter_mut().flatten() {
            s.reset();
        }
    }

    /// The single-conversation slot (the bench/spec drivers, which manage one token stream and
    /// never contend): slot 0, created empty on first use.
    #[cfg(target_os = "macos")]
    fn single(&mut self) -> &mut Option<crate::seam::SeamKv> {
        if self.slots.is_empty() {
            self.slots.push(None);
            self.last_used.push(self.tick);
        }
        &mut self.slots[0]
    }

    /// Pick (and prepare) the slot for `prompt`; returns its index. See the struct doc for the
    /// policy. A freshly created slot is `None` — the runner's first call uploads the weights.
    ///
    /// This best-prefix choice is prefix-OPTIMAL for a real per-position KV cache (dense/
    /// attention arches): the picked slot always has the longest reusable prefix. For qwen35
    /// (gated-DeltaNet: an append-only recurrent summary, not a per-position cache — see the
    /// no-rewind rule in `seam::generate_dense_backend`) a `prefix_score` match is only
    /// scored on shared TOKENS, not on whether the state can actually rewind to it — so the pick
    /// here is merely CORRECT, not necessarily optimal: the runner independently re-checks EXACT
    /// extension (`prompt` extends `cached` verbatim) before reusing the slot's state, and
    /// zero-resets it otherwise. A suboptimal pick for qwen35 just costs an extra full
    /// re-prefill; it can never reuse a wrong recurrent state.
    fn pick(
        &mut self,
        be: &dyn infr_core::backend::Backend,
        cfg: &crate::Config,
        ec: &crate::EngineConfig,
        prompt: &[u32],
    ) -> Result<usize> {
        // Seeding shorter prefixes than this isn't worth the copy submit.
        const MIN_SEED: usize = 16;
        let max_slots: usize = ec.kv.slots;
        self.tick += 1;
        if self.slots.is_empty() {
            self.slots.push(None);
            self.last_used.push(self.tick);
            return Ok(0);
        }
        let score =
            |st: &Option<crate::seam::SeamKv>| st.as_ref().map_or(0, |s| s.prefix_score(prompt));
        // A slot whose cache the prompt EXTENDS (or equals) is this conversation continuing — pick
        // the one with the LONGEST reusable prefix, not merely the first (see `pick_continuation`).
        let cont = pick_continuation(
            (0..self.slots.len()).filter_map(|i| {
                self.slots[i]
                    .as_ref()
                    .map(|s| (i, s.prefix_score(prompt), s.cached_len()))
            }),
            prompt.len(),
        );
        if let Some(i) = cont {
            self.last_used[i] = self.tick;
            return Ok(i);
        }
        // Different conversation: the best shared prefix (if any) seeds the slot we hand out.
        let (best_i, best_s) = (0..self.slots.len())
            .map(|i| (i, score(&self.slots[i])))
            .max_by_key(|&(_, s)| s)
            .unwrap();
        // An EMPTY slot (freshly created, or reset — e.g. by the warmup discard) holds no cached
        // prefix worth keeping, so reusing it can never lose another conversation's KV. Prefer it
        // (LRU first) over forking: forking here would permanently strand a whole full-ctx KV
        // allocation per process (every `infr run` / first serve request pays one otherwise).
        let empty = (0..self.slots.len())
            .filter(|&i| self.slots[i].as_ref().is_none_or(|s| s.cached_len() == 0))
            .min_by_key(|&i| self.last_used[i]);
        let target = if let Some(i) = empty {
            i
        } else if self.slots.len() < max_slots {
            // Fork a fresh slot off any initialized one (shared weights, own KV). A fork that
            // fails the VRAM budget (each slot is a whole max_ctx KV cache — on a big model one
            // slot can own most of free VRAM) degrades to recycling the LRU slot instead of
            // failing the request: correctness is identical (the slot re-prefills from scratch),
            // only cross-conversation KV reuse is lost.
            let src = self
                .slots
                .iter()
                .flatten()
                .next()
                .expect("pick_slot: no initialized slot to fork from");
            match src.fork(be, cfg, ec) {
                Ok(fresh) => {
                    self.slots.push(Some(fresh));
                    self.last_used.push(self.tick);
                    self.slots.len() - 1
                }
                Err(e) => {
                    tracing::warn!(
                        "kv slots: fork of a {}th slot failed ({e}); recycling the LRU slot \
                         instead (fewer concurrent conversations keep their KV)",
                        self.slots.len() + 1,
                    );
                    (0..self.slots.len())
                        .min_by_key(|&i| self.last_used[i])
                        .unwrap()
                }
            }
        } else {
            // Recycle the least-recently-used slot.
            (0..self.slots.len())
                .min_by_key(|&i| self.last_used[i])
                .unwrap()
        };
        if best_s >= MIN_SEED && best_i != target {
            // Give the slot the shared prefix (system prompt etc.) via device-side KV copy —
            // only when it beats whatever prefix the slot already shares with the prompt.
            if best_s > score(&self.slots[target]) {
                let src = self.slots[best_i].take().expect("scored slot is Some");
                if let Some(dst) = self.slots[target].as_mut() {
                    dst.seed_from(be, cfg, ec, &src, best_s)?;
                }
                self.slots[best_i] = Some(src);
            }
        }
        self.last_used[target] = self.tick;
        Ok(target)
    }
}

/// A persistent Metal seam session — the Apple-GPU twin of [`DenseVulkanSession`]: owns the
/// backend and the conversation [`SlotPool`], so every later
/// [`SeamModel::generate_metal_session`] call prefills only the suffix that differs from its
/// slot's previous turn, and concurrent conversations (serve) each keep their own KV slot off
/// the one shared weight upload. Slot switches re-record the decode replay tape (its fingerprint
/// covers the bound KV/IO buffer addresses) — one graph-walk token per switch, never a stale
/// replay.
#[cfg(target_os = "macos")]
pub type DenseMetalSession = DenseSession<infr_metal::MetalBackend>;

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SeamModel {
    /// The engine configuration this model was loaded with (see the `ecfg` field). Handed to every
    /// backend this model — or a caller holding it (`ParallelSeam`, `DenseSeamChat`, the MTP entry
    /// points) — opens, so nothing downstream has to re-read the environment (S4/S5a).
    pub fn cfg(&self) -> &std::sync::Arc<crate::EngineConfig> {
        &self.ecfg
    }

    /// Load a model on a caller-supplied engine configuration. `tokenizer_path` overrides the
    /// GGUF's embedded vocab when given.
    ///
    /// THE constructor: no environment, no globals, no process-wide ordering hazard, and two
    /// models with different configurations can coexist in one process. `infr-cli` resolves ONE
    /// `Arc<Config>` in `main()` and hands the same handle to every model it loads (§5.1). S4's
    /// `load(gguf, tok)` — which built a config from the environment for callers that had none —
    /// was deleted in S8 once every caller had one to give.
    pub fn load_with(
        gguf_path: &Path,
        tokenizer_path: Option<&Path>,
        ecfg: std::sync::Arc<crate::EngineConfig>,
    ) -> Result<Self> {
        let g = Gguf::open(gguf_path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let mut cfg = Config::from_gguf(&g)?;
        let tokenizer = match tokenizer_path {
            Some(p) => Tokenizer::from_file(p).map_err(|e| anyhow!("load tokenizer: {e}"))?,
            None => build_tokenizer(&g)?,
        };
        add_chat_eos(&mut cfg, &tokenizer);
        // `token_embd.weight` is NOT dequantized here — see the field's doc. `Config::from_gguf`
        // above already read its shape, so a model missing the tensor still fails at load, not on
        // the lazy path below.
        let per_layer_embd = build_per_layer_embd(&g, &cfg)?;
        Ok(Self {
            gguf: g,
            cfg,
            ecfg,
            token_embd: std::sync::OnceLock::new(),
            per_layer_embd,
            tokenizer,
            footprint: std::sync::OnceLock::new(),
        })
    }

    /// The ENGINE configuration this model was loaded with (see the `ecfg` field). Every seam path
    /// below borrows it; the sessions this model opens clone the `Arc` so a warm call always reads
    /// the same values its buffers were placed with.
    pub fn engine_cfg(&self) -> &std::sync::Arc<crate::EngineConfig> {
        &self.ecfg
    }

    /// The model's resident weight footprint, computed once and memoized (see the `footprint`
    /// field). `WeightFootprint` is `Copy`, so callers get it by value.
    fn footprint(&self) -> crate::weights::WeightFootprint {
        *self
            .footprint
            .get_or_init(|| crate::weights::weight_footprint(&self.gguf))
    }

    /// Open a persistent Vulkan seam session: weights uploaded ONCE, the KV cache sized to
    /// `max_ctx`, and the materialized-token cache that makes every later
    /// [`generate_vulkan_session`](Self::generate_vulkan_session) call prefill only the suffix
    /// that differs from the previous turn (ChatSession-style KV reuse on the agnostic seam).
    pub fn vulkan_session(&self, max_ctx: usize) -> Result<DenseVulkanSession> {
        self.vulkan_session_on(None, max_ctx)
    }

    /// [`vulkan_session`](Self::vulkan_session) pinned to physical device `dev`
    /// (`Some(idx)` = `VulkanN`, bypassing `INFR_DEV`/the discrete-default rule; `None` = the
    /// default device, byte-identical to `vulkan_session`). The multi-device foundation: two
    /// sessions built on different indices hold two live backends, each owning its own weights + KV
    /// on its own GPU — data-parallel hosting at the session level (Slice 1).
    pub fn vulkan_session_on(
        &self,
        dev: Option<usize>,
        max_ctx: usize,
    ) -> Result<DenseVulkanSession> {
        let vk = open_backend(dev, self.ecfg.clone())?;
        Ok(DenseVulkanSession {
            be: vk,
            pool: SlotPool::new(),
            max_ctx,
            cfg: self.ecfg.clone(),
            ext: VulkanSessionPins(std::sync::Arc::new(crate::seam::PlacementPins::default())),
        })
    }

    /// [`vulkan_session`](Self::vulkan_session) with the DEFAULT context window: the model's
    /// trained context (`n_ctx_train`) clamped so weights + KV cache + activation headroom fit
    /// the VRAM budget — a long-context model's trained window (128k+) would otherwise allocate
    /// a KV cache that blows VRAM at startup or on the first request. Explicit contexts
    /// (INFR_CTX → `vulkan_session(ctx)`) are NEVER clamped; the Vulkan allocation budget
    /// guard still fails a truly-oversized one cleanly at alloc time.
    pub fn vulkan_session_default(&self) -> Result<DenseVulkanSession> {
        self.vulkan_session_default_on(None)
    }

    /// [`vulkan_session_default`](Self::vulkan_session_default) pinned to physical device `dev`
    /// (`Some(idx)` = `VulkanN`; `None` = the default device, byte-identical). Each device's own
    /// VRAM budget clamps its own default context — a session on the weak iGPU gets a window sized
    /// to the iGPU's memory, independent of the discrete card's.
    pub fn vulkan_session_default_on(&self, dev: Option<usize>) -> Result<DenseVulkanSession> {
        let vk = open_backend(dev, self.ecfg.clone())?;
        // The clamp's auto-q8 rung may pin this session's KV format — do it under THIS session's
        // placement scope so the decision lands on its own pins, not a process-global cell.
        let pins = std::sync::Arc::new(crate::seam::PlacementPins::default());
        let scope = crate::seam::PlacementScope::enter(pins.clone());
        let max_ctx = self.clamp_default_ctx(&vk, self.cfg.n_ctx_train);
        drop(scope);
        let max_ctx = max_ctx?;
        Ok(DenseVulkanSession {
            be: vk,
            pool: SlotPool::new(),
            max_ctx,
            cfg: self.ecfg.clone(),
            ext: VulkanSessionPins(pins),
        })
    }

    /// [`vulkan_session`](Self::vulkan_session) sized as a FRACTION of the device's free-VRAM KV
    /// capacity (`INFR_CTX=50%` → half the tokens that would fit after weights + headroom —
    /// device-appropriate %-base: this KV cache lives in VRAM). Uses the same fit math as
    /// [`vulkan_session_default`]'s clamp, scaled by `frac`; unlike an explicit token count the
    /// result is inherently within budget, and the alloc-time guard stays the backstop.
    pub fn vulkan_session_frac(&self, frac: f64) -> Result<DenseVulkanSession> {
        self.vulkan_session_frac_on(None, frac)
    }

    /// [`vulkan_session_frac`](Self::vulkan_session_frac) pinned to physical device `dev`
    /// (`Some(idx)` = `VulkanN`; `None` = the default device, byte-identical). The fraction is of
    /// the pinned device's own free-VRAM KV capacity.
    pub fn vulkan_session_frac_on(
        &self,
        dev: Option<usize>,
        frac: f64,
    ) -> Result<DenseVulkanSession> {
        let vk = open_backend(dev, self.ecfg.clone())?;
        let pins = std::sync::Arc::new(crate::seam::PlacementPins::default());
        let scope = crate::seam::PlacementScope::enter(pins.clone());
        // A pure recurrent-state arch has no per-token KV to size a fraction of — fall back to
        // the trained context (same shape as the default path's `kv_per_tok == 0` escape).
        let fit = self.kv_fit_ctx(&vk).unwrap_or(self.cfg.n_ctx_train);
        drop(scope);
        let max_ctx = ((fit as f64 * frac) as usize).max(1024);
        Ok(DenseVulkanSession {
            be: vk,
            pool: SlotPool::new(),
            max_ctx,
            cfg: self.ecfg.clone(),
            ext: VulkanSessionPins(pins),
        })
    }

    /// Resolve a DEFAULT context length against the VRAM budget: the full weight footprint + one
    /// KV cache (priced as the exact allocation, [`kv_fit_ctx`](Self::kv_fit_ctx)) + the dense
    /// activation reserve must fit the live free bytes. The rung ladder, in order:
    ///
    ///  1. **f16 fits `want`** → use it, say nothing. (The overwhelmingly common case, and — since
    ///     the fit now walks the same prefill-chunk ladder placement walks — the case gemma-3-12b
    ///     at ctx 131072 belongs in and used to miss.)
    ///  2. **f16 misses, q8_0 reaches `want`** → pin auto-q8 and `tracing::warn!`. Only a FULL
    ///     rescue pins q8; a partial one keeps the predictable f16 cache, because auto-q8 exists
    ///     to avoid losing capability, not to trade decode speed for a still-clamped window.
    ///  3. **`INFR_KV_OVERFLOW`** → keep `want` with the cache in system RAM.
    ///  4. **Nothing serves even [`MIN_SESSION_CTX`](crate::seam::MIN_SESSION_CTX)** → `Err`. The
    ///     weights leave no room for a usable cache on this device, so refusing with the numbers
    ///     beats a session that errors at its first activation alloc.
    ///  5. Otherwise **clamp and log**, exactly as before.
    ///
    /// Rung 4 is deliberately narrow. It fires on "no usable context at all", NOT on "the trained
    /// window is bigger than this card" — a 262144-token trained window on a 24 GiB card is
    /// completely ordinary (Qwen3-30B-A3B) and must keep clamping to a working ~50k, not refuse.
    /// A context the USER asked for explicitly (`--ctx N` / `INFR_CTX=N`) never reaches this
    /// function at all: those are taken verbatim and backstopped by the alloc-time VRAM guard.
    ///
    /// Extra KV slots (INFR_KV_SLOTS forks) and MoE expert host-offload aren't modeled — the
    /// alloc-time budget guard remains the backstop for those.
    fn clamp_default_ctx(&self, vk: &infr_vulkan::VulkanBackend, want: usize) -> Result<usize> {
        let Some(fit) = self.kv_fit_ctx(vk) else {
            return Ok(want); // pure recurrent-state arch: no per-token KV to size.
        };
        if fit >= want {
            return Ok(want);
        }
        // Auto-q8 KV rung, clamp flavor (see `crate::seam::PlacementPins` for the policy): before
        // shrinking the DEFAULT context below the trained window, try a Q8_0 KV cache — roughly
        // half the bytes per token. Dense non-MoE models only, matching the placement rung this
        // was validated on (MoE placement budgets pager arenas separately from this fit math).
        let may_auto_q8 = self.cfg.moe.is_none()
            && !crate::seam::kv_auto_q8()
            && crate::seam::kv_unset(&self.ecfg)
            && crate::seam::kv_q8_layout_ok(&self.cfg);
        let fit_q8 = may_auto_q8.then(|| self.kv_fit_ctx_fmt(vk, true)).flatten();
        if fit_q8.is_some_and(|f| f >= want) {
            crate::seam::pin_kv_auto_q8();
            tracing::warn!(
                requested_ctx = want,
                f16_fit_ctx = fit,
                kv_dtype = "q8_0",
                "kv auto-quant: q8_0 KV cache — an f16 cache would clamp the default context \
                 {want} -> {fit}; set INFR_KV_TYPE_K/V=f16 to force f16 (decode is ~10-16% \
                 slower on q8), or --ctx to pick the window yourself"
            );
            return Ok(want);
        }
        // Placement ladder's LAST rung (SWA ring → auto-q8 → THIS): if the context still does
        // not fit VRAM and the user opted into KV overflow, keep the requested window and let
        // the KV cache live in system RAM (`INFR_KV_OVERFLOW`, honored by the Vulkan backend's
        // `BufferUsage::KvCache` alloc). Read over PCIe by attention — slower, but the big
        // context actually runs instead of clamping. Flag off ⇒ today's clamp/error unchanged.
        // The SAME flag the backend's own KV placement reads (`kv.overflow`, off the
        // `Config` the backend was built with) — one value, so the clamp ladder and the
        // allocator cannot disagree about whether the spill is armed.
        if vk.cfg().kv.overflow {
            tracing::warn!(
                "ctx overflow: keeping default context {want} (would clamp to {fit} in VRAM); \
                 INFR_KV_OVERFLOW=1 places the KV cache in system RAM, read over PCIe"
            );
            return Ok(want);
        }
        // f16 cannot reach even the floor but q8 can: pin q8 and clamp to THAT. A partial rescue
        // normally keeps f16 (see the ladder doc), but the choice here is not "slower cache vs
        // smaller window" — it is "slower cache vs no session at all".
        let q8_rescues_the_floor = fit < crate::seam::MIN_SESSION_CTX
            && fit_q8.is_some_and(|f| f >= crate::seam::MIN_SESSION_CTX);
        if q8_rescues_the_floor {
            let f8 = fit_q8.expect("guarded by q8_rescues_the_floor");
            crate::seam::pin_kv_auto_q8();
            tracing::warn!(
                requested_ctx = want,
                f16_fit_ctx = fit,
                q8_fit_ctx = f8,
                kv_dtype = "q8_0",
                "kv auto-quant: q8_0 KV cache — an f16 cache leaves only {fit} tokens of context, \
                 below the {} usable minimum; clamping the default context {want} -> {f8}",
                crate::seam::MIN_SESSION_CTX
            );
            return Ok(f8);
        }
        let vram = vk.vram();
        let fp = self.footprint();
        // Refuse rung: the BEST cache this model may pick on its own (q8_0 when the user set no
        // format, else the format they did set) cannot serve even a minimally-useful window.
        let best = fit.max(fit_q8.unwrap_or(0));
        if best < crate::seam::MIN_SESSION_CTX {
            let free = vram.available.saturating_sub(fp.total());
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            let (k_fmt, v_fmt) = self.kv_fit_fmts(false);
            let need = crate::seam::kv_bytes_estimate_fmt(
                &self.cfg,
                want,
                crate::seam::kv_ring_wanted(&self.cfg, &self.ecfg),
                crate::seam::ubatch_rows(&self.ecfg),
                k_fmt,
                v_fmt,
            );
            return Err(anyhow!(
                "this model cannot be served on this device: no usable KV cache fits.\n  \
                 requested context   : {want} tokens\n  \
                 fits at {k_fmt:?}/{v_fmt:?} KV : {fit} tokens\n  \
                 fits at q8_0 KV     : {}\n  \
                 KV cache needed     : {:.2} GiB at {want} tokens in {k_fmt:?}/{v_fmt:?}\n  \
                 free after weights  : {:.2} GiB ({:.2} GiB {} - {:.2} GiB weights)\n\
                 Try a smaller window (--ctx N / INFR_CTX=N), an explicitly quantized cache \
                 (INFR_KV_TYPE_K=q8_0 INFR_KV_TYPE_V=q8_0), INFR_KV_OVERFLOW=1 to place the cache \
                 in system RAM, or a smaller quant of the model.",
                match fit_q8 {
                    Some(f) => format!("{f} tokens"),
                    None => "not available (an explicit INFR_KV_TYPE_K/V wins)".to_string(),
                },
                gib(need),
                gib(free),
                gib(vram.available),
                if vram.live { "free, live" } else { "heap" },
                gib(fp.total()),
            ));
        }
        tracing::warn!(
            "ctx clamp: default context {want} -> {fit} to fit VRAM (weights {:.2} GiB vs \
             {:.2} GiB available{}); set INFR_CTX to override",
            fp.total() as f64 / (1u64 << 30) as f64,
            vram.available as f64 / (1u64 << 30) as f64,
            if vram.live { ", live" } else { ", total heap" },
        );
        Ok(fit)
    }

    /// The VRAM-fit KV capacity in tokens: the LARGEST context whose KV cache — priced as the
    /// exact allocation the runner will make — plus the dense activation reserve still fits the
    /// device's AVAILABLE memory after the full weight footprint (live free bytes when
    /// VK_EXT_memory_budget is present, so call this BEFORE the weights upload while `available`
    /// still includes their space). Honors the per-layer KV geometry, the SWA ring, and the
    /// runner's KV-dtype env overrides (INFR_KV_TYPE_K/V, INFR_KV_Q8).
    ///
    /// RAW: may return a value below [`crate::seam::MIN_SESSION_CTX`] (down to `0`) when the
    /// weights leave no usable room — [`clamp_default_ctx`](Self::clamp_default_ctx)'s refuse
    /// rung is the reader of that. `None` for a pure recurrent-state arch (no per-token KV).
    /// Extra KV slots (INFR_KV_SLOTS forks) aren't modeled — the alloc-time budget guard remains
    /// the backstop.
    fn kv_fit_ctx(&self, vk: &infr_vulkan::VulkanBackend) -> Option<usize> {
        // Price whatever the runner will actually allocate: the auto-q8 pin (if the placement
        // ladder set it earlier in this process) or the plain env-driven formats.
        self.kv_fit_ctx_fmt(vk, crate::seam::kv_auto_q8())
    }

    /// [`kv_fit_ctx`](Self::kv_fit_ctx) at an EXPLICIT assumed-when-env-unset KV format:
    /// `auto_q8 = true` prices both sides Q8_0 wherever the user set nothing — how the ctx
    /// clamp's auto-q8 rung asks "would the trained window fit if we quantized the cache?"
    /// BEFORE pinning that choice process-wide.
    ///
    /// All the arithmetic lives in [`crate::seam::kv_fit_ctx_for`] (GPU-free, unit-tested); this
    /// only collects the device's VRAM figures and this model's weight footprint and dtypes.
    fn kv_fit_ctx_fmt(&self, vk: &infr_vulkan::VulkanBackend, auto_q8: bool) -> Option<usize> {
        let vram = vk.vram();
        let fp = self.footprint();
        let (k_fmt, v_fmt) = self.kv_fit_fmts(auto_q8);
        // The snapshot goes in whole: `kv_fit_ctx_for` derives its budget from the allocator's OWN
        // ceiling (`VramInfo::alloc_room`), not from the raw free figure — the VRAM guard reserves
        // a fixed headroom below it and refuses anything reaching into it. Budgeting against
        // `vram.available` planned a context 256 MiB past what could ever be allocated, and — now
        // that the rest of this math is exact rather than 5%-shy — that overhang landed as a
        // mid-prefill alloc failure. The placement sweeps derive from the same function, so the
        // context this validates is one placement will actually take (`seam`'s drift tests).
        crate::seam::kv_fit_ctx_for(&self.cfg, &self.ecfg, fp.total(), &vram, k_fmt, v_fmt)
    }

    /// The per-side KV dtypes a fit/footprint estimate must price: the user's explicit
    /// `INFR_KV_TYPE_K`/`_V` when set, else the default ladder (legacy `INFR_KV_Q8`, then a
    /// placement-pinned or candidate auto-q8, then f16) — the same ladder `parse_kv_fmt` walks in
    /// the runner. ESTIMATE ONLY in one direction: the runner additionally gates each format on
    /// backend/alignment and falls back to f16, so a gated-out low-bit request under-estimates
    /// here and the alloc-time VRAM guard backstops it.
    fn kv_fit_fmts(&self, auto_q8: bool) -> (DType, DType) {
        let side = |want: Option<DType>| match want {
            Some(dt) => dt,
            None if self.ecfg.kv.force_q8 || auto_q8 => DType::Q8_0,
            None => DType::F16,
        };
        (side(self.ecfg.kv.type_k), side(self.ecfg.kv.type_v))
    }

    /// Greedy generation on the Vulkan seam through a persistent session (see
    /// [`vulkan_session`](Self::vulkan_session)). `stats.n_prompt` reports the tokens actually
    /// PREFILLED (the un-cached suffix) — the TTFT-honest count.
    pub fn generate_vulkan_session(
        &self,
        session: &mut DenseVulkanSession,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        self.generate_vulkan_session_constrained(session, prompt, max_new, None, req, on_piece)
    }

    /// [`generate_vulkan_session`](Self::generate_vulkan_session) with an optional llguidance
    /// grammar constraint (serve's forced tool_choice) applied to the decode.
    pub fn generate_vulkan_session_constrained(
        &self,
        session: &mut DenseVulkanSession,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: Option<&crate::sampling::RequestCtx>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let slot = session
            .pool
            .pick(&session.be, &self.cfg, &session.cfg, &prompt_tokens)?;
        let max_ctx = session.max_ctx;
        // Cap the reply to the context that's actually left ("a turn also caps to remaining
        // context" — the CLI's generation ceiling is a default, not a demand): a VRAM-clamped
        // default session (e.g. a 21.9 GB model on a 24 GB card clamps 262k -> ~1.7k) would
        // otherwise hard-error on `infr run`'s default max_new=2048 before generating a single
        // token. EOS ends almost every reply long before this cap; an over-long PROMPT still
        // errors cleanly in the runner (its `prompt + gen + 1 > max_ctx` guard stays).
        let max_new = max_new.min(max_ctx.saturating_sub(prompt_tokens.len() + 1));
        // Resolve `ubatch_rows`/`kv_auto_q8` against THIS session's pins (warm calls must agree
        // with the buffers placement sized) — never a process-global cell shared across models.
        let _scope = crate::seam::PlacementScope::enter(session.pins().clone());
        let (_generated, stats) = crate::seam::generate_dense_vulkan_session(
            &session.be,
            &self.gguf,
            &self.cfg,
            &session.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
            &mut session.pool.slots[slot],
            max_ctx,
            constraint,
            req,
        )?;
        // The cold init may have re-clamped the window against the memory the device reported free
        // once the weights were resident (`crate::seam::reclamp_ctx_to_live_room`), so the slot's
        // KV cache — not this field's opening value — is the authority. Refresh it, or every later
        // turn would cap `max_new` against a window that was never allocated.
        if let Some(kv) = session.pool.slots[slot].as_ref() {
            session.max_ctx = kv.max_ctx();
        }
        Ok(stats)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The open GGUF handle — MTP Phase 2 (issue #33) needs it to resolve/upload the head's own
    /// tensors (`crate::mtp::load_mtp_head` / `MtpHeadSession::new_cpu`/`new_vulkan`).
    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    /// The host f32 token-embedding table (`token_embd.weight`), dequantized ONCE on first call
    /// and cached — MTP Phase 2 (issue #33) gathers embedding rows from this on the host,
    /// mirroring every other embed-gather call site on this seam (see `crate::mtp::MtpHeadSession`).
    ///
    /// Lazy on purpose: the Vulkan/Metal dense path uploads `token_embd.weight` to the device in
    /// its native dtype and never calls this, so it must not pay the dequant (~4s / ~3.1 GiB on
    /// Qwen3-14B). `Config::from_gguf` validated the tensor exists at load; a truncated/corrupt file
    /// can still fail the dequant here, so this is FALLIBLE (see [`crate::seam::TokenEmbd::get`]).
    pub fn token_embd(&self) -> Result<&[f32]> {
        self.embd().get()
    }

    /// The LAZY handle to the host token-embedding table, as threaded into the seam runners.
    /// Prefer this over [`token_embd`](Self::token_embd) at any call site that only PASSES the
    /// table on: the GPU/Metal dense runners never read it, so handing them the handle (rather
    /// than a materialized slice) keeps the ~4s / ~3.1 GiB dequant off the GPU load path.
    pub(crate) fn embd(&self) -> crate::seam::TokenEmbd<'_> {
        crate::seam::TokenEmbd::new(&self.token_embd, &self.gguf)
    }

    /// Tokenize raw text with the model's own tokenizer (no chat template) — for callers that
    /// need token ids directly (e.g. a raw-forward validation harness), as opposed to
    /// [`render_chat`](Self::render_chat) + the generation loop.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Raw tokenizer accessor for callers outside this module that need incremental detok (the
    /// diffusion decode loop, Phase 3) via the shared [`crate::stream_token`] helper.
    pub(crate) fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// gemma4-E2B per-layer embedding tensors — threaded into the seam runners by callers that
    /// drive `generate_dense_vulkan_session` directly (`crate::parallel::ParallelSeam`).
    pub(crate) fn per_layer_embd(&self) -> Option<&PerLayerEmbd> {
        self.per_layer_embd.as_ref()
    }

    /// The PER-SLOT context for an N-slot `infr serve --parallel N` session.
    ///
    /// N slots means N KV caches, so the derived per-slot window is the VRAM-fit window DIVIDED by
    /// N: `min(n_ctx_train, kv_fit / N)`.
    ///
    /// **The invariant this buys is "cannot OOM", NOT "same VRAM".** Total KV across the N slots is
    /// `N * (kv_fit / N) = kv_fit` at most — i.e. it is bounded by exactly the same VRAM fit a
    /// 1-slot server is bounded by, so raising `-np` can never overflow a device that `-np 1` fit.
    /// It does NOT mean the footprint is unchanged: when the model's trained window sits BELOW the
    /// fit (Qwen3-14B: trained 40960, fit ~84000 on a 24 GiB card), `-np 1` only allocates the
    /// trained 40960 while `-np 4` allocates 4 x 21097 — more total KV, but still inside budget.
    /// The visible cost of parallelism is the per-request window shrinking (40960 -> 21097 here).
    ///
    /// `want` (from `--ctx` / `INFR_CTX`, sharing `infr_core::parse_size`'s grammar):
    /// - `None` — derive as above.
    /// - `Bytes(c)` — an explicit token count, used VERBATIM per slot and never clamped (the Vulkan
    ///   alloc-time budget guard errors cleanly if `N * c` truly doesn't fit — the user asked).
    /// - `Percent(f)` — `f` of the free-VRAM KV capacity IN TOTAL, split across the N slots
    ///   (`kv_fit * f / N`), so `--ctx 50%` means the same total VRAM at any `-np`.
    ///
    /// `n_slots <= 1` with `want: None` is EXACTLY [`clamp_default_ctx`](Self::clamp_default_ctx) —
    /// byte-for-byte today's default sizing, auto-q8 rung and all.
    pub(crate) fn vulkan_slot_ctx(
        &self,
        vk: &infr_vulkan::VulkanBackend,
        n_slots: usize,
        want: Option<infr_core::SizeSpec>,
    ) -> Result<usize> {
        let n_slots = n_slots.max(1);
        // An explicit token count is the user's demand — honour it verbatim, at any N.
        if let Some(infr_core::SizeSpec::Bytes(c)) = want {
            return Ok((c as usize).max(1));
        }
        if n_slots == 1 && want.is_none() {
            return self.clamp_default_ctx(vk, self.cfg.n_ctx_train);
        }
        let trained = self.cfg.n_ctx_train;
        let Some(fit) = self.kv_fit_ctx(vk) else {
            return Ok(trained); // pure recurrent-state arch: no per-token KV to divide.
        };
        let budget = match want {
            Some(infr_core::SizeSpec::Percent(f)) => (fit as f64 * f) as usize,
            _ => fit,
        };
        // `MIN_SESSION_CTX` is the runner's own floor — below it a slot is useless and the alloc
        // guard's clear error is the better outcome than a silently-crippled window.
        let per_slot = (budget / n_slots).max(crate::seam::MIN_SESSION_CTX);
        let ctx = trained.min(per_slot);
        if ctx < trained {
            tracing::warn!(
                "ctx clamp: --parallel {n_slots} -> per-slot context {trained} -> {ctx} \
                 (each of the {n_slots} slots owns a full KV cache; their total stays within the \
                 same VRAM budget a single slot is held to). Set --ctx to override."
            );
        }
        Ok(ctx)
    }

    /// Detokenize ids back to text (`encode`'s twin, `skip_special_tokens=true` — matches
    /// [`crate::stream_token`]'s convention so a thinking model's `<|channel>thought`/`<channel|>`
    /// markers, which aren't in the tokenizer's added-specials set, still come through as text) —
    /// for callers that drive a decode loop directly on token ids (the diffusion decode loop's own
    /// tests) instead of through one of the `generate_*` string-in/string-out helpers.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// DiffusionGemma Phase-1 validation: a causal prefill of `tokens` on the CPU reference
    /// backend, returning the LAST token's raw logits (`[vocab]`, pre-softmax, post-softcap). Not
    /// specific to diffusion-gemma — works for any arch on this seam — but this is its only
    /// caller today (see `docs/diffusion-gemma.md`).
    pub fn prefill_logits_cpu(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        crate::seam::verify_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// [`prefill_logits_cpu`](Self::prefill_logits_cpu)'s MTP Phase 1 twin (issue #33,
    /// docs/mtp.md): ALSO returns the LM-head INPUT row (post-`output_norm`, pre-lm_head) for the
    /// same last-prompt-token row the logits came from — the `h_p` primitive Phase 2's MTP driver
    /// needs, validated here via `lm_head(h) == logits`. Returns `(logits, h)`.
    pub fn prefill_logits_and_h_cpu(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        crate::seam::verify_dense_cpu_with_h(
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// [`prefill_logits_and_h_cpu`](Self::prefill_logits_and_h_cpu)'s ALL-ROWS twin (MTP Phase 2,
    /// issue #33): returns the LM-head input row for EVERY one of `tokens`, not just the last — the
    /// shape `crate::mtp::catch_up` needs to prime the head's KV over a whole prompt in one call
    /// (`docs/mtp.md`'s `process()` hook runs after every target ubatch, not just the sampled row).
    /// Dense non-MoE models only. Returns `(logits [tokens.len()*vocab], h [tokens.len()*n_embd])`.
    pub fn verify_logits_and_h_cpu(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        crate::seam::verify_rows_cpu_with_h(
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// [`prefill_logits_cpu`](Self::prefill_logits_cpu)'s Vulkan twin, for the CPU/Vulkan
    /// cross-backend parity check.
    pub fn prefill_logits_vulkan(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let vk = infr_vulkan::VulkanBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("vulkan init: {e}"))?;
        crate::seam::verify_dense_vulkan(
            &vk,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            tokens,
        )
    }

    /// Open a Phase-2 DiffusionGemma denoise session on the CPU reference backend (see
    /// `docs/diffusion-gemma.md`): [`prefill`](DiffusionGemmaCpuSession::prefill) causally
    /// prefills the prompt ONCE (encoder scalars, KV rows `0..P`), then repeated
    /// [`denoise`](DiffusionGemmaCpuSession::denoise) calls forward the C-row canvas (decoder
    /// scalars, the bidirectional `Canvas` mask) against the SAME session — each call OVERWRITES
    /// KV rows `P..P+C`, so a caller can denoise the same block with a different partially-
    /// unmasked canvas over and over (the loop Phase 3 drives). `max_ctx` sizes the session's KV
    /// cache (must fit the whole prompt + canvas_length + any headroom for later blocks).
    pub fn diffusion_gemma_cpu_session(&self, max_ctx: usize) -> DiffusionGemmaCpuSession {
        DiffusionGemmaCpuSession {
            be: CpuBackend::new_with(self.ecfg.clone()),
            state: None,
            max_ctx,
        }
    }

    /// [`diffusion_gemma_cpu_session`](Self::diffusion_gemma_cpu_session)'s Vulkan twin, for the
    /// CPU/Vulkan cross-backend parity check.
    pub fn diffusion_gemma_vulkan_session(
        &self,
        max_ctx: usize,
    ) -> Result<DiffusionGemmaVulkanSession> {
        let vk = infr_vulkan::VulkanBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("vulkan init: {e}"))?;
        Ok(DiffusionGemmaVulkanSession {
            be: vk,
            state: None,
            max_ctx,
            pins: std::sync::Arc::new(crate::seam::PlacementPins::default()),
        })
    }

    /// [`diffusion_gemma_vulkan_session`](Self::diffusion_gemma_vulkan_session)'s Metal twin
    /// (Phase D — see docs/diffusion-gemma.md's Metal note). The denoise path
    /// (`generate_dense_backend`'s `denoise_req` branch) is backend-generic, so this is exactly
    /// the Vulkan constructor with Metal's own weight-upload closure.
    #[cfg(target_os = "macos")]
    pub fn diffusion_gemma_metal_session(
        &self,
        max_ctx: usize,
    ) -> Result<DiffusionGemmaMetalSession> {
        let mtl = infr_metal::MetalBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("metal init: {e}"))?;
        Ok(DiffusionGemmaMetalSession {
            be: mtl,
            state: None,
            max_ctx,
        })
    }

    /// Token-level bench on the CPU reference backend (no GPU): prefill `n_prompt` dummy tokens, then
    /// decode `n_gen`, returning the timing ([`crate::GenStats`] has `prompt_secs`/`decode_secs`). Lets
    /// `infr bench -ngl 0` measure prefill (pp = n_prompt/prompt_secs) and decode (tg = n_gen/decode_secs)
    /// directly comparable to `llama-bench -ngl 0`. Dummy tokens — timing is data-independent.
    pub fn bench(&self, n_prompt: usize, n_gen: usize) -> Result<crate::GenStats> {
        let prompt: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i % 100) as u32).collect();
        let (_, stats) = crate::seam::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt,
            n_gen,
            None, // req: bench is a sole sequence — env sampling, no abort latch, no gate
            |_| {},
        )?;
        Ok(stats)
    }

    /// Run the dense decode through the agnostic compute seam on the **Vulkan** backend — the GPU
    /// twin of [`generate_cpu`](Self::generate_cpu). Each native-dtype GGUF weight is padded + uploaded
    /// to VRAM (the CPU path maps it zero-copy instead); the per-token [`infr_core::graph::Graph`] is
    /// compiled + executed by `VulkanBackend`; greedy tokens are detokenized. Same graph, two
    /// backends — this is the end-to-end dense CPU↔GPU parity path.
    pub fn generate_dense_vulkan(&self, prompt: &str, max_new: usize) -> Result<String> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let vk = infr_vulkan::VulkanBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (generated, _stats) = crate::seam::generate_dense_vulkan(
            &vk,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |_| {},
        )?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Token-level Vulkan greedy generation: prefill the given prompt token ids and stream each
    /// GENERATED token id through `on_id` (BOS/template handling is entirely the caller's — nothing
    /// is prepended). Returns the generated ids. The id-exact, Vulkan-backed counterpart to
    /// [`Self::generate_cpu_ids`] — used for token-identity checks (CPU oracle, paged-vs-resident
    /// MoE) that need raw ids rather than detokenized text. A fresh [`infr_vulkan::VulkanBackend`]
    /// per call (like [`Self::generate_dense_vulkan`]) — every weight re-uploads, so the MoE
    /// placement decision (resident/host-visible/paged — see `generate_dense_vulkan_session`) is
    /// re-made fresh each call, which is exactly what a paged-vs-resident A/B needs.
    pub fn generate_vulkan_ids(
        &self,
        prompt_tokens: &[u32],
        max_new: usize,
        on_id: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let vk = infr_vulkan::VulkanBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (generated, _stats) = crate::seam::generate_dense_vulkan(
            &vk,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            prompt_tokens,
            max_new,
            on_id,
        )?;
        Ok(generated)
    }

    /// Multi-GPU PIPELINE (layer-split) twin of [`generate_dense_vulkan`](Self::generate_dense_vulkan):
    /// split this model's transformer layers across the physical `devices` (e.g. `[0, 1]`), each
    /// layer's weights + KV resident on its device, the residual handed across at the boundary.
    /// BIT-IDENTICAL to the single-device path (same ops + per-device kernels). Dense models only.
    pub fn generate_dense_vulkan_pipeline(
        &self,
        devices: &[usize],
        prompt: &str,
        max_new: usize,
    ) -> Result<String> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let (generated, _stats) = crate::seam::generate_dense_vulkan_pipeline(
            devices,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |_| {},
        )?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Token-id twin of [`generate_dense_vulkan_pipeline`](Self::generate_dense_vulkan_pipeline)
    /// (raw ids for a bit-identity check vs [`generate_vulkan_ids`](Self::generate_vulkan_ids)).
    pub fn generate_pipeline_ids(
        &self,
        devices: &[usize],
        prompt_tokens: &[u32],
        max_new: usize,
        on_id: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let (generated, _stats) = crate::seam::generate_dense_vulkan_pipeline(
            devices,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            prompt_tokens,
            max_new,
            on_id,
        )?;
        Ok(generated)
    }

    /// Multi-GPU TENSOR-PARALLEL (dense) twin of [`generate_dense_vulkan`](Self::generate_dense_vulkan):
    /// each layer's weight matrices are SHARDED across the physical `devices` (column-parallel
    /// q/k/v/gate/up, row-parallel o/down), every device computes its shard and the partials are
    /// all-reduced per attention + per FFN. Output equals single-device to reduction-order tolerance.
    /// Dense models only. `devices` of length 1 is the identity (single-rank) reference.
    pub fn generate_dense_vulkan_tp(
        &self,
        devices: &[usize],
        prompt: &str,
        max_new: usize,
    ) -> Result<String> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let (generated, _stats) = crate::seam::generate_dense_vulkan_tp(
            devices,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |_| {},
        )?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Token-id twin of [`generate_dense_vulkan_tp`](Self::generate_dense_vulkan_tp) — raw ids for
    /// the `tensor_parallel_matches_single_device` correctness check (single-rank vs multi-rank).
    pub fn generate_tp_ids(
        &self,
        devices: &[usize],
        prompt_tokens: &[u32],
        max_new: usize,
        on_id: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let (generated, _stats) = crate::seam::generate_dense_vulkan_tp(
            devices,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            prompt_tokens,
            max_new,
            on_id,
        )?;
        Ok(generated)
    }

    /// Multi-GPU EXPERT-PARALLEL (MoE) twin of [`generate_dense_vulkan`](Self::generate_dense_vulkan):
    /// the model's experts are split across the physical `devices` (rank `r` owns experts
    /// `[r·E/W, (r+1)·E/W)`), the router + attention run replicated on every rank, and one P2P
    /// all-reduce per MoE layer combines the partial expert outputs. Output equals single-device to
    /// reduction-order tolerance. MoE models only. `devices` of length 1 is the identity reference.
    pub fn generate_moe_vulkan_ep(
        &self,
        devices: &[usize],
        prompt: &str,
        max_new: usize,
    ) -> Result<String> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let (generated, _stats) = crate::seam::generate_moe_vulkan_ep(
            devices,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |_| {},
        )?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    /// Token-id twin of [`generate_moe_vulkan_ep`](Self::generate_moe_vulkan_ep) — raw ids for the
    /// `expert_parallel_matches_single_device` correctness check (single-rank vs multi-rank).
    pub fn generate_ep_ids(
        &self,
        devices: &[usize],
        prompt_tokens: &[u32],
        max_new: usize,
        on_id: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let (generated, _stats) = crate::seam::generate_moe_vulkan_ep(
            devices,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            prompt_tokens,
            max_new,
            on_id,
        )?;
        Ok(generated)
    }

    /// Token-level bench on the Vulkan seam, llama-bench-comparable: ONE weight upload (a
    /// persistent session) + an untimed pipeline warmup, then per rep — reset the KV, warm it to
    /// `depth` (untimed), and time ONE metric: `pg` = a whole (P prefill + G decode) turn,
    /// `n_gen > 0` = decode at depth, else prefill of `n_prompt` at depth. Returns the per-rep
    /// tokens/sec samples AND the [`BenchPlacement`] the run actually used, so the number can be
    /// attributed after the fact instead of being an unlabelled throughput figure.
    pub fn bench_vulkan(
        &self,
        n_prompt: usize,
        n_gen: usize,
        depth: usize,
        pg: Option<(usize, usize)>,
        reps: usize,
    ) -> Result<(Vec<f64>, BenchPlacement)> {
        let vk = infr_vulkan::VulkanBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("vulkan init: {e}"))?;
        let (p_eff, g_eff) = pg.unwrap_or((n_prompt, n_gen));
        // +16: the untimed warmup runs an 8-prompt + 2-gen turn through the SAME session, so the
        // KV must fit it even when the measured shape is tiny (pp2 with +8 sized the cache to 10
        // and the warmup itself overflowed it).
        let want = depth + p_eff.max(1) + g_eff + 16;
        let dummy = |n: usize| -> Vec<u32> { (0..n.max(1)).map(|i| (i % 100) as u32).collect() };
        let mut state: Option<crate::seam::SeamKv> = None;
        let run = |prompt_len: usize,
                   gen: usize,
                   state: &mut Option<crate::seam::SeamKv>|
         -> Result<crate::GenStats> {
            let (_, stats) = crate::seam::generate_dense_vulkan_session(
                &vk,
                &self.gguf,
                &self.cfg,
                &self.ecfg,
                self.embd(),
                self.per_layer_embd.as_ref(),
                &dummy(prompt_len),
                gen,
                |_| {},
                state,
                want,
                None, // constraint
                None, // req: bench is a sole sequence — env sampling, no gate
            )?;
            Ok(stats)
        };
        // Untimed work must stay out of the INFR_PROF_OPS profile: the warmup turn's m7 batched
        // rows and the depth warm's huge prefill would otherwise dominate/pollute the per-shape
        // aggregate for the tiny timed shape (recorders read the suppression flag at construction,
        // so suppressing around a run() disables their timestamps entirely). Same pattern as
        // DenseSeamChat::warmup. Routes through `with_profiling_suppressed` — an AtomicBool flag, NOT
        // an `INFR_PROF_OPS` env mutation, so it never races the rayon-parallel forward inside `run`
        // (env is a process-wide table; `set_var` is `unsafe` under edition 2024). `gpu_reset`
        // additionally drops anything profiled before the timed reps (e.g. session-init submits)
        // from the exit aggregate.
        let unprofiled = |prompt_len: usize,
                          gen: usize,
                          state: &mut Option<crate::seam::SeamKv>|
         -> Result<crate::GenStats> {
            crate::with_profiling_suppressed(|| run(prompt_len, gen, state))
        };
        // ONE rep, exactly as measured: reset the KV, warm it to `depth` (untimed), time the
        // metric, return it as tokens/sec. A closure so the discarded warm rep below and the
        // measured reps run BYTE-IDENTICAL code — a warm rep that differs from a timed one in any
        // way is not a warmup, it is a second measurement of something else.
        let one_rep = |state: &mut Option<crate::seam::SeamKv>| -> Result<f64> {
            if let Some(st) = state.as_mut() {
                st.reset();
            }
            if depth > 0 {
                // `depth + 1`, for the SAME reason the timed prefill below adds one: at `gen == 0`
                // the decode loop breaks before feeding the frontier token, so a warm of `n` tokens
                // materializes only `n - 1` KV rows and records exactly those as resident (see
                // `resident_after_gen` — it deliberately stopped recording un-written rows, because
                // the next turn would otherwise attend a stale/zero row).
                //
                // Warming with a bare `depth` therefore left the cache one row SHORT, and the timed
                // run silently absorbed that row: `pp4 @ d4096` batch-prefilled positions
                // `depth-1 ..= depth+4` — five rows — while still dividing by `n_prompt = 4`. Shape
                // profiling shows it plainly (`mrow_streamed:m5:...` where it must be `m4`), and it
                // under-reported this shape by ~20% against the pre-`a8b1809` numbers even though
                // per-row throughput had IMPROVED. Feeding one extra token here leaves exactly
                // `depth` rows resident, so the timed window holds precisely the work it reports —
                // llama-bench's `-d` semantics.
                unprofiled(depth + 1, 0, state)?; // warm the cache to `depth` (untimed)
            }
            if let Some((p, g)) = pg {
                // coding-agent turn: prompt ingest + reply generation timed together.
                let s = run(depth + p, g, state)?;
                Ok((p + g) as f64 / (s.prompt_secs + s.decode_secs).max(1e-9))
            } else if n_gen > 0 {
                // decode at depth: 1-token suffix feeds the loop, the timed part is the decode.
                let s = run(depth + 1, n_gen, state)?;
                Ok(n_gen as f64 / s.decode_secs.max(1e-9))
            } else {
                // +1: the suffix's LAST token is the decode feed and is never processed at
                // gen=0, and a suffix of <= 2 skips batched prefill entirely — so `depth + N`
                // measured N-1 batched rows (and pp2 measured nothing, reporting the 1e-9
                // floor). With +1, exactly N rows batch-prefill (positions depth..depth+N) and
                // prompt_secs covers precisely them — llama-bench's -p N semantics.
                let s = run(depth + n_prompt + 1, 0, state)?;
                Ok(n_prompt as f64 / s.prompt_secs.max(1e-9))
            }
        };
        // Untimed warmup: uploads the weights and compiles the pipelines a generic turn hits.
        unprofiled(8, 2, &mut state)?;
        // Untimed warm rep AT THE MEASURED SHAPE, discarded (backlog B6/B16). The (8, 2) turn
        // above does NOT cover the shape being timed — it prefills 7 rows and decodes 2 — so the
        // first TIMED rep was paying that shape's one-time costs (its pipeline variants, its
        // first-touch scratch pools) inside the measured window, and `avg_ts` averaged them in.
        //
        // Measured, Qwen3.6-35B-A3B UD-IQ3_S `pp4 @ d4096`, per-rep wall for the timed m=4 chunk
        // (`INFR_PROF_STAGES=1`) across six processes: rep 1 came in at 24.6 / 40.1 / 45.6 / 42.3
        // / 24.1 / 46.9 ms while reps 2-3 sat at 13.3-14.1 ms every single time. So rep 1 cost
        // 1.8-3.5x steady state, and the SPREAD of that one rep was the whole cell's variance:
        // avg_ts moved 220.1-251.2 t/s (14%) while the steady reps moved under 6%. The same
        // effect is 3% on `pp512` (155.5 vs 150.4 ms) because there the fixed cost is small next
        // to the work — which is exactly why the two prefill columns disagree about
        // reproducibility, and why the shortest metric is the worst offender.
        //
        // This is what `bench_vulkan`'s doc always claimed to do, and llama-bench does the same
        // (a discarded warmup iteration per shape). It removes a cold-start cost from the
        // average; it does not make any kernel faster.
        crate::with_profiling_suppressed(|| one_rep(&mut state))?;
        infr_prof_rt::gpu_reset();
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps.max(1) {
            samples.push(one_rep(&mut state)?);
        }
        // Read the placement AFTER the reps: the pins are set by the binder on the first load, and
        // the submit cap can only have moved during the runs. `bench_vulkan` holds no
        // `PlacementScope`, so these read the process-wide fallback pins this session wrote.
        Ok((
            samples,
            BenchPlacement {
                ubatch: crate::seam::ubatch_rows(&self.ecfg),
                ubatch_pinned: self.ecfg.device.ubatch_specified,
                kv_q8: state.as_ref().is_some_and(crate::seam::SeamKv::kv_q8),
                submit_cap: vk.submit_cap_now(),
            },
        ))
    }

    /// Open a persistent Metal seam session (the Apple-GPU twin of
    /// [`vulkan_session`](Self::vulkan_session)): weights uploaded ONCE, KV sized to `max_ctx`,
    /// later calls prefill only the un-cached suffix.
    #[cfg(target_os = "macos")]
    pub fn metal_session(&self, max_ctx: usize) -> Result<DenseMetalSession> {
        let mtl = infr_metal::MetalBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("metal init: {e}"))?;
        Ok(DenseMetalSession {
            be: mtl,
            pool: SlotPool::new(),
            max_ctx,
            cfg: self.ecfg.clone(),
            ext: (),
        })
    }

    /// Greedy generation on the Metal seam through a persistent session (see
    /// [`metal_session`](Self::metal_session)). `stats.n_prompt` reports the tokens actually
    /// PREFILLED (the un-cached suffix) — the TTFT-honest count.
    #[cfg(target_os = "macos")]
    pub fn generate_metal_session(
        &self,
        session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        self.generate_metal_session_constrained(session, prompt, max_new, None, req, on_piece)
    }

    /// [`generate_metal_session`](Self::generate_metal_session) with an optional llguidance
    /// grammar constraint (serve's forced tool_choice) applied to the decode.
    #[cfg(target_os = "macos")]
    pub fn generate_metal_session_constrained(
        &self,
        session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: Option<&crate::sampling::RequestCtx>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let slot = session
            .pool
            .pick(&session.be, &self.cfg, &session.cfg, &prompt_tokens)?;
        let (_generated, stats) = crate::seam::generate_dense_metal_session(
            &session.be,
            &self.gguf,
            &self.cfg,
            &session.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
            &mut session.pool.slots[slot],
            session.max_ctx,
            constraint,
            req,
        )?;
        Ok(stats)
    }

    /// Greedy generation on the reference **Metal** backend through the agnostic seam (the
    /// Apple-GPU twin of [`generate_cpu`](Self::generate_cpu)): weights are uploaded to Metal
    /// buffers, the per-token [`infr_core::graph::Graph`] is compiled + executed by `MetalBackend`,
    /// and generated tokens stream through `on_piece`.
    #[cfg(target_os = "macos")]
    pub fn generate_metal(
        &self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let mtl = infr_metal::MetalBackend::new_with(self.ecfg.clone())
            .map_err(|e| anyhow!("metal init: {e}"))?;
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::seam::generate_dense_metal(
            &mtl,
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            req,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }

    /// Speculative decoding on the Metal seam (`self` = TARGET, `draft` = the small
    /// same-tokenizer model): the draft session proposes `k` greedy tokens, ONE batched verify
    /// forward of the target checks all of them (LM head on every candidate row), the matching
    /// prefix commits plus the target's own next token as a bonus. Greedy-only (INFR_TEMP=0) —
    /// every committed token is checked against (or produced by) a verify-forward argmax, so
    /// the committed stream is the target's greedy stream over the VERIFY forward. That equals
    /// target-only greedy decode exactly unless a near-tie logit splits between the batched
    /// f16 verify kernels and decode's exact-f32 GEMV; end-to-end equality is pinned by
    /// `metal_spec_decode_matches_target_only_greedy`. Rollback is the session prefix-diff:
    /// rejected rows just get overwritten by the next round's suffix prefill.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_metal_spec(
        &self,
        session: &mut DenseMetalSession,
        draft: &SeamModel,
        draft_session: &mut DenseMetalSession,
        prompt: &str,
        max_new: usize,
        k: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let sampler = crate::sampling::Sampler::from_cfg(&self.ecfg.sampling);
        if sampler.temp > 0.0 {
            return Err(anyhow!(
                "speculative decoding is greedy-only — set INFR_TEMP=0"
            ));
        }
        let vocab = self.cfg.vocab;
        let argmax = |row: &[f32]| -> u32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        };
        let mut committed: Vec<u32> = self.encode(prompt)?;
        let n_prompt = committed.len();

        // Conversation slots, like the plain session chat: pick the best-prefix slot in BOTH
        // sessions from the prompt, once per call (the indices stay stable — LRU recycling only
        // happens inside pick). A returning conversation suffix-prefills its own slots; a
        // different conversation forks/seeds instead of clobbering — multi-user spec serve
        // stops paying a full re-prefill of BOTH models on every conversation switch.
        let t_slot = session
            .pool
            .pick(&session.be, &self.cfg, &session.cfg, &committed)?;
        let d_slot = draft_session.pool.pick(
            &draft_session.be,
            &draft.cfg,
            &draft_session.cfg,
            &committed,
        )?;

        // Initial fill: the target's normal (chunked-prefill) path produces the first token —
        // verify forwards are only for the small k+1 suffixes.
        let t0 = std::time::Instant::now();
        let mut acc_buf: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (first, _stats) = crate::seam::generate_dense_metal_session(
            &session.be,
            &self.gguf,
            &self.cfg,
            &session.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &committed,
            1,
            |_| {},
            &mut session.pool.slots[t_slot],
            session.max_ctx,
            None,
            None,
        )?;
        let prompt_secs = t0.elapsed().as_secs_f64();
        let mut t_next = *first.first().ok_or_else(|| anyhow!("empty first token"))?;
        let mut out: Vec<u32> = Vec::new();
        // Persistent verify feed (= committed ++ this round's candidates). Reused across rounds:
        // each round syncs only the tokens committed SINCE the last round (`feed_committed_len`
        // tracks the committed prefix already in `feed`) then appends the fresh candidates, so the
        // total copy work is O(generation) rather than the O(n²) of cloning all of `committed`
        // every verify round.
        let mut feed: Vec<u32> = Vec::new();
        let mut feed_committed_len = 0usize;
        let ignore_eos = self.ecfg.sampling.ignore_eos;
        let t1 = std::time::Instant::now();
        // Adaptive draft length: k tracks recent acceptance (EMA) — code-shaped output
        // accepts ~3/round and deserves the full k; high-entropy text accepts ~1-2 and a
        // long draft just adds verify rows that get thrown away (measured net-NEGATIVE at
        // fixed k=4 on prose). Bounds [1, k].
        let mut ema = 2.0f64;
        'outer: while out.len() < max_new {
            // Commit the token the target already chose (initial sample or verify bonus).
            let is_eos =
                !ignore_eos && (self.cfg.eos_ids.contains(&t_next) || t_next == self.cfg.eos);
            out.push(t_next);
            if !is_eos {
                stream_token(
                    &self.tokenizer,
                    &mut acc_buf,
                    &mut printed,
                    t_next,
                    &mut on_piece,
                );
            }
            committed.push(t_next);
            if is_eos || out.len() >= max_new {
                break;
            }
            // Draft proposes up to k greedy continuations of the committed stream (its session
            // suffix-prefills the divergence from last round automatically).
            let k_now = (ema.round() as usize).clamp(1, k);
            let budget = k_now.min(max_new - out.len());
            let td = std::time::Instant::now();
            let (cand, _) = crate::seam::generate_dense_metal_session(
                &draft_session.be,
                &draft.gguf,
                &draft.cfg,
                &draft_session.cfg,
                draft.embd(),
                draft.per_layer_embd.as_ref(),
                &committed,
                budget,
                |_| {},
                &mut draft_session.pool.slots[d_slot],
                draft_session.max_ctx,
                None,
                None,
            )?;
            // Verify: one batched target forward over [t_next, cand..]; row i's argmax is the
            // target's choice after consuming everything up to and including suffix row i. Rebuild
            // `feed` = committed ++ cand WITHOUT re-cloning committed: drop last round's candidate
            // tail, append only the newly-committed tokens, then this round's candidates.
            feed.truncate(feed_committed_len);
            feed.extend_from_slice(&committed[feed_committed_len..]);
            feed_committed_len = committed.len();
            feed.extend_from_slice(&cand);
            if feed.len() + 1 > session.max_ctx {
                break; // context full: the committed stream is still exact
            }
            let td = td.elapsed();
            let tv = std::time::Instant::now();
            let (logits, vstats) = crate::seam::verify_dense_metal2(
                &session.be,
                &self.gguf,
                &self.cfg,
                &session.cfg,
                self.embd(),
                self.per_layer_embd.as_ref(),
                &feed,
                &mut session.pool.slots[t_slot],
                session.max_ctx,
            )?;
            if self.ecfg.spec.debug {
                tracing::info!(
                    "[spec] draft {:.1}ms verify {:.1}ms (exec {:.1}ms) cand={}",
                    td.as_secs_f64() * 1e3,
                    tv.elapsed().as_secs_f64() * 1e3,
                    vstats * 1e3,
                    cand.len()
                );
            }
            let m = logits.len() / vocab;
            // Rows cover feed's un-cached suffix; the candidate checks use the LAST cand.len()+1
            // rows (the row before cand[0] is t_next's — its argmax checks cand[0]).
            let base = m - (cand.len() + 1);
            // Target's greedy choice at each candidate row plus the bonus row, then the pure
            // accept decision (unit-tested in `spec_accept_tests`, incl. the rejection branch the
            // self-spec e2e test can't reach): the longest prefix the target ratifies and the
            // next committed token — the target's correction at the first mismatch, or the bonus
            // token when all candidates accept.
            let varg: Vec<u32> = (0..=cand.len())
                .map(|j| argmax(&logits[(base + j) * vocab..(base + j + 1) * vocab]))
                .collect();
            let (accepted, next_tok) = spec_accept(&cand, &varg);
            for &c in &cand[..accepted] {
                let is_eos = !ignore_eos && (self.cfg.eos_ids.contains(&c) || c == self.cfg.eos);
                out.push(c);
                if !is_eos {
                    stream_token(
                        &self.tokenizer,
                        &mut acc_buf,
                        &mut printed,
                        c,
                        &mut on_piece,
                    );
                }
                committed.push(c);
                if is_eos || out.len() >= max_new {
                    break 'outer;
                }
            }
            ema = 0.7 * ema + 0.3 * (accepted as f64 + 1.0);
            // Roll the committed view back past the rejected tail: the session's `cached` holds
            // all of `feed`, and the next prefix diff overwrites the stale rows.
            t_next = next_tok;
        }
        Ok(crate::GenStats {
            n_prompt,
            prompt_secs,
            n_gen: out.len(),
            decode_secs: t1.elapsed().as_secs_f64(),
        })
    }

    /// Token-level bench on the **Metal** backend through the agnostic seam (the Apple-GPU twin of
    /// [`bench`](Self::bench)): prefill `n_prompt` dummy tokens, decode `n_gen`, return the timing.
    /// Lets `infr bench` (with `INFR_METAL=1`) measure pp/tg directly comparable to `llama-bench`
    /// on the Metal build.
    /// Runs through a persistent [`DenseMetalSession`] so backend, uploaded weights, compiled
    /// pipelines, and the dequant/repack weight caches all survive across reps — a fresh backend
    /// per rep re-paid every one-time cost inside the measurement (a factored-format checkpoint
    /// re-repacked hundreds of MB per rep). `reset_tokens` starts a fresh repetition; leaving it
    /// false preserves a preceding untimed depth warm so the next call measures only its suffix.
    /// `profile` controls whether this call contributes to the backend's profiler summary.
    #[cfg(target_os = "macos")]
    pub fn bench_metal(
        &self,
        session: &mut DenseMetalSession,
        n_prompt: usize,
        n_gen: usize,
        reset_tokens: bool,
        profile: bool,
    ) -> Result<crate::GenStats> {
        let prompt: Vec<u32> = (0..n_prompt.max(1)).map(|i| (i % 100) as u32).collect();
        if reset_tokens {
            if let Some(s) = session.pool.single().as_mut() {
                s.reset_tokens();
            }
        }
        let _profile_guard = (!profile).then(|| session.be.suppress_profiling());
        let (_, stats) = crate::seam::generate_dense_metal_session(
            &session.be,
            &self.gguf,
            &self.cfg,
            &session.cfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt,
            n_gen,
            |_| {},
            session.pool.single(),
            session.max_ctx,
            None,
            None,
        )?;
        Ok(stats)
    }

    /// Render a user turn with the model's OWN embedded chat template (so an instruct model — Gemma,
    /// Qwen, … — answers coherently). Errors if the GGUF has no `tokenizer.chat_template` or it fails
    /// to render — infr only supports models that ship one (no fabricated-ChatML fallback).
    pub fn render_chat(&self, user: &str) -> Result<String> {
        render_chat_user(&self.gguf, &self.tokenizer, self.cfg.eos, user, &self.ecfg)
            .ok_or_else(no_template_err)
    }

    /// Render a multi-turn conversation `(role, content)` through the model's OWN embedded chat
    /// template — the [`crate::chat::ChatModel::render`] primitive for the CPU dense/MoE path, so the
    /// shared [`crate::chat::Chat`] can drive a history-based REPL. Same template + error contract as
    /// [`render_chat`](Self::render_chat), generalized past a single user turn.
    pub fn render_chat_messages(&self, messages: &[(&str, &str)]) -> Result<String> {
        render_chat_jinja(
            &self.gguf,
            &self.tokenizer,
            self.cfg.eos,
            messages,
            true,
            &self.ecfg,
        )
        .ok_or_else(no_template_err)
    }

    /// Greedy generation on the CPU reference backend (no GPU). Returns the decoded text plus
    /// timing/counts ([`crate::GenStats`]) for the caller's stats line.
    /// The generated text is delivered through `on_piece` as it streams; only timing/counts are
    /// returned.
    pub fn generate_cpu(
        &self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        // Stream each generated token: incrementally detokenize and emit the new suffix.
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::seam::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            req,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }

    /// [`Self::generate_cpu`] on the REFERENCE CPU backend ([`infr_cpu::CpuBackend::reference`]):
    /// quantized weights are decoded to f32 and dotted in f32, so the run carries the weight's own
    /// quantization error and nothing else. The production path quantizes ACTIVATIONS to int8
    /// (llama.cpp's `vec_dot_*_q8_K` regime) at ~4e-3 relative error, which is invisible at Q4_K+
    /// but flips greedy tokens at Q2_K — so this, not `generate_cpu`, is the oracle a GPU-vs-CPU
    /// token comparison must be scored against. Slower; not what the CPU goldens pin.
    pub fn generate_cpu_reference(
        &self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::GenStats> {
        let prompt_tokens: Vec<u32> = self.encode(prompt)?;
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::seam::generate_dense_cpu_mode(
            infr_cpu::CpuBackend::reference_with(self.ecfg.clone()),
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            req,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }

    /// Token-level CPU greedy generation: prefill the given prompt token ids and stream each
    /// GENERATED token id through `on_id` (BOS/template handling is entirely the caller's — nothing
    /// is prepended). Returns the generated ids. The id-exact counterpart to [`Self::generate_cpu`],
    /// used for token-identity checks against a reference implementation.
    pub fn generate_cpu_ids(
        &self,
        prompt_tokens: &[u32],
        max_new: usize,
        on_id: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        let (generated, _stats) = crate::seam::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.ecfg,
            self.embd(),
            self.per_layer_embd.as_ref(),
            prompt_tokens,
            max_new,
            None, // req: token-identity checks are a sole sequence — env sampling, no gate
            on_id,
        )?;
        Ok(generated)
    }
}

/// A persistent CPU-reference session for DiffusionGemma's two-pass forward (Phase 2 — see
/// docs/diffusion-gemma.md and [`SeamModel::diffusion_gemma_cpu_session`]). Model-independent (like
/// [`DenseVulkanSession`]/[`DenseMetalSession`]) — `prefill`/`denoise` take the `&SeamModel` per
/// call instead of borrowing it at construction, so a [`crate::chat::ChatModel`] can hold both an
/// owned `SeamModel` and a persistent session side by side (Phase 3 — no self-referential borrow).
pub struct DiffusionGemmaCpuSession {
    be: CpuBackend,
    state: Option<crate::seam::SeamKv>,
    max_ctx: usize,
}

/// [`DiffusionGemmaCpuSession`]'s Vulkan twin (see [`SeamModel::diffusion_gemma_vulkan_session`]).
pub struct DiffusionGemmaVulkanSession {
    be: infr_vulkan::VulkanBackend,
    state: Option<crate::seam::SeamKv>,
    max_ctx: usize,
    /// This session's OWN placement pins (see [`crate::seam::PlacementPins`]) — DG binds through
    /// the shared `vulkan_moe_binder`, so its placement decisions must stay per-session too.
    pins: std::sync::Arc<crate::seam::PlacementPins>,
}

/// [`DiffusionGemmaVulkanSession`]'s Metal twin (Phase D — see
/// [`SeamModel::diffusion_gemma_metal_session`]).
#[cfg(target_os = "macos")]
pub struct DiffusionGemmaMetalSession {
    be: infr_metal::MetalBackend,
    state: Option<crate::seam::SeamKv>,
    max_ctx: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaCpuSession {
    /// Causal prefill of `tokens` (encoder scalars, chunked/per-token like every other dense
    /// prefill on this seam) — writes KV rows `0..tokens.len()`. Call once per block before
    /// [`denoise`](Self::denoise); a second call with a prompt that EXTENDS the previous one
    /// continues the session (ChatSession-style prefix reuse), matching every other session on
    /// this seam.
    pub fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        let bind = crate::seam::cpu_upload_bind(&self.be);
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            tokens,
            1, // rides the ordinary per-token causal-prefill loop (see `verify_dense_cpu`'s doc)
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        Ok(())
    }

    /// ONE canvas denoise forward over the session's already-prefilled prompt (see
    /// `DenoiseReq`'s doc): `canvas_tokens.len()` MUST match every call on this session (the
    /// model's `canvas_length`). Returns `[canvas_tokens.len() * vocab]` raw logits. `sc_logits`
    /// is the PREVIOUS step's raw canvas logits for self-conditioning (`None` = off, matching the
    /// reference's step-0 gate); `temp_inv` is the self-conditioning softmax temperature divisor
    /// (unused when `sc_logits` is `None`). Panics-free even before [`prefill`](Self::prefill) is
    /// ever called — errors instead (an empty prompt, P=0).
    pub fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        let mut out_logits = Vec::new();
        let mut reduced = None; // CPU never requests the GPU reducer (`u: None` below) — stays `None`
        let bind = crate::seam::cpu_upload_bind(&self.be);
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            &[], // denoise never touches the prompt/generation token stream — see `DenoiseReq`
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            Some(crate::seam::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
                u: None,
                sample_temp_inv: 0.0,
                reduced: &mut reduced,
            }),
            None,
        )?;
        Ok(out_logits)
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaVulkanSession {
    /// [`DiffusionGemmaCpuSession::prefill`]'s Vulkan twin. Weights bind through the SHARED
    /// [`crate::seam::vulkan_moe_binder`] so DG gets the same MoE expert placement tiers
    /// (resident / `INFR_CACHE` / auto-paged) as every other MoE model — DG's fused
    /// `ffn_gate_up_exps` bank pages under `Role::Gate` and its mixed Q5_0/Q8_0 down banks split
    /// into per-byte-size pools (see `infr_vulkan::pager`'s MoE-session doc).
    pub fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        let bind = crate::seam::vulkan_moe_binder(
            &self.be,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            self.state.is_none(),
            self.max_ctx,
        )?;
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            tokens,
            1,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        // Once per prefill (a denoise step would print per step — far too noisy).
        self.be.print_moe_pager_stats();
        Ok(())
    }

    /// [`DiffusionGemmaCpuSession::denoise`]'s Vulkan twin. Perf slice 3
    /// (docs/diffusion-gemma.md): when `u` is `Some`, this asks `generate_dense_backend` to try the
    /// GPU entropy-bound sampler reducer on this step's logits (see
    /// [`crate::seam::DenoiseOutcome`]) — `sample_temp_inv` is THIS step's sampler temperature
    /// divisor (`denoise_block`'s local `temp_inv`, distinct from `temp_inv` above, which is the
    /// PREVIOUS step's self-conditioning divisor) and `u` is `canvas_tokens.len()` host-drawn
    /// uniform `[0,1)` floats (the seeded CDF-inversion draw). `u: None` always takes the ordinary
    /// full-logits path (`gpu_seam_matches_cpu_diffusion_gemma_denoise`'s direct row-by-row
    /// comparison needs the full array); the backend may ALSO decline on its own (falls back to
    /// `DenoiseOutcome::Logits`) even when `u` is `Some` — the caller (`diffusion.rs::denoise_block`,
    /// or a direct test) handles both outcomes.
    pub fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
        sample_temp_inv: f32,
        u: Option<&[f32]>,
    ) -> Result<crate::seam::DenoiseOutcome> {
        let mut out_logits = Vec::new();
        let mut reduced = None;
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        // The shared placement-aware binder (see `prefill`): only ever CALLED when this denoise
        // is the session's first load (no prior `prefill` — a direct-denoise test), where it must
        // make the same placement decision prefill would have.
        let bind = crate::seam::vulkan_moe_binder(
            &self.be,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            self.state.is_none(),
            self.max_ctx,
        )?;
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            &[],
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            Some(crate::seam::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
                u,
                sample_temp_inv,
                reduced: &mut reduced,
            }),
            None,
        )?;
        Ok(match reduced {
            Some(r) => crate::seam::DenoiseOutcome::Reduced(r),
            None => crate::seam::DenoiseOutcome::Logits(out_logits),
        })
    }
}

/// Phase D: Metal twin of [`DiffusionGemmaVulkanSession`] (see
/// [`SeamModel::diffusion_gemma_metal_session`]). `generate_dense_backend`'s `denoise_req` branch
/// is backend-generic (verified — nothing in it besides the Phase-B `gpu_sc` gate distinguishes
/// Vulkan, and that gate now includes Metal too, see `seam.rs`), so this only differs from
/// the Vulkan twin in its weight-upload closure: Metal uploads weights in their NATIVE GGUF dtype
/// with no padding (matching `generate_dense_metal_session`'s closure), unlike Vulkan's
/// `pad_to_u32_align`.
#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
impl DiffusionGemmaMetalSession {
    /// [`DiffusionGemmaCpuSession::prefill`]'s Metal twin.
    pub fn prefill(&mut self, model: &SeamModel, tokens: &[u32]) -> Result<()> {
        let bind = crate::seam::metal_upload_bind(&self.be);
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            tokens,
            1,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        Ok(())
    }

    /// [`DiffusionGemmaCpuSession::denoise`]'s Metal twin.
    pub fn denoise(
        &mut self,
        model: &SeamModel,
        canvas_tokens: &[u32],
        sc_logits: Option<&[f32]>,
        temp_inv: f32,
    ) -> Result<Vec<f32>> {
        let mut out_logits = Vec::new();
        let mut reduced = None; // Metal never requests the GPU reducer (`u: None` below, Phase D — Vulkan only for this slice)
        let bind = crate::seam::metal_upload_bind(&self.be);
        crate::seam::generate_dense_backend(
            &self.be,
            &*bind,
            &model.gguf,
            &model.cfg,
            &model.ecfg,
            model.embd(),
            model.per_layer_embd.as_ref(),
            &[],
            0,
            |_| {},
            &mut self.state,
            self.max_ctx,
            None,
            None,
            None,
            None,
            None,
            Some(crate::seam::DenoiseReq {
                canvas_tokens,
                sc_logits,
                temp_inv,
                out_logits: &mut out_logits,
                u: None,
                sample_temp_inv: 0.0,
                reduced: &mut reduced,
            }),
            None,
        )?;
        Ok(out_logits)
    }
}

/// The speculative-decoding accept decision for one verify round (pure — no backend). Given the
/// draft `cand` and `verify_argmax` — the target verify-forward's greedy choice at each candidate
/// row plus the bonus row (`cand.len() + 1` entries; index `j` is the target's token after
/// consuming `cand[..j]`) — return the number of leading candidates the target ratifies (the
/// longest prefix where its argmax equals the draft) and the next committed token: the target's
/// correction at the first mismatch, or the bonus token when every candidate is accepted.
///
/// Correctness property (see `spec_accept_tests`): the committed stream `cand[..accepted] ++
/// [next]` is always exactly `verify_argmax[..=accepted]` — the target's own greedy stream — no
/// matter what the draft proposed. That is what makes speculative decoding output-identical to
/// target-only greedy; a wrong draft only shortens the accepted prefix, never commits a wrong
/// token. Backend-agnostic (no `cfg` gate): the macOS spec driver above and `crate::mtp`'s Vulkan
/// MTP spec driver (issue #33) both call this — one pure acceptance rule for every spec flavor.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn spec_accept(cand: &[u32], verify_argmax: &[u32]) -> (usize, u32) {
    debug_assert_eq!(verify_argmax.len(), cand.len() + 1);
    let accepted = cand
        .iter()
        .zip(verify_argmax)
        .take_while(|(c, v)| c == v)
        .count();
    (accepted, verify_argmax[accepted])
}

/// Temperature-aware speculative accept rule — the stochastic sibling of [`spec_accept`], used
/// when `INFR_TEMP > 0` (the greedy rule above requires the trunk's argmax as its one-and-only
/// target, which is why pure greedy makes thinking models degenerate — see `crate::sampling`'s
/// `Sampler` doc — and why MTP needs this at all). Implements the standard speculative-sampling
/// acceptance rule (Leviathan & Kalman 2023 / Chen et al. 2023):
///
/// Draft head proposes `cand[i]` from its own truncated proposal distribution `q_dists[i]`
/// (top-k/top-p-truncated at the SAME `Sampler` config the target uses — `sampling::truncated_dist`
/// applied to the head's own logits). The target trunk's verify forward gives the truncated
/// distributions `p_dists[0..=cand.len()]` at those same positions (row `j` = the target's
/// distribution conditioned on `cand[..j]`; `p_dists[cand.len()]` is the bonus row, conditioned on
/// every candidate being accepted). For `i` in `0..cand.len()`: draw `u ~ Uniform(0,1)`; accept
/// `cand[i]` iff `u < min(1, p_i(cand[i]) / q_i(cand[i]))`. On the first rejection, sample the next
/// token from the normalized residual `max(p_i - q_i, 0)` and STOP this cycle (no bonus). If every
/// candidate is accepted, sample the bonus token from `p_dists[cand.len()]`.
///
/// This provably preserves the TARGET's sampling distribution — unlike `spec_accept`, the
/// committed stream is not required to equal any single model's argmax/sample stream token-for-
/// token across a re-run; it's a sample from the target's distribution either way (see
/// `spec_accept_stochastic_tests` for the acceptance-rule properties this is pinned to: identical
/// p/q always accepts, an off-support draft always rejects, the residual always renormalizes, and
/// a fixed seed is deterministic).
fn dist_prob(dist: &[(u32, f32)], x: u32) -> f32 {
    dist.iter()
        .find(|&&(id, _)| id == x)
        .map(|&(_, p)| p)
        .unwrap_or(0.0)
}

/// The rejection-sampling residual `max(p - q, 0)`, renormalized to sum to 1 — the distribution
/// [`spec_accept_stochastic`] resamples from on a coin-flip rejection. Only entries in `p`'s own
/// support can contribute a positive residual (an entry outside `p` has `p == 0`, which clamps to
/// 0 regardless of `q`), so the returned ids are always a subset of `p`'s ids. Empty when `p` and
/// `q` agree pointwise everywhere on `p`'s support — the degenerate case `spec_accept_stochastic`
/// falls back to `p`'s own top choice for (see `spec_accept_stochastic_tests` for both properties).
fn residual_dist(p: &[(u32, f32)], q: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut residual: Vec<(u32, f32)> = Vec::with_capacity(p.len());
    let mut sum = 0.0f32;
    for &(id, pv) in p {
        let r = (pv - dist_prob(q, id)).max(0.0);
        if r > 0.0 {
            residual.push((id, r));
            sum += r;
        }
    }
    if sum > 0.0 {
        for e in residual.iter_mut() {
            e.1 /= sum;
        }
    }
    residual
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn spec_accept_stochastic(
    cand: &[u32],
    q_dists: &[Vec<(u32, f32)>],
    p_dists: &[Vec<(u32, f32)>],
    rng: &mut u64,
) -> (usize, u32) {
    debug_assert_eq!(q_dists.len(), cand.len());
    debug_assert_eq!(p_dists.len(), cand.len() + 1);
    for i in 0..cand.len() {
        let x = cand[i];
        let q_x = dist_prob(&q_dists[i], x);
        let p_x = dist_prob(&p_dists[i], x);
        // q_x > 0 always holds in practice (x was drawn from q_dists[i]'s own support), but guard
        // the division for a caller that hands in an inconsistent (cand, q_dists) pair.
        let ratio = if q_x > 0.0 { (p_x / q_x).min(1.0) } else { 0.0 };
        let u = crate::sampling::next_uniform(rng);
        if u < ratio {
            continue; // accept — draw the next position's coin
        }
        // Reject: sample the next committed token from the normalized residual max(p_i - q_i, 0).
        let residual = residual_dist(&p_dists[i], &q_dists[i]);
        let next = if !residual.is_empty() {
            crate::sampling::sample_from_dist(&residual, rng)
        } else {
            // Degenerate: q_i == p_i pointwise on p_i's support, so the residual is empty, yet the
            // coin still rejected (only possible if x itself sat outside p_i's support, p_x == 0).
            // Fall back to p_i's own top choice so the cycle always makes forward progress. p_i
            // itself being EMPTY is a caller contract violation (a truncated target distribution
            // always has ≥1 entry) — panic with a clear message rather than silently emit token 0.
            p_dists[i]
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&(id, _)| id)
                .expect(
                    "spec_accept_stochastic: p_dists[i] is empty — a truncated target \
                     distribution must carry at least one entry",
                )
        };
        return (i, next);
    }
    // Every candidate accepted: sample the bonus token from the target's own distribution.
    let bonus = crate::sampling::sample_from_dist(&p_dists[cand.len()], rng);
    (cand.len(), bonus)
}

#[cfg(test)]
mod spec_accept_tests {
    use super::spec_accept;

    #[test]
    fn all_accepted_returns_bonus() {
        // Draft fully matches the target → accept all three, next = the bonus-row token.
        assert_eq!(spec_accept(&[10, 11, 12], &[10, 11, 12, 99]), (3, 99));
    }

    #[test]
    fn reject_at_zero_returns_correction() {
        // Draft wrong at position 0 → accept none, next = the target's correction.
        assert_eq!(spec_accept(&[10, 11, 12], &[7, 11, 12, 99]), (0, 7));
    }

    #[test]
    fn reject_in_middle_commits_prefix() {
        // Accept [10], reject at 1 → next = the correction 8; the rest is discarded.
        assert_eq!(spec_accept(&[10, 11, 12], &[10, 8, 12, 99]), (1, 8));
    }

    #[test]
    fn empty_draft_returns_sole_token() {
        // No candidates (adaptive k floored mid-round) → next = the sole bonus token.
        assert_eq!(spec_accept(&[], &[42]), (0, 42));
    }

    #[test]
    fn committed_stream_is_target_greedy_for_any_draft() {
        // The rejection-sampling invariant the self-spec e2e test cannot reach (its draft always
        // agrees, so acceptance is always full): whatever the draft proposes, the committed
        // tokens equal a prefix of the target's own argmax stream — never a token the target
        // wouldn't have produced. This is the branch the original review flagged as untested.
        let varg = [5u32, 6, 7, 8]; // the target's greedy choice at each of the four rows
        for draft in [
            vec![5, 6, 7], // perfect draft → accept 3
            vec![9, 6, 7], // wrong at 0    → accept 0
            vec![5, 9, 7], // wrong at 1    → accept 1
            vec![5, 6, 9], // wrong at 2    → accept 2
            vec![0, 0, 0], // all wrong     → accept 0
        ] {
            let (accepted, next) = spec_accept(&draft, &varg);
            let mut committed: Vec<u32> = draft[..accepted].to_vec();
            committed.push(next);
            assert_eq!(
                committed,
                varg[..=accepted].to_vec(),
                "draft {draft:?} committed a non-greedy stream"
            );
        }
    }
}

#[cfg(test)]
mod spec_accept_stochastic_tests {
    use super::{residual_dist, spec_accept_stochastic};

    #[test]
    #[should_panic(expected = "p_dists[i] is empty")]
    fn empty_target_distribution_panics_not_token_zero() {
        // Degenerate caller contract violation: p_dists[0] is EMPTY. Then p_x == 0 ⇒ ratio 0 ⇒ the
        // coin always rejects; the residual is empty (no p mass); and p_dists[0] has no top choice.
        // The old fallback silently returned token id 0 — now it must panic with a clear message
        // instead of emitting a possibly-invalid token.
        let cand = [5u32];
        let q_dists = vec![vec![(5u32, 1.0f32)]];
        let p_dists = vec![vec![], vec![(10u32, 1.0f32)]]; // len == cand.len() + 1
        let mut rng = 1u64;
        let _ = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng);
    }

    #[test]
    fn identical_p_q_always_accepts() {
        // q == p everywhere ⇒ the ratio is exactly 1 at every position, so `u < 1` always holds
        // (next_uniform draws are in [0,1)) ⇒ every candidate is accepted, every seed.
        let dist = vec![(10u32, 0.5f32), (11, 0.3), (12, 0.2)];
        let cand = [10u32, 11, 12];
        let q_dists = vec![dist.clone(), dist.clone(), dist.clone()];
        let p_dists = vec![dist.clone(), dist.clone(), dist.clone(), dist.clone()];
        for seed in 1..50u64 {
            let mut rng = seed | 1;
            let (accepted, next) = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng);
            assert_eq!(accepted, cand.len(), "seed {seed}: expected full accept");
            // The bonus token must come from the (only nonzero-mass) support of p_dists[3].
            assert!(dist.iter().any(|&(id, _)| id == next));
        }
    }

    #[test]
    fn zero_target_prob_always_rejects() {
        // The drafted token (99) has zero probability under the target's truncated distribution
        // (it isn't in p_dist's support at all) ⇒ ratio == 0 ⇒ always rejected regardless of the
        // coin draw, and the residual must be resampled (never token 99 itself).
        let q_dist = vec![(99u32, 1.0f32)]; // draft head is CERTAIN of a token the target rejects
        let p_dist = vec![(10u32, 0.6f32), (11, 0.4)]; // target never puts mass on 99
        let cand = [99u32];
        let q_dists = vec![q_dist];
        let p_dists = vec![p_dist.clone(), p_dist.clone()];
        for seed in 1..50u64 {
            let mut rng = seed | 1;
            let (accepted, next) = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng);
            assert_eq!(accepted, 0, "seed {seed}: p(x)=0 must always reject");
            assert_ne!(
                next, 99,
                "seed {seed}: rejected token must not be re-emitted"
            );
            assert!(p_dist.iter().any(|&(id, _)| id == next));
        }
    }

    #[test]
    fn residual_normalizes_and_is_nonnegative() {
        // p has mass q doesn't (10, 12); q has mass p doesn't (99, ignored); both have 11 with
        // q's share larger than p's (clamps to 0 there).
        let p = vec![(10u32, 0.5f32), (11, 0.2), (12, 0.3)];
        let q = vec![(11u32, 0.9f32), (99, 0.1)];
        let r = residual_dist(&p, &q);
        // Only p's support can appear; 11's residual (0.2 - 0.9 clamped to 0) must be dropped.
        assert!(r.iter().all(|&(id, _)| id == 10 || id == 12));
        assert!(!r.iter().any(|&(id, _)| id == 11));
        assert!(r.iter().all(|&(_, p)| p >= 0.0));
        let sum: f32 = r.iter().map(|&(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "residual must renormalize to 1, got {sum}"
        );
    }

    #[test]
    fn residual_empty_when_q_dominates_p_everywhere() {
        // q >= p pointwise on p's whole support ⇒ every residual clamps to 0 ⇒ empty.
        let p = vec![(10u32, 0.3f32), (11, 0.3)];
        let q = vec![(10u32, 0.5f32), (11, 0.5)];
        assert!(residual_dist(&p, &q).is_empty());
    }

    #[test]
    fn deterministic_given_fixed_seed() {
        let q_dist = vec![(5u32, 1.0f32)];
        let p_dist = vec![(5u32, 0.4f32), (6, 0.6)];
        let cand = [5u32];
        let q_dists = vec![q_dist];
        let p_dists = vec![p_dist.clone(), p_dist];
        // A fixed, non-env-dependent seed (not `sampling::seed_rng()`, which falls back to a
        // wall-clock seed and would make this test flaky under parallel `cargo test`). Any NONZERO
        // value will do — zero is xorshift64's forbidden fixed point (see `legal_xorshift_state`).
        let mut rng_a = 1u64;
        let mut rng_b = 1u64;
        let a = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng_a);
        let b = spec_accept_stochastic(&cand, &q_dists, &p_dists, &mut rng_b);
        assert_eq!(a, b, "same seed must reproduce the same accept decision");
        assert_eq!(rng_a, rng_b, "same seed must reproduce the same rng stream");
    }
}

#[cfg(test)]
mod pick_continuation_tests {
    use super::pick_continuation;

    #[test]
    fn picks_longest_prefix_not_first() {
        // Two slots both continue (prompt extends their cache: score == cached_len); slot 2 has the
        // LONGER reusable prefix, so it must win even though slot 0 appears first. Slot 1 is a
        // different conversation (score below both its cache and prompt_len).
        let candidates = [
            (0usize, 20usize, 20usize), // extends: score 20 == cached 20
            (1, 5, 40),                 // no: 5 != 40 and 5 != 100
            (2, 60, 60),                // extends: score 60 == cached 60 (longest)
        ];
        assert_eq!(pick_continuation(candidates, 100), Some(2));
    }

    #[test]
    fn accepts_exact_equal_prompt() {
        // score == prompt_len (the prompt EQUALS the cache) qualifies even when cached_len differs.
        let candidates = [(7usize, 30usize, 50usize)];
        assert_eq!(pick_continuation(candidates, 30), Some(7));
    }

    #[test]
    fn none_when_no_slot_qualifies() {
        // A partial-but-diverged prefix (score < cached_len and < prompt_len) does not continue.
        let candidates = [(0usize, 10usize, 40usize), (1, 0, 0)];
        assert_eq!(pick_continuation(candidates, 100), None);
        // Empty candidate set.
        assert_eq!(pick_continuation(std::iter::empty(), 100), None);
    }
}
