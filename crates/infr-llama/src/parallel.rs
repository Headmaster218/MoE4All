//! [`ParallelSeam`] — the N-slot concurrent generation engine behind `infr serve --parallel N`.
//!
//! # What this is (and what it is not)
//!
//! It is **interleaved concurrent generation**: N sequences are in flight at once, each on its own
//! thread with its own KV slot, and they take turns on the GPU at TOKEN granularity via a fair
//! round-robin baton ([`crate::sampling::StepGate`]). A request is never head-of-line blocked
//! behind another request's whole generation — only behind one step of it.
//!
//! It is **not** continuous batching. llama.cpp gathers the N active sequences' next tokens into
//! ONE forward at `m = n_active`, which amortises the weight traffic across them and makes
//! aggregate throughput RISE with concurrency. We cannot express that today: [`infr_core::Op`]'s
//! `Attention` binds exactly one `k_cache`/`v_cache`, one `kv_len` and one `pos` for all its query
//! rows, so N rows cannot attend to N different KV caches in one dispatch. Getting there needs a
//! per-row `(cache, kv_len, pos)` indirection (a block table) plumbed through ~8 recorder attention
//! entry points and ~12 Vulkan shaders, plus the CPU/Metal backends, plus inverting this crate's
//! monolithic `generate_dense_backend` into a per-step API. That is a real project, and it is the
//! ONLY thing standing between this engine and the mrow-kernel throughput win.
//!
//! So: aggregate throughput here is roughly FLAT in N (each sequence still re-streams the whole
//! weight matrix for its own decode step). What N buys is *fairness and latency* — 4 agent tool
//! calls finish in ~the time of the slowest, not the sum. That is the difference between usable and
//! unusable for a fan-out coding agent, and it is an honest fraction of the win.
//!
//! # VRAM: how `-np` interacts with `--ctx`
//!
//! N slots means N KV caches. The per-slot context is therefore `min(n_ctx_train, kv_fit_ctx / N)`
//! by default, so the N slots TOGETHER are bounded by exactly the VRAM fit that bounds one slot:
//! raising `-np` can never OOM a device that `-np 1` fit. (It is NOT the same footprint — when the
//! trained window is below the fit, `-np 4` does allocate more total KV than `-np 1`; what it
//! cannot do is exceed the budget. The visible cost is the per-request window shrinking.) An
//! explicit `--ctx C` is used verbatim per slot, and the Vulkan alloc-time budget guard is left to
//! fail it cleanly if `N * C` truly doesn't fit.
//!
//! Slots are forked EAGERLY at startup (weights are shared through `Arc<SeamWeights>`; a fork costs
//! only its own KV + IO buffers). That means a VRAM refusal happens at boot with a clear message,
//! never halfway through serving.

use crate::sampling::{RequestCtx, StepGate};
use crate::seam::SeamKv;
use crate::{Config, GenStats, SeamModel};
use anyhow::{anyhow, Result};
use std::sync::{Arc, Condvar, Mutex};

/// Pure continuation-slot selection (the "this conversation continuing" case of [`checkout`], and
/// the twin of `seam::model::SlotPool::pick`'s first arm). Given `(slot_idx, prefix_score,
/// cached_len)` for each candidate free slot and the `prompt_len`, pick the qualifying slot with
/// the LONGEST reusable prefix — a slot qualifies when the prompt EXTENDS its cache (`score ==
/// cached_len`) or EQUALS it (`score == prompt_len`), and its score is positive. Returns the
/// winning `slot_idx`, or `None` if no slot qualifies.
///
/// Split out as a pure fn so this decision is unit-testable without a live Vulkan backend / KV
/// slots (the lock-drop and device-side `seed_from` around it stay integration-only).
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

/// One KV slot: its cache (moved OUT while a request holds it, so the generation gets the
/// `&mut Option<SeamKv>` the runner wants without holding the pool lock), plus the bookkeeping the
/// prefix-match/LRU policy needs.
struct Slot {
    /// `None` while checked out by an in-flight request, or before the slot is initialized.
    kv: Option<SeamKv>,
    busy: bool,
    /// LRU stamp.
    tick: u64,
}

/// The N-slot pool. Deliberately a plain `Mutex` + `Condvar` rather than the sequential
/// [`crate::seam::model`] `SlotPool`: checkout has to MOVE the `SeamKv` out (a generation holds it
/// for its whole lifetime, which is far too long to hold a lock over) and has to consider only the
/// slots that are actually free.
struct Pool {
    slots: Vec<Slot>,
    tick: u64,
}

/// A checked-out slot. Returns its KV to the pool on drop — including on error or panic, so a
/// failed request can never permanently burn a slot.
struct SlotGuard<'a> {
    engine: &'a ParallelSeam,
    idx: usize,
    kv: Option<SeamKv>,
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        let mut p = match self.engine.pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        p.tick += 1;
        let tick = p.tick;
        let s = &mut p.slots[self.idx];
        s.kv = self.kv.take();
        s.busy = false;
        s.tick = tick;
        drop(p);
        self.engine.freed.notify_one();
    }
}

/// The concurrent seam engine. `Sync`: `&self` is all a request needs, so N of them run at once.
pub struct ParallelSeam {
    model: SeamModel,
    vk: infr_vulkan::VulkanBackend,
    pool: Mutex<Pool>,
    /// Signalled when a slot is returned — a queued request waits here.
    freed: Condvar,
    /// The GPU baton. `None` when `n_slots == 1`: a lone sequence must not pay even a mutex per
    /// token, and single-request decode speed is a hard non-regression requirement.
    gate: Option<Arc<StepGate>>,
    max_ctx: usize,
    /// This model's OWN placement pins (pinned prefill chunk / auto-q8 KV — see
    /// [`crate::seam::PlacementPins`]). Per-engine so a multi-model host (`infr multi` runs N of
    /// these concurrently) never leaks one model's ladder decision to another. Entered as the
    /// current [`crate::seam::PlacementScope`] around placement (construction/warmup) and every
    /// request's decode; concurrent requests on THIS engine all point at this one shared cell.
    pins: Arc<crate::seam::PlacementPins>,
}

impl ParallelSeam {
    /// Build an N-slot engine: upload the weights once (via a warmup generation on slot 0, which is
    /// also what compiles every lazily-built pipeline), then fork N-1 sibling slots off it.
    ///
    /// `want_ctx` is the `--ctx` / `INFR_CTX` spec (token count or `%` of the free-VRAM KV
    /// capacity); `None` derives the per-slot window. See [`SeamModel::vulkan_slot_ctx`].
    pub fn new(
        model: SeamModel,
        n_slots: usize,
        want_ctx: Option<infr_core::SizeSpec>,
    ) -> Result<Self> {
        Self::new_on(None, model, n_slots, want_ctx)
    }

    /// [`new`](Self::new) pinned to physical device `dev`: `Some(idx)` binds `VulkanN`
    /// ([`infr_vulkan::VulkanBackend::new_on_with`], bypassing `device.dev`/the discrete-default
    /// rule), `None` is the default device (byte-identical to `new`). This is what lets `infr multi` host
    /// several concurrent-slot engines side by side, each on its own GPU: the whole engine (weights,
    /// N KV slots, recorder) lives on the ONE backend this constructs, so nothing crosses devices.
    pub fn new_on(
        dev: Option<usize>,
        model: SeamModel,
        n_slots: usize,
        want_ctx: Option<infr_core::SizeSpec>,
    ) -> Result<Self> {
        let n_slots = n_slots.max(1);
        let ecfg = model.cfg().clone();
        let vk = match dev {
            Some(idx) => infr_vulkan::VulkanBackend::new_on_with(idx, ecfg)
                .map_err(|e| anyhow!("vulkan init (Vulkan{idx}): {e}"))?,
            None => infr_vulkan::VulkanBackend::new_with(ecfg)
                .map_err(|e| anyhow!("vulkan init: {e}"))?,
        };
        Self::new_with_backend(model, n_slots, want_ctx, vk)
    }

    /// Build on a caller-owned Vulkan backend. Used by the unified service path so auxiliary
    /// engines can later derive clients from the exact device and elastic arena initialized by
    /// this LLM warmup.
    pub fn new_with_backend(
        model: SeamModel,
        n_slots: usize,
        want_ctx: Option<infr_core::SizeSpec>,
        vk: infr_vulkan::VulkanBackend,
    ) -> Result<Self> {
        let n_slots = n_slots.max(1);
        // This engine's own placement pins, entered as the current scope for the whole
        // placement phase (the clamp inside `vulkan_slot_ctx` + the `init_slots` warmup, which is
        // where the binder pins the prefill chunk / auto-q8 KV). See `PlacementPins`.
        let pins = Arc::new(crate::seam::PlacementPins::default());
        let scope = crate::seam::PlacementScope::enter(pins.clone());
        let max_ctx = model.vulkan_slot_ctx(&vk, n_slots, want_ctx)?;
        let mut engine = Self {
            model,
            vk,
            pool: Mutex::new(Pool {
                slots: Vec::new(),
                tick: 0,
            }),
            freed: Condvar::new(),
            // A 1-slot server has nothing to take turns with — keep it on the exact uncontended
            // path `infr run` takes (see `RequestCtx::gate_pass`: `None` constructs nothing).
            gate: (n_slots > 1).then(|| Arc::new(StepGate::new())),
            max_ctx,
            pins,
        };
        engine.init_slots(n_slots)?;
        drop(scope);
        Ok(engine)
    }

    pub fn fork_embedding_backend(&self) -> Result<infr_vulkan::VulkanBackend> {
        self.vk
            .fork_embedding_client()
            .map_err(|error| anyhow!("derive unified Embedding backend: {error}"))
    }

    pub fn unified_vram_stats(&self) -> Option<infr_vulkan::unified::UnifiedVramStats> {
        self.vk.unified_vram().map(|pool| pool.stats())
    }

    /// Materialize slot 0 (weights + KV + pipelines) with a throwaway generation, then fork the
    /// rest off it. `&mut self` — this runs at startup, before the engine is shared.
    fn init_slots(&mut self, n_slots: usize) -> Result<()> {
        self.vk.defer_session_finalization(true);
        let initialized = self.init_slots_before_host_import(n_slots);
        if let Err(error) = initialized {
            self.vk.defer_session_finalization(false);
            return Err(error);
        }
        self.vk
            .finish_deferred_session_allocations()
            .map_err(|error| anyhow!("finalize Vulkan session allocations: {error}"))
    }

    /// Build every persistent slot before optional WDDM Host DMA aliases are admitted. The warmup
    /// still sees the proportionally preloaded RAM tier; only its faster Vulkan alias is deferred.
    fn init_slots_before_host_import(&mut self, n_slots: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        // The warmup generation both uploads the weights and compiles every lazily-built pipeline,
        // so the first REAL request pays neither. INFR_PROF_OPS is suppressed for it (recorders read
        // it at construction; warmup submits would pollute a later bench's per-op aggregate) via the
        // shared `with_profiling_suppressed` helper.
        let mut slot0: Option<SeamKv> = None;
        crate::with_profiling_suppressed(|| {
            crate::seam::generate_dense_vulkan_session(
                &self.vk,
                self.model.gguf(),
                self.model.config(),
                self.model.engine_cfg(),
                self.model.embd(),
                self.model.per_layer_embd(),
                &[1u32],
                2,
                |_| {},
                &mut slot0,
                self.max_ctx,
                Some(crate::seam::TurnCheckpoint::Enable),
                None, // constraint
                None, // req: startup, not a request — env sampling, no gate
            )
        })?;
        let mut slot0 = slot0.ok_or_else(|| anyhow!("warmup did not initialize a KV slot"))?;
        // The warmup is what loads the weights, so it is also where the cold init re-clamps the
        // context against the memory the device reports free once they are resident. Take the
        // window that was actually allocated: every slot forked below is sized from it, and it is
        // what the server advertises and admits requests against.
        self.max_ctx = slot0.max_ctx();
        // Drop the warmup tokens so the first real prompt prefills a clean slot from row 0 instead
        // of forking off a garbage prefix.
        slot0.reset();

        let mut slots = Vec::with_capacity(n_slots);
        for i in 1..n_slots {
            // A fork shares the `Arc<SeamWeights>` — it costs only its own KV + IO buffers. If VRAM
            // refuses, say so HERE, at boot, with the two knobs that fix it. Never mid-request.
            let kv = slot0
                .fork(&self.vk, self.model.config(), self.model.engine_cfg())
                .map_err(|e| {
                    anyhow!(
                        "could not allocate KV slot {}/{n_slots} at ctx {}: {e}\n\
                     lower --parallel, or lower --ctx (each slot owns a full context of KV cache)",
                        i + 1,
                        self.max_ctx,
                    )
                })?;
            slots.push(Slot {
                kv: Some(kv),
                busy: false,
                tick: 0,
            });
        }
        slots.insert(
            0,
            Slot {
                kv: Some(slot0),
                busy: false,
                tick: 0,
            },
        );
        self.pool.get_mut().expect("fresh pool").slots = slots;
        tracing::info!(
            "slots: {n_slots} x {} ctx ready in {:.1}s",
            self.max_ctx,
            t0.elapsed().as_secs_f32()
        );
        Ok(())
    }

    pub fn n_slots(&self) -> usize {
        self.pool.lock().expect("pool poisoned").slots.len()
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// The physical device this engine's backend bound (e.g. the discrete GPU or the iGPU). Used by
    /// `infr multi` to print the model→device routing table — two engines pinned to different
    /// indices report different names.
    pub fn device_name(&self) -> String {
        use infr_core::backend::Backend;
        self.vk.capabilities().name
    }

    pub fn model(&self) -> &SeamModel {
        &self.model
    }

    /// A fresh per-sequence context wired to this engine's baton — one per request.
    pub fn request_ctx(&self, sampling: crate::sampling::RequestSampling) -> RequestCtx {
        match &self.gate {
            Some(g) => RequestCtx::with_gate(sampling, g.clone()),
            None => RequestCtx::new(sampling),
        }
    }

    /// Take a slot for `prompt`, blocking until one is free.
    ///
    /// Slot choice preserves the cross-request KV prefix cache: a prompt that EXTENDS (or equals) a
    /// free slot's cached tokens continues that slot — this is the persistent prefix cache that
    /// makes a repeated system prompt ~7x cheaper on TTFT, and it is why the pick runs BEFORE the
    /// generation rather than round-robining blindly. Otherwise the least-recently-used free slot is
    /// recycled, seeded (device-side KV copy) from whichever free slot shares the longest prefix.
    ///
    /// Only FREE slots are considered: a busy slot's KV is checked out and cannot be read or
    /// recycled. So under load the prefix cache degrades gracefully (fewer candidate slots) rather
    /// than corrupting an in-flight sequence.
    fn checkout(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
        /// Seeding shorter prefixes than this isn't worth the copy submit.
        const MIN_SEED: usize = 16;
        let cfg: &Config = self.model.config();
        let ec = self.model.engine_cfg();
        let mut p = self.pool.lock().expect("pool poisoned");
        loop {
            let free: Vec<usize> = (0..p.slots.len()).filter(|&i| !p.slots[i].busy).collect();
            if free.is_empty() {
                // Every slot is generating. The server's admission semaphore normally prevents this
                // (it bounds in-flight requests to n_slots), so this is the belt-and-braces path.
                p = self.freed.wait(p).expect("pool poisoned");
                continue;
            }
            let score = |s: &Slot| s.kv.as_ref().map_or(0, |k| k.prefix_score(prompt));
            // 1. This conversation continuing: the free slot with the LONGEST reusable prefix among
            //    those the prompt extends (or equals) — not merely the first such slot (which would
            //    re-prefill more suffix). Pure decision, unit-tested via `pick_continuation`.
            let cont = pick_continuation(
                free.iter().filter_map(|&i| {
                    p.slots[i].kv.as_ref().and_then(|k| {
                        k.continuation_prefix_len(prompt)
                            .map(|prefix| (i, prefix, prefix))
                    })
                }),
                prompt.len(),
            );
            // 2. Otherwise the LRU free slot, preferring an already-empty one (nothing to lose).
            let target = match cont {
                Some(i) => i,
                None => *free
                    .iter()
                    .min_by_key(|&&i| {
                        let empty = p.slots[i].kv.as_ref().is_none_or(|k| k.cached_len() == 0);
                        (!empty, p.slots[i].tick)
                    })
                    .expect("free is non-empty"),
            };
            // Seed the target with the best shared prefix among the other FREE slots (a common
            // system prompt), via a device-side KV copy instead of re-prefilling it.
            if cont.is_none() {
                let best = free
                    .iter()
                    .copied()
                    .filter(|&i| i != target)
                    .max_by_key(|&i| score(&p.slots[i]));
                if let Some(best) = best {
                    let best_s = score(&p.slots[best]);
                    if best_s >= MIN_SEED && best_s > score(&p.slots[target]) {
                        // `seed_from` is a device-side KV copy — it RECORDS, so it takes a turn on
                        // the baton like any other GPU submit. The module doc forbids holding the
                        // pool `Mutex` across a submit (it would serialize every other request's
                        // checkout/drop behind this copy). So: RESERVE both slots (mark them busy so
                        // no concurrent checkout can select them), take their KV out, DROP the lock
                        // across `seed_from`, then re-acquire to put them back.
                        p.slots[best].busy = true;
                        p.slots[target].busy = true;
                        let src = p.slots[best].kv.take().expect("scored slot is Some");
                        let mut dst = p.slots[target].kv.take();
                        drop(p);
                        let r = {
                            let _gp = req.gate_pass();
                            match dst.as_mut() {
                                Some(dst) => dst.seed_from(&self.vk, cfg, ec, &src, best_s),
                                None => Ok(()),
                            }
                        };
                        p = self.pool.lock().expect("pool poisoned");
                        // Return the source slot's KV and release its reservation; `target` stays
                        // reserved (busy) — it is checked out just below.
                        p.slots[best].kv = Some(src);
                        p.slots[best].busy = false;
                        p.slots[target].kv = dst;
                        // A failed seed costs only the prefix reuse — the slot re-prefills from
                        // scratch and the answer is identical. Never fail the request for it.
                        if let Err(e) = r {
                            tracing::warn!(
                                "kv slots: prefix seed failed ({e}); re-prefilling instead"
                            );
                        }
                        // `best` just went free again — wake a waiter that may want it.
                        self.freed.notify_one();
                    }
                }
            }
            p.tick += 1;
            let tick = p.tick;
            p.slots[target].busy = true;
            p.slots[target].tick = tick;
            let kv = p.slots[target].kv.take();
            return Ok(SlotGuard {
                engine: self,
                idx: target,
                kv,
            });
        }
    }

    /// Render an OpenAI conversation through the model's own chat template.
    pub fn render_chat_messages(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    /// Generate one sequence: check out a slot, run the ordinary seam decode on it (taking turns on
    /// the GPU baton at every step), and return the slot. `&self` — N of these run concurrently.
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: &RequestCtx,
        on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_turn(prompt, None, max_new, constraint, req, on_piece)
    }

    /// [`generate`](Self::generate) with a stable rendered-history prefix for recurrent state
    /// checkpointing. Attention-only models ignore the resulting boundary in the runner.
    pub fn generate_turn(
        &self,
        prompt: &str,
        stable_prefix: Option<&str>,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: &RequestCtx,
        mut on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        let prompt_tokens = self.model.encode(prompt)?;
        let turn_checkpoint = self.model.turn_checkpoint(&prompt_tokens, stable_prefix)?;
        let mut guard = self.checkout(&prompt_tokens, req)?;
        // Cap the reply to the context actually left in THIS slot (a per-slot ctx is smaller than
        // the model's trained window under `-np N`), mirroring the sequential session path: a
        // generation ceiling is a default, not a demand. An over-long PROMPT still errors cleanly
        // in the runner.
        let max_new = max_new.min(self.max_ctx.saturating_sub(prompt_tokens.len() + 1));
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        // Resolve `ubatch_rows`/`kv_auto_q8` against THIS engine's pins for the decode (warm calls
        // must agree with the buffers placement sized). Concurrent requests share the one cell.
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        let (_ids, stats) = crate::seam::generate_dense_vulkan_session(
            &self.vk,
            self.model.gguf(),
            self.model.config(),
            self.model.engine_cfg(),
            self.model.embd(),
            self.model.per_layer_embd(),
            &prompt_tokens,
            max_new,
            |id| {
                crate::stream_token(
                    self.model.tokenizer(),
                    &mut acc,
                    &mut printed,
                    id,
                    &mut on_piece,
                )
            },
            &mut guard.kv,
            self.max_ctx,
            turn_checkpoint,
            constraint,
            Some(req),
        )?;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::pick_continuation;

    #[test]
    fn continuation_picks_longest_prefix_not_first() {
        // Three free slots, prompt_len = 100. Slots 0 and 2 both qualify (prompt extends their
        // cache: score == cached_len); slot 2 has the LONGER reusable prefix, so it must win even
        // though slot 0 appears first. Slot 1 is a different conversation (score below its cache).
        let candidates = [
            (0usize, 20usize, 20usize), // extends: score 20 == cached 20
            (1, 5, 40),                 // no: 5 != 40 and 5 != 100
            (2, 60, 60),                // extends: score 60 == cached 60 (longest)
        ];
        assert_eq!(pick_continuation(candidates, 100), Some(2));
    }

    #[test]
    fn continuation_accepts_exact_equal_prompt() {
        // score == prompt_len (the prompt EQUALS the cache) qualifies even when cached_len differs.
        let candidates = [(7usize, 30usize, 50usize)];
        assert_eq!(pick_continuation(candidates, 30), Some(7));
    }

    #[test]
    fn continuation_none_when_no_slot_qualifies() {
        // A partial-but-diverged prefix (score < cached_len and < prompt_len) does not continue.
        let candidates = [(0usize, 10usize, 40usize), (1, 0, 0)];
        assert_eq!(pick_continuation(candidates, 100), None);
        // Empty candidate set.
        assert_eq!(pick_continuation(std::iter::empty(), 100), None);
    }
}
